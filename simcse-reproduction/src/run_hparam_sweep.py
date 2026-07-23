"""Run a learning-rate x weight-decay x seed grid and print a summary table.

Follow-up to the earlier 5-learning-rate x 10-seed study in
`results/sweep/` (see `results/sweep/RESULTS.md`). That study found STS-B
Spearman still climbing at the top of its grid (best at 1e-4) and only varied
learning rate. This sweep:

  1. Extends the learning-rate grid upward (to 3e-4, 1e-3) to look for where
     it stops climbing.
  2. Adds weight_decay as a second swept dimension (0.0, 0.01, 0.1) --
     previously held fixed at 0.0.
  3. Uses fewer seeds per cell (3 instead of 10) since the grid is now 2-D;
     this estimates seed noise per cell without the full cost of a 10-seed
     x 7-LR x 3-WD grid.

Each run:
  1. Fine-tunes BERT-base with unsupervised SimCSE (train_simcse.train).
  2. Scores the resulting checkpoint on STS-B: Spearman correlation, alignment,
     and uniformity (evaluate_sts.evaluate_checkpoint).
  3. Keeps the checkpoint by default -- GLUE classification probes
     (see run_glue_eval.py) and other geometric metrics (RankMe, etc.) need
     the actual weights, not just the per-run metrics. Pass
     --delete-checkpoints to discard them after scoring.

Results are written incrementally to --results-dir as one JSON file per run,
so a killed sweep can be re-run and will skip runs already on disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_sts import evaluate_checkpoint  # noqa: E402
from train_simcse import load_yaml, parse_settings, train  # noqa: E402

DEFAULT_LEARNING_RATES = [1.0e-5, 3.0e-5, 5.0e-5, 7.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
DEFAULT_WEIGHT_DECAYS = [0.0, 0.01, 0.1]
DEFAULT_SEEDS = [42, 43, 44]


def run_key(learning_rate: float, weight_decay: float, seed: int) -> str:
    return f"lr{learning_rate:g}_wd{weight_decay:g}_seed{seed}"


def run_one(
    base_config: dict,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    work_dir: Path,
    results_dir: Path,
    eval_config: dict,
    keep_checkpoints: bool,
) -> dict:
    key = run_key(learning_rate, weight_decay, seed)
    run_result_file = results_dir / f"{key}.json"
    if run_result_file.is_file():
        print(f"[skip] {key} already has results at {run_result_file}")
        return json.loads(run_result_file.read_text(encoding="utf-8"))

    settings = parse_settings(base_config)
    model_dir = work_dir / key
    settings = dataclasses.replace(
        settings,
        experiment_name=f"unsup-simcse-hparam-sweep-{key}",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        output_dir=model_dir,
        results_file=results_dir / f"{key}_training.json",
        overwrite_output_dir=True,
    )

    print(f"[run] {key}: learning_rate={learning_rate:g} weight_decay={weight_decay:g} seed={seed}")
    train_summary = train(settings)

    eval_result = evaluate_checkpoint(
        model=str(model_dir),
        split=eval_config.get("split", "test"),
        batch_size=int(eval_config.get("eval_batch_size", 64)),
        max_seq_length=int(settings.max_seq_length),
        pooler_type="cls_before_pooler",
        device=settings.requested_device,
        positive_threshold=float(eval_config.get("positive_threshold", 0.8)),
        uniformity_max_pairs=int(eval_config.get("uniformity_max_pairs", 2_000_000)),
        seed=42,  # fixed independent of training seed: keeps eval noise out of the grid comparison
    )

    if not keep_checkpoints:
        shutil.rmtree(model_dir, ignore_errors=True)

    row = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "checkpoint_dir": str(model_dir) if keep_checkpoints else None,
        "mean_training_loss": train_summary["mean_training_loss"],
        "last_training_loss": train_summary["last_training_loss"],
        "train_elapsed_seconds": train_summary["elapsed_seconds"],
        "stsb_spearman": eval_result["stsb_spearman"],
        "alignment": eval_result["alignment"],
        "uniformity": eval_result["uniformity"],
        "rankme": eval_result["rankme"],
        "idest": eval_result["idest"],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    run_result_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    cells = sorted({(row["learning_rate"], row["weight_decay"]) for row in rows})
    for learning_rate, weight_decay in cells:
        group = [
            row
            for row in rows
            if row["learning_rate"] == learning_rate and row["weight_decay"] == weight_decay
        ]
        entry = {"learning_rate": learning_rate, "weight_decay": weight_decay, "n_seeds": len(group)}
        for metric in ("stsb_spearman", "alignment", "uniformity"):
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)
    return summary


def format_table(summary: list[dict]) -> str:
    columns = [
        ("Learning rate", 14),
        ("Weight decay", 13),
        ("n", 3),
        ("STS-B Spearman", 20),
        ("Alignment (lower better)", 26),
        ("Uniformity (lower better)", 26),
    ]
    top = "┌" + "┬".join("─" * (w + 2) for _, w in columns) + "┐"
    sep = "├" + "┼".join("─" * (w + 2) for _, w in columns) + "┤"
    bottom = "└" + "┴".join("─" * (w + 2) for _, w in columns) + "┘"
    header = "│" + "│".join(f" {name:^{w}} " for name, w in columns) + "│"

    lines = [top, header, sep]
    best = max(summary, key=lambda e: e["stsb_spearman_mean"])
    for entry in summary:
        marker = " ★" if entry is best else "  "
        lr_cell = f"{entry['learning_rate']:g}{marker}"
        wd_cell = f"{entry['weight_decay']:g}"
        sts_cell = f"{entry['stsb_spearman_mean']:.4f} ± {entry['stsb_spearman_std']:.4f}"
        align_cell = f"{entry['alignment_mean']:.4f} ± {entry['alignment_std']:.4f}"
        unif_cell = f"{entry['uniformity_mean']:.4f} ± {entry['uniformity_std']:.4f}"
        cells = [lr_cell, wd_cell, str(entry["n_seeds"]), sts_cell, align_cell, unif_cell]
        lines.append("│" + "│".join(f" {c:^{w}} " for c, (_, w) in zip(cells, columns)) + "│")
    lines.append(bottom)
    lines.append("★ = highest mean STS-B Spearman across the grid")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=DEFAULT_WEIGHT_DECAYS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--work-dir", type=Path, default=Path("models/hparam_sweep"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/hparam_sweep"))
    parser.add_argument(
        "--delete-checkpoints",
        action="store_true",
        help="Delete each checkpoint after scoring it. Default is to keep all checkpoints, "
        "since GLUE probes (run_glue_eval.py) and other geometric metrics need the weights.",
    )
    args = parser.parse_args()
    keep_checkpoints = not args.delete_checkpoints

    base_config = load_yaml(args.config)
    eval_config = base_config.get("evaluation", {})
    total = len(args.learning_rates) * len(args.weight_decays) * len(args.seeds)

    rows = []
    completed = 0
    for learning_rate in args.learning_rates:
        for weight_decay in args.weight_decays:
            for seed in args.seeds:
                completed += 1
                print(f"--- {completed}/{total} ---")
                row = run_one(
                    base_config=base_config,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    seed=seed,
                    work_dir=args.work_dir,
                    results_dir=args.results_dir,
                    eval_config=eval_config,
                    keep_checkpoints=keep_checkpoints,
                )
                rows.append(row)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.results_dir / "hparam_sweep_raw.json"
    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary = summarize(rows)
    summary_path = args.results_dir / "hparam_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = format_table(summary)
    print("\n" + table)
    print(f"\nSaved {len(rows)} run results to: {raw_path}")
    print(f"Saved per-cell summary to: {summary_path}")
    if keep_checkpoints:
        print(f"Checkpoints kept under: {args.work_dir}")


if __name__ == "__main__":
    main()
