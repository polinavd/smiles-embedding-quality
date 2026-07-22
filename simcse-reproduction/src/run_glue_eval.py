"""Batch-evaluate SST-2 and QNLI accuracy for every checkpoint saved by a sweep.

This is a separate, post-hoc pass over checkpoints that a sweep
(run_sweep.py / run_hparam_sweep.py) already trained and kept on disk -- it
does not retrain anything. Point it at the same --work-dir the sweep used
(checkpoints must have been kept, i.e. the sweep was NOT run with
--delete-checkpoints).

Each checkpoint subdirectory's name is parsed to recover its learning_rate /
weight_decay / seed for the summary table; it understands both naming
schemes in this repo:
  - run_sweep.py:        lr{lr:g}_seed{seed}
  - run_hparam_sweep.py:  lr{lr:g}_wd{wd:g}_seed{seed}

Results are written incrementally to --results-dir as one JSON file per
checkpoint, so a killed run can be re-run and will skip checkpoints already
scored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_glue import TASK_FIELDS, evaluate_all  # noqa: E402

RUN_KEY_PATTERN = re.compile(
    r"^lr(?P<lr>[^_]+)(?:_wd(?P<wd>[^_]+))?_seed(?P<seed>\d+)$"
)


def parse_run_key(name: str) -> dict:
    match = RUN_KEY_PATTERN.match(name)
    if not match:
        return {"learning_rate": None, "weight_decay": None, "seed": None}
    return {
        "learning_rate": float(match.group("lr")),
        "weight_decay": (float(match.group("wd")) if match.group("wd") is not None else None),
        "seed": int(match.group("seed")),
    }


def is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("models/hparam_sweep"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/hparam_sweep_glue"))
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_FIELDS), default=sorted(TASK_FIELDS))
    parser.add_argument("--max-train", type=int, default=10_000, help="0 or negative disables capping.")
    parser.add_argument("--max-val", type=int, default=2_000, help="0 or negative disables capping.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=32)
    parser.add_argument(
        "--pooler-type",
        choices=["cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls_before_pooler",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.work_dir.is_dir():
        raise FileNotFoundError(
            f"--work-dir does not exist: {args.work_dir}. "
            "Point it at the checkpoint directory a sweep was run with (checkpoints must be kept)."
        )

    checkpoints = sorted(p for p in args.work_dir.iterdir() if is_checkpoint_dir(p))
    if not checkpoints:
        raise RuntimeError(
            f"No checkpoints found under {args.work_dir}. "
            "Was the sweep run with --delete-checkpoints? If so, there is nothing to score here."
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    max_train = args.max_train if args.max_train > 0 else None
    max_val = args.max_val if args.max_val > 0 else None

    rows = []
    for index, checkpoint_dir in enumerate(checkpoints, start=1):
        key = checkpoint_dir.name
        result_file = args.results_dir / f"{key}.json"
        print(f"--- {index}/{len(checkpoints)}: {key} ---")
        if result_file.is_file():
            print(f"[skip] {key} already scored at {result_file}")
            rows.append(json.loads(result_file.read_text(encoding="utf-8")))
            continue

        task_results = evaluate_all(
            model=str(checkpoint_dir),
            tasks=args.tasks,
            max_train=max_train,
            max_val=max_val,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            pooler_type=args.pooler_type,
            device=args.device,
            seed=args.seed,
        )
        row = {
            "checkpoint": key,
            "checkpoint_dir": str(checkpoint_dir),
            **parse_run_key(key),
            **{f"{task}_accuracy": result["accuracy"] for task, result in task_results.items()},
            "tasks": task_results,
        }
        result_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    raw_path = args.results_dir / "glue_sweep_raw.json"
    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nScored {len(rows)} checkpoints. Saved to: {raw_path}")

    for task in args.tasks:
        key = f"{task}_accuracy"
        best = max((r for r in rows if key in r), key=lambda r: r[key], default=None)
        if best is not None:
            print(f"Best {task} accuracy: {best[key]:.4f} ({best['checkpoint']})")


if __name__ == "__main__":
    main()
