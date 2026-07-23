# Unsupervised SimCSE — Learning-Rate Sensitivity Study

A scoped partial reproduction of **SimCSE: Simple Contrastive Learning of Sentence
Embeddings** (Gao, Yao & Chen, 2021 — [arXiv:2104.08821](https://arxiv.org/abs/2104.08821),
[princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE)).

This is **not** an attempt to reproduce the paper's headline numbers. It is a
controlled **sensitivity study**: we hold the entire unsupervised-SimCSE recipe
fixed and vary **only the learning rate**, then ask how the learning rate moves
three quantities the paper cares about:

1. **Alignment** — how close known-positive pairs sit on the unit hypersphere.
2. **Uniformity** — how evenly embeddings spread over the hypersphere.
3. **Downstream STS quality** — Spearman correlation with human similarity judgements.

---

## The pipeline (what actually happens)

```
BERT-base (pretrained)
      │
      │  unsupervised SimCSE fine-tune on unlabeled Wikipedia sentences
      │  (dual-dropout positives, in-batch negatives, temperature 0.05)
      ▼
fine-tuned encoder  ──►  generate sentence embeddings
      │                          │
      │                          ▼
      │                   alignment + uniformity   (measured on STS-B)
      ▼
downstream STS evaluation  ──►  Spearman correlation
   (7 SentEval STS tasks, averaged)
```

Each of the four stages already has an implementation in `src/`:

| Stage | Script | Status |
|-------|--------|--------|
| Fine-tune | [`src/train_simcse.py`](src/train_simcse.py) | ✅ implemented |
| Embed | [`src/generate_embeddings.py`](src/generate_embeddings.py) | ✅ implemented |
| Alignment / uniformity | [`src/geometric_metrics.py`](src/geometric_metrics.py) | ✅ implemented |
| Downstream STS | [`src/evaluate_sts.py`](src/evaluate_sts.py) | ⚠️ **STS-B only** — full 7-task SentEval not yet wired up |
| 50-run sweep driver | [`src/run_sweep.py`](src/run_sweep.py) | ✅ implemented — runs all 50 (LR × seed) experiments and prints a summary table |

---

## Experimental design — 50 runs

**1 model × 5 learning rates × 10 seeds = 50 experiments.**

| Factor | Values | Notes |
|--------|--------|-------|
| Base model | `google-bert/bert-base-uncased` | RoBERTa-base is out of scope for this study |
| **Learning rate** (the only swept factor) | `1e-5, 3e-5, 5e-5, 7e-5, 1e-4` | `3e-5` is the paper's BERT-base value (the anchor). Adjust in the config if desired. |
| Seed | `42, 43, 44, 45, 46, 47, 48, 49, 50, 51` | 10 seeds per LR give variance bands for the LR curves |

Everything else is held **constant** at the values in
[`configs/baseline.yaml`](configs/baseline.yaml). The seed controls parameter
initialisation of the projection head, dropout masks, and data-shuffle order, so
10 seeds yield genuine run-to-run variance on the 10k pilot corpus.

### Fixed configuration (held constant across all 50 runs)

| Setting | Value | Matches paper? |
|---------|-------|----------------|
| Objective | unsupervised SimCSE (dual dropout) | ✅ |
| Pooler (train) | `cls` + train-time MLP, `mlp_only_train: true` | ✅ |
| Pooler (eval) | `cls_before_pooler` (MLP dropped at test time) | ✅ |
| Temperature | `0.05` | ✅ |
| Dropout | `0.10` | ✅ |
| Max sequence length | `32` | ✅ |
| Epochs | `1` | ✅ |
| Batch size | `16` | ⚠️ **deviation** — paper uses 64 for BERT-base (MPS memory constraint) |
| Training corpus | `data/pilot/wiki10k.txt` (10k) | ⚠️ **deviation** — paper uses 1M sentences |
| Optimizer | AdamW, `weight_decay 0.0`, `warmup_ratio 0.0` | ✅ |

---

## Metrics

- **Alignment** (`α = 2`, lower is better): mean squared distance between
  L2-normalized positive pairs. Positive pairs are STS-B pairs with normalized
  score `> 0.8` (the paper's original `score > 4/5`). Implemented in
  [`src/geometric_metrics.py`](src/geometric_metrics.py).
- **Uniformity** (`t = 2`, lower is better): log of the mean Gaussian potential
  between all unique STS-B sentence embeddings.
- **Downstream STS quality**: Spearman correlation between cosine similarity and
  human scores. Reported as the **average over the 7 SentEval STS tasks**
  (STS12–16, STS-B, SICK-R) — see *What still needs building*.

> **Note on alignment/uniformity:** following the paper, these are computed on
> **STS-B**, not on the training corpus. Alignment uses the human labels to
> select positive pairs, so it is not fully label-free in this evaluation
> (this is stated honestly in `evaluate_sts.py`).

---

## How to run

Install the extra direct dependency:

```bash
uv add pyyaml
```

### 1. Baseline sanity — evaluate the untouched pretrained BERT first

```bash
uv run python src/evaluate_sts.py \
  --model google-bert/bert-base-uncased \
  --split test \
  --output results/bert_pretrained_stsb.json
```

### 2. Fine-tune one run

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python src/train_simcse.py --config configs/baseline.yaml
```

### 3. Evaluate a fine-tuned checkpoint (STS-B + geometry)

```bash
uv run python src/evaluate_sts.py \
  --model models/baseline \
  --split test \
  --output results/simcse_baseline_stsb.json
```

### 4. Generate a standalone embedding matrix (optional)

```bash
uv run python src/generate_embeddings.py \
  --model models/baseline \
  --input data/pilot/wiki10k.txt \
  --output results/wiki10k_embeddings.npy \
  --max-samples 1000
```

### 5. Run the full 50-experiment sweep

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python src/run_sweep.py --config configs/baseline.yaml
```

[`src/run_sweep.py`](src/run_sweep.py) runs all 5 learning rates × 10 seeds
(the defaults baked into the script; override with `--learning-rates` /
`--seeds` if needed). For each of the 50 runs it:

1. Fine-tunes BERT-base from `configs/baseline.yaml` with that run's
   `learning_rate` and `seed` substituted in (everything else fixed).
2. Scores the checkpoint on STS-B (Spearman, alignment, uniformity) via the
   same `evaluate_checkpoint` function `evaluate_sts.py` uses for single runs.
3. Deletes the fine-tuned checkpoint (50 BERT-base checkpoints is 20+ GB and
   only the metrics are needed) — pass `--keep-checkpoints` to retain them.

**Output artifacts** (under `results/sweep/` by default):

- `<key>.json` — one file per run (e.g. `lr3e-05_seed42.json`), written as
  soon as that run finishes. If the sweep is killed and re-run, any run whose
  file already exists is **skipped**, not repeated.
- `lr_sweep_raw.json` — all 50 rows once the sweep completes.
- `lr_sweep_summary.json` — mean ± std of `stsb_spearman`, `alignment`, and
  `uniformity` per learning rate, aggregated over the 10 seeds.

The script also prints a summary table to the terminal, e.g.:

```
┌────────────────┬─────┬──────────────────────┬────────────────────────────┬────────────────────────────┐
│ Learning rate  │  n  │    STS-B Spearman    │  Alignment (lower better)  │ Uniformity (lower better)  │
├────────────────┼─────┼──────────────────────┼────────────────────────────┼────────────────────────────┤
│    1e-05       │ 10  │   0.xxxx ± 0.xxxx    │      0.xxxx ± 0.xxxx       │     -x.xxxx ± 0.xxxx       │
│    3e-05       │ 10  │   0.xxxx ± 0.xxxx    │      0.xxxx ± 0.xxxx       │     -x.xxxx ± 0.xxxx       │
│    5e-05 ★     │ 10  │   0.xxxx ± 0.xxxx    │      0.xxxx ± 0.xxxx       │     -x.xxxx ± 0.xxxx       │
│    7e-05       │ 10  │   0.xxxx ± 0.xxxx    │      0.xxxx ± 0.xxxx       │     -x.xxxx ± 0.xxxx       │
│    1e-04       │ 10  │   0.xxxx ± 0.xxxx    │      0.xxxx ± 0.xxxx       │     -x.xxxx ± 0.xxxx       │
└────────────────┴─────┴──────────────────────┴────────────────────────────┴────────────────────────────┘
★ = highest mean STS-B Spearman across the sweep
```

Expect roughly **3.5–4 hours** for all 50 runs on the 10k-sentence corpus on
Apple-silicon MPS (the single-run pilot took ~230s/epoch; add STS-B encoding
overhead per run).

---

## What still needs building

- **Full 7-task SentEval evaluation.** `evaluate_sts.py` (and therefore
  `run_sweep.py`) only scores STS-B. To report the paper-style downstream
  number you must add STS12, STS13, STS14, STS15, STS16, and SICK-R and
  average the seven Spearman scores. Keep alignment/uniformity on STS-B as
  above.

---

## Caveats — read before interpreting results

- **The 10k pilot is not paper-comparable.** With only 10k sentences and batch
  size 16, the contrastive task is easy and training loss collapses toward zero
  (the pilot in `results/training_baseline.json` reached `last_training_loss ≈
  3.5e-5`). **Absolute** alignment/uniformity/STS numbers will not match the
  paper. Treat every result as **relative** — the object of study is the *shape*
  of each metric as a function of learning rate, not its absolute value.
- **Batch size 16 vs the paper's 64** reduces in-batch negatives from 63 to 15,
  which directly affects uniformity and downstream quality. Held constant here,
  but it is a real deviation.
- **No model selection.** The paper checkpoints on the STS-B dev set every 250
  steps and keeps the best; here we train one epoch and evaluate the final state.
- **Learning-rate effects are noisy on a small corpus** — this is exactly why the
  design uses 10 seeds per LR. Report mean ± spread, not single-seed points.
```