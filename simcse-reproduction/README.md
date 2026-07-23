# Unsupervised SimCSE — Learning-Rate Sensitivity Study

A scoped partial reproduction of **SimCSE: Simple Contrastive Learning of Sentence
Embeddings** (Gao, Yao & Chen, 2021 — [arXiv:2104.08821](https://arxiv.org/abs/2104.08821),
[princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE)).

This is **not** an attempt to reproduce the paper's headline numbers. It is a
controlled **sensitivity study**: we hold the entire unsupervised-SimCSE recipe
fixed and vary hyperparameters one at a time (learning rate, then weight
decay), asking how each moves:

1. **Alignment** — how close known-positive pairs sit on the unit hypersphere.
2. **Uniformity** — how evenly embeddings spread over the hypersphere.
3. **RankMe** and **IdEst** — label-free rank / intrinsic-dimension diagnostics.
4. **Downstream STS quality** — Spearman correlation with human similarity judgements.
5. **Downstream classification quality** — SST-2 and QNLI accuracy via a frozen-embedding + logistic-regression probe.

The study ran in three stages, each with its own results writeup:

| Stage | What varied | Results |
|---|---|---|
| 1. Learning-rate sweep | 5 LRs × 10 seeds = 50 runs | [`results/sweep/RESULTS.md`](results/sweep/RESULTS.md) |
| 2. Learning-rate extension | +5 higher LRs, 1 seed each, checkpoints kept, + SST-2/QNLI + RankMe/IdEst | [`results/lr_extended/RESULTS.md`](results/lr_extended/RESULTS.md) |
| 3. Weight-decay grid at the best LR | 5 weight decays × 5 seeds = 25 runs | [`results/wd_sweep_multiseed/RESULTS.md`](results/wd_sweep_multiseed/RESULTS.md) (supersedes the single-seed [`results/wd_sweep/RESULTS.md`](results/wd_sweep/RESULTS.md)) |

**Headline findings so far:** the learning-rate curve is fully bracketed —
STS-B Spearman peaks around **7e-5 to 1e-4** (both statistically tied) and
training **collapses outright** at ≥5e-4 (loss pins at exactly `ln(16)`,
confirmed independently by STS-B, SST-2, and QNLI all degrading to
chance/majority-baseline simultaneously). Weight decay, by contrast, has
**no statistically significant effect** on any metric across 0.0–0.1 (all
ANOVA p-values > 0.3) — leave it at the default.

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

Each stage already has an implementation in `src/`:

| Stage | Script | Status |
|-------|--------|--------|
| Fine-tune | [`src/train_simcse.py`](src/train_simcse.py) | ✅ implemented |
| Embed | [`src/generate_embeddings.py`](src/generate_embeddings.py) | ✅ implemented |
| Alignment / uniformity / RankMe / IdEst | [`src/geometric_metrics.py`](src/geometric_metrics.py) | ✅ implemented (ASMI deferred, formula pending) |
| Downstream STS | [`src/evaluate_sts.py`](src/evaluate_sts.py) | ⚠️ **STS-B only** — full 7-task SentEval not yet wired up |
| Downstream classification (SST-2, QNLI) | [`src/evaluate_glue.py`](src/evaluate_glue.py) | ✅ implemented — frozen embeddings + logistic-regression probe |
| Learning-rate × seed sweep driver | [`src/run_sweep.py`](src/run_sweep.py) | ✅ implemented — runs (LR × seed) experiments and prints a summary table |
| Learning-rate × weight-decay × seed grid driver | [`src/run_hparam_sweep.py`](src/run_hparam_sweep.py) | ✅ implemented |
| Batch GLUE eval over a sweep's saved checkpoints | [`src/run_glue_eval.py`](src/run_glue_eval.py) | ✅ implemented — post-hoc, does not retrain |

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
2. Scores the checkpoint on STS-B (Spearman, alignment, uniformity, RankMe,
   IdEst) via the same `evaluate_checkpoint` function `evaluate_sts.py` uses
   for single runs.
3. **Keeps the checkpoint by default** — later work (GLUE probes, RankMe/IdEst
   on other embedding sets) needs the actual weights. Pass
   `--delete-checkpoints` to discard them after scoring (BERT-base checkpoints
   are ~400MB each, so a large grid can add up fast).

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

### 6. Extend the learning-rate grid (or run a weight-decay × seed grid)

Same script, different axes:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Learning-rate grid at a single seed (used to find the collapse point):
uv run python src/run_sweep.py --config configs/baseline.yaml \
  --learning-rates 1e-5 3e-5 5e-5 7e-5 1e-4 2e-4 3e-4 5e-4 7e-4 1e-3 \
  --seeds 42 --work-dir models/lr_extended --results-dir results/lr_extended

# Weight-decay x seed grid at a fixed (best) learning rate:
uv run python src/run_hparam_sweep.py --config configs/baseline.yaml \
  --learning-rates 1e-4 --weight-decays 0.0 0.001 0.01 0.05 0.1 \
  --seeds 42 43 44 45 46 \
  --work-dir models/wd_sweep_multiseed --results-dir results/wd_sweep_multiseed
```

### 7. Batch-evaluate SST-2 and QNLI over a sweep's saved checkpoints

Post-hoc — does not retrain, just scores whatever checkpoints a sweep already
kept on disk:

```bash
uv run python src/run_glue_eval.py \
  --work-dir models/lr_extended --results-dir results/lr_extended_glue \
  --tasks sst2 qnli
```

---

## What still needs building

- **Full 7-task SentEval evaluation.** `evaluate_sts.py` (and therefore
  `run_sweep.py` / `run_hparam_sweep.py`) only scores STS-B. To report the
  paper-style downstream number you must add STS12, STS13, STS14, STS15,
  STS16, and SICK-R and average the seven Spearman scores. Keep
  alignment/uniformity on STS-B as above.
- **ASMI metric.** Deferred by request — formula/paper not yet provided.
  RankMe and IdEst are done and computed automatically by every
  `evaluate_checkpoint()` call.

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
  The learning-rate extension and weight-decay grids that keep checkpoints use
  fewer seeds per point (1 and 5 respectively) — treat single-seed comparisons
  there as leads, not conclusions, unless the effect is as large and
  unambiguous as the high-LR training collapse was.
- **RankMe and IdEst are unreliable once training has collapsed** (see
  `results/lr_extended/RESULTS.md`) — both metrics are computed from
  covariance/distance structure that becomes pure floating-point noise once
  all embeddings collapse to near-identical vectors. Trust them only in the
  non-collapsed regime.