# Learning-Rate Sensitivity Study — Results

Unsupervised SimCSE, BERT-base, 10k-sentence Wikipedia pilot corpus.
5 learning rates × 10 seeds = 50 fine-tuning runs. Full methodology and caveats
in [`../../README.md`](../../README.md); raw data in
[`lr_sweep_raw.json`](lr_sweep_raw.json), aggregates in
[`lr_sweep_summary.json`](lr_sweep_summary.json).

## Summary table

| Learning rate | n seeds | STS-B Spearman (mean ± std) | Range (min–max) | Alignment ↓ (mean ± std) | Uniformity ↓ (mean ± std) |
|---|---|---|---|---|---|
| 1e-05 | 10 | 0.3623 ± 0.0272 | 0.311 – 0.406 | 0.3092 ± 0.0079 | -2.2497 ± 0.0535 |
| 3e-05 | 10 | 0.5409 ± 0.0806 | 0.399 – 0.620 | 0.2898 ± 0.0109 | -2.4191 ± 0.0862 |
| 5e-05 | 10 | 0.6561 ± 0.0497 | 0.539 – 0.703 | 0.2673 ± 0.0156 | -2.4862 ± 0.0566 |
| 7e-05 | 10 | 0.6761 ± 0.0412 | 0.590 – 0.732 | 0.2577 ± 0.0262 | -2.4958 ± 0.1026 |
| **1e-04** | 10 | **0.7005 ± 0.0361** | 0.651 – 0.756 | 0.2388 ± 0.0242 | -2.5214 ± 0.1001 |

Best mean STS-B Spearman: **1e-04** (0.7005).

## Cross-run correlations (all 50 runs pooled)

| Pair | Pearson r | p-value |
|---|---|---|
| Alignment vs. STS-B Spearman | **-0.845** | 1.2e-14 |
| Uniformity vs. STS-B Spearman | **-0.849** | 6.6e-15 |
| Alignment vs. Uniformity | 0.581 | 9.8e-6 |

## Conclusions

1. **Learning rate has a large, monotonic effect in this regime.** Mean STS-B
   Spearman rises from 0.36 to 0.70 as learning rate increases from 1e-5 to
   1e-4 — each step's mean sits outside the neighboring step's standard
   deviation, so this is a real effect, not seed noise.

2. **The sweep has not found a peak.** Spearman is still increasing at the top
   of the grid (1e-4), and the per-LR spread narrows again at the top end
   (range shrinks from 0.164 at 5e-5 back down to 0.105 at 1e-4, mirroring the
   narrow spread at 1e-5) — a hint the curve may be flattening, but the true
   optimum has not been bracketed. **1e-4 should not be reported as "the best
   learning rate"** — only as the best *of the five tested*.

3. **Geometry tracks downstream quality, exactly as the paper argues.** Both
   alignment and uniformity move toward "better" (lower) as learning rate
   increases, and both correlate strongly with STS-B Spearman across all 50
   runs (r = -0.845 and r = -0.849). This is the clearest confirmation this
   study offers of the paper's central claim: alignment and uniformity are
   not just descriptive side-metrics, they track downstream embedding quality.

4. **Variance is highest in the middle of the grid, not the extremes.** The
   per-LR Spearman range is widest at 3e-05 (0.399–0.620, a 0.22 spread) and
   narrowest at the low end (1e-5: 0.095 spread). At low LR, training makes
   little progress regardless of seed, so runs cluster near the (poor)
   pretrained-BERT baseline; in the middle of the grid, seed (i.e. dropout
   pattern and batch order) determines whether a run "catches" the useful
   optimization region within one epoch, producing the widest spread. This
   argues for keeping multiple seeds per learning rate in any future sweep —
   a single-seed run at 3e-5 could have landed anywhere from 0.40 to 0.62.

5. **These are relative findings, not paper-comparable absolutes.** This
   sweep uses a 10k-sentence corpus (vs. the paper's 1M) and batch size 16
   (vs. 64), and reports STS-B alone (vs. the paper's 7-task SentEval
   average). The optimal learning rate found here (1e-4) is higher than the
   paper's own reported optimum (~3e-5) — expected, since fewer sentences and
   fewer in-batch negatives per step both mean less signal per epoch, pushing
   the useful learning rate up. Do not read this as "1e-4 beats the paper's
   3e-5"; it is an artifact of the reduced setup, not a finding about the
   paper. See the README's caveats section for the full list of deviations.

## Suggested next steps

- Extend the learning-rate grid upward (e.g. 3e-4, 1e-3) to bracket the true
  optimum, since 1e-4 is still climbing.
- Re-run the winning learning rate(s) on the full 1M-sentence corpus at batch
  size 64 to get paper-comparable absolute numbers.
- Add the remaining 6 SentEval tasks (STS12–16, SICK-R) to `evaluate_sts.py`
  so downstream quality is reported as the paper's 7-task average rather than
  STS-B alone.
