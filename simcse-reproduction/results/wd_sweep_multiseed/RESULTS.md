# Weight-Decay Grid at the Best Learning Rate — 5 Seeds Each

Supersedes the single-seed version in
[`../wd_sweep/RESULTS.md`](../wd_sweep/RESULTS.md), which flagged weight_decay
= 0.05 as a "weak lead" but explicitly couldn't distinguish signal from
single-seed noise. This run re-does the same 5-value grid with 5 seeds per
value (25 checkpoints total) specifically to settle that question.

**Design:** learning_rate=1e-4 (fixed), weight_decay in {0.0, 0.001, 0.01,
0.05, 0.1}, seeds {42, 43, 44, 45, 46} — 5 × 5 = 25 checkpoints, BERT-base,
10k-sentence corpus, checkpoints kept.

## Summary table (mean ± std over 5 seeds)

| Weight decay | STS-B Spearman | SST-2 acc | QNLI acc | Alignment ↓ | Uniformity ↓ | RankMe | IdEst |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.7229 ± 0.0211 | 0.8273 ± 0.0100 | 0.6801 ± 0.0191 | 0.2282 ± 0.0220 | -2.5387 ± 0.1032 | 156.94 ± 9.40 | 10.45 ± 0.45 |
| 0.001 | 0.7305 ± 0.0212 | 0.8266 ± 0.0111 | 0.6775 ± 0.0169 | 0.2237 ± 0.0210 | -2.5455 ± 0.1064 | 157.24 ± 8.86 | 10.39 ± 0.26 |
| 0.01 | 0.7292 ± 0.0193 | 0.8323 ± 0.0070 | 0.6814 ± 0.0157 | 0.2309 ± 0.0198 | -2.5785 ± 0.0944 | 155.29 ± 10.37 | 10.53 ± 0.50 |
| 0.05 | 0.7457 ± 0.0167 | 0.8362 ± 0.0083 | 0.6751 ± 0.0137 | 0.2135 ± 0.0198 | -2.5714 ± 0.0659 | 153.89 ± 7.66 | 10.24 ± 0.25 |
| 0.1 | 0.7392 ± 0.0130 | 0.8236 ± 0.0125 | 0.6765 ± 0.0095 | 0.2188 ± 0.0093 | -2.5711 ± 0.1030 | 153.66 ± 4.35 | 10.31 ± 0.32 |

## One-way ANOVA across the 5 weight-decay groups

| Metric | F | p-value |
|---|---|---|
| STS-B Spearman | 1.169 | 0.354 |
| SST-2 accuracy | 1.261 | 0.318 |
| QNLI accuracy | 0.143 | 0.964 |
| Alignment | 0.688 | 0.609 |
| Uniformity | 0.170 | 0.951 |

## Conclusion

**Weight decay has no statistically detectable effect on any metric in this
setup, across 0.0–0.1.** Every p-value is well above the conventional 0.05
threshold — for QNLI and uniformity, p > 0.9, meaning the five weight-decay
groups are essentially indistinguishable from random regroupings of the same
25 runs. STS-B's numeric mean does drift upward toward 0.05 (0.7229 → 0.7457),
mirroring the single-seed grid's "0.05 looks best" lead — but p=0.354 means
that drift is well within what 5-seed sampling noise alone would produce.

This is exactly why the single-seed version of this grid was flagged as
unreliable: **the apparent "0.05 is best" pattern did not disappear, but it
also never became significant.** With 5 seeds it's still just a trend, not a
finding. It would take more seeds still (or a much larger weight_decay range)
to know whether it's real.

**Practically:** for this corpus size, batch size, and learning rate, weight
decay in the standard 0.0–0.1 range can be left at its default (0.0) without
meaningfully sacrificing STS-B, SST-2, or QNLI performance, or embedding
geometry. This is a genuinely different outcome than the learning-rate
sweep, where the effect was large enough to be obvious even from a single
seed (and where going too high caused outright collapse) — weight decay, at
least in this range, is simply not a lever that moves these numbers by more
than noise.

## Open item

ASMI remains deferred (formula/source pending, on hold by request).
