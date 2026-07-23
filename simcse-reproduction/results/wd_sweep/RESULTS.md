# Weight-Decay Grid at the Best Learning Rate (1e-4)

> **Superseded.** This was a single-seed grid (see caveat in Conclusion #1
> below) and could not tell signal from noise. It was re-run with 5 seeds per
> weight-decay value in
> [`../wd_sweep_multiseed/RESULTS.md`](../wd_sweep_multiseed/RESULTS.md),
> which found **no statistically significant effect of weight decay** on any
> metric (all ANOVA p-values > 0.3). Read that file instead; this one is kept
> only for provenance.

Follow-up to [`../lr_extended/RESULTS.md`](../lr_extended/RESULTS.md). That
study bracketed the learning-rate curve (winning region ~7e-5 to 1e-4,
collapse at ≥5e-4) but held weight_decay fixed at 0.0 throughout. This run
fixes learning_rate=1e-4 (the higher-mean pick of the two statistically-tied
candidates) and sweeps weight_decay instead.

**Design:** 5 checkpoints, BERT-base, 10k-sentence corpus, learning_rate=1e-4,
single seed (42), weight_decay in {0.0, 0.001, 0.01, 0.05, 0.1}. Single seed
per point — same caveat as the LR extension: read as one sample per cell, not
a variance estimate.

## Full table

| Weight decay | STS-B Spearman | SST-2 acc | QNLI acc | Alignment ↓ | Uniformity ↓ | RankMe | IdEst |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.7233 | 0.8394 | 0.6850 | 0.2286 | -2.6690 | 153.74 | 10.16 |
| 0.001 | 0.7282 | 0.8383 | 0.6845 | 0.2156 | -2.6238 | 146.80 | 9.90 |
| 0.01 | 0.7213 | 0.8383 | 0.6770 | 0.2216 | -2.6668 | 153.01 | 9.85 |
| **0.05** | **0.7352** | 0.8394 | **0.6905** | **0.2100** | -2.6114 | 156.65 | 9.81 |
| 0.1 | 0.7252 | 0.8394 | 0.6795 | 0.2246 | -2.4886 | 152.86 | 10.58 |

## Conclusions

1. **Weight decay's effect here is small — likely within single-seed noise,
   not a real signal.** STS-B Spearman spans only 0.7213–0.7352 (a 0.014
   range) across all 5 weight-decay values. Compare that to the *seed* noise
   measured in the original 10-seed study at the same learning-rate region:
   std of 0.036 (at 7e-5) and 0.036 (at 1e-4) — 2.5x larger than the entire
   spread seen here across weight decay. **This grid cannot distinguish "weight
   decay matters" from "this is what one seed's worth of noise looks like."**
   A real answer would need multiple seeds per weight-decay value, the same
   fix that mattered for the learning-rate study.

2. **If forced to pick one value from this single-seed data, 0.05 looks
   best**, winning on STS-B, QNLI, and alignment simultaneously (and tying on
   SST-2) — but per point 1, this should be treated as a weak lead, not a
   conclusion. Unlike the learning-rate collapse (which was unambiguous even
   from a single seed — loss pinned exactly at ln(16)), nothing here rises
   above the noise floor.

3. **No collapse anywhere in this grid.** All 5 weight-decay values keep
   healthy STS-B/SST-2/QNLI numbers and sane RankMe/IdEst values (unlike the
   erratic RankMe/IdEst readings seen once learning rate collapsed training in
   the previous study) — weight decay in this range doesn't destabilize
   training the way high learning rate did.

## Suggested next step

Repeat this exact grid with multiple seeds (e.g. 5) per weight-decay value
before concluding anything about weight decay specifically — at single-seed
resolution, this grid is underpowered relative to the noise it needs to
detect.

## Open item

ASMI is still deferred (formula/source pending, on hold by request).
