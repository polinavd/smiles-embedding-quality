# Extended Learning-Rate Grid: STS-B, SST-2, QNLI, and Geometry

Follow-up to [`../sweep/RESULTS.md`](../sweep/RESULTS.md). That study swept 5
learning rates (10 seeds each) and found STS-B Spearman still climbing at the
top of the grid. This run extends the grid upward, keeps checkpoints on disk,
and adds two classification tasks and two more geometric metrics.

**Design:** 10 checkpoints, BERT-base, 10k-sentence corpus, weight_decay=0.0,
single seed (42) per learning rate — 1e-5, 3e-5, 5e-5, 7e-5, 1e-4 (the
original grid) plus 2e-4, 3e-4, 5e-4, 7e-4, 1e-3 (the extension). Single seed
per point, so read this as one sample per LR, not a variance estimate (unlike
the 10-seed study).

Each checkpoint was scored on:
- **STS-B Spearman** (downstream similarity quality) + **alignment** /
  **uniformity** (paper's geometry), on the STS-B unique-sentence embeddings.
- **RankMe** (effective rank, spectral entropy of embedding covariance) and
  **IdEst** (MST-length-scaling intrinsic dimension), same embedding matrix.
- **SST-2** and **QNLI** accuracy: encoder frozen, sentences embedded once,
  scikit-learn logistic regression fit on top (`evaluate_glue.py`) — mirrors
  the SimCSE paper's own SentEval transfer-task protocol.

## Full table

| LR | STS-B Spearman | SST-2 acc | QNLI acc | Alignment ↓ | Uniformity ↓ | RankMe | IdEst |
|---|---|---|---|---|---|---|---|
| 1e-05 | 0.3515 | 0.8372 | 0.6500 | 0.3141 | -2.2522 | 116.11 | 11.25 |
| 3e-05 | 0.5460 | 0.8372 | 0.6675 | 0.3017 | -2.5081 | 149.37 | 11.18 |
| 5e-05 | 0.7027 | 0.8326 | 0.6835 | 0.2605 | -2.5211 | 166.92 | 11.39 |
| **7e-05** | **0.7321** | **0.8429** | **0.6860** | 0.2306 | -2.5465 | 159.00 | 10.40 |
| 1e-04 | 0.7253 | 0.8314 | 0.6785 | 0.2270 | -2.6597 | 153.19 | 10.15 |
| 2e-04 | 0.6521 | 0.7947 | 0.6405 | 0.3267 | -2.6843 | 185.62 | 11.38 |
| 3e-04 | 0.6724 | 0.7821 | 0.6755 | 0.3085 | -2.8778 | 158.29 | 10.76 |
| 5e-04 | -0.0149 | 0.5092 | 0.5155 | 0.0000 | -0.0000 | 114.59 | 58.71 |
| 7e-04 | -0.0328 | 0.5092 | 0.5155 | 0.0000 | -0.0000 | 1.64 | 11.06 |
| 1e-03 | 0.0495 | 0.5092 | 0.5155 | 0.0000 | -0.0000 | 292.34 | 85.04 |

**7e-05 wins on every task simultaneously** — STS-B, SST-2, and QNLI all peak
at the same learning rate.

## Conclusions

1. **The curve is now fully bracketed.** The 5-LR study left it unclear
   whether 1e-4 was a true optimum or just the top of an unfinished climb.
   It's neither: the real peak is **7e-05** (STS-B 0.7321), 1e-4 is already
   slightly past it (0.7253), and by 5e-4 training has **collapsed outright**.

2. **Training collapse is real and confirmed by an exact number, not a fluke.**
   At 5e-4, 7e-4, and 1e-3, the training loss plateaus at **ln(16) = 2.7726**
   — the exact cross-entropy value for a model predicting uniformly at random
   over the batch's 16 examples. That means the encoder has stopped
   distinguishing *any* of its 16 in-batch examples from each other: total
   representation collapse, not merely "worse." Alignment and uniformity go
   to exactly 0.0 / -0.0 at these points because every embedding has become
   identical.

3. **Collapse shows up identically on tasks that share nothing with STS-B's
   evaluation recipe.** SST-2 accuracy drops to exactly 0.5092 and QNLI to
   exactly 0.5155 at all three collapsed learning rates — those are the
   majority-class baseline accuracies for each dataset. A classifier trained
   on collapsed (constant) embeddings can do no better than always predicting
   the majority label. This is independent confirmation (via a completely
   different evaluation pipeline — supervised logistic regression, not cosine
   similarity) that the collapse is in the embeddings themselves, not an
   artifact of the STS-B scoring code.

4. **Classification tasks are far less sensitive to learning rate than STS-B
   is, exactly as the alignment/uniformity theory predicts.** Across the
   entire healthy range (1e-5 to 1e-4), STS-B Spearman more than doubles
   (0.35 → 0.73) while SST-2 accuracy barely moves (0.837 → 0.843, about 1
   point) and QNLI moves modestly (0.650 → 0.686). This is the concrete data
   behind the earlier claim that classification tasks — where a supervised
   linear probe is fit on top of frozen embeddings — can partially compensate
   for embedding-geometry differences that STS-B (pure cosine similarity, no
   supervision) cannot. Good uniformity clearly matters much more for
   similarity/retrieval-style evaluation than for classification here.

5. **RankMe and IdEst become unreliable once the embedding space has
   collapsed — don't read the numbers at 5e-4/7e-4/1e-3 as real geometry.**
   In the healthy range both metrics behave sensibly (RankMe roughly
   116–186, IdEst roughly 10–11.4). In the collapsed range they swing
   wildly and non-monotonically (RankMe: 114.59 → 1.64 → 292.34; IdEst:
   58.71 → 11.06 → 85.04) with no consistent pattern. This is because both
   metrics are computed from the covariance/distance structure of the
   embeddings, and once that structure is destroyed (near-identical vectors),
   the metrics are measuring floating-point noise around zero variance, not
   signal. Treat RankMe/IdEst as diagnostic only in the non-collapsed regime.

## Caveats carried over from the earlier study

Single seed per learning rate here (vs. 10 seeds in the original 5-LR study),
10k-sentence corpus (not the paper's 1M), batch size 16 (not 64), STS-B/SST-2/
QNLI only (not the paper's full 7-task STS average or full GLUE/SentEval
suite). See [`../sweep/RESULTS.md`](../sweep/RESULTS.md) and the main
[README](../../README.md) for the full caveat list — all of it still applies.

## Open item

ASMI is still not implemented — its formula/source is still pending (deferred
by request). RankMe and IdEst are done and wired into every future
`evaluate_checkpoint()` call automatically.
