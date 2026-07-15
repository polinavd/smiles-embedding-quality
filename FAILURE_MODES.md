# Failure-mode analysis

The transfer hypothesis was falsified: no label-free geometric metric in
`embq/metrics.py` reliably predicts single-cell downstream recovery. This
document is the diagnostic follow-up — for each metric, *where and why* does it
produce a misleading verdict? These are the failures a new, more robust
label-free metric would have to avoid.

Reproduce with:

```bash
python scripts/failure_mode_analysis.py
```

It reads the checked-in tables (`results/build_sweep_averaged.csv`,
`results/build_sweep_perseed.csv`) and, for FM4, the saved embedding matrices in
`results/embeddings/`; it writes one CSV + one figure per mode. Logic lives in
`embq/failure_modes.py`.

## Samples & provenance (read this before comparing numbers)

**FM1, FM2 and FM3 now all read the same table** — `build_sweep_averaged.csv`
(seed-averaged: 6×PCA + 2×ICA + 2×NMF + 2×AE + 2×scVI = **14 real encoders** +
3 controls = **17 total**). An earlier version of this analysis ran FM1-3 on
the smaller positive-control table (`posctrl_full_table.csv`, 8 real + 3
controls, n=11) while FM4 already used the 14-encoder sweep table — an
unintentional *n* mismatch across the four failure modes. `build_sweep_averaged.csv`
carries every column FM1-3 need (`anisotropy`, `effective_rank`, `linear_probe`,
`ari`, ...), so there was no reason for the split; the positive-control table
is no longer used by this analysis.

Two samples remain, and they are **not interchangeable** — an *n* or a
correlation from one is not comparable to the other:

| Sample | Source table | Encoders | *n* | Seeds / readout |
|---|---|---|---|---|
| **B — sweep, seed-averaged** | `build_sweep_averaged.csv` | 6×PCA + 2×ICA + 2×NMF + 2×AE + 2×scVI (+3 controls) | **14 real / 17 total** | averaged over 3 seeds/encoder |
| **B-perseed — sweep, per-seed** | `build_sweep_perseed.csv` | same 14 real encoders | **14 real** | 3 seeds/encoder, not averaged (used only for FM4's variance decomposition) |
| **C — saved-embedding seed study** | `results/embeddings/*.npy` + `labels.npy` | pca_10, pca_50, scvi_30 (+ random control) | **3 real** | **30** KMeans seeds on fixed geometry |

Which sample backs which claim:

- **FM1, FM2, FM3** use **Sample B** (n=14 real / 17 total) — unified.
- **FM4** uses **Sample C** as the primary estimate (30 seeds), with
  **Sample B-perseed** (3 seeds, n=14 — the *same* 14 real encoders as FM1-3)
  reported alongside only to show how unstable the low-seed estimate is.
- The within-encoder KMeans-noise fraction is **16%** in Sample C (30 seeds,
  CI [7%, 23%]) versus a fragile **28%** in Sample B-perseed (3 seeds, CI
  [12%, 55%]). These two numbers are unchanged by the FM1-3 unification —
  FM4 never used the positive-control table to begin with.

> **`results/failure_mode_results.csv` is not part of this pipeline.**
> It has a different schema (`case_id`/`family`/`role`/`expected_failure`
> columns, synthetic cases like `dim_confound_006`, `nonlinear_moons`) and
> nothing in `failure_mode_analysis.py` / `embq/failure_modes.py` reads or
> writes it — it was never used to build `otchet_failure_modes.pdf`. It
> looks like a stray copy of an unrelated synthetic-case suite and should
> not be cited as a source for this report.

## FM1 — effective rank measures capacity, not quality

`results/fm1_effrank_vs_dim.{csv,png}` — Sample B (n=14 real / 17 total).

Across PCA encoders, effective rank is a near-perfect linear function of the
latent dimension (Spearman(effrank, dim) = **+1.00**, slope ≈ 0.94 rank/dim);
its correlation with ARI is weaker (+0.83, still not tight enough to trust as
a quality proxy). At a **fixed** latent dim of 30 (8 encoders: 5 real families
+ 3 controls), two of the three uninformative controls — `Noise-30` and
`RandProj-30` — take the **2nd and 3rd** highest effective-rank scores,
ahead of `scVI-30`, `PCA-30`, `ShufPCA-30` and `AE-30`, while their ARI ≈ 0.
Meanwhile `NMF-30`, the encoder with the **best** downstream ARI in the group
(0.85), has the **lowest** effective rank of all eight. *Why it fails:*
spectral entropy rewards spreading variance over many directions, which pure
noise does close to maximally — the metric cannot tell "many directions
because there is signal" from "many directions because it's noise".

## FM2 — control-leverage artifact

`results/fm2_control_leverage.{csv,png}` — Sample B (n=17 all / n=14 real).

The metrics that look predictive on the full benchmark owe part of their
correlation to the negative controls sitting in one convenient corner of the
(metric, ARI) plane. Dropping the controls attenuates every one of them:

| metric | ρ (all, n=17) | ρ (real only, n=14) |
|---|---|---|
| anisotropy | +0.59 | +0.34 |
| kNN-consistency | +0.45 | +0.02 |
| uniformity | +0.50 | +0.21 |
| alignment | −0.50 | −0.19 |

*Why it fails:* a correlation driven by high-leverage extreme points is a
benchmark-construction artifact, not evidence that the metric ranks *real*
encoders. On real encoders alone, kNN-consistency is essentially zero and the
rest are materially weaker than the full-benchmark number would suggest.

## FM3 — readout-dependence (compared by rank, not by raw value)

`results/fm3_readout_dependence.{csv,png}` — Sample B (real only, n=14).

The two "ground truths" order the encoders differently. Linear probe accuracy
and ARI are **different metrics on different scales**, so their raw values are
not directly comparable. The defensible comparison is between the **rankings**
each readout induces over the encoders:

- Rank agreement is weak and not significant: **Spearman ρ = +0.41** (p =
  0.15), **Kendall τ = +0.28** (p = 0.19), n=14.
- **33 of 91 encoder pairs are reversed** (the two readouts disagree on which
  encoder is better). The full reversal list is printed by the driver and in
  the CSV. The largest-gap reversal: probe ranks **ICA-30 first** (probe
  0.960) but its ARI is only 0.62, while **NMF-30** — probe 0.940, ranked
  below it — reaches ARI 0.85 (a 0.23 ARI gap in the "wrong" direction). The
  scVI case from the earlier (n=8) analysis still holds too: probe ranks
  **scVI-30/scVI-10 above PCA-2**, while ARI ranks them **below**.

*Why it fails:* linear separability (probe) and cluster compactness
(KMeans/ARI) are different geometric demands. No scalar geometric predictor
can be simultaneously right when the two readouts rank encoders differently —
"quality" is a (task, readout) pair, not a number.

## FM4 — the ARI target is ill-conditioned

`results/fm4_target_conditioning.{csv,png}` — **Sample C** primary (30 KMeans
seeds on the saved fixed embeddings), **Sample B-perseed** (3 seeds, n=14 —
same 14 real encoders as FM1-3) for contrast.

Part of the target is pure KMeans seed noise on a fixed embedding —
unpredictable by construction from seed-invariant geometry. Quantifying it
honestly requires many seeds; the 3-seed estimate is too fragile to carry the
claim on its own.

- **Sample C (30 seeds, fixed geometry, n=3 real):** within-encoder KMeans-noise
  is **16% of total ARI variance, 95% CI [7%, 23%]** (bootstrap over seeds).
- **Sample B-perseed (3 seeds, n=14):** the same quantity reads **28%**, but its
  bootstrap CI is **[12%, 55%]** — so wide it cannot support a precise claim.
  This is exactly why the recomputation on 30 seeds was needed.
- The panel-independent core statistic is the **per-encoder ARI std across
  seeds**: pca_10 0.012, pca_50 **0.054**, scvi_30 0.006 (random control 0.023).
  Most of the seed noise is carried by one encoder (pca_50, which occasionally
  drops from ~0.75 to ~0.58); scVI and pca_10 are nearly seed-stable.
- The real-encoder ARI range stays compressed (Sample B-perseed **[0.49, 0.86]**;
  Sample C's three encoders span **[0.58, 0.75]**).

*Why it fails:* a compressed target with a modest but non-zero stochastic
component leaves a limited between-encoder signal for any predictor, geometric
or otherwise, to latch onto — though the unpredictable fraction (~16%) is
smaller than the 3-seed estimate suggested, so this is stated as a contributing
factor, not the dominant one.

> **Caveat.** The variance *fraction* depends on the encoder panel (Sample C has
> only 3 saved real encoders). The per-encoder seed std above is
> panel-independent and is the more robust way to read "how stochastic is ARI".
> Re-run `fm4_kmeans_seed_study` once more encoders' embeddings are saved to
> `results/embeddings/` to tighten the between-encoder estimate.

## Takeaway for the next step

Each mode points at a concrete requirement for a better label-free metric /
target:

- FM1 → the metric must be invariant to (or normalized for) latent dimension.
- FM2 → validate on real encoders, never let controls carry the correlation.
- FM3 → fix the readout first; predict geometry *for that readout*.
- FM4 → use a less compressed, less stochastic, non-circular downstream target
  (e.g. an independent CITE-seq protein label, per the reformulation).
