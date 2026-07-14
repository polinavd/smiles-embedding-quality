# Results

Outputs from the embedding-quality experiments (encoder sweep, cross-modality
checks, and positive controls). Each file below has a one-line caption.

## Figures

- **build_summary.png** — summary plot of quality metrics across the encoder/dim sweep.
- **effrank_dim_contrapositive.png** — effective rank vs. embedding dimensionality, showing the contrapositive relationship used as a sanity check.
- **positive_control.png** — positive-control run: known-good embeddings recover expected downstream performance.
- **metric_vs_ari.png** — scatter of each unsupervised quality metric against downstream ARI (clustering agreement).
- **crossmodality_signs.png** — sign consistency of metric-vs-ARI correlations across modalities (e.g. PBMC vs. text).
- **rankme_vision_cifar100_scatter.png** / **rankme_vision_cifar10_scatter.png** — RankMe reproduction: effective rank vs. linear-probe accuracy for 17 JE-SSL ResNet-50 checkpoints, colored by SSL method, on CIFAR-100 and CIFAR-10 respectively.

## Tables

- **build_sweep_averaged.csv** — quality metrics (anisotropy, effective rank, intrinsic dim, uniformity, alignment, kNN consistency, ARI/NMI/kNN-acc) averaged over seeds, per encoder/dim/family.
- **build_sweep_perseed.csv** — same sweep, per-seed (not averaged), for variance inspection.
- **build_bootstrap_family.csv** — bootstrap confidence intervals (rho, CI lo/hi) for metric families, with whether the CI excludes zero.
- **embedding_quality_results.csv** — per-embedding quality metrics and downstream scores, one row per encoder.
- **posctrl_full_table.csv** — full metric table for the positive-control run (adds silhouette, linear probe, kNN label purity).
- **posctrl_correlations.csv** — correlation of each metric with ARI in the positive-control setting.
- **metric_downstream_correlations.csv** — Spearman correlation of each unsupervised metric with downstream ARI, with p-values and CIs.
- **partial_corr_dimcontrolled.csv** — partial correlations with embedding dimensionality controlled for, vs. raw correlations.
- **crossmodality_correlations.csv** — metric-vs-ARI correlations computed separately per modality (e.g. PBMC).
- **text_arm_results.csv** — per-seed results for the text-modality arm of the experiment.
- **text_arm_averaged.csv** — text-modality arm results averaged over seeds.
- **rankme_vision_cifar100.csv** / **rankme_vision_cifar10.csv** — per-checkpoint effective rank and linear-probe accuracy for the RankMe vision reproduction (one row per JE-SSL ResNet-50 checkpoint), on CIFAR-100 and CIFAR-10.
- **rankme_vision_cifar100.md** / **rankme_vision_cifar10.md** — RankMe reproduction summaries: Spearman ρ / Kendall τ, bootstrap CI, quality-ladder and within-method-family correlations, negative-control result, checkpoint provenance, seeds/preprocessing, and mechanistic interpretation, per dataset.
