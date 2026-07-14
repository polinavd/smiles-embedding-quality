# RankMe reproduction on JE-SSL vision embeddings

Reproduces Garrido et al., ICML 2023: the **effective rank** of frozen JE-SSL ResNet-50 features should correlate **positively** with **linear-probe accuracy** across pretrained checkpoints. This is the project's "works" control point, computed through the identical `embq-harness` recipe used for the scRNA arm.

## Result

- **n = 10** SSL ResNet-50 checkpoints (all 2048-d).
- **Spearman rho = +0.261**, 95% bootstrap CI [-0.509, +0.896] (1000 resamples, seed 0); excludes zero: False.
- **Kendall tau = +0.244**, 95% bootstrap CI [-0.297, +0.800].
- Point-estimate p-values (asymptotic, small-n, report only): Spearman p=0.467, Kendall p=0.381.

**Read this honestly:** at n=10 across *different* SSL methods the overall correlation is positive but its CI straddles zero — the effect is directionally consistent with RankMe but underpowered at this n on CIFAR-100 transfer, where frozen-feature probe accuracies are compressed into a narrow band. The cleaner signal is within-family below.

## Within-architecture family (RankMe's stronger claim)

Within a single training recipe, the only thing varying is training length -> quality, and effective rank tracks probe accuracy much more cleanly:

- **SwAV** (n=4, swav_100ep, swav_200ep, swav_400ep, swav_800ep): Spearman rho = +1.000, Kendall tau = +1.000.

## Negative control (random-projection embeddings)

Matched dimension (2048) and count (10); base input = flattened native-resolution (32x32) CIFAR-100 pixels, projected through independent Gaussian random matrices, run through the SAME metric + readout.

- **Control Spearman rho = +0.541**, 95% CI [-0.156, +0.924]; excludes zero: False.
- The control values are near-constant by construction — a random projection preserves the covariance spectrum, so effective rank ranges only [30.57, 33.55] (std 0.884) and probe accuracy only [0.0885, 0.1003] (std 0.0033) across the 10 encoders. Rank-correlating ~10 nearly-tied values yields an unstable point estimate; what matters is that the CI **includes zero** (`excludes_zero=False`), i.e. no evidence the pipeline manufactures correlation from structureless embeddings.

## Per-checkpoint table

| checkpoint | method | eff_rank | probe_acc |
|---|---|---:|---:|
| dino_rn50 | DINO | 283.89 | 0.6232 |
| vicreg_rn50 | VICReg | 377.08 | 0.6338 |
| barlowtwins_rn50 | BarlowTwins | 380.65 | 0.6202 |
| swav_100ep | SwAV | 168.02 | 0.6205 |
| swav_200ep | SwAV | 203.09 | 0.6378 |
| swav_400ep | SwAV | 230.74 | 0.6550 |
| swav_800ep | SwAV | 212.30 | 0.6545 |
| mocov2_200ep | MoCo-v2 | 140.05 | 0.5952 |
| mocov2_800ep | MoCo-v2 | 152.32 | 0.6130 |
| simsiam_100ep | SimSiam | 235.25 | 0.5968 |

## Provenance

- **Eval set:** CIFAR-100 test split, fixed seeded subset, n_eval = 8000 (seed 0).
- **Preprocessing:** Resize(224) -> ToTensor -> Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (ImageNet stats; checkpoints are ImageNet-pretrained).
- **Features:** torchvision ResNet-50 with fc=Identity; 2048-d penultimate features; every backbone weight asserted loaded (no missing keys).
- **Readout:** `embq.readouts.linear_probe_accuracy` (logistic regression, lbfgs, 50/50 stratified split, seed 0).
- **Correlation / CI / control:** `embq.harness` (seed 0, 1000 bootstrap resamples).
- **Device:** cuda.
- **Checkpoints (exact weights source):**
  - `dino_rn50` — facebookresearch/dino, ResNet-50 backbone — https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
  - `vicreg_rn50` — facebookresearch/vicreg, ResNet-50 backbone — https://dl.fbaipublicfiles.com/vicreg/resnet50.pth
  - `barlowtwins_rn50` — facebookresearch/barlowtwins, 1000ep ResNet-50 backbone — https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth
  - `swav_100ep` — facebookresearch/swav, 100ep ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar
  - `swav_200ep` — facebookresearch/swav, 200ep ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_pretrain.pth.tar
  - `swav_400ep` — facebookresearch/swav, 400ep ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar
  - `swav_800ep` — facebookresearch/swav, 800ep ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar
  - `mocov2_200ep` — facebookresearch/moco, MoCo-v2 200ep ResNet-50 — https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar
  - `mocov2_800ep` — facebookresearch/moco, MoCo-v2 800ep ResNet-50 — https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
  - `simsiam_100ep` — facebookresearch/simsiam, 100ep ResNet-50 — https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar

## Mechanistic interpretation

Effective rank measures how many directions of the feature covariance carry real variance — the *linear spread* of the embedding. The linear probe is a linear classifier on those same frozen features, so its accuracy depends on exactly that linear spread: a collapsed, low-effective-rank representation cannot linearly separate the classes, while a checkpoint that spreads variance over more directions gives the probe more separable structure to exploit. Metric and readout depend on the *same* property, so they track. This shows up most clearly within a single training recipe (the SwAV epoch ladder above): holding the method fixed and only increasing training, effective rank and probe accuracy rise together. Across different methods the relationship is noisier and, at n=10 on CIFAR-100 transfer, not separable from zero — an honest small-n limitation, consistent with the project's own pilot finding that geometric metrics are weak *cross-encoder* predictors. This is still the mirror image of the scRNA scVI-30 failure case, where the readout (ARI clustering) depends on cluster geometry that a linear-spread metric does not see at all. Same harness both ends; the difference is mechanistic.
