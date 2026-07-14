# RankMe reproduction on JE-SSL vision embeddings (CIFAR-100)

Reproduces Garrido et al., ICML 2023: the **effective rank** of frozen JE-SSL ResNet-50 features should correlate **positively** with **linear-probe accuracy** across pretrained checkpoints. This is the project's "works" control point, computed through the identical `embq-harness` recipe used for the scRNA arm.

## Result

- **n = 17** SSL ResNet-50 checkpoints (all 2048-d).
- **Spearman rho = +0.404**, 95% bootstrap CI [-0.111, +0.785] (1000 resamples, seed 0); excludes zero: False.
- **Kendall tau = +0.309**, 95% bootstrap CI [-0.055, +0.651].
- Point-estimate p-values (asymptotic, small-n, report only): Spearman p=0.107, Kendall p=0.091.
- Permutation test (monte_carlo, 10000 relabellings), Spearman: one-sided p=0.051, two-sided p=0.107 — a bootstrap-independent check.

**Read this honestly:** at n=17 across *different* SSL methods the overall correlation is only weakly positive on CIFAR-100 transfer, where frozen-feature probe accuracies are compressed into a narrow band. Whether its CI clears zero is reported above; the cleaner, mechanistically interpretable signal is the within-family ladder below.

## Quality ladders (RankMe's stronger claim)

Monotone sub-sequences where only training length varies, so 'quality' increases step by step and effective rank should track probe accuracy most cleanly. CI bootstrapped on the ladder itself:

- **SwAV epoch ladder (100->800ep)** (n=4): Spearman rho = +1.000, 95% bootstrap CI [+1.000, +1.000]; Kendall tau = +1.000. **Exact permutation p (24 relabellings): one-sided 0.042, two-sided 0.083.**
  - checkpoints: swav_100ep, swav_200ep, swav_400ep, swav_800ep
- **MoCo ladder (v1-200ep -> v2-200ep -> v2-800ep)** (n=3): Spearman rho = +1.000, 95% bootstrap CI [+1.000, +1.000]; Kendall tau = +1.000. **Exact permutation p (6 relabellings): one-sided 0.167, two-sided 0.333.**
  - checkpoints: mocov1_200ep, mocov2_200ep, mocov2_800ep

_Trust the permutation p over the bootstrap CI here. A perfectly monotone ladder gives rho = +1.0 on every valid bootstrap resample, so its [+1.0, +1.0] CI 'excludes zero' by construction — but the exact permutation test shows the true significance: at n=4 the best possible two-sided p is 2/24 = 0.083 and at n=3 it is 2/6 = 0.333, so these ladders are directional confirmations, NOT significant results. Only a larger group can clear p < 0.05 (see the CIFAR-10 SwAV family below)._

## Within-method family (broad, includes recipe variants)

A whole method's checkpoints, now also varying the recipe (crop count, batch size) rather than only training length. Extending n *within* SwAV forces in these variants (only 4 pure-epoch SwAV RN50 checkpoints exist), and they vary quality along axes effective rank does not track — so the broad within-family correlation is weaker and wider than the pure epoch ladder above. That contrast is itself the finding:

- **SwAV** (n=7): Spearman rho = +0.179, 95% bootstrap CI [-0.750, +1.000] (1000 resamples); Kendall tau = +0.238. **Exact permutation p (5040 relabellings): one-sided 0.357, two-sided 0.713.**
  - checkpoints: swav_100ep, swav_200ep, swav_400ep, swav_800ep, swav_400ep_2x224, swav_200ep_bs256, swav_400ep_bs256

## Negative control (random-projection embeddings)

Matched dimension (2048) and count (17); base input = flattened native-resolution (32x32) CIFAR-100 pixels, projected through independent Gaussian random matrices, run through the SAME metric + readout.

- **Control Spearman rho = +0.237**, 95% CI [-0.256, +0.671]; excludes zero: False.
- The control values are near-constant by construction — a random projection preserves the covariance spectrum, so effective rank ranges only [29.58, 33.55] (std 1.036) and probe accuracy only [0.0885, 0.1003] (std 0.0032) across the 17 encoders. Rank-correlating ~10 nearly-tied values yields an unstable point estimate; what matters is that the CI **includes zero** (`excludes_zero=False`), i.e. no evidence the pipeline manufactures correlation from structureless embeddings.

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
| swav_400ep_2x224 | SwAV | 236.42 | 0.6118 |
| swav_200ep_bs256 | SwAV | 197.89 | 0.6178 |
| swav_400ep_bs256 | SwAV | 216.86 | 0.6380 |
| deepclusterv2_400ep | DeepCluster-v2 | 196.11 | 0.6577 |
| deepclusterv2_800ep | DeepCluster-v2 | 253.73 | 0.6620 |
| selav2_400ep | SeLa-v2 | 171.70 | 0.6092 |
| mocov1_200ep | MoCo-v1 | 71.10 | 0.4420 |

## Provenance

- **Eval set:** CIFAR-100 test split, fixed seeded subset, n_eval = 8000 (seed 0).
- **Preprocessing:** Resize(224) -> ToTensor -> Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (ImageNet stats; checkpoints are ImageNet-pretrained).
- **Features:** torchvision ResNet-50 with fc=Identity; 2048-d penultimate features; every backbone weight asserted loaded (no missing keys).
- **Readout:** `embq.readouts.linear_probe_accuracy` (logistic regression, lbfgs, 50/50 stratified split, seed 0).
- **Correlation / CI / control:** `embq.harness` (seed 0, 1000 bootstrap resamples; exact permutation test enumerates all n! relabellings for n<=8, else Monte-Carlo).
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
  - `swav_400ep_2x224` — facebookresearch/swav, 400ep 2x224-crop (no multi-crop) ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_2x224_pretrain.pth.tar
  - `swav_200ep_bs256` — facebookresearch/swav, 200ep batch-256 ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_bs256_pretrain.pth.tar
  - `swav_400ep_bs256` — facebookresearch/swav, 400ep batch-256 ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_bs256_pretrain.pth.tar
  - `deepclusterv2_400ep` — facebookresearch/swav (deepcluster), DeepCluster-v2 400ep ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_pretrain.pth.tar
  - `deepclusterv2_800ep` — facebookresearch/swav (deepcluster), DeepCluster-v2 800ep ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar
  - `selav2_400ep` — facebookresearch/swav (deepcluster), SeLa-v2 400ep ResNet-50 — https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar
  - `mocov1_200ep` — facebookresearch/moco, MoCo-v1 200ep ResNet-50 — https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar

## Mechanistic interpretation

Effective rank measures how many directions of the feature covariance carry real variance — the *linear spread* of the embedding. The linear probe is a linear classifier on those same frozen features, so its accuracy depends on exactly that linear spread: a collapsed, low-effective-rank representation cannot linearly separate the classes, while a checkpoint that spreads variance over more directions gives the probe more separable structure to exploit. Metric and readout depend on the *same* property, so they track. This shows up most clearly within a single training recipe (the SwAV epoch ladder above): holding the method fixed and only increasing training, effective rank and probe accuracy rise together. Across different methods the relationship is noisier and wider — effective rank is a weaker predictor once the training recipe (not just its length) changes. This is still the mirror image of the scRNA scVI-30 failure case, where the readout (ARI clustering) depends on cluster geometry that a linear-spread metric does not see at all. Same harness both ends; the difference is mechanistic.
