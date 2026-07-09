"""Sanity check: three synthetic embedding clouds of different quality."""

import numpy as np
from embq.metrics import compute_all

rng = np.random.default_rng(0)
N, D = 2000, 64

# 1. GOOD: isotropic Gaussian, uses the whole space
good = rng.standard_normal((N, D))

# 2. COLLAPSED: everything squeezed toward one direction (cone effect)
direction = rng.standard_normal(D)
collapsed = 0.05 * rng.standard_normal((N, D)) + direction

# 3. LOW-RANK: lives on a 3D subspace embedded in 64D
basis = rng.standard_normal((3, D))
low_rank = rng.standard_normal((N, 3)) @ basis

for name, X in [("GOOD (isotropic)", good),
                ("COLLAPSED (cone)", collapsed),
                ("LOW-RANK (3D)", low_rank)]:
    m = compute_all(X)
    print(f"\n{name}")
    for k, v in m.items():
        print(f"  {k:22s} {v:+.4f}")
