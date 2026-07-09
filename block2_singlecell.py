"""
Block 2 (single-cell variant): does unsupervised geometry predict
downstream biological quality?

We simulate embeddings of KNOWN cell types at varying quality levels,
then check whether the Block-1 metrics rank-correlate with real
downstream scores (ARI / kNN accuracy) computed against the labels.

No torch, no heavy deps. Runs in seconds. This is the shape of the
real experiment; swap the synthetic generator for scVI/PCA/etc later.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from embq.metrics import compute_all

rng = np.random.default_rng(0)


def make_embedding(n_types=6, per_type=300, dim=32, separation=1.0,
                   noise=1.0, dead_dims=0, seed=0):
    """Synthetic cell-type embedding of tunable quality.

    separation : distance between cell-type centroids (higher = cleaner)
    noise      : within-type scatter (higher = messier)
    dead_dims  : dims carrying no signal (simulates wasted capacity)
    """
    r = np.random.default_rng(seed)
    centroids = r.standard_normal((n_types, dim)) * separation
    X, y = [], []
    for t in range(n_types):
        pts = centroids[t] + r.standard_normal((per_type, dim)) * noise
        X.append(pts)
        y += [t] * per_type
    X = np.vstack(X)
    y = np.array(y)
    if dead_dims > 0:
        X[:, :dead_dims] = r.standard_normal((X.shape[0], dead_dims)) * 5.0
    # shuffle
    idx = r.permutation(len(y))
    return X[idx], y[idx]


def downstream_scores(X, y):
    """Ground-truth quality using the labels (this is what we want to predict)."""
    n_types = len(np.unique(y))
    km = KMeans(n_clusters=n_types, n_init=10, random_state=0).fit(X)
    ari = adjusted_rand_score(y, km.labels_)
    knn = KNeighborsClassifier(n_neighbors=15)
    knn_acc = cross_val_score(knn, X, y, cv=3).mean()
    return {"ARI": ari, "kNN_acc": knn_acc}


def perturb(X, sigma=0.3, seed=1):
    r = np.random.default_rng(seed)
    return X + r.standard_normal(X.shape) * sigma


# --- build a spectrum of embeddings from great to terrible -------------
configs = [
    dict(separation=3.0, noise=0.5, dead_dims=0,  seed=1),   # excellent
    dict(separation=2.0, noise=0.8, dead_dims=0,  seed=2),   # good
    dict(separation=1.5, noise=1.0, dead_dims=4,  seed=3),   # ok
    dict(separation=1.0, noise=1.2, dead_dims=8,  seed=4),   # mediocre
    dict(separation=0.6, noise=1.5, dead_dims=12, seed=5),   # poor
    dict(separation=0.3, noise=2.0, dead_dims=16, seed=6),   # bad
]

rows = []
print(f"{'cfg':>4} | {'ARI':>6} {'kNN':>6} | "
      f"{'anis':>6} {'eff_rk':>7} {'idim':>6} {'unif':>6} {'nn_cons':>7}")
print("-" * 70)

for i, cfg in enumerate(configs):
    X, y = make_embedding(**cfg)
    down = downstream_scores(X, y)
    m = compute_all(X, X_perturbed=perturb(X))
    rows.append({**down, **m})
    print(f"{i:>4} | {down['ARI']:>6.3f} {down['kNN_acc']:>6.3f} | "
          f"{m['anisotropy']:>6.3f} {m['effective_rank']:>7.2f} "
          f"{m['intrinsic_dim_twonn']:>6.2f} {m['uniformity']:>6.2f} "
          f"{m['neighborhood_consistency']:>7.3f}")

# --- correlate each unsupervised metric with downstream ARI ------------
print("\nSpearman correlation with downstream ARI (n=6 embeddings):")
ari = np.array([r["ARI"] for r in rows])
for metric in ["anisotropy", "effective_rank", "intrinsic_dim_twonn",
               "uniformity", "neighborhood_consistency"]:
    vals = np.array([r[metric] for r in rows])
    rho, p = spearmanr(vals, ari)
    print(f"  {metric:24s} rho={rho:+.3f}  (p={p:.3f})")
