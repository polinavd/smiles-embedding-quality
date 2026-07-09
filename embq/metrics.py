"""
Unsupervised embedding quality metrics (Block 1).

Every metric takes an embedding matrix X of shape (n_samples, dim)
and returns a single float. No labels, no downstream task required.
"""

import numpy as np


def _l2_normalize(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def anisotropy(X, n_pairs=10000, seed=0):
    """Mean cosine similarity between random pairs.

    ~0.0  -> vectors point in all directions (good, isotropic).
    ->1.0 -> everything collapsed into a narrow cone (bad).
    """
    rng = np.random.default_rng(seed)
    Xn = _l2_normalize(X)
    n = Xn.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    sims = np.sum(Xn[i[mask]] * Xn[j[mask]], axis=1)
    return float(np.mean(sims))


def effective_rank(X):
    """Spectral entropy of the covariance -> effective number of used dimensions.

    Higher = more directions carry real variance (good).
    Bounded above by min(n_samples, dim).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total <= 0:
        return 0.0
    p = eigvals / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def intrinsic_dimensionality_twonn(X, discard_fraction=0.1, seed=0):
    """TwoNN estimator (Facco et al. 2017).

    Uses the ratio of distances to the 1st and 2nd nearest neighbour.
    Returns the estimated dimension of the manifold the data lives on.
    """
    from scipy.spatial import cKDTree

    n = X.shape[0]
    tree = cKDTree(X)
    # k=3 -> self + two nearest neighbours
    dists, _ = tree.query(X, k=3)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    valid = r1 > 0
    mu = r2[valid] / r1[valid]
    mu = np.sort(mu)
    # discard the largest ratios (noise) before the linear fit
    keep = int((1 - discard_fraction) * len(mu))
    mu = mu[:keep]
    F = np.arange(1, len(mu) + 1) / len(mu)
    x = np.log(mu)
    y = -np.log(1 - F + 1e-12)
    # slope through origin -> intrinsic dimension
    d = float(np.sum(x * y) / np.sum(x * x))
    return d


def uniformity(X, t=2.0, n_pairs=10000, seed=0):
    """Uniformity term from Wang & Isola (2020).

    Lower (more negative) = points spread evenly over the hypersphere (good).
    """
    rng = np.random.default_rng(seed)
    Xn = _l2_normalize(X)
    n = Xn.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    sq_dist = np.sum((Xn[i[mask]] - Xn[j[mask]]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-t * sq_dist))))


def alignment(X_a, X_b, alpha=2.0):
    """Alignment term from Wang & Isola (2020). PAIRED mode.

    X_a[i] and X_b[i] must be embeddings of a 'similar' pair
    (augmentation, paraphrase, positive match, etc).

    Lower = positive pairs land close together (good).
    """
    Xa = _l2_normalize(X_a)
    Xb = _l2_normalize(X_b)
    dist = np.sum((Xa - Xb) ** 2, axis=1)
    return float(np.mean(dist ** (alpha / 2)))


def neighborhood_consistency(X, X_perturbed, k=10):
    """Stability of the k-NN graph under perturbation/augmentation.

    For each point, fraction of its k nearest neighbours that stay
    neighbours after perturbation, averaged over all points.

    ->1.0 = local structure is robust (good).  ->0.0 = fragile.
    """
    from scipy.spatial import cKDTree

    def knn_sets(M):
        tree = cKDTree(M)
        _, idx = tree.query(M, k=k + 1)  # +1 for self
        return [set(row[1:]) for row in idx]

    a = knn_sets(X)
    b = knn_sets(X_perturbed)
    overlaps = [len(sa & sb) / k for sa, sb in zip(a, b)]
    return float(np.mean(overlaps))


def compute_all(X, X_pair=None, X_perturbed=None):
    """Run the full Block 1 metric suite on one embedding matrix.

    Unpaired metrics always run. If X_pair is given (matched rows),
    alignment is added. If X_perturbed is given, neighborhood
    consistency is added.
    """
    out = {
        "anisotropy": anisotropy(X),
        "effective_rank": effective_rank(X),
        "intrinsic_dim_twonn": intrinsic_dimensionality_twonn(X),
        "uniformity": uniformity(X),
    }
    if X_pair is not None:
        out["alignment"] = alignment(X, X_pair)
    if X_perturbed is not None:
        out["neighborhood_consistency"] = neighborhood_consistency(X, X_perturbed)
    return out
