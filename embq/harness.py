"""
embq-harness statistics: the single, domain-general evaluation core.

Every metric x readout evaluation (scRNA, vision, text) routes correlation,
bootstrap CI, and the random-projection negative control through THESE
functions, so a "metric works" result and a "metric fails" result are computed
with identical machinery (the embq-harness recipe).

Nothing here knows about a domain. Callers supply two 1-D arrays (metric value
and readout value, one entry per encoder/checkpoint); the negative control
helper supplies matched random-projection embeddings to run back through the
same metric + readout path.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, kendalltau


def _spearman(x, y):
    return float(spearmanr(x, y).statistic)


def _kendall(x, y):
    return float(kendalltau(x, y).statistic)


_STATS = {"spearman": _spearman, "kendall": _kendall}


def spearman_kendall(x, y):
    """Both rank correlations between a metric and a readout, plus n.

    Reports Spearman rho and Kendall tau (tau is more stable at the small n we
    always have here). n is returned so it can never be dropped from a report.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    sp = spearmanr(x, y)
    kt = kendalltau(x, y)
    return {
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
        "n": int(x.shape[0]),
    }


def bootstrap_ci(x, y, statistic="spearman", n_resamples=1000, ci=0.95, seed=0):
    """Percentile bootstrap CI for a rank correlation, resampling encoders.

    Resample the (metric, readout) pairs with replacement (n_resamples >= 1000
    by convention), recompute the correlation each time, and take the percentile
    interval. Matches the column contract of results/build_bootstrap_family.csv
    (rho, ci_lo, ci_hi, excludes_zero).

    A point estimate without this CI is not a result: with n ~ 6-15 real
    encoders the interval is wide, and that width is part of the finding.
    """
    if statistic not in _STATS:
        raise ValueError(f"statistic must be one of {sorted(_STATS)}")
    fn = _STATS[statistic]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    boots = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        # A resample with a constant column has an undefined rank correlation.
        if np.unique(xs).size < 2 or np.unique(ys).size < 2:
            continue
        val = fn(xs, ys)
        if np.isfinite(val):
            boots.append(val)
    boots = np.asarray(boots, dtype=float)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    point = fn(x, y)
    return {
        "statistic": statistic,
        "rho": float(point),
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n": int(n),
        "n_resamples": int(boots.size),
        "ci": float(ci),
    }


def random_projection_embeddings(base, out_dim, n_encoders, seed=0):
    """Matched random-projection embeddings for the negative control.

    Produces `n_encoders` embeddings of dimension `out_dim` (matched to the real
    encoders) by projecting a fixed `base` input matrix (n_samples x n_features)
    through independent Gaussian random matrices. Feed each back through the SAME
    metric and readout as the real embeddings, then correlate: expect rho ~ 0
    with a CI straddling zero. If the control correlates, the pipeline is leaking
    and the main result is void.
    """
    base = np.asarray(base, dtype=float)
    if base.ndim != 2:
        raise ValueError("base must be a 2-D (n_samples, n_features) matrix")
    n_features = base.shape[1]
    rng = np.random.default_rng(seed)
    embeddings = []
    for _ in range(n_encoders):
        proj = rng.standard_normal((n_features, out_dim)) / np.sqrt(n_features)
        embeddings.append(base @ proj)
    return embeddings
