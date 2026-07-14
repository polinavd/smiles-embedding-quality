"""Harness statistics + readouts: correlation, bootstrap CI, control, probe."""

import numpy as np
import pytest

from embq.harness import (
    spearman_kendall,
    bootstrap_ci,
    random_projection_embeddings,
)
from embq.readouts import linear_probe_accuracy, clustering_readout

rng = np.random.default_rng(7)


def test_spearman_kendall_reports_n_and_sign():
    x = np.arange(10.0)
    y = 2 * x + 1  # perfectly monotone increasing
    out = spearman_kendall(x, y)
    assert out["n"] == 10
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["kendall_tau"] == pytest.approx(1.0)


def test_spearman_kendall_requires_equal_length_1d():
    with pytest.raises(ValueError):
        spearman_kendall(np.arange(5.0), np.arange(4.0))


def test_bootstrap_ci_excludes_zero_when_correlated():
    x = np.arange(30.0)
    y = x + rng.standard_normal(30) * 0.1  # tight positive relationship
    out = bootstrap_ci(x, y, n_resamples=2000, seed=0)
    assert out["n"] == 30
    assert out["n_resamples"] > 0
    assert out["ci_lo"] <= out["rho"] <= out["ci_hi"]
    assert out["rho"] > 0.9
    assert out["excludes_zero"] is True


def test_bootstrap_ci_straddles_zero_when_random():
    x = rng.standard_normal(40)
    y = rng.standard_normal(40)  # independent
    out = bootstrap_ci(x, y, n_resamples=2000, seed=1)
    assert out["ci_lo"] < 0 < out["ci_hi"]
    assert out["excludes_zero"] is False


def test_bootstrap_ci_kendall_statistic():
    x = np.arange(20.0)
    y = x ** 2  # monotone -> tau = 1
    out = bootstrap_ci(x, y, statistic="kendall", n_resamples=1000, seed=0)
    assert out["statistic"] == "kendall"
    assert out["rho"] == pytest.approx(1.0)


def test_random_projection_embeddings_shapes():
    base = rng.standard_normal((200, 64))
    embs = random_projection_embeddings(base, out_dim=16, n_encoders=5, seed=0)
    assert len(embs) == 5
    assert all(e.shape == (200, 16) for e in embs)
    # independent projections -> not identical
    assert not np.allclose(embs[0], embs[1])


def test_linear_probe_separable_vs_random():
    # two well-separated blobs -> probe near-perfect
    centers = rng.standard_normal((3, 20)) * 6.0
    y = np.repeat(np.arange(3), 200)
    feats = np.vstack([centers[c] + rng.standard_normal((200, 20)) for c in range(3)])
    acc = linear_probe_accuracy(feats, y, seed=0)
    assert acc > 0.9

    # random features vs labels -> near chance (1/3)
    rand_feats = rng.standard_normal((600, 20))
    chance = linear_probe_accuracy(rand_feats, y, seed=0)
    assert chance < 0.6


def test_clustering_readout_recovers_blobs():
    centers = rng.standard_normal((4, 16)) * 8.0
    y = np.repeat(np.arange(4), 150)
    emb = np.vstack([centers[c] + rng.standard_normal((150, 16)) for c in range(4)])
    out = clustering_readout(emb, y, seed=0)
    assert out["ARI"] > 0.9
    assert out["NMI"] > 0.9
