"""Harness statistics + readouts: correlation, bootstrap CI, control, probe."""

import numpy as np
import pytest

from embq.harness import (
    spearman_kendall,
    bootstrap_ci,
    random_projection_embeddings,
    permutation_test,
    holm_bonferroni,
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


def test_permutation_test_exact_small_n():
    # n=6 -> exact enumeration of 720 permutations, no sampling.
    x = np.arange(6.0)
    y = x.copy()  # perfectly monotone -> only identity ordering hits rho=1
    out = permutation_test(x, y, seed=0)
    assert out["method"] == "exact"
    assert out["n_permutations"] == 720
    assert out["observed"] == pytest.approx(1.0)
    # exactly one of 720 permutations reaches rho=+1 (one-sided p = 1/720)
    assert out["p_greater"] == pytest.approx(1 / 720)
    # two orderings reach |rho|=1 (monotone up and down) -> 2/720
    assert out["p_two_sided"] == pytest.approx(2 / 720)


def test_permutation_test_null_is_not_significant():
    x = rng.standard_normal(7)
    y = rng.standard_normal(7)  # independent -> p should be far from 0
    out = permutation_test(x, y, seed=1)
    assert out["method"] == "exact"  # 7! = 5040 <= 8!
    assert out["p_two_sided"] > 0.1


def test_permutation_test_monte_carlo_large_n():
    x = np.arange(12.0)
    y = x + rng.standard_normal(12) * 0.1
    out = permutation_test(x, y, n_permutations=2000, seed=0)
    assert out["method"] == "monte_carlo"
    assert out["p_greater"] < 0.01  # strong positive relationship


def test_holm_bonferroni_step_down():
    # Worked example: p = [0.01, 0.04, 0.03, 0.005], m=4, alpha=0.05.
    # Holm rejects the two smallest (0.005, 0.01); 0.03/0.04 fail.
    out = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert out["m"] == 4
    assert out["bonferroni_threshold"] == pytest.approx(0.0125)
    assert out["reject"] == [True, False, False, True]
    assert out["adjusted"][3] == pytest.approx(0.02)   # 4 * 0.005
    assert out["adjusted"][0] == pytest.approx(0.03)   # 3 * 0.01
    assert out["any_reject"] is True


def test_holm_bonferroni_nothing_survives():
    # The RankMe family: smallest p=0.024 across m=10 -> nothing survives,
    # because 0.024 > alpha/m = 0.005 and Holm's first step fails.
    ps = [0.024, 0.107, 0.653, 0.713, 0.083, 0.333, 0.333, 1.0, 0.45, 0.6]
    out = holm_bonferroni(ps, alpha=0.05)
    assert out["any_reject"] is False
    assert min(out["adjusted"]) == pytest.approx(0.24)  # 10 * 0.024


def test_clustering_readout_recovers_blobs():
    centers = rng.standard_normal((4, 16)) * 8.0
    y = np.repeat(np.arange(4), 150)
    emb = np.vstack([centers[c] + rng.standard_normal((150, 16)) for c in range(4)])
    out = clustering_readout(emb, y, seed=0)
    assert out["ARI"] > 0.9
    assert out["NMI"] > 0.9
