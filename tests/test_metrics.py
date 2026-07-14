"""Directionality tests: each metric must move the right way on known cases."""

import numpy as np
import pytest

from embq.metrics import (
    anisotropy,
    effective_rank,
    intrinsic_dimensionality_twonn,
    uniformity,
    alignment,
    neighborhood_consistency,
    compute_all,
)

rng = np.random.default_rng(42)
N, D = 1500, 32


@pytest.fixture
def isotropic():
    return rng.standard_normal((N, D))


@pytest.fixture
def cone():
    d = rng.standard_normal(D)
    return 0.05 * rng.standard_normal((N, D)) + d


def test_anisotropy_cone_higher(isotropic, cone):
    # collapsed cloud must be far more anisotropic than isotropic one
    assert anisotropy(cone) > 0.9
    assert anisotropy(isotropic) < 0.1


def test_effective_rank_lowrank_smaller(isotropic):
    basis = rng.standard_normal((3, D))
    low_rank = rng.standard_normal((N, 3)) @ basis
    assert effective_rank(low_rank) < effective_rank(isotropic)
    assert effective_rank(low_rank) < 6  # ~3 true dims


def test_effective_rank_vision_shape_backward_compatible():
    # Vision-shaped embeddings (large dim, e.g. ResNet-50 2048-d) must work
    # unchanged: effective_rank already takes any (n_samples, dim) matrix.
    # Covers both n < dim and n > dim, and the low-rank ordering still holds.
    for n in (500, 4000):  # n < 2048 and n > 2048
        full = rng.standard_normal((n, 2048))
        basis = rng.standard_normal((20, 2048))
        low_rank = rng.standard_normal((n, 20)) @ basis
        er_full = effective_rank(full)
        er_low = effective_rank(low_rank)
        assert np.isfinite(er_full) and np.isfinite(er_low)
        assert er_full <= min(n, 2048)          # bounded by min(n_samples, dim)
        assert er_low < er_full                  # low-rank uses fewer directions
        assert er_low < 40                        # ~20 true dims


def test_intrinsic_dim_recovers_subspace():
    basis = rng.standard_normal((5, D))
    data = rng.standard_normal((N, 5)) @ basis
    est = intrinsic_dimensionality_twonn(data)
    assert 3.0 < est < 8.0  # should land near 5


def test_uniformity_isotropic_lower(isotropic, cone):
    # spread-out cloud has lower (more negative) uniformity than a cone
    assert uniformity(isotropic) < uniformity(cone)


def test_alignment_identical_pairs_zero(isotropic):
    # identical pairs -> alignment ~ 0
    assert alignment(isotropic, isotropic.copy()) < 1e-6


def test_alignment_random_pairs_larger(isotropic):
    shuffled = isotropic[rng.permutation(N)]
    assert alignment(isotropic, shuffled) > alignment(isotropic, isotropic.copy())


def test_neighborhood_consistency_identical_is_one(isotropic):
    assert neighborhood_consistency(isotropic, isotropic.copy(), k=10) == pytest.approx(1.0)


def test_neighborhood_consistency_noise_degrades(isotropic):
    heavy = isotropic + rng.standard_normal((N, D)) * 2.0
    assert neighborhood_consistency(isotropic, heavy, k=10) < 0.5


def test_compute_all_optional_keys(isotropic):
    base = compute_all(isotropic)
    assert set(base) == {"anisotropy", "effective_rank",
                         "intrinsic_dim_twonn", "uniformity"}
    full = compute_all(isotropic, X_pair=isotropic.copy(),
                       X_perturbed=isotropic.copy())
    assert "alignment" in full and "neighborhood_consistency" in full
