"""Tests for embq.failure_modes — the FM1-FM4 diagnostic suite."""

import numpy as np
import pandas as pd
import pytest

from embq.failure_modes import (
    fm1_effrank_tracks_dimension,
    fm2_control_leverage,
    fm3_readout_dependence,
    fm4_ill_conditioned_target,
    fm4_kmeans_seed_study,
)


# --------------------------------------------------------------------------
# FM1 — effective rank tracks dimension, not quality
# --------------------------------------------------------------------------

def test_fm1_flags_control_as_top_effrank(posctrl_df):
    out = fm1_effrank_tracks_dimension(posctrl_df)
    assert out["fixed_dim30_top_is_control"] is True
    assert out["fixed_dim30_top_ari"] < 0.1


def test_fm1_effrank_tracks_dim_more_than_ari(posctrl_df):
    out = fm1_effrank_tracks_dimension(posctrl_df)
    assert out["pca_effrank_vs_dim_rho"] > abs(out["pca_effrank_vs_ari_rho"])


# --------------------------------------------------------------------------
# FM2 — control-leverage: correlation collapses once controls are dropped
# --------------------------------------------------------------------------

def test_fm2_correlation_inflated_by_controls(posctrl_df):
    out = fm2_control_leverage(posctrl_df, metric="anisotropy")
    assert out["n_with_controls"] > out["n_real_only"]
    assert out["rho_with_controls"] > out["rho_real_only"]
    assert out["attenuation"] > 0


# --------------------------------------------------------------------------
# FM3 — readout-dependence: probe and clustering can disagree on ranking
# --------------------------------------------------------------------------

def test_fm3_detects_reversal(posctrl_df):
    out = fm3_readout_dependence(posctrl_df)
    assert out["n_reversed_pairs"] >= 1
    assert out["n_encoder_pairs"] > 0
    assert out["scvi30_ari"] == pytest.approx(0.70)


def test_fm3_perfect_agreement_has_no_reversals():
    df = pd.DataFrame({
        "embedding": ["e1", "e2", "e3", "e4"],
        "group": ["real"] * 4,
        "linear_probe": [0.9, 0.8, 0.7, 0.6],
        "ari": [0.95, 0.85, 0.75, 0.65],
    })
    out = fm3_readout_dependence(df)
    assert out["n_reversed_pairs"] == 0
    assert out["probe_vs_ari_rho_real"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# FM4a — ARI variance decomposition (sweep table, few seeds)
# --------------------------------------------------------------------------

def test_fm4_within_fraction_zero_when_seeds_agree():
    # every seed gives the identical ari per encoder -> no seed noise, all
    # variance is between-encoder
    df = pd.DataFrame({
        "encoder": ["a", "a", "a", "b", "b", "b"],
        "family": ["real"] * 6,
        "ari": [0.5, 0.5, 0.5, 0.9, 0.9, 0.9],
    })
    out = fm4_ill_conditioned_target(df, n_boot=50)
    assert out["within_encoder_variance_fraction"] == pytest.approx(0.0)


def test_fm4_within_fraction_high_when_encoders_are_identical():
    # both encoders share the same mean ari; all spread is seed jitter
    df = pd.DataFrame({
        "encoder": ["a", "a", "a", "b", "b", "b"],
        "family": ["real"] * 6,
        "ari": [0.4, 0.5, 0.6, 0.4, 0.5, 0.6],
    })
    out = fm4_ill_conditioned_target(df, n_boot=50)
    assert out["within_encoder_variance_fraction"] == pytest.approx(1.0)


def test_fm4_excludes_control_family():
    df = pd.DataFrame({
        "encoder": ["a", "a", "ctrl", "ctrl"],
        "family": ["real", "real", "control", "control"],
        "ari": [0.5, 0.6, 0.0, 0.0],
    })
    out = fm4_ill_conditioned_target(df, n_boot=50)
    assert out["n_encoders"] == 1


# --------------------------------------------------------------------------
# FM4b — high-seed KMeans recomputation on saved embedding matrices
# --------------------------------------------------------------------------

def test_fm4_kmeans_seed_study_separates_real_from_control(tmp_path):
    rng = np.random.default_rng(0)
    n_per_cluster = 60
    centers = np.array([[0, 0], [10, 10]])
    labels = np.repeat([0, 1], n_per_cluster)
    good_emb = np.vstack([
        centers[c] + rng.normal(scale=0.3, size=(n_per_cluster, 2))
        for c in [0, 1]
    ])
    medium_emb = np.vstack([
        centers[c] + rng.normal(scale=4.0, size=(n_per_cluster, 2))
        for c in [0, 1]
    ])
    random_emb = rng.normal(size=(2 * n_per_cluster, 2))

    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "good_encoder.npy", good_emb)
    np.save(tmp_path / "medium_encoder.npy", medium_emb)
    np.save(tmp_path / "random.npy", random_emb)

    out = fm4_kmeans_seed_study(str(tmp_path), n_seeds=5, n_boot=50,
                                 control_names=("random",))
    assert out["n_real_encoders"] == 2
    assert out["per_encoder_ari_mean"]["good_encoder"] > 0.9
    assert "random" not in out["per_encoder_ari_mean"]


def test_fm4_kmeans_seed_study_missing_labels_raises(tmp_path):
    np.save(tmp_path / "enc.npy", np.zeros((10, 2)))
    with pytest.raises(FileNotFoundError):
        fm4_kmeans_seed_study(str(tmp_path))
