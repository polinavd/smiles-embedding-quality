"""Shared pytest fixtures for the FM1-FM4 failure-mode tests."""

import pandas as pd
import pytest


@pytest.fixture
def posctrl_df() -> pd.DataFrame:
    """Small synthetic stand-in for results/posctrl_full_table.csv.

    Shaped to reproduce, in miniature, the three real findings FM1-FM3 check
    for: a PCA sweep where effective_rank tracks dim (not ari); a metric
    (anisotropy) whose apparent correlation with ari is a control-leverage
    artifact; and readout disagreement (linear_probe vs. ari) with at least
    one reversed encoder pair.
    """
    return pd.DataFrame([
        {"embedding": "PCA-5", "group": "PCA", "dim": 5, "effective_rank": 5,
         "ari": 0.30, "linear_probe": 0.50, "anisotropy": 0.60},
        {"embedding": "PCA-10", "group": "PCA", "dim": 10, "effective_rank": 10,
         "ari": 0.60, "linear_probe": 0.55, "anisotropy": 0.30},
        {"embedding": "PCA-20", "group": "PCA", "dim": 20, "effective_rank": 19,
         "ari": 0.35, "linear_probe": 0.60, "anisotropy": 0.70},
        {"embedding": "PCA-30", "group": "PCA", "dim": 30, "effective_rank": 29,
         "ari": 0.50, "linear_probe": 0.75, "anisotropy": 0.40},
        {"embedding": "NMF-30", "group": "NMF", "dim": 30, "effective_rank": 25,
         "ari": 0.85, "linear_probe": 0.65, "anisotropy": 0.50},
        {"embedding": "ICA-30", "group": "ICA", "dim": 30, "effective_rank": 27,
         "ari": 0.62, "linear_probe": 0.90, "anisotropy": 0.55},
        {"embedding": "scVI-30", "group": "scVI", "dim": 30, "effective_rank": 28,
         "ari": 0.70, "linear_probe": 0.80, "anisotropy": 0.45},
        {"embedding": "control-30", "group": "control", "dim": 30, "effective_rank": 31,
         "ari": 0.02, "linear_probe": 0.05, "anisotropy": 0.05},
    ])
