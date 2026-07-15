"""
Failure-mode analysis of the label-free metric suite.

The transfer hypothesis was falsified (see the pre-defense results): no
label-free geometric metric in ``embq.metrics`` reliably predicts downstream
cell-type recovery. This module is the *diagnostic* follow-up. Instead of
asking "does the metric work?", each function isolates ONE mechanism by which
a metric produces a misleading verdict, and returns numbers that quantify it.

Everything here consumes the already-computed result tables in ``results/``
(the real scVI/AE/ICA/NMF embeddings cannot be recomputed without a GPU box),
so the diagnosis is fully reproducible from the checked-in CSVs.

Four failure modes, each answering "where and why does the metric miss?":

  FM1  effective_rank tracks embedding dimensionality, not quality.
  FM2  control-leverage: correlations survive only because uninformative
       negative controls sit in a convenient corner.
  FM3  readout-dependence: the same embedding gets opposite verdicts from
       linear probe vs. clustering ARI.
  FM4  the ARI target is ill-conditioned: compressed range + KMeans
       stochasticity, so there is little signal to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# --------------------------------------------------------------------------
# FM1 — effective rank is a proxy for dimensionality, not quality
# --------------------------------------------------------------------------

def fm1_effrank_tracks_dimension(posctrl: pd.DataFrame) -> dict:
    """Show effective_rank ~ latent dim, and that it rewards useless controls.

    Two pieces of evidence:
      * across PCA encoders, effective_rank is almost perfectly linear in the
        latent dimension (it is measuring capacity, not signal);
      * at a *fixed* latent dim (30), the uninformative controls attain the
        HIGHEST effective_rank while their ARI is ~0. A quality metric must
        not rank noise above signal.

    Returns a dict with the PCA slope/correlation and the fixed-dim ranking.
    """
    pca = posctrl[posctrl["group"] == "PCA"].sort_values("dim")
    # effective rank vs dim across the PCA quality gradient
    rho_dim, _ = spearmanr(pca["dim"], pca["effective_rank"])
    slope, intercept = np.polyfit(pca["dim"], pca["effective_rank"], 1)
    # correlation of effrank with the thing it should track (ARI) vs dim
    rho_effrank_ari, _ = spearmanr(pca["effective_rank"], pca["ari"])

    # fixed latent dim = 30: real encoder(s) vs controls
    d30 = posctrl[posctrl["dim"] == 30].copy()
    d30 = d30[["embedding", "group", "effective_rank", "ari"]]
    d30 = d30.sort_values("effective_rank", ascending=False).reset_index(drop=True)
    top = d30.iloc[0]

    return {
        "pca_effrank_vs_dim_rho": round(float(rho_dim), 3),
        "pca_effrank_vs_dim_slope": round(float(slope), 3),
        "pca_effrank_vs_ari_rho": round(float(rho_effrank_ari), 3),
        "fixed_dim30_ranking": d30,
        "fixed_dim30_top_embedding": str(top["embedding"]),
        "fixed_dim30_top_is_control": bool(top["group"] == "control"),
        "fixed_dim30_top_ari": round(float(top["ari"]), 3),
    }


# --------------------------------------------------------------------------
# FM2 — control-leverage: correlation exists only because of the controls
# --------------------------------------------------------------------------

def fm2_control_leverage(posctrl: pd.DataFrame, metric: str) -> dict:
    """Compare metric-vs-ARI correlation WITH vs WITHOUT negative controls.

    The pilot correlations (anisotropy, kNN-consistency) look strong on the
    full benchmark, but the controls occupy a single extreme corner of the
    (metric, ARI) plane. Dropping them collapses the correlation. That is the
    signature of a leverage artifact, not a real predictive signal.
    """
    full = posctrl
    real = posctrl[posctrl["group"] != "control"]

    rho_all, p_all = spearmanr(full[metric], full["ari"])
    rho_real, p_real = spearmanr(real[metric], real["ari"])

    return {
        "metric": metric,
        "rho_with_controls": round(float(rho_all), 3),
        "p_with_controls": round(float(p_all), 3),
        "n_with_controls": int(len(full)),
        "rho_real_only": round(float(rho_real), 3),
        "p_real_only": round(float(p_real), 3),
        "n_real_only": int(len(real)),
        "attenuation": round(float(rho_all - rho_real), 3),
    }


# --------------------------------------------------------------------------
# FM3 — readout-dependence: probe and clustering disagree on the same X
# --------------------------------------------------------------------------

def fm3_readout_dependence(posctrl: pd.DataFrame) -> dict:
    """Quantify how much the encoder *ranking* depends on the readout.

    linear_probe (supervised linear-separability) and ARI (unsupervised
    clustering) are different metrics on different scales, so their raw values
    are NOT directly comparable — 0.94 vs 0.54 is not a "0.40 drop". The
    defensible comparison is between the *rankings* the two readouts induce
    over the encoders: if the two "ground truths" order the encoders
    differently, no single geometric predictor can be right for both, and
    "quality" is a (task, readout) pair rather than a scalar.

    We report rank agreement (Spearman ρ and Kendall τ over the encoder ranks),
    the number of discordant (reversed) encoder pairs, and the explicit list of
    reversals — pairs A, B where the probe prefers A but ARI prefers B. scVI-30
    is kept only as one illustrative reversal, not as a scale comparison.
    """
    real = posctrl[posctrl["group"] != "control"].copy()

    # rank agreement between the two readouts (this is the honest comparison)
    rho_probe_ari, p_rho = spearmanr(real["linear_probe"], real["ari"])
    tau, p_tau = kendalltau(real["linear_probe"], real["ari"])

    # ranks: 1 = best. Compared on rank, never on raw value/scale.
    real["probe_rank"] = real["linear_probe"].rank(ascending=False)
    real["ari_rank"] = real["ari"].rank(ascending=False)
    real["rank_gap"] = (real["probe_rank"] - real["ari_rank"]).abs()

    # enumerate every encoder pair and flag the reversals (discordant pairs)
    names = real["embedding"].tolist()
    probe = dict(zip(names, real["linear_probe"]))
    ari = dict(zip(names, real["ari"]))
    reversals = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = names[i], names[j]
            # discordant: the two readouts disagree on the ordering of a, b
            if (probe[a] - probe[b]) * (ari[a] - ari[b]) < 0:
                hi_probe, lo_probe = (a, b) if probe[a] > probe[b] else (b, a)
                reversals.append({
                    "probe_prefers": hi_probe,
                    "ari_prefers": lo_probe if ari[lo_probe] > ari[hi_probe] else hi_probe,
                    "probe_gap": round(abs(probe[a] - probe[b]), 3),
                    "ari_gap": round(abs(ari[a] - ari[b]), 3),
                })
    n_pairs = n * (n - 1) // 2

    worst = real.sort_values("rank_gap", ascending=False).iloc[0]
    scvi = posctrl[posctrl["embedding"] == "scVI-30"]
    scvi_row = scvi.iloc[0] if len(scvi) else None

    return {
        "probe_vs_ari_rho_real": round(float(rho_probe_ari), 3),
        "probe_vs_ari_p_real": round(float(p_rho), 3),
        "probe_vs_ari_kendall_tau": round(float(tau), 3),
        "probe_vs_ari_kendall_p": round(float(p_tau), 3),
        "n_encoder_pairs": int(n_pairs),
        "n_reversed_pairs": int(len(reversals)),
        "reversed_pairs": reversals,
        "max_rank_gap_embedding": str(worst["embedding"]),
        "max_rank_gap": float(worst["rank_gap"]),
        "scvi30_linear_probe": None if scvi_row is None else round(float(scvi_row["linear_probe"]), 3),
        "scvi30_ari": None if scvi_row is None else round(float(scvi_row["ari"]), 3),
        "table": real[["embedding", "group", "linear_probe", "ari",
                       "probe_rank", "ari_rank", "rank_gap"]]
                 .sort_values("rank_gap", ascending=False)
                 .reset_index(drop=True),
    }


# --------------------------------------------------------------------------
# FM4 — the ARI target itself is ill-conditioned
# --------------------------------------------------------------------------

def _within_fraction(df: pd.DataFrame, enc_col: str = "encoder") -> float:
    """Fraction of total ARI variance that is *within*-encoder (seed noise)."""
    grand_mean = df["ari"].mean()
    group_means = df.groupby(enc_col)["ari"].transform("mean")
    ss_total = float(((df["ari"] - grand_mean) ** 2).sum())
    ss_within = float(((df["ari"] - group_means) ** 2).sum())
    return ss_within / ss_total if ss_total > 0 else float("nan")


def fm4_ill_conditioned_target(perseed: pd.DataFrame, n_boot: int = 3000,
                               seed: int = 0) -> dict:
    """Decompose ARI variance into between- vs within-encoder (seed noise).

    The prediction target is ARI averaged over seeds. But KMeans is stochastic:
    part of the total ARI spread is just seed-to-seed noise on a FIXED
    embedding, which is by construction unpredictable from seed-invariant
    geometry. This estimate is from the *sweep* table (``build_sweep_perseed``,
    n=14 real encoders) at only **3 seeds**, so the point estimate is fragile;
    we attach a bootstrap 95% CI (resampling encoders) to make that explicit.
    See ``fm4_kmeans_seed_study`` for the higher-seed recomputation. The
    per-encoder ARI std is panel-independent and is the robust core statistic;
    the *fraction* additionally depends on the between-encoder spread.
    """
    real = perseed[perseed["family"] != "control"].copy()
    within_frac = _within_fraction(real)

    # bootstrap CI over encoders — with 3 seeds this interval is wide on purpose
    rng = np.random.default_rng(seed)
    encoders = real["encoder"].unique()
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(encoders, size=len(encoders), replace=True)
        sub = pd.concat([real[real["encoder"] == e] for e in pick], ignore_index=True)
        sub["_eid"] = np.repeat(np.arange(len(pick)), real.groupby("encoder").size().iloc[0])
        boot.append(_within_fraction(sub, "_eid"))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    per_enc = real.groupby("encoder")["ari"].mean()
    n_seeds = int(real.groupby("encoder").size().max())
    return {
        "within_encoder_variance_fraction": round(within_frac, 3),
        "within_frac_ci95_lo": round(float(lo), 3),
        "within_frac_ci95_hi": round(float(hi), 3),
        "between_encoder_variance_fraction": round(1 - within_frac, 3),
        "ari_min_real": round(float(per_enc.min()), 3),
        "ari_max_real": round(float(per_enc.max()), 3),
        "ari_range_real": round(float(per_enc.max() - per_enc.min()), 3),
        "n_encoders": int(per_enc.size),
        "n_seeds": n_seeds,
        "per_encoder_ari_std": real.groupby("encoder")["ari"].std().round(3).to_dict(),
    }


def fm4_kmeans_seed_study(emb_dir: str, n_seeds: int = 30, n_boot: int = 3000,
                          boot_seed: int = 0, control_names=("random",)) -> dict:
    """Recompute the ARI variance decomposition at high seed count.

    Reads the saved encoder embedding matrices (``<emb_dir>/*.npy``) and the
    ground-truth labels (``<emb_dir>/labels.npy``), then reruns KMeans with
    ``n_seeds`` different seeds on each FIXED embedding. Because the geometry is
    held fixed, all ARI variation here is pure KMeans seed noise — the exact
    quantity the 3-seed sweep could only estimate crudely.

    Returns per-encoder ARI mean/std over seeds (panel-independent), the
    within-encoder variance fraction on real encoders with a bootstrap CI over
    seeds, and a tidy per-(encoder, seed) frame for plotting. Requires
    scikit-learn; raises FileNotFoundError if the embeddings are not present.
    """
    import os
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    labels_path = os.path.join(emb_dir, "labels.npy")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"{labels_path} not found — cannot compute ARI without ground-truth "
            f"labels. Save them with block2_real.py on the data machine.")
    y = np.load(labels_path)
    k = int(np.unique(y).size)

    names = sorted(f[:-4] for f in os.listdir(emb_dir)
                   if f.endswith(".npy") and f != "labels.npy")
    rows = []
    for name in names:
        X = np.load(os.path.join(emb_dir, f"{name}.npy"))
        grp = "control" if name in control_names else "real"
        for s in range(n_seeds):
            km = KMeans(n_clusters=k, n_init=10, random_state=s).fit(X)
            rows.append((name, grp, s, float(adjusted_rand_score(y, km.labels_))))
    df = pd.DataFrame(rows, columns=["encoder", "group", "seed", "ari"])

    real = df[df["group"] == "real"].copy()
    within_frac = _within_fraction(real)

    # bootstrap CI over seeds (geometry is fixed, so the seed is the unit)
    rng = np.random.default_rng(boot_seed)
    encoders = real["encoder"].unique()
    boot = []
    for _ in range(n_boot):
        pick_seeds = rng.choice(np.arange(n_seeds), size=n_seeds, replace=True)
        sub = pd.concat(
            [real[real["encoder"] == e].set_index("seed").loc[pick_seeds].reset_index()
             for e in encoders], ignore_index=True)
        boot.append(_within_fraction(sub))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    per_mean = real.groupby("encoder")["ari"].mean()
    return {
        "n_seeds": int(n_seeds),
        "n_real_encoders": int(real["encoder"].nunique()),
        "encoders": names,
        "within_encoder_variance_fraction": round(within_frac, 3),
        "within_frac_ci95_lo": round(float(lo), 3),
        "within_frac_ci95_hi": round(float(hi), 3),
        "between_encoder_variance_fraction": round(1 - within_frac, 3),
        "ari_min_real": round(float(per_mean.min()), 3),
        "ari_max_real": round(float(per_mean.max()), 3),
        "ari_range_real": round(float(per_mean.max() - per_mean.min()), 3),
        "per_encoder_ari_mean": per_mean.round(4).to_dict(),
        "per_encoder_ari_std": df.groupby("encoder")["ari"].std().round(4).to_dict(),
        "table": df,
    }
