"""
Failure-mode analysis driver.

Runs the four diagnostics in ``embq.failure_modes`` over the checked-in result
tables, prints a readable report, and writes one CSV + one figure per failure
mode into ``results/``.

    python scripts/failure_mode_analysis.py

Outputs:
    results/fm1_effrank_vs_dim.csv        results/fm1_effrank_vs_dim.png
    results/fm2_control_leverage.csv      results/fm2_control_leverage.png
    results/fm3_readout_dependence.csv    results/fm3_readout_dependence.png
    results/fm4_target_conditioning.csv   results/fm4_target_conditioning.png
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from embq.failure_modes import (
    fm1_effrank_tracks_dimension,
    fm2_control_leverage,
    fm3_readout_dependence,
    fm4_ill_conditioned_target,
    fm4_kmeans_seed_study,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
# FM1/FM2/FM3 and the sweep-based half of FM4 all read from the SAME table so
# that "n" is not silently different per failure mode. build_sweep_averaged
# has 14 real encoders + 3 controls (posctrl_full_table had only 8 real + 3
# controls) and carries every column FM1-3 need (anisotropy, effective_rank,
# linear_probe, ari, ...). Renamed to match the group/embedding naming the
# analysis functions expect.
UNIFIED = os.path.join(RESULTS, "build_sweep_averaged.csv")
PERSEED = os.path.join(RESULTS, "build_sweep_perseed.csv")
EMB_DIR = os.path.join(RESULTS, "embeddings")
N_KMEANS_SEEDS = 30

# consistent colour language across all four figures
C_REAL = "#2c6fbb"    # real encoders (blue)
C_CTRL = "#c23b3b"    # negative controls (red)
C_SCVI = "#e08b1a"    # the scVI-30 headline case (orange)


def _save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# FM1
# --------------------------------------------------------------------------

def run_fm1(posctrl):
    print("\n" + "=" * 72)
    print("FM1  effective_rank tracks dimensionality, not quality")
    print("=" * 72)
    r = fm1_effrank_tracks_dimension(posctrl)
    print(f"  PCA: Spearman(effrank, dim)  = {r['pca_effrank_vs_dim_rho']:+.3f}"
          f"   (slope {r['pca_effrank_vs_dim_slope']:.3f} rank/dim)")
    print(f"  PCA: Spearman(effrank, ARI)  = {r['pca_effrank_vs_ari_rho']:+.3f}"
          f"   (weaker: it is not tracking quality)")
    print(f"  At fixed dim=30 the highest effective_rank belongs to "
          f"'{r['fixed_dim30_top_embedding']}' "
          f"(control={r['fixed_dim30_top_is_control']}, ARI={r['fixed_dim30_top_ari']:.3f})")
    print("  -> effective_rank rewards a noise embedding above every real one.")

    r["fixed_dim30_ranking"].to_csv(
        os.path.join(RESULTS, "fm1_effrank_vs_dim.csv"), index=False)

    pca = posctrl[posctrl["group"] == "PCA"].sort_values("dim")
    d30 = r["fixed_dim30_ranking"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(pca["dim"], pca["effective_rank"], "o-", color=C_REAL)
    lim = [0, pca["dim"].max() * 1.05]
    ax1.plot(lim, lim, "--", color="gray", lw=1, label="effrank = dim")
    ax1.set_xlabel("latent dimension"); ax1.set_ylabel("effective rank")
    ax1.set_title("(a) PCA: effective rank ≈ latent dim")
    ax1.legend(frameon=False)

    colors = [C_CTRL if g == "control" else C_REAL for g in d30["group"]]
    y = np.arange(len(d30))
    ax2.barh(y, d30["effective_rank"], color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{e}  (ARI={a:.2f})"
                         for e, a in zip(d30["embedding"], d30["ari"])], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("effective rank")
    ax2.set_title("(b) fixed dim=30: red=control, blue=real")
    fig.suptitle("FM1 — effective_rank measures capacity, not quality",
                 fontweight="bold")
    print("  wrote", _save(fig, "fm1_effrank_vs_dim.png"))


# --------------------------------------------------------------------------
# FM2
# --------------------------------------------------------------------------

def run_fm2(posctrl):
    print("\n" + "=" * 72)
    print("FM2  control-leverage: correlation vanishes without the controls")
    print("=" * 72)
    metrics = ["anisotropy", "knn_consistency", "uniformity", "alignment"]
    rows = [fm2_control_leverage(posctrl, m) for m in metrics]
    df = pd.DataFrame(rows)
    for _, row in df.iterrows():
        print(f"  {row['metric']:16s}  rho(all n={row['n_with_controls']})="
              f"{row['rho_with_controls']:+.3f}"
              f"   ->  rho(real n={row['n_real_only']})={row['rho_real_only']:+.3f}"
              f"   (attenuation {row['attenuation']:+.3f})")
    df.to_csv(os.path.join(RESULTS, "fm2_control_leverage.csv"), index=False)

    # figure: the headline pair, anisotropy, showing the leverage corner
    m = "anisotropy"
    real = posctrl[posctrl["group"] != "control"]
    ctrl = posctrl[posctrl["group"] == "control"]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.scatter(real[m], real["ari"], color=C_REAL, s=60, label="real encoders")
    ax.scatter(ctrl[m], ctrl["ari"], color=C_CTRL, s=90, marker="X",
               label="negative controls")
    # regression lines with and without controls
    xs = np.linspace(posctrl[m].min(), posctrl[m].max(), 50)
    for subset, col, lab in [(posctrl, "gray", "fit incl. controls"),
                             (real, C_REAL, "fit real only")]:
        a, b = np.polyfit(subset[m], subset["ari"], 1)
        ax.plot(xs, a * xs + b, "--", color=col, lw=1.5, label=lab)
    ax.set_xlabel(f"{m}"); ax.set_ylabel("downstream ARI")
    ax.set_title("FM2 — anisotropy–ARI is a control-leverage artifact",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    print("  wrote", _save(fig, "fm2_control_leverage.png"))


# --------------------------------------------------------------------------
# FM3
# --------------------------------------------------------------------------

def run_fm3(posctrl):
    print("\n" + "=" * 72)
    print("FM3  readout-dependence: probe and clustering RANK encoders differently")
    print("=" * 72)
    r = fm3_readout_dependence(posctrl)
    print(f"  rank agreement on real encoders (n={len(posctrl[posctrl['group']!='control'])}):")
    print(f"    Spearman(probe, ARI) = {r['probe_vs_ari_rho_real']:+.3f} "
          f"(p={r['probe_vs_ari_p_real']:.3f})   "
          f"Kendall tau = {r['probe_vs_ari_kendall_tau']:+.3f} "
          f"(p={r['probe_vs_ari_kendall_p']:.3f})")
    print(f"  {r['n_reversed_pairs']} of {r['n_encoder_pairs']} encoder pairs are "
          f"REVERSED (probe and ARI disagree on which is better):")
    for rv in r["reversed_pairs"]:
        print(f"    probe prefers {rv['probe_prefers']:8s} (+{rv['probe_gap']:.3f} probe) "
              f"but ARI prefers {rv['ari_prefers']:8s} (+{rv['ari_gap']:.3f} ARI)")
    print(f"  illustrative reversal — scVI-30: high probe rank, low ARI rank "
          f"(probe {r['scvi30_linear_probe']:.2f}, ARI {r['scvi30_ari']:.2f}; "
          f"note: different scales, compared by RANK not value)")
    r["table"].to_csv(os.path.join(RESULTS, "fm3_readout_dependence.csv"), index=False)

    # Figure: rank vs rank (comparable), NOT raw value vs value (different scales).
    real = r["table"]
    n = len(real)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([1, n], [1, n], "--", color="gray", lw=1, zorder=1,
            label="readouts agree")
    for _, row in real.iterrows():
        is_scvi = row["embedding"] == "scVI-30"
        reversed_pair = row["rank_gap"] > 0
        ax.scatter(row["probe_rank"], row["ari_rank"],
                   color=C_SCVI if is_scvi else (C_CTRL if reversed_pair else C_REAL),
                   s=110 if is_scvi else 70, zorder=3)
        ax.annotate(row["embedding"], (row["probe_rank"], row["ari_rank"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    color=C_SCVI if is_scvi else "black",
                    fontweight="bold" if is_scvi else "normal")
    ax.set_xlabel("rank by linear probe  (1 = best separability)")
    ax.set_ylabel("rank by ARI  (1 = best clustering)")
    ax.set_title("FM3 — encoder ranks disagree across readouts\n"
                 f"Spearman {r['probe_vs_ari_rho_real']:+.2f}, "
                 f"{r['n_reversed_pairs']}/{r['n_encoder_pairs']} pairs reversed",
                 fontweight="bold", fontsize=11)
    ax.invert_xaxis(); ax.invert_yaxis()   # best (rank 1) at top-right
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    print("  wrote", _save(fig, "fm3_readout_dependence.png"))


# --------------------------------------------------------------------------
# FM4
# --------------------------------------------------------------------------

def run_fm4(perseed):
    print("\n" + "=" * 72)
    print("FM4  the ARI target is ill-conditioned (compressed + stochastic)")
    print("=" * 72)

    # low-seed sweep estimate (n=14 real encoders, 3 seeds) — with bootstrap CI
    r3 = fm4_ill_conditioned_target(perseed)
    print(f"  [sweep table: n={r3['n_encoders']} real encoders, "
          f"{r3['n_seeds']} seeds]")
    print(f"    within-encoder KMeans-noise fraction = "
          f"{r3['within_encoder_variance_fraction']:.3f}  "
          f"95% CI [{r3['within_frac_ci95_lo']:.3f}, {r3['within_frac_ci95_hi']:.3f}]"
          f"  (wide: only {r3['n_seeds']} seeds)")

    # high-seed recomputation on the saved fixed embeddings, if available
    r = None
    try:
        r = fm4_kmeans_seed_study(EMB_DIR, n_seeds=N_KMEANS_SEEDS)
        print(f"  [saved embeddings: n={r['n_real_encoders']} real encoders, "
              f"{r['n_seeds']} KMeans seeds on fixed geometry]")
        print(f"    within-encoder KMeans-noise fraction = "
              f"{r['within_encoder_variance_fraction']:.3f}  "
              f"95% CI [{r['within_frac_ci95_lo']:.3f}, {r['within_frac_ci95_hi']:.3f}]")
        print("    per-encoder ARI std over seeds (panel-independent, the robust core):")
        for enc, sd in r["per_encoder_ari_std"].items():
            print(f"      {enc:10s} std={sd:.3f}")
        print(f"    real-encoder ARI range: [{r['ari_min_real']:.2f}, "
              f"{r['ari_max_real']:.2f}]  (span {r['ari_range_real']:.2f})")
    except FileNotFoundError as e:
        print(f"  [saved-embedding recomputation skipped: {e}]")
        print("   -> using the 3-seed sweep estimate only; add labels.npy to recompute.")

    # persist: prefer the high-seed study, keep the 3-seed row labelled
    out_rows = [{"source": f"sweep_table_{r3['n_seeds']}seed_n{r3['n_encoders']}",
                 **{k: v for k, v in r3.items() if k != "per_encoder_ari_std"}}]
    if r is not None:
        out_rows.append({
            "source": f"saved_emb_{r['n_seeds']}seed_n{r['n_real_encoders']}",
            "within_encoder_variance_fraction": r["within_encoder_variance_fraction"],
            "within_frac_ci95_lo": r["within_frac_ci95_lo"],
            "within_frac_ci95_hi": r["within_frac_ci95_hi"],
            "between_encoder_variance_fraction": r["between_encoder_variance_fraction"],
            "ari_min_real": r["ari_min_real"], "ari_max_real": r["ari_max_real"],
            "ari_range_real": r["ari_range_real"],
            "n_encoders": r["n_real_encoders"], "n_seeds": r["n_seeds"]})
    pd.DataFrame(out_rows).to_csv(
        os.path.join(RESULTS, "fm4_target_conditioning.csv"), index=False)

    # figure: (a) per-encoder ARI spread over the high-seed run; (b) decomposition
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if r is not None:
        tab = r["table"]
        order = tab.groupby("encoder")["ari"].mean().sort_values().index
        data = [tab[tab["encoder"] == e]["ari"].values for e in order]
        cols = [C_CTRL if tab[tab["encoder"] == e]["group"].iloc[0] == "control"
                else C_REAL for e in order]
        ax1.boxplot(data, showfliers=False)
        for i, (d, c) in enumerate(zip(data, cols), start=1):
            ax1.scatter(np.full(len(d), i), d, color=c, s=18, zorder=3, alpha=0.6)
        ax1.set_xticks(np.arange(1, len(order) + 1))
        ax1.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("ARI (per KMeans seed)")
        ax1.set_title(f"(a) ARI spread over {r['n_seeds']} seeds (fixed geometry)")
        wf, lo, hi = (r["within_encoder_variance_fraction"],
                      r["within_frac_ci95_lo"], r["within_frac_ci95_hi"])
        nseed = r["n_seeds"]
    else:
        real = perseed[perseed["family"] != "control"]
        order = real.groupby("encoder")["ari"].mean().sort_values().index
        data = [real[real["encoder"] == e]["ari"].values for e in order]
        ax1.boxplot(data, showfliers=False)
        ax1.scatter(np.repeat(np.arange(1, len(order) + 1), [len(d) for d in data]),
                    np.concatenate(data), color=C_REAL, s=25, zorder=3)
        ax1.set_xticks(np.arange(1, len(order) + 1))
        ax1.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("ARI (per seed)")
        ax1.set_title("(a) per-encoder ARI spread across seeds")
        wf, lo, hi = (r3["within_encoder_variance_fraction"],
                      r3["within_frac_ci95_lo"], r3["within_frac_ci95_hi"])
        nseed = r3["n_seeds"]

    ax2.bar(["within-encoder\n(seed noise)", "between-encoder\n(signal)"],
            [wf, 1 - wf], color=[C_CTRL, C_REAL])
    ax2.errorbar([0], [wf], yerr=[[wf - lo], [hi - wf]], fmt="none",
                 ecolor="black", capsize=6, lw=1.5)
    ax2.set_ylabel("fraction of total ARI variance")
    ax2.set_ylim(0, 1)
    ax2.set_title(f"(b) decomposition: within = {wf:.0%} "
                  f"[{lo:.0%}, {hi:.0%}], {nseed} seeds")
    fig.suptitle("FM4 — compressed, stochastic ARI target", fontweight="bold")
    print("  wrote", _save(fig, "fm4_target_conditioning.png"))


def main():
    posctrl = (pd.read_csv(UNIFIED)
               .rename(columns={"encoder": "embedding", "family": "group"}))
    perseed = pd.read_csv(PERSEED)
    run_fm1(posctrl)
    run_fm2(posctrl)
    run_fm3(posctrl)
    run_fm4(perseed)
    print("\nDone. Four CSVs + four PNGs written to results/.")


if __name__ == "__main__":
    main()
