"""
Block 2 — REAL single-cell pipeline (GPU-ready).

Flow:
  AnnData (raw counts + cell-type labels)
     -> several ENCODERS produce embeddings of differing quality
     -> unsupervised metrics (Block 1) on each embedding      [no labels]
     -> downstream metrics (ARI / NMI / kNN acc) vs labels    [ground truth]
     -> Spearman correlation: do the unsupervised metrics rank
        the encoders the same way the downstream scores do?

The encoder registry is deliberately pluggable. PCA and a random
baseline run anywhere. scVI/scanpy encoders are wrapped in try/except
so the file imports cleanly even without torch installed — fill them
in on your GPU box.

Usage:
    python block2_real.py --h5ad data/pbmc.h5ad --label cell_type
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

from embq.metrics import compute_all


# --------------------------------------------------------------------------
# Encoders: each takes raw counts (n_cells x n_genes) -> embedding (n_cells x d)
# --------------------------------------------------------------------------

def encode_pca(counts: np.ndarray, n_comps: int = 50, seed: int = 0) -> np.ndarray:
    """Log-normalize + PCA. The classic scanpy default, no deep learning."""
    from sklearn.decomposition import PCA

    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    x = np.log1p(counts / lib * 1e4)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    return PCA(n_components=n_comps, random_state=seed).fit_transform(x)


def encode_random(counts: np.ndarray, n_comps: int = 50, seed: int = 0) -> np.ndarray:
    """Random projection — deliberately weak lower-bound baseline."""
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((counts.shape[1], n_comps))
    return np.log1p(counts) @ proj


def encode_scvi(counts: np.ndarray, n_latent: int = 30, seed: int = 0,
                max_epochs: int = 200) -> np.ndarray:
    """scVI latent space. Requires scvi-tools + torch (GPU strongly preferred).

    Left as a thin wrapper: uncomment on your machine. Kept import-safe
    so the pipeline runs with PCA/random even where torch is absent.
    """
    try:
        import scvi
        import anndata as ad
    except ImportError as e:
        raise RuntimeError(
            "scvi-tools/anndata not installed. Run this encoder on the GPU box, "
            "or drop it from ENCODERS to test with PCA only."
        ) from e

    adata = ad.AnnData(counts.astype("float32"))
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    return model.get_latent_representation()


# Add / remove encoders here. Keys become row labels in the results table.
ENCODERS: dict[str, Callable[..., np.ndarray]] = {
    "random": encode_random,
    "pca_10": lambda c: encode_pca(c, n_comps=10),
    "pca_50": lambda c: encode_pca(c, n_comps=50),
    "scvi_30": lambda c: encode_scvi(c, n_latent=30),
}


# --------------------------------------------------------------------------
# Downstream ground truth (uses labels)
# --------------------------------------------------------------------------

def downstream_scores(emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    n_types = len(np.unique(labels))
    km = KMeans(n_clusters=n_types, n_init=10, random_state=0).fit(emb)
    knn = KNeighborsClassifier(n_neighbors=15)
    return {
        "ARI": adjusted_rand_score(labels, km.labels_),
        "NMI": normalized_mutual_info_score(labels, km.labels_),
        "kNN_acc": float(cross_val_score(knn, emb, labels, cv=3).mean()),
    }


def perturb(emb: np.ndarray, sigma: float = 0.3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return emb + rng.standard_normal(emb.shape) * emb.std() * sigma


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

@dataclass
class Result:
    encoder: str
    downstream: dict[str, float]
    unsupervised: dict[str, float]


def run(counts: np.ndarray, labels: np.ndarray, save_dir: str | None = None) -> list[Result]:
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

    results = []
    for name, fn in ENCODERS.items():
        emb = fn(counts)
        down = downstream_scores(emb, labels)
        uns = compute_all(emb, X_perturbed=perturb(emb))
        results.append(Result(name, down, uns))
        if save_dir:
            np.save(f"{save_dir}/{name}.npy", emb)
        print(f"[{name}] ARI={down['ARI']:.3f} "
              f"kNN={down['kNN_acc']:.3f} | "
              + " ".join(f"{k}={v:.2f}" for k, v in uns.items()))
    return results


def correlate(results: list[Result], target: str = "ARI") -> dict[str, tuple]:
    y = np.array([r.downstream[target] for r in results])
    metric_names = results[0].unsupervised.keys()
    out = {}
    for m in metric_names:
        x = np.array([r.unsupervised[m] for r in results])
        rho, p = spearmanr(x, y)
        out[m] = (float(rho), float(p))
    return out


def load_h5ad(path: str, label_key: str):
    import anndata as ad
    adata = ad.read_h5ad(path)
    counts = adata.X
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    labels = adata.obs[label_key].astype("category").cat.codes.to_numpy()
    return counts.astype("float32"), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", help="path to AnnData .h5ad with raw counts")
    ap.add_argument("--label", default="cell_type", help="obs column with cell types")
    ap.add_argument("--save-embeddings", metavar="DIR",
                     help="if set, save each encoder's embedding matrix as DIR/<name>.npy")
    args = ap.parse_args()

    if args.h5ad:
        counts, labels = load_h5ad(args.h5ad, args.label)
    else:
        # fallback demo so the file runs with no data / no torch
        from block2_singlecell import make_embedding
        print("No --h5ad given: running on synthetic counts as a smoke test.\n")
        counts, labels = make_embedding(n_types=6, per_type=200, dim=200,
                                        separation=2.0, noise=1.0)
        counts = np.abs(counts)  # fake 'counts'

    results = run(counts, labels, save_dir=args.save_embeddings)

    ari_vals = [r.downstream["ARI"] for r in results]
    if len(set(np.round(ari_vals, 6))) == 1:
        print(f"\n[!] All encoders scored ARI={ari_vals[0]:.3f} — no variance to "
              "correlate. Use harder data or weaker encoders to spread the scores.")
    else:
        print(f"\nSpearman(metric, ARI) across {len(results)} encoders:")
        for m, (rho, p) in correlate(results, "ARI").items():
            print(f"  {m:24s} rho={rho:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
