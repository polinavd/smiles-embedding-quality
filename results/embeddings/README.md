# Embedding matrices

Saved encoder outputs from `block2_real.py --h5ad data/pbmc3k.h5ad --label cell_type
--save-embeddings results/embeddings`, run on PBMC3k (2,638 cells, 8 `louvain` cell-type
labels, matching `first_pass_report.md`). Each `.npy` is `n_cells x n_comps`.

- **random.npy** — (2638, 50) random-projection baseline (negative control, ARI 0.319).
- **pca_10.npy** — (2638, 10) PCA embedding (ARI 0.750).
- **pca_50.npy** — (2638, 50) PCA embedding (ARI 0.749).

scVI encoders are not included here — the registry entry is commented out in
`block2_real.py` pending a GPU run.
