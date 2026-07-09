# smiles-embedding-quality

Unsupervised embedding quality estimation for single-cell data (SMILES 2026, supervisors Egor Shvetsov / Egor Surkov).

## Install

```bash
pip install -e .
```

## Run tests

```bash
pytest -v
```

## Run the real single-cell pipeline

```bash
python block2_real.py --h5ad data/pbmc.h5ad --label cell_type
```

Without `--h5ad`, it falls back to a synthetic smoke test (`block2_singlecell.py`).
