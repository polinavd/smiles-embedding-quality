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
python scripts/block2_real.py --h5ad data/pbmc.h5ad --label cell_type
```

Without `--h5ad`, it falls back to a synthetic smoke test (`scripts/block2_singlecell.py`).

## Run the failure-mode analysis

```bash
python scripts/failure_mode_analysis.py
```

Reproduces `results/fm1..fm4_*.{csv,png}` from the checked-in tables. See
`FAILURE_MODES.md` for methodology and sample provenance.

## Repo layout

- `embq/` — metrics and failure-mode diagnostics (the package)
- `scripts/` — runnable entry points (pipelines, analysis drivers)
- `tests/` — pytest suite
- `results/` — checked-in output tables/figures referenced by `FAILURE_MODES.md`
- `reports/` — local-only report drafts and PDF build (not tracked in git)
