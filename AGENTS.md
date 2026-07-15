# AGENTS.md — smiles-embedding-quality

Context and working conventions for any agent or collaborator on this repo.

## Mission

SMILES 2026 thesis: can embedding quality for scRNA-seq encoders be estimated
label-free, from geometric properties alone? Core finding (validated,
pre-defense): no single label-free metric (effective rank, anisotropy,
intrinsic dim, ...) reliably predicts downstream ARI/NMI across readouts.

Current phase: FM1-FM4 failure-mode diagnostics — characterize *where and why*
existing metrics fail, using the real scVI/PCA/ICA/NMF embeddings in
`results/` and a synthetic ground-truth testbed (separate repo). Only after
that: derive a new metric motivated by the diagnosis, still label-free.

## Source of truth

- Numbers, correlations, CIs: the checked-in tables in `results/`. Never
  invent a number — if it isn't in a CSV, it's `TBD`.
- Sample provenance (which n, which table, why): `FAILURE_MODES.md`. Read it
  before comparing any two numbers across figures — samples differ in size
  (n=8/11/14/17) and which controls are included.

## Workspace layout

- `embq/` — the package: `metrics.py` (label-free geometric metrics),
  `failure_modes.py` (FM1-4 diagnostics).
- `scripts/` — runnable entry points (pipelines, analysis drivers).
- `tests/` — pytest suite, one `test_<module>.py` per `embq/<module>.py`,
  shared fixtures in `tests/conftest.py`.
- `results/` — checked-in output tables/figures. Committed — this is the data
  the whole analysis is traceable back to.
- `reports/` — report drafts, deck corrections, the PDF build script. Local
  only, gitignored — never commit.

## Tests ship with the code, not after it

- Every new function in `embq/` gets its test in the *same* commit as the
  function — not a follow-up "add tests" commit. If you can't state the test
  case, the function's contract isn't clear yet.
- Shared fixtures go in `tests/conftest.py`, not duplicated per test file.
- Prefer small synthetic fixtures with a hand-computable answer (a
  well-separated 2-cluster embedding, a perfectly-linear PCA sweep) over
  sampling real `results/` data into a test — real data drifts as the
  pipeline changes; synthetic fixtures document intent and stay stable.
- Run `pytest -v` before every commit touching `embq/` or `scripts/`.

## Git

- Conventional commits: `type: summary` (`feat`, `fix`, `test`, `chore`,
  `docs`), body explains *why*, not just what changed.
- Atomic commits — don't bundle an unrelated reorg with a feature.
- No AI/assistant attribution in commit messages or trailers.

## Don't

- Don't fabricate experimental numbers — pull from `results/` or mark `TBD`.
- Don't commit `reports/` — drafts and the PDF build stay local.
- Don't add CI or a LICENSE file — out of scope for this repo.
