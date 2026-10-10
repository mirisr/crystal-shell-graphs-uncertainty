# Public release audit

This repository is a separate public release for the final NCA crystal-shell graph and patient-conditioned uncertainty manuscript.

## Included

- the final graph construction, image embedding, forecaster, evaluation, Monte Carlo, conformal, and pCR probe code;
- the exact selected-model configuration;
- five trained fold checkpoints with SHA-256 hashes;
- the patient-level fold assignment for public collection identifiers;
- the recorded Cradle software versions and idempotent setup script;
- data-free tests and a checkpoint-backed synthetic rollout;
- frozen aggregate manuscript result and selection artifacts.

## Excluded

- raw or derived patient imaging;
- tumor masks, registrations, and graph tensors;
- patient-level predictions and Monte Carlo samples;
- generated manuscript and submission packages;
- experiment logs and exploratory model families;
- local provenance inventories and credentials.

## Separation from the earlier release

The intended remote is `mirisr/crystal-shell-graphs-uncertainty`. It does not reuse the Git history, remote, weights, or release artifacts of `mirisr/breast-dce-mri-graph-rollouts`.

## Required checks

```bash
python scripts/audit_public_release.py
python scripts/smoke_test.py
python -m pytest -q \
  lsgc/tests/test_edge_modes.py \
  lsgc/tests/test_forecaster.py \
  lsgc/tests/test_image_to_graph.py \
  datasets/ispy2/scripts/tests \
  experiments/consistent_rollout/tests \
  experiments/consistent_rollout/test_paper_model_uq_ispy2_ranking.py
```
