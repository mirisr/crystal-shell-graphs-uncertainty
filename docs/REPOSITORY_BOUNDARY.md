# Repository boundary

This repository contains only the final code and artifacts associated with the NCA manuscript. Its Git history and remote are independent from the earlier `breast-dce-mri-graph-rollouts` release.

## Included in this release

- source for the final data preparation, frozen ResNet-18 embedding, crystal-shell graph forecaster, deterministic evaluation, residual Monte Carlo, conformal calibration, and pCR probe stages;
- the final model configuration and five fold checkpoints;
- SHA-256 checkpoint hashes;
- the public patient identifier to fold mapping;
- frozen aggregate results and method-selection records;
- exact environment pins, tests, and reproduction commands.

## Kept outside Git

- raw DICOM and downloaded clinical workbooks;
- processed image volumes, registration fields, tumor masks, and supervoxel graph tensors;
- patient-level predictions, interval outputs, and Monte Carlo draws;
- manuscript source, compiled PDFs, cover letters, and submission packages;
- patient-level figures and videos;
- experiment logs, exploratory model families, and historical archives;
- credentials and workstation or cluster provenance inventories.

The excluded materials remain available in the local research workspace or on Cradle where permitted. Their exclusion prevents accidental redistribution of large derived datasets and keeps the public repository aligned with the final manuscript.
