# Local data workspace

This directory is a local workspace for I-SPY2 and ACRIN-6698 source and derived artifacts. Git tracks only this document and the preprocessing scripts under `scripts/`.

Expected local structure:

```text
datasets/ispy2/
├── raw/                                  source DICOM or downloaded collection files
├── derived/                              PE, SER, support masks, and preprocessing outputs
├── registered/                           visit-to-baseline registrations
├── graphs_consistent/                    persistent longitudinal graph tensors
├── graphs_consistent_torchvision_resnet18_embed/
├── cohort.parquet                        local patient-level cohort table
└── folds.parquet                         local working copy of patient-level folds
```

The source collections must be obtained from TCIA:

- I-SPY2 DCE-MRI: https://doi.org/10.7937/TCIA.D8Z0-9T85
- ACRIN-6698 DWI: https://doi.org/10.7937/TCIA.KK02-6D95

Do not commit raw data, derived volumes, clinical metadata, patient-level predictions, or graph tensors. The frozen public fold assignment is released separately at `splits/ispy2_folds.parquet`.
