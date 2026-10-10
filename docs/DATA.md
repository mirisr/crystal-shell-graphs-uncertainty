# Data access and local preparation

## Collections

The study uses publicly available, de-identified imaging distributed by The Cancer Imaging Archive:

1. **I-SPY2 breast DCE-MRI**, DOI [10.7937/TCIA.D8Z0-9T85](https://doi.org/10.7937/TCIA.D8Z0-9T85).
2. **ACRIN-6698 breast DWI**, DOI [10.7937/TCIA.KK02-6D95](https://doi.org/10.7937/TCIA.KK02-6D95).

Users are responsible for reviewing and following the current TCIA collection licenses and usage notes before downloading the data.

## Cohort boundary

The reported analysis began with collection metadata, required all four longitudinal visits, successful registration and segmentation, and available pathologic outcome data. The resulting analysis cohort contained 556 I-SPY2 training patients and 202 ACRIN-6698 held-out patients.

ACRIN-6698 is nested within the I-SPY2 trial family. It is held out from model fitting, calibration, and method selection in the reported protocol, but it is not an independent trial.

## Local preparation stages

The public scripts support the following stages:

```text
collection download
  -> series indexing and filtering
  -> PE/SER and mask preprocessing
  -> longitudinal registration
  -> supervoxel extraction
  -> persistent region matching
  -> graph tensor construction
  -> patient-level folds
  -> frozen image embeddings
```

Important entry points include:

- `build_series_index.py`
- `filter_series.py`
- `preprocess_patient.py`
- `match_visits.py`
- `build_patient_graph.py`
- `make_folds.py`
- `experiments/image_to_graph/attach_torchvision_patch_embeddings.py`

## Released and excluded artifacts

The repository releases the patient-level fold assignment using the public TCIA collection identifiers. It also releases the five final model checkpoints and aggregate result tables.

The repository does not redistribute DICOM, NIfTI, masks, registered volumes, clinical workbooks, graph tensors, patient-level predictions, or Monte Carlo samples. Those remain local and are excluded by `.gitignore`.
