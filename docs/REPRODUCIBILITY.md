# Final NCA reproducibility protocol

This protocol distinguishes artifact verification from full numerical reproduction. Artifact verification uses the released checkpoints and synthetic data. Full reproduction requires the TCIA collections, the accompanying clinical tables, registration and graph preprocessing, GPU training, and patient-level intermediate outputs.

## 1. Verify the source release

Create the Python 3.11 environment described in the main README, then run:

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

The smoke test verifies the SHA-256 digest of all five final checkpoints, reconstructs each model from its embedded configuration, and runs a synthetic T0-to-T3 rollout through the selected architecture.

## 2. Recorded environment

The final Cradle run used the following core environment:

- Python 3.11.15;
- PyTorch 2.5.1 with CUDA 12.1;
- torchvision 0.20.1 with CUDA 12.1;
- PyTorch Geometric 2.6.1;
- NumPy 1.26.4;
- pandas 2.2.3;
- SciPy 1.15.3;
- scikit-learn 1.8.0;
- scikit-image 0.26.0;
- pyarrow 24.0.0;
- pydicom 3.0.2.

`environment/setup_cradle.sh` creates an isolated environment and installs the pinned project requirements. The scripts use paths relative to the repository and accept local overrides through environment variables.

## 3. Source data

Obtain the source collections from TCIA and review their current terms:

- I-SPY2 DCE-MRI: https://doi.org/10.7937/TCIA.D8Z0-9T85
- ACRIN-6698: https://doi.org/10.7937/TCIA.KK02-6D95

The source workbooks and imaging are not redistributed here. The release includes `splits/ispy2_folds.parquet`, which records the patient-level fold assignment used by the pipeline.

## 4. Data preparation and leakage controls

The primary preprocessing entry points are under `datasets/ispy2/scripts/`:

```bash
python datasets/ispy2/scripts/build_series_index.py --help
python datasets/ispy2/scripts/filter_series.py --help
python datasets/ispy2/scripts/preprocess_patient.py --help
python datasets/ispy2/scripts/match_visits.py --help
python datasets/ispy2/scripts/build_patient_graph.py --help
python datasets/ispy2/scripts/build_cohort.py --help
python datasets/ispy2/scripts/make_folds.py --help
```

All partitions are patient-level. Visits from one patient must remain in one fold. The 202 ACRIN-6698 held-out patients must remain outside model fitting, residual calibration, uncertainty-method ranking, threshold selection, and pCR model selection.

The final model used persistent supervoxels created after registration to the pretreatment examination. The final graph tensors contained six structured channels and a 512-dimensional frozen image embedding.

## 5. Frozen image embeddings

Attach the ResNet-18 embeddings after the consistent graphs have been built:

```bash
python experiments/image_to_graph/attach_torchvision_patch_embeddings.py \
  --graph-root datasets/ispy2/graphs_consistent \
  --derived-root datasets/ispy2/derived \
  --out-root datasets/ispy2/graphs_consistent_torchvision_resnet18_embed
```

The encoder uses torchvision ResNet-18 with `IMAGENET1K_V1` weights. The final model does not update the ResNet-18 weights.

## 6. Train the five final folds

The exact training arguments are frozen in `scripts/train_nca_fold.sh`. Run one job per fold:

```bash
for fold in 0 1 2 3 4; do
  FOLD="$fold" scripts/train_nca_fold.sh
done
```

On SLURM, submit the same script as an array and set `SLURM_ARRAY_TASK_ID` from 0 through 4. The final output contract is:

```text
models/checkpoints/fold0/best.pt
models/checkpoints/fold1/best.pt
models/checkpoints/fold2/best.pt
models/checkpoints/fold3/best.pt
models/checkpoints/fold4/best.pt
```

A retraining run should record new hashes rather than overwrite the released hash manifest without review.

## 7. Deterministic held-out evaluation

Run:

```bash
scripts/evaluate_nca_model.sh
```

The primary pretreatment-to-presurgery checks include deterministic FTV MAE, sliced Wasserstein distance, Chamfer distance, and active-subregion count error. Compare aggregate output against `reports/reproducibility/nca/deterministic_simulation_summary.json` and `reports/reproducibility/paper_results.csv`.

## 8. Residual Monte Carlo features

Generate the I-SPY2 out-of-fold and ACRIN-6698 held-out features separately:

```bash
SPLIT=ispy2_oof scripts/run_nca_monte_carlo.sh
SPLIT=acrin_holdout scripts/run_nca_monte_carlo.sh
```

The reported protocol uses 256 samples, a 0.90 interval target, and seed 42. I-SPY2 residuals supply calibration. ACRIN-6698 rows must never enter the calibration pool.

## 9. Patient-conditioned uncertainty selection

Run:

```bash
scripts/run_nca_uq_methods.sh
python experiments/consistent_rollout/test_paper_model_uq_ispy2_ranking.py
```

Candidate methods are ranked using I-SPY2 out-of-fold interval score while requiring coverage within 0.02 of the 0.90 target. The winner is frozen before the held-out cohort is scored. The released winner is the full-feature ExtraTrees locally scaled residual conformal method.

The expected held-out result is recorded in `paper_model_uq_frozen_acrin.csv`. The final table is in `paper_model_uq_t0t3_summary.csv`.

## 10. Secondary pCR analysis

The final pCR analysis uses `train_pcr_sklearn_probe.py` and the helper routines in `train_pcr_clinical_stack.py`. The internally selected L1 logistic classifier combines clinical covariates, graph-temporal summaries, and graph-derived model scores. It is selected on I-SPY2 out-of-fold predictions and scored once on ACRIN-6698.

The pCR analysis includes observed T1 and T2 information. It must not be described as a pretreatment-only classifier. The final aggregate rows and fold-level internal results are included under `reports/reproducibility/nca/`.

## 11. Numerical acceptance checks

A full rerun should verify all of the following:

- 556 I-SPY2 patients in the fitting, out-of-fold selection, and calibration cohort;
- 202 ACRIN-6698 patients in the untouched held-out cohort;
- five patient-level folds with no cross-fold patient leakage;
- model tag `resnet_crystal_edgegate_rd_k8`;
- 256 Monte Carlo samples for the reported uncertainty analysis;
- no ACRIN-6698 rows in uncertainty-method ranking;
- the frozen locally scaled conformal method selected on I-SPY2 out-of-fold results;
- consistent endpoint definitions and visit labels;
- the pCR feature timing and selection protocol described above.

Small differences can arise after GPU retraining. A difference in cohort size, split membership, selected method, evaluation definition, or leakage control is a protocol difference and should not be classified as ordinary numerical variation.

## 12. Reproduction language

Use the following terms precisely:

- **Artifacts verified:** the hashes, checkpoint loading, synthetic rollout, audit, and tests pass.
- **Pipeline rerun:** all data preparation, training, and evaluation stages execute, but numerical agreement has not been assessed.
- **Numerically reproduced:** the rerun follows the frozen protocol and matches the stated aggregate tolerances.
- **Independently validated:** a distinct external cohort was evaluated. The reported ACRIN-6698 analysis does not meet this definition because it is nested within the I-SPY2 trial family.
