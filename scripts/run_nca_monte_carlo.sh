#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${CRYSTAL_UQ_PYTHON:-python}"
ASSET_DIR="${ASSET_DIR:-reports/acrin_holdout_paper_variants}"
SPLIT="${SPLIT:-acrin_holdout}"

"$PYTHON" experiments/stage1_forecaster/make_acrin_holdout_assets.py --out-dir "$ASSET_DIR"

if [[ "$SPLIT" == "ispy2_oof" ]]; then
  PATIENT_LIST="$ASSET_DIR/ispy2_train.txt"
  EXTRA_ARGS=()
elif [[ "$SPLIT" == "acrin_holdout" ]]; then
  PATIENT_LIST="$ASSET_DIR/ispy2_acrin_eval.txt"
  EXTRA_ARGS=(
    --calibration-patient-list "$ASSET_DIR/ispy2_train.txt"
    --eval-patient-list "$ASSET_DIR/acrin_test.txt"
  )
else
  echo "SPLIT must be ispy2_oof or acrin_holdout." >&2
  exit 2
fi

"$PYTHON" experiments/consistent_rollout/run_conditional_mc.py \
  --runs-dir "${RUNS_DIR:-models/checkpoints}" \
  --checkpoint-name best.pt \
  --graphs-root "${GRAPH_ROOT:-datasets/ispy2/graphs_consistent_torchvision_resnet18_embed}" \
  --folds "${FOLDS:-$ASSET_DIR/ispy2_acrin_folds.parquet}" \
  --cohort "${COHORT:-datasets/ispy2/cohort.parquet}" \
  --patient-list "$PATIENT_LIST" \
  "${EXTRA_ARGS[@]}" \
  --out-dir "${OUT_DIR:-reports/nca_reproduction/features/$SPLIT}" \
  --start-visits 0 1 2 --n-mc "${N_MC:-256}" --metric-draws 0 \
  --eps-pcr 0.1 --interval 0.90 --seed 42 --device "${DEVICE:-auto}" \
  --residual-stratify-by none --residual-sampler-mode bucket_baseline
