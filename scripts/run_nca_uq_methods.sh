#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${CRYSTAL_UQ_PYTHON:-python}"
FEATURE_ROOT="${FEATURE_ROOT:-reports/nca_reproduction/features}"

"$PYTHON" experiments/consistent_rollout/train_paper_model_uq_methods.py \
  --train-per-patient "$FEATURE_ROOT/ispy2_oof/conditional_mc_per_patient.parquet" \
  --test-per-patient "$FEATURE_ROOT/acrin_holdout/conditional_mc_per_patient.parquet" \
  --out-dir "${OUT_DIR:-reports/nca_reproduction/uq_methods}" \
  --visit-pair T0_to_T3 --feature-modes paper_scalar paper_full \
  --interval 0.90 --seed 42 --student-epochs 500 --student-patience 50
