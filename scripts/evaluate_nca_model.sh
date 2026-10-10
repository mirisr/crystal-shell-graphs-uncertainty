#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${CRYSTAL_UQ_PYTHON:-python}"
ASSET_DIR="${ASSET_DIR:-reports/acrin_holdout_paper_variants}"

"$PYTHON" experiments/stage1_forecaster/make_acrin_holdout_assets.py --out-dir "$ASSET_DIR"
"$PYTHON" experiments/stage1_forecaster/run_simulation_eval.py \
  --runs-dir "${RUNS_DIR:-models/checkpoints}" \
  --graphs-root "${GRAPH_ROOT:-datasets/ispy2/graphs_consistent_torchvision_resnet18_embed}" \
  --folds "${FOLDS:-$ASSET_DIR/acrin_test_folds.parquet}" \
  --cohort "${COHORT:-datasets/ispy2/cohort.parquet}" \
  --checkpoint-name best.pt \
  --out-dir "${OUT_DIR:-reports/nca_reproduction/deterministic}"
