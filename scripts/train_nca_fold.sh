#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${CRYSTAL_UQ_PYTHON:-python}"
FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-}}"
[[ "$FOLD" =~ ^[0-4]$ ]] || { echo "Set FOLD to an integer from 0 through 4." >&2; exit 2; }

ASSET_DIR="${ASSET_DIR:-reports/acrin_holdout_paper_variants}"
GRAPH_ROOT="${GRAPH_ROOT:-datasets/ispy2/graphs_consistent_torchvision_resnet18_embed_byfold/fold${FOLD}}"
OUT_DIR="${OUT_DIR:-models/checkpoints/fold${FOLD}}"

"$PYTHON" experiments/stage1_forecaster/make_acrin_holdout_assets.py --out-dir "$ASSET_DIR"
mkdir -p "$OUT_DIR"

"$PYTHON" experiments/stage1_forecaster/train_consistent_forecaster_v2.py \
  --graphs-root "$GRAPH_ROOT" \
  --patient-list "$ASSET_DIR/ispy2_train.txt" \
  --folds-parquet "${FOLDS_PARQUET:-splits/ispy2_folds.parquet}" \
  --cohort-parquet "${COHORT_PARQUET:-datasets/ispy2/cohort.parquet}" \
  --fold "$FOLD" \
  --out-dir "$OUT_DIR" \
  --pretrained "$ASSET_DIR/no_pretrain.pt" \
  --epochs 170 --patience 45 --freeze-backbone-epochs 0 --lr 5e-5 \
  --hidden 64 --num-layers 2 --scheduled-sampling --eps-max 0.5 \
  --eps-warmup-epochs 8 --eps-anneal-epochs 60 --random-start \
  --core-in-channels 6 --ssl-adapter-dim 16 --ssl-adapter-dropout 0.10 \
  --predict-core-only --lambda-horizon 0.75 --lambda-ftv 0.35 \
  --lambda-alive-mass 0.01 --lambda-alive-bce 0.01 \
  --bio-warmup-epochs 0 --bio-anneal-epochs 12 \
  --selection-primary rollout --selection-ftv-weight 1.25 \
  --selection-alive-weight 0.00025 --graph-ftv-affine \
  --lambda-terminal-ftv 0.125 --terminal-ftv-loss-space ml \
  --edge-attr-dropout 0.10 --lambda-edge-response 0.15 \
  --lambda-edge-alive 0.05 --lambda-response-diffusion 0.02 \
  --k-spatial 8 --edge-mode crystal_shell_union \
  --edge-attr-mode ssl_radial_bio --conv-type relational \
  --reaction-diffusion-head --lambda-reaction-diffusion-residual 0.02 \
  --reaction-diffusion-treatment-proxy 1.0
