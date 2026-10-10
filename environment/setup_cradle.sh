#!/usr/bin/env bash
# One-shot environment setup for the crystal-graph UQ project on Cradle
# (UTRGV). Creates a Miniforge + conda env with PyTorch 2.5 (cu121) and
# torch-geometric 2.6, then runs a GPU smoke test on a gpul40q node.
#
# Usage (from the Cradle login node, after `git clone` of this repo):
#   bash environment/setup_cradle.sh
#
# The script is idempotent: re-running skips completed steps.
#
# Requires outbound HTTPS to github.com, download.pytorch.org, data.pyg.org,
# and pypi.org. No sudo required; everything lives under $HOME.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-crystal_uq}"
PY_VER="3.11"
TORCH_VER="2.5.1"
TORCHVISION_VER="0.20.1"
TORCH_CUDA="cu121"
PYG_VER="2.6.1"
PYG_WHL_INDEX="https://data.pyg.org/whl/torch-2.5.0+${TORCH_CUDA}.html"
GPU_PARTITION="${GPU_PARTITION:-gpul40q}"

log() { printf "\n\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[warn]\033[0m %s\n" "$*" >&2; }
die() { printf "\n\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

command -v wget >/dev/null || die "wget is required but not found"

############################################
# Step 1: Miniforge
############################################
if [[ ! -x "$INSTALL_DIR/bin/conda" ]]; then
    log "Installing Miniforge into $INSTALL_DIR"
    TMP=$(mktemp -d)
    wget -q \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O "$TMP/miniforge.sh"
    bash "$TMP/miniforge.sh" -b -p "$INSTALL_DIR" > "$TMP/install.log" 2>&1 \
        || { cat "$TMP/install.log"; die "Miniforge installer failed"; }
    rm -rf "$TMP"
    "$INSTALL_DIR/bin/conda" init bash > /dev/null
else
    log "Miniforge already present at $INSTALL_DIR"
fi

# shellcheck disable=SC1091
source "$INSTALL_DIR/etc/profile.d/conda.sh"

############################################
# Step 2: Conda env
############################################
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Creating conda env '$ENV_NAME' (Python $PY_VER)"
    conda create -n "$ENV_NAME" "python=$PY_VER" -y
else
    log "Conda env '$ENV_NAME' already exists"
fi

conda activate "$ENV_NAME"
pip install --quiet --upgrade pip setuptools wheel

############################################
# Step 3: PyTorch
############################################
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VER') else 1)" 2>/dev/null; then
    log "Installing PyTorch $TORCH_VER ($TORCH_CUDA)"
    pip install "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
else
    log "PyTorch $TORCH_VER already installed"
fi

############################################
# Step 4: torch-geometric stack
############################################
if ! python -c "import torch_geometric, sys; sys.exit(0 if torch_geometric.__version__.startswith('$PYG_VER') else 1)" 2>/dev/null; then
    log "Installing torch-geometric $PYG_VER"
    pip install "torch-geometric==${PYG_VER}"
else
    log "torch-geometric $PYG_VER already installed"
fi

if ! python -c "import torch_scatter, torch_sparse, torch_cluster" 2>/dev/null; then
    log "Installing torch-scatter / torch-sparse / torch-cluster from PyG wheel index"
    pip install torch-scatter torch-sparse torch-cluster -f "$PYG_WHL_INDEX"
else
    log "PyG companions (scatter/sparse/cluster) already installed"
fi

############################################
# Step 5: Project dependencies
############################################
log "Installing the recorded NCA project dependencies"
pip install --quiet -r "$REPO_ROOT/requirements.txt"
pip install --quiet -e "$REPO_ROOT" --no-deps

############################################
# Step 6: GPU smoke test
############################################
log "Submitting GPU smoke test to partition '$GPU_PARTITION' (may queue briefly)"

SMOKE_OUT=$(mktemp)
trap 'rm -f "$SMOKE_OUT"' EXIT

if srun -p "$GPU_PARTITION" --gres=gpu:1 -t 00:05:00 -J crystal_uq_smoke bash -c "
    source '$INSTALL_DIR/etc/profile.d/conda.sh'
    conda activate '$ENV_NAME'
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
    python - <<'PY'
import torch, torch_geometric
from torch_geometric.nn import CGConv, TransformerConv
print('torch', torch.__version__)
print('pyg', torch_geometric.__version__)
print('cuda.is_available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA not available on compute node'
print('device 0:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
x = torch.randn(64, 32, device='cuda')
ei = torch.randint(0, 64, (2, 256), device='cuda')
ea = torch.randn(256, 4, device='cuda')
_ = CGConv(32, dim=4).cuda()(x, ei, ea)
_ = TransformerConv(32, 32, edge_dim=4).cuda()(x, ei, ea)
print('GPU smoke test PASSED')
PY
" 2>&1 | tee "$SMOKE_OUT"; then
    if grep -q 'GPU smoke test PASSED' "$SMOKE_OUT"; then
        log "Environment verified on $GPU_PARTITION"
    else
        warn "srun completed but smoke test output did not confirm success"
    fi
else
    die "srun failed — check cluster availability and try again"
fi

############################################
# Done
############################################
cat <<EOF

========================================================================
 Setup complete.

 To use the environment in a new shell:
   source $INSTALL_DIR/etc/profile.d/conda.sh
   conda activate $ENV_NAME

 In a SLURM job script, include:
   source $INSTALL_DIR/etc/profile.d/conda.sh
   conda activate $ENV_NAME
   module load cuda/12.3    # optional; torch wheel bundles its own cudart

 Versions pinned:
   Python       $PY_VER
   PyTorch      $TORCH_VER ($TORCH_CUDA)
   PyG          $PYG_VER
   Project deps installed via pip

 Partition default for smoke test: $GPU_PARTITION
========================================================================
EOF
