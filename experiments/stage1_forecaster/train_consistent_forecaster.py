#!/usr/bin/env python3
"""Train the *consistent-graph* forecaster on the 30-patient prototype set.

Unlike the Stage-1 forecaster (which relies on NN matching for supervision),
this model uses **persistent node identities** from deformable registration:
  - node i at visit k corresponds *exactly* to node i at visit k+1
  - temporal edges are deterministic (src=i_v, dst=i_{v+1})
  - targets are exact: Δpos = pos[v+1][i] - pos[v][i], Δfeat = x[v+1][i] - x[v][i]
  - no alive head (disappearing supervoxels are handled via a loss mask)

Architecture: same LSGC backbone (S1.8: bio edges + edge gating), reused from
the best Stage-1 checkpoint as a warm start, with new heads fine-tuned on the
consistent-graph objectives.

Outputs
-------
    runs/consistent_forecaster/best.pt
    runs/consistent_forecaster/last.pt
    runs/consistent_forecaster/train_log.json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lsgc.forecaster import LSGCForecaster

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISITS = ("T0", "T1", "T2", "T3")


# ---------------------------------------------------------------------------
# Graph loading + edge building
# ---------------------------------------------------------------------------

def spatial_knn_edges(pos: Tensor, k: int = 8) -> Tensor:
    """Symmetric k-NN edges among nodes in `pos` (intra-visit spatial)."""
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    k_eff = min(k, pos.shape[0] - 1)
    _, idx = torch.topk(d, k=k_eff, largest=False, dim=1)
    src = torch.arange(pos.shape[0]).unsqueeze(1).expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    # Symmetric
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return ei.unique(dim=1)


def build_edges(g: dict, k_spatial: int = 8) -> Tensor:
    """Build intra-visit spatial kNN + deterministic temporal identity edges.

    Temporal edges: for each node id i and each consecutive visit pair (v, v+1),
    add edge from i_v → i_{v+1}.  This is exact correspondence — no NN noise.
    """
    offsets = g["visit_offsets"].tolist()
    pos = g["pos"]
    T = len(offsets) - 1

    all_src, all_dst = [], []

    for v in range(T):
        sl = slice(int(offsets[v]), int(offsets[v + 1]))
        pos_v = pos[sl]
        n_v = pos_v.shape[0]
        if n_v > 1:
            ei_v = spatial_knn_edges(pos_v, k=k_spatial)
            all_src.append(ei_v[0] + offsets[v])
            all_dst.append(ei_v[1] + offsets[v])

        # Deterministic temporal edges: node i at visit v → node i at visit v+1
        if v < T - 1:
            n_nodes = int(offsets[v + 1]) - int(offsets[v])
            assert n_nodes == int(offsets[v + 2]) - int(offsets[v + 1]), \
                "Consistent graph must have same node count at every visit"
            src_t = torch.arange(n_nodes) + offsets[v]
            dst_t = torch.arange(n_nodes) + offsets[v + 1]
            all_src.append(src_t); all_dst.append(dst_t)
            # Backward temporal edge too (allows signal to flow forward in time)
            all_src.append(dst_t); all_dst.append(src_t)

    edge_index = torch.stack([
        torch.cat(all_src), torch.cat(all_dst)
    ], dim=0).long()
    return edge_index


@dataclass
class ConsistentSample:
    patient_id: str
    x: Tensor          # (4N, C) features, all 4 visits stacked
    pos: Tensor        # (4N, 3)
    t: Tensor          # (4N, 1) visit index as float
    edge_index: Tensor # (2, E)
    visit_offsets: Tensor  # (5,) prefix sums [0, N, 2N, 3N, 4N]
    alive: Tensor      # (4N,) 1=supervoxel present, 0=dropped by registration
    n_sv: int          # supervoxels per visit (N)


def load_sample(path: Path) -> Optional[ConsistentSample]:
    try:
        g = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    try:
        edge_index = build_edges(g)
    except Exception:
        return None
    n_sv = int(g["n_supervoxels"])
    return ConsistentSample(
        patient_id=g["patient_id"],
        x=g["x"].float(),
        pos=g["pos"].float(),
        t=g["t"].float(),
        edge_index=edge_index,
        visit_offsets=g["visit_offsets"].long(),
        alive=g["alive"].float(),
        n_sv=n_sv,
    )


# ---------------------------------------------------------------------------
# Z-score normalisation (fit on train set)
# ---------------------------------------------------------------------------

def compute_stats(samples: List[ConsistentSample]):
    xs = torch.cat([s.x for s in samples], dim=0)
    return xs.mean(0), xs.std(0).clamp_min(1e-6)


# ---------------------------------------------------------------------------
# Per-transition loss
# ---------------------------------------------------------------------------

def transition_loss(
    out: dict,
    pos_src: Tensor, pos_dst: Tensor,
    x_src: Tensor, x_dst: Tensor,
    alive_dst: Tensor,
    pos_weight: float = 1.0,
    feat_weight: float = 0.5,
) -> tuple[Tensor, dict]:
    """Compute Δpos MSE + Δfeat MSE on supervoxels alive at destination visit."""
    mask = alive_dst.bool()
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True), {}

    # Positions are already centroid-relative (centroid subtracted in graph builder).
    # Ground-truth displacement = how much each supervoxel moved from src to dst.
    delta_pos_gt = (pos_dst - pos_src)[mask]

    delta_feat_gt = (x_dst - x_src)[mask]

    delta_pos_pred  = out["delta_pos"][mask]
    delta_feat_pred = out["delta_feat"][mask]

    pos_loss  = F.mse_loss(delta_pos_pred, delta_pos_gt)
    feat_loss = F.mse_loss(delta_feat_pred, delta_feat_gt)
    total = pos_weight * pos_loss + feat_weight * feat_loss

    with torch.no_grad():
        pos_mae = (delta_pos_pred - delta_pos_gt).norm(dim=-1).mean().item()

    return total, {"pos_loss": pos_loss.item(), "feat_loss": feat_loss.item(),
                   "pos_mae_mm": pos_mae, "n_alive": mask.sum().item()}


# ---------------------------------------------------------------------------
# Single forward pass over all transitions
# ---------------------------------------------------------------------------

def forward_patient(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: Tensor, std: Tensor,
    device: str,
) -> tuple[Tensor, dict]:
    x = ((s.x.to(device) - mean) / std)
    pos = s.pos.to(device)
    t   = s.t.to(device)
    ei  = s.edge_index.to(device)
    alive = s.alive.to(device)
    off = s.visit_offsets

    out = model(x, pos, t, ei)

    total_loss = torch.tensor(0.0, device=device)
    info = {"transitions": []}
    T = len(off) - 1
    for v in range(T - 1):
        sl_src = slice(int(off[v]),   int(off[v + 1]))
        sl_dst = slice(int(off[v+1]), int(off[v + 2]))

        loss_v, info_v = transition_loss(
            {k: val[sl_src] for k, val in out.items() if isinstance(val, Tensor)},
            pos[sl_src], pos[sl_dst],
            x[sl_src],   x[sl_dst],
            alive[sl_dst],
        )
        total_loss = total_loss + loss_v
        info["transitions"].append(info_v)

    return total_loss / max(T - 1, 1), info


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class Config:
    graphs_root: str = "datasets/ispy2/graphs_consistent"
    patient_list: str = "reports/all_patients_4visit.txt"
    out_dir: str = "runs/consistent_forecaster"
    pretrained_ckpt: str = (
        "experiments/stage1_forecaster/runs/bio_gated_5fold/fold0/best.pt"
    )
    hidden: int = 64
    num_layers: int = 2
    k_spatial: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 50
    val_frac: float = 0.2       # only used when fold == -1
    seed: int = 42
    freeze_backbone_epochs: int = 30   # fine-tune heads first, then unfreeze
    fold: int = -1              # -1 = random split; 0-4 = stratified 5-fold CV
    folds_parquet: str = "datasets/ispy2/folds.parquet"


def train(cfg: Config):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load graphs
    pids = Path(cfg.patient_list).read_text().splitlines()
    pids = [p.strip() for p in pids if p.strip()]
    samples_all = []
    for pid in pids:
        s = load_sample(Path(cfg.graphs_root) / f"{pid}.pt")
        if s: samples_all.append(s)
    print(f"Loaded {len(samples_all)}/{len(pids)} graphs")

    # Train / val split: either stratified 5-fold or random fraction
    if cfg.fold >= 0:
        import pandas as pd
        folds_df = pd.read_parquet(cfg.folds_parquet)
        fold_map = dict(zip(folds_df["patient_id"], folds_df["fold"]))
        val_samples   = [s for s in samples_all if fold_map.get(s.patient_id) == cfg.fold]
        train_samples = [s for s in samples_all if fold_map.get(s.patient_id) != cfg.fold]
        print(f"Fold {cfg.fold}/5  Train: {len(train_samples)}  Val: {len(val_samples)}")
    else:
        random.shuffle(samples_all)
        n_val = max(1, int(len(samples_all) * cfg.val_frac))
        val_samples   = samples_all[:n_val]
        train_samples = samples_all[n_val:]
        print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    mean, std = compute_stats(train_samples)
    mean, std = mean.to(DEVICE), std.to(DEVICE)

    # Build model — reuse feature dims from consistent graph
    in_ch  = samples_all[0].x.shape[1]
    feat_out = in_ch

    # Try to warm-start from S1.8 checkpoint (only backbone weights)
    ckpt_path = Path(cfg.pretrained_ckpt)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        s1_in_ch = int(ck["state_dict"]["embed.weight"].shape[1])
        print(f"Pretrained checkpoint: in_ch={s1_in_ch} (consistent in_ch={in_ch})")
    else:
        ck = None
        print("No pretrained checkpoint found — training from scratch")

    model = LSGCForecaster(
        in_channels=in_ch,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        feat_out_dim=feat_out,
        use_delta_t=True,
        use_edge_gating=True,
        edge_attr_dim=0,   # consistent graph has no bio edge attrs (yet)
    ).to(DEVICE)

    # Warm-start: load backbone weights if dims match
    if ck is not None and s1_in_ch == in_ch:
        sd = model.state_dict()
        pretrained_sd = ck["state_dict"]
        loaded, skipped = [], []
        for k, v in pretrained_sd.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v; loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(sd)
        print(f"Warm-start: loaded {len(loaded)} tensors, skipped {len(skipped)}")
    else:
        print("Skipping warm-start (dim mismatch or no ckpt)")

    # Phase 1: heads only
    def _set_backbone_grad(requires_grad: bool):
        for name, p in model.named_parameters():
            if name.startswith("embed") or name.startswith("convs"):
                p.requires_grad_(requires_grad)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = float("inf"); patience_cnt = 0
    log = []

    for epoch in range(1, cfg.epochs + 1):
        # Unfreeze backbone after warmup
        if epoch == cfg.freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr * 0.3, weight_decay=cfg.weight_decay
            )
            print(f"  [Epoch {epoch}] Unfreezing backbone, lr → {cfg.lr * 0.3:.1e}")

        model.train()
        random.shuffle(train_samples)
        train_losses, train_mae = [], []
        for s in train_samples:
            optimizer.zero_grad()
            loss, info = forward_patient(model, s, mean, std, DEVICE)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(loss.item())
            for tx in info["transitions"]:
                if "pos_mae_mm" in tx:
                    train_mae.append(tx["pos_mae_mm"])

        model.eval()
        val_losses, val_mae = [], []
        with torch.no_grad():
            for s in val_samples:
                loss, info = forward_patient(model, s, mean, std, DEVICE)
                val_losses.append(loss.item())
                for tx in info["transitions"]:
                    if "pos_mae_mm" in tx:
                        val_mae.append(tx["pos_mae_mm"])

        tr_l = np.mean(train_losses) if train_losses else float("nan")
        vl_l = np.mean(val_losses)   if val_losses   else float("nan")
        tr_m = np.mean(train_mae)    if train_mae     else float("nan")
        vl_m = np.mean(val_mae)      if val_mae       else float("nan")

        log.append(dict(epoch=epoch,
                        train_loss=tr_l, val_loss=vl_l,
                        train_mae_mm=tr_m, val_mae_mm=vl_m))

        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  "
                  f"train_loss={tr_l:.4f}  val_loss={vl_l:.4f}  "
                  f"train_mae={tr_m:.2f}mm  val_mae={vl_m:.2f}mm")

        if vl_l < best_val:
            best_val = vl_l; patience_cnt = 0
            torch.save(dict(
                state_dict=model.state_dict(),
                mean=mean.cpu(), std=std.cpu(),
                config=asdict(cfg),
                feature_names=samples_all[0].x.shape,
                epoch=epoch,
                val_loss=vl_l,
                val_mae_mm=vl_m,
            ), out_dir / "best.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print(f"Early stop at epoch {epoch}")
                break

    torch.save(dict(
        state_dict=model.state_dict(),
        mean=mean.cpu(), std=std.cpu(),
        config=asdict(cfg),
    ), out_dir / "last.pt")

    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nBest val_loss={best_val:.4f}  "
          f"val_mae={log[next(i for i,e in enumerate(log) if e['val_loss']==best_val)]['val_mae_mm']:.2f}mm")
    print(f"Outputs: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-root",  default="datasets/ispy2/graphs_consistent")
    ap.add_argument("--patient-list", default="reports/all_patients_4visit.txt")
    ap.add_argument("--out-dir",      default="runs/consistent_forecaster")
    ap.add_argument("--pretrained",   default="experiments/stage1_forecaster/runs/bio_gated_5fold/fold0/best.pt")
    ap.add_argument("--epochs",       type=int, default=300)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--hidden",       type=int, default=64)
    ap.add_argument("--num-layers",   type=int, default=2)
    ap.add_argument("--patience",     type=int, default=50)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=30)
    ap.add_argument("--fold",         type=int, default=-1,
                    help="Val fold index (0-4) for stratified 5-fold CV. "
                         "-1 uses random val_frac split.")
    ap.add_argument("--folds-parquet", default="datasets/ispy2/folds.parquet")
    args = ap.parse_args()

    cfg = Config(
        graphs_root=args.graphs_root,
        patient_list=args.patient_list,
        out_dir=args.out_dir,
        pretrained_ckpt=args.pretrained,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        num_layers=args.num_layers,
        patience=args.patience,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        fold=args.fold,
        folds_parquet=args.folds_parquet,
    )
    train(cfg)


if __name__ == "__main__":
    main()
