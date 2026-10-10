#!/usr/bin/env python3
"""Rollout-aware training for the consistent forecaster (v2).

Adds three training upgrades on top of train_consistent_forecaster.py:
1) scheduled sampling (self-feeding),
2) multi-horizon cloud loss,
3) random-start conditioning (start from T0/T1/T2).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lsgc.forecaster import LSGCForecaster
from experiments.stage1_forecaster.train_consistent_forecaster import (
    ConsistentSample,
    compute_stats,
    load_sample,
)
from experiments.stage1_forecaster.edge_modes import (
    EDGE_ATTR_MODE_CHOICES,
    EDGE_MODE_CHOICES,
    NUM_RELATIONS,
    build_history_graph,
    build_history_graph_with_types,
    edge_attr_dim,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISITS = ("T0", "T1", "T2", "T3")
RD_ACTION_CONTEXT_MODES = ("phase", "components", "components_subtype")
RD_ACTION_COMPONENTS = (
    "paclitaxel",
    "pembrolizumab",
    "carboplatin",
    "abt888",
    "mk2206",
    "neratinib",
    "ganitumab",
    "ganetespib",
    "amg386",
    "trastuzumab",
    "pertuzumab",
    "tdm1",
)
RD_ACTION_FAMILIES = (
    "paclitaxel_only",
    "paclitaxel_targeted",
    "paclitaxel_immunotherapy",
    "paclitaxel_dna_damage",
    "her2_directed",
    "other",
    "unknown",
)
RD_ACTION_SUBTYPES = ("TNBC", "HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "other")


def _epsilon_schedule(epoch: int, warmup: int, anneal: int, eps_max: float) -> float:
    if epoch < warmup:
        return 0.0
    if anneal <= 0:
        return float(eps_max)
    progress = min(1.0, (epoch - warmup) / max(1, anneal))
    return float(eps_max * progress)


def sliced_wasserstein_tensor(
    X: Tensor, Y: Tensor, n_projections: int = 64, resample: int = 256
) -> Tensor:
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return torch.zeros((), device=X.device, dtype=X.dtype)
    if X.shape[0] > resample:
        idx = torch.randperm(X.shape[0], device=X.device)[:resample]
        X = X[idx]
    if Y.shape[0] > resample:
        idx = torch.randperm(Y.shape[0], device=Y.device)[:resample]
        Y = Y[idx]
    dirs = torch.randn((n_projections, X.shape[1]), device=X.device, dtype=X.dtype)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    px = X @ dirs.T
    py = Y @ dirs.T
    q = min(px.shape[0], py.shape[0])
    q_grid = torch.linspace(0, 1, q, device=X.device, dtype=X.dtype)
    qx = torch.quantile(px, q_grid, dim=0)
    qy = torch.quantile(py, q_grid, dim=0)
    return (qx - qy).abs().mean()


def chamfer_feat_loss(A: Tensor, B: Tensor) -> Tensor:
    if A.shape[0] == 0 or B.shape[0] == 0:
        return torch.zeros((), device=A.device, dtype=A.dtype)
    d = torch.cdist(A, B, p=2.0)
    return 0.5 * (d.min(dim=1).values.pow(2).mean() + d.min(dim=0).values.pow(2).mean())


def _raw_features(xn: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return xn * std + mean


def _apply_delta_feat(src_xn: Tensor, delta_feat: Tensor) -> Tensor:
    """Apply a predicted feature delta, allowing core-only deltas on SSL graphs."""
    if delta_feat.shape[1] == src_xn.shape[1]:
        return src_xn + delta_feat
    if delta_feat.shape[1] > src_xn.shape[1]:
        raise ValueError(
            f"delta_feat width {delta_feat.shape[1]} exceeds state width {src_xn.shape[1]}"
        )
    out = src_xn.clone()
    out[:, : delta_feat.shape[1]] = out[:, : delta_feat.shape[1]] + delta_feat
    return out


def _sample_ftv_ml(s: ConsistentSample, visit_idx: int, volume_idx: int) -> float:
    off = [int(v) for v in s.visit_offsets.tolist()]
    x_visit = s.x[off[visit_idx]:off[visit_idx + 1]]
    return float(x_visit[:, volume_idx].clamp_min(0.0).sum().item())


def _burden_context_from_visit(
    xn: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    volume_idx: int,
    alive: Tensor | None = None,
) -> Tensor:
    """Prediction-available tumor-burden context used for regime conditioning."""
    raw = _raw_features(xn, mean, std)
    volume = raw[:, volume_idx].clamp_min(0.0)
    if alive is None:
        alive_weight = torch.ones_like(volume)
    else:
        alive_weight = alive.to(device=xn.device, dtype=xn.dtype).reshape(-1)
    ftv = (volume * alive_weight).sum()
    alive_count = alive_weight.sum()
    node_count = torch.as_tensor(float(xn.shape[0]), device=xn.device, dtype=xn.dtype)
    mean_active_volume = ftv / alive_count.clamp_min(1.0)
    alive_fraction = alive_count / node_count.clamp_min(1.0)
    return torch.stack(
        [
            torch.log1p(ftv),
            torch.log1p(alive_count),
            torch.log1p(node_count),
            torch.log1p(mean_active_volume),
            alive_fraction.clamp(0.0, 1.0),
        ]
    )


def _compute_train_ftv_tail_threshold(
    samples: list[ConsistentSample],
    *,
    volume_idx: int,
    quantile: float,
) -> float:
    if quantile <= 0.0 or quantile >= 1.0:
        return 0.0
    values: list[float] = []
    for s in samples:
        n_visits = int(len(s.visit_offsets) - 1)
        for visit_idx in range(1, n_visits):
            values.append(_sample_ftv_ml(s, visit_idx, volume_idx))
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _compute_train_visit_ftv_tail_threshold(
    samples: list[ConsistentSample],
    *,
    visit_idx: int,
    volume_idx: int,
    quantile: float,
) -> float:
    if quantile <= 0.0 or quantile >= 1.0:
        return 0.0
    values: list[float] = []
    for s in samples:
        n_visits = int(len(s.visit_offsets) - 1)
        if n_visits > visit_idx:
            values.append(_sample_ftv_ml(s, visit_idx, volume_idx))
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _compute_train_response_ratio_tail_threshold(
    samples: list[ConsistentSample],
    *,
    volume_idx: int,
    quantile: float,
    eps_ml: float = 1e-3,
) -> float:
    if quantile <= 0.0 or quantile >= 1.0:
        return 0.0
    ratios: list[float] = []
    for s in samples:
        n_visits = int(len(s.visit_offsets) - 1)
        if n_visits >= 4:
            ftv0 = _sample_ftv_ml(s, 0, volume_idx)
            ftv3 = _sample_ftv_ml(s, 3, volume_idx)
            ratios.append(float((ftv3 + eps_ml) / (ftv0 + eps_ml)))
    if not ratios:
        return 0.0
    return float(np.quantile(np.asarray(ratios, dtype=np.float64), quantile))


def _bio_rollout_loss(
    alive_logit: Tensor,
    pred_xn: Tensor,
    dst_xn: Tensor,
    dst_alive: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    volume_idx: int,
    lambda_alive_bce: float,
    lambda_alive_mass: float,
    lambda_ftv: float,
    graph_ftv_log: Tensor | None = None,
    graph_ftv_affine: Tensor | None = None,
    graph_ftv_residual: Tensor | None = None,
    lambda_graph_ftv: float = 0.0,
    lambda_graph_ftv_consistency: float = 0.0,
    graph_ftv_residual_cap_ml: float = 1.0,
    lambda_graph_ftv_residual_l2: float = 0.0,
    ftv_tail_threshold_ml: float = 0.0,
    ftv_tail_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Aggregate alive/FTV penalties for one predicted visit."""
    total = torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
    stats = {
        "bio_alive_bce": float("nan"),
        "bio_alive_mass": float("nan"),
        "bio_log_ftv": float("nan"),
        "bio_pred_ftv_ml": float("nan"),
        "bio_obs_ftv_ml": float("nan"),
        "bio_pred_alive_mass": float("nan"),
        "bio_obs_alive_count": float("nan"),
        "bio_graph_ftv_ml": float("nan"),
        "bio_graph_log_ftv": float("nan"),
        "bio_graph_node_consistency": float("nan"),
        "bio_affine_ftv_ml": float("nan"),
        "bio_affine_scale": float("nan"),
        "bio_affine_bias_ml": float("nan"),
        "bio_ftv_residual_ml": float("nan"),
        "bio_ftv_residual_l2": float("nan"),
        "bio_log_ftv_weighted": float("nan"),
        "bio_ftv_loss_weight": float("nan"),
    }
    alive_target = dst_alive.float().reshape(-1)
    alive_prob = torch.sigmoid(alive_logit.reshape(-1))
    pred_alive_mass = alive_prob.sum()
    obs_alive_count = alive_target.sum()

    if lambda_alive_bce > 0.0:
        alive_bce = F.binary_cross_entropy_with_logits(alive_logit.reshape(-1), alive_target)
        total = total + (lambda_alive_bce * alive_bce)
        stats["bio_alive_bce"] = float(alive_bce.detach().item())

    if lambda_alive_mass > 0.0:
        alive_mass = (pred_alive_mass - obs_alive_count).abs()
        total = total + (lambda_alive_mass * alive_mass)
        stats["bio_alive_mass"] = float(alive_mass.detach().item())

    pred_raw = _raw_features(pred_xn, mean, std)
    obs_raw = _raw_features(dst_xn, mean, std)
    pred_vol = pred_raw[:, volume_idx].clamp_min(0.0)
    obs_vol = obs_raw[:, volume_idx].clamp_min(0.0)
    pred_ftv_node = (pred_vol * alive_prob).sum()
    pred_ftv = pred_ftv_node
    obs_ftv = obs_vol.sum()

    if graph_ftv_affine is not None:
        affine = graph_ftv_affine.reshape(-1)
        if affine.numel() != 2:
            raise ValueError(f"graph_ftv_affine must have 2 values, got {tuple(graph_ftv_affine.shape)}")
        scale = F.softplus(affine[0]).clamp(0.25, 4.0)
        bias = affine[1]
        pred_ftv = (scale * pred_ftv_node + bias).clamp_min(0.0)
        stats["bio_affine_ftv_ml"] = float(pred_ftv.detach().item())
        stats["bio_affine_scale"] = float(scale.detach().item())
        stats["bio_affine_bias_ml"] = float(bias.detach().item())

    if graph_ftv_residual is not None and graph_ftv_residual_cap_ml > 0.0:
        cap = torch.as_tensor(
            float(graph_ftv_residual_cap_ml),
            device=pred_ftv.device,
            dtype=pred_ftv.dtype,
        )
        resid = cap * torch.tanh(graph_ftv_residual.reshape(()))
        pred_ftv = (pred_ftv + resid).clamp_min(0.0)
        resid_l2 = (resid / cap.clamp_min(1e-8)).pow(2)
        stats["bio_ftv_residual_ml"] = float(resid.detach().item())
        stats["bio_ftv_residual_l2"] = float(resid_l2.detach().item())
        if lambda_graph_ftv_residual_l2 > 0.0:
            total = total + (lambda_graph_ftv_residual_l2 * resid_l2)

    if lambda_ftv > 0.0:
        log_ftv = (torch.log1p(pred_ftv) - torch.log1p(obs_ftv)).abs()
        ftv_weight = 1.0
        if ftv_tail_weight > 1.0 and ftv_tail_threshold_ml > 0.0:
            if float(obs_ftv.detach().item()) >= float(ftv_tail_threshold_ml):
                ftv_weight = float(ftv_tail_weight)
        weighted_log_ftv = log_ftv * ftv_weight
        total = total + (lambda_ftv * weighted_log_ftv)
        stats["bio_log_ftv"] = float(log_ftv.detach().item())
        stats["bio_log_ftv_weighted"] = float(weighted_log_ftv.detach().item())
        stats["bio_ftv_loss_weight"] = float(ftv_weight)
        stats["bio_pred_ftv_ml"] = float(pred_ftv.detach().item())
        stats["bio_obs_ftv_ml"] = float(obs_ftv.detach().item())

    if graph_ftv_log is not None:
        target_log = torch.log1p(obs_ftv.detach())
        graph_loss = F.smooth_l1_loss(graph_ftv_log.reshape(()), target_log.reshape(()))
        graph_ftv = torch.expm1(graph_ftv_log.reshape(())).clamp_min(0.0)
        stats["bio_graph_ftv_ml"] = float(graph_ftv.detach().item())
        stats["bio_graph_log_ftv"] = float(graph_loss.detach().item())
        if lambda_graph_ftv > 0.0:
            total = total + (lambda_graph_ftv * graph_loss)
        if lambda_graph_ftv_consistency > 0.0:
            consistency = F.smooth_l1_loss(graph_ftv_log.reshape(()), torch.log1p(pred_ftv_node.detach()).reshape(()))
            total = total + (lambda_graph_ftv_consistency * consistency)
            stats["bio_graph_node_consistency"] = float(consistency.detach().item())

    stats["bio_pred_alive_mass"] = float(pred_alive_mass.detach().item())
    stats["bio_obs_alive_count"] = float(obs_alive_count.detach().item())
    return total, stats


def _predict_ftv_from_outputs(
    out: dict[str, Tensor],
    pred_xn: Tensor,
    alive_prob: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    volume_idx: int,
    graph_ftv_residual_cap_ml: float,
) -> Tensor:
    """Return the FTV estimate used by rollout/eval for a predicted state."""
    pred_raw = _raw_features(pred_xn, mean, std)
    node_ftv = (pred_raw[:, volume_idx].clamp_min(0.0) * alive_prob.reshape(-1)).sum()
    pred_ftv = node_ftv
    if "graph_ftv_affine" in out:
        affine = out["graph_ftv_affine"].reshape(-1)
        scale = F.softplus(affine[0]).clamp(0.25, 4.0)
        pred_ftv = (scale * node_ftv + affine[1]).clamp_min(0.0)
    if "graph_ftv_residual" in out and graph_ftv_residual_cap_ml > 0.0:
        cap = torch.as_tensor(
            float(graph_ftv_residual_cap_ml),
            device=pred_ftv.device,
            dtype=pred_ftv.dtype,
        )
        resid = cap * torch.tanh(out["graph_ftv_residual"].reshape(()))
        pred_ftv = (pred_ftv + resid).clamp_min(0.0)
    if "graph_ftv_log" in out:
        pred_ftv = torch.expm1(out["graph_ftv_log"].reshape(())).clamp_min(0.0)
    return pred_ftv


def _response_direction_loss(
    pred_log_ratio: Tensor,
    obs_log_ratio: Tensor,
    *,
    shrink_log_threshold: float,
    grow_log_threshold: float,
    margin: float,
) -> Tensor:
    """Penalize wrong shrink/stable/grow direction in log-response space."""
    if float(obs_log_ratio.detach().item()) <= float(shrink_log_threshold):
        return F.relu(float(margin) + pred_log_ratio)
    if float(obs_log_ratio.detach().item()) >= float(grow_log_threshold):
        return F.relu(float(margin) - pred_log_ratio)
    return F.relu(pred_log_ratio.abs() - float(margin))


def _response_diffusion_loss(
    source_xn: Tensor,
    pred_xn: Tensor,
    alive_prob: Tensor,
    edge_index: Tensor,
    last_slice: slice,
    mean: Tensor,
    std: Tensor,
    *,
    volume_idx: int,
) -> Tensor:
    """Edge-smooth predicted node burden changes within the current tumor state."""
    if edge_index.numel() == 0:
        return torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
    start = int(last_slice.start or 0)
    stop = int(last_slice.stop or start)
    src, dst = edge_index[0], edge_index[1]
    mask = (src >= start) & (src < stop) & (dst >= start) & (dst < stop)
    if not bool(mask.any()):
        return torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
    src_local = src[mask] - start
    dst_local = dst[mask] - start
    source_raw = _raw_features(source_xn, mean, std)
    pred_raw = _raw_features(pred_xn, mean, std)
    source_node = source_raw[:, volume_idx].clamp_min(0.0)
    pred_node = pred_raw[:, volume_idx].clamp_min(0.0) * alive_prob.reshape(-1)
    delta = pred_node - source_node
    return F.smooth_l1_loss(delta[src_local], delta[dst_local])


def _edge_response_alignment_losses(
    source_xn: Tensor,
    pred_xn: Tensor,
    dst_xn: Tensor,
    alive_prob: Tensor,
    dst_alive: Tensor,
    edge_index: Tensor,
    last_slice: slice,
    mean: Tensor,
    std: Tensor,
    *,
    volume_idx: int,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Preserve observed pairwise response contrast on the current graph edges."""
    zero = torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
    stats = {
        "bio_edge_response_alignment": float("nan"),
        "bio_edge_alive_alignment": float("nan"),
        "bio_edge_alignment_edges": float("nan"),
    }
    if edge_index.numel() == 0:
        return zero, zero, stats
    start = int(last_slice.start or 0)
    stop = int(last_slice.stop or start)
    src, dst = edge_index[0], edge_index[1]
    mask = (src >= start) & (src < stop) & (dst >= start) & (dst < stop)
    if not bool(mask.any()):
        return zero, zero, stats

    src_local = src[mask] - start
    dst_local = dst[mask] - start
    n = min(
        int(source_xn.shape[0]),
        int(pred_xn.shape[0]),
        int(dst_xn.shape[0]),
        int(alive_prob.reshape(-1).shape[0]),
        int(dst_alive.reshape(-1).shape[0]),
    )
    if n <= 1:
        return zero, zero, stats
    valid = (src_local < n) & (dst_local < n)
    if not bool(valid.any()):
        return zero, zero, stats
    src_local = src_local[valid]
    dst_local = dst_local[valid]

    source_raw = _raw_features(source_xn[:n], mean, std)
    pred_raw = _raw_features(pred_xn[:n], mean, std)
    obs_raw = _raw_features(dst_xn[:n], mean, std)
    source_node = source_raw[:, volume_idx].clamp_min(0.0)
    pred_alive = alive_prob.reshape(-1)[:n].clamp(0.0, 1.0)
    obs_alive = dst_alive.reshape(-1)[:n].to(dtype=pred_xn.dtype).clamp(0.0, 1.0)
    pred_node = pred_raw[:, volume_idx].clamp_min(0.0) * pred_alive
    obs_node = obs_raw[:, volume_idx].clamp_min(0.0) * obs_alive

    eps_ml = torch.as_tensor(1e-3, device=pred_xn.device, dtype=pred_xn.dtype)
    pred_response = torch.log(pred_node + eps_ml) - torch.log(source_node + eps_ml)
    obs_response = torch.log(obs_node + eps_ml) - torch.log(source_node + eps_ml)
    response_loss = F.smooth_l1_loss(
        pred_response[src_local] - pred_response[dst_local],
        obs_response[src_local] - obs_response[dst_local],
    )
    alive_loss = F.smooth_l1_loss(
        pred_alive[src_local] - pred_alive[dst_local],
        obs_alive[src_local] - obs_alive[dst_local],
    )
    stats["bio_edge_response_alignment"] = float(response_loss.detach().item())
    stats["bio_edge_alive_alignment"] = float(alive_loss.detach().item())
    stats["bio_edge_alignment_edges"] = float(src_local.numel())
    return response_loss, alive_loss, stats


def _reaction_diffusion_residual_loss(
    source_xn: Tensor,
    pred_xn: Tensor,
    alive_prob: Tensor,
    edge_index: Tensor,
    last_slice: slice,
    mean: Tensor,
    std: Tensor,
    reaction_diffusion_raw: Tensor | None,
    *,
    volume_idx: int,
    capacity_ml: float,
    treatment_proxy: float,
    dt: float,
) -> tuple[Tensor, dict[str, float]]:
    """Constrain predicted burden change with a graph reaction-diffusion residual."""
    if reaction_diffusion_raw is None or edge_index.numel() == 0:
        zero = torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
        return zero, {}
    start = int(last_slice.start or 0)
    stop = int(last_slice.stop or start)
    n_nodes = max(0, stop - start)
    if n_nodes <= 1:
        zero = torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
        return zero, {}

    src, dst = edge_index[0], edge_index[1]
    mask = (src >= start) & (src < stop) & (dst >= start) & (dst < stop)
    if not bool(mask.any()):
        zero = torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
        return zero, {}

    src_local = src[mask] - start
    dst_local = dst[mask] - start
    source_raw = _raw_features(source_xn, mean, std)
    pred_raw = _raw_features(pred_xn, mean, std)
    source_node = source_raw[:, volume_idx].clamp_min(0.0)
    pred_node = pred_raw[:, volume_idx].clamp_min(0.0) * alive_prob.reshape(-1)
    delta_node = pred_node - source_node

    neigh_delta = source_node[dst_local] - source_node[src_local]
    lap = torch.zeros_like(source_node)
    degree = torch.zeros_like(source_node)
    lap.scatter_add_(0, src_local, neigh_delta)
    degree.scatter_add_(0, src_local, torch.ones_like(neigh_delta))
    lap = lap / degree.clamp_min(1.0)

    params = F.softplus(reaction_diffusion_raw.reshape(-1)[:3])
    d_hat, r_hat, kappa_hat = params[0], params[1], params[2]
    if capacity_ml > 0.0:
        k_cap = torch.as_tensor(float(capacity_ml), device=source_node.device, dtype=source_node.dtype)
    else:
        k_cap = source_node.detach().quantile(0.95).clamp_min(1e-3) * 1.25
    growth = r_hat * source_node * (1.0 - (source_node / k_cap).clamp_min(0.0))
    diffusion = d_hat * lap
    kill = kappa_hat * float(treatment_proxy) * source_node
    rd_delta = float(dt) * (diffusion + growth - kill)
    loss = F.smooth_l1_loss(delta_node, rd_delta)
    info = {
        "reaction_diffusion_residual_loss": float(loss.detach().item()),
        "reaction_diffusion_D": float(d_hat.detach().item()),
        "reaction_diffusion_r": float(r_hat.detach().item()),
        "reaction_diffusion_kappa": float(kappa_hat.detach().item()),
        "reaction_diffusion_delta_obs_mean": float(delta_node.detach().mean().item()),
        "reaction_diffusion_delta_model_mean": float(rd_delta.detach().mean().item()),
    }
    return loss, info


def _terminal_t0_t3_ftv_loss(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: Tensor,
    std: Tensor,
    *,
    k_spatial: int,
    edge_mode: str,
    edge_attr_mode: str,
    use_edge_types: bool,
    edge_attr_dropout: float,
    volume_idx: int,
    graph_ftv_residual_cap_ml: float,
    loss_space: str,
    horizon_weights: str = "",
    terminal_ftv_tail_threshold_ml: float = 0.0,
    terminal_ftv_tail_weight: float = 1.0,
    lambda_terminal_response_ratio: float = 0.0,
    lambda_terminal_direction: float = 0.0,
    terminal_response_tail_threshold: float = 0.0,
    terminal_response_tail_weight: float = 1.0,
    response_direction_margin: float = 0.05,
    response_shrink_threshold: float = 0.80,
    response_grow_threshold: float = 1.20,
    start_burden_tail_threshold_ml: float = 0.0,
    start_burden_tail_weight: float = 1.0,
    response_ratio_eps_ml: float = 1e-3,
) -> tuple[Tensor, dict[str, float]]:
    """Autoregress from T0 and penalize future FTV directly."""
    x_all = ((s.x.to(DEVICE) - mean) / std)
    pos_all = s.pos.to(DEVICE)
    alive_all = s.alive.to(DEVICE)
    off = [int(v) for v in s.visit_offsets.tolist()]
    T = len(off) - 1
    if T < 4:
        zero = torch.zeros((), device=DEVICE, dtype=x_all.dtype)
        return zero, {
            "terminal_t3_ftv_abs_ml": float("nan"),
            "terminal_t3_ftv_bias_ml": float("nan"),
            "terminal_t3_ftv_loss": float("nan"),
        }

    obs_xn = [x_all[off[v]:off[v + 1]] for v in range(T)]
    obs_pos = [pos_all[off[v]:off[v + 1]] for v in range(T)]
    obs_alive = [alive_all[off[v]:off[v + 1]] for v in range(T)]
    obs_raw0 = _raw_features(obs_xn[0], mean, std)
    start_ftv = (obs_raw0[:, volume_idx].clamp_min(0.0) * obs_alive[0].float()).sum().detach()
    burden_context = _burden_context_from_visit(
        obs_xn[0],
        mean,
        std,
        volume_idx=volume_idx,
        alive=obs_alive[0],
    )
    start_burden_multiplier = 1.0
    if (
        start_burden_tail_weight > 1.0
        and start_burden_tail_threshold_ml > 0.0
        and float(start_ftv.detach().item()) >= float(start_burden_tail_threshold_ml)
    ):
        start_burden_multiplier = float(start_burden_tail_weight)
    history_x = [obs_xn[0]]
    history_pos = [obs_pos[0]]
    history_t = [torch.zeros((obs_xn[0].shape[0],), device=DEVICE)]
    pred_ftv = torch.zeros((), device=DEVICE, dtype=x_all.dtype)
    if horizon_weights.strip():
        weights = [float(x) for x in horizon_weights.replace(";", ",").split(",") if x.strip()]
        if len(weights) != T - 1:
            raise ValueError(
                f"terminal FTV horizon weights must have {T - 1} values for this sample; "
                f"got {len(weights)} from {horizon_weights!r}"
            )
    else:
        weights = [0.0] * max(0, T - 2) + [1.0]
    weight_sum = float(sum(max(0.0, w) for w in weights))
    if weight_sum <= 0.0:
        zero = torch.zeros((), device=DEVICE, dtype=x_all.dtype)
        return zero, {
            "terminal_t3_ftv_abs_ml": float("nan"),
            "terminal_t3_ftv_bias_ml": float("nan"),
            "terminal_t3_ftv_loss": float("nan"),
        }

    total_loss = torch.zeros((), device=DEVICE, dtype=x_all.dtype)
    info: dict[str, float] = {}

    for k in range(T - 1):
        x_cat = torch.cat(history_x, dim=0)
        p_cat = torch.cat(history_pos, dim=0)
        t_cat = torch.cat(history_t, dim=0)
        edge_index, edge_attr, edge_type = build_history_graph_with_types(
            history_pos,
            history_x,
            k_spatial=k_spatial,
            edge_mode=edge_mode,
            edge_attr_mode=edge_attr_mode,
        )
        if edge_attr is not None and edge_attr_dropout > 0.0 and model.training:
            edge_attr = F.dropout(edge_attr, p=edge_attr_dropout, training=True)
        out = model(
            x_cat,
            p_cat,
            t_cat,
            edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type if use_edge_types else None,
            burden_context=burden_context,
        )
        n_last = history_x[-1].shape[0]
        sl = slice(x_cat.shape[0] - n_last, x_cat.shape[0])
        pred_pos = history_pos[-1] + out["delta_pos"][sl]
        pred_xn = _apply_delta_feat(history_x[-1], out["delta_feat"][sl])
        alive_prob = torch.sigmoid(out["alive_logit"][sl])
        pred_ftv = _predict_ftv_from_outputs(
            out,
            pred_xn,
            alive_prob,
            mean,
            std,
            volume_idx=volume_idx,
            graph_ftv_residual_cap_ml=graph_ftv_residual_cap_ml,
        )
        obs_raw_h = _raw_features(obs_xn[k + 1], mean, std)
        obs_ftv_h = obs_raw_h[:, volume_idx].clamp_min(0.0).sum()
        obs_response_ratio = float(
            ((obs_ftv_h.detach() + response_ratio_eps_ml) / (start_ftv + response_ratio_eps_ml)).item()
        )
        response_tail_multiplier = 1.0
        if (
            terminal_response_tail_weight > 1.0
            and terminal_response_tail_threshold > 0.0
            and obs_response_ratio <= float(terminal_response_tail_threshold)
        ):
            response_tail_multiplier = float(terminal_response_tail_weight)
        if loss_space == "ml":
            loss_h = F.smooth_l1_loss(
                pred_ftv.reshape(()),
                obs_ftv_h.detach().reshape(()),
                beta=5.0,
            )
        else:
            loss_h = F.smooth_l1_loss(
                torch.log1p(pred_ftv).reshape(()),
                torch.log1p(obs_ftv_h.detach()).reshape(()),
            )
        w = max(0.0, float(weights[k]))
        ftv_loss_weight = w * start_burden_multiplier
        if (
            w > 0.0
            and terminal_ftv_tail_weight > 1.0
            and terminal_ftv_tail_threshold_ml > 0.0
            and float(obs_ftv_h.detach().item()) >= float(terminal_ftv_tail_threshold_ml)
        ):
            ftv_loss_weight *= float(terminal_ftv_tail_weight)
        if w > 0.0 and response_tail_multiplier > 1.0:
            ftv_loss_weight *= response_tail_multiplier
        if ftv_loss_weight > 0.0:
            total_loss = total_loss + (ftv_loss_weight * loss_h)
        response_ratio_loss = torch.zeros((), device=DEVICE, dtype=x_all.dtype)
        direction_loss = torch.zeros((), device=DEVICE, dtype=x_all.dtype)
        pred_response_ratio = float("nan")
        if w > 0.0 and (lambda_terminal_response_ratio > 0.0 or lambda_terminal_direction > 0.0):
            pred_log_ratio = torch.log(pred_ftv.clamp_min(0.0) + response_ratio_eps_ml) - torch.log(
                start_ftv + response_ratio_eps_ml
            )
            obs_log_ratio = torch.log(obs_ftv_h.detach() + response_ratio_eps_ml) - torch.log(
                start_ftv + response_ratio_eps_ml
            )
            response_ratio_loss = F.smooth_l1_loss(
                pred_log_ratio.reshape(()),
                obs_log_ratio.reshape(()),
            )
            if lambda_terminal_response_ratio > 0.0:
                total_loss = total_loss + (
                    w
                    * start_burden_multiplier
                    * response_tail_multiplier
                    * float(lambda_terminal_response_ratio)
                    * response_ratio_loss
                )
            if lambda_terminal_direction > 0.0:
                direction_loss = _response_direction_loss(
                    pred_log_ratio.reshape(()),
                    obs_log_ratio.reshape(()),
                    shrink_log_threshold=float(np.log(response_shrink_threshold)),
                    grow_log_threshold=float(np.log(response_grow_threshold)),
                    margin=response_direction_margin,
                )
                total_loss = total_loss + (
                    w
                    * start_burden_multiplier
                    * response_tail_multiplier
                    * float(lambda_terminal_direction)
                    * direction_loss
                )
            pred_response_ratio = float(
                ((pred_ftv.detach() + response_ratio_eps_ml) / (start_ftv + response_ratio_eps_ml)).item()
            )
        visit = VISITS[k + 1] if k + 1 < len(VISITS) else f"T{k + 1}"
        bias_h = pred_ftv - obs_ftv_h
        info[f"terminal_{visit.lower()}_ftv_abs_ml"] = float(bias_h.detach().abs().item())
        info[f"terminal_{visit.lower()}_ftv_bias_ml"] = float(bias_h.detach().item())
        info[f"terminal_{visit.lower()}_ftv_loss"] = float(loss_h.detach().item())
        info[f"terminal_{visit.lower()}_loss_weight"] = float(ftv_loss_weight)
        info[f"terminal_{visit.lower()}_response_ratio_obs"] = float(obs_response_ratio)
        info[f"terminal_{visit.lower()}_response_ratio_pred"] = float(pred_response_ratio)
        info[f"terminal_{visit.lower()}_response_ratio_loss"] = float(response_ratio_loss.detach().item())
        info[f"terminal_{visit.lower()}_direction_loss"] = float(direction_loss.detach().item())
        info[f"terminal_{visit.lower()}_response_tail_weight"] = float(response_tail_multiplier)
        info[f"terminal_{visit.lower()}_start_burden_weight"] = float(start_burden_multiplier)
        history_x.append(pred_xn)
        history_pos.append(pred_pos)
        history_t.append(torch.full((pred_xn.shape[0],), float(k + 1), device=DEVICE))

    obs_raw = _raw_features(obs_xn[3], mean, std)
    obs_ftv = obs_raw[:, volume_idx].clamp_min(0.0).sum()
    bias = pred_ftv - obs_ftv
    info.update({
        "terminal_t3_ftv_abs_ml": float(bias.detach().abs().item()),
        "terminal_t3_ftv_bias_ml": float(bias.detach().item()),
        "terminal_t3_ftv_loss": info.get("terminal_t3_ftv_loss", float("nan")),
        "terminal_weighted_ftv_loss": float((total_loss / weight_sum).detach().item()),
        "terminal_start_ftv_ml": float(start_ftv.detach().item()),
        "terminal_start_burden_weight": float(start_burden_multiplier),
    })
    return total_loss / weight_sum, info


def _load_pcr_map(cohort_parquet: str | None) -> dict[str, float]:
    if not cohort_parquet:
        return {}
    path = Path(cohort_parquet)
    if not path.is_file():
        return {}
    import pandas as pd

    df = pd.read_parquet(path)
    if "patient_id" not in df.columns or "pCR" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for pid, pcr in zip(df["patient_id"].astype(str), df["pCR"]):
        if pd.notna(pcr):
            out[str(pid)] = float(pcr)
    return out


def _rd_action_context_dim(mode: str) -> int:
    if mode not in RD_ACTION_CONTEXT_MODES:
        raise ValueError(f"Unknown reaction-diffusion action context mode: {mode}")
    dim = 3
    if mode in {"components", "components_subtype"}:
        dim += len(RD_ACTION_COMPONENTS) + len(RD_ACTION_FAMILIES)
    if mode == "components_subtype":
        dim += 3 + len(RD_ACTION_SUBTYPES) + 2
    return dim


def _safe_float_meta(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    return v if np.isfinite(v) else float(default)


def _normalise_arm_text(arm: Any) -> str:
    if arm is None:
        return ""
    text = str(arm).lower()
    for old, new in {
        "-": " ",
        "+": " ",
        "/": " ",
        "_": " ",
        "  ": " ",
    }.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _rd_arm_components(arm: Any) -> np.ndarray:
    text = _normalise_arm_text(arm)
    flags = {
        "paclitaxel": "paclitaxel" in text,
        "pembrolizumab": "pembrolizumab" in text,
        "carboplatin": "carboplatin" in text,
        "abt888": "abt 888" in text or "abt888" in text,
        "mk2206": "mk 2206" in text or "mk2206" in text,
        "neratinib": "neratinib" in text,
        "ganitumab": "ganitumab" in text,
        "ganetespib": "ganetespib" in text,
        "amg386": "amg 386" in text or "amg386" in text,
        "trastuzumab": "trastuzumab" in text,
        "pertuzumab": "pertuzumab" in text,
        "tdm1": "t dm1" in text or "tdm1" in text or "t dm 1" in text,
    }
    return np.asarray([1.0 if flags[name] else 0.0 for name in RD_ACTION_COMPONENTS], dtype=np.float32)


def _rd_arm_family(arm: Any) -> str:
    text = _normalise_arm_text(arm)
    if not text:
        return "unknown"
    has_pac = "paclitaxel" in text
    has_immuno = "pembrolizumab" in text
    has_dna = "carboplatin" in text or "abt 888" in text or "abt888" in text
    has_her2 = (
        "trastuzumab" in text
        or "pertuzumab" in text
        or "t dm1" in text
        or "tdm1" in text
        or "t dm 1" in text
    )
    targeted_terms = (
        "mk 2206",
        "mk2206",
        "neratinib",
        "ganitumab",
        "ganetespib",
        "amg 386",
        "amg386",
    )
    if has_immuno:
        return "paclitaxel_immunotherapy"
    if has_dna:
        return "paclitaxel_dna_damage"
    if has_her2:
        return "her2_directed"
    if has_pac and any(term in text for term in targeted_terms):
        return "paclitaxel_targeted"
    if has_pac:
        return "paclitaxel_only"
    return "other"


def _one_hot(value: str, choices: tuple[str, ...]) -> np.ndarray:
    out = np.zeros((len(choices),), dtype=np.float32)
    if value in choices:
        out[choices.index(value)] = 1.0
    elif choices:
        out[-1] = 1.0
    return out


def _load_rd_action_meta(cohort_parquet: str | None) -> dict[str, dict[str, Any]]:
    if not cohort_parquet:
        return {}
    path = Path(cohort_parquet)
    if not path.is_file():
        return {}
    import pandas as pd

    df = pd.read_parquet(path)
    if "patient_id" not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        arm = row.get("arm", "")
        subtype = str(row.get("subtype", "other"))
        if subtype not in RD_ACTION_SUBTYPES:
            subtype = "other"
        out[pid] = {
            "arm": arm,
            "component_flags": _rd_arm_components(arm),
            "family": _rd_arm_family(arm),
            "hr": _safe_float_meta(row.get("HR", 0.0)),
            "her2": _safe_float_meta(row.get("HER2", 0.0)),
            "mp": _safe_float_meta(row.get("MP", 0.0)),
            "subtype": subtype,
            "age": _safe_float_meta(row.get("age", 0.0)),
        }
    return out


def _shuffle_rd_action_meta(
    meta_by_patient: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Shuffle only arm-derived fields, keeping patient subtype/burden unchanged."""
    pids = sorted(meta_by_patient)
    donor_pids = pids[:]
    rng = random.Random(int(seed))
    rng.shuffle(donor_pids)
    out: dict[str, dict[str, Any]] = {}
    for pid, donor_pid in zip(pids, donor_pids):
        own = dict(meta_by_patient[pid])
        donor = meta_by_patient[donor_pid]
        own["arm"] = donor.get("arm", "")
        own["component_flags"] = np.asarray(donor.get("component_flags", np.zeros(len(RD_ACTION_COMPONENTS))), dtype=np.float32)
        own["family"] = str(donor.get("family", "unknown"))
        out[pid] = own
    return out


def _rd_action_context_tensor(
    patient_id: str,
    transition_idx: int,
    source_ftv_ml: Tensor,
    meta_by_patient: dict[str, dict[str, Any]] | None,
    shuffle_meta_by_patient: dict[str, dict[str, Any]] | None,
    *,
    mode: str,
    device: str,
    dtype: torch.dtype,
) -> Tensor | None:
    if mode not in RD_ACTION_CONTEXT_MODES:
        raise ValueError(f"Unknown reaction-diffusion action context mode: {mode}")
    meta = (meta_by_patient or {}).get(str(patient_id), {})
    arm_meta = (shuffle_meta_by_patient or {}).get(str(patient_id), meta)

    phase = np.zeros((3,), dtype=np.float32)
    phase[int(max(0, min(2, transition_idx)))] = 1.0
    parts: list[np.ndarray] = [phase]

    if mode in {"components", "components_subtype"}:
        if transition_idx < 2:
            components = np.asarray(
                arm_meta.get("component_flags", np.zeros((len(RD_ACTION_COMPONENTS),), dtype=np.float32)),
                dtype=np.float32,
            )
            family = _one_hot(str(arm_meta.get("family", "unknown")), RD_ACTION_FAMILIES)
        else:
            components = np.zeros((len(RD_ACTION_COMPONENTS),), dtype=np.float32)
            family = _one_hot("other", RD_ACTION_FAMILIES)
        parts.extend([components, family])

    if mode == "components_subtype":
        clinical = np.asarray(
            [
                _safe_float_meta(meta.get("hr", 0.0)),
                _safe_float_meta(meta.get("her2", 0.0)),
                _safe_float_meta(meta.get("mp", 0.0)),
            ],
            dtype=np.float32,
        )
        subtype = _one_hot(str(meta.get("subtype", "other")), RD_ACTION_SUBTYPES)
        source_ftv = float(source_ftv_ml.detach().clamp_min(0.0).cpu().item())
        age = _safe_float_meta(meta.get("age", 0.0))
        burden = np.asarray(
            [
                np.log1p(source_ftv),
                age / 100.0 if age > 0.0 else 0.0,
            ],
            dtype=np.float32,
        )
        parts.extend([clinical, subtype, burden])

    arr = np.concatenate(parts).astype(np.float32)
    expected = _rd_action_context_dim(mode)
    if int(arr.shape[0]) != expected:
        raise RuntimeError(f"RD action context width {arr.shape[0]} != expected {expected}")
    return torch.as_tensor(arr, device=device, dtype=dtype)


def _transition_tf_loss(
    dp: Tensor,
    dh: Tensor,
    src_pos: Tensor,
    dst_pos: Tensor,
    src_xn: Tensor,
    dst_xn: Tensor,
    dst_alive: Tensor,
    lambda_pos: float,
    lambda_feat: float,
) -> tuple[Tensor, float]:
    mask = dst_alive.bool()
    if mask.sum() == 0:
        z = torch.zeros((), device=dp.device, dtype=dp.dtype)
        return z, float("nan")
    gt_dp = (dst_pos - src_pos)[mask]
    gt_dh = (dst_xn[:, : dh.shape[1]] - src_xn[:, : dh.shape[1]])[mask]
    pred_dp = dp[mask]
    pred_dh = dh[mask]
    l_pos = F.mse_loss(pred_dp, gt_dp)
    l_feat = F.mse_loss(pred_dh, gt_dh)
    pos_mae = float((pred_dp - gt_dp).norm(dim=-1).mean().detach().item())
    return lambda_pos * l_pos + lambda_feat * l_feat, pos_mae


def _latent_change_losses(
    pred_xn: Tensor,
    src_xn: Tensor,
    dst_xn: Tensor,
    dst_alive: Tensor,
    *,
    start: int,
    end: int,
    lambda_delta: float,
    lambda_anticopy: float,
    margin: float,
    change_threshold: float,
) -> tuple[Tensor, dict[str, float]]:
    z = torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
    stats = {
        "patch_recon_latent_delta_loss": float("nan"),
        "patch_recon_anticopy_loss": float("nan"),
        "patch_recon_pred_latent_err": float("nan"),
        "patch_recon_copy_latent_err": float("nan"),
        "patch_recon_changed_frac": float("nan"),
    }
    if end <= start or start < 0 or end > int(pred_xn.shape[1]):
        return z, stats
    if lambda_delta <= 0.0 and lambda_anticopy <= 0.0:
        return z, stats
    mask = dst_alive.bool()
    if int(mask.sum().item()) == 0:
        return z, stats

    pred_lat = pred_xn[mask, start:end]
    src_lat = src_xn[mask, start:end]
    dst_lat = dst_xn[mask, start:end]
    pred_delta = pred_lat - src_lat
    true_delta = dst_lat - src_lat

    total = z
    delta_loss = F.smooth_l1_loss(pred_delta, true_delta)
    stats["patch_recon_latent_delta_loss"] = float(delta_loss.detach().item())
    if lambda_delta > 0.0:
        total = total + (float(lambda_delta) * delta_loss)

    pred_err_node = F.smooth_l1_loss(pred_lat, dst_lat, reduction="none").mean(dim=1)
    copy_err_node = F.smooth_l1_loss(src_lat, dst_lat, reduction="none").mean(dim=1)
    stats["patch_recon_pred_latent_err"] = float(pred_err_node.detach().mean().item())
    stats["patch_recon_copy_latent_err"] = float(copy_err_node.detach().mean().item())
    changed = copy_err_node.detach() > float(change_threshold)
    stats["patch_recon_changed_frac"] = float(changed.to(dtype=torch.float32).mean().detach().item())
    if lambda_anticopy > 0.0 and int(changed.sum().item()) > 0:
        rank_loss = F.relu(pred_err_node[changed] - copy_err_node[changed] + float(margin)).mean()
        stats["patch_recon_anticopy_loss"] = float(rank_loss.detach().item())
        total = total + (float(lambda_anticopy) * rank_loss)
    elif lambda_anticopy > 0.0:
        stats["patch_recon_anticopy_loss"] = 0.0
    return total, stats


def curriculum_forward_patient(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: Tensor,
    std: Tensor,
    *,
    epsilon: float,
    rng: random.Random,
    k_spatial: int,
    edge_mode: str,
    edge_attr_mode: str,
    use_edge_types: bool,
    lambda_pos: float,
    lambda_feat: float,
    lambda_cloud: float,
    lambda_cloud_feat: float,
    lambda_horizon: float,
    lambda_alive_bce: float,
    lambda_alive_mass: float,
    lambda_ftv: float,
    lambda_graph_ftv: float,
    lambda_graph_ftv_consistency: float,
    graph_ftv_residual_cap_ml: float,
    lambda_graph_ftv_residual_l2: float,
    ftv_tail_threshold_ml: float,
    ftv_tail_weight: float,
    edge_attr_dropout: float,
    lambda_pcr_aux: float,
    pcr_target: float | None,
    lambda_terminal_ftv: float,
    terminal_ftv_loss_space: str,
    terminal_ftv_horizon_weights: str,
    terminal_ftv_tail_threshold_ml: float,
    terminal_ftv_tail_weight: float,
    lambda_terminal_response_ratio: float,
    lambda_step_response_ratio: float,
    lambda_terminal_direction: float,
    lambda_step_direction: float,
    lambda_response_diffusion: float,
    lambda_edge_response: float,
    lambda_edge_alive: float,
    lambda_reaction_diffusion_residual: float,
    reaction_diffusion_capacity_ml: float,
    reaction_diffusion_treatment_proxy: float,
    reaction_diffusion_action_context: bool,
    reaction_diffusion_action_context_mode: str,
    reaction_diffusion_action_meta: dict[str, dict[str, Any]] | None,
    reaction_diffusion_shuffle_meta: dict[str, dict[str, Any]] | None,
    terminal_response_tail_threshold: float,
    terminal_response_tail_weight: float,
    response_direction_margin: float,
    response_shrink_threshold: float,
    response_grow_threshold: float,
    start_burden_tail_threshold_ml: float,
    start_burden_tail_weight: float,
    recon_latent_start: int,
    recon_latent_end: int,
    lambda_recon_latent_delta: float,
    lambda_recon_anticopy: float,
    recon_anticopy_margin: float,
    recon_change_threshold: float,
    volume_idx: int,
    random_start: bool,
) -> tuple[Tensor, dict]:
    x_all = ((s.x.to(DEVICE) - mean) / std)
    pos_all = s.pos.to(DEVICE)
    alive_all = s.alive.to(DEVICE)
    off = [int(v) for v in s.visit_offsets.tolist()]
    T = len(off) - 1

    obs_xn = [x_all[off[v]:off[v + 1]] for v in range(T)]
    obs_pos = [pos_all[off[v]:off[v + 1]] for v in range(T)]
    obs_alive = [alive_all[off[v]:off[v + 1]] for v in range(T)]

    start = rng.randint(0, T - 2) if random_start else 0
    history_x = [obs_xn[v] for v in range(start + 1)]
    history_pos = [obs_pos[v] for v in range(start + 1)]
    history_t = [torch.full((obs_xn[v].shape[0],), float(v), device=DEVICE) for v in range(start + 1)]
    burden_context = _burden_context_from_visit(
        obs_xn[start],
        mean,
        std,
        volume_idx=volume_idx,
        alive=obs_alive[start],
    )
    selffed_steps = 0
    maes: list[float] = []
    bio_stats: dict[str, list[float]] = {}
    pcr_stats: dict[str, list[float]] = {}
    total = torch.zeros((), device=DEVICE)

    for k in range(start, T - 1):
        x_cat = torch.cat(history_x, dim=0)
        p_cat = torch.cat(history_pos, dim=0)
        t_cat = torch.cat(history_t, dim=0)
        edge_index, edge_attr, edge_type = build_history_graph_with_types(
            history_pos,
            history_x,
            k_spatial=k_spatial,
            edge_mode=edge_mode,
            edge_attr_mode=edge_attr_mode,
        )
        if edge_attr is not None and edge_attr_dropout > 0.0 and model.training:
            edge_attr = F.dropout(edge_attr, p=edge_attr_dropout, training=True)
        rd_context = None
        if reaction_diffusion_action_context:
            source_raw_for_context = _raw_features(history_x[-1], mean, std)
            source_ftv_for_context = source_raw_for_context[:, volume_idx].clamp_min(0.0).sum().detach()
            rd_context = _rd_action_context_tensor(
                s.patient_id,
                k,
                source_ftv_for_context,
                reaction_diffusion_action_meta,
                reaction_diffusion_shuffle_meta,
                mode=reaction_diffusion_action_context_mode,
                device=DEVICE,
                dtype=x_cat.dtype,
            )
        out = model(
            x_cat,
            p_cat,
            t_cat,
            edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type if use_edge_types else None,
            burden_context=burden_context,
            reaction_diffusion_context=rd_context,
        )
        n_last = history_x[-1].shape[0]
        sl = slice(x_cat.shape[0] - n_last, x_cat.shape[0])
        src_xn_step = history_x[-1]
        dp = out["delta_pos"][sl]
        dh = out["delta_feat"][sl]
        alive_logit = out["alive_logit"][sl]

        pred_pos = history_pos[-1] + dp
        pred_xn = _apply_delta_feat(history_x[-1], dh)
        dst_pos = obs_pos[k + 1]
        dst_xn = obs_xn[k + 1]
        dst_alive = obs_alive[k + 1]

        use_self_feed = (k > start) and (rng.random() < epsilon)
        if use_self_feed:
            selffed_steps += 1
            l = lambda_cloud * sliced_wasserstein_tensor(pred_pos, dst_pos)
            if lambda_cloud_feat > 0.0:
                l = l + (lambda_cloud_feat * chamfer_feat_loss(pred_xn, dst_xn))
            total = total + l
            maes.append(float(sliced_wasserstein_tensor(pred_pos.detach(), dst_pos.detach()).item()))
            history_x.append(pred_xn)
            history_pos.append(pred_pos)
        else:
            l_tf, pos_mae = _transition_tf_loss(
                dp, dh,
                history_pos[-1], dst_pos,
                history_x[-1], dst_xn,
                dst_alive,
                lambda_pos=lambda_pos,
                lambda_feat=lambda_feat,
            )
            total = total + l_tf
            if np.isfinite(pos_mae):
                maes.append(float(pos_mae))
            history_x.append(dst_xn)
            history_pos.append(dst_pos)

        recon_l, recon_info = _latent_change_losses(
            pred_xn,
            src_xn_step,
            dst_xn,
            dst_alive,
            start=int(recon_latent_start),
            end=int(recon_latent_end),
            lambda_delta=float(lambda_recon_latent_delta),
            lambda_anticopy=float(lambda_recon_anticopy),
            margin=float(recon_anticopy_margin),
            change_threshold=float(recon_change_threshold),
        )
        total = total + recon_l
        for key, val in recon_info.items():
            if np.isfinite(val):
                bio_stats.setdefault(key, []).append(float(val))

        bio_l, bio_info = _bio_rollout_loss(
            alive_logit,
            pred_xn,
            dst_xn,
            dst_alive,
            mean,
            std,
            volume_idx=volume_idx,
            lambda_alive_bce=lambda_alive_bce,
            lambda_alive_mass=lambda_alive_mass,
            lambda_ftv=lambda_ftv,
            graph_ftv_log=out.get("graph_ftv_log"),
            graph_ftv_affine=out.get("graph_ftv_affine"),
            graph_ftv_residual=out.get("graph_ftv_residual"),
            lambda_graph_ftv=lambda_graph_ftv,
            lambda_graph_ftv_consistency=lambda_graph_ftv_consistency,
            graph_ftv_residual_cap_ml=graph_ftv_residual_cap_ml,
            lambda_graph_ftv_residual_l2=lambda_graph_ftv_residual_l2,
            ftv_tail_threshold_ml=ftv_tail_threshold_ml,
            ftv_tail_weight=ftv_tail_weight,
        )
        total = total + bio_l
        if (
            lambda_step_response_ratio > 0.0
            or lambda_step_direction > 0.0
            or lambda_response_diffusion > 0.0
            or lambda_edge_response > 0.0
            or lambda_edge_alive > 0.0
            or lambda_reaction_diffusion_residual > 0.0
        ):
            alive_prob = torch.sigmoid(alive_logit)
            pred_ftv_step = _predict_ftv_from_outputs(
                out,
                pred_xn,
                alive_prob,
                mean,
                std,
                volume_idx=volume_idx,
                graph_ftv_residual_cap_ml=graph_ftv_residual_cap_ml,
            )
            src_raw_step = _raw_features(src_xn_step, mean, std)
            dst_raw_step = _raw_features(dst_xn, mean, std)
            src_ftv_step = src_raw_step[:, volume_idx].clamp_min(0.0).sum().detach()
            obs_ftv_step = dst_raw_step[:, volume_idx].clamp_min(0.0).sum().detach()
            eps_ml = 1e-3
            pred_log_ratio = torch.log(pred_ftv_step.clamp_min(0.0) + eps_ml) - torch.log(src_ftv_step + eps_ml)
            obs_log_ratio = torch.log(obs_ftv_step + eps_ml) - torch.log(src_ftv_step + eps_ml)
            step_ratio_loss = F.smooth_l1_loss(pred_log_ratio.reshape(()), obs_log_ratio.reshape(()))
            if lambda_step_response_ratio > 0.0:
                total = total + (float(lambda_step_response_ratio) * step_ratio_loss)
            bio_stats.setdefault("step_response_ratio_loss", []).append(float(step_ratio_loss.detach().item()))
            if lambda_step_direction > 0.0:
                step_direction_loss = _response_direction_loss(
                    pred_log_ratio.reshape(()),
                    obs_log_ratio.reshape(()),
                    shrink_log_threshold=float(np.log(response_shrink_threshold)),
                    grow_log_threshold=float(np.log(response_grow_threshold)),
                    margin=response_direction_margin,
                )
                total = total + (float(lambda_step_direction) * step_direction_loss)
                bio_stats.setdefault("step_direction_loss", []).append(float(step_direction_loss.detach().item()))
            if lambda_response_diffusion > 0.0:
                diffusion_loss = _response_diffusion_loss(
                    src_xn_step,
                    pred_xn,
                    alive_prob,
                    edge_index,
                    sl,
                    mean,
                    std,
                    volume_idx=volume_idx,
                )
                total = total + (float(lambda_response_diffusion) * diffusion_loss)
                bio_stats.setdefault("response_diffusion_loss", []).append(float(diffusion_loss.detach().item()))
            if lambda_edge_response > 0.0 or lambda_edge_alive > 0.0:
                edge_response_loss, edge_alive_loss, edge_info = _edge_response_alignment_losses(
                    src_xn_step,
                    pred_xn,
                    dst_xn,
                    alive_prob,
                    dst_alive,
                    edge_index,
                    sl,
                    mean,
                    std,
                    volume_idx=volume_idx,
                )
                if lambda_edge_response > 0.0:
                    total = total + (float(lambda_edge_response) * edge_response_loss)
                if lambda_edge_alive > 0.0:
                    total = total + (float(lambda_edge_alive) * edge_alive_loss)
                for key, val in edge_info.items():
                    if np.isfinite(val):
                        bio_stats.setdefault(key, []).append(float(val))
            if lambda_reaction_diffusion_residual > 0.0:
                rd_loss, rd_info = _reaction_diffusion_residual_loss(
                    src_xn_step,
                    pred_xn,
                    alive_prob,
                    edge_index,
                    sl,
                    mean,
                    std,
                    out.get("reaction_diffusion_raw"),
                    volume_idx=volume_idx,
                    capacity_ml=reaction_diffusion_capacity_ml,
                    treatment_proxy=reaction_diffusion_treatment_proxy,
                    dt=1.0,
                )
                total = total + (float(lambda_reaction_diffusion_residual) * rd_loss)
                for key, val in rd_info.items():
                    if np.isfinite(val):
                        bio_stats.setdefault(key, []).append(float(val))
        for key, val in bio_info.items():
            if np.isfinite(val):
                bio_stats.setdefault(key, []).append(float(val))

        if (
            lambda_pcr_aux > 0.0
            and pcr_target is not None
            and k == T - 2
            and "graph_pcr_logit" in out
        ):
            target = torch.tensor(float(pcr_target), dtype=alive_logit.dtype, device=DEVICE)
            pcr_loss = F.binary_cross_entropy_with_logits(
                out["graph_pcr_logit"].reshape(()),
                target.reshape(()),
            )
            total = total + (lambda_pcr_aux * pcr_loss)
            pcr_prob = torch.sigmoid(out["graph_pcr_logit"].reshape(()))
            pcr_stats.setdefault("pcr_aux_bce", []).append(float(pcr_loss.detach().item()))
            pcr_stats.setdefault("pcr_aux_prob", []).append(float(pcr_prob.detach().item()))
            pcr_stats.setdefault("pcr_aux_target", []).append(float(target.detach().item()))

        history_t.append(torch.full((history_x[-1].shape[0],), float(k + 1), device=DEVICE))

        if lambda_horizon > 0.0 and k == T - 2:
            total = total + (lambda_horizon * sliced_wasserstein_tensor(pred_pos, dst_pos))

    terminal_info: dict[str, float] = {}
    if lambda_terminal_ftv > 0.0:
        terminal_loss, terminal_info = _terminal_t0_t3_ftv_loss(
            model,
            s,
            mean,
            std,
            k_spatial=k_spatial,
            edge_mode=edge_mode,
            edge_attr_mode=edge_attr_mode,
            use_edge_types=use_edge_types,
            edge_attr_dropout=edge_attr_dropout,
            volume_idx=volume_idx,
            graph_ftv_residual_cap_ml=graph_ftv_residual_cap_ml,
            loss_space=terminal_ftv_loss_space,
            horizon_weights=terminal_ftv_horizon_weights,
            terminal_ftv_tail_threshold_ml=terminal_ftv_tail_threshold_ml,
            terminal_ftv_tail_weight=terminal_ftv_tail_weight,
            lambda_terminal_response_ratio=lambda_terminal_response_ratio,
            lambda_terminal_direction=lambda_terminal_direction,
            terminal_response_tail_threshold=terminal_response_tail_threshold,
            terminal_response_tail_weight=terminal_response_tail_weight,
            response_direction_margin=response_direction_margin,
            response_shrink_threshold=response_shrink_threshold,
            response_grow_threshold=response_grow_threshold,
            start_burden_tail_threshold_ml=start_burden_tail_threshold_ml,
            start_burden_tail_weight=start_burden_tail_weight,
        )
        total = total + (lambda_terminal_ftv * terminal_loss)

    n_steps = max(1, T - 1 - start)
    bio_summary = {
        key: float(np.mean(vals)) if vals else float("nan")
        for key, vals in bio_stats.items()
    }
    pcr_summary = {
        key: float(np.mean(vals)) if vals else float("nan")
        for key, vals in pcr_stats.items()
    }
    return total / n_steps, {
        "start_visit": start,
        "n_self_fed": selffed_steps,
        "mean_step_mae": float(np.mean(maes)) if maes else float("nan"),
        **bio_summary,
        **pcr_summary,
        **terminal_info,
    }


@torch.no_grad()
def rollout_eval_metrics(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: Tensor,
    std: Tensor,
    k_spatial: int,
    edge_mode: str,
    edge_attr_mode: str,
    use_edge_types: bool,
    volume_idx: int,
    graph_ftv_residual_cap_ml: float = 1.0,
) -> dict[str, float]:
    x_all = ((s.x.to(DEVICE) - mean) / std)
    pos_all = s.pos.to(DEVICE)
    alive_all = s.alive.to(DEVICE)
    off = [int(v) for v in s.visit_offsets.tolist()]
    T = len(off) - 1

    obs_xn = [x_all[off[v]:off[v + 1]] for v in range(T)]
    obs_pos = [pos_all[off[v]:off[v + 1]] for v in range(T)]
    obs_alive = [alive_all[off[v]:off[v + 1]] for v in range(T)]

    history_x = [obs_xn[0]]
    history_pos = [obs_pos[0]]
    history_t = [torch.zeros((obs_xn[0].shape[0],), device=DEVICE)]
    burden_context = _burden_context_from_visit(
        obs_xn[0],
        mean,
        std,
        volume_idx=volume_idx,
        alive=obs_alive[0],
    )
    maes: list[float] = []
    ftv_abs: list[float] = []
    ftv_bias: list[float] = []
    alive_abs: list[float] = []
    alive_bias: list[float] = []
    t3_ftv_abs = float("nan")
    t3_ftv_bias = float("nan")
    t3_alive_abs = float("nan")
    t3_alive_bias = float("nan")

    for k in range(T - 1):
        x_cat = torch.cat(history_x, dim=0)
        p_cat = torch.cat(history_pos, dim=0)
        t_cat = torch.cat(history_t, dim=0)
        edge_index, edge_attr, edge_type = build_history_graph_with_types(
            history_pos,
            history_x,
            k_spatial=k_spatial,
            edge_mode=edge_mode,
            edge_attr_mode=edge_attr_mode,
        )
        out = model(
            x_cat,
            p_cat,
            t_cat,
            edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type if use_edge_types else None,
            burden_context=burden_context,
        )
        n_last = history_x[-1].shape[0]
        sl = slice(x_cat.shape[0] - n_last, x_cat.shape[0])
        dp = out["delta_pos"][sl]
        dh = out["delta_feat"][sl]
        alive_prob = torch.sigmoid(out["alive_logit"][sl])

        pred_pos = history_pos[-1] + dp
        pred_xn = _apply_delta_feat(history_x[-1], dh)
        dst_pos = obs_pos[k + 1]
        dst_alive = obs_alive[k + 1].bool()
        if dst_alive.sum() > 0:
            diff = (pred_pos - dst_pos)[dst_alive]
            maes.append(float(diff.norm(dim=-1).mean().item()))

        pred_raw = _raw_features(pred_xn, mean, std)
        obs_raw = _raw_features(obs_xn[k + 1], mean, std)
        node_ftv = (pred_raw[:, volume_idx].clamp_min(0.0) * alive_prob).sum()
        pred_ftv = node_ftv
        if "graph_ftv_affine" in out:
            affine = out["graph_ftv_affine"].reshape(-1)
            scale = F.softplus(affine[0]).clamp(0.25, 4.0)
            pred_ftv = (scale * node_ftv + affine[1]).clamp_min(0.0)
        if "graph_ftv_residual" in out and graph_ftv_residual_cap_ml > 0.0:
            cap = torch.as_tensor(
                float(graph_ftv_residual_cap_ml),
                device=node_ftv.device,
                dtype=node_ftv.dtype,
            )
            resid = cap * torch.tanh(out["graph_ftv_residual"].reshape(()))
            pred_ftv = (pred_ftv + resid).clamp_min(0.0)
        if "graph_ftv_log" in out:
            pred_ftv = torch.expm1(out["graph_ftv_log"].reshape(())).clamp_min(0.0)
        obs_ftv = obs_raw[:, volume_idx].clamp_min(0.0).sum()
        ftv_err = float((pred_ftv - obs_ftv).detach().item())
        alive_err = float((alive_prob.sum() - dst_alive.float().sum()).detach().item())
        ftv_abs.append(abs(ftv_err))
        ftv_bias.append(ftv_err)
        alive_abs.append(abs(alive_err))
        alive_bias.append(alive_err)
        if k == T - 2:
            t3_ftv_abs = abs(ftv_err)
            t3_ftv_bias = ftv_err
            t3_alive_abs = abs(alive_err)
            t3_alive_bias = alive_err
        history_x.append(pred_xn)
        history_pos.append(pred_pos)
        history_t.append(torch.full((pred_xn.shape[0],), float(k + 1), device=DEVICE))

    return {
        "rollout_mae_mm": float(np.mean(maes)) if maes else float("nan"),
        "rollout_ftv_abs_err_ml": float(np.mean(ftv_abs)) if ftv_abs else float("nan"),
        "rollout_ftv_bias_ml": float(np.mean(ftv_bias)) if ftv_bias else float("nan"),
        "rollout_alive_abs_err": float(np.mean(alive_abs)) if alive_abs else float("nan"),
        "rollout_alive_bias": float(np.mean(alive_bias)) if alive_bias else float("nan"),
        "t3_ftv_abs_err_ml": t3_ftv_abs,
        "t3_ftv_bias_ml": t3_ftv_bias,
        "t3_alive_abs_err": t3_alive_abs,
        "t3_alive_bias": t3_alive_bias,
    }


@torch.no_grad()
def rollout_eval_mae(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: Tensor,
    std: Tensor,
    k_spatial: int,
    edge_mode: str,
    edge_attr_mode: str,
    use_edge_types: bool = False,
) -> float:
    return rollout_eval_metrics(
        model,
        s,
        mean,
        std,
        k_spatial,
        edge_mode=edge_mode,
        edge_attr_mode=edge_attr_mode,
        use_edge_types=use_edge_types,
        volume_idx=1,
        graph_ftv_residual_cap_ml=1.0,
    )["rollout_mae_mm"]


@dataclass
class Config:
    graphs_root: str = "datasets/ispy2/graphs_consistent"
    patient_list: str = "reports/all_patients_4visit.txt"
    out_dir: str = "runs/consistent_forecaster_v2"
    pretrained_ckpt: str = "experiments/stage1_forecaster/runs/bio_gated_5fold/fold0/best.pt"
    hidden: int = 64
    num_layers: int = 2
    k_spatial: int = 8
    edge_mode: str = "full"
    edge_attr_mode: str = "none"
    conv_type: str = "lsgc"
    num_relations: int = NUM_RELATIONS
    graph_ftv_readout: bool = False
    graph_pcr_readout: bool = False
    graph_ftv_affine: bool = False
    graph_ftv_embed_affine: bool = False
    graph_ftv_residual: bool = False
    reaction_diffusion_head: bool = False
    core_in_channels: int = 0
    ssl_adapter_dim: int = 0
    ssl_adapter_dropout: float = 0.10
    ssl_adapter_zero_init: bool = True
    use_burden_conditioning: bool = False
    burden_context_hidden: int = 32
    burden_num_experts: int = 3
    burden_adapter_scale: float = 0.5
    predict_core_only: bool = False
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 180
    patience: int = 40
    val_frac: float = 0.2
    seed: int = 42
    freeze_backbone_epochs: int = 30
    fold: int = -1
    folds_parquet: str = "datasets/ispy2/folds.parquet"
    cohort_parquet: str = "datasets/ispy2/cohort.parquet"

    scheduled_sampling: bool = False
    eps_max: float = 0.5
    eps_warmup_epochs: int = 60
    eps_anneal_epochs: int = 60
    random_start: bool = False
    lambda_pos: float = 1.0
    lambda_feat: float = 0.5
    lambda_cloud: float = 1.0
    lambda_cloud_feat: float = 0.0
    lambda_horizon: float = 0.0
    lambda_alive_bce: float = 0.0
    lambda_alive_mass: float = 0.0
    lambda_ftv: float = 0.0
    lambda_graph_ftv: float = 0.0
    lambda_graph_ftv_consistency: float = 0.0
    graph_ftv_residual_cap_ml: float = 1.0
    lambda_graph_ftv_residual_l2: float = 0.0
    ftv_tail_quantile: float = 0.0
    ftv_tail_threshold_ml: float = 0.0
    ftv_tail_weight: float = 1.0
    edge_attr_dropout: float = 0.0
    lambda_pcr_aux: float = 0.0
    lambda_terminal_ftv: float = 0.0
    terminal_ftv_loss_space: str = "log"
    terminal_ftv_horizon_weights: str = ""
    terminal_ftv_tail_quantile: float = 0.0
    terminal_ftv_tail_threshold_ml: float = 0.0
    terminal_ftv_tail_weight: float = 1.0
    lambda_terminal_response_ratio: float = 0.0
    lambda_step_response_ratio: float = 0.0
    lambda_terminal_direction: float = 0.0
    lambda_step_direction: float = 0.0
    lambda_response_diffusion: float = 0.0
    lambda_edge_response: float = 0.0
    lambda_edge_alive: float = 0.0
    lambda_reaction_diffusion_residual: float = 0.0
    reaction_diffusion_capacity_ml: float = 0.0
    reaction_diffusion_treatment_proxy: float = 1.0
    reaction_diffusion_action_context: bool = False
    reaction_diffusion_action_context_mode: str = "components_subtype"
    reaction_diffusion_shuffle_action_context: bool = False
    reaction_diffusion_context_hidden: int = 32
    reaction_diffusion_context_dim: int = 0
    terminal_response_tail_quantile: float = 0.0
    terminal_response_tail_threshold: float = 0.0
    terminal_response_tail_weight: float = 1.0
    response_direction_margin: float = 0.05
    response_shrink_threshold: float = 0.80
    response_grow_threshold: float = 1.20
    start_burden_tail_quantile: float = 0.0
    start_burden_tail_threshold_ml: float = 0.0
    start_burden_tail_weight: float = 1.0
    recon_latent_start: int = -1
    recon_latent_end: int = -1
    lambda_recon_latent_delta: float = 0.0
    lambda_recon_anticopy: float = 0.0
    recon_anticopy_margin: float = 0.02
    recon_change_threshold: float = 0.01
    volume_idx: int = 1
    bio_warmup_epochs: int = 0
    bio_anneal_epochs: int = 0
    selection_ftv_weight: float = 0.0
    selection_alive_weight: float = 0.0
    selection_primary: str = "rollout"
    patient_limit: int = 0


def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pids = [p.strip() for p in Path(cfg.patient_list).read_text().splitlines() if p.strip()]
    if cfg.patient_limit > 0:
        pids = pids[: cfg.patient_limit]
    samples_all: list[ConsistentSample] = []
    for pid in pids:
        s = load_sample(Path(cfg.graphs_root) / f"{pid}.pt")
        if s is not None:
            samples_all.append(s)
    print(f"Loaded {len(samples_all)}/{len(pids)} graphs")
    if not samples_all:
        raise RuntimeError("No samples loaded.")

    if cfg.fold >= 0:
        import pandas as pd

        folds_df = pd.read_parquet(cfg.folds_parquet)
        fold_map = dict(zip(folds_df["patient_id"], folds_df["fold"]))
        val_samples = [s for s in samples_all if fold_map.get(s.patient_id) == cfg.fold]
        train_samples = [s for s in samples_all if fold_map.get(s.patient_id) != cfg.fold]
        print(f"Fold {cfg.fold}/5 Train: {len(train_samples)} Val: {len(val_samples)}")
    else:
        random.shuffle(samples_all)
        n_val = max(1, int(len(samples_all) * cfg.val_frac))
        val_samples = samples_all[:n_val]
        train_samples = samples_all[n_val:]
        print(f"Train: {len(train_samples)} Val: {len(val_samples)}")

    mean, std = compute_stats(train_samples)
    mean, std = mean.to(DEVICE), std.to(DEVICE)
    in_ch = samples_all[0].x.shape[1]

    ckpt_path = Path(cfg.pretrained_ckpt)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        s1_in_ch = int(ck["state_dict"]["embed.weight"].shape[1])
        print(f"Pretrained checkpoint in_ch={s1_in_ch} (target in_ch={in_ch})")
    else:
        ck = None
        s1_in_ch = -1
        print("No pretrained checkpoint found — training from scratch")

    core_in_ch = int(cfg.core_in_channels) if cfg.core_in_channels > 0 else in_ch
    feat_out_dim = core_in_ch if cfg.predict_core_only else in_ch
    rd_action_meta: dict[str, dict[str, Any]] = {}
    rd_shuffle_meta: dict[str, dict[str, Any]] = {}
    rd_context_dim = 0
    if cfg.reaction_diffusion_action_context:
        rd_context_dim = _rd_action_context_dim(cfg.reaction_diffusion_action_context_mode)
        cfg.reaction_diffusion_context_dim = rd_context_dim
        rd_action_meta = _load_rd_action_meta(cfg.cohort_parquet)
        if cfg.reaction_diffusion_shuffle_action_context:
            rd_shuffle_meta = _shuffle_rd_action_meta(rd_action_meta, seed=cfg.seed + 7919)
        print(
            "Reaction-diffusion action context enabled: "
            f"mode={cfg.reaction_diffusion_action_context_mode} "
            f"dim={rd_context_dim} shuffled={int(cfg.reaction_diffusion_shuffle_action_context)} "
            f"patients={len(rd_action_meta)}"
        )

    conv_kwargs = {
        "use_edge_gating": True,
        "edge_attr_dim": edge_attr_dim(cfg.edge_attr_mode),
    }
    if cfg.conv_type == "relational":
        conv_kwargs["num_relations"] = int(cfg.num_relations)
    model = LSGCForecaster(
        in_channels=in_ch,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        feat_out_dim=feat_out_dim,
        use_delta_t=True,
        conv_type=cfg.conv_type,
        graph_ftv_readout=cfg.graph_ftv_readout,
        graph_pcr_readout=cfg.graph_pcr_readout,
        graph_ftv_affine=cfg.graph_ftv_affine,
        graph_ftv_embed_affine=cfg.graph_ftv_embed_affine,
        graph_ftv_residual=cfg.graph_ftv_residual,
        reaction_diffusion_head=cfg.reaction_diffusion_head,
        reaction_diffusion_context_dim=rd_context_dim,
        reaction_diffusion_context_hidden=cfg.reaction_diffusion_context_hidden,
        core_in_channels=core_in_ch,
        ssl_adapter_dim=cfg.ssl_adapter_dim,
        ssl_adapter_dropout=cfg.ssl_adapter_dropout,
        ssl_adapter_zero_init=cfg.ssl_adapter_zero_init,
        burden_context_dim=5 if cfg.use_burden_conditioning else 0,
        burden_context_hidden=cfg.burden_context_hidden,
        burden_num_experts=cfg.burden_num_experts,
        burden_adapter_scale=cfg.burden_adapter_scale,
        **conv_kwargs,
    ).to(DEVICE)

    if ck is not None:
        sd = model.state_dict()
        loaded = 0
        skipped = 0
        for k, v in ck["state_dict"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(sd)
        print(f"Warm-start loaded {loaded} tensors; skipped {skipped} shape-mismatched/missing tensors")
    else:
        print("Skipping warm-start (dim mismatch or missing checkpoint)")

    def _set_backbone_grad(requires_grad: bool):
        for name, p in model.named_parameters():
            if name.startswith("embed") or name.startswith("convs"):
                p.requires_grad_(requires_grad)

    _set_backbone_grad(False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best_val = float("inf")
    patience_cnt = 0
    log: list[dict] = []
    rng = random.Random(cfg.seed + 123)
    pcr_map = _load_pcr_map(cfg.cohort_parquet) if cfg.lambda_pcr_aux > 0.0 else {}
    ftv_tail_threshold_ml = float(cfg.ftv_tail_threshold_ml)
    if ftv_tail_threshold_ml <= 0.0 and cfg.ftv_tail_weight > 1.0:
        ftv_tail_threshold_ml = _compute_train_ftv_tail_threshold(
            train_samples,
            volume_idx=cfg.volume_idx,
            quantile=cfg.ftv_tail_quantile,
        )
    if ftv_tail_threshold_ml > 0.0 and cfg.ftv_tail_weight > 1.0:
        print(
            f"FTV tail weighting enabled: threshold={ftv_tail_threshold_ml:.3f} ml "
            f"weight={cfg.ftv_tail_weight:.3f}"
        )
    cfg.ftv_tail_threshold_ml = ftv_tail_threshold_ml
    terminal_ftv_tail_threshold_ml = float(cfg.terminal_ftv_tail_threshold_ml)
    if (
        terminal_ftv_tail_threshold_ml <= 0.0
        and cfg.terminal_ftv_tail_weight > 1.0
        and cfg.terminal_ftv_tail_quantile > 0.0
    ):
        terminal_ftv_tail_threshold_ml = _compute_train_visit_ftv_tail_threshold(
            train_samples,
            visit_idx=3,
            volume_idx=cfg.volume_idx,
            quantile=cfg.terminal_ftv_tail_quantile,
        )
    if terminal_ftv_tail_threshold_ml > 0.0 and cfg.terminal_ftv_tail_weight > 1.0:
        print(
            f"Terminal FTV tail weighting enabled: threshold={terminal_ftv_tail_threshold_ml:.3f} ml "
            f"weight={cfg.terminal_ftv_tail_weight:.3f}"
        )
    cfg.terminal_ftv_tail_threshold_ml = terminal_ftv_tail_threshold_ml
    terminal_response_tail_threshold = float(cfg.terminal_response_tail_threshold)
    if (
        terminal_response_tail_threshold <= 0.0
        and cfg.terminal_response_tail_weight > 1.0
        and cfg.terminal_response_tail_quantile > 0.0
    ):
        terminal_response_tail_threshold = _compute_train_response_ratio_tail_threshold(
            train_samples,
            volume_idx=cfg.volume_idx,
            quantile=cfg.terminal_response_tail_quantile,
        )
    if terminal_response_tail_threshold > 0.0 and cfg.terminal_response_tail_weight > 1.0:
        print(
            f"Terminal response-tail weighting enabled: threshold={terminal_response_tail_threshold:.3f} "
            f"weight={cfg.terminal_response_tail_weight:.3f}"
        )
    cfg.terminal_response_tail_threshold = terminal_response_tail_threshold
    start_burden_tail_threshold_ml = float(cfg.start_burden_tail_threshold_ml)
    if (
        start_burden_tail_threshold_ml <= 0.0
        and cfg.start_burden_tail_weight > 1.0
        and cfg.start_burden_tail_quantile > 0.0
    ):
        start_burden_tail_threshold_ml = _compute_train_visit_ftv_tail_threshold(
            train_samples,
            visit_idx=0,
            volume_idx=cfg.volume_idx,
            quantile=cfg.start_burden_tail_quantile,
        )
    if start_burden_tail_threshold_ml > 0.0 and cfg.start_burden_tail_weight > 1.0:
        print(
            f"Start-burden tail weighting enabled: threshold={start_burden_tail_threshold_ml:.3f} ml "
            f"weight={cfg.start_burden_tail_weight:.3f}"
        )
    cfg.start_burden_tail_threshold_ml = start_burden_tail_threshold_ml

    for epoch in range(1, cfg.epochs + 1):
        if epoch == cfg.freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.3, weight_decay=cfg.weight_decay)
            print(f"[Epoch {epoch}] Unfreezing backbone, lr={cfg.lr * 0.3:.1e}")

        eps = (
            _epsilon_schedule(epoch, cfg.eps_warmup_epochs, cfg.eps_anneal_epochs, cfg.eps_max)
            if cfg.scheduled_sampling
            else 0.0
        )
        bio_scale = _epsilon_schedule(epoch, cfg.bio_warmup_epochs, cfg.bio_anneal_epochs, 1.0)
        lambda_alive_bce = cfg.lambda_alive_bce * bio_scale
        lambda_alive_mass = cfg.lambda_alive_mass * bio_scale
        lambda_ftv = cfg.lambda_ftv * bio_scale
        lambda_graph_ftv = cfg.lambda_graph_ftv * bio_scale
        lambda_graph_ftv_consistency = cfg.lambda_graph_ftv_consistency * bio_scale
        lambda_graph_ftv_residual_l2 = cfg.lambda_graph_ftv_residual_l2 * bio_scale
        lambda_pcr_aux = cfg.lambda_pcr_aux * bio_scale
        lambda_terminal_ftv = cfg.lambda_terminal_ftv * bio_scale
        lambda_terminal_response_ratio = cfg.lambda_terminal_response_ratio * bio_scale
        lambda_step_response_ratio = cfg.lambda_step_response_ratio * bio_scale
        lambda_terminal_direction = cfg.lambda_terminal_direction * bio_scale
        lambda_step_direction = cfg.lambda_step_direction * bio_scale
        lambda_response_diffusion = cfg.lambda_response_diffusion * bio_scale
        lambda_edge_response = cfg.lambda_edge_response * bio_scale
        lambda_edge_alive = cfg.lambda_edge_alive * bio_scale
        lambda_reaction_diffusion_residual = cfg.lambda_reaction_diffusion_residual * bio_scale
        lambda_recon_latent_delta = cfg.lambda_recon_latent_delta * bio_scale
        lambda_recon_anticopy = cfg.lambda_recon_anticopy * bio_scale

        model.train()
        random.shuffle(train_samples)
        tr_loss, tr_step_mae, tr_selffed = [], [], []
        tr_bio: dict[str, list[float]] = {}
        for s in train_samples:
            optimizer.zero_grad()
            loss, info = curriculum_forward_patient(
                model,
                s,
                mean,
                std,
                epsilon=eps,
                rng=rng,
                k_spatial=cfg.k_spatial,
                edge_mode=cfg.edge_mode,
                edge_attr_mode=cfg.edge_attr_mode,
                use_edge_types=cfg.conv_type == "relational",
                lambda_pos=cfg.lambda_pos,
                lambda_feat=cfg.lambda_feat,
                lambda_cloud=cfg.lambda_cloud,
                lambda_cloud_feat=cfg.lambda_cloud_feat,
                lambda_horizon=cfg.lambda_horizon,
                lambda_alive_bce=lambda_alive_bce,
                lambda_alive_mass=lambda_alive_mass,
                lambda_ftv=lambda_ftv,
                lambda_graph_ftv=lambda_graph_ftv,
                lambda_graph_ftv_consistency=lambda_graph_ftv_consistency,
                graph_ftv_residual_cap_ml=cfg.graph_ftv_residual_cap_ml,
                lambda_graph_ftv_residual_l2=lambda_graph_ftv_residual_l2,
                ftv_tail_threshold_ml=ftv_tail_threshold_ml,
                ftv_tail_weight=cfg.ftv_tail_weight,
                edge_attr_dropout=cfg.edge_attr_dropout,
                lambda_pcr_aux=lambda_pcr_aux,
                pcr_target=pcr_map.get(s.patient_id),
                lambda_terminal_ftv=lambda_terminal_ftv,
                terminal_ftv_loss_space=cfg.terminal_ftv_loss_space,
                terminal_ftv_horizon_weights=cfg.terminal_ftv_horizon_weights,
                terminal_ftv_tail_threshold_ml=terminal_ftv_tail_threshold_ml,
                terminal_ftv_tail_weight=cfg.terminal_ftv_tail_weight,
                lambda_terminal_response_ratio=lambda_terminal_response_ratio,
                lambda_step_response_ratio=lambda_step_response_ratio,
                lambda_terminal_direction=lambda_terminal_direction,
                lambda_step_direction=lambda_step_direction,
                lambda_response_diffusion=lambda_response_diffusion,
                lambda_edge_response=lambda_edge_response,
                lambda_edge_alive=lambda_edge_alive,
                lambda_reaction_diffusion_residual=lambda_reaction_diffusion_residual,
                reaction_diffusion_capacity_ml=cfg.reaction_diffusion_capacity_ml,
                reaction_diffusion_treatment_proxy=cfg.reaction_diffusion_treatment_proxy,
                reaction_diffusion_action_context=cfg.reaction_diffusion_action_context,
                reaction_diffusion_action_context_mode=cfg.reaction_diffusion_action_context_mode,
                reaction_diffusion_action_meta=rd_action_meta,
                reaction_diffusion_shuffle_meta=rd_shuffle_meta if cfg.reaction_diffusion_shuffle_action_context else None,
                terminal_response_tail_threshold=terminal_response_tail_threshold,
                terminal_response_tail_weight=cfg.terminal_response_tail_weight,
                response_direction_margin=cfg.response_direction_margin,
                response_shrink_threshold=cfg.response_shrink_threshold,
                response_grow_threshold=cfg.response_grow_threshold,
                start_burden_tail_threshold_ml=start_burden_tail_threshold_ml,
                start_burden_tail_weight=cfg.start_burden_tail_weight,
                recon_latent_start=cfg.recon_latent_start,
                recon_latent_end=cfg.recon_latent_end,
                lambda_recon_latent_delta=lambda_recon_latent_delta,
                lambda_recon_anticopy=lambda_recon_anticopy,
                recon_anticopy_margin=cfg.recon_anticopy_margin,
                recon_change_threshold=cfg.recon_change_threshold,
                volume_idx=cfg.volume_idx,
                random_start=cfg.random_start,
            )
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tr_loss.append(loss.item())
            if np.isfinite(info["mean_step_mae"]):
                tr_step_mae.append(info["mean_step_mae"])
            tr_selffed.append(info["n_self_fed"])
            for key, val in info.items():
                if key.startswith("bio_") and np.isfinite(val):
                    tr_bio.setdefault(key, []).append(float(val))
                if key.startswith("pcr_") and np.isfinite(val):
                    tr_bio.setdefault(key, []).append(float(val))
                if key.startswith("terminal_") and np.isfinite(val):
                    tr_bio.setdefault(key, []).append(float(val))
                if key.startswith("reaction_diffusion_") and np.isfinite(val):
                    tr_bio.setdefault(key, []).append(float(val))
                if key.startswith("patch_") and np.isfinite(val):
                    tr_bio.setdefault(key, []).append(float(val))

        model.eval()
        val_loss = []
        val_metrics: dict[str, list[float]] = {}
        with torch.no_grad():
            for s in val_samples:
                l, _ = curriculum_forward_patient(
                    model,
                    s,
                    mean,
                    std,
                    epsilon=0.0,
                    rng=rng,
                    k_spatial=cfg.k_spatial,
                    edge_mode=cfg.edge_mode,
                    edge_attr_mode=cfg.edge_attr_mode,
                    use_edge_types=cfg.conv_type == "relational",
                    lambda_pos=cfg.lambda_pos,
                    lambda_feat=cfg.lambda_feat,
                    lambda_cloud=cfg.lambda_cloud,
                    lambda_cloud_feat=cfg.lambda_cloud_feat,
                    lambda_horizon=cfg.lambda_horizon,
                    lambda_alive_bce=lambda_alive_bce,
                    lambda_alive_mass=lambda_alive_mass,
                    lambda_ftv=lambda_ftv,
                    lambda_graph_ftv=lambda_graph_ftv,
                    lambda_graph_ftv_consistency=lambda_graph_ftv_consistency,
                    graph_ftv_residual_cap_ml=cfg.graph_ftv_residual_cap_ml,
                    lambda_graph_ftv_residual_l2=lambda_graph_ftv_residual_l2,
                    ftv_tail_threshold_ml=ftv_tail_threshold_ml,
                    ftv_tail_weight=cfg.ftv_tail_weight,
                    edge_attr_dropout=0.0,
                    lambda_pcr_aux=lambda_pcr_aux,
                    pcr_target=pcr_map.get(s.patient_id),
                    lambda_terminal_ftv=lambda_terminal_ftv,
                    terminal_ftv_loss_space=cfg.terminal_ftv_loss_space,
                    terminal_ftv_horizon_weights=cfg.terminal_ftv_horizon_weights,
                    terminal_ftv_tail_threshold_ml=terminal_ftv_tail_threshold_ml,
                    terminal_ftv_tail_weight=cfg.terminal_ftv_tail_weight,
                    lambda_terminal_response_ratio=lambda_terminal_response_ratio,
                    lambda_step_response_ratio=lambda_step_response_ratio,
                lambda_terminal_direction=lambda_terminal_direction,
                lambda_step_direction=lambda_step_direction,
                lambda_response_diffusion=lambda_response_diffusion,
                lambda_edge_response=lambda_edge_response,
                lambda_edge_alive=lambda_edge_alive,
                lambda_reaction_diffusion_residual=lambda_reaction_diffusion_residual,
                    reaction_diffusion_capacity_ml=cfg.reaction_diffusion_capacity_ml,
                    reaction_diffusion_treatment_proxy=cfg.reaction_diffusion_treatment_proxy,
                    reaction_diffusion_action_context=cfg.reaction_diffusion_action_context,
                    reaction_diffusion_action_context_mode=cfg.reaction_diffusion_action_context_mode,
                    reaction_diffusion_action_meta=rd_action_meta,
                    reaction_diffusion_shuffle_meta=rd_shuffle_meta if cfg.reaction_diffusion_shuffle_action_context else None,
                    terminal_response_tail_threshold=terminal_response_tail_threshold,
                    terminal_response_tail_weight=cfg.terminal_response_tail_weight,
                    response_direction_margin=cfg.response_direction_margin,
                    response_shrink_threshold=cfg.response_shrink_threshold,
                    response_grow_threshold=cfg.response_grow_threshold,
                    start_burden_tail_threshold_ml=start_burden_tail_threshold_ml,
                    start_burden_tail_weight=cfg.start_burden_tail_weight,
                    recon_latent_start=cfg.recon_latent_start,
                    recon_latent_end=cfg.recon_latent_end,
                    lambda_recon_latent_delta=lambda_recon_latent_delta,
                    lambda_recon_anticopy=lambda_recon_anticopy,
                    recon_anticopy_margin=cfg.recon_anticopy_margin,
                    recon_change_threshold=cfg.recon_change_threshold,
                    volume_idx=cfg.volume_idx,
                    random_start=False,
                )
                val_loss.append(l.item())
                metrics = rollout_eval_metrics(
                    model,
                    s,
                    mean,
                    std,
                    cfg.k_spatial,
                    edge_mode=cfg.edge_mode,
                    edge_attr_mode=cfg.edge_attr_mode,
                    use_edge_types=cfg.conv_type == "relational",
                    volume_idx=cfg.volume_idx,
                    graph_ftv_residual_cap_ml=cfg.graph_ftv_residual_cap_ml,
                )
                for key, val in metrics.items():
                    if np.isfinite(val):
                        val_metrics.setdefault(key, []).append(float(val))

        tr_l = float(np.mean(tr_loss)) if tr_loss else float("nan")
        vl_l = float(np.mean(val_loss)) if val_loss else float("nan")
        tr_m = float(np.mean(tr_step_mae)) if tr_step_mae else float("nan")
        val_means = {
            key: float(np.mean(vals)) if vals else float("nan")
            for key, vals in val_metrics.items()
        }
        train_bio_means = {
            f"train_{key}": float(np.mean(vals)) if vals else float("nan")
            for key, vals in tr_bio.items()
        }
        vl_roll = val_means.get("rollout_mae_mm", float("nan"))
        tr_sf = float(np.mean(tr_selffed)) if tr_selffed else 0.0

        log_row = {
            "epoch": epoch,
            "epsilon": eps,
            "bio_scale": bio_scale,
            "lambda_alive_bce_eff": lambda_alive_bce,
            "lambda_alive_mass_eff": lambda_alive_mass,
            "lambda_ftv_eff": lambda_ftv,
            "lambda_graph_ftv_eff": lambda_graph_ftv,
            "lambda_graph_ftv_consistency_eff": lambda_graph_ftv_consistency,
            "lambda_graph_ftv_residual_l2_eff": lambda_graph_ftv_residual_l2,
            "lambda_pcr_aux_eff": lambda_pcr_aux,
            "lambda_terminal_ftv_eff": lambda_terminal_ftv,
            "lambda_terminal_response_ratio_eff": lambda_terminal_response_ratio,
            "lambda_step_response_ratio_eff": lambda_step_response_ratio,
            "lambda_terminal_direction_eff": lambda_terminal_direction,
            "lambda_step_direction_eff": lambda_step_direction,
            "lambda_response_diffusion_eff": lambda_response_diffusion,
            "lambda_edge_response_eff": lambda_edge_response,
            "lambda_edge_alive_eff": lambda_edge_alive,
            "lambda_reaction_diffusion_residual_eff": lambda_reaction_diffusion_residual,
            "lambda_recon_latent_delta_eff": lambda_recon_latent_delta,
            "lambda_recon_anticopy_eff": lambda_recon_anticopy,
            "recon_latent_start": cfg.recon_latent_start,
            "recon_latent_end": cfg.recon_latent_end,
            "recon_anticopy_margin": cfg.recon_anticopy_margin,
            "recon_change_threshold": cfg.recon_change_threshold,
            "reaction_diffusion_capacity_ml": cfg.reaction_diffusion_capacity_ml,
            "reaction_diffusion_treatment_proxy": cfg.reaction_diffusion_treatment_proxy,
            "reaction_diffusion_action_context": int(cfg.reaction_diffusion_action_context),
            "reaction_diffusion_action_context_mode": cfg.reaction_diffusion_action_context_mode,
            "reaction_diffusion_shuffle_action_context": int(cfg.reaction_diffusion_shuffle_action_context),
            "reaction_diffusion_context_hidden": cfg.reaction_diffusion_context_hidden,
            "terminal_ftv_loss_space": cfg.terminal_ftv_loss_space,
            "terminal_ftv_horizon_weights": cfg.terminal_ftv_horizon_weights,
            "terminal_ftv_tail_threshold_ml": terminal_ftv_tail_threshold_ml,
            "terminal_ftv_tail_weight": cfg.terminal_ftv_tail_weight,
            "terminal_response_tail_threshold": terminal_response_tail_threshold,
            "terminal_response_tail_weight": cfg.terminal_response_tail_weight,
            "start_burden_tail_threshold_ml": start_burden_tail_threshold_ml,
            "start_burden_tail_weight": cfg.start_burden_tail_weight,
            "use_burden_conditioning": int(cfg.use_burden_conditioning),
            "burden_num_experts": cfg.burden_num_experts,
            "ftv_tail_threshold_ml": ftv_tail_threshold_ml,
            "ftv_tail_weight": cfg.ftv_tail_weight,
            "edge_attr_dropout": cfg.edge_attr_dropout,
            "train_loss": tr_l,
            "val_loss": vl_l,
            "train_step_mae_mm": tr_m,
            "val_rollout_mae_mm": vl_roll,
            "train_self_fed_steps_mean": tr_sf,
            **{f"val_{key}": val for key, val in val_means.items()},
            **train_bio_means,
        }
        log.append(log_row)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} eps={eps:.3f} train_loss={tr_l:.4f} val_loss={vl_l:.4f} "
                f"train_step_mae={tr_m:.3f} val_rollout_mae={vl_roll:.3f} "
                f"val_t3_ftv_abs={val_means.get('t3_ftv_abs_err_ml', float('nan')):.3f} "
                f"val_t3_alive_abs={val_means.get('t3_alive_abs_err', float('nan')):.3f} "
                f"graph_ftv={int(cfg.graph_ftv_readout)} "
                f"pcr_aux={int(cfg.graph_pcr_readout)} "
                f"selffed={tr_sf:.2f}"
            )

        # Early-stop on a selected primary metric, with optional biology terms.
        t3_ftv_abs = val_means.get("t3_ftv_abs_err_ml", float("nan"))
        t3_alive_abs = val_means.get("t3_alive_abs_err", float("nan"))
        if cfg.selection_primary == "ftv" and np.isfinite(t3_ftv_abs):
            score = t3_ftv_abs
        elif cfg.selection_primary == "loss" and np.isfinite(vl_l):
            score = vl_l
        else:
            score = vl_roll if np.isfinite(vl_roll) else vl_l
        if np.isfinite(score):
            if cfg.selection_ftv_weight > 0.0 and np.isfinite(t3_ftv_abs):
                score += cfg.selection_ftv_weight * t3_ftv_abs
            if cfg.selection_alive_weight > 0.0 and np.isfinite(t3_alive_abs):
                score += cfg.selection_alive_weight * t3_alive_abs
        if score < best_val:
            best_val = score
            patience_cnt = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": vl_l,
                    "val_rollout_mae_mm": vl_roll,
                    "val_t3_ftv_abs_err_ml": val_means.get("t3_ftv_abs_err_ml", float("nan")),
                    "val_t3_ftv_bias_ml": val_means.get("t3_ftv_bias_ml", float("nan")),
                    "val_t3_alive_abs_err": val_means.get("t3_alive_abs_err", float("nan")),
                    "val_t3_alive_bias": val_means.get("t3_alive_bias", float("nan")),
                    "selection_score": score,
                },
                out_dir / "best.pt",
            )
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print(f"Early stop at epoch {epoch}")
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.cpu(),
            "std": std.cpu(),
            "config": asdict(cfg),
        },
        out_dir / "last.pt",
    )
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"Best val rollout MAE: {best_val:.4f} mm")
    print(f"Outputs: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-root", default="datasets/ispy2/graphs_consistent")
    ap.add_argument("--patient-list", default="reports/all_patients_4visit.txt")
    ap.add_argument("--out-dir", default="runs/consistent_forecaster_v2")
    ap.add_argument(
        "--pretrained",
        default="experiments/stage1_forecaster/runs/bio_gated_5fold/fold0/best.pt",
    )
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=30)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--folds-parquet", default="datasets/ispy2/folds.parquet")
    ap.add_argument("--cohort-parquet", default="datasets/ispy2/cohort.parquet")

    ap.add_argument("--k-spatial", type=int, default=8)
    ap.add_argument(
        "--edge-mode",
        choices=EDGE_MODE_CHOICES,
        default="full",
        help="Dynamic intra-visit edge mode; all non-none modes keep temporal identity links.",
    )
    ap.add_argument(
        "--edge-attr-mode",
        choices=EDGE_ATTR_MODE_CHOICES,
        default="none",
        help="Optional dynamic edge attributes. radial_bio adds tumor-centered geometry and feature-contrast channels.",
    )
    ap.add_argument("--conv-type", choices=("lsgc", "relational"), default="lsgc")
    ap.add_argument("--num-relations", type=int, default=NUM_RELATIONS)
    ap.add_argument("--graph-ftv-readout", action="store_true")
    ap.add_argument("--graph-pcr-readout", action="store_true")
    ap.add_argument("--graph-ftv-affine", action="store_true")
    ap.add_argument("--graph-ftv-embed-affine", action="store_true")
    ap.add_argument("--graph-ftv-residual", action="store_true")
    ap.add_argument("--reaction-diffusion-head", action="store_true")
    ap.add_argument("--core-in-channels", type=int, default=0)
    ap.add_argument("--ssl-adapter-dim", type=int, default=0)
    ap.add_argument("--ssl-adapter-dropout", type=float, default=0.10)
    ap.add_argument("--no-ssl-adapter-zero-init", action="store_true")
    ap.add_argument("--use-burden-conditioning", action="store_true")
    ap.add_argument("--burden-context-hidden", type=int, default=32)
    ap.add_argument("--burden-num-experts", type=int, default=3)
    ap.add_argument("--burden-adapter-scale", type=float, default=0.5)
    ap.add_argument("--predict-core-only", action="store_true")
    ap.add_argument("--lambda-pos", type=float, default=1.0)
    ap.add_argument("--lambda-feat", type=float, default=0.5)
    ap.add_argument("--lambda-cloud", type=float, default=1.0)
    ap.add_argument("--lambda-cloud-feat", type=float, default=0.0)
    ap.add_argument("--lambda-horizon", type=float, default=0.0)
    ap.add_argument("--lambda-alive-bce", type=float, default=0.0)
    ap.add_argument("--lambda-alive-mass", type=float, default=0.0)
    ap.add_argument("--lambda-ftv", type=float, default=0.0)
    ap.add_argument("--lambda-graph-ftv", type=float, default=0.0)
    ap.add_argument("--lambda-graph-ftv-consistency", type=float, default=0.0)
    ap.add_argument("--graph-ftv-residual-cap-ml", type=float, default=1.0)
    ap.add_argument("--lambda-graph-ftv-residual-l2", type=float, default=0.0)
    ap.add_argument("--ftv-tail-quantile", type=float, default=0.0)
    ap.add_argument("--ftv-tail-threshold-ml", type=float, default=0.0)
    ap.add_argument("--ftv-tail-weight", type=float, default=1.0)
    ap.add_argument("--edge-attr-dropout", type=float, default=0.0)
    ap.add_argument("--lambda-pcr-aux", type=float, default=0.0)
    ap.add_argument("--lambda-terminal-ftv", type=float, default=0.0)
    ap.add_argument("--terminal-ftv-loss-space", choices=("log", "ml"), default="log")
    ap.add_argument(
        "--terminal-ftv-horizon-weights",
        default="",
        help="Comma-separated T1,T2,T3 weights for autoregressive FTV loss. Empty means T3 only.",
    )
    ap.add_argument("--terminal-ftv-tail-quantile", type=float, default=0.0)
    ap.add_argument("--terminal-ftv-tail-threshold-ml", type=float, default=0.0)
    ap.add_argument("--terminal-ftv-tail-weight", type=float, default=1.0)
    ap.add_argument(
        "--lambda-terminal-response-ratio",
        type=float,
        default=0.0,
        help="Optional autoregressive T3/T0 log-response-ratio loss, applied inside terminal FTV rollout.",
    )
    ap.add_argument(
        "--terminal-response-tail-quantile",
        type=float,
        default=0.0,
        help="Optional lower quantile of training T3/T0 response ratio used as the strong-response threshold.",
    )
    ap.add_argument(
        "--terminal-response-tail-threshold",
        type=float,
        default=0.0,
        help="Direct strong-response threshold on observed T3/T0 FTV ratio; values below this are upweighted.",
    )
    ap.add_argument("--terminal-response-tail-weight", type=float, default=1.0)
    ap.add_argument("--lambda-step-response-ratio", type=float, default=0.0)
    ap.add_argument("--lambda-terminal-direction", type=float, default=0.0)
    ap.add_argument("--lambda-step-direction", type=float, default=0.0)
    ap.add_argument("--lambda-response-diffusion", type=float, default=0.0)
    ap.add_argument(
        "--lambda-edge-response",
        type=float,
        default=0.0,
        help="Auxiliary loss matching observed pairwise node response contrast on current graph edges.",
    )
    ap.add_argument(
        "--lambda-edge-alive",
        type=float,
        default=0.0,
        help="Auxiliary loss matching observed pairwise alive-probability contrast on current graph edges.",
    )
    ap.add_argument("--lambda-reaction-diffusion-residual", type=float, default=0.0)
    ap.add_argument(
        "--reaction-diffusion-capacity-ml",
        type=float,
        default=0.0,
        help="Optional fixed per-node carrying capacity for the reaction-diffusion residual; <=0 uses source q95.",
    )
    ap.add_argument("--reaction-diffusion-treatment-proxy", type=float, default=1.0)
    ap.add_argument("--reaction-diffusion-action-context", action="store_true")
    ap.add_argument(
        "--reaction-diffusion-action-context-mode",
        choices=RD_ACTION_CONTEXT_MODES,
        default="components_subtype",
    )
    ap.add_argument("--reaction-diffusion-shuffle-action-context", action="store_true")
    ap.add_argument("--reaction-diffusion-context-hidden", type=int, default=32)
    ap.add_argument("--response-direction-margin", type=float, default=0.05)
    ap.add_argument("--response-shrink-threshold", type=float, default=0.80)
    ap.add_argument("--response-grow-threshold", type=float, default=1.20)
    ap.add_argument(
        "--start-burden-tail-quantile",
        type=float,
        default=0.0,
        help="Optional upper quantile of training T0 FTV used to upweight large-burden starts.",
    )
    ap.add_argument("--start-burden-tail-threshold-ml", type=float, default=0.0)
    ap.add_argument("--start-burden-tail-weight", type=float, default=1.0)
    ap.add_argument(
        "--recon-latent-start",
        type=int,
        default=-1,
        help="Start column for reconstruction-AE latent features. Negative disables reconstruction-aware losses.",
    )
    ap.add_argument(
        "--recon-latent-end",
        type=int,
        default=-1,
        help="End column for reconstruction-AE latent features.",
    )
    ap.add_argument(
        "--lambda-recon-latent-delta",
        type=float,
        default=0.0,
        help="Auxiliary loss on forecasted reconstruction-AE latent change.",
    )
    ap.add_argument(
        "--lambda-recon-anticopy",
        type=float,
        default=0.0,
        help="Ranking loss requiring the predicted latent to beat current-state carry-forward on changed nodes.",
    )
    ap.add_argument("--recon-anticopy-margin", type=float, default=0.02)
    ap.add_argument("--recon-change-threshold", type=float, default=0.01)
    ap.add_argument("--volume-idx", type=int, default=1)
    ap.add_argument("--bio-warmup-epochs", type=int, default=0)
    ap.add_argument("--bio-anneal-epochs", type=int, default=0)
    ap.add_argument("--selection-ftv-weight", type=float, default=0.0)
    ap.add_argument("--selection-alive-weight", type=float, default=0.0)
    ap.add_argument("--selection-primary", choices=("rollout", "loss", "ftv"), default="rollout")
    ap.add_argument("--patient-limit", type=int, default=0)

    ap.add_argument("--scheduled-sampling", action="store_true")
    ap.add_argument("--eps-max", type=float, default=0.5)
    ap.add_argument("--eps-warmup-epochs", type=int, default=60)
    ap.add_argument("--eps-anneal-epochs", type=int, default=60)
    ap.add_argument("--random-start", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        graphs_root=args.graphs_root,
        patient_list=args.patient_list,
        out_dir=args.out_dir,
        pretrained_ckpt=args.pretrained,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        num_layers=args.num_layers,
        patience=args.patience,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        fold=args.fold,
        folds_parquet=args.folds_parquet,
        cohort_parquet=args.cohort_parquet,
        k_spatial=args.k_spatial,
        edge_mode=args.edge_mode,
        edge_attr_mode=args.edge_attr_mode,
        conv_type=args.conv_type,
        num_relations=args.num_relations,
        graph_ftv_readout=args.graph_ftv_readout,
        graph_pcr_readout=args.graph_pcr_readout,
        graph_ftv_affine=args.graph_ftv_affine,
        graph_ftv_embed_affine=args.graph_ftv_embed_affine,
        graph_ftv_residual=args.graph_ftv_residual,
        reaction_diffusion_head=args.reaction_diffusion_head,
        core_in_channels=args.core_in_channels,
        ssl_adapter_dim=args.ssl_adapter_dim,
        ssl_adapter_dropout=args.ssl_adapter_dropout,
        ssl_adapter_zero_init=not args.no_ssl_adapter_zero_init,
        use_burden_conditioning=args.use_burden_conditioning,
        burden_context_hidden=args.burden_context_hidden,
        burden_num_experts=args.burden_num_experts,
        burden_adapter_scale=args.burden_adapter_scale,
        predict_core_only=args.predict_core_only,
        lambda_pos=args.lambda_pos,
        lambda_feat=args.lambda_feat,
        lambda_cloud=args.lambda_cloud,
        lambda_cloud_feat=args.lambda_cloud_feat,
        lambda_horizon=args.lambda_horizon,
        lambda_alive_bce=args.lambda_alive_bce,
        lambda_alive_mass=args.lambda_alive_mass,
        lambda_ftv=args.lambda_ftv,
        lambda_graph_ftv=args.lambda_graph_ftv,
        lambda_graph_ftv_consistency=args.lambda_graph_ftv_consistency,
        graph_ftv_residual_cap_ml=args.graph_ftv_residual_cap_ml,
        lambda_graph_ftv_residual_l2=args.lambda_graph_ftv_residual_l2,
        ftv_tail_quantile=args.ftv_tail_quantile,
        ftv_tail_threshold_ml=args.ftv_tail_threshold_ml,
        ftv_tail_weight=args.ftv_tail_weight,
        edge_attr_dropout=args.edge_attr_dropout,
        lambda_pcr_aux=args.lambda_pcr_aux,
        lambda_terminal_ftv=args.lambda_terminal_ftv,
        terminal_ftv_loss_space=args.terminal_ftv_loss_space,
        terminal_ftv_horizon_weights=args.terminal_ftv_horizon_weights,
        terminal_ftv_tail_quantile=args.terminal_ftv_tail_quantile,
        terminal_ftv_tail_threshold_ml=args.terminal_ftv_tail_threshold_ml,
        terminal_ftv_tail_weight=args.terminal_ftv_tail_weight,
        lambda_terminal_response_ratio=args.lambda_terminal_response_ratio,
        lambda_step_response_ratio=args.lambda_step_response_ratio,
        lambda_terminal_direction=args.lambda_terminal_direction,
        lambda_step_direction=args.lambda_step_direction,
        lambda_response_diffusion=args.lambda_response_diffusion,
        lambda_edge_response=args.lambda_edge_response,
        lambda_edge_alive=args.lambda_edge_alive,
        lambda_reaction_diffusion_residual=args.lambda_reaction_diffusion_residual,
        reaction_diffusion_capacity_ml=args.reaction_diffusion_capacity_ml,
        reaction_diffusion_treatment_proxy=args.reaction_diffusion_treatment_proxy,
        reaction_diffusion_action_context=args.reaction_diffusion_action_context,
        reaction_diffusion_action_context_mode=args.reaction_diffusion_action_context_mode,
        reaction_diffusion_shuffle_action_context=args.reaction_diffusion_shuffle_action_context,
        reaction_diffusion_context_hidden=args.reaction_diffusion_context_hidden,
        terminal_response_tail_quantile=args.terminal_response_tail_quantile,
        terminal_response_tail_threshold=args.terminal_response_tail_threshold,
        terminal_response_tail_weight=args.terminal_response_tail_weight,
        response_direction_margin=args.response_direction_margin,
        response_shrink_threshold=args.response_shrink_threshold,
        response_grow_threshold=args.response_grow_threshold,
        start_burden_tail_quantile=args.start_burden_tail_quantile,
        start_burden_tail_threshold_ml=args.start_burden_tail_threshold_ml,
        start_burden_tail_weight=args.start_burden_tail_weight,
        recon_latent_start=args.recon_latent_start,
        recon_latent_end=args.recon_latent_end,
        lambda_recon_latent_delta=args.lambda_recon_latent_delta,
        lambda_recon_anticopy=args.lambda_recon_anticopy,
        recon_anticopy_margin=args.recon_anticopy_margin,
        recon_change_threshold=args.recon_change_threshold,
        volume_idx=args.volume_idx,
        bio_warmup_epochs=args.bio_warmup_epochs,
        bio_anneal_epochs=args.bio_anneal_epochs,
        selection_ftv_weight=args.selection_ftv_weight,
        selection_alive_weight=args.selection_alive_weight,
        selection_primary=args.selection_primary,
        patient_limit=args.patient_limit,
        scheduled_sampling=args.scheduled_sampling,
        eps_max=args.eps_max,
        eps_warmup_epochs=args.eps_warmup_epochs,
        eps_anneal_epochs=args.eps_anneal_epochs,
        random_start=args.random_start,
    )
    train(cfg)


if __name__ == "__main__":
    main()
