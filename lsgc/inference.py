"""Checkpoint loading and deterministic rollout for the NCA paper model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from experiments.stage1_forecaster.edge_modes import (
    NUM_RELATIONS,
    build_history_graph_with_types,
    edge_attr_dim,
)
from lsgc.forecaster import LSGCForecaster

VISITS = ("T0", "T1", "T2", "T3")


def load_model(path: Path) -> tuple[LSGCForecaster, torch.Tensor, torch.Tensor]:
    """Load one fold checkpoint using the architecture stored in that file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    in_channels = int(mean.shape[0])
    conv_type = str(config.get("conv_type", "lsgc"))
    conv_kwargs: dict[str, Any] = {
        "use_edge_gating": True,
        "edge_attr_dim": edge_attr_dim(config.get("edge_attr_mode", "none")),
    }
    if conv_type == "relational":
        conv_kwargs["num_relations"] = int(config.get("num_relations", NUM_RELATIONS))

    core_in_channels = int(config.get("core_in_channels", 0) or in_channels)
    feat_out_dim = core_in_channels if bool(config.get("predict_core_only", False)) else in_channels
    model = LSGCForecaster(
        in_channels=in_channels,
        hidden=int(config["hidden"]),
        num_layers=int(config["num_layers"]),
        feat_out_dim=feat_out_dim,
        use_delta_t=True,
        conv_type=conv_type,
        graph_ftv_readout=bool(config.get("graph_ftv_readout", False)),
        graph_pcr_readout=bool(config.get("graph_pcr_readout", False)),
        graph_ftv_affine=bool(config.get("graph_ftv_affine", False)),
        graph_ftv_embed_affine=bool(config.get("graph_ftv_embed_affine", False)),
        graph_ftv_residual=bool(config.get("graph_ftv_residual", False)),
        reaction_diffusion_head=bool(config.get("reaction_diffusion_head", False)),
        reaction_diffusion_context_dim=int(config.get("reaction_diffusion_context_dim", 0) or 0),
        reaction_diffusion_context_hidden=int(config.get("reaction_diffusion_context_hidden", 32) or 32),
        core_in_channels=core_in_channels,
        ssl_adapter_dim=int(config.get("ssl_adapter_dim", 0) or 0),
        ssl_adapter_dropout=float(config.get("ssl_adapter_dropout", 0.10)),
        ssl_adapter_zero_init=bool(config.get("ssl_adapter_zero_init", True)),
        burden_context_dim=(
            5
            if bool(config.get("use_burden_conditioning", False))
            else int(config.get("burden_context_dim", 0) or 0)
        ),
        burden_context_hidden=int(config.get("burden_context_hidden", 32) or 32),
        burden_num_experts=int(config.get("burden_num_experts", 0) or 0),
        burden_adapter_scale=float(config.get("burden_adapter_scale", 0.5)),
        **conv_kwargs,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model._graph_ftv_residual_cap_ml = float(config.get("graph_ftv_residual_cap_ml", 1.0))
    model.eval()
    return model, mean, std


def _apply_delta_feat(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if delta.shape[1] == x.shape[1]:
        return x + delta
    if delta.shape[1] > x.shape[1]:
        raise ValueError(f"delta feature width {delta.shape[1]} exceeds state width {x.shape[1]}")
    out = x.clone()
    out[:, : delta.shape[1]] = out[:, : delta.shape[1]] + delta
    return out


def _burden_context_from_visit(
    x_normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    volume_idx: int = 1,
    alive: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = x_normalized * std + mean
    volume = raw[:, volume_idx].clamp_min(0.0)
    alive_weight = (
        torch.ones_like(volume)
        if alive is None
        else alive.to(device=x_normalized.device, dtype=x_normalized.dtype).reshape(-1)
    )
    ftv = (volume * alive_weight).sum()
    alive_count = alive_weight.sum()
    node_count = torch.as_tensor(float(x_normalized.shape[0]), device=x_normalized.device, dtype=x_normalized.dtype)
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


@torch.no_grad()
def run_inference(
    model: LSGCForecaster,
    mean: torch.Tensor,
    std: torch.Tensor,
    graph: dict,
    k_spatial: int = 8,
    edge_mode: str = "full",
    edge_attr_mode: str = "none",
) -> dict:
    """Return teacher-forced per-visit node changes for one graph."""
    x = (graph["x"].float() - mean) / std
    pos = graph["pos"].float()
    t = graph["t"].float()
    offsets = graph["visit_offsets"].tolist()
    history_x = [x[int(offsets[v]): int(offsets[v + 1])] for v in range(len(offsets) - 1)]
    history_pos = [pos[int(offsets[v]): int(offsets[v + 1])] for v in range(len(offsets) - 1)]
    burden_context = None
    if getattr(model, "burden_context_dim", 0):
        alive0 = graph["alive"][int(offsets[0]): int(offsets[1])].float()
        burden_context = _burden_context_from_visit(history_x[0], mean, std, alive=alive0)
    edge_index, edge_attr, edge_type = build_history_graph_with_types(
        history_pos,
        history_x,
        k_spatial=k_spatial,
        edge_mode=edge_mode,
        edge_attr_mode=edge_attr_mode,
    )
    output = model(
        x,
        pos,
        t,
        edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type if getattr(model, "conv_type", "lsgc") == "relational" else None,
        burden_context=burden_context,
    )

    n_supervoxels = graph["n_supervoxels"]
    result: dict[str, Any] = {}
    for visit_index, visit in enumerate(VISITS[:-1]):
        node_slice = slice(int(offsets[visit_index]), int(offsets[visit_index + 1]))
        result[f"{visit}_dpos"] = output["delta_pos"][node_slice].numpy()
        result[f"{visit}_dfeat"] = output["delta_feat"][node_slice].numpy()
    predicted_positions = {VISITS[0]: pos[:n_supervoxels].numpy()}
    for visit_index, visit in enumerate(VISITS[:-1]):
        next_visit = VISITS[visit_index + 1]
        predicted_positions[next_visit] = predicted_positions[visit] + result[f"{visit}_dpos"]
    result["pos_pred"] = predicted_positions
    result["pos_actual"] = {
        visit: pos[int(offsets[v]): int(offsets[v + 1])].numpy()
        for v, visit in enumerate(VISITS)
    }
    result["alive"] = {
        visit: graph["alive"][int(offsets[v]): int(offsets[v + 1])].numpy().astype(bool)
        for v, visit in enumerate(VISITS)
    }
    return result


@torch.no_grad()
def rollout_from_visit(
    model: LSGCForecaster,
    mean: torch.Tensor,
    std: torch.Tensor,
    graph: dict,
    start_visit: int = 0,
    k_spatial: int = 8,
    edge_mode: str = "full",
    edge_attr_mode: str = "none",
) -> list[dict[str, Any]]:
    """Autoregressively forecast all visits after ``start_visit``."""
    if start_visit < 0 or start_visit >= len(VISITS) - 1:
        raise ValueError(f"start_visit must be in [0, {len(VISITS) - 2}]")

    x = (graph["x"].float() - mean) / std
    pos = graph["pos"].float()
    offsets = graph["visit_offsets"].tolist()
    n_visits = len(offsets) - 1
    history_x: list[torch.Tensor] = []
    history_pos: list[torch.Tensor] = []
    history_t: list[torch.Tensor] = []
    for visit_index in range(start_visit + 1):
        node_slice = slice(int(offsets[visit_index]), int(offsets[visit_index + 1]))
        x_visit = x[node_slice].clone()
        pos_visit = pos[node_slice].clone()
        history_x.append(x_visit)
        history_pos.append(pos_visit)
        history_t.append(torch.full((x_visit.shape[0],), float(visit_index), dtype=x.dtype))

    burden_context = None
    if getattr(model, "burden_context_dim", 0):
        start_slice = slice(int(offsets[start_visit]), int(offsets[start_visit + 1]))
        burden_context = _burden_context_from_visit(
            history_x[-1], mean, std, alive=graph["alive"][start_slice].float()
        )

    rows: list[dict[str, Any]] = []
    for predicted_visit in range(start_visit + 1, n_visits):
        x_cat = torch.cat(history_x, dim=0)
        pos_cat = torch.cat(history_pos, dim=0)
        t_cat = torch.cat(history_t, dim=0)
        edge_index, edge_attr, edge_type = build_history_graph_with_types(
            history_pos,
            history_x,
            k_spatial=k_spatial,
            edge_mode=edge_mode,
            edge_attr_mode=edge_attr_mode,
        )
        output = model(
            x_cat,
            pos_cat,
            t_cat,
            edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type if getattr(model, "conv_type", "lsgc") == "relational" else None,
            burden_context=burden_context,
        )
        n_last = int(history_x[-1].shape[0])
        last_slice = slice(x_cat.shape[0] - n_last, x_cat.shape[0])
        delta_pos = output["delta_pos"][last_slice]
        delta_feat = output["delta_feat"][last_slice]
        alive_prob = torch.sigmoid(output["alive_logit"][last_slice])
        x_next = _apply_delta_feat(history_x[-1], delta_feat)
        pos_next = history_pos[-1] + delta_pos
        x_next_raw = x_next * std + mean
        node_ftv = (x_next_raw[:, 1].clamp_min(0.0) * alive_prob).sum()
        graph_ftv = None
        if "graph_ftv_affine" in output:
            affine = output["graph_ftv_affine"].reshape(-1)
            scale = torch.nn.functional.softplus(affine[0]).clamp(0.25, 4.0)
            graph_ftv = float((scale * node_ftv + affine[1]).clamp_min(0.0).cpu().item())
        if "graph_ftv_residual" in output:
            base_ftv = node_ftv
            if "graph_ftv_affine" in output:
                affine = output["graph_ftv_affine"].reshape(-1)
                scale = torch.nn.functional.softplus(affine[0]).clamp(0.25, 4.0)
                base_ftv = (scale * node_ftv + affine[1]).clamp_min(0.0)
            cap_ml = float(getattr(model, "_graph_ftv_residual_cap_ml", 1.0))
            residual = (
                torch.as_tensor(cap_ml, device=node_ftv.device, dtype=node_ftv.dtype)
                * torch.tanh(output["graph_ftv_residual"].reshape(()))
                if cap_ml > 0.0
                else output["graph_ftv_residual"].reshape(())
            )
            graph_ftv = float((base_ftv + residual).clamp_min(0.0).cpu().item())
        if "graph_ftv_log" in output:
            graph_ftv = float(torch.expm1(output["graph_ftv_log"].reshape(())).clamp_min(0.0).cpu().item())

        observed_slice = slice(int(offsets[predicted_visit]), int(offsets[predicted_visit + 1]))
        rows.append(
            {
                "visit_idx": predicted_visit,
                "visit_name": VISITS[predicted_visit],
                "pos_pred": pos_next.cpu().numpy(),
                "x_pred": x_next_raw.cpu().numpy(),
                "alive_prob": alive_prob.cpu().numpy(),
                "graph_ftv_pred_ml": graph_ftv,
                "pos_obs": graph["pos"][observed_slice].float().cpu().numpy(),
                "x_obs": graph["x"][observed_slice].float().cpu().numpy(),
                "alive_obs": graph["alive"][observed_slice].cpu().numpy().astype(bool),
            }
        )
        history_x.append(x_next)
        history_pos.append(pos_next)
        history_t.append(torch.full((x_next.shape[0],), float(predicted_visit), dtype=x.dtype))
    return rows
