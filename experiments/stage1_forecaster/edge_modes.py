"""Dynamic edge construction for consistent graph rollout experiments."""

from __future__ import annotations

import re

import torch
from torch import Tensor


EDGE_MODE_CHOICES = (
    "full",
    "spatial_knn",
    "none",
    "temporal_only",
    "radial_knn",
    "feature_knn",
    "feature_knn_all",
    "feature_knn_pe_ser",
    "feature_knn_volume",
    "ssl_feature_knn",
    "radial_ssl_plus2",
    "radial_ssl_plus4",
    "hybrid_spatial_feature_a25",
    "hybrid_spatial_feature_a50",
    "hybrid_spatial_feature_a75",
    "multi_rel_union",
    "multi_rel_ssl_union",
    "directed_radial_bio",
    "learned_union_bio",
    "crystal_shell_union",
    "local_ssl_radius_knn",
    "crystal_shell_shuffle_control",
    "history_response_knn",
    "spatial_history_union",
    "spatial_local_ssl_history_union",
)

EDGE_ATTR_MODE_CHOICES = (
    "none",
    "radial_geometry",
    "radial_bio",
    "ssl_radial_bio",
)

RELATION_NAMES = (
    "spatial",
    "radial_same_shell",
    "feature_pe_ser",
    "feature_volume",
    "temporal_forward",
    "temporal_backward",
    "radial_inward_to_target",
    "radial_outward_to_target",
    "ssl_embedding",
    "coordination_shell_1",
    "coordination_shell_2",
    "coordination_shell_3",
)
NUM_RELATIONS = len(RELATION_NAMES)


def edge_attr_dim(edge_attr_mode: str = "none") -> int:
    """Return the per-edge attribute width for a named encoding."""
    mode = str(edge_attr_mode)
    if mode == "none":
        return 0
    if mode == "radial_geometry":
        return 9
    if mode == "radial_bio":
        return 12
    if mode == "ssl_radial_bio":
        return 20
    raise ValueError(f"unknown edge_attr_mode={edge_attr_mode!r}")


def _empty_edges(device: torch.device) -> Tensor:
    return torch.zeros((2, 0), dtype=torch.long, device=device)


def _empty_edge_attr(device: torch.device, dim: int) -> Tensor:
    return torch.zeros((0, int(dim)), dtype=torch.float32, device=device)


def _positive_scale(d: Tensor) -> Tensor:
    vals = d[torch.isfinite(d) & (d > 0)]
    if vals.numel() == 0:
        return torch.ones((), dtype=d.dtype, device=d.device)
    return vals.median().clamp_min(1e-6)


def _knn_from_distance(d: Tensor, k: int) -> Tensor:
    n = int(d.shape[0])
    if n <= 1 or k <= 0:
        return _empty_edges(d.device)
    d = d.clone()
    d.fill_diagonal_(float("inf"))
    k_eff = min(int(k), n - 1)
    _, idx = torch.topk(d, k=k_eff, largest=False, dim=1)
    src = torch.arange(n, device=d.device).unsqueeze(1).expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return ei.unique(dim=1)


def _typed_knn_from_distance(d: Tensor, k: int, relation: int) -> tuple[Tensor, Tensor]:
    ei = _knn_from_distance(d, k)
    et = torch.full((ei.shape[1],), int(relation), dtype=torch.long, device=d.device)
    return ei, et


def _directed_radial_edges(pos: Tensor, k: int) -> tuple[Tensor, Tensor]:
    n = int(pos.shape[0])
    if n <= 1 or k <= 0:
        return _empty_edges(pos.device), torch.zeros((0,), dtype=torch.long, device=pos.device)
    radius = pos.norm(dim=1)
    d_rad = (radius.view(-1, 1) - radius.view(1, -1)).abs()
    src_all: list[Tensor] = []
    dst_all: list[Tensor] = []
    rel_all: list[Tensor] = []
    nodes = torch.arange(n, device=pos.device)
    for dst in range(n):
        inward = nodes[radius < radius[dst]]
        outward = nodes[radius > radius[dst]]
        for candidates, relation in ((inward, 6), (outward, 7)):
            if candidates.numel() == 0:
                continue
            k_eff = min(int(k), int(candidates.numel()))
            cand_dist = d_rad[dst, candidates]
            _, order = torch.topk(cand_dist, k=k_eff, largest=False)
            src = candidates[order]
            dst_t = torch.full_like(src, int(dst))
            src_all.append(src)
            dst_all.append(dst_t)
            rel_all.append(torch.full_like(src, int(relation)))
    if not src_all:
        return _empty_edges(pos.device), torch.zeros((0,), dtype=torch.long, device=pos.device)
    edge_index = torch.stack([torch.cat(src_all), torch.cat(dst_all)], dim=0).long()
    edge_type = torch.cat(rel_all).long()
    return edge_index, edge_type


def _crystal_shell_edges(pos: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Coordination-shell edges inspired by crystal graph neighborhoods.

    Each node receives directed neighbors from the first, second, and third
    spatial coordination shells. Relation labels let the relational model treat
    immediate contacts differently from broader-shell context.
    """
    n = int(pos.shape[0])
    if n <= 1 or k <= 0:
        return _empty_edges(pos.device), torch.zeros((0,), dtype=torch.long, device=pos.device)
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    max_neighbors = min(3 * int(k), n - 1)
    if max_neighbors <= 0:
        return _empty_edges(pos.device), torch.zeros((0,), dtype=torch.long, device=pos.device)
    _, idx = torch.topk(d, k=max_neighbors, largest=False, dim=1)
    src_all: list[Tensor] = []
    dst_all: list[Tensor] = []
    rel_all: list[Tensor] = []
    for shell_idx, relation in enumerate((9, 10, 11)):
        start = shell_idx * int(k)
        stop = min((shell_idx + 1) * int(k), max_neighbors)
        if start >= stop:
            continue
        neigh = idx[:, start:stop]
        dst = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(neigh).reshape(-1)
        src = neigh.reshape(-1)
        src_all.append(src)
        dst_all.append(dst)
        rel_all.append(torch.full_like(src, int(relation)))
    if not src_all:
        return _empty_edges(pos.device), torch.zeros((0,), dtype=torch.long, device=pos.device)
    return torch.stack([torch.cat(src_all), torch.cat(dst_all)], dim=0).long(), torch.cat(rel_all).long()


def _shuffle_edge_targets(edge_index: Tensor, n_nodes: int) -> Tensor:
    """Deterministic edge-target shuffle for topology controls."""
    n = int(n_nodes)
    if edge_index.numel() == 0 or n <= 2:
        return edge_index
    src = edge_index[0].long()
    dst = edge_index[1].long()
    shift = max(1, n // 2)
    dst_shuf = (dst + shift) % n
    self_mask = dst_shuf == src
    if bool(self_mask.any()):
        dst_shuf = dst_shuf.clone()
        dst_shuf[self_mask] = (dst_shuf[self_mask] + 1) % n
    return torch.stack([src, dst_shuf], dim=0).long()


def _local_ssl_radius_edges(pos: Tensor, x: Tensor | None, k: int) -> Tensor:
    """Choose SSL-similar neighbors from a local spatial candidate pool."""
    n = int(pos.shape[0])
    if n <= 1 or k <= 0:
        return _empty_edges(pos.device)
    ssl_x = _ssl_feature_view(x)
    if ssl_x is None:
        return _knn_from_distance(torch.cdist(pos, pos), k)

    d_pos = torch.cdist(pos, pos)
    d_pos.fill_diagonal_(float("inf"))
    pool_k = min(max(2 * int(k), int(k) + 4), n - 1)
    if pool_k <= 0:
        return _empty_edges(pos.device)
    _, pool_idx = torch.topk(d_pos, k=pool_k, largest=False, dim=1)

    d_ssl = torch.cdist(ssl_x, ssl_x)
    ssl_scores = d_ssl.gather(1, pool_idx)
    k_eff = min(int(k), pool_k)
    _, order = torch.topk(ssl_scores, k=k_eff, largest=False, dim=1)
    idx = pool_idx.gather(1, order)
    src = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return ei.unique(dim=1)


def _history_response_edges(prev_x: Tensor | None, x: Tensor | None, k: int) -> Tensor:
    """Connect nodes with similar observed/predicted prior volume response."""
    if prev_x is None or x is None or x.shape[0] <= 1 or k <= 0:
        device = x.device if x is not None else (prev_x.device if prev_x is not None else torch.device("cpu"))
        return _empty_edges(device)
    n = min(int(prev_x.shape[0]), int(x.shape[0]))
    if n <= 1:
        return _empty_edges(x.device)
    if prev_x.shape[1] >= 2 and x.shape[1] >= 2:
        response = (x[:n, 1] - prev_x[:n, 1]).reshape(-1, 1)
    else:
        response = (x[:n, 0] - prev_x[:n, 0]).reshape(-1, 1)
    return _knn_from_distance(torch.cdist(response, response), k)


def _feature_view(x: Tensor, mode: str) -> Tensor:
    if mode == "ssl_feature_knn":
        if x.shape[1] > 6:
            return x[:, 6:]
        return x
    if mode == "feature_knn_pe_ser":
        # Current feature order: voxel_count_log, volume_ml, pe_mean, pe_std, ser_mean, ser_std.
        if x.shape[1] >= 6:
            return x[:, 2:6]
        return x
    if mode == "feature_knn_volume":
        if x.shape[1] >= 2:
            return x[:, 0:2]
        return x
    return x


def _ssl_feature_view(x: Tensor | None) -> Tensor | None:
    if x is None or x.shape[1] <= 6:
        return None
    return x[:, 6:]


def _hybrid_alpha(mode: str) -> float:
    m = re.search(r"_a(\d+)$", mode)
    if not m:
        return 0.50
    return float(int(m.group(1))) / 100.0


def _visit_edges(
    pos: Tensor,
    x: Tensor | None,
    k: int,
    mode: str,
    prev_x: Tensor | None = None,
) -> Tensor:
    if mode in {"full", "spatial_knn"}:
        return _knn_from_distance(torch.cdist(pos, pos), k)

    if mode == "radial_knn":
        radius = pos.norm(dim=1, keepdim=True)
        return _knn_from_distance(torch.cdist(radius, radius), k)

    if mode.startswith("radial_ssl_plus"):
        radius = pos.norm(dim=1, keepdim=True)
        radial_edges = _knn_from_distance(torch.cdist(radius, radius), k)
        ssl_x = _ssl_feature_view(x)
        if ssl_x is None:
            return radial_edges
        m = re.search(r"_plus(\d+)$", mode)
        k_ssl = int(m.group(1)) if m else 2
        ssl_edges = _knn_from_distance(torch.cdist(ssl_x, ssl_x), k_ssl)
        if radial_edges.numel() == 0:
            return ssl_edges
        if ssl_edges.numel() == 0:
            return radial_edges
        return torch.cat([radial_edges, ssl_edges], dim=1).unique(dim=1)

    if mode == "local_ssl_radius_knn":
        return _local_ssl_radius_edges(pos, x, k)

    if mode == "history_response_knn":
        history_edges = _history_response_edges(prev_x, x, k)
        if history_edges.numel() > 0:
            return history_edges
        return _knn_from_distance(torch.cdist(pos, pos), k)

    if mode == "spatial_history_union":
        spatial_edges = _knn_from_distance(torch.cdist(pos, pos), k)
        history_edges = _history_response_edges(prev_x, x, k)
        if history_edges.numel() == 0:
            return spatial_edges
        return torch.cat([spatial_edges, history_edges], dim=1).unique(dim=1)

    if mode == "spatial_local_ssl_history_union":
        edge_parts = [
            _knn_from_distance(torch.cdist(pos, pos), k),
            _local_ssl_radius_edges(pos, x, k),
            _history_response_edges(prev_x, x, k),
        ]
        edge_parts = [ei for ei in edge_parts if ei.numel() > 0]
        if not edge_parts:
            return _empty_edges(pos.device)
        return torch.cat(edge_parts, dim=1).unique(dim=1)

    if mode.startswith("feature_knn") or mode == "ssl_feature_knn":
        if x is None:
            raise ValueError(f"edge_mode={mode!r} requires history_x")
        return _knn_from_distance(torch.cdist(_feature_view(x, mode), _feature_view(x, mode)), k)

    if mode.startswith("hybrid_spatial_feature"):
        if x is None:
            raise ValueError(f"edge_mode={mode!r} requires history_x")
        alpha = _hybrid_alpha(mode)
        d_pos = torch.cdist(pos, pos)
        d_feat = torch.cdist(x, x)
        score = alpha * (d_pos / _positive_scale(d_pos)) + (1.0 - alpha) * (d_feat / _positive_scale(d_feat))
        return _knn_from_distance(score, k)

    if mode in {
        "multi_rel_union",
        "multi_rel_ssl_union",
        "directed_radial_bio",
        "learned_union_bio",
        "crystal_shell_union",
        "crystal_shell_shuffle_control",
    }:
        return _relation_visit_edges(pos, x, k, mode)[0]

    raise ValueError(f"unknown edge_mode={mode!r}")


def _relation_visit_edges(
    pos: Tensor,
    x: Tensor | None,
    k: int,
    mode: str,
) -> tuple[Tensor, Tensor]:
    edges: list[Tensor] = []
    types: list[Tensor] = []

    def add(ei: Tensor, et: Tensor) -> None:
        if ei.numel() == 0:
            return
        edges.append(ei)
        types.append(et)

    add(*_typed_knn_from_distance(torch.cdist(pos, pos), k, relation=0))

    radius = pos.norm(dim=1, keepdim=True)
    add(*_typed_knn_from_distance(torch.cdist(radius, radius), k, relation=1))

    if x is not None:
        if x.shape[1] >= 6:
            add(*_typed_knn_from_distance(torch.cdist(x[:, 2:6], x[:, 2:6]), k, relation=2))
        if x.shape[1] >= 2:
            add(*_typed_knn_from_distance(torch.cdist(x[:, 0:2], x[:, 0:2]), k, relation=3))

    if mode in {"multi_rel_ssl_union"}:
        ssl_x = _ssl_feature_view(x)
        if ssl_x is not None:
            add(*_typed_knn_from_distance(torch.cdist(ssl_x, ssl_x), k, relation=8))

    if mode == "directed_radial_bio":
        add(*_directed_radial_edges(pos, k))

    if mode in {"crystal_shell_union", "crystal_shell_shuffle_control"}:
        add(*_crystal_shell_edges(pos, k))
        add(*_directed_radial_edges(pos, max(1, k // 2)))

    if not edges:
        return _empty_edges(pos.device), torch.zeros((0,), dtype=torch.long, device=pos.device)
    edge_index = torch.cat(edges, dim=1).long()
    edge_type = torch.cat(types).long()
    if mode == "crystal_shell_shuffle_control":
        edge_index = _shuffle_edge_targets(edge_index, int(pos.shape[0]))
    return edge_index, edge_type


def build_history_edges(
    history_pos: list[Tensor],
    history_x: list[Tensor] | None = None,
    *,
    k_spatial: int = 8,
    edge_mode: str = "full",
) -> Tensor:
    """Build dynamic rollout edges for a list of visit histories.

    All non-``none`` modes keep deterministic identity temporal edges between
    visits. ``temporal_only`` keeps those identity edges without adding
    intra-visit neighbors.
    """
    if not history_pos:
        raise ValueError("history_pos must contain at least one visit")
    mode = "spatial_knn" if edge_mode == "full" else str(edge_mode)
    if mode == "none":
        return _empty_edges(history_pos[0].device)
    if mode not in EDGE_MODE_CHOICES:
        raise ValueError(f"unknown edge_mode={edge_mode!r}")
    if history_x is None:
        history_x = [None] * len(history_pos)  # type: ignore[list-item]
    if len(history_x) != len(history_pos):
        raise ValueError("history_x and history_pos must have the same number of visits")

    all_src: list[Tensor] = []
    all_dst: list[Tensor] = []
    offsets: list[int] = []
    cur = 0
    for pos in history_pos:
        offsets.append(cur)
        cur += int(pos.shape[0])
    offsets.append(cur)

    for v, pos in enumerate(history_pos):
        off = offsets[v]
        n_v = int(pos.shape[0])
        if mode != "temporal_only" and n_v > 1 and k_spatial > 0:
            prev_x = history_x[v - 1] if v > 0 else None
            ei_v = _visit_edges(pos, history_x[v], int(k_spatial), mode, prev_x=prev_x)
            if ei_v.numel():
                all_src.append(ei_v[0] + off)
                all_dst.append(ei_v[1] + off)
        if v < len(history_pos) - 1:
            n_next = int(history_pos[v + 1].shape[0])
            n_link = min(n_v, n_next)
            if n_link > 0:
                src_t = torch.arange(n_link, device=pos.device) + off
                dst_t = torch.arange(n_link, device=pos.device) + offsets[v + 1]
                all_src += [src_t, dst_t]
                all_dst += [dst_t, src_t]

    if not all_src:
        return _empty_edges(history_pos[0].device)
    return torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0).long()


def build_history_graph_with_types(
    history_pos: list[Tensor],
    history_x: list[Tensor] | None = None,
    *,
    k_spatial: int = 8,
    edge_mode: str = "full",
    edge_attr_mode: str = "none",
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Build dynamic rollout edges, attributes, and optional relation labels."""
    if not history_pos:
        raise ValueError("history_pos must contain at least one visit")
    mode = "spatial_knn" if edge_mode == "full" else str(edge_mode)
    if mode not in EDGE_MODE_CHOICES:
        raise ValueError(f"unknown edge_mode={edge_mode!r}")
    if history_x is None:
        history_x = [None] * len(history_pos)  # type: ignore[list-item]
    if len(history_x) != len(history_pos):
        raise ValueError("history_x and history_pos must have the same number of visits")

    if mode == "none":
        edge_index = _empty_edges(history_pos[0].device)
        edge_type = torch.zeros((0,), dtype=torch.long, device=history_pos[0].device)
    else:
        all_src: list[Tensor] = []
        all_dst: list[Tensor] = []
        all_type: list[Tensor] = []
        offsets: list[int] = []
        cur = 0
        for pos in history_pos:
            offsets.append(cur)
            cur += int(pos.shape[0])
        offsets.append(cur)

        relation_mode = mode in {
            "multi_rel_union",
            "multi_rel_ssl_union",
            "directed_radial_bio",
            "learned_union_bio",
            "crystal_shell_union",
            "crystal_shell_shuffle_control",
        }
        for v, pos in enumerate(history_pos):
            off = offsets[v]
            n_v = int(pos.shape[0])
            if mode != "temporal_only" and n_v > 1 and k_spatial > 0:
                if relation_mode:
                    ei_v, et_v = _relation_visit_edges(pos, history_x[v], int(k_spatial), mode)
                else:
                    prev_x = history_x[v - 1] if v > 0 else None
                    ei_v = _visit_edges(pos, history_x[v], int(k_spatial), mode, prev_x=prev_x)
                    et_v = torch.zeros((ei_v.shape[1],), dtype=torch.long, device=pos.device)
                if ei_v.numel():
                    all_src.append(ei_v[0] + off)
                    all_dst.append(ei_v[1] + off)
                    all_type.append(et_v)
            if v < len(history_pos) - 1:
                n_next = int(history_pos[v + 1].shape[0])
                n_link = min(n_v, n_next)
                if n_link > 0:
                    src_t = torch.arange(n_link, device=pos.device) + off
                    dst_t = torch.arange(n_link, device=pos.device) + offsets[v + 1]
                    all_src += [src_t, dst_t]
                    all_dst += [dst_t, src_t]
                    all_type += [
                        torch.full((n_link,), 4, dtype=torch.long, device=pos.device),
                        torch.full((n_link,), 5, dtype=torch.long, device=pos.device),
                    ]

        if all_src:
            edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0).long()
            edge_type = torch.cat(all_type).long()
        else:
            edge_index = _empty_edges(history_pos[0].device)
            edge_type = torch.zeros((0,), dtype=torch.long, device=history_pos[0].device)

    if edge_attr_dim(edge_attr_mode) == 0:
        return edge_index, None, edge_type
    pos_cat = torch.cat(history_pos, dim=0)
    x_cat = torch.cat(history_x, dim=0) if history_x is not None else None
    t_cat = torch.cat([
        torch.full((p.shape[0],), float(v), dtype=pos_cat.dtype, device=pos_cat.device)
        for v, p in enumerate(history_pos)
    ])
    edge_attr = _edge_attr_from_index(
        pos_cat,
        x_cat,
        t_cat,
        edge_index,
        edge_attr_mode=edge_attr_mode,
    )
    return edge_index, edge_attr, edge_type


def _edge_attr_from_index(
    pos: Tensor,
    x: Tensor | None,
    t: Tensor,
    edge_index: Tensor,
    *,
    edge_attr_mode: str = "none",
) -> Tensor | None:
    mode = str(edge_attr_mode)
    dim = edge_attr_dim(mode)
    if dim == 0:
        return None
    if edge_index.numel() == 0:
        return _empty_edge_attr(pos.device, dim).to(dtype=pos.dtype)

    src = edge_index[0].long()
    dst = edge_index[1].long()
    p_src = pos[src]
    p_dst = pos[dst]
    delta = p_src - p_dst
    dist = delta.norm(dim=-1, keepdim=True)
    r_src = p_src.norm(dim=-1, keepdim=True)
    r_dst = p_dst.norm(dim=-1, keepdim=True)
    radial_delta = r_src - r_dst

    dist_scale = _positive_scale(dist)
    radius_scale = _positive_scale(torch.cat([r_src, r_dst], dim=0))
    radial_delta_scale = _positive_scale(radial_delta.abs())
    cos_radial = (p_src * p_dst).sum(dim=-1, keepdim=True) / (r_src * r_dst + 1e-6)
    same_shell = torch.exp(-radial_delta.abs() / radial_delta_scale)
    dt = (t[src].float().view(-1, 1) - t[dst].float().view(-1, 1)).abs()
    dt_scale = _positive_scale(dt)
    is_temporal = (dt > 0).to(dtype=pos.dtype)

    attrs = [
        (dist / dist_scale).clamp(0.0, 6.0),
        (radial_delta.abs() / radial_delta_scale).clamp(0.0, 6.0),
        (radial_delta / radial_delta_scale).clamp(-6.0, 6.0),
        (r_src / radius_scale).clamp(0.0, 6.0),
        (r_dst / radius_scale).clamp(0.0, 6.0),
        cos_radial.clamp(-1.0, 1.0),
        same_shell.clamp(0.0, 1.0),
        (dt / dt_scale).clamp(0.0, 6.0),
        is_temporal,
    ]

    if mode in {"radial_bio", "ssl_radial_bio"}:
        if x is None:
            x = torch.zeros((pos.shape[0], 0), dtype=pos.dtype, device=pos.device)
        if x.shape[1] >= 2:
            vol_diff = (x[src, 0:2] - x[dst, 0:2]).norm(dim=-1, keepdim=True)
        else:
            vol_diff = torch.zeros_like(dist)
        if x.shape[1] >= 6:
            pe_ser_diff = (x[src, 2:6] - x[dst, 2:6]).norm(dim=-1, keepdim=True)
        else:
            pe_ser_diff = torch.zeros_like(dist)
        if x.shape[1] > 0:
            feat_diff = (x[src] - x[dst]).norm(dim=-1, keepdim=True)
        else:
            feat_diff = torch.zeros_like(dist)
        attrs += [
            (vol_diff / _positive_scale(vol_diff)).clamp(0.0, 6.0),
            (pe_ser_diff / _positive_scale(pe_ser_diff)).clamp(0.0, 6.0),
            (feat_diff / _positive_scale(feat_diff)).clamp(0.0, 6.0),
        ]

    if mode == "ssl_radial_bio":
        if x is None:
            x = torch.zeros((pos.shape[0], 0), dtype=pos.dtype, device=pos.device)
        if x.shape[1] >= 6:
            core_delta = x[src, :6] - x[dst, :6]
            signed_core = torch.stack(
                [
                    core_delta[:, 0],
                    core_delta[:, 1],
                    core_delta[:, 2],
                    core_delta[:, 4],
                ],
                dim=-1,
            )
            core_scale = _positive_scale(signed_core.abs().reshape(-1, 1))
            signed_core = (signed_core / core_scale).clamp(-6.0, 6.0)
        else:
            signed_core = torch.zeros((edge_index.shape[1], 4), dtype=pos.dtype, device=pos.device)

        ssl_x = _ssl_feature_view(x)
        if ssl_x is not None:
            ssl_src = ssl_x[src]
            ssl_dst = ssl_x[dst]
            ssl_l2 = (ssl_src - ssl_dst).norm(dim=-1, keepdim=True)
            ssl_l1 = (ssl_src - ssl_dst).abs().mean(dim=-1, keepdim=True)
            ssl_cos = torch.nn.functional.cosine_similarity(ssl_src, ssl_dst, dim=-1).view(-1, 1)
            ssl_scale = _positive_scale(ssl_l2)
            ssl_l2_scaled = (ssl_l2 / ssl_scale).clamp(0.0, 6.0)
            ssl_l1_scaled = (ssl_l1 / _positive_scale(ssl_l1)).clamp(0.0, 6.0)
            ssl_same = torch.exp(-ssl_l2 / ssl_scale).clamp(0.0, 1.0)
            ssl_attrs = torch.cat(
                [
                    ssl_cos.clamp(-1.0, 1.0),
                    ssl_l2_scaled,
                    ssl_l1_scaled,
                    ssl_same,
                ],
                dim=-1,
            )
        else:
            ssl_attrs = torch.zeros((edge_index.shape[1], 4), dtype=pos.dtype, device=pos.device)
        attrs += [signed_core, ssl_attrs]

    out = torch.cat(attrs, dim=-1).to(dtype=pos.dtype)
    if out.shape[1] != dim:
        raise RuntimeError(f"edge_attr_mode={mode!r} produced width {out.shape[1]} not {dim}")
    return out


def build_history_graph(
    history_pos: list[Tensor],
    history_x: list[Tensor] | None = None,
    *,
    k_spatial: int = 8,
    edge_mode: str = "full",
    edge_attr_mode: str = "none",
) -> tuple[Tensor, Tensor | None]:
    """Build dynamic rollout edges plus optional geometric/biology edge attributes."""
    edge_index, edge_attr, _ = build_history_graph_with_types(
        history_pos,
        history_x,
        k_spatial=k_spatial,
        edge_mode=edge_mode,
        edge_attr_mode=edge_attr_mode,
    )
    return edge_index, edge_attr
