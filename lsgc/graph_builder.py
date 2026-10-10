"""Build the unified spatio-temporal supervoxel graph consumed by LSGCConv.

Given a sequence of per-visit supervoxel tables for one patient, return a
single PyG edge_index (plus pos, t, and a node-offset table) in which:

    * intra-visit edges connect the k_spatial nearest supervoxel centroids
      *within the same visit* (symmetric kNN).
    * inter-visit edges connect each supervoxel at visit v to its k_temporal
      nearest supervoxels at visit v+1 using a combined distance in
      position + feature space. No rigid registration is assumed — the
      operator learns edge weights from delta_pos and delta_t.

This is intentionally a small, testable builder; the heavy lifting (supervoxel
segmentation, feature extraction) happens upstream in the preprocessing
pipeline. The returned graph is ready for LSGCNet(...).forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass
class SpatioTemporalGraph:
    x: Tensor           # (N, C) node features
    pos: Tensor         # (N, 3) centroids in mm
    t: Tensor           # (N,)   visit index (long)
    edge_index: Tensor  # (2, E) PyG edges, source -> target
    edge_attr: Tensor | None  # (E, A) optional edge attributes
    edge_type: Tensor | None  # (E,) optional relation id for relational GNNs
    visit_offsets: Tensor  # (T+1,) prefix sum of per-visit node counts


def _pairwise_knn(
    query: Tensor, key: Tensor, k: int, exclude_self: bool = False
) -> Tensor:
    """Return indices into `key` of the k nearest neighbors for each row of `query`.

    Distances are plain Euclidean on the supplied feature tensors. Works on CPU
    and GPU, O(N*M) memory -- adequate for supervoxel counts (hundreds to a
    few thousand). For larger N switch to torch_cluster.knn.
    """
    if query.dim() != 2 or key.dim() != 2 or query.size(1) != key.size(1):
        raise ValueError("query and key must be 2D with matching feature dim")
    d = torch.cdist(query, key)  # (Nq, Nk)
    if exclude_self and query.size(0) == key.size(0):
        d = d + torch.eye(query.size(0), device=d.device) * 1e9
    k_eff = min(k, key.size(0))
    _, idx = torch.topk(d, k=k_eff, largest=False, dim=1)
    return idx  # (Nq, k_eff)


def _bio_edge_attrs(
    f_a: Tensor, f_b: Tensor, src_local: Tensor, dst_local: Tensor,
    h_a: Tensor | None, h_b: Tensor | None, *,
    is_inter_visit: bool, adc_idx: int, adc_missing_idx: int | None,
    dce_idx_start: int, dce_n_phases: int,
    adc_scale_mm2_per_s: float = 1500.0,
) -> Tensor:
    """Compute 4-channel biological edge attributes.

    Channels (continuous, all roughly in [0, 1] or [-1, 1]):
      ch0  Δ_adc        : |ADC_i − ADC_j| / adc_scale, capped at 1.0; 0 if either
                          endpoint has the adc_missing flag set (no DWI). Identifies
                          tissue-boundary edges (necrotic ↔ cellular tumor).
      ch1  cross_habitat: 1 if endpoints have different DCE-perfusion habitats, else 0.
                          Marks heterogeneity-interface edges.
      ch2  dce_cosine   : 1 − cos(DCE_curve_i, DCE_curve_j) ∈ [0, 2]; small for
                          edges between vasculature-shared regions, large for edges
                          crossing perfusion phenotype boundaries.
      ch3  is_inter_visit: 1 if edge spans visits (temporal), 0 if intra-visit (spatial).
    """
    n_e = src_local.numel()
    out = torch.zeros((n_e, 4), dtype=torch.float32)

    if adc_idx is not None and 0 <= adc_idx < f_a.shape[1] and 0 <= adc_idx < f_b.shape[1]:
        adc_a = f_a[src_local, adc_idx]
        adc_b = f_b[dst_local, adc_idx]
        delta = (adc_a - adc_b).abs() / float(max(adc_scale_mm2_per_s, 1e-6))
        delta = delta.clamp(0.0, 1.0)
        if adc_missing_idx is not None and adc_missing_idx < f_a.shape[1]:
            mi_a = f_a[src_local, adc_missing_idx]
            mi_b = f_b[dst_local, adc_missing_idx]
            valid = (mi_a < 0.5) & (mi_b < 0.5)
            delta = delta * valid.float()
        out[:, 0] = delta.float()

    if h_a is not None and h_b is not None:
        out[:, 1] = (h_a[src_local] != h_b[dst_local]).float()

    if dce_n_phases > 0:
        ce = slice(dce_idx_start, dce_idx_start + dce_n_phases)
        if ce.stop <= f_a.shape[1] and ce.stop <= f_b.shape[1]:
            ca = f_a[src_local, ce]
            cb = f_b[dst_local, ce]
            na = ca.norm(dim=1).clamp_min(1e-6)
            nb = cb.norm(dim=1).clamp_min(1e-6)
            cos = (ca * cb).sum(dim=1) / (na * nb)
            out[:, 2] = (1.0 - cos).clamp(0.0, 2.0)

    if is_inter_visit:
        out[:, 3] = 1.0

    return out


def _relation_types(
    src_local: Tensor,
    dst_local: Tensor,
    h_a: Tensor | None,
    h_b: Tensor | None,
    *,
    is_inter_visit: bool,
) -> Tensor:
    """Assign 4 biological relation IDs to edges.

    0 = intra-habitat spatial, 1 = cross-habitat spatial,
    2 = temporal progression, 3 = temporal transition.
    If habitat labels are absent, edges default to same-habitat / progression.
    """
    if h_a is not None and h_b is not None:
        cross_habitat = h_a[src_local] != h_b[dst_local]
    else:
        cross_habitat = torch.zeros_like(src_local, dtype=torch.bool)

    if is_inter_visit:
        base = torch.full_like(src_local, 2, dtype=torch.long)
        return torch.where(cross_habitat, torch.full_like(base, 3), base)
    base = torch.zeros_like(src_local, dtype=torch.long)
    return torch.where(cross_habitat, torch.full_like(base, 1), base)


def build_spatiotemporal_graph(
    visit_features: Sequence[Tensor],
    visit_positions: Sequence[Tensor],
    *,
    k_spatial: int = 8,
    k_temporal: int = 4,
    feature_weight: float = 1.0,
    position_weight: float = 1.0,
    symmetric: bool = True,
    temporal_skip_hops: Sequence[int] = (1,),
    edge_mode: str = "geometric",
    spatial_alpha: float = 0.7,
    add_edge_attr: bool = False,
    habitat_labels: Sequence[Tensor] | None = None,
    edge_attr_mode: str = "legacy",
    adc_idx: int | None = None,
    adc_missing_idx: int | None = None,
    dce_idx_start: int = 0,
    dce_n_phases: int = 0,
) -> SpatioTemporalGraph:
    """Construct a single unified spatio-temporal graph for one patient.

    Parameters
    ----------
    visit_features
        List of length T; each entry is (N_v, C) per-supervoxel features
        at visit v (e.g. PE/SER summaries, kinetic curve).
    visit_positions
        List of length T; each entry is (N_v, 3) supervoxel centroids in mm.
    k_spatial
        Neighbors per node for intra-visit edges.
    k_temporal
        Neighbors per node for inter-visit edges (per skip-hop).
    feature_weight, position_weight
        Weights on feature- vs. position-distance when finding temporal
        neighbors. Both are applied after per-component z-scoring.
    symmetric
        If True, every edge is also added in the reverse direction.
    temporal_skip_hops
        Set of visit-index gaps for which to add inter-visit edges. The
        default ``(1,)`` connects only consecutive visits (v -> v+1), the
        original behavior. Passing e.g. ``(1, 2, 3)`` additionally adds
        v -> v+2 and v -> v+3 edges, shrinking the temporal diameter of
        the graph from 3 hops to 1 on a four-visit patient. The LSGC
        operator consumes the skip edges natively because Delta t is
        already a signed-scalar input to the edge filter.
    """
    if len(visit_features) != len(visit_positions):
        raise ValueError("visit_features and visit_positions must have equal length")
    if len(visit_features) == 0:
        raise ValueError("need at least one visit")

    T = len(visit_features)
    sizes = [f.size(0) for f in visit_features]
    offsets = torch.tensor([0, *sizes]).cumsum(dim=0)  # (T+1,)

    x = torch.cat(list(visit_features), dim=0)
    pos = torch.cat(list(visit_positions), dim=0)
    t = torch.cat([torch.full((n,), v, dtype=torch.long) for v, n in enumerate(sizes)])

    edges: list[Tensor] = []
    attrs: list[Tensor] = []
    types: list[Tensor] = []

    # Intra-visit spatial kNN -------------------------------------------------
    for v in range(T):
        p = visit_positions[v]
        if p.size(0) == 0:
            continue
        if edge_mode.startswith("mixed"):
            f = visit_features[v]
            eps = 1e-6
            p_std = p.std(dim=0, keepdim=True).clamp_min(eps)
            f_std = f.std(dim=0, keepdim=True).clamp_min(eps)
            mix = torch.cat(
                [spatial_alpha * p / p_std, (1.0 - spatial_alpha) * f / f_std], dim=1
            )
            nbr = _pairwise_knn(mix, mix, k=k_spatial, exclude_self=True)
        else:
            nbr = _pairwise_knn(p, p, k=k_spatial, exclude_self=True)  # (N_v, k)
        src = torch.arange(p.size(0)).repeat_interleave(nbr.size(1))
        dst = nbr.reshape(-1)
        src = src + offsets[v]
        dst = dst + offsets[v]
        edges.append(torch.stack([src, dst], dim=0))
        h = None if habitat_labels is None else habitat_labels[v]
        src_local = src - offsets[v]
        dst_local = dst - offsets[v]
        types.append(_relation_types(
            src_local, dst_local, h, h, is_inter_visit=False
        ))
        if add_edge_attr:
            f = visit_features[v]
            if edge_attr_mode == "bio":
                attrs.append(_bio_edge_attrs(
                    f, f, src_local, dst_local,
                    h_a=h, h_b=h, is_inter_visit=False,
                    adc_idx=adc_idx, adc_missing_idx=adc_missing_idx,
                    dce_idx_start=dce_idx_start, dce_n_phases=dce_n_phases,
                ))
            else:
                same_h = (
                    (h[src_local] == h[dst_local]).float()
                    if h is not None else torch.zeros_like(src, dtype=torch.float32)
                )
                pe_diff = (f[src_local, 2] - f[dst_local, 2]).abs()
                vol_diff = (f[src_local, 1] - f[dst_local, 1]).abs()
                attrs.append(
                    torch.stack(
                        [
                            same_h,
                            (pe_diff < 0.25).float(),
                            (vol_diff > 0.5).float(),
                            same_h,
                        ],
                        dim=1,
                    )
                )

    # Inter-visit feature+position kNN ---------------------------------------
    skips = sorted({int(h) for h in temporal_skip_hops})
    if any(h < 1 for h in skips):
        raise ValueError("temporal_skip_hops entries must be >= 1")
    for hop in skips:
        for v in range(T - hop):
            p_a, p_b = visit_positions[v], visit_positions[v + hop]
            f_a, f_b = visit_features[v], visit_features[v + hop]
            if p_a.size(0) == 0 or p_b.size(0) == 0:
                continue
            eps = 1e-6
            p_ab = torch.cat([p_a, p_b], dim=0)
            f_ab = torch.cat([f_a, f_b], dim=0)
            p_std = p_ab.std(dim=0, keepdim=True).clamp_min(eps)
            f_std = f_ab.std(dim=0, keepdim=True).clamp_min(eps)
            q = torch.cat(
                [position_weight * p_a / p_std, feature_weight * f_a / f_std], dim=1
            )
            k_ = torch.cat(
                [position_weight * p_b / p_std, feature_weight * f_b / f_std], dim=1
            )
            nbr = _pairwise_knn(q, k_, k=k_temporal, exclude_self=False)
            src = torch.arange(p_a.size(0)).repeat_interleave(nbr.size(1))
            dst = nbr.reshape(-1)
            src = src + offsets[v]
            dst = dst + offsets[v + hop]
            edges.append(torch.stack([src, dst], dim=0))
            h_a = None if habitat_labels is None else habitat_labels[v]
            h_b = None if habitat_labels is None else habitat_labels[v + hop]
            src_local = src - offsets[v]
            dst_local = dst - offsets[v + hop]
            types.append(_relation_types(
                src_local, dst_local, h_a, h_b, is_inter_visit=True
            ))
            if add_edge_attr:
                if edge_attr_mode == "bio":
                    attrs.append(_bio_edge_attrs(
                        f_a, f_b, src_local, dst_local,
                        h_a=h_a, h_b=h_b, is_inter_visit=True,
                        adc_idx=adc_idx, adc_missing_idx=adc_missing_idx,
                        dce_idx_start=dce_idx_start, dce_n_phases=dce_n_phases,
                    ))
                else:
                    if h_a is not None and h_b is not None:
                        same_h = (h_a[src_local] == h_b[dst_local]).float()
                    else:
                        same_h = torch.zeros_like(src, dtype=torch.float32)
                    pe_diff = (f_a[src_local, 2] - f_b[dst_local, 2]).abs()
                    vol_diff = (f_a[src_local, 1] - f_b[dst_local, 1]).abs()
                    attrs.append(
                        torch.stack(
                            [
                                same_h,
                                (pe_diff < 0.25).float(),
                                (vol_diff > 0.5).float(),
                                torch.zeros_like(same_h),
                            ],
                            dim=1,
                        )
                    )

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.cat(edges, dim=1)
        edge_attr = torch.cat(attrs, dim=0) if attrs else None
        edge_type = torch.cat(types, dim=0) if types else None
        if symmetric:
            rev = edge_index.flip(0)
            edge_index = torch.cat([edge_index, rev], dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
            if edge_type is not None:
                edge_type = torch.cat([edge_type, edge_type], dim=0)

    if not edges:
        edge_attr = None

    return SpatioTemporalGraph(
        x=x, pos=pos, t=t, edge_index=edge_index, edge_attr=edge_attr,
        edge_type=edge_type, visit_offsets=offsets
    )
