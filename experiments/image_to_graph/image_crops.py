from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from experiments.image_to_graph.image_patch_io import (
    fixed_patch_view,
    finite_or_zero,
    load_visit_bundle,
    world_to_voxel_zyx,
)


@dataclass(frozen=True)
class ImageChannelStats:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_values(cls, mean: Sequence[float], std: Sequence[float]) -> "ImageChannelStats":
        return cls(
            mean=torch.as_tensor(mean, dtype=torch.float32),
            std=torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6),
        )


def node_world_positions(payload: dict, visit_idx: int) -> np.ndarray:
    offsets = payload["visit_offsets"]
    if torch.is_tensor(offsets):
        offsets = offsets.detach().cpu().tolist()
    start, end = int(offsets[visit_idx]), int(offsets[visit_idx + 1])
    pos = payload["pos"][start:end]
    if torch.is_tensor(pos):
        pos_np = pos.detach().cpu().numpy().astype(np.float32)
    else:
        pos_np = np.asarray(pos, dtype=np.float32)
    visit_centroids = payload.get("visit_centroids")
    if visit_centroids is not None:
        centroid = np.asarray(visit_centroids[visit_idx], dtype=np.float32).reshape(1, 3)
        pos_np = pos_np + centroid
    return pos_np


def extract_visit_crops(
    *,
    derived_root: Path,
    payload: dict,
    visit_idx: int,
    crop_size: int,
    channel_stats: ImageChannelStats,
    node_indices: Sequence[int] | None = None,
    include_label: bool = True,
) -> torch.Tensor:
    patient_id = str(payload["patient_id"])
    visit_ids = list(payload.get("visit_ids") or [])
    if not visit_ids:
        raise KeyError(f"{patient_id}: graph payload is missing visit_ids")
    visit_id = str(visit_ids[int(visit_idx)])
    bundle = load_visit_bundle(derived_root, patient_id, visit_id)
    positions = node_world_positions(payload, int(visit_idx))
    if node_indices is not None:
        idx = np.asarray(list(node_indices), dtype=np.int64)
        positions = positions[idx]

    means = channel_stats.mean.detach().cpu().numpy().astype(np.float32)
    stds = channel_stats.std.detach().cpu().numpy().astype(np.float32)
    crops = np.zeros(
        (positions.shape[0], 3 if include_label else 2, crop_size, crop_size, crop_size),
        dtype=np.float32,
    )
    for i, world in enumerate(positions):
        center = world_to_voxel_zyx(world, bundle.origin, bundle.spacing)
        pe = fixed_patch_view(bundle.pe, center, crop_size)
        ser = fixed_patch_view(bundle.ser, center, crop_size)
        label = fixed_patch_view(bundle.labels, center, crop_size)
        crops[i, 0] = (finite_or_zero(pe) - means[0]) / max(float(stds[0]), 1e-6)
        crops[i, 1] = (finite_or_zero(ser) - means[1]) / max(float(stds[1]), 1e-6)
        if include_label:
            crops[i, 2] = (label > 0).astype(np.float32)
    return torch.from_numpy(crops)


def extract_patient_crops(
    *,
    derived_root: Path,
    payload: dict,
    crop_size: int,
    channel_stats: ImageChannelStats,
    node_indices_by_visit: Sequence[Sequence[int] | None] | None = None,
    include_label: bool = True,
) -> torch.Tensor:
    offsets = payload["visit_offsets"]
    if torch.is_tensor(offsets):
        offsets = offsets.detach().cpu().tolist()
    n_visits = len(offsets) - 1
    pieces: list[torch.Tensor] = []
    for visit_idx in range(n_visits):
        node_indices = None
        if node_indices_by_visit is not None:
            node_indices = node_indices_by_visit[visit_idx]
        pieces.append(
            extract_visit_crops(
                derived_root=derived_root,
                payload=payload,
                visit_idx=visit_idx,
                crop_size=crop_size,
                channel_stats=channel_stats,
                node_indices=node_indices,
                include_label=include_label,
            )
        )
    return torch.cat(pieces, dim=0)


def sample_image_channel_stats(
    *,
    derived_root: Path,
    samples: Sequence[object],
    max_voxels_per_visit: int = 50000,
    seed: int = 42,
) -> ImageChannelStats:
    """Estimate PE/SER normalization stats from train-fold image volumes only."""
    rng = np.random.default_rng(seed)
    sums = np.zeros(2, dtype=np.float64)
    sums_sq = np.zeros(2, dtype=np.float64)
    count = 0
    for sample in samples:
        payload = getattr(sample, "payload")
        patient_id = str(payload["patient_id"])
        for visit_id in list(payload.get("visit_ids") or []):
            bundle = load_visit_bundle(derived_root, patient_id, str(visit_id))
            pe = finite_or_zero(bundle.pe).reshape(-1)
            ser = finite_or_zero(bundle.ser).reshape(-1)
            n_vox = int(min(pe.shape[0], ser.shape[0]))
            if n_vox <= 0:
                continue
            if n_vox > max_voxels_per_visit:
                idx = rng.choice(n_vox, size=max_voxels_per_visit, replace=False)
                pe_vals = pe[idx].astype(np.float64, copy=False)
                ser_vals = ser[idx].astype(np.float64, copy=False)
            else:
                pe_vals = pe[:n_vox].astype(np.float64, copy=False)
                ser_vals = ser[:n_vox].astype(np.float64, copy=False)
            sums[0] += pe_vals.sum()
            sums[1] += ser_vals.sum()
            sums_sq[0] += np.square(pe_vals, dtype=np.float64).sum()
            sums_sq[1] += np.square(ser_vals, dtype=np.float64).sum()
            count += int(pe_vals.shape[0])
    if count == 0:
        return ImageChannelStats.from_values([0.0, 0.0], [1.0, 1.0])
    mean = sums / count
    var = np.maximum((sums_sq / count) - np.square(mean), 1e-6)
    return ImageChannelStats.from_values(mean.tolist(), np.sqrt(var).tolist())
