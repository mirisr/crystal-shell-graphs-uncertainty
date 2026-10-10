from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VisitImageBundle:
    patient_id: str
    visit_id: str
    pe: np.ndarray
    ser: np.ndarray
    labels: np.ndarray
    spacing: np.ndarray
    origin: np.ndarray
    supervoxels: pd.DataFrame
    meta: dict


REQUIRED_VISIT_FILES = (
    "pe_ser.npz",
    "supervoxel_labels.npz",
    "supervoxels.parquet",
    "meta.json",
)


def required_visit_paths(derived_root: Path, patient_id: str, visit_id: str) -> dict[str, Path]:
    visit_dir = derived_root / patient_id / visit_id
    return {name: visit_dir / name for name in REQUIRED_VISIT_FILES}


def missing_visit_files(derived_root: Path, patient_id: str, visit_id: str) -> list[str]:
    paths = required_visit_paths(derived_root, patient_id, visit_id)
    return [name for name, path in paths.items() if not path.exists()]


def has_complete_visit(derived_root: Path, patient_id: str, visit_id: str) -> bool:
    return not missing_visit_files(derived_root, patient_id, visit_id)


def load_visit_bundle(derived_root: Path, patient_id: str, visit_id: str) -> VisitImageBundle:
    paths = required_visit_paths(derived_root, patient_id, visit_id)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{patient_id}/{visit_id} is missing required files: {', '.join(missing)}"
        )

    meta = json.loads(paths["meta.json"].read_text())
    pe_ser = np.load(paths["pe_ser.npz"])
    labels_npz = np.load(paths["supervoxel_labels.npz"])

    spacing = np.asarray(
        pe_ser["voxel_spacing"] if "voxel_spacing" in pe_ser.files else meta["voxel_spacing_mm"],
        dtype=np.float32,
    )
    origin = np.asarray(
        pe_ser["origin"] if "origin" in pe_ser.files else meta["origin_mm"],
        dtype=np.float32,
    )

    return VisitImageBundle(
        patient_id=patient_id,
        visit_id=visit_id,
        pe=np.asarray(pe_ser["pe"], dtype=np.float32),
        ser=np.asarray(pe_ser["ser"], dtype=np.float32),
        labels=np.asarray(labels_npz["labels"], dtype=np.int32),
        spacing=spacing,
        origin=origin,
        supervoxels=pd.read_parquet(paths["supervoxels.parquet"]),
        meta=meta,
    )


def world_to_voxel_zyx(world_zyx: np.ndarray, origin_zyx: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    """Map a world-space z/y/x point in mm to nearest z/y/x voxel index."""
    return np.rint((np.asarray(world_zyx, dtype=np.float32) - origin_zyx) / spacing_zyx).astype(np.int64)


def radius_mm_to_voxels(radius_mm: float, spacing_zyx: np.ndarray) -> np.ndarray:
    radius = np.ceil(float(radius_mm) / np.maximum(spacing_zyx, 1e-6)).astype(np.int64)
    return np.maximum(radius, 1)


def clipped_slices(center_zyx: np.ndarray, radius_zyx: np.ndarray, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    center = np.asarray(center_zyx, dtype=np.int64)
    radius = np.asarray(radius_zyx, dtype=np.int64)
    starts = np.maximum(center - radius, 0)
    stops = np.minimum(center + radius + 1, np.asarray(shape, dtype=np.int64))
    return tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))


def patch_view(volume: np.ndarray, center_zyx: np.ndarray, radius_zyx: np.ndarray) -> np.ndarray:
    return volume[clipped_slices(center_zyx, radius_zyx, volume.shape)]


def fixed_patch_view(
    volume: np.ndarray,
    center_zyx: np.ndarray,
    size: int | Sequence[int],
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Return a fixed z/y/x patch centered on ``center_zyx`` with padding."""
    arr = np.asarray(volume)
    if isinstance(size, int):
        size_zyx = np.asarray([size, size, size], dtype=np.int64)
    else:
        size_zyx = np.asarray(list(size), dtype=np.int64)
    if size_zyx.shape != (3,):
        raise ValueError("size must be an int or three z/y/x sizes")
    if np.any(size_zyx <= 0):
        raise ValueError("patch sizes must be positive")

    center = np.asarray(center_zyx, dtype=np.int64)
    starts = center - (size_zyx // 2)
    stops = starts + size_zyx
    src_starts = np.maximum(starts, 0)
    src_stops = np.minimum(stops, np.asarray(arr.shape[:3], dtype=np.int64))
    dst_starts = src_starts - starts
    dst_stops = dst_starts + np.maximum(src_stops - src_starts, 0)

    out = np.full(tuple(int(x) for x in size_zyx), pad_value, dtype=arr.dtype)
    if np.any(src_stops <= src_starts):
        return out
    src = tuple(slice(int(a), int(b)) for a, b in zip(src_starts, src_stops))
    dst = tuple(slice(int(a), int(b)) for a, b in zip(dst_starts, dst_stops))
    out[dst] = arr[src]
    return out


def finite_or_zero(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)


def summarize_vector(values: np.ndarray) -> tuple[float, float, float, float, float]:
    arr = finite_or_zero(values).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(arr.mean()),
        float(arr.std()),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
    )
