"""Tumor-masked SLIC supervoxels and per-supervoxel feature aggregation.

Given:
    * a per-visit tumor mask (Z, Y, X) bool
    * a PE map (Z, Y, X) float (used as the SLIC intensity channel)
    * a SER map and the full (P, Z, Y, X) DCE stack (for feature aggregation)
    * voxel spacing and origin in mm

we produce a set of 3-D SLIC supervoxels confined to the mask, then
aggregate per-supervoxel features (voxel count, volume in mL, centroid in mm,
mean PE, mean SER, per-phase normalized-enhancement curve).

SLIC is applied only inside the bounding box of the mask with a mask-aware
parameterization: n_segments is derived from the target voxel count per
supervoxel so tumor size and SLIC granularity stay coupled (small tumors get
fewer segments, big tumors get more), which is what we want for graphs with
a roughly-consistent node degree across patients.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from skimage.segmentation import slic


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Supervoxels:
    """Output of `slic_tumor_supervoxels`."""

    labels: np.ndarray            # (Z, Y, X) int32; 0 == outside tumor, >=1 == supervoxel id
    n_supervoxels: int            # number of distinct >=1 labels
    target_voxels: int            # requested voxels-per-supervoxel
    compactness: float            # SLIC compactness parameter used


# ---------------------------------------------------------------------------
# SLIC
# ---------------------------------------------------------------------------


def slic_tumor_supervoxels(
    pe_map: np.ndarray,
    tumor_mask: np.ndarray,
    voxel_spacing: np.ndarray,
    *,
    target_voxels_per_sv: int = 150,
    min_n_segments: int = 20,
    max_n_segments: int = 600,
    compactness: float = 0.10,
    sigma: float = 1.0,
) -> Supervoxels:
    """SLIC on the PE map, confined to the tumor mask.

    The mask is tight-cropped for SLIC to avoid wasting segments on the
    background volume, then the resulting labels are pasted back into the
    full-size label volume.

    Returns a `Supervoxels` with labels in {0, 1, ..., n_supervoxels};
    label 0 is reserved for "outside tumor".
    """
    if pe_map.shape != tumor_mask.shape:
        raise ValueError(
            f"pe_map {pe_map.shape} and tumor_mask {tumor_mask.shape} must match"
        )
    if tumor_mask.dtype != bool:
        tumor_mask = tumor_mask.astype(bool)
    if not tumor_mask.any():
        return Supervoxels(
            labels=np.zeros_like(pe_map, dtype=np.int32),
            n_supervoxels=0,
            target_voxels=target_voxels_per_sv,
            compactness=compactness,
        )

    n_tumor = int(tumor_mask.sum())
    # Scale n_seg by tumor size, but never create segments smaller than
    # `min_voxels_per_sv` -- 1-voxel supervoxels are meaningless for feature
    # aggregation. The `min_n_segments` floor only applies if the tumor is
    # big enough that each segment would still hold ~target_voxels_per_sv/2
    # voxels; tiny post-treatment tumors stay at 1-2 supervoxels, which is
    # the honest answer.
    min_voxels_per_sv = max(target_voxels_per_sv // 4, 4)
    ideal = round(n_tumor / max(target_voxels_per_sv, 1))
    n_seg = max(1, ideal)
    if n_tumor >= min_n_segments * (target_voxels_per_sv // 2 or 1):
        n_seg = max(n_seg, min_n_segments)
    n_seg = min(n_seg, max_n_segments)
    # Hard cap: never ask for more segments than we have usable voxels.
    n_seg = min(n_seg, max(1, n_tumor // min_voxels_per_sv))

    # Tight crop for SLIC (faster and keeps segments inside the tumor).
    zs, ys, xs = np.where(tumor_mask)
    z0, z1 = zs.min(), zs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    pe_crop = pe_map[z0:z1, y0:y1, x0:x1]
    mask_crop = tumor_mask[z0:z1, y0:y1, x0:x1]

    # Normalize PE into [0, 1] inside the mask before SLIC so compactness is
    # interpretable across patients with wildly different PE ranges.
    pe_in = pe_crop[mask_crop]
    lo, hi = float(np.percentile(pe_in, 1)), float(np.percentile(pe_in, 99))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    pe_norm = np.clip((pe_crop - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    # SLIC in 3-D. `spacing` tells SLIC the relative scale of each axis so
    # voxel anisotropy (typ. 2.5 mm slice thickness vs ~1 mm in-plane) is
    # respected. skimage's SLIC returns an all-zero label volume when
    # `n_segments == 1`, so short-circuit that case: paint the mask with
    # label 1 and be done.
    if n_seg <= 1:
        labels_crop = mask_crop.astype(np.int32)
    else:
        labels_crop = slic(
            pe_norm,
            n_segments=n_seg,
            compactness=compactness,
            sigma=sigma,
            spacing=tuple(voxel_spacing.tolist()),
            mask=mask_crop,
            channel_axis=None,
            start_label=1,
        ).astype(np.int32)
    # skimage assigns 0 to the "unmasked / background" region already.

    labels = np.zeros_like(pe_map, dtype=np.int32)
    labels[z0:z1, y0:y1, x0:x1] = labels_crop
    uniq = np.unique(labels)
    n_sv = int((uniq > 0).sum())

    return Supervoxels(
        labels=labels,
        n_supervoxels=n_sv,
        target_voxels=target_voxels_per_sv,
        compactness=compactness,
    )


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------


def _voxel_to_mm(coord_vox: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Convert (N, 3) voxel indices (z, y, x) -> (N, 3) positions in mm (z, y, x)."""
    return coord_vox.astype(np.float64) * spacing[None, :] + origin[None, :]


def _safe_distribution_stats(values: np.ndarray) -> tuple[float, float, float]:
    """Return finite-value std, skew, and excess kurtosis."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0, 0.0
    std = float(v.std())
    if std < 1e-8:
        return 0.0, 0.0, 0.0
    centered = (v - float(v.mean())) / std
    skew = float(np.mean(centered ** 3))
    kurtosis = float(np.mean(centered ** 4) - 3.0)
    return std, skew, kurtosis


def _surface_area_mm2(mask: np.ndarray, spacing: np.ndarray) -> float:
    """Approximate binary-object surface area by counting exposed voxel faces."""
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    dz = np.count_nonzero(np.diff(padded, axis=0))
    dy = np.count_nonzero(np.diff(padded, axis=1))
    dx = np.count_nonzero(np.diff(padded, axis=2))
    sz, sy, sx = [float(v) for v in spacing]
    return float(dz * sy * sx + dy * sz * sx + dx * sz * sy)


def _shape_features(
    mask_sv: np.ndarray,
    voxel_spacing: np.ndarray,
    voxel_count: int,
    voxel_volume_ml: float,
) -> dict[str, float]:
    """Compact physical shape descriptors for one supervoxel."""
    coords = np.column_stack(np.where(mask_sv)).astype(np.float64)
    if coords.size == 0:
        return {
            "surface_area_mm2": 0.0,
            "sphericity": 0.0,
            "elongation": 1.0,
            "major_axis_length_mm": 0.0,
            "minor_axis_length_mm": 0.0,
        }

    extents_mm = (coords.max(axis=0) - coords.min(axis=0) + 1.0) * voxel_spacing
    major = float(extents_mm.max())
    minor = float(max(extents_mm.min(), 1e-6))
    elongation = major / minor

    surface_area = _surface_area_mm2(mask_sv, voxel_spacing)
    volume_mm3 = voxel_count * voxel_volume_ml * 1000.0
    if surface_area <= 0.0 or volume_mm3 <= 0.0:
        sphericity = 0.0
    else:
        sphericity = float(
            (np.pi ** (1.0 / 3.0))
            * ((6.0 * volume_mm3) ** (2.0 / 3.0))
            / surface_area
        )

    return {
        "surface_area_mm2": surface_area,
        "sphericity": sphericity,
        "elongation": float(elongation),
        "major_axis_length_mm": major,
        "minor_axis_length_mm": minor,
    }


def _read_dicom_volume(series_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(series_dir)
    frames: list[tuple[float, np.ndarray]] = []
    spacing_yx: tuple[float, float] | None = None
    slice_thickness = 1.0
    for f in files:
        ds = pydicom.dcmread(f)
        px = ds.pixel_array.astype(np.float32)
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None else float(getattr(ds, "SliceLocation", 0.0))
        frames.append((z, px))
        if spacing_yx is None and getattr(ds, "PixelSpacing", None) is not None:
            spacing_yx = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
        if getattr(ds, "SliceThickness", None) is not None:
            slice_thickness = float(ds.SliceThickness)
    frames.sort(key=lambda x: x[0])
    vol = np.stack([px for _, px in frames], axis=0)
    if len(frames) > 1:
        dz = float(abs(frames[1][0] - frames[0][0]))
        if dz < 1e-6:
            dz = slice_thickness
    else:
        dz = slice_thickness
    sy, sx = spacing_yx if spacing_yx is not None else (1.0, 1.0)
    spacing = np.asarray([dz, sy, sx], dtype=np.float64)
    origin = np.asarray([frames[0][0], 0.0, 0.0], dtype=np.float64)
    return vol, spacing, origin


def _resample_to_reference(
    src: np.ndarray,
    src_spacing: np.ndarray,
    src_origin: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_spacing: np.ndarray,
    ref_origin: np.ndarray,
) -> np.ndarray:
    def nearest_axis_idx(
        ref0: float,
        ref_step: float,
        ref_n: int,
        src0: float,
        src_step: float,
        src_n: int,
    ) -> np.ndarray:
        ref_coords = ref0 + np.arange(ref_n) * ref_step
        src_coords = src0 + np.arange(src_n) * src_step
        return np.abs(ref_coords[:, None] - src_coords[None, :]).argmin(axis=1)

    iz = nearest_axis_idx(ref_origin[0], ref_spacing[0], ref_shape[0], src_origin[0], src_spacing[0], src.shape[0])
    iy = nearest_axis_idx(ref_origin[1], ref_spacing[1], ref_shape[1], src_origin[1], src_spacing[1], src.shape[1])
    ix = nearest_axis_idx(ref_origin[2], ref_spacing[2], ref_shape[2], src_origin[2], src_spacing[2], src.shape[2])
    return src[np.ix_(iz, iy, ix)].astype(np.float32)


def load_ucsf_maps(
    visit_dir: Path,
    *,
    ref_shape: tuple[int, int, int],
    ref_spacing: np.ndarray,
    ref_origin: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load UCSF VOLSER maps (PE/SER/FTV), resampled to the DCE grid."""
    if not visit_dir.is_dir():
        return None, None, None

    def first_series_with_token(tokens: tuple[str, ...]) -> Path | None:
        for p in sorted(visit_dir.iterdir()):
            if not p.is_dir():
                continue
            name = p.name
            if not any(tok in name for tok in tokens):
                continue
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if subdirs:
                return subdirs[0]
        return None

    pe_dir = first_series_with_token(("_PE2", "_PE5", "_PE6"))
    ser_dir = first_series_with_token(("_SER",))
    ftv_dir = first_series_with_token(("_FTV",))

    def load_one(series_dir: Path | None) -> np.ndarray | None:
        if series_dir is None:
            return None
        try:
            vol, spacing, origin = _read_dicom_volume(series_dir)
            return _resample_to_reference(
                vol, spacing, origin, ref_shape, ref_spacing, ref_origin
            )
        except Exception:
            return None

    return load_one(pe_dir), load_one(ser_dir), load_one(ftv_dir)


def aggregate_features(
    supervoxels: Supervoxels,
    dce: np.ndarray,
    pe_map: np.ndarray,
    ser_map: np.ndarray,
    voxel_spacing: np.ndarray,
    origin: np.ndarray,
    *,
    pre_phase: int = 0,
    pe_source: str = "recomputed",
    adc_map: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-supervoxel feature table.

    Columns:
        supervoxel_id          int
        voxel_count            int
        volume_ml              float
        centroid_{z,y,x}_mm    float  (3 cols)
        mean_pe                float
        mean_ser               float
        phase{p}_mean_enh      float  (one col per phase p = 0..P-1;
                                       phase{pre_phase}_mean_enh is 0 by construction)

    Returns an empty DataFrame with the right columns when n_supervoxels == 0.
    """
    P = int(dce.shape[0])
    phase_cols = [f"phase{p}_mean_enh" for p in range(P)]
    cols = [
        "supervoxel_id", "voxel_count", "volume_ml",
        "centroid_z_mm", "centroid_y_mm", "centroid_x_mm",
        "pe_source",
        "mean_pe", "pe_std", "pe_skew", "pe_kurtosis",
        "mean_ser", "ser_std", "ser_skew", "ser_kurtosis",
        "mean_adc", "adc_std", "adc_skew", "adc_kurtosis", "adc_missing",
        "surface_area_mm2", "sphericity", "elongation",
        "major_axis_length_mm", "minor_axis_length_mm",
        *phase_cols,
    ]
    if supervoxels.n_supervoxels == 0:
        return pd.DataFrame(columns=cols)

    labels = supervoxels.labels
    voxel_volume_ml = float(np.prod(voxel_spacing)) / 1000.0  # mm^3 -> mL

    # Precompute per-phase normalized enhancement to avoid reshaping P times.
    s_pre = dce[pre_phase].astype(np.float32)
    eps = 1e-3
    valid_pre = s_pre > eps

    rows: list[dict] = []
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    for sv_id in unique_ids:
        mask_sv = labels == sv_id
        n = int(mask_sv.sum())
        if n == 0:
            continue

        # Centroid in voxel units (z, y, x), then converted to mm.
        zs, ys, xs = np.where(mask_sv)
        centroid_vox = np.asarray([zs.mean(), ys.mean(), xs.mean()], dtype=np.float64)
        centroid_mm = centroid_vox * voxel_spacing + origin

        pe_values = pe_map[mask_sv]
        ser_values = ser_map[mask_sv]
        adc_values = None if adc_map is None else adc_map[mask_sv]
        mean_pe = float(np.nanmean(pe_values))
        mean_ser = float(np.nanmean(ser_values))
        pe_std, pe_skew, pe_kurtosis = _safe_distribution_stats(pe_values)
        ser_std, ser_skew, ser_kurtosis = _safe_distribution_stats(ser_values)
        if adc_values is None:
            mean_adc, adc_std, adc_skew, adc_kurtosis, adc_missing = 0.0, 0.0, 0.0, 0.0, 1.0
        else:
            mean_adc = float(np.nanmean(adc_values))
            adc_std, adc_skew, adc_kurtosis = _safe_distribution_stats(adc_values)
            adc_missing = 0.0
        shape = _shape_features(mask_sv, voxel_spacing, n, voxel_volume_ml)

        phase_means = []
        sv_valid = mask_sv & valid_pre
        if not sv_valid.any():
            phase_means = [0.0] * P
        else:
            s_pre_sv = s_pre[sv_valid]
            for p in range(P):
                s_p = dce[p].astype(np.float32)[sv_valid]
                phase_means.append(float(((s_p - s_pre_sv) / s_pre_sv).mean()))

        row = {
            "supervoxel_id": int(sv_id),
            "voxel_count": n,
            "volume_ml": n * voxel_volume_ml,
            "centroid_z_mm": float(centroid_mm[0]),
            "centroid_y_mm": float(centroid_mm[1]),
            "centroid_x_mm": float(centroid_mm[2]),
            "pe_source": pe_source,
            "mean_pe": mean_pe,
            "pe_std": pe_std,
            "pe_skew": pe_skew,
            "pe_kurtosis": pe_kurtosis,
            "mean_ser": mean_ser,
            "ser_std": ser_std,
            "ser_skew": ser_skew,
            "ser_kurtosis": ser_kurtosis,
            "mean_adc": mean_adc,
            "adc_std": adc_std,
            "adc_skew": adc_skew,
            "adc_kurtosis": adc_kurtosis,
            "adc_missing": adc_missing,
            **shape,
        }
        for p, v in enumerate(phase_means):
            row[f"phase{p}_mean_enh"] = v
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    return df
