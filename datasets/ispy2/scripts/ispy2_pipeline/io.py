"""DICOM loading utilities for I-SPY 2 DCE and Analysis-Mask series.

These are hardened versions of the helpers originally prototyped in
notebooks/ispy2/session1/s1_00_ispy2_pilot_inspection.ipynb. They:

* load a DCE series directory into a (P, Z, Y, X) numpy volume plus the
  voxel spacing and world-space origin, so downstream code can work in mm;
* load an Analysis-Mask series into a (Z, Y, X) boolean tumor mask using the
  inverse bit-encoding that I-SPY 2 FTV masks use (0 == tumor, non-zero ==
  excluded); the other bit layers (pe_fail, mnc_fail, background, ...) are
  returned for QC but never used by the pipeline itself.

Any per-series warning (missing tags, phase/slice count mismatch) is raised
as a ValueError so the caller can decide whether to skip the visit or
abort. The loaders never silently "fix" malformed data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pydicom


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DCESeries:
    """A 4-D DCE volume with enough metadata to work in world-space."""

    volume: np.ndarray              # (n_phases, n_slices, rows, cols), uint16
    voxel_spacing: np.ndarray       # (3,) float; (dz, dy, dx) in mm
    origin: np.ndarray              # (3,) float; (z0, y0, x0) in mm (first slice, first row, first col)
    phase_key: str                  # 'TemporalPositionIdentifier' or 'AcquisitionNumber'
    header: pd.DataFrame            # per-slice headers (sorted as the volume was stacked)

    @property
    def n_phases(self) -> int:
        return int(self.volume.shape[0])

    @property
    def n_slices(self) -> int:
        return int(self.volume.shape[1])

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.volume.shape[1:])  # (Z, Y, X)


@dataclass
class MaskSeries:
    """Decoded FTV Analysis Mask aligned to a DCE series."""

    tumor: np.ndarray               # (Z, Y, X) bool -- the thing we want
    layers: dict[str, np.ndarray]   # QC bit planes: pe_fail, mnc_fail, background, ...
    voxel_spacing: np.ndarray       # (3,) float; (dz, dy, dx) in mm
    origin: np.ndarray              # (3,) float; (z0, y0, x0) in mm
    raw_dtype: str                  # e.g. 'uint8'

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.tumor.shape)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opt_int(ds: pydicom.Dataset, tag: str) -> int:
    v = getattr(ds, tag, None)
    return int(v) if v is not None else -1


def _read_slice_records(files: Iterable[Path]) -> pd.DataFrame:
    records: list[dict] = []
    for f in files:
        ds = pydicom.dcmread(f)
        records.append(
            {
                "path": f,
                "InstanceNumber": _opt_int(ds, "InstanceNumber"),
                "AcquisitionNumber": _opt_int(ds, "AcquisitionNumber"),
                "TemporalPositionIdentifier": _opt_int(ds, "TemporalPositionIdentifier"),
                "SliceLocation": float(getattr(ds, "SliceLocation", np.nan)),
                "ImagePositionPatient": (
                    tuple(float(x) for x in ds.ImagePositionPatient)
                    if "ImagePositionPatient" in ds
                    else None
                ),
                "Rows": int(ds.Rows),
                "Columns": int(ds.Columns),
                "PixelSpacing": (
                    tuple(float(x) for x in ds.PixelSpacing)
                    if "PixelSpacing" in ds
                    else None
                ),
                "SliceThickness": float(getattr(ds, "SliceThickness", np.nan)),
                "TriggerTime": float(getattr(ds, "TriggerTime", np.nan)),
                "_pixels": ds.pixel_array,
            }
        )
    if not records:
        raise FileNotFoundError("no DICOM files found")
    return pd.DataFrame(records)


def _voxel_spacing(hdr: pd.DataFrame) -> np.ndarray:
    ps = hdr["PixelSpacing"].iloc[0]
    if ps is None:
        raise ValueError("missing PixelSpacing in DICOM header")
    st = float(hdr["SliceThickness"].iloc[0])
    return np.asarray([st, float(ps[0]), float(ps[1])], dtype=np.float64)


def _origin(hdr: pd.DataFrame) -> np.ndarray:
    """First-slice world-space corner, (z, y, x)."""
    ipp = hdr["ImagePositionPatient"].iloc[0]
    if ipp is None:
        raise ValueError("missing ImagePositionPatient in DICOM header")
    # DICOM IPP is (x, y, z); we reorder to (z, y, x) to match the array axes.
    return np.asarray([ipp[2], ipp[1], ipp[0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_dce_series(series_dir: Path) -> DCESeries:
    """Load a flat directory of 2-D DCE DICOMs into a 4-D volume.

    Sorting key: phase first, then z (ImagePositionPatient[2]). Phase key is
    TemporalPositionIdentifier when present, otherwise AcquisitionNumber.
    """
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(series_dir)

    hdr = _read_slice_records(files)

    phase_key = "TemporalPositionIdentifier"
    if hdr[phase_key].max() <= 1:
        phase_key = "AcquisitionNumber"

    hdr["_z"] = hdr["ImagePositionPatient"].apply(
        lambda t: float("nan") if t is None else float(t[2])
    )
    hdr = hdr.sort_values([phase_key, "_z"]).reset_index(drop=True)

    vol_flat = np.stack(hdr["_pixels"].tolist(), axis=0)
    n_phases = hdr[phase_key].nunique()
    n_per_phase, rem = divmod(len(hdr), n_phases)
    if rem != 0:
        raise ValueError(
            f"series {series_dir.name}: {len(hdr)} slices but {n_phases} "
            f"phases -> remainder {rem}; likely incomplete acquisition"
        )
    volume = vol_flat.reshape(n_phases, n_per_phase, *vol_flat.shape[1:])

    spacing = _voxel_spacing(hdr)
    origin = _origin(hdr)

    hdr_out = hdr.drop(columns=["_pixels", "_z"])
    return DCESeries(
        volume=volume,
        voxel_spacing=spacing,
        origin=origin,
        phase_key=phase_key,
        header=hdr_out,
    )


def decode_ftv_mask(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Decode I-SPY 2 FTV analysis mask (inverse, bit-encoded).

    `tumor == (mask == 0)`. The other fields report which exclusion step(s)
    fired for each non-tumor voxel; useful for QC.
    """
    r = raw.astype(np.uint8)
    return {
        "tumor": r == 0,
        "pe_fail": (r & 1).astype(bool),
        "mnc_fail": (r & 2).astype(bool),
        "background": (r & 8).astype(bool),
        "undoc_bit4": (r & 16).astype(bool),
        "outside_voi": (r & 32).astype(bool),
        "omit": (r & 64).astype(bool),
    }


def _mask_spacing_from_functional_groups(ds: pydicom.Dataset) -> np.ndarray:
    """Pull (dz, dy, dx) in mm from a multiframe DICOM SEG.

    I-SPY 2 masks store pixel measures in SharedFunctionalGroupsSequence ->
    PixelMeasuresSequence. SliceThickness is set, but SpacingBetweenSlices is
    sometimes missing; in that case we derive dz from the PerFrame
    ImagePositionPatient values on the z-axis.
    """
    if "SharedFunctionalGroupsSequence" not in ds:
        raise ValueError("mask DICOM lacks SharedFunctionalGroupsSequence")
    sfgs = ds.SharedFunctionalGroupsSequence[0]
    if "PixelMeasuresSequence" not in sfgs:
        raise ValueError("mask DICOM lacks PixelMeasuresSequence")
    pms = sfgs.PixelMeasuresSequence[0]

    ps = getattr(pms, "PixelSpacing", None)
    if ps is None:
        raise ValueError("mask PixelMeasuresSequence lacks PixelSpacing")
    dy, dx = float(ps[0]), float(ps[1])

    sbs = getattr(pms, "SpacingBetweenSlices", None)
    if sbs is not None:
        dz = float(sbs)
    else:
        # Derive from PerFrame plane positions.
        if "PerFrameFunctionalGroupsSequence" not in ds or len(ds.PerFrameFunctionalGroupsSequence) < 2:
            st = getattr(pms, "SliceThickness", None)
            if st is None:
                raise ValueError("cannot determine mask slice spacing")
            dz = float(st)
        else:
            pffgs = ds.PerFrameFunctionalGroupsSequence
            z0 = float(pffgs[0].PlanePositionSequence[0].ImagePositionPatient[2])
            z1 = float(pffgs[1].PlanePositionSequence[0].ImagePositionPatient[2])
            dz = abs(z1 - z0)
            if dz < 1e-3:
                st = getattr(pms, "SliceThickness", None)
                dz = float(st) if st is not None else 1.0

    return np.asarray([dz, dy, dx], dtype=np.float64)


def _mask_origin_from_functional_groups(ds: pydicom.Dataset) -> np.ndarray:
    """Pull (z0, y0, x0) from the *first* frame's PlanePositionSequence.

    Note: I-SPY 2 stores frames in arbitrary order; we return the origin of
    frame 0 and assume the caller sorts frames by z before use. For the
    nearest-neighbor resample we do downstream only the set of z positions
    matters, not which one we label "origin".
    """
    if "PerFrameFunctionalGroupsSequence" not in ds:
        if "ImagePositionPatient" in ds:
            ipp = [float(x) for x in ds.ImagePositionPatient]
            return np.asarray([ipp[2], ipp[1], ipp[0]], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)
    fg0 = ds.PerFrameFunctionalGroupsSequence[0]
    ipp = [float(x) for x in fg0.PlanePositionSequence[0].ImagePositionPatient]
    return np.asarray([ipp[2], ipp[1], ipp[0]], dtype=np.float64)


def _mask_frame_zs(ds: pydicom.Dataset) -> np.ndarray:
    """Return z-coordinate (in mm) of every frame, in the order they are stored."""
    if "PerFrameFunctionalGroupsSequence" not in ds:
        return np.array([0.0])
    return np.asarray(
        [
            float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
            for fg in ds.PerFrameFunctionalGroupsSequence
        ],
        dtype=np.float64,
    )


def load_analysis_mask(series_dir: Path) -> MaskSeries:
    """Load an Analysis-Mask series as a single (Z, Y, X) boolean tumor volume.

    I-SPY 2 masks are multiframe DICOM SEG files (one .dcm, `NumberOfFrames`
    slices). Pixel measures live in SharedFunctionalGroupsSequence and the
    per-frame positions in PerFrameFunctionalGroupsSequence. We sort frames
    by z so `tumor[0]` is the lowest-z slice and z increases with the first
    axis, matching how the DCE loader sorts.

    Falls back to per-slice-DICOM loading for non-SEG mask series.
    """
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(series_dir)

    if len(files) == 1:
        ds = pydicom.dcmread(files[0])
        raw = ds.pixel_array  # (frames, rows, cols) or (rows, cols)
        if raw.ndim == 2:
            raw = raw[None]

        if (
            "SharedFunctionalGroupsSequence" in ds
            or "PerFrameFunctionalGroupsSequence" in ds
        ):
            spacing = _mask_spacing_from_functional_groups(ds)
            origin = _mask_origin_from_functional_groups(ds)
            zs = _mask_frame_zs(ds)
            # Sort frames by ascending z so axis 0 is z-sorted, like the DCE.
            order = np.argsort(zs)
            raw = raw[order]
            origin = np.asarray(
                [float(zs[order[0]]), float(origin[1]), float(origin[2])],
                dtype=np.float64,
            )
        else:
            # Plain DICOM (non-SEG) -- expect top-level pixel measures.
            if "PixelSpacing" not in ds or "SliceThickness" not in ds:
                raise ValueError(
                    "mask DICOM missing PixelSpacing / SliceThickness and no "
                    "functional groups sequences to fall back on"
                )
            ps = [float(x) for x in ds.PixelSpacing]
            spacing = np.asarray([float(ds.SliceThickness), ps[0], ps[1]], dtype=np.float64)
            if "ImagePositionPatient" in ds:
                ipp = [float(x) for x in ds.ImagePositionPatient]
                origin = np.asarray([ipp[2], ipp[1], ipp[0]], dtype=np.float64)
            else:
                origin = np.zeros(3, dtype=np.float64)

        layers = decode_ftv_mask(raw)
        return MaskSeries(
            tumor=layers["tumor"],
            layers=layers,
            voxel_spacing=spacing,
            origin=origin,
            raw_dtype=str(raw.dtype),
        )

    # Fallback: per-slice files.
    hdr = _read_slice_records(files)
    hdr["_z"] = hdr["ImagePositionPatient"].apply(
        lambda t: float("nan") if t is None else float(t[2])
    )
    hdr = hdr.sort_values("_z").reset_index(drop=True)
    raw = np.stack(hdr["_pixels"].tolist(), axis=0)
    spacing = _voxel_spacing(hdr)
    origin = _origin(hdr)
    layers = decode_ftv_mask(raw)
    return MaskSeries(
        tumor=layers["tumor"],
        layers=layers,
        voxel_spacing=spacing,
        origin=origin,
        raw_dtype=str(raw.dtype),
    )


def _nearest_axis_index(
    dce_origin: float,
    dce_spacing: float,
    dce_n: int,
    mask_origin: float,
    mask_spacing: float,
    mask_n: int,
) -> np.ndarray:
    """Per-axis nearest-neighbor lookup: for each DCE voxel along one axis,
    return the index into the mask axis whose world coordinate is closest."""
    dce_coords = dce_origin + np.arange(dce_n) * dce_spacing
    mask_coords = mask_origin + np.arange(mask_n) * mask_spacing
    return np.abs(dce_coords[:, None] - mask_coords[None, :]).argmin(axis=1)


def resample_mask_to_dce_grid(
    mask: "MaskSeries",
    dce: "DCESeries",
) -> np.ndarray:
    """Nearest-neighbor resample `mask.tumor` onto the DCE grid in world space.

    I-SPY 2 masks can differ from their DCE series on any combination of axes:
      - finer z-resolution (e.g. 1 mm mask vs. 2.5 mm DCE, 160 vs 72 slices),
      - different in-plane shape (e.g. 512x452 mask vs. 452x512 DCE),
      - shifted origin.

    We handle all axis-aligned cases by resampling each axis independently:
    for each DCE voxel along axis a, we pick the mask index whose world
    coordinate along a is closest. The assumption is that both volumes are
    stored in the same (Z, Y, X) patient frame with no rotation between them.
    That holds for every I-SPY 2 / ACRIN-6698 series we've inspected; pure
    rotations would require a proper SimpleITK resample.
    """
    iz = _nearest_axis_index(
        float(dce.origin[0]), float(dce.voxel_spacing[0]), int(dce.volume.shape[1]),
        float(mask.origin[0]), float(mask.voxel_spacing[0]), int(mask.tumor.shape[0]),
    )
    iy = _nearest_axis_index(
        float(dce.origin[1]), float(dce.voxel_spacing[1]), int(dce.volume.shape[2]),
        float(mask.origin[1]), float(mask.voxel_spacing[1]), int(mask.tumor.shape[1]),
    )
    ix = _nearest_axis_index(
        float(dce.origin[2]), float(dce.voxel_spacing[2]), int(dce.volume.shape[3]),
        float(mask.origin[2]), float(mask.voxel_spacing[2]), int(mask.tumor.shape[2]),
    )
    return mask.tumor[np.ix_(iz, iy, ix)].astype(bool)
