#!/usr/bin/env python3
"""Extract downloaded UCSF DCE feature maps into compact per-visit arrays.

The feature-map download is intentionally kept separate from the original
minimal DCE/mask download. This script reads map DICOM series from:

    datasets/ispy2/raw_expanded/feature_maps/<PID>/<T*>/<SeriesDescription>/<UID>/

and writes compact arrays under:

    datasets/ispy2/derived_rich_maps/<PID>/<T*>/feature_maps.npz
    datasets/ispy2/derived_rich_maps/<PID>/<T*>/feature_maps_meta.json

The output is aligned to the existing derived DCE grid by nearest-neighbor
resampling, using the reference shape, spacing, and origin from the existing
`derived`/`derived_v2` visit metadata. Existing derived data are read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


VISITS = ("T0", "T1", "T2", "T3")
MAP_KEYS = ("ucsf_pe2", "ucsf_pe5", "ucsf_pe6", "ucsf_ser")


def _slug_name(path: Path) -> str:
    return path.name.lower()


def _has_map_token(series_name: str, map_token: str) -> bool:
    """Match map tokens as slug parts so SER does not match the VOLSER prefix."""
    parts = series_name.replace("-", "_").split("_")
    return map_token.lower() in parts


def _candidate_series_dirs(search_dir: Path, map_token: str) -> list[Path]:
    if not search_dir.is_dir():
        return []
    out: list[Path] = []
    for parent in sorted(search_dir.iterdir()):
        if not parent.is_dir():
            continue
        name = _slug_name(parent)
        if not _has_map_token(name, map_token):
            continue
        subdirs = [p for p in sorted(parent.iterdir()) if p.is_dir()]
        out.extend(subdirs)
    return out


def _choose_series(search_dirs: list[Path], map_token: str, uid_to_visit: dict[str, str], visit: str) -> Path | None:
    candidates: list[Path] = []
    for search_dir in search_dirs:
        for series_dir in _candidate_series_dirs(search_dir, map_token):
            if uid_to_visit:
                assigned_visit = uid_to_visit.get(series_dir.name)
                if assigned_visit != visit:
                    continue
            candidates.append(series_dir)
    if not candidates:
        return None

    # Prefer unilateral cropped maps because the current tumor graph is based on
    # the cropped DCE/mask package when available. Fall back to bilateral.
    def priority(path: Path) -> tuple[int, str]:
        full = str(path.parent).lower()
        if "uni_lateral_cropped" in full:
            return (0, str(path))
        if "bi_lateral" in full:
            return (1, str(path))
        return (2, str(path))

    return sorted(candidates, key=priority)[0]


def _parse_date(value) -> datetime | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    for fmt in ("%m-%d-%Y", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def load_uid_visit_map(manifest_path: Path | None) -> dict[str, dict[str, str]]:
    """Infer visit labels for feature-map series when TCIA lacks Timepoint."""
    if manifest_path is None or not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    required = {"PatientID", "SeriesInstanceUID", "StudyDate"}
    if not required.issubset(df.columns):
        return {}

    visit_by_pid_uid: dict[str, dict[str, str]] = {}
    for pid, grp in df.groupby(df["PatientID"].astype(str)):
        if "Timepoint" in grp.columns and grp["Timepoint"].notna().any():
            mapping = {
                str(row.SeriesInstanceUID): str(row.Timepoint)
                for row in grp.itertuples(index=False)
                if str(getattr(row, "Timepoint", "")).startswith("T")
            }
            if mapping:
                visit_by_pid_uid[pid] = mapping
                continue

        date_by_uid: dict[str, datetime] = {}
        for row in grp.itertuples(index=False):
            dt = _parse_date(getattr(row, "StudyDate", None))
            if dt is None:
                dt = _parse_date(getattr(row, "SeriesDate", None))
            if dt is not None:
                date_by_uid[str(row.SeriesInstanceUID)] = dt
        sorted_dates = sorted(set(date_by_uid.values()))
        date_to_visit = {dt: VISITS[i] for i, dt in enumerate(sorted_dates[: len(VISITS)])}
        visit_by_pid_uid[pid] = {
            uid: date_to_visit[dt]
            for uid, dt in date_by_uid.items()
            if dt in date_to_visit
        }
    return visit_by_pid_uid


def _read_scalar_dicom_volume(series_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(series_dir)

    frames: list[tuple[float, np.ndarray]] = []
    spacing_yx: tuple[float, float] | None = None
    slice_thickness = 1.0
    origin_yx: tuple[float, float] | None = None

    for file in files:
        ds = pydicom.dcmread(file)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            x0, y0, z0 = (float(v) for v in ipp)
        else:
            x0, y0, z0 = 0.0, 0.0, float(getattr(ds, "SliceLocation", 0.0))
        frames.append((z0, arr))

        if spacing_yx is None and getattr(ds, "PixelSpacing", None) is not None:
            spacing_yx = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
        if origin_yx is None:
            origin_yx = (y0, x0)
        if getattr(ds, "SliceThickness", None) is not None:
            slice_thickness = float(ds.SliceThickness)

    frames.sort(key=lambda item: item[0])
    vol = np.stack([arr for _, arr in frames], axis=0).astype(np.float32)
    if len(frames) > 1:
        dz = float(abs(frames[1][0] - frames[0][0]))
        if dz < 1e-6:
            dz = slice_thickness
    else:
        dz = slice_thickness
    sy, sx = spacing_yx if spacing_yx is not None else (1.0, 1.0)
    oy, ox = origin_yx if origin_yx is not None else (0.0, 0.0)
    origin = np.asarray([frames[0][0], oy, ox], dtype=np.float64)
    spacing = np.asarray([dz, sy, sx], dtype=np.float64)
    return vol, spacing, origin


def _nearest_axis_idx(
    ref0: float,
    ref_step: float,
    ref_n: int,
    src0: float,
    src_step: float,
    src_n: int,
) -> np.ndarray:
    ref_coords = ref0 + np.arange(ref_n, dtype=np.float64) * ref_step
    src_coords = src0 + np.arange(src_n, dtype=np.float64) * src_step
    return np.abs(ref_coords[:, None] - src_coords[None, :]).argmin(axis=1)


def _resample_to_reference(
    src: np.ndarray,
    src_spacing: np.ndarray,
    src_origin: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_spacing: np.ndarray,
    ref_origin: np.ndarray,
) -> np.ndarray:
    iz = _nearest_axis_idx(ref_origin[0], ref_spacing[0], ref_shape[0], src_origin[0], src_spacing[0], src.shape[0])
    iy = _nearest_axis_idx(ref_origin[1], ref_spacing[1], ref_shape[1], src_origin[1], src_spacing[1], src.shape[1])
    ix = _nearest_axis_idx(ref_origin[2], ref_spacing[2], ref_shape[2], src_origin[2], src_spacing[2], src.shape[2])
    return src[np.ix_(iz, iy, ix)].astype(np.float32)


def _reference_grid(base_visit_dir: Path) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray]:
    meta = json.loads((base_visit_dir / "meta.json").read_text())
    pe_ser = np.load(base_visit_dir / "pe_ser.npz")
    shape = tuple(int(v) for v in pe_ser["pe"].shape)
    spacing = np.asarray(meta["voxel_spacing_mm"], dtype=np.float64)
    origin = np.asarray(meta["origin_mm"], dtype=np.float64)
    return shape, spacing, origin


def process_visit(
    pid: str,
    visit: str,
    raw_maps_root: Path,
    base_derived: Path,
    out_root: Path,
    uid_visit_map: dict[str, dict[str, str]],
) -> dict:
    base_visit = base_derived / pid / visit
    if not (base_visit / "meta.json").exists() or not (base_visit / "pe_ser.npz").exists():
        return {"patient_id": pid, "visit": visit, "status": "missing_base_derived"}

    search_dirs = [raw_maps_root / pid / visit]
    tunk_dir = raw_maps_root / pid / "Tunk"
    if tunk_dir not in search_dirs:
        search_dirs.append(tunk_dir)
    out_visit = out_root / pid / visit
    out_npz = out_visit / "feature_maps.npz"
    out_meta = out_visit / "feature_maps_meta.json"

    ref_shape, ref_spacing, ref_origin = _reference_grid(base_visit)
    zero = np.zeros(ref_shape, dtype=np.float32)
    maps: dict[str, np.ndarray] = {}
    present: dict[str, bool] = {}
    sources: dict[str, str | None] = {}

    map_specs = {
        "ucsf_pe2": "pe2",
        "ucsf_pe5": "pe5",
        "ucsf_pe6": "pe6",
        "ucsf_ser": "ser",
    }

    for key, map_token in map_specs.items():
        series_dir = _choose_series(search_dirs, map_token, uid_visit_map.get(pid, {}), visit)
        if series_dir is None:
            maps[key] = zero.copy()
            present[key] = False
            sources[key] = None
            continue
        try:
            vol, spacing, origin = _read_scalar_dicom_volume(series_dir)
            maps[key] = _resample_to_reference(vol, spacing, origin, ref_shape, ref_spacing, ref_origin)
            present[key] = True
            sources[key] = str(series_dir)
        except Exception as exc:  # noqa: BLE001
            maps[key] = zero.copy()
            present[key] = False
            sources[key] = f"ERROR {type(exc).__name__}: {exc}"

    out_visit.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        **maps,
        present=np.asarray([present[key] for key in MAP_KEYS], dtype=np.uint8),
        map_keys=np.asarray(MAP_KEYS),
        ref_shape=np.asarray(ref_shape, dtype=np.int32),
        ref_spacing=ref_spacing.astype(np.float32),
        ref_origin=ref_origin.astype(np.float32),
    )
    out_meta.write_text(
        json.dumps(
            {
                "patient_id": pid,
                "visit": visit,
                "map_keys": list(MAP_KEYS),
                "present": present,
                "sources": sources,
                "ref_shape": list(ref_shape),
                "ref_spacing_mm": ref_spacing.tolist(),
                "ref_origin_mm": ref_origin.tolist(),
            },
            indent=2,
        )
    )
    return {
        "patient_id": pid,
        "visit": visit,
        "status": "ok",
        **{f"{key}_present": present[key] for key in MAP_KEYS},
    }


def iter_patients(args: argparse.Namespace) -> list[str]:
    if args.patient:
        return [args.patient]
    if args.patient_list:
        return [ln.strip() for ln in args.patient_list.read_text().splitlines() if ln.strip()]
    return sorted(p.name for p in args.base_derived.iterdir() if p.is_dir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--patient-list", type=Path, default=None)
    ap.add_argument("--raw-maps-root", type=Path, default=Path("datasets/ispy2/raw_expanded/feature_maps"))
    ap.add_argument("--base-derived", type=Path, default=Path("datasets/ispy2/derived"))
    ap.add_argument("--out-root", type=Path, default=Path("datasets/ispy2/derived_rich_maps"))
    ap.add_argument(
        "--series-manifest",
        type=Path,
        default=Path("reports/ispy2_download_scope_audit/manifests/feature_maps_manifest.csv"),
        help="Feature-map manifest used to infer T0/T1/T2/T3 from StudyDate when downloads are under Tunk.",
    )
    ap.add_argument("--array-index", type=int, default=None)
    ap.add_argument("--array-count", type=int, default=None)
    args = ap.parse_args()

    patients = iter_patients(args)
    if args.array_index is not None and args.array_count:
        patients = patients[args.array_index :: args.array_count]

    args.out_root.mkdir(parents=True, exist_ok=True)
    uid_visit_map = load_uid_visit_map(args.series_manifest)
    all_rows = []
    n_ok = 0
    for pid in patients:
        print(f"[{pid}]")
        for visit in VISITS:
            row = process_visit(pid, visit, args.raw_maps_root, args.base_derived, args.out_root, uid_visit_map)
            all_rows.append(row)
            print(" ", visit, row["status"], {k: row.get(f"{k}_present") for k in MAP_KEYS if f"{k}_present" in row})
            if row["status"] == "ok":
                n_ok += 1

    log_path = args.out_root / "_feature_map_extract_log.jsonl"
    with log_path.open("a") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"processed_patients={len(patients)} ok_visits={n_ok} log={log_path}")
    return 0 if n_ok > 0 or not patients else 1


if __name__ == "__main__":
    raise SystemExit(main())
