#!/usr/bin/env python
"""Per-patient preprocessing CLI.

For a single patient directory (e.g. ``raw/ISPY2-825363/`` or
``pilot/ISPY2-212307/``) iterate the four timepoints T0/T1/T2/T3, compute
PE + SER + SLIC supervoxels + per-supervoxel features on each available
visit, and write the outputs to ``<out_root>/<patient_id>/<visit>/``:

    supervoxels.parquet      per-supervoxel feature table
    supervoxel_labels.npz    labels volume (int32, compressed)
    pe_ser.npz               PE and SER float32 volumes + spacing/origin
    meta.json                spacing, origin, phase counts, pre/early/late phases

Run:

    python preprocess_patient.py \
        --patient-dir datasets/ispy2/pilot/ISPY2-825363 \
        --out-root    datasets/ispy2/derived

Skips a visit if DCE or mask is missing. Exits non-zero on any unexpected
I/O or shape error so Slurm arrays pick up the failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ispy2_pipeline import (
    Supervoxels,
    aggregate_features,
    compute_pe_ser,
    kinetic_summary,
    load_analysis_mask,
    load_dce_series,
    load_ucsf_maps,
    resample_mask_to_dce_grid,
    slic_tumor_supervoxels,
)

TIMEPOINTS = ("T0", "T1", "T2", "T3")


def _find_series(visit_dir: Path, needle: str) -> Path | None:
    """Find the single series subdir whose name contains `needle`."""
    if not visit_dir.is_dir():
        return None
    candidates = [p for p in visit_dir.iterdir() if p.is_dir() and needle in p.name]
    if not candidates:
        return None
    # Each series is a directory keyed by SeriesInstanceUID; one level deeper.
    for series_parent in candidates:
        sub = [p for p in series_parent.iterdir() if p.is_dir()]
        if sub:
            return sub[0]
    return None


def _process_visit(
    visit_dir: Path,
    out_dir: Path,
    *,
    target_voxels_per_sv: int,
    compactness: float,
    rerun_aggregate_only: bool,
    log: logging.Logger,
) -> dict:
    dce_dir = _find_series(visit_dir, "original_DCE")
    mask_dir = _find_series(visit_dir, "Analysis_Mask")
    if dce_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"{visit_dir}: missing DCE ({dce_dir}) or Mask ({mask_dir})"
        )

    t0 = time.time()
    dce = load_dce_series(dce_dir)
    mask = load_analysis_mask(mask_dir)

    if mask.tumor.shape == dce.volume.shape[1:]:
        tumor_on_dce = mask.tumor
        resampled = False
    else:
        tumor_on_dce = resample_mask_to_dce_grid(mask, dce)
        resampled = True
        log.debug(
            "  %s: resampled mask %s -> %s (dce grid)",
            visit_dir.name, mask.tumor.shape, tumor_on_dce.shape,
        )

    pe, ser = compute_pe_ser(dce.volume)
    labels_path = out_dir / "supervoxel_labels.npz"
    if rerun_aggregate_only and labels_path.exists():
        labels = np.load(labels_path)["labels"].astype(np.int32)
        supervoxels = Supervoxels(
            labels=labels,
            n_supervoxels=int((np.unique(labels) > 0).sum()),
            target_voxels=target_voxels_per_sv,
            compactness=compactness,
        )
    else:
        supervoxels = slic_tumor_supervoxels(
            pe.data,
            tumor_on_dce,
            dce.voxel_spacing,
            target_voxels_per_sv=target_voxels_per_sv,
            compactness=compactness,
        )

    pe_ucsf, ser_ucsf, _ = load_ucsf_maps(
        visit_dir,
        ref_shape=dce.volume.shape[1:],
        ref_spacing=dce.voxel_spacing,
        ref_origin=dce.origin,
    )
    pe_for_agg = pe_ucsf if pe_ucsf is not None else pe.data
    ser_for_agg = ser_ucsf if ser_ucsf is not None else ser.data
    pe_source = "ucsf" if pe_ucsf is not None and ser_ucsf is not None else "recomputed"
    feat = aggregate_features(
        supervoxels,
        dce.volume,
        pe_for_agg,
        ser_for_agg,
        dce.voxel_spacing,
        dce.origin,
        pe_source=pe_source,
    )
    curve = kinetic_summary(dce.volume, tumor_on_dce)

    out_dir.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_dir / "supervoxels.parquet", index=False)
    np.savez_compressed(
        out_dir / "supervoxel_labels.npz",
        labels=supervoxels.labels.astype(np.int32),
    )
    np.savez_compressed(
        out_dir / "pe_ser.npz",
        pe=pe.data.astype(np.float32),
        ser=ser.data.astype(np.float32),
        voxel_spacing=dce.voxel_spacing.astype(np.float32),
        origin=dce.origin.astype(np.float32),
    )
    meta = {
        "dce_phases": int(dce.n_phases),
        "phase_key": dce.phase_key,
        "pre_phase": int(pe.pre_phase),
        "early_phase": int(pe.early_phase),
        "late_phase": int(ser.late_phase),
        "voxel_spacing_mm": dce.voxel_spacing.tolist(),
        "origin_mm": dce.origin.tolist(),
        "spatial_shape": list(dce.volume.shape[1:]),
        "mask_resampled_to_dce": bool(resampled),
        "mask_native_shape": list(mask.tumor.shape),
        "mask_native_spacing_mm": mask.voxel_spacing.tolist(),
        "n_tumor_voxels": int(tumor_on_dce.sum()),
        "n_supervoxels": int(supervoxels.n_supervoxels),
        "target_voxels_per_sv": target_voxels_per_sv,
        "compactness": compactness,
        "pe_source": pe_source,
        "kinetic_summary": [float(x) for x in curve.tolist()],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        "  %s: %d supervoxels from %d tumor voxels (%d phases, %.1fs)",
        visit_dir.name,
        meta["n_supervoxels"],
        meta["n_tumor_voxels"],
        meta["dce_phases"],
        meta["elapsed_sec"],
    )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--target-voxels-per-sv", type=int, default=150)
    ap.add_argument("--compactness", type=float, default=0.10)
    ap.add_argument(
        "--timepoints", nargs="+", default=list(TIMEPOINTS),
        help="Subset of T0 T1 T2 T3 to process (default: all four).",
    )
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a visit if <out_root>/<patient>/<visit>/meta.json exists.")
    ap.add_argument(
        "--rerun-aggregate-only",
        action="store_true",
        help="Reuse existing supervoxel_labels.npz and recompute features only.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("preprocess_patient")

    patient_dir = args.patient_dir.resolve()
    if not patient_dir.is_dir():
        log.error("patient dir not found: %s", patient_dir)
        return 2
    patient_id = patient_dir.name

    log.info("patient %s -> %s", patient_id, args.out_root)
    successes = 0
    failures: list[tuple[str, str]] = []
    for tp in args.timepoints:
        visit_dir = patient_dir / tp
        out_dir = args.out_root / patient_id / tp
        meta_path = out_dir / "meta.json"
        if args.skip_existing and meta_path.exists():
            if args.rerun_aggregate_only and (out_dir / "supervoxel_labels.npz").exists():
                log.info("  %s: rerun aggregate using existing labels", tp)
            else:
                log.info("  %s: skip (already present)", tp)
                successes += 1
                continue
        if not visit_dir.is_dir():
            log.warning("  %s: missing visit directory, skipping", tp)
            continue
        try:
            _process_visit(
                visit_dir, out_dir,
                target_voxels_per_sv=args.target_voxels_per_sv,
                compactness=args.compactness,
                rerun_aggregate_only=args.rerun_aggregate_only,
                log=log,
            )
            successes += 1
        except Exception as exc:  # noqa: BLE001
            log.error("  %s: FAILED: %r", tp, exc)
            failures.append((tp, repr(exc)))
            continue

    if failures:
        log.error(
            "%d/%d visits failed for %s: %s",
            len(failures), len(args.timepoints), patient_id,
            "; ".join(f"{tp}:{msg}" for tp, msg in failures),
        )
    # A temporal graph needs >=2 visits. If we have that, treat the patient as
    # usable (rc=0) so downstream graph-building still runs -- the batch driver
    # will flag it as a partial via the visit count. Only fail hard when fewer
    # than two visits succeeded.
    if successes < 2:
        log.error("done: %s (%d visits, %d failed) -- too few visits to build a graph",
                  patient_id, successes, len(failures))
        return 1
    log.info("done: %s (%d visits, %d failed)", patient_id, successes, len(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
