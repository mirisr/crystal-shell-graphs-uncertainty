#!/usr/bin/env python
"""Batch driver: preprocess + graph-build every patient in raw/.

For each patient directory under ``--raw-root`` (e.g. ``ISPY2-100899``,
``ACRIN-6698-102212``), run in sequence:

    1. preprocess_patient.py  (raw DICOM -> per-visit supervoxel tables)
    2. build_patient_graph.py (per-visit tables -> single .pt graph)

Resumable: a patient whose final ``<graphs_root>/<PID>.pt`` exists is
skipped entirely. A per-patient "preprocessed" check reuses existing
``<derived_root>/<PID>/<visit>/meta.json`` files by delegating to
``preprocess_patient.py --skip-existing``.

Parallelism: ``--jobs`` runs multiple patients in parallel as subprocess
workers. Each worker is process-isolated so a crash in one patient does
not take down the batch. Use ``--jobs 8`` or so on a typical
many-core compute node; the bottleneck is per-visit SLIC which is
CPU-bound but not huge.

Outputs:

    <logs_root>/batch.log            stdout+stderr of this driver
    <logs_root>/<PID>.log            per-patient worker log (on failure
                                     or --keep-all-logs)
    <logs_root>/status.csv           one row per patient with status
                                     (ok / fail_preprocess / fail_graph)

Run on Cradle (from repo root) with e.g.:

    python datasets/ispy2/scripts/batch_preprocess.py \
        --raw-root     datasets/ispy2/raw \
        --derived-root datasets/ispy2/derived \
        --graphs-root  datasets/ispy2/graphs \
        --logs-root    datasets/ispy2/logs \
        --jobs 8
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS = SCRIPT_DIR / "preprocess_patient.py"
BUILD_GRAPH = SCRIPT_DIR / "build_patient_graph.py"


def _iter_patients(raw_root: Path) -> list[str]:
    """Return patient IDs sorted: ISPY2-* then ACRIN-*."""
    out: list[str] = []
    for p in sorted(raw_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("ISPY2-") or p.name.startswith("ACRIN-"):
            out.append(p.name)
    return out


def _run(cmd: list[str], log_file: Path) -> tuple[int, str]:
    """Invoke a subprocess, tee its output to log_file, return (rc, tail)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("wb") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    # Tail last ~400 chars for the status file.
    tail = ""
    try:
        tail = log_file.read_bytes()[-400:].decode(errors="replace")
    except OSError:
        pass
    return proc.returncode, tail


def _worker(args: tuple) -> dict:
    (
        patient_id,
        raw_root,
        derived_root,
        graphs_root,
        logs_root,
        k_spatial,
        k_temporal,
        feature_set,
        target_voxels_per_sv,
        compactness,
        rerun_aggregate_only,
        keep_all_logs,
    ) = args

    graphs_path = graphs_root / f"{patient_id}.pt"
    if graphs_path.exists() and not rerun_aggregate_only:
        return {"patient_id": patient_id, "status": "skip_existing", "detail": str(graphs_path.name)}

    t0 = time.time()
    log_file = logs_root / f"{patient_id}.log"

    rc, tail = _run(
        [
            sys.executable, str(PREPROCESS),
            "--patient-dir", str(raw_root / patient_id),
            "--out-root", str(derived_root),
            "--target-voxels-per-sv", str(target_voxels_per_sv),
            "--compactness", str(compactness),
            "--skip-existing",
            *(["--rerun-aggregate-only"] if rerun_aggregate_only else []),
        ],
        log_file,
    )
    if rc != 0:
        return {
            "patient_id": patient_id,
            "status": "fail_preprocess",
            "detail": tail.splitlines()[-1] if tail else "rc!=0",
            "elapsed_sec": round(time.time() - t0, 1),
        }

    rc, tail = _run(
        [
            sys.executable, str(BUILD_GRAPH),
            "--derived-root", str(derived_root),
            "--patient-id", patient_id,
            "--out-root", str(graphs_root),
            "--k-spatial", str(k_spatial),
            "--k-temporal", str(k_temporal),
            "--feature-set", str(feature_set),
        ],
        log_file,
    )
    if rc != 0:
        return {
            "patient_id": patient_id,
            "status": "fail_graph",
            "detail": tail.splitlines()[-1] if tail else "rc!=0",
            "elapsed_sec": round(time.time() - t0, 1),
        }

    if not keep_all_logs and log_file.exists():
        log_file.unlink()

    return {
        "patient_id": patient_id,
        "status": "ok",
        "detail": str(graphs_path.name),
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--derived-root", type=Path, required=True)
    ap.add_argument("--graphs-root", type=Path, required=True)
    ap.add_argument("--logs-root", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--k-spatial", type=int, default=8)
    ap.add_argument("--k-temporal", type=int, default=4)
    ap.add_argument("--feature-set", choices=("v1", "v2", "v3"), default="v1")
    ap.add_argument("--target-voxels-per-sv", type=int, default=150)
    ap.add_argument("--compactness", type=float, default=0.10)
    ap.add_argument(
        "--rerun-aggregate-only",
        action="store_true",
        help="Reuse existing supervoxel labels and refresh feature tables only.",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N patients (smoke tests).")
    ap.add_argument("--keep-all-logs", action="store_true",
                    help="Keep per-patient log files even on success.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Process only these patient IDs (debug).")
    args = ap.parse_args()

    args.derived_root.mkdir(parents=True, exist_ok=True)
    args.graphs_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.logs_root / "batch.log", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("batch")

    patients = args.only or _iter_patients(args.raw_root)
    if args.limit:
        patients = patients[: args.limit]

    log.info(
        "batch start: %d patients, jobs=%d, raw=%s",
        len(patients), args.jobs, args.raw_root,
    )

    status_csv = args.logs_root / "status.csv"
    new_csv = not status_csv.exists()
    status_fh = status_csv.open("a", newline="")
    writer = csv.DictWriter(
        status_fh,
        fieldnames=["ts", "patient_id", "status", "detail", "elapsed_sec"],
    )
    if new_csv:
        writer.writeheader()

    work = [
        (
            pid, args.raw_root, args.derived_root, args.graphs_root, args.logs_root,
            args.k_spatial, args.k_temporal, args.feature_set, args.target_voxels_per_sv,
            args.compactness, args.rerun_aggregate_only, args.keep_all_logs,
        )
        for pid in patients
    ]

    counts = {"ok": 0, "skip_existing": 0, "fail_preprocess": 0, "fail_graph": 0}
    t_start = time.time()
    n_total = len(work)

    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_worker, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            writer.writerow({
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
                "patient_id": res["patient_id"],
                "status": res["status"],
                "detail": res.get("detail", ""),
                "elapsed_sec": res.get("elapsed_sec", ""),
            })
            status_fh.flush()
            if i % 10 == 0 or i == n_total:
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-6)
                eta_min = (n_total - i) / max(rate, 1e-6) / 60
                log.info(
                    "[%d/%d] ok=%d skip=%d fail_pre=%d fail_graph=%d  "
                    "%.1f pt/min  eta %.1f min",
                    i, n_total, counts.get("ok", 0),
                    counts.get("skip_existing", 0),
                    counts.get("fail_preprocess", 0),
                    counts.get("fail_graph", 0),
                    rate * 60, eta_min,
                )
            if res["status"].startswith("fail"):
                log.warning("  %s: %s (%s)", res["patient_id"], res["status"],
                            res.get("detail", "")[-200:])

    status_fh.close()
    log.info(
        "batch done: ok=%d skip=%d fail_pre=%d fail_graph=%d  elapsed=%.1fs",
        counts.get("ok", 0), counts.get("skip_existing", 0),
        counts.get("fail_preprocess", 0), counts.get("fail_graph", 0),
        time.time() - t_start,
    )
    return 0 if (counts.get("fail_preprocess", 0) + counts.get("fail_graph", 0)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
