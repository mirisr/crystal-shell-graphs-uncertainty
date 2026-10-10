"""Resumable DICOM downloader for a filtered I-SPY 2 manifest.

Layout on disk:

    <out>/<PatientID>/<Timepoint>/<SeriesDescription-slug>/<SeriesInstanceUID>/*.dcm

Resume is automatic: any series whose target directory already exists and
contains at least one .dcm file is skipped. A download log CSV is appended to
<out>/_download_log.csv so you can diff runs.

Typical use:

    python scripts/download_ispy2.py \
        --manifest metadata/pilot_manifest.csv \
        --out pilot

For the full 500 GB pull, point --manifest at filtered_manifest.csv and set
--out to /weka/scratch/<user>/ispy2_raw (after HPC grants access).
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from tcia_utils import nbia

HERE = Path(__file__).resolve().parent
META = HERE.parent / "metadata"

_SLUG = re.compile(r"[^A-Za-z0-9]+")


def slug(s: str) -> str:
    return _SLUG.sub("_", str(s)).strip("_")


def series_target_dir(out: Path, row: pd.Series) -> Path:
    tp = row.get("Timepoint")
    tp = tp if isinstance(tp, str) and tp else "Tunk"
    return (
        out
        / str(row["PatientID"])
        / tp
        / slug(row["SeriesDescription"])
        / str(row["SeriesInstanceUID"])
    )


def already_downloaded(target: Path) -> bool:
    if not target.exists():
        return False
    dcms = list(target.glob("*.dcm"))
    return len(dcms) > 0


def download_one(row: pd.Series, out: Path, staging_root: Path) -> tuple[bool, str]:
    target = series_target_dir(out, row)
    if already_downloaded(target):
        return True, "skip-existing"

    uid = str(row["SeriesInstanceUID"])
    # Per-thread staging dir so concurrent downloads never collide.
    staging = staging_root / f"t{threading.get_ident()}"
    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / uid
    if tmp.exists():
        shutil.rmtree(tmp)

    try:
        nbia.downloadSeries(
            [uid],
            path=str(staging),
            input_type="list",
        )
    except Exception as exc:
        return False, f"error:{exc!r}"

    src = staging / uid
    if not src.exists() or not any(src.glob("*.dcm")):
        return False, "no-dicoms-returned"

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))
    return True, "ok"


def _append_log(log_csv: Path, row: dict) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not log_csv.exists()
    with log_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination root (e.g. pilot, raw, /weka/scratch/user/ispy2)",
    )
    p.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="Temp dir for tcia_utils downloads (default: <out>/_staging)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N series (useful for smoke tests)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent downloads (default 1). 4-6 typically saturates NBIA.",
    )
    p.add_argument(
        "--quiet-tcia",
        action="store_true",
        help="Suppress tcia_utils INFO chatter (one line per series).",
    )
    p.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="SLURM_ARRAY_TASK_ID (0-based)",
    )
    p.add_argument(
        "--array-count",
        type=int,
        default=None,
        help="SLURM_ARRAY_TASK_COUNT",
    )
    args = p.parse_args()

    if args.quiet_tcia:
        logging.getLogger("tcia_utils").setLevel(logging.WARNING)
        logging.getLogger("tcia_utils.nbia").setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    df = pd.read_csv(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    staging = args.staging or (args.out / "_staging")

    if args.array_index is not None and args.array_count:
        df = df.iloc[args.array_index :: args.array_count].reset_index(drop=True)
        print(
            f"array task {args.array_index}/{args.array_count}: {len(df)} series"
        )

    if args.limit:
        df = df.head(args.limit)

    total_bytes = int(df["FileSize"].sum()) if "FileSize" in df.columns else 0
    print(
        f"Manifest: {len(df):,} series, "
        f"{df['PatientID'].nunique()} patients, "
        f"{total_bytes / 1e9:.2f} GB"
    )
    print(f"Destination: {args.out}  jobs={args.jobs}")
    sys.stdout.flush()

    log_csv = args.out / "_download_log.csv"
    started = datetime.utcnow().isoformat()

    state = {"ok": 0, "skipped": 0, "failed": 0, "bytes_done": 0}
    state_lock = threading.Lock()
    log_lock = threading.Lock()
    t0 = time.time()
    n_total = len(df)

    def _process(row: pd.Series) -> tuple[pd.Series, bool, str]:
        res, detail = download_one(row, args.out, staging)
        with log_lock:
            _append_log(
                log_csv,
                {
                    "ts": datetime.utcnow().isoformat(),
                    "run_started": started,
                    "PatientID": row["PatientID"],
                    "Timepoint": row.get("Timepoint"),
                    "SeriesInstanceUID": row["SeriesInstanceUID"],
                    "SeriesDescription": row["SeriesDescription"],
                    "FileSize": row.get("FileSize"),
                    "status": "ok" if res else "fail",
                    "detail": detail,
                },
            )
        return row, res, detail

    def _progress(n: int) -> None:
        elapsed = max(time.time() - t0, 1e-6)
        with state_lock:
            s = dict(state)
        rate_mb = s["bytes_done"] / 1e6 / elapsed
        eta_s = (total_bytes - s["bytes_done"]) / max(s["bytes_done"] / elapsed, 1.0)
        eta_h = eta_s / 3600
        print(
            f"  [{n:>5}/{n_total}] ok={s['ok']} skip={s['skipped']} "
            f"fail={s['failed']}  "
            f"{s['bytes_done'] / 1e9:.2f}/{total_bytes / 1e9:.2f} GB  "
            f"{rate_mb:.1f} MB/s  eta {eta_h:.1f} h"
        )
        sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(_process, row) for _, row in df.iterrows()]
        for n, fut in enumerate(as_completed(futs), 1):
            row, res, detail = fut.result()
            with state_lock:
                if res:
                    if detail == "skip-existing":
                        state["skipped"] += 1
                    else:
                        state["ok"] += 1
                    state["bytes_done"] += int(row.get("FileSize", 0) or 0)
                else:
                    state["failed"] += 1
            if n % 10 == 0 or n == n_total:
                _progress(n)

    if staging.exists():
        try:
            shutil.rmtree(staging)
        except OSError:
            pass

    print(
        f"\nDone. ok={state['ok']} skip={state['skipped']} fail={state['failed']}  "
        f"elapsed={time.time() - t0:.1f}s"
    )
    return 0 if state["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
