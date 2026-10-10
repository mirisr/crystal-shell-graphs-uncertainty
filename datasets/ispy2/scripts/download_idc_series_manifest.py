"""Download an I-SPY 2 series manifest from the public IDC S3 bucket.

The target layout is identical to ``download_ispy2.py`` so the existing
preprocessing code can consume either downloader's output.  A series is only
accepted when the sum of its DICOM file sizes exactly matches ``FileSize`` in
the manifest.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from idc_index import IDCClient


_SLUG = re.compile(r"[^A-Za-z0-9]+")


def slug(value: str) -> str:
    return _SLUG.sub("_", str(value)).strip("_")


def target_dir(out: Path, row: dict[str, str]) -> Path:
    return (
        out
        / row["PatientID"]
        / row["Timepoint"]
        / slug(row["SeriesDescription"])
        / row["SeriesInstanceUID"]
    )


def dicom_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.glob("*.dcm")) if path.exists() else 0


def download_series(row: dict[str, str], out: Path) -> tuple[str, str, int]:
    uid = row["SeriesInstanceUID"]
    expected = int(row["FileSize"])
    destination = target_dir(out, row)
    current = dicom_bytes(destination)
    if current == expected:
        return uid, "skip-exact", current
    if current:
        return uid, f"refuse-existing-size-{current}-expected-{expected}", current

    destination.mkdir(parents=True, exist_ok=True)
    urls = IDCClient().get_series_file_URLs(uid, source_bucket_location="aws")
    if not urls:
        return uid, "no-idc-objects", 0

    for url in urls:
        filename = Path(url).name
        final = destination / filename
        partial = destination / f".{filename}.part"
        completed = subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                url,
                str(partial),
                "--no-sign-request",
                "--only-show-errors",
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            partial.unlink(missing_ok=True)
            detail = completed.stderr.strip().replace("\n", " ")
            return uid, f"aws-error:{detail}", dicom_bytes(destination)
        partial.replace(final)

    actual = dicom_bytes(destination)
    if actual != expected:
        return uid, f"size-mismatch-{actual}-expected-{expected}", actual
    return uid, "ok", actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--per-series",
        action="store_true",
        help="Use aws-cli per object instead of the default batched s5cmd transfer.",
    )
    return parser.parse_args()


def download_batch(rows: list[dict[str, str]], out: Path) -> list[tuple[dict[str, str], str, int]]:
    """Use IDC's bundled s5cmd to fetch all incomplete series concurrently."""
    results: list[tuple[dict[str, str], str, int]] = []
    pending: list[dict[str, str]] = []
    for row in rows:
        actual = dicom_bytes(target_dir(out, row))
        if actual == int(row["FileSize"]):
            results.append((row, "skip-exact", actual))
        else:
            pending.append(row)
    if not pending:
        return results

    staging = out / "_idc_staging_nine_case_movies"
    staging.mkdir(parents=True, exist_ok=True)
    IDCClient().download_from_selection(
        str(staging),
        seriesInstanceUID=[row["SeriesInstanceUID"] for row in pending],
        quiet=True,
        show_progress_bar=True,
        dirTemplate="%PatientID/%SeriesInstanceUID",
        source_bucket_location="aws",
    )
    failures = 0
    for row in pending:
        source = staging / row["PatientID"] / row["SeriesInstanceUID"]
        destination = target_dir(out, row)
        destination.mkdir(parents=True, exist_ok=True)
        source_files = list(source.glob("*.dcm"))
        if not source_files:
            results.append((row, "no-idc-objects", dicom_bytes(destination)))
            failures += 1
            continue
        for source_file in source_files:
            shutil.copy2(source_file, destination / source_file.name)
        for partial in destination.glob(".*.dcm.part"):
            partial.unlink()
        actual = dicom_bytes(destination)
        expected = int(row["FileSize"])
        detail = "ok" if actual == expected else f"size-mismatch-{actual}-expected-{expected}"
        failures += int(detail != "ok")
        results.append((row, detail, actual))
    if failures == 0:
        shutil.rmtree(staging)
    return results


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "PatientID",
        "Timepoint",
        "SeriesDescription",
        "SeriesInstanceUID",
        "FileSize",
    }
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Manifest is empty or lacks required columns: {args.manifest}")

    args.out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    state = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0}
    started = time.time()
    print(
        f"IDC manifest: {len(rows)} series, "
        f"{len({row['PatientID'] for row in rows})} patients, "
        f"{sum(int(row['FileSize']) for row in rows) / 1e9:.2f} GB"
    )
    print(f"Destination: {args.out}  jobs={max(1, args.jobs)}")
    sys.stdout.flush()

    if not args.per_series:
        outcomes = download_batch(rows, args.out)
        iterator = ((row, detail, actual) for row, detail, actual in outcomes)
    else:
        executor = ThreadPoolExecutor(max_workers=max(1, args.jobs))
        futures = {executor.submit(download_series, row, args.out): row for row in rows}
        iterator = (
            (futures[future], *future.result()[1:])
            for future in as_completed(futures)
        )
    try:
        for index, (row, detail, actual) in enumerate(iterator, 1):
            with lock:
                if detail == "ok":
                    state["ok"] += 1
                elif detail == "skip-exact":
                    state["skip"] += 1
                else:
                    state["fail"] += 1
                state["bytes"] += actual if detail in {"ok", "skip-exact"} else 0
                snapshot = dict(state)
            print(
                f"[{index:02d}/{len(rows)}] {row['PatientID']} {row['Timepoint']} "
                f"{row['Modality']} {detail} | ok={snapshot['ok']} "
                f"skip={snapshot['skip']} fail={snapshot['fail']}"
            )
            sys.stdout.flush()
    finally:
        if args.per_series:
            executor.shutdown(wait=True)

    elapsed = time.time() - started
    print(
        f"Done in {elapsed:.1f}s: ok={state['ok']} skip={state['skip']} "
        f"fail={state['fail']} verified={state['bytes'] / 1e9:.2f} GB"
    )
    return 0 if state["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
