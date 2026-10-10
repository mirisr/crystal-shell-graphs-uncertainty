"""Choose a small pilot set from a filtered manifest and write its subset
manifest. Defaults: 1 pCR=1 + 1 pCR=0, both with all four timepoints present
for every series in MINIMAL_DESCRIPTIONS, preferably from the same scanner
vendor + site for consistency.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
META = HERE.parent / "metadata"


def _coarse_series_type(description: str) -> str:
    """Collapse uni-/bi-lateral variants into a single type tag.

    A given patient has either uni- OR bi-lateral scans (determined by
    acquisition matrix size), never both. For coverage checking we only care
    about the type (DCE vs. Mask) plus the timepoint.
    """
    s = str(description).lower()
    if "analysis mask" in s:
        return "MASK"
    if "original dce" in s:
        return "DCE"
    if ": ser" in s:
        return "SER"
    if ": pe2" in s:
        return "PE2"
    if ": pe5" in s:
        return "PE5"
    if ": pe6" in s:
        return "PE6"
    return description


def _patients_with_full_coverage(df: pd.DataFrame) -> pd.Index:
    """Patients that have every (Timepoint, coarse-series-type) combination."""
    work = df.copy()
    work["_type"] = work["SeriesDescription"].map(_coarse_series_type)
    required_types = sorted(work["_type"].unique())
    timepoints = ["T0", "T1", "T2", "T3"]
    expected = len(required_types) * len(timepoints)
    pt = (
        work.groupby("PatientID")
        .apply(
            lambda g: g[["Timepoint", "_type"]].drop_duplicates().shape[0],
            include_groups=False,
        )
    )
    return pt[pt >= expected].index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=META / "filtered_manifest.csv",
        help="Input manifest (output of filter_series.py)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=META / "pilot_manifest.csv",
        help="Output pilot manifest",
    )
    p.add_argument("--n-pcr1", type=int, default=1, help="How many pCR=1 patients")
    p.add_argument("--n-pcr0", type=int, default=1, help="How many pCR=0 patients")
    p.add_argument(
        "--vendor",
        default=None,
        help="Restrict to patients with this Manufacturer (e.g. 'SIEMENS', 'GE MEDICAL SYSTEMS')",
    )
    p.add_argument("--site", default=None, help="Restrict to this Site")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"{args.manifest} not found; run filter_series.py first"
        )
    df = pd.read_csv(args.manifest)

    pool = df.copy()
    if args.vendor:
        pool = pool[pool["Manufacturer"].str.contains(args.vendor, case=False, na=False)]
    if args.site:
        pool = pool[pool["Site"].str.contains(args.site, case=False, na=False)]

    full_cov = _patients_with_full_coverage(pool)
    pool = pool[pool["PatientID"].isin(full_cov)]
    print(f"Patients with full 4-timepoint coverage: {pool['PatientID'].nunique()}")

    patient_meta = (
        pool.drop_duplicates("PatientID")[
            [
                "PatientID",
                "pCR",
                "HR",
                "HER2",
                "MP",
                "Arm",
                "Site",
                "Manufacturer",
                "ManufacturerModelName",
                "PatientIntID",
            ]
        ]
    )
    pcr1 = patient_meta[patient_meta["pCR"] == 1].sample(
        n=min(args.n_pcr1, (patient_meta["pCR"] == 1).sum()),
        random_state=args.seed,
    )
    pcr0 = patient_meta[patient_meta["pCR"] == 0].sample(
        n=min(args.n_pcr0, (patient_meta["pCR"] == 0).sum()),
        random_state=args.seed,
    )
    picked = pd.concat([pcr1, pcr0], ignore_index=True)

    if picked.empty:
        raise SystemExit("No candidate patients matched filters")

    print("\nPicked patients:")
    print(
        picked[
            ["PatientID", "pCR", "HR", "HER2", "Manufacturer", "ManufacturerModelName", "Site"]
        ].to_string(index=False)
    )

    subset = pool[pool["PatientID"].isin(picked["PatientID"])].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.out, index=False)
    print(
        f"\nWrote {args.out} with {len(subset)} series "
        f"({subset['FileSize'].sum() / 1e9:.2f} GB) across "
        f"{subset['PatientID'].nunique()} patients"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
