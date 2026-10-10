"""Filter the full per-series NBIA index down to a download manifest.

Joins:
  * ispy2_series_index.csv + acrin6698_series_index.csv   (NBIA REST dump)
  * ISPY2-Cohort1-nbia-digest.xlsx                        (timepoint via Study ID)
  * ISPY2-Cohort-1-Clinical-Data.xlsx                     (pCR / HR / HER2 / Arm)

Writes <metadata>/filtered_manifest.csv. The "minimal" default keeps only the
cropped-DCE + FTV-Analysis-Mask series (uni- and bi-lateral). Expand with
--extra-descriptions if you want SER / PE2 maps etc. later.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
META = HERE.parent / "metadata"

MINIMAL_DESCRIPTIONS = [
    "ISPY2: VOLSER: uni-lateral cropped: original DCE",
    "ISPY2: VOLSER: uni-lateral cropped: Analysis Mask",
    "ISPY2: VOLSER: bi-lateral: original DCE",
    "ISPY2: VOLSER: bi-lateral: Analysis Mask",
]
FEATURES_DESCRIPTIONS = [
    "ISPY2: VOLSER: uni-lateral cropped: SER",
    "ISPY2: VOLSER: uni-lateral cropped: PE2",
    "ISPY2: VOLSER: uni-lateral cropped: PE5",
    "ISPY2: VOLSER: uni-lateral cropped: PE6",
    "ISPY2: VOLSER: bi-lateral: SER",
    "ISPY2: VOLSER: bi-lateral: PE2",
]
DWI_DESCRIPTION_PATTERNS = [
    r"^ACRIN-6698: (?:TrT[01]: )?ADC: from S.*: bVals=0,100,600,800$",
    r"^ACRIN-6698: (?:TrT[01]: )?DWI MASK: from S.*: Whole Tumor Manual$",
]

TIMEPOINT_RE = re.compile(r"_T(\d)$")
TRT_RE = re.compile(r"\bTrT([0-3])\b", re.IGNORECASE)


def _strip_prefix(pid: str) -> int | None:
    m = re.search(r"(\d+)$", str(pid))
    return int(m.group(1)) if m else None


def load_series_index() -> pd.DataFrame:
    frames = []
    for name in ("ispy2_series_index.csv", "acrin6698_series_index.csv"):
        path = META / name
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run build_series_index.py first.")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def load_digest_timepoints() -> pd.DataFrame:
    path = META / "ISPY2-Cohort1-nbia-digest.xlsx"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    dig = pd.read_excel(path, sheet_name=0)
    keep = dig[["Series Instance UID", "Study ID"]].rename(
        columns={"Series Instance UID": "SeriesInstanceUID", "Study ID": "StudyID"}
    )
    keep["Timepoint"] = (
        keep["StudyID"].astype(str).str.extract(TIMEPOINT_RE, expand=False)
    )
    keep["Timepoint"] = keep["Timepoint"].map(
        lambda x: f"T{x}" if pd.notna(x) else None
    )
    return keep


def load_clinical() -> pd.DataFrame:
    path = META / "ISPY2-Cohort-1-Clinical-Data.xlsx"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    return pd.read_excel(path, sheet_name=0)


def build_manifest(
    descriptions: list[str],
    require_clinical: bool = True,
    *,
    description_patterns: list[str] | None = None,
) -> pd.DataFrame:
    series = load_series_index()
    digest = load_digest_timepoints()
    clinical = load_clinical()

    if description_patterns:
        pattern = "|".join(f"(?:{p})" for p in description_patterns)
        desc = series["SeriesDescription"].fillna("")
        match_desc = desc.str.contains(pattern, regex=True)
        match_pid = series["PatientID"].fillna("").str.startswith("ACRIN-6698-")
        df = series[match_desc & match_pid].copy()
    else:
        df = series[series["SeriesDescription"].isin(descriptions)].copy()
    df["PatientIntID"] = df["PatientID"].map(_strip_prefix)

    df = df.merge(digest, on="SeriesInstanceUID", how="left")
    missing_tp = df["Timepoint"].isna()
    if missing_tp.any():
        inferred = (
            df.loc[missing_tp, "SeriesDescription"]
            .fillna("")
            .str.extract(TRT_RE, expand=False)
            .map(lambda x: f"T{x}" if pd.notna(x) else None)
        )
        df.loc[missing_tp, "Timepoint"] = inferred.values

    clin_cols = ["Patient_ID", "Arm", "HR", "HER2", "MP", "pCR", "Age_at_Screening"]
    df = df.merge(
        clinical[clin_cols].rename(columns={"Patient_ID": "PatientIntID"}),
        on="PatientIntID",
        how="left",
    )

    if require_clinical:
        before = df["PatientID"].nunique()
        df = df[df["pCR"].notna()].copy()
        after = df["PatientID"].nunique()
        print(
            f"  filter: require clinical join -> kept {after}/{before} patients"
        )
    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print(f"  series:   {len(df):,}")
    print(f"  size:     {df['FileSize'].sum() / 1e9:.2f} GB")
    print(f"  patients: {df['PatientID'].nunique()}")
    if "Timepoint" in df.columns:
        print("  timepoint coverage:")
        tp = (
            df.dropna(subset=["Timepoint"])
            .groupby("Timepoint")["PatientID"]
            .nunique()
            .sort_index()
        )
        for t, n in tp.items():
            print(f"    {t}: {n} patients")
    if "pCR" in df.columns:
        pcr = df.groupby("pCR")["PatientID"].nunique()
        print(f"  pCR balance: {pcr.to_dict()}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scope",
        choices=("minimal", "features", "dwi"),
        default="minimal",
        help=(
            "minimal = cropped DCE + masks; "
            "features = +SER/PE2/PE5/PE6 maps; "
            "dwi = ACRIN ADC maps + DWI whole-tumor masks"
        ),
    )
    p.add_argument(
        "--extra-descriptions",
        nargs="*",
        default=[],
        help="Additional exact SeriesDescription strings to include",
    )
    p.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep ACRIN-6698 patients that lack a clinical/pCR row",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=META / "filtered_manifest.csv",
        help="Output CSV path",
    )
    args = p.parse_args()

    desc: list[str] = list(MINIMAL_DESCRIPTIONS)
    patterns: list[str] | None = None
    if args.scope == "features":
        desc.extend(FEATURES_DESCRIPTIONS)
    elif args.scope == "dwi":
        desc = []
        patterns = list(DWI_DESCRIPTION_PATTERNS)
    desc.extend(args.extra_descriptions)
    print(f"Scope: {args.scope}")
    if patterns:
        print(f"  regex patterns: {len(patterns)}")
        for pat in patterns:
            print(f"  - {pat}")
    else:
        print(f"  exact descriptions: {len(desc)}")
        for d in desc:
            print(f"  - {d}")

    df = build_manifest(
        desc,
        require_clinical=not args.keep_unlabeled,
        description_patterns=patterns,
    )
    summarize(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
