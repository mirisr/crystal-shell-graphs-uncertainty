#!/usr/bin/env python3
"""Audit I-SPY2/ACRIN download scope and prepare expansion manifests.

This script uses only the checked-in metadata:

* metadata/ispy2_series_index.csv
* metadata/acrin6698_series_index.csv
* metadata/filtered_manifest.csv

It compares the current minimal download manifest against all TCIA/NBIA series
for the clinical Cohort-1 patients represented in `filtered_manifest.csv`.
Outputs are written to a report directory so the large raw-data decision can be
made before starting any download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
META = REPO / "datasets" / "ispy2" / "metadata"
MANIFESTS = REPO / "datasets" / "ispy2" / "manifests"


def _tib(bytes_series: pd.Series) -> float:
    return float(pd.to_numeric(bytes_series, errors="coerce").fillna(0).sum() / 1024**4)


def _gb(bytes_series: pd.Series) -> float:
    return float(pd.to_numeric(bytes_series, errors="coerce").fillna(0).sum() / 1024**3)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for name in ("ispy2_series_index.csv", "acrin6698_series_index.csv"):
        path = META / name
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        frames.append(pd.read_csv(path))
    series = pd.concat(frames, ignore_index=True)

    selected_path = META / "filtered_manifest.csv"
    if not selected_path.exists():
        raise SystemExit(f"Missing {selected_path}")
    selected = pd.read_csv(selected_path)
    return series, selected


def classify_description(desc: object) -> str:
    d = str(desc).lower()
    if "volser" in d and "original dce" in d:
        return "selected-style DCE/mask family"
    if "analysis mask" in d:
        return "selected-style DCE/mask family"
    if "volser" in d and ("ser" in d or "pe2" in d or "pe5" in d or "pe6" in d):
        return "DCE derived maps"
    if "adc" in d or "dwi" in d or "diff" in d or "bval" in d:
        return "DWI/ADC"
    if "t2" in d or "fse" in d:
        return "T2"
    if "local" in d or "scout" in d or "survey" in d or "loc" in d:
        return "localizer/scout"
    if "dyn" in d or "vibrant" in d or "ethrive" in d or "pre/post" in d:
        return "original dynamic DCE"
    if "mip" in d or "cad" in d:
        return "CAD/MIP/other derived"
    return "other MR/SEG"


def build_scope_tables(series: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical_ids = set(selected["PatientID"].astype(str))
    full = series[series["PatientID"].astype(str).isin(clinical_ids)].copy()
    selected_uids = set(selected["SeriesInstanceUID"].astype(str))
    full["selected_minimal"] = full["SeriesInstanceUID"].astype(str).isin(selected_uids)
    full["download_class"] = full["SeriesDescription"].map(classify_description)
    unselected = full[~full["selected_minimal"]].copy()
    return full, unselected


def write_manifest(df: pd.DataFrame, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["Collection", "PatientID", "StudyDate", "SeriesDescription", "SeriesInstanceUID"])
    df.to_csv(out_path, index=False)
    return {
        "path": str(out_path),
        "series": int(len(df)),
        "patients": int(df["PatientID"].nunique()) if len(df) else 0,
        "gb": round(_gb(df["FileSize"]) if len(df) else 0.0, 3),
        "tib": round(_tib(df["FileSize"]) if len(df) else 0.0, 3),
    }


def tcia_manifest_counts() -> list[dict]:
    rows = []
    for path in sorted(MANIFESTS.glob("*.tcia")):
        lines = [ln.strip() for ln in path.read_text(errors="ignore").splitlines() if ln.strip()]
        series_lines = [ln for ln in lines if ln[0].isdigit()]
        rows.append({"manifest": path.name, "series": len(series_lines)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "reports" / "ispy2_download_scope_audit",
    )
    args = ap.parse_args()

    out_root = args.out_root
    manifest_root = out_root / "manifests"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    series, selected = load_inputs()
    full, unselected = build_scope_tables(series, selected)

    scope_summary = pd.DataFrame(
        [
            {
                "scope": "current_minimal_manifest",
                "series": int(full["selected_minimal"].sum()),
                "patients": int(full.loc[full["selected_minimal"], "PatientID"].nunique()),
                "gb": round(_gb(full.loc[full["selected_minimal"], "FileSize"]), 3),
                "tib": round(_tib(full.loc[full["selected_minimal"], "FileSize"]), 3),
            },
            {
                "scope": "full_clinical_cohort_series",
                "series": int(len(full)),
                "patients": int(full["PatientID"].nunique()),
                "gb": round(_gb(full["FileSize"]), 3),
                "tib": round(_tib(full["FileSize"]), 3),
            },
            {
                "scope": "unselected_clinical_series",
                "series": int(len(unselected)),
                "patients": int(unselected["PatientID"].nunique()),
                "gb": round(_gb(unselected["FileSize"]), 3),
                "tib": round(_tib(unselected["FileSize"]), 3),
            },
        ]
    )
    scope_summary.to_csv(out_root / "download_scope_summary.csv", index=False)

    class_summary = (
        unselected.groupby("download_class", dropna=False)
        .agg(
            series=("SeriesInstanceUID", "count"),
            patients=("PatientID", "nunique"),
            gb=("FileSize", _gb),
            tib=("FileSize", _tib),
        )
        .reset_index()
        .sort_values("gb", ascending=False)
    )
    class_summary.to_csv(out_root / "unselected_class_summary.csv", index=False)

    desc_summary = (
        unselected.groupby(["Collection", "download_class", "SeriesDescription"], dropna=False)
        .agg(
            series=("SeriesInstanceUID", "count"),
            patients=("PatientID", "nunique"),
            gb=("FileSize", _gb),
            tib=("FileSize", _tib),
        )
        .reset_index()
        .sort_values("gb", ascending=False)
    )
    desc_summary.to_csv(out_root / "unselected_description_summary.csv", index=False)

    feature_maps = unselected[unselected["download_class"].eq("DCE derived maps")].copy()
    dwi_adc = unselected[unselected["download_class"].eq("DWI/ADC")].copy()
    t2 = unselected[unselected["download_class"].eq("T2")].copy()
    original_dynamic_dce = unselected[unselected["download_class"].eq("original dynamic DCE")].copy()

    manifest_info = {
        "feature_maps": write_manifest(feature_maps, manifest_root / "feature_maps_manifest.csv"),
        "dwi_adc": write_manifest(dwi_adc, manifest_root / "dwi_adc_manifest.csv"),
        "t2": write_manifest(t2, manifest_root / "t2_manifest.csv"),
        "original_dynamic_dce": write_manifest(
            original_dynamic_dce, manifest_root / "original_dynamic_dce_manifest.csv"
        ),
        "full_unselected_clinical": write_manifest(
            unselected, manifest_root / "full_unselected_clinical_manifest.csv"
        ),
    }

    audit = {
        "scope_summary": scope_summary.to_dict(orient="records"),
        "unselected_class_summary": class_summary.to_dict(orient="records"),
        "prepared_manifests": manifest_info,
        "local_tcia_manifest_counts": tcia_manifest_counts(),
    }
    (out_root / "download_scope_audit.json").write_text(json.dumps(audit, indent=2))

    md = [
        "# I-SPY2 Download Scope Audit",
        "",
        "This audit compares the current minimal manifest against all series in the local NBIA series indexes for Cohort-1 clinical patients.",
        "",
        "## Scope summary",
        "",
        scope_summary.to_markdown(index=False),
        "",
        "## Unselected series by class",
        "",
        class_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Prepared candidate manifests",
        "",
    ]
    for name, info in manifest_info.items():
        md.append(f"- `{name}`: `{info['path']}` ({info['series']:,} series, {info['patients']} patients, {info['gb']:.1f} GB)")
    md.extend(
        [
            "",
            "## Recommendation",
            "",
            "Do not start with the full unselected manifest. Start with the smaller, hypothesis-driven manifests:",
            "",
            "1. `feature_maps_manifest.csv` for SER/PE maps that may enrich graph features without needing the full raw archive.",
            "2. `dwi_adc_manifest.csv` because DWI/ADC may add complementary biology and is directly relevant to patient-specific response.",
            "3. `t2_manifest.csv` if morphology or surrounding tissue context becomes part of the graph.",
            "4. `original_dynamic_dce_manifest.csv` only if the derived maps are insufficient and storage/time are approved.",
            "",
            "The full unselected clinical manifest should be treated as an archival mirror, not the first modeling step.",
            "",
        ]
    )
    (out_root / "download_scope_audit.md").write_text("\n".join(md))

    print((out_root / "download_scope_audit.md").relative_to(REPO))
    print(scope_summary.to_string(index=False))
    print(class_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
