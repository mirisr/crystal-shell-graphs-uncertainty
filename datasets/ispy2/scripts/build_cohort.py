#!/usr/bin/env python
"""Build the canonical I-SPY 2 cohort table used for all downstream stages.

Joins three sources into a single ``cohort.parquet``:

    1. ``metadata/ISPY2-Cohort-1-Clinical-Data.xlsx``
       -- 985 rows, one per clinical trial patient. Columns: ``Patient_ID``,
       ``Arm``, ``HR``, ``HER2``, ``MP``, ``pCR``, ``Age_at_Screening``,
       demographics. All label columns are non-null for every patient.
    2. ``metadata/graphs_manifest.txt``
       -- one line per ``<PID>.pt`` graph actually present on Cradle
       (e.g. ``ISPY2-756412``, ``ACRIN-6698-407539``). 959 rows.
    3. ``metadata/graphs_visit_counts.csv``
       -- per-graph visit count and node count, produced by running
       torch.load on every .pt and reading ``visit_offsets``.
    4. (reference only) ``metadata/Multi-feature-MRI-NACT-Data.xlsx``
       -- 384 patients used in the published 4-visit multi-feature analysis;
       joined as an indicator column ``in_mfmri`` for sanity-checking.

Clinical ``Patient_ID`` is the numeric suffix of the TCIA PatientID:
``ISPY2-756412`` or ``ACRIN-6698-756412`` -> ``756412``. In our 959-graph
cohort there is no ID collision between the two collections, so the join
is clean 1:1.

Output schema (one row per graph on disk):

    patient_id            TCIA-style, e.g. ``ISPY2-756412``
    collection            ``ISPY2`` or ``ACRIN-6698``
    clinical_patient_id   int, e.g. ``756412``
    arm                   string, trial arm (12 values)
    pCR                   int 0/1, pathological complete response
    HR                    int 0/1, hormone-receptor status
    HER2                  int 0/1
    MP                    int 0/1, MammaPrint high vs ultra-high risk
    subtype               derived from HR/HER2: HR+/HER2-, HR+/HER2+,
                          HR-/HER2+, TNBC
    age                   float, age at screening
    visit_count           int, number of timepoints present in the graph
    num_nodes             int, total supervoxels across visits
    has_graph             bool (always True for rows emitted)
    in_mfmri              bool, patient present in Multi-feature 4-visit set
    label_source          ``cohort1`` for all rows (the only source)
    notes                 free-text audit hints

Also writes ``cohort_audit.md`` summarizing join coverage, pCR prevalence
by arm and subtype, and visit-count distribution.

Usage::

    python datasets/ispy2/scripts/build_cohort.py \
        --metadata-dir datasets/ispy2/metadata \
        --out-parquet  datasets/ispy2/cohort.parquet \
        --out-audit    datasets/ispy2/cohort_audit.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


# --- pure helpers (covered by unit tests) -----------------------------------


def split_tcia_id(pid: str) -> tuple[str, int]:
    """Split a TCIA PatientID into (collection, clinical_patient_id).

    Accepts ``ISPY2-<int>`` and ``ACRIN-6698-<int>``. Raises ``ValueError``
    for any other shape so malformed entries fail loudly rather than get
    silently dropped from the join.
    """
    if pid.startswith("ISPY2-"):
        suffix = pid[len("ISPY2-"):]
        return "ISPY2", int(suffix)
    if pid.startswith("ACRIN-6698-"):
        suffix = pid[len("ACRIN-6698-"):]
        return "ACRIN-6698", int(suffix)
    raise ValueError(f"unrecognized TCIA PatientID: {pid!r}")


def derive_subtype(hr: int, her2: int) -> str:
    """Map (HR, HER2) to a human-readable breast-cancer subtype label.

    The four subtypes are standard in I-SPY 2 write-ups; TNBC is called out
    explicitly because it drives several stratification decisions
    downstream. Behavior at NaN is caller's responsibility (``pandas.apply``
    will handle via ``NaN`` propagation).
    """
    hr = int(hr); her2 = int(her2)
    if hr == 1 and her2 == 0:
        return "HR+/HER2-"
    if hr == 1 and her2 == 1:
        return "HR+/HER2+"
    if hr == 0 and her2 == 1:
        return "HR-/HER2+"
    return "TNBC"


def build_cohort_frame(
    clinical: pd.DataFrame,
    graph_ids: Iterable[str],
    visit_counts: pd.DataFrame,
    mfmri_ids: set[int],
) -> pd.DataFrame:
    """Return the cohort DataFrame (one row per graph on disk).

    Pure function over already-loaded inputs so tests can feed synthetic
    frames without touching the filesystem.
    """
    rows = []
    vc_by_pid = dict(zip(visit_counts["patient_id"], visit_counts["visit_count"]))
    nn_by_pid = dict(zip(visit_counts["patient_id"], visit_counts["num_nodes"]))
    clin_by_intid = clinical.set_index("Patient_ID")

    for pid in graph_ids:
        coll, int_id = split_tcia_id(pid)
        if int_id not in clin_by_intid.index:
            rows.append({
                "patient_id": pid, "collection": coll, "clinical_patient_id": int_id,
                "arm": None, "pCR": None, "HR": None, "HER2": None, "MP": None,
                "subtype": None, "age": None,
                "visit_count": vc_by_pid.get(pid, 0),
                "num_nodes": nn_by_pid.get(pid, 0),
                "has_graph": True,
                "in_mfmri": int_id in mfmri_ids,
                "label_source": "none",
                "notes": "no_clinical_row",
            })
            continue
        c = clin_by_intid.loc[int_id]
        rows.append({
            "patient_id": pid, "collection": coll, "clinical_patient_id": int_id,
            "arm": c["Arm"], "pCR": int(c["pCR"]),
            "HR": int(c["HR"]), "HER2": int(c["HER2"]), "MP": int(c["MP"]),
            "subtype": derive_subtype(c["HR"], c["HER2"]),
            "age": float(c["Age_at_Screening"]) if pd.notna(c["Age_at_Screening"]) else None,
            "visit_count": int(vc_by_pid.get(pid, 0)),
            "num_nodes": int(nn_by_pid.get(pid, 0)),
            "has_graph": True,
            "in_mfmri": int_id in mfmri_ids,
            "label_source": "cohort1",
            "notes": "",
        })
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def write_audit_md(cohort: pd.DataFrame, clinical: pd.DataFrame, out_path: Path) -> None:
    """Write a human-readable audit markdown alongside ``cohort.parquet``."""
    lines: list[str] = ["# I-SPY 2 cohort audit",
                        "",
                        "Auto-generated by `build_cohort.py`. Describes the join",
                        "between the clinical XLSX and the graphs on disk.",
                        ""]
    n_clin = len(clinical)
    n_graphs = len(cohort)
    n_labeled = cohort["pCR"].notna().sum()
    n_no_label = (cohort["label_source"] == "none").sum()
    four_visit = (cohort["visit_count"] == 4).sum()
    three_visit = (cohort["visit_count"] == 3).sum()
    two_visit = (cohort["visit_count"] == 2).sum()
    lines += ["## Coverage",
              "",
              f"- Clinical rows (Cohort-1 XLSX): **{n_clin}**",
              f"- Graphs on disk: **{n_graphs}**",
              f"- Graphs with a clinical row: **{n_labeled}** "
              f"({n_labeled / n_graphs:.1%})",
              f"- Graphs without a clinical row: **{n_no_label}**",
              f"- Clinical rows without a graph (dropouts): "
              f"**{n_clin - (cohort['label_source'] == 'cohort1').sum()}**",
              "",
              "## Visit coverage of graphs",
              "",
              f"- 4 visits: **{four_visit}** (encoder-validation working set)",
              f"- 3 visits: **{three_visit}**",
              f"- 2 visits: **{two_visit}**",
              ""]

    labeled = cohort[cohort["pCR"].notna()]
    if len(labeled):
        lines += ["## pCR prevalence",
                  "",
                  f"- Overall: **{labeled['pCR'].mean():.1%}** ({int(labeled['pCR'].sum())} / {len(labeled)})",
                  ""]
        lines += ["### By subtype", ""]
        for s, g in labeled.groupby("subtype"):
            lines.append(f"- **{s}**: {g['pCR'].mean():.1%} ({int(g['pCR'].sum())} / {len(g)})")
        lines += ["", "### By arm (top 12)", ""]
        by_arm = labeled.groupby("arm")["pCR"].agg(["mean", "sum", "count"]).sort_values("count", ascending=False)
        for arm, row in by_arm.iterrows():
            lines.append(f"- **{arm}**: {row['mean']:.1%} ({int(row['sum'])} / {int(row['count'])})")
        lines += ["", "### By collection (graphs)", ""]
        for coll, g in labeled.groupby("collection"):
            lines.append(f"- **{coll}**: n={len(g)}, pCR={g['pCR'].mean():.1%}")
        lines += ["", "### Four-visit working set",
                  ""]
        fv = labeled[labeled["visit_count"] == 4]
        lines.append(f"- n=**{len(fv)}**, pCR=**{fv['pCR'].mean():.1%}**")
        in_mf = cohort["in_mfmri"].astype(bool)
        mf_fv = labeled[labeled["in_mfmri"].astype(bool) & (labeled["visit_count"] == 4)]
        lines += ["", "## Multi-feature reference check",
                  "",
                  f"- Patients in Multi-feature XLSX that are in our cohort: "
                  f"**{int(in_mf.sum())}** of 384 (diagnostic only)",
                  f"- Of those, in our 4-visit subset: **{len(mf_fv)}** "
                  f"(pCR {mf_fv['pCR'].mean():.1%} if n>0)" if len(mf_fv) else
                  "- Of those, in our 4-visit subset: 0",
                  ""]

    out_path.write_text("\n".join(lines) + "\n")


# --- CLI entry point --------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "metadata")
    ap.add_argument("--out-parquet", type=Path,
                    default=Path(__file__).resolve().parent.parent / "cohort.parquet")
    ap.add_argument("--out-audit", type=Path,
                    default=Path(__file__).resolve().parent.parent / "cohort_audit.md")
    args = ap.parse_args()

    meta = args.metadata_dir
    clinical = pd.read_excel(meta / "ISPY2-Cohort-1-Clinical-Data.xlsx", sheet_name=0)
    graph_ids = (meta / "graphs_manifest.txt").read_text().strip().splitlines()
    visit_counts = pd.read_csv(meta / "graphs_visit_counts.csv")
    mfmri = pd.read_excel(meta / "Multi-feature-MRI-NACT-Data.xlsx", sheet_name=0)
    mfmri_ids = set(mfmri["CLINICAL-TRIAL-SUBJECT-ID"].astype(int).tolist())

    cohort = build_cohort_frame(clinical, graph_ids, visit_counts, mfmri_ids)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(args.out_parquet, index=False)
    write_audit_md(cohort, clinical, args.out_audit)

    print(f"wrote {args.out_parquet}  rows={len(cohort)}")
    print(f"wrote {args.out_audit}")
    print(f"  labeled: {cohort['pCR'].notna().sum()}")
    print(f"  unlabeled: {(cohort['label_source'] == 'none').sum()}")
    print(f"  4/3/2-visit: "
          f"{(cohort['visit_count']==4).sum()}/{(cohort['visit_count']==3).sum()}/{(cohort['visit_count']==2).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
