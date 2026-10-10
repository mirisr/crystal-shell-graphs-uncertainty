#!/usr/bin/env python
"""Produce the canonical 5-fold split shared by every training stage.

Stratification key: ``subtype`` x ``pCR`` (8 strata). We chose this over
``arm`` x ``pCR`` (26 strata) because several trial arms have <25
patients, which ``StratifiedKFold`` cannot split 5 ways without dropping
them. Subtype dominates the biology of pCR anyway, so balancing on
subtype + pCR is the more defensible choice.

Output: ``datasets/ispy2/folds.parquet`` with columns

    patient_id          str     TCIA-style, matches ``cohort.parquet``
    clinical_patient_id int     numeric ID (grouping key; no leakage anyway)
    fold                int     0-4, validation fold assignment
    strat               str     ``{subtype}|{pCR}`` (for auditing)

Per the proposal, the pre-Stage encoder-validation checkpoint uses
fold 0 as its single val split; Stages 1-3 reuse all five.

Also writes ``folds_audit.md`` summarizing per-fold size and pCR balance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SEED = 20260424  # pinned; changing it invalidates every downstream run


def make_folds(cohort: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Assign each patient to a fold 0..n_splits-1, stratified by subtype+pCR.

    Drops any patient missing ``pCR`` or ``subtype`` (should be 0 given the
    current Cohort-1 join, but this keeps the function safe to run on
    partial data).
    """
    keep = cohort.dropna(subset=["pCR", "subtype"]).copy()
    keep["pCR"] = keep["pCR"].astype(int)
    keep["strat"] = keep["subtype"].astype(str) + "|" + keep["pCR"].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(keep), -1, dtype=np.int64)
    for f, (_, val_idx) in enumerate(skf.split(keep, keep["strat"])):
        fold[val_idx] = f

    out = keep[["patient_id", "clinical_patient_id", "strat"]].copy()
    out["fold"] = fold
    return out.sort_values("patient_id").reset_index(drop=True)


def audit_folds(folds: pd.DataFrame, cohort: pd.DataFrame) -> str:
    """Write a small markdown with per-fold size and pCR balance."""
    merged = folds.merge(
        cohort[["patient_id", "pCR", "subtype", "arm", "visit_count"]],
        on="patient_id", how="left",
    )
    lines = ["# Folds audit",
             "",
             f"- Total patients: **{len(merged)}**",
             f"- Seed: **{SEED}**",
             f"- Stratification: subtype x pCR (8 strata)",
             "",
             "## Per-fold composition",
             "",
             "| Fold | n | pCR rate | 4-visit | 3-visit | 2-visit |",
             "|---|---|---|---|---|---|"]
    for f in sorted(merged["fold"].unique()):
        sub = merged[merged["fold"] == f]
        vc = sub["visit_count"].value_counts().to_dict()
        lines.append(f"| {f} | {len(sub)} | {sub['pCR'].mean():.1%} | "
                     f"{vc.get(4, 0)} | {vc.get(3, 0)} | {vc.get(2, 0)} |")
    lines += ["", "## Per-fold subtype breakdown (counts)", "",
              "| Fold | " + " | ".join(sorted(merged["subtype"].unique())) + " |",
              "|---|" + "---|" * merged["subtype"].nunique()]
    subtypes = sorted(merged["subtype"].unique())
    for f in sorted(merged["fold"].unique()):
        sub = merged[merged["fold"] == f]
        counts = [str((sub["subtype"] == s).sum()) for s in subtypes]
        lines.append(f"| {f} | " + " | ".join(counts) + " |")

    lines += ["", "## Fold-0 (encoder-validation checkpoint val set)", "",
              f"Size: **{(merged['fold'] == 0).sum()}**",
              f"pCR rate: **{merged[merged['fold'] == 0]['pCR'].mean():.1%}**",
              f"4-visit subset inside fold 0: "
              f"**{((merged['fold'] == 0) & (merged['visit_count'] == 4)).sum()}**",
              ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path,
                    default=Path(__file__).resolve().parent.parent / "cohort.parquet")
    ap.add_argument("--out-parquet", type=Path,
                    default=Path(__file__).resolve().parent.parent / "folds.parquet")
    ap.add_argument("--out-audit", type=Path,
                    default=Path(__file__).resolve().parent.parent / "folds_audit.md")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    cohort = pd.read_parquet(args.cohort)
    folds = make_folds(cohort, n_splits=args.n_splits)
    folds.to_parquet(args.out_parquet, index=False)
    args.out_audit.write_text(audit_folds(folds, cohort))

    print(f"wrote {args.out_parquet}  rows={len(folds)}  (seed={SEED})")
    print(f"wrote {args.out_audit}")
    for f in sorted(folds["fold"].unique()):
        sub = folds[folds["fold"] == f]
        merged = sub.merge(cohort[["patient_id", "pCR"]], on="patient_id")
        print(f"  fold {f}: n={len(sub)} pCR={merged['pCR'].mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
