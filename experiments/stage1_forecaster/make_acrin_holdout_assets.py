#!/usr/bin/env python3
"""Build ISPY2-train and ACRIN-6698 held-out split assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _has_graph(graphs_root: Path, patient_id: str) -> bool:
    return (graphs_root / f"{patient_id}.pt").is_file()


def _write_list(path: Path, patient_ids: list[str]) -> None:
    path.write_text("\n".join(patient_ids) + ("\n" if patient_ids else ""))


def _fold_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["fold"].value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def _collection_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["collection"].value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _write_fold_parquet(path: Path, df: pd.DataFrame) -> None:
    cols = ["patient_id", "fold"]
    df.loc[:, cols].sort_values(["fold", "patient_id"]).to_parquet(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, default=Path("datasets/ispy2/cohort.parquet"))
    ap.add_argument("--folds", type=Path, default=Path("datasets/ispy2/folds.parquet"))
    ap.add_argument("--graphs-root", type=Path, default=Path("datasets/ispy2/graphs_consistent"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/acrin_holdout_paper_variants"))
    ap.add_argument("--train-collection", default="ISPY2")
    ap.add_argument("--test-collection", default="ACRIN-6698")
    ap.add_argument("--min-visits", type=int, default=4)
    args = ap.parse_args()

    cohort = pd.read_parquet(args.cohort).copy()
    folds = pd.read_parquet(args.folds).copy()
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    folds["patient_id"] = folds["patient_id"].astype(str)
    folds["fold"] = folds["fold"].astype(int)

    merged = cohort.merge(folds[["patient_id", "fold"]], on="patient_id", how="inner")
    merged["has_split_graph"] = merged["patient_id"].map(lambda pid: _has_graph(args.graphs_root, str(pid)))
    eligible = merged[
        (merged["visit_count"].fillna(0).astype(int) >= int(args.min_visits))
        & merged["has_split_graph"]
    ].copy()

    train_df = eligible[eligible["collection"].astype(str) == str(args.train_collection)].copy()
    test_df = eligible[eligible["collection"].astype(str) == str(args.test_collection)].copy()
    union_df = pd.concat([train_df, test_df], ignore_index=True)

    train_ids = train_df.sort_values(["fold", "patient_id"])["patient_id"].astype(str).tolist()
    test_ids = test_df.sort_values(["fold", "patient_id"])["patient_id"].astype(str).tolist()
    union_ids = union_df.sort_values(["fold", "patient_id"])["patient_id"].astype(str).tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_list(args.out_dir / "ispy2_train.txt", train_ids)
    _write_list(args.out_dir / "acrin_test.txt", test_ids)
    _write_list(args.out_dir / "ispy2_acrin_eval.txt", union_ids)
    _write_fold_parquet(args.out_dir / "ispy2_train_folds.parquet", train_df)
    _write_fold_parquet(args.out_dir / "acrin_test_folds.parquet", test_df)
    _write_fold_parquet(args.out_dir / "ispy2_acrin_folds.parquet", union_df)

    manifest: dict[str, Any] = {
        "cohort": str(args.cohort),
        "folds": str(args.folds),
        "graphs_root": str(args.graphs_root),
        "train_collection": str(args.train_collection),
        "test_collection": str(args.test_collection),
        "min_visits": int(args.min_visits),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_union": int(len(union_df)),
        "train_fold_counts": _fold_counts(train_df),
        "test_fold_counts": _fold_counts(test_df),
        "union_collection_counts": _collection_counts(union_df),
        "outputs": {
            "train_list": str(args.out_dir / "ispy2_train.txt"),
            "test_list": str(args.out_dir / "acrin_test.txt"),
            "union_list": str(args.out_dir / "ispy2_acrin_eval.txt"),
            "train_folds": str(args.out_dir / "ispy2_train_folds.parquet"),
            "test_folds": str(args.out_dir / "acrin_test_folds.parquet"),
            "union_folds": str(args.out_dir / "ispy2_acrin_folds.parquet"),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
