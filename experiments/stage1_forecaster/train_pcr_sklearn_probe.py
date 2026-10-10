#!/usr/bin/env python3
"""Sklearn pCR probes over clinical, graph-temporal, and model-stack features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.stage1_forecaster.train_pcr_clinical_stack import (  # noqa: E402
    _clinical_table,
    _graph_temporal_table,
    _ids,
    _load_prediction_features,
    _merge_features,
    _metrics,
    _youden_threshold,
)


def _build_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    cohort = pd.read_parquet(args.cohort)
    train_df = _clinical_table(cohort, pd.read_parquet(args.train_folds), _ids(args.train_list))
    test_df = _clinical_table(cohort, pd.read_parquet(args.test_folds), _ids(args.test_list))
    train_graph = _graph_temporal_table(args.graph_root, train_df["patient_id"].astype(str).tolist())
    test_graph = _graph_temporal_table(args.graph_root, test_df["patient_id"].astype(str).tolist())
    train_df = train_df.merge(train_graph, on="patient_id", how="left")
    test_df = test_df.merge(test_graph, on="patient_id", how="left")
    pred_df = _load_prediction_features(list(args.prediction_csv))
    train_df = _merge_features(train_df, pred_df, "ispy2_oof")
    test_df = _merge_features(test_df, pred_df, "acrin_external")
    train_df, test_df = train_df.align(test_df, join="outer", axis=1, fill_value=np.nan)
    id_cols = {"patient_id", "pCR", "fold"}
    all_non_model = [c for c in train_df.columns if c not in id_cols and not (c.startswith("prob__") or c.startswith("logit__"))]
    graph_cols = [c for c in all_non_model if c.startswith(("T0_", "T1_", "T2_", "ftv_", "alive_"))]
    clinical_cols = [c for c in all_non_model if c not in graph_cols]
    model_cols = [c for c in train_df.columns if c.startswith("prob__") or c.startswith("logit__")]
    feature_sets = {
        "clinical_only": clinical_cols,
        "graph_temporal_only": graph_cols,
        "clinical_plus_graph_temporal": clinical_cols + graph_cols,
        "model_stack_only": model_cols,
        "clinical_plus_model_stack": clinical_cols + model_cols,
        "clinical_graph_model_stack": clinical_cols + graph_cols + model_cols,
    }
    return train_df, test_df, feature_sets


def _matrix(train_df: pd.DataFrame, eval_df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_train = train_df[cols].to_numpy(dtype=float)
    x_eval = eval_df[cols].to_numpy(dtype=float)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_eval = np.where(np.isfinite(x_eval), x_eval, med)
    return x_train.astype(np.float32), x_eval.astype(np.float32)


def _model_specs(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logreg_l2_c0p1": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed)),
        "logreg_l2_c1": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed)),
        "logreg_l1_c0p1": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed)),
        "random_forest": RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=8, class_weight="balanced_subsample", random_state=seed, n_jobs=4),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=seed, n_jobs=4),
        "gradient_boost": GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, max_depth=2, min_samples_leaf=8, random_state=seed),
        "hist_gradient_boost": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_leaf_nodes=8, l2_regularization=0.1, random_state=seed),
    }


def _predict_prob(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(float)
    score = model.decision_function(x)
    return (1.0 / (1.0 + np.exp(-score))).astype(float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, default=Path("datasets/ispy2/cohort.parquet"))
    ap.add_argument("--train-list", type=Path, default=Path("reports/acrin_holdout_paper_variants/ispy2_train.txt"))
    ap.add_argument("--test-list", type=Path, default=Path("reports/acrin_holdout_paper_variants/acrin_test.txt"))
    ap.add_argument("--train-folds", type=Path, default=Path("reports/acrin_holdout_paper_variants/ispy2_train_folds.parquet"))
    ap.add_argument("--test-folds", type=Path, default=Path("reports/acrin_holdout_paper_variants/acrin_test_folds.parquet"))
    ap.add_argument("--graph-root", type=Path, default=Path("datasets/ispy2/graphs_consistent_reconae_maisi_continue_struct_acrinholdout/fold0"))
    ap.add_argument("--prediction-csv", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.set_num_threads(2)
    train_df, test_df, feature_sets = _build_tables(args)
    y_train = train_df["pCR"].to_numpy(dtype=int)
    y_test = test_df["pCR"].to_numpy(dtype=int)
    fold_ids = train_df["fold"].to_numpy(dtype=int)
    summaries: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    for feature_label, cols in feature_sets.items():
        if not cols:
            continue
        x_train_all, x_test = _matrix(train_df, test_df, cols)
        for model_name, model in _model_specs(int(args.seed)).items():
            oof = np.full(len(y_train), np.nan, dtype=float)
            for fold in sorted(np.unique(fold_ids)):
                fit = fold_ids != int(fold)
                val = fold_ids == int(fold)
                if not val.any() or len(np.unique(y_train[fit])) < 2:
                    continue
                m = _model_specs(int(args.seed) + int(fold))[model_name]
                m.fit(x_train_all[fit], y_train[fit])
                oof[val] = _predict_prob(m, x_train_all[val])
            oof_thr = _youden_threshold(y_train[np.isfinite(oof)], oof[np.isfinite(oof)])
            model.fit(x_train_all, y_train)
            train_prob = _predict_prob(model, x_train_all)
            threshold = _youden_threshold(y_train, train_prob)
            test_prob = _predict_prob(model, x_test)
            summaries.append({"feature_set": feature_label, "model": model_name, "dataset": "ispy2_oof", **_metrics(y_train, oof, oof_thr)})
            summaries.append({"feature_set": feature_label, "model": model_name, "dataset": "acrin_external", **_metrics(y_test, test_prob, threshold)})
            for pid, fold, y, p in zip(train_df["patient_id"], fold_ids, y_train, oof):
                preds.append({"feature_set": feature_label, "model": model_name, "dataset": "ispy2_oof", "patient_id": str(pid), "fold": int(fold), "pCR": int(y), "pCR_prob": float(p) if np.isfinite(p) else float("nan")})
            for pid, y, p in zip(test_df["patient_id"], y_test, test_prob):
                preds.append({"feature_set": feature_label, "model": model_name, "dataset": "acrin_external", "patient_id": str(pid), "fold": "", "pCR": int(y), "pCR_prob": float(p)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries).sort_values(["dataset", "auc"], ascending=[True, False])
    pred = pd.DataFrame(preds)
    summary.to_csv(args.out_dir / "sklearn_pcr_summary.csv", index=False)
    pred.to_csv(args.out_dir / "sklearn_pcr_predictions.csv", index=False)
    manifest = {
        "prediction_csv": [str(p) for p in args.prediction_csv],
        "out_dir": str(args.out_dir),
        "seed": int(args.seed),
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    (args.out_dir / "sklearn_pcr_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary[summary["dataset"].eq("acrin_external")].head(25).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
