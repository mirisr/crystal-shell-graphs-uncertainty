#!/usr/bin/env python3
"""Stack model-derived pCR scores with baseline clinical covariates."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class FitConfig:
    epochs: int = 1200
    lr: float = 0.02
    weight_decay: float = 1e-2
    hidden_dim: int = 32
    dropout: float = 0.20
    seed: int = 42


def roc_auc_score_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            ranks[inv == idx] = ranks[inv == idx].mean()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    return float(((tp / (np.arange(len(ys)) + 1.0)) * ys).sum() / n_pos)


def _youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    best_t, best_j = 0.5, -float("inf")
    for t in np.unique(p):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    pred = (p >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "n": float(len(y)),
        "pcr_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": roc_auc_score_rank(y, p),
        "average_precision": average_precision(y, p),
        "brier": float(np.mean((p - y.astype(float)) ** 2)) if len(y) else float("nan"),
        "threshold": float(threshold),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


class LinearHead(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Linear(int(n_features), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(n_features), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, cfg: FitConfig, model_type: str) -> np.ndarray:
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    x_train = np.asarray(x_train, dtype=np.float32)
    x_eval = np.asarray(x_eval, dtype=np.float32)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_eval = np.where(np.isfinite(x_eval), x_eval, med)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    xt = torch.as_tensor((x_train - mean) / std, dtype=torch.float32)
    xe = torch.as_tensor((x_eval - mean) / std, dtype=torch.float32)
    yt = torch.as_tensor(y_train.astype(np.float32), dtype=torch.float32)
    if model_type == "mlp":
        model: nn.Module = MLPHead(xt.shape[1], int(cfg.hidden_dim), float(cfg.dropout))
    else:
        model = LinearHead(xt.shape[1])
    n_pos = float(yt.sum().item())
    n_neg = float(len(yt) - n_pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    model.train()
    for _ in range(int(cfg.epochs)):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xe)).detach().cpu().numpy().astype(float)


def _ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _clinical_table(cohort: pd.DataFrame, folds: pd.DataFrame, patient_ids: list[str]) -> pd.DataFrame:
    cohort = cohort.copy()
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    folds = folds[["patient_id", "fold"]].copy()
    folds["patient_id"] = folds["patient_id"].astype(str)
    df = cohort[cohort["patient_id"].isin(patient_ids)].merge(folds, on="patient_id", how="left")
    df = df[pd.to_numeric(df["pCR"], errors="coerce").notna()].copy()
    df["pCR"] = df["pCR"].astype(int)
    for col in ("HR", "HER2", "MP", "age"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cat = pd.get_dummies(df[["subtype", "arm"]].fillna("missing"), prefix=["subtype", "arm"], dtype=float)
    num = df[["age", "HR", "HER2", "MP"]].astype(float)
    out = pd.concat([df[["patient_id", "pCR", "fold"]].reset_index(drop=True), num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    return out


def _graph_temporal_table(graph_root: Path, patient_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pid in patient_ids:
        gp = graph_root / f"{pid}.pt"
        if not gp.is_file():
            continue
        try:
            g = torch.load(gp, map_location="cpu", weights_only=False)
            offsets = [int(v) for v in g["visit_offsets"].tolist()]
            row: dict[str, Any] = {"patient_id": pid}
            ftv: list[float] = []
            alive_counts: list[float] = []
            for visit_idx, visit_name in enumerate(("T0", "T1", "T2")):
                x = g["x"][offsets[visit_idx]:offsets[visit_idx + 1]].float()
                alive = g["alive"][offsets[visit_idx]:offsets[visit_idx + 1]].bool()
                if bool(alive.any()):
                    x_alive = x[alive]
                else:
                    x_alive = x
                vol = x_alive[:, 1].detach().cpu().numpy().astype(float) if x_alive.shape[1] > 1 else np.ones((int(x_alive.shape[0]),), dtype=float)
                vol = np.nan_to_num(vol, nan=0.0)
                ftv_value = float(np.clip(vol, 0.0, None).sum())
                ftv.append(ftv_value)
                alive_counts.append(float(x_alive.shape[0]))
                row[f"{visit_name}_ftv"] = ftv_value
                row[f"{visit_name}_node_count"] = float(x_alive.shape[0])
                row[f"{visit_name}_max_node_volume"] = float(np.max(vol)) if len(vol) else 0.0
                core = x_alive[:, : min(6, int(x_alive.shape[1]))].detach().cpu().numpy().astype(float)
                if core.size:
                    for j in range(core.shape[1]):
                        row[f"{visit_name}_core{j}_mean"] = float(np.nanmean(core[:, j]))
                        row[f"{visit_name}_core{j}_max"] = float(np.nanmax(core[:, j]))
            for a, b in ((0, 1), (1, 2), (0, 2)):
                va = ftv[a]
                vb = ftv[b]
                row[f"ftv_delta_T{b}_T{a}"] = float(vb - va)
                row[f"ftv_ratio_T{b}_T{a}"] = float(vb / va) if va > 1e-8 else 0.0
                row[f"alive_delta_T{b}_T{a}"] = float(alive_counts[b] - alive_counts[a])
                row[f"alive_ratio_T{b}_T{a}"] = float(alive_counts[b] / alive_counts[a]) if alive_counts[a] > 1e-8 else 0.0
            rows.append(row)
        except Exception as exc:
            print(f"[warn] graph temporal features failed for {pid}: {exc}", flush=True)
    return pd.DataFrame(rows)


def _load_prediction_features(paths: list[Path]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        name_parts = []
        for col in ("condition", "feature_set", "model"):
            if col in df.columns:
                name_parts.append(df[col].astype(str))
        if name_parts:
            name = name_parts[0]
            for part in name_parts[1:]:
                name = name + "__" + part
        else:
            name = pd.Series([path.stem] * len(df), index=df.index)
        out = df[["dataset", "patient_id", "pCR", "pCR_prob"]].copy()
        out["feature_name"] = path.parent.name + "__" + name.astype(str)
        pieces.append(out)
    if not pieces:
        return pd.DataFrame(columns=["dataset", "patient_id", "pCR"])
    long = pd.concat(pieces, ignore_index=True)
    long["pCR_prob"] = pd.to_numeric(long["pCR_prob"], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    long["pCR_logit"] = np.log(long["pCR_prob"] / (1.0 - long["pCR_prob"]))
    prob_wide = long.pivot_table(index=["dataset", "patient_id", "pCR"], columns="feature_name", values="pCR_prob", aggfunc="mean")
    logit_wide = long.pivot_table(index=["dataset", "patient_id", "pCR"], columns="feature_name", values="pCR_logit", aggfunc="mean")
    prob_wide.columns = [f"prob__{c}" for c in prob_wide.columns]
    logit_wide.columns = [f"logit__{c}" for c in logit_wide.columns]
    return pd.concat([prob_wide, logit_wide], axis=1).reset_index()


def _merge_features(base: pd.DataFrame, pred: pd.DataFrame, dataset: str) -> pd.DataFrame:
    p = pred[pred["dataset"] == dataset].drop(columns=["dataset", "pCR"], errors="ignore").copy()
    return base.merge(p, on="patient_id", how="left")


def _evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], label: str, cfg: FitConfig, model_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_train = train_df["pCR"].to_numpy(dtype=int)
    y_test = test_df["pCR"].to_numpy(dtype=int)
    x_train = train_df[feature_cols].to_numpy(dtype=float)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    oof = np.full(len(y_train), np.nan, dtype=float)
    for fold in sorted(train_df["fold"].astype(int).unique()):
        fit = train_df["fold"].to_numpy() != int(fold)
        val = train_df["fold"].to_numpy() == int(fold)
        if val.any() and len(np.unique(y_train[fit])) == 2:
            oof[val] = _fit_predict(x_train[fit], y_train[fit], x_train[val], FitConfig(**{**asdict(cfg), "seed": int(cfg.seed) + int(fold)}), model_type)
    oof_thr = _youden_threshold(y_train[np.isfinite(oof)], oof[np.isfinite(oof)])
    train_prob = _fit_predict(x_train, y_train, x_train, cfg, model_type)
    threshold = _youden_threshold(y_train, train_prob)
    test_prob = _fit_predict(x_train, y_train, x_test, cfg, model_type)
    summary = [
        {"feature_set": label, "model": model_type, "dataset": "ispy2_oof", **_metrics(y_train, oof, oof_thr)},
        {"feature_set": label, "model": model_type, "dataset": "acrin_external", **_metrics(y_test, test_prob, threshold)},
    ]
    preds: list[dict[str, Any]] = []
    for pid, fold, y, p in zip(train_df["patient_id"], train_df["fold"], y_train, oof):
        preds.append({"feature_set": label, "model": model_type, "dataset": "ispy2_oof", "patient_id": str(pid), "fold": int(fold), "pCR": int(y), "pCR_prob": float(p) if np.isfinite(p) else float("nan")})
    for pid, y, p in zip(test_df["patient_id"], y_test, test_prob):
        preds.append({"feature_set": label, "model": model_type, "dataset": "acrin_external", "patient_id": str(pid), "fold": "", "pCR": int(y), "pCR_prob": float(p)})
    return summary, preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, default=Path("datasets/ispy2/cohort.parquet"))
    ap.add_argument("--train-list", type=Path, default=Path("reports/acrin_holdout_paper_variants/ispy2_train.txt"))
    ap.add_argument("--test-list", type=Path, default=Path("reports/acrin_holdout_paper_variants/acrin_test.txt"))
    ap.add_argument("--train-folds", type=Path, default=Path("reports/acrin_holdout_paper_variants/ispy2_train_folds.parquet"))
    ap.add_argument("--test-folds", type=Path, default=Path("reports/acrin_holdout_paper_variants/acrin_test_folds.parquet"))
    ap.add_argument("--prediction-csv", type=Path, action="append", default=[])
    ap.add_argument("--graph-root", type=Path, default=Path("datasets/ispy2/graphs_consistent_reconae_maisi_continue_struct_acrinholdout/fold0"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=FitConfig.epochs)
    ap.add_argument("--lr", type=float, default=FitConfig.lr)
    ap.add_argument("--weight-decay", type=float, default=FitConfig.weight_decay)
    ap.add_argument("--hidden-dim", type=int, default=FitConfig.hidden_dim)
    ap.add_argument("--dropout", type=float, default=FitConfig.dropout)
    ap.add_argument("--seed", type=int, default=FitConfig.seed)
    args = ap.parse_args()
    cfg = FitConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
    )
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
    clinical_cols = [c for c in train_df.columns if c not in id_cols and not (c.startswith("prob__") or c.startswith("logit__"))]
    graph_cols = [c for c in clinical_cols if c.startswith(("T0_", "T1_", "T2_", "ftv_", "alive_"))]
    clinical_base_cols = [c for c in clinical_cols if c not in graph_cols]
    model_cols = [c for c in train_df.columns if c.startswith("prob__") or c.startswith("logit__")]
    feature_sets = {
        "clinical_only": clinical_base_cols,
        "graph_temporal_only": graph_cols,
        "clinical_plus_graph_temporal": clinical_base_cols + graph_cols,
        "model_stack_only": model_cols,
        "clinical_plus_model_stack": clinical_base_cols + model_cols,
        "clinical_graph_model_stack": clinical_base_cols + graph_cols + model_cols,
    }
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for label, cols in feature_sets.items():
        if not cols:
            continue
        for model_type in ("linear", "mlp"):
            s, p = _evaluate(train_df, test_df, cols, label, cfg, model_type)
            summaries.extend(s)
            predictions.extend(p)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries).sort_values(["feature_set", "model", "dataset"])
    pred_out = pd.DataFrame(predictions)
    summary_df.to_csv(args.out_dir / "clinical_stack_pcr_summary.csv", index=False)
    pred_out.to_csv(args.out_dir / "clinical_stack_pcr_predictions.csv", index=False)
    manifest = {
        "prediction_csv": [str(p) for p in args.prediction_csv],
        "out_dir": str(args.out_dir),
        "fit": asdict(cfg),
        "clinical_features": clinical_cols,
        "graph_temporal_features": graph_cols,
        "model_features": model_cols,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    (args.out_dir / "clinical_stack_pcr_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary_df.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
