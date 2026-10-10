#!/usr/bin/env python3
"""Post-hoc UQ method sweep for the selected paper model.

The inputs are per-patient conditional MC summaries from the fixed
ResNet-Crystal-EdgeGate-MC paper model. I-SPY2 out-of-fold rows are used for
fitting, conformal calibration, and method ranking. ACRIN is scored once after
the I-SPY2 winner is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - Cradle has scipy; local smoke may not.
    scipy_stats = None


EXCLUDE_FEATURE_SUBSTRINGS = (
    "obs",
    "coverage",
    "nonconformity",
    "conformal",
    "crps",
    "abs_err",
    "swd",
    "chamfer",
    "dice",
    "iou",
    "hausdorff",
    "surface",
    "betti",
)

EXCLUDE_FEATURE_COLS = {
    "patient_id",
    "collection",
    "pCR",
    "subtype",
    "start_visit",
    "predicted_visit",
    "residual_stratify_by",
    "residual_stratum",
    "residual_sampler_mode",
    "sampler_selection_rule",
    "pred_ftv_source",
    "obs_ftv_ml",
}


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _log_obs(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(0.0, np.asarray(y, dtype=float)))


def _log_to_ml(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(z, dtype=float)))


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    level = min(1.0, math.ceil((scores.size + 1) * (1.0 - float(alpha))) / scores.size)
    return float(np.quantile(scores, level, method="higher"))


def _student_t_quantile(prob: float, df: float) -> float:
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(float(prob), df=float(df)))
    z = NormalDist().inv_cdf(float(prob))
    nu = max(float(df), 1.0)
    # Cornish-Fisher expansion for Student-t; accurate enough for smoke tests.
    return float(
        z
        + (z**3 + z) / (4.0 * nu)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * nu**2)
    )


def _interval_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return {
            "n": 0.0,
            "mae_ml": float("nan"),
            "rmse_ml": float("nan"),
            "bias_ml": float("nan"),
            "coverage": float("nan"),
            "width_ml": float("nan"),
            "interval_score": float("nan"),
        }
    yy = y[ok]
    pp = pred[ok]
    ll = np.maximum(0.0, lo[ok])
    hh = np.maximum(ll, hi[ok])
    lower_miss = np.maximum(0.0, ll - yy)
    upper_miss = np.maximum(0.0, yy - hh)
    score = (hh - ll) + (2.0 / float(alpha)) * (lower_miss + upper_miss)
    return {
        "n": float(ok.sum()),
        "mae_ml": float(mean_absolute_error(yy, pp)),
        "rmse_ml": float(mean_squared_error(yy, pp) ** 0.5),
        "bias_ml": float(np.mean(pp - yy)),
        "coverage": float(np.mean((yy >= ll) & (yy <= hh))),
        "width_ml": float(np.mean(hh - ll)),
        "interval_score": float(np.mean(score)),
    }


def _fold_ids(train: pd.DataFrame) -> np.ndarray:
    if "fold" not in train.columns:
        raise RuntimeError("Training table must include fold for OOF calibration.")
    return pd.to_numeric(train["fold"], errors="coerce").fillna(-1).astype(int).to_numpy()


def _load_table(path: Path, visit_pair: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if visit_pair != "all":
        start, pred = visit_pair.split("_to_")
        df = df[(df["start_visit"] == start) & (df["predicted_visit"] == pred)].copy()
    if df.empty:
        raise RuntimeError(f"No rows in {path} for visit_pair={visit_pair}")
    return df.reset_index(drop=True)


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("pred_ftv_det_ml", "ftv_mc_mean_ml", "ftv_mc_std_ml", "ftv_raw_width90_ml", "ftv_t0_ml", "alive_mass_det"):
        if col in out.columns:
            out[f"log1p_{col}"] = np.log1p(np.maximum(0.0, pd.to_numeric(out[col], errors="coerce")))
    if {"ftv_mc_std_ml", "ftv_mc_mean_ml"}.issubset(out.columns):
        mean = np.maximum(1.0, pd.to_numeric(out["ftv_mc_mean_ml"], errors="coerce"))
        out["mc_cv"] = pd.to_numeric(out["ftv_mc_std_ml"], errors="coerce") / mean
    if {"ftv_raw_width90_ml", "ftv_mc_mean_ml"}.issubset(out.columns):
        mean = np.maximum(1.0, pd.to_numeric(out["ftv_mc_mean_ml"], errors="coerce"))
        out["raw_width_to_mean"] = pd.to_numeric(out["ftv_raw_width90_ml"], errors="coerce") / mean
    if {"ftv_mc_mean_ml", "ftv_t0_ml"}.issubset(out.columns):
        base = np.maximum(1.0, pd.to_numeric(out["ftv_t0_ml"], errors="coerce"))
        out["mc_mean_to_t0"] = pd.to_numeric(out["ftv_mc_mean_ml"], errors="coerce") / base
    if "subtype" in out.columns:
        dummies = pd.get_dummies(out["subtype"].astype(str), prefix="subtype", dummy_na=True)
        out = pd.concat([out, dummies], axis=1)
    if "start_visit" in out.columns:
        out = pd.concat([out, pd.get_dummies(out["start_visit"].astype(str), prefix="start")], axis=1)
    if "predicted_visit" in out.columns:
        out = pd.concat([out, pd.get_dummies(out["predicted_visit"].astype(str), prefix="pred")], axis=1)
    return out


def _feature_columns(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> list[str]:
    common = [c for c in train.columns if c in test.columns and c not in EXCLUDE_FEATURE_COLS]
    numeric = [
        c for c in common
        if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c])
    ]
    numeric = [
        c for c in numeric
        if not any(token in c.lower() for token in EXCLUDE_FEATURE_SUBSTRINGS)
    ]
    if mode == "paper_scalar":
        prefixes = (
            "log1p_",
            "mc_",
            "raw_width_to_mean",
            "start_",
            "pred_",
            "subtype_",
        )
        keep_names = {
            "fold",
            "start_idx",
            "pred_idx",
            "rollout_depth",
            "n_mc",
            "n_supervoxels",
            "n_calibration_patients",
            "pred_ftv_det_ml",
            "ftv_mc_mean_ml",
            "ftv_mc_std_ml",
            "ftv_mc_median_ml",
            "ftv_raw_p05_ml",
            "ftv_raw_p95_ml",
            "ftv_raw_width90_ml",
            "pcr_prob_mc",
            "alive_mass_det",
            "ftv_t0_ml",
            "mc_mean_to_t0",
        }
        keep = [c for c in numeric if c in keep_names or c.startswith(prefixes)]
    elif mode == "paper_full":
        keep = numeric
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    keep = sorted([c for c in keep if c != "fold"])
    if not keep:
        raise RuntimeError(f"No feature columns selected for mode {mode}")
    return keep


def _to_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def _fit_quantile_boosting(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    x_test: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    alphas = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
    oof = {k: np.full(len(x_train), np.nan, dtype=float) for k in alphas}
    test = {k: np.full(len(x_test), np.nan, dtype=float) for k in alphas}
    for key, alpha in alphas.items():
        for fold in sorted(set(int(v) for v in folds if v >= 0)):
            tr = folds != fold
            va = folds == fold
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=float(alpha),
                    n_estimators=300,
                    learning_rate=0.035,
                    max_depth=2,
                    min_samples_leaf=12,
                    random_state=seed + fold,
                ),
            )
            model.fit(x_train[tr], y_train_log[tr])
            oof[key][va] = model.predict(x_train[va])
        final = make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                loss="quantile",
                alpha=float(alpha),
                n_estimators=340,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=12,
                random_state=seed + 101,
            ),
        )
        final.fit(x_train, y_train_log)
        test[key] = final.predict(x_test)
    return oof, test


def _fit_extra_trees_quantiles(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    x_test: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof = {k: np.full(len(x_train), np.nan, dtype=float) for k in ("q05", "q50", "q95")}
    test = {k: np.full(len(x_test), np.nan, dtype=float) for k in ("q05", "q50", "q95")}
    for fold in sorted(set(int(v) for v in folds if v >= 0)):
        tr = folds != fold
        va = folds == fold
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=600,
                min_samples_leaf=8,
                max_features=0.75,
                random_state=seed + fold,
                n_jobs=-1,
            ),
        )
        model.fit(x_train[tr], y_train_log[tr])
        forest = model[-1]
        xva = model[:-1].transform(x_train[va])
        tree_preds = np.stack([est.predict(xva) for est in forest.estimators_], axis=0)
        oof["q05"][va], oof["q50"][va], oof["q95"][va] = np.quantile(tree_preds, [0.05, 0.50, 0.95], axis=0)
    final = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=8,
            max_features=0.75,
            random_state=seed + 101,
            n_jobs=-1,
        ),
    )
    final.fit(x_train, y_train_log)
    forest = final[-1]
    xt = final[:-1].transform(x_test)
    tree_preds = np.stack([est.predict(xt) for est in forest.estimators_], axis=0)
    test["q05"], test["q50"], test["q95"] = np.quantile(tree_preds, [0.05, 0.50, 0.95], axis=0)
    return oof, test


def _fit_scaled_abs_residual(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    center_train_log: np.ndarray,
    x_test: np.ndarray,
    center_test_log: np.ndarray,
    folds: np.ndarray,
    seed: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.abs(y_train_log - center_train_log)
    oof_scale = np.full(len(x_train), np.nan, dtype=float)
    for fold in sorted(set(int(v) for v in folds if v >= 0)):
        tr = folds != fold
        va = folds == fold
        if method == "extra_trees":
            reg = ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=8,
                max_features=0.75,
                random_state=seed + fold,
                n_jobs=-1,
            )
        elif method == "gbrt":
            reg = GradientBoostingRegressor(
                loss="squared_error",
                n_estimators=240,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=12,
                random_state=seed + fold,
            )
        else:
            raise ValueError(method)
        model = make_pipeline(SimpleImputer(strategy="median"), reg)
        model.fit(x_train[tr], target[tr])
        oof_scale[va] = np.maximum(1e-4, model.predict(x_train[va]))
    if method == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=8,
            max_features=0.75,
            random_state=seed + 101,
            n_jobs=-1,
        )
    else:
        reg = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=280,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=12,
            random_state=seed + 101,
        )
    final = make_pipeline(SimpleImputer(strategy="median"), reg)
    final.fit(x_train, target)
    test_scale = np.maximum(1e-4, final.predict(x_test))
    _ = center_test_log
    return np.maximum(1e-4, oof_scale), test_scale


@dataclass
class StudentConfig:
    hidden: int = 96
    epochs: int = 500
    patience: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    df: float = 4.0


class StudentHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_scale = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu(h).squeeze(-1)
        scale = F.softplus(self.log_scale(h).squeeze(-1)) + 1e-4
        return mu, scale


def _student_t_nll(y: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, df: float) -> torch.Tensor:
    z = (y - mu) / scale
    nu = torch.as_tensor(float(df), dtype=y.dtype, device=y.device)
    log_norm = (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu * torch.pi)
        - torch.log(scale)
    )
    log_prob = log_norm - ((nu + 1.0) / 2.0) * torch.log1p((z * z) / nu)
    return -log_prob.mean()


def _fit_one_student(
    x_scaled: np.ndarray,
    y: np.ndarray,
    dev_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    cfg: StudentConfig,
) -> StudentHead:
    torch.manual_seed(seed)
    random.seed(seed)
    model = StudentHead(x_scaled.shape[1], cfg.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.from_numpy(x_scaled)
    yt = torch.from_numpy(y)
    best_state = None
    best_val = float("inf")
    bad = 0
    for _epoch in range(int(cfg.epochs)):
        model.train()
        opt.zero_grad()
        mu, scale = model(xt[dev_idx])
        loss = _student_t_nll(yt[dev_idx], mu, scale, cfg.df)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            mu_v, scale_v = model(xt[val_idx])
            val_loss = float(_student_t_nll(yt[val_idx], mu_v, scale_v, cfg.df).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(cfg.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _inner_split(idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.array(idx, dtype=int)
    rng.shuffle(order)
    split = max(1, int(0.8 * len(order)))
    if split >= len(order):
        split = max(1, len(order) - 1)
    return order[:split], order[split:]


def _fit_student_head(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    x_test: np.ndarray,
    folds: np.ndarray,
    seed: int,
    cfg: StudentConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imputer.fit_transform(x_train)).astype(np.float32)
    x_test_scaled = scaler.transform(imputer.transform(x_test)).astype(np.float32)
    y = y_train_log.astype(np.float32)

    oof_mu = np.full(len(x_train), np.nan, dtype=float)
    oof_scale = np.full(len(x_train), np.nan, dtype=float)
    for fold in sorted(set(int(v) for v in folds if v >= 0)):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        if len(tr) < 8 or len(va) == 0:
            continue
        dev_idx, val_idx = _inner_split(tr, seed + fold)
        model = _fit_one_student(x_scaled, y, dev_idx, val_idx, seed + fold, cfg)
        with torch.no_grad():
            mu_va, scale_va = model(torch.from_numpy(x_scaled[va]))
        oof_mu[va] = mu_va.numpy()
        oof_scale[va] = scale_va.numpy()

    holdout_fold = sorted(set(int(v) for v in folds if v >= 0))[-1]
    val = folds == holdout_fold
    if int(val.sum()) < 20:
        dev_idx, val_idx = _inner_split(np.arange(len(x_train)), seed)
    else:
        dev_idx, val_idx = np.where(folds != holdout_fold)[0], np.where(val)[0]
    final = _fit_one_student(x_scaled, y, dev_idx, val_idx, seed, cfg)
    with torch.no_grad():
        mu_tr, scale_tr = final(torch.from_numpy(x_scaled))
        mu_te, scale_te = final(torch.from_numpy(x_test_scaled))
    return (
        {"mu": oof_mu, "scale": np.maximum(1e-4, oof_scale)},
        {"mu": mu_tr.numpy(), "scale": scale_tr.numpy()},
        {"mu": mu_te.numpy(), "scale": scale_te.numpy()},
    )


def _oof_conformal_quantiles(scores: np.ndarray, folds: np.ndarray, alpha: float) -> np.ndarray:
    """Conformal quantile for each row using other folds only."""
    out = np.full(len(scores), np.nan, dtype=float)
    scores = np.asarray(scores, dtype=float)
    folds = np.asarray(folds, dtype=int)
    for fold in sorted(set(int(v) for v in folds if v >= 0)):
        cal = (folds != fold) & np.isfinite(scores)
        out[folds == fold] = _conformal_quantile(scores[cal], alpha)
    return out


def _add_summary(
    rows: list[dict[str, Any]],
    preds: list[pd.DataFrame],
    *,
    split: str,
    feature_mode: str,
    method: str,
    y: np.ndarray,
    pred: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float,
    base: pd.DataFrame,
    extra: dict[str, Any] | None = None,
) -> None:
    metrics = _interval_metrics(y, pred, lo, hi, alpha)
    row: dict[str, Any] = {"split": split, "feature_mode": feature_mode, "method": method, **metrics}
    if extra:
        row.update(extra)
    rows.append(row)
    out = base[["patient_id", "start_visit", "predicted_visit", "fold", "collection", "subtype", "pCR", "obs_ftv_ml"]].copy()
    out["split"] = split
    out["feature_mode"] = feature_mode
    out["method"] = method
    out["pred_ml"] = pred
    out["lo90_ml"] = np.maximum(0.0, lo)
    out["hi90_ml"] = np.maximum(np.maximum(0.0, lo), hi)
    out["covered90"] = (pd.to_numeric(out["obs_ftv_ml"], errors="coerce") >= out["lo90_ml"]) & (
        pd.to_numeric(out["obs_ftv_ml"], errors="coerce") <= out["hi90_ml"]
    )
    preds.append(out)


FEATURE_INDEPENDENT_METHODS = {
    "paper_mc_raw",
    "paper_mc_existing_conformal",
    "paper_log_conformal_global",
    "paper_log_conformal_by_visit",
}
RANKING_EXCLUDED_METHODS = {
    "paper_mc_raw",
    "paper_mc_existing_conformal",
}
RANK_KEYS = ("interval_score", "coverage", "width_ml")
COVERAGE_TARGET = 0.90
COVERAGE_TOLERANCE = 0.02


def _emit_method(
    rows: list[dict[str, Any]],
    preds: list[pd.DataFrame],
    *,
    feature_mode: str,
    method: str,
    alpha: float,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_oof: np.ndarray,
    lo_oof: np.ndarray,
    hi_oof: np.ndarray,
    pred_te: np.ndarray,
    lo_te: np.ndarray,
    hi_te: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> None:
    _add_summary(
        rows,
        preds,
        split="ispy2_oof",
        feature_mode=feature_mode,
        method=method,
        y=y_train,
        pred=pred_oof,
        lo=lo_oof,
        hi=hi_oof,
        alpha=alpha,
        base=train,
        extra=extra,
    )
    _add_summary(
        rows,
        preds,
        split="acrin_holdout",
        feature_mode=feature_mode,
        method=method,
        y=y_test,
        pred=pred_te,
        lo=lo_te,
        hi=hi_te,
        alpha=alpha,
        base=test,
        extra=extra,
    )


def _visit_oof_quantiles(
    train: pd.DataFrame,
    scores: np.ndarray,
    folds: np.ndarray,
    alpha: float,
) -> np.ndarray:
    q = np.full(len(train), np.nan, dtype=float)
    start = train["start_idx"].to_numpy()
    pred = train["pred_idx"].to_numpy()
    for fold in sorted(set(int(v) for v in folds if v >= 0)):
        for s, p in pd.DataFrame({"s": start, "p": pred}).drop_duplicates().itertuples(index=False):
            cal = (folds != fold) & (start == s) & (pred == p) & np.isfinite(scores)
            va = (folds == fold) & (start == s) & (pred == p)
            if not va.any():
                continue
            q[va] = _conformal_quantile(scores[cal], alpha)
    return q


def _run_feature_mode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_mode: str,
    alpha: float,
    seed: int,
    student_cfg: StudentConfig,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[str]]:
    cols = _feature_columns(train, test, feature_mode)
    x_train = _to_matrix(train, cols)
    x_test = _to_matrix(test, cols)
    y_train = pd.to_numeric(train["obs_ftv_ml"], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test["obs_ftv_ml"], errors="coerce").to_numpy(dtype=float)
    y_train_log = _log_obs(y_train)
    folds = _fold_ids(train)
    rows: list[dict[str, Any]] = []
    preds: list[pd.DataFrame] = []

    mc_train = pd.to_numeric(train["ftv_mc_mean_ml"], errors="coerce").to_numpy(dtype=float)
    mc_test = pd.to_numeric(test["ftv_mc_mean_ml"], errors="coerce").to_numpy(dtype=float)
    mc_train_log = _log_obs(mc_train)
    mc_test_log = _log_obs(mc_test)
    emit = dict(
        rows=rows,
        preds=preds,
        feature_mode=feature_mode,
        alpha=alpha,
        train=train,
        test=test,
        y_train=y_train,
        y_test=y_test,
    )

    _emit_method(
        **emit,
        method="paper_mc_raw",
        pred_oof=mc_train,
        lo_oof=pd.to_numeric(train["ftv_raw_p05_ml"], errors="coerce").to_numpy(dtype=float),
        hi_oof=pd.to_numeric(train["ftv_raw_p95_ml"], errors="coerce").to_numpy(dtype=float),
        pred_te=mc_test,
        lo_te=pd.to_numeric(test["ftv_raw_p05_ml"], errors="coerce").to_numpy(dtype=float),
        hi_te=pd.to_numeric(test["ftv_raw_p95_ml"], errors="coerce").to_numpy(dtype=float),
    )
    if {"ftv_conformal_p05_ml", "ftv_conformal_p95_ml"}.issubset(train.columns) and {
        "ftv_conformal_p05_ml",
        "ftv_conformal_p95_ml",
    }.issubset(test.columns):
        _emit_method(
            **emit,
            method="paper_mc_existing_conformal",
            pred_oof=mc_train,
            lo_oof=pd.to_numeric(train["ftv_conformal_p05_ml"], errors="coerce").to_numpy(dtype=float),
            hi_oof=pd.to_numeric(train["ftv_conformal_p95_ml"], errors="coerce").to_numpy(dtype=float),
            pred_te=mc_test,
            lo_te=pd.to_numeric(test["ftv_conformal_p05_ml"], errors="coerce").to_numpy(dtype=float),
            hi_te=pd.to_numeric(test["ftv_conformal_p95_ml"], errors="coerce").to_numpy(dtype=float),
        )

    abs_log_resid = np.abs(y_train_log - mc_train_log)
    q_oof = _oof_conformal_quantiles(abs_log_resid, folds, alpha)
    q_global = _conformal_quantile(abs_log_resid, alpha)
    _emit_method(
        **emit,
        method="paper_log_conformal_global",
        pred_oof=mc_train,
        lo_oof=_log_to_ml(mc_train_log - q_oof),
        hi_oof=_log_to_ml(mc_train_log + q_oof),
        pred_te=mc_test,
        lo_te=_log_to_ml(mc_test_log - q_global),
        hi_te=_log_to_ml(mc_test_log + q_global),
        extra={"q": q_global},
    )

    q_visit_oof = _visit_oof_quantiles(train, abs_log_resid, folds, alpha)
    lo_bucket = np.full(len(test), np.nan)
    hi_bucket = np.full(len(test), np.nan)
    for key, _grp in train.groupby(["start_idx", "pred_idx"], dropna=False):
        tr_mask = (train["start_idx"] == key[0]) & (train["pred_idx"] == key[1])
        q = _conformal_quantile(abs_log_resid[tr_mask.to_numpy()], alpha)
        te_mask = (test["start_idx"] == key[0]) & (test["pred_idx"] == key[1])
        idx = np.where(te_mask.to_numpy())[0]
        lo_bucket[idx] = _log_to_ml(mc_test_log[idx] - q)
        hi_bucket[idx] = _log_to_ml(mc_test_log[idx] + q)
    _emit_method(
        **emit,
        method="paper_log_conformal_by_visit",
        pred_oof=mc_train,
        lo_oof=_log_to_ml(mc_train_log - q_visit_oof),
        hi_oof=_log_to_ml(mc_train_log + q_visit_oof),
        pred_te=mc_test,
        lo_te=lo_bucket,
        hi_te=hi_bucket,
    )

    for method_name, fitter in (
        ("extra_trees_quantile", _fit_extra_trees_quantiles),
        ("quantile_gbrt", _fit_quantile_boosting),
    ):
        oof, te = fitter(x_train, y_train_log, x_test, folds, seed)
        pred_oof = _log_to_ml(oof["q50"])
        pred_te = _log_to_ml(te["q50"])
        _emit_method(
            **emit,
            method=f"{method_name}_raw",
            pred_oof=pred_oof,
            lo_oof=_log_to_ml(oof["q05"]),
            hi_oof=_log_to_ml(oof["q95"]),
            pred_te=pred_te,
            lo_te=_log_to_ml(te["q05"]),
            hi_te=_log_to_ml(te["q95"]),
        )
        score = np.maximum(oof["q05"] - y_train_log, y_train_log - oof["q95"])
        q_oof_cqr = _oof_conformal_quantiles(score, folds, alpha)
        q_cqr = _conformal_quantile(score, alpha)
        _emit_method(
            **emit,
            method=f"{method_name}_cqr",
            pred_oof=pred_oof,
            lo_oof=_log_to_ml(oof["q05"] - q_oof_cqr),
            hi_oof=_log_to_ml(oof["q95"] + q_oof_cqr),
            pred_te=pred_te,
            lo_te=_log_to_ml(te["q05"] - q_cqr),
            hi_te=_log_to_ml(te["q95"] + q_cqr),
            extra={"q": q_cqr},
        )

    for scale_method in ("extra_trees", "gbrt"):
        oof_scale, test_scale = _fit_scaled_abs_residual(
            x_train,
            y_train_log,
            mc_train_log,
            x_test,
            mc_test_log,
            folds,
            seed,
            scale_method,
        )
        scaled_score = np.abs(y_train_log - mc_train_log) / oof_scale
        q_oof_scaled = _oof_conformal_quantiles(scaled_score, folds, alpha)
        q_scaled = _conformal_quantile(scaled_score, alpha)
        _emit_method(
            **emit,
            method=f"{scale_method}_scaled_residual_conformal",
            pred_oof=mc_train,
            lo_oof=_log_to_ml(mc_train_log - q_oof_scaled * oof_scale),
            hi_oof=_log_to_ml(mc_train_log + q_oof_scaled * oof_scale),
            pred_te=mc_test,
            lo_te=_log_to_ml(mc_test_log - q_scaled * test_scale),
            hi_te=_log_to_ml(mc_test_log + q_scaled * test_scale),
            extra={"q": q_scaled},
        )

    st_oof, _st_train, st_test = _fit_student_head(x_train, y_train_log, x_test, folds, seed, student_cfg)
    st_q = _student_t_quantile(1.0 - alpha / 2.0, df=student_cfg.df)
    _emit_method(
        **emit,
        method="student_t_raw",
        pred_oof=_log_to_ml(st_oof["mu"]),
        lo_oof=_log_to_ml(st_oof["mu"] - st_q * st_oof["scale"]),
        hi_oof=_log_to_ml(st_oof["mu"] + st_q * st_oof["scale"]),
        pred_te=_log_to_ml(st_test["mu"]),
        lo_te=_log_to_ml(st_test["mu"] - st_q * st_test["scale"]),
        hi_te=_log_to_ml(st_test["mu"] + st_q * st_test["scale"]),
        extra={"df": student_cfg.df},
    )
    st_score = np.abs(y_train_log - st_oof["mu"]) / st_oof["scale"]
    st_q_oof = _oof_conformal_quantiles(st_score, folds, alpha)
    st_cq = _conformal_quantile(st_score, alpha)
    _emit_method(
        **emit,
        method="student_t_scaled_conformal",
        pred_oof=_log_to_ml(st_oof["mu"]),
        lo_oof=_log_to_ml(st_oof["mu"] - st_q_oof * st_oof["scale"]),
        hi_oof=_log_to_ml(st_oof["mu"] + st_q_oof * st_oof["scale"]),
        pred_te=_log_to_ml(st_test["mu"]),
        lo_te=_log_to_ml(st_test["mu"] - st_cq * st_test["scale"]),
        hi_te=_log_to_ml(st_test["mu"] + st_cq * st_test["scale"]),
        extra={"q": st_cq, "df": student_cfg.df},
    )
    return rows, preds, cols


def _assert_ranking_is_ispy2_only(ranking: pd.DataFrame) -> None:
    forbidden = ("acrin", "holdout")
    bad_cols = [c for c in ranking.columns if any(tok in str(c).lower() for tok in forbidden)]
    if bad_cols:
        raise AssertionError(f"I-SPY2 ranking columns leak ACRIN: {bad_cols}")
    if set(ranking["split"].astype(str)) != {"ispy2_oof"}:
        raise AssertionError(f"ranking split must be ispy2_oof only, got {sorted(set(ranking['split'].astype(str)))}")
    for key in RANK_KEYS:
        if any(tok in key.lower() for tok in forbidden):
            raise AssertionError(f"ranking sort key leaks ACRIN: {key}")


def _select_winner(oof_summary: pd.DataFrame) -> pd.Series:
    _assert_ranking_is_ispy2_only(oof_summary)
    work = oof_summary.copy()
    work = work[~work["method"].isin(RANKING_EXCLUDED_METHODS)].copy()
    dup = work["method"].isin(FEATURE_INDEPENDENT_METHODS) & (work["feature_mode"] != "paper_scalar")
    work = work[~dup].copy()
    if work.empty:
        raise RuntimeError("No I-SPY2 OOF methods available for ranking.")
    eligible = work[work["coverage"] >= COVERAGE_TARGET - COVERAGE_TOLERANCE]
    pool = eligible if len(eligible) else work
    ranked = pool.sort_values(list(RANK_KEYS), ascending=[True, False, True], kind="mergesort")
    return ranked.iloc[0]


def _row_metrics(row: pd.Series) -> dict[str, float]:
    return {
        "n": float(row["n"]),
        "mae_ml": float(row["mae_ml"]),
        "rmse_ml": float(row["rmse_ml"]),
        "bias_ml": float(row["bias_ml"]),
        "coverage": float(row["coverage"]),
        "width_ml": float(row["width_ml"]),
        "interval_score": float(row["interval_score"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-per-patient", type=Path, required=True)
    ap.add_argument("--test-per-patient", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/acrin_paper_model_uq/methods/resnet_crystal_edgegate_rd_k8"))
    ap.add_argument("--visit-pair", default="T0_to_T3", choices=("all", "T0_to_T1", "T0_to_T2", "T0_to_T3", "T1_to_T2", "T1_to_T3", "T2_to_T3"))
    ap.add_argument("--feature-modes", nargs="+", default=["paper_scalar", "paper_full"], choices=("paper_scalar", "paper_full"))
    ap.add_argument("--interval", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--student-epochs", type=int, default=500)
    ap.add_argument("--student-patience", type=int, default=50)
    args = ap.parse_args()

    alpha = 1.0 - float(args.interval)
    if not (0.0 < alpha < 1.0):
        raise ValueError("--interval must be between 0 and 1")

    train = _add_engineered_features(_load_table(args.train_per_patient, args.visit_pair))
    test = _add_engineered_features(_load_table(args.test_per_patient, args.visit_pair))
    student_cfg = StudentConfig(epochs=int(args.student_epochs), patience=int(args.student_patience))

    all_rows: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []
    feature_cols: dict[str, list[str]] = {}
    for mode in args.feature_modes:
        rows, preds, cols = _run_feature_mode(train, test, mode, alpha, int(args.seed), student_cfg)
        all_rows.extend(rows)
        all_preds.extend(preds)
        feature_cols[mode] = cols

    summary = pd.DataFrame(all_rows)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    oof_rank = summary[summary["split"] == "ispy2_oof"].copy()
    _assert_ranking_is_ispy2_only(oof_rank)
    oof_rank = oof_rank.sort_values(list(RANK_KEYS), ascending=[True, False, True], kind="mergesort")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "paper_model_uq_method_summary.csv"
    ranking_path = args.out_dir / "paper_model_uq_ispy2_oof_ranking.csv"
    frozen_path = args.out_dir / "paper_model_uq_frozen_acrin.csv"
    predictions_path = args.out_dir / "paper_model_uq_method_predictions.csv"
    manifest_path = args.out_dir / "paper_model_uq_method_manifest.json"
    summary_md_path = args.out_dir / "paper_model_uq_method_summary.md"
    summary.to_csv(summary_path, index=False)
    oof_rank.to_csv(ranking_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    winner_row = None
    frozen = pd.DataFrame()
    if args.visit_pair == "T0_to_T3":
        winner_row = _select_winner(oof_rank)
        frozen = summary[
            (summary["split"] == "acrin_holdout")
            & (summary["feature_mode"] == winner_row["feature_mode"])
            & (summary["method"] == winner_row["method"])
        ].copy()
        if len(frozen) != 1:
            raise RuntimeError("Expected exactly one frozen ACRIN row for the I-SPY2 winner.")
        frozen.to_csv(frozen_path, index=False)

    manifest: dict[str, Any] = {
        "train_per_patient": str(args.train_per_patient),
        "test_per_patient": str(args.test_per_patient),
        "visit_pair": args.visit_pair,
        "interval": float(args.interval),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seed": int(args.seed),
        "selection_split": "ispy2_oof",
        "selection_visit_pair": "T0_to_T3",
        "coverage_target": COVERAGE_TARGET,
        "coverage_tolerance": COVERAGE_TOLERANCE,
        "rank_keys": list(RANK_KEYS),
        "rank_keys_exclude_acrin": True,
        "feature_cols": feature_cols,
    }
    if winner_row is not None:
        acrin = frozen.iloc[0]
        manifest["winner"] = {
            "feature_mode": str(winner_row["feature_mode"]),
            "method": str(winner_row["method"]),
            "ispy2_oof": _row_metrics(winner_row),
            "acrin_holdout": _row_metrics(acrin),
        }
        manifest["note"] = (
            "Winner frozen on I-SPY2 OOF T0-to-T3 only; ACRIN metrics were scored once after freeze "
            "and were not used to rank methods."
        )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    lines = [
        "# Paper-Model UQ Method Experiments",
        "",
        f"Train/calibration source: I-SPY2 OOF paper-model MC rows, n={len(train)}.",
        f"Held-out score source: ACRIN-6698 paper-model MC rows, n={len(test)}.",
        f"Visit pair: `{args.visit_pair}`.",
        f"Target interval coverage: {args.interval:.2f}.",
        "Method ranking uses I-SPY2 OOF only; ACRIN is scored after freeze.",
        "",
        "| Split | Feature mode | Method | MAE | Coverage | Width | Interval score |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    display = summary.sort_values(["split"] + list(RANK_KEYS), ascending=[True, True, False, True], kind="mergesort")
    for _, row in display.iterrows():
        lines.append(
            f"| {row['split']} | {row['feature_mode']} | {row['method']} | "
            f"{row['mae_ml']:.3f} | {row['coverage']:.3f} | "
            f"{row['width_ml']:.3f} | {row['interval_score']:.3f} |"
        )
    if winner_row is not None:
        lines.extend(
            [
                "",
                f"Frozen winner (`{winner_row['feature_mode']}` / `{winner_row['method']}`): "
                f"I-SPY2 OOF interval score {float(winner_row['interval_score']):.3f}, "
                f"coverage {float(winner_row['coverage']):.3f}; "
                f"ACRIN interval score {float(frozen.iloc[0]['interval_score']):.3f}, "
                f"coverage {float(frozen.iloc[0]['coverage']):.3f}.",
            ]
        )
    summary_md_path.write_text("\n".join(lines) + "\n")

    print(oof_rank.head(20).to_string(index=False), flush=True)
    if winner_row is not None:
        print(
            f"FROZEN_WINNER {winner_row['feature_mode']} {winner_row['method']} "
            f"ispy2_interval_score={float(winner_row['interval_score']):.6f} "
            f"acrin_interval_score={float(frozen.iloc[0]['interval_score']):.6f}",
            flush=True,
        )
    print(summary_path, flush=True)
    print(ranking_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
