#!/usr/bin/env python3
"""Summarize paper-model uncertainty sampler experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLERS = (
    "bucket_baseline",
    "predicted_ftv_stratified",
    "baseline_ftv_stratified",
    "alive_mass_stratified",
    "hybrid_backoff",
)


def _safe_float(x: object) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    lo = np.maximum(0.0, np.asarray(lo, dtype=float))
    hi = np.maximum(lo, np.asarray(hi, dtype=float))
    y = np.asarray(y, dtype=float)
    lower_miss = np.maximum(0.0, lo - y)
    upper_miss = np.maximum(0.0, y - hi)
    return (hi - lo) + (2.0 / float(alpha)) * (lower_miss + upper_miss)


def _summarize_group(df: pd.DataFrame, label: str, alpha: float) -> dict[str, object]:
    obs = pd.to_numeric(df["obs_ftv_ml"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(df["ftv_mc_mean_ml"], errors="coerce").to_numpy(dtype=float)
    det = pd.to_numeric(df["pred_ftv_det_ml"], errors="coerce").to_numpy(dtype=float)
    raw_lo = pd.to_numeric(df["ftv_raw_p05_ml"], errors="coerce").to_numpy(dtype=float)
    raw_hi = pd.to_numeric(df["ftv_raw_p95_ml"], errors="coerce").to_numpy(dtype=float)
    conf_lo = pd.to_numeric(df["ftv_conformal_p05_ml"], errors="coerce").to_numpy(dtype=float)
    conf_hi = pd.to_numeric(df["ftv_conformal_p95_ml"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    raw_ok = ok & np.isfinite(raw_lo) & np.isfinite(raw_hi)
    conf_ok = ok & np.isfinite(conf_lo) & np.isfinite(conf_hi)
    out: dict[str, object] = {
        "bucket": label,
        "n_rows": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else int(len(df)),
        "det_mae_ml": _safe_float(np.mean(np.abs(det[ok] - obs[ok]))) if ok.any() else float("nan"),
        "mc_mean_mae_ml": _safe_float(np.mean(np.abs(pred[ok] - obs[ok]))) if ok.any() else float("nan"),
        "mc_mean_bias_ml": _safe_float(np.mean(pred[ok] - obs[ok])) if ok.any() else float("nan"),
        "crps_ftv_ml": _safe_float(pd.to_numeric(df.get("crps_ftv"), errors="coerce").mean()),
        "raw_coverage90": _safe_float(np.mean((obs[raw_ok] >= raw_lo[raw_ok]) & (obs[raw_ok] <= raw_hi[raw_ok]))) if raw_ok.any() else float("nan"),
        "raw_width90_ml": _safe_float(np.mean(raw_hi[raw_ok] - raw_lo[raw_ok])) if raw_ok.any() else float("nan"),
        "raw_interval_score": _safe_float(np.mean(_interval_score(obs[raw_ok], raw_lo[raw_ok], raw_hi[raw_ok], alpha))) if raw_ok.any() else float("nan"),
        "conformal_coverage90": _safe_float(np.mean((obs[conf_ok] >= conf_lo[conf_ok]) & (obs[conf_ok] <= conf_hi[conf_ok]))) if conf_ok.any() else float("nan"),
        "conformal_width90_ml": _safe_float(np.mean(conf_hi[conf_ok] - conf_lo[conf_ok])) if conf_ok.any() else float("nan"),
        "conformal_interval_score": _safe_float(np.mean(_interval_score(obs[conf_ok], conf_lo[conf_ok], conf_hi[conf_ok], alpha))) if conf_ok.any() else float("nan"),
        "mean_calibration_patients": _safe_float(pd.to_numeric(df.get("n_calibration_patients"), errors="coerce").mean()),
        "mean_bucket_calibration_patients": _safe_float(pd.to_numeric(df.get("n_bucket_calibration_patients"), errors="coerce").mean()),
    }
    if "sampler_selection_rule" in df.columns:
        out["n_selection_rules"] = int(df["sampler_selection_rule"].astype(str).nunique())
        out["top_selection_rule"] = str(df["sampler_selection_rule"].astype(str).mode().iloc[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("reports/acrin_paper_model_uq/mc"))
    ap.add_argument("--model-tag", default="resnet_crystal_edgegate_rd_k8")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/acrin_paper_model_uq/summary"))
    ap.add_argument("--interval", type=float, default=0.90)
    args = ap.parse_args()

    alpha = 1.0 - float(args.interval)
    rows: list[dict[str, object]] = []
    combined: list[pd.DataFrame] = []
    for sampler in SAMPLERS:
        path = args.root / args.model_tag / sampler / "conditional_mc_per_patient.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["sampler_mode"] = sampler
        combined.append(df)
        rows.append({"sampler_mode": sampler, **_summarize_group(df, "all_conditioning", alpha)})
        t3 = df[(df["start_visit"] == "T0") & (df["predicted_visit"] == "T3")]
        if len(t3):
            rows.append({"sampler_mode": sampler, **_summarize_group(t3, "T0_to_T3", alpha)})
        for start_visit, predicted_visit in [("T0", "T1"), ("T0", "T2"), ("T1", "T3"), ("T2", "T3")]:
            g = df[(df["start_visit"] == start_visit) & (df["predicted_visit"] == predicted_visit)]
            if len(g):
                rows.append({
                    "sampler_mode": sampler,
                    **_summarize_group(g, f"{start_visit}_to_{predicted_visit}", alpha),
                })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "paper_model_uq_sampler_summary.csv", index=False)
    if combined:
        pd.concat(combined, ignore_index=True).to_parquet(
            args.out_dir / "paper_model_uq_sampler_per_patient.parquet",
            index=False,
        )
    manifest = {
        "root": str(args.root),
        "model_tag": args.model_tag,
        "samplers_found": sorted(summary["sampler_mode"].unique().tolist()) if len(summary) else [],
        "summary_rows": int(len(summary)),
        "primary_bucket": "T0_to_T3",
    }
    (args.out_dir / "paper_model_uq_sampler_manifest.json").write_text(json.dumps(manifest, indent=2))
    if len(summary):
        primary = summary[summary["bucket"] == "T0_to_T3"].sort_values(
            ["conformal_interval_score", "conformal_width90_ml"],
            ascending=[True, True],
        )
        if len(primary):
            primary.to_csv(args.out_dir / "paper_model_uq_sampler_t0t3_ranked.csv", index=False)
            print(primary.to_string(index=False), flush=True)
    print(args.out_dir / "paper_model_uq_sampler_summary.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
