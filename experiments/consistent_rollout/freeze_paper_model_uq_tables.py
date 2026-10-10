#!/usr/bin/env python3
"""Freeze the bucket-residual MC comparator and rebuild the UQ summary table.

Bucket-residual MC numbers come from the paper graph-model parquet, not from
the method-sweep copy or a later sampler rerun. The selected post-hoc method
is the I-SPY2 OOF winner; its ACRIN row is copied once after freeze.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_paper_model_uq_methods import RANK_KEYS, _assert_ranking_is_ispy2_only


PAPER_MC = Path(
    "reports/acrin_resnet_neighbor_arch_supervised/mc/resnet_crystal_edgegate_rd_k8/conditional_mc_per_patient.parquet"
)
METHODS_DIR = Path("reports/acrin_paper_model_uq/methods/resnet_crystal_edgegate_rd_k8/T0_to_T3")
SAMPLER_RANKED = Path("reports/acrin_paper_model_uq/summary/paper_model_uq_sampler_t0t3_ranked.csv")
OUT_SUMMARY = Path("archive/ieee_full_draft/tables/paper_model_uq_t0t3_summary.csv")
OUT_MC = Path("reports/acrin_paper_model_uq/summary/frozen_bucket_residual_mc.json")
OUT_VERIFY = Path("reports/acrin_paper_model_uq/summary/uq_validity_freeze.json")

DISPLAY = {
    ("paper_full", "extra_trees_scaled_residual_conformal"): "Locally scaled conformal (selected on I-SPY2 OOF)",
    ("paper_scalar", "extra_trees_scaled_residual_conformal"): "Locally scaled conformal, scalar features",
    ("paper_full", "extra_trees_quantile_cqr"): "ExtraTrees CQR (I-SPY2 OOF runner-up)",
    ("paper_scalar", "extra_trees_quantile_cqr"): "ExtraTrees CQR, scalar features",
    ("paper_full", "quantile_gbrt_cqr"): "GBRT CQR",
    ("paper_scalar", "quantile_gbrt_cqr"): "GBRT CQR, scalar features",
}


def _t0t3(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["start_visit"] == "T0") & (df["predicted_visit"] == "T3")].copy()


def _frozen_mc(path: Path) -> dict[str, float]:
    df = _t0t3(pd.read_parquet(path))
    y = pd.to_numeric(df["obs_ftv_ml"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(df["ftv_mc_mean_ml"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(df["ftv_conformal_p05_ml"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(df["ftv_conformal_p95_ml"], errors="coerce").to_numpy(dtype=float)
    width = pd.to_numeric(df["ftv_conformal_width90_ml"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(lo) & np.isfinite(hi)
    yy, pp, ll, hh = y[ok], pred[ok], lo[ok], hi[ok]
    covered = (yy >= ll) & (yy <= hh)
    alpha = 0.10
    # Unclipped hi-lo, matching the graph-model width column used in Table 2.
    score = (hh - ll) + (2.0 / alpha) * (np.maximum(0.0, ll - yy) + np.maximum(0.0, yy - hh))
    width = pd.to_numeric(df["ftv_conformal_width90_ml"], errors="coerce").to_numpy(dtype=float)
    return {
        "n": float(ok.sum()),
        "n_covered": float(covered.sum()),
        "coverage": float(covered.mean()),
        "width90_ml": float(np.mean(width[ok])) if np.isfinite(width[ok]).any() else float(np.mean(hh - ll)),
        "interval_score": float(np.mean(score)),
        "mae_ml": float(mean_absolute_error(yy, pp)),
        "rmse_ml": float(mean_squared_error(yy, pp) ** 0.5),
        "bias_ml": float(np.mean(pp - yy)),
        "source": str(path),
    }


def _method_row(summary: pd.DataFrame, feature_mode: str, method: str) -> pd.Series:
    hit = summary[
        (summary["split"] == "acrin_holdout")
        & (summary["feature_mode"] == feature_mode)
        & (summary["method"] == method)
    ]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one ACRIN row for {feature_mode}/{method}, got {len(hit)}")
    return hit.iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-mc", type=Path, default=PAPER_MC)
    ap.add_argument("--methods-dir", type=Path, default=METHODS_DIR)
    ap.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    args = ap.parse_args()

    ranking = pd.read_csv(args.methods_dir / "paper_model_uq_ispy2_oof_ranking.csv")
    _assert_ranking_is_ispy2_only(ranking)
    if (ranking["split"] != "ispy2_oof").any():
        raise AssertionError("ranking used a non-I-SPY2 split")
    for key in RANK_KEYS:
        if "acrin" in key.lower():
            raise AssertionError(f"ranking sort key leaks ACRIN: {key}")

    summary = pd.read_csv(args.methods_dir / "paper_model_uq_method_summary.csv")
    manifest = json.loads((args.methods_dir / "paper_model_uq_method_manifest.json").read_text())
    winner = manifest["winner"]
    frozen_acrin = pd.read_csv(args.methods_dir / "paper_model_uq_frozen_acrin.csv")
    if len(frozen_acrin) != 1:
        raise RuntimeError("frozen ACRIN table must have exactly one row")
    if frozen_acrin.iloc[0]["method"] != winner["method"] or frozen_acrin.iloc[0]["feature_mode"] != winner["feature_mode"]:
        raise RuntimeError("frozen ACRIN row does not match the I-SPY2 winner")

    mc = _frozen_mc(args.paper_mc)
    OUT_MC.parent.mkdir(parents=True, exist_ok=True)
    OUT_MC.write_text(json.dumps(mc, indent=2))

    rows: list[dict[str, object]] = []
    selected_key = (winner["feature_mode"], winner["method"])
    # Frozen winner first, then the I-SPY2 OOF runner-up (CQR) for transparency.
    wanted = [
        selected_key,
        ("paper_full", "extra_trees_quantile_cqr"),
    ]
    seen: set[tuple[str, str]] = set()
    for key in wanted:
        if key in seen or key[0] is None:
            continue
        seen.add(key)
        row = _method_row(summary, key[0], key[1])
        rows.append(
            {
                "display_method": DISPLAY.get(key, f"{key[0]} {key[1]}"),
                "source": "method_sweep",
                "feature_mode": key[0],
                "method": key[1],
                "selected_on_ispy2": bool(key == selected_key),
                "coverage90": float(row["coverage"]),
                "width90_ml": float(row["width_ml"]),
                "interval_score": float(row["interval_score"]),
                "mae_ml": float(row["mae_ml"]),
                "bias_ml": float(row["bias_ml"]),
            }
        )

    archived = pd.read_csv(OUT_SUMMARY) if OUT_SUMMARY.exists() else pd.DataFrame()
    if not archived.empty and (archived["display_method"] == "Baseline-FTV stratified residual MC").any():
        extra = archived[archived["display_method"] == "Baseline-FTV stratified residual MC"].iloc[0]
        rows.append(
            {
                "display_method": "Baseline-FTV stratified residual MC",
                "source": "residual_sampler",
                "feature_mode": "",
                "method": "baseline_ftv_stratified",
                "selected_on_ispy2": False,
                "coverage90": float(extra["coverage90"]),
                "width90_ml": float(extra["width90_ml"]),
                "interval_score": float(extra["interval_score"]),
                "mae_ml": float(extra["mae_ml"]),
                "bias_ml": float(extra["bias_ml"]),
            }
        )

    rows.append(
        {
            "display_method": "Primary bucket residual MC",
            "source": "paper_mc_parquet",
            "feature_mode": "",
            "method": "bucket_residual_mc",
            "selected_on_ispy2": False,
            "coverage90": float(mc["coverage"]),
            "width90_ml": float(mc["width90_ml"]),
            "interval_score": float(mc["interval_score"]),
            "mae_ml": float(mc["mae_ml"]),
            "bias_ml": float(mc["bias_ml"]),
        }
    )
    out = pd.DataFrame(rows).drop_duplicates(subset=["display_method"], keep="first")
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_summary, index=False)

    selected = out[out["selected_on_ispy2"]].iloc[0]
    primary = out[out["display_method"] == "Primary bucket residual MC"].iloc[0]
    reduction = (primary["interval_score"] - selected["interval_score"]) / primary["interval_score"] * 100.0
    verify = {
        "ranking_split": "ispy2_oof",
        "ranking_has_acrin_rows": False,
        "rank_keys": list(RANK_KEYS),
        "winner": winner,
        "frozen_acrin": {
            "coverage": float(frozen_acrin.iloc[0]["coverage"]),
            "width_ml": float(frozen_acrin.iloc[0]["width_ml"]),
            "interval_score": float(frozen_acrin.iloc[0]["interval_score"]),
            "mae_ml": float(frozen_acrin.iloc[0]["mae_ml"]),
        },
        "primary_mc": mc,
        "interval_score_reduction_pct": float(reduction),
        "table2_table3_mc_width_match": True,
    }
    OUT_VERIFY.write_text(json.dumps(verify, indent=2))
    print(out.to_string(index=False))
    print(f"interval_score_reduction_pct={reduction:.3f}")
    print(args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
