#!/usr/bin/env python3
"""Run simulation evaluation on registered consistent graphs.

Evaluates deterministic rollouts from the consistent forecaster using:
  1) teacher-forced one-step transitions
  2) autoregressive rollout with increasing observed context

Writes per-patient/per-visit metrics and aggregated summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lsgc import inference as lib  # noqa: E402
from lsgc.metrics import compute_all_cloud_metrics  # noqa: E402


def _list_folds(runs_dir: Path) -> list[Path]:
    return sorted([p for p in runs_dir.glob("fold*") if p.is_dir()])


def _load_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    model, mean, std = lib.load_model(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return model, mean, std, dict(raw.get("config", {}))


def _patient_ids_for_fold(
    fold_dir: Path,
    folds_df: pd.DataFrame,
    patient_list: Path | None = None,
) -> list[str]:
    if patient_list is not None and patient_list.is_file():
        ids = [ln.strip() for ln in patient_list.read_text().splitlines() if ln.strip()]
        return ids
    fold = int(fold_dir.name.replace("fold", ""))
    ids = folds_df.loc[folds_df["fold"] == fold, "patient_id"].astype(str).tolist()
    return ids


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _apply_delta_feat_np(x: np.ndarray, dfeat: np.ndarray) -> np.ndarray:
    if dfeat.shape[1] == x.shape[1]:
        return x + dfeat
    if dfeat.shape[1] > x.shape[1]:
        raise ValueError(f"delta feature width {dfeat.shape[1]} exceeds state width {x.shape[1]}")
    out = x.copy()
    out[:, : dfeat.shape[1]] = out[:, : dfeat.shape[1]] + dfeat
    return out


def _teacher_forced_rows(
    pid: str,
    g: dict,
    infer: dict,
    cohort_row: dict[str, Any] | None,
    volume_idx: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    off = g["visit_offsets"].tolist()
    ftv_t0 = _safe_float(g["x"][int(off[0]): int(off[1]), volume_idx].sum().item())

    for v, vid in enumerate(lib.VISITS[:-1]):
        nxt = lib.VISITS[v + 1]
        pos_src = infer["pos_actual"][vid]
        dpos = infer[f"{vid}_dpos"]
        pred_pos = pos_src + dpos
        obs_pos = infer["pos_actual"][nxt]

        x_src = g["x"][int(off[v]): int(off[v + 1])].detach().cpu().numpy()
        x_pred = _apply_delta_feat_np(x_src, infer[f"{vid}_dfeat"])
        x_obs = g["x"][int(off[v + 1]): int(off[v + 2])].detach().cpu().numpy()

        pred_ftv = _safe_float(np.clip(x_pred[:, volume_idx], 0.0, None).sum())
        obs_ftv = _safe_float(x_obs[:, volume_idx].sum())

        metrics = compute_all_cloud_metrics(
            pred_pos=pred_pos,
            obs_pos=obs_pos,
            pred_alive=None,
            obs_alive=g["alive"][int(off[v + 1]): int(off[v + 2])].detach().cpu().numpy(),
            pred_ftv=pred_ftv,
            obs_ftv=obs_ftv,
            ftv_t0=ftv_t0,
            compute_topology=True,
        )
        row = {
            "patient_id": pid,
            "conditioning": "teacher_forced",
            "start_visit": vid,
            "predicted_visit": nxt,
            "rollout_depth": 1,
            "ftv_t0_ml": ftv_t0,
            "pred_ftv_ml": pred_ftv,
            "obs_ftv_ml": obs_ftv,
            "obs_ftv_ratio_tk_t0": _safe_float(obs_ftv / ftv_t0) if ftv_t0 > 0 else float("nan"),
            **metrics,
        }
        if cohort_row:
            row.update(cohort_row)
        rows.append(row)
    return rows


def _rollout_rows(
    pid: str,
    g: dict,
    rollout: list[dict[str, Any]],
    start_visit: int,
    cohort_row: dict[str, Any] | None,
    volume_idx: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    off = g["visit_offsets"].tolist()
    ftv_t0 = _safe_float(g["x"][int(off[0]): int(off[1]), volume_idx].sum().item())
    for idx, step in enumerate(rollout):
        pred_ftv_arr = np.clip(step["x_pred"][:, volume_idx], 0.0, None) * np.asarray(step["alive_prob"])
        pred_ftv_node = _safe_float(np.sum(pred_ftv_arr))
        pred_ftv = _safe_float(step.get("graph_ftv_pred_ml", pred_ftv_node))
        if not np.isfinite(pred_ftv):
            pred_ftv = pred_ftv_node
        obs_ftv = _safe_float(step["x_obs"][:, volume_idx].sum())
        metrics = compute_all_cloud_metrics(
            pred_pos=step["pos_pred"],
            obs_pos=step["pos_obs"],
            pred_alive=np.asarray(step["alive_prob"]),
            obs_alive=np.asarray(step["alive_obs"]),
            pred_ftv=pred_ftv,
            obs_ftv=obs_ftv,
            ftv_t0=ftv_t0,
            compute_topology=True,
        )
        row = {
            "patient_id": pid,
            "conditioning": f"rollout_from_{lib.VISITS[start_visit]}",
            "start_visit": lib.VISITS[start_visit],
            "predicted_visit": step["visit_name"],
            "rollout_depth": idx + 1,
            "ftv_t0_ml": ftv_t0,
            "pred_ftv_ml": pred_ftv,
            "pred_ftv_node_sum_ml": pred_ftv_node,
            "pred_ftv_source": "graph_readout" if step.get("graph_ftv_pred_ml") is not None else "node_sum",
            "obs_ftv_ml": obs_ftv,
            "obs_ftv_ratio_tk_t0": _safe_float(obs_ftv / ftv_t0) if ftv_t0 > 0 else float("nan"),
            **metrics,
        }
        if cohort_row:
            row.update(cohort_row)
        rows.append(row)
    return rows


def _aggregate(df: pd.DataFrame) -> dict[str, Any]:
    metric_cols = [
        c for c in df.columns
        if c.endswith("_mm")
        or c.endswith("_err")
        or c in ("dice", "iou", "surface_dice", "ftv_abs_err_ml", "ftv_rel_err")
    ]
    summary_rows = []
    for (cond, visit), grp in df.groupby(["conditioning", "predicted_visit"], dropna=False):
        row: dict[str, Any] = {
            "conditioning": cond,
            "predicted_visit": visit,
            "n": int(len(grp)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna().values
            row[f"{col}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{col}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
        summary_rows.append(row)
    return {
        "n_rows": int(len(df)),
        "n_patients": int(df["patient_id"].nunique() if len(df) else 0),
        "by_conditioning_visit": summary_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/consistent_forecaster_5fold"))
    ap.add_argument("--checkpoint-name", default="best.pt")
    ap.add_argument("--graphs-root", type=Path, default=Path("datasets/ispy2/graphs_consistent"))
    ap.add_argument("--folds", type=Path, default=Path("datasets/ispy2/folds.parquet"))
    ap.add_argument("--cohort", type=Path, default=Path("datasets/ispy2/cohort.parquet"))
    ap.add_argument("--patient-list", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/simulation_eval"))
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on patient count per fold.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds_df = pd.read_parquet(args.folds)
    cohort_df = pd.read_parquet(args.cohort).set_index("patient_id")
    fold_dirs = _list_folds(args.runs_dir)
    if not fold_dirs:
        print(f"No fold directories found under {args.runs_dir}")
        return 1

    all_rows: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / args.checkpoint_name
        if not ckpt_path.is_file():
            print(f"skip {fold_dir}: missing {args.checkpoint_name}")
            continue
        graphs_root_for_fold = args.graphs_root / fold_dir.name
        if not graphs_root_for_fold.is_dir():
            graphs_root_for_fold = args.graphs_root
        model, mean, std, cfg = _load_checkpoint(ckpt_path)
        k_spatial = int(cfg.get("k_spatial", 8))
        edge_mode = str(cfg.get("edge_mode", "full"))
        edge_attr_mode = str(cfg.get("edge_attr_mode", "none"))
        patient_ids = _patient_ids_for_fold(fold_dir, folds_df, args.patient_list)
        if args.limit > 0:
            patient_ids = patient_ids[: args.limit]

        fold_rows: list[dict[str, Any]] = []
        for pid in patient_ids:
            gp = graphs_root_for_fold / f"{pid}.pt"
            if not gp.is_file():
                continue
            try:
                g = torch.load(gp, map_location="cpu", weights_only=False)
            except Exception as exc:
                print(f"{pid}: load error {exc}")
                continue
            cohort_row = None
            if pid in cohort_df.index:
                cr = cohort_df.loc[pid]
                cohort_row = {
                    "collection": str(cr["collection"]) if pd.notna(cr.get("collection", None)) else None,
                    "pCR": int(cr["pCR"]) if pd.notna(cr.get("pCR", None)) else None,
                    "subtype": str(cr["subtype"]) if pd.notna(cr.get("subtype", None)) else None,
                }

            try:
                infer = lib.run_inference(
                    model,
                    mean,
                    std,
                    g,
                    k_spatial=k_spatial,
                    edge_mode=edge_mode,
                    edge_attr_mode=edge_attr_mode,
                )
                fold_rows.extend(_teacher_forced_rows(pid, g, infer, cohort_row))
                for start_visit in (0, 1, 2):
                    rollout = lib.rollout_from_visit(
                        model,
                        mean,
                        std,
                        g,
                        start_visit=start_visit,
                        k_spatial=k_spatial,
                        edge_mode=edge_mode,
                        edge_attr_mode=edge_attr_mode,
                    )
                    fold_rows.extend(_rollout_rows(pid, g, rollout, start_visit, cohort_row))
            except Exception as exc:
                print(f"{pid}: eval error {exc}")
                continue

        if not fold_rows:
            print(f"skip {fold_dir}: no evaluable rows")
            continue
        fold_df = pd.DataFrame(fold_rows)
        fold_out_dir = args.out_dir / fold_dir.name
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        fold_df.to_parquet(fold_out_dir / "simulation_per_patient.parquet", index=False)
        fold_summary = _aggregate(fold_df)
        (fold_out_dir / "simulation_summary.json").write_text(json.dumps(fold_summary, indent=2))
        print(f"[{fold_dir.name}] wrote {len(fold_df)} rows")
        all_rows.extend(fold_rows)

    if all_rows:
        all_df = pd.DataFrame(all_rows)
        all_df.to_parquet(args.out_dir / "simulation_per_patient.parquet", index=False)
        overall = _aggregate(all_df)
        (args.out_dir / "simulation_summary.json").write_text(json.dumps(overall, indent=2))
        print(f"[overall] wrote {len(all_df)} rows to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
