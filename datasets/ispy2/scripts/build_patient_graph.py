#!/usr/bin/env python
"""Assemble a per-patient spatio-temporal supervoxel graph.

Consumes the per-visit outputs of `preprocess_patient.py`:

    <derived_root>/<patient_id>/T{0,1,2,3}/supervoxels.parquet
    <derived_root>/<patient_id>/T{0,1,2,3}/meta.json

and produces a single ``.pt`` payload consumable by LSGCNet:

    {
        "x":             FloatTensor (N_total, C_feat),
        "pos":           FloatTensor (N_total, 3),       # (z, y, x) in mm
        "t":             LongTensor  (N_total,),         # visit index in {0,1,2,3}
        "edge_index":    LongTensor  (2, E_total),
        "visit_offsets": LongTensor  (V+1,),
        "visit_ids":     list[str],                      # e.g. ['T0','T1','T2','T3']
        "feature_names": list[str],
        "patient_id":    str,
        "meta":          dict of per-visit metadata (phase info, spacing, ...)
    }

Uses `build_spatiotemporal_graph` from `lsgc.graph_builder` to form kNN
edges intra-visit (spatial) and inter-visit (temporal), so the loaded
tensors plug straight into `LSGCConv`.

Run from this repository root so `import lsgc` resolves:

    python datasets/ispy2/scripts/build_patient_graph.py \\
        --derived-root datasets/ispy2/derived \\
        --patient-id   ISPY2-212307 \\
        --out-root     datasets/ispy2/graphs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lsgc.graph_builder import build_spatiotemporal_graph  # noqa: E402


FEATURE_NAMES_V1 = ["log_voxel_count", "volume_ml", "mean_pe", "mean_ser", "enh_late"]
PHASE_FEATURE_COUNT_V2 = 8
FEATURE_NAMES_V2 = [
    "log_voxel_count",
    "volume_ml",
    "mean_pe",
    "pe_std",
    "pe_skew",
    "pe_kurtosis",
    "mean_ser",
    "ser_std",
    "ser_skew",
    "ser_kurtosis",
    "enh_late",
    "surface_area_mm2",
    "sphericity",
    "elongation",
    "major_axis_length_mm",
    "minor_axis_length_mm",
    *[f"phase{p}_mean_enh" for p in range(PHASE_FEATURE_COUNT_V2)],
]
FEATURE_NAMES_V3 = [
    *FEATURE_NAMES_V2,
    "mean_adc",
    "adc_std",
    "adc_skew",
    "adc_kurtosis",
    "adc_missing",
]
TIMEPOINTS = ("T0", "T1", "T2", "T3")


def _column_or_zeros(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return df[name].to_numpy(dtype=np.float32)


def _visit_feature_tensor(
    df: pd.DataFrame,
    meta: dict,
    *,
    feature_set: str = "v1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x, pos) tensors for a single visit.

    x: (N, C) float32
    pos: (N, 3) float32 -- (z, y, x) in mm
    """
    late_phase = int(meta["late_phase"])
    late_col = f"phase{late_phase}_mean_enh"
    if late_col not in df.columns:
        raise KeyError(
            f"expected column {late_col} in supervoxels.parquet (P={meta['dce_phases']})"
        )

    if feature_set == "v1":
        feature_names = FEATURE_NAMES_V1
    elif feature_set == "v2":
        feature_names = FEATURE_NAMES_V2
    elif feature_set == "v3":
        feature_names = FEATURE_NAMES_V3
    else:
        raise ValueError(f"unknown feature_set={feature_set!r}")

    arrays: dict[str, np.ndarray] = {
        "log_voxel_count": np.log1p(df["voxel_count"].to_numpy(dtype=np.float32)),
        "volume_ml": df["volume_ml"].to_numpy(dtype=np.float32),
        "mean_pe": df["mean_pe"].to_numpy(dtype=np.float32),
        "mean_ser": df["mean_ser"].to_numpy(dtype=np.float32),
        "enh_late": df[late_col].to_numpy(dtype=np.float32),
    }
    for name in FEATURE_NAMES_V3:
        if name in arrays:
            continue
        arrays[name] = _column_or_zeros(df, name)

    x = np.stack([arrays[name] for name in feature_names], axis=1)
    pos = df[["centroid_z_mm", "centroid_y_mm", "centroid_x_mm"]].to_numpy(
        dtype=np.float32
    )
    return torch.from_numpy(x.copy()), torch.from_numpy(pos.copy())


def _load_visit(
    patient_root: Path,
    tp: str,
    *,
    feature_set: str,
) -> tuple[torch.Tensor, torch.Tensor, dict] | None:
    visit_dir = patient_root / tp
    pq = visit_dir / "supervoxels.parquet"
    mj = visit_dir / "meta.json"
    if not pq.exists() or not mj.exists():
        return None
    df = pd.read_parquet(pq)
    if df.empty:
        return None
    meta = json.loads(mj.read_text())
    x, pos = _visit_feature_tensor(df, meta, feature_set=feature_set)
    return x, pos, meta


def build_graph_for_patient(
    derived_root: Path,
    patient_id: str,
    *,
    k_spatial: int = 8,
    k_temporal: int = 4,
    feature_weight: float = 1.0,
    position_weight: float = 1.0,
    feature_set: str = "v1",
    log: logging.Logger | None = None,
) -> dict:
    log = log or logging.getLogger("build_graph")
    patient_root = derived_root / patient_id
    if not patient_root.is_dir():
        raise FileNotFoundError(patient_root)

    visit_feats: list[torch.Tensor] = []
    visit_pos: list[torch.Tensor] = []
    visit_ids: list[str] = []
    per_visit_meta: dict[str, dict] = {}
    if feature_set == "v1":
        feature_names = FEATURE_NAMES_V1
    elif feature_set == "v2":
        feature_names = FEATURE_NAMES_V2
    else:
        feature_names = FEATURE_NAMES_V3

    for tp in TIMEPOINTS:
        loaded = _load_visit(patient_root, tp, feature_set=feature_set)
        if loaded is None:
            log.warning("  %s: missing / empty, skipping", tp)
            continue
        x, pos, meta = loaded
        visit_feats.append(x)
        visit_pos.append(pos)
        visit_ids.append(tp)
        per_visit_meta[tp] = meta
        log.info("  %s: %d nodes", tp, x.shape[0])

    if len(visit_feats) < 2:
        raise ValueError(
            f"{patient_id}: need >=2 visits to build a temporal graph, got {len(visit_feats)}"
        )

    g = build_spatiotemporal_graph(
        visit_feats, visit_pos,
        k_spatial=k_spatial,
        k_temporal=k_temporal,
        feature_weight=feature_weight,
        position_weight=position_weight,
    )

    t = torch.zeros(g.x.shape[0], dtype=torch.long)
    for v_idx in range(len(visit_ids)):
        start = int(g.visit_offsets[v_idx].item())
        end = int(g.visit_offsets[v_idx + 1].item())
        t[start:end] = v_idx

    payload = {
        "x": g.x.float(),
        "pos": g.pos.float(),
        "t": t,
        "edge_index": g.edge_index.long(),
        "visit_offsets": g.visit_offsets.long(),
        "visit_ids": visit_ids,
        "feature_names": feature_names,
        "feature_set": feature_set,
        "patient_id": patient_id,
        "meta": per_visit_meta,
        "k_spatial": int(k_spatial),
        "k_temporal": int(k_temporal),
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived-root", type=Path, required=True)
    ap.add_argument("--patient-id", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--k-spatial", type=int, default=8)
    ap.add_argument("--k-temporal", type=int, default=4)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--position-weight", type=float, default=1.0)
    ap.add_argument("--feature-set", choices=("v1", "v2", "v3"), default="v1")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("build_patient_graph")

    log.info("patient %s -> %s", args.patient_id, args.out_root)
    payload = build_graph_for_patient(
        args.derived_root, args.patient_id,
        k_spatial=args.k_spatial,
        k_temporal=args.k_temporal,
        feature_weight=args.feature_weight,
        position_weight=args.position_weight,
        feature_set=args.feature_set,
        log=log,
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    out_path = args.out_root / f"{args.patient_id}.pt"
    torch.save(payload, out_path)

    log.info(
        "saved %s: x=%s, edges=%d, visits=%s",
        out_path.name,
        tuple(payload["x"].shape),
        int(payload["edge_index"].shape[1]),
        payload["visit_ids"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
