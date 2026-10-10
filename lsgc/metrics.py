"""Shared 3D simulation metrics for registered graph evaluation.

This module centralizes geometric, morphology, topology, overlap, and
probabilistic metrics used by simulation evaluation scripts/notebooks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, cKDTree, distance


def _as_points(arr: np.ndarray | list[list[float]] | list[tuple[float, float, float]]) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    return pts


def _valid_pair(A: np.ndarray, B: np.ndarray) -> bool:
    return A.shape[0] > 0 and B.shape[0] > 0


def chamfer_position(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance in mm."""
    A = _as_points(A)
    B = _as_points(B)
    if not _valid_pair(A, B):
        return float("nan")
    d_ab = cKDTree(B).query(A, k=1)[0]
    d_ba = cKDTree(A).query(B, k=1)[0]
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


def hausdorff_95(A: np.ndarray, B: np.ndarray) -> float:
    """95th-percentile Hausdorff distance in mm."""
    A = _as_points(A)
    B = _as_points(B)
    if not _valid_pair(A, B):
        return float("nan")
    d_ab = cKDTree(B).query(A, k=1)[0]
    d_ba = cKDTree(A).query(B, k=1)[0]
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def sliced_wasserstein(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 64,
    resample: int = 512,
    rng: np.random.Generator | None = None,
) -> float:
    """Approximate sliced Wasserstein-1 between 3D point clouds."""
    X = _as_points(X)
    Y = _as_points(Y)
    if not _valid_pair(X, Y):
        return float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    if X.shape[0] > resample:
        X = X[rng.choice(X.shape[0], size=resample, replace=False)]
    if Y.shape[0] > resample:
        Y = Y[rng.choice(Y.shape[0], size=resample, replace=False)]
    dirs = rng.normal(size=(n_projections, 3))
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    vals: list[float] = []
    for d in dirs:
        sx = np.sort(X @ d)
        sy = np.sort(Y @ d)
        q = min(sx.shape[0], sy.shape[0])
        if q == 0:
            continue
        tx = np.quantile(sx, np.linspace(0.0, 1.0, q))
        ty = np.quantile(sy, np.linspace(0.0, 1.0, q))
        vals.append(float(np.mean(np.abs(tx - ty))))
    return float(np.mean(vals)) if vals else float("nan")


def displacement_mae(pred_pos: np.ndarray, obs_pos: np.ndarray) -> float:
    """Mean point-wise displacement error (requires aligned node order)."""
    P = _as_points(pred_pos)
    O = _as_points(obs_pos)
    if P.shape != O.shape or P.shape[0] == 0:
        return float("nan")
    return float(np.linalg.norm(P - O, axis=1).mean())


def _voxel_set(pos: np.ndarray, voxel_size: float = 2.0) -> set[tuple[int, int, int]]:
    P = _as_points(pos)
    if P.shape[0] == 0:
        return set()
    vox = np.floor(P / float(voxel_size)).astype(np.int32)
    return set(map(tuple, vox.tolist()))


def voxelized_dice_iou(A: np.ndarray, B: np.ndarray, voxel_size: float = 2.0) -> tuple[float, float]:
    a = _voxel_set(A, voxel_size=voxel_size)
    b = _voxel_set(B, voxel_size=voxel_size)
    if not a and not b:
        return 1.0, 1.0
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    union = len(a | b)
    dice = (2.0 * inter) / (len(a) + len(b) + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


def _surface_points_from_voxels(vox: set[tuple[int, int, int]], voxel_size: float) -> np.ndarray:
    if not vox:
        return np.empty((0, 3), dtype=np.float64)
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    surf: list[tuple[int, int, int]] = []
    for v in vox:
        if any((v[0] + dx, v[1] + dy, v[2] + dz) not in vox for dx, dy, dz in neighbors):
            surf.append(v)
    arr = np.asarray(surf, dtype=np.float64)
    return (arr + 0.5) * float(voxel_size)


def surface_dice(A: np.ndarray, B: np.ndarray, tau: float = 2.0, voxel_size: float = 2.0) -> float:
    """Surface Dice at tolerance tau in mm."""
    a_vox = _voxel_set(A, voxel_size=voxel_size)
    b_vox = _voxel_set(B, voxel_size=voxel_size)
    a_s = _surface_points_from_voxels(a_vox, voxel_size)
    b_s = _surface_points_from_voxels(b_vox, voxel_size)
    if a_s.shape[0] == 0 and b_s.shape[0] == 0:
        return 1.0
    if a_s.shape[0] == 0 or b_s.shape[0] == 0:
        return 0.0
    d_ab = cKDTree(b_s).query(a_s, k=1)[0]
    d_ba = cKDTree(a_s).query(b_s, k=1)[0]
    hit_a = float(np.mean(d_ab <= tau))
    hit_b = float(np.mean(d_ba <= tau))
    denom = a_s.shape[0] + b_s.shape[0]
    return float((hit_a * a_s.shape[0] + hit_b * b_s.shape[0]) / max(denom, 1))


def convex_hull_ratio(pos: np.ndarray, voxel_size: float = 2.0) -> float:
    """Voxel cloud volume divided by convex hull volume."""
    P = _as_points(pos)
    if P.shape[0] < 4:
        return float("nan")
    vox_volume = len(_voxel_set(P, voxel_size=voxel_size)) * (voxel_size ** 3)
    try:
        hull = ConvexHull(P)
        if hull.volume <= 0:
            return float("nan")
        return float(vox_volume / hull.volume)
    except Exception:
        return float("nan")


def sphericity(pos: np.ndarray) -> float:
    """Sphericity based on convex hull area and volume."""
    P = _as_points(pos)
    if P.shape[0] < 4:
        return float("nan")
    try:
        hull = ConvexHull(P)
    except Exception:
        return float("nan")
    if hull.area <= 0 or hull.volume <= 0:
        return float("nan")
    return float((math.pi ** (1.0 / 3.0)) * ((6.0 * hull.volume) ** (2.0 / 3.0)) / hull.area)


def inertia_ratios(pos: np.ndarray) -> dict[str, float]:
    """Planarity, linearity, isotropy from covariance eigenvalues."""
    P = _as_points(pos)
    if P.shape[0] < 3:
        return {"planarity": float("nan"), "linearity": float("nan"), "isotropy": float("nan")}
    X = P - P.mean(axis=0, keepdims=True)
    cov = np.cov(X.T)
    vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    if vals[0] <= 1e-12:
        return {"planarity": float("nan"), "linearity": float("nan"), "isotropy": float("nan")}
    l1, l2, l3 = float(vals[0]), float(vals[1]), float(vals[2])
    return {
        "planarity": (l2 - l3) / l1,
        "linearity": (l1 - l2) / l1,
        "isotropy": l3 / l1,
    }


def longest_diameter(pos: np.ndarray) -> float:
    """Max pairwise Euclidean distance (RECIST-style proxy)."""
    P = _as_points(pos)
    if P.shape[0] < 2:
        return 0.0
    return float(distance.pdist(P).max(initial=0.0))


def sa_to_vol_ratio(pos: np.ndarray, voxel_size: float = 2.0) -> float:
    """Surface-area-to-volume proxy from voxelized cloud."""
    vox = _voxel_set(pos, voxel_size=voxel_size)
    if not vox:
        return float("nan")
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    exposed_faces = 0
    for v in vox:
        for dx, dy, dz in neighbors:
            if (v[0] + dx, v[1] + dy, v[2] + dz) not in vox:
                exposed_faces += 1
    surface_area = exposed_faces * (voxel_size ** 2)
    volume = len(vox) * (voxel_size ** 3)
    if volume <= 0:
        return float("nan")
    return float(surface_area / volume)


def ftv_ratio(ftv_pred: float, ftv_t0: float) -> float:
    if not np.isfinite(ftv_pred) or not np.isfinite(ftv_t0) or ftv_t0 <= 0:
        return float("nan")
    return float(ftv_pred / ftv_t0)


def response_category(ratio: float) -> str:
    """RECIST-like bucket from tumor burden ratio."""
    if not np.isfinite(ratio):
        return "UNK"
    if ratio <= 0.05:
        return "pCR"
    if ratio <= 0.70:
        return "PR"
    if ratio >= 1.20:
        return "PD"
    return "SD"


def betti_numbers(pos: np.ndarray, max_dim: int = 1, threshold: float | None = None) -> dict[str, float]:
    """Betti numbers from ripser diagrams at threshold radius.

    If ripser is unavailable, returns NaNs so callers can degrade gracefully.
    """
    P = _as_points(pos)
    if P.shape[0] == 0:
        return {"betti0": 0.0, "betti1": 0.0}
    try:
        from ripser import ripser  # type: ignore
    except Exception:
        return {"betti0": float("nan"), "betti1": float("nan")}

    if threshold is None:
        nn = cKDTree(P).query(P, k=min(2, P.shape[0]))[0]
        threshold = float(np.percentile(nn[:, -1], 90)) if nn.shape[1] > 1 else 0.0
        threshold = max(threshold, 1e-6)
    dgms = ripser(P, maxdim=max_dim)["dgms"]

    def _betti_at(diag: np.ndarray, t: float) -> float:
        if diag.size == 0:
            return 0.0
        births = diag[:, 0]
        deaths = diag[:, 1]
        alive = (births <= t) & ((deaths > t) | np.isinf(deaths))
        return float(alive.sum())

    b0 = _betti_at(np.asarray(dgms[0]), threshold)
    b1 = _betti_at(np.asarray(dgms[1]), threshold) if len(dgms) > 1 else 0.0
    return {"betti0": b0, "betti1": b1}


def betti_agreement(pred_pos: np.ndarray, obs_pos: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    p = betti_numbers(pred_pos, threshold=threshold)
    o = betti_numbers(obs_pos, threshold=threshold)
    if not np.isfinite(p["betti0"]) or not np.isfinite(o["betti0"]):
        return {"betti0_abs_err": float("nan"), "betti1_abs_err": float("nan"), "betti_l1_err": float("nan")}
    e0 = abs(p["betti0"] - o["betti0"])
    e1 = abs(p["betti1"] - o["betti1"])
    return {"betti0_abs_err": float(e0), "betti1_abs_err": float(e1), "betti_l1_err": float(e0 + e1)}


def ftv_errors(pred: float, obs: float) -> dict[str, float]:
    if not np.isfinite(pred) or not np.isfinite(obs):
        return {"ftv_abs_err_ml": float("nan"), "ftv_rel_err": float("nan"), "ftv_volume_ratio": float("nan")}
    abs_err = abs(pred - obs)
    rel_err = abs_err / obs if obs > 1e-8 else float("nan")
    ratio = pred / obs if obs > 1e-8 else float("nan")
    return {"ftv_abs_err_ml": float(abs_err), "ftv_rel_err": float(rel_err), "ftv_volume_ratio": float(ratio)}


def alive_count_error(pred_alive: np.ndarray, obs_n: int, threshold: float = 0.5) -> int:
    pa = np.asarray(pred_alive, dtype=np.float64)
    if pa.size == 0:
        return int(abs(obs_n))
    pred_n = int((pa >= threshold).sum())
    return int(abs(pred_n - int(obs_n)))


def coverage_90(samples: np.ndarray | list[float], true_val: float) -> int:
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(true_val):
        return 0
    lo, hi = np.percentile(arr, [5, 95])
    return int(lo <= true_val <= hi)


def crps_empirical(samples: np.ndarray | list[float], true_val: float) -> float:
    """CRPS for an empirical predictive distribution."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(true_val):
        return float("nan")
    term1 = np.mean(np.abs(x - true_val))
    pair = np.abs(x[:, None] - x[None, :])
    term2 = 0.5 * np.mean(pair)
    return float(term1 - term2)


def compute_all_cloud_metrics(
    pred_pos: np.ndarray,
    obs_pos: np.ndarray,
    pred_alive: np.ndarray | None = None,
    obs_alive: np.ndarray | None = None,
    pred_ftv: float | None = None,
    obs_ftv: float | None = None,
    ftv_t0: float | None = None,
    voxel_size: float = 2.0,
    compute_topology: bool = True,
) -> dict[str, Any]:
    """Compute a flat dict of metrics for one predicted-vs-observed visit."""
    P = _as_points(pred_pos)
    O = _as_points(obs_pos)
    dice, iou = voxelized_dice_iou(P, O, voxel_size=voxel_size)
    out: dict[str, Any] = {
        "swd_mm": sliced_wasserstein(P, O),
        "chamfer_mm": chamfer_position(P, O),
        "hausdorff95_mm": hausdorff_95(P, O),
        "displacement_mae_mm": displacement_mae(P, O),
        "dice": dice,
        "iou": iou,
        "surface_dice": surface_dice(P, O, tau=voxel_size, voxel_size=voxel_size),
        "sphericity_pred": sphericity(P),
        "sphericity_obs": sphericity(O),
        "diameter_pred_mm": longest_diameter(P),
        "diameter_obs_mm": longest_diameter(O),
        "sa_to_vol_pred": sa_to_vol_ratio(P, voxel_size=voxel_size),
        "sa_to_vol_obs": sa_to_vol_ratio(O, voxel_size=voxel_size),
        "convex_hull_ratio_pred": convex_hull_ratio(P, voxel_size=voxel_size),
        "convex_hull_ratio_obs": convex_hull_ratio(O, voxel_size=voxel_size),
    }

    ir_p = inertia_ratios(P)
    ir_o = inertia_ratios(O)
    for k, v in ir_p.items():
        out[f"{k}_pred"] = v
    for k, v in ir_o.items():
        out[f"{k}_obs"] = v

    if pred_ftv is not None and obs_ftv is not None:
        out.update(ftv_errors(float(pred_ftv), float(obs_ftv)))
    else:
        out.update({"ftv_abs_err_ml": float("nan"), "ftv_rel_err": float("nan"), "ftv_volume_ratio": float("nan")})

    if pred_ftv is not None and ftv_t0 is not None:
        r = ftv_ratio(float(pred_ftv), float(ftv_t0))
        out["ftv_ratio_tk_t0"] = r
        out["response_category_pred"] = response_category(r)
    else:
        out["ftv_ratio_tk_t0"] = float("nan")
        out["response_category_pred"] = "UNK"

    if pred_alive is not None:
        obs_n = int(np.asarray(obs_alive).sum()) if obs_alive is not None else O.shape[0]
        out["alive_count_abs_err"] = alive_count_error(pred_alive, obs_n)
    else:
        out["alive_count_abs_err"] = float("nan")

    if compute_topology:
        out.update(betti_agreement(P, O))
        bp = betti_numbers(P)
        bo = betti_numbers(O)
        out["betti0_pred"] = bp["betti0"]
        out["betti1_pred"] = bp["betti1"]
        out["betti0_obs"] = bo["betti0"]
        out["betti1_obs"] = bo["betti1"]
    else:
        out.update(
            {
                "betti0_abs_err": float("nan"),
                "betti1_abs_err": float("nan"),
                "betti_l1_err": float("nan"),
                "betti0_pred": float("nan"),
                "betti1_pred": float("nan"),
                "betti0_obs": float("nan"),
                "betti1_obs": float("nan"),
            }
        )
    return out


__all__ = [
    "alive_count_error",
    "betti_agreement",
    "betti_numbers",
    "chamfer_position",
    "compute_all_cloud_metrics",
    "convex_hull_ratio",
    "coverage_90",
    "crps_empirical",
    "displacement_mae",
    "ftv_errors",
    "ftv_ratio",
    "hausdorff_95",
    "inertia_ratios",
    "longest_diameter",
    "response_category",
    "sa_to_vol_ratio",
    "sliced_wasserstein",
    "sphericity",
    "surface_dice",
    "voxelized_dice_iou",
]
