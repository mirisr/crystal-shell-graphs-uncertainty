#!/usr/bin/env python
"""Precompute inter-visit supervoxel matches for the Stage-1 forecaster.

For every patient graph under ``--graphs-root`` we produce a sidecar
``<patient>.match.pt`` alongside it with the following payload:

    {
      "patient_id":       str,
      "method":           str,           # "sinkhorn" | "nn"
      "num_visits":       int,           # T
      "transitions": [                   # one entry per (k -> k+1)
        {
          "k":             int,          # source visit index
          "target_idx":    LongTensor (n_k,),   # target visit index, -1 if dead
          "alive":         BoolTensor (n_k,),
          "pair_cost":     FloatTensor (n_k,),
          "threshold":     float,
          "delta_pos":     FloatTensor (n_live, 3),  # matched-only gt drift
          "delta_feat":    FloatTensor (n_live, C),  # matched-only gt feature change
          "src_live_idx":  LongTensor (n_live,),     # which source rows were alive
        },
        ...
      ],
      "feature_names":    list[str],     # copied from graph
      "alpha_pos":        float,
      "beta_feat":        float,
      "cost_quantile":    float,
    }

Features are z-scored patient-wide before matching so positions (mm) and
features (unitless) are on commensurate scales; the returned
``delta_pos`` is in millimeters and ``delta_feat`` is in the *original*
feature units (i.e. as saved on the graph).

Usage
-----

Single patient::

    python datasets/ispy2/scripts/match_visits.py \
        --graphs-root datasets/ispy2/graphs --patient-id ISPY2-212307

Whole cohort, in parallel::

    python datasets/ispy2/scripts/match_visits.py \
        --graphs-root datasets/ispy2/graphs --all --num-workers 8

The script is resumable: existing ``.match.pt`` files are skipped
unless ``--overwrite`` is passed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lsgc.matching import match_visits, standardize_features  # noqa: E402


def _extract_visit_tensors(payload: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return per-visit feature and position lists from a graph payload."""
    offsets = payload["visit_offsets"].tolist()
    x = payload["x"]
    pos = payload["pos"]
    h_list = [x[offsets[v]: offsets[v + 1]] for v in range(len(offsets) - 1)]
    p_list = [pos[offsets[v]: offsets[v + 1]] for v in range(len(offsets) - 1)]
    return h_list, p_list


def compute_patient_matches(
    graph_path: Path,
    *,
    method: str = "sinkhorn",
    alpha_pos: float = 1.0,
    beta_feat: float = 1.0,
    cost_quantile_alive: float = 0.90,
    sinkhorn_reg: float = 0.1,
    sinkhorn_iters: int = 200,
    center_positions: bool = True,
) -> dict:
    """Return the matching sidecar dict for one patient graph.

    When ``center_positions`` is True (default), the positions of each
    visit are translated so their centroid is at the origin before
    computing the matching cost. This removes the bulk "patient
    repositioning" offset between visits (observed in practice to be
    tens of millimeters because DCE-MRI visits are acquired weeks apart
    with different compression/orientation), so that matches reflect
    tumor *shape* evolution rather than global positioning. The saved
    ``delta_pos`` is also in this centered frame; the per-visit
    centroids are saved so absolute positions can be reconstructed if
    needed.
    """
    payload = torch.load(graph_path, map_location="cpu", weights_only=False)

    h_raw, p_raw = _extract_visit_tensors(payload)
    if len(h_raw) < 2:
        raise ValueError(f"{graph_path.name}: need >=2 visits, got {len(h_raw)}")

    centroids = [p.mean(dim=0) if p.shape[0] > 0
                 else torch.zeros(3, dtype=p.dtype) for p in p_raw]
    if center_positions:
        p_list = [p - c for p, c in zip(p_raw, centroids)]
    else:
        p_list = p_raw

    # Standardize features patient-wide so alpha=beta=1 is sensible.
    h_list = standardize_features(h_raw)

    transitions: list[dict] = []
    for k in range(len(h_list) - 1):
        res = match_visits(
            p_list[k], h_list[k], p_list[k + 1], h_list[k + 1],
            method=method,
            alpha=alpha_pos, beta=beta_feat,
            sinkhorn_reg=sinkhorn_reg, sinkhorn_iters=sinkhorn_iters,
            cost_quantile_alive=cost_quantile_alive,
        )
        alive = res.alive
        src_live_idx = torch.nonzero(alive, as_tuple=False).flatten()
        tgt_live_idx = res.target_idx[alive]

        if src_live_idx.numel() > 0:
            delta_pos = p_list[k + 1][tgt_live_idx] - p_list[k][src_live_idx]
            delta_feat = h_raw[k + 1][tgt_live_idx] - h_raw[k][src_live_idx]
        else:
            delta_pos = torch.empty(0, 3, dtype=p_list[k].dtype)
            delta_feat = torch.empty(0, h_raw[k].shape[1], dtype=h_raw[k].dtype)

        transitions.append({
            "k": k,
            "target_idx": res.target_idx.long(),
            "alive": res.alive,
            "pair_cost": res.pair_cost.float(),
            "threshold": float(res.threshold),
            "delta_pos": delta_pos.float(),
            "delta_feat": delta_feat.float(),
            "src_live_idx": src_live_idx.long(),
            "effective_method": res.method,
        })

    return {
        "patient_id": payload.get("patient_id", graph_path.stem),
        "method": method,
        "num_visits": len(h_list),
        "transitions": transitions,
        "feature_names": payload.get("feature_names", []),
        "alpha_pos": float(alpha_pos),
        "beta_feat": float(beta_feat),
        "cost_quantile": float(cost_quantile_alive),
        "center_positions": bool(center_positions),
        "visit_centroids": torch.stack(centroids, dim=0).float(),  # (T, 3)
    }


def _process_one(args_tuple: tuple[Path, Path, str, float, float, float, bool]) -> tuple[str, str]:
    # Keep each worker single-threaded; otherwise 16 workers x 16 torch threads
    # stampede the node and wall-time explodes.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # can only be set once per process; ignore on reuse.

    graph_path, out_path, method, alpha, beta, cq, overwrite = args_tuple
    try:
        if out_path.exists() and not overwrite:
            return (graph_path.stem, "skip_existing")
        data = compute_patient_matches(
            graph_path, method=method,
            alpha_pos=alpha, beta_feat=beta,
            cost_quantile_alive=cq,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        live = sum(int(t["alive"].sum().item()) for t in data["transitions"])
        total = sum(int(t["alive"].numel()) for t in data["transitions"])
        return (graph_path.stem, f"ok live={live}/{total}")
    except Exception as exc:  # pragma: no cover -- logged by caller
        return (graph_path.stem, f"fail {type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-root", type=Path, required=True,
                    help="Directory containing <patient>.pt files.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Where to write <patient>.match.pt (default: same as graphs-root).")
    ap.add_argument("--patient-id", default=None,
                    help="Process a single patient.")
    ap.add_argument("--all", action="store_true",
                    help="Process every *.pt under --graphs-root (excluding *.match.pt).")
    ap.add_argument("--method", choices=("sinkhorn", "nn"), default="sinkhorn")
    ap.add_argument("--alpha-pos", type=float, default=1.0)
    ap.add_argument("--beta-feat", type=float, default=1.0)
    ap.add_argument("--cost-quantile", type=float, default=0.90)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("match_visits")

    out_root = args.out_root or args.graphs_root
    out_root.mkdir(parents=True, exist_ok=True)

    def _sidecar_name(graph_path: Path) -> Path:
        suffix = ".match.pt" if args.method == "sinkhorn" else f".match-{args.method}.pt"
        return out_root / (graph_path.stem + suffix)

    if args.patient_id:
        graph_paths = [args.graphs_root / f"{args.patient_id}.pt"]
    elif args.all:
        graph_paths = sorted(
            p for p in args.graphs_root.glob("*.pt") if ".match" not in p.name
        )
    else:
        ap.error("specify --patient-id or --all")

    log.info("matching %d patients (method=%s) -> %s",
             len(graph_paths), args.method, out_root)

    work = [
        (gp, _sidecar_name(gp), args.method, args.alpha_pos,
         args.beta_feat, args.cost_quantile, args.overwrite)
        for gp in graph_paths
    ]

    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    if args.num_workers <= 1:
        for item in work:
            pid, status = _process_one(item)
            _log_one(log, pid, status)
            n_ok, n_skip, n_fail = _tally(status, n_ok, n_skip, n_fail)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futs = {pool.submit(_process_one, item): item[0].stem for item in work}
            for i, fut in enumerate(as_completed(futs), start=1):
                pid, status = fut.result()
                _log_one(log, pid, status)
                n_ok, n_skip, n_fail = _tally(status, n_ok, n_skip, n_fail)
                if i % 50 == 0:
                    log.info("progress %d/%d (ok=%d skip=%d fail=%d)",
                             i, len(work), n_ok, n_skip, n_fail)

    log.info("done in %.1fs: ok=%d skip=%d fail=%d",
             time.time() - t0, n_ok, n_skip, n_fail)
    return 0


def _log_one(log: logging.Logger, pid: str, status: str) -> None:
    if status.startswith("fail"):
        log.warning("%s: %s", pid, status)
    else:
        log.debug("%s: %s", pid, status)


def _tally(status: str, n_ok: int, n_skip: int, n_fail: int) -> tuple[int, int, int]:
    if status.startswith("ok"):
        n_ok += 1
    elif status == "skip_existing":
        n_skip += 1
    else:
        n_fail += 1
    return n_ok, n_skip, n_fail


if __name__ == "__main__":
    raise SystemExit(main())
