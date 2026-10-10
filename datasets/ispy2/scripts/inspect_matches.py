#!/usr/bin/env python
"""Sanity-check visualization + stats for the precomputed match sidecars.

For one or more patients, prints a compact summary of the matching
quality and optionally dumps a PNG with the pair-cost distribution and
a 2-D projection of matched pairs for every transition.

Usage
-----

Stats only (no plotting; safe over SSH)::

    python datasets/ispy2/scripts/inspect_matches.py \
        --graphs-root datasets/ispy2/graphs \
        --patient-id ISPY2-212307

Stats + figure::

    python datasets/ispy2/scripts/inspect_matches.py \
        --graphs-root datasets/ispy2/graphs \
        --patient-id ISPY2-212307 \
        --out-png /tmp/ISPY2-212307.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def describe_patient(graph_path: Path, match_path: Path) -> dict:
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    match = torch.load(match_path, map_location="cpu", weights_only=False)

    offsets = graph["visit_offsets"].tolist()
    pos = graph["pos"]

    print(f"\n=== {match['patient_id']} ({match['method']}) ===")
    print(f"visits: {graph['visit_ids']}  (T={match['num_visits']})")

    per_transition: list[dict] = []
    for t in match["transitions"]:
        k = t["k"]
        n_src = int(t["alive"].numel())
        n_live = int(t["alive"].sum().item())
        n_dead = n_src - n_live
        frac_live = n_live / max(n_src, 1)
        if t["delta_pos"].shape[0] > 0:
            dist_mm = t["delta_pos"].norm(dim=1)
            median_drift = dist_mm.median().item()
            p95_drift = dist_mm.quantile(0.95).item()
        else:
            median_drift = p95_drift = float("nan")
        print(
            f"  k={k}: n_src={n_src:4d} live={n_live:4d} ({_fmt_pct(frac_live)}) "
            f"dead={n_dead:4d}  drift median={median_drift:6.2f}mm "
            f"p95={p95_drift:6.2f}mm  cost_thresh={t['threshold']:.3f}"
        )
        per_transition.append({
            "k": k, "n_src": n_src, "n_live": n_live,
            "median_drift_mm": median_drift, "p95_drift_mm": p95_drift,
        })

    return {
        "patient_id": match["patient_id"],
        "visits": graph["visit_ids"],
        "offsets": offsets,
        "per_transition": per_transition,
    }


def plot_patient(graph_path: Path, match_path: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    match = torch.load(match_path, map_location="cpu", weights_only=False)

    offsets = graph["visit_offsets"].tolist()
    pos = graph["pos"].numpy()
    visit_ids = graph["visit_ids"]
    transitions = match["transitions"]

    # Two rows per transition: 2-D projection + cost histogram.
    n_tr = len(transitions)
    fig, axes = plt.subplots(n_tr, 2, figsize=(12, 3.3 * n_tr), squeeze=False)

    for row, t in enumerate(transitions):
        k = t["k"]
        off_k = offsets[k]; off_kp = offsets[k + 1]; off_kp2 = offsets[k + 2]
        p_src = pos[off_k: off_kp]
        p_tgt = pos[off_kp: off_kp2]

        ax = axes[row, 0]
        # Axial (y, x) projection, mm.
        ax.scatter(p_src[:, 2], p_src[:, 1], s=6, c="tab:blue", alpha=0.5,
                   label=f"{visit_ids[k]} src ({p_src.shape[0]})")
        ax.scatter(p_tgt[:, 2], p_tgt[:, 1], s=6, c="tab:orange", alpha=0.5,
                   label=f"{visit_ids[k + 1]} tgt ({p_tgt.shape[0]})")
        alive = t["alive"].numpy()
        tgt_idx = t["target_idx"].numpy()
        for i, (a, j) in enumerate(zip(alive, tgt_idx)):
            if not a:
                continue
            x0, y0 = p_src[i, 2], p_src[i, 1]
            x1, y1 = p_tgt[j, 2], p_tgt[j, 1]
            ax.plot([x0, x1], [y0, y1], c="gray", lw=0.3, alpha=0.35)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{visit_ids[k]} → {visit_ids[k + 1]}  "
                     f"alive={int(alive.sum())}/{len(alive)}")
        ax.legend(fontsize=7, loc="best")

        ax = axes[row, 1]
        cost = t["pair_cost"].numpy()
        ax.hist(cost, bins=40, color="tab:gray", alpha=0.8)
        ax.axvline(t["threshold"], c="tab:red", lw=1.2, label=f"thresh={t['threshold']:.2f}")
        ax.set_xlabel("pair cost"); ax.set_ylabel("count")
        ax.set_title("per-pair matching cost")
        ax.legend(fontsize=7)

    fig.suptitle(f"{match['patient_id']} — inter-visit supervoxel matches", y=1.0)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-root", type=Path, required=True)
    ap.add_argument("--patient-id", required=True)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--method", choices=("sinkhorn", "nn"), default="sinkhorn")
    args = ap.parse_args()

    graph_path = args.graphs_root / f"{args.patient_id}.pt"
    suffix = ".match.pt" if args.method == "sinkhorn" else f".match-{args.method}.pt"
    match_path = args.graphs_root / f"{args.patient_id}{suffix}"
    if not match_path.exists():
        raise SystemExit(f"{match_path} does not exist -- run match_visits.py first")

    describe_patient(graph_path, match_path)
    if args.out_png is not None:
        plot_patient(graph_path, match_path, args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
