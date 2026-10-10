#!/usr/bin/env python3
"""Load the released NCA checkpoint and run a synthetic T0-to-T3 rollout."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lsgc.inference import load_model, rollout_from_visit


def _expected_checksums() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (ROOT / "models" / "checksums.sha256").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        entries[relative.strip()] = digest
    return entries


def _verify_checkpoints() -> list[Path]:
    checkpoints: list[Path] = []
    for relative, expected in _expected_checksums().items():
        path = ROOT / "models" / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checkpoint checksum mismatch: {path}")
        load_model(path)
        checkpoints.append(path)
    if len(checkpoints) != 5:
        raise RuntimeError(f"expected five fold checkpoints, found {len(checkpoints)}")
    return checkpoints


def _synthetic_graph(n_nodes: int = 7, n_features: int = 518) -> dict:
    torch.manual_seed(17)
    visits = 4
    x = torch.randn(n_nodes * visits, n_features)
    x[:, 1] = torch.rand(n_nodes * visits) * 2.0
    positions = torch.cat(
        [torch.randn(n_nodes, 3) + visit * 0.15 for visit in range(visits)]
    )
    return {
        "x": x,
        "pos": positions,
        "t": torch.repeat_interleave(torch.arange(visits).float(), n_nodes),
        "visit_offsets": torch.arange(0, (visits + 1) * n_nodes, n_nodes),
        "alive": torch.ones(n_nodes * visits, dtype=torch.bool),
        "n_supervoxels": n_nodes,
    }


def main() -> None:
    checkpoints = _verify_checkpoints()
    model, mean, std = load_model(checkpoints[0])
    rollout = rollout_from_visit(
        model,
        mean,
        std,
        _synthetic_graph(),
        start_visit=0,
        k_spatial=8,
        edge_mode="crystal_shell_union",
        edge_attr_mode="ssl_radial_bio",
    )
    if len(rollout) != 3:
        raise RuntimeError(f"expected three forecast visits, found {len(rollout)}")
    for row in rollout:
        for key in ("pos_pred", "x_pred", "alive_prob"):
            if not np.isfinite(row[key]).all():
                raise RuntimeError(f"non-finite values in {row['visit_name']} {key}")
    print(
        "NCA checkpoint smoke test passed:",
        f"checkpoints={len(checkpoints)}",
        f"forecast_visits={len(rollout)}",
        f"final_ftv_ml={rollout[-1]['graph_ftv_pred_ml']:.6f}",
    )


if __name__ == "__main__":
    main()
