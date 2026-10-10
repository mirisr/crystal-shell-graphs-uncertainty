#!/usr/bin/env python3
"""Attach frozen torchvision patch embeddings to consistent graph payloads."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.image_to_graph.image_crops import (  # noqa: E402
    ImageChannelStats,
    extract_visit_crops,
)


def _read_patient_list(path: Path | None, graph_root: Path, patients: Sequence[str] | None) -> list[str]:
    if patients:
        return [str(p) for p in patients]
    if path is not None and path.is_file():
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return sorted(p.stem for p in graph_root.glob("*.pt"))


def _visit_offsets(payload: dict) -> list[int]:
    offsets = payload.get("visit_offsets")
    if offsets is None:
        raise KeyError("graph payload is missing visit_offsets")
    if torch.is_tensor(offsets):
        return [int(x) for x in offsets.detach().cpu().tolist()]
    return [int(x) for x in offsets]


def _load_resnet18(device: torch.device) -> tuple[nn.Module, int]:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, 512


def _robust_scale_2d(images: torch.Tensor) -> torch.Tensor:
    """Scale PE/SER projection channels independently to [0, 1]."""
    out = images.clone()
    for channel in (0, 1):
        flat = out[:, channel].flatten(1)
        lo = torch.quantile(flat, 0.01, dim=1).view(-1, 1, 1)
        hi = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1)
        denom = (hi - lo).clamp_min(1e-6)
        out[:, channel] = ((out[:, channel] - lo) / denom).clamp(0.0, 1.0)
    out[:, 2] = out[:, 2].clamp(0.0, 1.0)
    return out


def _crops_to_images(crops: torch.Tensor, projection: str) -> torch.Tensor:
    if projection == "mip":
        images = crops.amax(dim=2)
    elif projection == "center":
        images = crops[:, :, crops.shape[2] // 2]
    else:
        raise ValueError(f"unsupported projection: {projection}")
    images = _robust_scale_2d(images.float())
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _encode_in_batches(
    model: nn.Module,
    crops: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    projection: str,
) -> torch.Tensor:
    if crops.numel() == 0:
        return torch.zeros((0, 512), dtype=torch.float32)
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(crops.shape[0]), int(batch_size)):
            piece = crops[start:start + int(batch_size)].to(device=device, dtype=torch.float32)
            images = _crops_to_images(piece, projection=projection)
            chunks.append(model(images).detach().cpu().float())
    return torch.cat(chunks, dim=0)


def attach_torchvision_embeddings_to_payload(
    payload: dict,
    *,
    derived_root: Path,
    model: nn.Module,
    encoder_meta: dict,
    device: torch.device,
    batch_size: int,
    concat_to_x: bool,
) -> tuple[dict, list[dict]]:
    patient_id = str(payload.get("patient_id") or "")
    if not patient_id:
        raise KeyError("graph payload is missing patient_id")
    visit_ids = list(payload.get("visit_ids") or [])
    offsets = _visit_offsets(payload)
    if len(visit_ids) + 1 != len(offsets):
        raise ValueError(
            f"{patient_id}: visit_ids length {len(visit_ids)} does not match visit_offsets {len(offsets)}"
        )

    n_nodes = int(payload["x"].shape[0])
    embedding_dim = int(encoder_meta["embedding_dim"])
    x_embed = torch.zeros((n_nodes, embedding_dim), dtype=torch.float32)
    node_ok = torch.zeros((n_nodes,), dtype=torch.bool)
    reports: list[dict] = []
    raw_stats = ImageChannelStats.from_values([0.0, 0.0], [1.0, 1.0])

    for visit_idx, visit_id in enumerate(visit_ids):
        start, end = offsets[visit_idx], offsets[visit_idx + 1]
        expected = int(end - start)
        try:
            crops = extract_visit_crops(
                derived_root=derived_root,
                payload=payload,
                visit_idx=visit_idx,
                crop_size=int(encoder_meta["crop_size"]),
                channel_stats=raw_stats,
                node_indices=None,
                include_label=True,
            )
            emb = _encode_in_batches(
                model,
                crops,
                device=device,
                batch_size=batch_size,
                projection=str(encoder_meta["projection"]),
            )
            if emb.shape != (expected, embedding_dim):
                raise RuntimeError(
                    f"{patient_id} {visit_id}: expected embedding {(expected, embedding_dim)}, got {tuple(emb.shape)}"
                )
            x_embed[start:end] = emb
            node_ok[start:end] = True
            reports.append(
                {
                    "visit_id": str(visit_id),
                    "status": "ok",
                    "nodes_expected": expected,
                    "nodes_attached": expected,
                }
            )
        except Exception as exc:
            reports.append(
                {
                    "visit_id": str(visit_id),
                    "status": "failed",
                    "nodes_expected": expected,
                    "nodes_attached": 0,
                    "detail": str(exc),
                }
            )

    out = dict(payload)
    out["x_ssl_patch_embedding"] = x_embed
    out["ssl_patch_node_ok"] = node_ok
    out["ssl_patch_feature_names"] = [f"torchvision_resnet18_emb_{i:03d}" for i in range(embedding_dim)]
    out["ssl_patch_meta"] = {
        "source": "torchvision_resnet18_imagenet1k_v1",
        "encoder": encoder_meta,
        "concat_to_x": bool(concat_to_x),
        "visits": reports,
    }

    if concat_to_x:
        original_x = out["x"].float()
        out["x_pre_ssl_patch_embedding"] = original_x
        out["x"] = torch.cat([original_x, x_embed], dim=1)
        original_names = list(out.get("feature_names") or [f"x{i}" for i in range(original_x.shape[1])])
        out["pre_ssl_feature_names"] = original_names
        out["feature_names"] = original_names + out["ssl_patch_feature_names"]

    return out, reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived-root", type=Path, default=Path("datasets/ispy2/derived"))
    ap.add_argument("--graph-root", type=Path, default=Path("datasets/ispy2/graphs_consistent"))
    ap.add_argument("--out-root", type=Path, default=Path("datasets/ispy2/graphs_consistent_torchvision_resnet18_embed"))
    ap.add_argument("--patient-list", type=Path, default=None)
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--crop-size", type=int, default=32)
    ap.add_argument("--projection", choices=("mip", "center"), default="mip")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--concat-to-x", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("attach_torchvision_patch_embeddings")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, embedding_dim = _load_resnet18(device)
    encoder_meta = {
        "model": "torchvision.resnet18",
        "weights": "IMAGENET1K_V1",
        "embedding_dim": int(embedding_dim),
        "crop_size": int(args.crop_size),
        "projection": str(args.projection),
        "input_channels": ["PE", "SER", "support"],
        "normalization": "per-crop 1st/99th percentile for PE/SER plus ImageNet channel normalization",
    }

    patients = _read_patient_list(args.patient_list, args.graph_root, args.patients)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    t0 = time.time()
    for idx, patient_id in enumerate(patients, start=1):
        graph_path = args.graph_root / f"{patient_id}.pt"
        out_path = args.out_root / f"{patient_id}.pt"
        if out_path.exists() and not args.overwrite:
            manifest.append({"patient_id": patient_id, "status": "exists", "path": str(out_path)})
            continue
        if not graph_path.exists():
            manifest.append({"patient_id": patient_id, "status": "missing_graph", "path": ""})
            continue
        try:
            payload = torch.load(graph_path, map_location="cpu", weights_only=False)
            augmented, reports = attach_torchvision_embeddings_to_payload(
                payload,
                derived_root=args.derived_root,
                model=model,
                encoder_meta=encoder_meta,
                device=device,
                batch_size=int(args.batch_size),
                concat_to_x=bool(args.concat_to_x),
            )
            torch.save(augmented, out_path)
            ok_nodes = int(augmented["ssl_patch_node_ok"].sum().item())
            total_nodes = int(augmented["ssl_patch_node_ok"].numel())
            status = "ok" if ok_nodes == total_nodes else "partial"
            manifest.append(
                {
                    "patient_id": patient_id,
                    "status": status,
                    "path": str(out_path),
                    "nodes_attached": ok_nodes,
                    "nodes_total": total_nodes,
                    "failed_visits": sum(1 for r in reports if r.get("status") != "ok"),
                }
            )
        except Exception as exc:
            log.exception("failed %s", patient_id)
            manifest.append({"patient_id": patient_id, "status": "failed", "path": "", "detail": str(exc)})
        if idx % int(args.progress_every) == 0:
            log.info("processed %d/%d elapsed=%.1fs", idx, len(patients), time.time() - t0)

    manifest_path = args.out_root / "torchvision_patch_embedding_manifest.json"
    manifest_path.write_text(json.dumps({"encoder": encoder_meta, "patients": manifest}, indent=2))
    ok = sum(1 for row in manifest if row.get("status") in {"ok", "exists"})
    partial = sum(1 for row in manifest if row.get("status") == "partial")
    failed = sum(1 for row in manifest if row.get("status") == "failed")
    log.info("wrote %s ok=%d partial=%d failed=%d", manifest_path, ok, partial, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
