"""Unit tests for LSGCForecaster."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from lsgc.forecaster import LSGCForecaster  # noqa: E402
from lsgc.graph_builder import build_spatiotemporal_graph  # noqa: E402


def _dummy_patient():
    torch.manual_seed(0)
    visit_feats = [torch.randn(n, 5) for n in (10, 8, 7)]
    visit_pos = [torch.randn(n, 3) * 10 for n in (10, 8, 7)]
    return build_spatiotemporal_graph(
        visit_feats, visit_pos, k_spatial=3, k_temporal=2,
        temporal_skip_hops=(1, 2),
    ), visit_feats, visit_pos


def test_forecaster_output_shapes():
    g, feats, pos = _dummy_patient()
    n = g.x.shape[0]
    model = LSGCForecaster(in_channels=5, hidden=16, num_layers=2)
    out = model(g.x, g.pos, g.t, g.edge_index)
    assert out["delta_pos"].shape == (n, 3)
    assert out["delta_feat"].shape == (n, 5)
    assert out["alive_logit"].shape == (n,)
    assert out["hidden"].shape == (n, 16)


def test_forecaster_respects_use_delta_t_flag():
    g, _, _ = _dummy_patient()
    torch.manual_seed(1)
    m_on = LSGCForecaster(in_channels=5, hidden=16, num_layers=2, use_delta_t=True)
    torch.manual_seed(1)
    m_off = LSGCForecaster(in_channels=5, hidden=16, num_layers=2, use_delta_t=False)

    out_on_1 = m_on(g.x, g.pos, g.t, g.edge_index, delta_t=1.0)
    out_on_3 = m_on(g.x, g.pos, g.t, g.edge_index, delta_t=3.0)
    # Varying delta_t must change predictions when use_delta_t is on.
    assert not torch.allclose(out_on_1["delta_pos"], out_on_3["delta_pos"])

    out_off_1 = m_off(g.x, g.pos, g.t, g.edge_index, delta_t=1.0)
    out_off_3 = m_off(g.x, g.pos, g.t, g.edge_index, delta_t=3.0)
    # With the flag off, predictions are invariant to delta_t.
    assert torch.allclose(out_off_1["delta_pos"], out_off_3["delta_pos"])


def test_forecaster_gradients_flow_to_backbone():
    g, _, _ = _dummy_patient()
    model = LSGCForecaster(in_channels=5, hidden=16, num_layers=2)
    out = model(g.x, g.pos, g.t, g.edge_index)
    loss = out["delta_pos"].square().mean() + out["delta_feat"].square().mean() \
        + out["alive_logit"].abs().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.convs.parameters())
    assert model.embed.weight.grad is not None


def test_forecaster_ssl_adapter_uses_core_embed_and_core_delta_width():
    torch.manual_seed(2)
    n = 6
    x = torch.randn(n, 9)
    pos = torch.randn(n, 3)
    t = torch.zeros(n)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    model = LSGCForecaster(
        in_channels=9,
        core_in_channels=5,
        ssl_adapter_dim=4,
        hidden=16,
        num_layers=1,
        feat_out_dim=5,
    )
    out = model(x, pos, t, edge_index)
    assert model.embed.weight.shape == (16, 5)
    assert out["delta_feat"].shape == (n, 5)
    assert out["delta_pos"].shape == (n, 3)


def test_forecaster_optional_graph_heads():
    torch.manual_seed(3)
    n = 7
    x = torch.randn(n, 6)
    pos = torch.randn(n, 3)
    t = torch.zeros(n)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    model = LSGCForecaster(
        in_channels=6,
        hidden=12,
        num_layers=1,
        graph_ftv_readout=True,
        graph_pcr_readout=True,
        graph_ftv_affine=True,
    )
    out = model(x, pos, t, edge_index)
    assert out["graph_ftv_log"].shape == ()
    assert out["graph_pcr_logit"].shape == ()
    assert out["graph_ftv_affine"].shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
