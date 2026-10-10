from __future__ import annotations

import torch

from experiments.stage1_forecaster.edge_modes import (
    NUM_RELATIONS,
    build_history_graph_with_types,
    edge_attr_dim,
)


def test_ssl_radial_bio_edge_attributes_have_expected_width() -> None:
    pos = [
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
    ]
    x = [torch.randn(4, 70)]

    edge_index, edge_attr, edge_type = build_history_graph_with_types(
        pos,
        x,
        k_spatial=2,
        edge_mode="radial_knn",
        edge_attr_mode="ssl_radial_bio",
    )

    assert edge_index.shape[0] == 2
    assert edge_attr is not None
    assert edge_attr.shape == (edge_index.shape[1], edge_attr_dim("ssl_radial_bio"))
    assert edge_type is not None
    assert edge_type.shape[0] == edge_index.shape[1]
    assert torch.isfinite(edge_attr).all()


def test_multi_rel_ssl_union_adds_ssl_relation() -> None:
    pos = [torch.randn(5, 3)]
    x = [torch.randn(5, 70)]

    edge_index, edge_attr, edge_type = build_history_graph_with_types(
        pos,
        x,
        k_spatial=1,
        edge_mode="multi_rel_ssl_union",
        edge_attr_mode="ssl_radial_bio",
    )

    assert NUM_RELATIONS >= 9
    assert edge_index.shape[1] > 0
    assert edge_attr is not None
    assert edge_attr.shape[1] == 20
    assert edge_type is not None
    assert 8 in set(edge_type.tolist())


def test_radial_ssl_plus_adds_extra_ssl_neighbors() -> None:
    pos = [torch.randn(8, 3)]
    x = [torch.randn(8, 70)]
    base_edge_index, _, _ = build_history_graph_with_types(
        pos,
        x,
        k_spatial=2,
        edge_mode="radial_knn",
        edge_attr_mode="ssl_radial_bio",
    )
    plus_edge_index, edge_attr, _ = build_history_graph_with_types(
        pos,
        x,
        k_spatial=2,
        edge_mode="radial_ssl_plus2",
        edge_attr_mode="ssl_radial_bio",
    )
    assert plus_edge_index.shape[1] >= base_edge_index.shape[1]
    assert edge_attr is not None
    assert edge_attr.shape == (plus_edge_index.shape[1], edge_attr_dim("ssl_radial_bio"))


def test_temporal_only_keeps_identity_edges_without_visit_neighbors() -> None:
    pos = [torch.randn(3, 3), torch.randn(2, 3)]
    x = [torch.randn(3, 70), torch.randn(2, 70)]

    edge_index, edge_attr, edge_type = build_history_graph_with_types(
        pos,
        x,
        k_spatial=2,
        edge_mode="temporal_only",
        edge_attr_mode="ssl_radial_bio",
    )

    assert edge_index.shape == (2, 4)
    assert edge_attr is not None
    assert edge_attr.shape == (4, edge_attr_dim("ssl_radial_bio"))
    assert edge_type is not None
    assert set(edge_type.tolist()) == {4, 5}
