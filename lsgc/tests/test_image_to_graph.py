from __future__ import annotations

import numpy as np

from experiments.image_to_graph.image_crops import ImageChannelStats
from experiments.image_to_graph.image_patch_io import fixed_patch_view, world_to_voxel_zyx


def test_fixed_patch_view_pads_at_image_boundary() -> None:
    vol = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    patch = fixed_patch_view(vol, np.array([0, 0, 0]), 5, pad_value=-1.0)
    assert patch.shape == (5, 5, 5)
    assert patch[2, 2, 2] == vol[0, 0, 0]
    assert patch[0, 0, 0] == -1.0


def test_world_to_voxel_zyx_uses_origin_and_spacing() -> None:
    world = np.array([12.0, 24.0, 36.0], dtype=np.float32)
    origin = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    spacing = np.array([2.0, 2.0, 3.0], dtype=np.float32)
    assert np.array_equal(world_to_voxel_zyx(world, origin, spacing), np.array([1, 2, 2]))


def test_image_channel_stats_from_values_clamps_std() -> None:
    stats = ImageChannelStats.from_values([1.0, 2.0], [0.0, 3.0])
    assert bool((stats.std > 0).all())
