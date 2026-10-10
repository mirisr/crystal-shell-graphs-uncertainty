from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_patient_graph import FEATURE_NAMES_V1, FEATURE_NAMES_V2, _visit_feature_tensor
from ispy2_pipeline.supervoxels import Supervoxels, aggregate_features


def test_aggregate_features_adds_heterogeneity_and_shape_columns() -> None:
    labels = np.zeros((2, 2, 3), dtype=np.int32)
    labels[:, :, :2] = 1
    labels[:, :, 2:] = 2
    supervoxels = Supervoxels(
        labels=labels,
        n_supervoxels=2,
        target_voxels=4,
        compactness=0.1,
    )
    dce = np.stack(
        [
            np.ones_like(labels, dtype=np.float32) * 10,
            np.ones_like(labels, dtype=np.float32) * 15,
            np.ones_like(labels, dtype=np.float32) * 20,
        ]
    )
    pe = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)
    ser = pe / 10.0

    df = aggregate_features(
        supervoxels,
        dce,
        pe,
        ser,
        np.asarray([2.0, 1.0, 1.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    expected = {
        "pe_std",
        "pe_skew",
        "pe_kurtosis",
        "ser_std",
        "ser_skew",
        "ser_kurtosis",
        "surface_area_mm2",
        "sphericity",
        "elongation",
        "major_axis_length_mm",
        "minor_axis_length_mm",
    }
    assert expected.issubset(df.columns)
    assert len(df) == 2
    assert (df["surface_area_mm2"] > 0).all()
    assert (df["sphericity"] > 0).all()


def test_visit_feature_tensor_keeps_v1_and_adds_padded_v2_features() -> None:
    df = pd.DataFrame(
        {
            "voxel_count": [3, 7],
            "volume_ml": [0.1, 0.2],
            "mean_pe": [1.0, 2.0],
            "pe_std": [0.1, 0.2],
            "pe_skew": [0.0, 0.1],
            "pe_kurtosis": [0.0, -0.1],
            "mean_ser": [0.5, 0.7],
            "ser_std": [0.01, 0.02],
            "ser_skew": [0.0, 0.0],
            "ser_kurtosis": [0.0, 0.0],
            "surface_area_mm2": [10.0, 20.0],
            "sphericity": [0.8, 0.7],
            "elongation": [1.2, 1.5],
            "major_axis_length_mm": [4.0, 5.0],
            "minor_axis_length_mm": [2.0, 2.5],
            "phase0_mean_enh": [0.0, 0.0],
            "phase1_mean_enh": [0.5, 0.6],
            "centroid_z_mm": [0.0, 1.0],
            "centroid_y_mm": [0.0, 1.0],
            "centroid_x_mm": [0.0, 1.0],
        }
    )
    meta = {"late_phase": 1, "dce_phases": 2}

    x_v1, pos = _visit_feature_tensor(df, meta, feature_set="v1")
    x_v2, _ = _visit_feature_tensor(df, meta, feature_set="v2")

    assert x_v1.shape == (2, len(FEATURE_NAMES_V1))
    assert x_v2.shape == (2, len(FEATURE_NAMES_V2))
    assert pos.shape == (2, 3)
    phase7_idx = FEATURE_NAMES_V2.index("phase7_mean_enh")
    assert np.allclose(x_v2[:, phase7_idx].numpy(), 0.0)
