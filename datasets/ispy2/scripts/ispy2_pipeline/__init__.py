"""I-SPY 2 preprocessing pipeline.

Pure-python, no Cradle-specific bits. Turns the raw DCE + Analysis-Mask
DICOMs downloaded under datasets/ispy2/{raw,pilot}/PATIENT/T{0-3}/ into
per-visit supervoxel feature tables and, ultimately, per-patient
spatio-temporal graphs that LSGCNet consumes.

The public entry points are:

    io.load_dce_series(series_dir)            -> DCESeries
    io.load_analysis_mask(series_dir)         -> MaskSeries
    maps.compute_pe_ser(dce, pre_idx, early_idx, late_idx)  -> PEMap, SERMap
    supervoxels.slic_tumor_supervoxels(...)   -> Supervoxels
    supervoxels.aggregate_features(...)       -> pandas.DataFrame

The CLIs `preprocess_patient.py` and `build_patient_graph.py` wrap these
into reproducible per-patient outputs.
"""
from .io import (
    DCESeries,
    MaskSeries,
    load_analysis_mask,
    load_dce_series,
    resample_mask_to_dce_grid,
)
from .maps import compute_pe_ser, kinetic_summary
from .supervoxels import Supervoxels, aggregate_features, load_ucsf_maps, slic_tumor_supervoxels

__all__ = [
    "DCESeries",
    "MaskSeries",
    "load_dce_series",
    "load_analysis_mask",
    "resample_mask_to_dce_grid",
    "compute_pe_ser",
    "kinetic_summary",
    "Supervoxels",
    "slic_tumor_supervoxels",
    "aggregate_features",
    "load_ucsf_maps",
]
