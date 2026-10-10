"""Unit tests for the clinical-join helpers in ``build_cohort.py``.

The file-writing CLI is covered indirectly by the main run; these tests pin
the pure helpers so future refactors to the join logic can't silently
drop or duplicate patients.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from build_cohort import (  # noqa: E402
    build_cohort_frame,
    derive_subtype,
    split_tcia_id,
)


def test_split_tcia_id_both_collections():
    assert split_tcia_id("ISPY2-123456") == ("ISPY2", 123456)
    assert split_tcia_id("ACRIN-6698-407539") == ("ACRIN-6698", 407539)


def test_split_tcia_id_rejects_malformed():
    with pytest.raises(ValueError):
        split_tcia_id("random-string")
    with pytest.raises(ValueError):
        split_tcia_id("ISPY3-100001")


def test_derive_subtype_matrix():
    assert derive_subtype(1, 0) == "HR+/HER2-"
    assert derive_subtype(1, 1) == "HR+/HER2+"
    assert derive_subtype(0, 1) == "HR-/HER2+"
    assert derive_subtype(0, 0) == "TNBC"


def _fixture_clinical() -> pd.DataFrame:
    return pd.DataFrame([
        {"Patient_ID": 100001, "Arm": "Paclitaxel", "HR": 1, "HER2": 0,
         "MP": 0, "pCR": 0, "Age_at_Screening": 45.0},
        {"Patient_ID": 100002, "Arm": "Paclitaxel + Pembrolizumab", "HR": 0,
         "HER2": 0, "MP": 1, "pCR": 1, "Age_at_Screening": 52.0},
        {"Patient_ID": 100003, "Arm": "Paclitaxel + Neratinib", "HR": 1,
         "HER2": 1, "MP": 1, "pCR": 1, "Age_at_Screening": 38.0},
    ])


def _fixture_visit_counts() -> pd.DataFrame:
    return pd.DataFrame([
        {"patient_id": "ISPY2-100001", "visit_count": 4, "num_nodes": 400},
        {"patient_id": "ACRIN-6698-100002", "visit_count": 3, "num_nodes": 250},
        {"patient_id": "ISPY2-100003", "visit_count": 2, "num_nodes": 90},
        # unlabeled graph (no clinical row)
        {"patient_id": "ISPY2-999999", "visit_count": 4, "num_nodes": 300},
    ])


def test_build_cohort_frame_happy_path():
    clinical = _fixture_clinical()
    vc = _fixture_visit_counts()
    graph_ids = vc["patient_id"].tolist()
    mfmri_ids = {100001}

    cohort = build_cohort_frame(clinical, graph_ids, vc, mfmri_ids)

    assert len(cohort) == 4
    assert cohort["has_graph"].all()

    ispy_100001 = cohort.loc[cohort["patient_id"] == "ISPY2-100001"].iloc[0]
    assert ispy_100001["collection"] == "ISPY2"
    assert ispy_100001["clinical_patient_id"] == 100001
    assert ispy_100001["pCR"] == 0
    assert ispy_100001["subtype"] == "HR+/HER2-"
    assert ispy_100001["visit_count"] == 4
    assert ispy_100001["in_mfmri"] is True or ispy_100001["in_mfmri"] == True  # noqa
    assert ispy_100001["label_source"] == "cohort1"

    acrin = cohort.loc[cohort["patient_id"] == "ACRIN-6698-100002"].iloc[0]
    assert acrin["collection"] == "ACRIN-6698"
    assert acrin["subtype"] == "TNBC"

    unlabeled = cohort.loc[cohort["patient_id"] == "ISPY2-999999"].iloc[0]
    assert unlabeled["label_source"] == "none"
    assert pd.isna(unlabeled["pCR"]) or unlabeled["pCR"] is None
    assert unlabeled["notes"] == "no_clinical_row"


def test_build_cohort_frame_deterministic_order():
    """Output is sorted by patient_id so repeated runs are byte-identical."""
    clinical = _fixture_clinical()
    vc = _fixture_visit_counts()
    shuffled = vc.sample(frac=1, random_state=0)["patient_id"].tolist()

    c1 = build_cohort_frame(clinical, shuffled, vc, set())
    c2 = build_cohort_frame(clinical, vc["patient_id"].tolist(), vc, set())
    assert c1["patient_id"].tolist() == c2["patient_id"].tolist()


def test_build_cohort_frame_handles_missing_visit_counts():
    """A PID in graph_ids but missing from visit_counts gets 0s, not crash."""
    clinical = _fixture_clinical()
    vc = _fixture_visit_counts().iloc[:1]  # only ISPY2-100001 has counts
    graph_ids = ["ISPY2-100001", "ISPY2-100002"]
    cohort = build_cohort_frame(clinical, graph_ids, vc, set())
    row = cohort.loc[cohort["patient_id"] == "ISPY2-100002"].iloc[0]
    assert row["visit_count"] == 0
    assert row["num_nodes"] == 0
