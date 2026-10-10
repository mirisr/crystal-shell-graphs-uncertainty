from __future__ import annotations

import numpy as np

from experiments.consistent_rollout.run_conditional_mc import (
    ResidualRecord,
    candidate_residuals,
    conformal_correction,
    sample_alive_mask_gumbel_topk,
)


def test_alive_sampler_is_reproducible_and_count_bounded():
    probs = np.array([0.05, 0.2, 0.9, 0.7, 0.4])
    a = sample_alive_mask_gumbel_topk(probs, 3, np.random.default_rng(7))
    b = sample_alive_mask_gumbel_topk(probs, 3, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert a.dtype == bool
    assert int(a.sum()) == 3

    assert int(sample_alive_mask_gumbel_topk(probs, -10, np.random.default_rng(1)).sum()) == 0
    assert int(sample_alive_mask_gumbel_topk(probs, 99, np.random.default_rng(1)).sum()) == len(probs)


def test_candidate_residuals_exclude_target_patient():
    bucket = [
        ResidualRecord(
            patient_id="p1",
            fold=0,
            start_idx=0,
            pred_idx=3,
            n_sv=10,
            pred_ftv_ml=12.0,
            obs_ftv_ml=13.0,
            ftv_t0_ml=20.0,
            alive_mass_det=8.0,
            collection="ISPY2",
            pCR=0,
            subtype="TNBC",
            ftv_resid=1.0,
            alive_count_resid=2.0,
            global_disp_resid=np.ones(3),
            local_disp_resids=np.ones((2, 3)),
            log_volume_resids=np.array([0.1, 0.2]),
        ),
        ResidualRecord(
            patient_id="p2",
            fold=1,
            start_idx=0,
            pred_idx=3,
            n_sv=10,
            pred_ftv_ml=8.0,
            obs_ftv_ml=7.0,
            ftv_t0_ml=11.0,
            alive_mass_det=5.0,
            collection="ISPY2",
            pCR=1,
            subtype="HR+/HER2-",
            ftv_resid=-1.0,
            alive_count_resid=-2.0,
            global_disp_resid=np.zeros(3),
            local_disp_resids=np.zeros((1, 3)),
            log_volume_resids=np.array([-0.1]),
        ),
    ]

    patient_resids, local, logv = candidate_residuals(bucket, "p1")
    assert [r.patient_id for r in patient_resids] == ["p2"]
    assert local.shape == (1, 3)
    assert logv.tolist() == [-0.1]


def test_conformal_correction_uses_higher_rank_and_empty_zero():
    assert conformal_correction([], alpha=0.10) == 0.0
    # n=5, ceil((n+1)*0.9)=6 -> clipped to rank 5, the max score.
    assert conformal_correction([0.0, 0.5, 1.0, 2.0, 3.0], alpha=0.10) == 3.0
    # n=10, ceil(11*0.8)=9 -> 9th sorted value.
    assert conformal_correction(range(10), alpha=0.20) == 8.0
