"""Voxel-wise DCE feature maps: Percent Enhancement (PE) and Signal Enhancement Ratio (SER).

Both are standard I-SPY 2 features, defined on the DCE intensity time-series
S(phase, z, y, x):

    PE  = 100 * (S_early - S_pre) / S_pre
    SER = (S_early - S_pre) / (S_late - S_pre)

* `pre`   = first phase (pre-contrast baseline, index 0 by convention).
* `early` = first phase acquired post-contrast (~90 s after injection).
* `late`  = last phase acquired (~7.5 min after injection).

When the series has more than three phases (I-SPY 2 Siemens cohorts often
have 9) the extra intermediate phases are still useful as a kinetic-curve
summary -- see `kinetic_summary`. They are *not* required for PE/SER.

All numerics are float32 by default to keep memory down on 4-D volumes.
Divide-by-zero at voxels with `S_pre == 0` (or `S_late == S_pre`) is handled
by returning 0.0 at those voxels, which is a conservative choice -- a pure
0 in feature space is clearly excludable downstream if we need to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-3  # absolute intensity floor; DCE is uint16 so this is well below noise


@dataclass
class PEMap:
    data: np.ndarray          # (Z, Y, X) float32, units: percent
    pre_phase: int
    early_phase: int


@dataclass
class SERMap:
    data: np.ndarray          # (Z, Y, X) float32, unitless
    pre_phase: int
    early_phase: int
    late_phase: int


def compute_pe_ser(
    dce: np.ndarray,
    *,
    pre_phase: int = 0,
    early_phase: int | None = None,
    late_phase: int | None = None,
) -> tuple[PEMap, SERMap]:
    """Compute PE and SER maps from a (P, Z, Y, X) DCE volume.

    Defaults:
        pre_phase   = 0            (first phase)
        early_phase = 1 if P >= 3 else 1  (second phase)
        late_phase  = P - 1        (last phase)
    """
    if dce.ndim != 4:
        raise ValueError(f"dce must be 4-D (P, Z, Y, X); got shape {dce.shape}")
    p = dce.shape[0]
    if p < 2:
        raise ValueError(f"need at least 2 phases for PE; got P={p}")

    if early_phase is None:
        early_phase = 1 if p >= 2 else p - 1
    if late_phase is None:
        late_phase = p - 1

    for name, idx in (("pre", pre_phase), ("early", early_phase), ("late", late_phase)):
        if not 0 <= idx < p:
            raise IndexError(f"{name}_phase={idx} out of range for P={p}")

    s_pre = dce[pre_phase].astype(np.float32)
    s_early = dce[early_phase].astype(np.float32)

    pe = np.zeros_like(s_pre)
    valid_pre = s_pre > EPS
    pe[valid_pre] = 100.0 * (s_early[valid_pre] - s_pre[valid_pre]) / s_pre[valid_pre]

    if late_phase == pre_phase:
        ser = np.zeros_like(s_pre)
    else:
        s_late = dce[late_phase].astype(np.float32)
        num = s_early - s_pre
        den = s_late - s_pre
        ser = np.zeros_like(num)
        valid_den = np.abs(den) > EPS
        ser[valid_den] = num[valid_den] / den[valid_den]

    return (
        PEMap(data=pe, pre_phase=pre_phase, early_phase=early_phase),
        SERMap(
            data=ser,
            pre_phase=pre_phase,
            early_phase=early_phase,
            late_phase=late_phase,
        ),
    )


def kinetic_summary(
    dce: np.ndarray,
    mask: np.ndarray,
    *,
    pre_phase: int = 0,
) -> np.ndarray:
    """Compact kinetic-curve summary per phase, averaged over `mask`.

    For each phase `p`, returns the mean of `(S_p - S_pre) / S_pre` inside
    the mask (so `curve[pre_phase] == 0` by construction). Useful as a small
    per-visit global feature you can concatenate onto node features.

    Returns: shape (P,) float32.
    """
    if dce.ndim != 4:
        raise ValueError(f"dce must be 4-D (P, Z, Y, X); got {dce.shape}")
    if mask.shape != dce.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} != dce spatial {dce.shape[1:]}")
    if not mask.any():
        return np.zeros(dce.shape[0], dtype=np.float32)

    s_pre = dce[pre_phase].astype(np.float32)
    s_pre_m = s_pre[mask]
    # Mean inside mask of normalized enhancement for each phase.
    curve = np.zeros(dce.shape[0], dtype=np.float32)
    valid = s_pre_m > EPS
    denom = s_pre_m[valid]
    if denom.size == 0:
        return curve
    for p in range(dce.shape[0]):
        s_p = dce[p].astype(np.float32)[mask][valid]
        curve[p] = float(((s_p - denom) / denom).mean())
    return curve
