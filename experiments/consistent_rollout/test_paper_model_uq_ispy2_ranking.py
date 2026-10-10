#!/usr/bin/env python3
"""Ranking must use I-SPY2 OOF only; ACRIN columns/rows fail the assertion."""

from __future__ import annotations

import pandas as pd

from train_paper_model_uq_methods import _assert_ranking_is_ispy2_only, _select_winner


def _row(method: str, feature_mode: str, interval_score: float, coverage: float, width: float) -> dict:
    return {
        "split": "ispy2_oof",
        "feature_mode": feature_mode,
        "method": method,
        "n": 556.0,
        "mae_ml": 3.0,
        "rmse_ml": 5.0,
        "bias_ml": 0.0,
        "coverage": coverage,
        "width_ml": width,
        "interval_score": interval_score,
    }


def main() -> int:
    ranking = pd.DataFrame(
        [
            _row("extra_trees_quantile_cqr", "paper_full", 22.0, 0.90, 18.0),
            _row("quantile_gbrt_cqr", "paper_full", 21.0, 0.87, 16.0),  # under-covered
            _row("paper_mc_existing_conformal", "paper_scalar", 20.0, 0.91, 17.0),  # excluded comparator
            _row("paper_log_conformal_global", "paper_scalar", 24.0, 0.90, 21.0),
            _row("paper_log_conformal_global", "paper_full", 24.0, 0.90, 21.0),  # feature-independent dup
        ]
    )
    _assert_ranking_is_ispy2_only(ranking)
    winner = _select_winner(ranking)
    assert winner["method"] == "extra_trees_quantile_cqr", winner["method"]
    assert winner["feature_mode"] == "paper_full"

    leaked = ranking.copy()
    leaked["acrin_interval_score"] = 20.63
    try:
        _assert_ranking_is_ispy2_only(leaked)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected ACRIN column leak to fail")

    leaked_rows = ranking.copy()
    leaked_rows.loc[0, "split"] = "acrin_holdout"
    try:
        _assert_ranking_is_ispy2_only(leaked_rows)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected ACRIN ranking rows to fail")

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
