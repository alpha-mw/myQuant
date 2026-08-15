from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.governance.admission import (
    _cohort_means,
    _maturity_passed,
    _metric_blockers,
)
from quant_investor.factors.governance.statistics import (
    probability_of_backtest_overfitting,
    redundancy_clusters,
)
from quant_investor.factors.governance.weights import largest_remainder_weights


def test_pbo_requires_complete_finite_10_by_n_and_runs_all_252_splits() -> None:
    complete = pd.DataFrame(
        {
            "configuration-a": np.linspace(0.01, 0.10, 10),
            "configuration-b": np.linspace(0.02, -0.02, 10),
        }
    )
    result = probability_of_backtest_overfitting(complete)
    assert result["complete"] is True
    assert result["block_count"] == 10
    assert result["config_count"] == 2
    assert result["split_count"] == 252

    missing = complete.copy()
    missing.loc[3, "configuration-b"] = np.nan
    failed = probability_of_backtest_overfitting(missing)
    assert failed["complete"] is False
    assert failed["pbo"] == 1.0
    assert failed["split_count"] == 0
    assert failed["config_count"] == 2

    one_configuration = complete[["configuration-a"]]
    one = probability_of_backtest_overfitting(one_configuration)
    assert one["complete"] is False
    assert one["pbo"] == 1.0
    assert one["split_count"] == 0


def test_redundancy_union_uses_slot_or_correlation_and_is_transitive() -> None:
    index = pd.date_range("2025-01-01", periods=12, freq="D")
    series = {
        "configuration-a": pd.Series([1.0, 0.0, -1.0, 0.5, -0.5], index=index[:5]),
        "configuration-b": pd.Series(np.arange(12, dtype=float), index=index),
        "configuration-c": pd.Series(np.arange(12, dtype=float) * -2.0, index=index),
    }
    clusters = redundancy_clusters(
        series,
        normalized_slots={
            "configuration-a": "liquidity:amount",
            "configuration-b": "liquidity:amount",
            "configuration-c": "momentum:return",
        },
    )
    assert clusters == (("configuration-a", "configuration-b", "configuration-c"),)

    short = redundancy_clusters(
        {
            "configuration-a": pd.Series(range(11), dtype=float),
            "configuration-b": pd.Series(range(11), dtype=float),
        },
        normalized_slots={
            "configuration-a": "family-a:primitive",
            "configuration-b": "family-b:primitive",
        },
    )
    assert short == (("configuration-a",), ("configuration-b",))


@pytest.mark.parametrize(
    ("daily", "months", "cohorts", "expected"),
    [
        (299, 12, 8, False),
        (300, 11, 8, False),
        (300, 12, 7, False),
        (300, 12, 8, True),
    ],
)
def test_maturity_is_exact_conjunctive_300_12_8(
    daily: int, months: int, cohorts: int, expected: bool
) -> None:
    assert (
        _maturity_passed(
            valid_daily_sessions=daily,
            closed_month_ends=months,
            disjoint_cohorts=cohorts,
        )
        is expected
    )


def test_cohorts_are_fixed_canonical_session_ordinals_and_never_stitched() -> None:
    index = pd.Index([f"session-{value:03d}" for value in range(360)])
    values = pd.Series(np.arange(360, dtype=float), index=index)
    assert len(_cohort_means(values)) == 12

    # One missing observation invalidates its exact [0, 30) ordinal cohort.  The
    # next valid value cannot slide backward to create a stitched replacement.
    values.iloc[29] = np.nan
    means = _cohort_means(values)
    assert len(means) == 11
    assert means[0] == pytest.approx(np.mean(np.arange(30, 60, dtype=float)))

    # Exactly eight disjoint canonical cohorts survive; this is the maturity
    # unit, independent of calendar/business-day deltas.
    eight = pd.Series(np.nan, index=index, dtype=float)
    eight.iloc[: 8 * 30] = 0.01
    assert len(_cohort_means(eight)) == 8


def test_admission_threshold_equalities_and_cpcv_44_45_boundaries() -> None:
    exact = _metric_blockers(
        valid_daily_sessions=300,
        closed_month_ends=12,
        disjoint_cohorts=8,
        t_statistic=3.000000000001,
        dsr=0.95,
        pbo_complete=True,
        pbo_split_count=252,
        pbo=0.50,
        bh_q_value=0.10,
        cpcv_path_count=45,
        positive_path_ratio=0.55,
        turnover=Decimal("12"),
    )
    assert exact == []

    t_equal = _metric_blockers(
        valid_daily_sessions=300,
        closed_month_ends=12,
        disjoint_cohorts=8,
        t_statistic=3.0,
        dsr=0.95,
        pbo_complete=True,
        pbo_split_count=252,
        pbo=0.50,
        bh_q_value=0.10,
        cpcv_path_count=45,
        positive_path_ratio=0.55,
        turnover=Decimal("12"),
    )
    assert t_equal == ["T_STATISTIC_FAILED"]

    forty_four = _metric_blockers(
        valid_daily_sessions=300,
        closed_month_ends=12,
        disjoint_cohorts=8,
        t_statistic=3.1,
        dsr=0.95,
        pbo_complete=True,
        pbo_split_count=252,
        pbo=0.50,
        bh_q_value=0.10,
        cpcv_path_count=44,
        positive_path_ratio=1.0,
        turnover=Decimal("12"),
    )
    assert forty_four == ["CPCV_INCOMPLETE"]


def test_largest_remainder_is_exact_and_all_zero_fails_closed() -> None:
    weights = largest_remainder_weights({"factor-b": Decimal("1"), "factor-a": Decimal("1")})
    assert weights == {
        "factor-a": "0.500000000000",
        "factor-b": "0.500000000000",
    }
    with pytest.raises(Exception, match="all shrunk IC values are zero"):
        largest_remainder_weights({"factor-a": Decimal("0"), "factor-b": Decimal("0")})
