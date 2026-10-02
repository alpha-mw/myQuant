from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from scipy import stats as scipy_stats

from quant_investor.factors.governance.admission import (
    _cohort_icir,
    _cohort_means,
    _cohort_test,
    _maturity_passed,
    _metric_blockers,
    _preliminary_candidate_metrics,
    _sharpe,
)
from quant_investor.factors.governance.statistics import (
    cohort_overlap_variance_inflation,
    deflated_sharpe_ratio,
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


def test_admission_threshold_equalities_and_block_pair_44_45_boundaries() -> None:
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
        block_pair_count=45,
        positive_block_pair_ratio=0.55,
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
        block_pair_count=45,
        positive_block_pair_ratio=0.55,
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
        block_pair_count=44,
        positive_block_pair_ratio=1.0,
        turnover=Decimal("12"),
    )
    assert forty_four == ["BLOCK_PAIR_STABILITY_INCOMPLETE"]


def test_largest_remainder_is_exact_and_all_zero_fails_closed() -> None:
    weights = largest_remainder_weights({"factor-b": Decimal("1"), "factor-a": Decimal("1")})
    assert weights == {
        "factor-a": "0.500000000000",
        "factor-b": "0.500000000000",
    }
    with pytest.raises(Exception, match="all shrunk IC values are zero"):
        largest_remainder_weights({"factor-a": Decimal("0"), "factor-b": Decimal("0")})


def _worthless_persistent_rank_ic(rng: np.random.Generator) -> pd.Series:
    """Daily RankIC of a persistent signal with no skill against a 30-session label.

    Each session's IC is the sum of the next 30 sessions' independent return
    shocks, so neighbouring sessions share 29 of them.
    """
    shocks = rng.standard_normal(390)
    cumulative = np.concatenate([[0.0], np.cumsum(shocks)])
    return pd.Series(cumulative[31:391] - cumulative[1:361], dtype=float)


def test_cohort_overlap_inflation_matches_the_shared_label_window() -> None:
    # Adjacent 30-session cohort means of a 30-session label share a triangular
    # window whose correlation is 4495/18010; nothing is shared two cohorts apart.
    correlation = 4495 / 18010
    mean_inflation = 1 + 2 * (11 / 12) * correlation
    expected = mean_inflation / (1 - (mean_inflation - 1) / 11)
    assert cohort_overlap_variance_inflation(12) == pytest.approx(expected)
    assert expected == pytest.approx(1.5209, abs=1e-4)

    assert cohort_overlap_variance_inflation(1) == 1.0
    # A one-session label leaves disjoint cohorts with nothing in common.
    assert cohort_overlap_variance_inflation(12, cohort_size=30, horizon_sessions=1) == 1.0


def test_cohort_t_statistic_is_deflated_by_the_overlap_and_keeps_n_minus_one_freedom() -> None:
    values = [0.03, 0.01, 0.04, 0.02, 0.05, 0.00, 0.03, 0.02, 0.04, 0.01, 0.03, 0.02]
    raw, _ = scipy_stats.ttest_1samp(values, 0.0)

    statistic, p_value = _cohort_test(values)

    assert statistic == pytest.approx(float(raw) / np.sqrt(cohort_overlap_variance_inflation(12)))
    assert p_value == pytest.approx(2 * scipy_stats.t.sf(statistic, 11))
    assert _cohort_test([0.03]) == (0.0, 1.0)


def test_deflated_sharpe_is_taken_over_cohort_means_with_the_effective_sample() -> None:
    series = _worthless_persistent_rank_ic(np.random.default_rng(11)) + 2.0
    open_sessions = list(pd.bdate_range("2025-01-02", periods=390).strftime("%Y-%m-%d"))
    series.index = open_sessions[:360]
    cohort_icir = _cohort_icir(series)

    preliminary, _, _ = _preliminary_candidate_metrics(
        [{"configuration_id": "configuration-a", "factor_id": "factor-a", "family": "liquidity"}],
        {"configuration-a": series},
        open_sessions,
        {"configuration-a": cohort_icir},
        trial_sharpe_std=0.0,
        effective_trials=1,
        trial_icir_complete=True,
    )

    assert cohort_icir == pytest.approx(_sharpe(pd.Series(_cohort_means(series))))
    assert preliminary["configuration-a"]["dsr"] == pytest.approx(
        deflated_sharpe_ratio(
            observed_sharpe=cohort_icir,
            trial_sharpe_std=0.0,
            trial_count=1,
            sample_size=12 / cohort_overlap_variance_inflation(12),
            skew=0.0,
            kurtosis=3.0,
        )
    )


def test_worthless_persistent_signal_passes_the_gates_at_about_the_nominal_rate() -> None:
    rng = np.random.default_rng(20261002)
    trials = 2000
    t_passes = dsr_passes = daily_dsr_passes = 0
    for _ in range(trials):
        series = _worthless_persistent_rank_ic(rng)
        cohorts = _cohort_means(series)
        statistic, _ = _cohort_test(cohorts)
        t_passes += statistic > 3.0
        dsr_passes += (
            deflated_sharpe_ratio(
                observed_sharpe=_cohort_icir(series),
                trial_sharpe_std=0.0,
                trial_count=1,
                sample_size=len(cohorts) / cohort_overlap_variance_inflation(len(cohorts)),
                skew=0.0,
                kurtosis=3.0,
            )
            >= 0.95
        )
        # The retired computation: ICIR and sample size of the overlapping daily series.
        daily_dsr_passes += (
            deflated_sharpe_ratio(
                observed_sharpe=_sharpe(series),
                trial_sharpe_std=0.0,
                trial_count=1,
                sample_size=360,
                skew=0.0,
                kurtosis=3.0,
            )
            >= 0.95
        )

    # One-sided t > 3 with 11 degrees of freedom is 0.6%; a single-trial DSR of
    # 0.95 is 5%.  The daily-series DSR passed a worthless factor far more often.
    assert t_passes / trials < 0.015
    assert dsr_passes / trials < 0.07
    assert daily_dsr_passes / trials > 0.25
