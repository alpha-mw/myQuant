from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.governance.statistics import (
    HARVEY_LIU_ZHU_T_HURDLE,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    infer_cohort_size,
    nonoverlap_t_statistic,
    probability_of_backtest_overfitting,
)


class TestExpectedMaxSharpe:
    def test_more_trials_raise_the_bar(self) -> None:
        few = expected_max_sharpe_under_null(trial_sharpe_std=0.5, trial_count=10)
        many = expected_max_sharpe_under_null(
            trial_sharpe_std=0.5, trial_count=10_000
        )

        assert many > few > 0.0

    def test_no_dispersion_means_no_selection_bias(self) -> None:
        assert expected_max_sharpe_under_null(
            trial_sharpe_std=0.0, trial_count=500
        ) == pytest.approx(0.0)

    def test_a_single_trial_has_no_selection_bias(self) -> None:
        assert expected_max_sharpe_under_null(
            trial_sharpe_std=0.5, trial_count=1
        ) == pytest.approx(0.0)


class TestDeflatedSharpe:
    def _dsr(self, **overrides: float) -> float:
        kwargs = {
            "observed_sharpe": 0.60,
            "trial_sharpe_std": 0.30,
            "trial_count": 10,
            "sample_size": 60,
            "skew": 0.0,
            "kurtosis": 3.0,
        }
        kwargs.update(overrides)
        return deflated_sharpe_ratio(**kwargs)  # type: ignore[arg-type]

    def test_a_strong_result_from_few_trials_survives(self) -> None:
        assert self._dsr(trial_count=2) > 0.95

    def test_the_same_result_from_many_trials_does_not(self) -> None:
        assert self._dsr(trial_count=10_000) < 0.95

    def test_deflation_is_monotone_in_trial_count(self) -> None:
        values = [self._dsr(trial_count=n) for n in (2, 10, 100, 1_000, 10_000)]

        assert values == sorted(values, reverse=True)

    def test_negative_skew_and_fat_tails_deflate_further(self) -> None:
        normal = self._dsr()
        ugly = self._dsr(skew=-1.5, kurtosis=9.0)

        assert ugly < normal

    def test_a_longer_sample_supports_a_higher_deflated_sharpe(self) -> None:
        assert self._dsr(sample_size=400) > self._dsr(sample_size=40)

    def test_a_sharpe_below_the_selection_bar_fails(self) -> None:
        assert self._dsr(observed_sharpe=0.05, trial_count=1_000) < 0.5

    def test_degenerate_inputs_fail_closed(self) -> None:
        assert deflated_sharpe_ratio(
            observed_sharpe=1.0,
            trial_sharpe_std=0.3,
            trial_count=10,
            sample_size=1,
            skew=0.0,
            kurtosis=3.0,
        ) == 0.0


class TestProbabilityOfBacktestOverfitting:
    def test_a_genuinely_dominant_config_is_not_overfit(self) -> None:
        rng = np.random.default_rng(7)
        blocks, configs = 10, 12
        values = rng.normal(0.0, 0.01, size=(blocks, configs))
        # Config 0 beats everything in every block, by a lot.
        values[:, 0] += 0.20
        frame = pd.DataFrame(values, columns=[f"c{i}" for i in range(configs)])

        evidence = probability_of_backtest_overfitting(frame)

        assert evidence["pbo"] < 0.1
        assert evidence["split_count"] == 252

    def test_pure_noise_is_a_coin_flip_on_average(self) -> None:
        """PBO centres on 0.5 under the null, but only across matrices.

        Every split reuses the same matrix, so a config that happens to draw a
        high mean over all blocks wins in sample *and* ranks high out of sample
        across many splits.  A single noise draw therefore ranges roughly 0.10
        to 0.89 (sd about 0.19); only the average over draws is 0.5.  Read a
        single run's PBO as a diagnostic with real sampling error, not as a
        precise threshold.
        """

        values = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            frame = pd.DataFrame(
                rng.normal(size=(10, 20)),
                columns=[f"c{i}" for i in range(20)],
            )
            values.append(probability_of_backtest_overfitting(frame)["pbo"])

        assert 0.4 < float(np.mean(values)) < 0.6

    def test_an_inverted_leader_is_maximally_overfit(self) -> None:
        # Whichever config wins in one half is built to lose in the other.
        blocks, configs = 10, 10
        values = np.zeros((blocks, configs))
        for config in range(configs):
            values[: blocks // 2, config] = config
            values[blocks // 2 :, config] = -config
        frame = pd.DataFrame(values, columns=[f"c{i}" for i in range(configs)])

        evidence = probability_of_backtest_overfitting(frame)

        assert evidence["pbo"] > 0.5

    def test_too_few_configs_or_blocks_fails_closed(self) -> None:
        frame = pd.DataFrame({"c0": [1.0, 2.0]})

        evidence = probability_of_backtest_overfitting(frame)

        assert evidence["pbo"] == 1.0
        assert evidence["split_count"] == 0


class TestInferCohortSize:
    """The cohort size is in observations; the horizon is in sessions.

    A daily IC series needs 30 observations to span a 30-session horizon; a
    month-end series needs 2.  Hard-coding 30 silently failed closed on every
    monthly series, which is what the whole production candidate set uses.
    """

    def test_a_daily_series_needs_about_a_horizon_of_observations(self) -> None:
        index = pd.bdate_range("2021-01-04", periods=900)

        assert 25 <= infer_cohort_size(index, horizon_sessions=30) <= 35

    def test_a_month_end_series_needs_only_two(self) -> None:
        index = pd.date_range("2021-01-31", periods=60, freq="ME")

        assert infer_cohort_size(index, horizon_sessions=30) == 2

    def test_a_biweekly_series_sits_in_between(self) -> None:
        index = pd.bdate_range("2021-01-04", periods=600)[::10]

        size = infer_cohort_size(index, horizon_sessions=30)
        assert 2 <= size <= 4

    def test_an_unusable_index_falls_back_to_a_single_observation(self) -> None:
        assert infer_cohort_size(pd.DatetimeIndex([]), horizon_sessions=30) == 1


class TestNonoverlapTStatistic:
    def _series(self, values: np.ndarray) -> pd.Series:
        dates = pd.bdate_range("2021-01-04", periods=len(values))
        return pd.Series(values, index=dates)

    def _monthly(self, values: np.ndarray) -> pd.Series:
        dates = pd.date_range("2021-01-31", periods=len(values), freq="ME")
        return pd.Series(values, index=dates)

    def test_a_monthly_series_is_not_failed_closed(self) -> None:
        """A ~58-observation month-end series is what the miner actually has."""

        rng = np.random.default_rng(8)
        series = self._monthly(rng.normal(0.05, 0.03, size=58))

        t_stat, p_value, sample = nonoverlap_t_statistic(series)

        assert sample == 29
        assert t_stat > HARVEY_LIU_ZHU_T_HURDLE
        assert p_value < 0.01

    def test_a_weak_monthly_series_does_not_clear_the_hurdle(self) -> None:
        rng = np.random.default_rng(12)
        series = self._monthly(rng.normal(0.004, 0.05, size=58))

        t_stat, _p, _n = nonoverlap_t_statistic(series)

        assert abs(t_stat) < HARVEY_LIU_ZHU_T_HURDLE

    def test_a_consistent_edge_clears_the_hurdle(self) -> None:
        rng = np.random.default_rng(3)
        series = self._series(rng.normal(0.06, 0.02, size=900))

        t_stat, p_value, sample = nonoverlap_t_statistic(series)

        assert t_stat > HARVEY_LIU_ZHU_T_HURDLE
        assert p_value < 0.01
        assert sample == 30

    def test_noise_does_not_clear_the_hurdle(self) -> None:
        rng = np.random.default_rng(4)
        series = self._series(rng.normal(0.0, 0.05, size=900))

        t_stat, _p, _n = nonoverlap_t_statistic(series)

        assert abs(t_stat) < HARVEY_LIU_ZHU_T_HURDLE

    def test_cohorts_do_not_share_a_forward_window(self) -> None:
        # 900 sessions at a cohort size of 30 gives exactly 30 disjoint cohorts.
        rng = np.random.default_rng(5)
        series = self._series(rng.normal(0.02, 0.02, size=900))

        _t, _p, sample = nonoverlap_t_statistic(series, cohort_size=30)

        assert sample == 30

    def test_a_series_too_short_to_split_fails_closed(self) -> None:
        t_stat, p_value, sample = nonoverlap_t_statistic(
            self._series(np.array([0.1, 0.2]))
        )

        assert t_stat == 0.0
        assert p_value == 1.0
        assert sample <= 1
