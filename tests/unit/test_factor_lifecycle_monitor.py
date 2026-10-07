import math

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import lifecycle_monitor as lm


def _origins(count: int) -> pd.Index:
    return pd.Index(pd.bdate_range("2025-01-02", periods=count).strftime("%Y%m%d"))


def test_ew_mean_weights_recent_values_by_half_life():
    series = pd.Series([0.0, 0.0, 1.0], index=_origins(3))
    result = lm.exponentially_weighted_mean(series, half_life=1.0)
    assert result["mean"] == pytest.approx(1.0 / (1.0 + 0.5 + 0.25))
    assert result["count"] == 3
    assert 1.0 < result["effective_n"] < 3.0
    assert lm.exponentially_weighted_mean(pd.Series(dtype=float), half_life=5)["mean"] is None


def test_cohort_means_are_disjoint_and_drop_partial_tail():
    series = pd.Series(np.arange(11, dtype=float), index=_origins(11))
    cohorts = lm.cohort_means(series, cohort_size=5)
    assert cohorts.tolist() == [2.0, 7.0]
    assert list(cohorts.index) == [series.index[4], series.index[9]]


def test_cusum_detects_a_one_sigma_drop_quickly():
    rng = np.random.default_rng(11)
    delays = []
    for _ in range(200):
        values = rng.normal(-1.0, 1.0, 200)
        result = lm.lower_cusum(values, reference=0.0, scale=1.0, slack=0.5, threshold=4.0)
        assert result["ever_alarmed"]
        delays.append(result["first_alarm_index"] + 1)
    assert 6 <= float(np.mean(delays)) <= 13


def test_cusum_in_control_false_alarm_rate_is_low():
    rng = np.random.default_rng(12)
    alarms = 0
    trials = 400
    for _ in range(trials):
        values = rng.normal(0.0, 1.0, 24)
        result = lm.lower_cusum(values, reference=0.0, scale=1.0, slack=0.5, threshold=4.0)
        alarms += int(result["ever_alarmed"])
    assert alarms / trials < 0.15


def test_cusum_rejects_invalid_input():
    with pytest.raises(lm.LifecycleMonitorError):
        lm.lower_cusum([0.1, math.nan], reference=0.0, scale=1.0, slack=0.5, threshold=4.0)
    with pytest.raises(lm.LifecycleMonitorError):
        lm.lower_cusum([0.1], reference=0.0, scale=0.0, slack=0.5, threshold=4.0)


def test_posterior_returns_prior_without_data_and_respects_prior_strength():
    prior_only = lm.normal_posterior([], prior_mean=0.03, prior_sd=0.02, observation_sd_floor=0.01)
    assert prior_only["mean"] == 0.03 and not prior_only["updated_from_prior"]
    data = [-0.02, -0.01, -0.03, -0.02]
    weak = lm.normal_posterior(data, prior_mean=0.05, prior_sd=1.0, observation_sd_floor=0.01)
    strong = lm.normal_posterior(data, prior_mean=0.05, prior_sd=1e-4, observation_sd_floor=0.01)
    assert weak["mean"] == pytest.approx(np.mean(data), abs=1e-3)
    assert weak["p_positive"] < 0.05
    assert strong["mean"] == pytest.approx(0.05, abs=1e-3)
    assert strong["p_positive"] > 0.99


def test_posterior_floor_prevents_overconfidence_from_smooth_short_series():
    result = lm.normal_posterior(
        [0.001, 0.001], prior_mean=0.0, prior_sd=0.05, observation_sd_floor=0.02
    )
    assert result["p_positive"] < 0.6


def test_classification_separates_source_gaps_from_alpha_failure():
    policy = lm.MonitorPolicy(min_cohorts=4)
    state, reasons = lm.classify_monitor_state(
        cohort_count=0,
        available_count=0,
        unavailable_count=3,
        posterior_p_positive=0.5,
        cusum_alarm=False,
        policy=policy,
    )
    assert state == lm.SOURCE_BLOCKED and reasons == ["NO_AVAILABLE_RANK_IC"]
    state, _ = lm.classify_monitor_state(
        cohort_count=2,
        available_count=10,
        unavailable_count=0,
        posterior_p_positive=0.99,
        cusum_alarm=False,
        policy=policy,
    )
    assert state == lm.INSUFFICIENT_EVIDENCE


def test_many_short_horizon_cohorts_do_not_bypass_the_session_minimum():
    state, reasons = lm.classify_monitor_state(
        cohort_count=19,
        available_count=19,
        unavailable_count=0,
        posterior_p_positive=0.99,
        cusum_alarm=False,
        policy=lm.MonitorPolicy(),
    )
    assert state == lm.INSUFFICIENT_EVIDENCE
    assert reasons == ["ORIGIN_SESSIONS_BELOW_60"]


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=_origins(len(values)))


def test_evaluate_healthy_factor():
    rng = np.random.default_rng(3)
    series = _series(rng.normal(0.05, 0.05, 200))
    result = lm.evaluate_ic_series(
        series,
        horizon=5,
        policy=lm.MonitorPolicy(prior_mean=0.05, prior_sd=0.03),
    )
    assert result["cohort_count"] == 40
    assert result["state"] == lm.HEALTHY
    assert result["nonoverlap_t"]["t"] > 3


def test_evaluate_decayed_factor_raises_alarm():
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.normal(0.05, 0.05, 100), rng.normal(-0.04, 0.05, 150)])
    result = lm.evaluate_ic_series(
        _series(values),
        horizon=5,
        policy=lm.MonitorPolicy(prior_mean=0.0, prior_sd=0.05, half_life_sessions=40),
        reference_ic=0.05,
    )
    assert result["cusum"]["alarm"]
    assert result["ew_rank_ic"]["mean"] < 0
    assert result["state"] in {lm.WATCH, lm.ALARM}


def test_evaluate_short_series_is_insufficient_evidence():
    result = lm.evaluate_ic_series(_series(np.full(12, 0.08)), horizon=5, policy=lm.MonitorPolicy())
    assert result["state"] == lm.INSUFFICIENT_EVIDENCE
    assert result["nonoverlap_t"]["t"] == 0.0


def test_residual_rank_ic_removes_a_clone_and_keeps_new_information():
    rng = np.random.default_rng(5)
    size = 500
    pool = pd.Series(rng.normal(size=size))
    new = pd.Series(rng.normal(size=size))
    returns = 0.5 * pool + 0.5 * new + rng.normal(scale=0.5, size=size)
    clone = pool + rng.normal(scale=0.05, size=size)
    assert abs(lm.residual_rank_ic(clone, pool, returns)) < 0.1
    assert lm.residual_rank_ic(new, pool, returns) > 0.3


def test_size_exposure_sign():
    sizes = pd.Series([1e5, 1e6, 1e7, 1e8, 1e9])
    small_cap_signal = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    assert lm.size_exposure(small_cap_signal, sizes) == pytest.approx(-1.0)


def _payload(origin: str, horizon: int, rank_ic: str | None) -> dict:
    diagnostics = {
        "rank_ic": (
            {"state": "AVAILABLE", "value": rank_ic, "reasons": []}
            if rank_ic is not None
            else {"state": "UNAVAILABLE", "value": None, "reasons": ["END_CLOSE_MISSING"]}
        ),
        "securities": {
            "A": {"signal": "1.0", "raw_price_return": "0.01"},
            "B": {"signal": "2.0", "raw_price_return": None},
        },
    }
    return {"horizon": horizon, "origin_session": origin, "diagnostics": diagnostics}


def test_report_counts_source_gaps_and_grants_nothing():
    ref = {"path": "results/factors/outcomes/x/y.json", "sha256": "0" * 64}
    rows = [
        lm.outcome_row_from_payload(_payload("20260901", 1, "0.1"), factor_id="F", outcome_ref=ref),
        lm.outcome_row_from_payload(_payload("20260902", 1, None), factor_id="F", outcome_ref=ref),
    ]
    assert rows[0].returns.to_dict() == {"A": 0.01}
    report = lm.build_monitor_report(rows, policy=lm.MonitorPolicy(), input_refs={})
    horizon = report["factors"]["F"]["horizons"]["1"]
    assert horizon["available_origin_count"] == 1
    assert horizon["unavailable_origin_count"] == 1
    assert horizon["source_gap_reasons"] == {"END_CLOSE_MISSING": 1}
    assert horizon["state"] == lm.INSUFFICIENT_EVIDENCE
    assert report["authority"] == lm.MONITOR_AUTHORITY
    assert report["grants_weight_change"] is False and report["grants_admission"] is False
    assert report["factors"]["F"]["size_exposure"]["state"] == "UNAVAILABLE"


def test_non_finite_available_rank_ic_fails_closed():
    with pytest.raises(lm.LifecycleMonitorError):
        lm.outcome_row_from_payload(_payload("20260901", 1, "nan"), factor_id="F", outcome_ref={})


def test_policy_validation():
    with pytest.raises(lm.LifecycleMonitorError):
        lm.MonitorPolicy(watch_posterior=0.4, alarm_posterior=0.5)
