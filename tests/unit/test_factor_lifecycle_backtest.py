import math

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import lifecycle_backtest as lb


def _months(count: int) -> pd.DatetimeIndex:
    return pd.date_range("2012-01-31", periods=count, freq="ME")


def test_forward_label_enters_at_next_close():
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 15.0, 16.0]})
    label = lb.forward_label(close, horizon=2)
    assert label["A"].iloc[0] == pytest.approx(15.0 / 11.0 - 1.0)
    assert label["A"].iloc[2:].isna().all()


def test_month_end_origins_pick_last_session_per_month():
    dates = pd.to_datetime(["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-27"])
    assert lb.month_end_origins(dates) == list(pd.to_datetime(["2026-01-30", "2026-02-27"]))


def test_cross_sectional_rank_ic_requires_a_real_cross_section():
    signal = pd.Series(range(10), dtype=float)
    assert math.isnan(lb.cross_sectional_rank_ic(signal, signal))
    wide = pd.Series(range(50), dtype=float)
    assert lb.cross_sectional_rank_ic(wide, wide) == pytest.approx(1.0)


def test_ew_estimate_matches_expanding_mean_at_infinite_half_life():
    values = np.array([0.01, 0.03, np.nan, 0.05])
    mean, sd, effective = lb.ew_estimate(values, math.inf)
    assert mean == pytest.approx(0.03)
    assert sd == pytest.approx(np.std([0.01, 0.03, 0.05], ddof=1))
    assert effective == pytest.approx(3.0)


def test_half_life_selection_prefers_short_memory_for_regime_switching_ic():
    rng = np.random.default_rng(1)
    regimes = np.repeat([0.08, -0.04, 0.08, -0.04, 0.08, -0.04], 24)
    ic = pd.DataFrame(
        {f"f{j}": regimes + rng.normal(0, 0.02, regimes.size) for j in range(4)},
        index=_months(regimes.size),
    )
    result = lb.select_half_life(ic, half_lives=(2.0, 6.0, math.inf), min_history=6)
    assert result["best_half_life_periods"] in (2.0, 6.0)


def test_half_life_selection_prefers_long_memory_for_stable_ic():
    rng = np.random.default_rng(2)
    ic = pd.DataFrame(
        {f"f{j}": 0.04 + rng.normal(0, 0.05, 120) for j in range(4)}, index=_months(120)
    )
    result = lb.select_half_life(ic, half_lives=(1.0, 3.0, math.inf), min_history=12)
    assert result["best_half_life_periods"] == "inf"


def test_posterior_uses_prior_until_two_observations():
    mean, p, count = lb.posterior_p_positive(
        np.array([0.1]), half_life=6.0, prior_mean=0.0, prior_sd=0.05
    )
    assert (mean, count) == (0.0, 1) and p == pytest.approx(0.5)


def test_rule_validation_and_grid_size():
    with pytest.raises(lb.LifecycleBacktestError):
        lb.LifecycleRule(watch_p=0.4, retire_p=0.5)
    assert len(lb.rule_grid()) == 64


def test_retired_factor_reenters_with_a_fresh_evidence_window():
    rng = np.random.default_rng(6)
    values = np.concatenate(
        [
            0.05 + rng.normal(0, 0.02, 30),
            -0.05 + rng.normal(0, 0.02, 24),
            0.06 + rng.normal(0, 0.02, 60),
        ]
    )
    ic = pd.DataFrame({"cyclical": values}, index=_months(values.size))
    with_reentry = lb.simulate_lifecycle(ic, lb.LifecycleRule(reentry_cooldown=12))
    without = lb.simulate_lifecycle(ic, lb.LifecycleRule(reentry_cooldown=None))
    assert with_reentry["retirements"] and with_reentry["reentries"]
    assert with_reentry["final_states"]["cyclical"] in {lb.PROBATION, lb.ACTIVE}
    assert without["final_states"]["cyclical"] == lb.RETIRED
    assert with_reentry["portfolio_ic"].iloc[-12:].notna().all()
    assert without["portfolio_ic"].iloc[-12:].isna().all()


def test_cap_weights_redistributes_and_never_breaches_cap():
    capped = lb.cap_weights(np.array([0.7, 0.2, 0.1]), 0.4)
    assert capped.max() <= 0.4 + 1e-12
    assert capped.sum() == pytest.approx(1.0)
    assert capped[1] / capped[2] == pytest.approx(2.0)
    lonely = lb.cap_weights(np.array([1.0, 0.0]), 0.4)
    assert lonely.tolist() == pytest.approx([0.4, 0.0])
    assert lb.cap_weights(np.array([0.7, 0.3]), None).tolist() == [0.7, 0.3]


def test_fair_baseline_uses_only_past_sign():
    index = _months(40)
    ic = pd.DataFrame({"good": [0.05] * 40, "bad": [-0.05] * 40}, index=index)
    fair = lb.expanding_positive_weighted(ic, warmup_periods=6)
    naive = lb.static_equal_weight(ic, warmup_periods=6)
    assert fair.iloc[6:].tolist() == pytest.approx([0.05] * 34)
    assert naive.iloc[6:].tolist() == pytest.approx([0.0] * 34)


def test_lifecycle_retires_a_decayed_factor_and_keeps_a_live_one():
    rng = np.random.default_rng(3)
    periods = 96
    live = 0.05 + rng.normal(0, 0.03, periods)
    decayed = np.concatenate([0.05 + rng.normal(0, 0.03, 36), -0.04 + rng.normal(0, 0.03, 60)])
    ic = pd.DataFrame({"live": live, "decayed": decayed}, index=_months(periods))
    run = lb.simulate_lifecycle(ic, lb.LifecycleRule(half_life_periods=6.0, reentry_cooldown=None))
    assert run["final_states"]["live"] == lb.ACTIVE
    assert run["final_states"]["decayed"] == lb.RETIRED
    assert run["retirements"][0]["factor"] == "decayed"
    reentered = lb.simulate_lifecycle(ic, lb.LifecycleRule(half_life_periods=6.0))
    assert reentered["final_states"]["decayed"] == lb.PREREGISTERED
    assert reentered["portfolio_ic"].iloc[-24:].tolist() == pytest.approx(live[-24:].tolist())
    tail = run["portfolio_ic"].iloc[-24:]
    assert tail.mean() == pytest.approx(live[-24:].mean(), abs=0.01)
    static = lb.static_equal_weight(ic, warmup_periods=12).iloc[-24:]
    assert tail.mean() > static.mean()


def test_lifecycle_decisions_do_not_see_the_current_period():
    ic = pd.DataFrame({"a": [0.05] * 20 + [-1.0]}, index=_months(21))
    run = lb.simulate_lifecycle(ic, lb.LifecycleRule(probation_periods=6, rebalance_every=1))
    assert run["portfolio_ic"].iloc[-1] == pytest.approx(-1.0)


def test_rebalance_step_is_capped():
    ic = pd.DataFrame({"a": [0.05] * 30, "b": [0.05] * 30}, index=_months(30))
    run = lb.simulate_lifecycle(ic, lb.LifecycleRule(max_step=0.25, rebalance_every=1))
    assert run["turnover"] <= 1.0 + 1e-9


def test_summary_and_deflation():
    rng = np.random.default_rng(4)
    summaries = [
        lb.summarize_ic_series(pd.Series(rng.normal(0.02 + 0.002 * i, 0.05, 120)))
        for i in range(10)
    ]
    deflated = lb.deflated_best_rule(summaries)
    assert deflated["trial_count"] == 10
    assert 0.0 <= deflated["dsr"] <= 1.0
    with pytest.raises(lb.LifecycleBacktestError):
        lb.deflated_best_rule(summaries[:1])


def test_regime_attribution_reports_correlations_and_event_windows():
    index = _months(36)
    regime = pd.Series(np.linspace(-1, 1, 36), index=index)
    ic = pd.DataFrame({"a": regime * 0.1, "b": -regime * 0.1}, index=index)
    result = lb.regime_attribution(
        ic, pd.DataFrame({"size": regime}), events={"early": ("2012-01-01", "2012-12-31")}
    )
    assert result["regime_correlations"]["a"]["size"] == pytest.approx(1.0)
    assert result["regime_correlations"]["b"]["size"] == pytest.approx(-1.0)
    assert result["event_window_mean_ic"]["early"]["a"] < 0
