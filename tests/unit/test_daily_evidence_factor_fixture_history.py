"""Extending audit prehistory must not revise already generated price observations."""

import pandas as pd
from _native_daily_factor_fixture import NativeFactorInputs


def test_extra_history_preserves_existing_synthetic_observations(tmp_path):
    original = NativeFactorInputs(tmp_path / "original", count=10).day(1)
    extended = NativeFactorInputs(tmp_path / "extended", count=10).day(1, extra_history=9)
    before = pd.read_parquet(original["market_history_path"])
    after = pd.read_parquet(extended["market_history_path"])
    common = after[after["trade_date"].isin(before["trade_date"].unique())].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, common)
    assert len(after["trade_date"].unique()) == 101
    assert original["as_of"] == extended["as_of"] == "20260825"


def test_five_day_dag_window_respects_actual_research_policy(tmp_path):
    from quant_investor.intelligence.storage import approved_theme_policy_v2

    policy = approved_theme_policy_v2()["payload"]
    fixture = NativeFactorInputs(tmp_path, count=10, extra_future_sessions=3)
    eligible = [
        day.strftime("%Y%m%d")
        for day in fixture.sessions[90:]
        if day.strftime("%Y%m%d") >= policy["effective_signal_date"]
        and day.isoformat() + "T07:00:00Z" >= policy["effective_from"]
    ]
    assert eligible == ["20260827", "20260828", "20260831", "20260901", "20260902"]
    assert fixture.sessions[90].strftime("%Y%m%d") == "20260824"
