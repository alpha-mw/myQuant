from __future__ import annotations

import pandas as pd

from quant_investor.market.dag.packets import _build_symbol_market_state


def test_symbol_market_state_prepares_frame_once(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=80, freq="B"),
            "close": [10.0 + idx * 0.1 for idx in range(80)],
            "vol": [1_000_000 + idx * 1_000 for idx in range(80)],
        }
    )
    copy_calls = {"count": 0}
    original_copy = pd.DataFrame.copy

    def _counting_copy(self, *args, **kwargs):
        copy_calls["count"] += 1
        return original_copy(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", _counting_copy)

    state = _build_symbol_market_state(
        frame,
        trend_windows=(20, 60),
        volume_spike_threshold=1.35,
        breakout_distance_pct=0.06,
    )

    assert state["rows"] == 80
    assert state["latest_close"] == 17.9
    assert state["trend_windows"] == [20, 60]
    assert copy_calls["count"] == 0


def test_symbol_market_state_computes_latest_trend_without_full_rolling(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=80, freq="B"),
            "close": [10.0 + idx * 0.1 for idx in range(80)],
            "vol": [1_000_000 + idx * 1_000 for idx in range(80)],
        }
    )
    rolling_calls = {"count": 0}
    original_rolling = pd.Series.rolling

    def _counting_rolling(self, *args, **kwargs):
        rolling_calls["count"] += 1
        return original_rolling(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "rolling", _counting_rolling)

    state = _build_symbol_market_state(
        frame,
        trend_windows=(20, 60),
        volume_spike_threshold=1.35,
        breakout_distance_pct=0.06,
    )

    assert state["trend_stability"] > 0.99
    assert rolling_calls["count"] == 0


def test_symbol_market_state_computes_drawdown_without_pandas_cummax(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=140, freq="B"),
            "close": [
                10.0 + idx * 0.1 if idx < 80 else 18.0 - (idx - 80) * 0.05
                for idx in range(140)
            ],
            "vol": [1_000_000 + idx * 1_000 for idx in range(140)],
        }
    )
    cummax_calls = {"count": 0}
    original_cummax = pd.Series.cummax

    def _counting_cummax(self, *args, **kwargs):
        cummax_calls["count"] += 1
        return original_cummax(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "cummax", _counting_cummax)

    state = _build_symbol_market_state(
        frame,
        trend_windows=(20, 60, 120),
        volume_spike_threshold=1.35,
        breakout_distance_pct=0.06,
    )

    assert state["max_drawdown_pct"] > 0.0
    assert cummax_calls["count"] == 0


def test_symbol_market_state_computes_returns_without_pandas_pct_change(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=80, freq="B"),
            "close": [10.0 + idx * 0.1 for idx in range(80)],
            "vol": [1_000_000 + idx * 1_000 for idx in range(80)],
        }
    )
    pct_change_calls = {"count": 0}
    original_pct_change = pd.Series.pct_change

    def _counting_pct_change(self, *args, **kwargs):
        pct_change_calls["count"] += 1
        return original_pct_change(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "pct_change", _counting_pct_change)

    state = _build_symbol_market_state(
        frame,
        trend_windows=(20, 60),
        volume_spike_threshold=1.35,
        breakout_distance_pct=0.06,
    )

    assert state["average_return"] > 0.0
    assert state["volatility"] > 0.0
    assert pct_change_calls["count"] == 0


def test_symbol_market_state_reuses_numeric_close_for_breakout(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=80, freq="B"),
            "close": [10.0 + idx * 0.1 for idx in range(80)],
            "vol": [1_000_000 + idx * 1_000 for idx in range(80)],
        }
    )
    to_numeric_calls = {"count": 0}
    original_to_numeric = pd.to_numeric

    def _counting_to_numeric(*args, **kwargs):
        to_numeric_calls["count"] += 1
        return original_to_numeric(*args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", _counting_to_numeric)

    state = _build_symbol_market_state(
        frame,
        trend_windows=(20, 60),
        volume_spike_threshold=1.35,
        breakout_distance_pct=0.06,
    )

    assert state["breakout_readiness"] == 1.0
    assert to_numeric_calls["count"] <= 2


def test_symbol_market_state_skips_numeric_coercion_for_numeric_frame(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=80, freq="B"),
            "close": [10.0 + idx * 0.1 for idx in range(80)],
            "vol": [1_000_000 + idx * 1_000 for idx in range(80)],
        }
    )
    to_numeric_calls = {"count": 0}
    original_to_numeric = pd.to_numeric

    def _counting_to_numeric(*args, **kwargs):
        to_numeric_calls["count"] += 1
        return original_to_numeric(*args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", _counting_to_numeric)

    state = _build_symbol_market_state(
        frame,
        trend_windows=(20, 60),
        volume_spike_threshold=1.35,
        breakout_distance_pct=0.06,
    )

    assert state["latest_close"] == 17.9
    assert state["volume_spike_ratio"] > 0.0
    assert to_numeric_calls["count"] == 0
