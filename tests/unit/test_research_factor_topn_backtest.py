"""Research top-N backtest: signal parity with the governed code and execution rules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.governance.bootstrap import (
    BLEND_W80,
    CANONICAL_PARQUET,
    LOW_DOLLAR_VOLUME,
    compute_bootstrap_signals,
)

_SPEC = importlib.util.spec_from_file_location(
    "research_factor_topn_backtest",
    Path(__file__).resolve().parents[2] / "scripts/research_factor_topn_backtest.py",
)
backtest = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = backtest
_SPEC.loader.exec_module(backtest)

NO_COSTS = backtest.Frictions(
    "none", fees=False, half_spread_ticks=0.0, impact_coefficient_bps=0.0, participation_cap=0.0
)


def _panel() -> pd.DataFrame:
    """40 symbols over 130 sessions, with suspensions and one short history."""
    rng = np.random.default_rng(5)
    sessions = list(pd.bdate_range("2025-01-02", periods=130).strftime("%Y%m%d"))
    rows = []
    for number in range(40):
        symbol = f"{600000 + number}.SH"
        history = sessions[70:] if number == 0 else sessions
        close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(history))))
        for day, price in zip(history, close):
            if number % 7 == 3 and rng.random() < 0.15:
                continue  # suspended: the symbol simply has no bar that day
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "adj_close": price,
                    "vol": float(rng.uniform(1e3, 1e5)),
                    "amount": float(rng.uniform(1e3, 1e6)),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["ts_code", "trade_date"], kind="mergesort")
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("session", [-1, -9])
def test_vectorised_signals_match_the_governed_signal_code(session: int) -> None:
    bars = _panel()
    components = backtest.compute_components(bars)
    day = sorted(bars["trade_date"].unique())[session]
    cohort = bars.loc[bars["trade_date"] == day, "ts_code"]
    mine = components.loc[cohort.index].set_index(cohort.to_numpy())
    history = bars[bars["ts_code"].isin(set(cohort)) & (bars["trade_date"] <= day)]
    governed = compute_bootstrap_signals(
        {
            symbol: frame[["trade_date", "amount", "adj_close", "vol"]]
            for symbol, frame in history.groupby("ts_code", sort=False)
        },
        source_format=CANONICAL_PARQUET,
    )

    w80 = backtest.blend_w80(mine)
    low = governed[LOW_DOLLAR_VOLUME]
    assert mine["low_dollar"].isna().equals(low.reindex(mine.index).isna())
    assert w80.isna().equals(governed[BLEND_W80].reindex(w80.index).isna())
    assert w80.isna().any()  # the short history has no 90-session momentum
    np.testing.assert_allclose(
        mine["low_dollar"], low.reindex(mine.index), rtol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        w80, governed[BLEND_W80].reindex(w80.index), rtol=1e-12, equal_nan=True
    )
    assert backtest.rank_pool(mine["low_dollar"], w80, top_n=10) == backtest.rank_pool(
        low, governed[BLEND_W80], top_n=10
    )


def test_pool_averages_rank_percentiles_and_breaks_ties_by_symbol() -> None:
    low = pd.Series({"B": 2.0, "A": 2.0, "C": 1.0, "D": 3.0, "E": np.nan})
    w80 = pd.Series({"B": 0.5, "A": 0.5, "C": 0.9, "D": 0.1, "E": 0.9})

    # A and B tie on both factors; D is best on LOW and worst on W80, C the reverse.
    assert backtest.rank_pool(low, w80, top_n=4) == ["A", "B", "C", "D"]
    assert backtest.rank_pool(low, w80, top_n=2) == ["A", "B"]
    assert backtest.rank_pool(low.iloc[:1], w80.iloc[:1], top_n=2) == []


def test_limit_bands_follow_the_board_and_the_chinext_reform() -> None:
    assert backtest.limit_band("600000.SH", "20240102") == 0.10
    assert backtest.limit_band("300750.SZ", "20200821") == 0.10
    assert backtest.limit_band("300750.SZ", "20200824") == 0.20
    assert backtest.limit_band("688981.SH", "20200102") == 0.20
    assert backtest.limit_band("830799.BJ", "20240102") == 0.30


def test_orders_are_blocked_only_on_the_wrong_side_of_a_limit_open() -> None:
    # 10% board, previous close 9.87: limits are 10.86 and 8.88.
    assert backtest.buy_blocked(10.86, 10.86, 9.87, 0.10)
    assert not backtest.buy_blocked(10.85, 10.86, 9.87, 0.10)
    assert backtest.sell_blocked(8.88, 8.88, 9.87, 0.10)
    assert not backtest.sell_blocked(8.89, 8.88, 9.87, 0.10)
    # An ST-style open pinned on the 5% limit all day blocks; trading through it does not.
    assert backtest.buy_blocked(10.50, 10.50, 10.00, 0.10)
    assert not backtest.buy_blocked(10.50, 10.80, 10.00, 0.10)
    assert backtest.sell_blocked(9.50, 9.50, 10.00, 0.10)
    assert not backtest.sell_blocked(9.50, 9.20, 10.00, 0.10)


def test_fees_charge_the_minimum_commission_and_the_stamp_duty_of_the_day() -> None:
    assert backtest.trade_fees(10_000, side="BUY", trade_date="20240102") == pytest.approx(5.10)
    assert backtest.trade_fees(100_000, side="BUY", trade_date="20240102") == pytest.approx(11.0)
    assert backtest.trade_fees(10_000, side="SELL", trade_date="20230825") == pytest.approx(15.10)
    assert backtest.trade_fees(10_000, side="SELL", trade_date="20230828") == pytest.approx(10.10)


def _market(open_: list[list[float]], close: list[list[float]], **overrides) -> "backtest.Market":
    opens = np.array(open_, dtype=float)
    closes = np.array(close, dtype=float)
    sessions, names = opens.shape
    previous = np.vstack([closes[:1], closes[:-1]])
    values = {
        "dates": [f"2024010{day + 2}" for day in range(sessions)],
        "symbols": [f"60000{name}.SH" for name in range(names)],
        "open": opens,
        "high": np.fmax(opens, closes),
        "low": np.fmin(opens, closes),
        "pre_close": previous,
        "adj_factor": np.ones_like(opens),
        "adj_close_filled": pd.DataFrame(closes).ffill().to_numpy(),
        "adv_cny": np.full_like(opens, 1e9),
        "band": np.full_like(opens, 0.10),
        "lot": np.full(names, 100),
        "delist_index": np.full(names, sessions + 1),
    }
    return backtest.Market(**{**values, **overrides})


def test_entry_buys_whole_lots_at_the_open_and_marks_at_the_close() -> None:
    market = _market([[33.0], [34.0]], [[33.5], [35.0]])

    result = backtest.simulate(market, {0: [0]}, capital=10_000, top_n=1, frictions=NO_COSTS)

    # 10,000 / 33 = 303 shares, so three lots: 9,900 invested and 100 left in cash.
    assert result.nav == pytest.approx([100 + 300 * 33.5, 100 + 300 * 35.0])
    assert result.stats["buy_orders"] == 1 and result.stats["buy_value"] == 9_900


def test_limit_up_open_leaves_the_cash_idle_until_the_next_rebalance() -> None:
    # Session 0 closes at 10 (previous close 10); session 1 opens on the 11.00 limit.
    market = _market([[10.0], [11.0], [11.2]], [[10.0], [11.0], [11.5]])

    result = backtest.simulate(market, {1: [0]}, capital=10_000, top_n=1, frictions=NO_COSTS)

    assert result.stats["buy_blocked_limit"] == 1
    assert result.nav == pytest.approx([10_000, 10_000, 10_000])


def test_exit_waits_through_a_limit_down_and_a_suspension() -> None:
    # Name 0 is bought, then dropped from the pool at session 1, where it opens
    # limit-down; session 2 has no bar; it finally sells at the session 3 open.
    market = _market(
        [[10.0, 10.0], [9.0, 10.0], [np.nan, 10.0], [8.5, 10.0]],
        [[10.0, 10.0], [9.0, 10.0], [np.nan, 10.0], [8.6, 10.0]],
    )
    market.pre_close[3, 0] = 9.0

    result = backtest.simulate(
        market, {0: [0], 1: [1]}, capital=10_000, top_n=1, frictions=NO_COSTS
    )

    assert result.stats["sell_blocked_limit"] == 1
    assert result.stats["sell_blocked_suspended"] == 1
    assert result.stats["sell_value"] == 1_000 * 8.5
    # The replacement could not be funded on its rebalance day, so the proceeds sit in cash.
    assert result.nav == pytest.approx([10_000, 9_000, 9_000, 8_500])


def test_costs_come_out_of_the_fill_and_a_thin_name_is_only_partly_filled() -> None:
    market = _market([[10.0], [10.0]], [[10.0], [10.0]], adv_cny=np.full((2, 1), 50_000.0))
    frictions = backtest.Frictions("all", impact_coefficient_bps=0.0)

    result = backtest.simulate(market, {0: [0]}, capital=10_000, top_n=1, frictions=frictions)

    # The 10% cap on 50,000 of turnover allows 5,000: 500 shares. Half a tick on a
    # 10.00 stock is 5bp (2.50), and the commission is the 5.00 minimum plus 0.05 transfer.
    assert result.stats["buy_capped"] == 1 and result.stats["buy_value"] == 5_000
    assert result.stats["price_drag"] == pytest.approx(2.50)
    assert result.stats["fees"] == pytest.approx(5.05, abs=1e-3)
    assert result.nav[0] == pytest.approx(10_000 - 2.50 - 5.05, abs=1e-3)


def test_delisted_holding_is_written_down_to_the_recovery_rate() -> None:
    market = _market(
        [[10.0], [10.0], [np.nan]], [[10.0], [10.0], [np.nan]], delist_index=np.array([2])
    )

    lost = backtest.simulate(market, {0: [0]}, capital=10_000, top_n=1, frictions=NO_COSTS)
    kept = backtest.simulate(
        market, {0: [0]}, capital=10_000, top_n=1, frictions=NO_COSTS, delist_recovery=0.5
    )

    assert lost.nav[-1] == pytest.approx(0.0) and lost.stats["delisted_positions"] == 1
    assert kept.nav[-1] == pytest.approx(5_000.0)


def test_universe_benchmarks_use_only_listed_names_and_zero_a_suspended_day() -> None:
    days = ["20240102", "20240103", "20240104"]
    bars = pd.DataFrame(
        [
            ("A", days[0], 10.0, 100.0, True),
            ("A", days[1], 11.0, 100.0, True),
            ("A", days[2], 11.0, 100.0, True),
            ("B", days[0], 10.0, 300.0, True),
            ("B", days[2], 13.0, 300.0, True),  # no bar on the middle day
            ("C", days[0], 10.0, 100.0, False),
            ("C", days[1], 20.0, 100.0, False),
        ],
        columns=["ts_code", "trade_date", "adj_close", "total_mv", "listed"],
    )

    table = backtest.universe_benchmarks(bars, days[1:])

    # Day 2: A +10%, B suspended counts as 0%, unlisted C is ignored.
    assert table.loc[days[1], "equal_weight"] == pytest.approx(0.05)
    assert table.loc[days[1], "cap_weight"] == pytest.approx(0.10 * 100 / 400)
    # Day 3: only A traded the day before, and it was flat.
    assert table.loc[days[2], "equal_weight"] == pytest.approx(0.0)
