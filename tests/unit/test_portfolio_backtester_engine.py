"""
组合回测可信化测试
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtest import PortfolioBacktester, PortfolioConstructor, TransactionCostModel


def _make_backtest_frame(periods: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    rows = []
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    for day_idx, date in enumerate(dates):
        benchmark = 0.0002 + 0.0001 * np.sin(day_idx / 7)
        for sym_idx, symbol in enumerate(symbols):
            score = 1.5 - sym_idx * 0.4 + 0.1 * np.sin((day_idx + sym_idx) / 5)
            forward = 0.0015 * score + 0.0003 * np.cos((day_idx + 1) / 3)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 100 + day_idx + sym_idx,
                    "factor_score": score,
                    "forward_ret_1d": forward,
                    "benchmark_return": benchmark,
                    "industry": "TMT" if sym_idx < 2 else "消费",
                }
            )
    return pd.DataFrame(rows)


def test_build_weights_uses_history_before_rebalance_date(monkeypatch):
    df = _make_backtest_frame()
    backtester = PortfolioBacktester(df, construction_method="risk_parity", n_holdings=2)
    date = pd.Timestamp(df["date"].unique()[40])
    scores = df[df["date"] == date].set_index("symbol")["factor_score"]
    ret_matrix = df.pivot_table(index="date", columns="symbol", values="forward_ret_1d")
    captured = {}

    def _fake_risk_parity(scores_arg, ret_matrix_arg, n_top=10, lookback=60):
        captured["max_seen_date"] = ret_matrix_arg.index.max()
        return PortfolioConstructor.equal_weight(scores_arg, n_top=2)

    monkeypatch.setattr(PortfolioConstructor, "risk_parity", staticmethod(_fake_risk_parity))
    weights = backtester._build_weights(scores, ret_matrix, date, {})

    assert weights.sum() > 0
    assert captured["max_seen_date"] < date


def test_higher_costs_reduce_final_nav():
    df = _make_backtest_frame()
    low_cost = TransactionCostModel(commission_rate=0.0001, stamp_duty=0.0001, slippage_rate=0.0001)
    high_cost = TransactionCostModel(commission_rate=0.003, stamp_duty=0.003, slippage_rate=0.003)

    low = PortfolioBacktester(df, rebalance_freq="W", cost_model=low_cost, n_holdings=2).run_simple()
    high = PortfolioBacktester(df, rebalance_freq="W", cost_model=high_cost, n_holdings=2).run_simple()

    assert float(high.full_portfolio_nav.iloc[-1]) <= float(low.full_portfolio_nav.iloc[-1])
    assert high.combined_metrics.total_transaction_cost >= low.combined_metrics.total_transaction_cost


def test_rebalance_frequency_changes_turnover():
    df = _make_backtest_frame(periods=180)
    weekly = PortfolioBacktester(df, rebalance_freq="W", n_holdings=2).run_simple()
    monthly = PortfolioBacktester(df, rebalance_freq="M", n_holdings=2).run_simple()

    assert weekly.combined_metrics.annual_turnover != monthly.combined_metrics.annual_turnover


def test_annual_turnover_limit_is_enforced():
    df = _make_backtest_frame(periods=180)
    result = PortfolioBacktester(
        df,
        rebalance_freq="D",
        n_holdings=2,
        max_annual_turnover=0.30,
        min_holding_days=1,
    ).run_simple()

    assert result.combined_metrics.annual_turnover <= 0.31


def test_run_walkforward_consumes_training_window(monkeypatch):
    df = _make_backtest_frame(periods=320)
    backtester = PortfolioBacktester(df, rebalance_freq="W", n_holdings=2, wf_train_months=6, wf_test_months=2)
    calls = []
    original = backtester._prepare_training_context

    def _wrapped(train_df):
        calls.append((train_df["date"].min(), train_df["date"].max(), len(train_df)))
        return original(train_df)

    monkeypatch.setattr(backtester, "_prepare_training_context", _wrapped)
    result = backtester.run_walkforward()

    assert calls
    assert all(length > 0 for _, _, length in calls)
    assert result.windows


def test_state_history_records_execution_fields():
    df = _make_backtest_frame(periods=80)
    result = PortfolioBacktester(df, rebalance_freq="W", n_holdings=2).run_simple()
    last_state = result.state_history[-1]

    assert hasattr(last_state, "positions_shares")
    assert hasattr(last_state, "positions_value")
    assert hasattr(last_state, "transaction_cost")
    assert hasattr(last_state, "gross_exposure")
    assert isinstance(last_state.positions_value, dict)
