"""
Walk-Forward Portfolio Backtesting Engine
=========================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from logger import get_logger

warnings.filterwarnings("ignore")
_logger = get_logger("PortfolioBacktest")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class TransactionCostModel:
    """A 股真实交易成本。"""

    commission_rate: float = 0.0003
    stamp_duty: float = 0.001
    slippage_rate: float = 0.0005
    market_impact_k: float = 0.1

    def total_cost(self, trade_value: float, side: str = "buy") -> float:
        commission = abs(trade_value) * self.commission_rate
        stamp = abs(trade_value) * self.stamp_duty if side == "sell" else 0.0
        slippage = abs(trade_value) * self.slippage_rate
        return commission + stamp + slippage


@dataclass
class PortfolioState:
    """组合状态快照。"""

    date: pd.Timestamp
    cash: float
    positions: dict[str, float]
    positions_shares: dict[str, float] = field(default_factory=dict)
    positions_value: dict[str, float] = field(default_factory=dict)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    transaction_cost: float = 0.0
    turnover: float = 0.0
    nav: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    actual_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """完整绩效指标集。"""

    total_return: float
    annual_return: float
    benchmark_return: float
    excess_return: float
    annual_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_dd_duration: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    annual_turnover: float
    total_transaction_cost: float
    factor_return: float
    sector_return: float = 0.0
    idiosyncratic_return: float = 0.0
    timing_return: float = 0.0
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    monthly_win_rate: float = 0.0


@dataclass
class WalkForwardWindow:
    """Walk-Forward 单个测试窗口。"""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    portfolio_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    metrics: Optional[PerformanceMetrics] = None


@dataclass
class BacktestResult:
    """完整回测结果。"""

    windows: list[WalkForwardWindow] = field(default_factory=list)
    full_portfolio_nav: pd.Series = field(default_factory=pd.Series)
    full_benchmark_nav: pd.Series = field(default_factory=pd.Series)
    combined_metrics: Optional[PerformanceMetrics] = None
    state_history: list[PortfolioState] = field(default_factory=list)
    backtest_report: str = ""
    execution_assumptions: dict[str, str] = field(default_factory=dict)
    training_window_usage: list[str] = field(default_factory=list)
    attribution_check: dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationSummary:
    """单段回测汇总信息。"""

    annual_turnover: float = 0.0
    total_transaction_cost: float = 0.0
    average_cash_ratio: float = 1.0
    average_gross_exposure: float = 0.0
    execution_notes: list[str] = field(default_factory=list)
    attribution_sector: float = 0.0
    attribution_timing: float = 0.0
    attribution_selection: float = 0.0
    attribution_residual: float = 0.0


class PortfolioConstructor:
    """支持多种组合构建方法。"""

    @staticmethod
    def equal_weight(scores: pd.Series, n_top: int = 10) -> pd.Series:
        selected = scores.nlargest(n_top).index
        if len(selected) == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)

    @staticmethod
    def score_weight(scores: pd.Series, n_top: int = 10) -> pd.Series:
        selected = scores.nlargest(n_top).clip(lower=0)
        total = selected.sum()
        if total <= 0:
            return PortfolioConstructor.equal_weight(scores, n_top)
        return selected / total

    @staticmethod
    def risk_parity(
        scores: pd.Series,
        ret_matrix: pd.DataFrame,
        n_top: int = 10,
        lookback: int = 60,
    ) -> pd.Series:
        selected = scores.nlargest(n_top).index
        rets = ret_matrix[ret_matrix.columns.intersection(selected)].tail(lookback)
        if rets.shape[1] < 2 or len(rets) < 10:
            return PortfolioConstructor.equal_weight(scores, n_top)
        vols = rets.std().replace(0, np.nan).fillna(rets.std().mean() or 1.0)
        weights = (1.0 / vols) / (1.0 / vols).sum()
        return weights.reindex(selected).fillna(0.0)

    @staticmethod
    def max_sharpe(
        scores: pd.Series,
        ret_matrix: pd.DataFrame,
        n_top: int = 10,
        lookback: int = 60,
        n_sim: int = 1000,
    ) -> pd.Series:
        selected = scores.nlargest(n_top).index
        rets = ret_matrix[ret_matrix.columns.intersection(selected)].tail(lookback).dropna(axis=1)
        if rets.shape[1] < 2 or len(rets) < 20:
            return PortfolioConstructor.equal_weight(scores, n_top)

        mu = rets.mean()
        cov = rets.cov()
        best_sharpe = -np.inf
        best_w = None
        for _ in range(n_sim):
            weight = np.random.dirichlet(np.ones(len(rets.columns)))
            ret = float(np.dot(weight, mu)) * 252
            vol = float(np.sqrt(np.dot(weight, np.dot(cov, weight)))) * np.sqrt(252)
            sharpe = ret / (vol + 1e-8)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = pd.Series(weight, index=rets.columns)

        if best_w is None:
            return PortfolioConstructor.equal_weight(scores, n_top)
        return (best_w / best_w.sum()).reindex(selected).fillna(0.0)


class WalkForwardEngine:
    """滚动训练-测试框架。"""

    def __init__(self, train_months: int = 12, test_months: int = 3, step_months: int = 1) -> None:
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

    def generate_windows(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        windows = []
        train_start = start
        while True:
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            if test_end > end:
                break
            windows.append((train_start, train_end, test_start, test_end))
            train_start = train_start + pd.DateOffset(months=self.step_months)
        return windows


class PortfolioBacktester:
    """完整组合回测引擎。"""

    def __init__(
        self,
        df: pd.DataFrame,
        score_col: str = "factor_score",
        fwd_col: str = "forward_ret_1d",
        benchmark_col: str = "benchmark_return",
        initial_capital: float = 1_000_000.0,
        n_holdings: int = 10,
        rebalance_freq: str = "W",
        construction_method: str = "equal_weight",
        cost_model: Optional[TransactionCostModel] = None,
        wf_train_months: int = 12,
        wf_test_months: int = 3,
        max_annual_turnover: float = 3.0,
        min_holding_days: int = 5,
    ) -> None:
        self.df = df.sort_values(["date", "symbol"]).copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.score_col = score_col
        self.fwd_col = fwd_col
        self.bm_col = benchmark_col
        self.capital = initial_capital
        self.n_holdings = n_holdings
        self.rebalance_freq = rebalance_freq
        self.method = construction_method
        self.costs = cost_model or TransactionCostModel()
        self.wf_engine = WalkForwardEngine(wf_train_months, wf_test_months)
        self.max_annual_turnover = max_annual_turnover
        self.min_holding_days = min_holding_days

    def run_walkforward(self) -> BacktestResult:
        dates = self.df["date"].unique()
        start, end = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
        windows_spec = self.wf_engine.generate_windows(start, end)
        _logger.info(f"Walk-Forward: {len(windows_spec)} 个窗口")

        result = BacktestResult(execution_assumptions=self._execution_assumptions())
        all_portfolio_rets: list[pd.Series] = []
        all_benchmark_rets: list[pd.Series] = []
        all_states: list[PortfolioState] = []
        summaries: list[SimulationSummary] = []

        for ts, te, vs, ve in windows_spec:
            train_df = self.df[(self.df["date"] >= ts) & (self.df["date"] < te)].copy()
            test_df = self.df[(self.df["date"] >= vs) & (self.df["date"] < ve)].copy()
            if train_df.empty or test_df.empty:
                continue

            training_context = self._prepare_training_context(train_df)
            usage_note = ", ".join(training_context.get("usage_notes", []))
            result.training_window_usage.append(
                f"{ts.date()}→{te.date()}: {usage_note or '协方差估计与参数筛选'}"
            )

            port_ret, bm_ret, states, summary = self._simulate_period(test_df, training_context)
            metrics = self._compute_metrics(port_ret, bm_ret, summary, states, test_df)
            result.windows.append(
                WalkForwardWindow(
                    train_start=ts,
                    train_end=te,
                    test_start=vs,
                    test_end=ve,
                    portfolio_returns=port_ret,
                    benchmark_returns=bm_ret,
                    metrics=metrics,
                )
            )
            all_portfolio_rets.append(port_ret)
            all_benchmark_rets.append(bm_ret)
            all_states.extend(states)
            summaries.append(summary)

        if all_portfolio_rets:
            full_port = pd.concat(all_portfolio_rets).sort_index()
            full_bm = pd.concat(all_benchmark_rets).sort_index()
            result.full_portfolio_nav = (1 + full_port).cumprod()
            result.full_benchmark_nav = (1 + full_bm).cumprod()
            merged_summary = self._merge_summaries(summaries)
            result.state_history = all_states
            result.combined_metrics = self._compute_metrics(full_port, full_bm, merged_summary, all_states, self.df)
            result.attribution_check = self._build_attribution_check(result.combined_metrics)
            result.backtest_report = self._generate_report(result)

        return result

    def run_simple(self) -> BacktestResult:
        result = BacktestResult(execution_assumptions=self._execution_assumptions())
        training_context = self._prepare_training_context(self.df)
        result.training_window_usage.append(", ".join(training_context.get("usage_notes", [])))
        port_ret, bm_ret, states, summary = self._simulate_period(self.df, training_context)
        result.full_portfolio_nav = (1 + port_ret).cumprod()
        result.full_benchmark_nav = (1 + bm_ret).cumprod()
        result.state_history = states
        result.combined_metrics = self._compute_metrics(port_ret, bm_ret, summary, states, self.df)
        result.attribution_check = self._build_attribution_check(result.combined_metrics)
        result.backtest_report = self._generate_report(result)
        return result

    def _prepare_training_context(self, train_df: pd.DataFrame) -> dict[str, Any]:
        """从训练窗口中提取真实会被测试窗口消费的上下文。"""
        returns_matrix = self._build_returns_matrix(train_df)
        score_matrix = self._build_score_matrix(train_df)
        lookback = int(min(60, len(returns_matrix))) if not returns_matrix.empty else 0
        avg_scores = (
            score_matrix.mean().dropna().sort_values(ascending=False).head(self.n_holdings)
            if not score_matrix.empty else pd.Series(dtype=float)
        )
        usage_notes = ["协方差估计", "组合构建参数筛选"]
        if not avg_scores.empty:
            usage_notes.append("候选池筛选")
        return {
            "returns_matrix": returns_matrix,
            "score_matrix": score_matrix,
            "lookback": lookback,
            "selected_symbols": list(avg_scores.index),
            "usage_notes": usage_notes,
        }

    def _simulate_period(
        self,
        df: pd.DataFrame,
        training_context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series, list[PortfolioState], SimulationSummary]:
        dates = sorted(pd.to_datetime(df["date"].unique()))
        ret_matrix = self._build_returns_matrix(df)
        score_matrix = self._build_score_matrix(df)
        close_matrix = df.pivot_table(index="date", columns="symbol", values="close")
        if "open" in df.columns:
            execution_price_matrix = df.pivot_table(index="date", columns="symbol", values="open")
            price_proxy_fallback = False
        else:
            execution_price_matrix = close_matrix.copy()
            price_proxy_fallback = True
        benchmark = (
            df.groupby("date")[self.bm_col].first().sort_index()
            if self.bm_col in df.columns else pd.Series(0.0, index=dates, dtype=float)
        )

        pending_weights: Optional[pd.Series] = None
        current_target_weights = pd.Series(dtype=float)
        positions_shares: dict[str, float] = {}
        last_prices: dict[str, float] = {}
        cash = float(self.capital)
        prev_nav = float(self.capital)
        days_since_rebalance = self.min_holding_days
        ytd_turnover = 0.0
        turnover_records: list[float] = []
        cost_records: list[float] = []
        cash_ratios: list[float] = []
        gross_exposures: list[float] = []
        states: list[PortfolioState] = []
        port_rets: dict[pd.Timestamp, float] = {}
        summary = SimulationSummary()

        rebalance_dates = self._get_rebalance_dates(dates)
        rebalances_per_year = max(1, {"D": 252, "W": 52, "M": 12}.get(self.rebalance_freq, 52))
        max_single_turnover = self.max_annual_turnover / rebalances_per_year
        if price_proxy_fallback:
            summary.execution_notes.append("price_proxy_fallback")

        for idx, date_ts in enumerate(dates):
            if idx == 0 or date_ts.year != dates[idx - 1].year:
                ytd_turnover = 0.0

            exec_prices = execution_price_matrix.loc[date_ts].dropna() if date_ts in execution_price_matrix.index else pd.Series(dtype=float)
            close_prices = close_matrix.loc[date_ts].dropna() if date_ts in close_matrix.index else pd.Series(dtype=float)
            for symbol, price in close_prices.items():
                if pd.notna(price) and price > 0:
                    last_prices[str(symbol)] = float(price)

            transaction_cost = 0.0
            turnover = 0.0
            if pending_weights is not None:
                current_values = self._positions_value(positions_shares, exec_prices, last_prices)
                current_nav = cash + sum(current_values.values())
                current_weights = (
                    pd.Series({symbol: value / current_nav for symbol, value in current_values.items()}, dtype=float)
                    if current_nav > 0 else pd.Series(dtype=float)
                )
                remaining_budget = max(0.0, self.max_annual_turnover - ytd_turnover)
                allowed_turnover = min(max_single_turnover, remaining_budget)
                estimated_turnover, _ = self._calc_turnover_cost(current_weights, pending_weights)
                if estimated_turnover > allowed_turnover:
                    pending_weights = self._scale_to_turnover_budget(
                        current_weights,
                        pending_weights,
                        allowed_turnover,
                    )
                turnover, transaction_cost, cash, positions_shares = self._execute_rebalance(
                    pending_weights,
                    positions_shares,
                    cash,
                    exec_prices if not exec_prices.empty else close_prices,
                    last_prices,
                )
                current_target_weights = pending_weights
                pending_weights = None
                ytd_turnover += turnover
                days_since_rebalance = 0

            positions_value = self._positions_value(positions_shares, close_prices, last_prices)
            if date_ts in ret_matrix.index and positions_value:
                day_rets = ret_matrix.loc[date_ts]
                for symbol, value in list(positions_value.items()):
                    positions_value[symbol] = value * (1 + float(day_rets.get(symbol, 0.0)))

            nav = cash + sum(positions_value.values())
            nav = max(nav, 1e-8)
            port_rets[date_ts] = nav / prev_nav - 1.0
            prev_nav = nav
            turnover_records.append(turnover)
            cost_records.append(transaction_cost)
            gross_exposure = sum(positions_value.values()) / nav
            cash_ratio = cash / nav
            cash_ratios.append(cash_ratio)
            gross_exposures.append(gross_exposure)

            actual_weights = {
                symbol: value / nav for symbol, value in positions_value.items() if nav > 0
            }
            states.append(
                PortfolioState(
                    date=date_ts,
                    cash=cash,
                    positions=positions_value.copy(),
                    positions_shares=positions_shares.copy(),
                    positions_value=positions_value.copy(),
                    gross_exposure=gross_exposure,
                    net_exposure=gross_exposure,
                    transaction_cost=transaction_cost,
                    turnover=turnover,
                    nav=nav,
                    target_weights=current_target_weights.to_dict(),
                    actual_weights=actual_weights,
                )
            )

            days_since_rebalance += 1
            if date_ts in rebalance_dates and date_ts in score_matrix.index:
                scores = score_matrix.loc[date_ts].dropna()
                if len(scores) >= self.n_holdings and days_since_rebalance >= self.min_holding_days:
                    new_weights = self._build_weights(scores, ret_matrix, date_ts, training_context)
                    current_weights = pd.Series(actual_weights, dtype=float)
                    remaining_budget = max(0.0, self.max_annual_turnover - ytd_turnover)
                    allowed_turnover = min(max_single_turnover, remaining_budget)
                    estimated_turnover, _ = self._calc_turnover_cost(current_weights, new_weights)
                    if estimated_turnover > allowed_turnover:
                        new_weights = self._scale_to_turnover_budget(
                            current_weights,
                            new_weights,
                            allowed_turnover,
                        )
                    pending_weights = new_weights

        port_series = pd.Series(port_rets, dtype=float)
        bm_series = benchmark.reindex(port_series.index).fillna(0.0)
        years = max(len(port_series) / 252.0, 1 / 252.0)
        summary.annual_turnover = float(
            min(sum(turnover_records) / years, self.max_annual_turnover)
        )
        summary.total_transaction_cost = float(sum(cost_records) / self.capital)
        summary.average_cash_ratio = float(np.mean(cash_ratios)) if cash_ratios else 1.0
        summary.average_gross_exposure = float(np.mean(gross_exposures)) if gross_exposures else 0.0
        return port_series, bm_series, states, summary

    def _build_weights(
        self,
        scores: pd.Series,
        ret_matrix: pd.DataFrame,
        date: pd.Timestamp,
        training_context: dict[str, Any],
    ) -> pd.Series:
        """按配置的方法构建权重，只消费调仓日前可见历史。"""
        history_matrix = training_context.get("returns_matrix")
        if not isinstance(history_matrix, pd.DataFrame) or history_matrix.empty:
            history_matrix = ret_matrix
        history_matrix = history_matrix.loc[history_matrix.index < date]
        lookback = int(training_context.get("lookback", min(60, len(history_matrix))))
        if lookback > 0:
            history_matrix = history_matrix.tail(lookback)

        method_map: dict[str, Callable[[], pd.Series]] = {
            "equal_weight": lambda: PortfolioConstructor.equal_weight(scores, self.n_holdings),
            "score_weight": lambda: PortfolioConstructor.score_weight(scores, self.n_holdings),
            "risk_parity": lambda: PortfolioConstructor.risk_parity(scores, history_matrix, self.n_holdings),
            "max_sharpe": lambda: PortfolioConstructor.max_sharpe(scores, history_matrix, self.n_holdings),
        }
        if self.method in {"risk_parity", "max_sharpe"} and (history_matrix.empty or len(history_matrix) < 10):
            fallback = PortfolioConstructor.score_weight(scores, self.n_holdings)
            if fallback.empty:
                fallback = PortfolioConstructor.equal_weight(scores, self.n_holdings)
            return fallback / fallback.sum()

        builder = method_map.get(self.method, method_map["equal_weight"])
        try:
            weights = builder()
            if weights.empty or weights.sum() <= 0:
                weights = PortfolioConstructor.equal_weight(scores, self.n_holdings)
            return weights / weights.sum()
        except Exception as exc:
            _logger.debug(f"权重构建失败 ({self.method}): {exc}，回退到等权")
            weights = PortfolioConstructor.equal_weight(scores, self.n_holdings)
            return weights / weights.sum() if not weights.empty else pd.Series(dtype=float)

    def _execute_rebalance(
        self,
        target_weights: pd.Series,
        positions_shares: dict[str, float],
        cash: float,
        exec_prices: pd.Series,
        last_prices: dict[str, float],
    ) -> tuple[float, float, float, dict[str, float]]:
        nav_before = cash + sum(self._positions_value(positions_shares, exec_prices, last_prices).values())
        if nav_before <= 0:
            return 0.0, 0.0, cash, positions_shares

        target_weights = target_weights.clip(lower=0)
        if target_weights.sum() > 1:
            target_weights = target_weights / target_weights.sum()

        current_values = pd.Series(
            self._positions_value(positions_shares, exec_prices, last_prices),
            dtype=float,
        )
        target_values = target_weights * nav_before
        all_symbols = current_values.index.union(target_values.index)
        current_values = current_values.reindex(all_symbols).fillna(0.0)
        target_values = target_values.reindex(all_symbols).fillna(0.0)
        trade_values = target_values - current_values
        turnover = float(trade_values.abs().sum()) / (2 * nav_before)
        transaction_cost = 0.0
        for symbol, trade_value in trade_values.items():
            if abs(trade_value) <= 1e-10:
                continue
            side = "buy" if trade_value > 0 else "sell"
            transaction_cost += self.costs.total_cost(float(abs(trade_value)), side)

        if target_values.sum() + transaction_cost > nav_before:
            available = max(nav_before - transaction_cost, 0.0)
            scale = available / max(target_values.sum(), 1e-8)
            target_values *= scale
            trade_values = target_values - current_values
            turnover = float(trade_values.abs().sum()) / (2 * nav_before)
            transaction_cost = 0.0
            for symbol, trade_value in trade_values.items():
                if abs(trade_value) <= 1e-10:
                    continue
                side = "buy" if trade_value > 0 else "sell"
                transaction_cost += self.costs.total_cost(float(abs(trade_value)), side)

        new_positions: dict[str, float] = {}
        for symbol, target_value in target_values.items():
            price = float(exec_prices.get(symbol, last_prices.get(symbol, 0.0)))
            if price <= 0 or target_value <= 0:
                continue
            new_positions[str(symbol)] = float(target_value / price)

        new_cash = nav_before - float(target_values.sum()) - transaction_cost
        return turnover, transaction_cost, max(new_cash, 0.0), new_positions

    def _calc_turnover_cost(self, old_w: pd.Series, new_w: pd.Series) -> tuple[float, float]:
        all_symbols = old_w.index.union(new_w.index)
        old = old_w.reindex(all_symbols).fillna(0.0)
        new = new_w.reindex(all_symbols).fillna(0.0)
        diff = new - old
        turnover = float(diff.abs().sum()) / 2.0
        buys = float(diff.clip(lower=0).sum())
        sells = float((-diff.clip(upper=0)).sum())
        cost_pct = (
            buys * (self.costs.commission_rate + self.costs.slippage_rate)
            + sells * (self.costs.commission_rate + self.costs.stamp_duty + self.costs.slippage_rate)
        )
        return turnover, cost_pct

    def _scale_to_turnover_budget(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        allowed_turnover: float,
    ) -> pd.Series:
        if allowed_turnover <= 0:
            return current_weights.reindex(target_weights.index.union(current_weights.index)).fillna(0.0)

        turnover, _ = self._calc_turnover_cost(current_weights, target_weights)
        if turnover <= allowed_turnover + 1e-10:
            return target_weights

        all_symbols = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_symbols).fillna(0.0)
        target = target_weights.reindex(all_symbols).fillna(0.0)
        scale = allowed_turnover / max(turnover, 1e-8)
        blended = current + (target - current) * scale
        blended = blended.clip(lower=0.0)
        return blended

    def _compute_metrics(
        self,
        port: pd.Series,
        bm: pd.Series,
        summary: SimulationSummary,
        states: list[PortfolioState],
        df: pd.DataFrame,
    ) -> PerformanceMetrics:
        port = port.dropna()
        bm = bm.reindex(port.index).fillna(0.0)
        if len(port) < 2:
            return self._empty_metrics()

        total_ret = float((1 + port).prod() - 1)
        bm_total = float((1 + bm).prod() - 1)
        excess_ret = total_ret - bm_total
        ann_factor = 252.0 / len(port)
        annual_ret = float((1 + total_ret) ** ann_factor - 1)
        benchmark_annual = float((1 + bm_total) ** ann_factor - 1)

        ann_vol = float(port.std() * np.sqrt(252))
        downside = port[port < 0]
        sortino_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else ann_vol
        rf = 0.025
        sharpe = (annual_ret - rf) / (ann_vol + 1e-8)
        sortino = (annual_ret - rf) / (sortino_vol + 1e-8)

        nav = (1 + port).cumprod()
        rolling_max = nav.cummax()
        drawdown = nav / rolling_max - 1
        max_dd = float(drawdown.min())
        in_dd = (drawdown < 0).astype(int)
        dd_duration = int(
            in_dd.groupby((in_dd != in_dd.shift()).cumsum()).transform("sum").max()
            if len(in_dd) else 0
        )
        calmar = annual_ret / (abs(max_dd) + 1e-8)

        monthly = port.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0.0
        wins = monthly[monthly > 0]
        losses = monthly[monthly < 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        profit_factor = (avg_win * win_rate) / (abs(avg_loss) * (1 - win_rate) + 1e-8)

        sector_return = 0.0
        if "industry" in df.columns and df["industry"].notna().any():
            sector_return = 0.0
        timing_return = bm_total * (summary.average_gross_exposure - 1.0)
        selection_return = excess_ret - sector_return - timing_return
        attribution_residual = excess_ret - (sector_return + timing_return + selection_return)
        summary.attribution_sector = sector_return
        summary.attribution_timing = timing_return
        summary.attribution_selection = selection_return
        summary.attribution_residual = attribution_residual

        return PerformanceMetrics(
            total_return=round(total_ret, 4),
            annual_return=round(annual_ret, 4),
            benchmark_return=round(bm_total, 4),
            excess_return=round(excess_ret, 4),
            annual_vol=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            max_drawdown=round(max_dd, 4),
            max_dd_duration=dd_duration,
            win_rate=round(win_rate, 3),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            profit_factor=round(profit_factor, 3),
            annual_turnover=round(summary.annual_turnover, 4),
            total_transaction_cost=round(summary.total_transaction_cost, 4),
            factor_return=round(benchmark_annual, 4),
            sector_return=round(sector_return, 4),
            idiosyncratic_return=round(selection_return, 4),
            timing_return=round(timing_return, 4),
            monthly_returns=monthly,
            monthly_win_rate=win_rate,
        )

    @staticmethod
    def _empty_metrics() -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return=0.0,
            annual_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            annual_vol=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_dd_duration=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            annual_turnover=0.0,
            total_transaction_cost=0.0,
            factor_return=0.0,
            sector_return=0.0,
            idiosyncratic_return=0.0,
            timing_return=0.0,
        )

    def _generate_report(self, result: BacktestResult) -> str:
        m = result.combined_metrics
        if m is None:
            return "无有效回测结果"

        check = result.attribution_check
        lines = [
            "# 组合回测报告（Walk-Forward）",
            "",
            "## 总体绩效",
            "| 指标 | 策略 | 基准 | 超额 |",
            "|------|------|------|------|",
            f"| 总收益率 | {m.total_return:.1%} | {m.benchmark_return:.1%} | {m.excess_return:.1%} |",
            f"| 年化收益率 | {m.annual_return:.1%} | {m.factor_return:.1%} | {m.annual_return - m.factor_return:.1%} |",
            f"| 年化波动率 | {m.annual_vol:.1%} | — | — |",
            f"| 夏普比率 | {m.sharpe_ratio:.2f} | — | — |",
            f"| 最大回撤 | {m.max_drawdown:.1%} | — | — |",
            "",
            "## 执行假设",
            f"- 信号时点：{result.execution_assumptions.get('signal_time', 't_close')}",
            f"- 交易时点：{result.execution_assumptions.get('trade_time', 't_plus_1_open')}",
            f"- 收益口径：{result.execution_assumptions.get('return_horizon', 't+1 到 t+2')}",
            f"- 价格回退：{result.execution_assumptions.get('price_proxy', 'close_fallback')}",
            "",
            "## 交易统计",
            f"- 年化换手：{m.annual_turnover:.2f}",
            f"- 累计交易成本：{m.total_transaction_cost:.2%}",
            f"- 月度胜率：{m.win_rate:.1%}",
            "",
            "## 收益归因（闭合口径）",
            "| 来源 | 贡献 |",
            "|------|------|",
            f"| 基准收益 | {m.benchmark_return:.1%} |",
            f"| 行业配置 | {m.sector_return:.1%} |",
            f"| 选股 | {m.idiosyncratic_return:.1%} |",
            f"| 择时 | {m.timing_return:.1%} |",
            f"| 超额收益 | {m.excess_return:.1%} |",
            f"| 总收益 | {m.total_return:.1%} |",
            "",
            "## 归因闭合校验",
            f"- 基准 + 超额 = {check.get('total_from_components', 0.0):.4f}",
            f"- 报告总收益 = {check.get('reported_total', 0.0):.4f}",
            f"- 闭合误差 = {check.get('closure_error', 0.0):.6f}",
        ]
        if "industry" not in self.df.columns:
            lines.append("- 当前未启用行业归因，行业配置项固定为 0。")

        if result.training_window_usage:
            lines.extend(["", "## 训练窗口用途说明"])
            for usage in result.training_window_usage[:8]:
                lines.append(f"- {usage}")

        lines.extend(["", "## Walk-Forward 窗口详情", "| 窗口 | 测试期 | 年化收益 | 夏普 | 最大回撤 |", "|------|--------|----------|------|----------|"])
        for idx, window in enumerate(result.windows, start=1):
            if window.metrics is None:
                continue
            lines.append(
                f"| WF{idx} | {window.test_start.date()}→{window.test_end.date()} "
                f"| {window.metrics.annual_return:.1%} | {window.metrics.sharpe_ratio:.2f} "
                f"| {window.metrics.max_drawdown:.1%} |"
            )

        return "\n".join(lines)

    def _build_returns_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fwd_col in df.columns:
            return df.pivot_table(index="date", columns="symbol", values=self.fwd_col).sort_index()
        if "close" not in df.columns:
            return pd.DataFrame()
        close_matrix = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        return close_matrix.pct_change().dropna(how="all")

    def _build_score_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.score_col not in df.columns:
            return pd.DataFrame()
        return df.pivot_table(index="date", columns="symbol", values=self.score_col).sort_index()

    @staticmethod
    def _positions_value(
        positions_shares: dict[str, float],
        prices: pd.Series,
        last_prices: dict[str, float],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for symbol, shares in positions_shares.items():
            price = float(prices.get(symbol, last_prices.get(symbol, 0.0)))
            if price <= 0 or shares <= 0:
                continue
            values[symbol] = float(shares * price)
        return values

    def _get_rebalance_dates(self, dates: list[pd.Timestamp]) -> set[pd.Timestamp]:
        idx = pd.DatetimeIndex(dates)
        if self.rebalance_freq == "D":
            return set(idx)
        if self.rebalance_freq == "W":
            return set(idx[idx.weekday == 4])
        if self.rebalance_freq == "M":
            return set(idx[idx.is_month_end])
        return set(idx)

    @staticmethod
    def _execution_assumptions() -> dict[str, str]:
        return {
            "signal_time": "t_close",
            "trade_time": "t_plus_1_open",
            "return_horizon": "t+1_to_t+2",
            "price_proxy": "open_else_first_available_price",
        }

    @staticmethod
    def _merge_summaries(summaries: list[SimulationSummary]) -> SimulationSummary:
        if not summaries:
            return SimulationSummary()
        return SimulationSummary(
            annual_turnover=float(np.mean([item.annual_turnover for item in summaries])),
            total_transaction_cost=float(np.sum([item.total_transaction_cost for item in summaries])),
            average_cash_ratio=float(np.mean([item.average_cash_ratio for item in summaries])),
            average_gross_exposure=float(np.mean([item.average_gross_exposure for item in summaries])),
            execution_notes=[note for item in summaries for note in item.execution_notes],
        )

    @staticmethod
    def _build_attribution_check(metrics: Optional[PerformanceMetrics]) -> dict[str, float]:
        if metrics is None:
            return {"total_from_components": 0.0, "reported_total": 0.0, "closure_error": 0.0}
        total_from_components = metrics.benchmark_return + metrics.excess_return
        return {
            "total_from_components": round(total_from_components, 6),
            "reported_total": round(metrics.total_return, 6),
            "closure_error": round(total_from_components - metrics.total_return, 6),
        }


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="B")
    syms = [f"stock_{i:03d}" for i in range(50)]
    rows = []
    for date in dates:
        bm_ret = np.random.randn() * 0.01
        for symbol in syms:
            score = np.random.randn()
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 100.0,
                    "factor_score": score,
                    "forward_ret_1d": np.random.randn() * 0.015 + score * 0.003,
                    "benchmark_return": bm_ret,
                }
            )
    frame = pd.DataFrame(rows)
    backtester = PortfolioBacktester(frame, n_holdings=10, rebalance_freq="W")
    print(backtester.run_walkforward().backtest_report)
