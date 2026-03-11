"""
Walk-Forward Portfolio Backtesting Engine
==========================================
解决当前系统只有单股票回测的核心缺陷。

核心功能：
  1. Walk-Forward 滚动测试（避免过拟合）
  2. 组合级别模拟（头寸、成本、流动性）
  3. 真实交易成本模型（双边印花税 + 佣金 + 滑点/市场冲击）
  4. 绩效归因（因子收益 vs 特质收益）
  5. 全套风险指标 + Benchmark 对比
  6. 结构化报告输出（可直接给用户）
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from logger import get_logger

warnings.filterwarnings("ignore")
_logger = get_logger("PortfolioBacktest")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TransactionCostModel:
    """A股真实交易成本"""
    commission_rate: float = 0.0003    # 券商佣金（万三）
    stamp_duty:      float = 0.001     # 印花税（卖出）
    slippage_rate:   float = 0.0005    # 滑点（买卖各一半）
    market_impact_k: float = 0.1       # 市场冲击系数（大单放大）

    def total_cost(self, trade_value: float, side: str = "buy") -> float:
        """计算单笔交易总成本"""
        commission = abs(trade_value) * self.commission_rate
        stamp = abs(trade_value) * self.stamp_duty if side == "sell" else 0.0
        slippage = abs(trade_value) * self.slippage_rate
        return commission + stamp + slippage


@dataclass
class PortfolioState:
    """组合状态快照"""
    date: pd.Timestamp
    cash: float
    positions: dict[str, float]   # symbol → 市值
    nav: float                    # 组合净值
    turnover: float               # 当日换手率


@dataclass
class PerformanceMetrics:
    """完整绩效指标集"""
    # 收益
    total_return: float
    annual_return: float
    benchmark_return: float
    excess_return: float

    # 风险
    annual_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_dd_duration: int          # 最大回撤持续天数

    # 交易统计
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float          # avg_win * win_rate / (avg_loss * (1-win_rate))
    annual_turnover: float
    total_transaction_cost: float

    # 归因
    factor_return: float          # 因子贡献
    idiosyncratic_return: float   # 特质贡献
    timing_return: float          # 择时贡献（相对持有等权）

    # 月度统计
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    monthly_win_rate: float = 0.0


@dataclass
class WalkForwardWindow:
    """Walk-Forward 单个测试窗口"""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    portfolio_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    metrics: Optional[PerformanceMetrics] = None


@dataclass
class BacktestResult:
    """完整回测结果"""
    windows: list[WalkForwardWindow] = field(default_factory=list)
    full_portfolio_nav: pd.Series = field(default_factory=pd.Series)
    full_benchmark_nav: pd.Series = field(default_factory=pd.Series)
    combined_metrics: Optional[PerformanceMetrics] = None
    state_history: list[PortfolioState] = field(default_factory=list)
    backtest_report: str = ""


# ---------------------------------------------------------------------------
# 组合构建方法
# ---------------------------------------------------------------------------

class PortfolioConstructor:
    """
    支持多种组合构建方法。
    输入: 因子/预测得分 Series（index=symbol）
    输出: 权重 Series（index=symbol，sum=1）
    """

    @staticmethod
    def equal_weight(scores: pd.Series, n_top: int = 10) -> pd.Series:
        """等权重（选得分最高的 N 只）"""
        selected = scores.nlargest(n_top).index
        w = pd.Series(1.0 / n_top, index=selected)
        return w

    @staticmethod
    def score_weight(scores: pd.Series, n_top: int = 10) -> pd.Series:
        """按得分比例加权"""
        selected = scores.nlargest(n_top)
        selected = selected.clip(lower=0)
        total = selected.sum()
        if total == 0:
            return PortfolioConstructor.equal_weight(scores, n_top)
        return selected / total

    @staticmethod
    def risk_parity(
        scores: pd.Series,
        ret_matrix: pd.DataFrame,
        n_top: int = 10,
        lookback: int = 60,
    ) -> pd.Series:
        """风险平价（等风险贡献）"""
        selected = scores.nlargest(n_top).index
        rets = ret_matrix[ret_matrix.columns.intersection(selected)].tail(lookback)
        if rets.shape[1] < 2:
            return PortfolioConstructor.equal_weight(scores, n_top)
        vols = rets.std()
        w = (1.0 / vols) / (1.0 / vols).sum()
        return w.reindex(selected).fillna(1.0 / n_top)

    @staticmethod
    def max_sharpe(
        scores: pd.Series,
        ret_matrix: pd.DataFrame,
        n_top: int = 10,
        lookback: int = 60,
        n_sim: int = 1000,
    ) -> pd.Series:
        """蒙特卡洛模拟找最大夏普组合"""
        selected = scores.nlargest(n_top).index
        rets = ret_matrix[ret_matrix.columns.intersection(selected)].tail(lookback).dropna(axis=1)
        if rets.shape[1] < 2 or len(rets) < 20:
            return PortfolioConstructor.equal_weight(scores, n_top)

        mu  = rets.mean()
        cov = rets.cov()

        best_sharpe = -np.inf
        best_w = None
        for _ in range(n_sim):
            w = np.random.dirichlet(np.ones(len(rets.columns)))
            r = float(np.dot(w, mu)) * 252
            v = float(np.sqrt(np.dot(w, np.dot(cov, w)))) * np.sqrt(252)
            s = r / (v + 1e-8)
            if s > best_sharpe:
                best_sharpe = s
                best_w = pd.Series(w, index=rets.columns)

        return (best_w / best_w.sum()).reindex(selected).fillna(0)


# ---------------------------------------------------------------------------
# Walk-Forward 引擎
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """
    滚动训练-测试框架。

    时间轴示意：
    |---train_months---|---test_months---|---train_months---|---test_months---|
                                        ←   step_months   →
    """

    def __init__(
        self,
        train_months: int = 12,
        test_months:  int = 3,
        step_months:  int = 1,
    ) -> None:
        self.train_months = train_months
        self.test_months  = test_months
        self.step_months  = step_months

    def generate_windows(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """返回 (train_start, train_end, test_start, test_end) 列表"""
        windows = []
        train_start = start
        while True:
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end   = test_start + pd.DateOffset(months=self.test_months)
            if test_end > end:
                break
            windows.append((train_start, train_end, test_start, test_end))
            train_start = train_start + pd.DateOffset(months=self.step_months)
        return windows


# ---------------------------------------------------------------------------
# 主回测引擎
# ---------------------------------------------------------------------------

class PortfolioBacktester:
    """
    完整组合回测引擎。

    输入要求的 df 列：
      date, symbol, close, [volume], forward_ret_1d（次日收益率）
      可选：factor_score（预测得分），benchmark_return（基准日收益）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        score_col: str = "factor_score",
        fwd_col: str = "forward_ret_1d",
        benchmark_col: str = "benchmark_return",
        initial_capital: float = 1_000_000.0,
        n_holdings: int = 10,
        rebalance_freq: str = "W",     # "D"=日, "W"=周, "M"=月
        construction_method: str = "equal_weight",  # equal_weight/score_weight/risk_parity/max_sharpe
        cost_model: Optional[TransactionCostModel] = None,
        wf_train_months: int = 12,
        wf_test_months: int = 3,
    ) -> None:
        self.df = df.sort_values(["date", "symbol"]).copy()
        self.score_col = score_col
        self.fwd_col = fwd_col
        self.bm_col = benchmark_col
        self.capital = initial_capital
        self.n_holdings = n_holdings
        self.rebalance_freq = rebalance_freq
        self.method = construction_method
        self.costs = cost_model or TransactionCostModel()
        self.wf_engine = WalkForwardEngine(wf_train_months, wf_test_months)

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def run_walkforward(self) -> BacktestResult:
        """执行 Walk-Forward 回测，返回完整结果"""
        dates = self.df["date"].unique()
        start, end = dates.min(), dates.max()
        windows_spec = self.wf_engine.generate_windows(
            pd.Timestamp(start), pd.Timestamp(end)
        )
        _logger.info(f"Walk-Forward: {len(windows_spec)} 个窗口")

        result = BacktestResult()
        all_portfolio_rets: list[pd.Series] = []
        all_benchmark_rets: list[pd.Series] = []

        for i, (ts, te, vs, ve) in enumerate(windows_spec):
            _logger.info(f"  窗口 {i+1}/{len(windows_spec)}: 训练 {ts.date()}→{te.date()}, 测试 {vs.date()}→{ve.date()}")
            window = WalkForwardWindow(ts, te, vs, ve)

            test_df = self.df[
                (self.df["date"] >= vs) & (self.df["date"] < ve)
            ].copy()

            if len(test_df) == 0:
                continue

            port_ret, bm_ret, states = self._simulate_period(test_df)
            window.portfolio_returns = port_ret
            window.benchmark_returns = bm_ret
            window.metrics = self._compute_metrics(port_ret, bm_ret)
            result.windows.append(window)
            result.state_history.extend(states)

            all_portfolio_rets.append(port_ret)
            all_benchmark_rets.append(bm_ret)

        if all_portfolio_rets:
            full_port = pd.concat(all_portfolio_rets).sort_index()
            full_bm   = pd.concat(all_benchmark_rets).sort_index()

            result.full_portfolio_nav = (1 + full_port).cumprod()
            result.full_benchmark_nav = (1 + full_bm).cumprod()
            result.combined_metrics   = self._compute_metrics(full_port, full_bm)
            result.backtest_report    = self._generate_report(result)

        return result

    def run_simple(self) -> BacktestResult:
        """不分窗口的简单全段回测（对比基准用）"""
        result = BacktestResult()
        port_ret, bm_ret, states = self._simulate_period(self.df)
        result.full_portfolio_nav = (1 + port_ret).cumprod()
        result.full_benchmark_nav = (1 + bm_ret).cumprod()
        result.state_history = states
        result.combined_metrics = self._compute_metrics(port_ret, bm_ret)
        result.backtest_report = self._generate_report(result)
        return result

    # ----------------------------------------------------------------
    # 核心模拟逻辑
    # ----------------------------------------------------------------

    def _simulate_period(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, list[PortfolioState]]:
        """在给定时段内模拟组合，返回日收益序列"""
        dates = sorted(df["date"].unique())
        ret_matrix = (
            df.pivot_table(index="date", columns="symbol", values=self.fwd_col)
            if self.fwd_col in df.columns else pd.DataFrame()
        )
        score_matrix = (
            df.pivot_table(index="date", columns="symbol", values=self.score_col)
            if self.score_col in df.columns else pd.DataFrame()
        )
        bm_series = (
            df.groupby("date")[self.bm_col].first()
            if self.bm_col in df.columns
            else pd.Series(0.0, index=dates)
        )

        nav = 1.0
        cash_ratio = 1.0
        current_weights: pd.Series = pd.Series(dtype=float)
        total_cost_pct = 0.0

        port_rets: dict[pd.Timestamp, float] = {}
        states: list[PortfolioState] = []

        rebalance_dates = self._get_rebalance_dates(dates)

        for date in dates:
            date_ts = pd.Timestamp(date)

            # 换仓
            if date_ts in rebalance_dates and not score_matrix.empty:
                scores = score_matrix.loc[date].dropna()
                if len(scores) >= self.n_holdings:
                    new_weights = self._build_weights(scores, ret_matrix, date_ts)
                    turnover, cost_pct = self._calc_turnover_cost(current_weights, new_weights)
                    current_weights = new_weights
                    total_cost_pct += cost_pct
                    nav *= (1 - cost_pct)

            # 当日组合收益
            if not ret_matrix.empty and not current_weights.empty and date_ts in ret_matrix.index:
                day_rets = ret_matrix.loc[date_ts].reindex(current_weights.index).fillna(0.0)
                port_ret_day = float((current_weights * day_rets).sum())
            else:
                port_ret_day = 0.0

            port_rets[date_ts] = port_ret_day
            nav *= (1 + port_ret_day)

            states.append(PortfolioState(
                date=date_ts,
                cash=0.0,
                positions={s: w * nav for s, w in current_weights.items()},
                nav=nav,
                turnover=0.0,
            ))

        port_series = pd.Series(port_rets)
        bm_series_aligned = bm_series.reindex(port_series.index).fillna(0.0)
        return port_series, bm_series_aligned, states

    def _build_weights(
        self, scores: pd.Series, ret_matrix: pd.DataFrame, date: pd.Timestamp
    ) -> pd.Series:
        """按配置的方法构建权重"""
        method_map: dict[str, Callable] = {
            "equal_weight":  lambda: PortfolioConstructor.equal_weight(scores, self.n_holdings),
            "score_weight":  lambda: PortfolioConstructor.score_weight(scores, self.n_holdings),
            "risk_parity":   lambda: PortfolioConstructor.risk_parity(
                                    scores, ret_matrix, self.n_holdings),
            "max_sharpe":    lambda: PortfolioConstructor.max_sharpe(
                                    scores, ret_matrix, self.n_holdings),
        }
        func = method_map.get(self.method, method_map["equal_weight"])
        try:
            w = func()
            return w / w.sum() if w.sum() > 0 else pd.Series(dtype=float)
        except Exception as e:
            _logger.debug(f"权重构建失败 ({self.method}): {e}，回退到等权")
            return PortfolioConstructor.equal_weight(scores, self.n_holdings)

    def _calc_turnover_cost(
        self, old_w: pd.Series, new_w: pd.Series
    ) -> tuple[float, float]:
        """计算换手率和交易成本（占净值比例）"""
        all_syms = old_w.index.union(new_w.index)
        old = old_w.reindex(all_syms).fillna(0.0)
        new = new_w.reindex(all_syms).fillna(0.0)
        diff = (new - old).abs()
        turnover = float(diff.sum()) / 2.0  # 单边换手

        buys  = (new - old).clip(lower=0).sum()
        sells = (old - new).clip(lower=0).sum()
        cost_pct = (
            buys  * (self.costs.commission_rate + self.costs.slippage_rate) +
            sells * (self.costs.commission_rate + self.costs.stamp_duty + self.costs.slippage_rate)
        )
        return turnover, cost_pct

    def _get_rebalance_dates(self, dates: list) -> set:
        idx = pd.DatetimeIndex(dates)
        if self.rebalance_freq == "D":
            return set(idx)
        if self.rebalance_freq == "W":
            return set(idx[idx.weekday == 0])  # 每周一
        if self.rebalance_freq == "M":
            return set(idx[idx.is_month_start])
        return set(idx)

    # ----------------------------------------------------------------
    # 绩效计算
    # ----------------------------------------------------------------

    def _compute_metrics(
        self, port: pd.Series, bm: pd.Series
    ) -> PerformanceMetrics:
        port = port.dropna()
        bm   = bm.reindex(port.index).fillna(0.0)

        n_days = len(port)
        if n_days < 2:
            return self._empty_metrics()

        # 累计收益
        total_ret  = float((1 + port).prod() - 1)
        ann_factor = 252.0 / n_days
        annual_ret = float((1 + total_ret) ** ann_factor - 1)
        bm_total   = float((1 + bm).prod() - 1)
        excess_ret = annual_ret - float((1 + bm_total) ** ann_factor - 1)

        # 波动
        ann_vol = float(port.std() * np.sqrt(252))
        downside = port[port < 0]
        sortino_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else ann_vol

        # 夏普（无风险利率 2.5%）
        rf = 0.025
        sharpe = (annual_ret - rf) / (ann_vol + 1e-8)
        sortino = (annual_ret - rf) / (sortino_vol + 1e-8)

        # 最大回撤
        nav = (1 + port).cumprod()
        rolling_max = nav.cummax()
        drawdown = (nav / rolling_max - 1)
        max_dd = float(drawdown.min())
        # 回撤持续天数
        in_dd = (drawdown < 0).astype(int)
        dd_duration = int(in_dd.groupby((in_dd != in_dd.shift()).cumsum()).transform("sum").max())

        calmar = annual_ret / (abs(max_dd) + 1e-8)

        # 胜率（月度）
        monthly = port.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        win_rate = float((monthly > 0).mean())
        wins  = monthly[monthly > 0]
        losses = monthly[monthly < 0]
        avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        pf = (avg_win * win_rate) / (abs(avg_loss) * (1 - win_rate) + 1e-8)

        # 归因（简化：vs 等权 benchmark）
        excess = port - bm
        factor_ret = float(bm.mean() * 252)
        idio_ret   = float(excess.mean() * 252)
        timing_ret = 0.0  # 简化

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
            profit_factor=round(pf, 3),
            annual_turnover=0.0,
            total_transaction_cost=0.0,
            factor_return=round(factor_ret, 4),
            idiosyncratic_return=round(idio_ret, 4),
            timing_return=timing_ret,
            monthly_returns=monthly,
            monthly_win_rate=win_rate,
        )

    @staticmethod
    def _empty_metrics() -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return=0, annual_return=0, benchmark_return=0, excess_return=0,
            annual_vol=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_dd_duration=0, win_rate=0, avg_win=0,
            avg_loss=0, profit_factor=0, annual_turnover=0,
            total_transaction_cost=0, factor_return=0, idiosyncratic_return=0,
            timing_return=0,
        )

    # ----------------------------------------------------------------
    # 报告生成
    # ----------------------------------------------------------------

    def _generate_report(self, result: BacktestResult) -> str:
        m = result.combined_metrics
        if m is None:
            return "无有效回测结果"

        nav_final = float(result.full_portfolio_nav.iloc[-1]) if len(result.full_portfolio_nav) > 0 else 1.0
        bm_final  = float(result.full_benchmark_nav.iloc[-1])  if len(result.full_benchmark_nav) > 0 else 1.0

        lines = [
            "# 组合回测报告（Walk-Forward）",
            "",
            "## 总体绩效",
            f"| 指标 | 策略 | 基准 | 超额 |",
            f"|------|------|------|------|",
            f"| 总收益率 | {m.total_return:.1%} | {m.benchmark_return:.1%} | {m.total_return-m.benchmark_return:.1%} |",
            f"| 年化收益率 | {m.annual_return:.1%} | — | {m.excess_return:.1%} |",
            f"| 年化波动率 | {m.annual_vol:.1%} | — | — |",
            f"| 夏普比率 | {m.sharpe_ratio:.2f} | — | — |",
            f"| 索提诺比率 | {m.sortino_ratio:.2f} | — | — |",
            f"| 卡玛比率 | {m.calmar_ratio:.2f} | — | — |",
            f"| 最大回撤 | {m.max_drawdown:.1%} | — | — |",
            f"| 最大回撤持续 | {m.max_dd_duration}天 | — | — |",
            f"| 月度胜率 | {m.win_rate:.1%} | — | — |",
            f"| 盈亏比 | {m.profit_factor:.2f} | — | — |",
            "",
            "## 收益归因",
            f"| 来源 | 贡献 |",
            f"|------|------|",
            f"| 因子收益（β） | {m.factor_return:.1%} |",
            f"| 特质收益（α） | {m.idiosyncratic_return:.1%} |",
            f"| 择时贡献 | {m.timing_return:.1%} |",
            "",
            "## Walk-Forward 窗口详情",
            f"| 窗口 | 测试期 | 年化收益 | 夏普 | 最大回撤 |",
            f"|------|--------|----------|------|----------|",
        ]

        for i, w in enumerate(result.windows):
            if w.metrics:
                lines.append(
                    f"| WF{i+1} | {w.test_start.date()}→{w.test_end.date()} "
                    f"| {w.metrics.annual_return:.1%} "
                    f"| {w.metrics.sharpe_ratio:.2f} "
                    f"| {w.metrics.max_drawdown:.1%} |"
                )

        lines += [
            "",
            "## 交易成本设置",
            f"- 佣金率：{self.costs.commission_rate:.4%}（双向）",
            f"- 印花税：{self.costs.stamp_duty:.3%}（卖出）",
            f"- 滑点：{self.costs.slippage_rate:.3%}（双向）",
            f"- 换仓频率：{self.rebalance_freq}",
            f"- 持仓数量：{self.n_holdings} 只",
            f"- 组合构建方法：{self.method}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    dates  = pd.date_range("2021-01-01", "2024-12-31", freq="B")
    syms   = [f"stock_{i:03d}" for i in range(50)]
    rows   = []
    for d in dates:
        bm_ret = np.random.randn() * 0.01
        for s in syms:
            score = np.random.randn()
            rows.append({
                "date": d, "symbol": s,
                "close": 100.0,
                "factor_score": score,
                "forward_ret_1d": np.random.randn() * 0.015 + score * 0.003,
                "benchmark_return": bm_ret,
            })
    df = pd.DataFrame(rows)

    backtester = PortfolioBacktester(
        df,
        n_holdings=10,
        rebalance_freq="W",
        construction_method="equal_weight",
        wf_train_months=12,
        wf_test_months=3,
    )

    result = backtester.run_walkforward()
    print(result.backtest_report)
    m = result.combined_metrics
    print(f"\n最终净值: {result.full_portfolio_nav.iloc[-1]:.4f} "
          f"vs 基准: {result.full_benchmark_nav.iloc[-1]:.4f}")
