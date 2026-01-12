from __future__ import annotations

"""Backtrader strategy implementation with risk control and rebalancing."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import backtrader as bt
import pandas as pd


class StockData(bt.feeds.PandasData):
    lines = ("amount",)
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "vol"),
        ("amount", "amount"),
    )


class CommissionInfo(bt.CommInfoBase):
    params = (
        ("commission", 0.0005),
        ("stamp_duty", 0.00001),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        commission = abs(size) * price * self.p.commission
        stamp = abs(size) * price * self.p.stamp_duty if size < 0 else 0.0
        return commission + stamp


@dataclass
class BacktestConfig:
    top_n: int
    max_position_pct: float
    daily_stop_loss: float


@dataclass
class BacktestResult:
    portfolio_values: pd.Series
    trades: List[Dict[str, float]]


class MomentumVolatilityStrategy(bt.Strategy):
    params = (
        ("rebalance_dates", []),
        ("holdings", {}),
        ("max_position_pct", 0.1),
        ("daily_stop_loss", 0.02),
    )

    def __init__(self) -> None:
        self.rebalance_dates = set(pd.to_datetime(self.p.rebalance_dates))
        self.holdings = self.p.holdings
        self.order_refs = {}
        self.trades = []

    def next(self) -> None:
        current_date = self.datas[0].datetime.date(0)
        current_ts = pd.Timestamp(current_date)
        self._apply_stop_loss()
        if current_ts in self.rebalance_dates:
            self._rebalance(current_ts)

    def _rebalance(self, current_ts: pd.Timestamp) -> None:
        target = set(self.holdings.get(current_ts, []))
        current = {d._name for d in self.datas if self.getposition(d).size > 0}

        for data in self.datas:
            if data._name in current and data._name not in target:
                self.close(data=data)

        cash = self.broker.getcash()
        if not target:
            return
        max_value = self.broker.getvalue() * self.p.max_position_pct
        target_value = min(cash / len(target), max_value)

        for data in self.datas:
            if data._name in target:
                price = data.close[0]
                size = int(target_value / price)
                if size > 0:
                    self.order_target_size(data=data, target=size)

    def _apply_stop_loss(self) -> None:
        for data in self.datas:
            position = self.getposition(data)
            if position.size == 0:
                continue
            if len(data.close) < 2:
                continue
            prev_close = data.close[-1]
            if prev_close == 0:
                continue
            daily_return = data.close[0] / prev_close - 1
            if daily_return <= -self.p.daily_stop_loss:
                self.close(data=data)

    def notify_trade(self, trade) -> None:
        if trade.isclosed:
            self.trades.append({
                "pnl": trade.pnl,
                "pnl_comm": trade.pnlcomm,
                "size": trade.size,
            })


class PortfolioValueAnalyzer(bt.Analyzer):
    def __init__(self) -> None:
        self.values = []
        self.timestamps = []

    def next(self) -> None:
        self.values.append(self.strategy.broker.getvalue())
        self.timestamps.append(self.strategy.datas[0].datetime.date(0))

    def get_series(self) -> pd.Series:
        return pd.Series(self.values, index=pd.to_datetime(self.timestamps))


class BacktestEngine:
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        benchmark: Optional[pd.DataFrame] = None,
    ) -> None:
        self.price_data = price_data
        self.benchmark = benchmark

    def run(
        self,
        holdings: Dict[pd.Timestamp, List[str]],
        rebalance_dates: pd.DatetimeIndex,
        config: BacktestConfig,
    ) -> BacktestResult:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(1_000_000.0)
        cerebro.broker.addcommissioninfo(CommissionInfo())

        for ts_code, data in self.price_data.items():
            frame = data.copy()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            frame = frame.sort_values("trade_date").set_index("trade_date")
            frame = frame.rename(columns={"vol": "vol", "amount": "amount"})
            feed = StockData(dataname=frame, name=ts_code)
            cerebro.adddata(feed)

        cerebro.addanalyzer(PortfolioValueAnalyzer, _name="portfolio_values")
        cerebro.addstrategy(
            MomentumVolatilityStrategy,
            rebalance_dates=rebalance_dates,
            holdings=holdings,
            max_position_pct=config.max_position_pct,
            daily_stop_loss=config.daily_stop_loss,
        )

        results = cerebro.run()
        strategy = results[0]
        analyzer = strategy.analyzers.getbyname("portfolio_values")
        portfolio_values = analyzer.get_series()
        return BacktestResult(portfolio_values=portfolio_values, trades=strategy.trades)
