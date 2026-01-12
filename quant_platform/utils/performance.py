from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    annual_return: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    profit_loss_ratio: float
    holding_days: float


def compute_metrics(portfolio_values: pd.Series, trades: list[Dict[str, float]], risk_free_rate: float) -> PerformanceMetrics:
    returns = portfolio_values.pct_change().dropna()
    if returns.empty:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    excess = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() != 0 else 0.0

    pnl_values = [trade["pnl_comm"] for trade in trades if trade["pnl_comm"] != 0]
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]
    win_rate = len(wins) / len(pnl_values) if pnl_values else 0.0
    profit_loss_ratio = abs(np.mean(wins) / np.mean(losses)) if wins and losses else 0.0

    holding_days = len(returns)

    return PerformanceMetrics(
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        holding_days=holding_days,
    )
