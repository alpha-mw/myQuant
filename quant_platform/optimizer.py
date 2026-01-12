from __future__ import annotations

"""Optimization loop for factor weights."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from quant_platform.strategy.backtest import BacktestConfig, BacktestEngine
from quant_platform.strategy.factors import FactorInputs, build_factor_scores, compute_rebalance_dates
from quant_platform.strategy.selection import select_top_n
from quant_platform.utils.performance import PerformanceMetrics, compute_metrics


@dataclass
class OptimizationRecord:
    weight_momentum: float
    metrics: PerformanceMetrics


@dataclass
class OptimizationResult:
    best_weight: float
    best_metrics: PerformanceMetrics
    history: List[OptimizationRecord]


def optimize_weights(
    inputs: FactorInputs,
    price_data: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    top_n: int,
    max_position_pct: float,
    daily_stop_loss: float,
    risk_free_rate: float,
    initial_weight: float,
    step: float,
    max_iterations: int,
) -> OptimizationResult:
    history: List[OptimizationRecord] = []
    current_weight = initial_weight
    best_record: OptimizationRecord | None = None

    for _ in range(max_iterations):
        record = _evaluate_weight(
            inputs,
            price_data,
            benchmark,
            top_n,
            max_position_pct,
            daily_stop_loss,
            risk_free_rate,
            current_weight,
        )
        history.append(record)
        if best_record is None or record.metrics.sharpe > best_record.metrics.sharpe:
            best_record = record

        if _meets_targets(record.metrics):
            return OptimizationResult(best_weight=current_weight, best_metrics=record.metrics, history=history)

        candidates = [w for w in [current_weight + step, current_weight - step] if 0 <= w <= 1]
        candidate_records = [
            _evaluate_weight(
                inputs,
                price_data,
                benchmark,
                top_n,
                max_position_pct,
                daily_stop_loss,
                risk_free_rate,
                w,
            )
            for w in candidates
        ]
        history.extend(candidate_records)
        if candidate_records:
            candidate_records.sort(key=lambda x: (x.metrics.annual_return, -x.metrics.max_drawdown, x.metrics.sharpe), reverse=True)
            current_weight = candidate_records[0].weight_momentum
        else:
            break

    if not best_record:
        raise RuntimeError("Optimization failed to produce any result.")
    return OptimizationResult(best_weight=best_record.weight_momentum, best_metrics=best_record.metrics, history=history)


def _evaluate_weight(
    inputs: FactorInputs,
    price_data: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    top_n: int,
    max_position_pct: float,
    daily_stop_loss: float,
    risk_free_rate: float,
    weight: float,
) -> OptimizationRecord:
    scores, _, _ = build_factor_scores(inputs, weight)
    rebalance_dates = compute_rebalance_dates(scores)
    selection = select_top_n(scores, rebalance_dates, top_n)

    engine = BacktestEngine(price_data=price_data, benchmark=benchmark)
    result = engine.run(
        holdings=selection.holdings,
        rebalance_dates=rebalance_dates,
        config=BacktestConfig(top_n=top_n, max_position_pct=max_position_pct, daily_stop_loss=daily_stop_loss),
    )
    metrics = compute_metrics(result.portfolio_values, result.trades, risk_free_rate)
    return OptimizationRecord(weight_momentum=weight, metrics=metrics)


def _meets_targets(metrics: PerformanceMetrics) -> bool:
    return metrics.annual_return >= 0.15 and metrics.max_drawdown >= -0.10


def build_weight_grid(step: float) -> list[float]:
    return [round(x, 2) for x in np.arange(0, 1 + step, step)]
