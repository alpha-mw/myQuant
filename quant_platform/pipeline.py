from __future__ import annotations

"""Orchestrates data, backtest, optimization, reporting, and packaging."""

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Dict, List

import pandas as pd

from quant_platform.config import AppConfig
from quant_platform.data.data_manager import DataManager
from quant_platform.data.tushare_client import build_tushare_client
from quant_platform.optimizer import optimize_weights
from quant_platform.report.report_generator import ReportData, generate_report
from quant_platform.strategy.backtest import BacktestConfig, BacktestEngine
from quant_platform.strategy.factors import FactorInputs, build_factor_scores, compute_rebalance_dates
from quant_platform.strategy.selection import select_top_n
from quant_platform.utils.packaging import package_outputs
from quant_platform.utils.performance import compute_metrics


@dataclass
class PipelineOutputs:
    report_path: Path
    zip_path: Path
    best_weight: float


def run_pipeline(config: AppConfig) -> PipelineOutputs:
    tushare_client = build_tushare_client(config.data.tushare_token)
    data_manager = DataManager(config.data.data_dir, tushare_client)
    bundle = data_manager.load_or_fetch(config.data.start_date, config.data.end_date, config.data.benchmark_code)

    inputs = FactorInputs(prices=bundle.prices, market_caps=bundle.market_caps, industries=bundle.industries)
    optimization = optimize_weights(
        inputs=inputs,
        price_data=bundle.prices,
        benchmark=bundle.benchmark,
        top_n=config.strategy.top_n,
        max_position_pct=config.strategy.max_position_pct,
        daily_stop_loss=config.strategy.daily_stop_loss,
        risk_free_rate=config.strategy.risk_free_rate,
        initial_weight=config.strategy.initial_weight_momentum,
        step=config.strategy.step,
        max_iterations=config.strategy.max_iterations,
    )

    scores, _, _ = build_factor_scores(inputs, optimization.best_weight)
    rebalance_dates = compute_rebalance_dates(scores)
    selection = select_top_n(scores, rebalance_dates, config.strategy.top_n)

    engine = BacktestEngine(price_data=bundle.prices, benchmark=bundle.benchmark)
    result = engine.run(
        holdings=selection.holdings,
        rebalance_dates=rebalance_dates,
        config=BacktestConfig(
            top_n=config.strategy.top_n,
            max_position_pct=config.strategy.max_position_pct,
            daily_stop_loss=config.strategy.daily_stop_loss,
        ),
    )

    initial_scores, _, _ = build_factor_scores(inputs, config.strategy.initial_weight_momentum)
    initial_rebalance = compute_rebalance_dates(initial_scores)
    initial_selection = select_top_n(initial_scores, initial_rebalance, config.strategy.top_n)
    initial_result = engine.run(
        holdings=initial_selection.holdings,
        rebalance_dates=initial_rebalance,
        config=BacktestConfig(
            top_n=config.strategy.top_n,
            max_position_pct=config.strategy.max_position_pct,
            daily_stop_loss=config.strategy.daily_stop_loss,
        ),
    )

    initial_metrics = compute_metrics(initial_result.portfolio_values, initial_result.trades, config.strategy.risk_free_rate)
    best_metrics = compute_metrics(result.portfolio_values, result.trades, config.strategy.risk_free_rate)

    benchmark = bundle.benchmark.copy()
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark_series = benchmark.set_index("trade_date")["close"].sort_index()
    drawdown = _compute_drawdown(result.portfolio_values)

    holdings_distribution = _build_holdings_distribution(selection.holdings, bundle.industries)

    weight_history = [
        {"iteration": idx + 1, "weight_momentum": record.weight_momentum}
        for idx, record in enumerate(optimization.history)
    ]
    notes = []
    if config.data.tushare_token is None:
        notes.append("- Tushare Token 未配置，需使用 data/hs300_members.csv 与行情缓存文件，或切换 iFinD。")

    report_data = ReportData(
        initial_metrics=initial_metrics,
        best_metrics=best_metrics,
        portfolio_values=result.portfolio_values,
        benchmark=benchmark_series,
        drawdown=drawdown,
        weight_history=weight_history,
        holdings_distribution=holdings_distribution,
        notes=notes,
    )

    output_dir = config.report.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "backtest_report.pdf"
    generate_report(report_data, report_path)

    _collect_artifacts(output_dir)
    zip_path = package_outputs(output_dir)
    return PipelineOutputs(report_path=report_path, zip_path=zip_path, best_weight=optimization.best_weight)


def _compute_drawdown(values: pd.Series) -> pd.Series:
    cumulative = (1 + values.pct_change().fillna(0)).cumprod()
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


def _build_holdings_distribution(holdings: Dict[pd.Timestamp, List[str]], industries: pd.DataFrame) -> pd.Series:
    if not holdings:
        return pd.Series(dtype=int)
    latest_date = max(holdings.keys())
    latest_holdings = holdings[latest_date]
    industry_map = industries.set_index("ts_code")["industry"].to_dict()
    series = pd.Series([industry_map.get(code, "Unknown") for code in latest_holdings])
    return series.value_counts()


def _collect_artifacts(output_dir: Path) -> None:
    artifacts = [
        Path("requirements.txt"),
        Path("docs/deployment_manual.md"),
        Path("scripts/run_pipeline.py"),
        Path("config_template.yaml"),
    ]
    for artifact in artifacts:
        if artifact.exists():
            shutil.copy(artifact, output_dir / artifact.name)
    package_dir = Path("quant_platform")
    if package_dir.exists():
        target = output_dir / package_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(package_dir, target)
