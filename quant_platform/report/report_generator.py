from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from quant_platform.utils.performance import PerformanceMetrics


@dataclass
class ReportData:
    initial_metrics: PerformanceMetrics
    best_metrics: PerformanceMetrics
    portfolio_values: pd.Series
    benchmark: pd.Series
    drawdown: pd.Series
    weight_history: List[Dict[str, float]]
    holdings_distribution: pd.Series
    notes: List[str]


def generate_report(report_data: ReportData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _add_summary_page(report_data, pdf)
        _add_performance_charts(report_data, pdf)
        _add_weight_history(report_data, pdf)
        _add_holdings_distribution(report_data, pdf)


def _add_summary_page(report_data: ReportData, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    text_lines = [
        "沪深300 双因子策略回测报告",
        "",
        "策略逻辑：动量 + 波动率双因子，行业/市值中性化，月度调仓，等权配置。",
        "",
        "初始权重表现：",
        _format_metrics(report_data.initial_metrics),
        "",
        "最优权重表现：",
        _format_metrics(report_data.best_metrics),
        "",
        "风险与实盘注意事项：",
        "1. 实盘接入需配置交易接口（如同花顺模拟/券商CTP），并验证订单路由。",
        "2. 数据源需要每日更新，Tushare Token 请妥善保存。",
        "3. 风控已限制单票最大仓位10%，触发日内止损2%。",
        "",
        "备注：",
    ]
    text_lines.extend(report_data.notes)
    ax.text(0.05, 0.95, "\n".join(text_lines), va="top", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def _add_performance_charts(report_data: ReportData, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    portfolio = report_data.portfolio_values / report_data.portfolio_values.iloc[0]
    benchmark = report_data.benchmark / report_data.benchmark.iloc[0]

    axes[0].plot(portfolio.index, portfolio.values, label="策略净值")
    axes[0].plot(benchmark.index, benchmark.values, label="沪深300")
    axes[0].set_title("收益曲线")
    axes[0].legend()

    axes[1].plot(report_data.drawdown.index, report_data.drawdown.values, color="red")
    axes[1].set_title("回撤曲线")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    pdf.savefig(fig)
    plt.close(fig)


def _add_weight_history(report_data: ReportData, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 5.0))
    df = pd.DataFrame(report_data.weight_history)
    if not df.empty:
        ax.plot(df["iteration"], df["weight_momentum"], marker="o")
    ax.set_title("因子权重迭代过程")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Momentum Weight")
    pdf.savefig(fig)
    plt.close(fig)


def _add_holdings_distribution(report_data: ReportData, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 5.0))
    report_data.holdings_distribution.plot(kind="bar", ax=ax)
    ax.set_title("行业持仓分布")
    ax.set_xlabel("Industry")
    ax.set_ylabel("Count")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _format_metrics(metrics: PerformanceMetrics) -> str:
    return (
        f"年化收益: {metrics.annual_return:.2%}\n"
        f"最大回撤: {metrics.max_drawdown:.2%}\n"
        f"夏普比率: {metrics.sharpe:.2f}\n"
        f"胜率: {metrics.win_rate:.2%}\n"
        f"盈亏比: {metrics.profit_loss_ratio:.2f}\n"
        f"持仓天数: {metrics.holding_days:.0f}"
    )
