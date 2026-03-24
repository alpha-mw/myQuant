"""
Quant Investor V8.0 legacy frozen 入口。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from quant_investor.branch_contracts import PortfolioStrategy, ResearchPipelineResult, UnifiedDataBundle
from quant_investor.logger import get_logger
from quant_investor.pipeline.legacy_v8_pipeline import LegacyV8ParallelResearchPipeline
from quant_investor.versioning import (
    ARCHITECTURE_VERSION_V8,
    BRANCH_SCHEMA_VERSION_V8,
    CALIBRATION_SCHEMA_VERSION,
    DEBATE_TEMPLATE_VERSION,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)

_logger = get_logger("QuantInvestorV8")


@dataclass
class V8PipelineResult:
    """V8 legacy frozen 完整流水线结果。"""

    architecture_version: str = ARCHITECTURE_VERSION_V8
    branch_schema_version: str = BRANCH_SCHEMA_VERSION_V8
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    report_protocol_version: str = REPORT_PROTOCOL_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    debate_template_version: str = DEBATE_TEMPLATE_VERSION
    data_bundle: Optional[UnifiedDataBundle] = None
    branch_results: dict[str, Any] = field(default_factory=dict)
    calibrated_signals: dict[str, Any] = field(default_factory=dict)
    risk_results: Any = None
    final_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    final_report: str = ""
    execution_log: list[str] = field(default_factory=list)
    layer_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0

    raw_data: dict[str, Any] = field(default_factory=dict)
    factor_data: dict[str, Any] = field(default_factory=dict)
    model_predictions: dict[str, Any] = field(default_factory=dict)
    macro_signal: str = "🟡"
    macro_summary: str = ""
    v7_decision: Any = None
    llm_ensemble_results: dict[str, Any] = field(default_factory=dict)


class QuantInvestorV8:
    """V8 legacy frozen 入口。"""

    def __init__(
        self,
        stock_pool: list[str],
        market: str = "CN",
        lookback_years: float = 1.0,
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        enable_macro: bool = True,
        enable_backtest: bool = False,
        enable_alpha_mining: bool = True,
        enable_kline: bool = True,
        enable_intelligence: bool = True,
        enable_llm_debate: bool = True,
        kline_backend: str = "heuristic",
        allow_synthetic_for_research: bool = False,
        verbose: bool = True,
        enable_kronos: bool | None = None,
        enable_branch_debate: bool | None = None,
    ) -> None:
        self.stock_pool = stock_pool
        self.market = market
        self.lookback_years = lookback_years
        self.total_capital = total_capital
        self.risk_level = risk_level
        self.enable_macro = enable_macro
        self.enable_backtest = enable_backtest
        self.enable_alpha_mining = enable_alpha_mining
        self.enable_kline = enable_kline if enable_kronos is None else enable_kronos
        self.kline_backend = kline_backend
        self.enable_intelligence = enable_intelligence
        self.enable_llm_debate = enable_llm_debate
        self.enable_branch_debate = False if enable_branch_debate is None else bool(enable_branch_debate)
        self.allow_synthetic_for_research = allow_synthetic_for_research
        self.verbose = verbose
        self.result = V8PipelineResult()

    def _log(self, msg: str) -> None:
        self.result.execution_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if self.verbose:
            _logger.info(msg)

    def run(self) -> V8PipelineResult:
        t0 = time.time()
        self._log("=" * 60)
        self._log("🚀 Quant Investor V8.0 启动（legacy frozen）")
        self._log(f"标的: {self.stock_pool}")
        self._log(f"市场: {self.market}  资金: ¥{self.total_capital:,.0f}")
        self._log("=" * 60)

        pipeline = LegacyV8ParallelResearchPipeline(
            stock_pool=self.stock_pool,
            market=self.market,
            lookback_years=self.lookback_years,
            total_capital=self.total_capital,
            risk_level=self.risk_level,
            enable_alpha_mining=self.enable_alpha_mining,
            enable_kline=self.enable_kline,
            kline_backend=self.kline_backend,
            enable_intelligence=self.enable_intelligence,
            enable_llm_debate=self.enable_llm_debate,
            enable_macro=self.enable_macro,
            allow_synthetic_for_research=self.allow_synthetic_for_research,
            verbose=self.verbose,
        )
        pipeline_result = pipeline.run()
        self._populate_result(pipeline_result)

        self.result.total_time = time.time() - t0
        self._log(f"✅ V8 分析完成，总耗时 {self.result.total_time:.1f}s")
        return self.result

    def _populate_result(self, pipeline_result: ResearchPipelineResult) -> None:
        self.result.architecture_version = getattr(
            pipeline_result,
            "architecture_version",
            self.result.architecture_version,
        )
        self.result.branch_schema_version = getattr(
            pipeline_result,
            "branch_schema_version",
            self.result.branch_schema_version,
        )
        self.result.ic_protocol_version = getattr(
            pipeline_result,
            "ic_protocol_version",
            self.result.ic_protocol_version,
        )
        self.result.report_protocol_version = getattr(
            pipeline_result,
            "report_protocol_version",
            self.result.report_protocol_version,
        )
        self.result.calibration_schema_version = getattr(
            pipeline_result,
            "calibration_schema_version",
            self.result.calibration_schema_version,
        )
        self.result.debate_template_version = getattr(
            pipeline_result,
            "debate_template_version",
            self.result.debate_template_version,
        )
        self.result.data_bundle = pipeline_result.data_bundle
        self.result.branch_results = pipeline_result.branch_results
        self.result.calibrated_signals = pipeline_result.calibrated_signals
        self.result.risk_results = pipeline_result.risk_result
        self.result.final_strategy = pipeline_result.final_strategy
        self.result.final_report = pipeline_result.final_report
        self.result.layer_timings = pipeline_result.timings
        self.result.execution_log.extend(pipeline_result.execution_log)

        self.result.raw_data = pipeline_result.data_bundle.symbol_data
        quant_branch = pipeline_result.branch_results.get("quant")
        kline_branch = pipeline_result.branch_results.get("kline")
        llm_branch = pipeline_result.branch_results.get("llm_debate")
        macro_branch = pipeline_result.branch_results.get("macro")

        if quant_branch is not None:
            self.result.factor_data = quant_branch.signals
        if kline_branch is not None:
            self.result.model_predictions = kline_branch.signals.get("predicted_return", {})
        if llm_branch is not None:
            self.result.llm_ensemble_results = llm_branch.signals
        if macro_branch is not None:
            self.result.macro_signal = str(macro_branch.signals.get("liquidity_signal", "🟡"))
            self.result.macro_summary = str(macro_branch.explanation)

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print(self.result.final_report)
        print("=" * 70)

    def save_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.result.final_report)
        _logger.info(f"报告已保存到: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Investor V8.0 legacy frozen. V9 为 current architecture，最新默认版本为 V10。"
    )
    parser.add_argument("--stocks", nargs="+", default=["000001.SZ", "600519.SH"])
    parser.add_argument("--market", default="CN", choices=["CN", "US"])
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="模拟资金（元）")
    parser.add_argument("--risk", default="中等", choices=["保守", "中等", "积极"])
    parser.add_argument("--lookback", type=float, default=1.0, help="数据回溯年数")
    parser.add_argument("--no-macro", action="store_true", help="关闭宏观分支")
    parser.add_argument("--no-kline", "--no-kronos", action="store_true", help="关闭 K 线分析分支")
    parser.add_argument(
        "--kline-backend",
        default="heuristic",
        choices=["heuristic", "kronos", "chronos"],
        help="V8 legacy 冻结后端集合。",
    )
    parser.add_argument("--no-intelligence", action="store_true", help="关闭多维智能融合分支")
    parser.add_argument("--no-llm-debate", action="store_true", help="关闭 legacy llm_debate 顶层分支")
    parser.add_argument("--with-backtest", action="store_true", help="保留兼容参数，当前不主动触发独立回测")
    parser.add_argument("--allow-synthetic-for-research", action="store_true")
    parser.add_argument("--output", default="", help="报告输出路径（默认打印到控制台）")
    args = parser.parse_args()

    investor = QuantInvestorV8(
        stock_pool=args.stocks,
        market=args.market,
        lookback_years=args.lookback,
        total_capital=args.capital,
        risk_level=args.risk,
        enable_macro=not args.no_macro,
        enable_backtest=args.with_backtest,
        enable_kline=not args.no_kline,
        kline_backend=args.kline_backend,
        enable_intelligence=not args.no_intelligence,
        enable_llm_debate=not args.no_llm_debate,
        allow_synthetic_for_research=args.allow_synthetic_for_research,
        verbose=True,
    )
    result = investor.run()

    if args.output:
        investor.save_report(args.output)
        print(f"\n报告已保存到: {args.output}")
    else:
        investor.print_report()

    print("\n⏱ 各阶段耗时：")
    for layer, seconds in result.layer_timings.items():
        print(f"  {layer}: {seconds:.1f}s")
    print(f"  总计: {result.total_time:.1f}s")


if __name__ == "__main__":
    main()
