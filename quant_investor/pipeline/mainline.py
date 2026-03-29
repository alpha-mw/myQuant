"""
Quant-Investor single mainline.

Current mainline = deterministic research core + structured review layer +
single control chain (RiskGuard -> ICCoordinator -> PortfolioConstructor -> NarratorAgent).
"""

from __future__ import annotations

from copy import deepcopy
import time
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from quant_investor.agent_orchestrator import AgentOrchestrator as StructuredAgentOrchestrator
from quant_investor.agent_protocol import BranchVerdict, ICDecision, PortfolioPlan, ReportBundle
from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BaseBranchAgentOutput,
    MasterAgentOutput,
    RiskAgentOutput,
)
from quant_investor.agents.base import BaseAgent
from quant_investor.agents.llm_client import has_provider_for_model, resolve_default_model
from quant_investor.agents.orchestrator import AgentOrchestrator as ReviewAgentOrchestrator
from quant_investor.agents.prompts import format_agent_display_name
from quant_investor.branch_contracts import (
    LLMUsageRecord,
    LLMUsageSummary,
    PortfolioStrategy,
    TradeRecommendation,
    UnifiedDataBundle,
)
from quant_investor.config import config
from quant_investor.llm_gateway import (
    current_usage_session_id,
    end_usage_session,
    snapshot_usage,
    start_usage_session,
    update_usage_markdown,
)
from quant_investor.logger import get_logger
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline
from quant_investor.versioning import (
    AGENT_SCHEMA_VERSION,
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)

_logger = get_logger("QuantInvestor")

_CONVICTION_SCORE = {
    "strong_buy": 0.80,
    "buy": 0.35,
    "neutral": 0.0,
    "sell": -0.35,
    "strong_sell": -0.80,
}


@dataclass
class QuantInvestorPipelineResult:
    """Single-mainline pipeline result with baseline and review artifacts."""

    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    report_protocol_version: str = REPORT_PROTOCOL_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    data_bundle: Optional[UnifiedDataBundle] = None
    branch_results: dict[str, Any] = field(default_factory=dict)
    calibrated_signals: dict[str, Any] = field(default_factory=dict)
    risk_results: Any = None
    final_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    final_report: str = ""
    execution_log: list[str] = field(default_factory=list)
    layer_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    agent_orchestration: Optional[dict[str, Any]] = None
    agent_portfolio_plan: Any = None
    agent_report_bundle: Any = None
    agent_ic_decisions: Any = field(default_factory=dict)
    llm_usage_records: list[LLMUsageRecord] = field(default_factory=list)
    llm_usage_summary: LLMUsageSummary = field(default_factory=LLMUsageSummary)
    llm_usage_session_id: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)
    factor_data: dict[str, Any] = field(default_factory=dict)
    model_predictions: dict[str, Any] = field(default_factory=dict)
    macro_signal: str = "🟡"
    macro_summary: str = ""
    llm_ensemble_results: dict[str, Any] = field(default_factory=dict)
    baseline_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    baseline_risk_result: Any = None
    macro_verdict: BranchVerdict | None = None
    reviewed_research_by_symbol: dict[str, dict[str, BranchVerdict]] = field(default_factory=dict)
    reviewed_branch_summaries: dict[str, BranchVerdict] = field(default_factory=dict)
    branch_review_outputs: dict[str, BaseBranchAgentOutput | None] = field(default_factory=dict)
    master_review_output: MasterAgentOutput | None = None
    risk_review_output: RiskAgentOutput | None = None
    agent_layer_enabled: bool = False
    agent_schema_version: str = AGENT_SCHEMA_VERSION


@dataclass
class ResearchCoreSnapshot:
    data_bundle: UnifiedDataBundle
    branch_results: dict[str, Any] = field(default_factory=dict)
    calibrated_signals: dict[str, Any] = field(default_factory=dict)
    risk_result: Any = None
    baseline_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    market_regime: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    execution_log: list[str] = field(default_factory=list)
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    llm_usage_session_id: str = ""


class QuantInvestor:
    """Single supported public mainline."""

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
        enable_quant: bool = True,
        enable_kline: bool = True,
        enable_fundamental: bool = True,
        enable_intelligence: bool = True,
        kline_backend: str = "hybrid",
        allow_synthetic_for_research: bool = False,
        enable_document_semantics: bool = True,
        verbose: bool = True,
        enable_kronos: bool | None = None,
        enable_agent_layer: bool = True,
        agent_model: str = "claude-sonnet-4-6",
        master_model: str = "gpt-5.4-mini",
        agent_timeout: float = 15.0,
        master_timeout: float = 30.0,
        agent_total_timeout: float = 120.0,
        recall_context: dict[str, Any] | None = None,
    ) -> None:
        self.stock_pool = stock_pool
        self.market = market
        self.lookback_years = lookback_years
        self.total_capital = total_capital
        self.risk_level = risk_level
        self.enable_macro = enable_macro
        self.enable_backtest = enable_backtest
        self.enable_alpha_mining = enable_alpha_mining
        self.enable_quant = enable_quant
        self.enable_kline = enable_kline if enable_kronos is None else enable_kronos
        self.kline_backend = kline_backend
        self.enable_fundamental = enable_fundamental
        self.enable_intelligence = enable_intelligence
        self.allow_synthetic_for_research = allow_synthetic_for_research
        self.enable_document_semantics = enable_document_semantics
        self.verbose = verbose
        self.enable_agent_layer = enable_agent_layer
        self.agent_model, self.master_model = self._resolve_agent_models(
            agent_model=agent_model,
            master_model=master_model,
        )
        self.agent_timeout = agent_timeout
        self.master_timeout = master_timeout
        self.agent_total_timeout = agent_total_timeout
        self.recall_context = dict(recall_context or {})
        self.result = QuantInvestorPipelineResult()

    def _log(self, message: str) -> None:
        self.result.execution_log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if self.verbose:
            _logger.info(message)

    @staticmethod
    def _resolve_agent_models(
        agent_model: str,
        master_model: str,
    ) -> tuple[str, str]:
        return (
            str(agent_model or "").strip() or resolve_default_model(preferred_model="claude-sonnet-4-6"),
            str(master_model or "").strip() or resolve_default_model(preferred_model="gpt-5.4-mini"),
        )

    def run(self) -> QuantInvestorPipelineResult:
        t0 = time.time()
        managed_session = None
        if current_usage_session_id() is None:
            managed_session = start_usage_session(label="mainline")

        try:
            self._log("=" * 60)
            self._log("🚀 Quant-Investor 主线启动")
            self._log(f"标的: {self.stock_pool}")
            self._log(f"市场: {self.market}  资金: ¥{self.total_capital:,.0f}")
            self._log(f"Review Layer: {'启用' if self.enable_agent_layer else '禁用'}")
            self._log("=" * 60)

            if hasattr(config, "validate_runtime"):
                validation = config.validate_runtime(
                    market=self.market,
                    enable_agent_layer=self.enable_agent_layer,
                    agent_model=self.agent_model,
                    master_model=self.master_model,
                    kline_backend=self.kline_backend,
                )
                for message in validation["errors"]:
                    self._log(f"⚠️ {message}")
                for message in validation["warnings"]:
                    self._log(f"⚠️ {message}")

            snapshot = self._run_research_core()
            review_result, agent_layer_enabled = self._run_review_layer(snapshot)
            orchestration = self._run_unified_control_chain(snapshot, review_result)
            records, summary = snapshot_usage(current_usage_session_id() or snapshot.llm_usage_session_id)
            result = self._build_result(
                snapshot=snapshot,
                orchestration=orchestration,
                review_result=review_result,
                agent_layer_enabled=agent_layer_enabled,
                usage_records=records,
                usage_summary=summary,
                total_time=time.time() - t0,
            )
            self.result = result
            if summary.call_count:
                self._log(
                    f"LLM 可观测性: calls={summary.call_count}, tokens={summary.total_tokens}, cost=${summary.estimated_cost_usd:.6f}"
                )
            self._log(f"✅ 主线分析完成，总耗时 {result.total_time:.1f}s")
            return result
        finally:
            if managed_session is not None:
                end_usage_session(managed_session)

    def _run_research_core(
        self,
    ) -> ResearchCoreSnapshot:
        pipeline = ParallelResearchPipeline(
            stock_pool=self.stock_pool,
            market=self.market,
            lookback_years=self.lookback_years,
            total_capital=self.total_capital,
            risk_level=self.risk_level,
            enable_alpha_mining=self.enable_alpha_mining,
            enable_quant=self.enable_quant,
            enable_kline=self.enable_kline,
            enable_fundamental=self.enable_fundamental,
            enable_intelligence=self.enable_intelligence,
            enable_macro=self.enable_macro,
            kline_backend=self.kline_backend,
            allow_synthetic_for_research=self.allow_synthetic_for_research,
            enable_document_semantics=self.enable_document_semantics,
            verbose=self.verbose,
        )
        pipeline_any = cast(Any, pipeline)
        pipeline_any.enable_agent_orchestrator_bridge = False
        if hasattr(pipeline_any, "run_research_core"):
            return cast(ResearchCoreSnapshot, pipeline_any.run_research_core())

        legacy_result = pipeline.run()
        macro_branch = getattr(legacy_result, "branch_results", {}).get("macro")
        market_regime = None
        if macro_branch is not None and isinstance(getattr(macro_branch, "signals", {}), dict):
            market_regime = (
                macro_branch.signals.get("regime")
                or macro_branch.signals.get("macro_regime")
                or macro_branch.signals.get("risk_level")
            )
        return ResearchCoreSnapshot(
            data_bundle=legacy_result.data_bundle,
            branch_results=getattr(legacy_result, "branch_results", {}),
            calibrated_signals=getattr(legacy_result, "calibrated_signals", {}),
            risk_result=getattr(legacy_result, "risk_result", None),
            baseline_strategy=getattr(legacy_result, "final_strategy", PortfolioStrategy()),
            market_regime=market_regime,
            timings=getattr(legacy_result, "timings", {}),
            execution_log=getattr(legacy_result, "execution_log", []),
            branch_schema_version=getattr(legacy_result, "branch_schema_version", BRANCH_SCHEMA_VERSION),
            calibration_schema_version=getattr(
                legacy_result,
                "calibration_schema_version",
                CALIBRATION_SCHEMA_VERSION,
            ),
            llm_usage_session_id=getattr(legacy_result, "llm_usage_session_id", ""),
        )

    def _review_layer_available(self) -> bool:
        if not self.enable_agent_layer:
            return False
        return has_provider_for_model(self.agent_model) and has_provider_for_model(self.master_model)

    def _run_review_layer(
        self,
        snapshot: ResearchCoreSnapshot,
    ) -> tuple[AgentEnhancedStrategy | None, bool]:
        if not self._review_layer_available():
            return None, False

        effective_total_timeout = ReviewAgentOrchestrator.compute_recommended_total_timeout(
            timeout_per_agent=self.agent_timeout,
            master_timeout=self.master_timeout,
            existing_total_timeout=self.agent_total_timeout,
        )
        orchestrator = ReviewAgentOrchestrator(
            branch_model=self.agent_model,
            master_model=self.master_model,
            timeout_per_agent=self.agent_timeout,
            master_timeout=self.master_timeout,
            total_timeout=effective_total_timeout,
        )
        market_regime = snapshot.market_regime or self._derive_market_regime(snapshot)
        review_result = orchestrator.enhance_sync(
            branch_results=snapshot.branch_results,
            calibrated_signals=snapshot.calibrated_signals,
            risk_result=snapshot.risk_result,
            ensemble_output=self._build_ensemble_output(snapshot.baseline_strategy),
            data_bundle=snapshot.data_bundle,
            market_regime=market_regime,
            algorithmic_strategy=snapshot.baseline_strategy,
            recall_context=self.recall_context,
        )
        enabled = any(output is not None for output in review_result.branch_agent_outputs.values())
        return review_result, enabled

    def _run_unified_control_chain(
        self,
        snapshot: ResearchCoreSnapshot,
        review_result: AgentEnhancedStrategy | None,
    ) -> dict[str, Any]:
        structured = StructuredAgentOrchestrator()
        constraints = self._build_structured_constraints(snapshot, review_result)
        tradability_snapshot = structured._normalize_tradability_snapshot(snapshot.data_bundle, None)
        existing_portfolio = {"current_weights": {}}

        if review_result and any(output is not None for output in review_result.branch_agent_outputs.values()):
            macro_verdict = structured._build_macro_verdict_from_branch_results(
                snapshot.branch_results,
                snapshot.data_bundle,
            )
            reviewed_research = structured._build_symbol_verdicts_from_branch_results(
                snapshot.branch_results,
                snapshot.data_bundle,
            )
            macro_verdict = self._apply_review_to_verdict(
                macro_verdict,
                review_result.branch_agent_outputs.get("macro"),
                symbol=None,
            )
            reviewed_by_symbol: dict[str, dict[str, BranchVerdict]] = {}
            for symbol, branch_map in reviewed_research.items():
                reviewed_by_symbol[symbol] = {}
                for branch_name, verdict in branch_map.items():
                    reviewed_by_symbol[symbol][branch_name] = self._apply_review_to_verdict(
                        verdict,
                        review_result.branch_agent_outputs.get(branch_name),
                        symbol=symbol,
                    )
            return structured.run_with_structured_research(
                data_bundle=snapshot.data_bundle,
                macro_verdict=macro_verdict,
                research_by_symbol=reviewed_by_symbol,
                constraints=constraints,
                existing_portfolio=existing_portfolio,
                tradability_snapshot=tradability_snapshot,
                ic_hints_by_symbol=self._build_ic_hints_by_symbol(
                    review_result.agent_strategy,
                ),
                persist_outputs=False,
            )

        return structured.run_with_precomputed_research(
            data_bundle=snapshot.data_bundle,
            branch_results=snapshot.branch_results,
            constraints=constraints,
            existing_portfolio=existing_portfolio,
            tradability_snapshot=tradability_snapshot,
            persist_outputs=False,
        )

    def _build_result(
        self,
        *,
        snapshot: ResearchCoreSnapshot,
        orchestration: dict[str, Any],
        review_result: AgentEnhancedStrategy | None,
        agent_layer_enabled: bool,
        usage_records: list[Any],
        usage_summary: Any,
        total_time: float,
    ) -> QuantInvestorPipelineResult:
        portfolio_plan = orchestration["portfolio_plan"]
        report_bundle = orchestration["report_bundle"]
        final_strategy = self._portfolio_plan_to_strategy(
            portfolio_plan=portfolio_plan,
            ic_by_symbol=orchestration["ic_by_symbol"],
            report_bundle=report_bundle,
        )
        final_report = update_usage_markdown(
            report_bundle.markdown_report,
            usage_summary,
            title="## LLM 可观测性（含 Review Layer）",
        )
        final_report = self._append_workspace_report_sections(
            base_report=final_report,
            snapshot=snapshot,
            review_result=review_result,
            portfolio_plan=portfolio_plan,
            final_strategy=final_strategy,
            report_bundle=report_bundle,
            execution_log=self._build_execution_log(snapshot, review_result, agent_layer_enabled),
            layer_timings=self._build_layer_timings(snapshot, review_result, total_time),
        )
        report_bundle.markdown_report = final_report
        branch_summaries = {
            key: verdict
            for key, verdict in report_bundle.branch_verdicts.items()
        }

        result = QuantInvestorPipelineResult(
            architecture_version=ARCHITECTURE_VERSION,
            branch_schema_version=snapshot.branch_schema_version,
            ic_protocol_version=report_bundle.ic_protocol_version,
            report_protocol_version=report_bundle.report_protocol_version,
            calibration_schema_version=snapshot.calibration_schema_version,
            data_bundle=snapshot.data_bundle,
            branch_results=snapshot.branch_results,
            calibrated_signals=snapshot.calibrated_signals,
            risk_results=orchestration["risk_by_symbol"],
            final_strategy=final_strategy,
            final_report=final_report,
            execution_log=self._build_execution_log(snapshot, review_result, agent_layer_enabled),
            layer_timings=self._build_layer_timings(snapshot, review_result, total_time),
            total_time=total_time,
            agent_orchestration=orchestration,
            agent_portfolio_plan=portfolio_plan,
            agent_report_bundle=report_bundle,
            agent_ic_decisions=orchestration["ic_by_symbol"],
            llm_usage_records=list(usage_records),
            llm_usage_summary=usage_summary,
            llm_usage_session_id=current_usage_session_id() or snapshot.llm_usage_session_id,
            raw_data=snapshot.data_bundle.symbol_data,
            factor_data=snapshot.branch_results.get("quant").signals if snapshot.branch_results.get("quant") else {},
            model_predictions=(
                snapshot.branch_results.get("kline").signals.get("predicted_return", {})
                if snapshot.branch_results.get("kline")
                else {}
            ),
            macro_signal=str(getattr(report_bundle.macro_verdict, "metadata", {}).get("policy_signal", "🟡")),
            macro_summary=report_bundle.macro_verdict.thesis if report_bundle.macro_verdict else "",
            baseline_strategy=snapshot.baseline_strategy,
            baseline_risk_result=snapshot.risk_result,
            macro_verdict=report_bundle.macro_verdict,
            reviewed_research_by_symbol=orchestration["research_by_symbol"],
            reviewed_branch_summaries=branch_summaries,
            branch_review_outputs=(
                dict(review_result.branch_agent_outputs)
                if review_result
                else {}
            ),
            master_review_output=review_result.agent_strategy if review_result else None,
            risk_review_output=review_result.risk_agent_output if review_result else None,
            agent_layer_enabled=agent_layer_enabled,
        )
        result.final_strategy.metadata["agent_layer_enabled"] = agent_layer_enabled
        provenance_summary = dict(getattr(result.final_strategy, "provenance_summary", {}))
        provenance_summary["agent_layer_enabled"] = agent_layer_enabled
        result.final_strategy.provenance_summary = provenance_summary
        return result

    def _append_workspace_report_sections(
        self,
        *,
        base_report: str,
        snapshot: Any,
        review_result: AgentEnhancedStrategy | None,
        portfolio_plan: PortfolioPlan,
        final_strategy: PortfolioStrategy,
        report_bundle: ReportBundle,
        execution_log: list[str],
        layer_timings: dict[str, float],
    ) -> str:
        """Append workspace-only report sections after the base Narrator report."""
        master_output = review_result.agent_strategy if review_result else None
        branch_outputs = review_result.branch_agent_outputs if review_result else {}
        lines = [base_report.rstrip(), ""]
        lines.extend(self._render_data_overview(snapshot, report_bundle))
        lines.extend(self._render_market_overview(report_bundle, portfolio_plan, master_output))
        lines.extend(self._render_analysis_process(execution_log, layer_timings))
        lines.extend(self._render_subagent_reasoning(branch_outputs))
        lines.extend(self._render_master_reasoning(master_output))
        lines.extend(self._render_final_advice(report_bundle, final_strategy, master_output))
        lines.extend(self._render_trade_instructions(final_strategy))
        lines.extend(self._render_next_steps(report_bundle, master_output))
        return "\n".join(lines).rstrip() + "\n"

    def _render_data_overview(self, snapshot: Any, report_bundle: ReportBundle) -> list[str]:
        data_bundle = getattr(snapshot, "data_bundle", None)
        symbols = list(
            getattr(data_bundle, "symbols", [])
            or getattr(data_bundle, "stock_pool", [])
            or self.stock_pool
        )
        enabled_branches = sorted(getattr(snapshot, "branch_results", {}).keys())
        coverage_highlights = list(getattr(report_bundle, "coverage_summary", []) or [])[:4]
        lines = [
            "## 数据概览",
            f"- 市场: {self.market}",
            f"- 标的数量: {len(symbols)}",
            f"- 回看区间: {self.lookback_years} 年",
            f"- 启用分支: {', '.join(enabled_branches) if enabled_branches else '无'}",
        ]
        if coverage_highlights:
            lines.extend(f"- 覆盖提示: {item}" for item in coverage_highlights)
        else:
            lines.append("- 覆盖提示: 本轮未记录显著覆盖缺口。")
        lines.append("")
        return lines

    def _render_market_overview(
        self,
        report_bundle: ReportBundle,
        portfolio_plan: PortfolioPlan,
        master_output: MasterAgentOutput | None,
    ) -> list[str]:
        macro_verdict = report_bundle.macro_verdict
        market_view = list(getattr(report_bundle, "market_view", []) or [])
        lines = [
            "## 市场概览",
            f"- 宏观结论: {getattr(macro_verdict, 'thesis', '') or '未提供'}",
            (
                f"- 目标暴露: gross {float(getattr(portfolio_plan, 'target_gross_exposure', 0.0)):.1%}"
                f", net {float(getattr(portfolio_plan, 'target_net_exposure', 0.0)):.1%}"
                f", cash {float(getattr(portfolio_plan, 'cash_ratio', 0.0)):.1%}"
            ),
        ]
        if master_output is not None:
            lines.append(
                f"- Review Layer 暴露建议: {float(getattr(master_output, 'risk_adjusted_exposure', 0.0)):.1%}"
            )
        if market_view:
            lines.extend(f"- 市场要点: {item}" for item in market_view[:4])
        lines.append("")
        return lines

    def _render_analysis_process(
        self,
        execution_log: list[str],
        layer_timings: dict[str, float],
    ) -> list[str]:
        lines = ["## 分析过程"]
        if execution_log:
            lines.extend(f"- 过程日志: {item}" for item in execution_log[:12])
        else:
            lines.append("- 过程日志: 无执行日志。")
        if layer_timings:
            ordered_timings = sorted(layer_timings.items(), key=lambda item: item[0])
            lines.extend(f"- 阶段耗时: {key}={float(value):.2f}s" for key, value in ordered_timings[:12])
        else:
            lines.append("- 阶段耗时: 无耗时明细。")
        lines.append("")
        return lines

    def _render_subagent_reasoning(
        self,
        branch_outputs: dict[str, BaseBranchAgentOutput | None],
    ) -> list[str]:
        lines = ["## SubAgent 决策过程、逻辑和依据"]
        if not branch_outputs:
            lines.extend(["- 本轮未启用或未成功产出分支 review。", ""])
            return lines
        for branch_name in sorted(branch_outputs):
            output = branch_outputs[branch_name]
            if output is None:
                lines.append(f"- {branch_name}: 本轮未成功产出 review。")
                continue
            lines.append(f"### {branch_name}")
            lines.append(f"- Conviction: {output.conviction} / confidence={float(output.confidence):.2f}")
            if output.key_insights:
                lines.extend(f"- 关键洞察: {item}" for item in output.key_insights[:3])
            if output.risk_flags:
                lines.extend(f"- 风险标记: {item}" for item in output.risk_flags[:3])
            if output.disagreements_with_algo:
                lines.extend(f"- 与算法分歧: {item}" for item in output.disagreements_with_algo[:2])
            lines.append(f"- 推理摘要: {output.reasoning or '无'}")
        lines.append("")
        return lines

    def _render_master_reasoning(
        self,
        master_output: MasterAgentOutput | None,
    ) -> list[str]:
        lines = ["## Master Agent 决策过程、逻辑和依据"]
        if master_output is None:
            lines.extend(["- 本轮未成功产出 Master Agent review。", ""])
            return lines
        lines.extend(
            [
                f"- 最终 conviction: {master_output.final_conviction}",
                f"- 最终分数: {float(master_output.final_score):.2f}",
                f"- 置信度: {float(master_output.confidence):.2f}",
            ]
        )
        if master_output.conviction_drivers:
            lines.extend(f"- Conviction driver: {item}" for item in master_output.conviction_drivers[:4])
        if master_output.debate_rounds:
            lines.extend(f"- Debate round: {item}" for item in master_output.debate_rounds[:3])
        if master_output.debate_resolution:
            lines.extend(f"- Debate resolution: {item}" for item in master_output.debate_resolution[:3])
        if master_output.portfolio_narrative:
            lines.append(f"- 组合叙事: {master_output.portfolio_narrative}")
        lines.append("")
        return lines

    def _render_final_advice(
        self,
        report_bundle: ReportBundle,
        final_strategy: PortfolioStrategy,
        master_output: MasterAgentOutput | None,
    ) -> list[str]:
        ic_decision = report_bundle.ic_decision
        lines = ["## 最终投资建议"]
        lines.append(
            f"- Structured IC 动作: {getattr(getattr(ic_decision, 'action', None), 'value', 'hold')}"
        )
        lines.append(
            f"- Structured IC thesis: {getattr(ic_decision, 'thesis', '') or report_bundle.summary or '无'}"
        )
        if master_output is not None and master_output.top_picks:
            top_symbols = ", ".join(pick.symbol for pick in master_output.top_picks[:5])
            lines.append(f"- Review Layer top picks: {top_symbols}")
        research_mode = getattr(final_strategy, "research_mode", "")
        if research_mode:
            lines.append(f"- 执行模式: {research_mode}")
        lines.append("")
        return lines

    def _render_trade_instructions(self, final_strategy: PortfolioStrategy) -> list[str]:
        lines = ["## 仓位、买卖指令"]
        recommendations = list(
            getattr(final_strategy, "recommendations", [])
            or getattr(final_strategy, "trade_recommendations", [])
            or []
        )
        if not recommendations:
            lines.extend(["- 本轮没有生成可展示的交易指令。", ""])
            return lines
        for recommendation in recommendations[:12]:
            action = str(getattr(recommendation, "action", "hold")).strip().lower()
            weight = float(
                getattr(recommendation, "weight", getattr(recommendation, "suggested_weight", 0.0))
                or 0.0
            )
            confidence = float(getattr(recommendation, "confidence", 0.0) or 0.0)
            rationale = (
                str(getattr(recommendation, "rationale", "")).strip()
                or str(getattr(recommendation, "one_line_conclusion", "")).strip()
                or str(getattr(recommendation, "symbol", "")).strip()
            )
            instruction_type = "直接指令" if action in {"buy", "sell"} else "观察/持有"
            lines.append(
                f"- {getattr(recommendation, 'symbol', '')}: {instruction_type} {action}, "
                f"目标仓位 {weight:.1%}, 置信度 {confidence:.2f}, 依据 {rationale}"
            )
        lines.append("")
        return lines

    def _render_next_steps(
        self,
        report_bundle: ReportBundle,
        master_output: MasterAgentOutput | None,
    ) -> list[str]:
        lines = ["## 下一步计划"]
        coverage_summary = list(getattr(report_bundle, "coverage_summary", []) or [])
        warnings = list(getattr(report_bundle, "warnings", []) or [])
        diagnostics = list(getattr(report_bundle, "appendix_diagnostics", []) or [])
        if coverage_summary:
            lines.extend(f"- 补充覆盖: {item}" for item in coverage_summary[:3])
        if warnings:
            lines.extend(f"- 风险跟踪: {item}" for item in warnings[:3])
        if diagnostics:
            lines.extend(f"- 工程跟进: {item}" for item in diagnostics[:3])
        if master_output is not None and master_output.top_picks:
            for pick in master_output.top_picks[:3]:
                lines.append(
                    f"- 持续跟踪 {pick.symbol}: review 建议为 {pick.action}，依据 {pick.rationale or '待补充'}"
                )
        if len(lines) == 1:
            lines.append("- 当前没有额外待办。")
        lines.append("")
        return lines

    def _build_execution_log(
        self,
        snapshot: ResearchCoreSnapshot,
        review_result: AgentEnhancedStrategy | None,
        agent_layer_enabled: bool,
    ) -> list[str]:
        execution_log = list(snapshot.execution_log)
        if agent_layer_enabled and review_result is not None:
            execution_log.append("[INFO] review layer 已应用。")
            for branch_name, output in review_result.branch_agent_outputs.items():
                status = "reviewed" if output is not None else "skipped"
                execution_log.append(f"[INFO] review_branch[{branch_name}]={status}")
        else:
            execution_log.append("[INFO] review layer 未启用，使用 deterministic structured adapters。")
        execution_log.append("single-mainline 完成")
        return execution_log

    def _build_layer_timings(
        self,
        snapshot: ResearchCoreSnapshot,
        review_result: AgentEnhancedStrategy | None,
        total_time: float,
    ) -> dict[str, float]:
        timings = dict(snapshot.timings)
        if review_result is not None:
            for key, value in review_result.agent_layer_timings.items():
                timings[f"review_{key}"] = float(value)
        timings["total"] = total_time
        return timings

    @staticmethod
    def _build_ensemble_output(strategy: PortfolioStrategy) -> dict[str, Any]:
        ensemble_output: dict[str, Any] = {}
        branch_consensus = getattr(strategy, "branch_consensus", {})
        if branch_consensus:
            ensemble_output["branch_consensus"] = branch_consensus
            scores = list(branch_consensus.values())
            ensemble_output["aggregate_score"] = sum(scores) / len(scores)
        else:
            weights = list(strategy.target_weights.values())
            ensemble_output["aggregate_score"] = sum(weights) / len(weights) if weights else 0.0
        return ensemble_output

    def _derive_market_regime(
        self,
        snapshot: ResearchCoreSnapshot,
    ) -> str:
        macro_branch = snapshot.branch_results.get("macro")
        if macro_branch and isinstance(macro_branch.signals, dict):
            regime = macro_branch.signals.get("regime") or macro_branch.signals.get("risk_level")
            if regime:
                return str(regime)
        return "default"

    def _build_structured_constraints(
        self,
        snapshot: ResearchCoreSnapshot,
        review_result: AgentEnhancedStrategy | None,
    ) -> dict[str, Any]:
        target_weights = dict(snapshot.baseline_strategy.target_weights)
        gross_cap = float(snapshot.baseline_strategy.total_exposure or sum(target_weights.values()) or 1.0)
        max_weight = max(target_weights.values(), default=1.0)
        synthetic_symbols = getattr(snapshot.data_bundle, "synthetic_symbols", [])
        if callable(synthetic_symbols):
            synthetic_symbols = synthetic_symbols()
        constraints: dict[str, Any] = {
            "gross_exposure_cap": BaseAgent.clamp(gross_cap, 0.0, 1.0),
            "max_weight": BaseAgent.clamp(max_weight, 0.0, 1.0),
            "blocked_symbols": list(synthetic_symbols or []),
            "risk_flags": [],
            "position_limits": {},
        }
        if review_result and review_result.agent_strategy is not None:
            constraints["gross_exposure_cap"] = min(
                constraints["gross_exposure_cap"],
                float(review_result.agent_strategy.risk_adjusted_exposure),
            )
        if review_result and review_result.risk_agent_output is not None:
            risk_output = review_result.risk_agent_output
            constraints["gross_exposure_cap"] = min(
                constraints["gross_exposure_cap"],
                float(risk_output.max_recommended_exposure),
            )
            constraints["risk_flags"] = BaseAgent.dedupe_texts(
                list(constraints["risk_flags"]) + list(risk_output.risk_warnings)
            )
            position_limits = dict(constraints["position_limits"])
            for symbol, payload in risk_output.position_sizing_overrides.items():
                if not isinstance(payload, dict):
                    continue
                max_override = payload.get("max_weight")
                if max_override is None:
                    continue
                position_limits[str(symbol)] = min(
                    position_limits.get(str(symbol), constraints["max_weight"]),
                    BaseAgent.clamp(float(max_override), 0.0, 1.0),
                )
            constraints["position_limits"] = position_limits
            if position_limits:
                constraints["max_weight"] = min(
                    constraints["max_weight"],
                    max(position_limits.values()),
                )
        return constraints

    def _apply_review_to_verdict(
        self,
        baseline: BranchVerdict,
        review_output: BaseBranchAgentOutput | None,
        *,
        symbol: str | None,
    ) -> BranchVerdict:
        if review_output is None:
            return baseline

        payload = deepcopy(baseline)
        bounded_score = self._bounded_review_score(
            payload.final_score,
            float(getattr(review_output, "conviction_score", payload.final_score)),
        )
        merged_confidence = BaseAgent.clamp(
            (payload.final_confidence + float(review_output.confidence)) / 2.0,
            0.0,
            1.0,
        )
        insight = ""
        if symbol and review_output.symbol_views.get(symbol):
            insight = str(review_output.symbol_views[symbol]).strip()
        if not insight and review_output.key_insights:
            insight = str(review_output.key_insights[0]).strip()
        if not insight:
            insight = payload.thesis

        risk_notes, coverage_notes, diagnostic_notes = BaseAgent.partition_bucket_notes(
            list(review_output.risk_flags)
        )
        payload.thesis = insight or payload.thesis
        payload.final_score = bounded_score
        payload.final_confidence = merged_confidence
        payload.direction = BaseAgent.score_to_direction(bounded_score)
        payload.action = BaseAgent.score_to_action(bounded_score)
        payload.confidence_label = BaseAgent.confidence_to_label(merged_confidence)
        payload.investment_risks = BaseAgent.dedupe_texts(payload.investment_risks + risk_notes)
        payload.coverage_notes = BaseAgent.dedupe_texts(payload.coverage_notes + coverage_notes)
        payload.diagnostic_notes = BaseAgent.dedupe_texts(payload.diagnostic_notes + diagnostic_notes)
        payload.metadata = dict(payload.metadata)
        payload.metadata.update(
            {
                "llm_reviewed": True,
                "llm_review_agent": format_agent_display_name(review_output.branch_name),
                "llm_conviction": review_output.conviction,
                "llm_conviction_score": float(review_output.conviction_score),
                "llm_disagreements_with_algo": list(review_output.disagreements_with_algo),
            }
        )
        if payload.evidence:
            payload.evidence[0].summary = insight
            payload.evidence[0].score = bounded_score
            payload.evidence[0].confidence = merged_confidence
            payload.evidence[0].direction = payload.direction
            if symbol:
                payload.evidence[0].symbols = [symbol]
        return payload

    @staticmethod
    def _bounded_review_score(base_score: float, review_score: float) -> float:
        return BaseAgent.clamp(
            float(review_score),
            float(base_score) - 0.30,
            float(base_score) + 0.30,
        )

    def _build_ic_hints_by_symbol(
        self,
        master_output: MasterAgentOutput | None,
    ) -> dict[str, dict[str, Any]]:
        if master_output is None:
            return {}
        hints: dict[str, dict[str, Any]] = {}
        for pick in master_output.top_picks:
            hints[pick.symbol] = {
                "score": self._bounded_review_score(
                    float(master_output.final_score),
                    self._conviction_score(pick.conviction),
                ),
                "confidence": float(master_output.confidence),
                "action": str(pick.action).strip().lower(),
                "rationale_points": BaseAgent.dedupe_texts(
                    [pick.rationale] + list(master_output.conviction_drivers[:2]) + list(master_output.debate_resolution[:1])
                ),
            }
        return hints

    @staticmethod
    def _conviction_score(conviction: str) -> float:
        return _CONVICTION_SCORE.get(str(conviction).strip().lower(), 0.0)

    def _portfolio_plan_to_strategy(
        self,
        *,
        portfolio_plan: PortfolioPlan,
        ic_by_symbol: dict[str, ICDecision],
        report_bundle: ReportBundle,
    ) -> PortfolioStrategy:
        recommendations: list[TradeRecommendation] = []
        tracked_symbols = sorted(set(ic_by_symbol) | set(portfolio_plan.target_weights))
        branch_consensus = {
            branch_name: float(verdict.final_score)
            for branch_name, verdict in report_bundle.branch_verdicts.items()
        }
        execution_notes = BaseAgent.dedupe_texts(
            list(portfolio_plan.execution_notes) + list(portfolio_plan.construction_notes)
        )
        risk_decision = report_bundle.risk_decision
        if portfolio_plan.target_gross_exposure <= 0.0 or not portfolio_plan.target_weights:
            research_mode = "research_only"
        elif portfolio_plan.status.value != "success" or (
            risk_decision is not None and risk_decision.status.value != "success"
        ):
            research_mode = "degraded"
        else:
            research_mode = "production"
        for symbol in tracked_symbols:
            ic_decision = ic_by_symbol.get(symbol)
            weight = float(portfolio_plan.target_weights.get(symbol, 0.0))
            if ic_decision is None:
                action = "watch" if weight <= 0 else "hold"
                confidence = 0.0
                rationale = symbol
            else:
                action = ic_decision.action.value
                confidence = float(ic_decision.final_confidence)
                rationale = ic_decision.thesis or symbol
            recommendations.append(
                TradeRecommendation(
                    symbol=symbol,
                    action=action,
                    weight=weight,
                    confidence=confidence,
                    rationale=rationale,
                    one_line_conclusion=rationale,
                    support_drivers=list(getattr(ic_decision, "rationale_points", [])[:2]) if ic_decision else [],
                    drag_drivers=[],
                    weight_cap_reasons=list(portfolio_plan.execution_notes[:1]),
                    metadata={"generated_from": "mainline_structured_outputs"},
                )
            )
        return PortfolioStrategy(
            recommendations=recommendations,
            target_weights=dict(portfolio_plan.target_weights),
            target_positions=dict(portfolio_plan.target_positions),
            position_limits=dict(portfolio_plan.position_limits),
            blocked_symbols=list(portfolio_plan.blocked_symbols),
            rejected_symbols=list(portfolio_plan.rejected_symbols),
            total_exposure=float(portfolio_plan.target_exposure),
            gross_exposure=float(portfolio_plan.target_gross_exposure),
            net_exposure=float(portfolio_plan.target_net_exposure),
            cash_ratio=float(portfolio_plan.cash_ratio),
            summary=report_bundle.summary,
            notes=list(report_bundle.executive_summary),
            metadata={
                "generated_from": "mainline_structured_outputs",
                "report_headline": report_bundle.headline,
            },
            style_bias=str(
                getattr(report_bundle.macro_verdict, "metadata", {}).get("style_bias", "balanced")
            ),
            branch_consensus=branch_consensus,
            risk_summary={
                "risk_level": (
                    risk_decision.risk_level.value
                    if risk_decision is not None
                    else "low"
                ),
                "warnings": list(risk_decision.reasons) if risk_decision is not None else [],
                "blocked_symbols": list(risk_decision.blocked_symbols) if risk_decision is not None else [],
                "gross_exposure_cap": (
                    float(risk_decision.gross_exposure_cap)
                    if risk_decision is not None
                    else float(portfolio_plan.target_gross_exposure)
                ),
                "max_single_position": max(portfolio_plan.position_limits.values(), default=0.0),
                "action_cap": (
                    risk_decision.action_cap.value
                    if risk_decision is not None
                    else "buy"
                ),
            },
            execution_notes=execution_notes,
            research_mode=research_mode,
            provenance_summary={
                "generated_from": "mainline_structured_outputs",
                "agent_layer_enabled": getattr(self.result, "agent_layer_enabled", False),
                "tracked_symbols": tracked_symbols,
            },
        )

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print(self.result.final_report)
        print("=" * 70)

    def save_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.result.final_report)
        _logger.info(f"报告已保存到: {path}")
