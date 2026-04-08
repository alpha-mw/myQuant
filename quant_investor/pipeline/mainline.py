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

from quant_investor.agent_protocol import BranchVerdict, GlobalContext, ICDecision, PortfolioPlan, ReportBundle
from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BaseBranchAgentOutput,
    MasterAgentInput,
    MasterAgentOutput,
    RiskAgentOutput,
    ShortlistEvidencePack,
)
from quant_investor.agents.base import BaseAgent
from quant_investor.agents.llm_client import has_provider_for_model, resolve_default_model
from quant_investor.agents.prompts import format_agent_display_name
from quant_investor.bayesian.likelihood import SignalLikelihoodMapper
from quant_investor.bayesian.posterior import BayesianPosteriorEngine
from quant_investor.bayesian.prior import HierarchicalPriorBuilder
from quant_investor.bayesian.types import PosteriorResult
from quant_investor.funnel.deterministic_funnel import DeterministicFunnel, FunnelConfig, FunnelOutput
from quant_investor.model_roles import resolve_model_role
from quant_investor.branch_contracts import (
    BranchResult,
    LLMUsageRecord,
    LLMUsageSummary,
    PortfolioStrategy,
    TradeRecommendation,
    UnifiedDataBundle,
)
from quant_investor.config import config
from quant_investor.global_context.builder import GlobalContextBuilder
from quant_investor.llm_gateway import (
    current_usage_session_id,
    end_usage_session,
    snapshot_usage,
    start_usage_session,
    update_usage_markdown,
)
from quant_investor.logger import get_logger
from quant_investor.market.data_snapshot import build_market_data_snapshot
from quant_investor.reporting.conclusion_renderer import ConclusionRenderer
from quant_investor.reporting.run_artifacts import build_model_role_metadata
from quant_investor.versioning import (
    AGENT_SCHEMA_VERSION,
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    DEBATE_TEMPLATE_VERSION,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)

_logger = get_logger("QuantInvestor")


def _execute_market_dag(**kwargs: Any) -> dict[str, Any]:
    from quant_investor.market.dag_executor import execute_market_dag

    return execute_market_dag(**kwargs)

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
    agent_orchestration: Optional[dict[str, Any]] = None
    agent_portfolio_plan: Any = None
    agent_report_bundle: Any = None
    agent_ic_decisions: Any = field(default_factory=dict)
    agent_review_bundle: Any = None
    ic_hints_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_role_metadata: Any = None
    execution_trace: Any = None
    what_if_plan: Any = None
    llm_usage_records: list[LLMUsageRecord] = field(default_factory=list)
    llm_usage_summary: LLMUsageSummary = field(default_factory=LLMUsageSummary)
    llm_usage_session_id: str = ""
    data_snapshot: dict[str, Any] = field(default_factory=dict)
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
    review_bundle: Any = None
    symbol_review_bundle: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_layer_enabled: bool = False
    agent_schema_version: str = AGENT_SCHEMA_VERSION
    pipeline_mode: str = "legacy"
    global_context: Optional[GlobalContext] = None
    funnel_output: Optional[FunnelOutput] = None
    bayesian_records: list[Any] = field(default_factory=list)
    shortlist_evidence: list[Any] = field(default_factory=list)
    bayesian_shortlist_symbols: list[str] = field(default_factory=list)


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
        agent_model: str = "moonshot-v1-128k",
        master_model: str = "moonshot-v1-128k",
        agent_fallback_model: str = "deepseek-reasoner",
        master_fallback_model: str = "deepseek-chat",
        master_reasoning_effort: str = "",
        agent_timeout: float = 15.0,
        master_timeout: float = 30.0,
        agent_total_timeout: float = 120.0,
        universe_key: str = "full_a",
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
        self.primary_agent_model = str(agent_model or "").strip() or resolve_default_model(preferred_model="moonshot-v1-128k")
        self.primary_master_model = str(master_model or "").strip() or resolve_default_model(preferred_model="moonshot-v1-128k")
        self.agent_fallback_model = str(agent_fallback_model or "").strip()
        self.master_fallback_model = str(master_fallback_model or "").strip()
        self.agent_resolution = resolve_model_role(
            role="branch",
            primary_model=self.primary_agent_model,
            fallback_model=self.agent_fallback_model,
        )
        self.master_resolution = resolve_model_role(
            role="master",
            primary_model=self.primary_master_model,
            fallback_model=self.master_fallback_model,
        )
        self.agent_model = self.agent_resolution.resolved_model
        self.master_model = self.master_resolution.resolved_model
        self.master_reasoning_effort = str(master_reasoning_effort or "").strip() or "high"
        self.agent_timeout = agent_timeout
        self.master_timeout = master_timeout
        self.agent_total_timeout = agent_total_timeout
        self.universe_key = str(universe_key or "").strip() or "full_a"
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
            str(agent_model or "").strip() or resolve_default_model(preferred_model="moonshot-v1-128k"),
            str(master_model or "").strip() or resolve_default_model(preferred_model="moonshot-v1-128k"),
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
            self._log(f"市场: {self.market}  universe={self.universe_key}  资金: ¥{self.total_capital:,.0f}")
            self._log(
                "Review Layer: %s | branch_model=%s | master_model=%s | master_reasoning_effort=%s"
                % (
                    "启用" if self.enable_agent_layer else "禁用",
                    self.agent_model,
                    self.master_model,
                    self.master_reasoning_effort,
                )
            )
            if self.agent_resolution.fallback_used:
                self._log(
                    "branch model fallback: primary=%s fallback=%s reason=%s"
                    % (
                        self.agent_resolution.primary_model,
                        self.agent_resolution.fallback_model,
                        self.agent_resolution.fallback_reason,
                    )
                )
            if self.master_resolution.fallback_used:
                self._log(
                    "master model fallback: primary=%s fallback=%s reason=%s"
                    % (
                        self.master_resolution.primary_model,
                        self.master_resolution.fallback_model,
                        self.master_resolution.fallback_reason,
                    )
                )
            self._log("=" * 60)

            if hasattr(config, "validate_runtime"):
                validation = config.validate_runtime(
                    market=self.market,
                    enable_agent_layer=self.enable_agent_layer,
                    agent_model=self.agent_model,
                    master_model=self.master_model,
                    master_reasoning_effort=self.master_reasoning_effort,
                    kline_backend=self.kline_backend,
                )
                for message in validation["errors"]:
                    self._log(f"⚠️ {message}")
                for message in validation["warnings"]:
                    self._log(f"⚠️ {message}")
            data_snapshot = build_market_data_snapshot(
                market=self.market,
                universe=self.universe_key,
                requested_symbols=list(self.stock_pool),
            )
            missing_requested = list(data_snapshot.get("missing_requested_symbols", []) or [])
            unreadable_requested = list(data_snapshot.get("unreadable_requested_symbols", []) or [])
            if missing_requested or unreadable_requested:
                missing_text = ", ".join(missing_requested + unreadable_requested)
                raise ValueError(f"本地数据不可用于研究：{missing_text}")
            if data_snapshot.get("summary_text"):
                self._log(f"数据快照: {data_snapshot['summary_text']}")
            dag_artifacts = _execute_market_dag(
                market=self.market,
                symbols=list(self.stock_pool),
                universe=self.universe_key,
                mode="sample",
                batch_size=len(self.stock_pool),
                total_capital=self.total_capital,
                top_k=max(1, min(len(self.stock_pool), int(getattr(config, "BAYESIAN_SHORTLIST_SIZE", 20)))),
                verbose=self.verbose,
                enable_agent_layer=self.enable_agent_layer,
                agent_model=self.primary_agent_model,
                agent_fallback_model=self.agent_fallback_model,
                master_model=self.primary_master_model,
                master_fallback_model=self.master_fallback_model,
                master_reasoning_effort=self.master_reasoning_effort,
                agent_timeout=self.agent_timeout,
                master_timeout=self.master_timeout,
                recall_context=self.recall_context,
                data_snapshot=data_snapshot,
            )
            records, summary = snapshot_usage(current_usage_session_id() or "")
            result = self._build_result_from_dag(
                dag_artifacts=dag_artifacts,
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

    def _build_result_from_dag(
        self,
        *,
        dag_artifacts: dict[str, Any],
        usage_records: list[Any],
        usage_summary: Any,
        total_time: float,
    ) -> QuantInvestorPipelineResult:
        report_bundle = dag_artifacts.get("report_bundle")
        portfolio_decision = dag_artifacts.get("portfolio_decision")
        global_context = dag_artifacts.get("global_context")
        data_snapshot = dict(
            dag_artifacts.get("data_snapshot", {})
            or (
                getattr(global_context, "metadata", {}) or {}
            ).get("data_snapshot", {})
        )
        final_strategy = self._portfolio_decision_to_strategy(
            portfolio_decision=portfolio_decision,
            report_bundle=report_bundle,
            global_context=global_context,
        )
        final_report = update_usage_markdown(
            getattr(report_bundle, "markdown_report", "") or "",
            usage_summary,
            title="## LLM 可观测性（统一 DAG）",
        )
        result = QuantInvestorPipelineResult(
            architecture_version=ARCHITECTURE_VERSION,
            branch_schema_version=BRANCH_SCHEMA_VERSION,
            ic_protocol_version=getattr(report_bundle, "ic_protocol_version", IC_PROTOCOL_VERSION),
            report_protocol_version=getattr(report_bundle, "report_protocol_version", REPORT_PROTOCOL_VERSION),
            calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
            debate_template_version=DEBATE_TEMPLATE_VERSION,
            data_bundle=None,
            branch_results=dict(dag_artifacts.get("branch_results", {})),
            calibrated_signals={},
            risk_results=dag_artifacts.get("risk_decision"),
            final_strategy=final_strategy,
            final_report=final_report,
            execution_log=list(self.result.execution_log) + ["[INFO] unified market DAG executed", "single-mainline 完成"],
            layer_timings={"total": total_time},
            total_time=total_time,
            agent_orchestration=dict(dag_artifacts),
            agent_portfolio_plan=dag_artifacts.get("portfolio_plan"),
            agent_report_bundle=report_bundle,
            agent_ic_decisions=dag_artifacts.get("ic_decisions"),
            agent_review_bundle=dag_artifacts.get("review_bundle"),
            ic_hints_by_symbol=dict(getattr(dag_artifacts.get("review_bundle"), "ic_hints_by_symbol", {}) or {}),
            model_role_metadata=dag_artifacts.get("model_role_metadata"),
            execution_trace=dag_artifacts.get("execution_trace"),
            what_if_plan=dag_artifacts.get("what_if_plan"),
            llm_usage_records=list(usage_records),
            llm_usage_summary=usage_summary,
            llm_usage_session_id=current_usage_session_id() or "",
            data_snapshot=data_snapshot,
            raw_data={},
            factor_data={},
            model_predictions={},
            macro_signal=str(getattr(getattr(report_bundle, "macro_verdict", None), "metadata", {}).get("policy_signal", "🟡")),
            macro_summary=str(getattr(getattr(report_bundle, "macro_verdict", None), "thesis", "") or ""),
            baseline_strategy=final_strategy,
            baseline_risk_result=dag_artifacts.get("risk_decision"),
            macro_verdict=getattr(report_bundle, "macro_verdict", None),
            reviewed_research_by_symbol=dict(dag_artifacts.get("branch_verdicts_by_symbol", {})),
            reviewed_branch_summaries=dict(dag_artifacts.get("branch_summaries", {})),
            branch_review_outputs=(
                dict(dag_artifacts.get("branch_summaries", {}))
                if self.enable_agent_layer
                else {}
            ),
            master_review_output=(
                dag_artifacts.get("portfolio_master_output")
                if self.enable_agent_layer
                else None
            ),
            risk_review_output=dag_artifacts.get("risk_decision"),
            review_bundle=dag_artifacts.get("review_bundle"),
            symbol_review_bundle={},
            agent_layer_enabled=self.enable_agent_layer,
            pipeline_mode="bayesian",
            global_context=global_context,
            funnel_output=dag_artifacts.get("funnel_output"),
            bayesian_records=list(dag_artifacts.get("bayesian_records", [])),
            shortlist_evidence=list(dag_artifacts.get("shortlist_evidence", [])),
            bayesian_shortlist_symbols=[
                item.symbol
                for item in list(dag_artifacts.get("shortlist", []))
            ],
        )
        result.final_strategy.metadata["agent_layer_enabled"] = self.enable_agent_layer
        provenance_summary = dict(getattr(result.final_strategy, "provenance_summary", {}))
        provenance_summary["agent_layer_enabled"] = self.enable_agent_layer
        result.final_strategy.provenance_summary = provenance_summary
        return result

    def _portfolio_decision_to_strategy(
        self,
        *,
        portfolio_decision: Any,
        report_bundle: Any,
        global_context: Any,
    ) -> PortfolioStrategy:
        shortlist = list(getattr(portfolio_decision, "shortlist", []) or [])
        target_weights = dict(getattr(portfolio_decision, "target_weights", {}) or {})
        recommendations: list[TradeRecommendation] = []
        for item in shortlist:
            rationale = list(getattr(item, "rationale", []) or [])
            recommendations.append(
                TradeRecommendation(
                    symbol=str(getattr(item, "symbol", "")),
                    action=str(getattr(getattr(item, "action", None), "value", "hold")),
                    weight=float(target_weights.get(getattr(item, "symbol", ""), getattr(item, "suggested_weight", 0.0)) or 0.0),
                    suggested_weight=float(getattr(item, "suggested_weight", 0.0) or 0.0),
                    confidence=float(getattr(item, "confidence", 0.0) or 0.0),
                    expected_upside=float(getattr(item, "expected_upside", 0.0) or 0.0),
                    rationale="；".join(rationale[:2]) or str(getattr(item, "symbol", "")),
                    one_line_conclusion="；".join(rationale[:1]) or str(getattr(item, "symbol", "")),
                    risk_flags=list(getattr(item, "risk_flags", []) or []),
                    metadata={"company_name": str(getattr(item, "company_name", ""))},
                )
            )
        macro_verdict = getattr(report_bundle, "macro_verdict", None)
        return PortfolioStrategy(
            target_exposure=float(getattr(portfolio_decision, "target_exposure", 0.0) or 0.0),
            total_exposure=float(getattr(portfolio_decision, "target_exposure", 0.0) or 0.0),
            gross_exposure=float(getattr(portfolio_decision, "target_gross_exposure", 0.0) or 0.0),
            net_exposure=float(getattr(portfolio_decision, "target_net_exposure", 0.0) or 0.0),
            cash_ratio=float(getattr(portfolio_decision, "cash_ratio", 1.0) or 1.0),
            style_bias=str(getattr(global_context, "style_exposures", {}).get("style_bias", "均衡")),
            candidate_symbols=[str(getattr(item, "symbol", "")) for item in shortlist],
            target_weights=target_weights,
            target_positions=dict(getattr(portfolio_decision, "target_positions", {}) or {}),
            position_limits=dict(getattr(getattr(report_bundle, "portfolio_plan", None), "position_limits", {}) or {}),
            execution_notes=list(getattr(getattr(report_bundle, "portfolio_plan", None), "execution_notes", []) or []),
            branch_consensus={
                name: float(getattr(verdict, "final_score", 0.0) or 0.0)
                for name, verdict in dict(getattr(report_bundle, "branch_verdicts", {}) or {}).items()
            },
            risk_summary=dict(getattr(portfolio_decision, "risk_constraints", {}) or {}),
            provenance_summary={
                "generated_from": "market_dag",
                "tracked_symbols": [str(getattr(item, "symbol", "")) for item in shortlist],
            },
            research_mode="production",
            summary=str(getattr(report_bundle, "summary", "") or ""),
            notes=list(getattr(report_bundle, "executive_summary", []) or []),
            metadata={
                "generated_from": "market_dag",
                "report_headline": str(getattr(report_bundle, "headline", "") or ""),
            },
            recommendations=recommendations,
            trade_recommendations=recommendations,
            blocked_symbols=list(getattr(getattr(report_bundle, "portfolio_plan", None), "blocked_symbols", []) or []),
            rejected_symbols=list(getattr(getattr(report_bundle, "portfolio_plan", None), "rejected_symbols", []) or []),
            sector_preferences=[],
            stop_loss_policy={},
            symbol_convictions={symbol: float(weight) for symbol, weight in target_weights.items()},
        )

    def _run_research_core(
        self,
    ) -> ResearchCoreSnapshot:
        raise RuntimeError("legacy research core has been retired; use execute_market_dag()")

    def _build_global_context(
        self,
        snapshot: ResearchCoreSnapshot,
    ) -> GlobalContext:
        builder = GlobalContextBuilder()
        return builder.build(
            stock_pool=self.stock_pool,
            market=self.market,
            universe_key=self.universe_key,
            config=config,
            data_bundle=snapshot.data_bundle,
        )

    def _run_funnel(
        self,
        snapshot: ResearchCoreSnapshot,
        global_context: GlobalContext,
    ) -> FunnelOutput:
        funnel_config = FunnelConfig(
            max_candidates=getattr(config, "FUNNEL_MAX_CANDIDATES", 400),
        )
        funnel = DeterministicFunnel(config=funnel_config)
        return funnel.run(
            branch_results=snapshot.branch_results,
            global_context=global_context,
        )

    def _run_bayesian_decision(
        self,
        snapshot: ResearchCoreSnapshot,
        global_context: GlobalContext,
        funnel_output: FunnelOutput,
    ) -> list[PosteriorResult]:
        prior_builder = HierarchicalPriorBuilder()
        likelihood_mapper = SignalLikelihoodMapper()
        engine = BayesianPosteriorEngine()
        candidate_set = set(funnel_output.candidates)
        regime = snapshot.market_regime or global_context.macro_regime or "default"

        records: list[PosteriorResult] = []
        for symbol in funnel_output.candidates:
            prior = prior_builder.build_prior(symbol, global_context)
            likelihoods = likelihood_mapper.compute_likelihoods(
                branch_results=snapshot.branch_results,
                symbol=symbol,
                candidate_symbols=candidate_set,
            )
            is_degraded: dict[str, bool] = {}
            for branch_name, br in snapshot.branch_results.items():
                if isinstance(br, BranchResult) and getattr(br, "metadata", {}).get("degraded"):
                    is_degraded[branch_name] = True

            company_name = global_context.symbol_name_map.get(symbol, "")
            result = engine.compute_posterior(
                prior, likelihoods,
                symbol=symbol,
                company_name=company_name,
                regime=regime,
                is_degraded=is_degraded,
            )
            records.append(result)

        records.sort(key=lambda r: r.posterior_action_score, reverse=True)
        for rank, record in enumerate(records, 1):
            record.rank = rank
        return records

    def _build_shortlist_evidence(
        self,
        bayesian_records: list[PosteriorResult],
        snapshot: ResearchCoreSnapshot,
        global_context: GlobalContext,
    ) -> list[ShortlistEvidencePack]:
        shortlist_size = getattr(config, "BAYESIAN_SHORTLIST_SIZE", 20)
        shortlist = bayesian_records[:shortlist_size]
        packs: list[ShortlistEvidencePack] = []
        for record in shortlist:
            branch_summaries: dict[str, dict[str, Any]] = {}
            for branch_name, br in snapshot.branch_results.items():
                if not isinstance(br, BranchResult):
                    continue
                sym_score = br.symbol_scores.get(record.symbol, br.final_score)
                branch_summaries[branch_name] = {
                    "score": float(sym_score) if sym_score is not None else 0.0,
                    "confidence": float(br.final_confidence or br.confidence or 0.0),
                    "direction": "bullish" if (sym_score or 0) > 0 else "bearish" if (sym_score or 0) < 0 else "neutral",
                }
            pack = ShortlistEvidencePack(
                symbol=record.symbol,
                company_name=record.company_name,
                bayesian_record=record.to_dict(),
                branch_verdicts_summary=branch_summaries,
                risk_flags=[],
                key_catalysts=[],
                macro_summary=global_context.macro_regime,
            )
            packs.append(pack)
        return packs

    def _run_bayesian_pipeline(
        self,
        snapshot: ResearchCoreSnapshot,
    ) -> tuple[GlobalContext, FunnelOutput, list[PosteriorResult], list[ShortlistEvidencePack]]:
        t0 = time.time()
        self._log("🔬 Bayesian pipeline: building GlobalContext...")
        global_context = self._build_global_context(snapshot)
        self._log(
            f"GlobalContext: regime={global_context.macro_regime}, "
            f"universe={len(global_context.universe_tiers.get('total', []))}, "
            f"quarantine={len(global_context.data_quality_quarantine)}"
        )

        self._log("🔬 Bayesian pipeline: running deterministic funnel...")
        funnel_output = self._run_funnel(snapshot, global_context)
        self._log(
            f"Funnel: {len(self.stock_pool)} -> {len(funnel_output.candidates)} candidates, "
            f"{len(funnel_output.excluded_symbols)} excluded"
        )

        self._log("🔬 Bayesian pipeline: computing Bayesian posteriors...")
        bayesian_records = self._run_bayesian_decision(snapshot, global_context, funnel_output)
        if bayesian_records:
            top = bayesian_records[0]
            self._log(
                f"Bayesian top: {top.symbol} ({top.company_name}) "
                f"action_score={top.posterior_action_score:.3f} "
                f"win_rate={top.posterior_win_rate:.3f}"
            )

        self._log("🔬 Bayesian pipeline: building shortlist evidence packs...")
        shortlist_evidence = self._build_shortlist_evidence(bayesian_records, snapshot, global_context)
        self._log(f"Shortlist: {len(shortlist_evidence)} symbols for master discussion")

        elapsed = time.time() - t0
        self._log(f"🔬 Bayesian pipeline complete in {elapsed:.1f}s")
        return global_context, funnel_output, bayesian_records, shortlist_evidence

    def _review_layer_available(self) -> bool:
        if not self.enable_agent_layer:
            self._log("Review layer: 已通过配置标志禁用 (enable_agent_layer=False)")
            return False
        if not has_provider_for_model(self.agent_model):
            self._log(
                f"Review layer: agent_model={self.agent_model!r} 无对应 API Key，"
                "将使用 review layer 的安全降级路径（请检查 KIMI_API_KEY / DEEPSEEK_API_KEY 等环境变量）"
            )
        if not has_provider_for_model(self.master_model):
            self._log(
                f"Review layer: master_model={self.master_model!r} 无对应 API Key，"
                "将使用 review layer 的安全降级路径（请检查 KIMI_API_KEY / DEEPSEEK_API_KEY 等环境变量）"
            )
        return True

    def _run_review_layer(
        self,
        snapshot: ResearchCoreSnapshot,
        *,
        candidate_symbols: list[str] | None = None,
    ) -> tuple[AgentEnhancedStrategy | None, bool]:
        raise RuntimeError("legacy review-layer side path has been retired; use execute_market_dag()")

    def _run_unified_control_chain(
        self,
        snapshot: ResearchCoreSnapshot,
        review_result: AgentEnhancedStrategy | None,
    ) -> dict[str, Any]:
        raise RuntimeError("legacy unified-control-chain adapter has been retired; use execute_market_dag()")

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
        model_role_metadata = self._build_model_role_metadata(report_bundle)
        report_bundle.model_role_metadata = model_role_metadata
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
            debate_template_version=DEBATE_TEMPLATE_VERSION,
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
            agent_review_bundle=orchestration.get("review_bundle"),
            ic_hints_by_symbol=self._extract_ic_hints_by_symbol(review_result),
            model_role_metadata=model_role_metadata,
            execution_trace=getattr(report_bundle, "execution_trace", None),
            what_if_plan=getattr(report_bundle, "what_if_plan", None),
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
            review_bundle=orchestration.get("review_bundle"),
            symbol_review_bundle=(
                dict(getattr(review_result, "symbol_review_bundle", {}))
                if review_result
                else {}
            ),
            agent_layer_enabled=agent_layer_enabled,
        )
        result.final_strategy.metadata["agent_layer_enabled"] = agent_layer_enabled
        provenance_summary = dict(getattr(result.final_strategy, "provenance_summary", {}))
        provenance_summary["agent_layer_enabled"] = agent_layer_enabled
        result.final_strategy.provenance_summary = provenance_summary
        return result

    def _build_model_role_metadata(self, report_bundle: ReportBundle) -> Any:
        payload = dict(getattr(report_bundle, "model_role_metadata", {}) or {})
        payload.setdefault("branch_model", self.primary_agent_model)
        payload.setdefault("master_model", self.primary_master_model)
        payload.setdefault("agent_fallback_model", self.agent_fallback_model)
        payload.setdefault("master_fallback_model", self.master_fallback_model)
        payload.setdefault("resolved_branch_model", self.agent_resolution.resolved_model)
        payload.setdefault("resolved_master_model", self.master_resolution.resolved_model)
        payload.setdefault("master_reasoning_effort", self.master_reasoning_effort)
        payload.setdefault("branch_provider", payload.get("branch_provider", ""))
        payload.setdefault("master_provider", payload.get("master_provider", ""))
        payload.setdefault("branch_timeout", self.agent_timeout)
        payload.setdefault("master_timeout", self.master_timeout)
        payload.setdefault("agent_layer_enabled", self.enable_agent_layer)
        payload.setdefault("branch_fallback_used", self.agent_resolution.fallback_used)
        payload.setdefault("master_fallback_used", self.master_resolution.fallback_used)
        payload.setdefault("branch_fallback_reason", self.agent_resolution.fallback_reason)
        payload.setdefault("master_fallback_reason", self.master_resolution.fallback_reason)
        payload.setdefault("universe_key", self.universe_key)
        payload.setdefault("universe_size", len(self.stock_pool))
        payload.setdefault("universe_hash", "")
        return build_model_role_metadata(
            branch_model=str(payload.get("branch_model", self.primary_agent_model)),
            master_model=str(payload.get("master_model", self.primary_master_model)),
            agent_fallback_model=str(payload.get("agent_fallback_model", self.agent_fallback_model)),
            master_fallback_model=str(payload.get("master_fallback_model", self.master_fallback_model)),
            resolved_branch_model=str(payload.get("resolved_branch_model", self.agent_resolution.resolved_model)),
            resolved_master_model=str(payload.get("resolved_master_model", self.master_resolution.resolved_model)),
            master_reasoning_effort=str(payload.get("master_reasoning_effort", self.master_reasoning_effort)),
            branch_provider=str(payload.get("branch_provider", "")),
            master_provider=str(payload.get("master_provider", "")),
            branch_timeout=float(payload.get("branch_timeout", self.agent_timeout)),
            master_timeout=float(payload.get("master_timeout", self.master_timeout)),
            agent_layer_enabled=bool(payload.get("agent_layer_enabled", self.enable_agent_layer)),
            branch_fallback_used=bool(payload.get("branch_fallback_used", self.agent_resolution.fallback_used)),
            master_fallback_used=bool(payload.get("master_fallback_used", self.master_resolution.fallback_used)),
            branch_fallback_reason=str(payload.get("branch_fallback_reason", self.agent_resolution.fallback_reason)),
            master_fallback_reason=str(payload.get("master_fallback_reason", self.master_resolution.fallback_reason)),
            universe_key=str(payload.get("universe_key", self.universe_key)),
            universe_size=int(payload.get("universe_size", len(self.stock_pool))),
            universe_hash=str(payload.get("universe_hash", "")),
            metadata=payload,
        )

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
        lines.extend(
            ConclusionRenderer.render_run_context(
                getattr(report_bundle, "model_role_metadata", None),
                getattr(report_bundle, "execution_trace", None),
                getattr(report_bundle, "what_if_plan", None),
            )
        )
        lines.extend(self._render_market_overview(report_bundle, portfolio_plan, master_output))
        lines.extend(self._render_analysis_process(execution_log, layer_timings))
        review_bundle = getattr(review_result, "review_bundle", None) if review_result else None
        if review_bundle is not None:
            lines.extend(self._render_review_overview(review_bundle))
        if self.result.pipeline_mode == "bayesian":
            lines.extend(self._render_bayesian_overview())
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

    def _render_review_overview(self, review_bundle: Any) -> list[str]:
        lines = ["## LLM 复核总览"]
        fallback_reasons = list(getattr(review_bundle, "fallback_reasons", []) or [])
        if fallback_reasons:
            lines.extend(f"- 降级原因: {reason}" for reason in fallback_reasons[:4])
        symbol_hints = dict(getattr(review_bundle, "ic_hints_by_symbol", {}) or {})
        if symbol_hints:
            lines.append("- IC hints by symbol:")
            for symbol in sorted(symbol_hints):
                hint = symbol_hints[symbol]
                lines.append(
                    f"  - {symbol}: action={hint.get('action', 'hold')}, "
                    f"score={float(hint.get('score', 0.0)):.2f}, "
                    f"confidence={float(hint.get('confidence', 0.0)):.2f}"
                )
        lines.append("")
        return lines

    def _render_bayesian_overview(self) -> list[str]:
        lines = ["## Bayesian 决策层"]
        funnel = self.result.funnel_output
        if funnel is not None:
            lines.append(
                f"- 漏斗压缩: {len(self.stock_pool)} 只 -> {len(funnel.candidates)} 候选 "
                f"({len(funnel.excluded_symbols)} 被排除)"
            )
        records = self.result.bayesian_records
        if records:
            lines.append(f"- Bayesian 后验排名: 共 {len(records)} 只候选")
            for r in records[:10]:
                name_tag = f" ({r.company_name})" if r.company_name else ""
                lines.append(
                    f"  - #{r.rank} {r.symbol}{name_tag}: "
                    f"action_score={r.posterior_action_score:.3f}, "
                    f"win_rate={r.posterior_win_rate:.3f}, "
                    f"confidence={r.posterior_confidence:.3f}"
                )
        shortlist = self.result.shortlist_evidence
        if shortlist:
            lines.append(f"- Master Discussion 入选: {len(shortlist)} 只")
            lines.append(f"  - 标的: {', '.join(p.symbol for p in shortlist[:10])}")
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

    def _extract_ic_hints_by_symbol(self, review_result: AgentEnhancedStrategy | None) -> dict[str, dict[str, Any]]:
        if review_result is None:
            return {}
        if review_result.ic_hints_by_symbol:
            return dict(review_result.ic_hints_by_symbol)
        review_bundle = getattr(review_result, "review_bundle", None)
        if review_bundle is not None:
            hints = getattr(review_bundle, "ic_hints_by_symbol", {})
            if isinstance(hints, dict):
                return {str(symbol): dict(hint) for symbol, hint in hints.items() if isinstance(hint, dict)}
        if review_result.agent_strategy is not None:
            return self._build_ic_hints_by_symbol(review_result.agent_strategy)
        return {}

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
