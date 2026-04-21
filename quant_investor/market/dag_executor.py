#!/usr/bin/env python3
"""Three-layer market DAG executor.

This module replaces the legacy batch-centric internal mainline.
It builds:

1. GlobalContext
2. PerSymbolResearch
3. PortfolioDecision

The public entrypoints remain unchanged, but all internal execution now flows
through this DAG.
"""

from __future__ import annotations

import json
import asyncio
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    GlobalContext,
    MasterICHint,
    PortfolioDecision,
    PortfolioPlan,
    RiskDecision,
    StockReviewBundle,
)
from quant_investor.agents.agent_contracts import BaseBranchAgentOutput
from quant_investor.agents.fundamental_agent import FundamentalAgent
from quant_investor.agents.ic_coordinator import ICCoordinator
from quant_investor.agents.intelligence_agent import IntelligenceAgent
from quant_investor.agents.kline_agent import KlineAgent
from quant_investor.agents.macro_agent import MacroAgent
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.narrator_agent import NarratorAgent
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.agents.risk_guard import RiskGuard
from quant_investor.agents.llm_client import LLMClient as GatewayLLMClient
from quant_investor.bayesian.likelihood import SignalLikelihoodMapper
from quant_investor.bayesian.posterior import BayesianPosteriorEngine
from quant_investor.bayesian.prior import HierarchicalPriorBuilder
from quant_investor.config import config
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.deterministic_funnel import DeterministicFunnel
from quant_investor.market.cn_resolver import CNUniverseResolver
from quant_investor.market.config import get_market_settings, normalize_categories, normalize_universe
from quant_investor.market.data_snapshot import build_market_data_snapshot
from quant_investor.market.data_quality import build_data_quality_diagnostics
from quant_investor.market.dag.assembly import (
    _aggregate_branch_summaries,
    _attach_symbol_to_ic_decision,
    _build_branch_results,
)
from quant_investor.market.dag.common import _run_async_coroutine_safely, _score_to_action
from quant_investor.market.dag.context import _prepare_market_context
from quant_investor.market.dag.decision import (
    _run_bayesian_selection_phase,
    _run_portfolio_construction_phase,
)
from quant_investor.market.dag.evidence import (
    _build_master_evidence_pack,
    _compact_trace_fragments,
)
from quant_investor.market.dag.packets import (
    _build_market_snapshot,
    _build_symbol_bundle,
    _build_symbol_research_packet,
)
from quant_investor.market.dag.research import _run_candidate_research_phase
from quant_investor.market.dag.reporting import _build_reporting_artifacts
from quant_investor.market.dag.review import _portfolio_master_advisory
from quant_investor.market.dag.shortlist import _build_shortlist, _build_shortlist_from_bayesian_records
from quant_investor.market.provider_health import detect_provider_health
from quant_investor.market.shared_csv_reader import SharedCSVReader
from quant_investor.llm_provider_priority import resolve_runtime_role_models
from quant_investor.model_roles import ModelRoleResolution, resolve_model_role
from quant_investor.reporting.run_artifacts import (
    build_bayesian_trace,
    build_execution_trace,
    build_model_role_metadata,
    build_what_if_plan,
)
from quant_investor.data.universe.cn_universe import LOCAL_UNIVERSE_DIR


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))

def _load_company_name_map(market: str) -> dict[str, str]:
    if str(market or "").strip().upper() != "CN":
        return {}
    path = LOCAL_UNIVERSE_DIR / "stock_names.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(symbol).strip().upper(): str(name).strip()
        for symbol, name in raw.items()
        if str(symbol).strip() and str(name).strip()
    }


def _load_company_profile_map(market: str) -> dict[str, dict[str, str]]:
    if str(market or "").strip().upper() != "CN":
        return {}
    db_path = Path(str(getattr(config, "DB_PATH", "") or "")).expanduser()
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "stock_list" not in tables and "stock_profiles" not in tables:
            conn.close()
            return {}
        query = """
            SELECT
                s.ts_code AS ts_code,
                s.name AS name,
                COALESCE(NULLIF(s.industry, ''), NULLIF(p.industry, ''), NULLIF(p.sector, '')) AS industry,
                COALESCE(NULLIF(p.sector, ''), NULLIF(s.industry, ''), NULLIF(p.industry, '')) AS sector
            FROM stock_list s
            LEFT JOIN stock_profiles p ON s.ts_code = p.ts_code
        """
        rows = conn.execute(query).fetchall() if "stock_list" in tables else []
        conn.close()
    except Exception:
        return {}
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        symbol = str(row["ts_code"] or "").strip().upper()
        if not symbol:
            continue
        name = str(row["name"] or "").strip()
        industry = str(row["industry"] or "").strip()
        sector = str(row["sector"] or industry or "").strip()
        result[symbol] = {
            "name": name,
            "industry": industry,
            "sector": sector,
        }
    return result


def _is_quarantined_read_result(read_result: Any) -> bool:
    issues = list(getattr(read_result, "issues", []) or [])
    return bool(issues)

def _score_to_direction(score: float) -> str:
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"

def _branch_conviction_from_action(action: ActionLabel) -> str:
    if action == ActionLabel.BUY:
        return "buy"
    if action == ActionLabel.SELL:
        return "sell"
    return "neutral"


def _master_hint_to_ic_hint(hint: MasterICHint) -> dict[str, Any]:
    return {
        "score": float(hint.score_hint),
        "confidence": float(hint.confidence_hint),
        "action": hint.action.value if hasattr(hint.action, "value") else str(hint.action),
        "direction": hint.direction.value if hasattr(hint.direction, "value") else str(hint.direction),
        "rationale_points": list(hint.rationale_points[:4]),
        "agreement_points": list(hint.agreement_points[:3]),
        "conflict_points": list(hint.conflict_points[:3]),
        "risk_flags": list(hint.risk_flags[:5]),
        "score_delta": float(hint.score_delta),
        "confidence_delta": float(hint.confidence_delta),
        "status": hint.status.value if hasattr(hint.status, "value") else str(hint.status),
        "telemetry": hint.telemetry.to_dict() if hasattr(hint.telemetry, "to_dict") else asdict(hint.telemetry),
        "thesis": hint.thesis,
        "metadata": dict(hint.metadata or {}),
    }

def _branch_verdict_to_result(verdict: BranchVerdict, branch_name: str) -> BranchResult:
    metadata = dict(verdict.metadata or {})
    raw_symbol_scores = metadata.get("legacy_symbol_scores")
    symbol_scores: dict[str, float] = {}
    if isinstance(raw_symbol_scores, Mapping):
        for symbol, score in raw_symbol_scores.items():
            text = str(symbol or "").strip()
            if not text:
                continue
            try:
                symbol_scores[text] = float(score)
            except Exception:
                continue
    if not symbol_scores:
        symbol_scores = {str(verdict.symbol or branch_name): float(verdict.final_score)}
    return BranchResult(
        branch_name=branch_name,
        score=float(verdict.final_score),
        confidence=float(verdict.final_confidence),
        signals=dict(verdict.metadata or {}),
        risks=list(verdict.investment_risks),
        explanation=str(verdict.thesis),
        symbol_scores=symbol_scores,
        success=verdict.status.value != "vetoed",
        metadata=metadata,
        base_score=float(verdict.final_score),
        final_score=float(verdict.final_score),
        base_confidence=float(verdict.final_confidence),
        final_confidence=float(verdict.final_confidence),
        conclusion=str(verdict.thesis),
        thesis_points=list(verdict.coverage_notes[:3]),
        investment_risks=list(verdict.investment_risks),
        coverage_notes=list(verdict.coverage_notes),
        diagnostic_notes=list(verdict.diagnostic_notes),
        support_drivers=[],
        drag_drivers=[],
        weight_cap_reasons=[],
        module_coverage={},
    )


def _branch_output_to_verdict(output: BaseBranchAgentOutput, symbol: str) -> BranchVerdict:
    action = str(output.conviction).lower()
    direction = _score_to_direction(float(output.conviction_score))
    if action not in {"strong_buy", "buy", "neutral", "sell", "strong_sell"}:
        action = "neutral"
    return BranchVerdict(
        agent_name=str(output.branch_name or ""),
        thesis=str(output.reasoning or "") or "分支已完成结构化判断。",
        symbol=symbol,
        status=BranchVerdict.__dataclass_fields__["status"].default,
        direction=_score_to_direction(float(output.conviction_score)),
        action=_score_to_action(float(output.conviction_score)),
        confidence_label=BranchVerdict.__dataclass_fields__["confidence_label"].default,
        final_score=float(output.conviction_score),
        final_confidence=float(output.confidence),
        investment_risks=list(output.risk_flags),
        coverage_notes=list(output.key_insights),
        diagnostic_notes=list(output.disagreements_with_algo),
        metadata={
            "branch_name": output.branch_name,
            "reasoning": output.reasoning,
            "symbol_views": dict(output.symbol_views),
        },
    )


def _ensure_branch_verdict(value: Any, *, symbol: str, branch_name: str) -> BranchVerdict:
    if isinstance(value, BranchVerdict):
        payload = BranchVerdict(
            agent_name=value.agent_name or branch_name,
            thesis=value.thesis or f"{branch_name} 分支已生成结构化判断。",
            symbol=symbol,
            status=value.status,
            direction=value.direction,
            action=value.action,
            confidence_label=value.confidence_label,
            final_score=float(value.final_score),
            final_confidence=float(value.final_confidence),
            evidence=value.evidence,
            investment_risks=list(value.investment_risks),
            coverage_notes=list(value.coverage_notes),
            diagnostic_notes=list(value.diagnostic_notes),
            metadata=dict(value.metadata or {}),
        )
        payload.metadata.setdefault("symbol", symbol)
        payload.metadata.setdefault("branch_name", branch_name)
        return payload
    if isinstance(value, BaseBranchAgentOutput):
        return _branch_output_to_verdict(value, symbol=symbol)
    raise TypeError(f"unsupported branch verdict type: {type(value)!r}")


async def _execute_market_dag_async(
    *,
    market: str,
    symbols: list[str] | None = None,
    universe: str | None = None,
    categories: list[str] | None = None,
    mode: str = "sample",
    batch_size: int | None,
    total_capital: float,
    top_k: int,
    download_stage: Mapping[str, Any] | None = None,
    data_snapshot: Mapping[str, Any] | None = None,
    verbose: bool = True,
    enable_agent_layer: bool = True,
    review_model_priority: list[str] | None = None,
    agent_model: str = "",
    agent_fallback_model: str = "",
    master_model: str = "",
    master_fallback_model: str = "",
    master_reasoning_effort: str = "high",
    agent_timeout: float = config.DEFAULT_AGENT_TIMEOUT_SECONDS,
    master_timeout: float = config.DEFAULT_MASTER_TIMEOUT_SECONDS,
    agent_layer_enabled: bool = True,
    funnel_profile: str = config.FUNNEL_PROFILE,
    max_candidates: int = config.FUNNEL_MAX_CANDIDATES,
    trend_windows: list[int] | tuple[int, ...] | None = None,
    volume_spike_threshold: float = config.FUNNEL_VOLUME_SPIKE_THRESHOLD,
    breakout_distance_pct: float = config.FUNNEL_BREAKOUT_DISTANCE_PCT,
    sector_bucket_limit: int = config.FUNNEL_SECTOR_BUCKET_LIMIT,
    recall_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_market_settings(market)
    selected_categories = (
        normalize_universe(settings.market, universe)
        if universe is not None
        else normalize_categories(settings.market, categories)
    )
    universe_key = universe or (selected_categories[0] if len(selected_categories) == 1 else "custom")

    resolver = CNUniverseResolver(data_dir=settings.data_dir) if settings.market == "CN" else None
    shared_reader = SharedCSVReader(market=settings.market, data_dir=settings.data_dir, resolver=resolver)

    explicit_symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()))
    if explicit_symbols:
        symbols = explicit_symbols
    elif settings.market == "CN" and universe_key == "full_a":
        symbols = shared_reader.list_symbols("full_a")
    else:
        symbols = []
        for category in selected_categories:
            symbols.extend(shared_reader.list_symbols(category))
        symbols = list(dict.fromkeys(symbols))
    if mode == "sample":
        symbols = symbols[: (batch_size or settings.default_batch_size)]

    branch_config, master_config = resolve_runtime_role_models(
        review_model_priority=review_model_priority,
        agent_model=agent_model,
        agent_fallback_model=agent_fallback_model,
        master_model=master_model,
        master_fallback_model=master_fallback_model,
    )
    agent_model = branch_config.primary_model
    agent_fallback_model = branch_config.fallback_model
    master_model = master_config.primary_model
    master_fallback_model = master_config.fallback_model
    branch_candidate_models = list(branch_config.candidate_models)
    master_candidate_models = list(master_config.candidate_models)

    scoped_data_snapshot = dict(
        data_snapshot
        or build_market_data_snapshot(
            market=settings.market,
            universe=universe_key,
            categories=selected_categories,
            requested_symbols=explicit_symbols,
        )
    )

    if not symbols:
        empty_context = GlobalContext(
            market=settings.market,
            universe_key=universe_key,
            universe_symbols=[],
            latest_trade_date=str(scoped_data_snapshot.get("local_latest_trade_date", "")),
            freshness_mode=str(scoped_data_snapshot.get("freshness_mode", "stable")),
            effective_target_trade_date=str(scoped_data_snapshot.get("local_latest_trade_date", "")),
            metadata={
                "resolver": shared_reader.snapshot(),
                "data_snapshot": scoped_data_snapshot,
                "selection_profile": {
                    "funnel_profile": str(funnel_profile or config.FUNNEL_PROFILE).strip().lower() or config.FUNNEL_PROFILE,
                    "trend_windows": list(trend_windows or config.FUNNEL_TREND_WINDOWS),
                    "volume_spike_threshold": float(volume_spike_threshold or config.FUNNEL_VOLUME_SPIKE_THRESHOLD),
                    "breakout_distance_pct": float(breakout_distance_pct or config.FUNNEL_BREAKOUT_DISTANCE_PCT),
                    "max_candidates": int(max_candidates or config.FUNNEL_MAX_CANDIDATES),
                    "sector_bucket_limit": int(sector_bucket_limit if sector_bucket_limit is not None else config.FUNNEL_SECTOR_BUCKET_LIMIT),
                },
            },
        )
        empty_decision = PortfolioDecision()
        empty_trace = build_execution_trace(
            model_roles=build_model_role_metadata(
                branch_model=agent_model,
                master_model=master_model,
                agent_fallback_model=agent_fallback_model,
                master_fallback_model=master_fallback_model,
                resolved_branch_model=agent_model,
                resolved_master_model=master_model,
                master_reasoning_effort=master_reasoning_effort,
                agent_layer_enabled=agent_layer_enabled,
                universe_key=universe_key,
                universe_size=0,
                universe_hash="",
                metadata={"resolver": shared_reader.snapshot()},
            ),
            analysis_meta={"batch_count": 0, "category_count": 0, "total_stocks": 0},
            portfolio_plan={"selected_count": 0, "target_exposure": 0.0, "max_single_weight": 0.0, "risk_veto": False},
            download_stage=download_stage,
        )
        what_if = build_what_if_plan(
            portfolio_plan=empty_decision.to_dict(),
            market_summary={"candidate_count": 0, "macro_score": 0.0},
            model_roles=empty_trace.model_roles,
            candidate_count=0,
            selected_count=0,
        )
        return {
            "global_context": empty_context,
            "symbol_research_packets": {},
            "branch_verdicts_by_symbol": {},
            "branch_summaries": {},
            "macro_verdict": MacroAgent().run({"market_snapshot": {"regime": "neutral", "macro_score": 0.0, "liquidity_score": 0.0}}),
            "risk_decision": RiskDecision(),
            "ic_decisions": [],
            "shortlist": [],
            "portfolio_plan": PortfolioPlan(),
            "portfolio_decision": empty_decision,
            "review_bundle": StockReviewBundle(),
            "model_role_metadata": empty_trace.model_roles,
            "what_if_plan": what_if,
            "execution_trace": empty_trace,
            "data_quality_issues": [],
            "resolver": shared_reader.snapshot(),
            "data_snapshot": scoped_data_snapshot,
            "tradability_snapshot": {},
            "portfolio_master_output": None,
            "portfolio_master_meta": {"status": "empty"},
        }

    all_symbols = list(symbols)
    company_name_map = _load_company_name_map(settings.market)
    company_profile_map = _load_company_profile_map(settings.market)
    for symbol, payload in company_profile_map.items():
        name = str(payload.get("name", "")).strip()
        if name and symbol not in company_name_map:
            company_name_map[symbol] = name

    branch_model_resolution: ModelRoleResolution = resolve_model_role(
        role="branch",
        primary_model=agent_model,
        fallback_model=agent_fallback_model,
    )
    master_model_resolution: ModelRoleResolution = resolve_model_role(
        role="master",
        primary_model=master_model,
        fallback_model=master_fallback_model,
    )
    macro_agent = MacroAgent()
    kline_agent = KlineAgent()
    context_state = _prepare_market_context(
        market=settings.market,
        universe_key=universe_key,
        selected_categories=selected_categories,
        symbols=all_symbols,
        company_profile_map=company_profile_map,
        shared_reader=shared_reader,
        scoped_data_snapshot=scoped_data_snapshot,
        download_stage=download_stage,
        enable_agent_layer=enable_agent_layer,
        agent_timeout=agent_timeout,
        master_timeout=master_timeout,
        master_reasoning_effort=master_reasoning_effort,
        branch_model_resolution=branch_model_resolution,
        master_model_resolution=master_model_resolution,
        branch_candidate_models=branch_candidate_models,
        master_candidate_models=master_candidate_models,
        company_name_map=company_name_map,
        funnel_profile=str(funnel_profile or config.FUNNEL_PROFILE).strip().lower() or config.FUNNEL_PROFILE,
        max_candidates=max(1, int(max_candidates or config.FUNNEL_MAX_CANDIDATES)),
        trend_windows=tuple(int(item) for item in (trend_windows or config.FUNNEL_TREND_WINDOWS) if int(item) > 0) or tuple(config.FUNNEL_TREND_WINDOWS),
        volume_spike_threshold=float(volume_spike_threshold or config.FUNNEL_VOLUME_SPIKE_THRESHOLD),
        breakout_distance_pct=float(breakout_distance_pct or config.FUNNEL_BREAKOUT_DISTANCE_PCT),
        sector_bucket_limit=max(0, int(sector_bucket_limit if sector_bucket_limit is not None else config.FUNNEL_SECTOR_BUCKET_LIMIT)),
        macro_agent=macro_agent,
        kline_agent=kline_agent,
        funnel_cls=DeterministicFunnel,
        provider_health_detector=detect_provider_health,
        ensure_branch_verdict=_ensure_branch_verdict,
        branch_verdict_to_result=_branch_verdict_to_result,
    )
    read_results = context_state.read_results
    frames = context_state.frames
    tradability_snapshot = context_state.tradability_snapshot
    data_quality_issues = context_state.data_quality_issues
    quarantined_symbols = context_state.quarantined_symbols
    researchable_symbols = context_state.researchable_symbols
    candidate_symbols = context_state.candidate_symbols
    provider_health = context_state.provider_health
    market_snapshot = context_state.market_snapshot
    macro_verdict = context_state.macro_verdict
    global_quant_verdict = context_state.global_quant_verdict
    quant_result = context_state.quant_result
    global_context = context_state.global_context
    model_roles = context_state.model_roles
    full_market_kline_result = context_state.full_market_kline_result
    funnel_output = context_state.funnel_output

    fundamental_agent = FundamentalAgent()
    intelligence_agent = IntelligenceAgent()
    research_state = await _run_candidate_research_phase(
        candidate_symbols=candidate_symbols,
        company_name_map=company_name_map,
        market=settings.market,
        market_snapshot=market_snapshot,
        universe_key=universe_key,
        read_results=read_results,
        frames=frames,
        global_quant_verdict=global_quant_verdict,
        macro_verdict=macro_verdict,
        branch_model_resolution=branch_model_resolution,
        master_model_resolution=master_model_resolution,
        branch_candidate_models=branch_candidate_models,
        master_candidate_models=master_candidate_models,
        master_reasoning_effort=master_reasoning_effort,
        enable_agent_layer=enable_agent_layer,
        agent_timeout=agent_timeout,
        master_timeout=master_timeout,
        resolver_snapshot=context_state.resolver_snapshot,
        kline_agent=kline_agent,
        fundamental_agent=fundamental_agent,
        intelligence_agent=intelligence_agent,
        quant_result=quant_result,
        full_market_kline_result=full_market_kline_result,
        ensure_branch_verdict=_ensure_branch_verdict,
        master_hint_to_ic_hint=_master_hint_to_ic_hint,
    )
    symbol_research_packets = research_state.symbol_research_packets
    research_by_symbol = research_state.research_by_symbol
    review_bundle = research_state.review_bundle
    ic_hints_by_symbol = research_state.ic_hints_by_symbol
    branch_summaries = research_state.branch_summaries
    branch_results = research_state.branch_results

    selection_state = _run_bayesian_selection_phase(
        candidate_symbols=candidate_symbols,
        company_name_map=company_name_map,
        symbol_research_packets=symbol_research_packets,
        research_by_symbol=research_by_symbol,
        branch_summaries=branch_summaries,
        branch_results=branch_results,
        macro_verdict=macro_verdict,
        global_context=global_context,
        model_roles=model_roles,
        resolver_snapshot=shared_reader.snapshot(),
        data_quality_issues=data_quality_issues,
        top_k=top_k,
        all_symbols=all_symbols,
        funnel_output=funnel_output,
        provider_health=provider_health,
        master_timeout=master_timeout,
        master_reasoning_effort=master_reasoning_effort,
        master_model_resolution=master_model_resolution,
        master_candidate_models=master_candidate_models,
        recall_context=recall_context,
        hierarchical_prior_builder_cls=HierarchicalPriorBuilder,
        likelihood_mapper_cls=SignalLikelihoodMapper,
        posterior_engine_cls=BayesianPosteriorEngine,
        master_agent_cls=MasterAgent,
        llm_client_cls=GatewayLLMClient,
        portfolio_master_advisory_fn=_portfolio_master_advisory,
    )

    decision_state = _run_portfolio_construction_phase(
        shortlist=selection_state.shortlist,
        branch_summaries=branch_summaries,
        macro_verdict=macro_verdict,
        global_context=global_context,
        data_quality_issues=data_quality_issues,
        ic_hints_by_symbol=ic_hints_by_symbol,
        research_by_symbol=research_by_symbol,
        tradability_snapshot=tradability_snapshot,
        funnel_summary=selection_state.funnel_summary,
        bayesian_records=selection_state.bayesian_records,
        candidate_symbols=candidate_symbols,
        portfolio_master_output=selection_state.portfolio_master_output,
        portfolio_master_meta=selection_state.portfolio_master_meta,
        risk_guard_cls=RiskGuard,
        ic_coordinator_cls=ICCoordinator,
        portfolio_constructor_cls=PortfolioConstructor,
        attach_symbol_to_ic_decision_fn=_attach_symbol_to_ic_decision,
    )

    reporting_state = _build_reporting_artifacts(
        market=settings.market,
        universe_key=universe_key,
        all_symbols=all_symbols,
        researchable_symbols=researchable_symbols,
        candidate_symbols=candidate_symbols,
        quarantined_symbols=quarantined_symbols,
        data_quality_issues=data_quality_issues,
        read_results=read_results,
        shared_reader=shared_reader,
        global_context=global_context,
        provider_health=provider_health,
        model_roles=model_roles,
        funnel_summary=selection_state.funnel_summary,
        bayesian_records=selection_state.bayesian_records,
        review_bundle=review_bundle,
        ic_hints_by_symbol=ic_hints_by_symbol,
        macro_verdict=macro_verdict,
        branch_summaries=branch_summaries,
        branch_verdicts_by_symbol=research_by_symbol,
        branch_results=branch_results,
        ic_decisions=decision_state.ic_decisions,
        portfolio_plan=decision_state.portfolio_plan,
        portfolio_decision=decision_state.portfolio_decision,
        symbol_research_packets=symbol_research_packets,
        shortlist=selection_state.shortlist,
        portfolio_master_output=selection_state.portfolio_master_output,
        portfolio_master_meta=selection_state.portfolio_master_meta,
        portfolio_master_reliability=selection_state.portfolio_master_reliability,
        risk_decision=decision_state.risk_decision,
        tradability_snapshot=tradability_snapshot,
        scoped_data_snapshot=scoped_data_snapshot,
        download_stage=download_stage,
        category_count=len(selected_categories),
        funnel_output=funnel_output,
        global_quant_verdict=global_quant_verdict,
        narrator_agent_cls=NarratorAgent,
        build_data_quality_diagnostics_fn=build_data_quality_diagnostics,
        build_what_if_plan_fn=build_what_if_plan,
        build_execution_trace_fn=build_execution_trace,
        build_bayesian_trace_fn=build_bayesian_trace,
    )
    return reporting_state.dag_artifacts


def execute_market_dag(
    *,
    market: str,
    symbols: list[str] | None = None,
    universe: str | None = None,
    categories: list[str] | None = None,
    mode: str = "sample",
    batch_size: int | None,
    total_capital: float,
    top_k: int,
    download_stage: Mapping[str, Any] | None = None,
    data_snapshot: Mapping[str, Any] | None = None,
    verbose: bool = True,
    enable_agent_layer: bool = True,
    review_model_priority: list[str] | None = None,
    agent_model: str = "",
    agent_fallback_model: str = "",
    master_model: str = "",
    master_fallback_model: str = "",
    master_reasoning_effort: str = "high",
    agent_timeout: float = config.DEFAULT_AGENT_TIMEOUT_SECONDS,
    master_timeout: float = config.DEFAULT_MASTER_TIMEOUT_SECONDS,
    funnel_profile: str = config.FUNNEL_PROFILE,
    max_candidates: int = config.FUNNEL_MAX_CANDIDATES,
    trend_windows: list[int] | tuple[int, ...] | None = None,
    volume_spike_threshold: float = config.FUNNEL_VOLUME_SPIKE_THRESHOLD,
    breakout_distance_pct: float = config.FUNNEL_BREAKOUT_DISTANCE_PCT,
    sector_bucket_limit: int = config.FUNNEL_SECTOR_BUCKET_LIMIT,
    recall_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        _execute_market_dag_async(
            market=market,
            symbols=symbols,
            universe=universe,
            categories=categories,
            mode=mode,
            batch_size=batch_size,
            total_capital=total_capital,
            top_k=top_k,
            download_stage=download_stage,
            data_snapshot=data_snapshot,
            verbose=verbose,
            enable_agent_layer=enable_agent_layer,
            review_model_priority=review_model_priority,
            agent_model=agent_model,
            agent_fallback_model=agent_fallback_model,
            master_model=master_model,
            master_fallback_model=master_fallback_model,
            master_reasoning_effort=master_reasoning_effort,
            agent_timeout=agent_timeout,
            master_timeout=master_timeout,
            agent_layer_enabled=enable_agent_layer,
            funnel_profile=funnel_profile,
            max_candidates=max_candidates,
            trend_windows=trend_windows,
            volume_spike_threshold=volume_spike_threshold,
            breakout_distance_pct=breakout_distance_pct,
            sector_bucket_limit=sector_bucket_limit,
            recall_context=recall_context,
        )
    )


__all__ = [
    "_build_master_evidence_pack",
    "_compact_trace_fragments",
    "_build_shortlist",
    "_build_shortlist_from_bayesian_records",
    "_build_market_snapshot",
    "_build_symbol_bundle",
    "_build_symbol_research_packet",
    "_aggregate_branch_summaries",
    "_build_branch_results",
    "_run_async_coroutine_safely",
    "_prepare_market_context",
    "_run_candidate_research_phase",
    "_run_bayesian_selection_phase",
    "_run_portfolio_construction_phase",
    "_build_reporting_artifacts",
    "execute_market_dag",
]
