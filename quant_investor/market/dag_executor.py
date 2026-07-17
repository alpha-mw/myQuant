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

import asyncio
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ConfidenceLabel,
    Direction,
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
from quant_investor.agents.macro_agent import MacroAgent
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.narrator_agent import NarratorAgent
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.agents.risk_guard import RiskGuard
from quant_investor.agents.llm_client import LLMClient as GatewayLLMClient
from quant_investor.bayesian.likelihood import SignalLikelihoodMapper
from quant_investor.bayesian.posterior import BayesianPosteriorEngine
from quant_investor.bayesian.prior import HierarchicalPriorBuilder
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.config import config
from quant_investor.funnel.deterministic_funnel import DeterministicFunnel
from quant_investor.market.config import get_market_settings, normalize_categories, normalize_universe
from quant_investor.market.data_snapshot import build_market_data_snapshot
from quant_investor.market.data_quality import build_data_quality_diagnostics
from quant_investor.market.dag.assembly import (
    _aggregate_branch_summaries,
    _attach_symbol_to_ic_decision,
    _build_branch_results,
)
from quant_investor.market.dag.common import _run_async_coroutine_safely, _score_to_action
from quant_investor.market.dag.context import (
    DAG_SINGLE_NAME_WEIGHT_CAP,
    _blocked_macro_verdict,
    _prepare_market_context,
    _resolve_effective_data_state,
)
from quant_investor.market.dag.decision import (
    PortfolioConstructionState,
    _build_counterfactual_control_inputs,
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
from quant_investor.v16.runtime import (
    DEFAULT_CODEX_REVIEW_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_FACTOR_READINESS_PATH,
    DEFAULT_STAGE1_PROMPT_PATH,
    V16Stage1RuntimeError,
    load_v16_factor_readiness,
    prepare_v16_stage1_pending,
)
from quant_investor.market.dag.review import _portfolio_master_advisory
from quant_investor.market.dag.shortlist import _build_shortlist, _build_shortlist_from_bayesian_records
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.branch_readiness import (
    STATUS_BLOCK,
    assess_macro_readiness,
    load_macro_record,
    macro_generation_identity,
)
from quant_investor.market.macro_readiness_runtime import (
    FrozenMacroReadinessRuntime,
    freeze_macro_readiness_runtime,
)
from quant_investor.market.name_map import (
    load_company_name_map as _load_cached_company_name_map,
)
from quant_investor.market.pit_universe import filter_symbols_by_pit_status
from quant_investor.market.provider_health import detect_provider_health
from quant_investor.market.runtime_profile import profile_stage
from quant_investor.market.us_market_cap_filter import USMarketCapFilter
from quant_investor.llm_policy import llm_handoff_metadata, local_llm_disabled
from quant_investor.llm_provider_priority import resolve_runtime_role_models
from quant_investor.model_roles import ModelRoleResolution, resolve_model_role
from quant_investor.reporting.run_artifacts import (
    build_bayesian_trace,
    build_execution_trace,
    build_model_role_metadata,
    build_what_if_plan,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _codex_handoff_model_resolution(
    *,
    role: str,
    primary_model: str,
    fallback_model: str,
) -> ModelRoleResolution:
    metadata = {
        **llm_handoff_metadata(),
        "agent_layer_enabled": True,
        "review_layer_mode": "codex_handoff",
        "codex_handoff_pending": True,
    }
    return ModelRoleResolution(
        role=role,
        primary_model=primary_model,
        fallback_model=fallback_model,
        resolved_model="codex-handoff",
        fallback_used=False,
        fallback_reason=str(
            metadata.get("handoff_reason")
            or "local_llm_disabled_codex_handoff"
        ),
        provider_available=False,
        fallback_provider_available=False,
        metadata=metadata,
    )


def _counterfactual_replay_ready(selection_state: Any) -> tuple[bool, set[str]]:
    variants = {
        str(item.get("basis") or "")
        for item in selection_state.counterfactual_by_symbol.values()
    }
    return bool(selection_state.counterfactual_by_symbol) and len(variants) == 1, variants


def _empty_universe_macro_contract(
    *,
    as_of: str,
) -> tuple[
    dict[str, Any],
    Any,
    dict[str, str],
    BranchVerdict,
    FrozenMacroReadinessRuntime,
]:
    record, manifest = load_macro_record(as_of=as_of)
    runtime = freeze_macro_readiness_runtime(
        macro_logical_date=str(record.get("trade_date") or ""),
        target_session_date=as_of,
    )
    readiness = assess_macro_readiness(
        macro_record=record,
        manifest=manifest,
        as_of=as_of,
        decision_cutoff_at=runtime.decision_cutoff_at or None,
        macro_readiness_evidence=runtime.evidence,
    )
    identity = macro_generation_identity(manifest)
    if readiness.status == STATUS_BLOCK:
        verdict = _blocked_macro_verdict(
            blockers=list(readiness.blockers),
            generation_identity=identity,
        )
        return record, readiness, identity, verdict, runtime

    market_snapshot = {
        "regime": "neutral",
        "macro_score": float(record["macro_score"]),
        "liquidity_score": float(record["liquidity_score"]),
        "volatility_percentile": float(record["volatility_percentile"]),
        "policy_signal": str(record["policy_signal"]),
        "macro_data_readiness_status": readiness.status,
        "canonical_macro_generation": dict(identity),
        "decision_authorized": True,
    }
    verdict = MacroAgent().run({"market_snapshot": market_snapshot})
    verdict.metadata = dict(verdict.metadata or {})
    verdict.metadata.update(
        {
            "decision_authorized": True,
            "macro_data_readiness_status": readiness.status,
            "canonical_macro_generation": dict(identity),
        }
    )
    return record, readiness, identity, verdict, runtime


def _blocked_macro_control_state(
    *,
    global_context: GlobalContext,
    tradability_snapshot: Mapping[str, Mapping[str, Any]],
) -> PortfolioConstructionState:
    """Stop before the deterministic control-chain agents on blocked Macro."""

    identity = dict(
        global_context.metadata.get("canonical_macro_generation", {}) or {}
    )
    risk_decision = RiskDecision(
        status=AgentStatus.VETOED,
        hard_veto=True,
        veto=True,
        action_cap=ActionLabel.HOLD,
        max_weight=DAG_SINGLE_NAME_WEIGHT_CAP,
        gross_exposure_cap=0.55,
        target_exposure_cap=0.55,
        reasons=[
            "Canonical Macro readiness is blocked; no new portfolio "
            "decision is authorized."
        ],
        metadata={
            "decision_authorized": False,
            "branch_fusion_blocked": True,
            "canonical_macro_generation": identity,
        },
    )
    portfolio_plan = PortfolioPlan(
        status=AgentStatus.VETOED,
        metadata={
            "decision_authorized": False,
            "branch_fusion_blocked": True,
            "reason": "canonical_macro_readiness_blocked",
            "canonical_macro_generation": identity,
        },
    )
    portfolio_decision = PortfolioDecision(
        status=AgentStatus.VETOED,
        risk_constraints={
            "risk_decision": risk_decision.to_dict(),
            "tradability_snapshot": dict(tradability_snapshot),
        },
        metadata={
            "decision_authorized": False,
            "branch_fusion_blocked": True,
            "reason": "canonical_macro_readiness_blocked",
            "canonical_macro_generation": identity,
            "risk_summary": risk_decision.to_dict(),
        },
    )
    return PortfolioConstructionState(
        risk_decision=risk_decision,
        ic_decisions=[],
        portfolio_plan=portfolio_plan,
        portfolio_decision=portfolio_decision,
    )


def _load_company_name_map(market: str) -> dict[str, str]:
    if str(market or "").strip().upper() != "CN":
        return {}
    return _load_cached_company_name_map(market, allow_provider=False)


def _load_company_profile_map(market: str) -> dict[str, dict[str, str]]:
    if str(market or "").strip().upper() != "CN":
        return {}
    result = _load_parquet_company_profile_map(market)
    db_path = Path(str(getattr(config, "DB_PATH", "") or "")).expanduser()
    if not db_path.exists():
        return result
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "stock_list" not in tables and "stock_profiles" not in tables:
            conn.close()
            return result
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
        return result
    for row in rows:
        symbol = str(row["ts_code"] or "").strip().upper()
        if not symbol:
            continue
        name = str(row["name"] or "").strip()
        industry = str(row["industry"] or "").strip()
        sector = str(row["sector"] or industry or "").strip()
        payload = result.setdefault(symbol, {})
        if name and not str(payload.get("name") or "").strip():
            payload["name"] = name
        if industry and not str(payload.get("industry") or "").strip():
            payload["industry"] = industry
        if sector and not str(payload.get("sector") or "").strip():
            payload["sector"] = sector
        if not str(payload.get("profile_source") or "").strip():
            payload["profile_source"] = "sqlite_profile"
    return result


def _load_parquet_company_profile_map(market: str) -> dict[str, dict[str, str]]:
    market_key = str(market or "").strip().lower()
    if market_key != "cn":
        return {}
    data_root = Path(str(getattr(config, "DATA_DIR", "data") or "data")).expanduser()
    table_path = data_root / "parquet" / market_key / "dag_core_raw" / "table=stock_basic"
    if not table_path.exists():
        return {}
    try:
        frame = pd.read_parquet(table_path)
    except Exception:
        return {}
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ts_code" not in frame.columns:
        return {}
    result: dict[str, dict[str, str]] = {}
    for row in frame.itertuples(index=False):
        symbol = str(getattr(row, "ts_code", "") or "").strip().upper()
        if not symbol:
            continue
        name = str(getattr(row, "name", "") or "").strip()
        industry = str(getattr(row, "industry", "") or "").strip()
        sector = str(getattr(row, "sector", "") or "").strip() or industry
        if not any((name, industry, sector)):
            continue
        result[symbol] = {
            "name": name,
            "industry": industry,
            "sector": sector,
            "profile_source": "canonical_parquet_stock_basic",
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

def _branch_output_to_verdict(output: BaseBranchAgentOutput, symbol: str) -> BranchVerdict:
    action = str(output.conviction).lower()
    if action not in {"strong_buy", "buy", "neutral", "sell", "strong_sell"}:
        action = "neutral"
    return BranchVerdict(
        agent_name=str(output.branch_name or ""),
        thesis=str(output.reasoning or "") or "分支已完成结构化判断。",
        symbol=symbol,
        status=AgentStatus.SUCCESS,
        direction=Direction(_score_to_direction(float(output.conviction_score))),
        action=_score_to_action(float(output.conviction_score)),
        confidence_label=ConfidenceLabel.MEDIUM,
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
    shortlist_size: int | None = None,
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
    runtime_profiler: Any | None = None,
    decision_protocol: str = "v15",
    v16_factor_readiness_path: str = DEFAULT_FACTOR_READINESS_PATH,
    v16_review_root: str = DEFAULT_CODEX_REVIEW_ROOT,
    v16_config_path: str = str(DEFAULT_CONFIG_PATH),
    v16_prompt_path: str = str(DEFAULT_STAGE1_PROMPT_PATH),
    v16_run_id: str = "",
    v16_expiry_hours: float = 24.0,
) -> dict[str, Any]:
    protocol = str(decision_protocol or "v15").strip().lower()
    if protocol not in {"v15", "v16"}:
        raise ValueError("decision_protocol must be v15 or v16")
    settings = get_market_settings(market)
    selected_categories = (
        normalize_universe(settings.market, universe)
        if universe is not None
        else normalize_categories(settings.market, categories)
    )
    universe_key = universe or (selected_categories[0] if len(selected_categories) == 1 else "custom")

    shared_reader = MarketDataReader(market=settings.market)

    explicit_symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()))
    market_cap_filter_metadata: dict[str, Any] = {}
    with profile_stage(
        runtime_profiler,
        "dag_symbol_list",
        {
            "market": settings.market,
            "universe_key": universe_key,
            "category_count": len(selected_categories),
            "mode": mode,
            "explicit_symbol_count": len(explicit_symbols),
        },
    ) as stage_metadata:
        if explicit_symbols:
            symbols = explicit_symbols
        elif settings.market == "CN" and universe_key == "full_a":
            symbols = shared_reader.list_symbols("full_a")
        else:
            symbols = []
            for category in selected_categories:
                symbols.extend(shared_reader.list_symbols(category))
            symbols = list(dict.fromkeys(symbols))
        if settings.market == "US":
            symbols, market_cap_filter_metadata = USMarketCapFilter().filter_symbols(symbols, fetch_missing=True)
            if explicit_symbols:
                explicit_symbols = list(symbols)
        unsampled_symbol_count = len(symbols)
        if mode == "sample":
            symbols = symbols[: (batch_size or settings.default_batch_size)]
        stage_metadata["symbol_count"] = len(symbols)
        stage_metadata["unsampled_symbol_count"] = unsampled_symbol_count
        stage_metadata["sampled"] = bool(mode == "sample")

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
    if market_cap_filter_metadata:
        scoped_data_snapshot["market_cap_filter"] = market_cap_filter_metadata
    pit_universe_metadata: dict[str, Any] = {
        "enabled": bool(getattr(config, "PIT_UNIVERSE_ENABLED", False)),
        "required": bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)),
        "status": "disabled",
    }
    if settings.market == "CN" and bool(getattr(config, "PIT_UNIVERSE_ENABLED", False)):
        pit_as_of = str(
            scoped_data_snapshot.get("local_latest_trade_date")
            or scoped_data_snapshot.get("latest_trade_date")
            or ""
        )
        pit_binding = shared_reader.coverage_bound_pit()
        pit_records = pit_binding.get("records", {})
        pit_filter = filter_symbols_by_pit_status(
            symbols,
            as_of=pit_as_of,
            records=pit_records,
            required=bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)),
        )
        symbols = pit_filter.symbols
        pit_universe_metadata = dict(pit_filter.metadata)
        pit_universe_metadata["status"] = (
            "applied"
            if pit_binding.get("status") == "passed" and pit_records
            else "missing_store"
        )
        pit_manifest = dict(pit_binding.get("manifest", {}) or {})
        pit_universe_metadata["snapshot_id"] = str(
            pit_manifest.get("source_run_id") or ""
        )
        pit_universe_metadata["generation_id"] = str(
            pit_binding.get("generation_id") or ""
        )
        pit_universe_metadata["manifest_path"] = str(
            pit_binding.get("generation_manifest_path") or ""
        )
        pit_universe_metadata["canonical_path"] = str(
            pit_binding.get("canonical_path") or ""
        )
        pit_universe_metadata["canonical_sha256"] = str(
            pit_binding.get("canonical_sha256") or ""
        )
        pit_universe_metadata["binding_source"] = "market_coverage"
        pit_universe_metadata["binding_blockers"] = list(
            pit_binding.get("blockers", []) or []
        )
        pit_universe_metadata["quarantine_symbols"] = list(pit_filter.quarantine_symbols)
        pit_universe_metadata["untradable_symbols"] = list(pit_filter.untradable_symbols)
    # The current run's local PIT filter is authoritative.  Never accept a
    # caller-injected snapshot claim when PIT is disabled or unavailable.
    scoped_data_snapshot["pit_universe"] = pit_universe_metadata

    if not symbols:
        if protocol == "v16":
            raise V16Stage1RuntimeError(
                "v16 Stage 1 cannot be prepared from an empty universe"
            )
        (
            empty_completeness,
            empty_as_of,
            empty_freshness_mode,
        ) = _resolve_effective_data_state(
            scoped_data_snapshot=scoped_data_snapshot,
            download_stage=download_stage,
        )
        (
            pinned_macro_record,
            pinned_macro_readiness,
            pinned_macro_identity,
            empty_macro_verdict,
            pinned_macro_runtime,
        ) = _empty_universe_macro_contract(as_of=empty_as_of)
        macro_blocked = pinned_macro_readiness.status == STATUS_BLOCK
        baseline_target_exposure = float(
            empty_macro_verdict.metadata.get("target_gross_exposure", 0.55)
        )
        empty_status = (
            AgentStatus.VETOED if macro_blocked else AgentStatus.SUCCESS
        )
        empty_decision = PortfolioDecision(
            status=empty_status,
            metadata={
                "decision_authorized": False,
                "reason": "empty_universe",
                "canonical_macro_generation": dict(pinned_macro_identity),
            },
        )
        empty_plan = PortfolioPlan(
            status=empty_status,
            metadata={
                "decision_authorized": False,
                "reason": "empty_universe",
            },
        )
        empty_risk = RiskDecision(
            status=empty_status,
            hard_veto=macro_blocked,
            veto=macro_blocked,
            action_cap=(
                ActionLabel.HOLD if macro_blocked else ActionLabel.BUY
            ),
            max_weight=DAG_SINGLE_NAME_WEIGHT_CAP,
            gross_exposure_cap=baseline_target_exposure,
            target_exposure_cap=baseline_target_exposure,
            reasons=[
                "No portfolio decision is authorized for an empty universe."
            ],
            metadata={
                "decision_authorized": False,
                "canonical_macro_generation": dict(pinned_macro_identity),
            },
        )
        empty_context = GlobalContext(
            market=settings.market,
            universe_key=universe_key,
            universe_symbols=[],
            latest_trade_date=empty_as_of,
            freshness_mode=empty_freshness_mode,
            effective_target_trade_date=str(
                empty_completeness.get("effective_target_trade_date")
                or empty_as_of
            ),
            macro_regime=str(
                empty_macro_verdict.metadata.get("regime", "neutral")
            ),
            macro_data=(
                dict(pinned_macro_record) if not macro_blocked else {}
            ),
            risk_budget={
                "target_exposure": baseline_target_exposure,
                "max_single_weight": DAG_SINGLE_NAME_WEIGHT_CAP,
                "baseline_target_exposure": baseline_target_exposure,
                "baseline_max_single_weight": DAG_SINGLE_NAME_WEIGHT_CAP,
            },
            metadata={
                "resolver": shared_reader.snapshot(),
                "data_snapshot": scoped_data_snapshot,
                "market_cap_filter": market_cap_filter_metadata,
                "pit_universe": pit_universe_metadata,
                "canonical_macro_generation": dict(pinned_macro_identity),
                "canonical_macro_readiness": (
                    pinned_macro_readiness.to_dict()
                ),
                "macro_readiness_runtime": (
                    pinned_macro_runtime.metadata()
                ),
                "branch_fusion_blocked": macro_blocked,
                "decision_authorized": False,
                "empty_universe": True,
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
            portfolio_plan={
                "selected_count": 0,
                "target_exposure": baseline_target_exposure,
                "max_single_weight": DAG_SINGLE_NAME_WEIGHT_CAP,
                "risk_veto": macro_blocked,
                "action_cap": empty_risk.action_cap.value,
                "risk_summary": {
                    "status": empty_risk.status.value,
                    "hard_veto": empty_risk.hard_veto,
                    "veto": empty_risk.veto,
                    "action_cap": empty_risk.action_cap.value,
                    "gross_exposure_cap": empty_risk.gross_exposure_cap,
                    "target_exposure_cap": empty_risk.target_exposure_cap,
                    "max_weight": empty_risk.max_weight,
                    "decision_authorized": False,
                },
            },
            download_stage=download_stage,
        )
        what_if = build_what_if_plan(
            portfolio_plan=empty_decision.to_dict(),
            market_summary={
                "candidate_count": 0,
                "macro_score": float(empty_macro_verdict.final_score),
            },
            model_roles=empty_trace.model_roles,
            candidate_count=0,
            selected_count=0,
        )
        return {
            "global_context": empty_context,
            "symbol_research_packets": {},
            "branch_verdicts_by_symbol": {},
            "branch_summaries": {},
            "macro_verdict": empty_macro_verdict,
            "risk_decision": empty_risk,
            "ic_decisions": [],
            "shortlist": [],
            "portfolio_plan": empty_plan,
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

    codex_handoff_review = bool(enable_agent_layer) and local_llm_disabled()
    if enable_agent_layer and codex_handoff_review:
        branch_model_resolution = _codex_handoff_model_resolution(
            role="branch",
            primary_model=agent_model,
            fallback_model=agent_fallback_model,
        )
        master_model_resolution = _codex_handoff_model_resolution(
            role="master",
            primary_model=master_model,
            fallback_model=master_fallback_model,
        )
    elif enable_agent_layer:
        branch_model_resolution = resolve_model_role(
            role="branch",
            primary_model=agent_model,
            fallback_model=agent_fallback_model,
        )
        master_model_resolution = resolve_model_role(
            role="master",
            primary_model=master_model,
            fallback_model=master_fallback_model,
        )
    else:
        branch_model_resolution = ModelRoleResolution(
            role="branch",
            primary_model=agent_model,
            fallback_model=agent_fallback_model,
            resolved_model=agent_model or agent_fallback_model,
            fallback_reason="agent_layer_disabled",
            metadata={"agent_layer_enabled": False},
        )
        master_model_resolution = ModelRoleResolution(
            role="master",
            primary_model=master_model,
            fallback_model=master_fallback_model,
            resolved_model=master_model or master_fallback_model,
            fallback_reason="agent_layer_disabled",
            metadata={"agent_layer_enabled": False},
        )
    factor_readiness_v4 = None
    if protocol == "v16" and enable_agent_layer:
        # Factor readiness belongs to Eligibility and must be proved before
        # formal Quant is evaluated.  Missing/legacy evidence fails closed.
        factor_readiness_v4 = load_v16_factor_readiness(
            v16_factor_readiness_path
        )
    macro_agent = MacroAgent()
    with profile_stage(
        runtime_profiler,
        "dag_context_build",
        {"symbol_count": len(all_symbols), "universe_key": universe_key},
    ) as stage_metadata:
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
            funnel_cls=DeterministicFunnel,
            provider_health_detector=detect_provider_health,
            runtime_profiler=runtime_profiler,
            explicit_symbol_count=len(explicit_symbols),
            unsampled_symbol_count=unsampled_symbol_count,
            sampled=bool(mode == "sample"),
            recall_context=recall_context,
            full_market_branch_readiness=protocol == "v16",
            persist_branch_readiness=protocol != "v16",
        )
        stage_metadata["researchable_count"] = len(context_state.researchable_symbols)
        stage_metadata["candidate_count"] = len(context_state.candidate_symbols)
        stage_metadata["quarantined_count"] = len(context_state.quarantined_symbols)
    if protocol == "v16":
        pending = prepare_v16_stage1_pending(
            context_state=context_state,
            market=settings.market,
            mode=mode,
            enable_agent_layer=enable_agent_layer,
            factor_readiness_path=v16_factor_readiness_path,
            factor_readiness=factor_readiness_v4,
            review_root=v16_review_root,
            config_path=v16_config_path,
            prompt_path=v16_prompt_path,
            model_id=(
                branch_model_resolution.resolved_model
                or branch_model_resolution.primary_model
                or "codex-unresolved"
            ),
            run_id=v16_run_id,
            expiry_hours=v16_expiry_hours,
        )
        return {
            "decision_protocol": "v16",
            "status": pending["status"],
            "v16_stage1": pending,
            "formal_shortlist_generated": False,
            "new_risk_authorized": False,
            "data_snapshot": scoped_data_snapshot,
        }
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
    funnel_output = context_state.funnel_output

    fundamental_agent = FundamentalAgent()
    with profile_stage(
        runtime_profiler,
        "dag_candidate_research",
        {"candidate_count": len(candidate_symbols), "agent_layer_enabled": bool(enable_agent_layer)},
    ) as stage_metadata:
        research_state = await _run_candidate_research_phase(
            candidate_symbols=candidate_symbols,
            company_name_map=company_name_map,
            industry_map={
                symbol: str(
                    dict(tradability_snapshot.get(symbol, {}) or {}).get("industry")
                    or dict(tradability_snapshot.get(symbol, {}) or {}).get("sector")
                    or ""
                )
                for symbol in candidate_symbols
            },
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
            branch_data_readiness=context_state.branch_data_readiness,
            branch_data_payload=context_state.branch_data_payload,
            fundamental_agent=fundamental_agent,
            quant_result=quant_result,
            ensure_branch_verdict=_ensure_branch_verdict,
            master_hint_to_ic_hint=_master_hint_to_ic_hint,
        )
        stage_metadata["packet_count"] = len(research_state.symbol_research_packets)
        stage_metadata["branch_result_symbol_count"] = len(research_state.research_by_symbol)
    symbol_research_packets = research_state.symbol_research_packets
    research_by_symbol = research_state.research_by_symbol
    review_bundle = research_state.review_bundle
    ic_hints_by_symbol = research_state.ic_hints_by_symbol
    branch_summaries = research_state.branch_summaries
    branch_results = research_state.branch_results

    with profile_stage(
        runtime_profiler,
        "dag_bayesian_selection",
        {"candidate_count": len(candidate_symbols), "top_k": max(1, int(shortlist_size if shortlist_size is not None else top_k))},
    ) as stage_metadata:
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
            top_k=max(1, int(shortlist_size if shortlist_size is not None else top_k)),
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
        stage_metadata["shortlist_count"] = len(selection_state.shortlist)
        stage_metadata["bayesian_record_count"] = len(selection_state.bayesian_records)

    with profile_stage(
        runtime_profiler,
        "dag_control_chain",
        {"shortlist_count": len(selection_state.shortlist)},
    ) as stage_metadata:
        macro_control_blocked = bool(
            global_context.metadata.get("branch_fusion_blocked", False)
        )
        if macro_control_blocked:
            decision_state = _blocked_macro_control_state(
                global_context=global_context,
                tradability_snapshot=tradability_snapshot,
            )
            stage_metadata["status"] = "blocked_by_canonical_macro_readiness"
            stage_metadata["decision_authorized"] = False
        else:
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
        counterfactual_decision_state = None
        counterfactual_ready, counterfactual_variants = _counterfactual_replay_ready(
            selection_state
        )
        if counterfactual_ready and not macro_control_blocked:
            (
                counterfactual_research_by_symbol,
                counterfactual_branch_summaries,
            ) = _build_counterfactual_control_inputs(
                research_by_symbol=research_by_symbol,
                counterfactual_by_symbol=selection_state.counterfactual_by_symbol,
            )
            counterfactual_master_meta = {
                "status": "disabled_for_deterministic_counterfactual_replay",
                "reason": "actual_advisory_inputs_are_not_reused",
                "confidence": 0.0,
            }
            counterfactual_decision_state = _run_portfolio_construction_phase(
                shortlist=selection_state.counterfactual_shortlist,
                branch_summaries=counterfactual_branch_summaries,
                macro_verdict=counterfactual_branch_summaries["macro"],
                global_context=global_context,
                data_quality_issues=data_quality_issues,
                ic_hints_by_symbol={},
                research_by_symbol=counterfactual_research_by_symbol,
                tradability_snapshot=tradability_snapshot,
                funnel_summary=selection_state.funnel_summary,
                bayesian_records=selection_state.counterfactual_bayesian_records,
                candidate_symbols=candidate_symbols,
                portfolio_master_output=None,
                portfolio_master_meta=counterfactual_master_meta,
                risk_guard_cls=RiskGuard,
                ic_coordinator_cls=ICCoordinator,
                portfolio_constructor_cls=PortfolioConstructor,
                attach_symbol_to_ic_decision_fn=_attach_symbol_to_ic_decision,
            )
            decision_state.portfolio_decision.metadata[
                "fundamental_research_counterfactual_replay"
            ] = {
                "schema_version": "fundamental-control-chain-replay.v1",
                "measurement_only": True,
                "variant": next(iter(counterfactual_variants)),
                "branch_summaries": {
                    name: counterfactual_branch_summaries[name].to_dict()
                    for name in CANONICAL_BRANCH_ORDER
                },
                "branch_verdicts_by_symbol": {
                    symbol: {
                        name: branch_map[name].to_dict()
                        for name in CANONICAL_BRANCH_ORDER
                    }
                    for symbol, branch_map in counterfactual_research_by_symbol.items()
                },
                "bayesian_records": [
                    record.to_dict()
                    for record in selection_state.counterfactual_bayesian_records
                ],
                "shortlist": [
                    item.to_dict() for item in selection_state.counterfactual_shortlist
                ],
                "ic_hints_by_symbol": {},
                "risk_decision": counterfactual_decision_state.risk_decision.to_dict(),
                "ic_decisions": [
                    decision.to_dict()
                    for decision in counterfactual_decision_state.ic_decisions
                ],
                "portfolio_plan": counterfactual_decision_state.portfolio_plan.to_dict(),
                "portfolio_decision": counterfactual_decision_state.portfolio_decision.to_dict(),
            }
        elif selection_state.counterfactual_by_symbol:
            stage_metadata["counterfactual_replay_blocker"] = (
                "mixed_or_missing_fundamental_research_variant"
            )
        stage_metadata["ic_decision_count"] = len(decision_state.ic_decisions)
        stage_metadata["target_weight_count"] = len(getattr(decision_state.portfolio_decision, "target_weights", {}) or {})
        stage_metadata["counterfactual_target_weight_count"] = len(
            getattr(
                getattr(counterfactual_decision_state, "portfolio_decision", None),
                "target_weights",
                {},
            )
            or {}
        )

    with profile_stage(
        runtime_profiler,
        "dag_reporting_artifacts",
        {
            "candidate_count": len(candidate_symbols),
            "shortlist_count": len(selection_state.shortlist),
            "researchable_count": len(researchable_symbols),
        },
    ) as stage_metadata:
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
        stage_metadata["artifact_keys"] = sorted(str(key) for key in reporting_state.dag_artifacts.keys())
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
    shortlist_size: int | None = None,
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
    runtime_profiler: Any | None = None,
    decision_protocol: str = "v15",
    v16_factor_readiness_path: str = DEFAULT_FACTOR_READINESS_PATH,
    v16_review_root: str = DEFAULT_CODEX_REVIEW_ROOT,
    v16_config_path: str = str(DEFAULT_CONFIG_PATH),
    v16_prompt_path: str = str(DEFAULT_STAGE1_PROMPT_PATH),
    v16_run_id: str = "",
    v16_expiry_hours: float = 24.0,
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
            shortlist_size=shortlist_size,
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
            runtime_profiler=runtime_profiler,
            decision_protocol=decision_protocol,
            v16_factor_readiness_path=v16_factor_readiness_path,
            v16_review_root=v16_review_root,
            v16_config_path=v16_config_path,
            v16_prompt_path=v16_prompt_path,
            v16_run_id=v16_run_id,
            v16_expiry_hours=v16_expiry_hours,
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
