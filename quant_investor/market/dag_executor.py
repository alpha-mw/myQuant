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
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import threading
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import pandas as pd

from quant_investor.agent_protocol import (
    ActionLabel,
    BayesianDecisionRecord,
    BranchVerdict,
    DataQualityIssue,
    ExecutionTrace,
    GlobalContext,
    ICDecision,
    MasterICHint,
    ModelRoleMetadata,
    PortfolioDecision,
    PortfolioPlan,
    RiskDecision,
    Direction,
    ShortlistItem,
    StockReviewBundle,
    SymbolResearchPacket,
    WhatIfPlan,
)
from quant_investor.agents.agent_contracts import BaseBranchAgentOutput, MasterAgentInput
from quant_investor.agents.fundamental_agent import FundamentalAgent
from quant_investor.agents.ic_coordinator import ICCoordinator
from quant_investor.agents.intelligence_agent import IntelligenceAgent
from quant_investor.agents.kline_agent import KlineAgent
from quant_investor.agents.macro_agent import MacroAgent
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.narrator_agent import NarratorAgent
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.agents.risk_guard import RiskGuard
from quant_investor.agents.stock_reviewers import (
    BranchOverlayPacket,
    BranchOverlayReviewer,
    MasterICAgent,
    MasterSymbolPacket,
    _action_to_text,
)
from quant_investor.agents.llm_client import LLMClient as GatewayLLMClient
from quant_investor.bayesian.likelihood import SignalLikelihoodMapper
from quant_investor.bayesian.posterior import BayesianPosteriorEngine
from quant_investor.bayesian.prior import HierarchicalPriorBuilder
from quant_investor.config import config
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.funnel.deterministic_funnel import DeterministicFunnel, FunnelConfig, FunnelOutput
from quant_investor.market.cn_resolver import CNUniverseResolver
from quant_investor.market.config import get_market_settings, normalize_categories, normalize_universe
from quant_investor.market.data_snapshot import build_market_data_snapshot
from quant_investor.market.data_quality import build_data_quality_diagnostics
from quant_investor.market.provider_health import detect_provider_health
from quant_investor.market.shared_csv_reader import SharedCSVReader, SharedCSVReadResult
from quant_investor.model_roles import ModelRoleResolution, resolve_model_role
from quant_investor.llm_gateway import detect_provider
from quant_investor.llm_gateway import estimate_message_tokens
from quant_investor.reporting.conclusion_renderer import ConclusionRenderer
from quant_investor.reporting.run_artifacts import (
    build_bayesian_trace,
    build_funnel_summary,
    build_execution_trace,
    build_model_role_metadata,
    build_what_if_plan,
)
from quant_investor.data.universe.cn_universe import LOCAL_UNIVERSE_DIR


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


MASTER_EVIDENCE_PACK_TOKEN_LIMIT = 8_000
MASTER_EVIDENCE_PACK_SHORTLIST_LIMIT = 8
MASTER_EVIDENCE_PACK_FIELD_LIMIT = 7


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


def _is_quarantined_read_result(read_result: Any) -> bool:
    issues = list(getattr(read_result, "issues", []) or [])
    return bool(issues)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()  # type: ignore[call-arg]
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            return {}
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            return {}
    if is_dataclass(value):
        try:
            dumped = asdict(value)
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            return {}
    return {}


def _compact_trace_fragments(
    *,
    model_roles: ModelRoleMetadata | Mapping[str, Any] | Any,
    resolver_snapshot: Mapping[str, Any],
    data_quality_issues: list[DataQualityIssue],
    shortlist: list[ShortlistItem],
) -> dict[str, Any]:
    role_payload = _as_mapping(model_roles)
    return {
        "model_roles": {
            "branch_model": str(role_payload.get("branch_model", "")),
            "master_model": str(role_payload.get("master_model", "")),
            "branch_fallback_used": bool(role_payload.get("branch_fallback_used", False)),
            "master_fallback_used": bool(role_payload.get("master_fallback_used", False)),
            "master_reasoning_effort": str(role_payload.get("master_reasoning_effort", "")),
        },
        "resolver": {
            "resolution_strategy": str(resolver_snapshot.get("resolution_strategy", "")),
            "directory_priority": list(resolver_snapshot.get("directory_priority", []) or []),
            "physical_directories_used_for_full_a": list(
                resolver_snapshot.get("physical_directories_used_for_full_a", []) or []
            )[:4],
        },
        "data_quality_issue_count": len(data_quality_issues),
        "quarantined_count": len(data_quality_issues),
        "selected_count": len(shortlist),
    }


def _build_master_evidence_pack(
    *,
    shortlist: list[ShortlistItem],
    branch_summaries: Mapping[str, BranchVerdict],
    macro_verdict: BranchVerdict,
    risk_constraints: Mapping[str, Any],
    model_roles: Mapping[str, Any],
    resolver_snapshot: Mapping[str, Any],
    data_quality_issues: list[DataQualityIssue],
    company_name_map: Mapping[str, str],
    top_k: int,
) -> dict[str, Any]:
    shortlist_limit = max(1, min(int(top_k or 1), MASTER_EVIDENCE_PACK_SHORTLIST_LIMIT))
    shortlisted_items = shortlist[:shortlist_limit]

    def _shortlist_entry(item: ShortlistItem) -> dict[str, Any]:
        posterior_meta = dict(item.metadata or {})
        return {
            "symbol": item.symbol,
            "company_name": company_name_map.get(item.symbol, ""),
            "category": item.category,
            "rank_score": round(float(item.rank_score), 4),
            "action": item.action.value if hasattr(item.action, "value") else str(item.action),
            "confidence": round(float(item.confidence), 4),
            "expected_upside": round(float(item.expected_upside), 4),
            "suggested_weight": round(float(item.suggested_weight), 4),
            "posterior_action_score": round(float(posterior_meta.get("posterior_action_score", item.rank_score)), 4),
            "posterior_win_rate": round(float(posterior_meta.get("posterior_win_rate", 0.0)), 4),
            "posterior_confidence": round(float(posterior_meta.get("posterior_confidence", item.confidence)), 4),
            "posterior_edge_after_costs": round(float(posterior_meta.get("posterior_edge_after_costs", 0.0)), 4),
            "posterior_capacity_penalty": round(float(posterior_meta.get("posterior_capacity_penalty", 0.0)), 4),
            "rationale": list(item.rationale[:3]),
            "risk_flags": list(item.risk_flags[:3]),
        }

    def _branch_entry(name: str, verdict: BranchVerdict) -> dict[str, Any]:
        return {
            "branch": name,
            "score": round(float(verdict.final_score), 4),
            "confidence": round(float(verdict.final_confidence), 4),
            "thesis": str(verdict.thesis)[:240],
            "top_risks": list(verdict.investment_risks[:3]),
            "coverage_notes": list(verdict.coverage_notes[:2]),
        }

    key_risks = _dedupe_texts(
        _dedupe_texts([str(item) for item in risk_constraints.get("risk_notes", [])])[:4]
        + [issue.message for issue in data_quality_issues[:6]]
        + [item for verdict in branch_summaries.values() for item in verdict.investment_risks[:1]]
    )[:8]

    evidence_pack: dict[str, Any] = {
        "version": "evidence_pack.v1",
        "shortlist": [_shortlist_entry(item) for item in shortlisted_items],
        "branch_consensus": {
            name: _branch_entry(name, verdict)
            for name, verdict in branch_summaries.items()
        },
        "key_risks": key_risks,
        "portfolio_constraints": {
            "gross_exposure_cap": float(risk_constraints.get("gross_exposure_cap", 0.0) or 0.0),
            "target_exposure_cap": float(risk_constraints.get("target_exposure_cap", 0.0) or 0.0),
            "max_weight": float(risk_constraints.get("max_weight", 0.0) or 0.0),
            "action_cap": str(risk_constraints.get("action_cap", "hold")),
            "blocked_symbols": list(risk_constraints.get("blocked_symbols", [])[:10]),
            "selected_count": len(shortlisted_items),
        },
        "macro_summary": {
            "symbol": str(macro_verdict.symbol or ""),
            "score": round(float(macro_verdict.final_score), 4),
            "confidence": round(float(macro_verdict.final_confidence), 4),
            "thesis": str(macro_verdict.thesis)[:240],
            "regime": str(macro_verdict.metadata.get("regime", "neutral")),
        },
        "trace_fragments": _compact_trace_fragments(
            model_roles=model_roles,
            resolver_snapshot=resolver_snapshot,
            data_quality_issues=data_quality_issues,
            shortlist=shortlisted_items,
        ),
    }

    evidence_pack["trace_fragments"]["budget"] = {
        "field_limit": MASTER_EVIDENCE_PACK_FIELD_LIMIT,
        "token_limit": MASTER_EVIDENCE_PACK_TOKEN_LIMIT,
        "field_count": len(evidence_pack),
        "shortlist_count": len(shortlisted_items),
        "company_name_coverage": sum(1 for item in shortlisted_items if company_name_map.get(item.symbol, "")),
    }

    def _token_count(payload: dict[str, Any]) -> int:
        return estimate_message_tokens(
            [
                {"role": "system", "content": "evidence-pack"},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                },
            ]
        )

    token_count = _token_count(evidence_pack)
    if token_count <= MASTER_EVIDENCE_PACK_TOKEN_LIMIT:
        evidence_pack["trace_fragments"]["budget"]["token_count"] = token_count
        evidence_pack["trace_fragments"]["budget"]["truncated"] = False
        return evidence_pack

    trimmed = dict(evidence_pack)
    trimmed_shortlist = [_shortlist_entry(item) for item in shortlisted_items[: max(1, shortlist_limit // 2)]]
    trimmed["shortlist"] = trimmed_shortlist
    trimmed["branch_consensus"] = {
        name: {
            "branch": name,
            "score": round(float(verdict.final_score), 4),
            "confidence": round(float(verdict.final_confidence), 4),
            "thesis": str(verdict.thesis)[:120],
        }
        for name, verdict in branch_summaries.items()
    }
    trimmed["key_risks"] = key_risks[:5]
    trimmed["trace_fragments"] = {
        "model_roles": evidence_pack["trace_fragments"]["model_roles"],
        "resolver": evidence_pack["trace_fragments"]["resolver"],
        "data_quality_issue_count": len(data_quality_issues),
        "selected_count": len(trimmed_shortlist),
        "budget": {
            "field_limit": MASTER_EVIDENCE_PACK_FIELD_LIMIT,
            "token_limit": MASTER_EVIDENCE_PACK_TOKEN_LIMIT,
            "field_count": len(trimmed),
            "shortlist_count": len(trimmed_shortlist),
            "company_name_coverage": sum(1 for item in trimmed_shortlist if item.get("company_name")),
        },
    }
    token_count = _token_count(trimmed)
    trimmed["trace_fragments"]["budget"]["token_count"] = token_count
    trimmed["trace_fragments"]["budget"]["truncated"] = True
    return trimmed


def _score_to_direction(score: float) -> str:
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"


def _score_to_action(score: float) -> ActionLabel:
    if score >= 0.25:
        return ActionLabel.BUY
    if score <= -0.35:
        return ActionLabel.SELL
    return ActionLabel.HOLD


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


def _run_async_coroutine_safely(coro_factory: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # pragma: no cover - defensive thread boundary
            error_box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_box:
        raise error_box["error"]
    return result_box.get("value")


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


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "rows": 0,
            "latest_close": 0.0,
            "average_return": 0.0,
            "volatility": 0.0,
        }
    working = frame.copy()
    close_col = "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
    if not close_col:
        return {
            "rows": int(len(working)),
            "latest_close": 0.0,
            "average_return": 0.0,
            "volatility": 0.0,
        }
    close = pd.to_numeric(working[close_col], errors="coerce").dropna()
    average_return = 0.0
    volatility = 0.0
    if len(close) >= 2:
        returns = close.pct_change().dropna()
        average_return = float(returns.tail(20).mean()) if not returns.empty else 0.0
        volatility = float(returns.tail(60).std()) if len(returns) >= 3 else 0.0
    latest_close = float(close.iloc[-1]) if not close.empty else 0.0
    return {
        "rows": int(len(working)),
        "latest_close": latest_close,
        "average_return": average_return,
        "volatility": volatility,
    }


def _build_market_snapshot(
    *,
    market: str,
    universe_key: str,
    frames: dict[str, pd.DataFrame],
    global_summary: dict[str, Any],
    latest_trade_date: str,
    macro_overview: dict[str, Any],
) -> dict[str, Any]:
    closes = [summary["latest_close"] for summary in (_frame_summary(frame) for frame in frames.values()) if summary["latest_close"] > 0]
    frame_summaries = [_frame_summary(frame) for frame in frames.values() if not frame.empty]
    avg_return = fmean([summary["average_return"] for summary in frame_summaries]) if frame_summaries else 0.0
    volatility = fmean([summary["volatility"] for summary in frame_summaries]) if frame_summaries else 0.0
    breadth = 0.0
    if frames:
        positive = sum(1 for summary in frame_summaries if summary["average_return"] > 0)
        breadth = positive / max(len(frames), 1)
    return {
        "market": market,
        "universe_key": universe_key,
        "regime": macro_overview.get("regime", "neutral"),
        "policy_signal": macro_overview.get("policy_signal", "neutral"),
        "macro_score": float(macro_overview.get("macro_score", 0.0)),
        "liquidity_score": float(macro_overview.get("liquidity_score", 0.0)),
        "volatility_percentile": float(macro_overview.get("volatility_percentile", 50.0)),
        "candidate_count": int(global_summary.get("candidate_count", len(frames))),
        "symbol_count": int(len(frames)),
        "average_return": float(avg_return),
        "average_volatility": float(volatility),
        "breadth": float(breadth),
        "latest_trade_date": latest_trade_date,
        "latest_price": max(closes) if closes else 0.0,
    }


def _build_global_quant_verdict(
    *,
    cross_section_quant: Mapping[str, Any],
    symbol_count: int,
) -> BranchVerdict:
    average_return = float(cross_section_quant.get("average_return", 0.0))
    average_volatility = float(cross_section_quant.get("average_volatility", 0.0))
    breadth = float(cross_section_quant.get("breadth", 0.0))
    candidate_count = int(cross_section_quant.get("candidate_count", symbol_count))
    sample_count = int(cross_section_quant.get("sample_count", candidate_count))
    score = _clamp(average_return * 8.0 + (breadth - 0.5) * 0.6 - average_volatility * 0.4, -1.0, 1.0)
    confidence = _clamp(0.35 + min(sample_count, max(symbol_count, 1)) / max(symbol_count, 1) * 0.12, 0.0, 1.0)
    thesis = (
        "横截面量化结果已在全局上下文中一次性计算并收敛。"
        if score >= 0
        else "横截面量化结果显示全局环境偏谨慎，需降低预期。"
    )
    return BranchVerdict(
        agent_name="quant",
        thesis=thesis,
        symbol=None,
        final_score=score,
        final_confidence=confidence,
        investment_risks=[
            f"candidate_count={candidate_count}",
            f"sample_count={sample_count}",
            f"breadth={breadth:.3f}",
        ],
        coverage_notes=[
            "cross-sectional quant computed once in GlobalContext",
            f"average_return={average_return:+.4f}",
            f"average_volatility={average_volatility:.4f}",
        ],
        diagnostic_notes=[
            "global quant summary derived from shared context",
        ],
        metadata={
            "branch_name": "quant",
            "global_context_only": True,
            "candidate_count": candidate_count,
            "sample_count": sample_count,
            "average_return": average_return,
            "average_volatility": average_volatility,
            "breadth": breadth,
        },
    )


def _build_quant_branch_result(
    *,
    frames: Mapping[str, pd.DataFrame],
) -> BranchResult:
    symbol_scores: dict[str, float] = {}
    for symbol, frame in frames.items():
        summary = _frame_summary(frame)
        score = summary["average_return"] * 8.0 - summary["volatility"] * 2.0
        symbol_scores[symbol] = _clamp(score, -1.0, 1.0)
    return BranchResult(
        branch_name="quant",
        final_score=float(fmean(symbol_scores.values()) if symbol_scores else 0.0),
        final_confidence=_clamp(0.35 + min(len(symbol_scores), 50) / 120.0, 0.0, 1.0),
        symbol_scores=symbol_scores,
        conclusion="横截面量化分支已基于 shared context 与价格代理完成全市场压缩评分。",
        signals={
            "branch_mode": "cross_section_funnel",
            "alpha_factors": ["short_term_return", "volatility_penalty"],
        },
        investment_risks=["量化压缩当前未引入更重因子库。"],
        coverage_notes=[f"symbols={len(symbol_scores)}", "full_market_deterministic_funnel"],
        diagnostic_notes=["global_quant_branch_result"],
        metadata={"reliability": 0.70},
    )


def _build_symbol_quant_verdict(
    *,
    symbol: str,
    quant_result: BranchResult,
) -> BranchVerdict:
    score = float(quant_result.symbol_scores.get(symbol, quant_result.final_score))
    return BranchVerdict(
        agent_name="quant",
        thesis="量化分支当前基于收益/波动率横截面代理给出 deterministic 结论。",
        symbol=symbol,
        final_score=score,
        final_confidence=float(quant_result.final_confidence),
        investment_risks=list(quant_result.investment_risks),
        coverage_notes=list(quant_result.coverage_notes),
        diagnostic_notes=list(quant_result.diagnostic_notes),
        metadata={"branch_name": "quant", **dict(quant_result.metadata or {})},
    )


def _build_cross_section_quant(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    if not frames:
        return {
            "candidate_count": 0,
            "sample_count": 0,
            "average_return": 0.0,
            "average_volatility": 0.0,
            "breadth": 0.0,
        }
    summaries = [_frame_summary(frame) for frame in frames.values() if frame is not None and not frame.empty]
    if not summaries:
        return {
            "candidate_count": len(frames),
            "sample_count": 0,
            "average_return": 0.0,
            "average_volatility": 0.0,
            "breadth": 0.0,
        }
    positive = sum(1 for summary in summaries if summary["average_return"] > 0)
    return {
        "candidate_count": len(frames),
        "sample_count": len(summaries),
        "average_return": round(fmean(summary["average_return"] for summary in summaries), 6),
        "average_volatility": round(fmean(summary["volatility"] for summary in summaries), 6),
        "breadth": round(positive / max(len(summaries), 1), 6),
    }


def _build_symbol_tradability(symbol: str, read_result: SharedCSVReadResult, *, company_name: str = "") -> dict[str, Any]:
    frame = read_result.frame
    return {
        "symbol": symbol,
        "company_name": company_name,
        "tradable": bool(frame is not None and not frame.empty),
        "sector": "unknown",
        "source_path": read_result.path,
        "resolver_strategy": read_result.resolver_trace.get("resolution_strategy", ""),
        "data_quality_issue_count": len(read_result.issues),
    }


def _build_symbol_research_packet(
    *,
    symbol: str,
    company_name: str,
    market: str,
    universe_key: str,
    category: str,
    branch_verdicts: dict[str, BranchVerdict],
    read_result: SharedCSVReadResult,
    macro_verdict: BranchVerdict,
    global_quant_verdict: BranchVerdict,
    review_bundle: StockReviewBundle | None,
) -> SymbolResearchPacket:
    frame_summary = _frame_summary(read_result.frame)
    packet = SymbolResearchPacket(
        symbol=symbol,
        company_name=company_name,
        market=market,
        category=category,
        universe_key=universe_key,
        branch_verdicts=dict(branch_verdicts),
        branch_scores={name: float(verdict.final_score) for name, verdict in branch_verdicts.items()},
        branch_confidences={name: float(verdict.final_confidence) for name, verdict in branch_verdicts.items()},
        branch_theses={name: str(verdict.thesis) for name, verdict in branch_verdicts.items()},
        risk_flags=_dedupe_texts(
            [item for verdict in branch_verdicts.values() for item in verdict.investment_risks]
            + [issue.message for issue in read_result.issues]
        ),
        coverage_notes=_dedupe_texts(
            [item for verdict in branch_verdicts.values() for item in verdict.coverage_notes]
        ),
        diagnostic_notes=_dedupe_texts(
            [item for verdict in branch_verdicts.values() for item in verdict.diagnostic_notes]
        ),
        metadata={
            "company_name": company_name,
            "resolved_path": read_result.path,
            "resolver_trace": dict(read_result.resolver_trace),
            "macro_regime": macro_verdict.metadata.get("regime", "neutral"),
            "macro_score": float(macro_verdict.final_score),
            "global_quant_summary": global_quant_verdict.to_dict(),
            "latest_close": float(frame_summary.get("latest_close", 0.0)),
            "price_summary": frame_summary,
            "data_quality_issues": [issue.to_dict() for issue in read_result.issues],
            "review_fallback_reasons": list(review_bundle.fallback_reasons if review_bundle else []),
        },
    )
    return packet


def _build_symbol_bundle(
    *,
    symbol: str,
    frame: pd.DataFrame,
    read_result: SharedCSVReadResult,
    market: str,
    market_snapshot: Mapping[str, Any],
) -> UnifiedDataBundle:
    return UnifiedDataBundle(
        market=market,
        symbols=[symbol],
        symbol_data={symbol: frame},
        fundamentals={},
        event_data={},
        sentiment_data={},
        macro_data=dict(market_snapshot),
        metadata={
            "symbol_provenance": {
                symbol: {
                    "path": read_result.path,
                    "resolver_trace": read_result.resolver_trace,
                    "data_quality_issues": [issue.to_dict() for issue in read_result.issues],
                }
            },
        },
    )


def _aggregate_branch_summaries(
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
) -> dict[str, BranchVerdict]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scores": [],
            "confidences": [],
            "theses": [],
            "risks": [],
            "coverage": [],
            "diagnostic": [],
            "symbols": [],
        }
    )
    for symbol, branch_map in research_by_symbol.items():
        for branch_name, verdict in branch_map.items():
            bucket = buckets[branch_name]
            bucket["scores"].append(float(verdict.final_score))
            bucket["confidences"].append(float(verdict.final_confidence))
            bucket["theses"].append(str(verdict.thesis))
            bucket["risks"].extend(verdict.investment_risks)
            bucket["coverage"].extend(verdict.coverage_notes)
            bucket["diagnostic"].extend(verdict.diagnostic_notes)
            bucket["symbols"].append(symbol)
    result: dict[str, BranchVerdict] = {}
    for branch_name, bucket in buckets.items():
        avg_score = fmean(bucket["scores"]) if bucket["scores"] else 0.0
        avg_conf = fmean(bucket["confidences"]) if bucket["confidences"] else 0.0
        result[branch_name] = BranchVerdict(
            agent_name=branch_name,
            thesis=next((text for text in bucket["theses"] if text.strip()), f"{branch_name} 分支已完成全市场汇总。"),
            symbol=None,
            final_score=float(avg_score),
            final_confidence=float(avg_conf),
            investment_risks=_dedupe_texts(list(bucket["risks"]))[:8],
            coverage_notes=_dedupe_texts(list(bucket["coverage"]))[:8],
            diagnostic_notes=_dedupe_texts(list(bucket["diagnostic"]))[:8],
            metadata={
                "branch_name": branch_name,
                "symbol_count": len(bucket["symbols"]),
                "symbols": list(dict.fromkeys(bucket["symbols"]))[:15],
            },
        )
    return result


def _build_branch_results(
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
    branch_summaries: Mapping[str, BranchVerdict],
) -> dict[str, BranchResult]:
    branch_scores_by_symbol: dict[str, dict[str, float]] = defaultdict(dict)
    for symbol, branch_map in research_by_symbol.items():
        for branch_name, verdict in branch_map.items():
            branch_scores_by_symbol[branch_name][symbol] = float(verdict.final_score)

    results: dict[str, BranchResult] = {}
    for branch_name, verdict in branch_summaries.items():
        results[branch_name] = BranchResult(
            branch_name=branch_name,
            final_score=float(verdict.final_score),
            final_confidence=float(verdict.final_confidence),
            symbol_scores=dict(branch_scores_by_symbol.get(branch_name, {})),
            conclusion=str(verdict.thesis),
            investment_risks=list(verdict.investment_risks),
            coverage_notes=list(verdict.coverage_notes),
            diagnostic_notes=list(verdict.diagnostic_notes),
            metadata=dict(verdict.metadata),
        )
    return results


def _build_shortlist(
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
    ic_hints_by_symbol: Mapping[str, Mapping[str, Any]],
    macro_verdict: BranchVerdict,
    company_name_map: Mapping[str, str],
    top_k: int,
    allowed_symbols: set[str] | None = None,
    blocked_symbols: set[str] | None = None,
) -> list[ShortlistItem]:
    shortlist: list[ShortlistItem] = []
    for symbol, branch_map in research_by_symbol.items():
        if allowed_symbols is not None and symbol not in allowed_symbols:
            continue
        if blocked_symbols is not None and symbol in blocked_symbols:
            continue
        scores = [float(verdict.final_score) for verdict in branch_map.values()]
        confidences = [float(verdict.final_confidence) for verdict in branch_map.values()]
        avg_score = fmean(scores) if scores else 0.0
        avg_conf = fmean(confidences) if confidences else 0.0
        hint = dict(ic_hints_by_symbol.get(symbol, {}))
        hint_score = float(hint.get("score", avg_score))
        hint_conf = float(hint.get("confidence", avg_conf))
        rank_score = 0.55 * avg_score + 0.25 * hint_score + 0.20 * float(macro_verdict.final_score)
        action = _score_to_action(rank_score)
        rationale = _dedupe_texts(
            [verdict.thesis for verdict in branch_map.values()] + [str(hint.get("thesis", ""))]
        )
        shortlist.append(
            ShortlistItem(
                symbol=symbol,
                company_name=company_name_map.get(symbol, ""),
                category=str(next(iter(branch_map.values())).metadata.get("category", "")) if branch_map else "",
                rank_score=float(rank_score),
                action=action,
                confidence=float(max(avg_conf, hint_conf)),
                expected_upside=max(float(rank_score), 0.0),
                suggested_weight=max(float(hint.get("confidence", avg_conf)), 0.0) * 0.1,
                risk_flags=_dedupe_texts([item for verdict in branch_map.values() for item in verdict.investment_risks]),
                rationale=rationale[:5],
                metadata={
                    "branch_scores": {name: float(verdict.final_score) for name, verdict in branch_map.items()},
                    "macro_score": float(macro_verdict.final_score),
                    "ic_hint": hint,
                },
            )
        )
    shortlist.sort(key=lambda item: (-float(item.rank_score), item.symbol))
    return shortlist[:top_k]


def _build_shortlist_from_bayesian_records(
    *,
    posterior_results: list[Any],
    company_name_map: Mapping[str, str],
    top_k: int,
) -> list[ShortlistItem]:
    ranked = sorted(
        posterior_results,
        key=lambda item: (
            -float(getattr(item, "posterior_action_score", 0.0) if not isinstance(item, dict) else item.get("posterior_action_score", 0.0)),
            str(getattr(item, "symbol", "") if not isinstance(item, dict) else item.get("symbol", "")),
        ),
    )
    shortlist: list[ShortlistItem] = []
    for record in ranked[:top_k]:
        symbol = str(getattr(record, "symbol", "") if not isinstance(record, dict) else record.get("symbol", ""))
        company_name = (
            str(getattr(record, "company_name", "") if not isinstance(record, dict) else record.get("company_name", ""))
            or company_name_map.get(symbol, "")
        )
        action_score = float(
            getattr(record, "posterior_action_score", 0.0)
            if not isinstance(record, dict)
            else record.get("posterior_action_score", 0.0)
        )
        win_rate = float(
            getattr(record, "posterior_win_rate", 0.0)
            if not isinstance(record, dict)
            else record.get("posterior_win_rate", 0.0)
        )
        confidence = float(
            getattr(record, "posterior_confidence", 0.0)
            if not isinstance(record, dict)
            else record.get("posterior_confidence", 0.0)
        )
        expected_alpha = float(
            getattr(record, "posterior_expected_alpha", 0.0)
            if not isinstance(record, dict)
            else record.get("posterior_expected_alpha", 0.0)
        )
        metadata = (
            dict(getattr(record, "metadata", {}) or {})
            if not isinstance(record, dict)
            else dict(record.get("metadata", {}) or {})
        )
        shortlist.append(
            ShortlistItem(
                symbol=symbol,
                company_name=company_name,
                category=str(metadata.get("category", "")),
                rank_score=action_score,
                action=_score_to_action(action_score),
                confidence=confidence,
                expected_upside=max(expected_alpha, 0.0),
                suggested_weight=max(0.0, min(0.2, action_score * confidence * 0.2)),
                rationale=[
                    f"posterior_action_score={action_score:.3f}",
                    f"posterior_win_rate={win_rate:.3f}",
                ],
                metadata={
                    **metadata,
                    "posterior_action_score": action_score,
                    "posterior_win_rate": win_rate,
                    "posterior_confidence": confidence,
                    "posterior_expected_alpha": expected_alpha,
                    "posterior_edge_after_costs": float(metadata.get("posterior_edge_after_costs", 0.0)),
                    "posterior_capacity_penalty": float(metadata.get("posterior_capacity_penalty", 0.0)),
                    "rank": int(getattr(record, "rank", 0) if not isinstance(record, dict) else record.get("rank", 0)),
                },
            )
        )
    return shortlist


def _branch_verdicts_to_master_reports(
    branch_summaries: Mapping[str, BranchVerdict],
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
) -> dict[str, BaseBranchAgentOutput]:
    reports: dict[str, BaseBranchAgentOutput] = {}
    for branch_name, summary in branch_summaries.items():
        symbol_views: dict[str, str] = {}
        for symbol, branches in research_by_symbol.items():
            verdict = branches.get(branch_name)
            if verdict is None:
                continue
            symbol_views[symbol] = verdict.thesis
        action = _score_to_action(float(summary.final_score))
        reports[branch_name] = BaseBranchAgentOutput(
            branch_name=branch_name,
            conviction=_branch_conviction_from_action(action),
            conviction_score=float(summary.final_score),
            confidence=float(summary.final_confidence),
            key_insights=list(summary.coverage_notes[:4]) or [summary.thesis],
            risk_flags=list(summary.investment_risks[:4]),
            disagreements_with_algo=list(summary.diagnostic_notes[:4]),
            symbol_views=symbol_views,
            reasoning=str(summary.thesis),
        )
    return reports


def _attach_symbol_to_ic_decision(
    ic_decision: ICDecision,
    *,
    symbol: str,
    risk_decision: RiskDecision,
    current_weight: float,
    tradability_info: Mapping[str, Any],
    ic_hint: Mapping[str, Any] | None = None,
) -> ICDecision:
    payload = ic_decision
    payload.selected_symbols = [symbol] if symbol not in risk_decision.blocked_symbols and payload.action in {ActionLabel.BUY, ActionLabel.HOLD} else []
    payload.rejected_symbols = [] if payload.selected_symbols else [symbol]
    payload.metadata = dict(payload.metadata)
    payload.metadata.update(
        {
            "symbol": symbol,
            "company_name": str(tradability_info.get("company_name", "")),
            "risk_action_cap": risk_decision.action_cap.value,
            "current_weight": current_weight,
            "sector": str(tradability_info.get("sector", "unknown")),
            "symbol_candidates": [
                {
                    "symbol": symbol,
                    "company_name": str(tradability_info.get("company_name", "")),
                    "score": float(payload.final_score),
                    "confidence": float(payload.final_confidence),
                    "action": payload.action.value,
                    "current_weight": current_weight,
                    "one_line_conclusion": payload.thesis,
                }
            ],
        }
    )
    if ic_hint:
        payload.metadata["llm_master_hint"] = dict(ic_hint)
    return payload


def _portfolio_master_advisory(
    *,
    master_agent: MasterAgent,
    macro_verdict: BranchVerdict,
    shortlist: list[ShortlistItem],
    global_context: GlobalContext,
    evidence_pack: dict[str, Any],
    recall_context: Mapping[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    advice_meta: dict[str, Any] = {}
    try:
        agent_input = MasterAgentInput(
            evidence_pack=dict(evidence_pack),
            branch_reports={},
            risk_report=None,
            ensemble_baseline=dict(
                evidence_pack.get(
                    "portfolio_constraints",
                    {
                        "aggregate_score": float(fmean([item.rank_score for item in shortlist]) if shortlist else 0.0),
                        "selected_count": len(shortlist),
                    },
                )
            ),
            market_regime=str(global_context.macro_regime or macro_verdict.metadata.get("regime", "neutral")),
            candidate_symbols=[item.symbol for item in shortlist],
            recall_context=dict(recall_context or {}),
        )
        output = _run_async_coroutine_safely(lambda: master_agent.deliberate(agent_input))
        advice_meta = {
            "status": "success",
            "reason": "",
            "final_conviction": output.final_conviction,
            "final_score": float(output.final_score),
            "confidence": float(output.confidence),
            "top_picks": [item.model_dump() if hasattr(item, "model_dump") else asdict(item) for item in output.top_picks],
            "portfolio_narrative": output.portfolio_narrative,
            "risk_adjusted_exposure": float(output.risk_adjusted_exposure),
            "evidence_pack_token_count": int(
                evidence_pack.get("trace_fragments", {}).get("budget", {}).get("token_count", 0) or 0
            ),
        }
        return output, advice_meta
    except Exception as exc:
        advice_meta = {
            "status": "fallback",
            "reason": str(exc),
            "final_conviction": "neutral",
            "final_score": 0.0,
            "confidence": 0.5,
            "top_picks": [],
            "portfolio_narrative": "MasterAgent fallback advisory.",
            "risk_adjusted_exposure": float(global_context.risk_budget.get("target_exposure", 0.0)),
            "evidence_pack_token_count": int(
                evidence_pack.get("trace_fragments", {}).get("budget", {}).get("token_count", 0) or 0
            ),
        }
        return None, advice_meta


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
    agent_model: str = "",
    agent_fallback_model: str = "",
    master_model: str = "",
    master_fallback_model: str = "",
    master_reasoning_effort: str = "high",
    agent_timeout: float = 15.0,
    master_timeout: float = 30.0,
    agent_layer_enabled: bool = True,
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
    provider_health = detect_provider_health(
        agent_model=branch_model_resolution.primary_model,
        master_model=master_model_resolution.primary_model,
    )

    read_results: dict[str, SharedCSVReadResult] = {}
    frames: dict[str, pd.DataFrame] = {}
    tradability_snapshot: dict[str, dict[str, Any]] = {}
    data_quality_issues: list[DataQualityIssue] = []
    quarantined_symbols: list[str] = []
    researchable_symbols: list[str] = []
    for symbol in all_symbols:
        read_result = shared_reader.read_symbol_frame(symbol, universe_key=universe_key)
        read_results[symbol] = read_result
        frames[symbol] = read_result.frame
        tradability_snapshot[symbol] = _build_symbol_tradability(
            symbol,
            read_result,
            company_name=company_name_map.get(symbol, ""),
        )
        data_quality_issues.extend(read_result.issues)
        if _is_quarantined_read_result(read_result):
            quarantined_symbols.append(symbol)
        else:
            researchable_symbols.append(symbol)

    symbols = list(researchable_symbols)

    cross_section_quant = _build_cross_section_quant(frames)
    macro_overview = {
        "regime": "neutral",
        "macro_score": cross_section_quant.get("average_return", 0.0),
        "liquidity_score": cross_section_quant.get("breadth", 0.0),
        "volatility_percentile": min(95.0, max(5.0, cross_section_quant.get("average_volatility", 0.0) * 100.0 + 50.0)),
        "policy_signal": "neutral",
    }
    snapshot_latest_trade_date = str(scoped_data_snapshot.get("local_latest_trade_date", ""))
    snapshot_freshness_mode = str(scoped_data_snapshot.get("freshness_mode", "stable"))
    market_snapshot = _build_market_snapshot(
        market=settings.market,
        universe_key=universe_key,
        frames=frames,
        global_summary={"candidate_count": len(symbols)},
        latest_trade_date=(
            download_stage.get("completeness_after", {}).get("latest_trade_date", "")
            if download_stage
            else snapshot_latest_trade_date
        ),
        macro_overview=macro_overview,
    )

    macro_agent = MacroAgent()
    macro_verdict = macro_agent.run({"market_snapshot": market_snapshot})
    macro_overview["regime"] = str(macro_verdict.metadata.get("regime", "neutral"))
    macro_overview["macro_score"] = float(macro_verdict.final_score)
    macro_overview["liquidity_score"] = float(cross_section_quant.get("breadth", 0.0))
    market_snapshot.update(macro_overview)
    global_quant_verdict = _build_global_quant_verdict(
        cross_section_quant=cross_section_quant,
        symbol_count=len(symbols),
    )
    quant_result = _build_quant_branch_result(frames=frames)
    liquidity_scores = {
        symbol: min(1.0, max(0.0, _frame_summary(frame).get("rows", 0) / 250.0))
        for symbol, frame in frames.items()
    }
    illiquid_symbols = [symbol for symbol, score in liquidity_scores.items() if score < 0.10]

    completeness_payload = {}
    if download_stage:
        completeness_payload = dict(
            download_stage.get("completeness_after")
            or download_stage.get("completeness_before")
            or {}
        )
    effective_latest_trade_date = str(
        completeness_payload.get("latest_trade_date")
        or snapshot_latest_trade_date
    )
    effective_freshness_mode = str(
        completeness_payload.get("freshness_mode")
        or snapshot_freshness_mode
        or "stable"
    )
    global_context = GlobalContext(
        market=settings.market,
        universe_key=universe_key,
        rebalance_date=effective_latest_trade_date,
        latest_trade_date=effective_latest_trade_date,
        universe_symbols=list(all_symbols),
        universe_hash="",
        industry_map={},
        liquidity_filter={
            "candidate_count": len(all_symbols),
            "researchable_count": len(symbols),
            "quarantined_count": len(quarantined_symbols),
            "category_count": len(selected_categories),
            "suspended": list(quarantined_symbols),
            "illiquid": list(illiquid_symbols),
            "liquidity_scores": liquidity_scores,
        },
        macro_regime=str(macro_verdict.metadata.get("regime", "neutral")),
        cross_section_quant={**cross_section_quant, "macro_score": float(macro_verdict.final_score)},
        style_exposures={
            "style_bias": macro_verdict.metadata.get("style_bias", "balanced"),
        },
        correlation_matrix={},
        risk_budget={
            "target_exposure": float(macro_verdict.metadata.get("target_gross_exposure", 0.5)),
            "max_single_weight": 0.12,
        },
        data_quality_issues=data_quality_issues,
        data_quality_diagnostics=build_data_quality_diagnostics(
            total_symbols=all_symbols,
            researchable_symbols=researchable_symbols,
            shortlistable_symbols=[],
            final_selected_symbols=[],
            quarantined_symbols=quarantined_symbols,
            issues=data_quality_issues,
        ),
        model_capability_map=provider_health,
        symbol_name_map=company_name_map,
        data_quality_quarantine=list(quarantined_symbols),
        freshness_mode=effective_freshness_mode,
        effective_target_trade_date=str(
            completeness_payload.get("effective_target_trade_date")
            or effective_latest_trade_date
        ),
        universe_tiers={
            "total": list(all_symbols),
            "researchable": list(researchable_symbols),
            "shortlistable": [],
            "final_selected": [],
        },
        metadata={
            "resolver": shared_reader.snapshot(),
            "resolver_directory_priority": list((shared_reader.snapshot() or {}).get("directory_priority", [])),
            "physical_directories_used_for_full_a": list((shared_reader.snapshot() or {}).get("physical_directories_used_for_full_a", [])),
            "data_quality_issue_count": len(data_quality_issues),
            "candidate_count": len(all_symbols),
            "researchable_count": len(symbols),
            "quarantined_count": len(quarantined_symbols),
            "quarantined_symbols": list(quarantined_symbols[:32]),
            "global_quant_verdict": global_quant_verdict.to_dict(),
            "provider_health": provider_health,
            "data_snapshot": scoped_data_snapshot,
        },
    )
    global_context.universe_hash = ""
    if symbols:
        import hashlib

        global_context.universe_hash = hashlib.sha256(",".join(sorted(symbols)).encode("utf-8")).hexdigest()[:16]

    model_role_metadata = build_model_role_metadata(
        branch_model=branch_model_resolution.primary_model,
        master_model=master_model_resolution.primary_model,
        agent_fallback_model=branch_model_resolution.fallback_model,
        master_fallback_model=master_model_resolution.fallback_model,
        resolved_branch_model=branch_model_resolution.resolved_model,
        resolved_master_model=master_model_resolution.resolved_model,
        master_reasoning_effort=master_reasoning_effort,
        branch_provider=detect_provider(branch_model_resolution.resolved_model),
        master_provider=detect_provider(master_model_resolution.resolved_model),
        branch_timeout=agent_timeout,
        master_timeout=master_timeout,
        agent_layer_enabled=bool(enable_agent_layer),
        branch_fallback_used=bool(branch_model_resolution.fallback_used),
        master_fallback_used=bool(master_model_resolution.fallback_used),
        branch_fallback_reason=str(branch_model_resolution.fallback_reason),
        master_fallback_reason=str(master_model_resolution.fallback_reason),
        universe_key=universe_key,
        universe_size=len(symbols),
        universe_hash=global_context.universe_hash,
        metadata={
            "resolver": shared_reader.snapshot(),
            "data_quality_issue_count": len(data_quality_issues),
            "agent_layer_enabled": bool(agent_layer_enabled),
            "provider_health": provider_health,
        },
    )
    model_roles = model_role_metadata

    kline_agent = KlineAgent()
    fundamental_agent = FundamentalAgent()
    intelligence_agent = IntelligenceAgent()
    full_market_bundle = UnifiedDataBundle(
        market=settings.market,
        symbols=list(symbols),
        symbol_data={symbol: frames.get(symbol, pd.DataFrame()) for symbol in symbols},
        fundamentals={},
        event_data={},
        sentiment_data={},
        macro_data=dict(market_snapshot),
        metadata={
            "symbol_provenance": {
                symbol: {
                    "path": read_results[symbol].path,
                    "resolver_trace": read_results[symbol].resolver_trace,
                    "data_quality_issues": [issue.to_dict() for issue in read_results[symbol].issues],
                }
                for symbol in symbols
            }
        },
    )
    full_market_kline_verdict = _ensure_branch_verdict(
        kline_agent.run(
            {
                "data_bundle": full_market_bundle,
                "stock_pool": list(symbols),
                "market": settings.market,
                "verbose": False,
                "mode": "full_market",
            }
        ),
        symbol="__market__",
        branch_name="kline",
    )
    full_market_kline_result = _branch_verdict_to_result(full_market_kline_verdict, "kline")

    funnel = DeterministicFunnel(
        FunnelConfig(
            max_candidates=int(getattr(config, "FUNNEL_MAX_CANDIDATES", 400) or 400),
        )
    )
    funnel_output = funnel.run(
        quant_result=quant_result,
        kline_result=full_market_kline_result,
        global_context=global_context,
    )
    candidate_symbols = [symbol for symbol in funnel_output.candidates if symbol in researchable_symbols]
    if not candidate_symbols:
        candidate_symbols = list(researchable_symbols)
    global_context.universe_tiers = {
        "total": list(all_symbols),
        "researchable": list(researchable_symbols),
        "shortlistable": list(candidate_symbols),
        "final_selected": [],
    }
    global_context.data_quality_diagnostics = build_data_quality_diagnostics(
        total_symbols=all_symbols,
        researchable_symbols=researchable_symbols,
        shortlistable_symbols=candidate_symbols,
        final_selected_symbols=[],
        quarantined_symbols=quarantined_symbols,
        issues=data_quality_issues,
    )
    global_context.metadata["candidate_count"] = len(candidate_symbols)
    global_context.metadata["shortlistable_count"] = len(candidate_symbols)

    async def _research_symbol(
        symbol: str,
    ) -> tuple[
        str,
        dict[str, BranchVerdict],
        SymbolResearchPacket,
        MasterICHint | None,
        dict[str, Any],
        dict[str, Any],
        list[Any],
        list[str],
    ]:
        frame = frames.get(symbol, pd.DataFrame())
        read_result = read_results[symbol]
        bundle = _build_symbol_bundle(
            symbol=symbol,
            frame=frame,
            read_result=read_result,
            market=settings.market,
            market_snapshot=market_snapshot,
        )
        branch_payload = {
            "data_bundle": bundle,
            "stock_pool": [symbol],
            "market": settings.market,
            "verbose": False,
        }
        kline = _ensure_branch_verdict(
            kline_agent.run({**branch_payload, "mode": "shortlist"}),
            symbol=symbol,
            branch_name="kline",
        )
        fundamental = _ensure_branch_verdict(
            fundamental_agent.run({**branch_payload, "enable_document_semantics": True}),
            symbol=symbol,
            branch_name="fundamental",
        )
        intelligence = _ensure_branch_verdict(
            intelligence_agent.run({**branch_payload, "market_regime": macro_verdict.metadata.get("regime", "neutral")}),
            symbol=symbol,
            branch_name="intelligence",
        )
        base_branch_verdicts = {
            "kline": kline,
            "fundamental": fundamental,
            "intelligence": intelligence,
        }

        if not enable_agent_layer:
            packet = _build_symbol_research_packet(
                symbol=symbol,
                company_name=company_name_map.get(symbol, ""),
                market=settings.market,
                universe_key=universe_key,
                category=str(read_result.category or ""),
                branch_verdicts=base_branch_verdicts,
                read_result=read_result,
                macro_verdict=macro_verdict,
                global_quant_verdict=global_quant_verdict,
                review_bundle=None,
            )
            return symbol, base_branch_verdicts, packet, None, {}, {}, [], []

        review_llm = GatewayLLMClient(timeout=agent_timeout)
        review_master_llm = GatewayLLMClient(timeout=master_timeout)
        branch_names = list(base_branch_verdicts.keys())
        branch_overlay_verdicts: dict[str, Any] = {}
        telemetry: list[Any] = []
        fallback_reasons: list[str] = []
        for branch_name in branch_names:
            base_verdict = base_branch_verdicts[branch_name]
            packet = BranchOverlayPacket(
                symbol=symbol,
                branch_name=branch_name,
                base_score=float(base_verdict.final_score),
                base_confidence=float(base_verdict.final_confidence),
                thesis=str(base_verdict.thesis),
                direction=_score_to_direction(float(base_verdict.final_score)),
                action=_score_to_action(float(base_verdict.final_score)).value,
                agreement_points=_dedupe_texts(list(base_verdict.coverage_notes[:3]) or [base_verdict.thesis]),
                conflict_points=_dedupe_texts(list(base_verdict.diagnostic_notes[:3]) or list(base_verdict.investment_risks[:3])),
                risk_points=_dedupe_texts(list(base_verdict.investment_risks[:4])),
                branch_signals={"score": float(base_verdict.final_score), "confidence": float(base_verdict.final_confidence)},
                macro_summary=dict(market_snapshot),
                risk_summary={
                    "macro_score": float(macro_verdict.final_score),
                    "macro_regime": str(macro_verdict.metadata.get("regime", "neutral")),
                    "data_quality_issue_count": len(read_result.issues),
                },
                metadata={"source_branch": branch_name, "symbol": symbol, "resolver": read_result.resolver_trace},
            )
            reviewer = BranchOverlayReviewer(
                branch_name=branch_name,
                llm_client=review_llm,
                model=branch_model_resolution.resolved_model,
                timeout=agent_timeout,
                max_tokens=600,
            )
            overlay = await reviewer.review(packet)
            branch_overlay_verdicts[branch_name] = overlay
            telemetry.append(overlay.telemetry)
            if overlay.telemetry.fallback and overlay.telemetry.fallback_reason:
                fallback_reasons.append(f"{symbol}/{branch_name}: {overlay.telemetry.fallback_reason}")

        overlay_dicts = [overlay.to_dict() for overlay in branch_overlay_verdicts.values()]
        master_packet = MasterSymbolPacket(
            symbol=symbol,
            branch_overlay_summaries=overlay_dicts,
            macro_summary=dict(market_snapshot),
            risk_summary={
                "macro_score": float(macro_verdict.final_score),
                "macro_regime": str(macro_verdict.metadata.get("regime", "neutral")),
                "data_quality_issue_count": len(read_result.issues),
                "risk_flags": _dedupe_texts(
                    [issue.message for issue in read_result.issues[:2]]
                    + [item for item in base_branch_verdicts["kline"].investment_risks[:1]]
                ),
            },
            baseline_score=float(fmean([item["adjusted_score"] for item in overlay_dicts]) if overlay_dicts else 0.0),
            baseline_confidence=float(fmean([item["adjusted_confidence"] for item in overlay_dicts]) if overlay_dicts else 0.0),
            hard_veto=bool(False),
            metadata={"symbol": symbol, "resolver": read_result.resolver_trace},
        )
        master_reviewer = MasterICAgent(
            llm_client=review_master_llm,
            model=master_model_resolution.resolved_model,
            reasoning_effort=master_reasoning_effort,
            timeout=master_timeout,
            max_tokens=900,
        )
        master_hint = await master_reviewer.deliberate(master_packet)
        telemetry.append(master_hint.telemetry)
        if master_hint.telemetry.fallback and master_hint.telemetry.fallback_reason:
            fallback_reasons.append(f"{symbol}: {master_hint.telemetry.fallback_reason}")

        reviewed_branch_verdicts: dict[str, BranchVerdict] = {}
        for branch_name, base_verdict in base_branch_verdicts.items():
            overlay = branch_overlay_verdicts.get(branch_name)
            if overlay is None:
                reviewed_branch_verdicts[branch_name] = base_verdict
                continue
            reviewed_branch_verdicts[branch_name] = BranchVerdict(
                agent_name=base_verdict.agent_name,
                thesis=overlay.thesis or base_verdict.thesis,
                symbol=symbol,
                status=base_verdict.status,
                direction=overlay.direction if isinstance(overlay.direction, Direction) else base_verdict.direction,
                action=overlay.action if isinstance(overlay.action, ActionLabel) else base_verdict.action,
                confidence_label=base_verdict.confidence_label,
                final_score=float(overlay.adjusted_score),
                final_confidence=float(overlay.adjusted_confidence),
                investment_risks=_dedupe_texts(list(base_verdict.investment_risks) + list(overlay.risk_flags) + list(overlay.missing_risks)),
                coverage_notes=_dedupe_texts(list(base_verdict.coverage_notes) + list(overlay.agreement_points)),
                diagnostic_notes=_dedupe_texts(list(base_verdict.diagnostic_notes) + list(overlay.conflict_points) + list(overlay.contradictions)),
                metadata={
                    **dict(base_verdict.metadata or {}),
                    "branch_name": branch_name,
                    "overlay": overlay.to_dict(),
                },
            )
        packet = _build_symbol_research_packet(
            symbol=symbol,
            company_name=company_name_map.get(symbol, ""),
            market=settings.market,
            universe_key=universe_key,
            category=str(read_result.category or ""),
            branch_verdicts=reviewed_branch_verdicts,
            read_result=read_result,
            macro_verdict=macro_verdict,
            global_quant_verdict=global_quant_verdict,
            review_bundle=StockReviewBundle(
                agent_name="StockReviewOrchestrator",
                branch_overlay_verdicts_by_symbol={symbol: dict(branch_overlay_verdicts)},
                master_hints_by_symbol={symbol: master_hint},
                ic_hints_by_symbol={symbol: _master_hint_to_ic_hint(master_hint)},
                telemetry=telemetry,
                fallback_reasons=_dedupe_texts(fallback_reasons),
                metadata={
                    "branch_model": branch_model_resolution.resolved_model,
                    "master_model": master_model_resolution.resolved_model,
                    "branch_primary_model": branch_model_resolution.primary_model,
                    "branch_fallback_model": branch_model_resolution.fallback_model,
                    "master_primary_model": master_model_resolution.primary_model,
                    "master_fallback_model": master_model_resolution.fallback_model,
                    "branch_fallback_used": branch_model_resolution.fallback_used,
                    "master_fallback_used": master_model_resolution.fallback_used,
                    "branch_fallback_reason": branch_model_resolution.fallback_reason,
                    "master_fallback_reason": master_model_resolution.fallback_reason,
                    "master_reasoning_effort": master_reasoning_effort,
                    "agent_layer_enabled": bool(enable_agent_layer),
                    "universe_key": universe_key,
                    "symbol_count": len(candidate_symbols),
                    "resolver": read_result.resolver_trace,
                },
            ),
        )
        return symbol, reviewed_branch_verdicts, packet, master_hint, _master_hint_to_ic_hint(master_hint), dict(branch_overlay_verdicts), telemetry, fallback_reasons

    # Run per-symbol branch + review layer concurrently but bounded.
    semaphore = asyncio.Semaphore(8)

    async def _guarded(symbol: str):
        async with semaphore:
            return await _research_symbol(symbol)

    research_tasks = [_guarded(symbol) for symbol in candidate_symbols]
    research_results = await asyncio.gather(*research_tasks, return_exceptions=True)

    symbol_research_packets: dict[str, SymbolResearchPacket] = {}
    research_by_symbol: dict[str, dict[str, BranchVerdict]] = {}
    review_bundle = StockReviewBundle(
        agent_name="StockReviewOrchestrator",
        metadata={
            "branch_model": branch_model_resolution.resolved_model,
            "master_model": master_model_resolution.resolved_model,
            "branch_primary_model": branch_model_resolution.primary_model,
            "branch_fallback_model": branch_model_resolution.fallback_model,
            "master_primary_model": master_model_resolution.primary_model,
            "master_fallback_model": master_model_resolution.fallback_model,
            "branch_fallback_used": branch_model_resolution.fallback_used,
            "master_fallback_used": master_model_resolution.fallback_used,
            "branch_fallback_reason": branch_model_resolution.fallback_reason,
            "master_fallback_reason": master_model_resolution.fallback_reason,
            "master_reasoning_effort": master_reasoning_effort,
            "agent_layer_enabled": bool(enable_agent_layer),
            "universe_key": universe_key,
            "symbol_count": len(candidate_symbols),
            "resolver": shared_reader.snapshot(),
            "global_quant_summary": dict(global_quant_verdict.to_dict()),
            "candidate_symbols": list(candidate_symbols),
        },
    )
    ic_hints_by_symbol: dict[str, dict[str, Any]] = {}
    fallback_reasons: list[str] = []
    master_hints_by_symbol: dict[str, MasterICHint] = {}
    telemetry_items: list[Any] = []
    for item in research_results:
        if isinstance(item, Exception):
            raise item
        symbol, reviewed_branch_verdicts, packet, master_hint, ic_hint, branch_overlays, telemetry, fallbacks = item
        research_by_symbol[symbol] = reviewed_branch_verdicts
        symbol_research_packets[symbol] = packet
        review_bundle.branch_overlay_verdicts_by_symbol[symbol] = dict(branch_overlays)
        if master_hint is not None:
            master_hints_by_symbol[symbol] = master_hint
            review_bundle.master_hints_by_symbol[symbol] = master_hint
        if ic_hint is not None:
            ic_hints_by_symbol[symbol] = ic_hint
            review_bundle.ic_hints_by_symbol[symbol] = dict(ic_hint)
        else:
            review_bundle.ic_hints_by_symbol[symbol] = {}
        telemetry_items.extend(list(telemetry))
        fallback_reasons.extend(list(fallbacks))
    # The above per-symbol branch overlays were already embedded in the packet review bundle metadata;
    # for the persisted symbol review bundle we only need the hints and fallback reasons.
    review_bundle.telemetry = telemetry_items
    review_bundle.fallback_reasons = _dedupe_texts(fallback_reasons)

    branch_summaries = _aggregate_branch_summaries(research_by_symbol)
    branch_summaries["quant"] = global_quant_verdict
    branch_summaries["macro"] = macro_verdict
    branch_results = _build_branch_results(research_by_symbol, branch_summaries)
    branch_results["quant"] = quant_result
    branch_results["kline_funnel"] = full_market_kline_result

    prior_builder = HierarchicalPriorBuilder()
    likelihood_mapper = SignalLikelihoodMapper()
    posterior_engine = BayesianPosteriorEngine()
    bayesian_records: list[BayesianDecisionRecord] = []
    degraded_map = {
        "kline": provider_health.get("kline", {}).get("mode") != "hybrid",
        "fundamental": False,
        "intelligence": False,
        "quant": False,
    }
    for symbol in candidate_symbols:
        prior = prior_builder.build_prior(symbol, global_context)
        likelihoods = likelihood_mapper.compute_likelihoods(
            branch_results=branch_results,
            symbol=symbol,
            candidate_symbols=set(candidate_symbols),
        )
        posterior = posterior_engine.compute_posterior(
            prior,
            likelihoods,
            symbol=symbol,
            company_name=company_name_map.get(symbol, ""),
            regime=global_context.macro_regime or "未知",
            is_degraded=degraded_map,
        )
        bayesian_records.append(
            BayesianDecisionRecord(
                symbol=symbol,
                company_name=company_name_map.get(symbol, ""),
                prior=posterior.prior.to_dict(),
                likelihoods=posterior.likelihoods.to_dict(),
                posterior_win_rate=posterior.posterior_win_rate,
                posterior_expected_alpha=posterior.posterior_expected_alpha,
                posterior_confidence=posterior.posterior_confidence,
                posterior_action_score=posterior.posterior_action_score,
                posterior_edge_after_costs=posterior.posterior_edge_after_costs,
                posterior_capacity_penalty=posterior.posterior_capacity_penalty,
                correlation_discount=posterior.correlation_discount,
                coverage_discount=posterior.coverage_discount,
                data_quality_penalty=posterior.data_quality_penalty,
                fallback_penalty=posterior.fallback_penalty,
                regime_adjustment=posterior.regime_adjustment,
                evidence_sources=list(posterior.evidence_sources),
                action_threshold_used=posterior.action_threshold_used,
                metadata={
                    "category": str(symbol_research_packets[symbol].category),
                    "posterior_edge_after_costs": posterior.posterior_edge_after_costs,
                    "posterior_capacity_penalty": posterior.posterior_capacity_penalty,
                },
            )
        )
    bayesian_records.sort(key=lambda item: (-float(item.posterior_action_score), item.symbol))
    for index, record in enumerate(bayesian_records, start=1):
        record.rank = index
        record.metadata = dict(record.metadata or {})
        record.metadata["rank"] = index

    shortlist = _build_shortlist_from_bayesian_records(
        posterior_results=bayesian_records,
        company_name_map=company_name_map,
        top_k=top_k,
    )
    for item in shortlist:
        branch_map = research_by_symbol.get(item.symbol, {})
        item.risk_flags = _dedupe_texts([risk for verdict in branch_map.values() for risk in verdict.investment_risks])[:5]
        item.rationale = _dedupe_texts(
            list(item.rationale)
            + [verdict.thesis for verdict in branch_map.values()]
        )[:5]

    funnel_summary = build_funnel_summary(
        universe_size=len(all_symbols),
        candidates_count=len(candidate_symbols),
        shortlist_count=len(shortlist),
        final_selected_count=0,
        excluded_symbols=funnel_output.excluded_symbols,
        funnel_metadata=funnel_output.funnel_metadata,
    )
    evidence_pack = _build_master_evidence_pack(
        shortlist=shortlist,
        branch_summaries=branch_summaries,
        macro_verdict=macro_verdict,
        risk_constraints=global_context.risk_budget,
        model_roles=model_roles,
        resolver_snapshot=shared_reader.snapshot(),
        data_quality_issues=data_quality_issues,
        company_name_map=company_name_map,
        top_k=top_k,
    )

    portfolio_master_agent = MasterAgent(
        llm_client=GatewayLLMClient(timeout=master_timeout),
        model=master_model_resolution.resolved_model,
        reasoning_effort=master_reasoning_effort,
        timeout=master_timeout,
    )
    portfolio_master_output, portfolio_master_meta = _portfolio_master_advisory(
        master_agent=portfolio_master_agent,
        macro_verdict=macro_verdict,
        shortlist=shortlist,
        global_context=global_context,
        evidence_pack=evidence_pack,
        recall_context=recall_context,
    )
    portfolio_master_reliability = float(portfolio_master_meta.get("confidence", 0.0) or 0.0)

    risk_guard = RiskGuard()
    risk_decision = risk_guard.run(
        {
            "branch_verdicts": branch_summaries,
            "macro_verdict": macro_verdict,
            "portfolio_state": {
                "candidate_symbols": [item.symbol for item in shortlist],
                "current_weights": {},
            },
            "constraints": {
                "gross_exposure_cap": float(global_context.risk_budget.get("target_exposure", 0.55)),
                "max_weight": float(global_context.risk_budget.get("max_single_weight", 0.12)),
                "risk_flags": _dedupe_texts([issue.message for issue in data_quality_issues[:8]]),
                "data_quality_issue_count": len(data_quality_issues),
            },
        }
    )

    ic_coordinator = ICCoordinator()
    shortlisted_symbols = [item.symbol for item in shortlist]
    ic_decisions: list[ICDecision] = []
    for symbol in shortlisted_symbols:
        decision = ic_coordinator.run(
            {
                "branch_verdicts": research_by_symbol[symbol],
                "risk_decision": risk_decision,
                "ic_hints": ic_hints_by_symbol.get(symbol, {}),
            }
        )
        decision = _attach_symbol_to_ic_decision(
            decision,
            symbol=symbol,
            risk_decision=risk_decision,
            current_weight=0.0,
            tradability_info=tradability_snapshot[symbol],
            ic_hint=ic_hints_by_symbol.get(symbol, {}),
        )
        ic_decisions.append(decision)

    portfolio_constructor = PortfolioConstructor()
    portfolio_plan = portfolio_constructor.run(
        {
            "ic_decisions": ic_decisions,
            "macro_verdict": macro_verdict,
            "risk_limits": risk_decision,
            "existing_portfolio": {"current_weights": {}},
            "tradability_snapshot": tradability_snapshot,
        }
    )

    portfolio_decision = PortfolioDecision(
        status=portfolio_plan.status,
        shortlist=shortlist,
        target_exposure=float(portfolio_plan.target_exposure),
        target_gross_exposure=float(portfolio_plan.target_gross_exposure),
        target_net_exposure=float(portfolio_plan.target_net_exposure),
        cash_ratio=float(portfolio_plan.cash_ratio),
        target_weights=dict(portfolio_plan.target_weights),
        target_positions=dict(portfolio_plan.target_positions),
        risk_constraints={
            "risk_decision": risk_decision.to_dict(),
            "tradability_snapshot": tradability_snapshot,
        },
        master_hints={
            "portfolio_master_output": portfolio_master_output.model_dump() if portfolio_master_output is not None and hasattr(portfolio_master_output, "model_dump") else dict(portfolio_master_meta),
        },
        metadata={
            "portfolio_master_meta": portfolio_master_meta,
            "risk_summary": risk_decision.to_dict(),
            "branch_summary_count": len(branch_summaries),
            "funnel_summary": funnel_summary,
            "bayesian_record_count": len(bayesian_records),
            "bayesian_top_symbols": [record.symbol for record in bayesian_records[: min(len(bayesian_records), 10)]],
            "candidate_symbols": list(candidate_symbols),
            "shortlist_symbols": [item.symbol for item in shortlist],
        },
    )

    data_quality_summary = {
        "data_quality_issue_count": len(data_quality_issues),
        "quarantined_count": len(quarantined_symbols),
        "researchable_count": len(symbols),
        "shortlistable_count": len(candidate_symbols),
        "shortlist_count": len(shortlist),
        "quarantined_symbols": list(quarantined_symbols[:32]),
        "investment_risks": [issue.message for issue in data_quality_issues[:8]],
        "coverage_notes": [
            f"可研究覆盖 {len(symbols)} / 总覆盖 {len(all_symbols)}",
            f"漏斗候选 {len(candidate_symbols)} / 可研究覆盖 {len(symbols)}",
            f"Bayesian shortlist {len(shortlist)} / 漏斗候选 {len(candidate_symbols)}",
            f"resolver={shared_reader.snapshot().get('resolution_strategy', '')}",
        ],
        "diagnostic_notes": [
            f"{issue.symbol or 'unknown'}: {issue.message}"
            for issue in data_quality_issues[:8]
        ],
    }
    data_quality_diagnostics = build_data_quality_diagnostics(
        total_symbols=all_symbols,
        researchable_symbols=researchable_symbols,
        shortlistable_symbols=candidate_symbols,
        final_selected_symbols=list(portfolio_plan.target_weights),
        quarantined_symbols=quarantined_symbols,
        issues=data_quality_issues,
    )
    global_context.data_quality_diagnostics = data_quality_diagnostics
    global_context.universe_tiers = {
        "total": list(all_symbols),
        "researchable": list(researchable_symbols),
        "shortlistable": list(candidate_symbols),
        "final_selected": list(portfolio_plan.target_weights),
    }
    funnel_summary["final_selected_count"] = len(portfolio_plan.target_weights)
    funnel_summary["compression_ratio"] = (
        f"{len(all_symbols)} -> {len(candidate_symbols)} -> {len(shortlist)} -> {len(portfolio_plan.target_weights)}"
    )
    what_if_plan = build_what_if_plan(
        portfolio_plan=portfolio_plan,
        market_summary={
            "candidate_count": len(candidate_symbols),
            "selected_count": len(portfolio_plan.target_weights),
            "macro_score": float(macro_verdict.final_score),
        },
        model_roles=model_roles,
        candidate_count=len(candidate_symbols),
        selected_count=len(portfolio_plan.target_weights),
    )
    execution_trace = build_execution_trace(
        model_roles=model_roles,
        download_stage=download_stage,
        analysis_meta={
            "market": settings.market,
            "universe": universe_key,
            "batch_count": 1,
            "category_count": len(selected_categories),
            "total_stocks": len(all_symbols),
            "researchable_count": len(symbols),
            "quarantined_count": len(quarantined_symbols),
            "shortlistable_count": len(candidate_symbols),
            "shortlist_count": len(shortlist),
            "final_selected_count": len(portfolio_plan.target_weights),
            "quarantined_symbols": list(quarantined_symbols[:32]),
            "data_quality_issue_count": len(data_quality_issues),
            "fallback_reasons": list(review_bundle.fallback_reasons),
            "master_success": bool(portfolio_master_output is not None),
            "ic_hints_count": len(ic_hints_by_symbol),
            "company_name_coverage": sum(1 for item in shortlist if item.company_name),
            "evidence_pack_token_count": int(evidence_pack.get("trace_fragments", {}).get("budget", {}).get("token_count", 0) or 0),
            "evidence_pack_field_limit": MASTER_EVIDENCE_PACK_FIELD_LIMIT,
            "evidence_pack_shortlist_limit": MASTER_EVIDENCE_PACK_SHORTLIST_LIMIT,
            "candidate_count": len(candidate_symbols),
            "funnel_summary": funnel_summary,
            "bayesian_record_count": len(bayesian_records),
            "bayesian_trace": build_bayesian_trace(bayesian_records=bayesian_records),
            "resolver": shared_reader.snapshot(),
            "global_context": global_context.to_dict(),
            "provider_health": provider_health,
            "data_quality_diagnostics": data_quality_diagnostics.to_dict(),
        },
        portfolio_plan={
            "selected_count": len(portfolio_plan.target_weights),
            "target_exposure": float(portfolio_plan.target_exposure),
            "max_single_weight": float(portfolio_plan.position_limits.get(next(iter(portfolio_plan.position_limits), ""), 0.0)) if portfolio_plan.position_limits else 0.0,
            "risk_veto": bool(risk_decision.veto),
            "action_cap": risk_decision.action_cap.value,
            "risk_summary": risk_decision.to_dict(),
            "execution_notes": portfolio_plan.execution_notes,
            "target_weights": dict(portfolio_plan.target_weights),
            "reliability": portfolio_master_reliability,
            "style_bias": global_context.style_exposures.get("style_bias", "balanced"),
            "provider_health": provider_health,
        },
        persistence_note="结构化 DAG 结果已准备完毕。",
    )
    portfolio_decision.what_if_plan = what_if_plan
    portfolio_decision.execution_trace = execution_trace

    review_bundle.branch_summaries = branch_summaries
    review_bundle.risk_decision = risk_decision
    review_bundle.metadata.update(
        {
            "portfolio_master_output": portfolio_master_output.model_dump() if portfolio_master_output is not None and hasattr(portfolio_master_output, "model_dump") else portfolio_master_meta,
            "data_quality_summary": data_quality_summary,
        }
    )

    narrator_agent = NarratorAgent()
    report_bundle = narrator_agent.run(
        {
            "macro_verdict": macro_verdict,
            "branch_summaries": branch_summaries,
            "ic_decisions": ic_decisions,
            "portfolio_plan": portfolio_plan,
            "review_bundle": review_bundle,
            "ic_hints_by_symbol": ic_hints_by_symbol,
            "model_role_metadata": model_roles,
            "execution_trace": execution_trace,
            "what_if_plan": what_if_plan,
            "global_context": global_context,
            "symbol_research_packets": symbol_research_packets,
            "shortlist": shortlist,
            "portfolio_decision": portfolio_decision,
            "bayesian_records": bayesian_records,
            "funnel_summary": funnel_summary,
            "run_diagnostics": {
                **data_quality_summary,
                "coverage_notes": data_quality_summary["coverage_notes"]
                + [
                    f"physical_directories={len(shared_reader.snapshot().get('physical_directories_used_for_full_a', []))}",
                ],
                "diagnostic_notes": data_quality_summary["diagnostic_notes"]
                + [
                    f"{symbol}: {issue.message}"
                    for symbol, read_result in list(read_results.items())[:5]
                    for issue in read_result.issues[:1]
                ],
                "resolver": shared_reader.snapshot(),
                "global_context": global_context.to_dict(),
            },
        }
    )

    dag_artifacts = {
        "global_context": global_context,
        "symbol_research_packets": symbol_research_packets,
        "branch_verdicts_by_symbol": research_by_symbol,
        "branch_summaries": branch_summaries,
        "branch_results": branch_results,
        "funnel_output": funnel_output,
        "funnel_summary": funnel_summary,
        "bayesian_records": bayesian_records,
        "global_quant_verdict": global_quant_verdict,
        "macro_verdict": macro_verdict,
        "risk_decision": risk_decision,
        "ic_decisions": ic_decisions,
        "portfolio_plan": portfolio_plan,
        "portfolio_decision": portfolio_decision,
        "portfolio_master_output": portfolio_master_output,
        "portfolio_master_meta": portfolio_master_meta,
        "review_bundle": review_bundle,
        "model_role_metadata": model_roles,
        "what_if_plan": what_if_plan,
        "execution_trace": execution_trace,
        "tradability_snapshot": tradability_snapshot,
        "data_quality_issues": [issue.to_dict() for issue in data_quality_issues],
        "data_quality_summary": data_quality_summary,
        "data_quality_diagnostics": data_quality_diagnostics,
        "provider_health": provider_health,
        "resolver": shared_reader.snapshot(),
        "data_snapshot": scoped_data_snapshot,
        "report_bundle": report_bundle,
    }
    return dag_artifacts


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
    agent_model: str = "",
    agent_fallback_model: str = "",
    master_model: str = "",
    master_fallback_model: str = "",
    master_reasoning_effort: str = "high",
    agent_timeout: float = 15.0,
    master_timeout: float = 30.0,
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
            agent_model=agent_model,
            agent_fallback_model=agent_fallback_model,
            master_model=master_model,
            master_fallback_model=master_fallback_model,
            master_reasoning_effort=master_reasoning_effort,
            agent_timeout=agent_timeout,
            master_timeout=master_timeout,
            agent_layer_enabled=enable_agent_layer,
            recall_context=recall_context,
        )
    )
