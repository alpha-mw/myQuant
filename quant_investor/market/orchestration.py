from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    ExecutionTrace,
    GlobalContext,
    ModelRoleMetadata,
    PortfolioDecision,
    ShortlistItem,
    SymbolResearchPacket,
    WhatIfPlan,
)
from quant_investor.reporting.run_artifacts import (
    build_execution_trace,
    build_model_role_metadata,
    build_what_if_plan,
)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _hash_symbols(symbols: list[str]) -> str:
    joined = ",".join(sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()}))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _aggregate_branch_summary(all_results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for batches in all_results.values():
        for batch in batches:
            for name, branch in batch.get("branches", {}).items():
                bucket = aggregated.setdefault(
                    name,
                    {
                        "score_values": [],
                        "confidence_values": [],
                        "conclusions": [],
                        "investment_risks": [],
                        "coverage_notes": [],
                        "diagnostic_notes": [],
                    },
                )
                bucket["score_values"].append(float(branch.get("score", 0.0)))
                bucket["confidence_values"].append(float(branch.get("confidence", 0.0)))
                bucket["conclusions"].append(str(branch.get("conclusion", "")))
                bucket["investment_risks"].extend(str(item) for item in branch.get("investment_risks", []))
                bucket["coverage_notes"].extend(str(item) for item in branch.get("coverage_notes", []))
                bucket["diagnostic_notes"].extend(str(item) for item in branch.get("diagnostic_notes", []))
    return aggregated


def _average(values: list[float], default: float = 0.0) -> float:
    filtered = [float(value) for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else default


def _build_branch_verdict(branch_name: str, branch_payload: Mapping[str, Any], symbol: str) -> BranchVerdict:
    return BranchVerdict(
        agent_name=branch_name,
        thesis=str(branch_payload.get("conclusion", "")) or f"{branch_name} 分支完成结构化判断。",
        symbol=symbol,
        final_score=float(branch_payload.get("score", 0.0)),
        final_confidence=float(branch_payload.get("confidence", 0.0)),
        investment_risks=[str(item) for item in branch_payload.get("investment_risks", [])],
        coverage_notes=[str(item) for item in branch_payload.get("coverage_notes", [])],
        diagnostic_notes=[str(item) for item in branch_payload.get("diagnostic_notes", [])],
        metadata={
            "branch_name": branch_name,
            "support_drivers": list(branch_payload.get("support_drivers", [])),
            "drag_drivers": list(branch_payload.get("drag_drivers", [])),
        },
    )


def _action_from_text(text: str) -> ActionLabel:
    normalized = str(text or "").strip().lower()
    mapping = {
        "buy": ActionLabel.BUY,
        "买入": ActionLabel.BUY,
        "light_buy": ActionLabel.BUY,
        "轻仓试错": ActionLabel.BUY,
        "hold": ActionLabel.HOLD,
        "持有": ActionLabel.HOLD,
        "watch": ActionLabel.WATCH,
        "观察": ActionLabel.WATCH,
        "sell": ActionLabel.SELL,
        "减仓": ActionLabel.SELL,
        "清仓": ActionLabel.SELL,
        "avoid": ActionLabel.AVOID,
        "规避": ActionLabel.AVOID,
    }
    return mapping.get(normalized, ActionLabel.HOLD)


def _build_symbol_packets(all_results: dict[str, list[dict[str, Any]]]) -> dict[str, SymbolResearchPacket]:
    packets: dict[str, SymbolResearchPacket] = {}
    for category, batches in all_results.items():
        for batch in batches:
            for recommendation in batch.get("recommendations", []):
                symbol = str(recommendation.get("symbol", "")).strip()
                if not symbol:
                    continue
                packet = packets.setdefault(
                    symbol,
                    SymbolResearchPacket(
                        symbol=symbol,
                        market=str(batch.get("market", "")),
                        category=str(category),
                        universe_key="full_a",
                    ),
                )
                packet.category = packet.category or str(category)
                packet.metadata.setdefault("batch_ids", []).append(batch.get("batch_id"))
                packet.metadata.setdefault("recommendation_actions", []).append(recommendation.get("action"))
                packet.metadata.setdefault("category_name", recommendation.get("category_name", ""))
                packet.metadata.setdefault("confidence", recommendation.get("confidence", 0.0))
                packet.branch_scores.update(
                    {name: float(branch.get("score", 0.0)) for name, branch in batch.get("branches", {}).items()}
                )
                packet.branch_confidences.update(
                    {name: float(branch.get("confidence", 0.0)) for name, branch in batch.get("branches", {}).items()}
                )
                packet.branch_theses.update(
                    {name: str(branch.get("conclusion", "")) for name, branch in batch.get("branches", {}).items()}
                )
                packet.risk_flags.extend(str(item) for item in recommendation.get("risk_flags", []))
                packet.coverage_notes.extend(
                    str(item)
                    for branch in batch.get("branches", {}).values()
                    for item in branch.get("coverage_notes", [])
                )
                packet.diagnostic_notes.extend(
                    str(item)
                    for branch in batch.get("branches", {}).values()
                    for item in branch.get("diagnostic_notes", [])
                )
                packet.branch_verdicts.update(
                    {
                        name: _build_branch_verdict(name, branch, symbol)
                        for name, branch in batch.get("branches", {}).items()
                    }
                )
    return packets


def _build_shortlist(all_results: dict[str, list[dict[str, Any]]], market: str) -> list[ShortlistItem]:
    shortlist: list[ShortlistItem] = []
    seen: set[str] = set()
    for category, batches in all_results.items():
        for batch in batches:
            for recommendation in batch.get("recommendations", []):
                symbol = str(recommendation.get("symbol", "")).strip()
                if not symbol or symbol in seen:
                    continue
                seen.add(symbol)
                shortlist.append(
                    ShortlistItem(
                        symbol=symbol,
                        category=str(category),
                        rank_score=float(recommendation.get("rank_score", 0.0)),
                        action=_action_from_text(str(recommendation.get("action", "hold"))),
                        confidence=float(recommendation.get("confidence", 0.0)),
                        expected_upside=float(recommendation.get("expected_upside", 0.0)),
                        suggested_weight=float(recommendation.get("suggested_weight", 0.0)),
                        risk_flags=[str(item) for item in recommendation.get("risk_flags", [])],
                        rationale=[
                            str(item)
                            for item in (
                                recommendation.get("support_drivers", [])
                                + recommendation.get("drag_drivers", [])
                                + recommendation.get("weight_cap_reasons", [])
                            )
                        ],
                        metadata={
                            "market": market,
                            "category_name": recommendation.get("category_name", category),
                            "one_line_conclusion": recommendation.get("one_line_conclusion", ""),
                            "current_price": recommendation.get("current_price", 0.0),
                            "recommended_entry_price": recommendation.get("recommended_entry_price", 0.0),
                        },
                    )
                )
    shortlist.sort(key=lambda item: (-float(item.rank_score), item.symbol))
    return shortlist


def _build_global_context(
    *,
    market: str,
    universe: str,
    all_results: dict[str, list[dict[str, Any]]],
    analysis_meta: Mapping[str, Any],
    portfolio_plan: Mapping[str, Any],
    download_stage: Mapping[str, Any] | None,
) -> GlobalContext:
    symbols = list(dict.fromkeys(str(symbol) for symbol in analysis_meta.get("symbols", []) if str(symbol).strip()))
    latest_trade_date = ""
    if download_stage:
        completeness = _as_mapping(download_stage.get("completeness_after")) or _as_mapping(download_stage.get("completeness_before"))
        latest_trade_date = str(completeness.get("latest_trade_date", ""))
    branch_summary = _aggregate_branch_summary(all_results)
    macro_score = _average(branch_summary.get("macro", {}).get("score_values", []))
    risk_summary = _as_mapping(portfolio_plan.get("risk_summary"))
    return GlobalContext(
        market=market,
        universe_key=universe or "full_a",
        rebalance_date=latest_trade_date,
        latest_trade_date=latest_trade_date,
        universe_symbols=symbols,
        universe_hash=_hash_symbols(symbols),
        industry_map={},
        liquidity_filter={
            "candidate_count": int(analysis_meta.get("total_stocks", len(symbols))),
            "category_count": int(analysis_meta.get("category_count", len(all_results))),
        },
        macro_regime=str(analysis_meta.get("execution_trace", {}).get("key_parameters", {}).get("market_regime", "")) or "neutral",
        cross_section_quant={
            "batch_count": int(analysis_meta.get("batch_count", 0)),
            "category_count": int(analysis_meta.get("category_count", 0)),
            "total_stocks": int(analysis_meta.get("total_stocks", len(symbols))),
            "macro_score": macro_score,
        },
        style_exposures={
            "style_bias": portfolio_plan.get("style_bias", "均衡"),
            "category_exposure": dict(portfolio_plan.get("category_exposure", {})),
        },
        correlation_matrix={},
        risk_budget={
            "target_exposure": float(portfolio_plan.get("target_exposure", 0.0)),
            "max_single_weight": float(portfolio_plan.get("max_single_weight", 0.0)),
            "risk_summary": risk_summary,
        },
        metadata={
            "branch_model": analysis_meta.get("branch_model", ""),
            "master_model": analysis_meta.get("master_model", ""),
            "master_reasoning_effort": analysis_meta.get("master_reasoning_effort", ""),
            "selected_count": int(portfolio_plan.get("selected_count", 0)),
        },
    )


def build_market_dag_artifacts(
    *,
    market: str,
    universe: str,
    all_results: dict[str, list[dict[str, Any]]],
    analysis_meta: Mapping[str, Any],
    total_capital: float,
    top_k: int,
    download_stage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from quant_investor.market.analyze import build_full_market_trade_plan

    analysis_model_role_meta = _as_mapping(analysis_meta.get("model_role_metadata", {}))
    plan = build_full_market_trade_plan(
        all_results,
        market=market,
        total_capital=total_capital,
        top_k=top_k,
    )
    portfolio_plan = _as_mapping(plan.get("portfolio_plan"))
    shortlist = _build_shortlist(all_results, market=market)
    symbol_packets = _build_symbol_packets(all_results)
    global_context = _build_global_context(
        market=market,
        universe=universe,
        all_results=all_results,
        analysis_meta=analysis_meta,
        portfolio_plan=portfolio_plan,
        download_stage=download_stage,
    )
    model_roles = build_model_role_metadata(
        branch_model=str(analysis_meta.get("branch_model", "")),
        master_model=str(analysis_meta.get("master_model", "")),
        agent_fallback_model=str(analysis_model_role_meta.get("agent_fallback_model", analysis_meta.get("agent_fallback_model", ""))),
        master_fallback_model=str(analysis_model_role_meta.get("master_fallback_model", analysis_meta.get("master_fallback_model", ""))),
        resolved_branch_model=str(analysis_model_role_meta.get("resolved_branch_model", analysis_meta.get("branch_model", ""))),
        resolved_master_model=str(analysis_model_role_meta.get("resolved_master_model", analysis_meta.get("master_model", ""))),
        master_reasoning_effort=str(analysis_meta.get("master_reasoning_effort", "high")),
        branch_provider=str(analysis_model_role_meta.get("branch_provider", "")),
        master_provider=str(analysis_model_role_meta.get("master_provider", "")),
        branch_timeout=float(analysis_meta.get("agent_timeout", 0.0)),
        master_timeout=float(analysis_meta.get("master_timeout", 0.0)),
        agent_layer_enabled=bool(analysis_meta.get("agent_layer_enabled", False)),
        branch_fallback_used=bool(analysis_model_role_meta.get("branch_fallback_used", False)),
        master_fallback_used=bool(analysis_model_role_meta.get("master_fallback_used", False)),
        branch_fallback_reason=str(analysis_model_role_meta.get("branch_fallback_reason", "")),
        master_fallback_reason=str(analysis_model_role_meta.get("master_fallback_reason", "")),
        universe_key=str(analysis_meta.get("universe", universe)),
        universe_size=len(symbol_packets),
        universe_hash=str(analysis_model_role_meta.get("universe_hash", "")),
        metadata=dict(analysis_model_role_meta),
    )
    what_if_plan = build_what_if_plan(
        portfolio_plan=portfolio_plan,
        market_summary={
            "candidate_count": len(symbol_packets),
            "macro_score": float(portfolio_plan.get("risk_summary", {}).get("macro_score", 0.0)),
        },
        model_roles=model_roles,
        candidate_count=len(symbol_packets),
        selected_count=int(portfolio_plan.get("selected_count", 0)),
    )
    execution_trace = build_execution_trace(
        model_roles=model_roles,
        download_stage=download_stage,
        analysis_meta=analysis_meta,
        portfolio_plan={
            "selected_count": int(portfolio_plan.get("selected_count", 0)),
            "target_exposure": float(portfolio_plan.get("target_exposure", 0.0)),
            "max_single_weight": float(portfolio_plan.get("max_single_weight", 0.0)),
            "risk_veto": bool(portfolio_plan.get("risk_summary", {}).get("hard_veto", False)),
            "action_cap": ActionLabel.BUY.value,
            "risk_summary": portfolio_plan.get("risk_summary", {}),
            "execution_notes": portfolio_plan.get("execution_notes", []),
        },
    )
    portfolio_decision = PortfolioDecision(
        shortlist=list(shortlist),
        target_exposure=float(portfolio_plan.get("target_exposure", 0.0)),
        target_gross_exposure=float(portfolio_plan.get("target_exposure", 0.0)),
        target_net_exposure=float(portfolio_plan.get("target_exposure", 0.0)),
        cash_ratio=float(portfolio_plan.get("cash_reserve", 0.0) / max(total_capital, 1.0)),
        target_weights={item.symbol: float(item.suggested_weight) for item in shortlist},
        target_positions={item.symbol: float(item.suggested_weight) for item in shortlist},
        risk_constraints={
            "max_single_weight": float(portfolio_plan.get("max_single_weight", 0.0)),
            "risk_summary": portfolio_plan.get("risk_summary", {}),
        },
        master_hints={},
        what_if_plan=what_if_plan,
        execution_trace=execution_trace,
        metadata={
            "market": market,
            "universe": universe,
            "selected_count": int(portfolio_plan.get("selected_count", 0)),
        },
    )
    return {
        "global_context": global_context,
        "symbol_research_packets": symbol_packets,
        "shortlist": shortlist,
        "portfolio_decision": portfolio_decision,
        "model_role_metadata": model_roles,
        "what_if_plan": what_if_plan,
        "execution_trace": execution_trace,
    }
