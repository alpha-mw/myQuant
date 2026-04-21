from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable, Mapping

import pandas as pd

from quant_investor.agent_protocol import ActionLabel, BranchVerdict, Direction, MasterICHint, StockReviewBundle, SymbolResearchPacket
from quant_investor.agents.llm_client import LLMClient as GatewayLLMClient
from quant_investor.agents.stock_reviewers import (
    BranchOverlayPacket,
    BranchOverlayReviewer,
    MasterICAgent,
    MasterSymbolPacket,
)
from quant_investor.branch_contracts import BranchResult
from quant_investor.market.dag.assembly import _aggregate_branch_summaries, _build_branch_results
from quant_investor.market.dag.common import _dedupe_texts, _score_to_action
from quant_investor.market.dag.packets import _build_symbol_bundle, _build_symbol_research_packet
from quant_investor.model_roles import ModelRoleResolution


@dataclass
class CandidateResearchState:
    symbol_research_packets: dict[str, SymbolResearchPacket] = field(default_factory=dict)
    research_by_symbol: dict[str, dict[str, BranchVerdict]] = field(default_factory=dict)
    review_bundle: StockReviewBundle = field(default_factory=StockReviewBundle)
    ic_hints_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    branch_summaries: dict[str, BranchVerdict] = field(default_factory=dict)
    branch_results: dict[str, BranchResult] = field(default_factory=dict)


def _score_to_direction(score: float) -> str:
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"


async def _run_candidate_research_phase(
    *,
    candidate_symbols: list[str],
    company_name_map: Mapping[str, str],
    market: str,
    market_snapshot: Mapping[str, Any],
    universe_key: str,
    read_results: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    global_quant_verdict: BranchVerdict,
    macro_verdict: BranchVerdict,
    branch_model_resolution: ModelRoleResolution,
    master_model_resolution: ModelRoleResolution,
    branch_candidate_models: list[str],
    master_candidate_models: list[str],
    master_reasoning_effort: str,
    enable_agent_layer: bool,
    agent_timeout: float,
    master_timeout: float,
    resolver_snapshot: Mapping[str, Any],
    kline_agent: Any,
    fundamental_agent: Any,
    intelligence_agent: Any,
    quant_result: BranchResult,
    full_market_kline_result: BranchResult,
    ensure_branch_verdict: Callable[..., BranchVerdict],
    master_hint_to_ic_hint: Callable[[Any], dict[str, Any]],
) -> CandidateResearchState:
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
            market=market,
            market_snapshot=market_snapshot,
        )
        branch_payload = {
            "data_bundle": bundle,
            "stock_pool": [symbol],
            "market": market,
            "verbose": False,
        }
        kline = ensure_branch_verdict(
            kline_agent.run({**branch_payload, "mode": "shortlist"}),
            symbol=symbol,
            branch_name="kline",
        )
        fundamental = ensure_branch_verdict(
            fundamental_agent.run({**branch_payload, "enable_document_semantics": True}),
            symbol=symbol,
            branch_name="fundamental",
        )
        intelligence = ensure_branch_verdict(
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
                market=market,
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
                candidate_models=list(branch_candidate_models),
                fallback_model=branch_model_resolution.fallback_model,
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
            candidate_models=list(master_candidate_models),
            fallback_model=master_model_resolution.fallback_model,
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
            market=market,
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
                ic_hints_by_symbol={symbol: master_hint_to_ic_hint(master_hint)},
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
        return symbol, reviewed_branch_verdicts, packet, master_hint, master_hint_to_ic_hint(master_hint), dict(branch_overlay_verdicts), telemetry, fallback_reasons

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
            "resolver": dict(resolver_snapshot),
            "global_quant_summary": dict(global_quant_verdict.to_dict()),
            "candidate_symbols": list(candidate_symbols),
        },
    )
    ic_hints_by_symbol: dict[str, dict[str, Any]] = {}
    fallback_reasons: list[str] = []
    telemetry_items: list[Any] = []
    for item in research_results:
        if isinstance(item, Exception):
            raise item
        symbol, reviewed_branch_verdicts, packet, master_hint, ic_hint, branch_overlays, telemetry, fallbacks = item
        research_by_symbol[symbol] = reviewed_branch_verdicts
        symbol_research_packets[symbol] = packet
        review_bundle.branch_overlay_verdicts_by_symbol[symbol] = dict(branch_overlays)
        if master_hint is not None:
            review_bundle.master_hints_by_symbol[symbol] = master_hint
        if ic_hint is not None:
            ic_hints_by_symbol[symbol] = ic_hint
            review_bundle.ic_hints_by_symbol[symbol] = dict(ic_hint)
        else:
            review_bundle.ic_hints_by_symbol[symbol] = {}
        telemetry_items.extend(list(telemetry))
        fallback_reasons.extend(list(fallbacks))

    review_bundle.telemetry = telemetry_items
    review_bundle.fallback_reasons = _dedupe_texts(fallback_reasons)

    branch_summaries = _aggregate_branch_summaries(research_by_symbol)
    branch_summaries["quant"] = global_quant_verdict
    branch_summaries["macro"] = macro_verdict
    branch_results = _build_branch_results(research_by_symbol, branch_summaries)
    branch_results["quant"] = quant_result
    branch_results["kline_funnel"] = full_market_kline_result

    return CandidateResearchState(
        symbol_research_packets=symbol_research_packets,
        research_by_symbol=research_by_symbol,
        review_bundle=review_bundle,
        ic_hints_by_symbol=ic_hints_by_symbol,
        branch_summaries=branch_summaries,
        branch_results=branch_results,
    )
