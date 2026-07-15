from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable, Mapping

import pandas as pd

from quant_investor.agent_protocol import (
    AgentStatus,
    BranchOverlayVerdict,
    BranchVerdict,
    Direction,
    MasterICHint,
    ReviewTelemetry,
    StockReviewBundle,
    SymbolResearchPacket,
)
from quant_investor.agents.llm_client import LLMClient as GatewayLLMClient
from quant_investor.agents.stock_reviewers import (
    BranchOverlayPacket,
    BranchOverlayReviewer,
    MasterICAgent,
    MasterSymbolPacket,
)
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
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


def _direction_enum_from_score(score: float) -> Direction:
    direction = _score_to_direction(score)
    if direction == "bullish":
        return Direction.BULLISH
    if direction == "bearish":
        return Direction.BEARISH
    return Direction.NEUTRAL


def _is_codex_handoff_review(
    branch_model_resolution: ModelRoleResolution,
    master_model_resolution: ModelRoleResolution,
) -> bool:
    metadata = {
        **dict(branch_model_resolution.metadata or {}),
        **dict(master_model_resolution.metadata or {}),
    }
    return bool(
        metadata.get("review_layer_mode") == "codex_handoff"
        or metadata.get("codex_handoff_pending")
        or branch_model_resolution.resolved_model == "codex-handoff"
        or master_model_resolution.resolved_model == "codex-handoff"
    )


def _build_codex_handoff_overlay(
    packet: BranchOverlayPacket,
    *,
    model: str = "codex-handoff",
) -> BranchOverlayVerdict:
    """Create a neutral overlay record for Codex review."""

    telemetry = ReviewTelemetry(
        stage="review_branch_overlay",
        model=model or "codex-handoff",
        provider="codex",
        success=False,
        fallback=True,
        fallback_reason="codex_handoff_pending",
        score_delta=0.0,
        confidence_delta=0.0,
        metadata={
            "actor_name": f"{packet.symbol}:{packet.branch_name}",
            "codex_handoff_pending": True,
            "review_layer_mode": "codex_handoff",
        },
    )
    return BranchOverlayVerdict(
        symbol=packet.symbol,
        branch_name=packet.branch_name,
        status=AgentStatus.DEGRADED,
        thesis=(
            packet.thesis
            or f"{packet.symbol} {packet.branch_name} branch awaiting Codex review."
        ),
        direction=_direction_enum_from_score(float(packet.base_score)),
        action=_score_to_action(float(packet.base_score)),
        base_score=float(packet.base_score),
        adjusted_score=float(packet.base_score),
        base_confidence=float(packet.base_confidence),
        adjusted_confidence=float(packet.base_confidence),
        score_delta=0.0,
        confidence_delta=0.0,
        agreement_points=_dedupe_texts(list(packet.agreement_points)),
        conflict_points=[],
        missing_risks=[],
        contradictions=[],
        risk_flags=_dedupe_texts(list(packet.risk_points)),
        telemetry=telemetry,
        metadata={
            **dict(packet.metadata or {}),
            "model": model or "codex-handoff",
            "stage": "review_branch_overlay",
            "actor_name": f"{packet.symbol}:{packet.branch_name}",
            "branch_name": packet.branch_name,
            "symbol": packet.symbol,
            "review_layer_mode": "codex_handoff",
            "codex_handoff_pending": True,
            "codex_review_packet": packet.to_dict(),
        },
    )


def _build_codex_handoff_master(
    packet: MasterSymbolPacket,
    *,
    model: str = "codex-handoff",
) -> MasterICHint:
    """Create a neutral master hint for Codex review."""

    baseline_score = float(packet.baseline_score)
    baseline_confidence = float(packet.baseline_confidence)
    telemetry = ReviewTelemetry(
        stage="review_master_symbol",
        model=model or "codex-handoff",
        provider="codex",
        success=False,
        fallback=True,
        fallback_reason="codex_handoff_pending",
        score_delta=0.0,
        confidence_delta=0.0,
        metadata={
            "actor_name": f"IC:{packet.symbol}",
            "codex_handoff_pending": True,
            "review_layer_mode": "codex_handoff",
        },
    )
    return MasterICHint(
        symbol=packet.symbol,
        status=AgentStatus.DEGRADED,
        thesis=(
            f"{packet.symbol} per-symbol master review awaiting Codex handoff."
        ),
        action=_score_to_action(baseline_score),
        direction=_direction_enum_from_score(baseline_score),
        score_hint=baseline_score,
        confidence_hint=baseline_confidence,
        score_delta=0.0,
        confidence_delta=0.0,
        agreement_points=["base branch verdicts packaged for Codex master review"],
        conflict_points=[],
        rationale_points=[
            f"baseline_score={baseline_score:.3f}",
            f"baseline_confidence={baseline_confidence:.3f}",
        ],
        risk_flags=_dedupe_texts(
            [
                str(item)
                for item in packet.risk_summary.get("risk_flags", [])
                if str(item).strip()
            ]
        ),
        telemetry=telemetry,
        metadata={
            **dict(packet.metadata or {}),
            "model": model or "codex-handoff",
            "stage": "review_master_symbol",
            "actor_name": f"IC:{packet.symbol}",
            "symbol": packet.symbol,
            "review_layer_mode": "codex_handoff",
            "codex_handoff_pending": True,
            "hard_veto": packet.hard_veto,
            "baseline_score": baseline_score,
            "baseline_confidence": baseline_confidence,
            "codex_review_packet": packet.to_dict(),
        },
    )


def _build_symbol_quant_verdict(
    *,
    symbol: str,
    quant_result: BranchResult,
) -> BranchVerdict:
    score = float(quant_result.symbol_scores.get(symbol, quant_result.final_score))
    confidence = float(quant_result.final_confidence or 0.0)
    return BranchVerdict(
        agent_name="quant",
        thesis=str(quant_result.conclusion or "quant 分支完成全市场因子评分。"),
        symbol=symbol,
        direction=_direction_enum_from_score(score),
        action=_score_to_action(score),
        final_score=score,
        final_confidence=confidence,
        investment_risks=list(quant_result.investment_risks),
        coverage_notes=list(quant_result.coverage_notes),
        diagnostic_notes=list(quant_result.diagnostic_notes),
        metadata={
            **dict(quant_result.metadata or {}),
            "branch_name": "quant",
            "source": "full_market_quant_result",
        },
    )


def _build_symbol_macro_verdict(
    *,
    symbol: str,
    macro_verdict: BranchVerdict,
) -> BranchVerdict:
    score = float(macro_verdict.final_score)
    return BranchVerdict(
        agent_name="macro",
        thesis=str(macro_verdict.thesis or "macro 分支完成市场状态评估。"),
        symbol=symbol,
        direction=_direction_enum_from_score(score),
        action=_score_to_action(score),
        final_score=score,
        final_confidence=float(macro_verdict.final_confidence),
        investment_risks=list(macro_verdict.investment_risks),
        coverage_notes=list(macro_verdict.coverage_notes),
        diagnostic_notes=list(macro_verdict.diagnostic_notes),
        metadata={
            **dict(macro_verdict.metadata or {}),
            "branch_name": "macro",
            "source": "market_macro_verdict",
        },
    )


def _build_no_candidate_fundamental_outputs() -> tuple[BranchVerdict, BranchResult]:
    diagnostic = "not_run_no_candidates"
    generation_evidence = {
        "status": "UNCONFIRMED",
        "all_symbols_confirmed": False,
        "reason": diagnostic,
    }
    metadata = {
        "branch_name": "fundamental",
        "degraded_reason": diagnostic,
        "reliability": 0.0,
        "horizon_days": 5,
        "fundamental_data_generation_status": "UNCONFIRMED",
        "fundamental_data_generation_by_symbol": {},
        "fundamental_data_generation_status_by_symbol": {},
        "fundamental_data_generation_evidence": generation_evidence,
    }
    thesis = "No candidates entered Fundamental research; evidence is UNCONFIRMED."
    verdict = BranchVerdict(
        agent_name="fundamental",
        thesis=thesis,
        final_score=0.0,
        final_confidence=0.0,
        status=AgentStatus.DEGRADED,
        coverage_notes=["No candidate symbols were available for Fundamental research."],
        diagnostic_notes=[diagnostic],
        metadata=deepcopy(metadata),
    )
    result = BranchResult(
        branch_name="fundamental",
        success=False,
        final_score=0.0,
        final_confidence=0.0,
        symbol_scores={},
        conclusion=thesis,
        coverage_notes=list(verdict.coverage_notes),
        diagnostic_notes=[diagnostic],
        metadata=deepcopy(metadata),
    )
    return verdict, result


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
    fundamental_agent: Any,
    quant_result: BranchResult,
    ensure_branch_verdict: Callable[..., BranchVerdict],
    master_hint_to_ic_hint: Callable[[Any], dict[str, Any]],
    branch_data_readiness: Mapping[str, Any] | None = None,
    branch_data_payload: Mapping[str, Any] | None = None,
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
            branch_data_readiness=branch_data_readiness,
            branch_data_payload=branch_data_payload,
        )
        branch_payload = {
            "data_bundle": bundle,
            "stock_pool": [symbol],
            "market": market,
            "verbose": False,
        }
        quant = _build_symbol_quant_verdict(symbol=symbol, quant_result=quant_result)
        fundamental = ensure_branch_verdict(
            fundamental_agent.run({**branch_payload, "enable_document_semantics": True}),
            symbol=symbol,
            branch_name="fundamental",
        )
        macro = _build_symbol_macro_verdict(symbol=symbol, macro_verdict=macro_verdict)
        base_branch_verdicts = {
            "quant": quant,
            "fundamental": fundamental,
            "macro": macro,
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

        codex_handoff_review = _is_codex_handoff_review(
            branch_model_resolution,
            master_model_resolution,
        )
        if codex_handoff_review:
            review_llm = None
            review_master_llm = None
        else:
            review_llm = GatewayLLMClient(timeout=agent_timeout)
            review_master_llm = GatewayLLMClient(timeout=master_timeout)
        branch_names = list(base_branch_verdicts.keys())
        branch_overlay_verdicts: dict[str, Any] = {}
        telemetry: list[Any] = []
        fallback_reasons: list[str] = []
        for branch_name in branch_names:
            base_verdict = base_branch_verdicts[branch_name]
            overlay_packet = BranchOverlayPacket(
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
                metadata={
                    "source_branch": branch_name,
                    "symbol": symbol,
                    "resolver": read_result.resolver_trace,
                    "review_layer_mode": (
                        "codex_handoff"
                        if codex_handoff_review
                        else "local_llm"
                    ),
                },
            )
            if codex_handoff_review:
                overlay = _build_codex_handoff_overlay(
                    overlay_packet,
                    model=branch_model_resolution.resolved_model or "codex-handoff",
                )
            else:
                reviewer = BranchOverlayReviewer(
                    branch_name=branch_name,
                    llm_client=review_llm,
                    model=branch_model_resolution.resolved_model,
                    candidate_models=list(branch_candidate_models),
                    fallback_model=branch_model_resolution.fallback_model,
                    timeout=agent_timeout,
                    max_tokens=600,
                )
                overlay = await reviewer.review(overlay_packet)
            branch_overlay_verdicts[branch_name] = overlay
            telemetry.append(overlay.telemetry)
            if (
                (not codex_handoff_review)
                and overlay.telemetry.fallback
                and overlay.telemetry.fallback_reason
            ):
                fallback_reasons.append(
                    f"{symbol}/{branch_name}: "
                    f"{overlay.telemetry.fallback_reason}"
                )

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
                    + [
                        risk
                        for verdict in base_branch_verdicts.values()
                        for risk in verdict.investment_risks[:1]
                    ][:2]
                ),
            },
            baseline_score=float(fmean([item["adjusted_score"] for item in overlay_dicts]) if overlay_dicts else 0.0),
            baseline_confidence=float(fmean([item["adjusted_confidence"] for item in overlay_dicts]) if overlay_dicts else 0.0),
            hard_veto=bool(False),
            metadata={
                "symbol": symbol,
                "resolver": read_result.resolver_trace,
                "review_layer_mode": (
                    "codex_handoff"
                    if codex_handoff_review
                    else "local_llm"
                ),
            },
        )
        if codex_handoff_review:
            master_hint = _build_codex_handoff_master(
                master_packet,
                model=master_model_resolution.resolved_model or "codex-handoff",
            )
        else:
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
        if (not codex_handoff_review) and master_hint.telemetry.fallback and master_hint.telemetry.fallback_reason:
            fallback_reasons.append(f"{symbol}: {master_hint.telemetry.fallback_reason}")
        if codex_handoff_review:
            fallback_reasons.append(f"{symbol}: codex_handoff_pending")

        # LLM overlays remain report-only evidence. The deterministic DAG and
        # control chain consume the exact base verdict objects for every branch.
        reviewed_branch_verdicts = dict(base_branch_verdicts)
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
                    "advisory_only": True,
                    "deterministic_control_chain_isolated": True,
                    "review_layer_mode": (
                        "codex_handoff"
                        if codex_handoff_review
                        else "local_llm"
                    ),
                    "codex_handoff_pending": bool(codex_handoff_review),
                    "codex_handoff_packet_count": (
                        (len(branch_names) + 1)
                        if codex_handoff_review
                        else 0
                    ),
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
    codex_handoff_review = bool(enable_agent_layer) and _is_codex_handoff_review(
        branch_model_resolution,
        master_model_resolution,
    )
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
            "advisory_only": True,
            "deterministic_control_chain_isolated": True,
            "review_layer_mode": (
                "codex_handoff"
                if codex_handoff_review
                else ("local_llm" if enable_agent_layer else "disabled")
            ),
            "codex_handoff_pending": bool(codex_handoff_review),
            "codex_handoff_packet_count": (
                len(candidate_symbols) * 5
                if codex_handoff_review
                else 0
            ),
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
        if isinstance(item, BaseException):
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
    empty_fundamental_result: BranchResult | None = None
    if not candidate_symbols:
        empty_fundamental_verdict, empty_fundamental_result = (
            _build_no_candidate_fundamental_outputs()
        )
        branch_summaries["fundamental"] = empty_fundamental_verdict
    branch_summaries["quant"] = global_quant_verdict
    branch_summaries["macro"] = macro_verdict
    branch_results = _build_branch_results(research_by_symbol, branch_summaries)
    branch_results["quant"] = quant_result
    if empty_fundamental_result is not None:
        branch_results["fundamental"] = empty_fundamental_result
    branch_summaries = {
        name: branch_summaries[name]
        for name in CANONICAL_BRANCH_ORDER
        if name in branch_summaries
    }
    branch_results = {
        name: branch_results[name]
        for name in CANONICAL_BRANCH_ORDER
        if name in branch_results
    }

    return CandidateResearchState(
        symbol_research_packets=symbol_research_packets,
        research_by_symbol=research_by_symbol,
        review_bundle=review_bundle,
        ic_hints_by_symbol=ic_hints_by_symbol,
        branch_summaries=branch_summaries,
        branch_results=branch_results,
    )
