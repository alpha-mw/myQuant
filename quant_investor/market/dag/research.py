from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import Any, Callable, Mapping

import pandas as pd

from quant_investor.agent_protocol import (
    ActionLabel,
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
from quant_investor.branch_contracts import BranchResult
from quant_investor.fundamental_research.runtime import consume_overlay
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
            packet.thesis or f"{packet.symbol} {packet.branch_name} branch awaiting Codex review."
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
        thesis=(f"{packet.symbol} per-symbol master review awaiting Codex handoff."),
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
            [str(item) for item in packet.risk_summary.get("risk_flags", []) if str(item).strip()]
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


async def _run_candidate_research_phase(
    *,
    candidate_symbols: list[str],
    company_name_map: Mapping[str, str],
    market: str,
    market_snapshot: Mapping[str, Any],
    industry_map: Mapping[str, str],
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
    intelligence_agent: Any,
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
        deterministic_fundamental = fundamental
        fundamental_data_generation = str(
            dict(
                deterministic_fundamental.metadata.get("fundamental_data_generation_by_symbol", {})
                or {}
            ).get(symbol, "")
        )
        runtime_cutoff = market_snapshot.get("latest_trade_date", "")
        fundamental_runtime = consume_overlay(
            market=market,
            symbol=symbol,
            base_score=float(deterministic_fundamental.final_score),
            run_cutoff=runtime_cutoff,
            run_key=f"{market}:{universe_key}:{runtime_cutoff}",
            current_data_generation=fundamental_data_generation,
        )
        module_statuses = dict(
            deterministic_fundamental.metadata.get("structured_signals", {}).get(
                "module_coverages", {}
            )
            or {}
        )
        available_modules = sorted(
            name
            for name, status in module_statuses.items()
            if str(status) in {"available", "partial"}
        )
        missing_modules = sorted(
            name
            for name, status in module_statuses.items()
            if str(status) not in {"available", "partial"}
        )
        local_fundamental_context = dict(bundle.fundamentals.get(symbol, {}) or {})
        industry = str(
            local_fundamental_context.get("industry")
            or local_fundamental_context.get("sector")
            or industry_map.get(symbol)
            or "UNCONFIRMED"
        )
        explicit_peer_set_confirmed = (
            str(local_fundamental_context.get("peer_set_status", "")).lower() == "confirmed"
        )
        explicit_peer_symbols = (
            sorted(
                {
                    str(item).strip()
                    for item in list(local_fundamental_context.get("peer_symbols", []) or [])
                    if str(item).strip() and str(item).strip() != symbol
                }
            )
            if explicit_peer_set_confirmed
            else []
        )
        derived_peer_symbols = sorted(
            peer_symbol
            for peer_symbol, peer_industry in industry_map.items()
            if peer_symbol != symbol
            and industry != "UNCONFIRMED"
            and str(peer_industry).strip() == industry
        )[:10]
        if explicit_peer_set_confirmed:
            peer_symbols = explicit_peer_symbols[:10]
            peer_set_status = "confirmed"
            peer_set_source = "explicit_local_mart"
        elif derived_peer_symbols:
            peer_symbols = derived_peer_symbols
            peer_set_status = "confirmed"
            peer_set_source = "derived_local_industry"
        else:
            peer_symbols = []
            peer_set_status = "unconfirmed"
            peer_set_source = "UNCONFIRMED"
        valuation_price: float | None = None
        valuation_price_as_of = ""
        if frame is not None and not frame.empty and "close" in frame.columns:
            closes = pd.to_numeric(frame["close"], errors="coerce")
            valid_closes = closes[closes.map(lambda value: pd.notna(value) and float(value) > 0)]
            if not valid_closes.empty:
                last_index = valid_closes.index[-1]
                candidate_price = float(valid_closes.loc[last_index])
                if math.isfinite(candidate_price):
                    valuation_price = candidate_price
                    for date_column in ("date", "trade_date"):
                        if date_column not in frame.columns:
                            continue
                        raw_date = frame.loc[last_index, date_column]
                        if pd.isna(raw_date):
                            continue
                        parsed_date = pd.to_datetime(raw_date, errors="coerce")
                        if pd.notna(parsed_date):
                            valuation_price_as_of = parsed_date.date().isoformat()
                            break
        context_blockers = [] if peer_set_status == "confirmed" else ["peer_set_UNCONFIRMED"]
        if valuation_price is None or not valuation_price_as_of:
            context_blockers.append("valuation_price_UNCONFIRMED")
        deterministic_base_record = {
            "company_name": company_name_map.get(symbol, ""),
            "industry": industry,
            "peer_symbols": peer_symbols,
            "peer_set_status": peer_set_status,
            "peer_set_source": peer_set_source,
            "base_score": float(deterministic_fundamental.final_score),
            "base_confidence": float(deterministic_fundamental.final_confidence),
            "valuation_price": valuation_price,
            "valuation_price_as_of": valuation_price_as_of,
            "data_generation": fundamental_data_generation,
            "available_modules": available_modules,
            "missing_modules": missing_modules,
            "status": getattr(
                deterministic_fundamental.status,
                "value",
                str(deterministic_fundamental.status),
            ),
            "context_blockers": context_blockers,
            "runtime_audit": dict(fundamental_runtime.metadata),
        }
        fundamental = replace(
            deterministic_fundamental,
            final_score=(
                float(fundamental_runtime.adjusted_score)
                if fundamental_runtime.applied and fundamental_runtime.adjusted_score is not None
                else float(deterministic_fundamental.final_score)
            ),
            direction=(
                _direction_enum_from_score(float(fundamental_runtime.adjusted_score))
                if fundamental_runtime.applied and fundamental_runtime.adjusted_score is not None
                else deterministic_fundamental.direction
            ),
            action=(
                _score_to_action(float(fundamental_runtime.adjusted_score))
                if fundamental_runtime.applied and fundamental_runtime.adjusted_score is not None
                else deterministic_fundamental.action
            ),
            metadata={
                **dict(deterministic_fundamental.metadata or {}),
                "deterministic_base_score": float(deterministic_fundamental.final_score),
                "fundamental_deterministic_control_input": {
                    "thesis": deterministic_fundamental.thesis,
                    "status": deterministic_fundamental.status.value,
                    "direction": deterministic_fundamental.direction.value,
                    "action": deterministic_fundamental.action.value,
                    "confidence_label": (deterministic_fundamental.confidence_label.value),
                    "final_score": float(deterministic_fundamental.final_score),
                    "final_confidence": float(deterministic_fundamental.final_confidence),
                    "investment_risks": list(deterministic_fundamental.investment_risks),
                    "coverage_notes": list(deterministic_fundamental.coverage_notes),
                    "diagnostic_notes": list(deterministic_fundamental.diagnostic_notes),
                },
                "fundamental_research_runtime": dict(fundamental_runtime.metadata),
            },
        )
        intelligence = ensure_branch_verdict(
            intelligence_agent.run(
                {**branch_payload, "market_regime": macro_verdict.metadata.get("regime", "neutral")}
            ),
            symbol=symbol,
            branch_name="intelligence",
        )
        macro = _build_symbol_macro_verdict(symbol=symbol, macro_verdict=macro_verdict)
        base_branch_verdicts = {
            "quant": quant,
            "fundamental": fundamental,
            "intelligence": intelligence,
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
            packet.metadata["fundamental_deterministic_base"] = dict(deterministic_base_record)
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
        if fundamental_runtime.suppress_generic_fundamental_overlay:
            runtime_audit = dict(fundamental_runtime.metadata)
            runtime_telemetry = ReviewTelemetry(
                stage="fundamental_research_runtime",
                model="codex-external-dossier",
                provider="codex",
                success=True,
                fallback=False,
                score_delta=float(runtime_audit.get("computed_delta", 0.0)),
                confidence_delta=0.0,
                metadata=runtime_audit,
            )
            branch_overlay_verdicts["fundamental"] = BranchOverlayVerdict(
                symbol=symbol,
                branch_name="fundamental",
                status=fundamental.status,
                thesis=fundamental.thesis,
                direction=fundamental.direction,
                action=fundamental.action,
                base_score=float(deterministic_fundamental.final_score),
                adjusted_score=float(fundamental.final_score),
                base_confidence=float(deterministic_fundamental.final_confidence),
                adjusted_confidence=float(deterministic_fundamental.final_confidence),
                score_delta=float(runtime_audit.get("computed_delta", 0.0)),
                confidence_delta=0.0,
                agreement_points=["validated prior-run fundamental dossier applied"],
                risk_flags=list(fundamental.investment_risks),
                telemetry=runtime_telemetry,
                metadata={
                    "source": "fundamental_research_v13_2",
                    "generic_overlay_suppressed": True,
                    "runtime_audit": runtime_audit,
                },
            )
            telemetry.append(runtime_telemetry)
        for branch_name in branch_names:
            if (
                branch_name == "fundamental"
                and fundamental_runtime.suppress_generic_fundamental_overlay
            ):
                continue
            base_verdict = base_branch_verdicts[branch_name]
            overlay_packet = BranchOverlayPacket(
                symbol=symbol,
                branch_name=branch_name,
                base_score=float(base_verdict.final_score),
                base_confidence=float(base_verdict.final_confidence),
                thesis=str(base_verdict.thesis),
                direction=_score_to_direction(float(base_verdict.final_score)),
                action=_score_to_action(float(base_verdict.final_score)).value,
                agreement_points=_dedupe_texts(
                    list(base_verdict.coverage_notes[:3]) or [base_verdict.thesis]
                ),
                conflict_points=_dedupe_texts(
                    list(base_verdict.diagnostic_notes[:3])
                    or list(base_verdict.investment_risks[:3])
                ),
                risk_points=_dedupe_texts(list(base_verdict.investment_risks[:4])),
                branch_signals={
                    "score": float(base_verdict.final_score),
                    "confidence": float(base_verdict.final_confidence),
                },
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
                    "review_layer_mode": ("codex_handoff" if codex_handoff_review else "local_llm"),
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
                    f"{symbol}/{branch_name}: " f"{overlay.telemetry.fallback_reason}"
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
            baseline_score=float(
                fmean([item["adjusted_score"] for item in overlay_dicts]) if overlay_dicts else 0.0
            ),
            baseline_confidence=float(
                fmean([item["adjusted_confidence"] for item in overlay_dicts])
                if overlay_dicts
                else 0.0
            ),
            hard_veto=bool(False),
            metadata={
                "symbol": symbol,
                "resolver": read_result.resolver_trace,
                "review_layer_mode": ("codex_handoff" if codex_handoff_review else "local_llm"),
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
        if (
            (not codex_handoff_review)
            and master_hint.telemetry.fallback
            and master_hint.telemetry.fallback_reason
        ):
            fallback_reasons.append(f"{symbol}: {master_hint.telemetry.fallback_reason}")
        if codex_handoff_review:
            fallback_reasons.append(f"{symbol}: codex_handoff_pending")

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
                direction=(
                    overlay.direction
                    if isinstance(overlay.direction, Direction)
                    else base_verdict.direction
                ),
                action=(
                    overlay.action
                    if isinstance(overlay.action, ActionLabel)
                    else base_verdict.action
                ),
                confidence_label=base_verdict.confidence_label,
                final_score=float(overlay.adjusted_score),
                final_confidence=float(overlay.adjusted_confidence),
                investment_risks=_dedupe_texts(
                    list(base_verdict.investment_risks)
                    + list(overlay.risk_flags)
                    + list(overlay.missing_risks)
                ),
                coverage_notes=_dedupe_texts(
                    list(base_verdict.coverage_notes) + list(overlay.agreement_points)
                ),
                diagnostic_notes=_dedupe_texts(
                    list(base_verdict.diagnostic_notes)
                    + list(overlay.conflict_points)
                    + list(overlay.contradictions)
                ),
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
                    "review_layer_mode": ("codex_handoff" if codex_handoff_review else "local_llm"),
                    "codex_handoff_pending": bool(codex_handoff_review),
                    "codex_handoff_packet_count": (
                        (len(branch_names) + 1) if codex_handoff_review else 0
                    ),
                    "universe_key": universe_key,
                    "symbol_count": len(candidate_symbols),
                    "resolver": read_result.resolver_trace,
                },
            ),
        )
        packet.metadata["fundamental_deterministic_base"] = dict(deterministic_base_record)
        return (
            symbol,
            reviewed_branch_verdicts,
            packet,
            master_hint,
            master_hint_to_ic_hint(master_hint),
            dict(branch_overlay_verdicts),
            telemetry,
            fallback_reasons,
        )

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
            "review_layer_mode": (
                "codex_handoff"
                if codex_handoff_review
                else ("local_llm" if enable_agent_layer else "disabled")
            ),
            "codex_handoff_pending": bool(codex_handoff_review),
            "codex_handoff_packet_count": (
                len(candidate_symbols) * 5 if codex_handoff_review else 0
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
        (
            symbol,
            reviewed_branch_verdicts,
            packet,
            master_hint,
            ic_hint,
            branch_overlays,
            telemetry,
            fallbacks,
        ) = item
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
    review_bundle.metadata["fundamental_research_runtime"] = {
        symbol: dict(
            packet.metadata.get("fundamental_deterministic_base", {}).get("runtime_audit", {})
        )
        for symbol, packet in symbol_research_packets.items()
    }

    branch_summaries = _aggregate_branch_summaries(research_by_symbol)
    branch_summaries["quant"] = global_quant_verdict
    branch_summaries["macro"] = macro_verdict
    branch_results = _build_branch_results(research_by_symbol, branch_summaries)
    branch_results["quant"] = quant_result

    return CandidateResearchState(
        symbol_research_packets=symbol_research_packets,
        research_by_symbol=research_by_symbol,
        review_bundle=review_bundle,
        ic_hints_by_symbol=ic_hints_by_symbol,
        branch_summaries=branch_summaries,
        branch_results=branch_results,
    )
