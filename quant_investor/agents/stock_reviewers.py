"""
Symbol-scoped LLM review helpers.

These reviewers consume compact structured packets only. They do not read raw
market datasets directly and they always degrade to a safe deterministic
fallback when the provider is unavailable or the LLM call fails.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchOverlayVerdict,
    Direction,
    MasterICHint,
    ReviewTelemetry,
)
from quant_investor.agents.prompts import (
    BRANCH_OVERLAY_CONFIDENCE_CAP,
    BRANCH_OVERLAY_SCORE_CAP,
    MASTER_HINT_CONFIDENCE_CAP,
    MASTER_HINT_SCORE_CAP,
    build_branch_overlay_messages,
    build_master_symbol_messages,
    format_agent_display_name,
)
from quant_investor.llm_gateway import (
    LLMCallError,
    LLMClient,
    current_usage_session_id,
    has_provider_for_model,
    snapshot_usage,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _dedupe_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _coerce_text_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _score_to_direction(score: float) -> Direction:
    if score >= 0.15:
        return Direction.BULLISH
    if score <= -0.15:
        return Direction.BEARISH
    return Direction.NEUTRAL


def _score_to_action(score: float) -> ActionLabel:
    if score >= 0.25:
        return ActionLabel.BUY
    if score <= -0.35:
        return ActionLabel.SELL
    return ActionLabel.HOLD


def _direction_to_text(direction: Direction | str) -> str:
    label = direction.value if isinstance(direction, Direction) else str(direction or "").strip().lower()
    if label not in {"bullish", "bearish", "neutral"}:
        return "neutral"
    return label


def _action_to_text(action: ActionLabel | str) -> str:
    label = action.value if isinstance(action, ActionLabel) else str(action or "").strip().lower()
    if label not in {"buy", "hold", "sell", "watch", "avoid"}:
        return "hold"
    return label


@dataclass
class BranchOverlayPacket:
    symbol: str
    branch_name: str
    base_score: float
    base_confidence: float
    thesis: str
    direction: str
    action: str
    agreement_points: list[str] = field(default_factory=list)
    conflict_points: list[str] = field(default_factory=list)
    risk_points: list[str] = field(default_factory=list)
    branch_signals: dict[str, Any] = field(default_factory=dict)
    macro_summary: dict[str, Any] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MasterSymbolPacket:
    symbol: str
    branch_overlay_summaries: list[dict[str, Any]] = field(default_factory=list)
    macro_summary: dict[str, Any] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    baseline_score: float = 0.0
    baseline_confidence: float = 0.0
    hard_veto: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BranchOverlayReviewer:
    """Bounded branch-by-branch review for a single symbol."""

    def __init__(
        self,
        *,
        branch_name: str,
        llm_client: LLMClient,
        model: str,
        candidate_models: list[str] | None = None,
        fallback_model: str = "",
        timeout: float = 15.0,
        max_tokens: int = 600,
        score_cap: float | None = None,
        confidence_cap: float | None = None,
    ) -> None:
        self.branch_name = str(branch_name)
        self.llm_client = llm_client
        self.model = str(model or "").strip()
        self.candidate_models = [str(item).strip() for item in list(candidate_models or []) if str(item).strip()]
        self.fallback_model = str(fallback_model or "").strip()
        self.timeout = max(float(timeout), 0.1)
        self.max_tokens = max(int(max_tokens), 16)
        self.score_cap = float(score_cap if score_cap is not None else BRANCH_OVERLAY_SCORE_CAP.get(self.branch_name, 0.10))
        self.confidence_cap = float(
            confidence_cap if confidence_cap is not None else BRANCH_OVERLAY_CONFIDENCE_CAP.get(self.branch_name, 0.10)
        )

    async def review(self, packet: BranchOverlayPacket) -> BranchOverlayVerdict:
        t0 = time.monotonic()
        actor_name = f"{packet.symbol}:{self.branch_name}"
        stage = "review_branch_overlay"
        if (
            (not self.model or not has_provider_for_model(self.model))
            and (not self.fallback_model or not has_provider_for_model(self.fallback_model))
        ):
            return self._fallback(packet, stage=stage, actor_name=actor_name, reason="llm_provider_missing", started_at=t0)

        messages = build_branch_overlay_messages(self.branch_name, json.dumps(packet.to_dict(), ensure_ascii=False, sort_keys=True))
        try:
            raw = await self.llm_client.complete(
                messages=messages,
                model=self.model,
                candidate_models=self.candidate_models,
                fallback_model=self.fallback_model,
                max_tokens=self.max_tokens,
                response_json=True,
                stage=stage,
                actor_name=actor_name,
            )
            return self._from_raw(packet, raw, stage=stage, actor_name=actor_name, started_at=t0)
        except (LLMCallError, Exception) as exc:
            return self._fallback(packet, stage=stage, actor_name=actor_name, reason=str(exc), started_at=t0)

    def _from_raw(
        self,
        packet: BranchOverlayPacket,
        raw: Mapping[str, Any],
        *,
        stage: str,
        actor_name: str,
        started_at: float,
    ) -> BranchOverlayVerdict:
        score_delta = _clamp(raw.get("score_delta", 0.0), -self.score_cap, self.score_cap)
        confidence_delta = _clamp(raw.get("confidence_delta", 0.0), -self.confidence_cap, self.confidence_cap)
        adjusted_score = _clamp(packet.base_score + score_delta, -1.0, 1.0)
        adjusted_confidence = _clamp(packet.base_confidence + confidence_delta, 0.0, 1.0)
        direction = _score_to_direction(adjusted_score)
        action = _score_to_action(adjusted_score)
        thesis = str(raw.get("thesis") or packet.thesis).strip() or packet.thesis
        agreement_points = _dedupe_texts(_coerce_text_list(raw.get("agreement_points")) + packet.agreement_points[:2])
        conflict_points = _dedupe_texts(_coerce_text_list(raw.get("conflict_points")) + packet.conflict_points[:2])
        missing_risks = _dedupe_texts(_coerce_text_list(raw.get("missing_risks")) + packet.risk_points[:3])
        contradictions = _dedupe_texts(_coerce_text_list(raw.get("contradictions")))
        risk_flags = _dedupe_texts(_coerce_text_list(raw.get("risk_flags")) + packet.risk_points[:3])

        telemetry = self._build_telemetry(
            stage=stage,
            actor_name=actor_name,
            started_at=started_at,
            fallback=False,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
        )
        return BranchOverlayVerdict(
            symbol=packet.symbol,
            branch_name=packet.branch_name,
            status=AgentStatus.SUCCESS,
            thesis=thesis,
            direction=direction,
            action=action,
            base_score=packet.base_score,
            adjusted_score=adjusted_score,
            base_confidence=packet.base_confidence,
            adjusted_confidence=adjusted_confidence,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
            agreement_points=agreement_points,
            conflict_points=conflict_points,
            missing_risks=missing_risks,
            contradictions=contradictions,
            risk_flags=risk_flags,
            telemetry=telemetry,
            metadata={
                **(dict(raw.get("metadata", {})) if isinstance(raw.get("metadata", {}), Mapping) else {}),
                "model": self.model,
                "stage": stage,
                "actor_name": actor_name,
                "branch_name": packet.branch_name,
                "symbol": packet.symbol,
            },
        )

    def _fallback(
        self,
        packet: BranchOverlayPacket,
        *,
        stage: str,
        actor_name: str,
        reason: str,
        started_at: float,
    ) -> BranchOverlayVerdict:
        risk_pressure = min(len(packet.risk_points) * 0.02, self.score_cap)
        agreement_bias = min(len(packet.agreement_points) * 0.01, self.score_cap)
        score_delta = _clamp(agreement_bias - risk_pressure, -self.score_cap, self.score_cap)
        confidence_delta = _clamp(-0.03 - min(len(packet.conflict_points) * 0.01, self.confidence_cap), -self.confidence_cap, self.confidence_cap)
        adjusted_score = _clamp(packet.base_score + score_delta, -1.0, 1.0)
        adjusted_confidence = _clamp(packet.base_confidence + confidence_delta, 0.0, 1.0)
        telemetry = self._build_telemetry(
            stage=stage,
            actor_name=actor_name,
            started_at=started_at,
            fallback=True,
            fallback_reason=reason,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
        )
        return BranchOverlayVerdict(
            symbol=packet.symbol,
            branch_name=packet.branch_name,
            status=AgentStatus.DEGRADED,
            thesis=packet.thesis or f"{packet.symbol} {packet.branch_name} branch fallback review.",
            direction=_score_to_direction(adjusted_score),
            action=_score_to_action(adjusted_score),
            base_score=packet.base_score,
            adjusted_score=adjusted_score,
            base_confidence=packet.base_confidence,
            adjusted_confidence=adjusted_confidence,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
            agreement_points=_dedupe_texts(packet.agreement_points[:3]),
            conflict_points=_dedupe_texts(packet.conflict_points[:3]),
            missing_risks=_dedupe_texts(packet.risk_points[:3]),
            contradictions=[],
            risk_flags=_dedupe_texts(packet.risk_points[:5]),
            telemetry=telemetry,
            metadata={
                "model": self.model,
                "stage": stage,
                "actor_name": actor_name,
                "fallback_reason": reason,
                "branch_name": packet.branch_name,
                "symbol": packet.symbol,
                "deterministic_fallback": True,
            },
        )

    def _build_telemetry(
        self,
        *,
        stage: str,
        actor_name: str,
        started_at: float,
        fallback: bool,
        score_delta: float,
        confidence_delta: float,
        fallback_reason: str = "",
    ) -> ReviewTelemetry:
        latency_ms = int((time.monotonic() - started_at) * 1000)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        provider = ""
        session_id = current_usage_session_id()
        if session_id:
            records, _ = snapshot_usage(session_id)
            for record in reversed(records):
                if (
                    record.stage == stage
                    and record.branch_or_agent_name == actor_name
                    and record.model == self.model
                ):
                    prompt_tokens = int(record.prompt_tokens)
                    completion_tokens = int(record.completion_tokens)
                    total_tokens = int(record.total_tokens)
                    provider = str(record.provider or "")
                    if not fallback_reason and record.fallback:
                        fallback_reason = str(record.metadata.get("reason", ""))
                    break

        return ReviewTelemetry(
            stage=stage,
            model=self.model,
            provider=provider,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            success=not fallback,
            fallback=fallback,
            fallback_reason=fallback_reason,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
            metadata={
                "actor_name": actor_name,
                "model_label": format_agent_display_name(self.branch_name),
            },
        )


class MasterICAgent:
    """Bounded per-symbol master synthesis over overlay verdicts."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        model: str,
        candidate_models: list[str] | None = None,
        fallback_model: str = "",
        reasoning_effort: str = "high",
        timeout: float = 30.0,
        max_tokens: int = 900,
        score_cap: float = MASTER_HINT_SCORE_CAP,
        confidence_cap: float = MASTER_HINT_CONFIDENCE_CAP,
    ) -> None:
        self.llm_client = llm_client
        self.model = str(model or "").strip()
        self.candidate_models = [str(item).strip() for item in list(candidate_models or []) if str(item).strip()]
        self.fallback_model = str(fallback_model or "").strip()
        self.reasoning_effort = str(reasoning_effort or "").strip() or "high"
        self.timeout = max(float(timeout), 0.1)
        self.max_tokens = max(int(max_tokens), 16)
        self.score_cap = float(score_cap)
        self.confidence_cap = float(confidence_cap)

    async def deliberate(self, packet: MasterSymbolPacket) -> MasterICHint:
        t0 = time.monotonic()
        actor_name = f"IC:{packet.symbol}"
        stage = "review_master_symbol"
        if (
            (not self.model or not has_provider_for_model(self.model))
            and (not self.fallback_model or not has_provider_for_model(self.fallback_model))
        ):
            return self._fallback(packet, stage=stage, actor_name=actor_name, reason="llm_provider_missing", started_at=t0)

        messages = build_master_symbol_messages(packet.symbol, json.dumps(packet.to_dict(), ensure_ascii=False, sort_keys=True))
        try:
            raw = await self.llm_client.complete(
                messages=messages,
                model=self.model,
                candidate_models=self.candidate_models,
                fallback_model=self.fallback_model,
                max_tokens=self.max_tokens,
                response_json=True,
                stage=stage,
                actor_name=actor_name,
                reasoning_effort=self.reasoning_effort,
            )
            return self._from_raw(packet, raw, stage=stage, actor_name=actor_name, started_at=t0)
        except (LLMCallError, Exception) as exc:
            return self._fallback(packet, stage=stage, actor_name=actor_name, reason=str(exc), started_at=t0)

    def _from_raw(
        self,
        packet: MasterSymbolPacket,
        raw: Mapping[str, Any],
        *,
        stage: str,
        actor_name: str,
        started_at: float,
    ) -> MasterICHint:
        baseline_score = _clamp(packet.baseline_score, -1.0, 1.0)
        baseline_confidence = _clamp(packet.baseline_confidence, 0.0, 1.0)
        score_delta = _clamp(raw.get("score_delta", 0.0), -self.score_cap, self.score_cap)
        confidence_delta = _clamp(raw.get("confidence_delta", 0.0), -self.confidence_cap, self.confidence_cap)
        score_hint = _clamp(baseline_score + score_delta, -1.0, 1.0)
        confidence_hint = _clamp(baseline_confidence + confidence_delta, 0.0, 1.0)
        if packet.hard_veto:
            score_hint = min(score_hint, 0.0)
        direction = _score_to_direction(score_hint)
        action = _score_to_action(score_hint)
        if packet.hard_veto and action == ActionLabel.BUY:
            action = ActionLabel.HOLD

        telemetry = self._build_telemetry(
            stage=stage,
            actor_name=actor_name,
            started_at=started_at,
            fallback=False,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
        )
        return MasterICHint(
            symbol=packet.symbol,
            status=AgentStatus.VETOED if packet.hard_veto else AgentStatus.SUCCESS,
            thesis=str(raw.get("thesis") or f"{packet.symbol} Master hint synthesized.").strip(),
            action=action,
            direction=direction,
            score_hint=score_hint,
            confidence_hint=confidence_hint,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
            agreement_points=_dedupe_texts(_coerce_text_list(raw.get("agreement_points"))),
            conflict_points=_dedupe_texts(_coerce_text_list(raw.get("conflict_points"))),
            rationale_points=_dedupe_texts(_coerce_text_list(raw.get("rationale_points"))),
            risk_flags=_dedupe_texts(_coerce_text_list(raw.get("risk_flags"))),
            telemetry=telemetry,
            metadata={
                **(dict(raw.get("metadata", {})) if isinstance(raw.get("metadata", {}), Mapping) else {}),
                "model": self.model,
                "stage": stage,
                "actor_name": actor_name,
                "symbol": packet.symbol,
                "hard_veto": packet.hard_veto,
                "baseline_score": baseline_score,
                "baseline_confidence": baseline_confidence,
            },
        )

    def _fallback(
        self,
        packet: MasterSymbolPacket,
        *,
        stage: str,
        actor_name: str,
        reason: str,
        started_at: float,
    ) -> MasterICHint:
        branch_scores = [float(item.get("adjusted_score", item.get("base_score", 0.0))) for item in packet.branch_overlay_summaries]
        branch_confidences = [float(item.get("adjusted_confidence", item.get("base_confidence", 0.0))) for item in packet.branch_overlay_summaries]
        baseline_score = _clamp(sum(branch_scores) / max(len(branch_scores), 1), -1.0, 1.0)
        baseline_confidence = _clamp(sum(branch_confidences) / max(len(branch_confidences), 1), 0.0, 1.0)
        score_delta = _clamp((len(packet.macro_summary) - len(packet.risk_summary)) * 0.01, -self.score_cap, self.score_cap)
        confidence_delta = _clamp(-0.04 - min(len(packet.risk_summary) * 0.005, self.confidence_cap), -self.confidence_cap, self.confidence_cap)
        score_hint = _clamp(baseline_score + score_delta, -1.0, 1.0)
        confidence_hint = _clamp(baseline_confidence + confidence_delta, 0.0, 1.0)
        if packet.hard_veto:
            score_hint = min(score_hint, 0.0)
        telemetry = self._build_telemetry(
            stage=stage,
            actor_name=actor_name,
            started_at=started_at,
            fallback=True,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
            fallback_reason=reason,
        )
        return MasterICHint(
            symbol=packet.symbol,
            status=AgentStatus.VETOED if packet.hard_veto else AgentStatus.DEGRADED,
            thesis=f"{packet.symbol} Master hint fallback synthesis.",
            action=ActionLabel.HOLD if score_hint >= 0.0 else ActionLabel.AVOID,
            direction=_score_to_direction(score_hint),
            score_hint=score_hint,
            confidence_hint=confidence_hint,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
            agreement_points=["deterministic fallback synthesis"],
            conflict_points=_dedupe_texts(
                list(packet.risk_summary.get("conflicts", []))
                if isinstance(packet.risk_summary.get("conflicts", []), list)
                else []
            ),
            rationale_points=_dedupe_texts([
                f"baseline_score={baseline_score:.3f}",
                f"baseline_confidence={baseline_confidence:.3f}",
            ]),
            risk_flags=_dedupe_texts([str(item) for item in packet.risk_summary.get("risk_flags", []) if str(item).strip()]),
            telemetry=telemetry,
            metadata={
                "model": self.model,
                "stage": stage,
                "actor_name": actor_name,
                "symbol": packet.symbol,
                "fallback_reason": reason,
                "hard_veto": packet.hard_veto,
                "deterministic_fallback": True,
            },
        )

    def _build_telemetry(
        self,
        *,
        stage: str,
        actor_name: str,
        started_at: float,
        fallback: bool,
        score_delta: float,
        confidence_delta: float,
        fallback_reason: str = "",
    ) -> ReviewTelemetry:
        latency_ms = int((time.monotonic() - started_at) * 1000)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        provider = ""
        session_id = current_usage_session_id()
        if session_id:
            records, _ = snapshot_usage(session_id)
            for record in reversed(records):
                if (
                    record.stage == stage
                    and record.branch_or_agent_name == actor_name
                    and record.model == self.model
                ):
                    prompt_tokens = int(record.prompt_tokens)
                    completion_tokens = int(record.completion_tokens)
                    total_tokens = int(record.total_tokens)
                    provider = str(record.provider or "")
                    if not fallback_reason and record.fallback:
                        fallback_reason = str(record.metadata.get("reason", ""))
                    break
        return ReviewTelemetry(
            stage=stage,
            model=self.model,
            provider=provider,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            success=not fallback,
            fallback=fallback,
            fallback_reason=fallback_reason,
            score_delta=score_delta,
            confidence_delta=confidence_delta,
            metadata={
                "actor_name": actor_name,
                "model_label": "MasterICAgent",
            },
        )
