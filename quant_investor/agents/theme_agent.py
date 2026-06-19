"""Standalone non-canonical ThemeAgent wrapper."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    AgentStatus,
    BranchVerdict,
    CoverageScope,
    EvidenceItem,
)
from quant_investor.agents.base import BaseAgent
from quant_investor.themes.types import ThemePhase, ThemeScanResult, ThemeScore


_PHASE_ADJUSTMENTS = {
    ThemePhase.ACCUMULATION.value: 0.04,
    ThemePhase.EARLY_ACCELERATION.value: 0.08,
    ThemePhase.CONFIRMED_ROTATION.value: 0.10,
    ThemePhase.OVEREXTENDED.value: -0.15,
    ThemePhase.DISTRIBUTION.value: -0.25,
    ThemePhase.UNCLASSIFIED.value: 0.0,
}


class ThemeAgent(BaseAgent):
    """Convert compact theme-rotation metadata into one symbol-level verdict."""

    agent_name = "ThemeAgent"

    def run(self, payload: Mapping[str, Any]) -> BranchVerdict:
        envelope = self.ensure_payload(payload)
        self.require_keys(envelope, "symbol")
        symbol = str(envelope.get("symbol") or "").strip()
        theme_data = self._resolve_theme_data(envelope)
        if theme_data is None:
            return self._neutral_verdict(symbol)

        symbol_scores = self._mapping(theme_data.get("symbol_scores"))
        symbol_primary_theme = self._mapping(theme_data.get("symbol_primary_theme"))
        symbol_phase = self._mapping(theme_data.get("symbol_phase"))
        symbol_risk_flags = self._mapping(theme_data.get("symbol_risk_flags"))
        theme_scores = self._mapping(theme_data.get("theme_scores"))

        if symbol not in symbol_scores or symbol not in symbol_primary_theme:
            return self._neutral_verdict(symbol)

        symbol_theme_score = self.clamp(
            self._float(symbol_scores.get(symbol), 0.0),
            0.0,
            1.0,
        )
        primary_theme_id = str(symbol_primary_theme.get(symbol) or "")
        phase = self._normalize_phase(symbol_phase.get(symbol))
        phase_adjustment = _PHASE_ADJUSTMENTS.get(phase, 0.0)
        final_score = self.clamp((symbol_theme_score - 0.50) * 2.0 + phase_adjustment, -1.0, 1.0)
        if phase == ThemePhase.OVEREXTENDED.value:
            final_score = min(final_score, 0.25)
        if phase == ThemePhase.DISTRIBUTION.value:
            final_score = min(final_score, 0.0)

        theme_score = theme_scores.get(primary_theme_id)
        primary_theme_name = self._theme_value(theme_score, "theme_name", primary_theme_id)
        member_count = int(self._float(self._theme_value(theme_score, "member_count", 0), 0.0))
        evidence = self._theme_value(theme_score, "evidence", [])
        has_evidence = bool(evidence) if isinstance(evidence, list) else bool(str(evidence or "").strip())

        final_confidence = self._confidence(
            phase=phase,
            member_count=member_count,
            has_evidence=has_evidence,
        )
        risk_flags = self._risk_flags(
            symbol_risk_flags.get(symbol),
            phase=phase,
        )
        coverage_notes = self.dedupe_texts(
            [
                f"primary_theme_id={primary_theme_id or 'unknown'}",
                f"primary_theme_name={primary_theme_name or primary_theme_id or 'unknown'}",
                f"symbol_theme_score={symbol_theme_score:.3f}",
                f"phase={phase or ThemePhase.UNCLASSIFIED.value}",
            ]
        )
        metadata = {
            "branch_name": "theme",
            "deterministic_primary": True,
            "no_llm": True,
            "no_network": True,
            "theme_data_available": True,
            "symbol_theme_score": symbol_theme_score,
            "primary_theme_id": primary_theme_id,
            "primary_theme_name": primary_theme_name or primary_theme_id,
            "theme_phase": phase,
            "theme_risk_flags": list(risk_flags),
        }
        thesis = (
            f"Theme rotation metadata maps {symbol} to "
            f"{primary_theme_name or primary_theme_id or 'unknown theme'} "
            f"with phase={phase or ThemePhase.UNCLASSIFIED.value}."
        )
        return BranchVerdict(
            agent_name=self.agent_name,
            thesis=thesis,
            symbol=symbol,
            status=AgentStatus.SUCCESS,
            direction=self.score_to_direction(final_score),
            action=self.score_to_action(final_score),
            confidence_label=self.confidence_to_label(final_confidence),
            final_score=final_score,
            final_confidence=final_confidence,
            evidence=[
                EvidenceItem(
                    source=self.agent_name,
                    summary=thesis,
                    direction=self.score_to_direction(final_score),
                    score=final_score,
                    confidence=final_confidence,
                    scope=CoverageScope.SYMBOL,
                    symbols=[symbol],
                    metadata=dict(metadata),
                )
            ],
            investment_risks=risk_flags,
            coverage_notes=coverage_notes,
            diagnostic_notes=["theme_rotation_metadata", "non_canonical_theme_branch"],
            metadata=metadata,
        )

    def _neutral_verdict(self, symbol: str) -> BranchVerdict:
        metadata = {
            "branch_name": "theme",
            "deterministic_primary": True,
            "no_llm": True,
            "no_network": True,
            "theme_data_available": False,
        }
        return BranchVerdict(
            agent_name=self.agent_name,
            thesis="Theme rotation metadata is unavailable for this symbol.",
            symbol=symbol,
            status=AgentStatus.SUCCESS,
            direction=self.score_to_direction(0.0),
            action=self.score_to_action(0.0),
            confidence_label=self.confidence_to_label(0.0),
            final_score=0.0,
            final_confidence=0.0,
            diagnostic_notes=["theme_data_unavailable"],
            metadata=metadata,
        )

    def _resolve_theme_data(self, envelope: Mapping[str, Any]) -> dict[str, Any] | None:
        theme_scan = envelope.get("theme_scan")
        if isinstance(theme_scan, ThemeScanResult):
            return theme_scan.to_dict()
        if isinstance(theme_scan, Mapping):
            return dict(theme_scan)

        theme_rotation = envelope.get("theme_rotation")
        if isinstance(theme_rotation, Mapping):
            return dict(theme_rotation)

        global_context = envelope.get("global_context")
        metadata = getattr(global_context, "metadata", None)
        if isinstance(global_context, Mapping):
            metadata = global_context.get("metadata", metadata)
        if isinstance(metadata, Mapping):
            context_theme_rotation = metadata.get("theme_rotation")
            if isinstance(context_theme_rotation, Mapping):
                return dict(context_theme_rotation)
            context_theme_scan = metadata.get("theme_scan")
            if isinstance(context_theme_scan, ThemeScanResult):
                return context_theme_scan.to_dict()
            if isinstance(context_theme_scan, Mapping):
                return dict(context_theme_scan)
        return None

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _normalize_phase(value: Any) -> str:
        if isinstance(value, ThemePhase):
            return value.value
        if isinstance(value, Enum):
            return str(value.value)
        text = str(value or "").strip().lower()
        return text if text in _PHASE_ADJUSTMENTS else ThemePhase.UNCLASSIFIED.value

    @classmethod
    def _confidence(cls, *, phase: str, member_count: int, has_evidence: bool) -> float:
        confidence = 0.30
        if phase in {ThemePhase.EARLY_ACCELERATION.value, ThemePhase.CONFIRMED_ROTATION.value}:
            confidence += 0.15
        if member_count >= 10:
            confidence += 0.10
        if has_evidence:
            confidence += 0.05
        if phase in {ThemePhase.OVEREXTENDED.value, ThemePhase.DISTRIBUTION.value}:
            confidence -= 0.15
        return cls.clamp(confidence, 0.0, 0.80)

    @classmethod
    def _risk_flags(cls, value: Any, *, phase: str) -> list[str]:
        flags: list[str] = []
        if isinstance(value, list):
            flags.extend(str(item) for item in value if str(item).strip())
        elif value:
            flags.append(str(value))
        if phase == ThemePhase.OVEREXTENDED.value:
            flags.append("theme_overextended_no_chase")
        if phase == ThemePhase.DISTRIBUTION.value:
            flags.append("theme_distribution_risk")
        return cls.dedupe_texts(flags)

    @staticmethod
    def _theme_value(theme_score: Any, key: str, default: Any) -> Any:
        if isinstance(theme_score, ThemeScore):
            return getattr(theme_score, key, default)
        if isinstance(theme_score, Mapping):
            return theme_score.get(key, default)
        return default

    @staticmethod
    def _float(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if number == number else default
