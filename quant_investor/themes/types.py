from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Any


class ThemePhase(str, Enum):
    UNCLASSIFIED = "unclassified"
    ACCUMULATION = "accumulation"
    EARLY_ACCELERATION = "early_acceleration"
    CONFIRMED_ROTATION = "confirmed_rotation"
    OVEREXTENDED = "overextended"
    DISTRIBUTION = "distribution"


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return lower
    if not math.isfinite(numeric):
        return lower
    return max(lower, min(upper, numeric))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else 0.0
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


@dataclass
class ThemeScore:
    theme_id: str
    theme_name: str
    phase: ThemePhase = ThemePhase.UNCLASSIFIED
    score: float = 0.0
    confidence: float = 0.0
    member_count: int = 0
    breadth: float = 0.0
    momentum: float = 0.0
    acceleration: float = 0.0
    volume_confirmation: float = 0.0
    overextension_risk: float = 0.0
    fake_breakout_risk: float = 0.0
    raw_score: float | None = None
    smoothed_score: float | None = None
    heat_10d: float | None = None
    heat_delta_5d: float | None = None
    persistence_count: int = 0
    trend_state: str = "insufficient_history"
    smoothing_observation_count: int = 0
    smoothing_status: str = "insufficient_history"
    policy_catalyst_score: float = 0.0
    policy_confidence: float = 0.0
    policy_stage: str = "disabled"
    policy_evidence: list[str] = field(default_factory=list)
    policy_risk_flags: list[str] = field(default_factory=list)
    theme_turnover_share: float = 0.0
    turnover_share_sma10: float | None = None
    turnover_share_stretch: float = 0.0
    turnover_share_delta_5d: float | None = None
    turnover_share_trend: str = "disabled"
    theme_limitup_ratio: float = 0.0
    limitup_norm: float = 0.0
    member_turnover_concentration: float = 0.0
    crowding_risk: float = 0.0
    crowding_status: str = "disabled"
    crowding_diagnostic_notes: list[str] = field(default_factory=list)
    top_symbols: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": str(self.theme_id),
            "theme_name": str(self.theme_name),
            "phase": self.phase.value,
            "score": _jsonable(self.score),
            "confidence": _jsonable(self.confidence),
            "member_count": int(self.member_count),
            "breadth": _jsonable(self.breadth),
            "momentum": _jsonable(self.momentum),
            "acceleration": _jsonable(self.acceleration),
            "volume_confirmation": _jsonable(self.volume_confirmation),
            "overextension_risk": _jsonable(self.overextension_risk),
            "fake_breakout_risk": _jsonable(self.fake_breakout_risk),
            "raw_score": _jsonable(self.raw_score if self.raw_score is not None else self.score),
            "smoothed_score": _jsonable(self.smoothed_score),
            "heat_10d": _jsonable(self.heat_10d),
            "heat_delta_5d": _jsonable(self.heat_delta_5d),
            "persistence_count": int(self.persistence_count),
            "trend_state": str(self.trend_state or "insufficient_history"),
            "smoothing_observation_count": int(self.smoothing_observation_count),
            "smoothing_status": str(self.smoothing_status or "insufficient_history"),
            "policy_catalyst_score": _jsonable(self.policy_catalyst_score),
            "policy_confidence": _jsonable(self.policy_confidence),
            "policy_stage": str(self.policy_stage or "disabled"),
            "policy_evidence": _jsonable(self.policy_evidence),
            "policy_risk_flags": _jsonable(self.policy_risk_flags),
            "theme_turnover_share": _jsonable(self.theme_turnover_share),
            "turnover_share_sma10": _jsonable(self.turnover_share_sma10),
            "turnover_share_stretch": _jsonable(self.turnover_share_stretch),
            "turnover_share_delta_5d": _jsonable(self.turnover_share_delta_5d),
            "turnover_share_trend": str(self.turnover_share_trend or "disabled"),
            "theme_limitup_ratio": _jsonable(self.theme_limitup_ratio),
            "limitup_norm": _jsonable(self.limitup_norm),
            "member_turnover_concentration": _jsonable(self.member_turnover_concentration),
            "crowding_risk": _jsonable(self.crowding_risk),
            "crowding_status": str(self.crowding_status or "disabled"),
            "crowding_diagnostic_notes": _jsonable(self.crowding_diagnostic_notes),
            "top_symbols": _jsonable(self.top_symbols),
            "risk_flags": _jsonable(self.risk_flags),
            "evidence": _jsonable(self.evidence),
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class ThemeScanResult:
    market: str = "CN"
    universe_key: str = ""
    as_of: str = ""
    schema_version: str = "theme_rotation.v1"
    theme_scores: dict[str, ThemeScore] = field(default_factory=dict)
    symbol_scores: dict[str, float] = field(default_factory=dict)
    symbol_smoothed_scores: dict[str, float] = field(default_factory=dict)
    symbol_primary_theme: dict[str, str] = field(default_factory=dict)
    symbol_phase: dict[str, str] = field(default_factory=dict)
    symbol_risk_flags: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": str(self.market),
            "universe_key": str(self.universe_key),
            "as_of": str(self.as_of),
            "schema_version": str(self.schema_version),
            "theme_scores": {
                str(theme_id): score.to_dict()
                for theme_id, score in self.theme_scores.items()
            },
            "symbol_scores": _jsonable(self.symbol_scores),
            "symbol_smoothed_scores": _jsonable(self.symbol_smoothed_scores),
            "symbol_primary_theme": _jsonable(self.symbol_primary_theme),
            "symbol_phase": _jsonable(self.symbol_phase),
            "symbol_risk_flags": _jsonable(self.symbol_risk_flags),
            "metadata": _jsonable(self.metadata),
        }
