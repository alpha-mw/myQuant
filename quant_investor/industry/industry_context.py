"""Immutable industry context used by forward research scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

_SCORE_FIELDS = (
    "demand_score",
    "supply_score",
    "inventory_score",
    "pricing_power",
    "capex_score",
    "earnings_revision",
    "policy_score",
    "market_confirmation",
    "narrative_strength",
    "crowding_risk",
    "confidence",
)


def _validate_unit_interval(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _normalize_text_tuple(value: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of strings")
    normalized = tuple(value)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ValueError(f"{label} must contain only non-empty strings")
    return normalized


@dataclass(frozen=True)
class IndustryContext:
    """All required evidence dimensions for one industry score."""

    industry_id: str
    cycle_stage: str
    demand_score: float
    supply_score: float
    inventory_score: float
    pricing_power: float
    capex_score: float
    earnings_revision: float
    policy_score: float
    market_confirmation: float
    narrative_strength: float
    crowding_risk: float
    catalysts: Sequence[str]
    contrary_evidence: Sequence[str]
    confidence: float
    evidence_refs: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.industry_id, str) or not self.industry_id:
            raise ValueError("industry_id must be a non-empty string")
        if not isinstance(self.cycle_stage, str) or not self.cycle_stage:
            raise ValueError("cycle_stage must be a non-empty string")
        for field_name in _SCORE_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _validate_unit_interval(getattr(self, field_name), label=field_name),
            )
        object.__setattr__(
            self,
            "catalysts",
            _normalize_text_tuple(self.catalysts, label="catalysts"),
        )
        object.__setattr__(
            self,
            "contrary_evidence",
            _normalize_text_tuple(self.contrary_evidence, label="contrary_evidence"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _normalize_text_tuple(self.evidence_refs, label="evidence_refs"),
        )


__all__ = ["IndustryContext"]
