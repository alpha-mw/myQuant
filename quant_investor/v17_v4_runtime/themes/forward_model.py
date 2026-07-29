"""V17 v4 deterministic theme-exposure scoring with explicit evidence strength."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import cast, Sequence


class ThemeExposureType(str, Enum):
    DIRECT_BENEFICIARY = "DIRECT_BENEFICIARY"
    DIRECT = "DIRECT_BENEFICIARY"
    SUPPLIER = "SUPPLIER"
    SECOND_ORDER = "SECOND_ORDER"
    CONCEPT_ONLY = "CONCEPT_ONLY"


def _optional_unit_interval(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in [0, 1] or None")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _required_unit_interval(value: object, *, label: str) -> float:
    normalized = _optional_unit_interval(value, label=label)
    if normalized is None:
        raise ValueError(f"{label} is required")
    return normalized


@dataclass(frozen=True)
class ThemeExposure:
    """Exact caller-supplied inputs for one symbol/theme relationship."""

    symbol: str
    theme_id: str
    exposure_type: ThemeExposureType | str
    revenue_exposure: float | None
    product_exposure: float | None
    customer_exposure: float | None
    supply_chain_position: str
    confidence: float
    evidence_refs: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.theme_id, str) or not self.theme_id:
            raise ValueError("theme_id must be a non-empty string")
        try:
            exposure_type = ThemeExposureType(self.exposure_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported exposure_type") from exc
        object.__setattr__(self, "exposure_type", exposure_type)
        for field_name in (
            "revenue_exposure",
            "product_exposure",
            "customer_exposure",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_unit_interval(getattr(self, field_name), label=field_name),
            )
        if (
            self.revenue_exposure is None
            and self.product_exposure is None
            and self.customer_exposure is None
        ):
            raise ValueError("at least one revenue, product, or customer exposure is required")
        if not isinstance(self.supply_chain_position, str) or not self.supply_chain_position:
            raise ValueError("supply_chain_position must be a non-empty string")
        object.__setattr__(
            self,
            "confidence",
            _required_unit_interval(self.confidence, label="confidence"),
        )
        if isinstance(self.evidence_refs, (str, bytes)):
            raise ValueError("evidence_refs must be a sequence of strings")
        refs = tuple(self.evidence_refs)
        if not refs or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError("evidence_refs must contain at least one non-empty string")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class ThemeExposureScore:
    symbol: str
    theme_id: str
    exposure_type: ThemeExposureType
    base_score: float
    confidence: float
    type_weight: float
    score: float


def _type_weight(exposure: ThemeExposure) -> float:
    if exposure.exposure_type is ThemeExposureType.DIRECT_BENEFICIARY:
        return 1.0
    if exposure.exposure_type is ThemeExposureType.SUPPLIER and exposure.confidence >= 0.80:
        return 0.70
    return 0.0


def score_theme_exposure(exposure: ThemeExposure) -> ThemeExposureScore:
    """Score available revenue/product/customer evidence by exact type weight."""

    if not isinstance(exposure, ThemeExposure):
        raise ValueError("exposure must be ThemeExposure")
    components = tuple(
        value
        for value in (
            exposure.revenue_exposure,
            exposure.product_exposure,
            exposure.customer_exposure,
        )
        if value is not None
    )
    base_score = sum(components) / len(components)
    weight = _type_weight(exposure)
    return ThemeExposureScore(
        symbol=exposure.symbol,
        theme_id=exposure.theme_id,
        exposure_type=cast(ThemeExposureType, exposure.exposure_type),
        base_score=base_score,
        confidence=exposure.confidence,
        type_weight=weight,
        score=base_score * exposure.confidence * weight,
    )


theme_score = score_theme_exposure


__all__ = [
    "ThemeExposure",
    "ThemeExposureScore",
    "ThemeExposureType",
    "score_theme_exposure",
    "theme_score",
]
