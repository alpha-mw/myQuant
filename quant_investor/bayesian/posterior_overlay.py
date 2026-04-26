"""Shadow calibrated posterior overlay and edge-after-cost diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.bayesian.calibration_v2 import (
    CALIBRATION_V2_SCHEMA_VERSION,
    TARGET_POSTERIOR_WIN_RATE,
    CalibrationModelV2,
    normalize_score_to_unit_interval,
)
from quant_investor.bayesian.types import PosteriorResult
from quant_investor.versioning import POSTERIOR_OVERLAY_SCHEMA_VERSION


DEFAULT_EXPECTED_ALPHA_SCALE = 0.10
DEFAULT_MAX_PROBABILITY_ADJUSTMENT = 0.20
DEFAULT_CALIBRATION_BLEND_WEIGHT = 1.0
OVERLAY_MODE_OFF = "off"
OVERLAY_MODE_SHADOW = "shadow"
OVERLAY_METADATA_KEY = "calibrated_posterior_overlay"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_json_safe(value))


def _finite_float(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


def _non_negative_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def clamp_probability(value: float) -> float:
    number = _finite_float(value, "probability")
    return max(0.0, min(1.0, number))


def bps_to_decimal_return(value_bps: float) -> float:
    bps = _non_negative_float(value_bps, "value_bps")
    return bps / 10000.0


def horizon_label_for_days(horizon_days: int) -> str:
    days = int(horizon_days)
    if days <= 0:
        raise ValueError("horizon_days must be positive.")
    return f"{days}D"


@dataclass
class EdgeCostConfig:
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    risk_capital_charge: float = 0.0
    include_existing_capacity_penalty: bool = True
    expected_alpha_scale: float = DEFAULT_EXPECTED_ALPHA_SCALE
    max_probability_adjustment: float = DEFAULT_MAX_PROBABILITY_ADJUSTMENT
    calibration_blend_weight: float = DEFAULT_CALIBRATION_BLEND_WEIGHT
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.transaction_cost_bps = _non_negative_float(self.transaction_cost_bps, "transaction_cost_bps")
        self.slippage_bps = _non_negative_float(self.slippage_bps, "slippage_bps")
        self.market_impact_bps = _non_negative_float(self.market_impact_bps, "market_impact_bps")
        self.risk_capital_charge = _non_negative_float(self.risk_capital_charge, "risk_capital_charge")
        self.expected_alpha_scale = _finite_float(self.expected_alpha_scale, "expected_alpha_scale")
        if self.expected_alpha_scale <= 0.0:
            raise ValueError("expected_alpha_scale must be positive.")
        self.max_probability_adjustment = _finite_float(
            self.max_probability_adjustment,
            "max_probability_adjustment",
        )
        if not 0.0 <= self.max_probability_adjustment <= 1.0:
            raise ValueError("max_probability_adjustment must be in [0, 1].")
        self.calibration_blend_weight = _finite_float(
            self.calibration_blend_weight,
            "calibration_blend_weight",
        )
        if not 0.0 <= self.calibration_blend_weight <= 1.0:
            raise ValueError("calibration_blend_weight must be in [0, 1].")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EdgeCostConfig":
        data = dict(payload)
        return cls(
            transaction_cost_bps=float(data.get("transaction_cost_bps", 0.0) or 0.0),
            slippage_bps=float(data.get("slippage_bps", 0.0) or 0.0),
            market_impact_bps=float(data.get("market_impact_bps", 0.0) or 0.0),
            risk_capital_charge=float(data.get("risk_capital_charge", 0.0) or 0.0),
            include_existing_capacity_penalty=bool(data.get("include_existing_capacity_penalty", True)),
            expected_alpha_scale=float(data.get("expected_alpha_scale", DEFAULT_EXPECTED_ALPHA_SCALE) or 0.0),
            max_probability_adjustment=float(
                data.get("max_probability_adjustment", DEFAULT_MAX_PROBABILITY_ADJUSTMENT)
            ),
            calibration_blend_weight=float(
                data.get("calibration_blend_weight", DEFAULT_CALIBRATION_BLEND_WEIGHT)
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class CalibrationOverlayDiagnostics:
    schema_version: str = POSTERIOR_OVERLAY_SCHEMA_VERSION
    overlay_mode: str = OVERLAY_MODE_SHADOW
    model_id: str = ""
    calibration_schema_version: str = CALIBRATION_V2_SCHEMA_VERSION
    target_name: str = TARGET_POSTERIOR_WIN_RATE
    selected_curve_key: dict[str, Any] | None = None
    selected_curve_examples: int | None = None
    raw_win_rate: float = 0.50
    model_calibrated_win_rate: float = 0.50
    blended_calibrated_win_rate: float = 0.50
    probability_delta_before_cap: float = 0.0
    probability_delta_after_cap: float = 0.0
    cap_applied: bool = False
    normalized_raw_value: float = 0.50
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationOverlayDiagnostics":
        data = dict(payload)
        selected_curve_key = data.get("selected_curve_key")
        return cls(
            schema_version=str(data.get("schema_version", POSTERIOR_OVERLAY_SCHEMA_VERSION)),
            overlay_mode=str(data.get("overlay_mode", OVERLAY_MODE_SHADOW)),
            model_id=str(data.get("model_id", "")),
            calibration_schema_version=str(data.get("calibration_schema_version", CALIBRATION_V2_SCHEMA_VERSION)),
            target_name=str(data.get("target_name", TARGET_POSTERIOR_WIN_RATE)),
            selected_curve_key=dict(selected_curve_key) if isinstance(selected_curve_key, Mapping) else None,
            selected_curve_examples=(
                int(data["selected_curve_examples"])
                if data.get("selected_curve_examples") is not None
                else None
            ),
            raw_win_rate=float(data.get("raw_win_rate", 0.50) or 0.0),
            model_calibrated_win_rate=float(data.get("model_calibrated_win_rate", 0.50) or 0.0),
            blended_calibrated_win_rate=float(data.get("blended_calibrated_win_rate", 0.50) or 0.0),
            probability_delta_before_cap=float(data.get("probability_delta_before_cap", 0.0) or 0.0),
            probability_delta_after_cap=float(data.get("probability_delta_after_cap", 0.0) or 0.0),
            cap_applied=bool(data.get("cap_applied", False)),
            normalized_raw_value=float(data.get("normalized_raw_value", 0.50) or 0.0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class EdgeAfterCostBreakdown:
    raw_expected_alpha: float = 0.0
    calibrated_expected_alpha: float = 0.0
    existing_capacity_penalty: float = 0.0
    transaction_cost: float = 0.0
    slippage_cost: float = 0.0
    market_impact_cost: float = 0.0
    risk_capital_charge: float = 0.0
    total_cost_penalty: float = 0.0
    raw_edge_after_costs: float = 0.0
    calibrated_edge_after_costs: float = 0.0
    edge_delta: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EdgeAfterCostBreakdown":
        data = dict(payload)
        return cls(
            raw_expected_alpha=float(data.get("raw_expected_alpha", 0.0) or 0.0),
            calibrated_expected_alpha=float(data.get("calibrated_expected_alpha", 0.0) or 0.0),
            existing_capacity_penalty=float(data.get("existing_capacity_penalty", 0.0) or 0.0),
            transaction_cost=float(data.get("transaction_cost", 0.0) or 0.0),
            slippage_cost=float(data.get("slippage_cost", 0.0) or 0.0),
            market_impact_cost=float(data.get("market_impact_cost", 0.0) or 0.0),
            risk_capital_charge=float(data.get("risk_capital_charge", 0.0) or 0.0),
            total_cost_penalty=float(data.get("total_cost_penalty", 0.0) or 0.0),
            raw_edge_after_costs=float(data.get("raw_edge_after_costs", 0.0) or 0.0),
            calibrated_edge_after_costs=float(data.get("calibrated_edge_after_costs", 0.0) or 0.0),
            edge_delta=float(data.get("edge_delta", 0.0) or 0.0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class CalibratedPosteriorOverlay:
    schema_version: str = POSTERIOR_OVERLAY_SCHEMA_VERSION
    symbol: str = ""
    company_name: str = ""
    market: str = ""
    horizon_days: int = 20
    horizon_label: str = ""
    macro_regime: str = "未知"
    original_posterior_win_rate: float = 0.50
    calibrated_posterior_win_rate: float = 0.50
    original_posterior_expected_alpha: float = 0.0
    calibrated_posterior_expected_alpha: float = 0.0
    original_posterior_action_score: float = 0.0
    calibrated_posterior_action_score: float = 0.0
    original_edge_after_costs: float = 0.0
    calibrated_edge_after_costs: float = 0.0
    action_threshold_used: float = 0.0
    positive_edge: bool = False
    diagnostics: CalibrationOverlayDiagnostics = field(default_factory=CalibrationOverlayDiagnostics)
    edge_breakdown: EdgeAfterCostBreakdown = field(default_factory=EdgeAfterCostBreakdown)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibratedPosteriorOverlay":
        data = dict(payload)
        diagnostics_payload = data.get("diagnostics", {}) or {}
        edge_payload = data.get("edge_breakdown", {}) or {}
        return cls(
            schema_version=str(data.get("schema_version", POSTERIOR_OVERLAY_SCHEMA_VERSION)),
            symbol=str(data.get("symbol", "")),
            company_name=str(data.get("company_name", "")),
            market=str(data.get("market", "")),
            horizon_days=int(data.get("horizon_days", 20) or 20),
            horizon_label=str(data.get("horizon_label", "")),
            macro_regime=str(data.get("macro_regime", "未知")),
            original_posterior_win_rate=float(data.get("original_posterior_win_rate", 0.50) or 0.0),
            calibrated_posterior_win_rate=float(data.get("calibrated_posterior_win_rate", 0.50) or 0.0),
            original_posterior_expected_alpha=float(data.get("original_posterior_expected_alpha", 0.0) or 0.0),
            calibrated_posterior_expected_alpha=float(
                data.get("calibrated_posterior_expected_alpha", 0.0) or 0.0
            ),
            original_posterior_action_score=float(data.get("original_posterior_action_score", 0.0) or 0.0),
            calibrated_posterior_action_score=float(data.get("calibrated_posterior_action_score", 0.0) or 0.0),
            original_edge_after_costs=float(data.get("original_edge_after_costs", 0.0) or 0.0),
            calibrated_edge_after_costs=float(data.get("calibrated_edge_after_costs", 0.0) or 0.0),
            action_threshold_used=float(data.get("action_threshold_used", 0.0) or 0.0),
            positive_edge=bool(data.get("positive_edge", False)),
            diagnostics=CalibrationOverlayDiagnostics.from_dict(
                diagnostics_payload if isinstance(diagnostics_payload, Mapping) else {}
            ),
            edge_breakdown=EdgeAfterCostBreakdown.from_dict(
                edge_payload if isinstance(edge_payload, Mapping) else {}
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def _cap_probability_delta(raw_win_rate: float, blended_win_rate: float, max_adjustment: float) -> tuple[float, float, bool]:
    delta = blended_win_rate - raw_win_rate
    capped_delta = max(-max_adjustment, min(max_adjustment, delta))
    return capped_delta, raw_win_rate + capped_delta, abs(capped_delta - delta) > 1e-15


def build_calibrated_posterior_overlay(
    result: PosteriorResult,
    model: CalibrationModelV2,
    *,
    market: str = "",
    horizon_days: int = 20,
    horizon_label: str | None = None,
    macro_regime: str = "未知",
    edge_cost_config: EdgeCostConfig | None = None,
    overlay_mode: str = OVERLAY_MODE_SHADOW,
    metadata: Mapping[str, Any] | None = None,
) -> CalibratedPosteriorOverlay:
    config = edge_cost_config or EdgeCostConfig()
    resolved_horizon_label = horizon_label or horizon_label_for_days(horizon_days)
    raw_win_rate = clamp_probability(result.posterior_win_rate)
    normalized_raw_value = normalize_score_to_unit_interval(raw_win_rate)
    selected_curve = model.select_curve(
        TARGET_POSTERIOR_WIN_RATE,
        market=market,
        horizon_label=resolved_horizon_label,
        macro_regime=macro_regime,
    )
    model_calibrated_win_rate = clamp_probability(
        model.calibrate(
            TARGET_POSTERIOR_WIN_RATE,
            raw_win_rate,
            market=market,
            horizon_label=resolved_horizon_label,
            macro_regime=macro_regime,
        )
    )
    blended_win_rate = clamp_probability(
        raw_win_rate * (1.0 - config.calibration_blend_weight)
        + model_calibrated_win_rate * config.calibration_blend_weight
    )
    delta_before_cap = blended_win_rate - raw_win_rate
    capped_delta, calibrated_win_rate, cap_applied = _cap_probability_delta(
        raw_win_rate,
        blended_win_rate,
        config.max_probability_adjustment,
    )
    calibrated_win_rate = clamp_probability(calibrated_win_rate)
    calibrated_expected_alpha = (calibrated_win_rate - 0.50) * config.expected_alpha_scale
    existing_capacity_penalty = (
        _non_negative_float(result.posterior_capacity_penalty, "posterior_capacity_penalty")
        if config.include_existing_capacity_penalty
        else 0.0
    )
    transaction_cost = bps_to_decimal_return(config.transaction_cost_bps)
    slippage_cost = bps_to_decimal_return(config.slippage_bps)
    market_impact_cost = bps_to_decimal_return(config.market_impact_bps)
    risk_capital_charge = config.risk_capital_charge
    total_cost_penalty = (
        existing_capacity_penalty
        + transaction_cost
        + slippage_cost
        + market_impact_cost
        + risk_capital_charge
    )
    raw_expected_alpha = _finite_float(result.posterior_expected_alpha, "posterior_expected_alpha")
    raw_edge_after_costs = _finite_float(result.posterior_edge_after_costs, "posterior_edge_after_costs")
    calibrated_edge_after_costs = calibrated_expected_alpha - total_cost_penalty
    edge_delta = calibrated_edge_after_costs - raw_edge_after_costs
    calibrated_action_score = (
        calibrated_win_rate * 0.60
        + _finite_float(result.posterior_confidence, "posterior_confidence") * 0.25
        + max(0.0, calibrated_expected_alpha) * 5.0 * 0.15
    )

    diagnostics = CalibrationOverlayDiagnostics(
        schema_version=POSTERIOR_OVERLAY_SCHEMA_VERSION,
        overlay_mode=overlay_mode,
        model_id=model.model_id,
        calibration_schema_version=model.schema_version or CALIBRATION_V2_SCHEMA_VERSION,
        target_name=TARGET_POSTERIOR_WIN_RATE,
        selected_curve_key=selected_curve.key.to_dict() if selected_curve is not None else None,
        selected_curve_examples=selected_curve.total_examples if selected_curve is not None else None,
        raw_win_rate=raw_win_rate,
        model_calibrated_win_rate=model_calibrated_win_rate,
        blended_calibrated_win_rate=blended_win_rate,
        probability_delta_before_cap=delta_before_cap,
        probability_delta_after_cap=capped_delta,
        cap_applied=cap_applied,
        normalized_raw_value=normalized_raw_value,
        metadata={},
    )
    edge_breakdown = EdgeAfterCostBreakdown(
        raw_expected_alpha=raw_expected_alpha,
        calibrated_expected_alpha=calibrated_expected_alpha,
        existing_capacity_penalty=existing_capacity_penalty,
        transaction_cost=transaction_cost,
        slippage_cost=slippage_cost,
        market_impact_cost=market_impact_cost,
        risk_capital_charge=risk_capital_charge,
        total_cost_penalty=total_cost_penalty,
        raw_edge_after_costs=raw_edge_after_costs,
        calibrated_edge_after_costs=calibrated_edge_after_costs,
        edge_delta=edge_delta,
        metadata={},
    )
    overlay_metadata = {
        "posterior_result_metadata": _coerce_metadata(result.metadata),
        "input_metadata": _coerce_metadata(metadata),
        "posterior_overlay_schema_version": POSTERIOR_OVERLAY_SCHEMA_VERSION,
        "calibration_model_id": model.model_id,
        "calibration_schema_version": model.schema_version or CALIBRATION_V2_SCHEMA_VERSION,
        "edge_cost_config": config.to_dict(),
    }
    return CalibratedPosteriorOverlay(
        schema_version=POSTERIOR_OVERLAY_SCHEMA_VERSION,
        symbol=result.symbol,
        company_name=result.company_name,
        market=market,
        horizon_days=int(horizon_days),
        horizon_label=resolved_horizon_label,
        macro_regime=macro_regime,
        original_posterior_win_rate=raw_win_rate,
        calibrated_posterior_win_rate=calibrated_win_rate,
        original_posterior_expected_alpha=raw_expected_alpha,
        calibrated_posterior_expected_alpha=calibrated_expected_alpha,
        original_posterior_action_score=_finite_float(result.posterior_action_score, "posterior_action_score"),
        calibrated_posterior_action_score=calibrated_action_score,
        original_edge_after_costs=raw_edge_after_costs,
        calibrated_edge_after_costs=calibrated_edge_after_costs,
        action_threshold_used=_finite_float(result.action_threshold_used, "action_threshold_used"),
        positive_edge=calibrated_edge_after_costs > 0.0,
        diagnostics=diagnostics,
        edge_breakdown=edge_breakdown,
        metadata=overlay_metadata,
    )


def build_calibrated_posterior_overlays(
    results: Sequence[PosteriorResult],
    model: CalibrationModelV2,
    *,
    market: str = "",
    horizon_days: int = 20,
    horizon_label: str | None = None,
    macro_regime: str = "未知",
    edge_cost_config: EdgeCostConfig | None = None,
    overlay_mode: str = OVERLAY_MODE_SHADOW,
    metadata: Mapping[str, Any] | None = None,
) -> list[CalibratedPosteriorOverlay]:
    resolved_horizon_label = horizon_label or horizon_label_for_days(horizon_days)
    seen: set[tuple[str, str]] = set()
    overlays: list[CalibratedPosteriorOverlay] = []
    for result in results:
        key = (result.symbol, resolved_horizon_label)
        if key in seen:
            raise ValueError(f"Duplicate posterior overlay input for symbol+horizon: {key!r}.")
        seen.add(key)
        overlays.append(
            build_calibrated_posterior_overlay(
                result,
                model,
                market=market,
                horizon_days=horizon_days,
                horizon_label=resolved_horizon_label,
                macro_regime=macro_regime,
                edge_cost_config=edge_cost_config,
                overlay_mode=overlay_mode,
                metadata=metadata,
            )
        )
    return overlays


def attach_overlay_metadata(
    result: PosteriorResult,
    overlay: CalibratedPosteriorOverlay,
    *,
    mutate: bool = False,
) -> PosteriorResult:
    overlay_payload = overlay.to_dict()
    if mutate:
        result.metadata[OVERLAY_METADATA_KEY] = overlay_payload
        return result
    copied_metadata = dict(_json_safe(result.metadata))
    copied_metadata[OVERLAY_METADATA_KEY] = overlay_payload
    return replace(result, metadata=copied_metadata)


__all__ = [
    "DEFAULT_EXPECTED_ALPHA_SCALE",
    "DEFAULT_MAX_PROBABILITY_ADJUSTMENT",
    "DEFAULT_CALIBRATION_BLEND_WEIGHT",
    "OVERLAY_MODE_OFF",
    "OVERLAY_MODE_SHADOW",
    "OVERLAY_METADATA_KEY",
    "EdgeCostConfig",
    "CalibrationOverlayDiagnostics",
    "EdgeAfterCostBreakdown",
    "CalibratedPosteriorOverlay",
    "clamp_probability",
    "bps_to_decimal_return",
    "horizon_label_for_days",
    "build_calibrated_posterior_overlay",
    "build_calibrated_posterior_overlays",
    "attach_overlay_metadata",
]
