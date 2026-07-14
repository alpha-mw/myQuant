"""Shadow calibrated posterior overlay and edge-after-cost diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import (
    asdict,
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
OVERLAY_CUTOFF_PROOF_SCHEMA_VERSION = "2026-07-14.overlay-cutoff-proof.v1"
_POSTERIOR_SOURCE_DOMAIN = "myquant.posterior-result-source.v2"
_CALIBRATION_MODEL_DOMAIN = "myquant.calibration-model.v2"
_OVERLAY_CUTOFF_PROOF_DOMAIN = "myquant.overlay-cutoff-proof.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RESERVED_OVERLAY_METADATA_KEYS = {
    "schema_version",
    "overlay_mode",
    "report_only",
    "production_eligible",
    "production_weight",
    "decision_as_of",
    "source_sha256",
    "model_id",
    "model_sha256",
    "model_trained_at",
    "cutoff_proof",
    "cutoff_proof_sha256",
    "proof_sha256",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            normalized_key = str(key)
            if normalized_key in result:
                raise ValueError(
                    f"canonical JSON key collision: {normalized_key!r}"
                )
            result[normalized_key] = _json_safe(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_safe(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _canonical_datetime_text(value, "datetime")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not allow non-finite numbers")
    return value


def _canonical_datetime_text(value: datetime, field_name: str) -> str:
    if type(value) is not datetime:
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_aware_timestamp(value: Any, field_name: str) -> datetime:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _canonical_json_bytes(domain: str, value: Any) -> bytes:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    payload = {"domain": domain, "payload": _json_safe(value)}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical-JSON serializable") from exc
    return encoded.encode("utf-8")


def _domain_sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(domain, value)).hexdigest()


def posterior_result_source_sha256(result: PosteriorResult) -> str:
    """Hash every recursive dataclass field of one posterior result."""

    if type(result) is not PosteriorResult:
        raise TypeError("result must be an exact PosteriorResult")
    return _domain_sha256(_POSTERIOR_SOURCE_DOMAIN, asdict(result))


def calibration_model_sha256(model: CalibrationModelV2) -> str:
    if type(model) is not CalibrationModelV2:
        raise TypeError("loaded model must be an exact CalibrationModelV2")
    return _domain_sha256(_CALIBRATION_MODEL_DOMAIN, asdict(model))


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError(f"{label} must be an exact JSON object")
    data = dict(payload)
    if set(data) != expected:
        missing = sorted(expected - set(data))
        unknown = sorted(set(data) - expected)
        raise ValueError(
            f"{label} fields invalid; missing={missing!r}, unknown={unknown!r}"
        )
    return data


def _require_string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    return _finite_float(value, field_name)


def _require_float(value: Any, field_name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field_name} must be a float")
    return _finite_float(value, field_name)


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_PATTERN.fullmatch(value) is not None


def _reject_reserved_metadata(value: Any, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in _RESERVED_OVERLAY_METADATA_KEYS:
                raise ValueError(
                    f"reserved overlay metadata key forbidden: {path}.{normalized_key}"
                )
            _reject_reserved_metadata(item, f"{path}.{normalized_key}")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _reject_reserved_metadata(item, f"{path}[{index}]")


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


@dataclass(frozen=True, slots=True)
class OverlayCutoffProof:
    """External cutoff and model-integrity envelope; not an authenticity proof."""

    schema_version: str
    decision_as_of: str
    training_input_cutoff: str
    outcome_resolution_cutoff: str
    training_examples_sha256: str
    resolved_outcomes_sha256: str
    outcome_ledger_sha256: str
    model_id: str
    model_sha256: str
    model_trained_at: str
    proof_sha256: str

    def __post_init__(self) -> None:
        _validate_overlay_cutoff_proof(
            self,
            expected_decision_as_of=self.decision_as_of,
        )

    def to_dict(self) -> dict[str, Any]:
        _validate_overlay_cutoff_proof(
            self,
            expected_decision_as_of=self.decision_as_of,
        )
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OverlayCutoffProof":
        expected = {field.name for field in dataclass_fields(cls)}
        data = _require_exact_fields(payload, expected, "OverlayCutoffProof")
        proof = cls(
            schema_version=_require_string(
                data["schema_version"],
                "schema_version",
            ),
            decision_as_of=_require_string(
                data["decision_as_of"],
                "decision_as_of",
            ),
            training_input_cutoff=_require_string(
                data["training_input_cutoff"],
                "training_input_cutoff",
            ),
            outcome_resolution_cutoff=_require_string(
                data["outcome_resolution_cutoff"],
                "outcome_resolution_cutoff",
            ),
            training_examples_sha256=_require_string(
                data["training_examples_sha256"],
                "training_examples_sha256",
            ),
            resolved_outcomes_sha256=_require_string(
                data["resolved_outcomes_sha256"],
                "resolved_outcomes_sha256",
            ),
            outcome_ledger_sha256=_require_string(
                data["outcome_ledger_sha256"],
                "outcome_ledger_sha256",
            ),
            model_id=_require_string(data["model_id"], "model_id"),
            model_sha256=_require_string(
                data["model_sha256"],
                "model_sha256",
            ),
            model_trained_at=_require_string(
                data["model_trained_at"],
                "model_trained_at",
            ),
            proof_sha256=_require_string(
                data["proof_sha256"],
                "proof_sha256",
            ),
        )
        _validate_overlay_cutoff_proof(
            proof,
            expected_decision_as_of=proof.decision_as_of,
        )
        return proof


def overlay_cutoff_proof_sha256(
    proof: OverlayCutoffProof | Mapping[str, Any],
) -> str:
    if type(proof) is OverlayCutoffProof:
        payload = asdict(proof)
    elif type(proof) is dict:
        payload = dict(proof)
    else:
        raise TypeError("proof must be an OverlayCutoffProof or exact JSON object")
    payload.pop("proof_sha256", None)
    return _domain_sha256(_OVERLAY_CUTOFF_PROOF_DOMAIN, payload)


def _validate_overlay_cutoff_proof(
    proof: OverlayCutoffProof,
    *,
    expected_decision_as_of: str,
) -> None:
    if type(proof) is not OverlayCutoffProof:
        raise TypeError("cutoff proof must be an exact OverlayCutoffProof")
    if proof.schema_version != OVERLAY_CUTOFF_PROOF_SCHEMA_VERSION:
        raise ValueError("overlay cutoff proof schema invalid")
    if proof.decision_as_of != expected_decision_as_of:
        raise ValueError("overlay cutoff proof decision_as_of mismatch")
    for field_name in (
        "training_examples_sha256",
        "resolved_outcomes_sha256",
        "outcome_ledger_sha256",
        "model_sha256",
        "proof_sha256",
    ):
        if not _is_sha256(getattr(proof, field_name)):
            raise ValueError(f"overlay cutoff proof {field_name} invalid")
    _require_string(proof.model_id, "model_id")
    outcome_resolution = _parse_aware_timestamp(
        proof.outcome_resolution_cutoff,
        "outcome_resolution_cutoff",
    )
    training_input = _parse_aware_timestamp(
        proof.training_input_cutoff,
        "training_input_cutoff",
    )
    trained_at = _parse_aware_timestamp(
        proof.model_trained_at,
        "model_trained_at",
    )
    decision = _parse_aware_timestamp(
        proof.decision_as_of,
        "decision_as_of",
    )
    if not outcome_resolution <= training_input <= trained_at <= decision:
        raise ValueError(
            "overlay cutoff order must be outcome_resolution <= training_input "
            "<= model_trained_at <= decision_as_of"
        )
    if overlay_cutoff_proof_sha256(proof) != proof.proof_sha256:
        raise ValueError("overlay cutoff proof sha256 mismatch")


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


def _strict_metadata(value: Any, field_name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{field_name} must be an exact JSON object")
    _reject_reserved_metadata(value, field_name)
    normalized = _json_safe(value)
    if type(normalized) is not dict:
        raise TypeError(f"{field_name} must be an exact JSON object")
    _canonical_json_bytes("myquant.overlay-metadata.v2", normalized)
    return normalized


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        _validate_overlay_diagnostics(self)

    def to_dict(self) -> dict[str, Any]:
        _validate_overlay_diagnostics(self)
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationOverlayDiagnostics":
        expected = {field.name for field in dataclass_fields(cls)}
        data = _require_exact_fields(
            payload,
            expected,
            "CalibrationOverlayDiagnostics",
        )
        selected_curve_key = data["selected_curve_key"]
        if selected_curve_key is not None and type(selected_curve_key) is not dict:
            raise TypeError("selected_curve_key must be an exact JSON object or null")
        selected_curve_examples = data["selected_curve_examples"]
        if selected_curve_examples is not None:
            selected_curve_examples = _require_int(
                selected_curve_examples,
                "selected_curve_examples",
            )
        diagnostics = cls(
            schema_version=_require_string(data["schema_version"], "schema_version"),
            overlay_mode=_require_string(data["overlay_mode"], "overlay_mode"),
            model_id=_require_string(data["model_id"], "model_id"),
            calibration_schema_version=_require_string(
                data["calibration_schema_version"],
                "calibration_schema_version",
            ),
            target_name=_require_string(data["target_name"], "target_name"),
            selected_curve_key=(
                dict(_json_safe(selected_curve_key))
                if selected_curve_key is not None
                else None
            ),
            selected_curve_examples=(
                selected_curve_examples
            ),
            raw_win_rate=_require_number(data["raw_win_rate"], "raw_win_rate"),
            model_calibrated_win_rate=_require_number(
                data["model_calibrated_win_rate"],
                "model_calibrated_win_rate",
            ),
            blended_calibrated_win_rate=_require_number(
                data["blended_calibrated_win_rate"],
                "blended_calibrated_win_rate",
            ),
            probability_delta_before_cap=_require_number(
                data["probability_delta_before_cap"],
                "probability_delta_before_cap",
            ),
            probability_delta_after_cap=_require_number(
                data["probability_delta_after_cap"],
                "probability_delta_after_cap",
            ),
            cap_applied=_require_bool(data["cap_applied"], "cap_applied"),
            normalized_raw_value=_require_number(
                data["normalized_raw_value"],
                "normalized_raw_value",
            ),
            metadata=_strict_metadata(data["metadata"], "diagnostics.metadata"),
        )
        _validate_overlay_diagnostics(diagnostics)
        return diagnostics


def _validate_overlay_diagnostics(
    diagnostics: CalibrationOverlayDiagnostics,
) -> None:
    if type(diagnostics) is not CalibrationOverlayDiagnostics:
        raise TypeError("diagnostics must be exact CalibrationOverlayDiagnostics")
    if diagnostics.schema_version != POSTERIOR_OVERLAY_SCHEMA_VERSION:
        raise ValueError("diagnostics schema_version invalid")
    if diagnostics.overlay_mode != OVERLAY_MODE_SHADOW:
        raise ValueError("diagnostics overlay_mode must be exact shadow")
    _require_string(diagnostics.model_id, "diagnostics.model_id")
    _require_string(
        diagnostics.calibration_schema_version,
        "diagnostics.calibration_schema_version",
    )
    if diagnostics.target_name != TARGET_POSTERIOR_WIN_RATE:
        raise ValueError("diagnostics target_name invalid")
    if (
        diagnostics.selected_curve_key is not None
        and type(diagnostics.selected_curve_key) is not dict
    ):
        raise TypeError("diagnostics selected_curve_key invalid")
    if diagnostics.selected_curve_key is not None:
        _canonical_json_bytes(
            "myquant.overlay-selected-curve.v2",
            diagnostics.selected_curve_key,
        )
    if diagnostics.selected_curve_examples is not None:
        if (
            type(diagnostics.selected_curve_examples) is not int
            or diagnostics.selected_curve_examples < 0
        ):
            raise ValueError("diagnostics selected_curve_examples invalid")
    for field_name in (
        "raw_win_rate",
        "model_calibrated_win_rate",
        "blended_calibrated_win_rate",
        "normalized_raw_value",
    ):
        value = _require_number(
            getattr(diagnostics, field_name),
            f"diagnostics.{field_name}",
        )
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"diagnostics.{field_name} must be in [0, 1]")
    _require_number(
        diagnostics.probability_delta_before_cap,
        "diagnostics.probability_delta_before_cap",
    )
    _require_number(
        diagnostics.probability_delta_after_cap,
        "diagnostics.probability_delta_after_cap",
    )
    _require_bool(diagnostics.cap_applied, "diagnostics.cap_applied")
    _strict_metadata(diagnostics.metadata, "diagnostics.metadata")


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        _validate_edge_breakdown(self)

    def to_dict(self) -> dict[str, Any]:
        _validate_edge_breakdown(self)
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EdgeAfterCostBreakdown":
        expected = {field.name for field in dataclass_fields(cls)}
        data = _require_exact_fields(payload, expected, "EdgeAfterCostBreakdown")
        breakdown = cls(
            raw_expected_alpha=_require_number(
                data["raw_expected_alpha"],
                "raw_expected_alpha",
            ),
            calibrated_expected_alpha=_require_number(
                data["calibrated_expected_alpha"],
                "calibrated_expected_alpha",
            ),
            existing_capacity_penalty=_require_number(
                data["existing_capacity_penalty"],
                "existing_capacity_penalty",
            ),
            transaction_cost=_require_number(
                data["transaction_cost"],
                "transaction_cost",
            ),
            slippage_cost=_require_number(data["slippage_cost"], "slippage_cost"),
            market_impact_cost=_require_number(
                data["market_impact_cost"],
                "market_impact_cost",
            ),
            risk_capital_charge=_require_number(
                data["risk_capital_charge"],
                "risk_capital_charge",
            ),
            total_cost_penalty=_require_number(
                data["total_cost_penalty"],
                "total_cost_penalty",
            ),
            raw_edge_after_costs=_require_number(
                data["raw_edge_after_costs"],
                "raw_edge_after_costs",
            ),
            calibrated_edge_after_costs=_require_number(
                data["calibrated_edge_after_costs"],
                "calibrated_edge_after_costs",
            ),
            edge_delta=_require_number(data["edge_delta"], "edge_delta"),
            metadata=_strict_metadata(data["metadata"], "edge_breakdown.metadata"),
        )
        _validate_edge_breakdown(breakdown)
        return breakdown


def _validate_edge_breakdown(breakdown: EdgeAfterCostBreakdown) -> None:
    if type(breakdown) is not EdgeAfterCostBreakdown:
        raise TypeError("edge_breakdown must be exact EdgeAfterCostBreakdown")
    for dataclass_field in dataclass_fields(EdgeAfterCostBreakdown):
        if dataclass_field.name == "metadata":
            continue
        _require_number(
            getattr(breakdown, dataclass_field.name),
            f"edge_breakdown.{dataclass_field.name}",
        )
    for field_name in (
        "existing_capacity_penalty",
        "transaction_cost",
        "slippage_cost",
        "market_impact_cost",
        "risk_capital_charge",
        "total_cost_penalty",
    ):
        if getattr(breakdown, field_name) < 0.0:
            raise ValueError(f"edge_breakdown.{field_name} must be non-negative")
    if breakdown.edge_delta != (
        breakdown.calibrated_edge_after_costs
        - breakdown.raw_edge_after_costs
    ):
        raise ValueError("edge_breakdown edge_delta invariant invalid")
    _strict_metadata(breakdown.metadata, "edge_breakdown.metadata")


def _empty_cutoff_proof() -> OverlayCutoffProof:
    return OverlayCutoffProof(
        schema_version=OVERLAY_CUTOFF_PROOF_SCHEMA_VERSION,
        decision_as_of="",
        training_input_cutoff="",
        outcome_resolution_cutoff="",
        training_examples_sha256="",
        resolved_outcomes_sha256="",
        outcome_ledger_sha256="",
        model_id="",
        model_sha256="",
        model_trained_at="",
        proof_sha256="",
    )


@dataclass(frozen=True, slots=True)
class CalibratedPosteriorOverlay:
    schema_version: str = POSTERIOR_OVERLAY_SCHEMA_VERSION
    overlay_mode: str = OVERLAY_MODE_SHADOW
    report_only: bool = True
    production_eligible: bool = False
    production_weight: float = 0.0
    decision_as_of: str = ""
    source_sha256: str = ""
    model_id: str = ""
    model_sha256: str = ""
    model_trained_at: str = ""
    cutoff_proof_sha256: str = ""
    cutoff_proof: OverlayCutoffProof = field(default_factory=_empty_cutoff_proof)
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

    def __post_init__(self) -> None:
        _validate_calibrated_overlay(self)

    def to_dict(self) -> dict[str, Any]:
        _validate_calibrated_overlay(self)
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibratedPosteriorOverlay":
        expected = {field.name for field in dataclass_fields(cls)}
        data = _require_exact_fields(payload, expected, "CalibratedPosteriorOverlay")
        return cls(
            schema_version=_require_string(data["schema_version"], "schema_version"),
            overlay_mode=_require_string(data["overlay_mode"], "overlay_mode"),
            report_only=_require_bool(data["report_only"], "report_only"),
            production_eligible=_require_bool(
                data["production_eligible"],
                "production_eligible",
            ),
            production_weight=_require_float(
                data["production_weight"],
                "production_weight",
            ),
            decision_as_of=_require_string(
                data["decision_as_of"],
                "decision_as_of",
            ),
            source_sha256=_require_string(data["source_sha256"], "source_sha256"),
            model_id=_require_string(data["model_id"], "model_id"),
            model_sha256=_require_string(data["model_sha256"], "model_sha256"),
            model_trained_at=_require_string(
                data["model_trained_at"],
                "model_trained_at",
            ),
            cutoff_proof_sha256=_require_string(
                data["cutoff_proof_sha256"],
                "cutoff_proof_sha256",
            ),
            cutoff_proof=OverlayCutoffProof.from_dict(data["cutoff_proof"]),
            symbol=_require_string(data["symbol"], "symbol"),
            company_name=_require_string(
                data["company_name"],
                "company_name",
                allow_empty=True,
            ),
            market=_require_string(data["market"], "market", allow_empty=True),
            horizon_days=_require_int(data["horizon_days"], "horizon_days"),
            horizon_label=_require_string(data["horizon_label"], "horizon_label"),
            macro_regime=_require_string(data["macro_regime"], "macro_regime"),
            original_posterior_win_rate=_require_number(
                data["original_posterior_win_rate"],
                "original_posterior_win_rate",
            ),
            calibrated_posterior_win_rate=_require_number(
                data["calibrated_posterior_win_rate"],
                "calibrated_posterior_win_rate",
            ),
            original_posterior_expected_alpha=_require_number(
                data["original_posterior_expected_alpha"],
                "original_posterior_expected_alpha",
            ),
            calibrated_posterior_expected_alpha=float(
                _require_number(
                    data["calibrated_posterior_expected_alpha"],
                    "calibrated_posterior_expected_alpha",
                )
            ),
            original_posterior_action_score=_require_number(
                data["original_posterior_action_score"],
                "original_posterior_action_score",
            ),
            calibrated_posterior_action_score=_require_number(
                data["calibrated_posterior_action_score"],
                "calibrated_posterior_action_score",
            ),
            original_edge_after_costs=_require_number(
                data["original_edge_after_costs"],
                "original_edge_after_costs",
            ),
            calibrated_edge_after_costs=_require_number(
                data["calibrated_edge_after_costs"],
                "calibrated_edge_after_costs",
            ),
            action_threshold_used=_require_number(
                data["action_threshold_used"],
                "action_threshold_used",
            ),
            positive_edge=_require_bool(data["positive_edge"], "positive_edge"),
            diagnostics=CalibrationOverlayDiagnostics.from_dict(data["diagnostics"]),
            edge_breakdown=EdgeAfterCostBreakdown.from_dict(
                data["edge_breakdown"]
            ),
            metadata=_strict_metadata(data["metadata"], "metadata"),
        )


def _validate_calibrated_overlay(overlay: CalibratedPosteriorOverlay) -> None:
    if type(overlay) is not CalibratedPosteriorOverlay:
        raise TypeError("overlay must be exact CalibratedPosteriorOverlay")
    if overlay.schema_version != POSTERIOR_OVERLAY_SCHEMA_VERSION:
        raise ValueError("overlay schema_version invalid")
    if overlay.overlay_mode != OVERLAY_MODE_SHADOW:
        raise ValueError("overlay_mode must be exact shadow")
    if overlay.report_only is not True:
        raise ValueError("overlay must be report_only")
    if overlay.production_eligible is not False:
        raise ValueError("overlay cannot be production eligible")
    if type(overlay.production_weight) is not float:
        raise TypeError("production_weight must be a float")
    if not math.isfinite(overlay.production_weight) or overlay.production_weight != 0.0:
        raise ValueError("production_weight must be finite zero")
    _parse_aware_timestamp(overlay.decision_as_of, "decision_as_of")
    if not _is_sha256(overlay.source_sha256):
        raise ValueError("source_sha256 invalid")
    _require_string(overlay.model_id, "model_id")
    if not _is_sha256(overlay.model_sha256):
        raise ValueError("model_sha256 invalid")
    _parse_aware_timestamp(overlay.model_trained_at, "model_trained_at")
    if not _is_sha256(overlay.cutoff_proof_sha256):
        raise ValueError("cutoff_proof_sha256 invalid")
    _validate_overlay_cutoff_proof(
        overlay.cutoff_proof,
        expected_decision_as_of=overlay.decision_as_of,
    )
    if (
        overlay.cutoff_proof_sha256 != overlay.cutoff_proof.proof_sha256
        or overlay.model_id != overlay.cutoff_proof.model_id
        or overlay.model_sha256 != overlay.cutoff_proof.model_sha256
        or overlay.model_trained_at != overlay.cutoff_proof.model_trained_at
    ):
        raise ValueError("overlay proof binding mismatch")
    _require_string(overlay.symbol, "symbol")
    _require_string(overlay.company_name, "company_name", allow_empty=True)
    _require_string(overlay.market, "market", allow_empty=True)
    if type(overlay.horizon_days) is not int or overlay.horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")
    _require_string(overlay.horizon_label, "horizon_label")
    _require_string(overlay.macro_regime, "macro_regime")
    for field_name in (
        "original_posterior_win_rate",
        "calibrated_posterior_win_rate",
    ):
        value = _require_number(getattr(overlay, field_name), field_name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be in [0, 1]")
    for field_name in (
        "original_posterior_expected_alpha",
        "calibrated_posterior_expected_alpha",
        "original_posterior_action_score",
        "calibrated_posterior_action_score",
        "original_edge_after_costs",
        "calibrated_edge_after_costs",
        "action_threshold_used",
    ):
        _require_number(getattr(overlay, field_name), field_name)
    _require_bool(overlay.positive_edge, "positive_edge")
    if overlay.positive_edge is not (overlay.calibrated_edge_after_costs > 0.0):
        raise ValueError("positive_edge invariant invalid")
    _validate_overlay_diagnostics(overlay.diagnostics)
    if (
        overlay.diagnostics.model_id != overlay.model_id
        or overlay.diagnostics.schema_version != overlay.schema_version
        or overlay.diagnostics.overlay_mode != overlay.overlay_mode
        or overlay.diagnostics.raw_win_rate
        != overlay.original_posterior_win_rate
        or not math.isclose(
            overlay.diagnostics.blended_calibrated_win_rate
            - overlay.diagnostics.probability_delta_before_cap,
            overlay.original_posterior_win_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("overlay diagnostics binding mismatch")
    _validate_edge_breakdown(overlay.edge_breakdown)
    if (
        overlay.edge_breakdown.raw_expected_alpha
        != overlay.original_posterior_expected_alpha
        or overlay.edge_breakdown.calibrated_expected_alpha
        != overlay.calibrated_posterior_expected_alpha
        or overlay.edge_breakdown.raw_edge_after_costs
        != overlay.original_edge_after_costs
        or overlay.edge_breakdown.calibrated_edge_after_costs
        != overlay.calibrated_edge_after_costs
    ):
        raise ValueError("overlay edge breakdown binding mismatch")
    _strict_metadata(overlay.metadata, "metadata")


def _cap_probability_delta(raw_win_rate: float, blended_win_rate: float, max_adjustment: float) -> tuple[float, float, bool]:
    delta = blended_win_rate - raw_win_rate
    capped_delta = max(-max_adjustment, min(max_adjustment, delta))
    return capped_delta, raw_win_rate + capped_delta, abs(capped_delta - delta) > 1e-15


def _build_calibrated_posterior_shadow(
    result: PosteriorResult,
    model: CalibrationModelV2,
    *,
    decision_as_of: str,
    source_sha256: str,
    model_sha256: str,
    cutoff_proof: OverlayCutoffProof,
    market: str = "",
    horizon_days: int = 20,
    horizon_label: str | None = None,
    macro_regime: str = "未知",
    edge_cost_config: EdgeCostConfig | None = None,
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
        overlay_mode=OVERLAY_MODE_SHADOW,
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
    overlay_metadata = _strict_metadata(
        _coerce_metadata(metadata),
        "metadata",
    )
    return CalibratedPosteriorOverlay(
        schema_version=POSTERIOR_OVERLAY_SCHEMA_VERSION,
        overlay_mode=OVERLAY_MODE_SHADOW,
        report_only=True,
        production_eligible=False,
        production_weight=0.0,
        decision_as_of=decision_as_of,
        source_sha256=source_sha256,
        model_id=model.model_id,
        model_sha256=model_sha256,
        model_trained_at=model.trained_at,
        cutoff_proof_sha256=cutoff_proof.proof_sha256,
        cutoff_proof=cutoff_proof,
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


def _validate_overlay_mode(mode: Any) -> str:
    if type(mode) is not str or mode not in {
        OVERLAY_MODE_OFF,
        OVERLAY_MODE_SHADOW,
    }:
        raise ValueError("overlay mode must be exact 'off' or 'shadow'")
    return mode


def run_calibrated_posterior_overlay(
    *,
    mode: str = OVERLAY_MODE_OFF,
    result: PosteriorResult | None = None,
    decision_as_of: datetime | None = None,
    cutoff_proof: OverlayCutoffProof | None = None,
    model_loader: Callable[[], CalibrationModelV2] | None = None,
    market: str = "",
    horizon_days: int = 20,
    horizon_label: str | None = None,
    macro_regime: str = "未知",
    edge_cost_config: EdgeCostConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibratedPosteriorOverlay | None:
    """Run the report-only overlay after explicit cutoff and model validation."""

    resolved_mode = _validate_overlay_mode(mode)
    if resolved_mode == OVERLAY_MODE_OFF:
        return None

    decision_as_of_text = _canonical_datetime_text(
        decision_as_of,  # type: ignore[arg-type]
        "decision_as_of",
    )
    if type(cutoff_proof) is not OverlayCutoffProof:
        raise ValueError("shadow mode requires an overlay cutoff proof")
    _validate_overlay_cutoff_proof(
        cutoff_proof,
        expected_decision_as_of=decision_as_of_text,
    )
    if type(result) is not PosteriorResult:
        raise TypeError("shadow mode requires an exact PosteriorResult")
    source_sha256 = posterior_result_source_sha256(result)
    if edge_cost_config is None:
        config = EdgeCostConfig()
    elif type(edge_cost_config) is EdgeCostConfig:
        config = edge_cost_config
    else:
        raise TypeError("edge_cost_config must be an exact EdgeCostConfig")
    resolved_metadata = _strict_metadata(
        _coerce_metadata(metadata),
        "metadata",
    )
    if model_loader is None or not callable(model_loader):
        raise ValueError("shadow mode requires a callable model_loader")

    model = model_loader()
    if type(model) is not CalibrationModelV2:
        raise TypeError("model_loader must return an exact CalibrationModelV2")
    model_sha256 = calibration_model_sha256(model)
    if (
        model.model_id != cutoff_proof.model_id
        or model.trained_at != cutoff_proof.model_trained_at
        or model_sha256 != cutoff_proof.model_sha256
    ):
        raise ValueError("loaded model identity, trained_at, or sha256 mismatch")

    overlay = _build_calibrated_posterior_shadow(
        result,
        model,
        decision_as_of=decision_as_of_text,
        source_sha256=source_sha256,
        model_sha256=model_sha256,
        cutoff_proof=cutoff_proof,
        market=market,
        horizon_days=horizon_days,
        horizon_label=horizon_label,
        macro_regime=macro_regime,
        edge_cost_config=config,
        metadata=resolved_metadata,
    )
    if posterior_result_source_sha256(result) != source_sha256:
        raise ValueError("posterior source drifted during model calls")
    if calibration_model_sha256(model) != model_sha256:
        raise ValueError("calibration model drifted during model calls")
    return overlay


def build_calibrated_posterior_overlay(
    result: PosteriorResult,
    model: CalibrationModelV2,
    *,
    market: str = "",
    horizon_days: int = 20,
    horizon_label: str | None = None,
    macro_regime: str = "未知",
    edge_cost_config: EdgeCostConfig | None = None,
    overlay_mode: str = OVERLAY_MODE_OFF,
    decision_as_of: datetime | None = None,
    cutoff_proof: OverlayCutoffProof | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibratedPosteriorOverlay | None:
    """Compatibility wrapper; default-off and shadow-only through the runner."""

    return run_calibrated_posterior_overlay(
        mode=overlay_mode,
        result=result,
        decision_as_of=decision_as_of,
        cutoff_proof=cutoff_proof,
        model_loader=lambda: model,
        market=market,
        horizon_days=horizon_days,
        horizon_label=horizon_label,
        macro_regime=macro_regime,
        edge_cost_config=edge_cost_config,
        metadata=metadata,
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
    overlay_mode: str = OVERLAY_MODE_OFF,
    decision_as_of: datetime | None = None,
    cutoff_proof: OverlayCutoffProof | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[CalibratedPosteriorOverlay]:
    resolved_mode = _validate_overlay_mode(overlay_mode)
    if resolved_mode == OVERLAY_MODE_OFF:
        return []
    resolved_horizon_label = horizon_label or horizon_label_for_days(horizon_days)
    seen: set[tuple[str, str]] = set()
    preflight_results: list[PosteriorResult] = []
    for result in results:
        if type(result) is not PosteriorResult:
            raise TypeError("batch result must be an exact PosteriorResult")
        key = (result.symbol, resolved_horizon_label)
        if key in seen:
            raise ValueError(f"Duplicate posterior overlay input for symbol+horizon: {key!r}.")
        seen.add(key)
        preflight_results.append(result)
    overlays: list[CalibratedPosteriorOverlay] = []
    for result in preflight_results:
        overlay = build_calibrated_posterior_overlay(
            result,
            model,
            market=market,
            horizon_days=horizon_days,
            horizon_label=resolved_horizon_label,
            macro_regime=macro_regime,
            edge_cost_config=edge_cost_config,
            overlay_mode=resolved_mode,
            decision_as_of=decision_as_of,
            cutoff_proof=cutoff_proof,
            metadata=metadata,
        )
        if overlay is None:
            raise RuntimeError("shadow overlay wrapper returned no result")
        overlays.append(overlay)
    return overlays


def attach_overlay_metadata(
    result: PosteriorResult,
    overlay: CalibratedPosteriorOverlay,
    *,
    mutate: bool = False,
) -> PosteriorResult:
    if mutate:
        raise ValueError("attach_overlay_metadata mutate=True is forbidden")
    if type(result) is not PosteriorResult:
        raise TypeError("result must be an exact PosteriorResult")
    if type(overlay) is not CalibratedPosteriorOverlay:
        raise TypeError("overlay must be an exact CalibratedPosteriorOverlay")
    overlay_payload = overlay.to_dict()
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
    "OVERLAY_CUTOFF_PROOF_SCHEMA_VERSION",
    "EdgeCostConfig",
    "OverlayCutoffProof",
    "CalibrationOverlayDiagnostics",
    "EdgeAfterCostBreakdown",
    "CalibratedPosteriorOverlay",
    "clamp_probability",
    "bps_to_decimal_return",
    "horizon_label_for_days",
    "posterior_result_source_sha256",
    "calibration_model_sha256",
    "overlay_cutoff_proof_sha256",
    "run_calibrated_posterior_overlay",
    "build_calibrated_posterior_overlay",
    "build_calibrated_posterior_overlays",
    "attach_overlay_metadata",
]
