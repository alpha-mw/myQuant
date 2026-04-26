"""Offline structured risk tensor and execution feasibility contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from quant_investor.versioning import RISK_TENSOR_SCHEMA_VERSION


DEFAULT_RISK_TENSOR_DIR = Path("data/risk_tensor")
DEFAULT_SYMBOL_TENSORS_FILENAME = "symbol_tensors.jsonl"
DEFAULT_PORTFOLIO_TENSORS_FILENAME = "portfolio_tensors.jsonl"
DEFAULT_EXECUTION_REPORTS_FILENAME = "execution_reports.jsonl"

RISK_SEVERITY_INFO = "info"
RISK_SEVERITY_WARNING = "warning"
RISK_SEVERITY_BLOCKER = "blocker"

RISK_ISSUE_DATA_QUARANTINE = "data_quarantine"
RISK_ISSUE_UNTRADABLE = "untradable"
RISK_ISSUE_LOW_LIQUIDITY = "low_liquidity"
RISK_ISSUE_POSITION_TOO_LARGE = "position_too_large"
RISK_ISSUE_ADV_CAP_EXCEEDED = "adv_cap_exceeded"
RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED = "max_order_value_exceeded"
RISK_ISSUE_SECTOR_CONCENTRATION = "sector_concentration"
RISK_ISSUE_STYLE_EXPOSURE = "style_exposure"
RISK_ISSUE_BETA_EXPOSURE = "beta_exposure"
RISK_ISSUE_CORRELATION_CLUSTER = "correlation_cluster"
RISK_ISSUE_STRESS_LOSS = "stress_loss"
RISK_ISSUE_TURNOVER_EXCEEDED = "turnover_exceeded"

EXECUTION_FEASIBLE = "feasible"
EXECUTION_PARTIALLY_FEASIBLE = "partially_feasible"
EXECUTION_BLOCKED = "blocked"

_VALID_SEVERITIES = {
    RISK_SEVERITY_INFO,
    RISK_SEVERITY_WARNING,
    RISK_SEVERITY_BLOCKER,
}
_VALID_EXECUTION_STATUSES = {
    EXECUTION_FEASIBLE,
    EXECUTION_PARTIALLY_FEASIBLE,
    EXECUTION_BLOCKED,
}
_EXECUTION_REASON_ORDER = [
    RISK_ISSUE_UNTRADABLE,
    RISK_ISSUE_POSITION_TOO_LARGE,
    RISK_ISSUE_ADV_CAP_EXCEEDED,
    RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED,
    RISK_ISSUE_LOW_LIQUIDITY,
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_json_serializable(value: Any, label: str) -> Any:
    safe = _json_safe(value)
    try:
        json.dumps(safe, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise ValueError(f"{label} must contain only JSON-serializable values.") from exc
    return safe


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, "metadata"))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def validate_finite_number(value: float, *, field_name: str) -> None:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")


def _finite_float(value: Any, field_name: str) -> float:
    number = float(value)
    validate_finite_number(number, field_name=field_name)
    return number


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field_name)


def _non_negative_float_or_none(value: Any, field_name: str) -> float | None:
    number = _optional_finite_float(value, field_name)
    if number is not None and number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    number = _optional_finite_float(value, field_name)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _finite_float_dict(value: Mapping[str, Any] | None, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    result: dict[str, float] = {}
    for key in sorted(value):
        result[str(key)] = _finite_float(value[key], f"{field_name}.{key}")
    return result


def _ordered_unique(values: Sequence[str], *, preferred_order: Sequence[str] | None = None) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw_value in values:
        value = str(raw_value)
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    if preferred_order is None:
        return sorted(unique)
    order = {value: index for index, value in enumerate(preferred_order)}
    return sorted(unique, key=lambda value: (order.get(value, len(order)), value))


def _sort_issues(issues: Sequence["RiskIssue"]) -> list["RiskIssue"]:
    return sorted(
        issues,
        key=lambda issue: (
            issue.symbol or "",
            issue.market or "",
            issue.as_of,
            issue.issue_type,
            issue.severity,
            issue.issue_id,
            issue.message,
        ),
    )


def _slug(value: str | None) -> str:
    resolved = "portfolio" if value is None else str(value).strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved)
    return slug.strip("-") or "unknown"


def _short_hash(parts: Sequence[str]) -> str:
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hashable_none(value: str | None) -> str:
    return "<none>" if value is None else str(value)


def clamp_unit_interval(value: float) -> float:
    validate_finite_number(value, field_name="value")
    return max(0.0, min(1.0, float(value)))


def bps_to_decimal_return(value_bps: float) -> float:
    return _finite_float(value_bps, "value_bps") / 10000.0


def weighted_average(values: Mapping[str, float], weights: Mapping[str, float]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for key in sorted(values):
        if key not in weights:
            continue
        value = _finite_float(values[key], f"values.{key}")
        weight = _finite_float(weights[key], f"weights.{key}")
        numerator += value * weight
        denominator += abs(weight)
    if denominator == 0.0:
        return None
    return numerator / denominator


def make_risk_issue_id(
    *,
    symbol: str | None,
    market: str | None,
    as_of: str,
    issue_type: str,
    message: str,
) -> str:
    parts = [
        _hashable_none(market),
        _hashable_none(symbol),
        str(as_of),
        str(issue_type),
        str(message),
    ]
    return (
        f"risk-issue-{_slug(market)}-{_slug(symbol)}-{_slug(issue_type)}-"
        f"{_short_hash(parts)}"
    )


def make_symbol_tensor_id(*, symbol: str, market: str, as_of: str, latest_trade_date: str) -> str:
    parts = [str(market), str(symbol), str(as_of), str(latest_trade_date)]
    return (
        f"risk-symbol-{_slug(market)}-{_slug(symbol)}-{_slug(as_of)}-"
        f"{_slug(latest_trade_date)}-{_short_hash(parts)}"
    )


def make_portfolio_tensor_id(
    *,
    market: str,
    as_of: str,
    symbol_tensor_ids: Sequence[str],
) -> str:
    ordered_ids = sorted(str(value) for value in symbol_tensor_ids)
    parts = [str(market), str(as_of), *ordered_ids]
    return f"risk-portfolio-{_slug(market)}-{_slug(as_of)}-{len(ordered_ids)}-{_short_hash(parts)}"


def make_execution_report_id(
    *,
    market: str,
    as_of: str,
    symbol_tensor_ids: Sequence[str],
) -> str:
    ordered_ids = sorted(str(value) for value in symbol_tensor_ids)
    parts = [str(market), str(as_of), *ordered_ids]
    return f"execution-report-{_slug(market)}-{_slug(as_of)}-{len(ordered_ids)}-{_short_hash(parts)}"


@dataclass
class RiskIssue:
    schema_version: str = RISK_TENSOR_SCHEMA_VERSION
    issue_id: str = ""
    symbol: str | None = None
    market: str | None = None
    as_of: str = ""
    issue_type: str = ""
    severity: str = RISK_SEVERITY_WARNING
    message: str = ""
    value: float | None = None
    limit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_VALID_SEVERITIES)}; got {self.severity!r}.")
        self.symbol = _optional_str(self.symbol)
        self.market = _optional_str(self.market)
        self.value = _optional_finite_float(self.value, "value")
        self.limit = _optional_finite_float(self.limit, "limit")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RiskIssue":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", RISK_TENSOR_SCHEMA_VERSION)),
            issue_id=str(data.get("issue_id", "")),
            symbol=_optional_str(data.get("symbol")),
            market=_optional_str(data.get("market")),
            as_of=str(data.get("as_of", "")),
            issue_type=str(data.get("issue_type", "")),
            severity=str(data.get("severity", RISK_SEVERITY_WARNING)),
            message=str(data.get("message", "")),
            value=data.get("value"),
            limit=data.get("limit"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class SymbolExposure:
    schema_version: str = RISK_TENSOR_SCHEMA_VERSION
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    sector: str | None = None
    industry: str | None = None
    beta: float | None = None
    style_exposures: dict[str, float] = field(default_factory=dict)
    factor_exposures: dict[str, float] = field(default_factory=dict)
    correlation_cluster: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sector = _optional_str(self.sector)
        self.industry = _optional_str(self.industry)
        self.beta = _optional_finite_float(self.beta, "beta")
        self.style_exposures = _finite_float_dict(self.style_exposures, "style_exposures")
        self.factor_exposures = _finite_float_dict(self.factor_exposures, "factor_exposures")
        self.correlation_cluster = _optional_str(self.correlation_cluster)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolExposure":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", RISK_TENSOR_SCHEMA_VERSION)),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            sector=_optional_str(data.get("sector")),
            industry=_optional_str(data.get("industry")),
            beta=data.get("beta"),
            style_exposures=dict(data.get("style_exposures", {}) or {}),
            factor_exposures=dict(data.get("factor_exposures", {}) or {}),
            correlation_cluster=_optional_str(data.get("correlation_cluster")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class LiquidityProfile:
    schema_version: str = RISK_TENSOR_SCHEMA_VERSION
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    adv: float | None = None
    liquidity_score: float | None = None
    max_order_value: float | None = None
    max_participation_rate: float | None = None
    estimated_spread_bps: float | None = None
    estimated_market_impact_bps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.adv = _non_negative_float_or_none(self.adv, "adv")
        self.liquidity_score = _unit_float_or_none(self.liquidity_score, "liquidity_score")
        self.max_order_value = _non_negative_float_or_none(self.max_order_value, "max_order_value")
        self.max_participation_rate = _unit_float_or_none(
            self.max_participation_rate,
            "max_participation_rate",
        )
        self.estimated_spread_bps = _non_negative_float_or_none(
            self.estimated_spread_bps,
            "estimated_spread_bps",
        )
        self.estimated_market_impact_bps = _non_negative_float_or_none(
            self.estimated_market_impact_bps,
            "estimated_market_impact_bps",
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LiquidityProfile":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", RISK_TENSOR_SCHEMA_VERSION)),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            adv=data.get("adv"),
            liquidity_score=data.get("liquidity_score"),
            max_order_value=data.get("max_order_value"),
            max_participation_rate=data.get("max_participation_rate"),
            estimated_spread_bps=data.get("estimated_spread_bps"),
            estimated_market_impact_bps=data.get("estimated_market_impact_bps"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ExecutionFeasibility:
    schema_version: str = RISK_TENSOR_SCHEMA_VERSION
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    status: str = EXECUTION_FEASIBLE
    requested_weight: float = 0.0
    current_weight: float = 0.0
    requested_trade_value: float | None = None
    allowed_trade_value: float | None = None
    adv_usage: float | None = None
    max_order_value_usage: float | None = None
    blocked_reasons: list[str] = field(default_factory=list)
    warning_reasons: list[str] = field(default_factory=list)
    estimated_transaction_cost_bps: float | None = None
    estimated_slippage_bps: float | None = None
    estimated_market_impact_bps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_EXECUTION_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_EXECUTION_STATUSES)}; got {self.status!r}."
            )
        self.requested_weight = _finite_float(self.requested_weight, "requested_weight")
        self.current_weight = _finite_float(self.current_weight, "current_weight")
        self.requested_trade_value = _non_negative_float_or_none(
            self.requested_trade_value,
            "requested_trade_value",
        )
        self.allowed_trade_value = _non_negative_float_or_none(
            self.allowed_trade_value,
            "allowed_trade_value",
        )
        self.adv_usage = _non_negative_float_or_none(self.adv_usage, "adv_usage")
        self.max_order_value_usage = _non_negative_float_or_none(
            self.max_order_value_usage,
            "max_order_value_usage",
        )
        self.blocked_reasons = _ordered_unique(
            self.blocked_reasons,
            preferred_order=_EXECUTION_REASON_ORDER,
        )
        self.warning_reasons = _ordered_unique(
            self.warning_reasons,
            preferred_order=_EXECUTION_REASON_ORDER,
        )
        self.estimated_transaction_cost_bps = _non_negative_float_or_none(
            self.estimated_transaction_cost_bps,
            "estimated_transaction_cost_bps",
        )
        self.estimated_slippage_bps = _non_negative_float_or_none(
            self.estimated_slippage_bps,
            "estimated_slippage_bps",
        )
        self.estimated_market_impact_bps = _non_negative_float_or_none(
            self.estimated_market_impact_bps,
            "estimated_market_impact_bps",
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionFeasibility":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", RISK_TENSOR_SCHEMA_VERSION)),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            status=str(data.get("status", EXECUTION_FEASIBLE)),
            requested_weight=float(data.get("requested_weight", 0.0)),
            current_weight=float(data.get("current_weight", 0.0)),
            requested_trade_value=data.get("requested_trade_value"),
            allowed_trade_value=data.get("allowed_trade_value"),
            adv_usage=data.get("adv_usage"),
            max_order_value_usage=data.get("max_order_value_usage"),
            blocked_reasons=list(data.get("blocked_reasons", []) or []),
            warning_reasons=list(data.get("warning_reasons", []) or []),
            estimated_transaction_cost_bps=data.get("estimated_transaction_cost_bps"),
            estimated_slippage_bps=data.get("estimated_slippage_bps"),
            estimated_market_impact_bps=data.get("estimated_market_impact_bps"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class StressScenarioResult:
    schema_version: str = RISK_TENSOR_SCHEMA_VERSION
    scenario_name: str = ""
    symbol: str | None = None
    market: str | None = None
    as_of: str = ""
    shock_return: float = 0.0
    position_weight: float = 0.0
    estimated_loss: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = _optional_str(self.symbol)
        self.market = _optional_str(self.market)
        self.shock_return = _finite_float(self.shock_return, "shock_return")
        self.position_weight = _finite_float(self.position_weight, "position_weight")
        self.estimated_loss = _non_negative_float_or_none(self.estimated_loss, "estimated_loss") or 0.0
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StressScenarioResult":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", RISK_TENSOR_SCHEMA_VERSION)),
            scenario_name=str(data.get("scenario_name", "")),
            symbol=_optional_str(data.get("symbol")),
            market=_optional_str(data.get("market")),
            as_of=str(data.get("as_of", "")),
            shock_return=float(data.get("shock_return", 0.0)),
            position_weight=float(data.get("position_weight", 0.0)),
            estimated_loss=float(data.get("estimated_loss", 0.0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class SymbolRiskTensor:
    schema_version: str = RISK_TENSOR_SCHEMA_VERSION
    tensor_id: str = ""
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    latest_trade_date: str = ""
    target_weight: float = 0.0
    current_weight: float = 0.0
    data_quality_score: float | None = None
    is_researchable: bool = True
    is_tradable: bool = True
    quarantine: bool = False
    exposure: SymbolExposure = field(default_factory=SymbolExposure)
    liquidity: LiquidityProfile = field(default_factory=LiquidityProfile)
    execution: ExecutionFeasibility = field(default_factory=ExecutionFeasibility)
    stress_results: list[StressScenarioResult] = field(default_factory=list)
    issues: list[RiskIssue] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_weight = _finite_float(self.target_weight, "target_weight")
        self.current_weight = _finite_float(self.current_weight, "current_weight")
        self.data_quality_score = _unit_float_or_none(self.data_quality_score, "data_quality_score")
        if not isinstance(self.exposure, SymbolExposure):
            self.exposure = SymbolExposure.from_dict(self.exposure)
        if not isinstance(self.liquidity, LiquidityProfile):
            self.liquidity = LiquidityProfile.from_dict(self.liquidity)
        if not isinstance(self.execution, ExecutionFeasibility):
            self.execution = ExecutionFeasibility.from_dict(self.execution)
        self.stress_results = [
            result if isinstance(result, StressScenarioResult) else StressScenarioResult.from_dict(result)
            for result in self.stress_results
        ]
        self.stress_results = sorted(
            self.stress_results,
            key=lambda result: (result.scenario_name, result.symbol or "", result.market or ""),
        )
        self.issues = [
            issue if isinstance(issue, RiskIssue) else RiskIssue.from_dict(issue)
            for issue in self.issues
        ]
        self.issues = _sort_issues(self.issues)
        self.risk_score = clamp_unit_interval(self.risk_score)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tensor_id": self.tensor_id,
            "symbol": self.symbol,
            "market": self.market,
            "as_of": self.as_of,
            "latest_trade_date": self.latest_trade_date,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "data_quality_score": self.data_quality_score,
            "is_researchable": self.is_researchable,
            "is_tradable": self.is_tradable,
            "quarantine": self.quarantine,
            "exposure": self.exposure.to_dict(),
            "liquidity": self.liquidity.to_dict(),
            "execution": self.execution.to_dict(),
            "stress_results": [result.to_dict() for result in self.stress_results],
            "issues": [issue.to_dict() for issue in self.issues],
            "risk_score": self.risk_score,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolRiskTensor":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", RISK_TENSOR_SCHEMA_VERSION)),
            tensor_id=str(data.get("tensor_id", "")),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            latest_trade_date=str(data.get("latest_trade_date", "")),
            target_weight=float(data.get("target_weight", 0.0)),
            current_weight=float(data.get("current_weight", 0.0)),
            data_quality_score=data.get("data_quality_score"),
            is_researchable=bool(data.get("is_researchable", True)),
            is_tradable=bool(data.get("is_tradable", True)),
            quarantine=bool(data.get("quarantine", False)),
            exposure=SymbolExposure.from_dict(dict(data.get("exposure", {}) or {})),
            liquidity=LiquidityProfile.from_dict(dict(data.get("liquidity", {}) or {})),
            execution=ExecutionFeasibility.from_dict(dict(data.get("execution", {}) or {})),
            stress_results=[
                StressScenarioResult.from_dict(result)
                for result in list(data.get("stress_results", []) or [])
                if isinstance(result, Mapping)
            ],
            issues=[
                RiskIssue.from_dict(issue)
                for issue in list(data.get("issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            risk_score=float(data.get("risk_score", 0.0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class PortfolioRiskTensor:
    schema_version: str = RISK_TENSOR_SCHEMA_VERSION
    tensor_id: str = ""
    as_of: str = ""
    market: str = ""
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    turnover_estimate: float = 0.0
    sector_weights: dict[str, float] = field(default_factory=dict)
    style_exposures: dict[str, float] = field(default_factory=dict)
    factor_exposures: dict[str, float] = field(default_factory=dict)
    beta_exposure: float | None = None
    correlation_cluster_weights: dict[str, float] = field(default_factory=dict)
    symbol_tensors: list[SymbolRiskTensor] = field(default_factory=list)
    portfolio_issues: list[RiskIssue] = field(default_factory=list)
    blocked_symbols: list[str] = field(default_factory=list)
    max_weight_by_symbol: dict[str, float] = field(default_factory=dict)
    gross_exposure_cap: float | None = None
    risk_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gross_exposure = _finite_float(self.gross_exposure, "gross_exposure")
        self.net_exposure = _finite_float(self.net_exposure, "net_exposure")
        self.long_exposure = _finite_float(self.long_exposure, "long_exposure")
        self.short_exposure = _finite_float(self.short_exposure, "short_exposure")
        self.turnover_estimate = _finite_float(self.turnover_estimate, "turnover_estimate")
        self.sector_weights = _finite_float_dict(self.sector_weights, "sector_weights")
        self.style_exposures = _finite_float_dict(self.style_exposures, "style_exposures")
        self.factor_exposures = _finite_float_dict(self.factor_exposures, "factor_exposures")
        self.beta_exposure = _optional_finite_float(self.beta_exposure, "beta_exposure")
        self.correlation_cluster_weights = _finite_float_dict(
            self.correlation_cluster_weights,
            "correlation_cluster_weights",
        )
        self.symbol_tensors = [
            tensor if isinstance(tensor, SymbolRiskTensor) else SymbolRiskTensor.from_dict(tensor)
            for tensor in self.symbol_tensors
        ]
        self.symbol_tensors = sorted(self.symbol_tensors, key=lambda tensor: (tensor.symbol, tensor.market))
        self.portfolio_issues = [
            issue if isinstance(issue, RiskIssue) else RiskIssue.from_dict(issue)
            for issue in self.portfolio_issues
        ]
        self.portfolio_issues = _sort_issues(self.portfolio_issues)
        self.blocked_symbols = _ordered_unique(self.blocked_symbols)
        self.max_weight_by_symbol = _finite_float_dict(self.max_weight_by_symbol, "max_weight_by_symbol")
        self.gross_exposure_cap = _non_negative_float_or_none(
            self.gross_exposure_cap,
            "gross_exposure_cap",
        )
        self.risk_score = clamp_unit_interval(self.risk_score)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tensor_id": self.tensor_id,
            "as_of": self.as_of,
            "market": self.market,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "long_exposure": self.long_exposure,
            "short_exposure": self.short_exposure,
            "turnover_estimate": self.turnover_estimate,
            "sector_weights": dict(self.sector_weights),
            "style_exposures": dict(self.style_exposures),
            "factor_exposures": dict(self.factor_exposures),
            "beta_exposure": self.beta_exposure,
            "correlation_cluster_weights": dict(self.correlation_cluster_weights),
            "symbol_tensors": [tensor.to_dict() for tensor in self.symbol_tensors],
            "portfolio_issues": [issue.to_dict() for issue in self.portfolio_issues],
            "blocked_symbols": list(self.blocked_symbols),
            "max_weight_by_symbol": dict(self.max_weight_by_symbol),
            "gross_exposure_cap": self.gross_exposure_cap,
            "risk_score": self.risk_score,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioRiskTensor":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", RISK_TENSOR_SCHEMA_VERSION)),
            tensor_id=str(data.get("tensor_id", "")),
            as_of=str(data.get("as_of", "")),
            market=str(data.get("market", "")),
            gross_exposure=float(data.get("gross_exposure", 0.0)),
            net_exposure=float(data.get("net_exposure", 0.0)),
            long_exposure=float(data.get("long_exposure", 0.0)),
            short_exposure=float(data.get("short_exposure", 0.0)),
            turnover_estimate=float(data.get("turnover_estimate", 0.0)),
            sector_weights=dict(data.get("sector_weights", {}) or {}),
            style_exposures=dict(data.get("style_exposures", {}) or {}),
            factor_exposures=dict(data.get("factor_exposures", {}) or {}),
            beta_exposure=data.get("beta_exposure"),
            correlation_cluster_weights=dict(data.get("correlation_cluster_weights", {}) or {}),
            symbol_tensors=[
                SymbolRiskTensor.from_dict(tensor)
                for tensor in list(data.get("symbol_tensors", []) or [])
                if isinstance(tensor, Mapping)
            ],
            portfolio_issues=[
                RiskIssue.from_dict(issue)
                for issue in list(data.get("portfolio_issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            blocked_symbols=list(data.get("blocked_symbols", []) or []),
            max_weight_by_symbol=dict(data.get("max_weight_by_symbol", {}) or {}),
            gross_exposure_cap=data.get("gross_exposure_cap"),
            risk_score=float(data.get("risk_score", 0.0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ExecutionFeasibilityReport:
    schema_version: str = RISK_TENSOR_SCHEMA_VERSION
    report_id: str = ""
    as_of: str = ""
    market: str = ""
    total_symbols: int = 0
    feasible_symbols: list[str] = field(default_factory=list)
    partially_feasible_symbols: list[str] = field(default_factory=list)
    blocked_symbols: list[str] = field(default_factory=list)
    execution_by_symbol: dict[str, ExecutionFeasibility] = field(default_factory=dict)
    total_requested_trade_value: float | None = None
    total_allowed_trade_value: float | None = None
    aggregate_estimated_cost_bps: float | None = None
    issues: list[RiskIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_symbols < 0:
            raise ValueError("total_symbols must be non-negative.")
        self.feasible_symbols = _ordered_unique(self.feasible_symbols)
        self.partially_feasible_symbols = _ordered_unique(self.partially_feasible_symbols)
        self.blocked_symbols = _ordered_unique(self.blocked_symbols)
        self.execution_by_symbol = {
            str(symbol): (
                execution
                if isinstance(execution, ExecutionFeasibility)
                else ExecutionFeasibility.from_dict(execution)
            )
            for symbol, execution in sorted(self.execution_by_symbol.items())
        }
        self.total_requested_trade_value = _non_negative_float_or_none(
            self.total_requested_trade_value,
            "total_requested_trade_value",
        )
        self.total_allowed_trade_value = _non_negative_float_or_none(
            self.total_allowed_trade_value,
            "total_allowed_trade_value",
        )
        self.aggregate_estimated_cost_bps = _non_negative_float_or_none(
            self.aggregate_estimated_cost_bps,
            "aggregate_estimated_cost_bps",
        )
        self.issues = [
            issue if isinstance(issue, RiskIssue) else RiskIssue.from_dict(issue)
            for issue in self.issues
        ]
        self.issues = _sort_issues(self.issues)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "as_of": self.as_of,
            "market": self.market,
            "total_symbols": self.total_symbols,
            "feasible_symbols": list(self.feasible_symbols),
            "partially_feasible_symbols": list(self.partially_feasible_symbols),
            "blocked_symbols": list(self.blocked_symbols),
            "execution_by_symbol": {
                symbol: execution.to_dict()
                for symbol, execution in self.execution_by_symbol.items()
            },
            "total_requested_trade_value": self.total_requested_trade_value,
            "total_allowed_trade_value": self.total_allowed_trade_value,
            "aggregate_estimated_cost_bps": self.aggregate_estimated_cost_bps,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionFeasibilityReport":
        data = dict(payload)
        execution_payload = dict(data.get("execution_by_symbol", {}) or {})
        return cls(
            schema_version=str(data.get("schema_version", RISK_TENSOR_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            as_of=str(data.get("as_of", "")),
            market=str(data.get("market", "")),
            total_symbols=int(data.get("total_symbols", 0) or 0),
            feasible_symbols=list(data.get("feasible_symbols", []) or []),
            partially_feasible_symbols=list(data.get("partially_feasible_symbols", []) or []),
            blocked_symbols=list(data.get("blocked_symbols", []) or []),
            execution_by_symbol={
                str(symbol): ExecutionFeasibility.from_dict(execution)
                for symbol, execution in execution_payload.items()
                if isinstance(execution, Mapping)
            },
            total_requested_trade_value=data.get("total_requested_trade_value"),
            total_allowed_trade_value=data.get("total_allowed_trade_value"),
            aggregate_estimated_cost_bps=data.get("aggregate_estimated_cost_bps"),
            issues=[
                RiskIssue.from_dict(issue)
                for issue in list(data.get("issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


def _make_issue(
    *,
    symbol: str | None,
    market: str | None,
    as_of: str,
    issue_type: str,
    severity: str,
    message: str,
    value: float | None = None,
    limit: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RiskIssue:
    return RiskIssue(
        issue_id=make_risk_issue_id(
            symbol=symbol,
            market=market,
            as_of=as_of,
            issue_type=issue_type,
            message=message,
        ),
        symbol=symbol,
        market=market,
        as_of=as_of,
        issue_type=issue_type,
        severity=severity,
        message=message,
        value=value,
        limit=limit,
        metadata=_coerce_metadata(metadata),
    )


def _extract_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _issue_severity(value: Any) -> str | None:
    if isinstance(value, Mapping):
        severity = value.get("severity")
    else:
        severity = getattr(value, "severity", None)
    if severity is None:
        return None
    return str(severity)


def _get_optional_float_attr(source: Any, name: str) -> float | None:
    value = getattr(source, name, None)
    if value is None:
        return None
    return _finite_float(value, name)


def build_execution_feasibility(
    *,
    symbol: str,
    market: str,
    as_of: str,
    target_weight: float,
    current_weight: float = 0.0,
    portfolio_value: float | None = None,
    liquidity: LiquidityProfile | None = None,
    is_tradable: bool = True,
    tradability_reasons: Sequence[str] | None = None,
    max_weight: float | None = None,
    max_trade_value: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionFeasibility:
    resolved_target_weight = _finite_float(target_weight, "target_weight")
    resolved_current_weight = _finite_float(current_weight, "current_weight")
    resolved_portfolio_value = _non_negative_float_or_none(portfolio_value, "portfolio_value")
    resolved_max_weight = _non_negative_float_or_none(max_weight, "max_weight")
    resolved_max_trade_value = _non_negative_float_or_none(max_trade_value, "max_trade_value")
    resolved_liquidity = liquidity or LiquidityProfile(symbol=symbol, market=market, as_of=as_of)

    requested_trade_value = None
    if resolved_portfolio_value is not None:
        requested_trade_value = abs(resolved_target_weight - resolved_current_weight) * resolved_portfolio_value

    adv_allowed = None
    if (
        resolved_liquidity.adv is not None
        and resolved_liquidity.max_participation_rate is not None
    ):
        adv_allowed = resolved_liquidity.adv * resolved_liquidity.max_participation_rate

    candidates = [
        value
        for value in [
            resolved_liquidity.max_order_value,
            adv_allowed,
            resolved_max_trade_value,
        ]
        if value is not None
    ]
    allowed_trade_value = min(candidates) if candidates else None

    adv_usage = None
    if requested_trade_value is not None and resolved_liquidity.adv and resolved_liquidity.adv > 0.0:
        adv_usage = requested_trade_value / resolved_liquidity.adv

    max_order_value_usage = None
    if (
        requested_trade_value is not None
        and resolved_liquidity.max_order_value is not None
        and resolved_liquidity.max_order_value > 0.0
    ):
        max_order_value_usage = requested_trade_value / resolved_liquidity.max_order_value

    blocked_reasons: list[str] = []
    warning_reasons: list[str] = []

    if not is_tradable:
        blocked_reasons.append(RISK_ISSUE_UNTRADABLE)

    if resolved_max_weight is not None and abs(resolved_target_weight) > resolved_max_weight:
        blocked_reasons.append(RISK_ISSUE_POSITION_TOO_LARGE)

    if requested_trade_value is not None:
        if adv_allowed is not None and requested_trade_value > adv_allowed:
            warning_reasons.append(RISK_ISSUE_ADV_CAP_EXCEEDED)
        if (
            resolved_liquidity.max_order_value is not None
            and requested_trade_value > resolved_liquidity.max_order_value
        ):
            warning_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)
        if resolved_max_trade_value is not None and requested_trade_value > resolved_max_trade_value:
            warning_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)
        if allowed_trade_value is not None and requested_trade_value > allowed_trade_value:
            if allowed_trade_value <= 0.0:
                if RISK_ISSUE_ADV_CAP_EXCEEDED in warning_reasons:
                    blocked_reasons.append(RISK_ISSUE_ADV_CAP_EXCEEDED)
                if RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED in warning_reasons:
                    blocked_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)
                if not blocked_reasons:
                    blocked_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)
            else:
                if not any(
                    reason in warning_reasons
                    for reason in [
                        RISK_ISSUE_ADV_CAP_EXCEEDED,
                        RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED,
                    ]
                ):
                    warning_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)

    if resolved_liquidity.liquidity_score is not None and resolved_liquidity.liquidity_score < 0.20:
        warning_reasons.append(RISK_ISSUE_LOW_LIQUIDITY)

    extra_tradability_reasons = [
        reason
        for reason in _extract_reasons(tradability_reasons)
        if reason not in _EXECUTION_REASON_ORDER
    ]
    if not is_tradable:
        blocked_reasons.extend(extra_tradability_reasons)
    else:
        warning_reasons.extend(extra_tradability_reasons)

    blocked_reasons = _ordered_unique(blocked_reasons, preferred_order=_EXECUTION_REASON_ORDER)
    warning_reasons = _ordered_unique(warning_reasons, preferred_order=_EXECUTION_REASON_ORDER)
    warning_reasons = [reason for reason in warning_reasons if reason not in blocked_reasons]

    if blocked_reasons:
        status = EXECUTION_BLOCKED
    elif (
        requested_trade_value is not None
        and allowed_trade_value is not None
        and requested_trade_value > allowed_trade_value
        and allowed_trade_value > 0.0
    ):
        status = EXECUTION_PARTIALLY_FEASIBLE
    else:
        status = EXECUTION_FEASIBLE

    estimated_slippage_bps = None
    if resolved_liquidity.estimated_spread_bps is not None:
        estimated_slippage_bps = resolved_liquidity.estimated_spread_bps / 2.0
    estimated_market_impact_bps = resolved_liquidity.estimated_market_impact_bps
    cost_parts = [
        value
        for value in [estimated_slippage_bps, estimated_market_impact_bps]
        if value is not None
    ]
    estimated_transaction_cost_bps = sum(cost_parts) if cost_parts else None

    execution_metadata = _coerce_metadata(metadata)
    execution_metadata.setdefault("risk_tensor_schema_version", RISK_TENSOR_SCHEMA_VERSION)
    execution_metadata.setdefault("allowed_trade_value_sources", {
        "max_order_value": resolved_liquidity.max_order_value,
        "adv_participation": adv_allowed,
        "max_trade_value": resolved_max_trade_value,
    })

    return ExecutionFeasibility(
        symbol=symbol,
        market=market,
        as_of=as_of,
        status=status,
        requested_weight=resolved_target_weight,
        current_weight=resolved_current_weight,
        requested_trade_value=requested_trade_value,
        allowed_trade_value=allowed_trade_value,
        adv_usage=adv_usage,
        max_order_value_usage=max_order_value_usage,
        blocked_reasons=blocked_reasons,
        warning_reasons=warning_reasons,
        estimated_transaction_cost_bps=estimated_transaction_cost_bps,
        estimated_slippage_bps=estimated_slippage_bps,
        estimated_market_impact_bps=estimated_market_impact_bps,
        metadata=execution_metadata,
    )


def build_symbol_risk_tensor(
    *,
    symbol: str,
    market: str,
    as_of: str,
    latest_trade_date: str,
    target_weight: float,
    current_weight: float = 0.0,
    data_quality_assessment: Any | None = None,
    tradability_status: Any | None = None,
    exposure: SymbolExposure | None = None,
    liquidity: LiquidityProfile | None = None,
    portfolio_value: float | None = None,
    max_weight: float | None = None,
    max_trade_value: float | None = None,
    stress_shocks: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SymbolRiskTensor:
    resolved_target_weight = _finite_float(target_weight, "target_weight")
    resolved_current_weight = _finite_float(current_weight, "current_weight")
    assessment = data_quality_assessment
    status_source = tradability_status

    data_quality_score = getattr(assessment, "data_quality_score", None)
    if data_quality_score is not None:
        data_quality_score = clamp_unit_interval(float(data_quality_score))

    assessment_issues = list(getattr(assessment, "issues", []) or [])
    has_data_blocker = any(_issue_severity(issue) == RISK_SEVERITY_BLOCKER for issue in assessment_issues)
    quarantine = bool(getattr(assessment, "quarantine", False)) or has_data_blocker
    is_researchable = bool(getattr(assessment, "is_researchable", True)) and not quarantine
    if status_source is not None and hasattr(status_source, "is_tradable"):
        is_tradable = bool(getattr(status_source, "is_tradable"))
    elif assessment is not None and hasattr(assessment, "is_tradable"):
        is_tradable = bool(getattr(assessment, "is_tradable"))
    else:
        is_tradable = True

    tradability_reasons = _extract_reasons(getattr(status_source, "reasons", None))
    if not tradability_reasons:
        tradability_reasons = _extract_reasons(getattr(assessment, "tradability_reasons", None))
    tradability_reasons = _ordered_unique(tradability_reasons, preferred_order=_EXECUTION_REASON_ORDER)

    resolved_exposure = exposure or SymbolExposure(
        symbol=symbol,
        market=market,
        as_of=as_of,
    )
    if liquidity is not None:
        resolved_liquidity = liquidity
    else:
        resolved_liquidity = LiquidityProfile(
            symbol=symbol,
            market=market,
            as_of=as_of,
            adv=_get_optional_float_attr(status_source, "adv"),
            liquidity_score=_get_optional_float_attr(status_source, "liquidity_score"),
            max_order_value=_get_optional_float_attr(status_source, "max_order_value"),
        )

    execution = build_execution_feasibility(
        symbol=symbol,
        market=market,
        as_of=as_of,
        target_weight=resolved_target_weight,
        current_weight=resolved_current_weight,
        portfolio_value=portfolio_value,
        liquidity=resolved_liquidity,
        is_tradable=is_tradable,
        tradability_reasons=tradability_reasons,
        max_weight=max_weight,
        max_trade_value=max_trade_value,
        metadata={"builder": "build_symbol_risk_tensor"},
    )

    issues: list[RiskIssue] = []
    if quarantine:
        quarantine_reasons = _extract_reasons(getattr(assessment, "quarantine_reasons", None))
        message = "Symbol is quarantined by data quality assessment."
        if quarantine_reasons:
            message = f"{message} Reasons: {'; '.join(quarantine_reasons)}"
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_DATA_QUARANTINE,
                severity=RISK_SEVERITY_BLOCKER,
                message=message,
                value=data_quality_score,
                limit=None,
                metadata={"quarantine_reasons": quarantine_reasons},
            )
        )

    if not is_tradable:
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_UNTRADABLE,
                severity=RISK_SEVERITY_BLOCKER,
                message="Symbol is not tradable under supplied tradability status.",
                metadata={"tradability_reasons": tradability_reasons},
            )
        )

    if resolved_liquidity.liquidity_score is not None and resolved_liquidity.liquidity_score < 0.20:
        severity = (
            RISK_SEVERITY_BLOCKER
            if resolved_liquidity.liquidity_score <= 0.0
            else RISK_SEVERITY_WARNING
        )
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_LOW_LIQUIDITY,
                severity=severity,
                message="Liquidity score is below the Phase 6 diagnostic threshold.",
                value=resolved_liquidity.liquidity_score,
                limit=0.20,
            )
        )

    execution_issue_map = {
        RISK_ISSUE_POSITION_TOO_LARGE,
        RISK_ISSUE_ADV_CAP_EXCEEDED,
        RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED,
    }
    for reason in execution.blocked_reasons:
        if reason not in execution_issue_map:
            continue
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=reason,
                severity=RISK_SEVERITY_BLOCKER,
                message=f"Execution feasibility blocked by {reason}.",
                value=execution.requested_trade_value,
                limit=execution.allowed_trade_value,
            )
        )
    for reason in execution.warning_reasons:
        if reason not in execution_issue_map:
            continue
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=reason,
                severity=RISK_SEVERITY_WARNING,
                message=f"Execution feasibility warning: {reason}.",
                value=execution.requested_trade_value,
                limit=execution.allowed_trade_value,
            )
        )

    stress_results: list[StressScenarioResult] = []
    for scenario_name in sorted(dict(stress_shocks or {})):
        shock_return = _finite_float(dict(stress_shocks or {})[scenario_name], f"stress_shocks.{scenario_name}")
        estimated_loss = max(0.0, -shock_return * resolved_target_weight)
        stress_results.append(
            StressScenarioResult(
                scenario_name=str(scenario_name),
                symbol=symbol,
                market=market,
                as_of=as_of,
                shock_return=shock_return,
                position_weight=resolved_target_weight,
                estimated_loss=estimated_loss,
                metadata={"loss_convention": "positive_loss"},
            )
        )

    sorted_issues = _sort_issues(issues)
    severity_penalty = 0.0
    for issue in sorted_issues:
        if issue.severity == RISK_SEVERITY_BLOCKER:
            severity_penalty += 0.50
        elif issue.severity == RISK_SEVERITY_WARNING:
            severity_penalty += 0.20
        else:
            severity_penalty += 0.05
    max_stress_loss = max([result.estimated_loss for result in stress_results], default=0.0)
    risk_score = clamp_unit_interval(severity_penalty + min(0.50, max_stress_loss))

    tensor_metadata = _coerce_metadata(metadata)
    tensor_metadata.setdefault("risk_tensor_schema_version", RISK_TENSOR_SCHEMA_VERSION)
    if data_quality_score is not None:
        tensor_metadata.setdefault("data_quality_score_source", "data_quality_assessment")

    return SymbolRiskTensor(
        tensor_id=make_symbol_tensor_id(
            symbol=symbol,
            market=market,
            as_of=as_of,
            latest_trade_date=latest_trade_date,
        ),
        symbol=symbol,
        market=market,
        as_of=as_of,
        latest_trade_date=latest_trade_date,
        target_weight=resolved_target_weight,
        current_weight=resolved_current_weight,
        data_quality_score=data_quality_score,
        is_researchable=is_researchable,
        is_tradable=is_tradable,
        quarantine=quarantine,
        exposure=resolved_exposure,
        liquidity=resolved_liquidity,
        execution=execution,
        stress_results=stress_results,
        issues=sorted_issues,
        risk_score=risk_score,
        metadata=tensor_metadata,
    )


def _add_weight(target: dict[str, float], key: str, value: float) -> None:
    target[key] = target.get(key, 0.0) + value


def _aggregate_exposures(
    tensors: Sequence[SymbolRiskTensor],
    exposure_name: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for tensor in tensors:
        exposures = getattr(tensor.exposure, exposure_name)
        for name, exposure_value in exposures.items():
            _add_weight(result, name, tensor.target_weight * exposure_value)
    return {key: result[key] for key in sorted(result)}


def build_portfolio_risk_tensor(
    *,
    symbol_tensors: Sequence[SymbolRiskTensor],
    market: str,
    as_of: str,
    turnover_estimate: float = 0.0,
    sector_cap: float | None = None,
    gross_exposure_cap: float | None = None,
    max_weight: float | None = None,
    turnover_cap: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PortfolioRiskTensor:
    ordered_tensors = sorted(list(symbol_tensors), key=lambda tensor: (tensor.symbol, tensor.market))
    resolved_turnover = _non_negative_float_or_none(turnover_estimate, "turnover_estimate") or 0.0
    resolved_sector_cap = _non_negative_float_or_none(sector_cap, "sector_cap")
    resolved_gross_cap = _non_negative_float_or_none(gross_exposure_cap, "gross_exposure_cap")
    resolved_max_weight = _non_negative_float_or_none(max_weight, "max_weight")
    resolved_turnover_cap = _non_negative_float_or_none(turnover_cap, "turnover_cap")

    weights = [tensor.target_weight for tensor in ordered_tensors]
    gross_exposure = sum(abs(weight) for weight in weights)
    net_exposure = sum(weights)
    long_exposure = sum(max(weight, 0.0) for weight in weights)
    short_exposure = sum(min(weight, 0.0) for weight in weights)

    sector_weights: dict[str, float] = {}
    correlation_cluster_weights: dict[str, float] = {}
    for tensor in ordered_tensors:
        sector = tensor.exposure.sector or "UNKNOWN"
        _add_weight(sector_weights, sector, tensor.target_weight)
        cluster = tensor.exposure.correlation_cluster or "UNKNOWN"
        _add_weight(correlation_cluster_weights, cluster, abs(tensor.target_weight))
    sector_weights = {key: sector_weights[key] for key in sorted(sector_weights)}
    correlation_cluster_weights = {
        key: correlation_cluster_weights[key]
        for key in sorted(correlation_cluster_weights)
    }

    style_exposures = _aggregate_exposures(ordered_tensors, "style_exposures")
    factor_exposures = _aggregate_exposures(ordered_tensors, "factor_exposures")

    beta_values = [
        tensor.target_weight * tensor.exposure.beta
        for tensor in ordered_tensors
        if tensor.exposure.beta is not None
    ]
    beta_exposure = sum(beta_values) if beta_values else None

    blocked_symbols = _ordered_unique(
        [
            tensor.symbol
            for tensor in ordered_tensors
            if tensor.execution.status == EXECUTION_BLOCKED
            or any(issue.severity == RISK_SEVERITY_BLOCKER for issue in tensor.issues)
        ]
    )
    blocked_set = set(blocked_symbols)
    max_weight_by_symbol = {
        tensor.symbol: (
            0.0
            if tensor.symbol in blocked_set
            else resolved_max_weight
            if resolved_max_weight is not None
            else abs(tensor.target_weight)
        )
        for tensor in ordered_tensors
    }

    portfolio_issues: list[RiskIssue] = []
    if resolved_gross_cap is not None and gross_exposure > resolved_gross_cap:
        portfolio_issues.append(
            _make_issue(
                symbol=None,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_POSITION_TOO_LARGE,
                severity=RISK_SEVERITY_BLOCKER,
                message="Gross exposure exceeds configured Phase 6 cap.",
                value=gross_exposure,
                limit=resolved_gross_cap,
            )
        )
    if resolved_turnover_cap is not None and resolved_turnover > resolved_turnover_cap:
        portfolio_issues.append(
            _make_issue(
                symbol=None,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_TURNOVER_EXCEEDED,
                severity=RISK_SEVERITY_WARNING,
                message="Turnover estimate exceeds configured Phase 6 cap.",
                value=resolved_turnover,
                limit=resolved_turnover_cap,
            )
        )
    if resolved_sector_cap is not None:
        for sector, weight in sector_weights.items():
            if abs(weight) <= resolved_sector_cap:
                continue
            portfolio_issues.append(
                _make_issue(
                    symbol=None,
                    market=market,
                    as_of=as_of,
                    issue_type=RISK_ISSUE_SECTOR_CONCENTRATION,
                    severity=RISK_SEVERITY_WARNING,
                    message=f"Sector '{sector}' exceeds configured Phase 6 cap.",
                    value=abs(weight),
                    limit=resolved_sector_cap,
                    metadata={"sector": sector},
                )
            )
    if beta_exposure is not None and abs(beta_exposure) > 1.5:
        portfolio_issues.append(
            _make_issue(
                symbol=None,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_BETA_EXPOSURE,
                severity=RISK_SEVERITY_WARNING,
                message="Portfolio beta exposure exceeds Phase 6 diagnostic threshold.",
                value=abs(beta_exposure),
                limit=1.5,
            )
        )
    for cluster, weight in correlation_cluster_weights.items():
        if abs(weight) <= 0.50:
            continue
        portfolio_issues.append(
            _make_issue(
                symbol=None,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_CORRELATION_CLUSTER,
                severity=RISK_SEVERITY_WARNING,
                message=f"Correlation cluster '{cluster}' exceeds Phase 6 diagnostic threshold.",
                value=abs(weight),
                limit=0.50,
                metadata={"correlation_cluster": cluster},
            )
        )

    avg_symbol_risk = (
        sum(tensor.risk_score for tensor in ordered_tensors) / len(ordered_tensors)
        if ordered_tensors
        else 0.0
    )
    issue_penalty = 0.0
    for issue in portfolio_issues:
        if issue.severity == RISK_SEVERITY_BLOCKER:
            issue_penalty += 0.25
        elif issue.severity == RISK_SEVERITY_WARNING:
            issue_penalty += 0.10
        else:
            issue_penalty += 0.03
    risk_score = clamp_unit_interval(avg_symbol_risk + issue_penalty)

    tensor_metadata = _coerce_metadata(metadata)
    tensor_metadata.setdefault("risk_tensor_schema_version", RISK_TENSOR_SCHEMA_VERSION)

    return PortfolioRiskTensor(
        tensor_id=make_portfolio_tensor_id(
            market=market,
            as_of=as_of,
            symbol_tensor_ids=[tensor.tensor_id for tensor in ordered_tensors],
        ),
        as_of=as_of,
        market=market,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        turnover_estimate=resolved_turnover,
        sector_weights=sector_weights,
        style_exposures=style_exposures,
        factor_exposures=factor_exposures,
        beta_exposure=beta_exposure,
        correlation_cluster_weights=correlation_cluster_weights,
        symbol_tensors=ordered_tensors,
        portfolio_issues=_sort_issues(portfolio_issues),
        blocked_symbols=blocked_symbols,
        max_weight_by_symbol=max_weight_by_symbol,
        gross_exposure_cap=resolved_gross_cap,
        risk_score=risk_score,
        metadata=tensor_metadata,
    )


def build_execution_feasibility_report(
    *,
    symbol_tensors: Sequence[SymbolRiskTensor],
    market: str,
    as_of: str,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionFeasibilityReport:
    ordered_tensors = sorted(list(symbol_tensors), key=lambda tensor: (tensor.symbol, tensor.market))
    feasible_symbols: list[str] = []
    partially_feasible_symbols: list[str] = []
    blocked_symbols: list[str] = []
    execution_by_symbol: dict[str, ExecutionFeasibility] = {}
    issues: list[RiskIssue] = []

    requested_values: list[float] = []
    allowed_values: list[float] = []
    weighted_cost_numerator = 0.0
    weighted_cost_denominator = 0.0
    simple_costs: list[float] = []

    for tensor in ordered_tensors:
        execution = tensor.execution
        execution_by_symbol[tensor.symbol] = execution
        if execution.status == EXECUTION_BLOCKED:
            blocked_symbols.append(tensor.symbol)
            issues.extend(tensor.issues)
        elif execution.status == EXECUTION_PARTIALLY_FEASIBLE:
            partially_feasible_symbols.append(tensor.symbol)
            issues.extend(tensor.issues)
        else:
            feasible_symbols.append(tensor.symbol)

        if execution.requested_trade_value is not None:
            requested_values.append(execution.requested_trade_value)
        if execution.allowed_trade_value is not None:
            allowed_values.append(execution.allowed_trade_value)
        if execution.estimated_transaction_cost_bps is not None:
            simple_costs.append(execution.estimated_transaction_cost_bps)
            if execution.requested_trade_value is not None and execution.requested_trade_value > 0.0:
                weighted_cost_numerator += (
                    execution.estimated_transaction_cost_bps * execution.requested_trade_value
                )
                weighted_cost_denominator += execution.requested_trade_value

    if weighted_cost_denominator > 0.0:
        aggregate_estimated_cost_bps = weighted_cost_numerator / weighted_cost_denominator
    elif simple_costs:
        aggregate_estimated_cost_bps = sum(simple_costs) / len(simple_costs)
    else:
        aggregate_estimated_cost_bps = None

    report_metadata = _coerce_metadata(metadata)
    report_metadata.setdefault("risk_tensor_schema_version", RISK_TENSOR_SCHEMA_VERSION)

    return ExecutionFeasibilityReport(
        report_id=make_execution_report_id(
            market=market,
            as_of=as_of,
            symbol_tensor_ids=[tensor.tensor_id for tensor in ordered_tensors],
        ),
        as_of=as_of,
        market=market,
        total_symbols=len(ordered_tensors),
        feasible_symbols=feasible_symbols,
        partially_feasible_symbols=partially_feasible_symbols,
        blocked_symbols=blocked_symbols,
        execution_by_symbol=execution_by_symbol,
        total_requested_trade_value=sum(requested_values) if requested_values else None,
        total_allowed_trade_value=sum(allowed_values) if allowed_values else None,
        aggregate_estimated_cost_bps=aggregate_estimated_cost_bps,
        issues=_sort_issues(issues),
        metadata=report_metadata,
    )


def build_risk_guard_context_patch(
    portfolio_tensor: PortfolioRiskTensor,
    execution_report: ExecutionFeasibilityReport | None = None,
) -> dict[str, Any]:
    if execution_report is not None:
        execution_status_by_symbol = {
            symbol: execution.status
            for symbol, execution in sorted(execution_report.execution_by_symbol.items())
        }
        execution_blocked_symbols = list(execution_report.blocked_symbols)
        execution_partially_feasible_symbols = list(execution_report.partially_feasible_symbols)
        execution_report_id = execution_report.report_id
    else:
        execution_status_by_symbol = {
            tensor.symbol: tensor.execution.status
            for tensor in sorted(portfolio_tensor.symbol_tensors, key=lambda item: (item.symbol, item.market))
        }
        execution_blocked_symbols = _ordered_unique(
            [
                symbol
                for symbol, status in execution_status_by_symbol.items()
                if status == EXECUTION_BLOCKED
            ]
        )
        execution_partially_feasible_symbols = _ordered_unique(
            [
                symbol
                for symbol, status in execution_status_by_symbol.items()
                if status == EXECUTION_PARTIALLY_FEASIBLE
            ]
        )
        execution_report_id = None

    all_issues = list(portfolio_tensor.portfolio_issues)
    for tensor in portfolio_tensor.symbol_tensors:
        all_issues.extend(tensor.issues)
    risk_issues = [issue.to_dict() for issue in _sort_issues(all_issues)]
    metadata = {
        "risk_tensor_schema_version": RISK_TENSOR_SCHEMA_VERSION,
        "tensor_id": portfolio_tensor.tensor_id,
        "execution_report_id": execution_report_id,
        "blocked_symbol_count": len(portfolio_tensor.blocked_symbols),
        "issue_count": len(risk_issues),
    }
    payload = {
        "blocked_symbols": list(portfolio_tensor.blocked_symbols),
        "max_weight_by_symbol": dict(portfolio_tensor.max_weight_by_symbol),
        "gross_exposure_cap": portfolio_tensor.gross_exposure_cap,
        "risk_score": portfolio_tensor.risk_score,
        "risk_issues": risk_issues,
        "execution_status_by_symbol": execution_status_by_symbol,
        "execution_blocked_symbols": execution_blocked_symbols,
        "execution_partially_feasible_symbols": execution_partially_feasible_symbols,
        "metadata": metadata,
    }
    return dict(_ensure_json_serializable(payload, "risk_guard_context_patch"))


RecordT = TypeVar("RecordT", SymbolRiskTensor, PortfolioRiskTensor, ExecutionFeasibilityReport)


class RiskTensorStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_RISK_TENSOR_DIR
        self.symbol_tensors_path = self.root_dir / DEFAULT_SYMBOL_TENSORS_FILENAME
        self.portfolio_tensors_path = self.root_dir / DEFAULT_PORTFOLIO_TENSORS_FILENAME
        self.execution_reports_path = self.root_dir / DEFAULT_EXECUTION_REPORTS_FILENAME

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(_json_safe(payload)), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _read_jsonl(self, path: Path, record_cls: type[RecordT]) -> list[RecordT]:
        if not path.exists():
            return []
        records: list[RecordT] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path} line {line_number}: {exc.msg}") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"Expected JSON object in {path} line {line_number}.")
                records.append(record_cls.from_dict(payload))
        return records

    def append_symbol_tensor(self, tensor: SymbolRiskTensor) -> None:
        if tensor.tensor_id in self.get_symbol_tensor_ids():
            raise ValueError(f"Duplicate tensor_id in symbol tensor ledger: {tensor.tensor_id}")
        self._append_jsonl(self.symbol_tensors_path, tensor.to_dict())

    def append_symbol_tensors(self, tensors: Sequence[SymbolRiskTensor]) -> int:
        existing = self.get_symbol_tensor_ids()
        seen: set[str] = set()
        for tensor in tensors:
            if tensor.tensor_id in existing or tensor.tensor_id in seen:
                raise ValueError(f"Duplicate tensor_id in symbol tensor ledger: {tensor.tensor_id}")
            seen.add(tensor.tensor_id)
        for tensor in tensors:
            self._append_jsonl(self.symbol_tensors_path, tensor.to_dict())
        return len(tensors)

    def append_portfolio_tensor(self, tensor: PortfolioRiskTensor) -> None:
        if tensor.tensor_id in self.get_portfolio_tensor_ids():
            raise ValueError(f"Duplicate tensor_id in portfolio tensor ledger: {tensor.tensor_id}")
        self._append_jsonl(self.portfolio_tensors_path, tensor.to_dict())

    def append_execution_report(self, report: ExecutionFeasibilityReport) -> None:
        if report.report_id in self.get_execution_report_ids():
            raise ValueError(f"Duplicate report_id in execution report ledger: {report.report_id}")
        self._append_jsonl(self.execution_reports_path, report.to_dict())

    def read_symbol_tensors(self) -> list[SymbolRiskTensor]:
        return self._read_jsonl(self.symbol_tensors_path, SymbolRiskTensor)

    def read_portfolio_tensors(self) -> list[PortfolioRiskTensor]:
        return self._read_jsonl(self.portfolio_tensors_path, PortfolioRiskTensor)

    def read_execution_reports(self) -> list[ExecutionFeasibilityReport]:
        return self._read_jsonl(self.execution_reports_path, ExecutionFeasibilityReport)

    def get_symbol_tensor_ids(self) -> set[str]:
        return {tensor.tensor_id for tensor in self.read_symbol_tensors()}

    def get_portfolio_tensor_ids(self) -> set[str]:
        return {tensor.tensor_id for tensor in self.read_portfolio_tensors()}

    def get_execution_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_execution_reports()}


__all__ = [
    "DEFAULT_RISK_TENSOR_DIR",
    "DEFAULT_SYMBOL_TENSORS_FILENAME",
    "DEFAULT_PORTFOLIO_TENSORS_FILENAME",
    "DEFAULT_EXECUTION_REPORTS_FILENAME",
    "RISK_SEVERITY_INFO",
    "RISK_SEVERITY_WARNING",
    "RISK_SEVERITY_BLOCKER",
    "RISK_ISSUE_DATA_QUARANTINE",
    "RISK_ISSUE_UNTRADABLE",
    "RISK_ISSUE_LOW_LIQUIDITY",
    "RISK_ISSUE_POSITION_TOO_LARGE",
    "RISK_ISSUE_ADV_CAP_EXCEEDED",
    "RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED",
    "RISK_ISSUE_SECTOR_CONCENTRATION",
    "RISK_ISSUE_STYLE_EXPOSURE",
    "RISK_ISSUE_BETA_EXPOSURE",
    "RISK_ISSUE_CORRELATION_CLUSTER",
    "RISK_ISSUE_STRESS_LOSS",
    "RISK_ISSUE_TURNOVER_EXCEEDED",
    "EXECUTION_FEASIBLE",
    "EXECUTION_PARTIALLY_FEASIBLE",
    "EXECUTION_BLOCKED",
    "RiskIssue",
    "SymbolExposure",
    "LiquidityProfile",
    "ExecutionFeasibility",
    "StressScenarioResult",
    "SymbolRiskTensor",
    "PortfolioRiskTensor",
    "ExecutionFeasibilityReport",
    "RiskTensorStore",
    "make_risk_issue_id",
    "make_symbol_tensor_id",
    "make_portfolio_tensor_id",
    "make_execution_report_id",
    "clamp_unit_interval",
    "validate_finite_number",
    "bps_to_decimal_return",
    "weighted_average",
    "build_execution_feasibility",
    "build_symbol_risk_tensor",
    "build_portfolio_risk_tensor",
    "build_execution_feasibility_report",
    "build_risk_guard_context_patch",
]
