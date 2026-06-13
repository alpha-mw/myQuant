"""Offline structured risk tensor and execution feasibility contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    "make_risk_issue_id",
    "make_symbol_tensor_id",
    "make_portfolio_tensor_id",
    "make_execution_report_id",
    "clamp_unit_interval",
    "validate_finite_number",
    "bps_to_decimal_return",
    "weighted_average",
]
