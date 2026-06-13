"""Offline deterministic portfolio optimizer and walk-forward contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.versioning import PORTFOLIO_OPTIMIZER_SCHEMA_VERSION


DEFAULT_PORTFOLIO_OPTIMIZER_DIR = Path("data/portfolio_optimizer")
DEFAULT_OPTIMIZED_PLANS_FILENAME = "optimized_plans.jsonl"
DEFAULT_REBALANCE_RESULTS_FILENAME = "rebalance_results.jsonl"
DEFAULT_WALK_FORWARD_RESULTS_FILENAME = "walk_forward_results.jsonl"

PLAN_STATUS_OPTIMIZED = "optimized"
PLAN_STATUS_EMPTY = "empty"
PLAN_STATUS_INFEASIBLE = "infeasible"

CONSTRAINT_BLOCKED_SYMBOL = "blocked_symbol"
CONSTRAINT_MIN_EDGE = "min_edge"
CONSTRAINT_MAX_WEIGHT = "max_weight"
CONSTRAINT_GROSS_EXPOSURE = "gross_exposure"
CONSTRAINT_SECTOR_CAP = "sector_cap"
CONSTRAINT_TURNOVER_CAP = "turnover_cap"
CONSTRAINT_MAX_NAMES = "max_names"
CONSTRAINT_RISK_SCORE = "risk_score"

VIOLATION_INFO = "info"
VIOLATION_WARNING = "warning"
VIOLATION_BLOCKER = "blocker"

_VALID_PLAN_STATUSES = {
    PLAN_STATUS_OPTIMIZED,
    PLAN_STATUS_EMPTY,
    PLAN_STATUS_INFEASIBLE,
}
_VALID_VIOLATION_SEVERITIES = {
    VIOLATION_INFO,
    VIOLATION_WARNING,
    VIOLATION_BLOCKER,
}
_EPSILON = 1e-12
_UNKNOWN_SECTOR = "UNKNOWN"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
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


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return sorted({str(value) for value in values})


def _slug(value: str | None) -> str:
    resolved = "none" if value is None else str(value).strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved)
    return slug.strip("-") or "unknown"


def _short_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps([_json_safe(part) for part in parts], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    validate_finite_number(number, field_name=field_name)
    return number


def _finite_float(value: Any, field_name: str) -> float:
    number = float(value)
    validate_finite_number(number, field_name=field_name)
    return number


def _non_negative_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _non_negative_float_or_none(value: Any, field_name: str) -> float | None:
    number = _float_or_none(value, field_name)
    if number is not None and number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    number = _float_or_none(value, field_name)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _finite_float_dict(value: Mapping[str, Any] | None, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    result: dict[str, float] = {}
    for key in sorted(value, key=str):
        result[str(key)] = _finite_float(value[key], f"{field_name}.{key}")
    return result


def _clean_weight_dict(value: Mapping[str, float], *, keep_zero: bool = False) -> dict[str, float]:
    result: dict[str, float] = {}
    for symbol in sorted(value):
        weight = _finite_float(value[symbol], f"weights.{symbol}")
        if keep_zero or abs(weight) > _EPSILON:
            result[str(symbol)] = 0.0 if abs(weight) <= _EPSILON else weight
    return result


def _get_attr(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _get_nested_attr(source: Any, path: Sequence[str], default: Any = None) -> Any:
    value = source
    for part in path:
        value = _get_attr(value, part, default)
        if value is default:
            return default
    return value


def _metadata_float(metadata: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key not in metadata or metadata[key] is None:
            continue
        return _finite_float(metadata[key], key)
    return None


def _candidate_adjusted_score(candidate: "OptimizationCandidate", config: "PortfolioOptimizerConfig") -> float:
    risk_multiplier = max(0.0, 1.0 - config.risk_score_penalty_weight * candidate.risk_score)
    return candidate.edge_after_costs * max(candidate.confidence, 0.0) * risk_multiplier


def _make_violation(
    *,
    symbol: str | None,
    constraint_type: str,
    severity: str,
    value: float | None,
    limit: float | None,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> "ConstraintViolation":
    return ConstraintViolation(
        violation_id=make_constraint_violation_id(
            symbol=symbol,
            constraint_type=constraint_type,
            message=message,
        ),
        symbol=symbol,
        constraint_type=constraint_type,
        severity=severity,
        value=value,
        limit=limit,
        message=message,
        metadata=_coerce_metadata(metadata),
    )


def _sort_violations(violations: Sequence["ConstraintViolation"]) -> list["ConstraintViolation"]:
    return sorted(
        violations,
        key=lambda violation: (
            violation.symbol or "",
            violation.constraint_type,
            violation.severity,
            violation.message,
            violation.violation_id,
        ),
    )


def clamp_unit_interval(value: float) -> float:
    validate_finite_number(value, field_name="value")
    return max(0.0, min(1.0, float(value)))


def validate_finite_number(value: float, *, field_name: str) -> None:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")


def bps_to_decimal_return(value_bps: float) -> float:
    bps = _non_negative_float(value_bps, "value_bps")
    return bps / 10000.0


def estimate_turnover(current_weights: Mapping[str, float], target_weights: Mapping[str, float]) -> float:
    symbols = sorted(set(current_weights) | set(target_weights))
    turnover = 0.0
    for symbol in symbols:
        current = _finite_float(current_weights.get(symbol, 0.0), f"current_weights.{symbol}")
        target = _finite_float(target_weights.get(symbol, 0.0), f"target_weights.{symbol}")
        turnover += abs(target - current)
    return turnover


def compute_sector_weights(
    target_weights: Mapping[str, float],
    sector_by_symbol: Mapping[str, str | None],
) -> dict[str, float]:
    sector_weights: dict[str, float] = {}
    for symbol in sorted(target_weights):
        weight = _finite_float(target_weights[symbol], f"target_weights.{symbol}")
        sector = sector_by_symbol.get(symbol) or _UNKNOWN_SECTOR
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
    return {sector: sector_weights[sector] for sector in sorted(sector_weights)}


def compound_returns(returns: Sequence[float]) -> float:
    equity = 1.0
    for index, value in enumerate(returns):
        period_return = _finite_float(value, f"returns.{index}")
        equity *= 1.0 + period_return
    return equity - 1.0


def max_drawdown_from_returns(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for index, value in enumerate(returns):
        period_return = _finite_float(value, f"returns.{index}")
        equity *= 1.0 + period_return
        if equity > peak:
            peak = equity
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return max(0.0, max_drawdown)


def make_constraint_violation_id(*, symbol: str | None, constraint_type: str, message: str) -> str:
    return f"opt-violation-{_slug(symbol)}-{_slug(constraint_type)}-{_short_hash([symbol, constraint_type, message])}"


def make_plan_id(
    *,
    market: str,
    as_of: str,
    symbols: Sequence[str],
    config_hash: str | None = None,
) -> str:
    ordered_symbols = sorted(str(symbol) for symbol in symbols)
    parts = [market, as_of, ordered_symbols, config_hash or ""]
    return f"portfolio-plan-{_slug(market)}-{_slug(as_of)}-{len(ordered_symbols)}-{_short_hash(parts)}"


def make_rebalance_id(*, plan_id: str, evaluation_date: str) -> str:
    return f"rebalance-{_slug(evaluation_date)}-{_short_hash([plan_id, evaluation_date])}"


def make_walk_forward_run_id(
    *,
    market: str,
    start_date: str,
    end_date: str,
    rebalance_dates: Sequence[str],
) -> str:
    ordered_dates = [str(date) for date in rebalance_dates]
    return f"walk-forward-{_slug(market)}-{_slug(start_date)}-{_slug(end_date)}-{len(ordered_dates)}-{_short_hash([market, start_date, end_date, ordered_dates])}"


@dataclass
class PortfolioOptimizerConfig:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    max_weight: float = 0.10
    min_weight: float = 0.0
    gross_exposure_cap: float = 1.0
    sector_cap: float | None = 0.30
    turnover_cap: float | None = 0.30
    max_names: int | None = None
    min_edge_after_costs: float = 0.0
    max_risk_score: float | None = 0.80
    cash_buffer: float = 0.0
    allow_short: bool = False
    force_exit_blocked_symbols: bool = True
    risk_score_penalty_weight: float = 0.50
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_weight = _finite_float(self.max_weight, "max_weight")
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1].")
        self.min_weight = _finite_float(self.min_weight, "min_weight")
        if not 0.0 <= self.min_weight <= self.max_weight:
            raise ValueError("min_weight must be in [0, max_weight].")
        self.gross_exposure_cap = _finite_float(self.gross_exposure_cap, "gross_exposure_cap")
        if not 0.0 < self.gross_exposure_cap <= 1.0:
            raise ValueError("gross_exposure_cap must be in (0, 1].")
        self.sector_cap = _unit_float_or_none(self.sector_cap, "sector_cap")
        if self.sector_cap is not None and self.sector_cap <= 0.0:
            raise ValueError("sector_cap must be in (0, 1] when provided.")
        self.turnover_cap = _non_negative_float_or_none(self.turnover_cap, "turnover_cap")
        if self.max_names is not None:
            self.max_names = int(self.max_names)
            if self.max_names <= 0:
                raise ValueError("max_names must be positive when provided.")
        self.min_edge_after_costs = _finite_float(self.min_edge_after_costs, "min_edge_after_costs")
        self.max_risk_score = _unit_float_or_none(self.max_risk_score, "max_risk_score")
        self.cash_buffer = _finite_float(self.cash_buffer, "cash_buffer")
        if not 0.0 <= self.cash_buffer < self.gross_exposure_cap:
            raise ValueError("cash_buffer must be in [0, gross_exposure_cap).")
        if self.allow_short:
            raise ValueError("allow_short=True is not implemented in Phase 7; only long-only mode is supported.")
        self.risk_score_penalty_weight = _non_negative_float(
            self.risk_score_penalty_weight,
            "risk_score_penalty_weight",
        )
        self.transaction_cost_bps = _non_negative_float(self.transaction_cost_bps, "transaction_cost_bps")
        self.slippage_bps = _non_negative_float(self.slippage_bps, "slippage_bps")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioOptimizerConfig":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            max_weight=float(data.get("max_weight", 0.10)),
            min_weight=float(data.get("min_weight", 0.0)),
            gross_exposure_cap=float(data.get("gross_exposure_cap", 1.0)),
            sector_cap=data.get("sector_cap", 0.30),
            turnover_cap=data.get("turnover_cap", 0.30),
            max_names=data.get("max_names"),
            min_edge_after_costs=float(data.get("min_edge_after_costs", 0.0)),
            max_risk_score=data.get("max_risk_score", 0.80),
            cash_buffer=float(data.get("cash_buffer", 0.0)),
            allow_short=bool(data.get("allow_short", False)),
            force_exit_blocked_symbols=bool(data.get("force_exit_blocked_symbols", True)),
            risk_score_penalty_weight=float(data.get("risk_score_penalty_weight", 0.50)),
            transaction_cost_bps=float(data.get("transaction_cost_bps", 0.0)),
            slippage_bps=float(data.get("slippage_bps", 0.0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class OptimizationCandidate:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    company_name: str = ""
    sector: str | None = None
    current_weight: float = 0.0
    max_weight: float | None = None
    expected_alpha: float = 0.0
    edge_after_costs: float = 0.0
    confidence: float = 0.0
    action_score: float = 0.0
    risk_score: float = 0.0
    liquidity_score: float | None = None
    estimated_transaction_cost_bps: float | None = None
    estimated_slippage_bps: float | None = None
    estimated_market_impact_bps: float | None = None
    is_blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol)
        self.market = str(self.market)
        self.as_of = str(self.as_of)
        self.company_name = str(self.company_name)
        self.sector = None if self.sector is None else str(self.sector)
        self.current_weight = _finite_float(self.current_weight, "current_weight")
        self.max_weight = _non_negative_float_or_none(self.max_weight, "max_weight")
        self.expected_alpha = _finite_float(self.expected_alpha, "expected_alpha")
        self.edge_after_costs = _finite_float(self.edge_after_costs, "edge_after_costs")
        self.confidence = clamp_unit_interval(float(self.confidence))
        self.action_score = _finite_float(self.action_score, "action_score")
        self.risk_score = clamp_unit_interval(float(self.risk_score))
        self.liquidity_score = _unit_float_or_none(self.liquidity_score, "liquidity_score")
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
        self.block_reasons = _ordered_unique(self.block_reasons)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizationCandidate":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            company_name=str(data.get("company_name", "")),
            sector=data.get("sector"),
            current_weight=float(data.get("current_weight", 0.0)),
            max_weight=data.get("max_weight"),
            expected_alpha=float(data.get("expected_alpha", 0.0)),
            edge_after_costs=float(data.get("edge_after_costs", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            action_score=float(data.get("action_score", 0.0)),
            risk_score=float(data.get("risk_score", 0.0)),
            liquidity_score=data.get("liquidity_score"),
            estimated_transaction_cost_bps=data.get("estimated_transaction_cost_bps"),
            estimated_slippage_bps=data.get("estimated_slippage_bps"),
            estimated_market_impact_bps=data.get("estimated_market_impact_bps"),
            is_blocked=bool(data.get("is_blocked", False)),
            block_reasons=list(data.get("block_reasons", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ConstraintViolation:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    violation_id: str = ""
    symbol: str | None = None
    constraint_type: str = ""
    severity: str = VIOLATION_WARNING
    value: float | None = None
    limit: float | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_VIOLATION_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_VALID_VIOLATION_SEVERITIES)}; got {self.severity!r}.")
        self.symbol = None if self.symbol is None else str(self.symbol)
        self.value = _float_or_none(self.value, "value")
        self.limit = _float_or_none(self.limit, "limit")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConstraintViolation":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            violation_id=str(data.get("violation_id", "")),
            symbol=data.get("symbol"),
            constraint_type=str(data.get("constraint_type", "")),
            severity=str(data.get("severity", VIOLATION_WARNING)),
            value=data.get("value"),
            limit=data.get("limit"),
            message=str(data.get("message", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class OptimizedPortfolioPlan:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    plan_id: str = ""
    as_of: str = ""
    market: str = ""
    status: str = PLAN_STATUS_EMPTY
    objective_value: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    trade_weights: dict[str, float] = field(default_factory=dict)
    selected_symbols: list[str] = field(default_factory=list)
    blocked_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)
    cash_weight: float = 1.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    turnover_estimate: float = 0.0
    sector_weights: dict[str, float] = field(default_factory=dict)
    violations: list[ConstraintViolation] = field(default_factory=list)
    candidate_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_PLAN_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_PLAN_STATUSES)}; got {self.status!r}.")
        self.objective_value = _finite_float(self.objective_value, "objective_value")
        self.target_weights = _clean_weight_dict(self.target_weights)
        self.current_weights = _clean_weight_dict(self.current_weights, keep_zero=True)
        self.trade_weights = _clean_weight_dict(self.trade_weights)
        if self.target_weights:
            self.selected_symbols = sorted(
                self.target_weights,
                key=lambda symbol: (-self.target_weights[symbol], symbol),
            )
        else:
            self.selected_symbols = sorted({str(symbol) for symbol in self.selected_symbols})
        self.blocked_symbols = _ordered_unique(self.blocked_symbols)
        self.rejected_symbols = _ordered_unique(self.rejected_symbols)
        self.cash_weight = _non_negative_float(self.cash_weight, "cash_weight")
        self.gross_exposure = _non_negative_float(self.gross_exposure, "gross_exposure")
        self.net_exposure = _finite_float(self.net_exposure, "net_exposure")
        self.long_exposure = _non_negative_float(self.long_exposure, "long_exposure")
        self.turnover_estimate = _non_negative_float(self.turnover_estimate, "turnover_estimate")
        self.sector_weights = _finite_float_dict(self.sector_weights, "sector_weights")
        self.violations = [
            violation if isinstance(violation, ConstraintViolation) else ConstraintViolation.from_dict(violation)
            for violation in self.violations
        ]
        self.violations = _sort_violations(self.violations)
        if self.candidate_count < 0:
            raise ValueError("candidate_count must be non-negative.")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(_json_safe(asdict(self)))
        payload["violations"] = [violation.to_dict() for violation in self.violations]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizedPortfolioPlan":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            plan_id=str(data.get("plan_id", "")),
            as_of=str(data.get("as_of", "")),
            market=str(data.get("market", "")),
            status=str(data.get("status", PLAN_STATUS_EMPTY)),
            objective_value=float(data.get("objective_value", 0.0)),
            target_weights=dict(data.get("target_weights", {}) or {}),
            current_weights=dict(data.get("current_weights", {}) or {}),
            trade_weights=dict(data.get("trade_weights", {}) or {}),
            selected_symbols=list(data.get("selected_symbols", []) or []),
            blocked_symbols=list(data.get("blocked_symbols", []) or []),
            rejected_symbols=list(data.get("rejected_symbols", []) or []),
            cash_weight=float(data.get("cash_weight", 1.0)),
            gross_exposure=float(data.get("gross_exposure", 0.0)),
            net_exposure=float(data.get("net_exposure", 0.0)),
            long_exposure=float(data.get("long_exposure", 0.0)),
            turnover_estimate=float(data.get("turnover_estimate", 0.0)),
            sector_weights=dict(data.get("sector_weights", {}) or {}),
            violations=[
                ConstraintViolation.from_dict(violation)
                for violation in list(data.get("violations", []) or [])
                if isinstance(violation, Mapping)
            ],
            candidate_count=int(data.get("candidate_count", 0) or 0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class RebalanceInput:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    as_of: str = ""
    evaluation_date: str = ""
    market: str = ""
    candidates: list[OptimizationCandidate] = field(default_factory=list)
    forward_returns: dict[str, float] = field(default_factory=dict)
    benchmark_return: float | None = None
    current_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.candidates = [
            candidate if isinstance(candidate, OptimizationCandidate) else OptimizationCandidate.from_dict(candidate)
            for candidate in self.candidates
        ]
        self.forward_returns = _finite_float_dict(self.forward_returns, "forward_returns")
        self.benchmark_return = _float_or_none(self.benchmark_return, "benchmark_return")
        self.current_weights = _finite_float_dict(self.current_weights, "current_weights")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "evaluation_date": self.evaluation_date,
            "market": self.market,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "forward_returns": dict(self.forward_returns),
            "benchmark_return": self.benchmark_return,
            "current_weights": dict(self.current_weights),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RebalanceInput":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            as_of=str(data.get("as_of", "")),
            evaluation_date=str(data.get("evaluation_date", "")),
            market=str(data.get("market", "")),
            candidates=[
                OptimizationCandidate.from_dict(candidate)
                for candidate in list(data.get("candidates", []) or [])
                if isinstance(candidate, Mapping)
            ],
            forward_returns=dict(data.get("forward_returns", {}) or {}),
            benchmark_return=data.get("benchmark_return"),
            current_weights=dict(data.get("current_weights", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class RebalanceResult:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    rebalance_id: str = ""
    as_of: str = ""
    evaluation_date: str = ""
    market: str = ""
    plan_id: str = ""
    target_weights: dict[str, float] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    realized_gross_return: float = 0.0
    estimated_cost_return: float = 0.0
    realized_net_return: float = 0.0
    benchmark_return: float | None = None
    excess_return: float | None = None
    turnover_estimate: float = 0.0
    selected_symbols: list[str] = field(default_factory=list)
    missing_return_symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_weights = _clean_weight_dict(self.target_weights)
        self.current_weights = _clean_weight_dict(self.current_weights, keep_zero=True)
        self.realized_gross_return = _finite_float(self.realized_gross_return, "realized_gross_return")
        self.estimated_cost_return = _non_negative_float(self.estimated_cost_return, "estimated_cost_return")
        self.realized_net_return = _finite_float(self.realized_net_return, "realized_net_return")
        self.benchmark_return = _float_or_none(self.benchmark_return, "benchmark_return")
        self.excess_return = _float_or_none(self.excess_return, "excess_return")
        self.turnover_estimate = _non_negative_float(self.turnover_estimate, "turnover_estimate")
        self.selected_symbols = _ordered_unique(self.selected_symbols)
        self.missing_return_symbols = _ordered_unique(self.missing_return_symbols)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RebalanceResult":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            rebalance_id=str(data.get("rebalance_id", "")),
            as_of=str(data.get("as_of", "")),
            evaluation_date=str(data.get("evaluation_date", "")),
            market=str(data.get("market", "")),
            plan_id=str(data.get("plan_id", "")),
            target_weights=dict(data.get("target_weights", {}) or {}),
            current_weights=dict(data.get("current_weights", {}) or {}),
            realized_gross_return=float(data.get("realized_gross_return", 0.0)),
            estimated_cost_return=float(data.get("estimated_cost_return", 0.0)),
            realized_net_return=float(data.get("realized_net_return", 0.0)),
            benchmark_return=data.get("benchmark_return"),
            excess_return=data.get("excess_return"),
            turnover_estimate=float(data.get("turnover_estimate", 0.0)),
            selected_symbols=list(data.get("selected_symbols", []) or []),
            missing_return_symbols=list(data.get("missing_return_symbols", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class WalkForwardResult:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    run_id: str = ""
    market: str = ""
    start_date: str = ""
    end_date: str = ""
    rebalance_count: int = 0
    cumulative_gross_return: float = 0.0
    cumulative_net_return: float = 0.0
    cumulative_benchmark_return: float | None = None
    cumulative_excess_return: float | None = None
    annualized_net_return: float | None = None
    max_drawdown: float = 0.0
    average_turnover: float = 0.0
    total_estimated_cost_return: float = 0.0
    plans: list[OptimizedPortfolioPlan] = field(default_factory=list)
    rebalance_results: list[RebalanceResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rebalance_count < 0:
            raise ValueError("rebalance_count must be non-negative.")
        self.cumulative_gross_return = _finite_float(self.cumulative_gross_return, "cumulative_gross_return")
        self.cumulative_net_return = _finite_float(self.cumulative_net_return, "cumulative_net_return")
        self.cumulative_benchmark_return = _float_or_none(
            self.cumulative_benchmark_return,
            "cumulative_benchmark_return",
        )
        self.cumulative_excess_return = _float_or_none(
            self.cumulative_excess_return,
            "cumulative_excess_return",
        )
        self.annualized_net_return = _float_or_none(self.annualized_net_return, "annualized_net_return")
        self.max_drawdown = _non_negative_float(self.max_drawdown, "max_drawdown")
        self.average_turnover = _non_negative_float(self.average_turnover, "average_turnover")
        self.total_estimated_cost_return = _non_negative_float(
            self.total_estimated_cost_return,
            "total_estimated_cost_return",
        )
        self.plans = [
            plan if isinstance(plan, OptimizedPortfolioPlan) else OptimizedPortfolioPlan.from_dict(plan)
            for plan in self.plans
        ]
        self.rebalance_results = [
            result if isinstance(result, RebalanceResult) else RebalanceResult.from_dict(result)
            for result in self.rebalance_results
        ]
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "market": self.market,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rebalance_count": self.rebalance_count,
            "cumulative_gross_return": self.cumulative_gross_return,
            "cumulative_net_return": self.cumulative_net_return,
            "cumulative_benchmark_return": self.cumulative_benchmark_return,
            "cumulative_excess_return": self.cumulative_excess_return,
            "annualized_net_return": self.annualized_net_return,
            "max_drawdown": self.max_drawdown,
            "average_turnover": self.average_turnover,
            "total_estimated_cost_return": self.total_estimated_cost_return,
            "plans": [plan.to_dict() for plan in self.plans],
            "rebalance_results": [result.to_dict() for result in self.rebalance_results],
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WalkForwardResult":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            run_id=str(data.get("run_id", "")),
            market=str(data.get("market", "")),
            start_date=str(data.get("start_date", "")),
            end_date=str(data.get("end_date", "")),
            rebalance_count=int(data.get("rebalance_count", 0) or 0),
            cumulative_gross_return=float(data.get("cumulative_gross_return", 0.0)),
            cumulative_net_return=float(data.get("cumulative_net_return", 0.0)),
            cumulative_benchmark_return=data.get("cumulative_benchmark_return"),
            cumulative_excess_return=data.get("cumulative_excess_return"),
            annualized_net_return=data.get("annualized_net_return"),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            average_turnover=float(data.get("average_turnover", 0.0)),
            total_estimated_cost_return=float(data.get("total_estimated_cost_return", 0.0)),
            plans=[
                OptimizedPortfolioPlan.from_dict(plan)
                for plan in list(data.get("plans", []) or [])
                if isinstance(plan, Mapping)
            ],
            rebalance_results=[
                RebalanceResult.from_dict(result)
                for result in list(data.get("rebalance_results", []) or [])
                if isinstance(result, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )

__all__ = [
    "DEFAULT_PORTFOLIO_OPTIMIZER_DIR",
    "DEFAULT_OPTIMIZED_PLANS_FILENAME",
    "DEFAULT_REBALANCE_RESULTS_FILENAME",
    "DEFAULT_WALK_FORWARD_RESULTS_FILENAME",
    "PLAN_STATUS_OPTIMIZED",
    "PLAN_STATUS_EMPTY",
    "PLAN_STATUS_INFEASIBLE",
    "CONSTRAINT_BLOCKED_SYMBOL",
    "CONSTRAINT_MIN_EDGE",
    "CONSTRAINT_MAX_WEIGHT",
    "CONSTRAINT_GROSS_EXPOSURE",
    "CONSTRAINT_SECTOR_CAP",
    "CONSTRAINT_TURNOVER_CAP",
    "CONSTRAINT_MAX_NAMES",
    "CONSTRAINT_RISK_SCORE",
    "VIOLATION_INFO",
    "VIOLATION_WARNING",
    "VIOLATION_BLOCKER",
    "PortfolioOptimizerConfig",
    "OptimizationCandidate",
    "ConstraintViolation",
    "OptimizedPortfolioPlan",
    "RebalanceInput",
    "RebalanceResult",
    "WalkForwardResult",
    "make_constraint_violation_id",
    "make_plan_id",
    "make_rebalance_id",
    "make_walk_forward_run_id",
    "clamp_unit_interval",
    "validate_finite_number",
    "bps_to_decimal_return",
    "estimate_turnover",
    "compute_sector_weights",
    "compound_returns",
    "max_drawdown_from_returns",
]
