"""Offline cost, liquidity, and capacity diagnostics for factor research."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest import SingleFactorBacktestRun
from quant_investor.factors.matrix import FIELD_AMOUNT, MatrixDataBundle
from quant_investor.versioning import FACTOR_COST_CAPACITY_SCHEMA_VERSION


CAPACITY_VERDICT_PASS = "pass"
CAPACITY_VERDICT_WARN = "warn"
CAPACITY_VERDICT_FAIL = "fail"

CAPACITY_ISSUE_LOW_ADV = "low_adv"
CAPACITY_ISSUE_PARTICIPATION_BREACH = "participation_breach"
CAPACITY_ISSUE_COST_DRAG = "cost_drag"
CAPACITY_ISSUE_HIGH_TURNOVER = "high_turnover"
CAPACITY_ISSUE_LOW_TRADABILITY = "low_tradability"
CAPACITY_ISSUE_LOW_COVERAGE = "low_coverage"

SUPPORTED_CAPACITY_VERDICTS = {
    CAPACITY_VERDICT_PASS,
    CAPACITY_VERDICT_WARN,
    CAPACITY_VERDICT_FAIL,
}

SUPPORTED_CAPACITY_ISSUES = {
    CAPACITY_ISSUE_LOW_ADV,
    CAPACITY_ISSUE_PARTICIPATION_BREACH,
    CAPACITY_ISSUE_COST_DRAG,
    CAPACITY_ISSUE_HIGH_TURNOVER,
    CAPACITY_ISSUE_LOW_TRADABILITY,
    CAPACITY_ISSUE_LOW_COVERAGE,
}

_EPSILON = 1e-12


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
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
        json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON-serializable values.") from exc
    return safe


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, "metadata"))


def _non_empty_str(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite numeric.")
    return number


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field_name)


def _positive_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive; got {value!r}.")
    return number


def _non_negative_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
    number = _optional_finite_float(value, field_name)
    if number is not None and number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _unit_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _unit_float(value, field_name)


def _non_negative_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _slug(value: str | None) -> str:
    resolved = "none" if value is None else str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", resolved)
    return slug.strip("-") or "unknown"


def _short_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps(
        [_json_safe(part) for part in parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _to_valid_amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _config_identity_payload(config: "FactorCostCapacityConfig") -> dict[str, Any]:
    payload = config.to_dict()
    payload["config_id"] = ""
    return payload


@dataclass
class FactorCostCapacityConfig:
    schema_version: str = FACTOR_COST_CAPACITY_SCHEMA_VERSION
    config_id: str = ""
    target_capital: float = 0.0
    max_participation_rate: float = 0.10
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    max_average_turnover: float | None = None
    max_cost_drag_ratio: float | None = 0.50
    min_capacity: float | None = None
    min_tradability_ratio: float = 0.80
    min_coverage_ratio: float = 0.80
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_COST_CAPACITY_SCHEMA_VERSION)
        self.target_capital = _positive_float(self.target_capital, "target_capital")
        self.max_participation_rate = _unit_float(
            self.max_participation_rate,
            "max_participation_rate",
        )
        self.transaction_cost_bps = _non_negative_float(
            self.transaction_cost_bps,
            "transaction_cost_bps",
        )
        self.slippage_bps = _non_negative_float(self.slippage_bps, "slippage_bps")
        self.market_impact_bps = _non_negative_float(
            self.market_impact_bps,
            "market_impact_bps",
        )
        self.max_average_turnover = _unit_float_or_none(
            self.max_average_turnover,
            "max_average_turnover",
        )
        self.max_cost_drag_ratio = _unit_float_or_none(
            self.max_cost_drag_ratio,
            "max_cost_drag_ratio",
        )
        self.min_capacity = _optional_non_negative_float(self.min_capacity, "min_capacity")
        self.min_tradability_ratio = _unit_float(
            self.min_tradability_ratio,
            "min_tradability_ratio",
        )
        self.min_coverage_ratio = _unit_float(self.min_coverage_ratio, "min_coverage_ratio")
        self.metadata = _coerce_metadata(self.metadata)
        self.config_id = str(self.config_id or "").strip()
        if not self.config_id:
            self.config_id = make_cost_capacity_config_id(self)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorCostCapacityConfig":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_COST_CAPACITY_SCHEMA_VERSION)),
            config_id=str(data.get("config_id", "")),
            target_capital=float(data.get("target_capital", 0.0)),
            max_participation_rate=float(data.get("max_participation_rate", 0.10)),
            transaction_cost_bps=float(data.get("transaction_cost_bps", 0.0)),
            slippage_bps=float(data.get("slippage_bps", 0.0)),
            market_impact_bps=float(data.get("market_impact_bps", 0.0)),
            max_average_turnover=data.get("max_average_turnover"),
            max_cost_drag_ratio=data.get("max_cost_drag_ratio", 0.50),
            min_capacity=data.get("min_capacity"),
            min_tradability_ratio=float(data.get("min_tradability_ratio", 0.80)),
            min_coverage_ratio=float(data.get("min_coverage_ratio", 0.80)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorCostCapacityReport:
    schema_version: str = FACTOR_COST_CAPACITY_SCHEMA_VERSION
    report_id: str = ""
    factor_id: str | None = None
    factor_version: str | None = None
    backtest_run_id: str = ""
    generated_at: str = ""
    config: FactorCostCapacityConfig = field(default_factory=FactorCostCapacityConfig)
    average_turnover: float | None = None
    total_cost_bps: float = 0.0
    estimated_average_cost_return: float | None = None
    before_cost_sharpe: float | None = None
    after_cost_sharpe: float | None = None
    cost_drag_ratio: float | None = None
    estimated_capacity: float | None = None
    average_adv: float | None = None
    participation_breach_count: int = 0
    participation_breach_ratio: float | None = None
    tradability_ratio: float | None = None
    coverage_ratio: float | None = None
    issue_codes: list[str] = field(default_factory=list)
    verdict: str = CAPACITY_VERDICT_FAIL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_COST_CAPACITY_SCHEMA_VERSION)
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.factor_id = _optional_str(self.factor_id)
        self.factor_version = _optional_str(self.factor_version)
        self.backtest_run_id = _non_empty_str(self.backtest_run_id, "backtest_run_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        if not isinstance(self.config, FactorCostCapacityConfig):
            self.config = FactorCostCapacityConfig.from_dict(self.config)
        for field_name in (
            "average_turnover",
            "estimated_average_cost_return",
            "before_cost_sharpe",
            "after_cost_sharpe",
            "cost_drag_ratio",
            "estimated_capacity",
            "average_adv",
            "coverage_ratio",
        ):
            setattr(self, field_name, _optional_finite_float(getattr(self, field_name), field_name))
        self.total_cost_bps = _non_negative_float(self.total_cost_bps, "total_cost_bps")
        self.participation_breach_count = _non_negative_int(
            self.participation_breach_count,
            "participation_breach_count",
        )
        self.participation_breach_ratio = _unit_float_or_none(
            self.participation_breach_ratio,
            "participation_breach_ratio",
        )
        self.tradability_ratio = _unit_float_or_none(
            self.tradability_ratio,
            "tradability_ratio",
        )
        self.coverage_ratio = _unit_float_or_none(self.coverage_ratio, "coverage_ratio")
        self.cost_drag_ratio = _unit_float_or_none(self.cost_drag_ratio, "cost_drag_ratio")
        self.issue_codes = _ordered_unique(self.issue_codes)
        for issue_code in self.issue_codes:
            if issue_code not in SUPPORTED_CAPACITY_ISSUES:
                raise ValueError(f"issue_codes contains unsupported issue {issue_code!r}.")
        self.verdict = str(self.verdict)
        if self.verdict not in SUPPORTED_CAPACITY_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(SUPPORTED_CAPACITY_VERDICTS)}.")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(_json_safe(asdict(self)))
        payload["config"] = self.config.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorCostCapacityReport":
        data = dict(payload)
        config_payload = data.get("config", {}) or {}
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_COST_CAPACITY_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            factor_id=data.get("factor_id"),
            factor_version=data.get("factor_version"),
            backtest_run_id=str(data.get("backtest_run_id", "")),
            generated_at=str(data.get("generated_at", "")),
            config=FactorCostCapacityConfig.from_dict(config_payload)
            if isinstance(config_payload, Mapping)
            else config_payload,
            average_turnover=data.get("average_turnover"),
            total_cost_bps=float(data.get("total_cost_bps", 0.0)),
            estimated_average_cost_return=data.get("estimated_average_cost_return"),
            before_cost_sharpe=data.get("before_cost_sharpe"),
            after_cost_sharpe=data.get("after_cost_sharpe"),
            cost_drag_ratio=data.get("cost_drag_ratio"),
            estimated_capacity=data.get("estimated_capacity"),
            average_adv=data.get("average_adv"),
            participation_breach_count=int(data.get("participation_breach_count", 0)),
            participation_breach_ratio=data.get("participation_breach_ratio"),
            tradability_ratio=data.get("tradability_ratio"),
            coverage_ratio=data.get("coverage_ratio"),
            issue_codes=list(data.get("issue_codes", []) or []),
            verdict=str(data.get("verdict", CAPACITY_VERDICT_FAIL)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_cost_capacity_config_id(config: FactorCostCapacityConfig) -> str:
    payload = _config_identity_payload(config)
    parts = [payload.get("target_capital"), payload.get("max_participation_rate"), payload]
    return f"factor-cost-capacity-config-{_slug(str(payload.get('target_capital')))}-{_short_hash(parts)}"


def make_cost_capacity_report_id(
    *,
    backtest_run_id: str,
    generated_at: str,
    config_id: str,
) -> str:
    run_id = _non_empty_str(backtest_run_id, "backtest_run_id")
    timestamp = _non_empty_str(generated_at, "generated_at")
    resolved_config_id = _non_empty_str(config_id, "config_id")
    parts = [run_id, timestamp, resolved_config_id]
    return f"factor-cost-capacity-{_slug(run_id)}-{_slug(resolved_config_id)}-{_short_hash(parts)}"


def bps_to_decimal_return(value_bps: float) -> float:
    return _finite_float(value_bps, "value_bps") / 10000.0


def estimate_average_adv(bundle: MatrixDataBundle) -> float | None:
    if not bundle.has_field(FIELD_AMOUNT):
        return None
    amounts = bundle.get_field(FIELD_AMOUNT)
    valid_amounts: list[float] = []
    for row in amounts:
        for value in row:
            amount = _to_valid_amount(value)
            if amount is not None:
                valid_amounts.append(amount)
    return _mean(valid_amounts)


def _date_index(dates: Sequence[str], value: str) -> int | None:
    try:
        return list(dates).index(value)
    except ValueError:
        return None


def _active_weight_rows(run: SingleFactorBacktestRun, signal_index: int) -> list[int]:
    active_rows: list[int] = []
    for row_index, row in enumerate(run.weight_matrix.net_weights):
        value = row[signal_index]
        if value is not None and abs(float(value)) > _EPSILON:
            active_rows.append(row_index)
    return active_rows


def estimate_factor_capacity(
    run: SingleFactorBacktestRun,
    bundle: MatrixDataBundle,
    config: FactorCostCapacityConfig,
) -> tuple[float | None, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "model": "simple_offline_adv_participation_proxy",
        "description": (
            "Daily capacity is estimated as average active-symbol amount times "
            "max participation rate divided by daily turnover."
        ),
        "target_capital": config.target_capital,
        "max_participation_rate": config.max_participation_rate,
    }
    if not bundle.has_field(FIELD_AMOUNT):
        metadata.update(
            {
                "adv_available": False,
                "daily_capacity_count": 0,
                "participation_breach_count": 0,
                "participation_breach_dates": [],
            }
        )
        return None, metadata

    amounts = bundle.get_field(FIELD_AMOUNT)
    capacity_values: list[float] = []
    breach_dates: list[str] = []
    daily_estimates: list[dict[str, Any]] = []

    for record in run.daily_records:
        signal_index = _date_index(run.weight_matrix.dates, record.signal_date)
        trade_index = _date_index(bundle.contract.dates, record.execution_start_date)
        if trade_index is None:
            trade_index = _date_index(bundle.contract.dates, record.date)
        if signal_index is None or trade_index is None:
            continue
        active_rows = _active_weight_rows(run, signal_index)
        active_amounts = [
            amount
            for row_index in active_rows
            for amount in [_to_valid_amount(amounts[row_index][trade_index])]
            if amount is not None
        ]
        if not active_amounts:
            continue
        average_active_amount = sum(active_amounts) / len(active_amounts)
        allowed_trade_value = average_active_amount * config.max_participation_rate
        requested_trade_value = record.turnover * config.target_capital
        if requested_trade_value > allowed_trade_value + _EPSILON:
            breach_dates.append(record.date)
        daily_capacity = None
        if record.turnover > _EPSILON:
            daily_capacity = allowed_trade_value / max(record.turnover, _EPSILON)
            capacity_values.append(daily_capacity)
        daily_estimates.append(
            {
                "date": record.date,
                "active_symbol_count": len(active_amounts),
                "average_active_amount": average_active_amount,
                "allowed_trade_value": allowed_trade_value,
                "requested_trade_value": requested_trade_value,
                "daily_capacity": daily_capacity,
            }
        )

    metadata.update(
        {
            "adv_available": True,
            "daily_capacity_count": len(capacity_values),
            "participation_breach_count": len(breach_dates),
            "participation_breach_dates": sorted(breach_dates),
            "daily_estimates": daily_estimates,
        }
    )
    return _median(capacity_values), metadata


def _tradability_ratio(bundle: MatrixDataBundle) -> float | None:
    if bundle.tradability_mask is None:
        return None
    total = 0
    tradable = 0
    for row in bundle.tradability_mask:
        for value in row:
            total += 1
            if value:
                tradable += 1
    if total == 0:
        return None
    return tradable / total


def build_factor_cost_capacity_report(
    run: SingleFactorBacktestRun,
    bundle: MatrixDataBundle,
    *,
    config: FactorCostCapacityConfig,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorCostCapacityReport:
    turnovers = [float(record.turnover) for record in run.daily_records]
    average_turnover = _mean(turnovers)
    total_cost_bps = (
        config.transaction_cost_bps
        + config.slippage_bps
        + config.market_impact_bps
    )
    estimated_average_cost_return = (
        None
        if average_turnover is None
        else average_turnover * bps_to_decimal_return(total_cost_bps)
    )
    before_cost_sharpe = run.aggregate_result.before_cost_sharpe
    after_cost_sharpe = run.aggregate_result.after_cost_sharpe
    cost_drag_ratio = None
    if (
        before_cost_sharpe is not None
        and after_cost_sharpe is not None
        and before_cost_sharpe > 0.0
        and after_cost_sharpe > 0.0
    ):
        cost_drag_ratio = max(0.0, before_cost_sharpe - after_cost_sharpe) / abs(
            before_cost_sharpe
        )

    estimated_capacity, capacity_metadata = estimate_factor_capacity(run, bundle, config)
    average_adv = estimate_average_adv(bundle)
    participation_breach_count = int(capacity_metadata.get("participation_breach_count", 0) or 0)
    participation_breach_ratio = None
    if run.daily_records:
        participation_breach_ratio = participation_breach_count / len(run.daily_records)
    tradability_ratio = _tradability_ratio(bundle)
    coverage_ratio = run.aggregate_result.coverage_ratio

    issue_codes: list[str] = []
    if (
        config.max_average_turnover is not None
        and average_turnover is not None
        and average_turnover > config.max_average_turnover
    ):
        issue_codes.append(CAPACITY_ISSUE_HIGH_TURNOVER)
    if (
        config.max_cost_drag_ratio is not None
        and cost_drag_ratio is not None
        and cost_drag_ratio > config.max_cost_drag_ratio
    ):
        issue_codes.append(CAPACITY_ISSUE_COST_DRAG)
    if participation_breach_count > 0:
        issue_codes.append(CAPACITY_ISSUE_PARTICIPATION_BREACH)
    if tradability_ratio is not None and tradability_ratio < config.min_tradability_ratio:
        issue_codes.append(CAPACITY_ISSUE_LOW_TRADABILITY)
    if coverage_ratio < config.min_coverage_ratio:
        issue_codes.append(CAPACITY_ISSUE_LOW_COVERAGE)
    if average_adv is None:
        issue_codes.append(CAPACITY_ISSUE_LOW_ADV)
    if (
        config.min_capacity is not None
        and (estimated_capacity is None or estimated_capacity < config.min_capacity)
    ):
        issue_codes.append(CAPACITY_ISSUE_LOW_ADV)

    unique_issues = _ordered_unique(issue_codes)
    fail_issues: set[str] = set()
    if CAPACITY_ISSUE_LOW_COVERAGE in unique_issues:
        fail_issues.add(CAPACITY_ISSUE_LOW_COVERAGE)
    if CAPACITY_ISSUE_LOW_TRADABILITY in unique_issues:
        fail_issues.add(CAPACITY_ISSUE_LOW_TRADABILITY)
    if (
        config.min_capacity is not None
        and (estimated_capacity is None or estimated_capacity < config.min_capacity)
    ):
        fail_issues.add(CAPACITY_ISSUE_LOW_ADV)

    if fail_issues:
        verdict = CAPACITY_VERDICT_FAIL
    elif unique_issues:
        verdict = CAPACITY_VERDICT_WARN
    else:
        verdict = CAPACITY_VERDICT_PASS

    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "offline_only": True,
            "model": "simple_offline_adv_participation_proxy",
            "model_limitations": (
                "This is a deterministic ADV/participation proxy, not a broker "
                "execution model."
            ),
            "capacity_estimation": capacity_metadata,
            "pass": "phase9_pass4",
        }
    )
    return FactorCostCapacityReport(
        report_id=make_cost_capacity_report_id(
            backtest_run_id=run.run_id,
            generated_at=generated_at,
            config_id=config.config_id,
        ),
        factor_id=run.factor_id,
        factor_version=run.factor_version,
        backtest_run_id=run.run_id,
        generated_at=generated_at,
        config=config,
        average_turnover=average_turnover,
        total_cost_bps=total_cost_bps,
        estimated_average_cost_return=estimated_average_cost_return,
        before_cost_sharpe=before_cost_sharpe,
        after_cost_sharpe=after_cost_sharpe,
        cost_drag_ratio=cost_drag_ratio,
        estimated_capacity=estimated_capacity,
        average_adv=average_adv,
        participation_breach_count=participation_breach_count,
        participation_breach_ratio=participation_breach_ratio,
        tradability_ratio=tradability_ratio,
        coverage_ratio=coverage_ratio,
        issue_codes=unique_issues,
        verdict=verdict,
        metadata=resolved_metadata,
    )


__all__ = [
    "CAPACITY_VERDICT_PASS",
    "CAPACITY_VERDICT_WARN",
    "CAPACITY_VERDICT_FAIL",
    "CAPACITY_ISSUE_LOW_ADV",
    "CAPACITY_ISSUE_PARTICIPATION_BREACH",
    "CAPACITY_ISSUE_COST_DRAG",
    "CAPACITY_ISSUE_HIGH_TURNOVER",
    "CAPACITY_ISSUE_LOW_TRADABILITY",
    "CAPACITY_ISSUE_LOW_COVERAGE",
    "FactorCostCapacityConfig",
    "FactorCostCapacityReport",
    "make_cost_capacity_config_id",
    "make_cost_capacity_report_id",
    "bps_to_decimal_return",
    "estimate_average_adv",
    "estimate_factor_capacity",
    "build_factor_cost_capacity_report",
]
