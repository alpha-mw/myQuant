"""Offline robustness slicing and enhanced validation report helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest import SingleFactorBacktestRun
from quant_investor.factors.capacity import (
    CAPACITY_VERDICT_FAIL,
    CAPACITY_VERDICT_PASS,
    CAPACITY_VERDICT_WARN,
    FactorCostCapacityReport,
)
from quant_investor.factors.metrics import (
    ReturnMetricSummary,
    TurnoverMetricSummary,
    build_return_metric_summary,
    build_turnover_metric_summary,
)
from quant_investor.factors.schema import (
    FACTOR_STATUS_PAPER_TRADING,
    FACTOR_STATUS_REJECTED,
    FACTOR_STATUS_VALIDATED_RESEARCH,
    VALIDATION_VERDICT_FAIL,
    VALIDATION_VERDICT_PASS,
    VALIDATION_VERDICT_WARN,
    FactorValidationReport,
    FactorValidationThresholds,
)
from quant_investor.versioning import FACTOR_ROBUSTNESS_SCHEMA_VERSION


SLICE_FULL_SAMPLE = "full_sample"
SLICE_RECENT_1Y = "recent_1y"
SLICE_RECENT_3Y = "recent_3y"
SLICE_RECENT_5Y = "recent_5y"

ROBUSTNESS_VERDICT_PASS = "pass"
ROBUSTNESS_VERDICT_WARN = "warn"
ROBUSTNESS_VERDICT_FAIL = "fail"

SUPPORTED_ROBUSTNESS_VERDICTS = {
    ROBUSTNESS_VERDICT_PASS,
    ROBUSTNESS_VERDICT_WARN,
    ROBUSTNESS_VERDICT_FAIL,
}

_RECENT_WINDOWS = (
    (SLICE_RECENT_1Y, 252),
    (SLICE_RECENT_3Y, 756),
    (SLICE_RECENT_5Y, 1260),
)
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


def _non_negative_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric or None.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite numeric or None.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    number = _optional_finite_float(value, field_name)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
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


def _coerce_iso_date(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date; got {value!r}.") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"{field_name} must be a canonical ISO date; got {value!r}.")
    return text


def _coerce_dates(values: Sequence[str]) -> list[str]:
    dates = [_coerce_iso_date(str(value), "dates") for value in values]
    clean_dates = [value for value in dates if value is not None]
    return sorted(dict.fromkeys(clean_dates))


def _mean(values: Sequence[float | None]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


@dataclass
class FactorSliceSpec:
    schema_version: str = FACTOR_ROBUSTNESS_SCHEMA_VERSION
    slice_id: str = ""
    name: str = ""
    start_date: str | None = None
    end_date: str | None = None
    regime_label: str | None = None
    min_sample_days: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_ROBUSTNESS_SCHEMA_VERSION)
        self.name = _non_empty_str(self.name, "name")
        self.start_date = _coerce_iso_date(self.start_date, "start_date")
        self.end_date = _coerce_iso_date(self.end_date, "end_date")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")
        self.regime_label = _optional_str(self.regime_label)
        self.min_sample_days = _non_negative_int(self.min_sample_days, "min_sample_days")
        self.metadata = _coerce_metadata(self.metadata)
        self.slice_id = str(self.slice_id or "").strip()
        if not self.slice_id:
            self.slice_id = make_slice_id(
                name=self.name,
                start_date=self.start_date,
                end_date=self.end_date,
                regime_label=self.regime_label,
            )

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorSliceSpec":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_ROBUSTNESS_SCHEMA_VERSION)),
            slice_id=str(data.get("slice_id", "")),
            name=str(data.get("name", "")),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            regime_label=data.get("regime_label"),
            min_sample_days=int(data.get("min_sample_days", 20)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorSliceResult:
    schema_version: str = FACTOR_ROBUSTNESS_SCHEMA_VERSION
    slice_id: str = ""
    name: str = ""
    start_date: str | None = None
    end_date: str | None = None
    regime_label: str | None = None
    sample_days: int = 0
    before_cost_metrics: ReturnMetricSummary = field(default_factory=ReturnMetricSummary)
    after_cost_metrics: ReturnMetricSummary = field(default_factory=ReturnMetricSummary)
    excess_metrics: ReturnMetricSummary | None = None
    turnover_metrics: TurnoverMetricSummary = field(default_factory=TurnoverMetricSummary)
    coverage_ratio: float | None = None
    missing_ratio: float | None = None
    long_num_avg: float | None = None
    short_num_avg: float | None = None
    verdict: str = ROBUSTNESS_VERDICT_FAIL
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_ROBUSTNESS_SCHEMA_VERSION)
        self.slice_id = _non_empty_str(self.slice_id, "slice_id")
        self.name = _non_empty_str(self.name, "name")
        self.start_date = _coerce_iso_date(self.start_date, "start_date")
        self.end_date = _coerce_iso_date(self.end_date, "end_date")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")
        self.regime_label = _optional_str(self.regime_label)
        self.sample_days = _non_negative_int(self.sample_days, "sample_days")
        if not isinstance(self.before_cost_metrics, ReturnMetricSummary):
            self.before_cost_metrics = ReturnMetricSummary.from_dict(self.before_cost_metrics)
        if not isinstance(self.after_cost_metrics, ReturnMetricSummary):
            self.after_cost_metrics = ReturnMetricSummary.from_dict(self.after_cost_metrics)
        if self.excess_metrics is not None and not isinstance(self.excess_metrics, ReturnMetricSummary):
            self.excess_metrics = ReturnMetricSummary.from_dict(self.excess_metrics)
        if not isinstance(self.turnover_metrics, TurnoverMetricSummary):
            self.turnover_metrics = TurnoverMetricSummary.from_dict(self.turnover_metrics)
        self.coverage_ratio = _unit_float_or_none(self.coverage_ratio, "coverage_ratio")
        self.missing_ratio = _unit_float_or_none(self.missing_ratio, "missing_ratio")
        self.long_num_avg = _optional_finite_float(self.long_num_avg, "long_num_avg")
        self.short_num_avg = _optional_finite_float(self.short_num_avg, "short_num_avg")
        self.verdict = str(self.verdict)
        if self.verdict not in SUPPORTED_ROBUSTNESS_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(SUPPORTED_ROBUSTNESS_VERDICTS)}.")
        self.warnings = _ordered_unique(self.warnings)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(_json_safe(asdict(self)))
        payload["before_cost_metrics"] = self.before_cost_metrics.to_dict()
        payload["after_cost_metrics"] = self.after_cost_metrics.to_dict()
        payload["excess_metrics"] = (
            None if self.excess_metrics is None else self.excess_metrics.to_dict()
        )
        payload["turnover_metrics"] = self.turnover_metrics.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorSliceResult":
        data = dict(payload)
        excess_payload = data.get("excess_metrics")
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_ROBUSTNESS_SCHEMA_VERSION)),
            slice_id=str(data.get("slice_id", "")),
            name=str(data.get("name", "")),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            regime_label=data.get("regime_label"),
            sample_days=int(data.get("sample_days", 0)),
            before_cost_metrics=ReturnMetricSummary.from_dict(
                dict(data.get("before_cost_metrics", {}) or {})
            ),
            after_cost_metrics=ReturnMetricSummary.from_dict(
                dict(data.get("after_cost_metrics", {}) or {})
            ),
            excess_metrics=ReturnMetricSummary.from_dict(dict(excess_payload))
            if isinstance(excess_payload, Mapping)
            else None,
            turnover_metrics=TurnoverMetricSummary.from_dict(
                dict(data.get("turnover_metrics", {}) or {})
            ),
            coverage_ratio=data.get("coverage_ratio"),
            missing_ratio=data.get("missing_ratio"),
            long_num_avg=data.get("long_num_avg"),
            short_num_avg=data.get("short_num_avg"),
            verdict=str(data.get("verdict", ROBUSTNESS_VERDICT_FAIL)),
            warnings=list(data.get("warnings", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorRobustnessReport:
    schema_version: str = FACTOR_ROBUSTNESS_SCHEMA_VERSION
    report_id: str = ""
    factor_id: str | None = None
    factor_version: str | None = None
    backtest_run_id: str = ""
    generated_at: str = ""
    slice_results: list[FactorSliceResult] = field(default_factory=list)
    overall_verdict: str = ROBUSTNESS_VERDICT_FAIL
    failed_slices: list[str] = field(default_factory=list)
    warning_slices: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_ROBUSTNESS_SCHEMA_VERSION)
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.factor_id = _optional_str(self.factor_id)
        self.factor_version = _optional_str(self.factor_version)
        self.backtest_run_id = _non_empty_str(self.backtest_run_id, "backtest_run_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        self.slice_results = [
            result if isinstance(result, FactorSliceResult)
            else FactorSliceResult.from_dict(result)
            for result in self.slice_results
        ]
        self.slice_results = sorted(self.slice_results, key=lambda result: result.slice_id)
        self.overall_verdict = str(self.overall_verdict)
        if self.overall_verdict not in SUPPORTED_ROBUSTNESS_VERDICTS:
            raise ValueError(
                f"overall_verdict must be one of {sorted(SUPPORTED_ROBUSTNESS_VERDICTS)}."
            )
        self.failed_slices = _ordered_unique(self.failed_slices)
        self.warning_slices = _ordered_unique(self.warning_slices)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(_json_safe(asdict(self)))
        payload["slice_results"] = [result.to_dict() for result in self.slice_results]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorRobustnessReport":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_ROBUSTNESS_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            factor_id=data.get("factor_id"),
            factor_version=data.get("factor_version"),
            backtest_run_id=str(data.get("backtest_run_id", "")),
            generated_at=str(data.get("generated_at", "")),
            slice_results=[
                FactorSliceResult.from_dict(result)
                for result in list(data.get("slice_results", []) or [])
                if isinstance(result, Mapping)
            ],
            overall_verdict=str(data.get("overall_verdict", ROBUSTNESS_VERDICT_FAIL)),
            failed_slices=list(data.get("failed_slices", []) or []),
            warning_slices=list(data.get("warning_slices", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_slice_id(
    *,
    name: str,
    start_date: str | None,
    end_date: str | None,
    regime_label: str | None = None,
) -> str:
    resolved_name = _non_empty_str(name, "name")
    parts = [resolved_name, start_date, end_date, regime_label]
    return f"factor-slice-{_slug(resolved_name)}-{_slug(regime_label)}-{_short_hash(parts)}"


def make_robustness_report_id(*, backtest_run_id: str, generated_at: str) -> str:
    run_id = _non_empty_str(backtest_run_id, "backtest_run_id")
    timestamp = _non_empty_str(generated_at, "generated_at")
    parts = [run_id, timestamp]
    return f"factor-robustness-{_slug(run_id)}-{_short_hash(parts)}"


def build_default_recent_slice_specs(
    dates: Sequence[str],
    *,
    min_sample_days: int = 20,
) -> list[FactorSliceSpec]:
    resolved_min = _non_negative_int(min_sample_days, "min_sample_days")
    sorted_dates = _coerce_dates(dates)
    start_date = sorted_dates[0] if sorted_dates else None
    end_date = sorted_dates[-1] if sorted_dates else None
    specs = [
        FactorSliceSpec(
            name=SLICE_FULL_SAMPLE,
            start_date=start_date,
            end_date=end_date,
            min_sample_days=resolved_min,
            metadata={"window": "full"},
        )
    ]
    for slice_name, window_size in _RECENT_WINDOWS:
        if not sorted_dates:
            continue
        if len(sorted_dates) >= window_size:
            window_dates = sorted_dates[-window_size:]
        elif len(sorted_dates) >= resolved_min:
            window_dates = sorted_dates
        else:
            continue
        specs.append(
            FactorSliceSpec(
                name=slice_name,
                start_date=window_dates[0],
                end_date=window_dates[-1],
                min_sample_days=resolved_min,
                metadata={"window_records": len(window_dates), "target_window_records": window_size},
            )
        )
    return specs


def build_regime_slice_specs(
    dates: Sequence[str],
    regime_by_date: Mapping[str, str],
    *,
    min_sample_days: int = 20,
) -> list[FactorSliceSpec]:
    sorted_dates = _coerce_dates(dates)
    resolved_min = _non_negative_int(min_sample_days, "min_sample_days")
    dates_by_regime: dict[str, list[str]] = {}
    for current_date in sorted_dates:
        regime = str(regime_by_date.get(current_date, "")).strip()
        if not regime:
            continue
        dates_by_regime.setdefault(regime, []).append(current_date)
    specs: list[FactorSliceSpec] = []
    for regime_label in sorted(dates_by_regime):
        regime_dates = dates_by_regime[regime_label]
        specs.append(
            FactorSliceSpec(
                name=f"regime_{regime_label}",
                start_date=regime_dates[0],
                end_date=regime_dates[-1],
                regime_label=regime_label,
                min_sample_days=resolved_min,
                metadata={
                    "date_list": list(regime_dates),
                    "regime_by_date": {
                        current_date: regime_label
                        for current_date in regime_dates
                    },
                },
            )
        )
    return specs


def _record_in_slice(record_date: str, spec: FactorSliceSpec) -> bool:
    if spec.start_date is not None and record_date < spec.start_date:
        return False
    if spec.end_date is not None and record_date > spec.end_date:
        return False
    explicit_dates = spec.metadata.get("date_list") or spec.metadata.get("dates")
    if isinstance(explicit_dates, Sequence) and not isinstance(explicit_dates, (str, bytes)):
        if record_date not in {str(value) for value in explicit_dates}:
            return False
    regime_by_date = spec.metadata.get("regime_by_date")
    if spec.regime_label is not None and isinstance(regime_by_date, Mapping):
        if str(regime_by_date.get(record_date, "")) != spec.regime_label:
            return False
    return True


def evaluate_slice_result(
    run: SingleFactorBacktestRun,
    spec: FactorSliceSpec,
    *,
    min_after_cost_sharpe: float | None = None,
    max_drawdown: float | None = None,
    max_turnover: float | None = None,
    min_coverage_ratio: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorSliceResult:
    records = [
        record
        for record in run.daily_records
        if _record_in_slice(record.execution_end_date, spec)
    ]
    before_cost_returns = [record.long_short_return for record in records]
    after_cost_returns = [record.after_cost_return for record in records]
    excess_returns = [record.excess_return for record in records]
    turnovers = [record.turnover for record in records]
    coverage_ratio = _mean([record.coverage_ratio for record in records])
    missing_ratio = _mean([record.missing_ratio for record in records])
    long_num_avg = _mean([float(record.long_count) for record in records])
    short_num_avg = _mean([float(record.short_count) for record in records])
    before_cost_metrics = build_return_metric_summary(
        f"{spec.name}.before_cost",
        before_cost_returns,
        metadata={"slice_id": spec.slice_id},
    )
    after_cost_metrics = build_return_metric_summary(
        f"{spec.name}.after_cost",
        after_cost_returns,
        metadata={"slice_id": spec.slice_id},
    )
    excess_metrics = None
    if any(value is not None for value in excess_returns):
        excess_metrics = build_return_metric_summary(
            f"{spec.name}.excess",
            excess_returns,
            metadata={"slice_id": spec.slice_id},
        )
    turnover_metrics = build_turnover_metric_summary(
        f"{spec.name}.turnover",
        turnovers,
        turnover_budget=max_turnover,
        metadata={"slice_id": spec.slice_id},
    )

    warnings: list[str] = []
    verdict = ROBUSTNESS_VERDICT_PASS
    if len(records) < spec.min_sample_days:
        verdict = ROBUSTNESS_VERDICT_FAIL
        warnings.append("insufficient_sample_days")
    if min_after_cost_sharpe is not None:
        if after_cost_metrics.sharpe is None:
            warnings.append("missing_after_cost_sharpe")
        elif after_cost_metrics.sharpe < min_after_cost_sharpe:
            warnings.append("after_cost_sharpe_below_threshold")
    if max_drawdown is not None:
        if after_cost_metrics.max_drawdown is None:
            warnings.append("missing_max_drawdown")
        elif after_cost_metrics.max_drawdown > max_drawdown:
            warnings.append("max_drawdown_above_threshold")
    if max_turnover is not None and turnover_metrics.budget_breach_count > 0:
        warnings.append("turnover_budget_breach")
    if min_coverage_ratio is not None:
        if coverage_ratio is None:
            warnings.append("missing_coverage_ratio")
        elif coverage_ratio < min_coverage_ratio:
            warnings.append("coverage_ratio_below_threshold")
    if verdict != ROBUSTNESS_VERDICT_FAIL and warnings:
        verdict = ROBUSTNESS_VERDICT_WARN

    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "backtest_run_id": run.run_id,
            "record_dates": [record.date for record in records],
            "offline_only": True,
            "pass": "phase9_pass4",
        }
    )
    return FactorSliceResult(
        slice_id=spec.slice_id,
        name=spec.name,
        start_date=spec.start_date,
        end_date=spec.end_date,
        regime_label=spec.regime_label,
        sample_days=len(records),
        before_cost_metrics=before_cost_metrics,
        after_cost_metrics=after_cost_metrics,
        excess_metrics=excess_metrics,
        turnover_metrics=turnover_metrics,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        long_num_avg=long_num_avg,
        short_num_avg=short_num_avg,
        verdict=verdict,
        warnings=warnings,
        metadata=resolved_metadata,
    )


def build_factor_robustness_report(
    run: SingleFactorBacktestRun,
    *,
    slice_specs: Sequence[FactorSliceSpec] | None = None,
    regime_by_date: Mapping[str, str] | None = None,
    generated_at: str,
    min_after_cost_sharpe: float | None = None,
    max_drawdown: float | None = None,
    max_turnover: float | None = None,
    min_coverage_ratio: float | None = None,
    min_sample_days: int = 20,
    metadata: Mapping[str, Any] | None = None,
) -> FactorRobustnessReport:
    record_dates = [record.execution_end_date for record in run.daily_records]
    specs = list(slice_specs) if slice_specs is not None else build_default_recent_slice_specs(
        record_dates,
        min_sample_days=min_sample_days,
    )
    if regime_by_date is not None:
        specs.extend(
            build_regime_slice_specs(
                record_dates,
                regime_by_date,
                min_sample_days=min_sample_days,
            )
        )
    deduped_specs = {spec.slice_id: spec for spec in specs}
    ordered_specs = [deduped_specs[slice_id] for slice_id in sorted(deduped_specs)]
    slice_results = [
        evaluate_slice_result(
            run,
            spec,
            min_after_cost_sharpe=min_after_cost_sharpe,
            max_drawdown=max_drawdown,
            max_turnover=max_turnover,
            min_coverage_ratio=min_coverage_ratio,
            metadata={"report_generated_at": generated_at},
        )
        for spec in ordered_specs
    ]
    failed_slices = [
        result.slice_id
        for result in slice_results
        if result.verdict == ROBUSTNESS_VERDICT_FAIL
    ]
    warning_slices = [
        result.slice_id
        for result in slice_results
        if result.verdict == ROBUSTNESS_VERDICT_WARN
    ]
    if failed_slices:
        overall = ROBUSTNESS_VERDICT_FAIL
    elif warning_slices:
        overall = ROBUSTNESS_VERDICT_WARN
    else:
        overall = ROBUSTNESS_VERDICT_PASS
    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "slice_count": len(slice_results),
            "offline_only": True,
            "pass": "phase9_pass4",
        }
    )
    return FactorRobustnessReport(
        report_id=make_robustness_report_id(
            backtest_run_id=run.run_id,
            generated_at=generated_at,
        ),
        factor_id=run.factor_id,
        factor_version=run.factor_version,
        backtest_run_id=run.run_id,
        generated_at=generated_at,
        slice_results=slice_results,
        overall_verdict=overall,
        failed_slices=failed_slices,
        warning_slices=warning_slices,
        metadata=resolved_metadata,
    )


def _make_enhanced_validation_report_id(
    *,
    factor_id: str,
    factor_version: str,
    backtest_result_id: str,
    generated_at: str,
) -> str:
    parts = [factor_id, factor_version, backtest_result_id, generated_at, "enhanced"]
    return (
        f"factor-enhanced-validation-{_slug(factor_id)}-"
        f"{_slug(factor_version)}-{_short_hash(parts)}"
    )


def _add_gate(
    gate_results: dict[str, str],
    failed_gates: list[str],
    warning_gates: list[str],
    gate_name: str,
    verdict: str,
) -> None:
    gate_results[gate_name] = verdict
    if verdict == VALIDATION_VERDICT_FAIL:
        failed_gates.append(gate_name)
    elif verdict == VALIDATION_VERDICT_WARN:
        warning_gates.append(gate_name)


def _hard_min_gate(value: float | int | None, threshold: float, *, allow_equal: bool = True) -> str:
    if value is None:
        return VALIDATION_VERDICT_FAIL
    if allow_equal and float(value) >= threshold:
        return VALIDATION_VERDICT_PASS
    if not allow_equal and float(value) > threshold:
        return VALIDATION_VERDICT_PASS
    return VALIDATION_VERDICT_FAIL


def _optional_max_warning_gate(value: float | None, threshold: float) -> str:
    if value is None:
        return VALIDATION_VERDICT_FAIL
    if value > threshold:
        return VALIDATION_VERDICT_WARN
    return VALIDATION_VERDICT_PASS


def build_enhanced_factor_validation_report(
    *,
    run: SingleFactorBacktestRun,
    robustness_report: FactorRobustnessReport | None,
    cost_capacity_report: FactorCostCapacityReport | None,
    thresholds: FactorValidationThresholds | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorValidationReport:
    resolved_thresholds = thresholds or FactorValidationThresholds()
    aggregate = run.aggregate_result
    gate_results: dict[str, str] = {}
    failed_gates: list[str] = []
    warning_gates: list[str] = []

    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "sample_days",
        _hard_min_gate(aggregate.sample_days, resolved_thresholds.min_sample_days),
    )
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "coverage_ratio",
        _hard_min_gate(aggregate.coverage_ratio, resolved_thresholds.min_coverage_ratio),
    )
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "rank_ic_mean",
        _hard_min_gate(aggregate.rank_ic_mean, resolved_thresholds.min_rank_ic_mean),
    )
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "icir",
        _hard_min_gate(aggregate.icir, resolved_thresholds.min_icir),
    )
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "ic_t_stat",
        _hard_min_gate(aggregate.ic_t_stat, resolved_thresholds.min_ic_t_stat),
    )
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "after_cost_sharpe",
        _hard_min_gate(
            aggregate.after_cost_sharpe,
            resolved_thresholds.min_after_cost_sharpe,
        ),
    )
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "positive_ic_ratio",
        _hard_min_gate(
            aggregate.positive_ic_ratio,
            resolved_thresholds.min_positive_ic_ratio,
        ),
    )
    if resolved_thresholds.require_positive_after_cost_spread:
        spread_verdict = _hard_min_gate(
            aggregate.after_cost_top_bottom_spread,
            0.0,
            allow_equal=False,
        )
    else:
        spread_verdict = VALIDATION_VERDICT_PASS
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "after_cost_spread",
        spread_verdict,
    )
    if resolved_thresholds.max_drawdown is not None:
        _add_gate(
            gate_results,
            failed_gates,
            warning_gates,
            "max_drawdown",
            _optional_max_warning_gate(
                aggregate.max_drawdown,
                resolved_thresholds.max_drawdown,
            ),
        )
    if resolved_thresholds.max_turnover is not None:
        _add_gate(
            gate_results,
            failed_gates,
            warning_gates,
            "max_turnover",
            _optional_max_warning_gate(
                aggregate.turnover_avg,
                resolved_thresholds.max_turnover,
            ),
        )

    if robustness_report is None:
        robustness_verdict = VALIDATION_VERDICT_WARN
    elif robustness_report.overall_verdict == ROBUSTNESS_VERDICT_PASS:
        robustness_verdict = VALIDATION_VERDICT_PASS
    elif robustness_report.overall_verdict == ROBUSTNESS_VERDICT_WARN:
        robustness_verdict = VALIDATION_VERDICT_WARN
    else:
        robustness_verdict = VALIDATION_VERDICT_FAIL
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "robustness",
        robustness_verdict,
    )

    if cost_capacity_report is None:
        capacity_verdict = VALIDATION_VERDICT_WARN
    elif cost_capacity_report.verdict == CAPACITY_VERDICT_PASS:
        capacity_verdict = VALIDATION_VERDICT_PASS
    elif cost_capacity_report.verdict == CAPACITY_VERDICT_WARN:
        capacity_verdict = VALIDATION_VERDICT_WARN
    elif cost_capacity_report.verdict == CAPACITY_VERDICT_FAIL:
        capacity_verdict = VALIDATION_VERDICT_FAIL
    else:
        capacity_verdict = VALIDATION_VERDICT_FAIL
    _add_gate(
        gate_results,
        failed_gates,
        warning_gates,
        "cost_capacity",
        capacity_verdict,
    )

    resolved_metadata = _coerce_metadata(metadata)
    source_metadata = {
        **aggregate.metadata,
        **resolved_metadata,
    }
    point_in_time_snapshot: dict[str, Any] = {}
    if resolved_thresholds.require_point_in_time:
        point_in_time_passed = source_metadata.get("point_in_time_passed")
        point_in_time_failed = bool(source_metadata.get("point_in_time_failed", False))
        if point_in_time_passed is True and not point_in_time_failed:
            pit_verdict = VALIDATION_VERDICT_PASS
            point_in_time_snapshot["status"] = "passed"
        else:
            pit_verdict = VALIDATION_VERDICT_FAIL
            point_in_time_snapshot["status"] = "failed"
            point_in_time_snapshot["reason"] = "missing_or_failed_point_in_time_evidence"
        _add_gate(
            gate_results,
            failed_gates,
            warning_gates,
            "point_in_time",
            pit_verdict,
        )
    else:
        point_in_time_snapshot["status"] = "not_required"

    unique_failed = _ordered_unique(failed_gates)
    unique_warnings = _ordered_unique(warning_gates)
    if unique_failed:
        overall_verdict = VALIDATION_VERDICT_FAIL
        recommended_status = FACTOR_STATUS_REJECTED
        rationale = f"Rejected because hard gates failed: {', '.join(unique_failed)}."
    elif unique_warnings:
        overall_verdict = VALIDATION_VERDICT_WARN
        recommended_status = FACTOR_STATUS_PAPER_TRADING
        rationale = f"Core gates passed with warnings: {', '.join(unique_warnings)}."
    else:
        overall_verdict = VALIDATION_VERDICT_PASS
        recommended_status = FACTOR_STATUS_VALIDATED_RESEARCH
        rationale = "All enhanced validation gates passed; production approval remains manual."

    robustness_snapshot: dict[str, Any] = {}
    if robustness_report is not None:
        robustness_snapshot = {
            "report_id": robustness_report.report_id,
            "overall_verdict": robustness_report.overall_verdict,
            "failed_slices": list(robustness_report.failed_slices),
            "warning_slices": list(robustness_report.warning_slices),
            "slice_count": len(robustness_report.slice_results),
        }
    metric_snapshot = {
        "aggregate_result": aggregate.to_dict(),
        "robustness": robustness_snapshot,
    }
    capacity_snapshot = (
        {}
        if cost_capacity_report is None
        else {"cost_capacity_report": cost_capacity_report.to_dict()}
    )
    resolved_metadata.update(
        {
            "offline_only": True,
            "pass": "phase9_pass4",
            "production_auto_approved": False,
        }
    )
    return FactorValidationReport(
        report_id=_make_enhanced_validation_report_id(
            factor_id=aggregate.factor_id,
            factor_version=aggregate.factor_version,
            backtest_result_id=aggregate.result_id,
            generated_at=generated_at,
        ),
        factor_id=aggregate.factor_id,
        factor_version=aggregate.factor_version,
        generated_at=generated_at,
        backtest_result_id=aggregate.result_id,
        thresholds=resolved_thresholds,
        overall_verdict=overall_verdict,
        gate_results=gate_results,
        failed_gates=unique_failed,
        warning_gates=unique_warnings,
        metric_snapshot=metric_snapshot,
        correlation_snapshot={},
        capacity_snapshot=capacity_snapshot,
        point_in_time_snapshot=point_in_time_snapshot,
        recommended_status=recommended_status,
        rationale=rationale,
        metadata=resolved_metadata,
    )


__all__ = [
    "SLICE_FULL_SAMPLE",
    "SLICE_RECENT_1Y",
    "SLICE_RECENT_3Y",
    "SLICE_RECENT_5Y",
    "ROBUSTNESS_VERDICT_PASS",
    "ROBUSTNESS_VERDICT_WARN",
    "ROBUSTNESS_VERDICT_FAIL",
    "FactorSliceSpec",
    "FactorSliceResult",
    "FactorRobustnessReport",
    "make_slice_id",
    "make_robustness_report_id",
    "build_default_recent_slice_specs",
    "build_regime_slice_specs",
    "evaluate_slice_result",
    "build_factor_robustness_report",
    "build_enhanced_factor_validation_report",
]
