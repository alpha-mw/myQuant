"""Offline factor backtest alignment audit helpers.

This module diagnoses whether factor signals, delayed execution windows, price
fields, and execution-return matrices use the same time axis. It is deliberately
read-only and does not wire factors into stock selection, posterior scoring,
RiskGuard, PortfolioConstructor, orders, providers, LLMs, or execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest import (
    EXECUTION_PRICE_CLOSE,
    EXECUTION_PRICE_OPEN,
    EXECUTION_PRICE_VWAP,
    FactorDailyBacktestRecord,
    SingleFactorBacktestRun,
)
from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_CLOSE,
    FIELD_OPEN,
    FIELD_VOLUME,
    FIELD_VWAP,
    FactorMatrix,
    MatrixDataBundle,
)
from quant_investor.factors.schema import FactorBacktestConfig
from quant_investor.versioning import FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION


ALIGNMENT_AUDIT_PASS = "pass"
ALIGNMENT_AUDIT_WARN = "warn"
ALIGNMENT_AUDIT_FAIL = "fail"

ALIGNMENT_ISSUE_INFO = "info"
ALIGNMENT_ISSUE_WARNING = "warning"
ALIGNMENT_ISSUE_BLOCKER = "blocker"

ALIGNMENT_ISSUE_NON_POSITIVE_DELAY = "non_positive_delay"
ALIGNMENT_ISSUE_SAME_DAY_EXECUTION = "same_day_execution"
ALIGNMENT_ISSUE_EXECUTION_BEFORE_SIGNAL = "execution_before_signal"
ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL = "return_window_overlap_signal"
ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD = "return_matrix_lookahead"
ALIGNMENT_ISSUE_PRICE_FIELD_MISSING = "price_field_missing"
ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING = "derived_vwap_missing"
ALIGNMENT_ISSUE_DATE_ORDER_INVALID = "date_order_invalid"
ALIGNMENT_ISSUE_ALIGNMENT_GAP = "alignment_gap"
ALIGNMENT_ISSUE_UNEXPLAINED_DELAY_POLICY = "unexplained_delay_policy"
ALIGNMENT_ISSUE_INSUFFICIENT_DATES = "insufficient_dates"

ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1 = "signal_t_execute_t_plus_1"
ALIGNMENT_POLICY_CUSTOM = "custom"

DEFAULT_FACTOR_ALIGNMENT_AUDIT_DIR = Path("data/factor_library/alignment_audit")
DEFAULT_ALIGNMENT_AUDIT_REPORTS_FILENAME = "alignment_audit_reports.jsonl"
DEFAULT_ALIGNMENT_AUDIT_MARKDOWN_FILENAME = "alignment_audit_report.md"

ALIGNMENT_AUDIT_NON_RUNTIME_IMPACT_NOTE = (
    "This alignment audit is offline-only and does not alter official scoring, "
    "stock selection, posterior, RiskGuard, PortfolioConstructor, target weights, "
    "orders, providers, LLMs, or execution."
)

SUPPORTED_ALIGNMENT_AUDIT_VERDICTS = {
    ALIGNMENT_AUDIT_PASS,
    ALIGNMENT_AUDIT_WARN,
    ALIGNMENT_AUDIT_FAIL,
}
SUPPORTED_ALIGNMENT_ISSUE_SEVERITIES = {
    ALIGNMENT_ISSUE_INFO,
    ALIGNMENT_ISSUE_WARNING,
    ALIGNMENT_ISSUE_BLOCKER,
}
SUPPORTED_ALIGNMENT_ISSUE_CODES = {
    ALIGNMENT_ISSUE_NON_POSITIVE_DELAY,
    ALIGNMENT_ISSUE_SAME_DAY_EXECUTION,
    ALIGNMENT_ISSUE_EXECUTION_BEFORE_SIGNAL,
    ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL,
    ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD,
    ALIGNMENT_ISSUE_PRICE_FIELD_MISSING,
    ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING,
    ALIGNMENT_ISSUE_DATE_ORDER_INVALID,
    ALIGNMENT_ISSUE_ALIGNMENT_GAP,
    ALIGNMENT_ISSUE_UNEXPLAINED_DELAY_POLICY,
    ALIGNMENT_ISSUE_INSUFFICIENT_DATES,
}
SUPPORTED_ALIGNMENT_POLICIES = {
    ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1,
    ALIGNMENT_POLICY_CUSTOM,
}
SUPPORTED_ALIGNMENT_EXECUTION_PRICES = {
    EXECUTION_PRICE_OPEN,
    EXECUTION_PRICE_VWAP,
    EXECUTION_PRICE_CLOSE,
}

_FLOAT_TOLERANCE = 1e-12
_SEVERITY_ORDER = {
    ALIGNMENT_ISSUE_BLOCKER: 0,
    ALIGNMENT_ISSUE_WARNING: 1,
    ALIGNMENT_ISSUE_INFO: 2,
}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
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
    text = str(value).strip()
    return text or None


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be non-negative integer; got {value!r}.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive integer; got {value!r}.")
    number = int(value)
    if number < 1:
        raise ValueError(f"{field_name} must be >= 1; got {value!r}.")
    return number


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool; got {value!r}.")
    return value


def _validate_supported(value: str, field_name: str, supported: set[str]) -> None:
    if value not in supported:
        raise ValueError(f"{field_name} must be one of {sorted(supported)}; got {value!r}.")


def _resolve_execution_price(value: str | None) -> str:
    execution_price = str(value or EXECUTION_PRICE_VWAP).strip()
    _validate_supported(
        execution_price,
        "execution_price",
        SUPPORTED_ALIGNMENT_EXECUTION_PRICES,
    )
    return execution_price


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


def _to_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _to_positive_price(value: Any) -> float | None:
    number = _to_finite_float(value)
    if number is None or number <= 0.0:
        return None
    return number


def _sorted_issue_codes(values: Sequence[Any]) -> list[str]:
    issue_codes = sorted({str(value).strip() for value in values if str(value).strip()})
    for issue_code in issue_codes:
        _validate_supported(issue_code, "issue_code", SUPPORTED_ALIGNMENT_ISSUE_CODES)
    return issue_codes


def _issue_sort_key(
    issue: "FactorBacktestAlignmentIssue",
) -> tuple[int, str, str, str, str, str]:
    return (
        _SEVERITY_ORDER.get(issue.severity, 99),
        issue.issue_code,
        issue.signal_date or "",
        issue.execution_start_date or "",
        issue.execution_end_date or "",
        issue.issue_id,
    )


@dataclass
class FactorBacktestAlignmentIssue:
    schema_version: str = FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
    issue_id: str = ""
    issue_code: str = ""
    severity: str = ALIGNMENT_ISSUE_WARNING
    message: str = ""
    signal_date: str | None = None
    execution_start_date: str | None = None
    execution_end_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
        )
        self.issue_id = _non_empty_str(self.issue_id, "issue_id")
        self.issue_code = _non_empty_str(self.issue_code, "issue_code")
        _validate_supported(
            self.issue_code,
            "issue_code",
            SUPPORTED_ALIGNMENT_ISSUE_CODES,
        )
        self.severity = _non_empty_str(self.severity, "severity")
        _validate_supported(
            self.severity,
            "severity",
            SUPPORTED_ALIGNMENT_ISSUE_SEVERITIES,
        )
        self.message = _non_empty_str(self.message, "message")
        self.signal_date = _optional_str(self.signal_date)
        self.execution_start_date = _optional_str(self.execution_start_date)
        self.execution_end_date = _optional_str(self.execution_end_date)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "issue_id": self.issue_id,
            "issue_code": self.issue_code,
            "severity": self.severity,
            "message": self.message,
            "signal_date": self.signal_date,
            "execution_start_date": self.execution_start_date,
            "execution_end_date": self.execution_end_date,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorBacktestAlignmentIssue":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get(
                    "schema_version",
                    FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION,
                )
            ),
            issue_id=str(data.get("issue_id", "")),
            issue_code=str(data.get("issue_code", "")),
            severity=str(data.get("severity", ALIGNMENT_ISSUE_WARNING)),
            message=str(data.get("message", "")),
            signal_date=data.get("signal_date"),
            execution_start_date=data.get("execution_start_date"),
            execution_end_date=data.get("execution_end_date"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorBacktestAlignmentAuditConfig:
    schema_version: str = FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
    config_id: str = ""
    expected_policy: str = ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1
    expected_delay_days: int = 1
    expected_holding_period_days: int = 1
    execution_price: str = EXECUTION_PRICE_VWAP
    require_vwap_derivable: bool = True
    require_return_window_after_execution: bool = True
    allow_custom_policy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
        )
        self.config_id = _non_empty_str(self.config_id, "config_id")
        self.expected_policy = _non_empty_str(self.expected_policy, "expected_policy")
        _validate_supported(
            self.expected_policy,
            "expected_policy",
            SUPPORTED_ALIGNMENT_POLICIES,
        )
        self.expected_delay_days = _positive_int(
            self.expected_delay_days,
            "expected_delay_days",
        )
        self.expected_holding_period_days = _positive_int(
            self.expected_holding_period_days,
            "expected_holding_period_days",
        )
        self.execution_price = _resolve_execution_price(self.execution_price)
        self.require_vwap_derivable = _require_bool(
            self.require_vwap_derivable,
            "require_vwap_derivable",
        )
        self.require_return_window_after_execution = _require_bool(
            self.require_return_window_after_execution,
            "require_return_window_after_execution",
        )
        self.allow_custom_policy = _require_bool(
            self.allow_custom_policy,
            "allow_custom_policy",
        )
        if (
            self.expected_policy == ALIGNMENT_POLICY_CUSTOM
            and not self.allow_custom_policy
        ):
            raise ValueError(
                "expected_policy custom requires allow_custom_policy=True."
            )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_id": self.config_id,
            "expected_policy": self.expected_policy,
            "expected_delay_days": self.expected_delay_days,
            "expected_holding_period_days": self.expected_holding_period_days,
            "execution_price": self.execution_price,
            "require_vwap_derivable": self.require_vwap_derivable,
            "require_return_window_after_execution": (
                self.require_return_window_after_execution
            ),
            "allow_custom_policy": self.allow_custom_policy,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "FactorBacktestAlignmentAuditConfig":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get(
                    "schema_version",
                    FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION,
                )
            ),
            config_id=str(data.get("config_id", "")),
            expected_policy=str(
                data.get("expected_policy", ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1)
            ),
            expected_delay_days=int(data.get("expected_delay_days", 1)),
            expected_holding_period_days=int(
                data.get("expected_holding_period_days", 1)
            ),
            execution_price=str(data.get("execution_price", EXECUTION_PRICE_VWAP)),
            require_vwap_derivable=bool(data.get("require_vwap_derivable", True)),
            require_return_window_after_execution=bool(
                data.get("require_return_window_after_execution", True)
            ),
            allow_custom_policy=bool(data.get("allow_custom_policy", False)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class AlignmentAuditRecord:
    schema_version: str = FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
    record_id: str = ""
    signal_date: str = ""
    execution_start_date: str = ""
    execution_end_date: str = ""
    signal_index: int = 0
    execution_start_index: int = 0
    execution_end_index: int = 0
    delay_days: int = 1
    holding_period_days: int = 1
    execution_price: str = EXECUTION_PRICE_VWAP
    expected_return_source_index: int = 0
    observed_weight_source_index: int = 0
    passed: bool = True
    issue_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
        )
        self.record_id = _non_empty_str(self.record_id, "record_id")
        self.signal_date = _non_empty_str(self.signal_date, "signal_date")
        self.execution_start_date = _non_empty_str(
            self.execution_start_date,
            "execution_start_date",
        )
        self.execution_end_date = _non_empty_str(
            self.execution_end_date,
            "execution_end_date",
        )
        self.signal_index = _non_negative_int(self.signal_index, "signal_index")
        self.execution_start_index = _non_negative_int(
            self.execution_start_index,
            "execution_start_index",
        )
        self.execution_end_index = _non_negative_int(
            self.execution_end_index,
            "execution_end_index",
        )
        if self.execution_start_index <= self.signal_index:
            raise ValueError("execution_start_index must be greater than signal_index.")
        if self.execution_end_index <= self.execution_start_index:
            raise ValueError(
                "execution_end_index must be greater than execution_start_index."
            )
        self.delay_days = _positive_int(self.delay_days, "delay_days")
        self.holding_period_days = _positive_int(
            self.holding_period_days,
            "holding_period_days",
        )
        self.execution_price = _resolve_execution_price(self.execution_price)
        self.expected_return_source_index = _non_negative_int(
            self.expected_return_source_index,
            "expected_return_source_index",
        )
        self.observed_weight_source_index = _non_negative_int(
            self.observed_weight_source_index,
            "observed_weight_source_index",
        )
        self.passed = _require_bool(self.passed, "passed")
        self.issue_codes = _sorted_issue_codes(self.issue_codes)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "signal_date": self.signal_date,
            "execution_start_date": self.execution_start_date,
            "execution_end_date": self.execution_end_date,
            "signal_index": self.signal_index,
            "execution_start_index": self.execution_start_index,
            "execution_end_index": self.execution_end_index,
            "delay_days": self.delay_days,
            "holding_period_days": self.holding_period_days,
            "execution_price": self.execution_price,
            "expected_return_source_index": self.expected_return_source_index,
            "observed_weight_source_index": self.observed_weight_source_index,
            "passed": self.passed,
            "issue_codes": list(self.issue_codes),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AlignmentAuditRecord":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get(
                    "schema_version",
                    FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION,
                )
            ),
            record_id=str(data.get("record_id", "")),
            signal_date=str(data.get("signal_date", "")),
            execution_start_date=str(data.get("execution_start_date", "")),
            execution_end_date=str(data.get("execution_end_date", "")),
            signal_index=int(data.get("signal_index", 0)),
            execution_start_index=int(data.get("execution_start_index", 0)),
            execution_end_index=int(data.get("execution_end_index", 0)),
            delay_days=int(data.get("delay_days", 1)),
            holding_period_days=int(data.get("holding_period_days", 1)),
            execution_price=str(data.get("execution_price", EXECUTION_PRICE_VWAP)),
            expected_return_source_index=int(
                data.get("expected_return_source_index", 0)
            ),
            observed_weight_source_index=int(
                data.get("observed_weight_source_index", 0)
            ),
            passed=bool(data.get("passed", True)),
            issue_codes=list(data.get("issue_codes", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorBacktestAlignmentAuditReport:
    schema_version: str = FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    factor_matrix_id: str | None = None
    backtest_run_id: str | None = None
    config: FactorBacktestAlignmentAuditConfig = field(
        default_factory=FactorBacktestAlignmentAuditConfig
    )
    total_records: int = 0
    passed_records: int = 0
    failed_records: int = 0
    issue_count: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    records: list[AlignmentAuditRecord] = field(default_factory=list)
    issues: list[FactorBacktestAlignmentIssue] = field(default_factory=list)
    verdict: str = ALIGNMENT_AUDIT_PASS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
        )
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        self.factor_matrix_id = _optional_str(self.factor_matrix_id)
        self.backtest_run_id = _optional_str(self.backtest_run_id)
        if not isinstance(self.config, FactorBacktestAlignmentAuditConfig):
            self.config = FactorBacktestAlignmentAuditConfig.from_dict(self.config)
        self.records = [
            record if isinstance(record, AlignmentAuditRecord)
            else AlignmentAuditRecord.from_dict(record)
            for record in self.records
        ]
        self.records = sorted(self.records, key=lambda record: record.signal_index)
        self.issues = [
            issue if isinstance(issue, FactorBacktestAlignmentIssue)
            else FactorBacktestAlignmentIssue.from_dict(issue)
            for issue in self.issues
        ]
        self.issues = sorted(self.issues, key=_issue_sort_key)
        self.total_records = len(self.records)
        self.failed_records = sum(1 for record in self.records if not record.passed)
        self.passed_records = self.total_records - self.failed_records
        self.issue_count = len(self.issues)
        self.blocker_count = sum(
            1 for issue in self.issues if issue.severity == ALIGNMENT_ISSUE_BLOCKER
        )
        self.warning_count = sum(
            1 for issue in self.issues if issue.severity == ALIGNMENT_ISSUE_WARNING
        )
        self.info_count = sum(
            1 for issue in self.issues if issue.severity == ALIGNMENT_ISSUE_INFO
        )
        self.verdict = _non_empty_str(self.verdict, "verdict")
        _validate_supported(
            self.verdict,
            "verdict",
            SUPPORTED_ALIGNMENT_AUDIT_VERDICTS,
        )
        for field_name in (
            "total_records",
            "passed_records",
            "failed_records",
            "issue_count",
            "blocker_count",
            "warning_count",
            "info_count",
        ):
            _non_negative_int(getattr(self, field_name), field_name)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "factor_matrix_id": self.factor_matrix_id,
            "backtest_run_id": self.backtest_run_id,
            "config": self.config.to_dict(),
            "total_records": self.total_records,
            "passed_records": self.passed_records,
            "failed_records": self.failed_records,
            "issue_count": self.issue_count,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "records": [record.to_dict() for record in self.records],
            "issues": [issue.to_dict() for issue in self.issues],
            "verdict": self.verdict,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorBacktestAlignmentAuditReport":
        data = dict(payload)
        config_payload = data.get("config", {}) or {}
        return cls(
            schema_version=str(
                data.get(
                    "schema_version",
                    FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION,
                )
            ),
            report_id=str(data.get("report_id", "")),
            generated_at=str(data.get("generated_at", "")),
            factor_matrix_id=data.get("factor_matrix_id"),
            backtest_run_id=data.get("backtest_run_id"),
            config=FactorBacktestAlignmentAuditConfig.from_dict(config_payload)
            if isinstance(config_payload, Mapping)
            else config_payload,
            total_records=int(data.get("total_records", 0)),
            passed_records=int(data.get("passed_records", 0)),
            failed_records=int(data.get("failed_records", 0)),
            issue_count=int(data.get("issue_count", 0)),
            blocker_count=int(data.get("blocker_count", 0)),
            warning_count=int(data.get("warning_count", 0)),
            info_count=int(data.get("info_count", 0)),
            records=[
                AlignmentAuditRecord.from_dict(record)
                for record in list(data.get("records", []) or [])
                if isinstance(record, Mapping)
            ],
            issues=[
                FactorBacktestAlignmentIssue.from_dict(issue)
                for issue in list(data.get("issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            verdict=str(data.get("verdict", ALIGNMENT_AUDIT_PASS)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_alignment_audit_config_id(config: FactorBacktestAlignmentAuditConfig) -> str:
    parts = [
        config.schema_version,
        config.expected_policy,
        config.expected_delay_days,
        config.expected_holding_period_days,
        config.execution_price,
        config.require_vwap_derivable,
        config.require_return_window_after_execution,
        config.allow_custom_policy,
        config.metadata,
    ]
    return (
        f"alignment-audit-config-{_slug(config.expected_policy)}-"
        f"d{config.expected_delay_days}-h{config.expected_holding_period_days}-"
        f"{_slug(config.execution_price)}-{_short_hash(parts)}"
    )


def make_alignment_issue_id(
    *,
    issue_code: str,
    signal_date: str | None,
    execution_start_date: str | None,
    execution_end_date: str | None,
    message: str,
) -> str:
    parts = [
        str(issue_code),
        signal_date,
        execution_start_date,
        execution_end_date,
        str(message),
    ]
    return f"alignment-issue-{_slug(issue_code)}-{_short_hash(parts)}"


def make_alignment_record_id(
    *,
    signal_date: str,
    execution_start_date: str,
    execution_end_date: str,
    execution_price: str,
) -> str:
    parts = [
        str(signal_date),
        str(execution_start_date),
        str(execution_end_date),
        str(execution_price),
    ]
    return (
        f"alignment-record-{_slug(signal_date)}-{_slug(execution_start_date)}-"
        f"{_slug(execution_end_date)}-{_slug(execution_price)}-{_short_hash(parts)}"
    )


def make_alignment_audit_report_id(
    *,
    factor_matrix_id: str | None,
    backtest_run_id: str | None,
    generated_at: str,
) -> str:
    parts = [factor_matrix_id, backtest_run_id, str(generated_at)]
    return (
        f"alignment-audit-report-{_slug(factor_matrix_id)}-"
        f"{_slug(backtest_run_id)}-{_slug(generated_at)}-{_short_hash(parts)}"
    )


def validate_strictly_ascending_dates(dates: Sequence[str]) -> None:
    resolved_dates = [str(value).strip() for value in dates if str(value).strip()]
    if not resolved_dates:
        raise ValueError("dates must be non-empty.")
    parsed_dates: list[date] = []
    for value in resolved_dates:
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"dates must be ISO dates; got {value!r}.") from exc
        if parsed_value.isoformat() != value:
            raise ValueError(f"dates must be canonical ISO dates; got {value!r}.")
        parsed_dates.append(parsed_value)
    if len(set(resolved_dates)) != len(resolved_dates):
        raise ValueError("dates must not contain duplicates.")
    if any(current >= next_value for current, next_value in zip(parsed_dates, parsed_dates[1:])):
        raise ValueError("dates must be strictly ascending ISO dates.")


def expected_alignment_tuples(
    dates: Sequence[str],
    *,
    delay_days: int,
    holding_period_days: int,
    start_date: str | None = None,
    end_date: str | None = None,
    execution_price: str = EXECUTION_PRICE_VWAP,
) -> list[dict[str, Any]]:
    validate_strictly_ascending_dates(dates)
    resolved_dates = [str(value).strip() for value in dates if str(value).strip()]
    resolved_delay = _positive_int(delay_days, "delay_days")
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    resolved_execution_price = _resolve_execution_price(execution_price)
    if start_date is not None:
        date.fromisoformat(start_date)
    if end_date is not None:
        date.fromisoformat(end_date)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be <= end_date.")

    alignments: list[dict[str, Any]] = []
    for signal_index, signal_date in enumerate(resolved_dates):
        if start_date is not None and signal_date < start_date:
            continue
        if end_date is not None and signal_date > end_date:
            continue
        execution_start_index = signal_index + resolved_delay
        execution_end_index = execution_start_index + resolved_holding_period
        if execution_end_index >= len(resolved_dates):
            continue
        execution_start_date = resolved_dates[execution_start_index]
        execution_end_date = resolved_dates[execution_end_index]
        alignments.append(
            {
                "signal_date": signal_date,
                "execution_start_date": execution_start_date,
                "execution_end_date": execution_end_date,
                "signal_index": signal_index,
                "execution_start_index": execution_start_index,
                "execution_end_index": execution_end_index,
                "delay_days": resolved_delay,
                "holding_period_days": resolved_holding_period,
                "execution_price": resolved_execution_price,
            }
        )
    return alignments


def _make_issue(
    *,
    issue_code: str,
    severity: str,
    message: str,
    signal_date: str | None = None,
    execution_start_date: str | None = None,
    execution_end_date: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorBacktestAlignmentIssue:
    return FactorBacktestAlignmentIssue(
        issue_id=make_alignment_issue_id(
            issue_code=issue_code,
            signal_date=signal_date,
            execution_start_date=execution_start_date,
            execution_end_date=execution_end_date,
            message=message,
        ),
        issue_code=issue_code,
        severity=severity,
        message=message,
        signal_date=signal_date,
        execution_start_date=execution_start_date,
        execution_end_date=execution_end_date,
        metadata=dict(metadata or {}),
    )


def _field_for_execution_price(execution_price: str) -> str:
    resolved = _resolve_execution_price(execution_price)
    if resolved == EXECUTION_PRICE_OPEN:
        return FIELD_OPEN
    if resolved == EXECUTION_PRICE_CLOSE:
        return FIELD_CLOSE
    return FIELD_VWAP


def _blank_price_matrix(bundle: MatrixDataBundle) -> list[list[float | None]]:
    return [[None for _date in bundle.contract.dates] for _symbol in bundle.contract.symbols]


def _derive_vwap_matrix(bundle: MatrixDataBundle) -> list[list[float | None]]:
    amount = bundle.get_field(FIELD_AMOUNT)
    volume = bundle.get_field(FIELD_VOLUME)
    output = _blank_price_matrix(bundle)
    for row_index, (amount_row, volume_row) in enumerate(zip(amount, volume)):
        for column_index, (amount_value, volume_value) in enumerate(zip(amount_row, volume_row)):
            amount_number = _to_finite_float(amount_value)
            volume_number = _to_finite_float(volume_value)
            if amount_number is None or volume_number is None or volume_number == 0.0:
                continue
            output[row_index][column_index] = amount_number / volume_number
    return output


def _resolve_price_matrix(
    bundle: MatrixDataBundle,
    execution_price: str,
    *,
    require_vwap_derivable: bool = True,
) -> tuple[list[list[Any]] | None, list[FactorBacktestAlignmentIssue]]:
    resolved_execution_price = _resolve_execution_price(execution_price)
    field_name = _field_for_execution_price(resolved_execution_price)
    if field_name != FIELD_VWAP:
        if bundle.has_field(field_name):
            return bundle.get_field(field_name), []
        return None, [
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_PRICE_FIELD_MISSING,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message=f"execution price field {field_name!r} is missing from bundle.",
                metadata={"execution_price": resolved_execution_price, "field_name": field_name},
            )
        ]

    if bundle.has_field(FIELD_VWAP):
        return bundle.get_field(FIELD_VWAP), []
    if bundle.has_field(FIELD_AMOUNT) and bundle.has_field(FIELD_VOLUME):
        return _derive_vwap_matrix(bundle), []
    issue_code = (
        ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING
        if require_vwap_derivable
        else ALIGNMENT_ISSUE_PRICE_FIELD_MISSING
    )
    return None, [
        _make_issue(
            issue_code=issue_code,
            severity=ALIGNMENT_ISSUE_BLOCKER,
            message="vwap is missing and cannot be derived from amount/volume.",
            metadata={
                "execution_price": resolved_execution_price,
                "required_fields": [FIELD_AMOUNT, FIELD_VOLUME],
                "has_amount": bundle.has_field(FIELD_AMOUNT),
                "has_volume": bundle.has_field(FIELD_VOLUME),
            },
        )
    ]


def _validate_matrix_shape(
    matrix: Sequence[Sequence[Any]],
    *,
    rows: int,
    columns: int,
    field_name: str,
) -> None:
    if len(matrix) != rows:
        raise ValueError(f"{field_name} must have {rows} rows; got {len(matrix)}.")
    for row_index, row in enumerate(matrix):
        if len(row) != columns:
            raise ValueError(
                f"{field_name} row {row_index} must have {columns} columns; got {len(row)}."
            )


def _forward_return_at(
    price_row: Sequence[Any],
    column_index: int,
    holding_period_days: int,
) -> float | None:
    future_index = column_index + holding_period_days
    if future_index >= len(price_row):
        return None
    start_price = _to_positive_price(price_row[column_index])
    end_price = _to_positive_price(price_row[future_index])
    if start_price is None or end_price is None:
        return None
    return end_price / start_price - 1.0


def _previous_return_at(price_row: Sequence[Any], column_index: int) -> float | None:
    if column_index <= 0:
        return None
    previous_price = _to_positive_price(price_row[column_index - 1])
    current_price = _to_positive_price(price_row[column_index])
    if previous_price is None or current_price is None:
        return None
    return current_price / previous_price - 1.0


def _approximately_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= _FLOAT_TOLERANCE


def audit_execution_return_matrix_alignment(
    *,
    bundle: MatrixDataBundle,
    execution_return_matrix: Sequence[Sequence[float | None]],
    execution_price: str,
    holding_period_days: int,
) -> list[FactorBacktestAlignmentIssue]:
    resolved_execution_price = _resolve_execution_price(execution_price)
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    symbols = list(bundle.contract.symbols)
    dates = list(bundle.contract.dates)
    validate_strictly_ascending_dates(dates)
    _validate_matrix_shape(
        execution_return_matrix,
        rows=len(symbols),
        columns=len(dates),
        field_name="execution_return_matrix",
    )
    prices, issues = _resolve_price_matrix(
        bundle,
        resolved_execution_price,
        require_vwap_derivable=True,
    )
    if prices is None:
        return sorted(issues, key=_issue_sort_key)
    _validate_matrix_shape(
        prices,
        rows=len(symbols),
        columns=len(dates),
        field_name="execution_price_matrix",
    )

    output = list(issues)
    for row_index, symbol in enumerate(symbols):
        price_row = prices[row_index]
        observed_row = execution_return_matrix[row_index]
        for column_index, signal_date in enumerate(dates):
            execution_end_index = column_index + resolved_holding_period
            execution_end_date = (
                dates[execution_end_index]
                if execution_end_index < len(dates)
                else None
            )
            expected_return = _forward_return_at(
                price_row,
                column_index,
                resolved_holding_period,
            )
            observed_return = _to_finite_float(observed_row[column_index])
            if expected_return is None and observed_return is None:
                continue
            if _approximately_equal(observed_return, expected_return):
                continue
            previous_return = _previous_return_at(price_row, column_index)
            if (
                previous_return is not None
                and observed_return is not None
                and _approximately_equal(observed_return, previous_return)
            ):
                output.append(
                    _make_issue(
                        issue_code=ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD,
                        severity=ALIGNMENT_ISSUE_BLOCKER,
                        message=(
                            f"execution return for {symbol} on {signal_date} matches "
                            "a prior close-to-close return instead of the forward execution window."
                        ),
                        signal_date=signal_date,
                        execution_start_date=signal_date,
                        execution_end_date=execution_end_date,
                        metadata={
                            "symbol": symbol,
                            "row_index": row_index,
                            "date_index": column_index,
                            "expected_return": expected_return,
                            "observed_return": observed_return,
                            "prior_return": previous_return,
                            "execution_price": resolved_execution_price,
                            "holding_period_days": resolved_holding_period,
                        },
                    )
                )
                continue
            output.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                    severity=ALIGNMENT_ISSUE_WARNING,
                    message=(
                        f"execution return for {symbol} on {signal_date} does not match "
                        "the expected forward execution window."
                    ),
                    signal_date=signal_date,
                    execution_start_date=signal_date,
                    execution_end_date=execution_end_date,
                    metadata={
                        "symbol": symbol,
                        "row_index": row_index,
                        "date_index": column_index,
                        "expected_return": expected_return,
                        "observed_return": observed_return,
                        "execution_price": resolved_execution_price,
                        "holding_period_days": resolved_holding_period,
                    },
                )
            )
    return sorted(output, key=_issue_sort_key)


def _holding_period_from_run(run: SingleFactorBacktestRun | None) -> int | None:
    if run is None:
        return None
    candidates: list[Any] = []
    candidates.append(run.metadata.get("holding_period_days"))
    candidates.append(run.aggregate_result.metadata.get("holding_period_days"))
    for record in run.daily_records:
        candidates.append(record.metadata.get("holding_period_days"))
        alignment_payload = record.metadata.get("alignment")
        if isinstance(alignment_payload, Mapping):
            candidates.append(alignment_payload.get("holding_period_days"))
    for candidate in candidates:
        if candidate is None:
            continue
        return _positive_int(candidate, "holding_period_days")
    return None


def _build_default_audit_config(
    config: FactorBacktestConfig,
    run: SingleFactorBacktestRun | None,
) -> FactorBacktestAlignmentAuditConfig:
    holding_period_days = _holding_period_from_run(run) or 1
    expected_policy = (
        ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1
        if config.delay_days == 1 and holding_period_days == 1
        else ALIGNMENT_POLICY_CUSTOM
    )
    audit_config = FactorBacktestAlignmentAuditConfig(
        config_id="placeholder",
        expected_policy=expected_policy,
        expected_delay_days=config.delay_days,
        expected_holding_period_days=holding_period_days,
        execution_price=config.execution_price,
        allow_custom_policy=expected_policy == ALIGNMENT_POLICY_CUSTOM,
        metadata={"source": "FactorBacktestConfig"},
    )
    audit_config.config_id = make_alignment_audit_config_id(audit_config)
    return audit_config


def _factor_bundle_symbols_dates_match(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
) -> bool:
    return (
        list(factor_matrix.symbols) == list(bundle.contract.symbols)
        and list(factor_matrix.dates) == list(bundle.contract.dates)
    )


def _calendar_index(dates: Sequence[str]) -> dict[str, int]:
    return {date_value: index for index, date_value in enumerate(dates)}


def _record_chronology_issues(
    record: FactorDailyBacktestRecord,
    date_to_index: Mapping[str, int],
) -> list[FactorBacktestAlignmentIssue]:
    output: list[FactorBacktestAlignmentIssue] = []
    signal_index = date_to_index.get(record.signal_date)
    execution_start_index = date_to_index.get(record.execution_start_date)
    execution_end_index = date_to_index.get(record.execution_end_date)
    if signal_index is None or execution_start_index is None or execution_end_index is None:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                severity=ALIGNMENT_ISSUE_WARNING,
                message="run daily record contains dates outside the factor matrix calendar.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={
                    "record_date": record.date,
                    "signal_date_in_calendar": signal_index is not None,
                    "execution_start_date_in_calendar": execution_start_index is not None,
                    "execution_end_date_in_calendar": execution_end_index is not None,
                },
            )
        )
        return output
    if execution_start_index == signal_index:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_SAME_DAY_EXECUTION,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="run daily record executes on the same date as the signal.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={"record_date": record.date},
            )
        )
    if execution_start_index < signal_index:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_EXECUTION_BEFORE_SIGNAL,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="run daily record execution starts before the signal date.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={"record_date": record.date},
            )
        )
    if execution_end_index <= execution_start_index:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="run daily record return window does not end after execution start.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={"record_date": record.date},
            )
        )
    if record.date != record.execution_end_date:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_DATE_ORDER_INVALID,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="run daily record date must equal execution_end_date.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={"record_date": record.date},
            )
        )
    return output


def _run_record_alignment_issues(
    alignments: Sequence[Mapping[str, Any]],
    run: SingleFactorBacktestRun,
    dates: Sequence[str],
) -> list[FactorBacktestAlignmentIssue]:
    output: list[FactorBacktestAlignmentIssue] = []
    date_to_index = _calendar_index(dates)
    run_records = sorted(run.daily_records, key=lambda record: record.date)
    expected_count = len(alignments)
    observed_count = len(run_records)
    if expected_count != observed_count:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                severity=ALIGNMENT_ISSUE_WARNING,
                message=(
                    "run daily record count does not match expected alignment count "
                    f"({observed_count} observed vs {expected_count} expected)."
                ),
                metadata={
                    "observed_count": observed_count,
                    "expected_count": expected_count,
                    "run_id": run.run_id,
                },
            )
        )
    for record in run_records:
        output.extend(_record_chronology_issues(record, date_to_index))
    for index, alignment in enumerate(alignments):
        if index >= len(run_records):
            break
        record = run_records[index]
        expected_fields = {
            "signal_date": str(alignment["signal_date"]),
            "execution_start_date": str(alignment["execution_start_date"]),
            "execution_end_date": str(alignment["execution_end_date"]),
            "date": str(alignment["execution_end_date"]),
        }
        observed_fields = {
            "signal_date": record.signal_date,
            "execution_start_date": record.execution_start_date,
            "execution_end_date": record.execution_end_date,
            "date": record.date,
        }
        mismatches = {
            field_name: {
                "expected": expected_value,
                "observed": observed_fields[field_name],
            }
            for field_name, expected_value in expected_fields.items()
            if observed_fields[field_name] != expected_value
        }
        if mismatches:
            output.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                    severity=ALIGNMENT_ISSUE_WARNING,
                    message="run daily record does not match expected alignment tuple.",
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    metadata={
                        "run_id": run.run_id,
                        "record_index": index,
                        "record_date": record.date,
                        "mismatches": mismatches,
                    },
                )
            )
    return output


def _issues_for_alignment(
    issues: Sequence[FactorBacktestAlignmentIssue],
    alignment: Mapping[str, Any],
) -> list[FactorBacktestAlignmentIssue]:
    signal_date = str(alignment["signal_date"])
    execution_start_date = str(alignment["execution_start_date"])
    execution_end_date = str(alignment["execution_end_date"])
    return [
        issue for issue in issues
        if (
            issue.signal_date in (None, signal_date)
            and issue.execution_start_date in (None, execution_start_date)
            and issue.execution_end_date in (None, execution_end_date)
        )
    ]


def _verdict_from_issues(issues: Sequence[FactorBacktestAlignmentIssue]) -> str:
    if any(issue.severity == ALIGNMENT_ISSUE_BLOCKER for issue in issues):
        return ALIGNMENT_AUDIT_FAIL
    if any(issue.severity == ALIGNMENT_ISSUE_WARNING for issue in issues):
        return ALIGNMENT_AUDIT_WARN
    return ALIGNMENT_AUDIT_PASS


def _deduplicate_issues(
    issues: Sequence[FactorBacktestAlignmentIssue],
) -> list[FactorBacktestAlignmentIssue]:
    by_id: dict[str, FactorBacktestAlignmentIssue] = {}
    for issue in issues:
        by_id.setdefault(issue.issue_id, issue)
    return sorted(by_id.values(), key=_issue_sort_key)


def audit_factor_backtest_alignment(
    *,
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    config: FactorBacktestConfig,
    run: SingleFactorBacktestRun | None = None,
    audit_config: FactorBacktestAlignmentAuditConfig | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorBacktestAlignmentAuditReport:
    if not _factor_bundle_symbols_dates_match(factor_matrix, bundle):
        raise ValueError("factor_matrix symbols/dates must match bundle contract.")
    validate_strictly_ascending_dates(factor_matrix.dates)
    validate_strictly_ascending_dates(bundle.contract.dates)
    resolved_audit_config = audit_config or _build_default_audit_config(config, run)
    resolved_execution_price = _resolve_execution_price(resolved_audit_config.execution_price)
    issues: list[FactorBacktestAlignmentIssue] = []

    _price_matrix, price_issues = _resolve_price_matrix(
        bundle,
        resolved_execution_price,
        require_vwap_derivable=resolved_audit_config.require_vwap_derivable,
    )
    issues.extend(price_issues)

    if resolved_audit_config.expected_policy == ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1:
        if resolved_audit_config.expected_delay_days != 1:
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="T+1 alignment policy requires expected_delay_days=1.",
                    metadata={"expected_delay_days": resolved_audit_config.expected_delay_days},
                )
            )
        if resolved_audit_config.expected_holding_period_days != 1:
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="T+1 alignment policy requires expected_holding_period_days=1.",
                    metadata={
                        "expected_holding_period_days": (
                            resolved_audit_config.expected_holding_period_days
                        )
                    },
                )
            )
    if (
        resolved_audit_config.expected_policy == ALIGNMENT_POLICY_CUSTOM
        and not resolved_audit_config.allow_custom_policy
    ):
        issues.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_UNEXPLAINED_DELAY_POLICY,
                severity=ALIGNMENT_ISSUE_WARNING,
                message="custom alignment policy is not explicitly allowed.",
                metadata={"expected_policy": resolved_audit_config.expected_policy},
            )
        )

    alignments = expected_alignment_tuples(
        factor_matrix.dates,
        delay_days=resolved_audit_config.expected_delay_days,
        holding_period_days=resolved_audit_config.expected_holding_period_days,
        start_date=config.start_date or None,
        end_date=config.end_date or None,
        execution_price=resolved_execution_price,
    )
    if not alignments:
        issues.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_INSUFFICIENT_DATES,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="no signal/execution/return windows fit the requested audit config.",
                metadata={
                    "date_count": len(factor_matrix.dates),
                    "delay_days": resolved_audit_config.expected_delay_days,
                    "holding_period_days": resolved_audit_config.expected_holding_period_days,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                },
            )
        )

    for alignment in alignments:
        if int(alignment["delay_days"]) < 1:
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_NON_POSITIVE_DELAY,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="delay_days must be positive.",
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    metadata={"delay_days": alignment["delay_days"]},
                )
            )
        if int(alignment["execution_start_index"]) <= int(alignment["signal_index"]):
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_SAME_DAY_EXECUTION,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="execution_start_index must be after signal_index.",
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    metadata=dict(alignment),
                )
            )
        if int(alignment["execution_end_index"]) <= int(alignment["execution_start_index"]):
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="execution_end_index must be after execution_start_index.",
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    metadata=dict(alignment),
                )
            )

    if run is not None:
        issues.extend(_run_record_alignment_issues(alignments, run, factor_matrix.dates))

    issues = _deduplicate_issues(issues)
    records: list[AlignmentAuditRecord] = []
    for alignment in alignments:
        alignment_issues = _issues_for_alignment(issues, alignment)
        record_issue_codes = [issue.issue_code for issue in alignment_issues]
        record_passed = not any(
            issue.severity == ALIGNMENT_ISSUE_BLOCKER for issue in alignment_issues
        )
        records.append(
            AlignmentAuditRecord(
                record_id=make_alignment_record_id(
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    execution_price=resolved_execution_price,
                ),
                signal_date=str(alignment["signal_date"]),
                execution_start_date=str(alignment["execution_start_date"]),
                execution_end_date=str(alignment["execution_end_date"]),
                signal_index=int(alignment["signal_index"]),
                execution_start_index=int(alignment["execution_start_index"]),
                execution_end_index=int(alignment["execution_end_index"]),
                delay_days=int(alignment["delay_days"]),
                holding_period_days=int(alignment["holding_period_days"]),
                execution_price=resolved_execution_price,
                expected_return_source_index=int(alignment["execution_start_index"]),
                observed_weight_source_index=int(alignment["signal_index"]),
                passed=record_passed,
                issue_codes=record_issue_codes,
                metadata={
                    "return_window_after_signal": True,
                    "return_window_after_execution_start": (
                        int(alignment["execution_end_index"])
                        > int(alignment["execution_start_index"])
                    ),
                    "weight_source": "signal_date",
                    "return_source": "execution_start_date",
                },
            )
        )

    report_metadata = _coerce_metadata(metadata)
    report_metadata.update(
        {
            "factor_backtest_alignment_audit_schema_version": (
                FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
            ),
            "delay_days": resolved_audit_config.expected_delay_days,
            "holding_period_days": resolved_audit_config.expected_holding_period_days,
            "execution_price": resolved_execution_price,
            "non_runtime_impact": True,
        }
    )
    verdict = _verdict_from_issues(issues)
    return FactorBacktestAlignmentAuditReport(
        report_id=make_alignment_audit_report_id(
            factor_matrix_id=factor_matrix.matrix_id,
            backtest_run_id=run.run_id if run is not None else None,
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        factor_matrix_id=factor_matrix.matrix_id,
        backtest_run_id=run.run_id if run is not None else None,
        config=resolved_audit_config,
        total_records=len(records),
        passed_records=sum(1 for record in records if record.passed),
        failed_records=sum(1 for record in records if not record.passed),
        issue_count=len(issues),
        blocker_count=sum(1 for issue in issues if issue.severity == ALIGNMENT_ISSUE_BLOCKER),
        warning_count=sum(1 for issue in issues if issue.severity == ALIGNMENT_ISSUE_WARNING),
        info_count=sum(1 for issue in issues if issue.severity == ALIGNMENT_ISSUE_INFO),
        records=records,
        issues=issues,
        verdict=verdict,
        metadata=report_metadata,
    )


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_alignment_audit_markdown(
    report: FactorBacktestAlignmentAuditReport,
) -> str:
    lines = [
        "# Factor Backtest Alignment Audit",
        "",
        f"Generated at: `{_markdown_cell(report.generated_at)}`",
        "",
        f"Verdict: `{_markdown_cell(report.verdict)}`",
        "",
        "## Config Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Policy | `{_markdown_cell(report.config.expected_policy)}` |",
        f"| Delay days | `{report.config.expected_delay_days}` |",
        f"| Holding period days | `{report.config.expected_holding_period_days}` |",
        f"| Execution price | `{_markdown_cell(report.config.execution_price)}` |",
        f"| Require VWAP derivable | `{report.config.require_vwap_derivable}` |",
        f"| Require return window after execution | `{report.config.require_return_window_after_execution}` |",
        f"| Allow custom policy | `{report.config.allow_custom_policy}` |",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Total records | {report.total_records} |",
        f"| Passed records | {report.passed_records} |",
        f"| Failed records | {report.failed_records} |",
        f"| Issues | {report.issue_count} |",
        f"| Blockers | {report.blocker_count} |",
        f"| Warnings | {report.warning_count} |",
        f"| Info | {report.info_count} |",
        "",
        "## Alignment Records",
        "",
        "| Signal | Execute start | Execute end | Delay | Hold | Price | Weight idx | Return idx | Passed | Issues |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    if report.records:
        for record in report.records:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(record.signal_date),
                        _markdown_cell(record.execution_start_date),
                        _markdown_cell(record.execution_end_date),
                        str(record.delay_days),
                        str(record.holding_period_days),
                        _markdown_cell(record.execution_price),
                        str(record.observed_weight_source_index),
                        str(record.expected_return_source_index),
                        _markdown_cell(record.passed),
                        _markdown_cell(", ".join(record.issue_codes)),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _No records_ |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Issues",
            "",
            "| Severity | Code | Signal | Execute start | Execute end | Message |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if report.issues:
        for issue in report.issues:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(issue.severity),
                        _markdown_cell(issue.issue_code),
                        _markdown_cell(issue.signal_date),
                        _markdown_cell(issue.execution_start_date),
                        _markdown_cell(issue.execution_end_date),
                        _markdown_cell(issue.message),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _No issues_ |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Non-Runtime Impact",
            "",
            ALIGNMENT_AUDIT_NON_RUNTIME_IMPACT_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ALIGNMENT_AUDIT_PASS",
    "ALIGNMENT_AUDIT_WARN",
    "ALIGNMENT_AUDIT_FAIL",
    "ALIGNMENT_ISSUE_INFO",
    "ALIGNMENT_ISSUE_WARNING",
    "ALIGNMENT_ISSUE_BLOCKER",
    "ALIGNMENT_ISSUE_NON_POSITIVE_DELAY",
    "ALIGNMENT_ISSUE_SAME_DAY_EXECUTION",
    "ALIGNMENT_ISSUE_EXECUTION_BEFORE_SIGNAL",
    "ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL",
    "ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD",
    "ALIGNMENT_ISSUE_PRICE_FIELD_MISSING",
    "ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING",
    "ALIGNMENT_ISSUE_DATE_ORDER_INVALID",
    "ALIGNMENT_ISSUE_ALIGNMENT_GAP",
    "ALIGNMENT_ISSUE_UNEXPLAINED_DELAY_POLICY",
    "ALIGNMENT_ISSUE_INSUFFICIENT_DATES",
    "ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1",
    "ALIGNMENT_POLICY_CUSTOM",
    "DEFAULT_FACTOR_ALIGNMENT_AUDIT_DIR",
    "DEFAULT_ALIGNMENT_AUDIT_REPORTS_FILENAME",
    "DEFAULT_ALIGNMENT_AUDIT_MARKDOWN_FILENAME",
    "ALIGNMENT_AUDIT_NON_RUNTIME_IMPACT_NOTE",
    "FactorBacktestAlignmentIssue",
    "FactorBacktestAlignmentAuditConfig",
    "AlignmentAuditRecord",
    "FactorBacktestAlignmentAuditReport",
    "make_alignment_audit_config_id",
    "make_alignment_issue_id",
    "make_alignment_record_id",
    "make_alignment_audit_report_id",
    "validate_strictly_ascending_dates",
    "expected_alignment_tuples",
    "audit_execution_return_matrix_alignment",
    "audit_factor_backtest_alignment",
    "render_alignment_audit_markdown",
]
