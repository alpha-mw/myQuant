"""Contracts for offline factor backtest alignment audits."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest import (
    EXECUTION_PRICE_CLOSE,
    EXECUTION_PRICE_OPEN,
    EXECUTION_PRICE_VWAP,
)
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
    "SUPPORTED_ALIGNMENT_AUDIT_VERDICTS",
    "SUPPORTED_ALIGNMENT_ISSUE_SEVERITIES",
    "SUPPORTED_ALIGNMENT_ISSUE_CODES",
    "SUPPORTED_ALIGNMENT_POLICIES",
    "SUPPORTED_ALIGNMENT_EXECUTION_PRICES",
    "FactorBacktestAlignmentIssue",
    "FactorBacktestAlignmentAuditConfig",
    "AlignmentAuditRecord",
    "FactorBacktestAlignmentAuditReport",
    "make_alignment_audit_config_id",
    "make_alignment_issue_id",
    "make_alignment_record_id",
    "make_alignment_audit_report_id",
]

