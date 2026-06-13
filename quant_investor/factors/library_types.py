"""Contracts and helper types for production factor library audits."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.schema import (
    FactorAdmissionDecision,
    FactorDefinition,
    FactorLibraryEntry,
    FactorValidationReport,
)
from quant_investor.versioning import (
    FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
    FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION,
)


FACTOR_LIBRARY_ISSUE_INFO = "info"
FACTOR_LIBRARY_ISSUE_WARNING = "warning"
FACTOR_LIBRARY_ISSUE_BLOCKER = "blocker"

FACTOR_LIBRARY_ISSUE_MISSING_DEFINITION = "missing_definition"
FACTOR_LIBRARY_ISSUE_MISSING_VALIDATION_REPORT = "missing_validation_report"
FACTOR_LIBRARY_ISSUE_MISSING_ADMISSION_DECISION = "missing_admission_decision"
FACTOR_LIBRARY_ISSUE_NON_PRODUCTION_STATUS = "non_production_status"
FACTOR_LIBRARY_ISSUE_EXPIRED_VALIDATION = "expired_validation"
FACTOR_LIBRARY_ISSUE_REJECTED_FACTOR = "rejected_factor"
FACTOR_LIBRARY_ISSUE_DISABLED_FACTOR = "disabled_factor"
FACTOR_LIBRARY_ISSUE_DUPLICATE_FACTOR = "duplicate_factor"
FACTOR_LIBRARY_ISSUE_REDUNDANT_FACTOR = "redundant_factor"
FACTOR_LIBRARY_ISSUE_WEAK_CONTRIBUTION = "weak_contribution"
FACTOR_LIBRARY_ISSUE_MISSING_INCREMENTAL_REVIEW = "missing_incremental_review"

FACTOR_LIBRARY_AUDIT_PASS = "pass"
FACTOR_LIBRARY_AUDIT_WARN = "warn"
FACTOR_LIBRARY_AUDIT_FAIL = "fail"

FACTOR_GUARDRAIL_ALLOWED = "allowed"
FACTOR_GUARDRAIL_BLOCKED = "blocked"
FACTOR_GUARDRAIL_SHADOW_ONLY = "shadow_only"

SUPPORTED_FACTOR_LIBRARY_ISSUE_SEVERITIES = {
    FACTOR_LIBRARY_ISSUE_INFO,
    FACTOR_LIBRARY_ISSUE_WARNING,
    FACTOR_LIBRARY_ISSUE_BLOCKER,
}
SUPPORTED_FACTOR_LIBRARY_ISSUE_CODES = {
    FACTOR_LIBRARY_ISSUE_MISSING_DEFINITION,
    FACTOR_LIBRARY_ISSUE_MISSING_VALIDATION_REPORT,
    FACTOR_LIBRARY_ISSUE_MISSING_ADMISSION_DECISION,
    FACTOR_LIBRARY_ISSUE_NON_PRODUCTION_STATUS,
    FACTOR_LIBRARY_ISSUE_EXPIRED_VALIDATION,
    FACTOR_LIBRARY_ISSUE_REJECTED_FACTOR,
    FACTOR_LIBRARY_ISSUE_DISABLED_FACTOR,
    FACTOR_LIBRARY_ISSUE_DUPLICATE_FACTOR,
    FACTOR_LIBRARY_ISSUE_REDUNDANT_FACTOR,
    FACTOR_LIBRARY_ISSUE_WEAK_CONTRIBUTION,
    FACTOR_LIBRARY_ISSUE_MISSING_INCREMENTAL_REVIEW,
}
SUPPORTED_FACTOR_LIBRARY_AUDIT_VERDICTS = {
    FACTOR_LIBRARY_AUDIT_PASS,
    FACTOR_LIBRARY_AUDIT_WARN,
    FACTOR_LIBRARY_AUDIT_FAIL,
}
SUPPORTED_FACTOR_GUARDRAIL_STATUSES = {
    FACTOR_GUARDRAIL_ALLOWED,
    FACTOR_GUARDRAIL_BLOCKED,
    FACTOR_GUARDRAIL_SHADOW_ONLY,
}

_SEVERITY_ORDER = {
    FACTOR_LIBRARY_ISSUE_BLOCKER: 0,
    FACTOR_LIBRARY_ISSUE_WARNING: 1,
    FACTOR_LIBRARY_ISSUE_INFO: 2,
}
_FORMAL_USAGE = {"stock_selection", "portfolio_construction"}


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
    text = str(value).strip()
    return text or None


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive; got {value!r}.")
    return number


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool; got {value!r}.")
    return value


def _validate_supported(value: str, field_name: str, supported: set[str]) -> None:
    if value not in supported:
        raise ValueError(f"{field_name} must be one of {sorted(supported)}; got {value!r}.")


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


def _parse_iso_date(value: str) -> date:
    text = _non_empty_str(value, "date")
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(f"Expected ISO date string; got {value!r}.") from exc


def _factor_key(factor_id: str | None, factor_version: str | None) -> tuple[str, str]:
    return (str(factor_id or ""), str(factor_version or ""))


def _entry_key(entry: FactorLibraryEntry) -> tuple[str, str]:
    return _factor_key(entry.factor_id, entry.factor_version)


def _definition_key(definition: FactorDefinition) -> tuple[str, str]:
    return _factor_key(definition.factor_id, definition.version)


def _decision_key(decision: FactorAdmissionDecision) -> tuple[str, str]:
    return _factor_key(decision.factor_id, decision.factor_version)


def _report_key(report: FactorValidationReport) -> tuple[str, str]:
    return _factor_key(report.factor_id, report.factor_version)


def _latest_by_decided_at(
    decisions: Sequence[FactorAdmissionDecision],
) -> dict[tuple[str, str], FactorAdmissionDecision]:
    latest: dict[tuple[str, str], FactorAdmissionDecision] = {}
    for decision in decisions:
        key = _decision_key(decision)
        current = latest.get(key)
        if current is None or (decision.decided_at, decision.decision_id) > (
            current.decided_at,
            current.decision_id,
        ):
            latest[key] = decision
    return latest


def _get_value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _item_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    if hasattr(item, "to_dict"):
        payload = item.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _report_candidate_key(item: Any) -> tuple[str | None, str | None]:
    return (
        _optional_str(_get_value(item, "candidate_factor_id") or _get_value(item, "factor_id")),
        _optional_str(
            _get_value(item, "candidate_factor_version") or _get_value(item, "factor_version")
        ),
    )


def _sort_issues(issues: Sequence["FactorLibraryAuditIssue"]) -> list["FactorLibraryAuditIssue"]:
    return sorted(
        issues,
        key=lambda issue: (
            _SEVERITY_ORDER[issue.severity],
            issue.factor_id or "",
            issue.factor_version or "",
            issue.issue_code,
            issue.issue_id,
        ),
    )


def _make_issue(
    *,
    factor_id: str | None,
    factor_version: str | None,
    issue_code: str,
    severity: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> "FactorLibraryAuditIssue":
    return FactorLibraryAuditIssue(
        issue_id=make_factor_library_audit_issue_id(
            factor_id=factor_id,
            factor_version=factor_version,
            issue_code=issue_code,
            message=message,
        ),
        factor_id=factor_id,
        factor_version=factor_version,
        issue_code=issue_code,
        severity=severity,
        message=message,
        metadata=dict(metadata or {}),
    )


@dataclass
class FactorLibraryPolicy:
    schema_version: str = FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION
    policy_id: str = ""
    require_definition: bool = True
    require_validation_report: bool = True
    require_explicit_production_approval: bool = True
    require_current_validation: bool = True
    require_incremental_review: bool = True
    allow_redundant_factors: bool = False
    allow_negative_contribution: bool = False
    production_revalidation_days: int = 90
    paper_trading_revalidation_days: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION)
        for field_name in [
            "require_definition",
            "require_validation_report",
            "require_explicit_production_approval",
            "require_current_validation",
            "require_incremental_review",
            "allow_redundant_factors",
            "allow_negative_contribution",
        ]:
            setattr(self, field_name, _require_bool(getattr(self, field_name), field_name))
        self.production_revalidation_days = _positive_int(
            self.production_revalidation_days,
            "production_revalidation_days",
        )
        self.paper_trading_revalidation_days = _positive_int(
            self.paper_trading_revalidation_days,
            "paper_trading_revalidation_days",
        )
        self.metadata = _coerce_metadata(self.metadata)
        if not str(self.policy_id).strip():
            self.policy_id = make_factor_library_policy_id(self)
        else:
            self.policy_id = _non_empty_str(self.policy_id, "policy_id")

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorLibraryPolicy":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION)),
            policy_id=str(data.get("policy_id", "")),
            require_definition=data.get("require_definition", True),
            require_validation_report=data.get("require_validation_report", True),
            require_explicit_production_approval=data.get(
                "require_explicit_production_approval",
                True,
            ),
            require_current_validation=data.get("require_current_validation", True),
            require_incremental_review=data.get("require_incremental_review", True),
            allow_redundant_factors=data.get("allow_redundant_factors", False),
            allow_negative_contribution=data.get("allow_negative_contribution", False),
            production_revalidation_days=int(data.get("production_revalidation_days", 90)),
            paper_trading_revalidation_days=int(
                data.get("paper_trading_revalidation_days", 30)
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorLibraryAuditIssue:
    schema_version: str = FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION
    issue_id: str = ""
    factor_id: str | None = None
    factor_version: str | None = None
    issue_code: str = ""
    severity: str = FACTOR_LIBRARY_ISSUE_INFO
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION)
        self.issue_id = _non_empty_str(self.issue_id, "issue_id")
        self.factor_id = _optional_str(self.factor_id)
        self.factor_version = _optional_str(self.factor_version)
        self.issue_code = _non_empty_str(self.issue_code, "issue_code")
        _validate_supported(
            self.issue_code,
            "issue_code",
            SUPPORTED_FACTOR_LIBRARY_ISSUE_CODES,
        )
        self.severity = _non_empty_str(self.severity, "severity")
        _validate_supported(
            self.severity,
            "severity",
            SUPPORTED_FACTOR_LIBRARY_ISSUE_SEVERITIES,
        )
        self.message = _non_empty_str(self.message, "message")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorLibraryAuditIssue":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION)),
            issue_id=str(data.get("issue_id", "")),
            factor_id=data.get("factor_id"),
            factor_version=data.get("factor_version"),
            issue_code=str(data.get("issue_code", "")),
            severity=str(data.get("severity", FACTOR_LIBRARY_ISSUE_INFO)),
            message=str(data.get("message", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorLibraryAuditReport:
    schema_version: str = FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    policy: FactorLibraryPolicy = field(default_factory=FactorLibraryPolicy)
    library_id: str | None = None
    production_factor_count: int = 0
    paper_trading_factor_count: int = 0
    rejected_factor_count: int = 0
    deprecated_factor_count: int = 0
    disabled_factor_count: int = 0
    expired_factor_count: int = 0
    issue_count: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issues: list[FactorLibraryAuditIssue] = field(default_factory=list)
    allowed_factor_ids: list[str] = field(default_factory=list)
    blocked_factor_ids: list[str] = field(default_factory=list)
    shadow_only_factor_ids: list[str] = field(default_factory=list)
    verdict: str = FACTOR_LIBRARY_AUDIT_PASS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION)
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        if not isinstance(self.policy, FactorLibraryPolicy):
            self.policy = FactorLibraryPolicy.from_dict(self.policy)
        self.library_id = _optional_str(self.library_id)
        for field_name in [
            "production_factor_count",
            "paper_trading_factor_count",
            "rejected_factor_count",
            "deprecated_factor_count",
            "disabled_factor_count",
            "expired_factor_count",
            "issue_count",
            "blocker_count",
            "warning_count",
            "info_count",
        ]:
            setattr(self, field_name, _non_negative_int(getattr(self, field_name), field_name))
        self.issues = _sort_issues([
            issue
            if isinstance(issue, FactorLibraryAuditIssue)
            else FactorLibraryAuditIssue.from_dict(issue)
            for issue in self.issues
        ])
        self.allowed_factor_ids = _ordered_unique(self.allowed_factor_ids)
        self.blocked_factor_ids = _ordered_unique(self.blocked_factor_ids)
        self.shadow_only_factor_ids = _ordered_unique(self.shadow_only_factor_ids)
        self.verdict = _non_empty_str(self.verdict, "verdict")
        _validate_supported(self.verdict, "verdict", SUPPORTED_FACTOR_LIBRARY_AUDIT_VERDICTS)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "policy": self.policy.to_dict(),
            "library_id": self.library_id,
            "production_factor_count": self.production_factor_count,
            "paper_trading_factor_count": self.paper_trading_factor_count,
            "rejected_factor_count": self.rejected_factor_count,
            "deprecated_factor_count": self.deprecated_factor_count,
            "disabled_factor_count": self.disabled_factor_count,
            "expired_factor_count": self.expired_factor_count,
            "issue_count": self.issue_count,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "allowed_factor_ids": list(self.allowed_factor_ids),
            "blocked_factor_ids": list(self.blocked_factor_ids),
            "shadow_only_factor_ids": list(self.shadow_only_factor_ids),
            "verdict": self.verdict,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorLibraryAuditReport":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            generated_at=str(data.get("generated_at", "")),
            policy=FactorLibraryPolicy.from_dict(dict(data.get("policy", {}) or {})),
            library_id=data.get("library_id"),
            production_factor_count=int(data.get("production_factor_count", 0)),
            paper_trading_factor_count=int(data.get("paper_trading_factor_count", 0)),
            rejected_factor_count=int(data.get("rejected_factor_count", 0)),
            deprecated_factor_count=int(data.get("deprecated_factor_count", 0)),
            disabled_factor_count=int(data.get("disabled_factor_count", 0)),
            expired_factor_count=int(data.get("expired_factor_count", 0)),
            issue_count=int(data.get("issue_count", 0)),
            blocker_count=int(data.get("blocker_count", 0)),
            warning_count=int(data.get("warning_count", 0)),
            info_count=int(data.get("info_count", 0)),
            issues=[
                FactorLibraryAuditIssue.from_dict(issue)
                for issue in list(data.get("issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            allowed_factor_ids=list(data.get("allowed_factor_ids", []) or []),
            blocked_factor_ids=list(data.get("blocked_factor_ids", []) or []),
            shadow_only_factor_ids=list(data.get("shadow_only_factor_ids", []) or []),
            verdict=str(data.get("verdict", FACTOR_LIBRARY_AUDIT_PASS)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorProductionGuardrailResult:
    schema_version: str = FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION
    factor_id: str = ""
    factor_version: str = ""
    requested_usage: str = ""
    status: str = FACTOR_GUARDRAIL_BLOCKED
    allowed: bool = False
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION
        )
        self.factor_id = _non_empty_str(self.factor_id, "factor_id")
        self.factor_version = _non_empty_str(self.factor_version, "factor_version")
        self.requested_usage = _non_empty_str(self.requested_usage, "requested_usage")
        self.status = _non_empty_str(self.status, "status")
        _validate_supported(self.status, "status", SUPPORTED_FACTOR_GUARDRAIL_STATUSES)
        self.allowed = _require_bool(self.allowed, "allowed")
        self.reasons = _ordered_unique(self.reasons)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorProductionGuardrailResult":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get(
                    "schema_version",
                    FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION,
                )
            ),
            factor_id=str(data.get("factor_id", "")),
            factor_version=str(data.get("factor_version", "")),
            requested_usage=str(data.get("requested_usage", "")),
            status=str(data.get("status", FACTOR_GUARDRAIL_BLOCKED)),
            allowed=bool(data.get("allowed", False)),
            reasons=list(data.get("reasons", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_factor_library_policy_id(policy: FactorLibraryPolicy) -> str:
    payload = policy.to_dict() if hasattr(policy, "to_dict") else dict(asdict(policy))
    payload["policy_id"] = ""
    return f"factor-library-policy-{_short_hash([payload])}"


def make_factor_library_audit_issue_id(
    *,
    factor_id: str | None,
    factor_version: str | None,
    issue_code: str,
    message: str,
) -> str:
    parts = [factor_id, factor_version, issue_code, message]
    return (
        f"factor-library-issue-{_slug(factor_id)}-{_slug(factor_version)}-"
        f"{_slug(issue_code)}-{_short_hash(parts)}"
    )


def make_factor_library_audit_report_id(
    *,
    library_id: str | None,
    generated_at: str,
) -> str:
    parts = [library_id, generated_at]
    return f"factor-library-audit-{_slug(library_id)}-{_short_hash(parts)}"


def make_factor_guardrail_result_id(
    *,
    factor_id: str,
    factor_version: str,
    requested_usage: str,
) -> str:
    parts = [factor_id, factor_version, requested_usage]
    return (
        f"factor-guardrail-{_slug(factor_id)}-{_slug(factor_version)}-"
        f"{_slug(requested_usage)}-{_short_hash(parts)}"
    )


def iso_date_add_days(date_str: str, days: int) -> str:
    number = _positive_int(days, "days")
    return (_parse_iso_date(date_str) + timedelta(days=number)).isoformat()


def iso_date_is_after(left: str, right: str) -> bool:
    return _parse_iso_date(left) > _parse_iso_date(right)


def is_validation_current(
    *,
    last_revalidation_at: str | None,
    production_since: str | None,
    expires_at: str | None,
    as_of: str,
    policy: FactorLibraryPolicy,
) -> bool:
    if expires_at:
        return not iso_date_is_after(as_of, expires_at)
    base_date = last_revalidation_at or production_since
    if not base_date:
        return False
    expiry = iso_date_add_days(base_date, policy.production_revalidation_days)
    return not iso_date_is_after(as_of, expiry)

__all__ = [
    "FACTOR_LIBRARY_ISSUE_INFO",
    "FACTOR_LIBRARY_ISSUE_WARNING",
    "FACTOR_LIBRARY_ISSUE_BLOCKER",
    "FACTOR_LIBRARY_ISSUE_MISSING_DEFINITION",
    "FACTOR_LIBRARY_ISSUE_MISSING_VALIDATION_REPORT",
    "FACTOR_LIBRARY_ISSUE_MISSING_ADMISSION_DECISION",
    "FACTOR_LIBRARY_ISSUE_NON_PRODUCTION_STATUS",
    "FACTOR_LIBRARY_ISSUE_EXPIRED_VALIDATION",
    "FACTOR_LIBRARY_ISSUE_REJECTED_FACTOR",
    "FACTOR_LIBRARY_ISSUE_DISABLED_FACTOR",
    "FACTOR_LIBRARY_ISSUE_DUPLICATE_FACTOR",
    "FACTOR_LIBRARY_ISSUE_REDUNDANT_FACTOR",
    "FACTOR_LIBRARY_ISSUE_WEAK_CONTRIBUTION",
    "FACTOR_LIBRARY_ISSUE_MISSING_INCREMENTAL_REVIEW",
    "FACTOR_LIBRARY_AUDIT_PASS",
    "FACTOR_LIBRARY_AUDIT_WARN",
    "FACTOR_LIBRARY_AUDIT_FAIL",
    "FACTOR_GUARDRAIL_ALLOWED",
    "FACTOR_GUARDRAIL_BLOCKED",
    "FACTOR_GUARDRAIL_SHADOW_ONLY",
    "SUPPORTED_FACTOR_LIBRARY_ISSUE_SEVERITIES",
    "SUPPORTED_FACTOR_LIBRARY_ISSUE_CODES",
    "SUPPORTED_FACTOR_LIBRARY_AUDIT_VERDICTS",
    "SUPPORTED_FACTOR_GUARDRAIL_STATUSES",
    "FactorLibraryPolicy",
    "FactorLibraryAuditIssue",
    "FactorLibraryAuditReport",
    "FactorProductionGuardrailResult",
    "make_factor_library_policy_id",
    "make_factor_library_audit_issue_id",
    "make_factor_library_audit_report_id",
    "make_factor_guardrail_result_id",
    "iso_date_add_days",
    "iso_date_is_after",
    "is_validation_current",
]
