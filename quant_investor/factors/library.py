"""Offline production factor library construction and audit helpers.

The helpers in this module only inspect local governance artifacts. They do not
approve factors, fetch data, or connect factor outputs to stock selection,
``PortfolioConstructor``, ``RiskGuard``, providers, LLMs, or execution paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.contribution import (
    CONTRIBUTION_VERDICT_DEGRADES,
    CONTRIBUTION_VERDICT_INSUFFICIENT_DATA,
)
from quant_investor.factors.correlation import CORRELATION_VERDICT_REDUNDANT
from quant_investor.factors.schema import (
    ADMISSION_DECISION_APPROVE_PAPER_TRADING,
    ADMISSION_DECISION_APPROVE_PRODUCTION,
    ADMISSION_DECISION_DISABLE,
    ADMISSION_DECISION_REJECT,
    FACTOR_STATUS_DEPRECATED,
    FACTOR_STATUS_DISABLED,
    FACTOR_STATUS_PAPER_TRADING,
    FACTOR_STATUS_PRODUCTION,
    VALIDATION_VERDICT_FAIL,
    VALIDATION_VERDICT_PASS,
    VALIDATION_VERDICT_WARN,
    FactorAdmissionDecision,
    FactorDefinition,
    FactorLibraryEntry,
    FactorValidationReport,
    ProductionFactorLibrary,
    make_production_library_id,
)
from quant_investor.versioning import (
    FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
    FACTOR_LIBRARY_SCHEMA_VERSION,
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


def build_production_library_from_artifacts(
    *,
    definitions: Sequence[FactorDefinition],
    admission_decisions: Sequence[FactorAdmissionDecision],
    validation_reports: Sequence[FactorValidationReport],
    generated_at: str,
    policy: FactorLibraryPolicy | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProductionFactorLibrary:
    resolved_policy = policy or FactorLibraryPolicy()
    resolved_metadata = _coerce_metadata(metadata)
    definition_keys = [_definition_key(definition) for definition in definitions]
    if len(definition_keys) != len(set(definition_keys)):
        duplicates = sorted(
            {
                key
                for key in definition_keys
                if definition_keys.count(key) > 1
            }
        )
        raise ValueError(f"Duplicate factor definitions: {duplicates}")
    definitions_by_key = {_definition_key(definition): definition for definition in definitions}
    reports_by_id = {report.report_id: report for report in validation_reports}
    production_decisions = [
        decision
        for decision in admission_decisions
        if decision.decision == ADMISSION_DECISION_APPROVE_PRODUCTION
        and decision.target_status == FACTOR_STATUS_PRODUCTION
    ]
    latest_decisions = _latest_by_decided_at(production_decisions)
    entries: list[FactorLibraryEntry] = []
    seen: set[tuple[str, str]] = set()

    for key, decision in sorted(latest_decisions.items()):
        definition = definitions_by_key.get(key)
        if resolved_policy.require_definition and definition is None:
            continue
        report = reports_by_id.get(str(decision.validation_report_id or ""))
        if resolved_policy.require_validation_report and report is None:
            continue
        if report is not None:
            if _report_key(report) != key:
                continue
            if report.overall_verdict == VALIDATION_VERDICT_FAIL:
                continue
            if report.overall_verdict not in {VALIDATION_VERDICT_PASS, VALIDATION_VERDICT_WARN}:
                continue
        if key in seen:
            raise ValueError(f"Duplicate production factor entry: {key[0]} {key[1]}")
        seen.add(key)

        production_since = str(resolved_metadata.get("production_since") or decision.decided_at)
        expires_at = decision.expires_at or iso_date_add_days(
            generated_at,
            resolved_policy.production_revalidation_days,
        )
        owner = definition.owner if definition is not None else None
        entry_metadata = {
            "source": "build_production_library_from_artifacts",
            "policy_id": resolved_policy.policy_id,
            "factor_library_schema_version": FACTOR_LIBRARY_SCHEMA_VERSION,
        }
        if "entry_metadata" in resolved_metadata and isinstance(
            resolved_metadata["entry_metadata"],
            Mapping,
        ):
            entry_metadata.update(dict(resolved_metadata["entry_metadata"]))
        entry = FactorLibraryEntry(
            factor_id=decision.factor_id,
            factor_version=decision.factor_version,
            status=FACTOR_STATUS_PRODUCTION,
            admission_decision_id=decision.decision_id,
            validation_report_id=decision.validation_report_id,
            production_since=production_since,
            expires_at=expires_at,
            last_revalidation_at=report.generated_at if report is not None else None,
            owner=owner,
            tags=[],
            metadata=entry_metadata,
        )
        entries.append(entry)

    ordered = sorted(entries, key=lambda entry: (entry.factor_id, entry.factor_version))
    library_metadata = {
        **resolved_metadata,
        "factor_library_schema_version": FACTOR_LIBRARY_SCHEMA_VERSION,
        "factor_library_audit_schema_version": FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
        "offline_only": True,
        "requires_explicit_approve_production": (
            resolved_policy.require_explicit_production_approval
        ),
        "policy": resolved_policy.to_dict(),
    }
    return ProductionFactorLibrary(
        library_id=make_production_library_id(ordered),
        generated_at=generated_at,
        entries=ordered,
        metadata=library_metadata,
    )


def audit_factor_library(
    *,
    library: ProductionFactorLibrary | None,
    definitions: Sequence[FactorDefinition] | None = None,
    admission_decisions: Sequence[FactorAdmissionDecision] | None = None,
    validation_reports: Sequence[FactorValidationReport] | None = None,
    redundancy_reports: Sequence[Any] | None = None,
    contribution_reports: Sequence[Any] | None = None,
    policy: FactorLibraryPolicy | None = None,
    as_of: str,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorLibraryAuditReport:
    resolved_policy = policy or FactorLibraryPolicy()
    resolved_metadata = _coerce_metadata(metadata)
    definition_list = list(definitions or [])
    decision_list = list(admission_decisions or [])
    report_list = list(validation_reports or [])
    redundancy_list = list(redundancy_reports or [])
    contribution_list = list(contribution_reports or [])
    entries = list(library.entries if library is not None else [])

    definitions_by_key = {_definition_key(definition): definition for definition in definition_list}
    reports_by_id = {report.report_id: report for report in report_list}
    latest_decisions = _latest_by_decided_at(decision_list)
    production_keys = {_entry_key(entry) for entry in entries}
    issues: list[FactorLibraryAuditIssue] = []

    key_counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        key_counts[_entry_key(entry)] = key_counts.get(_entry_key(entry), 0) + 1
    for key, count in key_counts.items():
        if count > 1:
            issues.append(
                _make_issue(
                    factor_id=key[0],
                    factor_version=key[1],
                    issue_code=FACTOR_LIBRARY_ISSUE_DUPLICATE_FACTOR,
                    severity=FACTOR_LIBRARY_ISSUE_BLOCKER,
                    message=f"Duplicate production factor entry detected for {key[0]} {key[1]}.",
                )
            )

    redundancy_by_key: dict[tuple[str, str], list[Any]] = {}
    for report in redundancy_list:
        key = _report_candidate_key(report)
        if key[0] and key[1]:
            redundancy_by_key.setdefault((key[0], key[1]), []).append(report)
    contribution_by_key: dict[tuple[str, str], list[Any]] = {}
    for report in contribution_list:
        key = _report_candidate_key(report)
        if key[0] and key[1]:
            contribution_by_key.setdefault((key[0], key[1]), []).append(report)

    expired_keys: set[tuple[str, str]] = set()
    if (
        library is None
        and not definition_list
        and not decision_list
        and not report_list
        and not redundancy_list
        and not contribution_list
    ):
        issues.append(
            _make_issue(
                factor_id=None,
                factor_version=None,
                issue_code=FACTOR_LIBRARY_ISSUE_MISSING_ADMISSION_DECISION,
                severity=FACTOR_LIBRARY_ISSUE_WARNING,
                message="No factor governance artifacts were found for production library audit.",
            )
        )
    for entry in entries:
        key = _entry_key(entry)
        if resolved_policy.require_definition and key not in definitions_by_key:
            issues.append(
                _make_issue(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_MISSING_DEFINITION,
                    severity=FACTOR_LIBRARY_ISSUE_BLOCKER,
                    message=f"Production factor {entry.factor_id} lacks a matching definition.",
                )
            )
        report = reports_by_id.get(str(entry.validation_report_id or ""))
        if resolved_policy.require_validation_report and report is None:
            issues.append(
                _make_issue(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_MISSING_VALIDATION_REPORT,
                    severity=FACTOR_LIBRARY_ISSUE_BLOCKER,
                    message=f"Production factor {entry.factor_id} lacks a matching validation report.",
                )
            )
        decision = latest_decisions.get(key)
        if (
            resolved_policy.require_explicit_production_approval
            and (
                decision is None
                or decision.decision != ADMISSION_DECISION_APPROVE_PRODUCTION
                or decision.target_status != FACTOR_STATUS_PRODUCTION
            )
        ):
            issues.append(
                _make_issue(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_MISSING_ADMISSION_DECISION,
                    severity=FACTOR_LIBRARY_ISSUE_BLOCKER,
                    message=(
                        f"Production factor {entry.factor_id} lacks an explicit "
                        "approve_production admission decision."
                    ),
                )
            )
        if entry.status != FACTOR_STATUS_PRODUCTION:
            issues.append(
                _make_issue(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_NON_PRODUCTION_STATUS,
                    severity=FACTOR_LIBRARY_ISSUE_BLOCKER,
                    message=f"Entry {entry.factor_id} is not marked production.",
                )
            )
        if resolved_policy.require_current_validation and not is_validation_current(
            last_revalidation_at=entry.last_revalidation_at,
            production_since=entry.production_since,
            expires_at=entry.expires_at,
            as_of=as_of,
            policy=resolved_policy,
        ):
            expired_keys.add(key)
            issues.append(
                _make_issue(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_EXPIRED_VALIDATION,
                    severity=FACTOR_LIBRARY_ISSUE_BLOCKER,
                    message=f"Production factor {entry.factor_id} validation is expired.",
                    metadata={"as_of": as_of, "expires_at": entry.expires_at},
                )
            )
        if entry.status == FACTOR_STATUS_DISABLED:
            issues.append(
                _make_issue(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_DISABLED_FACTOR,
                    severity=FACTOR_LIBRARY_ISSUE_BLOCKER,
                    message=f"Production factor {entry.factor_id} is disabled.",
                )
            )

        matched_redundancy = redundancy_by_key.get(key, [])
        matched_contribution = contribution_by_key.get(key, [])
        if (
            resolved_policy.require_incremental_review
            and not matched_redundancy
            and not matched_contribution
        ):
            issues.append(
                _make_issue(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_MISSING_INCREMENTAL_REVIEW,
                    severity=FACTOR_LIBRARY_ISSUE_WARNING,
                    message=(
                        f"Production factor {entry.factor_id} lacks redundancy or "
                        "contribution review evidence."
                    ),
                )
            )
        for redundancy_report in matched_redundancy:
            if (
                _get_value(redundancy_report, "overall_verdict")
                == CORRELATION_VERDICT_REDUNDANT
                and not resolved_policy.allow_redundant_factors
            ):
                issues.append(
                    _make_issue(
                        factor_id=entry.factor_id,
                        factor_version=entry.factor_version,
                        issue_code=FACTOR_LIBRARY_ISSUE_REDUNDANT_FACTOR,
                        severity=FACTOR_LIBRARY_ISSUE_WARNING,
                        message=f"Production factor {entry.factor_id} is marked redundant.",
                        metadata={"report": _item_to_dict(redundancy_report)},
                    )
                )
        for contribution_report in matched_contribution:
            if (
                _get_value(contribution_report, "verdict")
                in {CONTRIBUTION_VERDICT_DEGRADES, CONTRIBUTION_VERDICT_INSUFFICIENT_DATA}
                and not resolved_policy.allow_negative_contribution
            ):
                issues.append(
                    _make_issue(
                        factor_id=entry.factor_id,
                        factor_version=entry.factor_version,
                        issue_code=FACTOR_LIBRARY_ISSUE_WEAK_CONTRIBUTION,
                        severity=FACTOR_LIBRARY_ISSUE_WARNING,
                        message=(
                            f"Production factor {entry.factor_id} has weak or "
                            "insufficient contribution evidence."
                        ),
                        metadata={"report": _item_to_dict(contribution_report)},
                    )
                )

    for decision in decision_list:
        key = _decision_key(decision)
        if decision.decision == ADMISSION_DECISION_REJECT:
            issues.append(
                _make_issue(
                    factor_id=decision.factor_id,
                    factor_version=decision.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_REJECTED_FACTOR,
                    severity=(
                        FACTOR_LIBRARY_ISSUE_BLOCKER
                        if key in production_keys
                        else FACTOR_LIBRARY_ISSUE_INFO
                    ),
                    message=f"Factor {decision.factor_id} has a rejected admission decision.",
                    metadata={"decision_id": decision.decision_id},
                )
            )
        if decision.decision == ADMISSION_DECISION_DISABLE:
            issues.append(
                _make_issue(
                    factor_id=decision.factor_id,
                    factor_version=decision.factor_version,
                    issue_code=FACTOR_LIBRARY_ISSUE_DISABLED_FACTOR,
                    severity=(
                        FACTOR_LIBRARY_ISSUE_BLOCKER
                        if key in production_keys
                        else FACTOR_LIBRARY_ISSUE_INFO
                    ),
                    message=f"Factor {decision.factor_id} has a disabled admission decision.",
                    metadata={"decision_id": decision.decision_id},
                )
            )

    for definition in definition_list:
        key = _definition_key(definition)
        if definition.status == FACTOR_STATUS_DISABLED:
            issues.append(
                _make_issue(
                    factor_id=definition.factor_id,
                    factor_version=definition.version,
                    issue_code=FACTOR_LIBRARY_ISSUE_DISABLED_FACTOR,
                    severity=(
                        FACTOR_LIBRARY_ISSUE_BLOCKER
                        if key in production_keys
                        else FACTOR_LIBRARY_ISSUE_INFO
                    ),
                    message=f"Factor definition {definition.factor_id} is disabled.",
                )
            )
        if definition.status == FACTOR_STATUS_DEPRECATED:
            issues.append(
                _make_issue(
                    factor_id=definition.factor_id,
                    factor_version=definition.version,
                    issue_code=FACTOR_LIBRARY_ISSUE_NON_PRODUCTION_STATUS,
                    severity=FACTOR_LIBRARY_ISSUE_INFO,
                    message=f"Factor definition {definition.factor_id} is deprecated.",
                )
            )

    issues = _sort_issues(issues)
    blocker_count = sum(1 for issue in issues if issue.severity == FACTOR_LIBRARY_ISSUE_BLOCKER)
    warning_count = sum(1 for issue in issues if issue.severity == FACTOR_LIBRARY_ISSUE_WARNING)
    info_count = sum(1 for issue in issues if issue.severity == FACTOR_LIBRARY_ISSUE_INFO)
    blocker_keys = {
        (issue.factor_id or "", issue.factor_version or "")
        for issue in issues
        if issue.severity == FACTOR_LIBRARY_ISSUE_BLOCKER and issue.factor_id
    }
    warning_keys = {
        (issue.factor_id or "", issue.factor_version or "")
        for issue in issues
        if issue.severity == FACTOR_LIBRARY_ISSUE_WARNING and issue.factor_id
    }
    allowed = [
        entry.factor_id
        for entry in entries
        if _entry_key(entry) not in blocker_keys
    ]
    blocked = [
        entry.factor_id
        for entry in entries
        if _entry_key(entry) in blocker_keys
    ]

    paper_trading_keys = {
        _decision_key(decision)
        for decision in decision_list
        if decision.decision == ADMISSION_DECISION_APPROVE_PAPER_TRADING
        or decision.target_status == FACTOR_STATUS_PAPER_TRADING
    }
    shadow_keys = {
        key
        for key in warning_keys | paper_trading_keys
        if key not in blocker_keys and key[0]
    }
    shadow = [key[0] for key in sorted(shadow_keys)]

    production_count = len(entries)
    paper_count = len(
        {
            _decision_key(decision)
            for decision in decision_list
            if decision.target_status == FACTOR_STATUS_PAPER_TRADING
            or decision.decision == ADMISSION_DECISION_APPROVE_PAPER_TRADING
        }
    )
    rejected_count = len(
        {
            _decision_key(decision)
            for decision in decision_list
            if decision.decision == ADMISSION_DECISION_REJECT
        }
    )
    deprecated_count = len(
        {_definition_key(definition) for definition in definition_list if definition.status == FACTOR_STATUS_DEPRECATED}
    )
    disabled_count = len(
        {
            *[
                _definition_key(definition)
                for definition in definition_list
                if definition.status == FACTOR_STATUS_DISABLED
            ],
            *[
                _decision_key(decision)
                for decision in decision_list
                if decision.decision == ADMISSION_DECISION_DISABLE
            ],
        }
    )
    verdict = (
        FACTOR_LIBRARY_AUDIT_FAIL
        if blocker_count
        else FACTOR_LIBRARY_AUDIT_WARN
        if warning_count
        else FACTOR_LIBRARY_AUDIT_PASS
    )
    report_id = make_factor_library_audit_report_id(
        library_id=library.library_id if library is not None else None,
        generated_at=generated_at,
    )
    report_metadata = {
        **resolved_metadata,
        "as_of": as_of,
        "offline_only": True,
        "factor_library_audit_schema_version": FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
        "factor_production_guardrail_schema_version": (
            FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION
        ),
    }
    return FactorLibraryAuditReport(
        report_id=report_id,
        generated_at=generated_at,
        policy=resolved_policy,
        library_id=library.library_id if library is not None else None,
        production_factor_count=production_count,
        paper_trading_factor_count=paper_count,
        rejected_factor_count=rejected_count,
        deprecated_factor_count=deprecated_count,
        disabled_factor_count=disabled_count,
        expired_factor_count=len(expired_keys),
        issue_count=len(issues),
        blocker_count=blocker_count,
        warning_count=warning_count,
        info_count=info_count,
        issues=issues,
        allowed_factor_ids=allowed,
        blocked_factor_ids=blocked,
        shadow_only_factor_ids=shadow,
        verdict=verdict,
        metadata=report_metadata,
    )


def evaluate_factor_usage_guardrail(
    *,
    factor_id: str,
    factor_version: str,
    requested_usage: str,
    library: ProductionFactorLibrary | None,
    audit_report: FactorLibraryAuditReport | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorProductionGuardrailResult:
    resolved_factor_id = _non_empty_str(factor_id, "factor_id")
    resolved_factor_version = _non_empty_str(factor_version, "factor_version")
    resolved_usage = _non_empty_str(requested_usage, "requested_usage")
    resolved_metadata = _coerce_metadata(metadata)
    key = (resolved_factor_id, resolved_factor_version)
    library_keys = {
        (entry.factor_id, entry.factor_version): entry
        for entry in (library.entries if library is not None else [])
    }
    in_library = key in library_keys
    blocked_ids = set(audit_report.blocked_factor_ids if audit_report is not None else [])
    blocked = resolved_factor_id in blocked_ids
    shadow_ids = set(audit_report.shadow_only_factor_ids if audit_report is not None else [])
    explicitly_rejected_or_disabled = bool(
        resolved_metadata.get("rejected") or resolved_metadata.get("disabled")
    )
    known_paper_trading = (
        resolved_metadata.get("status") == FACTOR_STATUS_PAPER_TRADING
        or resolved_factor_id in shadow_ids
    )

    reasons: list[str] = []
    if resolved_usage in _FORMAL_USAGE:
        if not in_library:
            reasons.append("missing_from_production_library")
        if blocked:
            reasons.append("blocked_by_factor_library_audit")
        status = FACTOR_GUARDRAIL_ALLOWED if in_library and not blocked else FACTOR_GUARDRAIL_BLOCKED
        allowed = status == FACTOR_GUARDRAIL_ALLOWED
    elif resolved_usage == "research_report":
        if in_library and not blocked:
            status = FACTOR_GUARDRAIL_ALLOWED
            allowed = True
            reasons.append("production_factor_available")
        elif explicitly_rejected_or_disabled:
            status = FACTOR_GUARDRAIL_BLOCKED
            allowed = False
            reasons.append("explicitly_rejected_or_disabled")
        else:
            status = FACTOR_GUARDRAIL_SHADOW_ONLY
            allowed = True
            reasons.append("research_report_shadow_only")
    elif resolved_usage == "shadow_portfolio":
        if explicitly_rejected_or_disabled:
            status = FACTOR_GUARDRAIL_BLOCKED
            allowed = False
            reasons.append("explicitly_rejected_or_disabled")
        elif (in_library and not blocked) or known_paper_trading:
            status = FACTOR_GUARDRAIL_ALLOWED
            allowed = True
            reasons.append("production_or_paper_trading_factor")
        else:
            status = FACTOR_GUARDRAIL_SHADOW_ONLY
            allowed = True
            reasons.append("shadow_only_without_production_admission")
    else:
        status = FACTOR_GUARDRAIL_SHADOW_ONLY if in_library and not blocked else FACTOR_GUARDRAIL_BLOCKED
        allowed = status != FACTOR_GUARDRAIL_BLOCKED
        reasons.append("unknown_usage_requires_shadow_or_block")

    result_metadata = {
        **resolved_metadata,
        "guardrail_result_id": make_factor_guardrail_result_id(
            factor_id=resolved_factor_id,
            factor_version=resolved_factor_version,
            requested_usage=resolved_usage,
        ),
        "library_id": library.library_id if library is not None else None,
        "audit_report_id": audit_report.report_id if audit_report is not None else None,
    }
    return FactorProductionGuardrailResult(
        factor_id=resolved_factor_id,
        factor_version=resolved_factor_version,
        requested_usage=resolved_usage,
        status=status,
        allowed=allowed,
        reasons=reasons,
        metadata=result_metadata,
    )


def build_production_factor_context_patch(
    library: ProductionFactorLibrary,
    audit_report: FactorLibraryAuditReport | None = None,
) -> dict[str, Any]:
    blocked = audit_report.blocked_factor_ids if audit_report is not None else []
    shadow = audit_report.shadow_only_factor_ids if audit_report is not None else []
    factor_versions_by_id: dict[str, list[str]] = {}
    for entry in library.entries:
        factor_versions_by_id.setdefault(entry.factor_id, []).append(entry.factor_version)
    payload = {
        "production_factor_ids": _ordered_unique([entry.factor_id for entry in library.entries]),
        "blocked_factor_ids": _ordered_unique(blocked),
        "shadow_only_factor_ids": _ordered_unique(shadow),
        "factor_versions_by_id": {
            factor_id: _ordered_unique(versions)
            for factor_id, versions in sorted(factor_versions_by_id.items())
        },
        "library_id": library.library_id,
        "audit_report_id": audit_report.report_id if audit_report is not None else None,
        "verdict": audit_report.verdict if audit_report is not None else None,
        "issue_count": audit_report.issue_count if audit_report is not None else 0,
        "metadata": {
            "factor_library_schema_version": FACTOR_LIBRARY_SCHEMA_VERSION,
            "factor_library_audit_schema_version": FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
            "factor_production_guardrail_schema_version": (
                FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION
            ),
            "offline_only": True,
            "not_runtime_wired": True,
        },
    }
    json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return dict(_json_safe(payload))


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
    "build_production_library_from_artifacts",
    "audit_factor_library",
    "evaluate_factor_usage_guardrail",
    "build_production_factor_context_patch",
]
