"""Offline production factor library construction and audit helpers.

The helpers in this module only inspect local governance artifacts. They do not
approve factors, fetch data, or connect factor outputs to stock selection,
``PortfolioConstructor``, ``RiskGuard``, providers, LLMs, or execution paths.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from quant_investor.factors.contribution import (
    CONTRIBUTION_VERDICT_DEGRADES,
    CONTRIBUTION_VERDICT_INSUFFICIENT_DATA,
)
from quant_investor.factors.correlation import CORRELATION_VERDICT_REDUNDANT
from quant_investor.factors.library_types import (
    FACTOR_GUARDRAIL_ALLOWED,
    FACTOR_GUARDRAIL_BLOCKED,
    FACTOR_GUARDRAIL_SHADOW_ONLY,
    FACTOR_LIBRARY_AUDIT_FAIL,
    FACTOR_LIBRARY_AUDIT_PASS,
    FACTOR_LIBRARY_AUDIT_WARN,
    FACTOR_LIBRARY_ISSUE_BLOCKER,
    FACTOR_LIBRARY_ISSUE_DISABLED_FACTOR,
    FACTOR_LIBRARY_ISSUE_DUPLICATE_FACTOR,
    FACTOR_LIBRARY_ISSUE_EXPIRED_VALIDATION,
    FACTOR_LIBRARY_ISSUE_INFO,
    FACTOR_LIBRARY_ISSUE_MISSING_ADMISSION_DECISION,
    FACTOR_LIBRARY_ISSUE_MISSING_DEFINITION,
    FACTOR_LIBRARY_ISSUE_MISSING_INCREMENTAL_REVIEW,
    FACTOR_LIBRARY_ISSUE_MISSING_VALIDATION_REPORT,
    FACTOR_LIBRARY_ISSUE_NON_PRODUCTION_STATUS,
    FACTOR_LIBRARY_ISSUE_REDUNDANT_FACTOR,
    FACTOR_LIBRARY_ISSUE_REJECTED_FACTOR,
    FACTOR_LIBRARY_ISSUE_WEAK_CONTRIBUTION,
    FACTOR_LIBRARY_ISSUE_WARNING,
    SUPPORTED_FACTOR_GUARDRAIL_STATUSES,
    SUPPORTED_FACTOR_LIBRARY_AUDIT_VERDICTS,
    SUPPORTED_FACTOR_LIBRARY_ISSUE_CODES,
    SUPPORTED_FACTOR_LIBRARY_ISSUE_SEVERITIES,
    FactorLibraryAuditIssue,
    FactorLibraryAuditReport,
    FactorLibraryPolicy,
    FactorProductionGuardrailResult,
    _FORMAL_USAGE,
    _coerce_metadata,
    _decision_key,
    _definition_key,
    _entry_key,
    _get_value,
    _item_to_dict,
    _json_safe,
    _latest_by_decided_at,
    _make_issue,
    _non_empty_str,
    _ordered_unique,
    _report_candidate_key,
    _report_key,
    _sort_issues,
    is_validation_current,
    iso_date_add_days,
    iso_date_is_after,
    make_factor_guardrail_result_id,
    make_factor_library_audit_issue_id,
    make_factor_library_audit_report_id,
    make_factor_library_policy_id,
)
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
