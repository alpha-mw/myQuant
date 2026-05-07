from __future__ import annotations

import json

import pytest

from quant_investor.factors.library import (
    FACTOR_GUARDRAIL_ALLOWED,
    FACTOR_GUARDRAIL_BLOCKED,
    FACTOR_GUARDRAIL_SHADOW_ONLY,
    FACTOR_LIBRARY_AUDIT_FAIL,
    FACTOR_LIBRARY_AUDIT_PASS,
    FACTOR_LIBRARY_AUDIT_WARN,
    FACTOR_LIBRARY_ISSUE_EXPIRED_VALIDATION,
    FACTOR_LIBRARY_ISSUE_MISSING_ADMISSION_DECISION,
    FACTOR_LIBRARY_ISSUE_MISSING_INCREMENTAL_REVIEW,
    FACTOR_LIBRARY_ISSUE_MISSING_VALIDATION_REPORT,
    FACTOR_LIBRARY_ISSUE_REDUNDANT_FACTOR,
    FACTOR_LIBRARY_ISSUE_REJECTED_FACTOR,
    FACTOR_LIBRARY_ISSUE_WEAK_CONTRIBUTION,
    FactorLibraryAuditReport,
    FactorLibraryPolicy,
    audit_factor_library,
    build_production_factor_context_patch,
    build_production_library_from_artifacts,
    evaluate_factor_usage_guardrail,
    is_validation_current,
    iso_date_add_days,
)
from quant_investor.factors.schema import (
    ADMISSION_DECISION_APPROVE_PRODUCTION,
    ADMISSION_DECISION_REJECT,
    FACTOR_FAMILY_MOMENTUM,
    FACTOR_STATUS_PRODUCTION,
    FACTOR_STATUS_REJECTED,
    FACTOR_STATUS_RESEARCH_CANDIDATE,
    VALIDATION_VERDICT_FAIL,
    VALIDATION_VERDICT_PASS,
    VALIDATION_VERDICT_WARN,
    FactorAdmissionDecision,
    FactorDefinition,
    FactorValidationReport,
    FactorValidationThresholds,
    make_admission_decision_id,
    make_factor_id,
    make_validation_report_id,
)


def _definition(
    *,
    suffix: str = "a",
    status: str = FACTOR_STATUS_RESEARCH_CANDIDATE,
) -> FactorDefinition:
    expression = f"close / delay(close, 20) - 1 + {len(suffix)}"
    return FactorDefinition(
        factor_id=make_factor_id(
            factor_name=f"Momentum {suffix}",
            factor_family=FACTOR_FAMILY_MOMENTUM,
            expression=expression,
        ),
        factor_name=f"Momentum {suffix}",
        factor_family=FACTOR_FAMILY_MOMENTUM,
        status=status,
        version="v1",
        expression=expression,
        input_fields=["close", "trade_date"],
        data_sources=["local_csv"],
        universe="CN",
        benchmark="CSI300",
        expected_direction=1.0,
        rebalance_frequency="weekly",
        lookback_window=20,
        delay_days=1,
        execution_price="next_open",
        economic_rationale="Fixture momentum rationale.",
        owner="research",
        created_at="2026-04-27",
    )


def _validation_report(
    definition: FactorDefinition,
    *,
    verdict: str = VALIDATION_VERDICT_PASS,
    generated_at: str = "2026-04-27",
) -> FactorValidationReport:
    backtest_result_id = f"bt-{definition.factor_id}"
    return FactorValidationReport(
        report_id=make_validation_report_id(
            factor_id=definition.factor_id,
            factor_version=definition.version,
            backtest_result_id=backtest_result_id,
        ),
        factor_id=definition.factor_id,
        factor_version=definition.version,
        generated_at=generated_at,
        backtest_result_id=backtest_result_id,
        thresholds=FactorValidationThresholds(),
        overall_verdict=verdict,
        gate_results={"sample_days": VALIDATION_VERDICT_PASS},
        failed_gates=[] if verdict != VALIDATION_VERDICT_FAIL else ["sample_days"],
        warning_gates=[] if verdict != VALIDATION_VERDICT_WARN else ["turnover"],
        metric_snapshot={"rank_ic_mean": 0.04},
        recommended_status=FACTOR_STATUS_RESEARCH_CANDIDATE,
        rationale="Fixture validation report.",
    )


def _decision(
    definition: FactorDefinition,
    report: FactorValidationReport | None,
    *,
    decision: str = ADMISSION_DECISION_APPROVE_PRODUCTION,
    target_status: str = FACTOR_STATUS_PRODUCTION,
    decided_at: str = "2026-04-27",
    expires_at: str | None = "2026-07-26",
) -> FactorAdmissionDecision:
    return FactorAdmissionDecision(
        decision_id=make_admission_decision_id(
            factor_id=definition.factor_id,
            factor_version=definition.version,
            decision=decision,
            decided_at=decided_at,
        ),
        factor_id=definition.factor_id,
        factor_version=definition.version,
        validation_report_id=report.report_id if report is not None else "missing-report",
        decision=decision,
        target_status=target_status,
        decided_at=decided_at,
        decided_by="research",
        rationale="Fixture admission decision.",
        expires_at=expires_at,
    )


def _library(
    *,
    definition: FactorDefinition | None = None,
    report: FactorValidationReport | None = None,
    decision: FactorAdmissionDecision | None = None,
    require_incremental_review: bool = False,
):
    resolved_definition = definition or _definition()
    resolved_report = report or _validation_report(resolved_definition)
    resolved_decision = decision or _decision(resolved_definition, resolved_report)
    return build_production_library_from_artifacts(
        definitions=[resolved_definition],
        admission_decisions=[resolved_decision],
        validation_reports=[resolved_report],
        generated_at="2026-04-27",
        policy=FactorLibraryPolicy(require_incremental_review=require_incremental_review),
    )


def test_policy_and_expiration_helpers_round_trip() -> None:
    policy = FactorLibraryPolicy(require_incremental_review=False, metadata={"b": 2})

    assert FactorLibraryPolicy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()
    assert iso_date_add_days("2026-04-27", 3) == "2026-04-30"
    assert is_validation_current(
        last_revalidation_at=None,
        production_since="2026-04-27",
        expires_at="2026-04-28",
        as_of="2026-04-28",
        policy=policy,
    )
    assert not is_validation_current(
        last_revalidation_at=None,
        production_since="2026-04-27",
        expires_at="2026-04-28",
        as_of="2026-04-29",
        policy=policy,
    )
    assert is_validation_current(
        last_revalidation_at="2026-04-27",
        production_since=None,
        expires_at=None,
        as_of="2026-07-26",
        policy=policy,
    )
    assert not is_validation_current(
        last_revalidation_at=None,
        production_since=None,
        expires_at=None,
        as_of="2026-04-27",
        policy=policy,
    )


def test_build_library_requires_explicit_approval_and_pass_or_warn_validation() -> None:
    definition = _definition()
    report = _validation_report(definition)
    library = build_production_library_from_artifacts(
        definitions=[definition],
        admission_decisions=[_decision(definition, report)],
        validation_reports=[report],
        generated_at="2026-04-27",
        policy=FactorLibraryPolicy(require_incremental_review=False),
    )

    assert [entry.factor_id for entry in library.entries] == [definition.factor_id]
    assert library.entries[0].status == FACTOR_STATUS_PRODUCTION
    assert library.entries[0].production_since == "2026-04-27"

    no_decision_library = build_production_library_from_artifacts(
        definitions=[definition],
        admission_decisions=[],
        validation_reports=[report],
        generated_at="2026-04-27",
        policy=FactorLibraryPolicy(require_incremental_review=False),
    )
    assert no_decision_library.entries == []

    failed_report = _validation_report(definition, verdict=VALIDATION_VERDICT_FAIL)
    failed_library = build_production_library_from_artifacts(
        definitions=[definition],
        admission_decisions=[_decision(definition, failed_report)],
        validation_reports=[failed_report],
        generated_at="2026-04-27",
        policy=FactorLibraryPolicy(require_incremental_review=False),
    )
    assert failed_library.entries == []


def test_build_library_missing_validation_policy_and_deterministic_ordering() -> None:
    definition_b = _definition(suffix="b")
    definition_a = _definition(suffix="a")
    report_b = _validation_report(definition_b, verdict=VALIDATION_VERDICT_WARN)
    report_a = _validation_report(definition_a)
    missing_validation_library = build_production_library_from_artifacts(
        definitions=[definition_a],
        admission_decisions=[_decision(definition_a, report_a)],
        validation_reports=[],
        generated_at="2026-04-27",
        policy=FactorLibraryPolicy(require_validation_report=True),
    )
    assert missing_validation_library.entries == []

    loose_library = build_production_library_from_artifacts(
        definitions=[definition_a],
        admission_decisions=[_decision(definition_a, report_a)],
        validation_reports=[],
        generated_at="2026-04-27",
        policy=FactorLibraryPolicy(require_validation_report=False),
    )
    assert [entry.factor_id for entry in loose_library.entries] == [definition_a.factor_id]

    ordered = build_production_library_from_artifacts(
        definitions=[definition_b, definition_a],
        admission_decisions=[
            _decision(definition_b, report_b),
            _decision(definition_a, report_a),
        ],
        validation_reports=[report_b, report_a],
        generated_at="2026-04-27",
        policy=FactorLibraryPolicy(require_incremental_review=False),
    )
    assert [entry.factor_id for entry in ordered.entries] == sorted(
        [definition_b.factor_id, definition_a.factor_id]
    )

    with pytest.raises(ValueError, match="Duplicate factor definitions"):
        build_production_library_from_artifacts(
            definitions=[definition_a, definition_a],
            admission_decisions=[_decision(definition_a, report_a)],
            validation_reports=[report_a],
            generated_at="2026-04-27",
        )


def test_audit_clean_library_passes_and_round_trips() -> None:
    definition = _definition()
    report = _validation_report(definition)
    decision = _decision(definition, report)
    library = _library(definition=definition, report=report, decision=decision)

    audit = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[decision],
        validation_reports=[report],
        redundancy_reports=[
            {"candidate_factor_id": definition.factor_id, "candidate_factor_version": "v1", "overall_verdict": "distinct"}
        ],
        contribution_reports=[
            {"candidate_factor_id": definition.factor_id, "candidate_factor_version": "v1", "verdict": "improves"}
        ],
        policy=FactorLibraryPolicy(require_incremental_review=True),
        as_of="2026-04-28",
        generated_at="2026-04-28",
    )

    assert audit.verdict == FACTOR_LIBRARY_AUDIT_PASS
    assert audit.allowed_factor_ids == [definition.factor_id]
    assert audit.blocked_factor_ids == []
    assert FactorLibraryAuditReport.from_dict(audit.to_dict()).to_dict() == audit.to_dict()


def test_audit_detects_missing_expired_rejected_and_incremental_issues() -> None:
    definition = _definition()
    report = _validation_report(definition)
    decision = _decision(definition, report, expires_at="2026-04-28")
    library = _library(definition=definition, report=report, decision=decision, require_incremental_review=True)
    rejected = _decision(
        definition,
        report,
        decision=ADMISSION_DECISION_REJECT,
        target_status=FACTOR_STATUS_REJECTED,
        decided_at="2026-04-29",
        expires_at=None,
    )

    audit = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[decision, rejected],
        validation_reports=[],
        policy=FactorLibraryPolicy(require_incremental_review=True),
        as_of="2026-05-01",
        generated_at="2026-05-01",
    )
    codes = [issue.issue_code for issue in audit.issues]

    assert audit.verdict == FACTOR_LIBRARY_AUDIT_FAIL
    assert FACTOR_LIBRARY_ISSUE_MISSING_VALIDATION_REPORT in codes
    assert FACTOR_LIBRARY_ISSUE_EXPIRED_VALIDATION in codes
    assert FACTOR_LIBRARY_ISSUE_REJECTED_FACTOR in codes
    assert FACTOR_LIBRARY_ISSUE_MISSING_INCREMENTAL_REVIEW in codes
    assert audit.blocked_factor_ids == [definition.factor_id]

    missing_decision_audit = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[],
        validation_reports=[report],
        policy=FactorLibraryPolicy(require_incremental_review=False),
        as_of="2026-04-28",
        generated_at="2026-04-28",
    )
    assert FACTOR_LIBRARY_ISSUE_MISSING_ADMISSION_DECISION in [
        issue.issue_code for issue in missing_decision_audit.issues
    ]


def test_audit_redundancy_and_contribution_warnings_are_policy_controlled() -> None:
    definition = _definition()
    report = _validation_report(definition)
    decision = _decision(definition, report)
    library = _library(definition=definition, report=report, decision=decision)
    redundancy = {
        "candidate_factor_id": definition.factor_id,
        "candidate_factor_version": "v1",
        "overall_verdict": "redundant",
    }
    contribution = {
        "candidate_factor_id": definition.factor_id,
        "candidate_factor_version": "v1",
        "verdict": "degrades",
    }

    audit = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[decision],
        validation_reports=[report],
        redundancy_reports=[redundancy],
        contribution_reports=[contribution],
        policy=FactorLibraryPolicy(require_incremental_review=True),
        as_of="2026-04-28",
        generated_at="2026-04-28",
    )
    assert audit.verdict == FACTOR_LIBRARY_AUDIT_WARN
    assert [issue.issue_code for issue in audit.issues] == [
        FACTOR_LIBRARY_ISSUE_REDUNDANT_FACTOR,
        FACTOR_LIBRARY_ISSUE_WEAK_CONTRIBUTION,
    ]
    assert audit.shadow_only_factor_ids == [definition.factor_id]

    allowed = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[decision],
        validation_reports=[report],
        redundancy_reports=[redundancy],
        contribution_reports=[contribution],
        policy=FactorLibraryPolicy(
            require_incremental_review=True,
            allow_redundant_factors=True,
            allow_negative_contribution=True,
        ),
        as_of="2026-04-28",
        generated_at="2026-04-28",
    )
    assert allowed.verdict == FACTOR_LIBRARY_AUDIT_PASS


def test_guardrail_and_context_patch_are_json_serializable() -> None:
    definition = _definition()
    report = _validation_report(definition)
    decision = _decision(definition, report)
    library = _library(definition=definition, report=report, decision=decision)
    audit = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[decision],
        validation_reports=[report],
        policy=FactorLibraryPolicy(require_incremental_review=False),
        as_of="2026-04-28",
        generated_at="2026-04-28",
    )

    assert evaluate_factor_usage_guardrail(
        factor_id=definition.factor_id,
        factor_version="v1",
        requested_usage="stock_selection",
        library=library,
        audit_report=audit,
    ).status == FACTOR_GUARDRAIL_ALLOWED
    assert evaluate_factor_usage_guardrail(
        factor_id="missing-factor",
        factor_version="v1",
        requested_usage="stock_selection",
        library=library,
        audit_report=audit,
    ).status == FACTOR_GUARDRAIL_BLOCKED
    assert evaluate_factor_usage_guardrail(
        factor_id="research-factor",
        factor_version="v1",
        requested_usage="research_report",
        library=library,
        audit_report=audit,
    ).status == FACTOR_GUARDRAIL_SHADOW_ONLY

    blocked_audit = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[],
        validation_reports=[report],
        policy=FactorLibraryPolicy(require_incremental_review=False),
        as_of="2026-04-28",
        generated_at="2026-04-28",
    )
    blocked_result = evaluate_factor_usage_guardrail(
        factor_id=definition.factor_id,
        factor_version="v1",
        requested_usage="portfolio_construction",
        library=library,
        audit_report=blocked_audit,
    )
    assert blocked_result.status == FACTOR_GUARDRAIL_BLOCKED

    patch = build_production_factor_context_patch(library, audit)
    json.dumps(patch, ensure_ascii=False, sort_keys=True)
    assert patch["production_factor_ids"] == [definition.factor_id]
