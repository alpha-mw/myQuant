from __future__ import annotations

import json

from quant_investor.factors.library import (
    FACTOR_LIBRARY_ISSUE_WARNING,
    FactorLibraryAuditIssue,
    FactorLibraryPolicy,
    audit_factor_library,
    build_production_library_from_artifacts,
)
from quant_investor.factors.report import (
    NON_RUNTIME_IMPACT_NOTE,
    build_factor_governance_dashboard_payload,
    render_factor_library_audit_markdown,
)
from tests.unit.test_factor_library import _decision, _definition, _validation_report


def _audit_with_pipe_issue():
    definition = _definition()
    report = _validation_report(definition)
    decision = _decision(definition, report)
    library = build_production_library_from_artifacts(
        definitions=[definition],
        admission_decisions=[decision],
        validation_reports=[report],
        generated_at="2026-04-27",
        policy=FactorLibraryPolicy(require_incremental_review=False),
    )
    audit = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[decision],
        validation_reports=[report],
        policy=FactorLibraryPolicy(require_incremental_review=False),
        as_of="2026-04-28",
        generated_at="2026-04-28",
    )
    audit.issues.append(
        FactorLibraryAuditIssue(
            issue_id="issue-pipe",
            factor_id=definition.factor_id,
            factor_version="v1",
            issue_code="missing_incremental_review",
            severity=FACTOR_LIBRARY_ISSUE_WARNING,
            message="message with | pipe",
        )
    )
    audit.issue_count = len(audit.issues)
    audit.warning_count = 1
    audit.verdict = "warn"
    return library, audit


def test_markdown_contains_required_sections_and_escapes_pipes() -> None:
    _library, audit = _audit_with_pipe_issue()

    markdown = render_factor_library_audit_markdown(audit)

    assert audit.report_id in markdown
    assert "Verdict" in markdown
    assert "Counts" in markdown
    assert "Issue Table" in markdown
    assert "message with \\| pipe" in markdown
    assert NON_RUNTIME_IMPACT_NOTE in markdown


def test_dashboard_payload_is_json_serializable_and_contains_expected_fields() -> None:
    library, audit = _audit_with_pipe_issue()

    payload = build_factor_governance_dashboard_payload(
        library=library,
        audit_report=audit,
        metadata={"unit": True},
    )

    json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert payload["verdict"] == "warn"
    assert payload["counts"]["issue_count"] == 1
    assert payload["library_id"] == library.library_id
    assert payload["issues"][0]["message"] == "message with | pipe"
    assert "FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION" in payload["schema_versions"]
