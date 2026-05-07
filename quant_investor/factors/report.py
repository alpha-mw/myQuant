"""Markdown and JSON payload renderers for offline factor governance audits."""

from __future__ import annotations

import json
from typing import Any, Mapping

from quant_investor.factors.library import FactorLibraryAuditReport
from quant_investor.factors.schema import ProductionFactorLibrary
from quant_investor.versioning import (
    FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
    FACTOR_LIBRARY_SCHEMA_VERSION,
    FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION,
)


NON_RUNTIME_IMPACT_NOTE = (
    "This factor library audit is generated offline and does not alter stock "
    "selection, PortfolioConstructor, RiskGuard, providers, LLMs, web, or "
    "broker/execution behavior."
)


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
    return value


def _escape_pipe(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _count_payload(report: FactorLibraryAuditReport) -> dict[str, int]:
    return {
        "production_factor_count": report.production_factor_count,
        "paper_trading_factor_count": report.paper_trading_factor_count,
        "rejected_factor_count": report.rejected_factor_count,
        "deprecated_factor_count": report.deprecated_factor_count,
        "disabled_factor_count": report.disabled_factor_count,
        "expired_factor_count": report.expired_factor_count,
        "issue_count": report.issue_count,
        "blocker_count": report.blocker_count,
        "warning_count": report.warning_count,
        "info_count": report.info_count,
    }


def _render_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- `{value}`" for value in values]


def render_factor_library_audit_markdown(report: FactorLibraryAuditReport) -> str:
    lines = [
        f"# Factor Library Audit: {report.report_id}",
        "",
        f"Generated at: `{report.generated_at}`",
        "",
        "## Verdict",
        "",
        f"`{report.verdict}`",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key, value in _count_payload(report).items():
        lines.append(f"| `{_escape_pipe(key)}` | {value} |")

    lines.extend(["", "## Allowed Factors", ""])
    lines.extend(_render_list(report.allowed_factor_ids))
    lines.extend(["", "## Blocked Factors", ""])
    lines.extend(_render_list(report.blocked_factor_ids))
    lines.extend(["", "## Shadow-only Factors", ""])
    lines.extend(_render_list(report.shadow_only_factor_ids))

    lines.extend(
        [
            "",
            "## Issue Table",
            "",
            "| Severity | Code | Factor | Version | Message |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    if report.issues:
        for issue in report.issues:
            lines.append(
                "| "
                f"`{_escape_pipe(issue.severity)}` | "
                f"`{_escape_pipe(issue.issue_code)}` | "
                f"{_escape_pipe(issue.factor_id or '')} | "
                f"{_escape_pipe(issue.factor_version or '')} | "
                f"{_escape_pipe(issue.message)} |"
            )
    else:
        lines.append("|  |  |  |  | No issues. |")

    lines.extend(
        [
            "",
            "## Policy Snapshot",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )
    for key, value in report.policy.to_dict().items():
        if key == "metadata":
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = value
        lines.append(f"| `{_escape_pipe(key)}` | `{_escape_pipe(rendered)}` |")

    lines.extend(
        [
            "",
            "## Runtime Impact",
            "",
            NON_RUNTIME_IMPACT_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def build_factor_governance_dashboard_payload(
    *,
    library: ProductionFactorLibrary | None,
    audit_report: FactorLibraryAuditReport,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "verdict": audit_report.verdict,
        "counts": _count_payload(audit_report),
        "allowed_factor_ids": list(audit_report.allowed_factor_ids),
        "blocked_factor_ids": list(audit_report.blocked_factor_ids),
        "shadow_only_factor_ids": list(audit_report.shadow_only_factor_ids),
        "issues": [issue.to_dict() for issue in audit_report.issues],
        "policy": audit_report.policy.to_dict(),
        "library_id": library.library_id if library is not None else audit_report.library_id,
        "schema_versions": {
            "FACTOR_LIBRARY_SCHEMA_VERSION": FACTOR_LIBRARY_SCHEMA_VERSION,
            "FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION": FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
            "FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION": (
                FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION
            ),
        },
        "metadata": {
            **dict(metadata or {}),
            "audit_report_id": audit_report.report_id,
            "offline_only": True,
            "not_runtime_wired": True,
        },
    }
    json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return dict(_json_safe(payload))


__all__ = [
    "NON_RUNTIME_IMPACT_NOTE",
    "render_factor_library_audit_markdown",
    "build_factor_governance_dashboard_payload",
]
