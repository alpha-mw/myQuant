"""Markdown renderers for offline tradability audit reports."""

from __future__ import annotations

from typing import Any

from quant_investor.factors.tradability_types import (
    EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE,
    TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE,
    FactorExecutionFeasibilityReport,
    FactorTradabilityAuditReport,
)


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_tradability_audit_markdown(report: FactorTradabilityAuditReport) -> str:
    lines = [
        "# A-share Tradability Audit",
        "",
        f"Generated at: {report.generated_at}",
        f"Verdict: {report.verdict}",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Symbols | {report.symbols_count} |",
        f"| Dates | {report.dates_count} |",
        f"| Tradable cells | {report.tradable_cell_count} |",
        f"| Blocked cells | {report.blocked_cell_count} |",
        f"| Buy-blocked cells | {report.buy_blocked_cell_count} |",
        f"| Sell-blocked cells | {report.sell_blocked_cell_count} |",
        f"| Research-eligible cells | {report.research_eligible_cell_count} |",
        f"| Issues | {report.issue_count} |",
        f"| Blockers | {report.blocker_count} |",
        f"| Warnings | {report.warning_count} |",
        f"| Info | {report.info_count} |",
        "",
        "## Issue Summary",
        "",
        "| Issue code | Count |",
        "| --- | ---: |",
    ]
    if report.issue_summary:
        for issue_code, count in report.issue_summary.items():
            lines.append(f"| {_escape_markdown_cell(issue_code)} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend([
        "",
        "## Issue Table",
        "",
        "| Severity | Symbol | Date | Issue code | Message |",
        "| --- | --- | --- | --- | --- |",
    ])
    if report.issues:
        for issue in report.issues:
            lines.append(
                "| "
                f"{_escape_markdown_cell(issue.severity)} | "
                f"{_escape_markdown_cell(issue.symbol or '')} | "
                f"{_escape_markdown_cell(issue.date or '')} | "
                f"{_escape_markdown_cell(issue.issue_code)} | "
                f"{_escape_markdown_cell(issue.message)} |"
            )
    else:
        lines.append("| none |  |  |  | No tradability issues. |")
    lines.extend([
        "",
        "## Non-runtime-impact Note",
        "",
        TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE,
        "",
    ])
    return "\n".join(lines)


def render_execution_feasibility_markdown(report: FactorExecutionFeasibilityReport) -> str:
    blocked_symbols = ", ".join(report.blocked_symbols) if report.blocked_symbols else "none"
    lines = [
        "# Factor Execution Feasibility Audit",
        "",
        f"Generated at: {report.generated_at}",
        f"Verdict: {report.verdict}",
        "",
        "## Transition Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total transitions | {report.total_transitions} |",
        f"| Feasible transitions | {report.feasible_transitions} |",
        f"| Blocked transitions | {report.blocked_transitions} |",
        f"| Partially feasible transitions | {report.partially_feasible_transitions} |",
        f"| Blocked buy transitions | {report.blocked_buy_count} |",
        f"| Blocked sell transitions | {report.blocked_sell_count} |",
        f"| Issues | {report.issue_count} |",
        f"| Blockers | {report.blocker_count} |",
        f"| Warnings | {report.warning_count} |",
        f"| Info | {report.info_count} |",
        "",
        "## Blocked Symbols",
        "",
        blocked_symbols,
        "",
        "## Issue Table",
        "",
        "| Severity | Symbol | Date | Issue code | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    if report.issues:
        for issue in report.issues:
            lines.append(
                "| "
                f"{_escape_markdown_cell(issue.severity)} | "
                f"{_escape_markdown_cell(issue.symbol or '')} | "
                f"{_escape_markdown_cell(issue.date or '')} | "
                f"{_escape_markdown_cell(issue.issue_code)} | "
                f"{_escape_markdown_cell(issue.message)} |"
            )
    else:
        lines.append("| none |  |  |  | No execution feasibility issues. |")
    lines.extend([
        "",
        "## Transition Sample",
        "",
        "| Execution date | Symbol | Direction | Previous | Target | Trade | Status | Issues |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for record in report.transition_records[:20]:
        lines.append(
            "| "
            f"{_escape_markdown_cell(record.execution_date)} | "
            f"{_escape_markdown_cell(record.symbol)} | "
            f"{_escape_markdown_cell(record.trade_direction)} | "
            f"{record.previous_weight:.6g} | "
            f"{record.target_weight:.6g} | "
            f"{record.trade_weight:.6g} | "
            f"{_escape_markdown_cell(record.status)} | "
            f"{_escape_markdown_cell(','.join(record.issue_codes))} |"
        )
    if not report.transition_records:
        lines.append("|  |  |  | 0 | 0 | 0 | none |  |")
    lines.extend([
        "",
        "## Non-runtime-impact Note",
        "",
        EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE,
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "render_tradability_audit_markdown",
    "render_execution_feasibility_markdown",
]
