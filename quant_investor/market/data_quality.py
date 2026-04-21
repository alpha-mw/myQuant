from __future__ import annotations

from typing import Iterable

from quant_investor.agent_protocol import DataQualityDiagnostics, DataQualityIssue


def _normalize_symbols(symbols: Iterable[str] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for symbol in symbols or []:
        text = str(symbol or "").strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def build_data_quality_diagnostics(
    *,
    total_symbols: Iterable[str],
    researchable_symbols: Iterable[str],
    shortlistable_symbols: Iterable[str],
    final_selected_symbols: Iterable[str],
    quarantined_symbols: Iterable[str],
    issues: Iterable[DataQualityIssue],
) -> DataQualityDiagnostics:
    total = _normalize_symbols(total_symbols)
    researchable = _normalize_symbols(researchable_symbols)
    shortlistable = _normalize_symbols(shortlistable_symbols)
    final_selected = _normalize_symbols(final_selected_symbols)
    quarantined = _normalize_symbols(quarantined_symbols)
    issue_list = list(issues)
    blocking_issue_count = sum(
        1
        for issue in issue_list
        if str(issue.severity).lower() in {"error", "critical"}
    )
    return DataQualityDiagnostics(
        total_universe_count=len(total),
        researchable_universe_count=len(researchable),
        shortlistable_universe_count=len(shortlistable),
        final_selected_universe_count=len(final_selected),
        quarantined_symbols=quarantined,
        issue_count=len(issue_list),
        blocking_issue_count=blocking_issue_count,
        coverage_tiers={
            "total": len(total),
            "researchable": len(researchable),
            "shortlistable": len(shortlistable),
            "final_selected": len(final_selected),
        },
        issues=issue_list,
        metadata={
            "quarantined_count": len(quarantined),
            "coverage_ratio_researchable": len(researchable) / max(len(total), 1),
            "coverage_ratio_shortlistable": len(shortlistable) / max(len(researchable), 1),
            "coverage_ratio_final_selected": len(final_selected) / max(len(shortlistable), 1),
        },
    )

