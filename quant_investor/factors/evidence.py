"""Offline multi-date factor shadow evidence collection.

This module aggregates read-only shadow scoring and audit artifacts across
local as-of dates. It does not fetch data, call providers, or wire factor
library signals into official selection, posterior scoring, risk, portfolio
construction, orders, or execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.evidence_types import (
    DEFAULT_EVIDENCE_DASHBOARD_FILENAME,
    DEFAULT_EVIDENCE_DATE_RESULTS_FILENAME,
    DEFAULT_EVIDENCE_MARKDOWN_FILENAME,
    DEFAULT_FACTOR_EVIDENCE_DIR,
    DEFAULT_MULTI_DATE_EVIDENCE_REPORTS_FILENAME,
    EVIDENCE_ISSUE_ALIGNMENT_AUDIT_FAIL,
    EVIDENCE_ISSUE_AUDIT_BLOCKER,
    EVIDENCE_ISSUE_EXECUTION_COST_WARN,
    EVIDENCE_ISSUE_INSUFFICIENT_OBSERVATION_DAYS,
    EVIDENCE_ISSUE_LARGE_RANK_DRIFT,
    EVIDENCE_ISSUE_LOW_FACTOR_COVERAGE,
    EVIDENCE_ISSUE_LOW_TOP_N_OVERLAP,
    EVIDENCE_ISSUE_MISSING_CANDIDATES,
    EVIDENCE_ISSUE_MISSING_FACTOR_MATRICES,
    EVIDENCE_ISSUE_MISSING_PRODUCTION_LIBRARY,
    EVIDENCE_ISSUE_TRADABILITY_AUDIT_FAIL,
    EVIDENCE_STATUS_FAIL,
    EVIDENCE_STATUS_INSUFFICIENT_DATA,
    EVIDENCE_STATUS_OK,
    EVIDENCE_STATUS_WARN,
    FactorAuditEvidenceSnapshot,
    FactorEvidenceCollectionConfig,
    FactorEvidenceDateInput,
    FactorShadowEvidenceDateResult,
    MultiDateFactorEvidenceReport,
    NON_RUNTIME_IMPACT_NOTE,
    _candidate_symbols,
    _json_safe,
    _metadata,
    _ordered_unique,
    make_evidence_collection_config_id,
    make_evidence_date_result_id,
    make_multi_date_evidence_report_id,
)
from quant_investor.factors.matrix import FactorMatrix
from quant_investor.factors.schema import (
    FACTOR_STATUS_PRODUCTION,
    ProductionFactorLibrary,
)
from quant_investor.factors.shadow_scoring import (
    ShadowScoringConfig,
    build_shadow_scoring_comparison_report,
)
from quant_investor.versioning import (
    FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION,
    FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION,
)


def load_json_file_safe(path: str | Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None or not str(path).strip():
        return None, ["missing_json_path"]
    resolved = Path(path)
    if not resolved.exists():
        return None, [f"missing_json_file:{resolved}"]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"malformed_json_file:{resolved}:{exc}"]
    if not isinstance(payload, Mapping):
        return None, [f"json_file_not_object:{resolved}"]
    return dict(payload), []


def load_jsonl_file_safe(path: str | Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    if path is None or not str(path).strip():
        return [], ["missing_jsonl_path"]
    resolved = Path(path)
    if not resolved.exists():
        return [], [f"missing_jsonl_file:{resolved}"]
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"malformed_jsonl_file:{resolved}:{exc}"]
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            warnings.append(f"malformed_jsonl_line:{resolved}:{line_number}:{exc.msg}")
            continue
        if not isinstance(payload, Mapping):
            warnings.append(f"jsonl_line_not_object:{resolved}:{line_number}")
            continue
        rows.append(dict(payload))
    return rows, warnings


def load_factor_matrices_from_paths(paths: Sequence[str | Path]) -> tuple[list[FactorMatrix], list[str]]:
    matrices: list[FactorMatrix] = []
    warnings: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            warnings.append(f"missing_factor_matrix_file:{path}")
            continue
        payloads: list[dict[str, Any]]
        if path.suffix.lower() == ".jsonl":
            payloads, row_warnings = load_jsonl_file_safe(path)
            warnings.extend(row_warnings)
        else:
            payload, json_warnings = load_json_file_safe(path)
            warnings.extend(json_warnings)
            if payload is None:
                continue
            if isinstance(payload.get("factor_matrices"), list):
                payloads = [
                    dict(item)
                    for item in payload["factor_matrices"]
                    if isinstance(item, Mapping)
                ]
            else:
                payloads = [payload]
        for index, payload in enumerate(payloads):
            try:
                matrices.append(FactorMatrix.from_dict(payload))
            except (TypeError, ValueError) as exc:
                warnings.append(f"malformed_factor_matrix:{path}:{index}:{exc}")
    return sorted(matrices, key=lambda matrix: matrix.matrix_id), warnings


def load_production_library_safe(path: str | Path | None) -> tuple[ProductionFactorLibrary | None, list[str]]:
    payload, warnings = load_json_file_safe(path)
    if payload is None:
        return None, warnings
    try:
        return ProductionFactorLibrary.from_dict(payload), warnings
    except (TypeError, ValueError) as exc:
        return None, [*warnings, f"malformed_production_library:{path}:{exc}"]


def _payloads_from_paths(paths: Sequence[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in paths:
        resolved = Path(path)
        if resolved.suffix.lower() == ".jsonl":
            rows, row_warnings = load_jsonl_file_safe(resolved)
            payloads.extend(rows)
            warnings.extend(row_warnings)
        else:
            payload, json_warnings = load_json_file_safe(resolved)
            warnings.extend(json_warnings)
            if payload is not None:
                payloads.append(payload)
    return payloads, warnings


def _int_from_payload(payload: Mapping[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
            return len(value)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _verdict_from_payload(payload: Mapping[str, Any]) -> str | None:
    for key in ("verdict", "status", "audit_status", "result"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return None


def _issue_codes_from_payload(payload: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    raw_codes = payload.get("issue_codes")
    if isinstance(raw_codes, Sequence) and not isinstance(raw_codes, (str, bytes, bytearray)):
        codes.extend(str(code).strip() for code in raw_codes if str(code).strip())
    raw_issues = payload.get("issues")
    if isinstance(raw_issues, Sequence) and not isinstance(raw_issues, (str, bytes, bytearray)):
        for issue in raw_issues:
            if isinstance(issue, Mapping):
                code = issue.get("issue_code") or issue.get("code")
                if code is not None and str(code).strip():
                    codes.append(str(code).strip())
    return _ordered_unique(codes)


def build_audit_evidence_snapshot(
    *,
    as_of: str,
    production_library: ProductionFactorLibrary | None,
    library_audit_payload: Mapping[str, Any] | None = None,
    alignment_audit_payloads: Sequence[Mapping[str, Any]] | None = None,
    tradability_audit_payloads: Sequence[Mapping[str, Any]] | None = None,
    execution_cost_payloads: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorAuditEvidenceSnapshot:
    alignment_payloads = list(alignment_audit_payloads or [])
    tradability_payloads = list(tradability_audit_payloads or [])
    execution_payloads = list(execution_cost_payloads or [])
    issue_codes: list[str] = []
    for payload in [library_audit_payload, *alignment_payloads, *tradability_payloads, *execution_payloads]:
        if isinstance(payload, Mapping):
            issue_codes.extend(_issue_codes_from_payload(payload))

    return FactorAuditEvidenceSnapshot(
        as_of=as_of,
        library_exists=production_library is not None,
        production_factor_count=(
            sum(1 for entry in production_library.entries if entry.status == FACTOR_STATUS_PRODUCTION)
            if production_library is not None
            else 0
        ),
        library_audit_verdict=(
            _verdict_from_payload(library_audit_payload)
            if isinstance(library_audit_payload, Mapping)
            else None
        ),
        library_blocker_count=(
            _int_from_payload(
                library_audit_payload,
                ["blocker_count", "library_blocker_count", "blocked_factor_count", "blocked_factor_ids"],
            )
            if isinstance(library_audit_payload, Mapping)
            else 0
        ),
        library_warning_count=(
            _int_from_payload(library_audit_payload, ["warning_count", "warnings", "warning_codes"])
            if isinstance(library_audit_payload, Mapping)
            else 0
        ),
        alignment_audit_verdicts=[
            verdict for verdict in (_verdict_from_payload(payload) for payload in alignment_payloads) if verdict
        ],
        tradability_audit_verdicts=[
            verdict for verdict in (_verdict_from_payload(payload) for payload in tradability_payloads) if verdict
        ],
        execution_cost_verdicts=[
            verdict for verdict in (_verdict_from_payload(payload) for payload in execution_payloads) if verdict
        ],
        audit_issue_codes=issue_codes,
        metadata=_metadata(metadata),
    )


def _status_from_warnings(warnings: set[str], fail: bool, insufficient: bool) -> str:
    if insufficient:
        return EVIDENCE_STATUS_INSUFFICIENT_DATA
    if fail:
        return EVIDENCE_STATUS_FAIL
    if warnings:
        return EVIDENCE_STATUS_WARN
    return EVIDENCE_STATUS_OK


def _rank_delta_metrics(report: Any, threshold: float | None) -> tuple[float | None, int | None, list[str]]:
    deltas = [
        (score.symbol, abs(int(score.rank_delta)))
        for score in report.candidate_scores
        if score.rank_delta is not None
    ]
    if not deltas:
        return None, None, []
    average = sum(delta for _symbol, delta in deltas) / len(deltas)
    max_delta = max(delta for _symbol, delta in deltas)
    if threshold is not None:
        large = [symbol for symbol, delta in deltas if delta > threshold]
    else:
        large = [
            symbol
            for symbol, _delta in sorted(deltas, key=lambda item: (-item[1], item[0]))[:5]
        ]
    return average, max_delta, sorted(large)


def collect_shadow_evidence_for_date(
    *,
    date_input: FactorEvidenceDateInput,
    config: FactorEvidenceCollectionConfig,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorShadowEvidenceDateResult:
    candidates = [dict(candidate) for candidate in date_input.candidates]
    library, library_warnings = load_production_library_safe(date_input.production_library_path)
    matrices, matrix_warnings = load_factor_matrices_from_paths(date_input.factor_matrix_paths)
    library_audit_payload, library_audit_warnings = load_json_file_safe(date_input.library_audit_path)
    alignment_payloads, alignment_warnings = _payloads_from_paths(date_input.alignment_audit_paths)
    tradability_payloads, tradability_warnings = _payloads_from_paths(date_input.tradability_audit_paths)
    execution_payloads, execution_warnings = _payloads_from_paths(date_input.execution_cost_report_paths)

    audit_snapshot = build_audit_evidence_snapshot(
        as_of=date_input.as_of,
        production_library=library,
        library_audit_payload=library_audit_payload,
        alignment_audit_payloads=alignment_payloads,
        tradability_audit_payloads=tradability_payloads,
        execution_cost_payloads=execution_payloads,
        metadata={
            "loader_warnings": sorted(
                library_warnings
                + matrix_warnings
                + library_audit_warnings
                + alignment_warnings
                + tradability_warnings
                + execution_warnings
            )
        },
    )

    shadow_config = ShadowScoringConfig(
        as_of=date_input.as_of,
        top_n=config.top_n,
        min_factor_coverage_ratio=config.min_average_factor_coverage,
        metadata={"evidence_config_id": config.config_id},
    )
    shadow_report = build_shadow_scoring_comparison_report(
        candidates=candidates,
        library=library,
        factor_matrices=matrices,
        audit_report=library_audit_payload,
        config=shadow_config,
        generated_at=generated_at,
        metadata={"evidence_collection": True},
    )

    average_abs_rank_delta, max_abs_rank_delta, large_symbols = _rank_delta_metrics(
        shadow_report,
        config.max_average_abs_rank_delta,
    )
    warning_codes = set(shadow_report.warning_codes)
    fail = False
    insufficient = False

    if library is None:
        warning_codes.add(EVIDENCE_ISSUE_MISSING_PRODUCTION_LIBRARY)
    if not matrices:
        warning_codes.add(EVIDENCE_ISSUE_MISSING_FACTOR_MATRICES)
    if not candidates:
        warning_codes.add(EVIDENCE_ISSUE_MISSING_CANDIDATES)
        insufficient = True
    if (
        shadow_report.average_factor_coverage_ratio is not None
        and shadow_report.average_factor_coverage_ratio < config.min_average_factor_coverage
    ):
        warning_codes.add(EVIDENCE_ISSUE_LOW_FACTOR_COVERAGE)
    if (
        shadow_report.overlap_ratio is not None
        and shadow_report.overlap_ratio < config.min_top_n_overlap_ratio
    ):
        warning_codes.add(EVIDENCE_ISSUE_LOW_TOP_N_OVERLAP)
    if config.max_average_abs_rank_delta is not None and average_abs_rank_delta is not None:
        if average_abs_rank_delta > config.max_average_abs_rank_delta:
            warning_codes.add(EVIDENCE_ISSUE_LARGE_RANK_DRIFT)
    if audit_snapshot.library_blocker_count > 0:
        warning_codes.add(EVIDENCE_ISSUE_AUDIT_BLOCKER)
        fail = config.require_library_audit_no_blocker
    if "fail" in audit_snapshot.alignment_audit_verdicts:
        warning_codes.add(EVIDENCE_ISSUE_ALIGNMENT_AUDIT_FAIL)
        fail = fail or config.require_alignment_audit_pass
    if "fail" in audit_snapshot.tradability_audit_verdicts:
        warning_codes.add(EVIDENCE_ISSUE_TRADABILITY_AUDIT_FAIL)
        fail = fail or config.require_tradability_audit_pass
    if config.require_execution_cost_review and (
        "warn" in audit_snapshot.execution_cost_verdicts
        or "fail" in audit_snapshot.execution_cost_verdicts
    ):
        warning_codes.add(EVIDENCE_ISSUE_EXECUTION_COST_WARN)

    return FactorShadowEvidenceDateResult(
        result_id=make_evidence_date_result_id(
            as_of=date_input.as_of,
            candidate_symbols=_candidate_symbols(candidates),
        ),
        as_of=date_input.as_of,
        candidate_count=shadow_report.candidate_count,
        production_factor_count=shadow_report.production_factor_count,
        used_factor_count=shadow_report.used_factor_count,
        scored_candidate_count=shadow_report.scored_candidate_count,
        average_factor_coverage_ratio=shadow_report.average_factor_coverage_ratio,
        official_top_symbols=shadow_report.official_top_symbols,
        shadow_top_symbols=shadow_report.shadow_top_symbols,
        overlap_top_symbols=shadow_report.overlap_top_symbols,
        top_n_overlap_ratio=shadow_report.overlap_ratio,
        average_abs_rank_delta=average_abs_rank_delta,
        max_abs_rank_delta=max_abs_rank_delta,
        large_rank_delta_symbols=large_symbols,
        shadow_report_id=shadow_report.report_id,
        audit_snapshot=audit_snapshot,
        warning_codes=list(warning_codes),
        status=_status_from_warnings(warning_codes, fail=fail, insufficient=insufficient),
        metadata={
            **_metadata(metadata),
            "factor_shadow_evidence_schema_version": FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION,
            "factor_evidence_dashboard_schema_version": FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION,
            "non_runtime_impact": True,
            "no_official_score_change": True,
            "no_portfolio_change": True,
            "date_input_metadata": dict(_json_safe(date_input.metadata)),
        },
    )


def _average(values: Sequence[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def build_multi_date_factor_evidence_report(
    *,
    date_inputs: Sequence[FactorEvidenceDateInput],
    config: FactorEvidenceCollectionConfig,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> MultiDateFactorEvidenceReport:
    sorted_inputs = sorted(date_inputs, key=lambda item: item.as_of)
    date_results = [
        collect_shadow_evidence_for_date(
            date_input=date_input,
            config=config,
            generated_at=generated_at,
            metadata=metadata,
        )
        for date_input in sorted_inputs
    ]
    observation_days = len(date_results)
    overlap_values = [result.top_n_overlap_ratio for result in date_results]
    coverage_values = [result.average_factor_coverage_ratio for result in date_results]
    rank_values = [result.average_abs_rank_delta for result in date_results]
    max_rank_values = [result.max_abs_rank_delta for result in date_results if result.max_abs_rank_delta is not None]
    warning_codes: set[str] = set()
    for result in date_results:
        warning_codes.update(result.warning_codes)

    audit_blocker_days = sum(1 for result in date_results if result.audit_snapshot.library_blocker_count > 0)
    alignment_fail_days = sum(1 for result in date_results if "fail" in result.audit_snapshot.alignment_audit_verdicts)
    tradability_fail_days = sum(1 for result in date_results if "fail" in result.audit_snapshot.tradability_audit_verdicts)
    execution_cost_warn_days = sum(
        1
        for result in date_results
        if "warn" in result.audit_snapshot.execution_cost_verdicts
        or "fail" in result.audit_snapshot.execution_cost_verdicts
    )

    fail = any(result.status == EVIDENCE_STATUS_FAIL for result in date_results)
    insufficient = observation_days < config.min_observation_days
    if insufficient:
        warning_codes.add(EVIDENCE_ISSUE_INSUFFICIENT_OBSERVATION_DAYS)
    average_overlap = _average(overlap_values)
    average_coverage = _average(coverage_values)
    average_rank_delta = _average(rank_values)
    if average_overlap is not None and average_overlap < config.min_top_n_overlap_ratio:
        warning_codes.add(EVIDENCE_ISSUE_LOW_TOP_N_OVERLAP)
    if average_coverage is not None and average_coverage < config.min_average_factor_coverage:
        warning_codes.add(EVIDENCE_ISSUE_LOW_FACTOR_COVERAGE)
    if config.max_average_abs_rank_delta is not None and average_rank_delta is not None:
        if average_rank_delta > config.max_average_abs_rank_delta:
            warning_codes.add(EVIDENCE_ISSUE_LARGE_RANK_DRIFT)

    report_id = make_multi_date_evidence_report_id(
        generated_at=generated_at,
        as_of_dates=[result.as_of for result in date_results],
    )
    return MultiDateFactorEvidenceReport(
        report_id=report_id,
        generated_at=generated_at,
        config=config,
        observation_days=observation_days,
        start_date=date_results[0].as_of if date_results else None,
        end_date=date_results[-1].as_of if date_results else None,
        average_top_n_overlap_ratio=average_overlap,
        min_top_n_overlap_ratio=(
            min(value for value in overlap_values if value is not None)
            if any(value is not None for value in overlap_values)
            else None
        ),
        average_factor_coverage_ratio=average_coverage,
        average_abs_rank_delta=average_rank_delta,
        max_abs_rank_delta=max(max_rank_values) if max_rank_values else None,
        audit_blocker_days=audit_blocker_days,
        alignment_fail_days=alignment_fail_days,
        tradability_fail_days=tradability_fail_days,
        execution_cost_warn_days=execution_cost_warn_days,
        date_results=date_results,
        warning_codes=list(warning_codes),
        status=_status_from_warnings(warning_codes, fail=fail, insufficient=insufficient),
        metadata={
            **_metadata(metadata),
            "factor_shadow_evidence_schema_version": FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION,
            "factor_evidence_dashboard_schema_version": FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION,
            "non_runtime_impact": True,
            "no_official_score_change": True,
            "no_portfolio_change": True,
        },
    )


def _escape_pipe(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_optional_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def render_multi_date_evidence_markdown(report: MultiDateFactorEvidenceReport) -> str:
    lines = [
        f"# Multi-Date Factor Shadow Evidence: {report.report_id}",
        "",
        f"Generated at: `{_escape_pipe(report.generated_at)}`",
        f"Status: `{_escape_pipe(report.status)}`",
        "",
        "## Observation Window",
        "",
        f"- Start date: `{_escape_pipe(report.start_date or '')}`",
        f"- End date: `{_escape_pipe(report.end_date or '')}`",
        f"- Observation days: `{report.observation_days}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Average top-N overlap ratio | `{_format_optional_float(report.average_top_n_overlap_ratio)}` |",
        f"| Minimum top-N overlap ratio | `{_format_optional_float(report.min_top_n_overlap_ratio)}` |",
        f"| Average factor coverage ratio | `{_format_optional_float(report.average_factor_coverage_ratio)}` |",
        f"| Average absolute rank delta | `{_format_optional_float(report.average_abs_rank_delta)}` |",
        f"| Maximum absolute rank delta | `{report.max_abs_rank_delta if report.max_abs_rank_delta is not None else ''}` |",
        "",
        "## Date-Level Summary",
        "",
        "| As of | Status | Candidates | Used Factors | Top-N Overlap | Coverage | Avg Abs Rank Delta | Warnings |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report.date_results:
        lines.append(
            "| "
            f"`{_escape_pipe(result.as_of)}` | "
            f"`{_escape_pipe(result.status)}` | "
            f"{result.candidate_count} | "
            f"{result.used_factor_count} | "
            f"`{_format_optional_float(result.top_n_overlap_ratio)}` | "
            f"`{_format_optional_float(result.average_factor_coverage_ratio)}` | "
            f"`{_format_optional_float(result.average_abs_rank_delta)}` | "
            f"`{_escape_pipe(', '.join(result.warning_codes))}` |"
        )
    if not report.date_results:
        lines.append("|  |  |  |  |  |  |  | No date results. |")

    lines.extend(
        [
            "",
            "## Audit Blocker/Fail Summary",
            "",
            "| Field | Days |",
            "| --- | ---: |",
            f"| Library audit blocker days | {report.audit_blocker_days} |",
            f"| Alignment audit fail days | {report.alignment_fail_days} |",
            f"| Tradability audit fail days | {report.tradability_fail_days} |",
            f"| Execution cost warn/fail days | {report.execution_cost_warn_days} |",
            "",
            "## Large Rank Drift Summary",
            "",
            "| As of | Symbols |",
            "| --- | --- |",
        ]
    )
    drift_rows = [result for result in report.date_results if result.large_rank_delta_symbols]
    if drift_rows:
        for result in drift_rows:
            lines.append(
                f"| `{_escape_pipe(result.as_of)}` | `{_escape_pipe(', '.join(result.large_rank_delta_symbols))}` |"
            )
    else:
        lines.append("|  | None |")

    lines.extend(["", "## Warnings", ""])
    if report.warning_codes:
        lines.extend([f"- `{_escape_pipe(code)}`" for code in report.warning_codes])
    else:
        lines.append("- None")
    lines.extend(["", "## Non-Runtime Impact", "", NON_RUNTIME_IMPACT_NOTE, ""])
    return "\n".join(lines)


def build_factor_evidence_dashboard_payload(report: MultiDateFactorEvidenceReport) -> dict[str, Any]:
    payload = {
        "schema_version": FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION,
        "status": report.status,
        "generated_at": report.generated_at,
        "observation_days": report.observation_days,
        "start_date": report.start_date,
        "end_date": report.end_date,
        "average_top_n_overlap_ratio": report.average_top_n_overlap_ratio,
        "average_factor_coverage_ratio": report.average_factor_coverage_ratio,
        "average_abs_rank_delta": report.average_abs_rank_delta,
        "audit_blocker_days": report.audit_blocker_days,
        "alignment_fail_days": report.alignment_fail_days,
        "tradability_fail_days": report.tradability_fail_days,
        "execution_cost_warn_days": report.execution_cost_warn_days,
        "warning_codes": list(report.warning_codes),
        "date_summaries": [
            {
                "as_of": result.as_of,
                "status": result.status,
                "candidate_count": result.candidate_count,
                "production_factor_count": result.production_factor_count,
                "used_factor_count": result.used_factor_count,
                "scored_candidate_count": result.scored_candidate_count,
                "top_n_overlap_ratio": result.top_n_overlap_ratio,
                "average_factor_coverage_ratio": result.average_factor_coverage_ratio,
                "average_abs_rank_delta": result.average_abs_rank_delta,
                "max_abs_rank_delta": result.max_abs_rank_delta,
                "warning_codes": list(result.warning_codes),
            }
            for result in report.date_results
        ],
        "metadata": dict(_json_safe(report.metadata)),
    }
    json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return dict(_json_safe(payload))


__all__ = [
    "EVIDENCE_STATUS_OK",
    "EVIDENCE_STATUS_WARN",
    "EVIDENCE_STATUS_FAIL",
    "EVIDENCE_STATUS_INSUFFICIENT_DATA",
    "EVIDENCE_ISSUE_MISSING_PRODUCTION_LIBRARY",
    "EVIDENCE_ISSUE_MISSING_FACTOR_MATRICES",
    "EVIDENCE_ISSUE_MISSING_CANDIDATES",
    "EVIDENCE_ISSUE_LOW_FACTOR_COVERAGE",
    "EVIDENCE_ISSUE_LOW_TOP_N_OVERLAP",
    "EVIDENCE_ISSUE_LARGE_RANK_DRIFT",
    "EVIDENCE_ISSUE_AUDIT_BLOCKER",
    "EVIDENCE_ISSUE_ALIGNMENT_AUDIT_FAIL",
    "EVIDENCE_ISSUE_TRADABILITY_AUDIT_FAIL",
    "EVIDENCE_ISSUE_EXECUTION_COST_WARN",
    "EVIDENCE_ISSUE_INSUFFICIENT_OBSERVATION_DAYS",
    "DEFAULT_FACTOR_EVIDENCE_DIR",
    "DEFAULT_EVIDENCE_DATE_RESULTS_FILENAME",
    "DEFAULT_MULTI_DATE_EVIDENCE_REPORTS_FILENAME",
    "DEFAULT_EVIDENCE_DASHBOARD_FILENAME",
    "DEFAULT_EVIDENCE_MARKDOWN_FILENAME",
    "FactorEvidenceCollectionConfig",
    "FactorEvidenceDateInput",
    "FactorAuditEvidenceSnapshot",
    "FactorShadowEvidenceDateResult",
    "MultiDateFactorEvidenceReport",
    "make_evidence_collection_config_id",
    "make_evidence_date_result_id",
    "make_multi_date_evidence_report_id",
    "load_json_file_safe",
    "load_jsonl_file_safe",
    "load_factor_matrices_from_paths",
    "load_production_library_safe",
    "build_audit_evidence_snapshot",
    "collect_shadow_evidence_for_date",
    "build_multi_date_factor_evidence_report",
    "render_multi_date_evidence_markdown",
    "build_factor_evidence_dashboard_payload",
]
