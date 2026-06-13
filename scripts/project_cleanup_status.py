"""Summarize project cleanup evidence across audit reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import (
    build_cleanup_inventory,
    build_code_retirement_reference_audit,
    get_repo_root,
)

SCHEMA_VERSION = "myquant.project_cleanup_status.v1"
DEFAULT_MAX_SOURCES = 20
DEFAULT_LARGE_MODULE_LINE_THRESHOLD = 1000
DEFAULT_MAX_LARGE_MODULES = 10

REPORT_PATTERNS = {
    "latest_cleanup_inventory_json": (
        "reports/project_cleanup/cleanup_inventory_*/cleanup_inventory.json"
    ),
    "code_retirement_reference_audit_json": (
        "reports/project_cleanup/"
        "code_retirement_reference_audit_*/code_retirement_reference_audit.json"
    ),
    "data_duplicate_audit_json": (
        "reports/project_cleanup/data_duplicate_audit_*/data_duplicate_audit.json"
    ),
    "data_cleanup_plan_json": (
        "reports/project_cleanup/data_cleanup_plan_*/data_cleanup_plan.json"
    ),
    "data_cleanup_restore_policy_json": (
        "reports/project_cleanup/"
        "data_cleanup_restore_policy_*/data_cleanup_restore_policy.json"
    ),
    "data_cleanup_restore_reference_audit_json": (
        "reports/project_cleanup/"
        "data_cleanup_restore_reference_audit_*/"
        "data_cleanup_restore_reference_audit.json"
    ),
    "data_cleanup_restore_readiness_json": (
        "reports/project_cleanup/"
        "data_cleanup_restore_readiness_*/data_cleanup_restore_readiness.json"
    ),
    "data_cleanup_restore_readback_json": (
        "reports/project_cleanup/"
        "data_cleanup_restore_readback_*/data_cleanup_restore_readback.json"
    ),
    "data_cleanup_gate_json": (
        "reports/project_cleanup/data_cleanup_gate_*/data_cleanup_gate.json"
    ),
    "data_cleanup_readback_json": (
        "reports/project_cleanup/data_cleanup_readback_*/data_cleanup_readback.json"
    ),
    "data_cleanup_whitelist_json": (
        "reports/project_cleanup/data_cleanup_whitelist_*/data_cleanup_whitelist.json"
    ),
    "data_cleanup_execute_json": (
        "reports/project_cleanup/data_cleanup_execute_*/data_cleanup_execute.json"
    ),
    "data_cleanup_reference_rewrite_json": (
        "reports/project_cleanup/"
        "data_cleanup_reference_rewrite_*/data_cleanup_reference_rewrite.json"
    ),
    "empty_cell_flags_compaction_json": (
        "reports/project_cleanup/"
        "empty_cell_flags_compaction_*/empty_cell_flags_compaction.json"
    ),
    "issue_cell_flags_compaction_json": (
        "reports/project_cleanup/"
        "issue_cell_flags_compaction_*/issue_cell_flags_compaction.json"
    ),
    "uniform_row_flags_compaction_json": (
        "reports/project_cleanup/"
        "uniform_row_flags_compaction_*/uniform_row_flags_compaction.json"
    ),
    "matrix_coverage_compaction_json": (
        "reports/project_cleanup/"
        "matrix_coverage_compaction_*/matrix_coverage_compaction.json"
    ),
    "architecture_performance_audit_json": (
        "reports/project_cleanup/architecture_rebaseline_*/architecture_performance_audit.json",
        "reports/project_cleanup/cleanup_baseline_*/architecture_performance_audit.json",
    ),
    "latest_runtime_profile_json": (
        "results/cn_analysis_full/CN_Runtime_Profile_*.json"
    ),
}


def _generated_at_timestamp(path: Path) -> float:
    payload = _load_json(path)
    if not payload:
        return 0.0
    value = payload.get("generated_at")
    if not isinstance(value, str) or not value.strip():
        return 0.0
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _latest_path(repo_root: Path, pattern: str | Sequence[str]) -> Path | None:
    patterns = (pattern,) if isinstance(pattern, str) else tuple(pattern)
    candidates = sorted(
        candidate
        for item in patterns
        for candidate in repo_root.glob(item)
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (_generated_at_timestamp(path), path.as_posix()),
    )


def _all_report_paths(repo_root: Path, pattern: str | Sequence[str]) -> list[Path]:
    patterns = (pattern,) if isinstance(pattern, str) else tuple(pattern)
    candidates = sorted(
        candidate
        for item in patterns
        for candidate in repo_root.glob(item)
    )
    return sorted(
        candidates,
        key=lambda path: (_generated_at_timestamp(path), path.as_posix()),
    )


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _report_sources(repo_root: Path) -> dict[str, str | None]:
    sources: dict[str, str | None] = {}
    for name, pattern in REPORT_PATTERNS.items():
        path = _latest_path(repo_root, pattern)
        sources[name] = str(path) if path is not None else None
    return sources


def _load_sources(sources: dict[str, str | None]) -> dict[str, dict[str, Any] | None]:
    loaded: dict[str, dict[str, Any] | None] = {}
    for name, path_text in sources.items():
        loaded[name] = _load_json(Path(path_text)) if path_text else None
    return loaded


def _summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    value = report.get("summary")
    return value if isinstance(value, dict) else {}


def _count_delete_allowed(
    inventory: dict[str, Any],
    classification: str,
) -> int:
    return sum(
        1
        for item in inventory.get("items", [])
        if item.get("classification") == classification
        and bool(item.get("delete_allowed"))
    )


def _existing_code_candidate_count(code_audit: dict[str, Any]) -> int:
    return sum(
        1
        for candidate in code_audit.get("candidates", [])
        if bool(candidate.get("exists"))
    )


def _code_status(code_audit: dict[str, Any]) -> str:
    if int(code_audit.get("production_reference_count", 0)) > 0:
        return "attention_required"
    if _existing_code_candidate_count(code_audit) > 0:
        return "removal_candidates_present"
    return "clear"


def _duplicate_storage_status(
    duplicate_audit: dict[str, Any] | None,
    cleanup_plan: dict[str, Any] | None,
    cleanup_whitelist: dict[str, Any] | None,
    cleanup_execution_summary: dict[str, Any],
) -> str:
    if not duplicate_audit:
        return "not_audited"
    audit_summary = _summary(duplicate_audit)
    plan_summary = _summary(cleanup_plan)
    if (
        int(audit_summary.get("duplicate_group_count", 0) or 0) == 0
        and int(plan_summary.get("candidate_group_count", 0) or 0) == 0
    ):
        return "clear"
    if int(cleanup_execution_summary.get("deleted_count", 0) or 0) > 0:
        return "partial_cleanup_executed"
    if int((cleanup_whitelist or {}).get("execute_allowed_count", 0) or 0) > 0:
        return "approved_pending_execute"
    if int(plan_summary.get("candidate_group_count", 0) or 0) > 0:
        return "review_only"
    if int(audit_summary.get("duplicate_group_count", 0) or 0) > 0:
        return "review_only"
    return "clear"


def _unnecessary_data_status(inventory: dict[str, Any]) -> str:
    if int(inventory.get("delete_candidate_count", 0) or 0) > 0:
        return "delete_candidates_present"
    return "clear"


def _structure_status(
    architecture_audit: dict[str, Any] | None,
    current_large_modules: list[dict[str, Any]],
    remediation: dict[str, Any] | None = None,
) -> str:
    if current_large_modules:
        return "review_required"
    if not architecture_audit:
        return "not_audited"
    if remediation and int(remediation.get("unresolved_finding_count", 0) or 0) > 0:
        return "review_required"
    if remediation and (
        int(remediation.get("resolved_finding_count", 0) or 0) > 0
        or int(remediation.get("resolved_large_module_candidate_count", 0) or 0) > 0
    ):
        return "remediated_pending_rebaseline"
    if architecture_audit.get("primary_findings") or architecture_audit.get(
        "large_module_candidates"
    ):
        return "review_required"
    return "clear"


def _slowest_stages(profile: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    stages = []
    for stage in profile.get("stages", []):
        stages.append(
            {
                "name": stage.get("name", ""),
                "seconds": float(stage.get("seconds", 0.0) or 0.0),
            }
        )
    return sorted(stages, key=lambda item: item["seconds"], reverse=True)[:limit]


def _source_line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())


def _current_large_modules(
    repo_root: Path,
    *,
    line_threshold: int = DEFAULT_LARGE_MODULE_LINE_THRESHOLD,
    limit: int = DEFAULT_MAX_LARGE_MODULES,
) -> list[dict[str, Any]]:
    source_root = repo_root / "quant_investor"
    if not source_root.exists():
        return []
    modules: list[dict[str, Any]] = []
    for path in source_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        lines = _source_line_count(path)
        if lines < line_threshold:
            continue
        modules.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "lines": lines,
            }
        )
    return sorted(
        modules,
        key=lambda item: (-int(item["lines"]), str(item["path"])),
    )[:limit]


def _stage_by_name(profile: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not profile:
        return {}
    return {
        str(stage.get("name", "")): stage
        for stage in profile.get("stages", [])
        if isinstance(stage, dict) and str(stage.get("name", "")).strip()
    }


def _batch_read_profile_proves_batch_path(profile: dict[str, Any] | None) -> bool:
    stage = _stage_by_name(profile).get("dag_batch_read") or {}
    metadata = stage.get("metadata") or {}
    return (
        int(metadata.get("batch_result_count", 0) or 0) > 0
        and int(metadata.get("per_symbol_fallback_count", 0) or 0) == 0
        and int(metadata.get("projected_column_count", 0) or 0) > 0
        and bool(str(metadata.get("runtime_lookback_start_date", "")).strip())
    )


def _runtime_profile_proves_stage_profile(profile: dict[str, Any] | None) -> bool:
    stages = set(_stage_by_name(profile))
    required = {
        "dag_symbol_list",
        "dag_batch_read",
        "dag_funnel",
        "dag_candidate_research",
        "dag_bayesian_selection",
        "dag_control_chain",
        "dag_reporting_artifacts",
        "analysis_report_persistence",
    }
    if not required.issubset(stages):
        return False
    return _batch_read_profile_proves_batch_path(profile)


def _file_contains(path: Path, patterns: set[str]) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return all(pattern in text for pattern in patterns)


def _reader_cache_evidence(repo_root: Path) -> bool:
    return _file_contains(
        repo_root / "quant_investor" / "market" / "market_data_reader.py",
        {
            "_latest_payload",
            "_snapshot_gate_cache",
            "_serving_symbols_cache",
            "_components_payload",
        },
    )


def _market_report_split_evidence(repo_root: Path) -> bool:
    required_paths = [
        "quant_investor/market/runtime_profile.py",
        "quant_investor/market/report_persistence.py",
        "quant_investor/market/full_report_helpers.py",
        "quant_investor/market/full_report_sections.py",
    ]
    return all((repo_root / path).exists() for path in required_paths)


def _strategy_profile_split_evidence(repo_root: Path) -> bool:
    required_paths = [
        "quant_investor/market/runtime_profile.py",
        "quant_investor/monitoring/cn_aggressive_review_layer.py",
        "quant_investor/monitoring/cn_aggressive_review_runtime.py",
        "quant_investor/monitoring/cn_aggressive_rebalance.py",
        "quant_investor/monitoring/cn_aggressive_report_renderer.py",
    ]
    return all((repo_root / path).exists() for path in required_paths)


def _current_path_under_threshold(
    repo_root: Path,
    relative_path: str,
    *,
    threshold: int = DEFAULT_LARGE_MODULE_LINE_THRESHOLD,
) -> bool:
    path = repo_root / relative_path
    return path.exists() and _source_line_count(path) < threshold


def _finding_resolution_evidence(
    finding_id: str,
    repo_root: Path,
    runtime_profile: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    if finding_id == "perf-001":
        if _batch_read_profile_proves_batch_path(runtime_profile):
            return (
                "resolved_current_evidence",
                ["latest runtime profile has dag_batch_read batch_result_count>0, projected columns, lookback, and fallback=0"],
            )
        return ("pending_reaudit", ["missing batch-read runtime profile proof"])
    if finding_id == "perf-002":
        if _runtime_profile_proves_stage_profile(runtime_profile):
            return (
                "resolved_current_evidence",
                ["latest runtime profile contains DAG, control-chain, reporting, and persistence stages"],
            )
        return ("pending_reaudit", ["missing complete stage profile proof"])
    if finding_id == "perf-003":
        if _reader_cache_evidence(repo_root) and _batch_read_profile_proves_batch_path(
            runtime_profile
        ):
            return (
                "resolved_current_evidence",
                ["MarketDataReader cache fields exist and runtime profile proves batch path fallback=0"],
            )
        return ("pending_reaudit", ["missing reader-cache or batch-path proof"])
    if finding_id == "arch-001":
        if _market_report_split_evidence(repo_root) and _current_path_under_threshold(
            repo_root,
            "quant_investor/market/analyze.py",
        ):
            return (
                "resolved_current_evidence",
                ["market runtime profile, report persistence, and full-report split modules exist; analyze.py is below threshold"],
            )
        return ("pending_reaudit", ["missing market report split proof"])
    if finding_id == "arch-002":
        if _strategy_profile_split_evidence(repo_root) and _current_path_under_threshold(
            repo_root,
            "quant_investor/monitoring/cn_aggressive_portfolio_tracker.py",
        ):
            return (
                "resolved_current_evidence",
                ["generic market runtime profiler exists and CN tracker split modules exist below threshold"],
            )
        return ("pending_reaudit", ["missing strategy profiler split proof"])
    return ("pending_reaudit", [f"no remediation rule for {finding_id or 'unknown'}"])


def _architecture_remediation_summary(
    repo_root: Path,
    architecture_audit: dict[str, Any] | None,
    runtime_profile: dict[str, Any] | None,
    current_large_modules: list[dict[str, Any]],
) -> dict[str, Any]:
    findings = (
        architecture_audit.get("primary_findings", [])
        if architecture_audit
        else []
    )
    remediation_items: list[dict[str, Any]] = []
    for finding in findings:
        finding_id = str(finding.get("id", "")).strip()
        status, evidence = _finding_resolution_evidence(
            finding_id,
            repo_root,
            runtime_profile,
        )
        remediation_items.append(
            {
                "id": finding_id,
                "status": status,
                "evidence": evidence,
            }
        )

    baseline_candidates = (
        architecture_audit.get("large_module_candidates", [])
        if architecture_audit
        else []
    )
    current_large_paths = {
        str(item.get("path", ""))
        for item in current_large_modules
        if str(item.get("path", "")).strip()
    }
    resolved_candidates = [
        str(candidate.get("path", ""))
        for candidate in baseline_candidates
        if str(candidate.get("path", "")).strip()
        and str(candidate.get("path", "")) not in current_large_paths
        and _current_path_under_threshold(
            repo_root,
            str(candidate.get("path", "")),
        )
    ]
    unresolved_count = sum(
        1
        for item in remediation_items
        if item.get("status") != "resolved_current_evidence"
    )
    return {
        "items": remediation_items,
        "resolved_finding_count": len(remediation_items) - unresolved_count,
        "unresolved_finding_count": unresolved_count,
        "resolved_large_module_candidate_count": len(resolved_candidates),
        "resolved_large_module_candidates": resolved_candidates[:10],
    }


def _runtime_profile_summary(
    profile: dict[str, Any] | None,
    source_path: str | None,
) -> dict[str, Any] | None:
    if not profile:
        return None
    fallback_count = 0
    for stage in profile.get("stages", []):
        metadata = stage.get("metadata") or {}
        fallback_count += int(metadata.get("per_symbol_fallback_count", 0) or 0)
    return {
        "source_json": source_path,
        "market": profile.get("market"),
        "universe": profile.get("universe"),
        "total_seconds": profile.get("total_seconds"),
        "stage_count": len(profile.get("stages", [])),
        "per_symbol_fallback_count": fallback_count,
        "slowest_stages": _slowest_stages(profile),
    }


def _cleanup_execution_summary(repo_root: Path) -> dict[str, Any]:
    paths = _all_report_paths(repo_root, REPORT_PATTERNS["data_cleanup_execute_json"])
    executed_report_count = 0
    deleted_count = 0
    deleted_reclaim_bytes = 0
    deleted_group_ids: set[str] = set()
    latest_summary: dict[str, Any] = {}
    latest_execution_performed = False
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        summary = _summary(payload)
        latest_summary = summary
        latest_execution_performed = bool(payload.get("execution_performed", False))
        if not (
            payload.get("apply_requested") is True
            and payload.get("confirm_token_valid") is True
            and payload.get("execution_performed") is True
        ):
            continue
        executed_report_count += 1
        deleted_count += int(summary.get("deleted_count", 0) or 0)
        deleted_reclaim_bytes += int(summary.get("deleted_reclaim_bytes", 0) or 0)
        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            if item.get("status") != "deleted":
                continue
            group_id = str(item.get("group_id", "")).strip()
            if group_id:
                deleted_group_ids.add(group_id)
    return {
        "execution_report_count": len(paths),
        "executed_report_count": executed_report_count,
        "latest_deleted_count": int(latest_summary.get("deleted_count", 0) or 0),
        "latest_deleted_reclaim_bytes": int(
            latest_summary.get("deleted_reclaim_bytes", 0) or 0
        ),
        "latest_execution_performed": latest_execution_performed,
        "deleted_count": deleted_count,
        "deleted_reclaim_bytes": deleted_reclaim_bytes,
        "deleted_group_count": len(deleted_group_ids),
        "deleted_group_ids": sorted(deleted_group_ids),
    }


def _empty_cell_flags_compaction_summary(repo_root: Path) -> dict[str, Any]:
    paths = _all_report_paths(
        repo_root,
        REPORT_PATTERNS["empty_cell_flags_compaction_json"],
    )
    executed_report_count = 0
    compacted_count = 0
    compacted_reclaim_bytes = 0
    orphan_deleted_count = 0
    orphan_deleted_reclaim_bytes = 0
    references_rewritten_count = 0
    latest_summary: dict[str, Any] = {}
    latest_execution_performed = False
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        summary = _summary(payload)
        latest_summary = summary
        latest_execution_performed = bool(payload.get("execution_performed", False))
        if not (
            payload.get("apply_requested") is True
            and payload.get("confirm_token_valid") is True
            and payload.get("execution_performed") is True
        ):
            continue
        executed_report_count += 1
        compacted_count += int(summary.get("compacted_count", 0) or 0)
        compacted_reclaim_bytes += int(
            summary.get("compacted_reclaim_bytes", 0) or 0
        )
        orphan_deleted_count += int(summary.get("orphan_deleted_count", 0) or 0)
        orphan_deleted_reclaim_bytes += int(
            summary.get("orphan_deleted_reclaim_bytes", 0) or 0
        )
        references_rewritten_count += int(
            summary.get("references_rewritten_count", 0) or 0
        )
    return {
        "report_count": len(paths),
        "executed_report_count": executed_report_count,
        "latest_candidate_count": int(latest_summary.get("candidate_count", 0) or 0),
        "latest_would_compact_count": int(
            latest_summary.get("would_compact_count", 0) or 0
        ),
        "latest_compacted_count": int(
            latest_summary.get("compacted_count", 0) or 0
        ),
        "latest_compacted_reclaim_bytes": int(
            latest_summary.get("compacted_reclaim_bytes", 0) or 0
        ),
        "latest_orphan_deleted_count": int(
            latest_summary.get("orphan_deleted_count", 0) or 0
        ),
        "latest_orphan_deleted_reclaim_bytes": int(
            latest_summary.get("orphan_deleted_reclaim_bytes", 0) or 0
        ),
        "latest_blocked_count": int(latest_summary.get("blocked_count", 0) or 0),
        "latest_execution_performed": latest_execution_performed,
        "compacted_count": compacted_count,
        "compacted_reclaim_bytes": compacted_reclaim_bytes,
        "orphan_deleted_count": orphan_deleted_count,
        "orphan_deleted_reclaim_bytes": orphan_deleted_reclaim_bytes,
        "references_rewritten_count": references_rewritten_count,
    }


def _uniform_row_flags_compaction_summary(repo_root: Path) -> dict[str, Any]:
    paths = _all_report_paths(
        repo_root,
        REPORT_PATTERNS["uniform_row_flags_compaction_json"],
    )
    executed_report_count = 0
    compacted_count = 0
    compacted_reclaim_bytes = 0
    references_rewritten_count = 0
    latest_summary: dict[str, Any] = {}
    latest_execution_performed = False
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        summary = _summary(payload)
        latest_summary = summary
        latest_execution_performed = bool(payload.get("execution_performed", False))
        if not (
            payload.get("apply_requested") is True
            and payload.get("confirm_token_valid") is True
            and payload.get("execution_performed") is True
        ):
            continue
        executed_report_count += 1
        compacted_count += int(summary.get("compacted_count", 0) or 0)
        compacted_reclaim_bytes += int(
            summary.get("compacted_reclaim_bytes", 0) or 0
        )
        references_rewritten_count += int(
            summary.get("references_rewritten_count", 0) or 0
        )
    return {
        "report_count": len(paths),
        "executed_report_count": executed_report_count,
        "latest_candidate_count": int(latest_summary.get("candidate_count", 0) or 0),
        "latest_would_compact_count": int(
            latest_summary.get("would_compact_count", 0) or 0
        ),
        "latest_compacted_count": int(
            latest_summary.get("compacted_count", 0) or 0
        ),
        "latest_compacted_reclaim_bytes": int(
            latest_summary.get("compacted_reclaim_bytes", 0) or 0
        ),
        "latest_blocked_count": int(latest_summary.get("blocked_count", 0) or 0),
        "latest_execution_performed": latest_execution_performed,
        "compacted_count": compacted_count,
        "compacted_reclaim_bytes": compacted_reclaim_bytes,
        "references_rewritten_count": references_rewritten_count,
    }


def _issue_cell_flags_compaction_summary(repo_root: Path) -> dict[str, Any]:
    paths = _all_report_paths(
        repo_root,
        REPORT_PATTERNS["issue_cell_flags_compaction_json"],
    )
    executed_report_count = 0
    compacted_count = 0
    compacted_reclaim_bytes = 0
    references_rewritten_count = 0
    issue_row_count = 0
    latest_summary: dict[str, Any] = {}
    latest_execution_performed = False
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        summary = _summary(payload)
        latest_summary = summary
        latest_execution_performed = bool(payload.get("execution_performed", False))
        if not (
            payload.get("apply_requested") is True
            and payload.get("confirm_token_valid") is True
            and payload.get("execution_performed") is True
        ):
            continue
        executed_report_count += 1
        compacted_count += int(summary.get("compacted_count", 0) or 0)
        compacted_reclaim_bytes += int(
            summary.get("compacted_reclaim_bytes", 0) or 0
        )
        references_rewritten_count += int(
            summary.get("references_rewritten_count", 0) or 0
        )
        issue_row_count += int(summary.get("issue_row_count", 0) or 0)
    return {
        "report_count": len(paths),
        "executed_report_count": executed_report_count,
        "latest_candidate_count": int(latest_summary.get("candidate_count", 0) or 0),
        "latest_would_compact_count": int(
            latest_summary.get("would_compact_count", 0) or 0
        ),
        "latest_compacted_count": int(
            latest_summary.get("compacted_count", 0) or 0
        ),
        "latest_compacted_reclaim_bytes": int(
            latest_summary.get("compacted_reclaim_bytes", 0) or 0
        ),
        "latest_blocked_count": int(latest_summary.get("blocked_count", 0) or 0),
        "latest_execution_performed": latest_execution_performed,
        "compacted_count": compacted_count,
        "compacted_reclaim_bytes": compacted_reclaim_bytes,
        "references_rewritten_count": references_rewritten_count,
        "issue_row_count": issue_row_count,
    }


def _matrix_coverage_compaction_summary(repo_root: Path) -> dict[str, Any]:
    paths = _all_report_paths(
        repo_root,
        REPORT_PATTERNS["matrix_coverage_compaction_json"],
    )
    executed_report_count = 0
    compacted_count = 0
    compacted_reclaim_bytes = 0
    references_rewritten_count = 0
    latest_summary: dict[str, Any] = {}
    latest_execution_performed = False
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        summary = _summary(payload)
        latest_summary = summary
        latest_execution_performed = bool(payload.get("execution_performed", False))
        if not (
            payload.get("apply_requested") is True
            and payload.get("confirm_token_valid") is True
            and payload.get("execution_performed") is True
        ):
            continue
        executed_report_count += 1
        compacted_count += int(summary.get("compacted_count", 0) or 0)
        compacted_reclaim_bytes += int(
            summary.get("compacted_reclaim_bytes", 0) or 0
        )
        references_rewritten_count += int(
            summary.get("references_rewritten_count", 0) or 0
        )
    return {
        "report_count": len(paths),
        "executed_report_count": executed_report_count,
        "latest_candidate_count": int(latest_summary.get("candidate_count", 0) or 0),
        "latest_would_compact_count": int(
            latest_summary.get("would_compact_count", 0) or 0
        ),
        "latest_compacted_count": int(
            latest_summary.get("compacted_count", 0) or 0
        ),
        "latest_compacted_reclaim_bytes": int(
            latest_summary.get("compacted_reclaim_bytes", 0) or 0
        ),
        "latest_blocked_count": int(latest_summary.get("blocked_count", 0) or 0),
        "latest_execution_performed": latest_execution_performed,
        "compacted_count": compacted_count,
        "compacted_reclaim_bytes": compacted_reclaim_bytes,
        "references_rewritten_count": references_rewritten_count,
    }


def _reference_rewrite_summary(repo_root: Path) -> dict[str, Any]:
    paths = _all_report_paths(
        repo_root,
        REPORT_PATTERNS["data_cleanup_reference_rewrite_json"],
    )
    executed_report_count = 0
    rewritten_deleted_count = 0
    rewritten_deleted_reclaim_bytes = 0
    references_rewritten_count = 0
    latest_summary: dict[str, Any] = {}
    latest_execution_performed = False
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        summary = _summary(payload)
        latest_summary = summary
        latest_execution_performed = bool(payload.get("execution_performed", False))
        rewritten_deleted = int(summary.get("rewritten_deleted_count", 0) or 0)
        if not (
            payload.get("apply_requested") is True
            and payload.get("execution_performed") is True
            and rewritten_deleted > 0
        ):
            continue
        executed_report_count += 1
        rewritten_deleted_count += rewritten_deleted
        rewritten_deleted_reclaim_bytes += int(
            summary.get("rewritten_deleted_reclaim_bytes", 0) or 0
        )
        references_rewritten_count += int(
            summary.get("references_rewritten_count", 0) or 0
        )
    return {
        "report_count": len(paths),
        "executed_report_count": executed_report_count,
        "latest_selected_group_count": int(
            latest_summary.get("selected_group_count", 0) or 0
        ),
        "latest_would_rewrite_delete_count": int(
            latest_summary.get("would_rewrite_delete_count", 0) or 0
        ),
        "latest_rewritten_deleted_count": int(
            latest_summary.get("rewritten_deleted_count", 0) or 0
        ),
        "latest_blocked_count": int(latest_summary.get("blocked_count", 0) or 0),
        "latest_execution_performed": latest_execution_performed,
        "rewritten_deleted_count": rewritten_deleted_count,
        "rewritten_deleted_reclaim_bytes": rewritten_deleted_reclaim_bytes,
        "references_rewritten_count": references_rewritten_count,
    }


def _effective_whitelist_summary(
    cleanup_whitelist: dict[str, Any] | None,
    whitelist_summary: dict[str, Any],
    cleanup_execution_summary: dict[str, Any],
) -> dict[str, Any]:
    if not cleanup_whitelist:
        return {
            "whitelist_item_count": 0,
            "pending_manual_approval_count": 0,
            "approved_for_delete_count": 0,
            "whitelist_execute_allowed_count": 0,
            "whitelist_potential_reclaim_bytes": 0,
        }

    items = [
        item
        for item in cleanup_whitelist.get("items", [])
        if isinstance(item, dict)
    ]
    if not items:
        return {
            "whitelist_item_count": whitelist_summary.get("whitelist_item_count", 0),
            "pending_manual_approval_count": whitelist_summary.get(
                "manual_approval_required_count",
                0,
            ),
            "approved_for_delete_count": whitelist_summary.get(
                "approved_for_delete_count",
                0,
            ),
            "whitelist_execute_allowed_count": cleanup_whitelist.get(
                "execute_allowed_count",
                0,
            ),
            "whitelist_potential_reclaim_bytes": whitelist_summary.get(
                "potential_reclaim_bytes",
                0,
            ),
        }

    deleted_group_ids = {
        str(group_id)
        for group_id in cleanup_execution_summary.get("deleted_group_ids", [])
    }
    active_items = [
        item
        for item in items
        if str(item.get("group_id", "")).strip() not in deleted_group_ids
    ]
    pending_count = sum(
        1
        for item in active_items
        if item.get("approval_status") == "pending_manual_approval"
    )
    approved_count = sum(
        1
        for item in active_items
        if item.get("approval_status") == "approved_for_delete"
    )
    execute_allowed_count = sum(
        1
        for item in active_items
        if item.get("execute_allowed") is True
    )
    return {
        "whitelist_item_count": len(items),
        "pending_manual_approval_count": pending_count,
        "approved_for_delete_count": approved_count,
        "whitelist_execute_allowed_count": execute_allowed_count,
        "whitelist_potential_reclaim_bytes": whitelist_summary.get(
            "potential_reclaim_bytes",
            0,
        ),
    }


def _objective(
    *,
    status: str,
    evidence: dict[str, Any],
    next_action: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    }


def build_project_cleanup_status(root: Path | None = None) -> dict[str, Any]:
    """Build a read-only summary over current cleanup evidence."""
    repo_root = get_repo_root(root)
    inventory = build_cleanup_inventory(repo_root)
    sources = _report_sources(repo_root)
    reports = _load_sources(sources)

    code_audit = reports["code_retirement_reference_audit_json"]
    if code_audit is None:
        code_audit = build_code_retirement_reference_audit(repo_root)

    duplicate_audit = reports["data_duplicate_audit_json"]
    cleanup_plan = reports["data_cleanup_plan_json"]
    restore_policy = reports["data_cleanup_restore_policy_json"]
    restore_reference_audit = reports["data_cleanup_restore_reference_audit_json"]
    restore_readiness = reports["data_cleanup_restore_readiness_json"]
    restore_readback = reports["data_cleanup_restore_readback_json"]
    cleanup_gate = reports["data_cleanup_gate_json"]
    cleanup_readback = reports["data_cleanup_readback_json"]
    cleanup_whitelist = reports["data_cleanup_whitelist_json"]
    cleanup_execute = reports["data_cleanup_execute_json"]
    reference_rewrite = reports["data_cleanup_reference_rewrite_json"]
    empty_cell_flags_compaction = reports["empty_cell_flags_compaction_json"]
    issue_cell_flags_compaction = reports["issue_cell_flags_compaction_json"]
    uniform_row_flags_compaction = reports["uniform_row_flags_compaction_json"]
    matrix_coverage_compaction = reports["matrix_coverage_compaction_json"]
    architecture_audit = reports["architecture_performance_audit_json"]
    runtime_profile = reports["latest_runtime_profile_json"]

    inventory_summary = _summary(inventory)
    plan_summary = _summary(cleanup_plan)
    restore_policy_summary = _summary(restore_policy)
    restore_reference_summary = _summary(restore_reference_audit)
    restore_readiness_summary = _summary(restore_readiness)
    restore_readback_summary = _summary(restore_readback)
    duplicate_summary = _summary(duplicate_audit)
    gate_summary = _summary(cleanup_gate)
    readback_summary = _summary(cleanup_readback)
    whitelist_summary = _summary(cleanup_whitelist)
    execute_summary = _summary(cleanup_execute)
    reference_rewrite_summary = _summary(reference_rewrite)
    empty_cell_flags_summary = _summary(empty_cell_flags_compaction)
    issue_cell_flags_summary = _summary(issue_cell_flags_compaction)
    uniform_row_flags_summary = _summary(uniform_row_flags_compaction)
    matrix_coverage_summary = _summary(matrix_coverage_compaction)
    cumulative_execute_summary = _cleanup_execution_summary(repo_root)
    cumulative_reference_rewrite_summary = _reference_rewrite_summary(repo_root)
    cumulative_empty_cell_flags_summary = _empty_cell_flags_compaction_summary(
        repo_root
    )
    cumulative_issue_cell_flags_summary = _issue_cell_flags_compaction_summary(
        repo_root
    )
    cumulative_uniform_row_flags_summary = _uniform_row_flags_compaction_summary(
        repo_root
    )
    cumulative_matrix_coverage_summary = _matrix_coverage_compaction_summary(
        repo_root
    )
    effective_whitelist_summary = _effective_whitelist_summary(
        cleanup_whitelist,
        whitelist_summary,
        cumulative_execute_summary,
    )

    active_runtime_delete_count = _count_delete_allowed(
        inventory,
        "active_runtime_source",
    )
    strategy_delete_count = _count_delete_allowed(inventory, "strategy_evidence")
    existing_code_count = _existing_code_candidate_count(code_audit)
    architecture_findings = (
        architecture_audit.get("primary_findings", [])
        if architecture_audit
        else []
    )
    large_modules = (
        architecture_audit.get("large_module_candidates", [])
        if architecture_audit
        else []
    )
    current_large_modules = _current_large_modules(repo_root)
    architecture_remediation = _architecture_remediation_summary(
        repo_root,
        architecture_audit,
        runtime_profile,
        current_large_modules,
    )

    objectives = {
        "redundant_code": _objective(
            status=_code_status(code_audit),
            evidence={
                "retired_candidate_count": code_audit.get("candidate_count", 0),
                "existing_candidate_count": existing_code_count,
                "production_reference_count": code_audit.get(
                    "production_reference_count",
                    0,
                ),
            },
            next_action=(
                "Keep the reference audit in CI; add candidates only after "
                "public CLI/API smoke tests protect the contract."
            ),
        ),
        "unnecessary_data": _objective(
            status=_unnecessary_data_status(inventory),
            evidence={
                "current_workspace_delete_candidate_count": inventory.get(
                    "delete_candidate_count",
                    0,
                ),
                "safe_cache_count": inventory_summary.get("safe_cache", 0),
                "derived_artifact_count": inventory_summary.get(
                    "derived_artifact",
                    0,
                ),
                "active_runtime_delete_candidate_count": (
                    active_runtime_delete_count
                ),
                "strategy_evidence_delete_candidate_count": strategy_delete_count,
            },
            next_action=(
                "Only apply workspace cleanup when the inventory contains "
                "safe_cache or derived_artifact targets."
            ),
        ),
        "unreasonable_structure": _objective(
            status=_structure_status(
                architecture_audit,
                current_large_modules,
                architecture_remediation,
            ),
            evidence={
                "primary_finding_count": len(architecture_findings),
                "baseline_large_module_count": len(large_modules),
                "current_large_module_count": len(current_large_modules),
                "large_module_count": len(current_large_modules),
                "remediated_baseline_finding_count": (
                    architecture_remediation["resolved_finding_count"]
                ),
                "unresolved_baseline_finding_count": (
                    architecture_remediation["unresolved_finding_count"]
                ),
                "remediated_baseline_large_module_candidate_count": (
                    architecture_remediation[
                        "resolved_large_module_candidate_count"
                    ]
                ),
                "top_large_modules": [
                    item.get("path", "") for item in current_large_modules[:5]
                ],
                "current_large_modules": current_large_modules[:5],
                "baseline_finding_remediation": (
                    architecture_remediation["items"][:10]
                ),
                "remediated_baseline_large_module_candidates": (
                    architecture_remediation[
                        "resolved_large_module_candidates"
                    ]
                ),
            },
            next_action=(
                "If status is remediated_pending_rebaseline, run a fresh "
                "architecture audit to replace the stale baseline; otherwise "
                "split current large modules along tested boundaries."
            ),
        ),
        "legacy_code_files": _objective(
            status=_code_status(code_audit),
            evidence={
                "retired_runtime_candidate_count": code_audit.get(
                    "candidate_count",
                    0,
                ),
                "existing_candidate_count": existing_code_count,
                "production_reference_count": code_audit.get(
                    "production_reference_count",
                    0,
                ),
            },
            next_action=(
                "Keep public compatibility aliases while internal retired "
                "runtime paths remain absent from production references."
            ),
        ),
        "duplicate_storage": _objective(
            status=_duplicate_storage_status(
                duplicate_audit,
                cleanup_plan,
                cleanup_whitelist,
                cumulative_execute_summary,
            ),
            evidence={
                "duplicate_group_count": duplicate_summary.get(
                    "duplicate_group_count",
                    0,
                ),
                "duplicate_file_count": duplicate_summary.get(
                    "duplicate_file_count",
                    0,
                ),
                "candidate_group_count": plan_summary.get(
                    "candidate_group_count",
                    0,
                ),
                "candidate_file_count": plan_summary.get(
                    "candidate_file_count",
                    0,
                ),
                "potential_reclaim_bytes": plan_summary.get(
                    "potential_reclaim_bytes",
                    0,
                ),
                "delete_candidate_count": (
                    cleanup_plan or {}
                ).get("delete_candidate_count", 0),
                "restore_policy_group_count": restore_policy_summary.get(
                    "restore_source_group_count",
                    0,
                ),
                "restore_policy_candidate_file_count": (
                    restore_policy_summary.get(
                        "restore_source_candidate_file_count",
                        0,
                    )
                ),
                "restore_policy_potential_reclaim_bytes": (
                    restore_policy_summary.get("potential_reclaim_bytes", 0)
                ),
                "restore_policy_delete_candidate_count": (
                    restore_policy or {}
                ).get("delete_candidate_count", 0),
                "restore_policy_high_risk_count": (
                    restore_policy_summary.get("risk_level_summary", {}).get(
                        "high",
                        0,
                    )
                ),
                "restore_policy_medium_risk_count": (
                    restore_policy_summary.get("risk_level_summary", {}).get(
                        "medium",
                        0,
                    )
                ),
                "restore_reference_candidate_path_count": (
                    restore_reference_summary.get("candidate_path_count", 0)
                ),
                "restore_reference_referenced_candidate_path_count": (
                    restore_reference_summary.get(
                        "referenced_candidate_path_count",
                        0,
                    )
                ),
                "restore_reference_unreferenced_candidate_path_count": (
                    restore_reference_summary.get(
                        "unreferenced_candidate_path_count",
                        0,
                    )
                ),
                "restore_reference_referenced_group_count": (
                    restore_reference_summary.get("referenced_group_count", 0)
                ),
                "restore_reference_unreferenced_group_count": (
                    restore_reference_summary.get("unreferenced_group_count", 0)
                ),
                "restore_reference_scan_file_count": (
                    restore_reference_summary.get("scan_file_count", 0)
                ),
                "restore_reference_scan_mode": (
                    restore_reference_summary.get("scan_mode", "")
                ),
                "restore_readiness_reference_free_group_count": (
                    restore_readiness_summary.get("reference_free_group_count", 0)
                ),
                "restore_readiness_referenced_group_count": (
                    restore_readiness_summary.get("referenced_group_count", 0)
                ),
                "restore_readiness_reference_unknown_group_count": (
                    restore_readiness_summary.get("reference_unknown_group_count", 0)
                ),
                "restore_readiness_reference_free_candidate_path_count": (
                    restore_readiness_summary.get(
                        "reference_free_candidate_path_count",
                        0,
                    )
                ),
                "restore_readiness_reference_free_potential_reclaim_bytes": (
                    restore_readiness_summary.get(
                        "reference_free_potential_reclaim_bytes",
                        0,
                    )
                ),
                "restore_readiness_delete_candidate_count": (
                    restore_readiness or {}
                ).get("delete_candidate_count", 0),
                "restore_readiness_retained_copy_readback_required_count": (
                    restore_readiness_summary.get(
                        "readiness_class_summary",
                        {},
                    ).get("review_retained_copy_readback_required", 0)
                ),
                "restore_readiness_manifest_rewrite_required_count": (
                    restore_readiness_summary.get(
                        "readiness_class_summary",
                        {},
                    ).get("review_manifest_rewrite_required", 0)
                ),
                "restore_readiness_blocked_high_risk_policy_count": (
                    restore_readiness_summary.get(
                        "readiness_class_summary",
                        {},
                    ).get("blocked_high_risk_policy", 0)
                ),
                "restore_readiness_blocked_referenced_candidate_count": (
                    restore_readiness_summary.get(
                        "readiness_class_summary",
                        {},
                    ).get("blocked_referenced_candidate", 0)
                ),
                "restore_readiness_missing_reference_audit_group_count": (
                    restore_readiness_summary.get(
                        "readiness_class_summary",
                        {},
                    ).get("blocked_missing_reference_audit_group", 0)
                ),
                "restore_readback_reviewed_group_count": (
                    restore_readback_summary.get("reviewed_group_count", 0)
                ),
                "restore_readback_passed_group_count": (
                    restore_readback_summary.get(
                        "retained_copy_readback_passed_count",
                        0,
                    )
                ),
                "restore_readback_blocked_group_count": (
                    restore_readback_summary.get("blocked_count", 0)
                ),
                "restore_readback_verified_reclaim_bytes": (
                    restore_readback_summary.get("verified_reclaim_bytes", 0)
                ),
                "restore_readback_delete_candidate_count": (
                    restore_readback or {}
                ).get("delete_candidate_count", 0),
                "clear_but_delete_disabled_count": gate_summary.get(
                    "clear_but_delete_disabled_count",
                    0,
                ),
                "hash_readback_passed_count": readback_summary.get(
                    "hash_readback_passed_count",
                    0,
                ),
                "whitelist_item_count": whitelist_summary.get(
                    "whitelist_item_count",
                    0,
                ),
                "pending_manual_approval_count": (
                    effective_whitelist_summary[
                        "pending_manual_approval_count"
                    ]
                ),
                "approved_for_delete_count": (
                    effective_whitelist_summary["approved_for_delete_count"]
                ),
                "whitelist_potential_reclaim_bytes": (
                    effective_whitelist_summary[
                        "whitelist_potential_reclaim_bytes"
                    ]
                ),
                "whitelist_execute_allowed_count": (
                    effective_whitelist_summary[
                        "whitelist_execute_allowed_count"
                    ]
                ),
                "latest_deleted_count": execute_summary.get("deleted_count", 0),
                "latest_deleted_reclaim_bytes": execute_summary.get(
                    "deleted_reclaim_bytes",
                    0,
                ),
                "execution_report_count": cumulative_execute_summary[
                    "execution_report_count"
                ],
                "executed_report_count": cumulative_execute_summary[
                    "executed_report_count"
                ],
                "deleted_count": cumulative_execute_summary["deleted_count"],
                "deleted_reclaim_bytes": cumulative_execute_summary[
                    "deleted_reclaim_bytes"
                ],
                "deleted_group_count": cumulative_execute_summary[
                    "deleted_group_count"
                ],
                "reference_rewrite_report_count": (
                    cumulative_reference_rewrite_summary["report_count"]
                ),
                "reference_rewrite_executed_report_count": (
                    cumulative_reference_rewrite_summary["executed_report_count"]
                ),
                "reference_rewrite_latest_selected_group_count": (
                    reference_rewrite_summary.get("selected_group_count", 0)
                ),
                "reference_rewrite_latest_would_rewrite_delete_count": (
                    reference_rewrite_summary.get("would_rewrite_delete_count", 0)
                ),
                "reference_rewrite_latest_rewritten_deleted_count": (
                    reference_rewrite_summary.get("rewritten_deleted_count", 0)
                ),
                "reference_rewrite_latest_blocked_count": (
                    reference_rewrite_summary.get("blocked_count", 0)
                ),
                "reference_rewrite_rewritten_deleted_count": (
                    cumulative_reference_rewrite_summary[
                        "rewritten_deleted_count"
                    ]
                ),
                "reference_rewrite_rewritten_deleted_reclaim_bytes": (
                    cumulative_reference_rewrite_summary[
                        "rewritten_deleted_reclaim_bytes"
                    ]
                ),
                "reference_rewrite_references_rewritten_count": (
                    cumulative_reference_rewrite_summary[
                        "references_rewritten_count"
                    ]
                ),
                "empty_cell_flags_report_count": (
                    cumulative_empty_cell_flags_summary["report_count"]
                ),
                "empty_cell_flags_executed_report_count": (
                    cumulative_empty_cell_flags_summary["executed_report_count"]
                ),
                "empty_cell_flags_latest_candidate_count": (
                    cumulative_empty_cell_flags_summary["latest_candidate_count"]
                ),
                "empty_cell_flags_latest_would_compact_count": (
                    cumulative_empty_cell_flags_summary[
                        "latest_would_compact_count"
                    ]
                ),
                "empty_cell_flags_latest_compacted_count": (
                    empty_cell_flags_summary.get("compacted_count", 0)
                ),
                "empty_cell_flags_latest_compacted_reclaim_bytes": (
                    empty_cell_flags_summary.get("compacted_reclaim_bytes", 0)
                ),
                "empty_cell_flags_latest_orphan_deleted_count": (
                    cumulative_empty_cell_flags_summary[
                        "latest_orphan_deleted_count"
                    ]
                ),
                "empty_cell_flags_latest_orphan_deleted_reclaim_bytes": (
                    cumulative_empty_cell_flags_summary[
                        "latest_orphan_deleted_reclaim_bytes"
                    ]
                ),
                "empty_cell_flags_latest_blocked_count": (
                    cumulative_empty_cell_flags_summary["latest_blocked_count"]
                ),
                "empty_cell_flags_compacted_count": (
                    cumulative_empty_cell_flags_summary["compacted_count"]
                ),
                "empty_cell_flags_compacted_reclaim_bytes": (
                    cumulative_empty_cell_flags_summary[
                        "compacted_reclaim_bytes"
                    ]
                ),
                "empty_cell_flags_references_rewritten_count": (
                    cumulative_empty_cell_flags_summary[
                        "references_rewritten_count"
                    ]
                ),
                "empty_cell_flags_orphan_deleted_count": (
                    cumulative_empty_cell_flags_summary["orphan_deleted_count"]
                ),
                "empty_cell_flags_orphan_deleted_reclaim_bytes": (
                    cumulative_empty_cell_flags_summary[
                        "orphan_deleted_reclaim_bytes"
                    ]
                ),
                "issue_cell_flags_report_count": (
                    cumulative_issue_cell_flags_summary["report_count"]
                ),
                "issue_cell_flags_executed_report_count": (
                    cumulative_issue_cell_flags_summary["executed_report_count"]
                ),
                "issue_cell_flags_latest_candidate_count": (
                    cumulative_issue_cell_flags_summary["latest_candidate_count"]
                ),
                "issue_cell_flags_latest_would_compact_count": (
                    cumulative_issue_cell_flags_summary[
                        "latest_would_compact_count"
                    ]
                ),
                "issue_cell_flags_latest_compacted_count": (
                    issue_cell_flags_summary.get("compacted_count", 0)
                ),
                "issue_cell_flags_latest_compacted_reclaim_bytes": (
                    issue_cell_flags_summary.get("compacted_reclaim_bytes", 0)
                ),
                "issue_cell_flags_latest_blocked_count": (
                    cumulative_issue_cell_flags_summary["latest_blocked_count"]
                ),
                "issue_cell_flags_compacted_count": (
                    cumulative_issue_cell_flags_summary["compacted_count"]
                ),
                "issue_cell_flags_compacted_reclaim_bytes": (
                    cumulative_issue_cell_flags_summary[
                        "compacted_reclaim_bytes"
                    ]
                ),
                "issue_cell_flags_references_rewritten_count": (
                    cumulative_issue_cell_flags_summary[
                        "references_rewritten_count"
                    ]
                ),
                "issue_cell_flags_issue_row_count": (
                    cumulative_issue_cell_flags_summary["issue_row_count"]
                ),
                "uniform_row_flags_report_count": (
                    cumulative_uniform_row_flags_summary["report_count"]
                ),
                "uniform_row_flags_executed_report_count": (
                    cumulative_uniform_row_flags_summary["executed_report_count"]
                ),
                "uniform_row_flags_latest_candidate_count": (
                    cumulative_uniform_row_flags_summary["latest_candidate_count"]
                ),
                "uniform_row_flags_latest_would_compact_count": (
                    cumulative_uniform_row_flags_summary[
                        "latest_would_compact_count"
                    ]
                ),
                "uniform_row_flags_latest_compacted_count": (
                    uniform_row_flags_summary.get("compacted_count", 0)
                ),
                "uniform_row_flags_latest_compacted_reclaim_bytes": (
                    uniform_row_flags_summary.get("compacted_reclaim_bytes", 0)
                ),
                "uniform_row_flags_latest_blocked_count": (
                    cumulative_uniform_row_flags_summary["latest_blocked_count"]
                ),
                "uniform_row_flags_compacted_count": (
                    cumulative_uniform_row_flags_summary["compacted_count"]
                ),
                "uniform_row_flags_compacted_reclaim_bytes": (
                    cumulative_uniform_row_flags_summary[
                        "compacted_reclaim_bytes"
                    ]
                ),
                "uniform_row_flags_references_rewritten_count": (
                    cumulative_uniform_row_flags_summary[
                        "references_rewritten_count"
                    ]
                ),
                "matrix_coverage_report_count": (
                    cumulative_matrix_coverage_summary["report_count"]
                ),
                "matrix_coverage_executed_report_count": (
                    cumulative_matrix_coverage_summary["executed_report_count"]
                ),
                "matrix_coverage_latest_candidate_count": (
                    cumulative_matrix_coverage_summary["latest_candidate_count"]
                ),
                "matrix_coverage_latest_would_compact_count": (
                    cumulative_matrix_coverage_summary[
                        "latest_would_compact_count"
                    ]
                ),
                "matrix_coverage_latest_compacted_count": (
                    matrix_coverage_summary.get("compacted_count", 0)
                ),
                "matrix_coverage_latest_compacted_reclaim_bytes": (
                    matrix_coverage_summary.get("compacted_reclaim_bytes", 0)
                ),
                "matrix_coverage_latest_blocked_count": (
                    cumulative_matrix_coverage_summary["latest_blocked_count"]
                ),
                "matrix_coverage_compacted_count": (
                    cumulative_matrix_coverage_summary["compacted_count"]
                ),
                "matrix_coverage_compacted_reclaim_bytes": (
                    cumulative_matrix_coverage_summary[
                        "compacted_reclaim_bytes"
                    ]
                ),
                "matrix_coverage_references_rewritten_count": (
                    cumulative_matrix_coverage_summary[
                        "references_rewritten_count"
                    ]
                ),
            },
            next_action=(
                "Do not delete duplicate storage until manual approval, fresh "
                "storage validation, and restore readback all pass in the same run."
            ),
        ),
    }

    latest_profile = _runtime_profile_summary(
        runtime_profile,
        sources["latest_runtime_profile_json"],
    )
    performance = {
        "latest_profile": latest_profile,
        "stage_profile_available": latest_profile is not None,
    }

    data_safety = {
        "active_runtime_delete_candidate_count": active_runtime_delete_count,
        "strategy_evidence_delete_candidate_count": strategy_delete_count,
        "protected_inventory_count": inventory.get("protected_count", 0),
        "data_cleanup_execution_performed": bool(
            cumulative_execute_summary["deleted_count"] > 0
        ),
        "data_cleanup_deleted_count": cumulative_execute_summary["deleted_count"],
        "data_cleanup_deleted_reclaim_bytes": (
            cumulative_execute_summary["deleted_reclaim_bytes"]
        ),
        "data_cleanup_latest_deleted_count": execute_summary.get("deleted_count", 0),
        "reference_rewrite_execution_performed": bool(
            cumulative_reference_rewrite_summary["rewritten_deleted_count"] > 0
        ),
        "reference_rewrite_rewritten_deleted_count": (
            cumulative_reference_rewrite_summary["rewritten_deleted_count"]
        ),
        "reference_rewrite_rewritten_deleted_reclaim_bytes": (
            cumulative_reference_rewrite_summary[
                "rewritten_deleted_reclaim_bytes"
            ]
        ),
        "reference_rewrite_references_rewritten_count": (
            cumulative_reference_rewrite_summary["references_rewritten_count"]
        ),
        "restore_policy_delete_candidate_count": (
            restore_policy or {}
        ).get("delete_candidate_count", 0),
        "restore_reference_delete_candidate_count": (
            restore_reference_audit or {}
        ).get("delete_candidate_count", 0),
        "restore_readiness_delete_candidate_count": (
            restore_readiness or {}
        ).get("delete_candidate_count", 0),
        "restore_readback_delete_candidate_count": (
            restore_readback or {}
        ).get("delete_candidate_count", 0),
        "empty_cell_flags_compaction_performed": bool(
            cumulative_empty_cell_flags_summary["compacted_count"] > 0
            or cumulative_empty_cell_flags_summary["orphan_deleted_count"] > 0
        ),
        "empty_cell_flags_compacted_count": (
            cumulative_empty_cell_flags_summary["compacted_count"]
        ),
        "empty_cell_flags_compacted_reclaim_bytes": (
            cumulative_empty_cell_flags_summary["compacted_reclaim_bytes"]
        ),
        "empty_cell_flags_references_rewritten_count": (
            cumulative_empty_cell_flags_summary["references_rewritten_count"]
        ),
        "empty_cell_flags_orphan_deleted_count": (
            cumulative_empty_cell_flags_summary["orphan_deleted_count"]
        ),
        "empty_cell_flags_orphan_deleted_reclaim_bytes": (
            cumulative_empty_cell_flags_summary["orphan_deleted_reclaim_bytes"]
        ),
        "empty_cell_flags_latest_blocked_count": (
            cumulative_empty_cell_flags_summary["latest_blocked_count"]
        ),
        "issue_cell_flags_compaction_performed": bool(
            cumulative_issue_cell_flags_summary["compacted_count"] > 0
        ),
        "issue_cell_flags_compacted_count": (
            cumulative_issue_cell_flags_summary["compacted_count"]
        ),
        "issue_cell_flags_compacted_reclaim_bytes": (
            cumulative_issue_cell_flags_summary["compacted_reclaim_bytes"]
        ),
        "issue_cell_flags_references_rewritten_count": (
            cumulative_issue_cell_flags_summary["references_rewritten_count"]
        ),
        "issue_cell_flags_issue_row_count": (
            cumulative_issue_cell_flags_summary["issue_row_count"]
        ),
        "issue_cell_flags_latest_blocked_count": (
            cumulative_issue_cell_flags_summary["latest_blocked_count"]
        ),
        "uniform_row_flags_compaction_performed": bool(
            cumulative_uniform_row_flags_summary["compacted_count"] > 0
        ),
        "uniform_row_flags_compacted_count": (
            cumulative_uniform_row_flags_summary["compacted_count"]
        ),
        "uniform_row_flags_compacted_reclaim_bytes": (
            cumulative_uniform_row_flags_summary["compacted_reclaim_bytes"]
        ),
        "uniform_row_flags_references_rewritten_count": (
            cumulative_uniform_row_flags_summary["references_rewritten_count"]
        ),
        "uniform_row_flags_latest_blocked_count": (
            cumulative_uniform_row_flags_summary["latest_blocked_count"]
        ),
        "matrix_coverage_compaction_performed": bool(
            cumulative_matrix_coverage_summary["compacted_count"] > 0
        ),
        "matrix_coverage_compacted_count": (
            cumulative_matrix_coverage_summary["compacted_count"]
        ),
        "matrix_coverage_compacted_reclaim_bytes": (
            cumulative_matrix_coverage_summary["compacted_reclaim_bytes"]
        ),
        "matrix_coverage_references_rewritten_count": (
            cumulative_matrix_coverage_summary["references_rewritten_count"]
        ),
        "matrix_coverage_latest_blocked_count": (
            cumulative_matrix_coverage_summary["latest_blocked_count"]
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "root": str(repo_root),
        "mutation_status": {
            "source_edits": False,
            "data_deletions": False,
            "read_only_summary": True,
        },
        "objectives": objectives,
        "data_safety": data_safety,
        "performance": performance,
        "sources": sources,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_project_cleanup_status_markdown(status: dict[str, Any]) -> str:
    """Render a compact Markdown cleanup status report."""
    lines = [
        "# Project Cleanup Status",
        "",
        f"- Schema: `{status.get('schema_version', '')}`",
        f"- Generated at: `{status.get('generated_at', '')}`",
        f"- Root: `{status.get('root', '')}`",
        "- Mode: read-only summary; no source edits or data deletions",
        "",
        "## Objectives",
        "",
        "| Objective | Status | Key Evidence | Next Action |",
        "| --- | --- | --- | --- |",
    ]
    for name, objective in status.get("objectives", {}).items():
        evidence = ", ".join(
            f"{key}={value}"
            for key, value in objective.get("evidence", {}).items()
            if not isinstance(value, (list, dict))
        )
        lines.append(
            "| {name} | {status_text} | {evidence} | {next_action} |".format(
                name=_markdown_cell(name),
                status_text=_markdown_cell(objective.get("status", "")),
                evidence=_markdown_cell(evidence or "-"),
                next_action=_markdown_cell(objective.get("next_action", "")),
            )
        )

    data_safety = status.get("data_safety", {})
    lines.extend(
        [
            "",
            "## Data Safety",
            "",
        ]
    )
    for key, value in data_safety.items():
        lines.append(f"- {key}: `{value}`")

    performance = status.get("performance", {})
    latest_profile = performance.get("latest_profile") or {}
    lines.extend(
        [
            "",
            "## Performance Evidence",
            "",
            f"- Stage profile available: `{performance.get('stage_profile_available')}`",
            f"- Latest profile: `{latest_profile.get('source_json')}`",
            f"- Total seconds: `{latest_profile.get('total_seconds')}`",
            f"- Stage count: `{latest_profile.get('stage_count')}`",
            (
                "- Per-symbol fallback count: "
                f"`{latest_profile.get('per_symbol_fallback_count')}`"
            ),
            "",
            "## Sources",
            "",
        ]
    )
    for key, value in status.get("sources", {}).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def write_project_cleanup_status(
    root: Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write JSON and Markdown cleanup status reports."""
    repo_root = get_repo_root(root)
    status = build_project_cleanup_status(repo_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"project_cleanup_status_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "project_cleanup_status.json"
    md_path = out_dir / "project_cleanup_status.md"
    json_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_project_cleanup_status_markdown(status),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "md": str(md_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a read-only project cleanup status report."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown cleanup status reports.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = get_repo_root(args.root)
    paths = write_project_cleanup_status(
        repo_root,
        output_dir=args.output_dir,
    )
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    print("project cleanup status:")
    print(f"workspace root: {payload['root']}")
    for name, objective in payload["objectives"].items():
        print(f"{name}: {objective['status']}")
    print("project cleanup status manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
