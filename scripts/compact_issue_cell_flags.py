"""Compact issue-backed Tushare cell-flag CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import get_repo_root

SCHEMA_VERSION = "myquant.issue_cell_flags_compaction.v1"
CONFIRM_TOKEN = "COMPACT_ISSUE_CELL_FLAGS"
DEFAULT_MAX_MARKDOWN_ITEMS = 500
DEFAULT_MAX_FILE_BYTES = 128 * 1024 * 1024
CELL_FLAGS_REFERENCE_RE = re.compile(
    r"data/cleaning_reports/tushare/[A-Za-z0-9_./=-]+_cell_flags\.csv"
)
REFERENCE_SCAN_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
REFERENCE_SCAN_ROOTS = (
    ("data", "cleaning_reports", "tushare"),
    ("data", "factor_readiness", "tushare"),
    ("reports",),
    ("results", "strategy_records"),
)
IGNORED_EXTERNAL_REFERENCE_PREFIXES = ("reports/storage/csv_inventory_",)


@dataclass(frozen=True)
class IssueCellFlagsCompactionItem:
    group_id: str
    cell_flags_path: str
    status: str
    action: str
    reclaimable_bytes: int
    cleaning_report_path: str | None
    factor_readiness_report_path: str | None
    issue_row_count: int
    references_rewritten: int
    ignored_external_reference_count: int
    errors: list[str]
    reason: str


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _latest_json(repo_root: Path, pattern: str) -> Path:
    candidates = sorted(repo_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no report found for {pattern}")
    return candidates[-1]


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative_path(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _cleaning_report_for_cell_flags(path: str) -> str:
    suffix = "_cell_flags.csv"
    if not path.endswith(suffix):
        return ""
    return path[: -len(suffix)] + "_cleaning_report.json"


def _factor_report_for_cell_flags(path: str) -> str:
    prefix = "data/cleaning_reports/tushare/"
    suffix = "_cell_flags.csv"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return ""
    body = path[len(prefix) : -len(suffix)]
    return f"data/factor_readiness/tushare/{body}_factor_readiness_report.json"


def _iter_external_reference_files(repo_root: Path) -> tuple[list[Path], int]:
    files: list[Path] = []
    skipped_project_cleanup = 0
    project_cleanup_root = repo_root / "reports" / "project_cleanup"
    for parts in REFERENCE_SCAN_ROOTS:
        root = repo_root.joinpath(*parts)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in REFERENCE_SCAN_SUFFIXES:
                continue
            if path.is_relative_to(project_cleanup_root):
                skipped_project_cleanup += 1
                continue
            files.append(path)
    return files, skipped_project_cleanup


def _scan_external_references(
    repo_root: Path,
    references: list[str],
    *,
    max_file_bytes: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    reference_map = {reference: [] for reference in references}
    if not references:
        return reference_map, {
            "external_reference_scan_file_count": 0,
            "external_reference_skipped_project_cleanup_count": 0,
            "external_reference_read_error_count": 0,
            "external_reference_skipped_large_file_count": 0,
            "external_reference_match_count": 0,
        }
    files, skipped_project_cleanup = _iter_external_reference_files(repo_root)
    reference_set = set(references)
    read_error_count = 0
    skipped_large_count = 0
    match_count = 0
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            read_error_count += 1
            continue
        if size > max_file_bytes:
            skipped_large_count += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            read_error_count += 1
            continue
        if "_cell_flags.csv" not in text:
            continue
        relative_path = _relative_path(repo_root, path)
        for reference in set(CELL_FLAGS_REFERENCE_RE.findall(text)):
            if reference not in reference_set:
                continue
            reference_map[reference].append(relative_path)
            match_count += 1
    return reference_map, {
        "external_reference_scan_file_count": len(files),
        "external_reference_skipped_project_cleanup_count": skipped_project_cleanup,
        "external_reference_read_error_count": read_error_count,
        "external_reference_skipped_large_file_count": skipped_large_count,
        "external_reference_match_count": match_count,
    }


def _is_ignored_external_reference(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in IGNORED_EXTERNAL_REFERENCE_PREFIXES)


def _candidate_items(
    policy: dict[str, Any],
    reference_audit: dict[str, Any],
) -> list[tuple[str, str]]:
    reference_groups = {
        str(group.get("group_id", "")): group
        for group in reference_audit.get("groups", [])
        if isinstance(group, dict)
    }
    items: list[tuple[str, str]] = []
    for group in policy.get("groups", []):
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id", ""))
        reference_group = reference_groups.get(group_id)
        if not reference_group:
            continue
        if group.get("policy_class") != "cross_symbol_generated_artifact_duplicate":
            continue
        if group.get("risk_level") != "high":
            continue
        if "cleaning_cell_flags" not in group.get("path_roles", []):
            continue
        referenced = {
            str(path)
            for path in reference_group.get("referenced_candidate_paths", [])
        }
        for candidate_path in group.get("candidate_paths", []):
            candidate_text = str(candidate_path)
            if not candidate_text.endswith("_cell_flags.csv"):
                continue
            if candidate_text not in referenced:
                continue
            items.append((group_id, candidate_text))
    return items


def _parse_primary_key(value: str) -> tuple[dict[str, Any] | None, str | None]:
    text = value.strip()
    if not text:
        return {}, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"primary_key_json is invalid: {exc}"
    if not isinstance(payload, dict):
        return None, "primary_key_json is not an object"
    return payload, None


def _none_if_empty(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _read_issue_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    required = {
        "row_index",
        "primary_key_json",
        "column",
        "issue_code",
        "severity",
        "message",
    }
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(required - columns)
            if missing:
                return [], "cell flags columns missing: " + ", ".join(missing)
            for row in reader:
                row_index_text = str(row.get("row_index", "")).strip()
                try:
                    row_index = int(row_index_text)
                except ValueError:
                    return [], "row_index is not an integer"
                primary_key, error = _parse_primary_key(
                    str(row.get("primary_key_json", ""))
                )
                if error:
                    return [], error
                rows.append(
                    {
                        "row_index": row_index,
                        "primary_key": primary_key,
                        "column": _none_if_empty(row.get("column")),
                        "issue_code": str(row.get("issue_code", "")).strip(),
                        "severity": str(row.get("severity", "")).strip(),
                        "message": str(row.get("message", "")).strip(),
                    }
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return [], f"cell flags read failed: {exc}"
    if not rows:
        return [], "cell flags file has no issue rows"
    return rows, None


def _issue_matches_row(issue: dict[str, Any], row: dict[str, Any]) -> bool:
    return (
        issue.get("row_index") == row["row_index"]
        and issue.get("primary_key") == row["primary_key"]
        and _none_if_empty(issue.get("column")) == row["column"]
        and str(issue.get("issue_code", "")).strip() == row["issue_code"]
        and str(issue.get("severity", "")).strip() == row["severity"]
        and str(issue.get("message", "")).strip() == row["message"]
    )


def _all_rows_embedded(
    payload: dict[str, Any] | None,
    issue_rows: list[dict[str, Any]],
) -> bool:
    if not payload:
        return False
    issue_dicts: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            issues = value.get("issues")
            if isinstance(issues, list):
                issue_dicts.extend(
                    issue for issue in issues if isinstance(issue, dict)
                )
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(payload)
    return bool(issue_dicts) and all(
        any(_issue_matches_row(issue, row) for issue in issue_dicts)
        for row in issue_rows
    )


def _set_compacted_metadata(
    payload: dict[str, Any],
    *,
    planned_path: str,
    issue_row_count: int,
) -> None:
    payload["cell_flags_path"] = None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata
    metadata.update(
        {
            "cell_flags_issue_backed_compacted": True,
            "cell_flags_path_suppressed": True,
            "cell_flags_planned_path": planned_path,
            "cell_flags_compaction_schema": SCHEMA_VERSION,
            "cell_flags_issue_row_count": issue_row_count,
            "cell_flags_issues_embedded_in_json": True,
        }
    )


def _rewrite_cell_flags_references(
    payload: Any,
    *,
    old_path: str,
    issue_row_count: int,
) -> int:
    rewritten = 0
    if isinstance(payload, dict):
        if payload.get("cell_flags_path") == old_path:
            _set_compacted_metadata(
                payload,
                planned_path=old_path,
                issue_row_count=issue_row_count,
            )
            rewritten += 1
        for value in payload.values():
            rewritten += _rewrite_cell_flags_references(
                value,
                old_path=old_path,
                issue_row_count=issue_row_count,
            )
    elif isinstance(payload, list):
        for value in payload:
            rewritten += _rewrite_cell_flags_references(
                value,
                old_path=old_path,
                issue_row_count=issue_row_count,
            )
    return rewritten


def _compaction_item(
    repo_root: Path,
    *,
    group_id: str,
    cell_flags_path: str,
    external_reference_map: dict[str, list[str]],
    apply: bool,
    confirm_token: str | None,
) -> IssueCellFlagsCompactionItem:
    cell_flags_abs = repo_root / cell_flags_path
    cleaning_report_path = _cleaning_report_for_cell_flags(cell_flags_path)
    cleaning_report_abs = repo_root / cleaning_report_path if cleaning_report_path else None
    errors: list[str] = []
    reclaimable_bytes = 0
    references_rewritten = 0
    issue_row_count = 0

    if not cell_flags_abs.exists():
        errors.append("cell flags file missing")
    elif not cell_flags_abs.is_file():
        errors.append("cell flags path is not a file")
    else:
        reclaimable_bytes = cell_flags_abs.stat().st_size

    issue_rows: list[dict[str, Any]] = []
    if not errors:
        issue_rows, issue_error = _read_issue_rows(cell_flags_abs)
        if issue_error:
            errors.append(issue_error)
        issue_row_count = len(issue_rows)

    cleaning_report = _load_json(cleaning_report_abs)
    if cleaning_report_abs is None:
        errors.append("could not derive owner cleaning report path")
    elif not cleaning_report:
        errors.append("owner cleaning report missing or invalid")
    elif cleaning_report.get("cell_flags_path") != cell_flags_path:
        errors.append("owner cleaning report does not reference cell flags path")

    metadata = cleaning_report.get("metadata") if cleaning_report else None
    if not isinstance(metadata, dict):
        metadata = {}
    factor_report_path = str(metadata.get("factor_readiness_report_path") or "")
    if not factor_report_path:
        factor_report_path = _factor_report_for_cell_flags(cell_flags_path)
    factor_report_abs = repo_root / factor_report_path if factor_report_path else None
    factor_report = _load_json(factor_report_abs)
    if not factor_report_abs:
        errors.append("could not derive factor readiness report path")
    elif not factor_report:
        errors.append("factor readiness report missing or invalid")

    if issue_rows:
        if not _all_rows_embedded(cleaning_report, issue_rows):
            errors.append("cell flag issue rows are not embedded in cleaning report")
        if not _all_rows_embedded(factor_report, issue_rows):
            errors.append("cell flag issue rows are not embedded in factor report")

    expected_references = {
        path
        for path in (cleaning_report_path, factor_report_path)
        if path
    }
    unexpected_references = [
        reference_path
        for reference_path in external_reference_map.get(cell_flags_path, [])
        if reference_path not in expected_references
        and not _is_ignored_external_reference(reference_path)
    ]
    ignored_external_reference_count = sum(
        1
        for reference_path in external_reference_map.get(cell_flags_path, [])
        if reference_path not in expected_references
        and _is_ignored_external_reference(reference_path)
    )
    if unexpected_references:
        errors.append(
            "unexpected references: "
            + ", ".join(sorted(set(unexpected_references))[:5])
        )

    updated_cleaning_report: dict[str, Any] | None = None
    updated_factor_report: dict[str, Any] | None = None
    if cleaning_report and factor_report and issue_rows:
        updated_cleaning_report = deepcopy(cleaning_report)
        cleaning_rewrites = _rewrite_cell_flags_references(
            updated_cleaning_report,
            old_path=cell_flags_path,
            issue_row_count=issue_row_count,
        )
        updated_factor_report = deepcopy(factor_report)
        factor_rewrites = _rewrite_cell_flags_references(
            updated_factor_report,
            old_path=cell_flags_path,
            issue_row_count=issue_row_count,
        )
        references_rewritten = cleaning_rewrites + factor_rewrites
        if cleaning_rewrites <= 0:
            errors.append("cleaning report reference rewrite did not occur")
        if factor_rewrites <= 0:
            errors.append("factor report reference rewrite did not occur")

    if errors:
        return IssueCellFlagsCompactionItem(
            group_id=group_id,
            cell_flags_path=cell_flags_path,
            status="blocked",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=cleaning_report_path or None,
            factor_readiness_report_path=factor_report_path or None,
            issue_row_count=issue_row_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=errors,
            reason="issue-backed cell flags compaction preconditions failed",
        )

    if not apply:
        return IssueCellFlagsCompactionItem(
            group_id=group_id,
            cell_flags_path=cell_flags_path,
            status="would_compact",
            action="dry_run_compact",
            reclaimable_bytes=reclaimable_bytes,
            cleaning_report_path=cleaning_report_path or None,
            factor_readiness_report_path=factor_report_path or None,
            issue_row_count=issue_row_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=[],
            reason="dry-run only; pass --apply with confirmation token",
        )

    if confirm_token != CONFIRM_TOKEN:
        return IssueCellFlagsCompactionItem(
            group_id=group_id,
            cell_flags_path=cell_flags_path,
            status="blocked_confirm_token_required",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=cleaning_report_path or None,
            factor_readiness_report_path=factor_report_path or None,
            issue_row_count=issue_row_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=["invalid or missing confirmation token"],
            reason="apply requires explicit confirmation token",
        )

    try:
        if (
            updated_cleaning_report is None
            or updated_factor_report is None
            or cleaning_report_abs is None
            or factor_report_abs is None
        ):
            raise OSError("updated reports missing")
        _write_json(cleaning_report_abs, updated_cleaning_report)
        _write_json(factor_report_abs, updated_factor_report)
        cell_flags_abs.unlink()
    except OSError as exc:
        return IssueCellFlagsCompactionItem(
            group_id=group_id,
            cell_flags_path=cell_flags_path,
            status="blocked_apply_failed",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=cleaning_report_path or None,
            factor_readiness_report_path=factor_report_path or None,
            issue_row_count=issue_row_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=[str(exc)],
            reason="failed while writing reports or deleting cell flags sidecar",
        )

    return IssueCellFlagsCompactionItem(
        group_id=group_id,
        cell_flags_path=cell_flags_path,
        status="compacted",
        action="compact",
        reclaimable_bytes=reclaimable_bytes,
        cleaning_report_path=cleaning_report_path or None,
        factor_readiness_report_path=factor_report_path or None,
        issue_row_count=issue_row_count,
        references_rewritten=references_rewritten,
        ignored_external_reference_count=ignored_external_reference_count,
        errors=[],
        reason="cell flags sidecar removed; issue rows remain embedded in JSON reports",
    )


def build_issue_cell_flags_compaction_report(
    policy: dict[str, Any],
    reference_audit: dict[str, Any],
    *,
    repo_root: Path,
    policy_json_path: Path | None = None,
    reference_audit_json_path: Path | None = None,
    apply: bool = False,
    confirm_token: str | None = None,
    max_items: int | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    candidates = _candidate_items(policy, reference_audit)
    skipped_by_limit = 0
    if max_items is not None:
        skipped_by_limit = max(0, len(candidates) - max_items)
        candidates = candidates[:max_items]
    candidate_paths = [path for _group_id, path in candidates]
    reference_map, reference_summary = _scan_external_references(
        repo_root,
        candidate_paths,
        max_file_bytes=max_file_bytes,
    )
    items = [
        asdict(
            _compaction_item(
                repo_root,
                group_id=group_id,
                cell_flags_path=cell_flags_path,
                external_reference_map=reference_map,
                apply=apply,
                confirm_token=confirm_token,
            )
        )
        for group_id, cell_flags_path in candidates
    ]
    status_summary: dict[str, int] = {}
    for item in items:
        status = str(item["status"])
        status_summary[status] = status_summary.get(status, 0) + 1
    blocked_count = sum(
        count
        for status, count in status_summary.items()
        if status.startswith("blocked") or status == "blocked"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "source_policy_json": str(policy_json_path) if policy_json_path else None,
        "source_reference_audit_json": (
            str(reference_audit_json_path) if reference_audit_json_path else None
        ),
        "root": str(repo_root),
        "apply_requested": apply,
        "confirm_token_valid": confirm_token == CONFIRM_TOKEN,
        "execution_performed": apply and status_summary.get("compacted", 0) > 0,
        "delete_candidate_count": 0,
        "summary": {
            **reference_summary,
            "candidate_count": len(candidates),
            "skipped_by_limit_count": skipped_by_limit,
            "would_compact_count": status_summary.get("would_compact", 0),
            "compacted_count": status_summary.get("compacted", 0),
            "blocked_count": blocked_count,
            "planned_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in items
                if item["status"] == "would_compact"
            ),
            "compacted_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in items
                if item["status"] == "compacted"
            ),
            "references_rewritten_count": sum(
                int(item["references_rewritten"])
                for item in items
                if item["status"] in {"would_compact", "compacted"}
            ),
            "issue_row_count": sum(
                int(item["issue_row_count"])
                for item in items
                if item["status"] in {"would_compact", "compacted"}
            ),
            "ignored_external_reference_count": sum(
                int(item["ignored_external_reference_count"])
                for item in items
            ),
            "status_summary": status_summary,
        },
        "items": items,
    }


def render_issue_cell_flags_compaction_markdown(
    report: dict[str, Any],
    *,
    max_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> str:
    summary = report.get("summary", {})
    items = report.get("items", [])
    visible_items = items[:max_items]
    lines = [
        "# Issue Cell Flags Compaction Report",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Apply requested: `{report.get('apply_requested', False)}`",
        f"- Execution performed: `{report.get('execution_performed', False)}`",
        f"- Candidates: `{summary.get('candidate_count', 0)}`",
        f"- Would compact: `{summary.get('would_compact_count', 0)}`",
        f"- Compacted: `{summary.get('compacted_count', 0)}`",
        f"- Blocked: `{summary.get('blocked_count', 0)}`",
        f"- Planned reclaim bytes: `{summary.get('planned_reclaim_bytes', 0)}`",
        (
            "- Compacted reclaim bytes: "
            f"`{summary.get('compacted_reclaim_bytes', 0)}`"
        ),
        "",
        "## Items",
        "",
        "| Group | Status | Cell Flags | Reclaim Bytes | Issue Rows | Rewrites |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in visible_items:
        lines.append(
            "| {group} | {status} | `{path}` | {bytes} | {rows} | {rewrites} |".format(
                group=item.get("group_id", ""),
                status=item.get("status", ""),
                path=item.get("cell_flags_path", ""),
                bytes=item.get("reclaimable_bytes", 0),
                rows=item.get("issue_row_count", 0),
                rewrites=item.get("references_rewritten", 0),
            )
        )
    if len(items) > len(visible_items):
        lines.append("")
        lines.append(
            f"_Item table truncated to {len(visible_items)} of {len(items)} rows._"
        )
    return "\n".join(lines) + "\n"


def write_issue_cell_flags_compaction_report(
    *,
    root: Path | None = None,
    policy_json_path: Path | None = None,
    reference_audit_json_path: Path | None = None,
    output_dir: Path | None = None,
    apply: bool = False,
    confirm_token: str | None = None,
    max_items: int | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_markdown_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    policy_path = policy_json_path or _latest_json(
        repo_root,
        "reports/project_cleanup/"
        "data_cleanup_restore_policy_*/data_cleanup_restore_policy.json",
    )
    reference_path = reference_audit_json_path or _latest_json(
        repo_root,
        "reports/project_cleanup/"
        "data_cleanup_restore_reference_audit_*/"
        "data_cleanup_restore_reference_audit.json",
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    reference_audit = json.loads(reference_path.read_text(encoding="utf-8"))
    report = build_issue_cell_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=repo_root,
        policy_json_path=policy_path,
        reference_audit_json_path=reference_path,
        apply=apply,
        confirm_token=confirm_token,
        max_items=max_items,
        max_file_bytes=max_file_bytes,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_dir or (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"issue_cell_flags_compaction_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "issue_cell_flags_compaction.json"
    md_path = out_dir / "issue_cell_flags_compaction.md"
    _write_json(json_path, report)
    md_path.write_text(
        render_issue_cell_flags_compaction_markdown(
            report,
            max_items=max_markdown_items,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compact issue-backed cell_flags CSV sidecars after proving the "
            "issue rows are embedded in owner JSON reports."
        )
    )
    parser.add_argument("--policy-json", type=Path, default=None)
    parser.add_argument("--reference-audit-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-markdown-items", type=int, default=DEFAULT_MAX_MARKDOWN_ITEMS)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-token",
        default=None,
        help=f"Required token for --apply: {CONFIRM_TOKEN}",
    )
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    max_items = args.max_items
    if max_items is not None and max_items < 0:
        max_items = None
    paths = write_issue_cell_flags_compaction_report(
        root=args.root,
        policy_json_path=args.policy_json,
        reference_audit_json_path=args.reference_audit_json,
        output_dir=args.output_dir,
        apply=args.apply,
        confirm_token=args.confirm_token,
        max_items=max_items,
        max_file_bytes=max(0, args.max_file_bytes),
        max_markdown_items=max(0, args.max_markdown_items),
    )
    payload = _load_json(Path(paths["json"])) or {}
    summary = payload.get("summary") or {}
    mode = "apply" if args.apply else "dry-run"
    print(f"issue cell flags compaction mode: {mode}")
    print(f"workspace root: {payload.get('root', '')}")
    print(f"candidates: {summary.get('candidate_count', 0)}")
    print(f"would compact: {summary.get('would_compact_count', 0)}")
    print(f"compacted: {summary.get('compacted_count', 0)}")
    print(f"blocked: {summary.get('blocked_count', 0)}")
    print("issue cell flags compaction manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
