"""Compact redundant matrix-coverage sidecars into readiness-report lineage."""

from __future__ import annotations

import argparse
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

SCHEMA_VERSION = "myquant.matrix_coverage_sidecar_compaction.v1"
CONFIRM_TOKEN = "COMPACT_MATRIX_COVERAGE_SIDECARS"
DEFAULT_MAX_MARKDOWN_ITEMS = 500
DEFAULT_MAX_FILE_BYTES = 128 * 1024 * 1024
MATRIX_REFERENCE_RE = re.compile(
    r"data/factor_readiness/tushare/[A-Za-z0-9_./=-]+_matrix_coverage\.json"
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
class MatrixCoverageCompactionItem:
    group_id: str
    matrix_coverage_path: str
    status: str
    action: str
    reclaimable_bytes: int
    cleaning_report_path: str | None
    factor_readiness_report_path: str | None
    summary_count: int
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


def _load_json(path: Path) -> dict[str, Any] | None:
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


def _cleaning_report_for_matrix(matrix_path: str) -> str:
    prefix = "data/factor_readiness/tushare/"
    suffix = "_matrix_coverage.json"
    if not matrix_path.startswith(prefix) or not matrix_path.endswith(suffix):
        return ""
    body = matrix_path[len(prefix) : -len(suffix)]
    return f"data/cleaning_reports/tushare/{body}_cleaning_report.json"


def _factor_report_for_matrix(matrix_path: str) -> str:
    suffix = "_matrix_coverage.json"
    if not matrix_path.endswith(suffix):
        return ""
    return matrix_path[: -len(suffix)] + "_factor_readiness_report.json"


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
        if "_matrix_coverage.json" not in text:
            continue
        relative_path = _relative_path(repo_root, path)
        for reference in set(MATRIX_REFERENCE_RE.findall(text)):
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
        if "factor_matrix_coverage" not in group.get("path_roles", []):
            continue
        referenced = {
            str(path)
            for path in reference_group.get("referenced_candidate_paths", [])
        }
        for candidate_path in group.get("candidate_paths", []):
            candidate_text = str(candidate_path)
            if not candidate_text.endswith("_matrix_coverage.json"):
                continue
            if candidate_text not in referenced:
                continue
            items.append((group_id, candidate_text))
    return items


def _set_compacted_metadata(
    payload: dict[str, Any],
    *,
    planned_path: str,
    factor_report_path: str,
    summary_count: int,
) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata
    metadata.update(
        {
            "matrix_coverage_path": None,
            "matrix_coverage_path_suppressed": True,
            "matrix_coverage_planned_path": planned_path,
            "matrix_coverage_compaction_schema": SCHEMA_VERSION,
            "matrix_coverage_embedded_in_factor_readiness_report": True,
            "matrix_coverage_factor_readiness_report_path": factor_report_path,
            "matrix_coverage_summary_count": summary_count,
        }
    )


def _coverage_matches_factor_report(
    matrix_payload: dict[str, Any],
    factor_payload: dict[str, Any],
) -> tuple[bool, str | None, int]:
    matrix_summaries = matrix_payload.get("summaries")
    factor_summaries = factor_payload.get("coverage_summaries")
    if matrix_summaries != factor_summaries:
        return False, "matrix coverage differs from factor readiness coverage", 0
    if matrix_payload.get("schema_version") != factor_payload.get("schema_version"):
        return False, "matrix coverage schema differs from factor readiness report", 0
    if matrix_payload.get("generated_at") != factor_payload.get("generated_at"):
        return False, "matrix coverage generated_at differs from factor report", 0
    if not isinstance(matrix_summaries, list) or not matrix_summaries:
        return False, "matrix coverage has no summaries", 0
    return True, None, len(matrix_summaries)


def _compaction_item(
    repo_root: Path,
    *,
    group_id: str,
    matrix_path: str,
    external_reference_map: dict[str, list[str]],
    apply: bool,
    confirm_token: str | None,
) -> MatrixCoverageCompactionItem:
    matrix_abs = repo_root / matrix_path
    cleaning_report_path = _cleaning_report_for_matrix(matrix_path)
    cleaning_report_abs = repo_root / cleaning_report_path if cleaning_report_path else None
    errors: list[str] = []
    reclaimable_bytes = 0
    references_rewritten = 0
    summary_count = 0

    if not matrix_abs.exists():
        errors.append("matrix coverage file missing")
    elif not matrix_abs.is_file():
        errors.append("matrix coverage path is not a file")
    else:
        reclaimable_bytes = matrix_abs.stat().st_size

    matrix_payload = _load_json(matrix_abs) if not errors else None
    if matrix_payload is None and not errors:
        errors.append("matrix coverage json missing or invalid")

    cleaning_report = (
        _load_json(cleaning_report_abs)
        if cleaning_report_abs is not None
        else None
    )
    if cleaning_report_abs is None:
        errors.append("could not derive owner cleaning report path")
    elif not cleaning_report:
        errors.append("owner cleaning report missing or invalid")
    metadata = cleaning_report.get("metadata") if cleaning_report else None
    if not isinstance(metadata, dict):
        metadata = {}
    if cleaning_report and metadata.get("matrix_coverage_path") != matrix_path:
        errors.append("owner cleaning report does not reference matrix coverage path")

    factor_report_path = str(metadata.get("factor_readiness_report_path") or "")
    if not factor_report_path:
        factor_report_path = _factor_report_for_matrix(matrix_path)
    factor_report_abs = repo_root / factor_report_path if factor_report_path else None
    factor_payload = _load_json(factor_report_abs) if factor_report_abs else None
    if not factor_report_abs:
        errors.append("could not derive factor readiness report path")
    elif not factor_payload:
        errors.append("factor readiness report missing or invalid")

    if matrix_payload and factor_payload:
        matches, match_error, count = _coverage_matches_factor_report(
            matrix_payload,
            factor_payload,
        )
        if not matches and match_error:
            errors.append(match_error)
        summary_count = count

    expected_references = {cleaning_report_path} if cleaning_report_path else set()
    external_references = external_reference_map.get(matrix_path, [])
    unexpected_references = [
        reference_path
        for reference_path in external_references
        if reference_path not in expected_references
        and not _is_ignored_external_reference(reference_path)
    ]
    ignored_external_reference_count = sum(
        1
        for reference_path in external_references
        if reference_path not in expected_references
        and _is_ignored_external_reference(reference_path)
    )
    if unexpected_references:
        errors.append(
            "unexpected references: "
            + ", ".join(sorted(set(unexpected_references))[:5])
        )

    updated_cleaning_report: dict[str, Any] | None = None
    if cleaning_report and factor_report_path and summary_count:
        updated_cleaning_report = deepcopy(cleaning_report)
        _set_compacted_metadata(
            updated_cleaning_report,
            planned_path=matrix_path,
            factor_report_path=factor_report_path,
            summary_count=summary_count,
        )
        references_rewritten = 1

    if errors:
        return MatrixCoverageCompactionItem(
            group_id=group_id,
            matrix_coverage_path=matrix_path,
            status="blocked",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=cleaning_report_path or None,
            factor_readiness_report_path=factor_report_path or None,
            summary_count=summary_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=errors,
            reason="matrix coverage compaction preconditions failed",
        )

    if not apply:
        return MatrixCoverageCompactionItem(
            group_id=group_id,
            matrix_coverage_path=matrix_path,
            status="would_compact",
            action="dry_run_compact",
            reclaimable_bytes=reclaimable_bytes,
            cleaning_report_path=cleaning_report_path or None,
            factor_readiness_report_path=factor_report_path or None,
            summary_count=summary_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=[],
            reason="dry-run only; pass --apply with confirmation token",
        )

    if confirm_token != CONFIRM_TOKEN:
        return MatrixCoverageCompactionItem(
            group_id=group_id,
            matrix_coverage_path=matrix_path,
            status="blocked_confirm_token_required",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=cleaning_report_path or None,
            factor_readiness_report_path=factor_report_path or None,
            summary_count=summary_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=["invalid or missing confirmation token"],
            reason="apply requires explicit confirmation token",
        )

    try:
        if updated_cleaning_report is None or cleaning_report_abs is None:
            raise OSError("updated cleaning report missing")
        _write_json(cleaning_report_abs, updated_cleaning_report)
        matrix_abs.unlink()
    except OSError as exc:
        return MatrixCoverageCompactionItem(
            group_id=group_id,
            matrix_coverage_path=matrix_path,
            status="blocked_apply_failed",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=cleaning_report_path or None,
            factor_readiness_report_path=factor_report_path or None,
            summary_count=summary_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=[str(exc)],
            reason="failed while writing cleaning report or deleting matrix sidecar",
        )

    return MatrixCoverageCompactionItem(
        group_id=group_id,
        matrix_coverage_path=matrix_path,
        status="compacted",
        action="compact",
        reclaimable_bytes=reclaimable_bytes,
        cleaning_report_path=cleaning_report_path or None,
        factor_readiness_report_path=factor_report_path or None,
        summary_count=summary_count,
        references_rewritten=references_rewritten,
        ignored_external_reference_count=ignored_external_reference_count,
        errors=[],
        reason="matrix coverage sidecar removed; coverage remains in factor readiness report",
    )


def build_matrix_coverage_compaction_report(
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
                matrix_path=matrix_path,
                external_reference_map=reference_map,
                apply=apply,
                confirm_token=confirm_token,
            )
        )
        for group_id, matrix_path in candidates
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
            "ignored_external_reference_count": sum(
                int(item["ignored_external_reference_count"])
                for item in items
            ),
            "status_summary": status_summary,
        },
        "items": items,
    }


def render_matrix_coverage_compaction_markdown(
    report: dict[str, Any],
    *,
    max_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> str:
    summary = report.get("summary", {})
    items = report.get("items", [])
    visible_items = items[:max_items]
    lines = [
        "# Matrix Coverage Sidecar Compaction Report",
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
        "| Group | Status | Matrix Coverage | Reclaim Bytes | Summaries | Rewrites |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in visible_items:
        lines.append(
            "| {group} | {status} | `{path}` | {bytes} | {summaries} | {rewrites} |".format(
                group=item.get("group_id", ""),
                status=item.get("status", ""),
                path=item.get("matrix_coverage_path", ""),
                bytes=item.get("reclaimable_bytes", 0),
                summaries=item.get("summary_count", 0),
                rewrites=item.get("references_rewritten", 0),
            )
        )
    if len(items) > len(visible_items):
        lines.append("")
        lines.append(
            f"_Item table truncated to {len(visible_items)} of {len(items)} rows._"
        )
    return "\n".join(lines) + "\n"


def write_matrix_coverage_compaction_report(
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
    report = build_matrix_coverage_compaction_report(
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
        / f"matrix_coverage_compaction_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "matrix_coverage_compaction.json"
    md_path = out_dir / "matrix_coverage_compaction.md"
    _write_json(json_path, report)
    md_path.write_text(
        render_matrix_coverage_compaction_markdown(
            report,
            max_items=max_markdown_items,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compact redundant matrix-coverage sidecars after proving the "
            "same coverage is embedded in factor readiness reports."
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
    paths = write_matrix_coverage_compaction_report(
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
    print(f"matrix coverage compaction mode: {mode}")
    print(f"workspace root: {payload.get('root', '')}")
    print(f"candidates: {summary.get('candidate_count', 0)}")
    print(f"would compact: {summary.get('would_compact_count', 0)}")
    print(f"compacted: {summary.get('compacted_count', 0)}")
    print(f"blocked: {summary.get('blocked_count', 0)}")
    print("matrix coverage compaction manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
