"""Compact uniform Tushare row-flag CSV artifacts with manifest rewrites."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import get_repo_root

SCHEMA_VERSION = "myquant.uniform_row_flags_compaction.v1"
CONFIRM_TOKEN = "COMPACT_UNIFORM_ROW_FLAGS"
DEFAULT_MAX_MARKDOWN_ITEMS = 500
DEFAULT_MAX_FILE_BYTES = 128 * 1024 * 1024
REFERENCE_SCAN_SUFFIXES = {".json", ".md", ".txt"}
REFERENCE_SCAN_ROOTS = (
    ("data", "cleaning_reports", "tushare"),
    ("data", "factor_readiness", "tushare"),
    ("reports",),
    ("results", "strategy_records"),
)
ROW_FLAGS_REFERENCE_RE = re.compile(
    r"data/cleaning_reports/tushare/[A-Za-z0-9_./=-]+_row_flags\.csv"
)
IGNORED_EXTERNAL_REFERENCE_PREFIXES = (
    "reports/storage/csv_inventory_",
)


@dataclass(frozen=True)
class RowFlagsCompactionItem:
    group_id: str
    row_flags_path: str
    status: str
    action: str
    reclaimable_bytes: int
    cleaning_report_path: str | None
    row_count: int
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


def _parse_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "True":
        return True
    if text == "False":
        return False
    if text == "":
        return ""
    try:
        return int(text)
    except ValueError:
        return text


def _uniform_row_flags_metadata(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            if "row_index" not in columns:
                return None, "row_index column missing"
            uniform_values: dict[str, Any] = {}
            row_count = 0
            for row_count, row in enumerate(reader, start=1):
                try:
                    row_index = int(str(row.get("row_index", "")).strip())
                except ValueError:
                    return None, "row_index is not an integer"
                if row_index != row_count - 1:
                    return None, "row_index is not contiguous from zero"
                for column in columns:
                    if column == "row_index":
                        continue
                    value = _parse_csv_value(str(row.get(column, "")))
                    if column not in uniform_values:
                        uniform_values[column] = value
                    elif uniform_values[column] != value:
                        return None, f"column is not uniform: {column}"
            if row_count <= 0:
                return None, "row flags file has no data rows"
            if not uniform_values:
                return None, "no compactable flag columns"
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return None, f"row flags read failed: {exc}"
    return (
        {
            "row_flags_compaction_schema": SCHEMA_VERSION,
            "row_flags_row_count": row_count,
            "row_flags_columns": columns,
            "row_flags_uniform_values": uniform_values,
        },
        None,
    )


def _cleaning_report_for_row_flags(path: Path) -> Path:
    stem = path.name.removesuffix("_row_flags.csv")
    return path.with_name(f"{stem}_cleaning_report.json")


def _set_compacted_metadata(
    payload: dict[str, Any],
    *,
    planned_path: str,
    metadata: dict[str, Any],
) -> None:
    payload["row_flags_path"] = None
    existing = payload.get("metadata")
    if not isinstance(existing, dict):
        existing = {}
        payload["metadata"] = existing
    existing.update(
        {
            "row_flags_compacted": True,
            "row_flags_path_suppressed": True,
            "row_flags_planned_path": planned_path,
            **metadata,
        }
    )


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
    read_error_count = 0
    skipped_large_count = 0
    match_count = 0
    reference_set = set(references)
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
        if "_row_flags.csv" not in text:
            continue
        relative_path = _relative_path(repo_root, path)
        for reference in set(ROW_FLAGS_REFERENCE_RE.findall(text)):
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
    return any(
        path.startswith(prefix)
        for prefix in IGNORED_EXTERNAL_REFERENCE_PREFIXES
    )


def _group_maps(
    policy: dict[str, Any],
    reference_audit: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    policy_groups = {
        str(group.get("group_id", "")): group
        for group in policy.get("groups", [])
        if isinstance(group, dict)
    }
    reference_groups = {
        str(group.get("group_id", "")): group
        for group in reference_audit.get("groups", [])
        if isinstance(group, dict)
    }
    return policy_groups, reference_groups


def _candidate_items(
    policy: dict[str, Any],
    reference_audit: dict[str, Any],
) -> list[tuple[str, str]]:
    policy_groups, reference_groups = _group_maps(policy, reference_audit)
    items: list[tuple[str, str]] = []
    for group_id, group in policy_groups.items():
        reference_group = reference_groups.get(group_id)
        if not reference_group:
            continue
        if group.get("policy_class") != "cross_symbol_generated_artifact_duplicate":
            continue
        if group.get("risk_level") != "high":
            continue
        if "cleaning_row_flags" not in group.get("path_roles", []):
            continue
        referenced = {
            str(path)
            for path in reference_group.get("referenced_candidate_paths", [])
        }
        for candidate_path in group.get("candidate_paths", []):
            candidate_text = str(candidate_path)
            if not candidate_text.endswith("_row_flags.csv"):
                continue
            if candidate_text not in referenced:
                continue
            items.append((group_id, candidate_text))
    return items


def _compaction_item(
    repo_root: Path,
    *,
    group_id: str,
    row_flags_path: str,
    external_reference_map: dict[str, list[str]],
    apply: bool,
    confirm_token: str | None,
) -> RowFlagsCompactionItem:
    row_flags_abs = repo_root / row_flags_path
    cleaning_report_abs = _cleaning_report_for_row_flags(row_flags_abs)
    relative_cleaning_report = (
        _relative_path(repo_root, cleaning_report_abs)
        if cleaning_report_abs.exists()
        else None
    )
    errors: list[str] = []
    references_rewritten = 0
    row_count = 0
    reclaimable_bytes = 0

    if not row_flags_abs.exists():
        errors.append("row flags file missing")
    elif not row_flags_abs.is_file():
        errors.append("row flags path is not a file")
    else:
        reclaimable_bytes = row_flags_abs.stat().st_size

    metadata: dict[str, Any] | None = None
    if not errors:
        metadata, metadata_error = _uniform_row_flags_metadata(row_flags_abs)
        if metadata_error:
            errors.append(metadata_error)
        elif metadata is not None:
            row_count = int(metadata.get("row_flags_row_count", 0) or 0)

    cleaning_report = _load_json(cleaning_report_abs)
    if not cleaning_report:
        errors.append("cleaning report missing or invalid")
    elif cleaning_report.get("row_flags_path") != row_flags_path:
        errors.append("cleaning report does not reference row flags path")

    expected_references = {relative_cleaning_report} if relative_cleaning_report else set()
    unexpected_references = [
        reference_path
        for reference_path in external_reference_map.get(row_flags_path, [])
        if reference_path not in expected_references
        and not _is_ignored_external_reference(reference_path)
    ]
    ignored_external_reference_count = sum(
        1
        for reference_path in external_reference_map.get(row_flags_path, [])
        if reference_path not in expected_references
        and _is_ignored_external_reference(reference_path)
    )
    if unexpected_references:
        errors.append(
            "unexpected references: "
            + ", ".join(sorted(set(unexpected_references))[:5])
        )

    updated_cleaning_report: dict[str, Any] | None = None
    if cleaning_report and metadata:
        updated_cleaning_report = json.loads(json.dumps(cleaning_report))
        _set_compacted_metadata(
            updated_cleaning_report,
            planned_path=row_flags_path,
            metadata=metadata,
        )
        references_rewritten = 1

    if errors:
        return RowFlagsCompactionItem(
            group_id=group_id,
            row_flags_path=row_flags_path,
            status="blocked",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=relative_cleaning_report,
            row_count=row_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=errors,
            reason="row flags compaction preconditions failed",
        )

    if not apply:
        return RowFlagsCompactionItem(
            group_id=group_id,
            row_flags_path=row_flags_path,
            status="would_compact",
            action="dry_run_compact",
            reclaimable_bytes=reclaimable_bytes,
            cleaning_report_path=relative_cleaning_report,
            row_count=row_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=[],
            reason="dry-run only; pass --apply with confirmation token",
        )

    if confirm_token != CONFIRM_TOKEN:
        return RowFlagsCompactionItem(
            group_id=group_id,
            row_flags_path=row_flags_path,
            status="blocked_confirm_token_required",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=relative_cleaning_report,
            row_count=row_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=["invalid or missing confirmation token"],
            reason="apply requires explicit confirmation token",
        )

    try:
        if updated_cleaning_report is None:
            raise OSError("updated cleaning report missing")
        _write_json(cleaning_report_abs, updated_cleaning_report)
        row_flags_abs.unlink()
    except OSError as exc:
        return RowFlagsCompactionItem(
            group_id=group_id,
            row_flags_path=row_flags_path,
            status="blocked_apply_failed",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=relative_cleaning_report,
            row_count=row_count,
            references_rewritten=references_rewritten,
            ignored_external_reference_count=ignored_external_reference_count,
            errors=[str(exc)],
            reason="failed while writing cleaning report or deleting row flags",
        )

    return RowFlagsCompactionItem(
        group_id=group_id,
        row_flags_path=row_flags_path,
        status="compacted",
        action="compact",
        reclaimable_bytes=reclaimable_bytes,
        cleaning_report_path=relative_cleaning_report,
        row_count=row_count,
        references_rewritten=references_rewritten,
        ignored_external_reference_count=ignored_external_reference_count,
        errors=[],
        reason="uniform row flags removed after cleaning report metadata rewrite",
    )


def build_uniform_row_flags_compaction_report(
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
                row_flags_path=row_flags_path,
                external_reference_map=reference_map,
                apply=apply,
                confirm_token=confirm_token,
            )
        )
        for group_id, row_flags_path in candidates
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


def render_uniform_row_flags_compaction_markdown(
    report: dict[str, Any],
    *,
    max_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> str:
    summary = report.get("summary", {})
    items = report.get("items", [])
    visible_items = items[:max_items]
    lines = [
        "# Uniform Row Flags Compaction Report",
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
        "| Group | Status | Row Flags | Reclaim Bytes | Rows | Rewrites |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in visible_items:
        lines.append(
            "| {group} | {status} | `{path}` | {bytes} | {rows} | {rewrites} |".format(
                group=item.get("group_id", ""),
                status=item.get("status", ""),
                path=item.get("row_flags_path", ""),
                bytes=item.get("reclaimable_bytes", 0),
                rows=item.get("row_count", 0),
                rewrites=item.get("references_rewritten", 0),
            )
        )
    if len(items) > len(visible_items):
        lines.append("")
        lines.append(
            f"_Item table truncated to {len(visible_items)} of {len(items)} rows._"
        )
    return "\n".join(lines) + "\n"


def write_uniform_row_flags_compaction_report(
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
    report = build_uniform_row_flags_compaction_report(
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
        / f"uniform_row_flags_compaction_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "uniform_row_flags_compaction.json"
    md_path = out_dir / "uniform_row_flags_compaction.md"
    _write_json(json_path, report)
    md_path.write_text(
        render_uniform_row_flags_compaction_markdown(
            report,
            max_items=max_markdown_items,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compact referenced cross-symbol uniform row-flag CSV artifacts "
            "by rewriting owner cleaning-report metadata."
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
    paths = write_uniform_row_flags_compaction_report(
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
    print(f"uniform row flags compaction mode: {mode}")
    print(f"workspace root: {payload.get('root', '')}")
    print(f"candidates: {summary.get('candidate_count', 0)}")
    print(f"would compact: {summary.get('would_compact_count', 0)}")
    print(f"compacted: {summary.get('compacted_count', 0)}")
    print(f"blocked: {summary.get('blocked_count', 0)}")
    print("uniform row flags compaction manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
