"""Compact empty Tushare cell-flag CSV artifacts with manifest rewrites."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import get_repo_root

SCHEMA_VERSION = "myquant.empty_cell_flags_compaction.v1"
CONFIRM_TOKEN = "COMPACT_EMPTY_CELL_FLAGS"
ORPHAN_DELETE_CONFIRM_TOKEN = "DELETE_ORPHAN_EMPTY_CELL_FLAGS"
DEFAULT_MAX_MARKDOWN_ITEMS = 500
REFERENCE_SCAN_SUFFIXES = {".json", ".md", ".txt"}
REFERENCE_SCAN_ROOTS = (
    ("data", "cleaning_reports", "tushare"),
    ("data", "factor_readiness", "tushare"),
    ("reports",),
    ("results", "strategy_records"),
)


@dataclass(frozen=True)
class CellFlagsCompactionItem:
    cell_flags_path: str
    status: str
    action: str
    reclaimable_bytes: int
    cleaning_report_path: str | None
    factor_readiness_report_path: str | None
    references_rewritten: int
    errors: list[str]
    reason: str


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _is_empty_cell_flags_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if not path.name.endswith("_cell_flags.csv"):
        return False
    if path.stat().st_size > 1:
        return False
    try:
        return path.read_text(encoding="utf-8").strip() == ""
    except UnicodeDecodeError:
        return False


def _cleaning_report_for_cell_flags(path: Path) -> Path:
    stem = path.name.removesuffix("_cell_flags.csv")
    return path.with_name(f"{stem}_cleaning_report.json")


def _set_sparse_metadata(payload: dict[str, Any], planned_path: str) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata
    metadata["cell_flags_empty"] = True
    metadata["cell_flags_path_suppressed"] = True
    metadata["cell_flags_planned_path"] = planned_path


def _rewrite_cell_flags_references(
    payload: Any,
    *,
    old_path: str,
) -> int:
    rewritten = 0
    if isinstance(payload, dict):
        if payload.get("cell_flags_path") == old_path:
            payload["cell_flags_path"] = None
            _set_sparse_metadata(payload, old_path)
            rewritten += 1
        for value in payload.values():
            rewritten += _rewrite_cell_flags_references(
                value,
                old_path=old_path,
            )
    elif isinstance(payload, list):
        for value in payload:
            rewritten += _rewrite_cell_flags_references(
                value,
                old_path=old_path,
            )
    return rewritten


def _factor_readiness_report_path(
    repo_root: Path,
    cleaning_report: dict[str, Any],
) -> Path | None:
    metadata = cleaning_report.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("factor_readiness_report_path")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path


def _json_contains_reference(payload: dict[str, Any] | None, reference: str) -> bool:
    if payload is None:
        return False
    return reference in json.dumps(payload, ensure_ascii=False)


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
) -> tuple[dict[str, list[str]], dict[str, int]]:
    reference_map = {reference: [] for reference in references}
    if not references:
        return reference_map, {
            "external_reference_scan_file_count": 0,
            "external_reference_skipped_project_cleanup_count": 0,
            "external_reference_read_error_count": 0,
            "external_reference_match_count": 0,
        }
    files, skipped_project_cleanup = _iter_external_reference_files(repo_root)
    read_error_count = 0
    match_count = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            read_error_count += 1
            continue
        relative_path = _relative_path(repo_root, path)
        for reference in references:
            if reference in text:
                reference_map[reference].append(relative_path)
                match_count += 1
    return reference_map, {
        "external_reference_scan_file_count": len(files),
        "external_reference_skipped_project_cleanup_count": skipped_project_cleanup,
        "external_reference_read_error_count": read_error_count,
        "external_reference_match_count": match_count,
    }


def _compaction_item(
    repo_root: Path,
    cell_flags_path: Path,
    *,
    apply: bool,
    confirm_token: str | None,
    allow_orphan_delete: bool,
    orphan_reference_map: dict[str, list[str]],
) -> CellFlagsCompactionItem:
    relative_cell_path = _relative_path(repo_root, cell_flags_path)
    reclaimable_bytes = cell_flags_path.stat().st_size
    cleaning_report_path = _cleaning_report_for_cell_flags(cell_flags_path)
    relative_cleaning_report = (
        _relative_path(repo_root, cleaning_report_path)
        if cleaning_report_path.exists()
        else None
    )
    errors: list[str] = []

    cleaning_report = _load_json(cleaning_report_path)
    if not cleaning_report:
        if allow_orphan_delete:
            external_references = orphan_reference_map.get(relative_cell_path, [])
            if external_references:
                errors.append(
                    "external references found: "
                    + ", ".join(external_references[:5])
                )
                return CellFlagsCompactionItem(
                    cell_flags_path=relative_cell_path,
                    status="blocked_orphan_referenced",
                    action="blocked",
                    reclaimable_bytes=0,
                    cleaning_report_path=relative_cleaning_report,
                    factor_readiness_report_path=None,
                    references_rewritten=0,
                    errors=errors,
                    reason=(
                        "orphan cell flags file is not deleted while external "
                        "references remain"
                    ),
                )
            if not apply:
                return CellFlagsCompactionItem(
                    cell_flags_path=relative_cell_path,
                    status="would_delete_orphan",
                    action="dry_run_delete_orphan",
                    reclaimable_bytes=reclaimable_bytes,
                    cleaning_report_path=relative_cleaning_report,
                    factor_readiness_report_path=None,
                    references_rewritten=0,
                    errors=[],
                    reason=(
                        "unreferenced empty orphan cell flags file; pass "
                        "--apply with orphan deletion confirmation token"
                    ),
                )
            if confirm_token != ORPHAN_DELETE_CONFIRM_TOKEN:
                return CellFlagsCompactionItem(
                    cell_flags_path=relative_cell_path,
                    status="blocked_orphan_confirm_token_required",
                    action="blocked",
                    reclaimable_bytes=0,
                    cleaning_report_path=relative_cleaning_report,
                    factor_readiness_report_path=None,
                    references_rewritten=0,
                    errors=["invalid or missing orphan deletion confirmation token"],
                    reason="orphan deletion requires explicit confirmation token",
                )
            try:
                cell_flags_path.unlink()
            except OSError as exc:
                return CellFlagsCompactionItem(
                    cell_flags_path=relative_cell_path,
                    status="blocked_orphan_delete_failed",
                    action="blocked",
                    reclaimable_bytes=0,
                    cleaning_report_path=relative_cleaning_report,
                    factor_readiness_report_path=None,
                    references_rewritten=0,
                    errors=[str(exc)],
                    reason="failed while deleting orphan cell flags file",
                )
            return CellFlagsCompactionItem(
                cell_flags_path=relative_cell_path,
                status="orphan_deleted",
                action="delete_orphan",
                reclaimable_bytes=reclaimable_bytes,
                cleaning_report_path=relative_cleaning_report,
                factor_readiness_report_path=None,
                references_rewritten=0,
                errors=[],
                reason=(
                    "empty orphan cell flags file removed after external "
                    "reference scan found no references"
                ),
            )
        errors.append(f"cleaning report missing or invalid: {relative_cleaning_report}")
        return CellFlagsCompactionItem(
            cell_flags_path=relative_cell_path,
            status="blocked_missing_cleaning_report",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=relative_cleaning_report,
            factor_readiness_report_path=None,
            references_rewritten=0,
            errors=errors,
            reason="cell flags file is not compacted without its cleaning report",
        )

    factor_report_path = _factor_readiness_report_path(repo_root, cleaning_report)
    factor_report = _load_json(factor_report_path) if factor_report_path else None
    relative_factor_report = (
        _relative_path(repo_root, factor_report_path)
        if factor_report_path and factor_report_path.exists()
        else None
    )

    updated_cleaning_report = json.loads(json.dumps(cleaning_report))
    cleaning_rewrites = _rewrite_cell_flags_references(
        updated_cleaning_report,
        old_path=relative_cell_path,
    )
    if cleaning_rewrites <= 0:
        errors.append("cleaning report does not reference the cell flags path")

    updated_factor_report: dict[str, Any] | None = None
    factor_rewrites = 0
    if factor_report is not None:
        updated_factor_report = json.loads(json.dumps(factor_report))
        factor_rewrites = _rewrite_cell_flags_references(
            updated_factor_report,
            old_path=relative_cell_path,
        )
        if (
            factor_rewrites <= 0
            and _json_contains_reference(factor_report, relative_cell_path)
        ):
            errors.append("factor readiness report still references the cell flags path")

    references_rewritten = cleaning_rewrites + factor_rewrites
    if errors:
        return CellFlagsCompactionItem(
            cell_flags_path=relative_cell_path,
            status="blocked_reference_rewrite_failed",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=relative_cleaning_report,
            factor_readiness_report_path=relative_factor_report,
            references_rewritten=references_rewritten,
            errors=errors,
            reason="manifest references could not be safely rewritten",
        )

    if not apply:
        return CellFlagsCompactionItem(
            cell_flags_path=relative_cell_path,
            status="would_compact",
            action="dry_run_compact",
            reclaimable_bytes=reclaimable_bytes,
            cleaning_report_path=relative_cleaning_report,
            factor_readiness_report_path=relative_factor_report,
            references_rewritten=references_rewritten,
            errors=[],
            reason="dry-run only; pass --apply with confirmation token",
        )

    if confirm_token != CONFIRM_TOKEN:
        return CellFlagsCompactionItem(
            cell_flags_path=relative_cell_path,
            status="blocked_confirm_token_required",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=relative_cleaning_report,
            factor_readiness_report_path=relative_factor_report,
            references_rewritten=references_rewritten,
            errors=["invalid or missing confirmation token"],
            reason="apply requires explicit confirmation token",
        )

    try:
        _write_json(cleaning_report_path, updated_cleaning_report)
        if factor_report_path is not None and updated_factor_report is not None:
            _write_json(factor_report_path, updated_factor_report)
        cell_flags_path.unlink()
    except OSError as exc:
        return CellFlagsCompactionItem(
            cell_flags_path=relative_cell_path,
            status="blocked_apply_failed",
            action="blocked",
            reclaimable_bytes=0,
            cleaning_report_path=relative_cleaning_report,
            factor_readiness_report_path=relative_factor_report,
            references_rewritten=references_rewritten,
            errors=[str(exc)],
            reason="failed while writing manifests or deleting cell flags",
        )

    return CellFlagsCompactionItem(
        cell_flags_path=relative_cell_path,
        status="compacted",
        action="compact",
        reclaimable_bytes=reclaimable_bytes,
        cleaning_report_path=relative_cleaning_report,
        factor_readiness_report_path=relative_factor_report,
        references_rewritten=references_rewritten,
        errors=[],
        reason="empty cell flags file removed after manifest references were suppressed",
    )


def build_empty_cell_flags_compaction_report(
    *,
    repo_root: Path,
    apply: bool = False,
    confirm_token: str | None = None,
    max_items: int | None = None,
    allow_orphan_delete: bool = False,
) -> dict[str, Any]:
    root = repo_root / "data" / "cleaning_reports" / "tushare"
    candidates = [
        path
        for path in sorted(root.rglob("*_cell_flags.csv"))
        if _is_empty_cell_flags_file(path)
    ]
    skipped_by_limit = 0
    if max_items is not None:
        skipped_by_limit = max(0, len(candidates) - max_items)
        candidates = candidates[:max_items]
    candidate_references = [_relative_path(repo_root, path) for path in candidates]
    orphan_reference_map, reference_scan_summary = (
        _scan_external_references(repo_root, candidate_references)
        if allow_orphan_delete
        else (
            {reference: [] for reference in candidate_references},
            {
                "external_reference_scan_file_count": 0,
                "external_reference_skipped_project_cleanup_count": 0,
                "external_reference_read_error_count": 0,
                "external_reference_match_count": 0,
            },
        )
    )

    items = [
        asdict(
            _compaction_item(
                repo_root,
                path,
                apply=apply,
                confirm_token=confirm_token,
                allow_orphan_delete=allow_orphan_delete,
                orphan_reference_map=orphan_reference_map,
            )
        )
        for path in candidates
    ]
    status_summary: dict[str, int] = {}
    for item in items:
        status = str(item["status"])
        status_summary[status] = status_summary.get(status, 0) + 1
    blocked_count = sum(
        count
        for status, count in status_summary.items()
        if status.startswith("blocked")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "root": str(repo_root),
        "apply_requested": apply,
        "allow_orphan_delete": allow_orphan_delete,
        "confirm_token_valid": confirm_token == CONFIRM_TOKEN
        or (
            allow_orphan_delete
            and confirm_token == ORPHAN_DELETE_CONFIRM_TOKEN
        ),
        "execution_performed": apply
        and (
            status_summary.get("compacted", 0) > 0
            or status_summary.get("orphan_deleted", 0) > 0
        ),
        "summary": {
            "candidate_count": len(items),
            "skipped_by_limit_count": skipped_by_limit,
            "would_compact_count": status_summary.get("would_compact", 0),
            "compacted_count": status_summary.get("compacted", 0),
            "would_delete_orphan_count": status_summary.get(
                "would_delete_orphan",
                0,
            ),
            "orphan_deleted_count": status_summary.get("orphan_deleted", 0),
            "blocked_count": blocked_count,
            "planned_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in items
                if item["status"] in {"would_compact", "would_delete_orphan"}
            ),
            "compacted_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in items
                if item["status"] == "compacted"
            ),
            "orphan_deleted_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in items
                if item["status"] == "orphan_deleted"
            ),
            "references_rewritten_count": sum(
                int(item["references_rewritten"])
                for item in items
                if item["status"] in {"would_compact", "compacted"}
            ),
            "status_summary": status_summary,
            **reference_scan_summary,
        },
        "items": items,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_empty_cell_flags_compaction_markdown(
    report: dict[str, Any],
    *,
    max_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> str:
    summary = report.get("summary", {})
    items = report.get("items", [])[:max_items]
    lines = [
        "# Empty Cell Flags Compaction Report",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Apply requested: `{report.get('apply_requested', False)}`",
        f"- Execution performed: `{report.get('execution_performed', False)}`",
        f"- Candidates: `{summary.get('candidate_count', 0)}`",
        f"- Would compact: `{summary.get('would_compact_count', 0)}`",
        f"- Compacted: `{summary.get('compacted_count', 0)}`",
        f"- Would delete orphan: `{summary.get('would_delete_orphan_count', 0)}`",
        f"- Orphan deleted: `{summary.get('orphan_deleted_count', 0)}`",
        f"- Blocked: `{summary.get('blocked_count', 0)}`",
        f"- Planned reclaim bytes: `{summary.get('planned_reclaim_bytes', 0)}`",
        f"- Compacted reclaim bytes: `{summary.get('compacted_reclaim_bytes', 0)}`",
        (
            "- Orphan deleted reclaim bytes: "
            f"`{summary.get('orphan_deleted_reclaim_bytes', 0)}`"
        ),
        "",
        "## Items",
        "",
        "| Cell Flags | Status | Reclaim Bytes | Cleaning Report |",
        "| --- | --- | ---: | --- |",
    ]
    for item in items:
        lines.append(
            "| {path} | {status} | {bytes} | {report} |".format(
                path=_markdown_cell(item.get("cell_flags_path", "")),
                status=_markdown_cell(item.get("status", "")),
                bytes=item.get("reclaimable_bytes", 0),
                report=_markdown_cell(item.get("cleaning_report_path", "")),
            )
        )
    if len(report.get("items", [])) > len(items):
        lines.append("")
        lines.append(
            f"_Item table truncated to {len(items)} of {len(report.get('items', []))} rows._"
        )
    return "\n".join(lines) + "\n"


def write_empty_cell_flags_compaction_report(
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
    apply: bool = False,
    confirm_token: str | None = None,
    max_items: int | None = None,
    allow_orphan_delete: bool = False,
    max_markdown_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    report = build_empty_cell_flags_compaction_report(
        repo_root=repo_root,
        apply=apply,
        confirm_token=confirm_token,
        max_items=max_items,
        allow_orphan_delete=allow_orphan_delete,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_dir or (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"empty_cell_flags_compaction_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "empty_cell_flags_compaction.json"
    md_path = out_dir / "empty_cell_flags_compaction.md"
    _write_json(json_path, report)
    md_path.write_text(
        render_empty_cell_flags_compaction_markdown(
            report,
            max_items=max_markdown_items,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or compact empty Tushare cell_flags CSV artifacts after "
            "rewriting JSON manifest references."
        )
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-token",
        default=None,
        help=(
            f"Required token for --apply: {CONFIRM_TOKEN}; use "
            f"{ORPHAN_DELETE_CONFIRM_TOKEN} with --allow-orphan-delete"
        ),
    )
    parser.add_argument(
        "--allow-orphan-delete",
        action="store_true",
        help=(
            "Allow deletion of empty cell_flags files that have no cleaning "
            "report and no external references. Requires a separate token."
        ),
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-markdown-items", type=int, default=DEFAULT_MAX_MARKDOWN_ITEMS)
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    max_items = args.max_items
    if max_items is not None and max_items < 0:
        max_items = None
    paths = write_empty_cell_flags_compaction_report(
        root=args.root,
        output_dir=args.output_dir,
        apply=args.apply,
        confirm_token=args.confirm_token,
        max_items=max_items,
        allow_orphan_delete=args.allow_orphan_delete,
        max_markdown_items=max(0, args.max_markdown_items),
    )
    payload = _load_json(Path(paths["json"])) or {}
    summary = payload.get("summary") or {}
    mode = "apply" if args.apply else "dry-run"
    print(f"empty cell flags compaction mode: {mode}")
    print(f"workspace root: {payload.get('root')}")
    print(f"candidates: {summary.get('candidate_count', 0)}")
    print(f"would compact: {summary.get('would_compact_count', 0)}")
    print(f"compacted: {summary.get('compacted_count', 0)}")
    print(f"would delete orphan: {summary.get('would_delete_orphan_count', 0)}")
    print(f"orphan deleted: {summary.get('orphan_deleted_count', 0)}")
    print(f"blocked: {summary.get('blocked_count', 0)}")
    print("empty cell flags compaction manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
