"""Controlled executor for approved data cleanup whitelist items."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import get_repo_root
from scripts.intelligence_retirement_evidence import (
    UnsafeRepositoryPath,
    is_protected_retirement_evidence_path,
    resolve_repo_relative_path,
)

SCHEMA_VERSION = "myquant.data_cleanup_execute.v1"
CONFIRM_TOKEN = "DELETE_APPROVED_CLEANUP_FILES"
APPROVED_STATUS = "approved_for_delete"
DEFAULT_MAX_MARKDOWN_ITEMS = 500


@dataclass(frozen=True)
class CleanupExecutionItem:
    group_id: str
    status: str
    action: str
    reclaimable_bytes: int
    candidate_paths: list[str]
    retained_paths: list[str]
    deleted_paths: list[str]
    errors: list[str]
    reason: str


def _latest_whitelist_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/"
            "data_cleanup_whitelist_*/data_cleanup_whitelist.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_cleanup_whitelist.json found under "
            "reports/project_cleanup"
        )
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_approved(item: dict[str, Any]) -> bool:
    return (
        item.get("approval_status") == APPROVED_STATUS
        and item.get("delete_allowed") is True
        and item.get("execute_allowed") is True
    )


def _candidate_metadata(item: dict[str, Any]) -> dict[str, tuple[int, str]]:
    paths = [str(path) for path in item.get("candidate_paths", [])]
    sizes = [int(size) for size in item.get("candidate_size_bytes", [])]
    hashes = [str(digest) for digest in item.get("candidate_sha256", [])]
    metadata: dict[str, tuple[int, str]] = {}
    for index, path in enumerate(paths):
        if index >= len(sizes) or index >= len(hashes):
            continue
        metadata[path] = (sizes[index], hashes[index])
    return metadata


def _validate_approved_item(
    item: dict[str, Any],
    candidate_paths: list[tuple[str, Path]],
    retained_paths: list[tuple[str, Path]],
) -> list[str]:
    errors: list[str] = []
    metadata = _candidate_metadata(item)

    for relative_path, path in candidate_paths:
        if not path.exists():
            errors.append(f"candidate missing: {relative_path}")
            continue
        expected = metadata.get(relative_path)
        if expected is None:
            errors.append(f"candidate metadata missing: {relative_path}")
            continue
        expected_size, expected_hash = expected
        try:
            actual_size = path.stat().st_size
            actual_hash = _hash_file(path)
        except OSError as exc:
            errors.append(f"candidate unreadable: {relative_path}: {exc}")
            continue
        if actual_size != expected_size:
            errors.append(f"candidate size mismatch: {relative_path}")
        if actual_hash != expected_hash:
            errors.append(f"candidate hash mismatch: {relative_path}")

    for relative_path, path in retained_paths:
        if not path.exists():
            errors.append(f"retained source missing: {relative_path}")

    return errors


def _execution_item(
    repo_root: Path,
    item: dict[str, Any],
    *,
    apply: bool,
    confirm_token: str | None,
) -> CleanupExecutionItem:
    group_id = str(item.get("group_id", ""))
    candidate_paths = [str(path) for path in item.get("candidate_paths", [])]
    retained_paths = [str(path) for path in item.get("retained_paths", [])]
    reclaimable_bytes = int(item.get("reclaimable_bytes") or 0)

    if not _is_approved(item):
        return CleanupExecutionItem(
            group_id=group_id,
            status="skipped_not_approved",
            action="skip",
            reclaimable_bytes=0,
            candidate_paths=candidate_paths,
            retained_paths=retained_paths,
            deleted_paths=[],
            errors=[],
            reason=(
                "item is not approved_for_delete with "
                "delete/execute allowed"
            ),
        )

    try:
        resolved_candidates = [
            (path, *resolve_repo_relative_path(repo_root, path))
            for path in candidate_paths
        ]
    except UnsafeRepositoryPath as exc:
        return CleanupExecutionItem(
            group_id=group_id,
            status="blocked_unsafe_candidate_path",
            action="blocked",
            reclaimable_bytes=0,
            candidate_paths=candidate_paths,
            retained_paths=retained_paths,
            deleted_paths=[],
            errors=[str(exc)],
            reason="candidate path must stay inside the repository",
        )

    try:
        resolved_retained = [
            (path, *resolve_repo_relative_path(repo_root, path))
            for path in retained_paths
        ]
    except UnsafeRepositoryPath as exc:
        return CleanupExecutionItem(
            group_id=group_id,
            status="blocked_unsafe_retained_path",
            action="blocked",
            reclaimable_bytes=0,
            candidate_paths=candidate_paths,
            retained_paths=retained_paths,
            deleted_paths=[],
            errors=[str(exc)],
            reason="retained path must stay inside the repository",
        )

    protected_paths = [
        original
        for original, _resolved, canonical in resolved_candidates
        if is_protected_retirement_evidence_path(canonical)
    ]
    if protected_paths:
        return CleanupExecutionItem(
            group_id=group_id,
            status="blocked_protected_retirement_evidence",
            action="blocked",
            reclaimable_bytes=0,
            candidate_paths=candidate_paths,
            retained_paths=retained_paths,
            deleted_paths=[],
            errors=[
                f"protected retirement evidence: {path}" for path in protected_paths
            ],
            reason="retired Intelligence evidence is immutable",
        )

    validation_errors = _validate_approved_item(
        item,
        [(original, resolved) for original, resolved, _ in resolved_candidates],
        [(original, resolved) for original, resolved, _ in resolved_retained],
    )
    if validation_errors:
        return CleanupExecutionItem(
            group_id=group_id,
            status="blocked_pre_delete_validation_failed",
            action="blocked",
            reclaimable_bytes=0,
            candidate_paths=candidate_paths,
            retained_paths=retained_paths,
            deleted_paths=[],
            errors=validation_errors,
            reason="pre-delete file validation failed",
        )

    if not apply:
        return CleanupExecutionItem(
            group_id=group_id,
            status="would_delete",
            action="dry_run_delete",
            reclaimable_bytes=reclaimable_bytes,
            candidate_paths=candidate_paths,
            retained_paths=retained_paths,
            deleted_paths=[],
            errors=[],
            reason="dry-run only; pass --apply with confirmation token",
        )

    if confirm_token != CONFIRM_TOKEN:
        return CleanupExecutionItem(
            group_id=group_id,
            status="blocked_confirm_token_required",
            action="blocked",
            reclaimable_bytes=0,
            candidate_paths=candidate_paths,
            retained_paths=retained_paths,
            deleted_paths=[],
            errors=["invalid or missing confirmation token"],
            reason="apply requires explicit confirmation token",
        )

    deleted_paths: list[str] = []
    errors: list[str] = []
    for relative_path, resolved_path, canonical_path in resolved_candidates:
        try:
            current_path, current_canonical = resolve_repo_relative_path(
                repo_root,
                relative_path,
            )
            if current_path != resolved_path or current_canonical != canonical_path:
                raise UnsafeRepositoryPath(
                    f"candidate path changed after validation: {relative_path}"
                )
            current_path.unlink()
            deleted_paths.append(relative_path)
        except (OSError, UnsafeRepositoryPath) as exc:
            errors.append(f"delete failed: {relative_path}: {exc}")

    status = "deleted" if not errors else "blocked_delete_failed"
    return CleanupExecutionItem(
        group_id=group_id,
        status=status,
        action="delete" if not errors else "blocked",
        reclaimable_bytes=reclaimable_bytes if not errors else 0,
        candidate_paths=candidate_paths,
        retained_paths=retained_paths,
        deleted_paths=deleted_paths,
        errors=errors,
        reason="approved files deleted" if not errors else "delete failed",
    )


def build_data_cleanup_execution_report(
    whitelist: dict[str, Any],
    *,
    repo_root: Path,
    whitelist_json_path: Path | None = None,
    apply: bool = False,
    confirm_token: str | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Build a dry-run or token-gated execution report."""
    items = list(whitelist.get("items", []))
    skipped_by_limit = 0
    if max_items is not None:
        skipped_by_limit = max(0, len(items) - max_items)
        items = items[:max_items]

    results = [
        _execution_item(
            repo_root,
            item,
            apply=apply,
            confirm_token=confirm_token,
        )
        for item in items
    ]
    payload = [asdict(item) for item in results]

    status_summary: dict[str, int] = {}
    for item in payload:
        status = str(item["status"])
        status_summary[status] = status_summary.get(status, 0) + 1

    would_delete_count = status_summary.get("would_delete", 0)
    deleted_count = status_summary.get("deleted", 0)
    blocked_count = sum(
        count
        for status, count in status_summary.items()
        if status.startswith("blocked")
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_whitelist_schema": whitelist.get("schema_version", ""),
        "source_whitelist_generated_at": whitelist.get("generated_at", ""),
        "source_whitelist_json": (
            str(whitelist_json_path) if whitelist_json_path else None
        ),
        "root": str(repo_root),
        "apply_requested": apply,
        "confirm_token_valid": confirm_token == CONFIRM_TOKEN,
        "execution_performed": apply and deleted_count > 0,
        "summary": {
            "reviewed_item_count": len(payload),
            "skipped_by_limit_count": skipped_by_limit,
            "skipped_not_approved_count": status_summary.get(
                "skipped_not_approved",
                0,
            ),
            "would_delete_count": would_delete_count,
            "deleted_count": deleted_count,
            "blocked_count": blocked_count,
            "planned_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in payload
                if item["status"] == "would_delete"
            ),
            "deleted_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in payload
                if item["status"] == "deleted"
            ),
            "status_summary": status_summary,
        },
        "items": payload,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_data_cleanup_execution_markdown(
    report: dict[str, Any],
    *,
    max_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> str:
    summary = report.get("summary", {})
    items = report.get("items", [])
    visible_items = items[:max_items]
    lines = [
        "# Data Cleanup Execution Report",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Source whitelist: `{report.get('source_whitelist_json', '')}`",
        f"- Apply requested: {report.get('apply_requested', False)}",
        f"- Execution performed: {report.get('execution_performed', False)}",
        f"- Reviewed items: {summary.get('reviewed_item_count', 0)}",
        (
            "- Skipped not approved: "
            f"{summary.get('skipped_not_approved_count', 0)}"
        ),
        f"- Would delete: {summary.get('would_delete_count', 0)}",
        f"- Deleted: {summary.get('deleted_count', 0)}",
        f"- Blocked: {summary.get('blocked_count', 0)}",
        f"- Planned reclaim bytes: {summary.get('planned_reclaim_bytes', 0)}",
        f"- Deleted reclaim bytes: {summary.get('deleted_reclaim_bytes', 0)}",
        "",
        "## Items",
        "",
        "| Group | Status | Action | Reclaim Bytes | First Candidate |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for item in visible_items:
        first_path = item.get("candidate_paths", [""])[0]
        lines.append(
            "| {group} | {status} | {action} | {reclaim} | `{path}` |".format(
                group=_markdown_cell(item.get("group_id", "")),
                status=_markdown_cell(item.get("status", "")),
                action=_markdown_cell(item.get("action", "")),
                reclaim=item.get("reclaimable_bytes", 0),
                path=_markdown_cell(first_path),
            )
        )
    if len(items) > len(visible_items):
        lines.extend(
            [
                "",
                (
                    f"_Item table truncated to {len(visible_items)} "
                    f"of {len(items)} rows._"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def write_data_cleanup_execution_report(
    whitelist_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    apply: bool = False,
    confirm_token: str | None = None,
    max_items: int | None = None,
    max_markdown_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    whitelist_path = whitelist_json_path.resolve()
    whitelist = _load_json(whitelist_path)
    report = build_data_cleanup_execution_report(
        whitelist,
        repo_root=repo_root,
        whitelist_json_path=whitelist_path,
        apply=apply,
        confirm_token=confirm_token,
        max_items=max_items,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_execute_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_execute.json"
    md_path = out_dir / "data_cleanup_execute.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_data_cleanup_execution_markdown(
            report,
            max_items=max_markdown_items,
        ),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "md": str(md_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or token-gated execution for approved cleanup files."
        )
    )
    parser.add_argument(
        "--whitelist-json",
        type=Path,
        default=None,
        help="Path to data_cleanup_whitelist.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete approved files. Requires --confirm-token.",
    )
    parser.add_argument(
        "--confirm-token",
        default=None,
        help=f"Required token for --apply: {CONFIRM_TOKEN}",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum whitelist items to review. Negative means all items.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown execution reports.",
    )
    parser.add_argument(
        "--max-markdown-items",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_ITEMS,
        help="Maximum item rows included in Markdown.",
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
    max_items = args.max_items
    if max_items is not None and max_items < 0:
        max_items = None
    try:
        whitelist_json = (
            args.whitelist_json or _latest_whitelist_json(repo_root)
        )
        paths = write_data_cleanup_execution_report(
            whitelist_json,
            root=repo_root,
            output_dir=args.output_dir,
            apply=args.apply,
            confirm_token=args.confirm_token,
            max_items=max_items,
            max_markdown_items=max(0, args.max_markdown_items),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    mode = "apply" if args.apply else "dry-run"
    print(f"data cleanup execution mode: {mode}")
    print(f"workspace root: {payload['root']}")
    print(f"reviewed items: {summary['reviewed_item_count']}")
    print(f"skipped not approved: {summary['skipped_not_approved_count']}")
    print(f"would delete: {summary['would_delete_count']}")
    print(f"deleted: {summary['deleted_count']}")
    print(f"blocked: {summary['blocked_count']}")
    print("data cleanup execution manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
