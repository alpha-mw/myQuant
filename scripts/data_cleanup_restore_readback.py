"""Run retained-copy readback for restore-source cleanup readiness groups."""

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

SCHEMA_VERSION = "myquant.data_cleanup_restore_readback.v1"
DEFAULT_READINESS_CLASS = "review_retained_copy_readback_required"
DEFAULT_MAX_GROUPS = 500
DEFAULT_MAX_MARKDOWN_GROUPS = 200


@dataclass(frozen=True)
class FileReadback:
    relative_path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    error: str | None = None


@dataclass(frozen=True)
class RestoreGroupReadback:
    group_id: str
    candidate_type: str
    policy_class: str
    risk_level: str
    readiness_class: str
    readback_status: str
    delete_allowed: bool
    reclaimable_bytes: int
    candidate_paths: list[str]
    retained_paths: list[str]
    candidate_files: list[dict[str, Any]]
    retained_files: list[dict[str, Any]]
    passed_checks: list[str]
    failed_checks: list[str]
    blockers: list[str]
    pending_checks: list[str]


def _latest_readiness_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/"
            "data_cleanup_restore_readiness_*/data_cleanup_restore_readiness.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_cleanup_restore_readiness.json found under "
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


def _file_readback(repo_root: Path, relative_path: str) -> FileReadback:
    path = repo_root / relative_path
    if not path.exists():
        return FileReadback(
            relative_path=relative_path,
            exists=False,
            size_bytes=None,
            sha256=None,
            error="missing",
        )
    try:
        size_bytes = path.stat().st_size
        digest = _hash_file(path)
    except OSError as exc:
        return FileReadback(
            relative_path=relative_path,
            exists=True,
            size_bytes=None,
            sha256=None,
            error=str(exc),
        )
    return FileReadback(
        relative_path=relative_path,
        exists=True,
        size_bytes=size_bytes,
        sha256=digest,
    )


def _selected_groups(
    readiness: dict[str, Any],
    *,
    readiness_class: str,
    max_groups: int | None,
) -> tuple[list[dict[str, Any]], int, int]:
    groups = [
        group
        for group in readiness.get("groups", [])
        if isinstance(group, dict)
    ]
    selected = [
        group
        for group in groups
        if str(group.get("readiness_class", "")) == readiness_class
    ]
    skipped_by_filter = len(groups) - len(selected)
    if max_groups is None:
        return selected, skipped_by_filter, 0
    return (
        selected[:max_groups],
        skipped_by_filter,
        max(0, len(selected) - max_groups),
    )


def _group_readback(
    repo_root: Path,
    group: dict[str, Any],
) -> RestoreGroupReadback:
    candidate_paths = [str(path) for path in group.get("candidate_paths", [])]
    retained_paths = [str(path) for path in group.get("retained_paths", [])]
    candidate_files = [
        asdict(_file_readback(repo_root, path)) for path in candidate_paths
    ]
    retained_files = [
        asdict(_file_readback(repo_root, path)) for path in retained_paths
    ]

    passed_checks: list[str] = []
    failed_checks: list[str] = []
    blockers = ["delete_disabled_by_policy"]
    pending_checks = [
        "manual_delete_approval_required",
        "pre_delete_storage_validate_required",
        "pre_delete_storage_validate_clean_required",
        "pre_delete_storage_diff_required",
    ]

    if candidate_files and all(file["exists"] for file in candidate_files):
        passed_checks.append("candidate_files_exist")
    else:
        failed_checks.append("candidate_files_exist")
        blockers.append("candidate_file_missing")

    if retained_files and all(file["exists"] for file in retained_files):
        passed_checks.append("retained_files_exist")
    else:
        failed_checks.append("retained_files_exist")
        blockers.append("retained_file_missing")

    retained_hashes = {
        str(file["sha256"])
        for file in retained_files
        if file.get("sha256")
    }
    candidate_hashes = [
        str(file["sha256"])
        for file in candidate_files
        if file.get("sha256")
    ]
    if candidate_hashes and all(
        digest in retained_hashes for digest in candidate_hashes
    ):
        passed_checks.append("candidate_hash_matches_retained")
    else:
        failed_checks.append("candidate_hash_matches_retained")
        blockers.append("hash_mismatch_or_unreadable")

    retained_sizes = {
        int(file["size_bytes"])
        for file in retained_files
        if file.get("size_bytes") is not None
    }
    candidate_sizes = [
        int(file["size_bytes"])
        for file in candidate_files
        if file.get("size_bytes") is not None
    ]
    if candidate_sizes and all(size in retained_sizes for size in candidate_sizes):
        passed_checks.append("candidate_size_matches_retained")
    else:
        failed_checks.append("candidate_size_matches_retained")
        blockers.append("size_mismatch_or_unreadable")

    readback_status = (
        "retained_copy_readback_passed" if not failed_checks else "blocked"
    )
    return RestoreGroupReadback(
        group_id=str(group.get("group_id", "")),
        candidate_type=str(
            group.get("candidate_type", "restore_source_duplicate_review")
        ),
        policy_class=str(group.get("policy_class", "")),
        risk_level=str(group.get("risk_level", "")),
        readiness_class=str(group.get("readiness_class", "")),
        readback_status=readback_status,
        delete_allowed=False,
        reclaimable_bytes=int(group.get("reclaimable_bytes", 0) or 0),
        candidate_paths=candidate_paths,
        retained_paths=retained_paths,
        candidate_files=candidate_files,
        retained_files=retained_files,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
        pending_checks=pending_checks,
    )


def build_restore_readback_report(
    readiness: dict[str, Any],
    *,
    repo_root: Path,
    readiness_json_path: Path | None = None,
    readiness_class: str = DEFAULT_READINESS_CLASS,
    max_groups: int | None = DEFAULT_MAX_GROUPS,
) -> dict[str, Any]:
    """Build a no-delete retained-copy readback report for readiness groups."""
    selected, skipped_by_filter, skipped_by_limit = _selected_groups(
        readiness,
        readiness_class=readiness_class,
        max_groups=max_groups,
    )
    results = [_group_readback(repo_root, group) for group in selected]
    groups = [asdict(item) for item in results]
    status_summary: dict[str, int] = {}
    blocker_summary: dict[str, int] = {}
    for item in groups:
        status = str(item["readback_status"])
        status_summary[status] = status_summary.get(status, 0) + 1
        for blocker in item["blockers"]:
            blocker_summary[blocker] = blocker_summary.get(blocker, 0) + 1

    passed = [
        item
        for item in groups
        if item["readback_status"] == "retained_copy_readback_passed"
    ]
    blocked = [
        item
        for item in groups
        if item["readback_status"] == "blocked"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_readiness_schema": readiness.get("schema_version", ""),
        "source_readiness_generated_at": readiness.get("generated_at", ""),
        "source_readiness_json": (
            str(readiness_json_path) if readiness_json_path else None
        ),
        "root": str(repo_root),
        "readiness_class_filter": readiness_class,
        "max_groups": max_groups,
        "delete_candidate_count": 0,
        "summary": {
            "reviewed_group_count": len(groups),
            "skipped_by_filter_count": skipped_by_filter,
            "skipped_by_limit_count": skipped_by_limit,
            "retained_copy_readback_passed_count": len(passed),
            "blocked_count": len(blocked),
            "verified_reclaim_bytes": sum(
                int(item["reclaimable_bytes"]) for item in passed
            ),
            "status_summary": dict(sorted(status_summary.items())),
            "blocker_summary": dict(sorted(blocker_summary.items())),
        },
        "groups": groups,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_restore_readback_markdown(
    report: dict[str, Any],
    *,
    max_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> str:
    summary = report.get("summary", {})
    groups = report.get("groups", [])
    visible_groups = groups[:max_groups]
    lines = [
        "# Data Cleanup Restore-Source Readback",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Source readiness: `{report.get('source_readiness_json', '')}`",
        f"- Readiness class filter: `{report.get('readiness_class_filter')}`",
        f"- Max groups: `{report.get('max_groups')}`",
        f"- Delete candidates: {report.get('delete_candidate_count', 0)}",
        f"- Reviewed groups: {summary.get('reviewed_group_count', 0)}",
        f"- Skipped by filter: {summary.get('skipped_by_filter_count', 0)}",
        f"- Skipped by limit: {summary.get('skipped_by_limit_count', 0)}",
        (
            "- Retained-copy readback passed: "
            f"{summary.get('retained_copy_readback_passed_count', 0)}"
        ),
        f"- Blocked: {summary.get('blocked_count', 0)}",
        (
            "- Verified reclaim bytes: "
            f"{summary.get('verified_reclaim_bytes', 0)}"
        ),
        "",
        "## Groups",
        "",
        "| Group | Policy Class | Status | Reclaim Bytes | First Candidate |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for group in visible_groups:
        candidate_paths = group.get("candidate_paths", [])
        first_path = candidate_paths[0] if candidate_paths else ""
        lines.append(
            "| {group_id} | {policy_class} | {status} | {reclaim} | `{path}` |".format(
                group_id=_markdown_cell(group.get("group_id", "")),
                policy_class=_markdown_cell(group.get("policy_class", "")),
                status=_markdown_cell(group.get("readback_status", "")),
                reclaim=group.get("reclaimable_bytes", 0),
                path=_markdown_cell(first_path),
            )
        )
    if len(groups) > len(visible_groups):
        lines.extend(
            [
                "",
                (
                    f"_Group table truncated to {len(visible_groups)} "
                    f"of {len(groups)} rows._"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def write_restore_readback_report(
    readiness_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    readiness_class: str = DEFAULT_READINESS_CLASS,
    max_groups: int | None = DEFAULT_MAX_GROUPS,
    max_markdown_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    readiness_path = readiness_json_path.resolve()
    readiness = _load_json(readiness_path)
    report = build_restore_readback_report(
        readiness,
        repo_root=repo_root,
        readiness_json_path=readiness_path,
        readiness_class=readiness_class,
        max_groups=max_groups,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_restore_readback_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_restore_readback.json"
    md_path = out_dir / "data_cleanup_restore_readback.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_restore_readback_markdown(
            report,
            max_groups=max_markdown_groups,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-delete retained-copy readback for restore-source groups."
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        default=None,
        help="Path to data_cleanup_restore_readiness.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--readiness-class",
        default=DEFAULT_READINESS_CLASS,
        help="Readiness class to hash-check.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=DEFAULT_MAX_GROUPS,
        help="Maximum groups to hash. Use -1 for all matching groups.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown readback reports.",
    )
    parser.add_argument(
        "--max-markdown-groups",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_GROUPS,
        help="Maximum group rows included in Markdown.",
    )
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = get_repo_root(args.root)
    max_groups = None if args.max_groups < 0 else max(0, args.max_groups)
    try:
        readiness_json = args.readiness_json or _latest_readiness_json(repo_root)
        paths = write_restore_readback_report(
            readiness_json,
            root=repo_root,
            output_dir=args.output_dir,
            readiness_class=args.readiness_class,
            max_groups=max_groups,
            max_markdown_groups=max(0, args.max_markdown_groups),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    print("data cleanup restore-source readback mode: dry-run")
    print(f"workspace root: {payload['root']}")
    print(f"reviewed groups: {summary['reviewed_group_count']}")
    print(
        "retained-copy readback passed: "
        f"{summary['retained_copy_readback_passed_count']}"
    )
    print(f"blocked: {summary['blocked_count']}")
    print(f"delete candidates: {payload['delete_candidate_count']}")
    print("data cleanup restore-source readback manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
