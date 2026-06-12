"""Dry-run inventory for high-risk data cleanup candidates."""

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

DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_FILE_ROWS = 5000

DATA_AUDIT_ROOTS: tuple[tuple[Path, str, str], ...] = (
    (
        Path("data") / "cn_market_full" / ".cache",
        "data_cache_candidate",
        "local market-data cache; audit lineage before deletion",
    ),
    (
        Path("data") / "factor_readiness" / "tushare",
        "duplicate_restore_source",
        "factor readiness lineage; retain unless duplicate is proven",
    ),
    (
        Path("data") / "cleaning_reports" / "tushare",
        "duplicate_restore_source",
        "Tushare cleaning/storage audit lineage; retain unless proven",
    ),
    (
        Path("data") / "raw_backups" / "tushare",
        "duplicate_restore_source",
        "raw restore source; retain unless another restore path is proven",
    ),
    (
        Path("reports") / "storage" / "csv_quarantine",
        "duplicate_restore_source",
        "CSV quarantine mirror; retain unless restore and hash proof exist",
    ),
)


@dataclass(frozen=True)
class DataAuditSourceRoot:
    relative_path: str
    classification: str
    reason: str
    exists: bool
    file_count: int
    size_bytes: int


@dataclass(frozen=True)
class DataAuditFile:
    relative_path: str
    source_root: str
    classification: str
    delete_allowed: bool
    reason: str
    size_bytes: int
    hash_status: str
    sha256: str | None
    duplicate_group_id: str | None
    error: str | None = None


@dataclass(frozen=True)
class DuplicateHashGroup:
    group_id: str
    sha256: str
    size_bytes: int
    file_count: int
    files: list[str]
    classifications: list[str]
    delete_allowed: bool
    reason: str


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        (child for child in path.rglob("*") if child.is_file()),
        key=lambda child: child.as_posix(),
    )


def _source_root_size(files: list[Path]) -> int:
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _audit_file(
    repo_root: Path,
    path: Path,
    *,
    source_root: Path,
    classification: str,
    reason: str,
    max_file_bytes: int,
) -> DataAuditFile:
    relative_path = path.relative_to(repo_root).as_posix()
    source_root_text = source_root.as_posix()

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return DataAuditFile(
            relative_path=relative_path,
            source_root=source_root_text,
            classification=classification,
            delete_allowed=False,
            reason=reason,
            size_bytes=0,
            hash_status="stat_error",
            sha256=None,
            duplicate_group_id=None,
            error=str(exc),
        )

    if size_bytes > max_file_bytes:
        return DataAuditFile(
            relative_path=relative_path,
            source_root=source_root_text,
            classification=classification,
            delete_allowed=False,
            reason=reason,
            size_bytes=size_bytes,
            hash_status="skipped_oversize",
            sha256=None,
            duplicate_group_id=None,
        )

    try:
        digest = _hash_file(path)
    except OSError as exc:
        return DataAuditFile(
            relative_path=relative_path,
            source_root=source_root_text,
            classification=classification,
            delete_allowed=False,
            reason=reason,
            size_bytes=size_bytes,
            hash_status="hash_error",
            sha256=None,
            duplicate_group_id=None,
            error=str(exc),
        )

    return DataAuditFile(
        relative_path=relative_path,
        source_root=source_root_text,
        classification=classification,
        delete_allowed=False,
        reason=reason,
        size_bytes=size_bytes,
        hash_status="hashed",
        sha256=digest,
        duplicate_group_id=None,
    )


def build_data_duplicate_audit(
    root: Path | None = None,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_file_rows: int | None = DEFAULT_MAX_FILE_ROWS,
) -> dict[str, Any]:
    """Build a no-delete duplicate/hash audit for high-risk data roots."""
    repo_root = get_repo_root(root)
    source_roots: list[DataAuditSourceRoot] = []
    audited_files: list[DataAuditFile] = []

    for relative_root, classification, reason in DATA_AUDIT_ROOTS:
        root_path = repo_root / relative_root
        files = _iter_files(root_path)
        source_roots.append(
            DataAuditSourceRoot(
                relative_path=relative_root.as_posix(),
                classification=classification,
                reason=reason,
                exists=root_path.exists(),
                file_count=len(files),
                size_bytes=_source_root_size(files),
            )
        )
        for file_path in files:
            audited_files.append(
                _audit_file(
                    repo_root,
                    file_path,
                    source_root=relative_root,
                    classification=classification,
                    reason=reason,
                    max_file_bytes=max_file_bytes,
                )
            )

    hash_groups: dict[str, list[DataAuditFile]] = {}
    for item in audited_files:
        if item.hash_status == "hashed" and item.sha256:
            hash_groups.setdefault(item.sha256, []).append(item)

    duplicate_groups: list[DuplicateHashGroup] = []
    duplicate_by_path: dict[str, str] = {}
    duplicate_index = 1
    for sha256, group_files in sorted(hash_groups.items()):
        if len(group_files) < 2:
            continue
        sorted_files = sorted(group_files, key=lambda item: item.relative_path)
        group_id = f"dup-{duplicate_index:04d}"
        duplicate_index += 1
        for item in sorted_files:
            duplicate_by_path[item.relative_path] = group_id
        duplicate_groups.append(
            DuplicateHashGroup(
                group_id=group_id,
                sha256=sha256,
                size_bytes=sorted_files[0].size_bytes,
                file_count=len(sorted_files),
                files=[item.relative_path for item in sorted_files],
                classifications=sorted(
                    {item.classification for item in sorted_files}
                ),
                delete_allowed=False,
                reason=(
                    "same sha256 across audited data roots; "
                    "audit-only, no deletion allowed"
                ),
            )
        )

    sorted_audited_files = sorted(
        audited_files,
        key=lambda entry: entry.relative_path,
    )
    all_files_with_groups = [
        asdict(
            DataAuditFile(
                relative_path=item.relative_path,
                source_root=item.source_root,
                classification=item.classification,
                delete_allowed=item.delete_allowed,
                reason=item.reason,
                size_bytes=item.size_bytes,
                hash_status=item.hash_status,
                sha256=item.sha256,
                duplicate_group_id=duplicate_by_path.get(item.relative_path),
                error=item.error,
            )
        )
        for item in sorted_audited_files
    ]
    priority_files = [
        item
        for item in all_files_with_groups
        if item["hash_status"] != "hashed"
        or item["duplicate_group_id"] is not None
        or item["classification"] == "data_cache_candidate"
    ]
    if max_file_rows is None:
        files_with_groups = all_files_with_groups
    else:
        selected_files = priority_files[:max_file_rows]
        selected_paths = {
            str(item["relative_path"]) for item in selected_files
        }
        if len(selected_files) < max_file_rows:
            for item in all_files_with_groups:
                if item["relative_path"] in selected_paths:
                    continue
                selected_files.append(item)
                if len(selected_files) >= max_file_rows:
                    break
        files_with_groups = selected_files
    source_root_payload = [asdict(item) for item in source_roots]
    duplicate_payload = [asdict(item) for item in duplicate_groups]

    classification_summary: dict[str, int] = {}
    hash_status_summary: dict[str, int] = {}
    for item in all_files_with_groups:
        classification = str(item["classification"])
        hash_status = str(item["hash_status"])
        classification_summary[classification] = (
            classification_summary.get(classification, 0) + 1
        )
        hash_status_summary[hash_status] = (
            hash_status_summary.get(hash_status, 0) + 1
        )

    duplicate_file_count = sum(
        group["file_count"] for group in duplicate_payload
    )

    return {
        "schema_version": "myquant.data_duplicate_audit.v1",
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "root": str(repo_root),
        "max_file_bytes": max_file_bytes,
        "delete_candidate_count": 0,
        "summary": {
            "source_root_count": len(source_root_payload),
            "scanned_file_count": len(all_files_with_groups),
            "included_file_count": len(files_with_groups),
            "files_truncated": (
                len(files_with_groups) < len(all_files_with_groups)
            ),
            "hashed_file_count": hash_status_summary.get("hashed", 0),
            "skipped_file_count": len(all_files_with_groups)
            - hash_status_summary.get("hashed", 0),
            "duplicate_group_count": len(duplicate_payload),
            "duplicate_file_count": duplicate_file_count,
            "classification_summary": classification_summary,
            "hash_status_summary": hash_status_summary,
        },
        "source_roots": source_root_payload,
        "duplicate_groups": duplicate_payload,
        "files": files_with_groups,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_data_duplicate_audit_markdown(manifest: dict[str, Any]) -> str:
    """Render a compact Markdown report for the duplicate data audit."""
    summary = manifest.get("summary", {})
    lines = [
        "# Data Duplicate Audit",
        "",
        f"- Schema: `{manifest.get('schema_version', '')}`",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Root: `{manifest.get('root', '')}`",
        f"- Max hashed file size: {manifest.get('max_file_bytes', 0)} bytes",
        f"- Delete candidates: {manifest.get('delete_candidate_count', 0)}",
        f"- Scanned files: {summary.get('scanned_file_count', 0)}",
        f"- File rows included: {summary.get('included_file_count', 0)}",
        f"- File rows truncated: {summary.get('files_truncated', False)}",
        f"- Hashed files: {summary.get('hashed_file_count', 0)}",
        f"- Skipped files: {summary.get('skipped_file_count', 0)}",
        f"- Duplicate groups: {summary.get('duplicate_group_count', 0)}",
        f"- Duplicate files: {summary.get('duplicate_file_count', 0)}",
        "",
        "## Source Roots",
        "",
        "| Classification | Exists | Files | Size Bytes | Path | Reason |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in manifest.get("source_roots", []):
        row_template = (
            "| {classification} | {exists} | {file_count} | "
            "{size_bytes} | `{path}` | {reason} |"
        )
        lines.append(
            row_template.format(
                classification=_markdown_cell(item.get("classification", "")),
                exists="yes" if item.get("exists") else "no",
                file_count=item.get("file_count", 0),
                size_bytes=item.get("size_bytes", 0),
                path=_markdown_cell(item.get("relative_path", "")),
                reason=_markdown_cell(item.get("reason", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Duplicate Hash Groups",
            "",
            "| Group | Files | Size Bytes | Delete | Paths |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for group in manifest.get("duplicate_groups", []):
        paths = "<br>".join(
            f"`{_markdown_cell(path)}`" for path in group.get("files", [])
        )
        row_template = (
            "| {group_id} | {file_count} | {size_bytes} | "
            "{delete_allowed} | {paths} |"
        )
        lines.append(
            row_template.format(
                group_id=_markdown_cell(group.get("group_id", "")),
                file_count=group.get("file_count", 0),
                size_bytes=group.get("size_bytes", 0),
                delete_allowed="yes" if group.get("delete_allowed") else "no",
                paths=paths,
            )
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "| Status | Duplicate Group | Size Bytes | Delete | Path |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for item in manifest.get("files", []):
        row_template = (
            "| {status} | {group_id} | {size_bytes} | "
            "{delete_allowed} | `{path}` |"
        )
        lines.append(
            row_template.format(
                status=_markdown_cell(item.get("hash_status", "")),
                group_id=_markdown_cell(item.get("duplicate_group_id") or ""),
                size_bytes=item.get("size_bytes", 0),
                delete_allowed="yes" if item.get("delete_allowed") else "no",
                path=_markdown_cell(item.get("relative_path", "")),
            )
        )
    return "\n".join(lines) + "\n"


def write_data_duplicate_audit_manifest(
    root: Path | None = None,
    *,
    output_dir: str | Path | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_file_rows: int | None = DEFAULT_MAX_FILE_ROWS,
) -> dict[str, str]:
    """Write JSON and Markdown data duplicate audit reports."""
    repo_root = get_repo_root(root)
    manifest = build_data_duplicate_audit(
        repo_root,
        max_file_bytes=max_file_bytes,
        max_file_rows=max_file_rows,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_duplicate_audit_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_duplicate_audit.json"
    md_path = out_dir / "data_duplicate_audit.md"
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(
        render_data_duplicate_audit_markdown(manifest),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "md": str(md_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dry-run duplicate/hash audit for high-risk data "
            "cleanup roots."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown reports.",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=DEFAULT_MAX_FILE_BYTES / 1024 / 1024,
        help=(
            "Maximum per-file size to hash. Larger files are listed "
            "but skipped."
        ),
    )
    parser.add_argument(
        "--max-file-rows",
        type=int,
        default=DEFAULT_MAX_FILE_ROWS,
        help=(
            "Maximum file detail rows to include in reports. "
            "Use -1 to include all file rows."
        ),
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
    max_file_bytes = max(0, int(args.max_file_mb * 1024 * 1024))
    max_file_rows = (
        None if args.max_file_rows < 0 else max(0, args.max_file_rows)
    )
    paths = write_data_duplicate_audit_manifest(
        args.root,
        output_dir=args.output_dir,
        max_file_bytes=max_file_bytes,
        max_file_rows=max_file_rows,
    )
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    summary = payload["summary"]

    print("data duplicate audit mode: dry-run")
    print(f"workspace root: {payload['root']}")
    print(f"scanned files: {summary['scanned_file_count']}")
    print(f"file rows included: {summary['included_file_count']}")
    print(f"files truncated: {summary['files_truncated']}")
    print(f"hashed files: {summary['hashed_file_count']}")
    print(f"skipped files: {summary['skipped_file_count']}")
    print(f"duplicate groups: {summary['duplicate_group_count']}")
    print(f"delete candidates: {payload['delete_candidate_count']}")
    print("data duplicate audit manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
