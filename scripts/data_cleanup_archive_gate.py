"""Validate archive-backed deletion readiness for data cleanup roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_cleanup_gate import (  # noqa: E402
    DEFAULT_MAX_TEXT_FILE_BYTES,
    REFERENCE_PATH_PATTERN,
    _runtime_reference_files,
    _strategy_record_files,
)
from scripts.workspace_layout import get_repo_root  # noqa: E402

SCHEMA_VERSION = "myquant.data_cleanup_archive_gate.v1"
DEFAULT_MAX_MARKDOWN_REFERENCES = 100


@dataclass(frozen=True)
class ArchiveGateResult:
    source_root: str
    archive_path: str
    delete_allowed: bool
    gate_status: str
    source_file_count: int
    source_directory_count: int
    source_size_bytes: int
    archive_size_bytes: int
    archive_sha256: str
    archive_member_count: int
    passed_checks: list[str]
    failed_checks: list[str]
    blockers: list[str]
    runtime_references: dict[str, list[str]]
    strategy_references: dict[str, list[str]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_source_files(source_root: Path) -> list[Path]:
    if not source_root.exists():
        return []
    return sorted(path for path in source_root.rglob("*") if path.is_file())


def _iter_source_dirs(source_root: Path) -> list[Path]:
    if not source_root.exists():
        return []
    return sorted(path for path in source_root.rglob("*") if path.is_dir())


def _source_size(files: Iterable[Path]) -> int:
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _tar_member_names(archive_path: Path) -> list[str]:
    with tarfile.open(archive_path, mode="r:*") as archive:
        return [member.name for member in archive.getmembers()]


def _read_text_if_small(path: Path, max_file_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_file_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _source_root_references(
    repo_root: Path,
    files: Iterable[Path],
    *,
    source_root: str,
    max_text_file_bytes: int,
) -> dict[str, list[str]]:
    references: dict[str, list[str]] = {}
    source_prefix = source_root.rstrip("/") + "/"
    for path in files:
        text = _read_text_if_small(path, max_text_file_bytes)
        if text is None:
            continue
        try:
            source_file = path.relative_to(repo_root).as_posix()
        except ValueError:
            source_file = str(path)
        for match in REFERENCE_PATH_PATTERN.finditer(text):
            referenced = match.group(0).rstrip(".;:")
            if referenced == source_root or referenced.startswith(source_prefix):
                references.setdefault(referenced, []).append(source_file)
    return references


def _append_check(
    *,
    passed: bool,
    check_name: str,
    blocker: str | None,
    passed_checks: list[str],
    failed_checks: list[str],
    blockers: list[str],
) -> None:
    if passed:
        passed_checks.append(check_name)
        return
    failed_checks.append(check_name)
    if blocker:
        blockers.append(blocker)


def build_archive_gate_report(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    manifest_json_path: Path | None = None,
    max_text_file_bytes: int = DEFAULT_MAX_TEXT_FILE_BYTES,
) -> dict[str, Any]:
    """Build an archive-backed deletion gate report for one cleanup root."""
    source_root_text = str(manifest.get("source_root", "")).strip().strip("/")
    archive_path_text = str(manifest.get("archive_path", "")).strip()
    archive_path = Path(archive_path_text)
    source_root = repo_root / source_root_text

    passed_checks: list[str] = []
    failed_checks: list[str] = []
    blockers: list[str] = []

    source_files = _iter_source_files(source_root)
    source_dirs = _iter_source_dirs(source_root)
    source_size = _source_size(source_files)

    _append_check(
        passed=bool(source_root_text and source_root.exists() and source_root.is_dir()),
        check_name="source_root_exists",
        blocker="source_root_missing",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )
    _append_check(
        passed=bool(archive_path_text and archive_path.exists() and archive_path.is_file()),
        check_name="archive_exists",
        blocker="archive_missing",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )

    actual_archive_size = 0
    actual_archive_sha = ""
    members: list[str] = []
    if archive_path.exists() and archive_path.is_file():
        try:
            actual_archive_size = archive_path.stat().st_size
            actual_archive_sha = _hash_file(archive_path)
            members = _tar_member_names(archive_path)
        except (OSError, tarfile.TarError) as exc:
            failed_checks.append("archive_readable")
            blockers.append(f"archive_unreadable:{exc}")
        else:
            passed_checks.append("archive_readable")

    expected_source_file_count = int(manifest.get("source_file_count") or 0)
    expected_source_dir_count = int(manifest.get("source_directory_count") or 0)
    expected_source_size = int(manifest.get("source_size_bytes") or 0)
    expected_archive_size = int(manifest.get("archive_size_bytes") or 0)
    expected_archive_sha = str(manifest.get("archive_sha256", "")).strip()
    expected_member_count = int(manifest.get("archive_member_count") or 0)
    archive_prefix = str(
        manifest.get("archive_source_prefix") or f"{source_root_text}/"
    ).strip()

    _append_check(
        passed=len(source_files) == expected_source_file_count,
        check_name="source_file_count_matches_manifest",
        blocker="source_file_count_mismatch",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )
    _append_check(
        passed=len(source_dirs) == expected_source_dir_count,
        check_name="source_directory_count_matches_manifest",
        blocker="source_directory_count_mismatch",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )
    _append_check(
        passed=source_size == expected_source_size,
        check_name="source_size_matches_manifest",
        blocker="source_size_mismatch",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )
    _append_check(
        passed=actual_archive_size == expected_archive_size,
        check_name="archive_size_matches_manifest",
        blocker="archive_size_mismatch",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )
    _append_check(
        passed=bool(actual_archive_sha and actual_archive_sha == expected_archive_sha),
        check_name="archive_sha256_matches_manifest",
        blocker="archive_sha256_mismatch",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )
    _append_check(
        passed=len(members) == expected_member_count,
        check_name="archive_member_count_matches_manifest",
        blocker="archive_member_count_mismatch",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )
    _append_check(
        passed=bool(
            members
            and archive_prefix
            and all(member == source_root_text or member.startswith(archive_prefix) for member in members)
        ),
        check_name="archive_members_stay_under_source_root",
        blocker="archive_member_outside_source_root",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )

    runtime_refs = _source_root_references(
        repo_root,
        _runtime_reference_files(repo_root),
        source_root=source_root_text,
        max_text_file_bytes=max_text_file_bytes,
    )
    strategy_refs = _source_root_references(
        repo_root,
        _strategy_record_files(repo_root),
        source_root=source_root_text,
        max_text_file_bytes=max_text_file_bytes,
    )
    _append_check(
        passed=not runtime_refs,
        check_name="runtime_reference_check",
        blocker="runtime_reference_present",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )
    _append_check(
        passed=not strategy_refs,
        check_name="strategy_record_reference_check",
        blocker="strategy_record_reference_present",
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
    )

    delete_allowed = not failed_checks
    gate_status = "delete_allowed" if delete_allowed else "blocked"
    result = ArchiveGateResult(
        source_root=source_root_text,
        archive_path=archive_path_text,
        delete_allowed=delete_allowed,
        gate_status=gate_status,
        source_file_count=len(source_files),
        source_directory_count=len(source_dirs),
        source_size_bytes=source_size,
        archive_size_bytes=actual_archive_size,
        archive_sha256=actual_archive_sha,
        archive_member_count=len(members),
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
        runtime_references=runtime_refs,
        strategy_references=strategy_refs,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_manifest_schema": manifest.get("schema_version", ""),
        "source_manifest_generated_at": manifest.get("generated_at", ""),
        "source_manifest_json": str(manifest_json_path) if manifest_json_path else None,
        "root": str(repo_root),
        "delete_candidate_count": 1 if delete_allowed else 0,
        "summary": {
            "gate_status": gate_status,
            "delete_allowed": delete_allowed,
            "passed_check_count": len(passed_checks),
            "failed_check_count": len(failed_checks),
            "runtime_reference_count": len(runtime_refs),
            "strategy_reference_count": len(strategy_refs),
            "source_size_bytes": source_size,
            "archive_size_bytes": actual_archive_size,
        },
        "result": asdict(result),
    }


def render_archive_gate_markdown(
    report: dict[str, Any],
    *,
    max_references: int = DEFAULT_MAX_MARKDOWN_REFERENCES,
) -> str:
    summary = report.get("summary", {})
    result = report.get("result", {})
    lines = [
        "# Data Cleanup Archive Gate",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Source manifest: `{report.get('source_manifest_json', '')}`",
        f"- Source root: `{result.get('source_root', '')}`",
        f"- Archive: `{result.get('archive_path', '')}`",
        f"- Gate status: `{summary.get('gate_status', '')}`",
        f"- Delete allowed: {summary.get('delete_allowed', False)}",
        f"- Failed checks: {summary.get('failed_check_count', 0)}",
        f"- Source size bytes: {summary.get('source_size_bytes', 0)}",
        f"- Archive size bytes: {summary.get('archive_size_bytes', 0)}",
        "",
        "## Failed Checks",
        "",
    ]
    failed = result.get("failed_checks", [])
    if failed:
        for item in failed:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Blockers", ""])
    blockers = result.get("blockers", [])
    if blockers:
        for item in blockers:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")

    references = {
        **dict(result.get("runtime_references", {})),
        **dict(result.get("strategy_references", {})),
    }
    lines.extend(["", "## References", ""])
    if references:
        for index, (path, sources) in enumerate(sorted(references.items())):
            if index >= max_references:
                lines.append(f"- Reference list truncated at {max_references} entries")
                break
            lines.append(f"- `{path}` referenced by {', '.join(f'`{item}`' for item in sources)}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_archive_gate_report(
    manifest_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    max_markdown_references: int = DEFAULT_MAX_MARKDOWN_REFERENCES,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    manifest_path = manifest_json_path.resolve()
    manifest = _load_json(manifest_path)
    report = build_archive_gate_report(
        manifest,
        repo_root=repo_root,
        manifest_json_path=manifest_path,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_archive_gate_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_archive_gate.json"
    md_path = out_dir / "data_cleanup_archive_gate.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        render_archive_gate_markdown(
            report,
            max_references=max_markdown_references,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate archive-backed cleanup readiness for a data root."
    )
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-markdown-references",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_REFERENCES,
    )
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = write_archive_gate_report(
        args.manifest_json,
        root=args.root,
        output_dir=args.output_dir,
        max_markdown_references=max(0, args.max_markdown_references),
    )
    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    print("data cleanup archive gate mode: dry-run")
    print(f"workspace root: {payload['root']}")
    print(f"gate status: {summary['gate_status']}")
    print(f"delete allowed: {summary['delete_allowed']}")
    print(f"failed checks: {summary['failed_check_count']}")
    print("data cleanup archive gate manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
