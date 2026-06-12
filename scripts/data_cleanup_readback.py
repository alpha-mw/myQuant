"""Run bounded hash readback for no-delete data cleanup candidates."""

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

SCHEMA_VERSION = "myquant.data_cleanup_readback.v1"
DEFAULT_MAX_CANDIDATES = 500
DEFAULT_MAX_MARKDOWN_CANDIDATES = 200


@dataclass(frozen=True)
class FileReadback:
    relative_path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    error: str | None = None


@dataclass(frozen=True)
class CandidateReadback:
    group_id: str
    candidate_type: str
    delete_allowed: bool
    readback_status: str
    reclaimable_bytes: int
    candidate_paths: list[str]
    retained_paths: list[str]
    candidate_files: list[dict[str, Any]]
    retained_files: list[dict[str, Any]]
    passed_checks: list[str]
    failed_checks: list[str]
    blockers: list[str]
    pending_checks: list[str]


def _latest_gate_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/"
            "data_cleanup_gate_*/data_cleanup_gate.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_cleanup_gate.json found under reports/project_cleanup"
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


def _selected_candidates(
    gate_report: dict[str, Any],
    *,
    candidate_type: str | None,
    max_candidates: int | None,
) -> tuple[list[dict[str, Any]], int]:
    clear_candidates = [
        candidate
        for candidate in gate_report.get("candidates", [])
        if candidate.get("gate_status") == "clear_but_delete_disabled"
    ]
    if candidate_type:
        clear_candidates = [
            candidate
            for candidate in clear_candidates
            if candidate.get("candidate_type") == candidate_type
        ]

    if max_candidates is None:
        return clear_candidates, 0

    return clear_candidates[:max_candidates], max(
        0,
        len(clear_candidates) - max_candidates,
    )


def _candidate_readback(
    repo_root: Path,
    candidate: dict[str, Any],
) -> CandidateReadback:
    candidate_paths = [
        str(path) for path in candidate.get("candidate_paths", [])
    ]
    retained_paths = [
        str(path) for path in candidate.get("retained_paths", [])
    ]
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
    ]

    if all(file["exists"] for file in candidate_files):
        passed_checks.append("candidate_files_exist")
    else:
        failed_checks.append("candidate_files_exist")
        blockers.append("candidate_file_missing")

    if all(file["exists"] for file in retained_files):
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
    if candidate_sizes and all(
        size in retained_sizes for size in candidate_sizes
    ):
        passed_checks.append("candidate_size_matches_retained")
    else:
        failed_checks.append("candidate_size_matches_retained")
        blockers.append("size_mismatch_or_unreadable")

    readback_status = (
        "hash_readback_passed" if not failed_checks else "blocked"
    )

    return CandidateReadback(
        group_id=str(candidate.get("group_id", "")),
        candidate_type=str(candidate.get("candidate_type", "")),
        delete_allowed=False,
        readback_status=readback_status,
        reclaimable_bytes=int(candidate.get("reclaimable_bytes") or 0),
        candidate_paths=candidate_paths,
        retained_paths=retained_paths,
        candidate_files=candidate_files,
        retained_files=retained_files,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        blockers=blockers,
        pending_checks=pending_checks,
    )


def build_data_cleanup_readback_report(
    gate_report: dict[str, Any],
    *,
    repo_root: Path,
    gate_json_path: Path | None = None,
    candidate_type: str | None = None,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Build a no-delete hash readback report for gate-clear candidates."""
    selected, skipped_by_limit = _selected_candidates(
        gate_report,
        candidate_type=candidate_type,
        max_candidates=max_candidates,
    )
    results = [
        _candidate_readback(repo_root, candidate) for candidate in selected
    ]
    results_payload = [asdict(item) for item in results]

    status_summary: dict[str, int] = {}
    blocker_summary: dict[str, int] = {}
    for item in results_payload:
        status = str(item["readback_status"])
        status_summary[status] = status_summary.get(status, 0) + 1
        for blocker in item["blockers"]:
            blocker_summary[blocker] = blocker_summary.get(blocker, 0) + 1

    passed_results = [
        item
        for item in results_payload
        if item["readback_status"] == "hash_readback_passed"
    ]
    blocked_results = [
        item
        for item in results_payload
        if item["readback_status"] == "blocked"
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_gate_schema": gate_report.get("schema_version", ""),
        "source_gate_generated_at": gate_report.get("generated_at", ""),
        "source_gate_json": str(gate_json_path) if gate_json_path else None,
        "root": str(repo_root),
        "candidate_type_filter": candidate_type,
        "max_candidates": max_candidates,
        "delete_candidate_count": 0,
        "summary": {
            "reviewed_candidate_count": len(results_payload),
            "skipped_by_limit_count": skipped_by_limit,
            "hash_readback_passed_count": len(passed_results),
            "blocked_count": len(blocked_results),
            "verified_reclaim_bytes": sum(
                int(item["reclaimable_bytes"]) for item in passed_results
            ),
            "status_summary": status_summary,
            "blocker_summary": blocker_summary,
        },
        "candidates": results_payload,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_data_cleanup_readback_markdown(
    report: dict[str, Any],
    *,
    max_candidates: int = DEFAULT_MAX_MARKDOWN_CANDIDATES,
) -> str:
    summary = report.get("summary", {})
    candidates = report.get("candidates", [])
    visible_candidates = candidates[:max_candidates]
    lines = [
        "# Data Cleanup Readback Report",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Source gate: `{report.get('source_gate_json', '')}`",
        f"- Candidate type filter: `{report.get('candidate_type_filter')}`",
        f"- Max candidates: `{report.get('max_candidates')}`",
        f"- Delete candidates: {report.get('delete_candidate_count', 0)}",
        f"- Reviewed candidates: {summary.get('reviewed_candidate_count', 0)}",
        f"- Skipped by limit: {summary.get('skipped_by_limit_count', 0)}",
        (
            "- Hash readback passed: "
            f"{summary.get('hash_readback_passed_count', 0)}"
        ),
        f"- Blocked: {summary.get('blocked_count', 0)}",
        (
            "- Verified reclaim bytes: "
            f"{summary.get('verified_reclaim_bytes', 0)}"
        ),
        "",
        "## Status Summary",
        "",
        "| Status | Groups |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary.get("status_summary", {}).items()):
        lines.append(f"| {_markdown_cell(status)} | {count} |")

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            (
                "| Group | Type | Status | Reclaim Bytes | Delete | "
                "First Candidate |"
            ),
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for candidate in visible_candidates:
        first_path = candidate.get("candidate_paths", [""])[0]
        row_template = (
            "| {group} | {candidate_type} | {status} | {reclaim} | "
            "{delete} | `{path}` |"
        )
        lines.append(
            row_template.format(
                group=_markdown_cell(candidate.get("group_id", "")),
                candidate_type=_markdown_cell(
                    candidate.get("candidate_type", "")
                ),
                status=_markdown_cell(candidate.get("readback_status", "")),
                reclaim=candidate.get("reclaimable_bytes", 0),
                delete="yes" if candidate.get("delete_allowed") else "no",
                path=_markdown_cell(first_path),
            )
        )
    if len(candidates) > len(visible_candidates):
        lines.extend(
            [
                "",
                (
                    f"_Candidate table truncated to {len(visible_candidates)} "
                    f"of {len(candidates)} rows._"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def write_data_cleanup_readback_report(
    gate_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    candidate_type: str | None = None,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    max_markdown_candidates: int = DEFAULT_MAX_MARKDOWN_CANDIDATES,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    gate_path = gate_json_path.resolve()
    gate_report = _load_json(gate_path)
    report = build_data_cleanup_readback_report(
        gate_report,
        repo_root=repo_root,
        gate_json_path=gate_path,
        candidate_type=candidate_type,
        max_candidates=max_candidates,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_readback_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_readback.json"
    md_path = out_dir / "data_cleanup_readback.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_data_cleanup_readback_markdown(
            report,
            max_candidates=max_markdown_candidates,
        ),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "md": str(md_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-delete hash readback for cleanup candidates."
    )
    parser.add_argument(
        "--gate-json",
        type=Path,
        default=None,
        help="Path to data_cleanup_gate.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--candidate-type",
        default=None,
        help=(
            "Optional candidate_type filter, such as "
            "quarantine_restore_mirror."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Maximum candidates to hash. Use -1 for all clear candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown readback reports.",
    )
    parser.add_argument(
        "--max-markdown-candidates",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_CANDIDATES,
        help="Maximum candidate rows included in Markdown.",
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
    max_candidates = (
        None if args.max_candidates < 0 else max(0, args.max_candidates)
    )
    try:
        gate_json = args.gate_json or _latest_gate_json(repo_root)
        paths = write_data_cleanup_readback_report(
            gate_json,
            root=repo_root,
            output_dir=args.output_dir,
            candidate_type=args.candidate_type,
            max_candidates=max_candidates,
            max_markdown_candidates=max(0, args.max_markdown_candidates),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    print("data cleanup readback mode: dry-run")
    print(f"workspace root: {payload['root']}")
    print(f"reviewed candidates: {summary['reviewed_candidate_count']}")
    print(f"hash readback passed: {summary['hash_readback_passed_count']}")
    print(f"blocked: {summary['blocked_count']}")
    print(f"delete candidates: {payload['delete_candidate_count']}")
    print("data cleanup readback manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
