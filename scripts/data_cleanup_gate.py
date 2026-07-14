"""Validate no-delete gates for data cleanup plan candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import get_repo_root
from scripts.intelligence_retirement_evidence import (
    UnsafeRepositoryPath,
    is_protected_retirement_evidence_path,
    resolve_repo_relative_path,
)

SCHEMA_VERSION = "myquant.data_cleanup_gate.v1"
DEFAULT_MAX_MARKDOWN_CANDIDATES = 500
DEFAULT_MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024

PROTECTED_RUNTIME_PREFIXES = (
    "data/parquet/cn/_latest.json",
    "data/parquet/cn/_catalog.json",
    "data/parquet/cn/bars/",
    "data/parquet_serving/cn/bars/",
)

REFERENCE_PATH_PATTERN = re.compile(
    r"(?:"
    r"reports/storage/csv_quarantine/[^\s\"'`),\]}]+"
    r"|data/raw_backups/tushare/[^\s\"'`),\]}]+"
    r"|data/factor_readiness/tushare/[^\s\"'`),\]}]+"
    r"|data/cleaning_reports/tushare/[^\s\"'`),\]}]+"
    r"|data/cn_market_full/\.cache/[^\s\"'`),\]}]+"
    r")"
)


@dataclass(frozen=True)
class ReferenceScanResult:
    scanned_file_count: int
    skipped_file_count: int
    total_bytes: int
    references_by_path: dict[str, list[str]]


@dataclass(frozen=True)
class CandidateGateResult:
    group_id: str
    candidate_type: str
    delete_allowed: bool
    gate_status: str
    reclaimable_bytes: int
    candidate_paths: list[str]
    retained_paths: list[str]
    blockers: list[str]
    passed_checks: list[str]
    failed_checks: list[str]
    pending_checks: list[str]
    runtime_references: dict[str, list[str]]
    strategy_references: dict[str, list[str]]


def _latest_cleanup_plan_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/"
            "data_cleanup_plan_*/data_cleanup_plan.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_cleanup_plan.json found under reports/project_cleanup"
        )
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_path_set(plan: dict[str, Any], repo_root: Path) -> set[str]:
    paths: set[str] = set()
    for candidate in plan.get("candidates", []):
        for path in candidate.get("candidate_paths", []):
            try:
                _resolved, canonical = resolve_repo_relative_path(repo_root, str(path))
            except UnsafeRepositoryPath:
                continue
            paths.add(canonical)
        for path in candidate.get("retained_paths", []):
            try:
                _resolved, canonical = resolve_repo_relative_path(repo_root, str(path))
            except UnsafeRepositoryPath:
                continue
            paths.add(canonical)
    return paths


def _extract_candidate_references(
    text: str,
    candidate_paths: set[str],
) -> set[str]:
    found: set[str] = set()
    for match in REFERENCE_PATH_PATTERN.finditer(text):
        value = match.group(0).rstrip(".;:")
        if value in candidate_paths:
            found.add(value)
    return found


def _read_text_if_small(path: Path, max_file_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_file_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _runtime_reference_files(repo_root: Path) -> list[Path]:
    paths = [
        repo_root / "data" / "parquet" / "cn" / "_latest.json",
        repo_root / "data" / "parquet" / "cn" / "_catalog.json",
    ]
    latest_path = paths[0]
    if latest_path.exists():
        try:
            latest = _load_json(latest_path)
        except (OSError, json.JSONDecodeError):
            latest = {}
        for key in ("manifest_path", "clean_manifest_path"):
            value = latest.get(key)
            if value:
                paths.append(repo_root / str(value))
    unique_paths: dict[str, Path] = {}
    for path in paths:
        if path.exists():
            unique_paths[str(path.resolve())] = path
    return list(unique_paths.values())


def _scan_reference_files(
    repo_root: Path,
    files: Iterable[Path],
    candidate_paths: set[str],
    *,
    max_file_bytes: int = DEFAULT_MAX_TEXT_FILE_BYTES,
) -> ReferenceScanResult:
    references_by_path: dict[str, list[str]] = {}
    scanned_file_count = 0
    skipped_file_count = 0
    total_bytes = 0

    for path in files:
        text = _read_text_if_small(path, max_file_bytes)
        if text is None:
            skipped_file_count += 1
            continue
        scanned_file_count += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
        try:
            source = path.relative_to(repo_root).as_posix()
        except ValueError:
            source = str(path)
        for referenced_path in _extract_candidate_references(
            text,
            candidate_paths,
        ):
            references_by_path.setdefault(referenced_path, []).append(source)

    return ReferenceScanResult(
        scanned_file_count=scanned_file_count,
        skipped_file_count=skipped_file_count,
        total_bytes=total_bytes,
        references_by_path=references_by_path,
    )


def _strategy_record_files(repo_root: Path) -> list[Path]:
    root = repo_root / "results" / "strategy_records"
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )


def _is_protected_runtime_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(prefix)
        for prefix in PROTECTED_RUNTIME_PREFIXES
    )


def _paths_exist(repo_root: Path, paths: list[str]) -> bool:
    try:
        resolved = [resolve_repo_relative_path(repo_root, path)[0] for path in paths]
    except UnsafeRepositoryPath:
        return False
    return all(path.exists() for path in resolved)


def _references_for_paths(
    paths: list[str],
    references_by_path: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        path: references_by_path[path]
        for path in paths
        if path in references_by_path
    }


def _candidate_gate_result(
    repo_root: Path,
    candidate: dict[str, Any],
    *,
    runtime_references_by_path: dict[str, list[str]],
    strategy_references_by_path: dict[str, list[str]],
) -> CandidateGateResult:
    candidate_paths = [
        str(path) for path in candidate.get("candidate_paths", [])
    ]
    retained_paths = [
        str(path) for path in candidate.get("retained_paths", [])
    ]
    canonical_candidate_paths: list[str] = []
    unsafe_candidate_paths: list[str] = []
    for path in candidate_paths:
        try:
            _resolved, canonical = resolve_repo_relative_path(repo_root, path)
        except UnsafeRepositoryPath:
            unsafe_candidate_paths.append(path)
        else:
            canonical_candidate_paths.append(canonical)
    canonical_retained_paths: list[str] = []
    unsafe_retained_paths: list[str] = []
    for path in retained_paths:
        try:
            _resolved, canonical = resolve_repo_relative_path(repo_root, path)
        except UnsafeRepositoryPath:
            unsafe_retained_paths.append(path)
        else:
            canonical_retained_paths.append(canonical)
    blockers = ["delete_disabled_by_policy"]
    passed_checks: list[str] = []
    failed_checks: list[str] = []
    pending_checks = ["restore_hash_readback_check"]

    if unsafe_candidate_paths:
        blockers.append("candidate_path_outside_repository")
        failed_checks.append("candidate_path_containment_check")
    else:
        passed_checks.append("candidate_path_containment_check")

    if unsafe_retained_paths:
        blockers.append("retained_path_outside_repository")
        failed_checks.append("retained_path_containment_check")
    else:
        passed_checks.append("retained_path_containment_check")

    if any(_is_protected_runtime_path(path) for path in canonical_candidate_paths):
        blockers.append("candidate_is_active_runtime_path")
        failed_checks.append("active_runtime_path_check")
    else:
        passed_checks.append("active_runtime_path_check")

    if any(
        is_protected_retirement_evidence_path(path)
        for path in canonical_candidate_paths
    ):
        blockers.append("candidate_is_protected_retirement_evidence")
        failed_checks.append("retirement_evidence_protection_check")
    else:
        passed_checks.append("retirement_evidence_protection_check")

    if _paths_exist(repo_root, candidate_paths):
        passed_checks.append("candidate_path_exists")
    else:
        blockers.append("candidate_path_missing")
        failed_checks.append("candidate_path_exists")

    if _paths_exist(repo_root, retained_paths):
        passed_checks.append("retained_path_exists")
    else:
        blockers.append("retained_path_missing")
        failed_checks.append("retained_path_exists")

    runtime_refs = _references_for_paths(
        canonical_candidate_paths,
        runtime_references_by_path,
    )
    if runtime_refs:
        blockers.append("runtime_reference_present")
        failed_checks.append("runtime_reference_check")
    else:
        passed_checks.append("runtime_reference_check")

    strategy_refs = _references_for_paths(
        canonical_candidate_paths,
        strategy_references_by_path,
    )
    if strategy_refs:
        blockers.append("strategy_record_reference_present")
        failed_checks.append("strategy_record_reference_check")
    else:
        passed_checks.append("strategy_record_reference_check")

    gate_status = "blocked" if failed_checks else "clear_but_delete_disabled"

    return CandidateGateResult(
        group_id=str(candidate.get("group_id", "")),
        candidate_type=str(candidate.get("candidate_type", "")),
        delete_allowed=False,
        gate_status=gate_status,
        reclaimable_bytes=int(candidate.get("reclaimable_bytes") or 0),
        candidate_paths=candidate_paths,
        retained_paths=retained_paths,
        blockers=blockers,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        pending_checks=pending_checks,
        runtime_references=runtime_refs,
        strategy_references=strategy_refs,
    )


def build_data_cleanup_gate_report(
    plan: dict[str, Any],
    *,
    repo_root: Path,
    plan_json_path: Path | None = None,
    max_text_file_bytes: int = DEFAULT_MAX_TEXT_FILE_BYTES,
) -> dict[str, Any]:
    """Build a no-delete gate report for cleanup plan candidates."""
    candidate_paths = _candidate_path_set(plan, repo_root)
    runtime_scan = _scan_reference_files(
        repo_root,
        _runtime_reference_files(repo_root),
        candidate_paths,
        max_file_bytes=max_text_file_bytes,
    )
    strategy_scan = _scan_reference_files(
        repo_root,
        _strategy_record_files(repo_root),
        candidate_paths,
        max_file_bytes=max_text_file_bytes,
    )

    gate_results = [
        _candidate_gate_result(
            repo_root,
            candidate,
            runtime_references_by_path=runtime_scan.references_by_path,
            strategy_references_by_path=strategy_scan.references_by_path,
        )
        for candidate in plan.get("candidates", [])
    ]
    results_payload = [asdict(item) for item in gate_results]

    status_summary: dict[str, int] = {}
    blocker_summary: dict[str, int] = {}
    runtime_candidate_reference_paths: set[str] = set()
    strategy_candidate_reference_paths: set[str] = set()
    for item in results_payload:
        status = str(item["gate_status"])
        status_summary[status] = status_summary.get(status, 0) + 1
        for blocker in item["blockers"]:
            blocker_summary[blocker] = blocker_summary.get(blocker, 0) + 1
        runtime_candidate_reference_paths.update(item["runtime_references"])
        strategy_candidate_reference_paths.update(item["strategy_references"])

    clear_results = [
        item
        for item in results_payload
        if item["gate_status"] == "clear_but_delete_disabled"
    ]
    blocked_results = [
        item for item in results_payload if item["gate_status"] == "blocked"
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_plan_schema": plan.get("schema_version", ""),
        "source_plan_generated_at": plan.get("generated_at", ""),
        "source_plan_json": str(plan_json_path) if plan_json_path else None,
        "root": str(repo_root),
        "delete_candidate_count": 0,
        "summary": {
            "candidate_group_count": len(results_payload),
            "clear_but_delete_disabled_count": len(clear_results),
            "blocked_count": len(blocked_results),
            "potential_reclaim_bytes_clear": sum(
                int(item["reclaimable_bytes"]) for item in clear_results
            ),
            "status_summary": status_summary,
            "blocker_summary": blocker_summary,
            "runtime_plan_path_reference_count": len(
                runtime_scan.references_by_path
            ),
            "strategy_plan_path_reference_count": len(
                strategy_scan.references_by_path
            ),
            "runtime_candidate_reference_count": len(
                runtime_candidate_reference_paths
            ),
            "strategy_candidate_reference_count": len(
                strategy_candidate_reference_paths
            ),
        },
        "reference_scans": {
            "runtime": asdict(runtime_scan),
            "strategy_records": asdict(strategy_scan),
        },
        "candidates": results_payload,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_data_cleanup_gate_markdown(
    report: dict[str, Any],
    *,
    max_candidates: int = DEFAULT_MAX_MARKDOWN_CANDIDATES,
) -> str:
    summary = report.get("summary", {})
    candidates = report.get("candidates", [])
    visible_candidates = candidates[:max_candidates]
    lines = [
        "# Data Cleanup Gate Report",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Source plan: `{report.get('source_plan_json', '')}`",
        f"- Delete candidates: {report.get('delete_candidate_count', 0)}",
        f"- Candidate groups: {summary.get('candidate_group_count', 0)}",
        (
            "- Clear but delete disabled: "
            f"{summary.get('clear_but_delete_disabled_count', 0)}"
        ),
        f"- Blocked: {summary.get('blocked_count', 0)}",
        (
            "- Potential reclaim bytes clear: "
            f"{summary.get('potential_reclaim_bytes_clear', 0)}"
        ),
        "",
        "## Reference Scans",
        "",
        (
            "| Scope | Scanned Files | Skipped Files | "
            "Plan Path Refs | Candidate Path Refs |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    scans = report.get("reference_scans", {})
    for scope, scan in scans.items():
        prefix = "runtime" if scope == "runtime" else "strategy"
        row_template = (
            "| {scope} | {scanned} | {skipped} | {plan_refs} | "
            "{candidate_refs} |"
        )
        lines.append(
            row_template.format(
                scope=_markdown_cell(scope),
                scanned=scan.get("scanned_file_count", 0),
                skipped=scan.get("skipped_file_count", 0),
                plan_refs=summary.get(
                    f"{prefix}_plan_path_reference_count",
                    0,
                ),
                candidate_refs=summary.get(
                    f"{prefix}_candidate_reference_count",
                    0,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Status Summary",
            "",
            "| Status | Groups |",
            "| --- | ---: |",
        ]
    )
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
                status=_markdown_cell(candidate.get("gate_status", "")),
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


def write_data_cleanup_gate_report(
    plan_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    max_markdown_candidates: int = DEFAULT_MAX_MARKDOWN_CANDIDATES,
    max_text_file_bytes: int = DEFAULT_MAX_TEXT_FILE_BYTES,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    plan_path = plan_json_path.resolve()
    plan = _load_json(plan_path)
    report = build_data_cleanup_gate_report(
        plan,
        repo_root=repo_root,
        plan_json_path=plan_path,
        max_text_file_bytes=max_text_file_bytes,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_gate_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_gate.json"
    md_path = out_dir / "data_cleanup_gate.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_data_cleanup_gate_markdown(
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
        description="Build no-delete gate report for a data cleanup plan."
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="Path to data_cleanup_plan.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown gate reports.",
    )
    parser.add_argument(
        "--max-markdown-candidates",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_CANDIDATES,
        help="Maximum candidate rows included in Markdown.",
    )
    parser.add_argument(
        "--max-text-file-mb",
        type=float,
        default=DEFAULT_MAX_TEXT_FILE_BYTES / 1024 / 1024,
        help="Maximum text file size to scan for path references.",
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
    try:
        plan_json = args.plan_json or _latest_cleanup_plan_json(repo_root)
        paths = write_data_cleanup_gate_report(
            plan_json,
            root=repo_root,
            output_dir=args.output_dir,
            max_markdown_candidates=max(0, args.max_markdown_candidates),
            max_text_file_bytes=max(
                0,
                int(args.max_text_file_mb * 1024 * 1024),
            ),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    print("data cleanup gate mode: dry-run")
    print(f"workspace root: {payload['root']}")
    print(f"candidate groups: {summary['candidate_group_count']}")
    print(
        "clear but delete disabled: "
        f"{summary['clear_but_delete_disabled_count']}"
    )
    print(f"blocked: {summary['blocked_count']}")
    print(f"delete candidates: {payload['delete_candidate_count']}")
    print("data cleanup gate manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
