"""Build a conservative no-delete plan from a data duplicate audit."""

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

SCHEMA_VERSION = "myquant.data_cleanup_plan.v1"
DEFAULT_MAX_MARKDOWN_CANDIDATES = 500

REQUIRED_VALIDATIONS = (
    "quant-investor market storage-validate --market CN",
    "quant-investor market storage-validate-clean --market CN",
    "quant-investor market storage-diff --market CN",
    "latest pointer and catalog reference check",
    "strategy records reference check",
    "restore path readback check",
)

BASE_BLOCKERS = (
    "delete_disabled_by_policy",
    "storage_validation_required",
    "restore_manifest_required",
    "latest_pointer_reference_check_required",
    "strategy_record_reference_check_required",
)

PATH_KIND_PREFIXES = (
    ("reports/storage/csv_quarantine/", "csv_quarantine"),
    ("data/raw_backups/tushare/", "raw_backup"),
    ("data/factor_readiness/tushare/", "factor_readiness"),
    ("data/cleaning_reports/tushare/", "cleaning_report"),
    ("data/cn_market_full/.cache/", "data_cache"),
)


@dataclass(frozen=True)
class CleanupPlanCandidate:
    group_id: str
    candidate_type: str
    delete_allowed: bool
    status: str
    sha256: str
    size_bytes: int
    reclaimable_bytes: int
    candidate_paths: list[str]
    retained_paths: list[str]
    path_kinds: list[str]
    blockers: list[str]
    required_validations: list[str]
    reason: str


def _path_kind(path: str) -> str:
    for prefix, kind in PATH_KIND_PREFIXES:
        if path.startswith(prefix):
            return kind
    return "other"


def _candidate_type(
    files: list[str],
    quarantine_paths: list[str],
    cache_paths: list[str],
) -> str:
    kinds = {_path_kind(path) for path in files}
    if quarantine_paths and len(quarantine_paths) < len(files):
        return "quarantine_restore_mirror"
    if kinds == {"csv_quarantine"}:
        return "quarantine_internal_duplicate_review"
    if cache_paths:
        return "data_cache_duplicate_review"
    return "restore_source_duplicate_review"


def _candidate_paths_for_group(
    files: list[str],
    candidate_type: str,
    quarantine_paths: list[str],
    cache_paths: list[str],
) -> tuple[list[str], list[str]]:
    if candidate_type == "quarantine_restore_mirror":
        candidate_paths = quarantine_paths
        retained_paths = [
            path for path in files if path not in candidate_paths
        ]
        return candidate_paths, retained_paths

    if (
        candidate_type == "data_cache_duplicate_review"
        and len(cache_paths) < len(files)
    ):
        candidate_paths = cache_paths
        retained_paths = [
            path for path in files if path not in candidate_paths
        ]
        return candidate_paths, retained_paths

    return files[1:], files[:1]


def _reason_for_candidate(candidate_type: str) -> str:
    if candidate_type == "quarantine_restore_mirror":
        return (
            "quarantine file has an identical audited restore copy; "
            "review only until restore validation passes"
        )
    if candidate_type == "quarantine_internal_duplicate_review":
        return "identical files inside quarantine; retain one audited copy"
    if candidate_type == "data_cache_duplicate_review":
        return "data cache duplicate; verify cache regeneration before removal"
    return "duplicate restore/audit source; review retained copy policy first"


def _blockers_for_candidate(candidate_type: str) -> list[str]:
    blockers = list(BASE_BLOCKERS)
    if candidate_type == "quarantine_restore_mirror":
        blockers.append("raw_restore_copy_must_remain_readable")
    elif candidate_type == "data_cache_duplicate_review":
        blockers.append("cache_regeneration_check_required")
    else:
        blockers.append("retained_copy_policy_review_required")
    return blockers


def _candidate_from_group(
    group: dict[str, Any],
) -> CleanupPlanCandidate | None:
    files = sorted(str(path) for path in group.get("files", []))
    if len(files) < 2:
        return None

    quarantine_paths = [
        path for path in files if _path_kind(path) == "csv_quarantine"
    ]
    cache_paths = [path for path in files if _path_kind(path) == "data_cache"]
    candidate_type = _candidate_type(files, quarantine_paths, cache_paths)
    candidate_paths, retained_paths = _candidate_paths_for_group(
        files,
        candidate_type,
        quarantine_paths,
        cache_paths,
    )
    if not candidate_paths or not retained_paths:
        return None

    size_bytes = int(group.get("size_bytes") or 0)
    return CleanupPlanCandidate(
        group_id=str(group.get("group_id", "")),
        candidate_type=candidate_type,
        delete_allowed=False,
        status="review_required",
        sha256=str(group.get("sha256", "")),
        size_bytes=size_bytes,
        reclaimable_bytes=size_bytes * len(candidate_paths),
        candidate_paths=candidate_paths,
        retained_paths=retained_paths,
        path_kinds=sorted({_path_kind(path) for path in files}),
        blockers=_blockers_for_candidate(candidate_type),
        required_validations=list(REQUIRED_VALIDATIONS),
        reason=_reason_for_candidate(candidate_type),
    )


def _latest_audit_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/"
            "data_duplicate_audit_*/data_duplicate_audit.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_duplicate_audit.json found under reports/project_cleanup"
        )
    return candidates[-1]


def load_data_duplicate_audit(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_data_cleanup_plan(
    audit: dict[str, Any],
    *,
    audit_json_path: Path | None = None,
) -> dict[str, Any]:
    """Build a conservative cleanup plan from duplicate hash groups."""
    candidates: list[CleanupPlanCandidate] = []
    for group in audit.get("duplicate_groups", []):
        candidate = _candidate_from_group(group)
        if candidate is not None:
            candidates.append(candidate)

    candidate_payload = [asdict(candidate) for candidate in candidates]
    type_summary: dict[str, int] = {}
    blocker_summary: dict[str, int] = {}
    for candidate in candidate_payload:
        candidate_type = str(candidate["candidate_type"])
        type_summary[candidate_type] = type_summary.get(candidate_type, 0) + 1
        for blocker in candidate["blockers"]:
            blocker_summary[blocker] = blocker_summary.get(blocker, 0) + 1

    potential_reclaim_bytes = sum(
        int(candidate["reclaimable_bytes"]) for candidate in candidate_payload
    )
    candidate_file_count = sum(
        len(candidate["candidate_paths"]) for candidate in candidate_payload
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_audit_schema": audit.get("schema_version", ""),
        "source_audit_generated_at": audit.get("generated_at", ""),
        "source_audit_json": str(audit_json_path) if audit_json_path else None,
        "root": audit.get("root", ""),
        "delete_candidate_count": 0,
        "summary": {
            "duplicate_group_count": len(audit.get("duplicate_groups", [])),
            "candidate_group_count": len(candidate_payload),
            "candidate_file_count": candidate_file_count,
            "potential_reclaim_bytes": potential_reclaim_bytes,
            "candidate_type_summary": type_summary,
            "blocker_summary": blocker_summary,
        },
        "required_validations": list(REQUIRED_VALIDATIONS),
        "candidates": candidate_payload,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_data_cleanup_plan_markdown(
    plan: dict[str, Any],
    *,
    max_candidates: int = DEFAULT_MAX_MARKDOWN_CANDIDATES,
) -> str:
    summary = plan.get("summary", {})
    candidates = plan.get("candidates", [])
    visible_candidates = candidates[:max_candidates]
    lines = [
        "# Data Cleanup Plan",
        "",
        f"- Schema: `{plan.get('schema_version', '')}`",
        f"- Generated at: `{plan.get('generated_at', '')}`",
        f"- Source audit: `{plan.get('source_audit_json', '')}`",
        f"- Delete candidates: {plan.get('delete_candidate_count', 0)}",
        f"- Candidate groups: {summary.get('candidate_group_count', 0)}",
        f"- Candidate files: {summary.get('candidate_file_count', 0)}",
        (
            "- Potential reclaim bytes: "
            f"{summary.get('potential_reclaim_bytes', 0)}"
        ),
        "",
        "## Required Validations",
        "",
    ]
    for validation in plan.get("required_validations", []):
        lines.append(f"- `{validation}`")

    lines.extend(
        [
            "",
            "## Candidate Type Summary",
            "",
            "| Candidate Type | Groups |",
            "| --- | ---: |",
        ]
    )
    for candidate_type, count in sorted(
        summary.get("candidate_type_summary", {}).items()
    ):
        lines.append(f"| {_markdown_cell(candidate_type)} | {count} |")

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            (
                "| Group | Type | Files | Reclaim Bytes | Delete | "
                "First Candidate |"
            ),
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for candidate in visible_candidates:
        first_path = candidate.get("candidate_paths", [""])[0]
        row_template = (
            "| {group} | {candidate_type} | {files} | {reclaim} | "
            "{delete} | `{path}` |"
        )
        lines.append(
            row_template.format(
                group=_markdown_cell(candidate.get("group_id", "")),
                candidate_type=_markdown_cell(
                    candidate.get("candidate_type", "")
                ),
                files=len(candidate.get("candidate_paths", [])),
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


def write_data_cleanup_plan(
    audit_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    max_markdown_candidates: int = DEFAULT_MAX_MARKDOWN_CANDIDATES,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    audit_path = audit_json_path.resolve()
    audit = load_data_duplicate_audit(audit_path)
    plan = build_data_cleanup_plan(audit, audit_json_path=audit_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_plan_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_plan.json"
    md_path = out_dir / "data_cleanup_plan.md"
    json_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_data_cleanup_plan_markdown(
            plan,
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
        description="Build a no-delete cleanup plan from a duplicate audit."
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="Path to data_duplicate_audit.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown cleanup plan reports.",
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
    try:
        audit_json = args.audit_json or _latest_audit_json(repo_root)
        paths = write_data_cleanup_plan(
            audit_json,
            root=repo_root,
            output_dir=args.output_dir,
            max_markdown_candidates=max(0, args.max_markdown_candidates),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    summary = payload["summary"]
    print("data cleanup plan mode: dry-run")
    print(f"workspace root: {payload['root']}")
    print(f"candidate groups: {summary['candidate_group_count']}")
    print(f"candidate files: {summary['candidate_file_count']}")
    print(f"potential reclaim bytes: {summary['potential_reclaim_bytes']}")
    print(f"delete candidates: {payload['delete_candidate_count']}")
    print("data cleanup plan manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
