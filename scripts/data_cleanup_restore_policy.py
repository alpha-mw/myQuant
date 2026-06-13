"""Build a conservative restore-source duplicate policy report."""

from __future__ import annotations

import argparse
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

SCHEMA_VERSION = "myquant.data_cleanup_restore_policy.v1"
DEFAULT_MAX_MARKDOWN_GROUPS = 500

SYMBOL_RE = re.compile(r"(?P<symbol>[0-9]{6}\.(?:SZ|SH|BJ))")

REQUIRED_VALIDATIONS = (
    "reference_scan_required",
    "retained_copy_policy_required",
    "quant-investor market storage-validate --market CN",
    "quant-investor market storage-validate-clean --market CN",
    "quant-investor market storage-diff --market CN",
)

GENERATED_ARTIFACT_ROLES = {
    "cleaning_row_flags",
    "cleaning_cell_flags",
    "cleaning_report_json",
    "factor_matrix_coverage",
    "factor_ready_masks",
    "factor_readiness_report",
    "factor_readiness_json",
}


@dataclass(frozen=True)
class RestorePolicyGroup:
    group_id: str
    policy_class: str
    risk_level: str
    delete_allowed: bool
    reclaimable_bytes: int
    candidate_paths: list[str]
    retained_paths: list[str]
    path_roles: list[str]
    symbols: list[str]
    blockers: list[str]
    required_validations: list[str]
    reason: str


def _latest_plan_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/data_cleanup_plan_*/data_cleanup_plan.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_cleanup_plan.json found under reports/project_cleanup"
        )
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_role(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if path.startswith("data/raw_backups/tushare/") and name.endswith("_raw.csv"):
        return "raw_backup"
    if path.startswith("data/cleaning_reports/tushare/"):
        if name.endswith("_row_flags.csv"):
            return "cleaning_row_flags"
        if name.endswith("_cell_flags.csv"):
            return "cleaning_cell_flags"
        if name.endswith("_cleaning_report.json"):
            return "cleaning_report_json"
        return "cleaning_report_artifact"
    if path.startswith("data/factor_readiness/tushare/"):
        if name.endswith("_matrix_coverage.json"):
            return "factor_matrix_coverage"
        if name.endswith("_factor_ready_masks.json"):
            return "factor_ready_masks"
        if name.endswith("_factor_readiness_report.json"):
            return "factor_readiness_report"
        return "factor_readiness_json"
    if "/.cache/" in path:
        return "cache"
    return "other_restore_source"


def _symbols(paths: list[str]) -> list[str]:
    found = {
        match.group("symbol")
        for path in paths
        for match in SYMBOL_RE.finditer(path)
    }
    return sorted(found)


def _policy_for(paths: list[str]) -> tuple[str, str, list[str], str]:
    roles = {_path_role(path) for path in paths}
    symbols = _symbols(paths)
    same_symbol = len(symbols) == 1
    all_generated = roles.issubset(GENERATED_ARTIFACT_ROLES)

    if roles == {"raw_backup"} and same_symbol:
        return (
            "same_symbol_raw_backup_duplicate",
            "medium",
            [
                "retained_raw_backup_policy_required",
                "reference_scan_required",
            ],
            (
                "same-symbol raw backups have identical content, but raw "
                "restore lineage must keep a retained-copy policy before delete"
            ),
        )
    if roles.issubset({"cleaning_row_flags", "cleaning_cell_flags"}) and same_symbol:
        return (
            "same_symbol_cleaning_artifact_duplicate",
            "medium",
            [
                "cleaning_report_manifest_check_required",
                "reference_rewrite_required",
            ],
            (
                "same-symbol cleaning artifacts duplicate content, but report "
                "references must be checked or rewritten before delete"
            ),
        )
    if roles.issubset(
        {
            "factor_matrix_coverage",
            "factor_ready_masks",
            "factor_readiness_report",
            "factor_readiness_json",
        }
    ) and same_symbol:
        return (
            "same_symbol_factor_readiness_duplicate",
            "medium",
            [
                "factor_readiness_manifest_check_required",
                "reference_rewrite_required",
            ],
            (
                "same-symbol factor-readiness artifacts duplicate content, but "
                "readiness manifests must be checked or rewritten before delete"
            ),
        )
    if all_generated and len(symbols) > 1:
        return (
            "cross_symbol_generated_artifact_duplicate",
            "high",
            ["cross_symbol_artifact_review_required"],
            (
                "generated artifacts are identical across different symbols; "
                "hash equality alone is not valid deletion evidence"
            ),
        )
    if roles == {"raw_backup"}:
        return (
            "cross_symbol_raw_backup_duplicate",
            "high",
            ["cross_symbol_restore_review_required"],
            "raw backups span multiple symbols or unknown symbols",
        )
    if "cache" in roles:
        return (
            "cache_restore_source_duplicate",
            "low",
            ["cache_regeneration_check_required"],
            "cache duplicate needs regeneration policy before deletion",
        )
    return (
        "mixed_restore_source_duplicate",
        "high",
        ["mixed_artifact_policy_required"],
        "duplicate group mixes restore-source roles and needs manual policy",
    )


def _restore_policy_group(candidate: dict[str, Any]) -> RestorePolicyGroup | None:
    if candidate.get("candidate_type") != "restore_source_duplicate_review":
        return None
    candidate_paths = [str(path) for path in candidate.get("candidate_paths", [])]
    retained_paths = [str(path) for path in candidate.get("retained_paths", [])]
    paths = candidate_paths + retained_paths
    if not candidate_paths or not retained_paths:
        return None
    policy_class, risk_level, blockers, reason = _policy_for(paths)
    return RestorePolicyGroup(
        group_id=str(candidate.get("group_id", "")),
        policy_class=policy_class,
        risk_level=risk_level,
        delete_allowed=False,
        reclaimable_bytes=int(candidate.get("reclaimable_bytes") or 0),
        candidate_paths=candidate_paths,
        retained_paths=retained_paths,
        path_roles=sorted({_path_role(path) for path in paths}),
        symbols=_symbols(paths),
        blockers=blockers,
        required_validations=list(REQUIRED_VALIDATIONS),
        reason=reason,
    )


def build_restore_source_policy(
    plan: dict[str, Any],
    *,
    plan_json_path: Path | None = None,
) -> dict[str, Any]:
    """Build a no-delete policy report for restore-source duplicates."""
    groups = [
        group
        for candidate in plan.get("candidates", [])
        if (group := _restore_policy_group(candidate)) is not None
    ]
    group_payload = [asdict(group) for group in groups]

    policy_class_summary: dict[str, int] = {}
    risk_level_summary: dict[str, int] = {}
    blocker_summary: dict[str, int] = {}
    for group in group_payload:
        policy_class = str(group["policy_class"])
        risk_level = str(group["risk_level"])
        policy_class_summary[policy_class] = (
            policy_class_summary.get(policy_class, 0) + 1
        )
        risk_level_summary[risk_level] = risk_level_summary.get(risk_level, 0) + 1
        for blocker in group["blockers"]:
            blocker_summary[blocker] = blocker_summary.get(blocker, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_plan_schema": plan.get("schema_version", ""),
        "source_plan_generated_at": plan.get("generated_at", ""),
        "source_plan_json": str(plan_json_path) if plan_json_path else None,
        "root": plan.get("root", ""),
        "delete_candidate_count": 0,
        "summary": {
            "restore_source_group_count": len(group_payload),
            "restore_source_candidate_file_count": sum(
                len(group["candidate_paths"]) for group in group_payload
            ),
            "potential_reclaim_bytes": sum(
                int(group["reclaimable_bytes"]) for group in group_payload
            ),
            "policy_class_summary": dict(sorted(policy_class_summary.items())),
            "risk_level_summary": dict(sorted(risk_level_summary.items())),
            "blocker_summary": dict(sorted(blocker_summary.items())),
        },
        "required_validations": list(REQUIRED_VALIDATIONS),
        "groups": group_payload,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_restore_source_policy_markdown(
    policy: dict[str, Any],
    *,
    max_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> str:
    summary = policy.get("summary", {})
    groups = policy.get("groups", [])
    visible_groups = groups[:max_groups]
    lines = [
        "# Data Cleanup Restore-Source Policy",
        "",
        f"- Schema: `{policy.get('schema_version', '')}`",
        f"- Generated at: `{policy.get('generated_at', '')}`",
        f"- Source plan: `{policy.get('source_plan_json', '')}`",
        f"- Delete candidates: {policy.get('delete_candidate_count', 0)}",
        f"- Restore-source groups: {summary.get('restore_source_group_count', 0)}",
        (
            "- Restore-source candidate files: "
            f"{summary.get('restore_source_candidate_file_count', 0)}"
        ),
        (
            "- Potential reclaim bytes: "
            f"{summary.get('potential_reclaim_bytes', 0)}"
        ),
        "",
        "## Policy Class Summary",
        "",
        "| Policy Class | Groups |",
        "| --- | ---: |",
    ]
    for policy_class, count in summary.get("policy_class_summary", {}).items():
        lines.append(f"| {_markdown_cell(policy_class)} | {count} |")

    lines.extend(
        [
            "",
            "## Groups",
            "",
            "| Group | Policy Class | Risk | Files | Reclaim Bytes | First Candidate |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for group in visible_groups:
        candidate_paths = group.get("candidate_paths", [])
        first_path = candidate_paths[0] if candidate_paths else ""
        lines.append(
            "| {group_id} | {policy_class} | {risk} | {files} | {reclaim} | `{path}` |".format(
                group_id=_markdown_cell(group.get("group_id", "")),
                policy_class=_markdown_cell(group.get("policy_class", "")),
                risk=_markdown_cell(group.get("risk_level", "")),
                files=len(candidate_paths),
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


def write_restore_source_policy(
    plan_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    max_markdown_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    plan_path = plan_json_path.resolve()
    plan = _load_json(plan_path)
    policy = build_restore_source_policy(plan, plan_json_path=plan_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_restore_policy_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_restore_policy.json"
    md_path = out_dir / "data_cleanup_restore_policy.md"
    json_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_restore_source_policy_markdown(
            policy,
            max_groups=max_markdown_groups,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a no-delete restore-source duplicate policy report."
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
        help="Directory for JSON/Markdown restore policy reports.",
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
    try:
        plan_json = args.plan_json or _latest_plan_json(repo_root)
        paths = write_restore_source_policy(
            plan_json,
            root=repo_root,
            output_dir=args.output_dir,
            max_markdown_groups=max(0, args.max_markdown_groups),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    print("data cleanup restore-source policy mode: dry-run")
    print(f"workspace root: {payload['root']}")
    print(f"restore-source groups: {summary['restore_source_group_count']}")
    print(
        "restore-source candidate files: "
        f"{summary['restore_source_candidate_file_count']}"
    )
    print(f"potential reclaim bytes: {summary['potential_reclaim_bytes']}")
    print(f"delete candidates: {payload['delete_candidate_count']}")
    print("data cleanup restore-source policy manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
