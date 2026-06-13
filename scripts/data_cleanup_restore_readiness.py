"""Build delete-readiness evidence for restore-source duplicate candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import get_repo_root

SCHEMA_VERSION = "myquant.data_cleanup_restore_readiness.v1"
DEFAULT_MAX_MARKDOWN_GROUPS = 500


def _latest_json(repo_root: Path, pattern: str) -> Path:
    candidates = sorted(repo_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no report found for pattern {pattern}")
    return candidates[-1]


def _latest_policy_json(repo_root: Path) -> Path:
    return _latest_json(
        repo_root,
        "reports/project_cleanup/"
        "data_cleanup_restore_policy_*/data_cleanup_restore_policy.json",
    )


def _latest_reference_audit_json(repo_root: Path) -> Path:
    return _latest_json(
        repo_root,
        "reports/project_cleanup/"
        "data_cleanup_restore_reference_audit_*/"
        "data_cleanup_restore_reference_audit.json",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_groups(reference_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(group.get("group_id", "")): group
        for group in reference_audit.get("groups", [])
        if isinstance(group, dict) and str(group.get("group_id", "")).strip()
    }


def _readiness_for(
    policy_group: dict[str, Any],
    reference_group: dict[str, Any] | None,
) -> tuple[str, list[str], list[str]]:
    if reference_group is None:
        return (
            "blocked_missing_reference_audit_group",
            ["reference_audit_group_missing"],
            ["rerun restore reference audit before considering deletion"],
        )

    referenced_paths = (
        reference_group.get("referenced_candidate_paths", [])
    )
    if referenced_paths:
        return (
            "blocked_referenced_candidate",
            ["candidate_path_still_referenced"],
            ["remove or rewrite references before considering deletion"],
        )

    policy_class = str(policy_group.get("policy_class", ""))
    risk_level = str(policy_group.get("risk_level", ""))
    if risk_level == "high" or policy_class.startswith("cross_symbol_"):
        return (
            "blocked_high_risk_policy",
            list(policy_group.get("blockers", [])) or ["high_risk_policy_review_required"],
            ["perform manual cross-symbol lineage review"],
        )
    if policy_class == "same_symbol_raw_backup_duplicate":
        return (
            "review_retained_copy_readback_required",
            [
                "retained_copy_readback_required",
                "manual_approval_required",
            ],
            [
                "verify retained raw backup path exists and hash matches",
                "run storage validation in the same approval window",
            ],
        )
    if policy_class in {
        "same_symbol_cleaning_artifact_duplicate",
        "same_symbol_factor_readiness_duplicate",
    }:
        return (
            "review_manifest_rewrite_required",
            [
                "reference_rewrite_required",
                "manifest_rewrite_required",
                "manual_approval_required",
            ],
            [
                "rewrite or compact report manifests before deletion",
                "run storage validation in the same approval window",
            ],
        )
    return (
        "blocked_policy_not_covered",
        list(policy_group.get("blockers", [])) or ["policy_not_covered"],
        ["add an explicit restore-source readiness policy"],
    )


def build_restore_readiness(
    policy: dict[str, Any],
    reference_audit: dict[str, Any],
    *,
    policy_json_path: Path | None = None,
    reference_audit_json_path: Path | None = None,
) -> dict[str, Any]:
    """Build a read-only delete-readiness report for restore-source groups."""
    reference_by_group = _reference_groups(reference_audit)
    groups: list[dict[str, Any]] = []
    readiness_summary: dict[str, int] = {}
    reference_free_group_count = 0
    referenced_group_count = 0
    reference_unknown_group_count = 0
    reference_free_candidate_path_count = 0
    reference_free_reclaim_bytes = 0

    for policy_group in policy.get("groups", []):
        if not isinstance(policy_group, dict):
            continue
        group_id = str(policy_group.get("group_id", ""))
        reference_group = reference_by_group.get(group_id)
        referenced_paths = (
            [str(path) for path in reference_group.get("referenced_candidate_paths", [])]
            if reference_group
            else []
        )
        unreferenced_paths = (
            [str(path) for path in reference_group.get("unreferenced_candidate_paths", [])]
            if reference_group
            else []
        )
        readiness_class, blockers, required_actions = _readiness_for(
            policy_group,
            reference_group,
        )
        readiness_summary[readiness_class] = (
            readiness_summary.get(readiness_class, 0) + 1
        )
        reclaimable_bytes = int(policy_group.get("reclaimable_bytes", 0) or 0)
        if referenced_paths:
            referenced_group_count += 1
        elif reference_group is None:
            reference_unknown_group_count += 1
        else:
            reference_free_group_count += 1
            reference_free_candidate_path_count += len(unreferenced_paths)
            reference_free_reclaim_bytes += reclaimable_bytes
        groups.append(
            {
                "group_id": group_id,
                "policy_class": str(policy_group.get("policy_class", "")),
                "risk_level": str(policy_group.get("risk_level", "")),
                "readiness_class": readiness_class,
                "delete_allowed": False,
                "reclaimable_bytes": reclaimable_bytes,
                "candidate_paths": [
                    str(path) for path in policy_group.get("candidate_paths", [])
                ],
                "retained_paths": [
                    str(path) for path in policy_group.get("retained_paths", [])
                ],
                "referenced_candidate_paths": referenced_paths,
                "unreferenced_candidate_paths": unreferenced_paths,
                "blockers": blockers,
                "required_actions": required_actions,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_policy_schema": policy.get("schema_version", ""),
        "source_policy_generated_at": policy.get("generated_at", ""),
        "source_policy_json": str(policy_json_path) if policy_json_path else None,
        "source_reference_audit_schema": reference_audit.get("schema_version", ""),
        "source_reference_audit_generated_at": reference_audit.get(
            "generated_at",
            "",
        ),
        "source_reference_audit_json": (
            str(reference_audit_json_path)
            if reference_audit_json_path
            else None
        ),
        "delete_candidate_count": 0,
        "summary": {
            "group_count": len(groups),
            "reference_free_group_count": reference_free_group_count,
            "referenced_group_count": referenced_group_count,
            "reference_unknown_group_count": reference_unknown_group_count,
            "reference_free_candidate_path_count": reference_free_candidate_path_count,
            "reference_free_potential_reclaim_bytes": reference_free_reclaim_bytes,
            "readiness_class_summary": dict(sorted(readiness_summary.items())),
        },
        "groups": groups,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_restore_readiness_markdown(
    readiness: dict[str, Any],
    *,
    max_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> str:
    summary = readiness.get("summary", {})
    groups = readiness.get("groups", [])
    visible_groups = groups[:max_groups]
    lines = [
        "# Data Cleanup Restore-Source Readiness",
        "",
        f"- Schema: `{readiness.get('schema_version', '')}`",
        f"- Generated at: `{readiness.get('generated_at', '')}`",
        f"- Source policy: `{readiness.get('source_policy_json', '')}`",
        (
            "- Source reference audit: "
            f"`{readiness.get('source_reference_audit_json', '')}`"
        ),
        f"- Delete candidates: {readiness.get('delete_candidate_count', 0)}",
        f"- Groups: {summary.get('group_count', 0)}",
        f"- Reference-free groups: {summary.get('reference_free_group_count', 0)}",
        f"- Referenced groups: {summary.get('referenced_group_count', 0)}",
        (
            "- Reference-free potential reclaim bytes: "
            f"{summary.get('reference_free_potential_reclaim_bytes', 0)}"
        ),
        "",
        "## Groups",
        "",
        "| Group | Policy Class | Readiness | Reclaim Bytes | Blockers |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for group in visible_groups:
        lines.append(
            "| {group_id} | {policy_class} | {readiness_class} | {reclaim} | {blockers} |".format(
                group_id=_markdown_cell(group.get("group_id", "")),
                policy_class=_markdown_cell(group.get("policy_class", "")),
                readiness_class=_markdown_cell(group.get("readiness_class", "")),
                reclaim=group.get("reclaimable_bytes", 0),
                blockers=_markdown_cell(",".join(group.get("blockers", []))),
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


def write_restore_readiness(
    policy_json_path: Path,
    reference_audit_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    max_markdown_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    policy_path = policy_json_path.resolve()
    reference_path = reference_audit_json_path.resolve()
    policy = _load_json(policy_path)
    reference_audit = _load_json(reference_path)
    readiness = build_restore_readiness(
        policy,
        reference_audit,
        policy_json_path=policy_path,
        reference_audit_json_path=reference_path,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_restore_readiness_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_restore_readiness.json"
    md_path = out_dir / "data_cleanup_restore_readiness.md"
    json_path.write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_restore_readiness_markdown(
            readiness,
            max_groups=max_markdown_groups,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a no-delete restore-source delete-readiness report."
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=None,
        help="Path to data_cleanup_restore_policy.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--reference-audit-json",
        type=Path,
        default=None,
        help=(
            "Path to data_cleanup_restore_reference_audit.json. Defaults to "
            "latest report."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown readiness reports.",
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
        policy_json = args.policy_json or _latest_policy_json(repo_root)
        reference_json = (
            args.reference_audit_json or _latest_reference_audit_json(repo_root)
        )
        paths = write_restore_readiness(
            policy_json,
            reference_json,
            root=repo_root,
            output_dir=args.output_dir,
            max_markdown_groups=max(0, args.max_markdown_groups),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    print("data cleanup restore-source readiness mode: dry-run")
    print(f"reference-free groups: {summary['reference_free_group_count']}")
    print(f"referenced groups: {summary['referenced_group_count']}")
    print(
        "reference-free potential reclaim bytes: "
        f"{summary['reference_free_potential_reclaim_bytes']}"
    )
    print(f"delete candidates: {payload['delete_candidate_count']}")
    print("data cleanup restore-source readiness manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
