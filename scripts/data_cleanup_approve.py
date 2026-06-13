"""Token-gated approval builder for data cleanup whitelist items."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_cleanup_execute import APPROVED_STATUS
from scripts.data_cleanup_whitelist import _approval_packet
from scripts.workspace_layout import get_repo_root

SCHEMA_VERSION = "myquant.data_cleanup_whitelist.v1"
APPROVAL_TOKEN = "APPROVE_DUPLICATE_CLEANUP_GROUPS"
DEFAULT_MAX_MARKDOWN_ITEMS = 500


def _latest_whitelist_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/"
            "data_cleanup_whitelist_*/data_cleanup_whitelist.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_cleanup_whitelist.json found under reports/project_cleanup"
        )
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_matches(item: dict[str, Any]) -> bool:
    candidate_hashes = {str(value) for value in item.get("candidate_sha256", [])}
    retained_hashes = {str(value) for value in item.get("retained_sha256", [])}
    candidate_sizes = {int(value) for value in item.get("candidate_size_bytes", [])}
    retained_sizes = {int(value) for value in item.get("retained_size_bytes", [])}
    if not candidate_hashes or not retained_hashes:
        return False
    if not candidate_sizes or not retained_sizes:
        return False
    return candidate_hashes.issubset(retained_hashes) and candidate_sizes.issubset(
        retained_sizes
    )


def _approval_result_for_item(
    item: dict[str, Any],
    *,
    approve_group_ids: set[str],
    approval_token: str | None,
) -> tuple[dict[str, Any], str]:
    group_id = str(item.get("group_id", ""))
    updated = copy.deepcopy(item)
    if group_id not in approve_group_ids:
        return updated, "not_requested"
    if approval_token != APPROVAL_TOKEN:
        return updated, "blocked_approval_token_required"
    if updated.get("approval_status") == APPROVED_STATUS:
        return updated, "already_approved"
    if updated.get("approval_status") != "pending_manual_approval":
        return updated, "blocked_not_pending_manual_approval"
    if not _metadata_matches(updated):
        return updated, "blocked_metadata_mismatch"

    updated["approval_status"] = APPROVED_STATUS
    updated["delete_allowed"] = True
    updated["execute_allowed"] = True
    updated["reason"] = (
        "manual approval token accepted for selected duplicate cleanup group; "
        "execution still requires data_cleanup_execute confirmation token"
    )
    return updated, APPROVED_STATUS


def _summarize_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    type_summary: dict[str, int] = {}
    for item in items:
        candidate_type = str(item.get("candidate_type", ""))
        type_summary[candidate_type] = type_summary.get(candidate_type, 0) + 1
    return {
        "whitelist_item_count": len(items),
        "candidate_file_count": sum(
            len(item.get("candidate_paths", [])) for item in items
        ),
        "potential_reclaim_bytes": sum(
            int(item.get("reclaimable_bytes") or 0) for item in items
        ),
        "candidate_type_summary": type_summary,
        "manual_approval_required_count": sum(
            1
            for item in items
            if item.get("approval_status") == "pending_manual_approval"
        ),
        "approved_for_delete_count": sum(
            1 for item in items if item.get("approval_status") == APPROVED_STATUS
        ),
    }


def build_data_cleanup_approval(
    whitelist: dict[str, Any],
    *,
    approve_group_ids: Sequence[str],
    approval_token: str | None = None,
    whitelist_json_path: Path | None = None,
) -> dict[str, Any]:
    """Return a new whitelist with explicitly approved duplicate groups."""
    approve_set = {str(group_id) for group_id in approve_group_ids if str(group_id)}
    results: list[dict[str, Any]] = []
    updated_items: list[dict[str, Any]] = []
    status_summary: dict[str, int] = {}
    for item in whitelist.get("items", []):
        updated, status = _approval_result_for_item(
            item,
            approve_group_ids=approve_set,
            approval_token=approval_token,
        )
        updated_items.append(updated)
        status_summary[status] = status_summary.get(status, 0) + 1
        if status != "not_requested":
            results.append(
                {
                    "group_id": str(item.get("group_id", "")),
                    "status": status,
                    "candidate_paths": [
                        str(path) for path in item.get("candidate_paths", [])
                    ],
                    "reclaimable_bytes": int(item.get("reclaimable_bytes") or 0),
                }
            )

    execute_allowed_count = sum(
        1 for item in updated_items if item.get("execute_allowed") is True
    )
    blocked_count = sum(
        count
        for status, count in status_summary.items()
        if status.startswith("blocked")
    )
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        **copy.deepcopy(whitelist),
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_whitelist_schema": whitelist.get("schema_version", ""),
        "source_whitelist_generated_at": whitelist.get("generated_at", ""),
        "source_whitelist_json": (
            str(whitelist_json_path) if whitelist_json_path else None
        ),
        "approval_generated_at": generated_at,
        "approval_token_valid": approval_token == APPROVAL_TOKEN,
        "approved_group_ids": sorted(approve_set),
        "delete_candidate_count": execute_allowed_count,
        "execute_allowed_count": execute_allowed_count,
        "summary": _summarize_items(updated_items),
        "approval_summary": {
            "requested_group_count": len(approve_set),
            "approved_count": status_summary.get(APPROVED_STATUS, 0),
            "already_approved_count": status_summary.get("already_approved", 0),
            "blocked_count": blocked_count,
            "status_summary": dict(sorted(status_summary.items())),
        },
        "approval_results": results,
        "approval_packet": _approval_packet(updated_items),
        "items": updated_items,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_data_cleanup_approval_markdown(
    approval: dict[str, Any],
    *,
    max_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> str:
    summary = approval.get("summary", {})
    approval_summary = approval.get("approval_summary", {})
    results = approval.get("approval_results", [])
    visible_results = results[:max_items]
    lines = [
        "# Data Cleanup Approval Report",
        "",
        f"- Schema: `{approval.get('schema_version', '')}`",
        f"- Generated at: `{approval.get('generated_at', '')}`",
        f"- Source whitelist: `{approval.get('source_whitelist_json', '')}`",
        f"- Approval token valid: `{approval.get('approval_token_valid')}`",
        f"- Requested groups: {approval_summary.get('requested_group_count', 0)}",
        f"- Approved: {approval_summary.get('approved_count', 0)}",
        f"- Already approved: {approval_summary.get('already_approved_count', 0)}",
        f"- Blocked: {approval_summary.get('blocked_count', 0)}",
        f"- Execute allowed: {approval.get('execute_allowed_count', 0)}",
        (
            "- Manual approval still required: "
            f"{summary.get('manual_approval_required_count', 0)}"
        ),
        "",
        "## Approval Results",
        "",
        "| Group | Status | Reclaim Bytes | First Candidate |",
        "| --- | --- | ---: | --- |",
    ]
    for item in visible_results:
        candidate_paths = item.get("candidate_paths", [])
        first_path = candidate_paths[0] if candidate_paths else ""
        lines.append(
            "| {group} | {status} | {reclaim} | `{path}` |".format(
                group=_markdown_cell(item.get("group_id", "")),
                status=_markdown_cell(item.get("status", "")),
                reclaim=item.get("reclaimable_bytes", 0),
                path=_markdown_cell(first_path),
            )
        )
    if len(results) > len(visible_results):
        lines.extend(
            [
                "",
                (
                    f"_Result table truncated to {len(visible_results)} "
                    f"of {len(results)} rows._"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def write_data_cleanup_approval(
    whitelist_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    approve_group_ids: Sequence[str],
    approval_token: str | None = None,
    max_markdown_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    whitelist_path = whitelist_json_path.resolve()
    whitelist = _load_json(whitelist_path)
    approval = build_data_cleanup_approval(
        whitelist,
        approve_group_ids=approve_group_ids,
        approval_token=approval_token,
        whitelist_json_path=whitelist_path,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_whitelist_{timestamp}_approved"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_whitelist.json"
    md_path = out_dir / "data_cleanup_approval.md"
    json_path.write_text(
        json.dumps(approval, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_data_cleanup_approval_markdown(
            approval,
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
        description="Build a token-gated approved data cleanup whitelist."
    )
    parser.add_argument(
        "--whitelist-json",
        type=Path,
        default=None,
        help="Path to data_cleanup_whitelist.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--approve-group",
        action="append",
        default=[],
        help="Group id to approve. Repeat for multiple groups.",
    )
    parser.add_argument(
        "--approval-token",
        default=None,
        help=f"Required token for approval: {APPROVAL_TOKEN}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for approved whitelist JSON and approval Markdown.",
    )
    parser.add_argument(
        "--max-markdown-items",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_ITEMS,
        help="Maximum approval result rows included in Markdown.",
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
        whitelist_json = args.whitelist_json or _latest_whitelist_json(repo_root)
        paths = write_data_cleanup_approval(
            whitelist_json,
            root=repo_root,
            output_dir=args.output_dir,
            approve_group_ids=args.approve_group,
            approval_token=args.approval_token,
            max_markdown_items=max(0, args.max_markdown_items),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    approval_summary = payload["approval_summary"]
    print("data cleanup approval mode: token-gated")
    print(f"workspace root: {payload.get('root', repo_root)}")
    print(f"requested groups: {approval_summary['requested_group_count']}")
    print(f"approved: {approval_summary['approved_count']}")
    print(f"blocked: {approval_summary['blocked_count']}")
    print(f"execute allowed: {payload['execute_allowed_count']}")
    print("data cleanup approval manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
