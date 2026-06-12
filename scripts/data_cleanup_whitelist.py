"""Build a manual-approval whitelist from data cleanup readback results."""

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

SCHEMA_VERSION = "myquant.data_cleanup_whitelist.v1"
DEFAULT_MAX_MARKDOWN_ITEMS = 500

REQUIRED_PRE_DELETE_GATES = (
    "manual_delete_approval_required",
    "quant-investor market storage-validate --market CN",
    "quant-investor market storage-validate-clean --market CN",
    "quant-investor market storage-diff --market CN",
    "fresh cleanup gate report must have zero candidate references",
    "fresh hash readback must pass for every candidate path",
)


@dataclass(frozen=True)
class CleanupWhitelistItem:
    group_id: str
    candidate_type: str
    approval_status: str
    delete_allowed: bool
    execute_allowed: bool
    reclaimable_bytes: int
    candidate_paths: list[str]
    retained_paths: list[str]
    candidate_sha256: list[str]
    retained_sha256: list[str]
    candidate_size_bytes: list[int]
    retained_size_bytes: list[int]
    required_pre_delete_gates: list[str]
    rollback_source_paths: list[str]
    reason: str


def _latest_readback_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/"
            "data_cleanup_readback_*/data_cleanup_readback.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_cleanup_readback.json found under reports/project_cleanup"
        )
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_values(files: list[dict[str, Any]], field: str) -> list[Any]:
    values: list[Any] = []
    for file_item in files:
        value = file_item.get(field)
        if value is not None:
            values.append(value)
    return values


def _whitelist_item(candidate: dict[str, Any]) -> CleanupWhitelistItem:
    candidate_files = list(candidate.get("candidate_files", []))
    retained_files = list(candidate.get("retained_files", []))
    return CleanupWhitelistItem(
        group_id=str(candidate.get("group_id", "")),
        candidate_type=str(candidate.get("candidate_type", "")),
        approval_status="pending_manual_approval",
        delete_allowed=False,
        execute_allowed=False,
        reclaimable_bytes=int(candidate.get("reclaimable_bytes") or 0),
        candidate_paths=[
            str(path) for path in candidate.get("candidate_paths", [])
        ],
        retained_paths=[
            str(path) for path in candidate.get("retained_paths", [])
        ],
        candidate_sha256=[
            str(value) for value in _file_values(candidate_files, "sha256")
        ],
        retained_sha256=[
            str(value) for value in _file_values(retained_files, "sha256")
        ],
        candidate_size_bytes=[
            int(value) for value in _file_values(candidate_files, "size_bytes")
        ],
        retained_size_bytes=[
            int(value) for value in _file_values(retained_files, "size_bytes")
        ],
        required_pre_delete_gates=list(REQUIRED_PRE_DELETE_GATES),
        rollback_source_paths=[
            str(path) for path in candidate.get("retained_paths", [])
        ],
        reason=(
            "candidate has matching retained restore source, but deletion "
            "requires explicit approval and fresh validation"
        ),
    )


def build_data_cleanup_whitelist(
    readback: dict[str, Any],
    *,
    readback_json_path: Path | None = None,
    candidate_type: str | None = None,
) -> dict[str, Any]:
    """Build a manual approval whitelist from hash-passed readback results."""
    items: list[CleanupWhitelistItem] = []
    for candidate in readback.get("candidates", []):
        if candidate.get("readback_status") != "hash_readback_passed":
            continue
        if (
            candidate_type
            and candidate.get("candidate_type") != candidate_type
        ):
            continue
        items.append(_whitelist_item(candidate))

    item_payload = [asdict(item) for item in items]
    type_summary: dict[str, int] = {}
    for item in item_payload:
        candidate_type_value = str(item["candidate_type"])
        type_summary[candidate_type_value] = (
            type_summary.get(candidate_type_value, 0) + 1
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_readback_schema": readback.get("schema_version", ""),
        "source_readback_generated_at": readback.get("generated_at", ""),
        "source_readback_json": (
            str(readback_json_path) if readback_json_path else None
        ),
        "root": readback.get("root", ""),
        "candidate_type_filter": candidate_type,
        "delete_candidate_count": 0,
        "execute_allowed_count": 0,
        "summary": {
            "whitelist_item_count": len(item_payload),
            "candidate_file_count": sum(
                len(item["candidate_paths"]) for item in item_payload
            ),
            "potential_reclaim_bytes": sum(
                int(item["reclaimable_bytes"]) for item in item_payload
            ),
            "candidate_type_summary": type_summary,
        },
        "required_pre_delete_gates": list(REQUIRED_PRE_DELETE_GATES),
        "items": item_payload,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_data_cleanup_whitelist_markdown(
    whitelist: dict[str, Any],
    *,
    max_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> str:
    summary = whitelist.get("summary", {})
    items = whitelist.get("items", [])
    visible_items = items[:max_items]
    lines = [
        "# Data Cleanup Approval Whitelist",
        "",
        f"- Schema: `{whitelist.get('schema_version', '')}`",
        f"- Generated at: `{whitelist.get('generated_at', '')}`",
        f"- Source readback: `{whitelist.get('source_readback_json', '')}`",
        f"- Candidate type filter: `{whitelist.get('candidate_type_filter')}`",
        f"- Delete candidates: {whitelist.get('delete_candidate_count', 0)}",
        f"- Execute allowed: {whitelist.get('execute_allowed_count', 0)}",
        f"- Whitelist items: {summary.get('whitelist_item_count', 0)}",
        f"- Candidate files: {summary.get('candidate_file_count', 0)}",
        (
            "- Potential reclaim bytes: "
            f"{summary.get('potential_reclaim_bytes', 0)}"
        ),
        "",
        "## Required Pre-Delete Gates",
        "",
    ]
    for gate in whitelist.get("required_pre_delete_gates", []):
        lines.append(f"- `{gate}`")

    lines.extend(
        [
            "",
            "## Items",
            "",
            (
                "| Group | Type | Approval | Reclaim Bytes | "
                "Execute | First Candidate |"
            ),
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for item in visible_items:
        first_path = item.get("candidate_paths", [""])[0]
        row_template = (
            "| {group} | {candidate_type} | {approval} | {reclaim} | "
            "{execute} | `{path}` |"
        )
        lines.append(
            row_template.format(
                group=_markdown_cell(item.get("group_id", "")),
                candidate_type=_markdown_cell(
                    item.get("candidate_type", "")
                ),
                approval=_markdown_cell(item.get("approval_status", "")),
                reclaim=item.get("reclaimable_bytes", 0),
                execute="yes" if item.get("execute_allowed") else "no",
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


def write_data_cleanup_whitelist(
    readback_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    candidate_type: str | None = None,
    max_markdown_items: int = DEFAULT_MAX_MARKDOWN_ITEMS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    readback_path = readback_json_path.resolve()
    readback = _load_json(readback_path)
    whitelist = build_data_cleanup_whitelist(
        readback,
        readback_json_path=readback_path,
        candidate_type=candidate_type,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_whitelist_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_whitelist.json"
    md_path = out_dir / "data_cleanup_whitelist.md"
    json_path.write_text(
        json.dumps(whitelist, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_data_cleanup_whitelist_markdown(
            whitelist,
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
        description="Build no-execute approval whitelist from readback report."
    )
    parser.add_argument(
        "--readback-json",
        type=Path,
        default=None,
        help="Path to data_cleanup_readback.json. Defaults to latest report.",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown whitelist reports.",
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
    try:
        readback_json = args.readback_json or _latest_readback_json(repo_root)
        paths = write_data_cleanup_whitelist(
            readback_json,
            root=repo_root,
            output_dir=args.output_dir,
            candidate_type=args.candidate_type,
            max_markdown_items=max(0, args.max_markdown_items),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    print("data cleanup whitelist mode: no-execute")
    print(f"workspace root: {payload['root']}")
    print(f"whitelist items: {summary['whitelist_item_count']}")
    print(f"candidate files: {summary['candidate_file_count']}")
    print(f"potential reclaim bytes: {summary['potential_reclaim_bytes']}")
    print(f"execute allowed: {payload['execute_allowed_count']}")
    print("data cleanup whitelist manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
