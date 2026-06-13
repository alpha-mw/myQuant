"""Audit references to restore-source duplicate candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import get_repo_root

SCHEMA_VERSION = "myquant.data_cleanup_restore_reference_audit.v1"
DEFAULT_MAX_MARKDOWN_GROUPS = 500
DEFAULT_MAX_REFERENCE_EXAMPLES = 5
DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024

DEFAULT_SCAN_ROOTS = (
    Path("data") / "cleaning_reports" / "tushare",
    Path("data") / "factor_readiness" / "tushare",
    Path("results") / "strategy_records",
    Path("reports") / "daily",
    Path("reports") / "storage",
)

SCAN_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
}

LINEAGE_REFERENCE_ROOTS = (
    Path("data") / "cleaning_reports" / "tushare",
    Path("data") / "factor_readiness" / "tushare",
)

LINEAGE_REFERENCE_REPORT_SUFFIXES = (
    "_cleaning_report.json",
    "_factor_readiness_report.json",
)

PATH_TOKEN_RE = re.compile(
    r"(?:/[^\s\"',;)\]}]+/myQuant/)?"
    r"(?:data|reports|results)/[^\s\"',;)\]}]+"
)


def _latest_policy_json(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob(
            "reports/project_cleanup/"
            "data_cleanup_restore_policy_*/data_cleanup_restore_policy.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no data_cleanup_restore_policy.json found under reports/project_cleanup"
        )
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _normalise_token(repo_root: Path, token: str) -> str | None:
    cleaned = token.strip().strip("\"'`")
    if not cleaned:
        return None
    if cleaned.startswith(str(repo_root)):
        try:
            return Path(cleaned).resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return None
    marker = "/myQuant/"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[1]
    if cleaned.startswith(("data/", "reports/", "results/")):
        return Path(cleaned).as_posix()
    return None


def _candidate_paths(policy: dict[str, Any]) -> set[str]:
    return {
        str(path)
        for group in policy.get("groups", [])
        if isinstance(group, dict)
        for path in group.get("candidate_paths", [])
    }


def _iter_scan_files(
    repo_root: Path,
    scan_roots: Sequence[Path],
    *,
    candidate_paths: set[str],
    include_all_text_files: bool,
) -> Iterable[Path]:
    project_cleanup_root = (repo_root / "reports" / "project_cleanup").resolve()
    owner_paths = _candidate_reference_owner_paths(repo_root, candidate_paths)
    for scan_root in scan_roots:
        root = scan_root if scan_root.is_absolute() else repo_root / scan_root
        if not root.exists():
            continue
        if not include_all_text_files and _is_lineage_scan_root(repo_root, root):
            for path in sorted(owner_paths):
                if not _path_is_under_scan_root(path, root):
                    continue
                if not path.exists() or not path.is_file():
                    continue
                if path.suffix.lower() not in SCAN_SUFFIXES:
                    continue
                yield path
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            if not include_all_text_files and not _is_default_reference_file(
                repo_root,
                path,
            ):
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            try:
                resolved.relative_to(project_cleanup_root)
                continue
            except ValueError:
                pass
            yield path


def _candidate_reference_owner_paths(
    repo_root: Path,
    candidate_paths: set[str],
) -> set[Path]:
    owners: set[Path] = set()
    for candidate in candidate_paths:
        for owner in _candidate_reference_owner_paths_for_candidate(candidate):
            owners.add(repo_root / owner)
    return owners


def _candidate_reference_owner_paths_for_candidate(
    candidate_path: str,
) -> set[Path]:
    owners: set[Path] = set()
    if candidate_path.startswith("data/raw_backups/tushare/"):
        relative = candidate_path.replace(
            "data/raw_backups/tushare/",
            "data/cleaning_reports/tushare/",
            1,
        )
        if relative.endswith("_raw.csv"):
            owners.add(
                Path(relative[: -len("_raw.csv")] + "_cleaning_report.json")
            )
    if candidate_path.startswith("data/cleaning_reports/tushare/"):
        if candidate_path.endswith("_row_flags.csv"):
            owners.add(
                Path(
                    candidate_path[: -len("_row_flags.csv")]
                    + "_cleaning_report.json"
                )
            )
        if candidate_path.endswith("_cell_flags.csv"):
            owners.add(
                Path(
                    candidate_path[: -len("_cell_flags.csv")]
                    + "_cleaning_report.json"
                )
            )
    if candidate_path.startswith("data/factor_readiness/tushare/"):
        cleaning_relative = candidate_path.replace(
            "data/factor_readiness/tushare/",
            "data/cleaning_reports/tushare/",
            1,
        )
        if candidate_path.endswith("_matrix_coverage.json"):
            owners.add(
                Path(
                    candidate_path[: -len("_matrix_coverage.json")]
                    + "_factor_readiness_report.json"
                )
            )
            owners.add(
                Path(
                    cleaning_relative[: -len("_matrix_coverage.json")]
                    + "_cleaning_report.json"
                )
            )
        if candidate_path.endswith("_factor_ready_masks.json"):
            owners.add(
                Path(
                    candidate_path[: -len("_factor_ready_masks.json")]
                    + "_factor_readiness_report.json"
                )
            )
            owners.add(
                Path(
                    cleaning_relative[: -len("_factor_ready_masks.json")]
                    + "_cleaning_report.json"
                )
            )
    return owners


def _is_lineage_scan_root(repo_root: Path, root: Path) -> bool:
    relative = Path(_relative_path(repo_root, root))
    return any(
        _is_relative_to(reference_root, relative)
        or _is_relative_to(relative, reference_root)
        for reference_root in LINEAGE_REFERENCE_ROOTS
    )


def _path_is_under_scan_root(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        resolved_path = path
        resolved_root = root
    if resolved_root.is_file():
        return resolved_path == resolved_root
    return _is_relative_to(resolved_path, resolved_root)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_default_reference_file(repo_root: Path, path: Path) -> bool:
    relative_path = _relative_path(repo_root, path)
    relative = Path(relative_path)
    for root in LINEAGE_REFERENCE_ROOTS:
        if _is_relative_to(relative, root):
            return path.name.endswith(LINEAGE_REFERENCE_REPORT_SUFFIXES)
    return True


def _scan_text_for_candidate_paths(
    repo_root: Path,
    text: str,
    candidate_paths: set[str],
) -> set[str]:
    matched: set[str] = set()
    for match in PATH_TOKEN_RE.finditer(text):
        normalised = _normalise_token(repo_root, match.group(0))
        if normalised in candidate_paths:
            matched.add(normalised)
    return matched


def _reference_index(
    repo_root: Path,
    candidate_paths: set[str],
    *,
    scan_roots: Sequence[Path],
    max_file_bytes: int,
    include_all_text_files: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    references: dict[str, list[dict[str, Any]]] = {
        path: [] for path in candidate_paths
    }
    stats = {
        "scan_file_count": 0,
        "scan_skipped_large_file_count": 0,
        "scan_read_error_count": 0,
    }
    for path in _iter_scan_files(
        repo_root,
        scan_roots,
        candidate_paths=candidate_paths,
        include_all_text_files=include_all_text_files,
    ):
        try:
            size = path.stat().st_size
        except OSError:
            stats["scan_read_error_count"] += 1
            continue
        if size > max_file_bytes:
            stats["scan_skipped_large_file_count"] += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            stats["scan_read_error_count"] += 1
            continue
        stats["scan_file_count"] += 1
        matched_paths = _scan_text_for_candidate_paths(
            repo_root,
            text,
            candidate_paths,
        )
        if not matched_paths:
            continue
        reference_path = _relative_path(repo_root, path)
        for candidate_path in sorted(matched_paths):
            references[candidate_path].append(
                {
                    "candidate_path": candidate_path,
                    "reference_path": reference_path,
                    "reference_kind": "path_token",
                }
            )
    return references, stats


def _summarize_policy_class(
    summary: dict[str, dict[str, int]],
    policy_class: str,
    *,
    referenced: bool,
) -> None:
    bucket = summary.setdefault(
        policy_class,
        {
            "group_count": 0,
            "referenced_group_count": 0,
            "unreferenced_group_count": 0,
        },
    )
    bucket["group_count"] += 1
    if referenced:
        bucket["referenced_group_count"] += 1
    else:
        bucket["unreferenced_group_count"] += 1


def build_restore_reference_audit(
    policy: dict[str, Any],
    *,
    root: Path | None = None,
    policy_json_path: Path | None = None,
    scan_roots: Sequence[Path] = DEFAULT_SCAN_ROOTS,
    max_reference_examples: int = DEFAULT_MAX_REFERENCE_EXAMPLES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    include_all_text_files: bool = False,
) -> dict[str, Any]:
    """Build a read-only reference audit for restore-source candidates."""
    repo_root = get_repo_root(root)
    candidates = _candidate_paths(policy)
    references, scan_stats = _reference_index(
        repo_root,
        candidates,
        scan_roots=scan_roots,
        max_file_bytes=max_file_bytes,
        include_all_text_files=include_all_text_files,
    )
    groups: list[dict[str, Any]] = []
    referenced_candidate_paths: set[str] = set()
    policy_class_summary: dict[str, dict[str, int]] = {}

    for group in policy.get("groups", []):
        if not isinstance(group, dict):
            continue
        candidate_paths = [str(path) for path in group.get("candidate_paths", [])]
        referenced_paths = [
            path for path in candidate_paths if references.get(path)
        ]
        unreferenced_paths = [
            path for path in candidate_paths if not references.get(path)
        ]
        for path in referenced_paths:
            referenced_candidate_paths.add(path)
        group_references = [
            reference
            for path in referenced_paths
            for reference in references.get(path, [])
        ]
        policy_class = str(group.get("policy_class", "unknown"))
        _summarize_policy_class(
            policy_class_summary,
            policy_class,
            referenced=bool(referenced_paths),
        )
        groups.append(
            {
                "group_id": str(group.get("group_id", "")),
                "policy_class": policy_class,
                "risk_level": str(group.get("risk_level", "")),
                "delete_allowed": False,
                "reclaimable_bytes": int(group.get("reclaimable_bytes", 0) or 0),
                "candidate_paths": candidate_paths,
                "referenced_candidate_paths": referenced_paths,
                "unreferenced_candidate_paths": unreferenced_paths,
                "reference_count": len(group_references),
                "reference_examples": group_references[:max_reference_examples],
            }
        )

    unreferenced_candidate_paths = candidates - referenced_candidate_paths
    referenced_group_count = sum(
        1 for group in groups if group["referenced_candidate_paths"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_policy_schema": policy.get("schema_version", ""),
        "source_policy_generated_at": policy.get("generated_at", ""),
        "source_policy_json": str(policy_json_path) if policy_json_path else None,
        "root": str(repo_root),
        "scan_roots": [
            path.as_posix() for path in scan_roots
        ],
        "delete_candidate_count": 0,
        "summary": {
            **scan_stats,
            "scan_mode": (
                "all_text_files"
                if include_all_text_files
                else "candidate_owner_reports"
            ),
            "group_count": len(groups),
            "candidate_path_count": len(candidates),
            "referenced_group_count": referenced_group_count,
            "unreferenced_group_count": len(groups) - referenced_group_count,
            "referenced_candidate_path_count": len(referenced_candidate_paths),
            "unreferenced_candidate_path_count": len(unreferenced_candidate_paths),
            "policy_class_reference_summary": dict(
                sorted(policy_class_summary.items())
            ),
        },
        "groups": groups,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_restore_reference_audit_markdown(
    audit: dict[str, Any],
    *,
    max_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> str:
    summary = audit.get("summary", {})
    groups = audit.get("groups", [])
    visible_groups = groups[:max_groups]
    lines = [
        "# Data Cleanup Restore-Source Reference Audit",
        "",
        f"- Schema: `{audit.get('schema_version', '')}`",
        f"- Generated at: `{audit.get('generated_at', '')}`",
        f"- Source policy: `{audit.get('source_policy_json', '')}`",
        f"- Delete candidates: {audit.get('delete_candidate_count', 0)}",
        f"- Candidate paths: {summary.get('candidate_path_count', 0)}",
        (
            "- Referenced candidate paths: "
            f"{summary.get('referenced_candidate_path_count', 0)}"
        ),
        (
            "- Unreferenced candidate paths: "
            f"{summary.get('unreferenced_candidate_path_count', 0)}"
        ),
        f"- Scanned files: {summary.get('scan_file_count', 0)}",
        "",
        "## Groups",
        "",
        "| Group | Policy Class | Risk | References | Referenced Paths | First Reference |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for group in visible_groups:
        examples = group.get("reference_examples", [])
        first_reference = examples[0].get("reference_path", "") if examples else ""
        lines.append(
            "| {group_id} | {policy_class} | {risk} | {references} | {paths} | `{reference}` |".format(
                group_id=_markdown_cell(group.get("group_id", "")),
                policy_class=_markdown_cell(group.get("policy_class", "")),
                risk=_markdown_cell(group.get("risk_level", "")),
                references=group.get("reference_count", 0),
                paths=len(group.get("referenced_candidate_paths", [])),
                reference=_markdown_cell(first_reference),
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


def write_restore_reference_audit(
    policy_json_path: Path,
    *,
    root: Path | None = None,
    output_dir: str | Path | None = None,
    scan_roots: Sequence[Path] = DEFAULT_SCAN_ROOTS,
    max_markdown_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
    max_reference_examples: int = DEFAULT_MAX_REFERENCE_EXAMPLES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    include_all_text_files: bool = False,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    policy_path = policy_json_path.resolve()
    policy = _load_json(policy_path)
    audit = build_restore_reference_audit(
        policy,
        root=repo_root,
        policy_json_path=policy_path,
        scan_roots=scan_roots,
        max_reference_examples=max_reference_examples,
        max_file_bytes=max_file_bytes,
        include_all_text_files=include_all_text_files,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_restore_reference_audit_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_restore_reference_audit.json"
    md_path = out_dir / "data_cleanup_restore_reference_audit.md"
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_restore_reference_audit_markdown(
            audit,
            max_groups=max_markdown_groups,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a no-delete restore-source reference audit report."
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=None,
        help="Path to data_cleanup_restore_policy.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=None,
        help="Relative or absolute root to scan. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown reference audit reports.",
    )
    parser.add_argument(
        "--max-markdown-groups",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_GROUPS,
        help="Maximum group rows included in Markdown.",
    )
    parser.add_argument(
        "--max-reference-examples",
        type=int,
        default=DEFAULT_MAX_REFERENCE_EXAMPLES,
        help="Maximum reference examples retained per group.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Skip text files larger than this size.",
    )
    parser.add_argument(
        "--include-all-text-files",
        action="store_true",
        help=(
            "Scan every supported text file under scan roots. By default the "
            "large Tushare lineage roots scan only known manifest/report files."
        ),
    )
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = get_repo_root(args.root)
    try:
        policy_json = args.policy_json or _latest_policy_json(repo_root)
        paths = write_restore_reference_audit(
            policy_json,
            root=repo_root,
            output_dir=args.output_dir,
            scan_roots=args.scan_root or DEFAULT_SCAN_ROOTS,
            max_markdown_groups=max(0, args.max_markdown_groups),
            max_reference_examples=max(0, args.max_reference_examples),
            max_file_bytes=max(0, args.max_file_bytes),
            include_all_text_files=bool(args.include_all_text_files),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = _load_json(Path(paths["json"]))
    summary = payload["summary"]
    print("data cleanup restore-source reference audit mode: dry-run")
    print(f"workspace root: {payload['root']}")
    print(f"candidate paths: {summary['candidate_path_count']}")
    print(f"referenced candidate paths: {summary['referenced_candidate_path_count']}")
    print(f"unreferenced candidate paths: {summary['unreferenced_candidate_path_count']}")
    print(f"referenced groups: {summary['referenced_group_count']}")
    print(f"unreferenced groups: {summary['unreferenced_group_count']}")
    print(f"scanned files: {summary['scan_file_count']}")
    print(f"delete candidates: {payload['delete_candidate_count']}")
    print("data cleanup restore-source reference audit manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
