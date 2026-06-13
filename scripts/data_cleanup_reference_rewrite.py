"""Rewrite owner-report references before deleting duplicate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import get_repo_root

SCHEMA_VERSION = "myquant.data_cleanup_reference_rewrite.v1"
CLEANING_CONFIRM_TOKEN = "REWRITE_REFERENCED_CLEANING_ARTIFACTS"
RESTORE_SOURCE_CONFIRM_TOKEN = "REWRITE_REFERENCED_RESTORE_SOURCES"
CONFIRM_TOKEN = CLEANING_CONFIRM_TOKEN
DEFAULT_MAX_GROUPS = 20
DEFAULT_MAX_MARKDOWN_GROUPS = 200
DEFAULT_MAX_FILE_BYTES = 128 * 1024 * 1024
REWRITABLE_POLICY_CLASSES = {
    "same_symbol_cleaning_artifact_duplicate",
    "same_symbol_factor_readiness_duplicate",
    "same_symbol_raw_backup_duplicate",
}
REWRITABLE_REFERENCE_KEYS = {
    "cell_flags_path",
    "factor_ready_masks_path",
    "matrix_coverage_path",
    "raw_backup_path",
    "row_flags_path",
}
POLICY_CLASS_SUFFIXES = {
    "same_symbol_cleaning_artifact_duplicate": ("_row_flags.csv", "_cell_flags.csv"),
    "same_symbol_factor_readiness_duplicate": (
        "_factor_ready_masks.json",
        "_matrix_coverage.json",
    ),
    "same_symbol_raw_backup_duplicate": ("_raw.csv",),
}
IGNORED_DERIVED_REFERENCE_PREFIXES = (
    "reports/storage/csv_inventory_",
)
BOUNDED_EXTERNAL_SCAN_ROOTS = (
    Path("reports") / "daily",
    Path("reports") / "storage",
    Path("results") / "strategy_records",
)
REFERENCE_SCAN_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
}


@dataclass(frozen=True)
class ReferenceRewriteItem:
    group_id: str
    policy_class: str
    risk_level: str
    status: str
    action: str
    delete_allowed: bool
    reclaimable_bytes: int
    candidate_path: str
    retained_path: str
    owner_paths: list[str]
    references_rewritten: int
    ignored_reference_count: int
    errors: list[str]
    reason: str


def _latest_json(repo_root: Path, pattern: str) -> Path:
    candidates = sorted(repo_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no report found for {pattern}")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative_path(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cleaning_report_for_artifact(candidate_path: str) -> str | None:
    if not candidate_path.startswith("data/cleaning_reports/tushare/"):
        return None
    if candidate_path.endswith("_row_flags.csv"):
        return candidate_path[: -len("_row_flags.csv")] + "_cleaning_report.json"
    if candidate_path.endswith("_cell_flags.csv"):
        return candidate_path[: -len("_cell_flags.csv")] + "_cleaning_report.json"
    return None


def _raw_backup_cleaning_report_for_candidate(candidate_path: str) -> str | None:
    if not candidate_path.startswith("data/raw_backups/tushare/"):
        return None
    relative = candidate_path.replace(
        "data/raw_backups/tushare/",
        "data/cleaning_reports/tushare/",
        1,
    )
    if relative.endswith("_raw.csv"):
        return relative[: -len("_raw.csv")] + "_cleaning_report.json"
    return None


def _raw_backup_factor_report_for_candidate(candidate_path: str) -> str | None:
    if not candidate_path.startswith("data/raw_backups/tushare/"):
        return None
    relative = candidate_path.replace(
        "data/raw_backups/tushare/",
        "data/factor_readiness/tushare/",
        1,
    )
    if relative.endswith("_raw.csv"):
        return relative[: -len("_raw.csv")] + "_factor_readiness_report.json"
    return None


def _factor_cleaning_report_for_candidate(candidate_path: str) -> str | None:
    if not candidate_path.startswith("data/factor_readiness/tushare/"):
        return None
    relative = candidate_path.replace(
        "data/factor_readiness/tushare/",
        "data/cleaning_reports/tushare/",
        1,
    )
    if relative.endswith("_matrix_coverage.json"):
        return relative[: -len("_matrix_coverage.json")] + "_cleaning_report.json"
    if relative.endswith("_factor_ready_masks.json"):
        return relative[: -len("_factor_ready_masks.json")] + "_cleaning_report.json"
    return None


def _factor_report_for_candidate(candidate_path: str) -> str | None:
    if not candidate_path.startswith("data/factor_readiness/tushare/"):
        return None
    if candidate_path.endswith("_matrix_coverage.json"):
        return candidate_path[: -len("_matrix_coverage.json")] + (
            "_factor_readiness_report.json"
        )
    if candidate_path.endswith("_factor_ready_masks.json"):
        return candidate_path[: -len("_factor_ready_masks.json")] + (
            "_factor_readiness_report.json"
        )
    return None


def _owner_paths_for_policy_candidate(
    repo_root: Path,
    policy_class: str,
    candidate_path: str,
) -> list[str]:
    owner_paths: list[str] = []
    if policy_class == "same_symbol_cleaning_artifact_duplicate":
        cleaning_report_path = _cleaning_report_for_artifact(candidate_path)
        if cleaning_report_path is None:
            return owner_paths
        owner_paths.append(cleaning_report_path)
        cleaning_payload = _load_json(repo_root / cleaning_report_path)
        if cleaning_payload is None:
            return owner_paths
        factor_report_path = _factor_report_from_cleaning_report(
            repo_root,
            cleaning_payload,
        )
        if factor_report_path and (repo_root / factor_report_path).exists():
            owner_paths.append(factor_report_path)
        return owner_paths
    if policy_class == "same_symbol_raw_backup_duplicate":
        cleaning_report_path = _raw_backup_cleaning_report_for_candidate(candidate_path)
        factor_report_path = _raw_backup_factor_report_for_candidate(candidate_path)
        for path in (cleaning_report_path, factor_report_path):
            if path and (repo_root / path).exists():
                owner_paths.append(path)
        return owner_paths
    if policy_class == "same_symbol_factor_readiness_duplicate":
        factor_report_path = _factor_report_for_candidate(candidate_path)
        cleaning_report_path = _factor_cleaning_report_for_candidate(candidate_path)
        for path in (factor_report_path, cleaning_report_path):
            if path and (repo_root / path).exists():
                owner_paths.append(path)
        return owner_paths
    return owner_paths


def _rewritable_artifact_suffix(policy_class: str, path: str) -> str | None:
    for suffix in POLICY_CLASS_SUFFIXES.get(policy_class, ()):
        if path.endswith(suffix):
            return suffix
    return None


def _candidate_paths_for_policy_group(group: dict[str, Any]) -> list[str]:
    policy_class = str(group.get("policy_class", ""))
    retained_paths = [str(path) for path in group.get("retained_paths", [])]
    if not retained_paths:
        return []
    retained_suffix = _rewritable_artifact_suffix(policy_class, retained_paths[0])
    if retained_suffix is None:
        return []
    candidate_paths: list[str] = []
    for candidate_path in group.get("candidate_paths", []):
        candidate_text = str(candidate_path)
        if _rewritable_artifact_suffix(policy_class, candidate_text) != retained_suffix:
            return []
        candidate_paths.append(candidate_text)
    return candidate_paths


def _valid_confirm_token_for_policy_class(
    policy_class: str,
    confirm_token: str | None,
) -> bool:
    if confirm_token == RESTORE_SOURCE_CONFIRM_TOKEN:
        return True
    return (
        policy_class == "same_symbol_cleaning_artifact_duplicate"
        and confirm_token == CLEANING_CONFIRM_TOKEN
    )


def _valid_confirm_token_for_groups(
    policy_groups: dict[str, dict[str, Any]],
    selected_group_ids: list[str],
    confirm_token: str | None,
) -> bool:
    if not selected_group_ids:
        return confirm_token in {
            CLEANING_CONFIRM_TOKEN,
            RESTORE_SOURCE_CONFIRM_TOKEN,
        }
    return all(
        _valid_confirm_token_for_policy_class(
            str(policy_groups[group_id].get("policy_class", "")),
            confirm_token,
        )
        for group_id in selected_group_ids
    )


def _factor_report_from_cleaning_report(
    repo_root: Path,
    cleaning_report: dict[str, Any],
) -> str | None:
    metadata = cleaning_report.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("factor_readiness_report_path")
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        try:
            return path.relative_to(repo_root).as_posix()
        except ValueError:
            return None
    return path.as_posix()


def _rewrite_reference_values(payload: Any, old_path: str, new_path: str) -> int:
    rewritten = 0
    if isinstance(payload, dict):
        matched_reference_key = False
        for key, value in list(payload.items()):
            if value == old_path:
                payload[key] = new_path
                rewritten += 1
                if key in REWRITABLE_REFERENCE_KEYS:
                    matched_reference_key = True
            else:
                rewritten += _rewrite_reference_values(value, old_path, new_path)
        if matched_reference_key:
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                payload["metadata"] = metadata
            metadata["duplicate_reference_rewritten"] = True
            metadata["duplicate_reference_rewrite_schema"] = SCHEMA_VERSION
    elif isinstance(payload, list):
        for index, value in enumerate(list(payload)):
            if value == old_path:
                payload[index] = new_path
                rewritten += 1
            else:
                rewritten += _rewrite_reference_values(value, old_path, new_path)
    return rewritten


def _group_maps(
    policy: dict[str, Any],
    reference_audit: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    policy_groups = {
        str(group.get("group_id", "")): group
        for group in policy.get("groups", [])
        if isinstance(group, dict)
    }
    reference_groups = {
        str(group.get("group_id", "")): group
        for group in reference_audit.get("groups", [])
        if isinstance(group, dict)
    }
    return policy_groups, reference_groups


def _candidate_group_ids(
    policy: dict[str, Any],
    reference_audit: dict[str, Any],
    *,
    max_groups: int | None,
) -> list[str]:
    policy_groups, reference_groups = _group_maps(policy, reference_audit)
    selected: list[str] = []
    for group_id, group in policy_groups.items():
        reference_group = reference_groups.get(group_id)
        if not reference_group:
            continue
        if group.get("policy_class") not in REWRITABLE_POLICY_CLASSES:
            continue
        if not _candidate_paths_for_policy_group(group):
            continue
        selected.append(group_id)
        if max_groups is not None and len(selected) >= max_groups:
            break
    return selected


def _is_ignored_derived_reference(reference_path: str) -> bool:
    return any(
        reference_path.startswith(prefix)
        for prefix in IGNORED_DERIVED_REFERENCE_PREFIXES
    )


def _owner_paths_for_candidate(
    repo_root: Path,
    policy_groups_by_candidate: dict[str, dict[str, Any]],
    candidate_path: str,
) -> list[str]:
    policy_group = policy_groups_by_candidate.get(candidate_path, {})
    return _owner_paths_for_policy_candidate(
        repo_root,
        str(policy_group.get("policy_class", "")),
        candidate_path,
    )


def _is_derived_reference_file(repo_root: Path, path: Path) -> bool:
    try:
        relative_path = _relative_path(repo_root, path)
    except ValueError:
        return False
    return _is_ignored_derived_reference(relative_path)


def _iter_bounded_reference_files(
    repo_root: Path,
    owner_paths_by_candidate: dict[str, list[str]],
) -> tuple[list[Path], int]:
    files: list[Path] = []
    seen: set[Path] = set()
    skipped_derived = 0

    def add_file(path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        files.append(path)

    for owner_paths in owner_paths_by_candidate.values():
        for owner_path in owner_paths:
            add_file(repo_root / owner_path)

    for scan_root in BOUNDED_EXTERNAL_SCAN_ROOTS:
        root = repo_root / scan_root
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in REFERENCE_SCAN_SUFFIXES:
                continue
            if _is_derived_reference_file(repo_root, path):
                skipped_derived += 1
                continue
            add_file(path)
    return files, skipped_derived


def _bounded_reference_index(
    repo_root: Path,
    candidate_paths: set[str],
    owner_paths_by_candidate: dict[str, list[str]],
    *,
    max_file_bytes: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    references: dict[str, list[dict[str, Any]]] = {
        candidate_path: [] for candidate_path in candidate_paths
    }
    files, skipped_derived = _iter_bounded_reference_files(
        repo_root,
        owner_paths_by_candidate,
    )
    stats = {
        "scan_file_count": 0,
        "scan_skipped_large_file_count": 0,
        "scan_read_error_count": 0,
        "scan_skipped_derived_reference_file_count": skipped_derived,
    }
    for path in files:
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
        try:
            reference_path = _relative_path(repo_root, path)
        except ValueError:
            reference_path = path.as_posix()
        for candidate_path in sorted(candidate_paths):
            if candidate_path in text:
                references[candidate_path].append(
                    {
                        "candidate_path": candidate_path,
                        "reference_path": reference_path,
                        "reference_kind": "path_token",
                    }
                )
    return references, stats


def _exhaustive_reference_index(
    repo_root: Path,
    candidate_paths: set[str],
    *,
    max_file_bytes: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    from scripts.data_cleanup_restore_reference_audit import (  # noqa: PLC0415
        DEFAULT_SCAN_ROOTS,
        _reference_index,
    )

    references, stats = _reference_index(
        repo_root,
        candidate_paths,
        scan_roots=DEFAULT_SCAN_ROOTS,
        max_file_bytes=max_file_bytes,
        include_all_text_files=True,
    )
    stats["scan_skipped_derived_reference_file_count"] = 0
    return references, stats


def _build_item(
    repo_root: Path,
    policy_group: dict[str, Any],
    reference_group: dict[str, Any],
    references: dict[str, list[dict[str, Any]]],
    *,
    candidate_path: str,
    apply: bool,
    confirm_token: str | None,
) -> ReferenceRewriteItem:
    group_id = str(policy_group.get("group_id", ""))
    retained_path = str(policy_group.get("retained_paths", [""])[0])
    errors: list[str] = []
    owner_paths: list[str] = []
    references_rewritten = 0
    candidate_reclaim_bytes = 0

    policy_class = str(policy_group.get("policy_class", ""))
    if policy_class not in REWRITABLE_POLICY_CLASSES:
        errors.append("policy_class_not_rewritable")
    if str(policy_group.get("risk_level", "")) != "medium":
        errors.append("risk_level_not_medium")
    if _rewritable_artifact_suffix(
        policy_class,
        candidate_path,
    ) != _rewritable_artifact_suffix(
        policy_class,
        retained_path,
    ):
        errors.append("artifact_suffix_mismatch")

    candidate_abs = repo_root / candidate_path
    retained_abs = repo_root / retained_path
    if not candidate_abs.exists():
        errors.append("candidate_file_missing")
    if not retained_abs.exists():
        errors.append("retained_file_missing")
    if candidate_abs.exists() and retained_abs.exists():
        try:
            candidate_reclaim_bytes = candidate_abs.stat().st_size
            if candidate_abs.stat().st_size != retained_abs.stat().st_size:
                errors.append("size_mismatch")
            if _hash_file(candidate_abs) != _hash_file(retained_abs):
                errors.append("hash_mismatch")
        except OSError as exc:
            errors.append(f"file_read_failed:{exc}")

    owner_paths = _owner_paths_for_policy_candidate(
        repo_root,
        policy_class,
        candidate_path,
    )
    if not owner_paths:
        errors.append("owner_paths_not_inferred")
    for owner_path in owner_paths:
        if _load_json(repo_root / owner_path) is None:
            errors.append(f"owner_missing_or_invalid:{owner_path}")

    expected_owner_paths = set(owner_paths)
    actual_references = references.get(candidate_path, [])
    ignored_reference_count = sum(
        1
        for reference in actual_references
        if _is_ignored_derived_reference(str(reference.get("reference_path", "")))
    )
    unexpected_references = [
        str(reference.get("reference_path", ""))
        for reference in actual_references
        if str(reference.get("reference_path", "")) not in expected_owner_paths
        and not _is_ignored_derived_reference(
            str(reference.get("reference_path", ""))
        )
    ]
    if unexpected_references:
        errors.append(
            "unexpected_references:"
            + ",".join(sorted(set(unexpected_references))[:5])
        )
    referenced_candidate_paths = [
        str(path) for path in reference_group.get("referenced_candidate_paths", [])
    ]
    if candidate_path not in referenced_candidate_paths:
        errors.append("candidate_not_referenced_by_reference_audit")

    updated_payloads: dict[str, dict[str, Any]] = {}
    for owner_path in owner_paths:
        payload = _load_json(repo_root / owner_path)
        if payload is None:
            continue
        updated = deepcopy(payload)
        rewritten = _rewrite_reference_values(updated, candidate_path, retained_path)
        if rewritten > 0:
            references_rewritten += rewritten
            updated_payloads[owner_path] = updated
    if references_rewritten <= 0:
        errors.append("no_owner_references_rewritten")

    if errors:
        return ReferenceRewriteItem(
            group_id=group_id,
            policy_class=str(policy_group.get("policy_class", "")),
            risk_level=str(policy_group.get("risk_level", "")),
            status="blocked",
            action="blocked",
            delete_allowed=False,
            reclaimable_bytes=0,
            candidate_path=candidate_path,
            retained_path=retained_path,
            owner_paths=owner_paths,
            references_rewritten=references_rewritten,
            ignored_reference_count=ignored_reference_count,
            errors=errors,
            reason="reference rewrite preconditions failed",
        )

    if not apply:
        return ReferenceRewriteItem(
            group_id=group_id,
            policy_class=str(policy_group.get("policy_class", "")),
            risk_level=str(policy_group.get("risk_level", "")),
            status="would_rewrite_delete",
            action="dry_run_rewrite_delete",
            delete_allowed=False,
            reclaimable_bytes=candidate_reclaim_bytes,
            candidate_path=candidate_path,
            retained_path=retained_path,
            owner_paths=owner_paths,
            references_rewritten=references_rewritten,
            ignored_reference_count=ignored_reference_count,
            errors=[],
            reason="dry-run only; pass --apply with confirmation token",
        )

    if not _valid_confirm_token_for_policy_class(policy_class, confirm_token):
        return ReferenceRewriteItem(
            group_id=group_id,
            policy_class=str(policy_group.get("policy_class", "")),
            risk_level=str(policy_group.get("risk_level", "")),
            status="blocked_confirm_token_required",
            action="blocked",
            delete_allowed=False,
            reclaimable_bytes=0,
            candidate_path=candidate_path,
            retained_path=retained_path,
            owner_paths=owner_paths,
            references_rewritten=references_rewritten,
            ignored_reference_count=ignored_reference_count,
            errors=["invalid or missing confirmation token"],
            reason="apply requires explicit confirmation token",
        )

    try:
        for owner_path, updated in updated_payloads.items():
            _write_json(repo_root / owner_path, updated)
        candidate_abs.unlink()
    except OSError as exc:
        return ReferenceRewriteItem(
            group_id=group_id,
            policy_class=str(policy_group.get("policy_class", "")),
            risk_level=str(policy_group.get("risk_level", "")),
            status="blocked_apply_failed",
            action="blocked",
            delete_allowed=False,
            reclaimable_bytes=0,
            candidate_path=candidate_path,
            retained_path=retained_path,
            owner_paths=owner_paths,
            references_rewritten=references_rewritten,
            ignored_reference_count=ignored_reference_count,
            errors=[str(exc)],
            reason="failed while writing owner reports or deleting candidate",
        )

    return ReferenceRewriteItem(
        group_id=group_id,
        policy_class=str(policy_group.get("policy_class", "")),
        risk_level=str(policy_group.get("risk_level", "")),
        status="rewritten_deleted",
        action="rewrite_delete",
        delete_allowed=False,
        reclaimable_bytes=candidate_reclaim_bytes,
        candidate_path=candidate_path,
        retained_path=retained_path,
        owner_paths=owner_paths,
        references_rewritten=references_rewritten,
        ignored_reference_count=ignored_reference_count,
        errors=[],
        reason="owner report references rewritten to retained duplicate before delete",
    )


def build_reference_rewrite_report(
    policy: dict[str, Any],
    reference_audit: dict[str, Any],
    *,
    repo_root: Path,
    policy_json_path: Path | None = None,
    reference_audit_json_path: Path | None = None,
    max_groups: int | None = DEFAULT_MAX_GROUPS,
    apply: bool = False,
    confirm_token: str | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    exhaustive_scan: bool = False,
) -> dict[str, Any]:
    policy_groups, reference_groups = _group_maps(policy, reference_audit)
    selected_group_ids = _candidate_group_ids(
        policy,
        reference_audit,
        max_groups=max_groups,
    )
    selected_candidates_by_group = {
        group_id: [
            candidate_path
            for candidate_path in _candidate_paths_for_policy_group(
                policy_groups[group_id]
            )
            if candidate_path
            in {
                str(path)
                for path in reference_groups[group_id].get(
                    "referenced_candidate_paths",
                    [],
                )
            }
        ]
        for group_id in selected_group_ids
    }
    selected_candidates_by_group = {
        group_id: candidate_paths
        for group_id, candidate_paths in selected_candidates_by_group.items()
        if candidate_paths
    }
    selected_group_ids = [
        group_id
        for group_id in selected_group_ids
        if group_id in selected_candidates_by_group
    ]
    policy_groups_by_candidate = {
        candidate_path: policy_groups[group_id]
        for group_id, candidate_paths in selected_candidates_by_group.items()
        for candidate_path in candidate_paths
    }
    selected_candidate_paths = {
        candidate_path
        for candidate_paths in selected_candidates_by_group.values()
        for candidate_path in candidate_paths
    }
    owner_paths_by_candidate = {
        candidate_path: _owner_paths_for_candidate(
            repo_root,
            policy_groups_by_candidate,
            candidate_path,
        )
        for candidate_path in selected_candidate_paths
    }
    references, scan_stats = (
        _exhaustive_reference_index(
            repo_root,
            selected_candidate_paths,
            max_file_bytes=max_file_bytes,
        )
        if exhaustive_scan
        else _bounded_reference_index(
            repo_root,
            selected_candidate_paths,
            owner_paths_by_candidate,
            max_file_bytes=max_file_bytes,
        )
    )
    items = []
    for group_id in selected_group_ids:
        for candidate_path in selected_candidates_by_group[group_id]:
            items.append(
                asdict(
                    _build_item(
                        repo_root,
                        policy_groups[group_id],
                        reference_groups[group_id],
                        references,
                        candidate_path=candidate_path,
                        apply=apply,
                        confirm_token=confirm_token,
                    )
                )
            )
    status_summary: dict[str, int] = {}
    for item in items:
        status = str(item["status"])
        status_summary[status] = status_summary.get(status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_policy_json": str(policy_json_path) if policy_json_path else None,
        "source_reference_audit_json": (
            str(reference_audit_json_path) if reference_audit_json_path else None
        ),
        "root": str(repo_root),
        "apply_requested": apply,
        "confirm_token_valid": _valid_confirm_token_for_groups(
            policy_groups,
            selected_group_ids,
            confirm_token,
        ),
        "execution_performed": apply
        and status_summary.get("rewritten_deleted", 0) > 0,
        "delete_candidate_count": 0,
        "summary": {
            **scan_stats,
            "scan_mode": (
                "exhaustive_all_text_files"
                if exhaustive_scan
                else "bounded_owner_external"
            ),
            "selected_group_count": len(selected_group_ids),
            "selected_candidate_path_count": len(selected_candidate_paths),
            "would_rewrite_delete_count": status_summary.get(
                "would_rewrite_delete",
                0,
            ),
            "rewritten_deleted_count": status_summary.get("rewritten_deleted", 0),
            "blocked_count": sum(
                count
                for status, count in status_summary.items()
                if status.startswith("blocked") or status == "blocked"
            ),
            "planned_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in items
                if item["status"] == "would_rewrite_delete"
            ),
            "rewritten_deleted_reclaim_bytes": sum(
                int(item["reclaimable_bytes"])
                for item in items
                if item["status"] == "rewritten_deleted"
            ),
            "references_rewritten_count": sum(
                int(item["references_rewritten"])
                for item in items
                if item["status"] in {"would_rewrite_delete", "rewritten_deleted"}
            ),
            "ignored_reference_count": sum(
                int(item["ignored_reference_count"])
                for item in items
            ),
            "status_summary": status_summary,
        },
        "items": items,
    }


def render_reference_rewrite_markdown(
    report: dict[str, Any],
    *,
    max_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> str:
    summary = report.get("summary", {})
    items = report.get("items", [])
    visible_items = items[:max_groups]
    lines = [
        "# Data Cleanup Reference Rewrite Report",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Apply requested: `{report.get('apply_requested', False)}`",
        f"- Execution performed: `{report.get('execution_performed', False)}`",
        f"- Selected groups: `{summary.get('selected_group_count', 0)}`",
        f"- Would rewrite/delete: `{summary.get('would_rewrite_delete_count', 0)}`",
        f"- Rewritten/deleted: `{summary.get('rewritten_deleted_count', 0)}`",
        f"- Blocked: `{summary.get('blocked_count', 0)}`",
        f"- Selected candidate paths: `{summary.get('selected_candidate_path_count', 0)}`",
        f"- References rewritten: `{summary.get('references_rewritten_count', 0)}`",
        f"- Planned reclaim bytes: `{summary.get('planned_reclaim_bytes', 0)}`",
        (
            "- Rewritten/deleted reclaim bytes: "
            f"`{summary.get('rewritten_deleted_reclaim_bytes', 0)}`"
        ),
        "",
        "## Items",
        "",
        "| Group | Status | Candidate | Retained | Reclaim Bytes | Rewrites |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in visible_items:
        lines.append(
            "| {group} | {status} | `{candidate}` | `{retained}` | {bytes} | {rewrites} |".format(
                group=item.get("group_id", ""),
                status=item.get("status", ""),
                candidate=item.get("candidate_path", ""),
                retained=item.get("retained_path", ""),
                bytes=item.get("reclaimable_bytes", 0),
                rewrites=item.get("references_rewritten", 0),
            )
        )
    if len(items) > len(visible_items):
        lines.append("")
        lines.append(
            f"_Item table truncated to {len(visible_items)} of {len(items)} rows._"
        )
    return "\n".join(lines) + "\n"


def write_reference_rewrite_report(
    *,
    root: Path | None = None,
    policy_json_path: Path | None = None,
    reference_audit_json_path: Path | None = None,
    output_dir: Path | None = None,
    max_groups: int | None = DEFAULT_MAX_GROUPS,
    apply: bool = False,
    confirm_token: str | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    exhaustive_scan: bool = False,
    max_markdown_groups: int = DEFAULT_MAX_MARKDOWN_GROUPS,
) -> dict[str, str]:
    repo_root = get_repo_root(root)
    policy_path = policy_json_path or _latest_json(
        repo_root,
        "reports/project_cleanup/"
        "data_cleanup_restore_policy_*/data_cleanup_restore_policy.json",
    )
    reference_path = reference_audit_json_path or _latest_json(
        repo_root,
        "reports/project_cleanup/"
        "data_cleanup_restore_reference_audit_*/"
        "data_cleanup_restore_reference_audit.json",
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    reference_audit = json.loads(reference_path.read_text(encoding="utf-8"))
    report = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=repo_root,
        policy_json_path=policy_path,
        reference_audit_json_path=reference_path,
        max_groups=max_groups,
        apply=apply,
        confirm_token=confirm_token,
        max_file_bytes=max_file_bytes,
        exhaustive_scan=exhaustive_scan,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_dir or (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"data_cleanup_reference_rewrite_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_cleanup_reference_rewrite.json"
    md_path = out_dir / "data_cleanup_reference_rewrite.md"
    _write_json(json_path, report)
    md_path.write_text(
        render_reference_rewrite_markdown(
            report,
            max_groups=max_markdown_groups,
        ),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite same-symbol restore-source owner references before "
            "deleting duplicate raw backup or artifact files."
        )
    )
    parser.add_argument("--policy-json", type=Path, default=None)
    parser.add_argument("--reference-audit-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-groups", type=int, default=DEFAULT_MAX_GROUPS)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-markdown-groups", type=int, default=DEFAULT_MAX_MARKDOWN_GROUPS)
    parser.add_argument(
        "--exhaustive-scan",
        action="store_true",
        help=(
            "Scan every supported text file under restore-reference roots. "
            "Default is a bounded owner-report plus external-evidence scan."
        ),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-token",
        default=None,
        help=(
            "Required token for --apply. Cleaning artifacts accept "
            f"{CLEANING_CONFIRM_TOKEN}; all supported restore-source "
            f"classes accept {RESTORE_SOURCE_CONFIRM_TOKEN}."
        ),
    )
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    max_groups = args.max_groups
    if max_groups is not None and max_groups < 0:
        max_groups = None
    paths = write_reference_rewrite_report(
        root=args.root,
        policy_json_path=args.policy_json,
        reference_audit_json_path=args.reference_audit_json,
        output_dir=args.output_dir,
        max_groups=max_groups,
        apply=args.apply,
        confirm_token=args.confirm_token,
        max_file_bytes=max(0, args.max_file_bytes),
        exhaustive_scan=args.exhaustive_scan,
        max_markdown_groups=max(0, args.max_markdown_groups),
    )
    payload = _load_json(Path(paths["json"])) or {}
    summary = payload.get("summary") or {}
    mode = "apply" if args.apply else "dry-run"
    print(f"data cleanup reference rewrite mode: {mode}")
    print(f"workspace root: {payload.get('root')}")
    print(f"selected groups: {summary.get('selected_group_count', 0)}")
    print(f"would rewrite/delete: {summary.get('would_rewrite_delete_count', 0)}")
    print(f"rewritten/deleted: {summary.get('rewritten_deleted_count', 0)}")
    print(f"blocked: {summary.get('blocked_count', 0)}")
    print("data cleanup reference rewrite manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
