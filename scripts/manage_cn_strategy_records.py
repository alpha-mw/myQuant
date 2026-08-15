#!/usr/bin/env python3
"""Offline manager for the registered CN strategy-record store.

Every mutating command is explicit and pointer-CAS guarded.  Archive rehearsal
only copies bytes and verifies a complete restore; it never retires, moves, or
deletes source records.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.strategy_records.store import (  # noqa: E402
    ARCHIVE_LOCATOR_SCHEMA,
    ARCHIVE_MANIFEST_SCHEMA,
    ARCHIVE_RESTORE_RECEIPT_SCHEMA,
    CATALOG_SCHEMA_V1,
    CATALOG_SCHEMA_V2,
    CATALOG_SCHEMA_V3,
    NEW_RECORD_MAX_FILE_BYTES,
    NEW_RECORD_MAX_FILES,
    NEW_RECORD_MAX_TOTAL_BYTES,
    NO_ACTION_RECEIPT_MAX_BYTES,
    STORE_DIRECTORY,
    StrategyRecordConflict,
    StrategyRecordStoreError,
    archive_final_root,
    bootstrap_catalog,
    canonical_json_bytes,
    content_sha256,
    load_registered_catalog,
    load_archive_binding,
    project_root_for_record_root,
    publish_catalog,
    regular_file_sha256,
    reselect_catalog,
)
from quant_investor.strategy_records.performance import (  # noqa: E402
    MAX_PERFORMANCE_JSON_BYTES,
    MAX_PERFORMANCE_PARQUET_BYTES,
    MONEY_QUANTUM,
    PERFORMANCE_MIGRATION_RECEIPT_SCHEMA,
    UNIT_QUANTUM,
    assert_private_tmp,
    apply_flow_neutral_unitization,
    build_lineage_index,
    build_manifest as build_performance_manifest,
    build_owner_declaration as build_performance_owner_declaration,
    build_performance_history_ref,
    build_seed_rows,
    decimal_text,
    extend_performance_rows,
    immutable_write,
    load_performance_history,
    normalize_registered_projection,
    read_performance_parquet,
    seal_semantic,
    validate_lineage_index,
    validate_cash_flow_artifact,
    validate_manifest as validate_performance_manifest,
    validate_owner_declaration as validate_performance_owner_declaration,
    validate_safe_regular_0600,
    validate_semantic as validate_performance_semantic,
    write_deterministic_parquet,
)

# Kept in this operational surface because the reader does not allocate archive
# extraction space.
ARCHIVE_MAX_MEMBERS = 10_000
ARCHIVE_MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
ARCHIVE_FREE_SPACE_RESERVE_BYTES = 256 * 1024 * 1024
_RECORD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STRICT_RUN_ID = re.compile(r"^[0-9]{8}_[0-9]{4}$")
_ARCHIVE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TRANSACTION_PLAN_SCHEMA = "myquant.strategy_record_quarantine_plan.v1"
TRANSACTION_EVENT_SCHEMA = "myquant.strategy_record_quarantine_event.v1"


def _sealed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = content_sha256(result)
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exact_once(path: Path, value: dict[str, Any]) -> str:
    raw = canonical_json_bytes(_sealed(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        existing = path.read_bytes()
        if existing != raw:
            raise StrategyRecordConflict("immutable transaction identity collision") from None
        return _sha(existing)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StrategyRecordStoreError("short immutable transaction write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if path.read_bytes() != raw:
        raise StrategyRecordStoreError("immutable transaction readback mismatch")
    return _sha(raw)


@contextmanager
def _operation_lock(record_root: str | os.PathLike[str]):
    root = Path(record_root)
    store = root / STORE_DIRECTORY
    store.mkdir(parents=True, exist_ok=True)
    lock_path = store / ".operation.v2.lock"
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StrategyRecordStoreError("operation lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: str | None) -> str:
    if value:
        return value
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _record_id(value: str) -> str:
    if not isinstance(value, str) or _RECORD_ID.fullmatch(value) is None:
        raise StrategyRecordStoreError("record_id is invalid")
    return value


def _strict_run_id(value: str) -> str:
    record_id = _record_id(value)
    if _STRICT_RUN_ID.fullmatch(record_id) is None:
        raise StrategyRecordStoreError(
            "new record_id must use YYYYMMDD_HHMM"
        )
    return record_id


def _regular_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise StrategyRecordStoreError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise StrategyRecordStoreError(f"{label} must be a real directory")
    return metadata


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise StrategyRecordStoreError("path is not canonical relative POSIX")
    return value


def _file_sha(path: Path, size: int) -> str:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StrategyRecordStoreError(
            "inventory file is unavailable or unsafe"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != size
        ):
            raise StrategyRecordStoreError(
                "inventory requires regular single-link files"
            )
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = lambda item: (  # noqa: E731
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if remaining or identity(before) != identity(after):
            raise StrategyRecordStoreError(
                "inventory file changed during read"
            )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def build_inventory(
    directory: Path,
    *,
    enforce_new_record_budget: bool,
) -> dict[str, Any]:
    """Seal a deterministic path/type/size/SHA inventory."""
    _regular_directory(directory, label="record directory")
    rows: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0
    casefold_paths: set[str] = set()
    for current, dirnames, filenames in os.walk(
        directory, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for name in [*dirnames, *filenames]:
            child = current_path / name
            relative = child.relative_to(directory).as_posix()
            _safe_relative(relative)
            folded = relative.casefold()
            if folded in casefold_paths:
                raise StrategyRecordStoreError(
                    "casefold-colliding inventory path"
                )
            casefold_paths.add(folded)
            metadata = os.lstat(child)
            if stat.S_ISLNK(metadata.st_mode):
                raise StrategyRecordStoreError("inventory symlink rejected")
            if stat.S_ISDIR(metadata.st_mode):
                rows.append(
                    {
                        "path": relative,
                        "type": "directory",
                        "size": 0,
                        "sha256": None,
                    }
                )
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StrategyRecordStoreError(
                    "inventory requires regular files without hard links"
                )
            if (
                enforce_new_record_budget
                and metadata.st_size > NEW_RECORD_MAX_FILE_BYTES
            ):
                raise StrategyRecordStoreError(
                    "new record file exceeds byte budget"
                )
            file_count += 1
            total_bytes += metadata.st_size
            if enforce_new_record_budget and file_count > NEW_RECORD_MAX_FILES:
                raise StrategyRecordStoreError(
                    "new record exceeds file-count budget"
                )
            if (
                enforce_new_record_budget
                and total_bytes > NEW_RECORD_MAX_TOTAL_BYTES
            ):
                raise StrategyRecordStoreError(
                    "new record exceeds total byte budget"
                )
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": metadata.st_size,
                    "sha256": _file_sha(child, metadata.st_size),
                }
            )
    inventory_sha = _sha(canonical_json_bytes(rows))
    return {
        "inventory": rows,
        "inventory_sha256": inventory_sha,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _record_entry(
    root: Path,
    relative_path: str,
    *,
    sealed_at: str,
    enforce_new_record_budget: bool,
) -> dict[str, Any]:
    relative = _safe_relative(relative_path)
    if (
        relative.startswith(f"{STORE_DIRECTORY}/")
        or relative == STORE_DIRECTORY
    ):
        raise StrategyRecordStoreError(
            "record path cannot be inside _record_store"
        )
    path = root / relative
    inventory = build_inventory(
        path, enforce_new_record_budget=enforce_new_record_budget
    )
    return {
        "record_id": _record_id(path.name),
        "relative_path": relative,
        "state": "ONLINE",
        "storage_state": "ONLINE",
        "sealed_at": sealed_at,
        **inventory,
    }


def _live_catalog_entries(root: Path, *, sealed_at: str) -> list[dict[str, Any]]:
    """Classify every live top-level child exactly once for registration."""

    records: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda value: value.name):
        if child.name == STORE_DIRECTORY:
            continue
        metadata = os.lstat(child)
        if stat.S_ISLNK(metadata.st_mode):
            raise StrategyRecordStoreError(
                f"top-level symlink is forbidden: {child.name}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            inventory = build_inventory(
                child, enforce_new_record_budget=False
            )
            strict = _STRICT_RUN_ID.fullmatch(child.name) is not None
            records.append(
                {
                    "record_id": _record_id(
                        child.name
                        if _RECORD_ID.fullmatch(child.name)
                        else "aux-" + child.name.lstrip("._")
                    ),
                    "relative_path": child.name,
                    "state": (
                        "ONLINE" if strict else "NONSTANDARD_RESEARCH_OUTPUT"
                    ),
                    "storage_state": (
                        "ONLINE" if strict else "NONSTANDARD_RESEARCH_OUTPUT"
                    ),
                    "record_class": (
                        "LEGACY_STRICT"
                        if strict
                        else "NONSTANDARD_RESEARCH_OUTPUT"
                    ),
                    "history_eligible": strict,
                    "archive_eligible": bool(
                        strict and child.name < "20260701_0000"
                    ),
                    "sealed_at": sealed_at,
                    **inventory,
                }
            )
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StrategyRecordStoreError(
                f"unsupported top-level object: {child.name}"
            )
        records.append(
            {
                "record_id": _record_id(child.name.lstrip(".") or "dot-file"),
                "relative_path": child.name,
                "state": "AUXILIARY_ROOT_FILE",
                "storage_state": "AUXILIARY_ROOT_FILE",
                "record_class": "AUXILIARY_ROOT_FILE",
                "history_eligible": False,
                "archive_eligible": False,
                "sealed_at": sealed_at,
                "file_count": 1,
                "total_bytes": metadata.st_size,
                "inventory": [
                    {
                        "path": child.name,
                        "type": "file",
                        "size": metadata.st_size,
                        "sha256": _file_sha(child, metadata.st_size),
                    }
                ],
            }
        )
        records[-1]["inventory_sha256"] = _sha(
            canonical_json_bytes(records[-1]["inventory"])
        )
    return records


def _attach_dashboard_closure(
    records: list[dict[str, Any]],
    projection: dict[str, Any],
    *,
    root: Path,
    project_root: Path,
) -> None:
    valid = projection.get("valid_records")
    if not isinstance(valid, list):
        raise StrategyRecordStoreError(
            "dashboard projection valid_records is missing"
        )
    by_id = {row["record_id"]: row for row in records}
    historical = projection.get("historical_records")
    if not isinstance(historical, list):
        raise StrategyRecordStoreError(
            "dashboard projection historical_records is missing"
        )
    accepted_ids = {
        item.get("record")
        for item in [*valid, *historical]
        if isinstance(item, dict)
    }
    for record in records:
        if record.get("state") == "ONLINE":
            record["history_eligible"] = record.get("record_id") in accepted_ids
    for projected in valid:
        if not isinstance(projected, dict):
            raise StrategyRecordStoreError(
                "dashboard projection contains an invalid record"
            )
        record_id = projected.get("record")
        catalog_row = by_id.get(record_id)
        if catalog_row is None:
            raise StrategyRecordStoreError(
                f"dashboard projection record is not ONLINE: {record_id}"
            )
        if catalog_row.get("state") == "ARCHIVED":
            continue
        if catalog_row.get("state") != "ONLINE":
            raise StrategyRecordStoreError(
                f"dashboard projection record is not ONLINE: {record_id}"
            )
        for key in (
            "manifest_path",
            "manual_manifest_path",
            "ledger_path",
            "pnl_path",
        ):
            value = projected.get(key)
            if value is None:
                continue
            try:
                resolved = (project_root / str(value)).resolve(strict=True)
                catalog_row[key] = resolved.relative_to(root.resolve()).as_posix()
            except (OSError, RuntimeError, ValueError) as exc:
                raise StrategyRecordStoreError(
                    f"dashboard closure {key} escapes record root"
                ) from exc
        for key in (
            "manifest_sha256",
            "manual_manifest_sha256",
            "ledger_sha256",
            "pnl_sha256",
            "financial_state_sha256",
        ):
            if projected.get(key) is not None:
                catalog_row[key] = projected[key]
        catalog_row["evidence_status"] = projected.get(
            "evidence_status", "HASH_VERIFIED"
        )
        catalog_row["summary"] = {
            "symbols": [
                item.get("symbol")
                for item in projected.get("positions", [])
                if isinstance(item, dict) and item.get("symbol")
            ],
            "actions": [],
        }


def _normalize_dashboard_projection_source_refs(
    projection: dict[str, Any],
) -> None:
    for key in ("valid_records", "historical_records"):
        rows = projection.get(key)
        if not isinstance(rows, list):
            raise StrategyRecordStoreError(
                "Dashboard projection is incomplete"
            )
        for row in rows:
            if not isinstance(row, dict) or not isinstance(
                row.get("source_refs"), list
            ):
                raise StrategyRecordStoreError(
                    "Dashboard projection source refs are invalid"
                )
            refs_by_path: dict[str, str] = {}
            for ref in row["source_refs"]:
                path = ref.get("path") if isinstance(ref, dict) else None
                digest = ref.get("sha256") if isinstance(ref, dict) else None
                if not isinstance(path, str) or not isinstance(digest, str):
                    raise StrategyRecordStoreError(
                        "Dashboard projection source ref is invalid"
                    )
                previous = refs_by_path.get(path)
                if previous is not None and previous != digest:
                    raise StrategyRecordStoreError(
                        "Dashboard projection source refs conflict"
                    )
                refs_by_path[path] = digest
            row["source_refs"] = [
                {"path": path, "sha256": refs_by_path[path]}
                for path in sorted(refs_by_path)
            ]


def _legacy_record_paths(root: Path, requested: list[str]) -> list[str]:
    if requested:
        return [_safe_relative(value) for value in requested]
    result: list[str] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.name == STORE_DIRECTORY or child.name.startswith("."):
            continue
        metadata = os.lstat(child)
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
            metadata.st_mode
        ):
            result.append(child.name)
    return result


def _pointer_sha(root: Path) -> str:
    path = root / STORE_DIRECTORY / "current.v1.json"
    try:
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StrategyRecordStoreError("pointer is unsafe")
        return _sha(path.read_bytes())
    except OSError as exc:
        raise StrategyRecordStoreError("pointer is unavailable") from exc


def _orphans(root: Path, catalog: dict[str, Any] | None) -> list[str]:
    registered = {
        str(record["relative_path"]).split("/", 1)[0]
        for record in (catalog or {}).get("records", [])
        if record.get("state", record.get("storage_state")) == "ONLINE"
    }
    return [
        child.name
        for child in sorted(root.iterdir(), key=lambda path: path.name)
        if child.name != STORE_DIRECTORY
        and not child.name.startswith(".")
        and stat.S_ISDIR(os.lstat(child).st_mode)
        and child.name not in registered
    ]


def command_inventory(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root)
    _regular_directory(root, label="record root")
    loaded = load_registered_catalog(root)
    pointer, catalog = loaded if loaded is not None else (None, None)
    result = {
        "registered": loaded is not None,
        "pointer_sha256": _pointer_sha(root) if loaded is not None else None,
        "active_record_id": (
            pointer.get("active_record_id") if pointer else None
        ),
        "previous_record_id": (
            pointer.get("previous_record_id") if pointer else None
        ),
        "record_count": catalog.get("record_count", 0) if catalog else 0,
        "records": catalog.get("records", []) if catalog else [],
        "orphan_record_dirs": _orphans(root, catalog),
        "orphans_preserved": True,
    }
    return result


def _load_projection(path: str | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyRecordStoreError(
            "dashboard projection JSON is invalid"
        ) from exc


def command_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root)
    sealed_at = _timestamp(args.published_at)
    paths = _legacy_record_paths(root, args.record_dir)
    records = [
        _record_entry(
            root,
            path,
            sealed_at=sealed_at,
            enforce_new_record_budget=False,
        )
        for path in paths
    ]
    active = args.active_record_id or (
        records[-1]["record_id"] if records else None
    )
    previous = args.previous_record_id
    if previous is None and len(records) > 1:
        previous = records[-2]["record_id"]
    return bootstrap_catalog(
        root,
        records=records,
        dashboard_projection=_load_projection(args.dashboard_projection_json),
        active_record_id=active,
        previous_record_id=previous,
        generation_id=args.generation_id,
        published_at=sealed_at,
    )


def command_bootstrap_live(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve()
    sealed_at = _timestamp(args.published_at)
    if args.record_dir:
        raise StrategyRecordStoreError(
            "bootstrap-live classifies the complete top-level inventory"
        )
    records = _live_catalog_entries(root, sealed_at=sealed_at)
    projection = _load_projection(args.dashboard_projection_json)
    if projection is None and args.project_root:
        try:
            from scripts.cn_dashboard_common import (
                build_dashboard_catalog_projection,
            )
        except ImportError as exc:
            raise StrategyRecordStoreError(
                "dashboard projection builder is unavailable"
            ) from exc
        projection = build_dashboard_catalog_projection(
            root, Path(args.project_root)
        )
    if not isinstance(projection, dict):
        raise StrategyRecordStoreError(
            "bootstrap-live requires a Dashboard projection"
        )
    valid = projection.get("valid_records")
    if not isinstance(valid, list) or len(valid) < 2:
        raise StrategyRecordStoreError(
            "bootstrap-live Dashboard projection has fewer than two valid records"
        )
    active = args.active_record_id or valid[-1].get("record")
    previous = args.previous_record_id or valid[-2].get("record")
    if args.expected_current_id and active != args.expected_current_id:
        raise StrategyRecordStoreError(
            "active record does not match expectation"
        )
    if args.expected_previous_id and previous != args.expected_previous_id:
        raise StrategyRecordStoreError(
            "previous record does not match expectation"
        )
    if args.expected_inventory_sha:
        by_id = {record["record_id"]: record for record in records}
        active_record = by_id.get(active)
        if (
            active_record is None
            or active_record["inventory_sha256"] != args.expected_inventory_sha
        ):
            raise StrategyRecordStoreError(
                "active inventory SHA does not match expectation"
            )
    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    _attach_dashboard_closure(
        records,
        projection,
        root=root,
        project_root=project_root,
    )
    return bootstrap_catalog(
        root,
        records=records,
        dashboard_projection=projection,
        active_record_id=active,
        previous_record_id=previous,
        generation_id=args.generation_id,
        published_at=sealed_at,
    )


def command_publish(args: argparse.Namespace) -> dict[str, Any]:
    try:
        records = json.loads(
            Path(args.records_json).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyRecordStoreError("records JSON is invalid") from exc
    if not isinstance(records, list):
        raise StrategyRecordStoreError("records JSON must be a list")
    return publish_catalog(
        args.record_root,
        expected_pointer_sha256=args.expected_pointer_sha,
        records=records,
        dashboard_projection=_load_projection(args.dashboard_projection_json),
        active_record_id=args.active_record_id,
        previous_record_id=args.previous_record_id,
        generation_id=args.generation_id,
        published_at=_timestamp(args.published_at),
    )


def command_stage_init(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root)
    if load_registered_catalog(root) is None:
        raise StrategyRecordStoreError(
            "stage-init requires a registered catalog"
        )
    record_id = _strict_run_id(args.record_id)
    stage_root = root / STORE_DIRECTORY / "staging"
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_metadata = _regular_directory(
        stage_root, label="strategy-record staging root"
    )
    if stage_metadata.st_dev != os.lstat(root).st_dev:
        raise StrategyRecordStoreError(
            "staging directory is not on record filesystem"
        )
    stage = stage_root / record_id
    try:
        stage.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        _regular_directory(stage, label="staging record")
        created = False
    return {
        "record_id": record_id,
        "staging_dir": str(stage),
        "created": created,
    }


def _same_inventory(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in (
            "inventory_sha256",
            "file_count",
            "total_bytes",
            "inventory",
        )
    )


def _reject_disabled_ledger_candidates(record_dir: Path) -> None:
    """Reject legacy ledger authorities before inventory hashing can open them."""

    for name in ("ledger.csv", "ledger_after_manual_switch.csv"):
        if os.path.lexists(record_dir / name):
            raise StrategyRecordStoreError(
                "catalog v3 publication refuses a legacy ledger candidate"
            )


def _record_root_relative(
    *, project: Path, root: Path, project_relative: Any, label: str
) -> str:
    if not isinstance(project_relative, str):
        raise StrategyRecordStoreError(f"{label} is absent")
    _safe_relative(project_relative)
    try:
        resolved = (project / project_relative).resolve(strict=True)
        return resolved.relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError) as exc:
        raise StrategyRecordStoreError(f"{label} escapes the record root") from exc


def _integer_positions(strict_record: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in strict_record.get("positions") or []:
        if not isinstance(row, dict):
            raise StrategyRecordStoreError("cash-flow position row is invalid")
        symbol = row.get("symbol")
        shares = row.get("shares")
        if not isinstance(symbol, str):
            raise StrategyRecordStoreError("cash-flow position symbol is invalid")
        try:
            numeric = float(shares)
            integral = int(numeric)
        except (TypeError, ValueError, OverflowError) as exc:
            raise StrategyRecordStoreError(
                "cash-flow position quantity is not integral"
            ) from exc
        if float(integral) != numeric:
            raise StrategyRecordStoreError(
                "cash-flow position quantity is not integral"
            )
        result[symbol] = integral
    return result


def _load_cash_flow_declaration(
    value: str, *, project: Path
) -> tuple[dict[str, Any], str]:
    try:
        path_text, expected_sha = value.rsplit("=", 1)
    except ValueError as exc:
        raise StrategyRecordStoreError(
            "cash-flow artifact must be PROJECT_RELATIVE_PATH=SHA256"
        ) from exc
    relative = _safe_relative(path_text)
    declaration, _ = _read_candidate_json(
        project / relative,
        expected_sha256=expected_sha,
        label="performance cash-flow declaration",
    )
    return declaration, expected_sha


def _seal_publish_catalog_v3(
    *,
    args: argparse.Namespace,
    root: Path,
    target: Path,
    record_id: str,
    sealed_at: str,
    pointer: dict[str, Any],
    catalog: dict[str, Any],
    records: list[dict[str, Any]],
    new_record: dict[str, Any],
) -> dict[str, Any]:
    """Advance record, lineage, performance, and catalog in one CAS candidate."""

    if not getattr(args, "project_root", None):
        raise StrategyRecordStoreError("catalog v3 seal-publish requires project_root")
    if not getattr(args, "generation_id", None):
        raise StrategyRecordStoreError("catalog v3 seal-publish requires generation_id")
    performance_generation = getattr(args, "performance_generation_id", None)
    if not performance_generation:
        raise StrategyRecordStoreError(
            "catalog v3 seal-publish requires performance_generation_id"
        )
    project = Path(args.project_root).resolve(strict=True)
    if project_root_for_record_root(root).resolve(strict=True) != project:
        raise StrategyRecordStoreError("project root does not bind the governed record root")
    try:
        from scripts.cn_dashboard_common import (
            DashboardInputError,
            load_json,
            stable_read,
            validate_record,
        )
    except ImportError as exc:
        raise StrategyRecordStoreError("Dashboard validation is unavailable") from exc

    manual_artifact = stable_read(target / "manual_execution_manifest.json", project)
    manual = load_json(manual_artifact)
    if not isinstance(manual, dict):
        raise StrategyRecordStoreError("manual manifest is not an object")
    effective_ledger = manual.get("effective_manual_ledger_path") or manual.get(
        "next_ledger_path"
    )
    if effective_ledger != "ledger_after_manual_switch.parquet":
        raise StrategyRecordStoreError(
            "catalog v3 financial state requires the exact Parquet ledger"
        )
    try:
        strict_record = validate_record(target, root, project)
    except DashboardInputError as exc:
        raise StrategyRecordStoreError(
            f"sealed record did not pass current-record validation: {exc}"
        ) from exc
    if strict_record.get("record") != record_id:
        raise StrategyRecordStoreError("sealed record identity drifted")
    if (
        strict_record.get("execution_kind") != "applied_effective_ledger"
        and strict_record.get("official_valuation") is not True
    ):
        raise StrategyRecordStoreError(
            "new no-trade financial state must be an official valuation"
        )

    for path_key in ("manifest_path", "manual_manifest_path", "ledger_path", "pnl_path"):
        new_record[path_key] = _record_root_relative(
            project=project,
            root=root,
            project_relative=strict_record.get(path_key),
            label=f"new record {path_key}",
        )
    if PurePosixPath(new_record["ledger_path"]).name != (
        "ledger_after_manual_switch.parquet"
    ):
        raise StrategyRecordStoreError("catalog v3 effective ledger is not Parquet")
    for sha_key in (
        "manifest_sha256",
        "manual_manifest_sha256",
        "ledger_sha256",
        "pnl_sha256",
        "financial_state_sha256",
    ):
        value = strict_record.get(sha_key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise StrategyRecordStoreError(f"new record {sha_key} is invalid")
        new_record[sha_key] = value
    new_record["history_eligible"] = True
    new_record["evidence_status"] = "HASH_VERIFIED"
    new_record["summary"] = {
        "symbols": [row["symbol"] for row in strict_record["positions"]],
        "actions": [],
    }
    records = [
        new_record if row.get("record_id") == record_id else row for row in records
    ]

    active_record_id = pointer.get("active_record_id")
    if not isinstance(active_record_id, str):
        raise StrategyRecordStoreError("catalog v3 active record is absent")
    if strict_record.get("source_record") != active_record_id:
        raise StrategyRecordStoreError(
            "new financial state must bind the pointer-selected active record"
        )
    performance = load_performance_history(root, catalog["performance_history_ref"])
    parent_rows = performance["rows"]
    parent_last = parent_rows[-1]
    if parent_last["record_id"] != active_record_id:
        raise StrategyRecordStoreError("active performance row drifted")
    same_date = strict_record.get("data_date") == parent_last["valuation_date"]
    status_text = str(strict_record.get("execution_status") or "").lower()
    explicit_correction = same_date and (
        "correction" in status_text or strict_record.get("funding_correction") is not None
    )
    if same_date and not explicit_correction:
        raise StrategyRecordStoreError("SAME_DATE_PERFORMANCE_CONFLICT")
    if strict_record.get("funding_correction") is not None:
        raise StrategyRecordStoreError(
            "PERFORMANCE_EXTERNAL_FLOW_CLOSURE_INCOMPLETE:flow correction requires "
            "an exact correction contract"
        )

    flow_refs = list(getattr(args, "cash_flow_artifact", None) or [])
    funding = strict_record.get("funding")
    post_flow_units = None
    external_flow_amount = Decimal("0.0000")
    if funding is None:
        if flow_refs:
            raise StrategyRecordStoreError(
                "cash-flow artifact is present without registered funding evidence"
            )
    else:
        if len(flow_refs) != 1:
            raise StrategyRecordStoreError(
                "PERFORMANCE_EXTERNAL_FLOW_CLOSURE_INCOMPLETE:one exact cash-flow "
                "artifact is required"
            )
        current_by_id = {
            row["record_id"]: row
            for row in catalog["records"]
            if row.get("state", row.get("storage_state")) == "ONLINE"
        }
        current_catalog_row = current_by_id.get(active_record_id)
        if not isinstance(current_catalog_row, dict):
            raise StrategyRecordStoreError("active cash-flow record is not ONLINE")
        if PurePosixPath(str(current_catalog_row.get("ledger_path"))).name != (
            "ledger_after_manual_switch.parquet"
        ):
            raise StrategyRecordStoreError("active cash-flow ledger is not Parquet")
        try:
            pre_record = validate_record(root / active_record_id, root, project)
        except DashboardInputError as exc:
            raise StrategyRecordStoreError(
                f"cash-flow pre-state validation failed: {exc}"
            ) from exc
        declaration, _ = _load_cash_flow_declaration(flow_refs[0], project=project)
        pre_row = {
            "record_id": active_record_id,
            "raw_nav_cny": pre_record["accounting"]["total_value_after"],
            "financial_state_sha256": pre_record["financial_state_sha256"],
            "manual_manifest_path": pre_record["manual_manifest_path"],
            "manual_manifest_sha256": pre_record["manual_manifest_sha256"],
        }
        post_row = {
            "record_id": record_id,
            "raw_nav_cny": strict_record["accounting"]["total_value_after"],
            "financial_state_sha256": strict_record["financial_state_sha256"],
            "manual_manifest_path": strict_record["manual_manifest_path"],
            "manual_manifest_sha256": strict_record["manual_manifest_sha256"],
        }
        external_flow_amount = validate_cash_flow_artifact(
            declaration,
            pre_row=pre_row,
            post_row=post_row,
            pre_positions=_integer_positions(pre_record),
            post_positions=_integer_positions(strict_record),
        )
        if float(external_flow_amount) != float(funding.get("amount")):
            raise StrategyRecordStoreError("cash-flow amount does not match funding closure")
        post_flow_units, _ = apply_flow_neutral_unitization(
            pre_nav=parent_last["raw_nav_cny"],
            pre_units=parent_last["unit_count"],
            pre_unit_nav=parent_last["unit_nav"],
            amount=external_flow_amount,
        )

    next_rows = extend_performance_rows(
        parent_rows,
        strict_record=strict_record,
        manual_manifest_sha256=new_record["manual_manifest_sha256"],
        ledger_parquet_sha256=new_record["ledger_sha256"],
        financial_state_sha256=new_record["financial_state_sha256"],
        post_flow_unit_count=post_flow_units,
        external_flow_amount=external_flow_amount,
        allow_same_date_correction=explicit_correction,
    )

    lineage = [dict(row) for row in catalog["lineage_index"]]
    execution_class = (
        "APPLIED_TRADES"
        if strict_record.get("execution_kind") == "applied_effective_ledger"
        else "NO_TRADE"
    )
    lineage.append(
        {
            "record_id": record_id,
            "source_record_id": active_record_id,
            "supersedes_record_id": active_record_id if explicit_correction else None,
            "valuation_date": strict_record["data_date"],
            "execution_class": execution_class,
            "publication_class": (
                "CORRECTION" if explicit_correction else "OFFICIAL_FINANCIAL_STATE"
            ),
            "storage_state": "ONLINE",
            "manifest_ref": {
                "path": new_record["manifest_path"],
                "sha256": new_record["manifest_sha256"],
            },
            "manual_manifest_ref": {
                "path": new_record["manual_manifest_path"],
                "sha256": new_record["manual_manifest_sha256"],
            },
            "effective_ledger_ref": {
                "path": new_record["ledger_path"],
                "sha256": new_record["ledger_sha256"],
            },
            "financial_state_sha256": new_record["financial_state_sha256"],
            "ledger_parquet_sha256": new_record["ledger_sha256"],
        }
    )
    validate_lineage_index(lineage, active_record_id=record_id)

    performance_generation = _record_id(str(performance_generation))
    prefix_text = f"_record_store/performance/{performance_generation}"
    prefix = root / prefix_text
    series_path = prefix / "series.parquet"
    series_sha, series_bytes = write_deterministic_parquet(next_rows, series_path)
    parent_manifest = performance["manifest"]
    owner = build_performance_owner_declaration(
        performance_generation_id=performance_generation,
        declared_at=sealed_at,
        series_path=f"{prefix_text}/series.parquet",
        series_sha256=series_sha,
        series_bytes=series_bytes,
        source_pointer_sha256=args.expected_pointer_sha,
        source_catalog_sha256=pointer["catalog_sha256"],
        normalized_projection_semantic_sha256=parent_manifest[
            "normalized_projection_semantic_sha256"
        ],
    )
    owner_raw = canonical_json_bytes(owner)
    owner_sha = immutable_write(
        prefix / "owner_declaration.v1.json",
        owner_raw,
        max_bytes=MAX_PERFORMANCE_JSON_BYTES,
    )
    manifest = build_performance_manifest(
        performance_generation_id=performance_generation,
        generated_at=sealed_at,
        identity_path=parent_manifest["identity_declaration"]["path"],
        identity_sha256=parent_manifest["identity_declaration"]["sha256"],
        parent_performance_manifest_sha256=catalog["performance_history_ref"][
            "manifest"
        ]["sha256"],
        source_pointer_sha256=args.expected_pointer_sha,
        source_catalog_generation_id=pointer["generation_id"],
        source_catalog_sha256=pointer["catalog_sha256"],
        dashboard_projection_sha256=parent_manifest[
            "source_dashboard_projection_sha256"
        ],
        normalized_projection_semantic_sha256=parent_manifest[
            "normalized_projection_semantic_sha256"
        ],
        series_path=f"{prefix_text}/series.parquet",
        series_sha256=series_sha,
        series_bytes=series_bytes,
        owner_path=f"{prefix_text}/owner_declaration.v1.json",
        owner_sha256=owner_sha,
        owner_bytes=len(owner_raw),
        rows=next_rows,
    )
    manifest_raw = canonical_json_bytes(manifest)
    manifest_sha = immutable_write(
        prefix / "manifest.v1.json",
        manifest_raw,
        max_bytes=MAX_PERFORMANCE_JSON_BYTES,
    )
    performance_ref = build_performance_history_ref(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        manifest_bytes=len(manifest_raw),
    )
    load_performance_history(root, performance_ref)
    if _pointer_sha(root) != args.expected_pointer_sha:
        raise StrategyRecordStoreError("source pointer drifted before catalog v3 CAS")
    result = publish_catalog(
        root,
        expected_pointer_sha256=args.expected_pointer_sha,
        records=records,
        active_record_id=record_id,
        previous_record_id=active_record_id,
        generation_id=args.generation_id,
        published_at=sealed_at,
        catalog_schema=CATALOG_SCHEMA_V3,
        inherit_history_registry=False,
        lineage_index=lineage,
        performance_history_ref=performance_ref,
    )
    readback = load_registered_catalog(root)
    if readback is None or readback != (result["pointer"], result["catalog"]):
        raise StrategyRecordStoreError("catalog v3 seal-publish readback mismatch")
    return {
        **result,
        "performance_generation_id": performance_generation,
        "performance_manifest_sha256": manifest_sha,
        "performance_series_sha256": series_sha,
        "performance_owner_declaration_sha256": owner_sha,
        "performance_contract_ready": True,
    }


def command_seal_publish(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve(strict=True)
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError(
            "seal-publish requires a registered catalog"
        )
    pointer, catalog = loaded
    observed_pointer_sha = _pointer_sha(root)
    if observed_pointer_sha != args.expected_pointer_sha:
        from quant_investor.strategy_records.store import (
            StrategyRecordCASMismatch,
        )

        raise StrategyRecordCASMismatch(
            args.expected_pointer_sha, observed_pointer_sha
        )
    record_id = _strict_run_id(args.record_id)
    stage = root / STORE_DIRECTORY / "staging" / record_id
    target = root / record_id
    adopting_existing = not stage.exists() and target.exists()
    sealed_at = _timestamp(args.published_at)
    source_candidate = stage if stage.exists() else target
    if catalog.get("schema_id") == CATALOG_SCHEMA_V3:
        _reject_disabled_ledger_candidates(source_candidate)
    if stage.exists():
        staged_inventory = build_inventory(
            stage, enforce_new_record_budget=True
        )
    elif target.exists():
        staged_inventory = build_inventory(
            target, enforce_new_record_budget=True
        )
    else:
        raise StrategyRecordStoreError("staging record is absent")
    if target.exists():
        target_inventory = build_inventory(
            target, enforce_new_record_budget=True
        )
        if not _same_inventory(staged_inventory, target_inventory):
            raise StrategyRecordConflict(
                "record identity exists with different bytes"
            )
    else:
        if os.lstat(stage).st_dev != os.lstat(root).st_dev:
            raise StrategyRecordStoreError(
                "staging and records are not on one filesystem"
            )
        os.replace(stage, target)
        target_inventory = build_inventory(
            target, enforce_new_record_budget=True
        )
        if not _same_inventory(staged_inventory, target_inventory):
            raise StrategyRecordStoreError("sealed record readback mismatch")
    new_record = {
        "record_id": record_id,
        "relative_path": record_id,
        "state": "ONLINE",
        "storage_state": "ONLINE",
        "sealed_at": sealed_at,
        **target_inventory,
    }
    records = [dict(record) for record in catalog["records"]]
    matches = [
        record for record in records if record["record_id"] == record_id
    ]
    if matches:
        if not _same_inventory(matches[0], new_record):
            raise StrategyRecordConflict("catalog record identity collision")
        if pointer.get("active_record_id") == record_id and (
            catalog.get("schema_id") == CATALOG_SCHEMA_V3
            or not getattr(args, "project_root", None)
        ):
            return {
                "idempotent": True,
                "pointer": pointer,
                "catalog": catalog,
                "pointer_sha256": observed_pointer_sha,
            }
        records = [
            new_record if record["record_id"] == record_id else record
            for record in records
        ]
    else:
        records.append(new_record)
    if catalog.get("schema_id") == CATALOG_SCHEMA_V3:
        return _seal_publish_catalog_v3(
            args=args,
            root=root,
            target=target,
            record_id=record_id,
            sealed_at=sealed_at,
            pointer=pointer,
            catalog=catalog,
            records=records,
            new_record=new_record,
        )
    dashboard_projection = None
    if getattr(args, "project_root", None):
        try:
            from scripts.cn_dashboard_common import (
                DashboardInputError,
                scan_historical_performance_records,
                scan_valid_records,
                validate_historical_performance_sequence,
                validate_historical_record,
                validate_record,
            )
        except ImportError as exc:
            raise StrategyRecordStoreError(
                "Dashboard validation is unavailable"
            ) from exc
        project_root = Path(args.project_root)
        archive_aware = any(
            row.get("state") == "ARCHIVED" for row in records
        )
        if archive_aware:
            existing_projection = catalog.get("dashboard_projection")
            if not isinstance(existing_projection, dict):
                raise StrategyRecordStoreError(
                    "archive-aware publication requires Dashboard projection"
                )
            dashboard_projection = json.loads(
                json.dumps(existing_projection, ensure_ascii=False)
            )
            current_row = validate_record(target, root, project_root)
            valid_by_id = {
                row["record"]: row
                for row in dashboard_projection["valid_records"]
            }
            valid_by_id[record_id] = current_row
            dashboard_projection["valid_records"] = [
                valid_by_id[key] for key in sorted(valid_by_id)
            ]
            dashboard_projection["rejected"] = [
                value
                for value in dashboard_projection["rejected"]
                if not value.startswith(record_id + ":")
            ]
            dashboard_projection["latest_seen"] = max(
                str(dashboard_projection.get("latest_seen") or record_id),
                record_id,
            )
            historical_by_id = {
                row["record"]: row
                for row in dashboard_projection["historical_records"]
            }
            historical_rejected = [
                value
                for value in dashboard_projection["historical_rejected"]
                if not value.startswith(record_id + ":")
            ]
            try:
                historical_by_id[record_id] = validate_historical_record(
                    record_dir=target,
                    record_root=root,
                    project_root=project_root,
                    strict_record=current_row,
                )
            except DashboardInputError as exc:
                historical_by_id.pop(record_id, None)
                historical_rejected.append(f"{record_id}:{exc}")
            historical = [
                historical_by_id[key] for key in sorted(historical_by_id)
            ]
            validate_historical_performance_sequence(historical)
            dashboard_projection["historical_records"] = historical
            dashboard_projection["historical_rejected"] = sorted(
                historical_rejected
            )
        else:
            valid, rejected, latest_seen = scan_valid_records(
                root, project_root
            )
            historical, historical_rejected = (
                scan_historical_performance_records(
                    record_root=root,
                    project_root=project_root,
                    strict_records=valid,
                )
            )
            dashboard_projection = {
                "valid_records": valid,
                "rejected": rejected,
                "latest_seen": latest_seen,
                "historical_records": historical,
                "historical_rejected": historical_rejected,
            }
        _normalize_dashboard_projection_source_refs(dashboard_projection)
        if adopting_existing:
            valid_by_id = {
                row["record"]: row
                for row in dashboard_projection["valid_records"]
            }
            registered_ids = {row["record_id"] for row in records}
            source_id = valid_by_id.get(record_id, {}).get("source_record")
            visited = {record_id}
            adopted: list[dict[str, Any]] = []
            while source_id and source_id not in registered_ids:
                if source_id in visited:
                    raise StrategyRecordStoreError(
                        "existing record source chain contains a cycle"
                    )
                visited.add(source_id)
                source_row = valid_by_id.get(source_id)
                if source_row is None:
                    raise StrategyRecordStoreError(
                        "existing record source chain is not Dashboard-valid"
                    )
                source_path = root / _strict_run_id(source_id)
                source_inventory = build_inventory(
                    source_path, enforce_new_record_budget=True
                )
                adopted.append(
                    {
                        "record_id": source_id,
                        "relative_path": source_id,
                        "state": "ONLINE",
                        "storage_state": "ONLINE",
                        "sealed_at": sealed_at,
                        **source_inventory,
                    }
                )
                registered_ids.add(source_id)
                source_id = source_row.get("source_record")
            if adopted:
                records = [
                    row for row in records if row["record_id"] != record_id
                ]
                records.extend(reversed(adopted))
                records.append(new_record)
        _attach_dashboard_closure(
            records,
            dashboard_projection,
            root=root,
            project_root=project_root,
        )
        if record_id not in {
            row.get("record") for row in dashboard_projection["valid_records"]
        }:
            raise StrategyRecordStoreError(
                "sealed record did not pass Dashboard current-record validation"
            )
    previous_record_id = pointer.get("active_record_id")
    if adopting_existing and dashboard_projection is not None:
        valid_by_id = {
            row["record"]: row
            for row in dashboard_projection["valid_records"]
        }
        direct_source = valid_by_id[record_id].get("source_record")
        if direct_source in {row["record_id"] for row in records}:
            previous_record_id = direct_source
    catalog_schema = (
        CATALOG_SCHEMA_V2
        if any(row.get("state") == "ARCHIVED" for row in records)
        else CATALOG_SCHEMA_V1
    )
    history_registry = None
    history_registry_ref = None
    if catalog_schema == CATALOG_SCHEMA_V2:
        if not args.generation_id:
            raise StrategyRecordStoreError(
                "archive-aware seal-publish requires generation_id"
            )
        registry_output = (
            root
            / STORE_DIRECTORY
            / "history_integrity"
            / f"{args.generation_id}.v2.json"
        )
        history_registry, history_registry_ref = (
            _build_candidate_history_registry(
                project=Path(args.project_root),
                root=root,
                txid=args.generation_id,
                generation_id=args.generation_id,
                projection=dashboard_projection,
                transformed=records,
                output=str(registry_output),
                generated_at=sealed_at,
            )
        )
    return publish_catalog(
        root,
        expected_pointer_sha256=args.expected_pointer_sha,
        records=records,
        dashboard_projection=dashboard_projection,
        active_record_id=record_id,
        previous_record_id=previous_record_id,
        generation_id=args.generation_id,
        published_at=sealed_at,
        catalog_schema=catalog_schema,
        history_registry=history_registry,
        history_registry_ref=history_registry_ref,
        inherit_history_registry=False,
    )


def command_no_action(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root)
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError(
            "no-action requires a registered catalog"
        )
    pointer, _ = loaded
    observed_pointer_sha = _pointer_sha(root)
    if observed_pointer_sha != args.expected_pointer_sha:
        from quant_investor.strategy_records.store import (
            StrategyRecordCASMismatch,
        )

        raise StrategyRecordCASMismatch(
            args.expected_pointer_sha, observed_pointer_sha
        )
    if not pointer.get("active_record_id") or not pointer.get(
        "active_closure"
    ):
        raise StrategyRecordStoreError(
            "no-action requires an active checkpoint"
        )
    receipt = {
        "schema_id": "myquant.strategy_record_no_action_receipt.v1",
        "receipt_id": _record_id(args.receipt_id),
        "created_at": _timestamp(args.published_at),
        "status": "NO_ACTION",
        "reason": args.reason,
        "active_record_id": pointer["active_record_id"],
        "active_checkpoint": dict(pointer["active_closure"]),
        "payload_copied": False,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
    }
    receipt["content_sha256"] = content_sha256(receipt)
    if len(canonical_json_bytes(receipt)) > NO_ACTION_RECEIPT_MAX_BYTES:
        raise StrategyRecordStoreError("no-action receipt exceeds byte budget")
    return publish_catalog(
        root,
        expected_pointer_sha256=args.expected_pointer_sha,
        receipts=[receipt],
        active_record_id=pointer.get("active_record_id"),
        previous_record_id=pointer.get("previous_record_id"),
        generation_id=args.generation_id,
        published_at=receipt["created_at"],
    )


def command_verify(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root)
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError("strategy-record store is unregistered")
    pointer, catalog = loaded
    verified = 0
    verified_archived = 0
    verified_archives: set[str] = set()
    for record in catalog["records"]:
        state = record.get("state", record.get("storage_state"))
        if state == "ARCHIVED":
            binding = load_archive_binding(root, record)
            verified_archived += 1
            verified_archives.add(binding["locator"]["archive_id"])
            continue
        if state != "ONLINE":
            continue
        observed = build_inventory(
            root / record["relative_path"], enforce_new_record_budget=False
        )
        if not _same_inventory(observed, record):
            raise StrategyRecordStoreError(
                f"registered record inventory mismatch: {record['record_id']}"
            )
        verified += 1
    result = {
        "valid": True,
        "pointer_sha256": _pointer_sha(root),
        "generation_id": pointer["generation_id"],
        "catalog_schema": catalog["schema_id"],
        "performance_contract_ready": catalog.get("schema_id") == CATALOG_SCHEMA_V3,
        "verified_online_records": verified,
        "verified_archived_records": verified_archived,
        "verified_archives": len(verified_archives),
        "orphan_record_dirs": _orphans(root, catalog),
        "orphans_preserved": True,
    }
    if catalog.get("schema_id") == CATALOG_SCHEMA_V3:
        performance = load_performance_history(root, catalog["performance_history_ref"])
        result["lineage_index_sha256"] = catalog["lineage_index_sha256"]
        result["performance_generation_id"] = catalog["performance_history_ref"][
            "performance_generation_id"
        ]
        result["performance_manifest_sha256"] = catalog["performance_history_ref"][
            "manifest"
        ]["sha256"]
        result["performance_series_sha256"] = performance["series_sha256"]
        result["performance_owner_declaration_sha256"] = catalog[
            "performance_history_ref"
        ]["owner_declaration"]["sha256"]
    return result


def command_reselect_catalog(args: argparse.Namespace) -> dict[str, Any]:
    if args.owner_approved_by != "maxwell":
        raise StrategyRecordStoreError(
            "catalog reselection requires explicit owner approval by maxwell"
        )
    reason = args.approval_reason
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > 500
        or "\n" in reason
        or "\r" in reason
    ):
        raise StrategyRecordStoreError("catalog reselection approval reason is invalid")
    result = reselect_catalog(
        args.record_root,
        expected_current_pointer_sha256=args.expected_current_pointer_sha,
        target_generation_id=args.target_generation_id,
        target_catalog_path=args.target_catalog_path,
        target_catalog_sha256=args.target_catalog_sha,
        published_at=_timestamp(args.published_at),
    )
    return {
        "catalog_reselected": True,
        "catalog_created": False,
        "pointer_sha256": result["pointer_sha256"],
        "generation_id": result["pointer"]["generation_id"],
        "catalog_path": result["pointer"]["catalog_path"],
        "catalog_sha256": result["pointer"]["catalog_sha256"],
        "catalog_schema": result["catalog"]["schema_id"],
        "performance_contract_ready": (
            result["catalog"].get("performance_contract_ready") is True
        ),
        "owner_approved_by": args.owner_approved_by,
        "approval_reason": reason,
        "v17_mainline_authority": False,
        "broker_order_execution_trade_authority": False,
    }


IDENTITY_RELATIVE_PATH = (
    "results/portfolio_cycle/CN/cn-aggressive-tech-manufacturing/"
    "governance/strategy_identity.v1.json"
)


def _read_candidate_bytes(
    path: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    label: str,
) -> bytes:
    validate_safe_regular_0600(path, label=label)
    digest, size = regular_file_sha256(path, label=label)
    if digest != expected_sha256:
        raise StrategyRecordStoreError(f"{label} SHA-256 mismatch")
    if size <= 0 or size > max_bytes:
        raise StrategyRecordStoreError(f"{label} exceeds byte budget")
    raw = path.read_bytes()
    if len(raw) != size or _sha(raw) != digest:
        raise StrategyRecordStoreError(f"{label} changed during readback")
    return raw


def _read_candidate_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_candidate_bytes(
        path,
        expected_sha256=expected_sha256,
        max_bytes=MAX_PERFORMANCE_JSON_BYTES,
        label=label,
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyRecordStoreError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise StrategyRecordStoreError(f"{label} is not canonical JSON")
    validate_performance_semantic(value, label=label)
    return value, raw


def command_declare_strategy_identity(args: argparse.Namespace) -> dict[str, Any]:
    from quant_investor.portfolio_cycle.contracts import (
        IDENTITY_DECLARATION_SCHEMA_ID,
        PROTOCOL,
        canonical_json_bytes as portfolio_canonical_json_bytes,
        seal_document,
    )
    from quant_investor.portfolio_cycle.identity import resolve_strategy_identity

    project = Path(args.project_root).resolve(strict=True)
    relative = args.identity_path or IDENTITY_RELATIVE_PATH
    if relative != IDENTITY_RELATIVE_PATH:
        raise StrategyRecordStoreError("strategy identity path is not canonical")
    declared_at = _timestamp(args.declared_at)
    provenance = args.provenance or (
        "Owner maxwell confirmed the one-to-one mapping from historical label "
        "aggressive_tech_manufacturing to canonical strategy ID "
        "cn-aggressive-tech-manufacturing. This declaration proves identity only; "
        "it creates no holdings pointer, V17 activation, Factor admission, risk or "
        "publisher permit, or broker/order/execution/trade authority."
    )
    declaration = seal_document(
        {
            "schema_id": IDENTITY_DECLARATION_SCHEMA_ID,
            "protocol": PROTOCOL,
            "historical_label": "aggressive_tech_manufacturing",
            "canonical_strategy_id": "cn-aggressive-tech-manufacturing",
            "declared_by": "maxwell",
            "declared_at": declared_at,
            "authority_kind": "owner_declaration",
            "provenance": provenance,
        }
    )
    raw = portfolio_canonical_json_bytes(declaration)
    target = project / relative
    digest = immutable_write(target, raw, max_bytes=MAX_PERFORMANCE_JSON_BYTES)
    resolved = resolve_strategy_identity(
        project,
        declaration_path=relative,
        declaration_sha256=digest,
        expected_historical_label="aggressive_tech_manufacturing",
    )
    if resolved.canonical_strategy_id != "cn-aggressive-tech-manufacturing":
        raise StrategyRecordStoreError("strategy identity readback mismatch")
    return {
        "created_or_idempotent": True,
        "identity_path": relative,
        "identity_sha256": digest,
        "canonical_strategy_id": resolved.canonical_strategy_id,
        "historical_label": resolved.historical_label,
        "authority_kind": resolved.authority_kind,
        "v17_activation_authority": False,
        "broker_order_execution_trade_authority": False,
    }


def command_prepare_performance_migration(args: argparse.Namespace) -> dict[str, Any]:
    from quant_investor.portfolio_cycle.identity import resolve_strategy_identity

    root = Path(args.record_root).resolve(strict=True)
    project = Path(args.project_root).resolve(strict=True)
    if project_root_for_record_root(root).resolve(strict=True) != project:
        raise StrategyRecordStoreError("project root does not bind the governed record root")
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError("performance migration requires a registered store")
    pointer, catalog = loaded
    observed_pointer_sha = _pointer_sha(root)
    if observed_pointer_sha != args.expected_pointer_sha:
        from quant_investor.strategy_records.store import StrategyRecordCASMismatch

        raise StrategyRecordCASMismatch(args.expected_pointer_sha, observed_pointer_sha)
    if catalog.get("schema_id") != CATALOG_SCHEMA_V2:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:seed requires registered catalog v2"
        )
    identity_path = args.identity_path
    if identity_path != IDENTITY_RELATIVE_PATH:
        raise StrategyRecordStoreError("performance identity path is not canonical")
    identity = resolve_strategy_identity(
        project,
        declaration_path=identity_path,
        declaration_sha256=args.identity_sha,
        expected_historical_label="aggressive_tech_manufacturing",
    )
    if identity.canonical_strategy_id != "cn-aggressive-tech-manufacturing":
        raise StrategyRecordStoreError("performance identity strategy mismatch")
    performance_generation = _record_id(args.performance_generation_id)
    generated_at = _timestamp(args.generated_at)
    owner_declared_at = _timestamp(args.owner_declared_at)
    candidate_dir = assert_private_tmp(Path(args.output_dir))
    if candidate_dir.exists():
        raise StrategyRecordConflict("performance candidate output already exists")
    candidate_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidate_dir.mkdir(mode=0o700)
    normalized, projection_sha, normalized_sha = normalize_registered_projection(catalog)
    rows = build_seed_rows(normalized, catalog=catalog)
    if rows[-1]["record_id"] != pointer.get("active_record_id"):
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:seed does not end at active record"
        )
    lineage = build_lineage_index(catalog)
    validate_lineage_index(lineage, active_record_id=pointer.get("active_record_id"))
    series_store_path = (
        f"_record_store/performance/{performance_generation}/series.parquet"
    )
    owner_store_path = (
        f"_record_store/performance/{performance_generation}/owner_declaration.v1.json"
    )
    series_path = candidate_dir / "series.parquet"
    replay_path = candidate_dir / ".series.replay.parquet"
    series_sha, series_bytes = write_deterministic_parquet(
        rows, series_path, replay_path=replay_path
    )
    readback_rows = read_performance_parquet(series_path)
    if readback_rows != rows:
        raise StrategyRecordStoreError("performance candidate Parquet readback mismatch")
    owner_candidate = build_performance_owner_declaration(
        performance_generation_id=performance_generation,
        declared_at=owner_declared_at,
        series_path=series_store_path,
        series_sha256=series_sha,
        series_bytes=series_bytes,
        source_pointer_sha256=observed_pointer_sha,
        source_catalog_sha256=pointer["catalog_sha256"],
        normalized_projection_semantic_sha256=normalized_sha,
    )
    owner_raw = canonical_json_bytes(owner_candidate)
    owner_sha = _sha(owner_raw)
    manifest = build_performance_manifest(
        performance_generation_id=performance_generation,
        generated_at=generated_at,
        identity_path=identity_path,
        identity_sha256=args.identity_sha,
        parent_performance_manifest_sha256=None,
        source_pointer_sha256=observed_pointer_sha,
        source_catalog_generation_id=pointer["generation_id"],
        source_catalog_sha256=pointer["catalog_sha256"],
        dashboard_projection_sha256=projection_sha,
        normalized_projection_semantic_sha256=normalized_sha,
        series_path=series_store_path,
        series_sha256=series_sha,
        series_bytes=series_bytes,
        owner_path=owner_store_path,
        owner_sha256=owner_sha,
        owner_bytes=len(owner_raw),
        rows=rows,
    )
    manifest_raw = canonical_json_bytes(manifest)
    manifest_path = candidate_dir / "manifest.v1.json"
    manifest_sha = immutable_write(
        manifest_path, manifest_raw, max_bytes=MAX_PERFORMANCE_JSON_BYTES
    )
    lineage_sha = _sha(canonical_json_bytes(lineage))
    lineage_document = seal_semantic(
        {
            "schema_id": "myquant.strategy_record_lineage_index.v1",
            "performance_generation_id": performance_generation,
            "active_record_id": pointer["active_record_id"],
            "previous_record_id": pointer["previous_record_id"],
            "lineage_index": lineage,
            "lineage_index_sha256": lineage_sha,
        }
    )
    lineage_raw = canonical_json_bytes(lineage_document)
    lineage_path = candidate_dir / "lineage_index.v1.json"
    lineage_document_sha = immutable_write(
        lineage_path, lineage_raw, max_bytes=MAX_PERFORMANCE_JSON_BYTES
    )
    receipt = seal_semantic(
        {
            "schema_id": PERFORMANCE_MIGRATION_RECEIPT_SCHEMA,
            "performance_generation_id": performance_generation,
            "generated_at": generated_at,
            "source_pointer_sha256": observed_pointer_sha,
            "source_catalog_generation_id": pointer["generation_id"],
            "source_catalog_sha256": pointer["catalog_sha256"],
            "dashboard_projection_sha256": projection_sha,
            "normalized_projection_semantic_sha256": normalized_sha,
            "identity_path": identity_path,
            "identity_sha256": args.identity_sha,
            "row_count": len(rows),
            "date_range": [rows[0]["valuation_date"], rows[-1]["valuation_date"]],
            "first_record_id": rows[0]["record_id"],
            "last_record_id": rows[-1]["record_id"],
            "first_raw_nav_cny": decimal_text(
                rows[0]["raw_nav_cny"], quantum=MONEY_QUANTUM
            ),
            "last_raw_nav_cny": decimal_text(
                rows[-1]["raw_nav_cny"], quantum=MONEY_QUANTUM
            ),
            "final_net_external_flow_cny": decimal_text(
                rows[-1]["excluded_external_flow_cny"], quantum=MONEY_QUANTUM
            ),
            "cumulative_return": decimal_text(
                rows[-1]["cumulative_return"], quantum=UNIT_QUANTUM
            ),
            "max_drawdown": decimal_text(
                min(row["drawdown"] for row in rows), quantum=UNIT_QUANTUM
            ),
            "candidate_manifest_sha256": manifest_sha,
            "candidate_manifest_bytes": len(manifest_raw),
            "series_sha256": series_sha,
            "series_bytes": series_bytes,
            "prospective_owner_declaration_sha256": owner_sha,
            "prospective_owner_declaration_bytes": len(owner_raw),
            "prospective_owner_declaration": owner_candidate,
            "lineage_index_sha256": lineage_sha,
            "lineage_document_sha256": lineage_document_sha,
            "lineage_count": len(lineage),
            "legacy_runtime_access": False,
            "candidate_only": True,
            "store_pointer_modified": False,
            "owner_approval_required": True,
            "v17_activation_authority": False,
            "broker_order_execution_trade_authority": False,
        }
    )
    receipt_raw = canonical_json_bytes(receipt)
    receipt_path = candidate_dir / "candidate_receipt.v1.json"
    receipt_sha = immutable_write(
        receipt_path, receipt_raw, max_bytes=MAX_PERFORMANCE_JSON_BYTES
    )
    if _pointer_sha(root) != observed_pointer_sha:
        raise StrategyRecordStoreError("source pointer drifted during migration prepare")
    return {
        "prepared": True,
        "published": False,
        "owner_approval_required": True,
        "candidate_dir": str(candidate_dir),
        "candidate_receipt_path": str(receipt_path),
        "candidate_receipt_sha256": receipt_sha,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": manifest_sha,
        "series_path": str(series_path),
        "series_sha256": series_sha,
        "prospective_owner_declaration_sha256": owner_sha,
        "lineage_document_path": str(lineage_path),
        "lineage_document_sha256": lineage_document_sha,
        "row_count": len(rows),
        "date_range": [rows[0]["valuation_date"], rows[-1]["valuation_date"]],
        "cumulative_return": receipt["cumulative_return"],
        "max_drawdown": receipt["max_drawdown"],
        "source_pointer_sha256": observed_pointer_sha,
        "store_pointer_modified": False,
    }


def command_seal_performance_owner_declaration(args: argparse.Namespace) -> dict[str, Any]:
    candidate_dir = assert_private_tmp(Path(args.candidate_dir)).resolve(strict=True)
    receipt, _ = _read_candidate_json(
        candidate_dir / "candidate_receipt.v1.json",
        expected_sha256=args.candidate_receipt_sha,
        label="performance candidate receipt",
    )
    if receipt.get("schema_id") != PERFORMANCE_MIGRATION_RECEIPT_SCHEMA:
        raise StrategyRecordStoreError("performance candidate receipt schema mismatch")
    for field, expected in (
        ("candidate_manifest_sha256", args.approved_manifest_sha),
        ("series_sha256", args.approved_series_sha),
        (
            "prospective_owner_declaration_sha256",
            args.approved_owner_declaration_sha,
        ),
    ):
        if receipt.get(field) != expected:
            raise StrategyRecordStoreError("owner approval does not match exact candidate")
    owner = receipt.get("prospective_owner_declaration")
    if not isinstance(owner, dict):
        raise StrategyRecordStoreError("prospective owner declaration is absent")
    raw = canonical_json_bytes(owner)
    if _sha(raw) != args.approved_owner_declaration_sha:
        raise StrategyRecordStoreError("prospective owner declaration SHA mismatch")
    target = candidate_dir / "owner_declaration.v1.json"
    digest = immutable_write(target, raw, max_bytes=MAX_PERFORMANCE_JSON_BYTES)
    manifest, _ = _read_candidate_json(
        candidate_dir / "manifest.v1.json",
        expected_sha256=args.approved_manifest_sha,
        label="performance candidate manifest",
    )
    validate_performance_owner_declaration(
        owner,
        expected_generation=receipt["performance_generation_id"],
        expected_series=manifest["series"],
    )
    return {
        "owner_declaration_sealed": True,
        "owner_declaration_path": str(target),
        "owner_declaration_sha256": digest,
        "approved_candidate_receipt_sha256": args.candidate_receipt_sha,
        "approved_manifest_sha256": args.approved_manifest_sha,
        "approved_series_sha256": args.approved_series_sha,
    }


def command_publish_performance_migration(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve(strict=True)
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError("performance migration publish requires a store")
    pointer, catalog = loaded
    observed_pointer_sha = _pointer_sha(root)
    if observed_pointer_sha != args.expected_pointer_sha:
        from quant_investor.strategy_records.store import StrategyRecordCASMismatch

        raise StrategyRecordCASMismatch(args.expected_pointer_sha, observed_pointer_sha)
    if catalog.get("schema_id") != CATALOG_SCHEMA_V2:
        raise StrategyRecordStoreError("performance migration source is no longer catalog v2")
    candidate_dir = assert_private_tmp(Path(args.candidate_dir)).resolve(strict=True)
    receipt, _ = _read_candidate_json(
        candidate_dir / "candidate_receipt.v1.json",
        expected_sha256=args.candidate_receipt_sha,
        label="performance candidate receipt",
    )
    manifest, manifest_raw = _read_candidate_json(
        candidate_dir / "manifest.v1.json",
        expected_sha256=args.candidate_manifest_sha,
        label="performance candidate manifest",
    )
    owner, owner_raw = _read_candidate_json(
        candidate_dir / "owner_declaration.v1.json",
        expected_sha256=args.owner_declaration_sha,
        label="performance owner declaration",
    )
    lineage_document, _ = _read_candidate_json(
        candidate_dir / "lineage_index.v1.json",
        expected_sha256=args.lineage_document_sha,
        label="performance lineage document",
    )
    series_raw = _read_candidate_bytes(
        candidate_dir / "series.parquet",
        expected_sha256=args.series_sha,
        max_bytes=MAX_PERFORMANCE_PARQUET_BYTES,
        label="performance series candidate",
    )
    if (
        receipt.get("source_pointer_sha256") != observed_pointer_sha
        or receipt.get("source_catalog_sha256") != pointer.get("catalog_sha256")
        or receipt.get("candidate_manifest_sha256") != args.candidate_manifest_sha
        or receipt.get("series_sha256") != args.series_sha
        or receipt.get("prospective_owner_declaration_sha256")
        != args.owner_declaration_sha
        or receipt.get("lineage_document_sha256") != args.lineage_document_sha
        or receipt.get("prospective_owner_declaration") != owner
    ):
        raise StrategyRecordStoreError("performance migration candidate closure mismatch")
    performance_generation = receipt.get("performance_generation_id")
    if performance_generation != args.performance_generation_id:
        raise StrategyRecordStoreError("performance generation mismatch")
    lineage = lineage_document.get("lineage_index")
    if (
        lineage_document.get("schema_id") != "myquant.strategy_record_lineage_index.v1"
        or lineage_document.get("active_record_id") != pointer.get("active_record_id")
        or lineage_document.get("previous_record_id") != pointer.get("previous_record_id")
        or lineage_document.get("lineage_index_sha256")
        != _sha(canonical_json_bytes(lineage))
    ):
        raise StrategyRecordStoreError("performance lineage closure mismatch")
    validate_lineage_index(lineage, active_record_id=pointer.get("active_record_id"))
    performance_ref = build_performance_history_ref(
        manifest=manifest,
        manifest_sha256=args.candidate_manifest_sha,
        manifest_bytes=len(manifest_raw),
    )
    validate_performance_manifest(manifest, expected_ref=performance_ref)
    validate_performance_owner_declaration(
        owner,
        expected_generation=performance_generation,
        expected_series=performance_ref["series"],
    )
    rows = read_performance_parquet(candidate_dir / "series.parquet")
    if (
        len(rows) != receipt.get("row_count")
        or rows[-1]["record_id"] != pointer.get("active_record_id")
    ):
        raise StrategyRecordStoreError("performance candidate active-row mismatch")
    prefix = root / STORE_DIRECTORY / "performance" / str(performance_generation)
    published = {
        "manifest": (
            prefix / "manifest.v1.json",
            manifest_raw,
            args.candidate_manifest_sha,
            MAX_PERFORMANCE_JSON_BYTES,
        ),
        "series": (
            prefix / "series.parquet",
            series_raw,
            args.series_sha,
            MAX_PERFORMANCE_PARQUET_BYTES,
        ),
        "owner_declaration": (
            prefix / "owner_declaration.v1.json",
            owner_raw,
            args.owner_declaration_sha,
            MAX_PERFORMANCE_JSON_BYTES,
        ),
    }
    for label, (target, raw, expected_sha, budget) in published.items():
        observed = immutable_write(target, raw, max_bytes=budget)
        if observed != expected_sha:
            raise StrategyRecordStoreError(f"published performance {label} SHA mismatch")
    load_performance_history(root, performance_ref)
    if _pointer_sha(root) != observed_pointer_sha:
        raise StrategyRecordStoreError("source pointer drifted before catalog CAS")
    result = publish_catalog(
        root,
        expected_pointer_sha256=args.expected_pointer_sha,
        records=[dict(row) for row in catalog["records"]],
        active_record_id=pointer["active_record_id"],
        previous_record_id=pointer["previous_record_id"],
        generation_id=args.catalog_generation_id,
        published_at=_timestamp(args.published_at),
        catalog_schema=CATALOG_SCHEMA_V3,
        inherit_history_registry=False,
        lineage_index=lineage,
        performance_history_ref=performance_ref,
    )
    readback = load_registered_catalog(root)
    if readback is None or readback != (result["pointer"], result["catalog"]):
        raise StrategyRecordStoreError("performance migration post-CAS readback mismatch")
    return {
        "published": True,
        "pointer_sha256": result["pointer_sha256"],
        "catalog_generation_id": result["pointer"]["generation_id"],
        "catalog_sha256": result["pointer"]["catalog_sha256"],
        "performance_generation_id": performance_generation,
        "performance_manifest_sha256": args.candidate_manifest_sha,
        "performance_series_sha256": args.series_sha,
        "performance_owner_declaration_sha256": args.owner_declaration_sha,
        "lineage_index_sha256": result["catalog"]["lineage_index_sha256"],
        "performance_contract_ready": True,
        "v17_activation_authority": False,
        "broker_order_execution_trade_authority": False,
    }


def _selected_before(
    catalog: dict[str, Any], before: str, from_record_id: str | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in catalog["records"]:
        if record.get("state", record.get("storage_state")) != "ONLINE":
            continue
        record_id = record.get("record_id")
        ordering = (
            record_id
            if isinstance(record_id, str)
            and _STRICT_RUN_ID.fullmatch(record_id)
            else record.get("sealed_at")
        )
        if not isinstance(ordering, str):
            raise StrategyRecordStoreError("record ordering field is invalid")
        if ordering < before and (
            from_record_id is None or ordering >= from_record_id
        ):
            result.append(record)
    return result


def _tar_members_safe(
    archive: tarfile.TarFile,
) -> tuple[list[tarfile.TarInfo], int]:
    members = archive.getmembers()
    if len(members) > ARCHIVE_MAX_MEMBERS:
        raise StrategyRecordStoreError("archive member-count budget exceeded")
    names: set[str] = set()
    folded: set[str] = set()
    expanded = 0
    for member in members:
        name = _safe_relative(member.name.rstrip("/"))
        if name in names or name.casefold() in folded:
            raise StrategyRecordStoreError("archive member path collision")
        names.add(name)
        folded.add(name.casefold())
        if not (member.isfile() or member.isdir()):
            raise StrategyRecordStoreError(
                "malicious archive member type rejected"
            )
        if member.isfile():
            expanded += member.size
            if expanded > ARCHIVE_MAX_EXPANDED_BYTES:
                raise StrategyRecordStoreError(
                    "archive expanded-byte budget exceeded"
                )
    return members, expanded


def _restore_tar(tar_path: Path, restore_root: Path) -> None:
    with tarfile.open(tar_path, mode="r:") as archive:
        members, _ = _tar_members_safe(archive)
        for member in members:
            destination = restore_root / member.name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=False)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise StrategyRecordStoreError(
                    "archive file member is unreadable"
                )
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            try:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise StrategyRecordStoreError(
                            "archive member is truncated"
                        )
                    os.write(descriptor, chunk)
                    remaining -= len(chunk)
            finally:
                os.close(descriptor)


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    _safe_relative(info.name.rstrip("/"))
    if not (info.isfile() or info.isdir()):
        raise StrategyRecordStoreError(
            "source contains an unsafe archive member type"
        )
    return info


def command_archive_rehearsal(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root)
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError(
            "archive rehearsal requires a registered catalog"
        )
    _, catalog = loaded
    selected = _selected_before(
        catalog, args.before, getattr(args, "from_record_id", None)
    )
    if not selected:
        raise StrategyRecordStoreError("archive rehearsal selection is empty")
    for record in selected:
        observed = build_inventory(
            root / record["relative_path"],
            enforce_new_record_budget=False,
        )
        if not _same_inventory(observed, record):
            raise StrategyRecordStoreError(
                f"source inventory mismatch: {record['record_id']}"
            )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _regular_directory(output_root, label="archive output root")
    logical_bytes = sum(int(record["total_bytes"]) for record in selected)
    required_free = 2 * logical_bytes + ARCHIVE_FREE_SPACE_RESERVE_BYTES
    if shutil.disk_usage(output_root).free < required_free:
        raise StrategyRecordStoreError(
            "insufficient free space for archive rehearsal"
        )
    zstd = shutil.which("zstd")
    if zstd is None:
        raise StrategyRecordStoreError("zstd executable is unavailable")
    safe_before = (
        re.sub(r"[^A-Za-z0-9._-]+", "-", args.before).strip("-") or "cutoff"
    )
    archive_path = (
        output_root / f"strategy-records-before-{safe_before}.tar.zst"
    )
    if archive_path.exists():
        raise StrategyRecordConflict("archive output already exists")
    with tempfile.TemporaryDirectory(
        prefix="strategy-record-archive-", dir=output_root
    ) as temp:
        temp_root = Path(temp)
        tar_path = temp_root / "records.tar"
        staged_archive = temp_root / "records.tar.zst"
        with tarfile.open(
            tar_path, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            for record in selected:
                archive.add(
                    root / record["relative_path"],
                    arcname=record["relative_path"],
                    recursive=True,
                    filter=_archive_filter,
                )
        subprocess.run(
            [
                zstd,
                "-q",
                "-T1",
                "-19",
                str(tar_path),
                "-o",
                str(staged_archive),
            ],
            check=True,
        )
        restored_tar = temp_root / "restored.tar"
        subprocess.run(
            [zstd, "-q", "-d", str(staged_archive), "-o", str(restored_tar)],
            check=True,
        )
        restore_root = temp_root / "restore"
        restore_root.mkdir()
        _restore_tar(restored_tar, restore_root)
        for record in selected:
            restored = build_inventory(
                restore_root / record["relative_path"],
                enforce_new_record_budget=False,
            )
            if not _same_inventory(restored, record):
                record_id = record["record_id"]
                raise StrategyRecordStoreError(
                    f"archive restore verification failed: {record_id}"
                )
        try:
            os.link(staged_archive, archive_path, follow_symlinks=False)
        except FileExistsError:
            raise StrategyRecordConflict(
                "archive output already exists"
            ) from None
    archive_raw = archive_path.read_bytes()
    return {
        "valid": True,
        "archive_path": str(archive_path),
        "archive_sha256": _sha(archive_raw),
        "archive_bytes": len(archive_raw),
        "logical_bytes": logical_bytes,
        "record_ids": [record["record_id"] for record in selected],
        "source_records_preserved": True,
        "moved": False,
        "deleted": False,
    }


def _project_relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise StrategyRecordStoreError("archive artifact is outside project root") from exc


def _copy_regular_exact_once(source: Path, target: Path) -> tuple[str, int]:
    source_sha, source_bytes = regular_file_sha256(source, label="rehearsal archive")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target_sha, target_bytes = regular_file_sha256(target, label="final archive")
        if (target_sha, target_bytes) != (source_sha, source_bytes):
            raise StrategyRecordConflict("final archive identity collision")
        return target_sha, target_bytes
    source_fd = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    target_fd = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StrategyRecordStoreError("rehearsal archive is unsafe")
        remaining = before.st_size
        while remaining:
            chunk = os.read(source_fd, min(1024 * 1024, remaining))
            if not chunk:
                raise StrategyRecordStoreError("rehearsal archive changed during copy")
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise StrategyRecordStoreError("short final archive write")
                view = view[written:]
            remaining -= len(chunk)
        os.fsync(target_fd)
        after = os.fstat(source_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise StrategyRecordStoreError("rehearsal archive changed during copy")
    finally:
        os.close(target_fd)
        os.close(source_fd)
    _fsync_directory(target.parent)
    target_sha, target_bytes = regular_file_sha256(
        target, expected_bytes=source_bytes, label="final archive"
    )
    if target_sha != source_sha:
        raise StrategyRecordStoreError("final archive readback mismatch")
    return target_sha, target_bytes


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyRecordStoreError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise StrategyRecordStoreError(f"{label} must be an object")
    return value


def command_archive_finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve(strict=True)
    project = Path(args.project_root).resolve(strict=True)
    if project_root_for_record_root(root) != project:
        raise StrategyRecordStoreError("project root does not match record root")
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError("archive finalize requires a registered catalog")
    pointer, catalog = loaded
    if _pointer_sha(root) != args.expected_pointer_sha:
        raise StrategyRecordStoreError("archive finalize pointer checkpoint mismatch")
    archive_id = args.archive_id
    if _ARCHIVE_ID.fullmatch(archive_id) is None:
        raise StrategyRecordStoreError("archive_id is invalid")
    if re.fullmatch(r"20[0-9]{2}-(0[1-9]|1[0-2])", args.archive_month) is None:
        raise StrategyRecordStoreError("archive_month is invalid")
    rehearsal_receipt_path = Path(args.rehearsal_receipt_json).resolve(strict=True)
    rehearsal_base = (
        project
        / "results/strategy_record_archives/CN/aggressive_tech_manufacturing"
    ).resolve(strict=True)
    if (
        rehearsal_base not in rehearsal_receipt_path.parents
        or not rehearsal_receipt_path.parent.name.startswith("rehearsal-")
    ):
        raise StrategyRecordStoreError("rehearsal receipt is outside the allowlisted root")
    rehearsal = _load_json_object(rehearsal_receipt_path, label="rehearsal receipt")
    if (
        rehearsal.get("ok") is not True
        and rehearsal.get("valid") is not True
    ) or (
        rehearsal.get("source_records_preserved") is not True
        or rehearsal.get("moved") is not False
        or rehearsal.get("deleted") is not False
    ):
        raise StrategyRecordStoreError("rehearsal receipt is not successful and copy-only")
    record_ids = rehearsal.get("record_ids")
    if (
        not isinstance(record_ids, list)
        or not record_ids
        or len(record_ids) != len(set(record_ids))
    ):
        raise StrategyRecordStoreError("rehearsal receipt record_ids are invalid")
    by_id = {record["record_id"]: record for record in catalog["records"]}
    selected: list[dict[str, Any]] = []
    expected_prefix = args.archive_month.replace("-", "")
    for record_id in record_ids:
        record = by_id.get(record_id)
        if (
            record is None
            or record.get("state", record.get("storage_state")) != "ONLINE"
            or not str(record_id).startswith(expected_prefix)
        ):
            raise StrategyRecordStoreError("rehearsal selection is not an ONLINE calendar month")
        observed = build_inventory(
            root / record["relative_path"], enforce_new_record_budget=False
        )
        if not _same_inventory(observed, record):
            raise StrategyRecordStoreError(f"source inventory mismatch: {record_id}")
        selected.append(dict(record))
    source_archive = (project / str(rehearsal.get("archive_path"))).resolve(strict=True)
    if source_archive.parent != rehearsal_receipt_path.parent:
        raise StrategyRecordStoreError("rehearsal archive is outside the receipt namespace")
    rehearsal_sha, rehearsal_bytes = regular_file_sha256(
        source_archive, label="rehearsal archive"
    )
    if (
        rehearsal.get("archive_sha256") != rehearsal_sha
        or rehearsal.get("archive_bytes") != rehearsal_bytes
    ):
        raise StrategyRecordStoreError("rehearsal receipt archive binding mismatch")
    month_root = archive_final_root(root, project_root=project) / args.archive_month
    final_archive = month_root / "archive.tar.zst"
    archive_sha, archive_bytes = _copy_regular_exact_once(source_archive, final_archive)
    archive_relative = _project_relative(project, final_archive)
    manifest_path = month_root / "archive-manifest.v1.json"
    manifest_relative = manifest_path.relative_to(project).as_posix()
    projection = catalog.get("dashboard_projection")
    if not isinstance(projection, dict):
        raise StrategyRecordStoreError("archive finalize requires a Dashboard projection")
    logical_refs_by_id: dict[str, dict[str, str]] = {}
    for key in ("valid_records", "historical_records"):
        rows = projection.get(key)
        if not isinstance(rows, list):
            raise StrategyRecordStoreError("Dashboard projection is incomplete")
        for projected in rows:
            if not isinstance(projected, dict) or not isinstance(
                projected.get("source_refs"), list
            ):
                raise StrategyRecordStoreError("Dashboard projection row is invalid")
            record_id = projected.get("record")
            refs_by_path = logical_refs_by_id.setdefault(record_id, {})
            for ref in projected["source_refs"]:
                path = ref.get("path") if isinstance(ref, dict) else None
                digest = ref.get("sha256") if isinstance(ref, dict) else None
                if not isinstance(path, str) or not isinstance(digest, str):
                    raise StrategyRecordStoreError(
                        "Dashboard projection source ref is invalid"
                    )
                previous_digest = refs_by_path.get(path)
                if previous_digest is not None and previous_digest != digest:
                    raise StrategyRecordStoreError(
                        "Dashboard projection refs conflict"
                    )
                refs_by_path[path] = digest
    manifest_records = []
    record_root_relative = root.relative_to(project).as_posix()
    for record in selected:
        projected_refs = logical_refs_by_id.get(record["record_id"])
        logical_source_refs = (
            [
                {"path": path, "sha256": digest}
                for path, digest in sorted(projected_refs.items())
            ]
            if projected_refs
            else None
        )
        if logical_source_refs is None:
            logical_source_refs = [
                {
                    "path": (
                        f"{record_root_relative}/{record['relative_path']}/{item['path']}"
                    ),
                    "sha256": item["sha256"],
                }
                for item in record["inventory"]
                if item.get("type") == "file"
            ]
        manifest_records.append(
            {
                "record_id": record["record_id"],
                "relative_path": record["relative_path"],
                "member_prefix": record["relative_path"],
                "inventory": record["inventory"],
                "inventory_sha256": record["inventory_sha256"],
                "file_count": record["file_count"],
                "total_bytes": record["total_bytes"],
                "logical_source_refs": logical_source_refs,
            }
        )
    manifest = _sealed(
        {
            "schema_id": ARCHIVE_MANIFEST_SCHEMA,
            "archive_id": archive_id,
            "created_at": _timestamp(args.published_at),
            "archive_format": "tar+zstd",
            "archive_path": archive_relative,
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "source_pointer_sha256": args.expected_pointer_sha,
            "source_catalog_generation_id": pointer["generation_id"],
            "source_catalog_path": pointer["catalog_path"],
            "source_catalog_sha256": pointer["catalog_sha256"],
            "records": manifest_records,
            "record_count": len(manifest_records),
            "file_count": sum(row["file_count"] for row in manifest_records),
            "logical_bytes": sum(row["total_bytes"] for row in manifest_records),
            "inventory_set_sha256": _sha(canonical_json_bytes(manifest_records)),
        }
    )
    manifest_sha = _write_exact_once(
        manifest_path,
        {
            key: value
            for key, value in manifest.items()
            if key != "content_sha256"
        },
    )
    zstd = shutil.which("zstd")
    if zstd is None:
        raise StrategyRecordStoreError("zstd executable is unavailable")
    with tempfile.TemporaryDirectory(
        prefix="strategy-record-final-restore-", dir="/private/tmp"
    ) as temp:
        temp_root = Path(temp)
        restored_tar = temp_root / "records.tar"
        subprocess.run(
            [zstd, "-q", "-d", str(final_archive), "-o", str(restored_tar)],
            check=True,
        )
        restore_root = temp_root / "restore"
        restore_root.mkdir()
        _restore_tar(restored_tar, restore_root)
        for record in selected:
            restored = build_inventory(
                restore_root / record["relative_path"],
                enforce_new_record_budget=False,
            )
            if not _same_inventory(restored, record):
                raise StrategyRecordStoreError(f"fresh restore mismatch: {record['record_id']}")
    receipt_path = month_root / "restore-receipt.v1.json"
    receipt_relative = receipt_path.relative_to(project).as_posix()
    receipt = {
        "schema_id": ARCHIVE_RESTORE_RECEIPT_SCHEMA,
        "archive_id": archive_id,
        "created_at": _timestamp(args.published_at),
        "archive_path": archive_relative,
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_sha,
        "record_ids": record_ids,
        "record_count": len(record_ids),
        "restored_file_count": manifest["file_count"],
        "restored_logical_bytes": manifest["logical_bytes"],
        "inventory_set_sha256": manifest["inventory_set_sha256"],
        "all_inventory_matched": True,
        "source_records_preserved": True,
    }
    receipt_sha = _write_exact_once(receipt_path, receipt)
    return {
        "valid": True,
        "archive_id": archive_id,
        "archive_path": archive_relative,
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_sha,
        "restore_receipt_path": receipt_relative,
        "restore_receipt_sha256": receipt_sha,
        "record_ids": record_ids,
        "record_count": len(record_ids),
        "source_records_preserved": True,
    }


def _transaction_root(root: Path, txid: str) -> Path:
    if _ARCHIVE_ID.fullmatch(txid) is None:
        raise StrategyRecordStoreError("transaction id is invalid")
    return root / STORE_DIRECTORY / "quarantine_transactions" / txid


def _quarantine_root(project: Path, txid: str) -> Path:
    return project / (
        "results/strategy_record_quarantine/CN/aggressive_tech_manufacturing"
    ) / txid / "records"


def _read_canonical_transaction(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = _load_json_object(path, label=label)
    if canonical_json_bytes(value) != raw or value.get("content_sha256") != content_sha256(value):
        raise StrategyRecordStoreError(f"{label} is not sealed canonical JSON")
    return value


def _archive_locator_from_manifest(
    root: Path, project: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_relative = _project_relative(project, manifest_path)
    manifest_sha, _ = regular_file_sha256(manifest_path, label="archive manifest")
    manifest = _read_canonical_transaction(manifest_path, label="archive manifest")
    if manifest.get("schema_id") != ARCHIVE_MANIFEST_SCHEMA:
        raise StrategyRecordStoreError("archive manifest schema is unsupported")
    receipt_path = manifest_path.parent / "restore-receipt.v1.json"
    receipt_relative = _project_relative(project, receipt_path)
    receipt_sha, _ = regular_file_sha256(receipt_path, label="archive restore receipt")
    receipt = _read_canonical_transaction(receipt_path, label="archive restore receipt")
    if receipt.get("schema_id") != ARCHIVE_RESTORE_RECEIPT_SCHEMA:
        raise StrategyRecordStoreError("archive restore receipt schema is unsupported")
    if (
        receipt.get("manifest_path") != manifest_relative
        or receipt.get("manifest_sha256") != manifest_sha
    ):
        raise StrategyRecordStoreError("archive restore receipt does not bind manifest")
    base = {
        "schema_id": ARCHIVE_LOCATOR_SCHEMA,
        "archive_id": manifest["archive_id"],
        "archive_path": manifest["archive_path"],
        "archive_sha256": manifest["archive_sha256"],
        "archive_bytes": manifest["archive_bytes"],
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_sha,
        "restore_receipt_path": receipt_relative,
        "restore_receipt_sha256": receipt_sha,
    }
    # Fail closed on the final allow root before the candidate is persisted.
    allowed = archive_final_root(root, project_root=project).resolve(strict=True)
    if allowed not in manifest_path.resolve(strict=True).parents:
        raise StrategyRecordStoreError("archive manifest is outside final archive root")
    return base, manifest


def _write_bytes_exact_once(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        existing = path.read_bytes()
        if existing != raw:
            raise StrategyRecordConflict("immutable artifact identity collision") from None
        return _sha(existing)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StrategyRecordStoreError("short immutable artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if path.read_bytes() != raw:
        raise StrategyRecordStoreError("immutable artifact readback mismatch")
    return _sha(raw)


def _build_candidate_history_registry(
    *,
    project: Path,
    root: Path,
    txid: str,
    generation_id: str,
    projection: dict[str, Any],
    transformed: list[dict[str, Any]],
    output: str | None,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        from scripts.cn_dashboard_common import (
            build_history_integrity_registry,
            canonical_json_bytes as dashboard_canonical_json_bytes,
            sha256_bytes as dashboard_sha256_bytes,
        )
    except ImportError as exc:
        raise StrategyRecordStoreError("Dashboard registry builder is unavailable") from exc
    normalized_projection = {
        "valid_records": [
            {
                **row,
                "source_refs": sorted(
                    row["source_refs"], key=lambda item: item["path"]
                ),
            }
            for row in projection["valid_records"]
        ],
        "rejected": projection["rejected"],
        "latest_seen": projection["latest_seen"],
        "historical_records": [
            {
                **row,
                "source_refs": sorted(
                    row["source_refs"], key=lambda item: item["path"]
                ),
            }
            for row in projection["historical_records"]
        ],
        "historical_rejected": projection["historical_rejected"],
    }
    projection_sha = dashboard_sha256_bytes(
        dashboard_canonical_json_bytes(normalized_projection)
    )
    by_id = {row["record_id"]: row for row in transformed}
    archive_bindings: dict[str, dict[str, Any]] = {}
    registry_records: list[dict[str, Any]] = []
    for original in projection["historical_records"]:
        row = dict(original)
        record_id = row["record"]
        catalog_row = by_id[record_id]
        logical_refs = sorted(row["source_refs"], key=lambda item: item["path"])
        row["logical_source_refs"] = logical_refs
        row["storage_state"] = catalog_row["state"]
        row["record_inventory_sha256"] = catalog_row["inventory_sha256"]
        registry_records.append(row)
        if catalog_row["state"] != "ARCHIVED":
            continue
        locator = catalog_row["archive_locator"]
        archive_bindings[record_id] = {
            "archive_storage_refs": [
                {
                    "path": locator["manifest_path"],
                    "sha256": locator["manifest_sha256"],
                    "bytes": (project / locator["manifest_path"]).stat().st_size,
                    "media_type": "application/json",
                },
                {
                    "path": locator["restore_receipt_path"],
                    "sha256": locator["restore_receipt_sha256"],
                    "bytes": (project / locator["restore_receipt_path"]).stat().st_size,
                    "media_type": "application/json",
                },
                {
                    "path": locator["archive_path"],
                    "sha256": locator["archive_sha256"],
                    "bytes": locator["archive_bytes"],
                    "media_type": "application/zstd",
                },
            ]
        }
    registry = build_history_integrity_registry(
        registry_records,
        generated_at=generated_at,
        intended_generation_id=generation_id,
        dashboard_projection_sha256=projection_sha,
        archive_bindings=archive_bindings,
    )
    registry_path = (
        Path(output).resolve()
        if output
        else _transaction_root(root, txid) / "candidate-history-integrity.v2.json"
    )
    try:
        relative = registry_path.relative_to(project).as_posix()
    except ValueError as exc:
        raise StrategyRecordStoreError("candidate history registry escapes project root") from exc
    raw = (
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = _write_bytes_exact_once(registry_path, raw)
    return registry, {"path": relative, "sha256": digest}


def command_archive_candidate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve(strict=True)
    project = Path(args.project_root).resolve(strict=True)
    if project_root_for_record_root(root) != project:
        raise StrategyRecordStoreError("project root does not match record root")
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError("archive candidate requires a registered catalog")
    pointer, catalog = loaded
    registered_ids = {record["record_id"] for record in catalog["records"]}
    unregistered_strict = sorted(
        child.name
        for child in root.iterdir()
        if _STRICT_RUN_ID.fullmatch(child.name) is not None
        and child.name not in registered_ids
    )
    if unregistered_strict:
        raise StrategyRecordStoreError(
            "unregistered strict record directory blocks archive candidate: "
            + ", ".join(unregistered_strict)
        )
    observed_pointer_sha = _pointer_sha(root)
    if observed_pointer_sha != args.expected_pointer_sha:
        raise StrategyRecordStoreError("archive candidate pointer checkpoint mismatch")
    if len(args.archive_manifest) != 4:
        raise StrategyRecordStoreError("archive candidate requires exactly four monthly manifests")
    checkpoint_path = Path(args.source_checkpoint_receipt).resolve(strict=True)
    checkpoint_sha, checkpoint_bytes = regular_file_sha256(
        checkpoint_path, label="source checkpoint reconstruction receipt"
    )
    if checkpoint_sha != args.source_checkpoint_receipt_sha:
        raise StrategyRecordStoreError("source checkpoint reconstruction receipt SHA mismatch")
    automation_hashes: dict[str, str] = {}
    for binding in args.automation_config_hash:
        if "=" not in binding:
            raise StrategyRecordStoreError("automation config hash binding is invalid")
        name, digest = binding.split("=", 1)
        if name in automation_hashes or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise StrategyRecordStoreError("automation config hash binding is invalid")
        automation_hashes[name] = digest
    if set(automation_hashes) != {"automation", "cn-dashboard", "myquant-cn", "a-2"}:
        raise StrategyRecordStoreError("all four automation config hashes are required")
    archive_rows: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    archive_bindings: list[dict[str, Any]] = []
    for value in args.archive_manifest:
        base, manifest = _archive_locator_from_manifest(
            root, project, Path(value).resolve(strict=True)
        )
        archive_bindings.append(
            {
                "archive_id": base["archive_id"],
                "manifest_path": base["manifest_path"],
                "manifest_sha256": base["manifest_sha256"],
                "restore_receipt_path": base["restore_receipt_path"],
                "restore_receipt_sha256": base["restore_receipt_sha256"],
                "archive_path": base["archive_path"],
                "archive_sha256": base["archive_sha256"],
            }
        )
        for manifest_record in manifest["records"]:
            record_id = manifest_record["record_id"]
            if record_id in archive_rows:
                raise StrategyRecordStoreError("archive manifests overlap")
            locator = {**base, "member_prefix": manifest_record["member_prefix"]}
            archive_rows[record_id] = (locator, manifest_record)
    expected_ids = {
        record["record_id"]
        for record in catalog["records"]
        if record.get("state", record.get("storage_state")) == "ONLINE"
        and record.get("record_id", "") < args.before
        and _STRICT_RUN_ID.fullmatch(record.get("record_id", "")) is not None
    }
    if set(archive_rows) != expected_ids:
        raise StrategyRecordStoreError("archive manifests do not exactly cover cutoff records")
    if (
        pointer.get("active_record_id") in expected_ids
        or pointer.get("previous_record_id") in expected_ids
    ):
        raise StrategyRecordStoreError("archive candidate would retire active authority")
    transformed: list[dict[str, Any]] = []
    for record in catalog["records"]:
        row = dict(record)
        if row["record_id"] in archive_rows:
            locator, manifest_record = archive_rows[row["record_id"]]
            for key in (
                "relative_path",
                "inventory",
                "inventory_sha256",
                "file_count",
                "total_bytes",
            ):
                if row.get(key) != manifest_record.get(key):
                    raise StrategyRecordStoreError(f"archive closure mismatch: {row['record_id']}")
            row["state"] = "ARCHIVED"
            row["storage_state"] = "ARCHIVED"
            row["archive_locator"] = locator
            row["logical_source_refs"] = manifest_record["logical_source_refs"]
            row["evidence_status"] = "ARCHIVE_HASH_VERIFIED"
        transformed.append(row)
    for row in transformed:
        if row.get("state") == "ARCHIVED":
            load_archive_binding(root, row, project_root=project)
    projection = _load_projection(args.dashboard_projection_json)
    if projection is None:
        projection = catalog.get("dashboard_projection")
    if not isinstance(projection, dict):
        raise StrategyRecordStoreError("archive candidate Dashboard projection is missing")
    history_registry, history_registry_ref = _build_candidate_history_registry(
        project=project,
        root=root,
        txid=args.transaction_id,
        generation_id=args.generation_id,
        projection=projection,
        transformed=transformed,
        output=args.history_registry_output,
        generated_at=_timestamp(args.published_at),
    )
    plan = {
        "schema_id": TRANSACTION_PLAN_SCHEMA,
        "transaction_id": args.transaction_id,
        "created_at": _timestamp(args.published_at),
        "project_root": str(project),
        "record_root": str(root),
        "quarantine_root": str(_quarantine_root(project, args.transaction_id)),
        "before": args.before,
        "source_pointer_sha256": observed_pointer_sha,
        "source_pointer": pointer,
        "source_catalog": catalog,
        "source_catalog_sha256": pointer["catalog_sha256"],
        "source_checkpoint_reconstruction_receipt": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "bytes": checkpoint_bytes,
        },
        "automation_config_hashes": automation_hashes,
        "candidate_generation_id": args.generation_id,
        "candidate_records": transformed,
        "candidate_dashboard_projection": projection,
        "candidate_history_registry": history_registry,
        "candidate_history_registry_ref": history_registry_ref,
        "archive_bindings": archive_bindings,
        "record_ids": sorted(expected_ids),
        "record_count": len(expected_ids),
        "protected_hashes": _load_projection(args.protected_hashes_json) or {},
    }
    tx_root = _transaction_root(root, args.transaction_id)
    plan_path = tx_root / "plan.v1.json"
    plan_sha = _write_exact_once(plan_path, plan)
    return {
        "valid": True,
        "transaction_id": args.transaction_id,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "record_count": len(expected_ids),
        "record_ids": sorted(expected_ids),
        "moved": False,
    }


def _load_plan(root: Path, txid: str) -> tuple[Path, dict[str, Any]]:
    path = _transaction_root(root, txid) / "plan.v1.json"
    plan = _read_canonical_transaction(path, label="quarantine transaction plan")
    if plan.get("schema_id") != TRANSACTION_PLAN_SCHEMA or plan.get("transaction_id") != txid:
        raise StrategyRecordStoreError("quarantine transaction plan identity mismatch")
    if plan.get("record_root") != str(root.resolve(strict=True)):
        raise StrategyRecordStoreError("quarantine transaction record root mismatch")
    return path, plan


def _verify_location(record_root: Path, quarantine_root: Path, record: dict[str, Any]) -> str:
    source = record_root / record["relative_path"]
    target = quarantine_root / record["relative_path"]
    source_exists = source.exists()
    target_exists = target.exists()
    if source_exists and target_exists:
        raise StrategyRecordStoreError(f"dual record location: {record['record_id']}")
    if not source_exists and not target_exists:
        raise StrategyRecordStoreError(f"lost record location: {record['record_id']}")
    present = source if source_exists else target
    observed = build_inventory(present, enforce_new_record_budget=False)
    if not _same_inventory(observed, record):
        raise StrategyRecordStoreError(f"quarantine inventory mismatch: {record['record_id']}")
    return "SOURCE" if source_exists else "QUARANTINE"


def _journal_event(tx_root: Path, phase: str, record: dict[str, Any]) -> None:
    event_path = tx_root / "events" / f"{record['record_id']}.{phase}.v1.json"
    if event_path.exists():
        existing = _read_canonical_transaction(event_path, label="quarantine event")
        if existing.get("record_id") != record["record_id"] or existing.get("phase") != phase:
            raise StrategyRecordStoreError("quarantine event identity collision")
        return
    _write_exact_once(
        event_path,
        {
            "schema_id": TRANSACTION_EVENT_SCHEMA,
            "transaction_id": tx_root.name,
            "record_id": record["record_id"],
            "relative_path": record["relative_path"],
            "inventory_sha256": record["inventory_sha256"],
            "phase": phase,
            "created_at": _timestamp(None),
        },
    )


def _move_records(plan: dict[str, Any], *, direction: str) -> int:
    root = Path(plan["record_root"])
    quarantine = Path(plan["quarantine_root"])
    tx_root = _transaction_root(root, plan["transaction_id"])
    quarantine.mkdir(parents=True, exist_ok=True)
    if os.lstat(root).st_dev != os.lstat(quarantine).st_dev:
        raise StrategyRecordStoreError("quarantine is not on the record filesystem")
    source_rows = {row["record_id"]: row for row in plan["source_catalog"]["records"]}
    moved = 0
    for record_id in plan["record_ids"]:
        record = source_rows[record_id]
        location = _verify_location(root, quarantine, record)
        expected = "SOURCE" if direction == "cutover" else "QUARANTINE"
        final = "QUARANTINE" if direction == "cutover" else "SOURCE"
        if location == final:
            _journal_event(tx_root, f"{direction}-complete", record)
            continue
        if location != expected:
            raise StrategyRecordStoreError("unexpected quarantine location state")
        _journal_event(tx_root, f"{direction}-intent", record)
        source = root / record["relative_path"]
        target = quarantine / record["relative_path"]
        if direction == "rollback":
            source, target = target, source
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.lstat(source).st_dev != os.lstat(target.parent).st_dev:
            raise StrategyRecordStoreError("record rename would cross filesystems")
        os.rename(source, target)
        _fsync_directory(source.parent)
        if target.parent != source.parent:
            _fsync_directory(target.parent)
        if _verify_location(root, quarantine, record) != final:
            raise StrategyRecordStoreError("record rename readback mismatch")
        _journal_event(tx_root, f"{direction}-complete", record)
        moved += 1
    return moved


def command_archive_quarantine_cutover(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve(strict=True)
    _, plan = _load_plan(root, args.transaction_id)
    if _pointer_sha(root) != plan["source_pointer_sha256"]:
        raise StrategyRecordStoreError("cutover source pointer checkpoint mismatch")
    quarantine = Path(plan["quarantine_root"])
    if quarantine.exists():
        raise StrategyRecordStoreError("cutover quarantine target already exists")
    quarantine_parent = quarantine.parent
    quarantine_parent.mkdir(parents=True, exist_ok=True)
    if any(quarantine_parent.iterdir()):
        raise StrategyRecordStoreError("cutover transaction quarantine namespace is not empty")
    if os.lstat(root).st_dev != os.lstat(quarantine_parent).st_dev:
        raise StrategyRecordStoreError("quarantine is not on the record filesystem")
    publication = publish_catalog(
        root,
        expected_pointer_sha256=plan["source_pointer_sha256"],
        records=plan["candidate_records"],
        dashboard_projection=plan["candidate_dashboard_projection"],
        history_registry=plan.get("candidate_history_registry"),
        history_registry_ref=plan.get("candidate_history_registry_ref"),
        active_record_id=plan["source_pointer"]["active_record_id"],
        previous_record_id=plan["source_pointer"]["previous_record_id"],
        generation_id=plan["candidate_generation_id"],
        published_at=_timestamp(args.published_at),
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    tx_root = _transaction_root(root, args.transaction_id)
    _write_exact_once(
        tx_root / "catalog-cutover.v1.json",
        {
            "schema_id": TRANSACTION_EVENT_SCHEMA,
            "transaction_id": args.transaction_id,
            "phase": "catalog-cutover",
            "source_pointer_sha256": plan["source_pointer_sha256"],
            "candidate_pointer_sha256": publication["pointer_sha256"],
            "candidate_generation_id": plan["candidate_generation_id"],
            "created_at": _timestamp(args.published_at),
        },
    )
    if args.publish_only:
        return {
            **publication,
            "transaction_id": args.transaction_id,
            "published_only": True,
            "moved_records": 0,
        }
    moved = _move_records(plan, direction="cutover")
    _write_exact_once(
        tx_root / "complete.v1.json",
        {
            "schema_id": TRANSACTION_EVENT_SCHEMA,
            "transaction_id": args.transaction_id,
            "phase": "complete",
            "candidate_pointer_sha256": publication["pointer_sha256"],
            "record_count": plan["record_count"],
            "created_at": _timestamp(args.published_at),
        },
    )
    return {**publication, "transaction_id": args.transaction_id, "moved_records": moved}


def command_archive_quarantine_resume(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve(strict=True)
    _, plan = _load_plan(root, args.transaction_id)
    loaded = load_registered_catalog(root)
    if loaded is None or loaded[0]["generation_id"] != plan["candidate_generation_id"]:
        raise StrategyRecordStoreError("resume requires the archive-aware candidate pointer")
    moved = _move_records(plan, direction="cutover")
    _write_exact_once(
        _transaction_root(root, args.transaction_id) / "complete.v1.json",
        {
            "schema_id": TRANSACTION_EVENT_SCHEMA,
            "transaction_id": args.transaction_id,
            "phase": "complete",
            "candidate_pointer_sha256": _pointer_sha(root),
            "record_count": plan["record_count"],
            "created_at": _timestamp(args.published_at),
        },
    )
    return {"valid": True, "transaction_id": args.transaction_id, "moved_records": moved}


def command_archive_quarantine_rollback(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve(strict=True)
    _, plan = _load_plan(root, args.transaction_id)
    loaded = load_registered_catalog(root)
    if loaded is None or loaded[0]["generation_id"] != plan["candidate_generation_id"]:
        raise StrategyRecordStoreError("rollback requires the archive-aware candidate pointer")
    moved = _move_records(plan, direction="rollback")
    source = plan["source_pointer"]
    publication = publish_catalog(
        root,
        expected_pointer_sha256=_pointer_sha(root),
        records=plan["source_catalog"]["records"],
        dashboard_projection=plan["source_catalog"].get("dashboard_projection"),
        history_registry=plan["source_catalog"].get("history_registry"),
        history_registry_ref=plan["source_catalog"].get("history_registry_ref"),
        inherit_history_registry=False,
        active_record_id=source["active_record_id"],
        previous_record_id=source["previous_record_id"],
        generation_id=args.generation_id,
        published_at=_timestamp(args.published_at),
        catalog_schema=(
            CATALOG_SCHEMA_V2
            if any(
                row.get("archive_locator") is not None
                for row in plan["source_catalog"]["records"]
            )
            else CATALOG_SCHEMA_V1
        ),
    )
    _write_exact_once(
        _transaction_root(root, args.transaction_id) / "rollback-complete.v1.json",
        {
            "schema_id": TRANSACTION_EVENT_SCHEMA,
            "transaction_id": args.transaction_id,
            "phase": "rollback-complete",
            "pointer_sha256": publication["pointer_sha256"],
            "created_at": _timestamp(args.published_at),
        },
    )
    return {**publication, "transaction_id": args.transaction_id, "restored_records": moved}


def command_archive_quarantine_verify(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.record_root).resolve(strict=True)
    _, plan = _load_plan(root, args.transaction_id)
    quarantine = Path(plan["quarantine_root"])
    source_rows = {row["record_id"]: row for row in plan["source_catalog"]["records"]}
    counts = {"SOURCE": 0, "QUARANTINE": 0}
    for record_id in plan["record_ids"]:
        counts[_verify_location(root, quarantine, source_rows[record_id])] += 1
    if counts["QUARANTINE"] == plan["record_count"]:
        complete = _read_canonical_transaction(
            _transaction_root(root, args.transaction_id) / "complete.v1.json",
            label="quarantine completion receipt",
        )
        if (
            complete.get("phase") != "complete"
            or complete.get("candidate_pointer_sha256") != _pointer_sha(root)
            or complete.get("record_count") != plan["record_count"]
        ):
            raise StrategyRecordStoreError(
                "quarantine completion receipt mismatch"
            )
    return {
        "valid": True,
        "transaction_id": args.transaction_id,
        "source_records": counts["SOURCE"],
        "quarantine_records": counts["QUARANTINE"],
        "dual_records": 0,
        "lost_records": 0,
    }


def _common_record_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record-root", required=True)


def _publication_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generation-id")
    parser.add_argument("--published-at")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    _common_record_root(inventory)
    inventory.add_argument("--json", action="store_true")
    inventory.set_defaults(handler=command_inventory)

    for name, handler in (
        ("bootstrap", command_bootstrap),
        ("bootstrap-live", command_bootstrap_live),
    ):
        child = subparsers.add_parser(name)
        _common_record_root(child)
        child.add_argument("--record-dir", action="append", default=[])
        child.add_argument("--active-record-id")
        child.add_argument("--previous-record-id")
        child.add_argument("--dashboard-projection-json")
        _publication_options(child)
        if name == "bootstrap-live":
            child.add_argument("--project-root")
            child.add_argument("--expected-current-id")
            child.add_argument("--expected-previous-id")
            child.add_argument("--expected-inventory-sha")
        child.set_defaults(handler=handler)
        child.set_defaults(mutating=True)

    publish = subparsers.add_parser("publish")
    _common_record_root(publish)
    publish.add_argument("--records-json", required=True)
    publish.add_argument("--expected-pointer-sha", required=True)
    publish.add_argument("--active-record-id")
    publish.add_argument("--previous-record-id")
    publish.add_argument("--dashboard-projection-json")
    _publication_options(publish)
    publish.set_defaults(handler=command_publish)
    publish.set_defaults(mutating=True)

    stage = subparsers.add_parser("stage-init")
    _common_record_root(stage)
    stage.add_argument("--record-id", required=True)
    stage.set_defaults(handler=command_stage_init)
    stage.set_defaults(mutating=True)

    seal = subparsers.add_parser("seal-publish")
    _common_record_root(seal)
    seal.add_argument("--record-id", required=True)
    seal.add_argument("--expected-pointer-sha", required=True)
    seal.add_argument(
        "--project-root",
        help="Required for governed CN publication and Dashboard revalidation",
    )
    seal.add_argument(
        "--performance-generation-id",
        help="Required when advancing a catalog v3 official financial state",
    )
    seal.add_argument(
        "--cash-flow-artifact",
        action="append",
        default=[],
        metavar="PROJECT_RELATIVE_PATH=SHA256",
        help="Exact owner-declared cash-flow closure for a non-zero funding state",
    )
    _publication_options(seal)
    seal.set_defaults(handler=command_seal_publish)
    seal.set_defaults(mutating=True)

    no_action = subparsers.add_parser("no-action")
    _common_record_root(no_action)
    no_action.add_argument("--receipt-id", required=True)
    no_action.add_argument("--reason", required=True)
    no_action.add_argument("--expected-pointer-sha", required=True)
    _publication_options(no_action)
    no_action.set_defaults(handler=command_no_action)
    no_action.set_defaults(mutating=True)

    verify = subparsers.add_parser("verify")
    _common_record_root(verify)
    verify.set_defaults(handler=command_verify)

    reselect = subparsers.add_parser("reselect-catalog")
    _common_record_root(reselect)
    reselect.add_argument("--expected-current-pointer-sha", required=True)
    reselect.add_argument("--target-generation-id", required=True)
    reselect.add_argument("--target-catalog-path", required=True)
    reselect.add_argument("--target-catalog-sha", required=True)
    reselect.add_argument("--published-at", required=True)
    reselect.add_argument("--owner-approved-by", required=True)
    reselect.add_argument("--approval-reason", required=True)
    reselect.set_defaults(handler=command_reselect_catalog, mutating=True)

    identity = subparsers.add_parser("declare-strategy-identity")
    _common_record_root(identity)
    identity.add_argument("--project-root", required=True)
    identity.add_argument("--identity-path", default=IDENTITY_RELATIVE_PATH)
    identity.add_argument("--declared-at", required=True)
    identity.add_argument("--provenance")
    identity.set_defaults(handler=command_declare_strategy_identity, mutating=True)

    prepare_performance = subparsers.add_parser("prepare-performance-migration")
    _common_record_root(prepare_performance)
    prepare_performance.add_argument("--project-root", required=True)
    prepare_performance.add_argument("--expected-pointer-sha", required=True)
    prepare_performance.add_argument("--performance-generation-id", required=True)
    prepare_performance.add_argument("--identity-path", required=True)
    prepare_performance.add_argument("--identity-sha", required=True)
    prepare_performance.add_argument("--generated-at", required=True)
    prepare_performance.add_argument("--owner-declared-at", required=True)
    prepare_performance.add_argument("--output-dir", required=True)
    prepare_performance.set_defaults(handler=command_prepare_performance_migration)

    seal_owner = subparsers.add_parser("seal-performance-owner-declaration")
    seal_owner.add_argument("--candidate-dir", required=True)
    seal_owner.add_argument("--candidate-receipt-sha", required=True)
    seal_owner.add_argument("--approved-manifest-sha", required=True)
    seal_owner.add_argument("--approved-series-sha", required=True)
    seal_owner.add_argument("--approved-owner-declaration-sha", required=True)
    seal_owner.set_defaults(handler=command_seal_performance_owner_declaration)

    publish_performance = subparsers.add_parser("publish-performance-migration")
    _common_record_root(publish_performance)
    publish_performance.add_argument("--expected-pointer-sha", required=True)
    publish_performance.add_argument("--candidate-dir", required=True)
    publish_performance.add_argument("--candidate-receipt-sha", required=True)
    publish_performance.add_argument("--candidate-manifest-sha", required=True)
    publish_performance.add_argument("--series-sha", required=True)
    publish_performance.add_argument("--owner-declaration-sha", required=True)
    publish_performance.add_argument("--lineage-document-sha", required=True)
    publish_performance.add_argument("--performance-generation-id", required=True)
    publish_performance.add_argument("--catalog-generation-id", required=True)
    publish_performance.add_argument("--published-at", required=True)
    publish_performance.set_defaults(
        handler=command_publish_performance_migration,
        mutating=True,
    )

    archive = subparsers.add_parser("archive-rehearsal")
    _common_record_root(archive)
    archive.add_argument("--before", required=True)
    archive.add_argument("--from-record-id")
    archive.add_argument("--output-root", required=True)
    archive.set_defaults(handler=command_archive_rehearsal)
    archive.set_defaults(mutating=True)

    finalize = subparsers.add_parser("archive-finalize")
    _common_record_root(finalize)
    finalize.add_argument("--project-root", required=True)
    finalize.add_argument("--archive-id", required=True)
    finalize.add_argument("--archive-month", required=True)
    finalize.add_argument("--rehearsal-receipt-json", required=True)
    finalize.add_argument("--expected-pointer-sha", required=True)
    finalize.add_argument("--published-at")
    finalize.set_defaults(handler=command_archive_finalize, mutating=True)

    candidate = subparsers.add_parser("archive-candidate")
    _common_record_root(candidate)
    candidate.add_argument("--project-root", required=True)
    candidate.add_argument("--transaction-id", required=True)
    candidate.add_argument("--archive-manifest", action="append", required=True)
    candidate.add_argument("--before", required=True)
    candidate.add_argument("--expected-pointer-sha", required=True)
    candidate.add_argument("--generation-id", required=True)
    candidate.add_argument("--dashboard-projection-json")
    candidate.add_argument(
        "--history-registry-output",
        help="Optional immutable project-relative output; defaults inside the transaction",
    )
    candidate.add_argument("--protected-hashes-json")
    candidate.add_argument("--source-checkpoint-receipt", required=True)
    candidate.add_argument("--source-checkpoint-receipt-sha", required=True)
    candidate.add_argument(
        "--automation-config-hash",
        action="append",
        required=True,
        help="Repeat exactly: automation=SHA, cn-dashboard=SHA, myquant-cn=SHA, a-2=SHA",
    )
    candidate.add_argument("--published-at")
    candidate.set_defaults(handler=command_archive_candidate, mutating=True)

    for name, handler in (
        ("archive-quarantine-cutover", command_archive_quarantine_cutover),
        ("archive-quarantine-resume", command_archive_quarantine_resume),
        ("archive-quarantine-rollback", command_archive_quarantine_rollback),
        ("archive-quarantine-verify", command_archive_quarantine_verify),
    ):
        child = subparsers.add_parser(name)
        _common_record_root(child)
        child.add_argument("--transaction-id", required=True)
        child.add_argument("--published-at")
        if name == "archive-quarantine-cutover":
            child.add_argument("--publish-only", action="store_true")
        if name == "archive-quarantine-rollback":
            child.add_argument("--generation-id", required=True)
        child.set_defaults(
            handler=handler,
            mutating=name != "archive-quarantine-verify",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "mutating", False):
            with _operation_lock(args.record_root):
                result = args.handler(args)
        else:
            result = args.handler(args)
    except (
        StrategyRecordStoreError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps({"ok": True, **result}, ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
