"""Crash-recoverable removal of the retired event-score catalog column.

This is a post-cutover transaction.  It deliberately does not reuse the
pre-cutover Intelligence mart approval receipt or its compiled evidence
hashes.  Production callers must supply the catalog, market pointer, source
table hashes, and a new run id explicitly.

The source Parquet file is immutable historical evidence.  Apply mode writes a
new generation without the retired column, prepares a durable journal, and
then atomically switches only ``_catalog.json``.  A fresh strict reader proves
that the retired column is no longer visible before the journal is committed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CONFIRM_TOKEN = "RETIRE_EVENT_SCORE_SCHEMA_V14"
REMOVED_COLUMN = "intelligence_score"
CATALOG_LOCK_NAME = "._catalog.json.intelligence-retirement.lock"
JOURNAL_SCHEMA_VERSION = "myquant.event-score-catalog-transaction.v14"
GENERATION_MANIFEST_SCHEMA_VERSION = (
    "myquant.event-score-schema-generation.v14"
)

PRODUCTION_REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CATALOG_PATH = Path("data/parquet/cn/_catalog.json")
CANONICAL_MARKET_POINTER_PATH = Path("data/parquet/cn/_latest.json")
CANONICAL_SOURCE_TABLE_PATH = Path(
    "data/parquet/cn/event_daily_score/part.parquet"
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{7,79}$")
_TERMINAL_JOURNAL_STATES = {"committed", "rolled_back"}


class CatalogLockBusy(RuntimeError):
    """Raised when another canonical catalog writer owns the shared lock."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _read_bytes(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, _sha256_bytes(payload)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_new(path: Path, *, mode: int = 0o700) -> None:
    os.mkdir(path, mode)
    _fsync_directory(path.parent)


def _ensure_directory(path: Path, *, mode: int = 0o700) -> None:
    if os.path.lexists(path):
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            raise OSError(f"unsafe directory: {path}")
        return
    path.mkdir(parents=True, mode=mode)
    os.chmod(path, mode)
    _fsync_directory(path.parent)


def _durable_create(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _atomic_write(
    path: Path, payload: bytes, *, mode: int | None = None
) -> None:
    target_mode = mode
    if target_mode is None:
        target_mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.event-schema-v14.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, target_mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _acquire_catalog_lock(lock_path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise CatalogLockBusy(f"catalog lock open failed: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise CatalogLockBusy("catalog lock is not a regular file")
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError, CatalogLockBusy) as exc:
        os.close(fd)
        raise CatalogLockBusy(f"catalog lock unavailable: {exc}") from exc
    return fd


def _release_catalog_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _repo_relative(repo_root: Path, path: Path) -> str:
    root = repo_root.resolve(strict=True)
    resolved = path.resolve(strict=False)
    return resolved.relative_to(root).as_posix()


def _validate_paths(
    *,
    repo_root: Path,
    catalog_path: Path,
    market_pointer_path: Path,
    source_table_path: Path,
) -> list[str]:
    blockers: list[str] = []
    try:
        root = repo_root.resolve(strict=True)
    except OSError as exc:
        return [f"repository_root_unreadable:{exc}"]
    for label, path in {
        "catalog": catalog_path,
        "market_pointer": market_pointer_path,
        "source_table": source_table_path,
    }.items():
        try:
            path.resolve(strict=False).relative_to(root)
        except (OSError, RuntimeError, ValueError):
            blockers.append(f"{label}_path_outside_repository")
        if path.is_symlink():
            blockers.append(f"{label}_symlink_not_allowed")
    if catalog_path.parent != market_pointer_path.parent:
        blockers.append("catalog_and_market_pointer_roots_differ")
    return blockers


def _catalog_residue(catalog: Any) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if "intelligence" in key.lower():
                    findings.append(
                        {"path": child_path, "value": key, "kind": "key"}
                    )
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str) and "intelligence" in value.lower():
            findings.append({"path": path, "value": value, "kind": "value"})

    walk(catalog, "")
    return findings


def _load_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_frame(payload: bytes, *, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"{label} unreadable as Parquet: {exc}") from exc
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} did not produce a DataFrame")
    return frame


def _exact_nonnegative_int(value: Any, expected: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and value == expected
    )


def _baseline_contract(
    *,
    catalog_path: Path,
    market_pointer_path: Path,
    source_table_path: Path,
    expected_catalog_sha256: str,
    expected_market_pointer_sha256: str,
    expected_source_table_sha256: str,
) -> tuple[dict[str, Any], bytes, bytes, pd.DataFrame, list[str]]:
    blockers: list[str] = []
    try:
        catalog_payload, catalog_sha = _read_bytes(catalog_path)
        pointer_payload, pointer_sha = _read_bytes(market_pointer_path)
        source_payload, source_sha = _read_bytes(source_table_path)
    except OSError as exc:
        return (
            {},
            b"",
            b"",
            pd.DataFrame(),
            [f"baseline_evidence_unreadable:{exc}"],
        )
    if catalog_sha != expected_catalog_sha256:
        blockers.append(f"catalog_sha256_mismatch:{catalog_sha}")
    if pointer_sha != expected_market_pointer_sha256:
        blockers.append(f"market_pointer_sha256_mismatch:{pointer_sha}")
    if source_sha != expected_source_table_sha256:
        blockers.append(f"source_table_sha256_mismatch:{source_sha}")
    try:
        catalog = _load_json_object(catalog_payload, label="catalog")
    except ValueError as exc:
        return (
            {},
            catalog_payload,
            source_payload,
            pd.DataFrame(),
            blockers + [str(exc)],
        )
    residues = _catalog_residue(catalog)
    if len(residues) != 1 or residues[0].get("value") != REMOVED_COLUMN:
        blockers.append(f"catalog_residual_contract_mismatch:{residues}")

    tables = catalog.get("tables")
    entry = (
        tables.get("event_daily_score")
        if isinstance(tables, Mapping)
        else None
    )
    if not isinstance(entry, Mapping):
        blockers.append("event_daily_score_catalog_entry_missing")
        return (
            catalog,
            catalog_payload,
            source_payload,
            pd.DataFrame(),
            blockers,
        )
    columns = entry.get("columns")
    if not isinstance(columns, list) or columns.count(REMOVED_COLUMN) != 1:
        blockers.append("event_daily_score_columns_contract_mismatch")
    raw_path = str(entry.get("path") or "").strip()
    if (
        not raw_path
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
    ):
        blockers.append("event_daily_score_source_path_invalid")
    else:
        declared_source = catalog_path.parent / raw_path
        try:
            if declared_source.resolve(
                strict=True
            ) != source_table_path.resolve(strict=True):
                blockers.append("event_daily_score_source_path_mismatch")
        except OSError:
            blockers.append("event_daily_score_source_path_unreadable")
    if str(entry.get("sha256") or "") != expected_source_table_sha256:
        blockers.append("event_daily_score_declared_sha256_mismatch")
    if not _exact_nonnegative_int(
        entry.get("size_bytes"), len(source_payload)
    ):
        blockers.append("event_daily_score_declared_size_mismatch")

    try:
        frame = _load_frame(source_payload, label="source table")
    except ValueError as exc:
        return (
            catalog,
            catalog_payload,
            source_payload,
            pd.DataFrame(),
            blockers + [str(exc)],
        )
    if isinstance(columns, list) and list(frame.columns) != list(columns):
        blockers.append("event_daily_score_declared_columns_mismatch")
    if not _exact_nonnegative_int(entry.get("row_count"), len(frame)):
        blockers.append("event_daily_score_declared_row_count_mismatch")
    return catalog, catalog_payload, source_payload, frame, blockers


def _retired_catalog(
    *,
    catalog: Mapping[str, Any],
    generation_id: str,
    table_relative: str,
    table_sha256: str,
    table_size: int,
    row_count: int,
    columns: list[str],
    manifest_relative: str,
    manifest_sha256: str,
    source_table_sha256: str,
) -> dict[str, Any]:
    output = dict(catalog)
    tables = dict(output["tables"])
    entry = dict(tables["event_daily_score"])
    entry.update(
        {
            "path": table_relative,
            "table_root": str(Path(table_relative).parent),
            "columns": columns,
            "row_count": row_count,
            "sha256": table_sha256,
            "size_bytes": table_size,
            "generation_id": generation_id,
            "generation_manifest": manifest_relative,
            "generation_manifest_sha256": manifest_sha256,
            "retirement_source_sha256": source_table_sha256,
        }
    )
    tables["event_daily_score"] = entry
    output["tables"] = tables
    residue = _catalog_residue(output)
    if residue:
        raise ValueError(
            f"retired catalog still contains residuals: {residue}"
        )
    return output


def _write_parquet(path: Path, frame: pd.DataFrame) -> tuple[bytes, str]:
    frame.to_parquet(path, index=False)
    os.chmod(path, 0o600)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    payload, digest = _read_bytes(path)
    round_trip = _load_frame(payload, label="new generation")
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True),
        round_trip.reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
        check_like=False,
    )
    return payload, digest


def _journal_transition(
    journal_path: Path, journal: Mapping[str, Any], *, state: str, detail: str
) -> dict[str, Any]:
    updated = dict(journal)
    transitions = list(updated.get("transitions", []) or [])
    transitions.append(
        {"state": state, "at_utc": _utc_now(), "detail": detail}
    )
    updated["state"] = state
    updated["transitions"] = transitions
    _atomic_write(journal_path, _json_bytes(updated), mode=0o600)
    return updated


def _load_journal(journal_path: Path) -> dict[str, Any]:
    payload, _digest = _read_bytes(journal_path)
    return _load_json_object(payload, label="transaction journal")


def _journal_contract_blockers(
    *,
    journal: Mapping[str, Any],
    repo_root: Path,
    catalog_path: Path,
    market_pointer_path: Path,
    source_table_path: Path,
    generation_path: Path,
    generation_id: str,
    expected_catalog_sha256: str,
    expected_market_pointer_sha256: str,
    expected_source_table_sha256: str,
) -> list[str]:
    expected = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "run_id": generation_id,
        "catalog_path": _repo_relative(repo_root, catalog_path),
        "market_pointer_path": _repo_relative(repo_root, market_pointer_path),
        "source_table_path": _repo_relative(repo_root, source_table_path),
        "generation_path": _repo_relative(repo_root, generation_path),
        "old_catalog_sha256": expected_catalog_sha256,
        "expected_market_pointer_sha256": expected_market_pointer_sha256,
        "source_table_sha256": expected_source_table_sha256,
    }
    blockers = [
        f"journal_{field}_mismatch"
        for field, value in expected.items()
        if journal.get(field) != value
    ]
    state = str(journal.get("state") or "")
    if state not in {
        "initializing",
        "prepared",
        "switched",
        *_TERMINAL_JOURNAL_STATES,
    }:
        blockers.append("journal_state_invalid")
    if state in {"prepared", "switched", "committed"}:
        for field in (
            "new_catalog_sha256",
            "new_generation_sha256",
            "generation_manifest_sha256",
        ):
            if not _DIGEST_RE.fullmatch(str(journal.get(field) or "")):
                blockers.append(f"journal_{field}_invalid")
        row_count = journal.get("row_count")
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count < 0
        ):
            blockers.append("journal_row_count_invalid")
        retained_columns = journal.get("retained_columns")
        if (
            not isinstance(retained_columns, list)
            or not retained_columns
            or REMOVED_COLUMN in retained_columns
            or any(not isinstance(item, str) for item in retained_columns)
        ):
            blockers.append("journal_retained_columns_invalid")
    return blockers


def _generation_blockers(
    *,
    repo_root: Path,
    catalog_path: Path,
    market_pointer_path: Path,
    source_table_path: Path,
    transaction_path: Path,
    generation_path: Path,
    journal: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    old_catalog_path = transaction_path / "old_catalog.json"
    new_catalog_path = transaction_path / "new_catalog.json"
    table_path = generation_path / "part.parquet"
    manifest_path = generation_path / "manifest.json"
    expected_digests = {
        old_catalog_path: str(journal.get("old_catalog_sha256") or ""),
        new_catalog_path: str(journal.get("new_catalog_sha256") or ""),
        market_pointer_path: str(
            journal.get("expected_market_pointer_sha256") or ""
        ),
        source_table_path: str(journal.get("source_table_sha256") or ""),
        table_path: str(journal.get("new_generation_sha256") or ""),
        manifest_path: str(journal.get("generation_manifest_sha256") or ""),
    }
    payloads: dict[Path, bytes] = {}
    for path, expected_sha in expected_digests.items():
        if not _DIGEST_RE.fullmatch(expected_sha):
            blockers.append(f"bound_digest_invalid:{path.name}")
            continue
        if path.is_symlink():
            blockers.append(f"bound_artifact_symlink:{path}")
            continue
        try:
            payload, digest = _read_bytes(path)
        except OSError as exc:
            blockers.append(f"bound_artifact_unreadable:{path}:{exc}")
            continue
        payloads[path] = payload
        if digest != expected_sha:
            blockers.append(f"bound_artifact_sha256_mismatch:{path}")
    if blockers:
        return blockers

    try:
        old_catalog = _load_json_object(
            payloads[old_catalog_path], label="old catalog"
        )
        new_catalog = _load_json_object(
            payloads[new_catalog_path], label="new catalog"
        )
        manifest = _load_json_object(
            payloads[manifest_path], label="generation manifest"
        )
        source_frame = _load_frame(
            payloads[source_table_path], label="source table"
        )
        new_frame = _load_frame(payloads[table_path], label="new generation")
    except ValueError as exc:
        return [str(exc)]

    old_residue = _catalog_residue(old_catalog)
    if len(old_residue) != 1 or old_residue[0].get("value") != REMOVED_COLUMN:
        blockers.append("journal_old_catalog_residual_contract_mismatch")
    if _catalog_residue(new_catalog):
        blockers.append("journal_new_catalog_contains_residual")
    expected_frame = source_frame.drop(
        columns=[REMOVED_COLUMN], errors="ignore"
    )
    if REMOVED_COLUMN not in source_frame.columns:
        blockers.append("journal_source_removed_column_missing")
    if REMOVED_COLUMN in new_frame.columns:
        blockers.append("journal_new_generation_removed_column_visible")
    try:
        pd.testing.assert_frame_equal(
            expected_frame.reset_index(drop=True),
            new_frame.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        blockers.append(f"new_generation_retained_data_mismatch:{exc}")

    table_relative = _repo_relative(catalog_path.parent, table_path)
    manifest_relative = _repo_relative(catalog_path.parent, manifest_path)
    entry = dict(new_catalog.get("tables", {}) or {}).get("event_daily_score")
    expected_entry_values = {
        "path": table_relative,
        "table_root": str(Path(table_relative).parent),
        "columns": list(new_frame.columns),
        "row_count": len(new_frame),
        "sha256": str(journal["new_generation_sha256"]),
        "size_bytes": len(payloads[table_path]),
        "generation_id": str(journal["run_id"]),
        "generation_manifest": manifest_relative,
        "generation_manifest_sha256": str(
            journal["generation_manifest_sha256"]
        ),
        "retirement_source_sha256": str(journal["source_table_sha256"]),
    }
    if not isinstance(entry, Mapping):
        blockers.append("journal_new_catalog_event_entry_missing")
    else:
        for field, expected_value in expected_entry_values.items():
            if entry.get(field) != expected_value:
                blockers.append(f"journal_new_catalog_{field}_mismatch")

    expected_manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "run_id": str(journal["run_id"]),
        "source_table_path": _repo_relative(repo_root, source_table_path),
        "source_table_sha256": str(journal["source_table_sha256"]),
        "old_catalog_sha256": str(journal["old_catalog_sha256"]),
        "expected_market_pointer_sha256": str(
            journal["expected_market_pointer_sha256"]
        ),
        "output_table_path": _repo_relative(repo_root, table_path),
        "output_table_sha256": str(journal["new_generation_sha256"]),
        "output_size_bytes": len(payloads[table_path]),
        "row_count": len(new_frame),
        "columns": list(new_frame.columns),
        "removed_columns": [REMOVED_COLUMN],
        "source_file_mutated": False,
    }
    for field, expected_value in expected_manifest.items():
        if manifest.get(field) != expected_value:
            blockers.append(f"generation_manifest_{field}_mismatch")
    return blockers


def _cas_catalog(
    *, catalog_path: Path, expected_payload: bytes, replacement_payload: bytes
) -> None:
    current_payload, current_sha = _read_bytes(catalog_path)
    expected_sha = _sha256_bytes(expected_payload)
    if current_sha != expected_sha or current_payload != expected_payload:
        raise RuntimeError(
            f"catalog_cas_mismatch:{current_sha}!={expected_sha}"
        )
    _atomic_write(catalog_path, replacement_payload)
    readback, readback_sha = _read_bytes(catalog_path)
    replacement_sha = _sha256_bytes(replacement_payload)
    if readback_sha != replacement_sha or readback != replacement_payload:
        raise RuntimeError("catalog_replace_readback_mismatch")


def _fresh_reader_probe(
    repo_root: Path,
    *,
    expected_row_count: int,
    expected_columns: Sequence[str],
) -> tuple[bool, str]:
    from quant_investor.market.market_data_reader import MarketDataReader

    try:
        reader = MarketDataReader(
            market="CN", data_root=repo_root / "data", mode_policy="strict"
        )
        frame = reader.read_table("event_daily_score")
    except Exception as exc:
        return False, f"unexpected_error:{type(exc).__name__}:{exc}"
    if REMOVED_COLUMN in frame.columns:
        return False, "retired column remained visible"
    missing = [
        column for column in expected_columns if column not in frame.columns
    ]
    if missing:
        return False, f"retained columns missing:{missing}"
    if len(frame) != expected_row_count:
        return False, f"row count mismatch:{len(frame)}!={expected_row_count}"
    return True, f"rows={len(frame)};removed_column_visible=false"


def _rollback_catalog(
    *,
    catalog_path: Path,
    transaction_path: Path,
    journal_path: Path,
    journal: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    old_payload, old_sha = _read_bytes(transaction_path / "old_catalog.json")
    new_payload, new_sha = _read_bytes(transaction_path / "new_catalog.json")
    current_payload, current_sha = _read_bytes(catalog_path)
    rollback_verified = False
    rollback_detail = ""
    try:
        if current_sha == new_sha and current_payload == new_payload:
            _cas_catalog(
                catalog_path=catalog_path,
                expected_payload=new_payload,
                replacement_payload=old_payload,
            )
        elif current_sha != old_sha or current_payload != old_payload:
            raise RuntimeError(f"rollback_catalog_cas_unsafe:{current_sha}")
        rollback_verified = _read_bytes(catalog_path)[1] == old_sha
        if not rollback_verified:
            raise RuntimeError("rollback_catalog_hash_mismatch")
        rollback_detail = old_sha
        updated = dict(journal)
        updated["rollback_blockers"] = list(blockers)
        _journal_transition(
            journal_path,
            updated,
            state="rolled_back",
            detail="post-switch validation failed",
        )
    except Exception as exc:
        return {
            "status": "critical_catalog_rollback_failed",
            "blockers": [*blockers, str(exc)],
            "rollback_verified": False,
            "rollback_detail": rollback_detail,
        }
    return {
        "status": "rolled_back_post_switch_validation_failed",
        "blockers": list(blockers),
        "rollback_verified": rollback_verified,
        "rollback_detail": rollback_detail,
    }


def _finish_prepared_transaction(
    *,
    repo_root: Path,
    catalog_path: Path,
    market_pointer_path: Path,
    source_table_path: Path,
    transaction_path: Path,
    generation_path: Path,
    journal_path: Path,
    journal: dict[str, Any],
    recovery: bool,
) -> dict[str, Any]:
    staged_path = transaction_path / "staged_generation"
    if not os.path.lexists(generation_path):
        if not staged_path.is_dir() or staged_path.is_symlink():
            return {
                "status": "blocked_generation_missing",
                "blockers": ["neither_staged_nor_published_generation_exists"],
            }
        if generation_path.parent.is_symlink():
            return {
                "status": "blocked_generation_path_unsafe",
                "blockers": ["generation_parent_symlink_not_allowed"],
            }
        os.replace(staged_path, generation_path)
        _fsync_directory(generation_path.parent)
        _fsync_directory(transaction_path)
    elif os.path.lexists(staged_path):
        return {
            "status": "blocked_duplicate_generation_state",
            "blockers": ["staged_and_published_generation_both_exist"],
        }

    blockers = _generation_blockers(
        repo_root=repo_root,
        catalog_path=catalog_path,
        market_pointer_path=market_pointer_path,
        source_table_path=source_table_path,
        transaction_path=transaction_path,
        generation_path=generation_path,
        journal=journal,
    )
    if blockers:
        return {
            "status": "blocked_bound_generation_invalid",
            "blockers": blockers,
        }

    old_payload, old_sha = _read_bytes(transaction_path / "old_catalog.json")
    new_payload, new_sha = _read_bytes(transaction_path / "new_catalog.json")
    current_payload, current_sha = _read_bytes(catalog_path)
    if current_sha == old_sha and current_payload == old_payload:
        _cas_catalog(
            catalog_path=catalog_path,
            expected_payload=old_payload,
            replacement_payload=new_payload,
        )
    elif current_sha != new_sha or current_payload != new_payload:
        return {
            "status": "critical_catalog_state_unbound",
            "blockers": [f"catalog_matches_neither_old_nor_new:{current_sha}"],
        }
    journal = _journal_transition(
        journal_path,
        journal,
        state="switched",
        detail="catalog points to new generation",
    )

    expected_columns = list(journal.get("retained_columns", []) or [])
    expected_rows = int(journal.get("row_count") or -1)
    probe_ok, probe_detail = _fresh_reader_probe(
        repo_root,
        expected_row_count=expected_rows,
        expected_columns=expected_columns,
    )
    post_blockers = _generation_blockers(
        repo_root=repo_root,
        catalog_path=catalog_path,
        market_pointer_path=market_pointer_path,
        source_table_path=source_table_path,
        transaction_path=transaction_path,
        generation_path=generation_path,
        journal=journal,
    )
    post_catalog_payload, post_catalog_sha = _read_bytes(catalog_path)
    if post_catalog_sha != new_sha or post_catalog_payload != new_payload:
        post_blockers.append("post_switch_catalog_changed")
    if not probe_ok:
        post_blockers.append(f"fresh_reader_probe_failed:{probe_detail}")
    if post_blockers:
        result = _rollback_catalog(
            catalog_path=catalog_path,
            transaction_path=transaction_path,
            journal_path=journal_path,
            journal=journal,
            blockers=post_blockers,
        )
        result["fresh_reader_probe"] = {
            "passed": probe_ok,
            "detail": probe_detail,
        }
        return result

    journal = _journal_transition(
        journal_path,
        journal,
        state="committed",
        detail="strict reader and evidence verified",
    )
    return {
        "status": (
            "recovered_committed"
            if recovery
            else "retired_event_score_residual"
        ),
        "blockers": [],
        "rollback_verified": False,
        "output_catalog_sha256": new_sha,
        "new_generation_sha256": str(journal["new_generation_sha256"]),
        "generation_manifest_sha256": str(
            journal["generation_manifest_sha256"]
        ),
        "fresh_reader_probe": {"passed": True, "detail": probe_detail},
    }


def _recover_transaction(
    *,
    repo_root: Path,
    catalog_path: Path,
    market_pointer_path: Path,
    source_table_path: Path,
    transaction_path: Path,
    generation_path: Path,
    generation_id: str,
    expected_catalog_sha256: str,
    expected_market_pointer_sha256: str,
    expected_source_table_sha256: str,
) -> dict[str, Any]:
    journal_path = transaction_path / "journal.json"
    if not journal_path.is_file() or journal_path.is_symlink():
        return {
            "status": "blocked_transaction_journal_missing",
            "blockers": ["existing transaction directory has no safe journal"],
        }
    try:
        journal = _load_journal(journal_path)
    except (OSError, ValueError) as exc:
        return {
            "status": "blocked_transaction_journal_invalid",
            "blockers": [str(exc)],
        }
    blockers = _journal_contract_blockers(
        journal=journal,
        repo_root=repo_root,
        catalog_path=catalog_path,
        market_pointer_path=market_pointer_path,
        source_table_path=source_table_path,
        generation_path=generation_path,
        generation_id=generation_id,
        expected_catalog_sha256=expected_catalog_sha256,
        expected_market_pointer_sha256=expected_market_pointer_sha256,
        expected_source_table_sha256=expected_source_table_sha256,
    )
    if blockers:
        return {
            "status": "blocked_transaction_journal_contract",
            "blockers": blockers,
        }
    state = str(journal["state"])
    if state == "initializing":
        current_sha = _read_bytes(catalog_path)[1]
        if current_sha != expected_catalog_sha256:
            return {
                "status": "critical_initializing_catalog_changed",
                "blockers": [
                    f"initializing_catalog_sha256_mismatch:{current_sha}"
                ],
            }
        _journal_transition(
            journal_path,
            journal,
            state="rolled_back",
            detail=(
                "recovery stopped incomplete initialization before "
                "catalog switch"
            ),
        )
        return {
            "status": "recovered_initialization_rolled_back",
            "blockers": [],
            "rollback_verified": True,
        }
    if state == "rolled_back":
        return {
            "status": "blocked_run_id_already_rolled_back",
            "blockers": ["use a new run id after reviewing rollback evidence"],
        }
    if state == "committed":
        blockers = _generation_blockers(
            repo_root=repo_root,
            catalog_path=catalog_path,
            market_pointer_path=market_pointer_path,
            source_table_path=source_table_path,
            transaction_path=transaction_path,
            generation_path=generation_path,
            journal=journal,
        )
        current_sha = _read_bytes(catalog_path)[1]
        if current_sha != str(journal.get("new_catalog_sha256") or ""):
            blockers.append("committed_catalog_sha256_mismatch")
        if blockers:
            return {
                "status": "critical_committed_evidence_invalid",
                "blockers": blockers,
            }
        probe_ok, probe_detail = _fresh_reader_probe(
            repo_root,
            expected_row_count=int(journal["row_count"]),
            expected_columns=list(journal["retained_columns"]),
        )
        return {
            "status": (
                "already_retired"
                if probe_ok
                else "critical_committed_probe_failed"
            ),
            "blockers": [] if probe_ok else [probe_detail],
            "fresh_reader_probe": {"passed": probe_ok, "detail": probe_detail},
            "output_catalog_sha256": str(journal["new_catalog_sha256"]),
            "new_generation_sha256": str(journal["new_generation_sha256"]),
        }
    return _finish_prepared_transaction(
        repo_root=repo_root,
        catalog_path=catalog_path,
        market_pointer_path=market_pointer_path,
        source_table_path=source_table_path,
        transaction_path=transaction_path,
        generation_path=generation_path,
        journal_path=journal_path,
        journal=journal,
        recovery=True,
    )


def _transaction(
    *,
    repo_root: Path,
    catalog_path: Path,
    market_pointer_path: Path,
    source_table_path: Path,
    generation_id: str,
    expected_catalog_sha256: str,
    expected_market_pointer_sha256: str,
    expected_source_table_sha256: str,
    apply: bool,
    confirm_token: str | None,
) -> dict[str, Any]:
    market_root = catalog_path.parent
    generations_root = market_root / "event_daily_score" / "_generations"
    transactions_root = market_root / "event_daily_score" / "_transactions"
    generation_path = generations_root / generation_id
    transaction_path = transactions_root / generation_id
    report: dict[str, Any] = {
        "schema_version": "myquant.event-score-residual-retirement.v14",
        "apply_requested": apply,
        "run_id": generation_id,
        "catalog_path": str(catalog_path),
        "market_pointer_path": str(market_pointer_path),
        "source_table_path": str(source_table_path),
        "generation_path": str(generation_path),
        "transaction_path": str(transaction_path),
        "expected_catalog_sha256": expected_catalog_sha256,
        "expected_market_pointer_sha256": expected_market_pointer_sha256,
        "expected_source_table_sha256": expected_source_table_sha256,
        "blockers": [],
    }

    config_blockers = _validate_paths(
        repo_root=repo_root,
        catalog_path=catalog_path,
        market_pointer_path=market_pointer_path,
        source_table_path=source_table_path,
    )
    for label, digest in {
        "catalog": expected_catalog_sha256,
        "market_pointer": expected_market_pointer_sha256,
        "source_table": expected_source_table_sha256,
    }.items():
        if not _DIGEST_RE.fullmatch(digest):
            config_blockers.append(f"expected_{label}_sha256_invalid")
    if not _RUN_ID_RE.fullmatch(generation_id):
        config_blockers.append("run_id_invalid")
    if "intelligence" in generation_id.lower():
        config_blockers.append("run_id_must_not_reintroduce_retired_name")
    if config_blockers:
        report["blockers"] = config_blockers
        report["status"] = "blocked_configuration"
        return report

    if apply and os.path.lexists(transaction_path):
        recovered = _recover_transaction(
            repo_root=repo_root,
            catalog_path=catalog_path,
            market_pointer_path=market_pointer_path,
            source_table_path=source_table_path,
            transaction_path=transaction_path,
            generation_path=generation_path,
            generation_id=generation_id,
            expected_catalog_sha256=expected_catalog_sha256,
            expected_market_pointer_sha256=expected_market_pointer_sha256,
            expected_source_table_sha256=expected_source_table_sha256,
        )
        report.update(recovered)
        report["recovery_attempted"] = True
        return report

    catalog, old_catalog_payload, source_payload, source_frame, blockers = (
        _baseline_contract(
            catalog_path=catalog_path,
            market_pointer_path=market_pointer_path,
            source_table_path=source_table_path,
            expected_catalog_sha256=expected_catalog_sha256,
            expected_market_pointer_sha256=expected_market_pointer_sha256,
            expected_source_table_sha256=expected_source_table_sha256,
        )
    )
    report["catalog_residuals"] = _catalog_residue(catalog) if catalog else []
    report["source_row_count"] = len(source_frame)
    report["source_columns"] = list(source_frame.columns)
    report["blockers"].extend(blockers)
    if blockers:
        report["status"] = "blocked_baseline_contract"
        return report
    retained = source_frame.drop(columns=[REMOVED_COLUMN])
    report["retained_columns"] = list(retained.columns)
    report["old_source_preserved_sha256"] = _sha256_bytes(source_payload)
    if not apply:
        report["status"] = "would_retire_event_score_residual"
        report["repository_writes"] = []
        return report
    if confirm_token != CONFIRM_TOKEN:
        report["blockers"].append("static_confirmation_token_required")
        report["status"] = "blocked_confirmation_required"
        return report
    if os.path.lexists(transaction_path) or os.path.lexists(generation_path):
        report["blockers"].append(
            "run_id_path_preexists_without_recoverable_transaction"
        )
        report["status"] = "blocked_run_id_collision"
        return report

    _ensure_directory(generations_root.parent)
    _ensure_directory(generations_root)
    _ensure_directory(transactions_root)
    _mkdir_new(transaction_path)
    journal_path = transaction_path / "journal.json"
    journal: dict[str, Any] = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "state": "initializing",
        "run_id": generation_id,
        "catalog_path": _repo_relative(repo_root, catalog_path),
        "market_pointer_path": _repo_relative(repo_root, market_pointer_path),
        "source_table_path": _repo_relative(repo_root, source_table_path),
        "generation_path": _repo_relative(repo_root, generation_path),
        "old_catalog_sha256": expected_catalog_sha256,
        "expected_market_pointer_sha256": expected_market_pointer_sha256,
        "source_table_sha256": expected_source_table_sha256,
        "new_catalog_sha256": "",
        "new_generation_sha256": "",
        "generation_manifest_sha256": "",
        "row_count": len(retained),
        "retained_columns": list(retained.columns),
        "transitions": [
            {
                "state": "initializing",
                "at_utc": _utc_now(),
                "detail": "journal created",
            }
        ],
    }
    _durable_create(journal_path, _json_bytes(journal))
    _durable_create(transaction_path / "old_catalog.json", old_catalog_payload)

    staged_path = transaction_path / "staged_generation"
    _mkdir_new(staged_path)
    new_table_path = staged_path / "part.parquet"
    new_table_payload, new_table_sha = _write_parquet(new_table_path, retained)
    final_table_path = generation_path / "part.parquet"
    final_manifest_path = generation_path / "manifest.json"
    manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": generation_id,
        "source_table_path": _repo_relative(repo_root, source_table_path),
        "source_table_sha256": expected_source_table_sha256,
        "old_catalog_sha256": expected_catalog_sha256,
        "expected_market_pointer_sha256": expected_market_pointer_sha256,
        "output_table_path": _repo_relative(repo_root, final_table_path),
        "output_table_sha256": new_table_sha,
        "output_size_bytes": len(new_table_payload),
        "row_count": len(retained),
        "columns": list(retained.columns),
        "removed_columns": [REMOVED_COLUMN],
        "source_file_mutated": False,
    }
    manifest_payload = _json_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_payload)
    _durable_create(staged_path / "manifest.json", manifest_payload)

    table_relative = _repo_relative(market_root, final_table_path)
    manifest_relative = _repo_relative(market_root, final_manifest_path)
    new_catalog = _retired_catalog(
        catalog=catalog,
        generation_id=generation_id,
        table_relative=table_relative,
        table_sha256=new_table_sha,
        table_size=len(new_table_payload),
        row_count=len(retained),
        columns=list(retained.columns),
        manifest_relative=manifest_relative,
        manifest_sha256=manifest_sha,
        source_table_sha256=expected_source_table_sha256,
    )
    new_catalog_payload = _json_bytes(new_catalog)
    new_catalog_sha = _sha256_bytes(new_catalog_payload)
    _durable_create(transaction_path / "new_catalog.json", new_catalog_payload)
    journal.update(
        {
            "new_catalog_sha256": new_catalog_sha,
            "new_generation_sha256": new_table_sha,
            "generation_manifest_sha256": manifest_sha,
        }
    )
    journal = _journal_transition(
        journal_path,
        journal,
        state="prepared",
        detail="old/new catalog and immutable generation hashes bound",
    )
    result = _finish_prepared_transaction(
        repo_root=repo_root,
        catalog_path=catalog_path,
        market_pointer_path=market_pointer_path,
        source_table_path=source_table_path,
        transaction_path=transaction_path,
        generation_path=generation_path,
        journal_path=journal_path,
        journal=journal,
        recovery=False,
    )
    report.update(result)
    return report


def retire_event_score_catalog_residual(
    *,
    repo_root: Path,
    catalog_path: Path,
    market_pointer_path: Path,
    source_table_path: Path,
    generation_id: str,
    expected_catalog_sha256: str,
    expected_market_pointer_sha256: str,
    expected_source_table_sha256: str,
    apply: bool = False,
    confirm_token: str | None = None,
) -> dict[str, Any]:
    """Run the residual transaction under the shared catalog lock."""

    def run_transaction() -> dict[str, Any]:
        return _transaction(
            repo_root=Path(repo_root),
            catalog_path=Path(catalog_path),
            market_pointer_path=Path(market_pointer_path),
            source_table_path=Path(source_table_path),
            generation_id=str(generation_id),
            expected_catalog_sha256=str(expected_catalog_sha256),
            expected_market_pointer_sha256=str(expected_market_pointer_sha256),
            expected_source_table_sha256=str(expected_source_table_sha256),
            apply=bool(apply),
            confirm_token=confirm_token,
        )

    lock_path = Path(catalog_path).parent / CATALOG_LOCK_NAME
    if not apply:
        result = run_transaction()
        result["catalog_lock"] = {
            "path": str(lock_path),
            "acquired": False,
            "reason": "dry_run",
        }
        return result
    try:
        lock_fd = _acquire_catalog_lock(lock_path)
    except CatalogLockBusy as exc:
        return {
            "schema_version": "myquant.event-score-residual-retirement.v14",
            "apply_requested": True,
            "run_id": generation_id,
            "catalog_lock": {
                "path": str(lock_path),
                "acquired": False,
                "non_blocking": True,
            },
            "blockers": [f"catalog_activation_lock_busy:{exc}"],
            "status": "blocked_catalog_lock_busy",
        }
    try:
        result = run_transaction()
        result["catalog_lock"] = {
            "path": str(lock_path),
            "acquired": True,
            "non_blocking": True,
            "held_through_transaction": True,
        }
        return result
    finally:
        _release_catalog_lock(lock_fd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an immutable event_daily_score generation without the "
            "retired column and atomically switch the canonical catalog."
        )
    )
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument("--expected-market-pointer-sha256", required=True)
    parser.add_argument("--expected-source-table-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-token", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = retire_event_score_catalog_residual(
        repo_root=PRODUCTION_REPO_ROOT,
        catalog_path=PRODUCTION_REPO_ROOT / CANONICAL_CATALOG_PATH,
        market_pointer_path=PRODUCTION_REPO_ROOT
        / CANONICAL_MARKET_POINTER_PATH,
        source_table_path=PRODUCTION_REPO_ROOT / CANONICAL_SOURCE_TABLE_PATH,
        generation_id=args.run_id,
        expected_catalog_sha256=args.expected_catalog_sha256,
        expected_market_pointer_sha256=args.expected_market_pointer_sha256,
        expected_source_table_sha256=args.expected_source_table_sha256,
        apply=args.apply,
        confirm_token=args.confirm_token,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    success = {
        "would_retire_event_score_residual",
        "retired_event_score_residual",
        "recovered_committed",
        "already_retired",
    }
    return 0 if report.get("status") in success else 2


if __name__ == "__main__":
    raise SystemExit(main())
