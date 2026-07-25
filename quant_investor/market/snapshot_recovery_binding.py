"""Validate immutable CN market snapshot-recovery evidence.

This module is deliberately version-neutral.  Market maintenance owns the
recovery pointer, intent, receipt and source snapshot; strategy runtimes may
bind those artifacts, but no strategy version owns their contract.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

RECOVERY_POINTER_SCHEMA = "cn-market-snapshot-recovery-pointer.v1"
RECOVERY_INTENT_SCHEMA = "cn-market-snapshot-recovery-intent.v1"
RECOVERY_RECEIPT_SCHEMA = "cn-market-snapshot-recovery-receipt.v1"
RECOVERY_BINDING_SCHEMA = "cn-market-snapshot-recovery-binding.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]

_RECOVERY_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRADE_DATE_RE = re.compile(r"^[0-9]{8}$")

_RECOVERY_POINTER_FIELDS = {
    "schema_version",
    "recovery_id",
    "previous_market_pointer_sha256",
    "source_snapshot_manifest_sha256",
    "acknowledged_trade_date",
    "reason",
    "intent_path",
    "intent_sha256",
    "receipt_path",
}
_RECOVERY_INTENT_FIELDS = {
    "schema_version",
    "recovery_id",
    "market",
    "snapshot_id",
    "created_at",
    "previous_market_pointer_sha256",
    "source_snapshot_manifest_path",
    "source_snapshot_manifest_sha256",
    "acknowledged_trade_date",
    "reason",
    "intent_path",
    "receipt_path",
    "source_validation",
}
_RECOVERY_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "recovery_id",
    "market",
    "snapshot_id",
    "activated_at",
    "previous_market_pointer_sha256",
    "new_market_pointer_sha256",
    "source_snapshot_manifest_path",
    "source_snapshot_manifest_sha256",
    "acknowledged_trade_date",
    "reason",
    "intent_path",
    "intent_sha256",
    "receipt_path",
    "source_validation",
}
_RECOVERY_SOURCE_VALIDATION_FIELDS = {
    "table_inventory_sha256",
    "serving_inventory_sha256",
    "table_logical_rowset_sha256",
    "serving_logical_rowset_sha256",
    "logical_column_names",
    "row_count",
    "key_count",
    "symbol_count",
    "latest_trade_date",
    "exact_date_symbol_count",
    "pit_membership_path",
    "pit_membership_sha256",
    "pit_generation_manifest_path",
    "pit_generation_manifest_sha256",
}


class MarketSnapshotRecoveryBindingError(ValueError):
    """A recovery pointer or one of its immutable artifacts is invalid."""

    exit_code = 2


def _require_finite_json(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise MarketSnapshotRecoveryBindingError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MarketSnapshotRecoveryBindingError(f"non-string JSON key at {path}")
            _require_finite_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")
        return
    raise MarketSnapshotRecoveryBindingError(
        f"unsupported JSON value at {path}: {type(value).__name__}"
    )


def _pairs_without_duplicates(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise MarketSnapshotRecoveryBindingError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's compact, sorted, finite JSON representation."""

    _require_finite_json(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_stable_regular_file(
    descriptor: int,
    *,
    source: str,
    max_bytes: int | None,
) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise MarketSnapshotRecoveryBindingError(f"input is not a regular file: {source}")
    limit = max_bytes if max_bytes is not None else max(before.st_size + 1, 1)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise MarketSnapshotRecoveryBindingError("input byte limit invalid")
    if before.st_size > limit:
        raise MarketSnapshotRecoveryBindingError(f"input exceeds size limit: {source}")

    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > limit:
        raise MarketSnapshotRecoveryBindingError(f"input exceeds size limit: {source}")

    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise MarketSnapshotRecoveryBindingError(f"input changed during read: {source}")
    return raw


def _open_stable_path(
    path: str | Path,
    *,
    max_bytes: int | None,
) -> bytes:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise MarketSnapshotRecoveryBindingError(f"symlink input rejected: {source}") from exc
        raise MarketSnapshotRecoveryBindingError(f"input unavailable: {source}") from exc
    try:
        raw = _read_stable_regular_file(
            descriptor,
            source=str(source),
            max_bytes=max_bytes,
        )
        opened = os.fstat(descriptor)
        try:
            current = os.lstat(source)
        except OSError as exc:
            raise MarketSnapshotRecoveryBindingError(
                f"input changed during read: {source}"
            ) from exc
        if (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise MarketSnapshotRecoveryBindingError(f"input changed during read: {source}")
        return raw
    finally:
        os.close(descriptor)


def file_sha256(path: str | Path) -> str:
    """Hash one stable regular file without following a leaf symlink."""

    return hashlib.sha256(_open_stable_path(path, max_bytes=None)).hexdigest()


def _decode_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                MarketSnapshotRecoveryBindingError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketSnapshotRecoveryBindingError(f"invalid JSON input: {source}") from exc
    if not isinstance(payload, dict):
        raise MarketSnapshotRecoveryBindingError(f"JSON input must be one object: {source}")
    _require_finite_json(payload)
    return payload


def read_json(
    path: str | Path,
    *,
    max_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Read one stable finite JSON object without following a leaf symlink."""

    source = str(path)
    return _decode_json_object(
        _open_stable_path(path, max_bytes=max_bytes),
        source=source,
    )


def _require_safe_relative_path(value: Any, *, label: str) -> Path:
    declared = Path(str(value or ""))
    if (
        not declared.parts
        or declared.is_absolute()
        or any(part in {"", ".", ".."} for part in declared.parts)
    ):
        raise MarketSnapshotRecoveryBindingError(f"{label} path mismatch")
    return declared


def _read_repository_json(
    relative_path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], str]:
    """Read through pinned directory FDs so no path component may be a symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    descriptors: list[int] = []
    entries: list[tuple[int, int, str]] = []
    try:
        try:
            current = os.open(REPO_ROOT, directory_flags)
        except OSError as exc:
            raise MarketSnapshotRecoveryBindingError(
                "repository root unavailable or symlinked"
            ) from exc
        descriptors.append(current)
        for part in relative_path.parts[:-1]:
            try:
                current = os.open(
                    part,
                    directory_flags,
                    dir_fd=current,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise MarketSnapshotRecoveryBindingError(f"{label} symlink rejected") from exc
                raise MarketSnapshotRecoveryBindingError(f"{label} unavailable") from exc
            descriptors.append(current)
            entries.append((descriptors[-2], current, part))
        try:
            artifact = os.open(relative_path.name, flags, dir_fd=current)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise MarketSnapshotRecoveryBindingError(f"{label} symlink rejected") from exc
            raise MarketSnapshotRecoveryBindingError(f"{label} unavailable") from exc
        descriptors.append(artifact)
        entries.append((current, artifact, relative_path.name))
        raw = _read_stable_regular_file(
            artifact,
            source=str(relative_path),
            max_bytes=max_bytes,
        )
        try:
            root_entry = os.lstat(REPO_ROOT)
        except OSError as exc:
            raise MarketSnapshotRecoveryBindingError(f"{label} changed during read") from exc
        root_opened = os.fstat(descriptors[0])
        if not _same_file_identity(root_entry, root_opened):
            raise MarketSnapshotRecoveryBindingError(f"{label} changed during read")
        for parent, child, name in reversed(entries):
            try:
                current_entry = os.stat(
                    name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise MarketSnapshotRecoveryBindingError(f"{label} changed during read") from exc
            if not _same_file_identity(current_entry, os.fstat(child)):
                raise MarketSnapshotRecoveryBindingError(f"{label} changed during read")
        return (
            _decode_json_object(raw, source=str(relative_path)),
            hashlib.sha256(raw).hexdigest(),
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _require_sha256(value: Any, *, label: str) -> str:
    normalized = str(value or "")
    if not _SHA256_RE.fullmatch(normalized):
        raise MarketSnapshotRecoveryBindingError(f"{label} SHA-256 invalid")
    return normalized


def _require_trade_date(value: Any, *, label: str) -> str:
    normalized = str(value or "")
    if not _TRADE_DATE_RE.fullmatch(normalized):
        raise MarketSnapshotRecoveryBindingError(f"{label} date must be YYYYMMDD")
    try:
        datetime.strptime(normalized, "%Y%m%d")
    except ValueError as exc:
        raise MarketSnapshotRecoveryBindingError(f"{label} date must be YYYYMMDD") from exc
    return normalized


def _recovery_artifact_path(
    raw_path: Any,
    *,
    recovery_id: str,
    filename: str,
    label: str,
) -> Path:
    declared = _require_safe_relative_path(raw_path, label=label)
    expected = Path("data/parquet/cn/_recoveries") / recovery_id / filename
    if declared != expected:
        raise MarketSnapshotRecoveryBindingError(f"{label} path mismatch")
    return declared


def _recovery_source_manifest_path(
    raw_path: Any,
    *,
    snapshot_id: str,
) -> Path:
    declared = _require_safe_relative_path(
        raw_path,
        label="recovery source snapshot manifest",
    )
    expected = Path("data/parquet/cn/_snapshots") / f"{snapshot_id}.json"
    if declared != expected:
        raise MarketSnapshotRecoveryBindingError("recovery source snapshot manifest path mismatch")
    return declared


def _validate_recovery_source_validation(
    raw: Any,
    *,
    acknowledged_trade_date: str,
    snapshot_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _RECOVERY_SOURCE_VALIDATION_FIELDS:
        raise MarketSnapshotRecoveryBindingError("recovery source validation schema mismatch")
    validation = dict(raw)
    for key in (
        "table_inventory_sha256",
        "serving_inventory_sha256",
        "table_logical_rowset_sha256",
        "serving_logical_rowset_sha256",
        "pit_membership_sha256",
        "pit_generation_manifest_sha256",
    ):
        _require_sha256(validation.get(key), label=f"recovery {key}")
    if validation["table_logical_rowset_sha256"] != validation["serving_logical_rowset_sha256"]:
        raise MarketSnapshotRecoveryBindingError("recovery table/serving semantic digest mismatch")

    logical_columns = validation.get("logical_column_names")
    if (
        not isinstance(logical_columns, list)
        or not logical_columns
        or any(not isinstance(value, str) or not value for value in logical_columns)
        or len(set(logical_columns)) != len(logical_columns)
    ):
        raise MarketSnapshotRecoveryBindingError("recovery logical column inventory invalid")
    for key in (
        "row_count",
        "key_count",
        "symbol_count",
        "exact_date_symbol_count",
    ):
        value = validation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MarketSnapshotRecoveryBindingError(f"recovery {key} invalid")
    if validation["row_count"] != validation["key_count"]:
        raise MarketSnapshotRecoveryBindingError("recovery row/key count mismatch")
    if validation.get("latest_trade_date") != acknowledged_trade_date:
        raise MarketSnapshotRecoveryBindingError("recovery validation trade date mismatch")

    coverage = snapshot_manifest.get("coverage")
    if not isinstance(coverage, Mapping):
        raise MarketSnapshotRecoveryBindingError("recovery source coverage missing")
    if coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4":
        raise MarketSnapshotRecoveryBindingError("recovery source coverage schema mismatch")
    if coverage.get("complete") is not True or coverage.get("blocking_incomplete_count") != 0:
        raise MarketSnapshotRecoveryBindingError("recovery source coverage is not complete")
    expected_manifest_values = {
        "row_count": snapshot_manifest.get("row_count"),
        "symbol_count": snapshot_manifest.get("symbol_count"),
        "latest_trade_date": snapshot_manifest.get("latest_complete_trade_date"),
        "pit_membership_path": coverage.get("pit_membership_path"),
        "pit_membership_sha256": coverage.get("pit_membership_sha256"),
        "pit_generation_manifest_path": coverage.get("pit_generation_manifest_path"),
        "pit_generation_manifest_sha256": coverage.get("pit_generation_manifest_sha256"),
    }
    for key, expected in expected_manifest_values.items():
        if validation.get(key) != expected:
            raise MarketSnapshotRecoveryBindingError(f"recovery source validation mismatch: {key}")
    return validation


def validate_recovery_pointer_binding(
    pointer_payload: Mapping[str, Any],
    *,
    pointer_sha256: str,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate and bind recovery evidence referenced by a CN market pointer."""

    if not isinstance(pointer_payload, Mapping):
        raise MarketSnapshotRecoveryBindingError("market pointer must be one object")
    if expected_binding is not None and not isinstance(expected_binding, Mapping):
        raise MarketSnapshotRecoveryBindingError("sealed recovery binding must be one object")
    pointer_sha = _require_sha256(pointer_sha256, label="market pointer")
    recovery = pointer_payload.get("recovery")
    if recovery is None:
        if expected_binding is not None:
            raise MarketSnapshotRecoveryBindingError(
                "sealed recovery binding present without recovery pointer"
            )
        return None
    if not isinstance(recovery, Mapping) or set(recovery) != _RECOVERY_POINTER_FIELDS:
        raise MarketSnapshotRecoveryBindingError("market recovery pointer schema mismatch")
    if recovery.get("schema_version") != RECOVERY_POINTER_SCHEMA:
        raise MarketSnapshotRecoveryBindingError("market recovery pointer version mismatch")

    recovery_id = str(recovery.get("recovery_id") or "")
    if not _RECOVERY_ID_RE.fullmatch(recovery_id):
        raise MarketSnapshotRecoveryBindingError("market recovery id invalid")
    previous_pointer_sha = _require_sha256(
        recovery.get("previous_market_pointer_sha256"),
        label="previous market pointer",
    )
    if previous_pointer_sha == pointer_sha:
        raise MarketSnapshotRecoveryBindingError("market recovery pointer did not change")
    source_manifest_sha = _require_sha256(
        recovery.get("source_snapshot_manifest_sha256"),
        label="recovery source snapshot manifest",
    )
    intent_sha = _require_sha256(
        recovery.get("intent_sha256"),
        label="recovery intent",
    )
    acknowledged_trade_date = _require_trade_date(
        recovery.get("acknowledged_trade_date"),
        label="recovery",
    )
    reason = str(recovery.get("reason") or "").strip()
    if not reason:
        raise MarketSnapshotRecoveryBindingError("market recovery reason missing")

    intent_path = _recovery_artifact_path(
        recovery.get("intent_path"),
        recovery_id=recovery_id,
        filename="intent.json",
        label="recovery intent",
    )
    receipt_path = _recovery_artifact_path(
        recovery.get("receipt_path"),
        recovery_id=recovery_id,
        filename="receipt.json",
        label="recovery receipt",
    )
    intent, observed_intent_sha = _read_repository_json(
        intent_path,
        label="recovery intent",
        max_bytes=4 * 1024 * 1024,
    )
    receipt, receipt_sha = _read_repository_json(
        receipt_path,
        label="recovery receipt",
        max_bytes=4 * 1024 * 1024,
    )
    if observed_intent_sha != intent_sha:
        raise MarketSnapshotRecoveryBindingError("recovery intent hash mismatch")
    if (
        set(intent) != _RECOVERY_INTENT_FIELDS
        or intent.get("schema_version") != RECOVERY_INTENT_SCHEMA
    ):
        raise MarketSnapshotRecoveryBindingError("recovery intent schema mismatch")
    if (
        set(receipt) != _RECOVERY_RECEIPT_FIELDS
        or receipt.get("schema_version") != RECOVERY_RECEIPT_SCHEMA
    ):
        raise MarketSnapshotRecoveryBindingError("recovery receipt schema mismatch")
    if receipt.get("status") != "activated":
        raise MarketSnapshotRecoveryBindingError("recovery receipt is not activated")

    snapshot_id = str(pointer_payload.get("snapshot_id") or "")
    if not _SNAPSHOT_ID_RE.fullmatch(snapshot_id):
        raise MarketSnapshotRecoveryBindingError("recovery snapshot id invalid")
    source_manifest_path_text = str(intent.get("source_snapshot_manifest_path") or "")
    source_manifest_path = _recovery_source_manifest_path(
        source_manifest_path_text,
        snapshot_id=snapshot_id,
    )
    source_manifest, observed_manifest_sha = _read_repository_json(
        source_manifest_path,
        label="recovery source snapshot manifest",
        max_bytes=16 * 1024 * 1024,
    )
    if observed_manifest_sha != source_manifest_sha:
        raise MarketSnapshotRecoveryBindingError("recovery source snapshot manifest hash mismatch")
    if (
        source_manifest.get("market") != "CN"
        or source_manifest.get("status") != "OK"
        or source_manifest.get("snapshot_id") != snapshot_id
        or source_manifest.get("manifest_path") != source_manifest_path_text
        or source_manifest.get("latest_complete_trade_date") != acknowledged_trade_date
    ):
        raise MarketSnapshotRecoveryBindingError(
            "recovery source snapshot manifest identity mismatch"
        )
    if (
        pointer_payload.get("manifest_path") != source_manifest_path_text
        or pointer_payload.get("latest_complete_trade_date") != acknowledged_trade_date
    ):
        raise MarketSnapshotRecoveryBindingError(
            "recovery pointer/source snapshot identity mismatch"
        )

    common_expected = {
        "recovery_id": recovery_id,
        "market": "CN",
        "snapshot_id": snapshot_id,
        "previous_market_pointer_sha256": previous_pointer_sha,
        "source_snapshot_manifest_path": source_manifest_path_text,
        "source_snapshot_manifest_sha256": source_manifest_sha,
        "acknowledged_trade_date": acknowledged_trade_date,
        "reason": reason,
        "intent_path": str(recovery.get("intent_path")),
        "receipt_path": str(recovery.get("receipt_path")),
    }
    for label, payload in (("intent", intent), ("receipt", receipt)):
        for key, expected in common_expected.items():
            if payload.get(key) != expected:
                raise MarketSnapshotRecoveryBindingError(
                    f"recovery {label} binding mismatch: {key}"
                )
    if receipt.get("intent_sha256") != intent_sha:
        raise MarketSnapshotRecoveryBindingError("recovery receipt intent hash mismatch")
    if receipt.get("new_market_pointer_sha256") != pointer_sha:
        raise MarketSnapshotRecoveryBindingError("recovery receipt/current pointer hash mismatch")
    if not str(intent.get("created_at") or "") or not str(receipt.get("activated_at") or ""):
        raise MarketSnapshotRecoveryBindingError("recovery evidence timestamp missing")

    intent_validation = _validate_recovery_source_validation(
        intent.get("source_validation"),
        acknowledged_trade_date=acknowledged_trade_date,
        snapshot_manifest=source_manifest,
    )
    receipt_validation = _validate_recovery_source_validation(
        receipt.get("source_validation"),
        acknowledged_trade_date=acknowledged_trade_date,
        snapshot_manifest=source_manifest,
    )
    if receipt_validation != intent_validation:
        raise MarketSnapshotRecoveryBindingError(
            "recovery intent/receipt source validation mismatch"
        )

    binding = {
        "schema_version": RECOVERY_BINDING_SCHEMA,
        "recovery_id": recovery_id,
        "intent_path": str(recovery.get("intent_path")),
        "intent_sha256": intent_sha,
        "receipt_path": str(recovery.get("receipt_path")),
        "receipt_sha256": receipt_sha,
        "previous_market_pointer_sha256": previous_pointer_sha,
        "new_market_pointer_sha256": pointer_sha,
        "source_snapshot_manifest_path": source_manifest_path_text,
        "source_snapshot_manifest_sha256": source_manifest_sha,
        "acknowledged_trade_date": acknowledged_trade_date,
        "restored_trade_date": str(pointer_payload.get("latest_complete_trade_date") or ""),
        "inventory_digests": {
            "table_sha256": intent_validation["table_inventory_sha256"],
            "serving_sha256": intent_validation["serving_inventory_sha256"],
        },
        "semantic_digests": {
            "table_sha256": intent_validation["table_logical_rowset_sha256"],
            "serving_sha256": intent_validation["serving_logical_rowset_sha256"],
        },
        "source_validation_facts": {
            key: intent_validation[key]
            for key in (
                "logical_column_names",
                "row_count",
                "key_count",
                "symbol_count",
                "latest_trade_date",
                "exact_date_symbol_count",
            )
        },
    }
    if expected_binding is not None and dict(expected_binding) != binding:
        raise MarketSnapshotRecoveryBindingError("sealed recovery binding mismatch")
    return binding


__all__ = [
    "MarketSnapshotRecoveryBindingError",
    "RECOVERY_BINDING_SCHEMA",
    "RECOVERY_INTENT_SCHEMA",
    "RECOVERY_POINTER_SCHEMA",
    "RECOVERY_RECEIPT_SCHEMA",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_sha256",
    "read_json",
    "validate_recovery_pointer_binding",
]
