"""Strict, compare-and-swap storage for the mined-factor registry.

Only explicitly authorized mining and health mutation paths should use this
module. Mutations default to dry-run and require ``write=True`` explicitly.
The store keeps registry-file and factor-record CAS preconditions separate so
an inverse mutation can restore one factor without overwriting unrelated
records changed after the original mutation.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from quant_investor.factors.governance import FactorRecord
from quant_investor.factors.runtime import MinedFactorRegistry


FACTOR_REGISTRY_MUTATION_SCHEMA_VERSION = "factor-registry-mutation.v1"
SUPPORTED_FACTOR_REGISTRY_SCHEMA_VERSION = "mined-factor-registry.v1"
FACTOR_RECORD_FIELDS = frozenset(FactorRecord(name="_").to_dict())


class FactorRegistryStoreError(ValueError):
    """Base error for strict registry storage failures."""


class FactorRegistryMissingError(FactorRegistryStoreError):
    """Raised when a registry required for mutation does not exist."""


class FactorRegistryMalformedError(FactorRegistryStoreError):
    """Raised when the registry is not valid UTF-8 JSON."""


class FactorRegistryValidationError(FactorRegistryStoreError):
    """Raised when registry or record structure is unsafe to mutate."""


class FactorRegistryConflictError(FactorRegistryStoreError):
    """Raised when a registry or record compare-and-swap check fails."""


class _MetadataAbsent:
    def __repr__(self) -> str:
        return "METADATA_ABSENT"


METADATA_ABSENT = _MetadataAbsent()


@dataclass(frozen=True)
class FactorRegistrySnapshot:
    """Strict readback plus file-level and record-level content hashes."""

    path: Path
    registry: MinedFactorRegistry
    payload: dict[str, Any]
    registry_sha256: str
    record_payloads: dict[str, dict[str, Any]]
    record_sha256s: dict[str, str]
    metadata_payload: dict[str, Any]
    metadata_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FactorRegistryValidationError(
            f"factor record is not canonical JSON serializable: {exc}"
        ) from exc
    return text.encode("utf-8")


def factor_record_sha256(record: FactorRecord | Mapping[str, Any]) -> str:
    """Return a stable SHA-256 for one factor record payload."""

    payload = (
        record.to_dict()
        if isinstance(record, FactorRecord)
        else dict(record)
    )
    return _sha256_bytes(_canonical_json_bytes(payload))


def registry_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 for the exact registry metadata object."""

    return _sha256_bytes(_canonical_json_bytes(dict(metadata)))


def _metadata_value_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes({"value": value}))


def registry_file_sha256(path: str | os.PathLike[str]) -> str:
    """Hash the exact on-disk bytes of an existing registry."""

    resolved = Path(path).expanduser()
    if not resolved.exists() or not resolved.is_file():
        raise FactorRegistryMissingError(
            f"factor registry missing: {resolved}"
        )
    return _sha256_bytes(resolved.read_bytes())


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _strict_payload_from_bytes(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FactorRegistryMalformedError(
            f"factor registry is not UTF-8: {path}"
        ) from exc
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        message = (
            exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        )
        raise FactorRegistryMalformedError(
            f"malformed factor registry JSON {path}: {message}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FactorRegistryValidationError(
            f"factor registry must be a JSON object: {path}"
        )
    return dict(payload)


def _validate_registry_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> tuple[MinedFactorRegistry, dict[str, dict[str, Any]]]:
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise FactorRegistryValidationError(
            f"factor registry schema_version is missing: {path}"
        )
    if schema_version != SUPPORTED_FACTOR_REGISTRY_SCHEMA_VERSION:
        raise FactorRegistryValidationError(
            "unsupported factor registry schema_version "
            f"{schema_version!r}; expected "
            f"{SUPPORTED_FACTOR_REGISTRY_SCHEMA_VERSION!r}: {path}"
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise FactorRegistryValidationError(
            f"factor registry metadata must be an object: {path}"
        )
    if any(not isinstance(key, str) or not key for key in metadata):
        raise FactorRegistryValidationError(
            f"factor registry metadata keys must be non-empty strings: {path}"
        )
    rows = payload.get("factors")
    if not isinstance(rows, list):
        raise FactorRegistryValidationError(
            f"factor registry factors must be a list: {path}"
        )

    record_payloads: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FactorRegistryValidationError(
                f"factor registry record {index} must be an object: {path}"
            )
        record_payload = dict(row)
        name = str(record_payload.get("name", "") or "").strip()
        if not name:
            raise FactorRegistryValidationError(
                f"factor registry record {index} has no name: {path}"
            )
        unknown_fields = sorted(set(record_payload) - FACTOR_RECORD_FIELDS)
        if unknown_fields:
            raise FactorRegistryValidationError(
                f"factor registry record {name} has unsupported fields: "
                f"{unknown_fields}"
            )
        if name in record_payloads:
            raise FactorRegistryValidationError(
                f"duplicate factor name in registry: {name}"
            )
        try:
            parsed = FactorRecord.from_dict(record_payload)
            parsed.to_dict()
            factor_record_sha256(record_payload)
        except (TypeError, ValueError) as exc:
            raise FactorRegistryValidationError(
                f"invalid factor registry record {name}: {exc}"
            ) from exc
        record_payloads[name] = record_payload

    try:
        registry = MinedFactorRegistry.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise FactorRegistryValidationError(
            f"invalid factor registry payload {path}: {exc}"
        ) from exc
    if len(registry.factors) != len(record_payloads):
        raise FactorRegistryValidationError(
            f"factor registry parsing dropped one or more records: {path}"
        )
    return registry, record_payloads


def load_registry_snapshot_strict(
    path: str | os.PathLike[str],
) -> FactorRegistrySnapshot:
    """Load without the runtime reader's forgiving empty fallback."""

    resolved = Path(path).expanduser()
    if not resolved.exists() or not resolved.is_file():
        raise FactorRegistryMissingError(
            f"factor registry missing: {resolved}"
        )
    raw = resolved.read_bytes()
    payload = _strict_payload_from_bytes(raw, resolved)
    registry, record_payloads = _validate_registry_payload(
        payload,
        path=resolved,
    )
    return FactorRegistrySnapshot(
        path=resolved,
        registry=registry,
        payload=payload,
        registry_sha256=_sha256_bytes(raw),
        record_payloads=copy.deepcopy(record_payloads),
        record_sha256s={
            name: factor_record_sha256(record)
            for name, record in record_payloads.items()
        },
        metadata_payload=copy.deepcopy(dict(payload.get("metadata", {}))),
        metadata_sha256=registry_metadata_sha256(
            dict(payload.get("metadata", {}))
        ),
    )


def _record_payload(
    name: str,
    record: FactorRecord | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if record is None:
        return None
    payload = (
        record.to_dict()
        if isinstance(record, FactorRecord)
        else dict(record)
    )
    payload_name = str(payload.get("name", "") or "").strip()
    if payload_name != name:
        raise FactorRegistryValidationError(
            f"patch key {name!r} does not match factor record name "
            f"{payload_name!r}"
        )
    unknown_fields = sorted(set(payload) - FACTOR_RECORD_FIELDS)
    if unknown_fields:
        raise FactorRegistryValidationError(
            f"factor record patch {name} has unsupported fields: "
            f"{unknown_fields}"
        )
    try:
        parsed = FactorRecord.from_dict(payload)
        parsed.to_dict()
        factor_record_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise FactorRegistryValidationError(
            f"invalid factor record patch {name}: {exc}"
        ) from exc
    return copy.deepcopy(payload)


def _serialize_registry_payload(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FactorRegistryValidationError(
            f"factor registry is not JSON serializable: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _serialize_mutation_journal(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FactorRegistryValidationError(
            f"factor registry mutation journal is not JSON serializable: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _atomic_write_mutation_journal(
    path: Path,
    payload: Mapping[str, Any],
    *,
    replace_existing: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        raise FactorRegistryValidationError(
            f"factor registry mutation journal already exists: {path}"
        )
    raw = _serialize_mutation_journal(payload)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not replace_existing:
            raise FactorRegistryValidationError(
                f"factor registry mutation journal already exists: {path}"
            )
        os.replace(tmp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if path.read_bytes() != raw:
            raise FactorRegistryStoreError(
                "factor registry mutation journal readback mismatch"
            )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _resolve_write_journal_path(
    registry_path: Path,
    journal_path: str | os.PathLike[str] | None,
) -> Path:
    if journal_path is None or not str(journal_path).strip():
        raise FactorRegistryValidationError(
            "journal_path is required for factor registry writes"
        )
    resolved = Path(journal_path).expanduser()
    if resolved.resolve() == registry_path.resolve():
        raise FactorRegistryValidationError(
            "factor registry mutation journal cannot replace the registry"
        )
    if resolved.parent.resolve() == registry_path.parent.resolve():
        raise FactorRegistryValidationError(
            "factor registry mutation journal must be outside the registry "
            "directory"
        )
    return resolved


def _attach_manifest_metadata(
    manifest: dict[str, Any],
    metadata: Mapping[str, Any] | None,
) -> None:
    for key, value in dict(metadata or {}).items():
        if not isinstance(key, str) or not key:
            raise FactorRegistryValidationError(
                "mutation manifest metadata keys must be non-empty strings"
            )
        if key in manifest:
            raise FactorRegistryValidationError(
                f"mutation manifest metadata cannot override {key}"
            )
        manifest[key] = copy.deepcopy(value)


def _assert_registry_sha(path: Path, expected_sha256: str) -> None:
    actual = registry_file_sha256(path)
    if actual != expected_sha256:
        raise FactorRegistryConflictError(
            "factor registry CAS conflict: "
            f"expected {expected_sha256}, found {actual}"
        )


@contextmanager
def _exclusive_registry_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_replace_registry(
    path: Path,
    raw: bytes,
    *,
    expected_registry_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), path.stat().st_mode & 0o777)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_registry_sha(path, expected_registry_sha256)
        os.replace(tmp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _build_patch_plan(
    snapshot: FactorRegistrySnapshot,
    patches: Mapping[str, FactorRecord | Mapping[str, Any] | None],
    *,
    expected_registry_sha256: str,
    expected_record_sha256s: Mapping[str, str | None],
    mutation_id: str,
    reason: str,
    metadata_updates: Mapping[str, Any] | None = None,
    metadata_delete_keys: Sequence[str] = (),
    expected_metadata_values: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    if not str(mutation_id or "").strip():
        raise FactorRegistryValidationError("mutation_id is required")
    if not str(reason or "").strip():
        raise FactorRegistryValidationError("mutation reason is required")
    resolved_metadata_updates = dict(metadata_updates or {})
    if any(
        not isinstance(key, str) or not key
        for key in resolved_metadata_updates
    ):
        raise FactorRegistryValidationError(
            "metadata update keys must be non-empty strings"
        )
    if any(
        not isinstance(key, str) or not key
        for key in metadata_delete_keys
    ):
        raise FactorRegistryValidationError(
            "metadata delete keys must be non-empty strings"
        )
    resolved_metadata_deletes = set(metadata_delete_keys)
    metadata_keys = set(resolved_metadata_updates) | resolved_metadata_deletes
    if set(resolved_metadata_updates) & resolved_metadata_deletes:
        raise FactorRegistryValidationError(
            "metadata keys cannot be both updated and deleted"
        )
    resolved_expected_metadata = dict(expected_metadata_values or {})
    if any(
        not isinstance(key, str) or not key
        for key in resolved_expected_metadata
    ):
        raise FactorRegistryValidationError(
            "expected metadata keys must be non-empty strings"
        )
    if metadata_keys != set(resolved_expected_metadata):
        raise FactorRegistryValidationError(
            "expected_metadata_values must contain exactly the patched "
            "metadata keys"
        )
    if not patches and not metadata_keys:
        raise FactorRegistryValidationError(
            "at least one record or metadata patch is required"
        )
    if not str(expected_registry_sha256 or "").strip():
        raise FactorRegistryValidationError(
            "expected_registry_sha256 is required"
        )
    if snapshot.registry_sha256 != expected_registry_sha256:
        raise FactorRegistryConflictError(
            "factor registry CAS conflict: "
            f"expected {expected_registry_sha256}, found "
            f"{snapshot.registry_sha256}"
        )

    patch_names = {str(name) for name in patches}
    expected_names = {str(name) for name in expected_record_sha256s}
    if patch_names != expected_names:
        raise FactorRegistryValidationError(
            "expected_record_sha256s must contain exactly the patched factor "
            "names"
        )

    normalized_patches: dict[str, dict[str, Any] | None] = {}
    changes: list[dict[str, Any]] = []
    for name in sorted(patch_names):
        current = snapshot.record_payloads.get(name)
        current_sha = snapshot.record_sha256s.get(name)
        expected_sha = expected_record_sha256s[name]
        if expected_sha is None:
            if current is not None:
                raise FactorRegistryConflictError(
                    f"factor record CAS conflict for {name}: expected absent"
                )
        elif current_sha != expected_sha:
            raise FactorRegistryConflictError(
                f"factor record CAS conflict for {name}: "
                f"expected {expected_sha}, found {current_sha or 'absent'}"
            )

        after = _record_payload(name, patches[name])
        after_sha = factor_record_sha256(after) if after is not None else None
        if current is None and after is None:
            raise FactorRegistryValidationError(
                f"factor record patch for {name} is a no-op absence"
            )
        if current_sha == after_sha:
            raise FactorRegistryValidationError(
                f"factor record patch for {name} does not change the record"
            )
        operation = (
            "add"
            if current is None
            else "delete"
            if after is None
            else "update"
        )
        normalized_patches[name] = after
        changes.append(
            {
                "name": name,
                "operation": operation,
                "before_record": copy.deepcopy(current),
                "after_record": copy.deepcopy(after),
                "before_record_sha256": current_sha,
                "after_record_sha256": after_sha,
            }
        )

    metadata_changes: list[dict[str, Any]] = []
    current_metadata = snapshot.metadata_payload
    for key in sorted(metadata_keys):
        before_exists = key in current_metadata
        before_value = copy.deepcopy(current_metadata.get(key))
        expected_value = resolved_expected_metadata[key]
        if expected_value is METADATA_ABSENT:
            if before_exists:
                raise FactorRegistryConflictError(
                    f"registry metadata CAS conflict for {key}: "
                    "expected absent"
                )
        elif not before_exists or (
            _metadata_value_sha256(before_value)
            != _metadata_value_sha256(expected_value)
        ):
            found = (
                _metadata_value_sha256(before_value)
                if before_exists
                else "absent"
            )
            raise FactorRegistryConflictError(
                f"registry metadata CAS conflict for {key}: found {found}"
            )

        after_exists = key not in resolved_metadata_deletes
        after_value = (
            copy.deepcopy(resolved_metadata_updates[key])
            if after_exists
            else None
        )
        if after_exists:
            _metadata_value_sha256(after_value)
        if before_exists == after_exists and (
            not before_exists
            or _metadata_value_sha256(before_value)
            == _metadata_value_sha256(after_value)
        ):
            raise FactorRegistryValidationError(
                f"registry metadata patch for {key} does not change the value"
            )
        metadata_changes.append(
            {
                "key": key,
                "before_exists": before_exists,
                "before_value": before_value,
                "before_value_sha256": (
                    _metadata_value_sha256(before_value)
                    if before_exists
                    else None
                ),
                "after_exists": after_exists,
                "after_value": after_value,
                "after_value_sha256": (
                    _metadata_value_sha256(after_value)
                    if after_exists
                    else None
                ),
            }
        )

    next_payload = copy.deepcopy(snapshot.payload)
    next_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in next_payload["factors"]:
        name = str(row.get("name", "") or "").strip()
        seen.add(name)
        if name not in normalized_patches:
            next_rows.append(dict(row))
            continue
        replacement = normalized_patches[name]
        if replacement is not None:
            next_rows.append(copy.deepcopy(replacement))
    for name in sorted(set(normalized_patches) - seen):
        replacement = normalized_patches[name]
        if replacement is not None:
            next_rows.append(copy.deepcopy(replacement))
    next_payload["factors"] = next_rows
    next_metadata = copy.deepcopy(snapshot.metadata_payload)
    for key in sorted(resolved_metadata_deletes):
        next_metadata.pop(key, None)
    for key, value in resolved_metadata_updates.items():
        next_metadata[str(key)] = copy.deepcopy(value)
    next_payload["metadata"] = next_metadata
    _validate_registry_payload(next_payload, path=snapshot.path)
    serialized = _serialize_registry_payload(next_payload)
    after_registry_sha256 = _sha256_bytes(serialized)

    manifest = {
        "schema_version": FACTOR_REGISTRY_MUTATION_SCHEMA_VERSION,
        "mutation_id": str(mutation_id),
        "reason": str(reason),
        "registry_path": str(snapshot.path),
        "before_registry_sha256": snapshot.registry_sha256,
        "after_registry_sha256": after_registry_sha256,
        "changed_record_count": len(changes),
        "changed_records": changes,
        "changed_metadata_count": len(metadata_changes),
        "metadata_changes": metadata_changes,
        "before_metadata_sha256": snapshot.metadata_sha256,
        "after_metadata_sha256": registry_metadata_sha256(next_metadata),
        "inverse_patch": {
            "records": {
                item["name"]: copy.deepcopy(item["before_record"])
                for item in changes
            },
            "expected_record_sha256s": {
                item["name"]: item["after_record_sha256"]
                for item in changes
            },
            "metadata_operations": [
                {
                    "key": item["key"],
                    "operation": (
                        "set" if item["before_exists"] else "delete"
                    ),
                    "value": copy.deepcopy(item["before_value"]),
                    "expected_current_exists": item["after_exists"],
                    "expected_current_value": copy.deepcopy(
                        item["after_value"]
                    ),
                    "expected_current_value_sha256": item[
                        "after_value_sha256"
                    ],
                }
                for item in metadata_changes
            ],
        },
        "write_requested": False,
        "applied": False,
        "status": "dry_run",
    }
    return next_payload, serialized, manifest


def apply_factor_record_patch(
    path: str | os.PathLike[str],
    patches: Mapping[str, FactorRecord | Mapping[str, Any] | None],
    *,
    expected_registry_sha256: str,
    expected_record_sha256s: Mapping[str, str | None],
    mutation_id: str,
    reason: str,
    metadata_updates: Mapping[str, Any] | None = None,
    metadata_delete_keys: Sequence[str] = (),
    expected_metadata_values: Mapping[str, Any] | None = None,
    manifest_metadata: Mapping[str, Any] | None = None,
    journal_path: str | os.PathLike[str] | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Apply record patches after file-level and record-level CAS checks.

    ``None`` means delete a record.  An expected record hash of ``None`` means
    the record must be absent, which is the only valid precondition for add.
    The default is a read-only dry run.
    """

    resolved = Path(path).expanduser()
    if not write:
        snapshot = load_registry_snapshot_strict(resolved)
        _next, _serialized, manifest = _build_patch_plan(
            snapshot,
            patches,
            expected_registry_sha256=expected_registry_sha256,
            expected_record_sha256s=expected_record_sha256s,
            mutation_id=mutation_id,
            reason=reason,
            metadata_updates=metadata_updates,
            metadata_delete_keys=metadata_delete_keys,
            expected_metadata_values=expected_metadata_values,
        )
        _attach_manifest_metadata(manifest, manifest_metadata)
        return manifest

    resolved_journal = _resolve_write_journal_path(resolved, journal_path)

    # Fail closed before creating the cooperative lock sidecar. The registry is
    # loaded again under the lock before either CAS check or write.
    load_registry_snapshot_strict(resolved)
    with _exclusive_registry_lock(resolved):
        snapshot = load_registry_snapshot_strict(resolved)
        _next, serialized, manifest = _build_patch_plan(
            snapshot,
            patches,
            expected_registry_sha256=expected_registry_sha256,
            expected_record_sha256s=expected_record_sha256s,
            mutation_id=mutation_id,
            reason=reason,
            metadata_updates=metadata_updates,
            metadata_delete_keys=metadata_delete_keys,
            expected_metadata_values=expected_metadata_values,
        )
        _attach_manifest_metadata(manifest, manifest_metadata)
        manifest["write_requested"] = True
        manifest["applied"] = False
        manifest["status"] = "prepared"
        manifest["journal_path"] = str(resolved_journal)
        _atomic_write_mutation_journal(
            resolved_journal,
            manifest,
            replace_existing=False,
        )
        _atomic_replace_registry(
            resolved,
            serialized,
            expected_registry_sha256=snapshot.registry_sha256,
        )
        readback = load_registry_snapshot_strict(resolved)
        if readback.registry_sha256 != manifest["after_registry_sha256"]:
            raise FactorRegistryStoreError(
                "factor registry atomic write readback hash mismatch"
            )
        manifest["write_requested"] = True
        manifest["applied"] = True
        manifest["status"] = "applied"
        manifest["readback_registry_sha256"] = readback.registry_sha256
        _atomic_write_mutation_journal(
            resolved_journal,
            manifest,
            replace_existing=True,
        )
        return manifest


def rollback_factor_record_patch(
    path: str | os.PathLike[str],
    mutation_manifest: Mapping[str, Any],
    *,
    mutation_id: str,
    reason: str = "factor record rollback",
    manifest_metadata: Mapping[str, Any] | None = None,
    journal_path: str | os.PathLike[str] | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Apply a manifest's inverse patch while preserving unrelated changes."""

    manifest = dict(mutation_manifest)
    if (
        manifest.get("schema_version")
        != FACTOR_REGISTRY_MUTATION_SCHEMA_VERSION
    ):
        raise FactorRegistryValidationError(
            "unsupported factor registry mutation manifest"
        )
    manifest_status = str(manifest.get("status", "") or "")
    valid_applied = (
        manifest_status == "applied"
        and bool(manifest.get("write_requested", False))
        and bool(manifest.get("applied", False))
    )
    valid_prepared = (
        manifest_status == "prepared"
        and bool(manifest.get("write_requested", False))
        and not bool(manifest.get("applied", False))
    )
    if not (valid_applied or valid_prepared):
        raise FactorRegistryValidationError(
            "only an applied or prepared write mutation manifest can be "
            "rolled back"
        )
    resolved = Path(path).expanduser()
    manifest_path = Path(str(manifest.get("registry_path", ""))).expanduser()
    if not str(manifest.get("registry_path", "")).strip() or (
        manifest_path.resolve() != resolved.resolve()
    ):
        raise FactorRegistryValidationError(
            "mutation manifest registry_path does not match rollback target"
        )
    inverse = manifest.get("inverse_patch")
    if not isinstance(inverse, Mapping):
        raise FactorRegistryValidationError(
            "mutation manifest has no inverse_patch"
        )
    records = inverse.get("records")
    expected = inverse.get("expected_record_sha256s")
    if not isinstance(records, Mapping) or not isinstance(expected, Mapping):
        raise FactorRegistryValidationError(
            "mutation inverse patch is malformed"
        )
    metadata_operations = inverse.get("metadata_operations", [])
    if not isinstance(metadata_operations, list) or any(
        not isinstance(item, Mapping) for item in metadata_operations
    ):
        raise FactorRegistryValidationError(
            "mutation inverse metadata patch is malformed"
        )
    metadata_updates: dict[str, Any] = {}
    metadata_delete_keys: list[str] = []
    expected_metadata_values: dict[str, Any] = {}
    for raw_item in metadata_operations:
        item = dict(raw_item)
        key = str(item.get("key", "") or "")
        operation = str(item.get("operation", "") or "")
        if not key or operation not in {"set", "delete"}:
            raise FactorRegistryValidationError(
                "mutation inverse metadata operation is malformed"
            )
        if operation == "set":
            metadata_updates[key] = copy.deepcopy(item.get("value"))
        else:
            metadata_delete_keys.append(key)
        expected_metadata_values[key] = (
            copy.deepcopy(item.get("expected_current_value"))
            if bool(item.get("expected_current_exists", False))
            else METADATA_ABSENT
        )

    snapshot = load_registry_snapshot_strict(resolved)
    rollback_metadata = copy.deepcopy(dict(manifest_metadata or {}))
    if "rollback_of" in rollback_metadata:
        raise FactorRegistryValidationError(
            "rollback manifest metadata cannot override rollback_of"
        )
    rollback_metadata["rollback_of"] = str(
        manifest.get("mutation_id", "")
    )
    rollback_manifest = apply_factor_record_patch(
        resolved,
        {str(name): value for name, value in records.items()},
        expected_registry_sha256=snapshot.registry_sha256,
        expected_record_sha256s={
            str(name): (None if value is None else str(value))
            for name, value in expected.items()
        },
        mutation_id=mutation_id,
        reason=reason,
        metadata_updates=metadata_updates,
        metadata_delete_keys=metadata_delete_keys,
        expected_metadata_values=expected_metadata_values,
        manifest_metadata=rollback_metadata,
        journal_path=journal_path,
        write=write,
    )
    return rollback_manifest


__all__ = [
    "FACTOR_REGISTRY_MUTATION_SCHEMA_VERSION",
    "METADATA_ABSENT",
    "FactorRegistryConflictError",
    "FactorRegistryMalformedError",
    "FactorRegistryMissingError",
    "FactorRegistrySnapshot",
    "FactorRegistryStoreError",
    "FactorRegistryValidationError",
    "apply_factor_record_patch",
    "factor_record_sha256",
    "load_registry_snapshot_strict",
    "registry_metadata_sha256",
    "registry_file_sha256",
    "rollback_factor_record_patch",
]
