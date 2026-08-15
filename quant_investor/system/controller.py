"""Build and verify the fixed-target stdlib emergency suspension controller."""

from __future__ import annotations

from collections.abc import Mapping
import os
import re
from typing import TYPE_CHECKING, Any, Final

from quant_investor.contracts import (
    SYSTEM_GENERATION_MANIFEST_CONTRACT,
    SYSTEM_GENERATION_MANIFEST_FIELDS,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)

from .errors import SystemContractError
from .storage import (
    ACTIVE_POINTER_PATH,
    EMPTY_POINTER_SHA256,
    GENERATIONS_ROOT,
    POINTER_HISTORY_ROOT,
    SYSTEM_ROOT,
)

if TYPE_CHECKING:
    from .store import SystemStore

CONTROL_ROOT: Final = SYSTEM_ROOT / "control"
EMERGENCY_CONTROLLER_PATH: Final = CONTROL_ROOT / "suspend.py"
EMERGENCY_CONTROLLER_DOMAIN: Final = "myquant-emergency-suspend-controller"
_METADATA_PREFIX: Final = b"# myquant-controller-metadata:"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_POINTER_FIELDS: Final = (
    "activated_at",
    "generation_id",
    "manifest_sha256",
    "os_actor",
    "previous_pointer_sha256",
)
_METADATA_FIELDS: Final = frozenset(
    {
        "active_pointer_path",
        "controller_contract_domain",
        "controller_path",
        "empty_pointer_sha256",
        "generation_manifest_path",
        "manifest_contract_sha256",
        "manifest_payload_fields",
        "pointer_fields",
        "pointer_history_path",
        "suspended_generation_id",
        "suspended_manifest_sha256",
        "target_generation_state",
    }
)

_CONTROLLER_BODY: Final = r"""import datetime
import fcntl
import hashlib
import json
import os
import secrets
import stat
import sys


class ControllerFailure(Exception):
    pass


def _canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ControllerFailure("duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(token):
    raise ControllerFailure("non-finite JSON value")


def _parse(raw):
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except ControllerFailure:
        raise
    except Exception as exc:
        raise ControllerFailure("invalid JSON") from exc
    if raw != _canonical(value):
        raise ControllerFailure("noncanonical JSON")
    return value


def _identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify_file(value, mode):
    if not stat.S_ISREG(value.st_mode):
        raise ControllerFailure("not a regular file")
    if value.st_uid != os.geteuid():
        raise ControllerFailure("owner mismatch")
    if stat.S_IMODE(value.st_mode) != mode:
        raise ControllerFailure("mode mismatch")
    if value.st_nlink != 1:
        raise ControllerFailure("hard-link count mismatch")


def _read(path, optional=False):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = None
    try:
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            if optional:
                return None
            raise ControllerFailure("required file absent") from None
        before = os.fstat(descriptor)
        _verify_file(before, 0o600)
        if before.st_size <= 0 or before.st_size > 8 * 1024 * 1024:
            raise ControllerFailure("file size outside bound")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        _verify_file(after, 0o600)
        if (
            _identity(before) != _identity(after)
            or _identity(after) != _identity(path_after)
            or len(raw) != after.st_size
        ):
            raise ControllerFailure("file changed during read")
        return raw
    except ControllerFailure:
        raise
    except OSError as exc:
        raise ControllerFailure("file cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_self():
    path = os.path.abspath(__file__)
    if path != METADATA["controller_path"]:
        raise ControllerFailure("controller path mismatch")
    _verify_directory(os.path.dirname(path))
    value = os.lstat(path)
    _verify_file(value, 0o500)


def _verify_target_manifest():
    generation_directory = os.path.dirname(METADATA["generation_manifest_path"])
    generations_directory = os.path.dirname(generation_directory)
    _verify_directory(os.path.dirname(generations_directory))
    _verify_directory(generations_directory)
    _verify_directory(generation_directory)
    raw = _read(METADATA["generation_manifest_path"])
    if hashlib.sha256(raw).hexdigest() != METADATA["suspended_manifest_sha256"]:
        raise ControllerFailure("target manifest byte hash mismatch")
    manifest = _parse(raw)
    if type(manifest) is not dict or set(manifest) != {
        "kind",
        "contract_sha256",
        "artifact_id",
        "created_at",
        "payload",
        "semantic_sha256",
    }:
        raise ControllerFailure("target manifest envelope invalid")
    payload = manifest.get("payload")
    if type(payload) is not dict:
        raise ControllerFailure("target manifest payload invalid")
    if (
        manifest.get("kind") != "system.generation_manifest"
        or manifest.get("contract_sha256") != METADATA["manifest_contract_sha256"]
        or manifest.get("semantic_sha256") != METADATA["suspended_generation_id"]
        or manifest.get("artifact_id") != payload.get("assembly_id")
        or set(payload) != set(METADATA["manifest_payload_fields"])
        or payload.get("generation_state") != METADATA["target_generation_state"]
    ):
        raise ControllerFailure("target manifest contract mismatch")
    preimage = {
        "domain": "myquant-artifact",
        "kind": manifest["kind"],
        "contract_sha256": manifest["contract_sha256"],
        "identity_field": "assembly_id",
        "artifact_id": manifest["artifact_id"],
        "created_at": manifest["created_at"],
        "payload": payload,
    }
    if hashlib.sha256(_canonical(preimage)).hexdigest() != manifest["semantic_sha256"]:
        raise ControllerFailure("target manifest semantic hash mismatch")
    if (
        payload.get("source_refs") != []
        or payload.get("factor_source_object_refs") != []
        or payload.get("factor_policy_ref") is not None
        or payload.get("factor_evidence_refs") != []
        or payload.get("factor_active_set_ref") is not None
        or payload.get("factor_validation_attestation_ref") is not None
        or payload.get("mainline_ref") is not None
        or payload.get("research_refs") != []
        or payload.get("migration_receipt_ref") is not None
        or payload.get("migration_marker_ref") is not None
        or payload.get("emergency_controller_sha256") is not None
    ):
        raise ControllerFailure("target manifest is not minimal suspended closure")


def _validate_pointer(raw):
    pointer = _parse(raw)
    if type(pointer) is not dict or set(pointer) != set(METADATA["pointer_fields"]):
        raise ControllerFailure("active pointer fields invalid")
    for field in ("generation_id", "manifest_sha256"):
        value = pointer.get(field)
        if type(value) is not str or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ControllerFailure("active pointer hash invalid")
    previous = pointer.get("previous_pointer_sha256")
    if previous != METADATA["empty_pointer_sha256"] and (
        type(previous) is not str
        or len(previous) != 64
        or any(character not in "0123456789abcdef" for character in previous)
    ):
        raise ControllerFailure("active pointer previous hash invalid")
    if type(pointer.get("activated_at")) is not str or type(pointer.get("os_actor")) is not str:
        raise ControllerFailure("active pointer audit fields invalid")
    return pointer


def _verify_directory(path):
    value = os.lstat(path)
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.geteuid():
        raise ControllerFailure("governed directory invalid")
    if stat.S_IMODE(value.st_mode) != 0o700:
        raise ControllerFailure("governed directory mode invalid")


def _ensure_history_directory():
    system_root = os.path.dirname(METADATA["active_pointer_path"])
    _verify_directory(system_root)
    history = METADATA["pointer_history_path"]
    try:
        os.mkdir(history, 0o700)
    except FileExistsError:
        pass
    _verify_directory(history)


def _write_all(descriptor, raw):
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ControllerFailure("short write")
        view = view[written:]


def _retain_previous(raw, digest):
    _ensure_history_directory()
    path = os.path.join(METADATA["pointer_history_path"], digest + ".json")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = None
    try:
        try:
            descriptor = os.open(path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, raw)
            os.fsync(descriptor)
            _verify_file(os.fstat(descriptor), 0o600)
        except FileExistsError:
            if _read(path) != raw:
                raise ControllerFailure("retained pointer identity conflict") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if _read(path) != raw:
        raise ControllerFailure("retained pointer exact-byte readback mismatch")
    history_fd = os.open(
        METADATA["pointer_history_path"],
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(history_fd)
    finally:
        os.close(history_fd)


def _cas(expected):
    active_path = METADATA["active_pointer_path"]
    active_parent = os.path.dirname(active_path)
    _verify_directory(active_parent)
    lock_path = os.path.join(active_parent, ".active.lock")
    common_lock_flags = (
        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        lock_fd = os.open(lock_path, common_lock_flags | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(lock_fd, 0o600)
    except FileExistsError:
        lock_fd = os.open(lock_path, common_lock_flags)
    temporary = None
    try:
        _verify_file(os.fstat(lock_fd), 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current = _read(active_path, optional=True)
        observed = (
            METADATA["empty_pointer_sha256"]
            if current is None
            else hashlib.sha256(current).hexdigest()
        )
        if observed != expected:
            raise ControllerFailure("active pointer compare-and-swap mismatch")
        if current is not None:
            _validate_pointer(current)
            _retain_previous(current, observed)
        pointer = {
            "generation_id": METADATA["suspended_generation_id"],
            "manifest_sha256": METADATA["suspended_manifest_sha256"],
            "previous_pointer_sha256": expected,
            "activated_at": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "os_actor": "uid:%d:emergency-suspend-controller" % os.geteuid(),
        }
        raw = _canonical(pointer)
        temporary = os.path.join(
            active_parent,
            ".active.emergency-%d-%s" % (os.getpid(), secrets.token_hex(8)),
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        temporary_fd = os.open(temporary, flags, 0o600)
        try:
            os.fchmod(temporary_fd, 0o600)
            _write_all(temporary_fd, raw)
            os.fsync(temporary_fd)
            _verify_file(os.fstat(temporary_fd), 0o600)
        finally:
            os.close(temporary_fd)
        os.replace(temporary, active_path)
        temporary = None
        parent_fd = os.open(
            active_parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        if _read(active_path) != raw:
            raise ControllerFailure("active pointer readback mismatch")
        return {
            "active_pointer_sha256": hashlib.sha256(raw).hexdigest(),
            "generation_id": METADATA["suspended_generation_id"],
            "manifest_sha256": METADATA["suspended_manifest_sha256"],
            "state": METADATA["target_generation_state"],
        }
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        os.close(lock_fd)


def _main():
    _verify_self()
    _verify_target_manifest()
    if len(sys.argv) != 2:
        raise ControllerFailure("exact expected pointer SHA argument required")
    expected = sys.argv[1]
    if expected == METADATA["empty_pointer_sha256"]:
        raise ControllerFailure("emergency suspension requires a non-empty active pointer")
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ControllerFailure("expected pointer SHA invalid")
    return _cas(expected)


if __name__ == "__main__":
    try:
        result = _main()
    except ControllerFailure:
        sys.stderr.buffer.write(_canonical({"code": "SYSTEM_EMERGENCY_CAS_FAILED"}) + b"\n")
        raise SystemExit(2)
    except Exception:
        sys.stderr.buffer.write(_canonical({"code": "SYSTEM_STORAGE_ERROR"}) + b"\n")
        raise SystemExit(3)
    sys.stdout.buffer.write(_canonical(result) + b"\n")
"""


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} must be lowercase SHA-256")
    return value


def _store(value: SystemStore | str | os.PathLike[str]) -> SystemStore:
    from .store import SystemStore as Store

    return value if isinstance(value, Store) else Store(value)


def _metadata(store: SystemStore, generation: Mapping[str, Any]) -> dict[str, Any]:
    generation_id = _require_sha256(generation.get("generation_id"), label="generation_id")
    manifest_sha = _require_sha256(
        generation.get("manifest_sha256"),
        label="manifest_sha256",
    )
    workspace = store.workspace_root
    return {
        "active_pointer_path": str(workspace / str(ACTIVE_POINTER_PATH)),
        "controller_contract_domain": EMERGENCY_CONTROLLER_DOMAIN,
        "controller_path": str(workspace / str(EMERGENCY_CONTROLLER_PATH)),
        "empty_pointer_sha256": EMPTY_POINTER_SHA256,
        "generation_manifest_path": str(
            workspace / str(GENERATIONS_ROOT / generation_id / "manifest.json")
        ),
        "manifest_contract_sha256": SYSTEM_GENERATION_MANIFEST_CONTRACT.contract_sha256,
        "manifest_payload_fields": sorted(SYSTEM_GENERATION_MANIFEST_FIELDS),
        "pointer_fields": list(_POINTER_FIELDS),
        "pointer_history_path": str(workspace / str(POINTER_HISTORY_ROOT)),
        "suspended_generation_id": generation_id,
        "suspended_manifest_sha256": manifest_sha,
        "target_generation_state": "SYSTEM_SUSPENDED",
    }


def _controller_bytes(metadata: Mapping[str, Any]) -> bytes:
    metadata_raw = canonical_json_bytes(dict(metadata))
    metadata_text = metadata_raw.decode("utf-8", errors="strict")
    return (
        b"#!/usr/bin/env python3\n"
        + _METADATA_PREFIX
        + metadata_raw
        + b"\n"
        + f"METADATA = __import__('json').loads({metadata_text!r})\n".encode(
            "utf-8", errors="strict"
        )
        + _CONTROLLER_BODY.encode("utf-8", errors="strict")
    )


def _parse_controller_metadata(raw: bytes) -> dict[str, Any]:
    lines = raw.splitlines()
    if len(lines) < 3 or lines[0] != b"#!/usr/bin/env python3":
        raise SystemContractError("emergency controller header is invalid")
    if not lines[1].startswith(_METADATA_PREFIX):
        raise SystemContractError("emergency controller metadata is absent")
    metadata_raw = lines[1].removeprefix(_METADATA_PREFIX)
    try:
        metadata = parse_canonical_json_bytes(metadata_raw, label="emergency controller metadata")
    except Exception as exc:
        raise SystemContractError("emergency controller metadata is invalid") from exc
    if type(metadata) is not dict or set(metadata) != set(_METADATA_FIELDS):
        raise SystemContractError("emergency controller metadata fields are not exact")
    return metadata


def verify_emergency_controller(
    store_or_workspace_root: SystemStore | str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify mode, owner, bytes, constants, and the fixed suspended target."""

    store = _store(store_or_workspace_root)
    stored = store._storage.read_executable(EMERGENCY_CONTROLLER_PATH)
    if expected_sha256 is not None and stored.byte_sha256 != _require_sha256(
        expected_sha256,
        label="emergency_controller_sha256",
    ):
        raise SystemContractError("emergency controller byte SHA mismatch")
    metadata = _parse_controller_metadata(stored.data)
    generation = store.verify_generation(metadata.get("suspended_generation_id"))
    if generation.get("generation_state") != "SYSTEM_SUSPENDED":
        raise SystemContractError("emergency controller target is not suspended")
    expected_metadata = _metadata(store, generation)
    if metadata != expected_metadata or stored.data != _controller_bytes(expected_metadata):
        raise SystemContractError("emergency controller exact bytes are invalid")
    return {
        "verified": True,
        "path": str(EMERGENCY_CONTROLLER_PATH),
        "byte_sha256": stored.byte_sha256,
        "generation_id": generation["generation_id"],
        "manifest_sha256": generation["manifest_sha256"],
    }


def build_emergency_controller(
    store_or_workspace_root: SystemStore | str | os.PathLike[str],
    *,
    suspended_generation_id: str,
) -> dict[str, Any]:
    """Install an immutable ``0500`` controller pinned to one suspended generation."""

    store = _store(store_or_workspace_root)
    generation = store.verify_generation(
        _require_sha256(suspended_generation_id, label="suspended_generation_id")
    )
    if generation.get("generation_state") != "SYSTEM_SUSPENDED":
        raise SystemContractError("emergency controller requires a suspended generation")
    raw = _controller_bytes(_metadata(store, generation))
    stored = store._storage.write_executable_exact_once(EMERGENCY_CONTROLLER_PATH, raw)
    return verify_emergency_controller(store, expected_sha256=stored.byte_sha256)


__all__ = [
    "CONTROL_ROOT",
    "EMERGENCY_CONTROLLER_DOMAIN",
    "EMERGENCY_CONTROLLER_PATH",
    "build_emergency_controller",
    "verify_emergency_controller",
]
