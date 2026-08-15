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

_CONTROLLER_BODY: Final = r"""import hashlib
import json
import os
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
    ).encode("utf-8")


def _read(path, mode):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_uid != os.geteuid()
            or stat.S_IMODE(value.st_mode) != mode
            or value.st_nlink != 1
        ):
            raise ControllerFailure("governed file identity invalid")
        raw = os.read(descriptor, value.st_size + 1)
        if len(raw) != value.st_size:
            raise ControllerFailure("governed file changed during read")
        return raw
    finally:
        os.close(descriptor)


def _parse(raw):
    value = json.loads(raw.decode("utf-8"))
    if raw != _canonical(value):
        raise ControllerFailure("noncanonical JSON")
    return value


def _main():
    manifest_raw = _read(METADATA["generation_manifest_path"], 0o600)
    if hashlib.sha256(manifest_raw).hexdigest() != METADATA["suspended_manifest_sha256"]:
        raise ControllerFailure("suspended manifest bytes differ")
    manifest = _parse(manifest_raw)
    if (
        manifest.get("semantic_sha256") != METADATA["suspended_generation_id"]
        or dict(manifest.get("payload") or {}).get("generation_state")
        != METADATA["target_generation_state"]
    ):
        raise ControllerFailure("suspended manifest semantics differ")
    payload = dict(manifest.get("payload") or {})
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
        raise ControllerFailure("suspended manifest is not minimal")
    if len(sys.argv) != 3:
        raise ControllerFailure("exact prepared pointer path and SHA are required")
    prepared_raw = _read(os.path.abspath(sys.argv[1]), 0o600)
    if hashlib.sha256(prepared_raw).hexdigest() != sys.argv[2]:
        raise ControllerFailure("prepared pointer byte SHA differs")
    pointer = _parse(prepared_raw)
    if (
        type(pointer) is not dict
        or set(pointer) != set(METADATA["pointer_fields"])
        or pointer.get("generation_id") != METADATA["suspended_generation_id"]
        or pointer.get("manifest_sha256") != METADATA["suspended_manifest_sha256"]
        or pointer.get("previous_pointer_sha256")
        == METADATA["empty_pointer_sha256"]
        or pointer.get("os_actor")
        != "uid:%d:emergency-suspend" % os.geteuid()
    ):
        raise ControllerFailure("prepared suspension pointer binding differs")
    active_raw = _read(METADATA["active_pointer_path"], 0o600)
    if hashlib.sha256(active_raw).hexdigest() != pointer["previous_pointer_sha256"]:
        raise ControllerFailure("prepared suspension preimage differs")
    return {
        "code": "SYSTEM_EMERGENCY_PREPARED_POINTER_VERIFIED",
        "pointer_sha256": hashlib.sha256(prepared_raw).hexdigest(),
        "write_performed": False,
    }


if __name__ == "__main__":
    try:
        result = _main()
    except Exception:
        sys.stderr.buffer.write(_canonical({"code": "SYSTEM_EMERGENCY_VERIFY_FAILED"}) + b"\n")
        raise SystemExit(2)
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
    """Verify exact bytes and the fixed target; controller write authority is zero."""

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
        "write_authority": "NONE",
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
    """Install an immutable ``0500`` verification-only incident controller."""

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
