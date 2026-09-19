"""Append-only owner event inbox and daily closure authority for official NAV.

An empty directory or missing record is never evidence.  A date is closed only
when the pointer-selected immutable generation contains an explicit closure
covering every event dimension named by the standing owner policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Final

from .store import canonical_json_bytes, content_sha256

EVENT_POINTER_SCHEMA: Final = "myquant.strategy_event_pointer.v1"
EVENT_GENERATION_SCHEMA: Final = "myquant.strategy_event_generation.v1"
EVENT_CLOSURE_SCHEMA: Final = "myquant.strategy_event_state_closure.v1"
EVENT_DIMENSIONS: Final = (
    "executions",
    "orders",
    "fills",
    "funding",
    "cost_basis_changes",
    "corporate_actions",
    "manual_changes",
)
EMPTY_POINTER_SHA256: Final = hashlib.sha256(b"").hexdigest()
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StrategyEventStoreError(RuntimeError):
    """Event store contract failure."""


class StrategyEventCASMismatch(StrategyEventStoreError):
    """Event pointer preimage changed."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    body["content_sha256"] = content_sha256(body)
    return body


def _validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    observed = value.get("content_sha256")
    if not isinstance(observed, str) or _SHA.fullmatch(observed) is None:
        raise StrategyEventStoreError(f"{label} content SHA is invalid")
    body = dict(value)
    del body["content_sha256"]
    if observed != content_sha256(body):
        raise StrategyEventStoreError(f"{label} content SHA mismatch")


def _read(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise StrategyEventStoreError(f"{label} is not a regular file")
    first = path.read_bytes()
    if first != path.read_bytes():
        raise StrategyEventStoreError(f"{label} was unstable")
    return first


def _write_exact_once(path: Path, raw: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path, label=label) != raw:
            raise StrategyEventStoreError(f"{label} identity collision")
        return
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def pointer_sha256(root: Path) -> str:
    path = root / "current.v1.json"
    return _sha(_read(path, label="event pointer")) if path.exists() else EMPTY_POINTER_SHA256


def build_empty_closure(
    *,
    trade_date: str,
    sealed_at: str,
    cutoff_at: str,
    policy_ref: Mapping[str, str],
    owner_declaration_ref: Mapping[str, str],
    source_receipt_ref: Mapping[str, str] | None,
) -> dict[str, Any]:
    from .event_contracts import event_date

    day = event_date(trade_date).isoformat()
    value = _seal(
        {
            "schema_id": EVENT_CLOSURE_SCHEMA,
            "trade_date": day,
            "sealed_at": sealed_at,
            "cutoff_at": cutoff_at,
            "status": "CLOSED_EMPTY",
            "dimensions": {
                name: {"status": "CLOSED_EMPTY", "events": []} for name in EVENT_DIMENSIONS
            },
            "policy_ref": dict(policy_ref),
            "owner_declaration_ref": dict(owner_declaration_ref),
            "source_receipt_ref": None if source_receipt_ref is None else dict(source_receipt_ref),
            "late_event_behavior": "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED",
            "actual_holdings_mutation_authority": False,
            "cash_mutation_authority": False,
            "broker_order_trade_authority": False,
        }
    )

    return validate_closure(value)


def validate_closure(value: Mapping[str, Any]) -> dict[str, Any]:
    from .event_contracts import validate_closure_contract

    return validate_closure_contract(value)


def publish_generation(
    root: Path,
    *,
    generation_id: str,
    generated_at: str,
    expected_pointer_sha256: str,
    closures: Sequence[Mapping[str, Any]],
    policy_ref: Mapping[str, str],
) -> dict[str, Any]:
    if type(generation_id) is not str or _ID.fullmatch(generation_id) is None:
        raise StrategyEventStoreError("event generation ID is invalid")
    if type(expected_pointer_sha256) is not str or _SHA.fullmatch(expected_pointer_sha256) is None:
        raise StrategyEventStoreError("expected event pointer SHA is invalid")
    from .event_contracts import validate_generation, source_ref, instant

    source_ref(policy_ref)
    rows = [validate_closure(row) for row in closures]
    rows.sort(key=lambda row: row["trade_date"])
    if not rows or len({row["trade_date"] for row in rows}) != len(rows):
        raise StrategyEventStoreError("event generation dates are empty or duplicated")
    generated = instant(generated_at, label="generation")
    if any(instant(row["sealed_at"], label="closure seal") > generated for row in rows):
        raise StrategyEventStoreError("event generation precedes closure seal")
    existing = None
    if expected_pointer_sha256 != EMPTY_POINTER_SHA256:
        if pointer_sha256(root) != expected_pointer_sha256:
            raise StrategyEventCASMismatch("event pointer preimage mismatch")
        existing = load_generation(root)
        current_by_date = {row["trade_date"]: row for row in existing["closures"]}
        candidate_by_date = {row["trade_date"]: row for row in rows}
        for day, current in current_by_date.items():
            if day not in candidate_by_date:
                raise StrategyEventStoreError("event successor dropped current closure set")
            if candidate_by_date[day] != current:
                raise StrategyEventStoreError("OFFICIAL_CLOSE_RESTATEMENT_REQUIRED")
        if candidate_by_date == current_by_date:
            return {**existing, "no_action": True}
    generation = _seal(
        {
            "schema_id": EVENT_GENERATION_SCHEMA,
            "generation_id": generation_id,
            "generated_at": generated_at,
            "policy_ref": dict(policy_ref),
            "trade_dates": [row["trade_date"] for row in rows],
            "closures": rows,
            "late_event_behavior": "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED",
            "broker_order_trade_authority": False,
        }
    )
    validate_generation(generation)
    generation_raw = canonical_json_bytes(generation)
    relative = f"generations/{generation_id}.v1.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path, label="event generation") != generation_raw:
            raise StrategyEventStoreError("event generation identity collision")
    else:
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        try:
            os.write(fd, generation_raw)
            os.fsync(fd)
        finally:
            os.close(fd)
    pointer = _seal(
        {
            "schema_id": EVENT_POINTER_SCHEMA,
            "generation_id": generation_id,
            "generation": {"path": relative, "sha256": _sha(generation_raw)},
            "trade_dates": generation["trade_dates"],
            "previous_pointer_sha256": (
                None if expected_pointer_sha256 == EMPTY_POINTER_SHA256 else expected_pointer_sha256
            ),
            "broker_order_trade_authority": False,
        }
    )
    pointer_raw = canonical_json_bytes(pointer)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".current.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        observed = pointer_sha256(root)
        if observed != expected_pointer_sha256:
            raise StrategyEventCASMismatch(
                f"event pointer CAS mismatch: expected {expected_pointer_sha256}, "
                f"observed {observed}"
            )
        if expected_pointer_sha256 != EMPTY_POINTER_SHA256:
            previous_raw = _read(root / "current.v1.json", label="event predecessor pointer")
            if _sha(previous_raw) != expected_pointer_sha256:
                raise StrategyEventCASMismatch("event predecessor pointer SHA mismatch")
            _write_exact_once(
                root / "pointer_history" / f"{expected_pointer_sha256}.json",
                previous_raw,
                label="event pointer history",
            )
        temporary = root / f".current.tmp-{os.getpid()}-{secrets.token_hex(4)}"
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        try:
            os.write(fd, pointer_raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, root / "current.v1.json")
    finally:
        os.close(descriptor)
    loaded = load_generation(root)
    return {**loaded, "pointer_sha256": _sha(pointer_raw)}


def _parse_native_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise StrategyEventStoreError(f"event {label} JSON invalid") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise StrategyEventStoreError(f"event {label} bytes are not canonical")
    return value


def load_frozen_generation(
    root: Path, *, pointer_bytes: bytes, expected_pointer_sha256: str
) -> dict[str, Any]:
    """Replay exact caller-retained pointer bytes; no current selection or write."""
    from quant_investor.system.storage import SecureSystemStorage
    from .event_contracts import validate_pointer, validate_generation

    if (
        type(pointer_bytes) is not bytes
        or not pointer_bytes
        or len(pointer_bytes) > 1024 * 1024
        or _sha(pointer_bytes) != expected_pointer_sha256
    ):
        raise StrategyEventStoreError("frozen event pointer SHA mismatch")
    pointer = validate_pointer(_parse_native_bytes(pointer_bytes, label="pointer"))
    reader = SecureSystemStorage(root)
    ref = pointer["generation"]
    stored = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024)
    if stored.byte_sha256 != ref["sha256"]:
        raise StrategyEventStoreError("event generation SHA mismatch")
    generation = validate_generation(
        _parse_native_bytes(stored.data, label="generation"), pointer=pointer
    )
    after = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024)
    if (after.data, after.stat_identity) != (stored.data, stored.stat_identity):
        raise StrategyEventStoreError("event generation changed during frozen read")
    return {
        "pointer": pointer,
        "generation": generation,
        "closures": generation["closures"],
        "pointer_sha256": expected_pointer_sha256,
        "pointer_bytes": pointer_bytes,
        "generation_sha256": stored.byte_sha256,
    }


def load_generation(root: Path) -> dict[str, Any]:
    raw = _read(root / "current.v1.json", label="event pointer")
    value = load_frozen_generation(root, pointer_bytes=raw, expected_pointer_sha256=_sha(raw))
    if _read(root / "current.v1.json", label="event pointer") != raw:
        raise StrategyEventStoreError("event pointer changed during read")
    value.pop("pointer_bytes")
    return value


def load_historical_generation(root: Path, *, expected_pointer_sha256: str) -> dict[str, Any]:
    """Read one exact registered ancestor, never infer it by directory ordering."""
    from quant_investor.system.storage import SecureSystemStorage

    if type(expected_pointer_sha256) is not str or _SHA.fullmatch(expected_pointer_sha256) is None:
        raise StrategyEventStoreError("historical event pointer SHA invalid")
    reader = SecureSystemStorage(root)
    current = reader.read_workspace_file_bytes("current.v1.json", maximum_bytes=1024 * 1024)
    selected = current
    seen: set[str] = set()
    for _ in range(4096):
        from .event_contracts import validate_pointer

        pointer = validate_pointer(_parse_native_bytes(selected.data, label="historical pointer"))
        if (
            pointer.get("schema_id") != EVENT_POINTER_SCHEMA
            or pointer.get("broker_order_trade_authority") is not False
        ):
            raise StrategyEventStoreError("historical event pointer contract invalid")
        if selected.byte_sha256 == expected_pointer_sha256:
            break
        previous = pointer.get("previous_pointer_sha256")
        if type(previous) is not str or _SHA.fullmatch(previous) is None or previous in seen:
            raise StrategyEventStoreError("requested event pointer is not in registered ancestry")
        seen.add(previous)
        selected = reader.read_workspace_file_bytes(
            f"pointer_history/{previous}.json", maximum_bytes=1024 * 1024
        )
        if selected.byte_sha256 != previous:
            raise StrategyEventStoreError("historical event predecessor SHA mismatch")
    else:
        raise StrategyEventStoreError("event pointer ancestry bound exceeded")
    result = load_frozen_generation(
        root, pointer_bytes=selected.data, expected_pointer_sha256=expected_pointer_sha256
    )
    after = reader.read_workspace_file_bytes("current.v1.json", maximum_bytes=1024 * 1024)
    if after.byte_sha256 != current.byte_sha256:
        raise StrategyEventStoreError("event pointer changed during historical read")
    return result


__all__ = [
    "EMPTY_POINTER_SHA256",
    "EVENT_CLOSURE_SCHEMA",
    "EVENT_DIMENSIONS",
    "EVENT_GENERATION_SCHEMA",
    "EVENT_POINTER_SCHEMA",
    "StrategyEventCASMismatch",
    "StrategyEventStoreError",
    "build_empty_closure",
    "load_generation",
    "load_frozen_generation",
    "load_historical_generation",
    "pointer_sha256",
    "publish_generation",
    "validate_closure",
]
