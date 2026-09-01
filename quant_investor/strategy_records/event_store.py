"""Append-only owner event inbox and daily closure authority for official NAV.

An empty directory or missing record is never evidence.  A date is closed only
when the pointer-selected immutable generation contains an explicit closure
covering every event dimension named by the standing owner policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
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
    day = date.fromisoformat(trade_date).isoformat()
    for value, label in ((sealed_at, "sealed_at"), (cutoff_at, "cutoff_at")):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise StrategyEventStoreError(f"event {label} is invalid") from exc
        if parsed.tzinfo is None:
            raise StrategyEventStoreError(f"event {label} timezone is missing")
    if datetime.fromisoformat(sealed_at.replace("Z", "+00:00")) < datetime.fromisoformat(
        cutoff_at.replace("Z", "+00:00")
    ):
        raise StrategyEventStoreError("event closure precedes owner cutoff")
    refs = [dict(policy_ref), dict(owner_declaration_ref)]
    if source_receipt_ref is not None:
        refs.append(dict(source_receipt_ref))
    for ref in refs:
        if set(ref) != {"path", "sha256"} or _SHA.fullmatch(str(ref.get("sha256"))) is None:
            raise StrategyEventStoreError("event closure ref is invalid")
    return _seal(
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


def validate_closure(value: Mapping[str, Any]) -> dict[str, Any]:
    _validate_seal(value, label="event closure")
    if value.get("schema_id") != EVENT_CLOSURE_SCHEMA or value.get("status") != "CLOSED_EMPTY":
        raise StrategyEventStoreError("event closure status/schema mismatch")
    date.fromisoformat(str(value.get("trade_date")))
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(EVENT_DIMENSIONS):
        raise StrategyEventStoreError("event closure dimensions are incomplete")
    if any(
        not isinstance(dimensions[name], dict)
        or dimensions[name] != {"status": "CLOSED_EMPTY", "events": []}
        for name in EVENT_DIMENSIONS
    ):
        raise StrategyEventStoreError("event closure contains an unclosed dimension")
    if any(
        value.get(name) is not False
        for name in (
            "actual_holdings_mutation_authority",
            "cash_mutation_authority",
            "broker_order_trade_authority",
        )
    ):
        raise StrategyEventStoreError("event closure claims forbidden authority")
    return dict(value)


def publish_generation(
    root: Path,
    *,
    generation_id: str,
    generated_at: str,
    expected_pointer_sha256: str,
    closures: Sequence[Mapping[str, Any]],
    policy_ref: Mapping[str, str],
) -> dict[str, Any]:
    if _ID.fullmatch(generation_id) is None:
        raise StrategyEventStoreError("event generation ID is invalid")
    if _SHA.fullmatch(expected_pointer_sha256) is None:
        raise StrategyEventStoreError("expected event pointer SHA is invalid")
    rows = [validate_closure(row) for row in closures]
    rows.sort(key=lambda row: row["trade_date"])
    if not rows or len({row["trade_date"] for row in rows}) != len(rows):
        raise StrategyEventStoreError("event generation dates are empty or duplicated")
    for ref in (policy_ref,):
        if set(ref) != {"path", "sha256"} or _SHA.fullmatch(str(ref.get("sha256"))) is None:
            raise StrategyEventStoreError("event policy ref is invalid")
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
                f"event pointer CAS mismatch: expected {expected_pointer_sha256}, observed {observed}"
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


def load_generation(root: Path) -> dict[str, Any]:
    pointer_raw = _read(root / "current.v1.json", label="event pointer")
    pointer = json.loads(pointer_raw)
    _validate_seal(pointer, label="event pointer")
    if pointer.get("schema_id") != EVENT_POINTER_SCHEMA:
        raise StrategyEventStoreError("event pointer schema mismatch")
    ref = pointer.get("generation")
    if not isinstance(ref, dict):
        raise StrategyEventStoreError("event generation ref is absent")
    raw = _read(root / str(ref.get("path")), label="event generation")
    if _sha(raw) != ref.get("sha256"):
        raise StrategyEventStoreError("event generation SHA mismatch")
    generation = json.loads(raw)
    _validate_seal(generation, label="event generation")
    if generation.get("schema_id") != EVENT_GENERATION_SCHEMA or generation.get(
        "generation_id"
    ) != pointer.get("generation_id"):
        raise StrategyEventStoreError("event generation closure mismatch")
    closures = [validate_closure(row) for row in generation.get("closures", [])]
    if [row["trade_date"] for row in closures] != pointer.get("trade_dates"):
        raise StrategyEventStoreError("event pointer date set mismatch")
    return {
        "pointer": pointer,
        "generation": generation,
        "closures": closures,
        "pointer_sha256": _sha(pointer_raw),
        "generation_sha256": _sha(raw),
    }


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
    "pointer_sha256",
    "publish_generation",
    "validate_closure",
]
