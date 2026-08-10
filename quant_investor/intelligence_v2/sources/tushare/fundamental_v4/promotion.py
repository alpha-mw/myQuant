"""Durable, exact-write-once Fundamental promotion journal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from ...._core import (
    canonical_bytes,
    common_fields,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from .models import (
    PROMOTION_EVENT_V1,
    FundamentalV4ContractError,
    fundamental_v4_contract,
)

EVENT_ORDER = (
    "INTENT",
    "PRECAS_VALIDATED",
    "CAS_COMMITTED",
    "POSTCHECK_PASSED",
    "ROLLBACK_COMMITTED",
    "TERMINAL",
)
TERMINAL_STATES = frozenset({"PROMOTED", "ROLLED_BACK", "NOT_PROMOTED", "PROMOTION_UNCERTAIN"})
ZERO_SHA256 = "0" * 64

_FIELDS = {
    "attempt_id",
    "authority",
    "decision_protocol",
    "event_at",
    "event_id",
    "event_type",
    "evidence",
    "frozen_v1_manifest_sha256",
    "ordinal",
    "previous_event_sha256",
    "production",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "version",
}
_EVIDENCE_FIELDS = {
    "INTENT": {
        "as_of",
        "authorized_arguments_sha256",
        "candidate_generation_id",
        "candidate_pointer_sha256",
        "expected_old_pointer_sha256",
        "implementation_sha256",
        "manifest_sha256",
        "package_sha256",
        "reconciliation_sha256",
        "scope_sha256",
    },
    "PRECAS_VALIDATED": {
        "expected_pointer_sha256",
        "generation_aggregate_sha256",
        "generation_id",
        "scope_sha256",
    },
    "CAS_COMMITTED": {
        "generation_id",
        "pointer_sha256",
        "previous_pointer_sha256",
    },
    "POSTCHECK_PASSED": {
        "generation_id",
        "pointer_sha256",
        "scope_sha256",
    },
    "ROLLBACK_COMMITTED": {
        "generation_id",
        "pointer_sha256",
        "rolled_back_from_sha256",
    },
    "TERMINAL": {
        "observed_pointer_sha256",
        "state",
    },
}


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise FundamentalV4ContractError(f"{label} must be a positive integer")
    return value


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 8 or not value.isdigit():
        raise FundamentalV4ContractError(f"{label} must be YYYYMMDD")
    return value


def _evidence(event_type: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if event_type not in _EVIDENCE_FIELDS:
        raise FundamentalV4ContractError("promotion event type is invalid")
    row = require_exact_keys(
        value,
        _EVIDENCE_FIELDS[event_type],
        label=f"{event_type} evidence",
    )
    if event_type == "INTENT":
        return {
            "as_of": _session(row["as_of"], label="as_of"),
            "authorized_arguments_sha256": sha256(
                row["authorized_arguments_sha256"],
                label="authorized_arguments_sha256",
            ),
            "candidate_generation_id": identifier(
                row["candidate_generation_id"],
                label="candidate_generation_id",
            ),
            "candidate_pointer_sha256": sha256(
                row["candidate_pointer_sha256"],
                label="candidate_pointer_sha256",
            ),
            "expected_old_pointer_sha256": sha256(
                row["expected_old_pointer_sha256"],
                label="expected_old_pointer_sha256",
            ),
            "implementation_sha256": sha256(
                row["implementation_sha256"],
                label="implementation_sha256",
            ),
            "manifest_sha256": sha256(
                row["manifest_sha256"],
                label="manifest_sha256",
            ),
            "package_sha256": sha256(
                row["package_sha256"],
                label="package_sha256",
            ),
            "reconciliation_sha256": sha256(
                row["reconciliation_sha256"],
                label="reconciliation_sha256",
            ),
            "scope_sha256": sha256(row["scope_sha256"], label="scope_sha256"),
        }
    if event_type == "TERMINAL":
        state = row.get("state")
        if state not in TERMINAL_STATES:
            raise FundamentalV4ContractError("promotion terminal state is invalid")
        return {
            "observed_pointer_sha256": sha256(
                row["observed_pointer_sha256"],
                label="observed_pointer_sha256",
            ),
            "state": state,
        }
    result = dict(row)
    for key, item in result.items():
        if key == "generation_id":
            result[key] = identifier(item, label=key)
        else:
            result[key] = sha256(item, label=key)
    return result


@fundamental_v4_contract
def build_promotion_event(
    *,
    attempt_id: str,
    event_type: str,
    ordinal: int,
    previous_event_sha256: str,
    evidence: Mapping[str, Any],
    event_at: str,
) -> dict[str, Any]:
    observed = timestamp(event_at, label="event_at")
    body = {
        **common_fields(timestamp_value=observed),
        "attempt_id": identifier(attempt_id, label="attempt_id"),
        "event_at": observed,
        "event_type": event_type,
        "evidence": _evidence(event_type, evidence),
        "ordinal": _positive_int(ordinal, label="ordinal"),
        "previous_event_sha256": sha256(
            previous_event_sha256,
            label="previous_event_sha256",
        ),
        "version": PROMOTION_EVENT_V1,
    }
    return seal(body, identity_field="event_id")


@fundamental_v4_contract
def validate_promotion_event(document: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_seal(document, identity_field="event_id")
    require_exact_keys(value, _FIELDS, label="Fundamental promotion event")
    if value.get("version") != PROMOTION_EVENT_V1:
        raise FundamentalV4ContractError("promotion event version mismatch")
    expected = build_promotion_event(
        attempt_id=value["attempt_id"],
        event_type=value["event_type"],
        ordinal=value["ordinal"],
        previous_event_sha256=value["previous_event_sha256"],
        evidence=value["evidence"],
        event_at=value["event_at"],
    )
    if value != expected:
        raise FundamentalV4ContractError("promotion event replay mismatch")
    return value


def _event_bytes(document: Mapping[str, Any]) -> bytes:
    return canonical_bytes(validate_promotion_event(document))


def _validate_chain_transitions(
    rows: Sequence[Mapping[str, Any]],
    *,
    event_types: Sequence[str],
    seen: set[str],
) -> None:
    if "CAS_COMMITTED" in seen and "PRECAS_VALIDATED" not in seen:
        raise FundamentalV4ContractError("CAS event lacks pre-CAS validation")
    if "POSTCHECK_PASSED" in seen and "CAS_COMMITTED" not in seen:
        raise FundamentalV4ContractError("postcheck event lacks CAS")
    if "ROLLBACK_COMMITTED" in seen and "CAS_COMMITTED" not in seen:
        raise FundamentalV4ContractError("rollback event lacks CAS")
    allowed_next = {
        "INTENT": {"PRECAS_VALIDATED", "TERMINAL"},
        "PRECAS_VALIDATED": {"CAS_COMMITTED", "TERMINAL"},
        "CAS_COMMITTED": {"POSTCHECK_PASSED", "ROLLBACK_COMMITTED", "TERMINAL"},
        "POSTCHECK_PASSED": {"TERMINAL"},
        "ROLLBACK_COMMITTED": {"TERMINAL"},
        "TERMINAL": set(),
    }
    if any(right not in allowed_next[left] for left, right in zip(event_types, event_types[1:])):
        raise FundamentalV4ContractError("promotion event transition is invalid")
    terminal = rows[-1] if rows[-1]["event_type"] == "TERMINAL" else None
    if terminal is None:
        return
    terminal_state = terminal["evidence"]["state"]
    if (
        (terminal_state == "PROMOTED" and "POSTCHECK_PASSED" not in seen)
        or (terminal_state == "ROLLED_BACK" and "ROLLBACK_COMMITTED" not in seen)
        or (terminal_state == "NOT_PROMOTED" and "CAS_COMMITTED" in seen)
    ):
        raise FundamentalV4ContractError("promotion terminal proof is inconsistent")


def validate_promotion_event_chain(
    values: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FundamentalV4ContractError("promotion event chain must be a sequence")
    rows = [validate_promotion_event(value) for value in values]
    if not rows or rows[0]["event_type"] != "INTENT":
        raise FundamentalV4ContractError("promotion event chain lacks INTENT")
    attempt_id = rows[0]["attempt_id"]
    previous = ZERO_SHA256
    last_time = ""
    seen: set[str] = set()
    event_types: list[str] = []
    for ordinal, row in enumerate(rows, start=1):
        event_type = row["event_type"]
        if (
            row["attempt_id"] != attempt_id
            or row["ordinal"] != ordinal
            or row["previous_event_sha256"] != previous
            or row["event_at"] < last_time
            or event_type in seen
        ):
            raise FundamentalV4ContractError("promotion event chain continuity failed")
        if event_type == "TERMINAL" and ordinal != len(rows):
            raise FundamentalV4ContractError("promotion terminal event is not final")
        seen.add(event_type)
        event_types.append(event_type)
        previous = hashlib.sha256(_event_bytes(row)).hexdigest()
        last_time = row["event_at"]
    _validate_chain_transitions(rows, event_types=event_types, seen=seen)
    return tuple(rows)


def _safe_journal_root(value: str | Path, *, must_exist: bool) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise FundamentalV4ContractError("promotion journal root must be absolute")
    if must_exist and path.resolve(strict=True) != path:
        raise FundamentalV4ContractError("promotion journal root is unsafe")
    return path


def _assert_private_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise FundamentalV4ContractError("promotion journal directory is unsafe")


def _write_once(path: Path, payload: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FundamentalV4ContractError("promotion journal write stalled")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if path.read_bytes() != payload or stat.S_IMODE(os.lstat(path).st_mode) != 0o600:
        raise FundamentalV4ContractError("promotion journal readback failed")


def create_promotion_journal(
    root: str | Path,
    *,
    intent: Mapping[str, Any],
) -> Path:
    value = validate_promotion_event(intent)
    if value["event_type"] != "INTENT" or value["ordinal"] != 1:
        raise FundamentalV4ContractError("promotion journal must start with INTENT")
    base = _safe_journal_root(root, must_exist=False)
    try:
        os.mkdir(base, 0o700)
    except OSError as exc:
        raise FundamentalV4ContractError("promotion journal creation failed") from exc
    _assert_private_directory(base)
    _write_once(base / "01_INTENT.json", _event_bytes(value))
    return base


def read_promotion_journal(root: str | Path) -> tuple[dict[str, Any], ...]:
    base = _safe_journal_root(root, must_exist=True)
    _assert_private_directory(base)
    names = sorted(item.name for item in os.scandir(base))
    if not names or len(names) != len({name.casefold() for name in names}):
        raise FundamentalV4ContractError("promotion journal files are invalid")
    rows: list[dict[str, Any]] = []
    for ordinal, name in enumerate(names, start=1):
        path = base / name
        metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise FundamentalV4ContractError("promotion journal event is unsafe")
        try:
            value = json.loads(path.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FundamentalV4ContractError("promotion journal event is invalid") from exc
        if canonical_bytes(value) != path.read_bytes():
            raise FundamentalV4ContractError("promotion journal event is not canonical")
        if name != f"{ordinal:02d}_{value.get('event_type')}.json":
            raise FundamentalV4ContractError("promotion journal filename is invalid")
        rows.append(value)
    return validate_promotion_event_chain(rows)


def append_promotion_journal_event(
    root: str | Path,
    *,
    event_type: str,
    evidence: Mapping[str, Any],
    event_at: str,
) -> dict[str, Any]:
    rows = read_promotion_journal(root)
    if rows[-1]["event_type"] == "TERMINAL":
        raise FundamentalV4ContractError("promotion journal is terminal")
    if event_type in {row["event_type"] for row in rows}:
        raise FundamentalV4ContractError("promotion event already exists")
    ordinal = len(rows) + 1
    event = build_promotion_event(
        attempt_id=rows[0]["attempt_id"],
        event_type=event_type,
        ordinal=ordinal,
        previous_event_sha256=hashlib.sha256(_event_bytes(rows[-1])).hexdigest(),
        evidence=evidence,
        event_at=event_at,
    )
    validate_promotion_event_chain([*rows, event])
    _write_once(
        Path(root) / f"{ordinal:02d}_{event_type}.json",
        _event_bytes(event),
    )
    read_promotion_journal(root)
    return event


def classify_promotion_recovery(
    events: Sequence[Mapping[str, Any]],
    *,
    observed_pointer_sha256_first: str,
    observed_pointer_sha256_second: str,
    candidate_generation_valid: bool,
    old_generation_valid: bool,
) -> str:
    rows = validate_promotion_event_chain(events)
    first = sha256(observed_pointer_sha256_first, label="observed pointer first")
    second = sha256(observed_pointer_sha256_second, label="observed pointer second")
    if (
        first != second
        or type(candidate_generation_valid) is not bool
        or type(old_generation_valid) is not bool
    ):
        return "PROMOTION_UNCERTAIN"
    terminal = next((row for row in rows if row["event_type"] == "TERMINAL"), None)
    if terminal is not None:
        if terminal["evidence"]["observed_pointer_sha256"] != first:
            return "PROMOTION_UNCERTAIN"
        return str(terminal["evidence"]["state"])
    by_type = {row["event_type"]: row for row in rows}
    intent = by_type["INTENT"]["evidence"]
    old_sha = intent["expected_old_pointer_sha256"]
    candidate_sha = intent["candidate_pointer_sha256"]
    if first == candidate_sha and candidate_generation_valid and "POSTCHECK_PASSED" in by_type:
        return "PROMOTED"
    if first == old_sha and old_generation_valid and "ROLLBACK_COMMITTED" in by_type:
        return "ROLLED_BACK"
    if first == old_sha and old_generation_valid and "CAS_COMMITTED" not in by_type:
        return "NOT_PROMOTED"
    return "PROMOTION_UNCERTAIN"


__all__ = [
    "EVENT_ORDER",
    "TERMINAL_STATES",
    "ZERO_SHA256",
    "append_promotion_journal_event",
    "build_promotion_event",
    "classify_promotion_recovery",
    "create_promotion_journal",
    "read_promotion_journal",
    "validate_promotion_event",
    "validate_promotion_event_chain",
]
