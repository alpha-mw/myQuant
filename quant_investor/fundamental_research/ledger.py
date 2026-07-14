"""Append-only, hash-chained private event ledger with compare-and-swap."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import JobState
from .storage import (
    MAX_JSON_BYTES,
    PRIVATE_MODE,
    _assert_contained,
    canonical_json_bytes,
    sha256_bytes,
)


class LedgerConflictError(RuntimeError):
    pass


_JOB_TRANSITIONS = {
    JobState.PREPARED: {JobState.EXPORTED, JobState.EXPIRED, JobState.SUPERSEDED},
    JobState.EXPORTED: {JobState.RECEIVED, JobState.EXPIRED, JobState.SUPERSEDED},
    JobState.RECEIVED: {
        JobState.VALIDATED,
        JobState.REJECTED,
        JobState.EXPIRED,
        JobState.SUPERSEDED,
    },
    JobState.VALIDATED: {JobState.SUPERSEDED},
    JobState.REJECTED: set(),
    JobState.EXPIRED: set(),
    JobState.SUPERSEDED: set(),
}


def validate_job_transition(previous: JobState | None, next_state: JobState) -> None:
    if previous is None:
        if next_state != JobState.PREPARED:
            raise ValueError("the first job state must be PREPARED")
        return
    if next_state not in _JOB_TRANSITIONS[previous]:
        raise ValueError(f"invalid job transition: {previous.value} -> {next_state.value}")


class HashChainLedger:
    def __init__(self, root: str | Path, path: str | Path) -> None:
        self.root = Path(root)
        _, self.path = _assert_contained(self.root, Path(path), allow_missing_leaf=True)

    @staticmethod
    def _event_hash(payload: dict[str, Any]) -> str:
        return sha256_bytes(canonical_json_bytes(payload))

    def read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        _, path = _assert_contained(self.root, self.path, allow_missing_leaf=False)
        if path.stat().st_mode & 0o777 != PRIVATE_MODE:
            raise ValueError("private ledger permissions are not 0600")
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("private ledger exceeds bounded size")
        records: list[dict[str, Any]] = []
        previous = ""
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank ledger line at {line_number}")
                record = json.loads(line)
                claimed_hash = str(record.pop("event_sha256", ""))
                if record.get("previous_event_sha256", "") != previous:
                    raise ValueError(f"ledger hash chain mismatch at line {line_number}")
                actual_hash = self._event_hash(record)
                if claimed_hash != actual_hash:
                    raise ValueError(f"ledger event hash mismatch at line {line_number}")
                record["event_sha256"] = claimed_hash
                records.append(record)
                previous = claimed_hash
        return records

    def head(self) -> str:
        records = self.read_records()
        return str(records[-1]["event_sha256"]) if records else ""

    def append(self, event: BaseModel, *, expected_head: str) -> str:
        _, path = _assert_contained(self.root, self.path, allow_missing_leaf=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, PRIVATE_MODE)
        try:
            os.fchmod(fd, PRIVATE_MODE)
            with os.fdopen(fd, "r+b", closefd=False) as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                records = self.read_records()
                current_head = str(records[-1]["event_sha256"]) if records else ""
                if current_head != expected_head:
                    raise LedgerConflictError(
                        f"ledger CAS mismatch: expected {expected_head!r}, found {current_head!r}"
                    )
                event_id = str(event.model_dump(mode="json").get("event_id", ""))
                if any(
                    str(item.get("event", {}).get("event_id", "")) == event_id for item in records
                ):
                    raise LedgerConflictError(f"duplicate event_id: {event_id}")
                record = {
                    "schema_version": "fundamental-research-ledger.v1",
                    "previous_event_sha256": current_head,
                    "event": event.model_dump(mode="json"),
                }
                event_hash = self._event_hash(record)
                record["event_sha256"] = event_hash
                encoded = canonical_json_bytes(record)
                if handle.seek(0, os.SEEK_END) + len(encoded) > MAX_JSON_BYTES:
                    raise ValueError("private ledger exceeds bounded size")
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return event_hash
        finally:
            os.close(fd)
