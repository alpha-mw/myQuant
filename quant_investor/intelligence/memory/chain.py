"""Immutable hash-chain records for successful and failed research cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    ZERO_SHA256,
    IntelligenceContractError,
    assert_no_authority,
    identifier,
    seal_content_addressed,
    sha256,
    timestamp,
    validate_content_addressed,
)

MEMORY_ENTRY_VERSION: Final = "myquant.v17.research-intelligence.memory-entry.v1"
MEMORY_EVENT_TYPES: Final = {
    "EVALUATED",
    "EVIDENCE_ADDED",
    "FAILED_CASE",
    "HYPOTHESIS_CREATED",
    "HYPOTHESIS_FALSIFIED",
    "HYPOTHESIS_SUPPORTED",
}
MEMORY_STATUSES: Final = {"ACTIVE", "FAILED", "FALSIFIED", "SUPPORTED", "UNRESOLVED"}
CONTENT_REF_FIELDS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "semantic_sha256",
}


def _content_refs(values: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise IntelligenceContractError("memory artifact_refs must be non-empty")
    result: list[dict[str, str]] = []
    for index, value in enumerate(values):
        if type(value) is not dict or set(value) != CONTENT_REF_FIELDS:
            raise IntelligenceContractError(f"artifact_refs[{index}] has invalid shape")
        result.append(
            {
                "artifact_id": str(value["artifact_id"]),
                "artifact_version": str(value["artifact_version"]),
                "byte_sha256": sha256(
                    value["byte_sha256"], label=f"artifact_refs[{index}].byte_sha256"
                ),
                "semantic_sha256": sha256(
                    value["semantic_sha256"],
                    label=f"artifact_refs[{index}].semantic_sha256",
                ),
            }
        )
    keys = [(row["artifact_id"], row["byte_sha256"]) for row in result]
    if len(keys) != len(set(keys)):
        raise IntelligenceContractError("duplicate memory artifact refs are rejected")
    return sorted(
        result,
        key=lambda row: (row["artifact_id"].encode(), row["byte_sha256"].encode()),
    )


def memory_tip(entries: Sequence[Mapping[str, Any]]) -> str:
    return ZERO_SHA256 if not entries else str(entries[-1].get("semantic_sha256", ""))


def validate_memory_chain(
    entries: Sequence[Mapping[str, Any]], *, expected_tip: str | None = None
) -> tuple[dict[str, Any], ...]:
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise IntelligenceContractError("memory chain must be a sequence")
    previous = ZERO_SHA256
    normalized: list[dict[str, Any]] = []
    for index, document in enumerate(entries):
        row = validate_content_addressed(document, identity_field="entry_id")
        if row.get("version") != MEMORY_ENTRY_VERSION:
            raise IntelligenceContractError(f"memory entry {index} version mismatch")
        if set(row) != {
            "artifact_refs",
            "authority",
            "entry_id",
            "event_type",
            "previous_entry_sha256",
            "production",
            "research_only",
            "semantic_sha256",
            "status",
            "subject_id",
            "summary",
            "timestamp",
            "version",
        }:
            raise IntelligenceContractError(f"memory entry {index} shape is not closed")
        if row.get("previous_entry_sha256") != previous:
            raise IntelligenceContractError(f"memory entry {index} breaks the hash chain")
        assert_no_authority(row)
        event_type = row.get("event_type")
        status = row.get("status")
        if event_type not in MEMORY_EVENT_TYPES or status not in MEMORY_STATUSES:
            raise IntelligenceContractError("memory event type/status is invalid")
        if event_type == "FAILED_CASE" and status != "FAILED":
            raise IntelligenceContractError("failed case status is invalid")
        summary = row.get("summary")
        if type(summary) is not str or not summary.strip() or len(summary.encode()) > 4000:
            raise IntelligenceContractError("memory summary is invalid")
        expected = seal_content_addressed(
            {
                "artifact_refs": _content_refs(row.get("artifact_refs", [])),
                "authority": dict(NO_AUTHORITY),
                "event_type": event_type,
                "previous_entry_sha256": previous,
                "production": False,
                "research_only": True,
                "status": status,
                "subject_id": identifier(row.get("subject_id"), label="subject_id"),
                "summary": summary.strip(),
                "timestamp": timestamp(row.get("timestamp"), label="timestamp"),
                "version": MEMORY_ENTRY_VERSION,
            },
            identity_field="entry_id",
        )
        if expected != row:
            raise IntelligenceContractError(f"memory entry {index} replay mismatch")
        previous = str(row["semantic_sha256"])
        normalized.append(row)
    if expected_tip is not None and sha256(expected_tip, label="expected_tip") != previous:
        raise IntelligenceContractError("memory tail deletion or substitution detected")
    return tuple(normalized)


def append_memory(
    entries: Sequence[Mapping[str, Any]],
    *,
    event_type: str,
    status: str,
    subject_id: str,
    summary: str,
    artifact_refs: Sequence[Mapping[str, Any]],
    timestamp_value: str,
    expected_tip: str,
) -> tuple[dict[str, Any], ...]:
    """Return a new chain; never mutate, overwrite, or delete the supplied chain."""

    chain = validate_memory_chain(entries, expected_tip=expected_tip)
    if event_type not in MEMORY_EVENT_TYPES or status not in MEMORY_STATUSES:
        raise IntelligenceContractError("memory event type/status is not allowlisted")
    if type(summary) is not str or not summary.strip() or len(summary.encode()) > 4000:
        raise IntelligenceContractError("memory summary is required and bounded")
    if event_type == "FAILED_CASE" and status != "FAILED":
        raise IntelligenceContractError("failed cases must retain FAILED status")
    entry = seal_content_addressed(
        {
            "artifact_refs": _content_refs(artifact_refs),
            "authority": dict(NO_AUTHORITY),
            "event_type": event_type,
            "previous_entry_sha256": expected_tip,
            "production": False,
            "research_only": True,
            "status": status,
            "subject_id": identifier(subject_id, label="subject_id"),
            "summary": summary.strip(),
            "timestamp": timestamp(timestamp_value, label="timestamp"),
            "version": MEMORY_ENTRY_VERSION,
        },
        identity_field="entry_id",
    )
    return chain + (entry,)


__all__ = [
    "MEMORY_ENTRY_VERSION",
    "append_memory",
    "memory_tip",
    "validate_memory_chain",
]
