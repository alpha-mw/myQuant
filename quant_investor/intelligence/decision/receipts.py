"""Common content-addressed receipt primitives for Sprint I1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    seal_content_addressed,
    validate_content_addressed,
)
from .models import (
    DECISION_PROTOCOL,
    DecisionContractError,
    canonical_timestamp,
    ensure_artifact_size,
    fail,
)

COMMON_FIELDS: Final = frozenset(
    {
        "authority",
        "broker",
        "decision_protocol",
        "execution",
        "mainline_authority",
        "operational_activation_unchanged",
        "order",
        "production",
        "research_only",
        "timestamp",
        "trade",
        "version",
    }
)


def authority_fields(timestamp_value: Any) -> dict[str, Any]:
    """Return the exact closed, research-only I1 authority fields."""

    return {
        "authority": dict(NO_AUTHORITY),
        "broker": False,
        "decision_protocol": DECISION_PROTOCOL,
        "execution": False,
        "mainline_authority": False,
        "operational_activation_unchanged": True,
        "order": False,
        "production": False,
        "research_only": True,
        "timestamp": canonical_timestamp(timestamp_value, label="timestamp"),
        "trade": False,
    }


def seal_artifact(
    *,
    version: str,
    identity_field: str,
    timestamp_value: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one new canonical I1 artifact and enforce the per-artifact size cap."""

    if type(version) is not str or not version:
        fail("I1_SHAPE_INVALID", "artifact version is required")
    if type(identity_field) is not str or not identity_field:
        fail("I1_SHAPE_INVALID", "artifact identity field is required")
    if type(payload) is not dict:
        fail("I1_SHAPE_INVALID", "artifact payload must be an object")
    if identity_field in payload or "semantic_sha256" in payload:
        fail("I1_SHAPE_INVALID", "artifact payload must be unsealed")
    if set(payload) & COMMON_FIELDS:
        fail("I1_SHAPE_INVALID", "artifact payload overlaps common fields")
    try:
        result = seal_content_addressed(
            {
                **authority_fields(timestamp_value),
                **dict(payload),
                "version": version,
            },
            identity_field=identity_field,
        )
    except DecisionContractError:
        raise
    except IntelligenceContractError as exc:
        fail("I1_SHAPE_INVALID", str(exc))
    ensure_artifact_size(result)
    return result


def validate_closed_artifact(
    document: Mapping[str, Any],
    *,
    version: str,
    identity_field: str,
    payload_fields: set[str],
) -> dict[str, Any]:
    """Validate seal, exact shape, protocol state, and closed authority."""

    if type(document) is not dict:
        fail("I1_SHAPE_INVALID", "artifact must be an object")
    expected_fields = (
        set(COMMON_FIELDS)
        | set(payload_fields)
        | {
            identity_field,
            "semantic_sha256",
        }
    )
    if set(document) != expected_fields or document.get("version") != version:
        fail("I1_SHAPE_INVALID", f"{version} shape/version mismatch")
    try:
        row = validate_content_addressed(document, identity_field=identity_field)
    except IntelligenceContractError as exc:
        fail("I1_REPLAY_MISMATCH", str(exc))
    ensure_artifact_size(row)
    if row.get("authority") != NO_AUTHORITY or any(
        row.get(field) is not False for field in ("broker", "execution", "order", "trade")
    ):
        fail("I1_AUTHORITY_OPEN", "artifact authority boundary is open")
    if (
        row.get("research_only") is not True
        or row.get("production") is not False
        or row.get("decision_protocol") != DECISION_PROTOCOL
        or row.get("mainline_authority") is not False
        or row.get("operational_activation_unchanged") is not True
    ):
        fail("I1_AUTHORITY_OPEN", "artifact protocol state is open")
    canonical_timestamp(row.get("timestamp"), label="artifact.timestamp")
    return row


__all__ = [
    "COMMON_FIELDS",
    "authority_fields",
    "seal_artifact",
    "validate_closed_artifact",
]
