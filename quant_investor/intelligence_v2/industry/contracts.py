"""Shared closed-artifact helpers for I2 Industry Intelligence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from .._core import (
    NO_AUTHORITY,
    common_fields,
    decimal_text,
    decimal_value,
    exact_ref,
    identifier,
    require_exact_keys,
    require_no_future,
    seal,
    timestamp,
    validate_seal,
)
from .models import IndustryContractError

COMMON_FIELDS = frozenset(
    {
        "authority",
        "decision_protocol",
        "frozen_v1_manifest_sha256",
        "production",
        "research_only",
        "timestamp",
        "version",
    }
)


def fail(message: str) -> None:
    raise IndustryContractError(message)


def artifact(
    *,
    version: str,
    identity_field: str,
    timestamp_value: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) & set(COMMON_FIELDS):
        fail("industry artifact payload shape is invalid")
    try:
        return seal(
            {
                **common_fields(timestamp_value=timestamp_value),
                **dict(payload),
                "version": version,
            },
            identity_field=identity_field,
        )
    except IndustryContractError:
        raise
    except Exception as exc:
        raise IndustryContractError(str(exc)) from exc


def closed_artifact(
    value: Mapping[str, Any],
    *,
    version: str,
    identity_field: str,
    payload_fields: set[str] | frozenset[str],
) -> dict[str, Any]:
    expected = (
        set(COMMON_FIELDS)
        | set(payload_fields)
        | {
            identity_field,
            "semantic_sha256",
        }
    )
    try:
        row = require_exact_keys(value, expected, label=version)
        validated = validate_seal(row, identity_field=identity_field)
        if validated["version"] != version:
            fail("industry artifact version mismatch")
        if (
            validated["authority"] != NO_AUTHORITY
            or validated["research_only"] is not True
            or validated["production"] is not False
        ):
            fail("industry authority boundary is open")
        timestamp(validated["timestamp"], label="artifact.timestamp")
        return validated
    except IndustryContractError:
        raise
    except Exception as exc:
        raise IndustryContractError(str(exc)) from exc


def entity(value: Any, *, label: str) -> str:
    try:
        return identifier(value, label=label)
    except Exception as exc:
        raise IndustryContractError(str(exc)) from exc


def decimal(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> str:
    try:
        return decimal_text(decimal_value(value, label=label, minimum=minimum, maximum=maximum))
    except Exception as exc:
        raise IndustryContractError(str(exc)) from exc


def source_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    try:
        return exact_ref(value, label=label)
    except Exception as exc:
        raise IndustryContractError(str(exc)) from exc


def no_future(*, available_at: str, as_of: str, label: str) -> None:
    try:
        require_no_future(available_at=available_at, as_of=as_of, label=label)
    except Exception as exc:
        raise IndustryContractError(str(exc)) from exc


def exact_sequence(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        fail(f"{label} must be a sequence")
    return list(value)


__all__ = [
    "COMMON_FIELDS",
    "artifact",
    "closed_artifact",
    "decimal",
    "entity",
    "exact_sequence",
    "fail",
    "no_future",
    "source_ref",
]
