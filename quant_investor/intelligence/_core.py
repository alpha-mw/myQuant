"""Deterministic, no-authority primitives for V17 research intelligence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import re
from typing import Any, Final

from quant_investor.v17_v4_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    validate_semantic_sha,
)

PROTOCOL_VERSION: Final = "myquant.v17.research-intelligence.i0"
DECIMAL_QUANTUM: Final = Decimal("0.000000000001")
ZERO_SHA256: Final = "0" * 64
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
SAFE_PATH_RE: Final = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")

NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "factor_governance_write": False,
    "formal_activation": False,
    "llm": False,
    "order": False,
    "portfolio": False,
    "provider": False,
    "research_runtime_default": False,
    "selector": False,
    "trade": False,
}


class IntelligenceContractError(ValueError):
    """Fail-closed I0 contract error."""

    exit_code = 2


def timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise IntelligenceContractError(f"{label} must be a UTC second timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntelligenceContractError(f"{label} is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        raise IntelligenceContractError(f"{label} must be canonical UTC seconds")
    return value


def identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or IDENTIFIER_RE.fullmatch(value) is None:
        raise IntelligenceContractError(f"{label} is not a canonical identifier")
    return value


def sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise IntelligenceContractError(f"{label} is not a canonical SHA-256")
    return value


def safe_path(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or value.startswith("/")
        or SAFE_PATH_RE.fullmatch(value) is None
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise IntelligenceContractError(f"{label} is not a safe relative path")
    return value


def decimal_value(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> Decimal:
    if type(value) is bool:
        raise IntelligenceContractError(f"{label} must be decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise IntelligenceContractError(f"{label} must be decimal") from exc
    if not parsed.is_finite():
        raise IntelligenceContractError(f"{label} must be finite")
    if minimum is not None and (parsed < minimum or (minimum_exclusive and parsed == minimum)):
        raise IntelligenceContractError(f"{label} is below its allowed domain")
    if maximum is not None and (parsed > maximum or (maximum_exclusive and parsed == maximum)):
        raise IntelligenceContractError(f"{label} is above its allowed domain")
    return parsed


def quantized(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        return value.quantize(DECIMAL_QUANTUM)


def decimal_text(value: Decimal) -> str:
    return format(quantized(value), "f")


def require_no_future(*, available_at: str, as_of: str, label: str) -> None:
    available = timestamp(available_at, label=f"{label}.available_at")
    cutoff = timestamp(as_of, label="as_of")
    if available > cutoff:
        raise IntelligenceContractError(f"{label} contains future evidence")


def content_identity(document: Mapping[str, Any], *, identity_field: str) -> str:
    body = dict(document)
    body.pop(identity_field, None)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def seal_content_addressed(
    document: Mapping[str, Any],
    *,
    identity_field: str,
) -> dict[str, Any]:
    if identity_field in document or "semantic_sha256" in document:
        raise IntelligenceContractError("document must be unsealed")
    result = dict(document)
    result[identity_field] = content_identity(result, identity_field=identity_field)
    return seal_semantic(result)


def validate_content_addressed(
    document: Mapping[str, Any],
    *,
    identity_field: str,
) -> dict[str, Any]:
    try:
        normalized = validate_semantic_sha(document)
    except Exception as exc:
        raise IntelligenceContractError("semantic SHA mismatch") from exc
    expected = content_identity(normalized, identity_field=identity_field)
    if normalized.get(identity_field) != expected:
        raise IntelligenceContractError(f"{identity_field} is not content addressed")
    return normalized


EXACT_REF_FIELDS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "cutoff",
    "relative_path",
    "semantic_sha256",
    "strategy_id",
}


def exact_ref(
    value: Mapping[str, Any],
    *,
    label: str,
    expected_versions: Sequence[str] | None = None,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != EXACT_REF_FIELDS:
        raise IntelligenceContractError(f"{label} must be an exact artifact reference")
    for field in ("artifact_id", "artifact_version"):
        if type(value[field]) is not str or not value[field]:
            raise IntelligenceContractError(f"{label}.{field} must be a string")
    result = {
        "artifact_id": value["artifact_id"],
        "artifact_version": value["artifact_version"],
        "byte_sha256": sha256(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "cutoff": timestamp(value["cutoff"], label=f"{label}.cutoff"),
        "relative_path": safe_path(value["relative_path"], label=f"{label}.relative_path"),
        "semantic_sha256": sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256"),
        "strategy_id": identifier(value["strategy_id"], label=f"{label}.strategy_id"),
    }
    if expected_versions is not None and result["artifact_version"] not in set(expected_versions):
        raise IntelligenceContractError(f"{label} version is not allowlisted")
    return result


def sorted_exact_refs(
    values: Sequence[Mapping[str, Any]],
    *,
    label: str,
    expected_versions: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceContractError(f"{label} must be a sequence")
    rows = [
        exact_ref(value, label=f"{label}[{index}]", expected_versions=expected_versions)
        for index, value in enumerate(values)
    ]
    keys = [(row["relative_path"], row["byte_sha256"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise IntelligenceContractError(f"{label} contains duplicate references")
    return sorted(
        rows,
        key=lambda row: (
            row["relative_path"].encode("ascii"),
            row["byte_sha256"].encode("ascii"),
        ),
    )


def content_ref(document: Mapping[str, Any], *, identity_field: str) -> dict[str, str]:
    normalized = validate_content_addressed(document, identity_field=identity_field)
    return {
        "artifact_id": str(normalized[identity_field]),
        "artifact_version": str(normalized["version"]),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(normalized)).hexdigest(),
        "semantic_sha256": str(normalized["semantic_sha256"]),
    }


def assert_no_authority(document: Mapping[str, Any]) -> None:
    if document.get("authority") != NO_AUTHORITY:
        raise IntelligenceContractError("authority boundary is not closed")
    if document.get("research_only") is not True or document.get("production") is not False:
        raise IntelligenceContractError("research/production boundary is not closed")


__all__ = [
    "DECIMAL_QUANTUM",
    "EXACT_REF_FIELDS",
    "IntelligenceContractError",
    "NO_AUTHORITY",
    "PROTOCOL_VERSION",
    "ZERO_SHA256",
    "assert_no_authority",
    "content_ref",
    "decimal_text",
    "decimal_value",
    "exact_ref",
    "identifier",
    "quantized",
    "require_no_future",
    "safe_path",
    "seal_content_addressed",
    "sha256",
    "sorted_exact_refs",
    "timestamp",
    "validate_content_addressed",
]
