"""Canonical, authority-closed primitives for Investment Intelligence v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import re
import unicodedata
from typing import Any, Final

DECIMAL_QUANTUM: Final = Decimal("0.000000000001")
MAX_ARTIFACT_BYTES: Final = 8 * 1024 * 1024
FROZEN_V1_MANIFEST_SHA256: Final = (
    "119e31882cbb3a68ffaf99eac2d6404d1c45e4284f46e5c8f54aa22b2cb908fc"
)
DECISION_PROTOCOL: Final = "myquant.v17.v4"
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
SAFE_PATH_RE: Final = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")

NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "factor_governance_write": False,
    "llm": False,
    "mainline_authority": False,
    "order": False,
    "portfolio": False,
    "production": False,
    "provider": False,
    "research_only": True,
    "selector": False,
    "trade": False,
}

EXACT_REF_FIELDS: Final = {
    "artifact_id",
    "artifact_version",
    "available_at",
    "byte_sha256",
    "cutoff",
    "relative_path",
    "semantic_sha256",
}
CONTENT_REF_FIELDS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "semantic_sha256",
}


class IntelligenceV2ContractError(ValueError):
    """Fail-closed error for a v2 research-intelligence contract."""

    exit_code = 2


def require_exact_keys(value: Any, fields: set[str] | frozenset[str], *, label: str) -> dict:
    if type(value) is not dict or set(value) != set(fields):
        raise IntelligenceV2ContractError(f"{label} shape is invalid")
    return value


def timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise IntelligenceV2ContractError(f"{label} must be a UTC second timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntelligenceV2ContractError(f"{label} is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        raise IntelligenceV2ContractError(f"{label} must be canonical UTC seconds")
    return value


def session_date(value: Any, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"\d{8}", value) is None:
        raise IntelligenceV2ContractError(f"{label} must be YYYYMMDD")
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise IntelligenceV2ContractError(f"{label} is invalid") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise IntelligenceV2ContractError(f"{label} is invalid")
    return value


def identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or IDENTIFIER_RE.fullmatch(value) is None:
        raise IntelligenceV2ContractError(f"{label} must be a canonical identifier")
    return value


def code(value: Any, *, label: str) -> str:
    if type(value) is not str or CODE_RE.fullmatch(value) is None:
        raise IntelligenceV2ContractError(f"{label} must be an uppercase code")
    return value


def sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise IntelligenceV2ContractError(f"{label} must be lowercase SHA-256")
    return value


def decimal_value(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if type(value) in {bool, float}:
        raise IntelligenceV2ContractError(f"{label} must not be binary float")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise IntelligenceV2ContractError(f"{label} must be decimal") from exc
    if not parsed.is_finite():
        raise IntelligenceV2ContractError(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise IntelligenceV2ContractError(f"{label} is below its allowed domain")
    if maximum is not None and parsed > maximum:
        raise IntelligenceV2ContractError(f"{label} is above its allowed domain")
    return parsed


def quantized(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        return value.quantize(DECIMAL_QUANTUM)


def decimal_text(value: Decimal) -> str:
    return format(quantized(value), "f")


def _validate_text(value: str, *, label: str) -> None:
    if unicodedata.normalize("NFC", value) != value:
        raise IntelligenceV2ContractError(f"{label} must be Unicode NFC")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise IntelligenceV2ContractError(f"{label} is not UTF-8") from exc
    if len(encoded) > 4000:
        raise IntelligenceV2ContractError(f"{label} exceeds 4000 UTF-8 bytes")


def _validate_list(value: list[Any], *, label: str, depth: int) -> None:
    for index, item in enumerate(value):
        _validate_value(item, label=f"{label}[{index}]", depth=depth + 1)


def _validate_dict(value: dict[str, Any], *, label: str, depth: int) -> None:
    for key, item in value.items():
        if type(key) is not str or not key or not key.isascii():
            raise IntelligenceV2ContractError(f"{label} keys must be nonempty ASCII")
        _validate_value(item, label=f"{label}.{key}", depth=depth + 1)


def _validate_value(value: Any, *, label: str = "$", depth: int = 0) -> None:
    if depth > 64:
        raise IntelligenceV2ContractError(f"{label} exceeds maximum depth")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is str:
        _validate_text(value, label=label)
        return
    if type(value) is list:
        _validate_list(value, label=label, depth=depth)
        return
    if type(value) is dict:
        _validate_dict(value, label=label, depth=depth)
        return
    if type(value) is float:
        raise IntelligenceV2ContractError(f"{label} contains binary float")
    raise IntelligenceV2ContractError(f"{label} contains unsupported value")


def canonical_bytes(value: Any) -> bytes:
    _validate_value(value)
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise IntelligenceV2ContractError("artifact exceeds 8 MiB")
    return raw


def _identity(value: Mapping[str, Any], *, identity_field: str) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def _semantic(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def seal(value: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    if type(value) is not dict or identity_field in value or "semantic_sha256" in value:
        raise IntelligenceV2ContractError("artifact must be an unsealed object")
    result = dict(value)
    result[identity_field] = _identity(result, identity_field=identity_field)
    result["semantic_sha256"] = _semantic(result)
    canonical_bytes(result)
    return result


def validate_seal(value: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise IntelligenceV2ContractError("artifact must be an object")
    result = dict(value)
    if sha256(result.get(identity_field), label=identity_field) != _identity(
        result, identity_field=identity_field
    ):
        raise IntelligenceV2ContractError(f"{identity_field} mismatch")
    if sha256(result.get("semantic_sha256"), label="semantic_sha256") != _semantic(result):
        raise IntelligenceV2ContractError("semantic_sha256 mismatch")
    canonical_bytes(result)
    return result


def _safe_path(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or value.startswith("/")
        or SAFE_PATH_RE.fullmatch(value) is None
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise IntelligenceV2ContractError(f"{label} must be a safe relative path")
    return value


def exact_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    require_exact_keys(value, EXACT_REF_FIELDS, label=label)
    artifact_id = str(value["artifact_id"])
    artifact_version = str(value["artifact_version"])
    if not artifact_id or not artifact_version:
        raise IntelligenceV2ContractError(f"{label} identity fields are required")
    available_at = timestamp(value["available_at"], label=f"{label}.available_at")
    cutoff = timestamp(value["cutoff"], label=f"{label}.cutoff")
    if cutoff > available_at:
        raise IntelligenceV2ContractError(f"{label} cutoff exceeds availability")
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "available_at": available_at,
        "byte_sha256": sha256(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "cutoff": cutoff,
        "relative_path": _safe_path(value["relative_path"], label=f"{label}.relative_path"),
        "semantic_sha256": sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256"),
    }


def content_ref(value: Mapping[str, Any], *, identity_field: str) -> dict[str, str]:
    document = validate_seal(value, identity_field=identity_field)
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(canonical_bytes(document)).hexdigest(),
        "semantic_sha256": str(document["semantic_sha256"]),
    }


def validate_content_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    require_exact_keys(value, CONTENT_REF_FIELDS, label=label)
    artifact_id = str(value["artifact_id"])
    artifact_version = str(value["artifact_version"])
    if not artifact_id or not artifact_version:
        raise IntelligenceV2ContractError(f"{label} identity fields are required")
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": sha256(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "semantic_sha256": sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256"),
    }


def sorted_unique(
    values: Sequence[Any],
    *,
    label: str,
    maximum: int = 256,
    allow_empty: bool = False,
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceV2ContractError(f"{label} must be a sequence")
    rows = [identifier(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if (not allow_empty and not rows) or len(rows) > maximum or len(rows) != len(set(rows)):
        raise IntelligenceV2ContractError(f"{label} cardinality or uniqueness is invalid")
    return sorted(rows, key=lambda item: item.encode("ascii"))


def require_no_future(*, available_at: str, as_of: str, label: str) -> None:
    if timestamp(available_at, label=f"{label}.available_at") > timestamp(as_of, label="as_of"):
        raise IntelligenceV2ContractError(f"{label} contains future evidence")


def common_fields(*, timestamp_value: str) -> dict[str, Any]:
    return {
        "authority": dict(NO_AUTHORITY),
        "decision_protocol": DECISION_PROTOCOL,
        "frozen_v1_manifest_sha256": FROZEN_V1_MANIFEST_SHA256,
        "production": False,
        "research_only": True,
        "timestamp": timestamp(timestamp_value, label="timestamp"),
    }


__all__ = [
    "CONTENT_REF_FIELDS",
    "DECIMAL_QUANTUM",
    "DECISION_PROTOCOL",
    "EXACT_REF_FIELDS",
    "FROZEN_V1_MANIFEST_SHA256",
    "IntelligenceV2ContractError",
    "NO_AUTHORITY",
    "canonical_bytes",
    "code",
    "common_fields",
    "content_ref",
    "decimal_text",
    "decimal_value",
    "exact_ref",
    "identifier",
    "quantized",
    "require_exact_keys",
    "require_no_future",
    "seal",
    "session_date",
    "sha256",
    "sorted_unique",
    "timestamp",
    "validate_content_ref",
    "validate_seal",
]
