"""Canonical, research-only primitives for Factor Governance v5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import re
import unicodedata
from typing import Any, Final

PROTOCOL_VERSION: Final = "factor-governance-protocol.v5"
DECIMAL_QUANTUM: Final = Decimal("0.000000000001")
MAX_ARTIFACT_BYTES: Final = 8 * 1024 * 1024
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "factor_governance_write": False,
    "mainline_authority": False,
    "order": False,
    "portfolio": False,
    "production": False,
    "provider": False,
    "research_only": True,
    "selector": False,
    "trade": False,
}


class FactorGovernanceV5Error(ValueError):
    """Fail-closed Factor Governance v5 contract error."""

    exit_code = 2


def timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise FactorGovernanceV5Error(f"{label} must be a UTC second timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FactorGovernanceV5Error(f"{label} is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        raise FactorGovernanceV5Error(f"{label} must be canonical UTC seconds")
    return value


def identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or IDENTIFIER_RE.fullmatch(value) is None:
        raise FactorGovernanceV5Error(f"{label} must be a canonical identifier")
    return value


def code(value: Any, *, label: str) -> str:
    if type(value) is not str or CODE_RE.fullmatch(value) is None:
        raise FactorGovernanceV5Error(f"{label} must be a canonical uppercase code")
    return value


def sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise FactorGovernanceV5Error(f"{label} must be lowercase SHA-256")
    return value


def decimal_value(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if type(value) is bool or type(value) is float:
        raise FactorGovernanceV5Error(f"{label} must not be binary float")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise FactorGovernanceV5Error(f"{label} must be decimal") from exc
    if not parsed.is_finite():
        raise FactorGovernanceV5Error(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise FactorGovernanceV5Error(f"{label} is below its allowed domain")
    if maximum is not None and parsed > maximum:
        raise FactorGovernanceV5Error(f"{label} is above its allowed domain")
    return parsed


def quantized(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        return value.quantize(DECIMAL_QUANTUM)


def decimal_text(value: Decimal) -> str:
    return format(quantized(value), "f")


def _validate_string(value: str, *, label: str) -> None:
    try:
        raw = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise FactorGovernanceV5Error(f"{label} is not UTF-8") from exc
    if len(raw) > 4000:
        raise FactorGovernanceV5Error(f"{label} exceeds 4000 UTF-8 bytes")
    if unicodedata.normalize("NFC", value) != value:
        raise FactorGovernanceV5Error(f"{label} must be Unicode NFC")


def _validate_list(value: list[Any], *, label: str, depth: int) -> None:
    for index, item in enumerate(value):
        _validate_value(item, label=f"{label}[{index}]", depth=depth + 1)


def _validate_dict(value: dict[str, Any], *, label: str, depth: int) -> None:
    for key, item in value.items():
        if type(key) is not str or not key or not key.isascii():
            raise FactorGovernanceV5Error(f"{label} keys must be nonempty ASCII")
        _validate_value(item, label=f"{label}.{key}", depth=depth + 1)


def _validate_value(value: Any, *, label: str = "$", depth: int = 0) -> None:
    if depth > 64:
        raise FactorGovernanceV5Error(f"{label} exceeds maximum depth")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is str:
        _validate_string(value, label=label)
        return
    if type(value) is list:
        _validate_list(value, label=label, depth=depth)
        return
    if type(value) is dict:
        _validate_dict(value, label=label, depth=depth)
        return
    if type(value) is float:
        raise FactorGovernanceV5Error(f"{label} contains binary float")
    raise FactorGovernanceV5Error(f"{label} contains unsupported value")


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
        raise FactorGovernanceV5Error("artifact exceeds 8 MiB")
    return raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FactorGovernanceV5Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes, *, label: str = "artifact") -> Any:
    if type(raw) is not bytes or len(raw) > MAX_ARTIFACT_BYTES:
        raise FactorGovernanceV5Error(f"{label} exceeds its byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FactorGovernanceV5Error(f"non-finite JSON constant: {token}")
            ),
            parse_float=lambda _token: (_ for _ in ()).throw(
                FactorGovernanceV5Error("native JSON floats are forbidden")
            ),
        )
    except FactorGovernanceV5Error:
        raise
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise FactorGovernanceV5Error(f"{label} is invalid JSON") from exc
    _validate_value(value)
    expected = canonical_bytes(value)
    if raw not in {expected, expected + b"\n"}:
        raise FactorGovernanceV5Error(f"{label} is not canonical JSON")
    return value


def semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def content_identity(value: Mapping[str, Any], *, identity_field: str) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def seal(value: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    if identity_field in value or "semantic_sha256" in value:
        raise FactorGovernanceV5Error("artifact must be unsealed")
    result = dict(value)
    result[identity_field] = content_identity(result, identity_field=identity_field)
    result["semantic_sha256"] = semantic_sha256(result)
    canonical_bytes(result)
    return result


def validate_seal(value: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise FactorGovernanceV5Error("artifact must be an object")
    result = dict(value)
    if sha256(result.get("semantic_sha256"), label="semantic_sha256") != semantic_sha256(result):
        raise FactorGovernanceV5Error("semantic_sha256 mismatch")
    if sha256(result.get(identity_field), label=identity_field) != content_identity(
        result, identity_field=identity_field
    ):
        raise FactorGovernanceV5Error(f"{identity_field} is not content addressed")
    canonical_bytes(result)
    return result


def sorted_unique_strings(values: Sequence[Any], *, label: str, maximum: int = 256) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FactorGovernanceV5Error(f"{label} must be a sequence")
    rows = [identifier(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if not rows or len(rows) > maximum or len(rows) != len(set(rows)):
        raise FactorGovernanceV5Error(f"{label} has invalid cardinality or duplicates")
    return sorted(rows, key=lambda value: value.encode("ascii"))


def common_fields(*, timestamp_value: str) -> dict[str, Any]:
    return {
        "authority": dict(NO_AUTHORITY),
        "decision_protocol": "myquant.v17.v4",
        "factor_protocol": PROTOCOL_VERSION,
        "timestamp": timestamp(timestamp_value, label="timestamp"),
    }


__all__ = [
    "DECIMAL_QUANTUM",
    "FactorGovernanceV5Error",
    "NO_AUTHORITY",
    "PROTOCOL_VERSION",
    "canonical_bytes",
    "code",
    "common_fields",
    "decimal_text",
    "decimal_value",
    "identifier",
    "quantized",
    "seal",
    "sha256",
    "sorted_unique_strings",
    "strict_json_loads",
    "timestamp",
    "validate_seal",
]
