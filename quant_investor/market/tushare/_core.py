"""Canonical primitives for stable Tushare market-data contracts."""

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
FORBIDDEN_VERSION_FIELDS: Final = frozenset(
    {
        "artifact_version",
        "decision_protocol",
        "frozen_v1_manifest_sha256",
        "protocol_version",
        "schema_version",
        "version",
    }
)
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
KIND_RE: Final = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")
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
    "available_at",
    "byte_sha256",
    "contract_sha256",
    "cutoff",
    "kind",
    "relative_path",
    "semantic_sha256",
}
CONTENT_REF_FIELDS: Final = {
    "artifact_id",
    "byte_sha256",
    "contract_sha256",
    "kind",
    "semantic_sha256",
}


class TushareDataContractError(ValueError):
    """Fail-closed error for stable market-data contracts."""

    exit_code = 2
    code = "TUSHARE_DATA_CONTRACT_INVALID"
    public_fields: dict[str, Any] = {}


def require_exact_keys(value: Any, fields: set[str] | frozenset[str], *, label: str) -> dict:
    if type(value) is not dict or set(value) != set(fields):
        raise TushareDataContractError(f"{label} shape is invalid")
    return value


def timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise TushareDataContractError(f"{label} must be a UTC second timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TushareDataContractError(f"{label} is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        raise TushareDataContractError(f"{label} must be canonical UTC seconds")
    return value


def session_date(value: Any, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"\d{8}", value) is None:
        raise TushareDataContractError(f"{label} must be YYYYMMDD")
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise TushareDataContractError(f"{label} is invalid") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise TushareDataContractError(f"{label} is invalid")
    return value


def identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or IDENTIFIER_RE.fullmatch(value) is None:
        raise TushareDataContractError(f"{label} must be a canonical identifier")
    return value


def code(value: Any, *, label: str) -> str:
    if type(value) is not str or CODE_RE.fullmatch(value) is None:
        raise TushareDataContractError(f"{label} must be an uppercase code")
    return value


def sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise TushareDataContractError(f"{label} must be lowercase SHA-256")
    return value


def decimal_value(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if type(value) in {bool, float}:
        raise TushareDataContractError(f"{label} must not be binary float")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TushareDataContractError(f"{label} must be decimal") from exc
    if not parsed.is_finite():
        raise TushareDataContractError(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise TushareDataContractError(f"{label} is below its allowed domain")
    if maximum is not None and parsed > maximum:
        raise TushareDataContractError(f"{label} is above its allowed domain")
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
        raise TushareDataContractError(f"{label} must be Unicode NFC")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TushareDataContractError(f"{label} is not UTF-8") from exc
    if len(encoded) > 4000:
        raise TushareDataContractError(f"{label} exceeds 4000 UTF-8 bytes")


def _validate_list(value: list[Any], *, label: str, depth: int) -> None:
    for index, item in enumerate(value):
        _validate_value(item, label=f"{label}[{index}]", depth=depth + 1)


def _validate_dict(value: dict[str, Any], *, label: str, depth: int) -> None:
    for key, item in value.items():
        if type(key) is not str or not key or not key.isascii() or key in FORBIDDEN_VERSION_FIELDS:
            raise TushareDataContractError(f"{label} keys must be nonempty ASCII")
        _validate_value(item, label=f"{label}.{key}", depth=depth + 1)


def _validate_value(value: Any, *, label: str = "$", depth: int = 0) -> None:
    if depth > 64:
        raise TushareDataContractError(f"{label} exceeds maximum depth")
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
        raise TushareDataContractError(f"{label} contains binary float")
    raise TushareDataContractError(f"{label} contains unsupported value")


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
        raise TushareDataContractError("artifact exceeds 8 MiB")
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


def contract_sha256(kind: Any) -> str:
    if type(kind) is not str or KIND_RE.fullmatch(kind) is None:
        raise TushareDataContractError("artifact kind is invalid")
    definition = {
        "contract_model": "exact-replay",
        "kind": kind,
        "unknown_fields": "REJECT",
    }
    return hashlib.sha256(canonical_bytes(definition)).hexdigest()


def seal(value: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    if (
        type(value) is not dict
        or identity_field in value
        or "contract_sha256" in value
        or "semantic_sha256" in value
    ):
        raise TushareDataContractError("artifact must be an unsealed object")
    result = dict(value)
    result["contract_sha256"] = contract_sha256(result.get("kind"))
    result[identity_field] = _identity(result, identity_field=identity_field)
    result["semantic_sha256"] = _semantic(result)
    canonical_bytes(result)
    return result


def validate_seal(value: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TushareDataContractError("artifact must be an object")
    result = dict(value)
    if sha256(
        result.get("contract_sha256"),
        label="contract_sha256",
    ) != contract_sha256(result.get("kind")):
        raise TushareDataContractError("contract_sha256 mismatch")
    if sha256(result.get(identity_field), label=identity_field) != _identity(
        result, identity_field=identity_field
    ):
        raise TushareDataContractError(f"{identity_field} mismatch")
    if sha256(result.get("semantic_sha256"), label="semantic_sha256") != _semantic(result):
        raise TushareDataContractError("semantic_sha256 mismatch")
    canonical_bytes(result)
    return result


def _safe_path(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or value.startswith("/")
        or SAFE_PATH_RE.fullmatch(value) is None
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise TushareDataContractError(f"{label} must be a safe relative path")
    return value


def exact_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    require_exact_keys(value, EXACT_REF_FIELDS, label=label)
    artifact_id = str(value["artifact_id"])
    kind = str(value["kind"])
    if not artifact_id or KIND_RE.fullmatch(kind) is None:
        raise TushareDataContractError(f"{label} identity fields are required")
    available_at = timestamp(value["available_at"], label=f"{label}.available_at")
    cutoff = timestamp(value["cutoff"], label=f"{label}.cutoff")
    if cutoff > available_at:
        raise TushareDataContractError(f"{label} cutoff exceeds availability")
    normalized_contract_sha256 = sha256(
        value["contract_sha256"],
        label=f"{label}.contract_sha256",
    )
    if normalized_contract_sha256 != contract_sha256(kind):
        raise TushareDataContractError(f"{label}.contract_sha256 mismatch")
    return {
        "artifact_id": artifact_id,
        "available_at": available_at,
        "byte_sha256": sha256(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "contract_sha256": normalized_contract_sha256,
        "cutoff": cutoff,
        "kind": kind,
        "relative_path": _safe_path(value["relative_path"], label=f"{label}.relative_path"),
        "semantic_sha256": sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256"),
    }


def content_ref(value: Mapping[str, Any], *, identity_field: str) -> dict[str, str]:
    document = validate_seal(value, identity_field=identity_field)
    return {
        "artifact_id": str(document[identity_field]),
        "byte_sha256": hashlib.sha256(canonical_bytes(document)).hexdigest(),
        "contract_sha256": str(document["contract_sha256"]),
        "kind": str(document["kind"]),
        "semantic_sha256": str(document["semantic_sha256"]),
    }


def validate_content_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    require_exact_keys(value, CONTENT_REF_FIELDS, label=label)
    artifact_id = str(value["artifact_id"])
    kind = str(value["kind"])
    if not artifact_id or KIND_RE.fullmatch(kind) is None:
        raise TushareDataContractError(f"{label} identity fields are required")
    normalized_contract_sha256 = sha256(
        value["contract_sha256"],
        label=f"{label}.contract_sha256",
    )
    if normalized_contract_sha256 != contract_sha256(kind):
        raise TushareDataContractError(f"{label}.contract_sha256 mismatch")
    return {
        "artifact_id": artifact_id,
        "byte_sha256": sha256(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "contract_sha256": normalized_contract_sha256,
        "kind": kind,
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
        raise TushareDataContractError(f"{label} must be a sequence")
    rows = [identifier(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if (not allow_empty and not rows) or len(rows) > maximum or len(rows) != len(set(rows)):
        raise TushareDataContractError(f"{label} cardinality or uniqueness is invalid")
    return sorted(rows, key=lambda item: item.encode("ascii"))


def require_no_future(*, available_at: str, as_of: str, label: str) -> None:
    if timestamp(available_at, label=f"{label}.available_at") > timestamp(as_of, label="as_of"):
        raise TushareDataContractError(f"{label} contains future evidence")


def common_fields(*, timestamp_value: str) -> dict[str, Any]:
    return {
        "authority": dict(NO_AUTHORITY),
        "production": False,
        "research_only": True,
        "timestamp": timestamp(timestamp_value, label="timestamp"),
    }


__all__ = [
    "CONTENT_REF_FIELDS",
    "DECIMAL_QUANTUM",
    "EXACT_REF_FIELDS",
    "FORBIDDEN_VERSION_FIELDS",
    "TushareDataContractError",
    "NO_AUTHORITY",
    "canonical_bytes",
    "code",
    "common_fields",
    "contract_sha256",
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
