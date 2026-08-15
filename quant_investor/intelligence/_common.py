"""Shared deterministic primitives for the stable Intelligence package.

The active Intelligence surface uses the repository-wide semantic artifact
envelope.  Domain payloads deliberately carry no numeric protocol or schema
labels; contract identity lives in the envelope's ``contract_sha256``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import ipaddress
import re
import unicodedata
from types import MappingProxyType
from typing import Any, Final, NoReturn
from urllib.parse import urlsplit

from quant_investor.contracts import (
    artifact_byte_sha256,
    canonical_json_bytes,
    seal_artifact,
    validate_artifact,
)

DECIMAL_QUANTUM: Final = Decimal("0.000000000001")
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
COMPANY_CODE_RE: Final = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
FORBIDDEN_VERSION_FIELDS: Final = frozenset({"version", "schema_version", "protocol_version"})
CONTROL_AUTHORITY_FIELDS: Final = frozenset(
    {
        "broker",
        "execution",
        "factor_governance_write",
        "llm_control",
        "mainline_activation",
        "order",
        "portfolio_activation",
        "provider",
        "selector_write",
        "trade",
    }
)
ACTIVATION_BINDING_FIELDS: Final = frozenset(
    {
        "activation_receipt",
        "activation_receipt_ref",
        "active_generation_id",
        "active_pointer",
        "generation_id",
        "pointer_ref",
        "readiness_ref",
    }
)
ARTIFACT_REF_FIELDS: Final = frozenset(
    {"kind", "contract_sha256", "artifact_id", "semantic_sha256", "byte_sha256"}
)
NO_AUTHORITY: Final = MappingProxyType(
    {
        "broker": False,
        "execution": False,
        "factor_governance_write": False,
        "llm_control": False,
        "mainline_activation": False,
        "order": False,
        "portfolio_activation": False,
        "provider": False,
        "selector_write": False,
        "trade": False,
    }
)


class IntelligenceError(ValueError):
    """Fail-closed stable Intelligence contract error."""

    exit_code = 2
    default_code = "INTELLIGENCE_VALIDATION_FAILED"

    def __init__(self, detail: str) -> None:
        self.code = self.default_code
        self.public_fields: dict[str, Any] = {}
        super().__init__(detail)


def _fail(message: str) -> NoReturn:
    raise IntelligenceError(message)


def timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        _fail(f"{label} must be a canonical UTC-second timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntelligenceError(f"{label} is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        _fail(f"{label} must be a canonical UTC-second timestamp")
    return value


def identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or IDENTIFIER_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical identifier")
    return value


def company_code(value: Any, *, label: str = "company_code") -> str:
    if type(value) is not str or COMPANY_CODE_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical A-share company code")
    return value


def sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def artifact_identity(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf-8", errors="strict")) > 512
        or any(ord(character) < 0x20 for character in value)
    ):
        _fail(f"{label} must be a canonical artifact identity")
    return value


def decimal_value(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if type(value) in {bool, float}:
        _fail(f"{label} must be a decimal value, not a binary float")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise IntelligenceError(f"{label} must be a decimal value") from exc
    if not parsed.is_finite():
        _fail(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        _fail(f"{label} is below its allowed domain")
    if maximum is not None and parsed > maximum:
        _fail(f"{label} is above its allowed domain")
    return parsed


def decimal_text(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        return format(value.quantize(DECIMAL_QUANTUM), "f")


def require_exact_keys(value: Any, keys: set[str] | frozenset[str], *, label: str) -> dict:
    if type(value) is not dict or set(value) != set(keys):
        _fail(f"{label} shape is invalid")
    return value


def _validate_json_value(  # noqa: C901 - closed recursive JSON grammar
    value: Any, *, label: str = "$", depth: int = 0
) -> None:
    if depth > 64:
        _fail(f"{label} exceeds maximum nesting depth")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is str:
        if unicodedata.normalize("NFC", value) != value:
            _fail(f"{label} must be Unicode NFC")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise IntelligenceError(f"{label} is not strict UTF-8") from exc
        return
    if type(value) is float:
        _fail(f"{label} contains a binary float")
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, label=f"{label}[{index}]", depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or not key or not key.isascii():
                _fail(f"{label} contains a non-canonical key")
            if key in FORBIDDEN_VERSION_FIELDS:
                _fail(f"{label}.{key} is forbidden in stable artifacts")
            _validate_json_value(item, label=f"{label}.{key}", depth=depth + 1)
        return
    _fail(f"{label} contains an unsupported value")


def canonical_value(value: Any) -> Any:
    """Validate a JSON value and return it unchanged for explicit call sites."""

    _validate_json_value(value)
    canonical_json_bytes(value)
    return value


def require_no_control_authority(value: Any, *, label: str = "payload") -> None:
    """Reject nested claims of operational authority in caller-owned content."""

    if type(value) is list:
        for index, item in enumerate(value):
            require_no_control_authority(item, label=f"{label}[{index}]")
        return
    if type(value) is not dict:
        return
    for key, item in value.items():
        if key in CONTROL_AUTHORITY_FIELDS and item is not False and item is not None:
            _fail(f"{label}.{key} claims forbidden control authority")
        require_no_control_authority(item, label=f"{label}.{key}")


def require_no_activation_binding(value: Any, *, label: str = "payload") -> None:
    """Reject generation, pointer, readiness, and activation receipt dependencies."""

    if type(value) is list:
        for index, item in enumerate(value):
            require_no_activation_binding(item, label=f"{label}[{index}]")
        return
    if type(value) is not dict:
        return
    for key, item in value.items():
        if key in ACTIVATION_BINDING_FIELDS:
            _fail(f"{label}.{key} contains a forbidden activation binding")
        require_no_activation_binding(item, label=f"{label}.{key}")


def sorted_unique_identifiers(
    values: Sequence[Any],
    *,
    label: str,
    allow_empty: bool = False,
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail(f"{label} must be a sequence")
    rows = [identifier(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if (not allow_empty and not rows) or len(rows) != len(set(rows)):
        _fail(f"{label} must be unique and nonempty")
    return sorted(rows, key=lambda item: item.encode("ascii"))


def business_identity(*, kind: str, identity_inputs: Mapping[str, Any]) -> str:
    """Derive a business identity from explicitly declared identity inputs only.

    The result is intentionally independent from the artifact envelope and from
    non-identity payload fields.  Builders may instead accept a caller-owned
    identity and pass it directly to :func:`build_artifact`.
    """

    artifact_kind = identifier(kind, label="kind")
    inputs = dict(identity_inputs)
    canonical_value(inputs)
    return hashlib.sha256(
        canonical_json_bytes({"identity_inputs": inputs, "kind": artifact_kind})
    ).hexdigest()


def research_payload(fields: Mapping[str, Any]) -> dict[str, Any]:
    if set(fields) & {"authority", "production", "research_only", "run_state"}:
        _fail("research payload fields cannot override inactivity controls")
    result = {
        "authority": dict(NO_AUTHORITY),
        "production": False,
        "research_only": True,
        "run_state": "INACTIVE",
        **dict(fields),
    }
    canonical_value(result)
    return result


def build_artifact(
    *,
    kind: str,
    identity_field: str,
    identity: str,
    fields: Mapping[str, Any],
    created_at: str,
    research_only: bool = True,
) -> dict[str, Any]:
    """Build one deterministic stable envelope with a domain-owned identity."""

    artifact_kind = identifier(kind, label="kind")
    identity_name = identifier(identity_field, label="identity_field")
    business_id = artifact_identity(identity, label=identity_name)
    instant = timestamp(created_at, label="created_at")
    body = dict(fields)
    if identity_name in body:
        _fail(f"{identity_name} must be derived, not caller supplied")
    if research_only:
        body = research_payload(body)
    else:
        canonical_value(body)
    payload = {identity_name: business_id, **body}
    artifact = seal_artifact(artifact_kind, payload, created_at=instant)
    return validate_stable_artifact(artifact, expected_kind=artifact_kind)


def validate_stable_artifact(
    artifact: Mapping[str, Any] | bytes,
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    try:
        normalized = validate_artifact(artifact, expected_kind=expected_kind)
    except Exception as exc:
        raise IntelligenceError("artifact envelope validation failed") from exc
    canonical_value(normalized)
    return normalized


def artifact_payload(
    artifact: Mapping[str, Any] | bytes,
    *,
    expected_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = validate_stable_artifact(artifact, expected_kind=expected_kind)
    payload = normalized.get("payload")
    if type(payload) is not dict:
        _fail("artifact payload is invalid")
    return normalized, payload


def artifact_ref(artifact: Mapping[str, Any] | bytes) -> dict[str, str]:
    normalized = validate_stable_artifact(artifact)
    return {
        "artifact_id": artifact_identity(normalized.get("artifact_id"), label="artifact_id"),
        "byte_sha256": artifact_byte_sha256(normalized),
        "contract_sha256": sha256(normalized.get("contract_sha256"), label="contract_sha256"),
        "kind": identifier(normalized.get("kind"), label="kind"),
        "semantic_sha256": sha256(normalized.get("semantic_sha256"), label="semantic_sha256"),
    }


def validate_artifact_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    row = require_exact_keys(value, ARTIFACT_REF_FIELDS, label=label)
    return {
        "artifact_id": artifact_identity(row["artifact_id"], label=f"{label}.artifact_id"),
        "byte_sha256": sha256(row["byte_sha256"], label=f"{label}.byte_sha256"),
        "contract_sha256": sha256(row["contract_sha256"], label=f"{label}.contract_sha256"),
        "kind": identifier(row["kind"], label=f"{label}.kind"),
        "semantic_sha256": sha256(row["semantic_sha256"], label=f"{label}.semantic_sha256"),
    }


def require_artifact_ref(
    reference: Mapping[str, Any],
    artifact: Mapping[str, Any] | bytes,
    *,
    label: str,
) -> dict[str, str]:
    expected = artifact_ref(artifact)
    observed = validate_artifact_ref(reference, label=label)
    if observed != expected:
        _fail(f"{label} does not bind the exact artifact")
    return observed


def require_no_future(artifact: Mapping[str, Any] | bytes, *, as_of: str, label: str) -> None:
    normalized = validate_stable_artifact(artifact)
    cutoff = timestamp(as_of, label="as_of")
    if timestamp(normalized.get("created_at"), label=f"{label}.created_at") > cutoff:
        _fail(f"{label} contains future evidence")


def validate_public_https_url(value: Any, *, label: str = "url") -> str:
    """Validate a literal public HTTPS URL without DNS or network activity."""

    if type(value) is not str or len(value.encode("utf-8")) > 2048:
        _fail(f"{label} is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise IntelligenceError(f"{label} is invalid") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port not in {None, 443}
    ):
        _fail(f"{label} must be an authority-closed HTTPS URL")
    host = parsed.hostname.rstrip(".").casefold()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        _fail(f"{label} host is not public")
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        if "." not in host or any(not part for part in host.split(".")):
            _fail(f"{label} host is not canonical")
    else:
        if not address.is_global:
            _fail(f"{label} host is not public")
    return value


__all__ = [
    "ACTIVATION_BINDING_FIELDS",
    "ARTIFACT_REF_FIELDS",
    "DECIMAL_QUANTUM",
    "FORBIDDEN_VERSION_FIELDS",
    "CONTROL_AUTHORITY_FIELDS",
    "IntelligenceError",
    "NO_AUTHORITY",
    "artifact_payload",
    "artifact_identity",
    "artifact_ref",
    "build_artifact",
    "business_identity",
    "canonical_value",
    "company_code",
    "decimal_text",
    "decimal_value",
    "identifier",
    "require_artifact_ref",
    "require_no_activation_binding",
    "require_exact_keys",
    "require_no_future",
    "require_no_control_authority",
    "research_payload",
    "sha256",
    "sorted_unique_identifiers",
    "timestamp",
    "validate_artifact_ref",
    "validate_public_https_url",
    "validate_stable_artifact",
]
