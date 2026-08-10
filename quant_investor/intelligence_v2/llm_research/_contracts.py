"""Local closed-artifact helpers for I5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
import ipaddress
import unicodedata
from typing import Any, Final, NoReturn
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .._core import (
    NO_AUTHORITY,
    canonical_bytes,
    common_fields,
    content_ref,
    decimal_text,
    decimal_value,
    exact_ref,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_content_ref,
    validate_seal,
)
from .models import I5ContractError

COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
    "version",
}


def fail(message: str) -> NoReturn:
    raise I5ContractError(message)


def text(value: Any, *, label: str, maximum: int = 4000) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        fail(f"{label} must be canonical nonempty text")
    if unicodedata.normalize("NFC", value) != value:
        fail(f"{label} must be Unicode NFC")
    if len(value.encode("utf-8", errors="strict")) > maximum:
        fail(f"{label} exceeds {maximum} UTF-8 bytes")
    return value


def texts(
    values: Sequence[Any],
    *,
    label: str,
    minimum: int = 0,
    maximum: int = 64,
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        fail(f"{label} must be a sequence")
    rows = [text(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if not minimum <= len(rows) <= maximum or len(rows) != len(set(rows)):
        fail(f"{label} cardinality or uniqueness is invalid")
    return rows


def identifiers(values: Sequence[Any], *, label: str, maximum: int = 256) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        fail(f"{label} must be a sequence")
    rows = [identifier(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if len(rows) > maximum or len(rows) != len(set(rows)):
        fail(f"{label} cardinality or uniqueness is invalid")
    return sorted(rows, key=lambda item: item.encode("ascii"))


def artifact(
    *,
    version: str,
    identity_field: str,
    timestamp_value: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) & COMMON_FIELDS:
        fail("I5 artifact payload shape is invalid")
    try:
        return seal(
            {
                "version": version,
                **common_fields(timestamp_value=timestamp_value),
                **dict(payload),
            },
            identity_field=identity_field,
        )
    except I5ContractError:
        raise
    except Exception as exc:
        raise I5ContractError(str(exc)) from exc


def closed_artifact(
    value: Mapping[str, Any],
    *,
    version: str,
    identity_field: str,
    payload_fields: set[str] | frozenset[str],
) -> dict[str, Any]:
    expected = COMMON_FIELDS | set(payload_fields) | {identity_field, "semantic_sha256"}
    try:
        row = require_exact_keys(value, expected, label=version)
        validated = validate_seal(row, identity_field=identity_field)
        if validated["version"] != version:
            fail("I5 artifact version mismatch")
        if validated["authority"] != NO_AUTHORITY:
            fail("I5 artifact authority is open")
        if validated["research_only"] is not True or validated["production"] is not False:
            fail("I5 artifact research boundary is open")
        timestamp(validated["timestamp"], label="artifact.timestamp")
        return validated
    except I5ContractError:
        raise
    except Exception as exc:
        raise I5ContractError(str(exc)) from exc


def same(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    if canonical_bytes(actual) != canonical_bytes(expected):
        fail(f"{label} differs from deterministic replay")


def ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    try:
        return validate_content_ref(value, label=label)
    except Exception as exc:
        raise I5ContractError(str(exc)) from exc


def exact_source_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    try:
        return exact_ref(value, label=label)
    except Exception as exc:
        raise I5ContractError(str(exc)) from exc


def artifact_ref(value: Mapping[str, Any], *, identity_field: str) -> dict[str, str]:
    try:
        return content_ref(value, identity_field=identity_field)
    except Exception as exc:
        raise I5ContractError(str(exc)) from exc


def digest_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    if type(value) is not bytes:
        fail(f"{label} must be bytes")
    return {"byte_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}


def validate_digest(value: Mapping[str, Any], *, raw: bytes, label: str) -> dict[str, Any]:
    require_exact_keys(value, {"byte_sha256", "size"}, label=label)
    if type(value["size"]) is not int or value["size"] < 0:
        fail(f"{label}.size is invalid")
    sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
    expected = digest_bytes(raw, label=label)
    if dict(value) != expected:
        fail(f"{label} bytes differ from closure")
    return expected


def decimal(
    value: Any,
    *,
    label: str,
    minimum: Decimal = Decimal("0"),
    maximum: Decimal = Decimal("1"),
) -> str:
    try:
        return decimal_text(decimal_value(value, label=label, minimum=minimum, maximum=maximum))
    except Exception as exc:
        raise I5ContractError(str(exc)) from exc


def when(value: Any, *, label: str) -> str:
    try:
        return timestamp(value, label=label)
    except Exception as exc:
        raise I5ContractError(str(exc)) from exc


def _canonical_host(parts: SplitResult, *, label: str) -> str:
    host = parts.hostname
    if host is None or not host or "%" in host:
        fail(f"{label} hostname is invalid")
    if any(ord(character) > 127 for character in host):
        fail(f"{label} IDNA hostname must already be canonical ASCII")
    normalized = host.lower()
    try:
        normalized.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise I5ContractError(f"{label} hostname IDNA is invalid") from exc
    if normalized != host:
        fail(f"{label} hostname must be lowercase canonical ASCII")
    return normalized


def _url_parts(value: Any, *, label: str) -> SplitResult:
    if type(value) is not str or not value or value != value.strip():
        fail(f"{label} URL is invalid")
    if any(character in value for character in ("\\", "\r", "\n", "\t", " ")):
        fail(f"{label} URL is ambiguous")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.fragment:
        fail(f"{label} must be fragment-free HTTP(S)")
    if parts.username is not None or parts.password is not None:
        fail(f"{label} userinfo is forbidden")
    return parts


def _url_host_and_port(parts: SplitResult, *, label: str) -> str:
    host = _canonical_host(parts, label=label)
    try:
        port = parts.port
    except ValueError as exc:
        raise I5ContractError(f"{label} port is invalid") from exc
    effective_port = port or (443 if parts.scheme == "https" else 80)
    if effective_port not in {80, 443}:
        fail(f"{label} port is forbidden")
    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        netloc_host = host
    else:
        netloc_host = (
            f"[{parsed_ip.compressed}]" if parsed_ip.version == 6 else parsed_ip.compressed
        )
    return netloc_host if port is None else f"{netloc_host}:{port}"


def canonical_url(value: Any, *, label: str) -> str:
    parts = _url_parts(value, label=label)
    netloc = _url_host_and_port(parts, label=label)
    path = parts.path or "/"
    canonical = urlunsplit((parts.scheme, netloc, path, parts.query, ""))
    if canonical != value:
        fail(f"{label} URL is not canonical")
    return canonical


__all__ = [
    "COMMON_FIELDS",
    "artifact",
    "artifact_ref",
    "canonical_url",
    "closed_artifact",
    "decimal",
    "digest_bytes",
    "exact_source_ref",
    "fail",
    "identifiers",
    "ref",
    "same",
    "text",
    "texts",
    "validate_digest",
    "when",
]
