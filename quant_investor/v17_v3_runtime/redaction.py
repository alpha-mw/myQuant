"""Public-envelope redaction for private strategy and market values."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Final, Mapping, Sequence

_SENSITIVE_KEYS: Final = frozenset(
    {
        "symbol",
        "symbols",
        "security",
        "security_code",
        "security_codes",
        "holdings",
        "review_only_holdings",
        "nav",
        "cash",
        "quantity",
        "quantities",
        "price",
        "prices",
        "target_weight",
        "target_weights",
        "current_weight",
        "current_weights",
        "trade",
        "trades",
        "orders",
    }
)
_SECURITY_CODE = re.compile(r"(?<![0-9])[0-9]{6}\.(?:SZ|SH|BJ)(?![A-Z])")


class PublicEnvelopeError(ValueError):
    """A public envelope retained private strategy content."""


def _key_is_sensitive(key: str) -> bool:
    normalized = key.casefold()
    return normalized in _SENSITIVE_KEYS or any(
        token in normalized for token in ("holding", "quantity", "target_weight", "current_weight")
    )


def public_digest(value: Any) -> str:
    """Hash private detail for equality/debug correlation without disclosure."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raw = type(value).__name__.encode("ascii", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def redact_public(value: Any) -> Any:
    """Drop sensitive fields and redact security codes from arbitrary metadata."""

    if value is None or type(value) in {bool, int, float}:
        return value
    if type(value) is str:
        return _SECURITY_CODE.sub("[REDACTED_SECURITY]", value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text = str(key)
            if _key_is_sensitive(text):
                continue
            result[text] = redact_public(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_public(item) for item in value]
    return type(value).__name__


def assert_public_envelope_safe(value: Any) -> None:
    """Reject public values containing sensitive keys or A-share security codes."""

    def walk(current: Any) -> None:
        if type(current) is str:
            if _SECURITY_CODE.search(current):
                raise PublicEnvelopeError("public envelope contains a security code")
            return
        if isinstance(current, Mapping):
            for key, item in current.items():
                if _key_is_sensitive(str(key)):
                    raise PublicEnvelopeError("public envelope contains a sensitive key")
                walk(item)
            return
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            for item in current:
                walk(item)

    walk(value)


__all__ = [
    "PublicEnvelopeError",
    "assert_public_envelope_safe",
    "public_digest",
    "redact_public",
]
