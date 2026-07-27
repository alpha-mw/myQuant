"""Canonical v4 identity primitives."""

from __future__ import annotations

import re
from typing import Any, Final

_OPAQUE_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$", re.ASCII)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_UTC_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)


class IdentityContractError(ValueError):
    """Raised when an identity is ambiguous or noncanonical."""

    exit_code = 2


def require_opaque_id(value: Any, *, label: str = "identifier") -> str:
    if type(value) is not str or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a lowercase canonical identifier")
    return value


def require_sha256(value: Any, *, label: str = "SHA-256") -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def require_utc_timestamp(value: Any, *, label: str = "timestamp") -> str:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a second-precision UTC timestamp")
    return value


__all__ = [
    "IdentityContractError",
    "require_opaque_id",
    "require_sha256",
    "require_utc_timestamp",
]
