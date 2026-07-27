"""Canonical identity primitives for protocol v3."""

from __future__ import annotations

from collections.abc import Collection, Sequence
import re
from typing import Any, Final

_OPAQUE_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$", re.ASCII)
_PATH_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$", re.ASCII)
_REGISTRY_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$", re.ASCII)
_SECURITY_CODE_RE: Final = re.compile(r"^[0-9]{6}\.(?:SZ|SH|BJ)$", re.ASCII)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_UTC_CUTOFF_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)


class IdentityContractError(ValueError):
    """Raised when an identifier is noncanonical or collision-prone."""

    exit_code = 2


def require_opaque_id(value: Any, *, label: str = "identifier") -> str:
    if type(value) is not str or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a lowercase canonical identifier")
    return value


def require_path_id(value: Any, *, label: str = "path identifier") -> str:
    if type(value) is not str or _PATH_ID_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a lowercase path-safe identifier")
    return value


def require_security_code(value: Any, *, label: str = "security code") -> str:
    if type(value) is not str or _SECURITY_CODE_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must match six digits plus .SZ, .SH, or .BJ")
    return value


def require_sha256(value: Any, *, label: str = "SHA-256") -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def require_utc_cutoff(value: Any, *, label: str = "cutoff") -> str:
    if type(value) is not str or _UTC_CUTOFF_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a second-precision UTC timestamp")
    return value


def _require_registry_token(value: Any, *, label: str) -> str:
    if type(value) is not str or _REGISTRY_TOKEN_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a canonical ASCII registry token")
    return value


def require_casefold_unique(
    values: Sequence[Any],
    *,
    label: str = "registry",
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise IdentityContractError(f"{label} must be an array")
    result: list[str] = []
    seen: dict[str, str] = {}
    for index, value in enumerate(values):
        token = _require_registry_token(value, label=f"{label}[{index}]")
        collision_key = token.lower()
        if collision_key in seen:
            raise IdentityContractError(
                f"{label} has ASCII-casefold collision: {seen[collision_key]!r} and {token!r}"
            )
        seen[collision_key] = token
        result.append(token)
    return tuple(result)


def require_registry_token(
    value: Any,
    *,
    registry: Collection[str],
    label: str = "registry token",
) -> str:
    if isinstance(registry, (str, bytes, bytearray)) or not isinstance(registry, Collection):
        raise IdentityContractError("registry must be a collection")
    entries = tuple(_require_registry_token(item, label="registry entry") for item in registry)
    require_casefold_unique(tuple(sorted(entries)), label="registry")
    token = _require_registry_token(value, label=label)
    if token not in frozenset(entries):
        raise IdentityContractError(f"{label} is not an exact registered token")
    return token


__all__ = [
    "IdentityContractError",
    "require_casefold_unique",
    "require_opaque_id",
    "require_path_id",
    "require_registry_token",
    "require_security_code",
    "require_sha256",
    "require_utc_cutoff",
]
