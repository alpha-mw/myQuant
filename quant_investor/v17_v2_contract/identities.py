"""Canonical identity primitives for the isolated v17 protocol-v2 contracts."""

from __future__ import annotations

from collections.abc import Collection, Sequence
import re
from typing import Any, Final

_OPAQUE_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$", re.ASCII)
_PATH_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$", re.ASCII)
_REGISTRY_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$", re.ASCII)
_SECURITY_CODE_RE: Final = re.compile(r"^[0-9]{6}\.(?:SZ|SH|BJ)$", re.ASCII)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


class IdentityContractError(ValueError):
    """Raised when an identity is noncanonical or collision-prone."""

    exit_code = 2


def require_opaque_id(value: Any, *, label: str = "identifier") -> str:
    """Require a lowercase ASCII opaque ID without applying normalization."""

    if type(value) is not str or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a canonical opaque identifier")
    return value


def require_path_id(value: Any, *, label: str = "path identifier") -> str:
    """Require one lowercase ASCII identifier safe as a single path component."""

    if type(value) is not str or _PATH_ID_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a canonical path identifier")
    return value


def require_security_code(value: Any, *, label: str = "security code") -> str:
    """Require a fully qualified canonical mainland China security code."""

    if type(value) is not str or _SECURITY_CODE_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must match six digits plus .SZ, .SH, or .BJ")
    return value


def require_sha256(value: Any, *, label: str = "SHA-256") -> str:
    """Require lowercase hexadecimal SHA-256 without case normalization."""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be lowercase SHA-256")
    return value


def _require_ascii_registry_token(value: Any, *, label: str) -> str:
    if type(value) is not str or _REGISTRY_TOKEN_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a canonical ASCII registry token")
    return value


def ascii_casefold_key(value: Any, *, label: str = "registry token") -> str:
    """Return the collision key for an ASCII token without changing the token."""

    token = _require_ascii_registry_token(value, label=label)
    return token.lower()


def require_ascii_casefold_unique(
    values: Sequence[Any],
    *,
    label: str = "registry",
) -> tuple[str, ...]:
    """Validate registry tokens and reject exact or ASCII-casefold collisions."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise IdentityContractError(f"{label} must be an array")
    normalized: list[str] = []
    seen: dict[str, str] = {}
    for index, value in enumerate(values):
        token = _require_ascii_registry_token(value, label=f"{label}[{index}]")
        collision_key = token.lower()
        previous = seen.get(collision_key)
        if previous is not None:
            raise IdentityContractError(
                f"{label} has ASCII-casefold collision: {previous!r} and {token!r}"
            )
        seen[collision_key] = token
        normalized.append(token)
    return tuple(normalized)


def require_registry_token(
    value: Any,
    *,
    registry: Collection[str],
    label: str = "registry token",
) -> str:
    """Require an exact registry member; never trim, lowercase, or normalize."""

    if isinstance(registry, (str, bytes, bytearray)) or not isinstance(registry, Collection):
        raise IdentityContractError("registry must be a collection of tokens")
    validated = [_require_ascii_registry_token(item, label="registry entry") for item in registry]
    ordered = sorted(validated)
    require_ascii_casefold_unique(ordered, label="registry")
    token = _require_ascii_registry_token(value, label=label)
    if token not in frozenset(validated):
        raise IdentityContractError(f"{label} is not an exact registered token")
    return token


__all__ = [
    "IdentityContractError",
    "ascii_casefold_key",
    "require_ascii_casefold_unique",
    "require_opaque_id",
    "require_path_id",
    "require_registry_token",
    "require_security_code",
    "require_sha256",
]
