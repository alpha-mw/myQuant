"""Identity validation helpers for V17 v5."""

from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Any

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$", re.ASCII)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.ASCII)


class IdentityContractError(ValueError):
    """Raised when an identity or path is not canonical."""

    exit_code = 2


def require_identifier(value: Any, *, label: str = "identifier") -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise IdentityContractError(f"{label} is not a canonical identifier")
    return value


def require_sha256(value: Any, *, label: str = "SHA-256") -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise IdentityContractError(f"{label} is not a lowercase SHA-256")
    return value


def require_git_commit(value: Any, *, label: str = "Git commit") -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise IdentityContractError(f"{label} is not a full lowercase Git commit")
    return value


def require_relative_path(value: Any, *, label: str = "relative path") -> str:
    if type(value) is not str or not value or "\\" in value:
        raise IdentityContractError(f"{label} is not a canonical POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise IdentityContractError(f"{label} escapes its trusted root")
    if path.as_posix() != value:
        raise IdentityContractError(f"{label} is not normalized")
    return value


__all__ = [
    "IdentityContractError",
    "require_git_commit",
    "require_identifier",
    "require_relative_path",
    "require_sha256",
]
