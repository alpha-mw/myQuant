"""Read the one release pointer the scheduled CN jobs run from.

``operations/releases/active.env`` names the frozen release install, its clean
checkout and the factor-loop context bound to it. Jobs read it instead of
carrying their own copies of those paths, so a repoint is one reviewed edit.
This reader mirrors ``scripts/operations/release_pointer.sh`` and fails closed
with a named blocker on any inconsistency; it never guesses a release.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

KNOWN_KEYS = (
    "RELEASE_COMMIT",
    "RELEASE_INSTALL_DIR",
    "RELEASE_CHECKOUT_DIR",
    "RELEASE_INSTALL_INPUT_SHA256",
    "FACTOR_LOOP_CONTEXT",
    "FACTOR_LOOP_CONTEXT_SHA256",
    "PRUNE_RELEASE_INSTALL_DIR",
)
REQUIRED_KEYS = KNOWN_KEYS[:-1]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ReleasePointerError(RuntimeError):
    """The pointer file is missing, malformed or does not match the filesystem."""

    exit_code = 2

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ReleasePointer:
    commit: str
    install_dir: Path
    checkout_dir: Path
    install_input_sha256: str
    factor_loop_context: Path
    factor_loop_context_sha256: str
    prune_install_dir: Path | None
    pointer_sha256: str

    @property
    def python(self) -> Path:
        return self.install_dir / "bin" / "python"

    @property
    def prune_python(self) -> Path:
        base = self.prune_install_dir or self.install_dir
        return base / "bin" / "python"


def parse_release_pointer(text: str) -> dict[str, str]:
    """KEY=VALUE lines; blanks and ``#`` comments are skipped, nothing is evaluated."""

    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not _KEY.fullmatch(key):
            raise ReleasePointerError("RELEASE_POINTER_LINE_INVALID")
        if key not in KNOWN_KEYS:
            raise ReleasePointerError(f"RELEASE_POINTER_KEY_UNKNOWN:{key}")
        values[key] = value
    missing = [key for key in REQUIRED_KEYS if not values.get(key)]
    if missing:
        raise ReleasePointerError(f"RELEASE_POINTER_KEY_MISSING:{missing[0]}")
    return values


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def read_release_pointer(path: Path, *, workspace_root: Path) -> ReleasePointer:
    """Read ``path`` and check every reference against the filesystem."""

    if not path.is_file():
        raise ReleasePointerError("RELEASE_POINTER_MISSING")
    raw = path.read_bytes()
    values = parse_release_pointer(raw.decode("utf-8"))
    commit = values["RELEASE_COMMIT"]
    if not _COMMIT.fullmatch(commit):
        raise ReleasePointerError("RELEASE_POINTER_COMMIT_INVALID")
    install_dir = Path(values["RELEASE_INSTALL_DIR"])
    if (
        not install_dir.is_absolute()
        or not install_dir.name.startswith(f"{commit}-")
        or not _executable(install_dir / "bin" / "python")
    ):
        raise ReleasePointerError("RELEASE_POINTER_INSTALL_INVALID")
    checkout_dir = Path(values["RELEASE_CHECKOUT_DIR"])
    if (
        not checkout_dir.is_absolute()
        or not checkout_dir.name.startswith(f"{commit}-")
        or not checkout_dir.is_dir()
    ):
        raise ReleasePointerError("RELEASE_POINTER_CHECKOUT_INVALID")
    input_sha = values["RELEASE_INSTALL_INPUT_SHA256"]
    context_sha = values["FACTOR_LOOP_CONTEXT_SHA256"]
    if not _SHA256.fullmatch(input_sha) or not _SHA256.fullmatch(context_sha):
        raise ReleasePointerError("RELEASE_POINTER_SHA_INVALID")
    context = Path(values["FACTOR_LOOP_CONTEXT"])
    if not context.is_absolute():
        context = workspace_root / context
    if not context.is_file() or hashlib.sha256(context.read_bytes()).hexdigest() != context_sha:
        raise ReleasePointerError("RELEASE_POINTER_CONTEXT_SHA_MISMATCH")
    prune_dir: Path | None = None
    if values.get("PRUNE_RELEASE_INSTALL_DIR"):
        prune_dir = Path(values["PRUNE_RELEASE_INSTALL_DIR"])
        if not prune_dir.is_absolute() or not _executable(prune_dir / "bin" / "python"):
            raise ReleasePointerError("RELEASE_POINTER_PRUNE_INSTALL_INVALID")
    return ReleasePointer(
        commit=commit,
        install_dir=install_dir,
        checkout_dir=checkout_dir,
        install_input_sha256=input_sha,
        factor_loop_context=context,
        factor_loop_context_sha256=context_sha,
        prune_install_dir=prune_dir,
        pointer_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "KNOWN_KEYS",
    "REQUIRED_KEYS",
    "ReleasePointer",
    "ReleasePointerError",
    "parse_release_pointer",
    "read_release_pointer",
]
