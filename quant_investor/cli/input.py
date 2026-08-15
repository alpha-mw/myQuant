"""Exact, local-only request loading for the public command line.

Candidate-producing commands accept one canonical JSON file plus its expected
byte SHA-256.  The loader deliberately rejects traversal, symlinks, foreign
ownership, writable-by-others files, hard links, non-canonical bytes, and files
that change while being read.  It never searches for a current pointer or
falls back to another request.
"""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from quant_investor.cli.output import CommandError
from quant_investor.contracts import parse_canonical_json_bytes

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_relative_path(value: str) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CommandError("REQUEST_PATH_INVALID")
    try:
        text.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise CommandError("REQUEST_PATH_INVALID") from exc
    return path


def _verify_file(value: os.stat_result) -> None:
    mode = stat.S_IMODE(value.st_mode)
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or value.st_nlink != 1
        or mode & 0o022
        or not mode & 0o400
        or value.st_size <= 0
        or value.st_size > _MAX_REQUEST_BYTES
    ):
        raise CommandError("REQUEST_FILE_UNSAFE")


def _open_parent(root: Path, path: PurePosixPath) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags | nofollow)
    try:
        for part in path.parts[:-1]:
            next_descriptor = os.open(part, flags | nofollow, dir_fd=descriptor)
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                raise CommandError("REQUEST_PATH_INVALID")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, path.name
    except BaseException:
        os.close(descriptor)
        raise


def _resolve_workspace_root(workspace_root: str | os.PathLike[str]) -> Path:
    try:
        root = Path(workspace_root).resolve(strict=True)
        metadata = os.lstat(root)
    except OSError as exc:
        raise CommandError("WORKSPACE_ROOT_INVALID") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise CommandError("WORKSPACE_ROOT_INVALID")
    return root


def _open_request(parent: int, leaf: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(leaf, flags, dir_fd=parent)
    except OSError as exc:
        code = (
            "REQUEST_SYMLINK_REFUSED"
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}
            else "REQUEST_UNAVAILABLE"
        )
        raise CommandError(code) from exc


def _read_descriptor(descriptor: int, expected_size: int) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks), remaining


def _parse_request(raw: bytes) -> dict[str, Any]:
    try:
        document = parse_canonical_json_bytes(raw, label="command request")
    except Exception as exc:
        raise CommandError("REQUEST_NOT_CANONICAL") from exc
    if type(document) is not dict:
        raise CommandError("REQUEST_NOT_OBJECT")
    return document


def read_exact_request(
    workspace_root: str | os.PathLike[str],
    relative_path: str,
    expected_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Read one exact canonical request below ``workspace_root``."""

    if _SHA256_RE.fullmatch(str(expected_sha256)) is None:
        raise CommandError("REQUEST_SHA256_INVALID")
    root = _resolve_workspace_root(workspace_root)
    parent, leaf = _open_parent(root, _canonical_relative_path(relative_path))
    descriptor: int | None = None
    try:
        descriptor = _open_request(parent, leaf)
        before = os.fstat(descriptor)
        _verify_file(before)
        raw, remaining = _read_descriptor(descriptor, before.st_size)
        after = os.fstat(descriptor)
        _verify_file(after)
        if remaining or len(raw) != after.st_size or _identity(before) != _identity(after):
            raise CommandError("REQUEST_CHANGED_DURING_READ")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise CommandError("REQUEST_SHA256_MISMATCH")
        return raw, _parse_request(raw)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


__all__ = ["read_exact_request"]
