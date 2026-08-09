"""Safe capture primitives for the durable provider-evidence directory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from ...._core import canonical_bytes
from .fileset import (
    REQUIRED_EVIDENCE_PATHS,
    validate_provider_evidence_fileset_manifest,
)
from .models import FundamentalV4ContractError, fundamental_v4_contract


def _regular_bytes(path: Path) -> tuple[bytes, tuple[int, int]]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise FundamentalV4ContractError("provider evidence file is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.getuid()
    ):
        raise FundamentalV4ContractError("provider evidence file is unsafe")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise FundamentalV4ContractError("provider evidence file open failed") from exc
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise FundamentalV4ContractError("provider evidence file changed during read")
    return b"".join(chunks), (after.st_dev, after.st_ino)


def _directory(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise FundamentalV4ContractError("provider evidence directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise FundamentalV4ContractError("provider evidence directory is unsafe")


def _actual_paths(root: Path) -> set[str]:
    values: list[str] = []
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        _directory(base)
        for name in names:
            _directory(base / name)
        for name in files:
            values.append((base / name).relative_to(root).as_posix())
    if len(values) != len({value.casefold() for value in values}):
        raise FundamentalV4ContractError("provider evidence has a casefold collision")
    return set(values)


@fundamental_v4_contract
def capture_provider_evidence_directory(root: str | Path) -> dict[str, bytes]:
    """Capture exact bytes after mode, topology, identity, and hash validation."""

    base = Path(root)
    if not base.is_absolute() or ".." in base.parts or base.resolve(strict=True) != base:
        raise FundamentalV4ContractError("provider evidence root must be absolute")
    _directory(base)
    required = {*REQUIRED_EVIDENCE_PATHS, "fileset_manifest.json"}
    if _actual_paths(base) != required:
        raise FundamentalV4ContractError("provider evidence fileset is not exact")
    manifest_bytes, manifest_inode = _regular_bytes(base / "fileset_manifest.json")
    try:
        manifest_value: Any = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FundamentalV4ContractError("fileset manifest is not JSON") from exc
    manifest = validate_provider_evidence_fileset_manifest(manifest_value)
    if manifest_bytes != canonical_bytes(manifest):
        raise FundamentalV4ContractError("fileset manifest bytes are not canonical")
    result = {"fileset_manifest.json": manifest_bytes}
    inodes = {manifest_inode}
    inventory = {row["relative_path"]: row for row in manifest["inventory"]}
    for relative_path in REQUIRED_EVIDENCE_PATHS:
        payload, inode = _regular_bytes(base / relative_path)
        row = inventory[relative_path]
        if (
            inode in inodes
            or len(payload) != row["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != row["byte_sha256"]
        ):
            raise FundamentalV4ContractError("provider evidence byte identity mismatch")
        inodes.add(inode)
        result[relative_path] = payload
    return result


__all__ = ["capture_provider_evidence_directory"]
