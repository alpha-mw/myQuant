"""Deterministic identity for the code actually serving public System reads."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes

from .errors import SystemSecurityError

INSTALLED_CODE_MANIFEST_DOMAIN: Final = "myquant-installed-code-manifest"
_PACKAGE_ROOT: Final = Path(__file__).resolve().parents[1]
_READ_FLAGS: Final = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_MAX_CODE_FILE_BYTES: Final = 64 * 1024 * 1024
_MAX_CODE_MANIFEST_BYTES: Final = 512 * 1024 * 1024
_IGNORED_PARTS: Final = frozenset({"__pycache__"})
_IGNORED_SUFFIXES: Final = frozenset({".pyc", ".pyo"})


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify_code_directory(value: os.stat_result) -> None:
    if not stat.S_ISDIR(value.st_mode):
        raise SystemSecurityError("installed code path is not a directory")
    if value.st_uid != os.geteuid():
        raise SystemSecurityError("installed code directory owner mismatch")
    if stat.S_IMODE(value.st_mode) & 0o022:
        raise SystemSecurityError("installed code directory is group/world writable")


def _runtime_paths() -> tuple[Path, ...]:  # noqa: C901
    rows: list[Path] = []
    try:
        _verify_code_directory(_PACKAGE_ROOT.lstat())
        for directory, directory_names, file_names in os.walk(
            _PACKAGE_ROOT,
            topdown=True,
            followlinks=False,
        ):
            current = Path(directory)
            _verify_code_directory(current.lstat())
            retained_directories: list[str] = []
            for name in sorted(directory_names):
                if name in _IGNORED_PARTS:
                    continue
                child = (current / name).lstat()
                if stat.S_ISLNK(child.st_mode):
                    raise SystemSecurityError(
                        "installed code manifest contains a directory symlink"
                    )
                _verify_code_directory(child)
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in sorted(file_names):
                path = current / name
                relative = path.relative_to(_PACKAGE_ROOT)
                if any(part in _IGNORED_PARTS for part in relative.parts):
                    continue
                if path.suffix in _IGNORED_SUFFIXES:
                    continue
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise SystemSecurityError("installed code manifest contains a symlink")
                if not stat.S_ISREG(metadata.st_mode):
                    raise SystemSecurityError("installed code manifest contains a non-file")
                rows.append(path)
    except SystemSecurityError:
        raise
    except OSError as exc:
        raise SystemSecurityError("installed code manifest cannot be enumerated") from exc
    return tuple(sorted(rows, key=lambda path: path.relative_to(_PACKAGE_ROOT).as_posix()))


def _hash_code_file(path: Path) -> tuple[str, int]:  # noqa: C901
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _READ_FLAGS)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemSecurityError("installed code entry is not a regular file")
        if before.st_uid != os.geteuid():
            raise SystemSecurityError("installed code entry owner mismatch")
        if stat.S_IMODE(before.st_mode) & 0o022:
            raise SystemSecurityError("installed code entry is group/world writable")
        if before.st_nlink != 1:
            raise SystemSecurityError("installed code entry must have one hard link")
        if before.st_size > _MAX_CODE_FILE_BYTES:
            raise SystemSecurityError("installed code entry is outside its byte bound")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        path_after = path.lstat()
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(path_after)
            or size != after.st_size
        ):
            raise SystemSecurityError("installed code entry changed during hashing")
        return digest.hexdigest(), size
    except SystemSecurityError:
        raise
    except OSError as exc:
        raise SystemSecurityError("installed code entry cannot be hashed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def installed_code_manifest() -> dict[str, Any]:
    """Hash the exact non-generated files in the imported ``quant_investor`` tree."""

    before = _runtime_paths()
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in before:
        digest, size = _hash_code_file(path)
        total_bytes += size
        if total_bytes > _MAX_CODE_MANIFEST_BYTES:
            raise SystemSecurityError("installed code manifest exceeds its byte bound")
        files.append(
            {
                "path": "quant_investor/" + path.relative_to(_PACKAGE_ROOT).as_posix(),
                "byte_sha256": digest,
                "size": size,
            }
        )
    if not files or _runtime_paths() != before:
        raise SystemSecurityError("installed code file set changed during hashing")
    return {
        "domain": INSTALLED_CODE_MANIFEST_DOMAIN,
        "files": files,
    }


def installed_code_manifest_sha256() -> str:
    """Return the semantic SHA of the exact code manifest for the running package."""

    return hashlib.sha256(canonical_json_bytes(installed_code_manifest())).hexdigest()


__all__ = [
    "INSTALLED_CODE_MANIFEST_DOMAIN",
    "installed_code_manifest",
    "installed_code_manifest_sha256",
]
