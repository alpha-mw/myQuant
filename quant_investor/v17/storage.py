"""Private, durable local storage helpers for v17 shadow artifacts.

This module performs no discovery and creates nothing at import time.  Callers
must supply explicit roots.  Writes use mode 0600, an exclusive temporary file,
file fsync, atomic rename, and parent-directory fsync; directories are forced to
0700.  Symlink and hard-link ambiguity is rejected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid
from typing import Any

from .contracts import V17ContractError
from .semantic import canonical_json_bytes, require_sha256

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_STREAM_OBJECT_BYTES = 8 * 1024 * 1024 * 1024
_STREAM_CHUNK_BYTES = 1024 * 1024
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_REGULAR_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V17ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> None:
    raise V17ContractError(f"non-finite JSON number rejected: {token}")


def _assert_no_symlink_components(path: Path, *, allow_missing_leaf: bool) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    if not parts:
        raise V17ContractError("empty filesystem path")
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        if not os.path.lexists(current):
            if allow_missing_leaf and index == len(parts) - 1:
                return
            continue
        try:
            mode = os.lstat(current).st_mode
        except OSError as exc:
            raise V17ContractError(f"filesystem identity unavailable: {current}") from exc
        if stat.S_ISLNK(mode):
            raise V17ContractError(f"symlink path component rejected: {current}")


def _assert_beneath(path: Path, root: Path) -> tuple[Path, Path]:
    # ``Path.absolute()`` is not a security boundary on every supported
    # Python/platform combination because lexical ``..`` components may be
    # retained.  ``abspath`` normalizes those components without following
    # symlinks; symlink rejection remains the separate check below.
    target = Path(os.path.abspath(os.fspath(path)))
    boundary = Path(os.path.abspath(os.fspath(root)))
    try:
        target.relative_to(boundary)
    except ValueError as exc:
        raise V17ContractError(f"path escapes fixed root: {target}") from exc
    return target, boundary


def _open_existing_directory(
    path: Path,
    *,
    create_missing: bool = False,
) -> tuple[int, os.stat_result]:
    """Open an absolute directory by walking every component without links."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute() or not absolute.anchor:
        raise V17ContractError(f"directory path must be absolute: {path}")
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, _DIRECTORY_OPEN_FLAGS)
        entry = os.fstat(descriptor)
        if not stat.S_ISDIR(entry.st_mode):
            raise V17ContractError(f"directory anchor is invalid: {absolute.anchor}")
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise V17ContractError(f"unsafe directory component: {component!r}")
            private_candidate = False
            try:
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create_missing:
                    raise
                private_candidate = True
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            next_entry = os.fstat(next_descriptor)
            if not stat.S_ISDIR(next_entry.st_mode):
                os.close(next_descriptor)
                raise V17ContractError(f"directory component is invalid: {absolute}")
            if private_candidate:
                os.fchmod(next_descriptor, 0o700)
                os.fsync(next_descriptor)
                next_entry = os.fstat(next_descriptor)
                if stat.S_IMODE(next_entry.st_mode) != 0o700:
                    os.close(next_descriptor)
                    raise V17ContractError(f"created private directory mode invalid: {absolute}")
            os.close(descriptor)
            descriptor = next_descriptor
            entry = next_entry
        return descriptor, entry
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise V17ContractError(f"directory path unavailable without symlinks: {absolute}") from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _create_or_open_private_child(parent_descriptor: int, name: str) -> tuple[int, os.stat_result]:
    if name in {"", ".", ".."} or "/" in name:
        raise V17ContractError("unsafe private directory component")
    created = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        created = True
        os.fsync(parent_descriptor)
    except FileExistsError:
        pass
    try:
        descriptor = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor)
    except OSError as exc:
        raise V17ContractError(f"private directory component unavailable: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise V17ContractError(f"private directory component invalid: {name}")
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino)
            or stat.S_IMODE(after.st_mode) != 0o700
        ):
            raise V17ContractError(f"private directory component identity drift: {name}")
        if not created and stat.S_IMODE(named.st_mode) != 0o700:
            raise V17ContractError(f"private directory component mode drift: {name}")
        return descriptor, after
    except Exception:
        os.close(descriptor)
        raise


def _open_private_directory(
    path: Path,
    *,
    root: Path,
) -> tuple[Path, int, os.stat_result]:
    """Create/open a private directory using openat for every component."""

    target, boundary = _assert_beneath(path, root)
    if boundary.parent == boundary:
        raise V17ContractError("filesystem root cannot be a private storage boundary")
    boundary_parent_descriptor, _ = _open_existing_directory(
        boundary.parent,
        create_missing=True,
    )
    descriptor: int | None = None
    try:
        descriptor, entry = _create_or_open_private_child(
            boundary_parent_descriptor,
            boundary.name,
        )
    finally:
        os.close(boundary_parent_descriptor)
    try:
        for component in target.relative_to(boundary).parts:
            next_descriptor, next_entry = _create_or_open_private_child(
                descriptor,
                component,
            )
            os.close(descriptor)
            descriptor = next_descriptor
            entry = next_entry
        return target, descriptor, entry
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def ensure_private_directory(path: str | Path, *, root: str | Path) -> Path:
    """Create *path* beneath *root* and enforce mode 0700 on created levels."""

    target, descriptor, entry = _open_private_directory(Path(path), root=Path(root))
    os.close(descriptor)
    _verify_directory_path_identity(target, entry)
    return target


def _regular_file_signature(entry: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        entry.st_dev,
        entry.st_ino,
        entry.st_nlink,
        entry.st_size,
        entry.st_mtime_ns,
        entry.st_ctime_ns,
        stat.S_IMODE(entry.st_mode),
    )


def file_sha256(path: str | Path, *, require_single_link: bool = True) -> str:
    source = Path(path)
    descriptor, before = _open_stable_regular_path(source)
    try:
        if require_single_link and before.st_nlink != 1:
            raise V17ContractError(f"hard-linked file rejected: {source}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _regular_file_signature(before) != _regular_file_signature(after):
            raise V17ContractError(f"file changed during hashing: {source}")
        _verify_source_path_identity(source, before)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_stream_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise V17ContractError("maximum_bytes must be a positive integer")
    return value


def _open_stable_regular_source(
    path: Path,
    *,
    maximum_bytes: int,
    expected_size_bytes: int | None,
) -> tuple[int, os.stat_result]:
    descriptor, entry = _open_stable_regular_path(path)
    try:
        if entry.st_nlink != 1:
            raise V17ContractError(f"hard-linked stream source rejected: {path}")
        if entry.st_size <= 0 or entry.st_size > maximum_bytes:
            raise V17ContractError(f"stream source size is outside fixed bounds: {path}")
        if expected_size_bytes is not None and entry.st_size != expected_size_bytes:
            raise V17ContractError(f"stream source size mismatch: {path}")
        return descriptor, entry
    except Exception:
        os.close(descriptor)
        raise


def _open_stable_regular_path(path: Path) -> tuple[int, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.parent == absolute or not absolute.name:
        raise V17ContractError(f"regular file path is invalid: {absolute}")
    parent_descriptor, parent_entry = _open_existing_directory(absolute.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            absolute.name,
            _REGULAR_OPEN_FLAGS,
            dir_fd=parent_descriptor,
        )
        entry = os.fstat(descriptor)
        named = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(entry.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _regular_file_signature(named) != _regular_file_signature(entry)
        ):
            raise V17ContractError(f"regular file path identity drift: {absolute}")
        _verify_directory_path_identity(
            absolute.parent,
            parent_entry,
            require_private_mode=False,
        )
        return descriptor, entry
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise V17ContractError(f"file unavailable without symlinks: {absolute}") from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        os.close(parent_descriptor)


def _verify_source_path_identity(path: Path, expected: os.stat_result) -> None:
    descriptor: int | None = None
    try:
        descriptor, observed = _open_stable_regular_path(path)
    except V17ContractError as exc:
        raise V17ContractError(f"stream source path unavailable after read: {path}") from exc
    try:
        if _regular_file_signature(observed) != _regular_file_signature(expected):
            raise V17ContractError(f"stream source path changed during read: {path}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_directory_path_identity(
    path: Path,
    expected: os.stat_result,
    *,
    require_private_mode: bool = True,
) -> None:
    descriptor: int | None = None
    try:
        descriptor, observed = _open_existing_directory(path)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino)
            or (require_private_mode and stat.S_IMODE(observed.st_mode) != 0o700)
        ):
            raise V17ContractError("stream target parent path identity drift")
    except V17ContractError as exc:
        raise V17ContractError("stream target parent path identity drift") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_private_parent(target: Path, *, boundary: Path) -> tuple[Path, int, os.stat_result]:
    parent, descriptor, entry = _open_private_directory(target.parent, root=boundary)
    if not stat.S_ISDIR(entry.st_mode) or stat.S_IMODE(entry.st_mode) != 0o700:
        os.close(descriptor)
        raise V17ContractError("atomic target parent is not a private directory")
    _verify_directory_path_identity(parent, entry)
    return parent, descriptor, entry


def _verify_private_parent(
    path: Path,
    descriptor: int,
    expected: os.stat_result,
) -> None:
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
        or (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise V17ContractError("atomic target parent identity drift")
    _verify_directory_path_identity(path, expected)


def _hash_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    expected_size_bytes: int,
    wait_for_single_link: bool = False,
) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    before: os.stat_result | None = None
    for attempt in range(101):
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise V17ContractError("exact-once target unavailable after installation") from exc
        before = os.fstat(descriptor)
        if before.st_nlink == 1:
            break
        os.close(descriptor)
        descriptor = None
        if not wait_for_single_link or before.st_nlink != 2 or attempt == 100:
            raise V17ContractError("exact-once target has invalid hard-link count")
        # A concurrent installer briefly exposes the target and its private
        # same-directory temporary hard link.  Wait only for that bounded CAS
        # window; a persistent or larger link count remains a hard failure.
        time.sleep(0.001)
    if descriptor is None or before is None:  # pragma: no cover - loop invariant
        raise V17ContractError("exact-once target unavailable after installation")
    try:
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != expected_size_bytes
        ):
            raise V17ContractError("exact-once target identity invalid after installation")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _STREAM_CHUNK_BYTES):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _regular_file_signature(after) != _regular_file_signature(before):
            raise V17ContractError("exact-once target changed during readback")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def atomic_copy_file_exact_once(
    source_path: str | Path,
    target_path: str | Path,
    *,
    root: str | Path,
    expected_source_sha256: str,
    expected_size_bytes: int | None = None,
    maximum_bytes: int = MAX_STREAM_OBJECT_BYTES,
) -> str:
    """Stream one stable file into immutable private CAS storage.

    The source is opened once without following links.  Hashing and copying use
    that descriptor, and the source descriptor plus lexical path are checked
    again before installation.  The target parent is held by directory FD and
    installation uses a same-directory hard-link CAS, so an existing name is
    never replaced.  Byte-identical concurrent installation is idempotent;
    any other existing target is rejected.
    """

    maximum = _validate_stream_limit(maximum_bytes)
    if expected_size_bytes is not None and (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes <= 0
        or expected_size_bytes > maximum
    ):
        raise V17ContractError("expected_size_bytes is outside fixed bounds")
    expected = require_sha256(
        expected_source_sha256,
        label="expected source byte SHA-256",
    )
    source = Path(source_path).absolute()
    target, boundary = _assert_beneath(Path(target_path), Path(root))
    if target == boundary:
        raise V17ContractError("stream target cannot equal its root")

    source_descriptor, source_before = _open_stable_regular_source(
        source,
        maximum_bytes=maximum,
        expected_size_bytes=expected_size_bytes,
    )
    try:
        parent, parent_descriptor, parent_before = _open_private_parent(
            target,
            boundary=boundary,
        )
    except Exception:
        os.close(source_descriptor)
        raise

    temporary_name = f".{target.name}.{uuid.uuid4().hex}.tmp"
    temporary_descriptor: int | None = None
    temporary_exists = False
    installed = False
    try:
        create_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temporary_descriptor = os.open(
            temporary_name,
            create_flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_exists = True
        digest = hashlib.sha256()
        copied = 0
        while chunk := os.read(source_descriptor, _STREAM_CHUNK_BYTES):
            digest.update(chunk)
            view = memoryview(chunk)
            offset = 0
            while offset < len(view):
                written = os.write(temporary_descriptor, view[offset:])
                if written <= 0:
                    raise V17ContractError("short streamed CAS write")
                offset += written
            copied += len(chunk)
        source_after = os.fstat(source_descriptor)
        if _regular_file_signature(source_after) != _regular_file_signature(source_before):
            raise V17ContractError(f"stream source changed during read: {source}")
        _verify_source_path_identity(source, source_before)
        if copied != source_before.st_size:
            raise V17ContractError("stream source read length mismatch")
        observed = digest.hexdigest()
        if observed != expected:
            raise V17ContractError("stream source byte SHA mismatch")

        os.fchmod(temporary_descriptor, 0o600)
        os.fsync(temporary_descriptor)
        temporary_entry = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(temporary_entry.st_mode)
            or temporary_entry.st_size != copied
            or stat.S_IMODE(temporary_entry.st_mode) != 0o600
        ):
            raise V17ContractError("streamed CAS temporary validation failed")
        os.close(temporary_descriptor)
        temporary_descriptor = None

        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            installed = True
            os.fsync(parent_descriptor)
        except FileExistsError:
            installed = False
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_exists = False
        os.fsync(parent_descriptor)

        target_digest = _hash_regular_at(
            parent_descriptor,
            target.name,
            expected_size_bytes=copied,
            wait_for_single_link=not installed,
        )
        if target_digest != expected:
            qualifier = "concurrent " if not installed else ""
            raise V17ContractError(f"{qualifier}exact-once target byte mismatch")
        parent_after = os.fstat(parent_descriptor)
        if (parent_after.st_dev, parent_after.st_ino) != (
            parent_before.st_dev,
            parent_before.st_ino,
        ):
            raise V17ContractError("stream target parent identity drift")
        _verify_directory_path_identity(parent, parent_before)
        return expected
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
        os.close(source_descriptor)
        os.close(parent_descriptor)


def read_json(
    path: str | Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
    require_single_link: bool = True,
) -> dict[str, Any]:
    """Read one stable, regular JSON object without following links."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise V17ContractError("max_bytes must be a positive integer")
    source = Path(path)
    descriptor, before = _open_stable_regular_path(source)
    try:
        if require_single_link and before.st_nlink != 1:
            raise V17ContractError(f"hard-linked JSON input rejected: {source}")
        if before.st_size > max_bytes:
            raise V17ContractError(f"JSON input exceeds size limit: {source}")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise V17ContractError(f"JSON input exceeds size limit: {source}")
        after = os.fstat(descriptor)
        if _regular_file_signature(before) != _regular_file_signature(after):
            raise V17ContractError(f"JSON input changed while reading: {source}")
        _verify_source_path_identity(source, before)
    finally:
        os.close(descriptor)

    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise V17ContractError(f"invalid UTF-8 JSON input: {source}") from exc
    if not isinstance(value, Mapping):
        raise V17ContractError(f"JSON root must be an object: {source}")
    # Canonical encoding validation catches unsupported/non-finite values.
    canonical_json_bytes(value)
    return dict(value)


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    root: str | Path,
) -> str:
    """Durably replace one file beneath an explicit private root."""

    if not isinstance(payload, bytes):
        raise V17ContractError("atomic payload must be bytes")
    target, boundary = _assert_beneath(Path(path), Path(root))
    if target == boundary:
        raise V17ContractError("atomic file target cannot equal its root")
    _assert_no_symlink_components(target, allow_missing_leaf=True)
    parent, parent_descriptor, parent_before = _open_private_parent(
        target,
        boundary=boundary,
    )
    try:
        current = os.stat(target.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        current = None
    except OSError as exc:
        os.close(parent_descriptor)
        raise V17ContractError("atomic target identity unavailable") from exc
    if current is not None and (
        not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode) or current.st_nlink != 1
    ):
        os.close(parent_descriptor)
        raise V17ContractError(f"atomic target is not a single-link regular file: {target}")

    temporary_name = f".{target.name}.{uuid.uuid4().hex}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    temporary_descriptor: int | None = None
    temporary_exists = False
    digest = hashlib.sha256(payload).hexdigest()
    try:
        temporary_descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_exists = True
        written = 0
        view = memoryview(payload)
        while written < len(payload):
            count = os.write(temporary_descriptor, view[written:])
            if count <= 0:
                raise V17ContractError("short atomic write")
            written += count
        os.fchmod(temporary_descriptor, 0o600)
        os.fsync(temporary_descriptor)
        before = os.fstat(temporary_descriptor)
        if before.st_size != len(payload) or not stat.S_ISREG(before.st_mode):
            raise V17ContractError("atomic temporary file validation failed")
        os.close(temporary_descriptor)
        temporary_descriptor = None
        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_exists = False
        os.fsync(parent_descriptor)
        if (
            _hash_regular_at(
                parent_descriptor,
                target.name,
                expected_size_bytes=len(payload),
            )
            != digest
        ):
            raise V17ContractError("atomic target readback mismatch")
        _verify_private_parent(parent, parent_descriptor, parent_before)
        return digest
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: str | Path,
) -> str:
    if not isinstance(payload, Mapping):
        raise V17ContractError("JSON write payload must be an object")
    raw = canonical_json_bytes(payload) + b"\n"
    return atomic_write_bytes(path, raw, root=root)


def _validate_exact_existing_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    *,
    wait_for_single_link: bool = False,
) -> str:
    expected = hashlib.sha256(payload).hexdigest()
    if (
        _hash_regular_at(
            parent_descriptor,
            name,
            expected_size_bytes=len(payload),
            wait_for_single_link=wait_for_single_link,
        )
        != expected
    ):
        raise V17ContractError("exact-once target already exists with different bytes")
    return expected


def atomic_write_bytes_exact_once(
    path: str | Path,
    payload: bytes,
    *,
    root: str | Path,
) -> str:
    """Install immutable bytes once; same bytes are idempotent, drift is rejected."""

    if not isinstance(payload, bytes):
        raise V17ContractError("atomic payload must be bytes")
    target, boundary = _assert_beneath(Path(path), Path(root))
    if target == boundary:
        raise V17ContractError("atomic file target cannot equal its root")
    _assert_no_symlink_components(target, allow_missing_leaf=True)
    parent, parent_descriptor, parent_before = _open_private_parent(
        target,
        boundary=boundary,
    )
    try:
        existing = os.stat(target.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        os.close(parent_descriptor)
        raise V17ContractError("exact-once target identity unavailable") from exc
    if existing is not None:
        try:
            digest = _validate_exact_existing_at(parent_descriptor, target.name, payload)
            _verify_private_parent(parent, parent_descriptor, parent_before)
            return digest
        finally:
            os.close(parent_descriptor)

    temporary_name = f".{target.name}.{uuid.uuid4().hex}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    temporary_descriptor: int | None = None
    temporary_exists = False
    installed = False
    try:
        temporary_descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_exists = True
        written = 0
        view = memoryview(payload)
        while written < len(payload):
            count = os.write(temporary_descriptor, view[written:])
            if count <= 0:
                raise V17ContractError("short exact-once write")
            written += count
        os.fchmod(temporary_descriptor, 0o600)
        os.fsync(temporary_descriptor)
        if os.fstat(temporary_descriptor).st_size != len(payload):
            raise V17ContractError("exact-once temporary file size mismatch")
        os.close(temporary_descriptor)
        temporary_descriptor = None
        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            installed = True
        except FileExistsError:
            installed = False
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_exists = False
        os.fsync(parent_descriptor)

        digest = _validate_exact_existing_at(
            parent_descriptor,
            target.name,
            payload,
            wait_for_single_link=not installed,
        )
        _verify_private_parent(parent, parent_descriptor, parent_before)
        return digest
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)


def atomic_write_json_exact_once(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: str | Path,
) -> str:
    if not isinstance(payload, Mapping):
        raise V17ContractError("JSON write payload must be an object")
    return atomic_write_bytes_exact_once(path, canonical_json_bytes(payload) + b"\n", root=root)


def ensure_v17_shadow_layout(repo_root: str | Path) -> dict[str, Path]:
    """Create only the fixed v17 shadow/private-source directory layout."""

    repo = Path(repo_root).absolute()
    results_root = repo / "results" / "v17_shadow"
    private_root = repo / "data" / "private" / "v17_sources"
    # The private roots are their own chmod boundaries.  In particular, never
    # pass the repository root as the private boundary: that would mutate the
    # repository (and possibly shared results/data parents) to mode 0700.
    layout = {
        "results": ensure_private_directory(results_root, root=results_root),
        "runs": ensure_private_directory(results_root / "runs", root=results_root),
        "models": ensure_private_directory(results_root / "models", root=results_root),
        "outcomes": ensure_private_directory(results_root / "outcomes", root=results_root),
        "latest": ensure_private_directory(results_root / "_latest", root=results_root),
        "private_sources": ensure_private_directory(private_root, root=private_root),
        "source_objects": ensure_private_directory(private_root / "objects", root=private_root),
        "source_manifests": ensure_private_directory(private_root / "manifests", root=private_root),
    }
    return layout


__all__ = [
    "MAX_JSON_BYTES",
    "MAX_STREAM_OBJECT_BYTES",
    "atomic_copy_file_exact_once",
    "atomic_write_bytes",
    "atomic_write_bytes_exact_once",
    "atomic_write_json",
    "atomic_write_json_exact_once",
    "ensure_private_directory",
    "ensure_v17_shadow_layout",
    "file_sha256",
    "read_json",
]
