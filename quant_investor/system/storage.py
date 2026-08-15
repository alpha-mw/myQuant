"""Descriptor-relative owner-only storage for unified system authority bytes."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from collections.abc import Callable, Iterator, Mapping
from typing import Final

from quant_investor.contracts import MAX_CANONICAL_JSON_BYTES
from quant_investor.system_authority import (
    ACTIVE_POINTER_PATH,
    EMPTY_POINTER_SHA256,
    MIGRATION_MARKER_PATH,
    SYSTEM_ROOT,
)

from .errors import (
    SystemError,
    SystemCASMismatch,
    SystemImmutableConflict,
    SystemNotFound,
    SystemSecurityError,
    SystemStorageError,
)

OBJECTS_ROOT: Final = SYSTEM_ROOT / "objects"
GENERATIONS_ROOT: Final = SYSTEM_ROOT / "generations"
POINTER_HISTORY_ROOT: Final = SYSTEM_ROOT / "pointer_history"
ACTIVATION_AUTHORIZATIONS_ROOT: Final = SYSTEM_ROOT / "activation_authorizations"
ACTIVATION_TRANSACTIONS_ROOT: Final = SYSTEM_ROOT / "activation_transactions"
FINAL_CUTOVER_AUTHORIZATIONS_ROOT: Final = SYSTEM_ROOT / "final_cutover_authorizations"
CANDIDATE_STATE_ROOT: Final = SYSTEM_ROOT / "candidate_state"
VALIDATION_RUNS_ROOT: Final = SYSTEM_ROOT / "validation_runs"
VALIDATION_REQUESTS_ROOT: Final = SYSTEM_ROOT / "validation_requests"
VALIDATION_CUSTODY_ROOT: Final = SYSTEM_ROOT / "validation_custody"
SOURCE_VERIFICATION_CACHE_ROOT: Final = SYSTEM_ROOT / "source_verification_cache"
_RESERVED_AUTHORITY_PATHS: Final = frozenset({ACTIVE_POINTER_PATH, MIGRATION_MARKER_PATH})

_DIRECTORY_FLAGS: Final = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS: Final = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_CREATE_FLAGS: Final = (
    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)


@dataclass(frozen=True, slots=True)
class StoredBytes:
    relative_path: str
    data: bytes
    byte_sha256: str


@dataclass(frozen=True, slots=True)
class StoredFile:
    relative_path: str
    byte_sha256: str
    size: int
    stat_identity: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class StoredSourceBytes:
    relative_path: str
    data: bytes
    byte_sha256: str
    stat_identity: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _PreparedInitialActivationWrite:
    """Exact bytes already deep-validated by ``SystemStore``."""

    pointer_raw: bytes
    receipt_raw: bytes
    final_authorization_raw: bytes
    activation_authorization_raw: bytes
    prepared_raw: bytes
    marker_raw: bytes


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def source_stat_identity(value: os.stat_result) -> dict[str, int]:
    """Return the exact immutable-source stat projection bound by validation."""

    return {
        "st_ctime_ns": value.st_ctime_ns,
        "st_dev": value.st_dev,
        "st_gid": value.st_gid,
        "st_ino": value.st_ino,
        "st_mode": value.st_mode,
        "st_mtime_ns": value.st_mtime_ns,
        "st_nlink": value.st_nlink,
        "st_size": value.st_size,
        "st_uid": value.st_uid,
    }


def _verify_directory(value: os.stat_result, *, governed: bool) -> None:
    if not stat.S_ISDIR(value.st_mode):
        raise SystemSecurityError("path component is not a directory")
    if governed and value.st_uid != os.geteuid():
        raise SystemSecurityError("governed directory owner mismatch")
    if governed and stat.S_IMODE(value.st_mode) != 0o700:
        raise SystemSecurityError("governed directory mode must be 0700")


def _verify_file(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise SystemSecurityError("artifact is not a regular file")
    if value.st_uid != os.geteuid():
        raise SystemSecurityError("artifact owner mismatch")
    if stat.S_IMODE(value.st_mode) != 0o600:
        raise SystemSecurityError("artifact mode must be 0600")
    if value.st_nlink != 1:
        raise SystemSecurityError("artifact must have exactly one hard link")


def _verify_executable_file(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise SystemSecurityError("controller is not a regular file")
    if value.st_uid != os.geteuid():
        raise SystemSecurityError("controller owner mismatch")
    if stat.S_IMODE(value.st_mode) != 0o500:
        raise SystemSecurityError("controller mode must be 0500")
    if value.st_nlink != 1:
        raise SystemSecurityError("controller must have exactly one hard link")


def _verify_source_file(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise SystemSecurityError("source is not a regular file")
    if value.st_uid != os.geteuid():
        raise SystemSecurityError("source owner mismatch")
    mode = stat.S_IMODE(value.st_mode)
    if mode & 0o077 or mode & 0o100 or not mode & 0o400:
        raise SystemSecurityError("source mode is not owner-only, non-executable, readable")
    if value.st_nlink != 1:
        raise SystemSecurityError("source must have exactly one hard link")


def _verify_source_directory(value: os.stat_result) -> None:
    if not stat.S_ISDIR(value.st_mode):
        raise SystemSecurityError("source path component is not a directory")
    if value.st_uid != os.geteuid():
        raise SystemSecurityError("source directory owner mismatch")
    if stat.S_IMODE(value.st_mode) & 0o022:
        raise SystemSecurityError("source directory is group/world writable")


def canonical_workspace_path(value: str | PurePosixPath) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        raise SystemSecurityError("source path must be relative text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SystemSecurityError("source path is not canonical relative POSIX")
    try:
        text.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise SystemSecurityError("source path must be ASCII") from exc
    if path == SYSTEM_ROOT or SYSTEM_ROOT in path.parents:
        raise SystemSecurityError("source path must be outside governed object storage")
    return path


def canonical_system_path(value: str | PurePosixPath) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        raise SystemSecurityError("system path must be relative text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SystemSecurityError("system path is not canonical relative POSIX")
    try:
        text.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise SystemSecurityError("system path must be ASCII") from exc
    if path != SYSTEM_ROOT and SYSTEM_ROOT not in path.parents:
        raise SystemSecurityError("system path is outside the governed root")
    return path


def _reject_reserved_authority_path(path: PurePosixPath) -> None:
    if path in _RESERVED_AUTHORITY_PATHS:
        raise SystemSecurityError("reserved System authority path requires a sealed operation")


def _is_governed_directory(parts: tuple[str, ...]) -> bool:
    path = PurePosixPath(*parts)
    return path == SYSTEM_ROOT or SYSTEM_ROOT in path.parents


class SecureSystemStorage:
    """Exact-byte storage used only through higher-level contract verification."""

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        max_read_bytes: int = MAX_CANONICAL_JSON_BYTES,
        max_directory_entries: int = 100_000,
    ) -> None:
        if (
            type(max_read_bytes) is not int
            or max_read_bytes <= 0
            or type(max_directory_entries) is not int
            or max_directory_entries <= 0
        ):
            raise SystemSecurityError("storage read bounds are invalid")
        try:
            root = Path(workspace_root).resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise SystemSecurityError("workspace root is unavailable") from exc
        try:
            root_stat = root.stat(follow_symlinks=False)
        except OSError as exc:
            raise SystemSecurityError("workspace root is unavailable") from exc
        _verify_directory(root_stat, governed=False)
        if root_stat.st_uid != os.geteuid():
            raise SystemSecurityError("workspace root owner mismatch")
        self.workspace_root = root
        self.max_read_bytes = max_read_bytes
        self.max_directory_entries = max_directory_entries

    def _open_workspace(self) -> int:
        descriptor: int | None = None
        try:
            descriptor = os.open("/", _DIRECTORY_FLAGS)
            for part in self.workspace_root.parts[1:]:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            workspace_stat = os.fstat(descriptor)
            _verify_directory(workspace_stat, governed=False)
            if workspace_stat.st_uid != os.geteuid():
                raise SystemSecurityError("workspace root owner mismatch")
            return descriptor
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise SystemSecurityError("workspace path is not securely openable") from exc
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise

    def _reject_casefold_alias(self, parent_fd: int, leaf: str) -> None:
        count = 0
        try:
            with os.scandir(parent_fd) as entries:
                for entry in entries:
                    count += 1
                    if count > self.max_directory_entries:
                        raise SystemSecurityError("directory collision check exceeded its bound")
                    if entry.name != leaf and entry.name.casefold() == leaf.casefold():
                        raise SystemSecurityError("casefold-colliding governed path")
        except SystemSecurityError:
            raise
        except OSError as exc:
            raise SystemSecurityError("governed directory cannot be enumerated") from exc

    def _open_directory(self, parts: tuple[str, ...], *, create: bool) -> int:  # noqa: C901
        descriptor = self._open_workspace()
        traversed: list[str] = []
        try:
            for part in parts:
                traversed.append(part)
                self._reject_casefold_alias(descriptor, part)
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise SystemNotFound("governed directory is absent") from None
                    child = -1
                    try:
                        created = False
                        try:
                            os.mkdir(part, mode=0o700, dir_fd=descriptor)
                            created = True
                        except FileExistsError:
                            # Another owner-equivalent writer may have won the
                            # same canonical directory creation race.
                            pass
                        child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                        if created and _is_governed_directory(tuple(traversed)):
                            os.fchmod(child, 0o700)
                    except OSError as exc:
                        if child >= 0:
                            os.close(child)
                        raise SystemSecurityError("governed directory cannot be created") from exc
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise SystemSecurityError(
                            "symlink or non-directory path component rejected"
                        ) from exc
                    raise SystemStorageError("governed directory cannot be opened") from exc
                _verify_directory(
                    os.fstat(child),
                    governed=_is_governed_directory(tuple(traversed)),
                )
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _open_source_directory(self, parts: tuple[str, ...]) -> int:
        descriptor = self._open_workspace()
        try:
            _verify_source_directory(os.fstat(descriptor))
            for part in parts:
                self._reject_casefold_alias(descriptor, part)
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except FileNotFoundError:
                    raise SystemNotFound("source directory is absent") from None
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise SystemSecurityError(
                            "source directory symlink/non-directory rejected"
                        ) from exc
                    raise SystemStorageError("source directory cannot be opened") from exc
                _verify_source_directory(os.fstat(child))
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _parent_leaf(
        self, value: str | PurePosixPath, *, create: bool
    ) -> tuple[int, str, PurePosixPath]:
        path = canonical_system_path(value)
        parent = self._open_directory(tuple(path.parts[:-1]), create=create)
        self._reject_casefold_alias(parent, path.name)
        return parent, path.name, path

    def _read_leaf(  # noqa: C901
        self,
        parent_fd: int,
        leaf: str,
        *,
        relative_path: PurePosixPath,
        optional: bool,
        executable: bool = False,
    ) -> StoredBytes | None:
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
            except FileNotFoundError:
                if optional:
                    return None
                raise SystemNotFound("governed artifact is absent") from None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise SystemSecurityError("artifact symlink rejected") from exc
                raise SystemStorageError("artifact cannot be opened") from exc
            verifier = _verify_executable_file if executable else _verify_file
            before = os.fstat(descriptor)
            verifier(before)
            if before.st_size <= 0 or before.st_size > self.max_read_bytes:
                raise SystemSecurityError("artifact size is outside the read bound")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            verifier(after)
            try:
                path_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise SystemSecurityError("artifact path changed during exact read") from exc
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(after) != _stat_identity(path_after)
                or len(raw) != after.st_size
            ):
                raise SystemSecurityError("artifact changed during exact read")
            return StoredBytes(str(relative_path), raw, _sha256(raw))
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def read(self, value: str | PurePosixPath) -> StoredBytes:
        parent, leaf, path = self._parent_leaf(value, create=False)
        try:
            result = self._read_leaf(parent, leaf, relative_path=path, optional=False)
            if result is None:  # pragma: no cover - optional=False is exhaustive
                raise SystemNotFound("governed artifact is absent")
            return result
        finally:
            os.close(parent)

    def read_optional(self, value: str | PurePosixPath) -> StoredBytes | None:
        try:
            parent, leaf, path = self._parent_leaf(value, create=False)
        except SystemNotFound:
            return None
        try:
            return self._read_leaf(parent, leaf, relative_path=path, optional=True)
        finally:
            os.close(parent)

    def read_executable(self, value: str | PurePosixPath) -> StoredBytes:
        parent, leaf, path = self._parent_leaf(value, create=False)
        try:
            result = self._read_leaf(
                parent,
                leaf,
                relative_path=path,
                optional=False,
                executable=True,
            )
            if result is None:  # pragma: no cover - optional=False is exhaustive
                raise SystemNotFound("governed executable is absent")
            return result
        finally:
            os.close(parent)

    def read_executable_optional(self, value: str | PurePosixPath) -> StoredBytes | None:
        try:
            parent, leaf, path = self._parent_leaf(value, create=False)
        except SystemNotFound:
            return None
        try:
            return self._read_leaf(
                parent,
                leaf,
                relative_path=path,
                optional=True,
                executable=True,
            )
        finally:
            os.close(parent)

    @staticmethod
    def _write_all(descriptor: int, raw: bytes) -> None:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SystemStorageError("short governed write")
            view = view[written:]

    def _write_temporary_file(
        self,
        parent_fd: int,
        leaf: str,
        raw: bytes,
        *,
        mode: int = 0o600,
    ) -> int:
        if mode not in {0o500, 0o600}:
            raise SystemSecurityError("governed file mode is invalid")
        descriptor = os.open(leaf, _CREATE_FLAGS, 0o600, dir_fd=parent_fd)
        try:
            os.fchmod(descriptor, mode)
            self._write_all(descriptor, raw)
            os.fsync(descriptor)
            verifier = _verify_executable_file if mode == 0o500 else _verify_file
            verifier(os.fstat(descriptor))
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _hash_leaf_stream(  # noqa: C901
        self,
        parent_fd: int,
        leaf: str,
        *,
        relative_path: PurePosixPath,
        source_policy: bool,
        maximum_bytes: int,
    ) -> StoredFile:
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
            except FileNotFoundError:
                raise SystemNotFound("streamed file is absent") from None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise SystemSecurityError("streamed file symlink rejected") from exc
                raise SystemStorageError("streamed file cannot be opened") from exc
            before = os.fstat(descriptor)
            if source_policy:
                _verify_source_file(before)
            else:
                _verify_file(before)
            if before.st_size <= 0 or before.st_size > maximum_bytes:
                raise SystemSecurityError("streamed file size is outside its bound")
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(descriptor)
            if source_policy:
                _verify_source_file(after)
            else:
                _verify_file(after)
            try:
                path_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise SystemSecurityError("streamed file path changed during read") from exc
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(after) != _stat_identity(path_after)
                or size != after.st_size
            ):
                raise SystemSecurityError("streamed file changed during exact read")
            return StoredFile(
                str(relative_path),
                digest.hexdigest(),
                size,
                source_stat_identity(after),
            )
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def hash_workspace_file(
        self,
        source_relative_path: str | PurePosixPath,
        *,
        maximum_bytes: int = 1024 * 1024 * 1024 * 1024,
    ) -> StoredFile:
        """Securely stream/hash one immutable file without copying its bytes."""

        source_path = canonical_workspace_path(source_relative_path)
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise SystemSecurityError("streamed source byte bound is invalid")
        source_parent = self._open_source_directory(tuple(source_path.parts[:-1]))
        try:
            self._reject_casefold_alias(source_parent, source_path.name)
            return self._hash_leaf_stream(
                source_parent,
                source_path.name,
                relative_path=source_path,
                source_policy=True,
                maximum_bytes=maximum_bytes,
            )
        finally:
            os.close(source_parent)

    def stat_workspace_file(
        self,
        source_relative_path: str | PurePosixPath,
        *,
        maximum_bytes: int = 1024 * 1024 * 1024 * 1024,
    ) -> StoredFile:
        """Securely stat one immutable source without reading its content."""

        source_path = canonical_workspace_path(source_relative_path)
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise SystemSecurityError("streamed source byte bound is invalid")
        parent = self._open_source_directory(tuple(source_path.parts[:-1]))
        descriptor: int | None = None
        try:
            self._reject_casefold_alias(parent, source_path.name)
            try:
                descriptor = os.open(source_path.name, _READ_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                raise SystemNotFound("streamed file is absent") from None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise SystemSecurityError("streamed file symlink rejected") from exc
                raise SystemStorageError("streamed file cannot be opened") from exc
            before = os.fstat(descriptor)
            _verify_source_file(before)
            if before.st_size <= 0 or before.st_size > maximum_bytes:
                raise SystemSecurityError("streamed file size is outside its bound")
            after = os.fstat(descriptor)
            path_after = os.stat(source_path.name, dir_fd=parent, follow_symlinks=False)
            if _stat_identity(before) != _stat_identity(after) or _stat_identity(
                after
            ) != _stat_identity(path_after):
                raise SystemSecurityError("streamed file changed during exact stat")
            return StoredFile(
                str(source_path),
                "",
                after.st_size,
                source_stat_identity(after),
            )
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def read_workspace_file_bytes(
        self,
        source_relative_path: str | PurePosixPath,
        *,
        maximum_bytes: int,
    ) -> StoredSourceBytes:
        """Securely read bounded source bytes with before/after identity checks."""

        source_path = canonical_workspace_path(source_relative_path)
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise SystemSecurityError("source byte read bound is invalid")
        parent = self._open_source_directory(tuple(source_path.parts[:-1]))
        descriptor: int | None = None
        try:
            self._reject_casefold_alias(parent, source_path.name)
            descriptor = os.open(source_path.name, _READ_FLAGS, dir_fd=parent)
            before = os.fstat(descriptor)
            _verify_source_file(before)
            if before.st_size <= 0 or before.st_size > maximum_bytes:
                raise SystemSecurityError("source byte read size is outside its bound")
            chunks: list[bytes] = []
            remaining = before.st_size
            digest = hashlib.sha256()
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            _verify_source_file(after)
            path_after = os.stat(source_path.name, dir_fd=parent, follow_symlinks=False)
            raw = b"".join(chunks)
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(after) != _stat_identity(path_after)
                or len(raw) != after.st_size
            ):
                raise SystemSecurityError("source changed during exact byte read")
            return StoredSourceBytes(
                str(source_path),
                raw,
                digest.hexdigest(),
                source_stat_identity(after),
            )
        except SystemError:
            raise
        except OSError as exc:
            raise SystemStorageError("source bytes cannot be read") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    @contextmanager
    def open_workspace_file(
        self,
        source_relative_path: str | PurePosixPath,
        *,
        maximum_bytes: int,
    ) -> Iterator[tuple[int, Mapping[str, int]]]:
        """Yield a secure read-only descriptor and verify its identity on exit."""

        source_path = canonical_workspace_path(source_relative_path)
        parent = self._open_source_directory(tuple(source_path.parts[:-1]))
        descriptor: int | None = None
        before: os.stat_result | None = None
        try:
            self._reject_casefold_alias(parent, source_path.name)
            descriptor = os.open(source_path.name, _READ_FLAGS, dir_fd=parent)
            before = os.fstat(descriptor)
            _verify_source_file(before)
            if before.st_size <= 0 or before.st_size > maximum_bytes:
                raise SystemSecurityError("source descriptor size is outside its bound")
            yield descriptor, source_stat_identity(before)
            after = os.fstat(descriptor)
            _verify_source_file(after)
            path_after = os.stat(source_path.name, dir_fd=parent, follow_symlinks=False)
            if _stat_identity(before) != _stat_identity(after) or _stat_identity(
                after
            ) != _stat_identity(path_after):
                raise SystemSecurityError("source changed while descriptor was in use")
        except SystemError:
            raise
        except OSError as exc:
            raise SystemStorageError("source descriptor cannot be opened") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def ensure_directory(self, value: str | PurePosixPath) -> None:
        """Create and verify one governed owner-only directory hierarchy."""

        path = canonical_system_path(value)
        descriptor = self._open_directory(tuple(path.parts), create=True)
        try:
            _verify_directory(os.fstat(descriptor), governed=True)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextmanager
    def exclusive_lock(self, value: str | PurePosixPath) -> Iterator[None]:
        """Hold one current-UID, mode-0600 governed advisory lock."""

        parent, leaf, _ = self._parent_leaf(value, create=True)
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(leaf, _CREATE_FLAGS, 0o600, dir_fd=parent)
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                os.fsync(parent)
            except FileExistsError:
                descriptor = os.open(
                    leaf,
                    os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent,
                )
            _verify_file(os.fstat(descriptor))
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except SystemError:
            raise
        except OSError as exc:
            raise SystemStorageError("governed lock cannot be used") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def read_exact_directory(
        self,
        value: str | PurePosixPath,
        *,
        expected_names: frozenset[str],
    ) -> dict[str, StoredBytes]:
        """Read an immutable directory whose file-name set is exact."""

        path = canonical_system_path(value)
        if not expected_names or any(
            type(name) is not str or not name or "/" in name or name in {".", ".."}
            for name in expected_names
        ):
            raise SystemSecurityError("immutable directory file set is invalid")
        descriptor = self._open_directory(tuple(path.parts), create=False)
        try:
            names: list[str] = []
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > self.max_directory_entries:
                        raise SystemSecurityError("immutable directory entry bound exceeded")
            if set(names) != set(expected_names) or len(names) != len(expected_names):
                raise SystemImmutableConflict("immutable directory file set mismatch")
            result: dict[str, StoredBytes] = {}
            for name in sorted(expected_names):
                relative = path / name
                row = self._read_leaf(
                    descriptor,
                    name,
                    relative_path=relative,
                    optional=False,
                )
                if row is None:  # pragma: no cover - optional=False
                    raise SystemNotFound("immutable directory file is absent")
                result[name] = row
            return result
        finally:
            os.close(descriptor)

    def list_directory_names(
        self,
        value: str | PurePosixPath,
        *,
        directories_only: bool,
    ) -> tuple[str, ...]:
        """List one bounded governed directory after exact entry-type checks."""

        if type(directories_only) is not bool:
            raise SystemSecurityError("directory listing policy is invalid")
        path = canonical_system_path(value)
        descriptor = self._open_directory(tuple(path.parts), create=False)
        try:
            names: list[str] = []
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > self.max_directory_entries:
                        raise SystemSecurityError("governed directory entry bound exceeded")
                    metadata = entry.stat(follow_symlinks=False)
                    if directories_only:
                        _verify_directory(metadata, governed=True)
                    else:
                        _verify_file(metadata)
            return tuple(sorted(names))
        except SystemError:
            raise
        except OSError as exc:
            raise SystemStorageError("governed directory cannot be listed") from exc
        finally:
            os.close(descriptor)

    def write_atomic_directory(  # noqa: C901
        self,
        value: str | PurePosixPath,
        files: Mapping[str, bytes],
    ) -> dict[str, StoredBytes]:
        """Publish an exact immutable multi-file directory with no replacement."""

        path = canonical_system_path(value)
        _reject_reserved_authority_path(path)
        normalized = dict(files)
        names = frozenset(normalized)
        if not names or any(type(raw) is not bytes or not raw for raw in normalized.values()):
            raise SystemSecurityError("immutable directory bytes are invalid")
        parent_path = PurePosixPath(*path.parts[:-1])
        parent = self._open_directory(tuple(parent_path.parts), create=True)
        staging = f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        staging_fd: int | None = None
        created = False
        try:
            self._reject_casefold_alias(parent, path.name)
            try:
                existing = self.read_exact_directory(path, expected_names=names)
                if any(existing[name].data != normalized[name] for name in names):
                    raise SystemImmutableConflict("immutable directory content conflict")
                return existing
            except SystemNotFound:
                pass
            os.mkdir(staging, mode=0o700, dir_fd=parent)
            created = True
            staging_fd = os.open(staging, _DIRECTORY_FLAGS, dir_fd=parent)
            os.fchmod(staging_fd, 0o700)
            _verify_directory(os.fstat(staging_fd), governed=True)
            for name in sorted(names):
                if "/" in name or name in {".", ".."}:
                    raise SystemSecurityError("immutable directory filename is invalid")
                file_fd = self._write_temporary_file(staging_fd, name, normalized[name])
                os.close(file_fd)
                row = self._read_leaf(
                    staging_fd,
                    name,
                    relative_path=path / name,
                    optional=False,
                )
                if row is None or row.data != normalized[name]:
                    raise SystemStorageError("staging directory readback mismatch")
            os.fsync(staging_fd)
            os.close(staging_fd)
            staging_fd = None
            try:
                os.rename(
                    staging,
                    path.name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                )
                created = False
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise SystemStorageError("immutable directory cannot be published") from exc
            os.fsync(parent)
            result = self.read_exact_directory(path, expected_names=names)
            if any(result[name].data != normalized[name] for name in names):
                raise SystemImmutableConflict("immutable directory content conflict")
            return result
        finally:
            if staging_fd is not None:
                os.close(staging_fd)
            if created:
                try:
                    cleanup = os.open(staging, _DIRECTORY_FLAGS, dir_fd=parent)
                    try:
                        with os.scandir(cleanup) as entries:
                            cleanup_names = [entry.name for entry in entries]
                        for name in cleanup_names:
                            os.unlink(name, dir_fd=cleanup)
                    finally:
                        os.close(cleanup)
                    os.rmdir(staging, dir_fd=parent)
                except FileNotFoundError:
                    pass
            os.close(parent)

    def write_exact_once(self, value: str | PurePosixPath, raw: bytes) -> StoredBytes:
        return self._write_exact_once(value, raw, allow_reserved_authority=False)

    def _write_exact_once(
        self,
        value: str | PurePosixPath,
        raw: bytes,
        *,
        allow_reserved_authority: bool,
    ) -> StoredBytes:
        if type(raw) is not bytes or not raw or len(raw) > self.max_read_bytes:
            raise SystemSecurityError("immutable artifact bytes are invalid")
        path = canonical_system_path(value)
        if not allow_reserved_authority:
            _reject_reserved_authority_path(path)
        parent, leaf, path = self._parent_leaf(path, create=True)
        temporary = f".{leaf}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        descriptor: int | None = None
        try:
            existing = self._read_leaf(parent, leaf, relative_path=path, optional=True)
            if existing is not None:
                if existing.data != raw:
                    raise SystemImmutableConflict("immutable artifact identity conflict")
                return existing
            descriptor = self._write_temporary_file(parent, temporary, raw)
            os.close(descriptor)
            descriptor = None
            try:
                os.link(
                    temporary,
                    leaf,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = self._read_leaf(parent, leaf, relative_path=path, optional=False)
                if existing is None or existing.data != raw:
                    raise SystemImmutableConflict("immutable artifact identity conflict") from None
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
            os.fsync(parent)
            readback = self._read_leaf(parent, leaf, relative_path=path, optional=False)
            if readback is None or readback.data != raw:
                raise SystemStorageError("immutable exact-byte readback mismatch")
            return readback
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    def write_executable_exact_once(
        self,
        value: str | PurePosixPath,
        raw: bytes,
    ) -> StoredBytes:
        """Write one immutable current-UID executable with exact mode ``0500``."""

        if type(raw) is not bytes or not raw or len(raw) > self.max_read_bytes:
            raise SystemSecurityError("immutable executable bytes are invalid")
        path = canonical_system_path(value)
        _reject_reserved_authority_path(path)
        parent, leaf, path = self._parent_leaf(path, create=True)
        temporary = f".{leaf}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        descriptor: int | None = None
        try:
            existing = self._read_leaf(
                parent,
                leaf,
                relative_path=path,
                optional=True,
                executable=True,
            )
            if existing is not None:
                if existing.data != raw:
                    raise SystemImmutableConflict("immutable executable identity conflict")
                return existing
            descriptor = self._write_temporary_file(parent, temporary, raw, mode=0o500)
            os.close(descriptor)
            descriptor = None
            try:
                os.link(
                    temporary,
                    leaf,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = self._read_leaf(
                    parent,
                    leaf,
                    relative_path=path,
                    optional=False,
                    executable=True,
                )
                if existing is None or existing.data != raw:
                    raise SystemImmutableConflict(
                        "immutable executable identity conflict"
                    ) from None
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
            os.fsync(parent)
            readback = self._read_leaf(
                parent,
                leaf,
                relative_path=path,
                optional=False,
                executable=True,
            )
            if readback is None or readback.data != raw:
                raise SystemStorageError("immutable executable exact-byte readback mismatch")
            return readback
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    def write_generation_manifest(self, generation_id: str, raw: bytes) -> StoredBytes:
        relative = GENERATIONS_ROOT / generation_id / "manifest.json"
        existing = self.read_optional(relative)
        if existing is not None:
            if existing.data != raw:
                raise SystemImmutableConflict("generation manifest identity conflict")
            return existing

        generations_fd = self._open_directory(tuple(GENERATIONS_ROOT.parts), create=True)
        staging = f".generation-{os.getpid()}-{secrets.token_hex(8)}"
        staging_fd: int | None = None
        manifest_fd: int | None = None
        staging_created = False
        try:
            self._reject_casefold_alias(generations_fd, generation_id)
            os.mkdir(staging, mode=0o700, dir_fd=generations_fd)
            staging_created = True
            staging_fd = os.open(staging, _DIRECTORY_FLAGS, dir_fd=generations_fd)
            os.fchmod(staging_fd, 0o700)
            _verify_directory(os.fstat(staging_fd), governed=True)
            manifest_fd = self._write_temporary_file(staging_fd, "manifest.json", raw)
            os.close(manifest_fd)
            manifest_fd = None
            os.fsync(staging_fd)
            os.close(staging_fd)
            staging_fd = None
            try:
                os.rename(
                    staging,
                    generation_id,
                    src_dir_fd=generations_fd,
                    dst_dir_fd=generations_fd,
                )
                staging_created = False
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise SystemStorageError("generation cannot be assembled") from exc
            os.fsync(generations_fd)
            readback = self.read(relative)
            if readback.data != raw:
                raise SystemImmutableConflict("generation manifest identity conflict")
            return readback
        finally:
            if manifest_fd is not None:
                os.close(manifest_fd)
            if staging_fd is not None:
                os.close(staging_fd)
            if staging_created:
                try:
                    cleanup_fd = os.open(staging, _DIRECTORY_FLAGS, dir_fd=generations_fd)
                    try:
                        try:
                            os.unlink("manifest.json", dir_fd=cleanup_fd)
                        except FileNotFoundError:
                            pass
                    finally:
                        os.close(cleanup_fd)
                    os.rmdir(staging, dir_fd=generations_fd)
                except FileNotFoundError:
                    pass
            os.close(generations_fd)

    def compare_and_swap_active(  # noqa: C901
        self, raw: bytes, *, expected_sha256: str
    ) -> StoredBytes:
        del raw, expected_sha256
        raise SystemSecurityError("active pointer requires System-owned activation authorization")

    def _compare_and_swap_active_authorized_nonempty(
        self, raw: bytes, *, expected_sha256: str
    ) -> StoredBytes:
        if expected_sha256 == EMPTY_POINTER_SHA256:
            raise SystemSecurityError("initial activation requires detached authorization")
        return self._compare_and_swap_pointer(
            raw,
            pointer_path=ACTIVE_POINTER_PATH,
            history_root=POINTER_HISTORY_ROOT,
            lock_path=SYSTEM_ROOT / ".active.lock",
            expected_sha256=expected_sha256,
            allow_reserved_authority=True,
        )

    def compare_and_swap_pointer(  # noqa: C901
        self,
        raw: bytes,
        *,
        pointer_path: str | PurePosixPath,
        history_root: str | PurePosixPath,
        lock_path: str | PurePosixPath,
        expected_sha256: str,
    ) -> StoredBytes:
        """CAS a non-reserved pointer while retaining its prior canonical bytes."""

        return self._compare_and_swap_pointer(
            raw,
            pointer_path=pointer_path,
            history_root=history_root,
            lock_path=lock_path,
            expected_sha256=expected_sha256,
            allow_reserved_authority=False,
        )

    def _compare_and_swap_pointer(  # noqa: C901
        self,
        raw: bytes,
        *,
        pointer_path: str | PurePosixPath,
        history_root: str | PurePosixPath,
        lock_path: str | PurePosixPath,
        expected_sha256: str,
        allow_reserved_authority: bool,
    ) -> StoredBytes:
        """CAS one exact pointer through an explicitly scoped authority lane."""

        if type(raw) is not bytes or not raw or len(raw) > self.max_read_bytes:
            raise SystemSecurityError("pointer bytes are invalid")
        pointer = canonical_system_path(pointer_path)
        if not allow_reserved_authority:
            _reject_reserved_authority_path(pointer)
        history = canonical_system_path(history_root)
        lock = canonical_system_path(lock_path)
        if pointer.parent != lock.parent:
            raise SystemSecurityError("pointer and lock directories differ")
        parent, leaf, path = self._parent_leaf(pointer, create=True)
        temporary = f".{leaf}.cas-{os.getpid()}-{secrets.token_hex(8)}"
        temporary_fd: int | None = None
        try:
            with self.exclusive_lock(lock):
                current = self._read_leaf(parent, leaf, relative_path=path, optional=True)
                observed = EMPTY_POINTER_SHA256 if current is None else current.byte_sha256
                if observed != expected_sha256:
                    raise SystemCASMismatch(expected_sha256, observed)
                if current is not None:
                    retained_path = history / f"{current.byte_sha256}.json"
                    retained = self.write_exact_once(retained_path, current.data)
                    if retained.byte_sha256 != current.byte_sha256:
                        raise SystemStorageError("previous pointer retention mismatch")

                temporary_fd = self._write_temporary_file(parent, temporary, raw)
                os.close(temporary_fd)
                temporary_fd = None
                os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
                readback = self._read_leaf(parent, leaf, relative_path=path, optional=False)
                if readback is None or readback.data != raw:
                    raise SystemStorageError("pointer exact-byte readback mismatch")
                return readback
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    def _commit_initial_activation(  # noqa: C901
        self,
        *,
        transaction: _PreparedInitialActivationWrite,
        lock_validator: Callable[[], None],
    ) -> dict[str, object]:
        """Persist one authorized first pointer and marker under the active lock.

        Exact pointer-only recovery publishes only the already-authorized marker.
        No state is ever rolled back to ``EMPTY``.
        """

        if type(transaction) is not _PreparedInitialActivationWrite:
            raise SystemSecurityError("initial activation requires a prepared transaction")
        pointer_raw = transaction.pointer_raw
        receipt_raw = transaction.receipt_raw
        final_authorization_raw = transaction.final_authorization_raw
        authorization_raw = transaction.activation_authorization_raw
        prepared_raw = transaction.prepared_raw
        marker_raw = transaction.marker_raw
        for label, raw in (
            ("pointer", pointer_raw),
            ("receipt", receipt_raw),
            ("final cutover authorization", final_authorization_raw),
            ("activation authorization", authorization_raw),
            ("prepared transaction", prepared_raw),
            ("marker", marker_raw),
        ):
            if type(raw) is not bytes or not raw or len(raw) > self.max_read_bytes:
                raise SystemSecurityError(f"initial activation {label} bytes are invalid")
        expected_pointer_sha = _sha256(pointer_raw)
        expected_marker_sha = _sha256(marker_raw)
        receipt_path = OBJECTS_ROOT / "system.migration.receipt" / f"{_sha256(receipt_raw)}.json"
        final_authorization_path = (
            OBJECTS_ROOT
            / "system.final_cutover_authorization"
            / f"{_sha256(final_authorization_raw)}.json"
        )
        authorization_path = (
            OBJECTS_ROOT / "system.activation_authorization" / f"{_sha256(authorization_raw)}.json"
        )
        authorization_index = ACTIVATION_AUTHORIZATIONS_ROOT / f"{expected_pointer_sha}.json"
        final_authorization_index = (
            FINAL_CUTOVER_AUTHORIZATIONS_ROOT / f"{expected_pointer_sha}.json"
        )
        prepared_path = (
            OBJECTS_ROOT / "system.activation_prepared" / f"{_sha256(prepared_raw)}.json"
        )
        prepared_index = ACTIVATION_TRANSACTIONS_ROOT / f"{expected_pointer_sha}.json"
        lock = SYSTEM_ROOT / ".active.lock"
        pointer_parent, pointer_leaf, pointer_path = self._parent_leaf(
            ACTIVE_POINTER_PATH, create=True
        )
        temporary = f".{pointer_leaf}.cas-{os.getpid()}-{secrets.token_hex(8)}"
        temporary_fd: int | None = None
        try:
            with self.exclusive_lock(lock):
                lock_validator()
                current = self._read_leaf(
                    pointer_parent,
                    pointer_leaf,
                    relative_path=pointer_path,
                    optional=True,
                )
                marker = self.read_optional(MIGRATION_MARKER_PATH)
                if current is None and marker is not None:
                    raise SystemImmutableConflict("migration marker exists without active pointer")
                if current is not None and current.data != pointer_raw:
                    raise SystemCASMismatch(EMPTY_POINTER_SHA256, current.byte_sha256)
                if marker is not None and marker.data != marker_raw:
                    raise SystemImmutableConflict("permanent migration marker conflicts")

                receipt = self.write_exact_once(receipt_path, receipt_raw)
                final_authorization = self.write_exact_once(
                    final_authorization_path, final_authorization_raw
                )
                final_authorization_indexed = self.write_exact_once(
                    final_authorization_index, final_authorization_raw
                )
                authorization = self.write_exact_once(authorization_path, authorization_raw)
                authorization_indexed = self.write_exact_once(
                    authorization_index, authorization_raw
                )
                prepared = self.write_exact_once(prepared_path, prepared_raw)
                prepared_indexed = self.write_exact_once(prepared_index, prepared_raw)
                cas_performed = False
                if current is None:
                    temporary_fd = self._write_temporary_file(
                        pointer_parent, temporary, pointer_raw
                    )
                    os.close(temporary_fd)
                    temporary_fd = None
                    os.replace(
                        temporary,
                        pointer_leaf,
                        src_dir_fd=pointer_parent,
                        dst_dir_fd=pointer_parent,
                    )
                    os.fsync(pointer_parent)
                    current = self._read_leaf(
                        pointer_parent,
                        pointer_leaf,
                        relative_path=pointer_path,
                        optional=False,
                    )
                    if current is None or current.data != pointer_raw:
                        raise SystemStorageError("initial pointer exact-byte readback mismatch")
                    cas_performed = True

                marker = self._write_exact_once(
                    MIGRATION_MARKER_PATH,
                    marker_raw,
                    allow_reserved_authority=True,
                )
                if (
                    current is None
                    or current.byte_sha256 != expected_pointer_sha
                    or marker.byte_sha256 != expected_marker_sha
                    or receipt.data != receipt_raw
                    or final_authorization.data != final_authorization_raw
                    or final_authorization_indexed.data != final_authorization_raw
                    or authorization.data != authorization_raw
                    or authorization_indexed.data != authorization_raw
                    or prepared.data != prepared_raw
                    or prepared_indexed.data != prepared_raw
                ):
                    raise SystemStorageError("initial activation exact-byte readback mismatch")
                return {
                    "authorization": authorization,
                    "authorization_index": authorization_indexed,
                    "cas_performed": cas_performed,
                    "final_authorization": final_authorization,
                    "final_authorization_index": final_authorization_indexed,
                    "marker": marker,
                    "pointer": current,
                    "prepared": prepared,
                    "prepared_index": prepared_indexed,
                    "receipt": receipt,
                }
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            try:
                os.unlink(temporary, dir_fd=pointer_parent)
            except FileNotFoundError:
                pass
            os.close(pointer_parent)


__all__ = [
    "ACTIVE_POINTER_PATH",
    "ACTIVATION_AUTHORIZATIONS_ROOT",
    "ACTIVATION_TRANSACTIONS_ROOT",
    "CANDIDATE_STATE_ROOT",
    "EMPTY_POINTER_SHA256",
    "FINAL_CUTOVER_AUTHORIZATIONS_ROOT",
    "GENERATIONS_ROOT",
    "MIGRATION_MARKER_PATH",
    "OBJECTS_ROOT",
    "POINTER_HISTORY_ROOT",
    "SOURCE_VERIFICATION_CACHE_ROOT",
    "SYSTEM_ROOT",
    "VALIDATION_CUSTODY_ROOT",
    "VALIDATION_RUNS_ROOT",
    "VALIDATION_REQUESTS_ROOT",
    "SecureSystemStorage",
    "StoredBytes",
    "StoredFile",
    "StoredSourceBytes",
    "canonical_system_path",
    "canonical_workspace_path",
    "source_stat_identity",
]
