"""Descriptor-relative durable storage for the four fixed protocol-v3 roots."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Final, Iterator

from quant_investor.v17_v3_contract.namespace import (
    NamespaceContractError,
    canonical_relative_path as contract_canonical_relative_path,
    root_for_path,
)

PRIVATE_SOURCES_ROOT: Final = PurePosixPath("data/private/v17_v3_sources")
PRIVATE_RUNS_ROOT: Final = PurePosixPath("data/private/v17_v3_runs")
SHADOW_RESULTS_ROOT: Final = PurePosixPath("results/v17_v3_shadow")
FORMAL_RESULTS_ROOT: Final = PurePosixPath("results/v17_v3_formal_research")
GOVERNED_ROOTS: Final = (
    PRIVATE_SOURCES_ROOT,
    PRIVATE_RUNS_ROOT,
    SHADOW_RESULTS_ROOT,
    FORMAL_RESULTS_ROOT,
)

EMPTY_SHA256: Final = "EMPTY"
DEFAULT_MAX_READ_BYTES: Final = 64 * 1024 * 1024
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


class StorageError(RuntimeError):
    """Base error for governed protocol-v3 storage."""


class StorageSecurityError(StorageError):
    """A path, inode, link, or permission check failed."""


class StorageNotFoundError(StorageError):
    """A governed artifact was not present."""


class ExactOnceConflictError(StorageError):
    """An immutable path already contains different bytes."""


class CASMismatchError(StorageError):
    """A mutable pointer did not match the caller's exact expected bytes."""

    def __init__(self, expected_sha256: str, observed_sha256: str) -> None:
        super().__init__("governed pointer CAS mismatch")
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256


class LockUnavailableError(StorageError):
    """A nonblocking governed lock is already held."""


@dataclass(frozen=True)
class StoredBytes:
    relative_path: str
    data: bytes
    byte_sha256: str


@dataclass(frozen=True)
class WriteResult:
    relative_path: str
    byte_sha256: str
    created: bool
    replaced: bool


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expected_sha(value: str) -> str:
    if value == EMPTY_SHA256:
        return value
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StorageSecurityError("expected SHA-256 is not canonical")
    return value


def canonical_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    """Validate one canonical path below exactly one fixed V3 root."""

    if not isinstance(value, (str, PurePosixPath)):
        raise StorageSecurityError("governed path must be text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise StorageSecurityError("governed path is not canonical POSIX")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise StorageSecurityError("governed paths must use ASCII") from exc
    try:
        contract_path = contract_canonical_relative_path(
            text,
            label="governed path",
        )
        root_for_path(str(contract_path))
    except NamespaceContractError as exc:
        raise StorageSecurityError("path violates V3 namespace isolation") from exc
    if not any(path == root or root in path.parents for root in GOVERNED_ROOTS):
        raise StorageSecurityError("path is outside the four protocol-v3 roots")
    return path


def _file_identity(st: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        st.st_dev,
        st.st_ino,
        st.st_mode,
        st.st_nlink,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )


def _verify_directory(st: os.stat_result, *, private: bool, label: str) -> None:
    if not stat.S_ISDIR(st.st_mode):
        raise StorageSecurityError(f"{label} is not a directory")
    if private and stat.S_IMODE(st.st_mode) != 0o700:
        raise StorageSecurityError(f"{label} directory mode must be 0700")


def _verify_regular_file(st: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise StorageSecurityError(f"{label} is not a regular file")
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise StorageSecurityError(f"{label} file mode must be 0600")
    if st.st_nlink != 1:
        raise StorageSecurityError(f"{label} file must have exactly one hard link")


def _is_governed(path: PurePosixPath) -> bool:
    return any(path == root or root in path.parents for root in GOVERNED_ROOTS)


@dataclass
class SecureStore:
    """Owner-private storage using descriptor-relative traversal and durable writes."""

    workspace_root: Path
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> None:
        root = Path(workspace_root)
        if not root.is_absolute():
            raise StorageSecurityError("workspace_root must be absolute")
        if any(part in {"", ".", ".."} for part in root.parts[1:]):
            raise StorageSecurityError("workspace_root must be a canonical absolute path")
        if type(max_read_bytes) is not int or max_read_bytes <= 0:
            raise StorageSecurityError("max_read_bytes must be positive")
        self.workspace_root = root
        self.max_read_bytes = max_read_bytes
        self._held_locks: ContextVar[frozenset[str]] = ContextVar(
            f"v17_v3_held_locks_{id(self)}",
            default=frozenset(),
        )

    def _open_workspace(self, *, require_private: bool) -> int:
        try:
            fd = os.open("/", _DIRECTORY_FLAGS)
            for part in self.workspace_root.parts[1:]:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = child
        except OSError as exc:
            try:
                os.close(fd)
            except (OSError, UnboundLocalError):
                pass
            raise StorageSecurityError("cannot open a physical workspace root") from exc
        try:
            _verify_directory(
                os.fstat(fd),
                private=require_private,
                label="workspace root",
            )
        except BaseException:
            os.close(fd)
            raise
        return fd

    @staticmethod
    def _reject_casefold_alias(parent_fd: int, leaf: str) -> None:
        try:
            names = os.listdir(parent_fd)
        except OSError as exc:
            raise StorageSecurityError("cannot enumerate governed parent") from exc
        aliases = [name for name in names if name.casefold() == leaf.casefold()]
        if any(name != leaf for name in aliases):
            raise StorageSecurityError("casefold-colliding governed path")

    def _open_directory(self, parts: tuple[str, ...], *, create: bool) -> int:
        fd = self._open_workspace(require_private=True)
        traversed: list[str] = []
        try:
            for part in parts:
                self._reject_casefold_alias(fd, part)
                traversed.append(part)
                current = PurePosixPath(*traversed)
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        raise StorageNotFoundError("governed directory is absent") from None
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                        os.chmod(part, 0o700, dir_fd=fd, follow_symlinks=False)
                        os.fsync(fd)
                    except FileExistsError:
                        pass
                    self._reject_casefold_alias(fd, part)
                    try:
                        child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                    except OSError as exc:
                        raise StorageSecurityError(
                            "cannot open a newly created governed directory"
                        ) from exc
                except OSError as exc:
                    raise StorageSecurityError("cannot open governed directory") from exc
                try:
                    _verify_directory(
                        os.fstat(child),
                        private=_is_governed(current),
                        label=str(current),
                    )
                except BaseException:
                    os.close(child)
                    raise
                os.close(fd)
                fd = child
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _open_parent(
        self,
        relative_path: str | PurePosixPath,
        *,
        create: bool,
    ) -> tuple[PurePosixPath, int, str]:
        path = canonical_relative_path(relative_path)
        if len(path.parts) < 2:
            raise StorageSecurityError("governed file must have a parent")
        parent_fd = self._open_directory(path.parent.parts, create=create)
        self._reject_casefold_alias(parent_fd, path.name)
        return path, parent_fd, path.name

    def initialize(self) -> None:
        """Create only the four frozen roots with mode 0700."""

        for root in GOVERNED_ROOTS:
            fd = self._open_directory(root.parts, create=True)
            os.close(fd)

    def _bounded_read_fd(self, fd: int, *, label: str) -> bytes:
        before = os.fstat(fd)
        _verify_regular_file(before, label=label)
        if before.st_size > self.max_read_bytes:
            raise StorageSecurityError("governed file exceeds the bounded read limit")
        chunks: list[bytes] = []
        remaining = self.max_read_bytes + 1
        while remaining:
            chunk = os.read(fd, min(131072, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if len(raw) > self.max_read_bytes:
            raise StorageSecurityError("governed file exceeds the bounded read limit")
        if _file_identity(before) != _file_identity(after) or len(raw) != after.st_size:
            raise StorageSecurityError("governed file changed during its stable read")
        return raw

    def read_optional(self, relative_path: str | PurePosixPath) -> StoredBytes | None:
        try:
            path, parent_fd, leaf = self._open_parent(relative_path, create=False)
        except StorageNotFoundError:
            return None
        try:
            try:
                fd = os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise StorageSecurityError("cannot open governed file") from exc
            try:
                raw = self._bounded_read_fd(fd, label=str(path))
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)
        return StoredBytes(str(path), raw, _sha256(raw))

    def read(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str | None = None,
    ) -> bytes:
        observed = self.read_optional(relative_path)
        if observed is None:
            raise StorageNotFoundError("governed file does not exist")
        if expected_sha256 is not None:
            expected = _expected_sha(expected_sha256)
            if expected == EMPTY_SHA256 or observed.byte_sha256 != expected:
                raise CASMismatchError(expected, observed.byte_sha256)
        return observed.data

    def relative_from_path(self, path: str | os.PathLike[str]) -> PurePosixPath:
        """Convert an absolute governed path without resolving symlinks."""

        target = Path(path)
        if not target.is_absolute():
            return canonical_relative_path(PurePosixPath(str(target)))
        try:
            relative = target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise StorageSecurityError("absolute path is outside workspace_root") from exc
        return canonical_relative_path(PurePosixPath(*relative.parts))

    def read_path(
        self,
        path: str | os.PathLike[str],
        expected_sha256: str | None = None,
    ) -> bytes:
        return self.read(self.relative_from_path(path), expected_sha256)

    @staticmethod
    def _write_all(fd: int, raw: bytes) -> None:
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError(errno.EIO, "write made no progress")
            written += count

    def _write_temp(self, parent_fd: int, leaf: str, raw: bytes) -> str:
        temp_leaf = f".{leaf}.tmp-{secrets.token_hex(16)}"
        fd: int | None = None
        try:
            fd = os.open(temp_leaf, _CREATE_FLAGS, 0o600, dir_fd=parent_fd)
            os.fchmod(fd, 0o600)
            _verify_regular_file(os.fstat(fd), label="temporary governed file")
            self._write_all(fd, raw)
            os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            if self._bounded_read_fd(fd, label="temporary governed file") != raw:
                raise StorageError("temporary file readback mismatch")
            return temp_leaf
        except BaseException:
            if temp_leaf:
                try:
                    os.unlink(temp_leaf, dir_fd=parent_fd)
                except OSError:
                    pass
            raise
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def _unlink_temp(parent_fd: int, temp_leaf: str | None) -> None:
        if temp_leaf is None:
            return
        try:
            os.unlink(temp_leaf, dir_fd=parent_fd)
        except FileNotFoundError:
            return

    def write_exact_once(
        self,
        relative_path: str | PurePosixPath,
        raw: bytes,
    ) -> WriteResult:
        """Atomically create immutable bytes, or accept byte-identical replay."""

        if type(raw) is not bytes:
            raise StorageSecurityError("governed payload must be bytes")
        path, parent_fd, leaf = self._open_parent(relative_path, create=True)
        temp_leaf: str | None = None
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            self._reject_casefold_alias(parent_fd, leaf)
            observed = self.read_optional(path)
            if observed is not None:
                if observed.data != raw:
                    raise ExactOnceConflictError("immutable path already contains different bytes")
                return WriteResult(str(path), observed.byte_sha256, False, False)
            temp_leaf = self._write_temp(parent_fd, leaf, raw)
            self._reject_casefold_alias(parent_fd, leaf)
            try:
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                observed = self.read_optional(path)
                if observed is not None and observed.data == raw:
                    return WriteResult(str(path), observed.byte_sha256, False, False)
                raise ExactOnceConflictError("immutable path appeared before atomic publication")
            os.replace(temp_leaf, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temp_leaf = None
            os.fsync(parent_fd)
            readback = self.read(path)
            if readback != raw:
                raise StorageError("immutable exact-byte readback mismatch")
            return WriteResult(str(path), _sha256(raw), True, False)
        finally:
            self._unlink_temp(parent_fd, temp_leaf)
            os.close(parent_fd)

    def replace_cas(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str,
        raw: bytes,
    ) -> WriteResult:
        """Atomically replace a pointer after locked exact-byte CAS."""

        if type(raw) is not bytes:
            raise StorageSecurityError("governed payload must be bytes")
        expected = _expected_sha(expected_sha256)
        path, parent_fd, leaf = self._open_parent(relative_path, create=True)
        temp_leaf: str | None = None
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            observed = self.read_optional(path)
            observed_sha = EMPTY_SHA256 if observed is None else observed.byte_sha256
            if observed is not None and observed.data == raw:
                return WriteResult(str(path), observed.byte_sha256, False, False)
            if observed_sha != expected:
                raise CASMismatchError(expected, observed_sha)
            temp_leaf = self._write_temp(parent_fd, leaf, raw)
            current = self.read_optional(path)
            current_sha = EMPTY_SHA256 if current is None else current.byte_sha256
            if current_sha != expected:
                raise CASMismatchError(expected, current_sha)
            os.replace(temp_leaf, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temp_leaf = None
            os.fsync(parent_fd)
            if self.read(path) != raw:
                raise StorageError("pointer exact-byte readback mismatch")
            return WriteResult(
                str(path),
                _sha256(raw),
                expected == EMPTY_SHA256,
                expected != EMPTY_SHA256,
            )
        finally:
            self._unlink_temp(parent_fd, temp_leaf)
            os.close(parent_fd)

    def remove_cas(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str,
    ) -> StoredBytes:
        """Durably remove one mutable pointer after exact-byte CAS."""

        expected = _expected_sha(expected_sha256)
        if expected == EMPTY_SHA256:
            raise StorageSecurityError("remove CAS requires existing exact bytes")
        path, parent_fd, leaf = self._open_parent(relative_path, create=False)
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            observed = self.read_optional(path)
            if observed is None:
                raise CASMismatchError(expected, EMPTY_SHA256)
            if observed.byte_sha256 != expected:
                raise CASMismatchError(expected, observed.byte_sha256)
            try:
                os.unlink(leaf, dir_fd=parent_fd)
            except OSError as exc:
                raise StorageSecurityError("cannot remove governed pointer") from exc
            os.fsync(parent_fd)
            return observed
        finally:
            os.close(parent_fd)

    def _ensure_lock_file(self, relative_path: str | PurePosixPath) -> None:
        self.write_exact_once(relative_path, b"")

    @contextmanager
    def locked(
        self,
        relative_path: str | PurePosixPath,
        *,
        blocking: bool = True,
    ) -> Iterator[None]:
        """Hold one permanent regular-file flock without symlink traversal."""

        path = canonical_relative_path(relative_path)
        key = str(path)
        held = self._held_locks.get()
        if key in held:
            raise StorageSecurityError("recursive governed lock acquisition")
        self._ensure_lock_file(path)
        _, parent_fd, leaf = self._open_parent(path, create=False)
        fd: int | None = None
        token = None
        try:
            fd = os.open(leaf, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            _verify_regular_file(os.fstat(fd), label=key)
            operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(fd, operation)
            except BlockingIOError as exc:
                raise LockUnavailableError("governed lock is unavailable") from exc
            token = self._held_locks.set(held | {key})
            yield
        finally:
            if token is not None:
                self._held_locks.reset(token)
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            os.close(parent_fd)


__all__ = [
    "CASMismatchError",
    "DEFAULT_MAX_READ_BYTES",
    "EMPTY_SHA256",
    "ExactOnceConflictError",
    "FORMAL_RESULTS_ROOT",
    "GOVERNED_ROOTS",
    "LockUnavailableError",
    "PRIVATE_RUNS_ROOT",
    "PRIVATE_SOURCES_ROOT",
    "SHADOW_RESULTS_ROOT",
    "SecureStore",
    "StorageError",
    "StorageNotFoundError",
    "StorageSecurityError",
    "StoredBytes",
    "WriteResult",
    "canonical_relative_path",
]
