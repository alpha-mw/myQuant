"""Descriptor-relative durable storage for the protocol-v2 fixed roots."""

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

from quant_investor.v17_v2_contract.identities import (
    IdentityContractError,
    require_sha256,
)
from quant_investor.v17_v2_contract.limits import LIMITS

from .gate import RESULTS_ROOT, SOURCES_ROOT, RuntimeGate

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
EMPTY_SHA: Final = "EMPTY"
LATEST_PATH: Final = PurePosixPath("results/v17_shadow/protocol-v2/_latest/shadow.json")
LATEST_LOCK_PATH: Final = PurePosixPath("results/v17_shadow/protocol-v2/_latest/.latest.lock")


class StorageError(RuntimeError):
    """Base class for fail-closed storage errors."""


class StorageSecurityError(StorageError):
    """Raised for symlink, type, permission, or link-count violations."""


class StorageNotFoundError(StorageError):
    """Raised when a requested governed file does not exist."""


class ExactOnceConflictError(StorageError):
    """Raised when an immutable path already contains different bytes."""


class CASMismatchError(StorageError):
    """Raised before payload writes when observed bytes do not match CAS."""

    def __init__(self, expected_sha256: str, observed_sha256: str) -> None:
        super().__init__(f"CAS mismatch: expected {expected_sha256}, observed {observed_sha256}")
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256


class LockUnavailableError(StorageError):
    """Raised when a nonblocking protocol lock is already held."""


class StorageCommitError(StorageError):
    """Raised when a write fails before or after its durability boundary."""

    def __init__(self, detail: str, *, possibly_committed: bool) -> None:
        super().__init__(detail)
        self.possibly_committed = possibly_committed


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


def _canonical_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        raise StorageSecurityError("relative_path must be text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise StorageSecurityError("relative_path is not canonical POSIX")
    if (
        path != RESULTS_ROOT
        and RESULTS_ROOT not in path.parents
        and path != SOURCES_ROOT
        and SOURCES_ROOT not in path.parents
    ):
        raise StorageSecurityError("path is outside the two protocol-v2 roots")
    return path


def _expected_sha(value: str) -> str:
    if value == EMPTY_SHA:
        return value
    try:
        return require_sha256(value, label="expected_sha256")
    except IdentityContractError as exc:
        raise StorageSecurityError(str(exc)) from exc


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


def _verify_directory(st: os.stat_result, *, governed: bool, label: str) -> None:
    if not stat.S_ISDIR(st.st_mode):
        raise StorageSecurityError(f"{label} is not a directory")
    if governed and stat.S_IMODE(st.st_mode) != 0o700:
        raise StorageSecurityError(f"{label} directory mode must be 0700")


def _verify_regular_file(st: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise StorageSecurityError(f"{label} is not a regular file")
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise StorageSecurityError(f"{label} file mode must be 0600")
    if st.st_nlink != 1:
        raise StorageSecurityError(f"{label} file must have exactly one link")


@dataclass
class SecureStore:
    """Safe storage rooted at one physical workspace directory."""

    workspace_root: Path
    max_read_bytes: int = LIMITS["general_json_bytes"]

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        max_read_bytes: int = LIMITS["general_json_bytes"],
    ) -> None:
        root = Path(workspace_root)
        if not root.is_absolute():
            raise StorageSecurityError("workspace_root must be absolute")
        if type(max_read_bytes) is not int or max_read_bytes <= 0:
            raise StorageSecurityError("max_read_bytes must be a positive integer")
        self.workspace_root = root
        self.max_read_bytes = max_read_bytes
        self._held_locks: ContextVar[frozenset[str]] = ContextVar(
            f"v17_v2_held_locks_{id(self)}",
            default=frozenset(),
        )

    def initialize(self) -> None:
        """Create only the two fixed roots and the permanent latest lock."""

        decision = RuntimeGate(self.workspace_root).classify(
            "SHADOW_PREPARE",
            "runtime-init",
            version="ABSENT",
            state="MISSING",
            checkpoint="PRE_IMPORT",
        )
        if not decision.allowed:
            raise StorageSecurityError(f"runtime gate rejected initialization: {decision.detail}")
        for root in (RESULTS_ROOT, SOURCES_ROOT):
            fd = self._open_directory(root.parts, create=True)
            os.close(fd)
        latest_parent = LATEST_LOCK_PATH.parent
        fd = self._open_directory(latest_parent.parts, create=True)
        os.close(fd)
        self._ensure_lock_file(LATEST_LOCK_PATH)

    def _open_workspace(self) -> int:
        try:
            fd = os.open(self.workspace_root, _DIRECTORY_FLAGS)
        except OSError as exc:
            raise StorageSecurityError("cannot open physical workspace root") from exc
        try:
            _verify_directory(os.fstat(fd), governed=False, label="workspace root")
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _open_directory(self, parts: tuple[str, ...], *, create: bool) -> int:
        fd = self._open_workspace()
        governed = False
        traversed: list[str] = []
        try:
            for part in parts:
                traversed.append(part)
                current = PurePosixPath(*traversed)
                governed = governed or current in {RESULTS_ROOT, SOURCES_ROOT}
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        raise StorageNotFoundError(f"directory does not exist: {current}") from None
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                        os.chmod(part, 0o700, dir_fd=fd, follow_symlinks=False)
                        os.fsync(fd)
                    except FileExistsError:
                        pass
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                except OSError as exc:
                    raise StorageSecurityError(
                        f"cannot open directory component: {current}"
                    ) from exc
                try:
                    _verify_directory(
                        os.fstat(child),
                        governed=governed,
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
        path = _canonical_relative_path(relative_path)
        if len(path.parts) <= 1:
            raise StorageSecurityError("governed file must have a parent")
        return path, self._open_directory(path.parent.parts, create=create), path.name

    def _bounded_read_fd(self, fd: int, *, label: str) -> bytes:
        before = os.fstat(fd)
        _verify_regular_file(before, label=label)
        if before.st_size > self.max_read_bytes:
            raise StorageSecurityError(f"{label} exceeds bounded read limit")
        chunks: list[bytes] = []
        remaining = self.max_read_bytes + 1
        while remaining:
            chunk = os.read(fd, min(131072, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        if len(data) > self.max_read_bytes:
            raise StorageSecurityError(f"{label} exceeds bounded read limit")
        if _file_identity(before) != _file_identity(after) or len(data) != after.st_size:
            raise StorageSecurityError(f"{label} changed during stable read")
        return data

    def _read_optional(self, relative_path: str | PurePosixPath) -> StoredBytes | None:
        path, parent_fd, leaf = self._open_parent(relative_path, create=False)
        try:
            try:
                fd = os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise StorageSecurityError(f"cannot open governed file: {path}") from exc
            try:
                data = self._bounded_read_fd(fd, label=str(path))
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)
        return StoredBytes(str(path), data, hashlib.sha256(data).hexdigest())

    def read(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str | None = None,
    ) -> bytes:
        """Return one bounded stable read, optionally bound to exact bytes."""

        observed = self._read_optional(relative_path)
        if observed is None:
            raise StorageNotFoundError(f"governed file does not exist: {relative_path}")
        if expected_sha256 is not None:
            expected = _expected_sha(expected_sha256)
            if expected == EMPTY_SHA or observed.byte_sha256 != expected:
                raise CASMismatchError(expected, observed.byte_sha256)
        return observed.data

    def _write_all(self, fd: int, data: bytes) -> None:
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError(errno.EIO, "write made no progress")
            written += count

    def _create_exact(self, path: PurePosixPath, data: bytes) -> WriteResult:
        _, parent_fd, leaf = self._open_parent(path, create=True)
        fd: int | None = None
        inode: tuple[int, int] | None = None
        file_durable = False
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            try:
                fd = os.open(leaf, _CREATE_FLAGS, 0o600, dir_fd=parent_fd)
            except FileExistsError:
                observed = self._read_optional(path)
                if observed is not None and observed.data == data:
                    return WriteResult(str(path), observed.byte_sha256, False, False)
                raise ExactOnceConflictError(
                    f"immutable path already contains different bytes: {path}"
                ) from None
            os.fchmod(fd, 0o600)
            st = os.fstat(fd)
            inode = (st.st_dev, st.st_ino)
            _verify_regular_file(st, label=str(path))
            self._write_all(fd, data)
            os.fsync(fd)
            file_durable = True
            os.lseek(fd, 0, os.SEEK_SET)
            if self._bounded_read_fd(fd, label=str(path)) != data:
                raise StorageCommitError(
                    f"exact-byte readback mismatch: {path}",
                    possibly_committed=True,
                )
            os.fsync(parent_fd)
            return WriteResult(str(path), hashlib.sha256(data).hexdigest(), True, False)
        except (ExactOnceConflictError, StorageCommitError):
            raise
        except BaseException as exc:
            if fd is not None and not file_durable and inode is not None:
                try:
                    current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                    if (
                        (current.st_dev, current.st_ino) == inode
                        and stat.S_ISREG(current.st_mode)
                        and current.st_nlink == 1
                    ):
                        os.unlink(leaf, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                except OSError:
                    pass
            raise StorageCommitError(
                f"exact-once write failed for {path}: {exc}",
                possibly_committed=file_durable,
            ) from exc
        finally:
            if fd is not None:
                os.close(fd)
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
            os.close(parent_fd)

    def write_exact_once(
        self,
        relative_path: str | PurePosixPath,
        data: bytes,
    ) -> WriteResult:
        """Create immutable bytes once; identical retries are idempotent."""

        path = _canonical_relative_path(relative_path)
        if type(data) is not bytes:
            raise StorageSecurityError("write payload must be bytes")
        if len(data) > self.max_read_bytes:
            raise StorageSecurityError("write payload exceeds bounded read limit")
        return self._create_exact(path, data)

    def _derived_lock(self, path: PurePosixPath) -> PurePosixPath:
        if path == LATEST_PATH:
            return LATEST_LOCK_PATH
        parts = path.parts
        run_prefix = RESULTS_ROOT.parts + ("runs",)
        if parts[: len(run_prefix)] == run_prefix and len(parts) >= len(run_prefix) + 2:
            run_id = parts[len(run_prefix)]
            return PurePosixPath(*run_prefix, run_id, ".ledger.lock")
        raise StorageSecurityError("CAS replacement is allowed only for run or latest state")

    def _replace_cas_locked(
        self,
        path: PurePosixPath,
        expected_sha256: str,
        data: bytes,
    ) -> WriteResult:
        observed = self._read_optional(path)
        observed_sha = EMPTY_SHA if observed is None else observed.byte_sha256
        if observed is not None and observed.data == data:
            return WriteResult(str(path), observed.byte_sha256, False, False)
        if observed_sha != expected_sha256:
            raise CASMismatchError(expected_sha256, observed_sha)

        _, parent_fd, leaf = self._open_parent(path, create=True)
        temporary = f".{leaf}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
        temp_fd: int | None = None
        replaced = False
        try:
            temp_fd = os.open(temporary, _CREATE_FLAGS, 0o600, dir_fd=parent_fd)
            os.fchmod(temp_fd, 0o600)
            _verify_regular_file(os.fstat(temp_fd), label=f"{path} temporary")
            self._write_all(temp_fd, data)
            os.fsync(temp_fd)
            os.lseek(temp_fd, 0, os.SEEK_SET)
            if self._bounded_read_fd(temp_fd, label=f"{path} temporary") != data:
                raise StorageCommitError(
                    f"CAS temporary readback mismatch: {path}",
                    possibly_committed=False,
                )
            current = self._read_optional(path)
            current_sha = EMPTY_SHA if current is None else current.byte_sha256
            if current_sha != expected_sha256:
                raise CASMismatchError(expected_sha256, current_sha)
            os.replace(
                temporary,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            replaced = True
            os.fsync(parent_fd)
            readback = self._read_optional(path)
            if readback is None or readback.data != data:
                raise StorageCommitError(
                    f"CAS exact-byte readback mismatch: {path}",
                    possibly_committed=True,
                )
            return WriteResult(str(path), readback.byte_sha256, observed is None, True)
        except (CASMismatchError, StorageCommitError):
            raise
        except BaseException as exc:
            raise StorageCommitError(
                f"CAS replacement failed for {path}: {exc}",
                possibly_committed=replaced,
            ) from exc
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            if not replaced:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            os.close(parent_fd)

    def replace_cas(
        self,
        relative_path: str | PurePosixPath,
        expected_sha: str,
        data: bytes,
    ) -> WriteResult:
        """Durably replace run/latest state under its protocol flock and CAS."""

        path = _canonical_relative_path(relative_path)
        expected = _expected_sha(expected_sha)
        if type(data) is not bytes:
            raise StorageSecurityError("CAS payload must be bytes")
        if len(data) > self.max_read_bytes:
            raise StorageSecurityError("CAS payload exceeds bounded read limit")
        try:
            preliminary = self._read_optional(path)
        except StorageNotFoundError:
            preliminary = None
        preliminary_sha = EMPTY_SHA if preliminary is None else preliminary.byte_sha256
        if preliminary is not None and preliminary.data == data:
            return WriteResult(str(path), preliminary.byte_sha256, False, False)
        if preliminary_sha != expected:
            raise CASMismatchError(expected, preliminary_sha)
        lock_path = self._derived_lock(path)
        if str(lock_path) in self._held_locks.get():
            return self._replace_cas_locked(path, expected, data)
        with self.locked(lock_path):
            return self._replace_cas_locked(path, expected, data)

    def _ensure_lock_file(self, lock_relative_path: str | PurePosixPath) -> None:
        path = _canonical_relative_path(lock_relative_path)
        _, parent_fd, leaf = self._open_parent(path, create=True)
        fd: int | None = None
        try:
            try:
                fd = os.open(leaf, _CREATE_FLAGS, 0o600, dir_fd=parent_fd)
                os.fchmod(fd, 0o600)
                os.fsync(fd)
                os.fsync(parent_fd)
            except FileExistsError:
                fd = os.open(leaf, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            _verify_regular_file(os.fstat(fd), label=str(path))
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)

    @contextmanager
    def locked(
        self,
        lock_relative_path: str | PurePosixPath,
        *,
        blocking: bool = True,
    ) -> Iterator[None]:
        """Hold one 0600 single-link flock, with optional fail-fast contention."""

        path = _canonical_relative_path(lock_relative_path)
        held = self._held_locks.get()
        if str(path) in held:
            yield
            return
        self._ensure_lock_file(path)
        _, parent_fd, leaf = self._open_parent(path, create=False)
        fd: int | None = None
        token = None
        try:
            fd = os.open(
                leaf,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            _verify_regular_file(os.fstat(fd), label=str(path))
            operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(fd, operation)
            except BlockingIOError as exc:
                raise LockUnavailableError(f"protocol lock is busy: {path}") from exc
            token = self._held_locks.set(held | {str(path)})
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
    "EMPTY_SHA",
    "ExactOnceConflictError",
    "LATEST_LOCK_PATH",
    "LATEST_PATH",
    "LockUnavailableError",
    "SecureStore",
    "StorageCommitError",
    "StorageError",
    "StorageNotFoundError",
    "StorageSecurityError",
    "StoredBytes",
    "WriteResult",
]
