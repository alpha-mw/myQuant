"""Owner-private, descriptor-relative storage for neutral runtime control."""

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

CONTROL_ROOT: Final = PurePosixPath("results/research_runtime_control")
V4_FORMAL_ROOT: Final = PurePosixPath("results/v17_v4_formal_research")
V4_CANARY_ROOT: Final = PurePosixPath("results/v17_v4_canary")
EMPTY_SHA256: Final = "EMPTY"
DEFAULT_MAX_READ_BYTES: Final = 16 * 1024 * 1024

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_CREATE_FLAGS = (
    os.O_RDWR
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class ControlStorageError(RuntimeError):
    """Base neutral control storage error."""


class ControlStorageSecurityError(ControlStorageError):
    """A path, link, permission, or inode invariant failed."""


class ControlNotFoundError(ControlStorageError):
    """A required governed object is absent."""


class ControlExactOnceConflict(ControlStorageError):
    """An immutable path already contains different bytes."""


class ControlCASMismatch(ControlStorageError):
    """A pointer did not equal the expected prevalue."""

    def __init__(self, expected_sha256: str, observed_sha256: str) -> None:
        super().__init__("runtime-control CAS mismatch")
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256


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


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expected(value: str) -> str:
    if value == EMPTY_SHA256:
        return value
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ControlStorageSecurityError("expected SHA-256 is not canonical")
    return value


def canonical_relative_path(
    value: str | PurePosixPath,
    *,
    writable: bool,
) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        raise ControlStorageSecurityError("governed path must be text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ControlStorageSecurityError("governed path is not canonical POSIX")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ControlStorageSecurityError("governed path must be ASCII") from exc
    roots = (
        (CONTROL_ROOT,)
        if writable
        else (CONTROL_ROOT, V4_FORMAL_ROOT, V4_CANARY_ROOT)
    )
    if not any(path == root or root in path.parents for root in roots):
        raise ControlStorageSecurityError("path is outside allowed runtime roots")
    return path


def _identity(st: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        st.st_dev,
        st.st_ino,
        st.st_mode,
        st.st_nlink,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )


def _verify_directory(st: os.stat_result, *, label: str) -> None:
    if not stat.S_ISDIR(st.st_mode):
        raise ControlStorageSecurityError(f"{label} is not a directory")
    if stat.S_IMODE(st.st_mode) != 0o700:
        raise ControlStorageSecurityError(f"{label} directory mode must be 0700")


def _verify_directory_type(st: os.stat_result, *, label: str) -> None:
    if not stat.S_ISDIR(st.st_mode):
        raise ControlStorageSecurityError(f"{label} is not a directory")


def _verify_regular(st: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise ControlStorageSecurityError(f"{label} is not a regular file")
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise ControlStorageSecurityError(f"{label} file mode must be 0600")
    if st.st_nlink != 1:
        raise ControlStorageSecurityError(
            f"{label} file must have exactly one hard link"
        )


@dataclass
class ControlStore:
    """No-follow storage whose write namespace is exactly the control root."""

    workspace_root: Path
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> None:
        root = Path(workspace_root)
        if not root.is_absolute() or any(
            part in {"", ".", ".."}
            for part in root.parts[1:]
        ):
            raise ControlStorageSecurityError(
                "workspace_root must be a canonical absolute path"
            )
        if type(max_read_bytes) is not int or max_read_bytes <= 0:
            raise ControlStorageSecurityError(
                "max_read_bytes must be positive"
            )
        self.workspace_root = root
        self.max_read_bytes = max_read_bytes
        self._held: ContextVar[tuple[str, ...]] = ContextVar(
            f"research_runtime_control_locks_{id(self)}",
            default=(),
        )

    def _open_workspace(self) -> int:
        fd: int | None = None
        try:
            fd = os.open("/", _DIRECTORY_FLAGS)
            for part in self.workspace_root.parts[1:]:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = child
            _verify_directory_type(os.fstat(fd), label="workspace root")
            return fd
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise

    @staticmethod
    def _reject_casefold_alias(parent_fd: int, leaf: str) -> None:
        try:
            names = os.listdir(parent_fd)
        except OSError as exc:
            raise ControlStorageSecurityError(
                "cannot enumerate governed parent"
            ) from exc
        if any(
            name != leaf and name.casefold() == leaf.casefold()
            for name in names
        ):
            raise ControlStorageSecurityError(
                "casefold-colliding governed path"
            )

    def _open_directory(
        self,
        parts: tuple[str, ...],
        *,
        create: bool,
    ) -> int:
        fd = self._open_workspace()
        try:
            traversed: list[str] = []
            for part in parts:
                traversed.append(part)
                self._reject_casefold_alias(fd, part)
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        raise ControlNotFoundError(
                            "governed directory is absent"
                        ) from None
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                        os.chmod(
                            part,
                            0o700,
                            dir_fd=fd,
                            follow_symlinks=False,
                        )
                        os.fsync(fd)
                    except FileExistsError:
                        pass
                    self._reject_casefold_alias(fd, part)
                    try:
                        child = os.open(
                            part,
                            _DIRECTORY_FLAGS,
                            dir_fd=fd,
                        )
                    except OSError as exc:
                        raise ControlStorageSecurityError(
                            "cannot open newly created directory"
                        ) from exc
                except OSError as exc:
                    raise ControlStorageSecurityError(
                        "cannot open governed directory"
                    ) from exc
                try:
                    current = PurePosixPath(*traversed)
                    governed = (
                        current == CONTROL_ROOT
                        or CONTROL_ROOT in current.parents
                        or current == V4_FORMAL_ROOT
                        or V4_FORMAL_ROOT in current.parents
                        or current == V4_CANARY_ROOT
                        or V4_CANARY_ROOT in current.parents
                    )
                    verifier = (
                        _verify_directory
                        if governed
                        else _verify_directory_type
                    )
                    verifier(os.fstat(child), label=part)
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
        writable: bool,
    ) -> tuple[PurePosixPath, int, str]:
        path = canonical_relative_path(relative_path, writable=writable)
        if len(path.parts) < 2:
            raise ControlStorageSecurityError(
                "governed file must have a parent"
            )
        parent = self._open_directory(path.parent.parts, create=create)
        self._reject_casefold_alias(parent, path.name)
        return path, parent, path.name

    def initialize(self) -> None:
        fd = self._open_directory(CONTROL_ROOT.parts, create=True)
        os.close(fd)

    def _bounded_read(self, fd: int, *, label: str) -> bytes:
        before = os.fstat(fd)
        _verify_regular(before, label=label)
        if before.st_size > self.max_read_bytes:
            raise ControlStorageSecurityError("governed file is too large")
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
        if (
            len(raw) > self.max_read_bytes
            or _identity(before) != _identity(after)
            or len(raw) != after.st_size
        ):
            raise ControlStorageSecurityError(
                "governed file changed during stable read"
            )
        return raw

    def read_optional(
        self,
        relative_path: str | PurePosixPath,
    ) -> StoredBytes | None:
        try:
            path, parent, leaf = self._open_parent(
                relative_path,
                create=False,
                writable=False,
            )
        except ControlNotFoundError:
            return None
        try:
            try:
                fd = os.open(leaf, _READ_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise ControlStorageSecurityError(
                    "cannot open governed file"
                ) from exc
            try:
                raw = self._bounded_read(fd, label=str(path))
            finally:
                os.close(fd)
        finally:
            os.close(parent)
        return StoredBytes(str(path), raw, _sha(raw))

    def read(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str | None = None,
    ) -> bytes:
        observed = self.read_optional(relative_path)
        if observed is None:
            raise ControlNotFoundError("governed file does not exist")
        if expected_sha256 is not None:
            expected = _expected(expected_sha256)
            if (
                expected == EMPTY_SHA256
                or observed.byte_sha256 != expected
            ):
                raise ControlCASMismatch(
                    expected,
                    observed.byte_sha256,
                )
        return observed.data

    @staticmethod
    def _write_all(fd: int, raw: bytes) -> None:
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError(errno.EIO, "write made no progress")
            written += count

    def _write_temp(self, parent: int, leaf: str, raw: bytes) -> str:
        temp = f".{leaf}.tmp-{secrets.token_hex(16)}"
        fd: int | None = None
        try:
            fd = os.open(temp, _CREATE_FLAGS, 0o600, dir_fd=parent)
            os.fchmod(fd, 0o600)
            _verify_regular(os.fstat(fd), label="temporary control file")
            self._write_all(fd, raw)
            os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            if self._bounded_read(fd, label="temporary control file") != raw:
                raise ControlStorageError("temporary readback mismatch")
            return temp
        except BaseException:
            try:
                os.unlink(temp, dir_fd=parent)
            except OSError:
                pass
            raise
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def _unlink_temp(parent: int, temp: str | None) -> None:
        if temp is None:
            return
        try:
            os.unlink(temp, dir_fd=parent)
        except FileNotFoundError:
            pass

    def write_exact_once(
        self,
        relative_path: str | PurePosixPath,
        raw: bytes,
    ) -> WriteResult:
        if type(raw) is not bytes:
            raise ControlStorageSecurityError("payload must be bytes")
        path, parent, leaf = self._open_parent(
            relative_path,
            create=True,
            writable=True,
        )
        temp: str | None = None
        try:
            fcntl.flock(parent, fcntl.LOCK_EX)
            self._reject_casefold_alias(parent, leaf)
            observed = self.read_optional(path)
            if observed is not None:
                if observed.data != raw:
                    raise ControlExactOnceConflict(
                        "immutable path contains different bytes"
                    )
                return WriteResult(
                    str(path),
                    observed.byte_sha256,
                    False,
                    False,
                )
            temp = self._write_temp(parent, leaf, raw)
            current = self.read_optional(path)
            if current is not None:
                if current.data == raw:
                    return WriteResult(
                        str(path),
                        current.byte_sha256,
                        False,
                        False,
                    )
                raise ControlExactOnceConflict(
                    "immutable path appeared before publication"
                )
            os.replace(temp, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            temp = None
            os.fsync(parent)
            if self.read(path) != raw:
                raise ControlStorageError("immutable readback mismatch")
            return WriteResult(str(path), _sha(raw), True, False)
        finally:
            self._unlink_temp(parent, temp)
            os.close(parent)

    def replace_cas(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str,
        raw: bytes,
    ) -> WriteResult:
        if type(raw) is not bytes:
            raise ControlStorageSecurityError("payload must be bytes")
        expected = _expected(expected_sha256)
        path, parent, leaf = self._open_parent(
            relative_path,
            create=True,
            writable=True,
        )
        temp: str | None = None
        try:
            fcntl.flock(parent, fcntl.LOCK_EX)
            observed = self.read_optional(path)
            observed_sha = (
                EMPTY_SHA256
                if observed is None
                else observed.byte_sha256
            )
            if observed_sha != expected:
                raise ControlCASMismatch(expected, observed_sha)
            if observed is not None and observed.data == raw:
                return WriteResult(
                    str(path),
                    observed.byte_sha256,
                    False,
                    False,
                )
            temp = self._write_temp(parent, leaf, raw)
            current = self.read_optional(path)
            current_sha = (
                EMPTY_SHA256
                if current is None
                else current.byte_sha256
            )
            if current_sha != expected:
                raise ControlCASMismatch(expected, current_sha)
            os.replace(temp, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            temp = None
            os.fsync(parent)
            if self.read(path) != raw:
                raise ControlStorageError("pointer readback mismatch")
            return WriteResult(
                str(path),
                _sha(raw),
                expected == EMPTY_SHA256,
                expected != EMPTY_SHA256,
            )
        finally:
            self._unlink_temp(parent, temp)
            os.close(parent)

    def _ensure_lock(self, path: str | PurePosixPath) -> None:
        self.write_exact_once(path, b"")

    @contextmanager
    def locked(
        self,
        relative_path: str | PurePosixPath,
        *,
        existing_only: bool = False,
    ) -> Iterator[None]:
        path = canonical_relative_path(
            relative_path,
            writable=not existing_only,
        )
        key = str(path)
        held = self._held.get()
        if key in held:
            raise ControlStorageSecurityError(
                "recursive lock acquisition"
            )
        if existing_only:
            self.read(path)
        else:
            self._ensure_lock(path)
        _, parent, leaf = self._open_parent(
            path,
            create=False,
            writable=not existing_only,
        )
        fd: int | None = None
        token = None
        try:
            fd = os.open(
                leaf,
                os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            _verify_regular(os.fstat(fd), label=key)
            fcntl.flock(fd, fcntl.LOCK_EX)
            token = self._held.set((*held, key))
            yield
        finally:
            if token is not None:
                self._held.reset(token)
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            os.close(parent)


__all__ = [
    "CONTROL_ROOT",
    "ControlCASMismatch",
    "ControlExactOnceConflict",
    "ControlNotFoundError",
    "ControlStorageError",
    "ControlStorageSecurityError",
    "ControlStore",
    "EMPTY_SHA256",
    "StoredBytes",
    "V4_FORMAL_ROOT",
    "V4_CANARY_ROOT",
    "WriteResult",
    "canonical_relative_path",
]
