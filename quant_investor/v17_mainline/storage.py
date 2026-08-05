"""Descriptor-relative owner-only storage for V17 mainline authority bytes."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Final

from .constants import EMPTY_SHA256, FORMAL_ROOT, MAINLINE_ROOT, SOURCE_ROOT
from .contracts import require_sha256

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
_GOVERNED_ROOTS: Final = tuple(
    PurePosixPath(value) for value in (MAINLINE_ROOT, FORMAL_ROOT, SOURCE_ROOT)
)


class MainlineStorageError(RuntimeError):
    pass


class MainlineStorageSecurityError(MainlineStorageError):
    pass


class MainlineNotFound(MainlineStorageError):
    pass


class MainlineCASMismatch(MainlineStorageError):
    def __init__(self, expected_sha256: str, observed_sha256: str) -> None:
        super().__init__("V17 mainline pointer CAS mismatch")
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256


class MainlineExactOnceConflict(MainlineStorageError):
    pass


@dataclass(frozen=True)
class StoredBytes:
    relative_path: str
    data: bytes
    byte_sha256: str


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def _verify_dir(st: os.stat_result, *, governed: bool) -> None:
    if not stat.S_ISDIR(st.st_mode):
        raise MainlineStorageSecurityError("path component is not a directory")
    if governed and stat.S_IMODE(st.st_mode) != 0o700:
        raise MainlineStorageSecurityError("governed directory mode must be 0700")
    if governed and st.st_uid != os.geteuid():
        raise MainlineStorageSecurityError("governed directory owner mismatch")


def _verify_file(st: os.stat_result) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise MainlineStorageSecurityError("artifact is not a regular file")
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise MainlineStorageSecurityError("artifact mode must be 0600")
    if st.st_uid != os.geteuid():
        raise MainlineStorageSecurityError("artifact owner mismatch")
    if st.st_nlink != 1:
        raise MainlineStorageSecurityError("artifact must have one hard link")


def canonical_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        raise MainlineStorageSecurityError("artifact path must be text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MainlineStorageSecurityError("artifact path is not canonical POSIX")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise MainlineStorageSecurityError("artifact path must be ASCII") from exc
    if not any(path == root or root in path.parents for root in _GOVERNED_ROOTS):
        raise MainlineStorageSecurityError("artifact path is outside V17 governed roots")
    return path


def _is_governed_directory(parts: tuple[str, ...]) -> bool:
    path = PurePosixPath(*parts)
    return any(path == root or root in path.parents for root in _GOVERNED_ROOTS)


class MainlineStore:
    """Low-level exact-byte storage; this is not a production publisher."""

    def __init__(
        self, workspace_root: str | os.PathLike[str], *, max_read_bytes: int = 64 * 1024 * 1024
    ) -> None:
        root = Path(workspace_root)
        if type(max_read_bytes) is not int or max_read_bytes <= 0:
            raise MainlineStorageSecurityError("workspace root or read bound is invalid")
        try:
            # Resolve the already-existing workspace root once.  This accepts
            # both a CLI-friendly relative root and macOS /tmp or /var system
            # aliases; every governed child is still opened with NOFOLLOW.
            root = root.resolve(strict=True)
        except OSError as exc:
            raise MainlineStorageSecurityError("workspace root is unavailable") from exc
        if not root.is_dir():
            raise MainlineStorageSecurityError("workspace root is not a directory")
        self.workspace_root = root
        self.max_read_bytes = max_read_bytes

    @staticmethod
    def _reject_casefold_alias(parent_fd: int, leaf: str) -> None:
        try:
            names = os.listdir(parent_fd)
        except OSError as exc:
            raise MainlineStorageSecurityError("cannot enumerate governed directory") from exc
        if any(name != leaf and name.casefold() == leaf.casefold() for name in names):
            raise MainlineStorageSecurityError("casefold-colliding path")

    def _open_workspace(self) -> int:
        fd: int | None = None
        try:
            fd = os.open("/", _DIRECTORY_FLAGS)
            for part in self.workspace_root.parts[1:]:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = child
            _verify_dir(os.fstat(fd), governed=False)
            return fd
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise

    def _open_directory(self, parts: tuple[str, ...], *, create: bool) -> int:
        fd = self._open_workspace()
        traversed: list[str] = []
        try:
            for part in parts:
                traversed.append(part)
                self._reject_casefold_alias(fd, part)
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        raise MainlineNotFound("artifact directory is absent") from None
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                        os.chmod(part, 0o700, dir_fd=fd, follow_symlinks=False)
                        child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                    except OSError as exc:
                        raise MainlineStorageSecurityError(
                            "cannot create governed directory"
                        ) from exc
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise MainlineStorageSecurityError(
                            "symlink or non-directory component"
                        ) from exc
                    raise
                _verify_dir(os.fstat(child), governed=_is_governed_directory(tuple(traversed)))
                os.close(fd)
                fd = child
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _parent_leaf(
        self, value: str | PurePosixPath, *, create: bool
    ) -> tuple[int, str, PurePosixPath]:
        path = canonical_relative_path(value)
        parent = self._open_directory(tuple(path.parts[:-1]), create=create)
        self._reject_casefold_alias(parent, path.name)
        return parent, path.name, path

    def _validate_write_path(self, value: str | PurePosixPath) -> PurePosixPath:
        path = canonical_relative_path(value)
        mainline_root = PurePosixPath(MAINLINE_ROOT)
        if path != mainline_root and mainline_root not in path.parents:
            raise MainlineStorageSecurityError("mainline store cannot publish external authority")
        return path

    def read(self, value: str | PurePosixPath, expected_sha256: str | None = None) -> StoredBytes:
        parent, leaf, path = self._parent_leaf(value, create=False)
        fd: int | None = None
        try:
            try:
                fd = os.open(leaf, _READ_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                raise MainlineNotFound("artifact is absent") from None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise MainlineStorageSecurityError("artifact symlink rejected") from exc
                raise
            before = os.fstat(fd)
            _verify_file(before)
            if before.st_size > self.max_read_bytes:
                raise MainlineStorageSecurityError("artifact exceeds read bound")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(fd)
            _verify_file(after)
            if _identity(before) != _identity(after) or len(raw) != after.st_size:
                raise MainlineStorageSecurityError("artifact changed during exact read")
            digest = _sha(raw)
            if expected_sha256 is not None:
                expected = require_sha256(expected_sha256, label="expected_sha256")
                if digest != expected:
                    raise MainlineCASMismatch(expected, digest)
            return StoredBytes(str(path), raw, digest)
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent)

    def read_optional(self, value: str | PurePosixPath) -> StoredBytes | None:
        try:
            return self.read(value)
        except MainlineNotFound:
            return None

    @staticmethod
    def _write_all(fd: int, raw: bytes) -> None:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise MainlineStorageError("short governed write")
            view = view[written:]

    def write_exact_once(self, value: str | PurePosixPath, raw: bytes) -> StoredBytes:
        if type(raw) is not bytes or len(raw) > self.max_read_bytes:
            raise MainlineStorageSecurityError("artifact payload is invalid")
        write_path = self._validate_write_path(value)
        parent, leaf, path = self._parent_leaf(write_path, create=True)
        temporary = f".{leaf}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        fd: int | None = None
        try:
            existing = self.read_optional(path)
            if existing is not None:
                if existing.data != raw:
                    raise MainlineExactOnceConflict("immutable artifact identity conflict")
                return existing
            fd = os.open(temporary, _CREATE_FLAGS, 0o600, dir_fd=parent)
            os.fchmod(fd, 0o600)
            self._write_all(fd, raw)
            os.fsync(fd)
            _verify_file(os.fstat(fd))
            os.close(fd)
            fd = None
            try:
                os.link(
                    temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False
                )
            except FileExistsError:
                existing = self.read(path)
                if existing.data != raw:
                    raise MainlineExactOnceConflict(
                        "immutable artifact identity conflict"
                    ) from None
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
            os.fsync(parent)
            readback = self.read(path)
            if readback.data != raw:
                raise MainlineStorageError("immutable exact-byte readback mismatch")
            return readback
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    def compare_and_swap(
        self, value: str | PurePosixPath, raw: bytes, *, expected_sha256: str
    ) -> StoredBytes:
        if type(raw) is not bytes or len(raw) > self.max_read_bytes:
            raise MainlineStorageSecurityError("CAS payload is invalid")
        expected = expected_sha256
        if expected != EMPTY_SHA256:
            expected = require_sha256(expected, label="expected_sha256")
        write_path = self._validate_write_path(value)
        parent, leaf, path = self._parent_leaf(write_path, create=True)
        lock_leaf = f".{leaf}.lock"
        temporary = f".{leaf}.cas-{os.getpid()}-{secrets.token_hex(8)}"
        lock_fd: int | None = None
        temp_fd: int | None = None
        try:
            try:
                lock_fd = os.open(lock_leaf, _CREATE_FLAGS, 0o600, dir_fd=parent)
            except FileExistsError:
                lock_fd = os.open(
                    lock_leaf, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent
                )
            os.fchmod(lock_fd, 0o600)
            _verify_file(os.fstat(lock_fd))
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            current = self.read_optional(path)
            observed = EMPTY_SHA256 if current is None else current.byte_sha256
            if observed != expected:
                raise MainlineCASMismatch(expected, observed)
            temp_fd = os.open(temporary, _CREATE_FLAGS, 0o600, dir_fd=parent)
            os.fchmod(temp_fd, 0o600)
            self._write_all(temp_fd, raw)
            os.fsync(temp_fd)
            _verify_file(os.fstat(temp_fd))
            os.close(temp_fd)
            temp_fd = None
            os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            readback = self.read(path)
            if readback.data != raw:
                raise MainlineStorageError("CAS exact-byte readback mismatch")
            return readback
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(parent)


__all__ = [
    "MainlineCASMismatch",
    "MainlineExactOnceConflict",
    "MainlineNotFound",
    "MainlineStorageError",
    "MainlineStorageSecurityError",
    "MainlineStore",
    "StoredBytes",
    "canonical_relative_path",
]
