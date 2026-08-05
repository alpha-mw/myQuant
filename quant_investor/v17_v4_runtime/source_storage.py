"""Owner-only, descriptor-relative storage for V17 v4 source artifacts."""

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

SOURCE_ROOT: Final = PurePosixPath("data/private/v17_v4_sources")
RUN_ROOT: Final = PurePosixPath("data/private/v17_v4_runs")
SHADOW_ROOT: Final = PurePosixPath("results/v17_v4_shadow")
FACTOR_CONTROL_ROOT: Final = PurePosixPath(
    "data/private/factor_governance_production_control_v1"
)
GOVERNED_ROOTS: Final = (
    SOURCE_ROOT,
    RUN_ROOT,
    SHADOW_ROOT,
)
REFERENCE_ROOTS: Final = (*GOVERNED_ROOTS, FACTOR_CONTROL_ROOT)
PIT_CATALOG_ROOT: Final = SOURCE_ROOT / "pit_catalog"
PIT_CATALOG_POINTER: Final = PIT_CATALOG_ROOT / "_latest.json"
PIT_CATALOG_LOCK: Final = PIT_CATALOG_ROOT / ".latest.lock"
EMPTY_SHA256: Final = "EMPTY"
DEFAULT_MAX_READ_BYTES: Final = 64 * 1024 * 1024
DEFAULT_MAX_HASH_BYTES: Final = 8 * 1024 * 1024 * 1024

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


class SourceStorageError(RuntimeError):
    pass


class SourceStorageSecurityError(SourceStorageError):
    pass


class SourceNotFoundError(SourceStorageError):
    pass


class SourceExactOnceConflict(SourceStorageError):
    pass


class SourceCASMismatch(SourceStorageError):
    def __init__(self, expected_sha256: str, observed_sha256: str) -> None:
        super().__init__("V17 v4 source CAS mismatch")
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256


@dataclass(frozen=True)
class StoredBytes:
    relative_path: str
    data: bytes
    byte_sha256: str


@dataclass(frozen=True)
class StoredFile:
    relative_path: str
    byte_sha256: str
    size: int


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
        raise SourceStorageSecurityError("expected SHA-256 is not canonical")
    return value


def canonical_source_path(value: str | PurePosixPath) -> PurePosixPath:
    return _canonical_path(
        value,
        roots=(SOURCE_ROOT,),
        label="V17 v4 source root",
    )


def canonical_governed_path(
    value: str | PurePosixPath,
) -> PurePosixPath:
    return _canonical_path(
        value,
        roots=GOVERNED_ROOTS,
        label="V17 v4 governed roots",
    )


def canonical_reference_path(
    value: str | PurePosixPath,
) -> PurePosixPath:
    return _canonical_path(
        value,
        roots=REFERENCE_ROOTS,
        label="V17 v4 exact-reference roots",
    )


def _canonical_path(
    value: str | PurePosixPath,
    *,
    roots: tuple[PurePosixPath, ...],
    label: str,
) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        raise SourceStorageSecurityError("source path must be text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SourceStorageSecurityError("source path is not canonical POSIX")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SourceStorageSecurityError("source path must be ASCII") from exc
    if not any(path == root or root in path.parents for root in roots):
        raise SourceStorageSecurityError(f"path is outside the {label}")
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


def _verify_directory(st: os.stat_result, *, private: bool, label: str) -> None:
    if not stat.S_ISDIR(st.st_mode):
        raise SourceStorageSecurityError(f"{label} is not a directory")
    if private and stat.S_IMODE(st.st_mode) != 0o700:
        raise SourceStorageSecurityError(f"{label} directory mode must be 0700")
    if private and st.st_uid != os.geteuid():
        raise SourceStorageSecurityError(
            f"{label} directory must be owned by the current user"
        )


def _verify_regular(st: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise SourceStorageSecurityError(f"{label} is not a regular file")
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise SourceStorageSecurityError(f"{label} file mode must be 0600")
    if st.st_uid != os.geteuid():
        raise SourceStorageSecurityError(
            f"{label} file must be owned by the current user"
        )
    if st.st_nlink != 1:
        raise SourceStorageSecurityError(
            f"{label} file must have exactly one hard link"
        )


@dataclass
class SourceStore:
    workspace_root: Path
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES,
    ) -> None:
        root = Path(workspace_root)
        if not root.is_absolute() or any(
            part in {"", ".", ".."} for part in root.parts[1:]
        ):
            raise SourceStorageSecurityError(
                "workspace_root must be a canonical absolute path"
            )
        if type(max_read_bytes) is not int or max_read_bytes <= 0:
            raise SourceStorageSecurityError(
                "max_read_bytes must be positive"
            )
        if type(max_hash_bytes) is not int or max_hash_bytes < max_read_bytes:
            raise SourceStorageSecurityError(
                "max_hash_bytes must be at least max_read_bytes"
            )
        self.workspace_root = root
        self.max_read_bytes = max_read_bytes
        self.max_hash_bytes = max_hash_bytes
        self._held: ContextVar[tuple[str, ...]] = ContextVar(
            f"v17_v4_source_locks_{id(self)}",
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
            _verify_directory(
                os.fstat(fd),
                private=False,
                label="workspace root",
            )
            return fd
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        return canonical_source_path(value)

    def _private_path(self, path: PurePosixPath) -> bool:
        return path == SOURCE_ROOT or SOURCE_ROOT in path.parents

    @staticmethod
    def _reject_casefold_alias(parent_fd: int, leaf: str) -> None:
        try:
            names = os.listdir(parent_fd)
        except OSError as exc:
            raise SourceStorageSecurityError(
                "cannot enumerate source parent"
            ) from exc
        if any(
            name != leaf and name.casefold() == leaf.casefold()
            for name in names
        ):
            raise SourceStorageSecurityError(
                "casefold-colliding source path"
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
                        raise SourceNotFoundError(
                            "source directory is absent"
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
                        child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                    except OSError as exc:
                        raise SourceStorageSecurityError(
                            "cannot open created source directory"
                        ) from exc
                except OSError as exc:
                    raise SourceStorageSecurityError(
                        "cannot open source directory"
                    ) from exc
                try:
                    current = PurePosixPath(*traversed)
                    private = self._private_path(current)
                    _verify_directory(
                        os.fstat(child),
                        private=private,
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
        path = self._canonical_path(relative_path)
        if len(path.parts) < 2:
            raise SourceStorageSecurityError(
                "source file must have a parent"
            )
        parent = self._open_directory(path.parent.parts, create=create)
        self._reject_casefold_alias(parent, path.name)
        return path, parent, path.name

    def initialize(self) -> None:
        fd = self._open_directory(SOURCE_ROOT.parts, create=True)
        os.close(fd)

    def _bounded_read(self, fd: int, *, label: str) -> bytes:
        before = os.fstat(fd)
        _verify_regular(before, label=label)
        if before.st_size > self.max_read_bytes:
            raise SourceStorageSecurityError("source file is too large")
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
            raise SourceStorageSecurityError(
                "source file changed during stable read"
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
            )
        except SourceNotFoundError:
            return None
        try:
            try:
                fd = os.open(leaf, _READ_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise SourceStorageSecurityError(
                    "cannot open source file"
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
            raise SourceNotFoundError("source file does not exist")
        if expected_sha256 is not None:
            expected = _expected(expected_sha256)
            if (
                expected == EMPTY_SHA256
                or observed.byte_sha256 != expected
            ):
                raise SourceCASMismatch(expected, observed.byte_sha256)
        return observed.data

    def verify_sha256(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str,
    ) -> StoredFile:
        expected = _expected(expected_sha256)
        if expected == EMPTY_SHA256:
            raise SourceStorageSecurityError(
                "file verification requires a SHA-256"
            )
        path, parent, leaf = self._open_parent(
            relative_path,
            create=False,
        )
        try:
            try:
                fd = os.open(leaf, _READ_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                raise SourceNotFoundError(
                    "source file does not exist"
                ) from None
            except OSError as exc:
                raise SourceStorageSecurityError(
                    "cannot open source file"
                ) from exc
            try:
                before = os.fstat(fd)
                _verify_regular(before, label=str(path))
                if before.st_size > self.max_hash_bytes:
                    raise SourceStorageSecurityError(
                        "source file exceeds the hash limit"
                    )
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_hash_bytes:
                        raise SourceStorageSecurityError(
                            "source file exceeds the hash limit"
                        )
                    digest.update(chunk)
                after = os.fstat(fd)
                if _identity(before) != _identity(after) or size != after.st_size:
                    raise SourceStorageSecurityError(
                        "source file changed during SHA readback"
                    )
            finally:
                os.close(fd)
        finally:
            os.close(parent)
        observed = digest.hexdigest()
        if observed != expected:
            raise SourceCASMismatch(expected, observed)
        return StoredFile(str(path), observed, size)

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
            _verify_regular(os.fstat(fd), label="temporary source file")
            self._write_all(fd, raw)
            os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            if self._bounded_read(fd, label="temporary source file") != raw:
                raise SourceStorageError("temporary source readback mismatch")
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
            raise SourceStorageSecurityError("source payload must be bytes")
        path, parent, leaf = self._open_parent(
            relative_path,
            create=True,
        )
        temp: str | None = None
        try:
            fcntl.flock(parent, fcntl.LOCK_EX)
            observed = self.read_optional(path)
            if observed is not None:
                if observed.data != raw:
                    raise SourceExactOnceConflict(
                        "immutable source path contains different bytes"
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
                raise SourceExactOnceConflict(
                    "immutable source path appeared before publication"
                )
            os.replace(temp, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            temp = None
            os.fsync(parent)
            if self.read(path) != raw:
                raise SourceStorageError("immutable source readback mismatch")
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
            raise SourceStorageSecurityError("source payload must be bytes")
        expected = _expected(expected_sha256)
        path, parent, leaf = self._open_parent(
            relative_path,
            create=True,
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
                raise SourceCASMismatch(expected, observed_sha)
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
                raise SourceCASMismatch(expected, current_sha)
            os.replace(temp, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            temp = None
            os.fsync(parent)
            if self.read(path) != raw:
                raise SourceStorageError("source pointer readback mismatch")
            return WriteResult(
                str(path),
                _sha(raw),
                expected == EMPTY_SHA256,
                expected != EMPTY_SHA256,
            )
        finally:
            self._unlink_temp(parent, temp)
            os.close(parent)

    @contextmanager
    def locked(
        self,
        relative_path: str | PurePosixPath,
    ) -> Iterator[None]:
        path = self._canonical_path(relative_path)
        key = str(path)
        held = self._held.get()
        if key in held:
            raise SourceStorageSecurityError("recursive source lock")
        self.write_exact_once(path, b"")
        _, parent, leaf = self._open_parent(path, create=False)
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


class GovernedStore(SourceStore):
    """Descriptor-relative storage across the fixed V17 v4 roots.

    This broad store is for exact-reference readback and tightly scoped
    publishers.  Individual writers must still enforce a narrower path set.
    """

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        return canonical_governed_path(value)

    def _private_path(self, path: PurePosixPath) -> bool:
        return any(
            path == root or root in path.parents
            for root in GOVERNED_ROOTS
        )


class ExactReferenceReader(GovernedStore):
    """Read-only exact-byte reader including Factor production control."""

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        return canonical_reference_path(value)

    def _private_path(self, path: PurePosixPath) -> bool:
        return any(
            path == root or root in path.parents
            for root in REFERENCE_ROOTS
        )

    def initialize(self) -> None:
        raise SourceStorageSecurityError(
            "exact-reference reader cannot initialize storage"
        )

    def write_exact_once(
        self,
        relative_path: str | PurePosixPath,
        raw: bytes,
    ) -> WriteResult:
        raise SourceStorageSecurityError(
            "exact-reference reader cannot write"
        )

    def replace_cas(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str,
        raw: bytes,
    ) -> WriteResult:
        raise SourceStorageSecurityError(
            "exact-reference reader cannot write"
        )

    @contextmanager
    def locked(
        self,
        relative_path: str | PurePosixPath,
    ) -> Iterator[None]:
        raise SourceStorageSecurityError(
            "exact-reference reader cannot lock"
        )
        yield


__all__ = [
    "EMPTY_SHA256",
    "ExactReferenceReader",
    "FACTOR_CONTROL_ROOT",
    "GOVERNED_ROOTS",
    "GovernedStore",
    "PIT_CATALOG_LOCK",
    "PIT_CATALOG_POINTER",
    "PIT_CATALOG_ROOT",
    "REFERENCE_ROOTS",
    "RUN_ROOT",
    "SHADOW_ROOT",
    "SOURCE_ROOT",
    "SourceCASMismatch",
    "SourceExactOnceConflict",
    "SourceNotFoundError",
    "SourceStorageError",
    "SourceStorageSecurityError",
    "SourceStore",
    "StoredBytes",
    "StoredFile",
    "WriteResult",
    "canonical_governed_path",
    "canonical_reference_path",
    "canonical_source_path",
]
