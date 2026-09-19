"""Restricted descriptor-relative storage for daily coordinator records.

Reuse the System storage's verified I/O primitives, not its System-root writer
API. No System path, activation method or caller-controlled root is exposed.
"""

from contextlib import contextmanager
import fcntl
import os
from pathlib import PurePosixPath
import secrets
from typing import Iterator

from quant_investor.system.storage import (
    SecureSystemStorage,
    StoredBytes,
    _verify_directory,
    _verify_file,
)
from quant_investor.system.errors import (
    SystemImmutableConflict,
    SystemNotFound,
    SystemSecurityError,
    SystemStorageError,
)
from .daily_contract import validate_ref

ROOT = PurePosixPath("results/operations/daily_production/CN")
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class JournalStorage:
    """Only journal read, lock, immutable write and derived-status replacement."""

    def __init__(self, workspace: str):
        self._io = SecureSystemStorage(workspace)

    @staticmethod
    def _path(value: str) -> PurePosixPath:
        validate_ref({"path": value, "sha256": "0" * 64})
        path = PurePosixPath(value)
        if ROOT not in path.parents or not value.isascii():
            raise SystemSecurityError("DAILY_JOURNAL_ROOT_INVALID")
        return path

    @staticmethod
    def _open_child(fd: int, part: str, *, create: bool) -> int:
        try:
            return os.open(part, DIRECTORY_FLAGS, dir_fd=fd)

        except FileNotFoundError:
            if not create:
                raise SystemNotFound("daily journal directory absent") from None
            try:
                os.mkdir(part, 0o700, dir_fd=fd)
                os.fsync(fd)
            except FileExistsError:
                pass
            return os.open(part, DIRECTORY_FLAGS, dir_fd=fd)

    @staticmethod
    def _governed_directory(path: PurePosixPath) -> bool:
        return path == ROOT or ROOT in path.parents

    def _parent(self, value: str, *, create: bool) -> tuple[int, str, PurePosixPath]:
        path = self._path(value)
        fd = self._io._open_workspace()
        walked: list[str] = []
        try:
            for part in path.parts[:-1]:
                walked.append(part)
                self._io._reject_casefold_alias(fd, part)
                child = self._open_child(fd, part, create=create)
                governed = PurePosixPath(*walked)
                try:
                    _verify_directory(os.fstat(child), governed=self._governed_directory(governed))
                except BaseException:
                    os.close(child)
                    raise
                os.close(fd)
                fd = child
            self._io._reject_casefold_alias(fd, path.name)
            return fd, path.name, path
        except BaseException:
            os.close(fd)
            raise

    def read(self, path: str) -> StoredBytes | None:
        try:
            parent, leaf, relative = self._parent(path, create=False)
        except SystemNotFound:
            return None
        try:
            return self._io._read_leaf(parent, leaf, relative_path=relative, optional=True)
        finally:
            os.close(parent)

    def _install(
        self, parent: int, temporary: str, relative: PurePosixPath, raw: bytes, *, projection: bool
    ) -> None:
        leaf = relative.name
        if projection:
            os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            return
        try:
            os.link(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        except FileExistsError:
            existing = self._io._read_leaf(parent, leaf, relative_path=relative, optional=False)
            if existing is None or existing.data != raw:
                raise SystemImmutableConflict("DAILY_IMMUTABLE_CONFLICT") from None
        os.unlink(temporary, dir_fd=parent)

    def write(self, path: str, raw: bytes, *, projection: bool = False) -> StoredBytes:
        if type(raw) is not bytes or not raw or len(raw) > self._io.max_read_bytes:
            raise SystemSecurityError("DAILY_JOURNAL_BYTES_INVALID")
        relative = self._path(path)
        if projection and (relative.name != "dag-status.v1.json" or len(relative.parts) != 6):
            raise SystemSecurityError("DAILY_PROJECTION_PATH_INVALID")
        parent, leaf, relative = self._parent(path, create=True)
        temporary = ".journal-" + secrets.token_hex(12)
        try:
            previous = self._io._read_leaf(parent, leaf, relative_path=relative, optional=True)
            if previous is not None:
                if previous.data == raw:
                    return previous
                if not projection:
                    raise SystemImmutableConflict("DAILY_IMMUTABLE_CONFLICT")
            descriptor = self._io._write_temporary_file(parent, temporary, raw)
            os.close(descriptor)
            self._install(parent, temporary, relative, raw, projection=projection)
            os.fsync(parent)
            result = self._io._read_leaf(parent, leaf, relative_path=relative, optional=False)
            if result is None or result.data != raw:
                raise SystemStorageError("DAILY_JOURNAL_READBACK_FAILED")
            return result
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    @contextmanager
    def lock(self, path: str, *, nonblocking: bool = False) -> Iterator[None]:
        parent, leaf, relative = self._parent(path, create=True)
        fd = None
        try:
            try:
                fd = os.open(
                    leaf, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
                )
                os.fsync(parent)
            except FileExistsError:
                fd = os.open(leaf, os.O_RDWR | os.O_NOFOLLOW, dir_fd=parent)
            _verify_file(os.fstat(fd))
            fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
            yield
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent)
