"""Read-only descriptor-relative exact I/O for portfolio-cycle artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Final

from .contracts import PortfolioCycleError, require_sha256

_DIRECTORY_FLAGS: Final = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS: Final = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


@dataclass(frozen=True)
class ExactBytes:
    relative_path: str
    data: bytes
    byte_sha256: str


def canonical_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        raise PortfolioCycleError("PORTFOLIO_CYCLE_PATH_INVALID", "artifact path must be text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_PATH_INVALID",
            "artifact path is not canonical relative POSIX",
        )
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_PATH_INVALID", "artifact path must be ASCII"
        ) from exc
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


def _verify_directory(st: os.stat_result, *, workspace: bool) -> None:
    if not stat.S_ISDIR(st.st_mode):
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_STORAGE_SECURITY",
            "path component is not a directory",
        )
    if workspace and st.st_uid != os.geteuid():
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_STORAGE_SECURITY",
            "workspace directory owner mismatch",
        )


def _verify_file(st: os.stat_result) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_STORAGE_SECURITY",
            "artifact is not a regular file",
        )
    if st.st_uid != os.geteuid():
        raise PortfolioCycleError("PORTFOLIO_CYCLE_STORAGE_SECURITY", "artifact owner mismatch")
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise PortfolioCycleError("PORTFOLIO_CYCLE_STORAGE_SECURITY", "artifact mode must be 0600")
    if st.st_nlink != 1:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_STORAGE_SECURITY",
            "artifact must have one hard link",
        )


class ExactReader:
    """A bounded reader that never creates, locks, caches, or selects files."""

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        max_read_bytes: int = 64 * 1024 * 1024,
        max_directory_entries: int = 100_000,
    ) -> None:
        if (
            type(max_read_bytes) is not int
            or max_read_bytes <= 0
            or type(max_directory_entries) is not int
            or max_directory_entries <= 0
        ):
            raise PortfolioCycleError("PORTFOLIO_CYCLE_STORAGE_SECURITY", "read bounds are invalid")
        try:
            root_text = os.path.abspath(os.fspath(workspace_root))
            root = Path(root_text)
        except (TypeError, ValueError, OSError) as exc:
            raise PortfolioCycleError(
                "PORTFOLIO_CYCLE_STORAGE_SECURITY",
                "workspace root is unavailable",
            ) from exc
        if not root.is_absolute() or "\x00" in root_text:
            raise PortfolioCycleError(
                "PORTFOLIO_CYCLE_STORAGE_SECURITY",
                "workspace root is not canonical absolute text",
            )
        self.workspace_root = root
        self.max_read_bytes = max_read_bytes
        self.max_directory_entries = max_directory_entries

    def _open_workspace(self) -> int:
        fd: int | None = None
        try:
            fd = os.open("/", _DIRECTORY_FLAGS)
            for part in self.workspace_root.parts[1:]:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = child
            _verify_directory(os.fstat(fd), workspace=True)
            return fd
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            raise PortfolioCycleError(
                "PORTFOLIO_CYCLE_STORAGE_SECURITY",
                "workspace path contains a symlink or non-directory component",
            ) from exc
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise

    def _reject_casefold_alias(self, parent_fd: int, leaf: str) -> None:
        count = 0
        try:
            with os.scandir(parent_fd) as entries:
                for entry in entries:
                    count += 1
                    if count > self.max_directory_entries:
                        raise PortfolioCycleError(
                            "PORTFOLIO_CYCLE_STORAGE_SECURITY",
                            "directory exceeds collision-check bound",
                        )
                    if entry.name != leaf and entry.name.casefold() == leaf.casefold():
                        raise PortfolioCycleError(
                            "PORTFOLIO_CYCLE_STORAGE_SECURITY",
                            "casefold-colliding path",
                        )
        except PortfolioCycleError:
            raise
        except OSError as exc:
            raise PortfolioCycleError(
                "PORTFOLIO_CYCLE_STORAGE_SECURITY",
                "cannot enumerate directory for collision check",
            ) from exc

    def _open_parent(self, path: PurePosixPath) -> int:
        fd = self._open_workspace()
        try:
            for part in path.parts[:-1]:
                self._reject_casefold_alias(fd, part)
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    raise PortfolioCycleError(
                        "PORTFOLIO_CYCLE_STORAGE_NOT_FOUND",
                        "artifact directory is absent",
                    ) from None
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise PortfolioCycleError(
                            "PORTFOLIO_CYCLE_STORAGE_SECURITY",
                            "symlink or non-directory path component",
                        ) from exc
                    raise
                _verify_directory(os.fstat(child), workspace=False)
                os.close(fd)
                fd = child
            self._reject_casefold_alias(fd, path.name)
            return fd
        except BaseException:
            os.close(fd)
            raise

    def read(self, relative_path: str | PurePosixPath, *, expected_sha256: str) -> ExactBytes:
        path = canonical_relative_path(relative_path)
        expected = require_sha256(expected_sha256, label="expected_sha256")
        parent = self._open_parent(path)
        fd: int | None = None
        try:
            try:
                fd = os.open(path.name, _READ_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                raise PortfolioCycleError(
                    "PORTFOLIO_CYCLE_STORAGE_NOT_FOUND", "artifact is absent"
                ) from None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise PortfolioCycleError(
                        "PORTFOLIO_CYCLE_STORAGE_SECURITY",
                        "artifact symlink rejected",
                    ) from exc
                raise
            before = os.fstat(fd)
            _verify_file(before)
            if before.st_size > self.max_read_bytes:
                raise PortfolioCycleError(
                    "PORTFOLIO_CYCLE_READ_BOUND_EXCEEDED",
                    "artifact exceeds the exact-read byte bound",
                )
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
                raise PortfolioCycleError(
                    "PORTFOLIO_CYCLE_STABLE_READ_FAILED",
                    "artifact changed during exact read",
                )
            observed = hashlib.sha256(raw).hexdigest()
            if observed != expected:
                raise PortfolioCycleError(
                    "PORTFOLIO_CYCLE_BYTE_SHA_MISMATCH",
                    f"expected {expected}, observed {observed}",
                )
            return ExactBytes(str(path), raw, observed)
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent)


__all__ = ["ExactBytes", "ExactReader", "canonical_relative_path"]
