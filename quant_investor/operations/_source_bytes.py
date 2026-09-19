"""Private descriptor-relative input reader; callers own the exact byte policy."""

from dataclasses import dataclass
import errno
import hashlib
import os
import stat

from quant_investor.system.errors import SystemSecurityError, SystemStorageError
from quant_investor.system.storage import SecureSystemStorage, canonical_workspace_path


class SourceBytesChanged(RuntimeError):
    """A safely opened source changed identity while being read."""


@dataclass(frozen=True)
class SourceBytes:
    data: bytes
    byte_sha256: str
    stat_identity: tuple[int, ...]


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify(
    value: os.stat_result,
    *,
    allowed_modes: frozenset[int],
    maximum_bytes: int,
    require_owner: bool,
    require_single_link: bool,
    require_non_executable: bool,
) -> None:
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISREG(value.st_mode):
        raise SystemSecurityError("source is not a regular file")
    if require_owner and value.st_uid != os.geteuid():
        raise SystemSecurityError("source owner mismatch")
    if require_single_link and value.st_nlink != 1:
        raise SystemSecurityError("source must have exactly one hard link")
    if mode not in allowed_modes or mode & 0o022 or not mode & 0o400:
        raise SystemSecurityError("source mode is outside its input profile")
    if require_non_executable and mode & 0o111:
        raise SystemSecurityError("source must not be executable")
    if value.st_size <= 0 or value.st_size > maximum_bytes:
        raise SystemSecurityError("source byte read size is outside its bound")


def read_source_bytes(
    storage: SecureSystemStorage,
    path: str,
    *,
    allowed_modes: frozenset[int],
    maximum_bytes: int,
    require_owner: bool,
    require_single_link: bool,
    require_non_executable: bool,
) -> SourceBytes:
    """Open exact bytes without writes, symlink following, or reader fallback."""
    if type(path) is not str or "\x00" in path:
        raise SystemSecurityError("source path must be canonical relative text")
    relative = canonical_workspace_path(path)
    if (
        type(maximum_bytes) is not int
        or maximum_bytes <= 0
        or type(allowed_modes) is not frozenset
        or not allowed_modes
        or any(type(mode) is not int or not 0 <= mode <= 0o777 for mode in allowed_modes)
        or any(
            type(flag) is not bool
            for flag in (require_owner, require_single_link, require_non_executable)
        )
    ):
        raise SystemSecurityError("source byte profile is invalid")

    def verify(value: os.stat_result) -> None:
        _verify(
            value,
            allowed_modes=allowed_modes,
            maximum_bytes=maximum_bytes,
            require_owner=require_owner,
            require_single_link=require_single_link,
            require_non_executable=require_non_executable,
        )

    parent = storage._open_source_directory(tuple(relative.parts[:-1]))
    descriptor = None
    try:
        storage._reject_casefold_alias(parent, relative.name)
        descriptor = os.open(
            relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        before = os.fstat(descriptor)
        verify(before)
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        verify(after)
        named = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
        verify(named)
        if (
            remaining
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(named)
        ):
            raise SourceBytesChanged("source changed during byte read")
        raw = b"".join(chunks)
        return SourceBytes(raw, hashlib.sha256(raw).hexdigest(), _identity(after))
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise SystemSecurityError("source symlink/non-directory rejected") from exc
        raise SystemStorageError("source bytes cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)
