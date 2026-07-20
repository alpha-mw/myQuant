"""Descriptor-bound reads for private and governed evidence-v2 inputs."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import stat
import sys
from collections.abc import Callable
from typing import Any

from .contracts import (
    ARTIFACT_MAX_BYTES,
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EvidenceRef,
    EvidenceV2Error,
    require_sha256,
)

_AclChecker = Callable[[int, str], bool]


@dataclass(frozen=True)
class RootPolicy:
    policy_id: str
    directory_mode: int | None
    file_mode: int | None
    require_current_uid: bool
    reject_group_world_write: bool
    require_no_extended_acl: bool


PRIVATE_EVIDENCE_POLICY = RootPolicy(
    policy_id="v16.private-evidence-root.v2",
    directory_mode=0o700,
    file_mode=0o600,
    require_current_uid=True,
    reject_group_world_write=True,
    require_no_extended_acl=True,
)
TRUST_MATERIAL_POLICY = RootPolicy(
    policy_id="v16.trust-material-root.v2",
    directory_mode=0o700,
    file_mode=0o600,
    require_current_uid=True,
    reject_group_world_write=True,
    require_no_extended_acl=True,
)
GOVERNED_DATA_POLICY = RootPolicy(
    policy_id="v16.governed-data-root.v2",
    directory_mode=None,
    file_mode=None,
    require_current_uid=False,
    reject_group_world_write=True,
    require_no_extended_acl=True,
)
_CANONICAL_POLICIES = {
    policy.policy_id: policy
    for policy in (
        PRIVATE_EVIDENCE_POLICY,
        TRUST_MATERIAL_POLICY,
        GOVERNED_DATA_POLICY,
    )
}


@dataclass(frozen=True)
class BoundBytes:
    absolute_path: str
    payload: bytes
    byte_sha256: str
    device: int
    inode: int
    size: int


def _darwin_acl_profile(fd: int, label: str) -> tuple[bool, bool]:
    """Return ``(has_extended_acl, has_allow_entry)`` for one descriptor."""

    if sys.platform != "darwin":
        raise EvidenceV2Error(
            f"{label} platform ACL verification is unsupported on {sys.platform}"
        )

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        acl_get_fd_np = libc.acl_get_fd_np
        acl_get_entry = libc.acl_get_entry
        acl_get_tag_type = libc.acl_get_tag_type
        acl_free = libc.acl_free
    except (AttributeError, OSError) as exc:
        raise EvidenceV2Error(f"{label} platform ACL API is unavailable") from exc

    acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_uint]
    acl_get_fd_np.restype = ctypes.c_void_p
    acl_get_entry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    acl_get_entry.restype = ctypes.c_int
    acl_get_tag_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    acl_get_tag_type.restype = ctypes.c_int
    acl_free.argtypes = [ctypes.c_void_p]
    acl_free.restype = ctypes.c_int

    # Darwin's acl_get_fd_np returns ENOENT when ACL_TYPE_EXTENDED is absent.
    acl_type_extended = 0x00000100
    ctypes.set_errno(0)
    acl = acl_get_fd_np(fd, acl_type_extended)
    if not acl:
        error_number = ctypes.get_errno()
        if error_number == errno.ENOENT:
            return False, False
        raise EvidenceV2Error(
            f"{label} platform ACL lookup failed: errno={error_number}"
        )

    has_allow_entry = False
    try:
        entry = ctypes.c_void_p()
        ctypes.set_errno(0)
        status = acl_get_entry(acl, 0, ctypes.byref(entry))
        while status == 0:
            tag_type = ctypes.c_int()
            ctypes.set_errno(0)
            if acl_get_tag_type(entry, ctypes.byref(tag_type)) != 0:
                error_number = ctypes.get_errno()
                raise EvidenceV2Error(
                    f"{label} platform ACL tag lookup failed: errno={error_number}"
                )
            if tag_type.value == 1:
                has_allow_entry = True
            elif tag_type.value != 2:
                raise EvidenceV2Error(
                    f"{label} platform ACL has an unknown tag: {tag_type.value}"
                )
            ctypes.set_errno(0)
            status = acl_get_entry(acl, 1, ctypes.byref(entry))
        error_number = ctypes.get_errno()
        if status != -1 or error_number != errno.EINVAL:
            raise EvidenceV2Error(
                f"{label} platform ACL iteration failed: errno={error_number}"
            )
    except Exception:
        acl_free(acl)
        raise

    free_status = acl_free(acl)
    if free_status != 0:
        raise EvidenceV2Error(f"{label} platform ACL release failed")
    return True, has_allow_entry


def platform_acl_absent(fd: int, label: str) -> bool:
    """Prove that a Darwin descriptor has no extended ACL entries."""

    has_extended_acl, _has_allow_entry = _darwin_acl_profile(fd, label)
    return not has_extended_acl


def _platform_ancestor_acl_safe(fd: int, label: str) -> bool:
    _has_extended_acl, has_allow_entry = _darwin_acl_profile(fd, label)
    return not has_allow_entry


def _flags(*, directory: bool = False) -> int:
    value = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        value |= getattr(os, "O_DIRECTORY", 0)
    return value


def _require_canonical_policy(policy: RootPolicy) -> None:
    expected = _CANONICAL_POLICIES.get(getattr(policy, "policy_id", None))
    if expected is None or policy != expected:
        raise EvidenceV2Error("secure evidence factory requires a canonical root policy")


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _check_permissions(
    metadata: os.stat_result,
    *,
    policy: RootPolicy,
    is_directory: bool,
    acl_checker: _AclChecker | None,
    fd: int,
    label: str,
) -> None:
    expected_mode = policy.directory_mode if is_directory else policy.file_mode
    actual_mode = stat.S_IMODE(metadata.st_mode)
    if expected_mode is not None and actual_mode != expected_mode:
        raise EvidenceV2Error(
            f"{label} mode mismatch: expected {expected_mode:04o}, got {actual_mode:04o}"
        )
    if policy.require_current_uid and metadata.st_uid != os.getuid():
        raise EvidenceV2Error(f"{label} owner is not the current uid")
    if policy.reject_group_world_write and actual_mode & 0o022:
        raise EvidenceV2Error(f"{label} is group/world writable")
    if policy.require_no_extended_acl:
        checker = acl_checker or platform_acl_absent
        try:
            acl_absent = checker(fd, label)
        except EvidenceV2Error:
            raise
        except Exception as exc:
            raise EvidenceV2Error(f"{label} ACL verification failed") from exc
        if acl_absent is not True:
            raise EvidenceV2Error(f"{label} has an extended ACL")


def _check_trusted_root_ancestor(
    metadata: os.stat_result,
    *,
    fd: int,
    label: str,
    acl_checker: _AclChecker | None,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise EvidenceV2Error(f"trusted root ancestor is not a directory: {label}")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise EvidenceV2Error(f"trusted root ancestor is group/world writable: {label}")
    try:
        acl_safe = (
            acl_checker(fd, label)
            if acl_checker is not None
            else _platform_ancestor_acl_safe(fd, label)
        )
    except EvidenceV2Error:
        raise
    except Exception as exc:
        raise EvidenceV2Error(
            f"trusted root ancestor ACL verification failed: {label}"
        ) from exc
    if acl_safe is not True:
        raise EvidenceV2Error(
            f"trusted root ancestor has an extended allow ACL: {label}"
        )


def _open_root(
    root: Path,
    *,
    policy: RootPolicy,
    acl_checker: _AclChecker | None,
) -> int:
    root_text = str(root)
    if (
        not root.is_absolute()
        or "\x00" in root_text
        or os.path.normpath(root_text) != root_text
        or root_text.startswith("//")
    ):
        raise EvidenceV2Error("trusted root must be absolute")
    descriptors: list[int] = []
    try:
        descriptor = os.open("/", _flags(directory=True))
        descriptors.append(descriptor)
        ancestor_path = Path("/")
        for part in root.parts[1:]:
            _check_trusted_root_ancestor(
                os.fstat(descriptor),
                fd=descriptor,
                label=str(ancestor_path),
                acl_checker=acl_checker,
            )
            descriptor = os.open(
                part,
                _flags(directory=True),
                dir_fd=descriptor,
            )
            descriptors.append(descriptor)
            ancestor_path /= part
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise EvidenceV2Error("trusted root is not a directory")
        _check_permissions(
            metadata,
            policy=policy,
            is_directory=True,
            acl_checker=acl_checker,
            fd=descriptor,
            label=str(root),
        )
    except OSError as exc:
        for opened_descriptor in reversed(descriptors):
            os.close(opened_descriptor)
        raise EvidenceV2Error(f"trusted root open failed: {root}") from exc
    except Exception:
        for opened_descriptor in reversed(descriptors):
            os.close(opened_descriptor)
        raise
    for ancestor_descriptor in descriptors[:-1]:
        os.close(ancestor_descriptor)
    return descriptor


def _relative_parts(root: Path, path: Path) -> tuple[str, ...]:
    path_text = str(path)
    if (
        not path.is_absolute()
        or "\x00" in path_text
        or os.path.normpath(path_text) != path_text
        or path_text.startswith("//")
    ):
        raise EvidenceV2Error("evidence path must be absolute and NUL-free")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise EvidenceV2Error("evidence path escapes its trusted root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise EvidenceV2Error("evidence path is not a strict root-relative leaf")
    return relative.parts


def _read_bound_bytes(
    *,
    root: str | Path,
    path: str | Path,
    policy: RootPolicy,
    expected_sha256: str,
    max_bytes: int = ARTIFACT_MAX_BYTES,
    acl_checker: _AclChecker | None = None,
) -> BoundBytes:
    """Read one regular file without path discovery or mutable-path races."""

    expected = require_sha256(expected_sha256, label="expected byte SHA")
    root_path = Path(root)
    target = Path(path)
    parts = _relative_parts(root_path, target)
    if max_bytes <= 0:
        raise EvidenceV2Error("max_bytes must be positive")

    descriptors: list[int] = []
    root_fd = _open_root(root_path, policy=policy, acl_checker=acl_checker)
    descriptors.append(root_fd)
    current_fd = root_fd
    try:
        for index, part in enumerate(parts[:-1]):
            try:
                next_fd = os.open(part, _flags(directory=True), dir_fd=current_fd)
            except OSError as exc:
                raise EvidenceV2Error(f"evidence directory open failed: {part}") from exc
            descriptors.append(next_fd)
            current_fd = next_fd
            metadata = os.fstat(current_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise EvidenceV2Error("evidence ancestor is not a directory")
            _check_permissions(
                metadata,
                policy=policy,
                is_directory=True,
                acl_checker=acl_checker,
                fd=current_fd,
                label=str(root_path.joinpath(*parts[: index + 1])),
            )

        leaf = parts[-1]
        try:
            file_fd = os.open(leaf, _flags(), dir_fd=current_fd)
        except OSError as exc:
            raise EvidenceV2Error(f"evidence file open failed: {target}") from exc
        descriptors.append(file_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceV2Error("evidence leaf is not a regular file")
        if before.st_nlink != 1:
            raise EvidenceV2Error("evidence leaf must have exactly one hard link")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise EvidenceV2Error("evidence leaf size is outside its bound")
        _check_permissions(
            before,
            policy=policy,
            is_directory=False,
            acl_checker=acl_checker,
            fd=file_fd,
            label=str(target),
        )
        path_before = os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)
        if (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino):
            raise EvidenceV2Error("evidence path and descriptor identity mismatch")

        chunks: list[bytes] = []
        total = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise EvidenceV2Error("evidence file exceeded its read bound")
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(file_fd)
        path_after = os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)
        if _signature(before) != _signature(after):
            raise EvidenceV2Error("evidence descriptor changed during read")
        if _signature(path_before) != _signature(path_after):
            raise EvidenceV2Error("evidence path changed during read")
        if (path_after.st_dev, path_after.st_ino) != (after.st_dev, after.st_ino):
            raise EvidenceV2Error("evidence path was replaced during read")
        actual = digest.hexdigest()
        if actual != expected:
            raise EvidenceV2Error("evidence byte SHA mismatch")
        return BoundBytes(
            absolute_path=str(target),
            payload=b"".join(chunks),
            byte_sha256=actual,
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_bound_canonical_json(
    *,
    root: str | Path,
    reference: EvidenceRef,
    policy: RootPolicy,
    max_bytes: int = ARTIFACT_MAX_BYTES,
    acl_checker: _AclChecker | None = None,
) -> dict[str, Any]:
    return _load_bound_canonical_artifact(
        root=root,
        reference=reference,
        policy=policy,
        max_bytes=max_bytes,
        acl_checker=acl_checker,
    ).read()


def _load_bound_canonical_artifact(
    *,
    root: str | Path,
    reference: EvidenceRef,
    policy: RootPolicy,
    max_bytes: int,
    acl_checker: _AclChecker | None,
) -> BoundCanonicalArtifact:
    if reference.root_policy != policy.policy_id:
        raise EvidenceV2Error("EvidenceRef root policy mismatch")
    bound = _read_bound_bytes(
        root=root,
        path=reference.absolute_path,
        policy=policy,
        expected_sha256=reference.byte_sha256,
        max_bytes=max_bytes,
        acl_checker=acl_checker,
    )
    artifact = BoundCanonicalArtifact(reference=reference, payload=bound.payload)
    artifact.read()
    return artifact


def load_bound_canonical_artifact(
    *,
    root: str | Path,
    reference: EvidenceRef,
    policy: RootPolicy,
    max_bytes: int = ARTIFACT_MAX_BYTES,
) -> BoundCanonicalArtifact:
    """Load canonical evidence with the platform ACL verifier."""

    _require_canonical_policy(policy)
    return _load_bound_canonical_artifact(
        root=root,
        reference=reference,
        policy=policy,
        max_bytes=max_bytes,
        acl_checker=None,
    )


def load_bound_raw_artifact(
    *,
    root: str | Path,
    reference: EvidenceRef,
    policy: RootPolicy,
    max_bytes: int = ARTIFACT_MAX_BYTES,
) -> BoundRawArtifact:
    """Load opaque evidence bytes with the platform ACL verifier."""

    _require_canonical_policy(policy)
    if reference.root_policy != policy.policy_id:
        raise EvidenceV2Error("EvidenceRef root policy mismatch")
    bound = _read_bound_bytes(
        root=root,
        path=reference.absolute_path,
        policy=policy,
        expected_sha256=reference.byte_sha256,
        max_bytes=max_bytes,
        acl_checker=None,
    )
    return BoundRawArtifact(reference=reference, payload=bound.payload)


__all__ = [
    "GOVERNED_DATA_POLICY",
    "PRIVATE_EVIDENCE_POLICY",
    "RootPolicy",
    "TRUST_MATERIAL_POLICY",
    "load_bound_canonical_artifact",
    "load_bound_raw_artifact",
    "platform_acl_absent",
]
