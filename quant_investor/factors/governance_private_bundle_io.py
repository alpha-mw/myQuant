"""Shared fail-closed private I/O for active Factor governance bundles.

This module deliberately contains no artifact schemas and no governance state
machine.  A caller supplies the exact bundle inventory and pure callbacks for
canonical serialization, per-artifact validation, complete cross-validation,
and construction of the readback report.  This module owns only the private
filesystem transaction.

Publication is Darwin-only.  The one final commit operation is
``renameatx_np(RENAME_EXCL)``; there is no clobbering fallback.  Tests may
replace that primitive and use the explicitly test-only hooks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import ctypes
from dataclasses import dataclass, field
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any


FACTOR_PRIVATE_ROOT_PREFIX = (
    "reports",
    "factor_governance",
    "private",
)
LOCK_FILENAME = ".factor_governance_private_bundle.lock"
QUARANTINE_DIRECTORY = ".quarantine"
RENAME_EXCL = 0x00000004
DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_BUNDLE_BYTES = 256 * 1024 * 1024

_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

Canonicalizer = Callable[[Mapping[str, Any]], bytes]
ArtifactValidator = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
CompleteValidator = Callable[
    [Mapping[str, Mapping[str, Any]]],
    Mapping[str, Mapping[str, Any]],
]
ReadbackReportBuilder = Callable[..., Mapping[str, Any]]
TestFaultHook = Callable[[str], None]


class FactorGovernancePrivateBundleIOError(ValueError):
    """A Factor private read or publication was rejected fail closed."""

    def __init__(
        self,
        message: str,
        *,
        status: str = "REJECTED_FAIL_CLOSED",
    ) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.accepted = False


@dataclass(frozen=True)
class PrivateBundleContract:
    """Pure callbacks and exact inventory for one private bundle type.

    ``build_readback_report`` is called with keyword arguments ``run_id``,
    ``artifacts`` (the normalized input artifacts), and
    ``artifact_bindings`` (byte/file metadata for those inputs).
    ``validate_complete`` must return the exact normalized full inventory,
    including the readback report, without changing canonical bytes.
    """

    root_suffix: tuple[str, ...]
    input_filenames: tuple[str, ...]
    readback_report_filename: str
    canonicalize: Canonicalizer = field(repr=False)
    validate_artifact: ArtifactValidator = field(repr=False)
    validate_complete: CompleteValidator = field(repr=False)
    build_readback_report: ReadbackReportBuilder = field(repr=False)
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES

    @property
    def canonical_filenames(self) -> tuple[str, ...]:
        return (*self.input_filenames, self.readback_report_filename)


@dataclass(frozen=True)
class _StableBundle:
    values: dict[str, dict[str, Any]]
    bindings: tuple[dict[str, Any], ...]
    raw: dict[str, bytes]
    file_identities: dict[str, tuple[int, ...]]
    directory_identity: tuple[int, ...]


def _error(
    message: str,
    *,
    status: str = "REJECTED_FAIL_CLOSED",
) -> FactorGovernancePrivateBundleIOError:
    return FactorGovernancePrivateBundleIOError(message, status=status)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the repository's compact, sorted, finite JSON encoding."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error(f"value is not canonical finite JSON: {exc}") from exc


def canonical_json_file_bytes(value: Mapping[str, Any]) -> bytes:
    """Return canonical JSON file bytes with one trailing newline."""

    return canonical_json_bytes(value) + b"\n"


def _absolute_normalized_path(
    value: str | os.PathLike[str],
    label: str,
) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise _error(f"{label} must be an absolute path string") from exc
    if (
        type(raw) is not str
        or not raw.startswith("/")
        or "\x00" in raw
        or raw == "/"
    ):
        raise _error(f"{label} must be an absolute normalized path")
    components = raw.split("/")[1:]
    if not components or any(part in {"", ".", ".."} for part in components):
        raise _error(f"{label} must be an absolute normalized path")
    if os.path.abspath(raw) != raw:
        raise _error(f"{label} must not contain aliases or traversal")
    return Path(raw)


def _safe_segment(value: str, label: str) -> str:
    if (
        type(value) is not str
        or _SAFE_SEGMENT.fullmatch(value) is None
        or ".." in value
        or value.startswith(".")
    ):
        raise _error(f"{label} must be one safe path segment")
    return value


def _validate_contract(contract: PrivateBundleContract) -> None:
    if not isinstance(contract, PrivateBundleContract):
        raise _error("contract must be a PrivateBundleContract")
    suffix = contract.root_suffix
    if type(suffix) is not tuple or len(suffix) <= len(FACTOR_PRIVATE_ROOT_PREFIX):
        raise _error("contract root_suffix must identify one Factor private lane")
    if suffix[: len(FACTOR_PRIVATE_ROOT_PREFIX)] != FACTOR_PRIVATE_ROOT_PREFIX:
        raise _error("contract root_suffix must stay under Factor private reports")
    for index, component in enumerate(suffix):
        _safe_segment(component, f"contract root_suffix[{index}]")

    inputs = contract.input_filenames
    if type(inputs) is not tuple or not inputs:
        raise _error("contract input_filenames must be a non-empty tuple")
    for filename in inputs:
        _safe_segment(filename, "contract artifact filename")
    if len(set(inputs)) != len(inputs):
        raise _error("contract input_filenames must be unique")
    _safe_segment(
        contract.readback_report_filename,
        "contract readback report filename",
    )
    if contract.readback_report_filename in inputs:
        raise _error("readback report filename must be distinct from inputs")
    if LOCK_FILENAME in contract.canonical_filenames:
        raise _error("contract inventory collides with the publication lock")
    if QUARANTINE_DIRECTORY in contract.canonical_filenames:
        raise _error("contract inventory collides with quarantine")
    for label, callback in (
        ("canonicalize", contract.canonicalize),
        ("validate_artifact", contract.validate_artifact),
        ("validate_complete", contract.validate_complete),
        ("build_readback_report", contract.build_readback_report),
    ):
        if not callable(callback):
            raise _error(f"contract {label} must be callable")
    if (
        type(contract.max_artifact_bytes) is not int
        or contract.max_artifact_bytes <= 0
    ):
        raise _error("contract max_artifact_bytes must be a positive integer")
    if (
        type(contract.max_bundle_bytes) is not int
        or contract.max_bundle_bytes < contract.max_artifact_bytes
    ):
        raise _error(
            "contract max_bundle_bytes must be at least max_artifact_bytes"
        )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_read_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return int(left.st_dev) == int(right.st_dev) and int(left.st_ino) == int(
        right.st_ino
    )


def _check_private_directory(value: os.stat_result, label: str) -> None:
    if not stat.S_ISDIR(value.st_mode):
        raise _error(f"{label} must be a directory")
    if int(value.st_uid) != os.getuid():
        raise _error(f"{label} owner mismatch")
    if stat.S_IMODE(value.st_mode) != 0o700:
        raise _error(f"{label} mode must be 0700")


def _check_private_file(value: os.stat_result, label: str) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise _error(f"{label} must be a regular non-symlink file")
    if int(value.st_uid) != os.getuid():
        raise _error(f"{label} owner mismatch")
    if stat.S_IMODE(value.st_mode) != 0o600:
        raise _error(f"{label} mode must be 0600")
    if int(value.st_nlink) != 1:
        raise _error(f"{label} hard-link count must be one")


def _open_absolute_directory(path: Path, *, private_leaf: bool) -> int:
    """Open every absolute component relative to an anchored nofollow dirfd."""

    descriptor = os.open("/", _directory_flags())
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise _error(
                    f"directory traversal rejected at {component}: {exc}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        opened = os.fstat(descriptor)
        if private_leaf:
            _check_private_directory(opened, str(path))
        elif not stat.S_ISDIR(opened.st_mode):
            raise _error(f"{path} must be a directory")
        path_value = os.lstat(path)
        if stat.S_ISLNK(path_value.st_mode) or not _same_object(
            path_value,
            opened,
        ):
            raise _error(f"directory path identity mismatch: {path}")
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise _error(f"directory path readback failed: {path}: {exc}") from exc
    except Exception:
        os.close(descriptor)
        raise


def _assert_directory_current(
    path: Path,
    descriptor: int,
    *,
    private: bool,
    label: str,
) -> None:
    opened = os.fstat(descriptor)
    if private:
        _check_private_directory(opened, label)
    elif not stat.S_ISDIR(opened.st_mode):
        raise _error(f"{label} must be a directory")
    try:
        path_value = os.lstat(path)
    except OSError as exc:
        raise _error(f"{label} disappeared") from exc
    if stat.S_ISLNK(path_value.st_mode) or not _same_object(path_value, opened):
        raise _error(f"{label} path identity changed")
    if private:
        _check_private_directory(path_value, label)


def _validate_private_root(
    value: str | os.PathLike[str],
    contract: PrivateBundleContract,
) -> tuple[Path, int]:
    root = _absolute_normalized_path(value, "private_root")
    suffix = contract.root_suffix
    if tuple(root.parts[-len(suffix) :]) != suffix:
        raise _error(
            "private_root does not end in the contract Factor private suffix"
        )
    descriptor = _open_absolute_directory(root, private_leaf=True)
    _assert_directory_current(
        root,
        descriptor,
        private=True,
        label="publication root",
    )
    return root, descriptor


def _lstat_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _error(f"path diagnostic failed for {name}: {exc}") from exc


def _assert_absent(directory_fd: int, name: str, label: str) -> None:
    if _lstat_at(directory_fd, name) is not None:
        raise _error(f"{label} already exists")


def _test_fault(hook: TestFaultHook | None, point: str) -> None:
    if hook is None:
        return
    try:
        hook(point)
    except Exception as exc:
        raise _error(f"injected test fault at {point}: {exc}") from exc


def _fsync_fd(
    descriptor: int,
    *,
    label: str,
    test_fault_hook: TestFaultHook | None = None,
    point: str,
) -> None:
    _test_fault(test_fault_hook, f"{point}:before")
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise _error(f"{label} fsync failed: {exc}") from exc
    _test_fault(test_fault_hook, f"{point}:after")


def _open_lock(
    root: Path,
    root_fd: int,
    *,
    test_fault_hook: TestFaultHook | None,
) -> int:
    flags = (
        os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    created = False
    try:
        try:
            descriptor = os.open(
                LOCK_FILENAME,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_fd,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(LOCK_FILENAME, flags, dir_fd=root_fd)
    except OSError as exc:
        raise _error(f"publication lock open rejected: {exc}") from exc

    try:
        if created:
            os.fchmod(descriptor, 0o600)
            _fsync_fd(
                descriptor,
                label="publication lock",
                test_fault_hook=test_fault_hook,
                point="lock:file-fsync",
            )
            _fsync_fd(
                root_fd,
                label="publication root after lock creation",
                test_fault_hook=test_fault_hook,
                point="lock:root-fsync",
            )
        opened = os.fstat(descriptor)
        _check_private_file(opened, "publication lock")
        if int(opened.st_size) != 0:
            raise _error("publication lock must be empty")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        path_value = _lstat_at(root_fd, LOCK_FILENAME)
        opened = os.fstat(descriptor)
        if path_value is None or not _same_object(path_value, opened):
            raise _error("publication lock path identity changed")
        _check_private_file(path_value, "publication lock")
        if int(path_value.st_size) != 0:
            raise _error("publication lock must be empty")
        _assert_directory_current(
            root,
            root_fd,
            private=True,
            label="publication root",
        )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _require_exclusive_rename_support() -> None:
    if sys.platform != "darwin":
        raise _error("renameatx_np(RENAME_EXCL) requires Darwin")
    try:
        function = getattr(ctypes.CDLL(None, use_errno=True), "renameatx_np")
    except (AttributeError, OSError) as exc:
        raise _error("renameatx_np symbol is unavailable") from exc
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int


def _renameatx_np_exclusive(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Perform the sole allowed no-clobber directory commit operation."""

    if sys.platform != "darwin":
        raise _error("renameatx_np(RENAME_EXCL) requires Darwin")
    try:
        function = getattr(ctypes.CDLL(None, use_errno=True), "renameatx_np")
    except (AttributeError, OSError) as exc:
        raise _error("renameatx_np symbol is unavailable") from exc
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        RENAME_EXCL,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number))
    raise _error(
        "renameatx_np(RENAME_EXCL) rejected publication: "
        + os.strerror(error_number or errno.EIO)
    )


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def exact_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _error(f"{label} is not strict finite JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(f"{label} must be a JSON object")
    return value


def _canonical_bytes(
    contract: PrivateBundleContract,
    value: Mapping[str, Any],
    *,
    label: str,
) -> bytes:
    try:
        raw = contract.canonicalize(copy.deepcopy(dict(value)))
    except Exception as exc:
        raise _error(f"{label} canonical serialization failed: {exc}") from exc
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
    ):
        raise _error(
            f"{label} canonicalizer must return bytes with one trailing newline"
        )
    return raw


def _validate_artifact(
    contract: PrivateBundleContract,
    filename: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        normalized = contract.validate_artifact(
            filename,
            copy.deepcopy(dict(value)),
        )
    except Exception as exc:
        raise _error(f"{filename} artifact validation failed: {exc}") from exc
    if not isinstance(normalized, Mapping):
        raise _error(f"{filename} artifact validator returned a non-object")
    result = copy.deepcopy(dict(normalized))
    _canonical_bytes(contract, result, label=filename)
    return result


def _binding(
    filename: str,
    raw: bytes,
    metadata: os.stat_result,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode": stat.S_IMODE(metadata.st_mode),
        "uid": int(metadata.st_uid),
        "nlink": int(metadata.st_nlink),
    }


def _write_all(
    descriptor: int,
    raw: bytes,
    *,
    label: str,
    test_fault_hook: TestFaultHook | None,
) -> None:
    remaining = memoryview(raw)
    chunk_index = 0
    while remaining:
        _test_fault(
            test_fault_hook,
            f"write:{label}:chunk-{chunk_index}:before",
        )
        try:
            count = os.write(descriptor, remaining)
        except OSError as exc:
            raise _error(f"{label} write failed: {exc}") from exc
        if count <= 0:
            raise _error(f"{label} write made no progress")
        remaining = remaining[count:]
        _test_fault(
            test_fault_hook,
            f"write:{label}:chunk-{chunk_index}:after",
        )
        chunk_index += 1


def _write_private_artifact(
    directory_fd: int,
    filename: str,
    value: Mapping[str, Any],
    *,
    contract: PrivateBundleContract,
    test_fault_hook: TestFaultHook | None,
) -> dict[str, Any]:
    raw = _canonical_bytes(contract, value, label=filename)
    if len(raw) > contract.max_artifact_bytes:
        raise _error(f"{filename} exceeds the private artifact size limit")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise _error(f"{filename} exclusive creation failed: {exc}") from exc
    try:
        os.fchmod(descriptor, 0o600)
        _check_private_file(os.fstat(descriptor), filename)
        _write_all(
            descriptor,
            raw,
            label=filename,
            test_fault_hook=test_fault_hook,
        )
        _fsync_fd(
            descriptor,
            label=filename,
            test_fault_hook=test_fault_hook,
            point=f"file-fsync:{filename}",
        )
        metadata = os.fstat(descriptor)
        _check_private_file(metadata, filename)
        if int(metadata.st_size) != len(raw):
            raise _error(f"{filename} size changed while writing")
        path_value = _lstat_at(directory_fd, filename)
        if path_value is None or not _same_object(path_value, metadata):
            raise _error(f"{filename} path identity changed while writing")
        _check_private_file(path_value, filename)
        return _binding(filename, raw, metadata)
    except OSError as exc:
        raise _error(f"{filename} write/fsync failed: {exc}") from exc
    finally:
        os.close(descriptor)


def _read_private_file(
    directory_fd: int,
    filename: str,
    *,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    path_value = _lstat_at(directory_fd, filename)
    if path_value is None:
        raise _error(f"required artifact missing: {filename}")
    _check_private_file(path_value, filename)
    if int(path_value.st_size) > max_bytes:
        raise _error(f"{filename} exceeds the private artifact size limit")
    try:
        descriptor = os.open(filename, _file_read_flags(), dir_fd=directory_fd)
    except OSError as exc:
        raise _error(f"{filename} safe open failed: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        _check_private_file(before, filename)
        if not _same_object(path_value, before):
            raise _error(f"{filename} path/open identity mismatch")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise _error(f"{filename} exceeds the private artifact size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise _error(f"{filename} changed while reading")
        final_path = _lstat_at(directory_fd, filename)
        if final_path is None or _identity(path_value) != _identity(final_path):
            raise _error(f"{filename} path changed while reading")
        raw = b"".join(chunks)
        if len(raw) != int(after.st_size):
            raise _error(f"{filename} read length mismatch")
        return raw, after
    except OSError as exc:
        raise _error(f"{filename} read failed: {exc}") from exc
    finally:
        os.close(descriptor)


def _list_exact(directory_fd: int, expected: Sequence[str]) -> None:
    try:
        actual = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise _error(f"bundle directory listing failed: {exc}") from exc
    wanted = sorted(expected)
    if actual != wanted:
        missing = sorted(set(wanted) - set(actual))
        extra = sorted(set(actual) - set(wanted))
        raise _error(
            "bundle artifact set mismatch: "
            f"missing={','.join(missing) or '-'};"
            f"extra={','.join(extra) or '-'}"
        )


def _read_and_validate_files(
    directory_fd: int,
    filenames: Sequence[str],
    *,
    contract: PrivateBundleContract,
    phase: str,
    test_fault_hook: TestFaultHook | None,
) -> _StableBundle:
    _list_exact(directory_fd, filenames)
    first_directory_identity = _identity(os.fstat(directory_fd))
    values: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    first_raw: dict[str, bytes] = {}
    first_metadata: dict[str, tuple[int, ...]] = {}
    total = 0
    for filename in filenames:
        _test_fault(test_fault_hook, f"{phase}:pass-1:{filename}:before")
        raw, metadata = _read_private_file(
            directory_fd,
            filename,
            max_bytes=contract.max_artifact_bytes,
        )
        _test_fault(test_fault_hook, f"{phase}:pass-1:{filename}:after")
        total += len(raw)
        if total > contract.max_bundle_bytes:
            raise _error("bundle exceeds the private total size limit")
        parsed = _strict_json_object(raw, filename)
        normalized = _validate_artifact(contract, filename, parsed)
        if raw != _canonical_bytes(contract, normalized, label=filename):
            raise _error(f"{filename} is not exact canonical file bytes")
        values[filename] = normalized
        bindings.append(_binding(filename, raw, metadata))
        first_raw[filename] = raw
        first_metadata[filename] = _identity(metadata)

    for filename in filenames:
        _test_fault(test_fault_hook, f"{phase}:pass-2:{filename}:before")
        raw, metadata = _read_private_file(
            directory_fd,
            filename,
            max_bytes=contract.max_artifact_bytes,
        )
        _test_fault(test_fault_hook, f"{phase}:pass-2:{filename}:after")
        if raw != first_raw[filename]:
            raise _error(f"{filename} changed across stable readback passes")
        if _identity(metadata) != first_metadata[filename]:
            raise _error(f"{filename} identity changed across readback passes")
    _list_exact(directory_fd, filenames)
    final_directory_identity = _identity(os.fstat(directory_fd))
    if final_directory_identity != first_directory_identity:
        raise _error("bundle directory changed across stable readback passes")
    return _StableBundle(
        values=values,
        bindings=tuple(bindings),
        raw=first_raw,
        file_identities=first_metadata,
        directory_identity=final_directory_identity,
    )


def _build_readback_report(
    contract: PrivateBundleContract,
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        report = contract.build_readback_report(
            run_id=run_id,
            artifacts=copy.deepcopy(
                {name: dict(value) for name, value in artifacts.items()}
            ),
            artifact_bindings=tuple(
                copy.deepcopy(dict(binding)) for binding in artifact_bindings
            ),
        )
    except Exception as exc:
        raise _error(f"readback report build failed: {exc}") from exc
    if not isinstance(report, Mapping):
        raise _error("readback report builder returned a non-object")
    return _validate_artifact(
        contract,
        contract.readback_report_filename,
        report,
    )


def _verify_complete_snapshot(
    contract: PrivateBundleContract,
    *,
    run_id: str,
    snapshot: _StableBundle,
) -> dict[str, Any]:
    binding_by_name = {
        str(binding["filename"]): dict(binding)
        for binding in snapshot.bindings
    }
    input_values = {
        filename: snapshot.values[filename]
        for filename in contract.input_filenames
    }
    input_bindings = [
        binding_by_name[filename] for filename in contract.input_filenames
    ]
    expected_report = _build_readback_report(
        contract,
        run_id=run_id,
        artifacts=input_values,
        artifact_bindings=input_bindings,
    )
    actual_report = snapshot.values.get(contract.readback_report_filename)
    if actual_report is None:
        raise _error("readback report is missing")
    if _canonical_bytes(
        contract,
        actual_report,
        label=contract.readback_report_filename,
    ) != _canonical_bytes(
        contract,
        expected_report,
        label=contract.readback_report_filename,
    ):
        raise _error("readback report does not bind the live input artifacts")

    try:
        normalized = contract.validate_complete(
            copy.deepcopy(
                {name: dict(value) for name, value in snapshot.values.items()}
            )
        )
    except Exception as exc:
        raise _error(f"complete cross-validation failed: {exc}") from exc
    if not isinstance(normalized, Mapping):
        raise _error("complete cross-validator returned a non-object")
    if set(normalized) != set(contract.canonical_filenames):
        raise _error("complete cross-validator returned an incomplete inventory")
    for filename in contract.canonical_filenames:
        value = normalized.get(filename)
        if not isinstance(value, Mapping):
            raise _error(
                f"complete cross-validator returned invalid {filename}"
            )
        if _canonical_bytes(
            contract,
            value,
            label=filename,
        ) != _canonical_bytes(
            contract,
            snapshot.values[filename],
            label=filename,
        ):
            raise _error(
                f"complete cross-validator changed canonical {filename} bytes"
            )
    return expected_report


def _same_stable_snapshot(left: _StableBundle, right: _StableBundle) -> bool:
    return (
        left.raw == right.raw
        and left.file_identities == right.file_identities
        and left.directory_identity == right.directory_identity
    )


def _open_private_child_directory(
    parent_fd: int,
    name: str,
    label: str,
) -> int:
    path_value = _lstat_at(parent_fd, name)
    if path_value is None:
        raise _error(f"{label} is missing")
    if stat.S_ISLNK(path_value.st_mode):
        raise _error(f"{label} must not be a symlink")
    _check_private_directory(path_value, label)
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise _error(f"{label} open rejected: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        _check_private_directory(opened, label)
        if not _same_object(path_value, opened):
            raise _error(f"{label} path/open identity mismatch")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _assert_private_child_current(
    parent_fd: int,
    name: str,
    descriptor: int,
    *,
    label: str,
    expected_identity: tuple[int, ...] | None = None,
) -> None:
    opened = os.fstat(descriptor)
    _check_private_directory(opened, label)
    path_value = _lstat_at(parent_fd, name)
    if path_value is None or not _same_object(path_value, opened):
        raise _error(f"{label} path identity changed")
    _check_private_directory(path_value, label)
    if expected_identity is not None and _identity(opened) != expected_identity:
        raise _error(f"{label} identity changed after stable readback")


def _create_staging_directory(root_fd: int, run_id: str) -> tuple[str, int]:
    for _attempt in range(8):
        name = f".{run_id}.staging.{secrets.token_hex(12)}"
        try:
            os.mkdir(name, 0o700, dir_fd=root_fd)
        except FileExistsError:
            continue
        except OSError as exc:
            raise _error(f"staging directory creation failed: {exc}") from exc
        descriptor = _open_private_child_directory(
            root_fd,
            name,
            "staging directory",
        )
        return name, descriptor
    raise _error("could not allocate a unique staging directory")


def _open_or_create_quarantine(root_fd: int) -> int:
    value = _lstat_at(root_fd, QUARANTINE_DIRECTORY)
    if value is None:
        try:
            os.mkdir(QUARANTINE_DIRECTORY, 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise _error(f"quarantine creation failed: {exc}") from exc
    return _open_private_child_directory(
        root_fd,
        QUARANTINE_DIRECTORY,
        "quarantine directory",
    )


def _quarantine_directory(
    *,
    root_fd: int,
    source_name: str,
    run_id: str,
    reason: str,
    expected_source_identity: tuple[int, int] | None,
) -> str:
    quarantine_fd = _open_or_create_quarantine(root_fd)
    try:
        quarantine_opened = os.fstat(quarantine_fd)
        quarantine_path = _lstat_at(root_fd, QUARANTINE_DIRECTORY)
        if quarantine_path is None or not _same_object(
            quarantine_path,
            quarantine_opened,
        ):
            raise _error("quarantine directory path identity changed")
        _check_private_directory(quarantine_opened, "quarantine directory")
        source_before = _lstat_at(root_fd, source_name)
        if source_before is None:
            raise _error("quarantine source is missing")
        _check_private_directory(source_before, "quarantine source directory")
        source_identity = (
            int(source_before.st_dev),
            int(source_before.st_ino),
        )
        if (
            expected_source_identity is not None
            and source_identity != expected_source_identity
        ):
            raise _error(
                "quarantine source identity differs from the owned directory"
            )
        for _attempt in range(8):
            destination = f"{run_id}.{reason}.{secrets.token_hex(12)}"
            try:
                _renameatx_np_exclusive(
                    root_fd,
                    source_name,
                    quarantine_fd,
                    destination,
                )
            except FileExistsError:
                continue
            os.fsync(quarantine_fd)
            os.fsync(root_fd)
            quarantine_path = _lstat_at(root_fd, QUARANTINE_DIRECTORY)
            if quarantine_path is None or not _same_object(
                quarantine_path,
                quarantine_opened,
            ):
                raise _error("quarantine directory path identity changed")
            _check_private_directory(quarantine_path, "quarantine directory")
            if _lstat_at(root_fd, source_name) is not None:
                raise _error("quarantined source remains visible at its old path")
            moved = _lstat_at(quarantine_fd, destination)
            if moved is None:
                raise _error("quarantined directory is missing after recovery")
            _check_private_directory(moved, "quarantined directory")
            if (int(moved.st_dev), int(moved.st_ino)) != source_identity:
                raise _error(
                    "quarantined directory identity changed during recovery"
                )
            return destination
        raise _error("could not allocate a unique quarantine destination")
    except OSError as exc:
        raise _error(f"quarantine durability failed: {exc}") from exc
    finally:
        os.close(quarantine_fd)


def _descriptors(
    bundle_path: Path,
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(binding["filename"]): {
            "absolute_path": str(bundle_path / str(binding["filename"])),
            "byte_sha256": str(binding["byte_sha256"]),
            "size_bytes": int(binding["size_bytes"]),
            "mode": int(binding["mode"]),
            "uid": int(binding["uid"]),
            "nlink": int(binding["nlink"]),
        }
        for binding in bindings
    }


def read_private_canonical_json(
    path: str | os.PathLike[str],
    expected_sha256: str,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    canonicalizer: Canonicalizer = canonical_json_bytes,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """Stably read one explicit owner-private canonical JSON file.

    Every ancestor is opened from ``/`` through anchored ``O_NOFOLLOW``
    dirfds.  The leaf must be owner ``0600``, regular, and have one hard link.
    The return object contains ``value`` and an exact ``descriptor``.
    """

    target = _absolute_normalized_path(path, "path")
    filename = _safe_segment(target.name, "private JSON filename")
    if type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None:
        raise _error("expected_sha256 must be lowercase SHA-256")
    if not callable(validator):
        raise _error("validator must be callable")
    if not callable(canonicalizer):
        raise _error("canonicalizer must be callable")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise _error("max_bytes must be a positive integer")

    parent_fd = _open_absolute_directory(target.parent, private_leaf=False)
    try:
        first_raw, first_metadata = _read_private_file(
            parent_fd,
            filename,
            max_bytes=max_bytes,
        )
        second_raw, second_metadata = _read_private_file(
            parent_fd,
            filename,
            max_bytes=max_bytes,
        )
        if first_raw != second_raw:
            raise _error("private JSON changed across stable readback passes")
        if _identity(first_metadata) != _identity(second_metadata):
            raise _error(
                "private JSON identity changed across stable readback passes"
            )
        _assert_directory_current(
            target.parent,
            parent_fd,
            private=False,
            label="private JSON parent",
        )
        try:
            path_value = os.lstat(target)
        except OSError as exc:
            raise _error("private JSON path disappeared") from exc
        if stat.S_ISLNK(path_value.st_mode) or _identity(path_value) != _identity(
            second_metadata
        ):
            raise _error("private JSON absolute path identity changed")
        actual_sha256 = hashlib.sha256(first_raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise _error("private JSON byte SHA-256 mismatch")
        parsed = _strict_json_object(first_raw, filename)
        try:
            normalized = validator(copy.deepcopy(parsed))
        except Exception as exc:
            raise _error(f"private JSON validation failed: {exc}") from exc
        if not isinstance(normalized, Mapping):
            raise _error("private JSON validator returned a non-object")
        result = copy.deepcopy(dict(normalized))
        try:
            expected_raw = canonicalizer(copy.deepcopy(result))
        except Exception as exc:
            raise _error(
                f"private JSON canonical serialization failed: {exc}"
            ) from exc
        if type(expected_raw) is not bytes or first_raw != expected_raw:
            raise _error("private JSON is not exact canonical file bytes")
        return {
            "value": result,
            "descriptor": {
                "absolute_path": str(target),
                "byte_sha256": actual_sha256,
                "size_bytes": len(first_raw),
                "mode": stat.S_IMODE(second_metadata.st_mode),
                "uid": int(second_metadata.st_uid),
                "nlink": int(second_metadata.st_nlink),
            },
        }
    finally:
        os.close(parent_fd)


def readback_private_bundle(
    bundle_path: str | os.PathLike[str],
    *,
    contract: PrivateBundleContract,
    _test_fault_hook: TestFaultHook | None = None,
) -> dict[str, Any]:
    """Strictly and stably read one already-published private bundle."""

    _validate_contract(contract)
    if _test_fault_hook is not None and not callable(_test_fault_hook):
        raise _error("_test_fault_hook must be callable")
    path = _absolute_normalized_path(bundle_path, "bundle_path")
    run_id = _safe_segment(path.name, "run_id")
    root, root_fd = _validate_private_root(path.parent, contract)
    bundle_fd: int | None = None
    try:
        _assert_directory_current(
            root,
            root_fd,
            private=True,
            label="publication root",
        )
        bundle_fd = _open_private_child_directory(
            root_fd,
            run_id,
            "canonical private bundle",
        )
        snapshot = _read_and_validate_files(
            bundle_fd,
            contract.canonical_filenames,
            contract=contract,
            phase="canonical-readback",
            test_fault_hook=_test_fault_hook,
        )
        report = _verify_complete_snapshot(
            contract,
            run_id=run_id,
            snapshot=snapshot,
        )
        _assert_private_child_current(
            root_fd,
            run_id,
            bundle_fd,
            label="canonical private bundle",
            expected_identity=snapshot.directory_identity,
        )
        _assert_directory_current(
            root,
            root_fd,
            private=True,
            label="publication root",
        )
        return {
            "accepted": True,
            "bundle_path": str(path),
            "artifacts": {
                name: copy.deepcopy(dict(value))
                for name, value in snapshot.values.items()
            },
            "artifact_descriptors": _descriptors(path, snapshot.bindings),
            "readback_report": copy.deepcopy(report),
        }
    finally:
        if bundle_fd is not None:
            os.close(bundle_fd)
        os.close(root_fd)


def publish_private_bundle(
    *,
    private_root: str | os.PathLike[str],
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    contract: PrivateBundleContract,
    revalidate_inputs: Callable[[], None],
    _test_fault_hook: TestFaultHook | None = None,
    _test_race_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Atomically publish one complete, owner-private, no-clobber bundle.

    Test hooks are intentionally prefixed ``_test_`` and are not part of a
    production workflow.  ``revalidate_inputs`` runs under the secured flock
    immediately before the final absence check and exclusive directory rename.
    """

    _validate_contract(contract)
    _require_exclusive_rename_support()
    normalized_run_id = _safe_segment(run_id, "run_id")
    if not callable(revalidate_inputs):
        raise _error("revalidate_inputs must be a zero-argument callable")
    if _test_fault_hook is not None and not callable(_test_fault_hook):
        raise _error("_test_fault_hook must be callable")
    if _test_race_hook is not None and not callable(_test_race_hook):
        raise _error("_test_race_hook must be callable")
    if not isinstance(artifacts, Mapping):
        raise _error("artifacts must be an exact filename-to-object mapping")
    if any(type(filename) is not str for filename in artifacts):
        raise _error("artifact filenames must be exact strings")
    if set(artifacts) != set(contract.input_filenames):
        missing = sorted(set(contract.input_filenames) - set(artifacts))
        extra = sorted(set(artifacts) - set(contract.input_filenames))
        raise _error(
            "publication input artifact set mismatch: "
            f"missing={','.join(missing) or '-'};"
            f"extra={','.join(extra) or '-'}"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for filename in contract.input_filenames:
        value = artifacts[filename]
        if not isinstance(value, Mapping):
            raise _error(f"{filename} publication value must be an object")
        normalized[filename] = _validate_artifact(contract, filename, value)

    root, root_fd = _validate_private_root(private_root, contract)
    lock_fd: int | None = None
    staging_fd: int | None = None
    staging_name: str | None = None
    staging_identity: tuple[int, int] | None = None
    committed = False
    try:
        lock_fd = _open_lock(
            root,
            root_fd,
            test_fault_hook=_test_fault_hook,
        )
        _assert_absent(root_fd, normalized_run_id, "canonical private bundle")
        staging_name, staging_fd = _create_staging_directory(
            root_fd,
            normalized_run_id,
        )
        opened_staging = os.fstat(staging_fd)
        staging_identity = (
            int(opened_staging.st_dev),
            int(opened_staging.st_ino),
        )
        _check_private_directory(opened_staging, "staging directory")
        _fsync_fd(
            root_fd,
            label="publication root after staging creation",
            test_fault_hook=_test_fault_hook,
            point="staging-created:root-fsync",
        )

        for filename in contract.input_filenames:
            _write_private_artifact(
                staging_fd,
                filename,
                normalized[filename],
                contract=contract,
                test_fault_hook=_test_fault_hook,
            )
        _fsync_fd(
            staging_fd,
            label="input staging directory",
            test_fault_hook=_test_fault_hook,
            point="staging-input:directory-fsync",
        )
        input_snapshot = _read_and_validate_files(
            staging_fd,
            contract.input_filenames,
            contract=contract,
            phase="staging-input",
            test_fault_hook=_test_fault_hook,
        )
        report = _build_readback_report(
            contract,
            run_id=normalized_run_id,
            artifacts=input_snapshot.values,
            artifact_bindings=input_snapshot.bindings,
        )
        _write_private_artifact(
            staging_fd,
            contract.readback_report_filename,
            report,
            contract=contract,
            test_fault_hook=_test_fault_hook,
        )
        _fsync_fd(
            staging_fd,
            label="complete staging directory",
            test_fault_hook=_test_fault_hook,
            point="staging-complete:directory-fsync",
        )
        complete_snapshot = _read_and_validate_files(
            staging_fd,
            contract.canonical_filenames,
            contract=contract,
            phase="staging-complete",
            test_fault_hook=_test_fault_hook,
        )
        _verify_complete_snapshot(
            contract,
            run_id=normalized_run_id,
            snapshot=complete_snapshot,
        )
        _fsync_fd(
            root_fd,
            label="publication root before commit",
            test_fault_hook=_test_fault_hook,
            point="precommit:root-fsync",
        )

        try:
            revalidate_inputs()
        except Exception as exc:
            raise _error(f"input revalidation failed: {exc}") from exc
        _assert_directory_current(
            root,
            root_fd,
            private=True,
            label="publication root",
        )
        lock_path = _lstat_at(root_fd, LOCK_FILENAME)
        if lock_path is None or not _same_object(lock_path, os.fstat(lock_fd)):
            raise _error("publication lock changed before commit")
        _check_private_file(lock_path, "publication lock")
        _assert_absent(root_fd, normalized_run_id, "canonical private bundle")
        if _test_race_hook is not None:
            _test_race_hook()

        _assert_directory_current(
            root,
            root_fd,
            private=True,
            label="publication root",
        )
        staging_opened = os.fstat(staging_fd)
        _check_private_directory(staging_opened, "staging directory")
        staging_path = _lstat_at(root_fd, staging_name)
        if staging_path is None or not _same_object(staging_path, staging_opened):
            raise _error("staging directory path identity changed before commit")
        if (
            int(staging_opened.st_dev),
            int(staging_opened.st_ino),
        ) != staging_identity:
            raise _error("open staging directory identity changed before commit")
        final_snapshot = _read_and_validate_files(
            staging_fd,
            contract.canonical_filenames,
            contract=contract,
            phase="precommit-stable",
            test_fault_hook=_test_fault_hook,
        )
        _verify_complete_snapshot(
            contract,
            run_id=normalized_run_id,
            snapshot=final_snapshot,
        )
        if not _same_stable_snapshot(complete_snapshot, final_snapshot):
            raise _error("staging bundle changed after complete validation")
        if staging_name is None:
            raise _error("staging directory name disappeared before commit")
        _assert_private_child_current(
            root_fd,
            staging_name,
            staging_fd,
            label="staging directory",
            expected_identity=final_snapshot.directory_identity,
        )

        _test_fault(_test_fault_hook, "commit:rename:before")
        rename_error: Exception | None = None
        commit_source_name = staging_name
        try:
            _renameatx_np_exclusive(
                root_fd,
                commit_source_name,
                root_fd,
                normalized_run_id,
            )
        except FileExistsError as exc:
            destination = _lstat_at(root_fd, normalized_run_id)
            destination_is_staging = (
                destination is not None
                and staging_identity is not None
                and (
                    int(destination.st_dev),
                    int(destination.st_ino),
                )
                == staging_identity
            )
            if destination_is_staging:
                committed = True
                staging_name = None
                rename_error = exc
            else:
                raise _error(
                    "canonical private bundle appeared during exclusive commit"
                ) from exc
        except Exception as exc:
            destination = _lstat_at(root_fd, normalized_run_id)
            destination_is_staging = (
                destination is not None
                and staging_identity is not None
                and (
                    int(destination.st_dev),
                    int(destination.st_ino),
                )
                == staging_identity
            )
            if destination_is_staging:
                committed = True
                staging_name = None
                rename_error = exc
            else:
                raise
        else:
            destination = _lstat_at(root_fd, normalized_run_id)
            destination_is_staging = (
                destination is not None
                and staging_identity is not None
                and (
                    int(destination.st_dev),
                    int(destination.st_ino),
                )
                == staging_identity
            )
            if not destination_is_staging:
                unexpected_identity = (
                    (
                        int(destination.st_dev),
                        int(destination.st_ino),
                    )
                    if destination is not None
                    else None
                )
                try:
                    _renameatx_np_exclusive(
                        root_fd,
                        normalized_run_id,
                        root_fd,
                        commit_source_name,
                    )
                    os.fsync(root_fd)
                except Exception as recovery_exc:
                    if unexpected_identity is None:
                        raise _error(
                            "exclusive commit outcome is missing; rollback "
                            f"was not proven: {recovery_exc}",
                            status="AMBIGUOUS_DURABILITY_FAIL_CLOSED",
                        ) from recovery_exc
                    try:
                        _quarantine_directory(
                            root_fd=root_fd,
                            source_name=normalized_run_id,
                            run_id=normalized_run_id,
                            reason="unexpected-commit",
                            expected_source_identity=unexpected_identity,
                        )
                    except Exception as quarantine_exc:
                        raise _error(
                            "exclusive commit moved an unexpected directory; "
                            "neither rollback nor identity-bound quarantine "
                            f"was proven: rollback={recovery_exc}; "
                            f"quarantine={quarantine_exc}",
                            status="AMBIGUOUS_DURABILITY_FAIL_CLOSED",
                        ) from quarantine_exc
                    raise _error(
                        "exclusive commit moved an unexpected directory; "
                        "rollback failed and the canonical path was "
                        f"identity-quarantined: {recovery_exc}",
                        status="POSTCOMMIT_RECOVERED_FAIL_CLOSED",
                    ) from recovery_exc
                raise _error(
                    "staging source identity changed during exclusive commit"
                )
            committed = True
            staging_name = None
        try:
            _test_fault(_test_fault_hook, "commit:rename:after")
            if rename_error is not None:
                raise _error(
                    f"exclusive commit returned an uncertain error: {rename_error}"
                )
            canonical_path = _lstat_at(root_fd, normalized_run_id)
            if canonical_path is None:
                raise _error("canonical bundle missing immediately after commit")
            _check_private_directory(canonical_path, "canonical private bundle")
            if staging_identity is None or (
                int(canonical_path.st_dev),
                int(canonical_path.st_ino),
            ) != staging_identity:
                raise _error(
                    "canonical bundle identity differs from committed staging inode"
                )
            if not _same_object(canonical_path, os.fstat(staging_fd)):
                raise _error(
                    "canonical bundle does not match the open committed directory"
                )
            os.close(staging_fd)
            staging_fd = None
            _fsync_fd(
                root_fd,
                label="publication root after commit",
                test_fault_hook=_test_fault_hook,
                point="commit:root-fsync",
            )
            result = readback_private_bundle(
                root / normalized_run_id,
                contract=contract,
                _test_fault_hook=_test_fault_hook,
            )
            if not isinstance(result, Mapping) or result.get("accepted") is not True:
                raise _error("live canonical readback was not accepted")
        except Exception as postcommit_exc:
            try:
                _quarantine_directory(
                    root_fd=root_fd,
                    source_name=normalized_run_id,
                    run_id=normalized_run_id,
                    reason="postcommit",
                    expected_source_identity=staging_identity,
                )
            except Exception as recovery_exc:
                raise _error(
                    "post-commit durability/readback failed and recovery was "
                    f"not durably proven: {postcommit_exc}; "
                    f"recovery={recovery_exc}",
                    status="AMBIGUOUS_DURABILITY_FAIL_CLOSED",
                ) from recovery_exc
            raise _error(
                "post-commit durability/readback failed; canonical bundle was "
                f"quarantined: {postcommit_exc}",
                status="POSTCOMMIT_RECOVERED_FAIL_CLOSED",
            ) from postcommit_exc
        return dict(result)
    except Exception:
        if staging_fd is not None:
            os.close(staging_fd)
            staging_fd = None
        if not committed and staging_name is not None:
            try:
                _quarantine_directory(
                    root_fd=root_fd,
                    source_name=staging_name,
                    run_id=normalized_run_id,
                    reason="staging-failed",
                    expected_source_identity=staging_identity,
                )
                staging_name = None
            except Exception:
                # Preserve the primary rejection.  A hidden staging path is
                # never a final or accepted bundle.
                pass
        raise
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(root_fd)


__all__ = [
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "DEFAULT_MAX_BUNDLE_BYTES",
    "FACTOR_PRIVATE_ROOT_PREFIX",
    "FactorGovernancePrivateBundleIOError",
    "LOCK_FILENAME",
    "PrivateBundleContract",
    "QUARANTINE_DIRECTORY",
    "canonical_json_bytes",
    "canonical_json_file_bytes",
    "publish_private_bundle",
    "read_private_canonical_json",
    "readback_private_bundle",
]
