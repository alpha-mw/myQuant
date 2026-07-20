"""Fail-closed private publication for v4.1 DISCOVERY evidence.

The module owns only the final owner-private directory transaction.  Artifact
schemas and semantic hashes remain the responsibility of
``governance_discovery_v4_1``.  Publication is deliberately Darwin-only because
the required no-clobber directory commit uses ``renameatx_np(RENAME_EXCL)``;
there is no ``rename``/``replace`` fallback.

Nothing in this module discovers or reads market data, registries, result
trees, providers, portfolios, brokers, orders, or trades.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import ctypes
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

from quant_investor.factors import governance_discovery_v4_1 as _core


AQUANT_SOURCE_RECEIPT_FILENAME = "aquant_source_receipt.v4_1.json"
SOURCE_IDEA_AUDIT_FILENAME = "source_idea_audit.v4_1.json"
LOCAL_COMPATIBILITY_CONTRACT_FILENAME = (
    "local_compatibility_contract.v4_1.json"
)
DISCOVERY_CATALOG_FILENAME = "discovery_catalog.v4_1.json"
STRUCTURAL_COLLISION_AUDIT_FILENAME = "structural_collision_audit.v4_1.json"
DISCOVERY_SOURCE_NODE_FILENAME = "discovery_source_node.v4_1.json"
DISCOVERY_CYCLE_STATE_FILENAME = "cycle_state.discovery.v4_1.json"
DISCOVERY_READBACK_REPORT_FILENAME = "discovery_readback_report.v4_1.json"

INPUT_ARTIFACT_FILENAMES = (
    AQUANT_SOURCE_RECEIPT_FILENAME,
    SOURCE_IDEA_AUDIT_FILENAME,
    LOCAL_COMPATIBILITY_CONTRACT_FILENAME,
    DISCOVERY_CATALOG_FILENAME,
    STRUCTURAL_COLLISION_AUDIT_FILENAME,
    DISCOVERY_SOURCE_NODE_FILENAME,
    DISCOVERY_CYCLE_STATE_FILENAME,
)
CANONICAL_ARTIFACT_FILENAMES = (
    *INPUT_ARTIFACT_FILENAMES,
    DISCOVERY_READBACK_REPORT_FILENAME,
)

PRIVATE_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_1_cycle",
)
LOCK_FILENAME = ".factor_v4_1_discovery.lock"
QUARANTINE_DIRECTORY = ".quarantine"
RENAME_EXCL = 0x00000004
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024

FIXED_SIDE_EFFECTS: dict[str, bool] = {
    "registry": False,
    "wal": False,
    "budget": False,
    "production_receipt": False,
    "production_pointer": False,
    "proposal": False,
    "apply": False,
    "portfolio": False,
    "live_provider": False,
    "broker": False,
    "order": False,
    "trade": False,
    "network": False,
}

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_SEMANTIC_IDENTITY_FIELD_BY_FILENAME = {
    AQUANT_SOURCE_RECEIPT_FILENAME: "receipt_semantic_sha256",
    SOURCE_IDEA_AUDIT_FILENAME: "audit_semantic_sha256",
    LOCAL_COMPATIBILITY_CONTRACT_FILENAME: "contract_semantic_sha256",
    DISCOVERY_CATALOG_FILENAME: "catalog_semantic_sha256",
    STRUCTURAL_COLLISION_AUDIT_FILENAME: "audit_semantic_sha256",
    DISCOVERY_SOURCE_NODE_FILENAME: "semantic_sha256",
    DISCOVERY_CYCLE_STATE_FILENAME: "state_semantic_sha256",
    DISCOVERY_READBACK_REPORT_FILENAME: "report_semantic_sha256",
}


class FactorGovernanceDiscoveryReadbackV4_1Error(ValueError):
    """A private DISCOVERY transaction was rejected fail closed."""

    def __init__(
        self,
        message: str,
        *,
        status: str = "REJECTED_FAIL_CLOSED",
    ) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.accepted = False


def _error(
    message: str,
    *,
    status: str = "REJECTED_FAIL_CLOSED",
) -> FactorGovernanceDiscoveryReadbackV4_1Error:
    return FactorGovernanceDiscoveryReadbackV4_1Error(
        message,
        status=status,
    )


def _absolute_normalized_path(value: str | os.PathLike[str], label: str) -> Path:
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
    normalized = os.path.abspath(raw)
    if normalized != raw:
        raise _error(f"{label} must not contain aliases or traversal")
    return Path(raw)


def _safe_run_id(value: str) -> str:
    if (
        type(value) is not str
        or _SAFE_RUN_ID.fullmatch(value) is None
        or ".." in value
        or value.startswith(".")
    ):
        raise _error("run_id must be one safe non-hidden path segment")
    return value


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


def _directory_object_identity(value: os.stat_result) -> tuple[int, int]:
    _check_private_directory(value, "private directory")
    return int(value.st_dev), int(value.st_ino)


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
    """Open every component with ``O_NOFOLLOW`` and return the leaf dirfd."""

    fd = os.open("/", _directory_flags())
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=fd)
            except OSError as exc:
                raise _error(
                    f"directory traversal rejected at {component}: {exc.strerror}"
                ) from exc
            os.close(fd)
            fd = child
        opened = os.fstat(fd)
        if private_leaf:
            _check_private_directory(opened, str(path))
        else:
            if not stat.S_ISDIR(opened.st_mode):
                raise _error(f"{path} must be a directory")
        try:
            path_value = os.lstat(path)
        except OSError as exc:
            raise _error(f"directory path readback failed: {path}") from exc
        if stat.S_ISLNK(path_value.st_mode) or not _same_object(
            path_value, opened
        ):
            raise _error(f"directory path identity mismatch: {path}")
        return fd
    except Exception:
        os.close(fd)
        raise


def _assert_root_current(root: Path, root_fd: int) -> None:
    opened = os.fstat(root_fd)
    _check_private_directory(opened, "publication root")
    try:
        path_value = os.lstat(root)
    except OSError as exc:
        raise _error("publication root disappeared") from exc
    if stat.S_ISLNK(path_value.st_mode) or not _same_object(path_value, opened):
        raise _error("publication root path identity changed")
    _check_private_directory(path_value, "publication root")


def _validate_private_root(value: str | os.PathLike[str]) -> tuple[Path, int]:
    root = _absolute_normalized_path(value, "private_root")
    if tuple(root.parts[-len(PRIVATE_ROOT_SUFFIX) :]) != PRIVATE_ROOT_SUFFIX:
        raise _error(
            "private_root must end in reports/factor_governance/private/v4_1_cycle"
        )
    root_fd = _open_absolute_directory(root, private_leaf=True)
    _assert_root_current(root, root_fd)
    return root, root_fd


def _lstat_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _error(f"path diagnostic failed for {name}: {exc.strerror}") from exc


def _assert_absent(directory_fd: int, name: str, label: str) -> None:
    if _lstat_at(directory_fd, name) is not None:
        raise _error(f"{label} already exists")


def _open_lock(root: Path, root_fd: int) -> int:
    flags = (
        os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    created = False
    try:
        try:
            lock_fd = os.open(
                LOCK_FILENAME,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_fd,
            )
            created = True
        except FileExistsError:
            lock_fd = os.open(LOCK_FILENAME, flags, dir_fd=root_fd)
    except OSError as exc:
        raise _error(f"publication lock open rejected: {exc.strerror}") from exc

    try:
        if created:
            os.fchmod(lock_fd, 0o600)
            os.fsync(lock_fd)
            os.fsync(root_fd)
        _check_private_file(os.fstat(lock_fd), "publication lock")
        if int(os.fstat(lock_fd).st_size) != 0:
            raise _error("publication lock must be empty")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        path_value = _lstat_at(root_fd, LOCK_FILENAME)
        opened = os.fstat(lock_fd)
        if path_value is None or not _same_object(path_value, opened):
            raise _error("publication lock path identity changed")
        _check_private_file(path_value, "publication lock")
        if int(path_value.st_size) != 0:
            raise _error("publication lock must be empty")
        _assert_root_current(root, root_fd)
        return lock_fd
    except Exception:
        os.close(lock_fd)
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
    """Perform the sole allowed directory commit operation."""

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
    def reject_constant(value: str) -> None:
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


def _validate_core_filename_set() -> None:
    declared = getattr(_core, "CANONICAL_ARTIFACT_FILENAMES", None)
    if declared is None or tuple(declared) != CANONICAL_ARTIFACT_FILENAMES:
        raise _error("core/publication canonical artifact set mismatch")


def _validate_artifact(filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = _core.validate_discovery_artifact_v4_1(filename, value)
    except (TypeError, ValueError) as exc:
        raise _error(f"{filename} core validation failed: {exc}") from exc
    if not isinstance(normalized, Mapping):
        raise _error(f"{filename} core validator returned a non-object")
    result = dict(normalized)
    try:
        raw = _core.canonical_file_bytes(result)
    except (TypeError, ValueError) as exc:
        raise _error(f"{filename} is not canonical finite JSON: {exc}") from exc
    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise _error("core canonical_file_bytes contract mismatch")
    return result


def _binding_from_raw(
    filename: str,
    value: Mapping[str, Any],
    raw: bytes,
    metadata: os.stat_result,
) -> dict[str, Any]:
    semantic = _authoritative_semantic_sha256(filename, value)
    byte_digest = hashlib.sha256(raw).hexdigest()
    return {
        "filename": filename,
        "byte_sha256": byte_digest,
        "semantic_sha256": semantic,
        "size_bytes": len(raw),
        "mode": stat.S_IMODE(metadata.st_mode),
        "uid": int(metadata.st_uid),
        "nlink": int(metadata.st_nlink),
    }


def _authoritative_semantic_sha256(
    filename: str,
    value: Mapping[str, Any],
) -> str:
    field = _SEMANTIC_IDENTITY_FIELD_BY_FILENAME.get(filename)
    if field is None:
        raise _error(f"no authoritative semantic identity field for {filename}")
    observed = value.get(field)
    if type(observed) is not str or _SHA256.fullmatch(observed) is None:
        raise _error(f"{filename}.{field} must be lowercase SHA-256")
    semantic_payload = {
        key: item for key, item in value.items() if key != field
    }
    try:
        expected = _core.semantic_sha256(semantic_payload)
    except (TypeError, ValueError) as exc:
        raise _error(f"{filename} semantic hash failed: {exc}") from exc
    if observed != expected:
        raise _error(f"{filename}.{field} does not seal the validated artifact")
    return observed


def _write_all(descriptor: int, raw: bytes, label: str) -> None:
    remaining = memoryview(raw)
    while remaining:
        try:
            count = os.write(descriptor, remaining)
        except OSError as exc:
            raise _error(f"{label} write failed: {exc.strerror}") from exc
        if count <= 0:
            raise _error(f"{label} write made no progress")
        remaining = remaining[count:]


def _write_private_artifact(
    directory_fd: int,
    filename: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        raw = _core.canonical_file_bytes(value)
    except (TypeError, ValueError) as exc:
        raise _error(f"{filename} serialization failed: {exc}") from exc
    if len(raw) > MAX_ARTIFACT_BYTES:
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
        raise _error(f"{filename} exclusive creation failed: {exc.strerror}") from exc
    try:
        os.fchmod(descriptor, 0o600)
        _check_private_file(os.fstat(descriptor), filename)
        _write_all(descriptor, raw, filename)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        _check_private_file(metadata, filename)
        if int(metadata.st_size) != len(raw):
            raise _error(f"{filename} size changed while writing")
        path_value = _lstat_at(directory_fd, filename)
        if path_value is None or not _same_object(path_value, metadata):
            raise _error(f"{filename} path identity changed while writing")
        _check_private_file(path_value, filename)
        return _binding_from_raw(filename, value, raw, metadata)
    except OSError as exc:
        raise _error(f"{filename} write/fsync failed: {exc.strerror}") from exc
    finally:
        os.close(descriptor)


def _read_private_file(
    directory_fd: int,
    filename: str,
) -> tuple[bytes, os.stat_result]:
    path_value = _lstat_at(directory_fd, filename)
    if path_value is None:
        raise _error(f"required artifact missing: {filename}")
    _check_private_file(path_value, filename)
    if int(path_value.st_size) > MAX_ARTIFACT_BYTES:
        raise _error(f"{filename} exceeds the private artifact size limit")
    try:
        descriptor = os.open(filename, _file_read_flags(), dir_fd=directory_fd)
    except OSError as exc:
        raise _error(f"{filename} safe open failed: {exc.strerror}") from exc
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
            if total > MAX_ARTIFACT_BYTES:
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
        raise _error(f"{filename} read failed: {exc.strerror}") from exc
    finally:
        os.close(descriptor)


def _list_exact(directory_fd: int, expected: Sequence[str]) -> None:
    try:
        actual = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise _error(f"bundle directory listing failed: {exc.strerror}") from exc
    wanted = sorted(expected)
    if actual != wanted:
        missing = sorted(set(wanted) - set(actual))
        extra = sorted(set(actual) - set(wanted))
        raise _error(
            "bundle artifact set mismatch: "
            f"missing={','.join(missing) or '-'};extra={','.join(extra) or '-'}"
        )


def _read_and_validate_files(
    directory_fd: int,
    filenames: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    _list_exact(directory_fd, filenames)
    first_directory_identity = _identity(os.fstat(directory_fd))
    values: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    first_raw: dict[str, bytes] = {}
    first_metadata: dict[str, tuple[int, ...]] = {}
    total = 0
    for filename in filenames:
        raw, metadata = _read_private_file(directory_fd, filename)
        total += len(raw)
        if total > MAX_BUNDLE_BYTES:
            raise _error("bundle exceeds the private total size limit")
        parsed = _strict_json_object(raw, filename)
        normalized = _validate_artifact(filename, parsed)
        expected_raw = _core.canonical_file_bytes(normalized)
        if raw != expected_raw:
            raise _error(f"{filename} is not exact canonical file bytes")
        values[filename] = normalized
        bindings.append(_binding_from_raw(filename, normalized, raw, metadata))
        first_raw[filename] = raw
        first_metadata[filename] = _identity(metadata)

    # A second complete pass makes a successful readback a stable observation,
    # not merely a set of individually stable file reads.
    for filename in filenames:
        raw, metadata = _read_private_file(directory_fd, filename)
        if raw != first_raw[filename]:
            raise _error(f"{filename} changed across stable readback passes")
        if _identity(metadata) != first_metadata[filename]:
            raise _error(f"{filename} identity changed across readback passes")
    _list_exact(directory_fd, filenames)
    if _identity(os.fstat(directory_fd)) != first_directory_identity:
        raise _error("bundle directory changed across stable readback passes")
    return values, bindings, first_raw


def _build_readback_report(
    *,
    cycle_id: str,
    run_id: str,
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        report = _core.build_discovery_readback_report_v4_1(
            cycle_id=cycle_id,
            run_id=run_id,
            artifact_bindings=sorted(
                (dict(item) for item in artifact_bindings),
                key=lambda item: str(item["filename"]),
            ),
            side_effects=dict(FIXED_SIDE_EFFECTS),
        )
    except (TypeError, ValueError) as exc:
        raise _error(f"discovery readback report build failed: {exc}") from exc
    return _validate_artifact(DISCOVERY_READBACK_REPORT_FILENAME, report)


def _cycle_id_from_values(values: Mapping[str, Mapping[str, Any]]) -> str:
    state = values.get(DISCOVERY_CYCLE_STATE_FILENAME)
    if not isinstance(state, Mapping):
        raise _error("DISCOVERY cycle state is missing")
    cycle_id = state.get("cycle_id")
    if type(cycle_id) is not str or not cycle_id:
        raise _error("DISCOVERY cycle state cycle_id is invalid")
    return cycle_id


def _verify_complete_values(
    *,
    run_id: str,
    values: Mapping[str, Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_cross_artifact_bundle(values)
    binding_by_name = {str(item["filename"]): dict(item) for item in bindings}
    input_bindings = [binding_by_name[name] for name in INPUT_ARTIFACT_FILENAMES]
    expected = _build_readback_report(
        cycle_id=_cycle_id_from_values(values),
        run_id=run_id,
        artifact_bindings=input_bindings,
    )
    actual = values.get(DISCOVERY_READBACK_REPORT_FILENAME)
    if actual is None:
        raise _error("discovery readback report is missing")
    if _core.canonical_file_bytes(actual) != _core.canonical_file_bytes(expected):
        raise _error("discovery readback report does not bind the live artifacts")
    return expected


def _validate_cross_artifact_bundle(
    values: Mapping[str, Mapping[str, Any]],
    *,
    base_ontology: Mapping[str, Any] | None = None,
    base_catalog: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Delegate all intra-bundle lineage checks to the pure core contract."""

    if (base_ontology is None) != (base_catalog is None):
        raise _error("base_ontology and base_catalog must be supplied together")
    validation_kwargs: dict[str, Any] = {}
    if base_ontology is not None and base_catalog is not None:
        validation_kwargs = {
            "base_ontology": copy.deepcopy(dict(base_ontology)),
            "base_catalog": copy.deepcopy(dict(base_catalog)),
        }
    try:
        normalized = _core.validate_discovery_bundle_v4_1(
            {
                filename: copy.deepcopy(dict(value))
                for filename, value in values.items()
            },
            **validation_kwargs,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise _error(f"cross-artifact discovery bundle validation failed: {exc}") from exc
    if not isinstance(normalized, Mapping):
        raise _error("cross-artifact validator returned a non-object")
    if set(normalized) != set(CANONICAL_ARTIFACT_FILENAMES):
        raise _error("cross-artifact validator returned an incomplete artifact set")
    for filename in CANONICAL_ARTIFACT_FILENAMES:
        normalized_value = normalized.get(filename)
        if not isinstance(normalized_value, Mapping):
            raise _error(
                f"cross-artifact validator returned invalid {filename} value"
            )
        if _core.canonical_file_bytes(normalized_value) != _core.canonical_file_bytes(
            values[filename]
        ):
            raise _error(
                f"cross-artifact validator changed normalized {filename} bytes"
            )
    return {
        filename: copy.deepcopy(dict(normalized[filename]))
        for filename in CANONICAL_ARTIFACT_FILENAMES
    }


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
        raise _error(f"{label} open rejected: {exc.strerror}") from exc
    try:
        opened = os.fstat(descriptor)
        _check_private_directory(opened, label)
        if not _same_object(path_value, opened):
            raise _error(f"{label} path/open identity mismatch")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _create_staging_directory(root_fd: int, run_id: str) -> tuple[str, int]:
    for _attempt in range(8):
        name = f".{run_id}.staging.{secrets.token_hex(12)}"
        try:
            os.mkdir(name, 0o700, dir_fd=root_fd)
        except FileExistsError:
            continue
        except OSError as exc:
            raise _error(f"staging directory creation failed: {exc.strerror}") from exc
        descriptor = _open_private_child_directory(root_fd, name, "staging directory")
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
            raise _error(f"quarantine creation failed: {exc.strerror}") from exc
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
    expected_source_identity: tuple[int, int] | None = None,
) -> str:
    quarantine_fd = _open_or_create_quarantine(root_fd)
    try:
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
                "quarantine source identity differs from committed staging inode"
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
            if _lstat_at(root_fd, source_name) is not None:
                raise _error("quarantined source remains visible at its old path")
            moved = _lstat_at(quarantine_fd, destination)
            if moved is None:
                raise _error("quarantined directory is missing after recovery")
            _check_private_directory(moved, "quarantined directory")
            if (int(moved.st_dev), int(moved.st_ino)) != source_identity:
                raise _error("quarantined directory identity changed during recovery")
            return destination
        raise _error("could not allocate a unique quarantine destination")
    finally:
        os.close(quarantine_fd)


def _descriptors(
    bundle_path: Path,
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(item["filename"]): {
            "absolute_path": str(bundle_path / str(item["filename"])),
            "byte_sha256": str(item["byte_sha256"]),
            "semantic_sha256": str(item["semantic_sha256"]),
            "size_bytes": int(item["size_bytes"]),
            "mode": int(item["mode"]),
            "uid": int(item["uid"]),
            "nlink": int(item["nlink"]),
        }
        for item in bindings
    }


def readback_discovery_bundle_v4_1(
    bundle_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Strictly and stably read one already-published DISCOVERY bundle."""

    _validate_core_filename_set()
    path = _absolute_normalized_path(bundle_path, "bundle_path")
    run_id = _safe_run_id(path.name)
    root, root_fd = _validate_private_root(path.parent)
    bundle_fd: int | None = None
    try:
        _assert_root_current(root, root_fd)
        bundle_fd = _open_private_child_directory(
            root_fd,
            run_id,
            "canonical discovery bundle",
        )
        values, bindings, _raw = _read_and_validate_files(
            bundle_fd,
            CANONICAL_ARTIFACT_FILENAMES,
        )
        report = _verify_complete_values(
            run_id=run_id,
            values=values,
            bindings=bindings,
        )
        _assert_root_current(root, root_fd)
        return {
            "accepted": True,
            "readiness": "EXPLORATORY_DISCOVERY",
            "qualification": False,
            "formal_admission_authority": False,
            "production_apply_enabled": False,
            "bundle_path": str(path),
            "artifact_descriptors": _descriptors(path, bindings),
            "readback": report,
            "side_effects": dict(FIXED_SIDE_EFFECTS),
        }
    finally:
        if bundle_fd is not None:
            os.close(bundle_fd)
        os.close(root_fd)


def readback_discovery_bundle_values_v4_1(
    bundle_path: str | os.PathLike[str],
    *,
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Read stable normalized bundle values bound to the exact base artifacts.

    Unlike :func:`readback_discovery_bundle_v4_1`, this API returns the eight
    normalized artifact values for downstream pure producers.  Both external
    base artifacts are mandatory and are cross-validated against the bundle
    only after the anchored exact-inventory, double-pass readback succeeds.
    Returned values are deep copies and remain non-formal DISCOVERY evidence.
    """

    if not isinstance(base_ontology, Mapping) or not isinstance(
        base_catalog, Mapping
    ):
        raise _error("base_ontology and base_catalog must both be mappings")
    _validate_core_filename_set()
    path = _absolute_normalized_path(bundle_path, "bundle_path")
    run_id = _safe_run_id(path.name)
    root, root_fd = _validate_private_root(path.parent)
    bundle_fd: int | None = None
    try:
        _assert_root_current(root, root_fd)
        bundle_fd = _open_private_child_directory(
            root_fd,
            run_id,
            "canonical discovery bundle",
        )
        values, bindings, _raw = _read_and_validate_files(
            bundle_fd,
            CANONICAL_ARTIFACT_FILENAMES,
        )
        report = _verify_complete_values(
            run_id=run_id,
            values=values,
            bindings=bindings,
        )
        normalized = _validate_cross_artifact_bundle(
            values,
            base_ontology=base_ontology,
            base_catalog=base_catalog,
        )
        _assert_root_current(root, root_fd)
        return {
            "accepted": True,
            "readiness": "EXPLORATORY_DISCOVERY",
            "qualification": False,
            "formal_admission_authority": False,
            "production_apply_enabled": False,
            "bundle_path": str(path),
            "artifact_descriptors": _descriptors(path, bindings),
            "readback": report,
            "side_effects": dict(FIXED_SIDE_EFFECTS),
            "values": copy.deepcopy(normalized),
        }
    finally:
        if bundle_fd is not None:
            os.close(bundle_fd)
        os.close(root_fd)


def publish_discovery_bundle_v4_1(
    *,
    private_root: str | os.PathLike[str],
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    revalidate_inputs: Callable[[], None],
    race_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Atomically publish one complete private DISCOVERY bundle.

    ``revalidate_inputs`` is mandatory and is invoked under the secured lock
    immediately before the final target-absence diagnostic and exclusive
    directory rename.  ``race_hook`` exists only for deterministic fault/race
    testing and runs after that diagnostic.
    """

    _validate_core_filename_set()
    _require_exclusive_rename_support()
    normalized_run_id = _safe_run_id(run_id)
    if not callable(revalidate_inputs):
        raise _error("revalidate_inputs must be a zero-argument callable")
    if race_hook is not None and not callable(race_hook):
        raise _error("race_hook must be callable when supplied")
    if not isinstance(artifacts, Mapping):
        raise _error("artifacts must be an exact filename-to-object mapping")
    if any(type(filename) is not str for filename in artifacts):
        raise _error("artifact filenames must be exact strings")
    if set(artifacts) != set(INPUT_ARTIFACT_FILENAMES):
        missing = sorted(set(INPUT_ARTIFACT_FILENAMES) - set(artifacts))
        extra = sorted(set(artifacts) - set(INPUT_ARTIFACT_FILENAMES))
        raise _error(
            "publication input artifact set mismatch: "
            f"missing={','.join(missing) or '-'};extra={','.join(extra) or '-'}"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for filename in INPUT_ARTIFACT_FILENAMES:
        value = artifacts[filename]
        if not isinstance(value, Mapping):
            raise _error(f"{filename} publication value must be an object")
        normalized[filename] = _validate_artifact(filename, value)

    root, root_fd = _validate_private_root(private_root)
    lock_fd: int | None = None
    staging_fd: int | None = None
    staging_name: str | None = None
    staging_identity: tuple[int, int] | None = None
    committed = False
    try:
        lock_fd = _open_lock(root, root_fd)
        _assert_absent(root_fd, normalized_run_id, "canonical discovery bundle")
        staging_name, staging_fd = _create_staging_directory(
            root_fd,
            normalized_run_id,
        )
        staging_identity = _directory_object_identity(os.fstat(staging_fd))
        input_bindings: list[dict[str, Any]] = []
        for filename in INPUT_ARTIFACT_FILENAMES:
            input_bindings.append(
                _write_private_artifact(
                    staging_fd,
                    filename,
                    normalized[filename],
                )
            )
        os.fsync(staging_fd)
        staging_values, staging_bindings, _raw = _read_and_validate_files(
            staging_fd,
            INPUT_ARTIFACT_FILENAMES,
        )
        if [dict(item) for item in staging_bindings] != input_bindings:
            raise _error("staging artifact bindings changed after write")

        report = _build_readback_report(
            cycle_id=_cycle_id_from_values(staging_values),
            run_id=normalized_run_id,
            artifact_bindings=staging_bindings,
        )
        _write_private_artifact(
            staging_fd,
            DISCOVERY_READBACK_REPORT_FILENAME,
            report,
        )
        os.fsync(staging_fd)
        complete_values, complete_bindings, _raw = _read_and_validate_files(
            staging_fd,
            CANONICAL_ARTIFACT_FILENAMES,
        )
        _verify_complete_values(
            run_id=normalized_run_id,
            values=complete_values,
            bindings=complete_bindings,
        )

        try:
            revalidate_inputs()
        except Exception as exc:
            raise _error(f"input revalidation failed: {exc}") from exc
        _assert_root_current(root, root_fd)
        lock_path = _lstat_at(root_fd, LOCK_FILENAME)
        if lock_path is None or not _same_object(lock_path, os.fstat(lock_fd)):
            raise _error("publication lock changed before commit")
        _check_private_file(lock_path, "publication lock")
        _assert_absent(root_fd, normalized_run_id, "canonical discovery bundle")
        if race_hook is not None:
            race_hook()

        staging_opened = os.fstat(staging_fd)
        _check_private_directory(staging_opened, "staging directory")
        staging_path = _lstat_at(root_fd, staging_name)
        if staging_path is None or not _same_object(staging_path, staging_opened):
            raise _error("staging directory path identity changed before commit")
        _check_private_directory(staging_path, "staging directory")
        if _directory_object_identity(staging_opened) != staging_identity:
            raise _error("open staging directory identity changed before commit")
        try:
            _renameatx_np_exclusive(
                root_fd,
                staging_name,
                root_fd,
                normalized_run_id,
            )
        except FileExistsError as exc:
            raise _error(
                "canonical discovery bundle appeared during exclusive commit"
            ) from exc
        committed = True
        staging_name = None

        try:
            canonical_path = _lstat_at(root_fd, normalized_run_id)
            if canonical_path is None:
                raise _error("canonical bundle missing immediately after commit")
            _check_private_directory(canonical_path, "canonical discovery bundle")
            if (
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
            os.fsync(root_fd)
            result = readback_discovery_bundle_v4_1(
                root / normalized_run_id
            )
            if (
                not isinstance(result, Mapping)
                or result.get("accepted") is not True
                or result.get("readiness") != "EXPLORATORY_DISCOVERY"
                or result.get("qualification") is not False
                or result.get("side_effects") != FIXED_SIDE_EFFECTS
            ):
                raise _error(
                    "live canonical readback did not return the exact accepted "
                    "DISCOVERY contract"
                )
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
                    f"not durably proven: {postcommit_exc}; recovery={recovery_exc}",
                    status="AMBIGUOUS_DURABILITY_FAIL_CLOSED",
                ) from recovery_exc
            raise _error(
                "post-commit durability/readback failed; canonical bundle was "
                f"quarantined: {postcommit_exc}",
                status="POSTCOMMIT_RECOVERED_FAIL_CLOSED",
            ) from postcommit_exc
        return result
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
                # The canonical path is still absent.  Preserve the original
                # rejection while leaving the hidden staging tree unaccepted.
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
    "AQUANT_SOURCE_RECEIPT_FILENAME",
    "CANONICAL_ARTIFACT_FILENAMES",
    "DISCOVERY_CATALOG_FILENAME",
    "DISCOVERY_CYCLE_STATE_FILENAME",
    "DISCOVERY_READBACK_REPORT_FILENAME",
    "DISCOVERY_SOURCE_NODE_FILENAME",
    "FIXED_SIDE_EFFECTS",
    "FactorGovernanceDiscoveryReadbackV4_1Error",
    "INPUT_ARTIFACT_FILENAMES",
    "LOCAL_COMPATIBILITY_CONTRACT_FILENAME",
    "LOCK_FILENAME",
    "QUARANTINE_DIRECTORY",
    "SOURCE_IDEA_AUDIT_FILENAME",
    "STRUCTURAL_COLLISION_AUDIT_FILENAME",
    "publish_discovery_bundle_v4_1",
    "readback_discovery_bundle_v4_1",
    "readback_discovery_bundle_values_v4_1",
]
