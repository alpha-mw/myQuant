"""Fail-closed local byte readback for factor-governance replay evidence.

This module proves only that one explicitly selected local evidence graph was
read from stable bytes and recomputed consistently.  It does not authenticate
the producer, authorize a production mutation, or make the graph eligible for
production apply.

Immutable publish never unlinks a final destination during fault cleanup.  A
post-link failure is reconciled only through the fresh exact path, original
parent identity, directory fsync and stable readback.  Persistent ambiguity
may leave a correct immutable commit (including an exact selected receipt) or
an unselected orphan, so callers must run exact verification before retrying or
drawing an authority conclusion.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Sequence


DRAFT_SCHEMA_VERSION = "factor-governance-canonical-replay-draft.v1"
BUNDLE_SCHEMA_VERSION = "factor-governance-canonical-replay-bundle.v1"
RECEIPT_SCHEMA_VERSION = "factor-governance-canonical-replay-receipt.v1"
STAGE_SCHEMA_VERSION = "factor-governance-canonical-stage.v1"
CONTROL_SCHEMA_VERSION = "factor-governance-canonical-readback-control.v1"
PRODUCER_CONTRACT_SCHEMA_VERSION = (
    "factor-governance-canonical-producer-contract.v1"
)

ARM_NAMES = ("A", "B", "C", "D")
CONTROL_CHAIN_STAGES = (
    "quant",
    "theme",
    "bayesian",
    "risk_guard",
    "portfolio_constructor",
)
CODE_CONFIG_ROLES = (
    "quant_runtime",
    "theme_runtime",
    "bayesian_runtime",
    "risk_guard_runtime",
    "portfolio_constructor_runtime",
    "factor_governance_policy",
    "factor_registry_schema",
    "runtime_configuration",
)

DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_JSON_DEPTH = 32
DEFAULT_MAX_JSON_NODES = 250_000
GENESIS_SHA256 = "0" * 64
ALLOWED_CHALLENGER_REGISTRY_STATES = frozenset(
    {"shadow", "mature_candidate", "production_candidate"}
)


class CanonicalReplayError(ValueError):
    """Raised when canonical replay bytes or graph identities fail closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes without a trailing newline."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise CanonicalReplayError(f"value is not canonical JSON: {exc}") from exc


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _reject_constant(value: str) -> Any:
    raise CanonicalReplayError(f"non-finite JSON number is forbidden: {value}")


def _strict_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise CanonicalReplayError(
            f"non-finite JSON number is forbidden: {value}"
        )
    return number


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalReplayError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _check_json_limits(value: Any, *, max_depth: int, max_nodes: int) -> None:
    if type(max_depth) is not int or max_depth < 1:
        raise CanonicalReplayError("max_depth must be a positive integer")
    if type(max_nodes) is not int or max_nodes < 1:
        raise CanonicalReplayError("max_nodes must be a positive integer")
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise CanonicalReplayError("JSON node limit exceeded")
        if depth > max_depth:
            raise CanonicalReplayError("JSON depth limit exceeded")
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def strict_json_loads(
    raw: bytes,
    *,
    expected_fields: set[str] | frozenset[str] | None = None,
    max_depth: int = DEFAULT_MAX_JSON_DEPTH,
    max_nodes: int = DEFAULT_MAX_JSON_NODES,
    require_object: bool = True,
) -> Any:
    """Parse strict JSON, rejecting duplicates, non-finite values and limits."""

    if type(raw) is not bytes:
        raise CanonicalReplayError("strict JSON input must be bytes")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CanonicalReplayError("JSON input is not strict UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except CanonicalReplayError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as exc:
        raise CanonicalReplayError(f"invalid JSON: {exc}") from exc
    _check_json_limits(value, max_depth=max_depth, max_nodes=max_nodes)
    if require_object and not isinstance(value, dict):
        raise CanonicalReplayError("JSON top level must be an object")
    if expected_fields is not None:
        if not isinstance(value, dict):
            raise CanonicalReplayError("exact-field JSON must be an object")
        actual = set(value)
        expected = set(expected_fields)
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        if unknown:
            raise CanonicalReplayError(
                f"unknown JSON fields: {', '.join(unknown)}"
            )
        if missing:
            raise CanonicalReplayError(
                f"missing JSON fields: {', '.join(missing)}"
            )
    return value


def _exact_fields(
    payload: Any,
    expected: Iterable[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CanonicalReplayError(f"{label} must be an object")
    expected_set = set(expected)
    actual = set(payload)
    unknown = sorted(actual - expected_set)
    missing = sorted(expected_set - actual)
    if unknown:
        raise CanonicalReplayError(f"{label} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise CanonicalReplayError(f"{label} is missing fields: {', '.join(missing)}")
    return payload


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CanonicalReplayError(f"{label} must be an exact non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CanonicalReplayError(f"{label} must be lowercase SHA-256 hex")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalReplayError(f"{label} must be a finite JSON number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CanonicalReplayError(
            f"{label} must be a finite JSON number"
        ) from exc
    if not math.isfinite(number):
        raise CanonicalReplayError(f"{label} must be a finite JSON number")
    return number


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CanonicalReplayError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _iso_date(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 10:
        raise CanonicalReplayError(f"{label} must be ISO YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise CanonicalReplayError(f"{label} must be ISO YYYY-MM-DD") from exc
    normalized = parsed.isoformat()
    if text != normalized:
        raise CanonicalReplayError(f"{label} must be exact ISO YYYY-MM-DD")
    return normalized


def _ordered_distinct_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise CanonicalReplayError(f"{label} must be a list")
    result = [_text(item, f"{label}[]") for item in value]
    if result != sorted(result) or len(result) != len(set(result)):
        raise CanonicalReplayError(f"{label} must be sorted and distinct")
    return result


def canonical_replay_producer_contract() -> dict[str, Any]:
    """Return the pure, hashable local-readback producer contract."""

    return {
        "schema_version": PRODUCER_CONTRACT_SCHEMA_VERSION,
        "producer_id": "myquant.factor_governance_canonical_local_readback",
        "selection": {
            "registry": "explicit_exact_path_raw_byte_sha256",
            "receipt": "receipts/{registry_sha256}.json",
            "bundle": "bundles/{receipt.evidence_id}.json",
            "scan_glob_latest_fallback": False,
        },
        "read_safety": {
            "dirfd_from_root": True,
            "no_follow": True,
            "stable_pre_post_identity": True,
            "single_link_regular_files": True,
            "bounded_bytes_and_json": True,
            "private_mode": "dirs_0700_files_0600",
            "external_mode": "owner_and_not_group_or_world_writable",
        },
        "graph": {
            "arms": list(ARM_NAMES),
            "stages": list(CONTROL_CHAIN_STAGES),
            "stage_count": len(ARM_NAMES) * len(CONTROL_CHAIN_STAGES),
            "global_unique_paths_and_inodes": True,
            "predecessor_byte_and_semantic_hashes": True,
        },
        "registry_record_identity": {
            "algorithm": (
                "sha256_of_canonical_compact_sorted_json_registry_record"
            ),
            "slot_identity_algorithm": (
                "governance_protocol_v2_slot_identity_from_exact_registry_record"
            ),
            "family_fallback_fields": [
                "metadata.factor_family",
                "metadata.governance_family",
                "category",
            ],
            "cluster_fallback_fields": [
                "metadata.dominant_primitive_cluster",
                "metadata.dominant_primitives_sorted_join_plus",
            ],
            "slot_format": "{family}::{dominant_primitive_cluster}",
            "missing_selected_slot_identity": "reject",
            "duplicate_factor_names": "reject",
            "production_incumbent_state": "production_factor",
            "allowed_challenger_states": sorted(
                ALLOWED_CHALLENGER_REGISTRY_STATES
            ),
            "stage_binding_fields": [
                "family",
                "slot",
                "registry_state",
                "registry_record_sha256",
            ],
        },
        "temporal_evidence": {
            "cutoff": "window_end",
            "single_pit_snapshot_rule": "window_end_equals_as_of",
            "calendar_membership_required": [
                "window_start",
                "window_end",
                "as_of",
            ],
            "market_window": (
                "exact_inclusive_independent_open_day_slice_"
                "window_start_through_window_end"
            ),
            "month_end_rankic_dates_not_after_cutoff": True,
            "forward_cohort_start_end_not_after_cutoff": True,
            "walk_forward_all_dates_not_after_cutoff": True,
        },
        "publish": {
            "canonical_compact_sorted_json_newline": True,
            "atomic_hard_link_no_clobber": True,
            "new_owned_modes_born_exact_after_umask_preflight": (
                "directories_0700_files_0600"
            ),
            "per_destination_lock": (
                "persistent_owner_0600_regular_nlink1_flock_exclusive"
            ),
            "supported_umask_preflight": (
                "unique_file0600_and_directory0700_exact_mode_probes_"
                "in_existing_private_root_before_named_namespace_mutation"
            ),
            "unsupported_umask": (
                "reject_before_bundles_receipts_lock_temp_or_final_creation"
            ),
            "umask_probe_crash_orphan": (
                "ignored_internal_unselected_random_name_no_scan"
            ),
            "reserved_temp_name": (
                ".canonical-publish.{sha256(leaf)}."
                "{canonical_byte_sha256}.tmp"
            ),
            "reserved_namespace_is_internal": True,
            "recovery_selection": "exact_final_lock_and_reserved_temp_only_no_scan",
            "prepared_file_and_directory_fsync_required_before_final_link": True,
            "shared_inode_file_and_directory_fsync_before_post_link_unlink": True,
            "directory_fsync_after_every_reserved_temp_unlink": True,
            "crash_recovery": [
                "resume_exact_nlink1_temp_when_final_absent",
                "unlink_exact_same_inode_nlink2_temp_then_verify_final_nlink1",
                "preserve_same_byte_different_inode_final_and_clear_exact_temp",
                "clear_partial_temp_only_when_nlink1_and_final_absent_under_lock",
                "never_unlink_final",
            ],
            "authoritative_nlink": 1,
            "exception_cleanup": (
                "reserved_temp_reconciliation_under_lock_never_final_leaf"
            ),
            "post_link_exception_reconciliation": (
                "fresh_exact_path_original_parent_identity_"
                "directory_fsync_double_readback"
            ),
            "failed_or_ambiguous_call_may_leave_correct_commit_or_orphan": True,
            "exact_verification_required_after_ambiguous_failure": True,
            "priority": "no_clobber_and_no_unrelated_data_loss",
        },
        "authority": {
            "producer_implemented": True,
            "canonical_producer_authenticated": False,
            "production_apply_authorized": False,
            "production_apply_eligible": False,
        },
    }


def producer_contract_sha256() -> str:
    """Return the SHA-256 identity of the pure producer contract."""

    return _sha256_value(canonical_replay_producer_contract())


@dataclass(frozen=True)
class _FileIdentity:
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    nlink: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_FileIdentity":
        return cls(
            dev=value.st_dev,
            ino=value.st_ino,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
            mode=value.st_mode,
            nlink=value.st_nlink,
        )


@dataclass(frozen=True)
class _DirectoryIdentity:
    path: str
    dev: int
    ino: int
    mode: int
    uid: int
    gid: int

    @classmethod
    def from_stat(
        cls, path: str, value: os.stat_result
    ) -> "_DirectoryIdentity":
        return cls(
            path=path,
            dev=value.st_dev,
            ino=value.st_ino,
            mode=value.st_mode,
            uid=value.st_uid,
            gid=value.st_gid,
        )


@dataclass(frozen=True)
class _ReadRecord:
    identity: _FileIdentity
    sha256: str
    private: bool
    directories: tuple[_DirectoryIdentity, ...]


class SafeReadSession:
    """Read exact files through root-based dirfds and detect later drift."""

    def __init__(
        self,
        private_root: str | os.PathLike[str],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self.private_root = self._validate_absolute(private_root, "private_root")
        self._private_parts = self._parts(self.private_root)
        self.max_file_bytes = _integer(
            max_file_bytes, "max_file_bytes", minimum=1
        )
        self.max_total_bytes = _integer(
            max_total_bytes, "max_total_bytes", minimum=1
        )
        self._total_bytes = 0
        self._records: dict[str, _ReadRecord] = {}
        root_fd = self._open_directory_path(self.private_root, require_private=True)
        os.close(root_fd)

    @staticmethod
    def _validate_absolute(
        path: str | os.PathLike[str], label: str
    ) -> str:
        try:
            raw = os.fspath(path)
        except TypeError as exc:
            raise CanonicalReplayError(
                f"{label} must be an absolute path string"
            ) from exc
        if type(raw) is not str or not raw.startswith("/") or "\x00" in raw:
            raise CanonicalReplayError(f"{label} must be an absolute path")
        pieces = raw.split("/")[1:]
        if not pieces or any(piece in {"", ".", ".."} for piece in pieces):
            raise CanonicalReplayError(f"{label} must be a normalized absolute path")
        return raw

    @staticmethod
    def _parts(path: str) -> tuple[str, ...]:
        return tuple(path.split("/")[1:])

    def _is_private_parts(self, parts: Sequence[str]) -> bool:
        count = len(self._private_parts)
        return len(parts) >= count and tuple(parts[:count]) == self._private_parts

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )

    @staticmethod
    def _file_flags() -> int:
        return (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )

    def _check_private_directory(self, value: os.stat_result, label: str) -> None:
        if not stat.S_ISDIR(value.st_mode):
            raise CanonicalReplayError(f"{label} is not a directory")
        if value.st_uid != os.getuid():
            raise CanonicalReplayError(f"{label} owner mismatch")
        if stat.S_IMODE(value.st_mode) != 0o700:
            raise CanonicalReplayError(f"{label} private directory mode must be 0700")

    def _open_directory_path(
        self,
        path: str,
        *,
        require_private: bool,
        capture: list[_DirectoryIdentity] | None = None,
    ) -> int:
        parts = self._parts(path)
        fd = os.open("/", self._directory_flags())
        traversed: list[str] = []
        try:
            if capture is not None:
                capture.append(_DirectoryIdentity.from_stat("/", os.fstat(fd)))
            for component in parts:
                try:
                    child = os.open(component, self._directory_flags(), dir_fd=fd)
                except OSError as exc:
                    raise CanonicalReplayError(
                        f"directory traversal rejected at {component}: {exc.strerror}"
                    ) from exc
                os.close(fd)
                fd = child
                traversed.append(component)
                current_path = "/" + "/".join(traversed)
                if require_private and self._is_private_parts(traversed):
                    self._check_private_directory(
                        os.fstat(fd), current_path
                    )
                if capture is not None:
                    capture.append(
                        _DirectoryIdentity.from_stat(current_path, os.fstat(fd))
                    )
            return fd
        except Exception:
            os.close(fd)
            raise

    def _open_leaf(
        self, path: str
    ) -> tuple[
        int,
        int,
        str,
        bool,
        os.stat_result,
        tuple[_DirectoryIdentity, ...],
    ]:
        parts = self._parts(path)
        parent_parts = parts[:-1]
        private = self._is_private_parts(parts)
        parent_path = "/" + "/".join(parent_parts)
        directory_chain: list[_DirectoryIdentity] = []
        parent_fd = self._open_directory_path(
            parent_path,
            require_private=private,
            capture=directory_chain,
        )
        leaf = parts[-1]
        try:
            fd = os.open(leaf, self._file_flags(), dir_fd=parent_fd)
        except OSError as exc:
            os.close(parent_fd)
            raise CanonicalReplayError(
                f"safe file open rejected for {leaf}: {exc.strerror}"
            ) from exc
        try:
            before = os.fstat(fd)
            self._check_file(before, path, private=private)
        except Exception:
            os.close(fd)
            os.close(parent_fd)
            raise
        return fd, parent_fd, leaf, private, before, tuple(directory_chain)

    def _capture_directory_chain(
        self, path: str, *, private: bool
    ) -> tuple[_DirectoryIdentity, ...]:
        parent_parts = self._parts(path)[:-1]
        parent_path = "/" + "/".join(parent_parts)
        chain: list[_DirectoryIdentity] = []
        fd = self._open_directory_path(
            parent_path,
            require_private=private,
            capture=chain,
        )
        os.close(fd)
        return tuple(chain)

    def _assert_directory_chain(
        self,
        path: str,
        expected: tuple[_DirectoryIdentity, ...],
        *,
        private: bool,
    ) -> None:
        first = self._capture_directory_chain(path, private=private)
        second = self._capture_directory_chain(path, private=private)
        if first != expected or second != expected:
            raise CanonicalReplayError(
                f"ancestor directory path identity changed for {path}"
            )

    def _check_file(
        self, value: os.stat_result, label: str, *, private: bool
    ) -> None:
        if not stat.S_ISREG(value.st_mode):
            raise CanonicalReplayError(f"{label} must be a regular file")
        if value.st_nlink != 1:
            raise CanonicalReplayError(f"{label} link count must equal 1")
        if value.st_uid != os.getuid():
            raise CanonicalReplayError(f"{label} owner mismatch")
        mode = stat.S_IMODE(value.st_mode)
        if private:
            if mode != 0o600:
                raise CanonicalReplayError(f"{label} private file mode must be 0600")
        elif mode & 0o022:
            raise CanonicalReplayError(
                f"{label} external file must not be group/world writable"
            )
        if value.st_size > self.max_file_bytes:
            raise CanonicalReplayError(f"{label} size exceeds file byte limit")
        blocks = getattr(value, "st_blocks", None)
        if value.st_size and blocks is not None and blocks * 512 < value.st_size:
            raise CanonicalReplayError(f"{label} sparse files are forbidden")

    @staticmethod
    def _path_stat(parent_fd: int, leaf: str) -> os.stat_result:
        try:
            return os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise CanonicalReplayError(
                f"post-read path stat failed for {leaf}: {exc.strerror}"
            ) from exc

    def read_bytes(self, path: str | os.PathLike[str]) -> bytes:
        """Read one exact stable file and remember its byte identity."""

        exact = self._validate_absolute(path, "path")
        (
            fd,
            parent_fd,
            leaf,
            private,
            before,
            directory_chain,
        ) = self._open_leaf(exact)
        try:
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, min(65_536, remaining))
                if not chunk:
                    raise CanonicalReplayError(f"{exact} shrank during read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise CanonicalReplayError(f"{exact} grew during read")
            after = os.fstat(fd)
            path_after = self._path_stat(parent_fd, leaf)
            self._check_file(after, exact, private=private)
            self._check_file(path_after, exact, private=private)
            identity = _FileIdentity.from_stat(before)
            if identity != _FileIdentity.from_stat(after):
                raise CanonicalReplayError(f"{exact} changed during read")
            if identity != _FileIdentity.from_stat(path_after):
                raise CanonicalReplayError(f"{exact} path was replaced during read")
            self._assert_directory_chain(
                exact,
                directory_chain,
                private=private,
            )
            raw = b"".join(chunks)
            if len(raw) != before.st_size:
                raise CanonicalReplayError(f"{exact} byte count changed during read")
            self._total_bytes += len(raw)
            if self._total_bytes > self.max_total_bytes:
                raise CanonicalReplayError("aggregate read byte limit exceeded")
            record = _ReadRecord(
                identity,
                _sha256_bytes(raw),
                private,
                directory_chain,
            )
            previous = self._records.get(exact)
            if previous is not None and previous != record:
                raise CanonicalReplayError(f"{exact} drifted between reads")
            self._records[exact] = record
            return raw
        finally:
            os.close(fd)
            os.close(parent_fd)

    def assert_unchanged(self) -> None:
        """Re-open every recorded path and reject any graph-wide drift."""

        for path, record in self._records.items():
            (
                fd,
                parent_fd,
                leaf,
                private,
                current,
                directory_chain,
            ) = self._open_leaf(path)
            try:
                path_current = self._path_stat(parent_fd, leaf)
                self._check_file(current, path, private=private)
                self._check_file(path_current, path, private=private)
                if record.identity != _FileIdentity.from_stat(current):
                    raise CanonicalReplayError(f"source drift detected for {path}")
                if record.identity != _FileIdentity.from_stat(path_current):
                    raise CanonicalReplayError(f"source path drift detected for {path}")
                if directory_chain != record.directories:
                    raise CanonicalReplayError(
                        f"source ancestor directory drift detected for {path}"
                    )
                self._assert_directory_chain(
                    path,
                    record.directories,
                    private=private,
                )
            finally:
                os.close(fd)
                os.close(parent_fd)

    def identity(self, path: str | os.PathLike[str]) -> tuple[int, int]:
        """Return the previously read device/inode identity for an exact path."""

        exact = self._validate_absolute(path, "path")
        try:
            record = self._records[exact]
        except KeyError as exc:
            raise CanonicalReplayError(f"path was not read in this session: {exact}") from exc
        return (record.identity.dev, record.identity.ino)


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if type(relative_path) is not str or not relative_path:
        raise CanonicalReplayError("relative publish path must be non-empty")
    if relative_path.startswith("/"):
        raise CanonicalReplayError("publish path must be relative")
    parts = tuple(relative_path.split("/"))
    if any(
        part in {"", ".", ".."} or "\x00" in part for part in parts
    ):
        raise CanonicalReplayError("publish path must be normalized")
    try:
        for part in parts:
            part.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise CanonicalReplayError(
            "publish path components must be strict UTF-8"
        ) from exc
    return parts


def _open_publish_parent(
    private_root: str | os.PathLike[str], relative_path: str
) -> tuple[int, str, str, tuple[_DirectoryIdentity, ...]]:
    session = SafeReadSession(private_root)
    exact_root = session.private_root
    parts = _relative_parts(relative_path)
    directory_chain: list[_DirectoryIdentity] = []
    fd = session._open_directory_path(
        exact_root,
        require_private=True,
        capture=directory_chain,
    )
    current_path = exact_root
    try:
        for component in parts[:-1]:
            try:
                child = os.open(component, session._directory_flags(), dir_fd=fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                os.fsync(fd)
                child = os.open(component, session._directory_flags(), dir_fd=fd)
            except OSError as exc:
                raise CanonicalReplayError(
                    f"publish directory traversal rejected: {exc.strerror}"
                ) from exc
            try:
                session._check_private_directory(
                    os.fstat(child), f"private publish directory {component}"
                )
            except Exception:
                os.close(child)
                raise
            os.close(fd)
            fd = child
            current_path = current_path + "/" + component
            directory_chain.append(
                _DirectoryIdentity.from_stat(current_path, os.fstat(fd))
            )
        return (
            fd,
            parts[-1],
            exact_root + "/" + "/".join(parts),
            tuple(directory_chain),
        )
    except Exception:
        os.close(fd)
        raise


def _preflight_publish_umask(session: SafeReadSession) -> None:
    """Prove requested private modes are born exact before named mutation."""

    token = f"{os.getpid()}.{secrets.token_hex(16)}"
    file_name = f".canonical-publish.umask-file-probe.{token}"
    directory_name = f".canonical-publish.umask-dir-probe.{token}"
    chain: list[_DirectoryIdentity] = []
    root_fd = session._open_directory_path(
        session.private_root,
        require_private=True,
        capture=chain,
    )
    file_fd: int | None = None
    file_exists = False
    directory_exists = False
    failure: CanonicalReplayError | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_fd = os.open(file_name, flags, 0o600, dir_fd=root_fd)
        file_exists = True
        file_stat = os.fstat(file_fd)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.getuid()
            or file_stat.st_nlink != 1
            or file_stat.st_size != 0
            or stat.S_IMODE(file_stat.st_mode) != 0o600
        ):
            failure = CanonicalReplayError(
                "publish umask must preserve exact private file mode 0600"
            )
        os.mkdir(directory_name, 0o700, dir_fd=root_fd)
        directory_exists = True
        directory_stat = os.stat(
            directory_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.getuid()
            or stat.S_IMODE(directory_stat.st_mode) != 0o700
        ):
            failure = CanonicalReplayError(
                "publish umask must preserve exact private directory mode 0700"
            )
    except OSError as exc:
        raise CanonicalReplayError(
            f"publish umask preflight failed: {exc.strerror}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if file_exists:
            try:
                os.unlink(file_name, dir_fd=root_fd)
            except OSError:
                pass
        if directory_exists:
            try:
                os.rmdir(directory_name, dir_fd=root_fd)
            except OSError:
                pass
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    session._assert_directory_chain(
        session.private_root + "/preflight",
        tuple(chain),
        private=True,
    )
    if failure is not None:
        raise failure


def publish_immutable_json(
    private_root: str | os.PathLike[str],
    relative_path: str,
    payload: Mapping[str, Any],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_json_depth: int = DEFAULT_MAX_JSON_DEPTH,
    max_json_nodes: int = DEFAULT_MAX_JSON_NODES,
) -> dict[str, Any]:
    """Publish canonical JSON with locked, crash-recoverable no-clobber."""

    max_file_bytes = _integer(max_file_bytes, "max_file_bytes", minimum=1)
    raw = canonical_json_bytes(dict(payload)) + b"\n"
    if len(raw) > max_file_bytes:
        raise CanonicalReplayError("immutable publish size exceeds file byte limit")
    try:
        strict_json_loads(
            raw,
            max_depth=max_json_depth,
            max_nodes=max_json_nodes,
        )
    except CanonicalReplayError as exc:
        raise CanonicalReplayError(f"value is not canonical JSON: {exc}") from exc

    parent_guard = SafeReadSession(private_root)
    _preflight_publish_umask(parent_guard)
    parent_fd, leaf, exact_destination, parent_chain = _open_publish_parent(
        private_root, relative_path
    )
    try:
        result = {"sha256": _sha256_bytes(raw), "size": len(raw)}
        leaf_sha = _sha256_bytes(leaf.encode("utf-8", errors="strict"))
        lock_name = f".canonical-publish.{leaf_sha}.lock"
        temp_name = f".canonical-publish.{leaf_sha}.{result['sha256']}.tmp"
    except BaseException:
        os.close(parent_fd)
        raise
    lock_fd: int | None = None
    lock_held = False

    def assert_parent_current() -> None:
        parent_guard._assert_directory_chain(
            exact_destination,
            parent_chain,
            private=True,
        )

    def path_stat(name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CanonicalReplayError(
                f"immutable publish path stat failed: {exc.strerror}"
            ) from exc

    def check_publish_file(
        value: os.stat_result,
        label: str,
        *,
        allowed_nlinks: frozenset[int],
    ) -> None:
        if not stat.S_ISREG(value.st_mode):
            raise CanonicalReplayError(f"{label} must be a regular file")
        if value.st_uid != os.getuid():
            raise CanonicalReplayError(f"{label} owner mismatch")
        if stat.S_IMODE(value.st_mode) != 0o600:
            raise CanonicalReplayError(f"{label} mode must be 0600")
        if value.st_nlink not in allowed_nlinks:
            raise CanonicalReplayError(f"{label} link count is unsafe")
        if value.st_size > max_file_bytes:
            raise CanonicalReplayError(f"{label} size exceeds file byte limit")
        blocks = getattr(value, "st_blocks", None)
        if value.st_size and blocks is not None and blocks * 512 < value.st_size:
            raise CanonicalReplayError(f"{label} sparse files are forbidden")

    def read_reserved(
        name: str, *, allowed_nlinks: frozenset[int]
    ) -> tuple[bytes, os.stat_result]:
        assert_parent_current()
        path_before = path_stat(name)
        if path_before is None:
            raise CanonicalReplayError("immutable publish reserved path disappeared")
        check_publish_file(
            path_before,
            "immutable publish reserved file",
            allowed_nlinks=allowed_nlinks,
        )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise CanonicalReplayError(
                f"immutable publish reserved open failed: {exc.strerror}"
            ) from exc
        try:
            before = os.fstat(fd)
            check_publish_file(
                before,
                "immutable publish reserved file",
                allowed_nlinks=allowed_nlinks,
            )
            if _FileIdentity.from_stat(before) != _FileIdentity.from_stat(
                path_before
            ):
                raise CanonicalReplayError(
                    "immutable publish reserved path identity changed"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, min(65_536, remaining))
                if not chunk:
                    raise CanonicalReplayError(
                        "immutable publish reserved file shrank during read"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise CanonicalReplayError(
                    "immutable publish reserved file grew during read"
                )
            after = os.fstat(fd)
            path_after = path_stat(name)
            if path_after is None:
                raise CanonicalReplayError(
                    "immutable publish reserved path disappeared"
                )
            check_publish_file(
                after,
                "immutable publish reserved file",
                allowed_nlinks=allowed_nlinks,
            )
            check_publish_file(
                path_after,
                "immutable publish reserved file",
                allowed_nlinks=allowed_nlinks,
            )
            identity = _FileIdentity.from_stat(before)
            if identity != _FileIdentity.from_stat(after) or identity != (
                _FileIdentity.from_stat(path_after)
            ):
                raise CanonicalReplayError(
                    "immutable publish reserved file changed during read"
                )
            content = b"".join(chunks)
            if len(content) != before.st_size:
                raise CanonicalReplayError(
                    "immutable publish reserved byte count changed"
                )
        finally:
            os.close(fd)
        assert_parent_current()
        return content, path_before

    def read_destination() -> bytes:
        return SafeReadSession(
            private_root,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_file_bytes,
        ).read_bytes(exact_destination)

    def fsync_exact_file(
        name: str, *, allowed_nlinks: frozenset[int]
    ) -> None:
        content, expected = read_reserved(name, allowed_nlinks=allowed_nlinks)
        if content != raw:
            raise CanonicalReplayError(
                "immutable fsync target bytes do not match canonical bytes"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise CanonicalReplayError(
                f"immutable fsync target reopen failed: {exc.strerror}"
            ) from exc
        try:
            before = os.fstat(fd)
            check_publish_file(
                before,
                "immutable fsync target",
                allowed_nlinks=allowed_nlinks,
            )
            if _FileIdentity.from_stat(before) != _FileIdentity.from_stat(
                expected
            ):
                raise CanonicalReplayError(
                    "immutable fsync target identity changed before fsync"
                )
            os.fsync(fd)
            after = os.fstat(fd)
            path_after = path_stat(name)
            if path_after is None:
                raise CanonicalReplayError(
                    "immutable fsync target disappeared after fsync"
                )
            if _FileIdentity.from_stat(before) != _FileIdentity.from_stat(
                after
            ) or _FileIdentity.from_stat(before) != _FileIdentity.from_stat(
                path_after
            ):
                raise CanonicalReplayError(
                    "immutable fsync target changed during fsync"
                )
        finally:
            os.close(fd)
        assert_parent_current()
        content_after, checked_after = read_reserved(
            name, allowed_nlinks=allowed_nlinks
        )
        if content_after != raw or _FileIdentity.from_stat(checked_after) != (
            _FileIdentity.from_stat(expected)
        ):
            raise CanonicalReplayError(
                "immutable fsync target failed stable readback"
            )

    def same_inode(left: os.stat_result, right: os.stat_result) -> bool:
        return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)

    def unlink_reserved_temp(expected: os.stat_result) -> None:
        assert_parent_current()
        os.fsync(parent_fd)
        assert_parent_current()
        current = path_stat(temp_name)
        if current is None or _FileIdentity.from_stat(current) != (
            _FileIdentity.from_stat(expected)
        ):
            raise CanonicalReplayError(
                "immutable publish reserved temp identity changed before cleanup"
            )
        os.unlink(temp_name, dir_fd=parent_fd)

    def verify_committed() -> bool:
        assert_parent_current()
        fsync_exact_file(leaf, allowed_nlinks=frozenset({1}))
        os.fsync(parent_fd)
        assert_parent_current()
        first = read_destination()
        assert_parent_current()
        os.fsync(parent_fd)
        assert_parent_current()
        second = read_destination()
        assert_parent_current()
        return first == raw and second == raw

    def reconcile_exact_state() -> str:
        """Return absent, prepared, or committed for exact reserved paths."""

        assert_parent_current()
        temp_stat = path_stat(temp_name)
        final_stat = path_stat(leaf)
        if temp_stat is None:
            if final_stat is None:
                return "absent"
            final_bytes, checked_final = read_reserved(
                leaf, allowed_nlinks=frozenset({1})
            )
            if final_bytes != raw:
                raise CanonicalReplayError(
                    "immutable destination already exists with different bytes"
                )
            if not same_inode(final_stat, checked_final) or not verify_committed():
                raise CanonicalReplayError(
                    "immutable existing destination changed during readback"
                )
            return "committed"

        temp_bytes, checked_temp = read_reserved(
            temp_name, allowed_nlinks=frozenset({1, 2})
        )
        temp_stat = checked_temp
        final_stat = path_stat(leaf)
        if temp_stat.st_nlink == 2:
            if final_stat is None:
                raise CanonicalReplayError(
                    "immutable reserved temp has an ambiguous second link"
                )
            final_bytes, checked_final = read_reserved(
                leaf, allowed_nlinks=frozenset({2})
            )
            if (
                not same_inode(temp_stat, checked_final)
                or temp_bytes != raw
                or final_bytes != raw
            ):
                raise CanonicalReplayError(
                    "immutable post-link recovery identity or bytes mismatch"
                )
            fsync_exact_file(temp_name, allowed_nlinks=frozenset({2}))
            os.fsync(parent_fd)
            assert_parent_current()
            unlink_reserved_temp(temp_stat)
            os.fsync(parent_fd)
            assert_parent_current()
            if not verify_committed():
                raise CanonicalReplayError(
                    "immutable post-link recovery readback mismatch"
                )
            return "committed"

        if temp_stat.st_nlink != 1:
            raise CanonicalReplayError(
                "immutable reserved temp link count is unsafe"
            )
        if final_stat is None:
            if temp_bytes == raw:
                return "prepared"
            unlink_reserved_temp(temp_stat)
            os.fsync(parent_fd)
            assert_parent_current()
            return "absent"

        if temp_bytes != raw:
            raise CanonicalReplayError(
                "immutable partial reserved temp conflicts with existing destination"
            )
        final_bytes, checked_final = read_reserved(
            leaf, allowed_nlinks=frozenset({1})
        )
        if same_inode(temp_stat, checked_final):
            raise CanonicalReplayError(
                "immutable publish link identities are inconsistent"
            )
        unlink_reserved_temp(temp_stat)
        os.fsync(parent_fd)
        assert_parent_current()
        if final_bytes != raw:
            raise CanonicalReplayError(
                "immutable destination already exists with different bytes"
            )
        if not verify_committed():
            raise CanonicalReplayError(
                "immutable existing destination changed during readback"
            )
        return "committed"

    def create_prepared_temp() -> None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        assert_parent_current()
        try:
            temp_fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            raise CanonicalReplayError(
                f"immutable publish temp creation failed: {exc.strerror}"
            ) from exc
        try:
            temp_stat = os.fstat(temp_fd)
            check_publish_file(
                temp_stat,
                "immutable publish temp",
                allowed_nlinks=frozenset({1}),
            )
            view = memoryview(raw)
            while view:
                written = os.write(temp_fd, view)
                if written <= 0:
                    raise CanonicalReplayError(
                        "immutable publish write made no progress"
                    )
                view = view[written:]
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
        assert_parent_current()

    try:
        assert_parent_current()
        lock_base_flags = (
            os.O_RDWR
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        lock_created = False
        try:
            lock_fd = os.open(
                lock_name,
                lock_base_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            lock_created = True
        except FileExistsError:
            try:
                lock_fd = os.open(
                    lock_name,
                    lock_base_flags,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise CanonicalReplayError(
                    f"immutable publish lock open failed: {exc.strerror}"
                ) from exc
        except OSError as exc:
            raise CanonicalReplayError(
                f"immutable publish lock open failed: {exc.strerror}"
            ) from exc
        if lock_created:
            os.fsync(lock_fd)
            os.fsync(parent_fd)
            assert_parent_current()
        check_publish_file(
            os.fstat(lock_fd),
            "immutable publish lock",
            allowed_nlinks=frozenset({1}),
        )
        if os.fstat(lock_fd).st_size != 0:
            raise CanonicalReplayError("immutable publish lock must be empty")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        lock_held = True
        lock_path_stat = path_stat(lock_name)
        if lock_path_stat is None or _FileIdentity.from_stat(lock_path_stat) != (
            _FileIdentity.from_stat(os.fstat(lock_fd))
        ):
            raise CanonicalReplayError("immutable publish lock path identity changed")
        check_publish_file(
            lock_path_stat,
            "immutable publish lock",
            allowed_nlinks=frozenset({1}),
        )
        if lock_path_stat.st_size != 0:
            raise CanonicalReplayError("immutable publish lock must be empty")
        assert_parent_current()

        linked = False
        try:
            state = reconcile_exact_state()
            if state == "committed":
                return result
            if state == "absent":
                create_prepared_temp()
                state = reconcile_exact_state()
            if state != "prepared":
                raise CanonicalReplayError(
                    "immutable publish did not reach a prepared state"
                )
            fsync_exact_file(temp_name, allowed_nlinks=frozenset({1}))
            os.fsync(parent_fd)
            assert_parent_current()
            os.link(
                temp_name,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            linked = True
            assert_parent_current()
            if reconcile_exact_state() != "committed":
                raise CanonicalReplayError(
                    "immutable publish did not reconcile a committed state"
                )
            return result
        except Exception:
            if linked:
                try:
                    if reconcile_exact_state() == "committed":
                        return result
                except Exception:
                    pass
            raise
    finally:
        if lock_fd is not None:
            try:
                if lock_held:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(parent_fd)


_SOURCE_FIELDS = frozenset({"path", "sha256"})
_DRAFT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "run_id",
        "as_of",
        "window_start",
        "window_end",
        "producer_contract_sha256",
        "registry",
        "factor_set",
        "snapshot_pointer",
        "snapshot_manifest",
        "calendar",
        "pit_manifest",
        "pit_canonical",
        "market_data",
        "code_config_manifest",
        "comparison",
        "stages",
    }
)
_BUNDLE_FIELDS = frozenset(
    (set(_DRAFT_FIELDS) - {"schema_version"})
    | {"schema_version", "recomputed"}
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "registry_sha256",
        "bundle_sha256",
        "factor_set",
        "factor_set_sha256",
        "snapshot_pointer_sha256",
        "snapshot_manifest_sha256",
        "calendar_sha256",
        "pit_manifest_sha256",
        "pit_canonical_sha256",
        "market_data_sha256",
        "code_config_manifest_sha256",
        "producer_contract_sha256",
    }
)
_CONTEXT_FIELDS = frozenset(
    {
        "registry_sha256",
        "factor_set_sha256",
        "snapshot_pointer_sha256",
        "snapshot_manifest_sha256",
        "calendar_sha256",
        "pit_manifest_sha256",
        "pit_canonical_sha256",
        "market_data_sha256",
        "code_config_manifest_sha256",
    }
)
_STAGE_REF_FIELDS = frozenset(
    {"arm", "stage", "path", "sha256", "semantic_sha256"}
)
_STAGE_FIELDS = frozenset(
    {
        "schema_version",
        "arm",
        "stage",
        "run_id",
        "as_of",
        "window_start",
        "window_end",
        "context",
        "predecessor",
        "output",
    }
)


def _source_ref(value: Any, label: str) -> dict[str, str]:
    payload = _exact_fields(value, _SOURCE_FIELDS, label)
    if type(payload["path"]) is not str:
        raise CanonicalReplayError(f"{label}.path must be an exact string")
    return {
        "path": SafeReadSession._validate_absolute(payload["path"], f"{label}.path"),
        "sha256": _sha256(payload["sha256"], f"{label}.sha256"),
    }


def _read_ref(
    session: SafeReadSession,
    value: Any,
    label: str,
    *,
    expected_fields: Iterable[str] | None = None,
) -> tuple[dict[str, str], bytes, dict[str, Any]]:
    ref = _source_ref(value, label)
    raw = session.read_bytes(ref["path"])
    if _sha256_bytes(raw) != ref["sha256"]:
        raise CanonicalReplayError(f"{label} SHA-256 mismatch")
    payload = strict_json_loads(
        raw,
        expected_fields=set(expected_fields) if expected_fields is not None else None,
    )
    return ref, raw, payload


def _registry_slot_identity(
    raw_factor: Mapping[str, Any], *, index: int
) -> dict[str, str]:
    """Derive the protocol-v2 slot from one exact registry record."""

    raw_metadata = raw_factor.get("metadata")
    if raw_metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        raise CanonicalReplayError(
            f"registry factor {index}.metadata must be an object"
        )

    def exact_optional_text(value: Any, label: str) -> str:
        if value is None or value == "":
            return ""
        if type(value) is not str or value != value.strip():
            raise CanonicalReplayError(
                f"registry factor {index}.{label} must be an exact string"
            )
        return value

    family = ""
    for label, value in (
        ("metadata.factor_family", metadata.get("factor_family")),
        ("metadata.governance_family", metadata.get("governance_family")),
        ("category", raw_factor.get("category")),
    ):
        candidate = exact_optional_text(value, label)
        if candidate:
            family = candidate
            break
    cluster = exact_optional_text(
        metadata.get("dominant_primitive_cluster"),
        "metadata.dominant_primitive_cluster",
    )
    if not cluster:
        dominant = metadata.get("dominant_primitives", []) or []
        if not isinstance(dominant, list):
            raise CanonicalReplayError(
                f"registry factor {index}.metadata.dominant_primitives "
                "must be a list"
            )
        normalized = [
            _text(
                item,
                f"registry factor {index}.metadata.dominant_primitives[]",
            )
            for item in dominant
        ]
        if len(normalized) != len(set(normalized)):
            raise CanonicalReplayError(
                f"registry factor {index}.metadata.dominant_primitives "
                "must be distinct"
            )
        if normalized:
            cluster = "+".join(sorted(normalized))
    return {
        "family": family,
        "dominant_primitive_cluster": cluster,
        "slot": f"{family}::{cluster}" if family and cluster else "",
    }


def _registry_factor_set(
    raw: bytes,
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    registry = strict_json_loads(
        raw,
        expected_fields={"schema_version", "metadata", "factors"},
    )
    if registry.get("schema_version") != "mined-factor-registry.v1":
        raise CanonicalReplayError("unsupported canonical registry schema")
    if not isinstance(registry.get("metadata"), dict):
        raise CanonicalReplayError("registry metadata must be an object")
    factors = registry.get("factors")
    if not isinstance(factors, list):
        raise CanonicalReplayError("registry factors must be a list")
    names: set[str] = set()
    production: list[str] = []
    records_by_name: dict[str, dict[str, Any]] = {}
    for index, raw_factor in enumerate(factors):
        if not isinstance(raw_factor, dict):
            raise CanonicalReplayError(f"registry factor {index} must be an object")
        name = _text(raw_factor.get("name"), f"registry factor {index}.name")
        state = _text(raw_factor.get("state"), f"registry factor {index}.state")
        if name in names:
            raise CanonicalReplayError("registry factor names must be distinct")
        names.add(name)
        slot_identity = _registry_slot_identity(raw_factor, index=index)
        records_by_name[name] = {
            "name": name,
            "state": state,
            "record_sha256": _sha256_value(raw_factor),
            "record": dict(raw_factor),
            **slot_identity,
        }
        if state == "production_factor":
            production.append(name)
    return registry, sorted(production), records_by_name


def _validate_calendar(payload: dict[str, Any]) -> list[str]:
    _exact_fields(
        payload,
        {"schema_version", "market", "open_days"},
        "calendar",
    )
    if payload["schema_version"] != "independent-open-day-calendar.v1":
        raise CanonicalReplayError("unsupported independent calendar schema")
    if payload["market"] != "CN":
        raise CanonicalReplayError("independent calendar market must be CN")
    days = _ordered_distinct_strings(payload["open_days"], "calendar.open_days")
    for item in days:
        _iso_date(item, "calendar.open_days[]")
    if not days:
        raise CanonicalReplayError("independent calendar must not be empty")
    return days


def _validate_pit_canonical(payload: dict[str, Any], as_of: str) -> list[str]:
    _exact_fields(
        payload,
        {"schema_version", "as_of", "symbols"},
        "PIT canonical",
    )
    if payload["schema_version"] != "cn-pit-canonical.v1":
        raise CanonicalReplayError("unsupported PIT canonical schema")
    if _iso_date(payload["as_of"], "PIT canonical as_of") != as_of:
        raise CanonicalReplayError("PIT canonical as_of mismatch")
    symbols = _ordered_distinct_strings(payload["symbols"], "PIT canonical symbols")
    if not symbols:
        raise CanonicalReplayError("PIT canonical symbols must not be empty")
    return symbols


def _validate_market_data(
    payload: dict[str, Any],
    *,
    open_days: Sequence[str],
    pit_symbols: Sequence[str],
    window_start: str,
    window_end: str,
) -> dict[str, Any]:
    _exact_fields(
        payload,
        {"schema_version", "dates", "returns"},
        "market data",
    )
    if payload["schema_version"] != "factor-governance-market-data.v1":
        raise CanonicalReplayError("unsupported canonical market-data schema")
    dates = _ordered_distinct_strings(payload["dates"], "market data dates")
    expected_dates = [
        item for item in open_days if window_start <= item <= window_end
    ]
    if (
        not expected_dates
        or dates != expected_dates
        or dates[0] != window_start
        or dates[-1] != window_end
    ):
        raise CanonicalReplayError(
            "market data must contain the exact complete open-day replay window"
        )
    returns = payload["returns"]
    if not isinstance(returns, dict):
        raise CanonicalReplayError("market data returns must be an object")
    if set(returns) != set(pit_symbols):
        raise CanonicalReplayError("market data symbols must exactly match PIT symbols")
    normalized: dict[str, list[float]] = {}
    for symbol in sorted(returns):
        values = returns[symbol]
        if not isinstance(values, list) or len(values) != len(dates):
            raise CanonicalReplayError("market return series must align with dates")
        normalized[symbol] = [
            _finite(item, f"market return {symbol}") for item in values
        ]
    return {"dates": dates, "returns": normalized}


def _validate_snapshot_graph(
    *,
    pointer: dict[str, Any],
    pointer_ref: Mapping[str, str],
    manifest: dict[str, Any],
    manifest_ref: Mapping[str, str],
    calendar_ref: Mapping[str, str],
    pit_manifest: dict[str, Any],
    pit_manifest_ref: Mapping[str, str],
    pit_canonical_ref: Mapping[str, str],
    market_ref: Mapping[str, str],
    as_of: str,
    pit_symbols: Sequence[str],
) -> None:
    del pointer_ref
    _exact_fields(
        pointer,
        {"schema_version", "snapshot_id", "manifest_path", "manifest_sha256"},
        "snapshot pointer",
    )
    if pointer["schema_version"] != "strict-parquet-snapshot-pointer.v1":
        raise CanonicalReplayError("unsupported snapshot pointer schema")
    _text(pointer["snapshot_id"], "snapshot pointer id")
    if pointer["manifest_path"] != manifest_ref["path"]:
        raise CanonicalReplayError("snapshot pointer manifest path mismatch")
    if pointer["manifest_sha256"] != manifest_ref["sha256"]:
        raise CanonicalReplayError("snapshot pointer manifest SHA mismatch")

    _exact_fields(
        manifest,
        {
            "schema_version",
            "snapshot_id",
            "latest_complete_trade_date",
            "calendar_path",
            "calendar_sha256",
            "pit_manifest_path",
            "pit_manifest_sha256",
            "pit_canonical_path",
            "pit_canonical_sha256",
            "market_data_path",
            "market_data_sha256",
        },
        "snapshot manifest",
    )
    if manifest["schema_version"] != "strict-parquet-snapshot-manifest.v1":
        raise CanonicalReplayError("unsupported snapshot manifest schema")
    if manifest["snapshot_id"] != pointer["snapshot_id"]:
        raise CanonicalReplayError("snapshot id mismatch")
    if _iso_date(
        manifest["latest_complete_trade_date"],
        "latest complete trade date",
    ) != as_of:
        raise CanonicalReplayError("snapshot date does not match replay as_of")
    bindings = (
        ("calendar", calendar_ref),
        ("pit_manifest", pit_manifest_ref),
        ("pit_canonical", pit_canonical_ref),
        ("market_data", market_ref),
    )
    for prefix, ref in bindings:
        if manifest[f"{prefix}_path"] != ref["path"]:
            raise CanonicalReplayError(f"snapshot {prefix} path mismatch")
        if manifest[f"{prefix}_sha256"] != ref["sha256"]:
            raise CanonicalReplayError(f"snapshot {prefix} SHA mismatch")

    _exact_fields(
        pit_manifest,
        {
            "schema_version",
            "as_of",
            "canonical_path",
            "canonical_sha256",
            "symbols",
        },
        "PIT manifest",
    )
    if pit_manifest["schema_version"] != "cn-pit-manifest.v1":
        raise CanonicalReplayError("unsupported PIT manifest schema")
    if _iso_date(pit_manifest["as_of"], "PIT manifest as_of") != as_of:
        raise CanonicalReplayError("PIT manifest as_of mismatch")
    if pit_manifest["canonical_path"] != pit_canonical_ref["path"]:
        raise CanonicalReplayError("PIT canonical path mismatch")
    if pit_manifest["canonical_sha256"] != pit_canonical_ref["sha256"]:
        raise CanonicalReplayError("PIT canonical SHA mismatch")
    if _ordered_distinct_strings(
        pit_manifest["symbols"], "PIT manifest symbols"
    ) != list(pit_symbols):
        raise CanonicalReplayError("PIT manifest/canonical symbol mismatch")


def _validate_code_config_manifest(
    payload: dict[str, Any], session: SafeReadSession
) -> list[dict[str, str]]:
    _exact_fields(
        payload,
        {"schema_version", "files"},
        "code/config manifest",
    )
    if payload["schema_version"] != "factor-governance-code-config-manifest.v1":
        raise CanonicalReplayError("unsupported code/config manifest schema")
    files = payload["files"]
    if not isinstance(files, list):
        raise CanonicalReplayError("code/config manifest files must be a list")
    normalized: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for index, raw in enumerate(files):
        item = _exact_fields(raw, {"role", "path", "sha256"}, f"code file {index}")
        role = _text(item["role"], f"code file {index}.role")
        ref = _source_ref(
            {"path": item["path"], "sha256": item["sha256"]},
            f"code file {index}",
        )
        content = session.read_bytes(ref["path"])
        if _sha256_bytes(content) != ref["sha256"]:
            raise CanonicalReplayError(f"code/config file {role} SHA mismatch")
        identity = session.identity(ref["path"])
        if ref["path"] in seen_paths or identity in seen_inodes:
            raise CanonicalReplayError("code/config file paths and inodes must be unique")
        seen_paths.add(ref["path"])
        seen_inodes.add(identity)
        normalized.append({"role": role, **ref})
    if [item["role"] for item in normalized] != list(CODE_CONFIG_ROLES):
        raise CanonicalReplayError(
            "code/config manifest must contain the fixed complete role sequence"
        )
    return normalized


def _validate_forward_cohorts(
    value: Any,
    label: str,
    *,
    open_days: Sequence[str],
    cutoff: str,
) -> int:
    if not isinstance(value, list):
        raise CanonicalReplayError(f"{label} must be a list")
    cohorts: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    open_day_index = {item: index for index, item in enumerate(open_days)}
    for index, raw in enumerate(value):
        item = _exact_fields(
            raw,
            {"cohort_id", "start", "end", "horizon_days"},
            f"{label}[{index}]",
        )
        cohort_id = _text(item["cohort_id"], f"{label}[{index}].cohort_id")
        start = _iso_date(item["start"], f"{label}[{index}].start")
        end = _iso_date(item["end"], f"{label}[{index}].end")
        if start > cutoff or end > cutoff:
            raise CanonicalReplayError(
                "forward cohort dates must not be after the replay cutoff"
            )
        if _integer(item["horizon_days"], f"{label}[{index}].horizon_days") != 30:
            raise CanonicalReplayError("forward cohorts must have a 30-day horizon")
        if (
            end <= start
            or cohort_id in seen
            or start not in open_day_index
            or end not in open_day_index
            or open_day_index[end] - open_day_index[start] + 1 != 30
        ):
            raise CanonicalReplayError("forward cohorts are invalid or duplicated")
        cohorts.append((start, end, cohort_id))
        seen.add(cohort_id)
    cohorts.sort()
    nonoverlap = 0
    last_end: str | None = None
    for start, end, _ in cohorts:
        if last_end is None or start > last_end:
            nonoverlap += 1
            last_end = end
    return nonoverlap


def _validate_walk_forward(
    value: Any,
    label: str,
    *,
    open_days: Sequence[str],
    cutoff: str,
) -> bool:
    payload = _exact_fields(
        value,
        {"purged", "purge_days", "embargo_days", "folds"},
        label,
    )
    if type(payload["purged"]) is not bool:
        raise CanonicalReplayError(f"{label}.purged must be an exact boolean")
    purge_days = _integer(payload["purge_days"], f"{label}.purge_days")
    embargo_days = _integer(payload["embargo_days"], f"{label}.embargo_days")
    folds = payload["folds"]
    if type(folds) is not list:
        raise CanonicalReplayError(f"{label}.folds must be an exact list")
    semantic_passed = (
        payload["purged"]
        and purge_days >= 30
        and embargo_days == 30
        and bool(folds)
    )
    for index, raw in enumerate(folds):
        item = _exact_fields(
            raw,
            {"train_end", "validation_start", "validation_end", "evidence_hash"},
            f"{label}.folds[{index}]",
        )
        train_end = date.fromisoformat(
            _iso_date(item["train_end"], f"{label}.fold.train_end")
        )
        validation_start = date.fromisoformat(
            _iso_date(item["validation_start"], f"{label}.fold.validation_start")
        )
        validation_end = date.fromisoformat(
            _iso_date(item["validation_end"], f"{label}.fold.validation_end")
        )
        _sha256(item["evidence_hash"], f"{label}.fold.evidence_hash")
        if any(
            observed.isoformat() > cutoff
            for observed in (train_end, validation_start, validation_end)
        ):
            raise CanonicalReplayError(
                f"{label} fold dates must not be after the replay cutoff"
            )
        if (
            validation_start <= train_end
            or validation_end <= validation_start
            or (validation_start - train_end).days < purge_days
            or train_end.isoformat() not in open_days
            or validation_start.isoformat() not in open_days
            or validation_end.isoformat() not in open_days
        ):
            semantic_passed = False
    return semantic_passed


def _bh_passed(records: Sequence[dict[str, Any]], q: float = 0.10) -> set[str]:
    by_family: dict[str, list[tuple[float, str]]] = {}
    for item in records:
        by_family.setdefault(item["family"], []).append((item["p_value"], item["name"]))
    result: set[str] = set()
    for members in by_family.values():
        ordered = sorted(members)
        last_passing = 0
        for rank, (p_value, _) in enumerate(ordered, start=1):
            if p_value <= q * rank / len(ordered):
                last_passing = rank
        result.update(name for _, name in ordered[:last_passing])
    return result


def _validate_quant_output(
    value: Any,
    arm: str,
    *,
    open_days: Sequence[str],
    cutoff: str,
    registry_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _exact_fields(
        value,
        {"schema_version", "selected_factors", "factor_records"},
        f"arm {arm} quant output",
    )
    if payload["schema_version"] != "factor-governance-quant-stage-output.v1":
        raise CanonicalReplayError("unsupported quant stage output schema")
    selected = _ordered_distinct_strings(
        payload["selected_factors"], f"arm {arm} selected factors"
    )
    raw_records = payload["factor_records"]
    if not isinstance(raw_records, list):
        raise CanonicalReplayError("quant factor_records must be a list")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_records):
        item = _exact_fields(
            raw,
            {
                "name",
                "family",
                "slot",
                "registry_state",
                "registry_record_sha256",
                "p_value",
                "health_failure_windows",
                "month_end_rankic_dates",
                "forward_cohorts",
                "walk_forward",
            },
            f"arm {arm} factor record {index}",
        )
        name = _text(item["name"], "factor name")
        if name in names:
            raise CanonicalReplayError("quant factor record names must be distinct")
        names.add(name)
        registry_identity = registry_records.get(name)
        if registry_identity is None:
            raise CanonicalReplayError(
                f"arm {arm} factor {name} is absent from the complete registry"
            )
        registry_state = _text(
            item["registry_state"], "factor registry_state"
        )
        registry_record_sha = _sha256(
            item["registry_record_sha256"],
            "factor registry_record_sha256",
        )
        if registry_state != registry_identity["state"]:
            raise CanonicalReplayError(
                f"arm {arm} factor {name} registry state mismatch"
            )
        if registry_record_sha != registry_identity["record_sha256"]:
            raise CanonicalReplayError(
                f"arm {arm} factor {name} registry record identity mismatch"
            )
        family = _text(item["family"], "factor family")
        slot = _text(item["slot"], "factor slot")
        registry_family = registry_identity["family"]
        registry_slot = registry_identity["slot"]
        if not registry_family or not registry_identity[
            "dominant_primitive_cluster"
        ] or not registry_slot:
            raise CanonicalReplayError(
                f"arm {arm} factor {name} registry family/slot identity is missing"
            )
        if family != registry_family or slot != registry_slot:
            raise CanonicalReplayError(
                f"arm {arm} factor {name} family/slot does not match registry identity"
            )
        p_value = _finite(item["p_value"], "factor p_value")
        if not 0.0 <= p_value <= 1.0:
            raise CanonicalReplayError("factor p_value must be in [0,1]")
        failure_windows = _ordered_distinct_strings(
            item["health_failure_windows"], "health failure windows"
        )
        month_ends = _ordered_distinct_strings(
            item["month_end_rankic_dates"], "month-end RankIC dates"
        )
        month_end_by_period: dict[str, str] = {}
        for open_day in open_days:
            month_end_by_period[open_day[:7]] = open_day
        for observed in month_ends:
            valid_observed = _iso_date(observed, "month-end RankIC date")
            if valid_observed > cutoff:
                raise CanonicalReplayError(
                    "month-end RankIC date must not be after the replay cutoff"
                )
            if month_end_by_period.get(valid_observed[:7]) != valid_observed:
                raise CanonicalReplayError(
                    "month-end RankIC date is not the independent calendar month end"
                )
        cohort_count = _validate_forward_cohorts(
            item["forward_cohorts"],
            "forward cohorts",
            open_days=open_days,
            cutoff=cutoff,
        )
        walk_forward_passed = _validate_walk_forward(
            item["walk_forward"],
            "purged walk-forward",
            open_days=open_days,
            cutoff=cutoff,
        )
        records.append(
            {
                "name": name,
                "family": family,
                "slot": slot,
                "registry_state": registry_state,
                "registry_record_sha256": registry_record_sha,
                "p_value": p_value,
                "failure_count": len(failure_windows),
                "mature": len(month_ends) >= 12 or cohort_count >= 8,
                "walk_forward_passed": walk_forward_passed,
            }
        )
    fdr_passed = _bh_passed(records)
    eligible = sorted(
        item["name"]
        for item in records
        if item["name"] in fdr_passed
        and item["failure_count"] < 2
        and item["mature"]
        and item["walk_forward_passed"]
    )
    if selected != eligible:
        raise CanonicalReplayError(
            f"arm {arm} selected factors do not match raw health/FDR/maturity/walk-forward recomputation"
        )
    return {"selected_factors": selected, "records": records}


def _validate_theme_output(value: Any, pit_symbols: set[str]) -> list[str]:
    payload = _exact_fields(
        value,
        {"schema_version", "eligible_symbols"},
        "theme output",
    )
    if payload["schema_version"] != "factor-governance-theme-stage-output.v1":
        raise CanonicalReplayError("unsupported theme output schema")
    symbols = _ordered_distinct_strings(payload["eligible_symbols"], "theme symbols")
    if not set(symbols).issubset(pit_symbols):
        raise CanonicalReplayError("theme output contains a non-PIT symbol")
    return symbols


def _validate_bayesian_output(value: Any, eligible_symbols: set[str]) -> dict[str, float]:
    payload = _exact_fields(
        value,
        {"schema_version", "posterior_scores"},
        "Bayesian output",
    )
    if payload["schema_version"] != "factor-governance-bayesian-stage-output.v1":
        raise CanonicalReplayError("unsupported Bayesian output schema")
    scores = payload["posterior_scores"]
    if not isinstance(scores, dict) or set(scores) != eligible_symbols:
        raise CanonicalReplayError("Bayesian scores must exactly match theme symbols")
    return {key: _finite(scores[key], f"posterior score {key}") for key in sorted(scores)}


def _validate_risk_output(
    value: Any,
    *,
    market: Mapping[str, Any],
    eligible_symbols: set[str],
) -> dict[str, Any]:
    payload = _exact_fields(
        value,
        {"schema_version", "dates", "adjusted_returns"},
        "RiskGuard output",
    )
    if payload["schema_version"] != "factor-governance-risk-stage-output.v1":
        raise CanonicalReplayError("unsupported RiskGuard output schema")
    dates = _ordered_distinct_strings(payload["dates"], "RiskGuard dates")
    if dates != market["dates"]:
        raise CanonicalReplayError("RiskGuard dates must exactly match market dates")
    returns = payload["adjusted_returns"]
    if not isinstance(returns, dict) or set(returns) != eligible_symbols:
        raise CanonicalReplayError("RiskGuard symbols must exactly match eligible symbols")
    normalized: dict[str, list[float]] = {}
    for symbol in sorted(returns):
        values = returns[symbol]
        if not isinstance(values, list) or len(values) != len(dates):
            raise CanonicalReplayError("RiskGuard return series must align with dates")
        normalized[symbol] = [
            _finite(item, f"RiskGuard adjusted return {symbol}") for item in values
        ]
    return {"dates": dates, "adjusted_returns": normalized}


def _same_number(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def _validate_portfolio_output(
    value: Any,
    *,
    risk: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact_fields(
        value,
        {
            "schema_version",
            "dates",
            "weights",
            "costs",
            "after_cost_returns",
            "turnover",
            "slippage",
            "tail_risk",
        },
        "PortfolioConstructor output",
    )
    if payload["schema_version"] != "factor-governance-portfolio-stage-output.v1":
        raise CanonicalReplayError("unsupported PortfolioConstructor output schema")
    dates = _ordered_distinct_strings(payload["dates"], "portfolio dates")
    if dates != risk["dates"]:
        raise CanonicalReplayError("portfolio dates must exactly match RiskGuard dates")
    weights = payload["weights"]
    costs = payload["costs"]
    reported = payload["after_cost_returns"]
    if not all(isinstance(item, list) for item in (weights, costs, reported)):
        raise CanonicalReplayError("portfolio series must be lists")
    if not (len(weights) == len(costs) == len(reported) == len(dates)):
        raise CanonicalReplayError("portfolio series must align with dates")
    symbols = set(risk["adjusted_returns"])
    previous = {symbol: 0.0 for symbol in symbols}
    recomputed: list[float] = []
    turnover = 0.0
    normalized_weights: list[dict[str, float]] = []
    normalized_costs: list[float] = []
    for index, raw_weights in enumerate(weights):
        if not isinstance(raw_weights, dict) or not set(raw_weights).issubset(symbols):
            raise CanonicalReplayError("portfolio weights contain an unknown symbol")
        current = {
            symbol: _finite(raw_weights.get(symbol, 0.0), f"weight {symbol}")
            for symbol in sorted(symbols)
        }
        if any(number < 0.0 or number > 1.0 for number in current.values()):
            raise CanonicalReplayError("portfolio weights must be in [0,1]")
        if sum(current.values()) > 1.0 + 1e-12:
            raise CanonicalReplayError("portfolio weights exceed one")
        cost = _finite(costs[index], "portfolio cost")
        if cost < 0.0:
            raise CanonicalReplayError("portfolio costs must be non-negative")
        gross = sum(
            current[symbol] * risk["adjusted_returns"][symbol][index]
            for symbol in symbols
        )
        after_cost = gross - cost
        reported_value = _finite(reported[index], "after-cost return")
        if not _same_number(reported_value, after_cost):
            raise CanonicalReplayError(
                "after-cost return does not equal weights times adjusted returns minus costs"
            )
        turnover += 0.5 * sum(
            abs(current[symbol] - previous[symbol]) for symbol in symbols
        )
        previous = current
        normalized_weights.append(current)
        normalized_costs.append(cost)
        recomputed.append(after_cost)
    slippage = sum(normalized_costs)
    tail_risk = abs(min(0.0, min(recomputed)))
    if not _same_number(_finite(payload["turnover"], "turnover"), turnover):
        raise CanonicalReplayError("portfolio turnover recomputation mismatch")
    if not _same_number(_finite(payload["slippage"], "slippage"), slippage):
        raise CanonicalReplayError("portfolio slippage recomputation mismatch")
    if not _same_number(_finite(payload["tail_risk"], "tail risk"), tail_risk):
        raise CanonicalReplayError("portfolio tail-risk recomputation mismatch")
    return {
        "dates": dates,
        "after_cost_returns": recomputed,
        "turnover": turnover,
        "slippage": slippage,
        "tail_risk": tail_risk,
    }


def _validate_stage_graph(
    *,
    stage_refs_value: Any,
    session: SafeReadSession,
    expected: Mapping[str, str],
    run_id: str,
    as_of: str,
    window_start: str,
    window_end: str,
    market: Mapping[str, Any],
    pit_symbols: Sequence[str],
    open_days: Sequence[str],
    registry_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(stage_refs_value, list):
        raise CanonicalReplayError("stage references must be a list")
    expected_order = [
        (arm, stage) for arm in ARM_NAMES for stage in CONTROL_CHAIN_STAGES
    ]
    if len(stage_refs_value) != len(expected_order):
        raise CanonicalReplayError("canonical graph must contain exactly 20 stages")
    refs: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    seen_inodes: set[tuple[int, int]] = set()
    stages_by_arm: dict[str, dict[str, Any]] = {arm: {} for arm in ARM_NAMES}
    recomputed: dict[str, Any] = {}
    for index, raw_ref in enumerate(stage_refs_value):
        item = _exact_fields(raw_ref, _STAGE_REF_FIELDS, f"stage reference {index}")
        arm = _text(item["arm"], f"stage reference {index}.arm")
        stage = _text(item["stage"], f"stage reference {index}.stage")
        if (arm, stage) != expected_order[index]:
            raise CanonicalReplayError(
                "stage references must use the exact A/B/C/D control-chain order"
            )
        ref = {
            "arm": arm,
            "stage": stage,
            "path": SafeReadSession._validate_absolute(
                _text(item["path"], f"stage reference {index}.path"),
                f"stage reference {index}.path",
            ),
            "sha256": _sha256(item["sha256"], f"stage reference {index}.sha256"),
            "semantic_sha256": _sha256(
                item["semantic_sha256"],
                f"stage reference {index}.semantic_sha256",
            ),
        }
        if not session._is_private_parts(session._parts(ref["path"])):
            raise CanonicalReplayError(
                "all canonical stage artifacts must reside under private_root"
            )
        raw = session.read_bytes(ref["path"])
        identity = session.identity(ref["path"])
        if ref["path"] in seen_paths or identity in seen_inodes:
            raise CanonicalReplayError(
                "all 20 stage paths and device/inode identities must be globally unique"
            )
        seen_paths.add(ref["path"])
        seen_inodes.add(identity)
        if _sha256_bytes(raw) != ref["sha256"]:
            raise CanonicalReplayError(f"arm {arm} {stage} stage SHA mismatch")
        stage_payload = strict_json_loads(raw, expected_fields=set(_STAGE_FIELDS))
        if stage_payload["schema_version"] != STAGE_SCHEMA_VERSION:
            raise CanonicalReplayError("unsupported canonical stage schema")
        exact_identifiers = {
            "arm": arm,
            "stage": stage,
            "run_id": run_id,
            "as_of": as_of,
            "window_start": window_start,
            "window_end": window_end,
        }
        for key, expected_value in exact_identifiers.items():
            if stage_payload[key] != expected_value:
                raise CanonicalReplayError(f"stage {key} identity mismatch")
        context = _exact_fields(
            stage_payload["context"], _CONTEXT_FIELDS, "stage context"
        )
        normalized_context = {
            key: _sha256(context[key], f"stage context {key}")
            for key in sorted(_CONTEXT_FIELDS)
        }
        if normalized_context != dict(expected):
            raise CanonicalReplayError("stage context identity mismatch")
        predecessor = _exact_fields(
            stage_payload["predecessor"],
            {"kind", "byte_sha256", "semantic_sha256"},
            "stage predecessor",
        )
        if stage == CONTROL_CHAIN_STAGES[0]:
            expected_predecessor = {
                "kind": "genesis",
                "byte_sha256": GENESIS_SHA256,
                "semantic_sha256": GENESIS_SHA256,
            }
        else:
            previous_stage = CONTROL_CHAIN_STAGES[
                CONTROL_CHAIN_STAGES.index(stage) - 1
            ]
            previous = stages_by_arm[arm][previous_stage]
            expected_predecessor = {
                "kind": "stage",
                "byte_sha256": previous["sha256"],
                "semantic_sha256": previous["semantic_sha256"],
            }
        if predecessor != expected_predecessor:
            raise CanonicalReplayError(
                f"arm {arm} {stage} predecessor byte/semantic identity mismatch"
            )
        output = stage_payload["output"]
        semantic_sha256 = _sha256_value(output)
        if semantic_sha256 != ref["semantic_sha256"]:
            raise CanonicalReplayError(f"arm {arm} {stage} semantic SHA mismatch")
        refs.append(ref)
        stages_by_arm[arm][stage] = ref

        arm_result = recomputed.setdefault(arm, {})
        if stage == "quant":
            arm_result[stage] = _validate_quant_output(
                output,
                arm,
                open_days=open_days,
                cutoff=window_end,
                registry_records=registry_records,
            )
        elif stage == "theme":
            arm_result[stage] = {
                "eligible_symbols": _validate_theme_output(
                    output, set(pit_symbols)
                )
            }
        elif stage == "bayesian":
            eligible = set(arm_result["theme"]["eligible_symbols"])
            arm_result[stage] = {
                "posterior_scores": _validate_bayesian_output(output, eligible)
            }
        elif stage == "risk_guard":
            eligible = set(arm_result["theme"]["eligible_symbols"])
            arm_result[stage] = _validate_risk_output(
                output,
                market=market,
                eligible_symbols=eligible,
            )
        else:
            arm_result[stage] = _validate_portfolio_output(
                output,
                risk=arm_result["risk_guard"],
            )
    return {"refs": refs, "arms": recomputed, "stage_inodes": seen_inodes}


def _validate_abcd(
    *,
    arms: Mapping[str, Any],
    factor_set: Sequence[str],
    comparison_value: Any,
    registry_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    comparison = _exact_fields(
        comparison_value,
        {"incumbent", "challenger", "slot"},
        "A/B/C/D comparison",
    )
    incumbent = _text(comparison["incumbent"], "comparison incumbent")
    challenger = _text(comparison["challenger"], "comparison challenger")
    slot = _text(comparison["slot"], "comparison slot")
    if incumbent == challenger:
        raise CanonicalReplayError("one-slot comparison names must differ")
    selected = {
        arm: list(arms[arm]["quant"]["selected_factors"])
        for arm in ARM_NAMES
    }
    expected_a = list(factor_set)
    if incumbent not in expected_a or challenger in expected_a:
        raise CanonicalReplayError("comparison names do not identify one registry slot")
    expected_b = sorted(name for name in expected_a if name != incumbent)
    expected_c = sorted([*expected_b, challenger])
    if (
        selected["A"] != expected_a
        or selected["B"] != expected_b
        or selected["C"] != expected_c
        or selected["D"] != expected_c
    ):
        raise CanonicalReplayError(
            "A/B/C/D does not encode exactly one one-slot replacement"
        )
    record_maps = {
        arm: {item["name"]: item for item in arms[arm]["quant"]["records"]}
        for arm in ARM_NAMES
    }
    incumbent_record = record_maps["A"].get(incumbent)
    challenger_record = record_maps["C"].get(challenger)
    if incumbent_record is None or challenger_record is None:
        raise CanonicalReplayError("one-slot factor records are missing")
    incumbent_identity = registry_records.get(incumbent)
    challenger_identity = registry_records.get(challenger)
    if incumbent_identity is None:
        raise CanonicalReplayError("production incumbent is absent from registry")
    if incumbent_identity["state"] != "production_factor":
        raise CanonicalReplayError(
            "production incumbent registry state is not production_factor"
        )
    if challenger_identity is None:
        raise CanonicalReplayError("challenger is absent from complete registry")
    if challenger_identity["state"] not in ALLOWED_CHALLENGER_REGISTRY_STATES:
        raise CanonicalReplayError(
            "challenger registry state is not an explicitly permitted candidate state"
        )
    if any(
        not identity[field]
        for identity in (incumbent_identity, challenger_identity)
        for field in ("family", "dominant_primitive_cluster", "slot")
    ):
        raise CanonicalReplayError(
            "one-slot registry family/slot identity is missing"
        )
    if (
        incumbent_record["registry_record_sha256"]
        != incumbent_identity["record_sha256"]
        or challenger_record["registry_record_sha256"]
        != challenger_identity["record_sha256"]
    ):
        raise CanonicalReplayError("one-slot registry record identity mismatch")
    if (
        incumbent_identity["slot"] != slot
        or challenger_identity["slot"] != slot
        or incumbent_identity["family"] != challenger_identity["family"]
        or incumbent_identity["dominant_primitive_cluster"]
        != challenger_identity["dominant_primitive_cluster"]
        or incumbent_record["slot"] != incumbent_identity["slot"]
        or challenger_record["slot"] != challenger_identity["slot"]
        or incumbent_record["family"] != incumbent_identity["family"]
        or challenger_record["family"] != challenger_identity["family"]
    ):
        raise CanonicalReplayError("one-slot factor family/slot identity mismatch")
    return {
        "incumbent": incumbent,
        "challenger": challenger,
        "family": incumbent_identity["family"],
        "dominant_primitive_cluster": incumbent_identity[
            "dominant_primitive_cluster"
        ],
        "slot": slot,
        "incumbent_registry_state": incumbent_identity["state"],
        "incumbent_registry_record_sha256": incumbent_identity[
            "record_sha256"
        ],
        "challenger_registry_state": challenger_identity["state"],
        "challenger_registry_record_sha256": challenger_identity[
            "record_sha256"
        ],
        "arm_factor_sets": selected,
    }


def _validate_replay_payload(
    *,
    payload: dict[str, Any],
    schema_version: str,
    session: SafeReadSession,
    registry_path: str,
    registry_raw: bytes,
) -> dict[str, Any]:
    is_bundle = schema_version == BUNDLE_SCHEMA_VERSION
    expected_fields = _BUNDLE_FIELDS if is_bundle else _DRAFT_FIELDS
    _exact_fields(payload, expected_fields, "canonical replay payload")
    if payload["schema_version"] != schema_version:
        raise CanonicalReplayError("canonical replay payload schema mismatch")
    evidence_id = _text(payload["evidence_id"], "evidence_id")
    if (
        len(evidence_id) > 128
        or evidence_id[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for char in evidence_id
        )
    ):
        raise CanonicalReplayError("evidence_id is not a safe logical identifier")
    run_id = _text(payload["run_id"], "run_id")
    as_of = _iso_date(payload["as_of"], "as_of")
    window_start = _iso_date(payload["window_start"], "window_start")
    window_end = _iso_date(payload["window_end"], "window_end")
    if not window_start <= window_end or window_end != as_of:
        raise CanonicalReplayError("replay date window chronology is invalid")
    contract_hash = _sha256(
        payload["producer_contract_sha256"], "producer contract SHA"
    )
    if contract_hash != producer_contract_sha256():
        raise CanonicalReplayError("producer contract SHA mismatch")

    registry_ref = _source_ref(payload["registry"], "registry reference")
    if registry_ref["path"] != registry_path:
        raise CanonicalReplayError("registry reference is not the explicit registry path")
    registry_sha = _sha256_bytes(registry_raw)
    if registry_ref["sha256"] != registry_sha:
        raise CanonicalReplayError("registry reference SHA mismatch")
    _, production_factors, registry_records = _registry_factor_set(registry_raw)
    factor_set = _ordered_distinct_strings(payload["factor_set"], "factor_set")
    if factor_set != production_factors:
        raise CanonicalReplayError("factor_set does not match canonical registry bytes")

    source_payloads: dict[str, dict[str, Any]] = {}
    source_refs: dict[str, dict[str, str]] = {"registry": registry_ref}
    source_fields: dict[str, set[str]] = {
        "snapshot_pointer": {
            "schema_version",
            "snapshot_id",
            "manifest_path",
            "manifest_sha256",
        },
        "snapshot_manifest": {
            "schema_version",
            "snapshot_id",
            "latest_complete_trade_date",
            "calendar_path",
            "calendar_sha256",
            "pit_manifest_path",
            "pit_manifest_sha256",
            "pit_canonical_path",
            "pit_canonical_sha256",
            "market_data_path",
            "market_data_sha256",
        },
        "calendar": {"schema_version", "market", "open_days"},
        "pit_manifest": {
            "schema_version",
            "as_of",
            "canonical_path",
            "canonical_sha256",
            "symbols",
        },
        "pit_canonical": {"schema_version", "as_of", "symbols"},
        "market_data": {"schema_version", "dates", "returns"},
        "code_config_manifest": {"schema_version", "files"},
    }
    for key, fields in source_fields.items():
        ref, _, source = _read_ref(
            session,
            payload[key],
            key.replace("_", " "),
            expected_fields=fields,
        )
        source_refs[key] = ref
        source_payloads[key] = source
    primary_paths = [item["path"] for item in source_refs.values()]
    primary_inodes = [session.identity(item["path"]) for item in source_refs.values()]
    if len(primary_paths) != len(set(primary_paths)) or len(primary_inodes) != len(
        set(primary_inodes)
    ):
        raise CanonicalReplayError("primary evidence source paths/inodes must be unique")

    open_days = _validate_calendar(source_payloads["calendar"])
    open_day_set = set(open_days)
    missing_cut_points = [
        label
        for label, value in (
            ("window_start", window_start),
            ("window_end", window_end),
            ("as_of", as_of),
        )
        if value not in open_day_set
    ]
    if missing_cut_points:
        raise CanonicalReplayError(
            "replay calendar is missing required open-day cut points: "
            + ", ".join(missing_cut_points)
        )
    pit_symbols = _validate_pit_canonical(
        source_payloads["pit_canonical"], as_of
    )
    _validate_snapshot_graph(
        pointer=source_payloads["snapshot_pointer"],
        pointer_ref=source_refs["snapshot_pointer"],
        manifest=source_payloads["snapshot_manifest"],
        manifest_ref=source_refs["snapshot_manifest"],
        calendar_ref=source_refs["calendar"],
        pit_manifest=source_payloads["pit_manifest"],
        pit_manifest_ref=source_refs["pit_manifest"],
        pit_canonical_ref=source_refs["pit_canonical"],
        market_ref=source_refs["market_data"],
        as_of=as_of,
        pit_symbols=pit_symbols,
    )
    market = _validate_market_data(
        source_payloads["market_data"],
        open_days=open_days,
        pit_symbols=pit_symbols,
        window_start=window_start,
        window_end=window_end,
    )
    code_files = _validate_code_config_manifest(
        source_payloads["code_config_manifest"], session
    )
    factor_set_sha = _sha256_value(factor_set)
    context = {
        "calendar_sha256": source_refs["calendar"]["sha256"],
        "code_config_manifest_sha256": source_refs["code_config_manifest"]["sha256"],
        "factor_set_sha256": factor_set_sha,
        "market_data_sha256": source_refs["market_data"]["sha256"],
        "pit_canonical_sha256": source_refs["pit_canonical"]["sha256"],
        "pit_manifest_sha256": source_refs["pit_manifest"]["sha256"],
        "registry_sha256": registry_sha,
        "snapshot_manifest_sha256": source_refs["snapshot_manifest"]["sha256"],
        "snapshot_pointer_sha256": source_refs["snapshot_pointer"]["sha256"],
    }
    stage_graph = _validate_stage_graph(
        stage_refs_value=payload["stages"],
        session=session,
        expected=context,
        run_id=run_id,
        as_of=as_of,
        window_start=window_start,
        window_end=window_end,
        market=market,
        pit_symbols=pit_symbols,
        open_days=open_days,
        registry_records=registry_records,
    )
    evidence_paths = set(primary_paths)
    evidence_inodes = set(primary_inodes)
    for item in code_files:
        if item["path"] in evidence_paths or session.identity(item["path"]) in evidence_inodes:
            raise CanonicalReplayError("code/config identities overlap primary evidence")
        evidence_paths.add(item["path"])
        evidence_inodes.add(session.identity(item["path"]))
    if evidence_paths.intersection(item["path"] for item in stage_graph["refs"]):
        raise CanonicalReplayError("stage paths overlap another evidence source")
    if evidence_inodes.intersection(stage_graph["stage_inodes"]):
        raise CanonicalReplayError("stage inodes overlap another evidence source")
    comparison = _validate_abcd(
        arms=stage_graph["arms"],
        factor_set=factor_set,
        comparison_value=payload["comparison"],
        registry_records=registry_records,
    )
    recomputed = {
        "comparison": comparison,
        "arms": {
            arm: {
                "factor_set": stage_graph["arms"][arm]["quant"]["selected_factors"],
                "after_cost_returns": stage_graph["arms"][arm]["portfolio_constructor"]["after_cost_returns"],
                "turnover": stage_graph["arms"][arm]["portfolio_constructor"]["turnover"],
                "slippage": stage_graph["arms"][arm]["portfolio_constructor"]["slippage"],
                "tail_risk": stage_graph["arms"][arm]["portfolio_constructor"]["tail_risk"],
            }
            for arm in ARM_NAMES
        },
    }
    if is_bundle:
        if canonical_json_bytes(payload["recomputed"]) != canonical_json_bytes(
            recomputed
        ):
            raise CanonicalReplayError("bundle recomputed evidence mismatch")
        normalized = dict(payload)
    else:
        normalized = dict(payload)
        normalized["schema_version"] = BUNDLE_SCHEMA_VERSION
        normalized["recomputed"] = recomputed
    normalized["stages"] = stage_graph["refs"]
    normalized["factor_set"] = factor_set
    for key, ref in source_refs.items():
        normalized[key] = ref
    return normalized


def _receipt_for_bundle(
    bundle: Mapping[str, Any], *, registry_sha256: str, bundle_sha256: str
) -> dict[str, Any]:
    factor_set = list(bundle["factor_set"])
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "evidence_id": bundle["evidence_id"],
        "registry_sha256": registry_sha256,
        "bundle_sha256": bundle_sha256,
        "factor_set": factor_set,
        "factor_set_sha256": _sha256_value(factor_set),
        "snapshot_pointer_sha256": bundle["snapshot_pointer"]["sha256"],
        "snapshot_manifest_sha256": bundle["snapshot_manifest"]["sha256"],
        "calendar_sha256": bundle["calendar"]["sha256"],
        "pit_manifest_sha256": bundle["pit_manifest"]["sha256"],
        "pit_canonical_sha256": bundle["pit_canonical"]["sha256"],
        "market_data_sha256": bundle["market_data"]["sha256"],
        "code_config_manifest_sha256": bundle["code_config_manifest"]["sha256"],
        "producer_contract_sha256": bundle["producer_contract_sha256"],
    }


def _validate_receipt(
    receipt: dict[str, Any], *, registry_sha256: str
) -> dict[str, Any]:
    _exact_fields(receipt, _RECEIPT_FIELDS, "canonical receipt")
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise CanonicalReplayError("unsupported canonical receipt schema")
    if receipt["registry_sha256"] != registry_sha256:
        raise CanonicalReplayError("canonical receipt registry binding mismatch")
    _sha256(receipt["bundle_sha256"], "receipt bundle SHA")
    factor_set = _ordered_distinct_strings(receipt["factor_set"], "receipt factor_set")
    if receipt["factor_set_sha256"] != _sha256_value(factor_set):
        raise CanonicalReplayError("receipt factor-set SHA mismatch")
    for key in (
        "registry_sha256",
        "snapshot_pointer_sha256",
        "snapshot_manifest_sha256",
        "calendar_sha256",
        "pit_manifest_sha256",
        "pit_canonical_sha256",
        "market_data_sha256",
        "code_config_manifest_sha256",
        "producer_contract_sha256",
    ):
        _sha256(receipt[key], f"receipt {key}")
    if receipt["producer_contract_sha256"] != producer_contract_sha256():
        raise CanonicalReplayError("receipt producer contract SHA mismatch")
    evidence_id = _text(receipt["evidence_id"], "receipt evidence_id")
    if "/" in evidence_id or evidence_id in {".", ".."}:
        raise CanonicalReplayError("receipt evidence_id is unsafe")
    return receipt


def _control_result(
    *,
    evidence_id: str,
    registry_sha256: str,
    bundle_sha256: str,
    receipt_sha256: str,
    factor_set_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "registry_sha256": registry_sha256,
        "bundle_sha256": bundle_sha256,
        "receipt_sha256": receipt_sha256,
        "factor_set_sha256": factor_set_sha256,
        "producer_contract_sha256": producer_contract_sha256(),
        "producer_implemented": True,
        "local_bytes_readback_verified": True,
        "canonical_producer_authenticated": False,
        "production_apply_authorized": False,
        "production_apply_eligible": False,
        "blocker": "forward_factor_apply_not_authorized_pr4",
    }


def produce_canonical_replay(
    *,
    private_root: str | os.PathLike[str],
    registry_path: str | os.PathLike[str],
    draft_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate an explicit draft graph, publish it immutably, and read it back."""

    session = SafeReadSession(private_root)
    exact_registry = session._validate_absolute(registry_path, "registry_path")
    exact_draft = session._validate_absolute(draft_path, "draft_path")
    registry_raw = session.read_bytes(exact_registry)
    draft_raw = session.read_bytes(exact_draft)
    draft = strict_json_loads(draft_raw, expected_fields=set(_DRAFT_FIELDS))
    bundle = _validate_replay_payload(
        payload=draft,
        schema_version=DRAFT_SCHEMA_VERSION,
        session=session,
        registry_path=exact_registry,
        registry_raw=registry_raw,
    )
    session.assert_unchanged()
    evidence_id = str(bundle["evidence_id"])
    published_bundle = publish_immutable_json(
        private_root,
        f"bundles/{evidence_id}.json",
        bundle,
    )
    registry_sha = _sha256_bytes(registry_raw)
    receipt = _receipt_for_bundle(
        bundle,
        registry_sha256=registry_sha,
        bundle_sha256=published_bundle["sha256"],
    )
    publish_immutable_json(
        private_root,
        f"receipts/{registry_sha}.json",
        receipt,
    )
    return verify_canonical_replay(
        private_root=private_root,
        registry_path=exact_registry,
    )


def verify_canonical_replay(
    *,
    private_root: str | os.PathLike[str],
    registry_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Follow only registry-SHA receipt selection and verify the full byte graph."""

    session = SafeReadSession(private_root)
    exact_registry = session._validate_absolute(registry_path, "registry_path")
    registry_raw = session.read_bytes(exact_registry)
    registry_sha = _sha256_bytes(registry_raw)
    exact_root = session.private_root
    receipt_path = f"{exact_root}/receipts/{registry_sha}.json"
    try:
        receipt_raw = session.read_bytes(receipt_path)
    except CanonicalReplayError as exc:
        raise CanonicalReplayError(
            f"exact canonical receipt is unavailable for registry {registry_sha}"
        ) from exc
    receipt = strict_json_loads(
        receipt_raw,
        expected_fields=set(_RECEIPT_FIELDS),
    )
    if receipt_raw != canonical_json_bytes(receipt) + b"\n":
        raise CanonicalReplayError(
            "exact canonical receipt bytes are not compact sorted JSON newline"
        )
    receipt = _validate_receipt(receipt, registry_sha256=registry_sha)
    evidence_id = receipt["evidence_id"]
    bundle_path = f"{exact_root}/bundles/{evidence_id}.json"
    try:
        bundle_raw = session.read_bytes(bundle_path)
    except CanonicalReplayError as exc:
        raise CanonicalReplayError("exact receipt-selected bundle is unavailable") from exc
    bundle_sha = _sha256_bytes(bundle_raw)
    if bundle_sha != receipt["bundle_sha256"]:
        raise CanonicalReplayError("receipt-selected bundle byte SHA mismatch")
    bundle = strict_json_loads(bundle_raw, expected_fields=set(_BUNDLE_FIELDS))
    if bundle_raw != canonical_json_bytes(bundle) + b"\n":
        raise CanonicalReplayError(
            "receipt-selected bundle bytes are not compact sorted JSON newline"
        )
    normalized = _validate_replay_payload(
        payload=bundle,
        schema_version=BUNDLE_SCHEMA_VERSION,
        session=session,
        registry_path=exact_registry,
        registry_raw=registry_raw,
    )
    expected_receipt = _receipt_for_bundle(
        normalized,
        registry_sha256=registry_sha,
        bundle_sha256=bundle_sha,
    )
    if canonical_json_bytes(receipt) != canonical_json_bytes(expected_receipt):
        raise CanonicalReplayError("receipt and bundle internal identities mismatch")
    session.assert_unchanged()
    return _control_result(
        evidence_id=evidence_id,
        registry_sha256=registry_sha,
        bundle_sha256=bundle_sha,
        receipt_sha256=_sha256_bytes(receipt_raw),
        factor_set_sha256=receipt["factor_set_sha256"],
    )


__all__ = [
    "ARM_NAMES",
    "BUNDLE_SCHEMA_VERSION",
    "CODE_CONFIG_ROLES",
    "CONTROL_CHAIN_STAGES",
    "CanonicalReplayError",
    "SafeReadSession",
    "canonical_json_bytes",
    "canonical_replay_producer_contract",
    "producer_contract_sha256",
    "produce_canonical_replay",
    "publish_immutable_json",
    "strict_json_loads",
    "verify_canonical_replay",
]
