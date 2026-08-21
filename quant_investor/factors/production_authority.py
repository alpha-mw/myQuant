"""Isolated first-production authority for the two approved bootstrap Factors.

This module owns exactly one narrow authority root:
``results/factors``.  It never reads, writes, derives, or bridges to
``results/system/_active.json``.  Its optional read-only source-custody
adapter resolves immutable source objects only; it cannot expose System
authority state.  A verified Factor source closure may be prepared into exact
immutable bytes and then committed through one expected-``EMPTY`` CAS.

The Factor pointer is intentionally separate from unified System authority:
activating it grants neither Mainline, Investment, portfolio, Strategy Record,
broker, order, trade, nor funds-transfer authority.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
from typing import Any, ClassVar, Final, NoReturn

from quant_investor.contracts import (
    MAX_CANONICAL_JSON_BYTES,
    ContractError,
    artifact_byte_sha256,
    canonical_json_bytes,
    get_contract,
    parse_canonical_json_bytes,
    seal_artifact,
    validate_artifact,
)

from .governance.bootstrap import (
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    _set_rows,
    bootstrap_factor_definitions,
)
from .governance.errors import FactorGovernanceError

FACTOR_PRODUCTION_SCOPE: Final = "FACTOR_PRODUCTION"
FACTOR_ROOT: Final = PurePosixPath("results/factors")
FACTOR_ACTIVE_POINTER_PATH: Final = FACTOR_ROOT / "_active.json"
FACTOR_PRODUCTION_MARKER_PATH: Final = FACTOR_ROOT / "_production_complete.json"
FACTOR_EMPTY_POINTER_SHA256: Final = "EMPTY"
FACTOR_OBJECTS_ROOT: Final = FACTOR_ROOT / "objects"
FACTOR_GENERATIONS_ROOT: Final = FACTOR_ROOT / "generations"
FACTOR_PREPARATIONS_ROOT: Final = FACTOR_ROOT / "preparations"
FACTOR_ACTIVATION_BUNDLES_ROOT: Final = FACTOR_ROOT / "activation_bundles"
FACTOR_ACTIVATION_TRANSACTIONS_ROOT: Final = FACTOR_ROOT / "activation_transactions"
FACTOR_POINTER_HISTORY_ROOT: Final = FACTOR_ROOT / "pointer_history"
FACTOR_SOURCE_MIRRORS_ROOT: Final = FACTOR_ROOT / "source_mirrors"
FACTOR_ACTIVE_LOCK_PATH: Final = FACTOR_ROOT / ".active.lock"

FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND: Final = "factor.production_source_closure"
FACTOR_PRODUCTION_RECOMPUTATION_KIND: Final = "factor.production_recomputation_evidence"
FACTOR_LEGACY_ZERO_CALL_KIND: Final = "factor.production_legacy_zero_call_certificate"
FACTOR_MARKET_PIT_SELECTION_KIND: Final = "factor.production_market_pit_selection"
FACTOR_PRODUCTION_MARKET_INPUT_KIND: Final = "factor.production_market_input"
FACTOR_CALENDAR_CUSTODY_ATTESTATION_KIND: Final = (
    "factor.production_calendar_capture_custody_attestation"
)
FACTOR_PRODUCTION_GENERATION_KIND: Final = "factor.production_generation"
FACTOR_PRODUCTION_POINTER_KIND: Final = "factor.production_pointer"
FACTOR_PRODUCTION_RECEIPT_KIND: Final = "factor.production_generation_receipt"
FACTOR_PRODUCTION_BUNDLE_KIND: Final = "factor.production_activation_bundle"
FACTOR_PRODUCTION_PREPARED_KIND: Final = "factor.production_prepared"
FACTOR_PRODUCTION_MARKER_KIND: Final = "factor.production_marker"
FACTOR_PRODUCTION_ROLLOVER_BUNDLE_KIND: Final = "factor.production_rollover_bundle"
FACTOR_PRODUCTION_ROLLOVER_PREPARED_KIND: Final = "factor.production_rollover_prepared"
FACTOR_PRODUCTION_ROLLOVER_COMMIT_KIND: Final = "factor.production_rollover_commit"
FACTOR_ROLLOVER_BUNDLES_ROOT: Final = FACTOR_ROOT / "rollover_bundles"
FACTOR_ROLLOVER_TRANSACTIONS_ROOT: Final = FACTOR_ROOT / "rollover_transactions"
FACTOR_ROLLOVER_COMMITS_ROOT: Final = FACTOR_ROOT / "rollover_commits"
FACTOR_ROLLOVER_INPUT_INDEX_ROOT: Final = FACTOR_ROOT / "rollover_input_index"
FACTOR_PRODUCTION_OBSERVATIONS_ROOT: Final = FACTOR_ROOT / "observations"
FACTOR_POINTER_CHAIN_MAX: Final = 4096

FACTOR_READINESS_READY: Final = "READY"
FACTOR_AUTHORITY_ACTIVE: Final = "ACTIVE"
ADMISSION_ROUTE: Final = "BOOTSTRAP_EXCEPTION"
PRODUCER_IDENTITY: Final = "NOT_CLAIMED"
FUNDAMENTAL_NOT_USED: Final = "NOT_USED_BY_ACTIVE_FACTOR_SET"
FUNDAMENTAL_ADVISORY: Final = "ADVISORY"

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_TIMESTAMP_RE: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_ACTOR_RE: Final = re.compile(r"^uid:([0-9]+)$")
_OBJECT_REF_FIELDS: Final = frozenset(
    {"kind", "contract_sha256", "artifact_id", "semantic_sha256", "byte_sha256"}
)
_NO_AUTHORITY_FIELDS: Final = (
    "system_authority",
    "mainline_authority",
    "investment_authority",
    "portfolio_authority",
    "strategy_record_authority",
    "broker_authority",
)
_RESERVED_FACTOR_AUTHORITY_PATHS: Final = frozenset(
    {FACTOR_ACTIVE_POINTER_PATH, FACTOR_PRODUCTION_MARKER_PATH}
)
_LOCAL_FACTOR_CLOSURE_KINDS: Final = frozenset(
    {
        FACTOR_PRODUCTION_GENERATION_KIND,
        FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND,
        FACTOR_PRODUCTION_RECOMPUTATION_KIND,
        FACTOR_LEGACY_ZERO_CALL_KIND,
        FACTOR_PRODUCTION_MARKET_INPUT_KIND,
    }
)
_MAX_FACTOR_SOURCE_MIRROR_BYTES: Final = 512 * 1024 * 1024
_DARWIN_RENAME_EXCL: Final = 0x00000004
_LINUX_RENAME_NOREPLACE: Final = 0x00000001

_SOURCE_FIELDS: Final = frozenset(
    {
        "factor_production_source_closure_id",
        "state",
        "activation_scope",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_verification",
        "release_install_input_source_ref",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "factor_implementation_refs",
        "legacy_zero_call_ref",
        "market_input_ref",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    }
)
_RECOMPUTATION_FIELDS: Final = frozenset(
    {
        "factor_production_recomputation_id",
        "state",
        "activation_scope",
        "source_closure_ref",
        "deployed_release_ref",
        "factor_active_set_ref",
        "as_of",
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_values",
        "signal_statistics",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
    }
)
_LEGACY_ZERO_CALL_FIELDS: Final = frozenset(
    {
        "factor_legacy_zero_call_id",
        "state",
        "activation_scope",
        "final_commit",
        "final_tree",
        "resolver_inventory_ref",
        "active_legacy_import_count",
        "active_legacy_call_count",
        "active_legacy_path_hash_count",
        "legacy_entrypoint_count",
        "verification_module_path",
        "verification_module_sha256",
        "verification_command",
        "stdout_sha256",
        "stderr_sha256",
        "verified_at",
    }
)
_MARKET_INPUT_FIELDS: Final = frozenset(
    {
        "factor_market_input_id",
        "state",
        "activation_scope",
        "as_of",
        "market_pit_selection_ref",
        "market_pointer_source_ref",
        "market_snapshot_manifest_source_ref",
        "market_scope_source_ref",
        "market_history_source_ref",
        "market_pointer_sha256",
        "market_snapshot_manifest_sha256",
        "market_history_sha256",
        "market_snapshot_id",
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "pit_generation_id",
        "pit_membership_sha256",
        "producer_module_path",
        "producer_module_sha256",
    }
)
_GENERATION_FIELDS: Final = frozenset(
    {
        "factor_production_generation_id",
        "state",
        "activation_scope",
        "admission_route",
        "producer_identity",
        "as_of",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_verification",
        "release_install_input_source_ref",
        "source_closure_ref",
        "recomputation_evidence_ref",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "market_input_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "factor_implementation_refs",
        "legacy_zero_call_ref",
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_values",
        "signal_statistics",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    }
)
_POINTER_FIELDS: Final = frozenset(
    {
        "factor_production_pointer_id",
        "factor_generation_id",
        "factor_generation_sha256",
        "previous_pointer_sha256",
        "activated_at",
        "os_actor",
        "authority_scope",
        "pointer_raw_sha256",
    }
)
_ACTIVE_POINTER_FIELDS: Final = frozenset(
    {
        "factor_generation_id",
        "factor_generation_sha256",
        "previous_pointer_sha256",
        "activated_at",
        "os_actor",
        "authority_scope",
    }
)
_RECEIPT_FIELDS: Final = frozenset(
    {
        "factor_production_receipt_id",
        "state",
        "activation_scope",
        "source_closure_ref",
        "recomputation_evidence_ref",
        "factor_generation_ref",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_input_source_ref",
        "legacy_zero_call_ref",
        "market_input_ref",
        "low_signal_sha256",
        "w80_signal_sha256",
        "active_factor_rows",
        "control_rows",
        "factor_readiness",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    }
)
_BUNDLE_FIELDS: Final = frozenset(
    {
        "factor_production_activation_id",
        "state",
        "activation_scope",
        "factor_generation_receipt_ref",
        "target_factor_generation_id",
        "target_factor_generation_ref",
        "deployed_release_ref",
        "active_factor_rows",
        "control_rows",
        "low_signal_sha256",
        "w80_signal_sha256",
        "factor_readiness",
        "market_input_ref",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "target_factor_pointer_ref",
        "target_factor_pointer_path",
        "expected_factor_pointer_sha256",
        "prepared_at",
        "activated_at",
        "actor_uid",
        "os_actor",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    }
)
_PREPARED_FIELDS: Final = frozenset(
    {
        "factor_production_prepared_id",
        "state",
        "activation_scope",
        "activation_bundle_ref",
        "factor_generation_receipt_ref",
        "target_factor_pointer_ref",
        "expected_factor_pointer_sha256",
        "prepared_at",
        "actor_uid",
    }
)
_MARKER_FIELDS: Final = frozenset(
    {
        "factor_production_marker_id",
        "state",
        "activation_scope",
        "activation_bundle_ref",
        "prepared_transaction_ref",
        "factor_generation_receipt_ref",
        "factor_pointer_ref",
        "factor_generation_ref",
        "deployed_release_ref",
        "active_factor_rows",
        "control_rows",
        "factor_readiness",
        "factor_authority",
        "market_input_ref",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    }
)
_ROLLOVER_BUNDLE_FIELDS: Final = frozenset(
    {
        "factor_production_rollover_bundle_id",
        "state",
        "activation_scope",
        "predecessor_pointer_ref",
        "target_pointer_ref",
        "previous_pointer_sha256",
        "target_pointer_sha256",
        "factor_generation_receipt_ref",
        "target_factor_generation_ref",
        "maintenance_receipt_path",
        "maintenance_receipt_sha256",
        "market_pointer_sha256",
        "market_manifest_sha256",
        "pit_pointer_sha256",
        "pit_manifest_sha256",
        "target_date",
        "prepared_at",
        "actor_uid",
        *_NO_AUTHORITY_FIELDS,
    }
)
_ROLLOVER_PREPARED_FIELDS: Final = frozenset(
    {
        "factor_production_rollover_prepared_id",
        "state",
        "activation_scope",
        "rollover_bundle_ref",
        "expected_pointer_sha256",
        "target_pointer_sha256",
        "prepared_at",
        "actor_uid",
    }
)
_ROLLOVER_COMMIT_FIELDS: Final = frozenset(
    {
        "factor_production_rollover_commit_id",
        "state",
        "activation_scope",
        "rollover_bundle_ref",
        "rollover_prepared_ref",
        "previous_pointer_sha256",
        "target_pointer_sha256",
        "committed_at",
        "actor_uid",
        "cas_performed",
        *_NO_AUTHORITY_FIELDS,
    }
)


@dataclass(frozen=True, slots=True)
class FactorStoredBytes:
    """An owner-only exact-byte Factor authority artifact."""

    relative_path: str
    data: bytes
    byte_sha256: str


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rollover_input_key(*, previous_pointer_sha256: str, maintenance_sha256: str) -> str:
    return _sha256(
        canonical_json_bytes(
            {
                "domain": "factor-production-rollover-input",
                "previous_pointer_sha256": _require_sha(
                    previous_pointer_sha256, label="rollover input previous pointer"
                ),
                "maintenance_sha256": _require_sha(
                    maintenance_sha256, label="rollover input maintenance"
                ),
            }
        )
    )


def _atomic_no_replace_rename(
    source: str,
    destination: str,
    *,
    source_directory_fd: int,
    destination_directory_fd: int,
) -> None:
    """Atomically rename without replacement or a weaker fallback."""

    try:
        source_raw = source.encode("ascii", errors="strict")
        destination_raw = destination.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise FactorGovernanceError("Factor authority rename leaf must be ASCII") from exc
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        flags = _DARWIN_RENAME_EXCL
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        flags = _LINUX_RENAME_NOREPLACE
    else:
        operation = None
        flags = 0
    if operation is None:
        _raise("atomic no-replace rename is unavailable on this platform")
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = operation(
        source_directory_fd,
        source_raw,
        destination_directory_fd,
        destination_raw,
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise FactorGovernanceError(
        "atomic no-replace rename failed " f"(platform={sys.platform}, errno={error_number})"
    )


def _raise(detail: str) -> NoReturn:
    raise FactorGovernanceError(detail)


def _require_sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _raise(f"{label} must be lowercase SHA-256")
    return value


def _require_text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip() or not value.isascii():
        _raise(f"{label} must be canonical ASCII text")
    return value


def _timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
        _raise(f"{label} must be canonical UTC")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:  # pragma: no cover - regex narrows normal cases
        raise FactorGovernanceError(f"{label} must be canonical UTC") from exc


def _canonical_path(value: str | PurePosixPath) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        _raise("Factor authority path must be relative text")
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path != FACTOR_ROOT and FACTOR_ROOT not in path.parents)
    ):
        _raise("Factor authority path is outside its governed root")
    try:
        text.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise FactorGovernanceError("Factor authority path must be ASCII") from exc
    return path


def _stable_workspace_file_sha256(
    workspace_root: Path, value: str | os.PathLike[str], *, label: str
) -> str:
    """Hash one owner-controlled regular workspace file with stable double read."""

    try:
        root = workspace_root.resolve(strict=True)
        path = Path(value).resolve(strict=True)
        path.relative_to(root)
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_CANONICAL_JSON_BYTES
        ):
            _raise(f"{label} is not a safe bounded regular file")
        first = path.read_bytes()
        middle = path.lstat()
        second = path.read_bytes()
        after = path.lstat()
    except (OSError, ValueError) as exc:
        raise FactorGovernanceError(f"{label} is unavailable") from exc
    if (
        _stat_identity(before) != _stat_identity(middle)
        or _stat_identity(middle) != _stat_identity(after)
        or first != second
        or len(first) != after.st_size
    ):
        _raise(f"{label} changed during stable read")
    return _sha256(first)


def _is_reserved_factor_authority_path(path: PurePosixPath) -> bool:
    return path in _RESERVED_FACTOR_AUTHORITY_PATHS or any(
        reserved in path.parents for reserved in _RESERVED_FACTOR_AUTHORITY_PATHS
    )


def _artifact_ref(document: Mapping[str, Any] | bytes) -> dict[str, str]:
    try:
        artifact = validate_artifact(document)
    except ContractError as exc:
        raise FactorGovernanceError("Factor authority artifact contract failed") from exc
    return {
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "artifact_id": artifact["artifact_id"],
        "semantic_sha256": artifact["semantic_sha256"],
        "byte_sha256": artifact_byte_sha256(artifact),
    }


def _validate_ref(
    value: Any,
    *,
    label: str,
    expected_kind: str | None = None,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_OBJECT_REF_FIELDS):
        _raise(f"{label} fields are not exact")
    kind = _require_text(value.get("kind"), label=f"{label}.kind")
    contract_sha = _require_sha(value.get("contract_sha256"), label=f"{label}.contract_sha256")
    try:
        definition = get_contract(kind, contract_sha)
    except ContractError as exc:
        raise FactorGovernanceError(f"{label} contract pair is not compiled") from exc
    if expected_kind is not None and definition.kind != expected_kind:
        _raise(f"{label} kind differs")
    return {
        "kind": definition.kind,
        "contract_sha256": definition.contract_sha256,
        "artifact_id": _require_text(value.get("artifact_id"), label=f"{label}.artifact_id"),
        "semantic_sha256": _require_sha(
            value.get("semantic_sha256"), label=f"{label}.semantic_sha256"
        ),
        "byte_sha256": _require_sha(value.get("byte_sha256"), label=f"{label}.byte_sha256"),
    }


def _sorted_refs(value: Any, *, label: str) -> list[dict[str, str]]:
    if type(value) is not list or not value:
        _raise(f"{label} must be a nonempty list")
    rows = [_validate_ref(row, label=f"{label}[{index}]") for index, row in enumerate(value)]
    keys = [
        (
            row["kind"],
            row["contract_sha256"],
            row["artifact_id"],
            row["semantic_sha256"],
            row["byte_sha256"],
        )
        for row in rows
    ]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _raise(f"{label} must be sorted and unique")
    return rows


def _validate_factor_generation_ref(value: Any, *, label: str) -> dict[str, str]:
    ref = _validate_ref(value, label=label, expected_kind=FACTOR_PRODUCTION_GENERATION_KIND)
    generation_id = ref["artifact_id"]
    prefix = "factor-production-generation-"
    if (
        not generation_id.startswith(prefix)
        or _SHA256_RE.fullmatch(generation_id.removeprefix(prefix)) is None
    ):
        _raise(f"{label} generation identity differs")
    return ref


def _validate_release_install_verification(value: Any) -> dict[str, Any]:
    fields = {
        "state",
        "release_ref",
        "source_archive_sha256",
        "wheel_sha256",
        "code_tree_sha256",
        "installed_code_manifest_sha256",
        "contract_catalog_sha256",
        "import_origin",
    }
    if type(value) is not dict or set(value) != fields or value.get("state") != "PASS":
        _raise("Factor release-install verification fields differ")
    result = dict(value)
    _validate_ref(result["release_ref"], label="release-install verification release_ref")
    for field in (
        "source_archive_sha256",
        "wheel_sha256",
        "code_tree_sha256",
        "installed_code_manifest_sha256",
        "contract_catalog_sha256",
    ):
        _require_sha(result[field], label=f"release-install verification {field}")
    _require_text(result["import_origin"], label="release-install verification import_origin")
    return result


def _require_no_authority(payload: Mapping[str, Any]) -> None:
    for field in _NO_AUTHORITY_FIELDS:
        if payload.get(field) != "NONE":
            _raise(f"Factor production {field} must remain NONE")


def _identity(prefix: str, payload: Mapping[str, Any], identity_field: str) -> str:
    body = dict(payload)
    observed = body.pop(identity_field, None)
    expected = prefix + _sha256(canonical_json_bytes(body))
    if observed != expected:
        _raise(f"{identity_field} differs from canonical content identity")
    return expected


def _validate_artifact_kind(document: Mapping[str, Any] | bytes, kind: str) -> dict[str, Any]:
    try:
        return validate_artifact(
            document,
            expected_kind=kind,
            expected_contract_sha256=get_contract(kind).contract_sha256,
        )
    except ContractError as exc:
        raise FactorGovernanceError(f"{kind} contract failed") from exc


def _expected_factor_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active, control = _set_rows(bootstrap_factor_definitions())
    return active, control


def _validate_factor_policy_rows(
    active: Any, control: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_active, expected_control = _expected_factor_rows()
    if active != expected_active or control != expected_control:
        _raise("Factor production policy must be exactly LOW/W80 50/50 and W75 control-only")
    return [dict(row) for row in expected_active], [dict(row) for row in expected_control]


def _validate_source_closure(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND)
    payload = artifact["payload"]
    if set(payload) != _SOURCE_FIELDS:
        _raise("Factor production source closure fields differ")
    if (
        payload["state"] != "VERIFIED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["admission_route"] != ADMISSION_ROUTE
        or payload["producer_identity"] != PRODUCER_IDENTITY
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
    ):
        _raise("Factor production source closure policy differs")
    _require_no_authority(payload)
    _validate_ref(
        payload["deployed_release_ref"],
        label="source closure deployed_release_ref",
        expected_kind="system.release",
    )
    _validate_ref(
        payload["release_install_evidence_ref"],
        label="source closure release_install_evidence_ref",
        expected_kind="system.release_install_evidence",
    )
    _validate_release_install_verification(payload["release_install_verification"])
    _validate_ref(
        payload["release_install_input_source_ref"],
        label="source closure release_install_input_source_ref",
        expected_kind="system.source_object",
    )
    _validate_ref(
        payload["market_scope_source_ref"],
        label="source closure market_scope_source_ref",
        expected_kind="system.source_object",
    )
    _validate_ref(
        payload["calendar_capture_custody_attestation_ref"],
        label="source closure calendar_capture_custody_attestation_ref",
        expected_kind=FACTOR_CALENDAR_CUSTODY_ATTESTATION_KIND,
    )
    for field in (
        "market_pit_selection_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "factor_source_bundle_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
    ):
        _validate_ref(payload[field], label=f"source closure {field}")
    _validate_ref(
        payload["legacy_zero_call_ref"],
        label="source closure legacy_zero_call_ref",
        expected_kind=FACTOR_LEGACY_ZERO_CALL_KIND,
    )
    _validate_ref(
        payload["market_input_ref"],
        label="source closure market_input_ref",
        expected_kind=FACTOR_PRODUCTION_MARKET_INPUT_KIND,
    )
    if (
        len(
            _sorted_refs(
                payload["factor_implementation_refs"], label="source closure implementations"
            )
        )
        != 2
    ):
        _raise("Factor production source closure must bind exactly two active implementations")
    _identity("factor-production-source-", payload, "factor_production_source_closure_id")
    return artifact


def _validate_signal_statistics(
    value: Any,
    *,
    low_signal_sha256: str,
    w80_signal_sha256: str,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != 2:
        _raise("Factor production signal statistics must contain LOW and W80")
    expected_ids = [LOW_DOLLAR_VOLUME, BLEND_W80]
    expected_hashes = [low_signal_sha256, w80_signal_sha256]
    fields = {
        "factor_id",
        "signal_symbol_set_sha256",
        "source_symbol_count",
        "finite_count",
        "distinct_finite_count",
        "coverage_numerator",
        "coverage_denominator",
        "coverage_rate",
        "signal_sha256",
        "implementation_sha256",
    }
    normalized: list[dict[str, Any]] = []
    for index, (row, factor_id, signal_sha) in enumerate(zip(value, expected_ids, expected_hashes)):
        if type(row) is not dict or set(row) != fields or row.get("factor_id") != factor_id:
            _raise("Factor production signal statistics identity differs")
        for field in ("signal_symbol_set_sha256", "signal_sha256", "implementation_sha256"):
            _require_sha(row.get(field), label=f"signal statistics[{index}].{field}")
        if row["signal_sha256"] != signal_sha:
            _raise("Factor production signal statistics hash differs")
        for field in (
            "source_symbol_count",
            "finite_count",
            "distinct_finite_count",
            "coverage_numerator",
            "coverage_denominator",
        ):
            if type(row.get(field)) is not int or row[field] <= 0:
                _raise(f"signal statistics[{index}].{field} is invalid")
        if (
            row["distinct_finite_count"] <= 1
            or row["source_symbol_count"] != row["finite_count"]
            or row["coverage_numerator"] != row["finite_count"]
            or row["coverage_denominator"] != row["finite_count"]
            or row.get("coverage_rate") != "1.000000000000"
        ):
            _raise("Factor production signal statistics are empty, constant, or incomplete")
        normalized.append(dict(row))
    return normalized


def _validate_signal_values(value: Any) -> dict[str, dict[str, str]]:
    if type(value) is not dict or set(value) != {LOW_DOLLAR_VOLUME, BLEND_W80}:
        _raise("Factor production signal values must contain exactly LOW and W80")
    normalized: dict[str, dict[str, str]] = {}
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        rows = value[factor_id]
        if type(rows) is not dict or not rows:
            _raise(f"Factor production {factor_id} signal values are empty")
        normalized_rows: dict[str, str] = {}
        for symbol, encoded in rows.items():
            if (
                type(symbol) is not str
                or re.fullmatch(r"[0-9]{6}\.(?:SH|SZ|BJ)", symbol) is None
                or type(encoded) is not str
            ):
                _raise(f"Factor production {factor_id} signal value fields differ")
            try:
                decoded = float.fromhex(encoded)
            except ValueError as exc:
                raise FactorGovernanceError(
                    f"Factor production {factor_id} signal value is not float.hex"
                ) from exc
            if not math.isfinite(decoded) or decoded.hex() != encoded:
                _raise(f"Factor production {factor_id} signal value is not canonical finite hex")
            normalized_rows[symbol] = encoded
        normalized[factor_id] = normalized_rows
    if set(normalized[LOW_DOLLAR_VOLUME]) != set(normalized[BLEND_W80]):
        _raise("Factor production LOW/W80 signal cohorts differ")
    return normalized


def _validate_recomputation_evidence(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_RECOMPUTATION_KIND)
    payload = artifact["payload"]
    if set(payload) != _RECOMPUTATION_FIELDS:
        _raise("Factor production recomputation evidence fields differ")
    if (
        payload["state"] != "VERIFIED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["admission_route"] != ADMISSION_ROUTE
        or payload["producer_identity"] != PRODUCER_IDENTITY
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
    ):
        _raise("Factor production recomputation policy differs")
    _validate_ref(
        payload["source_closure_ref"],
        label="recomputation source_closure_ref",
        expected_kind=FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND,
    )
    _validate_ref(
        payload["deployed_release_ref"],
        label="recomputation deployed_release_ref",
        expected_kind="system.release",
    )
    _validate_ref(payload["factor_active_set_ref"], label="recomputation factor_active_set_ref")
    low_sha = _require_sha(payload["low_signal_sha256"], label="recomputation LOW signal")
    w80_sha = _require_sha(payload["w80_signal_sha256"], label="recomputation W80 signal")
    if type(payload["as_of"]) is not str or re.fullmatch(r"[0-9]{8}", payload["as_of"]) is None:
        _raise("Factor production recomputation as_of differs")
    _validate_signal_values(payload["signal_values"])
    _require_sha(payload["exact_replay_sha256"], label="recomputation exact replay")
    _validate_signal_statistics(
        payload["signal_statistics"], low_signal_sha256=low_sha, w80_signal_sha256=w80_sha
    )
    _validate_factor_policy_rows(payload["active_factor_rows"], payload["control_rows"])
    _identity("factor-production-recompute-", payload, "factor_production_recomputation_id")
    return artifact


def _validate_legacy_zero_call_certificate(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_LEGACY_ZERO_CALL_KIND)
    payload = artifact["payload"]
    if set(payload) != _LEGACY_ZERO_CALL_FIELDS:
        _raise("Factor legacy-zero-call certificate fields differ")
    if payload["state"] != "VERIFIED" or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE:
        _raise("Factor legacy-zero-call certificate is not verified for production")
    for field in ("final_commit", "final_tree"):
        if type(payload.get(field)) is not str or _GIT_OBJECT_RE.fullmatch(payload[field]) is None:
            _raise(f"Factor legacy-zero-call {field} is invalid")
    _validate_ref(payload["resolver_inventory_ref"], label="legacy-zero-call resolver inventory")
    for field in (
        "active_legacy_import_count",
        "active_legacy_call_count",
        "active_legacy_path_hash_count",
        "legacy_entrypoint_count",
    ):
        if type(payload.get(field)) is not int or payload[field] != 0:
            _raise("Factor legacy-zero-call certificate is nonzero")
    path = _require_text(
        payload.get("verification_module_path"), label="legacy-zero-call module path"
    )
    if not path.startswith("quant_investor/factors/") or "v17" in path.lower():
        _raise("Factor legacy-zero-call verifier is outside the Factor authority lane")
    _require_sha(payload.get("verification_module_sha256"), label="legacy-zero-call module SHA")
    _require_text(payload.get("verification_command"), label="legacy-zero-call command")
    _require_sha(payload.get("stdout_sha256"), label="legacy-zero-call stdout SHA")
    _require_sha(payload.get("stderr_sha256"), label="legacy-zero-call stderr SHA")
    _timestamp(payload.get("verified_at"), label="legacy-zero-call verified_at")
    _identity("factor-production-legacy-zero-call-", payload, "factor_legacy_zero_call_id")
    return artifact


def _validate_market_input(  # noqa: C901 - exact fail-closed contract validator
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate the typed bridge before the source lane performs deep replay."""

    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_MARKET_INPUT_KIND)
    payload = artifact["payload"]
    if set(payload) != _MARKET_INPUT_FIELDS:
        _raise("Factor production Market input fields differ")
    if payload["state"] != "VERIFIED" or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE:
        _raise("Factor production Market input state differs")
    if type(payload["as_of"]) is not str or re.fullmatch(r"[0-9]{8}", payload["as_of"]) is None:
        _raise("Factor production Market input as_of differs")
    try:
        if datetime.strptime(payload["as_of"], "%Y%m%d").strftime("%Y%m%d") != payload["as_of"]:
            _raise("Factor production Market input as_of differs")
    except ValueError as exc:
        raise FactorGovernanceError("Factor production Market input as_of differs") from exc
    _validate_ref(
        payload["market_pit_selection_ref"],
        label="Market input selection ref",
        expected_kind=FACTOR_MARKET_PIT_SELECTION_KIND,
    )
    for field in (
        "market_pointer_source_ref",
        "market_snapshot_manifest_source_ref",
        "market_scope_source_ref",
        "market_history_source_ref",
    ):
        _validate_ref(
            payload[field], label=f"Market input {field}", expected_kind="system.source_object"
        )
    for field in (
        "market_pointer_sha256",
        "market_snapshot_manifest_sha256",
        "market_history_sha256",
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "pit_membership_sha256",
        "producer_module_sha256",
    ):
        _require_sha(payload[field], label=f"Market input {field}")
    _require_text(payload["market_snapshot_id"], label="Market input snapshot id")
    _require_text(payload["pit_generation_id"], label="Market input PIT generation id")
    producer_path = _require_text(
        payload["producer_module_path"], label="Market input producer path"
    )
    if not producer_path.startswith("quant_investor/factors/"):
        _raise("Factor production Market input producer is outside Factor source authority")
    try:
        from .governance.production_authority import validate_factor_production_market_input

        replayed = validate_factor_production_market_input(artifact)
    except (ImportError, ContractError, ValueError) as exc:
        raise FactorGovernanceError(
            "Factor production Market input semantic validator is unavailable"
        ) from exc
    if replayed != artifact:
        _raise("Factor production Market input semantic replay differs")
    return replayed


def _validate_factor_generation(  # noqa: C901 - exact fail-closed contract validator
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate the unpublished Factor-only generation without System linkage."""

    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_GENERATION_KIND)
    payload = artifact["payload"]
    if set(payload) != _GENERATION_FIELDS:
        _raise("Factor production generation fields differ")
    if (
        payload["state"] != "OPERATIONAL"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["admission_route"] != ADMISSION_ROUTE
        or payload["producer_identity"] != PRODUCER_IDENTITY
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
    ):
        _raise("Factor production generation policy differs")
    _require_no_authority(payload)
    _validate_release_install_verification(payload["release_install_verification"])
    if type(payload["as_of"]) is not str or re.fullmatch(r"[0-9]{8}", payload["as_of"]) is None:
        _raise("Factor production generation as_of differs")
    try:
        if datetime.strptime(payload["as_of"], "%Y%m%d").strftime("%Y%m%d") != payload["as_of"]:
            _raise("Factor production generation as_of differs")
    except ValueError as exc:
        raise FactorGovernanceError("Factor production generation as_of differs") from exc
    for field, kind in (
        ("deployed_release_ref", "system.release"),
        ("release_install_evidence_ref", "system.release_install_evidence"),
        ("release_install_input_source_ref", "system.source_object"),
        ("source_closure_ref", FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND),
        ("recomputation_evidence_ref", FACTOR_PRODUCTION_RECOMPUTATION_KIND),
        ("market_pit_selection_ref", FACTOR_MARKET_PIT_SELECTION_KIND),
        ("market_scope_source_ref", "system.source_object"),
        (
            "calendar_capture_custody_attestation_ref",
            FACTOR_CALENDAR_CUSTODY_ATTESTATION_KIND,
        ),
        ("market_input_ref", FACTOR_PRODUCTION_MARKET_INPUT_KIND),
        ("legacy_zero_call_ref", FACTOR_LEGACY_ZERO_CALL_KIND),
    ):
        _validate_ref(payload[field], label=f"generation {field}", expected_kind=kind)
    for field in (
        "calendar_compilation_ref",
        "factor_source_bundle_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
    ):
        _validate_ref(payload[field], label=f"generation {field}")
    if (
        len(_sorted_refs(payload["factor_implementation_refs"], label="generation implementations"))
        != 2
    ):
        _raise("Factor production generation must bind exactly two active implementations")
    low_sha = _require_sha(payload["low_signal_sha256"], label="generation LOW signal")
    w80_sha = _require_sha(payload["w80_signal_sha256"], label="generation W80 signal")
    _require_sha(payload["exact_replay_sha256"], label="generation exact replay")
    _validate_signal_values(payload["signal_values"])
    _validate_signal_statistics(
        payload["signal_statistics"], low_signal_sha256=low_sha, w80_signal_sha256=w80_sha
    )
    _validate_factor_policy_rows(payload["active_factor_rows"], payload["control_rows"])
    generation_id = _identity(
        "factor-production-generation-", payload, "factor_production_generation_id"
    )
    if artifact["artifact_id"] != generation_id:
        _raise("Factor production generation artifact identity differs")
    try:
        from .governance.production_authority import (
            validate_factor_production_generation,
        )

        replayed = validate_factor_production_generation(artifact)
    except (ImportError, ContractError, ValueError) as exc:
        raise FactorGovernanceError(
            "Factor production generation semantic validator is unavailable"
        ) from exc
    if replayed != artifact:
        _raise("Factor production generation semantic replay differs")
    return replayed


def _cross_bind_source_inputs(
    source: Mapping[str, Any], recomputation: Mapping[str, Any], legacy: Mapping[str, Any]
) -> None:
    source_payload = source["payload"]
    recomputation_payload = recomputation["payload"]
    if recomputation_payload["source_closure_ref"] != _artifact_ref(source):
        _raise("recomputation source closure binding differs")
    for field in ("deployed_release_ref", "factor_active_set_ref"):
        if recomputation_payload[field] != source_payload[field]:
            _raise(f"recomputation {field} differs from source closure")
    for field in (
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
    ):
        if recomputation_payload[field] != source_payload[field]:
            _raise(f"recomputation {field} differs from source closure")
    if source_payload["legacy_zero_call_ref"] != _artifact_ref(legacy):
        _raise("legacy-zero-call certificate binding differs")


def _cross_bind_factor_generation(
    generation: Mapping[str, Any],
    source: Mapping[str, Any],
    recomputation: Mapping[str, Any],
    legacy: Mapping[str, Any],
    market: Mapping[str, Any],
) -> None:
    """Bind one immutable Factor generation to the complete direct closure."""

    _cross_bind_source_inputs(source, recomputation, legacy)
    source_payload = source["payload"]
    recomputation_payload = recomputation["payload"]
    generation_payload = generation["payload"]
    market_payload = market["payload"]
    if source_payload["market_input_ref"] != _artifact_ref(market):
        _raise("Factor production Market input binding differs")
    for field, expected in (
        ("source_closure_ref", _artifact_ref(source)),
        ("recomputation_evidence_ref", _artifact_ref(recomputation)),
        ("legacy_zero_call_ref", _artifact_ref(legacy)),
        ("market_input_ref", _artifact_ref(market)),
        ("as_of", market_payload["as_of"]),
    ):
        if generation_payload[field] != expected:
            _raise(f"Factor production generation {field} differs")
    for field in (
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_verification",
        "release_install_input_source_ref",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "factor_implementation_refs",
    ):
        if generation_payload[field] != source_payload[field]:
            _raise(f"Factor production generation {field} differs from source closure")
    for field in (
        "as_of",
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_values",
        "signal_statistics",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
    ):
        if generation_payload[field] != recomputation_payload[field]:
            _raise(f"Factor production generation {field} differs from recomputation")


def build_factor_production_generation_receipt(
    *,
    factor_generation: Mapping[str, Any] | bytes,
    source_closure: Mapping[str, Any] | bytes,
    recomputation_evidence: Mapping[str, Any] | bytes,
    legacy_zero_call_certificate: Mapping[str, Any] | bytes,
    market_input: Mapping[str, Any] | bytes,
    created_at: str,
) -> dict[str, Any]:
    """Seal the Factor-only slice of a verified generation.

    The caller must provide all three typed artifacts.  The receipt does not
    accept a generic System final-cutover or dirty-inventory artifact, and it
    does not grant any System-adjacent authority.
    """

    generation = _validate_factor_generation(factor_generation)
    source = _validate_source_closure(source_closure)
    recomputation = _validate_recomputation_evidence(recomputation_evidence)
    legacy = _validate_legacy_zero_call_certificate(legacy_zero_call_certificate)
    market = _validate_market_input(market_input)
    _cross_bind_factor_generation(generation, source, recomputation, legacy, market)
    source_payload = source["payload"]
    recomputation_payload = recomputation["payload"]
    body: dict[str, Any] = {
        "state": "VERIFIED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "source_closure_ref": _artifact_ref(source),
        "recomputation_evidence_ref": _artifact_ref(recomputation),
        "factor_generation_ref": _artifact_ref(generation),
        "deployed_release_ref": source_payload["deployed_release_ref"],
        "release_install_evidence_ref": source_payload["release_install_evidence_ref"],
        "release_install_input_source_ref": source_payload["release_install_input_source_ref"],
        "legacy_zero_call_ref": source_payload["legacy_zero_call_ref"],
        "market_input_ref": _artifact_ref(market),
        "low_signal_sha256": recomputation_payload["low_signal_sha256"],
        "w80_signal_sha256": recomputation_payload["w80_signal_sha256"],
        "active_factor_rows": recomputation_payload["active_factor_rows"],
        "control_rows": recomputation_payload["control_rows"],
        "factor_readiness": FACTOR_READINESS_READY,
        "admission_route": ADMISSION_ROUTE,
        "producer_identity": PRODUCER_IDENTITY,
        "fundamental_dependency_state": FUNDAMENTAL_NOT_USED,
        "fundamental_freshness_policy": FUNDAMENTAL_ADVISORY,
        **{field: "NONE" for field in _NO_AUTHORITY_FIELDS},
    }
    receipt_id = "factor-production-receipt-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_RECEIPT_KIND,
        {"factor_production_receipt_id": receipt_id, **body},
        created_at=created_at,
        contract_sha256=get_contract(FACTOR_PRODUCTION_RECEIPT_KIND).contract_sha256,
    )
    return validate_factor_production_generation_receipt(
        artifact,
        factor_generation=generation,
        source_closure=source,
        recomputation_evidence=recomputation,
        legacy_zero_call_certificate=legacy,
        market_input=market,
    )


def validate_factor_production_generation_receipt(  # noqa: C901 - atomic exact cross-binding
    document: Mapping[str, Any] | bytes,
    *,
    factor_generation: Mapping[str, Any] | bytes | None = None,
    source_closure: Mapping[str, Any] | bytes | None = None,
    recomputation_evidence: Mapping[str, Any] | bytes | None = None,
    legacy_zero_call_certificate: Mapping[str, Any] | bytes | None = None,
    market_input: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    """Validate receipt structure, and deep-cross-bind supplied source artifacts."""

    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_RECEIPT_KIND)
    payload = artifact["payload"]
    if set(payload) != _RECEIPT_FIELDS:
        _raise("Factor production receipt fields differ")
    if (
        payload["state"] != "VERIFIED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["factor_readiness"] != FACTOR_READINESS_READY
        or payload["admission_route"] != ADMISSION_ROUTE
        or payload["producer_identity"] != PRODUCER_IDENTITY
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
    ):
        _raise("Factor production receipt policy differs")
    _require_no_authority(payload)
    _validate_ref(
        payload["source_closure_ref"],
        label="receipt source_closure_ref",
        expected_kind=FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND,
    )
    _validate_ref(
        payload["recomputation_evidence_ref"],
        label="receipt recomputation_evidence_ref",
        expected_kind=FACTOR_PRODUCTION_RECOMPUTATION_KIND,
    )
    _validate_factor_generation_ref(
        payload["factor_generation_ref"],
        label="receipt factor_generation_ref",
    )
    _validate_ref(
        payload["deployed_release_ref"],
        label="receipt deployed_release_ref",
        expected_kind="system.release",
    )
    _validate_ref(
        payload["release_install_evidence_ref"],
        label="receipt release_install_evidence_ref",
        expected_kind="system.release_install_evidence",
    )
    _validate_ref(
        payload["release_install_input_source_ref"],
        label="receipt release_install_input_source_ref",
        expected_kind="system.source_object",
    )
    _validate_ref(
        payload["legacy_zero_call_ref"],
        label="receipt legacy_zero_call_ref",
        expected_kind=FACTOR_LEGACY_ZERO_CALL_KIND,
    )
    _validate_ref(
        payload["market_input_ref"],
        label="receipt market_input_ref",
        expected_kind=FACTOR_PRODUCTION_MARKET_INPUT_KIND,
    )
    low_sha = _require_sha(payload["low_signal_sha256"], label="receipt LOW signal")
    _require_sha(payload["w80_signal_sha256"], label="receipt W80 signal")
    _validate_factor_policy_rows(payload["active_factor_rows"], payload["control_rows"])
    _identity("factor-production-receipt-", payload, "factor_production_receipt_id")
    supplied = (
        factor_generation,
        source_closure,
        recomputation_evidence,
        legacy_zero_call_certificate,
        market_input,
    )
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            _raise("Factor production receipt deep closure must supply all five artifacts")
        generation = _validate_factor_generation(factor_generation)  # type: ignore[arg-type]
        source = _validate_source_closure(source_closure)  # type: ignore[arg-type]
        recomputation = _validate_recomputation_evidence(
            recomputation_evidence  # type: ignore[arg-type]
        )
        legacy = _validate_legacy_zero_call_certificate(
            legacy_zero_call_certificate  # type: ignore[arg-type]
        )
        market = _validate_market_input(market_input)  # type: ignore[arg-type]
        _cross_bind_factor_generation(generation, source, recomputation, legacy, market)
        source_payload = source["payload"]
        recomputation_payload = recomputation["payload"]
        if payload["source_closure_ref"] != _artifact_ref(source) or payload[
            "recomputation_evidence_ref"
        ] != _artifact_ref(recomputation):
            _raise("Factor production receipt source/recomputation binding differs")
        if payload["factor_generation_ref"] != _artifact_ref(generation):
            _raise("Factor production receipt generation binding differs")
        for field in (
            "deployed_release_ref",
            "release_install_evidence_ref",
            "release_install_input_source_ref",
            "legacy_zero_call_ref",
            "market_input_ref",
        ):
            if payload[field] != source_payload[field]:
                _raise(f"Factor production receipt {field} differs from source closure")
        for field in (
            "low_signal_sha256",
            "w80_signal_sha256",
            "active_factor_rows",
            "control_rows",
        ):
            if payload[field] != recomputation_payload[field]:
                _raise(f"Factor production receipt {field} differs from recomputation")
        if payload["low_signal_sha256"] != low_sha:
            _raise("Factor production receipt LOW signal differs")
    return artifact


def _build_factor_pointer(
    *,
    receipt: Mapping[str, Any],
    activated_at: str,
    actor_uid: int,
    previous_pointer_sha256: str = FACTOR_EMPTY_POINTER_SHA256,
) -> dict[str, Any]:
    _timestamp(activated_at, label="activated_at")
    if type(actor_uid) is not int or actor_uid < 0:
        _raise("Factor activation actor UID is invalid")
    receipt_payload = receipt["payload"]
    generation_ref = receipt_payload["factor_generation_ref"]
    generation_id = generation_ref["artifact_id"]
    if not generation_id.startswith("factor-production-generation-"):
        _raise("Factor pointer generation identity differs")
    pointer_payload = {
        "factor_generation_id": generation_id,
        "factor_generation_sha256": generation_ref["byte_sha256"],
        "previous_pointer_sha256": previous_pointer_sha256,
        "activated_at": activated_at,
        "os_actor": f"uid:{actor_uid}",
        "authority_scope": FACTOR_PRODUCTION_SCOPE,
    }
    return _factor_pointer_record_from_raw(canonical_json_bytes(pointer_payload))


def _factor_pointer_record_from_raw(raw: bytes) -> dict[str, Any]:
    pointer_payload = validate_factor_active_pointer(raw)
    pointer_raw_sha256 = _sha256(raw)
    body = {**pointer_payload, "pointer_raw_sha256": pointer_raw_sha256}
    pointer_id = "factor-production-pointer-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_POINTER_KIND,
        {"factor_production_pointer_id": pointer_id, **body},
        created_at=pointer_payload["activated_at"],
        contract_sha256=get_contract(FACTOR_PRODUCTION_POINTER_KIND).contract_sha256,
    )
    return validate_factor_production_pointer(artifact)


def validate_factor_production_pointer(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_POINTER_KIND)
    payload = artifact["payload"]
    if set(payload) != _POINTER_FIELDS:
        _raise("Factor production pointer fields differ")
    pointer_payload = {field: payload[field] for field in _ACTIVE_POINTER_FIELDS}
    validate_factor_active_pointer(canonical_json_bytes(pointer_payload))
    if payload["pointer_raw_sha256"] != _sha256(canonical_json_bytes(pointer_payload)):
        _raise("Factor production pointer raw SHA differs")
    _identity("factor-production-pointer-", payload, "factor_production_pointer_id")
    return artifact


def validate_factor_active_pointer(raw: bytes) -> dict[str, Any]:
    """Validate the exact narrow six-field bytes written at Factor active path."""

    try:
        value = parse_canonical_json_bytes(raw, label="Factor active pointer")
    except ContractError as exc:
        raise FactorGovernanceError("Factor active pointer bytes are not canonical") from exc
    if type(value) is not dict or set(value) != _ACTIVE_POINTER_FIELDS:
        _raise("Factor active pointer fields differ")
    if value["authority_scope"] != FACTOR_PRODUCTION_SCOPE:
        _raise("Factor active pointer scope differs")
    generation_id = _require_text(
        value["factor_generation_id"], label="Factor pointer factor_generation_id"
    )
    if (
        not generation_id.startswith("factor-production-generation-")
        or _SHA256_RE.fullmatch(generation_id.removeprefix("factor-production-generation-")) is None
    ):
        _raise("Factor pointer generation identity differs")
    _require_sha(value["factor_generation_sha256"], label="Factor pointer factor_generation_sha256")
    previous = value["previous_pointer_sha256"]
    if previous != FACTOR_EMPTY_POINTER_SHA256:
        _require_sha(previous, label="Factor pointer previous_pointer_sha256")
    _timestamp(value["activated_at"], label="Factor pointer activated_at")
    actor = value.get("os_actor")
    if type(actor) is not str or _ACTOR_RE.fullmatch(actor) is None:
        _raise("Factor active pointer actor differs")
    return value


def _factor_pointer_raw(pointer: Mapping[str, Any]) -> bytes:
    validated = validate_factor_production_pointer(pointer)
    return canonical_json_bytes(
        {field: validated["payload"][field] for field in _ACTIVE_POINTER_FIELDS}
    )


def _build_activation_bundle(
    *, receipt: Mapping[str, Any], pointer: Mapping[str, Any], prepared_at: str, actor_uid: int
) -> dict[str, Any]:
    prepared = _timestamp(prepared_at, label="prepared_at")
    pointer_payload = validate_factor_active_pointer(_factor_pointer_raw(pointer))
    activated = _timestamp(pointer_payload["activated_at"], label="activated_at")
    if activated < prepared:
        _raise("Factor activation activated_at precedes prepared_at")
    if (
        type(actor_uid) is not int
        or actor_uid < 0
        or pointer_payload["os_actor"] != f"uid:{actor_uid}"
    ):
        _raise("Factor activation actor does not bind the exact pointer")
    receipt_payload = receipt["payload"]
    pointer_ref = _artifact_ref(pointer)
    body: dict[str, Any] = {
        "state": "PREPARED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "factor_generation_receipt_ref": _artifact_ref(receipt),
        "target_factor_generation_id": pointer_payload["factor_generation_id"],
        "target_factor_generation_ref": receipt_payload["factor_generation_ref"],
        "deployed_release_ref": receipt_payload["deployed_release_ref"],
        "active_factor_rows": receipt_payload["active_factor_rows"],
        "control_rows": receipt_payload["control_rows"],
        "low_signal_sha256": receipt_payload["low_signal_sha256"],
        "w80_signal_sha256": receipt_payload["w80_signal_sha256"],
        "factor_readiness": FACTOR_READINESS_READY,
        "market_input_ref": receipt_payload["market_input_ref"],
        "admission_route": ADMISSION_ROUTE,
        "producer_identity": PRODUCER_IDENTITY,
        "fundamental_dependency_state": FUNDAMENTAL_NOT_USED,
        "fundamental_freshness_policy": FUNDAMENTAL_ADVISORY,
        "target_factor_pointer_ref": pointer_ref,
        "target_factor_pointer_path": str(FACTOR_ACTIVE_POINTER_PATH),
        "expected_factor_pointer_sha256": FACTOR_EMPTY_POINTER_SHA256,
        "prepared_at": prepared_at,
        "activated_at": pointer_payload["activated_at"],
        "actor_uid": actor_uid,
        "os_actor": pointer_payload["os_actor"],
        **{field: "NONE" for field in _NO_AUTHORITY_FIELDS},
    }
    bundle_id = "factor-production-activation-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_BUNDLE_KIND,
        {"factor_production_activation_id": bundle_id, **body},
        created_at=prepared_at,
        contract_sha256=get_contract(FACTOR_PRODUCTION_BUNDLE_KIND).contract_sha256,
    )
    return validate_factor_production_activation_bundle(artifact, receipt=receipt, pointer=pointer)


def validate_factor_production_activation_bundle(  # noqa: C901 - atomic exact cross-binding
    document: Mapping[str, Any] | bytes,
    *,
    receipt: Mapping[str, Any] | bytes | None = None,
    pointer: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_BUNDLE_KIND)
    payload = artifact["payload"]
    if set(payload) != _BUNDLE_FIELDS:
        _raise("Factor activation bundle fields differ")
    if (
        payload["state"] != "PREPARED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["factor_readiness"] != FACTOR_READINESS_READY
        or payload["admission_route"] != ADMISSION_ROUTE
        or payload["producer_identity"] != PRODUCER_IDENTITY
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
        or payload["target_factor_pointer_path"] != str(FACTOR_ACTIVE_POINTER_PATH)
        or payload["expected_factor_pointer_sha256"] != FACTOR_EMPTY_POINTER_SHA256
    ):
        _raise("Factor activation bundle policy differs")
    _require_no_authority(payload)
    receipt_ref = _validate_ref(
        payload["factor_generation_receipt_ref"],
        label="bundle factor_generation_receipt_ref",
        expected_kind=FACTOR_PRODUCTION_RECEIPT_KIND,
    )
    generation_ref = _validate_factor_generation_ref(
        payload["target_factor_generation_ref"],
        label="bundle target_factor_generation_ref",
    )
    _validate_ref(
        payload["deployed_release_ref"],
        label="bundle deployed_release_ref",
        expected_kind="system.release",
    )
    pointer_ref = _validate_ref(
        payload["target_factor_pointer_ref"],
        label="bundle target_factor_pointer_ref",
        expected_kind=FACTOR_PRODUCTION_POINTER_KIND,
    )
    _validate_ref(
        payload["market_input_ref"],
        label="bundle market_input_ref",
        expected_kind=FACTOR_PRODUCTION_MARKET_INPUT_KIND,
    )
    if payload["target_factor_generation_id"] != generation_ref["artifact_id"]:
        _raise("Factor activation bundle generation identity differs")
    _require_sha(payload["low_signal_sha256"], label="bundle LOW signal")
    _require_sha(payload["w80_signal_sha256"], label="bundle W80 signal")
    _validate_factor_policy_rows(payload["active_factor_rows"], payload["control_rows"])
    prepared = _timestamp(payload["prepared_at"], label="bundle prepared_at")
    activated = _timestamp(payload["activated_at"], label="bundle activated_at")
    if activated < prepared or type(payload["actor_uid"]) is not int or payload["actor_uid"] < 0:
        _raise("Factor activation bundle clock or actor differs")
    if payload["os_actor"] != f"uid:{payload['actor_uid']}":
        _raise("Factor activation bundle actor differs")
    _identity("factor-production-activation-", payload, "factor_production_activation_id")
    supplied = (receipt, pointer)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            _raise("Factor activation bundle must bind both receipt and pointer")
        receipt_document = validate_factor_production_generation_receipt(
            receipt  # type: ignore[arg-type]
        )
        pointer_document = validate_factor_production_pointer(pointer)  # type: ignore[arg-type]
        receipt_payload = receipt_document["payload"]
        pointer_payload = validate_factor_active_pointer(_factor_pointer_raw(pointer_document))
        if receipt_ref != _artifact_ref(receipt_document) or pointer_ref != _artifact_ref(
            pointer_document
        ):
            _raise("Factor activation bundle receipt/pointer reference differs")
        for field, expected in (
            ("target_factor_generation_ref", receipt_payload["factor_generation_ref"]),
            (
                "target_factor_generation_id",
                receipt_payload["factor_generation_ref"]["artifact_id"],
            ),
            ("deployed_release_ref", receipt_payload["deployed_release_ref"]),
            ("active_factor_rows", receipt_payload["active_factor_rows"]),
            ("control_rows", receipt_payload["control_rows"]),
            ("low_signal_sha256", receipt_payload["low_signal_sha256"]),
            ("w80_signal_sha256", receipt_payload["w80_signal_sha256"]),
            ("market_input_ref", receipt_payload["market_input_ref"]),
        ):
            if payload[field] != expected:
                _raise(f"Factor activation bundle {field} differs from receipt")
        if (
            payload["target_factor_generation_id"] != pointer_payload["factor_generation_id"]
            or payload["target_factor_generation_ref"]["byte_sha256"]
            != pointer_payload["factor_generation_sha256"]
            or payload["activated_at"] != pointer_payload["activated_at"]
            or payload["os_actor"] != pointer_payload["os_actor"]
        ):
            _raise("Factor activation bundle pointer binding differs")
    return artifact


def _build_prepared_transaction(
    *, bundle: Mapping[str, Any], receipt: Mapping[str, Any], pointer: Mapping[str, Any]
) -> dict[str, Any]:
    bundle_payload = bundle["payload"]
    body = {
        "state": "PREPARED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "activation_bundle_ref": _artifact_ref(bundle),
        "factor_generation_receipt_ref": _artifact_ref(receipt),
        "target_factor_pointer_ref": _artifact_ref(pointer),
        "expected_factor_pointer_sha256": FACTOR_EMPTY_POINTER_SHA256,
        "prepared_at": bundle_payload["prepared_at"],
        "actor_uid": bundle_payload["actor_uid"],
    }
    prepared_id = "factor-production-prepared-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_PREPARED_KIND,
        {"factor_production_prepared_id": prepared_id, **body},
        created_at=bundle_payload["prepared_at"],
        contract_sha256=get_contract(FACTOR_PRODUCTION_PREPARED_KIND).contract_sha256,
    )
    return validate_factor_production_prepared(
        artifact, bundle=bundle, receipt=receipt, pointer=pointer
    )


def validate_factor_production_prepared(
    document: Mapping[str, Any] | bytes,
    *,
    bundle: Mapping[str, Any] | bytes | None = None,
    receipt: Mapping[str, Any] | bytes | None = None,
    pointer: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_PREPARED_KIND)
    payload = artifact["payload"]
    if set(payload) != _PREPARED_FIELDS:
        _raise("Factor prepared transaction fields differ")
    if (
        payload["state"] != "PREPARED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["expected_factor_pointer_sha256"] != FACTOR_EMPTY_POINTER_SHA256
    ):
        _raise("Factor prepared transaction policy differs")
    bundle_ref = _validate_ref(
        payload["activation_bundle_ref"],
        label="prepared activation_bundle_ref",
        expected_kind=FACTOR_PRODUCTION_BUNDLE_KIND,
    )
    receipt_ref = _validate_ref(
        payload["factor_generation_receipt_ref"],
        label="prepared receipt_ref",
        expected_kind=FACTOR_PRODUCTION_RECEIPT_KIND,
    )
    pointer_ref = _validate_ref(
        payload["target_factor_pointer_ref"],
        label="prepared pointer_ref",
        expected_kind=FACTOR_PRODUCTION_POINTER_KIND,
    )
    _timestamp(payload["prepared_at"], label="prepared transaction prepared_at")
    if type(payload["actor_uid"]) is not int or payload["actor_uid"] < 0:
        _raise("Factor prepared transaction actor differs")
    _identity("factor-production-prepared-", payload, "factor_production_prepared_id")
    supplied = (bundle, receipt, pointer)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            _raise("Factor prepared transaction must bind bundle, receipt, and pointer")
        bundle_document = validate_factor_production_activation_bundle(
            bundle  # type: ignore[arg-type]
        )
        receipt_document = validate_factor_production_generation_receipt(
            receipt  # type: ignore[arg-type]
        )
        pointer_document = validate_factor_production_pointer(pointer)  # type: ignore[arg-type]
        if (
            bundle_ref != _artifact_ref(bundle_document)
            or receipt_ref != _artifact_ref(receipt_document)
            or pointer_ref != _artifact_ref(pointer_document)
            or payload["prepared_at"] != bundle_document["payload"]["prepared_at"]
            or payload["actor_uid"] != bundle_document["payload"]["actor_uid"]
        ):
            _raise("Factor prepared transaction binding differs")
    return artifact


def _build_marker(
    *,
    receipt: Mapping[str, Any],
    pointer: Mapping[str, Any],
    bundle: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_payload = receipt["payload"]
    body: dict[str, Any] = {
        "state": "COMPLETE",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "activation_bundle_ref": _artifact_ref(bundle),
        "prepared_transaction_ref": _artifact_ref(prepared),
        "factor_generation_receipt_ref": _artifact_ref(receipt),
        "factor_pointer_ref": _artifact_ref(pointer),
        "factor_generation_ref": receipt_payload["factor_generation_ref"],
        "deployed_release_ref": receipt_payload["deployed_release_ref"],
        "active_factor_rows": receipt_payload["active_factor_rows"],
        "control_rows": receipt_payload["control_rows"],
        "factor_readiness": FACTOR_READINESS_READY,
        "factor_authority": FACTOR_AUTHORITY_ACTIVE,
        "market_input_ref": receipt_payload["market_input_ref"],
        "admission_route": ADMISSION_ROUTE,
        "producer_identity": PRODUCER_IDENTITY,
        "fundamental_dependency_state": FUNDAMENTAL_NOT_USED,
        "fundamental_freshness_policy": FUNDAMENTAL_ADVISORY,
        **{field: "NONE" for field in _NO_AUTHORITY_FIELDS},
    }
    marker_id = "factor-production-marker-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_MARKER_KIND,
        {"factor_production_marker_id": marker_id, **body},
        created_at=bundle["payload"]["prepared_at"],
        contract_sha256=get_contract(FACTOR_PRODUCTION_MARKER_KIND).contract_sha256,
    )
    return validate_factor_production_marker(
        artifact, receipt=receipt, pointer=pointer, bundle=bundle, prepared=prepared
    )


def validate_factor_production_marker(
    document: Mapping[str, Any] | bytes,
    *,
    receipt: Mapping[str, Any] | bytes | None = None,
    pointer: Mapping[str, Any] | bytes | None = None,
    bundle: Mapping[str, Any] | bytes | None = None,
    prepared: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_MARKER_KIND)
    payload = artifact["payload"]
    if set(payload) != _MARKER_FIELDS:
        _raise("Factor production marker fields differ")
    if (
        payload["state"] != "COMPLETE"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["factor_readiness"] != FACTOR_READINESS_READY
        or payload["factor_authority"] != FACTOR_AUTHORITY_ACTIVE
        or payload["admission_route"] != ADMISSION_ROUTE
        or payload["producer_identity"] != PRODUCER_IDENTITY
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
    ):
        _raise("Factor production marker policy differs")
    _require_no_authority(payload)
    bundle_ref = _validate_ref(
        payload["activation_bundle_ref"],
        label="marker activation_bundle_ref",
        expected_kind=FACTOR_PRODUCTION_BUNDLE_KIND,
    )
    prepared_ref = _validate_ref(
        payload["prepared_transaction_ref"],
        label="marker prepared_transaction_ref",
        expected_kind=FACTOR_PRODUCTION_PREPARED_KIND,
    )
    receipt_ref = _validate_ref(
        payload["factor_generation_receipt_ref"],
        label="marker receipt_ref",
        expected_kind=FACTOR_PRODUCTION_RECEIPT_KIND,
    )
    pointer_ref = _validate_ref(
        payload["factor_pointer_ref"],
        label="marker pointer_ref",
        expected_kind=FACTOR_PRODUCTION_POINTER_KIND,
    )
    _validate_ref(
        payload["market_input_ref"],
        label="marker market_input_ref",
        expected_kind=FACTOR_PRODUCTION_MARKET_INPUT_KIND,
    )
    _validate_factor_generation_ref(
        payload["factor_generation_ref"],
        label="marker factor_generation_ref",
    )
    _validate_ref(
        payload["deployed_release_ref"],
        label="marker deployed_release_ref",
        expected_kind="system.release",
    )
    _validate_factor_policy_rows(payload["active_factor_rows"], payload["control_rows"])
    _identity("factor-production-marker-", payload, "factor_production_marker_id")
    supplied = (receipt, pointer, bundle, prepared)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            _raise(
                "Factor production marker must bind receipt, pointer, bundle, "
                "and prepared transaction"
            )
        assert (
            receipt is not None
            and pointer is not None
            and bundle is not None
            and prepared is not None
        )
        receipt_document = validate_factor_production_generation_receipt(
            receipt  # type: ignore[arg-type]
        )
        pointer_document = validate_factor_production_pointer(pointer)  # type: ignore[arg-type]
        bundle_document = validate_factor_production_activation_bundle(
            bundle, receipt=receipt_document, pointer=pointer_document  # type: ignore[arg-type]
        )
        prepared_document = validate_factor_production_prepared(
            prepared,
            bundle=bundle_document,
            receipt=receipt_document,
            pointer=pointer_document,  # type: ignore[arg-type]
        )
        receipt_payload = receipt_document["payload"]
        if (
            bundle_ref != _artifact_ref(bundle_document)
            or prepared_ref != _artifact_ref(prepared_document)
            or receipt_ref != _artifact_ref(receipt_document)
            or pointer_ref != _artifact_ref(pointer_document)
        ):
            _raise("Factor production marker artifact binding differs")
        for field, expected in (
            ("factor_generation_ref", receipt_payload["factor_generation_ref"]),
            ("deployed_release_ref", receipt_payload["deployed_release_ref"]),
            ("active_factor_rows", receipt_payload["active_factor_rows"]),
            ("control_rows", receipt_payload["control_rows"]),
            ("market_input_ref", receipt_payload["market_input_ref"]),
        ):
            if payload[field] != expected:
                _raise(f"Factor production marker {field} differs from receipt")
    return artifact


def _build_rollover_bundle(
    *,
    predecessor_pointer: Mapping[str, Any],
    target_pointer: Mapping[str, Any],
    receipt: Mapping[str, Any],
    maintenance: Mapping[str, str],
    canonical_inputs: Mapping[str, str],
    target_date: str,
    prepared_at: str,
    actor_uid: int,
) -> dict[str, Any]:
    predecessor = validate_factor_production_pointer(predecessor_pointer)
    target = validate_factor_production_pointer(target_pointer)
    validated_receipt = validate_factor_production_generation_receipt(receipt)
    predecessor_raw = _factor_pointer_raw(predecessor)
    target_raw = _factor_pointer_raw(target)
    previous_sha = _sha256(predecessor_raw)
    target_sha = _sha256(target_raw)
    target_payload = validate_factor_active_pointer(target_raw)
    if target_payload["previous_pointer_sha256"] != previous_sha:
        _raise("Factor rollover target pointer preimage differs")
    for key in (
        "receipt_path",
        "receipt_sha256",
        "market_pointer_sha256",
        "market_manifest_sha256",
        "pit_pointer_sha256",
        "pit_manifest_sha256",
    ):
        if key not in {"receipt_path"}:
            _require_sha(
                (maintenance if key.startswith("receipt_") else canonical_inputs)[key],
                label=f"Factor rollover {key}",
            )
    body: dict[str, Any] = {
        "state": "PREPARED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "predecessor_pointer_ref": _artifact_ref(predecessor),
        "target_pointer_ref": _artifact_ref(target),
        "previous_pointer_sha256": previous_sha,
        "target_pointer_sha256": target_sha,
        "factor_generation_receipt_ref": _artifact_ref(validated_receipt),
        "target_factor_generation_ref": validated_receipt["payload"]["factor_generation_ref"],
        "maintenance_receipt_path": _require_text(
            maintenance["receipt_path"], label="Factor rollover maintenance path"
        ),
        "maintenance_receipt_sha256": maintenance["receipt_sha256"],
        "market_pointer_sha256": canonical_inputs["market_pointer_sha256"],
        "market_manifest_sha256": canonical_inputs["market_manifest_sha256"],
        "pit_pointer_sha256": canonical_inputs["pit_pointer_sha256"],
        "pit_manifest_sha256": canonical_inputs["pit_manifest_sha256"],
        "target_date": _require_text(target_date, label="Factor rollover target date"),
        "prepared_at": prepared_at,
        "actor_uid": actor_uid,
        **{field: "NONE" for field in _NO_AUTHORITY_FIELDS},
    }
    identity = "factor-production-rollover-bundle-" + _sha256(canonical_json_bytes(body))
    return validate_factor_production_rollover_bundle(
        seal_artifact(
            FACTOR_PRODUCTION_ROLLOVER_BUNDLE_KIND,
            {"factor_production_rollover_bundle_id": identity, **body},
            created_at=prepared_at,
            contract_sha256=get_contract(FACTOR_PRODUCTION_ROLLOVER_BUNDLE_KIND).contract_sha256,
        ),
        predecessor_pointer=predecessor,
        target_pointer=target,
        receipt=validated_receipt,
    )


def validate_factor_production_rollover_bundle(
    document: Mapping[str, Any] | bytes,
    *,
    predecessor_pointer: Mapping[str, Any] | bytes | None = None,
    target_pointer: Mapping[str, Any] | bytes | None = None,
    receipt: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_ROLLOVER_BUNDLE_KIND)
    payload = artifact["payload"]
    if set(payload) != _ROLLOVER_BUNDLE_FIELDS:
        _raise("Factor rollover bundle fields differ")
    if payload["state"] != "PREPARED" or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE:
        _raise("Factor rollover bundle policy differs")
    _require_no_authority(payload)
    _require_sha(payload["previous_pointer_sha256"], label="rollover predecessor SHA")
    _require_sha(payload["target_pointer_sha256"], label="rollover target SHA")
    for field in (
        "maintenance_receipt_sha256",
        "market_pointer_sha256",
        "market_manifest_sha256",
        "pit_pointer_sha256",
        "pit_manifest_sha256",
    ):
        _require_sha(payload[field], label=f"rollover {field}")
    _require_text(payload["maintenance_receipt_path"], label="rollover maintenance path")
    target_date = _require_text(payload["target_date"], label="rollover target date")
    if len(target_date) != 8 or not target_date.isdigit():
        _raise("Factor rollover target date differs")
    _timestamp(payload["prepared_at"], label="rollover prepared_at")
    if type(payload["actor_uid"]) is not int or payload["actor_uid"] < 0:
        _raise("Factor rollover actor differs")
    predecessor_ref = _validate_ref(
        payload["predecessor_pointer_ref"],
        label="rollover predecessor pointer ref",
        expected_kind=FACTOR_PRODUCTION_POINTER_KIND,
    )
    target_ref = _validate_ref(
        payload["target_pointer_ref"],
        label="rollover target pointer ref",
        expected_kind=FACTOR_PRODUCTION_POINTER_KIND,
    )
    receipt_ref = _validate_ref(
        payload["factor_generation_receipt_ref"],
        label="rollover receipt ref",
        expected_kind=FACTOR_PRODUCTION_RECEIPT_KIND,
    )
    generation_ref = _validate_factor_generation_ref(
        payload["target_factor_generation_ref"], label="rollover generation ref"
    )
    _identity(
        "factor-production-rollover-bundle-",
        payload,
        "factor_production_rollover_bundle_id",
    )
    supplied = (predecessor_pointer, target_pointer, receipt)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            _raise("Factor rollover bundle closure is incomplete")
        predecessor = validate_factor_production_pointer(predecessor_pointer)  # type: ignore[arg-type]
        target = validate_factor_production_pointer(target_pointer)  # type: ignore[arg-type]
        validated_receipt = validate_factor_production_generation_receipt(receipt)  # type: ignore[arg-type]
        predecessor_raw = _factor_pointer_raw(predecessor)
        target_raw = _factor_pointer_raw(target)
        target_payload = validate_factor_active_pointer(target_raw)
        if (
            predecessor_ref != _artifact_ref(predecessor)
            or target_ref != _artifact_ref(target)
            or receipt_ref != _artifact_ref(validated_receipt)
            or generation_ref != validated_receipt["payload"]["factor_generation_ref"]
            or payload["previous_pointer_sha256"] != _sha256(predecessor_raw)
            or payload["target_pointer_sha256"] != _sha256(target_raw)
            or target_payload["previous_pointer_sha256"] != _sha256(predecessor_raw)
            or target_payload["factor_generation_sha256"] != generation_ref["byte_sha256"]
        ):
            _raise("Factor rollover bundle binding differs")
    return artifact


def _build_rollover_prepared(
    *, bundle: Mapping[str, Any], prepared_at: str, actor_uid: int
) -> dict[str, Any]:
    validated = validate_factor_production_rollover_bundle(bundle)
    payload = validated["payload"]
    body = {
        "state": "PREPARED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "rollover_bundle_ref": _artifact_ref(validated),
        "expected_pointer_sha256": payload["previous_pointer_sha256"],
        "target_pointer_sha256": payload["target_pointer_sha256"],
        "prepared_at": prepared_at,
        "actor_uid": actor_uid,
    }
    identity = "factor-production-rollover-prepared-" + _sha256(canonical_json_bytes(body))
    return validate_factor_production_rollover_prepared(
        seal_artifact(
            FACTOR_PRODUCTION_ROLLOVER_PREPARED_KIND,
            {"factor_production_rollover_prepared_id": identity, **body},
            created_at=prepared_at,
            contract_sha256=get_contract(FACTOR_PRODUCTION_ROLLOVER_PREPARED_KIND).contract_sha256,
        ),
        bundle=validated,
    )


def validate_factor_production_rollover_prepared(
    document: Mapping[str, Any] | bytes,
    *,
    bundle: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_ROLLOVER_PREPARED_KIND)
    payload = artifact["payload"]
    if set(payload) != _ROLLOVER_PREPARED_FIELDS:
        _raise("Factor rollover prepared fields differ")
    if payload["state"] != "PREPARED" or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE:
        _raise("Factor rollover prepared policy differs")
    bundle_ref = _validate_ref(
        payload["rollover_bundle_ref"],
        label="rollover prepared bundle ref",
        expected_kind=FACTOR_PRODUCTION_ROLLOVER_BUNDLE_KIND,
    )
    _require_sha(payload["expected_pointer_sha256"], label="rollover expected pointer SHA")
    _require_sha(payload["target_pointer_sha256"], label="rollover target pointer SHA")
    _timestamp(payload["prepared_at"], label="rollover prepared timestamp")
    if type(payload["actor_uid"]) is not int or payload["actor_uid"] < 0:
        _raise("Factor rollover prepared actor differs")
    _identity(
        "factor-production-rollover-prepared-",
        payload,
        "factor_production_rollover_prepared_id",
    )
    if bundle is not None:
        validated = validate_factor_production_rollover_bundle(bundle)
        if (
            bundle_ref != _artifact_ref(validated)
            or payload["expected_pointer_sha256"] != validated["payload"]["previous_pointer_sha256"]
            or payload["target_pointer_sha256"] != validated["payload"]["target_pointer_sha256"]
        ):
            _raise("Factor rollover prepared binding differs")
    return artifact


def _build_rollover_commit(
    *, bundle: Mapping[str, Any], prepared: Mapping[str, Any], committed_at: str, actor_uid: int
) -> dict[str, Any]:
    validated_bundle = validate_factor_production_rollover_bundle(bundle)
    validated_prepared = validate_factor_production_rollover_prepared(
        prepared, bundle=validated_bundle
    )
    bundle_payload = validated_bundle["payload"]
    body: dict[str, Any] = {
        "state": "COMMITTED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "rollover_bundle_ref": _artifact_ref(validated_bundle),
        "rollover_prepared_ref": _artifact_ref(validated_prepared),
        "previous_pointer_sha256": bundle_payload["previous_pointer_sha256"],
        "target_pointer_sha256": bundle_payload["target_pointer_sha256"],
        "committed_at": committed_at,
        "actor_uid": actor_uid,
        "cas_performed": True,
        **{field: "NONE" for field in _NO_AUTHORITY_FIELDS},
    }
    identity = "factor-production-rollover-commit-" + _sha256(canonical_json_bytes(body))
    return validate_factor_production_rollover_commit(
        seal_artifact(
            FACTOR_PRODUCTION_ROLLOVER_COMMIT_KIND,
            {"factor_production_rollover_commit_id": identity, **body},
            created_at=committed_at,
            contract_sha256=get_contract(FACTOR_PRODUCTION_ROLLOVER_COMMIT_KIND).contract_sha256,
        ),
        bundle=validated_bundle,
        prepared=validated_prepared,
    )


def validate_factor_production_rollover_commit(
    document: Mapping[str, Any] | bytes,
    *,
    bundle: Mapping[str, Any] | bytes | None = None,
    prepared: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    artifact = _validate_artifact_kind(document, FACTOR_PRODUCTION_ROLLOVER_COMMIT_KIND)
    payload = artifact["payload"]
    if set(payload) != _ROLLOVER_COMMIT_FIELDS:
        _raise("Factor rollover commit fields differ")
    if (
        payload["state"] != "COMMITTED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["cas_performed"] is not True
    ):
        _raise("Factor rollover commit policy differs")
    _require_no_authority(payload)
    bundle_ref = _validate_ref(
        payload["rollover_bundle_ref"],
        label="rollover commit bundle ref",
        expected_kind=FACTOR_PRODUCTION_ROLLOVER_BUNDLE_KIND,
    )
    prepared_ref = _validate_ref(
        payload["rollover_prepared_ref"],
        label="rollover commit prepared ref",
        expected_kind=FACTOR_PRODUCTION_ROLLOVER_PREPARED_KIND,
    )
    _require_sha(payload["previous_pointer_sha256"], label="rollover commit previous SHA")
    _require_sha(payload["target_pointer_sha256"], label="rollover commit target SHA")
    _timestamp(payload["committed_at"], label="rollover commit timestamp")
    if type(payload["actor_uid"]) is not int or payload["actor_uid"] < 0:
        _raise("Factor rollover commit actor differs")
    _identity(
        "factor-production-rollover-commit-",
        payload,
        "factor_production_rollover_commit_id",
    )
    supplied = (bundle, prepared)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            _raise("Factor rollover commit closure is incomplete")
        validated_bundle = validate_factor_production_rollover_bundle(bundle)  # type: ignore[arg-type]
        validated_prepared = validate_factor_production_rollover_prepared(
            prepared, bundle=validated_bundle  # type: ignore[arg-type]
        )
        if (
            bundle_ref != _artifact_ref(validated_bundle)
            or prepared_ref != _artifact_ref(validated_prepared)
            or payload["previous_pointer_sha256"]
            != validated_bundle["payload"]["previous_pointer_sha256"]
            or payload["target_pointer_sha256"]
            != validated_bundle["payload"]["target_pointer_sha256"]
        ):
            _raise("Factor rollover commit binding differs")
    return artifact


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


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify_factor_directory(value: os.stat_result, *, governed: bool) -> None:
    if not stat.S_ISDIR(value.st_mode):
        _raise("Factor authority path component is not a directory")
    if governed and (value.st_uid != os.geteuid() or stat.S_IMODE(value.st_mode) != 0o700):
        _raise("Factor authority directory must be owner-only")


def _verify_factor_file(value: os.stat_result) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o600
    ):
        _raise("Factor authority artifact must be an owner-only regular file")
    if value.st_nlink != 1:
        _raise("Factor authority artifact must have exactly one hard link")


def _factor_governed(parts: tuple[str, ...]) -> bool:
    path = PurePosixPath(*parts)
    return path == FACTOR_ROOT or FACTOR_ROOT in path.parents


class _FactorSecureStorage:
    """Descriptor-relative, no-follow storage dedicated to the Factor root.

    This is intentionally not a generic System storage subclass: it cannot
    canonicalize or reach ``results/system`` at all.
    """

    _test_fault_hook: ClassVar[Callable[[str], None] | None] = None

    def __init__(self, workspace_root: str | os.PathLike[str]) -> None:
        try:
            root = Path(workspace_root).resolve(strict=True)
            metadata = root.stat(follow_symlinks=False)
        except (OSError, TypeError, ValueError) as exc:
            raise FactorGovernanceError("Factor production workspace root is unavailable") from exc
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            _raise("Factor production workspace root is unsafe")
        self.workspace_root = root
        test_hook = type(self)._test_fault_hook
        self._fault_hook: Callable[[str], None] = test_hook or (lambda _point: None)

    def _open_workspace(self) -> int:
        descriptor: int | None = None
        try:
            descriptor = os.open("/", _DIRECTORY_FLAGS)
            for part in self.workspace_root.parts[1:]:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                _raise("Factor production workspace root is unsafe")
            return descriptor
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise FactorGovernanceError("Factor workspace cannot be securely opened") from exc
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise

    @staticmethod
    def _reject_casefold_alias(parent_fd: int, leaf: str) -> None:
        count = 0
        try:
            with os.scandir(parent_fd) as entries:
                for entry in entries:
                    count += 1
                    if count > 100_000:
                        _raise("Factor authority directory collision check exceeded its bound")
                    if entry.name != leaf and entry.name.casefold() == leaf.casefold():
                        _raise("Factor authority path has a casefold collision")
        except FactorGovernanceError:
            raise
        except OSError as exc:
            raise FactorGovernanceError("Factor authority directory cannot be enumerated") from exc

    def _open_directory(  # noqa: C901 - descriptor-safe traversal is atomic
        self, parts: tuple[str, ...], *, create: bool
    ) -> int:
        descriptor = self._open_workspace()
        traversed: list[str] = []
        try:
            for part in parts:
                traversed.append(part)
                self._reject_casefold_alias(descriptor, part)
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise FileNotFoundError(str(PurePosixPath(*traversed))) from None
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                    if _factor_governed(tuple(traversed)):
                        os.fchmod(child, 0o700)
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        _raise("Factor authority symlink/non-directory path is rejected")
                    raise FactorGovernanceError(
                        "Factor authority directory cannot be opened"
                    ) from exc
                _verify_factor_directory(
                    os.fstat(child), governed=_factor_governed(tuple(traversed))
                )
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _parent_leaf(
        self, value: str | PurePosixPath, *, create: bool
    ) -> tuple[int, str, PurePosixPath]:
        path = _canonical_path(value)
        if path == FACTOR_ROOT:
            _raise("Factor authority file path cannot be its root")
        parent = self._open_directory(tuple(path.parts[:-1]), create=create)
        self._reject_casefold_alias(parent, path.name)
        return parent, path.name, path

    def _read_leaf(  # noqa: C901 - descriptor identity/readback is atomic
        self,
        parent_fd: int,
        leaf: str,
        *,
        relative_path: PurePosixPath,
        optional: bool,
        maximum_bytes: int = MAX_CANONICAL_JSON_BYTES,
    ) -> FactorStoredBytes | None:
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
            except FileNotFoundError:
                if optional:
                    return None
                raise FileNotFoundError(str(relative_path)) from None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    _raise("Factor authority artifact symlink is rejected")
                raise FactorGovernanceError("Factor authority artifact cannot be opened") from exc
            before = os.fstat(descriptor)
            _verify_factor_file(before)
            if before.st_size <= 0 or before.st_size > maximum_bytes:
                _raise("Factor authority artifact size is invalid")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    _raise("Factor authority artifact read was short")
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            _verify_factor_file(after)
            path_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(after) != _stat_identity(path_after)
                or len(raw) != after.st_size
            ):
                _raise("Factor authority artifact changed during exact read")
            return FactorStoredBytes(str(relative_path), raw, _sha256(raw))
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def _write_all(descriptor: int, raw: bytes) -> None:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _raise("Factor authority write was short")
            view = view[written:]

    def read_optional(self, value: str | PurePosixPath) -> FactorStoredBytes | None:
        try:
            parent, leaf, path = self._parent_leaf(value, create=False)
        except FileNotFoundError:
            return None
        try:
            return self._read_leaf(parent, leaf, relative_path=path, optional=True)
        finally:
            os.close(parent)

    def read(self, value: str | PurePosixPath) -> FactorStoredBytes:
        stored = self.read_optional(value)
        if stored is None:
            _raise("Factor authority artifact is absent")
        return stored

    def read_blob(self, value: str | PurePosixPath, *, maximum_bytes: int) -> FactorStoredBytes:
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            _raise("Factor source mirror byte bound is invalid")
        try:
            parent, leaf, path = self._parent_leaf(value, create=False)
        except FileNotFoundError:
            _raise("Factor source mirror is absent")
        try:
            stored = self._read_leaf(
                parent, leaf, relative_path=path, optional=False, maximum_bytes=maximum_bytes
            )
            if stored is None:  # pragma: no cover - optional=False is exhaustive
                _raise("Factor source mirror is absent")
            return stored
        finally:
            os.close(parent)

    def write_exact_once(
        self,
        value: str | PurePosixPath,
        raw: bytes,
    ) -> FactorStoredBytes:
        """Public generic writer: reserved Factor authority is always closed."""

        return self._write_exact_once(value, raw)

    def _write_blob_exact_once(
        self,
        value: str | PurePosixPath,
        raw: bytes,
        *,
        maximum_bytes: int,
    ) -> FactorStoredBytes:
        return self._write_exact_once(
            value,
            raw,
            maximum_bytes=maximum_bytes,
        )

    def _write_exact_once(
        self,
        value: str | PurePosixPath,
        raw: bytes,
        *,
        maximum_bytes: int = MAX_CANONICAL_JSON_BYTES,
    ) -> FactorStoredBytes:
        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or type(raw) is not bytes
            or not raw
            or len(raw) > maximum_bytes
        ):
            _raise("Factor authority artifact bytes are invalid")
        path = _canonical_path(value)
        if _is_reserved_factor_authority_path(path):
            _raise("reserved Factor authority path requires sealed activation")
        parent, leaf, relative = self._parent_leaf(path, create=True)
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(leaf, _CREATE_FLAGS, 0o600, dir_fd=parent)
            except FileExistsError:
                stored = self._read_leaf(
                    parent,
                    leaf,
                    relative_path=relative,
                    optional=False,
                    maximum_bytes=maximum_bytes,
                )
                if stored is None or stored.data != raw:
                    _raise("Factor authority immutable artifact conflicts")
                return stored
            os.fchmod(descriptor, 0o600)
            self._write_all(descriptor, raw)
            os.fsync(descriptor)
            _verify_factor_file(os.fstat(descriptor))
            os.close(descriptor)
            descriptor = None
            os.fsync(parent)
            stored = self._read_leaf(
                parent, leaf, relative_path=relative, optional=False, maximum_bytes=maximum_bytes
            )
            if stored is None or stored.data != raw:
                _raise("Factor authority immutable artifact readback differs")
            return stored
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def _write_reserved_atomic_no_replace(  # noqa: C901 - atomic authority publication
        self,
        value: str | PurePosixPath,
        raw: bytes,
        *,
        idempotent_existing: bool,
        before_rename_point: str,
        after_rename_point: str,
    ) -> FactorStoredBytes:
        """Publish complete reserved bytes with a native no-replace rename."""

        if type(raw) is not bytes or not raw or len(raw) > MAX_CANONICAL_JSON_BYTES:
            _raise("Factor authority reserved bytes are invalid")
        path = _canonical_path(value)
        if path not in _RESERVED_FACTOR_AUTHORITY_PATHS:
            _raise("atomic reserved publication requires an exact authority path")
        parent, leaf, relative = self._parent_leaf(path, create=True)
        temporary = f".{leaf}.publish-{os.getpid()}-{secrets.token_hex(8)}"
        descriptor: int | None = None
        try:
            current = self._read_leaf(parent, leaf, relative_path=relative, optional=True)
            if current is not None:
                if idempotent_existing and current.data == raw:
                    return current
                if path == FACTOR_ACTIVE_POINTER_PATH:
                    _raise("Factor pointer preimage is no longer EMPTY")
                _raise("Factor permanent marker conflicts with exact prepared marker")
            descriptor = os.open(temporary, _CREATE_FLAGS, 0o600, dir_fd=parent)
            os.fchmod(descriptor, 0o600)
            self._write_all(descriptor, raw)
            os.fsync(descriptor)
            _verify_factor_file(os.fstat(descriptor))
            os.close(descriptor)
            descriptor = None
            self._fault_hook(before_rename_point)
            try:
                _atomic_no_replace_rename(
                    temporary,
                    leaf,
                    source_directory_fd=parent,
                    destination_directory_fd=parent,
                )
            except FileExistsError:
                observed = self._read_leaf(parent, leaf, relative_path=relative, optional=False)
                if idempotent_existing and observed is not None and observed.data == raw:
                    return observed
                if path == FACTOR_ACTIVE_POINTER_PATH:
                    _raise("Factor pointer preimage is no longer EMPTY")
                _raise("Factor permanent marker conflicts with exact prepared marker")
            os.fsync(parent)
            self._fault_hook(after_rename_point)
            stored = self._read_leaf(parent, leaf, relative_path=relative, optional=False)
            if stored is None or stored.data != raw:
                _raise("Factor authority atomic no-replace readback differs")
            return stored
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    @contextmanager
    def exclusive_lock(self, value: str | PurePosixPath) -> Iterator[None]:
        parent, leaf, _relative = self._parent_leaf(value, create=True)
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(leaf, _CREATE_FLAGS, 0o600, dir_fd=parent)
            except FileExistsError:
                try:
                    descriptor = os.open(
                        leaf,
                        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=parent,
                    )
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        _raise("Factor authority lock path is unsafe")
                    raise FactorGovernanceError(
                        f"Factor authority lock cannot be opened (errno={exc.errno})"
                    ) from exc
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    _raise("Factor authority lock path is unsafe")
                raise FactorGovernanceError(
                    f"Factor authority lock cannot be opened (errno={exc.errno})"
                ) from exc
            os.fchmod(descriptor, 0o600)
            _verify_factor_file(os.fstat(descriptor))
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
            os.close(parent)

    def write_initial_pointer_under_lock(self, raw: bytes) -> FactorStoredBytes:
        return self._write_reserved_atomic_no_replace(
            FACTOR_ACTIVE_POINTER_PATH,
            raw,
            idempotent_existing=False,
            before_rename_point="BEFORE_POINTER_RENAME",
            after_rename_point="AFTER_POINTER_RENAME",
        )

    def write_permanent_marker_under_lock(self, raw: bytes) -> FactorStoredBytes:
        return self._write_reserved_atomic_no_replace(
            FACTOR_PRODUCTION_MARKER_PATH,
            raw,
            idempotent_existing=True,
            before_rename_point="BEFORE_MARKER_RENAME",
            after_rename_point="AFTER_MARKER_RENAME",
        )

    def replace_active_pointer_under_lock(
        self, raw: bytes, *, expected_pointer_sha256: str
    ) -> FactorStoredBytes:
        """Cooperatively replace the active pointer after an exact under-lock preimage read."""

        _require_sha(expected_pointer_sha256, label="expected Factor pointer SHA")
        if type(raw) is not bytes or not raw or len(raw) > MAX_CANONICAL_JSON_BYTES:
            _raise("Factor rollover pointer bytes are invalid")
        validate_factor_active_pointer(raw)
        parent, leaf, relative = self._parent_leaf(FACTOR_ACTIVE_POINTER_PATH, create=False)
        temporary = f".{leaf}.rollover-{os.getpid()}-{secrets.token_hex(8)}"
        descriptor: int | None = None
        try:
            current = self._read_leaf(parent, leaf, relative_path=relative, optional=False)
            if current is None or current.byte_sha256 != expected_pointer_sha256:
                _raise("Factor rollover pointer preimage changed")
            descriptor = os.open(temporary, _CREATE_FLAGS, 0o600, dir_fd=parent)
            os.fchmod(descriptor, 0o600)
            self._write_all(descriptor, raw)
            os.fsync(descriptor)
            _verify_factor_file(os.fstat(descriptor))
            os.close(descriptor)
            descriptor = None
            self._fault_hook("BEFORE_ROLLOVER_POINTER_REPLACE")
            observed = self._read_leaf(parent, leaf, relative_path=relative, optional=False)
            if observed is None or observed.byte_sha256 != expected_pointer_sha256:
                _raise("Factor rollover pointer preimage changed")
            os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            self._fault_hook("AFTER_ROLLOVER_POINTER_REPLACE")
            stored = self._read_leaf(parent, leaf, relative_path=relative, optional=False)
            if stored is None or stored.data != raw:
                _raise("Factor rollover pointer readback differs")
            return stored
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)


class FactorReadOnlySystemCustody:
    """Narrow read-only resolver for immutable System source custody.

    The wrapper intentionally exposes only object lookup and strict source
    inspection/readback.  It has no active-pointer, marker, generation
    publication, activation, portfolio, or trading method.
    """

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        source_root: str | os.PathLike[str],
        source_root_id: str,
    ) -> None:
        # This import is deliberately local: Factor authority depends only on
        # the immutable System *source* custody API, never System activation.
        from quant_investor.system.store import SystemStore

        source_store = SystemStore(
            workspace_root,
            source_root=source_root,
            source_root_id=source_root_id,
        )
        from .governance.production_authority import system_store_source_resolver

        # Retain only the two read closures, not a public-capability-like
        # SystemStore handle that callers could use for activation methods.
        self._artifact_reader = source_store.get_object
        self._source_reader = system_store_source_resolver(source_store)

    def artifact_resolver(self, ref: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._artifact_reader(ref)

    def source_resolver(
        self, ref: Mapping[str, Any], maximum_bytes: int
    ) -> tuple[Mapping[str, Any], bytes]:
        return self._source_reader(ref, maximum_bytes)


class FactorProductionStore:
    """Owner-only, descriptor-safe storage for isolated Factor authority bytes."""

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        source_custody: FactorReadOnlySystemCustody | None = None,
        release_repository_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if source_custody is not None and type(source_custody) is not FactorReadOnlySystemCustody:
            _raise("Factor production source custody must be the read-only System custody adapter")
        self._storage = _FactorSecureStorage(workspace_root)
        self.workspace_root = self._storage.workspace_root
        self._source_custody = source_custody
        if release_repository_root is None:
            self._release_repository_root: Path | None = None
        else:
            try:
                repository_root = Path(release_repository_root).resolve(strict=True)
                repository_metadata = repository_root.stat(follow_symlinks=False)
            except (OSError, TypeError, ValueError) as exc:
                raise FactorGovernanceError(
                    "Factor release repository root is unavailable"
                ) from exc
            if (
                not stat.S_ISDIR(repository_metadata.st_mode)
                or repository_metadata.st_uid != os.geteuid()
            ):
                _raise("Factor release repository root is unsafe")
            self._release_repository_root = repository_root

    @classmethod
    def from_system_source_custody(
        cls,
        workspace_root: str | os.PathLike[str],
        *,
        source_root: str | os.PathLike[str],
        source_root_id: str,
        release_repository_root: str | os.PathLike[str],
    ) -> "FactorProductionStore":
        """Build the only supported live-source resolver without System activation.

        The resulting store cannot inspect System active-pointer state through
        this wrapper.  It may read exact immutable source objects only.
        """

        custody = FactorReadOnlySystemCustody(
            workspace_root, source_root=source_root, source_root_id=source_root_id
        )
        return cls(
            workspace_root,
            source_custody=custody,
            release_repository_root=release_repository_root,
        )

    def read_optional(self, value: str | PurePosixPath) -> FactorStoredBytes | None:
        return self._storage.read_optional(value)

    def read(self, value: str | PurePosixPath) -> FactorStoredBytes:
        return self._storage.read(value)

    def write_exact_once(self, value: str | PurePosixPath, raw: bytes) -> FactorStoredBytes:
        """Publish ordinary Factor artifacts but never either authority pointer."""

        return self._storage.write_exact_once(value, raw)

    @contextmanager
    def _active_lock(self) -> Iterator[None]:
        with self._storage.exclusive_lock(FACTOR_ACTIVE_LOCK_PATH):
            yield

    def _write_initial_pointer_under_lock(self, raw: bytes) -> FactorStoredBytes:
        return self._storage.write_initial_pointer_under_lock(raw)

    def _write_permanent_marker_under_lock(self, raw: bytes) -> FactorStoredBytes:
        return self._storage.write_permanent_marker_under_lock(raw)

    def _replace_active_pointer_under_lock(
        self, raw: bytes, *, expected_pointer_sha256: str
    ) -> FactorStoredBytes:
        return self._storage.replace_active_pointer_under_lock(
            raw, expected_pointer_sha256=expected_pointer_sha256
        )

    @staticmethod
    def _artifact_path(ref: Mapping[str, str]) -> PurePosixPath:
        kind = ref["kind"]
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{1,127}", kind):
            _raise("Factor artifact kind path is invalid")
        return FACTOR_OBJECTS_ROOT / kind / f"{ref['byte_sha256']}.json"

    def _publish_artifact(self, artifact: Mapping[str, Any] | bytes) -> FactorStoredBytes:
        document = validate_artifact(artifact)
        raw = canonical_json_bytes(document)
        ref = _artifact_ref(document)
        stored = self._storage.write_exact_once(self._artifact_path(ref), raw)
        if stored.byte_sha256 != ref["byte_sha256"]:
            _raise("Factor artifact object SHA differs")
        return stored

    def _read_artifact_ref(self, ref: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        normalized = _validate_ref(ref, label=label)
        stored = self._storage.read(self._artifact_path(normalized))
        if stored.byte_sha256 != normalized["byte_sha256"]:
            _raise(f"{label} stored byte SHA differs")
        document = _validate_artifact_kind(stored.data, normalized["kind"])
        if _artifact_ref(document) != normalized:
            _raise(f"{label} stored artifact ref differs")
        return document

    def _read_factor_generation_ref(self, ref: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        normalized = _validate_factor_generation_ref(ref, label=label)
        generation = _validate_factor_generation(self._read_artifact_ref(normalized, label=label))
        generation_path = (
            FACTOR_GENERATIONS_ROOT
            / generation["payload"]["factor_production_generation_id"]
            / "generation.json"
        )
        stored = self._storage.read(generation_path)
        if stored.data != canonical_json_bytes(generation):
            _raise(f"{label} immutable generation bytes differ")
        return generation

    def _read_generation_for_pointer(self, pointer_raw: bytes, *, label: str) -> dict[str, Any]:
        pointer = validate_factor_active_pointer(pointer_raw)
        generation_path = (
            FACTOR_GENERATIONS_ROOT / pointer["factor_generation_id"] / "generation.json"
        )
        stored = self._storage.read(generation_path)
        try:
            generation = validate_artifact(
                stored.data, expected_kind=FACTOR_PRODUCTION_GENERATION_KIND
            )
        except ContractError as exc:
            raise FactorGovernanceError(f"{label} generation contract differs") from exc
        if (
            stored.byte_sha256 != pointer["factor_generation_sha256"]
            or generation["payload"]["factor_production_generation_id"]
            != pointer["factor_generation_id"]
        ):
            _raise(f"{label} generation differs from pointer")
        return generation

    @staticmethod
    def _source_mirror_paths(ref: Mapping[str, str]) -> tuple[PurePosixPath, PurePosixPath]:
        normalized = _validate_ref(ref, label="Factor source mirror ref")
        token = normalized["byte_sha256"]
        return (
            FACTOR_SOURCE_MIRRORS_ROOT / f"{token}.json",
            FACTOR_SOURCE_MIRRORS_ROOT / f"{token}.bin",
        )

    def _capture_artifact_resolver(
        self,
        resolver: Callable[[Mapping[str, Any]], Mapping[str, Any] | bytes],
    ) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
        def capture(ref: Mapping[str, Any]) -> Mapping[str, Any]:
            normalized = _validate_ref(ref, label="live Factor closure artifact ref")
            if normalized["kind"] in _LOCAL_FACTOR_CLOSURE_KINDS:
                return self._read_artifact_ref(normalized, label="Factor-local closure artifact")
            document = _validate_artifact_kind(resolver(normalized), normalized["kind"])
            if _artifact_ref(document) != normalized:
                _raise("live Factor closure artifact resolver returned a different artifact")
            self._publish_artifact(document)
            return document

        return capture

    def _live_or_local_artifact_resolver(
        self,
        resolver: Callable[[Mapping[str, Any]], Mapping[str, Any] | bytes],
    ) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
        def resolve(ref: Mapping[str, Any]) -> Mapping[str, Any]:
            normalized = _validate_ref(ref, label="live Factor closure artifact ref")
            if normalized["kind"] in _LOCAL_FACTOR_CLOSURE_KINDS:
                return self._read_artifact_ref(normalized, label="Factor-local closure artifact")
            document = _validate_artifact_kind(resolver(normalized), normalized["kind"])
            if _artifact_ref(document) != normalized:
                _raise("live Factor closure artifact resolver returned a different artifact")
            return document

        return resolve

    def _capture_source_resolver(
        self,
        resolver: Callable[[Mapping[str, Any], int], tuple[Mapping[str, Any], bytes]],
    ) -> Callable[[Mapping[str, Any], int], tuple[Mapping[str, Any], bytes]]:
        def capture(ref: Mapping[str, Any], maximum_bytes: int) -> tuple[Mapping[str, Any], bytes]:
            normalized = _validate_ref(
                ref, label="live Factor source ref", expected_kind="system.source_object"
            )
            if (
                type(maximum_bytes) is not int
                or maximum_bytes <= 0
                or maximum_bytes > _MAX_FACTOR_SOURCE_MIRROR_BYTES
            ):
                _raise("Factor source mirror resolver byte bound is invalid")
            descriptor, raw = resolver(normalized, maximum_bytes)
            if (
                type(descriptor) is not dict
                or type(raw) is not bytes
                or not raw
                or len(raw) > maximum_bytes
            ):
                _raise("live Factor source resolver output is invalid")
            if (
                descriptor.get("source_object_ref") != normalized
                or descriptor.get("byte_sha256") != _sha256(raw)
                or type(descriptor.get("stat_identity")) is not dict
            ):
                _raise("live Factor source resolver closure differs")
            metadata_path, raw_path = self._source_mirror_paths(normalized)
            self._storage._write_blob_exact_once(
                raw_path,
                raw,
                maximum_bytes=_MAX_FACTOR_SOURCE_MIRROR_BYTES,
            )
            metadata = {
                "source_object_ref": normalized,
                "source_descriptor": dict(descriptor),
                "source_raw_sha256": _sha256(raw),
                "source_raw_path": str(raw_path),
            }
            self._storage.write_exact_once(metadata_path, canonical_json_bytes(metadata))
            return descriptor, raw

        return capture

    def _mirrored_source_resolver(
        self, ref: Mapping[str, Any], maximum_bytes: int
    ) -> tuple[Mapping[str, Any], bytes]:
        normalized = _validate_ref(
            ref, label="mirrored Factor source ref", expected_kind="system.source_object"
        )
        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or maximum_bytes > _MAX_FACTOR_SOURCE_MIRROR_BYTES
        ):
            _raise("Factor source mirror byte bound is invalid")
        metadata_path, raw_path = self._source_mirror_paths(normalized)
        metadata_stored = self._storage.read(metadata_path)
        try:
            metadata = parse_canonical_json_bytes(
                metadata_stored.data, label="Factor source mirror"
            )
        except ContractError as exc:
            raise FactorGovernanceError("Factor source mirror metadata is invalid") from exc
        if type(metadata) is not dict or set(metadata) != {
            "source_object_ref",
            "source_descriptor",
            "source_raw_sha256",
            "source_raw_path",
        }:
            _raise("Factor source mirror metadata fields differ")
        if (
            metadata["source_object_ref"] != normalized
            or metadata["source_raw_path"] != str(raw_path)
            or type(metadata["source_descriptor"]) is not dict
        ):
            _raise("Factor source mirror binding differs")
        raw_stored = self._storage.read_blob(raw_path, maximum_bytes=maximum_bytes)
        if metadata["source_raw_sha256"] != raw_stored.byte_sha256:
            _raise("Factor source mirror raw SHA differs")
        descriptor = metadata["source_descriptor"]
        if (
            descriptor.get("source_object_ref") != normalized
            or descriptor.get("byte_sha256") != raw_stored.byte_sha256
        ):
            _raise("Factor source mirror descriptor differs")
        return descriptor, raw_stored.data

    def _require_live_resolvers(
        self,
    ) -> tuple[
        Callable[[Mapping[str, Any]], Mapping[str, Any] | bytes],
        Callable[[Mapping[str, Any], int], tuple[Mapping[str, Any], bytes]],
        Path,
    ]:
        if self._source_custody is None or self._release_repository_root is None:
            _raise(
                "Factor production requires read-only canonical source custody "
                "and an exact release repository root"
            )
        return (
            self._source_custody.artifact_resolver,
            self._source_custody.source_resolver,
            self._release_repository_root,
        )

    @staticmethod
    def _deep_validate_factor_closure(
        *,
        factor_generation: Mapping[str, Any] | bytes,
        source_closure: Mapping[str, Any] | bytes,
        recomputation_evidence: Mapping[str, Any] | bytes,
        legacy_zero_call_certificate: Mapping[str, Any] | bytes,
        artifact_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any] | bytes],
        source_resolver: Callable[[Mapping[str, Any], int], tuple[Mapping[str, Any], bytes]],
        validation_mode: str,
        current_release_root: Path | None,
    ) -> None:
        """Invoke the code-owned deep Factor source/replay validators.

        A structural artifact alone is intentionally insufficient.  Older
        source modules without this explicit resolver API are rejected rather
        than silently treated as adequate production evidence.
        """

        try:
            from .governance.production_authority import (
                replay_factor_production_recomputation_evidence,
                replay_factor_production_generation,
                validate_factor_production_generation,
                validate_factor_production_source_closure,
            )

            validate_factor_production_source_closure(
                source_closure,
                artifact_resolver=artifact_resolver,
                source_resolver=source_resolver,
            )
            replay_factor_production_recomputation_evidence(
                recomputation_evidence,
                artifact_resolver=artifact_resolver,
                source_resolver=source_resolver,
                validation_mode=validation_mode,
                current_release_root=current_release_root,
            )
            validate_factor_production_generation(factor_generation)
            replay_factor_production_generation(
                factor_generation,
                artifact_resolver=artifact_resolver,
                source_resolver=source_resolver,
                validation_mode=validation_mode,
                current_release_root=current_release_root,
            )
        except ImportError as exc:
            raise FactorGovernanceError(
                "deep Factor production closure validator is unavailable"
            ) from exc
        except TypeError as exc:
            raise FactorGovernanceError(
                "deep Factor production closure validator does not expose the required resolver API"
            ) from exc

    def prepare_initial_activation(
        self,
        *,
        factor_generation: Mapping[str, Any] | bytes,
        source_closure: Mapping[str, Any] | bytes,
        recomputation_evidence: Mapping[str, Any] | bytes,
        legacy_zero_call_certificate: Mapping[str, Any] | bytes,
        market_input: Mapping[str, Any] | bytes,
        prepared_at: str,
        activated_at: str,
    ) -> dict[str, Any]:
        """Seal exact Factor activation bytes and persist no active pointer or marker."""

        actor_uid = os.geteuid()
        prepared_time = _timestamp(prepared_at, label="prepared_at")
        activated_time = _timestamp(activated_at, label="activated_at")
        if activated_time < prepared_time:
            _raise("Factor activation activated_at precedes prepared_at")
        generation = _validate_factor_generation(factor_generation)
        source = _validate_source_closure(source_closure)
        recomputation = _validate_recomputation_evidence(recomputation_evidence)
        legacy = _validate_legacy_zero_call_certificate(legacy_zero_call_certificate)
        market = _validate_market_input(market_input)
        _cross_bind_factor_generation(generation, source, recomputation, legacy, market)
        # Store the typed Factor roots first so the deep replay can resolve
        # Factor-owned source/recomputation/certificate refs without ever
        # asking the System source custody to discover Factor authority.
        for artifact in (generation, source, recomputation, legacy, market):
            self._publish_artifact(artifact)
        live_artifacts, live_sources, release_root = self._require_live_resolvers()
        self._deep_validate_factor_closure(
            factor_generation=generation,
            source_closure=source,
            recomputation_evidence=recomputation,
            legacy_zero_call_certificate=legacy,
            artifact_resolver=self._capture_artifact_resolver(live_artifacts),
            source_resolver=self._capture_source_resolver(live_sources),
            validation_mode="PRE_CAS_CURRENT",
            current_release_root=release_root,
        )
        receipt = build_factor_production_generation_receipt(
            factor_generation=generation,
            source_closure=source,
            recomputation_evidence=recomputation,
            legacy_zero_call_certificate=legacy,
            market_input=market,
            created_at=prepared_at,
        )
        pointer = _build_factor_pointer(
            receipt=receipt, activated_at=activated_at, actor_uid=actor_uid
        )
        bundle = _build_activation_bundle(
            receipt=receipt, pointer=pointer, prepared_at=prepared_at, actor_uid=actor_uid
        )
        prepared = _build_prepared_transaction(bundle=bundle, receipt=receipt, pointer=pointer)
        marker = _build_marker(receipt=receipt, pointer=pointer, bundle=bundle, prepared=prepared)
        for artifact in (
            generation,
            source,
            recomputation,
            legacy,
            market,
            receipt,
            pointer,
            bundle,
            prepared,
            marker,
        ):
            self._publish_artifact(artifact)
        generation_id = generation["payload"]["factor_production_generation_id"]
        generation_path = FACTOR_GENERATIONS_ROOT / generation_id / "generation.json"
        self.write_exact_once(generation_path, canonical_json_bytes(generation))
        pointer_ref = _artifact_ref(pointer)
        derived_root = FACTOR_PREPARATIONS_ROOT / pointer_ref["byte_sha256"]
        for name, artifact in (
            ("receipt.json", receipt),
            ("pointer.json", pointer),
            ("bundle.json", bundle),
            ("prepared.json", prepared),
            ("marker.json", marker),
        ):
            self.write_exact_once(derived_root / name, canonical_json_bytes(artifact))
        self.write_exact_once(
            FACTOR_ACTIVATION_BUNDLES_ROOT / f"{pointer_ref['byte_sha256']}.json",
            canonical_json_bytes(bundle),
        )
        self.write_exact_once(
            FACTOR_ACTIVATION_TRANSACTIONS_ROOT / f"{pointer_ref['byte_sha256']}.json",
            canonical_json_bytes(prepared),
        )
        if (
            self.read_optional(FACTOR_ACTIVE_POINTER_PATH) is not None
            or self.read_optional(FACTOR_PRODUCTION_MARKER_PATH) is not None
        ):
            _raise(
                "Factor initial activation must be prepared while both authority paths are absent"
            )
        return {
            "activation_scope": FACTOR_PRODUCTION_SCOPE,
            "cas_performed": False,
            "factor_generation": generation,
            "factor_generation_raw": canonical_json_bytes(generation),
            "factor_generation_path": str(generation_path),
            "factor_generation_receipt": receipt,
            "factor_generation_receipt_raw": canonical_json_bytes(receipt),
            "target_factor_pointer": pointer,
            "target_factor_pointer_raw": _factor_pointer_raw(pointer),
            "activation_bundle": bundle,
            "activation_bundle_raw": canonical_json_bytes(bundle),
            "prepared_transaction": prepared,
            "prepared_transaction_raw": canonical_json_bytes(prepared),
            "permanent_marker": marker,
            "permanent_marker_raw": canonical_json_bytes(marker),
            "factor_pointer_path": str(FACTOR_ACTIVE_POINTER_PATH),
            "factor_marker_path": str(FACTOR_PRODUCTION_MARKER_PATH),
        }

    def prepare_rollover_activation(
        self,
        *,
        factor_generation: Mapping[str, Any] | bytes,
        source_closure: Mapping[str, Any] | bytes,
        recomputation_evidence: Mapping[str, Any] | bytes,
        legacy_zero_call_certificate: Mapping[str, Any] | bytes,
        market_input: Mapping[str, Any] | bytes,
        expected_pointer_sha256: str,
        maintenance: Mapping[str, str],
        canonical_inputs: Mapping[str, str],
        prepared_at: str,
        activated_at: str,
    ) -> dict[str, Any]:
        """Seal one successor package without mutating the active pointer."""

        _require_sha(expected_pointer_sha256, label="expected Factor pointer SHA")
        current = self.read(FACTOR_ACTIVE_POINTER_PATH)
        marker = self.read_optional(FACTOR_PRODUCTION_MARKER_PATH)
        if current.byte_sha256 != expected_pointer_sha256 or marker is None:
            _raise("Factor rollover predecessor or genesis marker differs")
        predecessor_generation = self._read_generation_for_pointer(
            current.data, label="Factor rollover predecessor"
        )
        generation = _validate_factor_generation(factor_generation)
        source = _validate_source_closure(source_closure)
        recomputation = _validate_recomputation_evidence(recomputation_evidence)
        legacy = _validate_legacy_zero_call_certificate(legacy_zero_call_certificate)
        market = _validate_market_input(market_input)
        _cross_bind_factor_generation(generation, source, recomputation, legacy, market)
        target_date = generation["payload"]["as_of"]
        predecessor_date = predecessor_generation["payload"]["as_of"]
        if target_date < predecessor_date:
            _raise("Factor rollover target precedes active generation")
        if target_date != maintenance.get("target_date"):
            _raise("Factor rollover generation date differs from maintenance")
        actor_uid = os.geteuid()
        _timestamp(prepared_at, label="prepared_at")
        _timestamp(activated_at, label="activated_at")
        for artifact in (generation, source, recomputation, legacy, market):
            self._publish_artifact(artifact)
        live_artifacts, live_sources, release_root = self._require_live_resolvers()
        self._deep_validate_factor_closure(
            factor_generation=generation,
            source_closure=source,
            recomputation_evidence=recomputation,
            legacy_zero_call_certificate=legacy,
            artifact_resolver=self._capture_artifact_resolver(live_artifacts),
            source_resolver=self._capture_source_resolver(live_sources),
            validation_mode="PRE_CAS_CURRENT",
            current_release_root=release_root,
        )
        receipt = build_factor_production_generation_receipt(
            factor_generation=generation,
            source_closure=source,
            recomputation_evidence=recomputation,
            legacy_zero_call_certificate=legacy,
            market_input=market,
            created_at=prepared_at,
        )
        predecessor_pointer = _factor_pointer_record_from_raw(current.data)
        target_pointer = _build_factor_pointer(
            receipt=receipt,
            activated_at=activated_at,
            actor_uid=actor_uid,
            previous_pointer_sha256=expected_pointer_sha256,
        )
        bundle = _build_rollover_bundle(
            predecessor_pointer=predecessor_pointer,
            target_pointer=target_pointer,
            receipt=receipt,
            maintenance=maintenance,
            canonical_inputs=canonical_inputs,
            target_date=target_date,
            prepared_at=prepared_at,
            actor_uid=actor_uid,
        )
        prepared = _build_rollover_prepared(
            bundle=bundle, prepared_at=prepared_at, actor_uid=actor_uid
        )
        commit = _build_rollover_commit(
            bundle=bundle,
            prepared=prepared,
            committed_at=activated_at,
            actor_uid=actor_uid,
        )
        for artifact in (
            generation,
            source,
            recomputation,
            legacy,
            market,
            receipt,
            predecessor_pointer,
            target_pointer,
            bundle,
            prepared,
            commit,
        ):
            self._publish_artifact(artifact)
        generation_path = (
            FACTOR_GENERATIONS_ROOT
            / generation["payload"]["factor_production_generation_id"]
            / "generation.json"
        )
        self.write_exact_once(generation_path, canonical_json_bytes(generation))
        target_raw = _factor_pointer_raw(target_pointer)
        target_sha = _sha256(target_raw)
        derived_root = FACTOR_PREPARATIONS_ROOT / target_sha
        for name, artifact in (
            ("receipt.json", receipt),
            ("pointer.json", target_pointer),
            ("rollover-bundle.json", bundle),
            ("rollover-prepared.json", prepared),
            ("rollover-commit.json", commit),
        ):
            self.write_exact_once(derived_root / name, canonical_json_bytes(artifact))
        canonical_paths_document = {
            key: canonical_inputs[key]
            for key in (
                "market_pointer_path",
                "market_manifest_path",
                "pit_pointer_path",
                "pit_manifest_path",
            )
        }
        self.write_exact_once(
            derived_root / "canonical-paths.json",
            canonical_json_bytes(canonical_paths_document),
        )
        self.write_exact_once(
            FACTOR_ROLLOVER_BUNDLES_ROOT / f"{target_sha}.json", canonical_json_bytes(bundle)
        )
        self.write_exact_once(
            FACTOR_ROLLOVER_TRANSACTIONS_ROOT / f"{target_sha}.json",
            canonical_json_bytes(prepared),
        )
        input_key = _rollover_input_key(
            previous_pointer_sha256=expected_pointer_sha256,
            maintenance_sha256=maintenance["receipt_sha256"],
        )
        self.write_exact_once(
            FACTOR_ROLLOVER_INPUT_INDEX_ROOT / f"{input_key}.json",
            canonical_json_bytes(
                {
                    "schema_version": "factor-production-rollover-input-index.v1",
                    "previous_pointer_sha256": expected_pointer_sha256,
                    "maintenance_sha256": maintenance["receipt_sha256"],
                    "target_pointer_sha256": target_sha,
                }
            ),
        )
        return {
            "target_factor_pointer_raw": target_raw,
            "previous_factor_pointer_raw": current.data,
            "factor_generation_receipt_raw": canonical_json_bytes(receipt),
            "rollover_bundle_raw": canonical_json_bytes(bundle),
            "rollover_prepared_raw": canonical_json_bytes(prepared),
            "rollover_commit_raw": canonical_json_bytes(commit),
            "target_pointer_sha256": target_sha,
            "previous_pointer_sha256": expected_pointer_sha256,
            "target_date": target_date,
            "canonical_paths": canonical_paths_document,
        }

    def _validate_stored_activation_package(  # noqa: C901 - atomic pre-CAS closure
        self,
        *,
        target_factor_pointer_raw: bytes,
        factor_generation_receipt_raw: bytes,
        activation_bundle_raw: bytes,
        prepared_transaction_raw: bytes,
        permanent_marker_raw: bytes,
        validation_mode: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        active_pointer = validate_factor_active_pointer(target_factor_pointer_raw)
        receipt = validate_factor_production_generation_receipt(factor_generation_receipt_raw)
        bundle = validate_factor_production_activation_bundle(activation_bundle_raw)
        pointer = self._read_artifact_ref(
            bundle["payload"]["target_factor_pointer_ref"], label="Factor sealed pointer record"
        )
        pointer = validate_factor_production_pointer(pointer)
        if _factor_pointer_raw(pointer) != target_factor_pointer_raw:
            _raise("Factor active pointer differs from sealed pointer record")
        bundle = validate_factor_production_activation_bundle(
            bundle, receipt=receipt, pointer=pointer
        )
        prepared = validate_factor_production_prepared(
            prepared_transaction_raw, bundle=bundle, receipt=receipt, pointer=pointer
        )
        marker = validate_factor_production_marker(
            permanent_marker_raw,
            receipt=receipt,
            pointer=pointer,
            bundle=bundle,
            prepared=prepared,
        )
        for supplied, document, label in (
            (factor_generation_receipt_raw, receipt, "Factor receipt"),
            (activation_bundle_raw, bundle, "Factor activation bundle"),
            (prepared_transaction_raw, prepared, "Factor prepared transaction"),
            (permanent_marker_raw, marker, "Factor permanent marker"),
        ):
            if canonical_json_bytes(document) != supplied:
                _raise(f"{label} must preserve exact canonical bytes")
            stored = self._read_artifact_ref(_artifact_ref(document), label=label)
            if canonical_json_bytes(stored) != supplied:
                _raise(f"{label} object-store bytes differ")
        if _factor_pointer_raw(pointer) != target_factor_pointer_raw:
            _raise("Factor sealed pointer exact bytes differ")
        generation = self._read_factor_generation_ref(
            receipt["payload"]["factor_generation_ref"], label="Factor production generation"
        )
        source = self._read_artifact_ref(
            receipt["payload"]["source_closure_ref"], label="Factor source closure"
        )
        recomputation = self._read_artifact_ref(
            receipt["payload"]["recomputation_evidence_ref"], label="Factor recomputation evidence"
        )
        legacy = self._read_artifact_ref(
            receipt["payload"]["legacy_zero_call_ref"], label="Factor legacy-zero-call certificate"
        )
        market = self._read_artifact_ref(
            receipt["payload"]["market_input_ref"], label="Factor Market input"
        )
        validate_factor_production_generation_receipt(
            receipt,
            factor_generation=generation,
            source_closure=source,
            recomputation_evidence=recomputation,
            legacy_zero_call_certificate=legacy,
            market_input=market,
        )
        if validation_mode not in {"STRUCTURAL", "PRE_CAS_CURRENT", "HISTORICAL"}:
            _raise("Factor activation validation mode is invalid")
        if validation_mode != "STRUCTURAL":
            # A fresh CAS must prove the live immutable custody still matches.
            # Once an exact pointer exists, only the sealed historical mirror
            # may drive marker recovery or idempotent replay; a later checkout
            # advance cannot authorize a second CAS or strand the first one.
            if validation_mode == "PRE_CAS_CURRENT":
                live_artifacts, live_sources, release_root = self._require_live_resolvers()
                self._deep_validate_factor_closure(
                    factor_generation=generation,
                    source_closure=source,
                    recomputation_evidence=recomputation,
                    legacy_zero_call_certificate=legacy,
                    artifact_resolver=self._live_or_local_artifact_resolver(live_artifacts),
                    source_resolver=live_sources,
                    validation_mode="PRE_CAS_CURRENT",
                    current_release_root=release_root,
                )
            self._deep_validate_factor_closure(
                factor_generation=generation,
                source_closure=source,
                recomputation_evidence=recomputation,
                legacy_zero_call_certificate=legacy,
                artifact_resolver=lambda ref: self._read_artifact_ref(
                    ref, label="mirrored Factor closure artifact"
                ),
                source_resolver=self._mirrored_source_resolver,
                validation_mode="HISTORICAL_RECOVERY",
                current_release_root=None,
            )
        if active_pointer["os_actor"] != f"uid:{os.geteuid()}":
            _raise("Factor activation EUID differs from prepared exact pointer")
        if _timestamp(active_pointer["activated_at"], label="pointer activated_at") > datetime.now(
            timezone.utc
        ):
            _raise("Factor activation exact pointer time is in the future")
        return active_pointer, pointer, receipt, bundle, prepared, marker

    def activate_initial_generation(
        self,
        *,
        target_factor_pointer_raw: bytes,
        factor_generation_receipt_raw: bytes,
        activation_bundle_raw: bytes,
        prepared_transaction_raw: bytes,
        permanent_marker_raw: bytes,
    ) -> dict[str, Any]:
        """Perform the one Factor ``EMPTY`` CAS, or exact marker-only recovery.

        This function never synthesizes actor, time, pointer, prepared, or
        marker fields.  It consumes precisely the canonical bytes created by
        :meth:`prepare_initial_activation`.
        """

        package = self._validate_stored_activation_package(
            target_factor_pointer_raw=target_factor_pointer_raw,
            factor_generation_receipt_raw=factor_generation_receipt_raw,
            activation_bundle_raw=activation_bundle_raw,
            prepared_transaction_raw=prepared_transaction_raw,
            permanent_marker_raw=permanent_marker_raw,
            validation_mode="STRUCTURAL",
        )
        active_pointer, pointer, receipt, bundle, prepared, marker = package
        del active_pointer, receipt, bundle, prepared
        cas_performed = False
        marker_published = False
        with self._active_lock():
            current = self.read_optional(FACTOR_ACTIVE_POINTER_PATH)
            existing_marker = self.read_optional(FACTOR_PRODUCTION_MARKER_PATH)
            if current is None:
                if existing_marker is not None:
                    _raise("Factor permanent marker exists without an active pointer")
                # Re-read exact immutable inputs and live source custody under
                # the active lock immediately before the sole EMPTY CAS.
                self._validate_stored_activation_package(
                    target_factor_pointer_raw=target_factor_pointer_raw,
                    factor_generation_receipt_raw=factor_generation_receipt_raw,
                    activation_bundle_raw=activation_bundle_raw,
                    prepared_transaction_raw=prepared_transaction_raw,
                    permanent_marker_raw=permanent_marker_raw,
                    validation_mode="PRE_CAS_CURRENT",
                )
                current = self._write_initial_pointer_under_lock(target_factor_pointer_raw)
                cas_performed = True
            elif current.data != target_factor_pointer_raw:
                _raise("Factor initial activation preimage changed or pointer differs")
            else:
                # Pointer-only crash recovery and an exact replay validate the
                # sealed historical closure and never rerun a fresh CAS gate.
                self._validate_stored_activation_package(
                    target_factor_pointer_raw=target_factor_pointer_raw,
                    factor_generation_receipt_raw=factor_generation_receipt_raw,
                    activation_bundle_raw=activation_bundle_raw,
                    prepared_transaction_raw=prepared_transaction_raw,
                    permanent_marker_raw=permanent_marker_raw,
                    validation_mode="HISTORICAL",
                )
            if existing_marker is not None and existing_marker.data != permanent_marker_raw:
                _raise("Factor permanent marker differs from exact prepared marker")
            if existing_marker is None:
                stored_marker = self._write_permanent_marker_under_lock(permanent_marker_raw)
                marker_published = True
            else:
                stored_marker = existing_marker
            if (
                current.data != target_factor_pointer_raw
                or stored_marker.data != permanent_marker_raw
            ):
                _raise("Factor activation exact-byte readback differs")
        verification = self.verify_active()
        return {
            **verification,
            "activation": {
                "cas_performed": cas_performed,
                "factor_pointer_byte_sha256": _sha256(target_factor_pointer_raw),
                "factor_pointer_semantic_sha256": pointer["semantic_sha256"],
                "marker_byte_sha256": _sha256(permanent_marker_raw),
                "marker_semantic_sha256": marker["semantic_sha256"],
                "idempotent_replay": (not cas_performed and not marker_published),
                "marker_only_recovery": (not cas_performed and marker_published),
            },
        }

    def recover_initial_marker_from_active_pointer(self) -> dict[str, Any]:
        """Recover only the exact marker authorized by an existing initial pointer."""

        pointer_stored = self.read_optional(FACTOR_ACTIVE_POINTER_PATH)
        marker_stored = self.read_optional(FACTOR_PRODUCTION_MARKER_PATH)
        if pointer_stored is None or marker_stored is not None:
            _raise("Factor marker recovery requires exact pointer-only state")
        pointer_record = _factor_pointer_record_from_raw(pointer_stored.data)
        pointer_ref = _artifact_ref(pointer_record)
        preparation_root = FACTOR_PREPARATIONS_ROOT / pointer_ref["byte_sha256"]
        stored_pointer_record = self.read(preparation_root / "pointer.json")
        if stored_pointer_record.data != canonical_json_bytes(pointer_record):
            _raise("Factor pointer-only preparation record differs")
        receipt = self.read(preparation_root / "receipt.json")
        bundle = self.read(preparation_root / "bundle.json")
        prepared = self.read(preparation_root / "prepared.json")
        marker = self.read(preparation_root / "marker.json")
        recovered = self.activate_initial_generation(
            target_factor_pointer_raw=pointer_stored.data,
            factor_generation_receipt_raw=receipt.data,
            activation_bundle_raw=bundle.data,
            prepared_transaction_raw=prepared.data,
            permanent_marker_raw=marker.data,
        )
        activation = recovered.get("activation")
        if (
            type(activation) is not dict
            or activation.get("cas_performed") is not False
            or activation.get("marker_only_recovery") is not True
        ):
            _raise("Factor pointer-only recovery attempted a non-marker transition")
        return recovered

    def activate_rollover_generation(
        self,
        *,
        target_factor_pointer_raw: bytes,
        previous_factor_pointer_raw: bytes,
        factor_generation_receipt_raw: bytes,
        rollover_bundle_raw: bytes,
        rollover_prepared_raw: bytes,
        rollover_commit_raw: bytes,
        canonical_paths: Mapping[str, str],
    ) -> dict[str, Any]:
        """Perform or recover one forward-only cooperative exact-preimage rollover."""

        predecessor = _factor_pointer_record_from_raw(previous_factor_pointer_raw)
        target = _factor_pointer_record_from_raw(target_factor_pointer_raw)
        receipt = validate_factor_production_generation_receipt(factor_generation_receipt_raw)
        bundle = validate_factor_production_rollover_bundle(
            rollover_bundle_raw,
            predecessor_pointer=predecessor,
            target_pointer=target,
            receipt=receipt,
        )
        prepared = validate_factor_production_rollover_prepared(
            rollover_prepared_raw, bundle=bundle
        )
        commit = validate_factor_production_rollover_commit(
            rollover_commit_raw, bundle=bundle, prepared=prepared
        )
        bundle_payload = bundle["payload"]
        expected_sha = bundle_payload["previous_pointer_sha256"]
        target_sha = bundle_payload["target_pointer_sha256"]
        if (
            _sha256(previous_factor_pointer_raw) != expected_sha
            or _sha256(target_factor_pointer_raw) != target_sha
        ):
            _raise("Factor rollover pointer package SHA differs")
        expected_paths = {
            "market_pointer_path",
            "market_manifest_path",
            "pit_pointer_path",
            "pit_manifest_path",
        }
        if set(canonical_paths) != expected_paths:
            _raise("Factor rollover canonical path set differs")
        cas_performed = False
        recovered_commit = False
        with self._active_lock():
            marker = self.read_optional(FACTOR_PRODUCTION_MARKER_PATH)
            if marker is None:
                _raise("Factor rollover requires the immutable genesis marker")
            current = self.read(FACTOR_ACTIVE_POINTER_PATH)
            if current.byte_sha256 == expected_sha:
                if current.data != previous_factor_pointer_raw:
                    _raise("Factor rollover predecessor bytes differ")
                for path_key, sha_key in (
                    ("market_pointer_path", "market_pointer_sha256"),
                    ("market_manifest_path", "market_manifest_sha256"),
                    ("pit_pointer_path", "pit_pointer_sha256"),
                    ("pit_manifest_path", "pit_manifest_sha256"),
                ):
                    if (
                        _stable_workspace_file_sha256(
                            self.workspace_root,
                            canonical_paths[path_key],
                            label=f"Factor rollover {path_key}",
                        )
                        != bundle_payload[sha_key]
                    ):
                        _raise("Factor rollover canonical input changed before CAS")
                history_path = FACTOR_POINTER_HISTORY_ROOT / f"{expected_sha}.json"
                self.write_exact_once(history_path, previous_factor_pointer_raw)
                self._storage._fault_hook("AFTER_ROLLOVER_POINTER_HISTORY")
                replaced = self._replace_active_pointer_under_lock(
                    target_factor_pointer_raw,
                    expected_pointer_sha256=expected_sha,
                )
                if replaced.byte_sha256 != target_sha:
                    _raise("Factor rollover active pointer SHA differs")
                cas_performed = True
            elif current.byte_sha256 == target_sha and current.data == target_factor_pointer_raw:
                history = self.read(FACTOR_POINTER_HISTORY_ROOT / f"{expected_sha}.json")
                if history.data != previous_factor_pointer_raw:
                    _raise("Factor rollover predecessor history differs")
                recovered_commit = (
                    self.read_optional(FACTOR_ROLLOVER_COMMITS_ROOT / f"{target_sha}.json") is None
                )
            else:
                _raise("Factor rollover active pointer is neither predecessor nor target")
            stored_commit = self.write_exact_once(
                FACTOR_ROLLOVER_COMMITS_ROOT / f"{target_sha}.json",
                canonical_json_bytes(commit),
            )
            if stored_commit.data != rollover_commit_raw:
                _raise("Factor rollover commit readback differs")
        verification = self.verify_active()
        return {
            **verification,
            "rollover": {
                "cas_performed": cas_performed,
                "marker_only_recovery": False,
                "commit_recovered": (not cas_performed and recovered_commit),
                "idempotent_replay": (not cas_performed and not recovered_commit),
                "previous_pointer_sha256": expected_sha,
                "target_pointer_sha256": target_sha,
                "rollover_commit_ref": _artifact_ref(commit),
            },
        }

    def recover_rollover_for_inputs(
        self, *, expected_pointer_sha256: str, maintenance_sha256: str
    ) -> dict[str, Any]:
        """Resolve one exact prepared rollover by its immutable input index."""

        input_key = _rollover_input_key(
            previous_pointer_sha256=expected_pointer_sha256,
            maintenance_sha256=maintenance_sha256,
        )
        stored_index = self.read(FACTOR_ROLLOVER_INPUT_INDEX_ROOT / f"{input_key}.json")
        try:
            index = parse_canonical_json_bytes(
                stored_index.data, label="Factor rollover input index"
            )
        except ContractError as exc:
            raise FactorGovernanceError("Factor rollover input index is invalid") from exc
        if (
            type(index) is not dict
            or set(index)
            != {
                "schema_version",
                "previous_pointer_sha256",
                "maintenance_sha256",
                "target_pointer_sha256",
            }
            or index["schema_version"] != "factor-production-rollover-input-index.v1"
            or index["previous_pointer_sha256"] != expected_pointer_sha256
            or index["maintenance_sha256"] != maintenance_sha256
        ):
            _raise("Factor rollover input index fields differ")
        target_sha = _require_sha(
            index["target_pointer_sha256"], label="rollover indexed target pointer"
        )
        preparation_root = FACTOR_PREPARATIONS_ROOT / target_sha
        target_pointer = self.read(preparation_root / "pointer.json")
        target_raw = _factor_pointer_raw(validate_factor_production_pointer(target_pointer.data))
        if _sha256(target_raw) != target_sha:
            _raise("Factor rollover indexed target bytes differ")
        current = self.read(FACTOR_ACTIVE_POINTER_PATH)
        if current.byte_sha256 == expected_pointer_sha256:
            previous_raw = current.data
        elif current.byte_sha256 == target_sha:
            previous_raw = self.read(
                FACTOR_POINTER_HISTORY_ROOT / f"{expected_pointer_sha256}.json"
            ).data
        else:
            _raise("Factor rollover indexed recovery pointer conflicts")
        try:
            canonical_paths = parse_canonical_json_bytes(
                self.read(preparation_root / "canonical-paths.json").data,
                label="Factor rollover canonical paths",
            )
        except ContractError as exc:
            raise FactorGovernanceError("Factor rollover canonical paths are invalid") from exc
        if type(canonical_paths) is not dict:
            _raise("Factor rollover canonical paths fields differ")
        return self.activate_rollover_generation(
            target_factor_pointer_raw=target_raw,
            previous_factor_pointer_raw=previous_raw,
            factor_generation_receipt_raw=self.read(preparation_root / "receipt.json").data,
            rollover_bundle_raw=self.read(preparation_root / "rollover-bundle.json").data,
            rollover_prepared_raw=self.read(preparation_root / "rollover-prepared.json").data,
            rollover_commit_raw=self.read(preparation_root / "rollover-commit.json").data,
            canonical_paths={str(key): str(value) for key, value in canonical_paths.items()},
        )

    def verify_active(self) -> dict[str, Any]:
        """Revalidate the current Factor head and its immutable genesis lineage."""

        pointer_stored = self.read_optional(FACTOR_ACTIVE_POINTER_PATH)
        marker_stored = self.read_optional(FACTOR_PRODUCTION_MARKER_PATH)
        if pointer_stored is None and marker_stored is None:
            return {
                "activation_scope": FACTOR_PRODUCTION_SCOPE,
                "factor_readiness": "BLOCKED",
                "factor_authority": "INACTIVE",
                "order_authority": "NONE",
                "trade_authority": "NONE",
                "funds_transfer_authority": "NONE",
                "blockers": ["FACTOR_ACTIVE_POINTER_ABSENT"],
                "system_pointer_touched": False,
            }
        if pointer_stored is None:
            _raise("Factor permanent marker exists without active pointer")
        if marker_stored is None:
            pending_pointer = validate_factor_active_pointer(pointer_stored.data)
            return {
                "activation_scope": FACTOR_PRODUCTION_SCOPE,
                "factor_readiness": FACTOR_READINESS_READY,
                "factor_authority": "BLOCKED",
                "order_authority": "NONE",
                "trade_authority": "NONE",
                "funds_transfer_authority": "NONE",
                "blockers": ["FACTOR_PRODUCTION_MARKER_ABSENT"],
                "factor_generation_id": pending_pointer["factor_generation_id"],
                "factor_generation_sha256": pending_pointer["factor_generation_sha256"],
                "factor_pointer_byte_sha256": pointer_stored.byte_sha256,
                "system_pointer_touched": False,
            }
        active_pointer = validate_factor_active_pointer(pointer_stored.data)
        marker = validate_factor_production_marker(marker_stored.data)
        genesis_pointer = self._read_artifact_ref(
            marker["payload"]["factor_pointer_ref"], label="Factor marker sealed pointer record"
        )
        genesis_pointer = validate_factor_production_pointer(genesis_pointer)
        genesis_raw = _factor_pointer_raw(genesis_pointer)
        genesis_sha = _sha256(genesis_raw)
        if marker["payload"]["factor_pointer_ref"] != _artifact_ref(genesis_pointer):
            _raise("Factor marker pointer binding differs")
        genesis_bundle = self._read_artifact_ref(
            marker["payload"]["activation_bundle_ref"], label="Factor marker activation bundle"
        )
        genesis_prepared = self._read_artifact_ref(
            marker["payload"]["prepared_transaction_ref"],
            label="Factor marker prepared transaction",
        )
        genesis_receipt = self._read_artifact_ref(
            marker["payload"]["factor_generation_receipt_ref"], label="Factor marker receipt"
        )
        validate_factor_production_activation_bundle(
            genesis_bundle, receipt=genesis_receipt, pointer=genesis_pointer
        )
        validate_factor_production_prepared(
            genesis_prepared,
            bundle=genesis_bundle,
            receipt=genesis_receipt,
            pointer=genesis_pointer,
        )
        validate_factor_production_marker(
            marker,
            receipt=genesis_receipt,
            pointer=genesis_pointer,
            bundle=genesis_bundle,
            prepared=genesis_prepared,
        )

        chain_length = 0
        seen: set[str] = set()
        current_raw = pointer_stored.data
        current_date: str | None = None
        while _sha256(current_raw) != genesis_sha:
            chain_length += 1
            if chain_length > FACTOR_POINTER_CHAIN_MAX:
                _raise("FACTOR_POINTER_CHAIN_LIMIT_REACHED")
            current_sha = _sha256(current_raw)
            if current_sha in seen:
                _raise("Factor pointer lineage contains a cycle")
            seen.add(current_sha)
            current_pointer = validate_factor_active_pointer(current_raw)
            previous_sha = current_pointer["previous_pointer_sha256"]
            if previous_sha == FACTOR_EMPTY_POINTER_SHA256:
                _raise("Factor pointer lineage is detached from genesis")
            preparation_root = FACTOR_PREPARATIONS_ROOT / current_sha
            pointer_record = validate_factor_production_pointer(
                self.read(preparation_root / "pointer.json").data
            )
            receipt_record = validate_factor_production_generation_receipt(
                self.read(preparation_root / "receipt.json").data
            )
            bundle_record = validate_factor_production_rollover_bundle(
                self.read(preparation_root / "rollover-bundle.json").data
            )
            prepared_record = validate_factor_production_rollover_prepared(
                self.read(preparation_root / "rollover-prepared.json").data,
                bundle=bundle_record,
            )
            commit_record = validate_factor_production_rollover_commit(
                self.read(FACTOR_ROLLOVER_COMMITS_ROOT / f"{current_sha}.json").data,
                bundle=bundle_record,
                prepared=prepared_record,
            )
            del commit_record
            predecessor_raw = self.read(FACTOR_POINTER_HISTORY_ROOT / f"{previous_sha}.json").data
            predecessor_record = _factor_pointer_record_from_raw(predecessor_raw)
            validate_factor_production_rollover_bundle(
                bundle_record,
                predecessor_pointer=predecessor_record,
                target_pointer=pointer_record,
                receipt=receipt_record,
            )
            if _factor_pointer_raw(pointer_record) != current_raw:
                _raise("Factor rollover pointer record differs from lineage head")
            generation_record = self._read_factor_generation_ref(
                receipt_record["payload"]["factor_generation_ref"],
                label="Factor rollover generation",
            )
            generation_date = generation_record["payload"]["as_of"]
            if current_date is not None and generation_date > current_date:
                _raise("Factor pointer lineage dates regress")
            current_date = generation_date
            current_raw = predecessor_raw
        if current_raw != genesis_raw:
            _raise("Factor pointer lineage genesis bytes differ")

        current_pointer_record = _factor_pointer_record_from_raw(pointer_stored.data)
        if pointer_stored.byte_sha256 == genesis_sha:
            receipt = genesis_receipt
            pointer = genesis_pointer
        else:
            preparation_root = FACTOR_PREPARATIONS_ROOT / pointer_stored.byte_sha256
            receipt = validate_factor_production_generation_receipt(
                self.read(preparation_root / "receipt.json").data
            )
            pointer = validate_factor_production_pointer(
                self.read(preparation_root / "pointer.json").data
            )
        if _artifact_ref(pointer) != _artifact_ref(current_pointer_record):
            _raise("Factor current pointer artifact differs")
        generation = self._read_factor_generation_ref(
            receipt["payload"]["factor_generation_ref"], label="Factor production generation"
        )
        source = self._read_artifact_ref(
            receipt["payload"]["source_closure_ref"], label="Factor source closure"
        )
        recomputation = self._read_artifact_ref(
            receipt["payload"]["recomputation_evidence_ref"], label="Factor recomputation evidence"
        )
        legacy = self._read_artifact_ref(
            receipt["payload"]["legacy_zero_call_ref"], label="Factor legacy-zero-call certificate"
        )
        market = self._read_artifact_ref(
            receipt["payload"]["market_input_ref"], label="Factor Market input"
        )
        validate_factor_production_generation_receipt(
            receipt,
            factor_generation=generation,
            source_closure=source,
            recomputation_evidence=recomputation,
            legacy_zero_call_certificate=legacy,
            market_input=market,
        )
        self._deep_validate_factor_closure(
            factor_generation=generation,
            source_closure=source,
            recomputation_evidence=recomputation,
            legacy_zero_call_certificate=legacy,
            artifact_resolver=lambda ref: self._read_artifact_ref(
                ref, label="mirrored Factor closure artifact"
            ),
            source_resolver=self._mirrored_source_resolver,
            validation_mode="HISTORICAL_RECOVERY",
            current_release_root=None,
        )
        return {
            "activation_scope": FACTOR_PRODUCTION_SCOPE,
            "factor_readiness": FACTOR_READINESS_READY,
            "factor_authority": FACTOR_AUTHORITY_ACTIVE,
            "factor_generation_id": active_pointer["factor_generation_id"],
            "factor_generation_sha256": active_pointer["factor_generation_sha256"],
            "as_of": generation["payload"]["as_of"],
            "factor_pointer_byte_sha256": pointer_stored.byte_sha256,
            "factor_pointer_semantic_sha256": pointer["semantic_sha256"],
            "marker_byte_sha256": marker_stored.byte_sha256,
            "marker_semantic_sha256": marker["semantic_sha256"],
            "active_factors": receipt["payload"]["active_factor_rows"],
            "control_factors": receipt["payload"]["control_rows"],
            "pointer_chain_length": chain_length,
            "genesis_pointer_sha256": genesis_sha,
            "admission_route": ADMISSION_ROUTE,
            "producer_identity": PRODUCER_IDENTITY,
            "fundamental_dependency_state": FUNDAMENTAL_NOT_USED,
            "fundamental_freshness_policy": FUNDAMENTAL_ADVISORY,
            "system_authority": "NONE",
            "mainline_authority": "NONE",
            "investment_authority": "NONE",
            "portfolio_authority": "NONE",
            "strategy_record_authority": "NONE",
            "broker_authority": "NONE",
            "order_authority": "NONE",
            "trade_authority": "NONE",
            "funds_transfer_authority": "NONE",
            "system_pointer_touched": False,
            "blockers": [],
        }

    def read_active_signal(self, factor_id: str) -> dict[str, Any]:
        """Read one sealed active signal without consulting current sources."""

        if factor_id not in {LOW_DOLLAR_VOLUME, BLEND_W80}:
            _raise("Factor production signal reader accepts only active LOW or W80")
        verification = self.verify_active()
        if verification.get("factor_authority") != FACTOR_AUTHORITY_ACTIVE:
            _raise("Factor production signal read requires a complete active marker closure")
        pointer_stored = self._storage.read(FACTOR_ACTIVE_POINTER_PATH)
        marker_stored = self._storage.read(FACTOR_PRODUCTION_MARKER_PATH)
        if (
            pointer_stored.byte_sha256 != verification["factor_pointer_byte_sha256"]
            or marker_stored.byte_sha256 != verification["marker_byte_sha256"]
        ):
            _raise("Factor production authority changed during signal read")
        marker = validate_factor_production_marker(marker_stored.data)
        if pointer_stored.byte_sha256 == verification["genesis_pointer_sha256"]:
            receipt = self._read_artifact_ref(
                marker["payload"]["factor_generation_receipt_ref"],
                label="Factor signal receipt",
            )
        else:
            receipt = validate_factor_production_generation_receipt(
                self.read(
                    FACTOR_PREPARATIONS_ROOT / pointer_stored.byte_sha256 / "receipt.json"
                ).data
            )
        generation = self._read_factor_generation_ref(
            receipt["payload"]["factor_generation_ref"],
            label="Factor signal generation",
        )
        payload = generation["payload"]
        if (
            payload["factor_production_generation_id"] != verification["factor_generation_id"]
            or _artifact_ref(generation)["byte_sha256"]
            != validate_factor_active_pointer(pointer_stored.data)["factor_generation_sha256"]
        ):
            _raise("Factor signal generation differs from active pointer")
        values = payload["signal_values"][factor_id]
        statistic = next(
            row for row in payload["signal_statistics"] if row["factor_id"] == factor_id
        )
        signal_sha = (
            payload["low_signal_sha256"]
            if factor_id == LOW_DOLLAR_VOLUME
            else payload["w80_signal_sha256"]
        )
        return {
            "activation_scope": FACTOR_PRODUCTION_SCOPE,
            "factor_authority": FACTOR_AUTHORITY_ACTIVE,
            "factor_generation_id": payload["factor_production_generation_id"],
            "factor_generation_sha256": _artifact_ref(generation)["byte_sha256"],
            "as_of": payload["as_of"],
            "factor_id": factor_id,
            "signal_sha256": signal_sha,
            "signal_symbol_set_sha256": statistic["signal_symbol_set_sha256"],
            "symbol_count": len(values),
            "signal_values": {
                symbol: values[symbol]
                for symbol in sorted(values, key=lambda value: value.encode("utf-8"))
            },
            "system_authority": "NONE",
            "mainline_authority": "NONE",
            "investment_authority": "NONE",
            "portfolio_authority": "NONE",
            "broker_authority": "NONE",
            "order_authority": "NONE",
            "trade_authority": "NONE",
        }

    def read_active_observation_inputs(self) -> dict[str, Any]:
        """Read one verified active head for non-authorizing daily observation.

        The caller must hold ``_active_lock`` across this read and publication.
        This returns both active signals after one deep replay, avoiding two
        independent verification windows for the same observation batch.
        """

        verification = self.verify_active()
        if verification.get("factor_authority") != FACTOR_AUTHORITY_ACTIVE:
            _raise("Factor production observation requires complete active authority")
        pointer_stored = self.read(FACTOR_ACTIVE_POINTER_PATH)
        marker_stored = self.read(FACTOR_PRODUCTION_MARKER_PATH)
        if (
            pointer_stored.byte_sha256 != verification["factor_pointer_byte_sha256"]
            or marker_stored.byte_sha256 != verification["marker_byte_sha256"]
        ):
            _raise("Factor production authority changed during observation read")
        generation = self._read_generation_for_pointer(
            pointer_stored.data, label="Factor production observation"
        )
        payload = generation["payload"]
        if payload["as_of"] != verification["as_of"]:
            _raise("Factor production observation date differs from verified head")
        if pointer_stored.byte_sha256 == verification["genesis_pointer_sha256"]:
            _raise("Factor production genesis observation lacks exact PIT pointer binding")
        else:
            bundle = validate_factor_production_rollover_bundle(
                self.read(
                    FACTOR_PREPARATIONS_ROOT / pointer_stored.byte_sha256 / "rollover-bundle.json"
                ).data
            )["payload"]
            if (
                bundle["target_pointer_sha256"] != pointer_stored.byte_sha256
                or bundle["target_date"] != payload["as_of"]
            ):
                _raise("Factor observation rollover binding differs")
            market_pointer_sha256 = bundle["market_pointer_sha256"]
            market_manifest_sha256 = bundle["market_manifest_sha256"]
            pit_pointer_sha256 = bundle["pit_pointer_sha256"]
            pit_manifest_sha256 = bundle["pit_manifest_sha256"]
        statistics = {row["factor_id"]: row for row in payload["signal_statistics"]}
        factor_rows = []
        for factor_id, alias, signal_field in (
            (LOW_DOLLAR_VOLUME, "LOW", "low_signal_sha256"),
            (BLEND_W80, "W80", "w80_signal_sha256"),
        ):
            statistic = statistics[factor_id]
            factor_rows.append(
                {
                    "factor_id": factor_id,
                    "factor_alias": alias,
                    "signal_sha256": payload[signal_field],
                    "signal_symbol_set_sha256": statistic["signal_symbol_set_sha256"],
                    "symbol_count": len(payload["signal_values"][factor_id]),
                }
            )
        return {
            "signal_date": payload["as_of"],
            "factor_generation_id": payload["factor_production_generation_id"],
            "factor_generation_sha256": verification["factor_generation_sha256"],
            "factor_pointer_sha256": pointer_stored.byte_sha256,
            "market_pointer_sha256": market_pointer_sha256,
            "market_manifest_sha256": market_manifest_sha256,
            "pit_pointer_sha256": pit_pointer_sha256,
            "pit_manifest_sha256": pit_manifest_sha256,
            "pit_membership_sha256": self._read_artifact_ref(
                payload["market_input_ref"], label="Factor observation PIT input"
            )["payload"]["pit_membership_sha256"],
            "calendar_compilation_ref": payload["calendar_compilation_ref"],
            "calendar_capture_custody_attestation_ref": payload[
                "calendar_capture_custody_attestation_ref"
            ],
            "factor_rows": factor_rows,
        }


def verify_factor_production(
    workspace_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return Factor-only authority status; it deliberately does not read System."""

    return FactorProductionStore(workspace_root).verify_active()


def read_factor_production_signal(
    workspace_root: str | os.PathLike[str], *, factor_id: str
) -> dict[str, Any]:
    """Read one signal from the exact verified active Factor generation."""

    return FactorProductionStore(workspace_root).read_active_signal(factor_id)


__all__ = [
    "ADMISSION_ROUTE",
    "FACTOR_ACTIVE_POINTER_PATH",
    "FACTOR_AUTHORITY_ACTIVE",
    "FACTOR_EMPTY_POINTER_SHA256",
    "FACTOR_PRODUCTION_MARKER_PATH",
    "FACTOR_PRODUCTION_OBSERVATIONS_ROOT",
    "FACTOR_PRODUCTION_GENERATION_KIND",
    "FACTOR_PRODUCTION_MARKET_INPUT_KIND",
    "FACTOR_PRODUCTION_SCOPE",
    "FACTOR_READINESS_READY",
    "FactorProductionStore",
    "FactorReadOnlySystemCustody",
    "FUNDAMENTAL_ADVISORY",
    "FUNDAMENTAL_NOT_USED",
    "PRODUCER_IDENTITY",
    "build_factor_production_generation_receipt",
    "read_factor_production_signal",
    "validate_factor_production_activation_bundle",
    "validate_factor_production_generation_receipt",
    "validate_factor_production_marker",
    "validate_factor_production_pointer",
    "validate_factor_production_prepared",
    "verify_factor_production",
]
