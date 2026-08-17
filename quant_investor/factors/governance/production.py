"""Production-only assembler for the first unified Factor generation.

The operator accepts one sealed request whose every input is an explicit local
path plus expected byte hash.  It never discovers "latest" inputs, calls a
provider, or writes the System active pointer.  Strict Factor sources are
copied into a new owner-only staging root, registered, replayed, and then read
again by the normal generation verifier.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from io import FileIO
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from typing import Any, Final, cast
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    contract_catalog_sha256,
    get_contract,
    seal_artifact,
    validate_artifact,
)
from quant_investor.factors.governance import (
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    FactorValidationStore,
)
from quant_investor.factors.governance.bootstrap import (
    _factor_set_sha256,
    _set_rows,
    bootstrap_factor_definitions,
    compute_bootstrap_signals,
)
from quant_investor.factors.governance.contextual import (
    _bootstrap_sources,
    _signal_hashes,
    _signal_statistics,
)
from quant_investor.factors.governance.implementations import installed_semantic_row
from quant_investor.factors.governance.errors import FactorGovernanceError
from quant_investor.factors.governance.source import decode_source_role
from quant_investor.factors.governance.source import role_schema
from quant_investor.intelligence import assess_readiness
from quant_investor.market.fundamental_incremental import (
    SafeSuccessorError,
    validate_successor_provenance,
)
from quant_investor.market.exchange_calendar_closure import (
    validate_exchange_calendar_compilation,
    validate_historical_compilation_envelope,
)
from quant_investor.market.exchange_calendar_official import (
    EvidenceRole,
    decode_capture_projection,
    decoder_code_sha256,
    decoder_id,
)
from quant_investor.market.tushare_calendar_authority import (
    AUTHORITY_ROUTE as TRUSTED_PROVIDER_CALENDAR_ROUTE,
    COMPILATION_KIND as TRUSTED_PROVIDER_CALENDAR_COMPILATION_KIND,
    SOURCE_LIMITATIONS as TRUSTED_PROVIDER_CALENDAR_SOURCE_LIMITATIONS,
    validate_calendar_authority_policy,
    validate_trusted_provider_calendar_capability,
    validate_trusted_provider_calendar_capture_execution,
    validate_trusted_provider_calendar_capture_success,
    validate_trusted_provider_calendar_compilation,
    validate_trusted_provider_calendar_capture_transaction,
)

from quant_investor.system.controller import build_emergency_controller
from quant_investor.system.errors import (
    SystemContractError,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
)
from quant_investor.system.requests import ASSEMBLY_REQUEST_FIELDS
from quant_investor.system.components import (
    BOOTSTRAP_VALIDATION_PROFILE,
    MAXIMUM_DECODED_SOURCE_CELLS,
    MAXIMUM_DECODED_SOURCE_ROWS,
    MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES,
)
from quant_investor.system.bootstrap_receipt import (
    ASSEMBLER_MODULE_PATH,
    FUNDAMENTAL_SOURCE_BLOCKERS,
    INPUT_SOURCE_ROW_FIELDS,
    PRODUCTION_BOOTSTRAP_RECEIPT_CONTRACT_SHA256,
    PRODUCTION_BOOTSTRAP_RECEIPT_FIELDS,
    build_production_bootstrap_receipt,
    production_generation_intent_sha256,
    validate_production_bootstrap_receipt,
)
from quant_investor.system.storage import ACTIVE_POINTER_PATH, MIGRATION_MARKER_PATH
from quant_investor.system.store import (
    SystemStore,
    generation_assembly_identity,
    object_ref_for_artifact,
    validate_object_ref,
)
from quant_investor.system.suspension import build_suspended_generation

BOOTSTRAP_OPERATOR_REQUEST_KIND: Final = "system.bootstrap_operator_request"
BOOTSTRAP_OPERATOR_REQUEST_CONTRACT_SHA256: Final = get_contract(
    BOOTSTRAP_OPERATOR_REQUEST_KIND
).contract_sha256
BOOTSTRAP_OPERATOR_REQUEST_FIELDS: Final = frozenset(
    {
        "bootstrap_operation_id",
        "state",
        "source_root_id",
        "release_manifest_ref",
        "exchange_calendar_file_ref",
        "market_scope_file_ref",
        "market_pointer_file_ref",
        "market_snapshot_manifest_file_ref",
        "market_table_file_refs",
        "pit_pointer_file_ref",
        "pit_generation_manifest_file_ref",
        "pit_membership_file_ref",
        "calendar_runtime_json_file_ref",
        "calendar_compilation_file_ref",
        "calendar_authority_policy_file_ref",
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
        "trusted_provider_calendar_capability_file_ref",
        "trusted_provider_calendar_capture_transaction_file_ref",
        "trusted_provider_calendar_capture_execution_file_ref",
        "trusted_provider_calendar_capture_success_file_ref",
        "trusted_provider_release_install_input_file_ref",
        "fundamental_pointer_file_ref",
        "fundamental_generation_manifest_file_ref",
        "fundamental_table_file_refs",
        "fundamental_evidence_file_refs",
        "bootstrap_decision_file_ref",
        "skill_tree_sha256",
        "automation_semantic_sha256",
        "source_blockers",
        "trusted_at",
    }
)
FILE_REF_FIELDS: Final = frozenset({"relative_path", "byte_sha256"})
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_A_SHARE_SYMBOL_RE: Final = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_MAX_INPUT_BYTES: Final = 4 * 1024 * 1024 * 1024
_JSON_MAX_BYTES: Final = 64 * 1024 * 1024
_MAX_MARKET_TABLES: Final = 24
_MINIMUM_MARKET_SESSIONS: Final = 91
_PARQUET_BATCH_ROWS: Final = 2_048
_PARQUET_BATCH_BYTES: Final = 16 * 1024**2
_CALENDAR_INPUT_COUNT_LIMITS: Final = {
    "official_calendar_raw_file_refs": 1_152,
    "official_calendar_capture_file_refs": 1_024,
    "official_calendar_decoder_admission_file_refs": 128,
    "official_calendar_index_closure_file_refs": 128,
    "trusted_provider_calendar_raw_file_refs": 4,
    "trusted_provider_calendar_capture_file_refs": 3,
}
_CALENDAR_INPUT_BYTE_LIMITS: Final = {
    "official_calendar_raw_file_refs": 8 * 1024**2,
    "official_calendar_capture_file_refs": 1024**2,
    "official_calendar_decoder_admission_file_refs": 1024**2,
    "official_calendar_index_closure_file_refs": 4 * 1024**2,
    "trusted_provider_calendar_raw_file_refs": 4 * 1024**2,
    "trusted_provider_calendar_capture_file_refs": 1024**2,
    "trusted_provider_calendar_capability_file_ref": 1024**2,
    "trusted_provider_calendar_capture_transaction_file_ref": 1024**2,
    "trusted_provider_calendar_capture_execution_file_ref": 2 * 1024**2,
    "trusted_provider_calendar_capture_success_file_ref": 2 * 1024**2,
    "trusted_provider_release_install_input_file_ref": 64 * 1024**2,
    "calendar_authority_policy_file_ref": 1024**2,
    "calendar_runtime_json_file_ref": 16 * 1024**2,
    "calendar_compilation_file_ref": 16 * 1024**2,
    "exchange_calendar_file_ref": 64 * 1024**2,
}
_MAXIMUM_CALENDAR_REPLAY_BYTES: Final = 128 * 1024**2
# Historical decoder names remain only inside the explicitly disabled rejection
# functions below.  They are not request fields or production DAG roles.
_CALENDAR_MANIFEST_KIND: Final = "system.exchange_calendar_manifest"
_CALENDAR_MANIFEST_CONTRACT_SHA256: Final = get_contract(_CALENDAR_MANIFEST_KIND).contract_sha256
_CALENDAR_CAPTURE_KIND: Final = "system.exchange_calendar_capture"
_CALENDAR_CAPTURE_CONTRACT_SHA256: Final = get_contract(_CALENDAR_CAPTURE_KIND).contract_sha256
_CALENDAR_MANIFEST_FIELDS: Final = frozenset(
    {
        "calendar_manifest_id",
        "state",
        "coverage_start_date",
        "cutoff_date",
        "timezone",
        "calendar_file_ref",
        "transform_code_sha256",
        "exchange_rows",
    }
)
_CALENDAR_EXCHANGE_FIELDS: Final = frozenset(
    {
        "exchange_id",
        "issuer",
        "daily_status_source_url",
        "daily_status_captured_at",
        "daily_status_raw_file_ref",
        "daily_status_capture_file_ref",
        "session_rule_source_url",
        "session_rule_captured_at",
        "session_rule_raw_file_ref",
        "session_rule_capture_file_ref",
        "open_session_count",
        "open_session_sha256",
        "session_intervals",
    }
)
_CALENDAR_INTERVAL_FIELDS: Final = frozenset({"opens_local", "closes_local"})
_CALENDAR_CAPTURE_FIELDS: Final = frozenset()
_CALENDAR_STATUS_FIELDS: Final = frozenset({"date", "status"})
_OFFICIAL_CALENDAR_AUTHORITIES: Final = {
    "SSE": ("SSE_OFFICIAL", "www.sse.com.cn"),
    "SZSE": ("SZSE_OFFICIAL", "www.szse.cn"),
    "BSE": ("BSE_OFFICIAL", "www.bse.cn"),
}
_CN_CONTINUOUS_SESSION_INTERVALS: Final = [
    {"opens_local": "09:30:00", "closes_local": "11:30:00"},
    {"opens_local": "13:00:00", "closes_local": "15:00:00"},
]
_EXCHANGE_BY_SUFFIX: Final = {".SH": "SSE", ".SZ": "SZSE", ".BJ": "BSE"}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not a canonical identifier")
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not lowercase SHA-256")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not canonical UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} is not canonical UTC")
    if parsed > datetime.now(timezone.utc):
        raise SystemContractError(f"{label} may not be in the future")
    return value


def _file_ref(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != FILE_REF_FIELDS:
        raise SystemContractError(f"{label} fields are not exact")
    relative = value.get("relative_path")
    if type(relative) is not str:
        raise SystemContractError(f"{label}.relative_path is invalid")
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or str(path) != relative
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SystemContractError(f"{label}.relative_path is invalid")
    return {
        "relative_path": relative,
        "byte_sha256": _sha(value.get("byte_sha256"), label=f"{label}.byte_sha256"),
    }


def _file_refs(value: Any, *, label: str, minimum: int = 1) -> list[dict[str, str]]:
    if type(value) is not list or len(value) < minimum:
        raise SystemContractError(f"{label} must be a nonempty exact list")
    rows = [_file_ref(row, label=f"{label}[{index}]") for index, row in enumerate(value)]
    keys = [(row["relative_path"], row["byte_sha256"]) for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SystemContractError(f"{label} must be sorted and unique")
    return rows


def _calendar_reference_rows(
    normalized: Mapping[str, Any],
) -> list[tuple[str, int, Mapping[str, str]]]:
    rows: list[tuple[str, int, Mapping[str, str]]] = []
    for field, maximum_count in _CALENDAR_INPUT_COUNT_LIMITS.items():
        references = normalized[field]
        if len(references) > maximum_count:
            raise SystemSecurityError(f"{field} exceeds its production count bound")
        rows.extend((field, ordinal, reference) for ordinal, reference in enumerate(references))
    for field in (
        "calendar_authority_policy_file_ref",
        "calendar_runtime_json_file_ref",
        "calendar_compilation_file_ref",
        "exchange_calendar_file_ref",
    ):
        rows.append((field, 0, normalized["files"][field]))
    for field in (
        "trusted_provider_calendar_capability_file_ref",
        "trusted_provider_calendar_capture_transaction_file_ref",
        "trusted_provider_calendar_capture_execution_file_ref",
        "trusted_provider_calendar_capture_success_file_ref",
        "trusted_provider_release_install_input_file_ref",
    ):
        reference = normalized[field]
        if reference is not None:
            rows.append((field, 0, reference))
    return rows


def _validate_calendar_size_rows(
    rows: Sequence[tuple[str, int, int]],
) -> None:
    aggregate = 0
    for field, _ordinal, size in rows:
        maximum = _CALENDAR_INPUT_BYTE_LIMITS[field]
        if type(size) is not int or size <= 0 or size > maximum:
            raise SystemSecurityError(f"{field} exceeds its production byte bound")
        aggregate += size
        if aggregate > _MAXIMUM_CALENDAR_REPLAY_BYTES:
            raise SystemSecurityError("calendar replay exceeds its aggregate byte bound")


def _preflight_calendar_input_budget(*, normalized: Mapping[str, Any], input_root: Path) -> None:
    sizes: list[tuple[str, int, int]] = []
    for field, ordinal, reference in _calendar_reference_rows(normalized):
        with _open_input_leaf(
            input_root,
            reference,
            maximum_bytes=_CALENDAR_INPUT_BYTE_LIMITS[field],
        ) as (_descriptor, metadata, _parent, _leaf):
            sizes.append((field, ordinal, metadata.st_size))
    _validate_calendar_size_rows(sizes)


def validate_bootstrap_operator_request(  # noqa: C901
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Deep-validate an exact sealed production assembly request."""

    try:
        request = validate_artifact(
            document,
            expected_kind=BOOTSTRAP_OPERATOR_REQUEST_KIND,
            expected_contract_sha256=BOOTSTRAP_OPERATOR_REQUEST_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemContractError("bootstrap operator request contract failed") from exc
    payload = request["payload"]
    if set(payload) != BOOTSTRAP_OPERATOR_REQUEST_FIELDS:
        raise SystemContractError("bootstrap operator request fields are not exact")
    if payload["state"] != "SEALED":
        raise SystemContractError("bootstrap operator request is not SEALED")
    operation_id = _identifier(payload["bootstrap_operation_id"], label="bootstrap_operation_id")
    source_root_id = _identifier(payload["source_root_id"], label="source_root_id")
    trusted_at = _timestamp(payload["trusted_at"], label="trusted_at")
    if request["created_at"] != trusted_at:
        raise SystemContractError("request created_at/trusted_at binding differs")
    release_ref = validate_object_ref(payload["release_manifest_ref"], label="release_manifest_ref")
    source_blockers = payload["source_blockers"]
    if type(source_blockers) is not list:
        raise SystemContractError("source_blockers must be a list")
    normalized_blockers = [
        _identifier(row, label=f"source_blockers[{index}]")
        for index, row in enumerate(source_blockers)
    ]
    if normalized_blockers != sorted(set(normalized_blockers)):
        raise SystemContractError("source_blockers must be sorted and unique")
    scalar_files = {
        field: _file_ref(payload[field], label=field)
        for field in (
            "exchange_calendar_file_ref",
            "market_scope_file_ref",
            "market_pointer_file_ref",
            "market_snapshot_manifest_file_ref",
            "pit_pointer_file_ref",
            "pit_generation_manifest_file_ref",
            "pit_membership_file_ref",
            "calendar_authority_policy_file_ref",
            "calendar_runtime_json_file_ref",
            "calendar_compilation_file_ref",
            "fundamental_pointer_file_ref",
            "fundamental_generation_manifest_file_ref",
            "bootstrap_decision_file_ref",
        )
    }
    official_calendar_raw = _file_refs(
        payload["official_calendar_raw_file_refs"],
        label="official_calendar_raw_file_refs",
        minimum=0,
    )
    official_calendar_captures = _file_refs(
        payload["official_calendar_capture_file_refs"],
        label="official_calendar_capture_file_refs",
        minimum=0,
    )
    official_calendar_admissions = _file_refs(
        payload["official_calendar_decoder_admission_file_refs"],
        label="official_calendar_decoder_admission_file_refs",
        minimum=0,
    )
    official_calendar_indexes = _file_refs(
        payload["official_calendar_index_closure_file_refs"],
        label="official_calendar_index_closure_file_refs",
        minimum=0,
    )
    provider_calendar_raw = _file_refs(
        payload["trusted_provider_calendar_raw_file_refs"],
        label="trusted_provider_calendar_raw_file_refs",
        minimum=0,
    )
    provider_calendar_captures = _file_refs(
        payload["trusted_provider_calendar_capture_file_refs"],
        label="trusted_provider_calendar_capture_file_refs",
        minimum=0,
    )
    capability_value = payload["trusted_provider_calendar_capability_file_ref"]
    provider_calendar_capability = (
        None
        if capability_value is None
        else _file_ref(
            capability_value,
            label="trusted_provider_calendar_capability_file_ref",
        )
    )
    transaction_value = payload["trusted_provider_calendar_capture_transaction_file_ref"]
    provider_calendar_transaction = (
        None
        if transaction_value is None
        else _file_ref(
            transaction_value,
            label="trusted_provider_calendar_capture_transaction_file_ref",
        )
    )
    execution_value = payload["trusted_provider_calendar_capture_execution_file_ref"]
    provider_calendar_execution = (
        None
        if execution_value is None
        else _file_ref(
            execution_value,
            label="trusted_provider_calendar_capture_execution_file_ref",
        )
    )
    success_value = payload["trusted_provider_calendar_capture_success_file_ref"]
    provider_calendar_success = (
        None
        if success_value is None
        else _file_ref(
            success_value,
            label="trusted_provider_calendar_capture_success_file_ref",
        )
    )
    release_input_value = payload["trusted_provider_release_install_input_file_ref"]
    provider_release_input = (
        None
        if release_input_value is None
        else _file_ref(
            release_input_value,
            label="trusted_provider_release_install_input_file_ref",
        )
    )
    official_present = all(
        (
            official_calendar_raw,
            official_calendar_captures,
            official_calendar_admissions,
            official_calendar_indexes,
        )
    )
    official_absent = not any(
        (
            official_calendar_raw,
            official_calendar_captures,
            official_calendar_admissions,
            official_calendar_indexes,
        )
    )
    provider_present = (
        len(provider_calendar_raw) == 4
        and len(provider_calendar_captures) == 3
        and provider_calendar_capability is not None
        and provider_calendar_transaction is not None
        and provider_calendar_execution is not None
        and provider_calendar_success is not None
        and provider_release_input is not None
    )
    provider_absent = (
        not provider_calendar_raw
        and not provider_calendar_captures
        and provider_calendar_capability is None
        and provider_calendar_transaction is None
        and provider_calendar_execution is None
        and provider_calendar_success is None
        and provider_release_input is None
    )
    if not ((official_present and provider_absent) or (official_absent and provider_present)):
        raise SystemContractError("calendar authority route tombstones are not exact")
    market_tables = _file_refs(payload["market_table_file_refs"], label="market_table_file_refs")
    if len(market_tables) > _MAX_MARKET_TABLES:
        raise SystemContractError("market_table_file_refs exceeds its exact bound")
    fundamental_tables = _file_refs(
        payload["fundamental_table_file_refs"],
        label="fundamental_table_file_refs",
    )
    fundamental_evidence = _file_refs(
        payload["fundamental_evidence_file_refs"],
        label="fundamental_evidence_file_refs",
    )
    all_paths = [
        row["relative_path"]
        for row in [
            *scalar_files.values(),
            *official_calendar_raw,
            *official_calendar_captures,
            *official_calendar_admissions,
            *official_calendar_indexes,
            *provider_calendar_raw,
            *provider_calendar_captures,
            *market_tables,
            *fundamental_tables,
            *fundamental_evidence,
        ]
    ]
    if provider_calendar_capability is not None:
        all_paths.append(provider_calendar_capability["relative_path"])
    if provider_calendar_transaction is not None:
        all_paths.append(provider_calendar_transaction["relative_path"])
    for reference in (
        provider_calendar_execution,
        provider_calendar_success,
        provider_release_input,
    ):
        if reference is not None:
            all_paths.append(reference["relative_path"])
    if len(all_paths) != len(set(all_paths)):
        raise SystemContractError("bootstrap input paths must be globally unique")
    return {
        "document": request,
        "operation_id": operation_id,
        "source_root_id": source_root_id,
        "trusted_at": trusted_at,
        "release_manifest_ref": release_ref,
        "skill_tree_sha256": _sha(payload["skill_tree_sha256"], label="skill_tree_sha256"),
        "automation_semantic_sha256": _sha(
            payload["automation_semantic_sha256"],
            label="automation_semantic_sha256",
        ),
        "source_blockers": normalized_blockers,
        "files": scalar_files,
        "official_calendar_raw_file_refs": official_calendar_raw,
        "official_calendar_capture_file_refs": official_calendar_captures,
        "official_calendar_decoder_admission_file_refs": official_calendar_admissions,
        "official_calendar_index_closure_file_refs": official_calendar_indexes,
        "trusted_provider_calendar_raw_file_refs": provider_calendar_raw,
        "trusted_provider_calendar_capture_file_refs": provider_calendar_captures,
        "trusted_provider_calendar_capability_file_ref": provider_calendar_capability,
        "trusted_provider_calendar_capture_transaction_file_ref": (provider_calendar_transaction),
        "trusted_provider_calendar_capture_execution_file_ref": provider_calendar_execution,
        "trusted_provider_calendar_capture_success_file_ref": provider_calendar_success,
        "trusted_provider_release_install_input_file_ref": provider_release_input,
        "market_table_file_refs": market_tables,
        "fundamental_table_file_refs": fundamental_tables,
        "fundamental_evidence_file_refs": fundamental_evidence,
    }


def _absolute_input_root(input_root: Path) -> Path:
    candidate = input_root.absolute()
    if not candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise SystemSecurityError("bootstrap input root is not canonical")
    return candidate


def _verify_input_directory(metadata: os.stat_result) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or mode & 0o022:
        raise SystemSecurityError("bootstrap input directory security is invalid")


@contextmanager
def _open_input_root(input_root: Path) -> Iterator[tuple[Path, int, os.stat_result]]:
    """Walk every absolute root component without following a symlink."""

    root = _absolute_input_root(input_root)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory: int | None = None
    try:
        directory = os.open(root.anchor, directory_flags)
        for part in root.parts[1:]:
            child = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        before = os.fstat(directory)
        mode = stat.S_IMODE(before.st_mode)
        path_before = os.lstat(root)
        if (
            not stat.S_ISDIR(before.st_mode)
            or before.st_uid != os.geteuid()
            or mode != 0o700
            or _file_identity(before) != _file_identity(path_before)
        ):
            raise SystemSecurityError("bootstrap input root security is invalid")
        yield root, directory, before
        after = os.fstat(directory)
        path_after = os.lstat(root)
        if _file_identity(before) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(path_after):
            raise SystemSecurityError("bootstrap input root changed during descriptor use")
    except (SystemContractError, SystemSecurityError, SystemStorageError):
        raise
    except OSError as exc:
        raise SystemSecurityError(
            "bootstrap input root is unavailable or contains a symlink"
        ) from exc
    finally:
        if directory is not None:
            os.close(directory)


def _input_root_path(input_root: Path) -> Path:
    with _open_input_root(input_root) as (root, _descriptor, _metadata):
        return root


def _verify_input_file(metadata: os.stat_result, *, maximum_bytes: int) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or mode != 0o600
        or metadata.st_size <= 0
        or metadata.st_size > maximum_bytes
    ):
        raise SystemSecurityError("bootstrap input storage security is invalid")


@contextmanager
def _open_input_leaf(
    input_root: Path,
    reference: Mapping[str, str],
    *,
    maximum_bytes: int,
) -> Iterator[tuple[int, os.stat_result, int, str]]:
    """Pin an explicit input through no-follow directory descriptors."""

    relative = PurePosixPath(reference["relative_path"])
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    directory: int | None = None
    descriptor: int | None = None
    try:
        with _open_input_root(input_root) as (_root, root_descriptor, _root_metadata):
            directory = os.dup(root_descriptor)
            os.set_inheritable(directory, False)
            for part in relative.parts[:-1]:
                child = os.open(part, directory_flags, dir_fd=directory)
                _verify_input_directory(os.fstat(child))
                os.close(directory)
                directory = child
            leaf = relative.name
            path_before = os.stat(leaf, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISLNK(path_before.st_mode):
                raise SystemSecurityError("bootstrap input path contains a symlink")
            _verify_input_file(path_before, maximum_bytes=maximum_bytes)
            descriptor = os.open(leaf, read_flags, dir_fd=directory)
            before = os.fstat(descriptor)
            _verify_input_file(before, maximum_bytes=maximum_bytes)
            if _file_identity(path_before) != _file_identity(before):
                raise SystemSecurityError("bootstrap input path changed before open")
            yield descriptor, before, directory, leaf
            after = os.fstat(descriptor)
            path_after = os.stat(leaf, dir_fd=directory, follow_symlinks=False)
            _verify_input_file(after, maximum_bytes=maximum_bytes)
            _verify_input_file(path_after, maximum_bytes=maximum_bytes)
            if _file_identity(before) != _file_identity(after) or _file_identity(
                after
            ) != _file_identity(path_after):
                raise SystemSecurityError("bootstrap input path changed during descriptor use")
    except (SystemContractError, SystemSecurityError, SystemStorageError):
        raise
    except OSError as exc:
        raise SystemSecurityError(
            "bootstrap input path is unavailable or contains a symlink"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def _stable_input_path(input_root: Path, reference: Mapping[str, str]) -> Path:
    with _open_input_leaf(
        input_root,
        reference,
        maximum_bytes=_MAX_INPUT_BYTES,
    ):
        pass
    return _input_root_path(input_root) / reference["relative_path"]


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if _file_identity(before) != _file_identity(after) or size != before.st_size:
        raise SystemSecurityError("bootstrap input changed during read")
    return digest.hexdigest(), size


def _read_stable_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise SystemSecurityError("bootstrap input size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if remaining or _file_identity(before) != _file_identity(after) or len(raw) != after.st_size:
        raise SystemSecurityError("bootstrap input changed during exact read")
    return raw


@contextmanager
def _open_stable_parquet_path(
    path: Path,
    *,
    expected_sha256: str,
) -> Iterator[FileIO]:
    """Open one copied Parquet through one stable, owner-only descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    stream: FileIO | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or mode & 0o022
            or mode & 0o111
            or before.st_size <= 0
            or before.st_size > MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES
        ):
            raise SystemSecurityError("Parquet source descriptor security is invalid")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise SystemSecurityError("Parquet source descriptor hash differs")
        os.lseek(descriptor, 0, os.SEEK_SET)
        duplicate = os.dup(descriptor)
        os.set_inheritable(duplicate, False)
        stream = FileIO(duplicate, mode="rb", closefd=True)
        yield stream
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        if _file_identity(before) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(path_after):
            raise SystemSecurityError("Parquet source changed during descriptor replay")
    except (SystemContractError, SystemSecurityError, SystemStorageError):
        raise
    except OSError as exc:
        raise SystemStorageError("Parquet source descriptor cannot be opened") from exc
    finally:
        if stream is not None:
            stream.close()
        if descriptor is not None:
            os.close(descriptor)


def _iter_bounded_parquet_rows(  # noqa: C901
    *,
    columns: Sequence[str],
    path: Path | None = None,
    expected_sha256: str | None = None,
    store: SystemStore | None = None,
    source_ref: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield bounded rows from either a copied input or governed source object."""

    if (path is None) == (source_ref is None) or (store is None) != (source_ref is None):
        raise SystemContractError("Parquet replay source selection is invalid")

    def decoded_rows(stream: FileIO) -> Iterator[dict[str, Any]]:
        total_rows = 0
        try:
            parquet = pq.ParquetFile(stream)
            if not set(columns) <= set(parquet.schema_arrow.names):
                raise SystemContractError("Parquet replay columns are incomplete")
            for batch in parquet.iter_batches(
                columns=list(columns),
                batch_size=_PARQUET_BATCH_ROWS,
                use_threads=False,
            ):
                total_rows += batch.num_rows
                if (
                    total_rows > MAXIMUM_DECODED_SOURCE_ROWS
                    or total_rows * len(columns) > MAXIMUM_DECODED_SOURCE_CELLS
                    or batch.num_rows > _PARQUET_BATCH_ROWS
                    or batch.nbytes > _PARQUET_BATCH_BYTES
                ):
                    raise SystemSecurityError("Parquet replay exceeds decoded row/cell bounds")
                yield from batch.to_pylist()
        except (SystemContractError, SystemSecurityError, SystemStorageError):
            raise
        except Exception as exc:
            raise SystemContractError("Parquet source decode failed") from exc

    if path is not None:
        if type(expected_sha256) is not str:
            raise SystemContractError("Parquet copied input SHA is absent")
        with _open_stable_parquet_path(path, expected_sha256=expected_sha256) as stream:
            yield from decoded_rows(stream)
        return
    if store is None or source_ref is None:  # pragma: no cover - guarded above
        raise SystemContractError("governed Parquet source is absent")
    with store.open_source_object(
        source_ref,
        maximum_bytes=MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES,
        decoded_reservation_bytes=_PARQUET_BATCH_BYTES,
    ) as (_payload, stream):
        yield from decoded_rows(stream)


def _verify_input(
    input_root: Path,
    reference: Mapping[str, str],
) -> tuple[Path, int]:
    with _open_input_leaf(
        input_root,
        reference,
        maximum_bytes=_MAX_INPUT_BYTES,
    ) as (descriptor, _metadata, _parent, _leaf):
        first, size = _descriptor_digest(descriptor, maximum_bytes=_MAX_INPUT_BYTES)
        second, second_size = _descriptor_digest(descriptor, maximum_bytes=_MAX_INPUT_BYTES)
    if first != reference["byte_sha256"] or second != first or second_size != size:
        raise SystemSecurityError("bootstrap input exact hash changed")
    return _input_root_path(input_root) / reference["relative_path"], size


def _ensure_staging_root(workspace_root: Path, operation_id: str) -> Path:
    root = workspace_root / "data/private/system_source_staging" / operation_id
    current = workspace_root
    for part in PurePosixPath("data/private/system_source_staging").parts + (operation_id,):
        current = current / part
        if current.exists():
            metadata = os.lstat(current)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise SystemSecurityError("source staging directory is not owner-only")
        else:
            current.mkdir(mode=0o700)
    return root


def _staging_relative_root(operation_id: str) -> PurePosixPath:
    return PurePosixPath("data/private/system_source_staging") / operation_id


def _workspace_source_relative(operation_id: str, relative: str) -> str:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemSecurityError("source staging relative path is invalid")
    return str(_staging_relative_root(operation_id) / path)


def _prefix_copied_paths(
    copied: Mapping[str, list[str] | str], operation_id: str
) -> dict[str, list[str] | str]:
    prefixed: dict[str, list[str] | str] = {}
    for field, value in copied.items():
        if type(value) is str:
            prefixed[field] = _workspace_source_relative(operation_id, value)
        elif type(value) is list and all(type(row) is str for row in value):
            prefixed[field] = [_workspace_source_relative(operation_id, row) for row in value]
        else:
            raise SystemContractError(f"copied {field} path projection is invalid")
    return prefixed


def _destination(staging_root: Path, relative: str) -> Path:
    path = staging_root / relative
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in (path.parent, *path.parent.parents):
        if parent == staging_root.parent:
            break
        if staging_root == parent or staging_root in parent.parents:
            parent.chmod(0o700)
    return path


def _descriptor_digest(descriptor: int, *, maximum_bytes: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > maximum_bytes:
            raise SystemSecurityError("bootstrap input exceeds its descriptor byte bound")
        digest.update(chunk)
    after = os.fstat(descriptor)
    if _file_identity(before) != _file_identity(after) or size != after.st_size:
        raise SystemSecurityError("bootstrap input changed during descriptor hash")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


def _copy_verified_input(  # noqa: C901
    *,
    input_root: Path,
    reference: Mapping[str, str],
    target: Path,
    maximum_bytes: int,
) -> int:
    expected_sha = reference["byte_sha256"]
    with _open_input_leaf(
        input_root,
        reference,
        maximum_bytes=maximum_bytes,
    ) as (input_fd, before, parent_fd, leaf):
        if target.exists():
            first, size = _descriptor_digest(input_fd, maximum_bytes=maximum_bytes)
            second, second_size = _descriptor_digest(input_fd, maximum_bytes=maximum_bytes)
            observed, observed_size = _file_digest(target)
            if (
                first != expected_sha
                or second != first
                or second_size != size
                or observed != expected_sha
                or observed_size != size
            ):
                raise SystemStorageError("source staging exact-once conflict")
            return size

        temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        output_fd: int | None = None
        try:
            output_fd = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(input_fd, min(1024 * 1024, maximum_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_bytes:
                    raise SystemSecurityError("bootstrap input exceeds its copy byte bound")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(output_fd, view)
                    view = view[written:]
            after = os.fstat(input_fd)
            path_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if (
                size != before.st_size
                or digest.hexdigest() != expected_sha
                or _file_identity(before) != _file_identity(after)
                or _file_identity(after) != _file_identity(path_after)
            ):
                raise SystemSecurityError(
                    "bootstrap input exact hash or identity changed during bounded copy"
                )
            second, second_size = _descriptor_digest(input_fd, maximum_bytes=maximum_bytes)
            if second != expected_sha or second_size != size:
                raise SystemSecurityError("bootstrap input exact hash changed after copy")
            os.fsync(output_fd)
            os.close(output_fd)
            output_fd = None
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError as exc:
                raise SystemStorageError("source staging exact-once conflict") from exc
            temporary.unlink()
            os.chmod(target, 0o600, follow_symlinks=False)
            directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if output_fd is not None:
                os.close(output_fd)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    observed, observed_size = _file_digest(target)
    if observed != expected_sha or observed_size != size:
        raise SystemStorageError("source staging exact-byte readback mismatch")
    return size


def _write_exact_once(target: Path, raw: bytes) -> None:
    if type(raw) is not bytes or not raw:
        raise SystemStorageError("generated source bytes are empty")
    expected = _sha256(raw)
    if target.exists():
        observed, _ = _file_digest(target)
        if observed != expected:
            raise SystemStorageError("generated source exact-once conflict")
        return
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, target, follow_symlinks=False)
    except FileExistsError as exc:
        temporary.unlink()
        raise SystemStorageError("generated source exact-once conflict") from exc
    temporary.unlink()
    os.chmod(target, 0o600, follow_symlinks=False)
    observed, _ = _file_digest(target)
    if observed != expected:
        raise SystemStorageError("generated source exact-byte readback mismatch")


def _write_parquet_once(
    target: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
) -> str:
    """Publish deterministic strict Parquet without overwriting an existing file."""

    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    if not table.schema.equals(schema, check_metadata=True):
        raise SystemContractError("generated strict Parquet schema differs")
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            version="2.6",
        )
        os.chmod(temporary, 0o600, follow_symlinks=False)
        descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        expected, _ = _file_digest(temporary)
        if target.exists():
            observed, _ = _file_digest(target)
            if observed != expected:
                raise SystemStorageError("generated strict Parquet exact-once conflict")
        else:
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError as exc:
                raise SystemStorageError("generated strict Parquet exact-once conflict") from exc
            os.chmod(target, 0o600, follow_symlinks=False)
            directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    first, _ = _file_digest(target)
    second, _ = _file_digest(target)
    if first != expected or second != first:
        raise SystemStorageError("generated strict Parquet readback differs")
    return first


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise SystemContractError("source JSON contains duplicate or invalid keys")
        result[key] = value
    return result


def _parse_source_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _JSON_MAX_BYTES:
        raise SystemContractError(f"{label} JSON byte bound failed")
    try:
        text = raw.decode("utf-8")
        document = json.loads(text, object_pairs_hook=_reject_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemContractError(f"{label} JSON decode failed") from exc
    if type(document) is not dict:
        raise SystemContractError(f"{label} must be a JSON object")
    return document


def _read_copied_json(
    staging_root: Path,
    copied: Mapping[str, list[str] | str],
    field: str,
) -> dict[str, Any]:
    path = staging_root / _copied_path(copied, field)
    return _parse_source_json(_read_stable_bytes(path, maximum_bytes=_JSON_MAX_BYTES), label=field)


def _compact_date(value: Any, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9]{8}", value) is None:
        raise SystemContractError(f"{label} is not compact date")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise SystemContractError(f"{label} is not compact date") from exc
    return value


def _symbols(value: Any, *, label: str, minimum: int = 1) -> list[str]:
    if type(value) is not list or len(value) < minimum:
        raise SystemContractError(f"{label} must be a nonempty symbol list")
    rows = list(value)
    if any(
        type(row) is not str or _A_SHARE_SYMBOL_RE.fullmatch(row) is None for row in rows
    ) or rows != sorted(set(rows)):
        raise SystemContractError(f"{label} symbols are not canonical sorted unique")
    return rows


def _symbol_set_sha256(symbols: Sequence[str]) -> str:
    return _sha256("\n".join(symbols).encode("utf-8"))


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SystemContractError(f"{label} must be an object")
    return value


def _source(
    store: SystemStore,
    relative: str,
    *,
    source_format: str,
    media_type: str,
    created_at: str,
) -> dict[str, str]:
    return store.put_source_file(
        relative,
        source_object_id="bootstrap-source-" + hashlib.sha256(relative.encode()).hexdigest(),
        media_type=media_type,
        source_format=source_format,
        created_at=created_at,
    )


def _bundle(
    store: SystemStore,
    operation_id: str,
    role: str,
    rows: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    created_at: str,
) -> dict[str, str]:
    sources: list[dict[str, Any]] = [
        {"role": inner_role, "source_ref": validate_object_ref(source_ref)}
        for inner_role, source_ref in rows
    ]
    sources.sort(key=lambda row: row["role"].encode("utf-8"))
    artifact = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": f"{operation_id}-{role}",
            "state": "IMMUTABLE",
            "sources": sources,
        },
        created_at=created_at,
    )
    return store.put_object(artifact)


def _strict_projection(
    store: SystemStore,
    *,
    calendar_ref: Mapping[str, Any],
    pit_ref: Mapping[str, Any],
    market_ref: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    list[str],
    dict[str, dict[str, str | None]],
    dict[str, Any],
]:
    normalized: dict[str, str] = {}

    def project_calendar(table: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
        del binding
        frame = table.to_pandas()
        return {
            "open_sessions": [value.isoformat() for value in frame["open_session"]],
            "opens_at_utc": [value.isoformat() for value in frame["opens_at_utc"]],
            "closes_at_utc": [value.isoformat() for value in frame["closes_at_utc"]],
        }

    calendar = decode_source_role(
        system_store=store,
        source_object_ref=calendar_ref,
        role="exchange_calendar",
        projector=project_calendar,
    )
    normalized["exchange_calendar"] = calendar.binding["normalized_sha256"]
    pit = decode_source_role(
        system_store=store,
        source_object_ref=pit_ref,
        role="pit_universe",
        projector=lambda table, binding: {
            "all_symbols": table.to_pandas()["symbol"].tolist(),
            "eligible_symbols": table.to_pandas()
            .loc[lambda frame: frame["tradable"].eq(True) & frame["total_mv"].gt(0), "symbol"]
            .tolist(),
            "signal_session": table.to_pandas()["signal_session"].iloc[0].isoformat(),
        },
    )
    normalized["pit_universe"] = pit.binding["normalized_sha256"]
    eligible = pit.projection["eligible_symbols"]
    all_symbols = pit.projection["all_symbols"]
    if all_symbols != sorted(set(all_symbols)):
        raise SystemContractError("strict PIT full cohort is not exact")
    if eligible != sorted(set(eligible)):
        raise SystemContractError("strict PIT eligible cohort is not exact")

    def project_market(table: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
        del binding
        frame = table.to_pandas()
        frames = {
            symbol: group.drop(columns=["symbol"]).reset_index(drop=True)
            for symbol, group in frame.groupby("symbol", sort=True)
        }
        computed = compute_bootstrap_signals(frames, source_format="PARQUET")
        return {
            "latest_market_session": frame["trade_date"].max().isoformat(),
            "market_symbols": sorted(frame["symbol"].unique().tolist()),
            "market_sessions": sorted(
                value.isoformat() for value in frame["trade_date"].unique().tolist()
            ),
            "signals": {
                factor_id: {
                    symbol: None if pd.isna(value) else float(value).hex()
                    for symbol, value in computed[factor_id].sort_index().items()
                }
                for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
            },
        }

    market = decode_source_role(
        system_store=store,
        source_object_ref=market_ref,
        role="market_history",
        projector=project_market,
    )
    normalized["market_history"] = market.binding["normalized_sha256"]
    return (
        normalized,
        eligible,
        market.projection["signals"],
        {
            "calendar": calendar.projection,
            "all_pit_symbols": all_symbols,
            "eligible_pit_symbols": eligible,
            "market_symbols": market.projection["market_symbols"],
            "market_sessions": market.projection["market_sessions"],
            "latest_market_session": market.projection["latest_market_session"],
            "pit_signal_session": pit.projection["signal_session"],
        },
    )


def _copy_request_inputs(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    input_root: Path,
    staging_root: Path,
) -> dict[str, list[str] | str]:
    calendar_bytes = 0

    def copy_one(field: str, reference: Mapping[str, str], target: Path) -> None:
        nonlocal calendar_bytes
        maximum = _MAX_INPUT_BYTES
        if field in _CALENDAR_INPUT_BYTE_LIMITS:
            remaining = _MAXIMUM_CALENDAR_REPLAY_BYTES - calendar_bytes
            if remaining <= 0:
                raise SystemSecurityError("calendar replay exceeds its aggregate copy bound")
            maximum = min(_CALENDAR_INPUT_BYTE_LIMITS[field], remaining)
        size = _copy_verified_input(
            input_root=input_root,
            reference=reference,
            target=target,
            maximum_bytes=maximum,
        )
        if field in _CALENDAR_INPUT_BYTE_LIMITS:
            calendar_bytes += size
            if calendar_bytes > _MAXIMUM_CALENDAR_REPLAY_BYTES:
                raise SystemSecurityError("calendar replay exceeds its aggregate copy bound")

    destinations = {
        "exchange_calendar_file_ref": "calendar_replay/"
        + normalized["files"]["exchange_calendar_file_ref"]["relative_path"],
        "market_scope_file_ref": "closure/market_scope.json",
        "market_pointer_file_ref": "closure/market_pointer.json",
        "market_snapshot_manifest_file_ref": "closure/market_snapshot_manifest.json",
        "pit_pointer_file_ref": "closure/pit_pointer.json",
        "pit_generation_manifest_file_ref": "closure/pit_generation_manifest.json",
        "pit_membership_file_ref": "closure/pit_membership.parquet",
        "calendar_runtime_json_file_ref": "calendar_replay/"
        + normalized["files"]["calendar_runtime_json_file_ref"]["relative_path"],
        "calendar_compilation_file_ref": "calendar_replay/"
        + normalized["files"]["calendar_compilation_file_ref"]["relative_path"],
        "calendar_authority_policy_file_ref": "calendar_replay/"
        + normalized["files"]["calendar_authority_policy_file_ref"]["relative_path"],
        "bootstrap_decision_file_ref": ("operations/unified_cutover/bootstrap-decision.json"),
    }
    copied: dict[str, list[str] | str] = {}
    for field, relative in destinations.items():
        reference = normalized["files"][field]
        target = _destination(staging_root, relative)
        copy_one(field, reference, target)
        copied[field] = relative
    for field in (
        "fundamental_pointer_file_ref",
        "fundamental_generation_manifest_file_ref",
    ):
        reference = normalized["files"][field]
        relative = f"fundamental_replay/{reference['relative_path']}"
        target = _destination(staging_root, relative)
        copy_one(field, reference, target)
        copied[field] = relative
    for field, prefix in (("market_table_file_refs", "closure/market_tables"),):
        rows: list[str] = []
        for index, reference in enumerate(normalized[field]):
            suffix = PurePosixPath(reference["relative_path"]).suffix.lower()
            relative = f"{prefix}/{index:04d}{suffix}"
            target = _destination(staging_root, relative)
            copy_one(field, reference, target)
            rows.append(relative)
        copied[field] = rows
    for field in (
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
    ):
        rows = []
        for reference in normalized[field]:
            relative = "calendar_replay/" + reference["relative_path"]
            target = _destination(staging_root, relative)
            copy_one(field, reference, target)
            rows.append(relative)
        copied[field] = rows
    for field in (
        "trusted_provider_calendar_capability_file_ref",
        "trusted_provider_calendar_capture_transaction_file_ref",
        "trusted_provider_calendar_capture_execution_file_ref",
        "trusted_provider_calendar_capture_success_file_ref",
        "trusted_provider_release_install_input_file_ref",
    ):
        reference = normalized[field]
        if reference is not None:
            relative = "calendar_replay/" + reference["relative_path"]
            copy_one(field, reference, _destination(staging_root, relative))
            copied[field] = relative
    for field in ("fundamental_table_file_refs", "fundamental_evidence_file_refs"):
        rows = []
        for reference in normalized[field]:
            relative = f"fundamental_replay/{reference['relative_path']}"
            target = _destination(staging_root, relative)
            copy_one(field, reference, target)
            rows.append(relative)
        copied[field] = rows
    return copied


def _copied_path(copied: Mapping[str, list[str] | str], field: str) -> str:
    value = copied.get(field)
    if type(value) is not str:
        raise SystemContractError(f"copied {field} is not a scalar path")
    return value


def _copied_paths(copied: Mapping[str, list[str] | str], field: str) -> list[str]:
    value = copied.get(field)
    if type(value) is not list or any(type(row) is not str for row in value):
        raise SystemContractError(f"copied {field} is not a path list")
    return list(value)


def _coverage_document(value: Any, *, label: str) -> dict[str, Any]:
    coverage = _mapping(value, label=label)
    if (
        coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4"
        or coverage.get("complete") is not True
        or coverage.get("coverage_ratio") != 1.0
        or coverage.get("blocking_incomplete_count") != 0
        or coverage.get("categories_checked") != ["full_a"]
        or coverage.get("classification_sets_disjoint") is not True
        or coverage.get("true_missing_symbols") != []
    ):
        raise SystemContractError("market coverage is not exact clean full-A closure")
    return coverage


def _validate_market_closure(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    staging_root: Path,
    copied: Mapping[str, list[str] | str],
) -> dict[str, Any]:
    scope = _read_copied_json(staging_root, copied, "market_scope_file_ref")
    scope_symbols = _symbols(scope.get("full_a"), label="market scope full_a")
    stats = _mapping(scope.get("stats"), label="market scope stats")
    if stats.get("full_a") != len(scope_symbols):
        raise SystemContractError("market scope count differs")
    for alias in ("full_market", "all_a", "all"):
        if alias in scope and scope[alias] != scope_symbols:
            raise SystemContractError("market scope aliases differ")
    scope_sha = _symbol_set_sha256(scope_symbols)

    pointer = _read_copied_json(staging_root, copied, "market_pointer_file_ref")
    manifest = _read_copied_json(staging_root, copied, "market_snapshot_manifest_file_ref")
    if (
        pointer.get("status") != "OK"
        or pointer.get("blockers") != []
        or manifest.get("status") != "OK"
        or manifest.get("blockers") != []
        or pointer.get("snapshot_id") != manifest.get("snapshot_id")
    ):
        raise SystemContractError("market pointer/snapshot state differs")
    snapshot_id = _identifier(pointer.get("snapshot_id"), label="snapshot_id")
    manifest_path = pointer.get("manifest_path")
    if type(manifest_path) is not str or PurePosixPath(manifest_path).name != f"{snapshot_id}.json":
        raise SystemContractError("market pointer manifest path differs")
    pointer_coverage = _coverage_document(pointer.get("coverage"), label="market pointer coverage")
    manifest_coverage = _coverage_document(
        manifest.get("coverage"), label="market snapshot coverage"
    )
    if canonical_json_bytes(pointer_coverage) != canonical_json_bytes(manifest_coverage):
        raise SystemContractError("market pointer/snapshot coverage differs")
    cutoff = _compact_date(
        pointer.get("latest_complete_trade_date"),
        label="market latest_complete_trade_date",
    )
    if (
        cutoff != manifest.get("latest_complete_trade_date")
        or cutoff != pointer_coverage.get("latest_complete_trade_date")
        or cutoff != pointer_coverage.get("coverage_trade_date")
        or cutoff != pointer_coverage.get("upsert_target_trade_date")
        or pointer.get("latest_trade_date") != cutoff
        or manifest.get("latest_trade_date") != cutoff
    ):
        raise SystemContractError("market cutoff binding differs")
    if (
        pointer_coverage.get("expected_scope_count") != len(scope_symbols)
        or pointer_coverage.get("coverage_complete_count") != len(scope_symbols)
        or pointer_coverage.get("expected_scope_sha256") != scope_sha
    ):
        raise SystemContractError("market expected scope binding differs")
    suspended = _symbols(
        pointer_coverage.get("suspended_symbols"),
        label="suspended symbols",
        minimum=0,
    )
    inactive = _symbols(
        pointer_coverage.get("inactive_symbols"),
        label="inactive symbols",
        minimum=0,
    )
    absent = _symbols(
        pointer_coverage.get("non_blocking_absent_symbols"),
        label="non-blocking absent symbols",
        minimum=0,
    )
    if (
        set(suspended) & set(inactive)
        or sorted([*suspended, *inactive]) != absent
        or not set(absent) <= set(scope_symbols)
    ):
        raise SystemContractError("market absent classifications differ")
    eligible = sorted(set(scope_symbols) - set(absent))
    if pointer_coverage.get("observed_bar_count") != len(eligible):
        raise SystemContractError("market observed cohort count differs")
    if (
        pointer_coverage.get("pit_membership_sha256")
        != normalized["files"]["pit_membership_file_ref"]["byte_sha256"]
        or pointer_coverage.get("pit_generation_manifest_sha256")
        != normalized["files"]["pit_generation_manifest_file_ref"]["byte_sha256"]
    ):
        raise SystemContractError("market/PIT byte binding differs")
    return {
        "pointer": pointer,
        "manifest": manifest,
        "coverage": pointer_coverage,
        "scope_symbols": scope_symbols,
        "scope_sha256": scope_sha,
        "eligible_symbols": eligible,
        "suspended_symbols": suspended,
        "inactive_symbols": inactive,
        "cutoff": cutoff,
    }


def _read_raw_market_rows(  # noqa: C901
    *,
    staging_root: Path,
    copied: Mapping[str, list[str] | str],
    closure: Mapping[str, Any],
    normalized: Mapping[str, Any],
    store: SystemStore | None = None,
    observed_inputs: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    required = {"ts_code", "trade_date", "adj_close", "amount", "vol", "total_mv"}
    scope = set(closure["scope_symbols"])
    eligible = set(closure["eligible_symbols"])
    cutoff = closure["cutoff"]
    rows: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, str]] = set()
    cutoff_market_caps: dict[str, float] = {}
    sessions: set[str] = set()
    paths = _copied_paths(copied, "market_table_file_refs")
    refs = normalized["market_table_file_refs"]
    if len(paths) != len(refs):
        raise SystemContractError("raw market table input cardinality differs")
    total_input_rows = 0
    for ordinal, (relative, input_ref) in enumerate(zip(paths, refs, strict=True)):
        if observed_inputs is None:
            iterator = _iter_bounded_parquet_rows(
                path=staging_root / relative,
                expected_sha256=input_ref["byte_sha256"],
                columns=sorted(required),
            )
        else:
            if store is None:
                raise SystemContractError("raw market governed store is absent")
            iterator = _iter_bounded_parquet_rows(
                store=store,
                source_ref=observed_inputs[("market_table_file_refs", ordinal)],
                columns=sorted(required),
            )
        for index, row in enumerate(iterator):
            total_input_rows += 1
            if total_input_rows > MAXIMUM_DECODED_SOURCE_ROWS:
                raise SystemSecurityError("raw market replay exceeds row bound")
            symbol = row["ts_code"]
            trade_date = row["trade_date"]
            if (
                type(symbol) is not str
                or _A_SHARE_SYMBOL_RE.fullmatch(symbol) is None
                or type(trade_date) is not str
                or re.fullmatch(r"[0-9]{8}", trade_date) is None
            ):
                raise SystemContractError("raw market key is not canonical")
            _compact_date(trade_date, label=f"raw market trade_date[{index}]")
            if trade_date > cutoff or symbol not in scope:
                continue
            key = (trade_date, symbol)
            if key in observed_keys:
                raise SystemContractError("raw market contains duplicate keys")
            observed_keys.add(key)
            if symbol not in eligible:
                continue
            converted: dict[str, float | None] = {}
            for field in ("adj_close", "amount", "vol", "total_mv"):
                value = row[field]
                if value is None:
                    converted[field] = None
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SystemContractError("raw market numeric type is invalid")
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise SystemContractError("raw market numeric value is not finite")
                converted[field] = numeric
            if (
                converted["adj_close"] is None
                or converted["amount"] is None
                or converted["adj_close"] <= 0
                or converted["amount"] <= 0
                or (converted["vol"] is not None and converted["vol"] < 0)
            ):
                raise SystemContractError("raw market factor value is invalid")
            sessions.add(trade_date)
            rows.append(
                {
                    "trade_date": datetime.strptime(trade_date, "%Y%m%d").date(),
                    "symbol": symbol,
                    "adj_close": converted["adj_close"],
                    "amount": converted["amount"],
                    "vol": converted["vol"],
                }
            )
            if trade_date == cutoff:
                total_mv = converted["total_mv"]
                if total_mv is None or total_mv <= 0:
                    raise SystemContractError("raw market cutoff capitalization is invalid")
                cutoff_market_caps[symbol] = total_mv
    sorted_sessions = sorted(sessions)
    cutoff_symbols = sorted(cutoff_market_caps)
    if (
        len(sorted_sessions) < _MINIMUM_MARKET_SESSIONS
        or not sorted_sessions
        or sorted_sessions[-1] != cutoff
        or cutoff_symbols != closure["eligible_symbols"]
    ):
        raise SystemContractError("raw market session/cohort closure differs")
    rows.sort(key=lambda row: (row["trade_date"], row["symbol"]))
    return rows, cutoff_market_caps, sorted_sessions


def _read_raw_pit_rows(  # noqa: C901
    *,
    staging_root: Path,
    copied: Mapping[str, list[str] | str],
    closure: Mapping[str, Any],
    cutoff_market_caps: Mapping[str, float],
    normalized: Mapping[str, Any],
    store: SystemStore | None = None,
    observed_inputs: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    manifest = _read_copied_json(staging_root, copied, "pit_generation_manifest_file_ref")
    membership_path = staging_root / _copied_path(copied, "pit_membership_file_ref")
    membership_sha, _ = _file_digest(membership_path)
    coverage = closure["coverage"]
    if (
        manifest.get("generation_id") != coverage.get("pit_generation_id")
        or manifest.get("canonical_sha256") != membership_sha
        or manifest.get("status_counts") is None
        or manifest.get("membership_quality_counts") != {"ok": manifest.get("row_count")}
    ):
        raise SystemContractError("PIT generation manifest binding differs")
    required = {
        "symbol",
        "industry",
        "source_list_status",
        "list_date",
        "delist_date",
        "membership_quality",
    }
    membership_input = normalized["files"]["pit_membership_file_ref"]
    if observed_inputs is None:
        iterator = _iter_bounded_parquet_rows(
            path=membership_path,
            expected_sha256=membership_input["byte_sha256"],
            columns=sorted(required),
        )
    else:
        if store is None:
            raise SystemContractError("PIT governed store is absent")
        iterator = _iter_bounded_parquet_rows(
            store=store,
            source_ref=observed_inputs[("pit_membership_file_ref", 0)],
            columns=sorted(required),
        )
    records: dict[str, dict[str, Any]] = {}
    scope = set(closure["scope_symbols"])
    for row in iterator:
        symbol = row["symbol"]
        if symbol not in scope:
            continue
        if symbol in records:
            raise SystemContractError("PIT membership contains duplicate scope symbols")
        if row["membership_quality"] != "ok":
            raise SystemContractError("PIT membership quality is not exact")
        records[symbol] = row
    if sorted(records) != closure["scope_symbols"]:
        raise SystemContractError("PIT membership/scope symbol closure differs")
    cutoff = closure["cutoff"]
    eligible = set(closure["eligible_symbols"])
    suspended = set(closure["suspended_symbols"])
    output: list[dict[str, Any]] = []
    for symbol in closure["scope_symbols"]:
        record = records[symbol]
        list_date = record["list_date"]
        delist_date = record["delist_date"]
        active_listing = (
            type(list_date) is str
            and re.fullmatch(r"[0-9]{8}", list_date) is not None
            and list_date <= cutoff
            and (not delist_date or delist_date > cutoff)
            and record["source_list_status"] == "L"
        )
        if symbol in eligible and not active_listing:
            raise SystemContractError("eligible PIT symbol is not actively listed")
        if symbol in suspended and not active_listing:
            raise SystemContractError("suspended PIT symbol is not actively listed")
        industry = record["industry"]
        if industry == "":
            industry = None
        if industry is not None and type(industry) is not str:
            raise SystemContractError("PIT industry is invalid")
        output.append(
            {
                "signal_session": datetime.strptime(cutoff, "%Y%m%d").date(),
                "symbol": symbol,
                "industry": industry,
                "total_mv": cutoff_market_caps.get(symbol),
                "tradable": symbol in eligible,
            }
        )
    return output


def _materialize_market_and_pit(
    *,
    normalized: Mapping[str, Any],
    staging_root: Path,
    copied: Mapping[str, list[str] | str],
) -> dict[str, Any]:
    closure = _validate_market_closure(
        normalized=normalized,
        staging_root=staging_root,
        copied=copied,
    )
    market_rows, market_caps, sessions = _read_raw_market_rows(
        staging_root=staging_root,
        copied=copied,
        closure=closure,
        normalized=normalized,
    )
    pit_rows = _read_raw_pit_rows(
        staging_root=staging_root,
        copied=copied,
        closure=closure,
        cutoff_market_caps=market_caps,
        normalized=normalized,
    )
    market_path = _destination(staging_root, "bootstrap/market_history.parquet")
    pit_path = _destination(staging_root, "bootstrap/pit_universe.parquet")
    market_sha = _write_parquet_once(
        market_path, rows=market_rows, schema=role_schema("market_history")
    )
    pit_sha = _write_parquet_once(pit_path, rows=pit_rows, schema=role_schema("pit_universe"))
    return {
        **closure,
        "market_history_relative": "bootstrap/market_history.parquet",
        "market_history_sha256": market_sha,
        "pit_universe_relative": "bootstrap/pit_universe.parquet",
        "pit_universe_sha256": pit_sha,
        "market_sessions": sessions,
    }


class _FundamentalSealedFileset:
    """Resolve every safe-successor read to one predeclared immutable input."""

    _LIST_FIELDS: Final = (
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
        "market_table_file_refs",
        "fundamental_table_file_refs",
        "fundamental_evidence_file_refs",
    )

    def __init__(
        self,
        *,
        normalized: Mapping[str, Any],
        staging_root: Path,
        copied: Mapping[str, list[str] | str],
        store: SystemStore | None,
        observed_inputs: Mapping[tuple[str, int], Mapping[str, Any]] | None,
    ) -> None:
        if (store is None) != (observed_inputs is None):
            raise SystemContractError("Fundamental sealed-fileset authority is incomplete")
        self._store = store
        self._entries: dict[str, dict[str, Any]] = {}
        self._by_sha: dict[str, list[dict[str, Any]]] = {}

        def add(field: str, ordinal: int, path_text: str, reference: Mapping[str, Any]) -> None:
            path = Path(os.path.abspath(staging_root / path_text))
            digest = reference["byte_sha256"]
            source_ref = (
                None
                if observed_inputs is None
                else validate_object_ref(
                    observed_inputs[(field, ordinal)],
                    label=f"Fundamental sealed {field}[{ordinal}]",
                )
            )
            entry = {
                "path": path,
                "byte_sha256": digest,
                "source_ref": source_ref,
            }
            key = str(path)
            if key in self._entries:
                raise SystemContractError("Fundamental sealed path is duplicated")
            self._entries[key] = entry
            self._by_sha.setdefault(digest, []).append(entry)

        for field, reference in normalized["files"].items():
            add(field, 0, _copied_path(copied, field), reference)
        for field in self._LIST_FIELDS:
            paths = _copied_paths(copied, field)
            references = normalized[field]
            if len(paths) != len(references):
                raise SystemContractError("Fundamental sealed list cardinality differs")
            for ordinal, (path_text, reference) in enumerate(zip(paths, references, strict=True)):
                add(field, ordinal, path_text, reference)

    def _entry(self, path: Path, expected_sha256: str) -> dict[str, Any]:
        key = str(Path(os.path.abspath(path)))
        exact = self._entries.get(key)
        if exact is not None:
            if exact["byte_sha256"] != expected_sha256:
                raise SystemSecurityError("Fundamental sealed path SHA differs")
            return exact
        matches = self._by_sha.get(expected_sha256, [])
        if not matches:
            raise SystemContractError("Fundamental sealed source is not declared")
        if len(matches) != 1:
            raise SystemSecurityError("Fundamental sealed source path is not uniquely bound")
        return matches[0]

    def expected_sha256(self, path: Path, *, fallback_sha256: str | None = None) -> str:
        key = str(Path(os.path.abspath(path)))
        exact = self._entries.get(key)
        if exact is not None:
            if fallback_sha256 is not None and exact["byte_sha256"] != fallback_sha256:
                raise SystemSecurityError("Fundamental expected SHA differs")
            return cast(str, exact["byte_sha256"])
        if type(fallback_sha256) is not str:
            raise SystemSecurityError("Fundamental source path is not sealed")
        self._entry(path, fallback_sha256)
        return fallback_sha256

    def read_bytes(
        self,
        path: Path,
        *,
        expected_sha256: str,
        maximum_bytes: int,
    ) -> bytes:
        entry = self._entry(path, expected_sha256)
        source_ref = entry["source_ref"]
        if source_ref is not None:
            if self._store is None:  # pragma: no cover - constructor invariant
                raise SystemContractError("Fundamental System source store is absent")
            payload, raw = self._store.read_source_object_bytes(
                source_ref,
                maximum_bytes=min(maximum_bytes, MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES),
            )
            if (
                payload["byte_sha256"] != expected_sha256
                or payload["relative_path"]
                != entry["path"].relative_to(self._store.source_root).as_posix()
            ):
                raise SystemSecurityError("Fundamental source object binding differs")
            return raw
        metadata = os.lstat(entry["path"])
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > maximum_bytes
        ):
            raise SystemSecurityError("Fundamental copied source security differs")
        raw = _read_stable_bytes(entry["path"], maximum_bytes=maximum_bytes)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise SystemSecurityError("Fundamental copied source SHA differs")
        return raw

    @contextmanager
    def open_parquet(
        self,
        path: Path,
        *,
        expected_sha256: str,
        maximum_bytes: int,
        decoded_reservation_bytes: int,
    ) -> Iterator[FileIO]:
        if (
            type(decoded_reservation_bytes) is not int
            or decoded_reservation_bytes <= 0
            or decoded_reservation_bytes > _PARQUET_BATCH_BYTES
        ):
            raise SystemSecurityError("Fundamental decoded reservation differs")
        entry = self._entry(path, expected_sha256)
        source_ref = entry["source_ref"]
        if source_ref is not None:
            if self._store is None:  # pragma: no cover - constructor invariant
                raise SystemContractError("Fundamental System source store is absent")
            with self._store.open_source_object(
                source_ref,
                maximum_bytes=min(maximum_bytes, MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES),
                decoded_reservation_bytes=decoded_reservation_bytes,
            ) as (payload, stream):
                if (
                    payload["byte_sha256"] != expected_sha256
                    or payload["relative_path"]
                    != entry["path"].relative_to(self._store.source_root).as_posix()
                ):
                    raise SystemSecurityError("Fundamental Parquet source binding differs")
                yield stream
            return
        with _open_stable_parquet_path(entry["path"], expected_sha256=expected_sha256) as stream:
            if os.fstat(stream.fileno()).st_size > maximum_bytes:
                raise SystemSecurityError("Fundamental Parquet source byte bound exceeded")
            yield stream

    def files_under(self, root: Path) -> list[Path]:
        normalized_root = Path(os.path.abspath(root))
        return sorted(
            entry["path"]
            for entry in self._entries.values()
            if normalized_root in entry["path"].parents
        )


def _validate_fundamental_closure(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    staging_root: Path,
    copied: Mapping[str, list[str] | str],
    expected_cutoff: str,
    store: SystemStore | None = None,
    observed_inputs: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    sealed_fileset = _FundamentalSealedFileset(
        normalized=normalized,
        staging_root=staging_root,
        copied=copied,
        store=store,
        observed_inputs=observed_inputs,
    )

    def read_json(field: str) -> dict[str, Any]:
        reference = normalized["files"][field]
        raw = sealed_fileset.read_bytes(
            staging_root / _copied_path(copied, field),
            expected_sha256=reference["byte_sha256"],
            maximum_bytes=_JSON_MAX_BYTES,
        )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemContractError(f"Fundamental {field} is invalid JSON") from exc
        if type(value) is not dict:
            raise SystemContractError(f"Fundamental {field} must be an object")
        return value

    pointer = read_json("fundamental_pointer_file_ref")
    manifest = read_json("fundamental_generation_manifest_file_ref")
    generation_id = pointer.get("generation_id")
    if (
        type(generation_id) is not str
        or not generation_id
        or pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or manifest.get("schema_version") != "cn-fundamental-generation.v1"
        or pointer.get("status") != "OK"
        or manifest.get("status") != "OK"
        or manifest.get("generation_id") != generation_id
    ):
        raise SystemContractError("Fundamental pointer/generation state differs")
    manifest_path = pointer.get("manifest_path")
    if (
        type(manifest_path) is not str
        or manifest_path
        != str(PurePosixPath("_fundamental_generations") / generation_id / "manifest.json")
        or normalized["files"]["fundamental_pointer_file_ref"]["relative_path"]
        != "_fundamental_latest.json"
        or normalized["files"]["fundamental_generation_manifest_file_ref"]["relative_path"]
        != manifest_path
    ):
        raise SystemContractError("Fundamental manifest path binding differs")
    pointer_tables = _mapping(pointer.get("tables"), label="Fundamental pointer tables")
    manifest_tables = _mapping(manifest.get("tables"), label="Fundamental generation tables")
    expected_names = [
        "fundamental_daily",
        "fundamental_period",
        "fundamental_quarantine",
    ]
    if sorted(pointer_tables) != expected_names or sorted(manifest_tables) != expected_names:
        raise SystemContractError("Fundamental table identities differ")
    table_refs = normalized["fundamental_table_file_refs"]
    refs_by_name: dict[str, dict[str, str]] = {}
    for reference in table_refs:
        filename = PurePosixPath(reference["relative_path"]).name
        matches = [name for name in expected_names if filename == f"{name}.parquet"]
        if len(matches) != 1 or matches[0] in refs_by_name:
            raise SystemContractError("Fundamental table input path differs")
        refs_by_name[matches[0]] = reference
    if sorted(refs_by_name) != expected_names:
        raise SystemContractError("Fundamental table inputs are incomplete")
    for name in expected_names:
        table_path = pointer_tables[name]
        table_manifest = _mapping(manifest_tables[name], label=f"Fundamental table {name}")
        expected_sha = refs_by_name[name]["byte_sha256"]
        if (
            type(table_path) is not str
            or table_path
            != str(PurePosixPath("_fundamental_generations") / generation_id / f"{name}.parquet")
            or refs_by_name[name]["relative_path"] != table_path
            or table_manifest.get("sha256") != expected_sha
        ):
            raise SystemContractError("Fundamental table byte binding differs")

    replay_root = (
        staging_root / PurePosixPath(_copied_path(copied, "fundamental_pointer_file_ref")).parent
    )
    try:
        validated = validate_successor_provenance(
            pointer,
            manifest,
            generation_root=replay_root,
            historical_only=True,
            sealed_fileset=sealed_fileset,
        )
    except (SafeSuccessorError, SystemContractError, OSError, ValueError) as exc:
        raise SystemContractError(
            "Fundamental safe-successor provenance validation failed"
        ) from exc

    evidence_prefix = PurePosixPath("_fundamental_generations") / generation_id
    expected_evidence: dict[str, str] = {}
    for reference in normalized["fundamental_evidence_file_refs"]:
        path = PurePosixPath(reference["relative_path"])
        try:
            relative = path.relative_to(evidence_prefix)
        except ValueError as exc:
            raise SystemContractError("Fundamental evidence path is noncanonical") from exc
        if not relative.parts or relative.parts[0] != "provider_evidence":
            raise SystemContractError("Fundamental evidence path is noncanonical")
        expected_evidence[str(relative)] = reference["byte_sha256"]
    if validated["provider_evidence_files"] != expected_evidence:
        raise SystemContractError("Fundamental provider evidence fileset differs")

    target_bindings = validated["target_bindings"]
    expected_target_shas = {
        "market_pointer": normalized["files"]["market_pointer_file_ref"]["byte_sha256"],
        "pit_pointer": normalized["files"]["pit_pointer_file_ref"]["byte_sha256"],
        "pit_membership": normalized["files"]["pit_membership_file_ref"]["byte_sha256"],
        "expected_scope": normalized["files"]["market_scope_file_ref"]["byte_sha256"],
    }
    if validated["target_cutoff"] != expected_cutoff or any(
        target_bindings.get(name, {}).get("sha256") != digest
        for name, digest in expected_target_shas.items()
    ):
        raise SystemContractError("Fundamental target source binding differs")
    if manifest_tables["fundamental_daily"]["rows"] <= 0 or (
        manifest_tables["fundamental_period"]["rows"] <= 0
    ):
        raise SystemContractError("Fundamental production tables are empty")
    return {
        "generation_id": generation_id,
        "table_sha256s": validated["table_sha256"],
        "provenance_binding_sha256": validated["provenance_binding_sha256"],
        "machine_states": validated["machine_states"],
    }


def _derive_source_blockers(fundamental_closure: Mapping[str, Any]) -> list[str]:
    machine_states = fundamental_closure.get("machine_states")
    expected = {
        "mixed": True,
        "legacy_direct_reader_provenance": "limited",
        "binding_aware_research_ready": True,
        "homogeneous_history_ready": False,
    }
    if type(machine_states) is not dict or machine_states != expected:
        raise SystemContractError("Fundamental machine-state projection is unsupported")
    return sorted(FUNDAMENTAL_SOURCE_BLOCKERS)


def _calendar_date(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not an ISO date")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemContractError(f"{label} is not an ISO date") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise SystemContractError(f"{label} is not an ISO date")
    return value


def _symbol_exchanges(symbols: Sequence[str]) -> list[str]:
    exchanges: set[str] = set()
    for symbol in symbols:
        matches = [
            exchange for suffix, exchange in _EXCHANGE_BY_SUFFIX.items() if symbol.endswith(suffix)
        ]
        if len(matches) != 1:
            raise SystemContractError("strict PIT contains an unsupported exchange identity")
        exchanges.add(matches[0])
    return sorted(exchanges)


def _reject_legacy_calendar_capture(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    input_root: Path,
    capture_ref: Mapping[str, str],
    raw_ref: Mapping[str, str],
    exchange: str,
    issuer: str,
    source_url: str,
    captured_at: str,
    evidence_role: EvidenceRole,
    coverage_start: str,
    cutoff: str,
    sessions: Sequence[str],
    session_intervals: Sequence[Mapping[str, str]],
    transform_code_sha256: str,
) -> dict[str, Any]:
    raise SystemPreconditionError("legacy DAILY_STATUS calendar captures are not admitted")


def _legacy_calendar_capture_quarantine(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    input_root: Path,
    capture_ref: Mapping[str, str],
    raw_ref: Mapping[str, str],
    exchange: str,
    issuer: str,
    source_url: str,
    captured_at: str,
    evidence_role: EvidenceRole,
    coverage_start: str,
    cutoff: str,
    sessions: Sequence[str],
    session_intervals: Sequence[Mapping[str, str]],
    transform_code_sha256: str,
) -> dict[str, Any]:
    capture_path, _ = _verify_input(input_root, capture_ref)
    capture_raw = _read_stable_bytes(capture_path, maximum_bytes=_JSON_MAX_BYTES)
    try:
        capture = validate_artifact(
            capture_raw,
            expected_kind=_CALENDAR_CAPTURE_KIND,
            expected_contract_sha256=_CALENDAR_CAPTURE_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemContractError("official calendar capture contract failed") from exc
    payload = capture["payload"]
    if set(payload) != _CALENDAR_CAPTURE_FIELDS:
        raise SystemContractError("official calendar capture fields are not exact")
    request_url = payload["request_url"]
    effective_url = payload["effective_url"]
    redirect_chain = payload["redirect_chain"]
    issuer_host = _OFFICIAL_CALENDAR_AUTHORITIES[exchange][1]
    if (
        type(request_url) is not str
        or type(effective_url) is not str
        or type(redirect_chain) is not list
        or any(type(item) is not str for item in redirect_chain)
    ):
        raise SystemContractError("official calendar HTTP capture metadata is invalid")
    url_chain = [request_url, *redirect_chain]
    if not redirect_chain or redirect_chain[-1] != effective_url:
        url_chain.append(effective_url)
    if any(
        (parsed := urlsplit(item)).scheme != "https"
        or parsed.hostname != issuer_host
        or parsed.username is not None
        or parsed.password is not None
        for item in url_chain
    ):
        raise SystemContractError("official calendar HTTP redirect authority differs")
    raw_path, raw_stored_size = _verify_input(input_root, raw_ref)
    raw = _read_stable_bytes(raw_path, maximum_bytes=_JSON_MAX_BYTES)
    media_type = payload["raw_media_type"]
    if type(media_type) is not str or not media_type or media_type.strip() != media_type:
        raise SystemContractError("official calendar raw media type is invalid")
    code_sha = decoder_code_sha256()
    if (
        raw_stored_size != len(raw)
        or payload["raw_sha256"] != _sha256(raw)
        or payload["raw_sha256"] != raw_ref["byte_sha256"]
        or payload["raw_byte_length"] != len(raw)
        or payload["decoder_id"] != decoder_id(exchange, evidence_role)
        or payload["decoder_sha256"] != code_sha
    ):
        raise SystemContractError("official calendar raw/decoder identity differs")
    projection = dict(
        decode_capture_projection(exchange, evidence_role, raw, media_type=media_type)
    )
    projection_sha = _sha256(canonical_json_bytes(projection))
    if (
        payload["state"] != "IMMUTABLE"
        or payload["evidence_role"] != evidence_role
        or payload["exchange_id"] != exchange
        or payload["issuer"] != issuer
        or payload["source_url"] != source_url
        or request_url != source_url
        or payload["http_status"] != 200
        or payload["issuer_host"] != issuer_host
        or payload["tls_verified"] is not True
        or payload["captured_at"] != captured_at
        or payload["raw_file_ref"] != raw_ref
        or payload["timezone"] != "Asia/Shanghai"
        or payload["coverage_start_date"] != coverage_start
        or payload["cutoff_date"] != cutoff
        or payload["transform_code_sha256"] != transform_code_sha256
        or transform_code_sha256 != code_sha
        or payload["projection_sha256"] != projection_sha
    ):
        raise SystemContractError("official calendar capture binding differs")
    _identifier(payload["calendar_capture_id"], label="calendar_capture_id")
    if evidence_role == "DAILY_STATUS":
        rows = projection.get("daily_status_rows")
        if payload["daily_status_rows"] != rows or payload["session_intervals"] != []:
            raise SystemContractError("official daily projection binding differs")
        expected_date = datetime.strptime(coverage_start, "%Y-%m-%d").date()
        cutoff_date = datetime.strptime(cutoff, "%Y-%m-%d").date()
        observed_open: list[str] = []
        closed_count = 0
        if type(rows) is not list:
            raise SystemContractError("official daily OPEN/CLOSED evidence is absent")
        for index, row in enumerate(rows):
            if type(row) is not dict or set(row) != _CALENDAR_STATUS_FIELDS:
                raise SystemContractError("official daily status fields are not exact")
            date_text = _calendar_date(row["date"], label=f"daily_status_rows[{index}].date")
            if date_text != expected_date.isoformat() or expected_date > cutoff_date:
                raise SystemContractError("official daily status coverage is not consecutive")
            if row["status"] == "OPEN":
                observed_open.append(date_text)
            elif row["status"] == "CLOSED":
                closed_count += 1
            else:
                raise SystemContractError("official daily status is not OPEN/CLOSED")
            expected_date += timedelta(days=1)
        if (
            expected_date != cutoff_date + timedelta(days=1)
            or observed_open != list(sessions)
            or closed_count <= 0
        ):
            raise SystemContractError("official daily OPEN/CLOSED calendar differs")
    elif evidence_role == "SESSION_RULE":
        intervals = projection.get("session_intervals")
        if (
            payload["daily_status_rows"] != []
            or payload["session_intervals"] != intervals
            or intervals != list(session_intervals)
        ):
            raise SystemContractError("official session-rule projection binding differs")
    else:
        raise SystemContractError("official calendar evidence role is unsupported")
    if capture_ref not in normalized["official_calendar_capture_file_refs"]:
        raise SystemContractError("official calendar capture is outside request closure")
    return capture


def _reject_legacy_calendar_manifest(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    input_root: Path,
    cohort_symbols: Sequence[str],
    source_projection: Mapping[str, Any],
) -> dict[str, Any]:
    raise SystemPreconditionError("legacy calendar manifest authority is not admitted")


def _legacy_calendar_manifest_quarantine(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    input_root: Path,
    cohort_symbols: Sequence[str],
    source_projection: Mapping[str, Any],
) -> dict[str, Any]:
    reference = normalized["files"]["calendar_manifest_file_ref"]
    path, _ = _verify_input(input_root, reference)
    raw = _read_stable_bytes(path, maximum_bytes=_JSON_MAX_BYTES)
    if _sha256(raw) != reference["byte_sha256"]:
        raise SystemSecurityError("calendar manifest changed during exact read")
    try:
        manifest = validate_artifact(
            raw,
            expected_kind=_CALENDAR_MANIFEST_KIND,
            expected_contract_sha256=_CALENDAR_MANIFEST_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemContractError("official calendar manifest contract failed") from exc
    payload = manifest["payload"]
    if set(payload) != _CALENDAR_MANIFEST_FIELDS:
        raise SystemContractError("official calendar manifest fields are not exact")
    if payload["state"] != "IMMUTABLE" or payload["timezone"] != "Asia/Shanghai":
        raise SystemContractError("official calendar manifest state/timezone is invalid")
    _identifier(payload["calendar_manifest_id"], label="calendar_manifest_id")
    if payload["transform_code_sha256"] != decoder_code_sha256():
        raise SystemContractError("calendar transform code differs from installed decoder")
    coverage_start = _calendar_date(payload["coverage_start_date"], label="coverage_start_date")
    cutoff = _calendar_date(payload["cutoff_date"], label="cutoff_date")
    if coverage_start > "2024-01-01":
        raise SystemContractError("official calendar coverage starts after 2024-01-01")
    if payload["calendar_file_ref"] != normalized["files"]["exchange_calendar_file_ref"]:
        raise SystemContractError("official calendar strict-file binding differs")
    calendar_projection = source_projection.get("calendar")
    if type(calendar_projection) is not dict:
        raise SystemContractError("strict calendar projection is absent")
    sessions = calendar_projection.get("open_sessions")
    opens = calendar_projection.get("opens_at_utc")
    closes = calendar_projection.get("closes_at_utc")
    if (
        type(sessions) is not list
        or type(opens) is not list
        or type(closes) is not list
        or not sessions
        or len(sessions) != len(opens)
        or len(sessions) != len(closes)
        or sessions != sorted(set(sessions))
        or sessions[0] < coverage_start
        or sessions[-1] != cutoff
        or cutoff != source_projection.get("pit_signal_session")
        or cutoff != source_projection.get("latest_market_session")
    ):
        raise SystemContractError("calendar/market/PIT cutoff closure differs")
    session_sha = _sha256(canonical_json_bytes(sessions))
    market_sessions = source_projection.get("market_sessions")
    if (
        type(market_sessions) is not list
        or len(market_sessions) < _MINIMUM_MARKET_SESSIONS
        or not set(market_sessions) <= set(sessions)
    ):
        raise SystemContractError("strict market sessions are outside official calendar")
    local_zone = ZoneInfo("Asia/Shanghai")
    for index, (session, opens_at, closes_at) in enumerate(
        zip(sessions, opens, closes, strict=True)
    ):
        try:
            local_open = datetime.fromisoformat(opens_at).astimezone(local_zone)
            local_close = datetime.fromisoformat(closes_at).astimezone(local_zone)
        except (TypeError, ValueError) as exc:
            raise SystemContractError("strict calendar timestamps are invalid") from exc
        if (
            local_open.strftime("%Y-%m-%d") != session
            or local_close.strftime("%Y-%m-%d") != session
            or local_open.strftime("%H:%M:%S") != _CN_CONTINUOUS_SESSION_INTERVALS[0]["opens_local"]
            or local_close.strftime("%H:%M:%S")
            != _CN_CONTINUOUS_SESSION_INTERVALS[-1]["closes_local"]
        ):
            raise SystemContractError(
                f"strict calendar session envelope differs at ordinal {index}"
            )
    rows = payload["exchange_rows"]
    if type(rows) is not list or not rows:
        raise SystemContractError("official exchange rows are absent")
    expected_exchanges = _symbol_exchanges(cohort_symbols)
    raw_refs = normalized["official_calendar_raw_file_refs"]
    observed_raw_refs: list[dict[str, str]] = []
    observed_capture_refs: list[dict[str, str]] = []
    observed_exchanges: list[str] = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != _CALENDAR_EXCHANGE_FIELDS:
            raise SystemContractError("official exchange row fields are not exact")
        exchange = row["exchange_id"]
        if exchange not in _OFFICIAL_CALENDAR_AUTHORITIES:
            raise SystemContractError("official exchange identity is unsupported")
        issuer, hostname = _OFFICIAL_CALENDAR_AUTHORITIES[exchange]
        if row["issuer"] != issuer:
            raise SystemContractError("official exchange source authority differs")
        source_rows: dict[str, tuple[str, str, dict[str, str], dict[str, str]]] = {}
        for role, prefix in (("DAILY_STATUS", "daily_status"), ("SESSION_RULE", "session_rule")):
            source_url = row[f"{prefix}_source_url"]
            parsed_url = urlsplit(source_url) if type(source_url) is str else None
            if (
                parsed_url is None
                or parsed_url.scheme != "https"
                or parsed_url.hostname != hostname
                or parsed_url.username is not None
                or parsed_url.password is not None
            ):
                raise SystemContractError("official exchange source authority differs")
            captured_at = _timestamp(
                row[f"{prefix}_captured_at"],
                label=f"exchange_rows[{index}].{prefix}_captured_at",
            )
            raw_ref = _file_ref(
                row[f"{prefix}_raw_file_ref"],
                label=f"exchange_rows[{index}].{prefix}_raw_file_ref",
            )
            capture_ref = _file_ref(
                row[f"{prefix}_capture_file_ref"],
                label=f"exchange_rows[{index}].{prefix}_capture_file_ref",
            )
            source_rows[role] = (source_url, captured_at, raw_ref, capture_ref)
        intervals = row["session_intervals"]
        if (
            type(intervals) is not list
            or any(
                type(item) is not dict or set(item) != _CALENDAR_INTERVAL_FIELDS
                for item in intervals
            )
            or intervals != _CN_CONTINUOUS_SESSION_INTERVALS
            or row["open_session_count"] != len(sessions)
            or len(sessions) < 391
            or row["open_session_sha256"] != session_sha
        ):
            raise SystemContractError("official exchange sessions differ")
        for role in (
            cast(EvidenceRole, "DAILY_STATUS"),
            cast(EvidenceRole, "SESSION_RULE"),
        ):
            source_url, captured_at, raw_ref, capture_ref = source_rows[role]
            _reject_legacy_calendar_capture(
                normalized=normalized,
                input_root=input_root,
                capture_ref=capture_ref,
                raw_ref=raw_ref,
                exchange=exchange,
                issuer=issuer,
                source_url=source_url,
                captured_at=captured_at,
                evidence_role=role,
                coverage_start=coverage_start,
                cutoff=cutoff,
                sessions=sessions,
                session_intervals=intervals,
                transform_code_sha256=payload["transform_code_sha256"],
            )
            observed_raw_refs.append(raw_ref)
            observed_capture_refs.append(capture_ref)
        observed_exchanges.append(exchange)
    if (
        observed_exchanges != expected_exchanges
        or observed_exchanges != sorted(set(observed_exchanges))
        or sorted(observed_raw_refs, key=lambda value: value["relative_path"]) != raw_refs
        or sorted(observed_capture_refs, key=lambda value: value["relative_path"])
        != normalized["official_calendar_capture_file_refs"]
    ):
        raise SystemContractError("official calendar/PIT exchange closure differs")
    return manifest


def _validate_calendar_compilation_closure(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    staging_root: Path,
    copied: Mapping[str, list[str] | str],
    cohort_symbols: Sequence[str],
    source_projection: Mapping[str, Any],
    release_ref: Mapping[str, Any],
    repository_root: Path,
    store: SystemStore | None = None,
    observed_inputs: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    validation_mode: str = "PRE_CAS_CURRENT",
) -> dict[str, Any]:
    """Replay the sole admitted calendar DAG from exact source bytes."""

    if (store is None) != (observed_inputs is None):
        raise SystemContractError("calendar replay authority is incomplete")

    def copied_relative(field: str, ordinal: int) -> str:
        if field in normalized["files"]:
            return _copied_path(copied, field)
        if field in {
            "trusted_provider_calendar_capability_file_ref",
            "trusted_provider_calendar_capture_transaction_file_ref",
            "trusted_provider_calendar_capture_execution_file_ref",
            "trusted_provider_calendar_capture_success_file_ref",
            "trusted_provider_release_install_input_file_ref",
        }:
            return _copied_path(copied, field)
        return _copied_paths(copied, field)[ordinal]

    size_rows: list[tuple[str, int, int]] = []
    for field, ordinal, reference in _calendar_reference_rows(normalized):
        relative = copied_relative(field, ordinal)
        if observed_inputs is not None:
            assert store is not None
            source_ref = observed_inputs[(field, ordinal)]
            artifact = store.get_object(source_ref)
            payload = artifact.get("payload")
            if (
                artifact.get("kind") != "system.source_object"
                or type(payload) is not dict
                or payload.get("source_root_id") != store.source_root_id
                or payload.get("relative_path") != relative
                or payload.get("byte_sha256") != reference["byte_sha256"]
            ):
                raise SystemContractError("calendar replay source descriptor differs")
        try:
            root = staging_root.resolve(strict=True)
            path = root.joinpath(relative).resolve(strict=True)
            path.relative_to(root)
            metadata = os.lstat(path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemSecurityError("calendar replay source path is unavailable") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise SystemSecurityError("calendar replay source storage is invalid")
        size_rows.append((field, ordinal, metadata.st_size))
    _validate_calendar_size_rows(size_rows)

    def source_bytes(field: str, ordinal: int = 0) -> bytes:
        if observed_inputs is not None:
            assert store is not None
            _artifact, raw = store.read_source_object_bytes(
                observed_inputs[(field, ordinal)],
                maximum_bytes=_CALENDAR_INPUT_BYTE_LIMITS[field],
            )
            return raw
        if field in normalized["files"] or field in {
            "trusted_provider_calendar_capability_file_ref",
            "trusted_provider_calendar_capture_transaction_file_ref",
            "trusted_provider_calendar_capture_execution_file_ref",
            "trusted_provider_calendar_capture_success_file_ref",
            "trusted_provider_release_install_input_file_ref",
        }:
            relative = _copied_path(copied, field)
        else:
            relative = _copied_paths(copied, field)[ordinal]
        return _read_stable_bytes(
            staging_root / relative,
            maximum_bytes=_CALENDAR_INPUT_BYTE_LIMITS[field],
        )

    raw_by_ref: dict[bytes, bytes] = {}
    for field in (
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
    ):
        for ordinal, reference in enumerate(normalized[field]):
            raw_by_ref[canonical_json_bytes(reference)] = source_bytes(field, ordinal)
    for field in (
        "exchange_calendar_file_ref",
        "calendar_authority_policy_file_ref",
        "calendar_runtime_json_file_ref",
        "calendar_compilation_file_ref",
    ):
        reference = normalized["files"][field]
        raw_by_ref[canonical_json_bytes(reference)] = source_bytes(field)
    capability_ref = normalized["trusted_provider_calendar_capability_file_ref"]
    if capability_ref is not None:
        raw_by_ref[canonical_json_bytes(capability_ref)] = source_bytes(
            "trusted_provider_calendar_capability_file_ref"
        )
    transaction_ref = normalized["trusted_provider_calendar_capture_transaction_file_ref"]
    if transaction_ref is not None:
        raw_by_ref[canonical_json_bytes(transaction_ref)] = source_bytes(
            "trusted_provider_calendar_capture_transaction_file_ref"
        )
    execution_ref = normalized["trusted_provider_calendar_capture_execution_file_ref"]
    if execution_ref is not None:
        raw_by_ref[canonical_json_bytes(execution_ref)] = source_bytes(
            "trusted_provider_calendar_capture_execution_file_ref"
        )
    success_ref = normalized["trusted_provider_calendar_capture_success_file_ref"]
    if success_ref is not None:
        raw_by_ref[canonical_json_bytes(success_ref)] = source_bytes(
            "trusted_provider_calendar_capture_success_file_ref"
        )
    release_input_ref = normalized["trusted_provider_release_install_input_file_ref"]
    if release_input_ref is not None:
        raw_by_ref[canonical_json_bytes(release_input_ref)] = source_bytes(
            "trusted_provider_release_install_input_file_ref"
        )

    def raw_resolver(reference: Mapping[str, Any]) -> bytes:
        normalized_value = dict(reference)
        declared_size = normalized_value.pop("size", None)
        if declared_size is not None and (type(declared_size) is not int or declared_size <= 0):
            raise SystemContractError("calendar replay file size is invalid")
        normalized_ref = _file_ref(normalized_value, label="calendar replay file ref")
        try:
            raw = raw_by_ref[canonical_json_bytes(normalized_ref)]
        except KeyError as exc:
            raise SystemContractError(
                "calendar compiler requested a file outside request closure"
            ) from exc
        if _sha256(raw) != normalized_ref["byte_sha256"]:
            raise SystemSecurityError("calendar compiler source bytes differ")
        if declared_size is not None and declared_size != len(raw):
            raise SystemSecurityError("calendar compiler source size differs")
        return raw

    def documents(field: str) -> list[dict[str, Any]]:
        return [
            _parse_source_json(
                raw_resolver(reference),
                label=f"{field}[{ordinal}]",
            )
            for ordinal, reference in enumerate(normalized[field])
        ]

    compilation_raw = source_bytes("calendar_compilation_file_ref")
    compilation_document = _parse_source_json(
        compilation_raw, label="calendar_compilation_file_ref"
    )
    if (
        _sha256(compilation_raw)
        != normalized["files"]["calendar_compilation_file_ref"]["byte_sha256"]
    ):
        raise SystemSecurityError("calendar compilation exact bytes differ")
    expected_exchanges = _symbol_exchanges(cohort_symbols)
    market_sessions = source_projection.get("market_sessions")
    if type(market_sessions) is not list:
        raise SystemContractError("calendar market-session projection is absent")
    policy_raw = source_bytes("calendar_authority_policy_file_ref")
    policy_document = _parse_source_json(policy_raw, label="calendar_authority_policy_file_ref")
    policy = validate_calendar_authority_policy(
        policy_document,
        pit_exchange_ids=(
            expected_exchanges
            if policy_document["payload"]["authority_route"] == "EXCHANGE_OFFICIAL"
            else None
        ),
    )
    route = policy["payload"]["authority_route"]
    if route == "EXCHANGE_OFFICIAL":
        capture_documents = documents("official_calendar_capture_file_refs")
        admission_documents = documents("official_calendar_decoder_admission_file_refs")
        index_documents = documents("official_calendar_index_closure_file_refs")
        required_raw_refs: list[dict[str, Any]] = [
            _file_ref(
                document["payload"]["raw_file_ref"],
                label="calendar capture raw_file_ref",
            )
            for document in capture_documents
        ]
        for document in admission_documents:
            fixture_ref = dict(document["payload"]["fixture_raw_file_ref"])
            fixture_ref.pop("size", None)
            required_raw_refs.append(
                _file_ref(fixture_ref, label="calendar admission fixture_raw_file_ref")
            )
        if sorted(map(canonical_json_bytes, required_raw_refs)) != sorted(
            map(canonical_json_bytes, normalized["official_calendar_raw_file_refs"])
        ):
            raise SystemContractError("official calendar raw source closure differs")
        if validation_mode == "PRE_CAS_CURRENT":
            compilation = validate_exchange_calendar_compilation(
                compilation_document,
                pit_exchange_ids=expected_exchanges,
                market_session_dates=market_sessions,
                capture_documents=capture_documents,
                admission_documents=admission_documents,
                index_closure_documents=index_documents,
                raw_resolver=raw_resolver,
                expected_release_ref=release_ref,
                expected_policy_ref=object_ref_for_artifact(policy),
            )
        elif validation_mode == "HISTORICAL":
            compilation_payload = compilation_document.get("payload")
            if type(compilation_payload) is not dict:
                raise SystemContractError("historical calendar compilation payload is absent")
            historical_compiler_sha = compilation_payload.get("compiler_code_sha256")
            if type(historical_compiler_sha) is not str:
                raise SystemContractError("historical calendar compiler SHA is absent")
            compilation = validate_historical_compilation_envelope(
                compilation_document,
                expected_release_ref=release_ref,
                expected_policy_ref=object_ref_for_artifact(policy),
                expected_compiler_code_sha256=historical_compiler_sha,
            )
        else:
            raise SystemContractError("calendar validation mode is invalid")
        source_limitations: list[str] = []
        expected_kind = "system.exchange_calendar_compilation"
    elif route == TRUSTED_PROVIDER_CALENDAR_ROUTE:
        if (
            capability_ref is None
            or transaction_ref is None
            or execution_ref is None
            or success_ref is None
            or release_input_ref is None
        ):
            raise SystemContractError("trusted-provider authority tombstone is invalid")
        capability_raw = source_bytes("trusted_provider_calendar_capability_file_ref")
        capability_document = _parse_source_json(
            capability_raw,
            label="trusted_provider_calendar_capability_file_ref",
        )
        capture_documents = documents("trusted_provider_calendar_capture_file_refs")
        docs_ref = capability_document["payload"]["docs_raw_file_ref"]
        docs_raw = raw_resolver(docs_ref)
        validate_trusted_provider_calendar_capability(
            capability_document,
            docs_raw=docs_raw,
            historical=validation_mode == "HISTORICAL",
        )
        required_provider_raw = [
            _file_ref(docs_ref, label="provider docs raw_file_ref"),
            *[
                _file_ref(
                    document["payload"]["raw_file_ref"],
                    label="provider calendar raw_file_ref",
                )
                for document in capture_documents
            ],
        ]
        if sorted(map(canonical_json_bytes, required_provider_raw)) != sorted(
            map(canonical_json_bytes, normalized["trusted_provider_calendar_raw_file_refs"])
        ):
            raise SystemContractError("trusted-provider calendar raw closure differs")
        transaction_document = _parse_source_json(
            source_bytes("trusted_provider_calendar_capture_transaction_file_ref"),
            label="trusted_provider_calendar_capture_transaction_file_ref",
        )
        validate_trusted_provider_calendar_capture_transaction(
            transaction_document,
            documentation_raw_file_ref=_file_ref(
                docs_ref,
                label="provider docs raw_file_ref",
            ),
            capability_file_ref=capability_ref,
            policy_file_ref=normalized["files"]["calendar_authority_policy_file_ref"],
            provider_raw_file_refs=[
                _file_ref(
                    document["payload"]["raw_file_ref"],
                    label="provider calendar raw_file_ref",
                )
                for document in capture_documents
            ],
            provider_capture_file_refs=normalized["trusted_provider_calendar_capture_file_refs"],
        )
        execution_document = _parse_source_json(
            source_bytes("trusted_provider_calendar_capture_execution_file_ref"),
            label="trusted_provider_calendar_capture_execution_file_ref",
        )
        release_install_input_raw = source_bytes("trusted_provider_release_install_input_file_ref")
        validate_trusted_provider_calendar_capture_execution(
            execution_document,
            release_install_input_raw=release_install_input_raw,
            repository_root=repository_root,
            documentation_raw_file_ref=_file_ref(
                docs_ref,
                label="provider docs raw_file_ref",
            ),
            capability_file_ref=capability_ref,
            policy_file_ref=normalized["files"]["calendar_authority_policy_file_ref"],
            provider_raw_file_refs=[
                _file_ref(
                    document["payload"]["raw_file_ref"],
                    label="provider calendar raw_file_ref",
                )
                for document in capture_documents
            ],
            provider_capture_file_refs=normalized["trusted_provider_calendar_capture_file_refs"],
            capture_transaction_file_ref=transaction_ref,
            historical=validation_mode == "HISTORICAL",
        )
        success_document = _parse_source_json(
            source_bytes("trusted_provider_calendar_capture_success_file_ref"),
            label="trusted_provider_calendar_capture_success_file_ref",
        )
        published_refs = sorted(
            [
                _file_ref(docs_ref, label="provider docs raw_file_ref"),
                capability_ref,
                normalized["files"]["calendar_authority_policy_file_ref"],
                *normalized["trusted_provider_calendar_raw_file_refs"][1:],
                *normalized["trusted_provider_calendar_capture_file_refs"],
                transaction_ref,
                execution_ref,
                release_input_ref,
            ],
            key=lambda row: row["relative_path"],
        )
        validate_trusted_provider_calendar_capture_success(
            success_document,
            capture_transaction_file_ref=transaction_ref,
            capture_execution_file_ref=execution_ref,
            published_leaf_file_refs=published_refs,
        )
        if (
            success_document["payload"]["observed_completed_at"]
            < execution_document["payload"]["observed_completed_at"]
        ):
            raise SystemContractError("trusted-provider capture success precedes execution")
        if validation_mode not in {"PRE_CAS_CURRENT", "HISTORICAL"}:
            raise SystemContractError("calendar validation mode is invalid")
        compilation = validate_trusted_provider_calendar_compilation(
            compilation_document,
            policy=policy,
            capability=capability_document,
            capture_documents=capture_documents,
            docs_raw=docs_raw,
            raw_resolver=raw_resolver,
            expected_release_ref=release_ref,
            pit_exchange_ids=expected_exchanges,
            market_session_dates=market_sessions,
            historical=validation_mode == "HISTORICAL",
        )
        source_limitations = list(TRUSTED_PROVIDER_CALENDAR_SOURCE_LIMITATIONS)
        expected_kind = TRUSTED_PROVIDER_CALENDAR_COMPILATION_KIND
    else:
        raise SystemContractError("calendar authority route is unsupported")
    if compilation["kind"] != expected_kind:
        raise SystemContractError("calendar policy/compilation kind differs")
    payload = compilation["payload"]
    if policy["payload"]["expected_compilation_kind"] != compilation["kind"]:
        raise SystemContractError("calendar policy expected compilation kind differs")
    if route == TRUSTED_PROVIDER_CALENDAR_ROUTE and (
        payload.get("policy_ref") != object_ref_for_artifact(policy)
        or payload.get("source_limitations") != source_limitations
    ):
        raise SystemContractError("trusted-provider policy/limitation closure differs")
    if (
        payload["calendar_json_file_ref"] != normalized["files"]["calendar_runtime_json_file_ref"]
        or payload["calendar_parquet_file_ref"] != normalized["files"]["exchange_calendar_file_ref"]
        or payload["cutoff_date"] != source_projection.get("pit_signal_session")
        or payload["cutoff_date"] != source_projection.get("latest_market_session")
        or payload["pit_exchange_ids"] != expected_exchanges
        or payload["market_session_dates_sha256"] != _sha256(canonical_json_bytes(market_sessions))
    ):
        raise SystemContractError("calendar compilation/Factor source binding differs")
    calendar_projection = source_projection.get("calendar")
    runtime = payload["runtime_projection"]
    open_rows = [row for row in runtime if row["status"] == "OPEN"]
    if type(calendar_projection) is not dict or (
        calendar_projection.get("open_sessions") != [row["date"] for row in open_rows]
        or calendar_projection.get("opens_at_utc") != [row["opens_at_utc"] for row in open_rows]
        or calendar_projection.get("closes_at_utc") != [row["closes_at_utc"] for row in open_rows]
    ):
        raise SystemContractError("calendar compilation/strict Parquet replay differs")
    return compilation


def _receipt_input_source_rows(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    scalar_refs: Mapping[str, Mapping[str, Any]],
    list_refs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Bind every operator-request input to its immutable stored source object."""

    rows: list[dict[str, Any]] = []
    expected_scalar = set(normalized["files"])
    for field in (
        "trusted_provider_calendar_capability_file_ref",
        "trusted_provider_calendar_capture_transaction_file_ref",
        "trusted_provider_calendar_capture_execution_file_ref",
        "trusted_provider_calendar_capture_success_file_ref",
        "trusted_provider_release_install_input_file_ref",
    ):
        if normalized[field] is not None:
            expected_scalar.add(field)
    if set(scalar_refs) != expected_scalar:
        raise SystemContractError("production receipt scalar input closure differs")
    expected_lists = {
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
        "market_table_file_refs",
        "fundamental_table_file_refs",
        "fundamental_evidence_file_refs",
    }
    if set(list_refs) != expected_lists:
        raise SystemContractError("production receipt list input closure differs")
    for field, input_ref in normalized["files"].items():
        rows.append(
            {
                "field": field,
                "ordinal": 0,
                "input_file_ref": dict(input_ref),
                "source_object_ref": validate_object_ref(
                    scalar_refs[field], label=f"{field} source object"
                ),
            }
        )
    capability_ref = normalized["trusted_provider_calendar_capability_file_ref"]
    if capability_ref is not None:
        rows.append(
            {
                "field": "trusted_provider_calendar_capability_file_ref",
                "ordinal": 0,
                "input_file_ref": dict(capability_ref),
                "source_object_ref": validate_object_ref(
                    scalar_refs["trusted_provider_calendar_capability_file_ref"],
                    label="trusted_provider_calendar_capability_file_ref source object",
                ),
            }
        )
    transaction_ref = normalized["trusted_provider_calendar_capture_transaction_file_ref"]
    if transaction_ref is not None:
        rows.append(
            {
                "field": "trusted_provider_calendar_capture_transaction_file_ref",
                "ordinal": 0,
                "input_file_ref": dict(transaction_ref),
                "source_object_ref": validate_object_ref(
                    scalar_refs["trusted_provider_calendar_capture_transaction_file_ref"],
                    label=(
                        "trusted_provider_calendar_capture_transaction_file_ref " "source object"
                    ),
                ),
            }
        )
    for field in (
        "trusted_provider_calendar_capture_execution_file_ref",
        "trusted_provider_calendar_capture_success_file_ref",
        "trusted_provider_release_install_input_file_ref",
    ):
        reference = normalized[field]
        if reference is not None:
            rows.append(
                {
                    "field": field,
                    "ordinal": 0,
                    "input_file_ref": dict(reference),
                    "source_object_ref": validate_object_ref(
                        scalar_refs[field],
                        label=f"{field} source object",
                    ),
                }
            )
    for field in sorted(expected_lists):
        inputs = normalized[field]
        sources = list_refs[field]
        if len(inputs) != len(sources):
            raise SystemContractError(f"production receipt {field} cardinality differs")
        for ordinal, (input_ref, source_ref) in enumerate(zip(inputs, sources, strict=True)):
            rows.append(
                {
                    "field": field,
                    "ordinal": ordinal,
                    "input_file_ref": dict(input_ref),
                    "source_object_ref": validate_object_ref(
                        source_ref, label=f"{field}[{ordinal}] source object"
                    ),
                }
            )
    rows.sort(key=lambda row: (row["field"], row["ordinal"]))
    if len({row["source_object_ref"]["byte_sha256"] for row in rows}) != len(rows):
        raise SystemContractError("production receipt input source objects are duplicated")
    return rows


def _receipt_copied_paths(  # noqa: C901
    *,
    store: SystemStore,
    normalized: Mapping[str, Any],
    input_source_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str] | str], dict[tuple[str, int], dict[str, str]]]:
    """Replay exact request-to-source bindings without trusting receipt claims."""

    expected: dict[tuple[str, int], dict[str, str]] = {
        (field, 0): dict(reference) for field, reference in normalized["files"].items()
    }
    capability_ref = normalized["trusted_provider_calendar_capability_file_ref"]
    if capability_ref is not None:
        expected[("trusted_provider_calendar_capability_file_ref", 0)] = dict(capability_ref)
    transaction_ref = normalized["trusted_provider_calendar_capture_transaction_file_ref"]
    if transaction_ref is not None:
        expected[("trusted_provider_calendar_capture_transaction_file_ref", 0)] = dict(
            transaction_ref
        )
    for field in (
        "trusted_provider_calendar_capture_execution_file_ref",
        "trusted_provider_calendar_capture_success_file_ref",
        "trusted_provider_release_install_input_file_ref",
    ):
        reference = normalized[field]
        if reference is not None:
            expected[(field, 0)] = dict(reference)
    for field in (
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
        "market_table_file_refs",
        "fundamental_table_file_refs",
        "fundamental_evidence_file_refs",
    ):
        expected.update(
            {
                (field, ordinal): dict(reference)
                for ordinal, reference in enumerate(normalized[field])
            }
        )
    observed: dict[tuple[str, int], dict[str, str]] = {}
    copied: dict[str, list[str] | str] = {}
    source_refs_seen: set[tuple[str, ...]] = set()
    for row in input_source_rows:
        key = (row["field"], row["ordinal"])
        if key not in expected or key in observed or row["input_file_ref"] != expected[key]:
            raise SystemContractError("production receipt input/request binding differs")
        source_ref = validate_object_ref(
            row["source_object_ref"], label=f"production receipt input {key}"
        )
        ref_key = tuple(
            source_ref[field]
            for field in (
                "kind",
                "contract_sha256",
                "artifact_id",
                "semantic_sha256",
                "byte_sha256",
            )
        )
        if ref_key in source_refs_seen:
            raise SystemContractError("production receipt input source ref is duplicated")
        source_refs_seen.add(ref_key)
        source = store.get_object(source_ref)
        if source["kind"] != "system.source_object":
            raise SystemContractError("production receipt input is not a source object")
        payload = store._verify_source_object(source)
        if payload["byte_sha256"] != expected[key]["byte_sha256"]:
            raise SystemContractError("production receipt input source bytes differ")
        source_path = PurePosixPath(payload["relative_path"])
        try:
            source_path.relative_to(_staging_relative_root(normalized["operation_id"]))
        except ValueError as exc:
            raise SystemContractError(
                "production receipt source lies outside operation staging root"
            ) from exc
        observed[key] = source_ref
        field, ordinal = key
        if field in normalized["files"] or field in {
            "trusted_provider_calendar_capability_file_ref",
            "trusted_provider_calendar_capture_transaction_file_ref",
            "trusted_provider_calendar_capture_execution_file_ref",
            "trusted_provider_calendar_capture_success_file_ref",
            "trusted_provider_release_install_input_file_ref",
        }:
            copied[field] = payload["relative_path"]
        else:
            values = copied.setdefault(field, [])
            if type(values) is not list or ordinal != len(values):
                raise SystemContractError("production receipt list input order differs")
            values.append(payload["relative_path"])
    if set(observed) != set(expected):
        raise SystemContractError("production receipt input closure is incomplete")
    for field in (
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
        "market_table_file_refs",
        "fundamental_table_file_refs",
        "fundamental_evidence_file_refs",
    ):
        copied.setdefault(field, [])
    return copied, observed


def _canonical_table_scalar(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SystemContractError("strict source contains a non-finite float")
        return value.hex()
    if value is None or type(value) in {str, bool, int}:
        return value
    raise SystemContractError("strict source contains an unsupported scalar")


def _canonical_table_projection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        normalized = {key: _canonical_table_scalar(value) for key, value in sorted(row.items())}
        digest.update(canonical_json_bytes(normalized))
        digest.update(b"\n")
        count += 1
    return {"row_count": count, "canonical_rows_sha256": digest.hexdigest()}


def _strict_table_projection(
    store: SystemStore,
    source_ref: Mapping[str, Any],
    *,
    columns: Sequence[str],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for row in _iter_bounded_parquet_rows(
        store=store,
        source_ref=source_ref,
        columns=columns,
    ):
        normalized = {key: _canonical_table_scalar(value) for key, value in sorted(row.items())}
        digest.update(canonical_json_bytes(normalized))
        digest.update(b"\n")
        count += 1
    return {"row_count": count, "canonical_rows_sha256": digest.hexdigest()}


def _exact_source_bundle(
    store: SystemStore,
    bundle_ref: Mapping[str, Any],
    *,
    expected_sources: Sequence[tuple[str, Mapping[str, Any]]],
    label: str,
) -> dict[str, str]:
    ref = validate_object_ref(bundle_ref, label=f"{label} ref")
    artifact = store.get_object(ref)
    expected_rows = sorted(
        [
            {"role": role, "source_ref": validate_object_ref(source_ref)}
            for role, source_ref in expected_sources
        ],
        key=lambda row: cast(str, row["role"]).encode("utf-8"),
    )
    if (
        artifact["kind"] != "system.source_bundle"
        or artifact["payload"].get("state") != "IMMUTABLE"
        or artifact["payload"].get("sources") != expected_rows
    ):
        raise SystemContractError(f"{label} exact source topology differs")
    store._verify_source_bundle(artifact, require_sources=True)
    return ref


def _validate_operational_source_topology(
    *,
    store: SystemStore,
    operational_source_ref: Mapping[str, Any],
    observed_inputs: Mapping[tuple[str, int], Mapping[str, Any]],
    calendar_ref: Mapping[str, Any],
    market_ref: Mapping[str, Any],
    pit_ref: Mapping[str, Any],
) -> None:
    """Require the manifest's four-role closure to contain the replayed bytes."""

    def scalar(field: str) -> Mapping[str, Any]:
        return observed_inputs[(field, 0)]

    def values(field: str) -> list[Mapping[str, Any]]:
        rows = [
            (ordinal, ref)
            for (row_field, ordinal), ref in observed_inputs.items()
            if row_field == field
        ]
        rows.sort(key=lambda row: row[0])
        if [ordinal for ordinal, _ref in rows] != list(range(len(rows))):
            raise SystemContractError(f"{field} operational source ordinals differ")
        return [ref for _ordinal, ref in rows]

    calendar_top = _exact_source_bundle(
        store,
        _source_role_ref(store, operational_source_ref, "exchange_calendar"),
        label="calendar operational source",
        expected_sources=[
            ("calendar", calendar_ref),
            ("compilation", scalar("calendar_compilation_file_ref")),
            ("authority-policy", scalar("calendar_authority_policy_file_ref")),
            ("runtime-json", scalar("calendar_runtime_json_file_ref")),
            *[
                (f"official-raw-{index:04d}", ref)
                for index, ref in enumerate(values("official_calendar_raw_file_refs"))
            ],
            *[
                (f"official-capture-{index:04d}", ref)
                for index, ref in enumerate(values("official_calendar_capture_file_refs"))
            ],
            *[
                (f"decoder-admission-{index:04d}", ref)
                for index, ref in enumerate(values("official_calendar_decoder_admission_file_refs"))
            ],
            *[
                (f"index-closure-{index:04d}", ref)
                for index, ref in enumerate(values("official_calendar_index_closure_file_refs"))
            ],
            *[
                (f"provider-raw-{index:04d}", ref)
                for index, ref in enumerate(values("trusted_provider_calendar_raw_file_refs"))
            ],
            *[
                (f"provider-capture-{index:04d}", ref)
                for index, ref in enumerate(values("trusted_provider_calendar_capture_file_refs"))
            ],
            *(
                [
                    (
                        "provider-capability",
                        scalar("trusted_provider_calendar_capability_file_ref"),
                    )
                ]
                if ("trusted_provider_calendar_capability_file_ref", 0) in observed_inputs
                else []
            ),
            *(
                [
                    (
                        "provider-capture-transaction",
                        scalar("trusted_provider_calendar_capture_transaction_file_ref"),
                    )
                ]
                if (
                    "trusted_provider_calendar_capture_transaction_file_ref",
                    0,
                )
                in observed_inputs
                else []
            ),
            *[
                (role, scalar(field))
                for role, field in (
                    (
                        "provider-capture-execution",
                        "trusted_provider_calendar_capture_execution_file_ref",
                    ),
                    (
                        "provider-capture-success",
                        "trusted_provider_calendar_capture_success_file_ref",
                    ),
                    (
                        "provider-release-install-input",
                        "trusted_provider_release_install_input_file_ref",
                    ),
                )
                if (field, 0) in observed_inputs
            ],
        ],
    )
    market_top = _exact_source_bundle(
        store,
        _source_role_ref(store, operational_source_ref, "market_snapshot"),
        label="market operational source",
        expected_sources=[
            ("manifest", scalar("market_snapshot_manifest_file_ref")),
            ("pointer", scalar("market_pointer_file_ref")),
            ("scope", scalar("market_scope_file_ref")),
            ("table", market_ref),
            *[
                (f"raw-table-{index:04d}", ref)
                for index, ref in enumerate(values("market_table_file_refs"))
            ],
        ],
    )
    fundamental_top = _exact_source_bundle(
        store,
        _source_role_ref(store, operational_source_ref, "fundamental_generation"),
        label="Fundamental operational source",
        expected_sources=[
            ("manifest", scalar("fundamental_generation_manifest_file_ref")),
            ("pointer", scalar("fundamental_pointer_file_ref")),
            *[
                (f"table-{index:04d}", ref)
                for index, ref in enumerate(values("fundamental_table_file_refs"))
            ],
            *[
                (f"evidence-{index:04d}", ref)
                for index, ref in enumerate(values("fundamental_evidence_file_refs"))
            ],
        ],
    )
    pit_top = _exact_source_bundle(
        store,
        _source_role_ref(store, operational_source_ref, "pit_membership"),
        label="PIT operational source",
        expected_sources=[
            ("manifest", scalar("pit_generation_manifest_file_ref")),
            ("pointer", scalar("pit_pointer_file_ref")),
            ("membership", scalar("pit_membership_file_ref")),
            ("strict-universe", pit_ref),
        ],
    )
    _exact_source_bundle(
        store,
        operational_source_ref,
        label="operational source closure",
        expected_sources=[
            ("exchange_calendar", calendar_top),
            ("fundamental_generation", fundamental_top),
            ("market_snapshot", market_top),
            ("pit_membership", pit_top),
        ],
    )


def _source_role_ref(
    store: SystemStore,
    bundle_ref: Mapping[str, Any],
    role: str,
) -> dict[str, str]:
    bundle = store.get_object(validate_object_ref(bundle_ref))
    rows = bundle["payload"].get("sources")
    matches = [row for row in rows if row.get("role") == role] if type(rows) is list else []
    if len(matches) != 1 or set(matches[0]) != {"role", "source_ref"}:
        raise SystemContractError(f"operational source role {role} is not exact")
    return validate_object_ref(matches[0]["source_ref"], label=f"operational {role}")


_INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256: Final = (
    "88c3baff71fb855ba73e260c267519195853482b28f781ef2f23a4dfe25a3679"
)
_HISTORICAL_PRODUCTION_RECEIPT_CONTRACT_SHA256S: Final = frozenset(
    {
        _INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256,
        PRODUCTION_BOOTSTRAP_RECEIPT_CONTRACT_SHA256,
    }
)
_INITIAL_PRODUCTION_RECEIPT_FIELDS: Final = PRODUCTION_BOOTSTRAP_RECEIPT_FIELDS - {
    "calendar_authority_policy_ref",
    "calendar_compilation_ref",
    "calendar_capability_ref",
    "calendar_capture_execution_ref",
    "calendar_authorization_basis",
    "calendar_source_limitations",
}


def _historical_generation_intent_sha256(
    manifest_payload: Mapping[str, Any], *, created_at: str
) -> str:
    """Replay the frozen initial intent domain without current Factor policy code."""

    body = {
        "generation_state": manifest_payload["generation_state"],
        "contract_catalog_sha256": manifest_payload["contract_catalog_sha256"],
        "release_manifest_ref": manifest_payload["release_manifest_ref"],
        "source_refs": manifest_payload["source_refs"],
        "factor_source_object_refs": manifest_payload["factor_source_object_refs"],
        "factor_policy_ref": manifest_payload["factor_policy_ref"],
        "factor_evidence_refs": manifest_payload["factor_evidence_refs"],
        "factor_active_set_ref": manifest_payload["factor_active_set_ref"],
        "factor_validation_attestation_ref": manifest_payload["factor_validation_attestation_ref"],
        "mainline_ref": None,
        "research_role": "SOLE_PRODUCTION_BOOTSTRAP_RECEIPT",
        "migration_receipt_ref": None,
        "migration_marker_ref": None,
        "skill_tree_sha256": manifest_payload["skill_tree_sha256"],
        "automation_semantic_sha256": manifest_payload["automation_semantic_sha256"],
        "readiness_matrix_ref": manifest_payload["readiness_matrix_ref"],
        "emergency_controller_sha256": manifest_payload["emergency_controller_sha256"],
        "created_at": created_at,
        "assembly_id": manifest_payload["assembly_id"],
    }
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _validate_historical_production_bootstrap_generation_closure(  # noqa: C901
    *,
    store: SystemStore,
    verified_generation: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    historical_assembler_sha256: str | None,
) -> dict[str, Any]:
    """Verify the initial immutable graph without executing descendant semantics."""

    if verified_generation.get("generation_state") != "OPERATIONAL":
        raise SystemContractError("historical bootstrap generation is not OPERATIONAL")
    manifest = verified_generation.get("manifest")
    if type(manifest) is not dict or type(manifest.get("payload")) is not dict:
        raise SystemContractError("historical generation manifest is invalid")
    manifest_payload = manifest["payload"]
    research = verified_generation.get("research")
    research_refs = manifest_payload.get("research_refs")
    if type(research) is not list or len(research) != 1 or type(research_refs) is not list:
        raise SystemContractError("historical production receipt cardinality differs")
    if len(research_refs) != 1:
        raise SystemContractError("historical production receipt ref is absent")
    receipt = research[0]
    if (
        type(receipt) is not dict
        or receipt.get("kind") != "system.production_bootstrap_receipt"
        or receipt.get("contract_sha256") not in _HISTORICAL_PRODUCTION_RECEIPT_CONTRACT_SHA256S
        or type(receipt.get("payload")) is not dict
    ):
        raise SystemContractError("historical production receipt contract differs")
    receipt_ref = {
        "kind": receipt["kind"],
        "contract_sha256": receipt["contract_sha256"],
        "artifact_id": receipt["artifact_id"],
        "semantic_sha256": receipt["semantic_sha256"],
        "byte_sha256": hashlib.sha256(canonical_json_bytes(receipt)).hexdigest(),
    }
    if receipt_ref != research_refs[0]:
        raise SystemContractError("historical production receipt exact ref differs")
    payload = receipt["payload"]
    historical_fields = (
        _INITIAL_PRODUCTION_RECEIPT_FIELDS
        if receipt["contract_sha256"] == _INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256
        else PRODUCTION_BOOTSTRAP_RECEIPT_FIELDS
    )
    if set(payload) != historical_fields or payload["state"] != "VERIFIED":
        raise SystemContractError("historical production receipt fields/state differ")
    identity_body = {
        key: payload[key]
        for key in sorted(historical_fields)
        if key != "production_bootstrap_receipt_id"
    }
    expected_receipt_id = (
        "production-bootstrap-" + hashlib.sha256(canonical_json_bytes(identity_body)).hexdigest()
    )
    if payload["production_bootstrap_receipt_id"] != expected_receipt_id:
        raise SystemContractError("historical production receipt identity differs")

    release_ref = validate_object_ref(deployed_release_ref, label="historical deployed release")
    exact_manifest_bindings = {
        "deployed_release_ref": manifest_payload["release_manifest_ref"],
        "source_refs": manifest_payload["source_refs"],
        "factor_source_object_refs": manifest_payload["factor_source_object_refs"],
        "factor_policy_ref": manifest_payload["factor_policy_ref"],
        "factor_evidence_refs": manifest_payload["factor_evidence_refs"],
        "factor_active_set_ref": manifest_payload["factor_active_set_ref"],
        "factor_validation_attestation_ref": manifest_payload["factor_validation_attestation_ref"],
        "readiness_matrix_ref": manifest_payload["readiness_matrix_ref"],
        "emergency_controller_sha256": manifest_payload["emergency_controller_sha256"],
        "skill_tree_sha256": manifest_payload["skill_tree_sha256"],
        "automation_semantic_sha256": manifest_payload["automation_semantic_sha256"],
    }
    if any(payload[field] != value for field, value in exact_manifest_bindings.items()):
        raise SystemContractError("historical receipt/generation binding differs")
    release = verified_generation.get("release")
    if (
        payload["deployed_release_ref"] != release_ref
        or manifest_payload["release_manifest_ref"] != release_ref
        or type(release) is not dict
        or payload["release_code_manifest_sha256"]
        != release.get("payload", {}).get("code_manifest_sha256")
        or payload["source_root_id"] != store.source_root_id
        or payload["generation_created_at"] != manifest["created_at"]
        or receipt["created_at"] != manifest["created_at"]
        or payload["mainline_ref"] is not None
        or manifest_payload["mainline_ref"] is not None
        or manifest_payload["migration_receipt_ref"] is not None
        or manifest_payload["migration_marker_ref"] is not None
    ):
        raise SystemContractError("historical production generation envelope differs")
    expected_assembly_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "system.generation_assembly",
                "identity_inputs": {
                    "generation_state": "OPERATIONAL",
                    "release_id": release_ref["artifact_id"],
                    "readiness_id": manifest_payload["readiness_matrix_ref"]["artifact_id"],
                    "created_at": manifest["created_at"],
                },
            }
        )
    ).hexdigest()
    if (
        payload["expected_assembly_id"] != expected_assembly_id
        or manifest_payload["assembly_id"] != expected_assembly_id
        or payload["generation_intent_sha256"]
        != _historical_generation_intent_sha256(manifest_payload, created_at=manifest["created_at"])
    ):
        raise SystemContractError("historical generation identity/intent differs")
    if (
        payload["assembler_module_path"] != ASSEMBLER_MODULE_PATH
        or type(historical_assembler_sha256) is not str
        or payload["assembler_code_sha256"] != historical_assembler_sha256
    ):
        raise SystemContractError("historical assembler code identity differs")

    dispatch = verified_generation.get("historical_contract_dispatch")
    if type(dispatch) is not dict:
        raise SystemContractError("historical contract dispatch is absent")
    request = store._historical_get_object(
        payload["bootstrap_operator_request_ref"], dispatch=dispatch
    )
    if request.get("kind") != BOOTSTRAP_OPERATOR_REQUEST_KIND:
        raise SystemContractError("historical bootstrap request kind differs")
    request_payload = request.get("payload")
    if (
        type(request_payload) is not dict
        or request_payload.get("release_manifest_ref") != release_ref
        or request_payload.get("source_root_id") != payload["source_root_id"]
        or request_payload.get("trusted_at") != receipt["created_at"]
        or request_payload.get("skill_tree_sha256") != payload["skill_tree_sha256"]
        or request_payload.get("automation_semantic_sha256")
        != payload["automation_semantic_sha256"]
        or request_payload.get("source_blockers") != payload["source_blockers"]
    ):
        raise SystemContractError("historical bootstrap request binding differs")

    rows = payload["input_source_rows"]
    if type(rows) is not list or not rows:
        raise SystemContractError("historical input source rows are absent")
    observed_keys: list[tuple[str, int]] = []
    for row in rows:
        if type(row) is not dict or set(row) != INPUT_SOURCE_ROW_FIELDS:
            raise SystemContractError("historical input source row fields differ")
        field = row["field"]
        ordinal = row["ordinal"]
        input_ref = row["input_file_ref"]
        if (
            type(field) is not str
            or not field
            or type(ordinal) is not int
            or ordinal < 0
            or type(input_ref) is not dict
            or set(input_ref) != {"relative_path", "byte_sha256"}
        ):
            raise SystemContractError("historical input source row is invalid")
        source = store._historical_get_object(row["source_object_ref"], dispatch=dispatch)
        store._verify_historical_source_object(source)
        source_payload = source["payload"]
        if (
            source_payload.get("byte_sha256") != input_ref["byte_sha256"]
            or source_payload.get("source_root_id") != payload["source_root_id"]
        ):
            raise SystemContractError("historical input source bytes/root differ")
        observed_keys.append((field, ordinal))
    if observed_keys != sorted(observed_keys) or len(observed_keys) != len(set(observed_keys)):
        raise SystemContractError("historical input source rows are not sorted/unique")
    if payload["source_blockers"] != sorted(FUNDAMENTAL_SOURCE_BLOCKERS):
        raise SystemContractError("historical source blockers differ")
    if payload["fundamental_machine_states"] != {
        "mixed": True,
        "legacy_direct_reader_provenance": "limited",
        "binding_aware_research_ready": True,
        "homogeneous_history_ready": False,
    }:
        raise SystemContractError("historical Fundamental machine states differ")
    statistics = payload["signal_statistics"]
    if (
        type(statistics) is not list
        or len(statistics) != 2
        or any(
            type(row) is not dict
            or type(row.get("finite_count")) is not int
            or row["finite_count"] <= 0
            or type(row.get("distinct_finite_count")) is not int
            or row["distinct_finite_count"] <= 1
            for row in statistics
        )
        or hashlib.sha256(canonical_json_bytes(statistics)).hexdigest()
        != payload["signal_statistics_sha256"]
    ):
        raise SystemContractError("historical signal statistics proof differs")
    return receipt


def _validate_production_bootstrap_generation_closure(  # noqa: C901
    *,
    store: SystemStore,
    verified_generation: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    validation_mode: str,
    historical_assembler_sha256: str | None,
) -> dict[str, Any]:
    """Deep-replay the sole production receipt required for first activation."""

    if validation_mode not in {"PRE_CAS_CURRENT", "HISTORICAL"}:
        raise SystemContractError("production bootstrap validation mode is invalid")

    if validation_mode == "HISTORICAL":
        return _validate_historical_production_bootstrap_generation_closure(
            store=store,
            verified_generation=verified_generation,
            deployed_release_ref=deployed_release_ref,
            historical_assembler_sha256=historical_assembler_sha256,
        )

    if verified_generation.get("generation_state") != "OPERATIONAL":
        raise SystemContractError("production bootstrap receipt requires OPERATIONAL generation")
    manifest = verified_generation["manifest"]
    manifest_payload = manifest["payload"]
    research_refs = manifest_payload["research_refs"]
    research = verified_generation["research"]
    if len(research_refs) != 1 or len(research) != 1:
        raise SystemContractError("operational generation must bind one production receipt")
    receipt = validate_production_bootstrap_receipt(research[0])
    receipt_ref = object_ref_for_artifact(receipt)
    if receipt_ref != research_refs[0]:
        raise SystemContractError("production bootstrap receipt exact ref differs")
    payload = receipt["payload"]
    release_ref = validate_object_ref(deployed_release_ref, label="deployed_release_ref")
    exact_manifest_bindings = {
        "deployed_release_ref": manifest_payload["release_manifest_ref"],
        "source_refs": manifest_payload["source_refs"],
        "factor_source_object_refs": manifest_payload["factor_source_object_refs"],
        "factor_policy_ref": manifest_payload["factor_policy_ref"],
        "factor_evidence_refs": manifest_payload["factor_evidence_refs"],
        "factor_active_set_ref": manifest_payload["factor_active_set_ref"],
        "factor_validation_attestation_ref": manifest_payload["factor_validation_attestation_ref"],
        "readiness_matrix_ref": manifest_payload["readiness_matrix_ref"],
        "emergency_controller_sha256": manifest_payload["emergency_controller_sha256"],
        "skill_tree_sha256": manifest_payload["skill_tree_sha256"],
        "automation_semantic_sha256": manifest_payload["automation_semantic_sha256"],
    }
    if any(payload[field] != value for field, value in exact_manifest_bindings.items()):
        raise SystemContractError("production receipt/generation binding differs")
    if payload["deployed_release_ref"] != release_ref:
        raise SystemContractError("production receipt/deployed release differs")
    release = store.get_object(release_ref)
    if (
        payload["source_root_id"] != store.source_root_id
        or payload["release_code_manifest_sha256"] != release["payload"]["code_manifest_sha256"]
        or payload["generation_created_at"] != manifest["created_at"]
        or receipt["created_at"] != manifest["created_at"]
        or manifest_payload["assembly_id"] != payload["expected_assembly_id"]
        or manifest_payload["mainline_ref"] is not None
        or payload["mainline_ref"] is not None
    ):
        raise SystemContractError("production receipt generation envelope differs")
    expected_assembly_id = generation_assembly_identity(
        generation_state="OPERATIONAL",
        release_id=release_ref["artifact_id"],
        readiness_id=manifest_payload["readiness_matrix_ref"]["artifact_id"],
        created_at=manifest["created_at"],
    )
    if payload["expected_assembly_id"] != expected_assembly_id:
        raise SystemContractError("production generation assembly identity differs")
    expected_intent_sha = production_generation_intent_sha256(
        generation_state=manifest_payload["generation_state"],
        contract_catalog_sha256=manifest_payload["contract_catalog_sha256"],
        release_manifest_ref=manifest_payload["release_manifest_ref"],
        source_refs=manifest_payload["source_refs"],
        factor_source_object_refs=manifest_payload["factor_source_object_refs"],
        factor_policy_ref=manifest_payload["factor_policy_ref"],
        factor_evidence_refs=manifest_payload["factor_evidence_refs"],
        factor_active_set_ref=manifest_payload["factor_active_set_ref"],
        factor_validation_attestation_ref=manifest_payload["factor_validation_attestation_ref"],
        skill_tree_sha256=manifest_payload["skill_tree_sha256"],
        automation_semantic_sha256=manifest_payload["automation_semantic_sha256"],
        readiness_matrix_ref=manifest_payload["readiness_matrix_ref"],
        emergency_controller_sha256=manifest_payload["emergency_controller_sha256"],
        generation_created_at=manifest["created_at"],
        expected_assembly_id=manifest_payload["assembly_id"],
    )
    if payload["generation_intent_sha256"] != expected_intent_sha:
        raise SystemContractError("production generation intent differs")
    assembler_path = Path(__file__).resolve(strict=True)
    expected_assembler_sha = (
        _sha256(assembler_path.read_bytes())
        if validation_mode == "PRE_CAS_CURRENT"
        else historical_assembler_sha256
    )
    if payload["assembler_module_path"] != ASSEMBLER_MODULE_PATH or (
        type(expected_assembler_sha) is not str
        or payload["assembler_code_sha256"] != expected_assembler_sha
    ):
        raise SystemContractError("production receipt assembler code identity differs")

    request_ref = validate_object_ref(
        payload["bootstrap_operator_request_ref"], label="bootstrap_operator_request_ref"
    )
    request = store.get_object(request_ref)
    normalized = validate_bootstrap_operator_request(request)
    if normalized["release_manifest_ref"] != release_ref:
        raise SystemContractError("production request/release binding differs")
    if (
        normalized["skill_tree_sha256"] != payload["skill_tree_sha256"]
        or normalized["automation_semantic_sha256"] != payload["automation_semantic_sha256"]
        or normalized["source_blockers"] != payload["source_blockers"]
        or normalized["source_root_id"] != payload["source_root_id"]
        or normalized["trusted_at"] != receipt["created_at"]
        or normalized["trusted_at"] != payload["generation_created_at"]
    ):
        raise SystemContractError("production request/receipt control binding differs")
    copied, observed_inputs = _receipt_copied_paths(
        store=store,
        normalized=normalized,
        input_source_rows=payload["input_source_rows"],
    )

    factor_policy = verified_generation["factor_policy"]
    bundles, factor_sources, decoded, documents = _bootstrap_sources(store, factor_policy)
    del bundles
    ref_fields = (
        "kind",
        "contract_sha256",
        "artifact_id",
        "semantic_sha256",
        "byte_sha256",
    )
    if sorted(
        factor_sources.values(), key=lambda row: tuple(row[field] for field in ref_fields)
    ) != sorted(
        manifest_payload["factor_source_object_refs"],
        key=lambda row: tuple(row[field] for field in ref_fields),
    ):
        raise SystemContractError("production receipt Factor source closure differs")
    calendar_ref = factor_sources["exchange_calendar"]
    market_ref = factor_sources["market"]
    pit_ref = factor_sources["pit_universe"]
    if calendar_ref != observed_inputs[("exchange_calendar_file_ref", 0)]:
        raise SystemContractError("strict calendar/request source binding differs")
    if len(manifest_payload["source_refs"]) != 1:
        raise SystemContractError("production operational source cardinality differs")
    _validate_operational_source_topology(
        store=store,
        operational_source_ref=manifest_payload["source_refs"][0],
        observed_inputs=observed_inputs,
        calendar_ref=calendar_ref,
        market_ref=market_ref,
        pit_ref=pit_ref,
    )
    active = verified_generation["factor_active_set"]["payload"]
    expected_factor_rows, expected_control_rows = _set_rows(bootstrap_factor_definitions())
    readiness = verified_generation["readiness"]["payload"]
    if (
        active.get("factor_rows") != expected_factor_rows
        or active.get("control_rows") != expected_control_rows
        or active.get("producer_identity") != "NOT_CLAIMED"
        or active.get("admission_route") != "BOOTSTRAP_EXCEPTION"
        or readiness.get("factor_state") != "READY"
        or readiness.get("mainline_state") != "UNINITIALIZED"
        or readiness.get("investment_state") != "BLOCKED"
        or readiness.get("producer_identity") != "NOT_CLAIMED"
        or readiness.get("admission_route") != "BOOTSTRAP_EXCEPTION"
    ):
        raise SystemContractError("production receipt authority/readiness target differs")
    expected_readiness = assess_readiness(
        producer_identity=active["producer_identity"],
        assessed_at=manifest["created_at"],
        factor_status=verified_generation["factor_status"],
        source_blockers=payload["source_blockers"],
        readiness_id=readiness["readiness_id"],
    )
    if expected_readiness["payload"] != readiness:
        raise SystemContractError("production receipt readiness replay differs")
    _normalized_shas, eligible, strict_signals, source_projection = _strict_projection(
        store,
        calendar_ref=calendar_ref,
        pit_ref=pit_ref,
        market_ref=market_ref,
    )
    market_closure = _validate_market_closure(
        normalized=normalized,
        staging_root=store.source_root,
        copied=copied,
    )
    raw_market_rows, market_caps, raw_sessions = _read_raw_market_rows(
        staging_root=store.source_root,
        copied=copied,
        closure=market_closure,
        normalized=normalized,
        store=store,
        observed_inputs=observed_inputs,
    )
    raw_pit_rows = _read_raw_pit_rows(
        staging_root=store.source_root,
        copied=copied,
        closure=market_closure,
        cutoff_market_caps=market_caps,
        normalized=normalized,
        store=store,
        observed_inputs=observed_inputs,
    )
    if _strict_table_projection(
        store,
        market_ref,
        columns=("trade_date", "symbol", "adj_close", "amount", "vol"),
    ) != _canonical_table_projection(raw_market_rows):
        raise SystemContractError("strict market materialization differs from canonical input")
    if _strict_table_projection(
        store,
        pit_ref,
        columns=("signal_session", "symbol", "industry", "total_mv", "tradable"),
    ) != _canonical_table_projection(raw_pit_rows):
        raise SystemContractError("strict PIT materialization differs from canonical input")
    frames = {
        symbol: group.drop(columns=["symbol"]).reset_index(drop=True)
        for symbol, group in pd.DataFrame(raw_market_rows).groupby("symbol", sort=True)
    }
    recomputed = compute_bootstrap_signals(frames, source_format="PARQUET")
    raw_signals = {
        factor_id: {
            symbol: None if pd.isna(value) else float(value).hex()
            for symbol, value in recomputed[factor_id].sort_index().items()
        }
        for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
    }
    if raw_signals != strict_signals or strict_signals != decoded["signals"]:
        raise SystemContractError("production raw/strict Factor signal replay differs")
    raw_session_dates = [
        datetime.strptime(value, "%Y%m%d").date().isoformat() for value in raw_sessions
    ]
    if (
        market_closure["scope_symbols"] != source_projection["all_pit_symbols"]
        or market_closure["eligible_symbols"] != eligible
        or raw_session_dates != source_projection["market_sessions"]
        or source_projection["market_symbols"] != eligible
    ):
        raise SystemContractError("production market/PIT cohort replay differs")
    staging_prefix = _staging_relative_root(normalized["operation_id"])
    for field in (
        "exchange_calendar_file_ref",
        "calendar_authority_policy_file_ref",
        "calendar_runtime_json_file_ref",
        "calendar_compilation_file_ref",
    ):
        if copied[field] != str(
            staging_prefix / "calendar_replay" / normalized["files"][field]["relative_path"]
        ):
            raise SystemContractError("production calendar replay path differs")
    for field in (
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
    ):
        expected_paths = [
            str(staging_prefix / "calendar_replay" / row["relative_path"])
            for row in normalized[field]
        ]
        if copied[field] != expected_paths:
            raise SystemContractError("production calendar evidence replay paths differ")
    for field, label in (
        (
            "trusted_provider_calendar_capability_file_ref",
            "capability",
        ),
        (
            "trusted_provider_calendar_capture_transaction_file_ref",
            "capture transaction",
        ),
        (
            "trusted_provider_calendar_capture_execution_file_ref",
            "capture execution",
        ),
        (
            "trusted_provider_calendar_capture_success_file_ref",
            "capture success",
        ),
        (
            "trusted_provider_release_install_input_file_ref",
            "release install input",
        ),
    ):
        reference = normalized[field]
        if reference is not None and copied[field] != str(
            staging_prefix / "calendar_replay" / reference["relative_path"]
        ):
            raise SystemContractError(f"production calendar {label} replay path differs")
    calendar_compilation = _validate_calendar_compilation_closure(
        normalized=normalized,
        staging_root=store.source_root,
        copied=copied,
        cohort_symbols=source_projection["all_pit_symbols"],
        source_projection=source_projection,
        release_ref=release_ref,
        repository_root=store.workspace_root,
        store=store,
        observed_inputs=observed_inputs,
        validation_mode=validation_mode,
    )
    expected_capability_ref = observed_inputs.get(
        ("trusted_provider_calendar_capability_file_ref", 0)
    )
    expected_execution_ref = observed_inputs.get(
        ("trusted_provider_calendar_capture_execution_file_ref", 0)
    )
    expected_limitations = list(calendar_compilation["payload"].get("source_limitations", []))
    expected_calendar_basis = {
        "authority_route": calendar_compilation["payload"]["authority_route"],
        "policy_ref": observed_inputs[("calendar_authority_policy_file_ref", 0)],
        "compilation_ref": observed_inputs[("calendar_compilation_file_ref", 0)],
        "capability_ref": expected_capability_ref,
        "capture_execution_ref": expected_execution_ref,
        "source_limitations": expected_limitations,
    }
    if (
        payload["calendar_authority_policy_ref"]
        != observed_inputs[("calendar_authority_policy_file_ref", 0)]
        or payload["calendar_compilation_ref"]
        != observed_inputs[("calendar_compilation_file_ref", 0)]
        or payload["calendar_capability_ref"] != expected_capability_ref
        or payload["calendar_capture_execution_ref"] != expected_execution_ref
        or payload["calendar_authorization_basis"] != expected_calendar_basis
        or payload["calendar_source_limitations"] != expected_limitations
    ):
        raise SystemContractError("production receipt calendar authority binding differs")
    fundamental_closure = _validate_fundamental_closure(
        normalized=normalized,
        staging_root=store.source_root,
        copied=copied,
        expected_cutoff=market_closure["cutoff"],
        store=store,
        observed_inputs=observed_inputs,
    )
    derived_source_blockers = _derive_source_blockers(fundamental_closure)
    if (
        derived_source_blockers != normalized["source_blockers"]
        or derived_source_blockers != payload["source_blockers"]
        or fundamental_closure["machine_states"] != payload["fundamental_machine_states"]
    ):
        raise SystemContractError("production source blocker projection differs")

    recomputation = documents["recomputation"]["document"]
    if (
        recomputation.get("signal_statistics") != payload["signal_statistics"]
        or _sha256(canonical_json_bytes(payload["signal_statistics"]))
        != payload["signal_statistics_sha256"]
    ):
        raise SystemContractError("production receipt signal statistics replay differs")
    return receipt


def validate_production_bootstrap_generation_closure(
    *,
    store: SystemStore,
    verified_generation: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    validation_mode: str,
    historical_assembler_sha256: str | None = None,
) -> dict[str, Any]:
    """Translate only expected Factor replay failures into a System hard gate."""

    try:
        return _validate_production_bootstrap_generation_closure(
            store=store,
            verified_generation=verified_generation,
            deployed_release_ref=deployed_release_ref,
            validation_mode=validation_mode,
            historical_assembler_sha256=historical_assembler_sha256,
        )
    except FactorGovernanceError as exc:
        raise SystemContractError("production bootstrap Factor closure replay failed") from exc


def assemble_production_bootstrap(  # noqa: C901
    *,
    workspace_root: str | os.PathLike[str],
    input_root: str | os.PathLike[str],
    request_raw: bytes,
) -> dict[str, Any]:
    """Materialize, validate, and offline-assemble; never activate."""

    normalized = validate_bootstrap_operator_request(request_raw)
    workspace = Path(workspace_root).absolute()
    inputs = _input_root_path(Path(input_root))
    workspace_stat = os.lstat(workspace)
    inputs_stat = os.lstat(inputs)
    if (
        not stat.S_ISDIR(workspace_stat.st_mode)
        or stat.S_ISLNK(workspace_stat.st_mode)
        or not stat.S_ISDIR(inputs_stat.st_mode)
        or stat.S_ISLNK(inputs_stat.st_mode)
        or inputs_stat.st_uid != os.geteuid()
        or stat.S_IMODE(inputs_stat.st_mode) != 0o700
    ):
        raise SystemSecurityError("bootstrap roots must be directories")
    store = SystemStore(workspace)
    if normalized["source_root_id"] != store.source_root_id:
        raise SystemContractError("bootstrap request source root identity differs")
    if store.read_active() is not None:
        raise SystemPreconditionError("bootstrap assembly requires EMPTY System pointer")
    if (workspace / str(MIGRATION_MARKER_PATH)).exists():
        raise SystemPreconditionError("bootstrap assembly requires absent migration marker")
    release_ref = normalized["release_manifest_ref"]
    release = store.get_object(release_ref)
    if release["kind"] != "system.release":
        raise SystemContractError("deployed release ref kind is invalid")
    _preflight_calendar_input_budget(normalized=normalized, input_root=inputs)
    staging = _ensure_staging_root(workspace, normalized["operation_id"])

    copied_local = _copy_request_inputs(
        normalized=normalized,
        input_root=inputs,
        staging_root=staging,
    )
    materialized = _materialize_market_and_pit(
        normalized=normalized,
        staging_root=staging,
        copied=copied_local,
    )
    fundamental_closure = _validate_fundamental_closure(
        normalized=normalized,
        staging_root=staging,
        copied=copied_local,
        expected_cutoff=materialized["cutoff"],
    )
    derived_source_blockers = _derive_source_blockers(fundamental_closure)
    if normalized["source_blockers"] != derived_source_blockers:
        raise SystemContractError("declared source blockers differ from machine projection")
    bootstrap_operator_request_ref = store.put_object(normalized["document"])
    copied = _prefix_copied_paths(copied_local, normalized["operation_id"])
    created_at = normalized["trusted_at"]
    factor_store = FactorValidationStore.for_sealed_operation(
        system_store=store,
        trusted_at=created_at,
    )
    contextual_ref = store.build_contextual_validator_component(
        BOOTSTRAP_VALIDATION_PROFILE,
        release_manifest_ref=release_ref,
        created_at=created_at,
    )
    decoder_ref = store.build_source_decoder_component(
        release_manifest_ref=release_ref,
        created_at=created_at,
    )
    implementation_refs: dict[str, dict[str, str]] = {}
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        row = installed_semantic_row(factor_id)
        implementation_refs[factor_id] = store.build_installed_component(
            component_id=row["implementation_id"],
            component_role="SOURCE_IMPLEMENTATION",
            package_name="quant_investor.factors.governance",
            module_names=[row["module_name"]],
            entrypoint_specs=[(row["module_name"], row["qualified_name"])],
            release_manifest_ref=release_ref,
            created_at=created_at,
        )
    validator_manifest = factor_store.build_validator_manifest(
        release_manifest_ref=release_ref,
        contextual_validator_component_ref=contextual_ref,
        source_decoder_component_ref=decoder_ref,
        implementation_component_refs=implementation_refs,
    )
    validator_manifest_ref = object_ref_for_artifact(validator_manifest)

    calendar_ref = _source(
        store,
        _copied_path(copied, "exchange_calendar_file_ref"),
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
        created_at=created_at,
    )
    pit_ref = _source(
        store,
        _workspace_source_relative(
            normalized["operation_id"], materialized["pit_universe_relative"]
        ),
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
        created_at=created_at,
    )
    market_ref = _source(
        store,
        _workspace_source_relative(
            normalized["operation_id"], materialized["market_history_relative"]
        ),
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
        created_at=created_at,
    )
    normalized_shas, eligible, signals, source_projection = _strict_projection(
        store,
        calendar_ref=calendar_ref,
        pit_ref=pit_ref,
        market_ref=market_ref,
    )
    calendar_compilation = _validate_calendar_compilation_closure(
        normalized=normalized,
        staging_root=staging,
        copied=copied_local,
        cohort_symbols=source_projection["all_pit_symbols"],
        source_projection=source_projection,
        release_ref=release_ref,
        repository_root=workspace,
    )
    if (
        source_projection["all_pit_symbols"] != materialized["scope_symbols"]
        or eligible != materialized["eligible_symbols"]
        or source_projection["market_symbols"] != eligible
    ):
        raise SystemContractError("materialized market/PIT cohort replay differs")
    market_bundle_ref = _bundle(
        store,
        normalized["operation_id"],
        "market",
        [("market", market_ref)],
        created_at=created_at,
    )

    definitions = bootstrap_factor_definitions()
    factor_rows, control_rows = _set_rows(definitions)
    factor_set_sha = _factor_set_sha256(
        definitions=definitions,
        factor_rows=factor_rows,
        control_rows=control_rows,
    )
    implementation_raw = canonical_json_bytes(
        {
            "domain": "myquant-bootstrap-implementation-tree-manifest",
            "implementation_rows": validator_manifest["payload"]["implementation_rows"],
        }
    )
    implementation_sha = _sha256(implementation_raw)
    statistics = _signal_statistics(
        signals,
        eligible_symbols=eligible,
        implementation_sha256s={
            LOW_DOLLAR_VOLUME: implementation_sha,
            BLEND_W80: implementation_sha,
        },
        source_bundle_sha256=market_bundle_ref["byte_sha256"],
    )
    recomputation_raw = canonical_json_bytes(
        {
            "authority": "NON_AUTHORIZING",
            "domain": "myquant-bootstrap-recomputation",
            "factor_set_sha256": factor_set_sha,
            "factor_weights": [
                {"factor_id": row["factor_id"], "weight": row["weight"]} for row in factor_rows
            ],
            "implementation_rows": validator_manifest["payload"]["implementation_rows"],
            "normalized_source_sha256s": normalized_shas,
            "result": "EXACT_MATCH",
            "signal_sha256s": _signal_hashes(signals),
            "signal_statistics": statistics,
        }
    )

    source_rows = [
        {
            "role": role,
            "source_ref": ref,
            "source_byte_sha256": store.get_object(ref)["payload"]["byte_sha256"],
        }
        for role, ref in (
            ("exchange_calendar", calendar_ref),
            ("market", market_ref),
            ("pit_universe", pit_ref),
        )
    ]
    source_rows.sort(key=lambda row: row["role"])
    generation_body = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-source-generation",
        "reader_contract": {
            "reader": "MarketDataReader",
            "market": "CN",
            "mode_policy": "strict",
            "source_format": "PARQUET",
            "fallback_allowed": False,
        },
        "source_rows": source_rows,
    }
    source_generation_raw = canonical_json_bytes(
        {
            **generation_body,
            "generation_sha256": _sha256(canonical_json_bytes(generation_body)),
        }
    )
    generated_rows = {
        "implementation": ("bootstrap/implementation-tree.json", implementation_raw),
        "recomputation": ("bootstrap/recomputation.json", recomputation_raw),
        "source_generation": (
            "bootstrap/source-generation.json",
            source_generation_raw,
        ),
    }
    for relative, raw in generated_rows.values():
        target = _destination(staging, relative)
        _write_exact_once(target, raw)
    decision_ref = _source(
        store,
        _copied_path(copied, "bootstrap_decision_file_ref"),
        source_format="JSON",
        media_type="application/json",
        created_at=created_at,
    )
    implementation_ref = _source(
        store,
        _workspace_source_relative(normalized["operation_id"], generated_rows["implementation"][0]),
        source_format="JSON",
        media_type="application/json",
        created_at=created_at,
    )
    recomputation_ref = _source(
        store,
        _workspace_source_relative(normalized["operation_id"], generated_rows["recomputation"][0]),
        source_format="JSON",
        media_type="application/json",
        created_at=created_at,
    )
    source_generation_ref = _source(
        store,
        _workspace_source_relative(
            normalized["operation_id"], generated_rows["source_generation"][0]
        ),
        source_format="JSON",
        media_type="application/json",
        created_at=created_at,
    )
    bootstrap = factor_store.initialize_bootstrap(
        release_ref=release_ref,
        decision_source_bundle_ref=_bundle(
            store,
            normalized["operation_id"],
            "decision",
            [("bootstrap_decision", decision_ref)],
            created_at=created_at,
        ),
        exchange_calendar_bundle_ref=_bundle(
            store,
            normalized["operation_id"],
            "calendar",
            [("calendar", calendar_ref)],
            created_at=created_at,
        ),
        implementation_bundle_ref=_bundle(
            store,
            normalized["operation_id"],
            "implementation",
            [("implementation_tree_manifest", implementation_ref)],
            created_at=created_at,
        ),
        market_bundle_ref=market_bundle_ref,
        pit_universe_bundle_ref=_bundle(
            store,
            normalized["operation_id"],
            "pit",
            [("pit", pit_ref)],
            created_at=created_at,
        ),
        recomputation_bundle_ref=_bundle(
            store,
            normalized["operation_id"],
            "recomputation",
            [("recomputation", recomputation_ref)],
            created_at=created_at,
        ),
        source_generation_bundle_ref=_bundle(
            store,
            normalized["operation_id"],
            "source-generation",
            [("source_generation", source_generation_ref)],
            created_at=created_at,
        ),
    )
    validation_request = store.build_validation_run_request(
        release_manifest_ref=release_ref,
        factor_validator_manifest_ref=validator_manifest_ref,
        intrinsic_receipt_ref=bootstrap.intrinsic_receipt_ref,
    )
    validation = store.run_validation(validation_request["validation_request_ref"])
    status = factor_store.build_status(
        active_factor_set_ref=bootstrap.active_set_ref,
        active_validation_receipt_ref=bootstrap.intrinsic_receipt_ref,
        active_contextual_result_ref=validation["contextual_result_ref"],
        active_validation_attestation_ref=validation["validation_attestation_ref"],
    )
    status_ref = store.put_object(status)
    readiness = assess_readiness(
        producer_identity=status["payload"]["active"]["producer_identity"],
        assessed_at=created_at,
        factor_status=status,
        source_blockers=derived_source_blockers,
        readiness_id="bootstrap-readiness-" + normalized["operation_id"],
    )
    readiness_ref = store.put_object(readiness)

    raw_objects: dict[str, dict[str, str]] = {}
    json_fields = {
        "market_scope_file_ref",
        "market_pointer_file_ref",
        "market_snapshot_manifest_file_ref",
        "pit_pointer_file_ref",
        "pit_generation_manifest_file_ref",
        "calendar_authority_policy_file_ref",
        "calendar_runtime_json_file_ref",
        "calendar_compilation_file_ref",
        "fundamental_pointer_file_ref",
        "fundamental_generation_manifest_file_ref",
    }
    for field in json_fields:
        raw_objects[field] = _source(
            store,
            _copied_path(copied, field),
            source_format="JSON",
            media_type="application/json",
            created_at=created_at,
        )
    raw_objects["pit_membership_file_ref"] = _source(
        store,
        _copied_path(copied, "pit_membership_file_ref"),
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
        created_at=created_at,
    )
    calendar_list_refs: dict[str, list[dict[str, str]]] = {}
    for field in (
        "official_calendar_raw_file_refs",
        "trusted_provider_calendar_raw_file_refs",
    ):
        calendar_list_refs[field] = [
            _source(
                store,
                relative,
                source_format="BINARY",
                media_type="application/octet-stream",
                created_at=created_at,
            )
            for relative in _copied_paths(copied, field)
        ]
    for field in (
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_capture_file_refs",
    ):
        calendar_list_refs[field] = [
            _source(
                store,
                relative,
                source_format="JSON",
                media_type="application/json",
                created_at=created_at,
            )
            for relative in _copied_paths(copied, field)
        ]
    calendar_capability_ref: dict[str, str] | None = None
    if normalized["trusted_provider_calendar_capability_file_ref"] is not None:
        calendar_capability_ref = _source(
            store,
            _copied_path(copied, "trusted_provider_calendar_capability_file_ref"),
            source_format="JSON",
            media_type="application/json",
            created_at=created_at,
        )
    calendar_transaction_ref: dict[str, str] | None = None
    if normalized["trusted_provider_calendar_capture_transaction_file_ref"] is not None:
        calendar_transaction_ref = _source(
            store,
            _copied_path(
                copied,
                "trusted_provider_calendar_capture_transaction_file_ref",
            ),
            source_format="JSON",
            media_type="application/json",
            created_at=created_at,
        )
    calendar_execution_ref: dict[str, str] | None = None
    if normalized["trusted_provider_calendar_capture_execution_file_ref"] is not None:
        calendar_execution_ref = _source(
            store,
            _copied_path(
                copied,
                "trusted_provider_calendar_capture_execution_file_ref",
            ),
            source_format="JSON",
            media_type="application/json",
            created_at=created_at,
        )
    calendar_success_ref: dict[str, str] | None = None
    if normalized["trusted_provider_calendar_capture_success_file_ref"] is not None:
        calendar_success_ref = _source(
            store,
            _copied_path(
                copied,
                "trusted_provider_calendar_capture_success_file_ref",
            ),
            source_format="JSON",
            media_type="application/json",
            created_at=created_at,
        )
    provider_release_input_ref: dict[str, str] | None = None
    if normalized["trusted_provider_release_install_input_file_ref"] is not None:
        provider_release_input_ref = _source(
            store,
            _copied_path(
                copied,
                "trusted_provider_release_install_input_file_ref",
            ),
            source_format="JSON",
            media_type="application/json",
            created_at=created_at,
        )
    market_table_refs = [
        _source(
            store,
            relative,
            source_format="PARQUET",
            media_type="application/vnd.apache.parquet",
            created_at=created_at,
        )
        for relative in _copied_paths(copied, "market_table_file_refs")
    ]
    fundamental_table_refs = [
        _source(
            store,
            relative,
            source_format="PARQUET",
            media_type="application/vnd.apache.parquet",
            created_at=created_at,
        )
        for relative in _copied_paths(copied, "fundamental_table_file_refs")
    ]
    fundamental_evidence_refs: list[dict[str, str]] = []
    for relative in _copied_paths(copied, "fundamental_evidence_file_refs"):
        suffix = PurePosixPath(relative).suffix.lower()
        if suffix == ".json":
            source_format = "JSON"
            media_type = "application/json"
        elif suffix == ".parquet":
            source_format = "PARQUET"
            media_type = "application/vnd.apache.parquet"
        else:
            raise SystemContractError("Fundamental evidence media type is unsupported")
        fundamental_evidence_refs.append(
            _source(
                store,
                relative,
                source_format=source_format,
                media_type=media_type,
                created_at=created_at,
            )
        )
    calendar_top = _bundle(
        store,
        normalized["operation_id"],
        "calendar-top",
        [
            ("calendar", calendar_ref),
            ("compilation", raw_objects["calendar_compilation_file_ref"]),
            ("authority-policy", raw_objects["calendar_authority_policy_file_ref"]),
            ("runtime-json", raw_objects["calendar_runtime_json_file_ref"]),
            *[
                (f"official-raw-{index:04d}", ref)
                for index, ref in enumerate(calendar_list_refs["official_calendar_raw_file_refs"])
            ],
            *[
                (f"official-capture-{index:04d}", ref)
                for index, ref in enumerate(
                    calendar_list_refs["official_calendar_capture_file_refs"]
                )
            ],
            *[
                (f"decoder-admission-{index:04d}", ref)
                for index, ref in enumerate(
                    calendar_list_refs["official_calendar_decoder_admission_file_refs"]
                )
            ],
            *[
                (f"index-closure-{index:04d}", ref)
                for index, ref in enumerate(
                    calendar_list_refs["official_calendar_index_closure_file_refs"]
                )
            ],
            *[
                (f"provider-raw-{index:04d}", ref)
                for index, ref in enumerate(
                    calendar_list_refs["trusted_provider_calendar_raw_file_refs"]
                )
            ],
            *[
                (f"provider-capture-{index:04d}", ref)
                for index, ref in enumerate(
                    calendar_list_refs["trusted_provider_calendar_capture_file_refs"]
                )
            ],
            *(
                [("provider-capability", calendar_capability_ref)]
                if calendar_capability_ref is not None
                else []
            ),
            *(
                [("provider-capture-transaction", calendar_transaction_ref)]
                if calendar_transaction_ref is not None
                else []
            ),
            *(
                [("provider-capture-execution", calendar_execution_ref)]
                if calendar_execution_ref is not None
                else []
            ),
            *(
                [("provider-capture-success", calendar_success_ref)]
                if calendar_success_ref is not None
                else []
            ),
            *(
                [("provider-release-install-input", provider_release_input_ref)]
                if provider_release_input_ref is not None
                else []
            ),
        ],
        created_at=created_at,
    )
    market_top = _bundle(
        store,
        normalized["operation_id"],
        "market-top",
        [
            ("manifest", raw_objects["market_snapshot_manifest_file_ref"]),
            ("pointer", raw_objects["market_pointer_file_ref"]),
            ("scope", raw_objects["market_scope_file_ref"]),
            ("table", market_ref),
            *[(f"raw-table-{index:04d}", ref) for index, ref in enumerate(market_table_refs)],
        ],
        created_at=created_at,
    )
    fundamental_top = _bundle(
        store,
        normalized["operation_id"],
        "fundamental-top",
        [
            ("manifest", raw_objects["fundamental_generation_manifest_file_ref"]),
            ("pointer", raw_objects["fundamental_pointer_file_ref"]),
            *[(f"table-{index:04d}", ref) for index, ref in enumerate(fundamental_table_refs)],
            *[
                (f"evidence-{index:04d}", ref)
                for index, ref in enumerate(fundamental_evidence_refs)
            ],
        ],
        created_at=created_at,
    )
    pit_top = _bundle(
        store,
        normalized["operation_id"],
        "pit-top",
        [
            ("manifest", raw_objects["pit_generation_manifest_file_ref"]),
            ("pointer", raw_objects["pit_pointer_file_ref"]),
            ("membership", raw_objects["pit_membership_file_ref"]),
            ("strict-universe", pit_ref),
        ],
        created_at=created_at,
    )
    operational_sources = _bundle(
        store,
        normalized["operation_id"],
        "operational-source-closure",
        [
            ("exchange_calendar", calendar_top),
            ("fundamental_generation", fundamental_top),
            ("market_snapshot", market_top),
            ("pit_membership", pit_top),
        ],
        created_at=created_at,
    )

    suspended = build_suspended_generation(
        store,
        blockers=["PREPARED_EMERGENCY_TARGET"],
        created_at=created_at,
    )
    controller = build_emergency_controller(
        store,
        suspended_generation_id=suspended["generation_id"],
    )
    receipt = store.get_object(bootstrap.intrinsic_receipt_ref)
    input_source_rows = _receipt_input_source_rows(
        normalized=normalized,
        scalar_refs={
            "exchange_calendar_file_ref": calendar_ref,
            "market_scope_file_ref": raw_objects["market_scope_file_ref"],
            "market_pointer_file_ref": raw_objects["market_pointer_file_ref"],
            "market_snapshot_manifest_file_ref": raw_objects["market_snapshot_manifest_file_ref"],
            "pit_pointer_file_ref": raw_objects["pit_pointer_file_ref"],
            "pit_generation_manifest_file_ref": raw_objects["pit_generation_manifest_file_ref"],
            "pit_membership_file_ref": raw_objects["pit_membership_file_ref"],
            "calendar_runtime_json_file_ref": raw_objects["calendar_runtime_json_file_ref"],
            "calendar_compilation_file_ref": raw_objects["calendar_compilation_file_ref"],
            "calendar_authority_policy_file_ref": raw_objects["calendar_authority_policy_file_ref"],
            "fundamental_pointer_file_ref": raw_objects["fundamental_pointer_file_ref"],
            "fundamental_generation_manifest_file_ref": raw_objects[
                "fundamental_generation_manifest_file_ref"
            ],
            "bootstrap_decision_file_ref": decision_ref,
            **(
                {
                    "trusted_provider_calendar_capability_file_ref": calendar_capability_ref,
                }
                if calendar_capability_ref is not None
                else {}
            ),
            **(
                {
                    "trusted_provider_calendar_capture_transaction_file_ref": (
                        calendar_transaction_ref
                    ),
                }
                if calendar_transaction_ref is not None
                else {}
            ),
            **(
                {
                    "trusted_provider_calendar_capture_execution_file_ref": (
                        calendar_execution_ref
                    ),
                }
                if calendar_execution_ref is not None
                else {}
            ),
            **(
                {
                    "trusted_provider_calendar_capture_success_file_ref": (calendar_success_ref),
                }
                if calendar_success_ref is not None
                else {}
            ),
            **(
                {
                    "trusted_provider_release_install_input_file_ref": (provider_release_input_ref),
                }
                if provider_release_input_ref is not None
                else {}
            ),
        },
        list_refs={
            **calendar_list_refs,
            "market_table_file_refs": market_table_refs,
            "fundamental_table_file_refs": fundamental_table_refs,
            "fundamental_evidence_file_refs": fundamental_evidence_refs,
        },
    )
    production_receipt = build_production_bootstrap_receipt(
        bootstrap_operator_request_ref=bootstrap_operator_request_ref,
        source_root_id=store.source_root_id,
        input_source_rows=input_source_rows,
        deployed_release_ref=release_ref,
        calendar_authority_policy_ref=raw_objects["calendar_authority_policy_file_ref"],
        calendar_compilation_ref=raw_objects["calendar_compilation_file_ref"],
        calendar_capability_ref=calendar_capability_ref,
        calendar_capture_execution_ref=calendar_execution_ref,
        calendar_authorization_basis={
            "authority_route": calendar_compilation["payload"]["authority_route"],
            "policy_ref": raw_objects["calendar_authority_policy_file_ref"],
            "compilation_ref": raw_objects["calendar_compilation_file_ref"],
            "capability_ref": calendar_capability_ref,
            "capture_execution_ref": calendar_execution_ref,
            "source_limitations": list(
                calendar_compilation["payload"].get("source_limitations", [])
            ),
        },
        calendar_source_limitations=list(
            calendar_compilation["payload"].get("source_limitations", [])
        ),
        release_code_manifest_sha256=release["payload"]["code_manifest_sha256"],
        generation_created_at=created_at,
        expected_assembly_id=generation_assembly_identity(
            generation_state="OPERATIONAL",
            release_id=release_ref["artifact_id"],
            readiness_id=readiness_ref["artifact_id"],
            created_at=created_at,
        ),
        generation_intent_sha256=production_generation_intent_sha256(
            generation_state="OPERATIONAL",
            contract_catalog_sha256=contract_catalog_sha256(),
            release_manifest_ref=release_ref,
            source_refs=[operational_sources],
            factor_source_object_refs=validation["contextual_result"]["payload"][
                "source_object_refs"
            ],
            factor_policy_ref=bootstrap.policy_ref,
            factor_evidence_refs=receipt["payload"]["evidence_refs"],
            factor_active_set_ref=bootstrap.active_set_ref,
            factor_validation_attestation_ref=validation["validation_attestation_ref"],
            skill_tree_sha256=normalized["skill_tree_sha256"],
            automation_semantic_sha256=normalized["automation_semantic_sha256"],
            readiness_matrix_ref=readiness_ref,
            emergency_controller_sha256=controller["byte_sha256"],
            generation_created_at=created_at,
            expected_assembly_id=generation_assembly_identity(
                generation_state="OPERATIONAL",
                release_id=release_ref["artifact_id"],
                readiness_id=readiness_ref["artifact_id"],
                created_at=created_at,
            ),
        ),
        source_refs=[operational_sources],
        factor_source_object_refs=validation["contextual_result"]["payload"]["source_object_refs"],
        factor_policy_ref=bootstrap.policy_ref,
        factor_evidence_refs=receipt["payload"]["evidence_refs"],
        factor_active_set_ref=bootstrap.active_set_ref,
        factor_validation_attestation_ref=validation["validation_attestation_ref"],
        readiness_matrix_ref=readiness_ref,
        emergency_controller_sha256=controller["byte_sha256"],
        skill_tree_sha256=normalized["skill_tree_sha256"],
        automation_semantic_sha256=normalized["automation_semantic_sha256"],
        source_blockers=derived_source_blockers,
        fundamental_machine_states=fundamental_closure["machine_states"],
        signal_statistics=statistics,
        assembler_code_sha256=_sha256(Path(__file__).resolve(strict=True).read_bytes()),
        created_at=created_at,
    )
    production_receipt_ref = store.put_object(production_receipt)
    assembly_payload = {
        "generation_state": "OPERATIONAL",
        "release_manifest_ref": release_ref,
        "source_refs": [operational_sources],
        "factor_source_object_refs": validation["contextual_result"]["payload"][
            "source_object_refs"
        ],
        "factor_policy_ref": bootstrap.policy_ref,
        "factor_evidence_refs": receipt["payload"]["evidence_refs"],
        "factor_active_set_ref": bootstrap.active_set_ref,
        "factor_validation_attestation_ref": validation["validation_attestation_ref"],
        "mainline_ref": None,
        "research_refs": [production_receipt_ref],
        "migration_receipt_ref": None,
        "migration_marker_ref": None,
        "skill_tree_sha256": normalized["skill_tree_sha256"],
        "automation_semantic_sha256": normalized["automation_semantic_sha256"],
        "readiness_matrix_ref": readiness_ref,
        "emergency_controller_sha256": controller["byte_sha256"],
    }
    if set(assembly_payload) != set(ASSEMBLY_REQUEST_FIELDS) - {"assembly_request_id"}:
        raise SystemContractError("internal assembly payload fields differ")
    assembly_request = seal_artifact(
        "system.assembly_request",
        {
            "assembly_request_id": "bootstrap-assembly-"
            + _sha256(canonical_json_bytes(assembly_payload)),
            **assembly_payload,
        },
        created_at=created_at,
    )
    assembly_request_ref = store.put_object(assembly_request)
    generation = store.assemble_from_request(assembly_request)
    verified = store.verify_generation(
        generation["generation_id"],
        deployed_release_ref=release_ref,
    )
    validate_production_bootstrap_generation_closure(
        store=store,
        verified_generation=verified,
        deployed_release_ref=release_ref,
        validation_mode="PRE_CAS_CURRENT",
    )
    if (workspace / str(ACTIVE_POINTER_PATH)).exists() or (
        workspace / str(MIGRATION_MARKER_PATH)
    ).exists():
        raise SystemStorageError("bootstrap assembler touched production authority paths")
    return {
        "status": "OFFLINE_VERIFIED",
        "operation_id": normalized["operation_id"],
        "source_root": str(staging),
        "assembly_request_ref": assembly_request_ref,
        "production_bootstrap_receipt_ref": production_receipt_ref,
        "generation_id": generation["generation_id"],
        "generation": verified,
        "suspended_generation_id": suspended["generation_id"],
        "factor_status_ref": status_ref,
        "readiness_ref": readiness_ref,
        "signal_statistics": statistics,
        "active_pointer_write_count": 0,
        "marker_write_count": 0,
    }


__all__ = [
    "BOOTSTRAP_OPERATOR_REQUEST_CONTRACT_SHA256",
    "BOOTSTRAP_OPERATOR_REQUEST_FIELDS",
    "BOOTSTRAP_OPERATOR_REQUEST_KIND",
    "FILE_REF_FIELDS",
    "assemble_production_bootstrap",
    "validate_bootstrap_operator_request",
    "validate_production_bootstrap_generation_closure",
]
