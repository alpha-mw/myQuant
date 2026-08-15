"""Production-only assembler for the first unified Factor generation.

The operator accepts one sealed request whose every input is an explicit local
path plus expected byte hash.  It never discovers "latest" inputs, calls a
provider, or writes the System active pointer.  Strict Factor sources are
copied into a new owner-only staging root, registered, replayed, and then read
again by the normal generation verifier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from typing import Any, Final
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    parse_canonical_json_bytes,
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
    _signal_hashes,
    _signal_statistics,
)
from quant_investor.factors.governance.implementations import installed_semantic_row
from quant_investor.factors.governance.source import decode_source_role
from quant_investor.factors.governance.source import role_schema
from quant_investor.intelligence import assess_readiness
from quant_investor.market.fundamental_incremental import (
    SafeSuccessorError,
    validate_successor_provenance,
)

from quant_investor.system.controller import build_emergency_controller
from quant_investor.system.errors import (
    SystemContractError,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
)
from quant_investor.system.requests import ASSEMBLY_REQUEST_FIELDS
from quant_investor.system.components import BOOTSTRAP_VALIDATION_PROFILE
from quant_investor.system.storage import ACTIVE_POINTER_PATH, MIGRATION_MARKER_PATH
from quant_investor.system.store import SystemStore, object_ref_for_artifact, validate_object_ref
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
        "calendar_manifest_file_ref",
        "calendar_raw_file_refs",
        "calendar_capture_file_refs",
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
        "source_url",
        "captured_at",
        "raw_file_ref",
        "capture_file_ref",
        "open_session_count",
        "open_session_sha256",
        "session_intervals",
    }
)
_CALENDAR_INTERVAL_FIELDS: Final = frozenset({"opens_local", "closes_local"})
_CALENDAR_CAPTURE_FIELDS: Final = frozenset(
    {
        "calendar_capture_id",
        "state",
        "exchange_id",
        "issuer",
        "source_url",
        "request_url",
        "effective_url",
        "redirect_chain",
        "http_status",
        "issuer_host",
        "tls_verified",
        "captured_at",
        "raw_file_ref",
        "raw_sha256",
        "raw_byte_length",
        "raw_media_type",
        "decoder_id",
        "decoder_sha256",
        "timezone",
        "session_intervals",
        "coverage_start_date",
        "cutoff_date",
        "daily_status_rows",
        "transform_code_sha256",
    }
)
_CALENDAR_STATUS_FIELDS: Final = frozenset({"date", "status"})
_EXCHANGE_BY_SUFFIX: Final = {".SH": "SSE", ".SZ": "SZSE", ".BJ": "BSE"}
_OFFICIAL_CALENDAR_AUTHORITIES: Final = {
    "SSE": ("SSE_OFFICIAL", "www.sse.com.cn"),
    "SZSE": ("SZSE_OFFICIAL", "www.szse.cn"),
    "BSE": ("BSE_OFFICIAL", "www.bse.cn"),
}
_OFFICIAL_CALENDAR_DECODERS: Final = {
    "SSE": "myquant.exchange_calendar.sse_official_json_v1",
    "SZSE": "myquant.exchange_calendar.szse_official_json_v1",
    "BSE": "myquant.exchange_calendar.bse_official_json_v1",
}
_OFFICIAL_RAW_CALENDAR_FIELDS: Final = frozenset(
    {
        "schema_version",
        "exchange_id",
        "issuer",
        "timezone",
        "session_intervals",
        "coverage_start_date",
        "cutoff_date",
        "daily_status_rows",
    }
)
_OFFICIAL_CALENDAR_DECODER_SHA256: Final = hashlib.sha256(
    canonical_json_bytes(
        {
            "domain": "myquant-official-exchange-calendar-decoder",
            "decoder_ids": _OFFICIAL_CALENDAR_DECODERS,
            "raw_fields": sorted(_OFFICIAL_RAW_CALENDAR_FIELDS),
            "status_fields": sorted(_CALENDAR_STATUS_FIELDS),
        }
    )
).hexdigest()
_CN_CONTINUOUS_SESSION_INTERVALS: Final = [
    {"opens_local": "09:30:00", "closes_local": "11:30:00"},
    {"opens_local": "13:00:00", "closes_local": "15:00:00"},
]


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


def validate_bootstrap_operator_request(
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
            "calendar_manifest_file_ref",
            "fundamental_pointer_file_ref",
            "fundamental_generation_manifest_file_ref",
            "bootstrap_decision_file_ref",
        )
    }
    calendar_raw = _file_refs(payload["calendar_raw_file_refs"], label="calendar_raw_file_refs")
    calendar_captures = _file_refs(
        payload["calendar_capture_file_refs"],
        label="calendar_capture_file_refs",
    )
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
            *calendar_raw,
            *calendar_captures,
            *market_tables,
            *fundamental_tables,
            *fundamental_evidence,
        ]
    ]
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
        "calendar_raw_file_refs": calendar_raw,
        "calendar_capture_file_refs": calendar_captures,
        "market_table_file_refs": market_tables,
        "fundamental_table_file_refs": fundamental_tables,
        "fundamental_evidence_file_refs": fundamental_evidence,
    }


def _stable_input_path(input_root: Path, reference: Mapping[str, str]) -> Path:
    try:
        root = input_root.resolve(strict=True)
        candidate = root.joinpath(reference["relative_path"]).resolve(strict=True)
        candidate.relative_to(root)
        metadata = os.lstat(candidate)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemSecurityError("bootstrap input path is unavailable") from exc
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or mode & 0o022
        or mode & 0o111
        or metadata.st_size <= 0
        or metadata.st_size > _MAX_INPUT_BYTES
    ):
        raise SystemSecurityError("bootstrap input storage security is invalid")
    return candidate


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


def _verify_input(
    input_root: Path,
    reference: Mapping[str, str],
) -> tuple[Path, int]:
    path = _stable_input_path(input_root, reference)
    first, size = _file_digest(path)
    second, second_size = _file_digest(path)
    if first != reference["byte_sha256"] or second != first or second_size != size:
        raise SystemSecurityError("bootstrap input exact hash changed")
    return path, size


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


def _destination(staging_root: Path, relative: str) -> Path:
    path = staging_root / relative
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in (path.parent, *path.parent.parents):
        if parent == staging_root.parent:
            break
        if staging_root == parent or staging_root in parent.parents:
            parent.chmod(0o700)
    return path


def _copy_exact_once(source: Path, target: Path, expected_sha: str) -> None:  # noqa: C901
    if target.exists():
        observed, _ = _file_digest(target)
        if observed != expected_sha:
            raise SystemStorageError("source staging exact-once conflict")
        return
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    input_fd = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
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
        while True:
            chunk = os.read(input_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(output_fd, view)
                view = view[written:]
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
        os.close(input_fd)
        if output_fd is not None:
            os.close(output_fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    observed, _ = _file_digest(target)
    if observed != expected_sha:
        raise SystemStorageError("source staging exact-byte readback mismatch")


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


def _copy_request_inputs(
    *,
    normalized: Mapping[str, Any],
    input_root: Path,
    staging_root: Path,
) -> dict[str, list[str] | str]:
    destinations = {
        "exchange_calendar_file_ref": "bootstrap/exchange_calendar.parquet",
        "market_scope_file_ref": "closure/market_scope.json",
        "market_pointer_file_ref": "closure/market_pointer.json",
        "market_snapshot_manifest_file_ref": "closure/market_snapshot_manifest.json",
        "pit_pointer_file_ref": "closure/pit_pointer.json",
        "pit_generation_manifest_file_ref": "closure/pit_generation_manifest.json",
        "pit_membership_file_ref": "closure/pit_membership.parquet",
        "calendar_manifest_file_ref": "closure/calendar_manifest.json",
        "bootstrap_decision_file_ref": ("operations/unified_cutover/bootstrap-decision.json"),
    }
    copied: dict[str, list[str] | str] = {}
    for field, relative in destinations.items():
        reference = normalized["files"][field]
        source, _ = _verify_input(input_root, reference)
        target = _destination(staging_root, relative)
        _copy_exact_once(source, target, reference["byte_sha256"])
        copied[field] = relative
    for field in (
        "fundamental_pointer_file_ref",
        "fundamental_generation_manifest_file_ref",
    ):
        reference = normalized["files"][field]
        relative = f"fundamental_replay/{reference['relative_path']}"
        source, _ = _verify_input(input_root, reference)
        target = _destination(staging_root, relative)
        _copy_exact_once(source, target, reference["byte_sha256"])
        copied[field] = relative
    for field, prefix in (
        ("calendar_raw_file_refs", "closure/calendar_raw"),
        ("calendar_capture_file_refs", "closure/calendar_captures"),
        ("market_table_file_refs", "closure/market_tables"),
    ):
        rows: list[str] = []
        for index, reference in enumerate(normalized[field]):
            suffix = PurePosixPath(reference["relative_path"]).suffix.lower()
            relative = f"{prefix}/{index:04d}{suffix}"
            source, _ = _verify_input(input_root, reference)
            target = _destination(staging_root, relative)
            _copy_exact_once(source, target, reference["byte_sha256"])
            rows.append(relative)
        copied[field] = rows
    for field in ("fundamental_table_file_refs", "fundamental_evidence_file_refs"):
        rows = []
        for reference in normalized[field]:
            relative = f"fundamental_replay/{reference['relative_path']}"
            source, _ = _verify_input(input_root, reference)
            target = _destination(staging_root, relative)
            _copy_exact_once(source, target, reference["byte_sha256"])
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
) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    required = {"ts_code", "trade_date", "adj_close", "amount", "vol", "total_mv"}
    scope = set(closure["scope_symbols"])
    eligible = set(closure["eligible_symbols"])
    cutoff = closure["cutoff"]
    rows: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, str]] = set()
    cutoff_market_caps: dict[str, float] = {}
    sessions: set[str] = set()
    for relative in _copied_paths(copied, "market_table_file_refs"):
        path = staging_root / relative
        try:
            parquet = pq.ParquetFile(path)
            if not required <= set(parquet.schema_arrow.names):
                raise SystemContractError("raw market table columns are incomplete")
            table = parquet.read(columns=sorted(required), use_threads=False)
        except SystemContractError:
            raise
        except Exception as exc:
            raise SystemContractError("raw market table decode failed") from exc
        for index, row in enumerate(table.to_pylist()):
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
    try:
        parquet = pq.ParquetFile(membership_path)
        if not required <= set(parquet.schema_arrow.names):
            raise SystemContractError("PIT membership columns are incomplete")
        table = parquet.read(columns=sorted(required), use_threads=False)
    except SystemContractError:
        raise
    except Exception as exc:
        raise SystemContractError("PIT membership decode failed") from exc
    records: dict[str, dict[str, Any]] = {}
    scope = set(closure["scope_symbols"])
    for row in table.to_pylist():
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
    )
    pit_rows = _read_raw_pit_rows(
        staging_root=staging_root,
        copied=copied,
        closure=closure,
        cutoff_market_caps=market_caps,
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


def _validate_fundamental_closure(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    staging_root: Path,
    copied: Mapping[str, list[str] | str],
    expected_cutoff: str,
) -> dict[str, Any]:
    pointer = _read_copied_json(staging_root, copied, "fundamental_pointer_file_ref")
    manifest = _read_copied_json(staging_root, copied, "fundamental_generation_manifest_file_ref")
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

    replay_root = staging_root / "fundamental_replay"
    try:
        validated = validate_successor_provenance(
            pointer,
            manifest,
            generation_root=replay_root,
            historical_only=True,
        )
    except (SafeSuccessorError, OSError, ValueError) as exc:
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


def _validate_calendar_capture(  # noqa: C901
    *,
    normalized: Mapping[str, Any],
    input_root: Path,
    capture_ref: Mapping[str, str],
    raw_ref: Mapping[str, str],
    exchange: str,
    issuer: str,
    source_url: str,
    captured_at: str,
    coverage_start: str,
    cutoff: str,
    sessions: Sequence[str],
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
    if (
        raw_stored_size != len(raw)
        or payload["raw_sha256"] != _sha256(raw)
        or payload["raw_sha256"] != raw_ref["byte_sha256"]
        or payload["raw_byte_length"] != len(raw)
        or payload["raw_media_type"] != "application/json"
        or payload["decoder_id"] != _OFFICIAL_CALENDAR_DECODERS[exchange]
        or payload["decoder_sha256"] != _OFFICIAL_CALENDAR_DECODER_SHA256
    ):
        raise SystemContractError("official calendar raw/decoder identity differs")
    try:
        raw_document = parse_canonical_json_bytes(raw, label="official calendar raw response")
    except ContractError as exc:
        raise SystemContractError("official calendar raw response is not canonical JSON") from exc
    if type(raw_document) is not dict or set(raw_document) != _OFFICIAL_RAW_CALENDAR_FIELDS:
        raise SystemContractError("official calendar raw response fields are not exact")
    if (
        payload["state"] != "IMMUTABLE"
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
        or payload["session_intervals"] != _CN_CONTINUOUS_SESSION_INTERVALS
        or payload["coverage_start_date"] != coverage_start
        or payload["cutoff_date"] != cutoff
        or payload["transform_code_sha256"] != transform_code_sha256
        or raw_document["schema_version"] != "official-exchange-calendar-response.v1"
        or raw_document["exchange_id"] != exchange
        or raw_document["issuer"] != issuer
        or raw_document["timezone"] != payload["timezone"]
        or raw_document["session_intervals"] != payload["session_intervals"]
        or raw_document["coverage_start_date"] != coverage_start
        or raw_document["cutoff_date"] != cutoff
        or raw_document["daily_status_rows"] != payload["daily_status_rows"]
    ):
        raise SystemContractError("official calendar capture binding differs")
    _identifier(payload["calendar_capture_id"], label="calendar_capture_id")
    rows = payload["daily_status_rows"]
    if type(rows) is not list or not rows:
        raise SystemContractError("official daily OPEN/CLOSED evidence is absent")
    expected_date = datetime.strptime(coverage_start, "%Y-%m-%d").date()
    cutoff_date = datetime.strptime(cutoff, "%Y-%m-%d").date()
    observed_open: list[str] = []
    closed_count = 0
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
    if capture_ref not in normalized["calendar_capture_file_refs"]:
        raise SystemContractError("official calendar capture is outside request closure")
    return capture


def _validate_calendar_manifest(  # noqa: C901
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
    _sha(payload["transform_code_sha256"], label="transform_code_sha256")
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
    raw_refs = normalized["calendar_raw_file_refs"]
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
        source_url = row["source_url"]
        parsed_url = urlsplit(source_url) if type(source_url) is str else None
        if (
            row["issuer"] != issuer
            or parsed_url is None
            or parsed_url.scheme != "https"
            or parsed_url.hostname != hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise SystemContractError("official exchange source authority differs")
        _timestamp(row["captured_at"], label=f"exchange_rows[{index}].captured_at")
        raw_ref = _file_ref(row["raw_file_ref"], label=f"exchange_rows[{index}].raw_file_ref")
        capture_ref = _file_ref(
            row["capture_file_ref"],
            label=f"exchange_rows[{index}].capture_file_ref",
        )
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
        _validate_calendar_capture(
            normalized=normalized,
            input_root=input_root,
            capture_ref=capture_ref,
            raw_ref=raw_ref,
            exchange=exchange,
            issuer=issuer,
            source_url=source_url,
            captured_at=row["captured_at"],
            coverage_start=coverage_start,
            cutoff=cutoff,
            sessions=sessions,
            transform_code_sha256=payload["transform_code_sha256"],
        )
        observed_exchanges.append(exchange)
        observed_raw_refs.append(raw_ref)
        observed_capture_refs.append(capture_ref)
    if (
        observed_exchanges != expected_exchanges
        or observed_exchanges != sorted(set(observed_exchanges))
        or observed_raw_refs != raw_refs
        or observed_capture_refs != normalized["calendar_capture_file_refs"]
    ):
        raise SystemContractError("official calendar/PIT exchange closure differs")
    return manifest


def assemble_production_bootstrap(  # noqa: C901
    *,
    workspace_root: str | os.PathLike[str],
    input_root: str | os.PathLike[str],
    request_raw: bytes,
) -> dict[str, Any]:
    """Materialize, validate, and offline-assemble; never activate."""

    normalized = validate_bootstrap_operator_request(request_raw)
    workspace = Path(workspace_root).resolve(strict=True)
    inputs = Path(input_root).resolve(strict=True)
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
    staging = _ensure_staging_root(workspace, normalized["operation_id"])
    store = SystemStore(
        workspace,
        source_root=staging,
        source_root_id=normalized["source_root_id"],
    )
    if store.read_active() is not None:
        raise SystemPreconditionError("bootstrap assembly requires EMPTY System pointer")
    if (workspace / str(MIGRATION_MARKER_PATH)).exists():
        raise SystemPreconditionError("bootstrap assembly requires absent migration marker")
    release_ref = normalized["release_manifest_ref"]
    release = store.get_object(release_ref)
    if release["kind"] != "system.release":
        raise SystemContractError("deployed release ref kind is invalid")

    copied = _copy_request_inputs(
        normalized=normalized,
        input_root=inputs,
        staging_root=staging,
    )
    materialized = _materialize_market_and_pit(
        normalized=normalized,
        staging_root=staging,
        copied=copied,
    )
    _validate_fundamental_closure(
        normalized=normalized,
        staging_root=staging,
        copied=copied,
        expected_cutoff=materialized["cutoff"],
    )
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
        materialized["pit_universe_relative"],
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
        created_at=created_at,
    )
    market_ref = _source(
        store,
        materialized["market_history_relative"],
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
    _validate_calendar_manifest(
        normalized=normalized,
        input_root=inputs,
        cohort_symbols=source_projection["all_pit_symbols"],
        source_projection=source_projection,
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
        generated_rows["implementation"][0],
        source_format="JSON",
        media_type="application/json",
        created_at=created_at,
    )
    recomputation_ref = _source(
        store,
        generated_rows["recomputation"][0],
        source_format="JSON",
        media_type="application/json",
        created_at=created_at,
    )
    source_generation_ref = _source(
        store,
        generated_rows["source_generation"][0],
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
        source_blockers=normalized["source_blockers"],
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
        "calendar_manifest_file_ref",
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
    calendar_raw_refs = [
        _source(
            store,
            relative,
            source_format="BINARY",
            media_type="application/octet-stream",
            created_at=created_at,
        )
        for relative in _copied_paths(copied, "calendar_raw_file_refs")
    ]
    calendar_capture_refs = [
        _source(
            store,
            relative,
            source_format="JSON",
            media_type="application/json",
            created_at=created_at,
        )
        for relative in _copied_paths(copied, "calendar_capture_file_refs")
    ]
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
            ("manifest", raw_objects["calendar_manifest_file_ref"]),
            *[(f"official-raw-{index:04d}", ref) for index, ref in enumerate(calendar_raw_refs)],
            *[
                (f"official-capture-{index:04d}", ref)
                for index, ref in enumerate(calendar_capture_refs)
            ],
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
        "research_refs": [],
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
    if (workspace / str(ACTIVE_POINTER_PATH)).exists() or (
        workspace / str(MIGRATION_MARKER_PATH)
    ).exists():
        raise SystemStorageError("bootstrap assembler touched production authority paths")
    return {
        "status": "OFFLINE_VERIFIED",
        "operation_id": normalized["operation_id"],
        "source_root": str(staging),
        "assembly_request_ref": assembly_request_ref,
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
]
