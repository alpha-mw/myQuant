"""Atomic generation storage for the CN PIT fundamental mart."""

from __future__ import annotations

import fcntl
import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from .fundamental_provider_contract import (
    FUNDAMENTAL_DERIVATION_CONTRACT,
    FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA,
    FUNDAMENTAL_FETCH_CHECKPOINT_POINTER_SCHEMA,
    FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA,
    FUNDAMENTAL_FETCH_PIT_CONTRACT,
    FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA,
    HARD_INVALID_SUBCOUNTER_FIELDS,
    assert_frame_semantics_equal,
    canonical_json_sha256,
    frame_fingerprint,
    frame_logical_schema,
    strict_nonnegative_int,
    validate_outcome_accounting_v3,
)


FUNDAMENTAL_POINTER_FILENAME = "_fundamental_latest.json"
FUNDAMENTAL_GENERATIONS_DIRNAME = "_fundamental_generations"
FUNDAMENTAL_TABLES = (
    "fundamental_period",
    "fundamental_daily",
    "fundamental_quarantine",
)
FUNDAMENTAL_RAW_TABLES = (
    "fina_indicator",
    "income",
    "balancesheet",
    "cashflow",
    "daily_basic",
    "forecast",
)
PRIMARY_PROVENANCE_SCHEMA_VERSION = "cn-fundamental-primary-provenance.v2"
FUNDAMENTAL_PROMOTION_LOCK_FILENAME = ".fundamental-promotion.lock"
FUNDAMENTAL_HISTORY_MIN_MONTHLY_COVERAGE = 0.90
FUNDAMENTAL_HISTORY_MAX_CONSECUTIVE_MISSING_MONTHS = 2
FUNDAMENTAL_HISTORY_BOUNDARY_TOLERANCE_DAYS = 31


class FundamentalGenerationError(ValueError):
    """Raised when a fundamental generation cannot be read or published."""


_PRIMARY_GENERATION_CAPABILITY = object()


@dataclass(frozen=True)
class _PrimaryGenerationAttestation:
    capability: object
    source: str
    provider_manifest_sha256: str
    raw_table_fingerprints: tuple[tuple[str, str], ...]
    metadata_sha256: str
    table_fingerprints: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _CapturedFundamentalGeneration:
    generation_id: str
    pointer: dict[str, Any]
    manifest_bytes: bytes
    table_bytes: dict[str, bytes]
    replay_projection_fingerprints: dict[str, str]


@dataclass(frozen=True)
class _CapturedProviderCheckpoint:
    root: Path
    generation_id: str
    revision: int
    pointer_sha256: str
    manifest_sha256: str
    binding_sha256: str
    outcome_accounting_sha256: str
    table_evidence_sha256: str
    tables: dict[str, pd.DataFrame]
    outcomes: list[dict[str, Any]]


def _metadata_sha256(metadata: Mapping[str, Any]) -> str:
    try:
        return canonical_json_sha256(dict(metadata))
    except (TypeError, ValueError) as exc:
        raise FundamentalGenerationError(
            "fundamental generation metadata is not canonical JSON"
        ) from exc


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise FundamentalGenerationError(
            "fundamental generation tables must contain pandas DataFrames"
        )
    try:
        return frame_fingerprint(frame)
    except (TypeError, ValueError) as exc:
        raise FundamentalGenerationError(
            "fundamental generation table fingerprint failed"
        ) from exc


def _table_fingerprints(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[tuple[str, str], ...]:
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental publish table set mismatch")
    return tuple(
        (table_name, _frame_fingerprint(tables[table_name]))
        for table_name in FUNDAMENTAL_TABLES
    )


def _issue_primary_generation_attestation(
    *,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    source: str,
    provider_manifest_sha256: str,
    raw_table_fingerprints: tuple[tuple[str, str], ...],
) -> _PrimaryGenerationAttestation:
    if str(metadata.get("source_priority") or "").strip() != "tushare_primary":
        raise FundamentalGenerationError(
            "primary generation attestation requires tushare_primary metadata"
        )
    resolved_source = str(source or "").strip()
    if resolved_source not in {"live_tushare", "live_tushare_partial"}:
        raise FundamentalGenerationError(
            "primary generation attestation requires live Tushare source"
        )
    if str(metadata.get("provider_status") or "").strip() != resolved_source:
        raise FundamentalGenerationError(
            "primary generation provider status/source mismatch"
        )
    if (
        str(metadata.get("source_provenance") or "").strip()
        != "live_tushare_explicit"
    ):
        raise FundamentalGenerationError(
            "primary generation source provenance is not live Tushare"
        )
    if metadata.get("gate2_passed") is not True:
        raise FundamentalGenerationError(
            "primary generation requires a passed readiness gate"
        )
    manifest_hash = str(provider_manifest_sha256 or "").strip().lower()
    if not _valid_sha256(manifest_hash):
        raise FundamentalGenerationError(
            "primary generation provider manifest hash is invalid"
        )
    provider_manifest = dict(metadata.get("provider_manifest", {}) or {})
    if _metadata_sha256(provider_manifest) != manifest_hash:
        raise FundamentalGenerationError(
            "primary generation provider manifest hash mismatch"
        )
    raw_fingerprints = tuple(raw_table_fingerprints)
    if not _valid_named_fingerprints(
        raw_fingerprints,
        expected_names=FUNDAMENTAL_RAW_TABLES,
    ):
        raise FundamentalGenerationError(
            "primary generation raw table fingerprints are invalid"
        )
    return _PrimaryGenerationAttestation(
        capability=_PRIMARY_GENERATION_CAPABILITY,
        source=resolved_source,
        provider_manifest_sha256=manifest_hash,
        raw_table_fingerprints=raw_fingerprints,
        metadata_sha256=_metadata_sha256(metadata),
        table_fingerprints=_table_fingerprints(tables),
    )


def _primary_generation_attestation_matches(
    attestation: _PrimaryGenerationAttestation | None,
    *,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> bool:
    if not isinstance(attestation, _PrimaryGenerationAttestation):
        return False
    try:
        return bool(
            attestation.capability is _PRIMARY_GENERATION_CAPABILITY
            and attestation.metadata_sha256 == _metadata_sha256(metadata)
            and attestation.table_fingerprints == _table_fingerprints(tables)
        )
    except FundamentalGenerationError:
        return False


def _valid_sha256(value: str) -> bool:
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_named_fingerprints(
    values: tuple[tuple[str, str], ...],
    *,
    expected_names: tuple[str, ...],
) -> bool:
    return bool(
        tuple(name for name, _digest in values) == expected_names
        and all(_valid_sha256(str(digest)) for _name, digest in values)
    )


def _primary_provenance_envelope(
    attestation: _PrimaryGenerationAttestation,
    *,
    metadata: Mapping[str, Any],
    table_manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": PRIMARY_PROVENANCE_SCHEMA_VERSION,
        "status": "verified_live_tushare",
        "source": attestation.source,
        "source_priority": "tushare_primary",
        "source_provenance": "live_tushare_explicit",
        "provider_manifest_sha256": attestation.provider_manifest_sha256,
        "metadata_sha256": _metadata_sha256(metadata),
        "raw_table_fingerprints": dict(attestation.raw_table_fingerprints),
        "output_frame_fingerprints": {
            table_name: str(
                table_manifest[table_name].get("frame_fingerprint") or ""
            )
            for table_name in FUNDAMENTAL_TABLES
        },
        "output_parquet_sha256": {
            table_name: str(table_manifest[table_name].get("sha256") or "")
            for table_name in FUNDAMENTAL_TABLES
        },
    }
    body["envelope_sha256"] = _metadata_sha256(body)
    return body


def _verify_primary_provenance(
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    pointer_metadata = dict(pointer.get("metadata", {}) or {})
    manifest_metadata = dict(manifest.get("metadata", {}) or {})
    pointer_priority = str(pointer_metadata.get("source_priority") or "").strip()
    manifest_priority = str(manifest_metadata.get("source_priority") or "").strip()
    pointer_envelope = pointer.get("primary_provenance")
    manifest_envelope = manifest.get("primary_provenance")
    if pointer_priority != "tushare_primary" and manifest_priority != "tushare_primary":
        if pointer_envelope is not None or manifest_envelope is not None:
            raise FundamentalGenerationError(
                "non-primary fundamental generation has primary provenance"
            )
        return False
    if pointer_priority != manifest_priority:
        raise FundamentalGenerationError(
            "fundamental primary source priority mismatch"
        )
    if not isinstance(pointer_envelope, Mapping) or not isinstance(
        manifest_envelope,
        Mapping,
    ):
        raise FundamentalGenerationError(
            "fundamental primary provenance envelope missing"
        )
    envelope = dict(manifest_envelope)
    if dict(pointer_envelope) != envelope:
        raise FundamentalGenerationError(
            "fundamental primary provenance pointer/manifest mismatch"
        )
    envelope_hash = str(envelope.pop("envelope_sha256", "")).strip().lower()
    if not _valid_sha256(envelope_hash) or _metadata_sha256(envelope) != envelope_hash:
        raise FundamentalGenerationError(
            "fundamental primary provenance envelope hash mismatch"
        )
    if (
        envelope.get("schema_version") != PRIMARY_PROVENANCE_SCHEMA_VERSION
        or envelope.get("status") != "verified_live_tushare"
        or str(envelope.get("source") or "").strip()
        not in {"live_tushare", "live_tushare_partial"}
        or envelope.get("source_priority") != "tushare_primary"
        or envelope.get("source_provenance") != "live_tushare_explicit"
    ):
        raise FundamentalGenerationError(
            "fundamental primary provenance contract mismatch"
        )
    if str(manifest_metadata.get("provider_status") or "").strip() != str(
        envelope.get("source") or ""
    ).strip():
        raise FundamentalGenerationError(
            "fundamental primary provenance source mismatch"
        )
    if (
        str(manifest_metadata.get("source_provenance") or "").strip()
        != "live_tushare_explicit"
    ):
        raise FundamentalGenerationError(
            "fundamental primary provenance source lineage mismatch"
        )
    if (
        manifest_metadata.get("gate2_passed") is not True
        or pointer_metadata.get("gate2_passed") is not True
    ):
        raise FundamentalGenerationError(
            "fundamental primary provenance readiness gate mismatch"
        )
    provider_manifest_hash = str(
        envelope.get("provider_manifest_sha256") or ""
    ).strip().lower()
    if (
        not _valid_sha256(provider_manifest_hash)
        or _metadata_sha256(
            dict(manifest_metadata.get("provider_manifest", {}) or {})
        )
        != provider_manifest_hash
    ):
        raise FundamentalGenerationError(
            "fundamental primary provider manifest hash mismatch"
        )
    if str(envelope.get("metadata_sha256") or "").strip().lower() != (
        _metadata_sha256(manifest_metadata)
    ):
        raise FundamentalGenerationError(
            "fundamental primary metadata hash mismatch"
        )
    raw_fingerprint_map = dict(
        envelope.get("raw_table_fingerprints", {}) or {}
    )
    raw_fingerprints = tuple(
        (
            table_name,
            str(raw_fingerprint_map.get(table_name) or "").strip().lower(),
        )
        for table_name in FUNDAMENTAL_RAW_TABLES
    )
    if (
        set(raw_fingerprint_map) != set(FUNDAMENTAL_RAW_TABLES)
        or not _valid_named_fingerprints(
            raw_fingerprints,
            expected_names=FUNDAMENTAL_RAW_TABLES,
        )
    ):
        raise FundamentalGenerationError(
            "fundamental primary raw fingerprints mismatch"
        )
    output_fingerprint_map = dict(
        envelope.get("output_frame_fingerprints", {}) or {}
    )
    output_frame_fingerprints = tuple(
        (
            table_name,
            str(output_fingerprint_map.get(table_name) or "")
            .strip()
            .lower(),
        )
        for table_name in FUNDAMENTAL_TABLES
    )
    if (
        set(output_fingerprint_map) != set(FUNDAMENTAL_TABLES)
        or not _valid_named_fingerprints(
            output_frame_fingerprints,
            expected_names=FUNDAMENTAL_TABLES,
        )
    ):
        raise FundamentalGenerationError(
            "fundamental primary output fingerprints mismatch"
        )
    manifest_tables = dict(manifest.get("tables", {}) or {})
    manifest_frame_fingerprints = tuple(
        (
            table_name,
            str(
                dict(manifest_tables.get(table_name, {}) or {}).get(
                    "frame_fingerprint"
                )
                or ""
            )
            .strip()
            .lower(),
        )
        for table_name in FUNDAMENTAL_TABLES
    )
    if (
        not _valid_named_fingerprints(
            manifest_frame_fingerprints,
            expected_names=FUNDAMENTAL_TABLES,
        )
        or output_frame_fingerprints != manifest_frame_fingerprints
    ):
        raise FundamentalGenerationError(
            "fundamental primary output fingerprints do not bind manifest"
        )
    expected_parquet_hashes = {
        table_name: str(
            manifest_tables
            .get(table_name, {})
            .get("sha256", "")
        ).strip().lower()
        for table_name in FUNDAMENTAL_TABLES
    }
    if dict(envelope.get("output_parquet_sha256", {}) or {}) != (
        expected_parquet_hashes
    ) or not all(_valid_sha256(value) for value in expected_parquet_hashes.values()):
        raise FundamentalGenerationError(
            "fundamental primary parquet provenance mismatch"
        )
    return True


def _tables_claim_primary(
    tables: Mapping[str, pd.DataFrame],
) -> bool:
    for table_name in FUNDAMENTAL_TABLES:
        frame = tables.get(table_name)
        if not isinstance(frame, pd.DataFrame) or "source_priority" not in frame.columns:
            continue
        priorities = {
            str(value).strip()
            for value in frame["source_priority"].dropna().tolist()
        }
        if "tushare_primary" in priorities:
            return True
    return False


def fundamental_data_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if path.suffix.lower() == ".parquet":
        return path.parent.parent
    if path.name in FUNDAMENTAL_TABLES:
        return path.parent
    return path


def _absolute_data_root(root: str | Path) -> Path:
    raw = fundamental_data_root(root)
    if ".." in raw.parts:
        raise FundamentalGenerationError(
            "fundamental root must not contain parent traversal"
        )
    return raw if raw.is_absolute() else Path.cwd().resolve(strict=True) / raw


def _read_data_root(root: str | Path) -> Path:
    absolute = _absolute_data_root(root)
    cursor = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            return cursor.joinpath(*parts[index + 1:])
        except OSError as exc:
            raise FundamentalGenerationError(
                "fundamental root unreadable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FundamentalGenerationError(
                "fundamental root ancestor symlink rejected"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise FundamentalGenerationError(
                "fundamental root is not a directory"
            )
    return cursor.resolve(strict=True)


def _write_data_root(root: str | Path) -> Path:
    absolute = _absolute_data_root(root)
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            try:
                os.mkdir(cursor, mode=0o700)
                metadata = os.lstat(cursor)
            except FileExistsError:
                metadata = os.lstat(cursor)
        except OSError as exc:
            raise FundamentalGenerationError(
                "fundamental root unwritable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FundamentalGenerationError(
                "fundamental root ancestor symlink rejected"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise FundamentalGenerationError(
                "fundamental root is not a directory"
            )
    return cursor.resolve(strict=True)


def legacy_fundamental_table_path(
    root: str | Path,
    table_name: str,
) -> Path:
    if table_name not in FUNDAMENTAL_TABLES:
        raise FundamentalGenerationError(f"unsupported fundamental table: {table_name}")
    path = Path(root).expanduser()
    if path.suffix.lower() == ".parquet":
        return path
    base = fundamental_data_root(path)
    return base / table_name / "part.parquet"


def _safe_generation_id(run_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "").strip())
    value = value.strip("._")
    if not value:
        raise FundamentalGenerationError("fundamental generation run_id is empty")
    return value


def _file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_bytes(path: Path) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FundamentalGenerationError(
                f"fundamental artifact is not a regular file: {path}"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise FundamentalGenerationError(
            f"fundamental artifact unreadable: {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(before):
            raise FundamentalGenerationError(
                f"fundamental artifact changed during open: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        signature = _file_signature(before)
        if (
            _file_signature(after) != signature
            or _file_signature(current) != signature
        ):
            raise FundamentalGenerationError(
                f"fundamental artifact changed during read: {path}"
            )
        return b"".join(chunks), signature
    except OSError as exc:
        raise FundamentalGenerationError(
            f"fundamental artifact changed during read: {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _readback_table_contract(
    payload: bytes,
    *,
    table_name: str,
    table_manifest: Mapping[str, Any] | None = None,
    require_v2: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Authenticate one table from the exact captured Parquet bytes."""

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    declared = dict(table_manifest) if table_manifest is not None else None
    if declared is not None:
        declared_sha256 = str(declared.get("sha256") or "").strip().lower()
        if (
            not _valid_sha256(declared_sha256)
            or declared_sha256 != actual_sha256
        ):
            raise FundamentalGenerationError(
                f"fundamental table hash mismatch: {table_name}"
            )
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
        fingerprint = frame_fingerprint(frame)
        logical_schema = frame_logical_schema(frame)
    except Exception as exc:
        raise FundamentalGenerationError(
            f"fundamental table readback contract failed: {table_name}"
        ) from exc
    evidence: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": actual_sha256,
        "frame_fingerprint": fingerprint,
        "logical_schema": logical_schema,
    }
    if declared is None:
        return frame, evidence
    if "rows" in declared and (
        type(declared.get("rows")) is not int
        or declared.get("rows") != evidence["rows"]
    ):
        raise FundamentalGenerationError(
            f"fundamental table row count mismatch: {table_name}"
        )
    if "columns" in declared and (
        not isinstance(declared.get("columns"), list)
        or declared.get("columns") != evidence["columns"]
    ):
        raise FundamentalGenerationError(
            f"fundamental table columns mismatch: {table_name}"
        )
    v2_fields = {"rows", "columns", "frame_fingerprint", "logical_schema"}
    v2_identity_fields = {"frame_fingerprint", "logical_schema"}
    present_v2_identity_fields = v2_identity_fields.intersection(declared)
    if require_v2 and v2_fields.intersection(declared) != v2_fields:
        raise FundamentalGenerationError(
            f"fundamental table readback contract missing: {table_name}"
        )
    if (
        present_v2_identity_fields
        and v2_fields.intersection(declared) != v2_fields
    ):
        raise FundamentalGenerationError(
            f"fundamental table readback contract incomplete: {table_name}"
        )
    if not present_v2_identity_fields:
        return frame, evidence
    declared_fingerprint = str(
        declared.get("frame_fingerprint") or ""
    ).strip().lower()
    if (
        not _valid_sha256(declared_fingerprint)
        or declared_fingerprint != evidence["frame_fingerprint"]
    ):
        raise FundamentalGenerationError(
            f"fundamental table frame fingerprint mismatch: {table_name}"
        )
    if declared.get("logical_schema") != evidence["logical_schema"]:
        raise FundamentalGenerationError(
            f"fundamental table logical schema mismatch: {table_name}"
        )
    return frame, evidence


def _json_object_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise FundamentalGenerationError(
            f"invalid fundamental {label} JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise FundamentalGenerationError(
            f"fundamental {label} JSON object required"
        )
    return dict(value)


def _resolve_inside(base: Path, value: str, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise FundamentalGenerationError(f"{label} path is empty")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise FundamentalGenerationError(
            f"{label} must be a safe relative fundamental path"
        )
    cursor = base
    for part in candidate.parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise FundamentalGenerationError(
                f"{label} path missing: {cursor}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FundamentalGenerationError(
                f"{label} path contains symlink: {cursor}"
            )
    resolved = (base / candidate).resolve(strict=True)
    base_resolved = base.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise FundamentalGenerationError(f"{label} escapes fundamental root: {resolved}")
    return resolved


def _strict_provider_int(value: Any, *, label: str) -> int:
    try:
        return strict_nonnegative_int(value, label=label)
    except (TypeError, ValueError) as exc:
        raise FundamentalGenerationError(str(exc)) from exc


def _strict_provider_ratio(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FundamentalGenerationError(f"{label} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise FundamentalGenerationError(f"{label} must be between 0 and 1")
    return resolved


def _safe_checkpoint_root(value: Any) -> Path:
    raw = Path(str(value or "").strip()).expanduser()
    if not raw.is_absolute() or ".." in raw.parts:
        raise FundamentalGenerationError(
            "staged fundamental checkpoint root must be an absolute safe path"
        )
    try:
        metadata = os.lstat(raw)
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise FundamentalGenerationError(
            "staged fundamental checkpoint root is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != raw
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint root is unsafe"
        )
    return resolved


def _capture_provider_checkpoint_v3(
    provider: Mapping[str, Any],
) -> _CapturedProviderCheckpoint:
    """Capture and authenticate the exact accepted-raw checkpoint bytes."""

    if provider.get("schema_version") != FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA:
        raise FundamentalGenerationError(
            "staged fundamental provider manifest schema mismatch"
        )
    if provider.get("pit_contract_version") != FUNDAMENTAL_FETCH_PIT_CONTRACT:
        raise FundamentalGenerationError(
            "staged fundamental provider PIT contract mismatch"
        )
    if provider.get("authoritative_full_rebuild") is not True:
        raise FundamentalGenerationError(
            "staged fundamental provider is not an authoritative full rebuild"
        )
    checkpoint_value = provider.get("checkpoint")
    if not isinstance(checkpoint_value, Mapping):
        raise FundamentalGenerationError(
            "staged fundamental provider checkpoint evidence is missing"
        )
    checkpoint = dict(checkpoint_value)
    if checkpoint.get("schema_version") != FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA:
        raise FundamentalGenerationError(
            "staged fundamental checkpoint declaration schema mismatch"
        )
    root = _safe_checkpoint_root(checkpoint.get("root"))
    pointer_bytes, _signature = _stable_file_bytes(root / "latest.json")
    pointer_sha256 = hashlib.sha256(pointer_bytes).hexdigest()
    if pointer_sha256 != str(checkpoint.get("pointer_sha256") or "").lower():
        raise FundamentalGenerationError(
            "staged fundamental checkpoint pointer SHA mismatch"
        )
    pointer = _json_object_from_bytes(
        pointer_bytes,
        label="provider checkpoint pointer",
    )
    if pointer_bytes != _json_bytes(pointer):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint pointer is not canonical JSON"
        )
    if pointer.get("schema_version") != FUNDAMENTAL_FETCH_CHECKPOINT_POINTER_SCHEMA:
        raise FundamentalGenerationError(
            "staged fundamental checkpoint pointer schema mismatch"
        )
    pointer_revision = _strict_provider_int(
        pointer.get("revision"),
        label="checkpoint pointer revision",
    )
    declared_revision = _strict_provider_int(
        checkpoint.get("revision"),
        label="provider checkpoint revision",
    )
    if pointer_revision < 1 or declared_revision != pointer_revision:
        raise FundamentalGenerationError(
            "staged fundamental checkpoint revision mismatch"
        )
    generation_id = str(pointer.get("generation_id") or "").strip()
    if (
        not generation_id
        or _safe_generation_id(generation_id) != generation_id
        or str(checkpoint.get("generation_id") or "").strip() != generation_id
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint generation mismatch"
        )

    manifest_path = _resolve_inside(
        root,
        str(pointer.get("manifest_path") or ""),
        label="provider checkpoint manifest",
    )
    manifest_bytes, _signature = _stable_file_bytes(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        manifest_sha256 != str(pointer.get("manifest_sha256") or "").lower()
        or manifest_sha256 != str(checkpoint.get("manifest_sha256") or "").lower()
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint manifest SHA mismatch"
        )
    manifest = _json_object_from_bytes(
        manifest_bytes,
        label="provider checkpoint manifest",
    )
    if manifest_bytes != _json_bytes(manifest):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint manifest is not canonical JSON"
        )
    if manifest.get("schema_version") != FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA:
        raise FundamentalGenerationError(
            "staged fundamental checkpoint manifest schema mismatch"
        )
    manifest_revision = _strict_provider_int(
        manifest.get("revision"),
        label="checkpoint manifest revision",
    )
    if (
        manifest_revision != pointer_revision
        or str(manifest.get("generation_id") or "") != generation_id
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint pointer/manifest mismatch"
        )
    binding_value = manifest.get("binding")
    if not isinstance(binding_value, Mapping):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint binding is missing"
        )
    binding = dict(binding_value)
    binding_sha256 = canonical_json_sha256(binding)
    if (
        binding_sha256 != str(manifest.get("binding_sha256") or "").lower()
        or binding_sha256 != str(checkpoint.get("binding_sha256") or "").lower()
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint binding SHA mismatch"
        )
    provider_scope = provider.get("canonical_scope_evidence")
    if not isinstance(provider_scope, Mapping):
        raise FundamentalGenerationError(
            "staged fundamental provider canonical scope evidence is missing"
        )
    provider_request_fields = provider.get("request_fields")
    if not isinstance(provider_request_fields, Mapping):
        raise FundamentalGenerationError(
            "staged fundamental provider request fields are invalid"
        )
    if (
        binding.get("pit_contract_version") != FUNDAMENTAL_FETCH_PIT_CONTRACT
        or list(binding.get("tables", []) or []) != list(FUNDAMENTAL_RAW_TABLES)
        or dict(binding.get("request_fields", {}) or {})
        != dict(provider_request_fields)
        or str(binding.get("as_of") or "")
        != str(provider.get("strict_pit_as_of") or "")
        or str(binding.get("daily_start_date") or "")
        != str(provider.get("daily_start_date") or "")
        or str(binding.get("financial_start_date") or "")
        != str(provider.get("financial_start_date") or "")
        or _strict_provider_int(binding.get("years"), label="checkpoint years")
        != _strict_provider_int(provider.get("years"), label="provider years")
        or canonical_json_sha256(dict(binding.get("canonical_scope_evidence", {}) or {}))
        != canonical_json_sha256(dict(provider_scope))
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint/provider binding mismatch"
        )

    table_files_value = manifest.get("table_files")
    if not isinstance(table_files_value, Mapping) or set(table_files_value) != set(
        FUNDAMENTAL_RAW_TABLES
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint table manifest is incomplete"
        )
    generation_root = manifest_path.parent
    tables: dict[str, pd.DataFrame] = {}
    for table_name in FUNDAMENTAL_RAW_TABLES:
        raw_entry = table_files_value.get(table_name)
        if not isinstance(raw_entry, Mapping):
            raise FundamentalGenerationError(
                f"staged fundamental checkpoint table entry is invalid: {table_name}"
            )
        entry = dict(raw_entry)
        table_path = _resolve_inside(
            generation_root,
            str(entry.get("path") or ""),
            label=f"provider checkpoint {table_name}",
        )
        table_bytes, _signature = _stable_file_bytes(table_path)
        if hashlib.sha256(table_bytes).hexdigest() != str(
            entry.get("sha256") or ""
        ).lower():
            raise FundamentalGenerationError(
                f"staged fundamental checkpoint table SHA mismatch: {table_name}"
            )
        try:
            frame = pd.read_parquet(io.BytesIO(table_bytes))
            fingerprint = frame_fingerprint(frame)
            logical_schema = frame_logical_schema(frame)
        except Exception as exc:
            raise FundamentalGenerationError(
                f"staged fundamental checkpoint table readback failed: {table_name}"
            ) from exc
        row_count = _strict_provider_int(
            entry.get("row_count"),
            label=f"checkpoint {table_name} row_count",
        )
        columns = list(frame.columns)
        if (
            row_count != len(frame)
            or entry.get("columns") != columns
            or str(entry.get("frame_fingerprint") or "").lower() != fingerprint
            or entry.get("logical_schema") != logical_schema
        ):
            raise FundamentalGenerationError(
                f"staged fundamental checkpoint table contract mismatch: {table_name}"
            )
        tables[table_name] = frame

    outcomes_entry_value = manifest.get("request_outcomes")
    if not isinstance(outcomes_entry_value, Mapping):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint request outcomes are missing"
        )
    outcomes_entry = dict(outcomes_entry_value)
    outcomes_path = _resolve_inside(
        generation_root,
        str(outcomes_entry.get("path") or ""),
        label="provider checkpoint request outcomes",
    )
    outcomes_bytes, _signature = _stable_file_bytes(outcomes_path)
    if hashlib.sha256(outcomes_bytes).hexdigest() != str(
        outcomes_entry.get("sha256") or ""
    ).lower():
        raise FundamentalGenerationError(
            "staged fundamental checkpoint request outcomes SHA mismatch"
        )
    outcomes_payload = _json_object_from_bytes(
        outcomes_bytes,
        label="provider checkpoint request outcomes",
    )
    if outcomes_bytes != _json_bytes(outcomes_payload):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint outcomes are not canonical JSON"
        )
    outcomes_value = outcomes_payload.get("outcomes")
    if not isinstance(outcomes_value, list):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint outcomes must be a list"
        )
    if _strict_provider_int(
        outcomes_entry.get("count"), label="checkpoint outcome count"
    ) != len(outcomes_value):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint outcome count mismatch"
        )
    outcomes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_outcome in enumerate(outcomes_value):
        if not isinstance(raw_outcome, Mapping):
            raise FundamentalGenerationError(
                "staged fundamental checkpoint outcome must be an object"
            )
        outcome = dict(raw_outcome)
        key = (
            str(outcome.get("symbol") or "").strip().upper(),
            str(outcome.get("table") or "").strip(),
        )
        if not key[0] or key[1] not in FUNDAMENTAL_RAW_TABLES or key in seen:
            raise FundamentalGenerationError(
                "staged fundamental checkpoint outcome identity is invalid"
            )
        seen.add(key)
        try:
            validate_outcome_accounting_v3(
                outcome,
                label=f"checkpoint outcome {index}",
            )
        except (TypeError, ValueError) as exc:
            raise FundamentalGenerationError(str(exc)) from exc
        outcomes.append(outcome)
    if outcomes != sorted(
        outcomes,
        key=lambda item: (str(item.get("symbol")), str(item.get("table"))),
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint outcomes are not sorted"
        )
    outcome_accounting_sha256 = canonical_json_sha256(outcomes)
    if (
        outcome_accounting_sha256
        != str(outcomes_entry.get("accounting_sha256") or "").lower()
        or outcome_accounting_sha256
        != str(manifest.get("outcome_accounting_sha256") or "").lower()
        or outcome_accounting_sha256
        != str(checkpoint.get("outcome_accounting_sha256") or "").lower()
        or outcome_accounting_sha256
        != str(provider.get("request_outcome_accounting_sha256") or "").lower()
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint outcome accounting SHA mismatch"
        )
    provider_outcomes = provider.get("symbol_table_outcomes")
    if not isinstance(provider_outcomes, list) or provider_outcomes != outcomes:
        raise FundamentalGenerationError(
            "staged fundamental provider/checkpoint outcomes mismatch"
        )

    raw_counts_value = provider.get("raw_row_counts")
    raw_fingerprints_value = provider.get("raw_table_fingerprints")
    if not isinstance(raw_counts_value, Mapping) or not isinstance(
        raw_fingerprints_value,
        Mapping,
    ):
        raise FundamentalGenerationError(
            "staged fundamental raw table evidence is missing"
        )
    raw_counts = dict(raw_counts_value)
    raw_fingerprints = dict(raw_fingerprints_value)
    if set(raw_counts) != set(FUNDAMENTAL_RAW_TABLES) or set(
        raw_fingerprints
    ) != set(FUNDAMENTAL_RAW_TABLES):
        raise FundamentalGenerationError(
            "staged fundamental raw table evidence set mismatch"
        )
    for table_name in FUNDAMENTAL_RAW_TABLES:
        declared_count = _strict_provider_int(
            raw_counts.get(table_name),
            label=f"provider raw row count {table_name}",
        )
        outcome_rows = sum(
            _strict_provider_int(
                outcome.get("rows"),
                label=f"provider {table_name} outcome rows",
            )
            for outcome in outcomes
            if str(outcome.get("table") or "") == table_name
        )
        if declared_count != len(tables[table_name]) or outcome_rows != declared_count:
            raise FundamentalGenerationError(
                f"staged fundamental checkpoint raw row count mismatch: {table_name}"
            )
        if str(raw_fingerprints.get(table_name) or "").lower() != frame_fingerprint(
            tables[table_name]
        ):
            raise FundamentalGenerationError(
                f"staged fundamental checkpoint raw fingerprint mismatch: {table_name}"
            )

    table_evidence_sha256 = canonical_json_sha256(dict(table_files_value))
    if (
        table_evidence_sha256
        != str(manifest.get("table_evidence_sha256") or "").lower()
        or table_evidence_sha256
        != str(checkpoint.get("table_evidence_sha256") or "").lower()
    ):
        raise FundamentalGenerationError(
            "staged fundamental checkpoint table evidence SHA mismatch"
        )
    return _CapturedProviderCheckpoint(
        root=root,
        generation_id=generation_id,
        revision=pointer_revision,
        pointer_sha256=pointer_sha256,
        manifest_sha256=manifest_sha256,
        binding_sha256=binding_sha256,
        outcome_accounting_sha256=outcome_accounting_sha256,
        table_evidence_sha256=table_evidence_sha256,
        tables=tables,
        outcomes=outcomes,
    )


@lru_cache(maxsize=8)
def _validate_fundamental_pointer_cached(
    base_text: str,
    pointer_bytes: bytes,
    manifest_bytes: bytes,
    table_signatures: tuple[tuple[Any, ...], ...],
) -> dict[str, Any]:
    base = Path(base_text)
    pointer_path = base / FUNDAMENTAL_POINTER_FILENAME
    payload = _json_object_from_bytes(pointer_bytes, label="pointer")
    if (
        payload.get("schema_version") != "cn-fundamental-pointer.v1"
        or payload.get("status") != "OK"
    ):
        raise FundamentalGenerationError("fundamental pointer status is not OK")
    generation_id = str(payload.get("generation_id", "")).strip()
    if not generation_id:
        raise FundamentalGenerationError("fundamental pointer generation_id missing")
    _resolve_inside(
        base,
        str(payload.get("manifest_path", "")),
        label="manifest",
    )
    manifest = _json_object_from_bytes(manifest_bytes, label="manifest")
    if (
        manifest.get("schema_version") != "cn-fundamental-generation.v1"
        or manifest.get("status") != "OK"
        or str(manifest.get("generation_id", "")) != generation_id
    ):
        raise FundamentalGenerationError("fundamental pointer/manifest generation mismatch")
    tables = dict(payload.get("tables", {}) or {})
    manifest_tables = dict(manifest.get("tables", {}) or {})
    if (
        set(tables) != set(FUNDAMENTAL_TABLES)
        or set(manifest_tables) != set(FUNDAMENTAL_TABLES)
    ):
        raise FundamentalGenerationError("fundamental pointer table set mismatch")
    signatures_by_name = {
        str(item[0]): tuple(int(value) for value in item[1:])
        for item in table_signatures
    }
    primary_claimed = bool(
        str(
            dict(payload.get("metadata", {}) or {}).get("source_priority")
            or ""
        ).strip()
        == "tushare_primary"
        or str(
            dict(manifest.get("metadata", {}) or {}).get("source_priority") or ""
        ).strip()
        == "tushare_primary"
    )
    for table_name, table_value in tables.items():
        table_path = _resolve_inside(base, str(table_value), label=table_name)
        table_bytes, observed_signature = _stable_file_bytes(table_path)
        if observed_signature != signatures_by_name.get(table_name):
            raise FundamentalGenerationError(
                f"fundamental table changed before validation: {table_name}"
            )
        _readback_table_contract(
            table_bytes,
            table_name=table_name,
            table_manifest=dict(manifest_tables[table_name] or {}),
            require_v2=primary_claimed,
        )
    primary_provenance_verified = _verify_primary_provenance(payload, manifest)
    payload["primary_provenance_verified"] = primary_provenance_verified
    pointer_metadata = dict(payload.get("metadata", {}) or {})
    pointer_metadata["primary_provenance_verified"] = primary_provenance_verified
    payload["metadata"] = pointer_metadata
    payload["pointer_path"] = str(pointer_path)
    payload["manifest"] = manifest
    return payload


def load_fundamental_pointer(root: str | Path) -> dict[str, Any] | None:
    base = _read_data_root(root)
    pointer_path = base / FUNDAMENTAL_POINTER_FILENAME
    if not pointer_path.exists():
        return None
    pointer_bytes, _pointer_signature = _stable_file_bytes(pointer_path)
    pointer_payload = _json_object_from_bytes(
        pointer_bytes,
        label="pointer",
    )
    manifest_path = _resolve_inside(
        base,
        str(pointer_payload.get("manifest_path", "")),
        label="manifest",
    )
    manifest_bytes, _manifest_signature = _stable_file_bytes(manifest_path)
    tables = dict(pointer_payload.get("tables", {}) or {})
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental pointer table set mismatch")
    table_signatures: list[tuple[Any, ...]] = []
    for table_name in FUNDAMENTAL_TABLES:
        table_path = _resolve_inside(
            base,
            str(tables.get(table_name, "")),
            label=table_name,
        )
        table_signatures.append(
            (table_name, *_file_signature(os.lstat(table_path)))
        )
    return deepcopy(
        _validate_fundamental_pointer_cached(
            str(base),
            pointer_bytes,
            manifest_bytes,
            tuple(table_signatures),
        )
    )


def pointer_sha256(root: str | Path) -> str:
    """Return the exact current pointer hash, or an empty string when absent."""

    base = _read_data_root(root)
    pointer_path = base / FUNDAMENTAL_POINTER_FILENAME
    try:
        os.lstat(pointer_path)
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise FundamentalGenerationError("fundamental pointer unreadable") from exc
    pointer_bytes, _signature = _stable_file_bytes(pointer_path)
    return hashlib.sha256(pointer_bytes).hexdigest()


def load_fundamental_table(
    root: str | Path,
    table_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse the same immutable bytes that satisfy the generation hash."""

    if table_name not in FUNDAMENTAL_TABLES:
        raise FundamentalGenerationError(
            f"unsupported fundamental table: {table_name}"
        )
    base = _read_data_root(root)
    pointer = load_fundamental_pointer(base)
    if pointer is None:
        raise FundamentalGenerationError("fundamental pointer missing")
    table_path = _resolve_inside(
        base,
        str(dict(pointer.get("tables", {}) or {}).get(table_name, "")),
        label=table_name,
    )
    table_bytes, _signature = _stable_file_bytes(table_path)
    table_manifest = dict(
        dict(pointer.get("manifest", {}) or {})
        .get("tables", {})
        .get(table_name, {})
        or {}
    )
    frame, _evidence = _readback_table_contract(
        table_bytes,
        table_name=table_name,
        table_manifest=table_manifest,
    )
    return frame, pointer


def resolve_fundamental_table_path(
    root: str | Path,
    table_name: str,
) -> Path:
    if Path(root).expanduser().suffix.lower() == ".parquet":
        return Path(root).expanduser()
    base = _read_data_root(root)
    pointer = load_fundamental_pointer(base)
    if pointer is None:
        return legacy_fundamental_table_path(base, table_name)
    return _resolve_inside(
        base,
        str(dict(pointer.get("tables", {}) or {}).get(table_name, "")),
        label=table_name,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise FundamentalGenerationError(
            f"fundamental generation file fsync failed: {path.name}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FundamentalGenerationError(
                f"fundamental generation file is unsafe: {path.name}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_table_roundtrip(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    table_name: str,
) -> None:
    try:
        assert_frame_semantics_equal(
            expected,
            actual,
            label=f"fundamental generation {table_name}",
        )
    except Exception as exc:
        raise FundamentalGenerationError(
            f"fundamental table semantic readback mismatch: {table_name}"
        ) from exc


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FundamentalGenerationError(
            "fundamental promotion pointer is not canonical JSON"
        ) from exc


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise FundamentalGenerationError("fundamental promotion pointer is unsafe") from exc
    if existing is not None and (
        stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)
    ):
        raise FundamentalGenerationError("fundamental promotion pointer is unsafe")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        written, _signature = _stable_file_bytes(temp_path)
        if written != payload:
            raise FundamentalGenerationError("fundamental promotion pointer readback mismatch")
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _directory_identity(path: Path) -> tuple[int, int, int]:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise FundamentalGenerationError("fundamental promotion root changed") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FundamentalGenerationError("fundamental promotion root is unsafe")
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


@contextmanager
def _fundamental_promotion_lock(root: Path) -> Iterator[None]:
    root_identity = _directory_identity(root)
    lock_path = root / FUNDAMENTAL_PROMOTION_LOCK_FILENAME
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise FundamentalGenerationError("fundamental promotion lock is unsafe") from exc
    locked = False
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _file_signature(opened) != _file_signature(current)
            or opened.st_mode & 0o077
        ):
            raise FundamentalGenerationError("fundamental promotion lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        current = os.lstat(lock_path)
        if (
            _file_signature(os.fstat(descriptor)) != _file_signature(current)
            or _directory_identity(root) != root_identity
        ):
            raise FundamentalGenerationError("fundamental promotion lock changed")
    except OSError as exc:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise FundamentalGenerationError("fundamental promotion lock failed") from exc
    except Exception:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise
    try:
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _capture_staged_fundamental_generation(
    staging_root: str | Path,
) -> tuple[Path, _CapturedFundamentalGeneration]:
    base = _read_data_root(staging_root)
    validated = load_fundamental_pointer(base)
    if validated is None:
        raise FundamentalGenerationError("staged fundamental pointer missing")
    if (
        validated.get("primary_provenance_verified") is not True
        or dict(validated.get("metadata", {}) or {}).get("gate2_passed") is not True
    ):
        raise FundamentalGenerationError(
            "staged fundamental generation is not verified primary gate2"
        )

    pointer_bytes, _signature = _stable_file_bytes(base / FUNDAMENTAL_POINTER_FILENAME)
    pointer = _json_object_from_bytes(pointer_bytes, label="staged pointer")
    generation_id = str(pointer.get("generation_id") or "").strip()
    if _safe_generation_id(generation_id) != generation_id:
        raise FundamentalGenerationError("staged fundamental generation_id is unsafe")
    manifest_path = _resolve_inside(
        base,
        str(pointer.get("manifest_path") or ""),
        label="staged manifest",
    )
    manifest_bytes, _signature = _stable_file_bytes(manifest_path)
    manifest = _json_object_from_bytes(
        manifest_bytes,
        label="staged manifest",
    )
    if manifest != dict(validated.get("manifest", {}) or {}):
        raise FundamentalGenerationError("staged fundamental manifest changed after validation")
    if (
        manifest.get("generation_id") != generation_id
        or dict(pointer.get("metadata", {}) or {}).get("gate2_passed") is not True
        or dict(manifest.get("metadata", {}) or {}).get("gate2_passed") is not True
        or dict(pointer.get("primary_provenance", {}) or {}).get("schema_version")
        != PRIMARY_PROVENANCE_SCHEMA_VERSION
        or dict(manifest.get("primary_provenance", {}) or {}).get("schema_version")
        != PRIMARY_PROVENANCE_SCHEMA_VERSION
        or _verify_primary_provenance(pointer, manifest) is not True
    ):
        raise FundamentalGenerationError(
            "staged fundamental generation is not verified primary gate2"
        )

    pointer_tables = dict(pointer.get("tables", {}) or {})
    manifest_tables = dict(manifest.get("tables", {}) or {})
    if (
        set(pointer_tables) != set(FUNDAMENTAL_TABLES)
        or set(manifest_tables) != set(FUNDAMENTAL_TABLES)
    ):
        raise FundamentalGenerationError("staged fundamental table set mismatch")
    captured_tables: dict[str, bytes] = {}
    replay_projection_fingerprints: dict[str, str] = {}
    for table_name in FUNDAMENTAL_TABLES:
        table_path = _resolve_inside(
            base,
            str(pointer_tables.get(table_name) or ""),
            label=f"staged {table_name}",
        )
        table_payload, _signature = _stable_file_bytes(table_path)
        readback, _evidence = _readback_table_contract(
            table_payload,
            table_name=table_name,
            table_manifest=dict(manifest_tables[table_name] or {}),
            require_v2=True,
        )
        replay_projection_fingerprints[table_name] = frame_fingerprint(
            _deterministic_replay_projection(
                readback,
                table_name=table_name,
            )
        )
        del readback
        captured_tables[table_name] = table_payload
    return base, _CapturedFundamentalGeneration(
        generation_id=generation_id,
        pointer=pointer,
        manifest_bytes=manifest_bytes,
        table_bytes=captured_tables,
        replay_projection_fingerprints=replay_projection_fingerprints,
    )


def _promotion_date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    parsed = pd.to_datetime(frame[column], errors="coerce")
    if parsed.isna().any():
        raise FundamentalGenerationError(
            f"staged fundamental date column is invalid: {column}"
        )
    return parsed


def _daily_history_coverage_metrics(
    dates: pd.Series,
    *,
    expected_start: str,
    expected_end: str,
    allow_tail_gap: bool,
) -> dict[str, Any]:
    """Measure per-symbol daily history without allowing clustered rows to pass."""

    parsed = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    start_ts = pd.Timestamp(pd.to_datetime(expected_start, format="%Y%m%d"))
    end_ts = pd.Timestamp(pd.to_datetime(expected_end, format="%Y%m%d"))
    if start_ts > end_ts:
        raise FundamentalGenerationError(
            "fundamental expected daily history window is invalid"
        )
    observed_start = parsed.min() if not parsed.empty else pd.NaT
    observed_end = parsed.max() if not parsed.empty else pd.NaT
    evaluation_end = end_ts
    if allow_tail_gap and not parsed.empty:
        evaluation_end = min(end_ts, pd.Timestamp(observed_end))

    window = parsed[(parsed >= start_ts) & (parsed <= evaluation_end)]
    expected_months = pd.period_range(start=start_ts, end=evaluation_end, freq="M")
    observed_months = pd.PeriodIndex(window.dt.to_period("M").unique(), freq="M")
    observed_month_set = set(observed_months)
    missing_months = [month for month in expected_months if month not in observed_month_set]
    max_consecutive_missing = 0
    current_missing = 0
    for month in expected_months:
        if month in observed_month_set:
            current_missing = 0
        else:
            current_missing += 1
            max_consecutive_missing = max(max_consecutive_missing, current_missing)

    expected_month_count = int(len(expected_months))
    observed_month_count = int(expected_month_count - len(missing_months))
    monthly_coverage_ratio = (
        observed_month_count / expected_month_count if expected_month_count else 0.0
    )
    minimum_rows = max(
        1,
        int(max(0, (evaluation_end - start_ts).days) * 100 / 365),
    )
    tolerance = pd.Timedelta(days=FUNDAMENTAL_HISTORY_BOUNDARY_TOLERANCE_DAYS)
    start_ok = bool(
        not window.empty
        and ((window >= start_ts) & (window <= start_ts + tolerance)).any()
    )
    end_ok = bool(
        allow_tail_gap
        or (
            not window.empty
            and ((window >= end_ts - tolerance) & (window <= end_ts)).any()
        )
    )
    density_ok = int(window.nunique()) >= minimum_rows
    monthly_ok = (
        monthly_coverage_ratio >= FUNDAMENTAL_HISTORY_MIN_MONTHLY_COVERAGE
        and max_consecutive_missing
        <= FUNDAMENTAL_HISTORY_MAX_CONSECUTIVE_MISSING_MONTHS
    )
    return {
        "expected_history_start": start_ts.strftime("%Y%m%d"),
        "expected_history_end": end_ts.strftime("%Y%m%d"),
        "evaluated_history_end": evaluation_end.strftime("%Y%m%d"),
        "observed_history_start": (
            pd.Timestamp(observed_start).strftime("%Y%m%d")
            if not pd.isna(observed_start)
            else ""
        ),
        "observed_history_end": (
            pd.Timestamp(observed_end).strftime("%Y%m%d")
            if not pd.isna(observed_end)
            else ""
        ),
        "observed_history_rows": int(window.nunique()),
        "minimum_history_rows": minimum_rows,
        "expected_history_months": expected_month_count,
        "observed_history_months": observed_month_count,
        "monthly_history_coverage_ratio": float(monthly_coverage_ratio),
        "max_consecutive_missing_months": int(max_consecutive_missing),
        "history_start_complete": start_ok,
        "history_end_complete": end_ok,
        "history_density_complete": density_ok,
        "history_monthly_complete": monthly_ok,
        "history_complete": bool(start_ok and end_ok and density_ok and monthly_ok),
    }


def _membership_eligibility_from_bytes(
    payload: bytes,
    *,
    symbols: list[str],
    as_of: str,
    non_blocking_absent: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Recompute PIT listing/history bounds from the exact membership bytes."""

    try:
        membership = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise FundamentalGenerationError(
            "staged fundamental canonical membership is unreadable"
        ) from exc
    required_columns = {"symbol", "list_date", "effective_from", "effective_to"}
    if not required_columns.issubset(membership.columns):
        raise FundamentalGenerationError(
            "staged fundamental canonical membership schema is invalid"
        )
    symbol_set = set(symbols)
    membership = membership.copy()
    membership["_symbol"] = (
        membership["symbol"].astype("string").fillna("").str.strip().str.upper()
    )
    relevant_membership = membership["_symbol"].isin(symbol_set)
    if membership.loc[
        relevant_membership,
        ["list_date", "effective_from", "effective_to"],
    ].isna().any(axis=None):
        raise FundamentalGenerationError(
            "staged fundamental canonical membership required date is null"
        )
    list_date_text = membership["list_date"].astype("string").fillna("").str.strip()
    list_date_exact = list_date_text.str.fullmatch(r"\d{8}", na=False)
    membership["_list_date"] = pd.to_datetime(
        list_date_text.where(list_date_exact),
        format="%Y%m%d",
        errors="coerce",
    )
    if (
        relevant_membership
        & (~list_date_exact | membership["_list_date"].isna())
    ).any():
        raise FundamentalGenerationError(
            "staged fundamental canonical membership list_date is invalid"
        )
    effective_from_text = (
        membership["effective_from"].astype("string").fillna("").str.strip()
    )
    effective_from_exact = effective_from_text.str.fullmatch(r"\d{8}", na=False)
    membership["_effective_from"] = pd.to_datetime(
        effective_from_text.where(effective_from_exact),
        format="%Y%m%d",
        errors="coerce",
    )
    if (
        relevant_membership
        & (~effective_from_exact | membership["_effective_from"].isna())
    ).any():
        raise FundamentalGenerationError(
            "staged fundamental canonical membership effective_from is invalid"
        )
    effective_to_text = (
        membership["effective_to"].astype("string").fillna("").str.strip()
    )
    effective_to_open = effective_to_text.eq("")
    effective_to_exact = effective_to_text.str.fullmatch(r"\d{8}", na=False)
    membership["_effective_to"] = pd.to_datetime(
        effective_to_text.where(~effective_to_open & effective_to_exact),
        format="%Y%m%d",
        errors="coerce",
    )
    if (
        relevant_membership
        & ~effective_to_open
        & (~effective_to_exact | membership["_effective_to"].isna())
    ).any():
        raise FundamentalGenerationError(
            "staged fundamental canonical membership effective_to is invalid"
        )
    membership["_effective_to_open"] = effective_to_open
    if "delist_date" in membership.columns:
        delist_date_text = (
            membership["delist_date"].astype("string").fillna("").str.strip()
        )
    else:
        delist_date_text = pd.Series("", index=membership.index, dtype="string")
    delist_date_empty = delist_date_text.eq("")
    delist_date_exact = delist_date_text.str.fullmatch(r"\d{8}", na=False)
    membership["_delist_date"] = pd.to_datetime(
        delist_date_text.where(~delist_date_empty & delist_date_exact),
        format="%Y%m%d",
        errors="coerce",
    )
    if (
        relevant_membership
        & ~delist_date_empty
        & (~delist_date_exact | membership["_delist_date"].isna())
    ).any():
        raise FundamentalGenerationError(
            "staged fundamental canonical membership delist_date is invalid"
        )
    as_of_ts = pd.Timestamp(pd.to_datetime(as_of, format="%Y%m%d"))
    listing_dates: dict[str, str] = {}
    history_end_dates: dict[str, str] = {}

    for symbol in symbols:
        rows = membership[
            (membership["_symbol"] == symbol)
            & membership["_effective_from"].notna()
            & membership["_effective_from"].le(as_of_ts)
        ].sort_values("_effective_from")
        if rows.empty:
            raise FundamentalGenerationError(
                f"staged fundamental canonical membership missing symbol: {symbol}"
            )
        active_rows = rows[
            rows["_effective_to_open"]
            | rows["_effective_to"].ge(as_of_ts)
        ]
        selected_active = not active_rows.empty
        if selected_active:
            row = active_rows.iloc[-1]
        elif symbol in non_blocking_absent:
            row = rows.iloc[-1]
        else:
            raise FundamentalGenerationError(
                f"staged fundamental canonical membership interval expired: {symbol}"
            )
        list_date = row["_list_date"].strftime("%Y%m%d")
        if list_date > as_of:
            raise FundamentalGenerationError(
                f"staged fundamental canonical membership list_date is future: {symbol}"
            )
        end_candidates: list[str] = []
        effective_to = row.get("_effective_to")
        if not pd.isna(effective_to):
            effective_to_date = pd.Timestamp(effective_to).strftime("%Y%m%d")
            if effective_to_date <= as_of:
                end_candidates.append(effective_to_date)
        raw_delist_date = row.get("_delist_date")
        if not pd.isna(raw_delist_date):
            delist_date = pd.Timestamp(raw_delist_date).strftime("%Y%m%d")
            if delist_date <= as_of:
                end_candidates.append(delist_date)
        if (
            selected_active
            and any(candidate < as_of for candidate in end_candidates)
            and symbol not in non_blocking_absent
        ):
            raise FundamentalGenerationError(
                f"staged fundamental canonical membership active interval conflicts: {symbol}"
            )
        if not selected_active and not end_candidates:
            raise FundamentalGenerationError(
                f"staged fundamental canonical membership expired without end: {symbol}"
            )
        history_end = max(end_candidates) if end_candidates else as_of
        effective_from = row["_effective_from"].strftime("%Y%m%d")
        if history_end < list_date or history_end < effective_from:
            raise FundamentalGenerationError(
                f"staged fundamental canonical membership date order invalid: {symbol}"
            )
        listing_dates[symbol] = list_date
        history_end_dates[symbol] = history_end
    return listing_dates, history_end_dates


def _deterministic_replay_projection(
    frame: pd.DataFrame,
    *,
    table_name: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise FundamentalGenerationError(
            f"fundamental replay table is not a DataFrame: {table_name}"
        )
    excluded = {"fetched_at", "forecast_fetched_at"}
    columns = [column for column in frame.columns if column not in excluded]
    # v3 derivation already emits canonical table order.  Keeping that order in
    # the fingerprint detects row-order drift without materializing a second
    # full-table sort/copy during promotion.
    return frame.loc[:, columns]


def _revalidate_checkpoint_accepted_raw_v3(
    checkpoint: _CapturedProviderCheckpoint,
    *,
    as_of: str,
) -> dict[str, pd.DataFrame]:
    """Independently prove every checkpoint slice is already accepted raw."""

    from quant_investor.factors.pit_fundamentals import normalize_ts_code

    try:
        from .fundamental_mart import _strict_pit_cutoff
    except Exception as exc:
        raise FundamentalGenerationError(
            "staged fundamental accepted-raw validator is unavailable"
        ) from exc

    grouped_positions: dict[str, dict[str, Any]] = {}
    for table_name in FUNDAMENTAL_RAW_TABLES:
        frame = checkpoint.tables[table_name]
        if frame.empty:
            grouped_positions[table_name] = {}
            continue
        if "ts_code" not in frame.columns:
            raise FundamentalGenerationError(
                f"staged fundamental accepted raw is missing ts_code: {table_name}"
            )
        normalized_symbols = frame["ts_code"].map(normalize_ts_code)
        if normalized_symbols.eq("").any():
            raise FundamentalGenerationError(
                f"staged fundamental accepted raw contains an invalid symbol: {table_name}"
            )
        grouped_positions[table_name] = {
            str(symbol): positions
            for symbol, positions in frame.groupby(
                normalized_symbols,
                sort=False,
                observed=True,
            ).indices.items()
        }

    expected_symbols_by_table: dict[str, set[str]] = {
        table_name: set() for table_name in FUNDAMENTAL_RAW_TABLES
    }
    dirty_fields = (
        "rows_hard_invalid",
        "rows_filtered_future",
        "rows_filtered_missing_availability",
        "rows_filtered_core_values",
        "rows_deduplicated",
        "rows_discarded_request_malformed",
        *HARD_INVALID_SUBCOUNTER_FIELDS,
    )
    for outcome in checkpoint.outcomes:
        symbol = str(outcome.get("symbol") or "").strip().upper()
        table_name = str(outcome.get("table") or "").strip()
        expected_symbols_by_table[table_name].add(symbol)
        table_frame = checkpoint.tables[table_name]
        positions = grouped_positions[table_name].get(symbol, [])
        raw_slice = table_frame.iloc[positions].reset_index(drop=True)
        try:
            accepted, recomputed, malformed_reason = _strict_pit_cutoff(
                raw_slice,
                table=table_name,
                symbol=symbol,
                as_of=as_of,
            )
        except Exception as exc:
            raise FundamentalGenerationError(
                "staged fundamental accepted-raw PIT replay failed: "
                f"{symbol}/{table_name}: {exc}"
            ) from exc
        if malformed_reason or any(int(recomputed[field]) for field in dirty_fields):
            detail = malformed_reason or next(
                field for field in dirty_fields if int(recomputed[field])
            )
            raise FundamentalGenerationError(
                "staged fundamental accepted raw contains rejected rows: "
                f"{symbol}/{table_name}: {detail}"
            )
        if (
            int(recomputed["rows_received"]) != len(raw_slice)
            or int(recomputed["rows"]) != len(raw_slice)
        ):
            raise FundamentalGenerationError(
                "staged fundamental accepted-raw row accounting replay failed: "
                f"{symbol}/{table_name}"
            )
        try:
            assert_frame_semantics_equal(
                raw_slice,
                accepted,
                label=f"fundamental accepted raw {symbol}/{table_name}",
            )
        except (TypeError, ValueError) as exc:
            raise FundamentalGenerationError(
                "staged fundamental accepted raw changed under PIT replay: "
                f"{symbol}/{table_name}: {exc}"
            ) from exc
        recomputed_status = "success" if not accepted.empty else "empty"
        if str(outcome.get("status") or "") != recomputed_status:
            raise FundamentalGenerationError(
                "staged fundamental accepted-raw outcome status mismatch: "
                f"{symbol}/{table_name}"
            )
        declared_rows = _strict_provider_int(
            outcome.get("rows"),
            label=f"accepted-raw outcome {symbol}/{table_name} rows",
        )
        if declared_rows != len(raw_slice):
            raise FundamentalGenerationError(
                "staged fundamental accepted-raw outcome row count mismatch: "
                f"{symbol}/{table_name}"
            )

    for table_name, positions_by_symbol in grouped_positions.items():
        unexpected = set(positions_by_symbol).difference(
            expected_symbols_by_table[table_name]
        )
        if unexpected:
            raise FundamentalGenerationError(
                "staged fundamental accepted raw contains an unexpected symbol: "
                f"{table_name}"
            )
    return {
        table_name: checkpoint.tables[table_name]
        for table_name in FUNDAMENTAL_RAW_TABLES
    }


def _validate_raw_to_derived_replay_v3(
    *,
    captured: _CapturedFundamentalGeneration,
    manifest: Mapping[str, Any],
    provider: Mapping[str, Any],
    membership_bytes: bytes,
    membership_path: Path,
    outcome_symbols: list[str],
    non_blocking_absent: set[str],
) -> str:
    checkpoint = _capture_provider_checkpoint_v3(provider)
    checkpoint_identity = {
        "generation_id": checkpoint.generation_id,
        "revision": checkpoint.revision,
        "pointer_sha256": checkpoint.pointer_sha256,
        "manifest_sha256": checkpoint.manifest_sha256,
        "binding_sha256": checkpoint.binding_sha256,
        "outcome_accounting_sha256": checkpoint.outcome_accounting_sha256,
        "table_evidence_sha256": checkpoint.table_evidence_sha256,
    }
    primary_envelope = dict(manifest.get("primary_provenance", {}) or {})
    checkpoint_fingerprints = {
        table_name: frame_fingerprint(checkpoint.tables[table_name])
        for table_name in FUNDAMENTAL_RAW_TABLES
    }
    if (
        dict(primary_envelope.get("raw_table_fingerprints", {}) or {})
        != checkpoint_fingerprints
    ):
        raise FundamentalGenerationError(
            "staged fundamental primary provenance/checkpoint raw fingerprints mismatch"
        )
    audit_value = provider.get("endpoint_audit")
    if not isinstance(audit_value, Mapping):
        raise FundamentalGenerationError(
            "staged fundamental endpoint audit evidence is missing"
        )
    audit = dict(audit_value)
    policy_value = audit.get("policy")
    if not isinstance(policy_value, Mapping):
        raise FundamentalGenerationError(
            "staged fundamental endpoint audit policy is missing"
        )
    try:
        from .fundamental_mart import (
            FundamentalEndpointAuditPolicy,
            _attach_financial_coverage,
            _build_endpoint_audit,
            rederive_fundamental_tables_v3,
        )

        declared_policy = FundamentalEndpointAuditPolicy(
            **dict(policy_value)
        )
        policy = FundamentalEndpointAuditPolicy()
        if declared_policy != policy:
            raise FundamentalGenerationError(
                "staged fundamental endpoint audit policy is not "
                "the authoritative promotion policy"
            )
        verified_raw_tables = _revalidate_checkpoint_accepted_raw_v3(
            checkpoint,
            as_of=str(provider.get("strict_pit_as_of") or ""),
        )
        recomputed_outcomes = _attach_financial_coverage(
            outcome_symbols,
            checkpoint.outcomes,
            verified_raw_tables,
            financial_start=str(provider.get("financial_start_date") or ""),
            as_of=str(provider.get("strict_pit_as_of") or ""),
            scope_evidence=dict(provider.get("canonical_scope_evidence", {}) or {}),
            policy=policy,
        )
        if recomputed_outcomes != checkpoint.outcomes:
            raise FundamentalGenerationError(
                "staged fundamental financial coverage evidence mismatch"
            )
        history_end_dates = dict(
            dict(provider.get("canonical_scope_evidence", {}) or {}).get(
                "history_end_dates",
                {},
            )
            or {}
        )
        as_of_text = str(provider.get("strict_pit_as_of") or "")
        daily_tail_exceptions = sorted(
            symbol
            for symbol in non_blocking_absent
            if str(history_end_dates.get(symbol) or "") == as_of_text
        )
        recomputed_audit = _build_endpoint_audit(
            outcome_symbols,
            recomputed_outcomes,
            policy=policy,
            daily_basic_empty_exception_symbols=daily_tail_exceptions,
        )
    except FundamentalGenerationError:
        raise
    except Exception as exc:
        raise FundamentalGenerationError(
            f"staged fundamental endpoint audit replay failed: {exc}"
        ) from exc
    if recomputed_audit != audit:
        raise FundamentalGenerationError(
            "staged fundamental endpoint audit replay mismatch"
        )
    derivation_value = provider.get("derivation")
    if not isinstance(derivation_value, Mapping):
        raise FundamentalGenerationError(
            "staged fundamental derivation evidence is missing"
        )
    derivation = dict(derivation_value)
    membership_sha256 = hashlib.sha256(membership_bytes).hexdigest()
    expected_selection_rule = (
        "latest_active_membership_interval_as_of_else_latest_expired"
    )
    metadata = dict(manifest.get("metadata", {}) or {})
    run_id = str(metadata.get("run_id") or "").strip()
    source = str(metadata.get("provider_status") or "").strip()
    as_of = str(provider.get("strict_pit_as_of") or "").strip()
    derivation_timestamp = str(
        derivation.get("derivation_timestamp") or ""
    ).strip()
    try:
        declared_membership_path = Path(
            str(derivation.get("pit_membership_path") or "")
        ).resolve(strict=True)
    except OSError as exc:
        raise FundamentalGenerationError(
            "staged fundamental derivation membership path is invalid"
        ) from exc
    if (
        derivation.get("contract_version") != FUNDAMENTAL_DERIVATION_CONTRACT
        or declared_membership_path != membership_path
        or str(derivation.get("pit_membership_sha256") or "").lower()
        != membership_sha256
        or str(derivation.get("membership_sha256") or "").lower()
        != membership_sha256
        or str(derivation.get("as_of") or "") != as_of
        or str(derivation.get("selection_rule") or "")
        != expected_selection_rule
        or str(derivation.get("sector_selection_rule") or "")
        != expected_selection_rule
        or str(derivation.get("run_id") or "") != run_id
        or str(derivation.get("source") or "") != source
        or not derivation_timestamp
        or dict(derivation.get("raw_table_fingerprints", {}) or {})
        != checkpoint_fingerprints
        or str(provider.get("raw_to_derived_binding_sha256") or "").lower()
        != canonical_json_sha256(derivation)
    ):
        raise FundamentalGenerationError(
            "staged fundamental derivation contract or binding mismatch"
        )
    try:
        replayed_tables, replayed_evidence = rederive_fundamental_tables_v3(
            verified_raw_tables,
            membership_bytes=membership_bytes,
            membership_sha256=membership_sha256,
            as_of=as_of,
            symbols=outcome_symbols,
            non_blocking_absent_symbols=sorted(non_blocking_absent),
            run_id=run_id,
            source=source,
            derivation_timestamp=derivation_timestamp,
        )
    except Exception as exc:
        raise FundamentalGenerationError(
            f"staged fundamental deterministic replay failed: {exc}"
        ) from exc
    replayed_declaration = {
        "contract_version": FUNDAMENTAL_DERIVATION_CONTRACT,
        "pit_membership_path": str(membership_path),
        "pit_membership_sha256": membership_sha256,
        "sector_selection_rule": expected_selection_rule,
        **dict(replayed_evidence),
    }
    if (
        replayed_declaration != derivation
        or canonical_json_sha256(replayed_declaration)
        != str(provider.get("raw_to_derived_binding_sha256") or "").lower()
    ):
        raise FundamentalGenerationError(
            "staged fundamental replay derivation evidence mismatch"
        )
    if set(replayed_tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError(
            "staged fundamental replay output table set mismatch"
        )
    del verified_raw_tables, checkpoint
    gc.collect()
    replay_projection_fingerprints: dict[str, str] = {}
    for table_name in FUNDAMENTAL_TABLES:
        try:
            replayed_frame = replayed_tables.pop(table_name)
            replayed_projection = _deterministic_replay_projection(
                replayed_frame,
                table_name=table_name,
            )
            del replayed_frame
            replay_fingerprint = frame_fingerprint(replayed_projection)
            expected_fingerprint = captured.replay_projection_fingerprints.get(
                table_name,
            )
            if replay_fingerprint != expected_fingerprint:
                raise ValueError("projection fingerprint changed")
        except Exception as exc:
            raise FundamentalGenerationError(
                f"staged fundamental raw-to-derived replay mismatch: {table_name}: {exc}"
            ) from exc
        replay_projection_fingerprints[table_name] = replay_fingerprint
        del replayed_projection
    return canonical_json_sha256(
        {
            "checkpoint": checkpoint_identity,
            "derivation_binding_sha256": canonical_json_sha256(derivation),
            "projection_fingerprints": replay_projection_fingerprints,
        }
    )


def _validate_primary_rebuild_capture(
    captured: _CapturedFundamentalGeneration,
) -> str:
    manifest = _json_object_from_bytes(
        captured.manifest_bytes,
        label="staged manifest",
    )
    metadata = dict(manifest.get("metadata", {}) or {})
    provider = dict(metadata.get("provider_manifest", {}) or {})
    audit = dict(provider.get("endpoint_audit", {}) or {})
    scope = dict(provider.get("canonical_scope_evidence", {}) or {})
    request_fields = dict(provider.get("request_fields", {}) or {})
    if provider.get("schema_version") != FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA:
        raise FundamentalGenerationError(
            "staged fundamental provider manifest schema mismatch"
        )
    if provider.get("pit_contract_version") != FUNDAMENTAL_FETCH_PIT_CONTRACT:
        raise FundamentalGenerationError(
            "staged fundamental provider PIT contract mismatch"
        )
    if (
        provider.get("authoritative_full_rebuild") is not True
        or audit.get("schema_version") != FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA
        or set(request_fields) != set(FUNDAMENTAL_RAW_TABLES)
        or any(
            "ts_code" not in str(request_fields.get(endpoint) or "").split(",")
            for endpoint in FUNDAMENTAL_RAW_TABLES
        )
        or audit.get("passed") is not True
        or list(audit.get("blockers", []) or [])
    ):
        raise FundamentalGenerationError(
            "staged fundamental authoritative endpoint audit is not passed"
        )
    endpoints = dict(audit.get("endpoints", {}) or {})
    if set(endpoints) != set(FUNDAMENTAL_RAW_TABLES):
        raise FundamentalGenerationError(
            "staged fundamental endpoint audit set mismatch"
        )
    for endpoint in FUNDAMENTAL_RAW_TABLES:
        endpoint_audit = dict(endpoints.get(endpoint, {}) or {})
        if endpoint_audit.get("passed") is not True:
            raise FundamentalGenerationError(
                f"staged fundamental endpoint audit failed: {endpoint}"
            )
        success_ratio = _strict_provider_ratio(
            endpoint_audit.get("success_ratio"),
            label=f"endpoint {endpoint} success_ratio",
        )
        if endpoint != "forecast" and success_ratio < 0.95:
            raise FundamentalGenerationError(
                f"staged fundamental endpoint coverage below 95pct: {endpoint}"
            )
    symbol_count = _strict_provider_int(
        scope.get("symbol_count"), label="canonical scope symbol_count"
    )
    requests_attempted = _strict_provider_int(
        provider.get("requests_attempted"), label="provider requests_attempted"
    )
    requests_failed = _strict_provider_int(
        provider.get("requests_failed"), label="provider requests_failed"
    )
    requests_malformed = _strict_provider_int(
        provider.get("requests_malformed"), label="provider requests_malformed"
    )
    scope_sha = str(scope.get("symbol_set_sha256") or "").strip().lower()
    as_of = str(provider.get("strict_pit_as_of") or "").strip()
    start_date = str(provider.get("daily_start_date") or "").strip()
    outcomes = provider.get("symbol_table_outcomes")
    if (
        symbol_count < 1
        or not _valid_sha256(scope_sha)
        or not re.fullmatch(r"\d{8}", as_of)
        or not re.fullmatch(r"\d{8}", start_date)
        or str(scope.get("canonical_market_trade_date") or "") != as_of
        or requests_attempted != symbol_count * len(FUNDAMENTAL_RAW_TABLES)
        or requests_failed != 0
        or requests_malformed != 0
    ):
        raise FundamentalGenerationError(
            "staged fundamental provider accounting or PIT binding is invalid"
        )
    if not isinstance(outcomes, list):
        raise FundamentalGenerationError(
            "staged fundamental request outcome evidence is missing"
        )
    outcome_keys = {
        (
            str(dict(outcome).get("symbol") or "").strip().upper(),
            str(dict(outcome).get("table") or "").strip(),
        )
        for outcome in outcomes
        if isinstance(outcome, Mapping)
    }
    outcome_symbols = sorted({symbol for symbol, _table in outcome_keys if symbol})
    if (
        len(outcomes) != requests_attempted
        or len(outcome_keys) != requests_attempted
        or len(outcome_symbols) != symbol_count
        or hashlib.sha256("\n".join(outcome_symbols).encode("utf-8")).hexdigest()
        != scope_sha
        or {table for _symbol, table in outcome_keys} != set(FUNDAMENTAL_RAW_TABLES)
    ):
        raise FundamentalGenerationError(
            "staged fundamental request outcomes do not bind the canonical scope"
        )
    evidence_paths: dict[str, Path] = {}
    evidence_payloads: dict[str, bytes] = {}
    for path_field, hash_field in (
        ("canonical_path", "canonical_file_sha256"),
        ("canonical_market_pointer_path", "canonical_market_pointer_sha256"),
        ("canonical_membership_path", "canonical_membership_sha256"),
    ):
        evidence_path = Path(str(scope.get(path_field) or ""))
        expected_evidence_hash = str(scope.get(hash_field) or "").strip().lower()
        if not evidence_path.is_absolute() or not _valid_sha256(expected_evidence_hash):
            raise FundamentalGenerationError(
                "staged fundamental canonical scope evidence is invalid"
            )
        try:
            resolved_evidence_path = evidence_path.resolve(strict=True)
        except OSError as exc:
            raise FundamentalGenerationError(
                "staged fundamental canonical scope evidence is invalid"
            ) from exc
        if resolved_evidence_path != evidence_path:
            raise FundamentalGenerationError(
                "staged fundamental canonical scope evidence path is not canonical"
            )
        evidence_bytes, _signature = _stable_file_bytes(evidence_path)
        if hashlib.sha256(evidence_bytes).hexdigest() != expected_evidence_hash:
            raise FundamentalGenerationError(
                "staged fundamental canonical scope evidence drifted"
            )
        evidence_paths[path_field] = evidence_path
        evidence_payloads[path_field] = evidence_bytes
    canonical_scope_payload = _json_object_from_bytes(
        evidence_payloads["canonical_path"],
        label="canonical scope evidence",
    )
    canonical_market_pointer = _json_object_from_bytes(
        evidence_payloads["canonical_market_pointer_path"],
        label="canonical market pointer evidence",
    )
    declared_scope = sorted(
        {
            str(symbol).strip().upper()
            for symbol in list(canonical_scope_payload.get("full_a", []) or [])
            if str(symbol).strip()
        }
    )
    market_coverage = dict(canonical_market_pointer.get("coverage", {}) or {})
    pointer_membership_path = Path(
        str(market_coverage.get("pit_membership_path") or "")
    ).expanduser()
    if not pointer_membership_path.is_absolute():
        pointer_membership_path = Path.cwd().resolve(strict=True) / pointer_membership_path
    try:
        pointer_membership_path = pointer_membership_path.resolve(strict=True)
    except OSError as exc:
        raise FundamentalGenerationError(
            "staged fundamental canonical market membership path is invalid"
        ) from exc
    pointer_membership_sha256 = str(
        market_coverage.get("pit_membership_sha256") or ""
    ).strip().lower()
    membership_evidence_sha256 = hashlib.sha256(
        evidence_payloads["canonical_membership_path"]
    ).hexdigest()
    pointer_expected_scope_count = _strict_provider_int(
        market_coverage.get("expected_scope_count"),
        label="canonical market expected_scope_count",
    )
    if (
        declared_scope != outcome_symbols
        or pointer_expected_scope_count != symbol_count
        or str(market_coverage.get("expected_scope_sha256") or "").strip().lower()
        != scope_sha
        or str(canonical_market_pointer.get("latest_complete_trade_date") or "")
        != as_of
        or str(canonical_market_pointer.get("snapshot_id") or "")
        != str(scope.get("canonical_market_snapshot_id") or "")
        or pointer_membership_path != evidence_paths["canonical_membership_path"]
        or not _valid_sha256(pointer_membership_sha256)
        or pointer_membership_sha256 != membership_evidence_sha256
        or pointer_membership_sha256
        != str(scope.get("canonical_membership_sha256") or "").strip().lower()
    ):
        raise FundamentalGenerationError(
            "staged fundamental canonical market pointer binding is invalid"
        )
    listing_dates = {
        str(symbol).strip().upper(): str(value).strip()
        for symbol, value in dict(scope.get("listing_dates", {}) or {}).items()
    }
    history_end_dates = {
        str(symbol).strip().upper(): str(value).strip()
        for symbol, value in dict(scope.get("history_end_dates", {}) or {}).items()
    }
    non_blocking_absent = {
        str(symbol).strip().upper()
        for symbol in list(scope.get("non_blocking_absent_symbols", []) or [])
    }
    pointer_non_blocking_absent = {
        str(symbol).strip().upper()
        for symbol in list(
            market_coverage.get("non_blocking_absent_symbols", []) or []
        )
        if str(symbol).strip()
    }
    (
        recomputed_listing_dates,
        recomputed_history_end_dates,
    ) = _membership_eligibility_from_bytes(
        evidence_payloads["canonical_membership_path"],
        symbols=outcome_symbols,
        as_of=as_of,
        non_blocking_absent=non_blocking_absent,
    )
    eligibility_lines = [
        f"{symbol}|{listing_dates.get(symbol, '')}|{history_end_dates.get(symbol, '')}"
        for symbol in outcome_symbols
    ]
    if (
        set(listing_dates) != set(outcome_symbols)
        or set(history_end_dates) != set(outcome_symbols)
        or not non_blocking_absent.issubset(outcome_symbols)
        or pointer_non_blocking_absent != non_blocking_absent
        or listing_dates != recomputed_listing_dates
        or history_end_dates != recomputed_history_end_dates
        or any(
            not re.fullmatch(r"\d{8}", listing_dates[symbol])
            or not re.fullmatch(r"\d{8}", history_end_dates[symbol])
            for symbol in outcome_symbols
        )
        or hashlib.sha256("\n".join(eligibility_lines).encode("utf-8")).hexdigest()
        != str(scope.get("history_eligibility_sha256") or "").strip().lower()
    ):
        raise FundamentalGenerationError(
            "staged fundamental PIT membership eligibility binding is invalid"
        )

    required_columns = {
        "fundamental_period": {"ts_code", "end_date", "availability_date"},
        "fundamental_daily": {
            "ts_code",
            "trade_date",
            "end_date",
            "availability_date",
        },
        "fundamental_quarantine": {"ts_code", "quarantine_reason"},
    }
    frames: dict[str, pd.DataFrame] = {}
    for table_name, columns in required_columns.items():
        try:
            if table_name == "fundamental_quarantine":
                frame = pd.read_parquet(
                    io.BytesIO(captured.table_bytes[table_name])
                )
            else:
                frame = pd.read_parquet(
                    io.BytesIO(captured.table_bytes[table_name]),
                    columns=sorted(columns),
                )
        except Exception as exc:
            raise FundamentalGenerationError(
                f"staged fundamental table schema invalid: {table_name}"
            ) from exc
        missing_columns = columns.difference(frame.columns)
        if missing_columns and table_name == "fundamental_quarantine" and frame.empty:
            frame = frame.reindex(columns=sorted(columns))
        elif missing_columns:
            raise FundamentalGenerationError(
                f"staged fundamental required columns missing: {table_name}"
            )
        else:
            frame = frame.loc[:, sorted(columns)]
        if not frame.empty and frame["ts_code"].astype("string").str.strip().eq("").any():
            raise FundamentalGenerationError(
                f"staged fundamental empty symbol: {table_name}"
            )
        table_symbols = {
            str(symbol).strip().upper()
            for symbol in frame["ts_code"].astype("string")
            if str(symbol).strip()
        }
        if not table_symbols.issubset(outcome_symbols):
            raise FundamentalGenerationError(
                f"staged fundamental table contains out-of-scope symbols: {table_name}"
            )
        frames[table_name] = frame

    period = frames["fundamental_period"]
    daily = frames["fundamental_daily"]
    if period.empty or daily.empty:
        raise FundamentalGenerationError("staged fundamental production tables are empty")
    normalized_period = period.assign(
        _symbol=period["ts_code"].astype("string").str.strip().str.upper()
    )
    normalized_daily = daily.assign(
        _symbol=daily["ts_code"].astype("string").str.strip().str.upper()
    )
    if normalized_period.duplicated(
        ["_symbol", "end_date", "availability_date"]
    ).any():
        raise FundamentalGenerationError("staged fundamental period keys are duplicated")
    if normalized_daily.duplicated(["_symbol", "trade_date"]).any():
        raise FundamentalGenerationError("staged fundamental daily keys are duplicated")

    as_of_ts = pd.Timestamp(pd.to_datetime(as_of, format="%Y%m%d"))
    period_available = _promotion_date_series(period, "availability_date")
    period_end = _promotion_date_series(period, "end_date")
    daily_trade = _promotion_date_series(daily, "trade_date")
    daily_available = _promotion_date_series(daily, "availability_date")
    daily_end = _promotion_date_series(daily, "end_date")
    if period_available.gt(as_of_ts).any() or daily_trade.gt(as_of_ts).any():
        raise FundamentalGenerationError("staged fundamental contains future-known rows")
    if period_end.gt(period_available).any():
        raise FundamentalGenerationError(
            "staged fundamental period end is after availability"
        )
    if daily_available.gt(daily_trade).any():
        raise FundamentalGenerationError("staged fundamental daily PIT ordering is invalid")
    if daily_end.gt(daily_available).any():
        raise FundamentalGenerationError(
            "staged fundamental daily end is after availability"
        )
    if daily_trade.max() != as_of_ts:
        raise FundamentalGenerationError("staged fundamental daily period span is incomplete")
    daily_symbol_count = int(daily["ts_code"].astype("string").nunique())
    if daily_symbol_count / symbol_count < 0.95:
        raise FundamentalGenerationError(
            "staged fundamental output symbol coverage below 95pct"
        )
    daily_by_symbol = daily.assign(
        _trade_date=daily_trade,
        _symbol=normalized_daily["_symbol"],
    ).groupby("_symbol", observed=True)["_trade_date"]
    by_symbol = daily_by_symbol.agg(["min", "max", "nunique"])
    tail_gap_exceptions = {
        symbol
        for symbol in non_blocking_absent
        if history_end_dates.get(symbol) == as_of
    }
    missing_daily_symbols = set(outcome_symbols).difference(by_symbol.index)
    if missing_daily_symbols.difference(tail_gap_exceptions):
        raise FundamentalGenerationError(
            "staged fundamental eligible symbols are missing daily history"
        )
    for symbol, _row in by_symbol.iterrows():
        expected_start = max(start_date, listing_dates[symbol])
        expected_end = history_end_dates[symbol]
        if daily_by_symbol.get_group(symbol).gt(
            pd.Timestamp(pd.to_datetime(expected_end, format="%Y%m%d"))
        ).any():
            raise FundamentalGenerationError(
                f"staged fundamental daily history exceeds eligibility: {symbol}"
            )
        metrics = _daily_history_coverage_metrics(
            daily_by_symbol.get_group(symbol),
            expected_start=expected_start,
            expected_end=expected_end,
            allow_tail_gap=symbol in tail_gap_exceptions,
        )
        if metrics["history_complete"] is not True:
            raise FundamentalGenerationError(
                f"staged fundamental per-symbol daily history incomplete: {symbol}"
            )

    del (
        frames,
        frame,
        period,
        daily,
        normalized_period,
        normalized_daily,
        period_available,
        period_end,
        daily_trade,
        daily_available,
        daily_end,
        daily_by_symbol,
        by_symbol,
    )
    gc.collect()
    replay_validation_sha256 = _validate_raw_to_derived_replay_v3(
        captured=captured,
        manifest=manifest,
        provider=provider,
        membership_bytes=evidence_payloads["canonical_membership_path"],
        membership_path=evidence_paths["canonical_membership_path"],
        outcome_symbols=outcome_symbols,
        non_blocking_absent=non_blocking_absent,
    )
    aggregate = {
        "generation_id": captured.generation_id,
        "manifest_sha256": hashlib.sha256(captured.manifest_bytes).hexdigest(),
        "table_sha256": {
            table_name: hashlib.sha256(captured.table_bytes[table_name]).hexdigest()
            for table_name in FUNDAMENTAL_TABLES
        },
        "raw_to_derived_replay_sha256": replay_validation_sha256,
    }
    return _metadata_sha256(aggregate)


def _write_private_generation_file(path: Path, payload: bytes) -> None:
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise FundamentalGenerationError(
            "fundamental generation staging file creation failed"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FundamentalGenerationError(
                    "fundamental generation staging write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    readback, _signature = _stable_file_bytes(path)
    if (
        len(readback) != len(payload)
        or hashlib.sha256(readback).digest() != hashlib.sha256(payload).digest()
    ):
        raise FundamentalGenerationError("fundamental generation staging readback mismatch")


def _validated_canonical_pointer_bytes(
    root: Path,
    *,
    expected_sha256: str,
) -> bytes:
    pointer_path = root / FUNDAMENTAL_POINTER_FILENAME
    pointer_bytes, _signature = _stable_file_bytes(pointer_path)
    if hashlib.sha256(pointer_bytes).hexdigest() != expected_sha256:
        raise FundamentalGenerationError("fundamental canonical pointer CAS mismatch")
    pointer = _json_object_from_bytes(pointer_bytes, label="canonical pointer")
    if (
        pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or pointer.get("status") != "OK"
    ):
        raise FundamentalGenerationError("fundamental canonical pointer shape invalid")
    generation_id = str(pointer.get("generation_id") or "").strip()
    manifest_path = _resolve_inside(
        root,
        str(pointer.get("manifest_path") or ""),
        label="canonical manifest",
    )
    manifest_bytes, _signature = _stable_file_bytes(manifest_path)
    manifest = _json_object_from_bytes(manifest_bytes, label="canonical manifest")
    if (
        not generation_id
        or manifest.get("schema_version") != "cn-fundamental-generation.v1"
        or manifest.get("status") != "OK"
        or str(manifest.get("generation_id") or "") != generation_id
    ):
        raise FundamentalGenerationError(
            "fundamental canonical pointer/manifest generation mismatch"
        )
    pointer_tables = dict(pointer.get("tables", {}) or {})
    manifest_tables = dict(manifest.get("tables", {}) or {})
    if set(pointer_tables) != set(FUNDAMENTAL_TABLES) or set(manifest_tables) != set(
        FUNDAMENTAL_TABLES
    ):
        raise FundamentalGenerationError("fundamental canonical table set mismatch")
    for table_name in FUNDAMENTAL_TABLES:
        table_path = _resolve_inside(
            root,
            str(pointer_tables.get(table_name) or ""),
            label=f"canonical {table_name}",
        )
        table_bytes, _signature = _stable_file_bytes(table_path)
        _readback_table_contract(
            table_bytes,
            table_name=table_name,
            table_manifest=dict(manifest_tables[table_name] or {}),
        )
    if pointer_sha256(root) != expected_sha256:
        raise FundamentalGenerationError("fundamental canonical pointer changed during validation")
    return pointer_bytes


def _validate_installed_promotion_identity(
    *,
    canonical_base: Path,
    final_root: Path,
    captured: _CapturedFundamentalGeneration,
    expected_pointer_bytes: bytes,
    expected_pointer_sha256: str,
) -> dict[str, Any]:
    """Validate the exact installed copy without replaying the full rebuild."""

    pointer_bytes, _signature = _stable_file_bytes(
        canonical_base / FUNDAMENTAL_POINTER_FILENAME
    )
    if (
        pointer_bytes != expected_pointer_bytes
        or hashlib.sha256(pointer_bytes).hexdigest()
        != expected_pointer_sha256
    ):
        raise FundamentalGenerationError(
            "fundamental promoted pointer identity mismatch"
        )
    installed_manifest_bytes, _signature = _stable_file_bytes(
        final_root / "manifest.json"
    )
    if installed_manifest_bytes != captured.manifest_bytes:
        raise FundamentalGenerationError(
            "fundamental promoted manifest identity mismatch"
        )
    manifest = _json_object_from_bytes(
        installed_manifest_bytes,
        label="promoted manifest",
    )
    manifest_tables = dict(manifest.get("tables", {}) or {})
    if set(manifest_tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError(
            "fundamental promoted table manifest set mismatch"
        )
    for table_name in FUNDAMENTAL_TABLES:
        installed_table_bytes, _signature = _stable_file_bytes(
            final_root / f"{table_name}.parquet"
        )
        if installed_table_bytes != captured.table_bytes[table_name]:
            raise FundamentalGenerationError(
                f"fundamental promoted table identity mismatch: {table_name}"
            )
        readback, _evidence = _readback_table_contract(
            installed_table_bytes,
            table_name=table_name,
            table_manifest=dict(manifest_tables[table_name] or {}),
            require_v2=True,
        )
        del installed_table_bytes, readback

    installed = load_fundamental_pointer(canonical_base)
    if (
        installed is None
        or installed.get("generation_id") != captured.generation_id
        or installed.get("primary_provenance_verified") is not True
        or dict(installed.get("metadata", {}) or {}).get("gate2_passed")
        is not True
        or pointer_sha256(canonical_base) != expected_pointer_sha256
    ):
        raise FundamentalGenerationError(
            "fundamental promoted generation readback failed"
        )
    return installed


def promote_staged_fundamental_generation(
    *,
    staging_root: str | Path,
    canonical_root: str | Path,
    expected_pointer_sha256: str,
) -> dict[str, Any]:
    """CAS-promote one verified primary staging generation into canonical data."""

    expected_hash = str(expected_pointer_sha256 or "").strip().lower()
    if not _valid_sha256(expected_hash):
        raise FundamentalGenerationError("fundamental expected pointer SHA256 is invalid")
    staging_base, captured = _capture_staged_fundamental_generation(staging_root)
    generation_aggregate_sha256 = _validate_primary_rebuild_capture(captured)
    canonical_base = _read_data_root(canonical_root)
    if staging_base == canonical_base:
        raise FundamentalGenerationError("fundamental staging and canonical roots must differ")

    with _fundamental_promotion_lock(canonical_base):
        previous_pointer_bytes = _validated_canonical_pointer_bytes(
            canonical_base,
            expected_sha256=expected_hash,
        )
        generations_root = _write_data_root(canonical_base / FUNDAMENTAL_GENERATIONS_DIRNAME)
        final_root = generations_root / captured.generation_id
        try:
            os.lstat(final_root)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise FundamentalGenerationError(
                "fundamental canonical generation path is unsafe"
            ) from exc
        else:
            raise FundamentalGenerationError("fundamental canonical generation already exists")

        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{captured.generation_id}.promotion.",
                dir=generations_root,
            )
        )
        os.chmod(staging_directory, 0o700)
        generation_installed = False
        try:
            _write_private_generation_file(
                staging_directory / "manifest.json",
                captured.manifest_bytes,
            )
            for table_name in FUNDAMENTAL_TABLES:
                _write_private_generation_file(
                    staging_directory / f"{table_name}.parquet",
                    captured.table_bytes[table_name],
                )
            _fsync_directory(staging_directory)
            if pointer_sha256(canonical_base) != expected_hash:
                raise FundamentalGenerationError("fundamental canonical pointer CAS mismatch")
            os.replace(staging_directory, final_root)
            generation_installed = True
            _fsync_directory(final_root)
            _fsync_directory(generations_root)

            relative_root = final_root.relative_to(canonical_base)
            next_pointer = deepcopy(captured.pointer)
            next_pointer["manifest_path"] = str(relative_root / "manifest.json")
            next_pointer["tables"] = {
                table_name: str(relative_root / f"{table_name}.parquet")
                for table_name in FUNDAMENTAL_TABLES
            }
            next_pointer_bytes = _json_bytes(next_pointer)
            next_pointer_hash = hashlib.sha256(next_pointer_bytes).hexdigest()
            if pointer_sha256(canonical_base) != expected_hash:
                raise FundamentalGenerationError("fundamental canonical pointer CAS mismatch")
            _atomic_write_bytes(
                canonical_base / FUNDAMENTAL_POINTER_FILENAME,
                next_pointer_bytes,
            )

            try:
                installed = _validate_installed_promotion_identity(
                    canonical_base=canonical_base,
                    final_root=final_root,
                    captured=captured,
                    expected_pointer_bytes=next_pointer_bytes,
                    expected_pointer_sha256=next_pointer_hash,
                )
            except Exception as exc:
                try:
                    current_hash = pointer_sha256(canonical_base)
                except Exception:
                    current_hash = ""
                if current_hash != next_pointer_hash:
                    raise FundamentalGenerationError(
                        "fundamental post-switch validation failed after pointer drift; "
                        "rollback not attempted"
                    ) from exc
                try:
                    _atomic_write_bytes(
                        canonical_base / FUNDAMENTAL_POINTER_FILENAME,
                        previous_pointer_bytes,
                    )
                    if pointer_sha256(canonical_base) != expected_hash:
                        raise FundamentalGenerationError(
                            "fundamental pointer rollback readback failed"
                        )
                    _validated_canonical_pointer_bytes(
                        canonical_base,
                        expected_sha256=expected_hash,
                    )
                except Exception as rollback_exc:
                    raise FundamentalGenerationError(
                        "fundamental post-switch validation and CAS rollback failed"
                    ) from rollback_exc
                raise FundamentalGenerationError(
                    "fundamental post-switch validation failed; pointer rolled back"
                ) from exc
            return {
                "status": "OK",
                "promoted": True,
                "generation_id": captured.generation_id,
                "previous_pointer_sha256": expected_hash,
                "pointer_sha256": next_pointer_hash,
                "generation_aggregate_sha256": generation_aggregate_sha256,
                "pointer": installed,
            }
        finally:
            if staging_directory.exists() and not generation_installed:
                shutil.rmtree(staging_directory)


def _publish_fundamental_generation_locked(
    *,
    root: str | Path,
    run_id: str,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    _primary_attestation: _PrimaryGenerationAttestation | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    generation_id = _safe_generation_id(run_id)
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental publish table set mismatch")
    source_priority = str(metadata.get("source_priority") or "").strip()
    if source_priority == "tushare_primary":
        if not _primary_generation_attestation_matches(
            _primary_attestation,
            tables=tables,
            metadata=metadata,
        ):
            raise FundamentalGenerationError(
                "tushare_primary generation requires an internal primary capability"
            )
    elif _primary_attestation is not None:
        raise FundamentalGenerationError(
            "primary generation capability cannot publish non-primary metadata"
        )
    elif _tables_claim_primary(tables):
        raise FundamentalGenerationError(
            "non-primary generation contains tushare_primary row claims"
        )
    base = _write_data_root(root)
    generations_root = _write_data_root(
        base / FUNDAMENTAL_GENERATIONS_DIRNAME
    )
    final_root = generations_root / generation_id
    if final_root.exists():
        pointer = load_fundamental_pointer(base)
        if pointer is not None and pointer.get("generation_id") == generation_id:
            return (
                {
                    table_name: resolve_fundamental_table_path(base, table_name)
                    for table_name in FUNDAMENTAL_TABLES
                },
                pointer,
            )
        raise FundamentalGenerationError(
            f"fundamental generation already exists: {generation_id}"
        )
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations_root)
    )
    table_paths: dict[str, Path] = {}
    table_manifest: dict[str, dict[str, Any]] = {}
    try:
        for table_name in FUNDAMENTAL_TABLES:
            frame = tables[table_name]
            path = staging_root / f"{table_name}.parquet"
            frame.to_parquet(path, index=False)
            _fsync_regular_file(path)
            table_payload, _signature = _stable_file_bytes(path)
            readback, readback_evidence = _readback_table_contract(
                table_payload,
                table_name=table_name,
            )
            _assert_table_roundtrip(
                frame,
                readback,
                table_name=table_name,
            )
            table_manifest[table_name] = readback_evidence
        primary_provenance = (
            _primary_provenance_envelope(
                _primary_attestation,
                metadata=metadata,
                table_manifest=table_manifest,
            )
            if source_priority == "tushare_primary"
            and _primary_attestation is not None
            else None
        )
        manifest = {
            "schema_version": "cn-fundamental-generation.v1",
            "generation_id": generation_id,
            "status": "OK",
            "tables": table_manifest,
            "metadata": dict(metadata),
        }
        if primary_provenance is not None:
            manifest["primary_provenance"] = primary_provenance
        _atomic_write_json(staging_root / "manifest.json", manifest)
        _fsync_directory(staging_root)
        os.replace(staging_root, final_root)
        _fsync_directory(final_root)
        _fsync_directory(generations_root)
        final_manifest_bytes, _signature = _stable_file_bytes(
            final_root / "manifest.json"
        )
        if _json_object_from_bytes(
            final_manifest_bytes,
            label="generation manifest",
        ) != manifest:
            raise FundamentalGenerationError(
                "fundamental generation manifest readback mismatch"
            )
        for table_name in FUNDAMENTAL_TABLES:
            final_table_bytes, _signature = _stable_file_bytes(
                final_root / f"{table_name}.parquet"
            )
            _readback_table_contract(
                final_table_bytes,
                table_name=table_name,
                table_manifest=table_manifest[table_name],
                require_v2=True,
            )
        relative_root = final_root.relative_to(base)
        for table_name in FUNDAMENTAL_TABLES:
            table_paths[table_name] = final_root / f"{table_name}.parquet"
        pointer_metadata = {
            key: metadata.get(key)
            for key in (
                "run_id",
                "provider_status",
                "source_priority",
                "source_provenance",
                "storage_backend",
                "readiness",
                "gate2_passed",
                "merge",
            )
            if key in metadata
        }
        pointer = {
            "schema_version": "cn-fundamental-pointer.v1",
            "status": "OK",
            "generation_id": generation_id,
            "manifest_path": str(relative_root / "manifest.json"),
            "tables": {
                table_name: str(relative_root / f"{table_name}.parquet")
                for table_name in FUNDAMENTAL_TABLES
            },
            "metadata": pointer_metadata,
        }
        if primary_provenance is not None:
            pointer["primary_provenance"] = primary_provenance
        _atomic_write_json(base / FUNDAMENTAL_POINTER_FILENAME, pointer)
        return table_paths, pointer
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        if final_root.exists():
            try:
                current_pointer = load_fundamental_pointer(base)
            except FundamentalGenerationError:
                current_pointer = None
            if (
                current_pointer is not None
                and current_pointer.get("generation_id") == generation_id
            ):
                return table_paths, current_pointer
            shutil.rmtree(final_root)
        raise


def publish_fundamental_generation(
    *,
    root: str | Path,
    run_id: str,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    _primary_attestation: _PrimaryGenerationAttestation | None = None,
    expected_pointer_sha256: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Publish one generation while serializing every pointer writer."""

    base = _write_data_root(root)
    with _fundamental_promotion_lock(base):
        current_pointer_sha256 = pointer_sha256(base)
        if expected_pointer_sha256 is None:
            if current_pointer_sha256:
                raise FundamentalGenerationError(
                    "fundamental predecessor pointer SHA256 is required"
                )
        elif current_pointer_sha256 != str(expected_pointer_sha256).strip().lower():
            raise FundamentalGenerationError(
                "fundamental predecessor pointer CAS mismatch"
            )
        return _publish_fundamental_generation_locked(
            root=base,
            run_id=run_id,
            tables=tables,
            metadata=metadata,
            _primary_attestation=_primary_attestation,
        )


__all__ = [
    "FUNDAMENTAL_GENERATIONS_DIRNAME",
    "FUNDAMENTAL_POINTER_FILENAME",
    "FUNDAMENTAL_PROMOTION_LOCK_FILENAME",
    "FUNDAMENTAL_TABLES",
    "FundamentalGenerationError",
    "fundamental_data_root",
    "legacy_fundamental_table_path",
    "load_fundamental_pointer",
    "load_fundamental_table",
    "pointer_sha256",
    "promote_staged_fundamental_generation",
    "publish_fundamental_generation",
    "resolve_fundamental_table_path",
]
