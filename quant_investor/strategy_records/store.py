"""Fail-closed registry for immutable strategy-record generations.

The only registered state is ``_record_store/current.v1.json``.  It points to
one immutable catalog under ``_record_store/catalogs/<generation>/``.  Legacy
directories are deliberately invisible until a catalog is bootstrapped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from typing import Any, Final
from zoneinfo import ZoneInfo

POINTER_SCHEMA: Final = "myquant.strategy_record_store_current.v1"
CATALOG_SCHEMA_V1: Final = "myquant.strategy_record_catalog.v1"
CATALOG_SCHEMA_V2: Final = "myquant.strategy_record_catalog.v2"
CATALOG_SCHEMA_V3: Final = "myquant.strategy_record_catalog.v3"
# Backward-compatible public name. New archive-aware publications use v2.
CATALOG_SCHEMA: Final = CATALOG_SCHEMA_V1
ARCHIVE_MANIFEST_SCHEMA: Final = "myquant.strategy_record_archive_manifest.v1"
ARCHIVE_RESTORE_RECEIPT_SCHEMA: Final = "myquant.strategy_record_archive_restore_receipt.v1"
ARCHIVE_LOCATOR_SCHEMA: Final = "myquant.strategy_record_archive_locator.v1"
STORE_DIRECTORY: Final = "_record_store"
POINTER_RELATIVE_PATH: Final = "_record_store/current.v1.json"
CATALOG_MAX_BYTES: Final = 32 * 1024 * 1024
ARCHIVE_DOCUMENT_MAX_BYTES: Final = 64 * 1024 * 1024
POINTER_MAX_BYTES: Final = 64 * 1024
NO_ACTION_RECEIPT_MAX_BYTES: Final = 64 * 1024
NEW_RECORD_MAX_FILE_BYTES: Final = 8 * 1024 * 1024
NEW_RECORD_MAX_TOTAL_BYTES: Final = 16 * 1024 * 1024
NEW_RECORD_MAX_FILES: Final = 128
EMPTY_POINTER_SHA256: Final = hashlib.sha256(b"").hexdigest()
PUBLICATION_DELAY_SCHEMA: Final = "myquant.strategy_record_publication_delay.v1"
LATE_OFFICIAL_VALUATION_PUBLICATION: Final = "LATE_OFFICIAL_VALUATION_PUBLICATION"
BATCH_CATCH_UP_OFFICIAL_VALUATION: Final = "BATCH_CATCH_UP_OFFICIAL_VALUATION"
LATE_PUBLICATION_REASON: Final = "SHARED_CHECKOUT_SAFETY_GATE_DELAY"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RECORD_ID = re.compile(r"^(?:[0-9]{8}_[0-9]{4}|[0-9]{8}_[0-9]{6}-b[0-9]{2})$")
_SHANGHAI = ZoneInfo("Asia/Shanghai")


class StrategyRecordStoreError(RuntimeError):
    """The registered strategy-record state is missing, unsafe, or corrupt."""


class StrategyRecordCASMismatch(StrategyRecordStoreError):
    def __init__(self, expected: str, observed: str) -> None:
        super().__init__(
            f"strategy-record pointer CAS mismatch: expected {expected}, observed {observed}"
        )
        self.expected_sha256 = expected
        self.observed_sha256 = observed


class StrategyRecordConflict(StrategyRecordStoreError):
    """An immutable identity already exists with different exact bytes."""


RecordStoreError = StrategyRecordStoreError


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise StrategyRecordStoreError("value is not canonical JSON data") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_file_sha256(
    path: str | os.PathLike[str],
    *,
    expected_bytes: int | None = None,
    label: str = "archive file",
) -> tuple[str, int]:
    """Hash a stable regular single-link file without loading it in memory."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StrategyRecordStoreError(f"{label} is unavailable or unsafe") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StrategyRecordStoreError(f"{label} must be a regular single-link file")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise StrategyRecordStoreError(f"{label} byte length mismatch")
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = lambda item: (  # noqa: E731
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if remaining or identity(before) != identity(after):
            raise StrategyRecordStoreError(f"{label} changed during read")
        return digest.hexdigest(), after.st_size
    finally:
        os.close(descriptor)


def content_sha256(value: Mapping[str, Any]) -> str:
    """Hash a document body, excluding its self-referential content hash."""
    body = dict(value)
    body.pop("content_sha256", None)
    return _sha256(canonical_json_bytes(body))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = content_sha256(result)
    return result


def _validate_content_hash(value: Mapping[str, Any], *, label: str) -> None:
    observed = value.get("content_sha256")
    if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
        raise StrategyRecordStoreError(f"{label} content_sha256 is invalid")
    if observed != content_sha256(value):
        raise StrategyRecordStoreError(f"{label} content_sha256 mismatch")


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StrategyRecordStoreError(f"{label} is not a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise StrategyRecordStoreError(f"{label} is not a canonical relative path")
    return value


def _lstat_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise StrategyRecordStoreError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise StrategyRecordStoreError(f"{label} must be a real directory")
    return metadata


def _read_regular(path: Path, *, max_bytes: int, label: str) -> tuple[bytes, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StrategyRecordStoreError(f"{label} is unavailable or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StrategyRecordStoreError(f"{label} must be a regular single-link file")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise StrategyRecordStoreError(f"{label} exceeds its byte budget")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = lambda item: (  # noqa: E731 - local exact identity tuple
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after) or len(raw) != after.st_size:
            raise StrategyRecordStoreError(f"{label} changed during read")
        return raw, identity(after)
    finally:
        os.close(descriptor)


def _parse_canonical(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyRecordStoreError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise StrategyRecordStoreError(f"{label} bytes are not canonical JSON")
    _validate_content_hash(value, label=label)
    return value


def _validate_pointer(pointer: Mapping[str, Any]) -> None:
    required = {
        "schema_id",
        "generation_id",
        "catalog_path",
        "catalog_sha256",
        "active_record_id",
        "previous_record_id",
        "active_closure",
        "previous_pointer_sha256",
        "published_at",
        "v17_mainline_authority",
        "broker_order_trade_authority",
        "content_sha256",
    }
    if not required.issubset(pointer):
        raise StrategyRecordStoreError("pointer fields are incomplete")
    if pointer.get("schema_id") != POINTER_SCHEMA:
        raise StrategyRecordStoreError("pointer schema is unsupported")
    generation = pointer.get("generation_id")
    if not isinstance(generation, str) or _GENERATION.fullmatch(generation) is None:
        raise StrategyRecordStoreError("pointer generation_id is invalid")
    expected_paths = {
        f"_record_store/catalogs/{generation}/catalog.v1.json",
        f"_record_store/catalogs/{generation}/catalog.v2.json",
        f"_record_store/catalogs/{generation}/catalog.v3.json",
    }
    if pointer.get("catalog_path") not in expected_paths:
        raise StrategyRecordStoreError("pointer catalog_path is not generation-bound")
    digest = pointer.get("catalog_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise StrategyRecordStoreError("pointer catalog_sha256 is invalid")
    previous_sha = pointer.get("previous_pointer_sha256")
    if previous_sha is not None and (
        not isinstance(previous_sha, str) or _SHA256.fullmatch(previous_sha) is None
    ):
        raise StrategyRecordStoreError("pointer previous_pointer_sha256 is invalid")
    if pointer.get("v17_mainline_authority") is not False:
        raise StrategyRecordStoreError("pointer must not claim V17 authority")
    if pointer.get("broker_order_trade_authority") is not False:
        raise StrategyRecordStoreError("pointer must not claim trade authority")
    if not isinstance(pointer.get("active_closure"), dict):
        raise StrategyRecordStoreError("pointer active_closure is invalid")


def _validate_inventory_fields(record: Mapping[str, Any], *, label: str) -> None:
    inventory = record.get("inventory")
    if not isinstance(inventory, list):
        raise StrategyRecordStoreError(f"{label} inventory is invalid")
    digest = record.get("inventory_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise StrategyRecordStoreError(f"{label} inventory_sha256 is invalid")
    if digest != _sha256(canonical_json_bytes(inventory)):
        raise StrategyRecordStoreError(f"{label} inventory_sha256 mismatch")
    files = sum(1 for row in inventory if isinstance(row, dict) and row.get("type") == "file")
    total = sum(
        int(row.get("size", -1))
        for row in inventory
        if isinstance(row, dict) and row.get("type") == "file"
    )
    if record.get("file_count") != files or record.get("total_bytes") != total:
        raise StrategyRecordStoreError(f"{label} inventory summary mismatch")


def _validate_archive_locator_shape(locator: Any) -> None:
    if not isinstance(locator, dict) or locator.get("schema_id") != ARCHIVE_LOCATOR_SCHEMA:
        raise StrategyRecordStoreError("ARCHIVED record locator is invalid")
    required_strings = {
        "archive_id",
        "archive_path",
        "archive_sha256",
        "manifest_path",
        "manifest_sha256",
        "restore_receipt_path",
        "restore_receipt_sha256",
        "member_prefix",
    }
    for key in required_strings:
        value = locator.get(key)
        if not isinstance(value, str) or not value:
            raise StrategyRecordStoreError(f"archive locator {key} is invalid")
    for key in ("archive_path", "manifest_path", "restore_receipt_path", "member_prefix"):
        _canonical_relative_path(locator[key], label=f"archive locator {key}")
    for key in ("archive_sha256", "manifest_sha256", "restore_receipt_sha256"):
        if _SHA256.fullmatch(locator[key]) is None:
            raise StrategyRecordStoreError(f"archive locator {key} is invalid")
    if not isinstance(locator.get("archive_bytes"), int) or locator["archive_bytes"] < 0:
        raise StrategyRecordStoreError("archive locator archive_bytes is invalid")


def _publication_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise StrategyRecordStoreError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StrategyRecordStoreError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise StrategyRecordStoreError(f"{label} timezone is missing")
    return parsed


def _validate_publication_delay(record: Mapping[str, Any]) -> None:
    delay = record.get("publication_delay")
    if delay is None:
        return
    required = {
        "schema_id",
        "publication_class",
        "expected_valuation_date",
        "expected_publication_date",
        "publication_delay_reason",
        "actual_sealed_at",
        "actual_published_at",
        "actual_publication_local_date",
        "candidate_recorded_at",
        "continuity_receipt_id",
        "continuity_receipt_sha256",
        "continuity_receipt_created_at",
        "continuity_checkpoint_digest",
        "source_record",
        "evidence_date",
        "delay_days",
        "historical_holdings_storage_authority",
        "v17_mainline_authority",
        "broker_order_trade_authority",
    }
    if not isinstance(delay, dict) or set(delay) != required:
        raise StrategyRecordStoreError("publication_delay shape is invalid")
    if (
        delay.get("schema_id") != PUBLICATION_DELAY_SCHEMA
        or delay.get("publication_class") != LATE_OFFICIAL_VALUATION_PUBLICATION
        or delay.get("publication_delay_reason") != LATE_PUBLICATION_REASON
        or delay.get("delay_days") != 1
        or delay.get("historical_holdings_storage_authority") is not True
        or delay.get("v17_mainline_authority") is not False
        or delay.get("broker_order_trade_authority") is not False
    ):
        raise StrategyRecordStoreError("publication_delay contract is invalid")
    try:
        valuation_date = date.fromisoformat(str(delay.get("expected_valuation_date")))
        publication_date = date.fromisoformat(str(delay.get("expected_publication_date")))
        actual_local_date = date.fromisoformat(str(delay.get("actual_publication_local_date")))
    except ValueError as exc:
        raise StrategyRecordStoreError("publication_delay date is invalid") from exc
    if (publication_date - valuation_date).days != 1 or actual_local_date != publication_date:
        raise StrategyRecordStoreError("publication_delay day interval is invalid")
    sealed = _publication_timestamp(delay.get("actual_sealed_at"), label="delay sealed_at")
    published = _publication_timestamp(delay.get("actual_published_at"), label="delay published_at")
    recorded = _publication_timestamp(
        delay.get("candidate_recorded_at"), label="delay candidate_recorded_at"
    )
    receipt = _publication_timestamp(
        delay.get("continuity_receipt_created_at"),
        label="delay continuity_receipt_created_at",
    )
    if sealed != published or str(record.get("sealed_at")) != delay.get("actual_sealed_at"):
        raise StrategyRecordStoreError("publication_delay sealed/published timestamp mismatch")
    if not receipt <= recorded <= sealed:
        raise StrategyRecordStoreError("publication_delay timestamp ordering is invalid")
    if sealed.astimezone(_SHANGHAI).date() != publication_date:
        raise StrategyRecordStoreError("publication_delay sealed local date mismatch")
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or _RECORD_ID.fullmatch(record_id) is None:
        raise StrategyRecordStoreError("publication_delay record id is invalid")
    if recorded.astimezone(_SHANGHAI).strftime("%Y%m%d_%H%M") != record_id:
        raise StrategyRecordStoreError("publication_delay record minute mismatch")
    if recorded.astimezone(_SHANGHAI).date() != publication_date:
        raise StrategyRecordStoreError("publication_delay recorded local date mismatch")
    receipt_sha = delay.get("continuity_receipt_sha256")
    if not isinstance(receipt_sha, str) or _SHA256.fullmatch(receipt_sha) is None:
        raise StrategyRecordStoreError("publication_delay receipt SHA is invalid")
    receipt_id = delay.get("continuity_receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id:
        raise StrategyRecordStoreError("publication_delay receipt id is invalid")
    checkpoint_digest = delay.get("continuity_checkpoint_digest")
    if not isinstance(checkpoint_digest, str) or _SHA256.fullmatch(checkpoint_digest) is None:
        raise StrategyRecordStoreError("publication_delay checkpoint digest is invalid")
    if delay.get("evidence_date") != delay.get("expected_valuation_date"):
        raise StrategyRecordStoreError("publication_delay evidence date mismatch")
    if not isinstance(delay.get("source_record"), str) or not delay.get("source_record"):
        raise StrategyRecordStoreError("publication_delay source record is invalid")


def _validate_catalog(catalog: Mapping[str, Any], *, generation_id: str) -> None:
    schema = catalog.get("schema_id")
    if schema not in {CATALOG_SCHEMA_V1, CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}:
        raise StrategyRecordStoreError("catalog schema is unsupported")
    if catalog.get("generation_id") != generation_id:
        raise StrategyRecordStoreError("catalog generation mismatch")
    records = catalog.get("records")
    receipts = catalog.get("receipts")
    if not isinstance(records, list) or not isinstance(receipts, list):
        raise StrategyRecordStoreError("catalog records or receipts are invalid")
    if catalog.get("record_count") != len(records):
        raise StrategyRecordStoreError("catalog record_count mismatch")
    identifiers: set[str] = set()
    paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise StrategyRecordStoreError("catalog record is invalid")
        record_id = record.get("record_id")
        relative_path = _canonical_relative_path(
            record.get("relative_path"), label="record relative_path"
        )
        if not isinstance(record_id, str) or not record_id or record_id in identifiers:
            raise StrategyRecordStoreError("catalog record_id is invalid or duplicated")
        if relative_path in paths or relative_path.startswith("_record_store/"):
            raise StrategyRecordStoreError("catalog record path is invalid or duplicated")
        state = record.get("state", record.get("storage_state"))
        if schema in {CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3} and (
            record.get("state") != record.get("storage_state") or state is None
        ):
            raise StrategyRecordStoreError("catalog v2 state/storage_state mismatch")
        if state not in {
            "ONLINE",
            "ARCHIVED",
            "NONSTANDARD_RESEARCH_OUTPUT",
            "AUXILIARY_ROOT_FILE",
        }:
            raise StrategyRecordStoreError("catalog record state is invalid")
        if schema in {CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}:
            _validate_inventory_fields(record, label=f"record {record_id}")
            if "publication_delay" in record:
                if schema != CATALOG_SCHEMA_V3 or state not in {"ONLINE", "ARCHIVED"}:
                    raise StrategyRecordStoreError(
                        "publication_delay requires a governed catalog v3 record"
                    )
                _validate_publication_delay(record)
            if state == "ARCHIVED":
                _validate_archive_locator_shape(record.get("archive_locator"))
            elif "archive_locator" in record:
                raise StrategyRecordStoreError("non-ARCHIVED record has archive locator")
        identifiers.add(record_id)
        paths.add(relative_path)

    if schema in {CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}:
        active = catalog.get("active_record_id")
        previous = catalog.get("previous_record_id")
        if not isinstance(active, str) or not isinstance(previous, str) or active == previous:
            raise StrategyRecordStoreError("catalog active/previous must be distinct")
        by_id = {record["record_id"]: record for record in records}
        for label, record_id in (("active", active), ("previous", previous)):
            row = by_id.get(record_id)
            if row is None or row.get("state") != "ONLINE":
                raise StrategyRecordStoreError(f"catalog {label} record is not ONLINE")
            if schema == CATALOG_SCHEMA_V3:
                expected_ledger = f"{row.get('relative_path')}/ledger_after_manual_switch.parquet"
                if row.get("ledger_path") != expected_ledger:
                    raise StrategyRecordStoreError(
                        f"catalog v3 {label} record does not bind the Parquet ledger"
                    )
                for key in (
                    "manifest_sha256",
                    "manual_manifest_sha256",
                    "ledger_sha256",
                    "financial_state_sha256",
                ):
                    value = row.get(key)
                    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                        raise StrategyRecordStoreError(
                            f"catalog v3 {label} record {key} is invalid"
                        )
        registry_ref = catalog.get("history_registry_ref")
        if schema == CATALOG_SCHEMA_V3 and registry_ref is not None:
            raise StrategyRecordStoreError("catalog v3 cannot retain a history registry")
        if registry_ref is not None:
            if not isinstance(registry_ref, dict):
                raise StrategyRecordStoreError("history registry ref is invalid")
            _canonical_relative_path(registry_ref.get("path"), label="history registry ref path")
            digest = registry_ref.get("sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise StrategyRecordStoreError("history registry ref SHA-256 is invalid")
    if schema == CATALOG_SCHEMA_V3:
        if catalog.get("performance_contract_ready") is not True:
            raise StrategyRecordStoreError("catalog v3 performance contract is not ready")
        if "dashboard_projection" in catalog or "history_registry" in catalog:
            raise StrategyRecordStoreError("catalog v3 cannot retain legacy performance data")
        lineage = catalog.get("lineage_index")
        lineage_sha = catalog.get("lineage_index_sha256")
        if not isinstance(lineage_sha, str) or _SHA256.fullmatch(lineage_sha) is None:
            raise StrategyRecordStoreError("catalog v3 lineage_index_sha256 is invalid")
        if lineage_sha != _sha256(canonical_json_bytes(lineage)):
            raise StrategyRecordStoreError("catalog v3 lineage_index_sha256 mismatch")
        from .performance import (
            validate_lineage_index,
            validate_performance_history_ref_shape,
        )

        validate_lineage_index(lineage, active_record_id=catalog.get("active_record_id"))
        validate_performance_history_ref_shape(catalog.get("performance_history_ref"))


def project_root_for_record_root(record_root: str | os.PathLike[str]) -> Path:
    """Resolve the project root from the governed CN strategy-root shape."""
    root = Path(record_root).resolve(strict=True)
    parts = root.parts
    marker = ("results", "strategy_records", "CN", "aggressive_tech_manufacturing")
    if len(parts) < len(marker) or tuple(parts[-len(marker) :]) != marker:
        raise StrategyRecordStoreError("record root is not the governed CN strategy root")
    return Path(*parts[: -len(marker)])


def archive_final_root(
    record_root: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> Path:
    root = Path(record_root)
    project = (
        Path(project_root).resolve(strict=True)
        if project_root is not None
        else project_root_for_record_root(root)
    )
    expected_record_root = (
        project / "results/strategy_records/CN/aggressive_tech_manufacturing"
    ).resolve(strict=True)
    if root.resolve(strict=True) != expected_record_root:
        raise StrategyRecordStoreError("project root does not bind the record root")
    return project / (
        "results/strategy_record_archives/CN/aggressive_tech_manufacturing/monthly/v1"
    )


def _archive_project_path(
    record_root: Path,
    relative: str,
    *,
    label: str,
    project_root: Path | None,
) -> Path:
    relative = _canonical_relative_path(relative, label=label)
    project = project_root or project_root_for_record_root(record_root)
    allowed = archive_final_root(record_root, project_root=project)
    candidate = project / relative
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
        allowed_resolved = allowed.resolve(strict=True)
    except OSError as exc:
        raise StrategyRecordStoreError(f"{label} parent is unavailable") from exc
    if resolved_parent != allowed_resolved and allowed_resolved not in resolved_parent.parents:
        raise StrategyRecordStoreError(f"{label} is outside the final archive root")
    return candidate


def _read_archive_document(path: Path, *, label: str, expected_sha: str) -> dict[str, Any]:
    raw, _ = _read_regular(path, max_bytes=ARCHIVE_DOCUMENT_MAX_BYTES, label=label)
    if _sha256(raw) != expected_sha:
        raise StrategyRecordStoreError(f"{label} byte SHA-256 mismatch")
    return _parse_canonical(raw, label=label)


def load_archive_binding(
    record_root: str | os.PathLike[str],
    record: Mapping[str, Any],
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate and return one ARCHIVED row's immutable storage closure."""
    if record.get("state") != "ARCHIVED" or record.get("storage_state") != "ARCHIVED":
        raise StrategyRecordStoreError("archive binding requires an ARCHIVED record")
    locator = record.get("archive_locator")
    _validate_archive_locator_shape(locator)
    assert isinstance(locator, dict)
    root = Path(record_root)
    project = Path(project_root).resolve(strict=True) if project_root is not None else None
    archive_path = _archive_project_path(
        root, locator["archive_path"], label="archive path", project_root=project
    )
    manifest_path = _archive_project_path(
        root, locator["manifest_path"], label="archive manifest path", project_root=project
    )
    receipt_path = _archive_project_path(
        root,
        locator["restore_receipt_path"],
        label="archive restore receipt path",
        project_root=project,
    )
    archive_sha, archive_bytes = regular_file_sha256(
        archive_path,
        expected_bytes=locator["archive_bytes"],
        label="archive payload",
    )
    if archive_sha != locator["archive_sha256"]:
        raise StrategyRecordStoreError("archive payload SHA-256 mismatch")
    manifest = _read_archive_document(
        manifest_path,
        label="archive manifest",
        expected_sha=locator["manifest_sha256"],
    )
    receipt = _read_archive_document(
        receipt_path,
        label="archive restore receipt",
        expected_sha=locator["restore_receipt_sha256"],
    )
    if manifest.get("schema_id") != ARCHIVE_MANIFEST_SCHEMA:
        raise StrategyRecordStoreError("archive manifest schema is unsupported")
    if manifest.get("archive_id") != locator["archive_id"]:
        raise StrategyRecordStoreError("archive manifest identity mismatch")
    for key, expected in (
        ("archive_path", locator["archive_path"]),
        ("archive_sha256", archive_sha),
        ("archive_bytes", archive_bytes),
    ):
        if manifest.get(key) != expected:
            raise StrategyRecordStoreError(f"archive manifest {key} mismatch")
    manifest_records = manifest.get("records")
    if not isinstance(manifest_records, list) or manifest.get("record_count") != len(
        manifest_records
    ):
        raise StrategyRecordStoreError("archive manifest records are invalid")
    matches = [
        row
        for row in manifest_records
        if isinstance(row, dict) and row.get("record_id") == record.get("record_id")
    ]
    if len(matches) != 1:
        raise StrategyRecordStoreError("archive manifest record closure is missing")
    manifest_record = matches[0]
    for key in (
        "relative_path",
        "inventory",
        "inventory_sha256",
        "file_count",
        "total_bytes",
    ):
        if manifest_record.get(key) != record.get(key):
            raise StrategyRecordStoreError(f"archive manifest record {key} mismatch")
    if record.get("logical_source_refs") != manifest_record.get("logical_source_refs"):
        raise StrategyRecordStoreError("archive manifest logical source refs mismatch")
    if manifest_record.get("member_prefix") != locator["member_prefix"]:
        raise StrategyRecordStoreError("archive member prefix mismatch")
    if receipt.get("schema_id") != ARCHIVE_RESTORE_RECEIPT_SCHEMA:
        raise StrategyRecordStoreError("archive restore receipt schema is unsupported")
    if receipt.get("archive_id") != locator["archive_id"]:
        raise StrategyRecordStoreError("archive restore receipt identity mismatch")
    if (
        receipt.get("manifest_path") != locator["manifest_path"]
        or receipt.get("manifest_sha256") != locator["manifest_sha256"]
    ):
        raise StrategyRecordStoreError("archive restore receipt manifest binding mismatch")
    if (
        receipt.get("archive_path") != locator["archive_path"]
        or receipt.get("archive_sha256") != archive_sha
    ):
        raise StrategyRecordStoreError("archive restore receipt payload binding mismatch")
    record_ids = [row.get("record_id") for row in manifest_records]
    if (
        receipt.get("record_ids") != record_ids
        or receipt.get("record_count") != len(record_ids)
        or receipt.get("all_inventory_matched") is not True
        or receipt.get("restored_file_count") != manifest.get("file_count")
        or receipt.get("restored_logical_bytes") != manifest.get("logical_bytes")
    ):
        raise StrategyRecordStoreError("archive restore receipt closure mismatch")
    return {
        "locator": dict(locator),
        "archive_path": archive_path,
        "manifest_path": manifest_path,
        "receipt_path": receipt_path,
        "manifest": manifest,
        "receipt": receipt,
    }


def _validate_pointer_catalog_closure(
    pointer: Mapping[str, Any], catalog: Mapping[str, Any]
) -> None:
    by_id = {record["record_id"]: record for record in catalog["records"]}
    active = pointer.get("active_record_id")
    previous = pointer.get("previous_record_id")
    if catalog.get("schema_id") in {CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}:
        if (
            catalog.get("active_record_id") != active
            or catalog.get("previous_record_id") != previous
        ):
            raise StrategyRecordStoreError("pointer/catalog active identifiers mismatch")
        if active == previous:
            raise StrategyRecordStoreError("pointer active/previous identifiers collide")
    for label, record_id in (("active", active), ("previous", previous)):
        if record_id is None and catalog.get("schema_id") == CATALOG_SCHEMA_V1:
            continue
        record = by_id.get(record_id)
        if record is None or record.get("state", record.get("storage_state")) != "ONLINE":
            raise StrategyRecordStoreError(f"pointer {label} record is not ONLINE")
    expected_closure = _active_closure(catalog["records"], active)
    if pointer.get("active_closure") != expected_closure:
        raise StrategyRecordStoreError("pointer active_closure mismatch")
    active_record = by_id.get(active)
    active_delay = (
        active_record.get("publication_delay") if isinstance(active_record, Mapping) else None
    )
    if isinstance(active_delay, Mapping) and pointer.get("published_at") != active_delay.get(
        "actual_published_at"
    ) and not _has_current_active_no_action_receipt(
        catalog,
        active_record_id=active,
        published_at=pointer.get("published_at"),
    ):
        raise StrategyRecordStoreError("pointer late publication timestamp mismatch")


def _has_current_active_no_action_receipt(
    catalog: Mapping[str, Any],
    *,
    active_record_id: Any,
    published_at: Any,
) -> bool:
    if not isinstance(active_record_id, str) or not isinstance(published_at, str):
        return False
    try:
        active_checkpoint = _active_closure(catalog["records"], active_record_id)
    except (KeyError, StrategyRecordStoreError):
        return False
    matches = [
        receipt
        for receipt in catalog.get("receipts", [])
        if isinstance(receipt, Mapping)
        and receipt.get("schema_id") == "myquant.strategy_record_no_action_receipt.v1"
        and receipt.get("created_at") == published_at
        and receipt.get("status") == "NO_ACTION"
        and receipt.get("active_record_id") == active_record_id
        and receipt.get("active_checkpoint") == active_checkpoint
        and receipt.get("payload_copied") is False
        and receipt.get("v17_mainline_authority") is False
        and receipt.get("broker_order_trade_authority") is False
        and receipt.get("content_sha256") == content_sha256(receipt)
    ]
    return len(matches) == 1


def _late_document_declaration(document: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    nested = document.get("publication_delay")
    snapshot = document.get("data_snapshot")
    sources = [value for value in (nested, document, snapshot) if isinstance(value, Mapping)]

    def selected(*names: str) -> Any:
        for source in sources:
            for name in names:
                if source.get(name) is not None:
                    return source.get(name)
        return None

    result = {
        "publication_class": selected("publication_class"),
        "expected_valuation_date": selected(
            "expected_valuation_date", "valuation_date", "valuation_trade_date"
        ),
        "expected_publication_date": selected("expected_publication_date", "publication_date"),
        "publication_delay_reason": selected("publication_delay_reason", "reason"),
    }
    for key in ("expected_valuation_date", "expected_publication_date"):
        text = str(result[key] or "")
        if len(text) == 8 and text.isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            result[key] = date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise StrategyRecordStoreError(f"{label} {key} is invalid") from exc
    for source in sources:
        if (
            source.get("publication_class") is not None
            and source.get("publication_class") != result["publication_class"]
        ):
            raise StrategyRecordStoreError(f"{label} publication class declarations conflict")
        for canonical, names in (
            (
                "expected_valuation_date",
                ("expected_valuation_date", "valuation_date", "valuation_trade_date"),
            ),
            (
                "expected_publication_date",
                ("expected_publication_date", "publication_date"),
            ),
        ):
            for name in names:
                if source.get(name) is None:
                    continue
                text = str(source.get(name))
                if len(text) == 8 and text.isdigit():
                    text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
                try:
                    normalized_date = date.fromisoformat(text).isoformat()
                except ValueError as exc:
                    raise StrategyRecordStoreError(f"{label} {name} is invalid") from exc
                if normalized_date != result[canonical]:
                    raise StrategyRecordStoreError(f"{label} {canonical} declarations conflict")
        for name in ("publication_delay_reason", "reason"):
            if (
                source.get(name) is not None
                and source.get(name) != result["publication_delay_reason"]
            ):
                raise StrategyRecordStoreError(
                    f"{label} publication delay reason declarations conflict"
                )
    return result


def _record_json(root: Path, *, path_value: Any, expected_sha: Any, label: str) -> dict[str, Any]:
    relative = _canonical_relative_path(path_value, label=f"{label} path")
    if not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None:
        raise StrategyRecordStoreError(f"{label} SHA is invalid")
    raw, _ = _read_regular(
        root / relative,
        max_bytes=NEW_RECORD_MAX_FILE_BYTES,
        label=label,
    )
    if _sha256(raw) != expected_sha:
        raise StrategyRecordStoreError(f"{label} SHA mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyRecordStoreError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise StrategyRecordStoreError(f"{label} is not an object")
    return value


def _validate_late_external_binding(
    *,
    root: Path,
    catalog: Mapping[str, Any],
    record: Mapping[str, Any],
    lineage: Mapping[str, Any],
    performance_row: Mapping[str, Any],
) -> None:
    delay = record.get("publication_delay")
    if not isinstance(delay, dict):
        raise StrategyRecordStoreError("late publication metadata is absent")
    if record.get("state", record.get("storage_state")) != "ONLINE":
        raise StrategyRecordStoreError(
            "late publication external binding requires ONLINE source artifacts"
        )
    if catalog.get("active_record_id") == record.get("record_id") and catalog.get(
        "published_at"
    ) != delay.get("actual_published_at") and not _has_current_active_no_action_receipt(
        catalog,
        active_record_id=record.get("record_id"),
        published_at=catalog.get("published_at"),
    ):
        raise StrategyRecordStoreError("late publication catalog timestamp mismatch")
    manifest = _record_json(
        root,
        path_value=record.get("manifest_path"),
        expected_sha=record.get("manifest_sha256"),
        label="late publication manifest",
    )
    manual = _record_json(
        root,
        path_value=record.get("manual_manifest_path"),
        expected_sha=record.get("manual_manifest_sha256"),
        label="late publication manual manifest",
    )
    expected_declaration = {
        key: delay[key]
        for key in (
            "publication_class",
            "expected_valuation_date",
            "expected_publication_date",
            "publication_delay_reason",
        )
    }
    if (
        _late_document_declaration(manifest, label="late manifest") != expected_declaration
        or _late_document_declaration(manual, label="late manual") != expected_declaration
    ):
        raise StrategyRecordStoreError("late publication declaration mismatch")
    manifest_delay = manifest.get("publication_delay")
    manual_delay = manual.get("publication_delay")
    if not isinstance(manifest_delay, dict) or manifest_delay != manual_delay:
        raise StrategyRecordStoreError("late publication document delay mismatch")
    if (
        manifest_delay.get("schema_id") != PUBLICATION_DELAY_SCHEMA
        or manifest_delay.get("publication_class") != LATE_OFFICIAL_VALUATION_PUBLICATION
        or manifest_delay.get("delay_days") != 1
    ):
        raise StrategyRecordStoreError("late publication document delay contract mismatch")
    for key in (
        "source_record",
        "evidence_date",
        "continuity_receipt_id",
        "continuity_receipt_sha256",
        "continuity_receipt_created_at",
        "continuity_checkpoint_digest",
        "recorded_at_iso",
        "historical_holdings_storage_authority",
        "v17_mainline_authority",
        "broker_order_trade_authority",
    ):
        metadata_key = "candidate_recorded_at" if key == "recorded_at_iso" else key
        if manifest_delay.get(key) != delay.get(metadata_key):
            raise StrategyRecordStoreError(f"late publication document {key} binding mismatch")
    if (
        manifest.get("recorded_at_iso") != delay.get("candidate_recorded_at")
        or manual.get("recorded_at_iso") != delay.get("candidate_recorded_at")
        or manifest.get("source_record") != delay.get("source_record")
        or manual.get("source_record") != delay.get("source_record")
        or manifest.get("v17_mainline_authority") is not False
        or manifest.get("broker_order_trade_authority") is not False
        or manual.get("v17_mainline_authority") is not False
        or manual.get("broker_order_trade_authority") is not False
    ):
        raise StrategyRecordStoreError("late publication document authority mismatch")
    if (
        lineage.get("publication_class") != LATE_OFFICIAL_VALUATION_PUBLICATION
        or lineage.get("valuation_date") != delay.get("expected_valuation_date")
        or lineage.get("record_id") != record.get("record_id")
    ):
        raise StrategyRecordStoreError("late publication lineage mismatch")
    if (
        performance_row.get("record_id") != record.get("record_id")
        or performance_row.get("valuation_date") != delay.get("expected_valuation_date")
        or performance_row.get("evidence_kind") != "REGISTERED_OFFICIAL_FINANCIAL_STATE"
        or performance_row.get("manual_manifest_sha256") != record.get("manual_manifest_sha256")
        or performance_row.get("ledger_parquet_sha256") != record.get("ledger_sha256")
        or performance_row.get("financial_state_sha256") != record.get("financial_state_sha256")
    ):
        raise StrategyRecordStoreError("late publication performance mismatch")
    matching_receipts = [
        row
        for row in catalog.get("receipts", [])
        if isinstance(row, Mapping) and row.get("receipt_id") == delay.get("continuity_receipt_id")
    ]
    if len(matching_receipts) != 1:
        raise StrategyRecordStoreError("late publication receipt is not unique")
    receipt = matching_receipts[0]
    source_closure = _active_closure(catalog["records"], str(delay.get("source_record") or ""))
    if (
        receipt.get("schema_id") != "myquant.strategy_record_no_action_receipt.v1"
        or receipt.get("content_sha256") != delay.get("continuity_receipt_sha256")
        or receipt.get("content_sha256") != content_sha256(receipt)
        or receipt.get("created_at") != delay.get("continuity_receipt_created_at")
        or receipt.get("v17_mainline_authority") is not False
        or receipt.get("broker_order_trade_authority") is not False
        or receipt.get("status") != "NO_ACTION"
        or receipt.get("payload_copied") is not False
        or receipt.get("active_record_id") != delay.get("source_record")
        or receipt.get("active_checkpoint") != source_closure
        or content_sha256(receipt.get("active_checkpoint") or {})
        != delay.get("continuity_checkpoint_digest")
    ):
        raise StrategyRecordStoreError("late publication receipt binding mismatch")


def _validate_external_catalog_bindings(root: Path, catalog: Mapping[str, Any]) -> None:
    schema = catalog.get("schema_id")
    if schema not in {CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}:
        return
    archive_cache: dict[str, dict[str, Any]] = {}
    for record in catalog["records"]:
        if record.get("state") == "ARCHIVED":
            locator = record["archive_locator"]
            archive_id = locator["archive_id"]
            binding = archive_cache.get(archive_id)
            if binding is None:
                binding = load_archive_binding(root, record)
                archive_cache[archive_id] = binding
                continue
            for key in (
                "archive_path",
                "archive_sha256",
                "archive_bytes",
                "manifest_path",
                "manifest_sha256",
                "restore_receipt_path",
                "restore_receipt_sha256",
            ):
                if binding["locator"][key] != locator[key]:
                    raise StrategyRecordStoreError(
                        "archive identity has conflicting locator fields"
                    )
            matches = [
                item
                for item in binding["manifest"]["records"]
                if item.get("record_id") == record.get("record_id")
            ]
            if len(matches) != 1:
                raise StrategyRecordStoreError("archive manifest record closure is missing")
            manifest_record = matches[0]
            for key in (
                "relative_path",
                "inventory",
                "inventory_sha256",
                "file_count",
                "total_bytes",
                "logical_source_refs",
            ):
                if manifest_record.get(key) != record.get(key):
                    raise StrategyRecordStoreError(f"archive manifest record {key} mismatch")
            if manifest_record.get("member_prefix") != locator["member_prefix"]:
                raise StrategyRecordStoreError("archive member prefix mismatch")
    if schema == CATALOG_SCHEMA_V3:
        from .performance import load_performance_history

        performance = load_performance_history(root, catalog["performance_history_ref"])
        rows = performance["rows"]
        last = rows[-1]
        active_id = catalog.get("active_record_id")
        if last.get("record_id") != active_id:
            raise StrategyRecordStoreError(
                "active performance record does not match catalog active record"
            )
        active_record = next(
            (row for row in catalog["records"] if row.get("record_id") == active_id),
            None,
        )
        active_lineage = next(
            (row for row in catalog["lineage_index"] if row.get("record_id") == active_id),
            None,
        )
        if active_record is None or active_lineage is None:
            raise StrategyRecordStoreError("active performance lineage is absent")
        selected_by_id = {
            row.get("record_id"): row
            for row in catalog["records"]
            if row.get("record_id")
            in {catalog.get("active_record_id"), catalog.get("previous_record_id")}
        }
        lineage_by_id = {
            row.get("record_id"): row
            for row in catalog["lineage_index"]
            if row.get("record_id") in selected_by_id
        }
        for selected_id, selected_record in selected_by_id.items():
            selected_lineage = lineage_by_id.get(selected_id)
            if selected_lineage is None:
                raise StrategyRecordStoreError("selected lineage closure is absent")
            expected_refs = {
                "manifest_ref": {
                    "path": selected_record.get("manifest_path"),
                    "sha256": selected_record.get("manifest_sha256"),
                },
                "manual_manifest_ref": {
                    "path": selected_record.get("manual_manifest_path"),
                    "sha256": selected_record.get("manual_manifest_sha256"),
                },
                "effective_ledger_ref": {
                    "path": selected_record.get("ledger_path"),
                    "sha256": selected_record.get("ledger_sha256"),
                },
            }
            if any(
                selected_lineage.get(key) != expected for key, expected in expected_refs.items()
            ) or (
                selected_lineage.get("financial_state_sha256")
                != selected_record.get("financial_state_sha256")
                or selected_lineage.get("ledger_parquet_sha256")
                != selected_record.get("ledger_sha256")
            ):
                raise StrategyRecordStoreError("selected lineage exact closure mismatch")
        if last.get("valuation_date") != active_lineage.get("valuation_date"):
            raise StrategyRecordStoreError("active performance valuation date mismatch")
        for series_key, record_key in (
            ("manual_manifest_sha256", "manual_manifest_sha256"),
            ("ledger_parquet_sha256", "ledger_sha256"),
            ("financial_state_sha256", "financial_state_sha256"),
        ):
            expected = active_record.get(record_key)
            if not isinstance(expected, str) or last.get(series_key) != expected:
                raise StrategyRecordStoreError(
                    f"active performance {series_key} does not reconcile"
                )
        all_lineage = {
            row.get("record_id"): row
            for row in catalog["lineage_index"]
            if isinstance(row, Mapping)
        }
        performance_by_id = {row.get("record_id"): row for row in rows if isinstance(row, Mapping)}
        for record in catalog["records"]:
            if not isinstance(record, Mapping) or "publication_delay" not in record:
                continue
            record_id = record.get("record_id")
            lineage = all_lineage.get(record_id)
            performance_row = performance_by_id.get(record_id)
            if not isinstance(lineage, Mapping) or not isinstance(performance_row, Mapping):
                raise StrategyRecordStoreError(
                    "late publication lineage/performance binding is absent"
                )
            _validate_late_external_binding(
                root=root,
                catalog=catalog,
                record=record,
                lineage=lineage,
                performance_row=performance_row,
            )
        return
    registry_ref = catalog.get("history_registry_ref")
    if registry_ref is None:
        return
    project = project_root_for_record_root(root)
    registry_path = project / registry_ref["path"]
    try:
        resolved = registry_path.resolve(strict=True)
    except OSError as exc:
        raise StrategyRecordStoreError("history registry is unavailable") from exc
    if resolved != project and project not in resolved.parents:
        raise StrategyRecordStoreError("history registry escapes project root")
    digest, _ = regular_file_sha256(registry_path, label="history registry")
    if digest != registry_ref["sha256"]:
        raise StrategyRecordStoreError("history registry SHA-256 mismatch")
    if "history_registry" in catalog:
        try:
            observed_registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StrategyRecordStoreError("history registry is invalid JSON") from exc
        if observed_registry != catalog["history_registry"]:
            raise StrategyRecordStoreError("history registry body mismatch")


def load_registered_catalog(
    record_root: str | os.PathLike[str],
    *,
    max_pointer_bytes: int = POINTER_MAX_BYTES,
    max_catalog_bytes: int = CATALOG_MAX_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load the one registered catalog with a stable pointer double-read.

    ``None`` means the record root has no ``_record_store`` registration at
    all.  Once that directory exists, missing or partial state is corruption.
    """
    root = Path(record_root)
    _lstat_directory(root, label="record root")
    store_root = root / STORE_DIRECTORY
    if not store_root.exists():
        return None
    _lstat_directory(store_root, label="record store")
    pointer_path = root / POINTER_RELATIVE_PATH
    if not pointer_path.exists():
        raise StrategyRecordStoreError("registered store has no current pointer")
    first_raw, first_identity = _read_regular(
        pointer_path, max_bytes=max_pointer_bytes, label="strategy-record pointer"
    )
    pointer = _parse_canonical(first_raw, label="strategy-record pointer")
    _validate_pointer(pointer)
    catalog_relative = _canonical_relative_path(
        pointer["catalog_path"], label="pointer catalog_path"
    )
    catalog_raw, _ = _read_regular(
        root / catalog_relative,
        max_bytes=max_catalog_bytes,
        label="strategy-record catalog",
    )
    if _sha256(catalog_raw) != pointer["catalog_sha256"]:
        raise StrategyRecordStoreError("catalog byte SHA-256 mismatch")
    catalog = _parse_canonical(catalog_raw, label="strategy-record catalog")
    _validate_catalog(catalog, generation_id=pointer["generation_id"])
    _validate_pointer_catalog_closure(pointer, catalog)
    _validate_external_catalog_bindings(root, catalog)
    second_raw, second_identity = _read_regular(
        pointer_path, max_bytes=max_pointer_bytes, label="strategy-record pointer"
    )
    if first_raw != second_raw or first_identity != second_identity:
        raise StrategyRecordStoreError("strategy-record pointer was unstable")
    return pointer, catalog


def _loaded_or_supplied(
    record_root: str | os.PathLike[str],
    pointer: Mapping[str, Any] | None,
    catalog: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if pointer is None and catalog is None:
        loaded = load_registered_catalog(record_root)
        if loaded is None:
            raise StrategyRecordStoreError("strategy-record store is unregistered")
        return loaded
    if pointer is None or catalog is None:
        raise StrategyRecordStoreError("pointer and catalog must be supplied together")
    pointer_copy, catalog_copy = dict(pointer), dict(catalog)
    _validate_content_hash(pointer_copy, label="strategy-record pointer")
    _validate_pointer(pointer_copy)
    _validate_content_hash(catalog_copy, label="strategy-record catalog")
    _validate_catalog(catalog_copy, generation_id=pointer_copy["generation_id"])
    _validate_pointer_catalog_closure(pointer_copy, catalog_copy)
    return pointer_copy, catalog_copy


def _safe_record_directory(root: Path, relative_path: str) -> Path:
    relative = _canonical_relative_path(relative_path, label="record relative_path")
    path = root / relative
    _lstat_directory(path, label="registered record directory")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise StrategyRecordStoreError("registered record path is unavailable") from exc
    if resolved_path.parent != resolved_root and resolved_root not in resolved_path.parents:
        raise StrategyRecordStoreError("registered record escapes record root")
    return path


def catalog_online_record_dirs(
    record_root: str | os.PathLike[str],
    pointer: Mapping[str, Any] | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    root = Path(record_root)
    _, loaded_catalog = _loaded_or_supplied(record_root, pointer, catalog)
    return tuple(
        _safe_record_directory(root, record["relative_path"])
        for record in loaded_catalog["records"]
        if record.get("state", record.get("storage_state")) == "ONLINE"
    )


def resolve_active_record_dirs(
    record_root: str | os.PathLike[str],
    pointer: Mapping[str, Any] | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    root = Path(record_root)
    loaded_pointer, loaded_catalog = _loaded_or_supplied(record_root, pointer, catalog)
    by_id = {
        record["record_id"]: record
        for record in loaded_catalog["records"]
        if record.get("state", record.get("storage_state")) == "ONLINE"
    }
    ordered: list[Path] = []
    seen: set[str] = set()
    for key in ("active_record_id", "previous_record_id"):
        record_id = loaded_pointer.get(key)
        if record_id is None:
            continue
        if not isinstance(record_id, str) or record_id not in by_id:
            raise StrategyRecordStoreError(f"pointer {key} is not an ONLINE catalog record")
        if record_id not in seen:
            ordered.append(_safe_record_directory(root, by_id[record_id]["relative_path"]))
            seen.add(record_id)
    if not ordered:
        raise StrategyRecordStoreError("pointer has no resolvable active record")
    return tuple(ordered)


def catalog_history_entries(
    record_root: str | os.PathLike[str],
    pointer: Mapping[str, Any] | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    root = Path(record_root)
    _, loaded_catalog = _loaded_or_supplied(record_root, pointer, catalog)
    result: list[dict[str, Any]] = []
    for record in loaded_catalog["records"]:
        if record.get("history_eligible") is False:
            continue
        if record.get("state", record.get("storage_state")) not in {
            "ONLINE",
            "ARCHIVED",
        }:
            continue
        row = dict(record)
        state = row.get("state", row.get("storage_state"))
        row["state"] = state
        row["storage_state"] = state
        if state == "ONLINE":
            row["record_dir"] = str(_safe_record_directory(root, row["relative_path"]))
        else:
            row["record_dir"] = None
        result.append(row)
    return tuple(result)


# Writer functions are defined below this read boundary so callers can depend on
# the fail-closed API without importing a CLI or performing any publication.


def _generation_id(value: str | None) -> str:
    if value is None:
        from datetime import datetime, timezone

        value = datetime.now(timezone.utc).strftime("g%Y%m%dT%H%M%S") + "-" + secrets.token_hex(4)
    if _GENERATION.fullmatch(value) is None:
        raise StrategyRecordStoreError("generation_id is invalid")
    return value


def _published_at(value: str | None) -> str:
    if value is not None:
        if not isinstance(value, str) or not value:
            raise StrategyRecordStoreError("published_at is invalid")
        return value
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_exact_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.lstat(path.parent).st_dev != os.lstat(path.parent.parent).st_dev:
        raise StrategyRecordStoreError("immutable staging must use the same filesystem")
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StrategyRecordStoreError("short immutable write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            existing, _ = _read_regular(
                path,
                max_bytes=CATALOG_MAX_BYTES,
                label="immutable artifact",
            )
            if existing != raw:
                raise StrategyRecordConflict("immutable artifact identity collision") from None
    finally:
        temporary.unlink(missing_ok=True)
    stored, _ = _read_regular(path, max_bytes=CATALOG_MAX_BYTES, label="immutable artifact")
    if stored != raw:
        raise StrategyRecordStoreError("immutable artifact readback mismatch")


def _cas_pointer(root: Path, raw: bytes, *, expected_pointer_sha256: str | None) -> str:
    store_root = root / STORE_DIRECTORY
    store_root.mkdir(parents=True, exist_ok=True)
    pointer_path = root / POINTER_RELATIVE_PATH
    lock_path = store_root / ".current.v1.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if os.fstat(lock_fd).st_nlink != 1:
            raise StrategyRecordStoreError("pointer lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if pointer_path.exists():
            current, _ = _read_regular(
                pointer_path, max_bytes=POINTER_MAX_BYTES, label="strategy-record pointer"
            )
            observed = _sha256(current)
        else:
            observed = EMPTY_POINTER_SHA256
        expected = expected_pointer_sha256 or EMPTY_POINTER_SHA256
        if observed != expected:
            raise StrategyRecordCASMismatch(expected, observed)
        temporary = store_root / f".current.v1.cas-{os.getpid()}-{secrets.token_hex(6)}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.write(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, pointer_path)
        directory_fd = os.open(store_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        readback, _ = _read_regular(
            pointer_path, max_bytes=POINTER_MAX_BYTES, label="strategy-record pointer"
        )
        if readback != raw:
            raise StrategyRecordStoreError("pointer readback mismatch")
        return _sha256(readback)
    finally:
        os.close(lock_fd)


def _active_closure(
    records: Sequence[Mapping[str, Any]],
    active_record_id: str | None,
) -> dict[str, Any]:
    if active_record_id is None:
        return {}
    for record in records:
        if record.get("record_id") == active_record_id:
            return {
                key: record[key]
                for key in (
                    "record_id",
                    "relative_path",
                    "inventory_sha256",
                    "total_bytes",
                    "file_count",
                    "manifest_path",
                    "manifest_sha256",
                    "manual_manifest_path",
                    "manual_manifest_sha256",
                    "ledger_path",
                    "ledger_sha256",
                    "pnl_path",
                    "pnl_sha256",
                    "financial_state_sha256",
                )
                if key in record
            }
    raise StrategyRecordStoreError("active_record_id is absent from catalog")


def reselect_catalog(
    record_root: str | os.PathLike[str],
    *,
    expected_current_pointer_sha256: str,
    target_generation_id: str,
    target_catalog_path: str,
    target_catalog_sha256: str,
    published_at: str,
) -> dict[str, Any]:
    """CAS-select one already-published immutable catalog generation.

    This is the rollback primitive for an explicitly owner-approved cutover
    reversal.  It never creates or rewrites a catalog and it never infers a
    target from directory ordering, mtime, or a filename such as ``latest``.
    """

    root = Path(record_root)
    _lstat_directory(root, label="record root")
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError("catalog reselection requires a registered store")
    current_pointer, _ = loaded
    observed_pointer_sha = _sha256(canonical_json_bytes(current_pointer))
    if observed_pointer_sha != expected_current_pointer_sha256:
        raise StrategyRecordCASMismatch(expected_current_pointer_sha256, observed_pointer_sha)
    generation = _generation_id(target_generation_id)
    catalog_relative = _canonical_relative_path(target_catalog_path, label="target catalog path")
    expected_paths = {
        f"_record_store/catalogs/{generation}/catalog.v2.json",
        f"_record_store/catalogs/{generation}/catalog.v3.json",
    }
    if catalog_relative not in expected_paths:
        raise StrategyRecordStoreError(
            "target catalog path is not an exact v2/v3 generation binding"
        )
    if (
        not isinstance(target_catalog_sha256, str)
        or _SHA256.fullmatch(target_catalog_sha256) is None
    ):
        raise StrategyRecordStoreError("target catalog SHA-256 is invalid")
    catalog_raw, _ = _read_regular(
        root / catalog_relative,
        max_bytes=CATALOG_MAX_BYTES,
        label="target strategy-record catalog",
    )
    observed_catalog_sha = _sha256(catalog_raw)
    if observed_catalog_sha != target_catalog_sha256:
        raise StrategyRecordStoreError("target catalog byte SHA-256 mismatch")
    catalog = _parse_canonical(catalog_raw, label="target strategy-record catalog")
    _validate_catalog(catalog, generation_id=generation)
    if catalog.get("schema_id") not in {CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}:
        raise StrategyRecordStoreError("target catalog schema is not reselectable")
    _validate_external_catalog_bindings(root, catalog)
    active_record_id = catalog.get("active_record_id")
    previous_record_id = catalog.get("previous_record_id")
    pointer = _seal(
        {
            "schema_id": POINTER_SCHEMA,
            "generation_id": generation,
            "catalog_path": catalog_relative,
            "catalog_sha256": target_catalog_sha256,
            "active_record_id": active_record_id,
            "previous_record_id": previous_record_id,
            "active_closure": _active_closure(catalog["records"], active_record_id),
            "previous_pointer_sha256": expected_current_pointer_sha256,
            "published_at": _published_at(published_at),
            "v17_mainline_authority": False,
            "broker_order_trade_authority": False,
        }
    )
    _validate_pointer(pointer)
    _validate_pointer_catalog_closure(pointer, catalog)
    pointer_raw = canonical_json_bytes(pointer)
    if len(pointer_raw) > POINTER_MAX_BYTES:
        raise StrategyRecordStoreError("pointer exceeds byte budget")
    pointer_sha = _cas_pointer(
        root,
        pointer_raw,
        expected_pointer_sha256=expected_current_pointer_sha256,
    )
    readback = load_registered_catalog(root)
    if readback is None or readback != (pointer, catalog):
        raise StrategyRecordStoreError("reselected catalog readback mismatch")
    return {
        "pointer": pointer,
        "catalog": catalog,
        "pointer_sha256": pointer_sha,
        "catalog_reselected": True,
        "catalog_created": False,
    }


def publish_catalog(
    record_root: str | os.PathLike[str],
    *,
    expected_pointer_sha256: str | None,
    records: Sequence[Mapping[str, Any]] | None = None,
    dashboard_projection: Any = None,
    receipts: Sequence[Mapping[str, Any]] = (),
    active_record_id: str | None = None,
    previous_record_id: str | None = None,
    generation_id: str | None = None,
    published_at: str | None = None,
    catalog_schema: str | None = None,
    history_registry: Mapping[str, Any] | None = None,
    history_registry_ref: Mapping[str, Any] | None = None,
    inherit_history_registry: bool = True,
    lineage_index: Sequence[Mapping[str, Any]] | None = None,
    performance_history_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(record_root)
    _lstat_directory(root, label="record root")
    loaded = load_registered_catalog(root) if (root / POINTER_RELATIVE_PATH).exists() else None
    if loaded is None:
        old_pointer: dict[str, Any] | None = None
        old_catalog: dict[str, Any] = {"records": [], "receipts": []}
    else:
        old_pointer, old_catalog = loaded
    if records is None:
        record_rows = [dict(value) for value in old_catalog["records"]]
    else:
        record_rows = [dict(value) for value in records]
    if active_record_id is None:
        active_record_id = old_pointer.get("active_record_id") if old_pointer else None
    if previous_record_id is None:
        previous_record_id = old_pointer.get("previous_record_id") if old_pointer else None
    generation = _generation_id(generation_id)
    timestamp = _published_at(published_at)
    old_schema = old_catalog.get("schema_id")
    schema = catalog_schema or (
        old_schema
        if old_schema in {CATALOG_SCHEMA_V1, CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}
        else (
            CATALOG_SCHEMA_V2
            if any(record.get("archive_locator") is not None for record in record_rows)
            else CATALOG_SCHEMA_V1
        )
    )
    if schema not in {CATALOG_SCHEMA_V1, CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}:
        raise StrategyRecordStoreError("catalog publication schema is unsupported")
    catalog_body: dict[str, Any] = {
        "schema_id": schema,
        "generation_id": generation,
        "published_at": timestamp,
        "records": record_rows,
        "record_count": len(record_rows),
        "receipts": [dict(value) for value in old_catalog.get("receipts", [])]
        + [dict(value) for value in receipts],
    }
    if schema in {CATALOG_SCHEMA_V2, CATALOG_SCHEMA_V3}:
        catalog_body["active_record_id"] = active_record_id
        catalog_body["previous_record_id"] = previous_record_id
    if schema == CATALOG_SCHEMA_V3:
        selected_lineage = (
            [dict(row) for row in lineage_index]
            if lineage_index is not None
            else [dict(row) for row in old_catalog.get("lineage_index", [])]
        )
        selected_performance_ref = (
            dict(performance_history_ref)
            if performance_history_ref is not None
            else dict(old_catalog.get("performance_history_ref", {}))
        )
        catalog_body["lineage_index"] = selected_lineage
        catalog_body["lineage_index_sha256"] = _sha256(canonical_json_bytes(selected_lineage))
        catalog_body["performance_history_ref"] = selected_performance_ref
        catalog_body["performance_contract_ready"] = True
    else:
        if dashboard_projection is not None:
            catalog_body["dashboard_projection"] = dashboard_projection
        elif "dashboard_projection" in old_catalog:
            catalog_body["dashboard_projection"] = old_catalog["dashboard_projection"]
        if history_registry is not None:
            catalog_body["history_registry"] = dict(history_registry)
        elif inherit_history_registry and "history_registry" in old_catalog:
            catalog_body["history_registry"] = old_catalog["history_registry"]
        if history_registry_ref is not None:
            catalog_body["history_registry_ref"] = dict(history_registry_ref)
        elif inherit_history_registry and "history_registry_ref" in old_catalog:
            catalog_body["history_registry_ref"] = old_catalog["history_registry_ref"]
    catalog = _seal(catalog_body)
    _validate_catalog(catalog, generation_id=generation)
    _validate_external_catalog_bindings(root, catalog)
    catalog_raw = canonical_json_bytes(catalog)
    if len(catalog_raw) > CATALOG_MAX_BYTES:
        raise StrategyRecordStoreError("catalog exceeds byte budget")
    suffix = {
        CATALOG_SCHEMA_V1: "v1",
        CATALOG_SCHEMA_V2: "v2",
        CATALOG_SCHEMA_V3: "v3",
    }[schema]
    catalog_relative = f"_record_store/catalogs/{generation}/catalog.{suffix}.json"
    _write_exact_once(root / catalog_relative, catalog_raw)
    pointer = _seal(
        {
            "schema_id": POINTER_SCHEMA,
            "generation_id": generation,
            "catalog_path": catalog_relative,
            "catalog_sha256": _sha256(catalog_raw),
            "active_record_id": active_record_id,
            "previous_record_id": previous_record_id,
            "active_closure": _active_closure(record_rows, active_record_id),
            "previous_pointer_sha256": None if old_pointer is None else expected_pointer_sha256,
            "published_at": timestamp,
            "v17_mainline_authority": False,
            "broker_order_trade_authority": False,
        }
    )
    pointer_raw = canonical_json_bytes(pointer)
    if len(pointer_raw) > POINTER_MAX_BYTES:
        raise StrategyRecordStoreError("pointer exceeds byte budget")
    pointer_sha = _cas_pointer(root, pointer_raw, expected_pointer_sha256=expected_pointer_sha256)
    readback = load_registered_catalog(root)
    if readback is None or readback != (pointer, catalog):
        raise StrategyRecordStoreError("published catalog readback mismatch")
    return {"pointer": pointer, "catalog": catalog, "pointer_sha256": pointer_sha}


def bootstrap_catalog(
    record_root: str | os.PathLike[str],
    *,
    records: Sequence[Mapping[str, Any]],
    dashboard_projection: Any = None,
    active_record_id: str | None = None,
    previous_record_id: str | None = None,
    generation_id: str | None = None,
    published_at: str | None = None,
    catalog_schema: str | None = None,
    history_registry: Mapping[str, Any] | None = None,
    history_registry_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    store_root = Path(record_root) / STORE_DIRECTORY
    if store_root.exists():
        children = {child.name for child in store_root.iterdir()}
        if children - {".operation.v2.lock"}:
            raise StrategyRecordConflict("strategy-record store is already initialized")
    return publish_catalog(
        record_root,
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        records=records,
        dashboard_projection=dashboard_projection,
        active_record_id=active_record_id,
        previous_record_id=previous_record_id,
        generation_id=generation_id,
        published_at=published_at,
        catalog_schema=catalog_schema,
        history_registry=history_registry,
        history_registry_ref=history_registry_ref,
    )


__all__ = [name for name in globals() if not name.startswith("_")]
