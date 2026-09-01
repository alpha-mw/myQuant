"""Build and verify the read-only CN aggressive Dashboard v2 envelope.

Version 2 does not replace the canonical v1 history.  It binds that exact v1
document, proves that the registered Strategy Record Store still designates the
same holdings closure, and adds one view-only mark at the latest complete local
CN close.  The mark never writes a ledger, a performance generation, an order,
or any other governed state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import stat
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.strategy_records.store import (
    StrategyRecordStoreError,
    content_sha256 as store_content_sha256,
    load_registered_catalog,
)

SCHEMA_VERSION = "cn_aggressive_dashboard.v2"
V1_SCHEMA_VERSION = "cn_aggressive_dashboard.v1"
MARKET = "CN"
STRATEGY_LABEL = "aggressive_tech_manufacturing"
RETURN_METHOD = "initial_capital_return_excluding_external_flows"
INITIAL_CAPITAL = 1_000_000.0
VIEW_ONLY_AUTHORITY = "VIEW_ONLY_NO_STORE_OR_PERFORMANCE_AUTHORITY"
MARK_SOURCE_KIND = "STRICT_CN_EOD_CLOSE"
DAILY_SCOPE = "DAILY_SYNC_LATEST_VERIFIED_LOCAL_CLOSE"
NO_ACTION_RECEIPT_SCHEMA = "myquant.strategy_record_no_action_receipt.v1"
OFFICIAL_CLOSE_BATCH_RECEIPT_SCHEMA = "myquant.strategy_daily_close_receipt.v1"
LATE_PUBLICATION_CLASS = "LATE_OFFICIAL_VALUATION_PUBLICATION"
LATE_PUBLICATION_SCHEMA = "publication_delay.v1"
LATE_CATALOG_PUBLICATION_SCHEMA = "myquant.strategy_record_publication_delay.v1"
LATE_PUBLICATION_REASON = "SHARED_CHECKOUT_SAFETY_GATE_DELAY"
LATE_VALUATION_DATE = date(2026, 8, 21)
LATE_PUBLICATION_DATE = date(2026, 8, 22)
LATE_SOURCE_RECORD = "20260820_1321"
LATE_FRESHNESS_REASON = "LATE_OFFICIAL_FINANCIAL_PUBLICATION_FOR_LATEST_LOCAL_CLOSE"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_ATTEMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_RECORD_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{4}$")
_SHANGHAI_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+08:00$")
_SHANGHAI = ZoneInfo("Asia/Shanghai")


class DashboardV2Error(RuntimeError):
    """Raised when the v2 read-only closure cannot be proved exactly."""


@dataclass(frozen=True)
class _Artifact:
    path: Path
    relative_path: str
    raw: bytes
    sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    """Return the Dashboard canonical JSON representation (without newline)."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DashboardV2Error("value_is_not_canonical_json_data") from exc


def content_sha256(value: Mapping[str, Any]) -> str:
    """Hash a Dashboard document after removing its self-referential hash."""

    body = dict(value)
    body.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_float(value: Any) -> float:
    return float(value) if _finite_number(value) else math.nan


def _close(left: Any, right: Any, *, tolerance: float = 0.01) -> bool:
    return (
        _finite_number(left)
        and _finite_number(right)
        and math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=tolerance)
    )


def _project_relative(path: Path, project_root: Path) -> str:
    root = project_root.resolve()
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise DashboardV2Error(f"artifact_outside_project:{path}") from exc
    if resolved != path.absolute():
        raise DashboardV2Error(f"artifact_path_is_not_lexically_exact:{path}")
    return relative.as_posix()


def _stable_artifact(path: Path, project_root: Path) -> _Artifact:
    """Bind a regular, non-symlink project file with a stable double read."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise DashboardV2Error(f"artifact_unavailable:{path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise DashboardV2Error(f"artifact_not_regular_non_symlink:{path}")
    try:
        first = path.read_bytes()
        second = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise DashboardV2Error(f"artifact_unavailable:{path}") from exc
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if first != second or identity(before) != identity(after):
        raise DashboardV2Error(f"artifact_unstable_double_read:{path}")
    return _Artifact(
        path=path,
        relative_path=_project_relative(path, project_root),
        raw=first,
        sha256=hashlib.sha256(first).hexdigest(),
    )


def _artifact_from_bytes(path: Path, raw: bytes, project_root: Path) -> _Artifact:
    """Bind staged bytes to their intended project-relative final path."""

    root = project_root.resolve()
    absolute = path if path.is_absolute() else root / path
    absolute = absolute.absolute()
    try:
        relative = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise DashboardV2Error(f"artifact_outside_project:{absolute}") from exc
    if ".." in Path(relative).parts or not isinstance(raw, bytes):
        raise DashboardV2Error("staged_artifact_invalid")
    return _Artifact(
        path=absolute,
        relative_path=relative,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _source_path(value: Any, project_root: Path, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise DashboardV2Error(f"{label}_invalid")
    path = Path(text)
    return path if path.is_absolute() else project_root / path


def _json_object(artifact: _Artifact, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(artifact.raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DashboardV2Error(f"{label}_invalid_json") from exc
    if not isinstance(value, dict):
        raise DashboardV2Error(f"{label}_not_object")
    return value


def _date_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise DashboardV2Error(f"{label}_invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise DashboardV2Error(f"{label}_invalid") from exc
    return parsed.isoformat()


def _compact_date(value: str) -> str:
    return value.replace("-", "")


def _local_date_from_timestamp(value: Any, *, label: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise DashboardV2Error(f"{label}_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DashboardV2Error(f"{label}_invalid") from exc
    if parsed.tzinfo is None:
        raise DashboardV2Error(f"{label}_timezone_missing")
    return parsed.astimezone(_SHANGHAI).date()


def _catalog_record_closure(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one ONLINE catalog row into the pointer-closure shape."""

    keys = (
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
    closure = {key: record.get(key) for key in keys}
    if any(value is None for value in closure.values()):
        raise DashboardV2Error("financial_publication_source_closure_missing")
    return closure


def _validate_continuity_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_active_record_id: str,
    expected_checkpoint: Mapping[str, Any],
    generation_local_date: date,
    expected_created_local_date: date | None = None,
) -> None:
    """Validate one no-action receipt independently of Dashboard state."""

    expected_date = expected_created_local_date or generation_local_date
    if (
        receipt.get("schema_id") != NO_ACTION_RECEIPT_SCHEMA
        or receipt.get("status") != "NO_ACTION"
        or receipt.get("payload_copied") is not False
        or receipt.get("v17_mainline_authority") is not False
        or receipt.get("broker_order_trade_authority") is not False
        or receipt.get("active_record_id") != expected_active_record_id
        or receipt.get("active_checkpoint") != dict(expected_checkpoint)
        or receipt.get("content_sha256") != store_content_sha256(receipt)
        or _local_date_from_timestamp(receipt.get("created_at"), label="daily_receipt_created_at")
        != expected_date
    ):
        raise DashboardV2Error("daily_continuity_receipt_invalid")


def _reject_unrelated_same_day_receipts(
    receipts: Any, *, expected_receipt_id: str, generation_local_date: date
) -> None:
    if not isinstance(receipts, list):
        raise DashboardV2Error("daily_continuity_receipts_invalid")
    for candidate in receipts:
        if not isinstance(candidate, dict) or candidate.get("receipt_id") == expected_receipt_id:
            continue
        if candidate.get("schema_id") == OFFICIAL_CLOSE_BATCH_RECEIPT_SCHEMA:
            if (
                candidate.get("status") != "OFFICIAL_CLOSE_PREPARED"
                or candidate.get("payload_copied") is not False
                or candidate.get("actual_holdings_mutation_authority") is not False
                or candidate.get("cash_mutation_authority") is not False
                or candidate.get("broker_order_trade_authority") is not False
                or candidate.get("content_sha256") != store_content_sha256(candidate)
            ):
                raise DashboardV2Error("daily_continuity_receipt_unrelated")
            continue
        try:
            candidate_date = _local_date_from_timestamp(
                candidate.get("created_at"), label="daily_receipt_created_at"
            )
        except DashboardV2Error as exc:
            # A malformed receipt is not safely classifiable as historical
            # noise, so it cannot be ignored by a current Dashboard build.
            raise DashboardV2Error("daily_continuity_receipt_unrelated") from exc
        if candidate_date == generation_local_date:
            raise DashboardV2Error("daily_continuity_receipt_unrelated")


def _publication_delay_semantics(value: Any, *, label: str) -> dict[str, Any]:
    """Normalize one producer or catalog publication-delay projection.

    The producer keeps the self-contained ``publication_delay.v1`` object in
    both manifests.  Store-v3 keeps a manager-validated projection under the
    catalog row, with actual seal/publication timestamps.  They are different
    physical schemas but must describe the same immutable event.
    """

    if not isinstance(value, dict):
        raise DashboardV2Error(f"{label}_invalid")
    if value.get("schema_id") == LATE_PUBLICATION_SCHEMA or (
        value.get("schema_id") == LATE_CATALOG_PUBLICATION_SCHEMA
        and "recorded_at_iso" in value
        and "actual_sealed_at" not in value
    ):
        required = {
            "schema_id",
            "publication_class",
            "expected_valuation_date",
            "evidence_date",
            "expected_publication_date",
            "source_record",
            "continuity_receipt_id",
            "continuity_receipt_sha256",
            "continuity_receipt_created_at",
            "continuity_checkpoint_digest",
            "recorded_at_iso",
            "publication_delay_reason",
            "historical_holdings_storage_authority",
            "v17_mainline_authority",
            "broker_order_trade_authority",
            "delay_days",
        }
        if set(value) != required:
            raise DashboardV2Error(f"{label}_shape_invalid")
        result = {
            "publication_class": value.get("publication_class"),
            "valuation_date": value.get("expected_valuation_date"),
            "evidence_date": value.get("evidence_date"),
            "publication_date": value.get("expected_publication_date"),
            "source_record": value.get("source_record"),
            "continuity_receipt_id": value.get("continuity_receipt_id"),
            "continuity_receipt_sha256": value.get("continuity_receipt_sha256"),
            "continuity_receipt_created_at": value.get("continuity_receipt_created_at"),
            "continuity_checkpoint_digest": value.get("continuity_checkpoint_digest"),
            "recorded_at_iso": value.get("recorded_at_iso"),
            "reason": value.get("publication_delay_reason"),
            "delay_days": value.get("delay_days"),
            "actual_sealed_at": None,
            "actual_published_at": None,
            "candidate_recorded_at": value.get("recorded_at_iso"),
        }
        if (
            result["publication_class"] != LATE_PUBLICATION_CLASS
            or result["reason"] != LATE_PUBLICATION_REASON
            or result["source_record"] != LATE_SOURCE_RECORD
            or result["valuation_date"] != LATE_VALUATION_DATE.isoformat()
            or result["evidence_date"] != LATE_VALUATION_DATE.isoformat()
            or result["publication_date"] != LATE_PUBLICATION_DATE.isoformat()
            or result["delay_days"] != 1
            or not _valid_sha(result["continuity_receipt_sha256"])
            or not _valid_sha(result["continuity_checkpoint_digest"])
            or not isinstance(result["continuity_receipt_id"], str)
            or not result["continuity_receipt_id"]
            or not isinstance(result["recorded_at_iso"], str)
            or value.get("historical_holdings_storage_authority") is not True
            or value.get("v17_mainline_authority") is not False
            or value.get("broker_order_trade_authority") is not False
        ):
            raise DashboardV2Error(f"{label}_contract_invalid")
        return result

    if value.get("schema_id") == LATE_CATALOG_PUBLICATION_SCHEMA:
        required = {
            "schema_id",
            "publication_class",
            "expected_valuation_date",
            "evidence_date",
            "expected_publication_date",
            "publication_delay_reason",
            "source_record",
            "actual_sealed_at",
            "actual_published_at",
            "actual_publication_local_date",
            "candidate_recorded_at",
            "continuity_receipt_id",
            "continuity_receipt_sha256",
            "continuity_receipt_created_at",
            "continuity_checkpoint_digest",
            "delay_days",
            "historical_holdings_storage_authority",
            "v17_mainline_authority",
            "broker_order_trade_authority",
        }
        if set(value) != required:
            raise DashboardV2Error(f"{label}_shape_invalid")
        result = {
            "publication_class": value.get("publication_class"),
            "valuation_date": value.get("expected_valuation_date"),
            "evidence_date": value.get("evidence_date"),
            "publication_date": value.get("expected_publication_date"),
            "source_record": value.get("source_record"),
            "continuity_receipt_id": value.get("continuity_receipt_id"),
            "continuity_receipt_sha256": value.get("continuity_receipt_sha256"),
            "continuity_receipt_created_at": value.get("continuity_receipt_created_at"),
            "continuity_checkpoint_digest": value.get("continuity_checkpoint_digest"),
            "recorded_at_iso": value.get("candidate_recorded_at"),
            "reason": value.get("publication_delay_reason"),
            "delay_days": value.get("delay_days"),
            "actual_sealed_at": value.get("actual_sealed_at"),
            "actual_published_at": value.get("actual_published_at"),
            "actual_publication_local_date": value.get("actual_publication_local_date"),
            "candidate_recorded_at": value.get("candidate_recorded_at"),
        }
        if (
            result["publication_class"] != LATE_PUBLICATION_CLASS
            or result["reason"] != LATE_PUBLICATION_REASON
            or result["valuation_date"] != LATE_VALUATION_DATE.isoformat()
            or result["evidence_date"] != LATE_VALUATION_DATE.isoformat()
            or result["publication_date"] != LATE_PUBLICATION_DATE.isoformat()
            or result["source_record"] != LATE_SOURCE_RECORD
            or result["delay_days"] != 1
            or result["actual_publication_local_date"] != LATE_PUBLICATION_DATE.isoformat()
            or not _valid_sha(result["continuity_receipt_sha256"])
            or not _valid_sha(result["continuity_checkpoint_digest"])
            or not isinstance(result["continuity_receipt_id"], str)
            or not result["continuity_receipt_id"]
            or not isinstance(result["candidate_recorded_at"], str)
            or value.get("historical_holdings_storage_authority") is not True
            or value.get("v17_mainline_authority") is not False
            or value.get("broker_order_trade_authority") is not False
        ):
            raise DashboardV2Error(f"{label}_contract_invalid")
        return result

    raise DashboardV2Error(f"{label}_schema_invalid")


def _validate_late_publication_metadata(
    *,
    project_root: Path,
    record_root: Path,
    generation_local_date: date,
    record: Mapping[str, Any],
    lineage: Mapping[str, Any],
    artifacts: Mapping[str, _Artifact],
    catalog: Mapping[str, Any],
    pointer: Mapping[str, Any],
) -> tuple[dict[str, Any], bool, list[dict[str, str]]]:
    """Validate the late 2026-08-21 close and return its private projection."""

    record_id = str(record.get("record_id") or "")
    if not _RECORD_ID_RE.fullmatch(record_id):
        raise DashboardV2Error("late_publication_record_id_invalid")
    if lineage.get("publication_class") != LATE_PUBLICATION_CLASS:
        raise DashboardV2Error("late_publication_lineage_class_invalid")
    if lineage.get("valuation_date") != LATE_VALUATION_DATE.isoformat():
        raise DashboardV2Error("late_publication_lineage_valuation_date_invalid")
    if lineage.get("source_record_id") != LATE_SOURCE_RECORD:
        raise DashboardV2Error("late_publication_lineage_source_invalid")

    manifest = _json_object(artifacts["manifest"], label="active_manifest")
    manual = _json_object(artifacts["manual_manifest"], label="active_manual_manifest")
    manifest_delay = manifest.get("publication_delay")
    manual_delay = manual.get("publication_delay")
    if manifest.get("publication_class") != LATE_PUBLICATION_CLASS:
        raise DashboardV2Error("late_publication_manifest_class_invalid")
    if manual.get("publication_class") != LATE_PUBLICATION_CLASS:
        raise DashboardV2Error("late_publication_manual_class_invalid")
    if not isinstance(manifest_delay, dict) or manifest_delay != manual_delay:
        raise DashboardV2Error("late_publication_manifest_manual_delay_mismatch")
    manifest_semantics = _publication_delay_semantics(
        manifest_delay, label="late_publication_manifest_delay"
    )
    catalog_delay = record.get("publication_delay")
    catalog_semantics = _publication_delay_semantics(
        catalog_delay, label="late_publication_catalog_delay"
    )
    for key in (
        "publication_class",
        "valuation_date",
        "evidence_date",
        "publication_date",
        "continuity_receipt_id",
        "continuity_receipt_sha256",
        "continuity_receipt_created_at",
        "reason",
        "delay_days",
    ):
        if catalog_semantics.get(key) != manifest_semantics.get(key):
            raise DashboardV2Error("late_publication_catalog_manifest_mismatch")
    if catalog_semantics.get("source_record") not in (None, LATE_SOURCE_RECORD):
        raise DashboardV2Error("late_publication_catalog_source_invalid")

    valuation_date = date.fromisoformat(manifest_semantics["valuation_date"])
    publication_date = date.fromisoformat(manifest_semantics["publication_date"])
    if publication_date != LATE_PUBLICATION_DATE or valuation_date != LATE_VALUATION_DATE:
        raise DashboardV2Error("late_publication_fixed_dates_invalid")
    if generation_local_date < publication_date:
        raise DashboardV2Error("late_publication_generation_before_publication")
    if (
        manifest.get("source_record") != LATE_SOURCE_RECORD
        or manual.get("source_record") != LATE_SOURCE_RECORD
    ):
        raise DashboardV2Error("late_publication_manifest_source_invalid")
    if manifest.get("recorded_at_iso") != manual.get("recorded_at_iso"):
        raise DashboardV2Error("late_publication_recorded_at_mismatch")
    recorded_at = _local_date_from_timestamp(
        manifest.get("recorded_at_iso"), label="late_publication_recorded_at"
    )
    if recorded_at != publication_date:
        raise DashboardV2Error("late_publication_recorded_date_invalid")
    sealed_at = _local_date_from_timestamp(
        record.get("sealed_at"), label="late_publication_sealed_at"
    )
    if sealed_at != publication_date:
        raise DashboardV2Error("late_publication_sealed_date_invalid")
    if record_id[:8] != publication_date.strftime("%Y%m%d"):
        raise DashboardV2Error("late_publication_record_date_invalid")
    for key in ("published_at", "recorded_at"):
        if (
            record.get(key) is not None
            and _local_date_from_timestamp(record.get(key), label=f"late_publication_{key}")
            != publication_date
        ):
            raise DashboardV2Error(f"late_publication_{key}_date_invalid")
    if catalog_semantics.get("actual_sealed_at") is not None:
        if catalog_semantics["actual_sealed_at"] != record.get("sealed_at"):
            raise DashboardV2Error("late_publication_catalog_sealed_at_mismatch")
    if catalog_semantics.get("actual_published_at") is not None:
        if (
            _local_date_from_timestamp(
                catalog_semantics["actual_published_at"],
                label="late_publication_actual_published_at",
            )
            != publication_date
        ):
            raise DashboardV2Error("late_publication_published_date_invalid")
    if catalog_semantics.get("candidate_recorded_at") != manifest.get("recorded_at_iso"):
        raise DashboardV2Error("late_publication_candidate_recorded_at_mismatch")

    snapshot = manifest.get("data_snapshot")
    if not isinstance(snapshot, dict):
        raise DashboardV2Error("late_publication_data_snapshot_missing")
    for key in ("valuation_trade_date", "analysis_trade_date", "latest_complete_trade_date"):
        value = snapshot.get(key)
        if value is not None and str(value).replace("-", "") != "20260821":
            raise DashboardV2Error(f"late_publication_{key}_invalid")
    evidence_path = (manifest.get("files") or {}).get("valuation_evidence")
    manual_evidence_path = manual.get("valuation_evidence_path")
    if not isinstance(evidence_path, str) or not evidence_path:
        raise DashboardV2Error("late_publication_evidence_path_missing")
    if manual_evidence_path not in (None, evidence_path):
        raise DashboardV2Error("late_publication_evidence_path_mismatch")
    evidence_artifact = _stable_artifact(record_root / record_id / evidence_path, project_root)
    evidence = _json_object(evidence_artifact, label="late_publication_evidence")
    if evidence.get("schema_version") != "cn_dashboard_strict_market_close_evidence.v1":
        raise DashboardV2Error("late_publication_evidence_schema_invalid")
    if str(evidence.get("trade_date") or "").replace("-", "") != "20260821":
        raise DashboardV2Error("late_publication_evidence_date_invalid")
    if str(evidence.get("latest_complete_trade_date") or "").replace("-", "") != "20260821":
        raise DashboardV2Error("late_publication_evidence_close_date_invalid")
    for declared in (
        manual.get("valuation_evidence_sha256"),
        snapshot.get("valuation_evidence_sha256"),
    ):
        if declared is not None and declared != evidence_artifact.sha256:
            raise DashboardV2Error("late_publication_evidence_sha_mismatch")

    source_records = [
        row
        for row in catalog.get("records", [])
        if isinstance(row, dict) and row.get("record_id") == LATE_SOURCE_RECORD
    ]
    if len(source_records) != 1:
        raise DashboardV2Error("late_publication_source_record_invalid")
    source_closure = _catalog_record_closure(source_records[0])
    receipt_id = manifest_semantics["continuity_receipt_id"]
    receipt_matches = [
        value
        for value in catalog.get("receipts", [])
        if isinstance(value, dict) and value.get("receipt_id") == receipt_id
    ]
    if len(receipt_matches) != 1:
        raise DashboardV2Error("late_publication_receipt_invalid")
    receipt = receipt_matches[0]
    _validate_continuity_receipt(
        receipt,
        expected_active_record_id=LATE_SOURCE_RECORD,
        expected_checkpoint=source_closure,
        generation_local_date=generation_local_date,
        expected_created_local_date=valuation_date,
    )
    if (
        receipt.get("content_sha256") != manifest_semantics["continuity_receipt_sha256"]
        or receipt.get("created_at") != manifest_semantics["continuity_receipt_created_at"]
    ):
        raise DashboardV2Error("late_publication_receipt_binding_invalid")
    if manifest_semantics.get("continuity_checkpoint_digest") is not None and (
        manifest_semantics["continuity_checkpoint_digest"] != store_content_sha256(source_closure)
    ):
        raise DashboardV2Error("late_publication_receipt_checkpoint_invalid")

    projection = copy.deepcopy(catalog_delay)
    refs = [_source_ref(evidence_artifact)]
    current = generation_local_date == publication_date
    if current:
        _reject_unrelated_same_day_receipts(
            catalog.get("receipts"),
            expected_receipt_id=receipt_id,
            generation_local_date=valuation_date,
        )
    return projection, current, refs


def _generation_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or _SHANGHAI_TIMESTAMP_RE.fullmatch(value) is None:
        raise DashboardV2Error(f"{label}_invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DashboardV2Error(f"{label}_invalid") from exc
    if parsed.utcoffset() != _SHANGHAI.utcoffset(parsed):
        raise DashboardV2Error(f"{label}_timezone_invalid")
    return parsed


def _expected_change_type(previous_shares: float, current_shares: float) -> str:
    if previous_shares == 0 and current_shares > 0:
        return "NEW"
    if current_shares == 0 and previous_shares > 0:
        return "CLOSED"
    if current_shares > previous_shares:
        return "INCREASED"
    if current_shares < previous_shares:
        return "REDUCED"
    return "UNCHANGED"


def _validate_v1_changes(changes: Any) -> list[str]:
    if not isinstance(changes, list) or not changes:
        return ["canonical_v1_changes_missing"]

    errors: list[str] = []
    seen_symbols: set[str] = set()
    number_fields = (
        "previous_shares",
        "current_shares",
        "share_delta",
        "previous_market_value",
        "current_market_value",
        "market_value_delta",
        "nav_weight_delta",
        "equity_weight_delta",
    )
    for change in changes:
        if (
            not isinstance(change, dict)
            or not isinstance(change.get("symbol"), str)
            or _SYMBOL_RE.fullmatch(change["symbol"]) is None
            or change["symbol"] in seen_symbols
            or not isinstance(change.get("name"), str)
            or not change["name"]
            or change.get("change_type")
            not in {"NEW", "INCREASED", "REDUCED", "CLOSED", "UNCHANGED"}
        ):
            errors.append("canonical_v1_change_identity_invalid")
            continue
        seen_symbols.add(change["symbol"])
        if any(not _finite_number(change.get(field)) for field in number_fields):
            errors.append("canonical_v1_change_values_invalid")
            continue
        previous_shares = float(change["previous_shares"])
        current_shares = float(change["current_shares"])
        if abs(float(change["share_delta"]) - (current_shares - previous_shares)) > 1e-9 or change[
            "change_type"
        ] != _expected_change_type(previous_shares, current_shares):
            errors.append("canonical_v1_change_share_delta_invalid")
        if (
            abs(
                float(change["market_value_delta"])
                - (float(change["current_market_value"]) - float(change["previous_market_value"]))
            )
            > 0.01
        ):
            errors.append("canonical_v1_change_market_value_delta_invalid")
    return errors


def _validate_v1(v1_bundle: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(v1_bundle, dict):
        return ["canonical_v1_not_object"]
    if v1_bundle.get("schema_version") != V1_SCHEMA_VERSION:
        errors.append("canonical_v1_schema_invalid")
    observed_hash = v1_bundle.get("content_sha256")
    try:
        expected_hash = content_sha256(v1_bundle)
    except DashboardV2Error:
        expected_hash = None
    if (
        not isinstance(observed_hash, str)
        or _SHA256_RE.fullmatch(observed_hash) is None
        or observed_hash != expected_hash
    ):
        errors.append("canonical_v1_content_sha256_invalid")
    if v1_bundle.get("status") not in {"FRESH", "PARTIAL"}:
        errors.append("canonical_v1_not_usable")
    if v1_bundle.get("blockers") != []:
        errors.append("canonical_v1_has_blockers")
    if (
        v1_bundle.get("market") != MARKET
        or v1_bundle.get("strategy_label") != STRATEGY_LABEL
        or v1_bundle.get("read_only") is not True
    ):
        errors.append("canonical_v1_identity_invalid")
    portfolio = v1_bundle.get("portfolio")
    if not isinstance(portfolio, dict):
        errors.append("canonical_v1_portfolio_missing")
    else:
        if portfolio.get("return_method") != RETURN_METHOD:
            errors.append("canonical_v1_return_method_invalid")
        if not _close(portfolio.get("performance_initial_capital"), INITIAL_CAPITAL):
            errors.append("canonical_v1_initial_capital_invalid")
        if not _close(portfolio.get("excluded_external_flow"), 0.0):
            errors.append("canonical_v1_external_flow_not_zero")
    positions = v1_bundle.get("positions")
    if not isinstance(positions, list) or not positions:
        errors.append("canonical_v1_positions_missing")
    errors.extend(_validate_v1_changes(v1_bundle.get("changes")))
    return errors


def _source_ref(artifact: _Artifact) -> dict[str, str]:
    return {"path": artifact.relative_path, "sha256": artifact.sha256}


def _dedupe_source_refs(refs: list[dict[str, str]]) -> list[dict[str, str]]:
    by_path: dict[str, str] = {}
    for ref in refs:
        previous = by_path.setdefault(ref["path"], ref["sha256"])
        if previous != ref["sha256"]:
            raise DashboardV2Error(f"source_ref_hash_conflict:{ref['path']}")
    return [{"path": path, "sha256": digest} for path, digest in sorted(by_path.items())]


def _closure_artifacts(
    *,
    project_root: Path,
    record_root: Path,
    closure: Mapping[str, Any],
) -> tuple[dict[str, _Artifact], list[dict[str, str]]]:
    fields = {
        "manifest": ("manifest_path", "manifest_sha256"),
        "manual_manifest": (
            "manual_manifest_path",
            "manual_manifest_sha256",
        ),
        "ledger": ("ledger_path", "ledger_sha256"),
        "pnl": ("pnl_path", "pnl_sha256"),
    }
    artifacts: dict[str, _Artifact] = {}
    refs: list[dict[str, str]] = []
    for label, (path_key, sha_key) in fields.items():
        relative = closure.get(path_key)
        expected = closure.get(sha_key)
        if not isinstance(relative, str) or not relative:
            raise DashboardV2Error(f"active_closure_{path_key}_invalid")
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise DashboardV2Error(f"active_closure_{sha_key}_invalid")
        artifact = _stable_artifact(record_root / relative, project_root)
        if artifact.sha256 != expected:
            raise DashboardV2Error(f"active_closure_{label}_sha256_mismatch")
        artifacts[label] = artifact
        refs.append(_source_ref(artifact))
    return artifacts, refs


def _verify_v1_against_closure(
    v1_bundle: Mapping[str, Any],
    closure: Mapping[str, Any],
    ledger_artifact: _Artifact,
) -> None:
    if v1_bundle.get("latest_valid_record") != closure.get("record_id"):
        raise DashboardV2Error("canonical_v1_active_record_mismatch")
    evidence = v1_bundle.get("current_evidence")
    if not isinstance(evidence, dict):
        raise DashboardV2Error("canonical_v1_current_evidence_missing")
    for key in (
        "manifest_sha256",
        "manual_manifest_sha256",
        "ledger_sha256",
        "pnl_sha256",
        "financial_state_sha256",
    ):
        if evidence.get(key) != closure.get(key):
            raise DashboardV2Error(f"canonical_v1_active_closure_mismatch:{key}")
    try:
        ledger = pd.read_parquet(ledger_artifact.path)
    except Exception as exc:
        raise DashboardV2Error("active_ledger_parquet_unreadable") from exc
    required = {"symbol", "name", "shares", "avg_cost", "cost_basis"}
    if ledger.empty or not required.issubset(ledger.columns):
        raise DashboardV2Error("active_ledger_shape_invalid")
    if ledger["symbol"].astype(str).duplicated().any():
        raise DashboardV2Error("active_ledger_symbol_duplicate")
    rows = {str(row["symbol"]): row for row in ledger.to_dict(orient="records")}
    positions = v1_bundle.get("positions") or []
    if {row.get("symbol") for row in positions} != set(rows):
        raise DashboardV2Error("canonical_v1_active_symbols_mismatch")
    for position in positions:
        symbol = position.get("symbol")
        row = rows[symbol]
        if (
            str(position.get("name")) != str(row["name"])
            or not _close(position.get("shares"), row["shares"], tolerance=1e-9)
            or not _close(position.get("avg_cost"), row["avg_cost"])
            or not _close(position.get("cost_basis"), row["cost_basis"])
        ):
            raise DashboardV2Error(f"canonical_v1_active_ledger_mismatch:{symbol}")


def _result_frame(result: Any, *, symbol: str) -> pd.DataFrame:
    issues = getattr(result, "issues", None)
    if issues is None and isinstance(result, Mapping):
        issues = result.get("issues")
    if issues:
        raise DashboardV2Error(f"market_symbol_read_failed:{symbol}")
    frame = getattr(result, "frame", None)
    if frame is None and isinstance(result, Mapping):
        frame = result.get("frame")
    if not isinstance(frame, pd.DataFrame):
        raise DashboardV2Error(f"market_symbol_frame_invalid:{symbol}")
    return frame.copy()


def _row_date(row: Mapping[str, Any]) -> str:
    raw = row.get("trade_date", row.get("date", row.get("Date")))
    text = str(raw or "").replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise DashboardV2Error("market_row_trade_date_invalid")
    return date(int(text[:4]), int(text[4:6]), int(text[6:])).isoformat()


def _row_close(row: Mapping[str, Any]) -> float:
    value = row.get("close")
    if not _finite_number(value) or float(value) <= 0:
        raise DashboardV2Error("market_row_close_invalid")
    return float(value)


def _serving_artifact(reader: Any, symbol: str, project_root: Path) -> _Artifact:
    try:
        path = reader.resolve_symbol_path(symbol, universe_key="full_a")
    except Exception as exc:
        raise DashboardV2Error(f"market_serving_path_unavailable:{symbol}") from exc
    if not isinstance(path, Path):
        path = Path(path) if path else None
    if path is None:
        raise DashboardV2Error(f"market_serving_path_unavailable:{symbol}")
    return _stable_artifact(path, project_root)


def _mark_positions(
    *,
    v1_positions: list[dict[str, Any]],
    reader: Any,
    mark_date: str,
    suspended_symbols: set[str],
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    symbols = [str(row.get("symbol") or "") for row in v1_positions]
    if any(_SYMBOL_RE.fullmatch(symbol) is None for symbol in symbols):
        raise DashboardV2Error("canonical_v1_symbol_invalid")
    marked: list[dict[str, Any]] = []
    refs: list[dict[str, str]] = []
    for position in v1_positions:
        symbol = str(position["symbol"])
        serving = _serving_artifact(reader, symbol, project_root)
        refs.append(_source_ref(serving))
        try:
            exact_result = reader.read_symbol_frame(
                symbol,
                universe_key="full_a",
                start_date=_compact_date(mark_date),
                end_date=_compact_date(mark_date),
                columns=["symbol", "ts_code", "trade_date", "close"],
            )
        except Exception as exc:
            raise DashboardV2Error(f"market_symbol_read_failed:{symbol}") from exc
        frame = _result_frame(exact_result, symbol=symbol)
        rows = frame.to_dict(orient="records")
        exact = [row for row in rows if _row_date(row) == mark_date]
        if len(exact) > 1:
            raise DashboardV2Error(f"market_exact_close_duplicate:{symbol}")
        if exact:
            price_row = exact[0]
            evidence = "EXACT_CLOSE"
        else:
            if symbol not in suspended_symbols:
                raise DashboardV2Error(f"market_exact_close_missing_non_suspended:{symbol}")
            try:
                prior_result = reader.read_symbol_frame(
                    symbol,
                    universe_key="full_a",
                    end_date=_compact_date(mark_date),
                    columns=["symbol", "ts_code", "trade_date", "close"],
                )
            except Exception as exc:
                raise DashboardV2Error(f"market_suspension_carry_forward_failed:{symbol}") from exc
            prior = _result_frame(prior_result, symbol=symbol).to_dict(orient="records")
            prior = [row for row in prior if _row_date(row) < mark_date]
            if not prior:
                raise DashboardV2Error(f"market_suspension_prior_close_missing:{symbol}")
            price_row = max(prior, key=_row_date)
            evidence = "BOUND_SUSPENSION_CARRY_FORWARD"
        price_date = _row_date(price_row)
        price = _row_close(price_row)
        shares = float(position["shares"])
        avg_cost = float(position["avg_cost"])
        cost_basis = float(position["cost_basis"])
        if shares <= 0 or not _close(cost_basis, shares * avg_cost):
            raise DashboardV2Error(f"position_cost_identity_invalid:{symbol}")
        market_value = shares * price
        marked.append(
            {
                "symbol": symbol,
                "name": str(position["name"]),
                "shares": shares,
                "avg_cost": avg_cost,
                "cost_basis": cost_basis,
                "price": price,
                "price_date": price_date,
                "price_evidence_status": evidence,
                "market_value": market_value,
                "unrealized_pnl": market_value - cost_basis,
                "nav_weight": 0.0,
                "equity_weight": 0.0,
                "source_ref": _source_ref(serving),
            }
        )
    return marked, refs


def _updated_max_drawdown(v1_bundle: Mapping[str, Any], marked_total: float) -> float:
    portfolio = v1_bundle["portfolio"]
    points = portfolio.get("performance_points") or []
    values: list[float] = []
    for point in points:
        value = point.get("adjusted_total_value")
        if not _finite_number(value) or float(value) <= 0:
            raise DashboardV2Error("canonical_v1_performance_point_invalid")
        values.append(float(value))
    if len(values) < 2:
        raise DashboardV2Error("canonical_v1_performance_history_too_short")
    values.append(marked_total)
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    return drawdown


def build_v2_bundle(
    *,
    project_root: Path,
    v1_bundle: dict,
    v1_json_path: Path,
    record_root: Path,
    generation_local_date: date,
    generated_at: str,
    publication_attempt_id: str,
    market_reader: Any | None = None,
    v1_json_bytes_override: bytes | None = None,
) -> dict[str, Any]:
    """Build one exact, read-only v2 Dashboard bundle.

    The function performs no writes and makes no network calls.  When a reader
    is not injected it creates the repository's strict local
    :class:`MarketDataReader`.
    """

    root = project_root.resolve()
    records = record_root.resolve()
    if not isinstance(generation_local_date, date):
        raise DashboardV2Error("generation_local_date_invalid")
    generation_date_text = generation_local_date.isoformat()
    if (
        not isinstance(publication_attempt_id, str)
        or _ATTEMPT_RE.fullmatch(publication_attempt_id) is None
    ):
        raise DashboardV2Error("publication_attempt_id_invalid")
    if _generation_timestamp(generated_at, label="generated_at").date() != generation_local_date:
        raise DashboardV2Error("generated_at_local_date_mismatch")

    v1_errors = _validate_v1(v1_bundle)
    if v1_errors:
        raise DashboardV2Error("canonical_v1_invalid:" + ";".join(v1_errors))
    v1_artifact = (
        _artifact_from_bytes(v1_json_path, v1_json_bytes_override, root)
        if v1_json_bytes_override is not None
        else _stable_artifact(v1_json_path.resolve(), root)
    )
    if _json_object(v1_artifact, label="canonical_v1_file") != v1_bundle:
        raise DashboardV2Error("canonical_v1_file_body_mismatch")

    try:
        registered = load_registered_catalog(records)
    except StrategyRecordStoreError as exc:
        raise DashboardV2Error(f"record_store_invalid:{exc}") from exc
    if registered is None:
        raise DashboardV2Error("record_store_unregistered")
    pointer, catalog = registered
    pointer_artifact = _stable_artifact(records / "_record_store/current.v1.json", root)
    catalog_relative = pointer.get("catalog_path")
    if not isinstance(catalog_relative, str):
        raise DashboardV2Error("record_store_catalog_path_invalid")
    catalog_artifact = _stable_artifact(records / catalog_relative, root)
    if (
        _json_object(pointer_artifact, label="record_store_pointer") != pointer
        or _json_object(catalog_artifact, label="record_store_catalog") != catalog
    ):
        raise DashboardV2Error("record_store_loaded_body_drift")
    if catalog_artifact.sha256 != pointer.get("catalog_sha256"):
        raise DashboardV2Error("record_store_catalog_sha256_mismatch")

    closure = pointer.get("active_closure")
    if not isinstance(closure, dict) or not closure:
        raise DashboardV2Error("record_store_active_closure_missing")
    artifacts, closure_refs = _closure_artifacts(
        project_root=root, record_root=records, closure=closure
    )
    _verify_v1_against_closure(v1_bundle, closure, artifacts["ledger"])

    active_id = str(pointer.get("active_record_id") or "")
    active_lineage = [
        row
        for row in catalog.get("lineage_index", [])
        if isinstance(row, dict) and row.get("record_id") == active_id
    ]
    active_records = [
        row
        for row in catalog.get("records", [])
        if isinstance(row, dict) and row.get("record_id") == active_id
    ]
    financial_publication = False
    late_publication = False
    late_publication_current = False
    publication_delay_projection: dict[str, Any] | None = None
    late_publication_refs: list[dict[str, str]] = []
    if len(active_lineage) == 1 and len(active_records) == 1:
        lineage_row = active_lineage[0]
        record_row = active_records[0]
        if lineage_row.get("publication_class") == LATE_PUBLICATION_CLASS:
            (
                publication_delay_projection,
                late_publication_current,
                late_publication_refs,
            ) = _validate_late_publication_metadata(
                project_root=root,
                record_root=records,
                generation_local_date=generation_local_date,
                record=record_row,
                lineage=lineage_row,
                artifacts=artifacts,
                catalog=catalog,
                pointer=pointer,
            )
            late_publication = True
            financial_publication = late_publication_current
            delay_semantics = _publication_delay_semantics(
                publication_delay_projection, label="late_publication_catalog_delay"
            )
            receipt_id = (
                str(delay_semantics["continuity_receipt_id"])
                if late_publication_current
                else f"automation-{generation_local_date.strftime('%Y%m%d')}-daily-review-v1"
            )
        else:
            receipt_id = f"automation-{generation_local_date.strftime('%Y%m%d')}" "-daily-review-v1"
            financial_publication = (
                lineage_row.get("publication_class") == "OFFICIAL_FINANCIAL_STATE"
                and lineage_row.get("valuation_date") == generation_date_text
                and _local_date_from_timestamp(
                    record_row.get("sealed_at"), label="active_record_sealed_at"
                )
                == generation_local_date
            )
    else:
        receipt_id = f"automation-{generation_local_date.strftime('%Y%m%d')}" "-daily-review-v1"
    matching_receipts = [
        receipt
        for receipt in catalog.get("receipts", [])
        if isinstance(receipt, dict) and receipt.get("receipt_id") == receipt_id
    ]
    if late_publication:
        # The late record carries its own inherited receipt binding.  It is a
        # financial publication only on its actual publication date (8/22).
        # On a later required date it needs that date's exact no-action receipt
        # bound to the late record's active closure, just like any other active
        # financial checkpoint.
        if late_publication_current:
            continuity_status = "FINANCIAL_STATE_PUBLICATION"
            financial_state_changed = True
            continuity_receipt_id = None
            receipt_sha = None
            freshness_reason = LATE_FRESHNESS_REASON
        elif matching_receipts:
            if len(matching_receipts) != 1:
                raise DashboardV2Error("daily_continuity_receipt_duplicate")
            receipt = matching_receipts[0]
            _validate_continuity_receipt(
                receipt,
                expected_active_record_id=str(pointer.get("active_record_id") or ""),
                expected_checkpoint=closure,
                generation_local_date=generation_local_date,
            )
            continuity_status = "NO_ACTION_BOUND"
            financial_state_changed = False
            continuity_receipt_id = receipt_id
            receipt_sha = str(receipt["content_sha256"])
            freshness_reason = "CURRENT_DAILY_RECEIPT_AND_LATEST_LOCAL_CLOSE"
        else:
            _reject_unrelated_same_day_receipts(
                catalog.get("receipts"),
                expected_receipt_id=receipt_id,
                generation_local_date=generation_local_date,
            )
            continuity_status = "UNCONFIRMED"
            financial_state_changed = False
            continuity_receipt_id = None
            receipt_sha = None
            freshness_reason = "DAILY_CONTINUITY_RECEIPT_MISSING"
    elif matching_receipts:
        if len(matching_receipts) != 1:
            raise DashboardV2Error("daily_continuity_receipt_duplicate")
        receipt = matching_receipts[0]
        if financial_publication:
            # An official valuation can supersede exactly one inherited
            # predecessor receipt.  The receipt is expected to bind the
            # source record closure, not the newly published active closure.
            source_record_id = lineage_row.get("source_record_id")
            if not isinstance(source_record_id, str) or not source_record_id:
                raise DashboardV2Error("financial_publication_source_record_missing")
            source_records = [
                row
                for row in catalog.get("records", [])
                if isinstance(row, dict) and row.get("record_id") == source_record_id
            ]
            if len(source_records) != 1:
                raise DashboardV2Error("financial_publication_source_record_invalid")
            source_closure = _catalog_record_closure(source_records[0])
            _validate_continuity_receipt(
                receipt,
                expected_active_record_id=source_record_id,
                expected_checkpoint=source_closure,
                generation_local_date=generation_local_date,
            )
            continuity_status = "FINANCIAL_STATE_PUBLICATION"
            financial_state_changed = True
            continuity_receipt_id = None
            receipt_sha = None
            freshness_reason = "CURRENT_FINANCIAL_PUBLICATION_AND_LATEST_LOCAL_CLOSE"
        else:
            _validate_continuity_receipt(
                receipt,
                expected_active_record_id=str(pointer.get("active_record_id") or ""),
                expected_checkpoint=closure,
                generation_local_date=generation_local_date,
            )
            continuity_status = "NO_ACTION_BOUND"
            financial_state_changed = False
            continuity_receipt_id = receipt_id
            receipt_sha = str(receipt["content_sha256"])
            freshness_reason = "CURRENT_DAILY_RECEIPT_AND_LATEST_LOCAL_CLOSE"
    elif financial_publication:
        _reject_unrelated_same_day_receipts(
            catalog.get("receipts"),
            expected_receipt_id=receipt_id,
            generation_local_date=generation_local_date,
        )
        continuity_status = "FINANCIAL_STATE_PUBLICATION"
        financial_state_changed = True
        continuity_receipt_id = None
        receipt_sha = None
        freshness_reason = "CURRENT_FINANCIAL_PUBLICATION_AND_LATEST_LOCAL_CLOSE"
    else:
        # A receipt created on this local date but carrying another identifier
        # is not harmless noise: it can mask an unrelated checkpoint or a
        # malformed automation attempt.  Fail closed instead of downgrading it
        # to an apparently ordinary missing receipt.
        _reject_unrelated_same_day_receipts(
            catalog.get("receipts"),
            expected_receipt_id=receipt_id,
            generation_local_date=generation_local_date,
        )
        continuity_status = "UNCONFIRMED"
        financial_state_changed = False
        continuity_receipt_id = None
        receipt_sha = None
        freshness_reason = "DAILY_CONTINUITY_RECEIPT_MISSING"

    reader = market_reader or MarketDataReader(
        market=MARKET, data_root=root / "data", mode_policy="strict"
    )
    try:
        snapshot = reader.snapshot()
    except Exception as exc:
        raise DashboardV2Error("market_snapshot_unavailable") from exc
    coverage = snapshot.get("coverage")
    if (
        snapshot.get("healthy") is not True
        or snapshot.get("mode_policy") != "strict"
        or not isinstance(coverage, dict)
        or coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4"
        or coverage.get("complete") is not True
        or coverage.get("classification_sets_disjoint") is not True
        or coverage.get("true_missing_symbols") != []
    ):
        raise DashboardV2Error("market_snapshot_not_strict_complete")
    mark_compact = str(snapshot.get("latest_complete_trade_date") or "")
    if len(mark_compact) != 8 or not mark_compact.isdigit():
        raise DashboardV2Error("market_latest_complete_trade_date_invalid")
    mark_date = date(
        int(mark_compact[:4]), int(mark_compact[4:6]), int(mark_compact[6:])
    ).isoformat()
    if date.fromisoformat(mark_date) > generation_local_date:
        raise DashboardV2Error("market_mark_date_after_generation_date")
    if (
        late_publication
        and late_publication_current
        and mark_date != LATE_VALUATION_DATE.isoformat()
    ):
        # The late publication is a catch-up for the 8/21 close.  A weekend
        # build may serve that close on 8/22, but must never turn a synthetic
        # 8/22 market pointer into an economic valuation.
        raise DashboardV2Error("late_publication_market_date_mismatch")
    if coverage.get("latest_complete_trade_date") != mark_compact:
        raise DashboardV2Error("market_coverage_trade_date_mismatch")
    suspended = coverage.get("suspended_symbols")
    if not isinstance(suspended, list) or any(not isinstance(item, str) for item in suspended):
        raise DashboardV2Error("market_suspended_set_invalid")

    market_pointer_path = _source_path(
        snapshot.get("latest_pointer_path"),
        root,
        label="market_pointer_path",
    )
    market_pointer = _stable_artifact(
        market_pointer_path,
        root,
    )
    market_pointer_document = _json_object(market_pointer, label="market_pointer")
    market_manifest_path = _source_path(
        snapshot.get("manifest_path"),
        root,
        label="market_manifest_path",
    )
    market_manifest = _stable_artifact(
        market_manifest_path,
        root,
    )
    market_manifest_document = _json_object(market_manifest, label="market_manifest")
    expected_market_binding = {
        "snapshot_id": snapshot.get("snapshot_id"),
        "latest_complete_trade_date": mark_compact,
        "latest_trade_date": snapshot.get("latest_trade_date"),
        "coverage": coverage,
    }
    if any(
        market_pointer_document.get(key) != value for key, value in expected_market_binding.items()
    ):
        raise DashboardV2Error("market_pointer_snapshot_binding_mismatch")
    pointer_manifest_path = _source_path(
        market_pointer_document.get("manifest_path"),
        root,
        label="market_pointer_manifest_path",
    )
    if pointer_manifest_path != market_manifest_path:
        raise DashboardV2Error("market_pointer_manifest_path_mismatch")
    if (
        market_manifest_document.get("snapshot_id") != snapshot.get("snapshot_id")
        or market_manifest_document.get("latest_complete_trade_date") != mark_compact
        or market_manifest_document.get("latest_trade_date") != snapshot.get("latest_trade_date")
        or market_manifest_document.get("coverage") != coverage
    ):
        raise DashboardV2Error("market_manifest_snapshot_binding_mismatch")
    marked_positions, serving_refs = _mark_positions(
        v1_positions=copy.deepcopy(v1_bundle["positions"]),
        reader=reader,
        mark_date=mark_date,
        suspended_symbols=set(suspended),
        project_root=root,
    )
    market_pointer_after = _stable_artifact(market_pointer_path, root)
    if (
        market_pointer_after.raw != market_pointer.raw
        or market_pointer_after.sha256 != market_pointer.sha256
    ):
        raise DashboardV2Error("market_pointer_changed_during_mark")
    cash = float(v1_bundle["portfolio"]["cash"])
    if cash < 0:
        raise DashboardV2Error("canonical_v1_cash_invalid")
    market_value = sum(row["market_value"] for row in marked_positions)
    nav = cash + market_value
    if nav <= 0 or market_value <= 0:
        raise DashboardV2Error("research_mark_portfolio_value_invalid")
    unrealized = sum(row["unrealized_pnl"] for row in marked_positions)
    for row in marked_positions:
        row["nav_weight"] = row["market_value"] / nav
        row["equity_weight"] = row["market_value"] / market_value
    gross = market_value / nav
    cash_weight = cash / nav
    if not _close(gross + cash_weight, 1.0, tolerance=1e-10):
        raise DashboardV2Error("research_mark_weight_identity_invalid")

    portfolio = v1_bundle["portfolio"]
    anchor_date = _date_text(
        portfolio.get("performance_end_date"),
        label="canonical_v1_performance_end_date",
    )
    if anchor_date > mark_date:
        raise DashboardV2Error("research_mark_precedes_canonical_anchor")
    anchor_value = float(portfolio["adjusted_total_value"])
    if anchor_value <= 0:
        raise DashboardV2Error("canonical_v1_anchor_value_invalid")
    current_return = nav / INITIAL_CAPITAL - 1.0
    interval_return = nav / anchor_value - 1.0
    maximum_drawdown = _updated_max_drawdown(v1_bundle, nav)

    benchmark_dates = {
        _date_text(row.get("end_date"), label="benchmark_end_date")
        for row in v1_bundle.get("benchmarks", [])
        if isinstance(row, dict)
    }
    if len(benchmark_dates) != 1:
        raise DashboardV2Error("canonical_v1_benchmark_dates_not_aligned")
    benchmark_as_of = next(iter(benchmark_dates))
    if benchmark_as_of > mark_date:
        raise DashboardV2Error("canonical_v1_benchmark_date_after_mark")

    history = v1_bundle.get("history") or {}
    history_complete = (
        history.get("evidence_status") == "CANONICAL_PERFORMANCE_CLOSURE"
        and int(history.get("rejected_record_count") or 0) == 0
    )
    legacy_caveats = sorted(
        {
            str(item)
            for item in v1_bundle.get("warnings", [])
            if isinstance(item, str) and item.strip()
        }
    )
    source_refs = _dedupe_source_refs(
        [
            _source_ref(v1_artifact),
            _source_ref(pointer_artifact),
            _source_ref(catalog_artifact),
            _source_ref(market_pointer),
            _source_ref(market_manifest),
            *closure_refs,
            *serving_refs,
            *late_publication_refs,
        ]
    )
    freshness_updated = continuity_status != "UNCONFIRMED"
    holdings_valid_through = (
        mark_date
        if late_publication and late_publication_current
        else (
            generation_date_text
            if freshness_updated
            else _date_text(v1_bundle.get("latest_data_date"), label="anchor_data_date")
        )
    )
    freshness_valid_through = f"{generation_date_text}T23:59:59+08:00"
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "publication_attempt_id": publication_attempt_id,
        "generated_at": generated_at,
        "generation_local_date": generation_date_text,
        "canonical_v1": copy.deepcopy(v1_bundle),
        "canonical_v1_ref": _source_ref(v1_artifact),
        "integrity": {"status": "VERIFIED"},
        "continuity_authority": {
            "status": continuity_status,
            "anchor_record_id": str(pointer["active_record_id"]),
            "anchor_data_date": _date_text(
                v1_bundle.get("latest_data_date"), label="anchor_data_date"
            ),
            "anchor_financial_state_sha256": str(closure["financial_state_sha256"]),
            "active_ledger_sha256": str(closure["ledger_sha256"]),
            "holdings_valid_through": holdings_valid_through,
            "financial_state_changed": financial_state_changed,
            "receipt_id": continuity_receipt_id,
            "receipt_content_sha256": receipt_sha,
        },
        "publication_delay": publication_delay_projection,
        "freshness": {
            "status": "UPDATED" if freshness_updated else "STALE",
            "scope": DAILY_SCOPE,
            "mark_as_of": mark_date,
            "generated_at": generated_at,
            "valid_through": freshness_valid_through,
            "source_kind": MARK_SOURCE_KIND,
            "reason": freshness_reason,
        },
        "completeness": {
            "current_holdings": "COMPLETE" if freshness_updated else "STALE",
            "current_absolute_performance": ("COMPLETE" if freshness_updated else "STALE"),
            "canonical_history": "COMPLETE" if history_complete else "PARTIAL",
            "benchmark_relative": (
                "COMPLETE" if benchmark_as_of == mark_date else "AS_OF_PRIOR_DATE"
            ),
            "benchmark_as_of": benchmark_as_of,
            "legacy_caveats": legacy_caveats,
        },
        "research_mark": {
            "status": "AVAILABLE",
            "authority": VIEW_ONLY_AUTHORITY,
            "source_kind": MARK_SOURCE_KIND,
            "mark_date": mark_date,
            "anchor_record_id": str(pointer["active_record_id"]),
            "base_ledger_sha256": str(closure["ledger_sha256"]),
            "base_financial_state_sha256": str(closure["financial_state_sha256"]),
            "positions": marked_positions,
            "portfolio": {
                "cash": cash,
                "market_value": market_value,
                "nav": nav,
                "unrealized_pnl": unrealized,
                "cash_weight": cash_weight,
                "gross_exposure": gross,
            },
            "current_absolute_performance": {
                "point_date": mark_date,
                "anchor_date": anchor_date,
                "marked_nav": nav,
                "initial_capital": INITIAL_CAPITAL,
                "cumulative_return": current_return,
                "continuity_interval_return": interval_return,
                "max_drawdown": maximum_drawdown,
                "evidence_status": "HASH_BOUND_CONTINUITY_MARK",
                "authority": VIEW_ONLY_AUTHORITY,
            },
            "canonical_effect": "NONE",
            "ledger_effect": "NONE",
            "performance_effect": "NONE",
            "paper_effect": "NONE",
            "trade_effect": "NONE",
        },
        "source_refs": source_refs,
    }
    bundle["content_sha256"] = content_sha256(bundle)
    errors = validate_v2_shape(bundle)
    if errors:
        raise DashboardV2Error("v2_shape_invalid:" + ";".join(errors))
    return bundle


def validate_v2_shape(bundle: dict) -> list[str]:
    """Validate the v2 structural and accounting invariants."""

    if not isinstance(bundle, dict):
        return ["bundle_not_object"]
    required = {
        "schema_version",
        "publication_attempt_id",
        "generated_at",
        "generation_local_date",
        "canonical_v1",
        "canonical_v1_ref",
        "integrity",
        "continuity_authority",
        "publication_delay",
        "freshness",
        "completeness",
        "research_mark",
        "source_refs",
        "content_sha256",
    }
    errors: list[str] = []
    if set(bundle) != required:
        errors.append("bundle_keys_invalid")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    attempt_id = bundle.get("publication_attempt_id")
    if not isinstance(attempt_id, str) or _ATTEMPT_RE.fullmatch(attempt_id) is None:
        errors.append("publication_attempt_id_invalid")
    try:
        generation_date = _date_text(
            bundle.get("generation_local_date"), label="generation_local_date"
        )
    except DashboardV2Error:
        generation_date = ""
        errors.append("generation_local_date_invalid")
    try:
        generated = _generation_timestamp(bundle.get("generated_at"), label="generated_at")
    except DashboardV2Error:
        errors.append("generated_at_invalid")
    else:
        if generated.date().isoformat() != generation_date:
            errors.append("generated_at_local_date_mismatch")
    errors.extend(_validate_v1(bundle.get("canonical_v1")))
    ref = bundle.get("canonical_v1_ref")
    if not _valid_source_ref(ref):
        errors.append("canonical_v1_ref_invalid")
    integrity = bundle.get("integrity")
    if integrity != {"status": "VERIFIED"}:
        errors.append("integrity_invalid")

    publication_delay = bundle.get("publication_delay")
    late_projection = publication_delay is not None
    late_valuation_date: str | None = None
    if publication_delay is not None:
        try:
            late_valuation_date = _publication_delay_semantics(
                publication_delay, label="publication_delay_projection"
            )["valuation_date"]
        except DashboardV2Error as exc:
            errors.append(str(exc))

    continuity = bundle.get("continuity_authority")
    continuity_required = {
        "status",
        "anchor_record_id",
        "anchor_data_date",
        "anchor_financial_state_sha256",
        "active_ledger_sha256",
        "holdings_valid_through",
        "financial_state_changed",
        "receipt_id",
        "receipt_content_sha256",
    }
    if not isinstance(continuity, dict) or set(continuity) != continuity_required:
        errors.append("continuity_authority_invalid")
    else:
        status = continuity.get("status")
        if status not in {
            "NO_ACTION_BOUND",
            "FINANCIAL_STATE_PUBLICATION",
            "UNCONFIRMED",
        }:
            errors.append("continuity_status_invalid")
        try:
            holdings_valid_through = _date_text(
                continuity.get("holdings_valid_through"),
                label="holdings_valid_through",
            )
        except DashboardV2Error:
            holdings_valid_through = ""
            errors.append("continuity_valid_through_invalid")
        expected_holdings_date = generation_date
        if late_projection and status == "FINANCIAL_STATE_PUBLICATION":
            expected_holdings_date = late_valuation_date or ""
        if status != "UNCONFIRMED" and holdings_valid_through != expected_holdings_date:
            errors.append("continuity_valid_through_invalid")
        if not str(continuity.get("anchor_record_id") or ""):
            errors.append("continuity_anchor_record_invalid")
        for key in (
            "anchor_financial_state_sha256",
            "active_ledger_sha256",
        ):
            if not _valid_sha(continuity.get(key)):
                errors.append(f"continuity_{key}_invalid")
        if status == "NO_ACTION_BOUND":
            if (
                continuity.get("financial_state_changed") is not False
                or not str(continuity.get("receipt_id") or "")
                or not _valid_sha(continuity.get("receipt_content_sha256"))
            ):
                errors.append("continuity_no_action_binding_invalid")
        elif status == "FINANCIAL_STATE_PUBLICATION" and (
            continuity.get("financial_state_changed") is not True
            or continuity.get("receipt_id") is not None
            or continuity.get("receipt_content_sha256") is not None
        ):
            errors.append("continuity_financial_publication_invalid")
        elif status == "UNCONFIRMED" and (
            continuity.get("financial_state_changed") is not False
            or continuity.get("receipt_id") is not None
            or continuity.get("receipt_content_sha256") is not None
            or holdings_valid_through != continuity.get("anchor_data_date")
        ):
            errors.append("continuity_unconfirmed_invalid")

    freshness = bundle.get("freshness")
    freshness_required = {
        "status",
        "scope",
        "mark_as_of",
        "generated_at",
        "valid_through",
        "source_kind",
        "reason",
    }
    if not isinstance(freshness, dict) or set(freshness) != freshness_required:
        errors.append("freshness_invalid")
    else:
        if freshness.get("status") not in {"UPDATED", "STALE"}:
            errors.append("freshness_status_invalid")
        if freshness.get("scope") != DAILY_SCOPE:
            errors.append("freshness_scope_invalid")
        if freshness.get("source_kind") != MARK_SOURCE_KIND:
            errors.append("freshness_source_kind_invalid")
        if freshness.get("generated_at") != bundle.get("generated_at"):
            errors.append("freshness_generated_at_mismatch")
        if freshness.get("valid_through") != (f"{generation_date}T23:59:59+08:00"):
            errors.append("freshness_valid_through_invalid")
        if freshness.get("reason") not in {
            "CURRENT_DAILY_RECEIPT_AND_LATEST_LOCAL_CLOSE",
            "CURRENT_FINANCIAL_PUBLICATION_AND_LATEST_LOCAL_CLOSE",
            LATE_FRESHNESS_REASON,
            "DAILY_CONTINUITY_RECEIPT_MISSING",
        }:
            errors.append("freshness_reason_invalid")
        if (
            freshness.get("status") == "UPDATED"
            and freshness.get("reason") == "DAILY_CONTINUITY_RECEIPT_MISSING"
        ):
            errors.append("freshness_updated_reason_invalid")
        if (
            freshness.get("status") == "UPDATED"
            and freshness.get("reason") == LATE_FRESHNESS_REASON
            and (
                not late_projection
                or not isinstance(continuity, dict)
                or continuity.get("status") != "FINANCIAL_STATE_PUBLICATION"
                or freshness.get("mark_as_of") != late_valuation_date
                or continuity.get("holdings_valid_through") != freshness.get("mark_as_of")
            )
        ):
            errors.append("late_publication_freshness_binding_invalid")
        if (
            freshness.get("status") == "STALE"
            and freshness.get("reason") != "DAILY_CONTINUITY_RECEIPT_MISSING"
        ):
            errors.append("freshness_stale_reason_invalid")

    completeness = bundle.get("completeness")
    completeness_required = {
        "current_holdings",
        "current_absolute_performance",
        "canonical_history",
        "benchmark_relative",
        "benchmark_as_of",
        "legacy_caveats",
    }
    if not isinstance(completeness, dict) or set(completeness) != completeness_required:
        errors.append("completeness_invalid")
    else:
        if completeness.get("current_holdings") not in {"COMPLETE", "STALE"}:
            errors.append("current_holdings_completeness_invalid")
        if completeness.get("current_absolute_performance") not in {
            "COMPLETE",
            "STALE",
        }:
            errors.append("current_performance_completeness_invalid")
        if completeness.get("canonical_history") not in {"COMPLETE", "PARTIAL"}:
            errors.append("canonical_history_completeness_invalid")
        if completeness.get("benchmark_relative") not in {
            "COMPLETE",
            "AS_OF_PRIOR_DATE",
        }:
            errors.append("benchmark_relative_completeness_invalid")
        caveats = completeness.get("legacy_caveats")
        if not isinstance(caveats, list) or any(
            not isinstance(item, str) or not item for item in caveats
        ):
            errors.append("legacy_caveats_invalid")
        freshness_status = freshness.get("status") if isinstance(freshness, dict) else None
        expected_current = "COMPLETE" if freshness_status == "UPDATED" else "STALE"
        if (
            completeness.get("current_holdings") != expected_current
            or completeness.get("current_absolute_performance") != expected_current
        ):
            errors.append("current_completeness_freshness_mismatch")

    mark = bundle.get("research_mark")
    mark_required = {
        "status",
        "authority",
        "source_kind",
        "mark_date",
        "anchor_record_id",
        "base_ledger_sha256",
        "base_financial_state_sha256",
        "positions",
        "portfolio",
        "current_absolute_performance",
        "canonical_effect",
        "ledger_effect",
        "performance_effect",
        "paper_effect",
        "trade_effect",
    }
    if not isinstance(mark, dict) or set(mark) != mark_required:
        errors.append("research_mark_invalid")
    else:
        if (
            mark.get("status") != "AVAILABLE"
            or mark.get("authority") != VIEW_ONLY_AUTHORITY
            or mark.get("source_kind") != MARK_SOURCE_KIND
        ):
            errors.append("research_mark_identity_invalid")
        if any(
            mark.get(key) != "NONE"
            for key in (
                "canonical_effect",
                "ledger_effect",
                "performance_effect",
                "paper_effect",
                "trade_effect",
            )
        ):
            errors.append("research_mark_effect_invalid")
        _validate_mark_accounting(
            mark,
            bundle.get("canonical_v1"),
            errors,
        )

    refs = bundle.get("source_refs")
    if not isinstance(refs, list) or not refs:
        errors.append("source_refs_missing")
    elif any(not _valid_source_ref(item) for item in refs):
        errors.append("source_ref_invalid")
    elif refs != sorted(refs, key=lambda item: item["path"]):
        errors.append("source_refs_not_sorted")
    elif len({item["path"] for item in refs}) != len(refs):
        errors.append("source_refs_duplicate")
    elif _valid_source_ref(ref) and ref not in refs:
        errors.append("canonical_v1_ref_not_registered")
    else:
        _validate_required_source_refs(bundle, refs, errors)
    content_hash = bundle.get("content_sha256")
    try:
        expected_content_hash = content_sha256(bundle)
    except DashboardV2Error:
        expected_content_hash = None
    if not _valid_sha(content_hash) or content_hash != expected_content_hash:
        errors.append("content_sha256_invalid")
    return errors


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _valid_source_ref(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        return False
    path = value.get("path")
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith("/")
        and "\\" not in path
        and ".." not in Path(path).parts
        and _valid_sha(value.get("sha256"))
    )


def _validate_mark_accounting(
    mark: Mapping[str, Any],
    canonical_v1: Any,
    errors: list[str],
) -> None:
    positions = mark.get("positions")
    portfolio = mark.get("portfolio")
    performance = mark.get("current_absolute_performance")
    if not isinstance(positions, list) or not positions:
        errors.append("research_mark_positions_invalid")
        return
    position_required = {
        "symbol",
        "name",
        "shares",
        "avg_cost",
        "cost_basis",
        "price",
        "price_date",
        "price_evidence_status",
        "market_value",
        "unrealized_pnl",
        "nav_weight",
        "equity_weight",
        "source_ref",
    }
    symbols: set[str] = set()
    valid_rows: list[dict[str, Any]] = []
    for row in positions:
        if not isinstance(row, dict) or set(row) != position_required:
            errors.append("research_mark_position_shape_invalid")
            continue
        valid_rows.append(row)
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
            errors.append("research_mark_position_symbol_invalid")
        elif symbol in symbols:
            errors.append("research_mark_position_symbol_duplicate")
        symbols.add(str(symbol))
        if row.get("price_evidence_status") not in {
            "EXACT_CLOSE",
            "BOUND_SUSPENSION_CARRY_FORWARD",
        }:
            errors.append("research_mark_price_evidence_invalid")
        if not _valid_source_ref(row.get("source_ref")):
            errors.append("research_mark_position_source_ref_invalid")
        if (
            not _close(
                row.get("cost_basis"),
                _as_float(row.get("shares")) * _as_float(row.get("avg_cost")),
            )
            or not _close(
                row.get("market_value"),
                _as_float(row.get("shares")) * _as_float(row.get("price")),
            )
            or not _close(
                row.get("unrealized_pnl"),
                _as_float(row.get("market_value")) - _as_float(row.get("cost_basis")),
            )
        ):
            errors.append(f"research_mark_position_identity_invalid:{symbol}")
    portfolio_required = {
        "cash",
        "market_value",
        "nav",
        "unrealized_pnl",
        "cash_weight",
        "gross_exposure",
    }
    if not isinstance(portfolio, dict) or set(portfolio) != portfolio_required:
        errors.append("research_mark_portfolio_invalid")
        return
    market_value = sum(_as_float(row.get("market_value")) for row in valid_rows)
    unrealized = sum(_as_float(row.get("unrealized_pnl")) for row in valid_rows)
    nav = _as_float(portfolio.get("nav"))
    if (
        not _close(portfolio.get("market_value"), market_value)
        or not _close(portfolio.get("unrealized_pnl"), unrealized)
        or not _close(_as_float(portfolio.get("cash")) + market_value, nav)
        or not _close(
            portfolio.get("cash_weight", math.nan) + portfolio.get("gross_exposure", math.nan),
            1.0,
            tolerance=1e-9,
        )
        or not _close(
            sum(_as_float(row.get("nav_weight")) for row in valid_rows),
            portfolio.get("gross_exposure"),
            tolerance=1e-9,
        )
        or not _close(
            sum(_as_float(row.get("equity_weight")) for row in valid_rows),
            1.0,
            tolerance=1e-9,
        )
    ):
        errors.append("research_mark_portfolio_identity_invalid")
    performance_required = {
        "point_date",
        "anchor_date",
        "marked_nav",
        "initial_capital",
        "cumulative_return",
        "continuity_interval_return",
        "max_drawdown",
        "evidence_status",
        "authority",
    }
    if not isinstance(performance, dict) or set(performance) != performance_required:
        errors.append("current_absolute_performance_invalid")
        return
    canonical_portfolio = canonical_v1.get("portfolio") if isinstance(canonical_v1, dict) else None
    if not isinstance(canonical_portfolio, dict):
        errors.append("current_absolute_performance_canonical_missing")
        return
    anchor_date = canonical_portfolio.get("performance_end_date")
    anchor_nav = _as_float(canonical_portfolio.get("adjusted_total_value"))
    expected_interval_return = (
        nav / anchor_nav - 1.0 if math.isfinite(anchor_nav) and anchor_nav > 0 else math.nan
    )
    try:
        expected_drawdown = _updated_max_drawdown(canonical_v1, nav)
    except DashboardV2Error:
        expected_drawdown = math.nan
    if (
        performance.get("evidence_status") != "HASH_BOUND_CONTINUITY_MARK"
        or performance.get("authority") != VIEW_ONLY_AUTHORITY
        or performance.get("point_date") != mark.get("mark_date")
        or performance.get("anchor_date") != anchor_date
        or not _close(performance.get("marked_nav"), nav)
        or not _close(performance.get("initial_capital"), INITIAL_CAPITAL)
        or not _close(
            performance.get("cumulative_return"),
            nav / INITIAL_CAPITAL - 1.0,
            tolerance=1e-10,
        )
        or not _close(
            performance.get("continuity_interval_return"),
            expected_interval_return,
            tolerance=1e-10,
        )
        or not _close(
            performance.get("max_drawdown"),
            expected_drawdown,
            tolerance=1e-10,
        )
    ):
        errors.append("current_absolute_performance_identity_invalid")


def _validate_required_source_refs(
    bundle: Mapping[str, Any],
    refs: list[dict[str, str]],
    errors: list[str],
) -> None:
    paths = {ref["path"] for ref in refs}
    store_root = "results/strategy_records/CN/aggressive_tech_manufacturing"
    required = {
        f"{store_root}/_record_store/current.v1.json",
        "data/parquet/cn/_latest.json",
    }
    continuity = bundle.get("continuity_authority")
    anchor = str(continuity.get("anchor_record_id") or "") if isinstance(continuity, dict) else ""
    if anchor:
        required.update(
            {
                f"{store_root}/{anchor}/manifest.json",
                f"{store_root}/{anchor}/manual_execution_manifest.json",
                (f"{store_root}/{anchor}/" "ledger_after_manual_switch.parquet"),
                f"{store_root}/{anchor}/pnl_summary.csv",
            }
        )
    missing = sorted(required - paths)
    if missing:
        errors.append("required_source_refs_missing:" + ",".join(missing))
    if not any(
        path.startswith(f"{store_root}/_record_store/catalogs/")
        and path.endswith("/catalog.v3.json")
        for path in paths
    ):
        errors.append("store_catalog_source_ref_missing")
    if not any(
        path.startswith("data/parquet/cn/_snapshots/") and path.endswith(".json") for path in paths
    ):
        errors.append("market_snapshot_source_ref_missing")
    mark = bundle.get("research_mark")
    positions = mark.get("positions", []) if isinstance(mark, dict) else []
    for row in positions:
        if not isinstance(row, dict) or not _valid_source_ref(row.get("source_ref")):
            continue
        source = row["source_ref"]
        expected_suffix = f"/serving/bars/symbol={row.get('symbol')}/bars.parquet"
        if source["path"] not in paths:
            errors.append(f"position_source_ref_not_registered:{row.get('symbol')}")
        if not source["path"].endswith(expected_suffix):
            errors.append(f"position_source_ref_path_invalid:{row.get('symbol')}")


def verify_v2_source_refs(
    bundle: dict,
    project_root: Path,
    *,
    v1_bytes_override: bytes | None = None,
) -> list[str]:
    """Verify every v2 physical source reference by exact project bytes."""

    errors: list[str] = []
    refs = bundle.get("source_refs")
    if not isinstance(refs, list):
        return ["source_refs_missing"]
    canonical_ref = bundle.get("canonical_v1_ref")
    for index, ref in enumerate(refs):
        if not _valid_source_ref(ref):
            errors.append(f"source_ref_invalid:{index}")
            continue
        if v1_bytes_override is not None and ref == canonical_ref:
            raw = v1_bytes_override
        else:
            try:
                artifact = _stable_artifact(
                    project_root.resolve() / ref["path"],
                    project_root.resolve(),
                )
                raw = artifact.raw
            except DashboardV2Error as exc:
                errors.append(str(exc))
                continue
        if ref == canonical_ref:
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                errors.append("canonical_v1_source_invalid_json")
                continue
            if parsed != bundle.get("canonical_v1"):
                errors.append("canonical_v1_source_body_mismatch")
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            errors.append(f"source_ref_sha256_mismatch:{ref['path']}")
    mark = bundle.get("research_mark")
    if isinstance(mark, dict):
        registered = {(ref.get("path"), ref.get("sha256")) for ref in refs if isinstance(ref, dict)}
        for row in mark.get("positions", []):
            if isinstance(row, dict):
                source = row.get("source_ref")
                if (
                    isinstance(source, dict)
                    and (source.get("path"), source.get("sha256")) not in registered
                ):
                    errors.append(f"position_source_ref_not_registered:{row.get('symbol')}")
    return errors


__all__ = [
    "DashboardV2Error",
    "SCHEMA_VERSION",
    "build_v2_bundle",
    "canonical_json_bytes",
    "content_sha256",
    "validate_v2_shape",
    "verify_v2_source_refs",
]
