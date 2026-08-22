"""Shared fail-closed helpers for the CN aggressive holdings Dashboard.

This module is intentionally offline and read-only.  It consumes archived
strategy records and a local benchmark file; it never calls a provider, builds
portfolio candidates, mutates a strategy pointer, or creates execution state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import stat
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from quant_investor.strategy_records.store import (
    StrategyRecordStoreError,
    load_archive_binding,
    load_registered_catalog,
)
from quant_investor.strategy_records.performance import load_performance_history

SCHEMA_VERSION = "cn_aggressive_dashboard.v1"
HISTORY_INTEGRITY_SCHEMA_VERSION = "cn_aggressive_dashboard_history_integrity.v2"
ARCHIVE_LOCATOR_SCHEMA_VERSION = "myquant.strategy_record_archive_locator.v1"
ARCHIVE_MANIFEST_SCHEMA_VERSION = "myquant.strategy_record_archive_manifest.v1"
ARCHIVE_RESTORE_RECEIPT_SCHEMA_VERSION = "myquant.strategy_record_archive_restore_receipt.v1"
TRANSACTION_BACKFILL_PROVENANCE_SCHEMA_VERSION = "cn_aggressive_transaction_backfill_provenance.v1"
STRICT_MARKET_CLOSE_EVIDENCE_SCHEMA_VERSION = "cn_dashboard_strict_market_close_evidence.v1"
ORDINARY_PUBLICATION_CLASS = "ORDINARY_SAME_DAY_OFFICIAL_VALUATION"
LATE_PUBLICATION_CLASS = "LATE_OFFICIAL_VALUATION_PUBLICATION"
LATE_PUBLICATION_SCHEMA = "myquant.strategy_record_publication_delay.v1"
LATE_PUBLICATION_REASON = "SHARED_CHECKOUT_SAFETY_GATE_DELAY"
LATE_SOURCE_RECORD = "20260820_1321"
MARKET = "CN"
STRATEGY = "aggressive_tech_manufacturing"
LEGACY_RETURN_METHOD = "initial_capital_return_excluding_external_flows"
CANONICAL_RETURN_METHOD = "flow_neutral_unitization_v1"
RECORD_NAME_RE = re.compile(r"^[0-9]{8}_[0-9]{4}$")
SYMBOL_RE = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GENERATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ALLOWED_BENCHMARK_SOURCES = {
    "eastmoney.push2his.kline",
    "tushare.index_daily",
}
ALLOWED_BENCHMARK_COVERAGE = {"exact_close", "previous_trading_day_ffill"}
RISK_FREE_SOURCE = "chinabond.mof_govt_yield_curve"
RISK_FREE_TENOR = "1Y"
RISK_FREE_SOURCE_URL = "https://yield.chinabond.com.cn/cbweb-mn/pgxh/showHistory"
BENCHMARK_SPECS = (
    {
        "id": "CSI300",
        "name": "沪深300",
        "ts_code": "000300.SH",
        "point_prefix": "csi300",
    },
    {
        "id": "STAR50",
        "name": "科创50",
        "ts_code": "000688.SH",
        "point_prefix": "star50",
    },
    {
        "id": "CHINEXT",
        "name": "创业板指",
        "ts_code": "399006.SZ",
        "point_prefix": "chinext",
    },
)
REQUIRED_LEDGER_COLUMNS = {
    "symbol",
    "name",
    "shares",
    "avg_cost",
    "cost_basis",
    "current_price",
    "current_value",
    "unrealized_pnl",
    "equity_sleeve_weight",
    "nav_weight",
}
ACCOUNTING_FIELDS = (
    "cash_after",
    "market_value_after",
    "total_value_after",
    "portfolio_pnl_after",
    "realized_pnl_from_rebalance",
)
HISTORICAL_ACCOUNTING_FIELDS = (
    "cash_after",
    "market_value_after",
    "total_value_after",
    "portfolio_pnl_after",
    "realized_pnl_from_rebalance",
)
AUTHORITY_FLAGS = {
    "benchmark_provider_calls": False,
    "broker_calls": False,
    "candidate_generation": False,
    "holdings_writes": False,
    "order_calls": False,
    "portfolio_recomputation": False,
    "provider_calls": False,
    "strategy_record_writes": False,
    "trade_calls": False,
    "v17_pointer_mutation": False,
}


class DashboardInputError(RuntimeError):
    """Raised when a required Dashboard input cannot be closed exactly."""


@dataclass(frozen=True)
class StableArtifact:
    path: Path
    relative_path: str
    data: bytes
    sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise DashboardInputError(f"artifact_outside_project:{path}") from exc


def stable_read(path: Path, project_root: Path) -> StableArtifact:
    """Read a regular non-symlink file twice and bind its exact bytes."""

    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise DashboardInputError(f"artifact_missing:{path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DashboardInputError(f"artifact_not_regular_non_symlink:{path}")
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise DashboardInputError(f"artifact_unstable_double_read:{path}")
    return StableArtifact(
        path=path,
        relative_path=_relative_to_project(path, project_root),
        data=first,
        sha256=sha256_bytes(first),
    )


def _stable_stream_ref(ref: dict[str, Any], project_root: Path, *, label: str) -> dict[str, str]:
    """Verify a project-relative physical ref without loading it into RAM."""

    if set(ref) != {"path", "sha256", "bytes", "media_type"}:
        raise DashboardInputError(f"{label}_shape_invalid")
    relative_path = ref.get("path")
    declared_sha = ref.get("sha256")
    declared_bytes = ref.get("bytes")
    media_type = ref.get("media_type")
    if (
        not isinstance(relative_path, str)
        or relative_path.startswith("/")
        or "\\" in relative_path
        or ".." in Path(relative_path).parts
        or not isinstance(declared_sha, str)
        or not SHA256_RE.fullmatch(declared_sha)
        or not isinstance(declared_bytes, int)
        or isinstance(declared_bytes, bool)
        or declared_bytes < 0
        or not isinstance(media_type, str)
        or not media_type
    ):
        raise DashboardInputError(f"{label}_invalid")
    path = project_root / relative_path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(project_root.resolve())
        metadata_before = path.lstat()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise DashboardInputError(f"{label}_unavailable:{relative_path}") from exc
    if (
        resolved != path.absolute()
        or stat.S_ISLNK(metadata_before.st_mode)
        or not stat.S_ISREG(metadata_before.st_mode)
        or metadata_before.st_nlink != 1
        or metadata_before.st_size != declared_bytes
    ):
        raise DashboardInputError(f"{label}_unsafe:{relative_path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        metadata_after = path.lstat()
    except OSError as exc:
        raise DashboardInputError(f"{label}_unavailable:{relative_path}") from exc
    identity = lambda item: (  # noqa: E731 - exact stable identity
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(metadata_before) != identity(metadata_after):
        raise DashboardInputError(f"{label}_unstable:{relative_path}")
    if digest.hexdigest() != declared_sha:
        raise DashboardInputError(f"{label}_sha_mismatch:{relative_path}")
    return {"path": relative_path, "sha256": declared_sha}


def _stable_external_read(path: Path, project_root: Path) -> tuple[bytes, str]:
    """Read an allowlisted external provenance source twice."""

    allowed_roots = (
        (Path.home() / ".codex" / "automations").resolve(),
        (project_root / ".codex" / "automations").resolve(),
    )
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise DashboardInputError(f"backfill_history_path_outside_allowlist:{path}")
    try:
        metadata = resolved.lstat()
    except FileNotFoundError as exc:
        raise DashboardInputError(f"backfill_history_missing:{path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DashboardInputError(f"backfill_history_not_regular_non_symlink:{path}")
    first = resolved.read_bytes()
    second = resolved.read_bytes()
    if first != second:
        raise DashboardInputError(f"backfill_history_unstable:{path}")
    return first, sha256_bytes(first)


def _validate_transaction_backfill_provenance(
    record_dir: Path,
    project_root: Path,
    source_record: str | None,
    *,
    verify_history_source: bool = True,
) -> tuple[StableArtifact, set[tuple[str, str]]]:
    """Validate a post-hoc record inventory and its exact history source."""

    artifact = stable_read(record_dir / "backfill_provenance.json", project_root)
    payload = load_json(artifact)
    required = {
        "schema_version",
        "record",
        "backfilled_at",
        "source_record",
        "history_path",
        "history_sha256",
        "history_section",
        "record_inventory_sha256_before_provenance",
        "record_files_before_provenance",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise DashboardInputError("backfill_provenance_shape_invalid")
    if (
        payload.get("schema_version") != TRANSACTION_BACKFILL_PROVENANCE_SCHEMA_VERSION
        or payload.get("record") != record_dir.name
        or payload.get("source_record") != source_record
    ):
        raise DashboardInputError("backfill_provenance_identity_invalid")
    try:
        datetime.fromisoformat(str(payload.get("backfilled_at")))
    except ValueError as exc:
        raise DashboardInputError("backfill_provenance_timestamp_invalid") from exc
    for key in (
        "history_sha256",
        "record_inventory_sha256_before_provenance",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise DashboardInputError(f"backfill_provenance_{key}_invalid")
    history_path = payload.get("history_path")
    if not isinstance(history_path, str) or not Path(history_path).is_absolute():
        raise DashboardInputError("backfill_history_path_invalid")
    history_section = payload.get("history_section")
    if not isinstance(history_section, str) or not history_section.strip():
        raise DashboardInputError("backfill_history_section_missing")
    if verify_history_source:
        history_bytes, history_sha = _stable_external_read(Path(history_path), project_root)
        if history_sha != payload["history_sha256"]:
            raise DashboardInputError("backfill_history_sha_mismatch")
        try:
            history_text = history_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DashboardInputError("backfill_history_unreadable") from exc
        if history_section not in history_text:
            raise DashboardInputError("backfill_history_section_missing")

    rows = payload.get("record_files_before_provenance")
    if not isinstance(rows, list) or not rows:
        raise DashboardInputError("backfill_inventory_missing")
    identities: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise DashboardInputError("backfill_inventory_row_invalid")
        relative_path = row.get("relative_path")
        declared_sha = row.get("sha256")
        size_bytes = row.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not isinstance(declared_sha, str)
            or not SHA256_RE.fullmatch(declared_sha)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise DashboardInputError("backfill_inventory_row_invalid")
        source_path = _safe_same_record_path(record_dir, relative_path, "backfill_inventory_ref")
        source = stable_read(source_path, project_root)
        if len(source.data) != size_bytes or source.sha256 != declared_sha:
            raise DashboardInputError("backfill_inventory_readback_mismatch:" + relative_path)
        identity = (source.relative_path, source.sha256)
        if identity in identities:
            raise DashboardInputError("backfill_inventory_ref_duplicate")
        identities.add(identity)
    return artifact, identities


def load_json(artifact: StableArtifact) -> Any:
    try:
        return json.loads(artifact.data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DashboardInputError(f"json_unreadable:{artifact.relative_path}") from exc


def _safe_same_record_path(record_dir: Path, declared: Any, label: str) -> Path:
    if not isinstance(declared, str) or not declared.strip():
        raise DashboardInputError(f"{label}_missing")
    candidate = Path(declared)
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = (record_dir / candidate).resolve()
    record_resolved = record_dir.resolve()
    if candidate.parent != record_resolved:
        raise DashboardInputError(f"{label}_outside_record:{declared}")
    return candidate


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DashboardInputError(f"{label}_not_numeric") from exc
    if not math.isfinite(result):
        raise DashboardInputError(f"{label}_not_finite")
    return result


def _almost_equal(left: Any, right: Any, *, tolerance: float = 0.01) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_owner_corrected_initial_capital_path(
    points: list[dict[str, Any]],
) -> bool:
    """Recognize a canonical series that still follows the fixed-capital path.

    The registered performance store preserves the erroneous funding and its
    owner correction as provenance. The Dashboard may collapse those offsetting
    events only when every point independently proves the owner-corrected
    economic identity and the final external-flow balance is zero. A genuine
    flow-neutral series will normally fail the raw-minus-flow identity after
    contributed capital earns a return and must keep its canonical method.
    """

    if not points:
        return False
    try:
        initial_capital = float(points[0]["performance_initial_capital"])
    except (KeyError, TypeError, ValueError):
        return False
    if not _almost_equal(initial_capital, 1_000_000.0):
        return False
    if not _almost_equal(points[-1].get("excluded_external_flow"), 0.0):
        return False
    return all(
        _almost_equal(point.get("performance_initial_capital"), initial_capital)
        and _almost_equal(
            point.get("adjusted_total_value"),
            float(point["total_value"]) - float(point["excluded_external_flow"]),
        )
        and _almost_equal(
            point.get("adjusted_total_value"),
            float(point["unit_nav"]) * initial_capital,
        )
        for point in points
    )


def _csv_rows(artifact: StableArtifact) -> list[dict[str, str]]:
    try:
        text = artifact.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DashboardInputError(f"csv_unreadable:{artifact.relative_path}") from exc
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise DashboardInputError(f"csv_empty:{artifact.relative_path}")
    return rows


def _ledger_rows(artifact: StableArtifact) -> list[dict[str, Any]]:
    if artifact.path.suffix == ".csv":
        return _csv_rows(artifact)
    if artifact.path.suffix != ".parquet":
        raise DashboardInputError(f"effective_ledger_format_invalid:{artifact.relative_path}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(pa.BufferReader(artifact.data))
        rows = table.to_pylist()
    except Exception as exc:
        raise DashboardInputError(f"parquet_unreadable:{artifact.relative_path}") from exc
    if not rows:
        raise DashboardInputError(f"parquet_empty:{artifact.relative_path}")
    return rows


def _execution_kind(status_value: Any) -> str:
    status_text = str(status_value or "").strip().lower()
    if not status_text:
        raise DashboardInputError("manual_manifest_status_missing")
    if "no_action" in status_text or "carry_forward" in status_text:
        return "carry_forward"
    if any(token in status_text for token in ("filled", "success", "applied", "executed")):
        if any(token in status_text for token in ("pending", "rejected")):
            raise DashboardInputError("manual_manifest_status_ambiguous")
        return "applied_effective_ledger"
    raise DashboardInputError(f"manual_manifest_status_not_effective:{status_text}")


def _aware_publication_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise DashboardInputError(f"{label}_missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DashboardInputError(f"{label}_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DashboardInputError(f"{label}_timezone_missing")
    return parsed


def _validate_publication_timing(
    *, manifest: dict[str, Any], manual: dict[str, Any], record_dir: Path
) -> None:
    manifest_class = manifest.get("publication_class")
    manual_class = manual.get("publication_class")
    manifest_delay = manifest.get("publication_delay")
    manual_delay = manual.get("publication_delay")
    if (
        manifest_class is None
        and manual_class is None
        and manifest_delay is None
        and manual_delay is None
    ):
        return
    if manifest_class != manual_class:
        raise DashboardInputError("publication_class_readback_mismatch")
    if manifest.get("recorded_at_iso") != manual.get("recorded_at_iso"):
        raise DashboardInputError("publication_recorded_at_readback_mismatch")
    recorded = _aware_publication_timestamp(
        manual.get("recorded_at_iso"), label="publication_recorded_at"
    )
    recorded_shanghai = recorded.astimezone(ZoneInfo("Asia/Shanghai"))
    if record_dir.name != recorded_shanghai.strftime("%Y%m%d_%H%M"):
        raise DashboardInputError("publication_record_id_minute_mismatch")
    trade_date = _record_date(
        manual.get("valuation_trade_date") or manual.get("trade_date"),
        "publication_valuation_trade_date",
    )
    if manifest_class == ORDINARY_PUBLICATION_CLASS:
        if manifest_delay is not None or manual_delay is not None:
            raise DashboardInputError("ordinary_publication_delay_not_allowed")
        if recorded_shanghai.date().isoformat() != trade_date:
            raise DashboardInputError("ordinary_publication_must_be_same_day")
        return
    if manifest_class != LATE_PUBLICATION_CLASS:
        raise DashboardInputError("publication_class_invalid")
    if not isinstance(manifest_delay, dict) or manifest_delay != manual_delay:
        raise DashboardInputError("publication_delay_readback_mismatch")
    expected_keys = {
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
        "delay_days",
        "historical_holdings_storage_authority",
        "v17_mainline_authority",
        "broker_order_trade_authority",
    }
    if set(manifest_delay) != expected_keys:
        raise DashboardInputError("publication_delay_shape_invalid")
    if (
        manifest_delay.get("schema_id") != LATE_PUBLICATION_SCHEMA
        or manifest_delay.get("publication_class") != LATE_PUBLICATION_CLASS
        or manifest_delay.get("expected_valuation_date") != "2026-08-21"
        or manifest_delay.get("evidence_date") != "2026-08-21"
        or manifest_delay.get("expected_publication_date") != "2026-08-22"
        or manifest_delay.get("source_record") != LATE_SOURCE_RECORD
        or manifest_delay.get("publication_delay_reason") != LATE_PUBLICATION_REASON
        or manifest_delay.get("delay_days") != 1
        or manifest_delay.get("historical_holdings_storage_authority") is not True
        or manifest_delay.get("v17_mainline_authority") is not False
        or manifest_delay.get("broker_order_trade_authority") is not False
        or manifest.get("historical_holdings_storage_authority") is not True
        or manifest.get("v17_mainline_authority") is not False
        or manifest.get("broker_order_trade_authority") is not False
        or manual.get("historical_holdings_storage_authority") is not True
        or manual.get("v17_mainline_authority") is not False
        or manual.get("broker_order_trade_authority") is not False
    ):
        raise DashboardInputError("publication_delay_contract_invalid")
    if (
        trade_date != manifest_delay["expected_valuation_date"]
        or manifest.get("source_record") != manifest_delay["source_record"]
        or manual.get("source_record") != manifest_delay["source_record"]
        or recorded_shanghai.date().isoformat() != manifest_delay["expected_publication_date"]
        or manifest_delay.get("recorded_at_iso") != manual.get("recorded_at_iso")
    ):
        raise DashboardInputError("publication_delay_identity_mismatch")
    for key in (
        "continuity_receipt_id",
        "continuity_receipt_sha256",
        "continuity_receipt_created_at",
        "continuity_checkpoint_digest",
    ):
        if manifest_delay.get(key) != manual.get(key) or manifest_delay.get(key) != manifest.get(
            key
        ):
            raise DashboardInputError("publication_delay_receipt_binding_mismatch")
    receipt_at = _aware_publication_timestamp(
        manifest_delay["continuity_receipt_created_at"],
        label="publication_receipt_created_at",
    )
    if receipt_at > recorded:
        raise DashboardInputError("publication_delay_receipt_after_record")
    if (
        date.fromisoformat(manifest_delay["expected_publication_date"])
        - date.fromisoformat(manifest_delay["expected_valuation_date"])
    ).days != 1:
        raise DashboardInputError("publication_delay_not_exactly_one_day")
    if any(key in manifest or key in manual for key in ("sealed_at", "published_at")):
        raise DashboardInputError("publication_delay_seal_claim_not_allowed")


def _record_date(value: Any, label: str) -> str:
    text = str(value or "")
    try:
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    except ValueError as exc:
        raise DashboardInputError(f"{label}_invalid:{text}") from exc


def _validate_funding(
    manual: dict[str, Any], record_dir: Path, project_root: Path
) -> tuple[dict[str, Any] | None, list[StableArtifact]]:
    embedded = manual.get("manual_funding_supplement")
    if embedded in (None, {}):
        return None, []
    if not isinstance(embedded, dict):
        raise DashboardInputError("manual_funding_supplement_invalid")
    required = {
        "amount",
        "cash_before",
        "cash_after",
        "total_value_before",
        "total_value_after",
        "record_id",
        "schema_version",
        "status",
    }
    if not required.issubset(embedded):
        raise DashboardInputError("manual_funding_supplement_incomplete")
    if embedded["record_id"] != record_dir.name:
        raise DashboardInputError("manual_funding_record_id_mismatch")
    if embedded["schema_version"] != "cn_aggressive_manual_funding_supplement.v1":
        raise DashboardInputError("manual_funding_schema_invalid")
    if embedded["status"] != "local_manual_funding_recorded_no_broker_api":
        raise DashboardInputError("manual_funding_status_invalid")
    amount = _number(embedded["amount"], "manual_funding_amount")
    if amount == 0:
        raise DashboardInputError("manual_funding_amount_zero")
    if not _almost_equal(
        _number(embedded["total_value_before"], "manual_funding_total_before") + amount,
        embedded["total_value_after"],
    ):
        raise DashboardInputError("manual_funding_total_reconciliation_failed")
    path_value = manual.get("manual_funding_supplement_path")
    if not path_value:
        raise DashboardInputError("manual_funding_supplement_path_missing")
    supplement_path = _safe_same_record_path(record_dir, path_value, "manual_funding_supplement")
    supplement_artifact = stable_read(supplement_path, project_root)
    if load_json(supplement_artifact) != embedded:
        raise DashboardInputError("manual_funding_supplement_readback_mismatch")
    result = {
        "amount": amount,
        "total_value_before": _number(embedded["total_value_before"], "funding_total_before"),
        "total_value_after": _number(embedded["total_value_after"], "funding_total_after"),
        "evidence_path": supplement_artifact.relative_path,
        "evidence_sha256": supplement_artifact.sha256,
    }
    return result, [supplement_artifact]


def _validate_funding_correction(manual: dict[str, Any], record_dir: Path) -> dict[str, Any] | None:
    """Validate an explicit owner correction that reverses false funding.

    A correction changes the accounting interpretation of an already archived
    funding supplement; it is not a trade and must never change position
    quantities.  Repeated superseding records may carry the same correction,
    so the performance builder applies it once per reversed record.
    """

    correction_type = manual.get("correction_type")
    if correction_type in (None, ""):
        return None
    if correction_type != "reverse_erroneous_20260709_manual_funding":
        raise DashboardInputError("funding_correction_type_invalid")
    owner = manual.get("owner_correction")
    if not isinstance(owner, dict):
        raise DashboardInputError("funding_owner_correction_missing")
    required = {
        "declared_on",
        "declaration",
        "initial_capital_cny",
        "erroneous_funding_amount_reversed_cny",
        "erroneous_funding_record",
        "position_events_after_20260709_preserved",
    }
    if not required.issubset(owner):
        raise DashboardInputError("funding_owner_correction_incomplete")
    reversed_record = str(owner["erroneous_funding_record"])
    if not RECORD_NAME_RE.fullmatch(reversed_record):
        raise DashboardInputError("funding_correction_record_invalid")
    initial_capital = _number(owner["initial_capital_cny"], "funding_correction_initial_capital")
    reversed_amount = _number(
        owner["erroneous_funding_amount_reversed_cny"],
        "funding_correction_reversed_amount",
    )
    if initial_capital <= 0 or reversed_amount <= 0:
        raise DashboardInputError("funding_correction_amount_invalid")
    if owner["position_events_after_20260709_preserved"] is not True:
        raise DashboardInputError("funding_correction_position_proof_missing")
    if manual.get("no_trade_performed") is not True:
        raise DashboardInputError("funding_correction_no_trade_proof_missing")
    capital_cny = _number(manual.get("capital_cny"), "funding_correction_manual_capital")
    if not _almost_equal(capital_cny, initial_capital):
        raise DashboardInputError("funding_correction_capital_mismatch")
    return {
        "reversed_record": reversed_record,
        "reversed_amount": reversed_amount,
        "initial_capital": initial_capital,
        "correction_record": record_dir.name,
    }


def validate_record(record_dir: Path, record_root: Path, project_root: Path) -> dict[str, Any]:
    """Validate one timestamped record and return its closed snapshot."""

    if not RECORD_NAME_RE.fullmatch(record_dir.name):
        raise DashboardInputError("record_name_invalid")
    manifest_artifact = stable_read(record_dir / "manifest.json", project_root)
    manifest = load_json(manifest_artifact)
    if not isinstance(manifest, dict):
        raise DashboardInputError("manifest_not_object")
    if manifest.get("market") != MARKET:
        raise DashboardInputError("manifest_market_mismatch")
    if manifest.get("strategy") != STRATEGY:
        raise DashboardInputError("manifest_strategy_mismatch")
    if manifest.get("timestamp") != record_dir.name:
        raise DashboardInputError("manifest_timestamp_mismatch")

    source_record = manifest.get("source_record")
    if source_record:
        if not isinstance(source_record, str) or not RECORD_NAME_RE.fullmatch(source_record):
            raise DashboardInputError("manifest_source_record_invalid")
        source_dir = record_root / source_record
        if not source_dir.is_dir() or source_dir.is_symlink():
            raise DashboardInputError("manifest_source_record_missing")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DashboardInputError("manifest_file_refs_missing")
    manual_path = _safe_same_record_path(
        record_dir,
        files.get("manual_execution_manifest"),
        "manifest_manual_ref",
    )
    manual_artifact = stable_read(manual_path, project_root)
    manual = load_json(manual_artifact)
    if not isinstance(manual, dict):
        raise DashboardInputError("manual_manifest_not_object")
    if manual.get("record_timestamp") != record_dir.name:
        raise DashboardInputError("manual_manifest_timestamp_mismatch")
    embedded_manual = manifest.get("manual_execution")
    backfill_provenance_required = embedded_manual is None
    if embedded_manual is not None and embedded_manual != manual:
        raise DashboardInputError("manifest_manual_execution_readback_mismatch")
    _validate_publication_timing(
        manifest=manifest,
        manual=manual,
        record_dir=record_dir,
    )
    execution_kind = _execution_kind(manual.get("status") or manual.get("execution_status"))
    if manual.get("no_broker_api_called") is not True:
        raise DashboardInputError("manual_manifest_no_broker_proof_missing")

    ledger_declared = manual.get("effective_manual_ledger_path") or manual.get("next_ledger_path")
    ledger_path = _safe_same_record_path(record_dir, ledger_declared, "effective_ledger")
    if ledger_path.name not in {
        "ledger_after_manual_switch.csv",
        "ledger_after_manual_switch.parquet",
    }:
        raise DashboardInputError("effective_ledger_name_invalid")
    ledger_artifact = stable_read(ledger_path, project_root)
    declared_ledger_sha = (
        manual.get("next_ledger_sha256")
        or manual.get(
            "ledger_after_manual_switch_parquet_sha256"
            if ledger_path.suffix == ".parquet"
            else "ledger_after_manual_switch_csv_sha256"
        )
        or (manual.get("ledger_provenance") or {}).get("declared_sha256")
    )
    if not isinstance(declared_ledger_sha, str) or not SHA256_RE.fullmatch(declared_ledger_sha):
        raise DashboardInputError("effective_ledger_sha_not_declared")
    if declared_ledger_sha != ledger_artifact.sha256:
        raise DashboardInputError("effective_ledger_sha_mismatch")
    provenance = manual.get("ledger_provenance")
    if not isinstance(provenance, dict):
        raise DashboardInputError("ledger_provenance_missing")
    if not all(
        provenance.get(key) is True
        for key in (
            "contained_in_run_directory",
            "regular_non_symlink_file",
            "stable_double_read",
        )
    ):
        raise DashboardInputError("ledger_provenance_flags_invalid")
    if provenance.get("declared_sha256") != ledger_artifact.sha256:
        raise DashboardInputError("ledger_provenance_sha_mismatch")

    parquet_name = manual.get("ledger_after_manual_switch_parquet")
    parquet_artifact: StableArtifact | None = None
    if parquet_name:
        parquet_path = _safe_same_record_path(record_dir, parquet_name, "ledger_parquet")
        parquet_artifact = stable_read(parquet_path, project_root)
        declared_parquet_sha = manual.get("ledger_after_manual_switch_parquet_sha256")
        if declared_parquet_sha != parquet_artifact.sha256:
            raise DashboardInputError("ledger_parquet_sha_mismatch")

    rows = _ledger_rows(ledger_artifact)
    if not REQUIRED_LEDGER_COLUMNS.issubset(rows[0]):
        raise DashboardInputError("effective_ledger_columns_missing")
    positions: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        symbol = str(row.get("symbol") or "").strip()
        name = str(row.get("name") or "").strip()
        if not SYMBOL_RE.fullmatch(symbol) or not name or symbol in seen_symbols:
            raise DashboardInputError(f"effective_ledger_identity_invalid:{row_number}")
        seen_symbols.add(symbol)
        shares = _number(row.get("shares"), f"shares:{symbol}")
        if shares <= 0:
            raise DashboardInputError(f"shares_not_positive:{symbol}")
        position = {
            "symbol": symbol,
            "name": name,
            "shares": shares,
            "avg_cost": _number(row.get("avg_cost"), f"avg_cost:{symbol}"),
            "cost_basis": _number(row.get("cost_basis"), f"cost_basis:{symbol}"),
            "recorded_price": _number(row.get("current_price"), f"recorded_price:{symbol}"),
            "market_value": _number(row.get("current_value"), f"market_value:{symbol}"),
            "unrealized_pnl": _number(row.get("unrealized_pnl"), f"unrealized_pnl:{symbol}"),
            "realized_pnl": None,
            "nav_weight": _number(row.get("nav_weight"), f"nav_weight:{symbol}"),
            "equity_weight": _number(row.get("equity_sleeve_weight"), f"equity_weight:{symbol}"),
            "thesis_status": str(row.get("thesis_status") or "UNKNOWN").strip() or "UNKNOWN",
        }
        positions.append(position)
    positions.sort(key=lambda item: item["symbol"])
    expected_count = manual.get("effective_manual_holding_count")
    if expected_count is None or int(expected_count) != len(positions):
        raise DashboardInputError("effective_holding_count_mismatch")

    pnl_path = _safe_same_record_path(record_dir, files.get("pnl_summary"), "manifest_pnl_ref")
    pnl_artifact = stable_read(pnl_path, project_root)
    pnl_rows = _csv_rows(pnl_artifact)
    pnl = pnl_rows[-1]
    accounting: dict[str, float] = {}
    for field in ACCOUNTING_FIELDS:
        if field not in manual or field not in pnl:
            raise DashboardInputError(f"accounting_field_missing:{field}")
        manual_value = _number(manual[field], f"manual_{field}")
        if not _almost_equal(manual_value, pnl[field]):
            raise DashboardInputError(f"pnl_manual_reconciliation_failed:{field}")
        accounting[field] = manual_value
    if not _almost_equal(
        accounting["cash_after"] + accounting["market_value_after"],
        accounting["total_value_after"],
    ):
        raise DashboardInputError("accounting_cash_market_total_mismatch")
    ledger_market_value = sum(item["market_value"] for item in positions)
    if not _almost_equal(ledger_market_value, accounting["market_value_after"]):
        raise DashboardInputError("ledger_market_value_accounting_mismatch")
    if not isinstance(manual.get("financial_state_sha256"), str) or not SHA256_RE.fullmatch(
        manual["financial_state_sha256"]
    ):
        raise DashboardInputError("financial_state_sha_missing_or_invalid")

    backfill_provenance_artifact: StableArtifact | None = None
    if backfill_provenance_required:
        backfill_provenance_artifact, inventory = _validate_transaction_backfill_provenance(
            record_dir, project_root, source_record
        )
        required_inventory = {
            (artifact.relative_path, artifact.sha256)
            for artifact in (
                manifest_artifact,
                manual_artifact,
                ledger_artifact,
                pnl_artifact,
            )
        }
        if not required_inventory.issubset(inventory):
            raise DashboardInputError("backfill_inventory_required_closure_missing")

    data_snapshot = manifest.get("data_snapshot")
    if not isinstance(data_snapshot, dict):
        raise DashboardInputError("manifest_data_snapshot_missing")
    if backfill_provenance_required:
        if (
            data_snapshot.get("freshness_mode") != "strict_parquet_canonical_historical_revaluation"
            or not re.fullmatch(
                r"^[0-9]{8}T[0-9]{6}Z$",
                str(data_snapshot.get("snapshot_id_at_backfill") or ""),
            )
            or not SHA256_RE.fullmatch(
                str(data_snapshot.get("market_pointer_sha256_at_backfill") or "")
            )
        ):
            raise DashboardInputError("backfill_data_snapshot_provenance_invalid")
        data_date = _record_date(
            data_snapshot.get("valuation_trade_date"),
            "valuation_trade_date",
        )
    else:
        data_date = _record_date(
            data_snapshot.get("analysis_trade_date"),
            "analysis_trade_date",
        )
    valuation_status = data_snapshot.get("valuation_status")
    manual_valuation_status = manual.get("valuation_status")
    if (
        valuation_status not in (None, "")
        and manual_valuation_status not in (None, "")
        and valuation_status != manual_valuation_status
    ):
        raise DashboardInputError("valuation_status_readback_mismatch")
    valuation_status = str(manual_valuation_status or valuation_status or "").strip() or None
    if manual.get("official_valuation") is False:
        fallback_value = data_snapshot.get("last_strict_completed_trade_date_for_untouched_marks")
        fallback_price_date = (
            _record_date(fallback_value, "last_strict_completed_trade_date")
            if fallback_value not in (None, "")
            else None
        )
        owner_trade_dates: dict[str, str] = {}
        for trade in manual.get("applied_owner_declared_trades") or []:
            if not isinstance(trade, dict):
                raise DashboardInputError("owner_declared_trade_invalid")
            symbol = str(trade.get("symbol") or "")
            if not SYMBOL_RE.fullmatch(symbol) or symbol not in seen_symbols:
                raise DashboardInputError("owner_declared_trade_symbol_invalid:" + symbol)
            trade_date = _record_date(trade.get("trade_date"), "owner_declared_trade_date")
            if symbol in owner_trade_dates and owner_trade_dates[symbol] != trade_date:
                raise DashboardInputError("owner_declared_trade_date_ambiguous:" + symbol)
            owner_trade_dates[symbol] = trade_date
        for position in positions:
            position_symbol = position.get("symbol")
            if not isinstance(position_symbol, str):
                raise DashboardInputError("position_symbol_invalid")
            position["price_date"] = owner_trade_dates.get(position_symbol, fallback_price_date)
    valuation_evidence_artifact: StableArtifact | None = None
    if manual.get("official_valuation") is True:
        if manual.get("valuation_completeness_passed") is not True:
            raise DashboardInputError("official_valuation_completeness_missing")
        if str(valuation_status or "").startswith("BLOCKED"):
            raise DashboardInputError("official_valuation_status_blocked")
        evidence_path = _safe_same_record_path(
            record_dir,
            files.get("valuation_evidence"),
            "official_valuation_evidence",
        )
        valuation_evidence_artifact = stable_read(evidence_path, project_root)
        if (
            manual.get("valuation_evidence_sha256") != valuation_evidence_artifact.sha256
            or data_snapshot.get("valuation_evidence_sha256") != valuation_evidence_artifact.sha256
        ):
            raise DashboardInputError("official_valuation_evidence_sha_mismatch")
        evidence = load_json(valuation_evidence_artifact)
        if not isinstance(evidence, dict):
            raise DashboardInputError("official_valuation_evidence_contract_invalid")
        evidence_schema = evidence.get("schema_version")
        if evidence_schema == STRICT_MARKET_CLOSE_EVIDENCE_SCHEMA_VERSION:
            # The current producer binds local strict-Parquet evidence.  The
            # older provider-shaped contract remains readable for historical
            # records, but may not be emitted by the new offline path.
            evidence_text = json.dumps(evidence, ensure_ascii=False, sort_keys=True).lower()
            if any(
                token in evidence_text
                for token in ("tushare", "provider", "stock_api", "index_api")
            ):
                raise DashboardInputError("official_valuation_evidence_provider_claim")
            if (
                evidence.get("market") != MARKET
                or _record_date(
                    evidence.get("trade_date"),
                    "official_valuation_evidence_trade_date",
                )
                != data_date
            ):
                raise DashboardInputError("official_valuation_evidence_contract_invalid")
            strict_stock_rows = evidence.get("stocks")
            strict_index_rows = evidence.get("indices")
            if not isinstance(strict_stock_rows, list) or not isinstance(strict_index_rows, list):
                raise DashboardInputError("official_valuation_evidence_rows_invalid")
        elif (
            evidence_schema != "cn_dashboard_tushare_close_evidence.v1"
            or evidence.get("provider") != "tushare.pro"
            or evidence.get("stock_api") != "daily"
            or evidence.get("index_api") != "index_daily"
            or evidence.get("coverage") != "exact_close"
            or evidence.get("previous_trading_day_ffill") is not False
            or _record_date(
                evidence.get("trade_date"),
                "official_valuation_evidence_trade_date",
            )
            != data_date
        ):
            raise DashboardInputError("official_valuation_evidence_contract_invalid")
        stock_rows = evidence.get("stocks")
        index_rows = evidence.get("indices")
        if not isinstance(stock_rows, list) or not isinstance(index_rows, list):
            raise DashboardInputError("official_valuation_evidence_rows_invalid")
        symbol_key = (
            "symbol"
            if evidence_schema == STRICT_MARKET_CLOSE_EVIDENCE_SCHEMA_VERSION
            else "ts_code"
        )
        stocks = {str(row.get(symbol_key) or row.get("ts_code")): row for row in stock_rows}
        indices = {str(row.get("ts_code") or row.get("symbol")): row for row in index_rows}
        if set(stocks) != seen_symbols or set(indices) != {
            "000300.SH",
            "000688.SH",
            "399006.SZ",
        }:
            raise DashboardInputError("official_valuation_evidence_coverage_invalid")
        for position in positions:
            position_symbol = position.get("symbol")
            if not isinstance(position_symbol, str):
                raise DashboardInputError("position_symbol_invalid")
            row = stocks[position_symbol]
            if _record_date(
                row.get("trade_date"),
                "official_valuation_stock_trade_date",
            ) != data_date or not _almost_equal(row.get("close"), position["recorded_price"]):
                raise DashboardInputError(
                    "official_valuation_stock_close_mismatch:" + position_symbol
                )
            if evidence_schema == STRICT_MARKET_CLOSE_EVIDENCE_SCHEMA_VERSION:
                path_value = row.get("serving_parquet_path") or row.get("serving_path")
                sha_value = row.get("serving_parquet_sha256") or row.get("parquet_sha256")
                if (
                    not isinstance(path_value, str)
                    or path_value.startswith("/")
                    or ".." in Path(path_value).parts
                    or not isinstance(sha_value, str)
                    or not SHA256_RE.fullmatch(sha_value)
                ):
                    raise DashboardInputError(
                        "official_valuation_stock_source_ref_invalid:" + position_symbol
                    )
                stock_path = project_root / path_value
                stock_artifact = stable_read(stock_path, project_root)
                if stock_artifact.sha256 != sha_value:
                    raise DashboardInputError(
                        "official_valuation_stock_source_sha_mismatch:" + position_symbol
                    )
            position["price_date"] = data_date
        for code, row in indices.items():
            if (
                _record_date(
                    row.get("trade_date"),
                    "official_valuation_index_trade_date",
                )
                != data_date
                or _number(row.get("close"), f"index_close:{code}") <= 0
            ):
                raise DashboardInputError("official_valuation_index_close_invalid:" + code)
            if evidence_schema == STRICT_MARKET_CLOSE_EVIDENCE_SCHEMA_VERSION:
                benchmark_path = evidence.get("benchmark_input_path") or row.get(
                    "benchmark_input_path"
                )
                benchmark_sha = evidence.get("benchmark_input_sha256") or row.get(
                    "benchmark_input_sha256"
                )
                if (
                    not isinstance(benchmark_path, str)
                    or benchmark_path.startswith("/")
                    or ".." in Path(benchmark_path).parts
                    or not isinstance(benchmark_sha, str)
                    or not SHA256_RE.fullmatch(benchmark_sha)
                ):
                    raise DashboardInputError("official_valuation_benchmark_source_ref_invalid")
                benchmark_artifact = stable_read(project_root / benchmark_path, project_root)
                if benchmark_artifact.sha256 != benchmark_sha:
                    raise DashboardInputError("official_valuation_benchmark_source_sha_mismatch")
        if source_record:
            source_dir = record_root / source_record
            source_manifest = stable_read(source_dir / "manifest.json", project_root)
            source_manual = stable_read(source_dir / "manual_execution_manifest.json", project_root)
            source_manual_value = load_json(source_manual)
            source_ledger = stable_read(
                source_dir / str(source_manual_value.get("effective_manual_ledger_path")),
                project_root,
            )
            if (
                manifest.get("source_manifest_sha256") != source_manifest.sha256
                or manual.get("source_manifest_sha256") != source_manifest.sha256
                or manual.get("source_manual_manifest_sha256") != source_manual.sha256
                or manual.get("source_contained_ledger_sha256") != source_ledger.sha256
            ):
                raise DashboardInputError("official_valuation_source_closure_sha_mismatch")

    funding, funding_artifacts = _validate_funding(manual, record_dir, project_root)
    funding_correction = _validate_funding_correction(manual, record_dir)

    artifacts = [
        manifest_artifact,
        manual_artifact,
        ledger_artifact,
        pnl_artifact,
    ]
    if parquet_artifact is not None:
        artifacts.append(parquet_artifact)
    if backfill_provenance_artifact is not None:
        artifacts.append(backfill_provenance_artifact)
    if valuation_evidence_artifact is not None:
        artifacts.append(valuation_evidence_artifact)
    artifacts.extend(funding_artifacts)
    source_refs = [
        {"path": artifact.relative_path, "sha256": artifact.sha256} for artifact in artifacts
    ]
    return {
        "record": record_dir.name,
        "recorded_at": manifest.get("recorded_at") or manual.get("recorded_at"),
        "source_record": source_record,
        "data_date": data_date,
        "execution_status": str(manual.get("status") or manual.get("execution_status")),
        "execution_kind": execution_kind,
        "publication_class": manifest.get("publication_class"),
        "publication_delay": manifest.get("publication_delay"),
        "valuation_status": valuation_status,
        "official_valuation": manual.get("official_valuation"),
        "valuation_completeness_passed": manual.get("valuation_completeness_passed"),
        "price_basis": manual.get("price_basis"),
        "manifest_path": manifest_artifact.relative_path,
        "manifest_sha256": manifest_artifact.sha256,
        "manual_manifest_path": manual_artifact.relative_path,
        "manual_manifest_sha256": manual_artifact.sha256,
        "ledger_path": ledger_artifact.relative_path,
        "ledger_sha256": ledger_artifact.sha256,
        "pnl_path": pnl_artifact.relative_path,
        "pnl_sha256": pnl_artifact.sha256,
        "financial_state_sha256": manual["financial_state_sha256"],
        "positions": positions,
        "accounting": accounting,
        "funding": funding,
        "funding_correction": funding_correction,
        "source_refs": source_refs,
    }


def scan_valid_records(
    record_root: Path, project_root: Path
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    if not record_root.is_dir() or record_root.is_symlink():
        raise DashboardInputError(f"record_root_missing_or_invalid:{record_root}")
    valid: list[dict[str, Any]] = []
    warnings: list[str] = []
    candidates = sorted(
        path
        for path in record_root.iterdir()
        if path.is_dir() and not path.is_symlink() and RECORD_NAME_RE.fullmatch(path.name)
    )
    latest_seen = candidates[-1].name if candidates else None
    for record_dir in candidates:
        try:
            valid.append(validate_record(record_dir, record_root, project_root))
        except DashboardInputError as exc:
            warnings.append(f"{record_dir.name}:{exc}")
    return valid, warnings, latest_seen


def _validate_historical_manifest(
    record_dir: Path, record_root: Path, project_root: Path
) -> tuple[dict[str, Any], StableArtifact, str | None]:
    manifest_artifact = stable_read(record_dir / "manifest.json", project_root)
    manifest = load_json(manifest_artifact)
    if not isinstance(manifest, dict):
        raise DashboardInputError("historical_manifest_not_object")
    if manifest.get("market") != MARKET:
        raise DashboardInputError("historical_manifest_market_mismatch")
    if manifest.get("strategy") != STRATEGY:
        raise DashboardInputError("historical_manifest_strategy_mismatch")
    if manifest.get("timestamp") != record_dir.name:
        raise DashboardInputError("historical_manifest_timestamp_mismatch")
    source_record = manifest.get("source_record")
    if source_record:
        if not isinstance(source_record, str) or not RECORD_NAME_RE.fullmatch(source_record):
            raise DashboardInputError("historical_source_record_invalid")
        source_dir = record_root / source_record
        if not source_dir.is_dir() or source_dir.is_symlink():
            raise DashboardInputError("historical_source_record_missing")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DashboardInputError("historical_manifest_file_refs_missing")
    return manifest, manifest_artifact, source_record


def _historical_valuation_date(
    *,
    record_dir: Path,
    manifest: dict[str, Any],
    pnl: dict[str, Any] | None,
    strict_record: dict[str, Any] | None,
    backfill_provenance_validated: bool = False,
) -> str:
    quote_snapshot = str((pnl or {}).get("quote_snapshot") or "")
    quote_match = re.match(r"^(20[0-9]{6})", quote_snapshot)
    if quote_match:
        return _record_date(quote_match.group(1), "historical_quote_date")
    intraday_snapshot = str(
        (manifest.get("data_snapshot") or {}).get("intraday_quote_snapshot") or ""
    )
    intraday_match = re.match(r"^(20[0-9]{2})-([0-9]{2})-([0-9]{2})", intraday_snapshot)
    if intraday_match:
        return date(
            int(intraday_match.group(1)),
            int(intraday_match.group(2)),
            int(intraday_match.group(3)),
        ).isoformat()
    if strict_record is not None:
        return strict_record["data_date"]
    if backfill_provenance_validated:
        snapshot = manifest.get("data_snapshot") or {}
        if snapshot.get("freshness_mode") != "strict_parquet_canonical_historical_revaluation":
            raise DashboardInputError("historical_backfill_freshness_mode_invalid")
        return _record_date(
            snapshot.get("valuation_trade_date"),
            "historical_backfill_valuation_trade_date",
        )
    if pnl is None and manifest.get("source_record") in (None, ""):
        return _record_date(record_dir.name[:8], "historical_baseline_date")
    raise DashboardInputError("historical_valuation_date_unverified")


def _historical_funding(
    *,
    manifest: dict[str, Any],
    record_dir: Path,
    project_root: Path,
) -> tuple[dict[str, Any] | None, list[StableArtifact]]:
    embedded_manual = manifest.get("manual_execution")
    files = manifest.get("files") or {}
    manual_ref = files.get("manual_execution_manifest")
    embedded_has_funding = isinstance(embedded_manual, dict) and bool(
        embedded_manual.get("manual_funding_supplement")
    )
    if not manual_ref:
        if embedded_has_funding:
            raise DashboardInputError("historical_manual_ref_missing")
        return None, []
    manual_path = _safe_same_record_path(
        record_dir,
        manual_ref,
        "historical_manual_ref",
    )
    manual_artifact = stable_read(manual_path, project_root)
    manual = load_json(manual_artifact)
    if not isinstance(manual, dict):
        raise DashboardInputError("historical_manual_not_object")
    manual_has_funding = bool(manual.get("manual_funding_supplement"))
    if not manual_has_funding:
        if embedded_has_funding:
            raise DashboardInputError("historical_embedded_funding_missing_from_manual")
        return None, []
    if embedded_has_funding and manual != embedded_manual:
        raise DashboardInputError("historical_manual_readback_mismatch")
    if manual.get("record_timestamp") != record_dir.name:
        raise DashboardInputError("historical_manual_timestamp_mismatch")
    funding, artifacts = _validate_funding(manual, record_dir, project_root)
    assert funding is not None
    funding["binding_status"] = (
        "MANIFEST_EMBED_AND_INDEPENDENT_MANUAL_MATCH"
        if embedded_has_funding
        else "INDEPENDENT_MANUAL_AND_EXACT_SUPPLEMENT_ONLY"
    )
    return funding, [manual_artifact, *artifacts]


def _validate_historical_record(
    *,
    record_dir: Path,
    record_root: Path,
    project_root: Path,
    strict_record: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest, manifest_artifact, source_record = _validate_historical_manifest(
        record_dir, record_root, project_root
    )
    if strict_record is not None and (
        strict_record.get("official_valuation") is False
        or strict_record.get("valuation_completeness_passed") is False
        or str(strict_record.get("valuation_status") or "").startswith("BLOCKED")
    ):
        raise DashboardInputError("historical_official_valuation_incomplete")
    files = manifest["files"]
    source_artifacts = [manifest_artifact]
    ledger_artifact: StableArtifact | None = None
    ledger_ref = files.get("ledger_after_manual_switch") or files.get("ledger")
    if ledger_ref:
        ledger_path = _safe_same_record_path(record_dir, ledger_ref, "historical_ledger_ref")
        ledger_artifact = stable_read(ledger_path, project_root)
        source_artifacts.append(ledger_artifact)

    pnl_ref = files.get("pnl_summary")
    if not pnl_ref:
        if source_record not in (None, ""):
            raise DashboardInputError("historical_pnl_ref_missing")
        if ledger_artifact is None:
            raise DashboardInputError("historical_baseline_ledger_missing")
        capital = _number(manifest.get("capital_cny"), "historical_baseline_capital")
        if capital <= 0:
            raise DashboardInputError("historical_baseline_capital_not_positive")
        ledger_rows = _ledger_rows(ledger_artifact)
        if "current_value" not in ledger_rows[0]:
            raise DashboardInputError("historical_baseline_ledger_value_missing")
        market_value = sum(
            _number(row.get("current_value"), "historical_baseline_value") for row in ledger_rows
        )
        cash = capital - market_value
        if cash < -0.01:
            raise DashboardInputError("historical_baseline_negative_implied_cash")
        valuation_date = _historical_valuation_date(
            record_dir=record_dir,
            manifest=manifest,
            pnl=None,
            strict_record=strict_record,
        )
        return {
            "record": record_dir.name,
            "source_record": None,
            "valuation_date": valuation_date,
            "accounting": {
                "cash_after": max(cash, 0.0),
                "market_value_after": market_value,
                "total_value_after": capital,
                "portfolio_pnl_after": 0.0,
                "realized_pnl_from_rebalance": 0.0,
            },
            "capital_base": capital,
            "funding": None,
            "funding_correction": None,
            "evidence_status": "ARCHIVE_INCEPTION_EXACT_BYTES_NO_DECLARED_SHA",
            "manifest_path": manifest_artifact.relative_path,
            "manifest_sha256": manifest_artifact.sha256,
            "ledger_path": ledger_artifact.relative_path,
            "ledger_sha256": ledger_artifact.sha256,
            "pnl_path": None,
            "pnl_sha256": None,
            "formal_record": manifest.get("formal_record") is True,
            "completeness_passed": manifest.get("completeness_passed") is True,
            "source_refs": [
                {"path": artifact.relative_path, "sha256": artifact.sha256}
                for artifact in source_artifacts
            ],
        }

    pnl_path = _safe_same_record_path(record_dir, pnl_ref, "historical_pnl_ref")
    pnl_artifact = stable_read(pnl_path, project_root)
    pnl_rows = _csv_rows(pnl_artifact)
    pnl = pnl_rows[-1]
    source_artifacts.append(pnl_artifact)
    accounting: dict[str, float] = {}
    for field in HISTORICAL_ACCOUNTING_FIELDS:
        if field not in pnl:
            raise DashboardInputError(f"historical_accounting_field_missing:{field}")
        accounting[field] = _number(pnl[field], f"historical_{field}")
    if not _almost_equal(
        accounting["cash_after"] + accounting["market_value_after"],
        accounting["total_value_after"],
    ):
        raise DashboardInputError("historical_cash_market_total_mismatch")
    capital_value = pnl.get("initial_capital")
    if capital_value in (None, ""):
        capital_value = manifest.get("capital_cny")
    capital_base = (
        _number(capital_value, "historical_capital_base")
        if capital_value not in (None, "")
        else None
    )
    if capital_base is not None and not _almost_equal(
        accounting["total_value_after"] - capital_base,
        accounting["portfolio_pnl_after"],
    ):
        raise DashboardInputError("historical_portfolio_pnl_mismatch")
    funding, funding_artifacts = _historical_funding(
        manifest=manifest,
        record_dir=record_dir,
        project_root=project_root,
    )
    source_artifacts.extend(funding_artifacts)
    if funding and not _almost_equal(funding["total_value_after"], accounting["total_value_after"]):
        raise DashboardInputError("historical_funding_total_after_mismatch")
    backfill_provenance_validated = False
    snapshot = manifest.get("data_snapshot") or {}
    if (
        strict_record is None
        and snapshot.get("freshness_mode") == "strict_parquet_canonical_historical_revaluation"
    ):
        manual_ref = files.get("manual_execution_manifest")
        if not manual_ref:
            raise DashboardInputError("historical_backfill_manual_ref_missing")
        manual_artifact = stable_read(
            _safe_same_record_path(record_dir, manual_ref, "historical_backfill_manual_ref"),
            project_root,
        )
        if all(
            artifact.relative_path != manual_artifact.relative_path for artifact in source_artifacts
        ):
            source_artifacts.append(manual_artifact)
        provenance_artifact, inventory = _validate_transaction_backfill_provenance(
            record_dir,
            project_root,
            source_record,
            verify_history_source=False,
        )
        required_inventory = {
            (artifact.relative_path, artifact.sha256) for artifact in source_artifacts
        }
        if not required_inventory.issubset(inventory):
            raise DashboardInputError("historical_backfill_inventory_closure_missing")
        source_artifacts.append(provenance_artifact)
        backfill_provenance_validated = True
    funding_correction = (
        strict_record.get("funding_correction") if strict_record is not None else None
    )
    if funding_correction is not None:
        correction_manual = stable_read(
            project_root / strict_record["manual_manifest_path"],
            project_root,
        )
        if correction_manual.sha256 != strict_record["manual_manifest_sha256"]:
            raise DashboardInputError("historical_funding_correction_manual_sha_mismatch")
        if all(
            artifact.relative_path != correction_manual.relative_path
            for artifact in source_artifacts
        ):
            source_artifacts.append(correction_manual)
    valuation_date = _historical_valuation_date(
        record_dir=record_dir,
        manifest=manifest,
        pnl=pnl,
        strict_record=strict_record,
        backfill_provenance_validated=backfill_provenance_validated,
    )
    return {
        "record": record_dir.name,
        "source_record": source_record,
        "valuation_date": valuation_date,
        "accounting": accounting,
        "capital_base": capital_base,
        "funding": funding,
        "funding_correction": funding_correction,
        "evidence_status": (
            "HASH_BOUND_CURRENT_CLOSURE"
            if strict_record is not None
            else "LEGACY_EXACT_BYTES_NO_DECLARED_SHA"
        ),
        "manifest_path": manifest_artifact.relative_path,
        "manifest_sha256": manifest_artifact.sha256,
        "ledger_path": (ledger_artifact.relative_path if ledger_artifact else None),
        "ledger_sha256": ledger_artifact.sha256 if ledger_artifact else None,
        "pnl_path": pnl_artifact.relative_path,
        "pnl_sha256": pnl_artifact.sha256,
        "formal_record": manifest.get("formal_record") is True,
        "completeness_passed": manifest.get("completeness_passed") is True,
        "source_refs": [
            {"path": artifact.relative_path, "sha256": artifact.sha256}
            for artifact in source_artifacts
        ],
    }


def scan_historical_performance_records(
    *,
    record_root: Path,
    project_root: Path,
    strict_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    strict_by_record = {record["record"]: record for record in strict_records}
    records: list[dict[str, Any]] = []
    rejected: list[str] = []
    candidates = sorted(
        path
        for path in record_root.iterdir()
        if path.is_dir() and not path.is_symlink() and RECORD_NAME_RE.fullmatch(path.name)
    )
    for record_dir in candidates:
        try:
            records.append(
                _validate_historical_record(
                    record_dir=record_dir,
                    record_root=record_root,
                    project_root=project_root,
                    strict_record=strict_by_record.get(record_dir.name),
                )
            )
        except DashboardInputError as exc:
            rejected.append(f"{record_dir.name}:{exc}")
    validate_historical_performance_sequence(records)
    return records, rejected


def validate_historical_record(
    *,
    record_dir: Path,
    record_root: Path,
    project_root: Path,
    strict_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate one new history row without scanning archived hot paths."""

    return _validate_historical_record(
        record_dir=record_dir,
        record_root=record_root,
        project_root=project_root,
        strict_record=strict_record,
    )


def validate_historical_performance_sequence(
    records: list[dict[str, Any]],
) -> None:
    """Validate funding/capital continuity for a merged history projection."""

    if len(records) < 2:
        raise DashboardInputError("fewer_than_two_historical_performance_records")
    funding_by_record = {
        record["record"]: record["funding"]
        for record in records
        if record.get("funding") is not None
    }
    previous_capital: float | None = None
    for record in records:
        capital = record["capital_base"]
        correction = record.get("funding_correction")
        if correction is not None:
            reversed_funding = funding_by_record.get(correction["reversed_record"])
            if reversed_funding is None or not _almost_equal(
                reversed_funding["amount"], correction["reversed_amount"]
            ):
                raise DashboardInputError(
                    "historical_funding_correction_target_mismatch:" + record["record"]
                )
            if capital is None or not _almost_equal(capital, correction["initial_capital"]):
                raise DashboardInputError(
                    "historical_funding_correction_capital_mismatch:" + record["record"]
                )
        if (
            previous_capital is not None
            and capital is not None
            and not _almost_equal(previous_capital, capital)
            and record.get("funding") is None
            and correction is None
        ):
            raise DashboardInputError(
                "historical_capital_change_without_funding:" + record["record"]
            )
        if capital is not None:
            previous_capital = capital


def build_dashboard_catalog_projection(record_root: Path, project_root: Path) -> dict[str, Any]:
    """Build the catalog projection during an explicit legacy bootstrap.

    Registered Dashboard reads must never call the legacy directory scanners.
    This helper keeps their use visible and limited to migration/bootstrap and
    unregistered compatibility fixtures.
    """

    try:
        registered = load_registered_catalog(record_root)
    except StrategyRecordStoreError as exc:
        raise DashboardInputError(f"record_catalog_invalid:{exc}") from exc
    if registered is not None:
        raise DashboardInputError("legacy_dashboard_projection_bootstrap_registered_root")
    valid, rejected, latest_seen = scan_valid_records(record_root, project_root)
    historical_records, historical_rejected = scan_historical_performance_records(
        record_root=record_root,
        project_root=project_root,
        strict_records=valid,
    )

    # Legacy records can name the effective ledger through more than one
    # manifest field.  The raw scanners preserve those declarations, so the
    # same exact source may appear twice in one record.  Catalog rows use the
    # path as the identity and must be canonical before publication.
    for row in [*valid, *historical_records]:
        refs = row.get("source_refs")
        if not isinstance(refs, list):
            raise DashboardInputError("dashboard_projection_source_refs_invalid")
        by_path: dict[str, dict[str, str]] = {}
        for ref in refs:
            if (
                not isinstance(ref, dict)
                or set(ref) != {"path", "sha256"}
                or not isinstance(ref.get("path"), str)
                or not isinstance(ref.get("sha256"), str)
            ):
                raise DashboardInputError("dashboard_projection_source_ref_invalid")
            existing = by_path.get(ref["path"])
            if existing is not None and existing != ref:
                raise DashboardInputError("dashboard_projection_source_ref_conflict:" + ref["path"])
            by_path[ref["path"]] = ref
        row["source_refs"] = [by_path[path] for path in sorted(by_path)]
    return {
        "valid_records": valid,
        "rejected": rejected,
        "latest_seen": latest_seen,
        "historical_records": historical_records,
        "historical_rejected": historical_rejected,
    }


def _projection_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DashboardInputError(f"catalog_dashboard_projection_{label}_invalid")
    return list(value)


def _projection_records(
    value: Any, label: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list):
        raise DashboardInputError(f"catalog_dashboard_projection_{label}_invalid")
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict):
            raise DashboardInputError(f"catalog_dashboard_projection_{label}_invalid")
        record_id = row.get("record")
        if (
            not isinstance(record_id, str)
            or not RECORD_NAME_RE.fullmatch(record_id)
            or record_id in by_id
            or not isinstance(row.get("source_refs"), list)
        ):
            raise DashboardInputError(f"catalog_dashboard_projection_{label}_invalid")
        copied = dict(row)
        records.append(copied)
        by_id[record_id] = copied
    if list(by_id) != sorted(by_id):
        raise DashboardInputError(f"catalog_dashboard_projection_{label}_order_invalid")
    return records, by_id


def _logical_source_refs(value: Any, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise DashboardInputError(f"{label}_invalid")
    by_path: dict[str, dict[str, str]] = {}
    for ref in value:
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise DashboardInputError(f"{label}_invalid")
        relative_path = ref.get("path")
        declared_sha = ref.get("sha256")
        if (
            not isinstance(relative_path, str)
            or relative_path.startswith("/")
            or "\\" in relative_path
            or ".." in Path(relative_path).parts
            or not isinstance(declared_sha, str)
            or not SHA256_RE.fullmatch(declared_sha)
            or relative_path in by_path
        ):
            raise DashboardInputError(f"{label}_invalid")
        by_path[relative_path] = {
            "path": relative_path,
            "sha256": declared_sha,
        }
    return [by_path[path] for path in sorted(by_path)]


def _archive_catalog_binding(
    *,
    record_root: Path,
    catalog_row: dict[str, Any],
    logical_refs: list[dict[str, str]],
    project_root: Path,
) -> dict[str, Any]:
    record_id = catalog_row.get("record_id")
    relative_path = catalog_row.get("relative_path")
    locator = catalog_row.get("archive_locator")
    if (
        not isinstance(record_id, str)
        or not RECORD_NAME_RE.fullmatch(record_id)
        or not isinstance(relative_path, str)
        or not isinstance(locator, dict)
        or locator.get("schema_id") != ARCHIVE_LOCATOR_SCHEMA_VERSION
        or not isinstance(locator.get("archive_id"), str)
        or not locator["archive_id"]
        or not isinstance(locator.get("member_prefix"), str)
        or not locator["member_prefix"]
        or Path(locator["member_prefix"]).is_absolute()
        or ".." in Path(locator["member_prefix"]).parts
    ):
        raise DashboardInputError("record_catalog_archive_locator_invalid:" + str(record_id))
    try:
        loaded = load_archive_binding(
            record_root,
            catalog_row,
            project_root=project_root,
        )
    except StrategyRecordStoreError as exc:
        raise DashboardInputError(
            "record_catalog_archive_binding_invalid:" + record_id + ":" + str(exc)
        ) from exc
    manifest = loaded["manifest"]
    receipt = loaded["receipt"]
    full_refs = [
        {
            "path": locator["manifest_path"],
            "sha256": locator["manifest_sha256"],
            "bytes": loaded["manifest_path"].stat().st_size,
            "media_type": "application/json",
        },
        {
            "path": locator["restore_receipt_path"],
            "sha256": locator["restore_receipt_sha256"],
            "bytes": loaded["receipt_path"].stat().st_size,
            "media_type": "application/json",
        },
        {
            "path": locator["archive_path"],
            "sha256": locator["archive_sha256"],
            "bytes": locator["archive_bytes"],
            "media_type": "application/zstd",
        },
    ]
    physical_refs = [
        _stable_stream_ref(
            ref,
            project_root,
            label="record_catalog_archive_storage_ref",
        )
        for ref in full_refs
    ]
    matches = [
        item
        for item in manifest["records"]
        if isinstance(item, dict) and item.get("record_id") == record_id
    ]
    if len(matches) != 1:
        raise DashboardInputError("record_catalog_archive_manifest_record_missing:" + record_id)
    manifest_row = matches[0]
    inventory = catalog_row.get("inventory")
    expected = {
        "record_id": record_id,
        "relative_path": relative_path,
        "member_prefix": locator["member_prefix"],
        "inventory_sha256": catalog_row.get("inventory_sha256"),
        "file_count": catalog_row.get("file_count"),
        "total_bytes": catalog_row.get("total_bytes"),
        "inventory": inventory,
        "logical_source_refs": logical_refs,
    }
    if any(manifest_row.get(key) != value for key, value in expected.items()):
        raise DashboardInputError("record_catalog_archive_manifest_closure_mismatch:" + record_id)
    if record_id not in receipt.get("record_ids", []):
        raise DashboardInputError(
            "record_catalog_archive_restore_receipt_record_missing:" + record_id
        )
    return {
        "archive_id": locator["archive_id"],
        "storage_state": "ARCHIVED",
        "record_inventory_sha256": catalog_row.get("inventory_sha256"),
        "logical_source_refs": logical_refs,
        "archive_storage_refs": full_refs,
        "physical_source_refs": physical_refs,
    }


def _catalog_history_registry_binding(
    catalog: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], StableArtifact] | None:
    if catalog.get("schema_id") != "myquant.strategy_record_catalog.v2":
        return None
    ref = catalog.get("history_registry_ref")
    embedded = catalog.get("history_registry")
    if (
        not isinstance(ref, dict)
        or set(ref) != {"path", "sha256"}
        or not isinstance(ref.get("path"), str)
        or ref["path"].startswith("/")
        or "\\" in ref["path"]
        or ".." in Path(ref["path"]).parts
        or not isinstance(ref.get("sha256"), str)
        or not SHA256_RE.fullmatch(ref["sha256"])
        or not isinstance(embedded, dict)
    ):
        raise DashboardInputError("catalog_history_registry_ref_invalid")
    registry_path = project_root / ref["path"]
    try:
        resolved = registry_path.resolve(strict=True)
        resolved.relative_to(project_root.resolve())
    except (OSError, ValueError) as exc:
        raise DashboardInputError("catalog_history_registry_ref_invalid") from exc
    if resolved != registry_path.absolute():
        raise DashboardInputError("catalog_history_registry_ref_invalid")
    artifact = stable_read(registry_path, project_root)
    if artifact.sha256 != ref["sha256"]:
        raise DashboardInputError("catalog_history_registry_sha_mismatch")
    body = load_json(artifact)
    if body != embedded:
        raise DashboardInputError("catalog_history_registry_body_mismatch")
    historical_generation_id = embedded.get("intended_generation_id")
    if (
        not isinstance(historical_generation_id, str)
        or GENERATION_ID_RE.fullmatch(historical_generation_id) is None
    ):
        raise DashboardInputError("catalog_history_registry_generation_invalid")
    return embedded, artifact


def _registered_dashboard_projection(
    *,
    record_root: Path,
    project_root: Path,
    registered_override: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> (
    tuple[
        list[dict[str, Any]],
        list[str],
        str | None,
        list[dict[str, Any]],
        list[str],
        dict[str, Any],
        list[StableArtifact],
        dict[str, Any],
    ]
    | None
):
    """Load a registered catalog projection without raw-directory fallback."""

    if registered_override is None:
        try:
            registered = load_registered_catalog(record_root)
        except StrategyRecordStoreError as exc:
            raise DashboardInputError(f"record_catalog_invalid:{exc}") from exc
    else:
        registered = registered_override
    if registered is None:
        return None
    pointer, catalog = registered
    schema_id = catalog.get("schema_id")
    if schema_id == "myquant.strategy_record_catalog.v3":
        return _registered_dashboard_v3_projection(
            record_root=record_root,
            project_root=project_root,
            pointer=pointer,
            catalog=catalog,
            registered_override=registered_override,
        )
    if catalog.get("fixture_only_legacy_dashboard") is True:
        return _legacy_registered_dashboard_projection_body(
            record_root=record_root,
            project_root=project_root,
            pointer=pointer,
            catalog=catalog,
            registered_override=registered_override,
        )
    if schema_id in {
        "myquant.strategy_record_catalog.v1",
        "myquant.strategy_record_catalog.v2",
    }:
        raise DashboardInputError("CANONICAL_PERFORMANCE_CLOSURE_MISSING")
    raise DashboardInputError("record_catalog_schema_unsupported")


def _registered_dashboard_v3_projection(
    *,
    record_root: Path,
    project_root: Path,
    pointer: dict[str, Any],
    catalog: dict[str, Any],
    registered_override: tuple[dict[str, Any], dict[str, Any]] | None,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    str | None,
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
    list[StableArtifact],
    dict[str, Any],
]:
    """Build the Dashboard adapter solely from catalog v3 closures."""

    if catalog.get("performance_contract_ready") is not True:
        raise DashboardInputError("CANONICAL_PERFORMANCE_CLOSURE_MISSING")
    try:
        performance = load_performance_history(record_root, catalog["performance_history_ref"])
    except (KeyError, StrategyRecordStoreError) as exc:
        raise DashboardInputError(f"canonical_performance_invalid:{exc}") from exc
    catalog_by_id = {
        row["record_id"]: row
        for row in catalog.get("records", [])
        if isinstance(row, dict) and isinstance(row.get("record_id"), str)
    }
    active_id = pointer.get("active_record_id")
    previous_id = pointer.get("previous_record_id")
    if (
        not isinstance(active_id, str)
        or not isinstance(previous_id, str)
        or active_id == previous_id
    ):
        raise DashboardInputError("record_catalog_pointer_selection_invalid")

    def load_selected(record_id: str) -> dict[str, Any]:
        catalog_row = catalog_by_id.get(record_id)
        if (
            catalog_row is None
            or catalog_row.get("state", catalog_row.get("storage_state")) != "ONLINE"
            or catalog_row.get("ledger_path")
            != f"{catalog_row.get('relative_path')}/ledger_after_manual_switch.parquet"
        ):
            raise DashboardInputError(
                "catalog_v3_selected_record_parquet_closure_invalid:" + record_id
            )
        record_dir = record_root / str(catalog_row["relative_path"])
        row = validate_record(record_dir, record_root, project_root)
        for row_key, catalog_key in (
            ("manifest_sha256", "manifest_sha256"),
            ("manual_manifest_sha256", "manual_manifest_sha256"),
            ("ledger_sha256", "ledger_sha256"),
            ("financial_state_sha256", "financial_state_sha256"),
        ):
            if row.get(row_key) != catalog_row.get(catalog_key):
                raise DashboardInputError("catalog_v3_selected_record_sha_mismatch:" + record_id)
        row["storage_state"] = "ONLINE"
        row["record_inventory_sha256"] = catalog_row.get("inventory_sha256")
        row["evidence_status"] = "CATALOG_V3_ACTIVE_CLOSURE"
        return row

    previous = load_selected(previous_id)
    latest = load_selected(active_id)
    performance_ref = catalog["performance_history_ref"]
    try:
        record_root_relative = (
            record_root.resolve(strict=True)
            .relative_to(project_root.resolve(strict=True))
            .as_posix()
        )
    except (OSError, ValueError) as exc:
        raise DashboardInputError("catalog_v3_record_root_escape") from exc
    performance_source_refs = [
        {
            "path": f"{record_root_relative}/{performance_ref[key]['path']}",
            "sha256": performance_ref[key]["sha256"],
        }
        for key in ("manifest", "series", "owner_declaration")
    ]
    for selected in (previous, latest):
        selected["source_refs"] = [
            {
                "path": selected[key],
                "sha256": selected[key.replace("_path", "_sha256")],
            }
            for key in ("manifest_path", "manual_manifest_path", "ledger_path", "pnl_path")
            if selected.get(key) is not None
            and selected.get(key.replace("_path", "_sha256")) is not None
        ]
        selected["logical_source_refs"] = list(selected["source_refs"])
        selected["_physical_source_refs"] = list(selected["source_refs"])

    historical: list[dict[str, Any]] = []
    canonical_points: list[dict[str, Any]] = []
    funding_events: list[dict[str, Any]] = []
    prior_excluded = 0.0
    initial_units = 1_000_000.0
    for raw in performance["rows"]:
        record_id = raw["record_id"]
        stored = catalog_by_id.get(record_id, {})
        cash = float(raw["cash_cny"])
        equity = float(raw["equity_market_value_cny"])
        total = float(raw["raw_nav_cny"])
        # The canonical series carries cumulative declared external flow
        # separately from the flow-neutral adjusted NAV.  Re-deriving this as
        # raw NAV minus adjusted NAV would incorrectly include investment
        # gains or losses earned by contributed units.
        excluded = float(raw["excluded_external_flow_cny"])
        canonical_points.append(
            {
                "date": raw["valuation_date"],
                "record": record_id,
                "unit_nav": float(raw["unit_nav"]),
                "total_value": total,
                "excluded_external_flow": excluded,
                "adjusted_total_value": float(raw["unit_nav"]) * initial_units,
                "performance_initial_capital": initial_units,
                "evidence_status": "CANONICAL_PERFORMANCE_CLOSURE",
                "interval_return": float(raw["interval_return"]),
                "cumulative_return": float(raw["cumulative_return"]),
                "drawdown": float(raw["drawdown"]),
                "financial_state_sha256": raw.get("financial_state_sha256"),
            }
        )
        flow_delta = excluded - prior_excluded
        if not _almost_equal(flow_delta, 0.0):
            if not canonical_points[:-1]:
                raise DashboardInputError("canonical_initial_performance_point_contains_flow")
            previous_point = canonical_points[-2]
            funding_events.append(
                {
                    "record": record_id,
                    "date": raw["valuation_date"],
                    "amount": flow_delta,
                    "direction": "CONTRIBUTION" if flow_delta > 0 else "REDEMPTION_OR_CORRECTION",
                    "evidence_path": performance_source_refs[0]["path"],
                    "evidence_sha256": performance_ref["manifest"]["sha256"],
                    "binding_status": "CANONICAL_PERFORMANCE_CLOSURE",
                    "total_value_before": previous_point["total_value"],
                    "total_value_after": total,
                    "pre_flow_record": previous_point["record"],
                    "post_flow_record": record_id,
                }
            )
        prior_excluded = excluded
        historical.append(
            {
                "record": record_id,
                "valuation_date": raw["valuation_date"],
                "data_date": raw["valuation_date"],
                "recorded_at": raw["valuation_at"],
                "source_record": None,
                "accounting": {
                    "cash_after": cash,
                    "market_value_after": equity,
                    "total_value_after": total,
                    "portfolio_pnl_after": float(raw["portfolio_pnl_cny"]),
                    "realized_pnl_from_rebalance": 0.0,
                },
                "capital_base": initial_units,
                "funding": None,
                "funding_correction": None,
                "evidence_status": "CANONICAL_PERFORMANCE_CLOSURE",
                "manifest_path": performance_source_refs[0]["path"],
                "manifest_sha256": performance_ref["manifest"]["sha256"],
                "manual_manifest_path": stored.get("manual_manifest_path"),
                "manual_manifest_sha256": raw.get("manual_manifest_sha256"),
                "ledger_path": (
                    stored.get("ledger_path")
                    if raw.get("ledger_parquet_sha256") is not None
                    else None
                ),
                "ledger_sha256": raw.get("ledger_parquet_sha256"),
                "pnl_path": stored.get("pnl_path"),
                "pnl_sha256": stored.get("pnl_sha256"),
                "financial_state_sha256": raw.get("financial_state_sha256"),
                "source_refs": list(performance_source_refs),
                "logical_source_refs": list(performance_source_refs),
                "_physical_source_refs": list(performance_source_refs),
                "storage_state": stored.get("state", stored.get("storage_state", "ARCHIVED")),
                "record_inventory_sha256": stored.get("inventory_sha256"),
            }
        )

    catalog_artifacts: list[StableArtifact] = []
    if registered_override is None:
        pointer_artifact = stable_read(record_root / "_record_store/current.v1.json", project_root)
        catalog_artifact = stable_read(record_root / str(pointer["catalog_path"]), project_root)
        if load_json(pointer_artifact) != pointer or load_json(catalog_artifact) != catalog:
            raise DashboardInputError("catalog_v3_readback_mismatch")
        catalog_artifacts.extend([pointer_artifact, catalog_artifact])
        for ref in performance_source_refs:
            artifact = stable_read(project_root / ref["path"], project_root)
            if artifact.sha256 != ref["sha256"]:
                raise DashboardInputError("canonical_performance_source_sha_mismatch")
            catalog_artifacts.append(artifact)
    return (
        [previous, latest],
        [],
        active_id,
        historical,
        [],
        {
            "latest": latest,
            "previous": previous,
            "active_record_id": active_id,
            "previous_record_id": previous_id,
        },
        catalog_artifacts,
        {
            "publication_generation_id": pointer["generation_id"],
            "intended_generation_id": pointer["generation_id"],
            "dashboard_projection_sha256": None,
            "archive_bindings": {},
            "history_registry_ref": None,
            "history_registry": None,
            "canonical_performance_points": canonical_points,
            "canonical_funding_events": funding_events,
            "performance_history_ref": performance_ref,
            "lineage_index_sha256": catalog["lineage_index_sha256"],
        },
    )


def _legacy_registered_dashboard_projection_body(
    *,
    record_root: Path,
    project_root: Path,
    pointer: dict[str, Any],
    catalog: dict[str, Any],
    registered_override: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    str | None,
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
    list[StableArtifact],
    dict[str, Any],
]:

    # The code below is retained only as fixture/migration implementation.  All
    # governed registered roots returned above, so it cannot be reached.
    publication_generation_id = pointer.get("generation_id")
    if catalog.get("schema_id") == "myquant.strategy_record_catalog.v2" and (
        not isinstance(publication_generation_id, str)
        or GENERATION_ID_RE.fullmatch(publication_generation_id) is None
        or catalog.get("generation_id") != publication_generation_id
    ):
        raise DashboardInputError("record_catalog_generation_mismatch")
    history_registry_binding = _catalog_history_registry_binding(catalog, project_root)
    historical_generation_id = publication_generation_id
    if history_registry_binding is not None:
        historical_generation_id = history_registry_binding[0]["intended_generation_id"]
    projection = catalog.get("dashboard_projection")
    if not isinstance(projection, dict):
        raise DashboardInputError("catalog_dashboard_projection_missing")
    required = {
        "valid_records",
        "rejected",
        "latest_seen",
        "historical_records",
        "historical_rejected",
    }
    if not required.issubset(projection):
        raise DashboardInputError("catalog_dashboard_projection_incomplete")

    valid, valid_by_id = _projection_records(projection["valid_records"], "valid_records")
    historical_records, historical_by_id = _projection_records(
        projection["historical_records"], "historical_records"
    )
    rejected = _projection_string_list(projection["rejected"], "rejected")
    historical_rejected = _projection_string_list(
        projection["historical_rejected"], "historical_rejected"
    )
    latest_seen = projection["latest_seen"]
    if latest_seen is not None and (
        not isinstance(latest_seen, str) or not RECORD_NAME_RE.fullmatch(latest_seen)
    ):
        raise DashboardInputError("catalog_dashboard_projection_latest_seen_invalid")
    normalized_projection = {
        "valid_records": [
            {
                **row,
                "source_refs": _logical_source_refs(
                    row["source_refs"],
                    label="catalog_dashboard_projection_source_ref",
                ),
            }
            for row in valid
        ],
        "rejected": rejected,
        "latest_seen": latest_seen,
        "historical_records": [
            {
                **row,
                "source_refs": _logical_source_refs(
                    row["source_refs"],
                    label="catalog_dashboard_projection_source_ref",
                ),
            }
            for row in historical_records
        ],
        "historical_rejected": historical_rejected,
    }
    projection_sha256 = sha256_bytes(canonical_json_bytes(normalized_projection))

    catalog_rows = catalog.get("records")
    if not isinstance(catalog_rows, list):
        raise DashboardInputError("record_catalog_records_invalid")
    catalog_by_id: dict[str, dict[str, Any]] = {}
    online_paths: dict[str, Path] = {}
    online_inventory: dict[str, dict[Path, str]] = {}
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        record_id = row.get("record_id")
        if (
            isinstance(record_id, str)
            and RECORD_NAME_RE.fullmatch(record_id)
            and row.get("state") in {"ONLINE", "ARCHIVED"}
        ):
            if record_id in catalog_by_id:
                raise DashboardInputError("record_catalog_record_duplicate")
            catalog_by_id[record_id] = row
        if row.get("state") != "ONLINE":
            continue
        relative_path = row.get("relative_path")
        if (
            not isinstance(record_id, str)
            or not RECORD_NAME_RE.fullmatch(record_id)
            or not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or record_id in online_paths
        ):
            raise DashboardInputError("record_catalog_online_inventory_invalid")
        resolved = (record_root / relative_path).resolve()
        try:
            resolved.relative_to(record_root.resolve())
        except ValueError as exc:
            raise DashboardInputError("record_catalog_online_inventory_invalid") from exc
        if not resolved.is_dir() or resolved.is_symlink():
            raise DashboardInputError("record_catalog_online_record_missing:" + record_id)
        online_paths[record_id] = resolved
        inventory = row.get("inventory")
        if not isinstance(inventory, list):
            raise DashboardInputError("record_catalog_online_inventory_invalid")
        file_inventory: dict[Path, str] = {}
        for item in inventory:
            if not isinstance(item, dict) or item.get("type") not in {
                "file",
                "directory",
            }:
                raise DashboardInputError("record_catalog_online_inventory_invalid")
            item_path = item.get("path")
            if (
                not isinstance(item_path, str)
                or not item_path
                or Path(item_path).is_absolute()
                or ".." in Path(item_path).parts
            ):
                raise DashboardInputError("record_catalog_online_inventory_invalid")
            inventory_path = (resolved / item_path).resolve()
            try:
                inventory_path.relative_to(resolved)
            except ValueError as exc:
                raise DashboardInputError("record_catalog_online_inventory_invalid") from exc
            if item["type"] == "file":
                inventory_sha = item.get("sha256")
                if (
                    not isinstance(inventory_sha, str)
                    or not SHA256_RE.fullmatch(inventory_sha)
                    or inventory_path in file_inventory
                ):
                    raise DashboardInputError("record_catalog_online_inventory_invalid")
                file_inventory[inventory_path] = inventory_sha
        online_inventory[record_id] = file_inventory

    projected_ids = set(valid_by_id) | set(historical_by_id)
    missing_storage = sorted(projected_ids - set(catalog_by_id))
    if missing_storage:
        raise DashboardInputError(
            "catalog_dashboard_projection_record_unregistered:" + missing_storage[0]
        )

    active_record_id = pointer.get("active_record_id")
    previous_record_id = pointer.get("previous_record_id")
    if (
        not isinstance(active_record_id, str)
        or not isinstance(previous_record_id, str)
        or active_record_id == previous_record_id
        or active_record_id not in valid_by_id
        or previous_record_id not in valid_by_id
        or active_record_id not in online_paths
        or previous_record_id not in online_paths
    ):
        raise DashboardInputError("record_catalog_pointer_projection_mismatch")

    # The catalog projection preserves logical evidence paths. ONLINE rows are
    # re-read directly; ARCHIVED rows are closed through their immutable
    # archive manifest + restore receipt and never touch the old hot path.
    online_roots = tuple(online_paths.values())
    inventory_refs = {
        path: digest
        for inventory in online_inventory.values()
        for path, digest in inventory.items()
    }
    archive_bindings: dict[str, dict[str, Any]] = {}
    for row in [*valid, *historical_records]:
        record_id = row["record"]
        logical_refs = _logical_source_refs(
            row["source_refs"],
            label="catalog_dashboard_projection_source_ref",
        )
        row["source_refs"] = logical_refs
        row["logical_source_refs"] = logical_refs
        catalog_row = catalog_by_id[record_id]
        state = catalog_row.get("state")
        row["storage_state"] = state
        if state == "ARCHIVED":
            binding = _archive_catalog_binding(
                record_root=record_root,
                catalog_row=catalog_row,
                logical_refs=logical_refs,
                project_root=project_root,
            )
            archive_bindings[record_id] = binding
            row["record_inventory_sha256"] = binding["record_inventory_sha256"]
            row["_physical_source_refs"] = binding["physical_source_refs"]
            continue
        row["record_inventory_sha256"] = catalog_row.get("inventory_sha256")
        row["_physical_source_refs"] = logical_refs
        for ref in logical_refs:
            relative_path = ref["path"]
            declared_sha = ref["sha256"]
            artifact = stable_read(project_root / relative_path, project_root)
            if artifact.sha256 != declared_sha:
                raise DashboardInputError(
                    "catalog_dashboard_projection_source_sha_mismatch:" + relative_path
                )
            if inventory_refs.get(artifact.path) != declared_sha:
                raise DashboardInputError(
                    "catalog_dashboard_projection_source_inventory_mismatch:" + relative_path
                )
            if not any(
                artifact.path == root or artifact.path.is_relative_to(root) for root in online_roots
            ):
                raise DashboardInputError(
                    "catalog_dashboard_projection_source_not_online:" + relative_path
                )

    catalog_artifacts: list[StableArtifact] = []
    if registered_override is None:
        pointer_path = record_root / "_record_store" / "current.v1.json"
        catalog_path_value = pointer.get("catalog_path")
        if not isinstance(catalog_path_value, str):
            raise DashboardInputError("record_catalog_pointer_path_invalid")
        catalog_path = (record_root / catalog_path_value).resolve()
        pointer_artifact = stable_read(pointer_path, project_root)
        catalog_artifact = stable_read(catalog_path, project_root)
        if load_json(pointer_artifact) != pointer:
            raise DashboardInputError("record_catalog_pointer_readback_mismatch")
        if load_json(catalog_artifact) != catalog:
            raise DashboardInputError("record_catalog_readback_mismatch")
        if pointer.get("catalog_sha256") != catalog_artifact.sha256:
            raise DashboardInputError("record_catalog_pointer_sha_mismatch")
        catalog_artifacts = [pointer_artifact, catalog_artifact]

    # Put the pointer-selected rows first so callers cannot accidentally
    # recover current/previous authority from lexicographic ordering.
    selected = [
        valid_by_id[active_record_id],
        valid_by_id[previous_record_id],
    ]
    declared_projection_sha = catalog.get("dashboard_projection_sha256")
    if declared_projection_sha is not None and (declared_projection_sha != projection_sha256):
        raise DashboardInputError("catalog_dashboard_projection_sha_mismatch")
    if history_registry_binding is not None:
        registry_by_record, _registry_artifact = load_history_integrity_registry(
            history_registry_binding[1].path,
            project_root,
            intended_generation_id=historical_generation_id,
            dashboard_projection_sha256=projection_sha256,
            archive_bindings=archive_bindings,
        )
        if set(registry_by_record) != set(historical_by_id):
            raise DashboardInputError("catalog_history_registry_record_set_mismatch")
        for record in historical_records:
            registry_row = registry_by_record[record["record"]]
            if (
                registry_row["logical_source_refs"] != record["logical_source_refs"]
                or registry_row["storage_state"] != record["storage_state"]
                or registry_row["record_inventory_sha256"] != record["record_inventory_sha256"]
            ):
                raise DashboardInputError(
                    "catalog_history_registry_projection_mismatch:" + record["record"]
                )
    return (
        valid,
        rejected,
        latest_seen,
        historical_records,
        historical_rejected,
        {
            "latest": selected[0],
            "previous": selected[1],
            "active_record_id": active_record_id,
            "previous_record_id": previous_record_id,
        },
        catalog_artifacts,
        {
            "publication_generation_id": publication_generation_id,
            "intended_generation_id": historical_generation_id,
            "dashboard_projection_sha256": projection_sha256,
            "archive_bindings": archive_bindings,
            "history_registry_ref": (
                {
                    "path": history_registry_binding[1].relative_path,
                    "sha256": history_registry_binding[1].sha256,
                }
                if history_registry_binding is not None
                else None
            ),
            "history_registry": (
                history_registry_binding[0] if history_registry_binding is not None else None
            ),
        },
    )


def load_dashboard_catalog_projection(record_root: Path, project_root: Path) -> dict[str, Any]:
    """Load the validated projection for operator-side registry refreshes.

    A registered production root is catalog-only.  The legacy scanner remains
    available solely for an unregistered bootstrap root and test fixtures.
    """

    registered = _registered_dashboard_projection(
        record_root=record_root,
        project_root=project_root,
    )
    if registered is None:
        return build_dashboard_catalog_projection(record_root, project_root)
    (
        valid,
        rejected,
        latest_seen,
        historical_records,
        historical_rejected,
        _selection,
        _catalog_artifacts,
        integrity_context,
    ) = registered
    return {
        "valid_records": valid,
        "rejected": rejected,
        "latest_seen": latest_seen,
        "historical_records": historical_records,
        "historical_rejected": historical_rejected,
        "integrity_context": integrity_context,
    }


def load_dashboard_candidate_projection(
    *,
    record_root: Path,
    project_root: Path,
    pointer: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """Validate a not-yet-active catalog for shadow parity checks."""

    registered = _registered_dashboard_projection(
        record_root=record_root,
        project_root=project_root,
        registered_override=(pointer, catalog),
    )
    if registered is None:  # pragma: no cover - explicit override is present
        raise DashboardInputError("candidate_catalog_missing")
    (
        valid,
        rejected,
        latest_seen,
        historical_records,
        historical_rejected,
        selection,
        _catalog_artifacts,
        integrity_context,
    ) = registered
    return {
        "valid_records": valid,
        "rejected": rejected,
        "latest_seen": latest_seen,
        "historical_records": historical_records,
        "historical_rejected": historical_rejected,
        "selection": selection,
        "integrity_context": integrity_context,
    }


def build_history_integrity_registry(
    records: list[dict[str, Any]],
    *,
    generated_at: str,
    intended_generation_id: str | None = None,
    dashboard_projection_sha256: str | None = None,
    archive_bindings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a historical-state-bound Dashboard integrity declaration.

    The registry binds the generation where the historical projection changed
    and the normalized projection content, never the candidate catalog byte
    SHA (which would make catalog publication self-referential). Receipt-only
    catalog publications intentionally retain this historical generation.
    """

    archive_bindings = archive_bindings or {}
    if intended_generation_id is None:
        intended_generation_id = "UNREGISTERED_LEGACY"
    rows: list[dict[str, Any]] = []
    for record in records:
        record_id = record["record"]
        logical_refs = _logical_source_refs(
            record.get("logical_source_refs", record.get("source_refs")),
            label="history_integrity_logical_source_refs",
        )
        storage_state = record.get("storage_state", "ONLINE")
        if storage_state not in {"ONLINE", "ARCHIVED"}:
            raise DashboardInputError("history_integrity_storage_state_invalid:" + record_id)
        inventory_sha = record.get("record_inventory_sha256")
        if not isinstance(inventory_sha, str) or not SHA256_RE.fullmatch(inventory_sha):
            inventory_sha = sha256_bytes(canonical_json_bytes(logical_refs))
        archive_storage_refs: list[dict[str, Any]] = []
        if storage_state == "ARCHIVED":
            binding = archive_bindings.get(record_id)
            if not isinstance(binding, dict):
                raise DashboardInputError("history_integrity_archive_binding_missing:" + record_id)
            refs = binding.get("archive_storage_refs")
            if not isinstance(refs, list) or len(refs) != 3:
                raise DashboardInputError("history_integrity_archive_refs_invalid:" + record_id)
            archive_storage_refs = [dict(ref) for ref in refs]
        rows.append(
            {
                "record": record_id,
                "storage_state": storage_state,
                "record_inventory_sha256": inventory_sha,
                "logical_source_refs": logical_refs,
                "archive_storage_refs": archive_storage_refs,
            }
        )
    if dashboard_projection_sha256 is None:
        dashboard_projection_sha256 = sha256_bytes(canonical_json_bytes(rows))

    payload = {
        "schema_version": HISTORY_INTEGRITY_SCHEMA_VERSION,
        "market": MARKET,
        "strategy_label": STRATEGY,
        "generated_at": generated_at,
        "authority": "DASHBOARD_POST_HOC_INTEGRITY_DECLARATION",
        "intended_generation_id": intended_generation_id,
        "dashboard_projection_sha256": dashboard_projection_sha256,
        "record_count": len(rows),
        "records": rows,
    }
    payload["content_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def load_history_integrity_registry(
    path: Path,
    project_root: Path,
    *,
    intended_generation_id: str | None = None,
    dashboard_projection_sha256: str | None = None,
    archive_bindings: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], StableArtifact]:
    artifact = stable_read(path, project_root)
    registry = load_json(artifact)
    if not isinstance(registry, dict):
        raise DashboardInputError("history_integrity_registry_not_object")
    if (
        registry.get("schema_version") != HISTORY_INTEGRITY_SCHEMA_VERSION
        or registry.get("market") != MARKET
        or registry.get("strategy_label") != STRATEGY
        or registry.get("authority") != "DASHBOARD_POST_HOC_INTEGRITY_DECLARATION"
    ):
        raise DashboardInputError("history_integrity_registry_identity_invalid")
    if intended_generation_id is not None and (
        registry.get("intended_generation_id") != intended_generation_id
    ):
        raise DashboardInputError("history_integrity_registry_generation_mismatch")
    if dashboard_projection_sha256 is not None and (
        registry.get("dashboard_projection_sha256") != dashboard_projection_sha256
    ):
        raise DashboardInputError("history_integrity_registry_projection_mismatch")
    declared_content_sha = registry.get("content_sha256")
    without_hash = dict(registry)
    without_hash.pop("content_sha256", None)
    actual_content_sha = sha256_bytes(canonical_json_bytes(without_hash))
    if declared_content_sha != actual_content_sha:
        raise DashboardInputError("history_integrity_registry_hash_mismatch")
    rows = registry.get("records")
    if not isinstance(rows, list) or registry.get("record_count") != len(rows):
        raise DashboardInputError("history_integrity_registry_records_invalid")
    archive_bindings = archive_bindings or {}
    by_record: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "record",
            "storage_state",
            "record_inventory_sha256",
            "logical_source_refs",
            "archive_storage_refs",
        }:
            raise DashboardInputError("history_integrity_registry_row_invalid")
        record = row["record"]
        refs = row["logical_source_refs"]
        storage_state = row["storage_state"]
        inventory_sha = row["record_inventory_sha256"]
        if (
            not isinstance(record, str)
            or not RECORD_NAME_RE.fullmatch(record)
            or record in by_record
            or storage_state not in {"ONLINE", "ARCHIVED"}
            or not isinstance(inventory_sha, str)
            or not SHA256_RE.fullmatch(inventory_sha)
        ):
            raise DashboardInputError("history_integrity_registry_row_invalid")
        logical_refs = _logical_source_refs(refs, label="history_integrity_registry_ref")
        archive_refs = row["archive_storage_refs"]
        if storage_state == "ONLINE":
            if archive_refs != []:
                raise DashboardInputError("history_integrity_online_archive_refs_present")
            for ref in logical_refs:
                source = stable_read(project_root / ref["path"], project_root)
                if source.sha256 != ref["sha256"]:
                    raise DashboardInputError(
                        "history_integrity_source_sha_mismatch:" + ref["path"]
                    )
        else:
            binding = archive_bindings.get(record)
            if (
                not isinstance(binding, dict)
                or archive_refs != binding.get("archive_storage_refs")
                or inventory_sha != binding.get("record_inventory_sha256")
            ):
                raise DashboardInputError("history_integrity_archive_binding_mismatch:" + record)
            for index, ref in enumerate(archive_refs):
                if not isinstance(ref, dict):
                    raise DashboardInputError("history_integrity_archive_ref_invalid")
                _stable_stream_ref(
                    ref,
                    project_root,
                    label=f"history_integrity_archive_ref_{index}",
                )
        by_record[record] = {
            "storage_state": storage_state,
            "record_inventory_sha256": inventory_sha,
            "logical_source_refs": logical_refs,
            "archive_storage_refs": archive_refs,
        }
    return by_record, artifact


def _read_benchmark_rows(artifact: StableArtifact, ts_code: str) -> dict[str, dict[str, Any]]:
    rows = _csv_rows(artifact)
    required = {
        "date",
        "ts_code",
        "close",
        "source_system",
        "value_date",
        "coverage",
    }
    if not required.issubset(rows[0]):
        raise DashboardInputError("benchmark_columns_missing")
    selected: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        if row.get("ts_code") != ts_code:
            continue
        date_value = str(row.get("date") or "")
        value_date = str(row.get("value_date") or "")
        try:
            current_date = date.fromisoformat(date_value)
            source_date = date.fromisoformat(value_date)
        except ValueError as exc:
            raise DashboardInputError(f"benchmark_date_invalid:{row_number}") from exc
        source_system = str(row.get("source_system") or "")
        coverage = str(row.get("coverage") or "")
        if source_system not in ALLOWED_BENCHMARK_SOURCES:
            raise DashboardInputError(f"benchmark_source_forbidden:{source_system}")
        if coverage not in ALLOWED_BENCHMARK_COVERAGE:
            raise DashboardInputError(f"benchmark_coverage_invalid:{coverage}")
        if coverage == "exact_close" and source_date != current_date:
            raise DashboardInputError(f"benchmark_exact_value_date_mismatch:{date_value}")
        if coverage == "previous_trading_day_ffill" and source_date >= current_date:
            raise DashboardInputError(f"benchmark_ffill_value_date_invalid:{date_value}")
        if date_value in selected:
            raise DashboardInputError(f"benchmark_duplicate_date:{date_value}")
        selected[date_value] = {
            "date": date_value,
            "close": _number(row.get("close"), f"benchmark_close:{date_value}"),
            "source_system": source_system,
            "value_date": value_date,
            "coverage": coverage,
        }
    return selected


def _read_risk_free_rows(
    artifact: StableArtifact,
) -> dict[str, dict[str, Any]]:
    """Read exact official ChinaBond 1Y government-bond curve observations."""

    rows = _csv_rows(artifact)
    required = {
        "date",
        "tenor",
        "annual_yield_percent",
        "source_system",
        "source_url",
    }
    if not required.issubset(rows[0]):
        raise DashboardInputError("risk_free_columns_missing")
    selected: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        date_value = str(row.get("date") or "")
        try:
            date.fromisoformat(date_value)
        except ValueError as exc:
            raise DashboardInputError(f"risk_free_date_invalid:{row_number}") from exc
        if row.get("tenor") != RISK_FREE_TENOR:
            raise DashboardInputError(f"risk_free_tenor_invalid:{row_number}")
        if row.get("source_system") != RISK_FREE_SOURCE:
            raise DashboardInputError(f"risk_free_source_invalid:{row_number}")
        if row.get("source_url") != RISK_FREE_SOURCE_URL:
            raise DashboardInputError(f"risk_free_url_invalid:{row_number}")
        annual_yield = (
            _number(
                row.get("annual_yield_percent"),
                f"risk_free_annual_yield_percent:{date_value}",
            )
            / 100.0
        )
        if annual_yield < 0 or annual_yield >= 1:
            raise DashboardInputError(f"risk_free_annual_yield_out_of_range:{date_value}")
        if date_value in selected:
            raise DashboardInputError(f"risk_free_duplicate_date:{date_value}")
        selected[date_value] = {
            "date": date_value,
            "annual_yield": annual_yield,
        }
    if not selected:
        raise DashboardInputError("risk_free_rows_missing")
    return selected


def _align_risk_free_rows(
    rows: dict[str, dict[str, Any]], required_dates: list[str]
) -> list[dict[str, Any]]:
    """Align by exact date or the latest previously published workday."""

    published_dates = sorted(rows)
    aligned: list[dict[str, Any]] = []
    for required in required_dates:
        eligible = [value for value in published_dates if value <= required]
        if not eligible:
            raise DashboardInputError("risk_free_missing_prior_date:" + required)
        value_date = eligible[-1]
        aligned.append(
            {
                "annual_yield": rows[value_date]["annual_yield"],
                "value_date": value_date,
                "coverage": (
                    "exact_published_yield"
                    if value_date == required
                    else "previous_published_workday_ffill"
                ),
            }
        )
    return aligned


def _max_drawdown(values: Iterable[float]) -> float:
    peak: float | None = None
    result = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak > 0:
            result = min(result, value / peak - 1.0)
    return result


def _changes(current: dict[str, Any], previous: dict[str, Any]) -> list[dict[str, Any]]:
    current_by_symbol = {row["symbol"]: row for row in current["positions"]}
    previous_by_symbol = {row["symbol"]: row for row in previous["positions"]}
    result: list[dict[str, Any]] = []
    for symbol in sorted(set(current_by_symbol) | set(previous_by_symbol)):
        now = current_by_symbol.get(symbol)
        before = previous_by_symbol.get(symbol)
        before_shares = before["shares"] if before else 0.0
        now_shares = now["shares"] if now else 0.0
        before_market_value = before["market_value"] if before else 0.0
        now_market_value = now["market_value"] if now else 0.0
        delta = now_shares - before_shares
        if before is None:
            change_type = "NEW"
        elif now is None:
            change_type = "CLOSED"
        elif delta > 0:
            change_type = "INCREASED"
        elif delta < 0:
            change_type = "REDUCED"
        else:
            change_type = "UNCHANGED"
        row = now or before
        result.append(
            {
                "symbol": symbol,
                "name": row["name"],
                "change_type": change_type,
                "previous_shares": before_shares,
                "current_shares": now_shares,
                "share_delta": delta,
                "previous_market_value": before_market_value,
                "current_market_value": now_market_value,
                "market_value_delta": now_market_value - before_market_value,
                "nav_weight_delta": (now["nav_weight"] if now else 0.0)
                - (before["nav_weight"] if before else 0.0),
                "equity_weight_delta": (now["equity_weight"] if now else 0.0)
                - (before["equity_weight"] if before else 0.0),
            }
        )
    return result


def _ledger_quantity_turnover(
    changes: list[dict[str, Any]],
    current: dict[str, Any],
    previous: dict[str, Any],
    *,
    current_total_value: float,
    previous_total_value: float,
) -> float:
    """Estimate two-way turnover from ledger quantity changes only."""

    current_by_symbol = {row["symbol"]: row for row in current["positions"]}
    previous_by_symbol = {row["symbol"]: row for row in previous["positions"]}
    gross_notional = 0.0
    for change in changes:
        share_delta = abs(change["share_delta"])
        if share_delta == 0:
            continue
        position = current_by_symbol.get(change["symbol"])
        if position is None:
            position = previous_by_symbol[change["symbol"]]
        gross_notional += share_delta * position["recorded_price"]
    average_nav = 0.5 * (current_total_value + previous_total_value)
    if average_nav <= 0:
        raise DashboardInputError("turnover_average_nav_not_positive")
    return 0.5 * gross_notional / average_nav


def _economic_portfolio_view(
    record: dict[str, Any], performance_point: dict[str, Any]
) -> tuple[dict[str, Any], float, float]:
    """Return the one-million-base portfolio view for a strict record.

    The archived accounting closure remains the exact-byte evidence source.
    Cash and total value exposed to the Dashboard remove cumulative amounts
    explicitly classified as external to the March 17 portfolio. Holdings and
    their market values are unchanged.
    """

    adjusted_total = float(performance_point["adjusted_total_value"])
    excluded_flow = float(performance_point["excluded_external_flow"])
    market_value = float(record["accounting"]["market_value_after"])
    adjusted_cash = float(record["accounting"]["cash_after"]) - excluded_flow
    if adjusted_cash < 0:
        raise DashboardInputError("economic_portfolio_cash_negative:" + record["record"])
    if not _almost_equal(adjusted_cash + market_value, adjusted_total):
        raise DashboardInputError("economic_portfolio_accounting_mismatch:" + record["record"])
    if adjusted_total <= 0:
        raise DashboardInputError("economic_portfolio_total_not_positive:" + record["record"])

    view = dict(record)
    view["positions"] = []
    for original in record["positions"]:
        position = dict(original)
        position["nav_weight"] = position["market_value"] / adjusted_total
        view["positions"].append(position)
    return view, adjusted_cash, adjusted_total


def _official_financial_state_view(
    record: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    """Return holdings weights against the registered raw financial state.

    Catalog v3 performance unitization is a return-accounting projection.  It
    must not replace the cash, NAV, or exposure denominator of the exact
    pointer-selected financial state after a contribution or redemption.
    """

    accounting = record["accounting"]
    cash = float(accounting["cash_after"])
    market_value = float(accounting["market_value_after"])
    total = float(accounting["total_value_after"])
    if total <= 0:
        raise DashboardInputError("official_financial_state_total_not_positive:" + record["record"])
    if not _almost_equal(cash + market_value, total):
        raise DashboardInputError(
            "official_financial_state_accounting_mismatch:" + record["record"]
        )
    view = dict(record)
    view["positions"] = []
    for original in record["positions"]:
        position = dict(original)
        position["nav_weight"] = position["market_value"] / total
        view["positions"].append(position)
    return view, cash, total


def _exclude_external_funding(
    records_by_date: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a March-base return series after subtracting external funding."""

    first = records_by_date[0]
    initial_capital = first.get("capital_base")
    first_total = first["accounting"]["total_value_after"]
    if (
        not isinstance(initial_capital, (int, float))
        or initial_capital <= 0
        or not _almost_equal(first_total, float(initial_capital))
    ):
        raise DashboardInputError("performance_initial_capital_baseline_invalid")
    cumulative_external_flow = 0.0
    corrected_funding_records: set[str] = set()
    timeline: list[dict[str, Any]] = []
    for record in records_by_date:
        funding = record.get("funding")
        if funding:
            cumulative_external_flow += funding["amount"]
        correction = record.get("funding_correction")
        if correction and correction["reversed_record"] not in corrected_funding_records:
            cumulative_external_flow -= correction["reversed_amount"]
            corrected_funding_records.add(correction["reversed_record"])
            if _almost_equal(cumulative_external_flow, 0.0):
                cumulative_external_flow = 0.0
            if cumulative_external_flow < 0:
                raise DashboardInputError(
                    "funding_correction_exceeds_external_flow:" + record["record"]
                )
        total_after = record["accounting"]["total_value_after"]
        adjusted_total = total_after - cumulative_external_flow
        unit_nav = adjusted_total / float(initial_capital)
        if unit_nav <= 0 or not math.isfinite(unit_nav):
            raise DashboardInputError("external_flow_excluded_nav_invalid")
        timeline.append(
            {
                "date": record.get("valuation_date") or record["data_date"],
                "record": record["record"],
                "unit_nav": unit_nav,
                "total_value": total_after,
                "excluded_external_flow": cumulative_external_flow,
                "adjusted_total_value": adjusted_total,
                "performance_initial_capital": float(initial_capital),
                "evidence_status": record.get("evidence_status", "HASH_BOUND_CURRENT_CLOSURE"),
            }
        )
    return timeline


def build_bundle(
    *,
    project_root: Path,
    record_root: Path,
    benchmark_path: Path,
    risk_free_path: Path | None = None,
    generated_at: str,
    today: date,
    history_integrity_path: Path | None = None,
    benchmark_gap_policy: str = "strict",
) -> dict[str, Any]:
    if benchmark_gap_policy not in {"strict", "allow_trailing"}:
        raise DashboardInputError("benchmark_gap_policy_invalid")
    registered_projection = _registered_dashboard_projection(
        record_root=record_root, project_root=project_root
    )
    catalog_artifacts: list[StableArtifact] = []
    integrity_context: dict[str, Any] = {
        "intended_generation_id": "UNREGISTERED_LEGACY",
        "dashboard_projection_sha256": None,
        "archive_bindings": {},
    }
    if registered_projection is None:
        projection = build_dashboard_catalog_projection(record_root, project_root)
        valid = projection["valid_records"]
        rejected = projection["rejected"]
        latest_seen = projection["latest_seen"]
        historical_records = projection["historical_records"]
        historical_rejected = projection["historical_rejected"]
        latest = valid[-1] if valid else None
        previous = valid[-2] if len(valid) >= 2 else None
    else:
        (
            valid,
            rejected,
            latest_seen,
            historical_records,
            historical_rejected,
            pointer_selection,
            catalog_artifacts,
            integrity_context,
        ) = registered_projection
        latest = pointer_selection["latest"]
        previous = pointer_selection["previous"]
    if len(valid) < 2:
        raise DashboardInputError("fewer_than_two_hash_bound_valid_records")
    assert latest is not None and previous is not None
    canonical_performance_points = integrity_context.get("canonical_performance_points")
    if canonical_performance_points is not None and history_integrity_path is not None:
        raise DashboardInputError("catalog_v3_history_integrity_path_forbidden")
    history_integrity_artifact: StableArtifact | None = None
    history_registry_bound_count = 0
    required_registry_ref = integrity_context.get("history_registry_ref")
    if required_registry_ref is not None:
        if history_integrity_path is None:
            raise DashboardInputError("catalog_history_registry_path_required")
        try:
            supplied_registry_path = (
                history_integrity_path.resolve(strict=True)
                .relative_to(project_root.resolve())
                .as_posix()
            )
        except (OSError, ValueError) as exc:
            raise DashboardInputError("catalog_history_registry_path_mismatch") from exc
        if supplied_registry_path != required_registry_ref["path"]:
            raise DashboardInputError("catalog_history_registry_path_mismatch")
    if history_integrity_path is not None:
        registry_by_record, history_integrity_artifact = load_history_integrity_registry(
            history_integrity_path,
            project_root,
            intended_generation_id=integrity_context.get("intended_generation_id"),
            dashboard_projection_sha256=integrity_context.get("dashboard_projection_sha256"),
            archive_bindings=integrity_context.get("archive_bindings"),
        )
        if (
            required_registry_ref is not None
            and history_integrity_artifact.sha256 != required_registry_ref["sha256"]
        ):
            raise DashboardInputError("catalog_history_registry_sha_mismatch")
        expected_records = {record["record"] for record in historical_records}
        if set(registry_by_record) != expected_records:
            raise DashboardInputError("history_integrity_registry_record_set_mismatch")
        for record in historical_records:
            registered_row = registry_by_record[record["record"]]
            if (
                registered_row["logical_source_refs"]
                != record.get("logical_source_refs", record["source_refs"])
                or registered_row["storage_state"] != record.get("storage_state", "ONLINE")
                or registered_row["record_inventory_sha256"]
                != record.get(
                    "record_inventory_sha256",
                    registered_row["record_inventory_sha256"],
                )
            ):
                raise DashboardInputError(
                    "history_integrity_registry_ref_set_mismatch:" + record["record"]
                )
            if record["evidence_status"] != "HASH_BOUND_CURRENT_CLOSURE":
                record["evidence_status"] = "DASHBOARD_POST_HOC_SHA_REGISTRY_BOUND"
                history_registry_bound_count += 1
    if canonical_performance_points is not None:
        if not isinstance(canonical_performance_points, list):
            raise DashboardInputError("canonical_performance_points_invalid")
        unitized_raw = [dict(point) for point in canonical_performance_points]
    else:
        unitized_raw = _exclude_external_funding(historical_records)
    unitized_by_record = {point["record"]: point for point in unitized_raw}

    def economic_point_for(record: dict[str, Any]) -> dict[str, Any]:
        existing = unitized_by_record.get(record["record"])
        if existing is not None:
            return existing
        if record.get("funding") is not None or record.get("funding_correction") is not None:
            raise DashboardInputError(
                "unvalued_current_record_contains_funding_event:" + record["record"]
            )
        prior = unitized_raw[-1]
        if canonical_performance_points is not None:
            if record.get("financial_state_sha256") != prior.get(
                "financial_state_sha256"
            ) or not _almost_equal(
                float(record["accounting"]["total_value_after"]),
                float(prior["total_value"]),
            ):
                raise DashboardInputError(
                    "canonical_performance_financial_state_missing:" + record["record"]
                )
            return dict(prior)
        excluded_flow = float(prior["excluded_external_flow"])
        total_value = float(record["accounting"]["total_value_after"])
        adjusted_total = total_value - excluded_flow
        if adjusted_total <= 0:
            raise DashboardInputError(
                "unvalued_current_record_adjusted_total_invalid:" + record["record"]
            )
        return {
            "adjusted_total_value": adjusted_total,
            "excluded_external_flow": excluded_flow,
        }

    latest_economic_point = economic_point_for(latest)
    previous_economic_point = economic_point_for(previous)
    if canonical_performance_points is not None:
        latest_view, portfolio_cash, portfolio_total = _official_financial_state_view(latest)
        previous_view, _, previous_portfolio_total = _official_financial_state_view(previous)
    else:
        latest_view, portfolio_cash, portfolio_total = _economic_portfolio_view(
            latest, latest_economic_point
        )
        previous_view, _, previous_portfolio_total = _economic_portfolio_view(
            previous, previous_economic_point
        )
    performance_adjusted_total = float(latest_economic_point["adjusted_total_value"])
    collapsed_unitized: dict[str, dict[str, Any]] = {}
    for point in unitized_raw:
        collapsed_unitized[point["date"]] = point
    unitized = [collapsed_unitized[key] for key in sorted(collapsed_unitized)]
    if len(unitized) < 2:
        raise DashboardInputError("portfolio_performance_has_no_comparable_interval")
    owner_corrected_initial_capital_path = (
        canonical_performance_points is not None
        and _is_owner_corrected_initial_capital_path(unitized)
    )

    required_dates = [row["date"] for row in unitized]
    benchmark_artifact = stable_read(benchmark_path, project_root)
    benchmark_series: dict[str, dict[str, Any]] = {}
    for spec in BENCHMARK_SPECS:
        rows = _read_benchmark_rows(benchmark_artifact, spec["ts_code"])
        missing_dates = [value for value in required_dates if value not in rows]
        if missing_dates and benchmark_gap_policy == "strict":
            raise DashboardInputError(
                spec["id"].lower() + "_benchmark_missing_dates:" + ",".join(missing_dates)
            )
        if missing_dates:
            first_missing = required_dates.index(missing_dates[0])
            if any(value in rows for value in required_dates[first_missing + 1 :]):
                raise DashboardInputError(
                    spec["id"].lower() + "_benchmark_non_trailing_gap:" + ",".join(missing_dates)
                )
            selected_dates = required_dates[:first_missing]
        else:
            selected_dates = required_dates
        if len(selected_dates) < 2:
            raise DashboardInputError(spec["id"].lower() + "_benchmark_prefix_too_short")
        selected = [rows[value] for value in selected_dates]
        first_close = selected[0]["close"]
        if first_close <= 0:
            raise DashboardInputError(spec["id"].lower() + "_initial_close_not_positive")
        nav = [row["close"] / first_close for row in selected]
        benchmark_series[spec["id"]] = {
            "spec": spec,
            "rows": selected,
            "selected_dates": selected_dates,
            "missing_dates": missing_dates,
            "nav": nav,
            "return": nav[-1] / nav[0] - 1.0,
            "max_drawdown": _max_drawdown(nav),
        }

    if risk_free_path is None:
        risk_free_path = benchmark_path.with_name("cn_govt_bond_yield.csv")
    risk_free_artifact = stable_read(risk_free_path, project_root)
    risk_free_rows = _read_risk_free_rows(risk_free_artifact)
    aligned_risk_free = _align_risk_free_rows(risk_free_rows, required_dates)

    portfolio_nav = [row["unit_nav"] for row in unitized]
    cumulative_return = float(unitized[-1].get("cumulative_return", portfolio_nav[-1] - 1.0))
    changes = _changes(latest_view, previous_view)
    gross = latest["accounting"]["market_value_after"] / portfolio_total
    cash_weight = portfolio_cash / portfolio_total
    equity_weights = sorted(
        (position["equity_weight"] for position in latest["positions"]),
        reverse=True,
    )
    top1 = sum(equity_weights[:1])
    top3 = sum(equity_weights[:3])
    hhi = sum(weight * weight for weight in equity_weights)
    current_unrealized = sum(position["unrealized_pnl"] for position in latest["positions"])
    data_age_days = (today - date.fromisoformat(latest["data_date"])).days
    performance_age_days = (today - date.fromisoformat(required_dates[-1])).days
    rejection_counts = Counter(
        (item.split(":", 1)[1] if ":" in item else item).split(":", 1)[0] for item in rejected
    )
    historical_rejection_counts = Counter(
        (item.split(":", 1)[1] if ":" in item else item).split(":", 1)[0]
        for item in historical_rejected
    )
    legacy_history_count = sum(
        record["evidence_status"]
        in {
            "ARCHIVE_INCEPTION_EXACT_BYTES_NO_DECLARED_SHA",
            "LEGACY_EXACT_BYTES_NO_DECLARED_SHA",
        }
        for record in historical_records
    )
    reversed_funding_records = {
        record["funding_correction"]["reversed_record"]
        for record in historical_records
        if record.get("funding_correction") is not None
    }
    if canonical_performance_points is not None:
        funding_events = (
            []
            if owner_corrected_initial_capital_path
            else [dict(event) for event in integrity_context.get("canonical_funding_events", [])]
        )
    else:
        funding_events = [
            {
                "record": record["record"],
                "date": record["valuation_date"],
                "amount": record["funding"]["amount"],
                "total_value_before": record["funding"]["total_value_before"],
                "total_value_after": record["funding"]["total_value_after"],
                "evidence_path": record["funding"]["evidence_path"],
                "evidence_sha256": record["funding"]["evidence_sha256"],
                "binding_status": record["funding"]["binding_status"],
            }
            for record in historical_records
            if record.get("funding") is not None
            and record["record"] not in reversed_funding_records
        ]
    # Disclosure limitations remain visible, but they do not by themselves
    # make the core Dashboard stale.  PARTIAL is reserved for a gap in the
    # selected holdings/performance chain: stale data, a newer unusable
    # record, or legacy performance bytes that are not bound by the
    # Dashboard integrity registry.
    warnings = [
        "trade_fee_and_net_of_fee_basis_unknown",
        "per_position_realized_pnl_unavailable",
        "current_quote_unavailable_recorded_prices_only",
        "industry_and_theme_exposure_not_hash_bound_in_effective_ledger",
    ]
    status_gaps: list[str] = []
    publication_delay = latest.get("publication_delay")
    late_publication_current = (
        latest.get("publication_class") == "LATE_OFFICIAL_VALUATION_PUBLICATION"
        and isinstance(publication_delay, dict)
        and publication_delay.get("expected_valuation_date") == latest["data_date"]
        and publication_delay.get("expected_publication_date") == today.isoformat()
        and publication_delay.get("delay_days") == 1
        and latest.get("official_valuation") is True
        and latest.get("valuation_completeness_passed") is True
    )
    if late_publication_current:
        warnings.append("late_official_valuation_publication_delay_days:1")
    if performance_age_days > 0 and not late_publication_current:
        stale_warning = "latest_performance_stale_calendar_days:" f"{performance_age_days}"
        warnings.append(stale_warning)
        status_gaps.append(stale_warning)
    if (
        latest.get("official_valuation") is False
        or latest.get("valuation_completeness_passed") is False
        or str(latest.get("valuation_status") or "").startswith("BLOCKED")
    ):
        valuation_warning = "latest_current_valuation_incomplete:" + str(
            latest.get("valuation_status") or "UNKNOWN"
        )
        warnings.append(valuation_warning)
        status_gaps.append(valuation_warning)
    if latest_seen != latest["record"]:
        unusable_warning = f"newer_unusable_record_exists:{latest_seen}"
        warnings.append(unusable_warning)
        status_gaps.append(unusable_warning)
    if rejected:
        warnings.append(f"current_holdings_records_rejected:{len(rejected)}")
    if legacy_history_count:
        legacy_warning = (
            "historical_performance_legacy_exact_bytes_without_declared_sha:"
            f"{legacy_history_count}"
        )
        warnings.append(legacy_warning)
        status_gaps.append(legacy_warning)
    if historical_rejected:
        warnings.append("historical_performance_records_rejected:" f"{len(historical_rejected)}")
    benchmark_tail_warnings = []
    for factor_id, series in sorted(benchmark_series.items()):
        if series["missing_dates"]:
            warning = (
                "benchmark_relative_as_of_prior_date:"
                + factor_id
                + ":"
                + series["selected_dates"][-1]
                + ":missing="
                + ",".join(series["missing_dates"])
            )
            warnings.append(warning)
            status_gaps.append(warning)
            benchmark_tail_warnings.append(warning)
    status = "PARTIAL" if status_gaps else "FRESH"

    performance_points: list[dict[str, Any]] = []
    for index, row in enumerate(unitized):

        def benchmark_point(series: Mapping[str, Any], prefix: str) -> dict[str, Any]:
            if index >= len(series["nav"]):
                return {
                    prefix + "nav": None,
                    prefix + "cumulative_return": None,
                    prefix + "benchmark_coverage": "unavailable",
                    prefix + "benchmark_value_date": None,
                }
            return {
                prefix + "nav": series["nav"][index],
                prefix + "cumulative_return": series["nav"][index] / series["nav"][0] - 1.0,
                prefix + "benchmark_coverage": series["rows"][index]["coverage"],
                prefix + "benchmark_value_date": series["rows"][index]["value_date"],
            }

        csi300_values = benchmark_point(benchmark_series["CSI300"], "csi300_")
        point = {
            "date": row["date"],
            "record": row["record"],
            "total_value": row["total_value"],
            "excluded_external_flow": row["excluded_external_flow"],
            "adjusted_total_value": row["adjusted_total_value"],
            "portfolio_unit_nav": row["unit_nav"],
            "portfolio_cumulative_return": row["unit_nav"] - 1.0,
            "csi300_nav": csi300_values["csi300_nav"],
            "csi300_cumulative_return": csi300_values["csi300_cumulative_return"],
            "cumulative_excess_return": (
                None
                if csi300_values["csi300_nav"] is None
                else row["unit_nav"] - csi300_values["csi300_nav"]
            ),
            "benchmark_coverage": csi300_values["csi300_benchmark_coverage"],
            "benchmark_value_date": csi300_values["csi300_benchmark_value_date"],
            "evidence_status": row["evidence_status"],
            "risk_free_annual_yield": aligned_risk_free[index]["annual_yield"],
            "risk_free_coverage": aligned_risk_free[index]["coverage"],
            "risk_free_value_date": aligned_risk_free[index]["value_date"],
        }
        for spec in BENCHMARK_SPECS[1:]:
            prefix = spec["point_prefix"] + "_"
            point.update(benchmark_point(benchmark_series[spec["id"]], prefix))
        performance_points.append(point)

    benchmark_payload = [
        {
            "id": spec["id"],
            "name": spec["name"],
            "ts_code": spec["ts_code"],
            "source_path": benchmark_artifact.relative_path,
            "source_sha256": benchmark_artifact.sha256,
            "start_date": required_dates[0],
            "end_date": benchmark_series[spec["id"]]["selected_dates"][-1],
            "return": benchmark_series[spec["id"]]["return"],
            "excess_return": (
                portfolio_nav[len(benchmark_series[spec["id"]]["nav"]) - 1] / portfolio_nav[0]
                - 1.0
                - benchmark_series[spec["id"]]["return"]
            ),
            "max_drawdown": benchmark_series[spec["id"]]["max_drawdown"],
            "missing_dates": benchmark_series[spec["id"]]["missing_dates"],
            "coverage": [
                *[row["coverage"] for row in benchmark_series[spec["id"]]["rows"]],
                *["unavailable"] * len(benchmark_series[spec["id"]]["missing_dates"]),
            ],
        }
        for spec in BENCHMARK_SPECS
    ]

    source_refs: list[dict[str, str]] = []
    seen_refs: set[tuple[str, str]] = set()
    for record in valid:
        for ref in record.get("_physical_source_refs", record["source_refs"]):
            identity = (ref["path"], ref["sha256"])
            if identity not in seen_refs:
                source_refs.append(ref)
                seen_refs.add(identity)
    for record in historical_records:
        for ref in record.get("_physical_source_refs", record["source_refs"]):
            identity = (ref["path"], ref["sha256"])
            if identity not in seen_refs:
                source_refs.append(ref)
                seen_refs.add(identity)
    if history_integrity_artifact is not None:
        source_refs.append(
            {
                "path": history_integrity_artifact.relative_path,
                "sha256": history_integrity_artifact.sha256,
            }
        )
    for artifact in catalog_artifacts:
        identity = (artifact.relative_path, artifact.sha256)
        if identity not in seen_refs:
            source_refs.append({"path": artifact.relative_path, "sha256": artifact.sha256})
            seen_refs.add(identity)
    source_refs.append(
        {
            "path": benchmark_artifact.relative_path,
            "sha256": benchmark_artifact.sha256,
        }
    )
    source_refs.append(
        {
            "path": risk_free_artifact.relative_path,
            "sha256": risk_free_artifact.sha256,
        }
    )

    for position in latest_view["positions"]:
        position["price_date"] = position.get("price_date") or latest["data_date"]
        position["evidence_status"] = "HASH_BOUND_EFFECTIVE_LEDGER"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "market": MARKET,
        "strategy_label": STRATEGY,
        "strategy_id_kind": "HISTORICAL_DISPLAY_LABEL_NOT_V17_CANONICAL_ID",
        "read_only": True,
        "authority_flags": dict(AUTHORITY_FLAGS),
        "latest_record_seen": latest_seen,
        "latest_valid_record": latest["record"],
        "previous_valid_record": previous["record"],
        "latest_data_date": latest["data_date"],
        "data_age_calendar_days": data_age_days,
        "current_evidence": {
            key: latest[key]
            for key in (
                "manifest_path",
                "manifest_sha256",
                "manual_manifest_path",
                "manual_manifest_sha256",
                "ledger_path",
                "ledger_sha256",
                "pnl_path",
                "pnl_sha256",
                "financial_state_sha256",
                "execution_status",
                "execution_kind",
                "valuation_status",
                "official_valuation",
                "valuation_completeness_passed",
                "price_basis",
            )
        },
        "previous_evidence": {
            key: previous[key]
            for key in (
                "manifest_path",
                "manifest_sha256",
                "manual_manifest_path",
                "manual_manifest_sha256",
                "ledger_path",
                "ledger_sha256",
                "pnl_path",
                "pnl_sha256",
                "financial_state_sha256",
                "execution_status",
                "execution_kind",
            )
        },
        "history": {
            "archive_start_record": historical_records[0]["record"],
            "archive_start_date": unitized[0]["date"],
            "first_pnl_record": next(
                (
                    record["record"]
                    for record in historical_records
                    if record.get("pnl_path") is not None
                ),
                None,
            ),
            "first_pnl_date": next(
                (
                    record["valuation_date"]
                    for record in historical_records
                    if record.get("pnl_path") is not None
                ),
                None,
            ),
            "latest_performance_record": unitized[-1]["record"],
            "latest_performance_date": unitized[-1]["date"],
            "included_record_count": len(historical_records),
            "performance_point_count": len(unitized),
            "legacy_exact_byte_record_count": legacy_history_count,
            "dashboard_integrity_registry_record_count": (history_registry_bound_count),
            "hash_bound_current_record_count": sum(
                record["evidence_status"] == "HASH_BOUND_CURRENT_CLOSURE"
                for record in historical_records
            ),
            "funding_events": funding_events,
            "net_external_flow": sum(event["amount"] for event in funding_events),
            "rejected_record_count": len(historical_rejected),
            "rejected_record_reason_counts": dict(sorted(historical_rejection_counts.items())),
            "rejected_record_samples": historical_rejected[-12:],
            "evidence_status": (
                "CANONICAL_PERFORMANCE_CLOSURE"
                if canonical_performance_points is not None
                else (
                    "PARTIAL_LEGACY_EXACT_BYTES_NO_DECLARED_SHA"
                    if legacy_history_count
                    else (
                        "DASHBOARD_POST_HOC_SHA_REGISTRY_BOUND"
                        if history_registry_bound_count
                        else "HASH_BOUND_CURRENT_CLOSURE_ONLY"
                    )
                )
            ),
            "baseline_manifest_path": historical_records[0]["manifest_path"],
            "baseline_manifest_sha256": historical_records[0]["manifest_sha256"],
            "baseline_ledger_path": historical_records[0]["ledger_path"],
            "baseline_ledger_sha256": historical_records[0]["ledger_sha256"],
        },
        "positions": latest_view["positions"],
        "changes": changes,
        "portfolio": {
            "cash": portfolio_cash,
            "market_value": latest["accounting"]["market_value_after"],
            "total_value": portfolio_total,
            "cash_weight": cash_weight,
            "gross_exposure": gross,
            "portfolio_pnl": (
                performance_adjusted_total - unitized[0]["performance_initial_capital"]
            ),
            "current_unrealized_pnl": current_unrealized,
            "latest_record_realized_pnl_from_rebalance": latest["accounting"][
                "realized_pnl_from_rebalance"
            ],
            "cumulative_realized_pnl": None,
            "realized_pnl_evidence_status": "UNKNOWN",
            "performance_initial_capital": unitized[0]["performance_initial_capital"],
            "excluded_external_flow": latest_economic_point["excluded_external_flow"],
            "adjusted_total_value": performance_adjusted_total,
            "cumulative_profit_excluding_external_flow": (
                performance_adjusted_total - unitized[0]["performance_initial_capital"]
            ),
            "cumulative_return": cumulative_return,
            "current_valuation_status": latest.get("valuation_status"),
            "latest_record_interval_return": (portfolio_nav[-1] / portfolio_nav[-2] - 1.0),
            "max_drawdown": _max_drawdown(portfolio_nav),
            "latest_interval_turnover": _ledger_quantity_turnover(
                changes,
                latest_view,
                previous_view,
                current_total_value=portfolio_total,
                previous_total_value=previous_portfolio_total,
            ),
            "return_method": (
                LEGACY_RETURN_METHOD
                if owner_corrected_initial_capital_path or canonical_performance_points is None
                else CANONICAL_RETURN_METHOD
            ),
            "gross_or_net": "UNKNOWN",
            "fee_basis": "UNKNOWN",
            "performance_start_date": unitized[0]["date"],
            "performance_end_date": unitized[-1]["date"],
            "performance_points": performance_points,
        },
        "benchmarks": benchmark_payload,
        "risk_free": {
            "name": "中国1年期国债收益率",
            "tenor": RISK_FREE_TENOR,
            "source_system": RISK_FREE_SOURCE,
            "source_url": RISK_FREE_SOURCE_URL,
            "source_path": risk_free_artifact.relative_path,
            "source_sha256": risk_free_artifact.sha256,
            "start_date": required_dates[0],
            "end_date": required_dates[-1],
            "latest_annual_yield": aligned_risk_free[-1]["annual_yield"],
            "day_count": "ACT/365",
            "alignment": "interval_start_previous_published_workday",
            "missing_dates": [],
            "coverage": [row["coverage"] for row in aligned_risk_free],
        },
        "concentration": {
            "top1_equity_weight": top1,
            "top3_equity_weight": top3,
            "equity_hhi": hhi,
            "holding_count": len(latest["positions"]),
            "thesis_status_counts": {
                value: sum(1 for row in latest["positions"] if row["thesis_status"] == value)
                for value in sorted({row["thesis_status"] for row in latest["positions"]})
            },
        },
        "risks": [
            {
                "code": "RECORDED_PRICE_STALE",
                "severity": "HIGH" if data_age_days > 5 else "MEDIUM",
                "detail": f"持仓价格与估值数据日期为 {latest['data_date']}，不是当前行情。",
            },
            {
                "code": "EQUITY_CONCENTRATION",
                "severity": "HIGH" if top3 >= 0.8 else "MEDIUM",
                "detail": f"权益仓前三大权重 {top3:.2%}，权益 HHI {hhi:.4f}。",
            },
            {
                "code": "THESIS_STATUS_PRESSURE",
                "severity": "HIGH",
                "detail": "有效 ledger 中多数持仓的 thesis_status 为降级观察。",
            },
            {
                "code": "FEE_AND_REALIZED_PNL_UNKNOWN",
                "severity": "MEDIUM",
                "detail": "费用毛净口径与累计已实现盈亏缺少 hash-bound 分项证据。",
            },
        ]
        + (
            [
                {
                    "code": "LEGACY_HISTORY_EVIDENCE_PARTIAL",
                    "severity": "MEDIUM",
                    "detail": (
                        f"自归档起点纳入 {legacy_history_count} 条旧记录；"
                        "文件已 exact-byte 绑定，但旧 manifest 未声明 SHA，"
                        "不得作为当前持仓权威。"
                    ),
                }
            ]
            if legacy_history_count
            else []
        ),
        "i1_research": None,
        "i1_display_status": "NOT_DISPLAYED_NO_EXACT_HASH_BOUND_I1_ARTIFACT",
        "blockers": [],
        "warnings": sorted(set(warnings)),
        "valid_record_count": len(valid),
        "rejected_record_count": len(rejected),
        "rejected_record_reason_counts": dict(sorted(rejection_counts.items())),
        "rejected_record_samples": rejected[-12:],
        "source_refs": sorted(source_refs, key=lambda item: (item["path"], item["sha256"])),
    }
    payload["content_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_bundle_shape(bundle: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(bundle, dict):
        return ["bundle_not_object"]
    if bundle.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if bundle.get("status") not in {"FRESH", "PARTIAL", "BLOCKED"}:
        errors.append("status_invalid")
    if bundle.get("market") != MARKET or bundle.get("strategy_label") != STRATEGY:
        errors.append("identity_invalid")
    if bundle.get("read_only") is not True:
        errors.append("read_only_invalid")
    flags = bundle.get("authority_flags")
    if flags != AUTHORITY_FLAGS:
        errors.append("authority_flags_invalid")
    public_redacted = bundle.get("public_redacted") is True
    if public_redacted and (
        bundle.get("positions") != []
        or bundle.get("changes") != []
        or bundle.get("source_refs") != []
    ):
        errors.append("public_redaction_invalid")
    if bundle.get("status") in {"FRESH", "PARTIAL"}:
        if bundle.get("blockers") != []:
            errors.append("usable_bundle_has_blockers")
        positions = bundle.get("positions")
        position_nav_weights_valid = isinstance(positions, list) and all(
            isinstance(row, dict)
            and isinstance(row.get("nav_weight"), (int, float))
            and math.isfinite(float(row["nav_weight"]))
            for row in positions
        )
        if not positions and not public_redacted:
            errors.append("usable_bundle_has_no_positions")
        elif not position_nav_weights_valid:
            errors.append("position_nav_weight_invalid")
        changes = bundle.get("changes")
        if not isinstance(changes, list) or (not public_redacted and not changes):
            errors.append("holding_changes_invalid")
        elif not public_redacted:
            change_symbols: set[str] = set()
            base_change_number_fields = (
                "previous_shares",
                "current_shares",
                "share_delta",
                "nav_weight_delta",
                "equity_weight_delta",
            )
            market_value_fields = (
                "previous_market_value",
                "current_market_value",
                "market_value_delta",
            )
            for change in changes:
                if (
                    not isinstance(change, dict)
                    or not isinstance(change.get("name"), str)
                    or not change["name"]
                    or not isinstance(change.get("symbol"), str)
                    or not SYMBOL_RE.fullmatch(change["symbol"])
                    or change["symbol"] in change_symbols
                    or change.get("change_type")
                    not in {"NEW", "INCREASED", "REDUCED", "CLOSED", "UNCHANGED"}
                ):
                    errors.append("holding_change_identity_invalid")
                    continue
                change_symbols.add(change["symbol"])
                market_value_presence = [key in change for key in market_value_fields]
                if any(market_value_presence) and not all(market_value_presence):
                    errors.append("holding_change_market_value_group_invalid")
                    continue
                if any(not _finite_number(change.get(key)) for key in base_change_number_fields):
                    errors.append("holding_change_values_invalid")
                    continue
                if all(market_value_presence) and any(
                    not _finite_number(change.get(key)) for key in market_value_fields
                ):
                    errors.append("holding_change_values_invalid")
                    continue
                previous_shares = float(change["previous_shares"])
                current_shares = float(change["current_shares"])
                if previous_shares == 0 and current_shares > 0:
                    expected_change_type = "NEW"
                elif current_shares == 0 and previous_shares > 0:
                    expected_change_type = "CLOSED"
                elif current_shares > previous_shares:
                    expected_change_type = "INCREASED"
                elif current_shares < previous_shares:
                    expected_change_type = "REDUCED"
                else:
                    expected_change_type = "UNCHANGED"
                if (
                    not _almost_equal(
                        float(change["share_delta"]),
                        current_shares - previous_shares,
                        tolerance=1e-9,
                    )
                    or change["change_type"] != expected_change_type
                ):
                    errors.append("holding_change_share_delta_invalid")
                if all(market_value_presence):
                    previous_market_value = float(change["previous_market_value"])
                    current_market_value = float(change["current_market_value"])
                    if not _almost_equal(
                        float(change["market_value_delta"]),
                        current_market_value - previous_market_value,
                        tolerance=0.01,
                    ):
                        errors.append("holding_change_market_value_delta_invalid")
        benchmarks = bundle.get("benchmarks") or []
        expected_benchmarks = {
            spec["id"]: (spec["name"], spec["ts_code"]) for spec in BENCHMARK_SPECS
        }
        actual_benchmarks = {
            row.get("id"): (row.get("name"), row.get("ts_code"))
            for row in benchmarks
            if isinstance(row, dict)
        }
        if actual_benchmarks != expected_benchmarks:
            errors.append("usable_bundle_benchmark_set_invalid")
        elif (
            any(row.get("missing_dates") for row in benchmarks)
            and bundle.get("status") != "PARTIAL"
        ):
            errors.append("benchmark_gaps_require_partial_status")
        risk_free = bundle.get("risk_free")
        if (
            not isinstance(risk_free, dict)
            or risk_free.get("tenor") != RISK_FREE_TENOR
            or risk_free.get("source_system") != RISK_FREE_SOURCE
            or risk_free.get("source_url") != RISK_FREE_SOURCE_URL
            or risk_free.get("day_count") != "ACT/365"
            or risk_free.get("alignment") != "interval_start_previous_published_workday"
            or risk_free.get("missing_dates") != []
            or not isinstance(risk_free.get("latest_annual_yield"), (int, float))
        ):
            errors.append("usable_bundle_risk_free_invalid")
        history = bundle.get("history")
        portfolio = bundle.get("portfolio")
        if not isinstance(history, dict):
            errors.append("usable_bundle_has_no_history")
        elif not isinstance(portfolio, dict):
            errors.append("usable_bundle_has_no_portfolio")
        else:
            points = portfolio.get("performance_points")
            portfolio_numbers = (
                "cash",
                "market_value",
                "total_value",
                "cash_weight",
                "gross_exposure",
                "portfolio_pnl",
                "performance_initial_capital",
                "excluded_external_flow",
                "adjusted_total_value",
                "cumulative_profit_excluding_external_flow",
                "cumulative_return",
            )
            return_method = portfolio.get("return_method")
            canonical_return = return_method == CANONICAL_RETURN_METHOD
            if return_method not in {
                LEGACY_RETURN_METHOD,
                CANONICAL_RETURN_METHOD,
            }:
                errors.append("portfolio_return_method_invalid")
            elif any(
                not isinstance(portfolio.get(key), (int, float))
                or not math.isfinite(float(portfolio[key]))
                for key in portfolio_numbers
            ):
                errors.append("portfolio_return_values_invalid")
            if not isinstance(points, list) or len(points) < 2:
                errors.append("usable_bundle_has_no_performance_history")
            elif any(
                not isinstance(point, dict)
                or any(
                    not isinstance(point.get(key), (int, float))
                    or not math.isfinite(float(point[key]))
                    for key in (
                        "total_value",
                        "excluded_external_flow",
                        "adjusted_total_value",
                        "portfolio_unit_nav",
                        "risk_free_annual_yield",
                    )
                )
                for point in points
            ):
                errors.append("performance_values_invalid")
            elif any(
                not (
                    (
                        all(
                            _finite_number(point.get(key))
                            for key in (
                                "csi300_nav",
                                "csi300_cumulative_return",
                                "cumulative_excess_return",
                            )
                        )
                        and point.get("benchmark_coverage")
                        in {"exact_close", "previous_trading_day_ffill"}
                        and isinstance(point.get("benchmark_value_date"), str)
                    )
                    or (
                        all(
                            point.get(key) is None
                            for key in (
                                "csi300_nav",
                                "csi300_cumulative_return",
                                "cumulative_excess_return",
                            )
                        )
                        and point.get("benchmark_coverage") == "unavailable"
                        and point.get("benchmark_value_date") is None
                    )
                )
                for point in points
            ):
                errors.append("performance_csi300_availability_invalid")
            elif any(
                not (
                    (
                        all(
                            _finite_number(point.get(prefix + key))
                            for key in ("nav", "cumulative_return")
                        )
                        and point.get(prefix + "benchmark_coverage")
                        in {"exact_close", "previous_trading_day_ffill"}
                        and isinstance(point.get(prefix + "benchmark_value_date"), str)
                    )
                    or (
                        all(point.get(prefix + key) is None for key in ("nav", "cumulative_return"))
                        and point.get(prefix + "benchmark_coverage") == "unavailable"
                        and point.get(prefix + "benchmark_value_date") is None
                    )
                )
                for point in points
                for prefix in ("star50_", "chinext_")
            ):
                errors.append("performance_aux_benchmark_availability_invalid")
            elif any(
                (
                    not isinstance(row, dict)
                    or not isinstance(row.get("missing_dates"), list)
                    or not isinstance(row.get("coverage"), list)
                    or len(row["coverage"]) != len(points)
                    or row["coverage"].count("unavailable") != len(row["missing_dates"])
                    or row["missing_dates"]
                    != [point["date"] for point in points[-len(row["missing_dates"]) :]]
                    if row.get("missing_dates")
                    else False
                )
                for row in benchmarks
            ):
                errors.append("benchmark_trailing_gap_contract_invalid")
            elif portfolio.get("performance_start_date") != history.get(
                "archive_start_date"
            ) or points[0].get("record") != history.get("archive_start_record"):
                errors.append("performance_history_start_mismatch")
            elif not _almost_equal(
                float(points[0]["adjusted_total_value"]),
                float(portfolio["performance_initial_capital"]),
            ) or not _almost_equal(float(points[0]["portfolio_unit_nav"]), 1.0):
                errors.append("performance_initial_capital_mismatch")
            elif any(
                not _almost_equal(
                    float(point["adjusted_total_value"]),
                    (
                        float(point["portfolio_unit_nav"])
                        * float(portfolio["performance_initial_capital"])
                        if canonical_return
                        else float(point["total_value"]) - float(point["excluded_external_flow"])
                    ),
                )
                for point in points
            ):
                errors.append("performance_external_flow_exclusion_invalid")
            elif not _almost_equal(
                float(portfolio["cumulative_return"]),
                float(points[-1]["portfolio_unit_nav"]) - 1.0,
            ):
                errors.append("performance_cumulative_return_mismatch")
            elif not _almost_equal(
                float(portfolio.get("cash", math.nan))
                + float(portfolio.get("market_value", math.nan)),
                float(portfolio.get("total_value", math.nan)),
            ):
                errors.append("economic_portfolio_accounting_mismatch")
            elif not canonical_return and not _almost_equal(
                float(portfolio.get("total_value", math.nan)),
                float(portfolio["adjusted_total_value"]),
            ):
                errors.append("economic_portfolio_total_mismatch")
            elif not str(portfolio.get("current_valuation_status") or "").startswith(
                "BLOCKED"
            ) and not _almost_equal(
                float(
                    portfolio.get(
                        ("adjusted_total_value" if canonical_return else "total_value"),
                        math.nan,
                    )
                ),
                float(points[-1]["adjusted_total_value"]),
            ):
                errors.append("economic_portfolio_performance_total_mismatch")
            elif not _almost_equal(
                float(portfolio.get("portfolio_pnl", math.nan)),
                float(portfolio[("adjusted_total_value" if canonical_return else "total_value")])
                - float(portfolio["performance_initial_capital"]),
            ):
                errors.append("economic_portfolio_pnl_mismatch")
            elif not _almost_equal(
                float(portfolio.get("cash_weight", math.nan))
                + float(portfolio.get("gross_exposure", math.nan)),
                0.0 if bundle.get("public_redacted") is True else 1.0,
            ):
                errors.append("economic_portfolio_weight_mismatch")
            elif not _almost_equal(
                (
                    sum(float(row["nav_weight"]) for row in positions)
                    if position_nav_weights_valid
                    else math.nan
                ),
                float(portfolio.get("gross_exposure", math.nan)),
            ):
                errors.append("position_nav_weight_total_mismatch")
    if bundle.get("i1_research") is None and bundle.get("i1_display_status") != (
        "NOT_DISPLAYED_NO_EXACT_HASH_BOUND_I1_ARTIFACT"
    ):
        errors.append("i1_absence_status_invalid")
    content_sha = bundle.get("content_sha256")
    if not isinstance(content_sha, str) or not SHA256_RE.fullmatch(content_sha):
        errors.append("content_sha256_invalid")
    else:
        without_hash = dict(bundle)
        without_hash.pop("content_sha256", None)
        if sha256_bytes(canonical_json_bytes(without_hash)) != content_sha:
            errors.append("content_sha256_mismatch")
    return errors


def verify_source_refs(bundle: dict[str, Any], project_root: Path) -> list[str]:
    errors: list[str] = []
    refs = bundle.get("source_refs")
    if not isinstance(refs, list) or not refs:
        return ["source_refs_missing"]
    current_record_dirs = {
        Path(evidence.get("manifest_path", "")).parent.as_posix()
        for evidence in (
            bundle.get("current_evidence"),
            bundle.get("previous_evidence"),
        )
        if isinstance(evidence, dict)
    }
    seen: set[str] = set()
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            errors.append(f"source_ref_shape_invalid:{index}")
            continue
        relative_path = ref.get("path")
        declared_sha = ref.get("sha256")
        if (
            not isinstance(relative_path, str)
            or relative_path.startswith("/")
            or ".." in Path(relative_path).parts
        ):
            errors.append(f"source_ref_path_invalid:{index}")
            continue
        if relative_path in seen:
            errors.append(f"source_ref_duplicate:{relative_path}")
            continue
        seen.add(relative_path)
        if (
            "results/strategy_record_archives/CN/"
            "aggressive_tech_manufacturing/monthly/v1/" in relative_path
            and relative_path.endswith((".tar.zst", ".tzst", ".zst"))
        ):
            try:
                archive_path = project_root / relative_path
                _stable_stream_ref(
                    {
                        "path": relative_path,
                        "sha256": declared_sha,
                        "bytes": archive_path.lstat().st_size,
                        "media_type": "application/zstd",
                    },
                    project_root,
                    label="source_ref_archive",
                )
            except (DashboardInputError, OSError) as exc:
                errors.append(str(exc))
            continue
        try:
            artifact = stable_read(project_root / relative_path, project_root)
        except DashboardInputError as exc:
            errors.append(str(exc))
            continue
        if declared_sha != artifact.sha256:
            errors.append(f"source_ref_sha_mismatch:{relative_path}")
            continue
        if Path(relative_path).name == "backfill_provenance.json":
            try:
                manifest = load_json(
                    stable_read(
                        artifact.path.parent / "manifest.json",
                        project_root,
                    )
                )
                source_record = (
                    manifest.get("source_record") if isinstance(manifest, dict) else None
                )
                _validate_transaction_backfill_provenance(
                    artifact.path.parent,
                    project_root,
                    source_record,
                    verify_history_source=(
                        artifact.path.parent.relative_to(project_root).as_posix()
                        in current_record_dirs
                    ),
                )
            except DashboardInputError as exc:
                errors.append(str(exc))
    return errors
