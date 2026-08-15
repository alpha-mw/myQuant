"""Canonical performance closure for the governed CN strategy-record store.

This module deliberately has no legacy-history reader.  The one migration
adapter accepts only normalized economic values embedded in the registered
catalog projection.  It never resolves projection ``source_refs`` and it never
opens a record ledger.  New performance generations are immutable, decimal
Parquet closures bound by a manifest and an owner declaration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Final
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from .store import (
    StrategyRecordConflict,
    StrategyRecordStoreError,
    canonical_json_bytes,
    regular_file_sha256,
)


PERFORMANCE_MANIFEST_SCHEMA: Final = "myquant.strategy_performance_manifest.v1"
PERFORMANCE_OWNER_DECLARATION_SCHEMA: Final = (
    "myquant.strategy_performance_owner_declaration.v1"
)
PERFORMANCE_CASH_FLOW_SCHEMA: Final = "myquant.strategy_performance_cash_flow.v1"
PERFORMANCE_HISTORY_REF_SCHEMA: Final = "myquant.strategy_performance_history_ref.v1"
PERFORMANCE_SERIES_SCHEMA: Final = "myquant.strategy_performance_series.v1"
PERFORMANCE_MIGRATION_RECEIPT_SCHEMA: Final = (
    "myquant.strategy_performance_migration_candidate_receipt.v1"
)

HISTORICAL_LABEL: Final = "aggressive_tech_manufacturing"
CANONICAL_STRATEGY_ID: Final = "cn-aggressive-tech-manufacturing"
PERFORMANCE_INITIAL_CAPITAL: Final = Decimal("1000000.0000")
MONEY_QUANTUM: Final = Decimal("0.0001")
UNIT_QUANTUM: Final = Decimal("0.000000000001")
CENT_TOLERANCE: Final = Decimal("0.01")
UNIT_TOLERANCE: Final = Decimal("0.000000000001")
MAX_PERFORMANCE_JSON_BYTES: Final = 4 * 1024 * 1024
MAX_PERFORMANCE_PARQUET_BYTES: Final = 64 * 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RECORD_ID = re.compile(r"^(?P<day>[0-9]{8})_(?P<clock>[0-9]{4})$")
_UTC_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


def _payload_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StrategyRecordStoreError("performance value is not canonical JSON") from exc


def semantic_sha256(value: Any) -> str:
    """Return a semantic SHA over canonical JSON without a terminal newline."""

    return hashlib.sha256(_payload_bytes(value)).hexdigest()


def seal_semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("semantic_sha256", None)
    body["semantic_sha256"] = semantic_sha256(body)
    return body


def validate_semantic(value: Mapping[str, Any], *, label: str) -> None:
    observed = value.get("semantic_sha256")
    if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
        raise StrategyRecordStoreError(f"{label} semantic_sha256 is invalid")
    body = dict(value)
    del body["semantic_sha256"]
    if observed != semantic_sha256(body):
        raise StrategyRecordStoreError(f"{label} semantic_sha256 mismatch")


def _require_sha(value: Any, *, label: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StrategyRecordStoreError(f"{label} is not a canonical SHA-256")
    return value


def _require_generation(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _GENERATION.fullmatch(value) is None:
        raise StrategyRecordStoreError(f"{label} is invalid")
    return value


def _require_utc_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise StrategyRecordStoreError(f"{label} is not a canonical UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise StrategyRecordStoreError(f"{label} is not a real UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise StrategyRecordStoreError(f"{label} is not canonical")
    return value


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StrategyRecordStoreError(f"{label} is not a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise StrategyRecordStoreError(f"{label} is not a canonical relative path")
    return value


def _decimal(value: Any, *, label: str, quantum: Decimal) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise StrategyRecordStoreError(f"{label} is not a decimal value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise StrategyRecordStoreError(f"{label} is not a decimal value") from exc
    if not result.is_finite():
        raise StrategyRecordStoreError(f"{label} is not finite")
    return result.quantize(quantum, rounding=ROUND_HALF_EVEN)


def money(value: Any, *, label: str) -> Decimal:
    return _decimal(value, label=label, quantum=MONEY_QUANTUM)


def unit_decimal(value: Any, *, label: str) -> Decimal:
    return _decimal(value, label=label, quantum=UNIT_QUANTUM)


def decimal_text(value: Decimal, *, quantum: Decimal) -> str:
    return format(value.quantize(quantum, rounding=ROUND_HALF_EVEN), "f")


def _valuation_at(record_id: str, valuation_date: str) -> str:
    match = _RECORD_ID.fullmatch(record_id)
    if match is None:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:record ID cannot bind valuation time"
        )
    try:
        parsed_day = datetime.strptime(match.group("day"), "%Y%m%d").date()
        parsed_time = datetime.strptime(match.group("clock"), "%H%M").time()
        declared_day = date.fromisoformat(valuation_date)
    except ValueError as exc:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:valuation date is invalid"
        ) from exc
    # The record timestamp is the only permitted deterministic time carrier in
    # the normalized migration input.  Historical corrections may deliberately
    # value an earlier trade date, so the date components need not be equal.
    local = datetime.combine(parsed_day, parsed_time, tzinfo=ZoneInfo("Asia/Shanghai"))
    if declared_day > parsed_day:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:valuation date is after record time"
        )
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalized_funding(value: Any, *, record_id: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:funding is invalid"
        )
    amount = money(value.get("amount"), label=f"{record_id}.funding.amount")
    if amount == 0:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:funding amount is zero"
        )
    return {"amount_cny": decimal_text(amount, quantum=MONEY_QUANTUM)}


def _normalized_correction(value: Any, *, record_id: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:funding correction is invalid"
        )
    reversed_record = value.get("reversed_record")
    if not isinstance(reversed_record, str) or not reversed_record:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:correction target is invalid"
        )
    reversed_amount = money(
        value.get("reversed_amount"),
        label=f"{record_id}.funding_correction.reversed_amount",
    )
    initial = money(
        value.get("initial_capital"),
        label=f"{record_id}.funding_correction.initial_capital",
    )
    if reversed_amount <= 0 or initial != PERFORMANCE_INITIAL_CAPITAL:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:correction economics are invalid"
        )
    return {
        "reversed_record_id": reversed_record,
        "reversed_amount_cny": decimal_text(reversed_amount, quantum=MONEY_QUANTUM),
        "initial_capital_cny": decimal_text(initial, quantum=MONEY_QUANTUM),
    }


def normalize_registered_projection(
    catalog: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    """Normalize the catalog-bound migration source without consulting refs.

    Only the economic fields named in the migration contract are read.  The
    full projection semantic SHA proves which catalog-embedded value was
    frozen, while the normalized SHA proves the exact economic adapter output.
    """

    projection = catalog.get("dashboard_projection")
    if not isinstance(projection, dict):
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:registered projection is absent"
        )
    source_rows = projection.get("historical_records")
    if not isinstance(source_rows, list) or not source_rows:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:historical projection is absent"
        )
    projection_sha = semantic_sha256(projection)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(source_rows):
        if not isinstance(raw, dict):
            raise StrategyRecordStoreError(
                "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:historical row is invalid"
            )
        record_id = raw.get("record")
        valuation_date = raw.get("valuation_date")
        accounting = raw.get("accounting")
        if (
            not isinstance(record_id, str)
            or record_id in seen
            or not isinstance(valuation_date, str)
            or not isinstance(accounting, dict)
        ):
            raise StrategyRecordStoreError(
                "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:required row fields are invalid"
            )
        seen.add(record_id)
        cash = money(accounting.get("cash_after"), label=f"{record_id}.cash_after")
        equity = money(
            accounting.get("market_value_after"),
            label=f"{record_id}.market_value_after",
        )
        nav = money(
            accounting.get("total_value_after"),
            label=f"{record_id}.total_value_after",
        )
        pnl = money(
            accounting.get("portfolio_pnl_after"),
            label=f"{record_id}.portfolio_pnl_after",
        )
        if abs(nav - cash - equity) > CENT_TOLERANCE:
            raise StrategyRecordStoreError(
                "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:NAV accounting does not close"
            )
        capital = money(raw.get("capital_base"), label=f"{record_id}.capital_base")
        evidence = raw.get("evidence_status")
        if evidence is not None and not isinstance(evidence, str):
            raise StrategyRecordStoreError(
                "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:evidence status is invalid"
            )
        normalized.append(
            {
                "source_sequence_no": index + 1,
                "record_id": record_id,
                "valuation_at": _valuation_at(record_id, valuation_date),
                "valuation_date": valuation_date,
                "cash_cny": decimal_text(cash, quantum=MONEY_QUANTUM),
                "equity_market_value_cny": decimal_text(equity, quantum=MONEY_QUANTUM),
                "raw_nav_cny": decimal_text(nav, quantum=MONEY_QUANTUM),
                "portfolio_pnl_cny": decimal_text(pnl, quantum=MONEY_QUANTUM),
                "capital_base_cny": decimal_text(capital, quantum=MONEY_QUANTUM),
                "funding": _normalized_funding(raw.get("funding"), record_id=record_id),
                "funding_correction": _normalized_correction(
                    raw.get("funding_correction"), record_id=record_id
                ),
                "source_evidence_status": evidence or "UNKNOWN",
            }
        )
    return normalized, projection_sha, semantic_sha256(normalized)


def _catalog_evidence_by_record(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    rows = catalog.get("records")
    if not isinstance(rows, list):
        raise StrategyRecordStoreError("catalog records are unavailable")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("record_id"), str):
            continue
        result[row["record_id"]] = {
            "manual_manifest_sha256": _optional_sha(row.get("manual_manifest_sha256")),
            "ledger_parquet_sha256": _optional_sha(row.get("ledger_sha256")),
            "financial_state_sha256": _optional_sha(row.get("financial_state_sha256")),
        }
    return result


def _optional_sha(value: Any) -> str | None:
    return value if isinstance(value, str) and _SHA256.fullmatch(value) else None


def build_seed_rows(
    normalized: Sequence[Mapping[str, Any]],
    *,
    catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reproduce the registered owner correction before daily collapse."""

    if not normalized:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:no normalized rows"
        )
    evidence_by_record = _catalog_evidence_by_record(catalog)
    funding_by_record: dict[str, Decimal] = {}
    corrected: set[str] = set()
    cumulative_flow = Decimal("0.0000")
    timeline: list[dict[str, Any]] = []
    for raw in normalized:
        record_id = str(raw["record_id"])
        funding = raw.get("funding")
        if funding is not None:
            if not isinstance(funding, Mapping):
                raise StrategyRecordStoreError("normalized funding is invalid")
            amount = money(funding.get("amount_cny"), label=f"{record_id}.funding")
            if record_id in funding_by_record:
                raise StrategyRecordStoreError("funding record is duplicated")
            funding_by_record[record_id] = amount
            cumulative_flow += amount
        correction = raw.get("funding_correction")
        if correction is not None:
            if not isinstance(correction, Mapping):
                raise StrategyRecordStoreError("normalized correction is invalid")
            target = correction.get("reversed_record_id")
            amount = money(
                correction.get("reversed_amount_cny"), label=f"{record_id}.correction"
            )
            if target not in funding_by_record or funding_by_record[target] != amount:
                raise StrategyRecordStoreError(
                    "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:correction target does not close"
                )
            if target not in corrected:
                cumulative_flow -= amount
                corrected.add(str(target))
                if cumulative_flow < 0:
                    raise StrategyRecordStoreError(
                        "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:correction exceeds flow"
                    )
        raw_nav = money(raw.get("raw_nav_cny"), label=f"{record_id}.raw_nav")
        adjusted_nav = (raw_nav - cumulative_flow).quantize(MONEY_QUANTUM)
        if adjusted_nav <= 0:
            raise StrategyRecordStoreError(
                "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:adjusted NAV is not positive"
            )
        evidence = evidence_by_record.get(record_id, {})
        timeline.append(
            {
                "record_id": record_id,
                "valuation_at": str(raw["valuation_at"]),
                "valuation_date": str(raw["valuation_date"]),
                "cash_cny": money(raw.get("cash_cny"), label=f"{record_id}.cash"),
                "equity_market_value_cny": money(
                    raw.get("equity_market_value_cny"), label=f"{record_id}.equity"
                ),
                "raw_nav_cny": raw_nav,
                "portfolio_pnl_cny": money(
                    raw.get("portfolio_pnl_cny"), label=f"{record_id}.pnl"
                ),
                "excluded_external_flow_cny": cumulative_flow.quantize(MONEY_QUANTUM),
                "adjusted_nav_cny": adjusted_nav,
                "unit_count": unit_decimal(
                    PERFORMANCE_INITIAL_CAPITAL, label=f"{record_id}.unit_count"
                ),
                "unit_nav": unit_decimal(
                    adjusted_nav / PERFORMANCE_INITIAL_CAPITAL,
                    label=f"{record_id}.unit_nav",
                ),
                "evidence_kind": "OWNER_DECLARED_REGISTERED_PROJECTION_MIGRATION",
                "manual_manifest_sha256": evidence.get("manual_manifest_sha256"),
                "ledger_parquet_sha256": evidence.get("ledger_parquet_sha256"),
                "financial_state_sha256": evidence.get("financial_state_sha256"),
            }
        )

    by_day: dict[str, dict[str, Any]] = {}
    day_order: list[str] = []
    for row in timeline:
        day = row["valuation_date"]
        if day not in by_day:
            day_order.append(day)
        by_day[day] = row
    rows = [by_day[day] for day in day_order]
    if rows[0]["adjusted_nav_cny"] != PERFORMANCE_INITIAL_CAPITAL:
        raise StrategyRecordStoreError(
            "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:initial adjusted NAV is not CNY 1m"
        )
    previous_unit: Decimal | None = None
    initial_unit = rows[0]["unit_nav"]
    high_water = initial_unit
    previous_at: str | None = None
    for index, row in enumerate(rows, start=1):
        current = row["unit_nav"]
        if previous_at is not None and row["valuation_at"] <= previous_at:
            raise StrategyRecordStoreError(
                "CANONICAL_PERFORMANCE_SOURCE_UNAVAILABLE:valuation time is not increasing"
            )
        interval = (
            Decimal("0")
            if previous_unit is None
            else current / previous_unit - Decimal("1")
        )
        cumulative = current / initial_unit - Decimal("1")
        high_water = max(high_water, current)
        drawdown = current / high_water - Decimal("1")
        row["sequence_no"] = index
        row["interval_return"] = unit_decimal(interval, label="interval_return")
        row["cumulative_return"] = unit_decimal(cumulative, label="cumulative_return")
        row["drawdown"] = unit_decimal(drawdown, label="drawdown")
        previous_unit = current
        previous_at = row["valuation_at"]
    validate_performance_rows(rows)
    return rows


def validate_performance_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise StrategyRecordStoreError("performance series is empty")
    seen_records: set[str] = set()
    seen_dates: set[str] = set()
    previous_at: str | None = None
    previous_unit: Decimal | None = None
    initial_unit: Decimal | None = None
    high_water: Decimal | None = None
    for index, row in enumerate(rows, start=1):
        if row.get("sequence_no") != index:
            raise StrategyRecordStoreError("performance sequence_no mismatch")
        record_id = row.get("record_id")
        valuation_date = row.get("valuation_date")
        valuation_at = row.get("valuation_at")
        if (
            not isinstance(record_id, str)
            or record_id in seen_records
            or not isinstance(valuation_date, str)
            or valuation_date in seen_dates
            or not isinstance(valuation_at, str)
        ):
            raise StrategyRecordStoreError("performance identity is invalid or duplicated")
        _require_utc_timestamp(valuation_at, label="performance valuation_at")
        if previous_at is not None and valuation_at <= previous_at:
            raise StrategyRecordStoreError("performance valuation_at is not increasing")
        cash = money(row.get("cash_cny"), label="performance cash")
        equity = money(row.get("equity_market_value_cny"), label="performance equity")
        raw_nav = money(row.get("raw_nav_cny"), label="performance raw NAV")
        excluded = money(
            row.get("excluded_external_flow_cny"), label="performance external flow"
        )
        adjusted = money(row.get("adjusted_nav_cny"), label="performance adjusted NAV")
        units = unit_decimal(row.get("unit_count"), label="performance unit count")
        unit_nav = unit_decimal(row.get("unit_nav"), label="performance unit NAV")
        evidence_kind = row.get("evidence_kind")
        if evidence_kind not in {
            "OWNER_DECLARED_REGISTERED_PROJECTION_MIGRATION",
            "REGISTERED_CORRECTION",
            "REGISTERED_APPLIED_TRADES",
            "REGISTERED_OFFICIAL_FINANCIAL_STATE",
        }:
            raise StrategyRecordStoreError("performance evidence kind is invalid")
        if abs(raw_nav - cash - equity) > CENT_TOLERANCE:
            raise StrategyRecordStoreError("performance NAV accounting mismatch")
        if adjusted != (raw_nav - excluded).quantize(MONEY_QUANTUM):
            raise StrategyRecordStoreError("performance adjusted NAV mismatch")
        if units <= 0 or unit_nav <= 0:
            raise StrategyRecordStoreError("performance units must be positive")
        identity_nav = raw_nav if str(evidence_kind).startswith("REGISTERED_") else adjusted
        if abs(identity_nav / units - unit_nav) > UNIT_TOLERANCE:
            raise StrategyRecordStoreError("performance unit NAV identity mismatch")
        if initial_unit is None:
            initial_unit = unit_nav
            high_water = unit_nav
        assert high_water is not None
        expected_interval = (
            Decimal("0")
            if previous_unit is None
            else unit_nav / previous_unit - Decimal("1")
        )
        expected_cumulative = unit_nav / initial_unit - Decimal("1")
        high_water = max(high_water, unit_nav)
        expected_drawdown = unit_nav / high_water - Decimal("1")
        for key, expected in (
            ("interval_return", expected_interval),
            ("cumulative_return", expected_cumulative),
            ("drawdown", expected_drawdown),
        ):
            observed = unit_decimal(row.get(key), label=f"performance {key}")
            if abs(observed - expected) > UNIT_TOLERANCE:
                raise StrategyRecordStoreError(f"performance {key} mismatch")
        for key in (
            "manual_manifest_sha256",
            "ledger_parquet_sha256",
            "financial_state_sha256",
        ):
            _require_sha(row.get(key), label=f"performance {key}", nullable=True)
        seen_records.add(record_id)
        seen_dates.add(valuation_date)
        previous_at = valuation_at
        previous_unit = unit_nav


PERFORMANCE_ARROW_SCHEMA: Final = pa.schema(
    [
        pa.field("sequence_no", pa.int64(), nullable=False),
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("valuation_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("valuation_date", pa.date32(), nullable=False),
        pa.field("cash_cny", pa.decimal128(20, 4), nullable=False),
        pa.field("equity_market_value_cny", pa.decimal128(20, 4), nullable=False),
        pa.field("raw_nav_cny", pa.decimal128(20, 4), nullable=False),
        pa.field("portfolio_pnl_cny", pa.decimal128(20, 4), nullable=False),
        pa.field("excluded_external_flow_cny", pa.decimal128(20, 4), nullable=False),
        pa.field("adjusted_nav_cny", pa.decimal128(20, 4), nullable=False),
        pa.field("unit_count", pa.decimal128(28, 12), nullable=False),
        pa.field("unit_nav", pa.decimal128(28, 12), nullable=False),
        pa.field("interval_return", pa.decimal128(28, 12), nullable=False),
        pa.field("cumulative_return", pa.decimal128(28, 12), nullable=False),
        pa.field("drawdown", pa.decimal128(28, 12), nullable=False),
        pa.field("evidence_kind", pa.string(), nullable=False),
        pa.field("manual_manifest_sha256", pa.string(), nullable=True),
        pa.field("ledger_parquet_sha256", pa.string(), nullable=True),
        pa.field("financial_state_sha256", pa.string(), nullable=True),
    ],
    metadata={b"schema_id": PERFORMANCE_SERIES_SCHEMA.encode("ascii")},
)


def _arrow_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    values: dict[str, list[Any]] = {field.name: [] for field in PERFORMANCE_ARROW_SCHEMA}
    for row in rows:
        for field in PERFORMANCE_ARROW_SCHEMA:
            value = row.get(field.name)
            if field.name == "valuation_at":
                value = datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            elif field.name == "valuation_date":
                value = date.fromisoformat(str(value))
            values[field.name].append(value)
    return pa.Table.from_pydict(values, schema=PERFORMANCE_ARROW_SCHEMA)


def write_deterministic_parquet(
    rows: Sequence[Mapping[str, Any]], path: Path, *, replay_path: Path | None = None
) -> tuple[str, int]:
    validate_performance_rows(rows)
    table = _arrow_table(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    def write_one(target: Path) -> None:
        if target.exists():
            raise StrategyRecordConflict("performance candidate already exists")
        pq.write_table(
            table,
            target,
            version="2.6",
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=True,
            data_page_version="2.0",
            row_group_size=max(1, len(rows)),
            store_schema=True,
        )
        os.chmod(target, 0o600)
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    write_one(path)
    if replay_path is not None:
        write_one(replay_path)
        left = path.read_bytes()
        right = replay_path.read_bytes()
        if left != right:
            raise StrategyRecordStoreError("performance Parquet replay is not deterministic")
    digest, size = regular_file_sha256(
        path, label="performance Parquet candidate"
    )
    if size > MAX_PERFORMANCE_PARQUET_BYTES:
        raise StrategyRecordStoreError("performance Parquet exceeds byte budget")
    return digest, size


def read_performance_parquet(path: Path) -> list[dict[str, Any]]:
    digest, size = regular_file_sha256(path, label="performance Parquet")
    del digest
    if size > MAX_PERFORMANCE_PARQUET_BYTES:
        raise StrategyRecordStoreError("performance Parquet exceeds byte budget")
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise StrategyRecordStoreError("performance Parquet is unreadable") from exc
    if table.schema != PERFORMANCE_ARROW_SCHEMA:
        raise StrategyRecordStoreError("performance Parquet schema mismatch")
    rows: list[dict[str, Any]] = []
    for raw in table.to_pylist():
        row = dict(raw)
        at = row["valuation_at"]
        if not isinstance(at, datetime):
            raise StrategyRecordStoreError("performance valuation timestamp is invalid")
        row["valuation_at"] = at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        day = row["valuation_date"]
        if not isinstance(day, date):
            raise StrategyRecordStoreError("performance valuation date is invalid")
        row["valuation_date"] = day.isoformat()
        rows.append(row)
    validate_performance_rows(rows)
    return rows


def _artifact_ref(
    *, schema_id: str, path: str, sha256: str, byte_length: int
) -> dict[str, Any]:
    return {
        "schema_id": schema_id,
        "path": _safe_relative(path, label="performance artifact path"),
        "sha256": _require_sha(sha256, label="performance artifact SHA"),
        "bytes": byte_length,
    }


def build_owner_declaration(
    *,
    performance_generation_id: str,
    declared_at: str,
    series_path: str,
    series_sha256: str,
    series_bytes: int,
    source_pointer_sha256: str,
    source_catalog_sha256: str,
    normalized_projection_semantic_sha256: str,
    declared_by: str = "maxwell",
) -> dict[str, Any]:
    if declared_by != "maxwell":
        raise StrategyRecordStoreError("performance owner must be maxwell")
    return seal_semantic(
        {
            "schema_id": PERFORMANCE_OWNER_DECLARATION_SCHEMA,
            "historical_label": HISTORICAL_LABEL,
            "canonical_strategy_id": CANONICAL_STRATEGY_ID,
            "performance_generation_id": _require_generation(
                performance_generation_id, label="performance generation"
            ),
            "approved_series": _artifact_ref(
                schema_id=PERFORMANCE_SERIES_SCHEMA,
                path=series_path,
                sha256=series_sha256,
                byte_length=series_bytes,
            ),
            "source_store_pointer_sha256": _require_sha(
                source_pointer_sha256, label="source pointer SHA"
            ),
            "source_catalog_sha256": _require_sha(
                source_catalog_sha256, label="source catalog SHA"
            ),
            "normalized_projection_semantic_sha256": _require_sha(
                normalized_projection_semantic_sha256,
                label="normalized projection SHA",
            ),
            "declared_by": declared_by,
            "declared_at": _require_utc_timestamp(declared_at, label="declared_at"),
            "authority_kind": "owner_declaration",
            "approval_scope": "exact_candidate_bytes_only",
            "v17_activation_authority": False,
            "broker_authority": False,
            "order_authority": False,
            "execution_authority": False,
            "trade_authority": False,
        }
    )


def build_manifest(
    *,
    performance_generation_id: str,
    generated_at: str,
    identity_path: str,
    identity_sha256: str,
    parent_performance_manifest_sha256: str | None,
    source_pointer_sha256: str,
    source_catalog_generation_id: str,
    source_catalog_sha256: str,
    dashboard_projection_sha256: str,
    normalized_projection_semantic_sha256: str,
    series_path: str,
    series_sha256: str,
    series_bytes: int,
    owner_path: str,
    owner_sha256: str,
    owner_bytes: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    max_drawdown = min(unit_decimal(row["drawdown"], label="drawdown") for row in rows)
    return seal_semantic(
        {
            "schema_id": PERFORMANCE_MANIFEST_SCHEMA,
            "historical_label": HISTORICAL_LABEL,
            "canonical_strategy_id": CANONICAL_STRATEGY_ID,
            "performance_generation_id": _require_generation(
                performance_generation_id, label="performance generation"
            ),
            "generated_at": _require_utc_timestamp(generated_at, label="generated_at"),
            "identity_declaration": {
                "path": _safe_relative(identity_path, label="identity path"),
                "sha256": _require_sha(identity_sha256, label="identity SHA"),
            },
            "parent_performance_manifest_sha256": _require_sha(
                parent_performance_manifest_sha256,
                label="parent performance manifest SHA",
                nullable=True,
            ),
            "source_store_pointer_sha256": _require_sha(
                source_pointer_sha256, label="source pointer SHA"
            ),
            "source_catalog_generation_id": _require_generation(
                source_catalog_generation_id, label="source catalog generation"
            ),
            "source_catalog_sha256": _require_sha(
                source_catalog_sha256, label="source catalog SHA"
            ),
            "source_dashboard_projection_sha256": _require_sha(
                dashboard_projection_sha256, label="dashboard projection SHA"
            ),
            "normalized_projection_semantic_sha256": _require_sha(
                normalized_projection_semantic_sha256,
                label="normalized projection SHA",
            ),
            "historical_seed_method": (
                "owner_corrected_initial_capital_external_flow_excluded_v1"
            ),
            "extension_method": "flow_neutral_unitization_v1",
            "performance_initial_capital_cny": decimal_text(
                PERFORMANCE_INITIAL_CAPITAL, quantum=MONEY_QUANTUM
            ),
            "seed_start_date": first["valuation_date"],
            "seed_end_date": last["valuation_date"],
            "row_count": len(rows),
            "final_record_id": last["record_id"],
            "first_raw_nav_cny": decimal_text(first["raw_nav_cny"], quantum=MONEY_QUANTUM),
            "last_raw_nav_cny": decimal_text(last["raw_nav_cny"], quantum=MONEY_QUANTUM),
            "final_net_external_flow_cny": decimal_text(
                last["excluded_external_flow_cny"], quantum=MONEY_QUANTUM
            ),
            "cumulative_return": decimal_text(
                last["cumulative_return"], quantum=UNIT_QUANTUM
            ),
            "max_drawdown": decimal_text(max_drawdown, quantum=UNIT_QUANTUM),
            "series": _artifact_ref(
                schema_id=PERFORMANCE_SERIES_SCHEMA,
                path=series_path,
                sha256=series_sha256,
                byte_length=series_bytes,
            ),
            "owner_declaration": _artifact_ref(
                schema_id=PERFORMANCE_OWNER_DECLARATION_SCHEMA,
                path=owner_path,
                sha256=owner_sha256,
                byte_length=owner_bytes,
            ),
            "authority_kind": "owner_declaration",
            "v17_activation_authority": False,
            "broker_authority": False,
            "order_authority": False,
            "execution_authority": False,
            "trade_authority": False,
        }
    )


def build_performance_history_ref(
    *, manifest: Mapping[str, Any], manifest_sha256: str, manifest_bytes: int
) -> dict[str, Any]:
    generation = _require_generation(
        manifest.get("performance_generation_id"), label="performance generation"
    )
    prefix = f"_record_store/performance/{generation}"
    owner = manifest.get("owner_declaration")
    series = manifest.get("series")
    if not isinstance(owner, dict) or not isinstance(series, dict):
        raise StrategyRecordStoreError("performance manifest refs are invalid")
    return {
        "schema_id": PERFORMANCE_HISTORY_REF_SCHEMA,
        "performance_generation_id": generation,
        "manifest": _artifact_ref(
            schema_id=PERFORMANCE_MANIFEST_SCHEMA,
            path=f"{prefix}/manifest.v1.json",
            sha256=manifest_sha256,
            byte_length=manifest_bytes,
        ),
        "series": dict(series),
        "owner_declaration": dict(owner),
    }


def _validate_ref_shape(
    value: Any,
    *,
    label: str,
    schema_id: str,
    expected_path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema_id", "path", "sha256", "bytes"}:
        raise StrategyRecordStoreError(f"{label} ref shape is invalid")
    if value.get("schema_id") != schema_id:
        raise StrategyRecordStoreError(f"{label} schema is invalid")
    if _safe_relative(value.get("path"), label=f"{label} path") != expected_path:
        raise StrategyRecordStoreError(f"{label} path is not generation-bound")
    _require_sha(value.get("sha256"), label=f"{label} SHA")
    if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        raise StrategyRecordStoreError(f"{label} byte length is invalid")
    return dict(value)


def validate_performance_history_ref_shape(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_id",
        "performance_generation_id",
        "manifest",
        "series",
        "owner_declaration",
    }:
        raise StrategyRecordStoreError("performance_history_ref shape is invalid")
    if value.get("schema_id") != PERFORMANCE_HISTORY_REF_SCHEMA:
        raise StrategyRecordStoreError("performance_history_ref schema is invalid")
    generation = _require_generation(
        value.get("performance_generation_id"), label="performance generation"
    )
    prefix = f"_record_store/performance/{generation}"
    _validate_ref_shape(
        value.get("manifest"),
        label="performance manifest",
        schema_id=PERFORMANCE_MANIFEST_SCHEMA,
        expected_path=f"{prefix}/manifest.v1.json",
    )
    _validate_ref_shape(
        value.get("series"),
        label="performance series",
        schema_id=PERFORMANCE_SERIES_SCHEMA,
        expected_path=f"{prefix}/series.parquet",
    )
    _validate_ref_shape(
        value.get("owner_declaration"),
        label="performance owner declaration",
        schema_id=PERFORMANCE_OWNER_DECLARATION_SCHEMA,
        expected_path=f"{prefix}/owner_declaration.v1.json",
    )
    return dict(value)


def _read_exact_json(
    path: Path, *, expected_sha: str, expected_bytes: int, label: str
) -> dict[str, Any]:
    digest, size = regular_file_sha256(path, expected_bytes=expected_bytes, label=label)
    if digest != expected_sha:
        raise StrategyRecordStoreError(f"{label} SHA-256 mismatch")
    if size > MAX_PERFORMANCE_JSON_BYTES:
        raise StrategyRecordStoreError(f"{label} exceeds byte budget")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyRecordStoreError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise StrategyRecordStoreError(f"{label} bytes are not canonical JSON")
    validate_semantic(value, label=label)
    return value


def validate_owner_declaration(
    value: Mapping[str, Any], *, expected_generation: str, expected_series: Mapping[str, Any]
) -> None:
    validate_semantic(value, label="performance owner declaration")
    if (
        value.get("schema_id") != PERFORMANCE_OWNER_DECLARATION_SCHEMA
        or value.get("historical_label") != HISTORICAL_LABEL
        or value.get("canonical_strategy_id") != CANONICAL_STRATEGY_ID
        or value.get("performance_generation_id") != expected_generation
        or value.get("approved_series") != expected_series
        or value.get("declared_by") != "maxwell"
        or value.get("authority_kind") != "owner_declaration"
        or value.get("approval_scope") != "exact_candidate_bytes_only"
    ):
        raise StrategyRecordStoreError("performance owner declaration mismatch")
    _require_utc_timestamp(value.get("declared_at"), label="owner declared_at")
    for field in (
        "v17_activation_authority",
        "broker_authority",
        "order_authority",
        "execution_authority",
        "trade_authority",
    ):
        if value.get(field) is not False:
            raise StrategyRecordStoreError("performance owner declaration claims authority")


def validate_manifest(
    value: Mapping[str, Any], *, expected_ref: Mapping[str, Any]
) -> None:
    validate_semantic(value, label="performance manifest")
    generation = expected_ref["performance_generation_id"]
    if (
        value.get("schema_id") != PERFORMANCE_MANIFEST_SCHEMA
        or value.get("historical_label") != HISTORICAL_LABEL
        or value.get("canonical_strategy_id") != CANONICAL_STRATEGY_ID
        or value.get("performance_generation_id") != generation
        or value.get("series") != expected_ref["series"]
        or value.get("owner_declaration") != expected_ref["owner_declaration"]
        or value.get("historical_seed_method")
        != "owner_corrected_initial_capital_external_flow_excluded_v1"
        or value.get("extension_method") != "flow_neutral_unitization_v1"
        or value.get("performance_initial_capital_cny") != "1000000.0000"
    ):
        raise StrategyRecordStoreError("performance manifest contract mismatch")
    for field in (
        "v17_activation_authority",
        "broker_authority",
        "order_authority",
        "execution_authority",
        "trade_authority",
    ):
        if value.get(field) is not False:
            raise StrategyRecordStoreError("performance manifest claims authority")


def load_performance_history(
    record_root: Path, ref_value: Mapping[str, Any]
) -> dict[str, Any]:
    ref = validate_performance_history_ref_shape(ref_value)
    manifest_ref = ref["manifest"]
    series_ref = ref["series"]
    owner_ref = ref["owner_declaration"]
    manifest = _read_exact_json(
        record_root / manifest_ref["path"],
        expected_sha=manifest_ref["sha256"],
        expected_bytes=manifest_ref["bytes"],
        label="performance manifest",
    )
    validate_manifest(manifest, expected_ref=ref)
    owner = _read_exact_json(
        record_root / owner_ref["path"],
        expected_sha=owner_ref["sha256"],
        expected_bytes=owner_ref["bytes"],
        label="performance owner declaration",
    )
    validate_owner_declaration(
        owner,
        expected_generation=ref["performance_generation_id"],
        expected_series=series_ref,
    )
    series_path = record_root / series_ref["path"]
    digest, size = regular_file_sha256(
        series_path,
        expected_bytes=series_ref["bytes"],
        label="performance series",
    )
    if digest != series_ref["sha256"]:
        raise StrategyRecordStoreError("performance series SHA-256 mismatch")
    rows = read_performance_parquet(series_path)
    if manifest.get("row_count") != len(rows):
        raise StrategyRecordStoreError("performance manifest row_count mismatch")
    first = rows[0]
    last = rows[-1]
    expected_values = {
        "seed_start_date": first["valuation_date"],
        "seed_end_date": last["valuation_date"],
        "final_record_id": last["record_id"],
        "first_raw_nav_cny": decimal_text(first["raw_nav_cny"], quantum=MONEY_QUANTUM),
        "last_raw_nav_cny": decimal_text(last["raw_nav_cny"], quantum=MONEY_QUANTUM),
        "final_net_external_flow_cny": decimal_text(
            last["excluded_external_flow_cny"], quantum=MONEY_QUANTUM
        ),
        "cumulative_return": decimal_text(last["cumulative_return"], quantum=UNIT_QUANTUM),
        "max_drawdown": decimal_text(
            min(row["drawdown"] for row in rows), quantum=UNIT_QUANTUM
        ),
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise StrategyRecordStoreError(f"performance manifest {key} mismatch")
    return {
        "ref": ref,
        "manifest": manifest,
        "owner_declaration": owner,
        "rows": rows,
        "series_sha256": digest,
        "series_bytes": size,
    }


def build_lineage_index(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build a deterministic effective lineage from the registered projection.

    Legacy records occasionally name the same source after a correction.  The
    original source is retained, while ``supersedes_record_id`` serializes any
    deviation from the preceding registered record.  Effective-parent
    traversal therefore remains unique without rewriting the source evidence.
    """

    projection = catalog.get("dashboard_projection")
    records = catalog.get("records")
    if not isinstance(projection, dict) or not isinstance(records, list):
        raise StrategyRecordStoreError("lineage source is unavailable")
    projected: dict[str, dict[str, Any]] = {}
    for key in ("historical_records", "valid_records"):
        rows = projection.get(key)
        if not isinstance(rows, list):
            raise StrategyRecordStoreError("lineage projection is invalid")
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("record"), str):
                projected[row["record"]] = row
    stored = {
        row["record_id"]: row
        for row in records
        if isinstance(row, dict)
        and isinstance(row.get("record_id"), str)
        and row.get("state", row.get("storage_state")) in {"ONLINE", "ARCHIVED"}
        and row.get("record_id") in projected
    }
    if not stored:
        raise StrategyRecordStoreError("lineage has no governed records")
    ordered_ids = sorted(stored)
    result: list[dict[str, Any]] = []
    previous: str | None = None
    for record_id in ordered_ids:
        source = projected[record_id]
        declared_parent = source.get("source_record")
        if declared_parent is not None and declared_parent not in stored:
            declared_parent = None
        supersedes = source.get("supersedes_record")
        if supersedes is not None and supersedes not in stored:
            raise StrategyRecordStoreError("lineage supersedes parent is absent")
        if previous is not None and declared_parent != previous:
            supersedes = previous
        valuation_date = source.get("valuation_date") or source.get("data_date")
        if not isinstance(valuation_date, str):
            match = _RECORD_ID.fullmatch(record_id)
            if match is None:
                raise StrategyRecordStoreError("lineage valuation date is absent")
            valuation_date = datetime.strptime(match.group("day"), "%Y%m%d").date().isoformat()
        execution_kind = str(source.get("execution_kind") or "").lower()
        execution_status = str(source.get("execution_status") or "").lower()
        if "applied" in execution_kind or "filled" in execution_status:
            execution_class = "APPLIED_TRADES"
        elif "carry_forward" in execution_kind or "no_action" in execution_status:
            execution_class = "NO_TRADE"
        else:
            execution_class = "UNKNOWN_BLOCKED"
        if supersedes is not None or "correction" in execution_status or source.get(
            "funding_correction"
        ) is not None:
            publication_class = "CORRECTION"
        elif source.get("official_valuation") is True or execution_class == "APPLIED_TRADES":
            publication_class = "OFFICIAL_FINANCIAL_STATE"
        else:
            publication_class = "RECEIPT_ONLY_NO_ACTION"
        stored_row = stored[record_id]
        result.append(
            {
                "record_id": record_id,
                "source_record_id": declared_parent,
                "supersedes_record_id": supersedes,
                "valuation_date": valuation_date,
                "execution_class": execution_class,
                "publication_class": publication_class,
                "storage_state": stored_row.get("state", stored_row.get("storage_state")),
                "manifest_ref": _lineage_ref(
                    stored_row.get("manifest_path"), stored_row.get("manifest_sha256")
                ),
                "manual_manifest_ref": _lineage_ref(
                    stored_row.get("manual_manifest_path"),
                    stored_row.get("manual_manifest_sha256"),
                ),
                "effective_ledger_ref": _lineage_ledger_ref(
                    stored_row.get("ledger_path"), stored_row.get("ledger_sha256")
                ),
                "financial_state_sha256": _optional_sha(
                    stored_row.get("financial_state_sha256")
                ),
                "ledger_parquet_sha256": (
                    _optional_sha(stored_row.get("ledger_sha256"))
                    if _is_parquet_ledger_path(stored_row.get("ledger_path"))
                    else None
                ),
            }
        )
        previous = record_id
    validate_lineage_index(result, active_record_id=catalog.get("active_record_id"))
    return result


def _lineage_ref(path: Any, sha256: Any) -> dict[str, str] | None:
    if path is None and sha256 is None:
        return None
    if not isinstance(path, str):
        raise StrategyRecordStoreError("lineage artifact path is invalid")
    return {
        "path": _safe_relative(path, label="lineage artifact path"),
        "sha256": str(_require_sha(sha256, label="lineage artifact SHA")),
    }


def _is_parquet_ledger_path(path: Any) -> bool:
    return (
        isinstance(path, str)
        and PurePosixPath(path).name == "ledger_after_manual_switch.parquet"
    )


def _lineage_ledger_ref(path: Any, sha256: Any) -> dict[str, str] | None:
    # Historical migration rows without the registered Parquet ledger closure
    # remain explicitly unavailable.  A legacy alternate ledger is never
    # carried into the v3 authority graph.
    if not _is_parquet_ledger_path(path):
        return None
    return _lineage_ref(path, sha256)


def validate_lineage_index(
    value: Any, *, active_record_id: Any
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise StrategyRecordStoreError("lineage_index is invalid")
    by_id: dict[str, dict[str, Any]] = {}
    effective_children: dict[str, list[str]] = {}
    required = {
        "record_id",
        "source_record_id",
        "supersedes_record_id",
        "valuation_date",
        "execution_class",
        "publication_class",
        "storage_state",
        "manifest_ref",
        "manual_manifest_ref",
        "effective_ledger_ref",
        "financial_state_sha256",
        "ledger_parquet_sha256",
    }
    for row in value:
        if not isinstance(row, dict) or set(row) != required:
            raise StrategyRecordStoreError("lineage row shape is invalid")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in by_id:
            raise StrategyRecordStoreError("lineage record_id is invalid or duplicated")
        if row.get("execution_class") not in {
            "APPLIED_TRADES",
            "NO_TRADE",
            "UNKNOWN_BLOCKED",
        }:
            raise StrategyRecordStoreError("lineage execution_class is invalid")
        if row.get("publication_class") not in {
            "OFFICIAL_FINANCIAL_STATE",
            "RECEIPT_ONLY_NO_ACTION",
            "CORRECTION",
        }:
            raise StrategyRecordStoreError("lineage publication_class is invalid")
        if row.get("storage_state") not in {"ONLINE", "ARCHIVED"}:
            raise StrategyRecordStoreError("lineage storage_state is invalid")
        try:
            date.fromisoformat(str(row.get("valuation_date")))
        except ValueError as exc:
            raise StrategyRecordStoreError("lineage valuation_date is invalid") from exc
        for key in ("financial_state_sha256", "ledger_parquet_sha256"):
            _require_sha(row.get(key), label=f"lineage {key}", nullable=True)
        by_id[record_id] = row
    for record_id, row in by_id.items():
        source = row["source_record_id"]
        supersedes = row["supersedes_record_id"]
        for label, parent in (("source", source), ("supersedes", supersedes)):
            if parent is not None and parent not in by_id:
                raise StrategyRecordStoreError(f"lineage {label} parent is absent")
            if parent == record_id:
                raise StrategyRecordStoreError("lineage self-cycle is invalid")
        effective = supersedes if supersedes is not None else source
        if effective is not None:
            effective_children.setdefault(effective, []).append(record_id)
    if any(len(children) > 1 for children in effective_children.values()):
        raise StrategyRecordStoreError("lineage effective parent has a fork")
    if not isinstance(active_record_id, str) or active_record_id not in by_id:
        raise StrategyRecordStoreError("lineage active record is absent")
    chain: list[str] = []
    visited: set[str] = set()
    current: str | None = active_record_id
    while current is not None:
        if current in visited:
            raise StrategyRecordStoreError("lineage contains a cycle")
        visited.add(current)
        chain.append(current)
        row = by_id[current]
        current = row["supersedes_record_id"] or row["source_record_id"]
    chain.reverse()
    return tuple(chain)


def validate_cash_flow_artifact(
    value: Mapping[str, Any],
    *,
    pre_row: Mapping[str, Any],
    post_row: Mapping[str, Any],
    pre_positions: Mapping[str, int],
    post_positions: Mapping[str, int],
) -> Decimal:
    validate_semantic(value, label="performance cash-flow declaration")
    required = {
        "schema_id",
        "event_id",
        "historical_label",
        "canonical_strategy_id",
        "effective_at",
        "shanghai_trade_date",
        "direction",
        "amount_cny",
        "timing_convention",
        "pre_flow_record_id",
        "pre_flow_nav_cny",
        "pre_flow_financial_state_sha256",
        "post_flow_record_id",
        "post_flow_nav_cny",
        "post_flow_financial_state_sha256",
        "matching_manual_manifest_path",
        "matching_manual_manifest_sha256",
        "declared_by",
        "declared_at",
        "authority_kind",
        "v17_activation_authority",
        "broker_authority",
        "order_authority",
        "execution_authority",
        "trade_authority",
        "semantic_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_id") != PERFORMANCE_CASH_FLOW_SCHEMA
        or value.get("historical_label") != HISTORICAL_LABEL
        or value.get("canonical_strategy_id") != CANONICAL_STRATEGY_ID
        or value.get("timing_convention")
        != "between_exact_pre_and_post_financial_states"
        or value.get("authority_kind") != "owner_declaration"
        or value.get("declared_by") != "maxwell"
    ):
        raise StrategyRecordStoreError("cash-flow declaration contract mismatch")
    event_id = value.get("event_id")
    if not isinstance(event_id, str) or _GENERATION.fullmatch(event_id) is None:
        raise StrategyRecordStoreError("cash-flow event_id is invalid")
    effective_at = _require_utc_timestamp(
        value.get("effective_at"), label="cash-flow effective_at"
    )
    _require_utc_timestamp(value.get("declared_at"), label="cash-flow declared_at")
    try:
        shanghai_day = date.fromisoformat(str(value.get("shanghai_trade_date")))
    except ValueError as exc:
        raise StrategyRecordStoreError("cash-flow Shanghai trade date is invalid") from exc
    effective_day = (
        datetime.strptime(effective_at, "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc)
        .astimezone(ZoneInfo("Asia/Shanghai"))
        .date()
    )
    if shanghai_day != effective_day:
        raise StrategyRecordStoreError("cash-flow effective date is inconsistent")
    direction = value.get("direction")
    amount = money(value.get("amount_cny"), label="cash-flow amount")
    if value.get("amount_cny") != decimal_text(amount, quantum=MONEY_QUANTUM):
        raise StrategyRecordStoreError("cash-flow amount is not scale-4 canonical")
    if direction not in {"CONTRIBUTION", "REDEMPTION"} or amount == 0:
        raise StrategyRecordStoreError("cash-flow direction or amount is invalid")
    if (direction == "CONTRIBUTION" and amount < 0) or (
        direction == "REDEMPTION" and amount > 0
    ):
        raise StrategyRecordStoreError("cash-flow sign does not match direction")
    if dict(pre_positions) != dict(post_positions):
        raise StrategyRecordStoreError("cash-flow position quantities changed")
    pre_nav = money(pre_row.get("raw_nav_cny"), label="pre-flow NAV")
    post_nav = money(post_row.get("raw_nav_cny"), label="post-flow NAV")
    if (
        value.get("pre_flow_nav_cny")
        != decimal_text(pre_nav, quantum=MONEY_QUANTUM)
        or value.get("post_flow_nav_cny")
        != decimal_text(post_nav, quantum=MONEY_QUANTUM)
    ):
        raise StrategyRecordStoreError("cash-flow declared NAV binding mismatch")
    if abs(post_nav - pre_nav - amount) > CENT_TOLERANCE:
        raise StrategyRecordStoreError("cash-flow NAV bridge does not close")
    for side, row in (("pre", pre_row), ("post", post_row)):
        expected_record = value.get(f"{side}_flow_record_id")
        expected_sha = value.get(f"{side}_flow_financial_state_sha256")
        if row.get("record_id") != expected_record or row.get(
            "financial_state_sha256"
        ) != expected_sha:
            raise StrategyRecordStoreError("cash-flow financial-state binding mismatch")
    manual_path = _safe_relative(
        value.get("matching_manual_manifest_path"),
        label="cash-flow matching manual manifest path",
    )
    manual_sha = _require_sha(
        value.get("matching_manual_manifest_sha256"),
        label="cash-flow matching manual manifest SHA",
    )
    if (
        post_row.get("manual_manifest_path") != manual_path
        or post_row.get("manual_manifest_sha256") != manual_sha
    ):
        raise StrategyRecordStoreError("cash-flow manual manifest binding mismatch")
    for field in (
        "v17_activation_authority",
        "broker_authority",
        "order_authority",
        "execution_authority",
        "trade_authority",
    ):
        if value.get(field) is not False:
            raise StrategyRecordStoreError("cash-flow declaration claims authority")
    return amount


def extend_performance_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    *,
    strict_record: Mapping[str, Any],
    manual_manifest_sha256: str,
    ledger_parquet_sha256: str,
    financial_state_sha256: str,
    post_flow_unit_count: Decimal | None = None,
    external_flow_amount: Decimal = Decimal("0.0000"),
    allow_same_date_correction: bool = False,
) -> list[dict[str, Any]]:
    """Append or explicitly replace one registered financial-state row."""

    if not existing_rows:
        raise StrategyRecordStoreError("performance extension has no parent series")
    validate_performance_rows(existing_rows)
    rows = [dict(row) for row in existing_rows]
    record_id = strict_record.get("record")
    valuation_date = strict_record.get("data_date")
    accounting = strict_record.get("accounting")
    if (
        not isinstance(record_id, str)
        or not isinstance(valuation_date, str)
        or not isinstance(accounting, Mapping)
    ):
        raise StrategyRecordStoreError("new performance financial state is invalid")
    if any(row["record_id"] == record_id for row in rows):
        raise StrategyRecordStoreError("performance record_id is already present")
    cash = money(accounting.get("cash_after"), label="new performance cash")
    equity = money(
        accounting.get("market_value_after"), label="new performance equity"
    )
    raw_nav = money(accounting.get("total_value_after"), label="new performance NAV")
    pnl = money(
        accounting.get("portfolio_pnl_after"), label="new performance P&L"
    )
    if abs(raw_nav - cash - equity) > CENT_TOLERANCE:
        raise StrategyRecordStoreError("new performance NAV accounting mismatch")
    previous = rows[-1]
    flow_amount = money(external_flow_amount, label="new performance external flow")
    if (post_flow_unit_count is None) != (flow_amount == 0):
        raise StrategyRecordStoreError(
            "new performance cash-flow amount and unit count are inconsistent"
        )
    units = (
        unit_decimal(post_flow_unit_count, label="new performance unit count")
        if post_flow_unit_count is not None
        else unit_decimal(previous["unit_count"], label="parent performance unit count")
    )
    unit_nav = unit_decimal(raw_nav / units, label="new performance unit NAV")
    cumulative_flow = money(
        previous["excluded_external_flow_cny"],
        label="parent cumulative external flow",
    ) + flow_amount
    execution_kind = str(strict_record.get("execution_kind") or "").lower()
    execution_status = str(strict_record.get("execution_status") or "").lower()
    evidence_kind = (
        "REGISTERED_CORRECTION"
        if allow_same_date_correction
        else (
            "REGISTERED_APPLIED_TRADES"
            if "applied" in execution_kind or "filled" in execution_status
            else "REGISTERED_OFFICIAL_FINANCIAL_STATE"
        )
    )
    new_row: dict[str, Any] = {
        "sequence_no": len(rows) + 1,
        "record_id": record_id,
        "valuation_at": _valuation_at(record_id, valuation_date),
        "valuation_date": valuation_date,
        "cash_cny": cash,
        "equity_market_value_cny": equity,
        "raw_nav_cny": raw_nav,
        "portfolio_pnl_cny": pnl,
        "excluded_external_flow_cny": money(
            cumulative_flow, label="cumulative external flow"
        ),
        "adjusted_nav_cny": money(
            raw_nav - cumulative_flow, label="external-flow-excluded adjusted NAV"
        ),
        "unit_count": units,
        "unit_nav": unit_nav,
        "interval_return": Decimal("0.000000000000"),
        "cumulative_return": Decimal("0.000000000000"),
        "drawdown": Decimal("0.000000000000"),
        "evidence_kind": evidence_kind,
        "manual_manifest_sha256": _require_sha(
            manual_manifest_sha256, label="new manual manifest SHA"
        ),
        "ledger_parquet_sha256": _require_sha(
            ledger_parquet_sha256, label="new Parquet ledger SHA"
        ),
        "financial_state_sha256": _require_sha(
            financial_state_sha256, label="new financial-state SHA"
        ),
    }
    if valuation_date < previous["valuation_date"]:
        raise StrategyRecordStoreError("performance valuation date moved backwards")
    if valuation_date == previous["valuation_date"]:
        if not allow_same_date_correction:
            raise StrategyRecordStoreError("SAME_DATE_PERFORMANCE_CONFLICT")
        rows[-1] = new_row
    else:
        rows.append(new_row)
    initial_unit = unit_decimal(rows[0]["unit_nav"], label="initial unit NAV")
    previous_unit: Decimal | None = None
    high_water = initial_unit
    for index, row in enumerate(rows, start=1):
        current = unit_decimal(row["unit_nav"], label="unit NAV")
        interval = (
            Decimal("0")
            if previous_unit is None
            else current / previous_unit - Decimal("1")
        )
        cumulative = current / initial_unit - Decimal("1")
        high_water = max(high_water, current)
        drawdown = current / high_water - Decimal("1")
        row["sequence_no"] = index
        row["interval_return"] = unit_decimal(interval, label="interval return")
        row["cumulative_return"] = unit_decimal(cumulative, label="cumulative return")
        row["drawdown"] = unit_decimal(drawdown, label="drawdown")
        previous_unit = current
    validate_performance_rows(rows)
    return rows


def apply_flow_neutral_unitization(
    *, pre_nav: Decimal, pre_units: Decimal, pre_unit_nav: Decimal, amount: Decimal
) -> tuple[Decimal, Decimal]:
    if pre_nav <= 0 or pre_units <= 0 or pre_unit_nav <= 0 or amount == 0:
        raise StrategyRecordStoreError("cash-flow unitization inputs are invalid")
    post_nav = pre_nav + amount
    units_delta = unit_decimal(amount / pre_unit_nav, label="cash-flow units delta")
    post_units = unit_decimal(pre_units + units_delta, label="post-flow units")
    if post_nav <= 0 or post_units <= 0:
        raise StrategyRecordStoreError("cash-flow leaves non-positive NAV or units")
    post_unit_nav = unit_decimal(post_nav / post_units, label="post-flow unit NAV")
    if abs(post_unit_nav - pre_unit_nav) > UNIT_TOLERANCE:
        raise StrategyRecordStoreError("cash-flow itself changes unit NAV")
    return post_units, post_unit_nav


def assert_private_tmp(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    private_tmp = Path("/private/tmp").resolve(strict=True)
    if resolved == private_tmp or private_tmp not in resolved.parents:
        raise StrategyRecordStoreError("performance candidate must be under /private/tmp")
    for existing in (candidate for candidate in [*path.parents] if candidate.exists()):
        metadata = os.lstat(existing)
        if stat.S_ISLNK(metadata.st_mode):
            raise StrategyRecordStoreError("candidate output ancestry contains a symlink")
        if existing == private_tmp:
            break
    return resolved


def immutable_write(path: Path, raw: bytes, *, max_bytes: int) -> str:
    if len(raw) > max_bytes:
        raise StrategyRecordStoreError("immutable artifact exceeds byte budget")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError:
        existing = path.read_bytes()
        if existing != raw:
            raise StrategyRecordConflict("immutable performance artifact conflict") from None
        return hashlib.sha256(existing).hexdigest()
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StrategyRecordStoreError("short immutable performance write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    digest, size = regular_file_sha256(path, expected_bytes=len(raw), label="immutable artifact")
    if size != len(raw) or path.read_bytes() != raw:
        raise StrategyRecordStoreError("immutable performance readback mismatch")
    return digest


def validate_safe_regular_0600(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise StrategyRecordStoreError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
    ):
        raise StrategyRecordStoreError(f"{label} storage security is invalid")
    return metadata


__all__ = [name for name in globals() if not name.startswith("_")]
