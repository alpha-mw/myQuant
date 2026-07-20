"""Non-executable mark target and strict H00300 total-return evidence."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import date, datetime, timezone
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EvidenceRef,
    EvidenceV2Error,
    decode_f64,
    encode_f64,
    seal_semantic,
    semantic_sha256,
    sha256_bytes,
    validate_semantic_seal,
)
from .schedule import (
    ScheduleAnchorBinding,
    ScheduleAnchorBindingV3,
    validate_schedule_anchor_binding,
    validate_schedule_anchor_binding_v3,
)

TARGET_ID = "CN_20D_MARK_NET_TOTAL_RETURN_EXCESS_VS_CSI300_TRI_V1"
INDEX_MANIFEST_SCHEMA = "csi-index-total-return-manifest.v1"
INDEX_TABLE_SCHEMA = "csi-h00300-total-return.parquet.v1"
TERMINAL_SETTLEMENT_SCHEMA = "terminal-cash-settlement.v1"
TARGET_OUTCOME_SCHEMA = "v16.mark-target-outcome.v2"
COST_EVIDENCE_SCHEMA = "v16.mark-cost-evidence.v2"
STOCK_MARK_EVIDENCE_SCHEMA = "v16.stock-mark-evidence.v2"
STOCK_MARK_TABLE_SCHEMA = "cn-stock-mark-boundaries.parquet.v2"
ADJUSTMENT_FACTOR_EVIDENCE_SCHEMA = "v16.adjustment-factor-evidence.v2"
PIT_MEMBERSHIP_EVIDENCE_SCHEMA = "v16.pit-membership-evidence.v2"
SUSPENSION_EVIDENCE_SCHEMA = "v16.suspension-evidence.v2"
H00300_INSTRUMENT_ID = "H00300.CSI"
H00300_OFFICIAL_CODE = "H00300"
H00300_RETURN_TYPE = "gross_pre_tax_total_return"
H00300_CURRENCY = "CNY"
H00300_SOURCE_SYSTEM = "csindex_official"
INDEX_TABLE_MAX_BYTES = 64 * 1024 * 1024
INDEX_TABLE_MAX_ROWS = 100_000
COST_COMPONENT_ORDER = (
    "buy_commission",
    "sell_commission",
    "sell_stamp_duty",
    "buy_transfer_fee",
    "sell_transfer_fee",
    "buy_slippage",
    "sell_slippage",
    "market_impact",
)

EXPECTED_INDEX_SCHEMA = pa.schema(
    [
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("trade_date", pa.date32(), nullable=False),
        pa.field("close_total_return", pa.float64(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("return_type", pa.string(), nullable=False),
        pa.field("source_system", pa.string(), nullable=False),
        pa.field("source_observed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source_document_sha256", pa.string(), nullable=False),
    ]
)

EXPECTED_STOCK_MARK_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("trade_date", pa.date32(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("source_observed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source_document_sha256", pa.string(), nullable=False),
    ]
)


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not text or text != text.strip() or len(text) > 128:
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    if any(character not in allowed for character in text):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _symbol(value: Any) -> str:
    text = str(value or "")
    if not text or text != text.strip().upper() or len(text) > 32:
        raise EvidenceV2Error("target symbol must be normalized")
    if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in text):
        raise EvidenceV2Error("target symbol must be normalized")
    return text


def _target_sessions(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) != 20:
        raise EvidenceV2Error("20D target requires exactly 20 open sessions")
    sessions = [_iso_date(item, label="target_session") for item in value]
    if sessions != sorted(sessions) or len(sessions) != len(set(sessions)):
        raise EvidenceV2Error("20D target sessions must be strictly ordered and unique")
    return sessions


def _iso_date(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise EvidenceV2Error(f"{label} must be ISO date") from exc
    if parsed.isoformat() != text:
        raise EvidenceV2Error(f"{label} must be canonical ISO date")
    return text


def _utc(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceV2Error(f"{label} must be ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceV2Error(f"{label} must be UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if canonical != text:
        raise EvidenceV2Error(f"{label} must be canonical UTC")
    return text


def _positive_f64(value: Any, *, label: str) -> float:
    number = decode_f64(value, label=label)
    if number <= 0.0:
        raise EvidenceV2Error(f"{label} must be positive")
    return number


@dataclass(frozen=True)
class CostVector:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(COST_COMPONENT_ORDER):
            raise EvidenceV2Error("cost vector must contain exactly eight components")
        if any(not math.isfinite(value) or value < 0.0 for value in self.values):
            raise EvidenceV2Error("cost components must be finite and nonnegative")

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> "CostVector":
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise EvidenceV2Error("cost vector must be a sequence")
        if len(rows) != len(COST_COMPONENT_ORDER):
            raise EvidenceV2Error("cost vector must contain exactly eight components")
        values: list[float] = []
        for expected_name, raw in zip(COST_COMPONENT_ORDER, rows):
            row = _exact(raw, {"name", "value"}, label=f"cost {expected_name}")
            if row["name"] != expected_name:
                raise EvidenceV2Error("cost components are missing, extra, or reordered")
            number = decode_f64(row["value"], label=f"cost.{expected_name}")
            if number < 0.0:
                raise EvidenceV2Error("cost components must be nonnegative")
            values.append(number)
        return cls(tuple(values))

    @property
    def total(self) -> float:
        return math.fsum(self.values)

    def to_rows(self) -> list[dict[str, str]]:
        return [
            {"name": name, "value": encode_f64(value)}
            for name, value in zip(COST_COMPONENT_ORDER, self.values)
        ]


def build_cost_evidence(
    *,
    protocol_attempt_id: str,
    sample_id: str,
    costs: CostVector,
    cost_model_ref: EvidenceRef,
) -> dict[str, Any]:
    return seal_semantic(
        {
            "schema_version": COST_EVIDENCE_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "sample_id": _safe_id(sample_id, label="sample_id"),
            "costs": costs.to_rows(),
            "cost_model_ref": cost_model_ref.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_cost_evidence(value: Mapping[str, Any]) -> tuple[dict[str, Any], CostVector]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "sample_id",
        "costs",
        "cost_model_ref",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="cost evidence")
    if payload["schema_version"] != COST_EVIDENCE_SCHEMA:
        raise EvidenceV2Error("unsupported cost evidence schema")
    _safe_id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _safe_id(payload["sample_id"], label="sample_id")
    if not isinstance(payload["costs"], list):
        raise EvidenceV2Error("cost evidence rows must be a list")
    costs = CostVector.from_rows(payload["costs"])
    EvidenceRef.from_dict(payload["cost_model_ref"])
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("cost evidence must be nonauthorizing")
    return payload, costs


def validate_terminal_settlement(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "symbol",
        "currency",
        "tax_treatment",
        "raw_cash_per_terminal_share",
        "terminal_share_basis",
        "settlement_effective_date",
        "applicable_adj_factor",
        "adj_factor_effective_date",
        "share_basis_verified",
        "official_event_ref",
        "adj_factor_ref",
        "terminal_adjusted_mark",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="terminal settlement")
    if payload["schema_version"] != TERMINAL_SETTLEMENT_SCHEMA:
        raise EvidenceV2Error("unsupported terminal settlement schema")
    symbol = str(payload["symbol"])
    if not symbol or symbol != symbol.strip().upper():
        raise EvidenceV2Error("terminal settlement symbol must be normalized")
    if payload["currency"] != H00300_CURRENCY or payload["tax_treatment"] != "gross_pre_tax":
        raise EvidenceV2Error("terminal settlement currency/tax basis mismatch")
    if payload["terminal_share_basis"] != "terminal_registered_share":
        raise EvidenceV2Error("unsupported terminal settlement share basis")
    if payload["share_basis_verified"] is not True:
        raise EvidenceV2Error("terminal settlement share basis is unverified")
    settlement_date = _iso_date(
        payload["settlement_effective_date"], label="settlement_effective_date"
    )
    factor_date = _iso_date(payload["adj_factor_effective_date"], label="adj_factor_effective_date")
    if factor_date > settlement_date:
        raise EvidenceV2Error("terminal adjustment factor is effective after settlement")
    raw_cash = _positive_f64(
        payload["raw_cash_per_terminal_share"], label="raw_cash_per_terminal_share"
    )
    factor = _positive_f64(payload["applicable_adj_factor"], label="applicable_adj_factor")
    declared = _positive_f64(payload["terminal_adjusted_mark"], label="terminal_adjusted_mark")
    expected = raw_cash * factor
    if encode_f64(declared) != encode_f64(expected):
        raise EvidenceV2Error("terminal adjusted mark recomputation mismatch")
    EvidenceRef.from_dict(payload["official_event_ref"])
    EvidenceRef.from_dict(payload["adj_factor_ref"])
    return payload


def build_terminal_settlement(
    *,
    symbol: str,
    raw_cash_per_terminal_share: float,
    settlement_effective_date: str,
    applicable_adj_factor: float,
    adj_factor_effective_date: str,
    official_event_ref: EvidenceRef,
    adj_factor_ref: EvidenceRef,
) -> dict[str, Any]:
    terminal_mark = float(raw_cash_per_terminal_share) * float(applicable_adj_factor)
    payload = seal_semantic(
        {
            "schema_version": TERMINAL_SETTLEMENT_SCHEMA,
            "symbol": str(symbol).strip().upper(),
            "currency": H00300_CURRENCY,
            "tax_treatment": "gross_pre_tax",
            "raw_cash_per_terminal_share": encode_f64(raw_cash_per_terminal_share),
            "terminal_share_basis": "terminal_registered_share",
            "settlement_effective_date": settlement_effective_date,
            "applicable_adj_factor": encode_f64(applicable_adj_factor),
            "adj_factor_effective_date": adj_factor_effective_date,
            "share_basis_verified": True,
            "official_event_ref": official_event_ref.to_dict(),
            "adj_factor_ref": adj_factor_ref.to_dict(),
            "terminal_adjusted_mark": encode_f64(terminal_mark),
        }
    )
    return validate_terminal_settlement(payload)


@dataclass(frozen=True)
class MarkCandidate:
    exact_mark: float | None
    pit_listed: bool
    authoritative_suspension: bool
    stale_mark: float | None
    terminal_settlement: Mapping[str, Any] | None


def _candidate_to_dict(candidate: MarkCandidate) -> dict[str, Any]:
    def optional_mark(value: float | None) -> str | None:
        return None if value is None else encode_f64(value)

    return {
        "exact_mark": optional_mark(candidate.exact_mark),
        "pit_listed": candidate.pit_listed,
        "authoritative_suspension": candidate.authoritative_suspension,
        "stale_mark": optional_mark(candidate.stale_mark),
        "terminal_settlement": (
            None
            if candidate.terminal_settlement is None
            else validate_terminal_settlement(candidate.terminal_settlement)
        ),
    }


def _candidate_from_dict(value: Any) -> MarkCandidate:
    row = _exact(
        value,
        {
            "exact_mark",
            "pit_listed",
            "authoritative_suspension",
            "stale_mark",
            "terminal_settlement",
        },
        label="mark candidate",
    )
    if not isinstance(row["pit_listed"], bool) or not isinstance(
        row["authoritative_suspension"], bool
    ):
        raise EvidenceV2Error("mark candidate booleans are invalid")

    def optional_mark(raw: Any, *, label: str) -> float | None:
        if raw is None:
            return None
        return _positive_f64(raw, label=label)

    terminal = row["terminal_settlement"]
    return MarkCandidate(
        exact_mark=optional_mark(row["exact_mark"], label="exact_mark"),
        pit_listed=row["pit_listed"],
        authoritative_suspension=row["authoritative_suspension"],
        stale_mark=optional_mark(row["stale_mark"], label="stale_mark"),
        terminal_settlement=(None if terminal is None else validate_terminal_settlement(terminal)),
    )


def validate_stock_mark_parquet(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > INDEX_TABLE_MAX_BYTES:
        raise EvidenceV2Error("stock mark Parquet byte size is invalid")
    try:
        parquet_file = pq.ParquetFile(pa.BufferReader(payload))
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise EvidenceV2Error("stock mark Parquet parse failed") from exc
    if not parquet_file.schema_arrow.equals(EXPECTED_STOCK_MARK_SCHEMA, check_metadata=False):
        raise EvidenceV2Error("stock mark Arrow schema mismatch")
    row_count = parquet_file.metadata.num_rows
    if row_count <= 0 or row_count > INDEX_TABLE_MAX_ROWS:
        raise EvidenceV2Error("stock mark row count is invalid")
    projection = parquet_metadata_projection(parquet_file)
    try:
        table = parquet_file.read()
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise EvidenceV2Error("stock mark Parquet read failed") from exc
    if table.num_rows != row_count or any(
        table[name].null_count != 0 for name in table.column_names
    ):
        raise EvidenceV2Error("stock mark Parquet rows contain nulls or drift")
    normalized: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    for row in table.to_pylist():
        symbol = _symbol(row["symbol"])
        trade_date = _iso_date(row["trade_date"].isoformat(), label="stock trade_date")
        close = float(row["close"])
        observed = row["source_observed_at"]
        if (
            not math.isfinite(close)
            or close <= 0.0
            or observed.tzinfo is None
            or observed.utcoffset() != timezone.utc.utcoffset(observed)
        ):
            raise EvidenceV2Error("stock mark row domain mismatch")
        source_sha = str(row["source_document_sha256"])
        if len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha):
            raise EvidenceV2Error("stock mark source document SHA is invalid")
        keys.append((symbol, trade_date))
        normalized.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "close": encode_f64(close),
                "source_observed_at": observed.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "source_document_sha256": source_sha,
            }
        )
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise EvidenceV2Error("stock mark rows must be strictly ordered and unique")
    return {
        "rows": normalized,
        "row_count": row_count,
        "metadata_projection": projection,
        "parquet_metadata_semantic_sha256": semantic_sha256({"projection": projection}),
    }


def _validate_source_rows(
    value: Any,
    *,
    row_fields: set[str],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise EvidenceV2Error(f"{label} rows must be a nonempty list")
    rows = [_exact(row, row_fields, label=f"{label} row") for row in value]
    keys = [
        (_symbol(row["symbol"]), _iso_date(row["trade_date"], label="trade_date")) for row in rows
    ]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise EvidenceV2Error(f"{label} rows must be strictly ordered and unique")
    return rows


def build_adjustment_factor_evidence(
    *,
    generation_id: str,
    market_table_ref: EvidenceRef,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "symbol": _symbol(row["symbol"]),
                "trade_date": _iso_date(row["trade_date"], label="trade_date"),
                "adj_factor": encode_f64(row["adj_factor"]),
            }
        )
    normalized.sort(key=lambda row: (row["symbol"], row["trade_date"]))
    return seal_semantic(
        {
            "schema_version": ADJUSTMENT_FACTOR_EVIDENCE_SCHEMA,
            "generation_id": _safe_id(generation_id, label="generation_id"),
            "market_table_ref": market_table_ref.to_dict(),
            "rows": normalized,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_adjustment_factor_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "generation_id",
        "market_table_ref",
        "rows",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="adjustment factor evidence")
    if payload["schema_version"] != ADJUSTMENT_FACTOR_EVIDENCE_SCHEMA:
        raise EvidenceV2Error("unsupported adjustment factor evidence schema")
    _safe_id(payload["generation_id"], label="generation_id")
    EvidenceRef.from_dict(payload["market_table_ref"])
    rows = _validate_source_rows(
        payload["rows"],
        row_fields={"symbol", "trade_date", "adj_factor"},
        label="adjustment factor",
    )
    for row in rows:
        if _positive_f64(row["adj_factor"], label="adj_factor") <= 0.0:
            raise EvidenceV2Error("adjustment factor must be positive")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("adjustment factor evidence must be nonauthorizing")
    payload["rows"] = rows
    return payload


def build_pit_membership_evidence(
    *,
    generation_id: str,
    calendar_ref: EvidenceRef,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = [
        {
            "symbol": _symbol(row["symbol"]),
            "trade_date": _iso_date(row["trade_date"], label="trade_date"),
            "pit_listed": row["pit_listed"],
        }
        for row in rows
    ]
    if any(not isinstance(row["pit_listed"], bool) for row in normalized):
        raise EvidenceV2Error("PIT listed values must be boolean")
    normalized.sort(key=lambda row: (row["symbol"], row["trade_date"]))
    return seal_semantic(
        {
            "schema_version": PIT_MEMBERSHIP_EVIDENCE_SCHEMA,
            "generation_id": _safe_id(generation_id, label="generation_id"),
            "calendar_ref": calendar_ref.to_dict(),
            "rows": normalized,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_pit_membership_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "generation_id",
        "calendar_ref",
        "rows",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="PIT membership evidence")
    if payload["schema_version"] != PIT_MEMBERSHIP_EVIDENCE_SCHEMA:
        raise EvidenceV2Error("unsupported PIT membership evidence schema")
    _safe_id(payload["generation_id"], label="generation_id")
    EvidenceRef.from_dict(payload["calendar_ref"])
    rows = _validate_source_rows(
        payload["rows"],
        row_fields={"symbol", "trade_date", "pit_listed"},
        label="PIT membership",
    )
    if any(not isinstance(row["pit_listed"], bool) for row in rows):
        raise EvidenceV2Error("PIT listed values must be boolean")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("PIT membership evidence must be nonauthorizing")
    payload["rows"] = rows
    return payload


def build_suspension_evidence(
    *,
    generation_id: str,
    calendar_ref: EvidenceRef,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = []
    for row in rows:
        suspended = row["authoritative_suspension"]
        stale_date = row.get("stale_trade_date")
        if not isinstance(suspended, bool):
            raise EvidenceV2Error("suspension values must be boolean")
        trade_date = _iso_date(row["trade_date"], label="trade_date")
        if suspended:
            stale_date = _iso_date(stale_date, label="stale_trade_date")
            if stale_date >= trade_date:
                raise EvidenceV2Error("suspension stale date must precede its boundary")
        elif stale_date is not None:
            raise EvidenceV2Error("non-suspended row cannot carry a stale date")
        normalized.append(
            {
                "symbol": _symbol(row["symbol"]),
                "trade_date": trade_date,
                "authoritative_suspension": suspended,
                "stale_trade_date": stale_date,
            }
        )
    normalized.sort(key=lambda row: (row["symbol"], row["trade_date"]))
    return seal_semantic(
        {
            "schema_version": SUSPENSION_EVIDENCE_SCHEMA,
            "generation_id": _safe_id(generation_id, label="generation_id"),
            "calendar_ref": calendar_ref.to_dict(),
            "rows": normalized,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_suspension_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "generation_id",
        "calendar_ref",
        "rows",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="suspension evidence")
    if payload["schema_version"] != SUSPENSION_EVIDENCE_SCHEMA:
        raise EvidenceV2Error("unsupported suspension evidence schema")
    _safe_id(payload["generation_id"], label="generation_id")
    EvidenceRef.from_dict(payload["calendar_ref"])
    rows = _validate_source_rows(
        payload["rows"],
        row_fields={
            "symbol",
            "trade_date",
            "authoritative_suspension",
            "stale_trade_date",
        },
        label="suspension",
    )
    for row in rows:
        suspended = row["authoritative_suspension"]
        stale_date = row["stale_trade_date"]
        if not isinstance(suspended, bool):
            raise EvidenceV2Error("suspension values must be boolean")
        if suspended:
            normalized_stale = _iso_date(stale_date, label="stale_trade_date")
            if normalized_stale >= row["trade_date"]:
                raise EvidenceV2Error("suspension stale date must precede its boundary")
        elif stale_date is not None:
            raise EvidenceV2Error("non-suspended row cannot carry a stale date")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("suspension evidence must be nonauthorizing")
    payload["rows"] = rows
    return payload


def _build_stock_mark_evidence(
    *,
    protocol_attempt_id: str,
    sample_id: str,
    symbol: str,
    slot_id: str,
    schedule_ref: EvidenceRef,
    entry: MarkCandidate,
    exit: MarkCandidate,
    market_parquet_ref: EvidenceRef,
    adjustment_factor_ref: EvidenceRef,
    pit_membership_ref: EvidenceRef,
    suspension_ref: EvidenceRef,
) -> dict[str, Any]:
    return seal_semantic(
        {
            "schema_version": STOCK_MARK_EVIDENCE_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "sample_id": _safe_id(sample_id, label="sample_id"),
            "symbol": _symbol(symbol),
            "slot_id": _safe_id(slot_id, label="slot_id"),
            "schedule_ref": schedule_ref.to_dict(),
            "entry": _candidate_to_dict(entry),
            "exit": _candidate_to_dict(exit),
            "market_parquet_ref": market_parquet_ref.to_dict(),
            "adjustment_factor_ref": adjustment_factor_ref.to_dict(),
            "pit_membership_ref": pit_membership_ref.to_dict(),
            "suspension_ref": suspension_ref.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_stock_mark_evidence(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], MarkCandidate, MarkCandidate]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "sample_id",
        "symbol",
        "slot_id",
        "schedule_ref",
        "entry",
        "exit",
        "market_parquet_ref",
        "adjustment_factor_ref",
        "pit_membership_ref",
        "suspension_ref",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="stock mark evidence")
    if payload["schema_version"] != STOCK_MARK_EVIDENCE_SCHEMA:
        raise EvidenceV2Error("unsupported stock mark evidence schema")
    _safe_id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _safe_id(payload["sample_id"], label="sample_id")
    _symbol(payload["symbol"])
    _safe_id(payload["slot_id"], label="slot_id")
    for field in (
        "schedule_ref",
        "market_parquet_ref",
        "adjustment_factor_ref",
        "pit_membership_ref",
        "suspension_ref",
    ):
        EvidenceRef.from_dict(payload[field])
    entry = _candidate_from_dict(payload["entry"])
    exit = _candidate_from_dict(payload["exit"])
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("stock mark evidence must be nonauthorizing")
    return payload, entry, exit


@dataclass(frozen=True)
class StockMarkSourceBundle:
    market_parquet: BoundRawArtifact
    adjustment_factors: BoundCanonicalArtifact
    pit_membership: BoundCanonicalArtifact
    suspensions: BoundCanonicalArtifact


@dataclass(frozen=True)
class ValidatedStockMarkSources:
    bundle: StockMarkSourceBundle
    calendar_ref: EvidenceRef = dataclass_field(init=False)
    market_marks: tuple[tuple[tuple[str, str], float], ...] = dataclass_field(init=False)
    adjustment_factors: tuple[tuple[tuple[str, str], float], ...] = dataclass_field(init=False)
    pit_membership: tuple[tuple[tuple[str, str], bool], ...] = dataclass_field(init=False)
    suspensions: tuple[tuple[tuple[str, str], tuple[bool, str | None]], ...] = dataclass_field(
        init=False
    )

    def __post_init__(self) -> None:
        market_ref = self.bundle.market_parquet.reference
        if market_ref.artifact_schema != STOCK_MARK_TABLE_SCHEMA:
            raise EvidenceV2Error("stock mark table ref schema mismatch")
        table = validate_stock_mark_parquet(self.bundle.market_parquet.payload)
        if market_ref.semantic_sha256 != table["parquet_metadata_semantic_sha256"]:
            raise EvidenceV2Error("stock mark table semantic SHA mismatch")
        adjustment = validate_adjustment_factor_evidence(self.bundle.adjustment_factors.read())
        pit = validate_pit_membership_evidence(self.bundle.pit_membership.read())
        suspensions = validate_suspension_evidence(self.bundle.suspensions.read())
        if adjustment["market_table_ref"] != market_ref.to_dict():
            raise EvidenceV2Error("adjustment factors drift from stock mark table")
        pit_calendar = EvidenceRef.from_dict(pit["calendar_ref"])
        if suspensions["calendar_ref"] != pit_calendar.to_dict():
            raise EvidenceV2Error("stock PIT/suspension calendar lineage mismatch")
        object.__setattr__(self, "calendar_ref", pit_calendar)
        object.__setattr__(
            self,
            "market_marks",
            tuple(
                (
                    (row["symbol"], row["trade_date"]),
                    decode_f64(row["close"], label="stock close"),
                )
                for row in table["rows"]
            ),
        )
        object.__setattr__(
            self,
            "adjustment_factors",
            tuple(
                (
                    (row["symbol"], row["trade_date"]),
                    decode_f64(row["adj_factor"], label="adj_factor"),
                )
                for row in adjustment["rows"]
            ),
        )
        object.__setattr__(
            self,
            "pit_membership",
            tuple(
                (
                    (row["symbol"], row["trade_date"]),
                    bool(row["pit_listed"]),
                )
                for row in pit["rows"]
            ),
        )
        object.__setattr__(
            self,
            "suspensions",
            tuple(
                (
                    (row["symbol"], row["trade_date"]),
                    (
                        bool(row["authoritative_suspension"]),
                        (None if row["stale_trade_date"] is None else str(row["stale_trade_date"])),
                    ),
                )
                for row in suspensions["rows"]
            ),
        )


def prepare_stock_mark_sources(
    bundle: StockMarkSourceBundle,
) -> ValidatedStockMarkSources:
    return ValidatedStockMarkSources(bundle=bundle)


def _candidate_from_stock_sources(
    sources: ValidatedStockMarkSources,
    *,
    symbol: str,
    boundary_date: str,
) -> MarkCandidate:
    key = (_symbol(symbol), _iso_date(boundary_date, label="boundary_date"))
    market_marks = dict(sources.market_marks)
    factors = dict(sources.adjustment_factors)
    memberships = dict(sources.pit_membership)
    suspensions = dict(sources.suspensions)
    if key not in memberships or key not in suspensions:
        raise EvidenceV2Error("stock boundary lacks PIT or suspension evidence")
    pit_listed = memberships[key]
    suspended, stale_date = suspensions[key]
    if key in market_marks:
        if key not in factors:
            raise EvidenceV2Error("exact stock mark lacks its adjustment factor")
        if suspended:
            raise EvidenceV2Error("stock boundary has both an exact mark and suspension")
        return MarkCandidate(
            exact_mark=market_marks[key] * factors[key],
            pit_listed=pit_listed,
            authoritative_suspension=False,
            stale_mark=None,
            terminal_settlement=None,
        )
    if not suspended or stale_date is None:
        raise EvidenceV2Error("stock boundary lacks an exact mark or suspension route")
    stale_key = (key[0], stale_date)
    if stale_key not in market_marks or stale_key not in factors:
        raise EvidenceV2Error("stock suspension route lacks a bound stale mark")
    return MarkCandidate(
        exact_mark=None,
        pit_listed=pit_listed,
        authoritative_suspension=True,
        stale_mark=market_marks[stale_key] * factors[stale_key],
        terminal_settlement=None,
    )


def build_stock_mark_evidence_from_sources(
    *,
    sources: ValidatedStockMarkSources,
    protocol_attempt_id: str,
    sample_id: str,
    symbol: str,
    slot_id: str,
    schedule_ref: EvidenceRef,
    entry_date: str,
    exit_date: str,
) -> dict[str, Any]:
    return _build_stock_mark_evidence(
        protocol_attempt_id=protocol_attempt_id,
        sample_id=sample_id,
        symbol=symbol,
        slot_id=slot_id,
        schedule_ref=schedule_ref,
        entry=_candidate_from_stock_sources(
            sources,
            symbol=symbol,
            boundary_date=entry_date,
        ),
        exit=_candidate_from_stock_sources(
            sources,
            symbol=symbol,
            boundary_date=exit_date,
        ),
        market_parquet_ref=sources.bundle.market_parquet.reference,
        adjustment_factor_ref=sources.bundle.adjustment_factors.reference,
        pit_membership_ref=sources.bundle.pit_membership.reference,
        suspension_ref=sources.bundle.suspensions.reference,
    )


def validate_stock_mark_evidence_from_sources(
    value: Mapping[str, Any],
    *,
    sources: ValidatedStockMarkSources,
    entry_date: str,
    exit_date: str,
) -> tuple[dict[str, Any], MarkCandidate, MarkCandidate]:
    payload, entry, exit = validate_stock_mark_evidence(value)
    expected = build_stock_mark_evidence_from_sources(
        sources=sources,
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        sample_id=str(payload["sample_id"]),
        symbol=str(payload["symbol"]),
        slot_id=str(payload["slot_id"]),
        schedule_ref=EvidenceRef.from_dict(payload["schedule_ref"]),
        entry_date=entry_date,
        exit_date=exit_date,
    )
    if expected != payload:
        raise EvidenceV2Error("stock mark evidence drifts from bound market sources")
    return payload, entry, exit


def resolve_mark(
    *,
    symbol: str,
    boundary_date: str,
    candidate: MarkCandidate,
    phase: str,
) -> tuple[float, str]:
    """Resolve entry/exit under the terminal > exact > suspension precedence."""

    if phase not in {"entry", "exit"}:
        raise EvidenceV2Error("mark phase must be entry or exit")
    boundary = _iso_date(boundary_date, label="boundary_date")
    terminal: dict[str, Any] | None = None
    if candidate.terminal_settlement is not None:
        terminal = validate_terminal_settlement(candidate.terminal_settlement)
        if terminal["symbol"] != str(symbol).strip().upper():
            raise EvidenceV2Error("terminal settlement symbol mismatch")
        if terminal["settlement_effective_date"] <= boundary:
            return (
                _positive_f64(terminal["terminal_adjusted_mark"], label="terminal_adjusted_mark"),
                "terminal_cash_settlement",
            )
    if candidate.exact_mark is not None:
        exact = float(candidate.exact_mark)
        if not math.isfinite(exact) or exact <= 0.0 or candidate.pit_listed is not True:
            raise EvidenceV2Error("exact mark is invalid or not PIT-listed")
        return exact, "exact_adjusted_close"
    if candidate.authoritative_suspension:
        if candidate.pit_listed is not True:
            raise EvidenceV2Error("generic stale mark is forbidden for a delisted symbol")
        if terminal is not None and terminal["settlement_effective_date"] <= boundary:
            raise EvidenceV2Error("terminal settlement must precede suspension stale mark")
        stale = candidate.stale_mark
        if stale is None or not math.isfinite(float(stale)) or float(stale) <= 0.0:
            raise EvidenceV2Error("authoritative suspension lacks an eligible stale mark")
        return float(stale), f"pit_listed_suspension_stale_{phase}_mark"
    raise EvidenceV2Error("data_missing_or_corrupt")


def _build_mark_target_outcome(
    *,
    protocol_attempt_id: str,
    sample_id: str,
    symbol: str,
    target_sessions: Sequence[str],
    entry: MarkCandidate,
    exit: MarkCandidate,
    h00300_s1_close: float,
    h00300_s20_close: float,
    costs: CostVector,
    stock_evidence: EvidenceRef,
    benchmark_evidence: EvidenceRef,
    cost_evidence: EvidenceRef,
    schedule_ref: EvidenceRef,
) -> dict[str, Any]:
    normalized_sessions = _target_sessions(list(target_sessions))
    s1 = normalized_sessions[0]
    s20 = normalized_sessions[-1]
    normalized_sample_id = _safe_id(sample_id, label="sample_id")
    normalized_symbol = _symbol(symbol)
    entry_mark, entry_source = resolve_mark(
        symbol=normalized_symbol,
        boundary_date=s1,
        candidate=entry,
        phase="entry",
    )
    exit_mark, exit_source = resolve_mark(
        symbol=normalized_symbol,
        boundary_date=s20,
        candidate=exit,
        phase="exit",
    )
    benchmark_entry = float(h00300_s1_close)
    benchmark_exit = float(h00300_s20_close)
    if (
        not math.isfinite(benchmark_entry)
        or not math.isfinite(benchmark_exit)
        or benchmark_entry <= 0.0
        or benchmark_exit <= 0.0
    ):
        raise EvidenceV2Error("H00300 target marks must be positive finite values")
    mark_return = exit_mark / entry_mark - 1.0
    benchmark_return = benchmark_exit / benchmark_entry - 1.0
    alpha = mark_return - benchmark_return - costs.total
    if not all(math.isfinite(item) for item in (mark_return, benchmark_return, alpha)):
        raise EvidenceV2Error("mark target produced a non-finite value")
    return seal_semantic(
        {
            "schema_version": TARGET_OUTCOME_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "target_id": TARGET_ID,
            "sample_id": normalized_sample_id,
            "symbol": normalized_symbol,
            "s1_date": s1,
            "s20_date": s20,
            "target_sessions": normalized_sessions,
            "entry_mark": encode_f64(entry_mark),
            "entry_mark_source": entry_source,
            "exit_mark": encode_f64(exit_mark),
            "exit_mark_source": exit_source,
            "h00300_s1_close": encode_f64(benchmark_entry),
            "h00300_s20_close": encode_f64(benchmark_exit),
            "costs": costs.to_rows(),
            "mark_return": encode_f64(mark_return),
            "benchmark_return": encode_f64(benchmark_return),
            "realized_mark_alpha": encode_f64(alpha),
            "positive_outcome": alpha > 0.0,
            "non_executable_research_target": True,
            "stock_evidence": stock_evidence.to_dict(),
            "benchmark_evidence": benchmark_evidence.to_dict(),
            "cost_evidence": cost_evidence.to_dict(),
            "schedule_ref": schedule_ref.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_mark_target_outcome(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "target_id",
        "sample_id",
        "symbol",
        "s1_date",
        "s20_date",
        "target_sessions",
        "entry_mark",
        "entry_mark_source",
        "exit_mark",
        "exit_mark_source",
        "h00300_s1_close",
        "h00300_s20_close",
        "costs",
        "mark_return",
        "benchmark_return",
        "realized_mark_alpha",
        "positive_outcome",
        "non_executable_research_target",
        "stock_evidence",
        "benchmark_evidence",
        "cost_evidence",
        "schedule_ref",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="mark target outcome")
    if (
        payload["schema_version"] != TARGET_OUTCOME_SCHEMA
        or payload["target_id"] != TARGET_ID
        or payload["non_executable_research_target"] is not True
    ):
        raise EvidenceV2Error("mark target outcome identity mismatch")
    _safe_id(payload["sample_id"], label="sample_id")
    _safe_id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _symbol(payload["symbol"])
    sessions = _target_sessions(payload["target_sessions"])
    if payload["s1_date"] != sessions[0] or payload["s20_date"] != sessions[-1]:
        raise EvidenceV2Error("mark target boundary dates drift from target sessions")
    entry_mark = _positive_f64(payload["entry_mark"], label="entry_mark")
    exit_mark = _positive_f64(payload["exit_mark"], label="exit_mark")
    benchmark_entry = _positive_f64(
        payload["h00300_s1_close"],
        label="h00300_s1_close",
    )
    benchmark_exit = _positive_f64(
        payload["h00300_s20_close"],
        label="h00300_s20_close",
    )
    if payload["entry_mark_source"] not in {
        "terminal_cash_settlement",
        "exact_adjusted_close",
        "pit_listed_suspension_stale_entry_mark",
    } or payload["exit_mark_source"] not in {
        "terminal_cash_settlement",
        "exact_adjusted_close",
        "pit_listed_suspension_stale_exit_mark",
    }:
        raise EvidenceV2Error("mark target source label is invalid")
    if not isinstance(payload["costs"], list):
        raise EvidenceV2Error("mark target cost vector must be a list")
    costs = CostVector.from_rows(payload["costs"])
    mark_return = exit_mark / entry_mark - 1.0
    benchmark_return = benchmark_exit / benchmark_entry - 1.0
    alpha = mark_return - benchmark_return - costs.total
    declared = {
        "mark_return": decode_f64(payload["mark_return"], label="mark_return"),
        "benchmark_return": decode_f64(
            payload["benchmark_return"],
            label="benchmark_return",
        ),
        "realized_mark_alpha": decode_f64(
            payload["realized_mark_alpha"],
            label="realized_mark_alpha",
        ),
    }
    expected = {
        "mark_return": mark_return,
        "benchmark_return": benchmark_return,
        "realized_mark_alpha": alpha,
    }
    if any(encode_f64(declared[name]) != encode_f64(expected[name]) for name in expected):
        raise EvidenceV2Error("mark target outcome recomputation mismatch")
    if payload["positive_outcome"] is not (alpha > 0.0):
        raise EvidenceV2Error("mark target positive outcome mismatch")
    for field in (
        "stock_evidence",
        "benchmark_evidence",
        "cost_evidence",
        "schedule_ref",
    ):
        EvidenceRef.from_dict(payload[field])
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("mark target outcome must be nonauthorizing")
    return payload


def _normalize_stat_value(value: Any, field: pa.Field) -> Any:
    if value is None:
        return None
    if pa.types.is_string(field.type):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
    if pa.types.is_date32(field.type):
        if isinstance(value, date):
            return value.isoformat()
        return _iso_date(value, label=f"statistics.{field.name}")
    if pa.types.is_float64(field.type):
        return encode_f64(float(value))
    if pa.types.is_timestamp(field.type):
        if not isinstance(value, datetime):
            raise EvidenceV2Error("timestamp statistics must decode as datetime")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    raise EvidenceV2Error("unsupported Parquet statistics type")


def parquet_metadata_projection(parquet_file: pq.ParquetFile) -> dict[str, Any]:
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow
    raw_key_values = metadata.metadata or {}
    key_values: list[dict[str, str]] = []
    for raw_key, raw_value in raw_key_values.items():
        try:
            key = raw_key.decode("utf-8")
            value = raw_value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceV2Error("Parquet key-value metadata must be UTF-8") from exc
        key_values.append({"key": key, "value": value})
    key_values.sort(key=lambda item: item["key"])
    if len({item["key"] for item in key_values}) != len(key_values):
        raise EvidenceV2Error("Parquet key-value metadata keys must be unique")

    row_groups: list[dict[str, Any]] = []
    for group_index in range(metadata.num_row_groups):
        group = metadata.row_group(group_index)
        columns: list[dict[str, Any]] = []
        for column_index, field in enumerate(schema):
            column = group.column(column_index)
            statistics = column.statistics
            if statistics is None:
                raise EvidenceV2Error("Parquet column statistics are required")
            columns.append(
                {
                    "path_in_schema": column.path_in_schema,
                    "physical_type": column.physical_type,
                    "compression": column.compression,
                    "encodings": sorted(str(item) for item in column.encodings),
                    "num_values": column.num_values,
                    "total_compressed_size": column.total_compressed_size,
                    "total_uncompressed_size": column.total_uncompressed_size,
                    "statistics": {
                        "has_min_max": statistics.has_min_max,
                        "null_count": statistics.null_count,
                        "distinct_count": statistics.distinct_count,
                        "min": (
                            _normalize_stat_value(statistics.min, field)
                            if statistics.has_min_max
                            else None
                        ),
                        "max": (
                            _normalize_stat_value(statistics.max, field)
                            if statistics.has_min_max
                            else None
                        ),
                    },
                }
            )
        row_groups.append(
            {
                "num_rows": group.num_rows,
                "total_byte_size": group.total_byte_size,
                "columns": columns,
            }
        )
    return {
        "schema": [
            {"name": field.name, "arrow_type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
        "created_by": str(metadata.created_by or ""),
        "file_key_value_metadata": key_values,
        "row_count": metadata.num_rows,
        "row_group_count": metadata.num_row_groups,
        "row_groups": row_groups,
    }


def validate_h00300_parquet(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > INDEX_TABLE_MAX_BYTES:
        raise EvidenceV2Error("H00300 Parquet byte size is invalid")
    try:
        parquet_file = pq.ParquetFile(pa.BufferReader(payload))
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise EvidenceV2Error("H00300 Parquet parse failed") from exc
    if not parquet_file.schema_arrow.equals(EXPECTED_INDEX_SCHEMA, check_metadata=False):
        raise EvidenceV2Error("H00300 Arrow schema mismatch")
    row_count = parquet_file.metadata.num_rows
    if row_count <= 0 or row_count > INDEX_TABLE_MAX_ROWS:
        raise EvidenceV2Error("H00300 row count is invalid")
    projection = parquet_metadata_projection(parquet_file)
    try:
        table = parquet_file.read()
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise EvidenceV2Error("H00300 Parquet read failed") from exc
    if table.num_rows != row_count:
        raise EvidenceV2Error("H00300 decoded row count mismatch")
    for name in table.column_names:
        if table[name].null_count != 0:
            raise EvidenceV2Error(f"H00300 column {name} contains nulls")
    rows = table.to_pylist()
    dates: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        trade_date = row["trade_date"].isoformat()
        dates.append(trade_date)
        close = float(row["close_total_return"])
        observed = row["source_observed_at"]
        if observed.tzinfo is None or observed.utcoffset() != timezone.utc.utcoffset(observed):
            raise EvidenceV2Error("H00300 source_observed_at must be UTC")
        if (
            row["instrument_id"] != H00300_INSTRUMENT_ID
            or row["currency"] != H00300_CURRENCY
            or row["return_type"] != H00300_RETURN_TYPE
            or row["source_system"] != H00300_SOURCE_SYSTEM
            or not math.isfinite(close)
            or close <= 0.0
        ):
            raise EvidenceV2Error("H00300 row domain mismatch")
        source_sha = str(row["source_document_sha256"])
        if len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha):
            raise EvidenceV2Error("H00300 source document SHA is invalid")
        normalized_rows.append(
            {
                "instrument_id": H00300_INSTRUMENT_ID,
                "trade_date": trade_date,
                "close_total_return": encode_f64(close),
                "currency": H00300_CURRENCY,
                "return_type": H00300_RETURN_TYPE,
                "source_system": H00300_SOURCE_SYSTEM,
                "source_observed_at": observed.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "source_document_sha256": source_sha,
            }
        )
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise EvidenceV2Error("H00300 trade dates must be strictly increasing and unique")
    return {
        "rows": normalized_rows,
        "row_count": row_count,
        "min_trade_date": dates[0],
        "max_trade_date": dates[-1],
        "metadata_projection": projection,
        "parquet_metadata_semantic_sha256": semantic_sha256({"projection": projection}),
    }


def validate_h00300_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "generation_id",
        "created_at",
        "instrument_id",
        "official_index_code",
        "return_type",
        "currency",
        "source_system",
        "table_ref",
        "parquet_metadata_semantic_sha256",
        "row_count",
        "min_trade_date",
        "max_trade_date",
        "ordered_trade_dates_sha256",
        "official_source_receipt",
        "calendar_ref",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="H00300 manifest")
    if (
        payload["schema_version"] != INDEX_MANIFEST_SCHEMA
        or payload["instrument_id"] != H00300_INSTRUMENT_ID
        or payload["official_index_code"] != H00300_OFFICIAL_CODE
        or payload["return_type"] != H00300_RETURN_TYPE
        or payload["currency"] != H00300_CURRENCY
        or payload["source_system"] != H00300_SOURCE_SYSTEM
    ):
        raise EvidenceV2Error("H00300 manifest identity mismatch")
    if not str(payload["generation_id"]):
        raise EvidenceV2Error("H00300 generation_id must be nonempty")
    _utc(payload["created_at"], label="created_at")
    EvidenceRef.from_dict(payload["table_ref"])
    EvidenceRef.from_dict(payload["official_source_receipt"])
    EvidenceRef.from_dict(payload["calendar_ref"])
    for field in (
        "parquet_metadata_semantic_sha256",
        "ordered_trade_dates_sha256",
    ):
        text = str(payload[field])
        if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
            raise EvidenceV2Error(f"H00300 {field} is invalid")
    if (
        isinstance(payload["row_count"], bool)
        or not isinstance(payload["row_count"], int)
        or not 0 < payload["row_count"] <= INDEX_TABLE_MAX_ROWS
    ):
        raise EvidenceV2Error("H00300 manifest row_count is invalid")
    minimum = _iso_date(payload["min_trade_date"], label="min_trade_date")
    maximum = _iso_date(payload["max_trade_date"], label="max_trade_date")
    if maximum < minimum:
        raise EvidenceV2Error("H00300 manifest date range is reversed")
    return payload


def build_h00300_manifest(
    *,
    generation_id: str,
    created_at: str,
    table_ref: EvidenceRef,
    parquet_payload: bytes,
    official_source_receipt: EvidenceRef,
    calendar_ref: EvidenceRef,
) -> dict[str, Any]:
    if table_ref.artifact_schema != INDEX_TABLE_SCHEMA:
        raise EvidenceV2Error("H00300 table ref schema mismatch")
    if sha256_bytes(parquet_payload) != table_ref.byte_sha256:
        raise EvidenceV2Error("H00300 table ref byte SHA mismatch")
    table = validate_h00300_parquet(parquet_payload)
    if table_ref.semantic_sha256 != table["parquet_metadata_semantic_sha256"]:
        raise EvidenceV2Error("H00300 table ref semantic SHA mismatch")
    dates = [row["trade_date"] for row in table["rows"]]
    return seal_semantic(
        {
            "schema_version": INDEX_MANIFEST_SCHEMA,
            "generation_id": _safe_id(generation_id, label="generation_id"),
            "created_at": _utc(created_at, label="created_at"),
            "instrument_id": H00300_INSTRUMENT_ID,
            "official_index_code": H00300_OFFICIAL_CODE,
            "return_type": H00300_RETURN_TYPE,
            "currency": H00300_CURRENCY,
            "source_system": H00300_SOURCE_SYSTEM,
            "table_ref": table_ref.to_dict(),
            "parquet_metadata_semantic_sha256": table["parquet_metadata_semantic_sha256"],
            "row_count": table["row_count"],
            "min_trade_date": table["min_trade_date"],
            "max_trade_date": table["max_trade_date"],
            "ordered_trade_dates_sha256": semantic_sha256({"trade_dates": dates}),
            "official_source_receipt": official_source_receipt.to_dict(),
            "calendar_ref": calendar_ref.to_dict(),
        }
    )


def validate_h00300_manifest_with_parquet(
    manifest: Mapping[str, Any],
    *,
    parquet_payload: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = validate_h00300_manifest(manifest)
    table_ref = EvidenceRef.from_dict(normalized["table_ref"])
    if table_ref.artifact_schema != INDEX_TABLE_SCHEMA:
        raise EvidenceV2Error("H00300 bound table schema mismatch")
    if sha256_bytes(parquet_payload) != table_ref.byte_sha256:
        raise EvidenceV2Error("H00300 bound table byte SHA mismatch")
    table = validate_h00300_parquet(parquet_payload)
    dates = [row["trade_date"] for row in table["rows"]]
    comparisons = {
        "parquet_metadata_semantic_sha256": table["parquet_metadata_semantic_sha256"],
        "row_count": table["row_count"],
        "min_trade_date": table["min_trade_date"],
        "max_trade_date": table["max_trade_date"],
        "ordered_trade_dates_sha256": semantic_sha256({"trade_dates": dates}),
    }
    if any(normalized[field] != expected for field, expected in comparisons.items()):
        raise EvidenceV2Error("H00300 manifest/table recomputation mismatch")
    if table_ref.semantic_sha256 != table["parquet_metadata_semantic_sha256"]:
        raise EvidenceV2Error("H00300 bound table semantic SHA mismatch")
    return normalized, table


@dataclass(frozen=True)
class MarkTargetEvidenceBundle:
    schedule_anchor: ScheduleAnchorBinding
    stock_marks: BoundCanonicalArtifact
    stock_sources: StockMarkSourceBundle
    costs: BoundCanonicalArtifact
    benchmark_manifest: BoundCanonicalArtifact
    benchmark_parquet: bytes


@dataclass(frozen=True)
class ValidatedMarkTargetCommonEvidence:
    """Validated schedule and H00300 inputs shared by many target samples."""

    schedule_anchor: ScheduleAnchorBinding
    benchmark_manifest: BoundCanonicalArtifact
    benchmark_parquet: bytes
    protocol_attempt_id: str = dataclass_field(init=False)
    epoch: str = dataclass_field(init=False)
    schedule_id: str = dataclass_field(init=False)
    seed_hex: str = dataclass_field(init=False)
    calendar_ref: EvidenceRef = dataclass_field(init=False)
    calibration_universe_ref: EvidenceRef = dataclass_field(init=False)
    model_bundle_refs: tuple[tuple[str, EvidenceRef], ...] | None = dataclass_field(init=False)
    slots: tuple[tuple[str, tuple[str, ...]], ...] = dataclass_field(init=False)
    prediction_anchor_windows: tuple[tuple[str, str, str], ...] = dataclass_field(init=False)
    benchmark_closes: tuple[tuple[str, float], ...] = dataclass_field(init=False)

    def __post_init__(self) -> None:
        schedule = validate_schedule_anchor_binding(self.schedule_anchor)
        manifest, table = validate_h00300_manifest_with_parquet(
            self.benchmark_manifest.read(),
            parquet_payload=self.benchmark_parquet,
        )
        calendar_ref = EvidenceRef.from_dict(schedule["open_session_calendar"])
        if manifest["calendar_ref"] != calendar_ref.to_dict():
            raise EvidenceV2Error("mark target schedule/calendar lineage mismatch")
        object.__setattr__(
            self,
            "protocol_attempt_id",
            str(schedule["protocol_attempt_id"]),
        )
        object.__setattr__(self, "epoch", str(schedule["epoch"]))
        object.__setattr__(self, "schedule_id", str(schedule["schedule_id"]))
        object.__setattr__(self, "seed_hex", str(schedule["seed_hex"]))
        object.__setattr__(self, "calendar_ref", calendar_ref)
        raw_universe_ref = schedule["calibration_universe_ref"]
        if raw_universe_ref is None:
            raise EvidenceV2Error("mark target schedule lacks a calibration universe")
        object.__setattr__(
            self,
            "calibration_universe_ref",
            EvidenceRef.from_dict(raw_universe_ref),
        )
        raw_model_refs = schedule["model_bundle_refs"]
        object.__setattr__(
            self,
            "model_bundle_refs",
            (
                None
                if raw_model_refs is None
                else tuple(
                    (branch, EvidenceRef.from_dict(reference))
                    for branch, reference in raw_model_refs.items()
                )
            ),
        )
        object.__setattr__(
            self,
            "slots",
            tuple(
                (
                    str(slot["slot_id"]),
                    tuple(_target_sessions(slot["target_sessions"])),
                )
                for slot in schedule["slots"]
            ),
        )
        object.__setattr__(
            self,
            "prediction_anchor_windows",
            tuple(
                (
                    str(slot["slot_id"]),
                    str(slot["s0_close_at"]),
                    str(slot["s1_open_at"]),
                )
                for slot in schedule["slots"]
            ),
        )
        object.__setattr__(
            self,
            "benchmark_closes",
            tuple(
                (
                    str(row["trade_date"]),
                    decode_f64(
                        row["close_total_return"],
                        label=f"H00300 {row['trade_date']} close",
                    ),
                )
                for row in table["rows"]
            ),
        )


def prepare_mark_target_common_evidence(
    *,
    schedule_anchor: ScheduleAnchorBinding,
    benchmark_manifest: BoundCanonicalArtifact,
    benchmark_parquet: bytes,
) -> ValidatedMarkTargetCommonEvidence:
    return ValidatedMarkTargetCommonEvidence(
        schedule_anchor=schedule_anchor,
        benchmark_manifest=benchmark_manifest,
        benchmark_parquet=benchmark_parquet,
    )


def build_mark_target_outcome_from_common_evidence(
    *,
    common: ValidatedMarkTargetCommonEvidence,
    stock_marks: BoundCanonicalArtifact,
    stock_sources: ValidatedStockMarkSources,
    costs: BoundCanonicalArtifact,
) -> dict[str, Any]:
    stock, _, _ = validate_stock_mark_evidence(stock_marks.read())
    cost_payload, cost_vector = validate_cost_evidence(costs.read())
    schedule_ref = common.schedule_anchor.schedule.reference
    attempt_id = common.protocol_attempt_id
    if (
        stock["protocol_attempt_id"] != attempt_id
        or cost_payload["protocol_attempt_id"] != attempt_id
    ):
        raise EvidenceV2Error("mark target protocol attempt lineage mismatch")
    if stock["schedule_ref"] != schedule_ref.to_dict():
        raise EvidenceV2Error("mark target schedule lineage mismatch")
    if stock_sources.calendar_ref != common.calendar_ref:
        raise EvidenceV2Error("stock source calendar drifts from target schedule")
    if stock["sample_id"] != cost_payload["sample_id"]:
        raise EvidenceV2Error("mark target sample lineage mismatch")
    matching_slots = [
        target_sessions for slot_id, target_sessions in common.slots if slot_id == stock["slot_id"]
    ]
    if len(matching_slots) != 1:
        raise EvidenceV2Error("mark target schedule slot is missing or ambiguous")
    target_sessions = list(matching_slots[0])
    stock, entry, exit = validate_stock_mark_evidence_from_sources(
        stock_marks.read(),
        sources=stock_sources,
        entry_date=target_sessions[0],
        exit_date=target_sessions[-1],
    )
    rows_by_date = dict(common.benchmark_closes)
    missing_sessions = [session for session in target_sessions if session not in rows_by_date]
    if missing_sessions:
        raise EvidenceV2Error("H00300 lacks one or more exact target-session rows")
    return _build_mark_target_outcome(
        protocol_attempt_id=attempt_id,
        sample_id=stock["sample_id"],
        symbol=stock["symbol"],
        target_sessions=target_sessions,
        entry=entry,
        exit=exit,
        h00300_s1_close=rows_by_date[target_sessions[0]],
        h00300_s20_close=rows_by_date[target_sessions[-1]],
        costs=cost_vector,
        stock_evidence=stock_marks.reference,
        benchmark_evidence=common.benchmark_manifest.reference,
        cost_evidence=costs.reference,
        schedule_ref=schedule_ref,
    )


def build_mark_target_outcome_from_evidence(
    bundle: MarkTargetEvidenceBundle,
) -> dict[str, Any]:
    common = prepare_mark_target_common_evidence(
        schedule_anchor=bundle.schedule_anchor,
        benchmark_manifest=bundle.benchmark_manifest,
        benchmark_parquet=bundle.benchmark_parquet,
    )
    stock_sources = prepare_stock_mark_sources(bundle.stock_sources)
    return build_mark_target_outcome_from_common_evidence(
        common=common,
        stock_marks=bundle.stock_marks,
        stock_sources=stock_sources,
        costs=bundle.costs,
    )


def validate_mark_target_outcome_from_evidence(
    value: Mapping[str, Any],
    *,
    bundle: MarkTargetEvidenceBundle,
) -> dict[str, Any]:
    normalized = validate_mark_target_outcome(value)
    recomputed = build_mark_target_outcome_from_evidence(bundle)
    if recomputed != normalized:
        raise EvidenceV2Error("mark target outcome drifts from bound source evidence")
    return normalized


def validate_mark_target_outcome_from_common_evidence(
    value: Mapping[str, Any],
    *,
    common: ValidatedMarkTargetCommonEvidence,
    stock_marks: BoundCanonicalArtifact,
    stock_sources: ValidatedStockMarkSources,
    costs: BoundCanonicalArtifact,
) -> dict[str, Any]:
    normalized = validate_mark_target_outcome(value)
    recomputed = build_mark_target_outcome_from_common_evidence(
        common=common,
        stock_marks=stock_marks,
        stock_sources=stock_sources,
        costs=costs,
    )
    if recomputed != normalized:
        raise EvidenceV2Error("mark target outcome drifts from bound source evidence")
    return normalized


@dataclass(frozen=True)
class MarkTargetEvidenceBundleV3:
    """V3-only mark target inputs; schedule-v2 bindings are not accepted."""

    schedule_anchor: ScheduleAnchorBindingV3
    stock_marks: BoundCanonicalArtifact
    stock_sources: StockMarkSourceBundle
    costs: BoundCanonicalArtifact
    benchmark_manifest: BoundCanonicalArtifact
    benchmark_parquet: bytes


@dataclass(frozen=True)
class ValidatedMarkTargetCommonEvidenceV3:
    """Validated schedule-v3 and H00300 inputs shared by target samples."""

    schedule_anchor: ScheduleAnchorBindingV3
    benchmark_manifest: BoundCanonicalArtifact
    benchmark_parquet: bytes
    protocol_attempt_id: str = dataclass_field(init=False)
    epoch: str = dataclass_field(init=False)
    schedule_id: str = dataclass_field(init=False)
    seed_hex: str = dataclass_field(init=False)
    calendar_ref: EvidenceRef = dataclass_field(init=False)
    calibration_universe_ref: EvidenceRef = dataclass_field(init=False)
    model_bundle_refs: tuple[tuple[str, EvidenceRef], ...] | None = dataclass_field(init=False)
    slots: tuple[tuple[str, tuple[str, ...]], ...] = dataclass_field(init=False)
    prediction_anchor_windows: tuple[tuple[str, str, str], ...] = dataclass_field(init=False)
    benchmark_closes: tuple[tuple[str, float], ...] = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_anchor, ScheduleAnchorBindingV3):
            raise EvidenceV2Error("mark target v3 requires ScheduleAnchorBindingV3")
        schedule = validate_schedule_anchor_binding_v3(self.schedule_anchor)
        manifest, table = validate_h00300_manifest_with_parquet(
            self.benchmark_manifest.read(),
            parquet_payload=self.benchmark_parquet,
        )
        calendar_ref = EvidenceRef.from_dict(schedule["open_session_calendar"])
        if manifest["calendar_ref"] != calendar_ref.to_dict():
            raise EvidenceV2Error("mark target v3 schedule/calendar lineage mismatch")
        object.__setattr__(self, "protocol_attempt_id", str(schedule["protocol_attempt_id"]))
        object.__setattr__(self, "epoch", str(schedule["epoch"]))
        object.__setattr__(self, "schedule_id", str(schedule["schedule_id"]))
        object.__setattr__(self, "seed_hex", str(schedule["seed_hex"]))
        object.__setattr__(self, "calendar_ref", calendar_ref)
        raw_universe_ref = schedule["calibration_universe_ref"]
        if raw_universe_ref is None:
            raise EvidenceV2Error("mark target v3 schedule lacks a calibration universe")
        object.__setattr__(
            self,
            "calibration_universe_ref",
            EvidenceRef.from_dict(raw_universe_ref),
        )
        raw_model_refs = schedule["model_bundle_refs"]
        object.__setattr__(
            self,
            "model_bundle_refs",
            (
                None
                if raw_model_refs is None
                else tuple(
                    (branch, EvidenceRef.from_dict(reference))
                    for branch, reference in raw_model_refs.items()
                )
            ),
        )
        object.__setattr__(
            self,
            "slots",
            tuple(
                (
                    str(slot["slot_id"]),
                    tuple(_target_sessions(slot["target_sessions"])),
                )
                for slot in schedule["slots"]
            ),
        )
        object.__setattr__(
            self,
            "prediction_anchor_windows",
            tuple(
                (
                    str(slot["slot_id"]),
                    str(slot["s0_close_at"]),
                    str(slot["s1_open_at"]),
                )
                for slot in schedule["slots"]
            ),
        )
        object.__setattr__(
            self,
            "benchmark_closes",
            tuple(
                (
                    str(row["trade_date"]),
                    decode_f64(
                        row["close_total_return"],
                        label=f"H00300 {row['trade_date']} close",
                    ),
                )
                for row in table["rows"]
            ),
        )


def prepare_mark_target_common_evidence_v3(
    *,
    schedule_anchor: ScheduleAnchorBindingV3,
    benchmark_manifest: BoundCanonicalArtifact,
    benchmark_parquet: bytes,
) -> ValidatedMarkTargetCommonEvidenceV3:
    if not isinstance(schedule_anchor, ScheduleAnchorBindingV3):
        raise EvidenceV2Error("mark target v3 requires ScheduleAnchorBindingV3")
    return ValidatedMarkTargetCommonEvidenceV3(
        schedule_anchor=schedule_anchor,
        benchmark_manifest=benchmark_manifest,
        benchmark_parquet=benchmark_parquet,
    )


def build_mark_target_outcome_from_common_evidence_v3(
    *,
    common: ValidatedMarkTargetCommonEvidenceV3,
    stock_marks: BoundCanonicalArtifact,
    stock_sources: ValidatedStockMarkSources,
    costs: BoundCanonicalArtifact,
) -> dict[str, Any]:
    if not isinstance(common, ValidatedMarkTargetCommonEvidenceV3):
        raise EvidenceV2Error("mark target v3 common evidence has the wrong type")
    stock, _, _ = validate_stock_mark_evidence(stock_marks.read())
    cost_payload, cost_vector = validate_cost_evidence(costs.read())
    schedule_ref = common.schedule_anchor.evidence.schedule.reference
    attempt_id = common.protocol_attempt_id
    if (
        stock["protocol_attempt_id"] != attempt_id
        or cost_payload["protocol_attempt_id"] != attempt_id
    ):
        raise EvidenceV2Error("mark target v3 protocol attempt lineage mismatch")
    if stock["schedule_ref"] != schedule_ref.to_dict():
        raise EvidenceV2Error("mark target v3 schedule lineage mismatch")
    if stock_sources.calendar_ref != common.calendar_ref:
        raise EvidenceV2Error("stock source calendar drifts from target schedule v3")
    if stock["sample_id"] != cost_payload["sample_id"]:
        raise EvidenceV2Error("mark target v3 sample lineage mismatch")
    matching_slots = [
        target_sessions
        for slot_id, target_sessions in common.slots
        if slot_id == stock["slot_id"]
    ]
    if len(matching_slots) != 1:
        raise EvidenceV2Error("mark target v3 schedule slot is missing or ambiguous")
    target_sessions = list(matching_slots[0])
    stock, entry, exit = validate_stock_mark_evidence_from_sources(
        stock_marks.read(),
        sources=stock_sources,
        entry_date=target_sessions[0],
        exit_date=target_sessions[-1],
    )
    rows_by_date = dict(common.benchmark_closes)
    missing_sessions = [session for session in target_sessions if session not in rows_by_date]
    if missing_sessions:
        raise EvidenceV2Error("H00300 lacks one or more exact target-session rows")
    return _build_mark_target_outcome(
        protocol_attempt_id=attempt_id,
        sample_id=stock["sample_id"],
        symbol=stock["symbol"],
        target_sessions=target_sessions,
        entry=entry,
        exit=exit,
        h00300_s1_close=rows_by_date[target_sessions[0]],
        h00300_s20_close=rows_by_date[target_sessions[-1]],
        costs=cost_vector,
        stock_evidence=stock_marks.reference,
        benchmark_evidence=common.benchmark_manifest.reference,
        cost_evidence=costs.reference,
        schedule_ref=schedule_ref,
    )


def build_mark_target_outcome_from_evidence_v3(
    bundle: MarkTargetEvidenceBundleV3,
) -> dict[str, Any]:
    if not isinstance(bundle, MarkTargetEvidenceBundleV3):
        raise EvidenceV2Error("mark target v3 requires MarkTargetEvidenceBundleV3")
    common = prepare_mark_target_common_evidence_v3(
        schedule_anchor=bundle.schedule_anchor,
        benchmark_manifest=bundle.benchmark_manifest,
        benchmark_parquet=bundle.benchmark_parquet,
    )
    stock_sources = prepare_stock_mark_sources(bundle.stock_sources)
    return build_mark_target_outcome_from_common_evidence_v3(
        common=common,
        stock_marks=bundle.stock_marks,
        stock_sources=stock_sources,
        costs=bundle.costs,
    )


def validate_mark_target_outcome_from_evidence_v3(
    value: Mapping[str, Any],
    *,
    bundle: MarkTargetEvidenceBundleV3,
) -> dict[str, Any]:
    normalized = validate_mark_target_outcome(value)
    recomputed = build_mark_target_outcome_from_evidence_v3(bundle)
    if recomputed != normalized:
        raise EvidenceV2Error("mark target outcome drifts from bound v3 source evidence")
    return normalized


def validate_mark_target_outcome_from_common_evidence_v3(
    value: Mapping[str, Any],
    *,
    common: ValidatedMarkTargetCommonEvidenceV3,
    stock_marks: BoundCanonicalArtifact,
    stock_sources: ValidatedStockMarkSources,
    costs: BoundCanonicalArtifact,
) -> dict[str, Any]:
    normalized = validate_mark_target_outcome(value)
    recomputed = build_mark_target_outcome_from_common_evidence_v3(
        common=common,
        stock_marks=stock_marks,
        stock_sources=stock_sources,
        costs=costs,
    )
    if recomputed != normalized:
        raise EvidenceV2Error("mark target outcome drifts from bound v3 source evidence")
    return normalized


__all__ = [
    "ADJUSTMENT_FACTOR_EVIDENCE_SCHEMA",
    "COST_COMPONENT_ORDER",
    "COST_EVIDENCE_SCHEMA",
    "CostVector",
    "EXPECTED_INDEX_SCHEMA",
    "EXPECTED_STOCK_MARK_SCHEMA",
    "H00300_INSTRUMENT_ID",
    "INDEX_MANIFEST_SCHEMA",
    "INDEX_TABLE_SCHEMA",
    "INDEX_TABLE_MAX_BYTES",
    "MarkCandidate",
    "MarkTargetEvidenceBundle",
    "MarkTargetEvidenceBundleV3",
    "PIT_MEMBERSHIP_EVIDENCE_SCHEMA",
    "STOCK_MARK_EVIDENCE_SCHEMA",
    "STOCK_MARK_TABLE_SCHEMA",
    "StockMarkSourceBundle",
    "SUSPENSION_EVIDENCE_SCHEMA",
    "TARGET_ID",
    "TARGET_OUTCOME_SCHEMA",
    "TERMINAL_SETTLEMENT_SCHEMA",
    "ValidatedMarkTargetCommonEvidence",
    "ValidatedMarkTargetCommonEvidenceV3",
    "ValidatedStockMarkSources",
    "build_adjustment_factor_evidence",
    "build_cost_evidence",
    "build_h00300_manifest",
    "build_mark_target_outcome_from_common_evidence",
    "build_mark_target_outcome_from_common_evidence_v3",
    "build_mark_target_outcome_from_evidence",
    "build_mark_target_outcome_from_evidence_v3",
    "build_stock_mark_evidence_from_sources",
    "build_pit_membership_evidence",
    "build_suspension_evidence",
    "build_terminal_settlement",
    "parquet_metadata_projection",
    "prepare_mark_target_common_evidence",
    "prepare_mark_target_common_evidence_v3",
    "prepare_stock_mark_sources",
    "resolve_mark",
    "validate_cost_evidence",
    "validate_adjustment_factor_evidence",
    "validate_h00300_manifest",
    "validate_h00300_manifest_with_parquet",
    "validate_h00300_parquet",
    "validate_mark_target_outcome",
    "validate_mark_target_outcome_from_common_evidence",
    "validate_mark_target_outcome_from_common_evidence_v3",
    "validate_mark_target_outcome_from_evidence",
    "validate_mark_target_outcome_from_evidence_v3",
    "validate_stock_mark_evidence",
    "validate_stock_mark_evidence_from_sources",
    "validate_stock_mark_parquet",
    "validate_pit_membership_evidence",
    "validate_suspension_evidence",
    "validate_terminal_settlement",
]
