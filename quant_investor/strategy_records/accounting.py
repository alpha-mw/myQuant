"""Derived-only portfolio accounting and attribution contracts.

This module never mutates Strategy Record Store-v3, holdings, performance, or
execution state.  It provides deterministic FIFO and daily-close mechanics for
the prospective accounting lane plus an immutable genesis contract that keeps
historical evidence gaps explicit.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from .store import canonical_json_bytes

ACCOUNTING_GENESIS_SCHEMA: Final = "myquant.strategy_accounting_genesis.v1"
ACCOUNTING_POINTER_SCHEMA: Final = "myquant.strategy_accounting_pointer.v1"
HISTORICAL_GAP_AUDIT_SCHEMA: Final = "myquant.strategy_accounting_historical_gap_audit.v1"
PROSPECTIVE_OPENING_BALANCE = "PROSPECTIVE_OPENING_BALANCE"
_SHA: Final = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL: Final = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MONEY = Decimal("0.0001")
_UNIT = Decimal("0.00000001")


class StrategyAccountingError(RuntimeError):
    """Fail-closed accounting contract error."""


def _money(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise StrategyAccountingError(f"{label} is not money")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise StrategyAccountingError(f"{label} is not money") from exc
    if not result.is_finite():
        raise StrategyAccountingError(f"{label} is not finite")
    return result.quantize(_MONEY, rounding=ROUND_HALF_EVEN)


def money_text(value: Decimal) -> str:
    return format(value.quantize(_MONEY, rounding=ROUND_HALF_EVEN), "f")


def _unit(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise StrategyAccountingError(f"{label} is not a unit amount")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise StrategyAccountingError(f"{label} is not a unit amount") from exc
    if not result.is_finite():
        raise StrategyAccountingError(f"{label} is not finite")
    return result.quantize(_UNIT, rounding=ROUND_HALF_EVEN)


def _unit_text(value: Decimal) -> str:
    return format(value.quantize(_UNIT, rounding=ROUND_HALF_EVEN), "f")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _content_sha(document: Mapping[str, Any]) -> str:
    body = dict(document)
    body.pop("content_sha256", None)
    return _sha256(canonical_json_bytes(body))


def seal_document(document: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(document)
    value["content_sha256"] = _content_sha(value)
    canonical_json_bytes(value)
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise StrategyAccountingError(f"{label} is not SHA-256")
    return value


def _require_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise StrategyAccountingError(f"{label} is not an identifier")
    return value


def _require_date(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise StrategyAccountingError(f"{label} is not a date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise StrategyAccountingError(f"{label} is not a date") from exc
    if parsed.isoformat() != value:
        raise StrategyAccountingError(f"{label} is not canonical")
    return value


def _require_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise StrategyAccountingError(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise StrategyAccountingError(f"{label} is not a UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise StrategyAccountingError(f"{label} is not canonical")
    return value


def _require_source_refs(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise StrategyAccountingError("source_refs are absent")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise StrategyAccountingError("source_ref shape is invalid")
        path = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or ".." in Path(path).parts
            or path in seen
        ):
            raise StrategyAccountingError("source_ref path is invalid")
        rows.append({"path": path, "sha256": _require_sha(digest, label="source_ref")})
        seen.add(path)
    if rows != sorted(rows, key=lambda row: row["path"]):
        raise StrategyAccountingError("source_refs are not sorted")
    return rows


def effective_fills(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return exact applied fills after deterministic correction/duplicate rules."""

    fills: dict[str, dict[str, Any]] = {}
    voided: set[str] = set()
    for raw in events:
        if not isinstance(raw, Mapping):
            raise StrategyAccountingError("trade event is not an object")
        event = dict(raw)
        event_id = _require_identifier(event.get("event_id"), label="event_id")
        if event_id in fills or event_id in voided:
            raise StrategyAccountingError("duplicate trade event_id")
        event_type = event.get("event_type")
        if event_type == "CORRECTION_VOID":
            corrected = _require_identifier(
                event.get("corrects_event_id"), label="corrects_event_id"
            )
            if corrected not in fills or corrected in voided:
                raise StrategyAccountingError("correction target is invalid")
            if set(event) != {"event_id", "event_type", "corrects_event_id"}:
                raise StrategyAccountingError("correction event has economic fields")
            voided.add(corrected)
            continue
        if event_type != "FILL" or event.get("corrects_event_id") is not None:
            raise StrategyAccountingError("trade event type is invalid")
        symbol = event.get("symbol")
        if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
            raise StrategyAccountingError("trade symbol is invalid")
        if event.get("side") not in {"BUY", "SELL"}:
            raise StrategyAccountingError("trade side is invalid")
        shares = event.get("shares")
        if not isinstance(shares, int) or isinstance(shares, bool) or shares <= 0:
            raise StrategyAccountingError("trade shares are invalid")
        _require_date(event.get("trade_date"), label="trade_date")
        price = _money(event.get("price_cny"), label="price_cny")
        if price <= 0:
            raise StrategyAccountingError("trade price is not positive")
        fee_status = event.get("fee_status")
        fee = event.get("total_fee_cny")
        if fee_status == "KNOWN":
            fee_value = _money(fee, label="total_fee_cny")
            if fee_value < 0:
                raise StrategyAccountingError("trade fee is negative")
        elif fee_status in {"LEGACY_UNAVAILABLE", "EVIDENCE_UNAVAILABLE"}:
            if fee is not None:
                raise StrategyAccountingError("unknown trade fee must be null")
        else:
            raise StrategyAccountingError("trade fee_status is invalid")
        fills[event_id] = event
    return [fills[event_id] for event_id in fills if event_id not in voided]


def apply_fifo_events(
    opening_lots: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply exact fills to opening balances using FIFO cost accounting."""

    lots: dict[str, list[dict[str, Any]]] = {}
    seen_lots: set[str] = set()
    for raw in opening_lots:
        lot = dict(raw)
        lot_id = _require_identifier(lot.get("lot_id"), label="lot_id")
        symbol = lot.get("symbol")
        shares = lot.get("remaining_shares")
        if (
            lot_id in seen_lots
            or not isinstance(symbol, str)
            or _SYMBOL.fullmatch(symbol) is None
            or not isinstance(shares, int)
            or isinstance(shares, bool)
            or shares <= 0
            or lot.get("origin") != PROSPECTIVE_OPENING_BALANCE
        ):
            raise StrategyAccountingError("opening lot is invalid")
        unit_cost = _unit(lot.get("unit_cost_cny"), label="opening unit_cost_cny")
        if unit_cost < 0:
            raise StrategyAccountingError("opening unit cost is negative")
        lot["unit_cost_cny"] = _unit_text(unit_cost)
        lots.setdefault(symbol, []).append(lot)
        seen_lots.add(lot_id)

    realized_rows: list[dict[str, Any]] = []
    fees_complete = True
    for event in effective_fills(events):
        event_id = str(event["event_id"])
        symbol = str(event["symbol"])
        shares = int(event["shares"])
        price = _money(event["price_cny"], label="event price")
        fee_known = event["fee_status"] == "KNOWN"
        fee = _money(event["total_fee_cny"], label="event fee") if fee_known else None
        fees_complete = fees_complete and fee_known
        if event["side"] == "BUY":
            gross_cost = _money(price * shares, label="buy gross cost")
            book_cost = gross_cost + (fee or Decimal("0"))
            lots.setdefault(symbol, []).append(
                {
                    "lot_id": "lot-" + event_id,
                    "symbol": symbol,
                    "remaining_shares": shares,
                    "unit_cost_cny": _unit_text(book_cost / shares),
                    "origin": "PROSPECTIVE_FILL",
                    "open_event_id": event_id,
                    "fee_status": event["fee_status"],
                }
            )
            continue
        available = sum(int(lot["remaining_shares"]) for lot in lots.get(symbol, []))
        if available < shares:
            raise StrategyAccountingError("SELL exceeds FIFO shares")
        remaining = shares
        cost = Decimal("0")
        consumed: list[dict[str, Any]] = []
        for lot in lots.get(symbol, []):
            if remaining == 0:
                break
            take = min(remaining, int(lot["remaining_shares"]))
            cost += _unit(lot["unit_cost_cny"], label="lot unit cost") * take
            lot["remaining_shares"] = int(lot["remaining_shares"]) - take
            consumed.append({"lot_id": lot["lot_id"], "shares": take})
            remaining -= take
        gross_proceeds = _money(price * shares, label="sell gross proceeds")
        realized_rows.append(
            {
                "event_id": event_id,
                "symbol": symbol,
                "shares": shares,
                "consumed_lots": consumed,
                "cost_basis_cny": money_text(cost),
                "gross_proceeds_cny": money_text(gross_proceeds),
                "gross_realized_pnl_cny": money_text(gross_proceeds - cost),
                "net_realized_pnl_cny": (
                    money_text(gross_proceeds - cost - fee) if fee is not None else None
                ),
                "fee_status": event["fee_status"],
            }
        )
    remaining_lots = [
        lot for symbol in sorted(lots) for lot in lots[symbol] if int(lot["remaining_shares"]) > 0
    ]
    return {
        "remaining_lots": remaining_lots,
        "realized_rows": realized_rows,
        "fees_complete": fees_complete,
    }


def build_daily_close(
    *,
    trade_date: str,
    opening_cash_cny: Any,
    opening_nav_cny: Any,
    opening_realized_pnl_cny: Any,
    opening_unrealized_pnl_cny: Any,
    closing_cash_cny: Any,
    closing_market_value_cny: Any,
    closing_realized_pnl_cny: Any | None,
    closing_unrealized_pnl_cny: Any,
    external_flow_cny: Any,
    other_pnl_cny: Any,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one prospective close with cash/NAV and P&L reconciliation."""

    _require_date(trade_date, label="trade_date")
    opening_cash = _money(opening_cash_cny, label="opening_cash")
    opening_nav = _money(opening_nav_cny, label="opening_nav")
    opening_realized = _money(opening_realized_pnl_cny, label="opening_realized")
    opening_unrealized = _money(opening_unrealized_pnl_cny, label="opening_unrealized")
    closing_cash = _money(closing_cash_cny, label="closing_cash")
    closing_market = _money(closing_market_value_cny, label="closing_market")
    closing_unrealized = _money(closing_unrealized_pnl_cny, label="closing_unrealized")
    flow = _money(external_flow_cny, label="external_flow")
    other = _money(other_pnl_cny, label="other_pnl")
    fills = effective_fills(events)
    unknown_fees = [event["event_id"] for event in fills if event["fee_status"] != "KNOWN"]
    buys = sum(
        (
            _money(event["price_cny"], label="buy price") * int(event["shares"])
            for event in fills
            if event["side"] == "BUY"
        ),
        Decimal("0"),
    )
    sells = sum(
        (
            _money(event["price_cny"], label="sell price") * int(event["shares"])
            for event in fills
            if event["side"] == "SELL"
        ),
        Decimal("0"),
    )
    fees = (
        sum((_money(event["total_fee_cny"], label="fee") for event in fills), Decimal("0"))
        if not unknown_fees
        else None
    )
    expected_cash = opening_cash + sells - buys - fees + flow if fees is not None else None
    closing_nav = closing_cash + closing_market
    daily_pnl = closing_nav - opening_nav - flow
    closing_realized = (
        _money(closing_realized_pnl_cny, label="closing_realized")
        if closing_realized_pnl_cny is not None
        else None
    )
    pnl_bridge = (
        closing_realized - opening_realized + closing_unrealized - opening_unrealized + other
        if closing_realized is not None
        else None
    )
    blockers: list[str] = []
    if expected_cash is None:
        blockers.append("FEE_EVIDENCE_UNAVAILABLE")
    elif expected_cash != closing_cash:
        blockers.append("CASH_RECONCILIATION_FAILED")
    if pnl_bridge is None:
        blockers.append("NET_REALIZED_PNL_UNAVAILABLE")
    elif pnl_bridge != daily_pnl:
        blockers.append("PNL_RECONCILIATION_FAILED")
    return {
        "trade_date": trade_date,
        "status": "VERIFIED" if not blockers else "BLOCKED",
        "opening_cash_cny": money_text(opening_cash),
        "closing_cash_cny": money_text(closing_cash),
        "expected_closing_cash_cny": (
            money_text(expected_cash) if expected_cash is not None else None
        ),
        "closing_market_value_cny": money_text(closing_market),
        "closing_nav_cny": money_text(closing_nav),
        "external_flow_cny": money_text(flow),
        "daily_pnl_cny": money_text(daily_pnl),
        "pnl_bridge_cny": money_text(pnl_bridge) if pnl_bridge is not None else None,
        "fees_cny": money_text(fees) if fees is not None else None,
        "unknown_fee_event_ids": unknown_fees,
        "blockers": blockers,
    }


def build_genesis(
    *,
    generation_id: str,
    created_at: str,
    strategy_label: str,
    effective_date: str,
    source_store: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    cash_cny: Any,
    nav_cny: Any,
    portfolio_pnl_cny: Any,
    positions: Sequence[Mapping[str, Any]],
    historical_audit_ref: Mapping[str, Any],
    industry_rows: Sequence[Mapping[str, Any]],
    theme_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the derived-only prospective accounting genesis."""

    _require_identifier(generation_id, label="generation_id")
    _require_timestamp(created_at, label="created_at")
    _require_identifier(strategy_label, label="strategy_label")
    _require_date(effective_date, label="effective_date")
    refs = _require_source_refs(list(source_refs))
    required_store = {
        "pointer_sha256",
        "catalog_generation_id",
        "catalog_sha256",
        "performance_generation_id",
        "performance_manifest_sha256",
        "performance_series_sha256",
        "active_record_id",
        "active_ledger_sha256",
    }
    if not isinstance(source_store, Mapping) or set(source_store) != required_store:
        raise StrategyAccountingError("source_store shape is invalid")
    for name in (
        "pointer_sha256",
        "catalog_sha256",
        "performance_manifest_sha256",
        "performance_series_sha256",
        "active_ledger_sha256",
    ):
        _require_sha(source_store.get(name), label=name)
    cash = _money(cash_cny, label="cash")
    nav = _money(nav_cny, label="nav")
    portfolio_pnl = _money(portfolio_pnl_cny, label="portfolio_pnl")
    normalized_positions: list[dict[str, Any]] = []
    opening_lots: list[dict[str, Any]] = []
    symbols: set[str] = set()
    market_value = Decimal("0")
    cost_basis = Decimal("0")
    for raw in positions:
        if not isinstance(raw, Mapping):
            raise StrategyAccountingError("position is invalid")
        symbol = raw.get("symbol")
        shares = raw.get("shares")
        if (
            not isinstance(symbol, str)
            or _SYMBOL.fullmatch(symbol) is None
            or symbol in symbols
            or not isinstance(shares, int)
            or isinstance(shares, bool)
            or shares <= 0
        ):
            raise StrategyAccountingError("position identity is invalid")
        avg_cost = _unit(raw.get("avg_cost_cny"), label="avg_cost")
        row_cost = _money(raw.get("cost_basis_cny"), label="cost_basis")
        row_market = _money(raw.get("market_value_cny"), label="market_value")
        if row_cost != _money(avg_cost * shares, label="position cost identity"):
            raise StrategyAccountingError("position cost basis does not close")
        normalized_positions.append(
            {
                "symbol": symbol,
                "name": str(raw.get("name") or ""),
                "shares": shares,
                "avg_cost_cny": _unit_text(avg_cost),
                "cost_basis_cny": money_text(row_cost),
                "market_value_cny": money_text(row_market),
                "unrealized_pnl_cny": money_text(row_market - row_cost),
            }
        )
        opening_lots.append(
            {
                "lot_id": "opening-" + symbol.replace(".", "-").lower(),
                "symbol": symbol,
                "remaining_shares": shares,
                "unit_cost_cny": _unit_text(avg_cost),
                "origin": PROSPECTIVE_OPENING_BALANCE,
                "source_ledger_sha256": source_store["active_ledger_sha256"],
                "historical_acquisition_date": None,
                "historical_realized_pnl_cny": None,
            }
        )
        symbols.add(symbol)
        market_value += row_market
        cost_basis += row_cost
    normalized_positions.sort(key=lambda row: row["symbol"])
    opening_lots.sort(key=lambda row: row["symbol"])
    if nav != cash + market_value:
        raise StrategyAccountingError("genesis NAV does not close")
    unrealized = market_value - cost_basis
    historical_unallocated = portfolio_pnl - unrealized
    if not isinstance(historical_audit_ref, Mapping) or set(historical_audit_ref) != {
        "path",
        "sha256",
    }:
        raise StrategyAccountingError("historical_audit_ref is invalid")
    _require_sha(historical_audit_ref.get("sha256"), label="historical audit")
    industry = [dict(row) for row in industry_rows]
    if {row.get("symbol") for row in industry} != symbols:
        raise StrategyAccountingError("Industry coverage is incomplete")
    themes = [dict(row) for row in theme_rows]
    by_symbol: dict[str, Decimal] = {symbol: Decimal("0") for symbol in symbols}
    for row in themes:
        symbol = row.get("symbol")
        if symbol not in by_symbol:
            raise StrategyAccountingError("Theme row symbol is invalid")
        weight = Decimal(str(row.get("weight")))
        if not weight.is_finite() or weight < 0 or weight > 1:
            raise StrategyAccountingError("Theme weight is invalid")
        by_symbol[symbol] += weight
    if any(value != Decimal("1") for value in by_symbol.values()):
        raise StrategyAccountingError("Theme weights do not sum to one")
    blockers = [
        "HISTORICAL_SHARE_TRANSITIONS_UNEXPLAINED",
        "LEGACY_FEES_UNAVAILABLE",
        "HISTORICAL_REALIZED_PNL_UNALLOCATED",
        "HISTORICAL_SECURITY_CONTRIBUTION_UNAVAILABLE",
        "THEME_ECONOMIC_EXPOSURE_UNCLASSIFIED",
    ]
    return seal_document(
        {
            "schema_id": ACCOUNTING_GENESIS_SCHEMA,
            "generation_id": generation_id,
            "created_at": created_at,
            "strategy_label": strategy_label,
            "effective_date": effective_date,
            "derived_only": True,
            "authority": {
                "store_mutation": False,
                "performance_mutation": False,
                "holdings_mutation": False,
                "cash_mutation": False,
                "broker": False,
                "order": False,
                "trade": False,
            },
            "source_store": dict(source_store),
            "source_refs": refs,
            "historical_gap_audit_ref": dict(historical_audit_ref),
            "coverage": {
                "historical": "PARTIAL",
                "prospective": "READY",
                "prospective_effective_date": effective_date,
            },
            "status": {
                "data": "VERIFIED",
                "accounting": "PARTIAL",
                "attribution": "PARTIAL",
                "evidence": "PARTIAL",
            },
            "blockers": blockers,
            "cash_cny": money_text(cash),
            "market_value_cny": money_text(market_value),
            "nav_cny": money_text(nav),
            "cost_basis_cny": money_text(cost_basis),
            "unrealized_pnl_cny": money_text(unrealized),
            "historical_unallocated_pnl_cny": money_text(historical_unallocated),
            "positions": normalized_positions,
            "opening_lots": opening_lots,
            "industry_rows": industry,
            "primary_theme_rows": themes,
        }
    )


def validate_genesis(document: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(document)
    if value.get("schema_id") != ACCOUNTING_GENESIS_SCHEMA:
        raise StrategyAccountingError("accounting genesis schema is invalid")
    if value.get("content_sha256") != _content_sha(value):
        raise StrategyAccountingError("accounting genesis content SHA differs")
    if value.get("derived_only") is not True or any(value.get("authority", {}).values()):
        raise StrategyAccountingError("accounting genesis claims authority")
    if value.get("coverage") != {
        "historical": "PARTIAL",
        "prospective": "READY",
        "prospective_effective_date": value.get("effective_date"),
    }:
        raise StrategyAccountingError("accounting coverage is invalid")
    if value.get("status") != {
        "data": "VERIFIED",
        "accounting": "PARTIAL",
        "attribution": "PARTIAL",
        "evidence": "PARTIAL",
    }:
        raise StrategyAccountingError("accounting status overclaims completeness")
    _require_source_refs(value.get("source_refs"))
    cash = _money(value.get("cash_cny"), label="cash")
    market = _money(value.get("market_value_cny"), label="market")
    nav = _money(value.get("nav_cny"), label="nav")
    if nav != cash + market:
        raise StrategyAccountingError("accounting genesis NAV does not close")
    if not isinstance(value.get("blockers"), list) or not value["blockers"]:
        raise StrategyAccountingError("historical blockers are absent")
    return value


def load_accounting_generation(record_root: Path) -> dict[str, Any]:
    """Load one exact accounting pointer/genesis/audit and report Store drift."""

    root = record_root.resolve(strict=True)
    pointer_path = root / "_accounting_store/current.v1.json"
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise StrategyAccountingError("accounting pointer is unavailable")
    pointer_raw = pointer_path.read_bytes()
    if pointer_raw != pointer_path.read_bytes():
        raise StrategyAccountingError("accounting pointer is unstable")
    try:
        pointer = json.loads(pointer_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyAccountingError("accounting pointer is invalid JSON") from exc
    if (
        not isinstance(pointer, dict)
        or pointer.get("schema_id") != ACCOUNTING_POINTER_SCHEMA
        or pointer.get("content_sha256") != _content_sha(pointer)
        or pointer.get("derived_only") is not True
        or pointer.get("store_mutation_authority") is not False
        or pointer.get("holdings_mutation_authority") is not False
        or pointer.get("broker_order_trade_authority") is not False
    ):
        raise StrategyAccountingError("accounting pointer contract is invalid")
    relative = pointer.get("genesis_path")
    if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
        raise StrategyAccountingError("accounting genesis path is invalid")
    genesis_path = root / relative
    if not genesis_path.is_file() or genesis_path.is_symlink():
        raise StrategyAccountingError("accounting genesis is unavailable")
    genesis_raw = genesis_path.read_bytes()
    if genesis_raw != genesis_path.read_bytes() or _sha256(genesis_raw) != pointer.get(
        "genesis_sha256"
    ):
        raise StrategyAccountingError("accounting genesis bytes differ")
    try:
        genesis = validate_genesis(json.loads(genesis_raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyAccountingError("accounting genesis is invalid JSON") from exc
    audit_ref = genesis["historical_gap_audit_ref"]
    audit_path = root / str(audit_ref["path"])
    if not audit_path.is_file() or audit_path.is_symlink():
        raise StrategyAccountingError("historical audit is unavailable")
    audit_raw = audit_path.read_bytes()
    if (
        audit_raw != audit_path.read_bytes()
        or _sha256(audit_raw) != audit_ref["sha256"]
        or audit_ref["sha256"] != pointer.get("historical_audit_sha256")
    ):
        raise StrategyAccountingError("historical audit bytes differ")
    try:
        audit = json.loads(audit_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyAccountingError("historical audit is invalid JSON") from exc
    if (
        not isinstance(audit, dict)
        or audit.get("schema_id") != HISTORICAL_GAP_AUDIT_SCHEMA
        or audit.get("content_sha256") != _content_sha(audit)
        or audit.get("status") != "HISTORICAL_PARTIAL"
        or audit.get("prospective_lot_authority") is not False
    ):
        raise StrategyAccountingError("historical audit contract is invalid")
    store_pointer_path = root / "_record_store/current.v1.json"
    if not store_pointer_path.is_file() or store_pointer_path.is_symlink():
        raise StrategyAccountingError("source Store pointer is unavailable")
    store_raw = store_pointer_path.read_bytes()
    if store_raw != store_pointer_path.read_bytes():
        raise StrategyAccountingError("source Store pointer is unstable")
    current_store_sha = _sha256(store_raw)
    source_store_sha = pointer["source_store_pointer_sha256"]
    state = "VERIFIED" if current_store_sha == source_store_sha else "SOURCE_STORE_ADVANCED"
    return {
        "state": state,
        "pointer": pointer,
        "pointer_sha256": _sha256(pointer_raw),
        "genesis": genesis,
        "audit": audit,
        "current_store_pointer_sha256": current_store_sha,
        "source_store_pointer_sha256": source_store_sha,
    }


def immutable_write(path: Path, raw: bytes, *, max_bytes: int = 8 * 1024 * 1024) -> str:
    if not raw or len(raw) > max_bytes:
        raise StrategyAccountingError("accounting artifact exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise StrategyAccountingError("accounting immutable artifact conflicts")
        return _sha256(raw)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    if path.read_bytes() != raw:
        raise StrategyAccountingError("accounting immutable write readback differs")
    return _sha256(raw)


__all__ = [
    "ACCOUNTING_GENESIS_SCHEMA",
    "ACCOUNTING_POINTER_SCHEMA",
    "HISTORICAL_GAP_AUDIT_SCHEMA",
    "PROSPECTIVE_OPENING_BALANCE",
    "StrategyAccountingError",
    "apply_fifo_events",
    "build_daily_close",
    "build_genesis",
    "effective_fills",
    "immutable_write",
    "load_accounting_generation",
    "money_text",
    "seal_document",
    "validate_genesis",
]
