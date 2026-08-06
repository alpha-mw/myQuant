"""Exact current-holdings pointer/manifest/Parquet resolution."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
import os
import re
from typing import Any, Final, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import (
    ArtifactRef,
    HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID,
    HOLDINGS_LEDGER_SCHEMA_ID,
    HOLDINGS_MANIFEST_SCHEMA_ID,
    HOLDINGS_POINTER_SCHEMA_ID,
    HOLDINGS_PRICE_SOURCE_SCHEMA_ID,
    HoldingPosition,
    MoneyTotals,
    PROTOCOL,
    PortfolioCycleError,
    VerifiedHoldingsBaseline,
    parse_canonical_json,
    require_date,
    require_exact_fields,
    require_identifier,
    require_money,
    require_sha256,
    require_timestamp,
    validate_artifact_ref,
)
from .exact_io import ExactReader

_POINTER_FIELDS: Final = frozenset(
    {
        "schema_id",
        "protocol",
        "canonical_strategy_id",
        "updated_at",
        "manifest_ref",
        "semantic_sha256",
    }
)
_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_id",
        "protocol",
        "canonical_strategy_id",
        "account_id",
        "currency",
        "trade_date",
        "as_of",
        "valuation_at",
        "decision_cutoff",
        "accounting_policy_ref",
        "price_source_ref",
        "ledger_ref",
        "contributed_capital",
        "cash",
        "total_cost_basis",
        "total_market_value",
        "total_unrealized_pnl",
        "total_realized_pnl",
        "nav",
        "semantic_sha256",
    }
)
_ACCOUNTING_POLICY_FIELDS: Final = frozenset(
    {
        "schema_id",
        "protocol",
        "currency",
        "money_scale",
        "rounding_mode",
        "capital_identity",
        "semantic_sha256",
    }
)
_PRICE_SOURCE_FIELDS: Final = frozenset(
    {
        "schema_id",
        "protocol",
        "currency",
        "source_id",
        "as_of",
        "valuation_at",
        "semantic_sha256",
    }
)
_SYMBOL: Final = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_SCALE_4: Final = Decimal("0.0001")
_REQUIRED_LEDGER_TYPES: Final = {
    "symbol": pa.string(),
    "name": pa.string(),
    "shares": pa.int64(),
    "avg_cost": pa.decimal128(20, 4),
    "market_price": pa.decimal128(20, 4),
    "cost_basis": pa.decimal128(20, 4),
    "market_value": pa.decimal128(20, 4),
    "unrealized_pnl": pa.decimal128(20, 4),
    "realized_pnl": pa.decimal128(20, 4),
}


def _scale4(value: Decimal) -> Decimal:
    return value.quantize(_SCALE_4, rounding=ROUND_HALF_EVEN)


def _validate_pointer(
    value: Mapping[str, Any],
) -> tuple[str, str, ArtifactRef]:
    code = "PORTFOLIO_CYCLE_HOLDINGS_POINTER_INVALID"
    document = require_exact_fields(value, _POINTER_FIELDS, label="holdings pointer", code=code)
    if (
        document.get("schema_id") != HOLDINGS_POINTER_SCHEMA_ID
        or document.get("protocol") != PROTOCOL
    ):
        raise PortfolioCycleError(code, "holdings pointer schema/protocol mismatch")
    strategy = require_identifier(
        document.get("canonical_strategy_id"),
        label="canonical_strategy_id",
        code=code,
    )
    updated_at = document.get("updated_at")
    require_timestamp(updated_at, label="updated_at", code=code)
    manifest_ref = validate_artifact_ref(
        document.get("manifest_ref"),
        label="manifest_ref",
        code=code,
        expected_schema_id=HOLDINGS_MANIFEST_SCHEMA_ID,
    )
    return strategy, updated_at, manifest_ref


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    code = "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID"
    document = require_exact_fields(value, _MANIFEST_FIELDS, label="holdings manifest", code=code)
    if (
        document.get("schema_id") != HOLDINGS_MANIFEST_SCHEMA_ID
        or document.get("protocol") != PROTOCOL
    ):
        raise PortfolioCycleError(code, "holdings manifest schema/protocol mismatch")
    strategy = require_identifier(
        document.get("canonical_strategy_id"),
        label="canonical_strategy_id",
        code=code,
    )
    account_id = require_identifier(document.get("account_id"), label="account_id", code=code)
    if document.get("currency") != "CNY":
        raise PortfolioCycleError(code, "holdings currency must be CNY")
    trade_date_text = document.get("trade_date")
    trade_date = require_date(trade_date_text, label="trade_date", code=code)
    as_of_text = document.get("as_of")
    valuation_text = document.get("valuation_at")
    cutoff_text = document.get("decision_cutoff")
    as_of = require_timestamp(as_of_text, label="as_of", code=code)
    valuation_at = require_timestamp(valuation_text, label="valuation_at", code=code)
    decision_cutoff = require_timestamp(cutoff_text, label="decision_cutoff", code=code)
    if (
        trade_date > as_of.date()
        or trade_date > valuation_at.date()
        or trade_date > decision_cutoff.date()
        or not (as_of <= valuation_at <= decision_cutoff)
    ):
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_CHRONOLOGY_INVALID",
            "manifest time order is invalid",
        )
    totals = MoneyTotals(
        contributed_capital=require_money(
            document.get("contributed_capital"),
            label="contributed_capital",
            nonnegative=True,
        ),
        cash=require_money(document.get("cash"), label="cash", nonnegative=True),
        cost_basis=require_money(
            document.get("total_cost_basis"),
            label="total_cost_basis",
            nonnegative=True,
        ),
        market_value=require_money(
            document.get("total_market_value"),
            label="total_market_value",
            nonnegative=True,
        ),
        unrealized_pnl=require_money(
            document.get("total_unrealized_pnl"), label="total_unrealized_pnl"
        ),
        realized_pnl=require_money(document.get("total_realized_pnl"), label="total_realized_pnl"),
        nav=require_money(document.get("nav"), label="nav", nonnegative=True),
    )
    return {
        "canonical_strategy_id": strategy,
        "account_id": account_id,
        "currency": "CNY",
        "trade_date": trade_date_text,
        "as_of": as_of_text,
        "as_of_dt": as_of,
        "valuation_at": valuation_text,
        "valuation_at_dt": valuation_at,
        "decision_cutoff": cutoff_text,
        "decision_cutoff_dt": decision_cutoff,
        "accounting_policy_ref": validate_artifact_ref(
            document.get("accounting_policy_ref"),
            label="accounting_policy_ref",
            code=code,
            expected_schema_id=HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID,
        ),
        "price_source_ref": validate_artifact_ref(
            document.get("price_source_ref"),
            label="price_source_ref",
            code=code,
            expected_schema_id=HOLDINGS_PRICE_SOURCE_SCHEMA_ID,
        ),
        "ledger_ref": validate_artifact_ref(
            document.get("ledger_ref"),
            label="ledger_ref",
            code=code,
            expected_schema_id=HOLDINGS_LEDGER_SCHEMA_ID,
        ),
        "totals": totals,
    }


def _read_supporting_json(reader: ExactReader, ref: ArtifactRef, *, label: str) -> dict[str, Any]:
    stored = reader.read(ref.relative_path, expected_sha256=ref.byte_sha256)
    document = parse_canonical_json(stored.data)
    if document.get("schema_id") != ref.schema_id:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID",
            f"{label} schema_id does not match its exact ref",
        )
    return document


def _validate_accounting_policy(document: Mapping[str, Any]) -> None:
    code = "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID"
    value = require_exact_fields(
        document,
        _ACCOUNTING_POLICY_FIELDS,
        label="accounting policy",
        code=code,
    )
    if (
        value.get("schema_id") != HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID
        or value.get("protocol") != PROTOCOL
        or value.get("currency") != "CNY"
        or value.get("money_scale") != 4
        or value.get("rounding_mode") != "ROUND_HALF_EVEN"
        or value.get("capital_identity")
        != "NAV_EQUALS_CONTRIBUTED_CAPITAL_PLUS_REALIZED_PLUS_UNREALIZED"
    ):
        raise PortfolioCycleError(code, "accounting policy does not match the frozen CNY policy")


def _validate_price_source(document: Mapping[str, Any], *, as_of: str, valuation_at: str) -> None:
    code = "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID"
    value = require_exact_fields(
        document,
        _PRICE_SOURCE_FIELDS,
        label="price source",
        code=code,
    )
    if (
        value.get("schema_id") != HOLDINGS_PRICE_SOURCE_SCHEMA_ID
        or value.get("protocol") != PROTOCOL
        or value.get("currency") != "CNY"
    ):
        raise PortfolioCycleError(code, "price source schema/protocol/currency mismatch")
    require_identifier(value.get("source_id"), label="price_source.source_id", code=code)
    source_as_of = value.get("as_of")
    source_valuation = value.get("valuation_at")
    require_timestamp(source_as_of, label="price_source.as_of", code=code)
    require_timestamp(source_valuation, label="price_source.valuation_at", code=code)
    if source_as_of != as_of or source_valuation != valuation_at:
        raise PortfolioCycleError(code, "price source does not bind manifest valuation timestamps")


def _parse_ledger(raw: bytes) -> tuple[HoldingPosition, ...]:
    code = "PORTFOLIO_CYCLE_HOLDINGS_LEDGER_INVALID"
    try:
        parquet = pq.ParquetFile(pa.BufferReader(raw))
        metadata = parquet.metadata
        if metadata.num_rows > 10_000 or metadata.num_columns > 32:
            raise PortfolioCycleError(code, "holdings ledger exceeds row/column bounds")
        table = parquet.read()
    except PortfolioCycleError:
        raise
    except Exception as exc:
        raise PortfolioCycleError(code, "holdings ledger is not valid Parquet") from exc
    if table.num_rows > 10_000 or table.num_columns > 32:
        raise PortfolioCycleError(code, "holdings ledger exceeds row/column bounds")
    schema_metadata = table.schema.metadata or {}
    if schema_metadata.get(b"schema_id") != HOLDINGS_LEDGER_SCHEMA_ID.encode("ascii"):
        raise PortfolioCycleError(code, "holdings ledger schema_id metadata mismatch")
    names = set(table.column_names)
    if not set(_REQUIRED_LEDGER_TYPES).issubset(names):
        raise PortfolioCycleError(code, "holdings ledger required columns are missing")
    for column_name, expected_type in _REQUIRED_LEDGER_TYPES.items():
        field = table.schema.field(column_name)
        if field.type != expected_type or field.nullable:
            raise PortfolioCycleError(
                code,
                "holdings ledger column " f"{column_name} has invalid type/nullability",
            )
    if any(table.column(index).null_count for index in range(table.num_columns)):
        raise PortfolioCycleError(code, "holdings ledger contains null values")

    columns = {name: table.column(name).to_pylist() for name in _REQUIRED_LEDGER_TYPES}
    symbols = columns["symbol"]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise PortfolioCycleError(code, "holdings symbols must be strictly ascending and unique")

    positions: list[HoldingPosition] = []
    for index, symbol in enumerate(symbols):
        if type(symbol) is not str or _SYMBOL.fullmatch(symbol) is None:
            raise PortfolioCycleError(code, "holdings ledger contains an invalid A-share symbol")
        name = columns["name"][index]
        if type(name) is not str or not name or name != name.strip() or len(name) > 200:
            raise PortfolioCycleError(code, "holdings ledger contains an invalid security name")
        shares = columns["shares"][index]
        if type(shares) is not int or shares <= 0:
            raise PortfolioCycleError(code, "holdings shares must be positive int64")
        avg_cost = columns["avg_cost"][index]
        market_price = columns["market_price"][index]
        cost_basis = columns["cost_basis"][index]
        market_value = columns["market_value"][index]
        unrealized = columns["unrealized_pnl"][index]
        realized = columns["realized_pnl"][index]
        monetary = (
            avg_cost,
            market_price,
            cost_basis,
            market_value,
            unrealized,
            realized,
        )
        if any(type(item) is not Decimal or not item.is_finite() for item in monetary):
            raise PortfolioCycleError(code, "holdings ledger contains invalid decimal values")
        if avg_cost < 0 or market_price < 0 or cost_basis < 0 or market_value < 0:
            raise PortfolioCycleError(code, "holdings prices and values must be nonnegative")
        if cost_basis != _scale4(Decimal(shares) * avg_cost):
            raise PortfolioCycleError(code, "row cost_basis identity mismatch")
        if market_value != _scale4(Decimal(shares) * market_price):
            raise PortfolioCycleError(code, "row market_value identity mismatch")
        if unrealized != _scale4(market_value - cost_basis):
            raise PortfolioCycleError(code, "row unrealized_pnl identity mismatch")
        positions.append(
            HoldingPosition(
                symbol=symbol,
                name=name,
                shares=shares,
                avg_cost=avg_cost,
                market_price=market_price,
                cost_basis=cost_basis,
                market_value=market_value,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
            )
        )
    return tuple(positions)


def _sum_positions(positions: tuple[HoldingPosition, ...], field: str) -> Decimal:
    return _scale4(sum((getattr(position, field) for position in positions), Decimal("0")))


def _validate_totals(totals: MoneyTotals, positions: tuple[HoldingPosition, ...]) -> None:
    expected = {
        "cost_basis": _sum_positions(positions, "cost_basis"),
        "market_value": _sum_positions(positions, "market_value"),
        "unrealized_pnl": _sum_positions(positions, "unrealized_pnl"),
        "realized_pnl": _sum_positions(positions, "realized_pnl"),
    }
    for field, value in expected.items():
        if getattr(totals, field) != value:
            raise PortfolioCycleError(
                "PORTFOLIO_CYCLE_HOLDINGS_ACCOUNTING_MISMATCH",
                f"manifest {field} does not equal the ledger row sum",
            )
    if totals.nav != _scale4(totals.cash + totals.market_value):
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_ACCOUNTING_MISMATCH",
            "NAV does not equal cash plus market value",
        )
    policy_nav = _scale4(totals.contributed_capital + totals.realized_pnl + totals.unrealized_pnl)
    if totals.nav != policy_nav:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_ACCOUNTING_MISMATCH",
            "NAV does not satisfy the exact accounting-policy capital " "identity",
        )


def resolve_holdings_baseline(
    workspace_root: str | os.PathLike[str],
    *,
    pointer_path: str,
    pointer_sha256: str,
    expected_strategy_id: str,
) -> VerifiedHoldingsBaseline:
    expected_strategy = require_identifier(
        expected_strategy_id,
        label="expected_strategy_id",
        code="PORTFOLIO_CYCLE_HOLDINGS_STRATEGY_MISMATCH",
    )
    reader = ExactReader(workspace_root)
    pointer_stored = reader.read(
        pointer_path,
        expected_sha256=require_sha256(pointer_sha256, label="pointer_sha256"),
    )
    pointer_document = parse_canonical_json(pointer_stored.data)
    pointer_strategy, pointer_updated_text, manifest_ref = _validate_pointer(pointer_document)
    if pointer_strategy != expected_strategy:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_STRATEGY_MISMATCH",
            "holdings pointer strategy does not match the explicit strategy",
        )
    manifest_stored = reader.read(
        manifest_ref.relative_path, expected_sha256=manifest_ref.byte_sha256
    )
    manifest_document = parse_canonical_json(manifest_stored.data)
    manifest = _validate_manifest(manifest_document)
    if manifest["canonical_strategy_id"] != expected_strategy:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_STRATEGY_MISMATCH",
            "holdings manifest strategy does not match the explicit strategy",
        )
    pointer_updated = require_timestamp(
        pointer_updated_text,
        label="pointer.updated_at",
        code="PORTFOLIO_CYCLE_HOLDINGS_POINTER_INVALID",
    )
    if manifest["decision_cutoff_dt"] > pointer_updated:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_CHRONOLOGY_INVALID",
            "decision cutoff is after pointer update",
        )

    policy_ref = manifest["accounting_policy_ref"]
    price_ref = manifest["price_source_ref"]
    ledger_ref = manifest["ledger_ref"]
    policy_document = _read_supporting_json(reader, policy_ref, label="accounting policy")
    _validate_accounting_policy(policy_document)
    price_document = _read_supporting_json(reader, price_ref, label="price source")
    _validate_price_source(
        price_document,
        as_of=manifest["as_of"],
        valuation_at=manifest["valuation_at"],
    )
    ledger_stored = reader.read(ledger_ref.relative_path, expected_sha256=ledger_ref.byte_sha256)
    positions = _parse_ledger(ledger_stored.data)
    _validate_totals(manifest["totals"], positions)

    return VerifiedHoldingsBaseline(
        verified=True,
        canonical_strategy_id=expected_strategy,
        account_id=manifest["account_id"],
        currency="CNY",
        trade_date=manifest["trade_date"],
        as_of=manifest["as_of"],
        valuation_at=manifest["valuation_at"],
        decision_cutoff=manifest["decision_cutoff"],
        pointer_updated_at=pointer_updated_text,
        pointer_ref=ArtifactRef(
            schema_id=HOLDINGS_POINTER_SCHEMA_ID,
            relative_path=pointer_stored.relative_path,
            byte_sha256=pointer_stored.byte_sha256,
        ),
        manifest_ref=manifest_ref,
        accounting_policy_ref=policy_ref,
        price_source_ref=price_ref,
        ledger_ref=ledger_ref,
        totals=manifest["totals"],
        positions=positions,
    )


__all__ = ["resolve_holdings_baseline"]
