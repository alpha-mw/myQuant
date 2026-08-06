"""Frozen contracts shared by the read-only V17 portfolio-cycle foundation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Final, Literal, Mapping

from quant_investor.v17_mainline.contracts import (
    MainlineContractError,
    canonical_bytes as _mainline_canonical_bytes,
    parse_canonical as _mainline_parse_canonical,
    seal_document as _mainline_seal_document,
)

PROTOCOL: Final = "myquant.v17.v4"
IDENTITY_DECLARATION_SCHEMA_ID: Final = "myquant.v17.v4.strategy-identity-declaration.v1"
HOLDINGS_POINTER_SCHEMA_ID: Final = "myquant.v17.v4.holdings-pointer.v1"
HOLDINGS_MANIFEST_SCHEMA_ID: Final = "myquant.v17.v4.holdings-manifest.v1"
HOLDINGS_LEDGER_SCHEMA_ID: Final = "myquant.v17.v4.current-holdings-ledger.v1"
HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID: Final = "myquant.v17.v4.holdings-accounting-policy.v1"
HOLDINGS_PRICE_SOURCE_SCHEMA_ID: Final = "myquant.v17.v4.holdings-price-source.v1"

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MONEY: Final = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{4}$")
_REF_FIELDS: Final = frozenset({"schema_id", "relative_path", "byte_sha256"})


class PortfolioCycleError(RuntimeError):
    """Stable fail-closed error for the portfolio-cycle foundation."""

    def __init__(self, code: str, detail: str) -> None:
        if type(code) is not str or not code.startswith("PORTFOLIO_CYCLE_"):
            raise ValueError("portfolio-cycle error code is invalid")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}")


@dataclass(frozen=True)
class ArtifactRef:
    schema_id: str
    relative_path: str
    byte_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "schema_id": self.schema_id,
            "relative_path": self.relative_path,
            "byte_sha256": self.byte_sha256,
        }


@dataclass(frozen=True)
class MoneyTotals:
    contributed_capital: Decimal
    cash: Decimal
    cost_basis: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    nav: Decimal


@dataclass(frozen=True)
class HoldingPosition:
    symbol: str
    name: str
    shares: int
    avg_cost: Decimal
    market_price: Decimal
    cost_basis: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal


@dataclass(frozen=True)
class VerifiedStrategyIdentity:
    verified: Literal[True]
    declaration_ref: ArtifactRef
    historical_label: str
    canonical_strategy_id: str
    declared_by: str
    declared_at: str
    authority_kind: Literal["owner_declaration"]
    provenance: str


@dataclass(frozen=True)
class VerifiedHoldingsBaseline:
    verified: Literal[True]
    canonical_strategy_id: str
    account_id: str
    currency: Literal["CNY"]
    trade_date: str
    as_of: str
    valuation_at: str
    decision_cutoff: str
    pointer_updated_at: str
    pointer_ref: ArtifactRef
    manifest_ref: ArtifactRef
    accounting_policy_ref: ArtifactRef
    price_source_ref: ArtifactRef
    ledger_ref: ArtifactRef
    totals: MoneyTotals
    positions: tuple[HoldingPosition, ...]


def canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        return _mainline_canonical_bytes(document)
    except MainlineContractError as exc:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_JSON_INVALID",
            "document is not canonical JSON data",
        ) from exc


def seal_document(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return _mainline_seal_document(document)
    except MainlineContractError as exc:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_JSON_INVALID",
            "document cannot be semantically sealed",
        ) from exc


def parse_canonical_json(raw: bytes) -> dict[str, Any]:
    try:
        return _mainline_parse_canonical(raw)
    except MainlineContractError as exc:
        message = str(exc)
        code = (
            "PORTFOLIO_CYCLE_SEMANTIC_SHA_MISMATCH"
            if "semantic SHA-256 mismatch" in message
            else "PORTFOLIO_CYCLE_JSON_INVALID"
        )
        raise PortfolioCycleError(code, message) from exc


def require_exact_fields(
    value: Any, expected: frozenset[str] | set[str], *, label: str, code: str
) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != set(expected):
        raise PortfolioCycleError(code, f"{label} fields are not exact")
    return value


def require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_CONTRACT_INVALID",
            f"{label} is not a canonical SHA-256",
        )
    return value


def require_identifier(value: Any, *, label: str, code: str) -> str:
    if type(value) is not str or len(value) > 80 or _IDENTIFIER.fullmatch(value) is None:
        raise PortfolioCycleError(code, f"{label} is not a canonical identifier")
    return value


def require_text(value: Any, *, label: str, code: str, max_length: int = 1000) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > max_length
        or "\x00" in value
    ):
        raise PortfolioCycleError(code, f"{label} is not canonical non-empty text")
    return value


def require_date(value: Any, *, label: str, code: str) -> date:
    if type(value) is not str or len(value) != 10:
        raise PortfolioCycleError(code, f"{label} is not a canonical date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PortfolioCycleError(code, f"{label} is not a real date") from exc
    if parsed.isoformat() != value:
        raise PortfolioCycleError(code, f"{label} is not a canonical date")
    return parsed


def require_timestamp(value: Any, *, label: str, code: str) -> datetime:
    if type(value) is not str:
        raise PortfolioCycleError(code, f"{label} is not a canonical UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise PortfolioCycleError(code, f"{label} is not a real canonical UTC timestamp") from exc
    return parsed


def require_money(value: Any, *, label: str, nonnegative: bool = False) -> Decimal:
    if type(value) is not str or _MONEY.fullmatch(value) is None or value == "-0.0000":
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID",
            f"{label} is not a canonical scale-4 decimal string",
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID",
            f"{label} is not a finite decimal",
        ) from exc
    if not parsed.is_finite() or parsed.as_tuple().exponent != -4:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID",
            f"{label} must have decimal scale 4",
        )
    if nonnegative and parsed < 0:
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID",
            f"{label} must be nonnegative",
        )
    return parsed


def validate_artifact_ref(
    value: Any,
    *,
    label: str,
    code: str,
    expected_schema_id: str | None = None,
) -> ArtifactRef:
    document = require_exact_fields(value, _REF_FIELDS, label=label, code=code)
    schema_id = document.get("schema_id")
    if type(schema_id) is not str or not schema_id.startswith("myquant."):
        raise PortfolioCycleError(code, f"{label}.schema_id is invalid")
    if expected_schema_id is not None and schema_id != expected_schema_id:
        raise PortfolioCycleError(code, f"{label}.schema_id mismatch")
    path = document.get("relative_path")
    if type(path) is not str:
        raise PortfolioCycleError(code, f"{label}.relative_path is invalid")
    # Full path security validation occurs in ExactReader.  Reject the obvious
    # non-canonical forms here so a ref is never represented ambiguously.
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise PortfolioCycleError(code, f"{label}.relative_path is invalid")
    try:
        path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PortfolioCycleError(code, f"{label}.relative_path must be ASCII") from exc
    return ArtifactRef(
        schema_id=schema_id,
        relative_path=path,
        byte_sha256=require_sha256(document.get("byte_sha256"), label=f"{label}.byte_sha256"),
    )


__all__ = [
    "ArtifactRef",
    "HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID",
    "HOLDINGS_LEDGER_SCHEMA_ID",
    "HOLDINGS_MANIFEST_SCHEMA_ID",
    "HOLDINGS_POINTER_SCHEMA_ID",
    "HOLDINGS_PRICE_SOURCE_SCHEMA_ID",
    "HoldingPosition",
    "IDENTITY_DECLARATION_SCHEMA_ID",
    "MoneyTotals",
    "PROTOCOL",
    "PortfolioCycleError",
    "VerifiedHoldingsBaseline",
    "VerifiedStrategyIdentity",
    "canonical_json_bytes",
    "parse_canonical_json",
    "require_date",
    "require_exact_fields",
    "require_identifier",
    "require_money",
    "require_sha256",
    "require_text",
    "require_timestamp",
    "seal_document",
    "validate_artifact_ref",
]
