"""Exact, fail-closed holdings snapshot contracts for v17 shadow runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timezone
import math
from typing import Any

from .contracts import (
    Availability,
    V17ContractError,
    coerce_enum,
    parse_iso_date,
    parse_utc_timestamp,
    require_authority_false,
    require_bool,
    require_exact_keys,
    require_identifier,
    require_nonempty_string,
    require_number,
    require_symbol,
)
from .semantic import seal_semantic, validate_semantic_seal

HOLDINGS_VERSION = "myquant.v17.holdings-snapshot.v1"
HOLDINGS_IDENTITY_KEYS = frozenset(
    {
        "version",
        "snapshot_id",
        "strategy_id",
        "market",
        "availability",
        "authority",
        "semantic_sha256",
    }
)
HOLDINGS_AVAILABLE_KEYS = HOLDINGS_IDENTITY_KEYS | frozenset(
    {
        "pit_cutoff",
        "as_of",
        "nav",
        "cash",
        "declared_all_cash",
        "positions",
    }
)
HOLDINGS_UNAVAILABLE_KEYS = HOLDINGS_IDENTITY_KEYS | frozenset({"reason"})
POSITION_KEYS = frozenset({"symbol", "market_value"})


def _parse_cutoff(value: Any, *, label: str) -> datetime:
    if isinstance(value, str) and len(value) == 10:
        parsed = parse_iso_date(value, label=label)
        return datetime.combine(parsed, time.min, tzinfo=timezone.utc)
    return parse_utc_timestamp(value, label=label)


def validate_holdings_snapshot(
    payload: Mapping[str, Any],
    *,
    cutoff: str | None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise V17ContractError("holdings snapshot must be an object")
    sealed = validate_semantic_seal(payload)
    if sealed.get("version") != HOLDINGS_VERSION:
        raise V17ContractError("holdings snapshot version mismatch")
    require_identifier(sealed.get("snapshot_id"), label="snapshot_id")
    require_identifier(sealed.get("strategy_id"), label="strategy_id")
    if sealed.get("market") != "CN":
        raise V17ContractError("v17 holdings market must be CN")
    require_authority_false(sealed.get("authority"))
    availability = coerce_enum(sealed.get("availability"), Availability, label="availability")
    if availability is Availability.UNAVAILABLE:
        require_exact_keys(sealed, HOLDINGS_UNAVAILABLE_KEYS, label="UNAVAILABLE holdings")
        require_nonempty_string(sealed.get("reason"), label="reason", max_chars=512)
        return sealed

    require_exact_keys(sealed, HOLDINGS_AVAILABLE_KEYS, label="AVAILABLE holdings")
    pit_cutoff = _parse_cutoff(sealed.get("pit_cutoff"), label="pit_cutoff")
    as_of = parse_utc_timestamp(sealed.get("as_of"), label="as_of")
    if pit_cutoff > as_of:
        raise V17ContractError("holdings PIT cutoff is later than as_of")
    if cutoff is not None:
        validation_time = _parse_cutoff(cutoff, label="cutoff")
        if pit_cutoff > validation_time:
            raise V17ContractError("holdings contains post-cutoff evidence")
        if as_of > validation_time:
            raise V17ContractError("holdings as_of is later than validation cutoff")

    nav = require_number(sealed.get("nav"), label="nav", minimum=0.0, minimum_exclusive=True)
    cash = require_number(sealed.get("cash"), label="cash", minimum=0.0)
    if cash > nav:
        raise V17ContractError("holdings cash cannot exceed NAV")
    all_cash = require_bool(sealed.get("declared_all_cash"), label="declared_all_cash")
    positions_value = sealed.get("positions")
    if isinstance(positions_value, (str, bytes)) or not isinstance(positions_value, Sequence):
        raise V17ContractError("holdings positions must be an array")
    positions: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_market_value = 0.0
    for index, item in enumerate(positions_value):
        if not isinstance(item, Mapping):
            raise V17ContractError(f"positions[{index}] must be an object")
        require_exact_keys(item, POSITION_KEYS, label=f"positions[{index}]")
        symbol = require_symbol(item.get("symbol"), label=f"positions[{index}].symbol")
        if symbol in seen:
            raise V17ContractError(f"duplicate holding symbol: {symbol}")
        seen.add(symbol)
        market_value = require_number(
            item.get("market_value"),
            label=f"positions[{index}].market_value",
            minimum=0.0,
            minimum_exclusive=True,
        )
        total_market_value += market_value
        positions.append({"symbol": symbol, "market_value": market_value})

    tolerance = max(1e-8, nav * 1e-10)
    if all_cash:
        if positions or abs(cash - nav) > tolerance:
            raise V17ContractError(
                "declared all-cash holdings must have no positions and cash equal to NAV"
            )
    else:
        if not positions:
            raise V17ContractError("empty positions require an explicit all-cash declaration")
        if not math.isclose(cash + total_market_value, nav, rel_tol=1e-10, abs_tol=tolerance):
            raise V17ContractError("holdings cash and positions do not reconcile to NAV")
    return sealed


def build_available_holdings_snapshot(
    *,
    snapshot_id: str,
    strategy_id: str,
    market: str,
    pit_cutoff: str,
    as_of: str,
    nav: float,
    cash: float,
    declared_all_cash: bool,
    positions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = seal_semantic(
        {
            "version": HOLDINGS_VERSION,
            "snapshot_id": snapshot_id,
            "strategy_id": strategy_id,
            "market": market,
            "availability": Availability.AVAILABLE.value,
            "pit_cutoff": pit_cutoff,
            "as_of": as_of,
            "nav": nav,
            "cash": cash,
            "declared_all_cash": declared_all_cash,
            "positions": [dict(item) for item in positions],
            "authority": False,
        }
    )
    return validate_holdings_snapshot(payload, cutoff=None)


def build_unavailable_holdings_snapshot(
    *,
    snapshot_id: str,
    strategy_id: str,
    market: str,
    reason: str,
) -> dict[str, Any]:
    payload = seal_semantic(
        {
            "version": HOLDINGS_VERSION,
            "snapshot_id": snapshot_id,
            "strategy_id": strategy_id,
            "market": market,
            "availability": Availability.UNAVAILABLE.value,
            "reason": reason,
            "authority": False,
        }
    )
    return validate_holdings_snapshot(payload, cutoff=None)


@dataclass(frozen=True)
class HoldingsSnapshot:
    _payload: Mapping[str, Any]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        cutoff: str | None,
    ) -> "HoldingsSnapshot":
        return cls(validate_holdings_snapshot(payload, cutoff=cutoff))

    @property
    def availability(self) -> Availability:
        return Availability(self._payload["availability"])

    @property
    def held_symbols(self) -> frozenset[str]:
        if self.availability is Availability.UNAVAILABLE:
            raise V17ContractError(
                "UNAVAILABLE holdings cannot be interpreted as an empty portfolio"
            )
        return frozenset(item["symbol"] for item in self._payload["positions"])

    def to_dict(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(dict(self._payload))


__all__ = [
    "HOLDINGS_AVAILABLE_KEYS",
    "HOLDINGS_IDENTITY_KEYS",
    "HOLDINGS_UNAVAILABLE_KEYS",
    "HOLDINGS_VERSION",
    "HoldingsSnapshot",
    "POSITION_KEYS",
    "build_available_holdings_snapshot",
    "build_unavailable_holdings_snapshot",
    "validate_holdings_snapshot",
]
