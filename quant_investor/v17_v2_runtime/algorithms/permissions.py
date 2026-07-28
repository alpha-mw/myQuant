"""Deterministic shrink-only permission truth table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.v17_v2_contract.identities import require_security_code

from ._semantic import seal_semantic, validate_semantic_seal

VERSION = "myquant.v17.v2.trade-permission.v1"
FUNDAMENTAL_STATES = frozenset({"F_ELIGIBLE", "F_INELIGIBLE", "UNAVAILABLE"})
QUANT_STATES = frozenset({"BUY_NOW", "WATCH", "TRIM_TIMING"})
RESTRICTION_GATE_ORDER = ("risk", "optimizer")


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _base_permission(
    *,
    held: bool,
    tradable: bool,
    fundamental_eligibility: str,
    severe_red_flag: bool,
    quant_timing: str,
) -> tuple[bool, bool, bool, str]:
    if not tradable:
        return False, False, held, "untradable_absolute_block"
    if not held:
        if (
            quant_timing == "BUY_NOW"
            and fundamental_eligibility == "F_ELIGIBLE"
            and not severe_red_flag
        ):
            return True, False, False, "new_position_buy_permitted"
        return False, False, False, "new_position_not_permitted"
    if quant_timing == "TRIM_TIMING":
        return False, True, False, "held_trim_sell_only"
    if quant_timing == "WATCH":
        return False, False, True, "held_watch_locked"
    if (
        quant_timing == "BUY_NOW"
        and fundamental_eligibility == "F_ELIGIBLE"
        and not severe_red_flag
    ):
        return True, False, False, "held_add_permitted"
    return False, False, True, "held_buy_now_fundamental_lock"


def determine_trade_permission(
    *,
    symbol: str,
    held: bool,
    tradable: bool,
    fundamental_eligibility: str,
    severe_red_flag: bool,
    quant_timing: str,
) -> dict[str, Any]:
    symbol = require_security_code(symbol)
    held = _boolean(held, label="held")
    tradable = _boolean(tradable, label="tradable")
    severe_red_flag = _boolean(severe_red_flag, label="severe_red_flag")
    if fundamental_eligibility not in FUNDAMENTAL_STATES:
        raise ValueError("fundamental_eligibility invalid")
    if quant_timing not in QUANT_STATES:
        raise ValueError("quant_timing invalid")
    can_buy, can_sell, locked, basis = _base_permission(
        held=held,
        tradable=tradable,
        fundamental_eligibility=fundamental_eligibility,
        severe_red_flag=severe_red_flag,
        quant_timing=quant_timing,
    )
    return seal_semantic(
        {
            "version": VERSION,
            "symbol": symbol,
            "held": held,
            "tradable": tradable,
            "fundamental_eligibility": fundamental_eligibility,
            "severe_red_flag": severe_red_flag,
            "quant_timing": quant_timing,
            "can_buy": can_buy,
            "can_sell": can_sell,
            "position_locked": locked,
            "basis": basis,
            "restriction_chain": [],
            "authority": False,
        }
    )


def build_permission_restriction(
    *, gate: str, allow_buy: bool, allow_sell: bool, reason: str
) -> dict[str, Any]:
    if gate not in RESTRICTION_GATE_ORDER:
        raise ValueError("permission restriction gate must be risk or optimizer")
    if not isinstance(reason, str) or not reason or reason.strip() != reason:
        raise ValueError(f"{gate}.reason must be canonical and non-empty")
    return {
        "gate": gate,
        "allow_buy": _boolean(allow_buy, label=f"{gate}.allow_buy"),
        "allow_sell": _boolean(allow_sell, label=f"{gate}.allow_sell"),
        "reason": reason,
    }


def _validate_chain(value: object) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("restriction_chain must be an array")
    result: list[dict[str, Any]] = []
    gates: list[str] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "gate",
            "allow_buy",
            "allow_sell",
            "reason",
        }:
            raise ValueError("restriction item invalid")
        restriction = build_permission_restriction(
            gate=item["gate"],
            allow_buy=item["allow_buy"],
            allow_sell=item["allow_sell"],
            reason=item["reason"],
        )
        gates.append(restriction["gate"])
        result.append(restriction)
    if gates != [gate for gate in RESTRICTION_GATE_ORDER if gate in gates]:
        raise ValueError("restriction gates must be unique and ordered risk->optimizer")
    return result


def apply_permission_restrictions(
    permission: Mapping[str, Any],
    *,
    restrictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base = validate_trade_permission(permission)
    if base["restriction_chain"]:
        raise ValueError("restrictions may only be applied once")
    chain = _validate_chain(restrictions)
    can_buy = bool(base["can_buy"])
    can_sell = bool(base["can_sell"])
    for restriction in chain:
        can_buy = can_buy and restriction["allow_buy"]
        can_sell = can_sell and restriction["allow_sell"]
    unsealed = {key: value for key, value in base.items() if key != "semantic_sha256"}
    unsealed.update(
        {
            "can_buy": can_buy,
            "can_sell": can_sell,
            "position_locked": bool(base["held"] and not (can_buy or can_sell)),
            "restriction_chain": chain,
        }
    )
    return seal_semantic(unsealed)


def validate_trade_permission(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = validate_semantic_seal(dict(payload))
    expected_keys = {
        "version",
        "symbol",
        "held",
        "tradable",
        "fundamental_eligibility",
        "severe_red_flag",
        "quant_timing",
        "can_buy",
        "can_sell",
        "position_locked",
        "basis",
        "restriction_chain",
        "authority",
        "semantic_sha256",
    }
    if (
        set(sealed) != expected_keys
        or sealed["version"] != VERSION
        or sealed["authority"] is not False
    ):
        raise ValueError("trade permission contract mismatch")
    require_security_code(sealed["symbol"])
    fundamental = sealed["fundamental_eligibility"]
    timing = sealed["quant_timing"]
    if fundamental not in FUNDAMENTAL_STATES or timing not in QUANT_STATES:
        raise ValueError("trade permission state invalid")
    held = _boolean(sealed["held"], label="held")
    tradable = _boolean(sealed["tradable"], label="tradable")
    red = _boolean(sealed["severe_red_flag"], label="severe_red_flag")
    base_buy, base_sell, base_locked, basis = _base_permission(
        held=held,
        tradable=tradable,
        fundamental_eligibility=fundamental,
        severe_red_flag=red,
        quant_timing=timing,
    )
    chain = _validate_chain(sealed["restriction_chain"])
    expected_buy, expected_sell = base_buy, base_sell
    for restriction in chain:
        expected_buy = expected_buy and restriction["allow_buy"]
        expected_sell = expected_sell and restriction["allow_sell"]
    expected_locked = bool(held and not (expected_buy or expected_sell)) if chain else base_locked
    if (
        sealed["basis"] != basis
        or _boolean(sealed["can_buy"], label="can_buy") is not expected_buy
        or _boolean(sealed["can_sell"], label="can_sell") is not expected_sell
        or _boolean(sealed["position_locked"], label="position_locked") is not expected_locked
    ):
        raise ValueError("trade permission arithmetic mismatch")
    return sealed
