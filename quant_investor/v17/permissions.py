"""Deterministic v17 trade-permission truth table.

Permissions are advisory shadow outputs only.  Fundamental/Quant establish the
maximum action set; risk and optimizer gates can only intersect (shrink) it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    FundamentalEligibility,
    QuantTiming,
    V17ContractError,
    coerce_enum,
    require_authority_false,
    require_bool,
    require_exact_keys,
    require_nonempty_string,
    require_symbol,
)
from .semantic import seal_semantic, validate_semantic_seal

TRADE_PERMISSION_VERSION = "myquant.v17.trade-permission.v1"
TRADE_PERMISSION_KEYS = frozenset(
    {
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
)
RESTRICTION_KEYS = frozenset({"gate", "allow_buy", "allow_sell", "reason"})
RESTRICTION_GATE_ORDER = ("risk", "optimizer")


def _base_permission(
    *,
    held: bool,
    tradable: bool,
    fundamental_eligibility: FundamentalEligibility,
    severe_red_flag: bool,
    quant_timing: QuantTiming,
) -> tuple[bool, bool, bool, str]:
    if not tradable:
        return False, False, held, "untradable_absolute_block"
    if not held:
        if (
            quant_timing is QuantTiming.BUY_NOW
            and fundamental_eligibility is FundamentalEligibility.F_ELIGIBLE
            and not severe_red_flag
        ):
            return True, False, False, "new_position_buy_permitted"
        return False, False, False, "new_position_not_permitted"
    if quant_timing is QuantTiming.TRIM_TIMING:
        # Fundamental status/red flags never revoke an already-authorized trim,
        # and they never create the trim either.
        return False, True, False, "held_trim_sell_only"
    if quant_timing is QuantTiming.WATCH:
        return False, False, True, "held_watch_locked"
    if fundamental_eligibility is FundamentalEligibility.F_ELIGIBLE and not severe_red_flag:
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
    canonical_symbol = require_symbol(symbol)
    canonical_held = require_bool(held, label="held")
    canonical_tradable = require_bool(tradable, label="tradable")
    canonical_f = coerce_enum(
        fundamental_eligibility,
        FundamentalEligibility,
        label="fundamental_eligibility",
    )
    canonical_red = require_bool(severe_red_flag, label="severe_red_flag")
    canonical_q = coerce_enum(quant_timing, QuantTiming, label="quant_timing")
    can_buy, can_sell, locked, basis = _base_permission(
        held=canonical_held,
        tradable=canonical_tradable,
        fundamental_eligibility=canonical_f,  # type: ignore[arg-type]
        severe_red_flag=canonical_red,
        quant_timing=canonical_q,  # type: ignore[arg-type]
    )
    return seal_semantic(
        {
            "version": TRADE_PERMISSION_VERSION,
            "symbol": canonical_symbol,
            "held": canonical_held,
            "tradable": canonical_tradable,
            "fundamental_eligibility": canonical_f.value,
            "severe_red_flag": canonical_red,
            "quant_timing": canonical_q.value,
            "can_buy": can_buy,
            "can_sell": can_sell,
            "position_locked": locked,
            "basis": basis,
            "restriction_chain": [],
            "authority": False,
        }
    )


def build_permission_restriction(
    *,
    gate: str,
    allow_buy: bool,
    allow_sell: bool,
    reason: str,
) -> dict[str, Any]:
    if gate not in RESTRICTION_GATE_ORDER:
        raise V17ContractError("permission restriction gate must be risk or optimizer")
    return {
        "gate": gate,
        "allow_buy": require_bool(allow_buy, label=f"{gate}.allow_buy"),
        "allow_sell": require_bool(allow_sell, label=f"{gate}.allow_sell"),
        "reason": require_nonempty_string(reason, label=f"{gate}.reason", max_chars=256),
    }


def _validate_restriction_chain(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V17ContractError("restriction_chain must be an array")
    chain: list[dict[str, Any]] = []
    gates: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise V17ContractError(f"restriction_chain[{index}] must be an object")
        require_exact_keys(item, RESTRICTION_KEYS, label=f"restriction_chain[{index}]")
        gate = item.get("gate")
        if gate not in RESTRICTION_GATE_ORDER:
            raise V17ContractError("unknown permission restriction gate")
        gates.append(gate)
        chain.append(
            build_permission_restriction(
                gate=gate,
                allow_buy=item.get("allow_buy"),
                allow_sell=item.get("allow_sell"),
                reason=item.get("reason"),
            )
        )
    expected_order = [gate for gate in RESTRICTION_GATE_ORDER if gate in gates]
    if gates != expected_order:
        raise V17ContractError("restriction gates must be unique and ordered risk->optimizer")
    return chain


def apply_permission_restrictions(
    permission: Mapping[str, Any],
    *,
    restrictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base = validate_trade_permission(permission)
    if base["restriction_chain"]:
        raise V17ContractError("restrictions may only be applied once to base permission")
    chain = _validate_restriction_chain(restrictions)
    can_buy = bool(base["can_buy"])
    can_sell = bool(base["can_sell"])
    for restriction in chain:
        can_buy = can_buy and restriction["allow_buy"]
        can_sell = can_sell and restriction["allow_sell"]
    unsealed = {key: value for key, value in base.items() if key != "semantic_sha256"}
    unsealed["can_buy"] = can_buy
    unsealed["can_sell"] = can_sell
    unsealed["position_locked"] = bool(base["held"] and not (can_buy or can_sell))
    unsealed["restriction_chain"] = chain
    return seal_semantic(unsealed)


def validate_trade_permission(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, TRADE_PERMISSION_KEYS, label="trade permission")
    if sealed.get("version") != TRADE_PERMISSION_VERSION:
        raise V17ContractError("trade permission version mismatch")
    require_authority_false(sealed.get("authority"))
    symbol = require_symbol(sealed.get("symbol"))
    held = require_bool(sealed.get("held"), label="held")
    tradable = require_bool(sealed.get("tradable"), label="tradable")
    fundamental = coerce_enum(
        sealed.get("fundamental_eligibility"),
        FundamentalEligibility,
        label="fundamental_eligibility",
    )
    red = require_bool(sealed.get("severe_red_flag"), label="severe_red_flag")
    quant = coerce_enum(sealed.get("quant_timing"), QuantTiming, label="quant_timing")
    base_buy, base_sell, base_locked, basis = _base_permission(
        held=held,
        tradable=tradable,
        fundamental_eligibility=fundamental,  # type: ignore[arg-type]
        severe_red_flag=red,
        quant_timing=quant,  # type: ignore[arg-type]
    )
    if sealed.get("basis") != basis:
        raise V17ContractError("trade permission basis mismatch")
    chain = _validate_restriction_chain(sealed.get("restriction_chain"))
    expected_buy = base_buy
    expected_sell = base_sell
    for restriction in chain:
        expected_buy = expected_buy and restriction["allow_buy"]
        expected_sell = expected_sell and restriction["allow_sell"]
    expected_locked = bool(held and not (expected_buy or expected_sell))
    if not chain:
        expected_locked = base_locked
    for field, expected in (
        ("can_buy", expected_buy),
        ("can_sell", expected_sell),
        ("position_locked", expected_locked),
    ):
        actual = require_bool(sealed.get(field), label=field)
        if actual is not expected:
            raise V17ContractError(f"trade permission {field} mismatch")
    # Symbol is evaluated above; this assignment documents that canonicalized
    # value must equal the payload rather than silently rewriting it.
    if symbol != sealed["symbol"]:
        raise V17ContractError("trade permission symbol mismatch")
    return sealed


__all__ = [
    "RESTRICTION_GATE_ORDER",
    "TRADE_PERMISSION_VERSION",
    "apply_permission_restrictions",
    "build_permission_restriction",
    "determine_trade_permission",
    "validate_trade_permission",
]
