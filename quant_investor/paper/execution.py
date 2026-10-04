"""Pure deterministic execution math for the Paper account.

Sell side: risk-reducing exits. Buy side: policy-bounded entries (`entry_policy`
in `owner-paper-risk-execution-policy-20261004-v3`). Both sides fill at the next
session's raw open with adverse slippage and never touch a broker."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
import hashlib
from typing import Any, Mapping

from quant_investor.contracts import canonical_json_bytes

from .contracts import PaperError, seal_document

CENT = Decimal("0.01")
SCALE4 = Decimal("0.0001")
SLIPPAGE = Decimal("0.005")
COMMISSION_RATE = Decimal("0.0001")
COMMISSION_MINIMUM = Decimal("5.00")
TRANSFER_RATE = Decimal("0.00001")
STAMP_RATE = Decimal("0.0005")
EXPIRY_SESSIONS = 3
BUY_LOT = 100


def _money2(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _money4(value: Decimal) -> str:
    return format(value.quantize(SCALE4, rounding=ROUND_HALF_UP), ".4f")


def _price_tick_down(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_DOWN)


def _price_tick_up(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_UP)


def calculate_sell_shares(*, action: str, settled_shares: int) -> int:
    if type(settled_shares) is not int or settled_shares < 0:
        raise PaperError("PAPER_T1_INVALID", "settled shares invalid")
    if action == "EXIT_100":
        return settled_shares
    ratios = {"REDUCE_25": Decimal("0.25"), "REDUCE_50": Decimal("0.50")}
    if action not in ratios:
        raise PaperError("PAPER_ACTION_FORBIDDEN", "sell action invalid")
    raw = int(Decimal(settled_shares) * ratios[action])
    return raw // 100 * 100


def calculate_fees(gross: Decimal) -> dict[str, Decimal]:
    if not gross.is_finite() or gross <= 0:
        raise PaperError("PAPER_ACCOUNTING_INVALID", "gross proceeds invalid")
    commission = _money2(max(gross * COMMISSION_RATE, COMMISSION_MINIMUM))
    transfer = _money2(gross * TRANSFER_RATE)
    stamp = _money2(gross * STAMP_RATE)
    total = commission + transfer + stamp
    if gross - total <= 0:
        raise PaperError("PAPER_NET_CASH_NONPOSITIVE", "fees consume gross proceeds")
    return {
        "commission": commission,
        "transfer_fee": transfer,
        "stamp_duty": stamp,
        "total_fees": total,
        "net_cash_proceeds": gross - total,
    }


def calculate_buy_fees(gross: Decimal) -> dict[str, Decimal]:
    """Buy-side fees: commission and transfer only; stamp duty is sell-only."""

    if not gross.is_finite() or gross <= 0:
        raise PaperError("PAPER_ACCOUNTING_INVALID", "gross cost invalid")
    commission = _money2(max(gross * COMMISSION_RATE, COMMISSION_MINIMUM))
    transfer = _money2(gross * TRANSFER_RATE)
    total = commission + transfer
    return {
        "commission": commission,
        "transfer_fee": transfer,
        "stamp_duty": Decimal("0.00"),
        "total_fees": total,
        "total_cost": gross + total,
    }


def calculate_buy_shares(*, budget: Decimal, price: Decimal) -> int:
    """Largest whole 100-share lot whose fee-inclusive cost fits the budget."""

    if not budget.is_finite() or budget < 0:
        raise PaperError("PAPER_ACCOUNTING_INVALID", "entry budget invalid")
    if not price.is_finite() or price <= 0:
        raise PaperError("PAPER_ACCOUNTING_INVALID", "entry price invalid")
    shares = int(budget / price) // BUY_LOT * BUY_LOT
    while shares > 0:
        gross = _money2(price * Decimal(shares))
        if calculate_buy_fees(gross)["total_cost"] <= budget:
            return shares
        shares -= BUY_LOT
    return 0


def economic_action_key(
    *,
    account_id: str,
    policy_id: str,
    signal_date: str,
    symbol: str,
    action: str,
    shares: int,
) -> str:
    text = "|".join((account_id, policy_id, signal_date, symbol, action, str(shares)))
    return hashlib.sha256(text.encode("ascii", errors="strict")).hexdigest()


def execute_sell(
    *,
    intent: Mapping[str, Any],
    intent_ref: Mapping[str, str],
    eligibility: Mapping[str, Any],
    eligibility_ref: Mapping[str, str],
    position: Mapping[str, Any],
    cash_before: Decimal,
    evaluated_open_session_count: int,
) -> dict[str, Any]:
    """Return one fill or pending outcome without writing any state."""

    if (
        intent["account_id"] != eligibility["account_id"]
        or intent["symbol"] != eligibility["symbol"]
        or intent["signal_date"] != eligibility["signal_date"]
    ):
        raise PaperError("PAPER_EVIDENCE_SESSION_CONFLICT", "intent/eligibility identity differs")
    if eligibility["source_intent_ref"] != dict(intent_ref):
        raise PaperError("PAPER_INPUT_SHA_DRIFT", "eligibility intent ref differs")
    if position["symbol"] != intent["symbol"]:
        raise PaperError("PAPER_POSITION_MISMATCH", "position symbol differs")
    if (
        position["shares"] != intent["expected_position"]["shares"]
        or position["settled_shares"] != intent["expected_position"]["settled_shares"]
        or Decimal(str(position["avg_cost"]))
        != Decimal(str(intent["expected_position"]["avg_cost"]))
    ):
        raise PaperError("PAPER_POSITION_MISMATCH", "expected position differs")

    pending_base = {
        "schema_version": "paper-pending.v1",
        "pending_id": "paper-pending-" + intent["source_intent_id"],
        "source_intent_ref": dict(intent_ref),
        "account_id": intent["account_id"],
        "symbol": intent["symbol"],
        "first_eligible_trade_date": intent["eligible_from_trade_date"],
        "last_evaluated_trade_date": eligibility["evaluated_trade_date"],
        "evaluated_open_session_count": evaluated_open_session_count,
        "expiry_sessions": EXPIRY_SESSIONS,
    }

    def pending(status: str, blockers: list[str]) -> dict[str, Any]:
        terminal = evaluated_open_session_count >= EXPIRY_SESSIONS
        final_status = "EXPIRED_REEVALUATION_REQUIRED" if terminal else status
        final_blockers = ["PAPER_INTENT_EXPIRED"] if terminal else blockers
        return {
            "outcome": "EXPIRED" if terminal else "PENDING",
            "pending": seal_document(
                {**pending_base, "status": final_status, "blocker_codes": sorted(final_blockers)}
            ),
            "order": None,
            "fill": None,
            "accounting": None,
        }

    if eligibility["evidence_status"] in {"NOT_YET_AVAILABLE", "MISSING"}:
        return pending("PENDING_NEXT_SESSION", ["PAPER_EXECUTION_EVIDENCE_NOT_AVAILABLE"])
    if eligibility["evidence_status"] != "READY":
        raise PaperError("PAPER_ELIGIBILITY_INVALID", "unexpected evidence status")
    if eligibility["evaluated_trade_date"] < intent["eligible_from_trade_date"]:
        return pending("PENDING_NEXT_SESSION", ["PAPER_NEXT_SESSION_NOT_REACHED"])
    if eligibility["suspended"] is True:
        return pending("PENDING_SUSPENDED", ["PAPER_SYMBOL_SUSPENDED"])
    if eligibility["corporate_action_state"] != "CLEAR":
        return pending("PENDING_CORPORATE_ACTION", ["PAPER_CORPORATE_ACTION_PENDING"])

    settled = int(position["settled_shares"])
    shares = calculate_sell_shares(action=intent["action"], settled_shares=settled)
    if shares == 0:
        return pending("NO_ACTION_BELOW_MINIMUM_LOT", ["PAPER_BELOW_MINIMUM_LOT"])
    if shares > settled or shares > int(position["shares"]):
        return pending("PENDING_T1", ["PAPER_SETTLED_SHARES_INSUFFICIENT"])

    open_price = Decimal(eligibility["open_price"])
    limit_down = Decimal(eligibility["limit_down"])
    limit_up = Decimal(eligibility["limit_up"])
    if open_price <= 0 or limit_down <= 0 or limit_up < limit_down:
        raise PaperError("PAPER_PRICE_LIMIT_EVIDENCE_INVALID", "price range invalid")
    if open_price <= limit_down:
        return pending("PENDING_LIMIT_BLOCKED", ["PAPER_SELL_PRICE_BELOW_LIMIT"])
    # A gap-down open above the limit still trades, floored at the limit-down price.
    simulated = max(_price_tick_down(open_price * (Decimal("1") - SLIPPAGE)), limit_down)
    if simulated > limit_up:
        raise PaperError("PAPER_PRICE_LIMIT_EVIDENCE_INVALID", "sell price above limit")

    gross = _money2(simulated * Decimal(shares))
    fees = calculate_fees(gross)
    avg_cost = Decimal(position["avg_cost"])
    realized_delta = _money2(gross - fees["total_fees"] - avg_cost * Decimal(shares))
    shares_after = int(position["shares"]) - shares
    cash_after = _money2(cash_before + fees["net_cash_proceeds"])
    cost_basis_after = _money2(avg_cost * Decimal(shares_after))
    if shares_after < 0 or cash_after < cash_before:
        raise PaperError("PAPER_ACCOUNTING_INVALID", "sell accounting invariant failed")

    order_id = "paper-order-" + intent["source_intent_id"]
    order = seal_document(
        {
            "schema_version": "paper-order.v1",
            "order_id": order_id,
            "account_id": intent["account_id"],
            "source_intent_ref": dict(intent_ref),
            "policy_ref": dict(intent["policy_ref"]),
            "symbol": intent["symbol"],
            "side": "SELL",
            "action": intent["action"],
            "shares": shares,
            "trade_date": eligibility["evaluated_trade_date"],
            "price_type": "NEXT_VALID_TRADING_DAY_OPEN",
            "reference_open": _money4(open_price),
            "adverse_slippage_fraction": format(SLIPPAGE, ".4f"),
            "simulated_price": _money4(simulated),
            "status": "FILLED",
            "broker": False,
            "real_order": False,
        }
    )
    order_ref = {
        "path": "orders.v1.json",
        "sha256": hashlib.sha256(canonical_json_bytes(order)).hexdigest(),
    }
    fill = seal_document(
        {
            "schema_version": "paper-fill.v1",
            "fill_id": "paper-fill-" + intent["source_intent_id"],
            "order_ref": order_ref,
            "account_id": intent["account_id"],
            "symbol": intent["symbol"],
            "side": "SELL",
            "shares": shares,
            "trade_date": eligibility["evaluated_trade_date"],
            "simulated_price": _money4(simulated),
            "gross_proceeds": _money4(gross),
            "commission": _money4(fees["commission"]),
            "transfer_fee": _money4(fees["transfer_fee"]),
            "stamp_duty": _money4(fees["stamp_duty"]),
            "total_fees": _money4(fees["total_fees"]),
            "net_cash_proceeds": _money4(fees["net_cash_proceeds"]),
            "realized_pnl_delta": _money4(realized_delta),
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
    )
    return {
        "outcome": "FILLED",
        "pending": None,
        "order": order,
        "fill": fill,
        "accounting": {
            "shares_sold": shares,
            "shares_after": shares_after,
            "cash_after": _money4(cash_after),
            "cost_basis_after": _money4(cost_basis_after),
            "realized_pnl_delta": _money4(realized_delta),
            "cumulative_fees_delta": _money4(fees["total_fees"]),
        },
        "eligibility_ref": dict(eligibility_ref),
    }


def execute_buy(
    *,
    intent: Mapping[str, Any],
    intent_ref: Mapping[str, str],
    eligibility: Mapping[str, Any],
    eligibility_ref: Mapping[str, str],
    position: Mapping[str, Any] | None,
    cash_before: Decimal,
    account_nav: Decimal,
    evaluated_open_session_count: int,
) -> dict[str, Any]:
    """Return one entry fill or pending outcome without writing any state."""

    if (
        intent["account_id"] != eligibility["account_id"]
        or intent["symbol"] != eligibility["symbol"]
        or intent["signal_date"] != eligibility["signal_date"]
    ):
        raise PaperError("PAPER_EVIDENCE_SESSION_CONFLICT", "intent/eligibility identity differs")
    if eligibility["source_intent_ref"] != dict(intent_ref):
        raise PaperError("PAPER_INPUT_SHA_DRIFT", "eligibility intent ref differs")
    expected = intent["expected_position"]
    if expected is None:
        if position is not None:
            raise PaperError("PAPER_POSITION_MISMATCH", "entry expected no position")
    elif (
        position is None
        or position["symbol"] != intent["symbol"]
        or position["shares"] != expected["shares"]
        or position["settled_shares"] != expected["settled_shares"]
        or Decimal(str(position["avg_cost"])) != Decimal(str(expected["avg_cost"]))
    ):
        raise PaperError("PAPER_POSITION_MISMATCH", "expected position differs")

    pending_base = {
        "schema_version": "paper-pending.v1",
        "pending_id": "paper-pending-" + intent["source_intent_id"],
        "source_intent_ref": dict(intent_ref),
        "account_id": intent["account_id"],
        "symbol": intent["symbol"],
        "first_eligible_trade_date": intent["eligible_from_trade_date"],
        "last_evaluated_trade_date": eligibility["evaluated_trade_date"],
        "evaluated_open_session_count": evaluated_open_session_count,
        "expiry_sessions": EXPIRY_SESSIONS,
    }

    def pending(status: str, blockers: list[str]) -> dict[str, Any]:
        terminal = evaluated_open_session_count >= EXPIRY_SESSIONS
        return {
            "outcome": "EXPIRED" if terminal else "PENDING",
            "pending": seal_document(
                {
                    **pending_base,
                    "status": "EXPIRED_REEVALUATION_REQUIRED" if terminal else status,
                    "blocker_codes": sorted(["PAPER_INTENT_EXPIRED"] if terminal else blockers),
                }
            ),
            "order": None,
            "fill": None,
            "accounting": None,
        }

    def skipped(code: str) -> dict[str, Any]:
        """Terminal non-fill: the candidate is dropped, never partially filled."""

        return {
            "outcome": "SKIPPED",
            "pending": seal_document({**pending_base, "status": code, "blocker_codes": [code]}),
            "order": None,
            "fill": None,
            "accounting": None,
        }

    if eligibility["evidence_status"] in {"NOT_YET_AVAILABLE", "MISSING"}:
        return pending("PENDING_NEXT_SESSION", ["PAPER_EXECUTION_EVIDENCE_NOT_AVAILABLE"])
    if eligibility["evidence_status"] != "READY":
        raise PaperError("PAPER_ELIGIBILITY_INVALID", "unexpected evidence status")
    if eligibility["evaluated_trade_date"] < intent["eligible_from_trade_date"]:
        return pending("PENDING_NEXT_SESSION", ["PAPER_NEXT_SESSION_NOT_REACHED"])
    if eligibility["suspended"] is True:
        return pending("PENDING_SUSPENDED", ["PAPER_SYMBOL_SUSPENDED"])
    if eligibility["corporate_action_state"] != "CLEAR":
        return pending("PENDING_CORPORATE_ACTION", ["PAPER_CORPORATE_ACTION_PENDING"])
    if not cash_before.is_finite() or cash_before < 0 or not account_nav.is_finite():
        raise PaperError("PAPER_ACCOUNTING_INVALID", "cash or NAV invalid")
    if account_nav <= 0:
        raise PaperError("PAPER_ACCOUNTING_INVALID", "NAV is not positive")

    open_price = Decimal(eligibility["open_price"])
    limit_down = Decimal(eligibility["limit_down"])
    limit_up = Decimal(eligibility["limit_up"])
    if open_price <= 0 or limit_down <= 0 or limit_up < limit_down:
        raise PaperError("PAPER_PRICE_LIMIT_EVIDENCE_INVALID", "price range invalid")
    if open_price >= limit_up:
        return pending("PENDING_LIMIT_BLOCKED", ["PAPER_BUY_PRICE_AT_LIMIT_UP"])
    simulated = min(_price_tick_up(open_price * (Decimal("1") + SLIPPAGE)), limit_up)
    if simulated < limit_down:
        raise PaperError("PAPER_PRICE_LIMIT_EVIDENCE_INVALID", "buy price below limit")

    weight = Decimal(intent["target_weight"])
    minimum_cash = (account_nav * Decimal(intent["minimum_cash_fraction"])).quantize(
        CENT, rounding=ROUND_HALF_UP
    )
    held = Decimal(int(position["shares"])) if position is not None else Decimal("0")
    room = (account_nav * weight).quantize(CENT, rounding=ROUND_HALF_UP) - held * simulated
    budget = min(room, cash_before - minimum_cash)
    if budget <= 0:
        return skipped("PAPER_ENTRY_SKIPPED_NO_BUDGET")
    shares = calculate_buy_shares(budget=budget, price=simulated)
    if shares == 0:
        return skipped("PAPER_ENTRY_SKIPPED_BELOW_MINIMUM_LOT")

    gross = _money2(simulated * Decimal(shares))
    fees = calculate_buy_fees(gross)
    cash_after = _money2(cash_before - fees["total_cost"])
    shares_after = int(held) + shares
    avg_cost_before = Decimal(str(position["avg_cost"])) if position is not None else Decimal("0")
    cost_basis_before = (
        Decimal(str(position["cost_basis"])) if position is not None else Decimal("0")
    )
    if position is not None and cost_basis_before <= 0:
        cost_basis_before = avg_cost_before * held
    cost_basis_after = _money2(cost_basis_before + fees["total_cost"])
    avg_cost_after = (cost_basis_after / Decimal(shares_after)).quantize(
        SCALE4, rounding=ROUND_HALF_UP
    )
    if cash_after < 0 or cash_after < minimum_cash - CENT:
        raise PaperError("PAPER_ACCOUNTING_INVALID", "entry breaches the cash floor")

    order = seal_document(
        {
            "schema_version": "paper-order.v1",
            "order_id": "paper-order-" + intent["source_intent_id"],
            "account_id": intent["account_id"],
            "source_intent_ref": dict(intent_ref),
            "policy_ref": dict(intent["policy_ref"]),
            "symbol": intent["symbol"],
            "side": "BUY",
            "action": "ENTRY",
            "shares": shares,
            "trade_date": eligibility["evaluated_trade_date"],
            "price_type": "NEXT_VALID_TRADING_DAY_OPEN",
            "reference_open": _money4(open_price),
            "adverse_slippage_fraction": format(SLIPPAGE, ".4f"),
            "simulated_price": _money4(simulated),
            "budget_cny": _money4(budget),
            "status": "FILLED",
            "broker": False,
            "real_order": False,
        }
    )
    order_ref = {
        "path": "orders.v1.json",
        "sha256": hashlib.sha256(canonical_json_bytes(order)).hexdigest(),
    }
    fill = seal_document(
        {
            "schema_version": "paper-fill.v1",
            "fill_id": "paper-fill-" + intent["source_intent_id"],
            "order_ref": order_ref,
            "account_id": intent["account_id"],
            "symbol": intent["symbol"],
            "side": "BUY",
            "shares": shares,
            "trade_date": eligibility["evaluated_trade_date"],
            "simulated_price": _money4(simulated),
            "gross_cost": _money4(gross),
            "commission": _money4(fees["commission"]),
            "transfer_fee": _money4(fees["transfer_fee"]),
            "stamp_duty": _money4(fees["stamp_duty"]),
            "total_fees": _money4(fees["total_fees"]),
            "total_cost": _money4(fees["total_cost"]),
            "realized_pnl_delta": "0.0000",
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
    )
    return {
        "outcome": "FILLED",
        "pending": None,
        "order": order,
        "fill": fill,
        "accounting": {
            "shares_bought": shares,
            "shares_after": shares_after,
            "cash_after": _money4(cash_after),
            "cost_basis_after": _money4(cost_basis_after),
            "avg_cost_after": _money4(avg_cost_after),
            "realized_pnl_delta": "0.0000",
            "cumulative_fees_delta": _money4(fees["total_fees"]),
        },
        "eligibility_ref": dict(eligibility_ref),
    }


__all__ = [
    "calculate_buy_fees",
    "calculate_buy_shares",
    "calculate_fees",
    "calculate_sell_shares",
    "economic_action_key",
    "execute_buy",
    "execute_sell",
]
