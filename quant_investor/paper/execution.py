"""Pure deterministic execution math for sell-only Paper risk exits."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import hashlib
from typing import Any, Mapping

from quant_investor.contracts import canonical_json_bytes

from .contracts import PaperError, seal_document

CENT = Decimal("0.01")
SCALE4 = Decimal("0.0001")
SLIPPAGE = Decimal("0.05")
COMMISSION_RATE = Decimal("0.0001")
COMMISSION_MINIMUM = Decimal("5.00")
TRANSFER_RATE = Decimal("0.00001")
STAMP_RATE = Decimal("0.0005")
EXPIRY_SESSIONS = 3


def _money2(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _money4(value: Decimal) -> str:
    return format(value.quantize(SCALE4, rounding=ROUND_HALF_UP), ".4f")


def _price_tick_down(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_DOWN)


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
    simulated = _price_tick_down(open_price * (Decimal("1") - SLIPPAGE))
    if open_price <= limit_down or simulated < limit_down:
        return pending("PENDING_LIMIT_BLOCKED", ["PAPER_SELL_PRICE_BELOW_LIMIT"])
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
            "adverse_slippage_fraction": "0.0500",
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


__all__ = [
    "calculate_fees",
    "calculate_sell_shares",
    "economic_action_key",
    "execute_sell",
]
