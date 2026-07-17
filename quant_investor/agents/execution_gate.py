"""V16 order eligibility gate.

Execution feasibility is separate from the research decision.  A blocked BUY
remains a BUY with its original target weight and is marked
``order_eligible=False``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    EligibilityDecision,
    ExecutionDecision,
)
from quant_investor.agents.base import BaseAgent
from quant_investor.v16.candidate_pipeline import Stage2Decision


class ExecutionGate(BaseAgent):
    """Own suspension, quote, lot, cash, and human-authorization checks."""

    agent_name = "ExecutionGate"
    protocol_version = "v1"

    def run(self, payload: Mapping[str, Any]) -> ExecutionDecision:
        envelope = self.ensure_payload(payload)
        self.require_keys(
            envelope,
            "ic_decision",
            "eligibility_decision",
            "execution_context",
        )
        decision = envelope["ic_decision"]
        eligibility = envelope["eligibility_decision"]
        context = self.ensure_payload(envelope["execution_context"])
        if not isinstance(decision, Stage2Decision):
            raise TypeError("ExecutionGate ic_decision must be Stage2Decision")
        if not isinstance(eligibility, EligibilityDecision):
            raise TypeError("ExecutionGate eligibility_decision must be EligibilityDecision")
        if decision.symbol != eligibility.symbol:
            raise ValueError("ExecutionGate decision symbols do not match")
        research_action = ActionLabel(decision.action.lower())

        if research_action is not ActionLabel.BUY:
            return ExecutionDecision(
                symbol=decision.symbol,
                research_action=research_action,
                target_weight=decision.target_weight,
                order_eligible=False,
                raw_order_shares=0.0,
                executable_order_shares=0.0,
                rounding_delta=0.0,
                blockers=["non_buy_research_action"],
                checks={
                    "eligibility_ready": eligibility.research_eligible,
                    "not_suspended": not bool(context.get("halted") or context.get("suspended")),
                    "quote_fresh": self._quote_is_fresh(context),
                    "lot_valid": False,
                    "cash_sufficient": False,
                    "human_authorized": context.get("human_authorized") is True,
                },
            )

        raw_order_shares = self._raw_order_shares(context)
        executable_order_shares = self._executable_order_shares(
            context,
            raw_order_shares,
        )
        checks = {
            "eligibility_ready": eligibility.research_eligible,
            "not_suspended": not bool(context.get("halted") or context.get("suspended")),
            "quote_fresh": self._quote_is_fresh(context),
            "lot_valid": executable_order_shares > 0.0,
            "cash_sufficient": self._cash_is_sufficient(
                context,
                executable_order_shares,
            ),
            "human_authorized": context.get("human_authorized") is True,
        }
        blocker_names = {
            "eligibility_ready": "research_ineligible",
            "not_suspended": "symbol_suspended",
            "quote_fresh": "quote_not_fresh_or_unconfirmed",
            "lot_valid": "lot_not_valid_or_unconfirmed",
            "cash_sufficient": "cash_not_sufficient_or_unconfirmed",
            "human_authorized": "human_authorization_missing",
        }
        blockers = list(eligibility.blockers)
        blockers.extend(blocker_names[name] for name, passed in checks.items() if not passed)
        blockers = list(dict.fromkeys(blockers))
        return ExecutionDecision(
            symbol=decision.symbol,
            research_action=research_action,
            target_weight=decision.target_weight,
            order_eligible=all(checks.values()),
            raw_order_shares=raw_order_shares,
            executable_order_shares=executable_order_shares,
            rounding_delta=raw_order_shares - executable_order_shares,
            blockers=blockers,
            checks=checks,
        )

    @staticmethod
    def _executable_order_shares(
        context: Mapping[str, Any],
        raw_order_shares: float,
    ) -> float:
        lot_size = context.get("lot_size")
        if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0:
            return 0.0
        if raw_order_shares <= 0.0:
            return 0.0
        lots = math.floor(raw_order_shares / lot_size)
        return float(lots * lot_size) if lots >= 1 else 0.0

    @staticmethod
    def _quote_is_fresh(context: Mapping[str, Any]) -> bool:
        quote_price = context.get("quote_price")
        if isinstance(quote_price, bool):
            return False
        try:
            numeric = float(quote_price)
        except (TypeError, ValueError):
            return False
        return bool(context.get("quote_fresh") is True and math.isfinite(numeric) and numeric > 0.0)

    @staticmethod
    def _cash_is_sufficient(
        context: Mapping[str, Any],
        executable_order_shares: float,
    ) -> bool:
        if executable_order_shares <= 0.0:
            return False
        try:
            quote_price = float(context.get("quote_price"))
            available_cash = float(context.get("available_cash"))
        except (TypeError, ValueError):
            return False
        return bool(
            math.isfinite(quote_price)
            and quote_price > 0.0
            and math.isfinite(available_cash)
            and available_cash >= 0.0
            and available_cash + 1e-9 >= executable_order_shares * quote_price
        )

    @staticmethod
    def _raw_order_shares(context: Mapping[str, Any]) -> float:
        try:
            raw_target_shares = float(context.get("raw_target_shares"))
            existing_shares = float(context.get("existing_shares"))
        except (TypeError, ValueError):
            return 0.0
        if (
            not math.isfinite(raw_target_shares)
            or raw_target_shares < 0.0
            or not math.isfinite(existing_shares)
            or existing_shares < 0.0
        ):
            return 0.0
        return max(0.0, raw_target_shares - existing_shares)


__all__ = ["ExecutionGate"]
