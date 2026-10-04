"""Sealed-policy sell-signal rules for the Paper account.

This module is the signal layer only: it turns strict-close evidence and the
sealed owner policies into one paper action per position. It performs no I/O,
holds no state, and never writes an order; the writer (`runtime.py`) remains the
only component that can mutate a paper account.

Policy binding:

- `results/policies/paper/.../owner-paper-risk-execution-policy-20261002-v2.json`
  (`signal_policy`) maps owner-stop breaches and profit giveback bands to
  `CLEAR_RISK` / `REDUCE_RISK` fractions.
- `results/policies/risk/.../owner-stop-policy-20260828-v1.json` supplies
  per-symbol hard stops with a strict-close trigger.
- The ledger's trailing columns (`trailing_profit_review_price`,
  `trailing_profit_reduce_price`, `profit_giveback_ratio`) are the output of
  `owner-trailing-anchor-policy-20260901-v1` (0.80 review / 0.65 reduce
  retention) and are consumed as given.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Final, Mapping

from .contracts import PaperError

HOLD: Final = "HOLD"
REVIEW_ONLY: Final = "REVIEW_ONLY"
REDUCE_25: Final = "REDUCE_25"
REDUCE_50: Final = "REDUCE_50"
EXIT_100: Final = "EXIT_100"

GIVEBACK_REDUCE_THRESHOLD: Final = Decimal("0.35")
GIVEBACK_REVIEW_THRESHOLD: Final = Decimal("0.20")
# Owner materiality floor: a giveback band only acts on a position whose peak
# profit was real. Below the floor the lane reports for review instead.
PEAK_PROFIT_TO_COST_FLOOR: Final = Decimal("0.10")

_POSITION_FIELDS: Final = {
    "symbol",
    "shares",
    "settled_shares",
    "avg_cost",
    "close",
    "hard_stop",
    "hard_stop_source",
    "giveback_ratio",
    "peak_price",
    "review_price",
    "reduce_price",
    "deterioration_evidence",
}


def _decimal(value: Any, *, label: str, allow_none: bool = False) -> Decimal | None:
    if value is None:
        if allow_none:
            return None
        raise PaperError("PAPER_RULES_INVALID", f"{label} is missing")
    try:
        parsed = Decimal(str(value))
    except Exception as exc:  # noqa: BLE001 - normalized into a PaperError below
        raise PaperError("PAPER_RULES_INVALID", f"{label} is not a decimal") from exc
    if not parsed.is_finite():
        raise PaperError("PAPER_RULES_INVALID", f"{label} is not finite")
    return parsed


def _symbol(value: Any) -> str:
    if type(value) is not str or len(value) != 9 or value[6] != ".":
        raise PaperError("PAPER_RULES_INVALID", "symbol is invalid")
    return value


def evaluate_position(position: Mapping[str, Any]) -> dict[str, Any]:
    """Return the paper action for one position under the sealed sell policy."""

    if type(position) is not dict or set(position) != _POSITION_FIELDS:
        raise PaperError("PAPER_RULES_INVALID", "position fields differ")
    symbol = _symbol(position["symbol"])
    shares = position["shares"]
    settled = position["settled_shares"]
    if type(shares) is not int or shares < 0 or type(settled) is not int or settled < 0:
        raise PaperError("PAPER_RULES_INVALID", "share counts are invalid")
    if settled > shares:
        raise PaperError("PAPER_RULES_INVALID", "settled shares exceed shares")
    close = _decimal(position["close"], label="close")
    if close <= 0:
        raise PaperError("PAPER_RULES_INVALID", "close is not positive")
    avg_cost = _decimal(position["avg_cost"], label="avg_cost")
    if avg_cost <= 0:
        raise PaperError("PAPER_RULES_INVALID", "avg cost is not positive")
    hard_stop = _decimal(position["hard_stop"], label="hard_stop", allow_none=True)
    giveback = _decimal(position["giveback_ratio"], label="giveback_ratio", allow_none=True)
    if giveback is not None and giveback < 0:
        raise PaperError("PAPER_RULES_INVALID", "giveback ratio is negative")
    evidence = position["deterioration_evidence"]
    if type(evidence) is not list or any(type(item) is not str or not item for item in evidence):
        raise PaperError("PAPER_RULES_INVALID", "deterioration evidence is invalid")

    blocked: list[str] = []
    if hard_stop is not None and close <= hard_stop:
        return _signal(
            symbol=symbol,
            action=EXIT_100,
            policy_row="owner_stop_strict_close_breach",
            reasons=[f"HARD_STOP_BREACH:{hard_stop}"],
            evidence_refs=[str(position["hard_stop_source"])],
        )
    if giveback is not None:
        if not _peak_profit_is_material(position, cost=avg_cost):
            return _signal(
                symbol=symbol,
                action=REVIEW_ONLY,
                policy_row="trailing_peak_profit_below_materiality_floor",
                reasons=[
                    f"GIVEBACK:{giveback}",
                    "TRAILING_PEAK_PROFIT_BELOW_MATERIALITY_FLOOR",
                ],
                blocked=True,
            )
        if giveback >= GIVEBACK_REDUCE_THRESHOLD:
            return _signal(
                symbol=symbol,
                action=REDUCE_50,
                policy_row="profit_giveback_at_least_35_percent",
                reasons=[f"GIVEBACK:{giveback}"],
            )
        if giveback >= GIVEBACK_REVIEW_THRESHOLD:
            if evidence:
                return _signal(
                    symbol=symbol,
                    action=REDUCE_25,
                    policy_row="profit_giveback_20_to_35_percent_with_deterioration",
                    reasons=[f"GIVEBACK:{giveback}", *sorted(evidence)],
                )
            return _signal(
                symbol=symbol,
                action=REVIEW_ONLY,
                policy_row="profit_giveback_20_to_35_percent_without_deterioration",
                reasons=[f"GIVEBACK:{giveback}", "NO_DETERIORATION_EVIDENCE"],
                blocked=True,
            )
    if hard_stop is None and giveback is None:
        blocked.append("NO_HARD_STOP_AND_NO_TRAILING_ANCHOR")
    return _signal(
        symbol=symbol,
        action=HOLD,
        policy_row="no_sell_trigger",
        reasons=sorted(blocked) or ["NO_TRIGGER"],
        blocked=bool(blocked),
    )


def _peak_profit_is_material(position: Mapping[str, Any], *, cost: Decimal) -> bool:
    """Whether the trailing lane may act, per the owner materiality floor.

    A giveback ratio is only meaningful when there was a real peak profit: with a
    peak barely above cost the ratio saturates near 1.0 on noise. Missing peak
    evidence keeps the lane non-actionable (review), never actionable.
    """

    peak = _decimal(position["peak_price"], label="peak_price", allow_none=True)
    if peak is None or cost <= 0:
        return False
    return (peak - cost) / cost >= PEAK_PROFIT_TO_COST_FLOOR


def _signal(
    *,
    symbol: str,
    action: str,
    policy_row: str,
    reasons: list[str],
    blocked: bool = False,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "action": action,
        "policy_row": policy_row,
        "reasons": list(reasons),
        "evidence_refs": list(evidence_refs or []),
        "needs_review": blocked,
    }


def evaluate_portfolio(positions: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate every position; the portfolio view is the ordered signal list."""

    if type(positions) is not list:
        raise PaperError("PAPER_RULES_INVALID", "positions must be a list")
    signals = [evaluate_position(position) for position in positions]
    symbols = [signal["symbol"] for signal in signals]
    if len(set(symbols)) != len(symbols):
        raise PaperError("PAPER_RULES_INVALID", "duplicate symbol in portfolio")
    return {
        "signals": signals,
        "actionable_count": sum(
            1 for signal in signals if signal["action"] in {REDUCE_25, REDUCE_50, EXIT_100}
        ),
        "review_count": sum(1 for signal in signals if signal["needs_review"]),
        "hold_count": sum(1 for signal in signals if signal["action"] == HOLD),
    }


__all__ = [
    "EXIT_100",
    "GIVEBACK_REDUCE_THRESHOLD",
    "GIVEBACK_REVIEW_THRESHOLD",
    "HOLD",
    "REDUCE_25",
    "PEAK_PROFIT_TO_COST_FLOOR",
    "REDUCE_50",
    "REVIEW_ONLY",
    "evaluate_portfolio",
    "evaluate_position",
]
