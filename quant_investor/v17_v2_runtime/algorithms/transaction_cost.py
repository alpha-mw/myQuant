"""Protocol-v2 deterministic square-root market-impact cost model."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any


def _decimal(value: Any, *, label: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    if result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return result


@dataclass(frozen=True)
class TransactionCostEstimate:
    coefficient: Decimal
    notional: Decimal
    adv20: Decimal
    fraction: Decimal
    amount: Decimal
    authority: bool = False

    def to_wire(self) -> dict[str, object]:
        return {
            "version": "myquant.v17.v2.transaction-cost.v1",
            "coefficient": str(self.coefficient),
            "notional": str(self.notional),
            "adv20": str(self.adv20),
            "fraction": str(self.fraction),
            "amount": str(self.amount),
            "authority": False,
        }


def estimate_transaction_cost(
    *,
    coefficient: Decimal | str | int | float,
    notional: Decimal | str | int | float,
    adv20: Decimal | str | int | float,
) -> TransactionCostEstimate:
    """Return ``coefficient * sqrt(notional / ADV20)`` and its cash amount.

    Decimal arithmetic is used end-to-end so replay is independent of binary
    floating-point formatting.  The result is not capped or silently repaired.
    """

    coefficient_d = _decimal(coefficient, label="coefficient")
    notional_d = _decimal(notional, label="notional")
    adv20_d = _decimal(adv20, label="adv20", positive=True)
    with localcontext() as context:
        context.prec = 50
        fraction = coefficient_d * (notional_d / adv20_d).sqrt()
        amount = notional_d * fraction
    return TransactionCostEstimate(
        coefficient=coefficient_d,
        notional=notional_d,
        adv20=adv20_d,
        fraction=fraction,
        amount=amount,
    )
