"""Deterministic shrunk-IC and largest-remainder Factor weights."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, ROUND_FLOOR, localcontext
from typing import Final

from .common import SHRINKAGE_PSEUDO_COUNT, decimal_text, decimal_value
from .errors import FactorGovernanceError

WEIGHT_UNITS: Final = 10**12


def shrunk_ic(mean_ic: Decimal, observation_count: int) -> Decimal:
    if type(observation_count) is not int or observation_count < 1:
        raise FactorGovernanceError("observation_count must be positive")
    value = max(Decimal("0"), decimal_value(mean_ic, label="mean_ic"))
    count = Decimal(observation_count)
    return value * count / (count + SHRINKAGE_PSEUDO_COUNT)


def largest_remainder_weights(values: Mapping[str, Decimal]) -> dict[str, str]:
    """Allocate exactly 1e12 units, breaking equal remainders by factor ID."""

    normalized = {
        str(factor_id): decimal_value(value, label=f"shrunk_ic[{factor_id}]", minimum=Decimal("0"))
        for factor_id, value in values.items()
    }
    total = sum(normalized.values(), Decimal("0"))
    if not normalized or total <= 0:
        raise FactorGovernanceError("all shrunk IC values are zero")
    floors: dict[str, int] = {}
    remainders: list[tuple[Decimal, str]] = []
    with localcontext() as context:
        context.prec = 50
        for factor_id in sorted(normalized, key=lambda value: value.encode("utf-8")):
            exact_units = normalized[factor_id] * Decimal(WEIGHT_UNITS) / total
            floor_units = int(exact_units.to_integral_value(rounding=ROUND_FLOOR))
            floors[factor_id] = floor_units
            remainders.append((exact_units - Decimal(floor_units), factor_id))
    residual = WEIGHT_UNITS - sum(floors.values())
    if residual < 0 or residual > len(floors):
        raise FactorGovernanceError("largest-remainder residual is invalid")
    ordered = sorted(remainders, key=lambda row: (-row[0], row[1].encode("utf-8")))
    for _, factor_id in ordered[:residual]:
        floors[factor_id] += 1
    result = {
        factor_id: decimal_text(
            Decimal(units) / Decimal(WEIGHT_UNITS), label=f"weight[{factor_id}]"
        )
        for factor_id, units in floors.items()
    }
    if sum((Decimal(value) for value in result.values()), Decimal("0")) != Decimal("1"):
        raise FactorGovernanceError("factor weights do not sum to one")
    return result


__all__ = ["WEIGHT_UNITS", "largest_remainder_weights", "shrunk_ic"]
