"""Deep-only shrink/veto truth table for v17 v3."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

from .decimal_normalization import DECIMAL_PRECISION, DecimalInput, normalize_decimal

MAX_DEEP_PENALTY = Decimal("0.10")
DEEP_PENALTY_SCALE = Decimal("0.10")


@dataclass(frozen=True)
class DeepResearchDecision:
    status: str
    held: bool
    available: bool
    buy_veto: bool
    locked: bool
    signal: Decimal | None
    penalty: Decimal
    base_target: Decimal
    raw_adjusted_target: Decimal
    target: Decimal
    current_target: Decimal
    blockers: tuple[str, ...] = ()


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def evaluate_deep_research(
    *,
    held: bool,
    current_target: DecimalInput,
    base_target: DecimalInput,
    available: bool,
    signal: DecimalInput | None = None,
    veto_buy: bool = False,
) -> DeepResearchDecision:
    """Apply Deep as a veto/shrink-only layer, never a positive adjustment."""

    held_value = _strict_bool(held, label="held")
    available_value = _strict_bool(available, label="available")
    veto_value = _strict_bool(veto_buy, label="veto_buy")
    current = normalize_decimal(current_target, label="current_target")
    base = normalize_decimal(base_target, label="base_target")
    if current < 0 or base < 0:
        raise ValueError("current_target and base_target must be nonnegative")

    if not available_value:
        target = current if held_value else Decimal("0").quantize(current)
        return DeepResearchDecision(
            status="LOCK" if held_value else "BUY_VETO",
            held=held_value,
            available=False,
            buy_veto=not held_value,
            locked=held_value,
            signal=None,
            penalty=Decimal("0").quantize(current),
            base_target=base,
            raw_adjusted_target=base,
            target=target,
            current_target=current,
            blockers=("deep_research_unavailable",),
        )

    if signal is None:
        raise ValueError("available deep research requires a signal")
    signal_value = normalize_decimal(signal, label="signal")
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        raw_penalty = min(
            MAX_DEEP_PENALTY,
            max(
                Decimal("0"),
                DEEP_PENALTY_SCALE * max(-signal_value, Decimal("0")),
            ),
        )
        penalty = normalize_decimal(raw_penalty, label="penalty")
        raw_adjusted = normalize_decimal(
            base - abs(base) * penalty,
            label="adjusted_target",
        )
    buy_veto = veto_value and not held_value
    if veto_value and held_value:
        target = current
        status = "LOCK"
        locked = True
    elif buy_veto:
        target = Decimal("0").quantize(base)
        status = "BUY_VETO"
        locked = False
    elif held_value:
        target = max(current, raw_adjusted)
        status = "HOLDING_FLOOR" if target != raw_adjusted else "READY"
        locked = False
    else:
        target = raw_adjusted
        status = "READY"
        locked = False
    return DeepResearchDecision(
        status=status,
        held=held_value,
        available=True,
        buy_veto=buy_veto,
        locked=locked,
        signal=signal_value,
        penalty=penalty,
        base_target=base,
        raw_adjusted_target=raw_adjusted,
        target=normalize_decimal(target, label="target"),
        current_target=current,
    )


__all__ = [
    "DeepResearchDecision",
    "evaluate_deep_research",
]
