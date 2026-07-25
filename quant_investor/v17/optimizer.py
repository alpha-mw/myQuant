"""Lexicographic selector over externally generated feasible portfolios.

The v17 optimizer is intentionally unable to invent a trade.  It receives a
permission mask from the deterministic Fundamental/Quant/PreTrade chain and a
set of already-feasible portfolio candidates from the risk layer, rejects any
candidate that exceeds that mask, and selects by the frozen objective order.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Mapping
import math

ALLOWED_ACTIONS = frozenset({"BUY", "SELL", "LOCK"})


@dataclass(frozen=True)
class ProposedTrade:
    symbol: str
    action: str
    notional_fraction: float


@dataclass(frozen=True)
class FeasiblePortfolio:
    candidate_id: str
    target_weights: Mapping[str, float]
    trades: tuple[ProposedTrade, ...]
    expected_adjusted_q25: float
    transaction_cost: float
    turnover: float

    @property
    def net_adjusted_q25(self) -> float:
        return self.expected_adjusted_q25 - self.transaction_cost


@dataclass(frozen=True)
class OptimizerResult:
    status: str
    selected: FeasiblePortfolio | None
    rejected: Mapping[str, tuple[str, ...]]


def _validate_candidate(
    candidate: FeasiblePortfolio,
    *,
    permission_mask: Mapping[str, frozenset[str] | set[str] | tuple[str, ...]],
    current_weights: Mapping[str, float],
    effective_gross: float,
) -> tuple[tuple[str, ...], FeasiblePortfolio]:
    blockers: list[str] = []
    if not candidate.candidate_id.strip():
        blockers.append("candidate_id_empty")
    scalars = {
        "expected_adjusted_q25": candidate.expected_adjusted_q25,
        "transaction_cost": candidate.transaction_cost,
        "turnover": candidate.turnover,
    }
    for name, value in scalars.items():
        if not math.isfinite(float(value)):
            blockers.append(f"nonfinite_{name}")
    if math.isfinite(float(candidate.transaction_cost)) and candidate.transaction_cost < 0.0:
        blockers.append("negative_transaction_cost")
    if math.isfinite(float(candidate.turnover)) and candidate.turnover < 0.0:
        blockers.append("negative_turnover")

    weights: dict[str, float] = {}
    for raw_symbol, raw_weight in candidate.target_weights.items():
        symbol = str(raw_symbol).strip()
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            blockers.append(f"invalid_target_weight:{symbol}")
            continue
        if not symbol or not math.isfinite(weight) or weight < 0.0:
            blockers.append(f"invalid_target_weight:{symbol or 'EMPTY'}")
            continue
        if symbol in weights:
            blockers.append(f"duplicate_target_symbol_after_normalization:{symbol}")
            continue
        weights[symbol] = weight
    gross = sum((Decimal(str(value)) for value in weights.values()), Decimal("0"))
    if gross > Decimal(str(effective_gross)):
        blockers.append("effective_gross_exceeded")

    seen: set[str] = set()
    trades_by_symbol: dict[str, ProposedTrade] = {}
    for trade in candidate.trades:
        symbol = str(trade.symbol).strip()
        action = str(trade.action).strip().upper()
        if not symbol or symbol in seen:
            blockers.append(f"duplicate_or_empty_trade:{symbol or 'EMPTY'}")
            continue
        seen.add(symbol)
        try:
            amount = float(trade.notional_fraction)
        except (TypeError, ValueError):
            blockers.append(f"invalid_notional:{symbol}")
            continue
        canonical_trade = ProposedTrade(
            symbol=symbol,
            action=action,
            notional_fraction=amount,
        )
        trades_by_symbol[symbol] = canonical_trade
        if action not in ALLOWED_ACTIONS:
            blockers.append(f"invalid_action:{symbol}:{action}")
            continue
        allowed = frozenset(str(value).strip().upper() for value in permission_mask.get(symbol, ()))
        if action not in allowed:
            blockers.append(f"action_not_permitted:{symbol}:{action}")
        if not math.isfinite(amount) or amount < 0.0:
            blockers.append(f"invalid_notional:{symbol}")
        if action in {"BUY", "SELL"} and amount <= 0.0:
            blockers.append(f"nonpositive_trade_notional:{symbol}")
        if action == "LOCK" and amount != 0.0:
            blockers.append(f"lock_has_notional:{symbol}")

    # Every target change must already be enumerated and permitted.  This is
    # the key boundary that prevents the selector from creating a trade.
    for symbol in sorted(set(current_weights).union(weights)):
        current = Decimal(str(current_weights.get(symbol, 0.0)))
        target = Decimal(str(weights.get(symbol, 0.0)))
        delta = target - current
        proposed = trades_by_symbol.get(symbol)
        if delta == 0:
            if proposed is not None and str(proposed.action).strip().upper() in {"BUY", "SELL"}:
                blockers.append(f"trade_without_target_change:{symbol}")
            continue
        required_action = "BUY" if delta > 0 else "SELL"
        if proposed is None:
            blockers.append(f"target_change_without_trade:{symbol}:{required_action}")
            continue
        if proposed.action.strip().upper() != required_action:
            blockers.append(f"trade_direction_mismatch:{symbol}:{required_action}")
        try:
            proposed_notional = Decimal(str(proposed.notional_fraction))
        except (TypeError, ValueError):
            proposed_notional = Decimal("NaN")
        if not proposed_notional.is_finite() or proposed_notional != abs(delta):
            blockers.append(f"trade_notional_mismatch:{symbol}")
    for symbol in sorted(set(trades_by_symbol).difference(set(current_weights).union(weights))):
        blockers.append(f"trade_symbol_outside_candidate_scope:{symbol}")
    recomputed_turnover = sum(
        abs(Decimal(str(weights.get(symbol, 0.0))) - Decimal(str(current_weights.get(symbol, 0.0))))
        for symbol in set(current_weights).union(weights)
    )
    if Decimal(str(candidate.turnover)) != recomputed_turnover:
        blockers.append("reported_turnover_mismatch")
    canonical_candidate = FeasiblePortfolio(
        candidate_id=candidate.candidate_id.strip(),
        target_weights=dict(sorted(weights.items())),
        trades=tuple(trades_by_symbol[symbol] for symbol in sorted(trades_by_symbol)),
        expected_adjusted_q25=float(candidate.expected_adjusted_q25),
        transaction_cost=float(candidate.transaction_cost),
        turnover=float(candidate.turnover),
    )
    return tuple(dict.fromkeys(blockers)), canonical_candidate


def optimize_lexicographic(
    candidates: Iterable[FeasiblePortfolio],
    *,
    permission_mask: Mapping[str, frozenset[str] | set[str] | tuple[str, ...]],
    current_weights: Mapping[str, float],
    effective_gross: float,
) -> OptimizerResult:
    """Select max net q25, min turnover, then ascending security-code key."""

    if not math.isfinite(float(effective_gross)) or not 0.0 <= effective_gross <= 1.0:
        raise ValueError("effective_gross must be finite and within [0, 1]")
    normalized_mask: dict[str, frozenset[str]] = {}
    for raw_symbol, raw_actions in permission_mask.items():
        symbol = str(raw_symbol).strip()
        actions = frozenset(str(action).strip().upper() for action in raw_actions)
        if not symbol or not actions or not actions.issubset(ALLOWED_ACTIONS):
            raise ValueError(f"invalid permission mask entry: {raw_symbol!r}")
        if symbol in normalized_mask:
            raise ValueError(f"duplicate permission symbol after normalization: {symbol}")
        normalized_mask[symbol] = actions
    normalized_current: dict[str, float] = {}
    for raw_symbol, raw_weight in current_weights.items():
        symbol = str(raw_symbol).strip()
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid current weight: {raw_symbol!r}") from exc
        if not symbol or not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"invalid current weight: {raw_symbol!r}")
        if symbol in normalized_current:
            raise ValueError(f"duplicate current symbol after normalization: {symbol}")
        normalized_current[symbol] = weight
    if sum(
        (Decimal(str(value)) for value in normalized_current.values()),
        Decimal("0"),
    ) > Decimal("1"):
        raise ValueError("current weights exceed one")

    feasible: list[FeasiblePortfolio] = []
    rejected: dict[str, tuple[str, ...]] = {}
    seen_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate.candidate_id.strip()
        if candidate_id in seen_ids:
            rejected[candidate.candidate_id] = ("duplicate_candidate_id",)
            continue
        seen_ids.add(candidate_id)
        blockers, canonical_candidate = _validate_candidate(
            candidate,
            permission_mask=normalized_mask,
            current_weights=normalized_current,
            effective_gross=effective_gross,
        )
        if blockers:
            rejected[candidate.candidate_id] = blockers
        else:
            feasible.append(canonical_candidate)
    if not feasible:
        return OptimizerResult(
            status="SHADOW_PORTFOLIO_INFEASIBLE",
            selected=None,
            rejected=rejected,
        )

    def objective(
        candidate: FeasiblePortfolio,
    ) -> tuple[Decimal, Decimal, tuple[str, ...], str]:
        securities = tuple(
            sorted(
                symbol for symbol, weight in candidate.target_weights.items() if float(weight) > 0.0
            )
        )
        return (
            -(
                Decimal(str(candidate.expected_adjusted_q25))
                - Decimal(str(candidate.transaction_cost))
            ),
            Decimal(str(candidate.turnover)),
            securities,
            candidate.candidate_id,
        )

    selected = sorted(feasible, key=objective)[0]
    return OptimizerResult(
        status="PORTFOLIO_SELECTED",
        selected=selected,
        rejected=rejected,
    )


__all__ = [
    "ALLOWED_ACTIONS",
    "FeasiblePortfolio",
    "OptimizerResult",
    "ProposedTrade",
    "optimize_lexicographic",
]
