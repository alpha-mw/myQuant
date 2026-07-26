"""Decimal-exact lexicographic selector over supplied portfolio proposals."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping

ALLOWED_ACTIONS = frozenset({"BUY", "SELL", "LOCK"})


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class ProposedTrade:
    symbol: str
    action: str
    notional_fraction: Decimal | str | int | float

    def to_wire(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "notional_fraction": str(_decimal(self.notional_fraction, label="notional_fraction")),
        }


@dataclass(frozen=True)
class FeasiblePortfolio:
    candidate_id: str
    target_weights: Mapping[str, Decimal | str | int | float]
    trades: tuple[ProposedTrade, ...]
    expected_adjusted_q25: Decimal | str | int | float
    transaction_cost: Decimal | str | int | float
    turnover: Decimal | str | int | float

    @property
    def net_adjusted_q25(self) -> Decimal:
        return _decimal(self.expected_adjusted_q25, label="expected_adjusted_q25") - _decimal(
            self.transaction_cost, label="transaction_cost"
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "target_weights": {
                symbol: str(_decimal(weight, label=f"target_weights.{symbol}"))
                for symbol, weight in sorted(self.target_weights.items())
            },
            "trades": [trade.to_wire() for trade in self.trades],
            "expected_adjusted_q25": str(
                _decimal(self.expected_adjusted_q25, label="expected_adjusted_q25")
            ),
            "transaction_cost": str(_decimal(self.transaction_cost, label="transaction_cost")),
            "net_adjusted_q25": str(self.net_adjusted_q25),
            "turnover": str(_decimal(self.turnover, label="turnover")),
        }


@dataclass(frozen=True)
class OptimizerResult:
    status: str
    selected: FeasiblePortfolio | None
    rejected: Mapping[str, tuple[str, ...]]


def _normalize_symbol(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _validate_candidate(
    candidate: FeasiblePortfolio,
    *,
    permission_mask: Mapping[str, frozenset[str]],
    current_weights: Mapping[str, Decimal],
    effective_gross: Decimal,
) -> tuple[tuple[str, ...], FeasiblePortfolio]:
    blockers: list[str] = []
    try:
        candidate_id = _normalize_symbol(candidate.candidate_id, label="candidate_id")
    except ValueError:
        candidate_id = str(candidate.candidate_id).strip()
        blockers.append("candidate_id_empty")

    try:
        expected_q25 = _decimal(candidate.expected_adjusted_q25, label="expected_adjusted_q25")
    except ValueError:
        expected_q25 = Decimal("NaN")
        blockers.append("nonfinite_expected_adjusted_q25")
    try:
        cost = _decimal(candidate.transaction_cost, label="transaction_cost")
        if cost < 0:
            blockers.append("negative_transaction_cost")
    except ValueError:
        cost = Decimal("NaN")
        blockers.append("nonfinite_transaction_cost")
    try:
        reported_turnover = _decimal(candidate.turnover, label="turnover")
        if reported_turnover < 0:
            blockers.append("negative_turnover")
    except ValueError:
        reported_turnover = Decimal("NaN")
        blockers.append("nonfinite_turnover")

    weights: dict[str, Decimal] = {}
    for raw_symbol, raw_weight in candidate.target_weights.items():
        try:
            symbol = _normalize_symbol(raw_symbol, label="target symbol")
            weight = _decimal(raw_weight, label=f"target weight {symbol}")
        except ValueError:
            blockers.append(f"invalid_target_weight:{str(raw_symbol).strip() or 'EMPTY'}")
            continue
        if weight < 0:
            blockers.append(f"invalid_target_weight:{symbol}")
            continue
        if symbol in weights:
            blockers.append(f"duplicate_target_symbol_after_normalization:{symbol}")
            continue
        weights[symbol] = weight
    if sum(weights.values(), Decimal("0")) > effective_gross:
        blockers.append("effective_gross_exceeded")

    trades: dict[str, ProposedTrade] = {}
    trade_amounts: dict[str, Decimal] = {}
    for trade in candidate.trades:
        try:
            symbol = _normalize_symbol(trade.symbol, label="trade symbol")
        except ValueError:
            blockers.append("duplicate_or_empty_trade:EMPTY")
            continue
        if symbol in trades:
            blockers.append(f"duplicate_or_empty_trade:{symbol}")
            continue
        action = str(trade.action).strip().upper()
        try:
            amount = _decimal(trade.notional_fraction, label=f"trade notional {symbol}")
        except ValueError:
            amount = Decimal("NaN")
            blockers.append(f"invalid_notional:{symbol}")
        canonical = ProposedTrade(symbol=symbol, action=action, notional_fraction=amount)
        trades[symbol] = canonical
        trade_amounts[symbol] = amount
        if action not in ALLOWED_ACTIONS:
            blockers.append(f"invalid_action:{symbol}:{action}")
        elif action not in permission_mask.get(symbol, frozenset()):
            blockers.append(f"action_not_permitted:{symbol}:{action}")
        if not amount.is_finite() or amount < 0:
            blockers.append(f"invalid_notional:{symbol}")
        elif action in {"BUY", "SELL"} and amount <= 0:
            blockers.append(f"nonpositive_trade_notional:{symbol}")
        elif action == "LOCK" and amount != 0:
            blockers.append(f"lock_has_notional:{symbol}")

    scope = set(current_weights).union(weights)
    for symbol in sorted(scope):
        delta = weights.get(symbol, Decimal("0")) - current_weights.get(symbol, Decimal("0"))
        proposed = trades.get(symbol)
        if delta == 0:
            if proposed is not None and proposed.action in {"BUY", "SELL"}:
                blockers.append(f"trade_without_target_change:{symbol}")
            continue
        action = "BUY" if delta > 0 else "SELL"
        if proposed is None:
            blockers.append(f"target_change_without_trade:{symbol}:{action}")
            continue
        if proposed.action != action:
            blockers.append(f"trade_direction_mismatch:{symbol}:{action}")
        if trade_amounts[symbol] != abs(delta):
            blockers.append(f"trade_notional_mismatch:{symbol}")
    for symbol in sorted(set(trades).difference(scope)):
        blockers.append(f"trade_symbol_outside_candidate_scope:{symbol}")
    turnover = sum(
        (
            abs(weights.get(symbol, Decimal("0")) - current_weights.get(symbol, Decimal("0")))
            for symbol in scope
        ),
        Decimal("0"),
    )
    if reported_turnover != turnover:
        blockers.append("reported_turnover_mismatch")
    canonical_candidate = FeasiblePortfolio(
        candidate_id=candidate_id,
        target_weights=dict(sorted(weights.items())),
        trades=tuple(trades[symbol] for symbol in sorted(trades)),
        expected_adjusted_q25=expected_q25,
        transaction_cost=cost,
        turnover=reported_turnover,
    )
    return tuple(dict.fromkeys(blockers)), canonical_candidate


def optimize_lexicographic(
    candidates: Iterable[FeasiblePortfolio],
    *,
    permission_mask: Mapping[str, frozenset[str] | set[str] | tuple[str, ...]],
    current_weights: Mapping[str, Decimal | str | int | float],
    effective_gross: Decimal | str | int | float,
) -> OptimizerResult:
    """Maximize net q25, minimize turnover, then use the stable code/id key."""

    gross_cap = _decimal(effective_gross, label="effective_gross")
    if not Decimal("0") <= gross_cap <= Decimal("1"):
        raise ValueError("effective_gross must be within [0, 1]")
    normalized_mask: dict[str, frozenset[str]] = {}
    for raw_symbol, raw_actions in permission_mask.items():
        symbol = _normalize_symbol(raw_symbol, label="permission symbol")
        actions = frozenset(str(value).strip().upper() for value in raw_actions)
        if not actions or not actions.issubset(ALLOWED_ACTIONS):
            raise ValueError(f"invalid permission mask entry: {raw_symbol!r}")
        if symbol in normalized_mask:
            raise ValueError(f"duplicate permission symbol: {symbol}")
        normalized_mask[symbol] = actions
    current: dict[str, Decimal] = {}
    for raw_symbol, raw_weight in current_weights.items():
        symbol = _normalize_symbol(raw_symbol, label="current symbol")
        weight = _decimal(raw_weight, label=f"current weight {symbol}")
        if weight < 0:
            raise ValueError(f"invalid current weight: {symbol}")
        if symbol in current:
            raise ValueError(f"duplicate current symbol: {symbol}")
        current[symbol] = weight
    if sum(current.values(), Decimal("0")) > Decimal("1"):
        raise ValueError("current weights exceed one")

    feasible: list[FeasiblePortfolio] = []
    rejected: dict[str, tuple[str, ...]] = {}
    seen_ids: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate.candidate_id).strip()
        if candidate_key in seen_ids:
            rejected[str(candidate.candidate_id)] = ("duplicate_candidate_id",)
            continue
        seen_ids.add(candidate_key)
        blockers, normalized = _validate_candidate(
            candidate,
            permission_mask=normalized_mask,
            current_weights=current,
            effective_gross=gross_cap,
        )
        if blockers:
            rejected[str(candidate.candidate_id)] = blockers
        else:
            feasible.append(normalized)
    if not feasible:
        return OptimizerResult("SHADOW_PORTFOLIO_INFEASIBLE", None, rejected)

    def objective(
        candidate: FeasiblePortfolio,
    ) -> tuple[Decimal, Decimal, tuple[str, ...], str]:
        securities = tuple(
            sorted(
                symbol
                for symbol, weight in candidate.target_weights.items()
                if _decimal(weight, label="target weight") > 0
            )
        )
        return (
            -candidate.net_adjusted_q25,
            _decimal(candidate.turnover, label="turnover"),
            securities,
            candidate.candidate_id,
        )

    return OptimizerResult("PORTFOLIO_SELECTED", min(feasible, key=objective), rejected)
