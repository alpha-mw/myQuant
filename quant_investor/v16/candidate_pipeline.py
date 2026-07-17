"""Deterministic candidate and IC contracts for the v16 four-branch design.

This module intentionally owns no provider, registry, broker, or order side
effects.  It establishes the bounded candidate flow and validates Codex Stage
1/Stage 2 responses without allowing retrieval annotations or advisory risk
output to modify formal branch evidence or portfolio decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

FUNNEL_LIMIT = 500
SUPPLEMENTAL_LIMIT = 100
UNION_LIMIT = 600
MENU_LIMIT = 50
POSITIVE_WEIGHT_LIMIT = 12
WEIGHT_TOLERANCE = 1e-6

FORMAL_BRANCHES = ("quant", "fundamental", "macro", "llm")
RETRIEVAL_BRANCHES = frozenset({"quant", "fundamental", "macro"})
IC_ACTIONS = frozenset({"BUY", "HOLD", "AVOID", "SELL"})


def _normalise_symbol(value: str) -> str:
    symbol = str(value).strip().upper()
    if not symbol:
        raise ValueError("symbol must be non-empty")
    return symbol


def _stable_unique(symbols: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = _normalise_symbol(raw_symbol)
        if symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return tuple(result)


@dataclass(frozen=True)
class CandidateUnion:
    """Sealed, deterministic union of Quant funnel and LLM additions."""

    funnel_symbols: tuple[str, ...]
    supplemental_symbols: tuple[str, ...]
    symbols: tuple[str, ...]

    @property
    def source_by_symbol(self) -> Mapping[str, str]:
        funnel = set(self.funnel_symbols)
        return MappingProxyType(
            {
                symbol: "quant_funnel" if symbol in funnel else "llm_supplemental"
                for symbol in self.symbols
            }
        )


def build_candidate_union(
    funnel_symbols: Iterable[str],
    supplemental_symbols: Iterable[str],
    *,
    funnel_limit: int = FUNNEL_LIMIT,
    supplemental_limit: int = SUPPLEMENTAL_LIMIT,
    union_limit: int = UNION_LIMIT,
) -> CandidateUnion:
    """Build the stable 500+100/600 candidate union without implicit truncation.

    Input ordering is treated as a sealed ranking: funnel names retain their
    deterministic Quant order and genuinely new LLM names retain their Stage 1
    order.  Duplicate symbols are removed stably.  Exceeding any boundary is a
    protocol error rather than a request to silently discard evidence.
    """

    funnel = _stable_unique(funnel_symbols)
    supplemental_all = _stable_unique(supplemental_symbols)
    if len(funnel) > funnel_limit:
        raise ValueError(f"Quant funnel exceeds {funnel_limit} symbols")
    if len(supplemental_all) > supplemental_limit:
        raise ValueError(f"Stage 1 supplemental set exceeds {supplemental_limit} symbols")

    funnel_set = set(funnel)
    supplemental = tuple(symbol for symbol in supplemental_all if symbol not in funnel_set)
    symbols = funnel + supplemental
    if len(symbols) > union_limit:
        raise ValueError(f"candidate union exceeds {union_limit} symbols")
    return CandidateUnion(
        funnel_symbols=funnel,
        supplemental_symbols=supplemental,
        symbols=symbols,
    )


@dataclass(frozen=True)
class RetrievalEvidence:
    """Advisory Q/F/M note that cannot carry a score or likelihood."""

    symbol: str
    branch: str
    supporting_fact_ids: tuple[str, ...] = ()
    contradicting_fact_ids: tuple[str, ...] = ()
    conflict_note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalise_symbol(self.symbol))
        branch = str(self.branch).strip().lower()
        if branch not in RETRIEVAL_BRANCHES:
            raise ValueError("retrieval evidence branch must be quant, fundamental, or macro")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(
            self,
            "supporting_fact_ids",
            tuple(str(item).strip() for item in self.supporting_fact_ids if str(item).strip()),
        )
        object.__setattr__(
            self,
            "contradicting_fact_ids",
            tuple(str(item).strip() for item in self.contradicting_fact_ids if str(item).strip()),
        )
        if self.conflict_note is not None and not self.conflict_note.strip():
            raise ValueError("conflict_note must be non-empty when supplied")


@dataclass(frozen=True)
class LLMBranchVerdict:
    """Formal fourth-branch raw evidence for one sealed-union symbol."""

    symbol: str
    raw_score: float
    confidence: float
    supporting_fact_ids: tuple[str, ...]
    contradicting_fact_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalise_symbol(self.symbol))
        if not isfinite(self.raw_score) or not -1.0 <= self.raw_score <= 1.0:
            raise ValueError("LLM raw_score must be finite and within [-1, 1]")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("LLM confidence must be finite and within [0, 1]")
        if not self.rationale.strip():
            raise ValueError("LLM rationale must be non-empty")


@dataclass(frozen=True)
class FormalBranchEvidence:
    """Immutable raw evidence consumed by the v16 calibration layer."""

    symbol: str
    branch: str
    raw_score: float
    confidence: float
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalise_symbol(self.symbol))
        branch = str(self.branch).strip().lower()
        if branch not in FORMAL_BRANCHES:
            raise ValueError(f"formal branch must be one of {FORMAL_BRANCHES!r}")
        object.__setattr__(self, "branch", branch)
        if not isfinite(self.raw_score) or not -1.0 <= self.raw_score <= 1.0:
            raise ValueError("formal raw_score must be finite and within [-1, 1]")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("formal confidence must be finite and within [0, 1]")
        evidence_ids = tuple(str(item).strip() for item in self.evidence_ids if str(item).strip())
        if not evidence_ids:
            raise ValueError("formal evidence requires at least one evidence id")
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True)
class FourBranchEvidence:
    """Exactly one Q/F/M/LLM record for a single union symbol."""

    symbol: str
    branches: tuple[FormalBranchEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalise_symbol(self.symbol))
        branch_names = tuple(item.branch for item in self.branches)
        if branch_names != FORMAL_BRANCHES:
            raise ValueError(
                "four-branch evidence must use exact canonical order "
                f"{FORMAL_BRANCHES!r}; got {branch_names!r}"
            )
        if any(item.symbol != self.symbol for item in self.branches):
            raise ValueError("four-branch evidence contains symbol drift")


def seal_four_branch_evidence(
    candidate_union: CandidateUnion,
    records: Sequence[FormalBranchEvidence],
) -> tuple[FourBranchEvidence, ...]:
    """Seal strict evidence; a missing branch never receives a neutral value."""

    expected_symbols = set(candidate_union.symbols)
    grouped: dict[str, dict[str, FormalBranchEvidence]] = {
        symbol: {} for symbol in candidate_union.symbols
    }
    for record in records:
        if record.symbol not in expected_symbols:
            raise ValueError(f"formal evidence outside sealed union: {record.symbol}")
        if record.branch in grouped[record.symbol]:
            raise ValueError(f"duplicate formal evidence for {record.symbol}/{record.branch}")
        grouped[record.symbol][record.branch] = record

    sealed: list[FourBranchEvidence] = []
    for symbol in candidate_union.symbols:
        branch_map = grouped[symbol]
        missing = [branch for branch in FORMAL_BRANCHES if branch not in branch_map]
        if missing:
            raise ValueError(f"missing formal branches for {symbol}: {missing}")
        sealed.append(
            FourBranchEvidence(
                symbol=symbol,
                branches=tuple(branch_map[branch] for branch in FORMAL_BRANCHES),
            )
        )
    return tuple(sealed)


def validate_stage1_review(
    candidate_union: CandidateUnion,
    *,
    llm_verdicts: Sequence[LLMBranchVerdict],
    retrieval_evidence: Sequence[RetrievalEvidence],
) -> None:
    """Require exactly one formal LLM verdict for every union symbol.

    Retrieval evidence is deliberately validated as a separate annotation
    collection.  Its type has no score, confidence, probability, likelihood,
    or branch-weight fields, which prevents Stage 1 from overwriting Q/F/M.
    """

    expected = set(candidate_union.symbols)
    verdict_symbols = [item.symbol for item in llm_verdicts]
    if len(verdict_symbols) != len(set(verdict_symbols)):
        raise ValueError("duplicate LLM verdict symbol")
    if set(verdict_symbols) != expected:
        missing = sorted(expected - set(verdict_symbols))
        extra = sorted(set(verdict_symbols) - expected)
        raise ValueError(f"LLM verdict symbol-set drift: missing={missing}, extra={extra}")

    for item in retrieval_evidence:
        if item.symbol not in expected:
            raise ValueError(f"retrieval evidence outside sealed union: {item.symbol}")


@dataclass(frozen=True)
class PosteriorMenuItem:
    symbol: str
    posterior_win_rate: float
    posterior_expected_alpha: float
    posterior_edge_after_costs: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalise_symbol(self.symbol))
        if not isfinite(self.posterior_win_rate) or not 0.0 <= self.posterior_win_rate <= 1.0:
            raise ValueError("posterior_win_rate must be finite and within [0, 1]")
        if not isfinite(self.posterior_expected_alpha):
            raise ValueError("posterior_expected_alpha must be finite")
        if self.posterior_edge_after_costs is not None and not isfinite(
            self.posterior_edge_after_costs
        ):
            raise ValueError("posterior_edge_after_costs must be finite when available")


def build_posterior_menu(
    items: Iterable[PosteriorMenuItem],
    *,
    menu_limit: int = MENU_LIMIT,
) -> tuple[PosteriorMenuItem, ...]:
    """Sort and seal the Bayesian menu without filtering negative edge.

    Unavailable edge is retained but ranks after all available edge values.
    This is explicit fail-visible behavior; no cost-free or neutral substitute
    is invented.
    """

    materialised = tuple(items)
    symbols = [item.symbol for item in materialised]
    if len(symbols) != len(set(symbols)):
        raise ValueError("duplicate posterior symbol")
    if menu_limit < 0:
        raise ValueError("menu_limit must be non-negative")

    def sort_key(item: PosteriorMenuItem) -> tuple[bool, float, float, str]:
        edge = item.posterior_edge_after_costs
        return (
            edge is None,
            -(edge if edge is not None else 0.0),
            -item.posterior_win_rate,
            item.symbol,
        )

    return tuple(sorted(materialised, key=sort_key)[:menu_limit])


@dataclass(frozen=True)
class Stage2Decision:
    symbol: str
    action: str
    selected_for_portfolio: bool
    target_weight: float
    rationale: str
    risk_acceptance_rationale: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalise_symbol(self.symbol))
        action = str(self.action).strip().upper()
        if action not in IC_ACTIONS:
            raise ValueError(f"unsupported Stage 2 action: {action}")
        object.__setattr__(self, "action", action)
        if not isfinite(self.target_weight) or not 0.0 <= self.target_weight <= 1.0:
            raise ValueError("target_weight must be finite and within [0, 1]")
        if not self.rationale.strip():
            raise ValueError("Stage 2 rationale must be non-empty")
        if (
            self.risk_acceptance_rationale is not None
            and not self.risk_acceptance_rationale.strip()
        ):
            raise ValueError("risk_acceptance_rationale must be non-empty when supplied")


@dataclass(frozen=True)
class Stage2PortfolioDecision:
    decisions: tuple[Stage2Decision, ...]
    cash_ratio: float


@dataclass(frozen=True)
class CapitalTarget:
    symbol: str
    action: str
    target_weight: float
    target_capital: float
    raw_target_shares: float


@dataclass(frozen=True)
class CapitalMapping:
    total_capital: float
    positions: tuple[CapitalTarget, ...]
    cash_ratio: float
    cash_amount: float


def map_portfolio_capital(
    portfolio: Stage2PortfolioDecision,
    *,
    total_capital: float,
    reference_prices: Mapping[str, float],
    existing_shares: Mapping[str, float] | None,
) -> CapitalMapping:
    """Map authoritative weights to capital and unrounded target shares only.

    Reference prices are valuation inputs, not an execution-quote substitute.
    Lot rounding, quote freshness, available cash, and order feasibility remain
    the sole responsibility of ExecutionGate.
    """

    capital = float(total_capital)
    if not isfinite(capital) or capital <= 0.0:
        raise ValueError("total_capital must be finite and positive")
    if existing_shares is None:
        raise ValueError("existing share information is required")

    prices: dict[str, float] = {}
    for raw_symbol, raw_price in reference_prices.items():
        symbol = _normalise_symbol(raw_symbol)
        price = float(raw_price)
        if not isfinite(price) or price <= 0.0:
            raise ValueError(f"invalid reference price for {symbol}")
        prices[symbol] = price
    current_shares: dict[str, float] = {}
    for raw_symbol, raw_shares in existing_shares.items():
        symbol = _normalise_symbol(raw_symbol)
        shares = float(raw_shares)
        if not isfinite(shares) or shares < 0.0:
            raise ValueError(f"invalid existing shares for {symbol}")
        current_shares[symbol] = shares

    positions: list[CapitalTarget] = []
    for decision in portfolio.decisions:
        target_capital = capital * decision.target_weight
        if decision.action == "BUY":
            if decision.symbol not in prices:
                raise ValueError(f"BUY reference price missing: {decision.symbol}")
            raw_target_shares = target_capital / prices[decision.symbol]
        elif decision.action == "HOLD":
            if decision.symbol not in current_shares:
                raise ValueError(f"HOLD existing shares missing: {decision.symbol}")
            raw_target_shares = current_shares[decision.symbol]
        else:
            raw_target_shares = 0.0

        if decision.target_weight > WEIGHT_TOLERANCE:
            positions.append(
                CapitalTarget(
                    symbol=decision.symbol,
                    action=decision.action,
                    target_weight=decision.target_weight,
                    target_capital=target_capital,
                    raw_target_shares=raw_target_shares,
                )
            )

    return CapitalMapping(
        total_capital=capital,
        positions=tuple(positions),
        cash_ratio=portfolio.cash_ratio,
        cash_amount=capital * portfolio.cash_ratio,
    )


def validate_stage2_portfolio(
    menu: Sequence[PosteriorMenuItem],
    decisions: Sequence[Stage2Decision],
    *,
    cash_ratio: float,
    existing_weights: Mapping[str, float] | None,
    severe_risk_symbols: Iterable[str] = (),
    tolerance: float = WEIGHT_TOLERANCE,
) -> Stage2PortfolioDecision:
    """Validate the authoritative IC allocation without normalising or capping it.

    RiskAdvisor output is read only to enforce an audit explanation for a BUY
    that knowingly accepts a severe warning.  It never changes action, target
    weight, candidate rank, or selection.
    """

    if existing_weights is None:
        raise ValueError("complete existing holding information is required")
    if not isfinite(cash_ratio) or not 0.0 <= cash_ratio <= 1.0:
        raise ValueError("cash_ratio must be finite and within [0, 1]")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    menu_symbols = tuple(item.symbol for item in menu)
    if len(menu_symbols) != len(set(menu_symbols)):
        raise ValueError("sealed menu contains duplicate symbols")
    decision_symbols = [item.symbol for item in decisions]
    if len(decision_symbols) != len(set(decision_symbols)):
        raise ValueError("duplicate Stage 2 decision symbol")
    if set(decision_symbols) != set(menu_symbols):
        missing = sorted(set(menu_symbols) - set(decision_symbols))
        extra = sorted(set(decision_symbols) - set(menu_symbols))
        raise ValueError(f"Stage 2 symbol-set drift: missing={missing}, extra={extra}")

    normalised_existing: dict[str, float] = {}
    for raw_symbol, raw_weight in existing_weights.items():
        symbol = _normalise_symbol(raw_symbol)
        weight = float(raw_weight)
        if not isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(f"invalid existing weight for {symbol}")
        normalised_existing[symbol] = weight
    if set(menu_symbols) - set(normalised_existing):
        missing = sorted(set(menu_symbols) - set(normalised_existing))
        raise ValueError(f"existing holding information incomplete: {missing}")
    held_outside_menu = sorted(
        symbol
        for symbol, weight in normalised_existing.items()
        if weight > tolerance and symbol not in set(menu_symbols)
    )
    if held_outside_menu:
        raise ValueError(f"current holdings absent from sealed menu: {held_outside_menu}")

    severe = {_normalise_symbol(symbol) for symbol in severe_risk_symbols}
    positive_count = 0
    for item in decisions:
        existing = normalised_existing[item.symbol]
        positive = item.target_weight > tolerance
        if positive:
            positive_count += 1

        if item.action == "BUY":
            if not positive or not item.selected_for_portfolio:
                raise ValueError(f"BUY requires positive selected weight: {item.symbol}")
            if item.symbol in severe and not item.risk_acceptance_rationale:
                raise ValueError(
                    f"severe-risk BUY requires risk_acceptance_rationale: {item.symbol}"
                )
        elif item.action == "HOLD":
            if abs(item.target_weight - existing) > tolerance:
                raise ValueError(f"HOLD must preserve existing weight: {item.symbol}")
            if item.selected_for_portfolio != positive:
                raise ValueError(
                    f"HOLD selection must match positive existing weight: {item.symbol}"
                )
        else:
            if positive or item.selected_for_portfolio:
                raise ValueError(f"{item.action} requires zero unselected weight: {item.symbol}")

    if positive_count > POSITIVE_WEIGHT_LIMIT:
        raise ValueError(f"positive target weights exceed {POSITIVE_WEIGHT_LIMIT}")
    total = cash_ratio + sum(item.target_weight for item in decisions)
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"target weights plus cash_ratio must equal 1, got {total!r}")

    return Stage2PortfolioDecision(decisions=tuple(decisions), cash_ratio=cash_ratio)
