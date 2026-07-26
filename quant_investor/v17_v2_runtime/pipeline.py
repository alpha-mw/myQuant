"""Pure in-memory pipeline for the v17 protocol-v2 shadow runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .algorithms.deep_research import DeepResearchEvaluation, evaluate_deep_research
from .algorithms.forward_calibration import (
    assess_fundamental_eligibility,
    calibrate_forward_returns,
)
from .algorithms.fundamental_scoring import score_fundamental_universe
from .algorithms.optimizer import (
    FeasiblePortfolio,
    ProposedTrade,
    optimize_lexicographic,
)
from .algorithms.permissions import determine_trade_permission
from .algorithms.quant_timing import (
    TimingCalibration,
    calibrate_timing_probabilities,
    compute_latest_scores,
    decide_timing,
)
from .algorithms.regime_overlay import (
    build_available_overlay_input,
    build_disabled_overlay_input,
    compute_regime_portfolio_overlay,
)
from .algorithms.transaction_cost import estimate_transaction_cost

PROTOCOL_VERSION = "myquant.v17.v2"
RUNTIME_AUTHORITY = False


def _decimal(value: object, *, label: str, positive: bool = False) -> Decimal:
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


def _signed_decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _wire(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("wire output cannot contain a non-finite float")
        return result
    if isinstance(value, (tuple, list)):
        return [_wire(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _wire(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    raise ValueError(f"wire output contains unsupported type: {type(value).__name__}")


@dataclass(frozen=True)
class PipelineInput:
    cutoff: str
    strategy_id: str
    fundamental_rows: tuple[Mapping[str, Any], ...]
    fundamental_history: tuple[Mapping[str, Any], ...]
    forward_observations: tuple[Mapping[str, Any], ...]
    price_history: Mapping[str, tuple[Mapping[str, Any], ...]]
    timing_observations: tuple[Mapping[str, Any], ...]
    deep_response: Mapping[str, Any]
    sealed_evidence_ids: Mapping[str, tuple[str, ...]]
    holdings: Mapping[str, Decimal | str | int | float]
    cash: Decimal | str | int | float
    nav: Decimal | str | int | float
    risk_policy: Mapping[str, Any]
    cost_policy: Mapping[str, Any]
    tradability: Mapping[str, bool]
    risk_model: Mapping[str, Mapping[str, Any]]
    clusters: Mapping[str, str]
    macro: Mapping[str, Any]
    markov: Mapping[str, Any]
    portfolio_candidates: tuple[Mapping[str, Any], ...] = ()

    def to_wire(self) -> dict[str, Any]:
        return _wire(
            {
                "version": f"{PROTOCOL_VERSION}.pipeline-input.v1",
                "cutoff": self.cutoff,
                "strategy_id": self.strategy_id,
                "fundamental_rows": self.fundamental_rows,
                "fundamental_history": self.fundamental_history,
                "forward_observations": self.forward_observations,
                "price_history": self.price_history,
                "timing_observations": self.timing_observations,
                "deep_response": self.deep_response,
                "sealed_evidence_ids": self.sealed_evidence_ids,
                "holdings": self.holdings,
                "cash": self.cash,
                "nav": self.nav,
                "risk_policy": self.risk_policy,
                "cost_policy": self.cost_policy,
                "tradability": self.tradability,
                "risk_model": self.risk_model,
                "clusters": self.clusters,
                "macro": self.macro,
                "markov": self.markov,
                "portfolio_candidates": self.portfolio_candidates,
                "authority": False,
            }
        )


@dataclass(frozen=True)
class RankedCandidate:
    symbol: str
    fundamental_score: float | None
    fundamental_score_decile: int | None
    base_q25_by_horizon: Mapping[int, float]
    deep_status: str
    f_eligible: bool
    severe_red_flags: tuple[str, ...]
    adjusted_q25_252: float | None
    quant_score_decile: int | None
    probability_20d: float | None
    probability_60d: float | None
    timing_state: str
    selected_top24: bool
    appended_holding: bool
    blockers: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        return _wire(
            {
                "symbol": self.symbol,
                "fundamental_score": self.fundamental_score,
                "fundamental_score_decile": self.fundamental_score_decile,
                "base_q25_by_horizon": self.base_q25_by_horizon,
                "deep_status": self.deep_status,
                "f_eligible": self.f_eligible,
                "severe_red_flags": self.severe_red_flags,
                "adjusted_q25_252": self.adjusted_q25_252,
                "quant_score_decile": self.quant_score_decile,
                "probability_20d": self.probability_20d,
                "probability_60d": self.probability_60d,
                "timing_state": self.timing_state,
                "selected_top24": self.selected_top24,
                "appended_holding": self.appended_holding,
                "blockers": self.blockers,
                "authority": False,
            }
        )


@dataclass(frozen=True)
class RankOutput:
    initial_ranked_symbols: tuple[str, ...]
    eligible_ranked_symbols: tuple[str, ...]
    sealed_symbols: tuple[str, ...]
    rows: tuple[RankedCandidate, ...]
    authority: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "version": f"{PROTOCOL_VERSION}.rank-output.v1",
            "initial_ranked_symbols": list(self.initial_ranked_symbols),
            "eligible_ranked_symbols": list(self.eligible_ranked_symbols),
            "sealed_symbols": list(self.sealed_symbols),
            "rows": [row.to_wire() for row in self.rows],
            "authority": False,
        }


@dataclass(frozen=True)
class PortfolioOutput:
    candidate_id: str
    target_weights: Mapping[str, Decimal]
    shadow_trade_deltas: tuple[ProposedTrade, ...]
    expected_adjusted_q25: Decimal
    transaction_cost_fraction: Decimal
    net_adjusted_q25: Decimal
    turnover: Decimal
    regime_overlay: Mapping[str, Any]
    permissions: tuple[Mapping[str, Any], ...]
    optimizer_rejections: Mapping[str, tuple[str, ...]]
    authority: bool = False

    def to_wire(self) -> dict[str, Any]:
        return _wire(
            {
                "version": f"{PROTOCOL_VERSION}.portfolio-output.v1",
                "candidate_id": self.candidate_id,
                "target_weights": self.target_weights,
                "shadow_trade_deltas": tuple(trade.to_wire() for trade in self.shadow_trade_deltas),
                "expected_adjusted_q25": self.expected_adjusted_q25,
                "transaction_cost_fraction": self.transaction_cost_fraction,
                "net_adjusted_q25": self.net_adjusted_q25,
                "turnover": self.turnover,
                "regime_overlay": self.regime_overlay,
                "permissions": self.permissions,
                "optimizer_rejections": self.optimizer_rejections,
                "authority": False,
            }
        )


@dataclass(frozen=True)
class PipelineResult:
    rank_output: RankOutput
    portfolio_output: PortfolioOutput | None
    terminal_state: str
    blockers: tuple[str, ...]
    authority: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "version": f"{PROTOCOL_VERSION}.pipeline-result.v1",
            "rank_output": self.rank_output.to_wire(),
            "portfolio_output": (
                self.portfolio_output.to_wire() if self.portfolio_output is not None else None
            ),
            "terminal_state": self.terminal_state,
            "blockers": list(self.blockers),
            "authority": False,
        }


def _empty_result(blocker: str) -> PipelineResult:
    return PipelineResult(
        rank_output=RankOutput((), (), (), ()),
        portfolio_output=None,
        terminal_state="HARD_STOP_INVALID_EVIDENCE",
        blockers=(blocker,),
    )


def _score_deciles(scored: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for _, group in scored.loc[scored["status"] == "AVAILABLE"].groupby("industry", sort=True):
        ordered = group.sort_values(
            ["total_score", "symbol"], ascending=[True, True], kind="mergesort"
        )
        count = len(ordered)
        for position, (_, row) in enumerate(ordered.iterrows()):
            result[str(row["symbol"])] = min(10, (position * 10) // count + 1)
    return result


def _deep_responses(value: Mapping[str, Any]) -> Mapping[str, Mapping[str, object]]:
    if set(value).issuperset({"symbol", "layers", "coverage", "signals", "severe_red_flags"}):
        symbol = value.get("symbol")
        return {str(symbol): value}
    result: dict[str, Mapping[str, object]] = {}
    for symbol, response in value.items():
        if not isinstance(response, Mapping):
            raise ValueError(f"deep response for {symbol} must be an object")
        result[str(symbol)] = response
    return result


def _unavailable_deep(base_q25: float | None, reason: str) -> DeepResearchEvaluation:
    return DeepResearchEvaluation(
        "DEEP_RESEARCH_UNAVAILABLE",
        False,
        False,
        False,
        (),
        0.0,
        0.0,
        base_q25,
        None,
        (reason,),
    )


def _regime_input(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    return value if value else build_disabled_overlay_input(name=name)


def _permissions_to_actions(permission: Mapping[str, Any]) -> frozenset[str]:
    actions: set[str] = set()
    if permission["can_buy"]:
        actions.add("BUY")
    if permission["can_sell"]:
        actions.add("SELL")
    if permission["position_locked"]:
        actions.add("LOCK")
    return frozenset(actions or {"LOCK"})


def _candidate_risk_blockers(
    target: Mapping[str, Decimal],
    *,
    current: Mapping[str, Decimal],
    risk_policy: Mapping[str, Any],
    risk_model: Mapping[str, Mapping[str, Any]],
    clusters: Mapping[str, str],
    industry_by_symbol: Mapping[str, str],
    effective_gross: Decimal,
) -> tuple[str, ...]:
    blockers: list[str] = []
    gross = sum(target.values(), Decimal("0"))
    turnover = sum(
        (
            abs(target.get(symbol, Decimal("0")) - current.get(symbol, Decimal("0")))
            for symbol in set(target).union(current)
        ),
        Decimal("0"),
    )
    if gross > effective_gross:
        blockers.append("effective_gross_exceeded")
    caps = {
        "single_name_cap": Decimal("1"),
        "industry_cap": Decimal("1"),
        "cluster_cap": Decimal("1"),
        "beta_cap": Decimal("1"),
        "stress_loss_cap": Decimal("1"),
        "turnover_cap": Decimal("2"),
    }
    for name in tuple(caps):
        if name in risk_policy:
            caps[name] = _decimal(risk_policy[name], label=name)
    if any(weight > caps["single_name_cap"] for weight in target.values()):
        blockers.append("single_name_cap_exceeded")
    industry_weights: dict[str, Decimal] = {}
    cluster_weights: dict[str, Decimal] = {}
    beta = Decimal("0")
    stress = Decimal("0")
    for symbol, weight in target.items():
        if symbol not in risk_model:
            blockers.append(f"risk_model_missing:{symbol}")
            continue
        if symbol not in clusters:
            blockers.append(f"cluster_missing:{symbol}")
            continue
        industry = industry_by_symbol[symbol]
        cluster = clusters[symbol]
        industry_weights[industry] = industry_weights.get(industry, Decimal("0")) + weight
        cluster_weights[cluster] = cluster_weights.get(cluster, Decimal("0")) + weight
        beta += weight * _signed_decimal(risk_model[symbol].get("beta"), label=f"{symbol}.beta")
        stress += weight * _decimal(
            risk_model[symbol].get("stress_loss"), label=f"{symbol}.stress_loss"
        )
    if any(value > caps["industry_cap"] for value in industry_weights.values()):
        blockers.append("industry_cap_exceeded")
    if any(value > caps["cluster_cap"] for value in cluster_weights.values()):
        blockers.append("cluster_cap_exceeded")
    if abs(beta) > caps["beta_cap"]:
        blockers.append("beta_cap_exceeded")
    if stress > caps["stress_loss_cap"]:
        blockers.append("stress_cap_exceeded")
    if turnover > caps["turnover_cap"]:
        blockers.append("turnover_cap_exceeded")
    return tuple(dict.fromkeys(blockers))


def _portfolio(
    input: PipelineInput,
    *,
    rank: RankOutput,
    industry_by_symbol: Mapping[str, str],
) -> tuple[PortfolioOutput | None, tuple[str, ...]]:
    if not input.portfolio_candidates:
        return None, ("no_portfolio_candidates",)
    nav = _decimal(input.nav, label="nav", positive=True)
    _decimal(input.cash, label="cash")
    current = {
        symbol: _decimal(weight, label=f"holdings.{symbol}")
        for symbol, weight in input.holdings.items()
    }
    base = build_available_overlay_input(
        name="base",
        gross_cap=float(_decimal(input.risk_policy.get("gross_cap"), label="gross_cap")),
        cash_floor=float(_decimal(input.risk_policy.get("cash_floor"), label="cash_floor")),
    )
    overlay = compute_regime_portfolio_overlay(
        base=base,
        macro=_regime_input(input.macro, name="macro"),
        markov=_regime_input(input.markov, name="markov"),
    )
    if overlay["availability"] != "AVAILABLE":
        return None, ("regime_overlay_unavailable",)
    effective_gross = _decimal(overlay["effective_gross"], label="effective_gross")
    by_symbol = {row.symbol: row for row in rank.rows}
    if any(row.timing_state == "UNREADY" for row in rank.rows):
        return None, ("quant_timing_unready",)
    permissions: dict[str, Mapping[str, Any]] = {}
    permission_mask: dict[str, frozenset[str]] = {}
    for symbol in rank.sealed_symbols:
        row = by_symbol[symbol]
        if symbol not in input.tradability or type(input.tradability[symbol]) is not bool:
            raise ValueError(f"tradability missing or invalid: {symbol}")
        permission = determine_trade_permission(
            symbol=symbol,
            held=symbol in current,
            tradable=input.tradability[symbol],
            fundamental_eligibility="F_ELIGIBLE" if row.f_eligible else "F_INELIGIBLE",
            severe_red_flag=bool(row.severe_red_flags),
            quant_timing=row.timing_state,
        )
        permissions[symbol] = permission
        permission_mask[symbol] = _permissions_to_actions(permission)

    feasible: list[FeasiblePortfolio] = []
    rejected: dict[str, tuple[str, ...]] = {}
    coefficient = _decimal(input.cost_policy.get("coefficient"), label="cost coefficient")
    for raw in input.portfolio_candidates:
        if set(raw) != {"candidate_id", "target_weights"}:
            raise ValueError("portfolio candidate keys mismatch")
        candidate_id = raw["candidate_id"]
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id.strip() != candidate_id
        ):
            raise ValueError("candidate_id invalid")
        raw_weights = raw["target_weights"]
        if not isinstance(raw_weights, Mapping):
            raise ValueError(f"target_weights must be an object: {candidate_id}")
        target = {
            str(symbol): _decimal(weight, label=f"{candidate_id}.{symbol}")
            for symbol, weight in raw_weights.items()
        }
        unknown = sorted(set(target).difference(rank.sealed_symbols))
        if unknown:
            rejected[candidate_id] = tuple(f"target_symbol_unsealed:{symbol}" for symbol in unknown)
            continue
        risk_blockers = _candidate_risk_blockers(
            target,
            current=current,
            risk_policy=input.risk_policy,
            risk_model=input.risk_model,
            clusters=input.clusters,
            industry_by_symbol=industry_by_symbol,
            effective_gross=effective_gross,
        )
        if risk_blockers:
            rejected[candidate_id] = risk_blockers
            continue
        trades: list[ProposedTrade] = []
        total_cost = Decimal("0")
        for symbol in sorted(set(target).union(current)):
            delta = target.get(symbol, Decimal("0")) - current.get(symbol, Decimal("0"))
            if delta == 0:
                continue
            action = "BUY" if delta > 0 else "SELL"
            trades.append(ProposedTrade(symbol, action, abs(delta)))
            model = input.risk_model.get(symbol)
            if model is None:
                raise ValueError(f"risk_model missing: {symbol}")
            estimate = estimate_transaction_cost(
                coefficient=coefficient,
                notional=abs(delta) * nav,
                adv20=model.get("adv20"),
            )
            total_cost += estimate.amount
        turnover = sum(
            (abs(_decimal(trade.notional_fraction, label="trade")) for trade in trades),
            Decimal("0"),
        )
        expected = sum(
            (
                weight * Decimal(str(by_symbol[symbol].adjusted_q25_252))
                for symbol, weight in target.items()
                if by_symbol[symbol].adjusted_q25_252 is not None
            ),
            Decimal("0"),
        )
        feasible.append(
            FeasiblePortfolio(
                candidate_id,
                target,
                tuple(trades),
                expected,
                total_cost / nav,
                turnover,
            )
        )
    optimized = optimize_lexicographic(
        feasible,
        permission_mask=permission_mask,
        current_weights=current,
        effective_gross=effective_gross,
    )
    all_rejected = dict(rejected)
    all_rejected.update(optimized.rejected)
    if optimized.selected is None:
        return None, ("no_feasible_portfolio_candidate",)
    selected = optimized.selected
    expected = _decimal(selected.expected_adjusted_q25, label="expected_adjusted_q25")
    cost = _decimal(selected.transaction_cost, label="transaction_cost")
    return (
        PortfolioOutput(
            candidate_id=selected.candidate_id,
            target_weights={
                symbol: _decimal(weight, label=f"target_weights.{symbol}")
                for symbol, weight in selected.target_weights.items()
            },
            shadow_trade_deltas=selected.trades,
            expected_adjusted_q25=expected,
            transaction_cost_fraction=cost,
            net_adjusted_q25=expected - cost,
            turnover=_decimal(selected.turnover, label="turnover"),
            regime_overlay=overlay,
            permissions=tuple(permissions[symbol] for symbol in rank.sealed_symbols),
            optimizer_rejections=all_rejected,
        ),
        (),
    )


def run_deterministic_pipeline(input: PipelineInput) -> PipelineResult:
    """Run the isolated shadow calculation without any I/O or action authority."""

    if not isinstance(input, PipelineInput):
        raise TypeError("input must be PipelineInput")
    try:
        cutoff = pd.Timestamp(input.cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        if not isinstance(input.strategy_id, str) or not input.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        candidates = score_fundamental_universe(
            pd.DataFrame(input.fundamental_rows),
            pd.DataFrame(input.fundamental_history),
            cutoff=input.cutoff,
            holdings=tuple(input.holdings),
            top_n=24,
        )
        if not candidates.ranked_symbols:
            return _empty_result("fundamental_produced_no_ranked_candidates")
        calibration = calibrate_forward_returns(
            pd.DataFrame(input.forward_observations),
            cutoff=input.cutoff,
        )
        fundamental_deciles = _score_deciles(candidates.scored)
        if input.price_history:
            price_frames = {
                symbol: pd.DataFrame(rows) for symbol, rows in input.price_history.items()
            }
            latest = compute_latest_scores(
                price_frames,
                sealed_symbols=candidates.sealed_symbols,
                cutoff=input.cutoff,
            )
        else:
            latest = pd.DataFrame(
                {
                    "symbol": candidates.sealed_symbols,
                    "composite_score": [np.nan] * len(candidates.sealed_symbols),
                    "status": ["UNREADY"] * len(candidates.sealed_symbols),
                }
            )
        timing_calibration = calibrate_timing_probabilities(
            pd.DataFrame(input.timing_observations),
            cutoff=input.cutoff,
        )
        timing = decide_timing(latest, timing_calibration).set_index("symbol", drop=False)
        deep_responses = _deep_responses(input.deep_response)
        scored = candidates.scored.set_index("symbol", drop=False)
        rows: list[RankedCandidate] = []
        for symbol in candidates.sealed_symbols:
            score_row = scored.loc[symbol]
            decile = fundamental_deciles.get(symbol)
            if decile is None:
                base_values: Mapping[int, float] = {}
                base_eligible = False
                base_blockers = ("fundamental_score_unavailable",)
            else:
                base = assess_fundamental_eligibility(
                    calibration,
                    industry=str(score_row["industry"]),
                    score_decile=decile,
                    deep_research_complete=True,
                    severe_red_flags=False,
                )
                base_values = base.base_q25_by_horizon
                base_eligible = base.eligible
                base_blockers = base.blockers
            if symbol in deep_responses:
                deep = evaluate_deep_research(
                    deep_responses[symbol],
                    sealed_symbol=symbol,
                    sealed_evidence_ids=input.sealed_evidence_ids.get(symbol, ()),
                    base_q25_by_horizon=base_values,
                    base_eligible=base_eligible,
                )
            else:
                deep = _unavailable_deep(
                    base_values.get(252),
                    "deep_research_response_missing",
                )
            timed = timing.loc[symbol]
            quant_decile = timed["score_decile"]
            total_score = score_row["total_score"]
            rows.append(
                RankedCandidate(
                    symbol=symbol,
                    fundamental_score=float(total_score) if pd.notna(total_score) else None,
                    fundamental_score_decile=decile,
                    base_q25_by_horizon=dict(base_values),
                    deep_status=deep.status,
                    f_eligible=deep.f_eligible,
                    severe_red_flags=deep.severe_red_flags,
                    adjusted_q25_252=deep.adjusted_q25_252,
                    quant_score_decile=int(quant_decile) if not pd.isna(quant_decile) else None,
                    probability_20d=(
                        float(timed["probability_20d"])
                        if np.isfinite(timed["probability_20d"])
                        else None
                    ),
                    probability_60d=(
                        float(timed["probability_60d"])
                        if np.isfinite(timed["probability_60d"])
                        else None
                    ),
                    timing_state=str(timed["timing_state"]),
                    selected_top24=symbol in candidates.ranked_symbols,
                    appended_holding=symbol in candidates.appended_holdings,
                    blockers=tuple(
                        dict.fromkeys(
                            [
                                *base_blockers,
                                *deep.blockers,
                                *(
                                    timing_calibration.blockers
                                    if str(timed["timing_state"]) == "UNREADY"
                                    else ()
                                ),
                            ]
                        )
                    ),
                )
            )
        by_symbol = {row.symbol: row for row in rows}
        rank = RankOutput(
            initial_ranked_symbols=candidates.ranked_symbols,
            eligible_ranked_symbols=tuple(
                symbol for symbol in candidates.ranked_symbols if by_symbol[symbol].f_eligible
            ),
            sealed_symbols=candidates.sealed_symbols,
            rows=tuple(rows),
        )
        portfolio, portfolio_blockers = _portfolio(
            input,
            rank=rank,
            industry_by_symbol={
                symbol: str(scored.loc[symbol]["industry"]) for symbol in rank.sealed_symbols
            },
        )
        if portfolio is not None:
            terminal = "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION"
            blockers: tuple[str, ...] = ()
        elif portfolio_blockers == ("no_portfolio_candidates",):
            terminal = "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"
            blockers = portfolio_blockers
        else:
            terminal = "SHADOW_PORTFOLIO_INFEASIBLE"
            blockers = portfolio_blockers
        return PipelineResult(rank, portfolio, terminal, blockers)
    except (KeyError, TypeError, ValueError) as exc:
        return _empty_result(f"invalid_pipeline_input:{exc}")


__all__ = [
    "PipelineInput",
    "PipelineResult",
    "PortfolioOutput",
    "RankOutput",
    "RankedCandidate",
    "run_deterministic_pipeline",
]
