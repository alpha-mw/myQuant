"""Offline deterministic portfolio optimizer and walk-forward contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.portfolio_optimizer_types import (
    DEFAULT_OPTIMIZED_PLANS_FILENAME,
    DEFAULT_PORTFOLIO_OPTIMIZER_DIR,
    DEFAULT_REBALANCE_RESULTS_FILENAME,
    DEFAULT_WALK_FORWARD_RESULTS_FILENAME,
    PLAN_STATUS_EMPTY,
    PLAN_STATUS_INFEASIBLE,
    PLAN_STATUS_OPTIMIZED,
    CONSTRAINT_BLOCKED_SYMBOL,
    CONSTRAINT_GROSS_EXPOSURE,
    CONSTRAINT_MAX_NAMES,
    CONSTRAINT_MAX_WEIGHT,
    CONSTRAINT_MIN_EDGE,
    CONSTRAINT_RISK_SCORE,
    CONSTRAINT_SECTOR_CAP,
    CONSTRAINT_TURNOVER_CAP,
    VIOLATION_BLOCKER,
    VIOLATION_INFO,
    VIOLATION_WARNING,
    ConstraintViolation,
    OptimizationCandidate,
    OptimizedPortfolioPlan,
    PortfolioOptimizerConfig,
    RebalanceInput,
    RebalanceResult,
    WalkForwardResult,
    _EPSILON,
    _UNKNOWN_SECTOR,
    _candidate_adjusted_score,
    _clean_weight_dict,
    _coerce_metadata,
    _finite_float_dict,
    _float_or_none,
    _get_attr,
    _get_nested_attr,
    _json_safe,
    _make_violation,
    _metadata_float,
    _non_negative_float,
    _ordered_unique,
    _short_hash,
    bps_to_decimal_return,
    clamp_unit_interval,
    compound_returns,
    compute_sector_weights,
    estimate_turnover,
    make_constraint_violation_id,
    make_plan_id,
    make_rebalance_id,
    make_walk_forward_run_id,
    max_drawdown_from_returns,
    validate_finite_number,
)
from quant_investor.versioning import PORTFOLIO_OPTIMIZER_SCHEMA_VERSION


def build_candidate_from_overlay(
    overlay: Any,
    *,
    risk_tensor: Any | None = None,
    current_weight: float = 0.0,
    default_max_weight: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizationCandidate:
    input_metadata = _coerce_metadata(metadata)
    overlay_metadata = _get_attr(overlay, "metadata", {}) or {}
    diagnostics_metadata = _get_nested_attr(overlay, ["diagnostics", "metadata"], {}) or {}
    confidence = diagnostics_metadata.get("posterior_confidence")
    if confidence is None:
        confidence = overlay_metadata.get("posterior_confidence")
    if confidence is None:
        confidence = 0.0

    tensor_metadata = _get_attr(risk_tensor, "metadata", {}) or {}
    max_weight = _metadata_float(
        tensor_metadata if isinstance(tensor_metadata, Mapping) else {},
        ["max_weight", "max_symbol_weight", "target_max_weight", "position_cap"],
    )
    if max_weight is None:
        max_weight = default_max_weight

    execution = _get_attr(risk_tensor, "execution")
    liquidity = _get_attr(risk_tensor, "liquidity")
    exposure = _get_attr(risk_tensor, "exposure")

    execution_status = str(_get_attr(execution, "status", ""))
    is_blocked = (
        execution_status == "blocked"
        or bool(_get_attr(risk_tensor, "quarantine", False))
        or not bool(_get_attr(risk_tensor, "is_tradable", True))
    )
    execution_reasons = list(_get_attr(execution, "blocked_reasons", []) or [])
    issue_reasons: list[str] = []
    for issue in list(_get_attr(risk_tensor, "issues", []) or []):
        if str(_get_attr(issue, "severity", "")) == "blocker":
            issue_reasons.append(str(_get_attr(issue, "issue_type", "") or _get_attr(issue, "message", "")))
    block_reasons = _ordered_unique([reason for reason in [*execution_reasons, *issue_reasons] if reason])

    market = str(_get_attr(overlay, "market", "") or _get_attr(risk_tensor, "market", ""))
    as_of = str(_get_attr(overlay, "as_of", "") or _get_attr(risk_tensor, "as_of", "") or input_metadata.get("as_of", ""))
    estimated_slippage_bps = _get_attr(execution, "estimated_slippage_bps")
    if estimated_slippage_bps is None:
        spread_bps = _get_attr(liquidity, "estimated_spread_bps")
        estimated_slippage_bps = None if spread_bps is None else float(spread_bps) / 2.0

    return OptimizationCandidate(
        schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        symbol=str(_get_attr(overlay, "symbol", "")),
        market=market,
        as_of=as_of,
        company_name=str(_get_attr(overlay, "company_name", "")),
        sector=_get_attr(exposure, "sector"),
        current_weight=current_weight,
        max_weight=max_weight,
        expected_alpha=float(_get_attr(overlay, "calibrated_posterior_expected_alpha", 0.0)),
        edge_after_costs=float(_get_attr(overlay, "calibrated_edge_after_costs", 0.0)),
        confidence=float(confidence),
        action_score=float(_get_attr(overlay, "calibrated_posterior_action_score", 0.0)),
        risk_score=float(_get_attr(risk_tensor, "risk_score", 0.0) or 0.0),
        liquidity_score=_get_attr(liquidity, "liquidity_score"),
        estimated_transaction_cost_bps=_get_attr(execution, "estimated_transaction_cost_bps"),
        estimated_slippage_bps=estimated_slippage_bps,
        estimated_market_impact_bps=(
            _get_attr(execution, "estimated_market_impact_bps")
            if _get_attr(execution, "estimated_market_impact_bps") is not None
            else _get_attr(liquidity, "estimated_market_impact_bps")
        ),
        is_blocked=is_blocked,
        block_reasons=block_reasons,
        metadata={
            "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
            "input_metadata": input_metadata,
        },
    )


def build_candidates_from_overlays(
    overlays: Sequence[Any],
    *,
    risk_tensors_by_symbol: Mapping[str, Any] | None = None,
    current_weights: Mapping[str, float] | None = None,
    default_max_weight: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[OptimizationCandidate]:
    tensors = risk_tensors_by_symbol or {}
    weights = current_weights or {}
    candidates: list[OptimizationCandidate] = []
    seen: set[str] = set()
    for overlay in overlays:
        symbol = str(_get_attr(overlay, "symbol", ""))
        if symbol in seen:
            raise ValueError(f"Duplicate optimization overlay for symbol: {symbol!r}.")
        seen.add(symbol)
        candidates.append(
            build_candidate_from_overlay(
                overlay,
                risk_tensor=tensors.get(symbol),
                current_weight=float(weights.get(symbol, 0.0)),
                default_max_weight=default_max_weight,
                metadata=metadata,
            )
        )
    return candidates


def _allocate_greedy(
    eligible: Sequence[tuple[OptimizationCandidate, float]],
    *,
    config: PortfolioOptimizerConfig,
    gross_budget: float,
) -> tuple[dict[str, float], set[str], set[str]]:
    weights = {candidate.symbol: 0.0 for candidate, _score in eligible}
    sector_weights: dict[str, float] = {}
    remaining = [candidate for candidate, _score in eligible]
    score_by_symbol = {candidate.symbol: score for candidate, score in eligible}
    max_weight_capped: set[str] = set()
    sector_capped: set[str] = set()
    remaining_budget = gross_budget

    while remaining and remaining_budget > _EPSILON:
        total_score = sum(max(score_by_symbol[candidate.symbol], 0.0) for candidate in remaining)
        if total_score <= _EPSILON:
            break
        allocated_this_round = 0.0
        next_remaining: list[OptimizationCandidate] = []
        for candidate in remaining:
            desired = remaining_budget * max(score_by_symbol[candidate.symbol], 0.0) / total_score
            symbol_cap = min(config.max_weight, candidate.max_weight if candidate.max_weight is not None else config.max_weight)
            symbol_remaining = max(0.0, symbol_cap - weights[candidate.symbol])
            sector = candidate.sector or _UNKNOWN_SECTOR
            sector_remaining = math.inf
            if config.sector_cap is not None:
                sector_remaining = max(0.0, config.sector_cap - sector_weights.get(sector, 0.0))
            allowed = min(desired, symbol_remaining, sector_remaining, remaining_budget - allocated_this_round)
            if allowed > _EPSILON:
                weights[candidate.symbol] += allowed
                sector_weights[sector] = sector_weights.get(sector, 0.0) + allowed
                allocated_this_round += allowed
            if symbol_remaining <= desired + _EPSILON and symbol_remaining <= sector_remaining + _EPSILON:
                max_weight_capped.add(candidate.symbol)
                continue
            if sector_remaining <= desired + _EPSILON and sector_remaining <= symbol_remaining + _EPSILON:
                sector_capped.add(sector)
                continue
            if allowed > _EPSILON:
                next_remaining.append(candidate)
        remaining_budget -= allocated_this_round
        if allocated_this_round <= _EPSILON:
            break
        if len(next_remaining) == len(remaining) and remaining_budget <= _EPSILON:
            break
        remaining = next_remaining
    return _clean_weight_dict(weights), max_weight_capped, sector_capped


def optimize_portfolio(
    candidates: Sequence[OptimizationCandidate],
    *,
    config: PortfolioOptimizerConfig | None = None,
    market: str = "",
    as_of: str = "",
    current_weights: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizedPortfolioPlan:
    resolved_config = config or PortfolioOptimizerConfig()
    input_metadata = _coerce_metadata(metadata)
    resolved_current_weights = _finite_float_dict(current_weights, "current_weights") if current_weights is not None else {}
    if not resolved_current_weights:
        resolved_current_weights = {
            candidate.symbol: candidate.current_weight
            for candidate in candidates
            if abs(candidate.current_weight) > _EPSILON
        }

    violations: list[ConstraintViolation] = []
    blocked_symbols: list[str] = []
    rejected_symbols: list[str] = []
    eligible: list[tuple[OptimizationCandidate, float]] = []
    sector_by_symbol = {candidate.symbol: candidate.sector for candidate in candidates}

    for candidate in candidates:
        if candidate.is_blocked:
            blocked_symbols.append(candidate.symbol)
            rejected_symbols.append(candidate.symbol)
            violations.append(
                _make_violation(
                    symbol=candidate.symbol,
                    constraint_type=CONSTRAINT_BLOCKED_SYMBOL,
                    severity=VIOLATION_BLOCKER if resolved_config.force_exit_blocked_symbols else VIOLATION_INFO,
                    value=None,
                    limit=None,
                    message="Candidate blocked by execution or risk tensor constraints.",
                    metadata={"block_reasons": candidate.block_reasons},
                )
            )
            continue
        if candidate.edge_after_costs < resolved_config.min_edge_after_costs:
            rejected_symbols.append(candidate.symbol)
            violations.append(
                _make_violation(
                    symbol=candidate.symbol,
                    constraint_type=CONSTRAINT_MIN_EDGE,
                    severity=VIOLATION_INFO,
                    value=candidate.edge_after_costs,
                    limit=resolved_config.min_edge_after_costs,
                    message="Candidate edge after costs is below the configured minimum.",
                )
            )
            continue
        if resolved_config.max_risk_score is not None and candidate.risk_score > resolved_config.max_risk_score:
            rejected_symbols.append(candidate.symbol)
            violations.append(
                _make_violation(
                    symbol=candidate.symbol,
                    constraint_type=CONSTRAINT_RISK_SCORE,
                    severity=VIOLATION_WARNING,
                    value=candidate.risk_score,
                    limit=resolved_config.max_risk_score,
                    message="Candidate risk score is above the configured maximum.",
                )
            )
            continue
        score = _candidate_adjusted_score(candidate, resolved_config)
        eligible.append((candidate, score))

    eligible.sort(
        key=lambda item: (
            -item[1],
            -item[0].edge_after_costs,
            -item[0].action_score,
            item[0].symbol,
        )
    )
    if resolved_config.max_names is not None and len(eligible) > resolved_config.max_names:
        for candidate, _score in eligible[resolved_config.max_names :]:
            rejected_symbols.append(candidate.symbol)
            violations.append(
                _make_violation(
                    symbol=candidate.symbol,
                    constraint_type=CONSTRAINT_MAX_NAMES,
                    severity=VIOLATION_INFO,
                    value=float(len(eligible)),
                    limit=float(resolved_config.max_names),
                    message="Candidate excluded by max_names constraint.",
                )
            )
        eligible = eligible[: resolved_config.max_names]

    gross_budget = resolved_config.gross_exposure_cap - resolved_config.cash_buffer
    positive_scores = [score for _candidate, score in eligible if score > _EPSILON]
    forced_exit_symbols = {
        symbol
        for symbol in blocked_symbols
        if resolved_config.force_exit_blocked_symbols and abs(resolved_current_weights.get(symbol, 0.0)) > _EPSILON
    }

    raw_target_weights: dict[str, float] = {}
    max_weight_capped: set[str] = set()
    sector_capped: set[str] = set()
    if positive_scores:
        raw_target_weights, max_weight_capped, sector_capped = _allocate_greedy(
            [(candidate, score) for candidate, score in eligible if score > _EPSILON],
            config=resolved_config,
            gross_budget=gross_budget,
        )
    else:
        rejected_symbols.extend(candidate.symbol for candidate, _score in eligible)

    for symbol in max_weight_capped:
        limit = next(
            (
                min(resolved_config.max_weight, candidate.max_weight if candidate.max_weight is not None else resolved_config.max_weight)
                for candidate, _score in eligible
                if candidate.symbol == symbol
            ),
            resolved_config.max_weight,
        )
        violations.append(
            _make_violation(
                symbol=symbol,
                constraint_type=CONSTRAINT_MAX_WEIGHT,
                severity=VIOLATION_WARNING,
                value=raw_target_weights.get(symbol),
                limit=limit,
                message="Candidate allocation was clipped by max weight.",
            )
        )
    for sector in sector_capped:
        violations.append(
            _make_violation(
                symbol=None,
                constraint_type=CONSTRAINT_SECTOR_CAP,
                severity=VIOLATION_WARNING,
                value=None,
                limit=resolved_config.sector_cap,
                message=f"Sector allocation was clipped by cap: {sector}.",
                metadata={"sector": sector},
            )
        )

    proposed_targets = dict(raw_target_weights)
    for symbol in forced_exit_symbols:
        proposed_targets[symbol] = 0.0

    proposed_turnover = estimate_turnover(resolved_current_weights, proposed_targets)
    final_targets = dict(proposed_targets)
    if resolved_config.turnover_cap is not None and proposed_turnover > resolved_config.turnover_cap + _EPSILON:
        non_forced_symbols = sorted((set(resolved_current_weights) | set(proposed_targets)) - forced_exit_symbols)
        forced_turnover = sum(abs(resolved_current_weights.get(symbol, 0.0)) for symbol in forced_exit_symbols)
        non_forced_turnover = sum(
            abs(proposed_targets.get(symbol, 0.0) - resolved_current_weights.get(symbol, 0.0))
            for symbol in non_forced_symbols
        )
        available_turnover = max(0.0, resolved_config.turnover_cap - forced_turnover)
        scale = 0.0 if non_forced_turnover <= _EPSILON else min(1.0, available_turnover / non_forced_turnover)
        final_targets = {}
        for symbol in non_forced_symbols:
            current = resolved_current_weights.get(symbol, 0.0)
            proposed = proposed_targets.get(symbol, 0.0)
            target = current + (proposed - current) * scale
            if abs(target) > _EPSILON:
                final_targets[symbol] = target
                if symbol not in sector_by_symbol:
                    sector_by_symbol[symbol] = None
        for symbol in forced_exit_symbols:
            final_targets[symbol] = 0.0
        severity = VIOLATION_WARNING
        final_turnover_for_violation = estimate_turnover(resolved_current_weights, final_targets)
        if final_turnover_for_violation > resolved_config.turnover_cap + _EPSILON:
            severity = VIOLATION_BLOCKER
        violations.append(
            _make_violation(
                symbol=None,
                constraint_type=CONSTRAINT_TURNOVER_CAP,
                severity=severity,
                value=proposed_turnover,
                limit=resolved_config.turnover_cap,
                message="Proposed turnover exceeded the configured turnover cap and was scaled toward current weights.",
                metadata={
                    "scale": scale,
                    "final_turnover": final_turnover_for_violation,
                    "forced_exit_symbols": sorted(forced_exit_symbols),
                },
            )
        )

    clean_targets = _clean_weight_dict(final_targets)
    trade_weights = {
        symbol: final_targets.get(symbol, 0.0) - resolved_current_weights.get(symbol, 0.0)
        for symbol in sorted(set(resolved_current_weights) | set(final_targets))
        if abs(final_targets.get(symbol, 0.0) - resolved_current_weights.get(symbol, 0.0)) > _EPSILON
    }
    turnover_estimate = estimate_turnover(resolved_current_weights, clean_targets)
    gross_exposure = sum(abs(weight) for weight in clean_targets.values())
    net_exposure = sum(clean_targets.values())
    long_exposure = sum(weight for weight in clean_targets.values() if weight > 0.0)
    if gross_exposure > resolved_config.gross_exposure_cap + _EPSILON:
        violations.append(
            _make_violation(
                symbol=None,
                constraint_type=CONSTRAINT_GROSS_EXPOSURE,
                severity=VIOLATION_BLOCKER,
                value=gross_exposure,
                limit=resolved_config.gross_exposure_cap,
                message="Final gross exposure exceeds configured cap.",
            )
        )
    sector_weights = compute_sector_weights(clean_targets, sector_by_symbol)
    if resolved_config.sector_cap is not None:
        for sector, weight in sector_weights.items():
            if weight > resolved_config.sector_cap + _EPSILON:
                violations.append(
                    _make_violation(
                        symbol=None,
                        constraint_type=CONSTRAINT_SECTOR_CAP,
                        severity=VIOLATION_BLOCKER,
                        value=weight,
                        limit=resolved_config.sector_cap,
                        message=f"Final sector exposure exceeds cap: {sector}.",
                        metadata={"sector": sector},
                    )
                )

    score_by_symbol = {candidate.symbol: score for candidate, score in eligible}
    objective_value = sum(clean_targets.get(symbol, 0.0) * score_by_symbol.get(symbol, 0.0) for symbol in clean_targets)
    selected_symbols = sorted(clean_targets, key=lambda symbol: (-clean_targets[symbol], symbol))
    if candidates and len(blocked_symbols) == len(candidates):
        status = PLAN_STATUS_INFEASIBLE
    elif clean_targets:
        status = PLAN_STATUS_OPTIMIZED
    elif forced_exit_symbols:
        status = PLAN_STATUS_INFEASIBLE
    else:
        status = PLAN_STATUS_EMPTY

    config_snapshot = resolved_config.to_dict()
    config_hash = _short_hash([config_snapshot])
    plan_id = make_plan_id(market=market, as_of=as_of, symbols=selected_symbols, config_hash=config_hash)
    plan_metadata = {
        "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        "config": config_snapshot,
        "optimization_method": "deterministic_greedy_v1",
        "eligible_count": len(eligible),
        "rejected_count": len(_ordered_unique(rejected_symbols)),
        "input_metadata": input_metadata,
    }
    return OptimizedPortfolioPlan(
        schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        plan_id=plan_id,
        as_of=as_of,
        market=market,
        status=status,
        objective_value=objective_value,
        target_weights=clean_targets,
        current_weights=resolved_current_weights,
        trade_weights=trade_weights,
        selected_symbols=selected_symbols,
        blocked_symbols=blocked_symbols,
        rejected_symbols=rejected_symbols,
        cash_weight=max(0.0, 1.0 - gross_exposure),
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        long_exposure=long_exposure,
        turnover_estimate=turnover_estimate,
        sector_weights=sector_weights,
        violations=violations,
        candidate_count=len(candidates),
        metadata=plan_metadata,
    )


def evaluate_rebalance(
    plan: OptimizedPortfolioPlan,
    *,
    evaluation_date: str,
    forward_returns: Mapping[str, float],
    benchmark_return: float | None = None,
    transaction_cost_bps: float | None = None,
    slippage_bps: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RebalanceResult:
    supplied_returns = _finite_float_dict(forward_returns, "forward_returns")
    missing_return_symbols = sorted(symbol for symbol in plan.target_weights if symbol not in supplied_returns)
    realized_gross_return = sum(
        plan.target_weights[symbol] * supplied_returns[symbol]
        for symbol in sorted(plan.target_weights)
        if symbol in supplied_returns
    )
    config_payload = plan.metadata.get("config", {}) if isinstance(plan.metadata.get("config"), Mapping) else {}
    resolved_transaction_cost_bps = (
        _non_negative_float(transaction_cost_bps, "transaction_cost_bps")
        if transaction_cost_bps is not None
        else _non_negative_float(config_payload.get("transaction_cost_bps", 0.0), "transaction_cost_bps")
    )
    resolved_slippage_bps = (
        _non_negative_float(slippage_bps, "slippage_bps")
        if slippage_bps is not None
        else _non_negative_float(config_payload.get("slippage_bps", 0.0), "slippage_bps")
    )
    estimated_cost_return = plan.turnover_estimate * bps_to_decimal_return(
        resolved_transaction_cost_bps + resolved_slippage_bps
    )
    realized_net_return = realized_gross_return - estimated_cost_return
    resolved_benchmark = _float_or_none(benchmark_return, "benchmark_return")
    excess_return = None if resolved_benchmark is None else realized_net_return - resolved_benchmark
    result_metadata = {
        "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        "input_metadata": _coerce_metadata(metadata),
        "cost_bps": {
            "transaction_cost_bps": resolved_transaction_cost_bps,
            "slippage_bps": resolved_slippage_bps,
        },
    }
    return RebalanceResult(
        schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        rebalance_id=make_rebalance_id(plan_id=plan.plan_id, evaluation_date=evaluation_date),
        as_of=plan.as_of,
        evaluation_date=evaluation_date,
        market=plan.market,
        plan_id=plan.plan_id,
        target_weights=plan.target_weights,
        current_weights=plan.current_weights,
        realized_gross_return=realized_gross_return,
        estimated_cost_return=estimated_cost_return,
        realized_net_return=realized_net_return,
        benchmark_return=resolved_benchmark,
        excess_return=excess_return,
        turnover_estimate=plan.turnover_estimate,
        selected_symbols=plan.selected_symbols,
        missing_return_symbols=missing_return_symbols,
        metadata=result_metadata,
    )


def run_walk_forward_loop(
    rebalance_inputs: Sequence[RebalanceInput],
    *,
    config: PortfolioOptimizerConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> WalkForwardResult:
    if not rebalance_inputs:
        return WalkForwardResult(
            schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
            run_id=make_walk_forward_run_id(market="", start_date="", end_date="", rebalance_dates=[]),
            market="",
            start_date="",
            end_date="",
            rebalance_count=0,
            cumulative_gross_return=0.0,
            cumulative_net_return=0.0,
            cumulative_benchmark_return=None,
            cumulative_excess_return=None,
            annualized_net_return=None,
            max_drawdown=0.0,
            average_turnover=0.0,
            total_estimated_cost_return=0.0,
            plans=[],
            rebalance_results=[],
            metadata={
                "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
                "input_metadata": _coerce_metadata(metadata),
            },
        )

    resolved_inputs = sorted(
        [
            rebalance_input if isinstance(rebalance_input, RebalanceInput) else RebalanceInput.from_dict(rebalance_input)
            for rebalance_input in rebalance_inputs
        ],
        key=lambda item: (item.as_of, item.evaluation_date, item.market),
    )
    plans: list[OptimizedPortfolioPlan] = []
    results: list[RebalanceResult] = []
    rolling_current_weights: dict[str, float] = {}
    resolved_config = config or PortfolioOptimizerConfig()

    for rebalance_input in resolved_inputs:
        current_weights_for_period = (
            dict(rebalance_input.current_weights)
            if rebalance_input.current_weights
            else dict(rolling_current_weights)
        )
        plan = optimize_portfolio(
            rebalance_input.candidates,
            config=resolved_config,
            market=rebalance_input.market,
            as_of=rebalance_input.as_of,
            current_weights=current_weights_for_period,
            metadata=rebalance_input.metadata,
        )
        result = evaluate_rebalance(
            plan,
            evaluation_date=rebalance_input.evaluation_date,
            forward_returns=rebalance_input.forward_returns,
            benchmark_return=rebalance_input.benchmark_return,
        )
        plans.append(plan)
        results.append(result)
        rolling_current_weights = dict(plan.target_weights)

    gross_returns = [result.realized_gross_return for result in results]
    net_returns = [result.realized_net_return for result in results]
    benchmark_returns = [result.benchmark_return for result in results]
    cumulative_benchmark_return = (
        compound_returns([float(value) for value in benchmark_returns])
        if all(value is not None for value in benchmark_returns)
        else None
    )
    cumulative_net_return = compound_returns(net_returns)
    cumulative_excess_return = (
        None if cumulative_benchmark_return is None else cumulative_net_return - cumulative_benchmark_return
    )
    annualized_net_return = None
    if len(results) > 0 and cumulative_net_return > -1.0:
        annualized_net_return = (1.0 + cumulative_net_return) ** (252.0 / len(results)) - 1.0
    market = resolved_inputs[0].market
    start_date = resolved_inputs[0].as_of
    end_date = resolved_inputs[-1].evaluation_date or resolved_inputs[-1].as_of
    rebalance_dates = [rebalance_input.as_of for rebalance_input in resolved_inputs]
    return WalkForwardResult(
        schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        run_id=make_walk_forward_run_id(
            market=market,
            start_date=start_date,
            end_date=end_date,
            rebalance_dates=rebalance_dates,
        ),
        market=market,
        start_date=start_date,
        end_date=end_date,
        rebalance_count=len(results),
        cumulative_gross_return=compound_returns(gross_returns),
        cumulative_net_return=cumulative_net_return,
        cumulative_benchmark_return=cumulative_benchmark_return,
        cumulative_excess_return=cumulative_excess_return,
        annualized_net_return=annualized_net_return,
        max_drawdown=max_drawdown_from_returns(net_returns),
        average_turnover=sum(result.turnover_estimate for result in results) / len(results),
        total_estimated_cost_return=sum(result.estimated_cost_return for result in results),
        plans=plans,
        rebalance_results=results,
        metadata={
            "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
            "input_metadata": _coerce_metadata(metadata),
            "config": resolved_config.to_dict(),
        },
    )


def build_portfolio_constructor_patch(plan: OptimizedPortfolioPlan) -> dict[str, Any]:
    optimization_method = None
    if isinstance(plan.metadata, Mapping):
        optimization_method = plan.metadata.get("optimization_method")
    return {
        "target_weights": dict(sorted(plan.target_weights.items())),
        "blocked_symbols": list(plan.blocked_symbols),
        "rejected_symbols": list(plan.rejected_symbols),
        "turnover_estimate": plan.turnover_estimate,
        "sector_weights": dict(sorted(plan.sector_weights.items())),
        "gross_exposure": plan.gross_exposure,
        "net_exposure": plan.net_exposure,
        "violations": [violation.to_dict() for violation in plan.violations],
        "metadata": {
            "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
            "plan_id": plan.plan_id,
            "status": plan.status,
            "optimization_method": optimization_method,
        },
    }


class PortfolioOptimizerStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_PORTFOLIO_OPTIMIZER_DIR
        self.optimized_plans_path = self.root_dir / DEFAULT_OPTIMIZED_PLANS_FILENAME
        self.rebalance_results_path = self.root_dir / DEFAULT_REBALANCE_RESULTS_FILENAME
        self.walk_forward_results_path = self.root_dir / DEFAULT_WALK_FORWARD_RESULTS_FILENAME

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path} at line {line_number}.") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Malformed JSON in {path} at line {line_number}: expected object.")
                rows.append(payload)
        return rows

    def append_plan(self, plan: OptimizedPortfolioPlan) -> None:
        if plan.plan_id in self.get_plan_ids():
            raise ValueError(f"Duplicate plan_id: {plan.plan_id}")
        self._append_jsonl(self.optimized_plans_path, plan.to_dict())

    def append_rebalance_result(self, result: RebalanceResult) -> None:
        if result.rebalance_id in self.get_rebalance_ids():
            raise ValueError(f"Duplicate rebalance_id: {result.rebalance_id}")
        self._append_jsonl(self.rebalance_results_path, result.to_dict())

    def append_walk_forward_result(self, result: WalkForwardResult) -> None:
        if result.run_id in self.get_walk_forward_run_ids():
            raise ValueError(f"Duplicate run_id: {result.run_id}")
        self._append_jsonl(self.walk_forward_results_path, result.to_dict())

    def read_plans(self) -> list[OptimizedPortfolioPlan]:
        return [OptimizedPortfolioPlan.from_dict(row) for row in self._read_jsonl(self.optimized_plans_path)]

    def read_rebalance_results(self) -> list[RebalanceResult]:
        return [RebalanceResult.from_dict(row) for row in self._read_jsonl(self.rebalance_results_path)]

    def read_walk_forward_results(self) -> list[WalkForwardResult]:
        return [WalkForwardResult.from_dict(row) for row in self._read_jsonl(self.walk_forward_results_path)]

    def get_plan_ids(self) -> set[str]:
        return {plan.plan_id for plan in self.read_plans()}

    def get_rebalance_ids(self) -> set[str]:
        return {result.rebalance_id for result in self.read_rebalance_results()}

    def get_walk_forward_run_ids(self) -> set[str]:
        return {result.run_id for result in self.read_walk_forward_results()}


__all__ = [
    "DEFAULT_PORTFOLIO_OPTIMIZER_DIR",
    "DEFAULT_OPTIMIZED_PLANS_FILENAME",
    "DEFAULT_REBALANCE_RESULTS_FILENAME",
    "DEFAULT_WALK_FORWARD_RESULTS_FILENAME",
    "PLAN_STATUS_OPTIMIZED",
    "PLAN_STATUS_EMPTY",
    "PLAN_STATUS_INFEASIBLE",
    "CONSTRAINT_BLOCKED_SYMBOL",
    "CONSTRAINT_MIN_EDGE",
    "CONSTRAINT_MAX_WEIGHT",
    "CONSTRAINT_GROSS_EXPOSURE",
    "CONSTRAINT_SECTOR_CAP",
    "CONSTRAINT_TURNOVER_CAP",
    "CONSTRAINT_MAX_NAMES",
    "CONSTRAINT_RISK_SCORE",
    "VIOLATION_INFO",
    "VIOLATION_WARNING",
    "VIOLATION_BLOCKER",
    "PortfolioOptimizerConfig",
    "OptimizationCandidate",
    "ConstraintViolation",
    "OptimizedPortfolioPlan",
    "RebalanceInput",
    "RebalanceResult",
    "WalkForwardResult",
    "PortfolioOptimizerStore",
    "make_constraint_violation_id",
    "make_plan_id",
    "make_rebalance_id",
    "make_walk_forward_run_id",
    "clamp_unit_interval",
    "validate_finite_number",
    "bps_to_decimal_return",
    "estimate_turnover",
    "compute_sector_weights",
    "compound_returns",
    "max_drawdown_from_returns",
    "build_candidate_from_overlay",
    "build_candidates_from_overlays",
    "optimize_portfolio",
    "evaluate_rebalance",
    "run_walk_forward_loop",
    "build_portfolio_constructor_patch",
]
