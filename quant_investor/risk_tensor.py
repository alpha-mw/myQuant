"""Offline structured risk tensor builders and store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from quant_investor.risk_tensor_types import (
    DEFAULT_EXECUTION_REPORTS_FILENAME,
    DEFAULT_PORTFOLIO_TENSORS_FILENAME,
    DEFAULT_RISK_TENSOR_DIR,
    DEFAULT_SYMBOL_TENSORS_FILENAME,
    EXECUTION_BLOCKED,
    EXECUTION_FEASIBLE,
    EXECUTION_PARTIALLY_FEASIBLE,
    ExecutionFeasibility,
    ExecutionFeasibilityReport,
    LiquidityProfile,
    PortfolioRiskTensor,
    RISK_ISSUE_ADV_CAP_EXCEEDED,
    RISK_ISSUE_BETA_EXPOSURE,
    RISK_ISSUE_CORRELATION_CLUSTER,
    RISK_ISSUE_DATA_QUARANTINE,
    RISK_ISSUE_LOW_LIQUIDITY,
    RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED,
    RISK_ISSUE_POSITION_TOO_LARGE,
    RISK_ISSUE_SECTOR_CONCENTRATION,
    RISK_ISSUE_STRESS_LOSS,
    RISK_ISSUE_STYLE_EXPOSURE,
    RISK_ISSUE_TURNOVER_EXCEEDED,
    RISK_ISSUE_UNTRADABLE,
    RISK_SEVERITY_BLOCKER,
    RISK_SEVERITY_INFO,
    RISK_SEVERITY_WARNING,
    RISK_TENSOR_SCHEMA_VERSION,
    RiskIssue,
    StressScenarioResult,
    SymbolExposure,
    SymbolRiskTensor,
    _EXECUTION_REASON_ORDER,
    _coerce_metadata,
    _ensure_json_serializable,
    _finite_float,
    _json_safe,
    _non_negative_float_or_none,
    _ordered_unique,
    _sort_issues,
    bps_to_decimal_return,
    clamp_unit_interval,
    make_execution_report_id,
    make_portfolio_tensor_id,
    make_risk_issue_id,
    make_symbol_tensor_id,
    validate_finite_number,
    weighted_average,
)


def _make_issue(
    *,
    symbol: str | None,
    market: str | None,
    as_of: str,
    issue_type: str,
    severity: str,
    message: str,
    value: float | None = None,
    limit: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RiskIssue:
    return RiskIssue(
        issue_id=make_risk_issue_id(
            symbol=symbol,
            market=market,
            as_of=as_of,
            issue_type=issue_type,
            message=message,
        ),
        symbol=symbol,
        market=market,
        as_of=as_of,
        issue_type=issue_type,
        severity=severity,
        message=message,
        value=value,
        limit=limit,
        metadata=_coerce_metadata(metadata),
    )


def _extract_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _issue_severity(value: Any) -> str | None:
    if isinstance(value, Mapping):
        severity = value.get("severity")
    else:
        severity = getattr(value, "severity", None)
    if severity is None:
        return None
    return str(severity)


def _get_optional_float_attr(source: Any, name: str) -> float | None:
    value = getattr(source, name, None)
    if value is None:
        return None
    return _finite_float(value, name)


def build_execution_feasibility(
    *,
    symbol: str,
    market: str,
    as_of: str,
    target_weight: float,
    current_weight: float = 0.0,
    portfolio_value: float | None = None,
    liquidity: LiquidityProfile | None = None,
    is_tradable: bool = True,
    tradability_reasons: Sequence[str] | None = None,
    max_weight: float | None = None,
    max_trade_value: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionFeasibility:
    resolved_target_weight = _finite_float(target_weight, "target_weight")
    resolved_current_weight = _finite_float(current_weight, "current_weight")
    resolved_portfolio_value = _non_negative_float_or_none(portfolio_value, "portfolio_value")
    resolved_max_weight = _non_negative_float_or_none(max_weight, "max_weight")
    resolved_max_trade_value = _non_negative_float_or_none(max_trade_value, "max_trade_value")
    resolved_liquidity = liquidity or LiquidityProfile(symbol=symbol, market=market, as_of=as_of)

    requested_trade_value = None
    if resolved_portfolio_value is not None:
        requested_trade_value = abs(resolved_target_weight - resolved_current_weight) * resolved_portfolio_value

    adv_allowed = None
    if (
        resolved_liquidity.adv is not None
        and resolved_liquidity.max_participation_rate is not None
    ):
        adv_allowed = resolved_liquidity.adv * resolved_liquidity.max_participation_rate

    candidates = [
        value
        for value in [
            resolved_liquidity.max_order_value,
            adv_allowed,
            resolved_max_trade_value,
        ]
        if value is not None
    ]
    allowed_trade_value = min(candidates) if candidates else None

    adv_usage = None
    if requested_trade_value is not None and resolved_liquidity.adv and resolved_liquidity.adv > 0.0:
        adv_usage = requested_trade_value / resolved_liquidity.adv

    max_order_value_usage = None
    if (
        requested_trade_value is not None
        and resolved_liquidity.max_order_value is not None
        and resolved_liquidity.max_order_value > 0.0
    ):
        max_order_value_usage = requested_trade_value / resolved_liquidity.max_order_value

    blocked_reasons: list[str] = []
    warning_reasons: list[str] = []

    if not is_tradable:
        blocked_reasons.append(RISK_ISSUE_UNTRADABLE)

    if resolved_max_weight is not None and abs(resolved_target_weight) > resolved_max_weight:
        blocked_reasons.append(RISK_ISSUE_POSITION_TOO_LARGE)

    if requested_trade_value is not None:
        if adv_allowed is not None and requested_trade_value > adv_allowed:
            warning_reasons.append(RISK_ISSUE_ADV_CAP_EXCEEDED)
        if (
            resolved_liquidity.max_order_value is not None
            and requested_trade_value > resolved_liquidity.max_order_value
        ):
            warning_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)
        if resolved_max_trade_value is not None and requested_trade_value > resolved_max_trade_value:
            warning_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)
        if allowed_trade_value is not None and requested_trade_value > allowed_trade_value:
            if allowed_trade_value <= 0.0:
                if RISK_ISSUE_ADV_CAP_EXCEEDED in warning_reasons:
                    blocked_reasons.append(RISK_ISSUE_ADV_CAP_EXCEEDED)
                if RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED in warning_reasons:
                    blocked_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)
                if not blocked_reasons:
                    blocked_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)
            else:
                if not any(
                    reason in warning_reasons
                    for reason in [
                        RISK_ISSUE_ADV_CAP_EXCEEDED,
                        RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED,
                    ]
                ):
                    warning_reasons.append(RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED)

    if resolved_liquidity.liquidity_score is not None and resolved_liquidity.liquidity_score < 0.20:
        warning_reasons.append(RISK_ISSUE_LOW_LIQUIDITY)

    extra_tradability_reasons = [
        reason
        for reason in _extract_reasons(tradability_reasons)
        if reason not in _EXECUTION_REASON_ORDER
    ]
    if not is_tradable:
        blocked_reasons.extend(extra_tradability_reasons)
    else:
        warning_reasons.extend(extra_tradability_reasons)

    blocked_reasons = _ordered_unique(blocked_reasons, preferred_order=_EXECUTION_REASON_ORDER)
    warning_reasons = _ordered_unique(warning_reasons, preferred_order=_EXECUTION_REASON_ORDER)
    warning_reasons = [reason for reason in warning_reasons if reason not in blocked_reasons]

    if blocked_reasons:
        status = EXECUTION_BLOCKED
    elif (
        requested_trade_value is not None
        and allowed_trade_value is not None
        and requested_trade_value > allowed_trade_value
        and allowed_trade_value > 0.0
    ):
        status = EXECUTION_PARTIALLY_FEASIBLE
    else:
        status = EXECUTION_FEASIBLE

    estimated_slippage_bps = None
    if resolved_liquidity.estimated_spread_bps is not None:
        estimated_slippage_bps = resolved_liquidity.estimated_spread_bps / 2.0
    estimated_market_impact_bps = resolved_liquidity.estimated_market_impact_bps
    cost_parts = [
        value
        for value in [estimated_slippage_bps, estimated_market_impact_bps]
        if value is not None
    ]
    estimated_transaction_cost_bps = sum(cost_parts) if cost_parts else None

    execution_metadata = _coerce_metadata(metadata)
    execution_metadata.setdefault("risk_tensor_schema_version", RISK_TENSOR_SCHEMA_VERSION)
    execution_metadata.setdefault("allowed_trade_value_sources", {
        "max_order_value": resolved_liquidity.max_order_value,
        "adv_participation": adv_allowed,
        "max_trade_value": resolved_max_trade_value,
    })

    return ExecutionFeasibility(
        symbol=symbol,
        market=market,
        as_of=as_of,
        status=status,
        requested_weight=resolved_target_weight,
        current_weight=resolved_current_weight,
        requested_trade_value=requested_trade_value,
        allowed_trade_value=allowed_trade_value,
        adv_usage=adv_usage,
        max_order_value_usage=max_order_value_usage,
        blocked_reasons=blocked_reasons,
        warning_reasons=warning_reasons,
        estimated_transaction_cost_bps=estimated_transaction_cost_bps,
        estimated_slippage_bps=estimated_slippage_bps,
        estimated_market_impact_bps=estimated_market_impact_bps,
        metadata=execution_metadata,
    )


def build_symbol_risk_tensor(
    *,
    symbol: str,
    market: str,
    as_of: str,
    latest_trade_date: str,
    target_weight: float,
    current_weight: float = 0.0,
    data_quality_assessment: Any | None = None,
    tradability_status: Any | None = None,
    exposure: SymbolExposure | None = None,
    liquidity: LiquidityProfile | None = None,
    portfolio_value: float | None = None,
    max_weight: float | None = None,
    max_trade_value: float | None = None,
    stress_shocks: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SymbolRiskTensor:
    resolved_target_weight = _finite_float(target_weight, "target_weight")
    resolved_current_weight = _finite_float(current_weight, "current_weight")
    assessment = data_quality_assessment
    status_source = tradability_status

    data_quality_score = getattr(assessment, "data_quality_score", None)
    if data_quality_score is not None:
        data_quality_score = clamp_unit_interval(float(data_quality_score))

    assessment_issues = list(getattr(assessment, "issues", []) or [])
    has_data_blocker = any(_issue_severity(issue) == RISK_SEVERITY_BLOCKER for issue in assessment_issues)
    quarantine = bool(getattr(assessment, "quarantine", False)) or has_data_blocker
    is_researchable = bool(getattr(assessment, "is_researchable", True)) and not quarantine
    if status_source is not None and hasattr(status_source, "is_tradable"):
        is_tradable = bool(getattr(status_source, "is_tradable"))
    elif assessment is not None and hasattr(assessment, "is_tradable"):
        is_tradable = bool(getattr(assessment, "is_tradable"))
    else:
        is_tradable = True

    tradability_reasons = _extract_reasons(getattr(status_source, "reasons", None))
    if not tradability_reasons:
        tradability_reasons = _extract_reasons(getattr(assessment, "tradability_reasons", None))
    tradability_reasons = _ordered_unique(tradability_reasons, preferred_order=_EXECUTION_REASON_ORDER)

    resolved_exposure = exposure or SymbolExposure(
        symbol=symbol,
        market=market,
        as_of=as_of,
    )
    if liquidity is not None:
        resolved_liquidity = liquidity
    else:
        resolved_liquidity = LiquidityProfile(
            symbol=symbol,
            market=market,
            as_of=as_of,
            adv=_get_optional_float_attr(status_source, "adv"),
            liquidity_score=_get_optional_float_attr(status_source, "liquidity_score"),
            max_order_value=_get_optional_float_attr(status_source, "max_order_value"),
        )

    execution = build_execution_feasibility(
        symbol=symbol,
        market=market,
        as_of=as_of,
        target_weight=resolved_target_weight,
        current_weight=resolved_current_weight,
        portfolio_value=portfolio_value,
        liquidity=resolved_liquidity,
        is_tradable=is_tradable,
        tradability_reasons=tradability_reasons,
        max_weight=max_weight,
        max_trade_value=max_trade_value,
        metadata={"builder": "build_symbol_risk_tensor"},
    )

    issues: list[RiskIssue] = []
    if quarantine:
        quarantine_reasons = _extract_reasons(getattr(assessment, "quarantine_reasons", None))
        message = "Symbol is quarantined by data quality assessment."
        if quarantine_reasons:
            message = f"{message} Reasons: {'; '.join(quarantine_reasons)}"
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_DATA_QUARANTINE,
                severity=RISK_SEVERITY_BLOCKER,
                message=message,
                value=data_quality_score,
                limit=None,
                metadata={"quarantine_reasons": quarantine_reasons},
            )
        )

    if not is_tradable:
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_UNTRADABLE,
                severity=RISK_SEVERITY_BLOCKER,
                message="Symbol is not tradable under supplied tradability status.",
                metadata={"tradability_reasons": tradability_reasons},
            )
        )

    if resolved_liquidity.liquidity_score is not None and resolved_liquidity.liquidity_score < 0.20:
        severity = (
            RISK_SEVERITY_BLOCKER
            if resolved_liquidity.liquidity_score <= 0.0
            else RISK_SEVERITY_WARNING
        )
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_LOW_LIQUIDITY,
                severity=severity,
                message="Liquidity score is below the Phase 6 diagnostic threshold.",
                value=resolved_liquidity.liquidity_score,
                limit=0.20,
            )
        )

    execution_issue_map = {
        RISK_ISSUE_POSITION_TOO_LARGE,
        RISK_ISSUE_ADV_CAP_EXCEEDED,
        RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED,
    }
    for reason in execution.blocked_reasons:
        if reason not in execution_issue_map:
            continue
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=reason,
                severity=RISK_SEVERITY_BLOCKER,
                message=f"Execution feasibility blocked by {reason}.",
                value=execution.requested_trade_value,
                limit=execution.allowed_trade_value,
            )
        )
    for reason in execution.warning_reasons:
        if reason not in execution_issue_map:
            continue
        issues.append(
            _make_issue(
                symbol=symbol,
                market=market,
                as_of=as_of,
                issue_type=reason,
                severity=RISK_SEVERITY_WARNING,
                message=f"Execution feasibility warning: {reason}.",
                value=execution.requested_trade_value,
                limit=execution.allowed_trade_value,
            )
        )

    stress_results: list[StressScenarioResult] = []
    for scenario_name in sorted(dict(stress_shocks or {})):
        shock_return = _finite_float(dict(stress_shocks or {})[scenario_name], f"stress_shocks.{scenario_name}")
        estimated_loss = max(0.0, -shock_return * resolved_target_weight)
        stress_results.append(
            StressScenarioResult(
                scenario_name=str(scenario_name),
                symbol=symbol,
                market=market,
                as_of=as_of,
                shock_return=shock_return,
                position_weight=resolved_target_weight,
                estimated_loss=estimated_loss,
                metadata={"loss_convention": "positive_loss"},
            )
        )

    sorted_issues = _sort_issues(issues)
    severity_penalty = 0.0
    for issue in sorted_issues:
        if issue.severity == RISK_SEVERITY_BLOCKER:
            severity_penalty += 0.50
        elif issue.severity == RISK_SEVERITY_WARNING:
            severity_penalty += 0.20
        else:
            severity_penalty += 0.05
    max_stress_loss = max([result.estimated_loss for result in stress_results], default=0.0)
    risk_score = clamp_unit_interval(severity_penalty + min(0.50, max_stress_loss))

    tensor_metadata = _coerce_metadata(metadata)
    tensor_metadata.setdefault("risk_tensor_schema_version", RISK_TENSOR_SCHEMA_VERSION)
    if data_quality_score is not None:
        tensor_metadata.setdefault("data_quality_score_source", "data_quality_assessment")

    return SymbolRiskTensor(
        tensor_id=make_symbol_tensor_id(
            symbol=symbol,
            market=market,
            as_of=as_of,
            latest_trade_date=latest_trade_date,
        ),
        symbol=symbol,
        market=market,
        as_of=as_of,
        latest_trade_date=latest_trade_date,
        target_weight=resolved_target_weight,
        current_weight=resolved_current_weight,
        data_quality_score=data_quality_score,
        is_researchable=is_researchable,
        is_tradable=is_tradable,
        quarantine=quarantine,
        exposure=resolved_exposure,
        liquidity=resolved_liquidity,
        execution=execution,
        stress_results=stress_results,
        issues=sorted_issues,
        risk_score=risk_score,
        metadata=tensor_metadata,
    )


def _add_weight(target: dict[str, float], key: str, value: float) -> None:
    target[key] = target.get(key, 0.0) + value


def _aggregate_exposures(
    tensors: Sequence[SymbolRiskTensor],
    exposure_name: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for tensor in tensors:
        exposures = getattr(tensor.exposure, exposure_name)
        for name, exposure_value in exposures.items():
            _add_weight(result, name, tensor.target_weight * exposure_value)
    return {key: result[key] for key in sorted(result)}


def build_portfolio_risk_tensor(
    *,
    symbol_tensors: Sequence[SymbolRiskTensor],
    market: str,
    as_of: str,
    turnover_estimate: float = 0.0,
    sector_cap: float | None = None,
    gross_exposure_cap: float | None = None,
    max_weight: float | None = None,
    turnover_cap: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PortfolioRiskTensor:
    ordered_tensors = sorted(list(symbol_tensors), key=lambda tensor: (tensor.symbol, tensor.market))
    resolved_turnover = _non_negative_float_or_none(turnover_estimate, "turnover_estimate") or 0.0
    resolved_sector_cap = _non_negative_float_or_none(sector_cap, "sector_cap")
    resolved_gross_cap = _non_negative_float_or_none(gross_exposure_cap, "gross_exposure_cap")
    resolved_max_weight = _non_negative_float_or_none(max_weight, "max_weight")
    resolved_turnover_cap = _non_negative_float_or_none(turnover_cap, "turnover_cap")

    weights = [tensor.target_weight for tensor in ordered_tensors]
    gross_exposure = sum(abs(weight) for weight in weights)
    net_exposure = sum(weights)
    long_exposure = sum(max(weight, 0.0) for weight in weights)
    short_exposure = sum(min(weight, 0.0) for weight in weights)

    sector_weights: dict[str, float] = {}
    correlation_cluster_weights: dict[str, float] = {}
    for tensor in ordered_tensors:
        sector = tensor.exposure.sector or "UNKNOWN"
        _add_weight(sector_weights, sector, tensor.target_weight)
        cluster = tensor.exposure.correlation_cluster or "UNKNOWN"
        _add_weight(correlation_cluster_weights, cluster, abs(tensor.target_weight))
    sector_weights = {key: sector_weights[key] for key in sorted(sector_weights)}
    correlation_cluster_weights = {
        key: correlation_cluster_weights[key]
        for key in sorted(correlation_cluster_weights)
    }

    style_exposures = _aggregate_exposures(ordered_tensors, "style_exposures")
    factor_exposures = _aggregate_exposures(ordered_tensors, "factor_exposures")

    beta_values = [
        tensor.target_weight * tensor.exposure.beta
        for tensor in ordered_tensors
        if tensor.exposure.beta is not None
    ]
    beta_exposure = sum(beta_values) if beta_values else None

    blocked_symbols = _ordered_unique(
        [
            tensor.symbol
            for tensor in ordered_tensors
            if tensor.execution.status == EXECUTION_BLOCKED
            or any(issue.severity == RISK_SEVERITY_BLOCKER for issue in tensor.issues)
        ]
    )
    blocked_set = set(blocked_symbols)
    max_weight_by_symbol = {
        tensor.symbol: (
            0.0
            if tensor.symbol in blocked_set
            else resolved_max_weight
            if resolved_max_weight is not None
            else abs(tensor.target_weight)
        )
        for tensor in ordered_tensors
    }

    portfolio_issues: list[RiskIssue] = []
    if resolved_gross_cap is not None and gross_exposure > resolved_gross_cap:
        portfolio_issues.append(
            _make_issue(
                symbol=None,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_POSITION_TOO_LARGE,
                severity=RISK_SEVERITY_BLOCKER,
                message="Gross exposure exceeds configured Phase 6 cap.",
                value=gross_exposure,
                limit=resolved_gross_cap,
            )
        )
    if resolved_turnover_cap is not None and resolved_turnover > resolved_turnover_cap:
        portfolio_issues.append(
            _make_issue(
                symbol=None,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_TURNOVER_EXCEEDED,
                severity=RISK_SEVERITY_WARNING,
                message="Turnover estimate exceeds configured Phase 6 cap.",
                value=resolved_turnover,
                limit=resolved_turnover_cap,
            )
        )
    if resolved_sector_cap is not None:
        for sector, weight in sector_weights.items():
            if abs(weight) <= resolved_sector_cap:
                continue
            portfolio_issues.append(
                _make_issue(
                    symbol=None,
                    market=market,
                    as_of=as_of,
                    issue_type=RISK_ISSUE_SECTOR_CONCENTRATION,
                    severity=RISK_SEVERITY_WARNING,
                    message=f"Sector '{sector}' exceeds configured Phase 6 cap.",
                    value=abs(weight),
                    limit=resolved_sector_cap,
                    metadata={"sector": sector},
                )
            )
    if beta_exposure is not None and abs(beta_exposure) > 1.5:
        portfolio_issues.append(
            _make_issue(
                symbol=None,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_BETA_EXPOSURE,
                severity=RISK_SEVERITY_WARNING,
                message="Portfolio beta exposure exceeds Phase 6 diagnostic threshold.",
                value=abs(beta_exposure),
                limit=1.5,
            )
        )
    for cluster, weight in correlation_cluster_weights.items():
        if abs(weight) <= 0.50:
            continue
        portfolio_issues.append(
            _make_issue(
                symbol=None,
                market=market,
                as_of=as_of,
                issue_type=RISK_ISSUE_CORRELATION_CLUSTER,
                severity=RISK_SEVERITY_WARNING,
                message=f"Correlation cluster '{cluster}' exceeds Phase 6 diagnostic threshold.",
                value=abs(weight),
                limit=0.50,
                metadata={"correlation_cluster": cluster},
            )
        )

    avg_symbol_risk = (
        sum(tensor.risk_score for tensor in ordered_tensors) / len(ordered_tensors)
        if ordered_tensors
        else 0.0
    )
    issue_penalty = 0.0
    for issue in portfolio_issues:
        if issue.severity == RISK_SEVERITY_BLOCKER:
            issue_penalty += 0.25
        elif issue.severity == RISK_SEVERITY_WARNING:
            issue_penalty += 0.10
        else:
            issue_penalty += 0.03
    risk_score = clamp_unit_interval(avg_symbol_risk + issue_penalty)

    tensor_metadata = _coerce_metadata(metadata)
    tensor_metadata.setdefault("risk_tensor_schema_version", RISK_TENSOR_SCHEMA_VERSION)

    return PortfolioRiskTensor(
        tensor_id=make_portfolio_tensor_id(
            market=market,
            as_of=as_of,
            symbol_tensor_ids=[tensor.tensor_id for tensor in ordered_tensors],
        ),
        as_of=as_of,
        market=market,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        turnover_estimate=resolved_turnover,
        sector_weights=sector_weights,
        style_exposures=style_exposures,
        factor_exposures=factor_exposures,
        beta_exposure=beta_exposure,
        correlation_cluster_weights=correlation_cluster_weights,
        symbol_tensors=ordered_tensors,
        portfolio_issues=_sort_issues(portfolio_issues),
        blocked_symbols=blocked_symbols,
        max_weight_by_symbol=max_weight_by_symbol,
        gross_exposure_cap=resolved_gross_cap,
        risk_score=risk_score,
        metadata=tensor_metadata,
    )


def build_execution_feasibility_report(
    *,
    symbol_tensors: Sequence[SymbolRiskTensor],
    market: str,
    as_of: str,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionFeasibilityReport:
    ordered_tensors = sorted(list(symbol_tensors), key=lambda tensor: (tensor.symbol, tensor.market))
    feasible_symbols: list[str] = []
    partially_feasible_symbols: list[str] = []
    blocked_symbols: list[str] = []
    execution_by_symbol: dict[str, ExecutionFeasibility] = {}
    issues: list[RiskIssue] = []

    requested_values: list[float] = []
    allowed_values: list[float] = []
    weighted_cost_numerator = 0.0
    weighted_cost_denominator = 0.0
    simple_costs: list[float] = []

    for tensor in ordered_tensors:
        execution = tensor.execution
        execution_by_symbol[tensor.symbol] = execution
        if execution.status == EXECUTION_BLOCKED:
            blocked_symbols.append(tensor.symbol)
            issues.extend(tensor.issues)
        elif execution.status == EXECUTION_PARTIALLY_FEASIBLE:
            partially_feasible_symbols.append(tensor.symbol)
            issues.extend(tensor.issues)
        else:
            feasible_symbols.append(tensor.symbol)

        if execution.requested_trade_value is not None:
            requested_values.append(execution.requested_trade_value)
        if execution.allowed_trade_value is not None:
            allowed_values.append(execution.allowed_trade_value)
        if execution.estimated_transaction_cost_bps is not None:
            simple_costs.append(execution.estimated_transaction_cost_bps)
            if execution.requested_trade_value is not None and execution.requested_trade_value > 0.0:
                weighted_cost_numerator += (
                    execution.estimated_transaction_cost_bps * execution.requested_trade_value
                )
                weighted_cost_denominator += execution.requested_trade_value

    if weighted_cost_denominator > 0.0:
        aggregate_estimated_cost_bps = weighted_cost_numerator / weighted_cost_denominator
    elif simple_costs:
        aggregate_estimated_cost_bps = sum(simple_costs) / len(simple_costs)
    else:
        aggregate_estimated_cost_bps = None

    report_metadata = _coerce_metadata(metadata)
    report_metadata.setdefault("risk_tensor_schema_version", RISK_TENSOR_SCHEMA_VERSION)

    return ExecutionFeasibilityReport(
        report_id=make_execution_report_id(
            market=market,
            as_of=as_of,
            symbol_tensor_ids=[tensor.tensor_id for tensor in ordered_tensors],
        ),
        as_of=as_of,
        market=market,
        total_symbols=len(ordered_tensors),
        feasible_symbols=feasible_symbols,
        partially_feasible_symbols=partially_feasible_symbols,
        blocked_symbols=blocked_symbols,
        execution_by_symbol=execution_by_symbol,
        total_requested_trade_value=sum(requested_values) if requested_values else None,
        total_allowed_trade_value=sum(allowed_values) if allowed_values else None,
        aggregate_estimated_cost_bps=aggregate_estimated_cost_bps,
        issues=_sort_issues(issues),
        metadata=report_metadata,
    )


def build_risk_guard_context_patch(
    portfolio_tensor: PortfolioRiskTensor,
    execution_report: ExecutionFeasibilityReport | None = None,
) -> dict[str, Any]:
    if execution_report is not None:
        execution_status_by_symbol = {
            symbol: execution.status
            for symbol, execution in sorted(execution_report.execution_by_symbol.items())
        }
        execution_blocked_symbols = list(execution_report.blocked_symbols)
        execution_partially_feasible_symbols = list(execution_report.partially_feasible_symbols)
        execution_report_id = execution_report.report_id
    else:
        execution_status_by_symbol = {
            tensor.symbol: tensor.execution.status
            for tensor in sorted(portfolio_tensor.symbol_tensors, key=lambda item: (item.symbol, item.market))
        }
        execution_blocked_symbols = _ordered_unique(
            [
                symbol
                for symbol, status in execution_status_by_symbol.items()
                if status == EXECUTION_BLOCKED
            ]
        )
        execution_partially_feasible_symbols = _ordered_unique(
            [
                symbol
                for symbol, status in execution_status_by_symbol.items()
                if status == EXECUTION_PARTIALLY_FEASIBLE
            ]
        )
        execution_report_id = None

    all_issues = list(portfolio_tensor.portfolio_issues)
    for tensor in portfolio_tensor.symbol_tensors:
        all_issues.extend(tensor.issues)
    risk_issues = [issue.to_dict() for issue in _sort_issues(all_issues)]
    metadata = {
        "risk_tensor_schema_version": RISK_TENSOR_SCHEMA_VERSION,
        "tensor_id": portfolio_tensor.tensor_id,
        "execution_report_id": execution_report_id,
        "blocked_symbol_count": len(portfolio_tensor.blocked_symbols),
        "issue_count": len(risk_issues),
    }
    payload = {
        "blocked_symbols": list(portfolio_tensor.blocked_symbols),
        "max_weight_by_symbol": dict(portfolio_tensor.max_weight_by_symbol),
        "gross_exposure_cap": portfolio_tensor.gross_exposure_cap,
        "risk_score": portfolio_tensor.risk_score,
        "risk_issues": risk_issues,
        "execution_status_by_symbol": execution_status_by_symbol,
        "execution_blocked_symbols": execution_blocked_symbols,
        "execution_partially_feasible_symbols": execution_partially_feasible_symbols,
        "metadata": metadata,
    }
    return dict(_ensure_json_serializable(payload, "risk_guard_context_patch"))


RecordT = TypeVar("RecordT", SymbolRiskTensor, PortfolioRiskTensor, ExecutionFeasibilityReport)


class RiskTensorStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_RISK_TENSOR_DIR
        self.symbol_tensors_path = self.root_dir / DEFAULT_SYMBOL_TENSORS_FILENAME
        self.portfolio_tensors_path = self.root_dir / DEFAULT_PORTFOLIO_TENSORS_FILENAME
        self.execution_reports_path = self.root_dir / DEFAULT_EXECUTION_REPORTS_FILENAME

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(_json_safe(payload)), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _read_jsonl(self, path: Path, record_cls: type[RecordT]) -> list[RecordT]:
        if not path.exists():
            return []
        records: list[RecordT] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path} line {line_number}: {exc.msg}") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"Expected JSON object in {path} line {line_number}.")
                records.append(record_cls.from_dict(payload))
        return records

    def append_symbol_tensor(self, tensor: SymbolRiskTensor) -> None:
        if tensor.tensor_id in self.get_symbol_tensor_ids():
            raise ValueError(f"Duplicate tensor_id in symbol tensor ledger: {tensor.tensor_id}")
        self._append_jsonl(self.symbol_tensors_path, tensor.to_dict())

    def append_symbol_tensors(self, tensors: Sequence[SymbolRiskTensor]) -> int:
        existing = self.get_symbol_tensor_ids()
        seen: set[str] = set()
        for tensor in tensors:
            if tensor.tensor_id in existing or tensor.tensor_id in seen:
                raise ValueError(f"Duplicate tensor_id in symbol tensor ledger: {tensor.tensor_id}")
            seen.add(tensor.tensor_id)
        for tensor in tensors:
            self._append_jsonl(self.symbol_tensors_path, tensor.to_dict())
        return len(tensors)

    def append_portfolio_tensor(self, tensor: PortfolioRiskTensor) -> None:
        if tensor.tensor_id in self.get_portfolio_tensor_ids():
            raise ValueError(f"Duplicate tensor_id in portfolio tensor ledger: {tensor.tensor_id}")
        self._append_jsonl(self.portfolio_tensors_path, tensor.to_dict())

    def append_execution_report(self, report: ExecutionFeasibilityReport) -> None:
        if report.report_id in self.get_execution_report_ids():
            raise ValueError(f"Duplicate report_id in execution report ledger: {report.report_id}")
        self._append_jsonl(self.execution_reports_path, report.to_dict())

    def read_symbol_tensors(self) -> list[SymbolRiskTensor]:
        return self._read_jsonl(self.symbol_tensors_path, SymbolRiskTensor)

    def read_portfolio_tensors(self) -> list[PortfolioRiskTensor]:
        return self._read_jsonl(self.portfolio_tensors_path, PortfolioRiskTensor)

    def read_execution_reports(self) -> list[ExecutionFeasibilityReport]:
        return self._read_jsonl(self.execution_reports_path, ExecutionFeasibilityReport)

    def get_symbol_tensor_ids(self) -> set[str]:
        return {tensor.tensor_id for tensor in self.read_symbol_tensors()}

    def get_portfolio_tensor_ids(self) -> set[str]:
        return {tensor.tensor_id for tensor in self.read_portfolio_tensors()}

    def get_execution_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_execution_reports()}


__all__ = [
    "DEFAULT_RISK_TENSOR_DIR",
    "DEFAULT_SYMBOL_TENSORS_FILENAME",
    "DEFAULT_PORTFOLIO_TENSORS_FILENAME",
    "DEFAULT_EXECUTION_REPORTS_FILENAME",
    "RISK_SEVERITY_INFO",
    "RISK_SEVERITY_WARNING",
    "RISK_SEVERITY_BLOCKER",
    "RISK_ISSUE_DATA_QUARANTINE",
    "RISK_ISSUE_UNTRADABLE",
    "RISK_ISSUE_LOW_LIQUIDITY",
    "RISK_ISSUE_POSITION_TOO_LARGE",
    "RISK_ISSUE_ADV_CAP_EXCEEDED",
    "RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED",
    "RISK_ISSUE_SECTOR_CONCENTRATION",
    "RISK_ISSUE_STYLE_EXPOSURE",
    "RISK_ISSUE_BETA_EXPOSURE",
    "RISK_ISSUE_CORRELATION_CLUSTER",
    "RISK_ISSUE_STRESS_LOSS",
    "RISK_ISSUE_TURNOVER_EXCEEDED",
    "EXECUTION_FEASIBLE",
    "EXECUTION_PARTIALLY_FEASIBLE",
    "EXECUTION_BLOCKED",
    "RiskIssue",
    "SymbolExposure",
    "LiquidityProfile",
    "ExecutionFeasibility",
    "StressScenarioResult",
    "SymbolRiskTensor",
    "PortfolioRiskTensor",
    "ExecutionFeasibilityReport",
    "RiskTensorStore",
    "make_risk_issue_id",
    "make_symbol_tensor_id",
    "make_portfolio_tensor_id",
    "make_execution_report_id",
    "clamp_unit_interval",
    "validate_finite_number",
    "bps_to_decimal_return",
    "weighted_average",
    "build_execution_feasibility",
    "build_symbol_risk_tensor",
    "build_portfolio_risk_tensor",
    "build_execution_feasibility_report",
    "build_risk_guard_context_patch",
]
