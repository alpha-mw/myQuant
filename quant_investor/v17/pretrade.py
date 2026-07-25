"""Deterministic v17 pre-trade vetoes and explicit eight-part cost model."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, cast

from .contracts import (
    Availability,
    TradeSide,
    V17ContractError,
    coerce_enum,
    require_authority_false,
    require_bool,
    require_exact_keys,
    require_number,
    require_ratio,
    require_symbol,
)
from .permissions import validate_trade_permission
from .risk_policy import validate_portfolio_risk_policy_snapshot
from .semantic import (
    canonical_json_bytes,
    require_sha256,
    seal_semantic,
    semantic_sha256,
    validate_semantic_seal,
)

EXECUTION_COST_POLICY_VERSION = "myquant.v17.execution-cost-policy.v1"
PRETRADE_RESULT_VERSION = "myquant.v17.pretrade-result.v1"

COST_FIELD_ORDER = (
    "buy_commission",
    "sell_commission",
    "sell_stamp_tax",
    "buy_transfer_fee",
    "sell_transfer_fee",
    "buy_slippage",
    "sell_slippage",
    "market_impact",
)
COST_POLICY_KEYS = frozenset(
    {
        "version",
        *COST_FIELD_ORDER,
        "authority",
        "semantic_sha256",
    }
)
PRETRADE_INPUT_KEYS = frozenset(
    {
        "symbol",
        "side",
        "trade_notional",
        "adv20",
        "position_weight_after",
        "industry_weight_after",
        "cluster_weight_after",
        "beta_after",
        "stress_loss_after",
        "turnover_after",
    }
)
PRETRADE_RESULT_KEYS = frozenset(
    {
        "version",
        "symbol",
        "side",
        "passed",
        "checks",
        "cost",
        "proposal_semantic_sha256",
        "permission_semantic_sha256",
        "risk_policy_semantic_sha256",
        "cost_policy_semantic_sha256",
        "authority",
        "semantic_sha256",
    }
)
PRETRADE_CHECK_KEYS = frozenset({"name", "passed", "observed", "limit"})
COST_RESULT_KEYS = frozenset({"trade_notional", "fraction", "amount", "components"})
COST_COMPONENT_KEYS = frozenset({"component", "rate", "applied", "amount"})


def build_execution_cost_policy(
    *,
    buy_commission: float,
    sell_commission: float,
    sell_stamp_tax: float,
    buy_transfer_fee: float,
    sell_transfer_fee: float,
    buy_slippage: float,
    sell_slippage: float,
    market_impact: float,
) -> dict[str, Any]:
    return validate_execution_cost_policy(
        seal_semantic(
            {
                "version": EXECUTION_COST_POLICY_VERSION,
                "buy_commission": buy_commission,
                "sell_commission": sell_commission,
                "sell_stamp_tax": sell_stamp_tax,
                "buy_transfer_fee": buy_transfer_fee,
                "sell_transfer_fee": sell_transfer_fee,
                "buy_slippage": buy_slippage,
                "sell_slippage": sell_slippage,
                "market_impact": market_impact,
                "authority": False,
            }
        )
    )


def validate_execution_cost_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, COST_POLICY_KEYS, label="execution cost policy")
    if sealed.get("version") != EXECUTION_COST_POLICY_VERSION:
        raise V17ContractError("execution cost policy version mismatch")
    require_authority_false(sealed.get("authority"))
    for field in COST_FIELD_ORDER:
        require_ratio(sealed.get(field), label=field)
    buy_fraction = sum(
        float(sealed[field])
        for field in (
            "buy_commission",
            "buy_transfer_fee",
            "buy_slippage",
            "market_impact",
        )
    )
    sell_fraction = sum(
        float(sealed[field])
        for field in (
            "sell_commission",
            "sell_stamp_tax",
            "sell_transfer_fee",
            "sell_slippage",
            "market_impact",
        )
    )
    if buy_fraction >= 1.0 or sell_fraction >= 1.0:
        raise V17ContractError("applicable transaction cost fraction must be below one")
    return sealed


def estimate_transaction_cost(
    *,
    trade_notional: float,
    side: str,
    cost_policy: Mapping[str, Any],
) -> dict[str, Any]:
    notional = require_number(
        trade_notional,
        label="trade_notional",
        minimum=0.0,
        minimum_exclusive=True,
    )
    canonical_side = cast(TradeSide, coerce_enum(side, TradeSide, label="side"))
    policy = validate_execution_cost_policy(cost_policy)
    applicable = {
        TradeSide.BUY: {
            "buy_commission",
            "buy_transfer_fee",
            "buy_slippage",
            "market_impact",
        },
        TradeSide.SELL: {
            "sell_commission",
            "sell_stamp_tax",
            "sell_transfer_fee",
            "sell_slippage",
            "market_impact",
        },
    }[canonical_side]
    breakdown: list[dict[str, Any]] = []
    total = 0.0
    total_fraction = 0.0
    for field in COST_FIELD_ORDER:
        rate = float(policy[field])
        applied = field in applicable
        amount = notional * rate if applied else 0.0
        if applied:
            total_fraction += rate
            total += amount
        breakdown.append({"component": field, "rate": rate, "applied": applied, "amount": amount})
    return {
        "trade_notional": notional,
        "fraction": total_fraction,
        "amount": total,
        "components": breakdown,
    }


def _check(name: str, passed: bool, observed: Any, limit: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "limit": limit}


def evaluate_pretrade(
    proposal: Mapping[str, Any],
    *,
    permission: Mapping[str, Any],
    risk_policy: Mapping[str, Any],
    cost_policy: Mapping[str, Any],
    cutoff: str | None,
) -> dict[str, Any]:
    """Evaluate all fixed vetoes; never creates a permission or an order."""

    if not isinstance(proposal, Mapping):
        raise V17ContractError("pretrade proposal must be an object")
    require_exact_keys(proposal, PRETRADE_INPUT_KEYS, label="pretrade proposal")
    symbol = require_symbol(proposal.get("symbol"))
    side = cast(TradeSide, coerce_enum(proposal.get("side"), TradeSide, label="side"))
    notional = require_number(
        proposal.get("trade_notional"),
        label="trade_notional",
        minimum=0.0,
        minimum_exclusive=True,
    )
    adv20 = require_number(
        proposal.get("adv20"), label="adv20", minimum=0.0, minimum_exclusive=True
    )
    position_weight = require_ratio(
        proposal.get("position_weight_after"), label="position_weight_after"
    )
    industry_weight = require_ratio(
        proposal.get("industry_weight_after"), label="industry_weight_after"
    )
    cluster_weight = require_ratio(
        proposal.get("cluster_weight_after"), label="cluster_weight_after"
    )
    beta = require_number(proposal.get("beta_after"), label="beta_after")
    stress_loss = require_ratio(proposal.get("stress_loss_after"), label="stress_loss_after")
    # No half-turnover factor is used by policy.  A full rotation can therefore
    # reach 2.0 even though the owner policy cap itself remains a [0, 1] ratio.
    turnover = require_number(
        proposal.get("turnover_after"),
        label="turnover_after",
        minimum=0.0,
        maximum=2.0,
    )
    normalized_proposal = {
        "symbol": symbol,
        "side": side.value,
        "trade_notional": notional,
        "adv20": adv20,
        "position_weight_after": position_weight,
        "industry_weight_after": industry_weight,
        "cluster_weight_after": cluster_weight,
        "beta_after": beta,
        "stress_loss_after": stress_loss,
        "turnover_after": turnover,
    }

    validated_permission = validate_trade_permission(permission)
    if validated_permission["symbol"] != symbol:
        raise V17ContractError("pretrade permission symbol mismatch")
    validated_risk = validate_portfolio_risk_policy_snapshot(risk_policy, cutoff=cutoff)
    if validated_risk["availability"] != Availability.AVAILABLE.value:
        # Use equality, not truthiness/defaulting: an unavailable policy means
        # upstream must terminate with no portfolio.
        raise V17ContractError("pretrade requires AVAILABLE risk policy")
    validated_cost = validate_execution_cost_policy(cost_policy)

    permission_allowed = bool(
        validated_permission["can_buy"]
        if side is TradeSide.BUY
        else validated_permission["can_sell"]
    )
    checks = [
        _check("permission", permission_allowed, permission_allowed, True),
        _check(
            "tradability",
            bool(validated_permission["tradable"]),
            bool(validated_permission["tradable"]),
            True,
        ),
        _check(
            "adv20",
            notional <= adv20 * float(validated_risk["adv20_participation_cap"]),
            notional,
            adv20 * float(validated_risk["adv20_participation_cap"]),
        ),
        _check(
            "single_name",
            position_weight <= float(validated_risk["single_name_cap"]),
            position_weight,
            float(validated_risk["single_name_cap"]),
        ),
        _check(
            "industry",
            industry_weight <= float(validated_risk["industry_cap"]),
            industry_weight,
            float(validated_risk["industry_cap"]),
        ),
        _check(
            "beta",
            abs(beta) <= float(validated_risk["beta_cap"]),
            abs(beta),
            float(validated_risk["beta_cap"]),
        ),
        _check(
            "cluster",
            cluster_weight <= float(validated_risk["cluster_cap"]),
            cluster_weight,
            float(validated_risk["cluster_cap"]),
        ),
        _check(
            "stress",
            stress_loss <= float(validated_risk["stress_loss_cap"]),
            stress_loss,
            float(validated_risk["stress_loss_cap"]),
        ),
        _check(
            "turnover",
            turnover <= float(validated_risk["turnover_cap"]),
            turnover,
            float(validated_risk["turnover_cap"]),
        ),
    ]
    cost = estimate_transaction_cost(
        trade_notional=notional,
        side=side.value,
        cost_policy=validated_cost,
    )
    checks.append(_check("cost", True, cost["fraction"], "explicit_eight_component_model"))
    return seal_semantic(
        {
            "version": PRETRADE_RESULT_VERSION,
            "symbol": symbol,
            "side": side.value,
            "passed": all(item["passed"] for item in checks),
            "checks": checks,
            "cost": cost,
            "proposal_semantic_sha256": semantic_sha256(normalized_proposal),
            "permission_semantic_sha256": validated_permission["semantic_sha256"],
            "risk_policy_semantic_sha256": validated_risk["semantic_sha256"],
            "cost_policy_semantic_sha256": validated_cost["semantic_sha256"],
            "authority": False,
        }
    )


def validate_pretrade_result(
    payload: Mapping[str, Any],
    *,
    proposal: Mapping[str, Any],
    permission: Mapping[str, Any],
    risk_policy: Mapping[str, Any],
    cost_policy: Mapping[str, Any],
    cutoff: str | None,
) -> dict[str, Any]:
    """Validate a result against every sealed source and the exact proposal.

    A result is not self-authenticating merely because its internal arithmetic
    and semantic seal agree.  Requiring all source payloads here prevents a
    caller from re-sealing fabricated checks or swapping the evaluated
    proposal while retaining plausible source digest strings.
    """

    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, PRETRADE_RESULT_KEYS, label="pretrade result")
    if sealed.get("version") != PRETRADE_RESULT_VERSION:
        raise V17ContractError("pretrade result version mismatch")
    require_authority_false(sealed.get("authority"))
    require_symbol(sealed.get("symbol"))
    side = cast(TradeSide, coerce_enum(sealed.get("side"), TradeSide, label="side"))
    checks = sealed.get("checks")
    expected_names = [
        "permission",
        "tradability",
        "adv20",
        "single_name",
        "industry",
        "beta",
        "cluster",
        "stress",
        "turnover",
        "cost",
    ]
    if not isinstance(checks, list) or any(not isinstance(item, Mapping) for item in checks):
        raise V17ContractError("pretrade checks must be an array of objects")
    if [item.get("name") for item in checks] != expected_names:
        raise V17ContractError("pretrade check order mismatch")
    for index, item in enumerate(checks):
        require_exact_keys(item, PRETRADE_CHECK_KEYS, label=f"checks[{index}]")
        require_bool(item.get("passed"), label=f"checks[{index}].passed")
    for field in (
        "proposal_semantic_sha256",
        "permission_semantic_sha256",
        "risk_policy_semantic_sha256",
        "cost_policy_semantic_sha256",
    ):
        require_sha256(sealed.get(field), label=field)
    cost = sealed.get("cost")
    if not isinstance(cost, Mapping):
        raise V17ContractError("pretrade cost must be an object")
    require_exact_keys(cost, COST_RESULT_KEYS, label="pretrade cost")
    trade_notional = require_number(
        cost.get("trade_notional"),
        label="cost.trade_notional",
        minimum=0.0,
        minimum_exclusive=True,
    )
    declared_fraction = require_ratio(cost.get("fraction"), label="cost.fraction")
    declared_amount = require_number(cost.get("amount"), label="cost.amount", minimum=0.0)
    components = cost.get("components")
    if not isinstance(components, list) or any(
        not isinstance(item, Mapping) for item in components
    ):
        raise V17ContractError("pretrade cost components must be an array of objects")
    if [item.get("component") for item in components] != list(COST_FIELD_ORDER):
        raise V17ContractError("pretrade cost component order mismatch")
    applicable = {
        TradeSide.BUY: {
            "buy_commission",
            "buy_transfer_fee",
            "buy_slippage",
            "market_impact",
        },
        TradeSide.SELL: {
            "sell_commission",
            "sell_stamp_tax",
            "sell_transfer_fee",
            "sell_slippage",
            "market_impact",
        },
    }[side]
    expected_fraction = 0.0
    expected_amount = 0.0
    for index, item in enumerate(components):
        require_exact_keys(item, COST_COMPONENT_KEYS, label=f"cost.components[{index}]")
        component = item["component"]
        rate = require_ratio(item.get("rate"), label=f"cost.components[{index}].rate")
        applied = require_bool(item.get("applied"), label=f"cost.components[{index}].applied")
        if applied is not (component in applicable):
            raise V17ContractError("pretrade cost side applicability mismatch")
        amount = require_number(
            item.get("amount"),
            label=f"cost.components[{index}].amount",
            minimum=0.0,
        )
        calculated = trade_notional * rate if applied else 0.0
        if not math.isclose(amount, calculated, rel_tol=1e-12, abs_tol=1e-12):
            raise V17ContractError("pretrade cost component amount mismatch")
        if applied:
            expected_fraction += rate
            expected_amount += calculated
    if not math.isclose(declared_fraction, expected_fraction, rel_tol=1e-12, abs_tol=1e-12):
        raise V17ContractError("pretrade cost fraction mismatch")
    if not math.isclose(declared_amount, expected_amount, rel_tol=1e-12, abs_tol=1e-8):
        raise V17ContractError("pretrade cost total amount mismatch")

    for index, item in enumerate(checks):
        name = item["name"]
        if name in {"permission", "tradability"}:
            boolean_observed = require_bool(item.get("observed"), label=f"checks[{index}].observed")
            if item.get("limit") is not True or item["passed"] is not boolean_observed:
                raise V17ContractError(f"pretrade {name} check arithmetic mismatch")
            continue
        if name == "cost":
            cost_observed = require_ratio(item.get("observed"), label="checks[cost].observed")
            if (
                item.get("limit") != "explicit_eight_component_model"
                or item["passed"] is not True
                or not math.isclose(
                    cost_observed,
                    declared_fraction,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise V17ContractError("pretrade cost check mismatch")
            continue
        numeric_observed = require_number(
            item.get("observed"), label=f"checks[{index}].observed", minimum=0.0
        )
        limit = require_number(item.get("limit"), label=f"checks[{index}].limit", minimum=0.0)
        if item["passed"] is not (numeric_observed <= limit):
            raise V17ContractError(f"pretrade {name} check arithmetic mismatch")

    expected_passed = all(item["passed"] is True for item in checks)
    if require_bool(sealed.get("passed"), label="passed") is not expected_passed:
        raise V17ContractError("pretrade passed flag mismatch")
    expected = evaluate_pretrade(
        proposal,
        permission=permission,
        risk_policy=risk_policy,
        cost_policy=cost_policy,
        cutoff=cutoff,
    )
    if canonical_json_bytes(sealed) != canonical_json_bytes(expected):
        raise V17ContractError("pretrade result does not match sealed source payloads")
    return sealed


__all__ = [
    "COST_FIELD_ORDER",
    "EXECUTION_COST_POLICY_VERSION",
    "PRETRADE_RESULT_VERSION",
    "build_execution_cost_policy",
    "estimate_transaction_cost",
    "evaluate_pretrade",
    "validate_execution_cost_policy",
    "validate_pretrade_result",
]
