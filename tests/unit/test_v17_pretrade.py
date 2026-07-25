from __future__ import annotations

import pytest

from quant_investor.v17.contracts import V17ContractError
from quant_investor.v17.permissions import determine_trade_permission
from quant_investor.v17.pretrade import (
    COST_FIELD_ORDER,
    build_execution_cost_policy,
    estimate_transaction_cost,
    evaluate_pretrade,
    validate_pretrade_result,
)
from quant_investor.v17.risk_policy import (
    build_available_risk_policy_snapshot,
    build_unavailable_risk_policy_snapshot,
)
from quant_investor.v17.semantic import seal_semantic


def _cost() -> dict[str, object]:
    return build_execution_cost_policy(
        buy_commission=0.0003,
        sell_commission=0.0003,
        sell_stamp_tax=0.0005,
        buy_transfer_fee=0.00001,
        sell_transfer_fee=0.00001,
        buy_slippage=0.001,
        sell_slippage=0.0012,
        market_impact=0.0004,
    )


def _risk() -> dict[str, object]:
    return build_available_risk_policy_snapshot(
        policy_id="risk-v1",
        strategy_id="cn-shadow",
        market="CN",
        pit_cutoff="2026-07-21",
        as_of="2026-07-21T00:00:00Z",
        expires_at="2026-07-24T00:00:00Z",
        gross_cap=0.8,
        cash_floor=0.2,
        single_name_cap=0.1,
        industry_cap=0.3,
        cluster_cap=0.2,
        beta_cap=1.2,
        stress_loss_cap=0.15,
        adv20_participation_cap=0.05,
        turnover_cap=0.3,
        stress_scenario="cn_stress_v1",
        source_refs=[
            {
                "source_id": "owner",
                "path": "private/policy.json",
                "byte_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
            }
        ],
    )


def _permission() -> dict[str, object]:
    return determine_trade_permission(
        symbol="600000.SH",
        held=False,
        tradable=True,
        fundamental_eligibility="F_ELIGIBLE",
        severe_red_flag=False,
        quant_timing="BUY_NOW",
    )


def _proposal() -> dict[str, object]:
    return {
        "symbol": "600000.SH",
        "side": "BUY",
        "trade_notional": 40_000.0,
        "adv20": 1_000_000.0,
        "position_weight_after": 0.08,
        "industry_weight_after": 0.25,
        "cluster_weight_after": 0.15,
        "beta_after": 1.1,
        "stress_loss_after": 0.1,
        "turnover_after": 0.2,
    }


def test_cost_model_uses_fixed_eight_component_order_and_side_specific_rates() -> None:
    buy = estimate_transaction_cost(trade_notional=100_000.0, side="BUY", cost_policy=_cost())
    sell = estimate_transaction_cost(trade_notional=100_000.0, side="SELL", cost_policy=_cost())
    assert [row["component"] for row in buy["components"]] == list(COST_FIELD_ORDER)
    assert buy["fraction"] == pytest.approx(0.0003 + 0.00001 + 0.001 + 0.0004)
    assert sell["fraction"] == pytest.approx(0.0003 + 0.0005 + 0.00001 + 0.0012 + 0.0004)


def test_pretrade_passes_only_when_permission_and_all_risk_checks_pass() -> None:
    result = evaluate_pretrade(
        _proposal(),
        permission=_permission(),
        risk_policy=_risk(),
        cost_policy=_cost(),
        cutoff="2026-07-22T00:00:00Z",
    )
    assert result["passed"] is True
    assert [row["name"] for row in result["checks"]] == [
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
    assert result["authority"] is False
    assert (
        validate_pretrade_result(
            result,
            proposal=_proposal(),
            permission=_permission(),
            risk_policy=_risk(),
            cost_policy=_cost(),
            cutoff="2026-07-22T00:00:00Z",
        )
        == result
    )

    blocked = _proposal()
    blocked["trade_notional"] = 60_000.0
    result = evaluate_pretrade(
        blocked,
        permission=_permission(),
        risk_policy=_risk(),
        cost_policy=_cost(),
        cutoff="2026-07-22T00:00:00Z",
    )
    assert result["passed"] is False
    assert next(row for row in result["checks"] if row["name"] == "adv20")["passed"] is False


def test_pretrade_rejects_unavailable_risk_instead_of_defaulting() -> None:
    unavailable = build_unavailable_risk_policy_snapshot(
        policy_id="risk-v1",
        strategy_id="cn-shadow",
        market="CN",
        reason="owner_policy_missing",
    )
    with pytest.raises(V17ContractError, match="AVAILABLE risk"):
        evaluate_pretrade(
            _proposal(),
            permission=_permission(),
            risk_policy=unavailable,
            cost_policy=_cost(),
            cutoff="2026-07-22T00:00:00Z",
        )


def test_pretrade_cannot_resurrect_denied_permission() -> None:
    denied = determine_trade_permission(
        symbol="600000.SH",
        held=False,
        tradable=True,
        fundamental_eligibility="F_INELIGIBLE",
        severe_red_flag=False,
        quant_timing="BUY_NOW",
    )
    result = evaluate_pretrade(
        _proposal(),
        permission=denied,
        risk_policy=_risk(),
        cost_policy=_cost(),
        cutoff="2026-07-22T00:00:00Z",
    )
    assert result["passed"] is False
    assert result["checks"][0] == {
        "name": "permission",
        "passed": False,
        "observed": False,
        "limit": True,
    }


def test_pretrade_result_rejects_resealed_cost_or_check_arithmetic_drift() -> None:
    result = evaluate_pretrade(
        _proposal(),
        permission=_permission(),
        risk_policy=_risk(),
        cost_policy=_cost(),
        cutoff="2026-07-22T00:00:00Z",
    )
    drifted = {k: v for k, v in result.items() if k != "semantic_sha256"}
    drifted["cost"] = {
        **drifted["cost"],
        "amount": drifted["cost"]["amount"] + 1.0,
    }
    with pytest.raises(V17ContractError, match="total amount mismatch"):
        validate_pretrade_result(
            seal_semantic(drifted),
            proposal=_proposal(),
            permission=_permission(),
            risk_policy=_risk(),
            cost_policy=_cost(),
            cutoff="2026-07-22T00:00:00Z",
        )

    drifted = {k: v for k, v in result.items() if k != "semantic_sha256"}
    drifted["checks"] = [dict(item) for item in drifted["checks"]]
    drifted["checks"][2]["passed"] = False
    drifted["passed"] = False
    with pytest.raises(V17ContractError, match="adv20 check arithmetic mismatch"):
        validate_pretrade_result(
            seal_semantic(drifted),
            proposal=_proposal(),
            permission=_permission(),
            risk_policy=_risk(),
            cost_policy=_cost(),
            cutoff="2026-07-22T00:00:00Z",
        )

    drifted = {k: v for k, v in result.items() if k != "semantic_sha256"}
    drifted["cost"] = {**drifted["cost"]}
    drifted["cost"]["components"] = [dict(item) for item in drifted["cost"]["components"]]
    drifted["cost"]["components"][1]["applied"] = True
    with pytest.raises(V17ContractError, match="side applicability"):
        validate_pretrade_result(
            seal_semantic(drifted),
            proposal=_proposal(),
            permission=_permission(),
            risk_policy=_risk(),
            cost_policy=_cost(),
            cutoff="2026-07-22T00:00:00Z",
        )


def test_pretrade_result_cannot_be_resealed_for_a_different_proposal() -> None:
    result = evaluate_pretrade(
        _proposal(),
        permission=_permission(),
        risk_policy=_risk(),
        cost_policy=_cost(),
        cutoff="2026-07-22T00:00:00Z",
    )
    different = _proposal()
    different["adv20"] = 2_000_000.0
    with pytest.raises(V17ContractError, match="sealed source payloads"):
        validate_pretrade_result(
            result,
            proposal=different,
            permission=_permission(),
            risk_policy=_risk(),
            cost_policy=_cost(),
            cutoff="2026-07-22T00:00:00Z",
        )
