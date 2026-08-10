from __future__ import annotations

import copy
from decimal import Decimal
import hashlib

import pytest

from quant_investor.intelligence_v2._core import common_fields, seal
from quant_investor.intelligence_v2.portfolio import (
    PortfolioContractError,
    build_graduation_policy,
    build_paper_execution_policy,
    build_paper_capital_gate,
    build_portfolio_construction,
    build_portfolio_risk_policy,
    validate_portfolio_construction,
    validate_portfolio_risk_policy,
    validate_paper_capital_gate,
)
from quant_investor.intelligence_v2.portfolio import constructor as constructor_module

AT = "2026-08-09T01:00:00Z"


def _exact_ref(name: str, *, artifact_id: str | None = None) -> dict[str, str]:
    digest = hashlib.sha256(name.encode()).hexdigest()
    return {
        "artifact_id": artifact_id or name,
        "artifact_version": f"myquant.test.{name}.v1",
        "available_at": AT,
        "byte_sha256": digest,
        "cutoff": AT,
        "relative_path": f"data/private/{name}.json",
        "semantic_sha256": digest,
    }


def _content_ref(document: dict, identity_field: str) -> dict[str, str]:
    from quant_investor.intelligence_v2._core import content_ref

    return content_ref(document, identity_field=identity_field)


def _support_policies() -> tuple[dict, dict]:
    paper = build_paper_execution_policy(
        created_at=AT,
        effective_from_session="20260801",
        effective_through_session="20260831",
        lot_size=100,
        settlement_rule="T_PLUS_ONE",
        buy_commission_rate="0.0003",
        sell_commission_rate="0.0003",
        minimum_commission_cny="5",
        transfer_fee_rate="0.00001",
        sell_stamp_duty_rate="0.0005",
        slippage_rate="0.001",
        max_fill_adv_participation="0.10",
        fee_rounding_quantum_cny="0.0001",
        fee_rounding_mode="ROUND_HALF_EVEN",
        price_rounding_quantum_cny="0.01",
        allow_partial_fills=True,
        allow_odd_lot_full_exit=True,
        order_expiry_rule="EXPLICIT_SESSION",
        partial_fill_ordering="QUEUE_PRIORITY_THEN_ORDER_ID",
        corporate_action_policy="EXACT_SOURCE_CHRONOLOGY",
        listing_policy="LISTED_NOT_DELISTED",
        exchange_calendar_ref=_exact_ref("calendar"),
        exchange_calendar_sessions=[
            {"session": value, "source_ref": _exact_ref(f"calendar-{value}"), "status": "OPEN"}
            for value in ("20260808", "20260809")
        ],
        price_limit_rules=[
            {
                "board": "MAIN",
                "effective_from_session": "20260801",
                "effective_through_session": "20260831",
                "ipo_no_limit_sessions": 5,
                "limit_ratio": "0.10",
                "rule_id": "MAIN_NORMAL",
                "source_ref": _exact_ref("main-limit"),
                "st": False,
            }
        ],
    )
    graduation = build_graduation_policy(
        created_at=AT,
        required_horizons=[20, 60],
        benchmark_ref=_exact_ref("benchmark"),
        minimum_matured_observations=2,
        minimum_coverage="1",
        minimum_cost_adjusted_excess_return="0",
        maximum_drawdown="0.20",
        maximum_regime_changes=0,
        require_no_hard_risk_breach=True,
    )
    return paper, graduation


def _policy(*, turnover: str = "1", gross: str = "0.40", target_positions: int = 4) -> dict:
    paper, graduation = _support_policies()
    return build_portfolio_risk_policy(
        created_at=AT,
        target_positions=target_positions,
        target_gross=gross,
        cash_floor=str(Decimal("1") - Decimal(gross)),
        per_security_cap="0.20",
        industry_cap="0.20",
        theme_cap="0.20",
        max_adv_participation="0.10",
        turnover_cap=turnover,
        weight_quantum="0.01",
        drawdown_threshold="0.30",
        risk_threshold="0.80",
        hard_veto_codes=["FRAUD"],
        macro_regime_rules=[
            {
                "cash_floor": str(Decimal("1") - Decimal(gross)),
                "gross_cap": gross,
                "regime": "NORMAL",
                "risk_multiplier": "1",
                "veto_codes": ["FRAUD"],
            },
            {
                "cash_floor": "0.80",
                "gross_cap": "0.20",
                "regime": "RISK_OFF",
                "risk_multiplier": "0.50",
                "veto_codes": ["FRAUD", "MACRO_VETO"],
            },
        ],
        fundamental_staleness_allowance_sessions=1,
        paper_execution_policy_ref=_content_ref(paper, "policy_id"),
        graduation_policy_ref=_content_ref(graduation, "policy_id"),
    )


def _decision(company: str, percentile: str) -> dict:
    return seal(
        {
            **common_fields(timestamp_value=AT),
            "company_code": company,
            "deterministic_percentile": percentile,
            "state": "PAPER_CANDIDATE",
            "version": "myquant.v17.intelligence-v2.investment-decision-receipt.v2",
        },
        identity_field="decision_id",
    )


def _subject(index: int, *, same_industry: bool = False) -> dict:
    company = f"{index:06d}.SZ"
    deterministic = f"0.{9 - index}"
    return {
        "advisory_percentile": f"0.{index}",
        "adv_weight_capacity": "1",
        "company_code": company,
        "decision_receipt": _decision(company, deterministic),
        "decision_validation_closure": {"fixture": company},
        "deterministic_percentile": deterministic,
        "drawdown": "0.10",
        "fundamental_age_sessions": 0,
        "hard_veto_codes": [],
        "industry_code": "INDUSTRY" if same_industry else f"INDUSTRY_{index}",
        "industry_ref": _exact_ref(f"industry-{index}"),
        "liquidity_ref": _exact_ref(f"liquidity-{index}"),
        "risk_score": "0.20",
        "security_ref": _exact_ref(f"security-{index}"),
        "theme_codes": [f"THEME_{index}"],
        "theme_refs": [_exact_ref(f"theme-{index}")],
    }


def _current_position(index: int, *, current: str = "0.10") -> dict:
    company = f"{index:06d}.SZ"
    return {
        "adv_weight_capacity": "1",
        "company_code": company,
        "current_weight": current,
        "industry_code": f"INDUSTRY_{index}",
        "industry_ref": _exact_ref(f"industry-{index}"),
        "liquidity_ref": _exact_ref(f"liquidity-{index}"),
        "security_ref": _exact_ref(f"security-{index}"),
        "theme_codes": [f"THEME_{index}"],
        "theme_refs": [_exact_ref(f"theme-{index}")],
    }


@pytest.fixture(autouse=True)
def _decision_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(receipt, **closure):
        assert closure == {"fixture": receipt["company_code"]}
        return receipt

    monkeypatch.setattr(constructor_module, "validate_decision_receipt_v2", validate)


def test_policy_is_exact_no_default_and_macro_can_only_tighten() -> None:
    policy = _policy()
    assert validate_portfolio_risk_policy(policy) == policy
    assert policy["target_positions"] == 4
    assert policy["max_adv_participation"] == "0.100000000000"
    assert policy["fundamental_staleness_allowance_sessions"] == 1
    with pytest.raises(TypeError):
        build_portfolio_risk_policy(created_at=AT)  # type: ignore[call-arg]

    forged = copy.deepcopy(policy)
    forged.pop("semantic_sha256")
    forged.pop("policy_id")
    forged["macro_regime_rules"][1]["gross_cap"] = "0.50"
    with pytest.raises(PortfolioContractError, match="only tighten"):
        build_portfolio_risk_policy(
            created_at=forged["timestamp"],
            target_positions=forged["target_positions"],
            target_gross=forged["target_gross"],
            cash_floor=forged["cash_floor"],
            per_security_cap=forged["per_security_cap"],
            industry_cap=forged["industry_cap"],
            theme_cap=forged["theme_cap"],
            max_adv_participation=forged["max_adv_participation"],
            turnover_cap=forged["turnover_cap"],
            weight_quantum=forged["weight_quantum"],
            drawdown_threshold=forged["drawdown_threshold"],
            risk_threshold=forged["risk_threshold"],
            hard_veto_codes=forged["hard_veto_codes"],
            macro_regime_rules=forged["macro_regime_rules"],
            fundamental_staleness_allowance_sessions=forged[
                "fundamental_staleness_allowance_sessions"
            ],
            paper_execution_policy_ref=forged["paper_execution_policy_ref"],
            graduation_policy_ref=forged["graduation_policy_ref"],
        )


def test_deterministic_pd_pa_and_advisory_tv_fallback_is_exact_pd() -> None:
    subjects = [_subject(index) for index in range(1, 9)]
    receipt = build_portfolio_construction(
        subjects=subjects,
        current_positions=[],
        policy=_policy(),
        macro_regime="NORMAL",
        macro_ref=_exact_ref("macro"),
        as_of=AT,
    )
    assert receipt["status"] == "AVAILABLE"
    assert receipt["p_d"]["gross_weight"] == "0.400000000000"
    assert receipt["p_d"]["cash_weight"] == "0.600000000000"
    assert all(Decimal(row["final_weight"]) == Decimal("0.10") for row in receipt["p_d"]["targets"])
    assert Decimal(receipt["advisory_capital_tv"]) > Decimal("0.10")
    assert receipt["advisory_fallback"] is True
    assert receipt["final_portfolio"] == receipt["p_d"]
    assert (
        validate_portfolio_construction(
            receipt,
            subjects=subjects,
            current_positions=[],
            policy=_policy(),
            macro_regime="NORMAL",
            macro_ref=_exact_ref("macro"),
            as_of=AT,
        )
        == receipt
    )


def test_constraint_order_caps_and_infeasible_cash_block() -> None:
    subjects = [_subject(index, same_industry=True) for index in range(1, 9)]
    receipt = build_portfolio_construction(
        subjects=subjects,
        current_positions=[],
        policy=_policy(),
        macro_regime="NORMAL",
        macro_ref=_exact_ref("macro"),
        as_of=AT,
    )
    assert receipt["status"] == "BLOCKED"
    assert "INFEASIBLE_CASH" in receipt["blocker_codes"]
    assert Decimal(receipt["p_d"]["gross_weight"]) == Decimal("0.20")


def test_macro_only_reduces_gross_and_raises_cash() -> None:
    subjects = [_subject(index) for index in range(1, 9)]
    receipt = build_portfolio_construction(
        subjects=subjects,
        current_positions=[],
        policy=_policy(),
        macro_regime="RISK_OFF",
        macro_ref=_exact_ref("macro"),
        as_of=AT,
    )
    assert Decimal(receipt["p_d"]["gross_weight"]) == Decimal("0.20")
    assert Decimal(receipt["p_d"]["cash_weight"]) == Decimal("0.80")


def test_turnover_is_max_feasible_interpolation() -> None:
    subjects = [_subject(index) for index in range(1, 9)]
    current_positions = [_current_position(index) for index in range(5, 9)]
    receipt = build_portfolio_construction(
        subjects=subjects,
        current_positions=current_positions,
        policy=_policy(turnover="0.10"),
        macro_regime="NORMAL",
        macro_ref=_exact_ref("macro"),
        as_of=AT,
    )
    current = {row["company_code"]: Decimal(row["current_weight"]) for row in current_positions}
    final = {row["company_code"]: Decimal(row["final_weight"]) for row in receipt["p_d"]["targets"]}
    symbols = set(current) | set(final)
    current_cash = Decimal("1") - sum(current.values())
    final_cash = Decimal("1") - sum(final.values())
    tv = (
        sum(
            abs(current.get(code, Decimal("0")) - final.get(code, Decimal("0"))) for code in symbols
        )
        + abs(current_cash - final_cash)
    ) / 2
    assert tv <= Decimal("0.10")
    assert tv >= Decimal("0.09")
    assert "TURNOVER_CONSTRAINED" in receipt["p_d"]["blocker_codes"]


def test_round_robin_reallocates_clipped_cash_to_other_candidates() -> None:
    subjects = [_subject(index) for index in range(1, 6)]
    subjects[0]["adv_weight_capacity"] = "0.10"
    receipt = build_portfolio_construction(
        subjects=subjects,
        current_positions=[],
        policy=_policy(target_positions=5),
        macro_regime="NORMAL",
        macro_ref=_exact_ref("macro"),
        as_of=AT,
    )
    weights = {
        row["company_code"]: Decimal(row["final_weight"]) for row in receipt["p_d"]["targets"]
    }
    assert receipt["p_d"]["status"] == "AVAILABLE"
    assert receipt["p_d"]["gross_weight"] == "0.400000000000"
    assert weights["000001.SZ"] == Decimal("0.01")
    assert sum(weights.values()) == Decimal("0.40")


def test_non_admitted_current_position_can_only_decrease_or_exit() -> None:
    subjects = [_subject(index) for index in range(1, 5)]
    old = _current_position(9, current="0.10")
    receipt = build_portfolio_construction(
        subjects=subjects,
        current_positions=[old],
        policy=_policy(turnover="0.10"),
        macro_regime="NORMAL",
        macro_ref=_exact_ref("macro"),
        as_of=AT,
    )
    weights = {
        row["company_code"]: Decimal(row["final_weight"]) for row in receipt["p_d"]["targets"]
    }
    assert weights.get("000009.SZ", Decimal("0")) == Decimal("0")


def test_second_paper_capital_gate_boundary_fallback_and_replay() -> None:
    baseline = {"000001.SZ": "0.50", "CASH": "0.50"}
    boundary = {"000001.SZ": "0.40", "000002.SZ": "0.10", "CASH": "0.50"}
    closure = {
        "baseline_targets": baseline,
        "provisional_targets": boundary,
        "evaluated_at": AT,
    }
    accepted = build_paper_capital_gate(**closure)
    assert accepted["capital_tv"] == "0.100000000000"
    assert accepted["status"] == "ACCEPTED"
    assert validate_paper_capital_gate(accepted, **closure) == accepted

    rejected_closure = {
        **closure,
        "provisional_targets": {
            "000001.SZ": "0.39",
            "000002.SZ": "0.11",
            "CASH": "0.50",
        },
    }
    rejected = build_paper_capital_gate(**rejected_closure)
    assert rejected["status"] == "REJECTED"
    assert rejected["reason_codes"] == ["ADVISORY_CAPITAL_LIMIT_REJECTED"]
    assert rejected["final_targets"] == rejected["baseline_targets"]
    forged = copy.deepcopy(rejected)
    forged["status"] = "ACCEPTED"
    with pytest.raises(ValueError):
        validate_paper_capital_gate(forged, **rejected_closure)
