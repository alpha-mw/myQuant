from __future__ import annotations

import pytest

from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    RiskLevel,
)
from quant_investor.agents.ic_coordinator import V16ICCoordinator
from quant_investor.agents.risk_guard import RiskAdvisor
from quant_investor.market.dag.decision import (
    _run_v16_decision_phase,
    _run_v16_eligibility_phase,
)
from quant_investor.v16.candidate_pipeline import PosteriorMenuItem


def _branches(*risks: str) -> dict[str, BranchVerdict]:
    return {
        "quant": BranchVerdict(
            agent_name="quant",
            investment_risks=list(risks),
        )
    }


def test_risk_advisor_output_has_only_the_five_advisory_fields() -> None:
    assert RiskAdvisor.protocol_version == "v1"
    advisory = RiskAdvisor().run(
        {
            "branch_verdicts": _branches("停牌风险需跟踪"),
            "advisory_context": {
                "scenarios": ["复牌后价格跳空"],
                "suggestions": ["复核事件证据"],
            },
        }
    )

    assert advisory.severity is RiskLevel.EXTREME
    assert set(advisory.to_dict()) == {
        "severity",
        "flags",
        "scenarios",
        "suggestions",
        "rationale",
    }
    for forbidden in (
        "veto",
        "hard_veto",
        "action_cap",
        "blocked_symbols",
        "gross_exposure_cap",
        "target_exposure_cap",
        "max_weight",
        "position_limits",
    ):
        assert not hasattr(advisory, forbidden)


def test_risk_advisory_is_not_an_ic_action_rank_or_weight_input() -> None:
    stage2 = {
        "symbol": "000001.SZ",
        "action": "BUY",
        "selected_for_portfolio": True,
        "target_weight": 0.2,
        "rationale": "Stage2研究结论为买入。",
        "risk_acceptance_rationale": "已识别尾部风险并接受研究假设。",
    }
    coordinator = V16ICCoordinator()
    low = coordinator.run(
        {
            "stage2_decision": stage2,
            "risk_advisory": RiskAdvisor().run({"branch_verdicts": _branches()}),
        }
    )
    extreme = coordinator.run(
        {
            "stage2_decision": stage2,
            "risk_advisory": RiskAdvisor().run({"branch_verdicts": _branches("财务造假风险")}),
        }
    )

    assert low == extreme
    assert coordinator.protocol_version == "v16.codex-authoritative"
    assert low.action == "BUY"
    assert low.target_weight == pytest.approx(0.2)


def test_severe_buy_rationale_check_runs_after_stage2_validation() -> None:
    coordinator = V16ICCoordinator()
    with pytest.raises(ValueError, match="Stage2 action"):
        coordinator.run(
            {
                "stage2_decision": {
                    "symbol": "000001.SZ",
                    "action": "WATCH",
                    "selected_for_portfolio": False,
                    "target_weight": 0.0,
                    "rationale": "不在Stage2动作集合。",
                    "risk_acceptance_rationale": None,
                }
            }
        )

    validated = coordinator.run(
        {
            "stage2_decision": {
                "symbol": "000001.SZ",
                "action": "BUY",
                "selected_for_portfolio": True,
                "target_weight": 0.2,
                "rationale": "研究结论为买入。",
                "risk_acceptance_rationale": None,
            }
        }
    )
    severe = RiskAdvisor().run({"branch_verdicts": _branches("监管立案调查")})
    with pytest.raises(ValueError, match="requires risk_acceptance_rationale"):
        RiskAdvisor.validate_severe_buy_rationale(severe, [validated])


def test_missing_severe_buy_rationale_fails_before_capital_mapping() -> None:
    class ExplodingPortfolioConstructor:
        def run(self, _payload):
            raise AssertionError("capital mapping must not run")

    eligibility = _run_v16_eligibility_phase(
        symbols=["AAA"],
        readiness_by_symbol={
            "AAA": {
                "pit_ready": True,
                "data_ready": True,
                "factor_ready": True,
            }
        },
    )
    with pytest.raises(ValueError, match="requires risk_acceptance_rationale"):
        _run_v16_decision_phase(
            menu=[
                PosteriorMenuItem(
                    symbol="AAA",
                    posterior_win_rate=0.6,
                    posterior_expected_alpha=0.01,
                    posterior_edge_after_costs=0.005,
                )
            ],
            stage2_responses=[
                {
                    "symbol": "AAA",
                    "action": "BUY",
                    "selected_for_portfolio": True,
                    "target_weight": 0.2,
                    "rationale": "研究结论为买入。",
                    "risk_acceptance_rationale": None,
                }
            ],
            branch_summaries=_branches(),
            macro_verdict=None,
            cash_ratio=0.8,
            existing_weights={"AAA": 0.0},
            total_capital=10_000.0,
            reference_price_by_symbol={"AAA": 10.0},
            existing_shares_by_symbol={"AAA": 0.0},
            eligibility_by_symbol=eligibility,
            execution_context_by_symbol={},
            risk_advisory_context={"flags": ["fraud investigation"]},
            portfolio_constructor_cls=ExplodingPortfolioConstructor,
        )
