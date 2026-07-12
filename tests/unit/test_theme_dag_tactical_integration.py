from __future__ import annotations

from types import SimpleNamespace

from quant_investor.agent_protocol import AgentStatus, ICDecision, RiskDecision
from quant_investor.market.dag import decision as decision_module


def test_portfolio_phase_forwards_executable_tactical_lane(monkeypatch) -> None:
    lane = {
        "enabled": True,
        "status": "active",
        "formal_kill_switch": False,
        "protocol_hash": "a" * 64,
        "regime": "震荡低波",
        "non_tech_symbols": ["GOLD"],
        "nav_cap": 0.10,
        "max_positions": 1,
    }
    monkeypatch.setattr(
        decision_module,
        "build_theme_risk_constraints",
        lambda **_: {},
    )
    monkeypatch.setattr(
        decision_module,
        "build_theme_portfolio_constraints",
        lambda **_: {
            "theme_portfolio_cap_enabled": True,
            "theme_exposure_map": {},
            "theme_caps": {},
            "theme_names": {},
            "theme_phases": {},
            "theme_tactical_lane": lane,
            "diagnostic_notes": ["theme_tactical_lane_active"],
        },
    )

    class _RiskGuard:
        def run(self, _payload):
            return RiskDecision(
                max_weight=0.60,
                gross_exposure_cap=1.0,
                position_limits={"GOLD": 0.60},
            )

    class _Coordinator:
        def run(self, _payload):
            return ICDecision()

    captured: dict = {}

    class _PortfolioConstructor:
        def run(self, payload):
            captured.update(payload)
            return SimpleNamespace(
                status=AgentStatus.SUCCESS,
                target_exposure=0.10,
                target_gross_exposure=0.10,
                target_net_exposure=0.10,
                cash_ratio=0.90,
                target_weights={"GOLD": 0.10},
                target_positions={"GOLD": 100_000.0},
            )

    decision_module._run_portfolio_construction_phase(
        shortlist=[SimpleNamespace(symbol="GOLD", metadata={})],
        branch_summaries={},
        macro_verdict=SimpleNamespace(),
        global_context=SimpleNamespace(
            risk_budget={
                "target_exposure": 1.0,
                "max_single_weight": 0.60,
            },
            metadata={},
        ),
        data_quality_issues=[],
        ic_hints_by_symbol={},
        research_by_symbol={"GOLD": {}},
        tradability_snapshot={
            "GOLD": {"industry": "Metals", "liquidity_score": 1.0}
        },
        funnel_summary={},
        bayesian_records=[],
        candidate_symbols=["GOLD"],
        portfolio_master_output=None,
        portfolio_master_meta={},
        risk_guard_cls=_RiskGuard,
        ic_coordinator_cls=_Coordinator,
        portfolio_constructor_cls=_PortfolioConstructor,
        attach_symbol_to_ic_decision_fn=lambda decision, **_: decision,
    )

    assert captured["risk_limits"]["theme_tactical_lane"] == lane
    assert captured["risk_limits"]["theme_portfolio_diagnostic_notes"] == [
        "theme_tactical_lane_active"
    ]
