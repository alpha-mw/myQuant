from __future__ import annotations

from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    DataQualityDiagnostics,
    ExecutionTrace,
    ExecutionTraceStep,
    GlobalContext,
    ICDecision,
    ModelRoleMetadata,
    PortfolioDecision,
    PortfolioPlan,
    ShortlistItem,
    SymbolResearchPacket,
    WhatIfPlan,
    WhatIfScenario,
)
from quant_investor.agents.narrator_agent import NarratorAgent


def test_narrator_report_carries_bayesian_and_trace_closure_fields():
    narrator = NarratorAgent()
    shortlist = [
        ShortlistItem(
            symbol="000001.SZ",
            company_name="平安银行",
            rank_score=0.9,
            action=ActionLabel.BUY,
            confidence=0.8,
            rationale=["posterior top rank"],
            metadata={
                "posterior_action_score": 0.9,
                "posterior_win_rate": 0.68,
                "posterior_confidence": 0.8,
            },
        )
    ]
    bundle = narrator.run(
        {
            "macro_verdict": BranchVerdict(agent_name="macro", thesis="macro stable", final_score=0.2, final_confidence=0.7),
            "branch_summaries": {
                "quant": BranchVerdict(agent_name="quant", thesis="quant ok", final_score=0.4, final_confidence=0.8),
            },
            "ic_decisions": [
                ICDecision(
                    symbol="000001.SZ",
                    selected_symbols=["000001.SZ"],
                    action=ActionLabel.BUY,
                    final_confidence=0.8,
                    metadata={
                        "symbol_candidates": [
                            {
                                "symbol": "000001.SZ",
                                "company_name": "平安银行",
                                "action": "buy",
                                "confidence": 0.8,
                                "one_line_conclusion": "posterior top rank",
                            }
                        ]
                    },
                )
            ],
            "portfolio_plan": PortfolioPlan(
                target_exposure=0.45,
                target_gross_exposure=0.45,
                target_net_exposure=0.45,
                target_positions={"000001.SZ": 0.25},
            ),
            "review_bundle": None,
            "ic_hints_by_symbol": {"000001.SZ": {"action": "buy", "score": 0.9}},
            "model_role_metadata": ModelRoleMetadata(branch_model="deepseek-reasoner", master_model="moonshot-v1-128k"),
            "execution_trace": ExecutionTrace(
                key_parameters={"total_universe_count": 5000, "final_selected_count": 1},
                steps=[ExecutionTraceStep(stage="bayesian_decision", role="system", model="deterministic", success=True, conclusion="posterior ranked")],
                final_deterministic_outcome={"selected_count": 1},
            ),
            "what_if_plan": WhatIfPlan(
                scenarios=[WhatIfScenario(scenario_name="macro_turns_weaker", action="reduce risk")]
            ),
            "global_context": GlobalContext(
                market="CN",
                universe_key="full_a",
                universe_symbols=["000001.SZ"],
                universe_tiers={
                    "total": ["000001.SZ", "000002.SZ"],
                    "researchable": ["000001.SZ"],
                    "shortlistable": ["000001.SZ"],
                    "final_selected": ["000001.SZ"],
                },
                data_quality_diagnostics=DataQualityDiagnostics(total_universe_count=2, researchable_universe_count=1),
            ),
            "symbol_research_packets": {
                "000001.SZ": SymbolResearchPacket(symbol="000001.SZ", company_name="平安银行")
            },
            "shortlist": shortlist,
            "portfolio_decision": PortfolioDecision(shortlist=shortlist, target_weights={"000001.SZ": 0.25}),
            "bayesian_records": [
                {
                    "symbol": "000001.SZ",
                    "company_name": "平安银行",
                    "posterior_action_score": 0.9,
                    "posterior_win_rate": 0.68,
                    "posterior_confidence": 0.8,
                    "rank": 1,
                }
            ],
            "funnel_summary": {"compression_ratio": "5000 -> 400 -> 20 -> 1"},
            "run_diagnostics": {
                "coverage_summary": ["researchable 1/2"],
                "appendix_diagnostics": ["diag"],
            },
        }
    )

    assert bundle.global_context is not None
    assert bundle.symbol_research_packets["000001.SZ"].company_name == "平安银行"
    assert bundle.shortlist[0].company_name == "平安银行"
    assert bundle.portfolio_decision is not None
    assert "Bayesian 决策分解" in bundle.markdown_report
    assert "平安银行" in bundle.markdown_report
    assert "What-If 响应计划" in bundle.markdown_report
    assert bundle.metadata["shortlist_count"] == 1
    assert bundle.metadata["final_selected_count"] == 1
