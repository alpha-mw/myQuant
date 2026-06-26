from __future__ import annotations

from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    GlobalContext,
    ICDecision,
    PortfolioDecision,
    PortfolioPlan,
    ShortlistItem,
)
from quant_investor.agents.narrator_agent import NarratorAgent
from quant_investor.regime.types import REGIME_RANGE_HIGH_VOL, REGIME_TREND_DOWN
from quant_investor.reporting.conclusion_renderer import ConclusionRenderer


def _markov_payload() -> dict[str, object]:
    return {
        "dominant_regime": REGIME_RANGE_HIGH_VOL,
        "probabilities": {
            "趋势上涨": 0.12,
            "震荡低波": 0.18,
            REGIME_RANGE_HIGH_VOL: 0.45,
            REGIME_TREND_DOWN: 0.20,
            "未知": 0.05,
        },
        "confidence": 0.45,
        "transition_risk": 0.65,
        "execution_mode": "production",
        "suggested_gross_exposure_cap": 0.42,
        "suggested_max_single_weight": 0.09,
        "applied_gross_exposure_cap": 0.40,
        "applied_max_single_weight": 0.08,
        "turnover_cap": 0.30,
        "diagnostic_notes": ["fixture_note"],
    }


def test_render_regime_section_reads_global_context() -> None:
    lines = ConclusionRenderer.render_regime_section(
        GlobalContext(regime_params={"markov": _markov_payload()})
    )

    text = "\n".join(lines)
    assert "## Markov 市场状态" in text
    assert REGIME_RANGE_HIGH_VOL in text
    assert "应用状态: production" in text
    assert "应用后的 gross exposure cap" in text
    assert "应用后的 max single weight" in text
    assert "fixture_note" in text
    assert "shadow" not in text.lower()


def test_render_regime_section_can_show_disabled_marker() -> None:
    lines = ConclusionRenderer.render_regime_section(
        GlobalContext(
            regime_params={"markov": {"enabled": False, "status": "disabled"}}
        )
    )

    assert "Markov regime disabled by config." in "\n".join(lines)


def test_narrator_includes_regime_section_before_bayesian() -> None:
    shortlist = [ShortlistItem(symbol="000001.SZ", company_name="平安银行", rank_score=0.9)]
    bundle = NarratorAgent().run(
        {
            "macro_verdict": BranchVerdict(
                agent_name="macro",
                thesis="macro stable",
                final_score=0.2,
                final_confidence=0.7,
            ),
            "branch_summaries": {
                "quant": BranchVerdict(agent_name="quant", thesis="quant ok"),
            },
            "ic_decisions": [
                ICDecision(
                    symbol="000001.SZ",
                    selected_symbols=["000001.SZ"],
                    action=ActionLabel.BUY,
                    final_confidence=0.8,
                )
            ],
            "portfolio_plan": PortfolioPlan(
                target_exposure=0.30,
                target_gross_exposure=0.30,
                target_net_exposure=0.30,
                target_positions={"000001.SZ": 0.30},
            ),
            "global_context": GlobalContext(
                regime_params={"markov": _markov_payload()},
                metadata={"markov_regime": _markov_payload()},
            ),
            "shortlist": shortlist,
            "portfolio_decision": PortfolioDecision(
                shortlist=shortlist,
                target_weights={"000001.SZ": 0.30},
            ),
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
            "funnel_summary": {"compression_ratio": "2 -> 1"},
            "run_diagnostics": {
                "coverage_summary": ["researchable 1/2"],
                "appendix_diagnostics": ["diag"],
            },
        }
    )

    assert "Markov 市场状态" in bundle.markdown_report
    assert bundle.markdown_report.index("Markov 市场状态") < bundle.markdown_report.index("Bayesian 决策分解")
    assert "shadow" not in bundle.markdown_report.lower()
