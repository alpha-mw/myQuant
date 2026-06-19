from __future__ import annotations

import json

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.deterministic_funnel import FunnelConfig, FunnelOutput
from quant_investor.funnel.theme_boost_diagnostics import (
    build_theme_boost_diagnostics_from_outputs,
    compare_theme_boost_candidates,
)


def _theme_context() -> GlobalContext:
    return GlobalContext(
        metadata={
            "theme_rotation": {
                "status": "success",
                "symbol_scores": {"D": 0.88},
                "symbol_primary_theme": {"D": "industry::AI"},
                "symbol_phase": {"D": "confirmed_rotation"},
                "symbol_risk_flags": {"D": ["theme_low_breadth"]},
                "theme_scores": {
                    "industry::AI": {
                        "theme_name": "AI",
                        "score": 72.0,
                        "confidence": 0.7,
                        "member_count": 12,
                    }
                },
            }
        }
    )


def test_build_diagnostics_from_outputs_entered_and_dropped():
    baseline = FunnelOutput(
        candidates=["A", "B", "C"],
        candidate_scores={"A": 0.9, "B": 0.8, "C": 0.7},
    )
    boosted = FunnelOutput(
        candidates=["A", "D", "B"],
        candidate_scores={"A": 0.91, "D": 0.82, "B": 0.79},
    )

    diagnostics = build_theme_boost_diagnostics_from_outputs(
        baseline_output=baseline,
        boosted_output=boosted,
        global_context=GlobalContext(),
    )

    assert diagnostics.entered_symbols == ["D"]
    assert diagnostics.dropped_symbols == ["C"]
    assert diagnostics.overlap_count == 2
    assert diagnostics.overlap_ratio == 0.5
    assert diagnostics.deltas_by_symbol["B"].rank_delta == -1
    assert diagnostics.deteriorated_symbols[0].symbol == "B"
    assert "D" in diagnostics.deltas_by_symbol


def test_build_diagnostics_extracts_theme_metadata():
    baseline = FunnelOutput(
        candidates=["A", "B"],
        candidate_scores={"A": 0.9, "B": 0.8},
    )
    boosted = FunnelOutput(
        candidates=["D", "A"],
        candidate_scores={"D": 0.92, "A": 0.91},
    )

    diagnostics = build_theme_boost_diagnostics_from_outputs(
        baseline_output=baseline,
        boosted_output=boosted,
        global_context=_theme_context(),
    )

    delta = diagnostics.deltas_by_symbol["D"]
    assert delta.primary_theme_id == "industry::AI"
    assert delta.primary_theme_name == "AI"
    assert delta.theme_phase == "confirmed_rotation"
    assert "theme_low_breadth" in delta.theme_risk_flags
    assert diagnostics.phase_summary["confirmed_rotation"] == 1
    assert diagnostics.risk_flag_counts["theme_low_breadth"] == 1


def test_diagnostics_to_dict_and_markdown_safe():
    diagnostics = build_theme_boost_diagnostics_from_outputs(
        baseline_output=FunnelOutput(candidates=["A"], candidate_scores={"A": 0.9}),
        boosted_output=FunnelOutput(candidates=["A"], candidate_scores={"A": 0.91}),
        global_context=GlobalContext(),
    )

    json.dumps(diagnostics.to_dict())
    markdown = diagnostics.to_markdown()

    assert "## Theme Boost A/B Diagnostics" in markdown
    assert "baseline_count" in markdown
    assert "boosted_count" in markdown


def test_compare_theme_boost_candidates_does_not_mutate_config_if_feasible():
    symbol_state = {
        "A": {
            "momentum_strength": 0.70,
            "breakout_readiness": 0.70,
            "volume_confirmation": 0.50,
            "trend_stability": 0.60,
            "distance_from_high_pct": 0.02,
            "fake_breakout_risk": 0.05,
            "max_drawdown_pct": 0.03,
            "return_20d": 0.08,
        },
        "B": {
            "momentum_strength": 0.68,
            "breakout_readiness": 0.68,
            "volume_confirmation": 0.48,
            "trend_stability": 0.58,
            "distance_from_high_pct": 0.02,
            "fake_breakout_risk": 0.05,
            "max_drawdown_pct": 0.03,
            "return_20d": 0.07,
        },
    }
    global_context = GlobalContext(
        universe_symbols=["A", "B"],
        universe_tiers={"researchable": ["A", "B"]},
        liquidity_filter={"liquidity_scores": {"A": 1.0, "B": 1.0}},
        metadata={
            "symbol_market_state": symbol_state,
            "theme_rotation": {
                "status": "success",
                "symbol_scores": {"B": 0.95},
                "symbol_primary_theme": {"B": "industry::AI"},
                "symbol_phase": {"B": "confirmed_rotation"},
                "symbol_risk_flags": {"B": []},
                "theme_scores": {"industry::AI": {"theme_name": "AI"}},
            },
        },
    )
    quant_result = BranchResult(
        branch_name="quant",
        symbol_scores={"A": 0.6, "B": 0.55},
    )
    base_config = FunnelConfig(
        profile="momentum_leader",
        max_candidates=2,
        theme_boost_enabled=False,
        theme_boost_cap=0.04,
    )

    diagnostics = compare_theme_boost_candidates(
        quant_result=quant_result,
        global_context=global_context,
        base_config=base_config,
    )

    assert base_config.theme_boost_enabled is False
    assert base_config.theme_boost_cap == 0.04
    assert diagnostics.metadata["diagnostic_only"] is True
    assert diagnostics.metadata["no_llm"] is True
    assert diagnostics.metadata["no_network"] is True


def test_compare_theme_boost_candidates_uses_disabled_and_enabled_configs(monkeypatch):
    captured_configs: list[FunnelConfig] = []

    def _fake_run(self, *, quant_result, global_context):
        captured_configs.append(self.config)
        if self.config.theme_boost_enabled:
            return FunnelOutput(
                candidates=["B", "A"],
                candidate_scores={"B": 0.95, "A": 0.90},
            )
        return FunnelOutput(
            candidates=["A", "B"],
            candidate_scores={"A": 0.90, "B": 0.88},
        )

    monkeypatch.setattr(
        "quant_investor.funnel.theme_boost_diagnostics.DeterministicFunnel.run",
        _fake_run,
    )

    diagnostics = compare_theme_boost_candidates(
        quant_result=BranchResult(branch_name="quant", symbol_scores={"A": 0.9}),
        global_context=GlobalContext(),
        base_config=FunnelConfig(profile="momentum_leader", theme_boost_cap=0.03),
        boost_cap=0.07,
    )

    assert [config.theme_boost_enabled for config in captured_configs] == [False, True]
    assert captured_configs[1].theme_boost_cap == 0.07
    assert diagnostics.deltas_by_symbol["B"].rank_delta == 1


def test_empty_outputs_safe():
    diagnostics = build_theme_boost_diagnostics_from_outputs(
        baseline_output=FunnelOutput(),
        boosted_output=FunnelOutput(),
        global_context=GlobalContext(),
    )

    assert diagnostics.overlap_ratio == 0.0
    assert diagnostics.entered_symbols == []
    assert diagnostics.dropped_symbols == []
    assert "## Theme Boost A/B Diagnostics" in diagnostics.to_markdown()
