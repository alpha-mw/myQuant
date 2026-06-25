from __future__ import annotations

from types import SimpleNamespace

from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    GlobalContext,
    ICDecision,
    PortfolioDecision,
    PortfolioPlan,
    ReportBundle,
)
from quant_investor.agents.narrator_agent import NarratorAgent
from quant_investor.market.dag import reporting as reporting_module
from quant_investor.reporting.theme_shadow_renderer import (
    append_theme_production_overlay_section_once,
    append_theme_shadow_section_once,
    render_theme_production_overlay_markdown,
    render_theme_shadow_monitor_markdown,
)


def _success_monitor() -> dict[str, object]:
    return {
        "status": "success",
        "final_decision_source": "baseline",
        "candidate_overlap_ratio": 0.5,
        "entered_candidates": ["000003.SZ"],
        "dropped_candidates": ["000002.SZ"],
        "selected_overlap_ratio": 0.8,
        "portfolio_weight_deltas": [
            {
                "symbol": "000001.SZ",
                "baseline_weight": 0.2,
                "shadow_weight": 0.1,
                "weight_delta": -0.1,
                "primary_theme_id": "industry::AI",
                "primary_theme_name": "AI",
                "phase": "confirmed_rotation",
            }
        ],
        "theme_exposure_baseline": {"industry::AI": 0.5},
        "theme_exposure_shadow": {"industry::AI": 0.35},
        "risk_delta": {
            "baseline_action_cap": "buy",
            "shadow_action_cap": "hold",
            "theme_risk_flags": ["theme_overextended"],
        },
        "artifact_path": "results/theme_shadow/CN/20260619_full_a_theme_shadow.json",
    }


def test_render_theme_shadow_monitor_markdown_success() -> None:
    markdown = render_theme_shadow_monitor_markdown(_success_monitor())

    assert "主题 Shadow Monitor" in markdown
    assert "final executable decision remains baseline" in markdown
    assert "000003.SZ" in markdown


def test_render_theme_shadow_monitor_markdown_disabled_empty() -> None:
    assert render_theme_shadow_monitor_markdown({"status": "disabled"}) == ""


def test_render_theme_shadow_monitor_markdown_error() -> None:
    markdown = render_theme_shadow_monitor_markdown(
        {
            "status": "error",
            "diagnostic_notes": ["theme_shadow_error: boom"],
        }
    )

    assert "error" in markdown
    assert "boom" in markdown


def test_append_theme_shadow_section_once() -> None:
    markdown = append_theme_shadow_section_once("# report\n", _success_monitor())

    assert "主题 Shadow Monitor" in markdown
    assert "final executable decision remains baseline" in markdown

    repeated = append_theme_shadow_section_once(markdown, _success_monitor())
    assert repeated == markdown
    assert repeated.count("主题 Shadow Monitor") == 1

    assert append_theme_shadow_section_once("# report\n", {"status": "disabled"}) == "# report\n"


def test_render_theme_production_overlay_markdown_explicit_on() -> None:
    overlay = {
        "production_decision_source": "theme_overlay_baseline",
        "control_decision_source": "no_theme_baseline",
        "theme_overlay_applied_to_baseline": True,
        "theme_overlay_modules": {
            "funnel_boost": True,
            "risk_guard": True,
            "portfolio_cap": True,
        },
        "canonical_branch_unchanged": True,
        "theme_likelihood_added": False,
        "posterior_formula_changed": False,
    }
    markdown = render_theme_production_overlay_markdown(overlay)

    assert "主题 Production Overlay" in markdown
    assert "production_decision_source: theme_overlay_baseline" in markdown
    assert "control_decision_source: no_theme_baseline" in markdown
    assert "theme_likelihood_added: false" in markdown

    appended = append_theme_production_overlay_section_once("# report\n", overlay)
    assert appended.count("主题 Production Overlay") == 1
    assert append_theme_production_overlay_section_once(appended, overlay) == appended


def test_render_theme_production_overlay_markdown_default_off_empty() -> None:
    assert (
        render_theme_production_overlay_markdown(
            {
                "production_decision_source": "no_theme_baseline",
                "theme_overlay_applied_to_baseline": False,
            }
        )
        == ""
    )


def test_narrator_includes_theme_shadow_section_if_feasible() -> None:
    bundle = NarratorAgent().run(
        {
            "macro_verdict": BranchVerdict(
                agent_name="macro",
                thesis="macro stable",
                final_score=0.2,
                final_confidence=0.7,
            ),
            "branch_summaries": {
                "quant": BranchVerdict(
                    agent_name="quant",
                    thesis="quant ok",
                    final_score=0.4,
                    final_confidence=0.8,
                ),
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
                target_exposure=0.45,
                target_gross_exposure=0.45,
                target_net_exposure=0.45,
                target_weights={"000001.SZ": 0.25},
                target_positions={"000001.SZ": 0.25},
            ),
            "global_context": GlobalContext(),
            "portfolio_decision": PortfolioDecision(target_weights={"000001.SZ": 0.25}),
            "theme_shadow_monitor": _success_monitor(),
            "run_diagnostics": {
                "coverage_summary": ["researchable 1/2"],
                "appendix_diagnostics": ["diag"],
            },
        }
    )

    assert "主题 Shadow Monitor" in bundle.markdown_report
    assert "final executable decision remains baseline" in bundle.markdown_report


def test_reporting_artifacts_include_theme_shadow_when_enabled_if_feasible(monkeypatch) -> None:
    class FakeNarrator:
        payload: dict[str, object] = {}

        def run(self, payload: dict[str, object]) -> ReportBundle:
            type(self).payload = payload
            return ReportBundle(markdown_report="# report")

    class FakeSharedReader:
        def snapshot(self) -> dict[str, object]:
            return {"resolution_strategy": "fixture"}

    class FakeDiagnostics:
        def to_dict(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(reporting_module.config, "THEME_SHADOW_MODE_ENABLED", True)
    monkeypatch.setattr(reporting_module.config, "THEME_SHADOW_EXECUTION_TARGET", "baseline")
    monkeypatch.setattr(reporting_module.config, "THEME_SHADOW_FUNNEL_BOOST_ENABLED", False)
    monkeypatch.setattr(reporting_module.config, "THEME_SHADOW_RISK_GUARD_ENABLED", False)
    monkeypatch.setattr(reporting_module.config, "THEME_SHADOW_PORTFOLIO_CAP_ENABLED", False)
    monkeypatch.setattr(reporting_module.config, "THEME_SHADOW_ARTIFACT_ENABLED", False)
    monkeypatch.setattr(reporting_module.config, "THEME_SHADOW_ARTIFACT_DIR", "results/theme_shadow")
    monkeypatch.setattr(reporting_module.config, "THEME_SHADOW_MAX_ROWS", 10)
    monkeypatch.setattr(reporting_module.config, "THEME_FUNNEL_BOOST_ENABLED", True)
    monkeypatch.setattr(reporting_module.config, "THEME_RISK_GUARD_ENABLED", True)
    monkeypatch.setattr(reporting_module.config, "THEME_PORTFOLIO_CAP_ENABLED", True)

    portfolio_plan = PortfolioPlan(
        target_exposure=0.45,
        target_weights={"000001.SZ": 0.25},
        position_limits={"000001.SZ": 0.25},
    )
    state = reporting_module._build_reporting_artifacts(
        market="CN",
        universe_key="full_a",
        all_symbols=["000001.SZ"],
        researchable_symbols=["000001.SZ"],
        candidate_symbols=["000001.SZ"],
        quarantined_symbols=[],
        data_quality_issues=[],
        read_results={},
        shared_reader=FakeSharedReader(),
        global_context=GlobalContext(market="CN", universe_key="full_a"),
        provider_health={},
        model_roles=SimpleNamespace(),
        funnel_summary={},
        bayesian_records=[],
        review_bundle=SimpleNamespace(fallback_reasons=[], metadata={}),
        ic_hints_by_symbol={},
        macro_verdict=BranchVerdict(final_score=0.0),
        branch_summaries={},
        branch_verdicts_by_symbol={},
        branch_results={},
        ic_decisions=[],
        portfolio_plan=portfolio_plan,
        portfolio_decision=PortfolioDecision(target_weights={"000001.SZ": 0.25}),
        symbol_research_packets={},
        shortlist=[],
        portfolio_master_output=None,
        portfolio_master_meta={},
        portfolio_master_reliability=0.0,
        risk_decision=SimpleNamespace(
            veto=False,
            action_cap=SimpleNamespace(value="buy"),
            gross_exposure_cap=0.45,
            max_weight=0.25,
            position_limits={"000001.SZ": 0.25},
            blocked_symbols=[],
            to_dict=lambda: {},
        ),
        tradability_snapshot={},
        scoped_data_snapshot={},
        download_stage=None,
        category_count=1,
        funnel_output=SimpleNamespace(),
        global_quant_verdict=BranchVerdict(),
        narrator_agent_cls=FakeNarrator,
        build_data_quality_diagnostics_fn=lambda **kwargs: FakeDiagnostics(),
        build_what_if_plan_fn=lambda **kwargs: SimpleNamespace(),
        build_execution_trace_fn=lambda **kwargs: SimpleNamespace(),
        build_bayesian_trace_fn=lambda **kwargs: {},
    )

    assert "theme_shadow_monitor" in state.dag_artifacts
    assert state.dag_artifacts["theme_shadow_monitor"]["status"] == "success"
    assert state.dag_artifacts["theme_production_overlay"][
        "production_decision_source"
    ] == "theme_overlay_baseline"
    assert state.dag_artifacts["theme_shadow_monitor"][
        "production_decision_source"
    ] == "theme_overlay_baseline"
    assert FakeNarrator.payload["theme_shadow_monitor"] == state.dag_artifacts["theme_shadow_monitor"]
    assert FakeNarrator.payload["theme_production_overlay"] == state.dag_artifacts[
        "theme_production_overlay"
    ]
