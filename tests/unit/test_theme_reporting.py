from __future__ import annotations

from types import SimpleNamespace

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
from quant_investor.market.dag.reporting import _build_reporting_artifacts
from quant_investor.market.full_report import (
    CURRENT_MARKET_REPORT_SCHEMA_ENVELOPE,
    generate_full_report,
)
from quant_investor.reporting.theme_renderer import render_theme_rotation_markdown


def _theme_rotation() -> dict[str, object]:
    return {
        "schema_version": "theme_rotation.v1",
        "status": "success",
        "top_themes": [
            {
                "theme_id": "industry::AI",
                "theme_name": "AI",
                "score": 72.5,
                "smoothed_score": 61.3,
                "heat_10d": 61.3,
                "heat_delta_5d": 4.2,
                "persistence_count": 6,
                "trend_state": "warming",
                "phase": "confirmed_rotation",
                "confidence": 0.66,
                "member_count": 18,
                "top_symbols": ["000001.SZ", "000002.SZ"],
                "risk_flags": ["theme_low_breadth"],
            }
        ],
        "metadata": {
            "deterministic": True,
            "no_llm": True,
            "no_network": True,
        },
    }


def _theme_governance() -> dict[str, object]:
    return {
        "schema_version": "theme_governance.v1",
        "enabled": True,
        "status": "success",
        "market": "CN",
        "universe_key": "full_a",
        "as_of": "20260618",
        "summary_counts": {
            "admitted_shadow": 1,
            "watchlist_strong": 0,
            "watchlist_rebuild": 1,
            "rejected": 0,
            "umbrella_only": 0,
            "unavailable": 0,
        },
        "decisions": [
            {
                "theme_id": "industry::铜",
                "theme_name": "铜",
                "gate_label": "admitted_shadow",
                "score": 60.4,
                "confidence": 0.87,
                "breadth": 0.83,
                "member_count": 18,
                "phase": "accumulation",
                "style_tag": "",
                "reasons": ["shadow_admission_defaults_passed"],
            },
            {
                "theme_id": "industry::半导体",
                "theme_name": "半导体",
                "gate_label": "watchlist_rebuild",
                "score": 46.1,
                "confidence": 0.89,
                "breadth": 0.79,
                "member_count": 194,
                "phase": "accumulation",
                "style_tag": "",
                "reasons": ["needs_rebuild_or_calibration"],
            },
        ],
        "diagnostic_notes": [],
        "metadata": {
            "deterministic": True,
            "no_llm": True,
            "no_network": True,
            "shadow_only": True,
        },
    }


def test_render_theme_rotation_markdown_success():
    markdown = render_theme_rotation_markdown(_theme_rotation())

    assert "主题轮动雷达" in markdown
    assert "AI" in markdown
    assert "72.5" in markdown
    assert "10日热度" in markdown
    assert "61.3" in markdown
    assert "warming" in markdown
    assert "confirmed_rotation" in markdown
    assert "theme_low_breadth" in markdown
    assert "不影响组合权重" in markdown


def test_render_theme_rotation_markdown_disabled_safe():
    markdown = render_theme_rotation_markdown({"status": "disabled"})

    assert markdown == "" or "disabled" in markdown or "未启用" in markdown


def test_render_theme_rotation_markdown_error_safe():
    markdown = render_theme_rotation_markdown(
        {
            "status": "error",
            "diagnostic_notes": ["theme_scanner_error: boom"],
        }
    )

    assert "boom" in markdown
    assert "异常" in markdown or "error" in markdown


def test_render_theme_rotation_markdown_malformed_safe():
    markdown = render_theme_rotation_markdown(
        {
            "status": "success",
            "top_themes": "bad",
            "metadata": {"deterministic": object()},
        }
    )

    assert isinstance(markdown, str)


def test_narrator_includes_theme_section_if_feasible():
    shortlist = [
        ShortlistItem(
            symbol="000001.SZ",
            company_name="平安银行",
            rank_score=0.9,
            action=ActionLabel.BUY,
            confidence=0.8,
        )
    ]
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
                target_positions={"000001.SZ": 0.25},
            ),
            "global_context": GlobalContext(metadata={"theme_rotation": _theme_rotation()}),
            "shortlist": shortlist,
            "portfolio_decision": PortfolioDecision(
                shortlist=shortlist,
                target_weights={"000001.SZ": 0.25},
            ),
            "run_diagnostics": {
                "coverage_summary": ["researchable 1/2"],
                "appendix_diagnostics": ["diag"],
            },
        }
    )

    assert "主题轮动雷达" in bundle.markdown_report


def test_reporting_artifacts_includes_theme_rotation_if_feasible():
    class FakeNarrator:
        payload: dict[str, object] = {}

        def run(self, payload: dict[str, object]) -> object:
            type(self).payload = payload
            return SimpleNamespace(markdown_report="# report")

    class FakeSharedReader:
        def snapshot(self) -> dict[str, object]:
            return {"resolution_strategy": "fixture"}

    class FakeDiagnostics:
        def to_dict(self) -> dict[str, object]:
            return {}

    theme_rotation = _theme_rotation()
    global_context = GlobalContext(
        metadata={
            "theme_rotation": theme_rotation,
            "theme_scores": {"industry::AI": {"score": 72.5}},
            "symbol_theme_score": {"000001.SZ": 0.78},
        },
    )
    portfolio_plan = PortfolioPlan(
        target_exposure=0.45,
        target_weights={"000001.SZ": 0.25},
        position_limits={"000001.SZ": 0.25},
    )
    shortlist = [
        ShortlistItem(
            symbol="000001.SZ",
            company_name="平安银行",
            rank_score=0.9,
            action=ActionLabel.BUY,
            confidence=0.8,
        )
    ]

    state = _build_reporting_artifacts(
        market="CN",
        universe_key="full_a",
        all_symbols=["000001.SZ"],
        researchable_symbols=["000001.SZ"],
        candidate_symbols=["000001.SZ"],
        quarantined_symbols=[],
        data_quality_issues=[],
        read_results={},
        shared_reader=FakeSharedReader(),
        global_context=global_context,
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
        shortlist=shortlist,
        portfolio_master_output=None,
        portfolio_master_meta={},
        portfolio_master_reliability=0.0,
        risk_decision=SimpleNamespace(
            veto=False,
            action_cap=SimpleNamespace(value="buy"),
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

    assert FakeNarrator.payload["theme_rotation"] == theme_rotation
    assert state.dag_artifacts["shortlist"] == shortlist
    assert state.dag_artifacts["theme_rotation"] == theme_rotation
    assert state.dag_artifacts["theme_scores"] == {"industry::AI": {"score": 72.5}}
    assert state.dag_artifacts["symbol_theme_score"] == {"000001.SZ": 0.78}


def test_full_market_report_includes_theme_governance_section(tmp_path):
    all_results = {
        "full_a": [
            {
                **CURRENT_MARKET_REPORT_SCHEMA_ENVELOPE,
                "stock_count": 1,
                "stocks": ["000001.SZ"],
                "strategy": {
                    "target_exposure": 0.0,
                    "candidate_symbols": [],
                },
                "branches": {
                    "quant": {
                        "score": -0.1,
                        "confidence": 0.5,
                        "conclusion": "quant cautious",
                    },
                    "fundamental": {
                        "score": 0.0,
                        "confidence": 0.5,
                        "conclusion": "fundamental neutral",
                    },
                    "macro": {
                        "score": 0.0,
                        "confidence": 0.8,
                        "conclusion": "macro neutral",
                    },
                },
                "recommendations": [],
                "execution_log": [],
                "analysis_meta": {
                    **CURRENT_MARKET_REPORT_SCHEMA_ENVELOPE,
                    "market": "CN",
                    "universe": "full_a",
                    "theme_rotation": _theme_rotation(),
                    "theme_governance": _theme_governance(),
                    "data_snapshot": {"summary_text": "fixture snapshot"},
                },
            }
        ]
    }

    paths = generate_full_report(
        all_results,
        market="CN",
        output_dir=str(tmp_path),
        total_capital=1_000_000,
        top_k=12,
    )

    trade_report = tmp_path.joinpath(paths["trade_report"]).read_text(encoding="utf-8")
    summary_report = tmp_path.joinpath(paths["summary_report"]).read_text(encoding="utf-8")

    assert "Theme Governance Sidecar" in trade_report
    assert "shadow/governance only" in trade_report
    assert "final executable decision remains baseline" in trade_report
    assert "铜" in trade_report
    assert "admitted_shadow" in trade_report
    assert "Theme Governance Sidecar" in summary_report
