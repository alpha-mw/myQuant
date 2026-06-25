from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quant_investor.funnel.theme_boost_diagnostics import (
    ThemeBoostDiagnostics,
    ThemeBoostSymbolDelta,
)
from quant_investor.themes import ThemeShadowDelta, ThemeShadowMonitor
from quant_investor.themes.shadow import build_theme_shadow_monitor


def _context_with_theme_rotation() -> SimpleNamespace:
    return SimpleNamespace(
        market="CN",
        universe_key="full_a",
        latest_trade_date="20260619",
        metadata={
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "market": "CN",
                "universe_key": "full_a",
                "as_of": "20260619",
                "symbol_scores": {
                    "000001.SZ": 0.90,
                    "000002.SZ": 0.86,
                    "000003.SZ": 0.60,
                },
                "symbol_primary_theme": {
                    "000001.SZ": "industry::AI",
                    "000002.SZ": "industry::AI",
                    "000003.SZ": "industry::Robotics",
                },
                "symbol_phase": {
                    "000001.SZ": "confirmed_rotation",
                    "000002.SZ": "confirmed_rotation",
                    "000003.SZ": "confirmed_rotation",
                },
                "symbol_risk_flags": {
                    "000001.SZ": [],
                    "000002.SZ": [],
                    "000003.SZ": [],
                },
                "theme_scores": {
                    "industry::AI": {"theme_name": "AI", "score": 0.9},
                    "industry::Robotics": {"theme_name": "Robotics", "score": 0.7},
                },
                "top_themes": [],
            }
        },
    )


def _context_without_selected_theme_exposure() -> SimpleNamespace:
    return SimpleNamespace(
        market="CN",
        universe_key="full_a",
        latest_trade_date="20260619",
        metadata={
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "market": "CN",
                "universe_key": "full_a",
                "as_of": "20260619",
                "symbol_scores": {},
                "symbol_primary_theme": {},
                "symbol_phase": {},
                "symbol_risk_flags": {},
                "theme_scores": {},
                "top_themes": [],
            }
        },
    )


def test_theme_shadow_monitor_disabled() -> None:
    monitor = build_theme_shadow_monitor(dag_artifacts={}, enabled=False)

    assert monitor.status == "disabled"
    assert monitor.artifact_path == ""
    assert monitor.final_decision_source == "baseline"
    assert monitor.production_decision_source == "no_theme_baseline"
    assert monitor.theme_overlay_applied_to_baseline is False


def test_theme_shadow_monitor_records_production_overlay_source() -> None:
    monitor = build_theme_shadow_monitor(
        dag_artifacts={},
        enabled=True,
        funnel_boost_enabled=False,
        risk_guard_enabled=False,
        portfolio_cap_enabled=False,
        production_funnel_boost_enabled=True,
        production_risk_guard_enabled=True,
        production_portfolio_cap_enabled=True,
        artifact_enabled=False,
    )
    payload = monitor.to_dict()

    assert payload["final_decision_source"] == "baseline"
    assert payload["production_decision_source"] == "theme_overlay_baseline"
    assert payload["control_decision_source"] == "no_theme_baseline"
    assert payload["theme_overlay_applied_to_baseline"] is True
    assert payload["theme_overlay_modules"] == {
        "funnel_boost": True,
        "risk_guard": True,
        "portfolio_cap": True,
    }
    assert payload["canonical_branch_unchanged"] is True
    assert payload["theme_likelihood_added"] is False
    assert payload["posterior_formula_changed"] is False


def test_theme_shadow_monitor_unsupported_execution_target_ignored() -> None:
    monitor = build_theme_shadow_monitor(
        dag_artifacts={},
        enabled=True,
        execution_target="theme",
        funnel_boost_enabled=False,
        risk_guard_enabled=False,
        portfolio_cap_enabled=False,
        artifact_enabled=False,
    )

    assert monitor.status == "success"
    assert monitor.execution_target == "baseline"
    assert monitor.final_decision_source == "baseline"
    assert "unsupported_execution_target_ignored" in monitor.diagnostic_notes


def test_theme_shadow_monitor_candidate_diagnostics_from_existing_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compare_theme_boost_candidates(**_kwargs: object) -> ThemeBoostDiagnostics:
        return ThemeBoostDiagnostics(
            baseline_count=2,
            boosted_count=2,
            overlap_count=1,
            overlap_ratio=1 / 3,
            entered_symbols=["000003.SZ"],
            dropped_symbols=["000002.SZ"],
            improved_symbols=[
                ThemeBoostSymbolDelta(
                    symbol="000003.SZ",
                    baseline_selected=False,
                    boosted_selected=True,
                    primary_theme_id="industry::AI",
                    primary_theme_name="AI",
                )
            ],
        )

    monkeypatch.setattr(
        "quant_investor.funnel.theme_boost_diagnostics.compare_theme_boost_candidates",
        fake_compare_theme_boost_candidates,
    )
    monitor = build_theme_shadow_monitor(
        dag_artifacts={
            "global_context": _context_with_theme_rotation(),
            "branch_results": {"quant": SimpleNamespace(symbol_scores={})},
        },
        enabled=True,
        funnel_boost_enabled=True,
        risk_guard_enabled=False,
        portfolio_cap_enabled=False,
        artifact_enabled=False,
    )

    assert monitor.status == "success"
    assert monitor.entered_candidates == ["000003.SZ"]
    assert monitor.dropped_candidates == ["000002.SZ"]
    assert monitor.baseline_candidate_count == 2
    assert monitor.shadow_candidate_count == 2


def test_theme_shadow_monitor_lightweight_portfolio_cap() -> None:
    portfolio_plan = SimpleNamespace(
        target_weights={
            "000001.SZ": 0.30,
            "000002.SZ": 0.25,
            "000003.SZ": 0.10,
        }
    )

    monitor = build_theme_shadow_monitor(
        dag_artifacts={
            "global_context": _context_with_theme_rotation(),
            "portfolio_plan": portfolio_plan,
        },
        enabled=True,
        funnel_boost_enabled=False,
        risk_guard_enabled=False,
        portfolio_cap_enabled=True,
        artifact_enabled=False,
    )

    assert monitor.status == "success"
    assert monitor.theme_exposure_baseline["industry::AI"] == pytest.approx(0.55)
    assert monitor.theme_exposure_shadow["industry::AI"] <= 0.35 + 1e-9
    assert portfolio_plan.target_weights["000001.SZ"] == pytest.approx(0.30)
    assert "portfolio_shadow_lightweight_cap" in monitor.diagnostic_notes


def test_shadow_portfolio_no_theme_exposure_has_no_weight_deltas() -> None:
    portfolio_plan = SimpleNamespace(
        target_weights={
            "000520.SZ": 0.08032,
            "000065.SZ": 0.080357,
        }
    )

    monitor = build_theme_shadow_monitor(
        dag_artifacts={
            "global_context": _context_without_selected_theme_exposure(),
            "portfolio_plan": portfolio_plan,
        },
        enabled=True,
        funnel_boost_enabled=False,
        risk_guard_enabled=False,
        portfolio_cap_enabled=True,
        artifact_enabled=False,
    )

    assert monitor.status == "success"
    assert monitor.portfolio_weight_deltas == []
    assert monitor.theme_exposure_baseline == {}
    assert monitor.theme_exposure_shadow == {}
    assert "portfolio_shadow_no_theme_exposure" in monitor.diagnostic_notes
    assert portfolio_plan.target_weights["000520.SZ"] == pytest.approx(0.08032)


def test_shadow_portfolio_theme_cap_delta_only_when_cap_binds() -> None:
    portfolio_plan = SimpleNamespace(
        target_weights={
            "000001.SZ": 0.10,
            "000002.SZ": 0.10,
            "000003.SZ": 0.05,
        }
    )

    monitor = build_theme_shadow_monitor(
        dag_artifacts={
            "global_context": _context_with_theme_rotation(),
            "portfolio_plan": portfolio_plan,
        },
        enabled=True,
        funnel_boost_enabled=False,
        risk_guard_enabled=False,
        portfolio_cap_enabled=True,
        artifact_enabled=False,
    )

    assert monitor.status == "success"
    assert monitor.theme_exposure_baseline["industry::AI"] == pytest.approx(0.20)
    assert monitor.theme_exposure_shadow["industry::AI"] == pytest.approx(0.20)
    assert monitor.portfolio_weight_deltas == []


def test_shadow_portfolio_theme_cap_produces_attributed_delta_when_cap_binds() -> None:
    portfolio_plan = SimpleNamespace(
        target_weights={
            "000001.SZ": 0.30,
            "000002.SZ": 0.25,
            "000003.SZ": 0.10,
        }
    )

    monitor = build_theme_shadow_monitor(
        dag_artifacts={
            "global_context": _context_with_theme_rotation(),
            "portfolio_plan": portfolio_plan,
        },
        enabled=True,
        funnel_boost_enabled=False,
        risk_guard_enabled=False,
        portfolio_cap_enabled=True,
        artifact_enabled=False,
    )

    deltas = {delta.symbol: delta for delta in monitor.portfolio_weight_deltas}

    assert monitor.status == "success"
    assert monitor.theme_exposure_shadow["industry::AI"] <= 0.35 + 1e-9
    assert deltas["000001.SZ"].shadow_weight < deltas["000001.SZ"].baseline_weight
    assert deltas["000002.SZ"].shadow_weight < deltas["000002.SZ"].baseline_weight
    assert "000003.SZ" not in deltas
    assert "portfolio_shadow_lightweight_cap" in monitor.diagnostic_notes


def test_shadow_smoke_case_no_candidate_no_risk_no_theme_exposure_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compare_theme_boost_candidates(**_kwargs: object) -> ThemeBoostDiagnostics:
        return ThemeBoostDiagnostics(
            baseline_count=2,
            boosted_count=2,
            overlap_count=2,
            overlap_ratio=1.0,
            entered_symbols=[],
            dropped_symbols=[],
        )

    monkeypatch.setattr(
        "quant_investor.funnel.theme_boost_diagnostics.compare_theme_boost_candidates",
        fake_compare_theme_boost_candidates,
    )
    monitor = build_theme_shadow_monitor(
        dag_artifacts={
            "global_context": _context_without_selected_theme_exposure(),
            "branch_results": {"quant": SimpleNamespace(symbol_scores={})},
            "portfolio_plan": SimpleNamespace(
                target_weights={
                    "000520.SZ": 0.08032,
                    "000065.SZ": 0.080357,
                }
            ),
        },
        enabled=True,
        funnel_boost_enabled=True,
        risk_guard_enabled=True,
        portfolio_cap_enabled=True,
        artifact_enabled=False,
    )

    assert monitor.status == "success"
    assert monitor.candidate_overlap_ratio == pytest.approx(1.0)
    assert monitor.entered_candidates == []
    assert monitor.dropped_candidates == []
    assert monitor.risk_delta["theme_risk_flags"] == []
    assert monitor.risk_delta["theme_effect"] is False
    assert monitor.portfolio_weight_deltas == []
    assert monitor.theme_exposure_baseline == {}
    assert monitor.theme_exposure_shadow == {}
    assert "portfolio_shadow_no_theme_exposure" in monitor.diagnostic_notes


def test_theme_shadow_monitor_error_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_compare(**_kwargs: object) -> ThemeBoostDiagnostics:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "quant_investor.funnel.theme_boost_diagnostics.compare_theme_boost_candidates",
        raise_compare,
    )

    monitor = build_theme_shadow_monitor(
        dag_artifacts={
            "global_context": _context_with_theme_rotation(),
            "branch_results": {"quant": SimpleNamespace(symbol_scores={})},
        },
        enabled=True,
        funnel_boost_enabled=True,
        risk_guard_enabled=False,
        portfolio_cap_enabled=False,
        artifact_enabled=False,
    )

    assert monitor.status == "error"
    assert any("theme_shadow_error" in note for note in monitor.diagnostic_notes)


def test_theme_shadow_monitor_artifact_write(tmp_path) -> None:
    monitor = build_theme_shadow_monitor(
        dag_artifacts={"global_context": _context_with_theme_rotation()},
        enabled=True,
        funnel_boost_enabled=False,
        risk_guard_enabled=False,
        portfolio_cap_enabled=False,
        artifact_enabled=True,
        artifact_dir=tmp_path,
    )

    artifact_path = tmp_path / "CN" / "20260619_full_a_theme_shadow.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert monitor.status == "success"
    assert monitor.artifact_path == str(artifact_path)
    assert payload["artifact_path"] == str(artifact_path)
    assert payload["final_decision_source"] == "baseline"


def test_theme_shadow_monitor_to_dict_and_markdown() -> None:
    monitor = ThemeShadowMonitor(
        status="success",
        candidate_overlap_ratio=0.5,
        selected_overlap_ratio=0.8,
        portfolio_weight_deltas=[
            ThemeShadowDelta(
                symbol="000001.SZ",
                baseline_weight=0.2,
                shadow_weight=0.1,
                weight_delta=-0.1,
                baseline_selected=True,
                shadow_selected=True,
                primary_theme_id="industry::AI",
                primary_theme_name="AI",
            )
        ],
    )

    payload = monitor.to_dict()
    json.dumps(payload)
    markdown = monitor.to_markdown()

    assert payload["final_decision_source"] == "baseline"
    assert "Shadow Monitor" in markdown
    assert "final executable decision remains baseline" in markdown
