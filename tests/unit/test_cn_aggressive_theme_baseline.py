from __future__ import annotations

import argparse
import json
from pathlib import Path

import quant_investor.monitoring.cn_aggressive_daily_review as daily_review


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        base_dir=str(tmp_path / "strategy_records"),
        years=7,
        tracker_max_rounds=3,
        source_record=None,
        allowed_stale_symbols=[],
        categories=None,
        maintenance_years=3,
        maintenance_workers=4,
        maintenance_batch_size=50,
        maintenance_max_rounds=1,
        maintenance_max_batches_per_run=1,
        min_symbol_success_rate=0.95,
        target_date="auto",
        daily_window=True,
        skip_maintenance=True,
        skip_market_metrics_prewarm=False,
    )


def test_daily_review_attaches_theme_overlay_baseline_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daily_review.config, "THEME_FUNNEL_BOOST_ENABLED", True)
    monkeypatch.setattr(daily_review.config, "THEME_RISK_GUARD_ENABLED", True)
    monkeypatch.setattr(daily_review.config, "THEME_PORTFOLIO_CAP_ENABLED", True)

    run_dir = tmp_path / "strategy_records" / "20260623_theme"
    raw_dir = run_dir / "raw_exports"
    raw_dir.mkdir(parents=True)
    (run_dir / "analysis_report.md").write_text("# formal report\n", encoding="utf-8")
    (raw_dir / "aggressive_portfolio_20260623_theme_formal_report.md").write_text(
        "# raw formal report\n",
        encoding="utf-8",
    )
    manifest = {
        "timestamp": "20260623_theme",
        "raw_exports": {
            "report": "raw_exports/aggressive_portfolio_20260623_theme_formal_report.md"
        },
        "formal_diagnostics": {},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "market_snapshot.json").write_text(
        json.dumps({"formal_diagnostics": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        daily_review.tracker,
        "run_tracker",
        lambda _args: {"timestamp": "20260623_theme", "run_dir": str(run_dir)},
    )

    result = daily_review.run_daily_review(_args(tmp_path))

    overlay = result["theme_production_overlay"]
    assert overlay["production_decision_source"] == "theme_overlay_baseline"
    assert overlay["control_decision_source"] == "no_theme_baseline"
    assert overlay["theme_overlay_applied_to_baseline"] is True
    assert overlay["theme_overlay_modules"] == {
        "funnel_boost": True,
        "risk_guard": True,
        "portfolio_cap": True,
    }

    written_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    written_snapshot = json.loads(
        (run_dir / "market_snapshot.json").read_text(encoding="utf-8")
    )
    for payload in (written_manifest, written_snapshot):
        assert payload["theme_production_overlay"]["production_decision_source"] == (
            "theme_overlay_baseline"
        )
        assert payload["formal_diagnostics"]["theme_production_overlay"][
            "canonical_branch_unchanged"
        ] is True
        assert payload["formal_diagnostics"]["theme_production_overlay"][
            "theme_likelihood_added"
        ] is False

    report_text = (run_dir / "analysis_report.md").read_text(encoding="utf-8")
    raw_report_text = (
        raw_dir / "aggressive_portfolio_20260623_theme_formal_report.md"
    ).read_text(encoding="utf-8")
    assert "Theme Baseline Overlay" in report_text
    assert "主题 Production Overlay" in report_text
    assert "Theme Baseline Overlay" in raw_report_text
    assert "production_decision_source: theme_overlay_baseline" in raw_report_text


def test_daily_review_defaults_theme_baseline_on_without_explicit_rollback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    for name in (
        "THEME_FUNNEL_BOOST_ENABLED",
        "THEME_RISK_GUARD_ENABLED",
        "THEME_PORTFOLIO_CAP_ENABLED",
        "THEME_SNAPSHOT_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(daily_review.config, name, False)

    run_dir = tmp_path / "strategy_records" / "20260625_theme_default"
    raw_dir = run_dir / "raw_exports"
    raw_dir.mkdir(parents=True)
    (run_dir / "analysis_report.md").write_text("# formal report\n", encoding="utf-8")
    (raw_dir / "aggressive_portfolio_20260625_theme_default_formal_report.md").write_text(
        "# raw formal report\n",
        encoding="utf-8",
    )
    manifest = {
        "market": "CN",
        "timestamp": "20260625_theme_default",
        "data_snapshot": {"analysis_trade_date": "20260623"},
        "candidate_level_dag_status": {
            "dag_pipeline": {"universe": "full_a"},
        },
        "raw_exports": {
            "report": "raw_exports/aggressive_portfolio_20260625_theme_default_formal_report.md"
        },
        "formal_diagnostics": {},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "market_snapshot.json").write_text(
        json.dumps({"analysis_trade_date": "20260623", "formal_diagnostics": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        daily_review.tracker,
        "run_tracker",
        lambda _args: {"timestamp": "20260625_theme_default", "run_dir": str(run_dir)},
    )

    result = daily_review.run_daily_review(_args(tmp_path))

    overlay = result["theme_production_overlay"]
    assert overlay["production_decision_source"] == "theme_overlay_baseline"
    assert overlay["theme_overlay_modules"] == {
        "funnel_boost": True,
        "risk_guard": True,
        "portfolio_cap": True,
    }
    assert result["rollback_env"] == {}
    assert result["theme_policy_catalyst"]["status"] in {"enabled", "disabled"}

    written_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert written_manifest["theme_production_overlay"]["production_decision_source"] == (
        "theme_overlay_baseline"
    )
    assert written_manifest["rollback_env"] == {}
    assert written_manifest["theme_snapshot"]["status"] != "disabled"
