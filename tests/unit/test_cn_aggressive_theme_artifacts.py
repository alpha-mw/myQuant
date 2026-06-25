from __future__ import annotations

import argparse
import json
from pathlib import Path

import quant_investor.monitoring.cn_aggressive_daily_review as daily_review
from quant_investor.themes.storage import ThemeSnapshotStore


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


def test_daily_review_persists_theme_snapshot_and_shadow_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "strategy_records"
    run_dir = base_dir / "20260623_theme"
    raw_dir = run_dir / "raw_exports"
    raw_dir.mkdir(parents=True)
    (run_dir / "analysis_report.md").write_text("# formal report\n", encoding="utf-8")
    (raw_dir / "aggressive_portfolio_20260623_theme_formal_report.md").write_text(
        "# raw formal report\n",
        encoding="utf-8",
    )
    (base_dir / "latest_notes_payload.md").write_text("# notes\n", encoding="utf-8")
    manifest = {
        "market": "CN",
        "timestamp": "20260623_theme",
        "data_snapshot": {"analysis_trade_date": "20260618"},
        "candidate_level_dag_status": {
            "dag_pipeline": {"universe": "full_a"},
        },
        "files": {"analysis_report": "analysis_report.md"},
        "raw_exports": {
            "report": "raw_exports/aggressive_portfolio_20260623_theme_formal_report.md",
        },
        "formal_diagnostics": {},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "market_snapshot.json").write_text(
        json.dumps(
            {
                "analysis_trade_date": "20260618",
                "formal_diagnostics": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshot_root = tmp_path / "theme_snapshots"
    theme_snapshot_path = ThemeSnapshotStore(snapshot_root).save(
        {
            "status": "success",
            "market": "CN",
            "universe_key": "full_a",
            "as_of": "20260618",
            "theme_scores": {"industry::AI": 88.0},
            "top_themes": [{"theme_name": "AI"}],
            "metadata": {
                "input_scope": "full_market",
                "scanned_symbol_count": 100,
                "member_count_min": 5,
            },
        },
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        run_id="theme-run",
    )
    shadow_root = tmp_path / "theme_shadow"
    shadow_path = shadow_root / "CN" / "20260618_full_a_theme_shadow.json"
    shadow_path.parent.mkdir(parents=True)
    shadow_path.write_text(
        json.dumps(
            {
                "status": "success",
                "market": "CN",
                "universe_key": "full_a",
                "as_of": "20260618",
                "final_decision_source": "baseline",
                "production_decision_source": "theme_overlay_baseline",
                "control_decision_source": "no_theme_baseline",
                "theme_overlay_applied_to_baseline": True,
                "candidate_overlap_ratio": 1.0,
                "entered_candidates": [],
                "dropped_candidates": [],
                "selected_overlap_ratio": 1.0,
                "portfolio_weight_deltas": [],
                "theme_exposure_baseline": {"industry::AI": 0.1},
                "theme_exposure_shadow": {"industry::AI": 0.1},
                "risk_delta": {"theme_effect": False},
                "diagnostic_notes": ["portfolio_shadow_no_theme_delta"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(daily_review.config, "THEME_FUNNEL_BOOST_ENABLED", True)
    monkeypatch.setattr(daily_review.config, "THEME_RISK_GUARD_ENABLED", True)
    monkeypatch.setattr(daily_review.config, "THEME_PORTFOLIO_CAP_ENABLED", True)
    monkeypatch.setattr(daily_review.config, "THEME_SNAPSHOT_ENABLED", True)
    monkeypatch.setattr(daily_review.config, "THEME_SNAPSHOT_DIR", str(snapshot_root))
    monkeypatch.setattr(daily_review.config, "THEME_SHADOW_MODE_ENABLED", True)
    monkeypatch.setattr(daily_review.config, "THEME_SHADOW_ARTIFACT_DIR", str(shadow_root))
    monkeypatch.setattr(
        daily_review.tracker,
        "run_tracker",
        lambda _args: {"timestamp": "20260623_theme", "run_dir": str(run_dir)},
    )

    result = daily_review.run_daily_review(_args(tmp_path))

    assert result["theme_production_overlay"]["production_decision_source"] == (
        "theme_overlay_baseline"
    )
    written_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    written_snapshot = json.loads(
        (run_dir / "market_snapshot.json").read_text(encoding="utf-8")
    )
    for payload in (written_manifest, written_snapshot):
        assert payload["theme_snapshot"]["status"] == "success"
        assert payload["theme_snapshot"]["path"] == str(theme_snapshot_path)
        assert payload["theme_rotation"]["status"] == "success"
        assert payload["theme_shadow_monitor"]["status"] == "success"
        assert payload["theme_shadow_monitor"]["artifact_path"] == str(shadow_path)
        assert payload["theme_shadow_monitor"]["theme_snapshot_path"] == str(
            theme_snapshot_path
        )
        assert payload["formal_diagnostics"]["theme_snapshot"]["status"] == "success"
        assert payload["formal_diagnostics"]["theme_shadow_monitor"]["status"] == (
            "success"
        )

    assert written_manifest["files"]["theme_snapshot"] == "theme_snapshot.json"
    assert written_manifest["files"]["theme_shadow_monitor"] == "theme_shadow_monitor.json"
    assert (
        written_manifest["raw_exports"]["theme_snapshot"]
        == "raw_exports/aggressive_portfolio_20260623_theme_formal_theme_snapshot.json"
    )
    assert (
        written_manifest["raw_exports"]["theme_shadow_monitor"]
        == "raw_exports/aggressive_portfolio_20260623_theme_formal_theme_shadow_monitor.json"
    )
    assert (run_dir / "theme_snapshot.json").exists()
    assert (run_dir / "theme_shadow_monitor.json").exists()
    assert (raw_dir / "aggressive_portfolio_20260623_theme_formal_theme_snapshot.json").exists()
    assert (
        raw_dir / "aggressive_portfolio_20260623_theme_formal_theme_shadow_monitor.json"
    ).exists()

    for path in (
        run_dir / "analysis_report.md",
        raw_dir / "aggressive_portfolio_20260623_theme_formal_report.md",
        base_dir / "latest_notes_payload.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "Theme Baseline Overlay" in text
        assert "Theme Shadow Monitor" in text
        assert str(theme_snapshot_path) in text
        assert str(shadow_path) in text
