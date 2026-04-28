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
        allowed_stale_symbols=["601989.SH"],
        categories=None,
        maintenance_years=3,
        maintenance_workers=4,
        maintenance_batch_size=50,
        maintenance_max_rounds=1,
        skip_maintenance=False,
    )


def test_daily_review_runs_maintenance_before_tracker_and_attaches_preflight(monkeypatch, tmp_path):
    calls: list[str] = []

    def _fake_maintenance(**kwargs):
        calls.append("maintenance")
        assert kwargs["market"] == "CN"
        assert kwargs["fail_on_incomplete"] is False
        assert kwargs["allowed_stale_symbols"] == ["601989.SH"]
        return {
            "status": "maintained",
            "categories": ["full_a", "hs300", "zz500", "zz1000"],
            "completeness": {
                "complete": False,
                "latest_trade_date": "20260427",
                "blocking_incomplete_count": 8,
            },
        }

    def _fake_tracker_run(args):
        calls.append("tracker")
        run_dir = tmp_path / "record"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({"timestamp": "run-1"}), encoding="utf-8")
        (run_dir / "market_snapshot.json").write_text(json.dumps({"snapshot": True}), encoding="utf-8")
        return {"timestamp": "run-1", "run_dir": str(run_dir)}

    monkeypatch.setattr(daily_review, "run_market_maintenance", _fake_maintenance)
    monkeypatch.setattr(daily_review.tracker, "run_tracker", _fake_tracker_run)

    result = daily_review.run_daily_review(_args(tmp_path))

    assert calls == ["maintenance", "tracker"]
    assert result["maintenance_preflight"]["status"] == "incomplete"
    manifest = json.loads((tmp_path / "record" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["maintenance_preflight"]["completeness"]["blocking_incomplete_count"] == 8


def test_daily_review_maintenance_failure_does_not_block_tracker(monkeypatch, tmp_path):
    calls: list[str] = []

    def _fake_maintenance(**kwargs):
        calls.append("maintenance")
        raise RuntimeError("upstream not ready")

    def _fake_tracker_run(args):
        calls.append("tracker")
        return {"timestamp": "run-2", "run_dir": str(tmp_path / "missing-record")}

    monkeypatch.setattr(daily_review, "run_market_maintenance", _fake_maintenance)
    monkeypatch.setattr(daily_review.tracker, "run_tracker", _fake_tracker_run)

    result = daily_review.run_daily_review(_args(tmp_path))

    assert calls == ["maintenance", "tracker"]
    assert result["maintenance_preflight"]["status"] == "failed_non_blocking"
    assert "upstream not ready" in result["maintenance_preflight"]["error"]
