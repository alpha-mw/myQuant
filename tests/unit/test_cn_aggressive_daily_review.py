from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

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
        maintenance_batch_size=200,
        maintenance_max_rounds=1,
        maintenance_max_batches_per_run=200,
        min_symbol_success_rate=0.95,
        target_date="auto",
        daily_window=True,
        skip_maintenance=False,
        skip_market_metrics_prewarm=False,
    )


class FakePreflightDownloader:
    probe = {"applicable": True, "available": False, "coverage_ratio": 0.0}
    completeness = {
        "complete": False,
        "coverage_ratio": 0.97,
        "blocking_incomplete_count": 3,
        "latest_trade_date": "20260315",
        "strict_trade_date": "20260316",
        "stable_trade_date": "20260315",
        "effective_target_trade_date": "20260315",
        "categories": {},
    }

    def __init__(self, **_kwargs):
        self.strict_trade_date = "20260316"
        self.stable_trade_date = "20260315"
        self.latest_trade_date = "20260316"
        self.freshness_mode = "strict"

    def load_components(self):
        return {"full_a": ["000001.SZ"], "hs300": [], "zz500": [], "zz1000": []}

    def _resolve_target_categories(self, _components, categories):
        return categories or ["full_a"]

    def _probe_strict_same_day_close_availability(self, **_kwargs):
        return dict(self.probe)

    def build_completeness_report(self, **_kwargs):
        return dict(self.completeness)


@pytest.fixture(autouse=True)
def _stub_parquet_preflight_completeness(monkeypatch):
    monkeypatch.setattr(
        daily_review.tracker,
        "build_parquet_canonical_completeness_report",
        lambda **_kwargs: dict(FakePreflightDownloader.completeness),
    )


def test_daily_review_skips_staged_maintenance_when_same_day_unavailable_and_snapshot_sufficient(
    monkeypatch,
    tmp_path,
):
    calls: list[str] = []
    FakePreflightDownloader.probe = {"applicable": True, "available": False, "coverage_ratio": 0.0}
    FakePreflightDownloader.completeness = {
        "complete": False,
        "coverage_ratio": 0.50,
        "blocking_incomplete_count": 50,
        "latest_trade_date": "20260315",
        "strict_trade_date": "20260316",
        "stable_trade_date": "20260315",
        "effective_target_trade_date": "20260315",
        "categories": {},
    }

    def _fake_tracker_run(args):
        calls.append("tracker")
        run_dir = tmp_path / "record"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({"timestamp": "run-1"}), encoding="utf-8")
        (run_dir / "market_snapshot.json").write_text(json.dumps({"snapshot": True}), encoding="utf-8")
        return {"timestamp": "run-1", "run_dir": str(run_dir)}

    monkeypatch.setattr(daily_review, "CNFullMarketDownloader", FakePreflightDownloader, raising=False)
    monkeypatch.setattr(
        daily_review,
        "run_storage_validate",
        lambda **_kwargs: {
            "status": "passed",
            "snapshot_id": "snap-1",
            "latest_complete_trade_date": "20260315",
            "manifest_path": "/tmp/manifest.json",
            "coverage": {"symbol_count": 1},
            "blockers": [],
        },
        raising=False,
    )
    monkeypatch.setattr(
        daily_review,
        "run_staged_maintenance",
        lambda **_kwargs: calls.append("staged"),
        raising=False,
    )
    monkeypatch.setattr(daily_review.tracker, "run_tracker", _fake_tracker_run)

    result = daily_review.run_daily_review(_args(tmp_path))

    assert calls == ["tracker"]
    assert result["maintenance_preflight"]["status"] == "skipped"
    assert result["maintenance_preflight"]["maintenance_status"] == "skipped_same_day_unavailable"
    assert result["maintenance_preflight"]["parquet_canonical_status"] == "healthy"
    assert result["maintenance_preflight"]["decision_data_status"] == "sufficient_limited"
    manifest = json.loads((tmp_path / "record" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["maintenance_preflight"]["latest_healthy_snapshot"]["snapshot_id"] == "snap-1"
    assert manifest["maintenance_preflight"]["staged_progress"] == {}


def test_daily_review_runs_configured_staged_batches_when_stable_target_has_gaps(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    FakePreflightDownloader.probe = {"applicable": False}
    FakePreflightDownloader.completeness = {
        "complete": False,
        "coverage_ratio": 0.82,
        "blocking_incomplete_count": 10,
        "latest_trade_date": "20260316",
        "strict_trade_date": "20260316",
        "stable_trade_date": "20260315",
        "effective_target_trade_date": "20260316",
        "categories": {},
    }

    monkeypatch.setattr(daily_review, "CNFullMarketDownloader", FakePreflightDownloader, raising=False)
    monkeypatch.setattr(
        daily_review,
        "run_storage_validate",
        lambda **_kwargs: {
            "status": "passed",
            "snapshot_id": "snap-2",
            "latest_complete_trade_date": "20260315",
            "manifest_path": "/tmp/manifest.json",
            "blockers": [],
        },
        raising=False,
    )

    def _fake_staged(**kwargs):
        calls.append(kwargs)
        return {
            "status": "running",
            "maintenance_status": "running",
            "run_id": "run-1",
            "run_dir": str(tmp_path / "_maintenance_runs" / "run-1"),
            "progress_summary": {
                "status": "running",
                "remaining_batches": 4,
                "failed_symbols": ["000001.SZ"],
            },
            "failed_symbols": ["000001.SZ"],
        }

    monkeypatch.setattr(daily_review, "run_staged_maintenance", _fake_staged, raising=False)
    monkeypatch.setattr(daily_review.tracker, "run_tracker", lambda _args: {"timestamp": "run-2", "run_dir": ""})

    result = daily_review.run_daily_review(_args(tmp_path))

    assert calls and calls[0]["max_batches_per_run"] == 200
    assert calls[0]["batch_size"] == 200
    assert calls[0]["resume"] is True
    assert result["maintenance_preflight"]["maintenance_status"] == "running"
    assert result["maintenance_preflight"]["staged_progress"]["remaining_batches"] == 4
    assert result["maintenance_preflight"]["failed_symbols"] == ["000001.SZ"]


def test_daily_review_maintenance_failure_does_not_block_tracker(monkeypatch, tmp_path):
    calls: list[str] = []

    def _fake_probe_failure(**_kwargs):
        calls.append("preflight")
        raise RuntimeError("upstream not ready")

    def _fake_tracker_run(args):
        calls.append("tracker")
        return {"timestamp": "run-2", "run_dir": str(tmp_path / "missing-record")}

    monkeypatch.setattr(daily_review, "CNFullMarketDownloader", _fake_probe_failure, raising=False)
    monkeypatch.setattr(
        daily_review,
        "run_storage_validate",
        lambda **_kwargs: {"status": "passed", "snapshot_id": "snap-3", "blockers": []},
        raising=False,
    )
    monkeypatch.setattr(daily_review.tracker, "run_tracker", _fake_tracker_run)

    result = daily_review.run_daily_review(_args(tmp_path))

    assert calls == ["preflight", "tracker"]
    assert result["maintenance_preflight"]["status"] == "failed_non_blocking"
    assert "upstream not ready" in result["maintenance_preflight"]["error"]


def test_daily_review_forwards_market_metrics_prewarm_debug_flag(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    args = _args(tmp_path)
    args.skip_market_metrics_prewarm = True

    monkeypatch.setattr(daily_review, "CNFullMarketDownloader", FakePreflightDownloader, raising=False)
    monkeypatch.setattr(
        daily_review,
        "run_storage_validate",
        lambda **_kwargs: {"status": "passed", "snapshot_id": "snap-4", "blockers": []},
        raising=False,
    )
    monkeypatch.setattr(
        daily_review,
        "run_staged_maintenance",
        lambda **_kwargs: {"status": "complete", "progress_summary": {"status": "complete"}},
        raising=False,
    )

    def _fake_tracker_run(tracker_args):
        captured["skip_market_metrics_prewarm"] = tracker_args.skip_market_metrics_prewarm
        captured["manual_ledger_parquet_only"] = (
            tracker_args.manual_ledger_parquet_only
        )
        return {
            "timestamp": "run-3",
            "run_dir": str(tmp_path / "missing-record"),
            "market_metrics_prewarm": {"status": "skipped"},
        }

    monkeypatch.setattr(daily_review.tracker, "run_tracker", _fake_tracker_run)

    result = daily_review.run_daily_review(args)

    assert captured["skip_market_metrics_prewarm"] is True
    assert captured["manual_ledger_parquet_only"] is True
    assert result["full_market_metrics_cache"]["status"] == "skipped"
