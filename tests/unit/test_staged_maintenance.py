from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import quant_investor.market.staged_maintenance as staged


def _category_payload(symbols: list[str], *, coverage_ratio: float = 0.0) -> dict[str, Any]:
    return {
        "expected": len(symbols),
        "latest_trade_date": "20260316",
        "blocking_missing_symbols": list(symbols),
        "blocking_stale_symbols": [],
        "blocking_unreadable_symbols": [],
        "blocking_incomplete_count": len(symbols),
        "coverage_complete_count": 0,
        "coverage_ratio": coverage_ratio,
    }


def _report(symbols: list[str], *, coverage_ratio: float = 0.0, complete: bool = False) -> dict[str, Any]:
    return {
        "complete": complete,
        "blocking_incomplete_count": 0 if complete else len(symbols),
        "categories_checked": ["full_a"],
        "categories": {"full_a": _category_payload(symbols, coverage_ratio=coverage_ratio)},
        "latest_trade_date": "20260316",
        "strict_trade_date": "20260316",
        "stable_trade_date": "20260315",
        "effective_target_trade_date": "20260316",
        "freshness_mode": "strict",
        "coverage_ratio": 1.0 if complete else coverage_ratio,
        "coverage_complete_count": 100 if complete else int(coverage_ratio * 100),
        "expected_scope_count": 100,
        "coverage_threshold": 0.95,
        "resolver": {},
    }


def _multi_category_report(categories: dict[str, list[str]]) -> dict[str, Any]:
    unique_symbols = sorted({symbol for symbols in categories.values() for symbol in symbols})
    return {
        "complete": False,
        "blocking_incomplete_count": len(unique_symbols),
        "categories_checked": list(categories),
        "categories": {category: _category_payload(symbols) for category, symbols in categories.items()},
        "latest_trade_date": "20260316",
        "strict_trade_date": "20260316",
        "stable_trade_date": "20260315",
        "effective_target_trade_date": "20260316",
        "freshness_mode": "strict",
        "coverage_ratio": 0.0,
        "coverage_complete_count": 0,
        "expected_scope_count": len(unique_symbols),
        "coverage_threshold": 0.95,
        "resolver": {},
    }


class FakeDownloader:
    reports: list[dict[str, Any]] = []
    same_day_probe: dict[str, Any] = {"applicable": False}
    download_status_by_symbol: dict[str, str] = {}
    download_calls: list[dict[str, Any]] = []
    last: "FakeDownloader | None" = None

    REQUESTS_PER_STOCK = 2
    REQUESTS_PER_MINUTE_BUDGET = 500

    def __init__(self, *, data_dir: str | None = None, years: int = 3, max_workers: int = 4, batch_size: int = 50):
        self.data_dir = str(data_dir or "")
        self.years = years
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.strict_trade_date = "20260316"
        self.stable_trade_date = "20260315"
        self.latest_trade_date = "20260316"
        self.freshness_mode = "strict"
        self.coverage_threshold = 0.95
        self.start_date = datetime(2023, 3, 16)
        self.end_date = datetime(2026, 3, 16)
        self.stats = {"total": 0, "updated": 0, "cached": 0, "stale_cached": 0, "failed": 0}
        FakeDownloader.last = self

    def load_components(self) -> dict[str, Any]:
        return {
            "full_a": [f"00000{i}.SZ" for i in range(1, 10)],
            "hs300": [],
            "zz500": [],
            "zz1000": [],
            "stats": {"total_unique": 9},
        }

    def _resolve_target_categories(self, _components: dict[str, Any], categories: list[str] | None) -> list[str]:
        return categories or ["full_a"]

    def _probe_strict_same_day_close_availability(self, **_kwargs: Any) -> dict[str, Any]:
        return deepcopy(FakeDownloader.same_day_probe)

    def build_completeness_report(self, **_kwargs: Any) -> dict[str, Any]:
        if FakeDownloader.reports:
            return deepcopy(FakeDownloader.reports[0])
        return _report([])

    def _collect_blocking_symbols(self, category_report: dict[str, Any]) -> list[str]:
        symbols = list(category_report.get("blocking_missing_symbols", []))
        symbols.extend(item["symbol"] for item in category_report.get("blocking_stale_symbols", []))
        symbols.extend(item["symbol"] for item in category_report.get("blocking_unreadable_symbols", []))
        return symbols

    def download_category(
        self,
        symbols: list[str],
        category: str,
        target_trade_date: str | None = None,
        round_control: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del round_control
        FakeDownloader.download_calls.append(
            {"category": category, "symbols": list(symbols), "target_trade_date": target_trade_date}
        )
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            status = FakeDownloader.download_status_by_symbol.get(symbol, "updated")
            self.stats["total"] += 1
            self.stats[status if status in self.stats else "failed"] += 1
            rows.append(
                {
                    "symbol": symbol,
                    "category": category,
                    "status": status,
                    "local_status": "up_to_date" if status == "updated" else "missing",
                    "latest_trade_date": target_trade_date or self.latest_trade_date,
                    "latest_local_date": target_trade_date or self.latest_trade_date if status == "updated" else "",
                    "error": None if status == "updated" else "boom",
                    "api_calls": self.REQUESTS_PER_STOCK,
                }
            )
        return rows


@pytest.fixture(autouse=True)
def _fake_downloader(monkeypatch):
    FakeDownloader.reports = []
    FakeDownloader.same_day_probe = {"applicable": False}
    FakeDownloader.download_status_by_symbol = {}
    FakeDownloader.download_calls = []
    FakeDownloader.last = None
    monkeypatch.setattr(staged, "CNFullMarketDownloader", FakeDownloader)


def _read_batches(run_dir: Path) -> list[dict[str, str]]:
    payload = json.loads((run_dir / "batches.json").read_text(encoding="utf-8"))
    return list(payload["batches"])


def test_staged_maintenance_creates_run_and_processes_only_configured_batches(tmp_path):
    FakeDownloader.reports = [_report(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"])]

    result = staged.run_staged_maintenance(
        market="CN",
        categories=["full_a"],
        data_dir=str(tmp_path),
        batch_size=2,
        max_batches_per_run=1,
    )

    run_dir = Path(result["run_dir"])
    progress = json.loads((run_dir / "progress_summary.json").read_text(encoding="utf-8"))
    batches = _read_batches(run_dir)

    assert progress["total_symbols"] == 5
    assert progress["batch_size"] == 2
    assert progress["completed_batches"] == 1
    assert progress["remaining_batches"] == 2
    assert progress["status"] == "running"
    assert progress["complete"] is False
    assert [row["status"] for row in batches] == ["completed", "pending", "pending"]
    assert FakeDownloader.download_calls == [
        {"category": "full_a", "symbols": ["000001.SZ", "000002.SZ"], "target_trade_date": "20260316"}
    ]
    assert (run_dir / "stage_plan.json").exists()
    assert (run_dir / "batch_0001" / "manifest.json").exists()


def test_staged_maintenance_dedupes_symbols_across_overlapping_categories(tmp_path):
    FakeDownloader.reports = [
        _multi_category_report(
            {
                "full_a": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "hs300": ["000001.SZ", "000002.SZ"],
                "zz500": ["000003.SZ"],
                "zz1000": ["000004.SZ"],
            }
        )
    ]

    result = staged.run_staged_maintenance(
        market="CN",
        categories=["full_a", "hs300", "zz500", "zz1000"],
        data_dir=str(tmp_path),
        batch_size=10,
        max_batches_per_run=0,
    )

    run_dir = Path(result["run_dir"])
    progress = json.loads((run_dir / "progress_summary.json").read_text(encoding="utf-8"))
    stage_plan = json.loads((run_dir / "stage_plan.json").read_text(encoding="utf-8"))
    batches = _read_batches(run_dir)
    queued_symbols = [
        symbol
        for row in batches
        for symbol in row["symbols"]
        if symbol
    ]

    assert stage_plan["total_symbols"] == 4
    assert progress["total_symbols"] == 4
    assert queued_symbols == ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    assert len(queued_symbols) == len(set(queued_symbols))
    assert [row["category"] for row in batches] == ["full_a"]


def test_staged_maintenance_prefers_downloader_daily_batch_api(monkeypatch, tmp_path):
    class BatchOnlyDownloader(FakeDownloader):
        batch_calls: list[dict[str, Any]] = []

        def download_category(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            raise AssertionError("staged maintenance should use download_daily_batch when available")

        def download_daily_batch(
            self,
            symbols: list[str],
            category: str,
            target_trade_date: str | None = None,
        ) -> list[dict[str, Any]]:
            BatchOnlyDownloader.batch_calls.append(
                {"category": category, "symbols": list(symbols), "target_trade_date": target_trade_date}
            )
            return [
                {
                    "symbol": symbol,
                    "category": category,
                    "status": "updated",
                    "local_status": "up_to_date",
                    "latest_trade_date": target_trade_date,
                    "latest_local_date": target_trade_date,
                    "error": None,
                    "api_calls": 0,
                }
                for symbol in symbols
            ]

    FakeDownloader.reports = [_report(["000001.SZ", "000002.SZ"])]
    BatchOnlyDownloader.batch_calls = []
    monkeypatch.setattr(staged, "CNFullMarketDownloader", BatchOnlyDownloader)

    staged.run_staged_maintenance(
        market="CN",
        categories=["full_a"],
        data_dir=str(tmp_path),
        batch_size=2,
        max_batches_per_run=1,
    )

    assert BatchOnlyDownloader.batch_calls == [
        {"category": "full_a", "symbols": ["000001.SZ", "000002.SZ"], "target_trade_date": "20260316"}
    ]


def test_staged_maintenance_resume_skips_completed_batches(tmp_path):
    FakeDownloader.reports = [_report(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"])]
    first = staged.run_staged_maintenance(
        market="CN",
        categories=["full_a"],
        data_dir=str(tmp_path),
        batch_size=2,
        max_batches_per_run=1,
    )
    FakeDownloader.download_calls = []

    second = staged.run_staged_maintenance(
        market="CN",
        categories=["full_a"],
        data_dir=str(tmp_path),
        batch_size=2,
        max_batches_per_run=1,
        resume=True,
    )

    assert second["run_id"] == first["run_id"]
    assert FakeDownloader.download_calls == [
        {"category": "full_a", "symbols": ["000003.SZ", "000004.SZ"], "target_trade_date": "20260316"}
    ]
    assert json.loads((Path(second["run_dir"]) / "progress_summary.json").read_text(encoding="utf-8"))[
        "completed_batches"
    ] == 2


def test_staged_maintenance_records_failed_batch_manifest(tmp_path):
    FakeDownloader.reports = [_report(["000001.SZ", "000002.SZ"])]
    FakeDownloader.download_status_by_symbol = {"000002.SZ": "failed"}

    result = staged.run_staged_maintenance(
        market="CN",
        categories=["full_a"],
        data_dir=str(tmp_path),
        batch_size=2,
        max_batches_per_run=1,
    )

    run_dir = Path(result["run_dir"])
    manifest = json.loads((run_dir / "batch_0001" / "manifest.json").read_text(encoding="utf-8"))
    failed = json.loads((run_dir / "failed_batches.json").read_text(encoding="utf-8"))
    progress = json.loads((run_dir / "progress_summary.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "incomplete"
    assert manifest["failed_symbols"] == ["000002.SZ"]
    assert failed["failed_batches"][0]["batch_id"] == "0001"
    assert progress["failed_symbols"] == ["000002.SZ"]


def test_staged_maintenance_95pct_decision_data_is_not_complete(tmp_path):
    FakeDownloader.reports = [_report(["000001.SZ"], coverage_ratio=0.96)]

    result = staged.run_staged_maintenance(
        market="CN",
        categories=["full_a"],
        data_dir=str(tmp_path),
        batch_size=10,
        max_batches_per_run=0,
        min_symbol_success_rate=0.95,
    )

    progress = json.loads((Path(result["run_dir"]) / "progress_summary.json").read_text(encoding="utf-8"))
    assert progress["decision_data_sufficient"] is True
    assert progress["decision_data_status"] == "sufficient_limited"
    assert progress["complete"] is False
    assert progress["status"] == "incomplete"


def test_staged_maintenance_daily_window_narrows_downloader_dates(tmp_path):
    FakeDownloader.reports = [_report(["000001.SZ"])]

    staged.run_staged_maintenance(
        market="CN",
        categories=["full_a"],
        data_dir=str(tmp_path),
        batch_size=1,
        max_batches_per_run=1,
        target_date="20260316",
        daily_window=True,
    )

    assert FakeDownloader.last is not None
    assert FakeDownloader.last.start_date.strftime("%Y%m%d") == "20260306"
    assert FakeDownloader.last.end_date.strftime("%Y%m%d") == "20260316"
