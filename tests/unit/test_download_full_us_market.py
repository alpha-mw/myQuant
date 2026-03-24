"""
美股下载器单元测试。
"""

from __future__ import annotations

import importlib

import pandas as pd


def _load_module():
    module_name = "quant_investor.market.download_us"
    return importlib.import_module(module_name)


def test_build_completeness_report_detects_missing_and_stale(tmp_path):
    module = _load_module()
    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=1)

    large_dir = tmp_path / "large_cap"
    mid_dir = tmp_path / "mid_cap"
    small_dir = tmp_path / "small_cap"
    large_dir.mkdir(parents=True, exist_ok=True)
    mid_dir.mkdir(parents=True, exist_ok=True)
    small_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"Date": ["2026-03-19", "2026-03-20"], "Close": [10.0, 10.5]}).to_csv(
        large_dir / "AAA.csv", index=False
    )
    pd.DataFrame({"Date": ["2026-03-18", "2026-03-19"], "Close": [9.0, 9.1]}).to_csv(
        large_dir / "BBB.csv", index=False
    )
    pd.DataFrame({"Date": ["2026-03-20"], "Close": [20.0]}).to_csv(
        mid_dir / "CCC.csv", index=False
    )

    report = downloader.build_completeness_report(
        universe={
            "large_cap": ["AAA", "BBB", "MISSING"],
            "mid_cap": ["CCC"],
            "small_cap": [],
            "stats": {"total_unique": 4},
        }
    )

    assert report["latest_trade_date"] == "2026-03-20"
    assert report["complete"] is False
    assert report["blocking_incomplete_count"] == 2
    assert report["categories"]["large_cap"]["blocking_missing_symbols"] == ["MISSING"]
    assert report["categories"]["large_cap"]["blocking_stale_symbols"] == [
        {"symbol": "BBB", "latest_local_date": "2026-03-19"}
    ]


def test_download_stock_force_refresh_bypasses_cache(monkeypatch, tmp_path):
    module = _load_module()
    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=1)

    large_dir = tmp_path / "large_cap"
    large_dir.mkdir(parents=True, exist_ok=True)
    filepath = large_dir / "AAA.csv"
    pd.DataFrame({"Date": [f"2026-01-{day:02d}" for day in range(1, 31)], "Close": list(range(30))}).to_csv(
        filepath,
        index=False,
    )

    monkeypatch.setattr(
        downloader,
        "_download_from_tushare",
        lambda symbol: pd.DataFrame({"Date": ["2026-03-20"], "Close": [99.0]}),
    )
    monkeypatch.setattr(downloader, "_download_from_yfinance", lambda symbol: None)

    result = downloader.download_stock("AAA", "large_cap", force_refresh=True)

    assert result["status"] == "success"
    refreshed = pd.read_csv(filepath)
    assert refreshed["Date"].iloc[-1] == "2026-03-20"
