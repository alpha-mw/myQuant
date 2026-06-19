from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_investor.market.config import normalize_universe
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.market_data_store import MarketDataStore


def test_us_normalize_universe_uses_full_us():
    assert normalize_universe("US", "full_us") == ["full_us"]
    assert normalize_universe("US", "full_market") == ["full_us"]
    assert normalize_universe("US", "all_us") == ["full_us"]


def test_us_load_universe_canonicalizes_full_us(tmp_path, monkeypatch):
    module = importlib.import_module("quant_investor.market.download_us")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "TUSHARE_AVAILABLE", False)

    universe_path = tmp_path / "complete_us_universe.json"
    market_cap_cache = tmp_path / "market_caps.json"
    universe_path.write_text(
        json.dumps(
            {
                "large_cap": ["AAPL", "MSFT"],
                "mid_cap": ["IBM"],
                "small_cap": ["F", "GM"],
                "stats": {"total_unique": 5},
            }
        ),
        encoding="utf-8",
    )
    market_cap_cache.write_text(
        json.dumps(
            {
                "symbols": {
                    "AAPL": {"market_cap": 3_000_000_000_000},
                    "MSFT": {"market_cap": 2_500_000_000_000},
                    "IBM": {"market_cap": 180_000_000_000},
                    "F": {"market_cap": 5_000_000_000},
                    "GM": {"market_cap": 45_000_000_000},
                }
            }
        ),
        encoding="utf-8",
    )

    downloader = module.FullMarketDownloader(
        data_dir=str(tmp_path),
        years=3,
        market_cap_cache_file=str(market_cap_cache),
    )
    universe = downloader.load_universe(universe_file=str(universe_path))

    assert universe["full_us"] == ["AAPL", "MSFT", "IBM", "GM"]
    assert universe["full_market"] == universe["full_us"]
    assert universe["all_us"] == universe["full_us"]
    assert universe["small_cap"] == ["GM"]
    assert universe["stats"]["full_us"] == 4
    assert universe["metadata"]["market_cap_filter"]["threshold_usd"] == 10_000_000_000
    assert universe["metadata"]["market_cap_filter"]["below_threshold_count"] == 1


def _write_us_parquet(data_root: Path, symbol: str, end: str, close: float = 1.0) -> None:
    dates = pd.bdate_range(end=end, periods=220)
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1000,
        }
    )
    MarketDataStore(market="US", data_root=data_root).write_full_history_bars(
        frame,
        source="test_us_universe_full",
    )


def _read_us_frame(data_root: Path, symbol: str) -> pd.DataFrame:
    return MarketDataReader(market="US", data_root=data_root).read_symbol_frame(symbol).frame


def test_us_download_stock_refreshes_stale_cache(tmp_path, monkeypatch):
    module = importlib.import_module("quant_investor.market.download_us")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "TUSHARE_AVAILABLE", False)
    monkeypatch.delenv("MYQUANT_US_PRICE_PROVIDER", raising=False)

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)
    downloader.end_date = datetime(2026, 5, 27)
    _write_us_parquet(tmp_path, "AAPL", "2026-05-22", close=1.0)

    fresh = pd.DataFrame(
        {
            "Date": ["2026-05-25", "2026-05-26"],
            "Open": [2.0, 3.0],
            "High": [2.0, 3.0],
            "Low": [2.0, 3.0],
            "Close": [2.0, 3.0],
            "Volume": [1000, 1000],
        }
    )
    monkeypatch.setattr(downloader, "_download_from_tushare", lambda _symbol: None)
    monkeypatch.setattr(downloader, "_download_from_yfinance", lambda _symbol: fresh)

    result = downloader.download_stock("AAPL", "full_us")

    assert result["status"] == "success"
    assert result["latest_date"] == "2026-05-26"
    assert _read_us_frame(tmp_path, "AAPL")["trade_date"].iloc[-1] == "20260526"


def test_us_download_stock_keeps_cache_when_provider_is_stale(tmp_path, monkeypatch):
    module = importlib.import_module("quant_investor.market.download_us")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "TUSHARE_AVAILABLE", False)
    monkeypatch.delenv("MYQUANT_US_PRICE_PROVIDER", raising=False)

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)
    downloader.end_date = datetime(2026, 5, 27)
    _write_us_parquet(tmp_path, "AAPL", "2026-05-22", close=1.0)

    stale = pd.DataFrame(
        {
            "Date": ["2026-05-21", "2026-05-22"],
            "Open": [9.0, 9.0],
            "High": [9.0, 9.0],
            "Low": [9.0, 9.0],
            "Close": [9.0, 9.0],
            "Volume": [1000, 1000],
        }
    )
    monkeypatch.setattr(downloader, "_download_from_tushare", lambda _symbol: None)
    monkeypatch.setattr(downloader, "_download_from_yfinance", lambda _symbol: stale)

    result = downloader.download_stock("AAPL", "full_us")

    assert result["status"] == "source_stale"
    assert result["latest_date"] == "2026-05-22"
    assert _read_us_frame(tmp_path, "AAPL")["close"].iloc[-1] == 1.0


def test_us_download_stock_uses_yfinance_before_tushare_by_default(tmp_path, monkeypatch):
    module = importlib.import_module("quant_investor.market.download_us")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "TUSHARE_AVAILABLE", False)
    monkeypatch.delenv("MYQUANT_US_PRICE_PROVIDER", raising=False)

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)
    downloader.end_date = datetime(2026, 5, 27)
    fresh = pd.DataFrame(
        {
            "Date": ["2026-05-26"],
            "Open": [3.0],
            "High": [3.0],
            "Low": [3.0],
            "Close": [3.0],
            "Volume": [1000],
        }
    )
    calls = []
    monkeypatch.setattr(
        downloader,
        "_download_from_yfinance",
        lambda _symbol: calls.append("yfinance") or fresh,
    )
    monkeypatch.setattr(
        downloader,
        "_download_from_akshare",
        lambda _symbol: calls.append("akshare") or None,
    )
    monkeypatch.setattr(
        downloader,
        "_download_from_tushare",
        lambda _symbol: calls.append("tushare") or None,
    )

    result = downloader.download_stock("AAPL", "full_us")

    assert result["status"] == "success"
    assert result["source"] == "yfinance"
    assert result["attempted_sources"] == ["yfinance"]
    assert calls == ["yfinance"]


def test_us_download_stock_can_force_tushare_first(tmp_path, monkeypatch):
    module = importlib.import_module("quant_investor.market.download_us")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "TUSHARE_AVAILABLE", False)
    monkeypatch.setenv("MYQUANT_US_PRICE_PROVIDER", "tushare")

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)
    downloader.end_date = datetime(2026, 5, 27)
    fresh = pd.DataFrame(
        {
            "Date": ["2026-05-26"],
            "Open": [3.0],
            "High": [3.0],
            "Low": [3.0],
            "Close": [3.0],
            "Volume": [1000],
        }
    )
    calls = []
    monkeypatch.setattr(
        downloader,
        "_download_from_tushare",
        lambda _symbol: calls.append("tushare") or fresh,
    )
    monkeypatch.setattr(
        downloader,
        "_download_from_yfinance",
        lambda _symbol: calls.append("yfinance") or fresh,
    )
    monkeypatch.setattr(
        downloader,
        "_download_from_akshare",
        lambda _symbol: calls.append("akshare") or None,
    )

    result = downloader.download_stock("AAPL", "full_us")

    assert result["status"] == "success"
    assert result["source"] == "tushare"
    assert result["attempted_sources"] == ["tushare"]
    assert calls == ["tushare"]


def test_us_download_category_accepts_forced_refresh_symbols(tmp_path, monkeypatch):
    module = importlib.import_module("quant_investor.market.download_us")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "TUSHARE_AVAILABLE", False)

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3, batch_size=2, max_workers=1)
    calls = []

    def fake_download_stock(symbol: str, category: str, force_refresh: bool = False):
        calls.append((symbol, category, force_refresh))
        return {
            "symbol": symbol,
            "category": category,
            "status": "success",
            "records": 1,
            "error": None,
        }

    monkeypatch.setattr(downloader, "download_stock", fake_download_stock)

    results = downloader.download_category(
        ["AAPL", "MSFT", "GOOG"],
        "full_us",
        force_refresh_symbols={"MSFT"},
    )

    assert len(results) == 3
    assert sorted(calls) == [
        ("AAPL", "full_us", False),
        ("GOOG", "full_us", False),
        ("MSFT", "full_us", True),
    ]


def test_us_tushare_quota_detection_handles_frequency_limit_text(tmp_path, monkeypatch):
    module = importlib.import_module("quant_investor.market.download_us")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "TUSHARE_AVAILABLE", False)

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)

    assert downloader._is_tushare_quota_error("抱歉，您访问接口(us_daily)频率超限(2次/分钟)")


def test_us_local_symbol_listing_applies_market_cap_filter(tmp_path, monkeypatch):
    market_cap_cache = tmp_path / "market_caps.json"
    market_cap_cache.write_text(
        json.dumps(
            {
                "symbols": {
                    "AAPL": {"market_cap": 3_000_000_000_000},
                    "TINY": {"market_cap": 2_000_000_000},
                    "GM": {"market_cap": 45_000_000_000},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MYQUANT_US_MARKET_CAP_CACHE_FILE", str(market_cap_cache))

    _write_us_parquet(tmp_path, "AAPL", "2026-05-26")
    _write_us_parquet(tmp_path, "TINY", "2026-05-26")
    _write_us_parquet(tmp_path, "GM", "2026-05-26")

    analyze_module = importlib.import_module("quant_investor.market.analyze")

    assert analyze_module.get_all_local_symbols("full_us", market="US", data_dir=str(tmp_path)) == [
        "AAPL",
        "GM",
    ]


def test_us_market_cap_threshold_env_accepts_numeric_string(monkeypatch):
    module = importlib.import_module("quant_investor.market.us_market_cap_filter")
    monkeypatch.setenv("MYQUANT_US_MIN_MARKET_CAP_USD", "25,000,000,000")

    assert module.get_us_min_market_cap_usd() == 25_000_000_000


def test_us_data_snapshot_uses_filtered_inventory_when_symbols_not_requested(tmp_path, monkeypatch):
    market_cap_cache = tmp_path / "market_caps.json"
    market_cap_cache.write_text(
        json.dumps(
            {
                "symbols": {
                    "AAPL": {"market_cap": 3_000_000_000_000},
                    "TINY": {"market_cap": 2_000_000_000},
                    "GM": {"market_cap": 45_000_000_000},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MYQUANT_US_MARKET_CAP_CACHE_FILE", str(market_cap_cache))
    _write_us_parquet(tmp_path, "AAPL", "2026-05-26")
    _write_us_parquet(tmp_path, "TINY", "2026-05-25")
    _write_us_parquet(tmp_path, "GM", "2026-05-26")

    snapshot_module = importlib.import_module("quant_investor.market.data_snapshot")
    snapshot = snapshot_module.build_market_data_snapshot(
        market="US",
        universe="full_us",
        data_dir=tmp_path,
    )

    assert snapshot["local_latest_trade_date"] == "20260526"
    assert snapshot["observed_symbol_count"] == 2
    assert snapshot["inventory_symbol_count"] == 2
    assert "20260526" in snapshot["summary_text"]


def test_us_stock_names_load_from_market_cap_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names_path = tmp_path / "data/us_universe/stock_names.json"
    market_cap_cache = tmp_path / "data/us_universe/us_market_caps.json"
    market_cap_cache.parent.mkdir(parents=True)
    market_cap_cache.write_text(
        json.dumps(
            {
                "symbols": {
                    "BCE": {
                        "market_cap": 23_000_000_000,
                        "name": "BCE Inc.",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    analyze_module = importlib.import_module("quant_investor.market.analyze")
    analyze_module._STOCK_NAME_CACHE["US"] = {}
    names = analyze_module.load_stock_names("US", refresh=True)

    assert names["BCE"] == "BCE Inc."
    assert json.loads(names_path.read_text(encoding="utf-8"))["BCE"] == "BCE Inc."
