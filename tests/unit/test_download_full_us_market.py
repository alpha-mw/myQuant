from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import datetime

import pandas as pd


def _load_module(monkeypatch):
    fake_tushare = types.SimpleNamespace(pro_api=lambda token: object())
    monkeypatch.setitem(sys.modules, "tushare", fake_tushare)
    module_name = "quant_investor.market.download_us"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_detect_latest_available_trade_date_ignores_intraday_current_session(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_now_new_york", lambda: datetime(2026, 3, 30, 12, 0, 0))

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **_kwargs):
            return pd.DataFrame(
                {"Close": [560.0, 562.0]},
                index=pd.to_datetime(["2026-03-27", "2026-03-30"]),
            )

    monkeypatch.setattr(module, "yf", types.SimpleNamespace(Ticker=lambda symbol: FakeTicker(symbol)))

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)

    assert downloader.detect_latest_available_trade_date() == "2026-03-27"


def test_load_universe_normalizes_aliases_and_dedupes_categories(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)

    universe_path = tmp_path / "complete_us_universe.json"
    universe_path.write_text(
        json.dumps(
            {
                "large_cap": ["BRK.B", "AAPL"],
                "mid_cap": ["AAPL", "BF.B", "BF-B"],
                "small_cap": ["bf.b", "TSLA"],
            }
        ),
        encoding="utf-8",
    )

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path / "market"), years=3)
    universe = downloader.load_universe(str(universe_path))

    assert universe["large_cap"] == ["BRK-B", "AAPL"]
    assert universe["mid_cap"] == ["BF-B"]
    assert universe["small_cap"] == ["TSLA"]
    assert universe["all"] == ["BRK-B", "AAPL", "BF-B", "TSLA"]
    assert universe["stats"]["total_unique"] == 4


def test_build_completeness_report_detects_alias_missing_and_stale(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)
    large_dir = tmp_path / "large_cap"
    large_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"Date": "2026-03-27", "Close": 100.0}]).to_csv(large_dir / "AAPL.csv", index=False)
    pd.DataFrame([{"Date": "2026-03-26", "Close": 200.0}]).to_csv(large_dir / "BRK-B.csv", index=False)

    report = downloader.build_completeness_report(
        universe={
            "large_cap": ["AAPL", "BRK.B", "BF.B"],
            "mid_cap": [],
            "small_cap": [],
        },
        categories=["large_cap"],
        required_latest_trade_date="2026-03-27",
    )

    assert report["complete"] is False
    assert report["blocking_incomplete_count"] == 2
    assert report["categories"]["large_cap"]["blocking_missing_symbols"] == ["BF-B"]
    assert report["categories"]["large_cap"]["blocking_stale_symbols"] == [
        {"symbol": "BRK-B", "latest_local_date": "2026-03-26"}
    ]


def test_download_stock_uses_alias_candidates_for_yfinance(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)

    calls: list[str] = []

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **_kwargs):
            calls.append(self.symbol)
            if self.symbol != "BF-B":
                return pd.DataFrame()
            frame = pd.DataFrame(
                {
                    "Open": [25.0, 25.5],
                    "High": [26.0, 26.2],
                    "Low": [24.8, 25.1],
                    "Close": [25.7, 26.1],
                    "Volume": [1000, 1200],
                },
                index=pd.to_datetime(["2026-03-26", "2026-03-27"]),
            )
            frame.index.name = "Date"
            return frame

    monkeypatch.setattr(module, "yf", types.SimpleNamespace(Ticker=lambda symbol: FakeTicker(symbol)))

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)
    result = downloader.download_stock(
        "BF.B",
        "large_cap",
        force_refresh=True,
        required_latest_trade_date="2026-03-27",
    )

    assert result["status"] == "success"
    assert result["provider_symbol"] == "BF-B"
    assert calls[0] == "BF-B"
    assert (tmp_path / "large_cap" / "BF-B.csv").exists()


def test_download_all_respects_selected_categories_and_force_refresh_map(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)
    monkeypatch.setattr(downloader, "detect_latest_available_trade_date", lambda: "2026-03-27")

    captured: list[dict[str, object]] = []

    def _fake_download_category(
        symbols,
        category,
        force_refresh_symbols=None,
        required_latest_trade_date=None,
    ):
        captured.append(
            {
                "symbols": list(symbols),
                "category": category,
                "force_refresh_symbols": list(force_refresh_symbols or []),
                "required_latest_trade_date": required_latest_trade_date,
            }
        )
        return [{"symbol": "BF-B", "status": "updated"}]

    monkeypatch.setattr(downloader, "download_category", _fake_download_category)

    downloader.download_all(
        universe={
            "large_cap": ["AAPL"],
            "mid_cap": ["BF-B"],
            "small_cap": ["PLTR"],
        },
        categories=["mid_cap"],
        force_refresh_by_category={"mid_cap": ["BF-B"]},
    )

    assert captured == [
        {
            "symbols": ["BF-B"],
            "category": "mid_cap",
            "force_refresh_symbols": ["BF-B"],
            "required_latest_trade_date": "2026-03-27",
        }
    ]
