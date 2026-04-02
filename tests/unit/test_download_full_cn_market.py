"""
下载器单元测试
"""

from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime, timedelta

import pandas as pd
import pytest


def _load_module(monkeypatch):
    """在无真实 tushare 依赖的测试环境中导入下载模块。"""
    fake_tushare = types.SimpleNamespace(pro_api=lambda token: object())
    monkeypatch.setitem(sys.modules, "tushare", fake_tushare)
    module_name = "quant_investor.market.download_cn"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "TUSHARE_TOKEN", "dummy-token")
    return module


class FakePro:
    """最小化 Tushare Pro 假实现。"""

    def __init__(self) -> None:
        self.daily_calls: list[tuple[str, str, str]] = []

    def trade_cal(self, exchange: str, start_date: str, end_date: str, is_open: str):
        return pd.DataFrame({"cal_date": ["20260314", "20260316"]})

    def daily(self, ts_code: str, start_date: str, end_date: str):
        self.daily_calls.append((ts_code, start_date, end_date))
        return pd.DataFrame(
            [
                {
                    "ts_code": ts_code,
                    "trade_date": "20260311",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "change": 0.2,
                    "pct_chg": 2.0,
                    "vol": 1000,
                    "amount": 10000,
                },
                {
                    "ts_code": ts_code,
                    "trade_date": "20260312",
                    "open": 10.2,
                    "high": 10.6,
                    "low": 10.0,
                    "close": 10.4,
                    "pre_close": 10.2,
                    "change": 0.2,
                    "pct_chg": 1.96,
                    "vol": 1200,
                    "amount": 12000,
                },
                {
                    "ts_code": ts_code,
                    "trade_date": "20260313",
                    "open": 10.4,
                    "high": 10.8,
                    "low": 10.3,
                    "close": 10.7,
                    "pre_close": 10.4,
                    "change": 0.3,
                    "pct_chg": 2.88,
                    "vol": 1500,
                    "amount": 15000,
                },
                {
                    "ts_code": ts_code,
                    "trade_date": "20260316",
                    "open": 10.7,
                    "high": 11.0,
                    "low": 10.6,
                    "close": 10.9,
                    "pre_close": 10.7,
                    "change": 0.2,
                    "pct_chg": 1.87,
                    "vol": 1600,
                    "amount": 16000,
                },
            ]
        )

    def adj_factor(self, ts_code: str, start_date: str, end_date: str):
        return pd.DataFrame(
            [
                {"trade_date": "20260311", "adj_factor": 1.0},
                {"trade_date": "20260312", "adj_factor": 1.0},
                {"trade_date": "20260313", "adj_factor": 1.0},
                {"trade_date": "20260316", "adj_factor": 1.0},
            ]
        )

    def suspend_d(self, **_kwargs):
        return pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"])


def test_download_stock_incremental_update(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    file_path = tmp_path / "hs300" / "000001.SZ.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    old_rows = []
    start = datetime(2025, 1, 1)
    for idx in range(250):
        trade_date = (start + timedelta(days=idx)).strftime("%Y-%m-%d")
        old_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 9.0,
                "high": 9.5,
                "low": 8.8,
                "close": 9.2,
                "pre_close": 9.1,
                "change": 0.1,
                "pct_chg": 1.0,
                "vol": 1000,
                "amount": 10000,
                "adj_factor": 1.0,
                "adj_close": 9.2,
                "adj_open": 9.0,
                "adj_high": 9.5,
                "adj_low": 8.8,
            }
        )
    old_rows.append(
        {
            "ts_code": "000001.SZ",
            "trade_date": "2026-03-12",
            "open": 10.2,
            "high": 10.6,
            "low": 10.0,
            "close": 10.4,
            "pre_close": 10.2,
            "change": 0.2,
            "pct_chg": 1.96,
            "vol": 1200,
            "amount": 12000,
            "adj_factor": 1.0,
            "adj_close": 10.4,
            "adj_open": 10.2,
            "adj_high": 10.6,
            "adj_low": 10.0,
        }
    )
    pd.DataFrame(old_rows).to_csv(file_path, index=False)

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "updated"
    assert fake_pro.daily_calls[-1] == ("000001.SZ", "20260311", "20260316")

    updated_df = pd.read_csv(file_path)
    assert updated_df["trade_date"].iloc[-1] == "2026-03-16"
    assert updated_df["trade_date"].nunique() == len(updated_df)


def test_download_stock_skips_when_file_is_latest(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    file_path = tmp_path / "hs300" / "000001.SZ.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    start = datetime(2025, 2, 1)
    for idx in range(250):
        trade_date = (start + timedelta(days=idx)).strftime("%Y-%m-%d")
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 9.0,
                "high": 9.5,
                "low": 8.8,
                "close": 9.2,
                "pre_close": 9.1,
                "change": 0.1,
                "pct_chg": 1.0,
                "vol": 1000,
                "amount": 10000,
                "adj_factor": 1.0,
                "adj_close": 9.2,
                "adj_open": 9.0,
                "adj_high": 9.5,
                "adj_low": 8.8,
            }
        )
    rows.append(
        {
            "ts_code": "000001.SZ",
            "trade_date": "2026-03-16",
            "open": 10.7,
            "high": 11.0,
            "low": 10.6,
            "close": 10.9,
            "pre_close": 10.7,
            "change": 0.2,
            "pct_chg": 1.87,
            "vol": 1600,
            "amount": 16000,
            "adj_factor": 1.0,
            "adj_close": 10.9,
            "adj_open": 10.7,
            "adj_high": 11.0,
            "adj_low": 10.6,
        }
    )
    pd.DataFrame(rows).to_csv(file_path, index=False)

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "cached"
    assert result["api_calls"] == 0
    assert fake_pro.daily_calls == []


def test_download_stock_falls_back_to_eastmoney_when_tushare_is_empty(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)

    class EmptyDailyPro(FakePro):
        def daily(self, ts_code: str, start_date: str, end_date: str):
            self.daily_calls.append((ts_code, start_date, end_date))
            return pd.DataFrame()

        def adj_factor(self, ts_code: str, start_date: str, end_date: str):
            return pd.DataFrame()

    fake_pro = EmptyDailyPro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)

    def _fake_request(symbol: str, start_date_str: str, end_date_str: str, *, fqt: str):
        assert symbol == "000001.SZ"
        assert start_date_str == downloader.start_date.strftime("%Y%m%d")
        assert end_date_str == downloader.latest_trade_date
        if fqt == "0":
            return [
                "2026-03-13,10.40,10.70,10.80,10.30,1500,15000000,4.81,2.88,0.30,0.50",
                "2026-03-16,10.70,10.90,11.00,10.60,1600,16000000,3.74,1.87,0.20,0.45",
            ]
        return [
            "2026-03-13,20.80,21.40,21.60,20.60,1500,15000000,4.81,2.88,0.60,0.50",
            "2026-03-16,21.40,21.80,22.00,21.20,1600,16000000,3.74,1.87,0.40,0.45",
        ]

    monkeypatch.setattr(downloader, "_request_eastmoney_kline", _fake_request)

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "success"
    assert fake_pro.daily_calls[-1] == (
        "000001.SZ",
        downloader.start_date.strftime("%Y%m%d"),
        downloader.latest_trade_date,
    )

    updated_df = pd.read_csv(tmp_path / "hs300" / "000001.SZ.csv")
    assert updated_df["trade_date"].iloc[-1] == "2026-03-16"
    assert updated_df["amount"].iloc[-1] == pytest.approx(16000.0)
    assert updated_df["adj_close"].iloc[-1] == pytest.approx(21.8)
    assert updated_df["adj_factor"].iloc[-1] == pytest.approx(2.0)


def test_download_stock_falls_back_to_yfinance_when_other_sources_fail(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)

    class EmptyDailyPro(FakePro):
        def daily(self, ts_code: str, start_date: str, end_date: str):
            self.daily_calls.append((ts_code, start_date, end_date))
            return pd.DataFrame()

        def adj_factor(self, ts_code: str, start_date: str, end_date: str):
            return pd.DataFrame()

    fake_pro = EmptyDailyPro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    monkeypatch.setattr(
        downloader,
        "_fetch_stock_frame_from_eastmoney",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("eastmoney down")),
    )

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **_kwargs):
            assert self.symbol == "000001.SZ"
            return pd.DataFrame(
                {
                    "Open": [10.0, 10.2],
                    "High": [10.4, 10.6],
                    "Low": [9.9, 10.1],
                    "Close": [10.2, 10.5],
                    "Adj Close": [20.4, 21.0],
                    "Volume": [1000, 1200],
                },
                index=pd.to_datetime(["2026-03-13", "2026-03-16"]),
            )

    monkeypatch.setattr(module, "yf", types.SimpleNamespace(Ticker=FakeTicker))

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "success"
    assert result["source"] == "yfinance"
    updated_df = pd.read_csv(tmp_path / "hs300" / "000001.SZ.csv")
    assert updated_df["trade_date"].tolist() == ["2026-03-13", "2026-03-16"]
    assert updated_df["amount"].iloc[-1] == pytest.approx(10.5 * 1200 / 1000.0)
    assert updated_df["adj_factor"].iloc[-1] == pytest.approx(2.0)
    assert updated_df["adj_close"].iloc[-1] == pytest.approx(21.0)


def test_download_category_uses_yfinance_batch_when_fallback_preferred(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    downloader._prefer_eastmoney_fallback = True

    def _fake_download(symbols, **_kwargs):
        frames = {}
        for idx, symbol in enumerate(symbols):
            frames[symbol] = pd.DataFrame(
                {
                    "Open": [10.0 + idx, 10.2 + idx],
                    "High": [10.4 + idx, 10.6 + idx],
                    "Low": [9.9 + idx, 10.1 + idx],
                    "Close": [10.2 + idx, 10.5 + idx],
                    "Adj Close": [10.2 + idx, 10.5 + idx],
                    "Volume": [1000 + idx, 1200 + idx],
                },
                index=pd.to_datetime(["2026-03-13", "2026-03-16"]),
            )
        return pd.concat(frames, axis=1)

    monkeypatch.setattr(module, "yf", types.SimpleNamespace(download=_fake_download))
    downloader.YFINANCE_BATCH_SIZE = 2

    results = downloader.download_category(["000001.SZ", "000002.SZ"], "hs300")

    assert [item["status"] for item in results] == ["success", "success"]
    assert [item["source"] for item in results] == ["yfinance", "yfinance"]
    first = pd.read_csv(tmp_path / "hs300" / "000001.SZ.csv")
    second = pd.read_csv(tmp_path / "hs300" / "000002.SZ.csv")
    assert first["trade_date"].iloc[-1] == "2026-03-16"
    assert second["trade_date"].iloc[-1] == "2026-03-16"


def test_download_category_only_sleeps_when_api_called(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    sleep_calls: list[float] = []

    results = iter(
        [
            {
                "symbol": "000001.SZ",
                "category": "hs300",
                "status": "cached",
                "records": 250,
                "api_calls": 0,
                "error": None,
            },
            {
                "symbol": "000002.SZ",
                "category": "hs300",
                "status": "updated",
                "records": 260,
                "api_calls": downloader.REQUESTS_PER_STOCK,
                "error": None,
            },
        ]
    )

    monkeypatch.setattr(downloader, "download_stock", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    downloader.download_category(["000001.SZ", "000002.SZ"], "hs300")

    expected_sleep = downloader.REQUESTS_PER_STOCK * 60 / downloader.REQUESTS_PER_MINUTE_BUDGET
    assert sleep_calls == [expected_sleep]


def test_build_completeness_report_detects_blocking_stale_symbols(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    latest_path = tmp_path / "hs300" / "000001.SZ.csv"
    stale_path = tmp_path / "hs300" / "000002.SZ.csv"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "trade_date": "2026-03-16",
                "close": 10.0,
            }
        ]
    ).to_csv(latest_path, index=False)
    pd.DataFrame(
        [
            {
                "trade_date": "2026-03-15",
                "close": 10.0,
            }
        ]
    ).to_csv(stale_path, index=False)

    components = {"hs300": ["000001.SZ", "000002.SZ"], "zz500": [], "zz1000": []}

    report = downloader.build_completeness_report(components=components)
    assert report["complete"] is False
    assert report["blocking_incomplete_count"] == 1
    assert report["categories"]["hs300"]["blocking_stale_symbols"] == [
        {"symbol": "000002.SZ", "latest_local_date": "20260315"}
    ]

    allowed_report = downloader.build_completeness_report(
        components=components,
        allowed_stale_symbols={"000002.SZ"},
    )
    assert allowed_report["complete"] is True
    assert allowed_report["blocking_incomplete_count"] == 0


def test_main_scopes_check_complete_to_selected_category(monkeypatch):
    module = _load_module(monkeypatch)
    captured: dict[str, object] = {}

    class FakeDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def load_components(self):
            return {
                "hs300": ["000001.SZ"],
                "zz500": ["000002.SZ"],
                "zz1000": ["000003.SZ"],
                "stats": {"total_unique": 3},
            }

        def build_completeness_report(self, components=None, allowed_stale_symbols=None, categories=None):
            captured["categories"] = categories
            return {
                "complete": True,
                "latest_trade_date": "20260316",
                "blocking_incomplete_count": 0,
                "categories": {
                    "hs300": {
                        "expected": 1,
                        "date_counts": {"20260316": 1},
                        "blocking_incomplete_count": 0,
                    }
                },
            }

        def _print_completeness_summary(self, completeness):
            captured["printed"] = True

    monkeypatch.setattr(module, "CNFullMarketDownloader", FakeDownloader)
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_cn.py", "--category", "hs300", "--check-complete"],
    )

    module.main()

    assert captured["categories"] == ["hs300"]
    assert captured["printed"] is True


def test_main_applies_retry_flags_to_selected_category(monkeypatch):
    module = _load_module(monkeypatch)
    captured: dict[str, object] = {}

    class FakeDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def load_components(self):
            return {
                "hs300": ["000001.SZ"],
                "zz500": ["000002.SZ"],
                "zz1000": ["000003.SZ"],
                "stats": {"total_unique": 3},
            }

        def download_all(
            self,
            components=None,
            max_rounds=1,
            fail_on_incomplete=False,
            allowed_stale_symbols=None,
            categories=None,
        ):
            captured["max_rounds"] = max_rounds
            captured["fail_on_incomplete"] = fail_on_incomplete
            captured["allowed_stale_symbols"] = allowed_stale_symbols
            captured["categories"] = categories
            return {}

    monkeypatch.setattr(module, "CNFullMarketDownloader", FakeDownloader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_cn.py",
            "--category",
            "hs300",
            "--max-rounds",
            "3",
            "--fail-on-incomplete",
            "--allowed-stale-symbols",
            "000001.SZ",
        ],
    )

    module.main()

    assert captured["max_rounds"] == 3
    assert captured["fail_on_incomplete"] is True
    assert captured["allowed_stale_symbols"] == ["000001.SZ"]
    assert captured["categories"] == ["hs300"]


def test_build_completeness_report_treats_latest_suspend_as_complete(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()

    def _fake_suspend_d(**_kwargs):
        return pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20260316",
                    "suspend_type": "S",
                }
            ]
        )

    fake_pro.suspend_d = _fake_suspend_d
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    latest_path = tmp_path / "hs300" / "000001.SZ.csv"
    stale_path = tmp_path / "hs300" / "000002.SZ.csv"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(latest_path, index=False)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 10.0}]).to_csv(stale_path, index=False)

    components = {"hs300": ["000001.SZ", "000002.SZ"], "zz500": [], "zz1000": []}

    report = downloader.build_completeness_report(components=components)
    assert report["complete"] is True
    assert report["blocking_incomplete_count"] == 0
    assert report["categories"]["hs300"]["suspended_stale_symbols"] == [
        {"symbol": "000002.SZ", "latest_local_date": "20260315"}
    ]
