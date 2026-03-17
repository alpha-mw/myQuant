"""
下载器单元测试
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _load_module(monkeypatch):
    """在无真实 tushare 依赖的测试环境中导入下载模块。"""
    fake_tushare = types.SimpleNamespace(pro_api=lambda token: object())
    monkeypatch.setitem(sys.modules, "tushare", fake_tushare)
    module_name = "download_full_cn_market_testable"
    sys.modules.pop(module_name, None)
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "unified" / "download_full_cn_market.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
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
    assert fake_pro.daily_calls == []
