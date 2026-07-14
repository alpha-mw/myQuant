"""
下载器单元测试
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.market_data_store import MarketDataStore
from quant_investor.market.pit_universe import (
    LIST_STATUS_DELISTED,
    LIST_STATUS_LISTED,
    PITUniverseRecord,
    PITUniverseStore,
)


def _load_module(
    monkeypatch,
    *,
    freshness_mode: str = "strict",
    coverage_threshold: str = "0.95",
    early_stop_sample_size: str = "10",
    early_stop_stale_ratio: str = "0.80",
):
    """在无真实 tushare 依赖的测试环境中导入下载模块。"""
    monkeypatch.setenv("CN_FRESHNESS_MODE", freshness_mode)
    monkeypatch.setenv("CN_FRESHNESS_COVERAGE_THRESHOLD", coverage_threshold)
    monkeypatch.setenv("CN_STRICT_EARLY_STOP_SAMPLE_SIZE", early_stop_sample_size)
    monkeypatch.setenv("CN_STRICT_EARLY_STOP_STALE_RATIO", early_stop_stale_ratio)
    fake_tushare = types.SimpleNamespace(pro_api=lambda token: object())
    monkeypatch.setitem(sys.modules, "tushare", fake_tushare)
    for module_name in [
        "quant_investor.market.download_cn",
        "quant_investor.config",
        "quant_investor.market.config",
        "quant_investor.fetch_cn_index_components",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    module_name = "quant_investor.market.download_cn"
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "TUSHARE_TOKEN", "dummy-token")
    monkeypatch.setattr(module.config, "TUSHARE_AUTO_CLEAN", False, raising=False)
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

    def stock_basic(self, **_kwargs):
        return pd.DataFrame(columns=["ts_code", "list_date"])


class BatchFakePro(FakePro):
    def __init__(self) -> None:
        super().__init__()
        self.batch_daily_calls: list[tuple[str, str | None]] = []
        self.batch_adj_factor_calls: list[tuple[str, str | None]] = []
        self.daily_basic_calls: list[tuple[str, str | None]] = []

    def daily(
        self,
        ts_code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        trade_date: str | None = None,
        fields: str | None = None,
    ):
        if trade_date:
            self.batch_daily_calls.append((trade_date, fields))
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": trade_date,
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
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": trade_date,
                        "open": 20.7,
                        "high": 21.0,
                        "low": 20.6,
                        "close": 20.9,
                        "pre_close": 20.7,
                        "change": 0.2,
                        "pct_chg": 0.97,
                        "vol": 2600,
                        "amount": 26000,
                    },
                ]
            )
        if ts_code is None or start_date is None or end_date is None:
            raise TypeError("symbol-scoped daily requires ts_code/start_date/end_date")
        return super().daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def adj_factor(
        self,
        ts_code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        trade_date: str | None = None,
        fields: str | None = None,
    ):
        if trade_date:
            self.batch_adj_factor_calls.append((trade_date, fields))
            return pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "trade_date": trade_date, "adj_factor": 1.0},
                    {"ts_code": "000002.SZ", "trade_date": trade_date, "adj_factor": 1.0},
                ]
            )
        if ts_code is None or start_date is None or end_date is None:
            raise TypeError("symbol-scoped adj_factor requires ts_code/start_date/end_date")
        return super().adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def daily_basic(self, trade_date: str | None = None, fields: str | None = None):
        self.daily_basic_calls.append((trade_date or "", fields))
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "turnover_rate": 1.1, "pe": 12.0, "pb": 1.2},
                {"ts_code": "000002.SZ", "trade_date": trade_date, "turnover_rate": 1.2, "pe": 13.0, "pb": 1.3},
            ]
        )


class PartialBatchFakePro(BatchFakePro):
    def daily(
        self,
        ts_code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        trade_date: str | None = None,
        fields: str | None = None,
    ):
        frame = super().daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            trade_date=trade_date,
            fields=fields,
        )
        if trade_date:
            return frame[frame["ts_code"].eq("000001.SZ")].reset_index(drop=True)
        return frame


def _write_cn_parquet_rows(data_root: Path, symbol: str, rows: list[dict]) -> None:
    normalized_rows = []
    for row in rows:
        close = float(row.get("close", row.get("adj_close", 10.0)))
        open_price = float(row.get("open", close))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        adj_factor = float(row.get("adj_factor", 1.0))
        normalized_rows.append(
            {
                "ts_code": symbol,
                "trade_date": row.get("trade_date", ""),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "pre_close": float(row.get("pre_close", close)),
                "change": float(row.get("change", 0.0)),
                "pct_chg": float(row.get("pct_chg", 0.0)),
                "vol": float(row.get("vol", 1000.0)),
                "amount": float(row.get("amount", 10000.0)),
                "adj_factor": adj_factor,
                "adj_close": float(row.get("adj_close", close * adj_factor)),
                "adj_open": float(row.get("adj_open", open_price * adj_factor)),
                "adj_high": float(row.get("adj_high", high * adj_factor)),
                "adj_low": float(row.get("adj_low", low * adj_factor)),
            }
        )
    MarketDataStore(market="CN", data_root=data_root).write_full_history_bars(
        pd.DataFrame(normalized_rows),
        source="test_download_full_cn_market",
    )


def _write_cn_parquet_row(data_root: Path, symbol: str, trade_date: str, close: float = 10.0) -> None:
    _write_cn_parquet_rows(data_root, symbol, [{"trade_date": trade_date, "close": close}])


def _read_cn_parquet_frame(data_root: Path, symbol: str) -> pd.DataFrame:
    return MarketDataReader(market="CN", data_root=data_root).read_symbol_frame(symbol).frame


def _write_stale_daily_file(path, symbol: str) -> None:
    data_root = Path(path).parent.parent
    rows = []
    start = datetime(2025, 1, 1)
    for idx in range(250):
        trade_date = (start + timedelta(days=idx)).strftime("%Y-%m-%d")
        rows.append(
            {
                "ts_code": symbol,
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
            "ts_code": symbol,
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
    _write_cn_parquet_rows(data_root, symbol, rows)


def test_download_daily_batch_uses_trade_date_endpoint_once_for_stale_symbols(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = BatchFakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)
    monkeypatch.setattr(module.config, "TUSHARE_AUTO_CLEAN", False, raising=False)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_stale_daily_file(tmp_path / "hs300" / "000001.SZ.csv", "000001.SZ")
    _write_stale_daily_file(tmp_path / "hs300" / "000002.SZ.csv", "000002.SZ")

    results = downloader.download_daily_batch(
        ["000001.SZ", "000002.SZ"],
        "hs300",
        target_trade_date="20260316",
    )

    assert [row["status"] for row in results] == ["updated", "updated"]
    assert fake_pro.batch_daily_calls == [("20260316", "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount")]
    assert fake_pro.batch_adj_factor_calls == [("20260316", "ts_code,trade_date,adj_factor")]
    assert len(fake_pro.daily_calls) == 0
    assert _read_cn_parquet_frame(tmp_path, "000001.SZ")["trade_date"].iloc[-1] == "20260316"
    assert _read_cn_parquet_frame(tmp_path, "000002.SZ")["adj_close"].iloc[-1] == 20.9


def test_download_daily_batch_fills_missing_tushare_symbol_from_eastmoney(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = PartialBatchFakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)
    monkeypatch.setattr(module.config, "TUSHARE_AUTO_CLEAN", False, raising=False)

    eastmoney_calls: list[tuple[tuple[str, ...], str]] = []

    def fake_eastmoney(symbols, trade_date, **_kwargs):
        eastmoney_calls.append((tuple(symbols), trade_date))
        return pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "2026-03-16",
                    "open": 20.8,
                    "high": 21.2,
                    "low": 20.5,
                    "close": 21.0,
                    "pre_close": 20.7,
                    "change": 0.3,
                    "pct_chg": 1.4493,
                    "vol": 2700,
                    "amount": 27000,
                    "data_source": "eastmoney.push2his.kline",
                }
            ]
        )

    monkeypatch.setattr(module, "fetch_eastmoney_daily_batch_frame", fake_eastmoney)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_stale_daily_file(tmp_path / "hs300" / "000001.SZ.csv", "000001.SZ")
    _write_stale_daily_file(tmp_path / "hs300" / "000002.SZ.csv", "000002.SZ")

    results = downloader.download_daily_batch(
        ["000001.SZ", "000002.SZ"],
        "hs300",
        target_trade_date="20260316",
    )

    by_symbol = {row["symbol"]: row for row in results}
    assert eastmoney_calls == [(("000002.SZ",), "20260316")]
    assert by_symbol["000001.SZ"]["status"] == "updated"
    assert by_symbol["000001.SZ"]["data_source"] == "tushare.daily"
    assert by_symbol["000002.SZ"]["status"] == "updated"
    assert by_symbol["000002.SZ"]["data_source"] == "eastmoney.push2his.kline"
    assert _read_cn_parquet_frame(tmp_path, "000001.SZ")["close"].iloc[-1] == 10.9
    assert _read_cn_parquet_frame(tmp_path, "000002.SZ")["close"].iloc[-1] == 21.0


def test_fetch_eastmoney_daily_batch_frame_normalizes_kline_payload(monkeypatch):
    module = _load_module(monkeypatch)
    requested_urls: list[str] = []

    def fake_fetch_json(url: str):
        requested_urls.append(url)
        return {
            "data": {
                "klines": [
                    "2026-03-16,20.80,21.00,21.20,20.50,2700,27000000.00,3.38,1.45,0.30,1.20"
                ]
            }
        }

    frame = module.fetch_eastmoney_daily_batch_frame(
        ["000002.SZ"],
        "20260316",
        fetch_json=fake_fetch_json,
    )

    assert "secid=0.000002" in requested_urls[0]
    assert frame.to_dict("records") == [
        {
            "ts_code": "000002.SZ",
            "trade_date": "2026-03-16",
            "open": 20.8,
            "high": 21.2,
            "low": 20.5,
            "close": 21.0,
            "pre_close": 20.7,
            "change": 0.3,
            "pct_chg": 1.45,
            "vol": 2700.0,
            "amount": 27000.0,
            "data_source": "eastmoney.push2his.kline",
        }
    ]


def test_download_stock_incremental_update(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
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
    _write_cn_parquet_rows(tmp_path, "000001.SZ", old_rows)

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "updated"
    assert fake_pro.daily_calls[-1] == ("000001.SZ", "20260311", "20260316")

    updated_df = _read_cn_parquet_frame(tmp_path, "000001.SZ")
    assert updated_df["trade_date"].iloc[-1] == "20260316"
    assert updated_df["trade_date"].nunique() == len(updated_df)


def test_download_stock_full_a_targets_resolved_existing_path(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)

    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-15", close=9.0)

    result = downloader.download_stock("000001.SZ", "full_a")

    assert result["status"] == "updated"
    updated_df = _read_cn_parquet_frame(tmp_path, "000001.SZ")
    assert updated_df["trade_date"].iloc[-1] == "20260316"
    assert updated_df["close"].iloc[-1] == 10.9


def test_download_stock_full_a_new_symbol_uses_bucket_from_components(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    components_file = tmp_path / "cn_index_components.json"
    components_file.write_text(
        json.dumps(
            {
                "full_a": ["000001.SZ"],
                "all": ["000001.SZ"],
                "hs300": ["000001.SZ"],
                "zz500": [],
                "zz1000": [],
                "stats": {
                    "full_a": 1,
                    "hs300": 1,
                    "zz500": 0,
                    "zz1000": 0,
                    "total_unique": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    downloader.load_components(components_file=str(components_file))

    result = downloader.download_stock("000001.SZ", "full_a")

    assert result["status"] == "updated"
    assert _read_cn_parquet_frame(tmp_path, "000001.SZ")["trade_date"].iloc[-1] == "20260316"


def test_download_stock_full_a_lazy_loads_components_for_new_symbol(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    components = {
        "full_a": ["000001.SZ"],
        "all": ["000001.SZ"],
        "hs300": ["000001.SZ"],
        "zz500": [],
        "zz1000": [],
        "stats": {"total_unique": 1},
    }
    load_calls: list[bool] = []

    def fake_load_components(*_args, **_kwargs):
        load_calls.append(True)
        return components

    monkeypatch.setattr(downloader, "load_components", fake_load_components)

    result = downloader.download_stock("000001.SZ", "full_a")

    assert load_calls == [True]
    assert result["status"] == "updated"
    assert _read_cn_parquet_frame(tmp_path, "000001.SZ")["trade_date"].iloc[-1] == "20260316"


def test_download_stock_skips_when_file_is_latest(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
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
    _write_cn_parquet_rows(tmp_path, "000001.SZ", rows)

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "cached"
    assert result["api_calls"] == 0
    assert fake_pro.daily_calls == []


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
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")
    _write_cn_parquet_row(tmp_path, "000002.SZ", "2026-03-15")

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


def test_build_completeness_report_excludes_symbols_listed_after_target_trade_date(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="strict")
    fake_pro = FakePro()

    def _fake_stock_basic(**_kwargs):
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "list_date": "20200101"},
                {"ts_code": "301696.SZ", "list_date": "20260317"},
            ]
        )

    fake_pro.stock_basic = _fake_stock_basic
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")

    components = {"full_a": ["000001.SZ", "301696.SZ"], "stats": {"total_unique": 2}}
    report = downloader.build_completeness_report(
        components=components,
        categories=["full_a"],
        target_trade_date="20260316",
    )

    assert report["complete"] is True
    assert report["blocking_incomplete_count"] == 0
    assert report["expected_scope_count"] == 1
    assert report["coverage_complete_count"] == 1
    assert report["categories"]["full_a"]["expected"] == 1
    assert report["categories"]["full_a"]["pre_listing_symbols"] == [
        {"symbol": "301696.SZ", "list_date": "20260317"}
    ]


def test_build_completeness_report_stable_mode_rolls_back_when_strict_coverage_is_below_threshold(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="stable", coverage_threshold="0.95")
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")
    _write_cn_parquet_row(tmp_path, "000002.SZ", "2026-03-14")

    components = {"hs300": ["000001.SZ", "000002.SZ"], "zz500": [], "zz1000": []}
    report = downloader.build_completeness_report(components=components)

    assert report["strict_trade_date"] == "20260316"
    assert report["stable_trade_date"] == "20260314"
    assert report["effective_target_trade_date"] == "20260314"
    assert report["latest_trade_date"] == "20260314"
    assert report["coverage_ratio"] == 1.0
    assert report["complete"] is True


def test_build_completeness_report_stable_mode_keeps_strict_target_when_coverage_meets_threshold(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="stable", coverage_threshold="0.50")
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")
    _write_cn_parquet_row(tmp_path, "000002.SZ", "2026-03-14")

    components = {"hs300": ["000001.SZ", "000002.SZ"], "zz500": [], "zz1000": []}
    report = downloader.build_completeness_report(components=components)

    assert report["effective_target_trade_date"] == "20260316"
    assert report["latest_trade_date"] == "20260316"
    assert report["coverage_ratio"] == 0.5
    assert report["complete"] is False
    assert report["blocking_incomplete_count"] == 1


def test_build_completeness_report_strict_mode_does_not_roll_back_target(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, freshness_mode="strict", coverage_threshold="0.95")
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")
    _write_cn_parquet_row(tmp_path, "000002.SZ", "2026-03-14")

    components = {"hs300": ["000001.SZ", "000002.SZ"], "zz500": [], "zz1000": []}
    report = downloader.build_completeness_report(components=components)

    assert report["effective_target_trade_date"] == "20260316"
    assert report["latest_trade_date"] == "20260316"
    assert report["coverage_ratio"] == 0.5
    assert report["complete"] is False


def test_build_completeness_report_does_not_mutate_default_target_for_download_stock(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, freshness_mode="strict")
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000002.SZ", "2026-03-14")

    components = {"hs300": ["000002.SZ"], "zz500": [], "zz1000": []}
    report = downloader.build_completeness_report(
        components=components,
        categories=["hs300"],
        target_trade_date="20260314",
    )

    result = downloader.download_stock("000002.SZ", "hs300")

    assert report["effective_target_trade_date"] == "20260314"
    assert downloader.latest_trade_date == "20260316"
    assert result["status"] == "updated"
    assert result["latest_trade_date"] == "20260316"
    assert fake_pro.daily_calls[-1][0] == "000002.SZ"
    assert fake_pro.daily_calls[-1][2] == "20260316"


def test_build_completeness_report_coverage_uses_scope_expected_and_counts_suspended_only(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="strict")
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
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")
    _write_cn_parquet_row(tmp_path, "000002.SZ", "2026-03-15")
    _write_cn_parquet_row(tmp_path, "000003.SZ", "2026-03-15")

    components = {"hs300": ["000001.SZ", "000002.SZ", "000003.SZ"], "zz500": [], "zz1000": []}
    report = downloader.build_completeness_report(
        components=components,
        allowed_stale_symbols={"000003.SZ"},
    )

    assert report["expected_scope_count"] == 3
    assert report["coverage_complete_count"] == 2
    assert report["coverage_ratio"] == 2 / 3
    assert report["complete"] is True
    assert report["blocking_incomplete_count"] == 0
    assert report["categories"]["hs300"]["expected"] == 3
    assert report["categories"]["hs300"]["coverage_complete_count"] == 2
    assert report["categories"]["hs300"]["coverage_ratio"] == 2 / 3


def test_build_completeness_report_records_suspend_probe_exception(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="strict")
    fake_pro = FakePro()

    def _raise_suspend_error(**_kwargs):
        raise RuntimeError("suspend probe failed")

    fake_pro.suspend_d = _raise_suspend_error
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")

    components = {"hs300": ["000001.SZ"], "zz500": [], "zz1000": []}
    report = downloader.build_completeness_report(components=components)

    assert report["complete"] is True
    assert report["data_quality_issue_count"] == 1
    issue = report["data_quality_issues"][0]
    assert issue["source"] == "download_cn._load_latest_suspended_symbols"
    assert issue["issue_type"] == "suspend_lookup_exception"
    assert issue["severity"] == "warning"
    assert issue["message"] == "suspend probe failed"


def test_suspend_cache_v2_is_requeried_and_rewritten_with_bound_evidence(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="strict")

    class _SuspendPro(FakePro):
        def __init__(self):
            super().__init__()
            self.suspend_calls: list[dict[str, str]] = []

        def suspend_d(self, **kwargs):
            self.suspend_calls.append(dict(kwargs))
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": "20260316",
                        "suspend_type": "S",
                    }
                ]
            )

    fake_pro = _SuspendPro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)
    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    cache_path = downloader._suspend_cache_path("20260316")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": 2,
                "trade_date": "20260316",
                "symbols": ["999999.SZ"],
            }
        ),
        encoding="utf-8",
    )

    symbols = downloader._load_latest_suspended_symbols("20260316")

    assert symbols == {"000002.SZ"}
    assert fake_pro.suspend_calls == [{"trade_date": "20260316"}]
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    declared_sha256 = payload.pop("payload_sha256")
    computed_sha256 = module.canonical_json_sha256(payload)
    assert payload["version"] == 5
    assert payload["query_succeeded"] is True
    assert payload["source"] == "tushare.suspend_d"
    assert payload["query_variant"] == "trade_date"
    assert payload["query_params"] == {"trade_date": "20260316"}
    assert payload["continuation_state_complete"] is False
    assert payload["exact_date_rows_validated"] is True
    assert payload["raw_row_count"] == 1
    assert payload["matched_row_count"] == 1
    assert payload["exact_event_records"] == [
        {
            "ts_code": "000002.SZ",
            "trade_date": "20260316",
            "suspend_type": "S",
        }
    ]
    assert payload["resume_symbols"] == []
    assert payload["other_event_symbols"] == []
    assert payload["query_run_id"]
    assert declared_sha256 == computed_sha256

    downloader._load_latest_suspended_symbols(
        "20260316",
        force_refresh=True,
        query_run_id="shared-history-audit-run",
    )
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert fake_pro.suspend_calls == [
        {"trade_date": "20260316"},
        {"trade_date": "20260316"},
    ]
    assert refreshed["query_run_id"] == "shared-history-audit-run"


def test_suspend_cache_v5_preserves_resume_and_other_exact_events(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="strict")

    class _SuspendPro(FakePro):
        def suspend_d(self, **_kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260316",
                        "suspend_type": "S",
                    },
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": "20260316",
                        "suspend_type": "R",
                    },
                    {
                        "ts_code": "000003.SZ",
                        "trade_date": "20260316",
                        "suspend_type": "X",
                    },
                ]
            )

    fake_pro = _SuspendPro()
    monkeypatch.setattr(
        module,
        "create_tushare_pro",
        lambda *_args, **_kwargs: fake_pro,
    )
    downloader = module.CNFullMarketDownloader(
        data_dir=str(tmp_path),
        years=3,
    )

    assert downloader._load_latest_suspended_symbols("20260316") == {
        "000001.SZ"
    }
    payload = json.loads(
        downloader._suspend_cache_path("20260316").read_text(
            encoding="utf-8"
        )
    )
    assert payload["resume_symbols"] == ["000002.SZ"]
    assert payload["other_event_symbols"] == ["000003.SZ"]
    assert payload["exact_event_row_count"] == 3


def test_load_active_listing_dates_uses_pit_universe_when_enabled(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="strict")
    fake_pro = FakePro()
    stock_basic_calls = {"count": 0}

    def _count_stock_basic(**_kwargs):
        stock_basic_calls["count"] += 1
        return pd.DataFrame(columns=["ts_code", "list_date"])

    fake_pro.stock_basic = _count_stock_basic
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    pit_root = tmp_path / "parquet" / "cn" / "reference"
    PITUniverseStore(
        root_dir=pit_root,
        raw_root=tmp_path / "pit_raw",
        compatibility_path=tmp_path / "pit_compat.json",
    ).write_snapshot(
        raw_records=[
            PITUniverseRecord(
                symbol="000001.SZ",
                source_list_status=LIST_STATUS_LISTED,
                list_date="20200101",
                observed_at="2026-07-06T00:00:00Z",
                source_run_id="unit-test",
            ),
            PITUniverseRecord(
                symbol="000002.SZ",
                source_list_status=LIST_STATUS_DELISTED,
                list_date="20210101",
                delist_date="20250102",
                observed_at="2026-07-06T00:00:00Z",
                source_run_id="unit-test",
            ),
        ],
        observed_at="2026-07-06T00:00:00Z",
        source_run_id="unit-test",
    )
    monkeypatch.setattr(module.config, "PIT_UNIVERSE_ENABLED", True, raising=False)
    monkeypatch.setattr(module.config, "PIT_UNIVERSE_REQUIRED", False, raising=False)
    monkeypatch.setattr(module.config, "PIT_UNIVERSE_SOURCE_ROOT", str(pit_root), raising=False)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)

    assert downloader._load_active_listing_dates() == {
        "000001.SZ": "20200101",
        "000002.SZ": "20210101",
    }
    assert stock_basic_calls["count"] == 0


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
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")
    _write_cn_parquet_row(tmp_path, "000002.SZ", "2026-03-15")

    components = {"hs300": ["000001.SZ", "000002.SZ"], "zz500": [], "zz1000": []}

    report = downloader.build_completeness_report(components=components)
    assert report["complete"] is True
    assert report["blocking_incomplete_count"] == 0
    assert report["categories"]["hs300"]["suspended_stale_symbols"] == [
        {"symbol": "000002.SZ", "latest_local_date": "20260315"}
    ]


def test_full_a_local_universe_uses_existing_local_cache(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)

    for category, symbol in [
        ("hs300", "000001.SZ"),
        ("zz500", "000002.SZ"),
        ("other", "600001.SH"),
    ]:
        _write_cn_parquet_row(tmp_path, symbol, "2026-03-16", close=10.2)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    assert not (tmp_path / "full_a").exists()
    components = downloader.load_components(components_file=str(tmp_path / "cn_index_components.json"))
    report = downloader.build_completeness_report(components=components, categories=["full_a"])

    assert downloader.pro is None
    assert components["full_a"] == ["000001.SZ", "000002.SZ", "600001.SH"]
    assert report["complete"] is True
    assert report["categories"]["full_a"]["expected"] == 3


def test_tushare_unavailable_uses_locally_observable_stable_date_for_strict_and_stable(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="stable")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)

    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-15")

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)

    assert downloader.pro is None
    assert downloader.strict_trade_date == "20260315"
    assert downloader.stable_trade_date == "20260315"
    assert downloader.latest_trade_date == "20260315"


def test_full_a_resolver_uses_fixed_directory_priority(tmp_path):
    from quant_investor.market.cn_resolver import CNUniverseResolver

    for category, trade_date in [("hs300", "2026-03-16"), ("zz500", "2026-03-15"), ("other", "2026-03-14")]:
        file_path = tmp_path / category / "000001.SZ.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": trade_date, "close": 10.0}]).to_csv(file_path, index=False)

    resolver = CNUniverseResolver(data_dir=str(tmp_path))
    assert not (tmp_path / "full_a").exists()
    resolved = resolver.resolve_symbol_file("000001.SZ", universe_key="full_a")
    assert resolved == tmp_path / "hs300" / "000001.SZ.csv"
    inventory, source_paths = resolver.collect_full_a_inventory()
    assert inventory == ["000001.SZ"]
    assert source_paths["000001.SZ"] == str(tmp_path / "hs300" / "000001.SZ.csv")
    snapshot = resolver.snapshot()
    assert snapshot["physical_directories_used_for_full_a"] == [str(tmp_path / "hs300"), str(tmp_path / "zz500"), str(tmp_path / "other")]


def test_get_all_local_symbols_full_a_uses_existing_directories_only(tmp_path):
    from quant_investor.market.analyze import get_all_local_symbols

    for category, symbol in [("hs300", "000001.SZ"), ("zz500", "000002.SZ"), ("other", "600001.SH")]:
        _write_cn_parquet_row(tmp_path, symbol, "2026-03-16")

    assert not (tmp_path / "full_a").exists()
    symbols = get_all_local_symbols("full_a", market="CN", data_dir=str(tmp_path))
    assert symbols == ["000001.SZ", "000002.SZ", "600001.SH"]


def test_get_all_components_falls_back_to_local_universe(monkeypatch):
    module = importlib.import_module("quant_investor.fetch_cn_index_components")
    monkeypatch.setattr(module, "fetch_full_a", lambda _pro: [])
    monkeypatch.setattr(module, "fetch_hs300", lambda _pro: [])
    monkeypatch.setattr(module, "fetch_zz500", lambda _pro: [])
    monkeypatch.setattr(module, "fetch_zz1000", lambda _pro: [])
    monkeypatch.setattr(
        module,
        "_local_parquet_full_a_inventory",
        lambda: (
            ["000001.SZ", "600001.SH"],
            {"status": "OK", "latest_complete_trade_date": "20260316"},
        ),
    )

    components = module.get_all_components(pro=object())

    assert components["full_a"] == ["000001.SZ", "600001.SH"]
    assert components["all"] == ["000001.SZ", "600001.SH"]
    assert components["stats"]["full_a"] == 2
    assert components["stats"]["total_unique"] == 2
    assert components["resolver"]["resolution_strategy"] == "parquet_serving_inventory"
    assert components["resolver"]["parquet_inventory"]["status"] == "OK"


def test_evaluate_symbol_local_status_covers_fixed_contract(tmp_path):
    from quant_investor.market.cn_resolver import CNUniverseResolver
    from quant_investor.market.cn_symbol_status import evaluate_symbol_local_status

    frames = {
        "000001.SZ": pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]),
        "000002.SZ": pd.DataFrame([{"trade_date": "2026-03-15", "close": 10.0}]),
        "000003.SZ": pd.DataFrame([{"close": 10.0}]),
        "000005.SZ": pd.DataFrame([{"trade_date": "2026-03-15", "close": 9.5}]),
    }

    resolver = CNUniverseResolver(data_dir=str(tmp_path))

    class _FakeMarketReader:
        def resolve_symbol_path(self, symbol, **_kwargs):
            return tmp_path / f"{symbol}.parquet" if symbol in frames else None

        def read_symbol_frame(self, symbol, **_kwargs):
            return types.SimpleNamespace(frame=frames.get(symbol, pd.DataFrame()), issues=[])

    market_reader = _FakeMarketReader()

    up_to_date = evaluate_symbol_local_status(
        "000001.SZ",
        category="hs300",
        resolver=resolver,
        market_reader=market_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols=set(),
        suspended_symbols=set(),
    )
    stale = evaluate_symbol_local_status(
        "000002.SZ",
        category="hs300",
        resolver=resolver,
        market_reader=market_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols=set(),
        suspended_symbols=set(),
    )
    unreadable = evaluate_symbol_local_status(
        "000003.SZ",
        category="hs300",
        resolver=resolver,
        market_reader=market_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols=set(),
        suspended_symbols=set(),
    )
    missing = evaluate_symbol_local_status(
        "000004.SZ",
        category="hs300",
        resolver=resolver,
        market_reader=market_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols={"000004.SZ"},
        suspended_symbols=set(),
    )
    suspended_stale = evaluate_symbol_local_status(
        "000005.SZ",
        category="hs300",
        resolver=resolver,
        market_reader=market_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols=set(),
        suspended_symbols={"000005.SZ"},
    )
    stale_cached = stale.with_local_status("stale_cached")

    assert up_to_date.local_status == "up_to_date"
    assert up_to_date.is_complete is True and up_to_date.is_blocking is False
    assert stale.local_status == "stale"
    assert stale.is_complete is False and stale.is_blocking is True
    assert unreadable.local_status == "unreadable"
    assert unreadable.is_complete is False and unreadable.is_blocking is True
    assert missing.local_status == "missing"
    assert missing.is_complete is False and missing.is_blocking is False
    assert suspended_stale.local_status == "suspended_stale"
    assert suspended_stale.is_complete is True and suspended_stale.is_blocking is False
    assert stale_cached.local_status == "stale_cached"
    assert stale_cached.is_complete is False and stale_cached.is_blocking is True


def test_download_stock_returns_stale_cached_when_increment_is_empty(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-15")
    monkeypatch.setattr(downloader, "_fetch_stock_frame", lambda *_args, **_kwargs: pd.DataFrame())

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "stale_cached"
    assert result["local_status"] == "stale_cached"
    assert result["api_calls"] == downloader.REQUESTS_PER_STOCK


def test_download_stock_returns_stale_cached_when_fetch_lags_target(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, freshness_mode="strict")
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-14")

    def _lagging_frame(*_args, **_kwargs):
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "2026-03-14",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "change": 0.2,
                    "pct_chg": 2.0,
                    "vol": 1000,
                    "amount": 10000,
                }
            ]
        )

    monkeypatch.setattr(downloader, "_fetch_stock_frame", _lagging_frame)

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "stale_cached"
    assert result["local_status"] == "stale_cached"
    assert result["latest_local_date"] == "20260314"
    assert result["latest_trade_date"] == "20260316"
    assert result["api_calls"] == downloader.REQUESTS_PER_STOCK


def test_download_stock_full_a_lazy_loads_components_from_custom_data_root(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)
    monkeypatch.setattr(
        module,
        "get_all_components",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should use custom components file before remote refresh")
        ),
    )

    data_dir = tmp_path / "cn-market-data"
    downloader = module.CNFullMarketDownloader(data_dir=str(data_dir), years=3)
    components_dir = tmp_path / "cn_universe"
    components_dir.mkdir(parents=True, exist_ok=True)
    (components_dir / "cn_index_components.json").write_text(
        json.dumps(
            {
                "full_a": ["000001.SZ"],
                "all": ["000001.SZ"],
                "hs300": ["000001.SZ"],
                "zz500": [],
                "zz1000": [],
                "stats": {
                    "full_a": 1,
                    "hs300": 1,
                    "zz500": 0,
                    "zz1000": 0,
                    "total_unique": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    result = downloader.download_stock("000001.SZ", "full_a")

    assert result["status"] == "updated"
    assert _read_cn_parquet_frame(data_dir, "000001.SZ")["trade_date"].iloc[-1] == "20260316"


def test_download_all_skips_loop_when_preflight_is_complete(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    components = {"full_a": ["000001.SZ"], "hs300": [], "zz500": [], "zz1000": [], "stats": {"total_unique": 1}}
    preflight = {
        "latest_trade_date": "20260316",
        "complete": True,
        "blocking_incomplete_count": 0,
        "categories_checked": ["full_a"],
        "categories": {
            "full_a": {
                "expected": 1,
                "date_counts": {"20260316": 1},
                "blocking_incomplete_count": 0,
                "blocking_missing_symbols": [],
                "blocking_stale_symbols": [],
                "blocking_unreadable_symbols": [],
            }
        },
        "resolver": {},
    }
    monkeypatch.setattr(downloader, "build_completeness_report", lambda **_kwargs: preflight)
    monkeypatch.setattr(
        downloader,
        "download_category",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("download_category should not be called")),
    )

    result = downloader.download_all(components=components, categories=["full_a"])

    assert result["completeness"]["complete"] is True
    assert result["categories"]["full_a"] == []


def test_download_all_full_a_routes_fresh_symbol_into_bucket_from_components(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    components = {
        "full_a": ["000001.SZ"],
        "all": ["000001.SZ"],
        "hs300": ["000001.SZ"],
        "zz500": [],
        "zz1000": [],
        "stats": {"total_unique": 1},
    }

    result = downloader.download_all(components=components, categories=["full_a"])

    assert result["completeness"]["complete"] is True
    assert result["categories"]["full_a"][0]["status"] == "updated"
    assert _read_cn_parquet_frame(tmp_path, "000001.SZ")["trade_date"].iloc[-1] == "20260316"


def test_download_all_early_stop_rolls_back_to_stable_target_and_aborts_remaining_symbols(
    monkeypatch,
    tmp_path,
):
    module = _load_module(
        monkeypatch,
        freshness_mode="strict",
        early_stop_sample_size="10",
        early_stop_stale_ratio="0.80",
    )
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    symbols = [f"{idx:06d}.SZ" for idx in range(1, 13)]
    for symbol in symbols:
        _write_cn_parquet_row(tmp_path, symbol, "2026-03-14")

    components = {
        "full_a": symbols,
        "all": symbols,
        "hs300": symbols,
        "zz500": [],
        "zz1000": [],
        "stats": {"total_unique": len(symbols)},
    }
    monkeypatch.setattr(downloader, "_fetch_stock_frame", lambda *_args, **_kwargs: pd.DataFrame())

    result = downloader.download_all(components=components, categories=["full_a"])

    assert len(result["categories"]["full_a"]) == 10
    assert result["rounds"][0]["early_stop_reason"] == "strict_same_day_unavailable"
    assert result["config"]["early_stop_reason"] == "strict_same_day_unavailable"
    assert result["completeness"]["latest_trade_date"] == "20260314"
    assert result["completeness"]["effective_target_trade_date"] == "20260314"
    assert result["completeness"]["early_stop_reason"] == "strict_same_day_unavailable"
    assert result["completeness"]["complete"] is True
    assert downloader.stats["stale_cached"] == 10


def test_download_all_uses_trade_date_close_probe_before_sampling_symbols(
    monkeypatch,
    tmp_path,
):
    module = _load_module(
        monkeypatch,
        freshness_mode="strict",
        early_stop_sample_size="10",
        early_stop_stale_ratio="0.80",
    )

    class SameDayUnavailablePro(FakePro):
        def daily(self, *args, **kwargs):
            if kwargs.get("trade_date") == "20260316":
                return pd.DataFrame(columns=["ts_code", "trade_date", "close"])
            return super().daily(*args, **kwargs)

    fake_pro = SameDayUnavailablePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    symbols = [f"{idx:06d}.SZ" for idx in range(1, 13)]
    for symbol in symbols:
        _write_cn_parquet_row(tmp_path, symbol, "2026-03-14")

    components = {
        "full_a": symbols,
        "all": symbols,
        "hs300": symbols,
        "zz500": [],
        "zz1000": [],
        "stats": {"total_unique": len(symbols)},
    }
    monkeypatch.setattr(
        downloader,
        "download_category",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("download_category should not be called after same-day probe")
        ),
    )

    result = downloader.download_all(components=components, categories=["full_a"])

    assert result["rounds"] == []
    assert result["config"]["early_stop_reason"] == "strict_same_day_unavailable"
    assert result["config"]["same_day_close_probe"]["source"] == "daily"
    assert result["config"]["same_day_close_probe"]["available"] is False
    assert result["config"]["effective_target_trade_date"] == "20260314"
    assert result["completeness"]["effective_target_trade_date"] == "20260314"
    assert result["completeness"]["early_stop_reason"] == "strict_same_day_unavailable"
    assert result["completeness"]["complete"] is True
    assert result["categories"]["full_a"] == []


def test_download_all_early_stop_when_non_empty_fetch_lags_strict_target(
    monkeypatch,
    tmp_path,
):
    module = _load_module(
        monkeypatch,
        freshness_mode="strict",
        early_stop_sample_size="3",
        early_stop_stale_ratio="0.80",
    )
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    symbols = [f"{idx:06d}.SZ" for idx in range(1, 6)]
    for symbol in symbols:
        _write_cn_parquet_row(tmp_path, symbol, "2026-03-14")

    components = {
        "full_a": symbols,
        "all": symbols,
        "hs300": symbols,
        "zz500": [],
        "zz1000": [],
        "stats": {"total_unique": len(symbols)},
    }

    def _lagging_frame(symbol, *_args, **_kwargs):
        return pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "trade_date": "2026-03-14",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "change": 0.2,
                    "pct_chg": 2.0,
                    "vol": 1000,
                    "amount": 10000,
                }
            ]
        )

    monkeypatch.setattr(downloader, "_fetch_stock_frame", _lagging_frame)

    result = downloader.download_all(components=components, categories=["full_a"])

    assert len(result["categories"]["full_a"]) == 3
    assert {row["status"] for row in result["categories"]["full_a"]} == {"stale_cached"}
    assert result["rounds"][0]["early_stop_reason"] == "strict_same_day_unavailable"
    assert result["config"]["effective_target_trade_date"] == "20260314"
    assert result["completeness"]["complete"] is True


def test_download_category_progress_prints_new_counters(monkeypatch, tmp_path, capsys):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    results = iter(
        [
            {"symbol": "000001.SZ", "category": "hs300", "status": "cached", "api_calls": 0},
            {
                "symbol": "000002.SZ",
                "category": "hs300",
                "status": "stale_cached",
                "api_calls": downloader.REQUESTS_PER_STOCK,
            },
            {
                "symbol": "000003.SZ",
                "category": "hs300",
                "status": "updated",
                "api_calls": downloader.REQUESTS_PER_STOCK,
            },
            {
                "symbol": "000004.SZ",
                "category": "hs300",
                "status": "failed",
                "api_calls": downloader.REQUESTS_PER_STOCK,
            },
        ]
    )
    monkeypatch.setattr(downloader, "download_stock", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    downloader.download_category(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"], "hs300")
    output = capsys.readouterr().out

    assert "cached:" in output
    assert "stale_cached:" in output
    assert "updated:" in output
    assert "failed:" in output
    assert "缓存:" not in output


def test_invalid_cn_freshness_env_values_fall_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("CN_FRESHNESS_COVERAGE_THRESHOLD", "bad-threshold")
    monkeypatch.setenv("CN_STRICT_EARLY_STOP_SAMPLE_SIZE", "bad-sample")
    monkeypatch.setenv("CN_STRICT_EARLY_STOP_STALE_RATIO", "bad-ratio")
    fake_tushare = types.SimpleNamespace(pro_api=lambda token: object())
    monkeypatch.setitem(sys.modules, "tushare", fake_tushare)
    for module_name in [
        "quant_investor.market.download_cn",
        "quant_investor.config",
        "quant_investor.market.config",
        "quant_investor.fetch_cn_index_components",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    download_module = importlib.import_module("quant_investor.market.download_cn")
    monkeypatch.setattr(download_module, "TUSHARE_TOKEN", "dummy-token")
    monkeypatch.setattr(download_module, "create_tushare_pro", lambda *_args, **_kwargs: FakePro())

    downloader = download_module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)

    assert downloader.coverage_threshold == 0.95
    assert downloader.strict_early_stop_sample_size == 10
    assert downloader.strict_early_stop_stale_ratio == 0.80


def test_cn_market_data_dir_env_is_used(monkeypatch, tmp_path):
    env_data_dir = tmp_path / "cn-market-data"
    monkeypatch.setenv("CN_MARKET_DATA_DIR", str(env_data_dir))
    for module_name in [
        "quant_investor.config",
        "quant_investor.market.config",
        "quant_investor.market.download_cn",
        "quant_investor.fetch_cn_index_components",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    fake_tushare = types.SimpleNamespace(pro_api=lambda token: object())
    monkeypatch.setitem(sys.modules, "tushare", fake_tushare)

    market_config_module = importlib.import_module("quant_investor.market.config")
    assert market_config_module.get_market_settings("CN").data_dir == str(env_data_dir)

    download_module = importlib.import_module("quant_investor.market.download_cn")
    monkeypatch.setattr(download_module, "create_tushare_pro", lambda *_args, **_kwargs: FakePro())
    downloader = download_module.CNFullMarketDownloader(years=3)
    assert downloader.data_dir == str(env_data_dir)

    monkeypatch.delitem(
        sys.modules,
        "quant_investor.fetch_cn_index_components",
        raising=False,
    )
    components_module = importlib.import_module("quant_investor.fetch_cn_index_components")
    captured: dict[str, str] = {}

    class _FakeMarketDataReader:
        def __init__(self, *, market, data_root, mode_policy):
            captured["market"] = market
            captured["data_root"] = str(data_root)
            captured["mode_policy"] = mode_policy

        def list_symbols(self, universe_key="full_a", category=None):
            return []

        def snapshot(self):
            return {"status": "blocked", "blockers": ["fixture_empty"]}

    monkeypatch.setattr(components_module, "MarketDataReader", _FakeMarketDataReader)
    monkeypatch.setattr(components_module, "fetch_full_a", lambda _pro: [])
    monkeypatch.setattr(components_module, "fetch_hs300", lambda _pro: [])
    monkeypatch.setattr(components_module, "fetch_zz500", lambda _pro: [])
    monkeypatch.setattr(components_module, "fetch_zz1000", lambda _pro: [])
    components_module.get_all_components(pro=object())

    assert captured["market"] == "CN"
    assert captured["data_root"] == str(env_data_dir.parent)
    assert captured["mode_policy"] == "strict"


def test_build_completeness_report_counts_cached_symbol_as_complete(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")
    components = {"hs300": ["000001.SZ"], "zz500": [], "zz1000": []}

    result = downloader.download_stock("000001.SZ", "hs300")
    report = downloader.build_completeness_report(components=components)

    assert result["status"] == "cached"
    assert report["complete"] is True
    assert report["categories"]["hs300"]["blocking_incomplete_count"] == 0
    assert report["categories"]["hs300"]["date_counts"] == {"20260316": 1}


# ── Freshness index tests ──────────────────────────────────────────────────────


def test_freshness_index_written_after_completeness_check(monkeypatch, tmp_path):
    """Slow-path (peek) completeness check should bootstrap the freshness index."""
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-16")

    components = {"hs300": ["000001.SZ"], "zz500": [], "zz1000": []}
    downloader.build_completeness_report(components=components)

    index_path = tmp_path / ".cache" / "freshness_index.json"
    assert index_path.exists(), "freshness_index.json should be created after completeness check"
    data = json.loads(index_path.read_text())
    assert data["symbols"].get("000001.SZ") == "20260316"


def test_freshness_index_fast_path_skips_file_peek(monkeypatch, tmp_path):
    """When index is pre-populated, completeness check uses it and doesn't need CSV files."""
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)

    # Pre-populate the index with an up-to-date date (no CSV file on disk)
    (tmp_path / ".cache").mkdir(parents=True, exist_ok=True)
    index_payload = {
        "schema_version": 1,
        "written_at": "20260316T100000",
        "symbols": {"000001.SZ": "20260316"},
    }
    (tmp_path / ".cache" / "freshness_index.json").write_text(
        json.dumps(index_payload), encoding="utf-8"
    )

    components = {"hs300": ["000001.SZ"], "zz500": [], "zz1000": []}
    report = downloader.build_completeness_report(components=components)

    # Even though there is no CSV file, the index says up-to-date
    assert report["complete"] is True
    assert report["categories"]["hs300"]["date_counts"] == {"20260316": 1}


def test_freshness_index_read_exception_records_diagnostic(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    cache_path = tmp_path / ".cache" / "freshness_index.json"
    cache_path.mkdir(parents=True)

    assert downloader._load_freshness_index() == {}
    issue = downloader._download_data_quality_issues[-1]
    assert issue.source == "download_cn._load_freshness_index"
    assert issue.issue_type == "freshness_index_read_exception"
    assert issue.severity == "warning"


def test_freshness_index_written_after_download(monkeypatch, tmp_path):
    """download_category() should update the freshness index for every processed symbol."""
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    _write_cn_parquet_row(tmp_path, "000001.SZ", "2026-03-14")

    downloader.download_category(["000001.SZ"], "hs300")

    index_path = tmp_path / ".cache" / "freshness_index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    # After download, the symbol should have an updated (or equal) date
    assert "000001.SZ" in data["symbols"]


def test_freshness_index_only_advances_date(monkeypatch, tmp_path):
    """_flush_freshness_index never regresses an existing date entry."""
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)

    # Write a newer date to the index first
    downloader._flush_freshness_index({"000001.SZ": "20260316"})
    # Now try to write an older date — it must not overwrite
    downloader._flush_freshness_index({"000001.SZ": "20260314"})

    index = downloader._load_freshness_index()
    assert index["000001.SZ"] == "20260316", "Older date must not regress the index entry"
