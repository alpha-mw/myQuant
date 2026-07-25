from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pandas as pd


def _load_module(
    monkeypatch,
    tmp_path,
    *,
    auto_clean: str = "1",
    factor_readiness: str = "1",
    storage_audit: str = "1",
    parquet_shadow: str = "0",
):
    monkeypatch.setenv("CN_FRESHNESS_MODE", "strict")
    monkeypatch.setenv("MYQUANT_TUSHARE_AUTO_CLEAN", auto_clean)
    monkeypatch.setenv("MYQUANT_TUSHARE_FACTOR_READINESS", factor_readiness)
    monkeypatch.setenv("MYQUANT_TUSHARE_STORAGE_AUDIT", storage_audit)
    monkeypatch.setenv("MYQUANT_TUSHARE_PARQUET_SHADOW_WRITE", parquet_shadow)
    monkeypatch.setenv("MYQUANT_TUSHARE_CLEANING_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("MYQUANT_TUSHARE_RAW_BACKUP_DIR", str(tmp_path / "raw_backups"))
    monkeypatch.setenv("MYQUANT_TUSHARE_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    monkeypatch.setenv("MYQUANT_TUSHARE_FACTOR_READINESS_DIR", str(tmp_path / "readiness"))
    monkeypatch.setenv("MYQUANT_TUSHARE_PARQUET_DIR", str(tmp_path / "parquet"))
    fake_tushare = types.SimpleNamespace(pro_api=lambda token: object())
    monkeypatch.setitem(sys.modules, "tushare", fake_tushare)
    for module_name in [
        "quant_investor.market.download_cn",
        "quant_investor.config",
        "quant_investor.market.config",
        "quant_investor.fetch_cn_index_components",
    ]:
        sys.modules.pop(module_name, None)
    return importlib.import_module("quant_investor.market.download_cn")


class FakePro:
    def __init__(self, *, invalid: bool = False) -> None:
        self.invalid = invalid
        self.daily_calls: list[tuple[str, str, str]] = []

    def trade_cal(self, **_kwargs):
        return pd.DataFrame({"cal_date": ["20260314", "20260316"]})

    def daily(self, ts_code: str, start_date: str, end_date: str):
        self.daily_calls.append((ts_code, start_date, end_date))
        if self.invalid:
            return pd.DataFrame(
                [
                    {
                        "ts_code": ts_code,
                        "trade_date": "20260316",
                        "open": 10.0,
                        "high": 9.0,
                        "low": 9.5,
                        "close": 10.2,
                        "vol": -1,
                        "amount": -100,
                    }
                ]
            )
        return pd.DataFrame(
            [
                {
                    "ts_code": ts_code,
                    "trade_date": "20260316",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "vol": 1000,
                    "amount": 10000,
                }
            ]
        )

    def adj_factor(self, ts_code: str, start_date: str, end_date: str):
        if self.invalid:
            return pd.DataFrame([{"trade_date": "20260316", "adj_factor": 1.0}])
        return pd.DataFrame([{"trade_date": "20260316", "adj_factor": 1.0}])

    def suspend_d(self, **_kwargs):
        return pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"])


def test_download_stock_fails_closed_after_full_history_writer_retirement(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path / "market"), years=3)
    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "failed"
    assert result["error"] == "cn_full_history_writer_retired_use_parquet_direct"
    assert result["cleaning_status"] == "skipped"
    assert result["factor_readiness_status"] is None
    assert result["parquet_status"] == "skipped"
    assert result["cleaning_report_path"] is None


def test_env_can_disable_auto_cleaning(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path, auto_clean="0")
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path / "market"), years=3)
    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "failed"
    assert result["error"] == "cn_full_history_writer_retired_use_parquet_direct"
    assert result["cleaning_status"] == "skipped"
    assert result["cleaning_report_path"] is None


def test_factor_readiness_can_be_disabled_separately(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path, factor_readiness="0")
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path / "market"), years=3)
    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "failed"
    assert result["error"] == "cn_full_history_writer_retired_use_parquet_direct"
    assert result["cleaning_status"] == "skipped"
    assert result["factor_readiness_status"] is None


def test_failed_cleaning_preserves_existing_canonical(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    fake_pro = FakePro(invalid=True)
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    market_dir = tmp_path / "market"
    existing = market_dir / "hs300" / "000001.SZ.csv"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("not,a,valid,market,csv\n", encoding="utf-8")
    before = existing.read_text(encoding="utf-8")

    downloader = module.CNFullMarketDownloader(data_dir=str(market_dir), years=3)
    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "failed"
    assert result["cleaning_status"] == "fail"
    assert existing.read_text(encoding="utf-8") == before
    assert Path(result["cleaning_report_path"]).exists()
    report = json.loads(Path(result["cleaning_report_path"]).read_text(encoding="utf-8"))
    assert Path(report["raw_backup_path"]).exists()
    raw_backup = pd.read_csv(report["raw_backup_path"])
    downloaded_bad_row = raw_backup.loc[raw_backup["ts_code"] == "000001.SZ"].iloc[0]
    assert downloaded_bad_row["vol"] == -1
    assert downloaded_bad_row["amount"] == -100
