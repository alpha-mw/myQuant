"""
下载器单元测试
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import datetime, timedelta

import pandas as pd


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
        sys.modules.pop(module_name, None)
    module_name = "quant_investor.market.download_cn"
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


def test_download_stock_full_a_targets_resolved_existing_path(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)

    hs300_file = tmp_path / "hs300" / "000001.SZ.csv"
    zz500_file = tmp_path / "zz500" / "000001.SZ.csv"
    hs300_file.parent.mkdir(parents=True, exist_ok=True)
    zz500_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 9.0}]).to_csv(hs300_file, index=False)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 8.0}]).to_csv(zz500_file, index=False)

    result = downloader.download_stock("000001.SZ", "full_a")

    assert result["status"] == "updated"
    assert hs300_file.exists()
    assert zz500_file.exists()
    assert pd.read_csv(hs300_file)["trade_date"].iloc[-1] == "2026-03-16"
    assert pd.read_csv(zz500_file)["close"].iloc[-1] == 8.0
    assert not (tmp_path / "full_a").exists()


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
    assert (tmp_path / "hs300" / "000001.SZ.csv").exists()
    assert not (tmp_path / "other" / "000001.SZ.csv").exists()


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
    assert (tmp_path / "hs300" / "000001.SZ.csv").exists()
    assert not (tmp_path / "other" / "000001.SZ.csv").exists()


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


def test_build_completeness_report_stable_mode_rolls_back_when_strict_coverage_is_below_threshold(
    monkeypatch,
    tmp_path,
):
    module = _load_module(monkeypatch, freshness_mode="stable", coverage_threshold="0.95")
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    latest_path = tmp_path / "hs300" / "000001.SZ.csv"
    stable_path = tmp_path / "hs300" / "000002.SZ.csv"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(latest_path, index=False)
    pd.DataFrame([{"trade_date": "2026-03-14", "close": 10.0}]).to_csv(stable_path, index=False)

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
    latest_path = tmp_path / "hs300" / "000001.SZ.csv"
    stable_path = tmp_path / "hs300" / "000002.SZ.csv"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(latest_path, index=False)
    pd.DataFrame([{"trade_date": "2026-03-14", "close": 10.0}]).to_csv(stable_path, index=False)

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
    latest_path = tmp_path / "hs300" / "000001.SZ.csv"
    stable_path = tmp_path / "hs300" / "000002.SZ.csv"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(latest_path, index=False)
    pd.DataFrame([{"trade_date": "2026-03-14", "close": 10.0}]).to_csv(stable_path, index=False)

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
    file_path = tmp_path / "hs300" / "000002.SZ.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-14", "close": 10.0}]).to_csv(file_path, index=False)

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
    base_dir = tmp_path / "hs300"
    base_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(base_dir / "000001.SZ.csv", index=False)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 10.0}]).to_csv(base_dir / "000002.SZ.csv", index=False)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 10.0}]).to_csv(base_dir / "000003.SZ.csv", index=False)

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


def test_full_a_local_universe_uses_existing_local_cache(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)

    for category, symbol in [
        ("hs300", "000001.SZ"),
        ("zz500", "000002.SZ"),
        ("other", "600001.SH"),
    ]:
        file_path = tmp_path / category / f"{symbol}.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "trade_date": "2026-03-16",
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
        ).to_csv(file_path, index=False)

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

    hs300_file = tmp_path / "hs300" / "000001.SZ.csv"
    hs300_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 10.0}]).to_csv(hs300_file, index=False)

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
        file_path = tmp_path / category / f"{symbol}.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(file_path, index=False)

    assert not (tmp_path / "full_a").exists()
    symbols = get_all_local_symbols("full_a", market="CN", data_dir=str(tmp_path))
    assert symbols == ["000001.SZ", "000002.SZ", "600001.SH"]


def test_get_all_components_falls_back_to_local_universe(monkeypatch):
    module = importlib.import_module("quant_investor.fetch_cn_index_components")
    monkeypatch.setattr(module, "fetch_full_a", lambda _pro: [])
    monkeypatch.setattr(module, "fetch_hs300", lambda _pro: [])
    monkeypatch.setattr(module, "fetch_zz500", lambda _pro: [])
    monkeypatch.setattr(module, "fetch_zz1000", lambda _pro: [])
    class _FakeResolver:
        def __init__(self, *args, **kwargs):
            self.trace = types.SimpleNamespace(
                directory_priority=["hs300", "zz500", "zz1000", "other"],
                physical_directories_used_for_full_a=[],
                local_union_fallback_used=False,
                resolved_file_paths_by_symbol={},
                resolution_strategy="",
            )

        def collect_full_a_inventory(self, local_union_fallback_used=False):
            return ["000001.SZ", "600001.SH"], {
                "000001.SZ": "/tmp/hs300/000001.SZ.csv",
                "600001.SH": "/tmp/other/600001.SH.csv",
            }

        def physical_directories_for_full_a(self):
            return []

        def snapshot(self):
            return {
                "directory_priority": ["hs300", "zz500", "zz1000", "other"],
                "physical_directories_used_for_full_a": ["hs300", "other"],
                "local_union_fallback_used": True,
                "resolved_file_paths_by_symbol": {
                    "000001.SZ": "/tmp/hs300/000001.SZ.csv",
                    "600001.SH": "/tmp/other/600001.SH.csv",
                },
                "resolution_strategy": "local_union",
            }

    monkeypatch.setattr(module, "CNUniverseResolver", _FakeResolver)

    components = module.get_all_components(pro=object())

    assert components["full_a"] == ["000001.SZ", "600001.SH"]
    assert components["all"] == ["000001.SZ", "600001.SH"]
    assert components["stats"]["full_a"] == 2
    assert components["stats"]["total_unique"] == 2
    assert components["resolver"]["local_union_fallback_used"] is True


def test_evaluate_symbol_local_status_covers_fixed_contract(tmp_path):
    from quant_investor.market.cn_resolver import CNUniverseResolver
    from quant_investor.market.cn_symbol_status import evaluate_symbol_local_status
    from quant_investor.market.shared_csv_reader import SharedCSVReader

    latest_file = tmp_path / "hs300" / "000001.SZ.csv"
    stale_file = tmp_path / "hs300" / "000002.SZ.csv"
    unreadable_file = tmp_path / "hs300" / "000003.SZ.csv"
    suspended_file = tmp_path / "hs300" / "000005.SZ.csv"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(latest_file, index=False)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 10.0}]).to_csv(stale_file, index=False)
    pd.DataFrame([{"close": 10.0}]).to_csv(unreadable_file, index=False)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 9.5}]).to_csv(suspended_file, index=False)

    resolver = CNUniverseResolver(data_dir=str(tmp_path))
    csv_reader = SharedCSVReader(market="CN", data_dir=str(tmp_path), resolver=resolver)

    up_to_date = evaluate_symbol_local_status(
        "000001.SZ",
        category="hs300",
        resolver=resolver,
        csv_reader=csv_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols=set(),
        suspended_symbols=set(),
    )
    stale = evaluate_symbol_local_status(
        "000002.SZ",
        category="hs300",
        resolver=resolver,
        csv_reader=csv_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols=set(),
        suspended_symbols=set(),
    )
    unreadable = evaluate_symbol_local_status(
        "000003.SZ",
        category="hs300",
        resolver=resolver,
        csv_reader=csv_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols=set(),
        suspended_symbols=set(),
    )
    missing = evaluate_symbol_local_status(
        "000004.SZ",
        category="hs300",
        resolver=resolver,
        csv_reader=csv_reader,
        latest_trade_date="20260316",
        allowed_stale_symbols={"000004.SZ"},
        suspended_symbols=set(),
    )
    suspended_stale = evaluate_symbol_local_status(
        "000005.SZ",
        category="hs300",
        resolver=resolver,
        csv_reader=csv_reader,
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
    file_path = tmp_path / "hs300" / "000001.SZ.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-15", "close": 10.0}]).to_csv(file_path, index=False)
    monkeypatch.setattr(downloader, "_fetch_stock_frame", lambda *_args, **_kwargs: pd.DataFrame())

    result = downloader.download_stock("000001.SZ", "hs300")

    assert result["status"] == "stale_cached"
    assert result["local_status"] == "stale_cached"
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
    assert (data_dir / "hs300" / "000001.SZ.csv").exists()
    assert not (data_dir / "other" / "000001.SZ.csv").exists()


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
    assert (tmp_path / "hs300" / "000001.SZ.csv").exists()
    assert not (tmp_path / "other" / "000001.SZ.csv").exists()


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
    hs300_dir = tmp_path / "hs300"
    hs300_dir.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        pd.DataFrame([{"trade_date": "2026-03-14", "close": 10.0}]).to_csv(
            hs300_dir / f"{symbol}.csv",
            index=False,
        )

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
    assert result["completeness"]["latest_trade_date"] == "20260314"
    assert result["completeness"]["effective_target_trade_date"] == "20260314"
    assert result["completeness"]["complete"] is True
    assert downloader.stats["stale_cached"] == 10


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
        sys.modules.pop(module_name, None)

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
        sys.modules.pop(module_name, None)

    fake_tushare = types.SimpleNamespace(pro_api=lambda token: object())
    monkeypatch.setitem(sys.modules, "tushare", fake_tushare)

    market_config_module = importlib.import_module("quant_investor.market.config")
    assert market_config_module.get_market_settings("CN").data_dir == str(env_data_dir)

    download_module = importlib.import_module("quant_investor.market.download_cn")
    monkeypatch.setattr(download_module, "create_tushare_pro", lambda *_args, **_kwargs: FakePro())
    downloader = download_module.CNFullMarketDownloader(years=3)
    assert downloader.data_dir == str(env_data_dir)

    sys.modules.pop("quant_investor.fetch_cn_index_components", None)
    components_module = importlib.import_module("quant_investor.fetch_cn_index_components")
    captured: dict[str, str] = {}

    class _FakeResolver:
        def __init__(self, data_dir, *args, **kwargs):
            captured["data_dir"] = data_dir
            self.trace = types.SimpleNamespace(
                directory_priority=["hs300", "zz500", "zz1000", "other"],
                physical_directories_used_for_full_a=[],
                local_union_fallback_used=False,
                resolved_file_paths_by_symbol={},
                resolution_strategy="",
            )

        def collect_full_a_inventory(self, local_union_fallback_used=False):
            return [], {}

        def physical_directories_for_full_a(self):
            return []

        def snapshot(self):
            return {
                "directory_priority": ["hs300", "zz500", "zz1000", "other"],
                "physical_directories_used_for_full_a": [],
                "local_union_fallback_used": False,
                "resolved_file_paths_by_symbol": {},
                "resolution_strategy": "upstream_fetch",
            }

    monkeypatch.setattr(components_module, "CNUniverseResolver", _FakeResolver)
    monkeypatch.setattr(components_module, "fetch_full_a", lambda _pro: [])
    monkeypatch.setattr(components_module, "fetch_hs300", lambda _pro: [])
    monkeypatch.setattr(components_module, "fetch_zz500", lambda _pro: [])
    monkeypatch.setattr(components_module, "fetch_zz1000", lambda _pro: [])
    components_module.get_all_components(pro=object())

    assert captured["data_dir"] == str(env_data_dir)


def test_build_completeness_report_counts_cached_symbol_as_complete(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    file_path = tmp_path / "hs300" / "000001.SZ.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(file_path, index=False)
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
    (tmp_path / "hs300").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-16", "close": 10.0}]).to_csv(
        tmp_path / "hs300" / "000001.SZ.csv", index=False
    )

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


def test_freshness_index_written_after_download(monkeypatch, tmp_path):
    """download_category() should update the freshness index for every processed symbol."""
    module = _load_module(monkeypatch)
    fake_pro = FakePro()
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: fake_pro)

    downloader = module.CNFullMarketDownloader(data_dir=str(tmp_path), years=3)
    (tmp_path / "hs300").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_date": "2026-03-14", "close": 10.0}]).to_csv(
        tmp_path / "hs300" / "000001.SZ.csv", index=False
    )

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
