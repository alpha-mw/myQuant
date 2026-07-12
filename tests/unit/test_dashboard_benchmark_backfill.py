from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_backfill():
    spec = importlib.util.spec_from_file_location(
        "dashboard_benchmark_backfill",
        ROOT / "scripts" / "backfill_cn_dashboard_benchmark.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTusharePro:
    def index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        assert start_date == "20260215"
        assert end_date == "20260217"
        base = {"000300.SH": 1000.0, "000688.SH": 4000.0}[ts_code]
        return pd.DataFrame(
            [
                {"ts_code": ts_code, "trade_date": "20260216", "close": base},
                {"ts_code": ts_code, "trade_date": "20260217", "close": base * 1.01},
            ]
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_backfill_writes_verified_tushare_rows_and_preserves_existing_codes(tmp_path):
    backfill = _load_backfill()
    output = tmp_path / "cn_index_benchmark.csv"
    output.write_text(
        "\n".join(
            [
                "date,ts_code,close,source_system,value_date,coverage",
                "2026-02-16,399006.SZ,5000,tushare.index_daily,2026-02-16,exact_close",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = backfill.backfill_benchmark_file(
        _FakeTusharePro(),
        output_file=output,
        start_date="2026-02-15",
        end_date="2026-02-17",
        ts_codes=("000300.SH", "000688.SH"),
    )

    rows = _read_csv(output)
    assert summary["pulled_row_count"] == 4
    assert summary["existing_row_count"] == 1
    assert summary["output_row_count"] == 5
    assert rows[0] == {
        "date": "2026-02-16",
        "ts_code": "000300.SH",
        "close": "1000.000000",
        "source_system": "tushare.index_daily",
        "value_date": "2026-02-16",
        "coverage": "exact_close",
    }
    assert {row["ts_code"] for row in rows} == {"000300.SH", "000688.SH", "399006.SZ"}


def test_eastmoney_backfill_only_fills_missing_rows(tmp_path):
    backfill = _load_backfill()
    output = tmp_path / "cn_index_benchmark.csv"
    output.write_text(
        "\n".join(
            [
                "date,ts_code,close,source_system,value_date,coverage",
                "2026-06-26,000300.SH,4868.220000,tushare.index_daily,2026-06-26,exact_close",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_fetch_json(url: str):
        assert "secid=1.000300" in url
        return {
            "data": {
                "klines": [
                    "2026-06-26,4981.83,4868.22,5030.12,4850.01,1,2,3,4,5,6",
                    "2026-06-29,4866.04,4926.92,4930.00,4800.00,1,2,3,4,5,6",
                ]
            }
        }

    pulled = backfill.pull_eastmoney_kline_rows(
        start_date="2026-06-26",
        end_date="2026-06-29",
        ts_codes=("000300.SH",),
        fetch_json=fake_fetch_json,
    )
    assert pulled == [
        {
            "date": "2026-06-26",
            "ts_code": "000300.SH",
            "close": "4868.220000",
            "source_system": "eastmoney.push2his.kline",
            "value_date": "2026-06-26",
            "coverage": "exact_close",
        },
        {
            "date": "2026-06-29",
            "ts_code": "000300.SH",
            "close": "4926.920000",
            "source_system": "eastmoney.push2his.kline",
            "value_date": "2026-06-29",
            "coverage": "exact_close",
        },
    ]

    original_pull = backfill.pull_eastmoney_kline_rows
    try:
        backfill.pull_eastmoney_kline_rows = lambda **kwargs: pulled
        summary = backfill.backfill_benchmark_file(
            None,
            output_file=output,
            start_date="2026-06-26",
            end_date="2026-06-29",
            ts_codes=("000300.SH",),
            source="eastmoney",
            replace_existing=False,
        )
    finally:
        backfill.pull_eastmoney_kline_rows = original_pull

    rows = _read_csv(output)
    assert summary["source_system"] == "eastmoney.push2his.kline"
    assert summary["replace_existing"] is False
    assert rows == [
        {
            "date": "2026-06-26",
            "ts_code": "000300.SH",
            "close": "4868.220000",
            "source_system": "tushare.index_daily",
            "value_date": "2026-06-26",
            "coverage": "exact_close",
        },
        {
            "date": "2026-06-29",
            "ts_code": "000300.SH",
            "close": "4926.920000",
            "source_system": "eastmoney.push2his.kline",
            "value_date": "2026-06-29",
            "coverage": "exact_close",
        },
    ]


def test_eastmoney_kline_falls_back_to_same_domain_http_after_https_failure():
    backfill = _load_backfill()
    urls: list[str] = []

    def fake_fetch_json(url: str):
        urls.append(url)
        if url.startswith("https://"):
            raise RuntimeError("https unavailable")
        return {
            "data": {
                "klines": [
                    "2026-07-10,4041.86,3842.73,4060.53,3842.35,1,2,3,4,5,6",
                ]
            }
        }

    rows = backfill.pull_eastmoney_kline_rows(
        start_date="2026-07-10",
        end_date="2026-07-10",
        ts_codes=("399006.SZ",),
        fetch_json=fake_fetch_json,
    )

    assert [url.split(":", 1)[0] for url in urls] == ["https", "http"]
    assert rows[0]["close"] == "3842.730000"
    assert rows[0]["source_system"] == "eastmoney.push2his.kline"


def test_eastmoney_quote_rows_use_timestamp_and_verified_exact_date_axis():
    backfill = _load_backfill()

    def fake_fetch_json(url: str):
        assert "secid=0.399006" in url
        return {
            "data": {
                "f43": 384273,
                "f57": "399006",
                "f58": "创业板指",
                "f59": 2,
                "f60": 401817,
                "f86": 1783671087,
            }
        }

    rows = backfill.pull_eastmoney_quote_rows(
        start_date="2026-07-09",
        end_date="2026-07-10",
        exact_dates=("2026-07-09", "2026-07-10"),
        ts_codes=("399006.SZ",),
        fetch_json=fake_fetch_json,
    )

    assert rows == [
        {
            "date": "2026-07-09",
            "ts_code": "399006.SZ",
            "close": "4018.170000",
            "source_system": "eastmoney.push2.quote",
            "value_date": "2026-07-09",
            "coverage": "exact_close",
        },
        {
            "date": "2026-07-10",
            "ts_code": "399006.SZ",
            "close": "3842.730000",
            "source_system": "eastmoney.push2.quote",
            "value_date": "2026-07-10",
            "coverage": "exact_close",
        },
    ]


def test_eastmoney_backfill_uses_quote_when_kline_is_unavailable(tmp_path, monkeypatch):
    backfill = _load_backfill()
    output = tmp_path / "cn_index_benchmark.csv"
    output.write_text(
        "\n".join(
            [
                "date,ts_code,close,source_system,value_date,coverage",
                "2026-07-09,000300.SH,4876.31,tushare.index_daily,2026-07-09,exact_close",
                "2026-07-10,000300.SH,4780.79,tushare.index_daily,2026-07-10,exact_close",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        backfill,
        "pull_eastmoney_kline_rows",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("kline unavailable")),
    )
    monkeypatch.setattr(
        backfill,
        "pull_eastmoney_quote_rows",
        lambda **_kwargs: [
            {
                "date": "2026-07-09",
                "ts_code": "399006.SZ",
                "close": "4018.170000",
                "source_system": "eastmoney.push2.quote",
                "value_date": "2026-07-09",
                "coverage": "exact_close",
            },
            {
                "date": "2026-07-10",
                "ts_code": "399006.SZ",
                "close": "3842.730000",
                "source_system": "eastmoney.push2.quote",
                "value_date": "2026-07-10",
                "coverage": "exact_close",
            },
        ],
    )

    summary = backfill.backfill_benchmark_file(
        None,
        output_file=output,
        start_date="2026-07-09",
        end_date="2026-07-10",
        ts_codes=("399006.SZ",),
        source="eastmoney",
        replace_existing=False,
    )

    rows = _read_csv(output)
    assert summary["source_system"] == "eastmoney.push2.quote"
    assert summary["replace_existing"] is False
    assert summary["pulled_row_count"] == 2
    assert summary["provider_warnings"] == ["push2his.kline unavailable: kline unavailable"]
    assert [row["date"] for row in rows if row["ts_code"] == "399006.SZ"] == [
        "2026-07-09",
        "2026-07-10",
    ]


def test_fetch_json_uses_curl_fallback_without_shell(monkeypatch):
    backfill = _load_backfill()
    calls = []

    def fail_urlopen(*_args, **_kwargs):
        raise OSError("urllib unavailable")

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"data":{"klines":[]}}', stderr="")

    monkeypatch.setattr(backfill, "urlopen", fail_urlopen)
    monkeypatch.setattr(backfill.subprocess, "run", fake_run)

    payload = backfill._fetch_json("http://push2his.eastmoney.com/test", retries=1)

    assert payload == {"data": {"klines": []}}
    assert calls[0][0][-1] == "http://push2his.eastmoney.com/test"
    assert calls[0][1]["check"] is False
    assert "shell" not in calls[0][1]


def test_tushare_backfill_adds_auditable_previous_trading_day_ffill(tmp_path):
    backfill = _load_backfill()
    output = tmp_path / "cn_index_benchmark.csv"
    output.write_text(
        "\n".join(
            [
                "date,ts_code,close,source_system,value_date,coverage",
                "2026-07-10,000300.SH,5100,tushare.index_daily,2026-07-10,exact_close",
                "2026-07-10,000688.SH,2200,eastmoney.push2his.kline,2026-07-10,exact_close",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeWeekendPro:
        def index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
            assert ts_code in {"000300.SH", "000688.SH"}
            assert start_date == end_date == "20260712"
            return pd.DataFrame()

        def trade_cal(self, exchange: str, start_date: str, end_date: str) -> pd.DataFrame:
            assert exchange == ""
            assert start_date == end_date == "20260712"
            return pd.DataFrame(
                [{"cal_date": "20260712", "is_open": 0, "pretrade_date": "20260710"}]
            )

    summary = backfill.backfill_benchmark_file(
        FakeWeekendPro(),
        output_file=output,
        start_date="2026-07-12",
        end_date="2026-07-12",
        ts_codes=("000300.SH", "000688.SH"),
        replace_existing=False,
    )

    rows = _read_csv(output)
    weekend_rows = [row for row in rows if row["date"] == "2026-07-12"]
    assert summary["replace_existing"] is False
    assert summary["calendar_source_system"] == "tushare.trade_cal"
    assert summary["previous_trading_day_ffill_row_count"] == 2
    assert weekend_rows == [
        {
            "date": "2026-07-12",
            "ts_code": "000300.SH",
            "close": "5100",
            "source_system": "tushare.index_daily",
            "value_date": "2026-07-10",
            "coverage": "previous_trading_day_ffill",
        },
        {
            "date": "2026-07-12",
            "ts_code": "000688.SH",
            "close": "2200",
            "source_system": "eastmoney.push2his.kline",
            "value_date": "2026-07-10",
            "coverage": "previous_trading_day_ffill",
        },
    ]
