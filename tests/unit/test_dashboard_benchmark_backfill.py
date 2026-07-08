from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

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
