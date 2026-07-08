from __future__ import annotations

import json

import pandas as pd

from quant_investor.market.daily_cleaner import (
    DailyCleanConfig,
    clean_market_daily_data,
    latest_download_report_target,
)


def test_cn_cleaner_drops_invalid_rows_and_quarantines_adjustment_gaps(tmp_path):
    raw_dir = tmp_path / "cn_market_full"
    source_dir = raw_dir / "hs300"
    source_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-05-27",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": None,
                "vol": 1000,
                "amount": 10000,
                "adj_factor": 1.0,
                "adj_close": 10.5,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "1970-01-01",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": None,
                "vol": 1000,
                "amount": 10000,
                "adj_factor": 1.0,
                "adj_close": 10.5,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-05-28",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": None,
                "vol": 1000,
                "amount": 10000,
                "adj_factor": None,
                "adj_close": None,
            },
        ]
    ).to_parquet(source_dir / "000001.SZ.parquet", index=False)

    manifest = clean_market_daily_data(
        DailyCleanConfig(
            market="CN",
            raw_dir=raw_dir,
            clean_dir=tmp_path / "clean" / "cn_daily",
            audit_dir=tmp_path / "audit" / "cn",
            latest_required_date="2026-05-28",
        )
    )

    clean = pd.read_csv(tmp_path / "clean" / "cn_daily" / "hs300" / "000001.SZ.csv")
    quarantine_symbols = pd.read_csv(tmp_path / "audit" / "cn" / "quarantine_symbols.csv")
    quarantine_rows = pd.read_csv(tmp_path / "audit" / "cn" / "quarantine_rows.csv")

    assert manifest["totals"]["dropped_rows"] == 1
    assert clean["trade_date"].tolist() == ["2026-05-27", "2026-05-28"]
    assert "missing_adjustment" in quarantine_symbols["quarantine_reasons"].iloc[0]
    assert quarantine_rows["issues"].tolist() == ["invalid_date"]


def test_us_cleaner_uses_full_us_as_canonical_and_records_membership(tmp_path):
    raw_dir = tmp_path / "us_market_full"
    (raw_dir / "full_us").mkdir(parents=True)
    (raw_dir / "small_cap").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Date": "2026-05-27",
                "Open": 10,
                "High": 11,
                "Low": 9,
                "Close": 10.5,
                "Volume": 1000,
            }
        ]
    ).to_parquet(raw_dir / "full_us" / "ABC.parquet", index=False)
    pd.DataFrame(
        [
            {
                "Date": "2026-04-01",
                "Open": 0,
                "High": 0,
                "Low": 0,
                "Close": 1,
                "Volume": 100,
            }
        ]
    ).to_parquet(raw_dir / "small_cap" / "ABC.parquet", index=False)

    manifest = clean_market_daily_data(
        DailyCleanConfig(
            market="US",
            raw_dir=raw_dir,
            clean_dir=tmp_path / "clean" / "us_daily",
            audit_dir=tmp_path / "audit" / "us",
            latest_required_date="2026-05-27",
        )
    )

    clean = pd.read_csv(tmp_path / "clean" / "us_daily" / "ABC.csv")
    membership = pd.read_csv(tmp_path / "audit" / "us" / "membership.csv")

    assert manifest["totals"]["source_file_count"] == 2
    assert manifest["totals"]["processed_file_count"] == 1
    assert manifest["totals"]["duplicate_symbol_count"] == 1
    assert clean["trade_date"].tolist() == ["2026-05-27"]
    assert membership["has_duplicate_storage"].iloc[0].item() is True
    assert membership["canonical_category"].iloc[0] == "full_us"


def test_latest_download_report_target_reads_cn_and_us_reports(tmp_path):
    cn_dir = tmp_path / "cn"
    us_dir = tmp_path / "us"
    cn_dir.mkdir()
    us_dir.mkdir()
    (cn_dir / "download_report_20260528_094001.json").write_text(
        json.dumps({"config": {"strict_trade_date": "20260528"}}),
        encoding="utf-8",
    )
    (us_dir / "download_report_20260527_120110.json").write_text(
        json.dumps(
            {
                "categories": {
                    "full_us": [
                        {"symbol": "ABC", "latest_date": "2026-05-26"},
                        {"symbol": "XYZ", "latest_date": "2026-05-27"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert latest_download_report_target(cn_dir, "CN") == "2026-05-28"
    assert latest_download_report_target(us_dir, "US") == "2026-05-27"
