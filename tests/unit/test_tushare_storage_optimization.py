from __future__ import annotations

import pandas as pd

import quant_investor.market.tushare_data_cleaning as tdc
import quant_investor.market.tushare_cleaning_storage as storage
from quant_investor.market.tushare_data_cleaning import (
    PARQUET_STATUS_SHADOW_WRITTEN,
    PARQUET_STATUS_UNSUPPORTED,
    STORAGE_STATUS_WARN,
    TushareStorageOptimizationConfig,
    build_storage_audit_report,
    detect_parquet_backend,
    write_parquet_shadow_if_supported,
)


def test_storage_helpers_are_split_and_reexported():
    assert tdc.detect_parquet_backend is storage.detect_parquet_backend
    assert tdc.write_parquet_shadow_if_supported is storage.write_parquet_shadow_if_supported
    assert tdc.build_storage_audit_report is storage.build_storage_audit_report
    assert tdc.safe_json_dump is storage.safe_json_dump
    assert tdc.sha256_file is storage.sha256_file
    assert tdc.atomic_write_dataframe_csv is storage.atomic_write_dataframe_csv


def test_detect_parquet_backend_returns_status_without_failing():
    supported, backend, warnings = detect_parquet_backend()

    assert isinstance(supported, bool)
    assert backend is None or backend in {"pyarrow", "fastparquet"}
    assert isinstance(warnings, list)


def test_storage_audit_recommends_parquet_for_large_matrix_table_when_backend_supported(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(storage, "detect_parquet_backend", lambda: (True, "pyarrow", []))
    csv_path = tmp_path / "daily.csv"
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-03-11",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
            }
        ]
    ).to_csv(csv_path, index=False)

    report = build_storage_audit_report(
        table_name="daily",
        csv_path=csv_path,
        config=TushareStorageOptimizationConfig(
            min_csv_size_for_parquet_bytes=0,
            min_rows_for_parquet=0,
        ),
        generated_at="2026-03-12T00:00:00Z",
    )

    assert report.recommended_storage_format == "parquet"
    assert report.status == STORAGE_STATUS_WARN


def test_parquet_write_skips_cleanly_when_backend_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "detect_parquet_backend", lambda: (False, None, ["missing"]))
    csv_path = tmp_path / "daily.csv"
    csv_path.write_text("ts_code,trade_date\n000001.SZ,2026-03-11\n", encoding="utf-8")

    report = write_parquet_shadow_if_supported(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "2026-03-11"}]),
        table_name="daily",
        csv_path=csv_path,
        parquet_path=tmp_path / "daily.parquet",
        config=TushareStorageOptimizationConfig(parquet_shadow_write=True),
        generated_at="2026-03-12T00:00:00Z",
    )

    assert report.status == PARQUET_STATUS_UNSUPPORTED
    assert csv_path.exists()


def test_parquet_shadow_write_and_readback_when_backend_supported(tmp_path):
    supported, _backend, _warnings = detect_parquet_backend()
    csv_path = tmp_path / "daily.csv"
    csv_path.write_text("ts_code,trade_date\n000001.SZ,2026-03-11\n", encoding="utf-8")
    parquet_path = tmp_path / "daily.parquet"

    report = write_parquet_shadow_if_supported(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "2026-03-11"}]),
        table_name="daily",
        csv_path=csv_path,
        parquet_path=parquet_path,
        config=TushareStorageOptimizationConfig(parquet_shadow_write=True),
        generated_at="2026-03-12T00:00:00Z",
    )

    if supported:
        assert report.status == PARQUET_STATUS_SHADOW_WRITTEN
        assert report.readback_validated is True
        assert parquet_path.exists()
    else:
        assert report.status == PARQUET_STATUS_UNSUPPORTED
    assert csv_path.exists()


def test_csv_deletion_and_canonical_parquet_disabled_by_default():
    config = TushareStorageOptimizationConfig()

    assert config.parquet_canonical is False
    assert config.delete_redundant_csv is False
