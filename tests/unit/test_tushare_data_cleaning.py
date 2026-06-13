from __future__ import annotations

from pathlib import Path

import pandas as pd

import quant_investor.market.tushare_cleaning_core as core
import quant_investor.market.tushare_data_cleaning as tdc
from quant_investor.market.tushare_data_cleaning import (
    CLEANING_STATUS_FAIL,
    CLEANING_STATUS_PASS,
    CLEANING_STATUS_WARN,
    clean_tushare_dataframe,
    clean_tushare_dataframe_to_file,
)


def _valid_daily(rows: list[dict] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        rows
        or [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-03-12",
                "open": 10.2,
                "high": 10.6,
                "low": 10.0,
                "close": 10.4,
                "vol": 1200,
                "amount": 12000,
                "adj_factor": 1.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-03-11",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "vol": 1000,
                "amount": 10000,
                "adj_factor": 1.0,
            },
        ]
    )


def test_cleaning_core_helpers_are_split_and_reused():
    assert tdc._resolve_profile is core._resolve_profile
    assert tdc._normalize_date_value is core._normalize_date_value
    assert tdc._valid_ts_code is core._valid_ts_code
    assert tdc._issue is core._issue
    assert tdc._cell_flag is core._cell_flag
    assert tdc._build_cleaning_report is core._build_cleaning_report


def test_valid_daily_dataframe_passes_and_is_sorted_without_mutating_input():
    frame = _valid_daily()
    original = frame.copy(deep=True)

    cleaned, quarantined, row_flags, cell_flags, report = clean_tushare_dataframe(frame)

    assert report.status == CLEANING_STATUS_PASS
    assert quarantined is None
    assert cleaned["trade_date"].tolist() == ["2026-03-11", "2026-03-12"]
    assert row_flags is not None
    assert cell_flags is not None
    assert frame.equals(original)


def test_cleaning_to_file_skips_empty_cell_flags_artifact(tmp_path):
    result = clean_tushare_dataframe_to_file(
        _valid_daily(),
        canonical_path=tmp_path / "market" / "hs300" / "000001.SZ.csv",
        raw_backup_dir=tmp_path / "raw",
        quarantine_dir=tmp_path / "quarantine",
        report_dir=tmp_path / "reports",
        factor_readiness_dir=tmp_path / "readiness",
        enable_factor_readiness=False,
        enable_storage_audit=False,
        generated_at="2026-03-12T00:00:00Z",
        metadata={"category": "hs300", "symbol": "000001.SZ"},
    )

    report = result["cleaning_report"]
    assert result["cell_flags_df"].empty
    assert result["cell_flags_path"] is None
    assert report.cell_flags_path is None
    assert report.metadata["cell_flags_empty"] is True
    assert report.metadata["cell_flags_path_suppressed"] is True
    assert not Path(report.metadata["cell_flags_planned_path"]).exists()


def test_cleaning_to_file_writes_non_empty_cell_flags_artifact(tmp_path):
    frame = _valid_daily()
    frame.loc[0, "trade_date"] = "bad-date"
    frame.loc[0, "ts_code"] = "bad-code"

    result = clean_tushare_dataframe_to_file(
        frame,
        canonical_path=tmp_path / "market" / "hs300" / "000001.SZ.csv",
        raw_backup_dir=tmp_path / "raw",
        quarantine_dir=tmp_path / "quarantine",
        report_dir=tmp_path / "reports",
        factor_readiness_dir=tmp_path / "readiness",
        enable_factor_readiness=False,
        enable_storage_audit=False,
        generated_at="2026-03-12T00:00:00Z",
        metadata={"category": "hs300", "symbol": "000001.SZ"},
    )

    report = result["cleaning_report"]
    assert not result["cell_flags_df"].empty
    assert result["cell_flags_path"] == report.cell_flags_path
    assert result["cell_flags_path"] is not None
    assert Path(result["cell_flags_path"]).exists()
    assert report.metadata["cell_flags_empty"] is False
    assert report.metadata["cell_flags_path_suppressed"] is False


def test_cleaning_to_file_compacts_large_uniform_row_flags(tmp_path):
    rows = []
    for index in range(120):
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": f"2026-03-{(index % 20) + 1:02d}",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "vol": 1000,
                "amount": 10000,
                "adj_factor": 1.0,
            }
        )
    result = clean_tushare_dataframe_to_file(
        pd.DataFrame(rows),
        canonical_path=tmp_path / "market" / "hs300" / "000001.SZ.csv",
        raw_backup_dir=tmp_path / "raw",
        quarantine_dir=tmp_path / "quarantine",
        report_dir=tmp_path / "reports",
        factor_readiness_dir=tmp_path / "readiness",
        enable_factor_readiness=False,
        enable_storage_audit=False,
        generated_at="2026-03-12T00:00:00Z",
        metadata={"category": "hs300", "symbol": "000001.SZ"},
    )

    report = result["cleaning_report"]
    assert result["row_flags_path"] is None
    assert report.row_flags_path is None
    assert report.metadata["row_flags_compacted"] is True
    assert report.metadata["row_flags_path_suppressed"] is True
    assert report.metadata["row_flags_row_count"] == 120
    assert report.metadata["row_flags_uniform_values"] == {
        "missing_required_column": True,
        "quarantined": False,
        "dropped": False,
    }
    assert not Path(report.metadata["row_flags_planned_path"]).exists()


def test_per_symbol_daily_missing_ts_code_uses_metadata_symbol():
    frame = _valid_daily().drop(columns=["ts_code"])

    cleaned, _quarantined, _row_flags, _cell_flags, report = clean_tushare_dataframe(
        frame,
        metadata={"symbol": "000001.SZ"},
    )

    assert report.status == CLEANING_STATUS_PASS
    assert cleaned["ts_code"].tolist() == ["000001.SZ", "000001.SZ"]


def test_duplicate_and_conflicting_duplicate_are_reported_and_deduplicated():
    frame = _valid_daily(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-03-11",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "vol": 1000,
                "amount": 10000,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-03-11",
                "open": 10.1,
                "high": 10.6,
                "low": 9.9,
                "close": 10.3,
                "vol": 1001,
                "amount": 10001,
            },
        ]
    )

    cleaned, _quarantined, row_flags, _cell_flags, report = clean_tushare_dataframe(frame)

    assert report.status == CLEANING_STATUS_WARN
    assert report.duplicate_row_count == 2
    assert report.conflicting_duplicate_count == 2
    assert len(cleaned) == 1
    assert bool(row_flags["conflicting_duplicate_primary_key"].any())


def test_invalid_date_and_ts_code_are_quarantined_without_fabricating_rows():
    frame = _valid_daily()
    frame.loc[0, "trade_date"] = "bad-date"
    frame.loc[0, "ts_code"] = "bad-code"

    cleaned, quarantined, row_flags, cell_flags, report = clean_tushare_dataframe(frame)

    assert report.status == CLEANING_STATUS_WARN
    assert len(cleaned) == 1
    assert quarantined is not None
    assert len(quarantined) == 1
    assert bool(row_flags["invalid_date"].any())
    assert bool(row_flags["invalid_ts_code"].any())
    assert set(cell_flags["issue_code"]) >= {"invalid_date", "invalid_ts_code"}


def test_negative_volume_and_amount_are_quarantined():
    frame = _valid_daily()
    frame.loc[0, "vol"] = -1
    frame.loc[0, "amount"] = -5

    cleaned, quarantined, row_flags, _cell_flags, report = clean_tushare_dataframe(frame)

    assert report.status == CLEANING_STATUS_WARN
    assert len(cleaned) == 1
    assert quarantined is not None
    assert bool(row_flags["negative_volume"].any())
    assert bool(row_flags["negative_amount"].any())


def test_invalid_ohlc_relation_is_quarantined():
    frame = _valid_daily()
    frame.loc[0, "high"] = 9.0

    cleaned, quarantined, row_flags, _cell_flags, report = clean_tushare_dataframe(frame)

    assert report.status == CLEANING_STATUS_WARN
    assert len(cleaned) == 1
    assert quarantined is not None
    assert bool(row_flags["invalid_ohlc"].any())


def test_missing_required_column_fails_structural_validation():
    frame = _valid_daily().drop(columns=["close"])

    cleaned, quarantined, row_flags, _cell_flags, report = clean_tushare_dataframe(frame)

    assert report.status == CLEANING_STATUS_FAIL
    assert cleaned.empty
    assert quarantined is None
    assert row_flags is not None
    assert row_flags["missing_required_column"].all()
