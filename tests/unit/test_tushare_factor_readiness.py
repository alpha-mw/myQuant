from __future__ import annotations

import pandas as pd

from quant_investor.market.tushare_data_cleaning import (
    FACTOR_READINESS_NOT_READY,
    build_factor_readiness_report,
    build_factor_ready_mask_manifest,
    build_matrix_coverage_summary,
    clean_tushare_dataframe,
)


def _daily(with_adj: bool = True) -> pd.DataFrame:
    rows = [
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
            "ts_code": "000002.SZ",
            "trade_date": "2026-03-12",
            "open": 20.0,
            "high": 20.5,
            "low": 19.8,
            "close": 20.2,
            "vol": 2000,
            "amount": 20000,
        },
    ]
    if with_adj:
        for row in rows:
            row["adj_factor"] = 1.0
    return pd.DataFrame(rows)


def test_factor_ready_mask_manifest_builds_symbol_date_panel():
    cleaned, quarantined, _row_flags, _cell_flags, _report = clean_tushare_dataframe(_daily())

    manifest = build_factor_ready_mask_manifest(
        cleaned,
        quarantined_df=quarantined,
        symbols=["000001.SZ", "000002.SZ"],
        dates=["2026-03-11", "2026-03-12"],
    )

    assert manifest.symbols == ["000001.SZ", "000002.SZ"]
    assert manifest.dates == ["2026-03-11", "2026-03-12"]
    assert manifest.masks["has_row"] == [[True, False], [False, True]]
    assert manifest.masks["tradable"] == [[False, False], [False, False]]


def test_factor_eligible_false_for_quarantined_cells():
    frame = _daily()
    frame.loc[0, "high"] = 9.0
    cleaned, quarantined, _row_flags, _cell_flags, _report = clean_tushare_dataframe(frame)

    manifest = build_factor_ready_mask_manifest(
        cleaned,
        quarantined_df=quarantined,
        symbols=["000001.SZ", "000002.SZ"],
        dates=["2026-03-11", "2026-03-12"],
    )

    assert manifest.masks["has_row"][0][0] is False
    assert manifest.masks["factor_eligible"][0][0] is False


def test_adjusted_price_ready_false_when_adj_factor_missing():
    cleaned, _quarantined, _row_flags, _cell_flags, _report = clean_tushare_dataframe(_daily(with_adj=False))

    manifest = build_factor_ready_mask_manifest(
        cleaned,
        symbols=["000001.SZ"],
        dates=["2026-03-11"],
    )

    assert manifest.masks["adjusted_price_ready"] == [[False]]


def test_matrix_coverage_summary_counts_missing_symbol_date_cells():
    cleaned, quarantined, _row_flags, _cell_flags, _report = clean_tushare_dataframe(_daily())

    summary = build_matrix_coverage_summary(
        cleaned,
        expected_symbols=["000001.SZ", "000002.SZ"],
        expected_dates=["2026-03-11", "2026-03-12"],
        quarantined_df=quarantined,
    )

    assert summary.expected_cell_count == 4
    assert summary.observed_cell_count == 2
    assert summary.missing_cell_count == 2
    assert summary.field_coverage["adj_factor"] == 1.0


def test_readiness_can_fail_while_cleaning_passes():
    cleaned, quarantined, _row_flags, _cell_flags, cleaning_report = clean_tushare_dataframe(
        _daily(with_adj=False)
    )

    readiness = build_factor_readiness_report(
        table_reports={"daily": cleaning_report},
        cleaned_frames={"daily": cleaned},
        quarantined_frames={"daily": quarantined} if quarantined is not None else {},
    )

    assert cleaning_report.status == "pass"
    assert readiness.overall_status == FACTOR_READINESS_NOT_READY
    assert {issue.issue_code for issue in readiness.issues} >= {
        "missing_trade_cal",
        "missing_adj_factor",
    }
