from __future__ import annotations

import pandas as pd

from scripts.run_v17_cn_full_a_diagnostic import (
    _date_close_utc,
    _derive_ttm_events,
)


def test_date_close_utc_accepts_compact_and_timestamp_dates() -> None:
    compact = _date_close_utc(pd.Series(["20260724"]))
    timestamp = _date_close_utc(pd.Series([pd.Timestamp("2026-07-24")]))
    assert compact.iloc[0] == pd.Timestamp("2026-07-24T07:00:00Z")
    assert timestamp.iloc[0] == compact.iloc[0]


def test_ttm_events_use_current_ytd_plus_prior_annual_minus_prior_ytd() -> None:
    statements = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20220420",
                "f_ann_date": "20220420",
                "end_date": "20220331",
                "n_income_attr_p": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20230330",
                "f_ann_date": "20230330",
                "end_date": "20221231",
                "n_income_attr_p": 100.0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20230425",
                "f_ann_date": "20230425",
                "end_date": "20230331",
                "n_income_attr_p": 30.0,
            },
        ]
    )
    events = _derive_ttm_events(
        statements,
        value_columns=("n_income_attr_p",),
    )
    latest = events.iloc[-1]
    assert latest["report_end_date"] == "20230331"
    assert latest["n_income_attr_p_ttm"] == 120.0
