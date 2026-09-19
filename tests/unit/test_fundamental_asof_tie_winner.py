"""When several periods share a disclosure date, the newest one must win.

``merge_asof`` keeps the last right-hand row at or before each trade date, so a
day carrying more than one fiscal period resolves to exactly one of them. The
previous implementation sorted with pandas' default quicksort — unstable — and
took the last row, making the winner arbitrary. Restated vintages republish old
periods under new announcement dates, which took the share of
(ts_code, availability_date) groups holding several periods from 18.07% to
66.98% and pushed the daily panel's wrong-period rate from 2.27% to 2.90%.

Scenarios:
  W01  the newest period wins a shared disclosure date
  W02  input order does not change the winner
  W03  distinct disclosure dates all survive
  W04  symbols are independent
  W05  a frame without end_date still collapses one row per date
  W06  an empty frame passes through
"""

import pandas as pd

from quant_investor.market.fundamental_mart import _legacy_asof_tie_winners


def _rows(records: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [r[0] for r in records],
            "availability_date": pd.to_datetime([r[1] for r in records]),
            "end_date": [r[2] for r in records],
        }
    )


def test_W01_newest_period_wins_a_shared_disclosure_date():
    # An annual report and the following quarter released the same morning.
    won = _legacy_asof_tie_winners(
        _rows([
            ("600000.SH", "2023-04-29", "20221231"),
            ("600000.SH", "2023-04-29", "20230331"),
        ])
    )
    assert len(won) == 1
    assert won["end_date"].iloc[0] == "20230331"


def test_W02_input_order_does_not_change_the_winner():
    forward = _legacy_asof_tie_winners(
        _rows([
            ("600000.SH", "2023-04-29", "20221231"),
            ("600000.SH", "2023-04-29", "20230331"),
        ])
    )
    reversed_ = _legacy_asof_tie_winners(
        _rows([
            ("600000.SH", "2023-04-29", "20230331"),
            ("600000.SH", "2023-04-29", "20221231"),
        ])
    )
    assert forward["end_date"].tolist() == reversed_["end_date"].tolist() == ["20230331"]


def test_W03_distinct_disclosure_dates_all_survive():
    won = _legacy_asof_tie_winners(
        _rows([
            ("600000.SH", "2023-04-29", "20221231"),
            ("600000.SH", "2023-08-30", "20230630"),
            ("600000.SH", "2023-10-27", "20230930"),
        ])
    )
    assert len(won) == 3
    assert won.sort_values("availability_date")["end_date"].tolist() == [
        "20221231", "20230630", "20230930",
    ]


def test_W04_symbols_are_independent():
    won = _legacy_asof_tie_winners(
        _rows([
            ("600000.SH", "2023-04-29", "20221231"),
            ("600000.SH", "2023-04-29", "20230331"),
            ("000001.SZ", "2023-04-29", "20221231"),
        ])
    )
    assert len(won) == 2
    by_symbol = dict(zip(won["ts_code"], won["end_date"]))
    assert by_symbol["600000.SH"] == "20230331"
    assert by_symbol["000001.SZ"] == "20221231"


def test_W05_frame_without_end_date_still_collapses_one_row_per_date():
    frame = _rows([
        ("600000.SH", "2023-04-29", "20221231"),
        ("600000.SH", "2023-04-29", "20230331"),
    ]).drop(columns=["end_date"])
    won = _legacy_asof_tie_winners(frame)
    assert len(won) == 1


def test_W06_empty_frame_passes_through():
    assert _legacy_asof_tie_winners(pd.DataFrame()).empty
