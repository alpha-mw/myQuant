"""The daily projection must never travel backwards in report period.

``fundamental_daily`` joins period rows onto trade dates with
``merge_asof(direction="backward")`` keyed on ``availability_date`` alone, with
no regard for ``end_date``. That is safe only while every period is disclosed
once. With restated vintages in the period table, a restatement of 2023Q1
announced 2024-04-30 carries a later availability_date than the first filing of
2023Q2 — so without the guard, every trade date after it would be served a
year-old report period.

Scenarios:
  D01  a restatement of a superseded period does not displace the newer one
  D02  a restatement of the still-current period does take effect
  D03  the period stream stays monotone in end_date per symbol
  D04  symbols are independent
  D05  a single-vintage stream is untouched
"""

import pandas as pd

from quant_investor.market.fundamental_mart import _drop_superseded_period_vintages

Q1, Q2 = "20230331", "20230630"
Q1_FILED_AT, Q2_FILED_AT, Q1_RESTATED_AT = "2023-04-27", "2023-08-30", "2024-04-30"


def _period(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [row[0] for row in rows],
            "end_date": [row[1] for row in rows],
            "availability_date": pd.to_datetime([row[2] for row in rows]),
            "fin_net_profit": [row[3] for row in rows],
        }
    )


def test_D01_superseded_period_restatement_is_dropped():
    kept = _drop_superseded_period_vintages(
        _period(
            [
                ("600771.SH", Q1, Q1_FILED_AT, 9.6),
                ("600771.SH", Q2, Q2_FILED_AT, 53.1),
                ("600771.SH", Q1, Q1_RESTATED_AT, 28.2),  # older period, newer date
            ]
        )
    )
    assert list(kept["end_date"]) == [Q1, Q2]
    assert Q1_RESTATED_AT not in set(kept["availability_date"].astype(str))


def test_D02_current_period_restatement_survives():
    """Restating the newest period must still reach the daily panel."""
    kept = _drop_superseded_period_vintages(
        _period(
            [
                ("600771.SH", Q1, Q1_FILED_AT, 9.6),
                ("600771.SH", Q1, Q1_RESTATED_AT, 28.2),
            ]
        )
    )
    assert len(kept) == 2
    assert kept["fin_net_profit"].tolist() == [9.6, 28.2]


def test_D03_end_date_is_monotone_per_symbol():
    kept = _drop_superseded_period_vintages(
        _period(
            [
                ("600771.SH", Q1, Q1_FILED_AT, 9.6),
                ("600771.SH", Q2, Q2_FILED_AT, 53.1),
                ("600771.SH", Q1, Q1_RESTATED_AT, 28.2),
                ("600771.SH", "20230930", "2023-10-25", 80.3),
            ]
        )
    )
    ordered = kept.sort_values("availability_date")["end_date"].tolist()
    assert ordered == sorted(ordered)


def test_D04_symbols_do_not_interfere():
    kept = _drop_superseded_period_vintages(
        _period(
            [
                ("600771.SH", Q2, Q2_FILED_AT, 53.1),
                ("000001.SZ", Q1, Q1_RESTATED_AT, 28.2),
            ]
        )
    )
    assert len(kept) == 2


def test_D05_single_vintage_stream_is_untouched():
    frame = _period(
        [
            ("600771.SH", Q1, Q1_FILED_AT, 9.6),
            ("600771.SH", Q2, Q2_FILED_AT, 53.1),
        ]
    )
    kept = _drop_superseded_period_vintages(frame)
    assert len(kept) == len(frame)


def test_D06_empty_frame_passes_through():
    assert _drop_superseded_period_vintages(pd.DataFrame()).empty
