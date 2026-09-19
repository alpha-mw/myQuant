"""Stored rows must be in the order a PIT replay reproduces.

Promotion re-runs ``_strict_pit_cutoff`` over the rows a rebuild stored and
requires the result to be identical — same rows, same order, nothing dropped
(``_revalidate_checkpoint_accepted_raw_v3``). Storing rows in any other order
fails the whole generation with "accepted raw changed under PIT replay", which
is what happened once restatements arrived: the restated vintage was appended
with ``pd.concat`` and left at the end of the frame, while the replay sorted it
back into announcement order.

Same rows, different order, generation rejected. These tests pin the ordering
down as a fixed point.
"""
from __future__ import annotations

import pandas as pd

from quant_investor.market.fundamental_mart import (
    _canonical_accepted_order,
    _strict_pit_cutoff,
)
from quant_investor.market.fundamental_provider_contract import (
    assert_frame_semantics_equal,
)

AS_OF = "20260904"
DIRTY_COUNTERS = (
    "rows_hard_invalid",
    "rows_filtered_future",
    "rows_filtered_missing_availability",
    "rows_filtered_core_values",
    "rows_deduplicated",
    "rows_discarded_request_malformed",
)


def _balancesheet(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Rows are (ann_date, end_date, update_flag, total_assets)."""
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": ann_date,
                "f_ann_date": ann_date,
                "end_date": end_date,
                "update_flag": update_flag,
                "total_assets": assets,
                "total_liab": assets * 0.6,
            }
            for ann_date, end_date, update_flag, assets in rows
        ]
    )


def _assert_is_replay_fixed_point(frame: pd.DataFrame, table: str) -> None:
    accepted, stats, reason = _strict_pit_cutoff(
        frame, table=table, symbol="000001.SZ", as_of=AS_OF
    )
    assert reason == ""
    assert [counter for counter in DIRTY_COUNTERS if int(stats[counter])] == []
    assert int(stats["rows"]) == len(frame)
    assert_frame_semantics_equal(frame, accepted, label=f"accepted raw {table}")


def test_appended_restatement_is_sorted_back_into_announcement_order() -> None:
    primary = _balancesheet(
        [
            ("20200214", "20191231", "0", 100.0),
            ("20200421", "20200331", "0", 110.0),
            ("20210202", "20201231", "0", 130.0),
        ]
    )
    # A restatement of 2018Q4 announced 2020-02-14 and of 2019Q4 announced
    # 2021-02-02: both belong before the end of the frame, not after it.
    restated = _balancesheet(
        [
            ("20200214", "20181231", "0", 90.0),
            ("20210202", "20191231", "0", 105.0),
        ]
    )
    appended = pd.concat([primary, restated], ignore_index=True)

    ordered, duplicates = _canonical_accepted_order(appended)

    assert duplicates == 0
    assert list(ordered["end_date"]) == [
        "20181231",
        "20191231",
        "20200331",
        "20191231",
        "20201231",
    ]
    assert list(ordered["ann_date"]) == [
        "20200214",
        "20200214",
        "20200421",
        "20210202",
        "20210202",
    ]
    _assert_is_replay_fixed_point(ordered, "balancesheet")


def test_appending_without_reordering_is_not_a_fixed_point() -> None:
    """The defect this guards against, stated directly."""
    primary = _balancesheet([("20200214", "20191231", "0", 100.0)])
    restated = _balancesheet([("20210202", "20191231", "0", 105.0)])
    # Restated first, so plain concatenation is out of announcement order.
    appended = pd.concat([restated, primary], ignore_index=True)

    accepted, _stats, _reason = _strict_pit_cutoff(
        appended, table="balancesheet", symbol="000001.SZ", as_of=AS_OF
    )

    assert list(accepted["ann_date"]) == ["20200214", "20210202"]
    assert list(appended["ann_date"]) == ["20210202", "20200214"]


def test_cutoff_output_is_already_canonical() -> None:
    unsorted = _balancesheet(
        [
            ("20210202", "20201231", "0", 130.0),
            ("20200214", "20191231", "0", 100.0),
            ("20200421", "20200331", "0", 110.0),
        ]
    )

    accepted, _stats, _reason = _strict_pit_cutoff(
        unsorted, table="balancesheet", symbol="000001.SZ", as_of=AS_OF
    )

    _assert_is_replay_fixed_point(accepted, "balancesheet")


def test_exact_duplicates_are_dropped_and_counted() -> None:
    row = ("20200214", "20191231", "0", 100.0)
    frame = _balancesheet([row, row, ("20200421", "20200331", "0", 110.0)])

    ordered, duplicates = _canonical_accepted_order(frame)

    assert duplicates == 1
    assert len(ordered) == 2
    _assert_is_replay_fixed_point(ordered, "balancesheet")


def test_empty_frame_is_canonical() -> None:
    ordered, duplicates = _canonical_accepted_order(pd.DataFrame())

    assert duplicates == 0
    assert ordered.empty
