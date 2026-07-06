from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from quant_investor.config import Config, MAINLINE_ENV_DEFAULTS
from quant_investor.market.pit_universe import (
    LIST_STATUS_DELISTED,
    LIST_STATUS_LISTED,
    LIST_STATUS_PENDING,
    PITUniverseRecord,
    PITUniverseStore,
    build_pit_delisted_field,
    build_pit_universe_mask,
    dedupe_latest_records,
    estimate_historical_bar_backfill_cost,
    evaluate_listing_status,
    filter_symbols_by_pit_status,
    is_listed,
    refresh_pit_universe_from_tushare,
)


def _record(
    symbol: str,
    *,
    status: str = LIST_STATUS_LISTED,
    list_date: str = "20200101",
    delist_date: str = "",
    name: str = "Fixture",
) -> PITUniverseRecord:
    return PITUniverseRecord(
        symbol=symbol,
        name=name,
        source_list_status=status,
        list_date=list_date,
        delist_date=delist_date,
        observed_at="2026-07-06T00:00:00Z",
        source_run_id="unit-test",
    )


def test_pit_universe_defaults_are_default_off() -> None:
    expected = {
        "PIT_UNIVERSE_ENABLED": "0",
        "PIT_UNIVERSE_REQUIRED": "0",
        "PIT_UNIVERSE_SOURCE_ROOT": "data/parquet/cn/reference",
        "PIT_UNIVERSE_BACKFILL_ENABLED": "0",
    }
    for key, value in expected.items():
        assert MAINLINE_ENV_DEFAULTS[key] == value

    if "PIT_UNIVERSE_ENABLED" not in os.environ:
        assert Config.PIT_UNIVERSE_ENABLED is False
    if "PIT_UNIVERSE_REQUIRED" not in os.environ:
        assert Config.PIT_UNIVERSE_REQUIRED is False
    if "PIT_UNIVERSE_BACKFILL_ENABLED" not in os.environ:
        assert Config.PIT_UNIVERSE_BACKFILL_ENABLED is False


def test_listing_status_handles_active_prelisting_delisted_pending_and_missing() -> None:
    active = _record("000001.SZ")
    prelisting = evaluate_listing_status(active, symbol="000001.SZ", as_of="20191231")
    assert prelisting.in_universe is False
    assert prelisting.reason == "pre_listing"

    listed = evaluate_listing_status(active, symbol="000001.SZ", as_of="20260101")
    assert listed.in_universe is True
    assert listed.research_eligible is True
    assert listed.tradable is True
    assert listed.reason == "listed"

    delisted_record = _record(
        "000002.SZ",
        status=LIST_STATUS_DELISTED,
        delist_date="20250102",
    )
    before_delist = evaluate_listing_status(delisted_record, symbol="000002.SZ", as_of="20250101")
    assert before_delist.in_universe is True
    assert before_delist.tradable is True
    after_delist = evaluate_listing_status(delisted_record, symbol="000002.SZ", as_of="20250102")
    assert after_delist.in_universe is False
    assert after_delist.reason == "delisted"

    pending = evaluate_listing_status(
        _record("000003.SZ", status=LIST_STATUS_PENDING),
        symbol="000003.SZ",
        as_of="20260101",
    )
    assert pending.in_universe is True
    assert pending.research_eligible is True
    assert pending.tradable is False
    assert pending.reason == "pending"

    missing = evaluate_listing_status(None, symbol="000004.SZ", as_of="20260101")
    assert missing.in_universe is False
    assert missing.reason == "missing_pit_record"

    bad_delist = evaluate_listing_status(
        _record("000005.SZ", status=LIST_STATUS_DELISTED),
        symbol="000005.SZ",
        as_of="20260101",
    )
    assert bad_delist.in_universe is False
    assert bad_delist.reason == "missing_delist_date"


def test_dedupe_latest_records_marks_conflicts_and_prefers_delisted_row() -> None:
    latest = dedupe_latest_records(
        [
            _record("000001.SZ", status=LIST_STATUS_LISTED),
            _record("000001.SZ", status=LIST_STATUS_DELISTED, delist_date="20250102"),
            _record("000002.SZ", list_date="20200101"),
            _record("000002.SZ", list_date="20210101"),
        ]
    )

    by_symbol = {record.symbol: record for record in latest}
    assert by_symbol["000001.SZ"].source_list_status == LIST_STATUS_DELISTED
    assert by_symbol["000001.SZ"].delist_date == "20250102"
    assert by_symbol["000002.SZ"].membership_quality == "conflicting_status_rows"


def test_filter_symbols_fail_open_for_missing_optional_pit_records() -> None:
    records = [_record("000001.SZ")]

    optional = filter_symbols_by_pit_status(
        ["000001.SZ", "000002.SZ"],
        as_of="20260101",
        records=records,
        required=False,
    )
    assert optional.symbols == ["000001.SZ", "000002.SZ"]
    assert optional.quarantine_symbols == []
    assert optional.metadata["reasons"]["000002.SZ"] == "missing_pit_record"

    required = filter_symbols_by_pit_status(
        ["000001.SZ", "000002.SZ"],
        as_of="20260101",
        records=records,
        required=True,
    )
    assert required.symbols == ["000001.SZ"]
    assert required.quarantine_symbols == ["000002.SZ"]
    assert required.metadata["missing_count"] == 1


def test_build_pit_backtest_masks_for_universe_and_delisted_fields() -> None:
    records = [
        _record("000001.SZ", list_date="20200102"),
        _record(
            "000002.SZ",
            status=LIST_STATUS_DELISTED,
            list_date="20200101",
            delist_date="20200103",
        ),
    ]
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]

    optional_mask = build_pit_universe_mask(symbols, dates, records, required=False)
    assert optional_mask == [
        [False, True, True],
        [True, True, False],
        [True, True, True],
    ]

    required_mask = build_pit_universe_mask(symbols, dates, records, required=True)
    assert required_mask[2] == [False, False, False]
    assert build_pit_delisted_field(symbols, dates, records) == [
        [False, False, False],
        [False, False, True],
        [False, False, False],
    ]


def test_pit_store_round_trips_manifest_and_latest_records(tmp_path: Path) -> None:
    store = PITUniverseStore(
        root_dir=tmp_path / "reference",
        raw_root=tmp_path / "raw",
        compatibility_path=tmp_path / "compat" / "stock_basic.json",
    )
    manifest = store.write_snapshot(
        raw_records=[
            _record("000001.SZ"),
            _record("000002.SZ", status=LIST_STATUS_DELISTED, delist_date="20250102"),
        ],
        observed_at="2026-07-06T00:00:00Z",
        source_run_id="unit-test",
    )

    assert store.canonical_path.exists()
    assert store.manifest_path.exists()
    assert store.compatibility_path.exists()
    assert manifest["row_count"] == 2
    assert manifest["status_counts"] == {"D": 1, "L": 1}
    loaded = store.load_latest_records()
    assert [record.symbol for record in loaded] == ["000001.SZ", "000002.SZ"]
    assert store.is_listed("000001.SZ", "20260101") is True
    assert is_listed("000002.SZ", "20260101", loaded) is False


class _FakeTusharePro:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def stock_basic(self, *, exchange: str, list_status: str, fields: str) -> pd.DataFrame:
        self.calls.append(list_status)
        rows = {
            LIST_STATUS_LISTED: [
                {
                    "ts_code": "000001.SZ",
                    "name": "Active",
                    "area": "SZ",
                    "industry": "Bank",
                    "market": "主板",
                    "list_date": "20200101",
                    "delist_date": "",
                    "list_status": "L",
                }
            ],
            LIST_STATUS_DELISTED: [
                {
                    "ts_code": "000002.SZ",
                    "name": "Delisted",
                    "area": "SZ",
                    "industry": "Tech",
                    "market": "主板",
                    "list_date": "20200101",
                    "delist_date": "20250102",
                    "list_status": "D",
                }
            ],
            LIST_STATUS_PENDING: [],
        }
        return pd.DataFrame(rows[list_status])


def test_refresh_pit_universe_from_fake_tushare_dry_run_and_execute(tmp_path: Path) -> None:
    pro = _FakeTusharePro()
    store = PITUniverseStore(
        root_dir=tmp_path / "reference",
        raw_root=tmp_path / "raw",
        compatibility_path=tmp_path / "compat" / "stock_basic.json",
    )

    dry_run = refresh_pit_universe_from_tushare(
        pro,
        store=store,
        execute=False,
        observed_at="2026-07-06T00:00:00Z",
        source_run_id="unit-test",
    )
    assert pro.calls == ["L", "D", "P"]
    assert dry_run["provider_call_count"] == 3
    assert dry_run["row_count"] == 2
    assert not store.canonical_path.exists()

    execute = refresh_pit_universe_from_tushare(
        pro,
        store=store,
        execute=True,
        observed_at="2026-07-06T00:00:00Z",
        source_run_id="unit-test",
    )
    assert execute["manifest"]["row_count"] == 2
    assert store.canonical_path.exists()


def test_backfill_cost_estimator_is_deterministic() -> None:
    cost = estimate_historical_bar_backfill_cost(
        missing_trade_dates=5,
        endpoints_per_date=3,
        unresolved_symbol_dates=7,
        calls_per_symbol_date=2,
    )

    assert cost["stock_basic_refresh_calls"] == 3
    assert cost["date_scoped_bar_calls"] == 15
    assert cost["symbol_tail_calls"] == 14
    assert cost["total_estimated_calls"] == 32
