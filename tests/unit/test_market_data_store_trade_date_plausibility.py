"""Guard the canonical bars table against impossible session dates.

Two rows carrying `trade_date=19700101` (epoch-zero timestamps for 601989.SH
and 603056.SH) reached the strict CN Parquet table and were then hardlinked
forward into every later snapshot, in both the `table/` and `serving/` layers.
Nothing in the store rejected them: `_normalize_trade_date` is purely lexical,
so it accepts any 8 leading digits.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_investor.market.market_data_store import MarketDataStore


def _bars(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "trade_date": trade_date,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "vol": 1.0,
                "amount": 1.0,
                "adj_factor": 1.0,
            }
            for symbol, trade_date in rows
        ]
    )


def _health_events(tmp_path) -> list[dict]:
    path = tmp_path / "parquet" / "cn" / "_health_ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_epoch_zero_trade_date_is_rejected_on_incoming_bars(tmp_path):
    store = MarketDataStore(market="CN", data_root=tmp_path)
    frame = _bars(("601989.SH", "20260803"), ("603056.SH", "19700101"))

    with pytest.raises(ValueError, match="implausible trade_date"):
        store._normalize_bars_frame(frame)


def test_rejection_names_the_offending_rows(tmp_path):
    store = MarketDataStore(market="CN", data_root=tmp_path)
    frame = _bars(("603056.SH", "19700101"))

    with pytest.raises(ValueError) as excinfo:
        store._normalize_bars_frame(frame)

    message = str(excinfo.value)
    assert "603056.SH" in message
    assert "19700101" in message
    assert "19901219" in message  # the earliest valid CN session is reported


@pytest.mark.parametrize(
    "trade_date",
    [
        "19700101",  # epoch zero
        "19891231",  # before the first SSE session
        "20261332",  # not a real calendar date
        "20260230",  # not a real calendar date
    ],
)
def test_implausible_dates_are_detected(tmp_path, trade_date):
    store = MarketDataStore(market="CN", data_root=tmp_path)
    offenders = store.implausible_trade_date_rows(_bars(("601989.SH", trade_date)))
    assert offenders == [("601989.SH", trade_date)]


@pytest.mark.parametrize("trade_date", ["19901219", "20190102", "20260803"])
def test_real_sessions_are_accepted(tmp_path, trade_date):
    store = MarketDataStore(market="CN", data_root=tmp_path)
    assert store.implausible_trade_date_rows(_bars(("601989.SH", trade_date))) == []
    kept = store._normalize_bars_frame(_bars(("601989.SH", trade_date)))
    assert list(kept["trade_date"]) == [trade_date]


def test_existing_bytes_are_healed_not_wedged(tmp_path):
    """Republishing must not be blocked forever by historical corruption."""

    store = MarketDataStore(market="CN", data_root=tmp_path)
    frame = _bars(("601989.SH", "19700101"), ("601989.SH", "20260803"))

    kept = store._normalize_bars_frame(frame, strict=False)

    assert list(kept["trade_date"]) == ["20260803"]


def test_healing_records_an_auditable_health_event(tmp_path):
    store = MarketDataStore(market="CN", data_root=tmp_path)
    frame = _bars(("601989.SH", "19700101"), ("603056.SH", "20260803"))

    store._normalize_bars_frame(frame, strict=False)

    events = [
        event
        for event in _health_events(tmp_path)
        if event["event_type"] == "implausible_trade_date_quarantined"
    ]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["dropped_row_count"] == 1
    assert payload["dropped_rows"] == [["601989.SH", "19700101"]]
    assert payload["first_plausible_session"] == "19901219"


def test_merge_drops_corruption_carried_by_existing_snapshot(tmp_path):
    """This is the path that propagated 19700101 into every later snapshot."""

    store = MarketDataStore(market="CN", data_root=tmp_path)
    existing = _bars(("601989.SH", "19700101"), ("601989.SH", "20260731"))
    incoming = store._normalize_bars_frame(_bars(("601989.SH", "20260803")))

    merged = store._merge_bars(existing, incoming)

    assert sorted(merged["trade_date"]) == ["20260731", "20260803"]


def test_market_floor_is_per_market(tmp_path):
    cn = MarketDataStore(market="CN", data_root=tmp_path)
    us = MarketDataStore(market="US", data_root=tmp_path)

    assert cn.first_plausible_session() == "19901219"
    # A 1985 US session is real history; the CN floor must not reject it.
    assert us.implausible_trade_date_rows(_bars(("AAPL", "19850102"))) == []
    assert cn.implausible_trade_date_rows(_bars(("601989.SH", "19850102"))) != []
