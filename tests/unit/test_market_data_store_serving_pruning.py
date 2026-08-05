"""Bound the derived serving layer's storage growth.

`table/` hardlinks across snapshots because a daily upsert rewrites only the
affected `year=/month=` partitions. `serving/` is keyed by symbol, so appending
one session rewrites every symbol file — an unshared full copy per snapshot
(~400MB/day over 181 snapshots, the bulk of the 73GB parquet tree).
"""

from __future__ import annotations

import json

import pytest

from quant_investor.market.market_data_store import MarketDataStore


def _make_snapshot(root, snapshot_id: str, *, serving: bool = True) -> None:
    base = root / "parquet" / "cn" / "_snapshots" / snapshot_id
    table = base / "table" / "bars" / "year=2026" / "month=08"
    table.mkdir(parents=True, exist_ok=True)
    (table / "part.parquet").write_bytes(b"table-bytes")
    if serving:
        for symbol in ("601989.SH", "603056.SH"):
            symbol_dir = base / "serving" / "bars" / f"symbol={symbol}"
            symbol_dir.mkdir(parents=True, exist_ok=True)
            (symbol_dir / "bars.parquet").write_bytes(b"serving-bytes-" + symbol.encode())


def _set_active(root, snapshot_id: str) -> None:
    pointer = root / "parquet" / "cn" / "_latest.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8")


def _serving_exists(root, snapshot_id: str) -> bool:
    return (root / "parquet" / "cn" / "_snapshots" / snapshot_id / "serving").exists()


@pytest.fixture
def store(tmp_path):
    for index in range(1, 6):
        _make_snapshot(tmp_path, f"2026080{index}T000000Z")
    _set_active(tmp_path, "20260805T000000Z")
    return MarketDataStore(market="CN", data_root=tmp_path)


def test_dry_run_is_the_default_and_deletes_nothing(store, tmp_path):
    result = store.prune_snapshot_serving_layers(keep_recent=2)

    assert result["dry_run"] is True
    assert result["pruned"] == []
    assert [item["snapshot_id"] for item in result["candidates"]] == [
        "20260803T000000Z",
        "20260802T000000Z",
        "20260801T000000Z",
    ]
    for index in range(1, 6):
        assert _serving_exists(tmp_path, f"2026080{index}T000000Z")


def test_pruning_keeps_the_most_recent_snapshots(store, tmp_path):
    store.prune_snapshot_serving_layers(keep_recent=2, dry_run=False)

    assert _serving_exists(tmp_path, "20260805T000000Z")
    assert _serving_exists(tmp_path, "20260804T000000Z")
    assert not _serving_exists(tmp_path, "20260803T000000Z")
    assert not _serving_exists(tmp_path, "20260801T000000Z")


def test_table_layer_is_never_touched(store, tmp_path):
    store.prune_snapshot_serving_layers(keep_recent=1, dry_run=False)

    for index in range(1, 6):
        table = (
            tmp_path / "parquet" / "cn" / "_snapshots" / f"2026080{index}T000000Z"
            / "table" / "bars" / "year=2026" / "month=08" / "part.parquet"
        )
        assert table.read_bytes() == b"table-bytes"


def test_active_snapshot_is_protected_even_when_old(tmp_path):
    for index in range(1, 6):
        _make_snapshot(tmp_path, f"2026080{index}T000000Z")
    _set_active(tmp_path, "20260801T000000Z")  # oldest is pinned active
    store = MarketDataStore(market="CN", data_root=tmp_path)

    store.prune_snapshot_serving_layers(keep_recent=1, dry_run=False)

    assert _serving_exists(tmp_path, "20260801T000000Z")
    assert not _serving_exists(tmp_path, "20260803T000000Z")


def test_pruning_records_an_auditable_health_event(store, tmp_path):
    store.prune_snapshot_serving_layers(keep_recent=2, dry_run=False)

    ledger = tmp_path / "parquet" / "cn" / "_health_ledger.jsonl"
    events = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "snapshot_serving_layer_pruned"
    ]
    assert len(events) == 1
    assert sorted(events[0]["payload"]["pruned"]) == [
        "20260801T000000Z",
        "20260802T000000Z",
        "20260803T000000Z",
    ]


def test_keep_recent_must_be_positive(store):
    with pytest.raises(ValueError, match="keep_recent must be at least 1"):
        store.prune_snapshot_serving_layers(keep_recent=0)


def test_already_pruned_snapshots_are_not_recounted(store, tmp_path):
    store.prune_snapshot_serving_layers(keep_recent=2, dry_run=False)

    again = store.prune_snapshot_serving_layers(keep_recent=2, dry_run=False)

    assert again["candidates"] == []
    assert again["pruned"] == []
