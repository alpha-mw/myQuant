"""Bound the derived serving layer's storage growth.

`table/` hardlinks across snapshots because a daily upsert rewrites only the
affected `year=/month=` partitions. `serving/` is keyed by symbol, so appending
one session rewrites every symbol file — an unshared full copy per snapshot
(~400MB/day over 181 snapshots, the bulk of the 73GB parquet tree).
"""

from __future__ import annotations

import json
import os

import pytest

from quant_investor.market.market_data_store import MarketDataStore

_PUBLISHED = iter(range(1_700_000_000, 1_800_000_000, 60))


def _make_snapshot(root, snapshot_id: str, *, serving: bool = True) -> None:
    base = root / "parquet" / "cn" / "_snapshots" / snapshot_id
    table = base / "table" / "bars" / "year=2026" / "month=08"
    table.mkdir(parents=True, exist_ok=True)
    (table / "part.parquet").write_bytes(b"table-bytes")
    manifest = base.parent / f"{snapshot_id}.json"
    manifest.write_text(json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8")
    published = next(_PUBLISHED)
    os.utime(manifest, (published, published))
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
            tmp_path
            / "parquet"
            / "cn"
            / "_snapshots"
            / f"2026080{index}T000000Z"
            / "table"
            / "bars"
            / "year=2026"
            / "month=08"
            / "part.parquet"
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


def test_recency_follows_publication_not_snapshot_name(tmp_path):
    # Named backfill snapshots sort after timestamp ids lexically; publishing
    # one first must not let it displace the genuinely newest snapshots.
    _make_snapshot(tmp_path, "guarded-backfill-2014-20260812T175338Z")
    for index in range(1, 4):
        _make_snapshot(tmp_path, f"2026090{index}T000000Z")
    _set_active(tmp_path, "20260903T000000Z")
    store = MarketDataStore(market="CN", data_root=tmp_path)

    result = store.prune_snapshot_serving_layers(keep_recent=2)

    assert [item["snapshot_id"] for item in result["candidates"]] == [
        "20260901T000000Z",
        "guarded-backfill-2014-20260812T175338Z",
    ]


def test_post_publish_prune_keeps_newest_projection_and_never_raises(store, tmp_path, monkeypatch):
    store._prune_serving_after_publish()

    kept = [
        f"2026080{i}T000000Z"
        for i in range(1, 6)
        if _serving_exists(tmp_path, f"2026080{i}T000000Z")
    ]
    assert kept == ["20260803T000000Z", "20260804T000000Z", "20260805T000000Z"]

    def _boom(**_kwargs):
        raise OSError("disk unavailable")

    events = []
    monkeypatch.setattr(store, "prune_snapshot_serving_layers", _boom)
    monkeypatch.setattr(store, "append_health_event", lambda kind, payload: events.append(kind))
    store._prune_serving_after_publish()
    assert events == ["snapshot_serving_layer_prune_failed"]
