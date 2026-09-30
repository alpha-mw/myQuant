"""Exact historical strict snapshots cannot fall forward to current aliases."""

import hashlib
import json
from pathlib import Path

import pytest
from _native_daily_store_fixture import NativeStoreFixture, DAYS, STOCKS
from quant_investor.market.market_data_reader import MarketDataReader, MarketDataUnavailableError


def test_frozen_snapshot_survives_current_pointer_change(tmp_path):
    fixture = NativeStoreFixture(tmp_path)
    fixture.advance(DAYS[0])
    data = tmp_path / "data"
    latest = data / "parquet/cn/_latest.json"
    payload = json.loads(latest.read_bytes())
    manifest = data / "parquet/cn/_snapshots/synthetic-20260824.json"
    ref = {
        "path": str(manifest.relative_to(data)),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    latest.write_text('{"status":"BLOCKED","snapshot_id":"later"}')
    reader = MarketDataReader(data_root=data, frozen_snapshot_ref=ref)
    assert reader.snapshot()["latest_complete_trade_date"] == payload["latest_complete_trade_date"]
    assert reader.snapshot()["healthy"]
    assert reader.resolve_symbol_path(STOCKS[0]) == Path(reader.snapshot()["table_root"])
    for action in (
        lambda: reader.resolve_symbol_path(STOCKS[0], for_write=True),
        reader._load_catalog,
        reader._load_components,
    ):
        with pytest.raises(MarketDataUnavailableError, match="frozen snapshot"):
            action()
    bad = MarketDataReader(data_root=data, frozen_snapshot_ref={**ref, "sha256": "0" * 64})
    assert not bad.snapshot()["healthy"]
    assert "SHA mismatch" in str(bad.snapshot()["blockers"])


@pytest.mark.parametrize(
    "path",
    [
        "parquet/cn/_latest.json",
        "../escape.json",
        "/tmp/snapshot.json",
        "parquet/cn/_snapshots/../x.json",
    ],
)
def test_frozen_selector_rejects_alias_or_escape(tmp_path, path):
    with pytest.raises(MarketDataUnavailableError):
        MarketDataReader(data_root=tmp_path, frozen_snapshot_ref={"path": path, "sha256": "a" * 64})
