"""Daily native PIT identity can advance while old Market retains its exact binding."""

import hashlib
from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from quant_investor.market.market_data_reader import MarketDataReader


def test_native_daily_pit_dates_keep_old_market_generation_bound(tmp_path):
    fixture = NativeFactorInputs(tmp_path / "inputs", count=10)
    first_args = fixture.day(0)
    _, first = strict_market_from_factor_inputs(
        tmp_path, first_args, pit_observed_at="2026-08-24T00:00:00Z"
    )
    manifest = tmp_path / "data/parquet/cn/_snapshots/synthetic-native-factor-20260824.json"
    frozen = {
        "path": str(manifest.relative_to(tmp_path / "data")),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    _, second = strict_market_from_factor_inputs(
        tmp_path, fixture.day(1), pit_observed_at="2026-08-25T00:00:00Z"
    )
    assert first["generation_id"].startswith("pit-20260824-")
    assert second["generation_id"].startswith("pit-20260825-")
    old = MarketDataReader(data_root=tmp_path / "data", frozen_snapshot_ref=frozen)
    assert old.snapshot()["healthy"] is True
    assert old.coverage_bound_pit()["generation_id"] == first["generation_id"]
