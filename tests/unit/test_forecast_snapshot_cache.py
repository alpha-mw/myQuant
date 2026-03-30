from __future__ import annotations

import pytest

from quant_investor.branch_contracts import ForecastSnapshot
from quant_investor.enhanced_data_layer import EnhancedDataLayer
from quant_investor.forecast_snapshot_store import ForecastSnapshotStore


def _make_snapshot(symbol: str, as_of: str, *, available: bool = True) -> ForecastSnapshot:
    return ForecastSnapshot(
        symbol=symbol,
        available=available,
        as_of=as_of,
        source="offline_forecast_cache",
        provider="offline_forecast_cache",
        publish_time=f"{as_of}T00:00:00",
        effective_time=f"{as_of}T00:00:00",
        ingest_time=f"{as_of}T08:00:00",
        revision_id=f"forecast:offline_forecast_cache:{as_of}",
        is_estimated=not available,
        eps_growth=0.12 if available else 0.0,
        revenue_growth_forecast=0.08 if available else 0.0,
        forecast_revision=0.04 if available else 0.0,
        coverage_count=4 if available else 0,
        notes=[],
        data_quality={"reason": "" if available else "forecast_provider_empty"},
        provenance={"reason": "" if available else "forecast_provider_empty"},
    )


def test_forecast_snapshot_cache_hit_avoids_online_fetch(monkeypatch, tmp_path):
    layer = EnhancedDataLayer(market="CN", verbose=False)
    store = ForecastSnapshotStore(tmp_path)
    store.save_snapshot(_make_snapshot("000001.SZ", "2026-03-26"))
    layer._source._tushare._forecast_store = store

    monkeypatch.setattr(
        layer._source._tushare,
        "get_earnings_forecast_snapshot",
        lambda *args, **kwargs: pytest.fail("should not fetch forecast online during analysis"),
    )

    snapshot = layer.get_earnings_forecast_snapshot("000001.SZ", "2026-03-26")

    assert snapshot.available is True
    assert snapshot.source == "offline_forecast_cache"
    assert snapshot.forecast_revision == 0.04


def test_forecast_snapshot_cache_miss_returns_neutral_snapshot_without_online_fetch(monkeypatch, tmp_path):
    layer = EnhancedDataLayer(market="CN", verbose=False)
    layer._source._tushare._forecast_store = ForecastSnapshotStore(tmp_path)

    monkeypatch.setattr(
        layer._source._tushare,
        "get_earnings_forecast_snapshot",
        lambda *args, **kwargs: pytest.fail("should not fetch forecast online during analysis"),
    )

    snapshot = layer.get_earnings_forecast_snapshot("000001.SZ", "2026-03-26")

    assert snapshot.available is False
    assert snapshot.data_quality["reason"] == "forecast_cache_missing_or_stale"
    assert "forecast_cache_missing_or_stale" in snapshot.notes


def test_forecast_snapshot_cache_stale_returns_neutral_snapshot_without_online_fetch(monkeypatch, tmp_path):
    layer = EnhancedDataLayer(market="CN", verbose=False)
    store = ForecastSnapshotStore(tmp_path)
    store.save_snapshot(_make_snapshot("000001.SZ", "2026-03-20"))
    layer._source._tushare._forecast_store = store

    monkeypatch.setattr(
        layer._source._tushare,
        "get_earnings_forecast_snapshot",
        lambda *args, **kwargs: pytest.fail("should not fetch forecast online during analysis"),
    )

    snapshot = layer.get_earnings_forecast_snapshot("000001.SZ", "2026-03-26")

    assert snapshot.available is False
    assert snapshot.data_quality["reason"] == "forecast_cache_missing_or_stale"
    assert snapshot.metadata["cached_as_of"] == "2026-03-20"
