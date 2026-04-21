from __future__ import annotations

import pandas as pd

from quant_investor.branch_contracts import (
    CorporateDocumentSnapshot,
    ForecastSnapshot,
    FundamentalSnapshot,
    ManagementSnapshot,
    OwnershipSnapshot,
    UnifiedDataBundle,
)
from quant_investor.forecast_snapshot_store import ForecastSnapshotStore
from quant_investor.fundamental_branch import FundamentalBranch


def test_fundamental_branch_uses_local_forecast_snapshot_without_online_fetch(monkeypatch, tmp_path):
    store = ForecastSnapshotStore(tmp_path)
    snapshot = ForecastSnapshot(
        symbol="000001.SZ",
        available=True,
        as_of="2026-03-26",
        source="offline_forecast_cache",
        provider="offline_forecast_cache",
        publish_time="2026-03-26T00:00:00",
        effective_time="2026-03-26T00:00:00",
        ingest_time="2026-03-26T08:00:00",
        revision_id="forecast:offline_forecast_cache:2026-03-26",
        is_estimated=False,
        eps_growth=0.12,
        revenue_growth_forecast=0.08,
        forecast_revision=0.04,
        coverage_count=5,
        data_quality={"reason": "provider_payload"},
        provenance={"reason": "provider_payload"},
    )
    store.save_snapshot(snapshot)
    monkeypatch.setattr(store, "load_snapshot", lambda _symbol: snapshot)
    monkeypatch.setattr(store, "get_snapshot", lambda _symbol, _as_of: snapshot)

    class _FakeDataLayer:
        def __init__(self, forecast_store: ForecastSnapshotStore) -> None:
            self._forecast_store = forecast_store

        def _online_forecast_fetch(self, *_args, **_kwargs):
            raise AssertionError("online forecast fetch should not happen")

        def get_earnings_forecast_snapshot(self, symbol: str, as_of: str) -> ForecastSnapshot:
            cached = self._forecast_store.load_snapshot(symbol)
            if cached is not None and str(cached.as_of) <= str(as_of):
                return self._forecast_store.get_snapshot(symbol, as_of)
            return self._online_forecast_fetch(symbol, as_of)

        def get_point_in_time_fundamental_snapshot(self, symbol: str, as_of: str) -> FundamentalSnapshot:
            return FundamentalSnapshot(
            symbol=symbol,
            available=True,
            as_of=str(as_of),
            effective_time=f"{as_of}T00:00:00",
            roe=0.18,
            gross_margin=0.32,
            profit_growth=0.15,
            revenue_growth=0.14,
            debt_ratio=0.30,
            current_ratio=1.5,
            pe=12.0,
            pb=1.8,
            ps=2.0,
            dividend_yield=0.03,
        )

        def get_management_snapshot(self, symbol: str, as_of: str) -> ManagementSnapshot:
            return ManagementSnapshot(
            symbol=symbol,
            available=True,
            as_of=str(as_of),
            effective_time=f"{as_of}T00:00:00",
            management_stability=0.8,
            governance_score=0.7,
            management_alignment=0.6,
            key_executive_changes=False,
        )

        def get_ownership_snapshot(self, symbol: str, as_of: str) -> OwnershipSnapshot:
            return OwnershipSnapshot(
            symbol=symbol,
            available=True,
            as_of=str(as_of),
            effective_time=f"{as_of}T00:00:00",
            concentration_score=0.4,
            ownership_change_signal=0.2,
            institutional_holding_pct=0.25,
            top_holder_pct=0.12,
        )

        def get_document_semantic_snapshot(self, symbol: str, as_of: str) -> CorporateDocumentSnapshot:
            return CorporateDocumentSnapshot(
            symbol=symbol,
            available=False,
            as_of=str(as_of),
        )

    data_layer = _FakeDataLayer(store)
    monkeypatch.setattr(
        data_layer,
        "_online_forecast_fetch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("online forecast fetch should not happen")
        ),
    )
    branch = FundamentalBranch(
        data_layer=data_layer,
        stock_pool=["000001.SZ"],
        enable_document_semantics=False,
    )

    data_bundle = UnifiedDataBundle(
        market="CN",
        symbols=["000001.SZ"],
        symbol_data={
            "000001.SZ": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-03-26"]),
                    "close": [10.0],
                }
            )
        },
        metadata={"end_date": "2026-03-26"},
    )
    data_bundle.stock_pool = ["000001.SZ"]
    data_bundle.stock_data = data_bundle.symbol_data

    result = branch.run(data_bundle)

    assert result.signals["quality_breakdown"]["000001.SZ"]["forecast_available"] is True
    assert result.signals["component_scores"]["000001.SZ"]["forecast_revision"] > 0.0
    assert result.module_coverage["forecast_revision"]["available_symbols"] == 1
