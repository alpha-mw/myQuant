from __future__ import annotations

import pandas as pd

from quant_investor.branch_contracts import CorporateDocumentSnapshot, ForecastSnapshot, FundamentalSnapshot, ManagementSnapshot, OwnershipSnapshot, UnifiedDataBundle
from quant_investor.forecast_snapshot_store import ForecastSnapshotStore
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline


def test_fundamental_branch_uses_local_forecast_snapshot_without_online_fetch(monkeypatch, tmp_path):
    pipeline = ParallelResearchPipeline(
        stock_pool=["000001.SZ"],
        market="CN",
        enable_document_semantics=False,
        verbose=False,
    )

    store = ForecastSnapshotStore(tmp_path)
    store.save_snapshot(
        ForecastSnapshot(
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
    )
    pipeline.data_layer._source._tushare._forecast_store = store

    monkeypatch.setattr(
        pipeline.data_layer._source._tushare,
        "get_earnings_forecast_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("online forecast fetch should not happen")),
    )

    monkeypatch.setattr(
        pipeline.data_layer,
        "get_point_in_time_fundamental_snapshot",
        lambda symbol, as_of: FundamentalSnapshot(
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
        ),
    )
    monkeypatch.setattr(
        pipeline.data_layer,
        "get_management_snapshot",
        lambda symbol, as_of: ManagementSnapshot(
            symbol=symbol,
            available=True,
            as_of=str(as_of),
            effective_time=f"{as_of}T00:00:00",
            management_stability=0.8,
            governance_score=0.7,
            management_alignment=0.6,
            key_executive_changes=False,
        ),
    )
    monkeypatch.setattr(
        pipeline.data_layer,
        "get_ownership_snapshot",
        lambda symbol, as_of: OwnershipSnapshot(
            symbol=symbol,
            available=True,
            as_of=str(as_of),
            effective_time=f"{as_of}T00:00:00",
            concentration_score=0.4,
            ownership_change_signal=0.2,
            institutional_holding_pct=0.25,
            top_holder_pct=0.12,
        ),
    )
    monkeypatch.setattr(
        pipeline.data_layer,
        "get_document_semantic_snapshot",
        lambda symbol, as_of: CorporateDocumentSnapshot(
            symbol=symbol,
            available=False,
            as_of=str(as_of),
        ),
    )

    data_bundle = UnifiedDataBundle(
        stock_pool=["000001.SZ"],
        stock_data={
            "000001.SZ": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-03-26"]),
                    "close": [10.0],
                }
            )
        },
        metadata={"end_date": "2026-03-26"},
    )

    result = pipeline._run_fundamental_branch(data_bundle)

    assert result.signals["forecast_revisions"]["000001.SZ"]["forecast_revision"] == 0.04
    assert result.signals["module_scores"]["forecast_revision"] > 0.0
    assert result.signals["module_coverages"]["forecast_revision"] == "available"
