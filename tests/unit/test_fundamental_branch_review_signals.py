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
from quant_investor.fundamental_branch import FundamentalBranch


class _FakeDataLayer:
    def get_point_in_time_fundamental_snapshot(self, symbol: str, as_of: str) -> FundamentalSnapshot:
        return FundamentalSnapshot(
            symbol=symbol,
            available=True,
            as_of=as_of,
            effective_time="2026-03-22T00:00:00",
            roe=0.18,
            roa=0.11,
            gross_margin=0.35,
            net_margin=0.16,
            revenue_growth=0.14,
            profit_growth=0.12,
            debt_ratio=0.28,
            current_ratio=1.6,
            pe=12.5,
            pb=1.8,
            ps=2.2,
            dividend_yield=0.03,
        )

    def get_earnings_forecast_snapshot(self, symbol: str, as_of: str) -> ForecastSnapshot:
        return ForecastSnapshot(
            symbol=symbol,
            available=True,
            as_of=as_of,
            effective_time="2026-03-24T00:00:00",
            provider="offline_forecast_cache",
            source="offline_forecast_cache",
            eps_growth=0.10,
            revenue_growth_forecast=0.08,
            forecast_revision=0.05,
            coverage_count=6,
        )

    def get_management_snapshot(self, symbol: str, as_of: str) -> ManagementSnapshot:
        return ManagementSnapshot(
            symbol=symbol,
            available=True,
            as_of=as_of,
            effective_time="2026-03-21T00:00:00",
            management_stability=0.8,
            governance_score=0.75,
            management_alignment=0.65,
        )

    def get_ownership_snapshot(self, symbol: str, as_of: str) -> OwnershipSnapshot:
        return OwnershipSnapshot(
            symbol=symbol,
            available=True,
            as_of=as_of,
            effective_time="2026-03-20T00:00:00",
            concentration_score=0.42,
            ownership_change_signal=0.12,
            institutional_holding_pct=0.28,
            top_holder_pct=0.16,
        )

    def get_document_semantic_snapshot(self, symbol: str, as_of: str) -> CorporateDocumentSnapshot:
        return CorporateDocumentSnapshot(
            symbol=symbol,
            available=True,
            as_of=as_of,
            effective_time="2026-03-25T00:00:00",
            semantic_sentiment=0.30,
            execution_confidence=0.70,
            governance_red_flag=0.10,
        )


def test_fundamental_branch_emits_review_signal_payloads():
    branch = FundamentalBranch(
        data_layer=_FakeDataLayer(),
        stock_pool=["000001.SZ"],
        enable_document_semantics=True,
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

    result = branch.run(data_bundle)

    assert "component_scores" in result.signals
    assert result.signals["module_scores"]["financial_quality"] != 0.0
    assert result.signals["module_confidences"]["financial_quality"] == 1.0
    assert result.signals["module_coverages"]["forecast_revision"] == "available"
    assert result.signals["financial_quality"]["000001.SZ"]["roe"] == 0.18
    assert result.signals["forecast_revisions"]["000001.SZ"]["forecast_revision"] == 0.05
    assert result.signals["valuation_metrics"]["000001.SZ"]["pe"] == 12.5
    assert result.signals["governance_scores"]["000001.SZ"] == 0.75
    assert result.signals["ownership_signals"]["000001.SZ"]["concentration_score"] == 0.42
    assert result.signals["doc_sentiment"]["000001.SZ"] == 0.30
    assert result.signals["data_staleness_days"]["000001.SZ"] == 6
