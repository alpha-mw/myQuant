from __future__ import annotations

import pytest
import pandas as pd

from quant_investor.agent_protocol import AgentStatus, BranchVerdict
from quant_investor.agents.fundamental_agent import FundamentalAgent, _BundleFundamentalDataLayer
from quant_investor.branch_contracts import FundamentalSnapshot, UnifiedDataBundle
from quant_investor.fundamental_components import financial_quality_analyzer
from quant_investor.fundamental_branch import FundamentalBranch
from quant_investor.market.dag.assembly import _aggregate_branch_summaries, _build_branch_results
from quant_investor.market.dag.decision import _branch_degraded_map


def test_bundle_adapter_preserves_missing_fields_in_availability_mask():
    layer = _BundleFundamentalDataLayer(
        {
            "000001.SZ": {
                "trade_date": "2026-03-26",
                "availability_date": "2026-03-20",
                "fin_roe": 0.18,
                "fin_net_profit_yoy": -0.10,
            }
        }
    )

    snapshot = layer.get_point_in_time_fundamental_snapshot("000001.SZ", "2026-03-26")

    assert snapshot.data_quality["available_fields"] == ["roe", "profit_growth"]
    assert "gross_margin" in snapshot.data_quality["missing_fields"]
    assert financial_quality_analyzer(snapshot).score == pytest.approx(0.10)


def test_explicit_zero_is_scored_but_missing_zero_is_not():
    missing = FundamentalSnapshot(
        symbol="000001.SZ",
        available=True,
        roe=0.18,
        data_quality={"available_fields": ["roe"]},
    )
    explicit_zero = FundamentalSnapshot(
        symbol="000001.SZ",
        available=True,
        roe=0.18,
        gross_margin=0.0,
        data_quality={"available_fields": ["roe", "gross_margin"]},
    )

    assert financial_quality_analyzer(missing).score == 0.30
    assert financial_quality_analyzer(explicit_zero).score == pytest.approx(0.20)


def test_bundle_forecast_does_not_invent_analyst_coverage():
    layer = _BundleFundamentalDataLayer(
        {"000001.SZ": {"trade_date": "2026-03-26", "forecast_revision": 0.04}}
    )

    snapshot = layer.get_earnings_forecast_snapshot("000001.SZ", "2026-03-26")

    assert snapshot.coverage_count == 0
    assert snapshot.data_quality["forecast_kind"] == "corporate_guidance"
    assert snapshot.data_quality["available_fields"] == ["forecast_revision"]


def test_partial_bundle_fundamental_verdict_is_degraded():
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=["000001.SZ"],
        symbol_data={
            "000001.SZ": pd.DataFrame({"date": pd.to_datetime(["2026-03-26"]), "close": [10.0]})
        },
        fundamentals={
            "000001.SZ": {
                "trade_date": "2026-03-26",
                "availability_date": "2026-03-20",
                "fin_roe": 0.18,
                "forecast_revision": 0.04,
            }
        },
    )

    verdict = FundamentalAgent().run({"data_bundle": bundle, "stock_pool": ["000001.SZ"]})

    assert verdict.status == AgentStatus.DEGRADED
    assert verdict.metadata["degraded_reason"] == "fundamental_evidence_incomplete"
    assert verdict.metadata["horizon_days"] == 30
    assert verdict.metadata["structured_signals"]["quality_breakdown"]
    assert verdict.metadata["fundamental_data_generation_by_symbol"]["000001.SZ"].startswith(
        "fundamental-"
    )


def test_fundamental_generation_is_event_driven_not_daily_price_driven():
    def generation(price_date: str, *, events: list[dict[str, str]] | None = None) -> str:
        bundle = UnifiedDataBundle(
            market="CN",
            symbols=["000001.SZ"],
            symbol_data={
                "000001.SZ": pd.DataFrame({"date": pd.to_datetime([price_date]), "close": [10.0]})
            },
            fundamentals={
                "000001.SZ": {
                    "trade_date": "2026-03-26",
                    "availability_date": "2026-03-20",
                    "source_version": "fundamental-mart-7",
                    "fin_roe": 0.18,
                }
            },
            event_data={"000001.SZ": list(events or [])},
        )
        verdict = FundamentalAgent().run({"data_bundle": bundle, "stock_pool": ["000001.SZ"]})
        return verdict.metadata["fundamental_data_generation_by_symbol"]["000001.SZ"]

    assert generation("2026-03-26") == generation("2026-03-27")
    assert generation("2026-03-27", events=[{"event_id": "filing-2"}]) != generation("2026-03-27")


def test_assembly_preserves_degraded_status_and_fundamental_metadata():
    verdict = BranchVerdict(
        agent_name="FundamentalAgent",
        thesis="partial evidence",
        status=AgentStatus.DEGRADED,
        final_score=0.1,
        final_confidence=0.4,
        metadata={
            "branch_name": "fundamental",
            "reliability": 0.45,
            "structured_signals": {"x": 1},
        },
    )
    research = {"000001.SZ": {"fundamental": verdict}}

    summaries = _aggregate_branch_summaries(research)
    results = _build_branch_results(research, summaries)

    assert summaries["fundamental"].status == AgentStatus.DEGRADED
    assert summaries["fundamental"].metadata["degraded_symbols"] == ["000001.SZ"]
    assert results["fundamental"].metadata["degraded_reason"] == "symbol_research_degraded"
    assert results["fundamental"].metadata["reliability"] == 0.45
    assert results["fundamental"].signals["structured_signals_by_symbol"]["000001.SZ"] == {"x": 1}


def test_bayesian_degraded_map_uses_symbol_verdict_status():
    research = {
        "000001.SZ": {
            "fundamental": BranchVerdict(
                agent_name="FundamentalAgent",
                status=AgentStatus.DEGRADED,
            )
        }
    }

    degraded = _branch_degraded_map(
        symbol="000001.SZ",
        research_by_symbol=research,
        branch_summaries={},
        branch_results={},
    )

    assert degraded["fundamental"] is True
    assert degraded["quant"] is False


def test_symbol_conclusion_includes_primary_negative_driver():
    branch = object.__new__(FundamentalBranch)

    conclusion = branch._build_symbol_conclusion(
        symbol="000001.SZ",
        available_modules=["financial_quality"],
        missing_modules=["valuation"],
        support_points=["财务质量: ROE 处于较优区间。"],
        drag_points=["财务质量: 盈利增速转弱。"],
    )

    assert "主要风险为财务质量: 盈利增速转弱" in conclusion
