"""
Research agents 包装层单元测试。
"""

from __future__ import annotations

import pandas as pd

from quant_investor.agents.base import BaseAgent
from quant_investor.agents.fundamental_agent import FundamentalAgent
from quant_investor.agents.intelligence_agent import IntelligenceAgent
from quant_investor.agents.kline_agent import KlineAgent
from quant_investor.agents.macro_agent import MacroAgent
from quant_investor.agents.quant_agent import QuantAgent
from quant_investor.branch_contracts import (
    BranchResult,
    CorporateDocumentSnapshot,
    ForecastSnapshot,
    FundamentalSnapshot,
    ManagementSnapshot,
    OwnershipSnapshot,
    UnifiedDataBundle,
)


def _make_symbol_frame(symbol: str, scale: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=80)
    close = pd.Series(range(80), dtype=float) * 0.2 * scale + 100.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000,
            "amount": close * 1_000_000,
            "symbol": symbol,
            "market": "CN",
        }
    )


class _FakeFundamentalDataLayer:
    def get_point_in_time_fundamental_snapshot(self, symbol: str, as_of: str) -> FundamentalSnapshot:
        return FundamentalSnapshot(
            symbol=symbol,
            as_of=as_of,
            available=True,
            roe=0.18,
            gross_margin=0.36,
            profit_growth=0.15,
            revenue_growth=0.18,
            debt_ratio=0.28,
            current_ratio=1.35,
            pe=12.0,
            pb=1.8,
            ps=1.9,
            dividend_yield=0.03,
        )

    def get_earnings_forecast_snapshot(self, symbol: str, as_of: str) -> ForecastSnapshot:
        return ForecastSnapshot(
            symbol=symbol,
            as_of=as_of,
            available=False,
            source="neutral",
            data_quality={"provider_missing": True, "snapshot_missing": False, "missing_scope": "global"},
            provenance={"provider_missing": True},
            notes=["forecast_provider_missing"],
        )

    def get_management_snapshot(self, symbol: str, as_of: str) -> ManagementSnapshot:
        return ManagementSnapshot(
            symbol=symbol,
            as_of=as_of,
            available=True,
            management_stability=0.4,
            governance_score=0.5,
            management_alignment=0.3,
        )

    def get_ownership_snapshot(self, symbol: str, as_of: str) -> OwnershipSnapshot:
        return OwnershipSnapshot(
            symbol=symbol,
            as_of=as_of,
            available=True,
            concentration_score=0.2,
            institutional_holding_pct=0.25,
            ownership_change_signal=0.15,
        )

    def get_document_semantic_snapshot(self, symbol: str, as_of: str) -> CorporateDocumentSnapshot:
        return CorporateDocumentSnapshot(
            symbol=symbol,
            as_of=as_of,
            available=True,
            semantic_sentiment=0.3,
            execution_confidence=0.4,
            governance_red_flag=0.0,
        )


def test_all_research_agents_output_non_empty_thesis():
    symbols = ["000001.SZ", "600519.SH"]
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=symbols,
        symbol_data={symbol: _make_symbol_frame(symbol, scale=1.0 + idx * 0.2) for idx, symbol in enumerate(symbols)},
        event_data={symbol: [{"impact": 0.1}] for symbol in symbols},
        sentiment_data={symbol: {"fear_greed": 0.1, "money_flow": 0.2, "breadth": 0.1} for symbol in symbols},
    )

    kline = KlineAgent().run({"data_bundle": bundle, "mode": "full_market"})
    quant = QuantAgent().run({"data_bundle": bundle, "enable_alpha_mining": False})
    fundamental = FundamentalAgent().run({"data_bundle": bundle, "data_layer": _FakeFundamentalDataLayer()})
    intelligence = IntelligenceAgent().run({"data_bundle": bundle, "market_regime": "震荡低波"})
    macro = MacroAgent().run(
        {
            "market_snapshot": {
                "regime": "balanced",
                "macro_score": 0.1,
                "liquidity_score": 0.1,
                "volatility_percentile": 45.0,
            }
        }
    )

    for verdict in (kline, quant, fundamental, intelligence, macro):
        assert verdict.thesis.strip()


def test_provider_missing_enters_coverage_not_investment_risk():
    symbol = "000001.SZ"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_symbol_frame(symbol)},
        metadata={"end_date": "2026-03-23"},
    )

    verdict = FundamentalAgent().run(
        {
            "data_bundle": bundle,
            "data_layer": _FakeFundamentalDataLayer(),
        }
    )

    assert any("盈利预测" in note for note in verdict.coverage_notes)
    assert all("provider_missing" not in item for item in verdict.investment_risks)
    assert all("snapshot_missing" not in item for item in verdict.investment_risks)


def test_intelligence_does_not_repeat_financial_primary_scoring():
    symbol_a = "000001.SZ"
    symbol_b = "600519.SH"
    frame_a = _make_symbol_frame(symbol_a, scale=1.0)
    frame_b = _make_symbol_frame(symbol_b, scale=1.0)

    bundle_high_quality = UnifiedDataBundle(
        market="CN",
        symbols=[symbol_a, symbol_b],
        symbol_data={symbol_a: frame_a, symbol_b: frame_b},
        fundamentals={
            symbol_a: {"roe": 0.30, "gross_margin": 0.60, "debt_ratio": 0.10, "pe": 5},
            symbol_b: {"roe": -0.10, "gross_margin": 0.05, "debt_ratio": 0.90, "pe": 80},
        },
        event_data={symbol_a: [], symbol_b: []},
        sentiment_data={
            symbol_a: {"fear_greed": 0.1, "money_flow": 0.1, "breadth": 0.1},
            symbol_b: {"fear_greed": 0.1, "money_flow": 0.1, "breadth": 0.1},
        },
    )
    bundle_low_quality = UnifiedDataBundle(
        market="CN",
        symbols=[symbol_a, symbol_b],
        symbol_data={symbol_a: frame_a.copy(), symbol_b: frame_b.copy()},
        fundamentals={
            symbol_a: {"roe": -0.30, "gross_margin": 0.05, "debt_ratio": 0.95, "pe": 80},
            symbol_b: {"roe": 0.45, "gross_margin": 0.75, "debt_ratio": 0.05, "pe": 6},
        },
        event_data={symbol_a: [], symbol_b: []},
        sentiment_data={
            symbol_a: {"fear_greed": 0.1, "money_flow": 0.1, "breadth": 0.1},
            symbol_b: {"fear_greed": 0.1, "money_flow": 0.1, "breadth": 0.1},
        },
    )

    result_high = IntelligenceAgent().run({"data_bundle": bundle_high_quality})
    result_low = IntelligenceAgent().run({"data_bundle": bundle_low_quality})

    assert result_high.final_score == result_low.final_score
    assert "财务" not in result_high.thesis
    assert result_high.metadata["no_financial_primary_scoring"] is True


def test_kline_timeout_or_failure_still_returns_verdict():
    class _FailingBackend:
        def predict(self, symbol_data, stock_pool):
            raise TimeoutError("chronos timeout")

    symbol = "000001.SZ"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_symbol_frame(symbol)},
    )

    verdict = KlineAgent().run(
        {
            "data_bundle": bundle,
            "mode": "shortlist",
            "backend": _FailingBackend(),
        }
    )

    assert verdict.thesis.strip()
    assert any("Chronos" in note for note in verdict.diagnostic_notes)


def test_base_agent_reroutes_coverage_and_diagnostic_notes_out_of_investment_risk():
    class _DummyAgent(BaseAgent):
        agent_name = "DummyAgent"

        def run(self, payload):
            raise NotImplementedError

    verdict = _DummyAgent().branch_result_to_verdict(
        BranchResult(
            branch_name="dummy",
            explanation="结构化测试结果。",
            risks=[
                "provider_missing",
                "snapshot_missing",
                "timeout waiting for model",
                "真实投资风险保留",
            ],
            symbol_scores={"000001.SZ": 0.2},
        )
    )

    assert verdict.thesis.strip()
    assert verdict.investment_risks == ["真实投资风险保留"]
    assert any("provider_missing" in note for note in verdict.coverage_notes)
    assert any("timeout" in note for note in verdict.diagnostic_notes)
