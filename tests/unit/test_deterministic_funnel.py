"""Tests for the Deterministic Funnel."""

from __future__ import annotations

import pytest

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.candidate_filter import DataQualityGate, LiquidityGate, TradabilityGate
from quant_investor.funnel.deterministic_funnel import DeterministicFunnel, FunnelConfig, FunnelOutput


def _make_context(
    symbols: list[str],
    quarantine: list[str] | None = None,
    suspended: list[str] | None = None,
    illiquid: list[str] | None = None,
    industry_map: dict[str, str] | None = None,
    symbol_market_state: dict[str, dict[str, float]] | None = None,
) -> GlobalContext:
    return GlobalContext(
        universe_symbols=symbols,
        universe_tiers={"researchable": symbols, "total": symbols},
        industry_map=industry_map or {},
        data_quality_quarantine=quarantine or [],
        liquidity_filter={
            "suspended": suspended or [],
            "illiquid": illiquid or [],
        },
        metadata={"symbol_market_state": symbol_market_state or {}},
    )


def _make_branch(name: str, scores: dict[str, float]) -> BranchResult:
    return BranchResult(branch_name=name, symbol_scores=scores)


class TestGates:
    def test_data_quality_gate_excludes_quarantined(self):
        ctx = _make_context(["A", "B", "C"], quarantine=["B"])
        passed, excluded = DataQualityGate().filter(["A", "B", "C"], ctx)
        assert passed == ["A", "C"]
        assert excluded == {"B": "data_quality_quarantine"}

    def test_tradability_gate_excludes_suspended(self):
        ctx = _make_context(["A", "B"], suspended=["A"])
        passed, excluded = TradabilityGate().filter(["A", "B"], ctx)
        assert passed == ["B"]
        assert "A" in excluded

    def test_liquidity_gate_excludes_illiquid(self):
        ctx = _make_context(["A", "B"], illiquid=["B"])
        passed, excluded = LiquidityGate().filter(["A", "B"], ctx)
        assert passed == ["A"]
        assert "B" in excluded

    def test_liquidity_gate_uses_percentile_scores(self):
        ctx = GlobalContext(
            liquidity_filter={"liquidity_scores": {"A": 0.05, "B": 0.50}},
        )
        passed, excluded = LiquidityGate(percentile_min=0.10).filter(["A", "B"], ctx)
        assert passed == ["B"]
        assert "A" in excluded


class TestDeterministicFunnel:
    def test_funnel_compresses_universe(self):
        symbols = [f"S{i:04d}" for i in range(100)]
        ctx = _make_context(symbols)
        quant = _make_branch("quant", {s: float(i) / 100 for i, s in enumerate(symbols)})
        kline = _make_branch("kline", {s: float(99 - i) / 100 for i, s in enumerate(symbols)})

        funnel = DeterministicFunnel(FunnelConfig(max_candidates=20))
        output = funnel.run(quant_result=quant, kline_result=kline, global_context=ctx)

        assert isinstance(output, FunnelOutput)
        assert len(output.candidates) == 20
        assert len(output.candidate_scores) == 20
        assert output.funnel_metadata["total_universe"] == 100
        assert output.funnel_metadata["final_candidates"] == 20

    def test_funnel_applies_all_gates(self):
        symbols = ["GOOD1", "GOOD2", "BAD_Q", "BAD_S", "BAD_L"]
        ctx = _make_context(
            symbols,
            quarantine=["BAD_Q"],
            suspended=["BAD_S"],
            illiquid=["BAD_L"],
        )
        quant = _make_branch("quant", {s: 0.5 for s in symbols})
        kline = _make_branch("kline", {s: 0.5 for s in symbols})

        funnel = DeterministicFunnel(FunnelConfig(max_candidates=100))
        output = funnel.run(quant_result=quant, kline_result=kline, global_context=ctx)

        assert set(output.candidates) == {"GOOD1", "GOOD2"}
        assert "BAD_Q" in output.excluded_symbols
        assert "BAD_S" in output.excluded_symbols
        assert "BAD_L" in output.excluded_symbols

    def test_funnel_ranks_by_composite_score(self):
        symbols = ["A", "B", "C"]
        ctx = _make_context(symbols)
        quant = _make_branch("quant", {"A": 0.9, "B": 0.5, "C": 0.1})
        kline = _make_branch("kline", {"A": 0.1, "B": 0.5, "C": 0.9})

        funnel = DeterministicFunnel(FunnelConfig(max_candidates=2, quant_weight=0.6, kline_weight=0.4))
        output = funnel.run(quant_result=quant, kline_result=kline, global_context=ctx)

        assert len(output.candidates) == 2
        # A: 0.6*0.9 + 0.4*0.1 = 0.58
        # B: 0.6*0.5 + 0.4*0.5 = 0.50
        # C: 0.6*0.1 + 0.4*0.9 = 0.42
        assert output.candidates[0] == "A"
        assert output.candidates[1] == "B"
        assert pytest.approx(output.candidate_scores["A"], abs=0.01) == 0.58

    def test_funnel_empty_scores(self):
        symbols = ["A", "B"]
        ctx = _make_context(symbols)
        quant = _make_branch("quant", {})
        kline = _make_branch("kline", {})

        funnel = DeterministicFunnel(FunnelConfig(max_candidates=10))
        output = funnel.run(quant_result=quant, kline_result=kline, global_context=ctx)

        assert len(output.candidates) == 2
        assert all(score == 0.0 for score in output.candidate_scores.values())

    def test_momentum_leader_prefers_breakout_confirmation(self):
        symbols = ["A", "B", "C"]
        ctx = _make_context(
            symbols,
            symbol_market_state={
                "A": {
                    "momentum_strength": 0.90,
                    "breakout_readiness": 0.95,
                    "volume_confirmation": 0.80,
                    "trend_stability": 0.85,
                    "distance_from_high_pct": 0.01,
                    "fake_breakout_risk": 0.10,
                    "max_drawdown_pct": 0.04,
                    "return_20d": 0.16,
                },
                "B": {
                    "momentum_strength": 0.82,
                    "breakout_readiness": 0.90,
                    "volume_confirmation": 0.00,
                    "trend_stability": 0.55,
                    "distance_from_high_pct": 0.02,
                    "fake_breakout_risk": 0.95,
                    "max_drawdown_pct": 0.18,
                    "return_20d": 0.10,
                },
                "C": {
                    "momentum_strength": 0.55,
                    "breakout_readiness": 0.45,
                    "volume_confirmation": 0.30,
                    "trend_stability": 0.55,
                    "distance_from_high_pct": 0.08,
                    "fake_breakout_risk": 0.20,
                    "max_drawdown_pct": 0.06,
                    "return_20d": 0.05,
                },
            },
        )
        quant = _make_branch("quant", {"A": 0.4, "B": 0.5, "C": 0.2})
        kline = _make_branch("kline", {"A": 0.8, "B": 0.85, "C": 0.3})

        funnel = DeterministicFunnel(
            FunnelConfig(
                profile="momentum_leader",
                max_candidates=2,
                sector_bucket_limit=0,
            )
        )
        output = funnel.run(quant_result=quant, kline_result=kline, global_context=ctx)

        assert output.candidates == ["A", "C"]
        assert output.candidate_scores["A"] > output.candidate_scores["C"]
        assert "B" in output.excluded_symbols

    def test_momentum_leader_limits_sector_crowding(self):
        symbols = ["A", "B", "C"]
        ctx = _make_context(
            symbols,
            industry_map={"A": "半导体", "B": "半导体", "C": "银行"},
            symbol_market_state={
                "A": {"momentum_strength": 0.92, "breakout_readiness": 0.90, "volume_confirmation": 0.70, "trend_stability": 0.80, "distance_from_high_pct": 0.01, "fake_breakout_risk": 0.12, "max_drawdown_pct": 0.05, "return_20d": 0.18},
                "B": {"momentum_strength": 0.89, "breakout_readiness": 0.88, "volume_confirmation": 0.72, "trend_stability": 0.78, "distance_from_high_pct": 0.02, "fake_breakout_risk": 0.15, "max_drawdown_pct": 0.06, "return_20d": 0.17},
                "C": {"momentum_strength": 0.78, "breakout_readiness": 0.76, "volume_confirmation": 0.60, "trend_stability": 0.74, "distance_from_high_pct": 0.03, "fake_breakout_risk": 0.10, "max_drawdown_pct": 0.04, "return_20d": 0.12},
            },
        )
        quant = _make_branch("quant", {"A": 0.6, "B": 0.58, "C": 0.45})
        kline = _make_branch("kline", {"A": 0.8, "B": 0.79, "C": 0.62})

        funnel = DeterministicFunnel(
            FunnelConfig(
                profile="momentum_leader",
                max_candidates=2,
                sector_bucket_limit=1,
            )
        )
        output = funnel.run(quant_result=quant, kline_result=kline, global_context=ctx)

        assert output.candidates == ["A", "C"]
        assert output.excluded_symbols["B"] == "sector_bucket_limit"
