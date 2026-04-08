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
) -> GlobalContext:
    return GlobalContext(
        universe_symbols=symbols,
        universe_tiers={"researchable": symbols, "total": symbols},
        data_quality_quarantine=quarantine or [],
        liquidity_filter={
            "suspended": suspended or [],
            "illiquid": illiquid or [],
        },
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
