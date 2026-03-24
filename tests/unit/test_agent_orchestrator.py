"""
AgentOrchestrator 过渡编排测试。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_investor.agent_orchestrator import AgentOrchestrator
from quant_investor.agent_protocol import ActionLabel, BranchVerdict, PortfolioPlan, ReportBundle
from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.enhanced_data_layer import EnhancedDataLayer
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline


def _make_symbol_frame(symbol: str, scale: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=120)
    close = np.linspace(100, 100 * (1 + 0.08 * scale), len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(dates), 2_000_000),
            "amount": close * 2_000_000,
            "symbol": symbol,
            "market": "CN",
            "forward_ret_5d": pd.Series(close).shift(-5) / pd.Series(close) - 1,
        }
    )


class _CountingMacroAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, payload):
        self.calls += 1
        return BranchVerdict(
            agent_name="MacroAgent",
            thesis="宏观中性偏稳。",
            final_score=0.1,
            final_confidence=0.8,
            metadata={
                "regime": "balanced",
                "target_gross_exposure": 0.6,
                "style_bias": "balanced_quality",
                "symbol": None,
            },
        )


class _FakeResearchAgent:
    def __init__(self, agent_name: str, score_map: dict[str, float], risk_symbol: str | None = None) -> None:
        self.agent_name = agent_name
        self.score_map = score_map
        self.risk_symbol = risk_symbol
        self.calls: list[str] = []

    def run(self, payload):
        symbol = list(payload.get("stock_pool") or payload["data_bundle"].symbols)[0]
        self.calls.append(symbol)
        risks = ["fraud investigation ongoing"] if symbol == self.risk_symbol else []
        return BranchVerdict(
            agent_name=self.agent_name,
            thesis=f"{self.agent_name} 对 {symbol} 给出结构化判断。",
            symbol=symbol,
            final_score=self.score_map[symbol],
            final_confidence=0.72,
            investment_risks=risks,
            metadata={
                "legacy_symbol_scores": {symbol: self.score_map[symbol]},
            },
        )


class _FakeTerminal:
    def generate_risk_report(self):
        return SimpleNamespace(
            overall_signal="🟢",
            overall_risk_level="低风险",
            recommendation="积极布局",
        )


def test_agent_orchestrator_runs_fixed_chain_and_propagates_veto(tmp_path):
    symbols = ["AAA", "BBB"]
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=symbols,
        symbol_data={
            "AAA": _make_symbol_frame("AAA", scale=1.0),
            "BBB": _make_symbol_frame("BBB", scale=0.8),
        },
        fundamentals={
            "AAA": {"sector": "defensive"},
            "BBB": {"sector": "growth"},
        },
    )
    macro_agent = _CountingMacroAgent()
    kline_agent = _FakeResearchAgent("KlineAgent", {"AAA": 0.45, "BBB": 0.50})
    quant_agent = _FakeResearchAgent("QuantAgent", {"AAA": 0.40, "BBB": 0.55}, risk_symbol="BBB")
    fundamental_agent = _FakeResearchAgent("FundamentalAgent", {"AAA": 0.35, "BBB": 0.48})
    intelligence_agent = _FakeResearchAgent("IntelligenceAgent", {"AAA": 0.30, "BBB": 0.42})

    output = AgentOrchestrator(
        macro_agent=macro_agent,
        kline_agent=kline_agent,
        quant_agent=quant_agent,
        fundamental_agent=fundamental_agent,
        intelligence_agent=intelligence_agent,
    ).run(
        data_bundle=bundle,
        constraints={
            "gross_exposure_cap": 0.7,
            "max_weight": 0.35,
            "sector_caps": {"defensive": 0.5, "growth": 0.5},
            "veto_keywords": ["fraud"],
        },
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={
            "AAA": {"is_tradable": True, "sector": "defensive", "liquidity_score": 1.0},
            "BBB": {"is_tradable": True, "sector": "growth", "liquidity_score": 1.0},
        },
        persist_dir=tmp_path,
    )

    assert macro_agent.calls == 1
    assert kline_agent.calls == symbols
    assert quant_agent.calls == symbols
    assert fundamental_agent.calls == symbols
    assert intelligence_agent.calls == symbols

    assert set(output["ic_by_symbol"]) == set(symbols)
    assert isinstance(output["portfolio_plan"], PortfolioPlan)
    assert isinstance(output["report_bundle"], ReportBundle)
    assert output["ic_by_symbol"]["BBB"].status.name == "VETOED"
    assert output["ic_by_symbol"]["BBB"].action is ActionLabel.HOLD
    assert "BBB" not in output["portfolio_plan"].target_positions
    assert "AAA" in output["portfolio_plan"].target_positions
    assert Path(output["persisted_paths"]["markdown_report"]).exists()


def test_parallel_pipeline_bridge_attaches_agent_outputs_without_replacing_legacy(monkeypatch):
    frame = _make_symbol_frame("000001.SZ", scale=1.0)
    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: frame.copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: _FakeTerminal(),
    )

    pipeline = ParallelResearchPipeline(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
    )
    result = pipeline.run()

    assert result.final_strategy is not None
    assert result.final_report
    assert hasattr(result, "agent_portfolio_plan")
    assert hasattr(result, "agent_report_bundle")
    assert hasattr(result, "agent_ic_decisions")
    assert len(result.agent_ic_decisions) == 1


def test_risk_hold_cap_without_existing_position_cannot_open_new_position():
    symbol = "AAA"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_symbol_frame(symbol, scale=1.0)},
        fundamentals={symbol: {"sector": "defensive"}},
    )
    output = AgentOrchestrator(
        macro_agent=_CountingMacroAgent(),
        kline_agent=_FakeResearchAgent("KlineAgent", {symbol: 0.55}),
        quant_agent=_FakeResearchAgent("QuantAgent", {symbol: 0.52}),
        fundamental_agent=_FakeResearchAgent("FundamentalAgent", {symbol: 0.48}),
        intelligence_agent=_FakeResearchAgent("IntelligenceAgent", {symbol: 0.44}),
    ).run(
        data_bundle=bundle,
        constraints={
            "action_cap": ActionLabel.HOLD,
            "gross_exposure_cap": 0.5,
            "max_weight": 0.2,
        },
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={
            symbol: {"is_tradable": True, "sector": "defensive", "liquidity_score": 1.0},
        },
        persist_outputs=False,
    )

    decision = output["ic_by_symbol"][symbol]
    assert decision.action is ActionLabel.HOLD
    assert decision.metadata["symbol_candidates"][0]["position_mode"] == "watch"
    assert symbol not in output["portfolio_plan"].target_positions
