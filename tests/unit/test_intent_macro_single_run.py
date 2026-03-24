"""
设计意图 4：MacroAgent 每次运行只允许一次。

当前预期：在 AgentOrchestrator 固定调用链上通过。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_investor.agent_orchestrator import AgentOrchestrator
from quant_investor.agent_protocol import BranchVerdict
from quant_investor.branch_contracts import UnifiedDataBundle


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
            final_score=0.10,
            final_confidence=0.80,
            metadata={"regime": "balanced", "target_gross_exposure": 0.60, "style_bias": "balanced_quality"},
        )


class _FakeResearchAgent:
    def __init__(self, agent_name: str, score_map: dict[str, float]) -> None:
        self.agent_name = agent_name
        self.score_map = score_map

    def run(self, payload):
        symbol = list(payload.get("stock_pool") or payload["data_bundle"].symbols)[0]
        return BranchVerdict(
            agent_name=self.agent_name,
            thesis=f"{self.agent_name} 对 {symbol} 给出结构化判断。",
            symbol=symbol,
            final_score=self.score_map[symbol],
            final_confidence=0.72,
            metadata={"legacy_symbol_scores": {symbol: self.score_map[symbol]}},
        )


def test_macro_agent_runs_once_per_orchestrated_research_run() -> None:
    symbols = ["AAA", "BBB"]
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=symbols,
        symbol_data={
            "AAA": _make_symbol_frame("AAA", scale=1.0),
            "BBB": _make_symbol_frame("BBB", scale=0.8),
        },
        fundamentals={"AAA": {"sector": "defensive"}, "BBB": {"sector": "growth"}},
    )
    macro_agent = _CountingMacroAgent()

    AgentOrchestrator(
        macro_agent=macro_agent,
        kline_agent=_FakeResearchAgent("KlineAgent", {"AAA": 0.45, "BBB": 0.50}),
        quant_agent=_FakeResearchAgent("QuantAgent", {"AAA": 0.40, "BBB": 0.55}),
        fundamental_agent=_FakeResearchAgent("FundamentalAgent", {"AAA": 0.35, "BBB": 0.48}),
        intelligence_agent=_FakeResearchAgent("IntelligenceAgent", {"AAA": 0.30, "BBB": 0.42}),
    ).run(
        data_bundle=bundle,
        constraints={"gross_exposure_cap": 0.70, "max_weight": 0.35},
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={
            "AAA": {"is_tradable": True, "sector": "defensive", "liquidity_score": 1.0},
            "BBB": {"is_tradable": True, "sector": "growth", "liquidity_score": 1.0},
        },
        persist_outputs=False,
    )

    assert macro_agent.calls == 1

