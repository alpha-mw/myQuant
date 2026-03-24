"""
设计意图 2：PortfolioConstructor 必须 deterministic。

当前预期：
- 组件级 deterministic 通过
- current V9 主输出仍未切到 PortfolioConstructor，会暴露入口未收口 blocker
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import quant_investor
from quant_investor.agent_protocol import ActionLabel, BranchVerdict, ICDecision
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.enhanced_data_layer import EnhancedDataLayer


def _macro_verdict(target_gross_exposure: float = 0.55) -> BranchVerdict:
    return BranchVerdict(
        agent_name="MacroAgent",
        thesis="宏观要求控制总暴露。",
        final_score=0.10,
        final_confidence=0.82,
        metadata={
            "regime": "balanced",
            "target_gross_exposure": target_gross_exposure,
            "style_bias": "balanced_quality",
        },
    )


def _deterministic_payload() -> dict:
    return {
        "ic_decisions": [
            ICDecision(
                thesis="优先保留消费与银行龙头。",
                final_score=0.70,
                final_confidence=0.85,
                action=ActionLabel.BUY,
                metadata={
                    "symbol_candidates": [
                        {
                            "symbol": "000001.SZ",
                            "score": 0.55,
                            "confidence": 0.75,
                            "action": "hold",
                            "position_mode": "target",
                            "sector": "bank",
                        },
                        {
                            "symbol": "600519.SH",
                            "score": 0.80,
                            "confidence": 0.90,
                            "action": "buy",
                            "position_mode": "target",
                            "sector": "consumer",
                        },
                    ]
                },
            ),
            ICDecision(
                thesis="重复输入但顺序不同，用于验证稳定排序。",
                final_score=0.68,
                final_confidence=0.80,
                action=ActionLabel.BUY,
                metadata={
                    "symbol_candidates": [
                        {
                            "symbol": "600519.SH",
                            "score": 0.80,
                            "confidence": 0.90,
                            "action": "buy",
                            "position_mode": "target",
                            "sector": "consumer",
                        },
                        {
                            "symbol": "000001.SZ",
                            "score": 0.55,
                            "confidence": 0.75,
                            "action": "hold",
                            "position_mode": "target",
                            "sector": "bank",
                        },
                    ]
                },
            ),
        ],
        "macro_verdict": _macro_verdict(0.55),
        "risk_limits": {
            "gross_exposure_cap": 0.60,
            "max_weight": 0.30,
            "position_limits": {
                "600519.SH": 0.28,
                "000001.SZ": 0.25,
            },
            "sector_caps": {
                "consumer": 0.30,
                "bank": 0.30,
            },
        },
        "existing_portfolio": {
            "current_weights": {
                "000001.SZ": 0.05,
                "600519.SH": 0.10,
            }
        },
        "tradability_snapshot": {
            "600519.SH": {
                "is_tradable": True,
                "sector": "consumer",
                "liquidity_score": 1.0,
            },
            "000001.SZ": {
                "is_tradable": True,
                "sector": "bank",
                "liquidity_score": 0.9,
            },
        },
    }


def _make_frame(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=100)
    close = np.linspace(100, 110, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(dates), 1_000_000),
            "amount": close * 1_000_000,
            "symbol": symbol,
            "market": "CN",
            "forward_ret_5d": pd.Series(close).shift(-5) / pd.Series(close) - 1,
        }
    )


def _patch_v9_runtime(monkeypatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: frame.copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: type(
            "_FakeTerminal",
            (),
            {
                "generate_risk_report": lambda self: SimpleNamespace(
                    overall_signal="🟢",
                    overall_risk_level="低风险",
                    recommendation="积极布局",
                ),
            },
        )(),
    )


def test_portfolio_constructor_is_order_invariant_and_deterministic() -> None:
    constructor = PortfolioConstructor()
    payload = _deterministic_payload()
    reversed_payload = _deterministic_payload()
    reversed_payload["ic_decisions"] = list(reversed(reversed_payload["ic_decisions"]))
    reversed_payload["tradability_snapshot"] = {
        "000001.SZ": reversed_payload["tradability_snapshot"]["000001.SZ"],
        "600519.SH": reversed_payload["tradability_snapshot"]["600519.SH"],
    }

    plan_a = constructor.run(payload)
    plan_b = constructor.run(reversed_payload)

    assert plan_a == plan_b
    assert plan_a.metadata["deterministic"] is True
    assert plan_a.target_gross_exposure <= 0.55 + 1e-9


def test_current_v9_main_output_should_route_through_portfolio_constructor(monkeypatch) -> None:
    frame = _make_frame("000001.SZ")
    _patch_v9_runtime(monkeypatch, frame)

    result = quant_investor.QuantInvestorV9(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
    ).run()

    assert result.agent_portfolio_plan is not None
    assert result.final_strategy.__class__.__name__ == "PortfolioPlan", (
        "QuantInvestorV9 当前主输出仍是 legacy PortfolioStrategy，"
        "PortfolioConstructor 只挂在 agent_portfolio_plan sidecar；"
        "这是入口/主输出未收口，不是 deterministic 逻辑缺失。"
    )

