"""
设计意图 5：full-market 模式下 Kline 应走 fast-screen，deep model 仅用于 shortlist。

当前预期：通过。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_investor.agents.kline_agent import KlineAgent
from quant_investor.branch_contracts import BranchResult


def _make_symbol_frame(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=120)
    close = np.linspace(100, 108, len(dates))
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


def test_full_market_mode_uses_heuristic_fast_screen_only(monkeypatch) -> None:
    class _HeuristicBackend:
        def predict(self, symbol_data, stock_pool):
            return BranchResult(
                branch_name="kline",
                score=0.08,
                confidence=0.52,
                symbol_scores={stock_pool[0]: 0.20},
                metadata={"branch_mode": "kline_heuristic"},
                conclusion="启发式 K 线快筛已形成完整趋势结论。",
            )

    monkeypatch.setattr(
        "quant_investor.kline_backends.get_backend",
        lambda name, **kwargs: (_ for _ in ()).throw(
            AssertionError("full_market 快筛不应触发 deep-model backend 初始化")
        ),
    )

    verdict = KlineAgent().run(
        {
            "symbol_data": {f"S{i:03d}": _make_symbol_frame(f"S{i:03d}") for i in range(25)},
            "stock_pool": [f"S{i:03d}" for i in range(25)],
            "mode": "full_market",
            "backend_name": "hybrid",
            "heuristic_backend": _HeuristicBackend(),
        }
    )

    assert verdict.metadata["backend_used"] == "heuristic"
    assert verdict.metadata["mode"] == "full_market"
    assert "full market 快筛模式" in verdict.thesis


def test_shortlist_mode_can_use_deep_backend(monkeypatch) -> None:
    captured: list[str] = []

    class _HybridBackend:
        def predict(self, symbol_data, stock_pool):
            return BranchResult(
                branch_name="kline",
                score=0.18,
                confidence=0.71,
                symbol_scores={stock_pool[0]: 0.55},
                metadata={"branch_mode": "kline_dual_model"},
                conclusion="K 线深模型链路已形成完整趋势结论。",
            )

    def _fake_get_backend(name, **kwargs):
        captured.append(name)
        return _HybridBackend()

    monkeypatch.setattr("quant_investor.kline_backends.get_backend", _fake_get_backend)

    verdict = KlineAgent().run(
        {
            "symbol_data": {"000001.SZ": _make_symbol_frame("000001.SZ")},
            "stock_pool": ["000001.SZ"],
            "mode": "shortlist",
            "backend_name": "hybrid",
        }
    )

    assert captured == ["hybrid"]
    assert verdict.metadata["backend_used"] == "hybrid"
    assert verdict.metadata["mode"] == "shortlist"

