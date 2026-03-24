"""
full-market Kline fast-screen 回归测试。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import quant_investor.market.analyze as market_analyze
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline


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


def test_batch_mode_requests_heuristic_fast_screen(monkeypatch):
    captured = {}

    class _FakeCurrent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(
                final_strategy=SimpleNamespace(
                    trade_recommendations=[],
                    target_exposure=0.35,
                    style_bias="均衡",
                    candidate_symbols=[],
                    position_limits={},
                    branch_consensus={},
                    risk_summary={},
                    execution_notes=[],
                    research_mode="production",
                ),
                branch_results={},
                execution_log=[],
            )

    monkeypatch.setattr(market_analyze, "QuantInvestor", _FakeCurrent)

    result = market_analyze.analyze_batch(
        symbols=["600000.SH", "600519.SH"],
        category="hs300",
        batch_id=1,
        market="CN",
        verbose=False,
    )

    assert result is not None
    assert captured["kline_backend"] == "heuristic"


def test_shortlist_mode_allows_deep_model_backend(monkeypatch):
    captured: list[str] = []

    class _HybridBackend:
        reliability = 0.84
        horizon_days = 5

        def predict(self, symbol_data, stock_pool):
            return BranchResult(
                branch_name="kline",
                score=0.18,
                confidence=0.71,
                symbol_scores={stock_pool[0]: 0.55},
                metadata={"branch_mode": "kline_dual_model", "reliability": 0.84, "horizon_days": 5},
                conclusion="K 线深模型链路已形成完整趋势结论。",
            )

    def _fake_get_backend(name, **kwargs):
        captured.append(name)
        assert kwargs["evaluator_name"] == "placeholder"
        return _HybridBackend()

    monkeypatch.setattr("quant_investor.kline_backends.get_backend", _fake_get_backend)

    symbol = "000001.SZ"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_symbol_frame(symbol)},
    )
    result = ParallelResearchPipeline(
        stock_pool=[symbol],
        market="CN",
        kline_backend="hybrid",
        verbose=False,
    )._run_kline_branch(bundle)

    assert captured == ["hybrid"]
    assert result.metadata["requested_backend"] == "hybrid"
    assert result.metadata["effective_backend"] == "hybrid"
    assert result.metadata["runtime_backend"] == "hybrid"
    assert result.metadata["branch_mode"] == "kline_dual_model"
    assert result.conclusion.strip()


def test_kline_deep_backend_timeout_keeps_heuristic_verdict(monkeypatch):
    class _TimeoutBackend:
        reliability = 0.84
        horizon_days = 5

        def predict(self, symbol_data, stock_pool):
            raise TimeoutError("chronos timed out")

    class _HeuristicBackend:
        reliability = 0.62
        horizon_days = 5

        def predict(self, symbol_data, stock_pool):
            return BranchResult(
                branch_name="kline",
                score=0.08,
                confidence=0.52,
                symbol_scores={stock_pool[0]: 0.20},
                metadata={"branch_mode": "kline_heuristic", "reliability": 0.62, "horizon_days": 5},
                conclusion="启发式 K 线快筛已形成完整趋势结论。",
            )

    def _fake_get_backend(name, **kwargs):
        if name == "hybrid":
            return _TimeoutBackend()
        if name == "heuristic":
            return _HeuristicBackend()
        raise AssertionError(f"unexpected backend: {name}")

    monkeypatch.setattr("quant_investor.kline_backends.get_backend", _fake_get_backend)

    symbol = "000001.SZ"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_symbol_frame(symbol)},
    )
    result = ParallelResearchPipeline(
        stock_pool=[symbol],
        market="CN",
        kline_backend="hybrid",
        verbose=False,
    )._run_kline_branch(bundle)

    assert result.metadata["requested_backend"] == "hybrid"
    assert result.metadata["effective_backend"] == "hybrid"
    assert result.metadata["runtime_backend"] == "heuristic_fallback"
    assert result.metadata["branch_mode"] == "kline_heuristic"
    assert result.conclusion.strip()
    assert any("base result" in note for note in result.diagnostic_notes)


def test_heuristic_batch_path_never_touches_deep_backend(monkeypatch):
    captured: list[str] = []

    class _HeuristicBackend:
        reliability = 0.62
        horizon_days = 5

        def predict(self, symbol_data, stock_pool):
            return BranchResult(
                branch_name="kline",
                score=0.05,
                confidence=0.50,
                symbol_scores={stock_pool[0]: 0.12},
                metadata={"branch_mode": "kline_heuristic", "reliability": 0.62, "horizon_days": 5},
                conclusion="启发式 K 线快筛已形成完整趋势结论。",
            )

    def _fake_get_backend(name, **kwargs):
        captured.append(name)
        if name != "heuristic":
            raise AssertionError("batch fast-screen 不应触发 deep-model backend")
        return _HeuristicBackend()

    monkeypatch.setattr("quant_investor.kline_backends.get_backend", _fake_get_backend)

    symbol = "000001.SZ"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_symbol_frame(symbol)},
    )
    result = ParallelResearchPipeline(
        stock_pool=[symbol],
        market="CN",
        kline_backend="heuristic",
        verbose=False,
    )._run_kline_branch(bundle)

    assert captured == ["heuristic"]
    assert result.metadata["runtime_backend"] == "heuristic"
    assert result.metadata["llm_interface_reserved"] is False
