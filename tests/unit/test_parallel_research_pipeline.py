"""
并行研究架构测试
"""

from __future__ import annotations

import importlib
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_investor.config import config
from quant_investor.contracts import BranchResult, UnifiedDataBundle
from quant_investor.market.analyze import build_full_market_trade_plan
from quant_investor.enhanced_data_layer import EnhancedDataLayer
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline
from quant_investor.pipeline.quant_investor_v8 import QuantInvestorV8


def _make_symbol_frame(symbol: str, scale: float = 1.0, volatile: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=160)
    base_seed = abs(hash(symbol)) % 1000
    rng = np.random.default_rng(base_seed)
    shock_scale = 0.05 if volatile else 0.015
    returns = rng.normal(0.001 * scale, shock_scale, len(dates))
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, len(dates)))
    volume = rng.integers(1_000_000, 5_000_000, len(dates))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "symbol": symbol,
            "market": "CN",
        }
    )
    df["return_20d"] = df["close"].pct_change(20)
    df["momentum_12_1"] = df["close"].pct_change(120) - df["close"].pct_change(20)
    df["volatility_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
    df["rsi_14"] = 55 + scale * 5
    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["ma_bias_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).mean()
    df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
    df["label_return"] = df["close"].shift(-5) / df["close"] - 1
    df["forward_ret_5d"] = df["label_return"]
    df["roe"] = 0.1 + 0.02 * scale
    df["gross_margin"] = 0.25 + 0.03 * scale
    df["profit_growth"] = 0.08 * scale
    df["revenue_growth"] = 0.05 * scale
    df["debt_ratio"] = 0.35 if not volatile else 0.65
    df["pe"] = 14 - scale
    df["pb"] = 1.8
    df["ps"] = 1.4
    return df


class _FakeTerminal:
    def __init__(self, signal: str = "🟢", risk_level: str = "低风险", recommendation: str = "正常配置"):
        self.signal = signal
        self.risk_level = risk_level
        self.recommendation = recommendation

    def generate_risk_report(self):
        return SimpleNamespace(
            overall_signal=self.signal,
            overall_risk_level=self.risk_level,
            recommendation=self.recommendation,
        )


@pytest.fixture(autouse=True)
def _force_internal_fallback_models(monkeypatch, tmp_path):
    """单测固定走空模型目录，保证 K-line 输出稳定且不依赖本机权重状态。"""
    monkeypatch.setattr(config, "KRONOS_MODEL_PATH", str(tmp_path / "kronos"))
    monkeypatch.setattr(config, "CHRONOS_MODEL_NAME", str(tmp_path / "chronos-2"))
    monkeypatch.setattr(config, "KLINE_ALLOW_REMOTE_MODEL_DOWNLOAD", False)


def test_parallel_pipeline_runs_with_all_branches(monkeypatch):
    symbol_frames = {
        "000001.SZ": _make_symbol_frame("000001.SZ", scale=1.0),
        "600519.SH": _make_symbol_frame("600519.SH", scale=1.3),
        "000858.SZ": _make_symbol_frame("000858.SZ", scale=0.9),
    }

    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: symbol_frames[symbol].copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: _FakeTerminal(signal="🟢", risk_level="低风险", recommendation="积极布局"),
    )

    pipeline = ParallelResearchPipeline(
        stock_pool=list(symbol_frames.keys()),
        market="CN",
        verbose=False,
    )
    result = pipeline.run()

    assert set(result.branch_results.keys()) == {
        "kline",
        "quant",
        "llm_debate",
        "intelligence",
        "macro",
    }
    assert 0 <= result.final_strategy.target_exposure <= 1
    assert result.final_strategy.candidate_symbols
    assert result.final_strategy.trade_recommendations
    first_recommendation = result.final_strategy.trade_recommendations[0]
    assert first_recommendation.suggested_weight > 0
    assert first_recommendation.target_price > first_recommendation.stop_loss_price
    assert first_recommendation.suggested_shares % 100 == 0
    assert "组合级策略结论" in result.final_report
    assert "可执行交易计划" in result.final_report
    assert result.calibrated_signals["kline"].branch_mode == "kline_dual_model"
    assert result.calibrated_signals["llm_debate"].branch_mode == "structured_research_debate"
    assert result.final_strategy.provenance_summary["synthetic_symbols"] == []
    assert result.branch_results["kline"].metadata["effective_backend"] == "hybrid"
    assert result.branch_results["kline"].metadata["requested_backend"] == "hybrid"
    assert result.branch_results["kline"].metadata["llm_interface_reserved"] is True


def test_pipeline_degrades_when_single_branch_fails(monkeypatch):
    symbol_frames = {
        "000001.SZ": _make_symbol_frame("000001.SZ", scale=1.0),
        "600519.SH": _make_symbol_frame("600519.SH", scale=1.2),
    }

    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: symbol_frames[symbol].copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: _FakeTerminal(signal="🟡", risk_level="中风险", recommendation="控制仓位"),
    )

    def _boom(self, data_bundle):
        raise RuntimeError("kronos offline")

    monkeypatch.setattr(ParallelResearchPipeline, "_run_kline_branch", _boom)

    pipeline = ParallelResearchPipeline(stock_pool=list(symbol_frames.keys()), market="CN", verbose=False)
    result = pipeline.run()

    assert result.branch_results["kline"].success is False
    assert "kline" in result.branch_results["kline"].branch_name
    assert 0 <= result.final_strategy.target_exposure <= 1


def test_kline_branch_forces_hybrid_backend_and_records_requested_backend(monkeypatch):
    captured = {}

    class _FakeBackend:
        name = "hybrid"
        reliability = 0.84
        horizon_days = 5

        def predict(self, symbol_data, stock_pool):
            return BranchResult(
                branch_name="kline",
                score=0.12,
                confidence=0.66,
                signals={"predicted_return": {stock_pool[0]: 0.03}, "trend_regime": {stock_pool[0]: "上行"}},
                symbol_scores={stock_pool[0]: 0.4},
                metadata={"branch_mode": "kline_dual_model", "reliability": 0.73, "horizon_days": 5},
            )

    def _fake_get_backend(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return _FakeBackend()

    monkeypatch.setattr("quant_investor.kline_backends.get_backend", _fake_get_backend)

    symbol = "000001.SZ"
    pipeline = ParallelResearchPipeline(
        stock_pool=[symbol],
        market="CN",
        kline_backend="heuristic",
        verbose=False,
    )
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_symbol_frame(symbol)},
    )

    result = pipeline._run_kline_branch(bundle)

    assert captured["name"] == "hybrid"
    assert captured["kwargs"]["evaluator_name"] == "placeholder"
    assert captured["kwargs"]["allow_remote_download"] is False
    assert result.metadata["requested_backend"] == "heuristic"
    assert result.metadata["effective_backend"] == "hybrid"
    assert result.metadata["branch_mode"] == "kline_dual_model"


def test_pipeline_skips_quant_branch_when_disabled(monkeypatch):
    symbol_frames = {
        "000001.SZ": _make_symbol_frame("000001.SZ", scale=1.0),
        "600519.SH": _make_symbol_frame("600519.SH", scale=1.2),
    }

    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: symbol_frames[symbol].copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: _FakeTerminal(signal="🟡", risk_level="中风险", recommendation="控制仓位"),
    )

    def _quant_should_not_run(self, data_bundle):
        raise AssertionError("quant branch should remain disabled")

    monkeypatch.setattr(ParallelResearchPipeline, "_run_quant_branch", _quant_should_not_run)

    pipeline = ParallelResearchPipeline(
        stock_pool=list(symbol_frames.keys()),
        market="CN",
        enable_quant=False,
        verbose=False,
    )
    result = pipeline.run()

    assert "quant" not in result.branch_results
    assert set(result.branch_results.keys()) == {"kline", "llm_debate", "intelligence", "macro"}


def test_pipeline_degrades_timed_out_branch_without_aborting(monkeypatch, tmp_path):
    symbol_frames = {
        "000001.SZ": _make_symbol_frame("000001.SZ", scale=1.0),
        "600519.SH": _make_symbol_frame("600519.SH", scale=1.2),
    }
    marker_file = tmp_path / "llm_branch_finished.txt"

    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: symbol_frames[symbol].copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: _FakeTerminal(signal="🟢", risk_level="低风险", recommendation="积极布局"),
    )

    def _slow_llm_branch(self, data_bundle):
        time.sleep(0.2)
        marker_file.write_text("finished", encoding="utf-8")
        return BranchResult(branch_name="llm_debate", score=0.4, confidence=0.6)

    monkeypatch.setattr(ParallelResearchPipeline, "_run_llm_debate_branch", _slow_llm_branch)

    pipeline = ParallelResearchPipeline(
        stock_pool=list(symbol_frames.keys()),
        market="CN",
        branch_timeout=0.05,
        verbose=False,
    )
    result = pipeline.run()

    assert result.branch_results["llm_debate"].success is False
    assert result.branch_results["llm_debate"].metadata["degraded_reason"] == "branch_timeout"
    assert "超时" in result.branch_results["llm_debate"].explanation
    assert 0 <= result.final_strategy.target_exposure <= 1
    time.sleep(0.25)
    assert not marker_file.exists()


def test_synthetic_symbol_is_excluded_from_candidates(monkeypatch):
    symbol_frames = {
        "000001.SZ": _make_symbol_frame("000001.SZ", scale=1.0),
        "600519.SH": _make_symbol_frame("600519.SH", scale=1.2),
    }

    def _fetch(self, symbol, start_date, end_date, label_periods=5):
        if symbol == "600519.SH":
            raise RuntimeError("data offline")
        return symbol_frames[symbol].copy()

    monkeypatch.setattr(EnhancedDataLayer, "fetch_and_process", _fetch)
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: _FakeTerminal(signal="🟢", risk_level="低风险", recommendation="积极布局"),
    )

    pipeline = ParallelResearchPipeline(stock_pool=list(symbol_frames.keys()), market="CN", verbose=False)
    result = pipeline.run()

    assert "600519.SH" in result.data_bundle.synthetic_symbols()
    assert "600519.SH" not in result.final_strategy.candidate_symbols
    assert result.final_strategy.research_mode == "degraded"
    assert "可信度与降级状态" in result.final_report


def test_all_synthetic_symbols_degrade_to_research_only(monkeypatch):
    def _offline(self, symbol, start_date, end_date, label_periods=5):
        raise RuntimeError("market offline")

    monkeypatch.setattr(EnhancedDataLayer, "fetch_and_process", _offline)
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: _FakeTerminal(signal="🟡", risk_level="中风险", recommendation="控制仓位"),
    )

    pipeline = ParallelResearchPipeline(
        stock_pool=["000001.SZ", "600519.SH"],
        market="CN",
        verbose=False,
    )
    result = pipeline.run()

    assert result.final_strategy.research_mode == "research_only"
    assert result.final_strategy.target_exposure == 0
    assert result.final_strategy.position_limits == {}
    assert result.final_strategy.provenance_summary["research_only_reason"]


def test_quant_investor_exports_v8_entrypoint():
    unified = importlib.import_module("quant_investor")

    assert hasattr(unified, "QuantInvestorV8")
    assert hasattr(unified, "BranchResult")


def test_quant_investor_v8_forwards_enable_quant(monkeypatch):
    captured_kwargs = {}

    class _FakePipeline:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def run(self):
            return SimpleNamespace(
                data_bundle=UnifiedDataBundle(market="CN", symbols=["000001.SZ"], symbol_data={}),
                branch_results={},
                calibrated_signals={},
                risk_result=None,
                final_strategy=SimpleNamespace(),
                final_report="",
                execution_log=[],
                timings={},
            )

    monkeypatch.setattr("quant_investor.pipeline.quant_investor_v8.ParallelResearchPipeline", _FakePipeline)

    investor = QuantInvestorV8(
        stock_pool=["000001.SZ"],
        market="CN",
        enable_quant=False,
        verbose=False,
    )
    investor.run()

    assert captured_kwargs["enable_quant"] is False


def test_risk_and_ensemble_reduce_exposure_in_high_risk_case():
    symbols = ["000001.SZ", "600519.SH", "000858.SZ"]
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=symbols,
        symbol_data={
            symbol: _make_symbol_frame(symbol, scale=0.4 if idx == 0 else -0.2 * idx, volatile=True)
            for idx, symbol in enumerate(symbols)
        },
    )
    bundle.fundamentals = {
        symbol: {"roe": 0.06, "gross_margin": 0.18, "debt_ratio": 0.68, "pe": 22}
        for symbol in symbols
    }
    bundle.sentiment_data = {
        symbol: {"fear_greed": -0.5, "money_flow": -0.4, "breadth": -0.2}
        for symbol in symbols
    }
    bundle.event_data = {
        symbol: [{"type": "drawdown", "headline": "大幅回撤", "impact": -0.4}]
        for symbol in symbols
    }

    pipeline = ParallelResearchPipeline(stock_pool=symbols, market="CN", verbose=False)
    branch_results = {
        "kline": BranchResult("kline", score=0.7, confidence=0.7, symbol_scores={s: 0.8 for s in symbols}, metadata={"branch_mode": "kline_heuristic", "reliability": 0.62, "horizon_days": 5}),
        "quant": BranchResult("quant", score=-0.6, confidence=0.8, symbol_scores={s: -0.7 for s in symbols}),
        "llm_debate": BranchResult("llm_debate", score=-0.4, confidence=0.6, symbol_scores={s: -0.5 for s in symbols}),
        "intelligence": BranchResult("intelligence", score=-0.7, confidence=0.7, symbol_scores={s: -0.8 for s in symbols}),
        "macro": BranchResult(
            "macro",
            score=-0.9,
            confidence=0.9,
            signals={"liquidity_signal": "🔴", "risk_level": "高风险"},
            symbol_scores={s: -0.9 for s in symbols},
        ),
    }

    risk_result = pipeline._run_risk_layer(bundle, branch_results)
    strategy = pipeline._run_ensemble_layer(bundle, branch_results, risk_result)

    assert risk_result.risk_level in {"warning", "danger"}
    assert strategy.target_exposure <= 0.55
    assert strategy.risk_summary["risk_level"] in {"warning", "danger"}
    assert {"target_exposure", "style_bias", "candidate_symbols"} <= set(strategy.__dict__.keys())
    assert strategy.provenance_summary["synthetic_symbols"] == []


def test_full_market_trade_plan_builds_portfolio_actions():
    all_results = {
        "hs300": [
            {
                "stock_count": 30,
                "branches": {
                    "kline": {"score": 0.08},
                    "quant": {"score": 0.12},
                    "llm_debate": {"score": 0.05},
                    "intelligence": {"score": 0.10},
                    "macro": {"score": 0.02},
                },
                "strategy": {
                    "target_exposure": 0.46,
                    "style_bias": "均衡",
                    "candidate_symbols": ["600000.SH"],
                    "risk_summary": {"risk_level": "normal"},
                },
                "recommendations": [
                    {
                        "symbol": "600000.SH",
                        "action": "buy",
                        "data_source_status": "real",
                        "suggested_weight": 0.12,
                        "recommended_entry_price": 10.0,
                        "current_price": 10.2,
                        "target_price": 11.4,
                        "stop_loss_price": 9.2,
                        "expected_upside": 0.14,
                        "model_expected_return": 0.11,
                        "consensus_score": 0.32,
                        "confidence": 0.56,
                        "branch_positive_count": 4,
                        "lot_size": 100,
                        "entry_price_range": {"low": 9.8, "high": 10.2},
                        "risk_flags": ["波动率中等"],
                        "position_management": ["首次建仓 60%"],
                    }
                ],
            }
        ],
        "zz500": [
            {
                "stock_count": 30,
                "branches": {
                    "kline": {"score": 0.05},
                    "quant": {"score": 0.10},
                    "llm_debate": {"score": 0.03},
                    "intelligence": {"score": 0.07},
                    "macro": {"score": 0.02},
                },
                "strategy": {
                    "target_exposure": 0.42,
                    "style_bias": "成长",
                    "candidate_symbols": ["002001.SZ"],
                    "risk_summary": {"risk_level": "normal"},
                },
                "recommendations": [
                    {
                        "symbol": "002001.SZ",
                        "action": "buy",
                        "data_source_status": "real",
                        "suggested_weight": 0.09,
                        "recommended_entry_price": 20.0,
                        "current_price": 20.4,
                        "target_price": 23.0,
                        "stop_loss_price": 18.4,
                        "expected_upside": 0.15,
                        "model_expected_return": 0.10,
                        "consensus_score": 0.28,
                        "confidence": 0.52,
                        "branch_positive_count": 4,
                        "lot_size": 100,
                        "entry_price_range": {"low": 19.6, "high": 20.3},
                        "risk_flags": ["等待回踩"],
                        "position_management": ["目标价附近先止盈 50%"],
                    }
                ],
            }
        ],
    }

    plan = build_full_market_trade_plan(all_results, total_capital=1_000_000, top_k=2)

    assert plan["portfolio_plan"]["selected_count"] == 2
    assert plan["portfolio_plan"]["planned_investment"] > 0
    assert plan["recommendations"][0]["portfolio_shares"] % 100 == 0
    assert plan["recommendations"][1]["portfolio_shares"] % 100 == 0
