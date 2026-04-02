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

from quant_investor.branch_debate_engine import BranchDebateEngine
from quant_investor.branch_contracts import BranchResult, EvidencePacket, UnifiedDataBundle
from quant_investor.config import config
from quant_investor.ensemble_judge import EnsembleJudge
from quant_investor import QuantInvestor
from quant_investor.market.analyze import build_full_market_trade_plan
from quant_investor.enhanced_data_layer import EnhancedDataLayer
from quant_investor.pipeline.parallel_research_pipeline import BranchPerformanceTracker, ParallelResearchPipeline
from quant_investor.versioning import CURRENT_BRANCH_WEIGHTS


def _make_symbol_frame(symbol: str, scale: float = 1.0, volatile: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=160)
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
        "fundamental",
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
    assert result.calibrated_signals["kline"].branch_mode == "kline_symbol_parallel"
    assert result.calibrated_signals["fundamental"].branch_mode == "fundamental_symbol_parallel"
    assert result.final_strategy.provenance_summary["synthetic_symbols"] == []
    assert result.branch_results["kline"].metadata["effective_backend"] == "hybrid"
    assert result.branch_results["kline"].metadata["requested_backend"] == "hybrid"
    assert result.branch_results["kline"].metadata["llm_interface_reserved"] is True


def test_pipeline_branch_order_uses_fundamental_not_llm_debate():
    assert ParallelResearchPipeline.BRANCH_ORDER == [
        "kline",
        "quant",
        "fundamental",
        "intelligence",
        "macro",
    ]


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
    assert set(result.branch_results.keys()) == {"kline", "fundamental", "intelligence", "macro"}


def test_pipeline_degrades_timed_out_fundamental_branch_without_aborting(monkeypatch, tmp_path):
    symbol_frames = {
        "000001.SZ": _make_symbol_frame("000001.SZ", scale=1.0),
        "600519.SH": _make_symbol_frame("600519.SH", scale=1.2),
    }
    marker_file = tmp_path / "fundamental_branch_finished.txt"

    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: symbol_frames[symbol].copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: _FakeTerminal(signal="🟢", risk_level="低风险", recommendation="积极布局"),
    )

    def _slow_fundamental_branch(self, data_bundle):
        time.sleep(0.2)
        marker_file.write_text("finished", encoding="utf-8")
        return BranchResult(branch_name="fundamental", score=0.4, confidence=0.6)

    monkeypatch.setattr(ParallelResearchPipeline, "_run_fundamental_branch", _slow_fundamental_branch)

    pipeline = ParallelResearchPipeline(
        stock_pool=list(symbol_frames.keys()),
        market="CN",
        branch_timeout=0.05,
        verbose=False,
    )
    result = pipeline.run()

    assert result.branch_results["fundamental"].success is False
    assert result.branch_results["fundamental"].metadata["degraded_reason"] == "branch_timeout"
    assert "超时" in result.branch_results["fundamental"].explanation
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


def test_quant_investor_exports_mainline_entrypoint():
    unified = importlib.import_module("quant_investor")

    assert hasattr(unified, "QuantInvestor")
    assert hasattr(unified, "BranchResult")


def test_quant_investor_forwards_enable_quant(monkeypatch):
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

    monkeypatch.setattr("quant_investor.pipeline.mainline.ParallelResearchPipeline", _FakePipeline)

    investor = QuantInvestor(
        stock_pool=["000001.SZ"],
        market="CN",
        enable_quant=False,
        verbose=False,
    )
    investor.run()

    assert captured_kwargs["enable_quant"] is False


def test_branch_result_score_and_confidence_are_property_aliases():
    result = BranchResult(
        branch_name="fundamental",
        base_score=0.10,
        final_score=0.25,
        base_confidence=0.30,
        final_confidence=0.55,
    )

    assert result.score == pytest.approx(0.25)
    assert result.confidence == pytest.approx(0.55)

    result.score = 0.12
    result.confidence = 0.44

    assert result.final_score == pytest.approx(0.12)
    assert result.final_confidence == pytest.approx(0.44)


def test_v9_branch_tracker_archives_legacy_llm_history_without_remap(tmp_path, monkeypatch):
    history_path = tmp_path / "branch_ic_history.json"
    history_path.write_text(
        """
        {
          "kline": [0.1, 0.1, 0.1, 0.1, 0.1],
          "quant": [0.2, 0.2, 0.2, 0.2, 0.2],
          "llm_debate": [0.9, 0.9, 0.9, 0.9, 0.9]
        }
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(BranchPerformanceTracker, "HISTORY_PATH", str(history_path))

    tracker = BranchPerformanceTracker(
        default_weights=CURRENT_BRANCH_WEIGHTS,
        architecture_version="9.0.0-current",
        branch_schema_version="v9-fundamental-first-class",
    )

    assert tracker._history["fundamental"] == []
    assert tracker._archived_history["llm_debate"] == [0.9, 0.9, 0.9, 0.9, 0.9]


def test_branch_local_debate_degrades_without_llm_provider(monkeypatch):
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    pipeline = ParallelResearchPipeline(stock_pool=["000001.SZ"], market="CN", verbose=False)
    base_result = BranchResult(
        branch_name="quant",
        score=0.35,
        confidence=0.62,
        symbol_scores={"000001.SZ": 0.40},
        explanation="base quant result",
    )
    evidence = EvidencePacket(
        branch_name="quant",
        summary="quant evidence",
        top_symbols=["000001.SZ"],
        bull_points=["动量因子偏正"],
        used_features=["momentum_12_1"],
    )

    fused = pipeline._apply_branch_debate("quant", evidence, base_result)

    assert fused.final_score == pytest.approx(base_result.base_score)
    assert fused.final_confidence == pytest.approx(base_result.base_confidence)
    assert fused.debate_verdict.metadata["reason"] == "llm_provider_missing"


def test_branch_local_debate_score_adjustment_is_bounded():
    pipeline = ParallelResearchPipeline(stock_pool=["000001.SZ"], market="CN", verbose=False)
    pipeline.branch_debate_engine = BranchDebateEngine(
        enabled=True,
        responder=lambda **_: {
            "direction": "bearish",
            "confidence": 0.9,
            "score_adjustment": -9.0,
            "risk_flags": ["overheated"],
            "used_features": ["roe"],
            "hard_veto": False,
        },
    )
    base_result = BranchResult(
        branch_name="fundamental",
        score=0.55,
        confidence=0.70,
        symbol_scores={"000001.SZ": 0.60},
        explanation="base fundamental result",
    )
    evidence = EvidencePacket(
        branch_name="fundamental",
        summary="fundamental evidence",
        top_symbols=["000001.SZ"],
        bear_points=["估值过高"],
        used_features=["pe", "pb"],
    )

    fused = pipeline._apply_branch_debate("fundamental", evidence, base_result)

    assert fused.debate_verdict.score_adjustment >= -0.20
    assert fused.final_score >= 0.0
    assert fused.final_score == pytest.approx(0.35)


@pytest.mark.parametrize(
    ("branch_name", "expected_cap"),
    [
        ("kline", 0.10),
        ("quant", 0.10),
        ("fundamental", 0.20),
        ("intelligence", 0.15),
        ("macro", 0.10),
    ],
)
def test_branch_local_debate_caps_are_constant_controlled(branch_name, expected_cap):
    pipeline = ParallelResearchPipeline(stock_pool=["000001.SZ"], market="CN", verbose=False)
    pipeline.branch_debate_engine = BranchDebateEngine(
        enabled=True,
        responder=lambda **_: {
            "direction": "bullish",
            "confidence": 0.8,
            "score_adjustment": 9.0,
            "used_features": ["test_feature"],
        },
    )
    evidence = EvidencePacket(
        branch_name=branch_name,
        scope="market" if branch_name == "macro" else "symbol",
        top_symbols=[] if branch_name == "macro" else ["000001.SZ"],
        used_features=["test_feature"],
    )
    base_result = BranchResult(
        branch_name=branch_name,
        score=0.30,
        confidence=0.60,
        symbol_scores={} if branch_name == "macro" else {"000001.SZ": 0.40},
        explanation="base result",
    )

    fused = pipeline._apply_branch_debate(branch_name, evidence, base_result)

    assert fused.debate_verdict.score_adjustment <= expected_cap


def test_fundamental_branch_runs_without_document_data(monkeypatch):
    symbol = "000001.SZ"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_symbol_frame(symbol)},
        metadata={"end_date": "20260321"},
    )

    monkeypatch.setattr(
        EnhancedDataLayer,
        "get_document_semantic_snapshot",
        lambda self, symbol, as_of: self._doc_store.get_semantic_snapshot(symbol, as_of),
    )

    pipeline = ParallelResearchPipeline(stock_pool=[symbol], market="CN", verbose=False)
    result = pipeline._run_fundamental_branch(bundle)

    assert result.branch_name == "fundamental"
    assert symbol in result.symbol_scores
    assert result.data_quality["documents_missing_symbols"] == [symbol]
    assert result.success is True


def test_ensemble_regime_weights_do_not_contain_debate_key():
    for weights in EnsembleJudge.REGIME_WEIGHTS.values():
        assert "debate" not in weights
        assert "llm_debate" not in weights
        assert "fundamental" in weights


def test_intelligence_branch_no_longer_uses_financial_primary_scoring():
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

    pipeline = ParallelResearchPipeline(stock_pool=[symbol_a, symbol_b], market="CN", verbose=False)
    result_high = pipeline._run_intelligence_branch(bundle_high_quality)
    result_low = pipeline._run_intelligence_branch(bundle_low_quality)

    assert "financial_health_score" not in result_high.signals
    assert result_high.symbol_scores == pytest.approx(result_low.symbol_scores)


def test_macro_branch_debate_runs_once_market_level(monkeypatch):
    symbol_frames = {
        "000001.SZ": _make_symbol_frame("000001.SZ", scale=1.0),
        "600519.SH": _make_symbol_frame("600519.SH", scale=1.3),
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

    call_counter = {"macro": 0}
    original_evaluate = BranchDebateEngine.evaluate

    def _wrapped_evaluate(self, branch_name, evidence, base_result):
        if branch_name == "macro":
            call_counter["macro"] += 1
            assert evidence.scope == "market"
        return original_evaluate(self, branch_name, evidence, base_result)

    monkeypatch.setattr(BranchDebateEngine, "evaluate", _wrapped_evaluate)

    pipeline = ParallelResearchPipeline(stock_pool=list(symbol_frames.keys()), market="CN", verbose=False)
    pipeline.run()

    assert call_counter["macro"] == 1


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
        "fundamental": BranchResult("fundamental", score=-0.4, confidence=0.6, symbol_scores={s: -0.5 for s in symbols}),
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
                    "fundamental": {"score": 0.05},
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
                    "fundamental": {"score": 0.03},
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
