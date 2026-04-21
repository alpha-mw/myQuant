from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_investor.agents.kline_agent import KlineAgent
from quant_investor.branch_contracts import BranchResult


def _make_symbol_data() -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="D"),
            "open": [10.0 + i * 0.1 for i in range(40)],
            "high": [10.2 + i * 0.1 for i in range(40)],
            "low": [9.8 + i * 0.1 for i in range(40)],
            "close": [10.1 + i * 0.1 for i in range(40)],
            "volume": [1000 + i for i in range(40)],
            "amount": [10000 + i * 10 for i in range(40)],
        }
    )
    return {"000001.SZ": frame, "600519.SH": frame.copy()}


def _make_branch_result(branch_name: str, score: float, confidence: float, runtime_mode: str) -> BranchResult:
    return BranchResult(
        branch_name=branch_name,
        final_score=score,
        final_confidence=confidence,
        symbol_scores={"000001.SZ": score, "600519.SH": score},
        signals={
            "predicted_return": {"000001.SZ": score, "600519.SH": score},
            "trend_regime": {"000001.SZ": "上行", "600519.SH": "上行"},
            "model_runtime_mode": runtime_mode,
        },
        metadata={
            "reliability": 0.7,
            "model_runtime_mode": runtime_mode,
        },
        conclusion=f"{branch_name} conclusion",
        diagnostic_notes=[],
    )


def test_hybrid_engine_attaches_structured_trace_for_native_models(monkeypatch):
    from quant_investor.kline_backends.hybrid_engine import KlineHybridEngine
    import quant_investor.kline_backends.hybrid_engine as hybrid_module
    from quant_investor.kline_backends.evaluator import KLineEvaluationOutput

    @dataclass
    class _FakeChronosBackend:
        runtime_mode: str = "vendor_native"

        def __init__(self, **_kwargs):
            self.runtime_mode = "vendor_native"

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("chronos", 0.4, 0.65, self.runtime_mode)

    @dataclass
    class _FakeKronosBackend:
        runtime_mode: str = "vendor_native"

        def __init__(self, **_kwargs):
            self.runtime_mode = "vendor_native"

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("kline", 0.6, 0.75, self.runtime_mode)

    class _FakeEvaluator:
        name = "fake_evaluator"

        def evaluate(self, evaluation_input):
            kronos_result = evaluation_input.kronos_result
            chronos_result = evaluation_input.chronos_result
            return KLineEvaluationOutput(
                score=0.5,
                confidence=0.8,
                symbol_scores={"000001.SZ": 0.5, "600519.SH": 0.5},
                predicted_returns={"000001.SZ": 0.1, "600519.SH": 0.1},
                regimes={"000001.SZ": "上行", "600519.SH": "上行"},
                explanation=f"{kronos_result.branch_name}+{chronos_result.branch_name}",
                metadata={"evaluator_name": self.name, "reliability": 0.82},
            )

    monkeypatch.setattr(hybrid_module, "KronosBackend", _FakeKronosBackend)
    monkeypatch.setattr(hybrid_module, "ChronosBackend", _FakeChronosBackend)
    monkeypatch.setattr(hybrid_module, "get_kline_evaluator", lambda _name=None: _FakeEvaluator())

    engine = KlineHybridEngine()
    signal = engine.predict(_make_symbol_data(), ["000001.SZ", "600519.SH"])

    assert signal.branch_result.branch_name == "kline"
    trace = signal.execution_trace
    assert trace.kronos_status == "vendor_native"
    assert trace.chronos_status == "vendor_native"
    assert trace.fallback_mode == ""
    assert trace.hybrid_mode == "dual_model"
    assert trace.degradation_reason == ""
    assert signal.branch_result.metadata["kline_execution_trace"]["kronos_status"] == "vendor_native"
    assert signal.branch_result.metadata["model_mode"] == "hybrid"


def test_hybrid_engine_marks_statistical_only_when_both_models_fallback(monkeypatch):
    from quant_investor.kline_backends.hybrid_engine import KlineHybridEngine
    import quant_investor.kline_backends.hybrid_engine as hybrid_module
    from quant_investor.kline_backends.evaluator import KLineEvaluationOutput

    class _FakeChronosBackend:
        def __init__(self, **_kwargs):
            self.runtime_mode = "statistical_fallback"

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("chronos", 0.1, 0.3, self.runtime_mode)

    class _FakeKronosBackend:
        def __init__(self, **_kwargs):
            self.runtime_mode = "statistical_fallback"

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("kline", 0.15, 0.35, self.runtime_mode)

    class _FakeEvaluator:
        name = "fake_evaluator"

        def evaluate(self, evaluation_input):
            return KLineEvaluationOutput(
                score=0.12,
                confidence=0.32,
                symbol_scores={"000001.SZ": 0.12, "600519.SH": 0.12},
                predicted_returns={"000001.SZ": 0.02, "600519.SH": 0.02},
                regimes={"000001.SZ": "震荡", "600519.SH": "震荡"},
                explanation="degraded",
                metadata={"evaluator_name": self.name, "reliability": 0.44},
            )

    monkeypatch.setattr(hybrid_module, "KronosBackend", _FakeKronosBackend)
    monkeypatch.setattr(hybrid_module, "ChronosBackend", _FakeChronosBackend)
    monkeypatch.setattr(hybrid_module, "get_kline_evaluator", lambda _name=None: _FakeEvaluator())

    engine = KlineHybridEngine()
    signal = engine.predict(_make_symbol_data(), ["000001.SZ", "600519.SH"])

    trace = signal.execution_trace
    assert trace.kronos_status == "statistical_fallback"
    assert trace.chronos_status == "statistical_fallback"
    assert trace.fallback_mode == "statistical_only"
    assert "统计" in trace.degradation_reason
    assert signal.branch_result.metadata["kline_execution_trace"]["fallback_mode"] == "statistical_only"
    assert signal.branch_result.metadata["model_runtime_mode"] == "statistical_only"


def test_hybrid_engine_marks_structured_degraded_when_one_model_missing(monkeypatch):
    from quant_investor.kline_backends.hybrid_engine import KlineHybridEngine
    import quant_investor.kline_backends.hybrid_engine as hybrid_module
    from quant_investor.kline_backends.evaluator import KLineEvaluationOutput

    class _FakeChronosBackend:
        def __init__(self, **_kwargs):
            self.runtime_mode = "statistical_fallback"

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("chronos", 0.1, 0.3, self.runtime_mode)

    class _FakeKronosBackend:
        def __init__(self, **_kwargs):
            self.runtime_mode = "vendor_native"

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("kline", 0.2, 0.4, self.runtime_mode)

    class _FakeEvaluator:
        name = "fake_evaluator"

        def evaluate(self, evaluation_input):
            return KLineEvaluationOutput(
                score=0.18,
                confidence=0.38,
                symbol_scores={"000001.SZ": 0.18, "600519.SH": 0.18},
                predicted_returns={"000001.SZ": 0.03, "600519.SH": 0.03},
                regimes={"000001.SZ": "震荡", "600519.SH": "震荡"},
                explanation="structured degraded",
                metadata={"evaluator_name": self.name, "reliability": 0.55},
            )

    monkeypatch.setattr(hybrid_module, "KronosBackend", _FakeKronosBackend)
    monkeypatch.setattr(hybrid_module, "ChronosBackend", _FakeChronosBackend)
    monkeypatch.setattr(hybrid_module, "get_kline_evaluator", lambda _name=None: _FakeEvaluator())

    engine = KlineHybridEngine()
    signal = engine.predict(_make_symbol_data(), ["000001.SZ", "600519.SH"])

    trace = signal.execution_trace
    assert trace.kronos_status == "vendor_native"
    assert trace.chronos_status == "statistical_fallback"
    assert trace.fallback_mode == "structured_degraded"
    assert "结构化降级" in trace.degradation_reason
    assert signal.branch_result.metadata["kline_execution_trace"]["fallback_mode"] == "structured_degraded"


def test_hybrid_engine_shortlists_full_market_input_before_deep_models(monkeypatch):
    from quant_investor.kline_backends.hybrid_engine import FULL_MARKET_THRESHOLD, KlineHybridEngine
    import quant_investor.kline_backends.hybrid_engine as hybrid_module
    from quant_investor.kline_backends.evaluator import KLineEvaluationOutput

    symbol_data = _make_symbol_data()
    template = next(iter(symbol_data.values()))
    for idx in range(FULL_MARKET_THRESHOLD + 1):
        symbol = f"00{idx:04d}.SZ"
        symbol_data[symbol] = template.copy()
    full_pool_size = len(symbol_data)

    seen: dict[str, int] = {}

    class _FakeHeuristicBackend:
        def predict(self, symbol_data, stock_pool):
            return BranchResult(
                branch_name="kline",
                final_score=0.12,
                final_confidence=0.5,
                symbol_scores={symbol: float(len(stock_pool) - i) for i, symbol in enumerate(stock_pool)},
                signals={"predicted_return": {}, "trend_regime": {}},
                metadata={"model_runtime_mode": "heuristic"},
            )

    class _FakeKronosBackend:
        def __init__(self, **_kwargs):
            self.runtime_mode = "vendor_native"

        def predict(self, symbol_data, stock_pool):
            seen["kronos_pool_size"] = len(stock_pool)
            return _make_branch_result("kline", 0.25, 0.55, self.runtime_mode)

    def _fake_run_chronos_with_timeout(self, symbol_data, stock_pool, fallback_result):
        seen["chronos_pool_size"] = len(stock_pool)
        assert len(stock_pool) < full_pool_size
        return _make_branch_result("chronos", 0.22, 0.52, "vendor_native")

    class _FakeEvaluator:
        name = "fake_evaluator"

        def evaluate(self, evaluation_input):
            return KLineEvaluationOutput(
                score=0.2,
                confidence=0.5,
                symbol_scores={symbol: 0.2 for symbol in evaluation_input.stock_pool},
                predicted_returns={symbol: 0.03 for symbol in evaluation_input.stock_pool},
                regimes={symbol: "震荡" for symbol in evaluation_input.stock_pool},
                explanation="full market",
                metadata={"evaluator_name": self.name, "reliability": 0.6},
            )

    monkeypatch.setattr(hybrid_module, "HeuristicBackend", _FakeHeuristicBackend)
    monkeypatch.setattr(hybrid_module, "KronosBackend", _FakeKronosBackend)
    monkeypatch.setattr(KlineHybridEngine, "_run_chronos_with_timeout", _fake_run_chronos_with_timeout)
    monkeypatch.setattr(hybrid_module, "get_kline_evaluator", lambda _name=None: _FakeEvaluator())

    engine = KlineHybridEngine()
    signal = engine.predict(symbol_data, list(symbol_data))

    assert seen["kronos_pool_size"] < len(symbol_data)
    assert seen["chronos_pool_size"] < len(symbol_data)
    assert signal.execution_trace.screening_mode == "heuristic_shortlist"
    assert signal.execution_trace.full_market_mode is True
    assert len(signal.execution_trace.shortlist) <= 12


def test_hybrid_engine_trace_includes_latency_and_availability(monkeypatch):
    from quant_investor.kline_backends.hybrid_engine import KlineHybridEngine
    import quant_investor.kline_backends.hybrid_engine as hybrid_module
    from quant_investor.kline_backends.evaluator import KLineEvaluationOutput

    class _FakeChronosBackend:
        def __init__(self, **_kwargs):
            self.runtime_mode = "vendor_native"

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("chronos", 0.4, 0.65, self.runtime_mode)

    class _FakeKronosBackend:
        def __init__(self, **_kwargs):
            self.runtime_mode = "vendor_native"

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("kline", 0.6, 0.75, self.runtime_mode)

    class _FakeEvaluator:
        name = "fake_evaluator"

        def evaluate(self, evaluation_input):
            return KLineEvaluationOutput(
                score=0.5,
                confidence=0.8,
                symbol_scores={"000001.SZ": 0.5, "600519.SH": 0.5},
                predicted_returns={"000001.SZ": 0.1, "600519.SH": 0.1},
                regimes={"000001.SZ": "上行", "600519.SH": "上行"},
                explanation="ok",
                metadata={"evaluator_name": self.name, "reliability": 0.82},
            )

    monkeypatch.setattr(hybrid_module, "KronosBackend", _FakeKronosBackend)
    monkeypatch.setattr(hybrid_module, "ChronosBackend", _FakeChronosBackend)
    monkeypatch.setattr(hybrid_module, "get_kline_evaluator", lambda _name=None: _FakeEvaluator())

    engine = KlineHybridEngine()
    signal = engine.predict(_make_symbol_data(), ["000001.SZ", "600519.SH"])
    trace = signal.execution_trace

    assert trace.kronos_available is True
    assert trace.chronos_available is True
    assert trace.kronos_latency_ms >= 0.0
    assert trace.chronos_latency_ms >= 0.0
    assert trace.fusion_method == "weighted_average"
    assert trace.symbol_count == 2

    trace_dict = trace.to_dict()
    assert "kronos_latency_ms" in trace_dict
    assert "chronos_latency_ms" in trace_dict
    assert "fusion_method" in trace_dict
    assert "kronos_available" in trace_dict


def test_hybrid_engine_health_check(monkeypatch):
    from quant_investor.kline_backends.hybrid_engine import KlineHybridEngine
    import quant_investor.kline_backends.hybrid_engine as hybrid_module

    class _FakeKronosBackend:
        vendor_native_available = True

        def __init__(self, **_kwargs):
            pass

        def predict(self, symbol_data, stock_pool):
            return _make_branch_result("kline", 0.3, 0.5, "vendor_native")

    class _FakeChronosBackend:
        vendor_native_available = False

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(hybrid_module, "KronosBackend", _FakeKronosBackend)
    monkeypatch.setattr(hybrid_module, "ChronosBackend", _FakeChronosBackend)
    monkeypatch.setattr(hybrid_module, "get_kline_evaluator", lambda _name=None: None)

    engine = KlineHybridEngine()
    health = engine.health_check()
    assert health["kronos_available"] is True
    assert health["chronos_available"] is False
    assert health["mode"] == "kronos_only_degraded"


def test_kline_agent_preserves_hybrid_trace_metadata(monkeypatch):
    class _FakeBackend:
        def predict(self, symbol_data, stock_pool):
            return BranchResult(
                branch_name="kline",
                final_score=0.3,
                final_confidence=0.6,
                symbol_scores={"000001.SZ": 0.3},
                signals={"predicted_return": {"000001.SZ": 0.04}, "trend_regime": {"000001.SZ": "上行"}},
                metadata={
                    "kline_execution_trace": {
                        "kronos_status": "vendor_native",
                        "chronos_status": "vendor_native",
                        "hybrid_mode": "dual_model",
                    },
                    "model_runtime_mode": "hybrid",
                },
                conclusion="hybrid conclusion",
            )

    monkeypatch.setattr("quant_investor.kline_backends.get_backend", lambda *_args, **_kwargs: _FakeBackend())

    agent = KlineAgent()
    verdict = agent.run(
        {
            "mode": "shortlist",
            "symbol_data": _make_symbol_data(),
            "stock_pool": ["000001.SZ", "600519.SH"],
        }
    )

    assert verdict.metadata["kline_execution_trace"]["hybrid_mode"] == "dual_model"
    assert verdict.metadata["model_runtime_mode"] == "hybrid"
