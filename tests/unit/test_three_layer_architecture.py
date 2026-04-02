from __future__ import annotations

import pandas as pd

from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.global_context.builder import GlobalContextBuilder
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline
from quant_investor.enhanced_data_layer import EnhancedDataLayer


def _make_frame(symbol: str, end: pd.Timestamp, periods: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    close = pd.Series(range(periods), dtype=float) * 0.1 + 10.0
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000,
            "amount": close * 1_000_000,
            "symbol": symbol,
            "market": "CN",
        }
    )
    df["forward_ret_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["return_20d"] = df["close"].pct_change(20)
    df["momentum_12_1"] = df["close"].pct_change(40) - df["close"].pct_change(10)
    df["volatility_20d"] = df["close"].pct_change().rolling(20).std()
    df["rsi_14"] = 55.0
    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["ma_bias_20"] = 0.01
    df["volume_ratio_20d"] = 1.05
    df["roe"] = 0.1
    df["gross_margin"] = 0.25
    df["profit_growth"] = 0.08
    df["revenue_growth"] = 0.05
    df["debt_ratio"] = 0.35
    df["pe"] = 14.0
    df["pb"] = 1.8
    df["ps"] = 1.4
    return df


def test_global_context_builder_uses_cache(tmp_path, monkeypatch):
    now = pd.Timestamp.now().normalize()
    symbol = "000001.SZ"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[symbol],
        symbol_data={symbol: _make_frame(symbol, now)},
        fundamentals={symbol: {}},
        event_data={symbol: []},
        sentiment_data={symbol: {}},
        metadata={
            "end_date": now.strftime("%Y%m%d"),
            "symbol_provenance": {
                symbol: {"data_source_status": "real", "is_synthetic": False}
            },
        },
    )
    builder = GlobalContextBuilder(cache_dir=tmp_path)
    calls = {"count": 0}
    original = GlobalContextBuilder._compute_context

    def _wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GlobalContextBuilder, "_compute_context", _wrapped)

    first = builder.build(data_bundle=bundle, phase1_context={"macro_regime": "bull"})
    second = builder.build(data_bundle=bundle, phase1_context={"macro_regime": "bull"})

    assert calls["count"] == 1
    assert first.cache_key == second.cache_key
    assert first.phase1_context["macro_regime"] == "bull"
    assert first.cache_path == second.cache_path
    assert first.cache_path


def test_parallel_pipeline_exposes_three_layer_artifacts(monkeypatch):
    now = pd.Timestamp.now().normalize()
    symbol_frames = {
        "000001.SZ": _make_frame("000001.SZ", now),
        "600519.SH": _make_frame("600519.SH", now),
    }

    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: symbol_frames[symbol].copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: type(
            "T",
            (),
            {
                "generate_risk_report": lambda self: type(
                    "R",
                    (),
                    {"overall_signal": "🟢", "overall_risk_level": "低风险", "recommendation": "积极布局"},
                )(),
            },
        )(),
    )

    pipeline = ParallelResearchPipeline(stock_pool=list(symbol_frames.keys()), market="CN", verbose=False)
    result = pipeline.run()

    assert result.global_context is not None
    assert result.global_context.cache_key
    assert set(result.symbol_research_packets) == set(symbol_frames)
    assert result.shortlist
    assert len(result.portfolio_decisions) <= len(result.shortlist)
    assert "global_context" in result.data_bundle.metadata
    assert "symbol_research_packets" in result.data_bundle.metadata
    assert "shortlist" in result.data_bundle.metadata
    assert "portfolio_decisions" in result.data_bundle.metadata
