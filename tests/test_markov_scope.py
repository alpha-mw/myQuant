from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from quant_investor.agent_protocol import BranchVerdict
from quant_investor.funnel.deterministic_funnel import FunnelOutput
from quant_investor.market.dag.context import _prepare_market_context
from quant_investor.market.read_result import MarketDataReadResult
from quant_investor.regime.scope import deterministic_symbol_sample


def _frame(symbol: str, *, direction: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range("2026-03-01", periods=90, freq="D")
    base = 20.0 + (sum(ord(ch) for ch in symbol) % 17)
    closes = [base * (1.0 + direction * 0.001 * idx) for idx in range(len(dates))]
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "trade_date": [date.strftime("%Y%m%d") for date in dates],
            "close": closes,
            "vol": [1000.0 + idx for idx in range(len(dates))],
            "amount": [10000.0 + idx * 10.0 for idx in range(len(dates))],
        }
    )


class ScopeReader:
    def __init__(
        self,
        *,
        requested_symbols: list[str],
        reference_symbols: list[str],
    ) -> None:
        self.requested_symbols = list(requested_symbols)
        self.reference_symbols = list(reference_symbols)
        self.frames = {
            symbol: _frame(symbol, direction=1.0)
            for symbol in sorted(set(requested_symbols + reference_symbols))
        }
        self.batch_calls: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return {"resolution_strategy": "fixture"}

    def list_symbols(self, universe_key: str = "full_us", category: str | None = None) -> list[str]:
        return list(self.reference_symbols)

    def read_symbol_frames(self, symbols: list[str], **kwargs: Any) -> dict[str, MarketDataReadResult]:
        self.batch_calls.append({"symbols": list(symbols), "kwargs": dict(kwargs)})
        return {
            symbol: MarketDataReadResult(
                frame=self.frames.get(symbol, pd.DataFrame()),
                symbol=symbol,
                universe_key=str(kwargs.get("universe_key", "")),
            )
            for symbol in symbols
        }

    def read_symbol_frame(self, symbol: str, universe_key: str = "") -> MarketDataReadResult:
        return MarketDataReadResult(
            frame=self.frames.get(symbol, pd.DataFrame()),
            symbol=symbol,
            universe_key=universe_key,
        )


class ScopeMacroAgent:
    def run(self, payload: dict[str, object]) -> BranchVerdict:
        return BranchVerdict(
            agent_name="macro",
            thesis="fixture macro",
            final_score=0.25,
            metadata={
                "regime": "趋势上涨",
                "target_gross_exposure": 0.70,
                "style_bias": "balanced",
            },
        )


class ScopeFunnel:
    def __init__(self, config: object) -> None:
        self.config = config

    def run(self, *, quant_result: object, global_context: object) -> FunnelOutput:
        symbols = list(global_context.universe_tiers.get("researchable", []))
        return FunnelOutput(
            candidates=symbols,
            candidate_scores={symbol: 0.5 for symbol in symbols},
            excluded_symbols={},
            funnel_metadata={},
        )


def _patch_branch_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    readiness = SimpleNamespace(status="ok")

    def _assess(**kwargs: Any) -> SimpleNamespace:
        symbols = list(kwargs.get("candidate_symbols", []) or [])
        return SimpleNamespace(
            blocked_symbols=[],
            quantifiable_universe=symbols,
            investable_universe=symbols,
            readiness={"macro": readiness},
            branch_data={},
            to_dict=lambda include_branch_data=False: {"status": "ok"},
        )

    monkeypatch.setattr("quant_investor.market.dag.context.assess_branch_data_readiness", _assess)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.write_branch_readiness_report",
        lambda report: {"status": "disabled"},
    )


def _run_scope_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    requested_symbols: list[str],
    reference_symbols: list[str],
    explicit_symbol_count: int,
    min_sample: int = 5,
    max_reference_symbols: int = 6,
) -> tuple[Any, ScopeReader]:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "production")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_PERSIST_ENABLED", False)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_MIN_MARKET_SAMPLE", min_sample)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_MAX_REFERENCE_SYMBOLS", max_reference_symbols)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_REFERENCE_UNIVERSE_US", "full_us")

    reader = ScopeReader(
        requested_symbols=requested_symbols,
        reference_symbols=reference_symbols,
    )
    state = _prepare_market_context(
        market="US",
        universe_key="full_us",
        selected_categories=["full_us"],
        symbols=requested_symbols,
        company_profile_map={symbol: {"industry": "Technology"} for symbol in requested_symbols},
        shared_reader=reader,
        scoped_data_snapshot={"local_latest_trade_date": "20260529", "freshness_mode": "stable"},
        download_stage=None,
        enable_agent_layer=False,
        agent_timeout=0.0,
        master_timeout=0.0,
        master_reasoning_effort="",
        branch_model_resolution=SimpleNamespace(
            primary_model="deterministic",
            fallback_model="",
            resolved_model="deterministic",
            fallback_used=False,
            fallback_reason="",
            metadata={},
        ),
        master_model_resolution=SimpleNamespace(
            primary_model="deterministic",
            fallback_model="",
            resolved_model="deterministic",
            fallback_used=False,
            fallback_reason="",
            metadata={},
        ),
        branch_candidate_models=[],
        master_candidate_models=[],
        company_name_map={symbol: symbol for symbol in requested_symbols},
        funnel_profile="classic",
        max_candidates=10,
        trend_windows=(5, 20, 60),
        volume_spike_threshold=1.2,
        breakout_distance_pct=0.05,
        sector_bucket_limit=0,
        macro_agent=ScopeMacroAgent(),
        funnel_cls=ScopeFunnel,
        provider_health_detector=lambda **kwargs: {},
        explicit_symbol_count=explicit_symbol_count,
        unsampled_symbol_count=len(requested_symbols),
        sampled=False,
    )
    return state, reader


def test_explicit_small_stock_pool_does_not_define_global_market_regime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state, reader = _run_scope_context(
        monkeypatch,
        tmp_path,
        requested_symbols=["NVDA", "AMD", "AVGO"],
        reference_symbols=["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA", "XOM"],
        explicit_symbol_count=3,
    )

    markov = state.global_context.metadata["markov_regime"]
    assert markov["regime_scope"] == "market_reference"
    assert markov["source_universe_key"] == "full_us"
    assert markov["requested_symbol_count"] == 3
    assert markov["explicit_symbol_count"] == 3
    assert markov["source_symbol_count"] == 6
    assert markov["production_eligible"] is True
    assert markov["status"] == "applied"
    assert reader.batch_calls[-1]["symbols"] == ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"]


def test_changing_requested_pool_uses_same_reference_regime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    reference = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA", "XOM"]
    state_one, _ = _run_scope_context(
        monkeypatch,
        tmp_path / "one",
        requested_symbols=["NVDA", "AMD", "AVGO"],
        reference_symbols=reference,
        explicit_symbol_count=3,
    )
    state_two, _ = _run_scope_context(
        monkeypatch,
        tmp_path / "two",
        requested_symbols=["JPM", "BAC", "WFC"],
        reference_symbols=reference,
        explicit_symbol_count=3,
    )

    markov_one = state_one.global_context.metadata["markov_regime"]
    markov_two = state_two.global_context.metadata["markov_regime"]
    assert markov_one["scope_key"] == markov_two["scope_key"]
    assert markov_one["dominant_regime"] == markov_two["dominant_regime"]
    assert markov_one["probabilities"] == pytest.approx(markov_two["probabilities"])


def test_missing_reference_data_fails_closed_and_preserves_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state, _ = _run_scope_context(
        monkeypatch,
        tmp_path,
        requested_symbols=["NVDA", "AMD", "AVGO"],
        reference_symbols=["AAPL", "MSFT", "GOOG"],
        explicit_symbol_count=3,
        min_sample=5,
        max_reference_symbols=5,
    )

    markov = state.global_context.metadata["markov_regime"]
    assert markov["production_eligible"] is False
    assert markov["regime_scope"] == "insufficient"
    assert markov["status"] == "not_applied_insufficient_market_scope"
    assert state.global_context.macro_regime == "趋势上涨"
    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(0.70)
    assert state.global_context.risk_budget["max_single_weight"] == pytest.approx(0.50)
    assert "turnover_cap" not in state.global_context.risk_budget
    assert any("below_min" in note or "insufficient" in note for note in markov["diagnostic_notes"])


def test_legitimate_full_market_run_remains_production_eligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    symbols = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"]
    state, reader = _run_scope_context(
        monkeypatch,
        tmp_path,
        requested_symbols=symbols,
        reference_symbols=symbols,
        explicit_symbol_count=0,
        min_sample=5,
        max_reference_symbols=6,
    )

    markov = state.global_context.metadata["markov_regime"]
    assert markov["regime_scope"] == "full_market"
    assert markov["production_eligible"] is True
    assert markov["source_symbol_count"] == 6
    assert len(reader.batch_calls) == 1


def test_deterministic_broad_market_sample_is_reproducible() -> None:
    first, sampled_first, unsampled_first = deterministic_symbol_sample(["MSFT", "AAPL", "NVDA"], 2)
    second, sampled_second, unsampled_second = deterministic_symbol_sample(["NVDA", "MSFT", "AAPL"], 2)

    assert first == ["AAPL", "MSFT"]
    assert first == second
    assert sampled_first is True
    assert sampled_second is True
    assert unsampled_first == unsampled_second == 3
