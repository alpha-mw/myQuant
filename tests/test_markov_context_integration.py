from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from quant_investor.agent_protocol import BranchVerdict
from quant_investor.funnel.deterministic_funnel import FunnelOutput
from quant_investor.market.dag.context import _prepare_market_context
from quant_investor.market.read_result import MarketDataReadResult
from quant_investor.regime.types import REGIME_RANGE_HIGH_VOL, REGIME_TREND_DOWN


class FakeReader:
    def __init__(self) -> None:
        self.frames = {
            "000001.SZ": _frame([10.0, 10.2, 10.5, 10.8]),
            "000002.SZ": _frame([10.0, 9.9, 9.8, 9.7]),
        }

    def snapshot(self) -> dict[str, object]:
        return {"resolution_strategy": "fixture"}

    def read_symbol_frames(self, symbols: list[str], **kwargs: object) -> dict[str, MarketDataReadResult]:
        return {
            symbol: MarketDataReadResult(
                frame=self.frames[symbol],
                symbol=symbol,
                universe_key=str(kwargs.get("universe_key", "")),
            )
            for symbol in symbols
        }

    def read_symbol_frame(self, symbol: str, universe_key: str = "") -> MarketDataReadResult:
        return MarketDataReadResult(
            frame=self.frames[symbol],
            symbol=symbol,
            universe_key=universe_key,
        )


class FakeMacroAgent:
    def run(self, payload: dict[str, object]) -> BranchVerdict:
        return BranchVerdict(
            agent_name="macro",
            thesis="fixture macro",
            final_score=0.20,
            metadata={
                "regime": "趋势上涨",
                "target_gross_exposure": 0.70,
                "style_bias": "balanced",
            },
        )


class FakeFunnel:
    def __init__(self, config: object) -> None:
        self.config = config

    def run(self, *, quant_result: object, global_context: object) -> FunnelOutput:
        return FunnelOutput(
            candidates=["000001.SZ", "000002.SZ"],
            candidate_scores={"000001.SZ": 0.8, "000002.SZ": 0.6},
            excluded_symbols={},
            funnel_metadata={},
        )


def _frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-06-20", periods=len(closes), freq="D"),
            "close": closes,
            "volume": [1000.0 + 50.0 * idx for idx in range(len(closes))],
            "amount": [10000.0 + 100.0 * idx for idx in range(len(closes))],
        }
    )


def _patch_branch_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    readiness = SimpleNamespace(status="ok")
    report = SimpleNamespace(
        blocked_symbols=[],
        quantifiable_universe=["000001.SZ", "000002.SZ"],
        investable_universe=["000001.SZ", "000002.SZ"],
        readiness={"macro": readiness},
        branch_data={},
        to_dict=lambda include_branch_data=False: {"status": "ok"},
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.assess_branch_data_readiness",
        lambda **kwargs: report,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.write_branch_readiness_report",
        lambda report: {"status": "disabled"},
    )


def _context_kwargs() -> dict[str, Any]:
    return {
        "market": "CN",
        "universe_key": "full_a",
        "selected_categories": ["full_a"],
        "symbols": ["000001.SZ", "000002.SZ"],
        "company_profile_map": {
            "000001.SZ": {"industry": "Banking"},
            "000002.SZ": {"industry": "Technology"},
        },
        "shared_reader": FakeReader(),
        "scoped_data_snapshot": {"local_latest_trade_date": "20260625", "freshness_mode": "stable"},
        "download_stage": None,
        "enable_agent_layer": False,
        "agent_timeout": 0.0,
        "master_timeout": 0.0,
        "master_reasoning_effort": "",
        "branch_model_resolution": SimpleNamespace(
            primary_model="deterministic",
            fallback_model="",
            resolved_model="deterministic",
            fallback_used=False,
            fallback_reason="",
            metadata={},
        ),
        "master_model_resolution": SimpleNamespace(
            primary_model="deterministic",
            fallback_model="",
            resolved_model="deterministic",
            fallback_used=False,
            fallback_reason="",
            metadata={},
        ),
        "branch_candidate_models": [],
        "master_candidate_models": [],
        "company_name_map": {"000001.SZ": "平安银行", "000002.SZ": "万科A"},
        "funnel_profile": "classic",
        "max_candidates": 10,
        "trend_windows": (2, 3),
        "volume_spike_threshold": 1.2,
        "breakout_distance_pct": 0.05,
        "sector_bucket_limit": 0,
        "macro_agent": FakeMacroAgent(),
        "funnel_cls": FakeFunnel,
        "provider_health_detector": lambda **kwargs: {},
    }


def test_markov_context_shadow_mode_adds_metadata_without_capping(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "shadow")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_PERSIST_ENABLED", False)

    state = _prepare_market_context(**_context_kwargs())

    markov = state.global_context.metadata["markov_regime"]
    assert markov["dominant_regime"]
    assert state.global_context.regime_params["markov"] == markov
    assert state.global_context.metadata["macro_agent_regime"] == "趋势上涨"
    assert state.global_context.macro_regime == "趋势上涨"
    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(0.70)
    assert state.global_context.risk_budget["max_single_weight"] == pytest.approx(0.12)
    assert "markov_dominant_regime" not in state.global_context.risk_budget


def test_markov_context_production_mode_caps_risk_budget(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "production")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_PERSIST_ENABLED", False)

    state = _prepare_market_context(**_context_kwargs())
    markov = state.global_context.metadata["markov_regime"]

    assert state.global_context.macro_regime == markov["dominant_regime"]
    assert state.global_context.risk_budget["markov_regime_enabled"] is True
    assert state.global_context.risk_budget["markov_execution_target"] == "production"
    assert state.global_context.risk_budget["markov_dominant_regime"] == markov["dominant_regime"]
    assert state.global_context.risk_budget["target_exposure"] <= 0.70
    assert state.global_context.risk_budget["max_single_weight"] <= 0.12
    assert "turnover_cap" in state.global_context.risk_budget
    assert markov["dominant_regime"] in {
        "趋势上涨",
        "震荡低波",
        REGIME_RANGE_HIGH_VOL,
        REGIME_TREND_DOWN,
        "未知",
    }
