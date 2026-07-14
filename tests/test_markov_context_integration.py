from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import quant_investor.market.dag.context as context_module
from quant_investor.agent_protocol import BranchVerdict
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.deterministic_funnel import FunnelOutput
from quant_investor.market.dag.context import _prepare_market_context
from quant_investor.market.read_result import MarketDataReadResult
from quant_investor.regime.types import REGIME_RANGE_HIGH_VOL, REGIME_TREND_DOWN


class FakeReader:
    def __init__(self) -> None:
        self.frames = {
            "000001.SZ": _frame("000001.SZ", [10.0, 10.2, 10.5, 10.8]),
            "000002.SZ": _frame("000002.SZ", [10.0, 9.9, 9.8, 9.7]),
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


def _frame(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [symbol] * len(closes),
            "trade_date": pd.date_range(
                end="2026-06-25",
                periods=len(closes),
                freq="D",
            ),
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


def test_markov_context_default_production_caps_risk_budget_and_preserves_macro_agent_regime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "production")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_PERSIST_ENABLED", False)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_MIN_MARKET_SAMPLE", 2)

    state = _prepare_market_context(**_context_kwargs())

    markov = state.global_context.metadata["markov_regime"]
    assert markov["dominant_regime"]
    assert state.global_context.regime_params["markov"] == markov
    assert state.global_context.metadata["macro_agent_regime"] == "趋势上涨"
    assert state.global_context.macro_regime == markov["dominant_regime"]
    assert state.global_context.risk_budget["markov_regime_enabled"] is True
    assert state.global_context.risk_budget["markov_execution_mode"] == "production"
    assert state.global_context.risk_budget["markov_dominant_regime"] == markov["dominant_regime"]
    assert state.global_context.risk_budget["target_exposure"] <= 0.70
    assert state.global_context.risk_budget["max_single_weight"] <= 0.50
    assert state.global_context.risk_budget["markov_applied_gross_exposure_cap"] == state.global_context.risk_budget["target_exposure"]
    assert state.global_context.risk_budget["markov_applied_max_single_weight"] == state.global_context.risk_budget["max_single_weight"]
    assert markov["execution_mode"] == "production"


def test_markov_context_shadow_target_is_normalized_to_production(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "shadow")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_PERSIST_ENABLED", False)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_MIN_MARKET_SAMPLE", 2)

    state = _prepare_market_context(**_context_kwargs())
    markov = state.global_context.metadata["markov_regime"]

    assert state.global_context.macro_regime == markov["dominant_regime"]
    assert markov["execution_mode"] == "production"
    assert state.global_context.risk_budget["markov_execution_mode"] == "production"
    assert "markov_shadow_deprecated_normalized_to_production" in markov["diagnostic_notes"]
    assert markov["dominant_regime"] in {
        "趋势上涨",
        "震荡低波",
        REGIME_RANGE_HIGH_VOL,
        REGIME_TREND_DOWN,
        "未知",
    }


def test_markov_context_disabled_preserves_legacy_macro_and_risk_budget(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", False)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "production")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_MIN_MARKET_SAMPLE", 2)

    class ExplodingEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("MarkovRegimeEngine should be bypassed when disabled")

    monkeypatch.setattr("quant_investor.market.dag.context.MarkovRegimeEngine", ExplodingEngine)

    state = _prepare_market_context(**_context_kwargs())

    assert state.global_context.macro_regime == "趋势上涨"
    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(0.70)
    assert state.global_context.risk_budget["max_single_weight"] == pytest.approx(0.50)
    assert "turnover_cap" not in state.global_context.risk_budget
    assert "markov_dominant_regime" not in state.global_context.risk_budget
    markov = state.global_context.metadata["markov_regime"]
    assert markov["enabled"] is False
    assert markov["status"] == "disabled"
    assert markov["applied_target_exposure"] == pytest.approx(0.70)
    assert markov["applied_max_single_weight"] == pytest.approx(0.50)


def test_quant_and_cross_section_receive_only_researchable_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr(context_module.config, "MARKOV_REGIME_ENABLED", False)
    kwargs = _context_kwargs()
    reader = kwargs["shared_reader"]
    assert isinstance(reader, FakeReader)
    reader.frames["000002.SZ"] = reader.frames["000002.SZ"].iloc[:-1].copy()
    captured: dict[str, list[str]] = {}
    original_cross_section = context_module._build_cross_section_quant

    def capture_cross_section(frames, **kwargs):
        captured["cross_section"] = list(frames)
        return original_cross_section(frames, **kwargs)

    def capture_quant(frames, **_kwargs):
        captured["quant"] = list(frames)
        return (
            BranchResult(
                branch_name="quant",
                final_score=0.0,
                final_confidence=0.0,
                symbol_scores={symbol: 0.0 for symbol in frames},
                metadata={
                    "governance_status": "governance_blocked",
                    "factor_mode": "governance_blocked",
                    "production_eligible": False,
                },
            ),
            None,
        )

    monkeypatch.setattr(
        context_module,
        "_build_cross_section_quant",
        capture_cross_section,
    )
    monkeypatch.setattr(
        context_module,
        "_build_quant_branch_result_with_validation",
        capture_quant,
    )

    state = _prepare_market_context(**kwargs)

    assert state.researchable_symbols == ["000001.SZ"]
    assert state.quarantined_symbols == ["000002.SZ"]
    assert captured["cross_section"] == ["000001.SZ"]
    assert captured["quant"] == ["000001.SZ"]
    stale_issue = next(
        issue
        for issue in state.data_quality_issues
        if issue.symbol == "000002.SZ"
    )
    assert stale_issue.issue_type == "production_frame_terminal_date_mismatch"
    assert stale_issue.severity == "error"
    assert stale_issue.metadata["evaluation_as_of"] == "20260625"
    assert state.global_context.metadata["quant_frame_validation_blockers"] == {
        "000002.SZ": "production_frame_terminal_date_mismatch:000002.SZ",
    }


def test_markov_context_forwards_turnover_cap_when_signal_sets_it(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from quant_investor.regime.types import RegimeSignal

    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "production")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_MIN_MARKET_SAMPLE", 2)

    class FakeEngine:
        execution_target = "production"
        enabled = True

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> RegimeSignal:
            return RegimeSignal(
                as_of="20260625",
                market="CN",
                universe_key="full_a",
                dominant_regime=REGIME_RANGE_HIGH_VOL,
                probabilities={
                    "趋势上涨": 0.10,
                    "震荡低波": 0.15,
                    REGIME_RANGE_HIGH_VOL: 0.55,
                    REGIME_TREND_DOWN: 0.15,
                    "未知": 0.05,
                },
                transition_matrix={},
                confidence=0.55,
                transition_risk=0.70,
                risk_on_score=0.20,
                volatility_score=0.85,
                pressure_score=0.75,
                suggested_gross_exposure_cap=0.40,
                suggested_max_single_weight=0.08,
                turnover_cap=0.30,
                feature_snapshot={},
                diagnostic_notes=[],
            )

    monkeypatch.setattr("quant_investor.market.dag.context.MarkovRegimeEngine", FakeEngine)

    state = _prepare_market_context(**_context_kwargs())

    assert state.global_context.macro_regime == REGIME_RANGE_HIGH_VOL
    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(0.40)
    assert state.global_context.risk_budget["max_single_weight"] == pytest.approx(0.08)
    assert state.global_context.risk_budget["turnover_cap"] == pytest.approx(0.30)
    assert state.global_context.risk_budget["markov_turnover_cap"] == pytest.approx(0.30)


def test_markov_context_never_increases_baseline_risk_budget(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from quant_investor.regime.types import RegimeSignal

    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "production")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))

    class PermissiveEngine:
        execution_target = "production"
        enabled = True

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> RegimeSignal:
            return RegimeSignal(
                as_of="20260625",
                market="CN",
                universe_key="full_a",
                dominant_regime="趋势上涨",
                probabilities={
                    "趋势上涨": 0.80,
                    "震荡低波": 0.10,
                    REGIME_RANGE_HIGH_VOL: 0.05,
                    REGIME_TREND_DOWN: 0.03,
                    "未知": 0.02,
                },
                transition_matrix={},
                confidence=0.80,
                transition_risk=0.08,
                risk_on_score=0.90,
                volatility_score=0.10,
                pressure_score=0.05,
                suggested_gross_exposure_cap=0.95,
                suggested_max_single_weight=0.75,
                turnover_cap=None,
                feature_snapshot={},
                diagnostic_notes=[],
            )

    monkeypatch.setattr("quant_investor.market.dag.context.MarkovRegimeEngine", PermissiveEngine)

    state = _prepare_market_context(**_context_kwargs())

    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(0.70)
    assert state.global_context.risk_budget["max_single_weight"] == pytest.approx(0.50)
