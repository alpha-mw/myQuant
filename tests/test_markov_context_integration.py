from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import quant_investor.market.dag.context as context_module
from quant_investor.agent_protocol import BranchVerdict
from quant_investor.branch_contracts import BranchResult
from quant_investor.factors.governance import FactorLifecycleState, FactorRecord
from quant_investor.factors.runtime import MinedFactorRegistry, MinedFactorScorer
from quant_investor.funnel.deterministic_funnel import FunnelOutput
from quant_investor.market.dag.context import _prepare_market_context
from quant_investor.market.read_result import MarketDataReadResult
from quant_investor.market.runtime_profile import MarketRuntimeProfiler
from quant_investor.regime.types import REGIME_RANGE_HIGH_VOL, REGIME_TREND_DOWN
from tests.helpers.macro_fixture import make_v15_controls


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
    macro_record = {
        "trade_date": "2026-06-25",
        "macro_score": 0.2,
        "liquidity_score": 0.4,
        "volatility_percentile": 45.0,
        "policy_signal": "neutral",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "pit_status": "market_point_in_time",
        "fetched_at": "2026-06-25T08:00:00+00:00",
    }
    macro_manifest = {
        "generation_id": "fixture-macro-generation",
        "parquet_sha256": "a" * 64,
        "generation_manifest_sha256": "b" * 64,
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "provider_status": "verified_provider_snapshot",
        "production_eligible": True,
        "v15_controls": make_v15_controls(),
    }
    monkeypatch.setattr(
        "quant_investor.market.dag.context.load_macro_record",
        lambda **kwargs: (dict(macro_record), dict(macro_manifest)),
    )
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


def test_canonical_macro_is_loaded_once_and_drives_macro_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_investor.agents.macro_agent import MacroAgent

    records = [
        {
            "trade_date": "2026-06-25",
            "macro_score": -0.6,
            "liquidity_score": 0.2,
            "volatility_percentile": 50.0,
            "policy_signal": "neutral",
            "source": "tushare_primary",
            "source_priority": "tushare_primary",
            "pit_status": "market_point_in_time",
            "fetched_at": "2026-06-25T08:00:00+00:00",
        },
        {
            "trade_date": "2026-06-25",
            "macro_score": 0.6,
            "liquidity_score": 0.2,
            "volatility_percentile": 50.0,
            "policy_signal": "neutral",
            "source": "tushare_primary",
            "source_priority": "tushare_primary",
            "pit_status": "market_point_in_time",
            "fetched_at": "2026-06-25T08:00:00+00:00",
        },
    ]
    manifests = [
        {
            "generation_id": f"macro-g{index}",
            "parquet_sha256": str(index) * 64,
            "generation_manifest_sha256": chr(96 + index) * 64,
            "source": "tushare_primary",
            "source_priority": "tushare_primary",
            "provider_status": "verified_provider_snapshot",
            "production_eligible": True,
            "v15_controls": make_v15_controls(
                macro_score=float(records[index - 1]["macro_score"]),
                liquidity_score=float(records[index - 1]["liquidity_score"]),
                volatility_percentile=float(
                    records[index - 1]["volatility_percentile"]
                ),
                policy_signal=str(records[index - 1]["policy_signal"]),
            ),
        }
        for index in (1, 2)
    ]
    load_calls = 0
    pinned_inputs: list[tuple[object, object]] = []

    def load_once(**_kwargs: object):
        nonlocal load_calls
        record = records[load_calls]
        manifest = manifests[load_calls]
        load_calls += 1
        return record, manifest

    def assess_pinned(**kwargs: object) -> SimpleNamespace:
        record = kwargs["pinned_macro_record"]
        manifest = kwargs["pinned_macro_manifest"]
        pinned_inputs.append((record, manifest))
        symbols = list(kwargs.get("candidate_symbols", []) or [])
        return SimpleNamespace(
            blocked_symbols=[],
            quantifiable_universe=symbols,
            investable_universe=symbols,
            readiness={"macro": SimpleNamespace(status="pass")},
            branch_data={"macro_data": record},
            to_dict=lambda include_branch_data=False: {"status": "pass"},
        )

    monkeypatch.setattr(context_module, "load_macro_record", load_once)
    monkeypatch.setattr(
        context_module,
        "assess_branch_data_readiness",
        assess_pinned,
    )
    monkeypatch.setattr(
        context_module,
        "write_branch_readiness_report",
        lambda report: {"status": "disabled"},
    )
    monkeypatch.setattr(context_module.config, "MARKOV_REGIME_ENABLED", False)

    states = []
    for _ in range(2):
        kwargs = _context_kwargs()
        kwargs["macro_agent"] = MacroAgent()
        states.append(_prepare_market_context(**kwargs))

    assert load_calls == 2
    assert pinned_inputs == list(zip(records, manifests))
    for index, (record, manifest) in enumerate(pinned_inputs):
        assert record is records[index]
        assert manifest is manifests[index]
    assert states[0].macro_verdict.final_score < states[1].macro_verdict.final_score
    for index, state in enumerate(states, start=1):
        assert state.market_snapshot["macro_score"] == records[index - 1][
            "macro_score"
        ]
        assert state.market_snapshot["liquidity_score"] == 0.2
        assert state.macro_verdict.metadata["decision_authorized"] is True
        assert state.macro_verdict.metadata["canonical_macro_generation"] == {
            "generation_id": f"macro-g{index}",
            "parquet_sha256": str(index) * 64,
            "generation_manifest_sha256": chr(96 + index) * 64,
        }
        assert state.global_context.metadata["canonical_macro_generation"] == (
            state.macro_verdict.metadata["canonical_macro_generation"]
        )


def test_blocked_canonical_macro_skips_agent_and_markov_and_clears_holding_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    load_calls = 0
    load_as_of: list[object] = []
    pinned_inputs: list[tuple[object, object]] = []
    history_path = tmp_path / "history.jsonl"
    blocked_record = {
        "trade_date": "2026-06-24",
        "macro_score": 0.9,
        "liquidity_score": 0.8,
        "volatility_percentile": 10.0,
        "policy_signal": "supportive",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "pit_status": "market_point_in_time",
        "fetched_at": "2026-06-24T08:00:00+00:00",
    }
    blocked_manifest = {
        "generation_id": "stale-macro-generation",
        "parquet_sha256": "a" * 64,
        "generation_manifest_sha256": "b" * 64,
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "provider_status": "verified_provider_snapshot",
        "production_eligible": True,
    }

    def blocked_load(**_kwargs: object):
        nonlocal load_calls
        load_calls += 1
        load_as_of.append(_kwargs.get("as_of"))
        return blocked_record, blocked_manifest

    def blocked_assessment(**kwargs: object) -> SimpleNamespace:
        pinned_inputs.append(
            (
                kwargs["pinned_macro_record"],
                kwargs["pinned_macro_manifest"],
            )
        )
        return SimpleNamespace(
            blocked_symbols=["000001.SZ"],
            quantifiable_universe=["000001.SZ"],
            investable_universe=[],
            readiness={"macro": SimpleNamespace(status="block")},
            branch_data={"macro_data": {}},
            to_dict=lambda include_branch_data=False: {"status": "block"},
        )

    class ExplodingMacroAgent:
        def run(self, payload: object) -> BranchVerdict:
            raise AssertionError("blocked canonical Macro must bypass MacroAgent")

    class ExplodingMarkovEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("blocked canonical Macro must bypass Markov")

    monkeypatch.setattr(context_module, "load_macro_record", blocked_load)
    monkeypatch.setattr(
        context_module,
        "assess_branch_data_readiness",
        blocked_assessment,
    )
    monkeypatch.setattr(
        context_module,
        "write_branch_readiness_report",
        lambda report: {"status": "disabled"},
    )
    monkeypatch.setattr(
        context_module,
        "MarkovRegimeEngine",
        ExplodingMarkovEngine,
    )
    monkeypatch.setattr(context_module.config, "MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr(
        context_module.config,
        "MARKOV_REGIME_PERSIST_ENABLED",
        True,
    )
    monkeypatch.setattr(
        context_module.config,
        "MARKOV_REGIME_HISTORY_PATH",
        str(history_path),
    )
    kwargs = _context_kwargs()
    kwargs.update(
        {
            "symbols": ["000001.SZ"],
            "company_profile_map": {
                "000001.SZ": {"industry": "Banking"},
            },
            "company_name_map": {"000001.SZ": "平安银行"},
            "macro_agent": ExplodingMacroAgent(),
            "funnel_profile": "momentum_leader",
            "download_stage": {
                "completeness_after": {},
                "completeness_before": {},
            },
            "recall_context": {"holding_symbol": "000001.SZ"},
        }
    )

    state = _prepare_market_context(**kwargs)

    assert load_calls == 1
    assert load_as_of == ["20260625"]
    assert pinned_inputs == [(blocked_record, blocked_manifest)]
    assert state.macro_verdict.status.value == "vetoed"
    assert state.macro_verdict.direction.value == "neutral"
    assert state.macro_verdict.action.value == "hold"
    assert state.macro_verdict.final_score == 0.0
    assert state.macro_verdict.metadata["decision_authorized"] is False
    assert state.market_snapshot["macro_score"] == 0.0
    assert state.market_snapshot["liquidity_score"] == 0.0
    assert state.market_snapshot["decision_authorized"] is False
    assert state.global_context.macro_data == {}
    assert state.candidate_symbols == []
    assert state.funnel_output.candidates == []
    assert state.global_context.universe_tiers["shortlistable"] == []
    assert state.global_context.metadata["decision_authorized"] is False
    assert state.global_context.metadata["holding_review_funnel_override"] is False
    assert (
        state.global_context.metadata["holding_review_funnel_override_requested"]
        is True
    )
    assert (
        state.global_context.metadata["holding_review_branch_readiness_override"]
        is False
    )
    assert (
        state.global_context.metadata[
            "holding_review_branch_readiness_override_requested"
        ]
        is True
    )
    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(
        0.55
    )
    assert state.global_context.risk_budget["target_exposure"] == (
        state.global_context.risk_budget["baseline_target_exposure"]
    )
    assert state.global_context.risk_budget["max_single_weight"] == (
        state.global_context.risk_budget["baseline_max_single_weight"]
    )
    assert "turnover_cap" not in state.global_context.risk_budget
    markov = state.global_context.metadata["markov_regime"]
    assert markov["enabled"] is False
    assert markov["status"] == "blocked_by_canonical_macro_readiness"
    assert markov["execution_mode"] == "not_run"
    assert history_path.exists() is False

    missing_date_kwargs = _context_kwargs()
    missing_date_kwargs.update(
        {
            "scoped_data_snapshot": {"freshness_mode": "stable"},
            "macro_agent": ExplodingMacroAgent(),
        }
    )
    missing_date_state = _prepare_market_context(**missing_date_kwargs)

    assert load_calls == 2
    assert load_as_of == ["20260625", ""]
    assert missing_date_state.macro_verdict.status.value == "vetoed"
    assert "macro_as_of_missing" in missing_date_state.macro_verdict.metadata[
        "blockers"
    ]
    assert missing_date_state.global_context.metadata[
        "decision_authorized"
    ] is False
    assert missing_date_state.candidate_symbols == []
    assert history_path.exists() is False


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


def test_runtime_plan_filters_four_short_histories_before_all_quant_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr(context_module.config, "MARKOV_REGIME_ENABLED", False)
    symbols = [f"S{index:02d}" for index in range(24)]
    dates = pd.date_range(end="2026-06-25", periods=8, freq="D")
    frames = {
        symbol: pd.DataFrame(
            {
                "ts_code": [symbol] * 8,
                "trade_date": dates,
                "close": [10.0 + index + step / 10.0 for step in range(8)],
                "adj_close": [10.0 + index + step / 10.0 for step in range(8)],
                "vol": [1_000.0 + step for step in range(8)],
                "volume": [1_000.0 + step for step in range(8)],
                "amount": [10_000.0 + step for step in range(8)],
            }
        )
        for index, symbol in enumerate(symbols)
    }
    short_symbols = symbols[-4:]
    for symbol in short_symbols:
        frames[symbol] = frames[symbol].tail(5).copy()

    class PlanReader(FakeReader):
        def __init__(self) -> None:
            self.frames = frames

    records = [
        FactorRecord(
            name="pv_low_dollar_volume_5d",
            version="v1",
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            category="liquidity",
            implementation="price_volume:pv_low_dollar_volume_5d",
            weight=0.5,
            direction=1.0,
        ),
        FactorRecord(
            name="pv_volume_stability_5d",
            version="v1",
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            category="liquidity",
            implementation="price_volume:pv_volume_stability_5d",
            weight=0.5,
            direction=1.0,
        ),
    ]
    contracts = {
        records[0].name: {
            "required_columns": ["trade_date", "amount"],
            "lookback_rows": 5,
            "gate2_min_coverage_rate": 1.0,
            "min_cross_section": 20,
        },
        records[1].name: {
            "required_columns": ["trade_date", "vol"],
            "lookback_rows": 8,
            "gate2_min_coverage_rate": 1.0,
            "min_cross_section": 20,
        },
    }
    runtime_status = {
        "status": "ready",
        "factor_mode": "governed_mined_factors",
        "confidence_multiplier": 1.0,
        "production_eligible": True,
        "blockers": [],
        "factor_runtime_contracts": contracts,
    }
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records(records))
    runtime_contract_calls = 0

    def runtime_contract():
        nonlocal runtime_contract_calls
        runtime_contract_calls += 1
        return records, runtime_status

    monkeypatch.setattr(scorer, "_runtime_contract", runtime_contract)
    monkeypatch.setattr(context_module, "MinedFactorScorer", lambda: scorer)
    captured: dict[str, list[str]] = {}

    def capture_context(**kwargs):
        captured["context"] = list(kwargs["symbols"])
        return None, ["fixture_context_blocked"]

    original_cross_section = context_module._build_cross_section_quant

    def capture_cross_section(runtime_frames, **kwargs):
        captured["cross_section"] = list(runtime_frames)
        return original_cross_section(runtime_frames, **kwargs)

    def capture_quant(*, frames, **kwargs):
        captured["quant"] = list(frames)
        assert kwargs["production_runtime_plan"].eligible_symbols == tuple(
            sorted(frames)
        )
        assert kwargs["scorer"] is scorer
        return (
            BranchResult(
                branch_name="quant",
                symbol_scores={symbol: 0.0 for symbol in frames},
            ),
            None,
        )

    monkeypatch.setattr(
        context_module,
        "_build_production_evaluation_context",
        capture_context,
    )
    monkeypatch.setattr(context_module, "_build_cross_section_quant", capture_cross_section)
    monkeypatch.setattr(
        context_module,
        "_build_quant_branch_result_with_validation",
        capture_quant,
    )
    kwargs = _context_kwargs()
    profiler = MarketRuntimeProfiler(market="CN", universe="full_a")
    kwargs.update(
        {
            "symbols": symbols,
            "shared_reader": PlanReader(),
            "company_profile_map": {},
            "company_name_map": {},
            "runtime_profiler": profiler,
        }
    )

    state = _prepare_market_context(**kwargs)

    eligible = symbols[:-4]
    assert runtime_contract_calls == 1
    assert state.researchable_symbols == eligible
    assert state.quarantined_symbols == short_symbols
    assert captured == {
        "context": eligible,
        "cross_section": eligible,
        "quant": eligible,
    }
    assert set(state.global_context.metadata["quant_contract_eligibility_blockers"]) == set(short_symbols)
    tradability_stage = next(
        item
        for item in profiler.stages
        if item["name"] == "dag_tradability_snapshot"
    )
    assert tradability_stage["metadata"]["researchable_count"] == 20
    assert tradability_stage["metadata"]["quarantined_count"] == 4
    assert tradability_stage["metadata"]["issue_count"] == 4
    for symbol in short_symbols:
        issue = next(
            item
            for item in state.data_quality_issues
            if item.symbol == symbol
            and item.issue_type == "production_factor_runtime_ineligible"
        )
        assert issue.metadata["blockers"] == [
            f"factor_required_lookback_missing:{records[1].name}:{symbol}"
        ]


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
                regime_scope="full_market",
                source_symbol_count=2,
                unsampled_symbol_count=2,
                sampled=False,
                production_eligible=True,
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
                regime_scope="full_market",
                source_symbol_count=2,
                unsampled_symbol_count=2,
                sampled=False,
                production_eligible=True,
            )

    monkeypatch.setattr("quant_investor.market.dag.context.MarkovRegimeEngine", PermissiveEngine)

    state = _prepare_market_context(**_context_kwargs())

    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(0.70)
    assert state.global_context.risk_budget["max_single_weight"] == pytest.approx(0.50)
