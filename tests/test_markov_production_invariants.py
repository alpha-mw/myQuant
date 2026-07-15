from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from quant_investor.agent_protocol import ActionLabel, BranchVerdict, GlobalContext, ICDecision
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.agents.risk_guard import RiskGuard
from quant_investor.funnel.deterministic_funnel import FunnelOutput
from quant_investor.market.dag.context import _prepare_market_context
from quant_investor.market.read_result import MarketDataReadResult
from quant_investor.regime.types import REGIME_RANGE_HIGH_VOL, REGIME_TREND_UP, RegimeSignal


def _frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-04-01", periods=len(closes), freq="D"),
            "close": closes,
            "vol": [1000.0 + idx for idx in range(len(closes))],
            "amount": [10000.0 + idx for idx in range(len(closes))],
        }
    )


class InvariantReader:
    def __init__(self) -> None:
        self.frames = {
            "000001.SZ": _frame([10.0 + idx * 0.03 for idx in range(90)]),
            "000002.SZ": _frame([12.0 + idx * 0.02 for idx in range(90)]),
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
        return MarketDataReadResult(frame=self.frames[symbol], symbol=symbol, universe_key=universe_key)


class InvariantMacroAgent:
    def run(self, payload: dict[str, object]) -> BranchVerdict:
        return BranchVerdict(
            agent_name="macro",
            thesis="fixture macro",
            final_score=0.30,
            metadata={
                "regime": "趋势上涨",
                "target_gross_exposure": 0.70,
                "style_bias": "balanced",
            },
        )


class InvariantFunnel:
    def __init__(self, config: object) -> None:
        self.config = config

    def run(self, *, quant_result: object, global_context: GlobalContext) -> FunnelOutput:
        return FunnelOutput(
            candidates=["000001.SZ", "000002.SZ"],
            candidate_scores={"000001.SZ": 0.8, "000002.SZ": 0.6},
            excluded_symbols={},
            funnel_metadata={},
        )


def _patch_branch_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "quant_investor.market.dag.context.load_macro_record",
        lambda **kwargs: (
            {
                "trade_date": "2026-06-25",
                "macro_score": 0.2,
                "liquidity_score": 0.4,
                "volatility_percentile": 45.0,
                "policy_signal": "neutral",
                "source": "tushare_primary",
                "source_priority": "tushare_primary",
                "pit_status": "market_point_in_time",
                "fetched_at": "2026-06-25T08:00:00+00:00",
            },
            {
                "generation_id": "fixture-macro-generation",
                "parquet_sha256": "a" * 64,
                "generation_manifest_sha256": "b" * 64,
                "source": "tushare_primary",
                "source_priority": "tushare_primary",
                "provider_status": "verified_provider_snapshot",
                "production_eligible": True,
            },
        ),
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


def _context_with_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    gross_cap: float,
    max_weight: float,
    turnover_cap: float | None = None,
    dominant_regime: str = REGIME_TREND_UP,
) -> Any:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", True)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "production")
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_PERSIST_ENABLED", False)
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
                dominant_regime=dominant_regime,
                probabilities={
                    "趋势上涨": 0.75,
                    "震荡低波": 0.10,
                    REGIME_RANGE_HIGH_VOL: 0.10,
                    "趋势下跌": 0.03,
                    "未知": 0.02,
                },
                transition_matrix={},
                confidence=0.75,
                transition_risk=0.13,
                risk_on_score=0.80,
                volatility_score=0.10,
                pressure_score=0.05,
                suggested_gross_exposure_cap=gross_cap,
                suggested_max_single_weight=max_weight,
                turnover_cap=turnover_cap,
                feature_snapshot={},
                diagnostic_notes=[],
                regime_scope="full_market",
                scope_key="CN:full_market:full_a:symbols_2",
                base_universe_key="full_a",
                source_universe_key="full_a",
                requested_symbol_count=2,
                source_symbol_count=2,
                explicit_symbol_count=0,
                unsampled_symbol_count=2,
                sampled=False,
                production_eligible=True,
            )

    monkeypatch.setattr("quant_investor.market.dag.context.MarkovRegimeEngine", FakeEngine)
    return _prepare_market_context(
        market="CN",
        universe_key="full_a",
        selected_categories=["full_a"],
        symbols=["000001.SZ", "000002.SZ"],
        company_profile_map={
            "000001.SZ": {"industry": "Banking"},
            "000002.SZ": {"industry": "Technology"},
        },
        shared_reader=InvariantReader(),
        scoped_data_snapshot={"local_latest_trade_date": "20260625", "freshness_mode": "stable"},
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
        company_name_map={"000001.SZ": "平安银行", "000002.SZ": "万科A"},
        funnel_profile="classic",
        max_candidates=10,
        trend_windows=(5, 20, 60),
        volume_spike_threshold=1.2,
        breakout_distance_pct=0.05,
        sector_bucket_limit=0,
        macro_agent=InvariantMacroAgent(),
        funnel_cls=InvariantFunnel,
        provider_health_detector=lambda **kwargs: {},
    )


@pytest.mark.parametrize(
    ("suggested_gross", "suggested_weight", "expected_gross", "expected_weight"),
    [
        (0.95, 0.75, 0.70, 0.50),
        (0.40, 0.08, 0.40, 0.08),
        (0.70, 0.50, 0.70, 0.50),
    ],
)
def test_markov_can_only_tighten_baseline_caps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    suggested_gross: float,
    suggested_weight: float,
    expected_gross: float,
    expected_weight: float,
) -> None:
    state = _context_with_signal(
        monkeypatch,
        tmp_path,
        gross_cap=suggested_gross,
        max_weight=suggested_weight,
    )

    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(expected_gross)
    assert state.global_context.risk_budget["max_single_weight"] == pytest.approx(expected_weight)
    assert state.global_context.risk_budget["target_exposure"] <= state.global_context.risk_budget["baseline_target_exposure"]
    assert state.global_context.risk_budget["max_single_weight"] <= state.global_context.risk_budget["baseline_max_single_weight"]


def test_markov_turnover_cap_is_forwarded_only_when_applied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state = _context_with_signal(
        monkeypatch,
        tmp_path,
        gross_cap=0.45,
        max_weight=0.50,
        turnover_cap=0.30,
        dominant_regime=REGIME_RANGE_HIGH_VOL,
    )

    assert state.global_context.risk_budget["turnover_cap"] == pytest.approx(0.30)
    assert state.global_context.metadata["markov_regime"]["applied_turnover_cap"] == pytest.approx(0.30)


def test_risk_guard_keeps_more_restrictive_macro_cap() -> None:
    decision = RiskGuard().run(
        {
            "branch_verdicts": {
                "quant": BranchVerdict(agent_name="quant", thesis="ok"),
            },
            "macro_verdict": BranchVerdict(
                final_score=0.10,
                metadata={"target_gross_exposure": 0.35},
            ),
            "portfolio_state": {"candidate_symbols": ["000001.SZ"], "current_weights": {}},
            "constraints": {"gross_exposure_cap": 0.60, "max_weight": 0.20},
        }
    )

    assert decision.gross_exposure_cap == pytest.approx(0.35)
    assert decision.max_weight == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("risk_gross", "macro_gross", "expected"),
    [(0.25, 0.60, 0.25), (0.70, 0.30, 0.30)],
)
def test_portfolio_constructor_uses_minimum_applicable_gross_cap(
    risk_gross: float,
    macro_gross: float,
    expected: float,
) -> None:
    plan = PortfolioConstructor().run(
        {
            "ic_decisions": [
                ICDecision(
                    symbol="000001.SZ",
                    selected_symbols=["000001.SZ"],
                    action=ActionLabel.BUY,
                    final_score=0.90,
                    final_confidence=0.80,
                )
            ],
            "macro_verdict": BranchVerdict(metadata={"target_gross_exposure": macro_gross}),
            "risk_limits": {
                "gross_exposure_cap": risk_gross,
                "max_weight": 0.50,
                "position_limits": {"000001.SZ": 0.50},
                "blocked_symbols": [],
                "sector_caps": {},
            },
            "existing_portfolio": {"current_weights": {}},
            "tradability_snapshot": {
                "000001.SZ": {"tradable": True, "liquidity_score": 1.0, "sector": "Banking"}
            },
        }
    )

    assert plan.metadata["applied_gross_cap"] == pytest.approx(expected)
    assert plan.target_gross_exposure <= expected


def test_disabled_markov_preserves_legacy_macro_and_risk_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_branch_readiness(monkeypatch)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_ENABLED", False)
    monkeypatch.setattr("quant_investor.market.dag.context.config.MARKOV_REGIME_EXECUTION_TARGET", "production")

    class ExplodingEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("disabled Markov must not instantiate the engine")

    monkeypatch.setattr("quant_investor.market.dag.context.MarkovRegimeEngine", ExplodingEngine)
    state = _prepare_market_context(
        market="CN",
        universe_key="full_a",
        selected_categories=["full_a"],
        symbols=["000001.SZ", "000002.SZ"],
        company_profile_map={},
        shared_reader=InvariantReader(),
        scoped_data_snapshot={"local_latest_trade_date": "20260625", "freshness_mode": "stable"},
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
        company_name_map={},
        funnel_profile="classic",
        max_candidates=10,
        trend_windows=(5, 20, 60),
        volume_spike_threshold=1.2,
        breakout_distance_pct=0.05,
        sector_bucket_limit=0,
        macro_agent=InvariantMacroAgent(),
        funnel_cls=InvariantFunnel,
        provider_health_detector=lambda **kwargs: {},
    )

    assert state.global_context.macro_regime == "趋势上涨"
    assert state.global_context.risk_budget["target_exposure"] == pytest.approx(0.70)
    assert state.global_context.risk_budget["max_single_weight"] == pytest.approx(0.50)
    assert state.global_context.metadata["markov_regime"]["status"] == "disabled"
