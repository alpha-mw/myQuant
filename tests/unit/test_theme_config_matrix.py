from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_investor.agent_protocol import ActionLabel, BranchVerdict, GlobalContext, ICDecision
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.agents.risk_guard import RiskGuard
from quant_investor.funnel.deterministic_funnel import DeterministicFunnel, FunnelConfig
from quant_investor.market.dag.theme_context import persist_theme_rotation_snapshot
from quant_investor.themes.calibration import build_theme_calibration_report
from quant_investor.themes.replay import build_theme_calibration_dataset


SYMBOL = "000001.SZ"


def _theme_rotation(
    *,
    symbol: str = SYMBOL,
    symbol_score: float = 0.95,
    phase: str = "confirmed_rotation",
    risk_flags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "theme_rotation.v1",
        "status": "success",
        "symbol_scores": {symbol: symbol_score},
        "symbol_primary_theme": {symbol: "industry::semiconductor"},
        "symbol_phase": {symbol: phase},
        "symbol_risk_flags": {symbol: list(risk_flags or [])},
        "theme_scores": {
            "industry::semiconductor": {
                "theme_name": "Semiconductor",
                "score": 88.0,
                "confidence": 0.74,
                "member_count": 12,
                "phase": phase,
            }
        },
    }


def _market_state() -> dict[str, float]:
    return {
        "momentum_strength": 0.72,
        "breakout_readiness": 0.68,
        "volume_confirmation": 0.55,
        "trend_stability": 0.62,
        "distance_from_high_pct": 0.018,
        "fake_breakout_risk": 0.08,
        "max_drawdown_pct": 0.04,
        "return_20d": 0.09,
    }


def _global_context(metadata: dict[str, Any] | None = None) -> GlobalContext:
    payload = dict(metadata or {})
    payload.setdefault("symbol_market_state", {SYMBOL: _market_state()})
    return GlobalContext(
        market="CN",
        universe_key="full_a",
        universe_symbols=[SYMBOL],
        universe_tiers={"researchable": [SYMBOL]},
        industry_map={SYMBOL: "Semiconductor"},
        liquidity_filter={"liquidity_scores": {SYMBOL: 1.0}},
        data_quality_quarantine=[],
        metadata=payload,
    )


def _risk_payload(constraints: dict[str, object] | None = None) -> dict[str, object]:
    base_constraints: dict[str, object] = {
        "gross_exposure_cap": 0.90,
        "max_weight": 0.20,
        "risk_flags": [],
    }
    base_constraints.update(constraints or {})
    return {
        "branch_verdicts": {
            "quant": BranchVerdict(
                agent_name="quant",
                thesis="quant ok",
                symbol=SYMBOL,
                final_score=0.35,
                final_confidence=0.75,
            )
        },
        "macro_verdict": BranchVerdict(agent_name="macro", thesis="macro stable"),
        "portfolio_state": {
            "candidate_symbols": [SYMBOL],
            "current_weights": {},
        },
        "constraints": base_constraints,
    }


def _ic_decision(symbol: str, *, score: float = 0.8, confidence: float = 0.9) -> ICDecision:
    return ICDecision(
        symbol=symbol,
        action=ActionLabel.BUY,
        final_score=score,
        final_confidence=confidence,
        selected_symbols=[symbol],
        metadata={
            "symbol_scores": {symbol: score},
            "symbol_confidences": {symbol: confidence},
            "symbol_calibrated_confidences": {symbol: confidence},
            "symbol_momentum_strengths": {symbol: score},
            "symbol_actions": {symbol: ActionLabel.BUY},
            "symbol_modes": {symbol: "target"},
        },
    )


def _portfolio_payload(
    symbols: list[str],
    *,
    sectors: dict[str, str] | None = None,
    risk_limits: dict[str, object] | None = None,
) -> dict[str, object]:
    sector_map = sectors or {symbol: "Tech" for symbol in symbols}
    base_risk_limits: dict[str, object] = {
        "gross_exposure_cap": 1.0,
        "max_weight": 0.6,
        "position_limits": {symbol: 0.6 for symbol in symbols},
        "blocked_symbols": [],
        "sector_caps": {},
    }
    base_risk_limits.update(risk_limits or {})
    return {
        "ic_decisions": [_ic_decision(symbol) for symbol in symbols],
        "macro_verdict": BranchVerdict(metadata={"target_gross_exposure": 1.0}),
        "risk_limits": base_risk_limits,
        "existing_portfolio": {"current_weights": {}},
        "tradability_snapshot": {
            symbol: {
                "is_tradable": True,
                "sector": sector_map[symbol],
                "liquidity_score": 1.0,
            }
            for symbol in symbols
        },
    }


def _theme_exposure_map(theme_by_symbol: dict[str, str]) -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "primary_theme_id": theme_id,
            "primary_theme_name": theme_id.rsplit("::", maxsplit=1)[-1],
            "phase": "confirmed_rotation",
            "symbol_score": 0.9,
            "risk_flags": [],
        }
        for symbol, theme_id in theme_by_symbol.items()
    }


def _theme_total(plan: Any, theme_id: str) -> float:
    exposure_map = plan.metadata["theme_exposure_map"]
    return sum(
        weight
        for symbol, weight in plan.target_weights.items()
        if exposure_map.get(symbol, {}).get("primary_theme_id") == theme_id
    )


def _frame(closes: list[float], *, start: str = "2026-06-18") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range(start, periods=len(closes), freq="D"),
            "close": closes,
        }
    )


def test_funnel_boost_disabled_is_noop() -> None:
    funnel = DeterministicFunnel(
        FunnelConfig(theme_boost_enabled=False, theme_boost_cap=0.05)
    )

    boost, metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context({"theme_rotation": _theme_rotation()}),
    )

    assert boost == 0.0
    assert metadata["enabled"] is False
    assert metadata["reason"] == "disabled"


def test_funnel_boost_enabled_affects_only_funnel_score() -> None:
    context = _global_context({"theme_rotation": _theme_rotation()})
    disabled_score = DeterministicFunnel(
        FunnelConfig(
            profile="momentum_leader",
            theme_boost_enabled=False,
            theme_boost_cap=0.05,
        )
    )._momentum_leader_score(
        symbol=SYMBOL,
        quant_scores={SYMBOL: 0.42},
        global_context=context,
    )
    enabled_score = DeterministicFunnel(
        FunnelConfig(
            profile="momentum_leader",
            theme_boost_enabled=True,
            theme_boost_cap=0.05,
        )
    )._momentum_leader_score(
        symbol=SYMBOL,
        quant_scores={SYMBOL: 0.42},
        global_context=context,
    )

    difference = enabled_score - disabled_score
    assert difference > 0.0
    assert difference <= 0.05 + 1e-9


def test_risk_guard_theme_disabled_is_noop() -> None:
    risk_guard = RiskGuard()

    baseline = risk_guard.run(_risk_payload())
    disabled = risk_guard.run(
        _risk_payload(
            {
                "theme_risk_guard_enabled": False,
                "theme_risk_flags": ["theme_overextended"],
                "theme_action_cap": "hold",
                "theme_gross_exposure_cap": 0.60,
                "theme_position_limits": {SYMBOL: 0.10},
            }
        )
    )

    assert disabled.action_cap == baseline.action_cap
    assert disabled.gross_exposure_cap == pytest.approx(baseline.gross_exposure_cap)
    assert disabled.max_weight == pytest.approx(baseline.max_weight)
    assert disabled.position_limits == baseline.position_limits


def test_risk_guard_theme_enabled_changes_only_risk_limits() -> None:
    baseline = RiskGuard().run(_risk_payload())
    enabled = RiskGuard().run(
        _risk_payload(
            {
                "theme_risk_guard_enabled": True,
                "theme_risk_flags": ["theme_overextended"],
                "theme_action_cap": "hold",
                "theme_gross_exposure_cap": 0.60,
                "theme_position_limits": {SYMBOL: 0.10},
            }
        )
    )

    assert baseline.hard_veto is False
    assert enabled.hard_veto is False
    assert enabled.action_cap == ActionLabel.HOLD
    assert enabled.action_cap != baseline.action_cap
    assert enabled.gross_exposure_cap == pytest.approx(0.60)
    assert enabled.gross_exposure_cap < baseline.gross_exposure_cap
    assert enabled.position_limits[SYMBOL] == pytest.approx(0.10)
    assert enabled.position_limits[SYMBOL] < baseline.position_limits[SYMBOL]
    assert enabled.blocked_symbols == baseline.blocked_symbols


def test_portfolio_theme_cap_disabled_is_noop() -> None:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    constructor = PortfolioConstructor()

    baseline = constructor.run(_portfolio_payload(symbols))
    disabled = constructor.run(
        _portfolio_payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": False,
                "theme_exposure_map": _theme_exposure_map(
                    {symbol: "industry::AI" for symbol in symbols}
                ),
                "theme_caps": {"industry::AI": 0.10},
            },
        )
    )

    assert disabled.target_weights == baseline.target_weights
    assert disabled.target_gross_exposure == baseline.target_gross_exposure
    assert disabled.position_limits == baseline.position_limits


def test_portfolio_theme_cap_enabled_reduces_only_overcap_theme() -> None:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    theme_map = {
        "000001.SZ": "industry::AI",
        "000002.SZ": "industry::AI",
        "000003.SZ": "industry::Robotics",
    }
    constructor = PortfolioConstructor()

    baseline = constructor.run(_portfolio_payload(symbols))
    capped = constructor.run(
        _portfolio_payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": True,
                "theme_exposure_map": _theme_exposure_map(theme_map),
                "theme_caps": {"industry::AI": 0.25, "industry::Robotics": 0.80},
            },
        )
    )

    assert _theme_total(capped, "industry::AI") <= 0.25 + 1e-6
    assert capped.target_weights["000003.SZ"] <= baseline.target_weights["000003.SZ"] + 1e-6
    assert capped.target_gross_exposure <= baseline.target_gross_exposure + 1e-6


def test_snapshot_enabled_writes_but_does_not_modify_theme_rotation(tmp_path: Path) -> None:
    theme_rotation = _theme_rotation()
    original = copy.deepcopy(theme_rotation)

    status = persist_theme_rotation_snapshot(
        theme_rotation=theme_rotation,
        enabled=True,
        root_dir=tmp_path,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        run_id="matrix",
    )

    assert status["status"] == "success"
    assert Path(status["path"]).exists()
    assert theme_rotation == original
    payload = json.loads(Path(status["path"]).read_text(encoding="utf-8"))
    assert payload["theme_rotation"] == original


def test_replay_and_calibration_are_explicit_only() -> None:
    snapshot = {
        "snapshot_schema_version": "theme_snapshot.v1",
        "market": "CN",
        "universe_key": "full_a",
        "as_of": "20260618",
        "theme_rotation": _theme_rotation(),
    }
    frames = {SYMBOL: _frame([10.0, 10.4, 10.8, 11.0, 11.2, 11.5, 11.8])}

    dataset = build_theme_calibration_dataset(
        snapshots=[snapshot],
        frames=frames,
        horizons=(1, 3, 5),
        benchmark_horizons=(5,),
    )
    report = build_theme_calibration_report(dataset, min_sample=1)

    assert dataset.metadata["no_llm"] is True
    assert dataset.metadata["no_network"] is True
    assert report.metadata["offline_only"] is True
    assert report.metadata["no_llm"] is True
    assert report.metadata["no_network"] is True
    assert report.record_count == 1
    assert report.available_count == 1
