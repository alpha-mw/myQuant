from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_investor.agent_protocol import BranchVerdict
from quant_investor.regime.features import build_regime_feature_snapshot


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"close": [10.0, 10.2, 10.4], "volume": [100, 120, 140]})


def test_build_regime_feature_snapshot_extracts_cross_section_and_tradability() -> None:
    snapshot = build_regime_feature_snapshot(
        market="CN",
        universe_key="full_a",
        as_of="20260625",
        frames={"000001.SZ": _frame(), "000002.SZ": _frame()},
        tradability_snapshot={
            "000001.SZ": {
                "market_state": {
                    "momentum_strength": 0.70,
                    "breakout_readiness": 0.60,
                    "fake_breakout_risk": 0.20,
                    "max_drawdown_pct": 0.08,
                    "liquidity_score": 0.90,
                    "volume_confirmation": 0.55,
                }
            },
            "000002.SZ": {
                "market_state": {
                    "momentum_strength": 0.30,
                    "breakout_readiness": 0.40,
                    "fake_breakout_risk": 0.70,
                    "max_drawdown_pct": 0.16,
                    "liquidity_score": 0.60,
                    "volume_confirmation": 0.20,
                }
            },
        },
        cross_section_quant={
            "average_return": 0.012,
            "average_volatility": 0.024,
            "breadth": 0.68,
            "sample_count": 2,
        },
        macro_verdict=BranchVerdict(
            final_score=0.35,
            metadata={"target_gross_exposure": 0.62},
        ),
    )

    assert snapshot.average_return == pytest.approx(0.012)
    assert snapshot.average_volatility == pytest.approx(0.024)
    assert snapshot.breadth == pytest.approx(0.68)
    assert snapshot.momentum_share == pytest.approx(0.5)
    assert snapshot.breakout_ready_share == pytest.approx(0.5)
    assert snapshot.fake_breakout_share == pytest.approx(0.5)
    assert snapshot.median_drawdown == pytest.approx(0.12)
    assert snapshot.average_liquidity == pytest.approx(0.75)
    assert snapshot.average_volume_confirmation == pytest.approx(0.375)
    assert snapshot.macro_score == pytest.approx(0.35)
    assert snapshot.macro_target_gross_exposure == pytest.approx(0.62)
    assert snapshot.sample_count == 2
    json.dumps(snapshot.to_dict(), ensure_ascii=False)


def test_build_regime_feature_snapshot_empty_input_is_unknown_safe() -> None:
    snapshot = build_regime_feature_snapshot(
        market="CN",
        universe_key="full_a",
        as_of="20260625",
        frames={},
        tradability_snapshot={},
        cross_section_quant={},
        macro_verdict=None,
    )

    assert snapshot.average_return == 0.0
    assert snapshot.average_volatility == 0.0
    assert snapshot.breadth == 0.0
    assert snapshot.sample_count == 0
    assert snapshot.macro_target_gross_exposure == pytest.approx(0.55)
    assert snapshot.diagnostics
    assert any("empty" in note or "missing" in note for note in snapshot.diagnostics)


def test_build_regime_feature_snapshot_accepts_mapping_macro_verdict() -> None:
    snapshot = build_regime_feature_snapshot(
        market="CN",
        universe_key="full_a",
        as_of="20260625",
        frames={"000001.SZ": _frame()},
        tradability_snapshot={},
        cross_section_quant={"breadth": 1.5, "average_volatility": -1.0},
        macro_verdict={
            "final_score": 2.0,
            "metadata": {"target_gross_exposure": 1.5},
        },
    )

    assert snapshot.breadth == pytest.approx(1.0)
    assert snapshot.average_volatility == pytest.approx(0.0)
    assert snapshot.macro_score == pytest.approx(1.0)
    assert snapshot.macro_target_gross_exposure == pytest.approx(1.0)
