from __future__ import annotations

import json

import pytest

from quant_investor.config import MAINLINE_ENV_DEFAULTS, config
from quant_investor.regime.engine import MarkovRegimeEngine
from quant_investor.regime.types import (
    REGIME_RANGE_HIGH_VOL,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    REGIME_UNKNOWN,
)


def test_markov_config_defaults_are_production_first() -> None:
    assert MAINLINE_ENV_DEFAULTS["MARKOV_REGIME_ENABLED"] == "1"
    assert MAINLINE_ENV_DEFAULTS["MARKOV_REGIME_EXECUTION_TARGET"] == "production"
    assert config.MARKOV_REGIME_ENABLED is True
    assert config.MARKOV_REGIME_EXECUTION_TARGET == "production"


def test_markov_engine_defaults_and_shadow_target_normalize_to_production(tmp_path) -> None:
    default_engine = MarkovRegimeEngine(history_path=str(tmp_path / "default.jsonl"))
    assert default_engine.execution_target == "production"

    shadow_engine = MarkovRegimeEngine(
        history_path=str(tmp_path / "shadow.jsonl"),
        execution_target="shadow",
    )
    signal = shadow_engine.run(
        market="CN",
        universe_key="full_a",
        as_of="20260625",
        frames={},
        tradability_snapshot={},
        cross_section_quant={},
        macro_verdict=None,
        market_snapshot={},
    )

    assert shadow_engine.execution_target == "production"
    assert "markov_shadow_deprecated_normalized_to_production" in signal.diagnostic_notes


def _tradability(
    *,
    momentum: float,
    breakout: float,
    fake_breakout: float,
    drawdown: float,
    liquidity: float,
    volume_confirmation: float,
    count: int = 20,
) -> dict[str, dict[str, object]]:
    return {
        f"{idx:06d}.SZ": {
            "market_state": {
                "momentum_strength": momentum,
                "breakout_readiness": breakout,
                "fake_breakout_risk": fake_breakout,
                "max_drawdown_pct": drawdown,
                "liquidity_score": liquidity,
                "volume_confirmation": volume_confirmation,
            }
        }
        for idx in range(count)
    }


def test_markov_engine_bullish_features_prefer_trend_up(tmp_path) -> None:
    signal = MarkovRegimeEngine(history_path=str(tmp_path / "history.jsonl")).run(
        market="CN",
        universe_key="full_a",
        as_of="20260625",
        frames={},
        tradability_snapshot=_tradability(
            momentum=0.78,
            breakout=0.72,
            fake_breakout=0.10,
            drawdown=0.04,
            liquidity=0.88,
            volume_confirmation=0.70,
        ),
        cross_section_quant={
            "average_return": 0.018,
            "average_volatility": 0.012,
            "breadth": 0.76,
            "sample_count": 20,
        },
        macro_verdict={"final_score": 0.55, "metadata": {"target_gross_exposure": 0.68}},
        market_snapshot={},
    )

    assert signal.dominant_regime == REGIME_TREND_UP
    assert signal.probabilities[REGIME_TREND_UP] == pytest.approx(signal.confidence)
    assert signal.suggested_gross_exposure_cap <= 0.68
    json.dumps(signal.to_dict(), ensure_ascii=False)


def test_markov_engine_bearish_high_vol_features_prefer_defensive_regime(tmp_path) -> None:
    signal = MarkovRegimeEngine(history_path=str(tmp_path / "history.jsonl")).run(
        market="CN",
        universe_key="full_a",
        as_of="20260625",
        frames={},
        tradability_snapshot=_tradability(
            momentum=0.12,
            breakout=0.10,
            fake_breakout=0.75,
            drawdown=0.30,
            liquidity=0.35,
            volume_confirmation=0.12,
        ),
        cross_section_quant={
            "average_return": -0.024,
            "average_volatility": 0.065,
            "breadth": 0.18,
            "sample_count": 20,
        },
        macro_verdict={"final_score": -0.65, "metadata": {"target_gross_exposure": 0.50}},
        market_snapshot={},
    )

    assert signal.dominant_regime in {REGIME_TREND_DOWN, REGIME_RANGE_HIGH_VOL}
    assert signal.transition_risk >= 0.50
    assert signal.suggested_max_single_weight <= 0.09
    assert signal.turnover_cap is not None


def test_markov_engine_disabled_returns_unknown_safe_without_persisting(tmp_path) -> None:
    history_path = tmp_path / "history.jsonl"

    signal = MarkovRegimeEngine(
        history_path=str(history_path),
        enabled=False,
    ).run(
        market="CN",
        universe_key="full_a",
        as_of="20260625",
        frames={},
        tradability_snapshot={},
        cross_section_quant={},
        macro_verdict=None,
        market_snapshot={},
    )

    assert signal.dominant_regime == REGIME_UNKNOWN
    assert signal.probabilities[REGIME_UNKNOWN] == pytest.approx(1.0)
    assert signal.suggested_gross_exposure_cap <= 0.55
    assert not history_path.exists()


def test_markov_engine_avoids_duplicate_history_records(tmp_path) -> None:
    history_path = tmp_path / "history.jsonl"
    engine = MarkovRegimeEngine(history_path=str(history_path))
    kwargs = {
        "market": "CN",
        "universe_key": "full_a",
        "as_of": "20260625",
        "frames": {},
        "tradability_snapshot": _tradability(
            momentum=0.70,
            breakout=0.65,
            fake_breakout=0.10,
            drawdown=0.05,
            liquidity=0.80,
            volume_confirmation=0.60,
            count=5,
        ),
        "cross_section_quant": {
            "average_return": 0.015,
            "average_volatility": 0.015,
            "breadth": 0.70,
            "sample_count": 5,
        },
        "macro_verdict": None,
        "market_snapshot": {},
    }

    engine.run(**kwargs)
    engine.run(**kwargs)

    lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
