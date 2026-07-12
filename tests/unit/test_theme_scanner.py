from __future__ import annotations

import json

import pandas as pd

from quant_investor.themes import ThemePhase, ThemeScanner, ThemeScore
from quant_investor.market.dag.theme_context import build_theme_rotation_metadata


def _frame(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    data: dict[str, object] = {
        "trade_date": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        "close": closes,
    }
    if volumes is not None:
        data["volume"] = volumes
    return pd.DataFrame(data)


def _trend(start: float, end: float, periods: int = 30) -> list[float]:
    if periods <= 1:
        return [end]
    step = (end - start) / (periods - 1)
    return [start + step * idx for idx in range(periods)]


def test_theme_scan_empty_inputs_safe():
    result = ThemeScanner().scan(frames={}, industry_map={})

    assert result.theme_scores == {}
    assert result.symbol_scores == {}
    json.dumps(result.to_dict())


def test_theme_score_to_dict_serializable():
    score = ThemeScore(
        theme_id="industry::semiconductor",
        theme_name="Semiconductor",
        phase=ThemePhase.CONFIRMED_ROTATION,
        score=72.5,
        confidence=0.8,
        top_symbols=["000001.SZ"],
        risk_flags=["theme_fake_breakout_risk"],
    )

    payload = score.to_dict()

    assert payload["phase"] == "confirmed_rotation"
    assert payload["top_symbols"] == ["000001.SZ"]
    json.dumps(payload)


def test_theme_scanner_ranks_strong_industry_above_weak_industry():
    frames: dict[str, pd.DataFrame] = {}
    industry_map: dict[str, str] = {}

    strong_closes = _trend(10.0, 11.8)
    weak_closes = _trend(10.0, 9.6)
    strong_volumes = _trend(1000.0, 3000.0)
    weak_volumes = [1000.0] * 30

    for idx in range(6):
        strong_symbol = f"STR{idx:03d}.SZ"
        weak_symbol = f"WEAK{idx:03d}.SZ"
        frames[strong_symbol] = _frame(strong_closes, strong_volumes)
        frames[weak_symbol] = _frame(weak_closes, weak_volumes)
        industry_map[strong_symbol] = "Semiconductor"
        industry_map[weak_symbol] = "Banking"

    result = ThemeScanner().scan(frames=frames, industry_map=industry_map, min_member_count=5)

    strong = result.theme_scores["industry::semiconductor"]
    weak = result.theme_scores["industry::banking"]

    assert strong.score > weak.score
    assert strong.phase in {
        ThemePhase.ACCUMULATION,
        ThemePhase.EARLY_ACCELERATION,
        ThemePhase.CONFIRMED_ROTATION,
    }
    assert result.symbol_scores["STR000.SZ"] > result.symbol_scores["WEAK000.SZ"]


def test_theme_scanner_missing_volume_safe():
    frames = {f"S{idx:03d}.SZ": _frame(_trend(10.0, 11.0)) for idx in range(5)}
    industry_map = {symbol: "Robotics" for symbol in frames}

    result = ThemeScanner().scan(frames=frames, industry_map=industry_map, min_member_count=5)

    theme = result.theme_scores["industry::robotics"]
    assert theme.volume_confirmation == 0.0
    json.dumps(result.to_dict())


def test_theme_scanner_filters_small_themes():
    frames = {f"S{idx:03d}.SZ": _frame(_trend(10.0, 11.0)) for idx in range(4)}
    industry_map = {symbol: "Small Theme" for symbol in frames}

    result = ThemeScanner().scan(frames=frames, industry_map=industry_map, min_member_count=5)

    assert result.theme_scores == {}
    assert result.symbol_scores == {}


def test_theme_scanner_overextended_flag():
    closes = [10.0] * 24 + [10.2, 10.8, 11.7, 12.9, 14.2, 15.6]
    volumes = [1000.0] * 24 + [1600.0, 1800.0, 2200.0, 2600.0, 3200.0, 3800.0]
    frames = {f"S{idx:03d}.SZ": _frame(closes, volumes) for idx in range(6)}
    industry_map = {symbol: "AI Hardware" for symbol in frames}

    result = ThemeScanner().scan(frames=frames, industry_map=industry_map, min_member_count=5)

    theme = result.theme_scores["industry::ai-hardware"]
    assert "theme_overextended" in theme.risk_flags or theme.phase == ThemePhase.OVEREXTENDED


def test_theme_scanner_deterministic_tie_break():
    frames: dict[str, pd.DataFrame] = {}
    industry_map: dict[str, str] = {}
    closes = _trend(10.0, 11.0)
    volumes = _trend(1000.0, 2000.0)

    for idx in range(5):
        alpha_symbol = f"A{idx:03d}.SZ"
        beta_symbol = f"B{idx:03d}.SZ"
        frames[beta_symbol] = _frame(closes, volumes)
        frames[alpha_symbol] = _frame(closes, volumes)
        industry_map[beta_symbol] = "Beta"
        industry_map[alpha_symbol] = "Alpha"

    result = ThemeScanner().scan(frames=frames, industry_map=industry_map, min_member_count=5)

    assert list(result.theme_scores) == ["industry::alpha", "industry::beta"]


def test_theme_scanner_adds_smoothing_fields_without_overwriting_raw_scores():
    frames = {
        f"S{idx:03d}.SZ": _frame(_trend(10.0, 12.0), _trend(1000.0, 2600.0))
        for idx in range(6)
    }
    industry_map = {symbol: "Copper" for symbol in frames}

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
    )
    payload = result.to_dict()
    theme = payload["theme_scores"]["industry::copper"]

    assert "smoothed_score" in theme
    assert "heat_10d" in theme
    assert theme["smoothing_observation_count"] >= 5
    assert theme["smoothing_status"] == "success"
    assert result.symbol_scores["S000.SZ"] == result.theme_scores["industry::copper"].score / 100.0
    assert "S000.SZ" in result.symbol_smoothed_scores
    assert result.symbol_smoothed_scores["S000.SZ"] != result.symbol_scores["S000.SZ"]


def test_theme_rotation_metadata_exposes_symbol_smoothed_scores_separately():
    frames = {
        f"S{idx:03d}.SZ": _frame(_trend(10.0, 12.0), _trend(1000.0, 2600.0))
        for idx in range(6)
    }
    industry_map = {symbol: "Copper" for symbol in frames}

    payload = build_theme_rotation_metadata(
        frames=frames,
        industry_map=industry_map,
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        min_member_count=5,
        top_n=20,
    )

    assert payload["status"] == "success"
    assert "symbol_smoothed_scores" in payload
    assert payload["symbol_scores"]["S000.SZ"] != payload["symbol_smoothed_scores"]["S000.SZ"]
    assert payload["top_themes"][0]["heat_10d"] == payload["top_themes"][0]["smoothed_score"]


def test_attention_axes_include_horizons_turnover_breadth_new_high_and_leader_persistence():
    frames = {
        f"S{idx:03d}.SZ": _frame(
            _trend(10.0 + idx * 0.01, 16.0 + idx * 0.10, periods=130),
            _trend(1000.0, 3000.0 + idx * 50.0, periods=130),
        )
        for idx in range(6)
    }
    industry_map = {symbol: "AI" for symbol in frames}

    theme = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
    ).theme_scores["industry::ai"]

    assert theme.attention_5d is not None
    assert theme.attention_20d is not None
    assert theme.attention_60d is not None
    assert theme.attention_120d is not None
    assert theme.attention_turnover_share == 1.0
    assert theme.new_high_rate == 1.0
    assert theme.leader_persistence is not None
    assert theme.attention_history_coverage == 1.0
    assert 0.0 <= theme.attention <= 1.0
    assert 0.0 <= theme.market_confirmation <= 1.0


def test_insufficient_long_history_exports_null_axes_and_reduces_coverage_instead_of_fake_zero():
    frames = {
        f"S{idx:03d}.SZ": _frame(
            _trend(10.0, 11.0, periods=30),
            _trend(1000.0, 2000.0, periods=30),
        )
        for idx in range(6)
    }
    industry_map = {symbol: "Robotics" for symbol in frames}

    theme = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
    ).theme_scores["industry::robotics"]
    payload = theme.to_dict()

    assert payload["attention_5d"] is not None
    assert payload["attention_20d"] is not None
    assert payload["attention_60d"] is None
    assert payload["attention_120d"] is None
    assert payload["new_high_rate"] is None
    assert payload["attention_history_coverage"] < 0.30
