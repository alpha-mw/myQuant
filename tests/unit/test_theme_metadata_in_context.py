from __future__ import annotations

import json

import pandas as pd

from quant_investor.market.dag.theme_context import (
    build_disabled_theme_rotation_metadata,
    build_theme_rotation_metadata,
)
from quant_investor.market.dag.context import _theme_snapshot_scope_metadata


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


def _strong_weak_fixture() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
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
    return frames, industry_map


def test_build_theme_rotation_metadata_disabled():
    metadata = build_disabled_theme_rotation_metadata(
        market="CN",
        universe_key="full_a",
        as_of="20260618",
    )

    assert metadata["status"] == "disabled"
    assert metadata["enabled"] is False
    assert metadata["metadata"]["no_llm"] is True
    assert metadata["metadata"]["no_network"] is True
    assert metadata["theme_scores"] == {}
    json.dumps(metadata)


def test_build_theme_rotation_metadata_success():
    frames, industry_map = _strong_weak_fixture()

    metadata = build_theme_rotation_metadata(
        frames=frames,
        industry_map=industry_map,
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        min_member_count=5,
        top_n=20,
    )

    assert metadata["status"] == "success"
    assert metadata["enabled"] is True
    assert metadata["theme_scores"]
    assert metadata["top_themes"]
    assert metadata["symbol_scores"]
    assert metadata["metadata"]["no_llm"] is True
    assert metadata["metadata"]["no_network"] is True
    assert (
        metadata["theme_scores"]["industry::semiconductor"]["score"]
        > metadata["theme_scores"]["industry::banking"]["score"]
    )
    json.dumps(metadata)


def test_build_theme_rotation_metadata_symbol_limit():
    frames = {
        f"S{idx:03d}.SZ": _frame(_trend(10.0, 11.0), _trend(1000.0, 2200.0))
        for idx in range(12)
    }
    industry_map = {symbol: "Robotics" for symbol in frames}

    metadata = build_theme_rotation_metadata(
        frames=frames,
        industry_map=industry_map,
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        min_member_count=5,
        top_n=20,
        symbol_limit=3,
    )

    assert len(metadata["symbol_scores"]) <= 3
    assert metadata["metadata"]["truncated_symbol_count"] > 0


def test_build_theme_rotation_metadata_empty_industry_map_safe():
    metadata = build_theme_rotation_metadata(
        frames={"000001.SZ": _frame(_trend(10.0, 11.0))},
        industry_map={},
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        min_member_count=5,
        top_n=20,
    )

    assert metadata["status"] == "success"
    assert metadata["theme_scores"] == {}
    assert metadata["metadata"]["no_network"] is True


def test_build_theme_rotation_metadata_error_safe(monkeypatch):
    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "quant_investor.market.dag.theme_context.ThemeScanner.scan",
        _raise,
    )

    metadata = build_theme_rotation_metadata(
        frames={},
        industry_map={},
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="20260618",
    )

    assert metadata["status"] == "error"
    assert metadata["theme_scores"] == {}
    assert any("theme_scanner_error" in note for note in metadata["diagnostic_notes"])


def test_theme_snapshot_scope_separates_holding_single_from_full_market():
    full_market = _theme_snapshot_scope_metadata(
        universe_key="full_a",
        symbol_count=5500,
        explicit_symbol_count=0,
        unsampled_symbol_count=5500,
        sampled=False,
        recall_context={},
    )
    holding_single = _theme_snapshot_scope_metadata(
        universe_key="full_a",
        symbol_count=1,
        explicit_symbol_count=1,
        unsampled_symbol_count=1,
        sampled=False,
        recall_context={"holding_symbol": "688301.SH"},
    )

    assert full_market["input_scope"] == "full_market"
    assert full_market["snapshot_universe_key"] == "full_a"
    assert holding_single["input_scope"] == "holding_single"
    assert holding_single["snapshot_universe_key"] == "full_a_holding_single"
