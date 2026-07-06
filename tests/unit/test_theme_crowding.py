from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_investor.themes import ThemeScanner, ThemeScore
from quant_investor.themes.scanner import _is_limitup_latest, _limitup_threshold_pct


def _frame(
    closes: list[float],
    *,
    highs: list[float] | None = None,
    volumes: list[float] | None = None,
    amounts: list[float] | None = None,
    pct_chg: list[float] | None = None,
) -> pd.DataFrame:
    data: dict[str, object] = {
        "trade_date": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        "close": closes,
        "high": highs if highs is not None else closes,
        "vol": volumes if volumes is not None else [1000.0] * len(closes),
    }
    if amounts is not None:
        data["amount"] = amounts
    if pct_chg is not None:
        data["pct_chg"] = pct_chg
    return pd.DataFrame(data)


def _trend(start: float, end: float, periods: int = 30) -> list[float]:
    if periods <= 1:
        return [end]
    step = (end - start) / (periods - 1)
    return [start + step * idx for idx in range(periods)]


def _history(theme_id: str, shares: list[float]) -> list[dict[str, object]]:
    return [
        {
            "theme_rotation": {
                "theme_scores": {
                    theme_id: {
                        "score": 60.0,
                        "theme_turnover_share": share,
                    }
                }
            }
        }
        for share in shares
    ]


def test_theme_crowding_metrics_are_deterministic_hand_calc():
    closes = _trend(10.0, 11.0)
    members = {
        "000001.SZ": 100.0,
        "000002.SZ": 200.0,
        "000003.SZ": 300.0,
        "000004.SZ": 400.0,
    }
    frames = {
        symbol: _frame(
            closes,
            amounts=[amount] * len(closes),
            pct_chg=[0.01] * (len(closes) - 1) + ([9.8] if symbol == "000004.SZ" else [0.01]),
        )
        for symbol, amount in members.items()
    }
    frames["600000.SH"] = _frame(closes, amounts=[1000.0] * len(closes), pct_chg=[0.01] * len(closes))
    industry_map = {symbol: "AI" for symbol in members}
    industry_map["600000.SH"] = "Banking"

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=4,
        crowding_enabled=True,
        crowding_min_universe=5,
        snapshot_history=_history("industry::ai", [0.40] * 5),
    )

    theme = result.theme_scores["industry::ai"]
    assert theme.crowding_status == "success"
    assert theme.theme_turnover_share == pytest.approx(0.5)
    assert theme.turnover_share_stretch == pytest.approx(0.25)
    assert theme.theme_limitup_ratio == pytest.approx(0.25)
    assert theme.limitup_norm == pytest.approx(0.25 / 0.30)
    assert theme.member_turnover_concentration == pytest.approx(0.9)
    assert theme.crowding_risk == pytest.approx(0.584166, rel=1e-5)
    assert any(item.startswith("crowding_risk=") for item in theme.evidence)


def test_limitup_detection_handles_board_thresholds_and_pct_chg_units():
    assert _limitup_threshold_pct("688001.SH") == pytest.approx(19.5)
    assert _limitup_threshold_pct("300001.SZ") == pytest.approx(19.5)
    assert _limitup_threshold_pct("430001.BJ") == pytest.approx(29.5)
    assert _limitup_threshold_pct("600001.SH") == pytest.approx(9.5)

    assert _is_limitup_latest(
        "300001.SZ",
        close=[10.0, 12.0],
        high=[10.0, 12.0],
        pct_chg=[0.0, 0.196],
    )
    assert _is_limitup_latest(
        "600001.SH",
        close=[10.0, 10.98],
        high=[10.0, 10.98],
        pct_chg=[0.0, 9.8],
    )
    assert not _is_limitup_latest(
        "600002.SH",
        close=[10.0, 10.98],
        high=[10.0, 11.20],
        pct_chg=[0.0, 9.8],
    )


def test_crowding_approximates_missing_amount_and_records_diagnostic():
    frames = {
        f"00000{idx}.SZ": _frame(
            _trend(10.0, 11.0),
            volumes=[1000.0 + idx] * 30,
        )
        for idx in range(1, 5)
    }
    frames["600000.SH"] = _frame(_trend(10.0, 10.5), volumes=[3000.0] * 30)
    industry_map = {symbol: "Robotics" for symbol in frames if symbol != "600000.SH"}
    industry_map["600000.SH"] = "Banking"

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=4,
        crowding_enabled=True,
        crowding_min_universe=5,
        snapshot_history=_history("industry::robotics", [0.50] * 5),
    )

    theme = result.theme_scores["industry::robotics"]
    assert theme.theme_turnover_share > 0.0
    assert any("amount_approximated" in note for note in theme.crowding_diagnostic_notes)


def test_crowding_flag_adds_to_overextension_without_exceeding_one():
    frames = {
        f"00000{idx}.SZ": _frame(
            _trend(10.0, 11.0),
            amounts=[1000.0] * 30,
            pct_chg=[1.0] * 29 + [9.8],
        )
        for idx in range(1, 5)
    }
    industry_map = {symbol: "Crowded" for symbol in frames}

    disabled = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=4,
        crowding_enabled=False,
    )
    enabled = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=4,
        crowding_enabled=True,
        crowding_min_universe=4,
        snapshot_history=_history("industry::crowded", [0.10] * 5),
    )

    base = disabled.theme_scores["industry::crowded"]
    crowded = enabled.theme_scores["industry::crowded"]
    assert crowded.crowding_risk >= 0.70
    assert "theme_crowded" in crowded.risk_flags
    assert crowded.overextension_risk == pytest.approx(
        min(base.overextension_risk + 0.30 * crowded.crowding_risk, 1.0)
    )


def test_crowding_insufficient_universe_records_status_without_flags():
    frames = {
        f"00000{idx}.SZ": _frame(_trend(10.0, 11.0), amounts=[1000.0] * 30)
        for idx in range(1, 5)
    }
    industry_map = {symbol: "Small Universe" for symbol in frames}

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=4,
        crowding_enabled=True,
        crowding_min_universe=10,
    )

    theme = result.theme_scores["industry::small-universe"]
    assert theme.crowding_status == "insufficient_universe"
    assert theme.crowding_risk == 0.0
    assert "theme_crowded" not in theme.risk_flags
    assert any("insufficient_universe" in note for note in theme.crowding_diagnostic_notes)


def test_crowding_switch_off_keeps_new_fields_neutral_and_existing_payload_serializable():
    frames = {
        f"00000{idx}.SZ": _frame(_trend(10.0, 11.0), amounts=[1000.0] * 30)
        for idx in range(1, 5)
    }
    industry_map = {symbol: "Neutral" for symbol in frames}

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=4,
        crowding_enabled=False,
    )
    payload = result.to_dict()
    theme = payload["theme_scores"]["industry::neutral"]

    assert theme["crowding_status"] == "disabled"
    assert theme["crowding_risk"] == 0.0
    assert theme["theme_turnover_share"] == 0.0
    assert theme["theme_limitup_ratio"] == 0.0
    assert "theme_crowded" not in theme["risk_flags"]
    json.dumps(payload)


def test_old_snapshot_history_without_crowding_fields_is_compatible():
    frames = {
        f"00000{idx}.SZ": _frame(_trend(10.0, 11.0), amounts=[1000.0] * 30)
        for idx in range(1, 5)
    }
    industry_map = {symbol: "Compat" for symbol in frames}

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=4,
        crowding_enabled=True,
        crowding_min_universe=4,
        snapshot_history=[
            {
                "theme_rotation": {
                    "theme_scores": {
                        "industry::compat": {
                            "score": 72.0,
                        }
                    }
                }
            }
        ],
    )

    theme = result.theme_scores["industry::compat"]
    assert theme.crowding_status == "success"
    assert theme.turnover_share_stretch == 0.0
    assert any("insufficient_history" in note for note in theme.crowding_diagnostic_notes)


def test_theme_score_crowding_defaults_make_legacy_objects_serializable():
    payload = ThemeScore(theme_id="industry::legacy", theme_name="Legacy").to_dict()

    assert payload["crowding_status"] == "disabled"
    assert payload["crowding_risk"] == 0.0
    assert payload["crowding_diagnostic_notes"] == []
    json.dumps(payload)
