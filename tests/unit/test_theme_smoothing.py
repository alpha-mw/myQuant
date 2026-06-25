from __future__ import annotations

from quant_investor.themes.smoothing import (
    ThemeSmoothingConfig,
    smooth_theme_series,
)


def test_sma10_marks_single_day_spike_unconfirmed() -> None:
    result = smooth_theme_series(
        [42, 40, 41, 43, 44, 45, 47, 48, 50, 82],
        ThemeSmoothingConfig(window=10, min_observations=5),
    )

    payload = result.to_dict()

    assert payload["status"] == "success"
    assert payload["raw_score"] == 82.0
    assert payload["smoothed_score"] == 48.2
    assert payload["heat_10d"] == 48.2
    assert payload["persistence_count"] == 1
    assert payload["trend_state"] == "spike_unconfirmed"


def test_sma10_detects_persistent_warming() -> None:
    result = smooth_theme_series(
        [54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
        ThemeSmoothingConfig(window=10, min_observations=5),
    )

    payload = result.to_dict()

    assert payload["status"] == "success"
    assert payload["smoothed_score"] == 58.5
    assert payload["persistence_count"] == 9
    assert payload["trend_state"] == "warming"


def test_sma10_insufficient_history_does_not_fabricate_heat() -> None:
    result = smooth_theme_series(
        [80, 82, "bad", 86],
        ThemeSmoothingConfig(window=10, min_observations=5),
    )

    payload = result.to_dict()

    assert payload["status"] == "insufficient_history"
    assert payload["raw_score"] == 86.0
    assert payload["smoothed_score"] is None
    assert payload["heat_10d"] is None
    assert payload["trend_state"] == "insufficient_history"
    assert "malformed_theme_score_history_value" in payload["diagnostic_notes"]
