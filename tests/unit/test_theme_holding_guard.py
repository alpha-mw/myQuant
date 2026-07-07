from __future__ import annotations

from quant_investor.monitoring.theme_holding_guard import (
    evaluate_holding_theme_guard,
)


def _payload(*, phase: str = "confirmed_rotation", flags: list[str] | None = None):
    return {
        "theme_rotation": {
            "status": "success",
            "symbol_primary_theme": {"000001.SZ": "industry::ai"},
            "symbol_phase": {"000001.SZ": phase},
            "symbol_risk_flags": {"000001.SZ": list(flags or [])},
            "theme_scores": {
                "industry::ai": {
                    "theme_id": "industry::ai",
                    "theme_name": "人工智能",
                    "phase": phase,
                }
            },
        }
    }


def test_distribution_phase_tightens_holding_guard():
    signals = evaluate_holding_theme_guard(
        ["000001.SZ"],
        _payload(phase="distribution"),
    )

    signal = signals["000001.SZ"]
    assert signal.primary_theme_id == "industry::ai"
    assert signal.primary_theme_name == "人工智能"
    assert signal.phase == "distribution"
    assert signal.guard_level == "tighten"
    assert "phase_distribution" in signal.reasons


def test_overextended_phase_sets_watch_guard():
    signals = evaluate_holding_theme_guard(
        ["000001.SZ"],
        _payload(phase="overextended"),
    )

    assert signals["000001.SZ"].guard_level == "watch"
    assert "phase_overextended" in signals["000001.SZ"].reasons


def test_confirmed_rotation_is_neutral():
    signals = evaluate_holding_theme_guard(
        ["000001.SZ"],
        _payload(phase="confirmed_rotation"),
    )

    assert signals["000001.SZ"].guard_level == "none"
    assert "phase_confirmed_rotation" in signals["000001.SZ"].reasons


def test_risk_flags_trigger_without_phase():
    tighten = evaluate_holding_theme_guard(
        ["000001.SZ"],
        _payload(phase="", flags=["theme_distribution_risk"]),
    )
    watch = evaluate_holding_theme_guard(
        ["000001.SZ"],
        _payload(phase="", flags=["theme_fake_breakout_risk"]),
    )

    assert tighten["000001.SZ"].guard_level == "tighten"
    assert "flag_theme_distribution_risk" in tighten["000001.SZ"].reasons
    assert watch["000001.SZ"].guard_level == "watch"
    assert "flag_theme_fake_breakout_risk" in watch["000001.SZ"].reasons


def test_missing_snapshot_fails_open_with_diagnostic_reason():
    signals = evaluate_holding_theme_guard(["000001.SZ"], {})

    signal = signals["000001.SZ"]
    assert signal.guard_level == "none"
    assert signal.primary_theme_id == ""
    assert "theme_snapshot_unavailable" in signal.reasons
