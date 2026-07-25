from __future__ import annotations

import numpy as np
import pandas as pd

from quant_investor.v17.quant_timing import (
    assert_factor_source_binding,
    calibrate_timing_probabilities,
    compute_latest_scores,
    decide_timing,
)

CUTOFF = pd.Timestamp("2026-06-30T15:00:00+08:00")


def _price_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    for index in range(10):
        progression = np.linspace(10.0, 11.0 + index, len(dates))
        progression[::7] *= 0.99
        volume = 1000.0 + index * 10.0 + np.sin(np.arange(len(dates))) * 50.0
        frames[f"{index:06d}.SZ"] = pd.DataFrame(
            {
                "trade_date": dates,
                "availability": dates,
                "is_open_day": [True] * len(dates),
                "close": progression,
                "volume": volume,
                "amount": progression * volume,
            }
        )
    return frames


def _calibration_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2023-01-31T15:00:00+08:00", periods=24, freq="ME")
    for horizon in (20, 60):
        for decile in range(1, 11):
            wins = decile * 20
            for index in range(200):
                date = dates[index % len(dates)]
                target_end = date + pd.Timedelta(days=horizon * 2)
                rows.append(
                    {
                        "symbol": f"{300000 + decile * 1000 + index:06d}.SZ",
                        "horizon": horizon,
                        "score_decile": decile,
                        "cross_section_date": date,
                        "availability": target_end,
                        "age_open_days": index + 1,
                        "target_start_trade_date": date,
                        "target_end_trade_date": target_end,
                        "realized_open_days": horizon,
                        "is_mature": True,
                        "is_pit": True,
                        "target_definition": "EXCESS_RETURN_GT_ZERO",
                        "excess_return": 0.01 if index < wins else -0.01,
                    }
                )
    return pd.DataFrame(rows)


def test_factor_source_binding_and_sealed_three_factor_scores() -> None:
    assert assert_factor_source_binding() == (
        "a2c7ec5f2ec30a5367662c86151f8da42e6d5b6b3d99069c106be9aaf28ca168"
    )
    frames = _price_frames()
    scores = compute_latest_scores(frames, sealed_symbols=tuple(frames), cutoff=CUTOFF)
    assert len(scores) == 10
    assert set(scores["status"]) == {"READY"}
    assert scores["composite_score"].between(0.0, 1.0).all()


def test_jeffreys_pava_and_timing_thresholds() -> None:
    calibration = calibrate_timing_probabilities(_calibration_observations(), cutoff=CUTOFF)
    assert calibration.ready is True
    for horizon in (20, 60):
        probabilities = calibration.cells.loc[
            calibration.cells["horizon"] == horizon, "isotonic_probability"
        ].tolist()
        assert probabilities == sorted(probabilities)

    latest = pd.DataFrame(
        {
            "symbol": [f"{index:06d}.SZ" for index in range(10)],
            "composite_score": np.arange(10, dtype=float),
            "status": ["READY"] * 10,
        }
    )
    decisions = decide_timing(latest, calibration).set_index("symbol")
    assert decisions.loc["000000.SZ", "timing_state"] == "TRIM_TIMING"
    assert decisions.loc["000009.SZ", "timing_state"] == "BUY_NOW"
    assert decisions.loc["000005.SZ", "timing_state"] == "WATCH"


def test_any_decile_below_minimum_makes_quant_unready() -> None:
    observations = _calibration_observations()
    observations = observations.drop(
        observations.loc[
            (observations["horizon"] == 20) & (observations["score_decile"] == 1)
        ].index[:1]
    )
    calibration = calibrate_timing_probabilities(observations, cutoff=CUTOFF)
    assert calibration.ready is False
    assert "h20_d1:observations_below_200" in calibration.blockers


def test_duplicate_symbol_cross_section_horizon_cannot_inflate_calibration() -> None:
    observations = _calibration_observations()
    observations = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
    calibration = calibrate_timing_probabilities(observations, cutoff=CUTOFF)
    assert calibration.ready is False
    assert "duplicate_symbol_cross_section_horizon" in calibration.blockers


def test_latest_factor_scores_reject_post_cutoff_or_non_open_bars() -> None:
    frames = _price_frames()
    symbol = next(iter(frames))
    frames[symbol].loc[0, "availability"] = pd.Timestamp("2026-07-01")
    try:
        compute_latest_scores(frames, sealed_symbols=tuple(frames), cutoff=CUTOFF)
    except ValueError as exc:
        assert "price_frame_after_cutoff" in str(exc)
    else:
        raise AssertionError("post-cutoff price evidence was accepted")

    frames = _price_frames()
    frames[symbol].loc[0, "is_open_day"] = False
    try:
        compute_latest_scores(frames, sealed_symbols=tuple(frames), cutoff=CUTOFF)
    except ValueError as exc:
        assert "non-open session" in str(exc)
    else:
        raise AssertionError("non-open price row was accepted")


def test_timing_targets_must_be_mature_and_horizon_bound() -> None:
    observations = _calibration_observations()
    observations.loc[0, "availability"] = observations.loc[0, "cross_section_date"]
    observations.loc[1, "realized_open_days"] = 19
    observations.loc[2, "target_end_trade_date"] = CUTOFF + pd.Timedelta(days=1)
    calibration = calibrate_timing_probabilities(observations, cutoff=CUTOFF)
    assert calibration.ready is False
    assert "target_available_before_end" in calibration.blockers
    assert "target_horizon_mismatch" in calibration.blockers
    assert "target_end_after_cutoff" in calibration.blockers
