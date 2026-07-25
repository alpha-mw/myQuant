from __future__ import annotations

import numpy as np
import pandas as pd

from quant_investor.v17.forward_calibration import (
    assess_fundamental_eligibility,
    calibrate_forward_returns,
)

CUTOFF = pd.Timestamp("2026-06-30T15:00:00+08:00")


def _observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-31T15:00:00+08:00", periods=12, freq="ME")
    for horizon in (120, 252, 378):
        for index in range(120):
            start = dates[index % len(dates)]
            end = start + pd.Timedelta(days=horizon + 50)
            rows.append(
                {
                    "symbol": f"{index:06d}.SZ",
                    "industry": "synthetic-industry",
                    "score_decile": 10,
                    "horizon": horizon,
                    "cross_section_date": start,
                    "availability": end,
                    "age_open_days": 100 + index,
                    "realized_open_days": horizon,
                    "is_pit_month_end": True,
                    "is_mature": True,
                    "stock_start_trade_date": start,
                    "stock_end_trade_date": end,
                    "benchmark_start_trade_date": start,
                    "benchmark_end_trade_date": end,
                    "stock_total_return": 0.20 + index / 10_000.0,
                    "benchmark_total_return": 0.05,
                    "benchmark_symbol": "H00300.CSI",
                    "stock_return_includes_dividends": True,
                    "benchmark_return_is_pre_tax_total_return": True,
                    "delisted": False,
                    "official_terminal_cash_settlement": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame["availability"] = frame["stock_end_trade_date"]
    return frame


def test_three_horizon_positive_q25_is_f_eligible() -> None:
    calibrated = calibrate_forward_returns(_observations(), cutoff=CUTOFF)
    assert set(calibrated["status"]) == {"AVAILABLE"}
    eligibility = assess_fundamental_eligibility(
        calibrated,
        industry="synthetic-industry",
        score_decile=10,
        deep_research_complete=True,
        severe_red_flags=False,
    )
    assert eligibility.status == "F_ELIGIBLE"
    assert eligibility.optimizer_q25_252 is not None
    assert eligibility.optimizer_q25_252 > 0.0


def test_date_mismatch_invalidates_cell_instead_of_dropping_row() -> None:
    observations = _observations()
    observations.loc[0, "benchmark_end_trade_date"] += pd.Timedelta(days=1)
    calibrated = calibrate_forward_returns(observations, cutoff=CUTOFF)
    cell = calibrated.loc[calibrated["horizon"] == 120].iloc[0]
    assert cell["status"] == "UNAVAILABLE"
    assert "stock_benchmark_trade_dates_mismatch" in cell["blockers"]


def test_unofficial_delisting_cash_invalidates_cell() -> None:
    observations = _observations()
    observations.loc[0, "delisted"] = True
    calibrated = calibrate_forward_returns(observations, cutoff=CUTOFF)
    cell = calibrated.loc[calibrated["horizon"] == 120].iloc[0]
    assert cell["status"] == "UNAVAILABLE"
    assert "delisting_without_official_terminal_cash" in cell["blockers"]


def test_unrealized_or_premature_forward_target_invalidates_cell() -> None:
    observations = _observations()
    after_cutoff = CUTOFF + pd.Timedelta(days=1)
    observations.loc[0, "stock_end_trade_date"] = after_cutoff
    observations.loc[0, "benchmark_end_trade_date"] = after_cutoff
    observations.loc[0, "availability"] = after_cutoff
    calibrated = calibrate_forward_returns(observations, cutoff=CUTOFF)
    cell = calibrated.loc[calibrated["horizon"] == 120].iloc[0]
    assert cell["status"] == "UNAVAILABLE"
    assert "forward_end_after_cutoff" in cell["blockers"]

    observations = _observations()
    observations.loc[0, "availability"] = observations.loc[
        0, "stock_end_trade_date"
    ] - pd.Timedelta(microseconds=1)
    calibrated = calibrate_forward_returns(observations, cutoff=CUTOFF)
    cell = calibrated.loc[calibrated["horizon"] == 120].iloc[0]
    assert "forward_return_available_before_end" in cell["blockers"]


def test_forward_start_horizon_and_industry_are_fail_closed() -> None:
    observations = _observations()
    observations.loc[0, "stock_start_trade_date"] += pd.Timedelta(days=1)
    observations.loc[0, "benchmark_start_trade_date"] += pd.Timedelta(days=1)
    observations.loc[1, "realized_open_days"] = 119
    observations.loc[2, "industry"] = np.nan
    calibrated = calibrate_forward_returns(observations, cutoff=CUTOFF)

    normal_cell = calibrated.loc[
        (calibrated["horizon"] == 120) & calibrated["industry"].eq("synthetic-industry")
    ].iloc[0]
    assert "forward_start_not_cross_section" in normal_cell["blockers"]
    assert "forward_horizon_mismatch" in normal_cell["blockers"]
    unknown_cell = calibrated.loc[calibrated["industry"].eq("nan")].iloc[0]
    assert unknown_cell["status"] == "UNAVAILABLE"
    assert "industry_unknown" in unknown_cell["blockers"]


def test_duplicate_and_fractional_cell_keys_fail_closed() -> None:
    observations = _observations()
    observations = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
    observations["score_decile"] = observations["score_decile"].astype(float)
    observations.loc[observations["horizon"] == 252, "score_decile"] = 9.5
    calibrated = calibrate_forward_returns(observations, cutoff=CUTOFF)
    h120 = calibrated.loc[calibrated["horizon"] == 120].iloc[0]
    assert "duplicate_symbol_cross_section_horizon" in h120["blockers"]
    fractional = calibrated.loc[calibrated["horizon"] == 252].iloc[0]
    assert "invalid_score_decile" in fractional["blockers"]


def test_eligibility_rejects_coerced_controls_and_decile() -> None:
    calibrated = calibrate_forward_returns(_observations(), cutoff=CUTOFF)
    eligibility = assess_fundamental_eligibility(
        calibrated,
        industry="synthetic-industry",
        score_decile=10.0,  # type: ignore[arg-type]
        deep_research_complete=1,  # type: ignore[arg-type]
        severe_red_flags=0,  # type: ignore[arg-type]
    )
    assert eligibility.eligible is False
    assert "score_decile_invalid" in eligibility.blockers
    assert "deep_research_complete_not_strict_bool" in eligibility.blockers
    assert "severe_red_flags_not_strict_bool" in eligibility.blockers


def test_eligibility_missing_calibration_columns_fails_closed() -> None:
    eligibility = assess_fundamental_eligibility(
        pd.DataFrame({"industry": ["synthetic-industry"]}),
        industry="synthetic-industry",
        score_decile=10,
        deep_research_complete=True,
        severe_red_flags=False,
    )
    assert eligibility.eligible is False
    assert any(item.startswith("calibration_columns_missing:") for item in eligibility.blockers)
