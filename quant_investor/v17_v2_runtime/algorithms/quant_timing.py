"""Three-factor timing with deterministic Jeffreys/PAVA calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, cast

import numpy as np
import pandas as pd

from quant_investor.factors.price_volume import compute_price_volume_factor
from quant_investor.v17_v2_contract.identities import require_security_code

FACTOR_NAMES = (
    "pv_blend_volstab19x2_mom90_amihud5_w80",
    "pv_short_reversal_25d",
    "pv_downside_volatility_15d",
)
CALIBRATION_HORIZONS = (20, 60)
MIN_OBSERVATIONS_PER_DECILE = 200
MIN_CROSS_SECTIONS_PER_DECILE = 24
MAX_LOOKBACK_OPEN_DAYS = 1260
BUY_THRESHOLD = 0.60
TRIM_THRESHOLD = 0.40
REQUIRED_CALIBRATION_COLUMNS = frozenset(
    {
        "horizon",
        "symbol",
        "score_decile",
        "cross_section_date",
        "availability",
        "age_open_days",
        "target_start_trade_date",
        "target_end_trade_date",
        "realized_open_days",
        "is_mature",
        "is_pit",
        "target_definition",
        "excess_return",
    }
)


@dataclass(frozen=True)
class TimingCalibration:
    ready: bool
    cells: pd.DataFrame
    blockers: tuple[str, ...]


def _cutoff(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return result.tz_convert("UTC")


def _strict_true(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def compute_latest_scores(
    frames: Mapping[str, pd.DataFrame],
    *,
    sealed_symbols: tuple[str, ...] | list[str],
    cutoff: datetime | str | pd.Timestamp,
) -> pd.DataFrame:
    """Compute equal-weight factor ranks using only in-memory sealed frames."""

    cutoff_ts = _cutoff(cutoff)
    requested = tuple(
        require_security_code(value, label="sealed symbol") for value in sealed_symbols
    )
    sealed = tuple(dict.fromkeys(requested))
    if not sealed or len(sealed) != len(requested):
        raise ValueError("sealed_symbols must be unique and non-empty")
    missing = sorted(set(sealed).difference(frames))
    if missing:
        raise ValueError(f"sealed symbols missing price/volume frames: {missing}")
    selected: dict[str, pd.DataFrame] = {}
    for symbol in sealed:
        frame = frames[symbol].copy(deep=True)
        required = {"trade_date", "availability", "is_open_day"}
        missing_columns = sorted(required.difference(frame.columns))
        if missing_columns:
            raise ValueError(
                f"price/volume frame missing PIT columns for {symbol}: {missing_columns}"
            )
        trade_dates = pd.to_datetime(frame["trade_date"], utc=True, errors="coerce")
        availability = pd.to_datetime(frame["availability"], utc=True, errors="coerce")
        if trade_dates.isna().any() or availability.isna().any():
            raise ValueError(f"price/volume frame has invalid PIT timestamps: {symbol}")
        if trade_dates.duplicated(keep=False).any():
            raise ValueError(f"price/volume frame has duplicate sessions: {symbol}")
        if any(not _strict_true(value) for value in frame["is_open_day"]):
            raise ValueError(f"price/volume frame includes a non-open session: {symbol}")
        if (trade_dates > cutoff_ts).any() or (availability > cutoff_ts).any():
            raise ValueError(f"price_frame_after_cutoff:{symbol}")
        frame["trade_date"] = trade_dates
        frame["availability"] = availability
        selected[symbol] = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    raw = pd.DataFrame(index=pd.Index(sealed, name="symbol"))
    for name in FACTOR_NAMES:
        values = compute_price_volume_factor(name, selected)
        raw[name] = pd.to_numeric(values.reindex(sealed), errors="coerce")
    ready = np.isfinite(raw[list(FACTOR_NAMES)]).all(axis=1)
    raw["composite_score"] = (
        raw[list(FACTOR_NAMES)].rank(method="average", pct=True).mean(axis=1).where(ready, np.nan)
    )
    raw["status"] = np.where(ready, "READY", "UNREADY")
    return raw.reset_index()


def pava_non_decreasing(values: list[float], weights: list[int]) -> list[float]:
    """Weighted deterministic pool-adjacent-violators algorithm."""

    if len(values) != len(weights) or any(weight <= 0 for weight in weights):
        raise ValueError("PAVA values and positive weights must have equal length")
    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        if not np.isfinite(value):
            raise ValueError("PAVA values must be finite")
        blocks.append({"start": index, "end": index, "weight": weight, "mean": value})
        while len(blocks) >= 2 and float(blocks[-2]["mean"]) > float(blocks[-1]["mean"]):
            right = blocks.pop()
            left = blocks.pop()
            total = int(left["weight"]) + int(right["weight"])
            pooled = (
                float(left["mean"]) * int(left["weight"])
                + float(right["mean"]) * int(right["weight"])
            ) / total
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": total,
                    "mean": pooled,
                }
            )
    result = [0.0] * len(values)
    for block in blocks:
        for index in range(int(block["start"]), int(block["end"]) + 1):
            result[index] = float(block["mean"])
    return result


def calibrate_timing_probabilities(
    observations: pd.DataFrame,
    *,
    cutoff: datetime | str | pd.Timestamp,
) -> TimingCalibration:
    missing = sorted(REQUIRED_CALIBRATION_COLUMNS.difference(observations.columns))
    if missing:
        raise ValueError(f"timing observations missing required columns: {missing}")
    cutoff_ts = _cutoff(cutoff)
    frame = observations.copy(deep=True)
    timestamps = (
        "cross_section_date",
        "availability",
        "target_start_trade_date",
        "target_end_trade_date",
    )
    for column in timestamps:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame[list(timestamps)].isna().any().any():
        raise ValueError("timing observations contain invalid timestamps")
    frame = frame.loc[frame["cross_section_date"] <= cutoff_ts].copy()
    for column in (
        "horizon",
        "score_decile",
        "age_open_days",
        "realized_open_days",
        "excess_return",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    try:
        for value in frame["symbol"]:
            require_security_code(value, label="timing observation symbol")
    except ValueError as exc:
        raise ValueError("timing observations contain invalid symbols") from exc

    blockers: list[str] = []
    if frame.empty:
        blockers.append("no_mature_pit_observations")
    if frame.duplicated(["symbol", "cross_section_date", "horizon"], keep=False).any():
        blockers.append("duplicate_symbol_cross_section_horizon")
    if (frame["availability"] > cutoff_ts).any():
        blockers.append("evidence_after_cutoff")
    ages = frame["age_open_days"]
    if (
        ages.isna().any()
        or (ages % 1 != 0).any()
        or ((ages < 0) | (ages > MAX_LOOKBACK_OPEN_DAYS)).any()
    ):
        blockers.append("outside_1260_open_day_window")
    if any(not _strict_true(value) for value in frame["is_mature"]):
        blockers.append("immature_observation")
    if any(not _strict_true(value) for value in frame["is_pit"]):
        blockers.append("non_pit_observation")
    if not frame["target_definition"].astype(str).eq("EXCESS_RETURN_GT_ZERO").all():
        blockers.append("wrong_target_definition")
    if not np.isfinite(frame["excess_return"]).all():
        blockers.append("nonfinite_excess_return")
    if not frame["horizon"].isin(CALIBRATION_HORIZONS).all():
        blockers.append("invalid_horizon")
    if not frame["score_decile"].isin(range(1, 11)).all():
        blockers.append("invalid_score_decile")
    if not frame["target_start_trade_date"].eq(frame["cross_section_date"]).all():
        blockers.append("target_start_not_cross_section")
    if not (frame["target_end_trade_date"] > frame["target_start_trade_date"]).all():
        blockers.append("target_date_order_invalid")
    if (frame["target_end_trade_date"] > cutoff_ts).any():
        blockers.append("target_end_after_cutoff")
    if (frame["availability"] < frame["target_end_trade_date"]).any():
        blockers.append("target_available_before_end")
    realized = frame["realized_open_days"]
    if (
        realized.isna().any()
        or not np.isfinite(realized).all()
        or not realized.eq(realized.round()).all()
        or not realized.eq(frame["horizon"]).all()
    ):
        blockers.append("target_horizon_mismatch")

    cells: list[dict[str, object]] = []
    for horizon in CALIBRATION_HORIZONS:
        horizon_cells: list[dict[str, object]] = []
        for decile in range(1, 11):
            cell = frame.loc[(frame["horizon"] == horizon) & (frame["score_decile"] == decile)]
            cell_blockers: list[str] = []
            count = len(cell)
            dates = int(cell["cross_section_date"].nunique())
            if count < MIN_OBSERVATIONS_PER_DECILE:
                cell_blockers.append("observations_below_200")
            if dates < MIN_CROSS_SECTIONS_PER_DECILE:
                cell_blockers.append("cross_sections_below_24")
            wins = int((cell["excess_return"] > 0).sum())
            probability = (wins + 0.5) / (count + 1.0) if count else np.nan
            horizon_cells.append(
                {
                    "horizon": horizon,
                    "score_decile": decile,
                    "observation_count": int(count),
                    "cross_section_count": dates,
                    "wins": wins,
                    "jeffreys_probability": probability,
                    "isotonic_probability": np.nan,
                    "status": "AVAILABLE" if not cell_blockers else "UNAVAILABLE",
                    "blockers": tuple(cell_blockers),
                }
            )
            blockers.extend(f"h{horizon}_d{decile}:{item}" for item in cell_blockers)
        if all(item["status"] == "AVAILABLE" for item in horizon_cells):
            fitted = pava_non_decreasing(
                [float(cast(float, item["jeffreys_probability"])) for item in horizon_cells],
                [int(cast(int, item["observation_count"])) for item in horizon_cells],
            )
            for item, probability in zip(horizon_cells, fitted, strict=True):
                item["isotonic_probability"] = probability
        cells.extend(horizon_cells)
    return TimingCalibration(
        not blockers,
        pd.DataFrame(cells),
        tuple(dict.fromkeys(blockers)),
    )


def decide_timing(latest_scores: pd.DataFrame, calibration: TimingCalibration) -> pd.DataFrame:
    required = {"symbol", "composite_score", "status"}
    missing = sorted(required.difference(latest_scores.columns))
    if missing:
        raise ValueError(f"latest scores missing required columns: {missing}")
    result = latest_scores[["symbol", "composite_score", "status"]].copy(deep=True)
    result["symbol"] = result["symbol"].astype(str)
    result["score_decile"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["probability_20d"] = np.nan
    result["probability_60d"] = np.nan
    result["timing_state"] = "UNREADY"
    if not calibration.ready:
        return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    ready = result.loc[
        (result["status"] == "READY") & np.isfinite(result["composite_score"])
    ].sort_values(["composite_score", "symbol"], ascending=[True, True], kind="mergesort")
    count = len(ready)
    for position, index in enumerate(ready.index):
        decile = min(10, (position * 10) // count + 1)
        result.at[index, "score_decile"] = decile
        probabilities: dict[int, float] = {}
        for horizon in CALIBRATION_HORIZONS:
            cell = calibration.cells.loc[
                (calibration.cells["horizon"] == horizon)
                & (calibration.cells["score_decile"] == decile)
                & (calibration.cells["status"] == "AVAILABLE")
            ]
            if len(cell) != 1:
                probabilities = {}
                break
            probabilities[horizon] = float(cell.iloc[0]["isotonic_probability"])
        if set(probabilities) != set(CALIBRATION_HORIZONS):
            continue
        p20, p60 = probabilities[20], probabilities[60]
        result.at[index, "probability_20d"] = p20
        result.at[index, "probability_60d"] = p60
        result.at[index, "timing_state"] = (
            "BUY_NOW"
            if p20 >= BUY_THRESHOLD and p60 >= BUY_THRESHOLD
            else "TRIM_TIMING" if p20 <= TRIM_THRESHOLD and p60 <= TRIM_THRESHOLD else "WATCH"
        )
    return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)


__all__ = [
    "BUY_THRESHOLD",
    "CALIBRATION_HORIZONS",
    "FACTOR_NAMES",
    "TimingCalibration",
    "calibrate_timing_probabilities",
    "compute_latest_scores",
    "decide_timing",
    "pava_non_decreasing",
]
