"""Three-factor v17 timing and deterministic probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import pandas as pd

from quant_investor.factors.price_volume import compute_price_volume_factor

from .contracts import require_symbol
from .resources import load_json_resource
from .storage import file_sha256

FACTOR_NAMES = (
    "pv_blend_volstab19x2_mom90_amihud5_w80",
    "pv_short_reversal_25d",
    "pv_downside_volatility_15d",
)
EXPECTED_DEFINITION_SHA256 = (
    "a412e07a6c5df48f250577d821209670ffce36794c61b514c8eb8b3b412a499d",
    "17863a088f0bebaaf68ea137f13ab3fb7576bb24514f6c87735838b9643467df",
    "8bb6170626d59c45aa2d21171e1660ad06fdc5efc322006869bac529e79b4152",
)
FACTOR_RESOURCE_SHA256 = "670d18dd8f164f3390ee9838b626a7fd893f699042e58ab71d303c022eb47c56"
CALIBRATION_HORIZONS = (20, 60)
MIN_OBSERVATIONS_PER_DECILE = 200
MIN_CROSS_SECTIONS_PER_DECILE = 24
MAX_LOOKBACK_OPEN_DAYS = 1260
BUY_THRESHOLD = 0.60
TRIM_THRESHOLD = 0.40

_RESOURCE_PATH = Path(__file__).with_name("resources") / "quant_factor_set.v1.json"
_SOURCE_PATH = Path(__file__).resolve().parents[1] / "factors" / "price_volume.py"

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


def load_factor_policy() -> Mapping[str, object]:
    return load_json_resource(
        _RESOURCE_PATH.name,
        expected_sha256=FACTOR_RESOURCE_SHA256,
    )


def assert_factor_source_binding() -> str:
    policy = load_factor_policy()
    expected = str(policy.get("implementation_source_sha256", ""))
    actual = file_sha256(_SOURCE_PATH)
    if actual != expected:
        raise ValueError(
            "price_volume.py byte SHA mismatch: "
            f"expected={expected or 'MISSING'} actual={actual}"
        )
    factors = policy.get("factors")
    if not isinstance(factors, list) or not all(isinstance(item, dict) for item in factors):
        raise ValueError("quant factor policy factors must be objects")
    if tuple(item.get("name") for item in factors) != FACTOR_NAMES:
        raise ValueError("quant factor policy names/order mismatch")
    if tuple(item.get("definition_sha256") for item in factors) != EXPECTED_DEFINITION_SHA256:
        raise ValueError("quant factor definition SHA mismatch")
    return actual


def compute_latest_scores(
    frames: Mapping[str, pd.DataFrame],
    *,
    sealed_symbols: tuple[str, ...] | list[str],
    cutoff: datetime | str | pd.Timestamp,
) -> pd.DataFrame:
    """Compute equal-weight timing scores only inside the sealed universe."""

    assert_factor_source_binding()
    cutoff_ts = _cutoff_utc(cutoff)
    requested = tuple(require_symbol(value, label="sealed symbol") for value in sealed_symbols)
    sealed = tuple(dict.fromkeys(requested))
    if not sealed or len(sealed) != len(requested):
        raise ValueError("sealed_symbols must be unique, non-empty security codes")
    if any(not isinstance(key, str) for key in frames):
        raise ValueError("price/volume frame keys must be canonical symbol strings")
    missing = sorted(set(sealed).difference(frames))
    if missing:
        raise ValueError(f"sealed symbols missing price/volume frames: {missing}")
    selected: dict[str, pd.DataFrame] = {}
    for symbol in sealed:
        frame = frames[symbol].copy(deep=True)
        required_columns = {"trade_date", "availability", "is_open_day"}
        missing_columns = sorted(required_columns.difference(frame.columns))
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
        frame = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
        selected[symbol] = frame
    raw = pd.DataFrame(index=pd.Index(sealed, name="symbol"))
    for name in FACTOR_NAMES:
        values = compute_price_volume_factor(name, selected)
        raw[name] = pd.to_numeric(values.reindex(sealed), errors="coerce")
    ready = np.isfinite(raw[list(FACTOR_NAMES)]).all(axis=1)
    ranks = raw[list(FACTOR_NAMES)].rank(method="average", pct=True)
    raw["composite_score"] = ranks.mean(axis=1).where(ready, np.nan)
    raw["status"] = np.where(ready, "READY", "UNREADY")
    return raw.reset_index()


def _cutoff_utc(cutoff: datetime | str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(cutoff)
    if parsed.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return parsed.tz_convert("UTC")


def _strict_true(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _pava_non_decreasing(values: list[float], weights: list[int]) -> list[float]:
    """Weighted, deterministic pool-adjacent-violators algorithm."""

    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append({"start": index, "end": index, "weight": weight, "mean": value})
        while len(blocks) >= 2 and float(blocks[-2]["mean"]) > float(blocks[-1]["mean"]):
            right = blocks.pop()
            left = blocks.pop()
            total_weight = int(left["weight"]) + int(right["weight"])
            pooled = (
                float(left["mean"]) * int(left["weight"])
                + float(right["mean"]) * int(right["weight"])
            ) / total_weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": total_weight,
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
    """Fit Jeffreys win probabilities and monotone decile calibration."""

    missing = sorted(REQUIRED_CALIBRATION_COLUMNS.difference(observations.columns))
    if missing:
        raise ValueError(f"timing observations missing required columns: {missing}")
    cutoff_ts = _cutoff_utc(cutoff)
    frame = observations.copy(deep=True)
    frame["cross_section_date"] = pd.to_datetime(
        frame["cross_section_date"], utc=True, errors="coerce"
    )
    frame["availability"] = pd.to_datetime(frame["availability"], utc=True, errors="coerce")
    frame["target_start_trade_date"] = pd.to_datetime(
        frame["target_start_trade_date"], utc=True, errors="coerce"
    )
    frame["target_end_trade_date"] = pd.to_datetime(
        frame["target_end_trade_date"], utc=True, errors="coerce"
    )
    if (
        frame[
            [
                "cross_section_date",
                "availability",
                "target_start_trade_date",
                "target_end_trade_date",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("timing observations contain invalid timestamps")
    frame = frame.loc[frame["cross_section_date"] <= cutoff_ts].copy()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame["score_decile"] = pd.to_numeric(frame["score_decile"], errors="coerce")
    frame["age_open_days"] = pd.to_numeric(frame["age_open_days"], errors="coerce")
    frame["realized_open_days"] = pd.to_numeric(frame["realized_open_days"], errors="coerce")
    frame["excess_return"] = pd.to_numeric(frame["excess_return"], errors="coerce")
    try:
        for value in frame["symbol"]:
            require_symbol(value, label="timing observation symbol")
    except ValueError:
        raise ValueError("timing observations contain invalid symbols")
    duplicate_observations = frame.duplicated(
        ["symbol", "cross_section_date", "horizon"], keep=False
    )

    global_blockers: list[str] = []
    if frame.empty:
        global_blockers.append("no_mature_pit_observations")
    if bool(duplicate_observations.any()):
        global_blockers.append("duplicate_symbol_cross_section_horizon")
    if (frame["availability"] > cutoff_ts).any():
        global_blockers.append("evidence_after_cutoff")
    ages = frame["age_open_days"]
    if (
        ages.isna().any()
        or (ages % 1 != 0).any()
        or ((ages < 0) | (ages > MAX_LOOKBACK_OPEN_DAYS)).any()
    ):
        global_blockers.append("outside_1260_open_day_window")
    if any(not _strict_true(value) for value in frame["is_mature"]):
        global_blockers.append("immature_observation")
    if any(not _strict_true(value) for value in frame["is_pit"]):
        global_blockers.append("non_pit_observation")
    if not frame["target_definition"].astype(str).eq("EXCESS_RETURN_GT_ZERO").all():
        global_blockers.append("wrong_target_definition")
    if not np.isfinite(frame["excess_return"]).all():
        global_blockers.append("nonfinite_excess_return")
    if not frame["horizon"].isin(CALIBRATION_HORIZONS).all():
        global_blockers.append("invalid_horizon")
    if not frame["score_decile"].isin(range(1, 11)).all():
        global_blockers.append("invalid_score_decile")
    if not frame["target_start_trade_date"].eq(frame["cross_section_date"]).all():
        global_blockers.append("target_start_not_cross_section")
    if not (frame["target_end_trade_date"] > frame["target_start_trade_date"]).all():
        global_blockers.append("target_date_order_invalid")
    if (frame["target_end_trade_date"] > cutoff_ts).any():
        global_blockers.append("target_end_after_cutoff")
    if (frame["availability"] < frame["target_end_trade_date"]).any():
        global_blockers.append("target_available_before_end")
    realized = frame["realized_open_days"]
    if (
        realized.isna().any()
        or not np.isfinite(realized).all()
        or not realized.eq(realized.round()).all()
        or not realized.eq(frame["horizon"]).all()
    ):
        global_blockers.append("target_horizon_mismatch")

    cells: list[dict[str, object]] = []
    for horizon in CALIBRATION_HORIZONS:
        horizon_cells: list[dict[str, object]] = []
        for decile in range(1, 11):
            cell = frame.loc[(frame["horizon"] == horizon) & (frame["score_decile"] == decile)]
            blockers: list[str] = []
            count = int(len(cell))
            dates = int(cell["cross_section_date"].nunique())
            if count < MIN_OBSERVATIONS_PER_DECILE:
                blockers.append("observations_below_200")
            if dates < MIN_CROSS_SECTIONS_PER_DECILE:
                blockers.append("cross_sections_below_24")
            wins = int((cell["excess_return"] > 0.0).sum())
            probability = (wins + 0.5) / (count + 1.0) if count else np.nan
            horizon_cells.append(
                {
                    "horizon": horizon,
                    "score_decile": decile,
                    "observation_count": count,
                    "cross_section_count": dates,
                    "wins": wins,
                    "jeffreys_probability": probability,
                    "isotonic_probability": np.nan,
                    "status": "AVAILABLE" if not blockers else "UNAVAILABLE",
                    "blockers": tuple(blockers),
                }
            )
            global_blockers.extend(f"h{horizon}_d{decile}:{item}" for item in blockers)
        if all(item["status"] == "AVAILABLE" for item in horizon_cells):
            fitted = _pava_non_decreasing(
                [float(cast(float, item["jeffreys_probability"])) for item in horizon_cells],
                [int(cast(int, item["observation_count"])) for item in horizon_cells],
            )
            for item, probability in zip(horizon_cells, fitted, strict=True):
                item["isotonic_probability"] = probability
        cells.extend(horizon_cells)
    result = pd.DataFrame(cells)
    return TimingCalibration(
        ready=not global_blockers,
        cells=result,
        blockers=tuple(dict.fromkeys(global_blockers)),
    )


def decide_timing(
    latest_scores: pd.DataFrame,
    calibration: TimingCalibration,
) -> pd.DataFrame:
    """Map sealed latest scores to BUY_NOW/WATCH/TRIM_TIMING."""

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
    if count == 0:
        return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    for position, idx in enumerate(ready.index):
        decile = min(10, (position * 10) // count + 1)
        result.at[idx, "score_decile"] = decile
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
        result.at[idx, "probability_20d"] = p20
        result.at[idx, "probability_60d"] = p60
        if p20 >= BUY_THRESHOLD and p60 >= BUY_THRESHOLD:
            state = "BUY_NOW"
        elif p20 <= TRIM_THRESHOLD and p60 <= TRIM_THRESHOLD:
            state = "TRIM_TIMING"
        else:
            state = "WATCH"
        result.at[idx, "timing_state"] = state
    return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)


__all__ = [
    "BUY_THRESHOLD",
    "CALIBRATION_HORIZONS",
    "EXPECTED_DEFINITION_SHA256",
    "FACTOR_RESOURCE_SHA256",
    "FACTOR_NAMES",
    "TimingCalibration",
    "assert_factor_source_binding",
    "calibrate_timing_probabilities",
    "compute_latest_scores",
    "decide_timing",
    "load_factor_policy",
]
