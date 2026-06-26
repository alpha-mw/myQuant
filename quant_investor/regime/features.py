from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_investor.agent_protocol import BranchVerdict
from quant_investor.regime.types import RegimeFeatureSnapshot


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _clamp(value: Any, lower: float, upper: float, default: float = 0.0) -> float:
    numeric = _finite_float(value, default)
    return max(float(lower), min(float(upper), numeric))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _macro_metadata(macro_verdict: BranchVerdict | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if isinstance(macro_verdict, BranchVerdict):
        return _mapping(macro_verdict.metadata)
    if isinstance(macro_verdict, Mapping):
        return _mapping(macro_verdict.get("metadata"))
    return {}


def _macro_score(macro_verdict: BranchVerdict | Mapping[str, Any] | None) -> float:
    if isinstance(macro_verdict, BranchVerdict):
        return _clamp(macro_verdict.final_score, -1.0, 1.0)
    if isinstance(macro_verdict, Mapping):
        return _clamp(macro_verdict.get("final_score", 0.0), -1.0, 1.0)
    return 0.0


def _share(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for item in values if item) / len(values)


def _compact_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if "." in text and text.replace(".", "", 1).isdigit():
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _date_column(frame: pd.DataFrame) -> str:
    for column in ("trade_date", "date", "Date"):
        if column in frame.columns:
            return column
    return ""


def _truncate_frame_as_of(
    frame: pd.DataFrame,
    *,
    as_of: str,
) -> tuple[pd.DataFrame, list[str], bool]:
    diagnostics: list[str] = []
    if frame is None or frame.empty:
        return pd.DataFrame(), diagnostics, False
    date_col = _date_column(frame)
    if not date_col:
        return frame.copy(), ["regime_frame_date_column_missing"], False
    cutoff = _compact_date(as_of)
    if not cutoff:
        return frame.iloc[0:0].copy(), ["regime_as_of_invalid"], True
    normalized = frame[date_col].map(_compact_date)
    valid_mask = normalized.str.len().eq(8)
    invalid_count = int((~valid_mask).sum())
    if invalid_count:
        diagnostics.append(f"regime_frame_invalid_dates_ignored:{invalid_count}")
    future_count = int((valid_mask & (normalized > cutoff)).sum())
    if future_count:
        diagnostics.append(f"regime_frame_future_rows_truncated:{future_count}")
    result = frame.loc[valid_mask & (normalized <= cutoff)].copy()
    if not result.empty:
        result[date_col] = normalized.loc[result.index]
        result = result.sort_values(date_col, kind="stable").reset_index(drop=True)
    return result, diagnostics, True


def truncate_frames_as_of(
    frames: Mapping[str, pd.DataFrame],
    *,
    as_of: str,
) -> tuple[dict[str, pd.DataFrame], list[str], int]:
    truncated: dict[str, pd.DataFrame] = {}
    diagnostics: list[str] = []
    dated_count = 0
    seen_diagnostics: set[str] = set()
    for symbol, frame in (frames or {}).items():
        next_frame, notes, used_date_filter = _truncate_frame_as_of(frame, as_of=as_of)
        if used_date_filter:
            dated_count += 1
        for note in notes:
            if note not in seen_diagnostics:
                diagnostics.append(note)
                seen_diagnostics.add(note)
        truncated[str(symbol)] = next_frame
    return truncated, diagnostics, dated_count


def _numeric_series(frame: pd.DataFrame, *columns: str) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _pct_change_values(close: pd.Series) -> np.ndarray:
    values = close.to_numpy(dtype=float, copy=False)
    if values.size < 2:
        return np.array([], dtype=float)
    previous = values[:-1]
    current = values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = current / previous - 1.0
    return returns[np.isfinite(returns)]


def _window_return(close: pd.Series, window: int) -> float:
    if window <= 0 or len(close) <= window:
        return 0.0
    base = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if abs(base) <= 1e-8:
        return 0.0
    return latest / base - 1.0


def _drawdown(close: pd.Series, window: int = 120) -> float:
    if close.empty:
        return 0.0
    values = close.tail(max(int(window), 1)).to_numpy(dtype=float, copy=False)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    running_high = np.maximum.accumulate(values)
    valid = np.abs(running_high) > 1e-12
    if not valid.any():
        return 0.0
    drawdowns = np.zeros_like(values, dtype=float)
    drawdowns[valid] = 1.0 - values[valid] / running_high[valid]
    return _clamp(float(np.nanmax(drawdowns)), 0.0, 1.0)


def _derived_market_state(frame: pd.DataFrame) -> Mapping[str, Any]:
    close = _numeric_series(frame, "close", "Close", "adj_close")
    volume = _numeric_series(frame, "volume", "vol", "Volume")
    returns = _pct_change_values(close)
    avg_return = float(np.mean(returns[-20:])) if returns.size else 0.0
    volatility = float(np.std(returns[-60:], ddof=1)) if returns.size >= 3 else 0.0
    momentum_strength = _clamp((_window_return(close, 20) + 0.10) / 0.25, 0.0, 1.0)
    max_drawdown = _drawdown(close)
    latest_close = float(close.iloc[-1]) if not close.empty else 0.0
    recent_high = float(close.tail(120).max()) if not close.empty else 0.0
    if recent_high > 1e-8:
        distance_from_high = max(0.0, (recent_high - latest_close) / recent_high)
    else:
        distance_from_high = 1.0
    breakout_readiness = 1.0 - _clamp(distance_from_high / 0.06, 0.0, 1.0)
    if len(volume) >= 20:
        baseline_volume = float(volume.tail(20).mean())
    else:
        baseline_volume = float(volume.mean()) if len(volume) else 0.0
    latest_volume = float(volume.iloc[-1]) if len(volume) else 0.0
    volume_ratio = latest_volume / baseline_volume if baseline_volume > 0.0 else 0.0
    volume_confirmation = _clamp((volume_ratio - 1.0) / 0.35, 0.0, 1.0)
    liquidity_score = _clamp(0.70 * (len(close) / 250.0) + 0.30 * min(volume_ratio, 1.35) / 1.35, 0.0, 1.0)
    fake_breakout_risk = _clamp(
        breakout_readiness * (1.0 - volume_confirmation) * 0.55
        + _clamp(max_drawdown / 0.18, 0.0, 1.0) * 0.45,
        0.0,
        1.0,
    )
    return {
        "average_return": avg_return,
        "volatility": volatility,
        "momentum_strength": momentum_strength,
        "breakout_readiness": breakout_readiness,
        "fake_breakout_risk": fake_breakout_risk,
        "max_drawdown_pct": max_drawdown,
        "liquidity_score": liquidity_score,
        "volume_confirmation": volume_confirmation,
        "rows": int(len(frame)),
    }


def _derive_cross_section_and_states(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    states: list[Mapping[str, Any]] = []
    for frame in (frames or {}).values():
        if frame is None or frame.empty:
            continue
        state = _derived_market_state(frame)
        if int(state.get("rows", 0) or 0) > 0:
            states.append(state)
    if not states:
        return {
            "candidate_count": len(frames or {}),
            "sample_count": 0,
            "average_return": 0.0,
            "average_volatility": 0.0,
            "breadth": 0.0,
        }, []
    positive = sum(1 for state in states if float(state.get("average_return", 0.0)) > 0.0)
    return {
        "candidate_count": len(frames or {}),
        "sample_count": len(states),
        "average_return": float(sum(float(state.get("average_return", 0.0)) for state in states) / len(states)),
        "average_volatility": float(sum(float(state.get("volatility", 0.0)) for state in states) / len(states)),
        "breadth": positive / max(len(states), 1),
    }, states


def _market_states(
    tradability_snapshot: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    states: list[Mapping[str, Any]] = []
    for payload in tradability_snapshot.values():
        if not isinstance(payload, Mapping):
            continue
        state = payload.get("market_state")
        if isinstance(state, Mapping):
            states.append(state)
    return states


def build_regime_feature_snapshot(
    *,
    market: str,
    universe_key: str,
    as_of: str,
    frames: Mapping[str, pd.DataFrame],
    tradability_snapshot: Mapping[str, Mapping[str, Any]],
    cross_section_quant: Mapping[str, Any],
    macro_verdict: BranchVerdict | Mapping[str, Any] | None,
) -> RegimeFeatureSnapshot:
    diagnostics: list[str] = []
    truncated_frames, truncation_notes, dated_frame_count = truncate_frames_as_of(
        frames or {},
        as_of=str(as_of or ""),
    )
    diagnostics.extend(truncation_notes)
    if dated_frame_count:
        cross_section, states = _derive_cross_section_and_states(truncated_frames)
    else:
        cross_section = _mapping(cross_section_quant)
        states = _market_states(tradability_snapshot or {})
    frame_count = len(truncated_frames or {})

    if not cross_section:
        diagnostics.append("cross_section_quant_missing")
    if frame_count <= 0:
        diagnostics.append("frames_empty")
    if not states:
        diagnostics.append("tradability_market_state_empty")

    sample_count = int(max(_finite_float(cross_section.get("sample_count", frame_count), frame_count), 0.0))
    average_return = _clamp(cross_section.get("average_return", 0.0), -1.0, 1.0)
    average_volatility = _clamp(cross_section.get("average_volatility", 0.0), 0.0, 1.0)
    breadth = _clamp(cross_section.get("breadth", 0.0), 0.0, 1.0)

    momentum_flags: list[bool] = []
    breakout_flags: list[bool] = []
    fake_breakout_flags: list[bool] = []
    drawdowns: list[float] = []
    liquidity_scores: list[float] = []
    volume_confirmations: list[float] = []
    for state in states:
        momentum_strength = _clamp(state.get("momentum_strength", 0.0), 0.0, 1.0)
        breakout_readiness = _clamp(state.get("breakout_readiness", 0.0), 0.0, 1.0)
        fake_breakout_risk = _clamp(state.get("fake_breakout_risk", 0.0), 0.0, 1.0)
        max_drawdown_pct = _clamp(state.get("max_drawdown_pct", 0.0), 0.0, 1.0)
        liquidity_score = _clamp(state.get("liquidity_score", 0.0), 0.0, 1.0)
        volume_confirmation = _clamp(state.get("volume_confirmation", 0.0), 0.0, 1.0)

        momentum_flags.append(momentum_strength >= 0.55)
        breakout_flags.append(
            breakout_readiness >= 0.50 and volume_confirmation >= 0.35
        )
        fake_breakout_flags.append(fake_breakout_risk >= 0.60)
        drawdowns.append(max_drawdown_pct)
        liquidity_scores.append(liquidity_score)
        volume_confirmations.append(volume_confirmation)

    macro_meta = _macro_metadata(macro_verdict)
    macro_target_gross_exposure = _clamp(
        macro_meta.get("target_gross_exposure", 0.55),
        0.0,
        1.0,
        default=0.55,
    )

    return RegimeFeatureSnapshot(
        as_of=str(as_of or ""),
        market=str(market or ""),
        universe_key=str(universe_key or ""),
        average_return=average_return,
        average_volatility=average_volatility,
        breadth=breadth,
        momentum_share=_share(momentum_flags),
        breakout_ready_share=_share(breakout_flags),
        fake_breakout_share=_share(fake_breakout_flags),
        median_drawdown=float(median(drawdowns)) if drawdowns else 0.0,
        average_liquidity=float(sum(liquidity_scores) / len(liquidity_scores)) if liquidity_scores else 0.50,
        average_volume_confirmation=(
            float(sum(volume_confirmations) / len(volume_confirmations))
            if volume_confirmations
            else 0.0
        ),
        macro_score=_macro_score(macro_verdict),
        macro_target_gross_exposure=macro_target_gross_exposure,
        sample_count=sample_count,
        diagnostics=diagnostics,
        metadata={
            "frame_count": frame_count,
            "dated_frame_count": dated_frame_count,
            "tradability_market_state_count": len(states),
            "no_network": True,
            "no_llm": True,
        },
    )
