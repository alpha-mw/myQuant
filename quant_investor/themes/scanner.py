from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median
from typing import Any, Mapping

import pandas as pd

from quant_investor.themes.types import ThemePhase, ThemeScanResult, ThemeScore, clamp


_INVALID_THEME_NAMES = {"", "unknown", "none", "nan", "null"}
_DATE_COLUMNS = ("trade_date", "date", "Date", "datetime", "time")
_CLOSE_COLUMNS = ("close", "Close")
_VOLUME_COLUMNS = ("volume", "vol")


class ThemeScanner:
    def scan(
        self,
        *,
        frames: Mapping[str, pd.DataFrame],
        industry_map: Mapping[str, str],
        symbol_market_state: Mapping[str, Mapping[str, Any]] | None = None,
        market: str = "CN",
        universe_key: str = "",
        as_of: str = "",
        min_member_count: int = 5,
        top_n: int = 20,
    ) -> ThemeScanResult:
        state_map = symbol_market_state or {}
        min_count = max(0, int(min_member_count))
        limit = max(0, int(top_n))
        themes: dict[str, dict[str, Any]] = {}
        members_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scanned_symbol_count = 0

        for symbol in sorted(industry_map):
            theme_name = _valid_theme_name(industry_map.get(symbol))
            if theme_name is None:
                continue
            theme_id = f"industry::{_normalize_theme_id(theme_name)}"
            themes.setdefault(theme_id, {"theme_id": theme_id, "theme_name": theme_name})
            state = state_map.get(symbol, {})
            frame = frames.get(symbol)
            try:
                metrics = _symbol_metrics(frame, state)
            except Exception:
                metrics = _neutral_symbol_metrics()
            metrics["symbol"] = symbol
            members_by_theme[theme_id].append(metrics)
            scanned_symbol_count += 1

        scored: list[ThemeScore] = []
        for theme_id in sorted(members_by_theme):
            members = members_by_theme[theme_id]
            if len(members) < min_count:
                continue
            scored.append(
                _score_theme(
                    theme_id=theme_id,
                    theme_name=themes.get(theme_id, {}).get("theme_name", theme_id),
                    members=members,
                    min_member_count=min_count,
                )
            )

        selected = sorted(scored, key=lambda item: (-item.score, item.theme_id))[:limit]
        theme_scores = {score.theme_id: score for score in selected}

        symbol_scores: dict[str, float] = {}
        symbol_primary_theme: dict[str, str] = {}
        symbol_phase: dict[str, str] = {}
        symbol_risk_flags: dict[str, list[str]] = {}
        for theme_score in selected:
            for metrics in members_by_theme.get(theme_score.theme_id, []):
                symbol = str(metrics["symbol"])
                symbol_scores[symbol] = clamp(theme_score.score / 100.0)
                symbol_primary_theme[symbol] = theme_score.theme_id
                symbol_phase[symbol] = theme_score.phase.value
                symbol_risk_flags[symbol] = list(theme_score.risk_flags)

        return ThemeScanResult(
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            theme_scores=theme_scores,
            symbol_scores=symbol_scores,
            symbol_primary_theme=symbol_primary_theme,
            symbol_phase=symbol_phase,
            symbol_risk_flags=symbol_risk_flags,
            metadata={
                "theme_count": len(theme_scores),
                "scanned_symbol_count": scanned_symbol_count,
                "member_count_min": min_count,
                "top_n": limit,
                "deterministic": True,
                "no_llm": True,
                "no_network": True,
            },
        )


def _valid_theme_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _INVALID_THEME_NAMES:
        return None
    return text


def _normalize_theme_id(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9a-z_\-\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unclassified"


def _symbol_metrics(
    frame: pd.DataFrame | None,
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metrics = _neutral_symbol_metrics()
    state = state if isinstance(state, Mapping) else {}
    close, volume = _ordered_series(frame)

    if close:
        metrics["return_3d"] = _window_return(close, 3)
        metrics["return_5d"] = _window_return(close, 5)
        metrics["return_20d"] = _window_return(close, 20)
        metrics["has_ma20"] = len(close) >= 20
        metrics["above_ma20"] = bool(len(close) >= 20 and close[-1] > _mean(close[-20:]))
        metrics["data_coverage"] = clamp(len(close) / 21.0)
        metrics["fake_breakout_proxy"] = _fake_breakout_proxy(close)
    else:
        metrics["return_3d"] = _state_float(state, "return_3d", 0.0)
        metrics["return_5d"] = _state_float(state, "return_5d", 0.0)
        metrics["return_20d"] = _state_float(state, "return_20d", 0.0)

    if volume:
        tail = volume[-20:]
        average_volume = _mean(tail)
        if average_volume > 0:
            ratio = volume[-1] / average_volume
            metrics["volume_ratio"] = ratio
            metrics["symbol_volume_confirmation"] = clamp((ratio - 1.0) / 1.5)

    state_fake_risk = _state_float(state, "fake_breakout_risk", math.nan)
    metrics["fake_breakout_risk"] = (
        clamp(state_fake_risk)
        if math.isfinite(state_fake_risk)
        else metrics["fake_breakout_proxy"]
    )
    return metrics


def _neutral_symbol_metrics() -> dict[str, Any]:
    return {
        "return_3d": 0.0,
        "return_5d": 0.0,
        "return_20d": 0.0,
        "above_ma20": False,
        "has_ma20": False,
        "volume_ratio": None,
        "symbol_volume_confirmation": 0.0,
        "fake_breakout_proxy": 0.0,
        "fake_breakout_risk": 0.0,
        "data_coverage": 0.0,
    }


def _ordered_series(frame: pd.DataFrame | None) -> tuple[list[float], list[float]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return [], []

    close_col = _first_column(frame, _CLOSE_COLUMNS)
    if close_col is None:
        return [], []

    ordered = frame
    for date_col in _DATE_COLUMNS:
        if date_col in frame.columns:
            try:
                ordered = frame.sort_values(date_col, kind="mergesort")
            except Exception:
                ordered = frame
            break

    close = _finite_values(ordered[close_col])
    volume_col = _first_column(ordered, _VOLUME_COLUMNS)
    volume = _finite_values(ordered[volume_col]) if volume_col is not None else []
    return close, [value for value in volume if value >= 0.0]


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _finite_values(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    return [float(value) for value in values if math.isfinite(float(value))]


def _window_return(values: list[float], lookback: int) -> float:
    if len(values) <= lookback:
        return 0.0
    start = values[-1 - lookback]
    end = values[-1]
    if not math.isfinite(start) or not math.isfinite(end) or start <= 0:
        return 0.0
    return end / start - 1.0


def _score_theme(
    *,
    theme_id: str,
    theme_name: str,
    members: list[dict[str, Any]],
    min_member_count: int,
) -> ThemeScore:
    member_count = len(members)
    return_20d = _median_metric(members, "return_20d")
    return_5d = _median_metric(members, "return_5d")
    momentum = clamp((return_20d + 0.10) / 0.30)
    volume_confirmation = _theme_volume_confirmation(members)
    acceleration_base = clamp(((return_5d - return_20d / 4.0) + 0.03) / 0.12)
    acceleration = clamp(0.75 * acceleration_base + 0.25 * volume_confirmation)
    breadth = _theme_breadth(members)
    fake_breakout_risk = _median_metric(members, "fake_breakout_risk")
    overextension_risk = _theme_overextension_risk(return_5d, breadth, volume_confirmation)
    raw = (
        0.30 * momentum
        + 0.24 * acceleration
        + 0.22 * breadth
        + 0.16 * volume_confirmation
        - 0.08 * fake_breakout_risk
        - 0.10 * overextension_risk
    )
    score = clamp(raw, 0.0, 1.0) * 100.0
    data_coverage = _median_metric(members, "data_coverage")
    confidence = clamp(
        0.25
        + min(member_count, 20) / 20.0 * 0.30
        + breadth * 0.20
        + min(data_coverage, 1.0) * 0.20
        - fake_breakout_risk * 0.10,
        0.0,
        0.90,
    )
    phase = _infer_phase(
        score=score,
        breadth=breadth,
        acceleration=acceleration,
        volume_confirmation=volume_confirmation,
        overextension_risk=overextension_risk,
        fake_breakout_risk=fake_breakout_risk,
        momentum=momentum,
    )
    risk_flags = _risk_flags(
        score=score,
        breadth=breadth,
        overextension_risk=overextension_risk,
        fake_breakout_risk=fake_breakout_risk,
        member_count=member_count,
        min_member_count=min_member_count,
    )
    top_symbols = _top_symbols(members)

    evidence = [
        f"member_count={member_count}",
        f"momentum={momentum:.3f}",
        f"breadth={breadth:.3f}",
        f"volume_confirmation={volume_confirmation:.3f}",
    ]
    return ThemeScore(
        theme_id=theme_id,
        theme_name=theme_name,
        phase=phase,
        score=score,
        confidence=confidence,
        member_count=member_count,
        breadth=breadth,
        momentum=momentum,
        acceleration=acceleration,
        volume_confirmation=volume_confirmation,
        overextension_risk=overextension_risk,
        fake_breakout_risk=fake_breakout_risk,
        top_symbols=top_symbols,
        risk_flags=risk_flags,
        evidence=evidence,
        metadata={"theme_return_5d": return_5d, "theme_return_20d": return_20d},
    )


def _median_metric(members: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(member.get(key), 0.0) for member in members]
    return float(median(values)) if values else 0.0


def _theme_volume_confirmation(members: list[dict[str, Any]]) -> float:
    ratios = [
        _safe_float(member.get("volume_ratio"), math.nan)
        for member in members
        if member.get("volume_ratio") is not None
    ]
    ratios = [ratio for ratio in ratios if math.isfinite(ratio)]
    if not ratios:
        return 0.0
    return clamp((float(median(ratios)) - 1.0) / 1.5)


def _theme_breadth(members: list[dict[str, Any]]) -> float:
    if not members:
        return 0.0
    member_count = len(members)
    positive_20d = sum(1 for member in members if _safe_float(member.get("return_20d"), 0.0) > 0)
    positive_5d = sum(1 for member in members if _safe_float(member.get("return_5d"), 0.0) > 0)
    ma_members = [member for member in members if bool(member.get("has_ma20"))]
    ma_ratio = (
        sum(1 for member in ma_members if bool(member.get("above_ma20"))) / len(ma_members)
        if ma_members
        else 0.0
    )
    return clamp(((positive_20d / member_count) + (positive_5d / member_count) + ma_ratio) / 3.0)


def _theme_overextension_risk(
    theme_return_5d: float,
    breadth: float,
    volume_confirmation: float,
) -> float:
    risk = clamp((theme_return_5d - 0.08) / 0.12)
    if volume_confirmation > 0.85 and breadth < 0.50:
        risk = clamp(risk + 0.15)
    return risk


def _infer_phase(
    *,
    score: float,
    breadth: float,
    acceleration: float,
    volume_confirmation: float,
    overextension_risk: float,
    fake_breakout_risk: float,
    momentum: float,
) -> ThemePhase:
    if overextension_risk >= 0.70:
        return ThemePhase.OVEREXTENDED
    if fake_breakout_risk >= 0.70 and momentum < 0.45:
        return ThemePhase.DISTRIBUTION
    if score < 35:
        return ThemePhase.UNCLASSIFIED
    if score >= 70 and breadth >= 0.55 and overextension_risk < 0.60:
        return ThemePhase.CONFIRMED_ROTATION
    if score >= 55 and acceleration >= 0.55 and volume_confirmation >= 0.35:
        return ThemePhase.EARLY_ACCELERATION
    if score >= 40 and breadth >= 0.40:
        return ThemePhase.ACCUMULATION
    return ThemePhase.UNCLASSIFIED


def _risk_flags(
    *,
    score: float,
    breadth: float,
    overextension_risk: float,
    fake_breakout_risk: float,
    member_count: int,
    min_member_count: int,
) -> list[str]:
    flags: list[str] = []
    if overextension_risk >= 0.70:
        flags.append("theme_overextended")
    if fake_breakout_risk >= 0.65:
        flags.append("theme_fake_breakout_risk")
    if score >= 55 and breadth < 0.35:
        flags.append("theme_low_breadth")
    if member_count < min_member_count:
        flags.append("theme_low_member_count")
    return flags


def _top_symbols(members: list[dict[str, Any]]) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for member in members:
        return_20d = clamp((_safe_float(member.get("return_20d"), 0.0) + 0.10) / 0.30)
        return_5d = clamp((_safe_float(member.get("return_5d"), 0.0) + 0.03) / 0.12)
        volume_confirmation = clamp(_safe_float(member.get("symbol_volume_confirmation"), 0.0))
        fake_breakout_risk = clamp(_safe_float(member.get("fake_breakout_risk"), 0.0))
        symbol_score = (
            0.45 * return_20d
            + 0.25 * return_5d
            + 0.20 * volume_confirmation
            - 0.10 * fake_breakout_risk
        )
        ranked.append((clamp(symbol_score), str(member.get("symbol", ""))))
    return [symbol for _, symbol in sorted(ranked, key=lambda item: (-item[0], item[1]))[:5]]


def _fake_breakout_proxy(close: list[float]) -> float:
    if not close:
        return 0.0
    latest = close[-1]
    high_60 = max(close[-60:])
    recent_peak = max(close[-10:])
    if high_60 <= 0 or recent_peak <= 0:
        return 0.0
    distance_from_high = max(0.0, high_60 - latest) / high_60
    recent_pullback = max(0.0, recent_peak - latest) / recent_peak
    distance_risk = clamp((distance_from_high - 0.03) / 0.12)
    pullback_risk = clamp((recent_pullback - 0.02) / 0.10)
    return clamp(0.60 * distance_risk + 0.40 * pullback_risk)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _state_float(state: Mapping[str, Any], key: str, default: float) -> float:
    if not isinstance(state, Mapping):
        return default
    return _safe_float(state.get(key), default)
