from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median
from typing import Any, Mapping

import pandas as pd

from quant_investor.themes.policy import PolicyCatalystScanner
from quant_investor.themes.smoothing import ThemeSmoothingConfig, smooth_theme_series
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
        smoothing_window: int = 10,
        smoothing_min_observations: int = 5,
        policy_catalyst_enabled: bool | None = None,
        policy_catalyst_weight: float | None = None,
        policy_lookback_days: int | None = None,
        policy_event_path: str | None = None,
    ) -> ThemeScanResult:
        state_map = symbol_market_state or {}
        min_count = max(0, int(min_member_count))
        limit = max(0, int(top_n))
        policy_config = _resolve_policy_config(
            policy_catalyst_enabled=policy_catalyst_enabled,
            policy_catalyst_weight=policy_catalyst_weight,
            policy_lookback_days=policy_lookback_days,
            policy_event_path=policy_event_path,
        )
        smoothing_config = ThemeSmoothingConfig(
            window=max(int(smoothing_window or 10), 1),
            min_observations=max(int(smoothing_min_observations or 5), 1),
        )
        themes: dict[str, dict[str, Any]] = {}
        members_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
        history_members_by_theme: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
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
            for offset in range(smoothing_config.window):
                if offset == 0:
                    historical_metrics = dict(metrics)
                else:
                    historical_metrics = _symbol_metrics_at_offset(frame, state, offset)
                    historical_metrics["symbol"] = symbol
                history_members_by_theme[theme_id][offset].append(historical_metrics)
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

        policy_metadata = _apply_policy_catalysts(
            theme_scores=scored,
            members_by_theme=members_by_theme,
            as_of=as_of,
            enabled=policy_config["enabled"],
            weight=policy_config["weight"],
            lookback_days=policy_config["lookback_days"],
            event_path=policy_config["event_path"],
        )
        selected = sorted(scored, key=lambda item: (-item.score, item.theme_id))[:limit]
        for theme_score in selected:
            _apply_smoothing_to_theme_score(
                theme_score=theme_score,
                history_members_by_offset=history_members_by_theme.get(theme_score.theme_id, {}),
                min_member_count=min_count,
                config=smoothing_config,
            )
        theme_scores = {score.theme_id: score for score in selected}

        symbol_scores: dict[str, float] = {}
        symbol_smoothed_scores: dict[str, float] = {}
        symbol_primary_theme: dict[str, str] = {}
        symbol_phase: dict[str, str] = {}
        symbol_risk_flags: dict[str, list[str]] = {}
        for theme_score in selected:
            for metrics in members_by_theme.get(theme_score.theme_id, []):
                symbol = str(metrics["symbol"])
                symbol_scores[symbol] = clamp(theme_score.score / 100.0)
                if theme_score.smoothed_score is not None:
                    symbol_smoothed_scores[symbol] = clamp(theme_score.smoothed_score / 100.0)
                symbol_primary_theme[symbol] = theme_score.theme_id
                symbol_phase[symbol] = theme_score.phase.value
                symbol_risk_flags[symbol] = list(theme_score.risk_flags)

        return ThemeScanResult(
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            theme_scores=theme_scores,
            symbol_scores=symbol_scores,
            symbol_smoothed_scores=symbol_smoothed_scores,
            symbol_primary_theme=symbol_primary_theme,
            symbol_phase=symbol_phase,
            symbol_risk_flags=symbol_risk_flags,
            metadata={
                "theme_count": len(theme_scores),
                "scanned_symbol_count": scanned_symbol_count,
                "member_count_min": min_count,
                "top_n": limit,
                "smoothing_method": "sma",
                "smoothing_window": smoothing_config.window,
                "smoothing_min_observations": smoothing_config.min_observations,
                "policy_catalyst_status": policy_metadata["status"],
                "policy_catalyst_enabled": policy_metadata["enabled"],
                "policy_catalyst_weight": policy_metadata["weight"],
                "policy_catalyst_event_path": policy_metadata["event_path"],
                "policy_catalyst_matched_theme_count": policy_metadata["matched_theme_count"],
                "policy_catalyst_diagnostic_notes": policy_metadata["diagnostic_notes"],
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


def _symbol_metrics_at_offset(
    frame: pd.DataFrame | None,
    state: Mapping[str, Any] | None,
    trailing_offset: int,
) -> dict[str, Any]:
    if trailing_offset <= 0:
        return _symbol_metrics(frame, state)
    prefix = _frame_prefix(frame, trailing_offset)
    if prefix is None or prefix.empty:
        return _neutral_symbol_metrics()
    return _symbol_metrics(prefix, {})


def _frame_prefix(frame: pd.DataFrame | None, trailing_offset: int) -> pd.DataFrame | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    ordered = frame
    for date_col in _DATE_COLUMNS:
        if date_col in frame.columns:
            try:
                ordered = frame.sort_values(date_col, kind="mergesort")
            except Exception:
                ordered = frame
            break
    keep_count = len(ordered) - max(int(trailing_offset), 0)
    if keep_count <= 0:
        return None
    return ordered.iloc[:keep_count]


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


def _apply_smoothing_to_theme_score(
    *,
    theme_score: ThemeScore,
    history_members_by_offset: Mapping[int, list[dict[str, Any]]],
    min_member_count: int,
    config: ThemeSmoothingConfig,
) -> None:
    score_history: list[float] = []
    for offset in sorted(history_members_by_offset, reverse=True):
        members = list(history_members_by_offset.get(offset, []) or [])
        if len(members) < min_member_count:
            continue
        if _median_metric(members, "data_coverage") < 0.50:
            continue
        try:
            historical_score = _score_theme(
                theme_id=theme_score.theme_id,
                theme_name=theme_score.theme_name,
                members=members,
                min_member_count=min_member_count,
            )
        except Exception:
            continue
        score_history.append(float(historical_score.score))

    smoothing = smooth_theme_series(score_history, config)
    theme_score.raw_score = float(theme_score.score)
    theme_score.smoothed_score = smoothing.smoothed_score
    theme_score.heat_10d = smoothing.heat_10d
    theme_score.heat_delta_5d = smoothing.heat_delta_5d
    theme_score.persistence_count = int(smoothing.persistence_count)
    theme_score.trend_state = smoothing.trend_state
    theme_score.smoothing_observation_count = int(smoothing.observation_count)
    theme_score.smoothing_status = smoothing.status
    theme_score.metadata = {
        **dict(theme_score.metadata or {}),
        "smoothing_method": "sma",
        "smoothing_window": int(config.window),
        "smoothing_min_observations": int(config.min_observations),
        "smoothing_diagnostic_notes": list(smoothing.diagnostic_notes),
    }


def _resolve_policy_config(
    *,
    policy_catalyst_enabled: bool | None,
    policy_catalyst_weight: float | None,
    policy_lookback_days: int | None,
    policy_event_path: str | None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": False,
        "weight": 0.16,
        "lookback_days": 30,
        "event_path": "data/theme_policy_events.jsonl",
    }
    try:
        from quant_investor.config import Config

        defaults.update(
            {
                "enabled": bool(getattr(Config, "THEME_POLICY_CATALYST_ENABLED", False)),
                "weight": _safe_float(
                    getattr(Config, "THEME_POLICY_CATALYST_WEIGHT", 0.16),
                    0.16,
                ),
                "lookback_days": max(
                    int(getattr(Config, "THEME_POLICY_LOOKBACK_DAYS", 30) or 30),
                    1,
                ),
                "event_path": str(
                    getattr(
                        Config,
                        "THEME_POLICY_EVENT_PATH",
                        "data/theme_policy_events.jsonl",
                    )
                    or "data/theme_policy_events.jsonl"
                ),
            }
        )
    except Exception:
        pass

    if policy_catalyst_enabled is not None:
        defaults["enabled"] = bool(policy_catalyst_enabled)
    if policy_catalyst_weight is not None:
        defaults["weight"] = _safe_float(policy_catalyst_weight, defaults["weight"])
    if policy_lookback_days is not None:
        defaults["lookback_days"] = max(int(policy_lookback_days or 30), 1)
    if policy_event_path is not None:
        defaults["event_path"] = str(policy_event_path or defaults["event_path"])
    defaults["weight"] = clamp(defaults["weight"])
    return defaults


def _apply_policy_catalysts(
    *,
    theme_scores: list[ThemeScore],
    members_by_theme: Mapping[str, list[dict[str, Any]]],
    as_of: str,
    enabled: bool,
    weight: float,
    lookback_days: int,
    event_path: str,
) -> dict[str, Any]:
    metadata = {
        "enabled": bool(enabled),
        "status": "disabled",
        "weight": clamp(weight),
        "event_path": str(event_path or ""),
        "matched_theme_count": 0,
        "diagnostic_notes": [],
    }
    if not enabled:
        return metadata

    scanner = PolicyCatalystScanner(
        event_path=event_path or "data/theme_policy_events.jsonl",
        lookback_days=lookback_days,
    )
    events = scanner.load_events()
    metadata["diagnostic_notes"] = list(scanner.diagnostic_notes)
    if scanner.status != "success":
        metadata["status"] = "unavailable"
        for theme_score in theme_scores:
            theme_score.policy_stage = "unavailable"
        return metadata

    metadata["status"] = "success"
    matched = 0
    for theme_score in theme_scores:
        member_symbols = [
            str(member.get("symbol", ""))
            for member in list(members_by_theme.get(theme_score.theme_id, []) or [])
            if str(member.get("symbol", "")).strip()
        ]
        catalyst = scanner.score_theme(
            theme_id=theme_score.theme_id,
            theme_name=theme_score.theme_name,
            member_symbols=member_symbols,
            as_of=as_of,
            events=events,
        )
        _apply_policy_score_to_theme(theme_score, catalyst.to_dict(), weight=weight)
        if catalyst.policy_stage not in {"no_match", "unavailable"}:
            matched += 1

    metadata["matched_theme_count"] = matched
    return metadata


def _apply_policy_score_to_theme(
    theme_score: ThemeScore,
    catalyst: Mapping[str, Any],
    *,
    weight: float,
) -> None:
    policy_score = clamp(_safe_float(catalyst.get("policy_score"), 0.0))
    policy_boost = policy_score * clamp(weight) * 100.0
    theme_score.score = clamp(_safe_float(theme_score.score, 0.0) + policy_boost, 0.0, 100.0)
    theme_score.policy_catalyst_score = policy_score
    theme_score.policy_confidence = clamp(_safe_float(catalyst.get("confidence"), 0.0))
    theme_score.policy_stage = str(catalyst.get("policy_stage") or "no_match")
    theme_score.policy_evidence = [
        str(item)
        for item in list(catalyst.get("evidence", []) or [])
        if str(item).strip()
    ]
    theme_score.policy_risk_flags = [
        str(flag)
        for flag in list(catalyst.get("risk_flags", []) or [])
        if str(flag).startswith("policy_")
    ]
    theme_score.risk_flags = _dedupe_texts(
        [*list(theme_score.risk_flags or []), *theme_score.policy_risk_flags]
    )
    theme_score.metadata = {
        **dict(theme_score.metadata or {}),
        "policy_catalyst": {
            "policy_score": policy_score,
            "confidence": theme_score.policy_confidence,
            "stage": theme_score.policy_stage,
            "score_component": policy_boost,
            "weight_cap": clamp(weight),
        },
    }


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


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
