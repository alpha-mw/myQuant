from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Sequence


TREND_STATES = (
    "warming",
    "cooling",
    "stable",
    "spike_unconfirmed",
    "insufficient_history",
)


@dataclass(frozen=True)
class ThemeSmoothingConfig:
    window: int = 10
    min_observations: int = 5
    delta_lag: int = 5
    persistence_score_floor: float = 55.0
    warming_delta: float = 3.0
    cooling_delta: float = -3.0
    spike_delta: float = 12.0


@dataclass(frozen=True)
class ThemeSmoothingResult:
    raw_score: float | None = None
    smoothed_score: float | None = None
    heat_10d: float | None = None
    heat_delta_5d: float | None = None
    persistence_count: int = 0
    trend_state: str = "insufficient_history"
    observation_count: int = 0
    status: str = "insufficient_history"
    diagnostic_notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_score": self.raw_score,
            "smoothed_score": self.smoothed_score,
            "heat_10d": self.heat_10d,
            "heat_delta_5d": self.heat_delta_5d,
            "persistence_count": int(self.persistence_count),
            "trend_state": self.trend_state,
            "observation_count": int(self.observation_count),
            "status": self.status,
            "diagnostic_notes": list(self.diagnostic_notes),
        }


def smooth_theme_series(
    scores: Sequence[Any],
    config: ThemeSmoothingConfig | None = None,
) -> ThemeSmoothingResult:
    settings = config or ThemeSmoothingConfig()
    window = max(int(settings.window or 10), 1)
    min_observations = max(int(settings.min_observations or 5), 1)
    values, diagnostics = _clean_score_series(scores)
    if not values:
        return ThemeSmoothingResult(
            status="unavailable",
            diagnostic_notes=tuple(diagnostics or ["theme_score_history_missing"]),
        )

    raw_score = values[-1]
    observation_count = len(values)
    if observation_count < min_observations:
        return ThemeSmoothingResult(
            raw_score=raw_score,
            observation_count=observation_count,
            status="insufficient_history",
            trend_state="insufficient_history",
            diagnostic_notes=tuple(
                [
                    *diagnostics,
                    f"theme_smoothing_observations_below_min:{observation_count}/{min_observations}",
                ]
            ),
        )

    window_values = values[-window:]
    smoothed = _round_score(mean(window_values))
    delta = _delta_5d(values, settings)
    persistence_count = sum(
        1 for value in window_values if value >= float(settings.persistence_score_floor)
    )
    trend_state = _trend_state(
        raw_score=raw_score,
        smoothed_score=smoothed,
        heat_delta_5d=delta,
        config=settings,
    )
    return ThemeSmoothingResult(
        raw_score=raw_score,
        smoothed_score=smoothed,
        heat_10d=smoothed,
        heat_delta_5d=delta,
        persistence_count=persistence_count,
        trend_state=trend_state,
        observation_count=observation_count,
        status="success",
        diagnostic_notes=tuple(diagnostics),
    )


def smooth_numeric_series(
    values: Sequence[Any],
    *,
    lower: float,
    upper: float,
    config: ThemeSmoothingConfig | None = None,
) -> float | None:
    settings = config or ThemeSmoothingConfig()
    cleaned: list[float] = []
    for value in values:
        numeric = _safe_float(value)
        if numeric is None:
            continue
        cleaned.append(max(float(lower), min(float(upper), numeric)))
    if len(cleaned) < max(int(settings.min_observations or 5), 1):
        return None
    return float(round(mean(cleaned[-max(int(settings.window or 10), 1):]), 6))


def _clean_score_series(scores: Sequence[Any]) -> tuple[list[float], list[str]]:
    values: list[float] = []
    diagnostics: list[str] = []
    for item in list(scores or []):
        numeric = _safe_float(item)
        if numeric is None:
            diagnostics.append("malformed_theme_score_history_value")
            continue
        values.append(_round_score(max(0.0, min(100.0, numeric))))
    return values, _dedupe(diagnostics)


def _delta_5d(
    values: list[float],
    config: ThemeSmoothingConfig,
) -> float | None:
    lag = max(int(config.delta_lag or 5), 1)
    if len(values) <= lag:
        return None
    previous = values[-lag - 1 : -1]
    if not previous:
        return None
    return _round_score(values[-1] - mean(previous))


def _trend_state(
    *,
    raw_score: float,
    smoothed_score: float,
    heat_delta_5d: float | None,
    config: ThemeSmoothingConfig,
) -> str:
    delta = heat_delta_5d if heat_delta_5d is not None else 0.0
    floor = float(config.persistence_score_floor)
    if raw_score >= floor and smoothed_score < floor and delta >= float(config.spike_delta):
        return "spike_unconfirmed"
    if delta >= float(config.warming_delta):
        return "warming"
    if delta <= float(config.cooling_delta):
        return "cooling"
    return "stable"


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _round_score(value: float) -> float:
    return round(float(value), 6)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result
