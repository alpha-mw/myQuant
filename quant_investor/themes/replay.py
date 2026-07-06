from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_investor.themes.storage import ThemeSnapshotStore


_DATE_COLUMNS = ("trade_date", "date")
_CLOSE_COLUMNS = ("close", "Close")
_DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
_DEFAULT_BENCHMARK_HORIZONS = (5, 10, 20)
PIT_INDUSTRY_LABEL_NOTE = (
    "industry labels are as-of run date, not point-in-time; "
    "replay carries mild reclassification look-ahead"
)
_BASE_METADATA = {
    "deterministic": True,
    "no_llm": True,
    "no_network": True,
    "pit_industry_labels": False,
    "industry_label_note": PIT_INDUSTRY_LABEL_NOTE,
}


@dataclass
class ThemeReplayRecord:
    snapshot_path: str = ""
    market: str = "CN"
    universe_key: str = ""
    as_of: str = ""
    symbol: str = ""
    primary_theme_id: str = ""
    primary_theme_name: str = ""
    theme_type: str = "industry"
    membership_source: str = "industry_map"
    pit_membership: bool = False
    theme_memberships: list[str] = field(default_factory=list)
    phase: str = ""
    symbol_theme_score: float = 0.0
    theme_score: float | None = None
    theme_confidence: float | None = None
    theme_member_count: int | None = None
    risk_flags: list[str] = field(default_factory=list)

    forward_return_1d: float | None = None
    forward_return_3d: float | None = None
    forward_return_5d: float | None = None
    forward_return_10d: float | None = None
    forward_return_20d: float | None = None

    forward_alpha_5d: float | None = None
    forward_alpha_10d: float | None = None
    forward_alpha_20d: float | None = None

    max_drawdown_10d: float | None = None
    max_runup_10d: float | None = None
    hit_5d: bool | None = None
    hit_10d: bool | None = None
    data_available: bool = False
    unavailable_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThemeCalibrationDataset:
    records: list[ThemeReplayRecord] = field(default_factory=list)
    theme_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    phase_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_flag_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "theme_summary": dict(self.theme_summary),
            "phase_summary": dict(self.phase_summary),
            "risk_flag_summary": dict(self.risk_flag_summary),
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([record.to_dict() for record in self.records])

    def summary_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for summary_type, payload in (
            ("theme", self.theme_summary),
            ("phase", self.phase_summary),
            ("risk_flag", self.risk_flag_summary),
        ):
            for key in sorted(payload):
                rows.append(
                    {
                        "summary_type": summary_type,
                        "key": key,
                        **dict(payload.get(key, {}) or {}),
                    }
                )
        return pd.DataFrame(rows)

    def to_markdown(self, *, max_rows: int = 20) -> str:
        available_count = sum(1 for record in self.records if record.data_available)
        lines = [
            "## Theme Replay Calibration Dataset",
            "",
            f"Record count: {len(self.records)}",
            f"Available forward data count: {available_count}",
            f"Industry label note: {PIT_INDUSTRY_LABEL_NOTE}",
            "",
            "### Phase Summary",
        ]
        lines.extend(_summary_lines(self.phase_summary, max_rows=max_rows))
        lines.extend(["", "### Theme Summary"])
        lines.extend(_summary_lines(self.theme_summary, max_rows=max_rows))
        lines.extend(["", "### Risk Flag Summary"])
        lines.extend(_summary_lines(self.risk_flag_summary, max_rows=max_rows))
        return "\n".join(lines)


def normalize_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    close_column = _first_column(frame, _CLOSE_COLUMNS)
    if close_column is None:
        return pd.DataFrame()

    normalized = frame.copy(deep=True)
    date_column = _first_column(normalized, _DATE_COLUMNS)
    if date_column is not None:
        try:
            normalized = normalized.sort_values(date_column, kind="mergesort")
        except Exception:
            normalized = normalized.copy(deep=True)

    normalized["close"] = pd.to_numeric(normalized[close_column], errors="coerce")
    normalized = normalized.dropna(subset=["close"]).reset_index(drop=True)
    return normalized


def find_as_of_index(frame: pd.DataFrame, as_of: str) -> int | None:
    normalized_as_of = _date_key(as_of)
    if not normalized_as_of:
        return None

    normalized = normalize_symbol_frame(frame)
    if normalized.empty:
        return None

    date_column = _first_column(normalized, _DATE_COLUMNS)
    if date_column is None:
        return None

    best_index: int | None = None
    for idx, value in enumerate(normalized[date_column].tolist()):
        date_key = _date_key(value)
        if not date_key:
            continue
        if date_key == normalized_as_of:
            return idx
        if date_key <= normalized_as_of:
            best_index = idx
    return best_index


def compute_forward_metrics(
    *,
    frame: pd.DataFrame,
    as_of: str,
    horizons: tuple[int, ...] = _DEFAULT_HORIZONS,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        f"forward_return_{int(horizon)}d": None
        for horizon in horizons
    }
    metrics.update(
        {
            "max_drawdown_10d": None,
            "max_runup_10d": None,
            "data_available": False,
            "unavailable_reason": "",
        }
    )

    normalized = normalize_symbol_frame(frame)
    if normalized.empty:
        metrics["unavailable_reason"] = "missing_close"
        return metrics

    as_of_index = find_as_of_index(normalized, as_of)
    if as_of_index is None:
        metrics["unavailable_reason"] = "missing_as_of"
        return metrics

    close_values = [
        float(value)
        for value in pd.to_numeric(normalized["close"], errors="coerce").tolist()
        if _is_finite(value)
    ]
    if as_of_index >= len(close_values):
        metrics["unavailable_reason"] = "missing_as_of"
        return metrics

    start_price = close_values[as_of_index]
    if not _is_finite(start_price) or start_price <= 0.0:
        metrics["unavailable_reason"] = "invalid_start_price"
        return metrics

    for horizon in horizons:
        horizon_int = int(horizon)
        future_index = as_of_index + horizon_int
        key = f"forward_return_{horizon_int}d"
        if horizon_int <= 0 or future_index >= len(close_values):
            metrics[key] = None
            continue
        metrics[key] = _clean_float(close_values[future_index] / start_price - 1.0)

    window = close_values[as_of_index : min(len(close_values), as_of_index + 11)]
    if len(window) >= 2:
        forward_path = [_clean_float(value / start_price - 1.0) for value in window]
        metrics["max_drawdown_10d"] = _clean_float(min(forward_path))
        metrics["max_runup_10d"] = _clean_float(max(forward_path))

    metrics["data_available"] = any(
        metrics.get(f"forward_return_{int(horizon)}d") is not None
        for horizon in horizons
    )
    if not metrics["data_available"]:
        metrics["unavailable_reason"] = "missing_future_rows"
    return metrics


def build_benchmark_forward_returns(
    *,
    frames: Mapping[str, pd.DataFrame],
    as_of: str,
    horizons: tuple[int, ...] = _DEFAULT_BENCHMARK_HORIZONS,
) -> dict[int, float]:
    returns_by_horizon: dict[int, list[float]] = {int(horizon): [] for horizon in horizons}
    for frame in dict(frames or {}).values():
        metrics = compute_forward_metrics(
            frame=frame,
            as_of=as_of,
            horizons=tuple(int(horizon) for horizon in horizons),
        )
        for horizon in returns_by_horizon:
            value = metrics.get(f"forward_return_{horizon}d")
            if _is_finite(value):
                returns_by_horizon[horizon].append(float(value))
    return {
        horizon: _median(values) if values else 0.0
        for horizon, values in returns_by_horizon.items()
    }


def extract_snapshot_theme_rows(snapshot_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        rotation = _theme_rotation_payload(snapshot_payload)
        if not isinstance(rotation, Mapping):
            return []
        symbol_scores = rotation.get("symbol_scores")
        if not isinstance(symbol_scores, Mapping):
            return []
        primary_theme_map = _mapping_or_empty(rotation.get("symbol_primary_theme"))
        phase_map = _mapping_or_empty(rotation.get("symbol_phase"))
        risk_flag_map = _mapping_or_empty(rotation.get("symbol_risk_flags"))
        membership_map = _mapping_or_empty(rotation.get("symbol_theme_memberships"))
        theme_scores = _mapping_or_empty(rotation.get("theme_scores"))

        rows: list[dict[str, Any]] = []
        for symbol in sorted(symbol_scores):
            symbol_text = str(symbol or "")
            if not symbol_text:
                continue
            theme_id = str(primary_theme_map.get(symbol_text) or "")
            theme_payload = _mapping_or_empty(theme_scores.get(theme_id))
            rows.append(
                {
                    "symbol": symbol_text,
                    "symbol_theme_score": _safe_float(
                        symbol_scores.get(symbol_text),
                        0.0,
                    ),
                    "primary_theme_id": theme_id,
                    "primary_theme_name": str(
                        theme_payload.get("theme_name") or theme_id
                    ),
                    "theme_type": str(theme_payload.get("theme_type") or "industry"),
                    "membership_source": str(
                        theme_payload.get("membership_source") or "industry_map"
                    ),
                    "pit_membership": bool(theme_payload.get("pit_membership", False)),
                    "theme_memberships": _string_list(membership_map.get(symbol_text)),
                    "phase": str(phase_map.get(symbol_text) or ""),
                    "risk_flags": _string_list(risk_flag_map.get(symbol_text)),
                    "theme_score": _optional_float(theme_payload.get("score")),
                    "theme_confidence": _optional_float(theme_payload.get("confidence")),
                    "theme_member_count": _optional_int(
                        theme_payload.get("member_count")
                    ),
                }
            )
        return rows
    except Exception:
        return []


def build_theme_calibration_dataset(
    *,
    snapshots: Iterable[Mapping[str, Any] | str | Path],
    frames: Mapping[str, pd.DataFrame],
    horizons: tuple[int, ...] = _DEFAULT_HORIZONS,
    benchmark_horizons: tuple[int, ...] = _DEFAULT_BENCHMARK_HORIZONS,
) -> ThemeCalibrationDataset:
    records: list[ThemeReplayRecord] = []
    snapshot_count = 0
    malformed_snapshot_count = 0
    missing_frame_count = 0
    frame_map = dict(frames or {})
    horizon_tuple = tuple(int(horizon) for horizon in horizons)
    benchmark_horizon_tuple = tuple(int(horizon) for horizon in benchmark_horizons)

    for snapshot_input in list(snapshots or []):
        snapshot_count += 1
        snapshot_path = ""
        payload: Mapping[str, Any] | None
        if isinstance(snapshot_input, (str, Path)):
            snapshot_path = str(snapshot_input)
            payload = _load_snapshot_path(snapshot_input)
            if payload is None:
                malformed_snapshot_count += 1
                continue
        elif isinstance(snapshot_input, Mapping):
            payload = snapshot_input
        else:
            malformed_snapshot_count += 1
            continue

        rows = extract_snapshot_theme_rows(payload)
        if not rows and not _looks_like_empty_theme_rotation(payload):
            malformed_snapshot_count += 1
            continue

        rotation = _theme_rotation_payload(payload)
        market = str(payload.get("market") or rotation.get("market") or "CN")
        universe_key = str(
            payload.get("universe_key") or rotation.get("universe_key") or ""
        )
        as_of = str(payload.get("as_of") or rotation.get("as_of") or "")
        benchmark_returns = build_benchmark_forward_returns(
            frames=frame_map,
            as_of=as_of,
            horizons=benchmark_horizon_tuple,
        )

        for row in rows:
            symbol = str(row.get("symbol") or "")
            frame = frame_map.get(symbol)
            metrics: dict[str, Any]
            if isinstance(frame, pd.DataFrame):
                metrics = compute_forward_metrics(
                    frame=frame,
                    as_of=as_of,
                    horizons=horizon_tuple,
                )
            else:
                missing_frame_count += 1
                metrics = {
                    f"forward_return_{int(horizon)}d": None
                    for horizon in horizon_tuple
                }
                metrics.update(
                    {
                        "max_drawdown_10d": None,
                        "max_runup_10d": None,
                        "data_available": False,
                        "unavailable_reason": "missing_frame",
                    }
                )

            alpha_5d = _forward_alpha(metrics, benchmark_returns, 5)
            alpha_10d = _forward_alpha(metrics, benchmark_returns, 10)
            alpha_20d = _forward_alpha(metrics, benchmark_returns, 20)
            records.append(
                ThemeReplayRecord(
                    snapshot_path=snapshot_path,
                    market=market,
                    universe_key=universe_key,
                    as_of=as_of,
                    symbol=symbol,
                    primary_theme_id=str(row.get("primary_theme_id") or ""),
                    primary_theme_name=str(row.get("primary_theme_name") or ""),
                    theme_type=str(row.get("theme_type") or "industry"),
                    membership_source=str(row.get("membership_source") or "industry_map"),
                    pit_membership=bool(row.get("pit_membership", False)),
                    theme_memberships=_string_list(row.get("theme_memberships")),
                    phase=str(row.get("phase") or ""),
                    symbol_theme_score=_safe_float(
                        row.get("symbol_theme_score"),
                        0.0,
                    ),
                    theme_score=_optional_float(row.get("theme_score")),
                    theme_confidence=_optional_float(row.get("theme_confidence")),
                    theme_member_count=_optional_int(row.get("theme_member_count")),
                    risk_flags=_string_list(row.get("risk_flags")),
                    forward_return_1d=_optional_float(
                        metrics.get("forward_return_1d")
                    ),
                    forward_return_3d=_optional_float(
                        metrics.get("forward_return_3d")
                    ),
                    forward_return_5d=_optional_float(
                        metrics.get("forward_return_5d")
                    ),
                    forward_return_10d=_optional_float(
                        metrics.get("forward_return_10d")
                    ),
                    forward_return_20d=_optional_float(
                        metrics.get("forward_return_20d")
                    ),
                    forward_alpha_5d=alpha_5d,
                    forward_alpha_10d=alpha_10d,
                    forward_alpha_20d=alpha_20d,
                    max_drawdown_10d=_optional_float(
                        metrics.get("max_drawdown_10d")
                    ),
                    max_runup_10d=_optional_float(metrics.get("max_runup_10d")),
                    hit_5d=(alpha_5d > 0.0 if alpha_5d is not None else None),
                    hit_10d=(alpha_10d > 0.0 if alpha_10d is not None else None),
                    data_available=bool(metrics.get("data_available", False)),
                    unavailable_reason=str(metrics.get("unavailable_reason") or ""),
                    metadata={
                        **_BASE_METADATA,
                        "benchmark_returns": dict(benchmark_returns),
                    },
                )
            )

    dataset = ThemeCalibrationDataset(
        records=records,
        metadata={
            **_BASE_METADATA,
            "snapshot_count": snapshot_count,
            "record_count": len(records),
            "horizons": list(horizon_tuple),
            "benchmark_horizons": list(benchmark_horizon_tuple),
            "malformed_snapshot_count": malformed_snapshot_count,
            "missing_frame_count": missing_frame_count,
        },
    )
    dataset.theme_summary = _build_theme_summary(records)
    dataset.phase_summary = _build_phase_summary(records)
    dataset.risk_flag_summary = _build_risk_flag_summary(records)
    return dataset


def build_theme_calibration_dataset_from_store(
    *,
    store: ThemeSnapshotStore,
    frames: Mapping[str, pd.DataFrame],
    market: str = "CN",
    universe_key: str | None = None,
    max_snapshots: int | None = None,
) -> ThemeCalibrationDataset:
    paths = store.list_snapshots(market=market, universe_key=universe_key)
    if max_snapshots is not None:
        limit = max(int(max_snapshots), 0)
        paths = paths[-limit:] if limit else []
    return build_theme_calibration_dataset(snapshots=paths, frames=frames)


def _build_theme_summary(records: list[ThemeReplayRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ThemeReplayRecord]] = {}
    for record in records:
        key = record.primary_theme_id or "unknown_theme"
        grouped.setdefault(key, []).append(record)
    return {
        key: {
            "theme_name": next(
                (record.primary_theme_name for record in group if record.primary_theme_name),
                key,
            ),
            **_summary_payload(group, include_alpha_10d=True),
        }
        for key, group in sorted(grouped.items())
    }


def _build_phase_summary(records: list[ThemeReplayRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ThemeReplayRecord]] = {}
    for record in records:
        grouped.setdefault(record.phase or "unknown_phase", []).append(record)
    return {
        key: _summary_payload(group, include_alpha_10d=False)
        for key, group in sorted(grouped.items())
    }


def _build_risk_flag_summary(records: list[ThemeReplayRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ThemeReplayRecord]] = {}
    for record in records:
        for flag in record.risk_flags:
            grouped.setdefault(flag, []).append(record)
    return {
        key: {
            "record_count": len(group),
            "available_count": sum(1 for record in group if record.data_available),
            "avg_forward_alpha_5d": _mean_optional(
                record.forward_alpha_5d for record in group
            ),
            "hit_rate_5d": _hit_rate(record.hit_5d for record in group),
            "avg_max_drawdown_10d": _mean_optional(
                record.max_drawdown_10d for record in group
            ),
        }
        for key, group in sorted(grouped.items())
    }


def _summary_payload(
    records: list[ThemeReplayRecord],
    *,
    include_alpha_10d: bool,
) -> dict[str, Any]:
    payload = {
        "record_count": len(records),
        "available_count": sum(1 for record in records if record.data_available),
        "avg_symbol_theme_score": _mean_optional(
            record.symbol_theme_score for record in records
        ),
        "avg_forward_alpha_5d": _mean_optional(
            record.forward_alpha_5d for record in records
        ),
        "hit_rate_5d": _hit_rate(record.hit_5d for record in records),
        "avg_max_drawdown_10d": _mean_optional(
            record.max_drawdown_10d for record in records
        ),
    }
    if include_alpha_10d:
        payload["avg_forward_alpha_10d"] = _mean_optional(
            record.forward_alpha_10d for record in records
        )
        payload["hit_rate_10d"] = _hit_rate(record.hit_10d for record in records)
    return payload


def _summary_lines(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    max_rows: int,
) -> list[str]:
    if not summary:
        return ["- none"]
    rows = []
    for key in sorted(
        summary,
        key=lambda item: (
            -int(_safe_float(summary[item].get("record_count"), 0.0)),
            item,
        ),
    )[: max(int(max_rows), 0)]:
        payload = summary.get(key, {}) or {}
        rows.append(
            "- "
            f"{key}: records={payload.get('record_count', 0)}, "
            f"available={payload.get('available_count', 0)}, "
            f"alpha5={_format_optional(payload.get('avg_forward_alpha_5d'))}, "
            f"hit5={_format_optional(payload.get('hit_rate_5d'))}"
        )
    return rows


def _theme_rotation_payload(snapshot_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(snapshot_payload, Mapping):
        return {}
    rotation = snapshot_payload.get("theme_rotation")
    if isinstance(rotation, Mapping):
        return rotation
    return snapshot_payload


def _looks_like_empty_theme_rotation(payload: Mapping[str, Any]) -> bool:
    rotation = _theme_rotation_payload(payload)
    return isinstance(rotation, Mapping) and isinstance(rotation.get("symbol_scores"), Mapping)


def _load_snapshot_path(path: str | Path) -> Mapping[str, Any] | None:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _forward_alpha(
    metrics: Mapping[str, Any],
    benchmark_returns: Mapping[int, float],
    horizon: int,
) -> float | None:
    value = metrics.get(f"forward_return_{horizon}d")
    if not _is_finite(value):
        return None
    benchmark = float(benchmark_returns.get(horizon, 0.0) or 0.0)
    return _clean_float(float(value) - benchmark)


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _date_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = "".join(char for char in text if char.isdigit())
    return digits[:8] if len(digits) >= 8 else digits


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        return []
    try:
        items = list(value or [])
    except TypeError:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _optional_float(value: Any) -> float | None:
    if not _is_finite(value):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    numeric = _optional_float(value)
    return int(numeric) if numeric is not None else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if not _is_finite(value):
        return default
    return float(value)


def _is_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _median(values: list[float]) -> float:
    clean = sorted(float(value) for value in values if _is_finite(value))
    if not clean:
        return 0.0
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return _clean_float(clean[midpoint])
    return _clean_float((clean[midpoint - 1] + clean[midpoint]) / 2.0)


def _mean_optional(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if _is_finite(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _hit_rate(values: Iterable[bool | None]) -> float | None:
    labels = [value for value in values if isinstance(value, bool)]
    if not labels:
        return None
    return sum(1 for value in labels if value) / len(labels)


def _format_optional(value: Any) -> str:
    if value is None or not _is_finite(value):
        return "NA"
    return f"{float(value):.4f}"


def _clean_float(value: float) -> float:
    return round(float(value), 12)
