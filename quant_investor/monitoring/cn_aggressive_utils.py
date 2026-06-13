"""Shared helpers for the CN aggressive portfolio tracker."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, dict):
            return dict(_jsonable(payload))
    if isinstance(value, dict):
        return dict(_jsonable(value))
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return dict(_jsonable(data))
    return {}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and not isinstance(value, pd.DataFrame):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_pct(value: float, base: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return value / base
