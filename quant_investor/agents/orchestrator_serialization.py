"""Serialization helpers for the advisory review-layer orchestrator."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


def compact_review_mapping(payload: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                result[str(key)] = value
            elif isinstance(value, Mapping):
                nested = {
                    k: v
                    for k, v in value.items()
                    if isinstance(v, (int, float, str, bool)) or v is None
                }
                if nested:
                    result[str(key)] = nested
            elif isinstance(value, list):
                items = [item for item in value if isinstance(item, (int, float, str, bool))]
                if items:
                    result[str(key)] = items[:10]
        return result
    if hasattr(payload, "__dict__"):
        return compact_review_mapping(payload.__dict__)
    return {}


def serialize_review_item(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): serialize_review_item(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [serialize_review_item(item) for item in value]
    return value


def serialize_review_map(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): serialize_review_item(item)
        for key, item in value.items()
    }


__all__ = [
    "compact_review_mapping",
    "serialize_review_item",
    "serialize_review_map",
]
