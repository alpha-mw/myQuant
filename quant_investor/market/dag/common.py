from __future__ import annotations

import asyncio
import contextvars
import threading
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from quant_investor.agent_protocol import ActionLabel


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()  # type: ignore[call-arg]
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            return {}
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            return {}
    if is_dataclass(value):
        try:
            dumped = asdict(value)
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            return {}
    return {}


def _score_to_action(score: float) -> ActionLabel:
    if score >= 0.25:
        return ActionLabel.BUY
    if score <= -0.35:
        return ActionLabel.SELL
    return ActionLabel.HOLD


def _run_async_coroutine_safely(coro_factory: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}
    ctx = contextvars.copy_context()

    def _runner() -> None:
        try:
            result_box["value"] = ctx.run(asyncio.run, coro_factory())
        except BaseException as exc:  # pragma: no cover - defensive thread boundary
            error_box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_box:
        raise error_box["error"]
    return result_box.get("value")
