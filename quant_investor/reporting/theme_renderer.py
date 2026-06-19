"""Theme rotation markdown renderer."""

from __future__ import annotations

import math
from typing import Any, Mapping


def render_theme_rotation_markdown(
    theme_rotation: Mapping[str, Any] | None,
    *,
    max_themes: int = 10,
) -> str:
    if not isinstance(theme_rotation, Mapping) or not theme_rotation:
        return ""

    status = str(theme_rotation.get("status") or "").strip().lower()
    if status == "disabled":
        return ""
    if status == "error":
        note = _join_limited(theme_rotation.get("diagnostic_notes"), limit=3)
        suffix = f"：{note}" if note else "。"
        return f"## 主题轮动雷达\n\n主题扫描异常，已跳过主题轮动雷达{suffix}\n"
    if status and status != "success":
        return ""

    top_themes = _list_of_mappings(theme_rotation.get("top_themes"))[:max(max_themes, 0)]
    if not top_themes:
        return "## 主题轮动雷达\n\n主题扫描已完成，但未发现满足成员数/质量阈值的主题。\n"

    lines = [
        "## 主题轮动雷达",
        "",
        "| 排名 | 主题 | 分数 | 阶段 | 置信度 | 成员数 | 龙头/强势标的 | 风险 |",
        "|---:|---|---:|---|---:|---:|---|---|",
    ]
    for rank, theme in enumerate(top_themes, start=1):
        theme_id = str(_field(theme, "theme_id") or "")
        theme_name = str(_field(theme, "theme_name") or theme_id or "-")
        score = _format_float(_field(theme, "score"), digits=1)
        phase = str(_field(theme, "phase") or "-")
        confidence = _format_float(_field(theme, "confidence"), digits=2)
        member_count = _format_int(_field(theme, "member_count"))
        top_symbols = _join_limited(_field(theme, "top_symbols"), limit=5)
        risk_flags = _join_limited(_field(theme, "risk_flags"), limit=3)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    _cell(theme_name),
                    score,
                    _cell(phase),
                    confidence,
                    member_count,
                    _cell(top_symbols or "-"),
                    _cell(risk_flags or "-"),
                ]
            )
            + " |"
        )

    mode_parts = _mode_parts(theme_rotation.get("metadata"))
    lines.extend(
        [
            "",
            f"- 主题扫描状态：{status or 'success'}",
            f"- 数据模式：{' / '.join(mode_parts) if mode_parts else '-'}",
            "- 说明：主题分数仅用于结构化观察，本阶段不影响组合权重。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _field(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _list_of_mappings(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)):
        return []
    try:
        items = list(value or [])
    except TypeError:
        return []
    return [item for item in items if isinstance(item, Mapping) or hasattr(item, "__dict__")]


def _join_limited(value: Any, *, limit: int) -> str:
    if isinstance(value, (str, bytes)):
        values = [str(value)]
    else:
        try:
            values = [str(item) for item in list(value or [])]
        except TypeError:
            values = []
    cleaned: list[str] = []
    for item in values:
        text = item.strip()
        if text:
            cleaned.append(text)
    return ", ".join(cleaned[: max(limit, 0)])


def _mode_parts(metadata: Any) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []
    parts: list[str] = []
    for key in ("deterministic", "no_llm", "no_network"):
        if bool(metadata.get(key)):
            parts.append(key)
    return parts


def _format_float(value: Any, *, digits: int) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def _format_int(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return str(int(numeric))


def _cell(value: str) -> str:
    return str(value or "-").replace("|", "/").replace("\n", " ").strip() or "-"
