"""Markdown renderer for the theme governance sidecar."""

from __future__ import annotations

import math
from typing import Any, Mapping


SECTION_HEADING = "## Theme Governance Sidecar"


def render_theme_governance_markdown(
    payload: Mapping[str, Any] | None,
    *,
    max_rows: int = 20,
) -> str:
    if not isinstance(payload, Mapping) or not payload:
        return ""
    enabled = bool(payload.get("enabled", False))
    status = str(payload.get("status") or "").strip().lower()
    if not enabled and status in {"", "disabled"}:
        return ""

    decisions = _list_of_mappings(payload.get("decisions"))[: max(int(max_rows), 0)]
    notes = _join_limited(payload.get("diagnostic_notes"), limit=4)
    if status in {"error", "unavailable"} and not decisions:
        suffix = f" Diagnostics: {notes}" if notes else ""
        return (
            f"{SECTION_HEADING}\n\n"
            "- Scope: shadow/governance only; final executable decision remains baseline.\n"
            f"- Status: {status}.{suffix}\n"
        )
    if not decisions:
        return ""

    lines = [
        SECTION_HEADING,
        "",
        "- Scope: shadow/governance only; final executable decision remains baseline.",
        "",
        "| Theme | Gate | 当前分 | 10日热度 | 5日变化 | 持续天数 | 趋势状态 | Confidence | Breadth | Members | Phase | Style | Notes |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|",
    ]
    for decision in decisions:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(str(_field(decision, "theme_name") or _field(decision, "theme_id") or "-")),
                    _cell(str(_field(decision, "gate_label") or "-")),
                    _format_float(
                        _field(decision, "raw_score")
                        if _field(decision, "raw_score") is not None
                        else _field(decision, "score"),
                        digits=1,
                    ),
                    _format_float(_field(decision, "heat_10d"), digits=1),
                    _format_float(_field(decision, "heat_delta_5d"), digits=1),
                    _format_int(_field(decision, "persistence_count")),
                    _cell(str(_field(decision, "trend_state") or "-")),
                    _format_float(_field(decision, "confidence"), digits=2),
                    _format_float(_field(decision, "breadth"), digits=2),
                    _format_int(_field(decision, "member_count")),
                    _cell(str(_field(decision, "phase") or "-")),
                    _cell(str(_field(decision, "style_tag") or "-")),
                    _cell(_join_limited(_field(decision, "reasons"), limit=3) or "-"),
                ]
            )
            + " |"
        )
    summary = _summary_text(payload.get("summary_counts"))
    lines.extend(
        [
            "",
            f"- Status: {status or 'success'}",
            f"- Summary: {summary or '-'}",
            "- Note: labels are governance defaults for observation, not buy/sell signals.",
        ]
    )
    if notes:
        lines.append(f"- Diagnostics: {notes}")
    return "\n".join(lines).strip() + "\n"


def append_theme_governance_section_once(
    markdown_report: str,
    payload: Mapping[str, Any] | None,
    *,
    max_rows: int = 20,
) -> str:
    base = str(markdown_report or "")
    if SECTION_HEADING in base:
        return base
    section = render_theme_governance_markdown(payload, max_rows=max_rows)
    if not section:
        return base
    if not base.strip():
        return section
    return base.rstrip() + "\n\n" + section.strip() + "\n"


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


def _summary_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    parts = []
    for key in (
        "admitted_shadow",
        "watchlist_strong",
        "watchlist_rebuild",
        "rejected",
        "umbrella_only",
        "unavailable",
    ):
        count = value.get(key)
        if count:
            parts.append(f"{key}={count}")
    return ", ".join(parts)


def _format_float(value: Any, *, digits: int) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def _format_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return str(int(numeric))


def _cell(value: str) -> str:
    return str(value or "-").replace("|", "/").replace("\n", " ").strip() or "-"
