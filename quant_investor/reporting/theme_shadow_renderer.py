"""Theme shadow monitor markdown renderer."""

from __future__ import annotations

import math
from typing import Any, Mapping


def render_theme_shadow_monitor_markdown(
    monitor: Mapping[str, Any] | None,
    *,
    max_rows: int = 20,
) -> str:
    if not isinstance(monitor, Mapping) or not monitor:
        return ""

    status = str(monitor.get("status") or "").strip().lower()
    if status == "disabled":
        return ""
    if status == "error":
        notes = _join_limited(monitor.get("diagnostic_notes"), limit=4)
        applied = _truthy(monitor.get("theme_overlay_applied_to_baseline"))
        lines = [
            "## Theme Shadow Monitor",
            "",
            "- display_alias: 主题 Shadow Monitor",
            "- status: error",
            "- final_decision_source: baseline",
            f"- production_decision_source: {_text(monitor.get('production_decision_source')) or 'no_theme_baseline'}",
            f"- control_decision_source: {_text(monitor.get('control_decision_source')) or 'no_theme_baseline'}",
            f"- diagnostic_notes: {notes or '-'}",
            f"- {_source_note(applied)}",
        ]
        return "\n".join(lines).strip() + "\n"
    if status.startswith("not_persisted"):
        notes = _join_limited(monitor.get("diagnostic_notes"), limit=4)
        lines = [
            "## Theme Shadow Monitor",
            "",
            "- display_alias: 主题 Shadow Monitor",
            f"- status: {status}",
            "- final_decision_source: baseline",
            f"- artifact_path: {_text(monitor.get('artifact_path')) or '-'}",
            f"- theme_snapshot.path: {_theme_snapshot_path(monitor) or '-'}",
            f"- diagnostic_notes: {notes or '-'}",
            "- note: Shadow monitor only; final executable decision remains baseline.",
        ]
        return "\n".join(lines).strip() + "\n"
    if status != "success":
        return ""

    row_limit = max(int(max_rows or 0), 0)
    lines = [
        "## Theme Shadow Monitor",
        "",
        "- display_alias: 主题 Shadow Monitor",
        f"- status: {status}",
        f"- final_decision_source: {_text(monitor.get('final_decision_source')) or 'baseline'}",
        f"- production_decision_source: {_text(monitor.get('production_decision_source')) or 'no_theme_baseline'}",
        f"- control_decision_source: {_text(monitor.get('control_decision_source')) or 'no_theme_baseline'}",
        f"- theme_overlay_applied_to_baseline: {str(_truthy(monitor.get('theme_overlay_applied_to_baseline'))).lower()}",
        f"- theme_overlay_modules: {_module_summary(monitor.get('theme_overlay_modules'))}",
        f"- note: {_source_note(_truthy(monitor.get('theme_overlay_applied_to_baseline')))}",
        f"- candidate_overlap_ratio: {_format_float(monitor.get('candidate_overlap_ratio'))}",
        f"- entered_candidates: {_join_limited(monitor.get('entered_candidates'), limit=row_limit) or '-'}",
        f"- dropped_candidates: {_join_limited(monitor.get('dropped_candidates'), limit=row_limit) or '-'}",
        f"- selected_overlap_ratio: {_format_float(monitor.get('selected_overlap_ratio'))}",
        "",
        "### Largest Weight Deltas",
        _render_weight_delta_table(monitor.get("portfolio_weight_deltas"), row_limit),
        "",
        "### Theme Exposure",
        _render_exposure_table(
            monitor.get("theme_exposure_baseline"),
            monitor.get("theme_exposure_shadow"),
            row_limit,
        ),
        "",
        "### Risk Delta",
        _render_mapping(monitor.get("risk_delta"), row_limit),
    ]
    diagnostic_notes = _join_limited(
        monitor.get("diagnostic_notes"),
        limit=row_limit,
    )
    if diagnostic_notes:
        lines.extend(["", f"- diagnostic_notes: {diagnostic_notes}"])
    snapshot_path = _theme_snapshot_path(monitor)
    if snapshot_path:
        lines.extend(["", f"- theme_snapshot.path: {snapshot_path}"])
    artifact_path = _text(monitor.get("artifact_path"))
    if artifact_path:
        lines.extend(["", f"- artifact_path: {artifact_path}"])
    return "\n".join(lines).strip() + "\n"


def render_theme_production_overlay_markdown(
    overlay: Mapping[str, Any] | None,
) -> str:
    if not isinstance(overlay, Mapping) or not overlay:
        return ""
    if not _truthy(overlay.get("theme_overlay_applied_to_baseline")):
        return ""
    lines = [
        "## Theme Baseline Overlay",
        "",
        "- display_alias: 主题 Production Overlay",
        f"- production_decision_source: {_text(overlay.get('production_decision_source')) or 'theme_overlay_baseline'}",
        f"- control_decision_source: {_text(overlay.get('control_decision_source')) or 'no_theme_baseline'}",
        "- theme_overlay_applied_to_baseline: true",
        f"- theme_overlay_modules: {_module_summary(overlay.get('theme_overlay_modules'))}",
        f"- canonical_branch_unchanged: {str(_truthy(overlay.get('canonical_branch_unchanged'), default=True)).lower()}",
        f"- theme_likelihood_added: {str(_truthy(overlay.get('theme_likelihood_added'))).lower()}",
        f"- posterior_formula_changed: {str(_truthy(overlay.get('posterior_formula_changed'))).lower()}",
        "- note: Theme overlay is part of the production baseline only through explicit impact toggles; no-theme baseline is retained as control.",
    ]
    return "\n".join(lines).strip() + "\n"


def append_theme_production_overlay_section_once(
    markdown: str,
    overlay: Mapping[str, Any] | None,
) -> str:
    section = render_theme_production_overlay_markdown(overlay)
    if not section:
        return str(markdown or "")
    original = str(markdown or "")
    if "Theme Baseline Overlay" in original:
        return original
    if "主题 Production Overlay" in original:
        return original.replace(
            "主题 Production Overlay",
            "Theme Baseline Overlay / 主题 Production Overlay",
            1,
        )
    if "Theme Production Overlay" in original:
        return original.replace(
            "Theme Production Overlay",
            "Theme Baseline Overlay",
            1,
        )
    if not original.strip():
        return section.strip() + "\n"
    return original.rstrip() + "\n\n" + section.strip() + "\n"


def append_theme_shadow_section_once(
    markdown: str,
    monitor: Mapping[str, Any] | None,
    *,
    max_rows: int = 20,
) -> str:
    section = render_theme_shadow_monitor_markdown(monitor, max_rows=max_rows)
    if not section:
        return str(markdown or "")

    original = str(markdown or "")
    if "Theme Shadow Monitor" in original:
        return original
    if "主题 Shadow Monitor" in original:
        return original.replace(
            "主题 Shadow Monitor",
            "Theme Shadow Monitor / 主题 Shadow Monitor",
            1,
        )
    if not original.strip():
        return section.strip() + "\n"
    return original.rstrip() + "\n\n" + section.strip() + "\n"


def _theme_snapshot_path(monitor: Mapping[str, Any]) -> str:
    direct = _text(monitor.get("theme_snapshot_path"))
    if direct:
        return direct
    snapshot = monitor.get("theme_snapshot")
    if isinstance(snapshot, Mapping):
        return _text(snapshot.get("path")) or _text(snapshot.get("artifact_path"))
    return ""


def _render_weight_delta_table(value: Any, limit: int) -> str:
    rows = _list_of_mappings(value)[:limit]
    if not rows:
        return "_none_"
    lines = [
        "| symbol | baseline_weight | shadow_weight | delta | theme | phase |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        theme = _text(row.get("primary_theme_name")) or _text(row.get("primary_theme_id")) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("symbol")),
                    _format_float(row.get("baseline_weight")),
                    _format_float(row.get("shadow_weight")),
                    _format_float(row.get("weight_delta")),
                    _cell(theme),
                    _cell(row.get("phase")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_exposure_table(baseline: Any, shadow: Any, limit: int) -> str:
    baseline_map = _mapping_float_values(baseline)
    shadow_map = _mapping_float_values(shadow)
    theme_ids = sorted(set(baseline_map) | set(shadow_map))[:limit]
    if not theme_ids:
        return "_none_"
    lines = [
        "| theme | baseline | shadow | delta |",
        "|---|---:|---:|---:|",
    ]
    for theme_id in theme_ids:
        base = baseline_map.get(theme_id, 0.0)
        capped = shadow_map.get(theme_id, 0.0)
        lines.append(
            f"| {_cell(theme_id)} | {base:.6f} | {capped:.6f} | {capped - base:.6f} |"
        )
    return "\n".join(lines)


def _render_mapping(value: Any, limit: int) -> str:
    if not isinstance(value, Mapping) or not value:
        return "_none_"
    lines: list[str] = []
    for key in sorted(value)[:limit]:
        item = value.get(key)
        if isinstance(item, Mapping):
            rendered = ", ".join(
                f"{_text(inner_key)}={_text(inner_value)}"
                for inner_key, inner_value in list(item.items())[:limit]
            )
        elif isinstance(item, list):
            rendered = _join_limited(item, limit=limit)
        else:
            rendered = _text(item)
        lines.append(f"- {_text(key)}: {rendered or '-'}")
    return "\n".join(lines) if lines else "_none_"


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)):
        return []
    try:
        items = list(value or [])
    except TypeError:
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _mapping_float_values(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw_value in value.items():
        numeric = _optional_float(raw_value)
        if numeric is None:
            continue
        result[str(key)] = numeric
    return result


def _module_summary(value: Any) -> str:
    modules = value if isinstance(value, Mapping) else {}
    ordered = {
        "funnel_boost": bool(modules.get("funnel_boost", False)),
        "risk_guard": bool(modules.get("risk_guard", False)),
        "portfolio_cap": bool(modules.get("portfolio_cap", False)),
    }
    return ", ".join(
        f"{name}={str(enabled).lower()}"
        for name, enabled in ordered.items()
    )


def _source_note(overlay_applied: bool) -> str:
    if overlay_applied:
        return (
            "Theme impact toggles are part of the production baseline; "
            "no-theme baseline is retained as control."
        )
    return "Shadow monitor only; final executable decision remains baseline."


def _truthy(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off", "none"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return bool(default)


def _join_limited(value: Any, *, limit: int) -> str:
    if isinstance(value, (str, bytes)):
        values = [str(value)]
    else:
        try:
            values = [str(item) for item in list(value or [])]
        except TypeError:
            values = []
    cleaned = [item.strip() for item in values if item.strip()]
    suffix = f" (+{len(cleaned) - limit} more)" if limit >= 0 and len(cleaned) > limit else ""
    return ", ".join(cleaned[: max(limit, 0)]) + suffix


def _format_float(value: Any) -> str:
    numeric = _optional_float(value)
    return "-" if numeric is None else f"{numeric:.6f}"


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _text(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def _cell(value: Any) -> str:
    text = _text(value).replace("|", "/")
    return text or "-"
