from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_investor.themes.smoothing import (
    ThemeSmoothingConfig,
    ThemeSmoothingResult,
    smooth_numeric_series,
    smooth_theme_series,
)


GOVERNANCE_SCHEMA_VERSION = "theme_governance.v1"
GOVERNANCE_METADATA = {
    "deterministic": True,
    "no_llm": True,
    "no_network": True,
    "shadow_only": True,
}
GATE_LABELS = (
    "admitted_shadow",
    "watchlist_strong",
    "watchlist_rebuild",
    "rejected",
    "umbrella_only",
    "unavailable",
)
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ThemeGovernanceConfig:
    min_member_count: int = 5
    rejected_score_floor: float = 35.0
    rebuild_score_ceiling: float = 45.0
    rebuild_confidence_floor: float = 0.35
    rebuild_breadth_floor: float = 0.30
    admitted_score_floor: float = 55.0
    admitted_confidence_floor: float = 0.45
    admitted_breadth_floor: float = 0.40
    smoothing_enabled: bool = True
    smoothing_window: int = 10
    smoothing_min_observations: int = 5
    smoothing_admission_min_persistence: int = 3
    admitted_phases: tuple[str, ...] = (
        "accumulation",
        "early_acceleration",
        "confirmed_rotation",
    )
    severe_phases: tuple[str, ...] = (
        "overextended",
        "distribution",
    )
    severe_risk_flags: tuple[str, ...] = (
        "theme_overextended",
        "theme_overextended_no_chase",
        "theme_distribution_risk",
        "theme_fake_breakout_risk",
        "theme_low_breadth",
    )


@dataclass(frozen=True)
class ThemeGovernanceRegistryEntry:
    theme_id: str
    theme_type: str = "tradable"
    style_tag: str = ""
    parent_theme: str = ""
    theme_name: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "theme_type": self.theme_type,
            "style_tag": self.style_tag,
            "parent_theme": self.parent_theme,
            "theme_name": self.theme_name,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ThemeGovernanceRegistry:
    entries: dict[str, ThemeGovernanceRegistryEntry] = field(default_factory=dict)
    diagnostic_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ThemeGovernanceDecision:
    theme_id: str
    theme_name: str
    gate_label: str
    score: float | None = None
    raw_score: float | None = None
    smoothed_score: float | None = None
    heat_10d: float | None = None
    heat_delta_5d: float | None = None
    persistence_count: int = 0
    trend_state: str = "insufficient_history"
    smoothing_status: str = "insufficient_history"
    confidence: float | None = None
    breadth: float | None = None
    member_count: int | None = None
    phase: str = ""
    risk_flags: tuple[str, ...] = ()
    theme_type: str = "tradable"
    style_tag: str = ""
    parent_theme: str = ""
    reasons: tuple[str, ...] = ()
    diagnostic_notes: tuple[str, ...] = ()
    registry_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "gate_label": self.gate_label,
            "score": self.score,
            "raw_score": self.raw_score,
            "smoothed_score": self.smoothed_score,
            "heat_10d": self.heat_10d,
            "heat_delta_5d": self.heat_delta_5d,
            "persistence_count": int(self.persistence_count),
            "trend_state": self.trend_state,
            "smoothing_status": self.smoothing_status,
            "confidence": self.confidence,
            "breadth": self.breadth,
            "member_count": self.member_count,
            "phase": self.phase,
            "risk_flags": list(self.risk_flags),
            "theme_type": self.theme_type,
            "style_tag": self.style_tag,
            "parent_theme": self.parent_theme,
            "reasons": list(self.reasons),
            "diagnostic_notes": list(self.diagnostic_notes),
            "registry_notes": self.registry_notes,
        }


@dataclass(frozen=True)
class ThemeGovernanceResult:
    status: str
    enabled: bool
    market: str = ""
    universe_key: str = ""
    as_of: str = ""
    decisions: tuple[ThemeGovernanceDecision, ...] = ()
    diagnostic_notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: dict(GOVERNANCE_METADATA))

    def to_dict(self) -> dict[str, Any]:
        decisions = [decision.to_dict() for decision in self.decisions]
        return {
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "status": str(self.status),
            "market": str(self.market or ""),
            "universe_key": str(self.universe_key or ""),
            "as_of": str(self.as_of or ""),
            "decisions": decisions,
            "top_themes": decisions,
            "summary_counts": _summary_counts(self.decisions),
            "diagnostic_notes": list(self.diagnostic_notes),
            "metadata": dict(self.metadata or GOVERNANCE_METADATA),
        }


def load_theme_governance_registry(path: str | Path | None) -> ThemeGovernanceRegistry:
    """Load optional JSON theme ontology without adding parser dependencies."""

    if not path:
        return ThemeGovernanceRegistry(
            entries={},
            diagnostic_notes=["theme_governance_registry_unset"],
        )
    registry_path = Path(path)
    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            raw_payload = json.load(handle)
    except FileNotFoundError:
        return ThemeGovernanceRegistry(
            entries={},
            diagnostic_notes=[f"theme_governance_registry_missing: {registry_path}"],
        )
    except (OSError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ThemeGovernanceRegistry(
            entries={},
            diagnostic_notes=[f"theme_governance_registry_malformed: {exc}"],
        )
    if not isinstance(raw_payload, Mapping):
        return ThemeGovernanceRegistry(
            entries={},
            diagnostic_notes=["theme_governance_registry_malformed: root_not_mapping"],
        )

    raw_themes = raw_payload.get("themes", [])
    if isinstance(raw_themes, Mapping):
        raw_themes = list(raw_themes.values())
    if isinstance(raw_themes, (str, bytes)):
        raw_themes = []
    entries: dict[str, ThemeGovernanceRegistryEntry] = {}
    diagnostics: list[str] = []
    try:
        iterable = list(raw_themes or [])
    except TypeError:
        iterable = []
        diagnostics.append("theme_governance_registry_malformed: themes_not_iterable")
    for item in iterable:
        if not isinstance(item, Mapping):
            diagnostics.append("theme_governance_registry_entry_skipped: not_mapping")
            continue
        theme_id = _clean_text(item.get("theme_id"))
        if not theme_id:
            diagnostics.append("theme_governance_registry_entry_skipped: missing_theme_id")
            continue
        entries[theme_id] = ThemeGovernanceRegistryEntry(
            theme_id=theme_id,
            theme_type=_clean_text(item.get("theme_type")) or "tradable",
            style_tag=_clean_text(item.get("style_tag")),
            parent_theme=_clean_text(item.get("parent_theme")),
            theme_name=_clean_text(item.get("theme_name")),
            notes=_clean_text(item.get("notes")),
        )
    return ThemeGovernanceRegistry(entries=entries, diagnostic_notes=diagnostics)


def evaluate_theme_governance(
    theme_rotation: Mapping[str, Any] | None,
    config: ThemeGovernanceConfig | None = None,
    registry: ThemeGovernanceRegistry | None = None,
    history: Iterable[Mapping[str, Any]] | None = None,
) -> ThemeGovernanceResult:
    settings = config or ThemeGovernanceConfig()
    ontology = registry if registry is not None else load_theme_governance_registry(None)
    registry_notes = _dedupe_notes(getattr(ontology, "diagnostic_notes", []) or [])

    if not isinstance(theme_rotation, Mapping):
        return _result(
            status="unavailable",
            enabled=False,
            diagnostic_notes=[*registry_notes, "theme_rotation_missing"],
        )

    status = _clean_text(theme_rotation.get("status")).lower() or "success"
    enabled = bool(theme_rotation.get("enabled", status != "disabled"))
    market = _clean_text(theme_rotation.get("market"))
    universe_key = _clean_text(theme_rotation.get("universe_key"))
    as_of = _clean_text(theme_rotation.get("as_of"))
    input_notes = _string_list(theme_rotation.get("diagnostic_notes"))

    if not enabled or status == "disabled":
        return _result(
            status="disabled",
            enabled=False,
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            diagnostic_notes=[*registry_notes, *input_notes, "theme_rotation_disabled"],
        )
    if status == "error":
        return _result(
            status="error",
            enabled=True,
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            diagnostic_notes=[*registry_notes, *input_notes],
        )
    if status and status != "success":
        return _result(
            status="unavailable",
            enabled=True,
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            diagnostic_notes=[*registry_notes, *input_notes, f"theme_rotation_status_unhandled: {status}"],
        )

    raw_theme_scores = theme_rotation.get("theme_scores")
    if not isinstance(raw_theme_scores, Mapping):
        return _result(
            status="unavailable",
            enabled=True,
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            diagnostic_notes=[*registry_notes, *input_notes, "theme_scores_missing"],
        )

    decisions: list[ThemeGovernanceDecision] = []
    history_payloads = _rotation_history(history)
    for theme_id, raw_theme in raw_theme_scores.items():
        decisions.append(
            _evaluate_theme(
                fallback_theme_id=str(theme_id),
                raw_theme=raw_theme,
                config=settings,
                registry=ontology,
                history=history_payloads,
                current_as_of=as_of,
            )
        )
    decisions = sorted(
        decisions,
        key=lambda item: (
            _label_rank(item.gate_label),
            -(item.score if item.score is not None else -1.0),
            item.theme_id,
        ),
    )
    return _result(
        status="success",
        enabled=True,
        market=market,
        universe_key=universe_key,
        as_of=as_of,
        decisions=decisions,
        diagnostic_notes=[*registry_notes, *input_notes],
    )


def write_theme_governance_artifact(
    payload: Mapping[str, Any],
    root_dir: str | Path,
    *,
    market: str = "CN",
    universe_key: str = "",
    as_of: str = "",
    run_id: str = "",
) -> Path:
    safe_market = _safe_component(market, "unknown_market")
    safe_universe_key = _safe_component(universe_key or "unknown_universe", "unknown_universe")
    safe_as_of = _safe_component(as_of or "unknown_date", "unknown_date")
    safe_run_id = _safe_component(run_id or "governance", "governance")
    date_segment = _date_segment(as_of)
    parent = Path(root_dir) / safe_market / date_segment
    path = parent / f"{safe_universe_key}_{safe_as_of}_{safe_run_id}_theme_governance.json"
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(copy.deepcopy(dict(payload)), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return path


def _evaluate_theme(
    *,
    fallback_theme_id: str,
    raw_theme: Any,
    config: ThemeGovernanceConfig,
    registry: ThemeGovernanceRegistry,
    history: list[Mapping[str, Any]],
    current_as_of: str = "",
) -> ThemeGovernanceDecision:
    if not isinstance(raw_theme, Mapping):
        return ThemeGovernanceDecision(
            theme_id=fallback_theme_id,
            theme_name=fallback_theme_id,
            gate_label="unavailable",
            reasons=("theme_payload_not_mapping",),
            diagnostic_notes=("theme_payload_not_mapping",),
        )
    theme_id = _clean_text(raw_theme.get("theme_id")) or fallback_theme_id
    entry = registry.entries.get(theme_id)
    inferred_type = "tradable" if str(theme_id).startswith("industry::") else "unclassified"
    theme_type = _clean_text(entry.theme_type if entry else "") or inferred_type
    theme_name = (
        _clean_text(entry.theme_name if entry else "")
        or _clean_text(raw_theme.get("theme_name"))
        or theme_id
    )
    diagnostics: list[str] = []
    score = _numeric_field(raw_theme, "score", lower=0.0, upper=100.0, diagnostics=diagnostics)
    confidence = _numeric_field(raw_theme, "confidence", lower=0.0, upper=1.0, diagnostics=diagnostics)
    breadth = _numeric_field(raw_theme, "breadth", lower=0.0, upper=1.0, diagnostics=diagnostics)
    member_count = _integer_field(raw_theme, "member_count", diagnostics=diagnostics)
    phase = _clean_text(raw_theme.get("phase")).lower()
    risk_flags = tuple(_string_list(raw_theme.get("risk_flags")))

    if score is None or confidence is None or breadth is None or member_count is None:
        return ThemeGovernanceDecision(
            theme_id=theme_id,
            theme_name=theme_name,
            gate_label="unavailable",
            score=score,
            raw_score=score,
            confidence=confidence,
            breadth=breadth,
            member_count=member_count,
            phase=phase,
            risk_flags=risk_flags,
            theme_type=theme_type,
            style_tag=_clean_text(entry.style_tag if entry else ""),
            parent_theme=_clean_text(entry.parent_theme if entry else ""),
            reasons=("required_theme_fields_unavailable",),
            diagnostic_notes=tuple(diagnostics),
            registry_notes=_clean_text(entry.notes if entry else ""),
        )

    smoothing = _theme_smoothing(
        theme_id=theme_id,
        raw_theme=raw_theme,
        history=history,
        config=config,
        current_as_of=current_as_of,
    )
    diagnostics.extend(smoothing.diagnostic_notes)
    score_for_gate = (
        float(smoothing.smoothed_score)
        if config.smoothing_enabled
        and smoothing.status == "success"
        and smoothing.smoothed_score is not None
        else score
    )
    confidence_for_gate = _smoothed_numeric_for_gate(
        theme_id=theme_id,
        raw_theme=raw_theme,
        history=history,
        field_name="confidence",
        lower=0.0,
        upper=1.0,
        config=config,
        current_as_of=current_as_of,
    )
    breadth_for_gate = _smoothed_numeric_for_gate(
        theme_id=theme_id,
        raw_theme=raw_theme,
        history=history,
        field_name="breadth",
        lower=0.0,
        upper=1.0,
        config=config,
        current_as_of=current_as_of,
    )
    confidence_for_gate = confidence if confidence_for_gate is None else confidence_for_gate
    breadth_for_gate = breadth if breadth_for_gate is None else breadth_for_gate

    gate_label, reasons = _classify_theme(
        theme_type=theme_type,
        score=score_for_gate,
        raw_score=score,
        confidence=confidence_for_gate,
        breadth=breadth_for_gate,
        member_count=member_count,
        phase=phase,
        risk_flags=risk_flags,
        config=config,
        smoothing_status=smoothing.status,
        persistence_count=smoothing.persistence_count,
        trend_state=smoothing.trend_state,
    )
    return ThemeGovernanceDecision(
        theme_id=theme_id,
        theme_name=theme_name,
        gate_label=gate_label,
        score=score_for_gate,
        raw_score=score,
        smoothed_score=smoothing.smoothed_score,
        heat_10d=smoothing.heat_10d,
        heat_delta_5d=smoothing.heat_delta_5d,
        persistence_count=int(smoothing.persistence_count),
        trend_state=smoothing.trend_state,
        smoothing_status=smoothing.status,
        confidence=confidence_for_gate,
        breadth=breadth_for_gate,
        member_count=member_count,
        phase=phase,
        risk_flags=risk_flags,
        theme_type=theme_type,
        style_tag=_clean_text(entry.style_tag if entry else ""),
        parent_theme=_clean_text(entry.parent_theme if entry else ""),
        reasons=tuple(reasons),
        diagnostic_notes=tuple(diagnostics),
        registry_notes=_clean_text(entry.notes if entry else ""),
    )


def _classify_theme(
    *,
    theme_type: str,
    score: float,
    raw_score: float,
    confidence: float,
    breadth: float,
    member_count: int,
    phase: str,
    risk_flags: tuple[str, ...],
    config: ThemeGovernanceConfig,
    smoothing_status: str,
    persistence_count: int,
    trend_state: str,
) -> tuple[str, list[str]]:
    if theme_type == "umbrella":
        return "umbrella_only", ["registry_theme_type_umbrella"]
    if member_count < int(config.min_member_count):
        return "rejected", ["member_count_below_minimum"]
    if score < float(config.rejected_score_floor):
        return "rejected", ["score_below_rejected_floor"]

    flag_set = {str(flag).strip().lower() for flag in risk_flags if str(flag).strip()}
    severe_flags = sorted(flag_set.intersection(config.severe_risk_flags))
    severe_phase = phase in set(config.severe_phases)
    if score >= float(config.admitted_score_floor) and (severe_phase or severe_flags):
        reasons = []
        if severe_phase:
            reasons.append(f"phase_{phase}")
        reasons.extend(severe_flags)
        return "watchlist_strong", reasons or ["risk_prevents_shadow_admission"]

    smoothing_required = bool(config.smoothing_enabled)
    smoothing_confirmed = (
        smoothing_status == "success"
        and persistence_count >= int(config.smoothing_admission_min_persistence)
        and trend_state != "spike_unconfirmed"
    )
    if smoothing_required and raw_score >= float(config.admitted_score_floor) and not smoothing_confirmed:
        reasons = ["raw_spike_not_confirmed"]
        if smoothing_status != "success":
            reasons.append("theme_smoothing_history_insufficient")
        if persistence_count < int(config.smoothing_admission_min_persistence):
            reasons.append("theme_persistence_below_admission_floor")
        if trend_state == "spike_unconfirmed":
            reasons.append("theme_spike_unconfirmed")
        return "watchlist_strong", reasons
    if (
        smoothing_required
        and raw_score >= float(config.admitted_score_floor)
        and score < float(config.admitted_score_floor)
    ):
        return "watchlist_strong", ["raw_spike_not_confirmed", "smoothed_score_below_admission_floor"]

    rebuild_reasons: list[str] = []
    if confidence < float(config.rebuild_confidence_floor):
        rebuild_reasons.append("confidence_below_rebuild_floor")
    if breadth < float(config.rebuild_breadth_floor):
        rebuild_reasons.append("breadth_below_rebuild_floor")
    if score <= float(config.rebuild_score_ceiling):
        rebuild_reasons.append("score_in_rebuild_band")
    if rebuild_reasons:
        return "watchlist_rebuild", rebuild_reasons

    if (
        score >= float(config.admitted_score_floor)
        and confidence >= float(config.admitted_confidence_floor)
        and breadth >= float(config.admitted_breadth_floor)
        and phase in set(config.admitted_phases)
        and not severe_flags
        and not severe_phase
        and (not smoothing_required or smoothing_confirmed)
    ):
        return "admitted_shadow", ["shadow_admission_defaults_passed"]

    if score >= float(config.admitted_score_floor):
        return "watchlist_strong", ["promising_score_without_admission_phase"]
    return "watchlist_rebuild", ["needs_rebuild_or_calibration"]


def _theme_smoothing(
    *,
    theme_id: str,
    raw_theme: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    config: ThemeGovernanceConfig,
    current_as_of: str = "",
) -> ThemeSmoothingResult:
    smoothing_config = _smoothing_config(config)
    embedded = _embedded_smoothing(raw_theme)
    score_values = _theme_history_values(
        theme_id=theme_id,
        raw_theme=raw_theme,
        history=history,
        field_name="score",
        current_as_of=current_as_of,
    )
    result = smooth_theme_series(score_values, smoothing_config)
    if result.status == "success":
        return result
    if embedded is not None and embedded.status == "success":
        return embedded
    notes = list(result.diagnostic_notes)
    if result.status != "success":
        notes.append("theme_smoothing_history_insufficient")
    return ThemeSmoothingResult(
        raw_score=result.raw_score,
        smoothed_score=result.smoothed_score,
        heat_10d=result.heat_10d,
        heat_delta_5d=result.heat_delta_5d,
        persistence_count=result.persistence_count,
        trend_state=result.trend_state,
        observation_count=result.observation_count,
        status=result.status,
        diagnostic_notes=tuple(_dedupe_notes(notes)),
    )


def _embedded_smoothing(raw_theme: Mapping[str, Any]) -> ThemeSmoothingResult | None:
    smoothed = _safe_optional_float(raw_theme.get("smoothed_score"))
    heat_10d = _safe_optional_float(raw_theme.get("heat_10d"))
    if smoothed is None and heat_10d is None:
        return None
    raw_score = _safe_optional_float(raw_theme.get("raw_score"))
    if raw_score is None:
        raw_score = _safe_optional_float(raw_theme.get("score"))
    status = _clean_text(raw_theme.get("smoothing_status")) or "success"
    trend_state = _clean_text(raw_theme.get("trend_state")) or (
        "stable" if status == "success" else "insufficient_history"
    )
    return ThemeSmoothingResult(
        raw_score=raw_score,
        smoothed_score=smoothed if smoothed is not None else heat_10d,
        heat_10d=heat_10d if heat_10d is not None else smoothed,
        heat_delta_5d=_safe_optional_float(raw_theme.get("heat_delta_5d")),
        persistence_count=max(_safe_int(raw_theme.get("persistence_count")), 0),
        trend_state=trend_state,
        observation_count=max(_safe_int(raw_theme.get("smoothing_observation_count")), 0),
        status=status,
        diagnostic_notes=tuple(_string_list(raw_theme.get("smoothing_diagnostic_notes"))),
    )


def _smoothed_numeric_for_gate(
    *,
    theme_id: str,
    raw_theme: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    field_name: str,
    lower: float,
    upper: float,
    config: ThemeGovernanceConfig,
    current_as_of: str = "",
) -> float | None:
    values = _theme_history_values(
        theme_id=theme_id,
        raw_theme=raw_theme,
        history=history,
        field_name=field_name,
        current_as_of=current_as_of,
    )
    return smooth_numeric_series(
        values,
        lower=lower,
        upper=upper,
        config=_smoothing_config(config),
    )


def _theme_history_values(
    *,
    theme_id: str,
    raw_theme: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    field_name: str,
    current_as_of: str = "",
) -> list[Any]:
    resolved_current_as_of = _clean_text(current_as_of) or _clean_text(raw_theme.get("as_of"))
    values: list[Any] = []
    for rotation in history:
        if not isinstance(rotation, Mapping):
            continue
        if _clean_text(rotation.get("status")).lower() not in {"", "success"}:
            continue
        if resolved_current_as_of and _clean_text(rotation.get("as_of")) == resolved_current_as_of:
            continue
        theme = _theme_from_rotation(rotation, theme_id)
        if isinstance(theme, Mapping):
            values.append(theme.get(field_name))
    values.append(raw_theme.get(field_name))
    return values


def _theme_from_rotation(
    rotation: Mapping[str, Any],
    theme_id: str,
) -> Mapping[str, Any] | None:
    theme_scores = rotation.get("theme_scores")
    if isinstance(theme_scores, Mapping):
        theme = theme_scores.get(theme_id)
        if isinstance(theme, Mapping):
            return theme
    top_themes = rotation.get("top_themes")
    if isinstance(top_themes, (str, bytes)):
        return None
    try:
        iterator = iter(top_themes or [])
    except TypeError:
        return None
    for item in iterator:
        if isinstance(item, Mapping) and _clean_text(item.get("theme_id")) == theme_id:
            return item
    return None


def _rotation_history(history: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if history is None or isinstance(history, (str, bytes)):
        return []
    try:
        items = list(history)
    except TypeError:
        return []
    rotations: list[Mapping[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            rotation = item.get("theme_rotation")
            if isinstance(rotation, Mapping):
                rotations.append(rotation)
            else:
                rotations.append(item)
    return sorted(rotations, key=lambda item: _clean_text(item.get("as_of")))


def _smoothing_config(config: ThemeGovernanceConfig) -> ThemeSmoothingConfig:
    return ThemeSmoothingConfig(
        window=max(int(config.smoothing_window or 10), 1),
        min_observations=max(int(config.smoothing_min_observations or 5), 1),
    )


def _result(
    *,
    status: str,
    enabled: bool,
    market: str = "",
    universe_key: str = "",
    as_of: str = "",
    decisions: list[ThemeGovernanceDecision] | None = None,
    diagnostic_notes: list[str] | None = None,
) -> ThemeGovernanceResult:
    return ThemeGovernanceResult(
        status=str(status),
        enabled=bool(enabled),
        market=str(market or ""),
        universe_key=str(universe_key or ""),
        as_of=str(as_of or ""),
        decisions=tuple(decisions or ()),
        diagnostic_notes=tuple(_dedupe_notes(diagnostic_notes or [])),
        metadata=dict(GOVERNANCE_METADATA),
    )


def _summary_counts(decisions: tuple[ThemeGovernanceDecision, ...]) -> dict[str, int]:
    counts = {label: 0 for label in GATE_LABELS}
    for decision in decisions:
        label = decision.gate_label if decision.gate_label in counts else "unavailable"
        counts[label] += 1
    return counts


def _numeric_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    lower: float,
    upper: float,
    diagnostics: list[str],
) -> float | None:
    value = payload.get(field_name)
    if value is None or str(value).strip() == "":
        diagnostics.append(f"missing_theme_{field_name}")
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        diagnostics.append(f"malformed_theme_{field_name}")
        return None
    if not math.isfinite(numeric):
        diagnostics.append(f"malformed_theme_{field_name}")
        return None
    return max(float(lower), min(float(upper), numeric))


def _integer_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    diagnostics: list[str],
) -> int | None:
    value = payload.get(field_name)
    if value is None or str(value).strip() == "":
        diagnostics.append(f"missing_theme_{field_name}")
        return None
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        diagnostics.append(f"malformed_theme_{field_name}")
        return None
    if numeric < 0:
        diagnostics.append(f"malformed_theme_{field_name}")
        return None
    return numeric


def _label_rank(label: str) -> int:
    order = {
        "admitted_shadow": 0,
        "watchlist_strong": 1,
        "watchlist_rebuild": 2,
        "umbrella_only": 3,
        "rejected": 4,
        "unavailable": 5,
    }
    return order.get(label, 99)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        raw_items = [str(value)]
    else:
        try:
            raw_items = list(value or [])
        except TypeError:
            raw_items = []
    return _dedupe_notes(str(item).strip() for item in raw_items if str(item).strip())


def _dedupe_notes(items: Any) -> list[str]:
    result: list[str] = []
    for item in list(items or []):
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_int(value: Any) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return 0
    return numeric


def _safe_component(value: Any, default: str) -> str:
    text = str(value or "").strip() or default
    safe = _SAFE_COMPONENT_RE.sub("_", text).strip("_") or default
    return default if safe in {".", ".."} else safe


def _date_segment(as_of: str) -> str:
    digits = "".join(ch for ch in str(as_of or "") if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return _safe_component(as_of, "unknown_date")
