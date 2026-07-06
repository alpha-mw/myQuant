from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.themes import ThemeScanner, ThemeSnapshotStore
from quant_investor.themes.governance import (
    GOVERNANCE_METADATA,
    GOVERNANCE_SCHEMA_VERSION,
    evaluate_theme_governance,
    load_theme_governance_registry,
    write_theme_governance_artifact,
)


SCHEMA_VERSION = "theme_rotation.v1"
_BASE_INTEGRATION_METADATA = {
    "deterministic": True,
    "no_llm": True,
    "no_network": True,
}
_SYMBOL_THEME_FIELDS = {
    "available": False,
    "schema_version": "",
    "status": "missing",
    "symbol_score": 0.0,
    "symbol_score_100": 0.0,
    "primary_theme_id": "",
    "primary_theme_name": "",
    "phase": "",
    "risk_flags": [],
    "theme_score": None,
    "theme_confidence": None,
    "theme_member_count": None,
}


def build_disabled_theme_rotation_metadata(
    *,
    market: str = "",
    universe_key: str = "",
    as_of: str = "",
) -> dict[str, Any]:
    return _empty_theme_rotation_metadata(
        enabled=False,
        status="disabled",
        market=market,
        universe_key=universe_key,
        as_of=as_of,
        diagnostic_notes=["theme_scanner_disabled"],
    )


def build_theme_rotation_metadata(
    *,
    frames: Mapping[str, pd.DataFrame],
    industry_map: Mapping[str, str],
    symbol_market_state: Mapping[str, Mapping[str, Any]],
    market: str,
    universe_key: str,
    as_of: str,
    min_member_count: int = 5,
    top_n: int = 20,
    symbol_limit: int = 300,
    smoothing_window: int = 10,
    smoothing_min_observations: int = 5,
    snapshot_history: list[Mapping[str, Any]] | None = None,
    snapshot_dir: str | Path | None = None,
    history_limit: int = 10,
) -> dict[str, Any]:
    try:
        crowding_context = _resolve_crowding_scan_context(
            snapshot_history=snapshot_history,
            snapshot_dir=snapshot_dir,
            history_limit=history_limit,
            market=market,
            universe_key=universe_key,
        )
        result = ThemeScanner().scan(
            frames=dict(frames or {}),
            industry_map=dict(industry_map or {}),
            symbol_market_state={
                str(symbol): dict(state or {})
                for symbol, state in dict(symbol_market_state or {}).items()
            },
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            min_member_count=int(min_member_count),
            top_n=int(top_n),
            smoothing_window=int(smoothing_window),
            smoothing_min_observations=int(smoothing_min_observations),
            crowding_enabled=crowding_context["enabled"],
            crowding_min_universe=crowding_context["min_universe"],
            snapshot_history=crowding_context["snapshot_history"],
        )
    except Exception as exc:
        return _empty_theme_rotation_metadata(
            enabled=True,
            status="error",
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            diagnostic_notes=[f"theme_scanner_error: {exc}"],
        )

    theme_scores = {
        str(theme_id): score.to_dict() if hasattr(score, "to_dict") else dict(score)
        for theme_id, score in dict(result.theme_scores or {}).items()
    }
    raw_symbol_scores = {
        str(symbol): _safe_float(score)
        for symbol, score in dict(result.symbol_scores or {}).items()
    }
    raw_symbol_smoothed_scores = {
        str(symbol): _safe_float(score)
        for symbol, score in dict(result.symbol_smoothed_scores or {}).items()
    }
    selected_symbols = _bounded_symbols(raw_symbol_scores, symbol_limit)
    symbol_scores = {
        symbol: raw_symbol_scores[symbol]
        for symbol in selected_symbols
    }
    symbol_smoothed_scores = {
        symbol: raw_symbol_smoothed_scores[symbol]
        for symbol in selected_symbols
        if symbol in raw_symbol_smoothed_scores
    }
    symbol_primary_theme = {
        symbol: str(result.symbol_primary_theme.get(symbol, ""))
        for symbol in selected_symbols
        if symbol in result.symbol_primary_theme
    }
    symbol_phase = {
        symbol: str(result.symbol_phase.get(symbol, ""))
        for symbol in selected_symbols
        if symbol in result.symbol_phase
    }
    symbol_risk_flags = {
        symbol: [str(flag) for flag in list(result.symbol_risk_flags.get(symbol, []) or [])]
        for symbol in selected_symbols
    }
    top_themes = [
        _compact_top_theme(theme)
        for theme in sorted(
            theme_scores.values(),
            key=lambda item: (-_safe_float(item.get("score", 0.0)), str(item.get("theme_id", ""))),
        )[: max(int(top_n), 0)]
    ]
    metadata = {
        **_BASE_INTEGRATION_METADATA,
        **dict(result.metadata or {}),
        "theme_count": len(theme_scores),
        "scanned_symbol_count": int(
            _safe_float(
                dict(result.metadata or {}).get("scanned_symbol_count", len(industry_map or {}))
            )
        ),
        "symbol_limit": max(int(symbol_limit), 0),
        "truncated_symbol_count": max(len(raw_symbol_scores) - len(selected_symbols), 0),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "status": "success",
        "market": str(result.market or market),
        "universe_key": str(result.universe_key or universe_key),
        "as_of": str(result.as_of or as_of),
        "theme_scores": theme_scores,
        "symbol_scores": symbol_scores,
        "symbol_smoothed_scores": symbol_smoothed_scores,
        "symbol_primary_theme": symbol_primary_theme,
        "symbol_phase": symbol_phase,
        "symbol_risk_flags": symbol_risk_flags,
        "top_themes": top_themes,
        "diagnostic_notes": [],
        "metadata": metadata,
    }


def persist_theme_rotation_snapshot(
    *,
    theme_rotation: Mapping[str, Any],
    enabled: bool,
    root_dir: str | Path,
    market: str,
    universe_key: str,
    as_of: str,
    run_id: str = "",
    save_disabled: bool = False,
) -> dict[str, Any]:
    metadata = dict(_BASE_INTEGRATION_METADATA)
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "path": "",
            "error": "",
            "diagnostic_notes": ["theme_snapshot_disabled"],
            "metadata": metadata,
        }

    try:
        rotation_status = ""
        if isinstance(theme_rotation, Mapping):
            rotation_status = str(theme_rotation.get("status") or "").strip().lower()
        if rotation_status == "disabled" and not save_disabled:
            return {
                "enabled": True,
                "status": "skipped",
                "path": "",
                "error": "",
                "diagnostic_notes": ["theme_rotation_disabled_not_saved"],
                "metadata": metadata,
            }

        path = ThemeSnapshotStore(root_dir).save(
            theme_rotation,
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            run_id=run_id,
        )
        return {
            "enabled": True,
            "status": "success",
            "path": str(path),
            "error": "",
            "diagnostic_notes": [],
            "metadata": metadata,
        }
    except Exception as exc:
        error = str(exc)
        return {
            "enabled": True,
            "status": "error",
            "path": "",
            "error": error,
            "diagnostic_notes": [f"theme_snapshot_error: {error}"],
            "metadata": metadata,
        }


def build_disabled_theme_governance_metadata(
    *,
    market: str = "",
    universe_key: str = "",
    as_of: str = "",
) -> dict[str, Any]:
    return _empty_theme_governance_metadata(
        enabled=False,
        status="disabled",
        market=market,
        universe_key=universe_key,
        as_of=as_of,
        diagnostic_notes=["theme_governance_disabled"],
    )


def build_theme_governance_metadata(
    *,
    theme_rotation: Mapping[str, Any],
    enabled: bool,
    registry_path: str | Path | None = None,
    snapshot_history: list[Mapping[str, Any]] | None = None,
    snapshot_dir: str | Path | None = None,
    history_limit: int = 10,
    market: str = "",
    universe_key: str = "",
    as_of: str = "",
) -> dict[str, Any]:
    if not enabled:
        return build_disabled_theme_governance_metadata(
            market=market,
            universe_key=universe_key,
            as_of=as_of,
        )
    try:
        registry = load_theme_governance_registry(registry_path)
        history = list(snapshot_history or [])
        if not history and snapshot_dir:
            try:
                history = ThemeSnapshotStore(snapshot_dir).load_recent(
                    market=market or str(theme_rotation.get("market") or "CN"),
                    universe_key=universe_key or str(theme_rotation.get("universe_key") or "") or None,
                    limit=max(int(history_limit or 10), 0),
                )
            except Exception:
                history = []
        result = evaluate_theme_governance(
            theme_rotation,
            registry=registry,
            history=history,
        )
        payload = result.to_dict()
        payload["market"] = str(payload.get("market") or market or "")
        payload["universe_key"] = str(payload.get("universe_key") or universe_key or "")
        payload["as_of"] = str(payload.get("as_of") or as_of or "")
        return payload
    except Exception as exc:
        return _empty_theme_governance_metadata(
            enabled=True,
            status="error",
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            diagnostic_notes=[f"theme_governance_error: {exc}"],
        )


def persist_theme_governance_artifact(
    *,
    theme_governance: Mapping[str, Any],
    enabled: bool,
    root_dir: str | Path,
    market: str,
    universe_key: str,
    as_of: str,
    run_id: str = "",
) -> dict[str, Any]:
    metadata = dict(GOVERNANCE_METADATA)
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "path": "",
            "error": "",
            "diagnostic_notes": ["theme_governance_artifact_disabled"],
            "metadata": metadata,
        }
    try:
        governance_status = ""
        if isinstance(theme_governance, Mapping):
            governance_status = str(theme_governance.get("status") or "").strip().lower()
        if governance_status == "disabled":
            return {
                "enabled": True,
                "status": "skipped",
                "path": "",
                "error": "",
                "diagnostic_notes": ["theme_governance_disabled_not_saved"],
                "metadata": metadata,
            }
        path = write_theme_governance_artifact(
            theme_governance,
            root_dir=root_dir,
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            run_id=run_id,
        )
        return {
            "enabled": True,
            "status": "success",
            "path": str(path),
            "error": "",
            "diagnostic_notes": [],
            "metadata": metadata,
        }
    except Exception as exc:
        error = str(exc)
        return {
            "enabled": True,
            "status": "error",
            "path": "",
            "error": error,
            "diagnostic_notes": [f"theme_governance_artifact_error: {error}"],
            "metadata": metadata,
        }


def extract_symbol_theme_metadata(
    *,
    global_context: Any,
    symbol: str,
) -> dict[str, Any]:
    """Extract compact per-symbol theme metadata without affecting decisions."""

    symbol_text = str(symbol or "")
    try:
        metadata = getattr(global_context, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            return _empty_symbol_theme_metadata(status="missing")

        if "theme_rotation" in metadata:
            rotation_payload = metadata.get("theme_rotation")
            if not isinstance(rotation_payload, Mapping):
                return _empty_symbol_theme_metadata(
                    schema_version=SCHEMA_VERSION,
                    status="error",
                )
            schema_version = str(rotation_payload.get("schema_version") or SCHEMA_VERSION)
            status = str(rotation_payload.get("status") or "").strip().lower()
            if status == "disabled":
                return _empty_symbol_theme_metadata(
                    schema_version=schema_version,
                    status="disabled",
                )
            if status == "error":
                return _empty_symbol_theme_metadata(
                    schema_version=schema_version,
                    status="error",
                )
            return _extract_symbol_theme_payload(
                payload=rotation_payload,
                symbol=symbol_text,
                schema_version=schema_version,
            )

        if not any(
            key in metadata
            for key in (
                "symbol_theme_score",
                "symbol_primary_theme",
                "symbol_theme_phase",
                "theme_scores",
            )
        ):
            return _empty_symbol_theme_metadata(status="missing")

        fallback_payload = {
            "symbol_scores": metadata.get("symbol_theme_score"),
            "symbol_primary_theme": metadata.get("symbol_primary_theme"),
            "symbol_phase": metadata.get("symbol_theme_phase"),
            "symbol_risk_flags": metadata.get("symbol_risk_flags"),
            "theme_scores": metadata.get("theme_scores"),
            "top_themes": metadata.get("top_themes"),
        }
        return _extract_symbol_theme_payload(
            payload=fallback_payload,
            symbol=symbol_text,
            schema_version=SCHEMA_VERSION,
        )
    except Exception:
        return _empty_symbol_theme_metadata(
            schema_version=SCHEMA_VERSION,
            status="error",
        )


def build_theme_risk_constraints(
    *,
    global_context: Any,
    symbols: list[str],
    enabled: bool,
    overextended_gross_cap: float = 0.60,
    overextended_max_weight: float = 0.10,
    distribution_gross_cap: float = 0.45,
    distribution_max_weight: float = 0.08,
    fake_breakout_max_weight: float = 0.10,
) -> dict[str, Any]:
    """Build deterministic RiskGuard constraints from compact theme metadata."""

    checked_symbols = _dedupe_flags(symbols)
    overextended_gross_cap = _clamp(
        _safe_float(overextended_gross_cap, 0.60),
        0.0,
        1.0,
    )
    overextended_max_weight = _clamp(
        _safe_float(overextended_max_weight, 0.10),
        0.0,
        1.0,
    )
    distribution_gross_cap = _clamp(
        _safe_float(distribution_gross_cap, 0.45),
        0.0,
        1.0,
    )
    distribution_max_weight = _clamp(
        _safe_float(distribution_max_weight, 0.08),
        0.0,
        1.0,
    )
    fake_breakout_max_weight = _clamp(
        _safe_float(fake_breakout_max_weight, 0.10),
        0.0,
        1.0,
    )
    constraints = _empty_theme_risk_constraints(
        enabled=bool(enabled),
        symbols_checked=len(checked_symbols),
        overextended_gross_cap=overextended_gross_cap,
        overextended_max_weight=overextended_max_weight,
        distribution_gross_cap=distribution_gross_cap,
        distribution_max_weight=distribution_max_weight,
        fake_breakout_max_weight=fake_breakout_max_weight,
    )
    if not enabled:
        constraints["diagnostic_notes"] = ["theme_risk_guard_disabled"]
        return constraints

    risk_by_symbol: dict[str, Any] = {}
    position_limits: dict[str, float] = {}
    risk_flags: list[str] = []
    gross_cap: float | None = None
    action_cap = ""

    def add_flag(flag: str) -> None:
        text = str(flag or "").strip()
        if text and text not in risk_flags:
            risk_flags.append(text)

    def apply_symbol_cap(symbol: str, cap: float) -> None:
        cap = _clamp(float(cap), 0.0, 1.0)
        current = position_limits.get(symbol)
        position_limits[symbol] = cap if current is None else min(current, cap)

    def apply_gross_cap(cap: float) -> None:
        nonlocal gross_cap
        cap = _clamp(float(cap), 0.0, 1.0)
        gross_cap = cap if gross_cap is None else min(gross_cap, cap)

    for symbol in checked_symbols:
        metadata = extract_symbol_theme_metadata(
            global_context=global_context,
            symbol=symbol,
        )
        if not bool(metadata.get("available", False)):
            continue

        phase = str(metadata.get("phase") or "").strip().lower()
        flags = _dedupe_flags(metadata.get("risk_flags", []))
        flag_set = set(flags)
        symbol_has_risk = False

        if phase == "distribution" or "theme_distribution_risk" in flag_set:
            apply_symbol_cap(symbol, distribution_max_weight)
            apply_gross_cap(distribution_gross_cap)
            add_flag("theme_distribution_risk")
            action_cap = "hold"
            symbol_has_risk = True

        if (
            phase == "overextended"
            or "theme_overextended" in flag_set
            or "theme_overextended_no_chase" in flag_set
        ):
            apply_symbol_cap(symbol, overextended_max_weight)
            apply_gross_cap(overextended_gross_cap)
            add_flag("theme_overextended")
            action_cap = "hold"
            symbol_has_risk = True

        if "theme_fake_breakout_risk" in flag_set:
            apply_symbol_cap(symbol, fake_breakout_max_weight)
            add_flag("theme_fake_breakout_risk")
            symbol_has_risk = True

        if "theme_low_breadth" in flag_set:
            add_flag("theme_low_breadth")
            symbol_has_risk = True

        if symbol_has_risk:
            risk_by_symbol[symbol] = {
                "available": True,
                "phase": phase,
                "primary_theme_id": str(metadata.get("primary_theme_id") or ""),
                "primary_theme_name": str(metadata.get("primary_theme_name") or ""),
                "symbol_score": _clamp(
                    _safe_float(metadata.get("symbol_score", 0.0)),
                    0.0,
                    1.0,
                ),
                "risk_flags": flags,
            }

    constraints["theme_risk_by_symbol"] = risk_by_symbol
    constraints["theme_risk_flags"] = risk_flags
    constraints["theme_position_limits"] = position_limits
    constraints["theme_action_cap"] = action_cap
    constraints["theme_gross_exposure_cap"] = gross_cap
    constraints["metadata"] = {
        **dict(constraints["metadata"]),
        "symbols_with_theme_risk": len(risk_by_symbol),
    }
    if not risk_by_symbol:
        constraints["diagnostic_notes"] = ["theme_risk_guard_no_theme_risk"]
    return constraints


def build_theme_portfolio_constraints(
    *,
    global_context: Any,
    symbols: list[str],
    enabled: bool,
    max_theme_exposure: float = 0.35,
    overextended_max_theme_exposure: float = 0.25,
    distribution_max_theme_exposure: float = 0.15,
) -> dict[str, Any]:
    """Build deterministic PortfolioConstructor theme exposure caps."""

    checked_symbols = _dedupe_flags(symbols)
    max_theme_exposure = _clamp(
        _safe_float(max_theme_exposure, 0.35),
        0.0,
        1.0,
    )
    overextended_max_theme_exposure = _clamp(
        _safe_float(overextended_max_theme_exposure, 0.25),
        0.0,
        1.0,
    )
    distribution_max_theme_exposure = _clamp(
        _safe_float(distribution_max_theme_exposure, 0.15),
        0.0,
        1.0,
    )
    constraints = _empty_theme_portfolio_constraints(
        enabled=bool(enabled),
        symbols_checked=len(checked_symbols),
    )
    if not enabled:
        constraints["diagnostic_notes"] = ["theme_portfolio_cap_disabled"]
        return constraints

    exposure_map: dict[str, dict[str, Any]] = {}
    theme_names: dict[str, str] = {}
    theme_phases: dict[str, str] = {}
    overextended_themes: set[str] = set()
    distribution_themes: set[str] = set()

    for symbol in checked_symbols:
        metadata = extract_symbol_theme_metadata(
            global_context=global_context,
            symbol=symbol,
        )
        if not bool(metadata.get("available", False)):
            continue

        theme_id = str(metadata.get("primary_theme_id") or "").strip()
        if not theme_id:
            continue
        phase = str(metadata.get("phase") or "").strip().lower()
        flags = _dedupe_flags(metadata.get("risk_flags", []))
        flag_set = set(flags)

        exposure_map[symbol] = {
            "primary_theme_id": theme_id,
            "primary_theme_name": str(metadata.get("primary_theme_name") or ""),
            "phase": phase,
            "symbol_score": _clamp(
                _safe_float(metadata.get("symbol_score", 0.0)),
                0.0,
                1.0,
            ),
            "risk_flags": flags,
        }
        theme_names.setdefault(
            theme_id,
            str(metadata.get("primary_theme_name") or theme_id),
        )
        theme_phases.setdefault(theme_id, phase)

        if (
            phase == "overextended"
            or "theme_overextended" in flag_set
            or "theme_overextended_no_chase" in flag_set
        ):
            overextended_themes.add(theme_id)
        if phase == "distribution" or "theme_distribution_risk" in flag_set:
            distribution_themes.add(theme_id)

    theme_caps: dict[str, float] = {}
    for theme_id in sorted(theme_names):
        cap = max_theme_exposure
        if theme_id in overextended_themes:
            cap = min(cap, overextended_max_theme_exposure)
            theme_phases[theme_id] = "overextended"
        if theme_id in distribution_themes:
            cap = min(cap, distribution_max_theme_exposure)
            theme_phases[theme_id] = "distribution"
        theme_caps[theme_id] = _clamp(cap, 0.0, 1.0)

    constraints["theme_exposure_map"] = exposure_map
    constraints["theme_caps"] = theme_caps
    constraints["theme_names"] = theme_names
    constraints["theme_phases"] = theme_phases
    constraints["metadata"] = {
        **dict(constraints["metadata"]),
        "symbols_with_theme": len(exposure_map),
        "theme_count": len(theme_caps),
    }
    if not exposure_map:
        constraints["diagnostic_notes"] = ["theme_portfolio_cap_no_theme_data"]
    return constraints


def _empty_theme_rotation_metadata(
    *,
    enabled: bool,
    status: str,
    market: str,
    universe_key: str,
    as_of: str,
    diagnostic_notes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": bool(enabled),
        "status": str(status),
        "market": str(market or ""),
        "universe_key": str(universe_key or ""),
        "as_of": str(as_of or ""),
        "theme_scores": {},
        "symbol_scores": {},
        "symbol_smoothed_scores": {},
        "symbol_primary_theme": {},
        "symbol_phase": {},
        "symbol_risk_flags": {},
        "top_themes": [],
        "diagnostic_notes": list(diagnostic_notes),
        "metadata": dict(_BASE_INTEGRATION_METADATA),
    }


def _empty_theme_governance_metadata(
    *,
    enabled: bool,
    status: str,
    market: str,
    universe_key: str,
    as_of: str,
    diagnostic_notes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": GOVERNANCE_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "status": str(status),
        "market": str(market or ""),
        "universe_key": str(universe_key or ""),
        "as_of": str(as_of or ""),
        "decisions": [],
        "top_themes": [],
        "summary_counts": {
            "admitted_shadow": 0,
            "watchlist_strong": 0,
            "watchlist_rebuild": 0,
            "rejected": 0,
            "umbrella_only": 0,
            "unavailable": 0,
        },
        "diagnostic_notes": list(diagnostic_notes),
        "metadata": dict(GOVERNANCE_METADATA),
    }


def _empty_theme_risk_constraints(
    *,
    enabled: bool,
    symbols_checked: int,
    overextended_gross_cap: float,
    overextended_max_weight: float,
    distribution_gross_cap: float,
    distribution_max_weight: float,
    fake_breakout_max_weight: float,
) -> dict[str, Any]:
    return {
        "theme_risk_guard_enabled": bool(enabled),
        "theme_risk_by_symbol": {},
        "theme_risk_flags": [],
        "theme_position_limits": {},
        "theme_action_cap": "",
        "theme_gross_exposure_cap": None,
        "theme_overextended_gross_cap": float(overextended_gross_cap),
        "theme_overextended_max_weight": float(overextended_max_weight),
        "theme_distribution_gross_cap": float(distribution_gross_cap),
        "theme_distribution_max_weight": float(distribution_max_weight),
        "theme_fake_breakout_max_weight": float(fake_breakout_max_weight),
        "diagnostic_notes": [],
        "metadata": {
            "deterministic": True,
            "no_llm": True,
            "no_network": True,
            "enabled": bool(enabled),
            "symbols_checked": int(symbols_checked),
            "symbols_with_theme_risk": 0,
        },
    }


def _empty_theme_portfolio_constraints(
    *,
    enabled: bool,
    symbols_checked: int,
) -> dict[str, Any]:
    return {
        "theme_portfolio_cap_enabled": bool(enabled),
        "theme_exposure_map": {},
        "theme_caps": {},
        "theme_names": {},
        "theme_phases": {},
        "diagnostic_notes": [],
        "metadata": {
            "deterministic": True,
            "no_llm": True,
            "no_network": True,
            "enabled": bool(enabled),
            "symbols_checked": int(symbols_checked),
            "symbols_with_theme": 0,
            "theme_count": 0,
        },
    }


def _bounded_symbols(
    symbol_scores: Mapping[str, float],
    symbol_limit: int,
) -> list[str]:
    limit = max(int(symbol_limit), 0)
    if limit <= 0:
        return []
    ranked = sorted(
        symbol_scores,
        key=lambda symbol: (-_safe_float(symbol_scores.get(symbol, 0.0)), str(symbol)),
    )
    return ranked[:limit]


def _extract_symbol_theme_payload(
    *,
    payload: Mapping[str, Any],
    symbol: str,
    schema_version: str,
) -> dict[str, Any]:
    symbol_scores = _mapping_or_empty(payload.get("symbol_scores"))
    if symbol not in symbol_scores:
        return _empty_symbol_theme_metadata(
            schema_version=schema_version,
            status="unavailable",
        )

    symbol_score = _clamp(_safe_float(symbol_scores.get(symbol, 0.0)), 0.0, 1.0)
    primary_theme = str(
        _mapping_or_empty(payload.get("symbol_primary_theme")).get(symbol, "") or ""
    )
    phase = str(_mapping_or_empty(payload.get("symbol_phase")).get(symbol, "") or "")
    risk_flags = _dedupe_flags(
        _mapping_or_empty(payload.get("symbol_risk_flags")).get(symbol, [])
    )
    theme_scores = _mapping_or_empty(payload.get("theme_scores"))
    top_themes = payload.get("top_themes", [])
    top_theme = _find_top_theme(top_themes, primary_theme)
    theme_score = theme_scores.get(primary_theme) if primary_theme else None
    theme_name = (
        _theme_value(theme_score, "theme_name")
        or _theme_value(top_theme, "theme_name")
        or primary_theme
    )

    score_value = _theme_value(theme_score, "score")
    if score_value is None:
        score_value = _theme_value(top_theme, "score")
    confidence_value = _theme_value(theme_score, "confidence")
    if confidence_value is None:
        confidence_value = _theme_value(top_theme, "confidence")
    member_count_value = _theme_value(theme_score, "member_count")
    if member_count_value is None:
        member_count_value = _theme_value(top_theme, "member_count")

    return {
        "available": True,
        "schema_version": str(schema_version or SCHEMA_VERSION),
        "status": "success",
        "symbol_score": symbol_score,
        "symbol_score_100": symbol_score * 100.0,
        "primary_theme_id": primary_theme,
        "primary_theme_name": str(theme_name or ""),
        "phase": phase,
        "risk_flags": risk_flags,
        "theme_score": _optional_float(score_value),
        "theme_confidence": _optional_float(confidence_value),
        "theme_member_count": _optional_int(member_count_value),
    }


def _empty_symbol_theme_metadata(
    *,
    schema_version: str = "",
    status: str,
) -> dict[str, Any]:
    metadata = dict(_SYMBOL_THEME_FIELDS)
    metadata["risk_flags"] = []
    metadata["schema_version"] = str(schema_version or "")
    metadata["status"] = str(status or "missing")
    return metadata


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _theme_value(theme: Any, key: str) -> Any:
    if isinstance(theme, Mapping):
        return theme.get(key)
    return getattr(theme, key, None)


def _find_top_theme(top_themes: Any, theme_id: str) -> Any:
    if not theme_id or isinstance(top_themes, (str, bytes)):
        return None
    try:
        iterator = iter(top_themes or [])
    except TypeError:
        return None
    for item in iterator:
        if str(_theme_value(item, "theme_id") or "") == theme_id:
            return item
    return None


def _dedupe_flags(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        return []
    try:
        items = list(value or [])
    except TypeError:
        return []
    flags: list[str] = []
    seen: set[str] = set()
    for item in items:
        flag = str(item or "").strip()
        if not flag or flag in seen:
            continue
        seen.add(flag)
        flags.append(flag)
    return flags


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_int(value: Any) -> int | None:
    numeric = _optional_float(value)
    return int(numeric) if numeric is not None else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _resolve_crowding_scan_context(
    *,
    snapshot_history: list[Mapping[str, Any]] | None,
    snapshot_dir: str | Path | None,
    history_limit: int,
    market: str,
    universe_key: str,
) -> dict[str, Any]:
    enabled = False
    min_universe = 30
    resolved_snapshot_dir = snapshot_dir
    try:
        from quant_investor.config import Config

        enabled = bool(getattr(Config, "THEME_CROWDING_ENABLED", False))
        min_universe = max(int(getattr(Config, "THEME_CROWDING_MIN_UNIVERSE", 30) or 30), 1)
        if resolved_snapshot_dir is None:
            resolved_snapshot_dir = str(
                getattr(Config, "THEME_SNAPSHOT_DIR", "results/theme_snapshots")
                or "results/theme_snapshots"
            )
    except Exception:
        pass

    history = list(snapshot_history or [])
    if enabled and not history and resolved_snapshot_dir:
        try:
            history = ThemeSnapshotStore(resolved_snapshot_dir).load_recent(
                market=market or "CN",
                universe_key=universe_key or None,
                limit=max(int(history_limit or 10), 0),
            )
        except Exception:
            history = []
    return {
        "enabled": enabled,
        "min_universe": min_universe,
        "snapshot_history": history,
    }


def _compact_top_theme(theme: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "theme_id": str(theme.get("theme_id", "")),
        "theme_name": str(theme.get("theme_name", "")),
        "score": _safe_float(theme.get("score", 0.0)),
        "raw_score": _safe_float(theme.get("raw_score", theme.get("score", 0.0))),
        "smoothed_score": _optional_float(theme.get("smoothed_score")),
        "heat_10d": _optional_float(theme.get("heat_10d")),
        "heat_delta_5d": _optional_float(theme.get("heat_delta_5d")),
        "persistence_count": int(_safe_float(theme.get("persistence_count", 0))),
        "trend_state": str(theme.get("trend_state", "")),
        "smoothing_observation_count": int(
            _safe_float(theme.get("smoothing_observation_count", 0))
        ),
        "smoothing_status": str(theme.get("smoothing_status", "")),
        "phase": str(theme.get("phase", "")),
        "confidence": _safe_float(theme.get("confidence", 0.0)),
        "member_count": int(_safe_float(theme.get("member_count", 0))),
        "top_symbols": [str(symbol) for symbol in list(theme.get("top_symbols", []) or [])],
        "risk_flags": [str(flag) for flag in list(theme.get("risk_flags", []) or [])],
        "theme_turnover_share": _safe_float(theme.get("theme_turnover_share", 0.0)),
        "turnover_share_sma10": _optional_float(theme.get("turnover_share_sma10")),
        "turnover_share_stretch": _safe_float(theme.get("turnover_share_stretch", 0.0)),
        "turnover_share_delta_5d": _optional_float(theme.get("turnover_share_delta_5d")),
        "turnover_share_trend": str(theme.get("turnover_share_trend", "")),
        "theme_limitup_ratio": _safe_float(theme.get("theme_limitup_ratio", 0.0)),
        "limitup_norm": _safe_float(theme.get("limitup_norm", 0.0)),
        "member_turnover_concentration": _safe_float(
            theme.get("member_turnover_concentration", 0.0)
        ),
        "crowding_risk": _safe_float(theme.get("crowding_risk", 0.0)),
        "crowding_status": str(theme.get("crowding_status", "")),
        "crowding_diagnostic_notes": [
            str(note) for note in list(theme.get("crowding_diagnostic_notes", []) or [])
        ],
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default
