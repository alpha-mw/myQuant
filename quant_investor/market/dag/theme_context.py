from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.themes import (
    PeVcKnowledgeStore,
    ThemeEvidenceEvent,
    ThemeMembership,
    ThemeMembershipStore,
    ThemeScanner,
    ThemeSnapshotStore,
    ThemeTaxonomy,
    active_memberships_by_symbol,
    evaluate_theme_protocol_v2,
)
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
    concept_membership_enabled: bool | None = None,
    concept_membership_path: str | Path | None = None,
    concept_membership_required: bool | None = None,
    concept_primary_margin: float | None = None,
    theme_memberships: list[Mapping[str, Any]] | None = None,
    membership_v2_enabled: bool | None = None,
    membership_v2_path: str | Path | None = None,
    membership_v2_required: bool | None = None,
    membership_v2_expected_sha256: str | None = None,
    protocol_v2_enabled: bool | None = None,
    taxonomy_v2_path: str | Path | None = None,
    evidence_event_v1_path: str | Path | None = None,
    pevc_canonical_path: str | Path | None = None,
    protocol_previous_states: Mapping[str, Mapping[str, Any]] | None = None,
    protocol_downstream_gates: Mapping[str, Mapping[str, Any]] | None = None,
    markov_regime: str = "",
    formal_v2_enabled: bool | None = None,
    formal_v2_kill_switch: bool | None = None,
) -> dict[str, Any]:
    try:
        crowding_context = _resolve_crowding_scan_context(
            snapshot_history=snapshot_history,
            snapshot_dir=snapshot_dir,
            history_limit=history_limit,
            market=market,
            universe_key=universe_key,
        )
        concept_context = _resolve_concept_membership_scan_context(
            concept_membership_enabled=concept_membership_enabled,
            concept_membership_path=concept_membership_path,
            concept_membership_required=concept_membership_required,
            concept_primary_margin=concept_primary_margin,
            theme_memberships=theme_memberships,
        )
        membership_v2_context = _resolve_membership_v2_scan_context(
            enabled=membership_v2_enabled,
            path=membership_v2_path,
            required=membership_v2_required,
            expected_sha256=membership_v2_expected_sha256,
            as_of=as_of,
        )
        if (
            concept_context["enabled"]
            and concept_context["required"]
            and concept_context["status"] != "success"
        ):
            payload = _empty_theme_rotation_metadata(
                enabled=True,
                status="error",
                market=market,
                universe_key=universe_key,
                as_of=as_of,
                diagnostic_notes=list(concept_context["diagnostic_notes"]),
            )
            payload["metadata"] = {
                **dict(payload.get("metadata", {}) or {}),
                **_concept_membership_metadata(concept_context),
            }
            return payload
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
            concept_membership_enabled=concept_context["enabled"],
            concept_primary_margin=concept_context["primary_margin"],
            theme_memberships=concept_context["memberships"],
            membership_v2_enabled=membership_v2_context["enabled"],
            theme_memberships_v2=membership_v2_context["memberships"],
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
    # Machine-consumed membership and score maps must remain complete.  The
    # display limit is applied only to the explicitly named display payload so
    # changing a dashboard metadata knob cannot change the formal candidate set.
    display_symbols = _bounded_symbols(raw_symbol_scores, symbol_limit)
    machine_symbols = sorted(raw_symbol_scores)
    symbol_scores = dict(raw_symbol_scores)
    symbol_smoothed_scores = {
        symbol: raw_symbol_smoothed_scores[symbol]
        for symbol in machine_symbols
        if symbol in raw_symbol_smoothed_scores
    }
    symbol_primary_theme = {
        symbol: str(result.symbol_primary_theme.get(symbol, ""))
        for symbol in machine_symbols
        if symbol in result.symbol_primary_theme
    }
    symbol_theme_memberships = {
        symbol: [
            str(theme_id)
            for theme_id in list(result.symbol_theme_memberships.get(symbol, []) or [])
        ]
        for symbol in machine_symbols
        if symbol in result.symbol_theme_memberships
    }
    symbol_theme_membership_details = {
        symbol: [
            dict(detail)
            for detail in list(
                result.symbol_theme_membership_details.get(symbol, []) or []
            )
        ]
        for symbol in machine_symbols
        if symbol in result.symbol_theme_membership_details
    }
    symbol_phase = {
        symbol: str(result.symbol_phase.get(symbol, ""))
        for symbol in machine_symbols
        if symbol in result.symbol_phase
    }
    symbol_risk_flags = {
        symbol: [str(flag) for flag in list(result.symbol_risk_flags.get(symbol, []) or [])]
        for symbol in machine_symbols
    }
    display = {
        "symbol_scores": {
            symbol: raw_symbol_scores[symbol]
            for symbol in display_symbols
        },
        "symbol_smoothed_scores": {
            symbol: raw_symbol_smoothed_scores[symbol]
            for symbol in display_symbols
            if symbol in raw_symbol_smoothed_scores
        },
        "symbol_primary_theme": {
            symbol: str(result.symbol_primary_theme.get(symbol, ""))
            for symbol in display_symbols
            if symbol in result.symbol_primary_theme
        },
        "symbol_theme_memberships": {
            symbol: [
                str(theme_id)
                for theme_id in list(result.symbol_theme_memberships.get(symbol, []) or [])
            ]
            for symbol in display_symbols
            if symbol in result.symbol_theme_memberships
        },
        "symbol_theme_membership_details": {
            symbol: [
                dict(detail)
                for detail in list(
                    result.symbol_theme_membership_details.get(symbol, []) or []
                )
            ]
            for symbol in display_symbols
            if symbol in result.symbol_theme_membership_details
        },
        "symbol_phase": {
            symbol: str(result.symbol_phase.get(symbol, ""))
            for symbol in display_symbols
            if symbol in result.symbol_phase
        },
        "symbol_risk_flags": {
            symbol: [str(flag) for flag in list(result.symbol_risk_flags.get(symbol, []) or [])]
            for symbol in display_symbols
        },
    }
    top_themes = [
        _compact_top_theme(theme)
        for theme in sorted(
            theme_scores.values(),
            key=lambda item: (
                -_safe_float(item.get("effective_score", item.get("score", 0.0))),
                str(item.get("theme_id", "")),
            ),
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
        "machine_symbol_count": len(machine_symbols),
        "display_symbol_count": len(display_symbols),
        "truncated_symbol_count": max(len(raw_symbol_scores) - len(display_symbols), 0),
        "symbol_limit_applies_to": "display_metadata_only",
    }
    metadata = {
        **metadata,
        **_concept_membership_metadata(concept_context),
        **_membership_v2_metadata(membership_v2_context),
        "concept_membership_diagnostic_notes": _dedupe_flags(
            [
                *list(metadata.get("concept_membership_diagnostic_notes", []) or []),
                *list(concept_context.get("diagnostic_notes", []) or []),
            ]
        ),
    }
    canonical_membership_details: dict[str, list[dict[str, Any]]] = {}
    for raw_membership in list(membership_v2_context.get("memberships") or []):
        if not isinstance(raw_membership, Mapping):
            continue
        symbol = str(raw_membership.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        canonical_membership_details.setdefault(symbol, []).append(
            {
                **dict(raw_membership),
                "pit_membership": True,
                "canonical_membership_v2": True,
            }
        )
    protocol_v2 = _build_protocol_v2_metadata(
        enabled=protocol_v2_enabled,
        theme_scores=theme_scores,
        symbol_theme_memberships={
            symbol: [
                str(detail.get("theme_id") or "")
                for detail in details
                if str(detail.get("theme_id") or "")
            ]
            for symbol, details in canonical_membership_details.items()
        },
        symbol_theme_membership_details=canonical_membership_details,
        membership_v2_context=membership_v2_context,
        as_of=str(result.as_of or as_of),
        taxonomy_path=taxonomy_v2_path,
        evidence_path=evidence_event_v1_path,
        pevc_path=pevc_canonical_path,
        previous_states=(
            protocol_previous_states
            or _resolve_protocol_previous_states(
                snapshots=crowding_context.get("snapshot_history", []),
                snapshot_dir=snapshot_dir,
                history_limit=history_limit,
                market=market,
                universe_key=universe_key,
                as_of=str(result.as_of or as_of),
            )
        ),
        downstream_gates=protocol_downstream_gates,
        markov_regime=markov_regime,
        formal_enabled=formal_v2_enabled,
        formal_kill_switch=formal_v2_kill_switch,
        valid_trading_dates=_protocol_trading_dates(
            frames,
            as_of=str(result.as_of or as_of),
        ),
    )
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
        "symbol_theme_memberships": symbol_theme_memberships,
        "symbol_theme_membership_details": symbol_theme_membership_details,
        "symbol_phase": symbol_phase,
        "symbol_risk_flags": symbol_risk_flags,
        "display": display,
        "protocol_v2": protocol_v2,
        "membership_v2": _membership_v2_metadata(membership_v2_context),
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
            "channel_open": False,
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
    constraints["theme_tactical_lane"] = _theme_tactical_lane_constraints(
        global_context=global_context,
        symbols=checked_symbols,
    )
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
        "symbol_theme_memberships": {},
        "symbol_theme_membership_details": {},
        "symbol_phase": {},
        "symbol_risk_flags": {},
        "display": {},
        "protocol_v2": {
            "schema_version": "theme_protocol.v2",
            "status": "disabled" if status == "disabled" else "blocked",
            "observer_enabled": False,
            "formal_enabled": False,
            "formal_kill_switch": True,
            "rollback_status": "observer_only",
            "rollback_reason": f"theme_rotation_{status}",
            "formal_pool": [],
            "prequalified_pool": [],
            "forced_theme_count": 0,
        },
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
        "theme_tactical_lane": {
            "enabled": False,
            "status": "disabled",
            "regime": "",
            "non_tech_symbols": [],
            "nav_cap": 0.0,
            "max_positions": 0,
            "protocol_hash": "",
            "formal_kill_switch": True,
        },
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


def _theme_tactical_lane_constraints(
    *,
    global_context: Any,
    symbols: list[str],
) -> dict[str, Any]:
    context_metadata = _mapping_or_empty(getattr(global_context, "metadata", {}))
    rotation = _mapping_or_empty(context_metadata.get("theme_rotation"))
    protocol = _mapping_or_empty(rotation.get("protocol_v2"))
    cap = _mapping_or_empty(protocol.get("tactical_lane_cap"))
    states = _mapping_or_empty(protocol.get("states"))
    details_by_symbol = _mapping_or_empty(
        rotation.get("symbol_theme_membership_details")
    )
    prequalified = {
        str(theme_id)
        for theme_id in list(protocol.get("prequalified_pool") or [])
        if str(theme_id)
    }
    reconciliation = _mapping_or_empty(rotation.get("formal_reconciliation"))
    reconciled_by_symbol = _mapping_or_empty(reconciliation.get("per_symbol"))
    protocol_hash = str(protocol.get("protocol_hash") or "")
    prospective_active = (
        str(protocol.get("status") or "") in {"prequalified", "formal"}
        and protocol.get("formal_enabled") is True
        and protocol.get("formal_kill_switch") is not True
        and len(protocol_hash) == 64
    )
    non_tech_symbols: list[str] = []
    unclassified_symbols: list[str] = []
    classification_blockers: dict[str, list[str]] = {}
    protocol_as_of = str(protocol.get("as_of") or rotation.get("as_of") or "")
    for symbol in symbols:
        reconciled = _mapping_or_empty(reconciled_by_symbol.get(symbol))
        if reconciled.get("passed") is True:
            membership_ids = [
                str(theme_id)
                for theme_id in list(reconciled.get("theme_ids") or [])
                if str(theme_id) in prequalified
            ]
        else:
            membership_ids = []
            raw_details = details_by_symbol.get(symbol, [])
            for raw_detail in raw_details if isinstance(raw_details, list) else []:
                if not isinstance(raw_detail, Mapping):
                    continue
                try:
                    membership = ThemeMembership.from_mapping(raw_detail)
                except (TypeError, ValueError):
                    continue
                if (
                    membership.theme_id in prequalified
                    and membership.symbol == symbol
                    and membership.is_active(protocol_as_of)
                ):
                    membership_ids.append(membership.theme_id)
            membership_ids = _dedupe_flags(membership_ids)
        mandates = {
            str(_mapping_or_empty(states.get(theme_id)).get("mandate") or "")
            for theme_id in membership_ids
            if theme_id in states
        }
        mandates.discard("")
        if not mandates:
            unclassified_symbols.append(symbol)
            non_tech_symbols.append(symbol)
            classification_blockers[symbol] = [
                "prospective_pit_prequalified_membership_missing"
            ]
            continue
        tech_mandates = {"technology", "advanced_manufacturing"}
        if mandates.intersection(tech_mandates) and not mandates.issubset(tech_mandates):
            non_tech_symbols.append(symbol)
            classification_blockers[symbol] = ["mixed_theme_mandate_fail_closed"]
        elif mandates.isdisjoint(tech_mandates):
            non_tech_symbols.append(symbol)
    cap_enabled = cap.get("enabled") is True
    cap_contract_present = {
        "non_tech_nav_cap",
        "non_tech_max_positions",
    }.issubset(cap)
    status = (
        "active"
        if prospective_active and cap_enabled
        else "observer_only"
    )
    if not protocol:
        status = "protocol_missing"
    elif protocol.get("formal_kill_switch") is True:
        status = "formal_kill_switch_active"
    elif not prospective_active:
        status = "formal_not_active"
    elif not cap_enabled:
        status = "closed_by_markov"
    return {
        # ``enabled`` means the cap contract must be enforced.  In a downtrend
        # the channel is closed but enforcement remains enabled at 0/0.
        "enabled": bool(prospective_active and cap_contract_present),
        "channel_open": bool(prospective_active and cap_enabled),
        "status": status,
        "regime": str(cap.get("regime") or ""),
        "non_tech_symbols": sorted(non_tech_symbols),
        "nav_cap": _clamp(
            _safe_float(cap.get("non_tech_nav_cap"), 0.0),
            0.0,
            1.0,
        ),
        "max_positions": max(int(_safe_float(cap.get("non_tech_max_positions"), 0.0)), 0),
        "protocol_hash": protocol_hash,
        "formal_kill_switch": bool(protocol.get("formal_kill_switch", True)),
        "unclassified_symbols": sorted(unclassified_symbols),
        "classification_blockers": classification_blockers,
        "source": (
            "post_control_reconciliation"
            if reconciled_by_symbol
            else "prospective_prequalified_pit_memberships"
        ),
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


def _resolve_concept_membership_scan_context(
    *,
    concept_membership_enabled: bool | None,
    concept_membership_path: str | Path | None,
    concept_membership_required: bool | None,
    concept_primary_margin: float | None,
    theme_memberships: list[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    enabled = False
    required = False
    path = "data/theme_membership.jsonl"
    primary_margin = 0.05
    try:
        from quant_investor.config import Config

        enabled = bool(getattr(Config, "THEME_CONCEPT_MEMBERSHIP_ENABLED", False))
        required = bool(getattr(Config, "THEME_CONCEPT_MEMBERSHIP_REQUIRED", False))
        path = str(
            getattr(Config, "THEME_CONCEPT_MEMBERSHIP_PATH", path)
            or "data/theme_membership.jsonl"
        )
        primary_margin = _clamp(
            _safe_float(getattr(Config, "THEME_CONCEPT_PRIMARY_MARGIN", 0.05), 0.05),
            0.0,
            1.0,
        )
    except Exception:
        pass

    if concept_membership_enabled is not None:
        enabled = bool(concept_membership_enabled)
    if concept_membership_required is not None:
        required = bool(concept_membership_required)
    if concept_membership_path is not None:
        path = str(concept_membership_path)
    if concept_primary_margin is not None:
        primary_margin = _clamp(_safe_float(concept_primary_margin, primary_margin), 0.0, 1.0)

    if not enabled:
        return {
            "enabled": False,
            "required": required,
            "path": path,
            "primary_margin": primary_margin,
            "memberships": [],
            "membership_count": 0,
            "status": "disabled",
            "diagnostic_notes": [],
        }

    if theme_memberships is not None:
        memberships = list(theme_memberships or [])
        return {
            "enabled": True,
            "required": required,
            "path": "inline",
            "primary_margin": primary_margin,
            "memberships": memberships,
            "membership_count": len(memberships),
            "status": "success" if memberships else "empty",
            "diagnostic_notes": (
                [f"theme_membership_count={len(memberships)}"]
                if memberships
                else ["theme_membership_inline_empty"]
            ),
        }

    result = ThemeMembershipStore(path).load()
    return {
        "enabled": True,
        "required": required,
        "path": path,
        "primary_margin": primary_margin,
        "memberships": list(result.memberships),
        "membership_count": len(result.memberships),
        "status": str(result.status or "missing"),
        "diagnostic_notes": list(result.diagnostic_notes or []),
    }


def _concept_membership_metadata(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "concept_membership_enabled": bool(context.get("enabled", False)),
        "concept_membership_required": bool(context.get("required", False)),
        "concept_membership_path": str(context.get("path") or ""),
        "concept_membership_status": str(context.get("status") or "disabled"),
        "concept_membership_count": int(
            _safe_float(context.get("membership_count", 0), 0.0)
        ),
        "concept_primary_margin": _clamp(
            _safe_float(context.get("primary_margin", 0.05), 0.05),
            0.0,
            1.0,
        ),
    }


def _resolve_membership_v2_scan_context(
    *,
    enabled: bool | None,
    path: str | Path | None,
    required: bool | None,
    expected_sha256: str | None,
    as_of: str,
) -> dict[str, Any]:
    resolved_enabled = True
    resolved_required = False
    resolved_path = "private/theme_knowledge/theme_membership.v2.jsonl"
    resolved_expected_sha = ""
    try:
        from quant_investor.config import Config

        resolved_enabled = bool(
            getattr(Config, "THEME_MEMBERSHIP_V2_ENABLED", True)
        )
        resolved_required = bool(
            getattr(Config, "THEME_MEMBERSHIP_V2_REQUIRED", False)
        )
        resolved_path = str(
            getattr(Config, "THEME_MEMBERSHIP_V2_PATH", resolved_path)
            or resolved_path
        )
        resolved_expected_sha = str(
            getattr(Config, "THEME_MEMBERSHIP_V2_EXPECTED_SHA256", "")
            or ""
        ).strip().lower()
    except Exception:
        pass
    if enabled is not None:
        resolved_enabled = bool(enabled)
    if required is not None:
        resolved_required = bool(required)
    if path is not None:
        resolved_path = str(path)
    if expected_sha256 is not None:
        resolved_expected_sha = str(expected_sha256 or "").strip().lower()

    base = {
        "enabled": resolved_enabled,
        "required": resolved_required,
        "path": resolved_path,
        "expected_sha256": resolved_expected_sha,
        "artifact_sha256": "",
        "hash_verified": False,
        "permissions_0600": False,
        "memberships": [],
        "membership_count": 0,
        "active_membership_count": 0,
        "updated_at_status": "disabled" if not resolved_enabled else "unverified",
        "updated_at_invalid_count": 0,
        "pit_status": "disabled" if not resolved_enabled else "coverage_blocked",
        "status": "disabled" if not resolved_enabled else "missing",
        "diagnostic_notes": [],
    }
    if not resolved_enabled:
        return base

    source = Path(resolved_path)
    if not source.is_file():
        base["diagnostic_notes"] = ["theme_membership_v2_file_missing"]
        return base
    try:
        raw = source.read_bytes()
    except OSError as exc:
        base["status"] = "error"
        base["diagnostic_notes"] = [
            f"theme_membership_v2_file_read_error: {exc}"
        ]
        return base
    actual_sha = hashlib.sha256(raw).hexdigest()
    base["artifact_sha256"] = actual_sha
    base["permissions_0600"] = source.stat().st_mode & 0o777 == 0o600
    if not base["permissions_0600"]:
        base["status"] = "error"
        base["diagnostic_notes"] = [
            "theme_membership_v2_permissions_not_0600"
        ]
        return base
    if resolved_expected_sha and (
        len(resolved_expected_sha) != 64
        or any(
            character not in "0123456789abcdef"
            for character in resolved_expected_sha
        )
    ):
        base["status"] = "error"
        base["diagnostic_notes"] = [
            "theme_membership_v2_expected_sha256_invalid"
        ]
        return base
    if resolved_expected_sha and resolved_expected_sha != actual_sha:
        base["status"] = "error"
        base["diagnostic_notes"] = [
            "theme_membership_v2_artifact_sha256_mismatch"
        ]
        return base
    base["hash_verified"] = bool(
        resolved_expected_sha and resolved_expected_sha == actual_sha
    )

    memberships: list[ThemeMembership] = []
    try:
        for line_number, raw_line in enumerate(
            raw.decode("utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"line {line_number} is not an object")
            if str(payload.get("schema_version") or "") != (
                "theme_membership.v2"
            ):
                raise ValueError(
                    f"line {line_number} is not theme_membership.v2"
                )
            memberships.append(ThemeMembership.from_mapping(payload))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        base["status"] = "error"
        base["diagnostic_notes"] = [
            f"theme_membership_v2_file_format_error: {exc}"
        ]
        return base
    active_count = sum(
        len(items)
        for items in active_memberships_by_symbol(
            memberships,
            as_of=as_of,
        ).values()
    )
    invalid_updated_at_count = sum(
        1
        for membership in memberships
        if not _valid_membership_updated_at(membership.updated_at)
    )
    base.update(
        {
            "memberships": [membership.to_dict() for membership in memberships],
            "membership_count": len(memberships),
            "active_membership_count": active_count,
            "updated_at_status": (
                "success" if invalid_updated_at_count == 0 else "unverified"
            ),
            "updated_at_invalid_count": invalid_updated_at_count,
            "pit_status": "success" if active_count else "coverage_blocked",
            "status": "success" if memberships else "empty",
            "diagnostic_notes": [
                f"theme_membership_v2_count={len(memberships)}",
                f"theme_membership_v2_active_count={active_count}",
                (
                    "theme_membership_v2_updated_at_verified"
                    if invalid_updated_at_count == 0
                    else (
                        "theme_membership_v2_updated_at_missing_or_invalid="
                        f"{invalid_updated_at_count}"
                    )
                ),
                (
                    "theme_membership_v2_hash_verified"
                    if base["hash_verified"]
                    else "theme_membership_v2_hash_unpinned_observer_only"
                ),
            ],
        }
    )
    return base


def _membership_v2_metadata(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "membership_v2_enabled": bool(context.get("enabled", False)),
        "membership_v2_required": bool(context.get("required", False)),
        "membership_v2_path": str(context.get("path") or ""),
        "membership_v2_status": str(context.get("status") or "disabled"),
        "membership_v2_count": int(
            _safe_float(context.get("membership_count", 0), 0.0)
        ),
        "membership_v2_active_count": int(
            _safe_float(context.get("active_membership_count", 0), 0.0)
        ),
        "membership_v2_pit_status": str(
            context.get("pit_status") or "coverage_blocked"
        ),
        "membership_v2_updated_at_status": str(
            context.get("updated_at_status") or "unverified"
        ),
        "membership_v2_updated_at_invalid_count": int(
            _safe_float(context.get("updated_at_invalid_count", 0), 0.0)
        ),
        "membership_v2_artifact_sha256": str(
            context.get("artifact_sha256") or ""
        ),
        "membership_v2_expected_sha256": str(
            context.get("expected_sha256") or ""
        ),
        "membership_v2_hash_verified": bool(
            context.get("hash_verified", False)
        ),
        "membership_v2_permissions_0600": bool(
            context.get("permissions_0600", False)
        ),
        "membership_v2_diagnostic_notes": list(
            context.get("diagnostic_notes") or []
        ),
    }


def _membership_v2_formal_activation_blockers(
    context: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if context.get("enabled") is not True:
        blockers.append("theme_membership_v2_not_enabled")
    if context.get("required") is not True:
        blockers.append("theme_membership_v2_not_required")
    if str(context.get("status") or "") != "success":
        blockers.append("theme_membership_v2_not_success")
    if context.get("hash_verified") is not True:
        blockers.append("theme_membership_v2_hash_unverified")
    if str(context.get("pit_status") or "") != "success":
        blockers.append("theme_membership_v2_pit_unverified")
    if str(context.get("updated_at_status") or "") != "success":
        blockers.append("theme_membership_v2_updated_at_unverified")
    return blockers


def _valid_membership_updated_at(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _build_protocol_v2_metadata(
    *,
    enabled: bool | None,
    theme_scores: Mapping[str, Mapping[str, Any]],
    symbol_theme_memberships: Mapping[str, list[str]],
    symbol_theme_membership_details: Mapping[str, list[Mapping[str, Any]]],
    membership_v2_context: Mapping[str, Any],
    as_of: str,
    taxonomy_path: str | Path | None,
    evidence_path: str | Path | None,
    pevc_path: str | Path | None,
    previous_states: Mapping[str, Mapping[str, Any]] | None,
    downstream_gates: Mapping[str, Mapping[str, Any]] | None,
    markov_regime: str,
    formal_enabled: bool | None,
    formal_kill_switch: bool | None,
    valid_trading_dates: list[str],
) -> dict[str, Any]:
    resolved_enabled = True
    resolved_taxonomy_path = str(
        taxonomy_path or "quant_investor/themes/data/theme_taxonomy.v2.json"
    )
    resolved_evidence_path = str(
        evidence_path or "private/theme_knowledge/theme_evidence_events.jsonl"
    )
    resolved_pevc_path = str(
        pevc_path or "private/theme_knowledge/pevc_theses.jsonl"
    )
    resolved_formal_enabled = False
    resolved_kill_switch = True
    try:
        from quant_investor.config import Config

        resolved_enabled = bool(getattr(Config, "THEME_PROTOCOL_V2_ENABLED", True))
        if taxonomy_path is None:
            resolved_taxonomy_path = str(
                getattr(Config, "THEME_TAXONOMY_V2_PATH", resolved_taxonomy_path)
                or resolved_taxonomy_path
            )
        if evidence_path is None:
            resolved_evidence_path = str(
                getattr(Config, "THEME_EVIDENCE_EVENT_V1_PATH", resolved_evidence_path)
                or resolved_evidence_path
            )
        if pevc_path is None:
            resolved_pevc_path = str(
                getattr(Config, "THEME_PEVC_CANONICAL_PATH", resolved_pevc_path)
                or resolved_pevc_path
            )
        resolved_formal_enabled = bool(
            getattr(Config, "THEME_V2_FORMAL_ENABLED", False)
        )
        resolved_kill_switch = bool(
            getattr(Config, "THEME_V2_FORMAL_KILL_SWITCH", True)
        )
    except Exception:
        pass
    if enabled is not None:
        resolved_enabled = bool(enabled)
    if formal_enabled is not None:
        resolved_formal_enabled = bool(formal_enabled)
    if formal_kill_switch is not None:
        resolved_kill_switch = bool(formal_kill_switch)
    if not resolved_enabled:
        return {
            "schema_version": "theme_protocol.v2",
            "status": "disabled",
            "observer_enabled": False,
            "formal_enabled": False,
            "formal_kill_switch": True,
            "rollback_status": "observer_disabled",
            "rollback_reason": "protocol_v2_disabled",
            "formal_pool": [],
            "prequalified_pool": [],
            "forced_theme_count": 0,
            "diagnostic_notes": ["theme_protocol_v2_disabled"],
        }

    diagnostic_notes: list[str] = []
    try:
        taxonomy = ThemeTaxonomy.load(resolved_taxonomy_path)
        events = _load_protocol_events(resolved_evidence_path)
        if not Path(resolved_evidence_path).exists():
            diagnostic_notes.append("theme_evidence_event_file_missing_observer_continues")
        pevc_store = PeVcKnowledgeStore(resolved_pevc_path)
        theses = [thesis.to_dict() for thesis in pevc_store.load(as_of=as_of)]
        if not Path(resolved_pevc_path).exists():
            diagnostic_notes.append("pevc_canonical_file_missing_observer_continues")
        membership_ids = sorted(
            {
                str(theme_id)
                for memberships in symbol_theme_memberships.values()
                for theme_id in list(memberships or [])
                if str(theme_id)
            }
        )
        membership_details = [
            dict(detail)
            for details in symbol_theme_membership_details.values()
            for detail in list(details or [])
            if isinstance(detail, Mapping)
        ]
        payload = evaluate_theme_protocol_v2(
            theme_scores=theme_scores,
            taxonomy=taxonomy,
            as_of=as_of,
            evidence_events=events,
            pevc_theses=theses,
            valid_membership_theme_ids=membership_ids,
            theme_membership_details=membership_details,
            previous_states=previous_states,
            downstream_gates=downstream_gates,
            markov_regime=markov_regime,
            formal_enabled=resolved_formal_enabled,
            formal_kill_switch=resolved_kill_switch,
            valid_trading_dates=valid_trading_dates,
            formal_activation_blockers=(
                _membership_v2_formal_activation_blockers(
                    membership_v2_context
                )
                if resolved_formal_enabled
                else []
            ),
        )
        payload["diagnostic_notes"] = diagnostic_notes
        payload["taxonomy_path"] = resolved_taxonomy_path
        payload["evidence_path"] = resolved_evidence_path
        payload["pevc_path"] = resolved_pevc_path
        payload["membership_v2"] = _membership_v2_metadata(
            membership_v2_context
        )
        payload.pop("artifact_hash", None)
        payload["artifact_hash"] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return payload
    except Exception as exc:
        return {
            "schema_version": "theme_protocol.v2",
            "status": "error",
            "observer_enabled": False,
            "formal_enabled": False,
            "formal_kill_switch": True,
            "rollback_status": "observer_only",
            "rollback_reason": "protocol_v2_error_fail_closed",
            "formal_pool": [],
            "prequalified_pool": [],
            "forced_theme_count": 0,
            "diagnostic_notes": [f"theme_protocol_v2_error: {exc}"],
        }


def _protocol_trading_dates(
    frames: Mapping[str, pd.DataFrame],
    *,
    as_of: str,
) -> list[str]:
    point = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(point):
        return []
    dates: set[str] = set()
    for frame in frames.values():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        date_column = next(
            (
                column
                for column in ("trade_date", "date", "Date", "datetime", "time")
                if column in frame.columns
            ),
            None,
        )
        if date_column is None:
            continue
        parsed = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        for value in parsed:
            candidate = value.date()
            if candidate <= point.date() and candidate.weekday() < 5:
                dates.add(candidate.isoformat())
    return sorted(dates)


def _resolve_protocol_previous_states(
    *,
    snapshots: Any,
    snapshot_dir: str | Path | None,
    history_limit: int,
    market: str,
    universe_key: str,
    as_of: str,
) -> dict[str, Mapping[str, Any]]:
    history = list(snapshots or [])
    resolved_dir = snapshot_dir
    if resolved_dir is None:
        try:
            from quant_investor.config import Config

            resolved_dir = str(
                getattr(Config, "THEME_SNAPSHOT_DIR", "results/theme_snapshots")
                or "results/theme_snapshots"
            )
        except Exception:
            resolved_dir = "results/theme_snapshots"
    if not history and resolved_dir:
        try:
            history = ThemeSnapshotStore(resolved_dir).load_recent(
                market=market or "CN",
                universe_key=universe_key or None,
                limit=max(int(history_limit or 10), 1),
            )
        except Exception:
            history = []
    return _latest_protocol_v2_states(history, as_of=as_of)


def _latest_protocol_v2_states(
    snapshots: Any,
    *,
    as_of: str,
) -> dict[str, Mapping[str, Any]]:
    as_of_key = "".join(character for character in str(as_of) if character.isdigit())[:8]
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for snapshot in list(snapshots or []):
        if not isinstance(snapshot, Mapping):
            continue
        rotation = snapshot.get("theme_rotation")
        payload = rotation if isinstance(rotation, Mapping) else snapshot
        protocol = payload.get("protocol_v2") if isinstance(payload, Mapping) else None
        if not isinstance(protocol, Mapping):
            continue
        protocol_as_of = "".join(
            character
            for character in str(protocol.get("as_of") or payload.get("as_of") or "")
            if character.isdigit()
        )[:8]
        states = protocol.get("states")
        if (
            protocol_as_of
            and (not as_of_key or protocol_as_of <= as_of_key)
            and isinstance(states, Mapping)
        ):
            candidates.append((protocol_as_of, states))
    if not candidates:
        return {}
    _, latest = max(candidates, key=lambda item: item[0])
    return {
        str(theme_id): dict(state)
        for theme_id, state in latest.items()
        if isinstance(state, Mapping)
    }


def _load_protocol_events(path: str | Path) -> list[ThemeEvidenceEvent]:
    source = Path(path)
    if not source.exists():
        return []
    events: list[ThemeEvidenceEvent] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"evidence line {line_number} must be an object")
        events.append(ThemeEvidenceEvent.from_mapping(payload))
    return events


def _compact_top_theme(theme: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "theme_id": str(theme.get("theme_id", "")),
        "theme_name": str(theme.get("theme_name", "")),
        "theme_type": str(theme.get("theme_type", "industry") or "industry"),
        "membership_source": str(
            theme.get("membership_source", "industry_map") or "industry_map"
        ),
        "pit_membership": bool(theme.get("pit_membership", False)),
        "score": _safe_float(theme.get("score", 0.0)),
        "effective_score": _safe_float(
            theme.get("effective_score", theme.get("score", 0.0))
        ),
        "attention": _safe_float(theme.get("attention", 0.0)),
        "attention_5d": _optional_float(theme.get("attention_5d")),
        "attention_20d": _optional_float(theme.get("attention_20d")),
        "attention_60d": _optional_float(theme.get("attention_60d")),
        "attention_120d": _optional_float(theme.get("attention_120d")),
        "attention_turnover_share": _optional_float(
            theme.get("attention_turnover_share")
        ),
        "new_high_rate": _optional_float(theme.get("new_high_rate")),
        "leader_persistence": _optional_float(theme.get("leader_persistence")),
        "attention_history_coverage": _safe_float(
            theme.get("attention_history_coverage", 0.0)
        ),
        "market_confirmation": _safe_float(
            theme.get("market_confirmation", 0.0)
        ),
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
