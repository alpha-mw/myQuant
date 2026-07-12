from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.themes.membership import SCHEMA_VERSION as MEMBERSHIP_SCHEMA_VERSION
from quant_investor.themes.membership import ThemeMembership
from quant_investor.themes.taxonomy import SCHEMA_VERSION as TAXONOMY_SCHEMA_VERSION
from quant_investor.themes.taxonomy import ThemeTaxonomy


EVIDENCE_SCHEMA_VERSION = "theme_evidence_event.v1"
STATE_SCHEMA_VERSION = "theme_state.v2"
PROTOCOL_VERSION = "theme_protocol.v2"
RECONCILIATION_SCHEMA_VERSION = "theme_formal_reconciliation.v1"
_DATE_FORMATS = ("%Y-%m-%d", "%Y%m%d")


class ThemeLifecycle(str, Enum):
    DISCOVERY = "discovery"
    WARMING = "warming"
    VALIDATED_TREND = "validated_trend"
    CROWDED = "crowded"
    COOLING = "cooling"
    BROKEN = "broken"


_LIFECYCLE_ORDER = {
    ThemeLifecycle.DISCOVERY.value: 0,
    ThemeLifecycle.WARMING.value: 1,
    ThemeLifecycle.VALIDATED_TREND.value: 2,
    ThemeLifecycle.CROWDED.value: 3,
    ThemeLifecycle.COOLING.value: 1,
    ThemeLifecycle.BROKEN.value: -1,
}
_DOWNSTREAM_BLOCKERS = {
    "data_gate_blocked",
    "tradability_gate_blocked",
    "liquidity_gate_blocked",
    "positive_edge_buy_gate_blocked",
    "risk_guard_blocked",
    "portfolio_constructor_blocked",
}
_DOWNSTREAM_GATE_FIELDS = {
    "data_pass": "data_gate_blocked",
    "tradability_pass": "tradability_gate_blocked",
    "liquidity_pass": "liquidity_gate_blocked",
    "positive_edge_or_buy": "positive_edge_buy_gate_blocked",
    "risk_guard_pass": "risk_guard_blocked",
    "portfolio_constructor_pass": "portfolio_constructor_blocked",
}
_INDUSTRIAL_EVENT_TYPES = {
    "order",
    "capacity",
    "certification",
    "product",
    "customer_validation",
}


@dataclass(frozen=True)
class ThemeEvidenceEvent:
    event_id: str
    theme_id: str
    event_type: str
    event_date: str
    available_at: str
    direction: int = 1
    strength: float = 0.0
    confidence: float = 0.0
    source_ref: str = ""
    summary: str = ""
    expires_at: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ThemeEvidenceEvent":
        schema = str(payload.get("schema_version") or EVIDENCE_SCHEMA_VERSION).strip()
        if schema != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema_version={schema}")
        event_id = str(payload.get("event_id") or "").strip()
        theme_id = str(payload.get("theme_id") or "").strip()
        event_type = str(payload.get("event_type") or "").strip().lower()
        event_date = _date_text(payload.get("event_date"))
        available_at = _date_text(payload.get("available_at"))
        if not event_id or not theme_id or not event_type:
            raise ValueError("event_id, theme_id, and event_type are required")
        if event_type not in {
            "order",
            "capacity",
            "certification",
            "policy",
            "attention",
            "product",
            "customer_validation",
            "kill",
        }:
            raise ValueError(f"unsupported evidence event_type={event_type}")
        if not event_date or not available_at:
            raise ValueError("event_date and available_at are required")
        direction = -1 if int(_finite(payload.get("direction"), 1.0)) < 0 else 1
        return cls(
            event_id=event_id,
            theme_id=theme_id,
            event_type=event_type,
            event_date=event_date,
            available_at=available_at,
            direction=direction,
            strength=_clamp(payload.get("strength")),
            confidence=_clamp(payload.get("confidence")),
            source_ref=str(payload.get("source_ref") or "").strip(),
            summary=str(payload.get("summary") or "").strip(),
            expires_at=_date_text(payload.get("expires_at"), required=False),
        )

    def is_available(self, as_of: str | date | datetime) -> bool:
        point = _parse_date(as_of)
        available = _parse_date(self.available_at)
        event_date = _parse_date(self.event_date)
        expiry = _parse_date(self.expires_at)
        return bool(
            point is not None
            and available is not None
            and event_date is not None
            and available <= point
            and event_date <= point
            and (expiry is None or expiry >= point)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "event_id": self.event_id,
            "theme_id": self.theme_id,
            "event_type": self.event_type,
            "event_date": self.event_date,
            "available_at": self.available_at,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "source_ref": self.source_ref,
            "summary": self.summary,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class ThemeProtocolConfig:
    attention_min: float = 0.55
    industrial_validation_min: float = 0.45
    market_confirmation_min: float = 0.45
    evidence_confidence_min: float = 0.35
    crowding_block: float = 0.85
    valuation_risk_block: float = 0.80
    evidence_stale_days: int = 120
    pevc_max_percentile_adjustment: float = 0.10
    long_horizon_attention_min_coverage: float = 0.95
    upward_confirmation_days: int = 3
    cooling_confirmation_days: int = 2

    def __post_init__(self) -> None:
        unit_interval_fields = (
            "attention_min",
            "industrial_validation_min",
            "market_confirmation_min",
            "evidence_confidence_min",
            "crowding_block",
            "valuation_risk_block",
            "pevc_max_percentile_adjustment",
            "long_horizon_attention_min_coverage",
        )
        for field_name in unit_interval_fields:
            raw_value = getattr(self, field_name)
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, float)
            ):
                raise ValueError(f"{field_name} must be a finite 0..1 number")
            numeric = float(raw_value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{field_name} must be within 0..1")
        if float(self.pevc_max_percentile_adjustment) > 0.10:
            raise ValueError(
                "pevc_max_percentile_adjustment must not exceed 0.10"
            )
        for field_name in (
            "evidence_stale_days",
            "upward_confirmation_days",
            "cooling_confirmation_days",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_min": self.attention_min,
            "industrial_validation_min": self.industrial_validation_min,
            "market_confirmation_min": self.market_confirmation_min,
            "evidence_confidence_min": self.evidence_confidence_min,
            "crowding_block": self.crowding_block,
            "valuation_risk_block": self.valuation_risk_block,
            "evidence_stale_days": self.evidence_stale_days,
            "pevc_max_percentile_adjustment": self.pevc_max_percentile_adjustment,
            "long_horizon_attention_min_coverage": (
                self.long_horizon_attention_min_coverage
            ),
            "upward_confirmation_days": self.upward_confirmation_days,
            "cooling_confirmation_days": self.cooling_confirmation_days,
        }


@dataclass(frozen=True)
class ThemeStateV2:
    theme_id: str
    theme_name: str
    as_of: str
    attention: float
    industrial_validation: float
    market_confirmation: float
    crowding: float | None
    valuation_risk: float | None
    evidence_confidence: float
    attention_5d: float | None
    attention_20d: float | None
    attention_60d: float | None
    attention_120d: float | None
    attention_turnover_share: float | None
    new_high_rate: float | None
    leader_persistence: float | None
    attention_history_coverage: float
    lifecycle: str
    mandate: str
    tradable_node: bool
    lane: str
    base_rank_score: float
    member_count: int = 0
    nav_weight: float | None = None
    base_percentile_rank: float = 0.0
    pevc_prior: float = 0.0
    pevc_rank_adjustment: float = 0.0
    adjusted_percentile_rank: float = 0.0
    eligibility_blockers: tuple[str, ...] = ()
    prequalification_blockers: tuple[str, ...] = ()
    downstream_blockers: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    pending_transition: str = ""
    pending_confirmation_dates: tuple[str, ...] = ()
    latest_evidence_available_at: str = ""
    latest_industrial_evidence_available_at: str = ""
    thesis_id: str = ""
    thesis_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "as_of": self.as_of,
            "attention": self.attention,
            "industrial_validation": self.industrial_validation,
            "market_confirmation": self.market_confirmation,
            "crowding": self.crowding,
            "valuation_risk": self.valuation_risk,
            "evidence_confidence": self.evidence_confidence,
            "attention_5d": self.attention_5d,
            "attention_20d": self.attention_20d,
            "attention_60d": self.attention_60d,
            "attention_120d": self.attention_120d,
            "attention_turnover_share": self.attention_turnover_share,
            "new_high_rate": self.new_high_rate,
            "leader_persistence": self.leader_persistence,
            "attention_history_coverage": self.attention_history_coverage,
            "lifecycle": self.lifecycle,
            "mandate": self.mandate,
            "tradable_node": self.tradable_node,
            "lane": self.lane,
            "base_rank_score": self.base_rank_score,
            "member_count": self.member_count,
            "nav_weight": self.nav_weight,
            "base_percentile_rank": self.base_percentile_rank,
            "pevc_prior": self.pevc_prior,
            "pevc_rank_adjustment": self.pevc_rank_adjustment,
            "adjusted_percentile_rank": self.adjusted_percentile_rank,
            "eligibility_blockers": list(self.eligibility_blockers),
            "prequalification_blockers": list(self.prequalification_blockers),
            "downstream_blockers": list(self.downstream_blockers),
            "risk_flags": list(self.risk_flags),
            "pending_transition": self.pending_transition,
            "pending_confirmation_dates": list(self.pending_confirmation_dates),
            "latest_evidence_available_at": self.latest_evidence_available_at,
            "latest_industrial_evidence_available_at": (
                self.latest_industrial_evidence_available_at
            ),
            "thesis_id": self.thesis_id,
            "thesis_version": self.thesis_version,
        }


def evaluate_theme_protocol_v2(
    *,
    theme_scores: Mapping[str, Mapping[str, Any]],
    taxonomy: ThemeTaxonomy,
    as_of: str,
    evidence_events: Sequence[ThemeEvidenceEvent | Mapping[str, Any]] = (),
    pevc_theses: Sequence[Mapping[str, Any]] = (),
    valid_membership_theme_ids: Sequence[str] = (),
    theme_membership_details: Sequence[Mapping[str, Any]] = (),
    previous_states: Mapping[str, Mapping[str, Any]] | None = None,
    downstream_gates: Mapping[str, Mapping[str, Any]] | None = None,
    markov_regime: str = "",
    formal_enabled: bool = False,
    formal_kill_switch: bool = False,
    valid_trading_dates: Sequence[str] = (),
    formal_activation_blockers: Sequence[str] = (),
    config: ThemeProtocolConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ThemeProtocolConfig()
    as_of_date = _parse_date(as_of)
    if as_of_date is None:
        raise ValueError("as_of must be a valid date")
    as_of_text = as_of_date.isoformat()
    available_events = _available_events(evidence_events, as_of=as_of_text)
    events_by_theme: dict[str, list[ThemeEvidenceEvent]] = {}
    for event in available_events:
        events_by_theme.setdefault(event.theme_id, []).append(event)
    theses_by_theme = _approved_theses_by_theme(pevc_theses, as_of=as_of_text)
    legacy_membership_ids = sorted(
        {str(item) for item in valid_membership_theme_ids if str(item)}
    )
    membership_context = _active_v2_membership_context(
        theme_membership_details,
        as_of=as_of_text,
    )
    valid_memberships = set(membership_context["active_theme_ids"])
    previous = dict(previous_states or {})
    gates = dict(downstream_gates or {})
    confirmed_trading_dates = _valid_trading_date_set(
        valid_trading_dates,
        as_of=as_of_date,
    )
    activation_blockers = list(
        dict.fromkeys(
            str(blocker).strip()
            for blocker in formal_activation_blockers
            if str(blocker).strip()
        )
    )

    mutable_states: list[dict[str, Any]] = []
    for raw_theme_id, raw_score in sorted(theme_scores.items(), key=lambda item: str(item[0])):
        payload = dict(raw_score or {})
        theme_id = str(payload.get("theme_id") or raw_theme_id)
        theme_name = str(payload.get("theme_name") or theme_id)
        node = taxonomy.resolve(theme_id) or taxonomy.resolve(theme_name)
        theme_events = events_by_theme.get(theme_id, [])
        thesis = theses_by_theme.get(theme_id, {})
        attention = _attention_score(payload, theme_events)
        industrial_validation = _industrial_validation(theme_events)
        market_confirmation = _market_confirmation(payload)
        crowding = _available_risk_axis(
            payload,
            value_key="crowding_risk",
            status_key="crowding_status",
        )
        valuation_risk = _available_risk_axis(
            payload,
            value_key="valuation_risk",
            status_key="valuation_risk_status",
        )
        evidence_confidence = _evidence_confidence(theme_events, payload)
        membership_invalidated = theme_id in set(
            membership_context["invalidated_theme_ids"]
        )
        hard_kill = membership_invalidated or any(
            event.event_type == "kill" and event.direction < 0
            for event in theme_events
        )
        desired_lifecycle = _desired_lifecycle(
            attention=attention,
            industrial_validation=industrial_validation,
            market_confirmation=market_confirmation,
            crowding=crowding or 0.0,
            hard_kill=hard_kill,
        )
        lifecycle = transition_theme_lifecycle(
            previous.get(theme_id, {}),
            desired=desired_lifecycle,
            as_of=as_of_text,
            hard_kill=hard_kill,
            confirmation_date=_confirmation_trade_date(
                payload,
                valid_trading_dates=confirmed_trading_dates,
                as_of=as_of_date,
            ),
            valid_trading_dates=confirmed_trading_dates,
            config=cfg,
        )
        mandate = node.mandate if node is not None else "tactical"
        tradable_node = bool(node.tradable_node) if node is not None else True
        latest_available = max((event.available_at for event in theme_events), default="")
        latest_industrial_available = max(
            (
                event.available_at
                for event in theme_events
                if event.event_type in _INDUSTRIAL_EVENT_TYPES
                and event.direction > 0
            ),
            default="",
        )
        blockers = _eligibility_blockers(
            theme_id=theme_id,
            node_present=node is not None,
            tradable_node=tradable_node,
            valid_membership=theme_id in valid_memberships,
            attention_60d=_optional_axis(payload.get("attention_60d")),
            attention_120d=_optional_axis(payload.get("attention_120d")),
            attention_history_coverage=_clamp(
                payload.get("attention_history_coverage")
            ),
            attention=attention,
            industrial_validation=industrial_validation,
            market_confirmation=market_confirmation,
            crowding=crowding,
            valuation_risk=valuation_risk,
            evidence_confidence=evidence_confidence,
            latest_industrial_evidence_available_at=(
                latest_industrial_available
            ),
            as_of=as_of_date,
            lifecycle=lifecycle["lifecycle"],
            gates=dict(gates.get(theme_id) or {}),
            config=cfg,
        )
        prequalification_blockers = [
            blocker for blocker in blockers if blocker not in _DOWNSTREAM_BLOCKERS
        ]
        downstream_blockers = [
            blocker for blocker in blockers if blocker in _DOWNSTREAM_BLOCKERS
        ]
        base_score = _clamp(
            0.45 * attention
            + 0.35 * industrial_validation
            + 0.20 * market_confirmation
        )
        pevc_prior = _pevc_prior(thesis)
        lane = "market_observation"
        if mandate in {"technology", "advanced_manufacturing"} or pevc_prior > 0.0:
            lane = "tech_thesis_watch"
        mutable_states.append(
            {
                "theme_id": theme_id,
                "theme_name": theme_name,
                "attention": attention,
                "industrial_validation": industrial_validation,
                "market_confirmation": market_confirmation,
                "crowding": crowding,
                "valuation_risk": valuation_risk,
                "evidence_confidence": evidence_confidence,
                "attention_5d": _optional_axis(payload.get("attention_5d")),
                "attention_20d": _optional_axis(payload.get("attention_20d")),
                "attention_60d": _optional_axis(payload.get("attention_60d")),
                "attention_120d": _optional_axis(payload.get("attention_120d")),
                "attention_turnover_share": _optional_axis(
                    payload.get("attention_turnover_share")
                ),
                "new_high_rate": _optional_axis(payload.get("new_high_rate")),
                "leader_persistence": _optional_axis(
                    payload.get("leader_persistence")
                ),
                "attention_history_coverage": _clamp(
                    payload.get("attention_history_coverage")
                ),
                "lifecycle": lifecycle["lifecycle"],
                "mandate": mandate,
                "tradable_node": tradable_node,
                "lane": lane,
                "base_rank_score": base_score,
                "member_count": max(int(_finite(payload.get("member_count"), 0.0)), 0),
                "nav_weight": _optional_unit_value(payload.get("nav_weight")),
                "base_percentile_rank": 0.0,
                "pevc_prior": pevc_prior,
                "pevc_rank_adjustment": 0.0,
                "adjusted_percentile_rank": 0.0,
                "eligibility_blockers": tuple(blockers),
                "prequalification_blockers": tuple(prequalification_blockers),
                "downstream_blockers": tuple(downstream_blockers),
                "risk_flags": tuple(
                    _risk_flags(
                        crowding,
                        valuation_risk,
                        latest_industrial_available,
                        as_of_date,
                        cfg,
                    )
                ),
                "pending_transition": lifecycle["pending_transition"],
                "pending_confirmation_dates": tuple(lifecycle["pending_confirmation_dates"]),
                "latest_evidence_available_at": latest_available,
                "latest_industrial_evidence_available_at": (
                    latest_industrial_available
                ),
                "thesis_id": str(thesis.get("thesis_id") or ""),
                "thesis_version": str(thesis.get("version") or ""),
            }
        )

    _apply_percentile_ranks(
        mutable_states,
        max_adjustment=cfg.pevc_max_percentile_adjustment,
    )
    tactical = tactical_lane_cap(markov_regime)
    _apply_tactical_position_cap(mutable_states, tactical=tactical)
    states = [
        ThemeStateV2(as_of=as_of_text, **state)
        for state in sorted(
            mutable_states,
            key=lambda item: (-float(item["adjusted_percentile_rank"]), str(item["theme_id"])),
        )
    ]
    observation = [state.theme_id for state in states]
    evidence_prequalified = [
        state.theme_id
        for state in states
        if not state.prequalification_blockers
    ]
    prequalified = (
        []
        if formal_enabled and activation_blockers
        else evidence_prequalified
    )
    watch = [state.theme_id for state in states if state.lane == "tech_thesis_watch"]
    formal = [state.theme_id for state in states if state.lane == "formal_investable"]
    protocol_hash = build_theme_protocol_hash(taxonomy=taxonomy, config=cfg)
    if not formal_enabled or formal_kill_switch:
        status = "observer"
    elif activation_blockers:
        status = "blocked"
    elif prequalified:
        status = "prequalified"
    else:
        status = "blocked"
    rollback_reason = ""
    if formal_kill_switch:
        rollback_reason = "formal_kill_switch_active"
    elif not formal_enabled:
        rollback_reason = "formal_not_enabled_observer_only"
    result = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "implementation_code_sha256": theme_protocol_code_hash(),
        "taxonomy_schema_version": TAXONOMY_SCHEMA_VERSION,
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_version": taxonomy.version,
        "as_of": as_of_text,
        "status": status,
        "observer_enabled": True,
        "formal_enabled": bool(formal_enabled),
        "formal_kill_switch": bool(formal_kill_switch),
        "formal_activation_blockers": activation_blockers,
        "formal_activation_ready": bool(
            formal_enabled
            and not formal_kill_switch
            and not activation_blockers
        ),
        "rollback_status": "observer_only" if rollback_reason else "not_active",
        "rollback_reason": rollback_reason,
        "forced_theme_count": 0,
        "prequalified_pool": prequalified,
        "pit_membership_status": membership_context["status"],
        "pit_membership_count": membership_context["active_membership_count"],
        "pit_membership_hash": membership_context["membership_hash"],
        "pit_membership_invalidated_theme_ids": membership_context[
            "invalidated_theme_ids"
        ],
        "pit_membership_diagnostic_notes": membership_context["diagnostic_notes"],
        "lifecycle_trading_dates": sorted(confirmed_trading_dates),
        "legacy_membership_theme_ids_ignored": legacy_membership_ids,
        "downstream_gates_supplied": bool(downstream_gates),
        "lanes": {
            "market_observation": observation,
            "tech_thesis_watch": watch,
            "formal_investable": formal,
        },
        "formal_pool": formal,
        "formal_producer": "post_control_reconciliation_only",
        "states": {state.theme_id: state.to_dict() for state in states},
        "tactical_lane_cap": tactical,
        "config": cfg.to_dict(),
        "input_hashes": {
            "theme_scores_sha256": _stable_payload_hash(
                {str(key): dict(value or {}) for key, value in theme_scores.items()}
            ),
            "evidence_events_sha256": _stable_payload_hash(
                {event.event_id: event.to_dict() for event in available_events}
            ),
            "pevc_theses_sha256": _stable_payload_hash(
                {str(key): dict(value or {}) for key, value in theses_by_theme.items()}
            ),
            "previous_states_sha256": _stable_payload_hash(previous),
            "downstream_gates_sha256": _stable_payload_hash(gates),
            "trading_dates_sha256": _stable_payload_hash(
                {"dates": sorted(confirmed_trading_dates)}
            ),
            "formal_activation_blockers_sha256": _stable_payload_hash(
                {"blockers": activation_blockers}
            ),
        },
        "deterministic": True,
        "no_network": True,
        "no_llm": True,
    }
    result["artifact_hash"] = _stable_payload_hash(result)
    return result


def reconcile_theme_protocol_v2(
    *,
    prequalification: Mapping[str, Any],
    symbol_membership_details: Mapping[str, Sequence[Mapping[str, Any]]],
    symbol_outcomes: Mapping[str, Mapping[str, Any]],
    as_of: str,
    expected_protocol_hash: str,
    run_id: str = "",
) -> dict[str, Any]:
    """Produce the only Theme v2 formal pool after the full control chain.

    The input is an immutable prequalification snapshot plus per-symbol outcomes
    written after Bayesian edge/BUY, RiskGuard, and PortfolioConstructor.  No
    missing or non-boolean gate is inferred as passed.
    """

    protocol_hash = str(prequalification.get("protocol_hash") or "")
    if str(prequalification.get("schema_version") or "") != PROTOCOL_VERSION:
        raise ValueError("unsupported theme prequalification schema")
    artifact_hash = str(prequalification.get("artifact_hash") or "")
    unsigned_prequalification = dict(prequalification)
    unsigned_prequalification.pop("artifact_hash", None)
    if not artifact_hash or artifact_hash != _stable_payload_hash(
        unsigned_prequalification
    ):
        raise ValueError("theme prequalification artifact hash mismatch")
    if str(prequalification.get("implementation_code_sha256") or "") != (
        theme_protocol_code_hash()
    ):
        raise ValueError("theme prequalification implementation hash mismatch")
    if list(prequalification.get("formal_pool") or []):
        raise ValueError("prequalification must not contain a formal pool")
    if not expected_protocol_hash or protocol_hash != expected_protocol_hash:
        raise ValueError("theme protocol hash mismatch")
    point = _parse_date(as_of)
    protocol_as_of = _parse_date(prequalification.get("as_of"))
    if point is None or protocol_as_of is None or protocol_as_of != point:
        raise ValueError("reconciliation as_of must equal prequalification as_of")
    if prequalification.get("formal_enabled") is not True:
        raise ValueError("theme v2 formal switch is not enabled")
    if prequalification.get("formal_kill_switch") is True:
        raise ValueError("theme v2 formal kill switch is active")
    if str(prequalification.get("formal_producer") or "") != (
        "post_control_reconciliation_only"
    ):
        raise ValueError("prequalification does not require post-control reconciliation")

    prequalified = {
        str(theme_id)
        for theme_id in list(prequalification.get("prequalified_pool") or [])
        if str(theme_id)
    }
    states = {
        str(theme_id): dict(state or {})
        for theme_id, state in dict(prequalification.get("states") or {}).items()
        if isinstance(state, Mapping)
    }
    active_by_symbol = _active_v2_memberships_by_symbol(
        symbol_membership_details,
        as_of=point.isoformat(),
    )
    membership_hash = _membership_snapshot_hash(active_by_symbol)
    if membership_hash != str(prequalification.get("pit_membership_hash") or ""):
        raise ValueError("PIT membership snapshot hash mismatch")
    normalized_outcomes = {
        str(symbol or "").strip().upper(): dict(outcome or {})
        for symbol, outcome in symbol_outcomes.items()
        if str(symbol or "").strip()
    }
    per_symbol: dict[str, dict[str, Any]] = {}
    passed_symbols: list[dict[str, Any]] = []
    for symbol in sorted(set(active_by_symbol) | set(normalized_outcomes)):
        theme_ids = sorted(
            {
                membership.theme_id
                for membership in active_by_symbol.get(symbol, [])
                if membership.theme_id in prequalified
            }
        )
        outcome = normalized_outcomes.get(symbol, {})
        blockers = [
            blocker
            for key, blocker in _DOWNSTREAM_GATE_FIELDS.items()
            if outcome.get(key) is not True
        ]
        if not theme_ids:
            blockers.append("pit_prequalified_membership_missing")
        passed = not blockers
        record = {
            "symbol": symbol,
            "theme_ids": theme_ids,
            "passed": passed,
            "blockers": blockers,
            "gates": {
                key: outcome.get(key) is True
                for key in _DOWNSTREAM_GATE_FIELDS
            },
            "decision_id": str(outcome.get("decision_id") or ""),
            "portfolio_weight": _optional_unit_value(outcome.get("portfolio_weight")),
        }
        per_symbol[symbol] = record
        if passed:
            passed_symbols.append(record)

    tactical = dict(prequalification.get("tactical_lane_cap") or {})
    tech_passed: list[dict[str, Any]] = []
    non_tech_passed: list[dict[str, Any]] = []
    for record in passed_symbols:
        mandates = {
            str(states.get(theme_id, {}).get("mandate") or "")
            for theme_id in record["theme_ids"]
        }
        tech_mandates = {"technology", "advanced_manufacturing"}
        if mandates and mandates.issubset(tech_mandates):
            tech_passed.append(record)
        else:
            if mandates.intersection(tech_mandates):
                record["classification"] = "mixed_mandate_fail_closed_non_tech"
            elif not mandates:
                record["classification"] = "unknown_mandate_fail_closed_non_tech"
            else:
                record["classification"] = "non_tech"
            non_tech_passed.append(record)
    non_tech_passed.sort(
        key=lambda record: (
            -max(
                (
                    _finite(
                        states.get(theme_id, {}).get("adjusted_percentile_rank"),
                        0.0,
                    )
                    for theme_id in record["theme_ids"]
                ),
                default=0.0,
            ),
            record["symbol"],
        )
    )
    max_non_tech = max(int(tactical.get("non_tech_max_positions") or 0), 0)
    non_tech_with_weight: list[dict[str, Any]] = []
    for record in non_tech_passed:
        if record.get("portfolio_weight") is None:
            record["passed"] = False
            record["blockers"].append("tactical_portfolio_weight_missing")
            continue
        non_tech_with_weight.append(record)
    kept_non_tech = non_tech_with_weight[:max_non_tech]
    for record in non_tech_with_weight[max_non_tech:]:
        blocker = (
            "tactical_lane_closed_by_markov"
            if max_non_tech <= 0
            else "tactical_position_cap_exceeded"
        )
        record["passed"] = False
        record["blockers"].append(blocker)

    non_tech_nav_used = sum(
        float(record.get("portfolio_weight") or 0.0)
        for record in kept_non_tech
    )
    non_tech_nav_cap = _clamp(tactical.get("non_tech_nav_cap"))
    if non_tech_nav_used > non_tech_nav_cap + 1e-12:
        for record in kept_non_tech:
            record["passed"] = False
            record["blockers"].append("tactical_nav_cap_exceeded")
        kept_non_tech = []

    formal_records = tech_passed + kept_non_tech
    formal_symbols = sorted(record["symbol"] for record in formal_records)
    formal_theme_ids = {
        theme_id
        for record in formal_records
        for theme_id in record["theme_ids"]
    }
    formal_pool = sorted(
        formal_theme_ids,
        key=lambda theme_id: (
            -_finite(states.get(theme_id, {}).get("adjusted_percentile_rank"), 0.0),
            theme_id,
        ),
    )
    per_theme = {
        theme_id: {
            "theme_id": theme_id,
            "formal_symbols": sorted(
                record["symbol"]
                for record in formal_records
                if theme_id in record["theme_ids"]
            ),
            "adjusted_percentile_rank": _finite(
                states.get(theme_id, {}).get("adjusted_percentile_rank"),
                0.0,
            ),
            "mandate": str(states.get(theme_id, {}).get("mandate") or ""),
        }
        for theme_id in formal_pool
    }
    artifact: dict[str, Any] = {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "protocol_hash": protocol_hash,
        "as_of": point.isoformat(),
        "run_id": str(run_id or ""),
        "source_stage": "post_control_chain",
        "only_formal_producer": True,
        "status": "formal" if formal_symbols else "valid_empty",
        "formal_pool": formal_pool,
        "formal_symbols": formal_symbols,
        "per_theme": per_theme,
        "per_symbol": per_symbol,
        "tactical_lane_cap": tactical,
        "tactical_non_tech_nav_used": non_tech_nav_used,
        "tactical_non_tech_nav_cap": non_tech_nav_cap,
        "forced_theme_count": 0,
        "deterministic": True,
        "no_network": True,
        "no_llm": True,
    }
    artifact["reconciliation_hash"] = _stable_payload_hash(artifact)
    return artifact


def write_theme_formal_reconciliation_artifact(
    path: str | Path,
    artifact: Mapping[str, Any],
) -> Path:
    target = Path(path)
    payload = dict(artifact or {})
    expected_hash = str(payload.get("reconciliation_hash") or "")
    if not expected_hash or _stable_payload_hash(payload, exclude_hash=True) != expected_hash:
        raise ValueError("theme reconciliation hash mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    readback = json.loads(target.read_text(encoding="utf-8"))
    if _stable_payload_hash(readback, exclude_hash=True) != expected_hash:
        raise RuntimeError("theme reconciliation readback hash mismatch")
    return target


def persist_theme_formal_reconciliation_artifact(
    root_dir: str | Path,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(artifact or {})
    reconciliation_hash = str(payload.get("reconciliation_hash") or "")
    as_of_key = "".join(
        character for character in str(payload.get("as_of") or "") if character.isdigit()
    )[:8]
    if len(reconciliation_hash) != 64 or len(as_of_key) != 8:
        raise ValueError("invalid reconciliation identity")
    target = Path(root_dir) / as_of_key / f"{reconciliation_hash}.json"
    if target.exists():
        readback = json.loads(target.read_text(encoding="utf-8"))
        if readback != payload:
            raise RuntimeError("existing reconciliation artifact content mismatch")
        if target.stat().st_mode & 0o777 != 0o600:
            raise RuntimeError("existing reconciliation artifact permissions are not 0600")
        return {
            "status": "idempotent_readback",
            "path": str(target),
            "reconciliation_hash": reconciliation_hash,
            "readback_verified": True,
        }
    write_theme_formal_reconciliation_artifact(target, payload)
    return {
        "status": "persisted",
        "path": str(target),
        "reconciliation_hash": reconciliation_hash,
        "readback_verified": True,
    }


def transition_theme_lifecycle(
    previous: Mapping[str, Any],
    *,
    desired: str,
    as_of: str,
    hard_kill: bool = False,
    confirmation_date: str | None = None,
    valid_trading_dates: Sequence[str] = (),
    config: ThemeProtocolConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ThemeProtocolConfig()
    as_of_date = _parse_date(as_of)
    if as_of_date is None:
        raise ValueError("as_of must be a valid date")
    trading_dates = _valid_trading_date_set(
        valid_trading_dates,
        as_of=as_of_date,
    )
    current = str(previous.get("lifecycle") or ThemeLifecycle.DISCOVERY.value)
    if current not in _LIFECYCLE_ORDER:
        current = ThemeLifecycle.DISCOVERY.value
    if hard_kill:
        return {
            "lifecycle": ThemeLifecycle.BROKEN.value,
            "pending_transition": "",
            "pending_confirmation_dates": [],
        }
    if current == ThemeLifecycle.BROKEN.value or desired == current:
        return {
            "lifecycle": current,
            "pending_transition": "",
            "pending_confirmation_dates": [],
        }
    pending_transition = str(previous.get("pending_transition") or "")
    pending_dates = {
        parsed.isoformat()
        for value in list(previous.get("pending_confirmation_dates") or [])
        if (parsed := _parse_date(value)) is not None
        and parsed <= as_of_date
        and parsed.weekday() < 5
        and parsed.isoformat() in trading_dates
    }
    if pending_transition != desired:
        pending_dates = set()
    candidate = _parse_date(confirmation_date)
    if (
        candidate is not None
        and candidate <= as_of_date
        and candidate.weekday() < 5
        and candidate.isoformat() in trading_dates
    ):
        pending_dates.add(candidate.isoformat())
    is_cooling = desired == ThemeLifecycle.COOLING.value
    is_upward = _LIFECYCLE_ORDER.get(desired, -1) > _LIFECYCLE_ORDER.get(current, -1)
    required = cfg.cooling_confirmation_days if is_cooling else cfg.upward_confirmation_days
    if not is_upward and not is_cooling:
        required = cfg.cooling_confirmation_days
    if len(pending_dates) >= max(int(required), 1):
        return {
            "lifecycle": desired,
            "pending_transition": "",
            "pending_confirmation_dates": [],
        }
    return {
        "lifecycle": current,
        "pending_transition": desired,
        "pending_confirmation_dates": sorted(pending_dates),
    }


def _valid_trading_date_set(
    values: Sequence[str],
    *,
    as_of: date,
) -> set[str]:
    result: set[str] = set()
    for value in values:
        parsed = _parse_date(value)
        if parsed is None or parsed > as_of or parsed.weekday() >= 5:
            continue
        result.add(parsed.isoformat())
    return result


def _confirmation_trade_date(
    payload: Mapping[str, Any],
    *,
    valid_trading_dates: set[str],
    as_of: date,
) -> str | None:
    candidates = (
        payload.get("confirmation_trade_date"),
        payload.get("latest_trade_date"),
        payload.get("trade_date"),
        as_of.isoformat(),
    )
    for value in candidates:
        parsed = _parse_date(value)
        if (
            parsed is not None
            and parsed <= as_of
            and parsed.weekday() < 5
            and parsed.isoformat() in valid_trading_dates
        ):
            return parsed.isoformat()
    return None


def tactical_lane_cap(markov_regime: str) -> dict[str, Any]:
    caps = {
        "趋势上涨": (0.15, 2),
        "震荡低波": (0.10, 1),
        "震荡高波": (0.05, 1),
        "趋势下跌": (0.0, 0),
    }
    nav_cap, max_positions = caps.get(str(markov_regime or ""), (0.0, 0))
    return {
        "regime": str(markov_regime or "unknown"),
        "non_tech_nav_cap": nav_cap,
        "non_tech_max_positions": max_positions,
        "enabled": nav_cap > 0.0 and max_positions > 0,
    }


def _apply_tactical_position_cap(
    states: list[dict[str, Any]],
    *,
    tactical: Mapping[str, Any],
) -> None:
    non_tech_prequalified = sorted(
        (
            state
            for state in states
            if not state.get("prequalification_blockers")
            and state.get("mandate") not in {"technology", "advanced_manufacturing"}
        ),
        key=lambda state: (
            -float(state.get("adjusted_percentile_rank") or 0.0),
            str(state.get("theme_id") or ""),
        ),
    )
    max_positions = max(int(tactical.get("non_tech_max_positions") or 0), 0)
    for index, state in enumerate(non_tech_prequalified):
        if index < max_positions and tactical.get("enabled") is True:
            continue
        blockers = list(state.get("eligibility_blockers") or ())
        blocker = (
            "tactical_lane_closed_by_markov"
            if max_positions <= 0 or tactical.get("enabled") is not True
            else "tactical_position_cap_exceeded"
        )
        if blocker not in blockers:
            blockers.append(blocker)
        state["eligibility_blockers"] = tuple(blockers)
        prequalification_blockers = list(
            state.get("prequalification_blockers") or ()
        )
        if blocker not in prequalification_blockers:
            prequalification_blockers.append(blocker)
        state["prequalification_blockers"] = tuple(prequalification_blockers)
        if state.get("lane") == "formal_investable":
            state["lane"] = "market_observation"


def build_theme_protocol_hash(*, taxonomy: ThemeTaxonomy, config: ThemeProtocolConfig) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "implementation_code_sha256": theme_protocol_code_hash(),
        "taxonomy": taxonomy.to_dict(),
        "config": config.to_dict(),
        "rank_weights": {
            "attention": 0.45,
            "industrial_validation": 0.35,
            "market_confirmation": 0.20,
        },
        "lifecycle": [item.value for item in ThemeLifecycle],
        "tactical_caps": {
            regime: tactical_lane_cap(regime)
            for regime in ("趋势上涨", "震荡低波", "震荡高波", "趋势下跌")
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def theme_protocol_code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _available_events(
    values: Sequence[ThemeEvidenceEvent | Mapping[str, Any]],
    *,
    as_of: str,
) -> list[ThemeEvidenceEvent]:
    available: dict[str, ThemeEvidenceEvent] = {}
    for value in values:
        try:
            event = (
                value
                if isinstance(value, ThemeEvidenceEvent)
                else ThemeEvidenceEvent.from_mapping(value)
            )
        except (TypeError, ValueError):
            continue
        if not event.is_available(as_of):
            continue
        previous = available.get(event.event_id)
        if previous is None or event.available_at >= previous.available_at:
            available[event.event_id] = event
    return sorted(available.values(), key=lambda event: (event.available_at, event.event_id))


def _active_v2_membership_context(
    values: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
) -> dict[str, Any]:
    selected, counts = _dedupe_current_v2_memberships(values, as_of=as_of)
    active: dict[tuple[str, str], ThemeMembership] = {}
    invalidated_theme_ids: set[str] = set()
    point = _parse_date(as_of)
    for key, membership in selected.items():
        if membership.is_active(as_of):
            active[key] = membership
            continue
        available = _parse_date(membership.available_at)
        effective_from = _parse_date(membership.effective_from)
        if (
            point is not None
            and available is not None
            and effective_from is not None
            and available <= point
            and effective_from <= point
        ):
            invalidated_theme_ids.add(membership.theme_id)
    theme_ids = sorted({membership.theme_id for membership in active.values()})
    invalidated_theme_ids.difference_update(theme_ids)
    diagnostic_notes: list[str] = []
    if counts["non_v2_count"]:
        diagnostic_notes.append(
            f"non_v2_membership_ignored={counts['non_v2_count']}"
        )
    if counts["invalid_count"]:
        diagnostic_notes.append(
            f"invalid_v2_membership_ignored={counts['invalid_count']}"
        )
    if counts["duplicate_count"]:
        diagnostic_notes.append(
            f"duplicate_v2_membership_resolved={counts['duplicate_count']}"
        )
    if counts["future_revision_count"]:
        diagnostic_notes.append(
            f"future_v2_membership_revision_ignored={counts['future_revision_count']}"
        )
    diagnostic_notes.append(f"active_v2_membership_count={len(active)}")
    active_by_symbol: dict[str, list[ThemeMembership]] = {}
    for membership in active.values():
        active_by_symbol.setdefault(membership.symbol, []).append(membership)
    return {
        "status": "success" if active else "coverage_blocked",
        "active_theme_ids": theme_ids,
        "active_membership_count": len(active),
        "invalidated_theme_ids": sorted(invalidated_theme_ids),
        "membership_hash": _membership_snapshot_hash(active_by_symbol),
        "diagnostic_notes": diagnostic_notes,
    }


def _active_v2_memberships_by_symbol(
    values: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    as_of: str,
) -> dict[str, list[ThemeMembership]]:
    flattened = [
        dict(raw or {})
        for details in values.values()
        for raw in list(details or [])
        if isinstance(raw, Mapping)
    ]
    selected, _counts = _dedupe_current_v2_memberships(
        flattened,
        as_of=as_of,
    )
    result: dict[str, list[ThemeMembership]] = {}
    for (symbol, _theme_id), membership in selected.items():
        if membership.is_active(as_of):
            result.setdefault(symbol, []).append(membership)
    for symbol in result:
        result[symbol].sort(key=lambda item: (item.theme_id, item.membership_id))
    return result


def _dedupe_current_v2_memberships(
    values: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
) -> tuple[dict[tuple[str, str], ThemeMembership], dict[str, int]]:
    selected: dict[tuple[str, str], ThemeMembership] = {}
    invalid_count = 0
    non_v2_count = 0
    valid_count = 0
    future_revision_count = 0
    point = _parse_date(as_of)
    for raw in values:
        payload = dict(raw or {})
        if str(payload.get("schema_version") or "") != MEMBERSHIP_SCHEMA_VERSION:
            non_v2_count += 1
            continue
        try:
            membership = ThemeMembership.from_mapping(payload)
        except (TypeError, ValueError):
            invalid_count += 1
            continue
        available = _parse_date(membership.available_at)
        effective_from = _parse_date(membership.effective_from)
        if (
            point is None
            or available is None
            or effective_from is None
            or available > point
            or effective_from > point
        ):
            future_revision_count += 1
            continue
        valid_count += 1
        key = (membership.symbol, membership.theme_id)
        previous = selected.get(key)
        if previous is None or _membership_precedence_key(
            membership
        ) > _membership_precedence_key(previous):
            selected[key] = membership
    return selected, {
        "invalid_count": invalid_count,
        "non_v2_count": non_v2_count,
        "duplicate_count": max(valid_count - len(selected), 0),
        "future_revision_count": future_revision_count,
    }


def _membership_precedence_key(
    membership: ThemeMembership,
) -> tuple[float, int, str, str]:
    updated = str(membership.updated_at or "").strip()
    try:
        parsed = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        revision_timestamp = parsed.timestamp()
        has_valid_updated_at = 1
    except ValueError:
        fallback = (
            _parse_date(membership.available_at)
            or _parse_date(membership.effective_from)
            or date.min
        )
        revision_timestamp = datetime(
            fallback.year,
            fallback.month,
            fallback.day,
            tzinfo=timezone.utc,
        ).timestamp()
        has_valid_updated_at = 0
    payload_hash = hashlib.sha256(
        json.dumps(
            membership.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return (
        revision_timestamp,
        has_valid_updated_at,
        str(membership.membership_id or ""),
        payload_hash,
    )


def _membership_snapshot_hash(
    values: Mapping[str, Sequence[ThemeMembership]],
) -> str:
    records = [
        membership.to_dict()
        for symbol in sorted(values)
        for membership in sorted(
            values[symbol],
            key=lambda item: (item.theme_id, item.membership_id),
        )
    ]
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _approved_theses_by_theme(
    values: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
) -> dict[str, Mapping[str, Any]]:
    point = _parse_date(as_of)
    result: dict[str, Mapping[str, Any]] = {}
    for raw in values:
        payload = dict(raw or {})
        if str(payload.get("status") or "").lower() != "approved":
            continue
        available = _parse_date(payload.get("available_at") or payload.get("approved_at"))
        approved = _parse_date(payload.get("approved_at"))
        review_by = _parse_date(payload.get("review_by"))
        if (
            point is None
            or available is None
            or approved is None
            or available > point
            or approved > point
            or (review_by is not None and review_by < point)
        ):
            continue
        theme_id = str(payload.get("theme_id") or "")
        if not theme_id:
            continue
        previous = result.get(theme_id)
        if previous is None or _pevc_revision_key(payload) > _pevc_revision_key(
            previous
        ):
            result[theme_id] = payload
    return result


def _pevc_revision_key(
    payload: Mapping[str, Any],
) -> tuple[tuple[tuple[int, int | str], ...], date, str]:
    approved = _parse_date(payload.get("approved_at")) or date.min
    version_text = str(payload.get("version") or "").strip().lower()
    if version_text.startswith("v"):
        version_text = version_text[1:]
    natural_parts: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", version_text):
        if not part:
            continue
        natural_parts.append((1, int(part)) if part.isdigit() else (0, part))
    content_hash = str(payload.get("content_hash") or "").strip().lower()
    return tuple(natural_parts), approved, content_hash


def _attention_score(payload: Mapping[str, Any], events: Sequence[ThemeEvidenceEvent]) -> float:
    explicit = payload.get("attention")
    if explicit is not None:
        return _clamp(explicit)
    score = payload.get("smoothed_score")
    if score is None:
        score = payload.get("score")
    base = _normalize_score(score)
    attention_events = [event for event in events if event.event_type == "attention"]
    if not attention_events:
        return base
    event_score = _weighted_event_score(attention_events)
    return _clamp(0.75 * base + 0.25 * event_score)


def _industrial_validation(events: Sequence[ThemeEvidenceEvent]) -> float:
    industrial = [
        event
        for event in events
        if event.event_type in _INDUSTRIAL_EVENT_TYPES
    ]
    if not industrial:
        return 0.0
    type_diversity = len({event.event_type for event in industrial}) / 5.0
    return _clamp(0.80 * _weighted_event_score(industrial) + 0.20 * type_diversity)


def _market_confirmation(payload: Mapping[str, Any]) -> float:
    explicit = payload.get("market_confirmation")
    if explicit is not None:
        return _clamp(explicit)
    return _clamp(
        0.35 * _clamp(payload.get("breadth"))
        + 0.30 * _clamp(payload.get("momentum"))
        + 0.20 * _clamp(payload.get("volume_confirmation"))
        + 0.15 * _clamp(payload.get("acceleration"))
    )


def _evidence_confidence(
    events: Sequence[ThemeEvidenceEvent],
    payload: Mapping[str, Any],
) -> float:
    industrial = [event for event in events if event.event_type != "attention"]
    if industrial:
        return _clamp(sum(event.confidence for event in industrial) / len(industrial))
    return _clamp(payload.get("evidence_confidence"))


def _weighted_event_score(events: Sequence[ThemeEvidenceEvent]) -> float:
    weights = [max(event.confidence, 0.05) for event in events]
    denominator = sum(weights)
    if denominator <= 0:
        return 0.0
    return _clamp(
        sum(event.direction * event.strength * weight for event, weight in zip(events, weights))
        / denominator
    )


def _desired_lifecycle(
    *,
    attention: float,
    industrial_validation: float,
    market_confirmation: float,
    crowding: float,
    hard_kill: bool,
) -> str:
    if hard_kill:
        return ThemeLifecycle.BROKEN.value
    if crowding >= 0.70 and attention >= 0.55:
        return ThemeLifecycle.CROWDED.value
    if attention >= 0.55 and industrial_validation >= 0.45 and market_confirmation >= 0.45:
        return ThemeLifecycle.VALIDATED_TREND.value
    if attention >= 0.40 or market_confirmation >= 0.40:
        return ThemeLifecycle.WARMING.value
    if attention < 0.30 and market_confirmation < 0.30:
        return ThemeLifecycle.COOLING.value
    return ThemeLifecycle.DISCOVERY.value


def _eligibility_blockers(
    *,
    theme_id: str,
    node_present: bool,
    tradable_node: bool,
    valid_membership: bool,
    attention_60d: float | None,
    attention_120d: float | None,
    attention_history_coverage: float,
    attention: float,
    industrial_validation: float,
    market_confirmation: float,
    crowding: float | None,
    valuation_risk: float | None,
    evidence_confidence: float,
    latest_industrial_evidence_available_at: str,
    as_of: date,
    lifecycle: str,
    gates: Mapping[str, Any],
    config: ThemeProtocolConfig,
) -> list[str]:
    del theme_id
    blockers: list[str] = []
    if not node_present:
        blockers.append("taxonomy_node_missing")
    elif not tradable_node:
        blockers.append("taxonomy_node_not_tradable")
    if not valid_membership:
        blockers.append("pit_membership_missing")
    if attention_60d is None or attention_120d is None:
        blockers.append("long_horizon_attention_unknown")
    elif (
        attention_history_coverage
        < config.long_horizon_attention_min_coverage
    ):
        blockers.append("long_horizon_attention_insufficient")
    if attention < config.attention_min:
        blockers.append("attention_below_gate")
    if industrial_validation < config.industrial_validation_min:
        blockers.append("industrial_validation_below_gate")
    if market_confirmation < config.market_confirmation_min:
        blockers.append("market_confirmation_below_gate")
    if evidence_confidence < config.evidence_confidence_min:
        blockers.append("evidence_confidence_below_gate")
    if _is_stale(
        latest_industrial_evidence_available_at,
        as_of,
        config.evidence_stale_days,
    ):
        blockers.append("stale_industrial_evidence")
    if crowding is None:
        blockers.append("crowding_unavailable")
    elif crowding >= config.crowding_block:
        blockers.append("crowding_gate_blocked")
    if valuation_risk is None:
        blockers.append("valuation_risk_unavailable")
    elif valuation_risk >= config.valuation_risk_block:
        blockers.append("valuation_risk_gate_blocked")
    if lifecycle not in {ThemeLifecycle.VALIDATED_TREND.value, ThemeLifecycle.CROWDED.value}:
        blockers.append("lifecycle_not_validated")
    for key, blocker in _DOWNSTREAM_GATE_FIELDS.items():
        if gates.get(key) is not True:
            blockers.append(blocker)
    return blockers


def _risk_flags(
    crowding: float | None,
    valuation_risk: float | None,
    latest_industrial_available: str,
    as_of: date,
    config: ThemeProtocolConfig,
) -> list[str]:
    flags: list[str] = []
    if crowding is None:
        flags.append("crowding_unknown")
    elif crowding >= config.crowding_block:
        flags.append("crowding_high")
    if valuation_risk is None:
        flags.append("valuation_risk_unknown")
    elif valuation_risk >= config.valuation_risk_block:
        flags.append("valuation_risk_high")
    if _is_stale(
        latest_industrial_available,
        as_of,
        config.evidence_stale_days,
    ):
        flags.append("evidence_stale")
    return flags


def _is_stale(value: str, point: date, stale_days: int) -> bool:
    available = _parse_date(value)
    return available is None or (point - available).days > max(int(stale_days), 0)


def _pevc_prior(thesis: Mapping[str, Any]) -> float:
    if not thesis:
        return 0.0
    explicit = thesis.get("prior_score")
    if explicit is not None:
        return _clamp(explicit)
    return _clamp(
        0.35 * _clamp(thesis.get("confidence"))
        + 0.20 * _clamp(thesis.get("technology_maturity"))
        + 0.20 * _clamp(thesis.get("moat_strength"))
        + 0.15 * _clamp(thesis.get("customer_validation"))
        + 0.10 * _clamp(thesis.get("commercialization_stage"))
    )


def _apply_percentile_ranks(states: list[dict[str, Any]], *, max_adjustment: float) -> None:
    if not states:
        return
    ordered = sorted(
        states,
        key=lambda state: (
            float(state["base_rank_score"]),
            str(state["theme_id"]),
        ),
    )
    denominator = max(len(ordered) - 1, 1)
    for index, state in enumerate(ordered):
        state["base_percentile_rank"] = index / denominator if len(ordered) > 1 else 1.0
        state["adjusted_percentile_rank"] = state["base_percentile_rank"]
    eligible = [
        state
        for state in states
        if not state["prequalification_blockers"]
    ]
    eligible_ordered = sorted(
        eligible,
        key=lambda state: (
            float(state["base_rank_score"]),
            str(state["theme_id"]),
        ),
    )
    eligible_denominator = max(len(eligible_ordered) - 1, 1)
    for index, state in enumerate(eligible_ordered):
        state["base_percentile_rank"] = (
            index / eligible_denominator if len(eligible_ordered) > 1 else 1.0
        )
        adjustment = max(
            0.0,
            min(max_adjustment, float(state["pevc_prior"]) * max_adjustment),
        )
        state["pevc_rank_adjustment"] = adjustment
        state["adjusted_percentile_rank"] = _clamp(state["base_percentile_rank"] + adjustment)


def _normalize_score(value: Any) -> float:
    numeric = _finite(value, 0.0)
    return _clamp(numeric / 100.0 if numeric > 1.0 else numeric)


def _optional_axis(value: Any) -> float | None:
    if value is None:
        return None
    numeric = _finite(value, math.nan)
    return _clamp(numeric) if math.isfinite(numeric) else None


def _available_risk_axis(
    payload: Mapping[str, Any],
    *,
    value_key: str,
    status_key: str,
) -> float | None:
    status = str(payload.get(status_key) or "").strip().lower()
    explicitly_available = payload.get(f"{value_key}_available") is True
    if status not in {"success", "available", "fresh"} and not explicitly_available:
        return None
    return _optional_axis(payload.get(value_key))


def _optional_unit_value(value: Any) -> float | None:
    if value is None:
        return None
    numeric = _finite(value, math.nan)
    return _clamp(numeric) if math.isfinite(numeric) else None


def _stable_payload_hash(
    payload: Mapping[str, Any],
    *,
    exclude_hash: bool = False,
) -> str:
    value = dict(payload or {})
    if exclude_hash:
        value.pop("reconciliation_hash", None)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, _finite(value, 0.0)))


def _finite(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        candidate = text[:10] if fmt == "%Y-%m-%d" else text[:8]
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    return None


def _date_text(value: Any, *, required: bool = True) -> str:
    parsed = _parse_date(value)
    if parsed is None:
        if required:
            raise ValueError(f"invalid date={value}")
        return ""
    return parsed.isoformat()
