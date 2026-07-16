"""Deterministic point-in-time macro v2 snapshot construction."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime
from typing import Any, Iterable, Mapping

from quant_investor.macro.contracts import (
    MacroObservation,
    MacroSnapshot,
    SHANGHAI,
    canonical_hash,
    is_official_source,
    is_tushare_source,
    parse_timestamp,
    published_cutoff,
)
from quant_investor.macro.registry import (
    FRESHNESS_MAX_AGE_DAYS,
    INDUSTRY_CHAINS,
    INDUSTRY_COMPONENT_WEIGHTS,
    NATIONAL_DOMAIN_WEIGHTS,
    NATIONAL_INDICATORS,
    PERIOD_MAX_LAG_DAYS,
    REGISTRY_VERSION,
    SCORE_MODEL_VERSION,
    definition_for,
)


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _signal(history: list[float], polarity: float) -> float:
    if len(history) < 3:
        return 0.0
    latest = history[-1]
    baseline = history[:-1]
    median = statistics.median(baseline)
    deviations = [abs(value - median) for value in baseline]
    mad = statistics.median(deviations) if deviations else 0.0
    scale = max(mad * 1.4826, max(abs(median) * 0.05, 1e-9))
    return _clamp(((latest - median) / (2.5 * scale)) * polarity)


def _source_confidence(source: str) -> float:
    normalized = source.lower()
    if is_official_source(normalized):
        return 1.0
    if is_tushare_source(normalized):
        return 0.8
    if "strict_parquet" in normalized or "local_canonical" in normalized:
        return 1.0
    return 0.0


def _source_priority(source: str) -> int:
    normalized = source.lower()
    if is_official_source(normalized):
        return 3
    if "strict_parquet" in normalized or "local_canonical" in normalized:
        return 3
    if is_tushare_source(normalized):
        return 2
    return 0


def _select_vintages(
    observations: Iterable[Mapping[str, Any] | MacroObservation],
    *,
    cutoff: datetime,
    logical_as_of: date,
) -> tuple[list[MacroObservation], list[str]]:
    parsed: list[MacroObservation] = []
    blockers: list[str] = []
    for item in observations:
        try:
            payload = item.to_dict() if isinstance(item, MacroObservation) else item
            observation = MacroObservation.from_mapping(payload)
        except (TypeError, ValueError) as exc:
            blockers.append(str(exc))
            continue
        if observation.quality_status != "pass":
            blockers.append(f"quality_not_pass:{observation.indicator_id}")
            continue
        if (
            date.fromisoformat(observation.period_end) <= logical_as_of
            and parse_timestamp(
                observation.available_at,
                field_name="available_at",
            )
            <= cutoff
        ):
            parsed.append(observation)

    grouped: dict[tuple[str, str], list[MacroObservation]] = defaultdict(list)
    for item in parsed:
        grouped[(item.indicator_id, item.period_end)].append(item)

    selected: list[MacroObservation] = []
    for key in sorted(grouped):
        all_sources = grouped[key]
        best_priority = max(_source_priority(item.source_system) for item in all_sources)
        group = [item for item in all_sources if _source_priority(item.source_system) == best_priority]
        by_available_at: dict[str, list[MacroObservation]] = defaultdict(list)
        for item in group:
            by_available_at[item.available_at].append(item)
        for available_at, conflicts in by_available_at.items():
            values = {item.value for item in conflicts}
            if len(values) > 1:
                raise ValueError(f"conflicting_vintage:{key[0]}:{key[1]}:{available_at}")
        selected.append(sorted(group, key=lambda item: (item.available_at, item.vintage_id, item.content_hash))[-1])
    return selected, sorted(set(blockers))


def build_macro_snapshot(
    observations: Iterable[Mapping[str, Any] | MacroObservation],
    *,
    market: str = "CN",
    as_of: str,
    decision_cutoff_at: Any | None = None,
) -> MacroSnapshot:
    cutoff = published_cutoff(
        as_of if decision_cutoff_at is None else decision_cutoff_at
    )
    logical_as_of = published_cutoff(as_of).astimezone(SHANGHAI).date()
    selected_period_vintages, blockers = _select_vintages(
        observations,
        cutoff=cutoff,
        logical_as_of=logical_as_of,
    )
    histories: dict[str, list[MacroObservation]] = defaultdict(list)
    for item in selected_period_vintages:
        histories[item.indicator_id].append(item)
    for values in histories.values():
        values.sort(key=lambda item: (item.period_end, item.available_at, item.vintage_id, item.content_hash))

    latest_by_indicator = {key: values[-1] for key, values in histories.items() if values}
    indicator_signals: dict[str, float] = {}
    stale: list[str] = []
    stale_periods: list[str] = []
    insufficient_history: list[str] = []
    source_confidences: list[float] = []
    lineage: dict[str, Any] = {}
    for indicator_id in sorted(latest_by_indicator):
        latest = latest_by_indicator[indicator_id]
        lineage[indicator_id] = {
            "period_end": latest.period_end,
            "release_at": latest.release_at,
            "available_at": latest.available_at,
            "vintage_id": latest.vintage_id,
            "source_system": latest.source_system,
            "fallback": is_tushare_source(latest.source_system),
            "content_hash": latest.content_hash,
        }
        definition = definition_for(indicator_id, latest.frequency)
        if definition is None:
            blockers.append(f"unregistered_indicator:{indicator_id}")
            continue
        history = histories[indicator_id]
        if len(history) < 3:
            insufficient_history.append(indicator_id)
            continue
        max_age = FRESHNESS_MAX_AGE_DAYS.get(definition.frequency)
        age_days = (cutoff - parse_timestamp(latest.available_at, field_name="available_at")).total_seconds() / 86400.0
        if max_age is None or age_days > max_age:
            stale.append(indicator_id)
            continue
        period_lag_limit = PERIOD_MAX_LAG_DAYS.get(definition.frequency)
        period_end = datetime.fromisoformat(latest.period_end).date()
        period_lag_days = (cutoff.date() - period_end).days
        if period_lag_limit is None or period_lag_days < 0 or period_lag_days > period_lag_limit:
            stale_periods.append(indicator_id)
            continue
        indicator_signals[indicator_id] = _signal([item.value for item in history], definition.polarity)
        confidence = _source_confidence(latest.source_system)
        source_confidences.append(confidence)
        if confidence <= 0:
            blockers.append(f"untrusted_source:{indicator_id}")

    domain_values: dict[str, list[float]] = defaultdict(list)
    for indicator_id, signal in indicator_signals.items():
        definition = definition_for(indicator_id)
        if definition is not None and not indicator_id.startswith("industry."):
            domain_values[definition.domain].append(signal)
    national_states = {
        domain: round(sum(domain_values.get(domain, [])) / len(domain_values[domain]), 8)
        if domain_values.get(domain)
        else 0.0
        for domain in NATIONAL_DOMAIN_WEIGHTS
    }
    macro_score = sum(national_states[domain] * weight for domain, weight in NATIONAL_DOMAIN_WEIGHTS.items())

    chain_components: dict[str, dict[str, float]] = defaultdict(dict)
    for indicator_id, signal in indicator_signals.items():
        parts = indicator_id.split(".")
        if len(parts) == 3 and parts[0] == "industry" and parts[1] in INDUSTRY_CHAINS:
            chain_components[parts[1]][parts[2]] = signal
    industry_states: dict[str, float] = {}
    industry_coverage: dict[str, float] = {}
    for chain in INDUSTRY_CHAINS:
        components = chain_components.get(chain, {})
        coverage = len(components) / len(INDUSTRY_COMPONENT_WEIGHTS)
        industry_coverage[chain] = round(coverage, 8)
        if coverage < 0.70:
            industry_states[chain] = 0.0
            if components:
                blockers.append(f"industry_coverage_degraded:{chain}")
            continue
        denominator = sum(INDUSTRY_COMPONENT_WEIGHTS[name] for name in components)
        industry_states[chain] = round(
            sum(components[name] * INDUSTRY_COMPONENT_WEIGHTS[name] for name in components) / denominator,
            8,
        )

    expected_national = {item.indicator_id for item in NATIONAL_INDICATORS}
    national_present = expected_national.intersection(indicator_signals)
    national_coverage = len(national_present) / len(expected_national)
    confidence = national_coverage * (sum(source_confidences) / len(source_confidences) if source_confidences else 0.0)
    if national_coverage < 0.80:
        blockers.append("national_coverage_below_80pct")
    blockers.extend(f"stale_indicator:{item}" for item in stale)
    blockers.extend(f"stale_period:{item}" for item in stale_periods)
    blockers.extend(f"insufficient_history:{item}" for item in insufficient_history)
    readiness = "pass" if not blockers else ("degraded" if selected_period_vintages else "block")

    shadow_overlays: dict[str, Any] = {}
    for chain in INDUSTRY_CHAINS:
        chain_ready = industry_coverage[chain] >= 0.70 and readiness == "pass"
        theoretical = _clamp(5.0 * (0.40 * macro_score + 0.60 * industry_states[chain]) * confidence, -5.0, 5.0)
        shadow_overlays[chain] = {
            "applied": False,
            "delta_points": round(theoretical if chain_ready else 0.0, 8),
            "national_contribution": round(5.0 * 0.40 * macro_score * confidence, 8),
            "industry_contribution": round(5.0 * 0.60 * industry_states[chain] * confidence, 8),
            "confidence": round(confidence, 8),
            "reason": "observer_only" if chain_ready else "readiness_not_pass",
            "base_score_scale": "unresolved",
            "shadow_score_100": None,
        }

    snapshot = MacroSnapshot(
        market=str(market).upper(),
        as_of=str(as_of),
        published_cutoff=cutoff.isoformat(),
        registry_version=REGISTRY_VERSION,
        score_model_version=SCORE_MODEL_VERSION,
        selected_observation_hashes=tuple(sorted(item.content_hash for item in selected_period_vintages)),
        national_states=national_states,
        industry_chain_states=industry_states,
        market_confirmation={"state": national_states["market_confirmation"]},
        coverage={
            "national": round(national_coverage, 8),
            "industry_chains": industry_coverage,
        },
        freshness={
            "stale_availability": sorted(stale),
            "stale_periods": sorted(stale_periods),
            "insufficient_history": sorted(insufficient_history),
        },
        source_lineage=lineage,
        readiness_status=readiness,
        blockers=tuple(sorted(set(blockers))),
        macro_score=round(_clamp(macro_score), 8),
        confidence=round(_clamp(confidence, 0.0, 1.0), 8),
        shadow_overlays=shadow_overlays,
    )
    return replace(snapshot, snapshot_hash=canonical_hash(snapshot.hash_payload()))


__all__ = ["build_macro_snapshot"]
