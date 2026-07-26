"""Sealed-evidence deep-research weighting and red-flag evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

LAYER_NAMES = (
    "raw_facts",
    "derived_metrics",
    "research_inferences",
    "investment_judgments",
    "risk_alerts",
)
COVERAGE_SECTIONS = (
    "financial_reports_and_three_statement_reconciliation",
    "normalization_and_reversible_adjustments",
    "segments",
    "management_and_governance",
    "ownership",
    "industry_and_competition",
    "products_and_technology",
    "dcf",
    "reverse_dcf",
    "comparable_companies",
    "sotp_if_applicable",
    "bull_base_bear_scenarios",
    "catalysts",
    "counterevidence",
    "falsification_conditions",
    "continuous_monitoring_items",
)
SIGNAL_WEIGHTS: Mapping[str, float] = {
    "financial": 0.25,
    "business_model": 0.15,
    "industry": 0.15,
    "competitiveness": 0.20,
    "management": 0.10,
    "valuation": 0.15,
}
ALLOWED_SIGNAL_VALUES = frozenset({-1.0, -0.5, 0.0, 0.5, 1.0})
SEVERE_RED_FLAGS = (
    "audit_or_going_concern",
    "restatement_or_three_statement_failure",
    "fraud_or_material_penalty",
    "controller_appropriation_or_pledge_crisis",
    "material_related_party_or_governance_conflict",
    "liquidity_or_refinancing_break",
    "customer_or_supplier_concentration_break",
    "product_or_technology_obsolescence",
    "listing_or_delisting_risk",
    "core_thesis_falsified",
)
RESPONSE_KEYS = frozenset({"symbol", "layers", "coverage", "signals", "severe_red_flags"})
LAYER_ITEM_KEYS = frozenset({"layer", "content", "evidence_ids"})
COVERAGE_ITEM_KEYS = frozenset({"conclusion", "evidence_ids"})
SIGNAL_ITEM_KEYS = frozenset({"signal", "evidence_ids"})
RED_FLAG_ITEM_KEYS = frozenset({"triggered", "evidence_ids"})


@dataclass(frozen=True)
class DeepResearchEvaluation:
    status: str
    research_complete: bool
    f_eligible: bool
    buy_permission_revoked: bool
    severe_red_flags: tuple[str, ...]
    weighted_signal: float
    delta: float
    base_q25_252: float | None
    adjusted_q25_252: float | None
    blockers: tuple[str, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "status": self.status,
            "research_complete": self.research_complete,
            "f_eligible": self.f_eligible,
            "buy_permission_revoked": self.buy_permission_revoked,
            "severe_red_flags": list(self.severe_red_flags),
            "weighted_signal": self.weighted_signal,
            "delta": self.delta,
            "base_q25_252": self.base_q25_252,
            "adjusted_q25_252": self.adjusted_q25_252,
            "blockers": list(self.blockers),
            "authority": False,
        }


def _strict_bool(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _canonical_string(value: object) -> str | None:
    return value if isinstance(value, str) and value and value.strip() == value else None


def _evidence_ids(item: Mapping[str, object]) -> tuple[str, ...] | None:
    raw = item.get("evidence_ids", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        return None
    values: list[str] = []
    for value in raw:
        canonical = _canonical_string(value)
        if canonical is None:
            return None
        values.append(canonical)
    return tuple(values) if len(values) == len(set(values)) else None


def _validate_evidence_refs(
    *,
    item: object,
    path: str,
    sealed: frozenset[str],
    exact_keys: frozenset[str],
    require_nonempty: bool,
    blockers: list[str],
) -> None:
    if not isinstance(item, Mapping):
        blockers.append(f"invalid_object:{path}")
        return
    if set(item) != set(exact_keys):
        blockers.append(f"object_keys_mismatch:{path}")
    refs = _evidence_ids(item)
    if refs is None:
        blockers.append(f"invalid_evidence_ids:{path}")
        return
    if require_nonempty and not refs:
        blockers.append(f"missing_evidence:{path}")
    unknown = sorted(set(refs).difference(sealed))
    if unknown:
        blockers.append(f"unsealed_evidence:{path}:{','.join(unknown)}")


def evaluate_deep_research(
    response: Mapping[str, object],
    *,
    sealed_symbol: str,
    sealed_evidence_ids: Iterable[str],
    base_q25_by_horizon: Mapping[int, float],
    base_eligible: bool,
) -> DeepResearchEvaluation:
    """Validate only caller-supplied evidence and adjust q25 by at most +/-10%."""

    blockers: list[str] = []
    if not isinstance(response, Mapping):
        return DeepResearchEvaluation(
            "DEEP_RESEARCH_INVALID",
            False,
            False,
            False,
            (),
            0.0,
            0.0,
            None,
            None,
            ("response_not_object",),
        )
    if set(response) != set(RESPONSE_KEYS):
        blockers.append("response_keys_mismatch")
    expected_symbol = _canonical_string(sealed_symbol)
    if expected_symbol is None or _canonical_string(response.get("symbol")) != expected_symbol:
        blockers.append("sealed_symbol_mismatch")
    canonical_sealed: list[str] = []
    for value in tuple(sealed_evidence_ids):
        canonical = _canonical_string(value)
        if canonical is None:
            blockers.append("sealed_evidence_set_invalid")
            break
        canonical_sealed.append(canonical)
    sealed = frozenset(canonical_sealed)
    if not sealed or len(sealed) != len(canonical_sealed):
        blockers.append("sealed_evidence_set_invalid")
    if _strict_bool(base_eligible) is None:
        blockers.append("base_eligible_not_strict_bool")

    layers = response.get("layers")
    if not isinstance(layers, Mapping):
        blockers.append("layers_missing")
    else:
        if set(layers) != set(LAYER_NAMES):
            blockers.append("layer_set_mismatch")
        for layer in LAYER_NAMES:
            items = layers.get(layer)
            if not isinstance(items, list) or not items:
                blockers.append(f"layer_empty_or_invalid:{layer}")
                continue
            for index, item in enumerate(items):
                path = f"layers.{layer}[{index}]"
                _validate_evidence_refs(
                    item=item,
                    path=path,
                    sealed=sealed,
                    exact_keys=LAYER_ITEM_KEYS,
                    require_nonempty=True,
                    blockers=blockers,
                )
                if isinstance(item, Mapping) and item.get("layer") != layer:
                    blockers.append(f"layer_mixing:{path}")
                if isinstance(item, Mapping) and _canonical_string(item.get("content")) is None:
                    blockers.append(f"layer_content_invalid:{path}")

    coverage = response.get("coverage")
    if not isinstance(coverage, Mapping):
        blockers.append("coverage_missing")
    else:
        if set(coverage) != set(COVERAGE_SECTIONS):
            blockers.append("coverage_set_mismatch")
        for section in COVERAGE_SECTIONS:
            item = coverage.get(section)
            _validate_evidence_refs(
                item=item,
                path=f"coverage.{section}",
                sealed=sealed,
                exact_keys=COVERAGE_ITEM_KEYS,
                require_nonempty=True,
                blockers=blockers,
            )
            if isinstance(item, Mapping) and _canonical_string(item.get("conclusion")) is None:
                blockers.append(f"coverage_conclusion_invalid:{section}")

    weighted_signal = 0.0
    signals = response.get("signals")
    if not isinstance(signals, Mapping) or set(signals) != set(SIGNAL_WEIGHTS):
        blockers.append("signal_set_mismatch")
    else:
        for dimension, weight in SIGNAL_WEIGHTS.items():
            item = signals.get(dimension)
            _validate_evidence_refs(
                item=item,
                path=f"signals.{dimension}",
                sealed=sealed,
                exact_keys=SIGNAL_ITEM_KEYS,
                require_nonempty=True,
                blockers=blockers,
            )
            if not isinstance(item, Mapping):
                continue
            raw_signal = item.get("signal")
            if isinstance(raw_signal, (bool, np.bool_)) or not isinstance(
                raw_signal, (int, float, np.integer, np.floating)
            ):
                blockers.append(f"invalid_signal:{dimension}")
                continue
            signal = float(raw_signal)
            if not np.isfinite(signal) or signal not in ALLOWED_SIGNAL_VALUES:
                blockers.append(f"invalid_signal:{dimension}")
                continue
            weighted_signal += weight * signal

    triggered: list[str] = []
    flags = response.get("severe_red_flags")
    if not isinstance(flags, Mapping) or set(flags) != set(SEVERE_RED_FLAGS):
        blockers.append("red_flag_set_mismatch")
    else:
        for flag in SEVERE_RED_FLAGS:
            item = flags.get(flag)
            _validate_evidence_refs(
                item=item,
                path=f"severe_red_flags.{flag}",
                sealed=sealed,
                exact_keys=RED_FLAG_ITEM_KEYS,
                require_nonempty=False,
                blockers=blockers,
            )
            if not isinstance(item, Mapping):
                continue
            value = _strict_bool(item.get("triggered"))
            if value is None:
                blockers.append(f"invalid_red_flag_value:{flag}")
                continue
            refs = _evidence_ids(item) or ()
            if value and not refs:
                blockers.append(f"triggered_red_flag_without_evidence:{flag}")
            if value:
                triggered.append(flag)

    base_values: dict[int, float] = {}
    if set(base_q25_by_horizon) != {120, 252, 378}:
        blockers.append("base_q25_horizon_set_mismatch")
    for horizon in (120, 252, 378):
        raw = base_q25_by_horizon.get(horizon)
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.integer, np.floating)
        ):
            blockers.append(f"base_q25_missing:{horizon}")
            continue
        value = float(raw)
        if not np.isfinite(value):
            blockers.append(f"base_q25_nonfinite:{horizon}")
        else:
            base_values[horizon] = value
            if value <= 0:
                blockers.append(f"base_q25_not_positive:{horizon}")
    if _strict_bool(base_eligible) is not True:
        blockers.append("base_fundamental_ineligible")

    research_complete = not blockers
    delta = float(np.clip(0.10 * weighted_signal, -0.10, 0.10))
    base_252 = base_values.get(252)
    allowed = (
        research_complete
        and _strict_bool(base_eligible) is True
        and set(base_values) == {120, 252, 378}
        and all(value > 0 for value in base_values.values())
        and not triggered
    )
    adjusted = base_252 * (1.0 + delta) if allowed and base_252 is not None else None
    status = (
        "DEEP_RESEARCH_INVALID"
        if blockers
        else "DEEP_RESEARCH_COMPLETE_RED_FLAG" if triggered else "DEEP_RESEARCH_COMPLETE"
    )
    return DeepResearchEvaluation(
        status,
        research_complete,
        bool(allowed),
        bool(triggered),
        tuple(triggered),
        float(weighted_signal),
        delta,
        base_252,
        adjusted,
        tuple(dict.fromkeys(blockers)),
    )
