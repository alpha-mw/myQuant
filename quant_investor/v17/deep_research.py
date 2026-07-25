"""Sealed-evidence-only v17 Fundamental deep-research importer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .resources import load_json_resource
from .storage import file_sha256

TEMPLATE_SOURCE_SHA256 = "7dbda6970ea058aa656f1e7c12a0be24feb5ea7ce99d3e33e885dcc84f438e0e"
TEMPLATE_RESOURCE_SHA256 = "434cf726270d5f65eb7a0d2e2f2569363b281be23c16ee08304acb6962cc6537"
TEMPLATE_RESOURCE_NAME = "deep_research_template.v1.json"

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


def _strict_bool(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _canonical_string(value: object) -> str | None:
    if not isinstance(value, str) or not value or value.strip() != value:
        return None
    return value


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
    if len(values) != len(set(values)):
        return None
    return tuple(values)


def load_deep_research_template() -> Mapping[str, object]:
    """Load the exact packaged template and verify its frozen source binding.

    Runtime replay is intentionally self-contained in the package.  The
    original Markdown is verified when the canonical package is built, but an
    installed wheel must not depend on the user's Downloads directory still
    being present.
    """

    payload = load_json_resource(
        TEMPLATE_RESOURCE_NAME,
        expected_sha256=TEMPLATE_RESOURCE_SHA256,
    )
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("deep-research template source binding missing")
    if source.get("sha256") != TEMPLATE_SOURCE_SHA256:
        raise ValueError("deep-research source SHA binding mismatch")
    if _canonical_string(source.get("path")) is None:
        raise ValueError("deep-research source path binding invalid")
    if tuple(payload.get("layers", ())) != LAYER_NAMES:
        raise ValueError("deep-research layer template mismatch")
    if tuple(payload.get("coverage", ())) != COVERAGE_SECTIONS:
        raise ValueError("deep-research coverage template mismatch")
    if payload.get("signals") != dict(SIGNAL_WEIGHTS):
        raise ValueError("deep-research signal weight template mismatch")
    if tuple(payload.get("severe_red_flags", ())) != SEVERE_RED_FLAGS:
        raise ValueError("deep-research red-flag template mismatch")
    return payload


def verify_deep_research_template_source(source_path: str | Path) -> str:
    """Verify the owner-supplied template source during package generation."""

    observed = file_sha256(Path(source_path))
    if observed != TEMPLATE_SOURCE_SHA256:
        raise ValueError("deep-research frozen source byte SHA mismatch")
    return observed


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
    """Validate a sealed response and adjust only an eligible base 252d q25."""

    load_deep_research_template()
    blockers: list[str] = []
    if not isinstance(response, Mapping):
        return DeepResearchEvaluation(
            status="DEEP_RESEARCH_INVALID",
            research_complete=False,
            f_eligible=False,
            buy_permission_revoked=False,
            severe_red_flags=(),
            weighted_signal=0.0,
            delta=0.0,
            base_q25_252=None,
            adjusted_q25_252=None,
            blockers=("response_not_object",),
        )
    if set(response) != set(RESPONSE_KEYS):
        blockers.append("response_keys_mismatch")
    expected_symbol = _canonical_string(sealed_symbol)
    response_symbol = _canonical_string(response.get("symbol"))
    if expected_symbol is None or response_symbol != expected_symbol:
        blockers.append("sealed_symbol_mismatch")

    sealed_values = tuple(sealed_evidence_ids)
    canonical_sealed: list[str] = []
    for value in sealed_values:
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
            path = f"coverage.{section}"
            _validate_evidence_refs(
                item=item,
                path=path,
                sealed=sealed,
                exact_keys=COVERAGE_ITEM_KEYS,
                require_nonempty=True,
                blockers=blockers,
            )
            if isinstance(item, Mapping) and _canonical_string(item.get("conclusion")) is None:
                blockers.append(f"coverage_conclusion_invalid:{section}")

    signals = response.get("signals")
    weighted_signal = 0.0
    if not isinstance(signals, Mapping) or set(signals) != set(SIGNAL_WEIGHTS):
        blockers.append("signal_set_mismatch")
    else:
        for dimension, weight in SIGNAL_WEIGHTS.items():
            item = signals.get(dimension)
            path = f"signals.{dimension}"
            _validate_evidence_refs(
                item=item,
                path=path,
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

    flags = response.get("severe_red_flags")
    triggered: list[str] = []
    if not isinstance(flags, Mapping) or set(flags) != set(SEVERE_RED_FLAGS):
        blockers.append("red_flag_set_mismatch")
    else:
        for flag in SEVERE_RED_FLAGS:
            item = flags.get(flag)
            path = f"severe_red_flags.{flag}"
            _validate_evidence_refs(
                item=item,
                path=path,
                sealed=sealed,
                exact_keys=RED_FLAG_ITEM_KEYS,
                require_nonempty=False,
                blockers=blockers,
            )
            if not isinstance(item, Mapping):
                continue
            flag_value = _strict_bool(item.get("triggered"))
            if flag_value is None:
                blockers.append(f"invalid_red_flag_value:{flag}")
                continue
            refs = _evidence_ids(item) or ()
            if flag_value and not refs:
                blockers.append(f"triggered_red_flag_without_evidence:{flag}")
            if flag_value:
                triggered.append(flag)

    base_values: dict[int, float] = {}
    if set(base_q25_by_horizon) != {120, 252, 378}:
        blockers.append("base_q25_horizon_set_mismatch")
    for horizon in (120, 252, 378):
        raw_value = base_q25_by_horizon.get(horizon)
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, (int, float, np.integer, np.floating)
        ):
            blockers.append(f"base_q25_missing:{horizon}")
            continue
        base_value = float(raw_value)
        if not np.isfinite(base_value):
            blockers.append(f"base_q25_nonfinite:{horizon}")
        else:
            base_values[horizon] = base_value
            if base_value <= 0.0:
                blockers.append(f"base_q25_not_positive:{horizon}")
    if _strict_bool(base_eligible) is not True:
        blockers.append("base_fundamental_ineligible")

    research_complete = not blockers
    delta = float(np.clip(0.10 * weighted_signal, -0.10, 0.10))
    base_252 = base_values.get(252)
    adjustment_allowed = (
        research_complete
        and _strict_bool(base_eligible) is True
        and set(base_values) == {120, 252, 378}
        and all(value > 0.0 for value in base_values.values())
        and not triggered
    )
    adjusted = base_252 * (1.0 + delta) if adjustment_allowed and base_252 is not None else None
    if blockers:
        status = "DEEP_RESEARCH_INVALID"
    elif triggered:
        status = "DEEP_RESEARCH_COMPLETE_RED_FLAG"
    else:
        status = "DEEP_RESEARCH_COMPLETE"
    return DeepResearchEvaluation(
        status=status,
        research_complete=research_complete,
        f_eligible=bool(adjustment_allowed),
        buy_permission_revoked=bool(triggered),
        severe_red_flags=tuple(triggered),
        weighted_signal=float(weighted_signal),
        delta=delta,
        base_q25_252=base_252,
        adjusted_q25_252=adjusted,
        blockers=tuple(dict.fromkeys(blockers)),
    )


__all__ = [
    "ALLOWED_SIGNAL_VALUES",
    "COVERAGE_SECTIONS",
    "DeepResearchEvaluation",
    "LAYER_NAMES",
    "SEVERE_RED_FLAGS",
    "SIGNAL_WEIGHTS",
    "TEMPLATE_RESOURCE_SHA256",
    "TEMPLATE_SOURCE_SHA256",
    "evaluate_deep_research",
    "load_deep_research_template",
    "verify_deep_research_template_source",
]
