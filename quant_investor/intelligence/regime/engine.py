"""One-step causal Markov filtering for market, industry, and theme layers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    content_ref,
    decimal_text,
    decimal_value,
    seal_content_addressed,
    timestamp,
    validate_content_addressed,
)
from ..evidence.models import validate_evidence_set
from .input import validate_regime_input

MARKET_STATES: Final = ("BULL", "RANGE", "HIGH_VOL", "BEAR")
INDUSTRY_STATES: Final = (
    "EARLY_EXPANSION",
    "EXPANSION",
    "PEAK",
    "DECLINE",
    "RECOVERY",
)
THEME_STATES: Final = (
    "EMERGING",
    "ACCELERATING",
    "MAINSTREAM",
    "CROWDED",
    "DECLINING",
)
LAYER_STATES: Final = {
    "industry": INDUSTRY_STATES,
    "market": MARKET_STATES,
    "theme": THEME_STATES,
}
REGIME_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence.multilayer-regime-receipt.v1"


def _probability(value: Any, *, label: str) -> Decimal:
    return decimal_value(
        value,
        label=label,
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )


def _distribution(
    value: Mapping[str, Any], *, states: Sequence[str], label: str
) -> dict[str, Decimal]:
    if type(value) is not dict or set(value) != set(states):
        raise IntelligenceContractError(f"{label} must contain the fixed state domain")
    result = {state: _probability(value[state], label=f"{label}.{state}") for state in states}
    if sum(result.values()) != Decimal("1"):
        raise IntelligenceContractError(f"{label} probabilities must sum exactly to 1")
    return result


def _transition_matrix(
    value: Mapping[str, Any], *, states: Sequence[str], label: str
) -> dict[str, dict[str, Decimal]]:
    if type(value) is not dict or set(value) != set(states):
        raise IntelligenceContractError(f"{label} must contain the fixed state domain")
    return {
        state: _distribution(value[state], states=states, label=f"{label}.{state}")
        for state in states
    }


def _filter_layer(
    *,
    states: Sequence[str],
    previous: Mapping[str, Any],
    transition: Mapping[str, Any],
    emission: Mapping[str, Any],
    label: str,
) -> tuple[str, dict[str, str], str]:
    prior = _distribution(previous, states=states, label=f"{label}.previous")
    matrix = _transition_matrix(transition, states=states, label=f"{label}.transition")
    if type(emission) is not dict or set(emission) != set(states):
        raise IntelligenceContractError(f"{label}.emission must contain the fixed state domain")
    likelihoods = {
        state: _probability(emission.get(state), label=f"{label}.emission.{state}")
        for state in states
    }

    with localcontext() as context:
        context.prec = 50
        predicted = {
            target: sum(
                (prior[source] * matrix[source][target] for source in states),
                Decimal("0"),
            )
            for target in states
        }
        unnormalized = {state: predicted[state] * likelihoods[state] for state in states}
        normalizer = sum(unnormalized.values(), Decimal("0"))
        if normalizer <= 0:
            raise IntelligenceContractError(f"{label} emission has zero probability mass")
        posterior = {state: unnormalized[state] / normalizer for state in states}
    selected = max(states, key=lambda state: (posterior[state], -states.index(state)))
    return (
        selected,
        {state: decimal_text(posterior[state]) for state in states},
        decimal_text(predicted[selected]),
    )


def infer_multilayer_regime(
    *,
    regime_input: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    """Infer all required layers using one forward filter step and no smoothing."""

    cutoff = timestamp(as_of, label="as_of")
    input_row = validate_regime_input(regime_input, as_of=cutoff)
    previous_distributions = input_row["previous_distributions"]
    transition_matrices = input_row["transition_matrices"]
    emission_likelihoods = input_row["emission_likelihoods"]
    for name, value in {
        "emission_likelihoods": emission_likelihoods,
        "previous_distributions": previous_distributions,
        "transition_matrices": transition_matrices,
    }.items():
        if type(value) is not dict or set(value) != set(LAYER_STATES):
            raise IntelligenceContractError(f"{name} must contain market/industry/theme")
    evidence_rows = validate_evidence_set(evidence, as_of=cutoff)
    selected: dict[str, str] = {}
    posteriors: dict[str, dict[str, str]] = {}
    transitions: dict[str, str] = {}
    for layer in ("market", "industry", "theme"):
        state, posterior, transition_probability = _filter_layer(
            states=LAYER_STATES[layer],
            previous=previous_distributions[layer],
            transition=transition_matrices[layer],
            emission=emission_likelihoods[layer],
            label=layer,
        )
        selected[layer] = state
        posteriors[layer] = posterior
        transitions[layer] = transition_probability

    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "evidence_refs": [
                content_ref(row, identity_field="evidence_id") for row in evidence_rows
            ],
            "industry_state": selected["industry"],
            "input_ref": content_ref(input_row, identity_field="input_id"),
            "market_state": selected["market"],
            "posterior": posteriors,
            "production": False,
            "research_only": True,
            "theme_state": selected["theme"],
            "timestamp": cutoff,
            "transition_probability": transitions,
            "version": REGIME_RECEIPT_VERSION,
        },
        identity_field="receipt_id",
    )


def validate_regime_receipt(
    document: Mapping[str, Any],
    *,
    regime_input: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    normalized = validate_content_addressed(document, identity_field="receipt_id")
    if normalized.get("version") != REGIME_RECEIPT_VERSION:
        raise IntelligenceContractError("regime receipt version mismatch")
    expected_refs = normalized.get("evidence_refs")
    if type(expected_refs) is not list:
        raise IntelligenceContractError("regime evidence refs are missing")
    admitted = [
        row
        for row in validate_evidence_set(evidence, as_of=as_of)
        if content_ref(row, identity_field="evidence_id") in expected_refs
    ]
    expected = infer_multilayer_regime(
        regime_input=regime_input,
        evidence=admitted,
        as_of=as_of,
    )
    if expected != normalized:
        raise IntelligenceContractError("regime receipt replay mismatch")
    return normalized


__all__ = [
    "INDUSTRY_STATES",
    "MARKET_STATES",
    "REGIME_RECEIPT_VERSION",
    "THEME_STATES",
    "infer_multilayer_regime",
    "validate_regime_receipt",
]
