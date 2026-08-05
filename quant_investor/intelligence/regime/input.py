"""Source-bound causal inputs for multi-layer regime inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .._core import (
    IntelligenceContractError,
    exact_ref,
    require_no_future,
    seal_content_addressed,
    timestamp,
    validate_content_addressed,
)

REGIME_INPUT_VERSION: Final = "myquant.v17.research-intelligence.regime-input.v1"


def build_regime_input(
    *,
    previous_distributions: Mapping[str, Mapping[str, Any]],
    transition_matrices: Mapping[str, Mapping[str, Any]],
    emission_likelihoods: Mapping[str, Mapping[str, Any]],
    source_refs: Sequence[Mapping[str, Any]],
    observed_at: str,
    available_at: str,
) -> dict[str, Any]:
    """Bind every posterior-driving input to exact sources and availability time."""

    if not source_refs:
        raise IntelligenceContractError("regime input requires exact source refs")
    refs = [
        exact_ref(value, label=f"source_refs[{index}]") for index, value in enumerate(source_refs)
    ]
    keys = [(ref["relative_path"], ref["byte_sha256"]) for ref in refs]
    if len(keys) != len(set(keys)):
        raise IntelligenceContractError("regime input source refs must be unique")
    refs.sort(key=lambda ref: (ref["relative_path"].encode(), ref["byte_sha256"].encode()))
    observed = timestamp(observed_at, label="observed_at")
    available = timestamp(available_at, label="available_at")
    if observed > available:
        raise IntelligenceContractError("regime input cannot precede observation")
    if any(ref["cutoff"] > available for ref in refs):
        raise IntelligenceContractError("regime source is not available at available_at")
    for label, value in {
        "emission_likelihoods": emission_likelihoods,
        "previous_distributions": previous_distributions,
        "transition_matrices": transition_matrices,
    }.items():
        if type(value) is not dict:
            raise IntelligenceContractError(f"{label} must be an object")
    return seal_content_addressed(
        {
            "available_at": available,
            "emission_likelihoods": dict(emission_likelihoods),
            "observed_at": observed,
            "previous_distributions": dict(previous_distributions),
            "source_refs": refs,
            "transition_matrices": dict(transition_matrices),
            "version": REGIME_INPUT_VERSION,
        },
        identity_field="input_id",
    )


def validate_regime_input(document: Mapping[str, Any], *, as_of: str) -> dict[str, Any]:
    normalized = validate_content_addressed(document, identity_field="input_id")
    if normalized.get("version") != REGIME_INPUT_VERSION:
        raise IntelligenceContractError("regime input version mismatch")
    expected = build_regime_input(
        previous_distributions=normalized.get("previous_distributions", {}),
        transition_matrices=normalized.get("transition_matrices", {}),
        emission_likelihoods=normalized.get("emission_likelihoods", {}),
        source_refs=normalized.get("source_refs", []),
        observed_at=normalized.get("observed_at"),
        available_at=normalized.get("available_at"),
    )
    if expected != normalized:
        raise IntelligenceContractError("regime input replay mismatch")
    require_no_future(
        available_at=str(normalized["available_at"]),
        as_of=as_of,
        label="regime_input",
    )
    return normalized


__all__ = ["REGIME_INPUT_VERSION", "build_regime_input", "validate_regime_input"]
