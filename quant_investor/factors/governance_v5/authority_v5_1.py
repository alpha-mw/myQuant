"""Frozen authority surface for the non-admission Factor v5.1 phase."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any, Final

from ._core import (
    FactorGovernanceV5Error,
    canonical_bytes,
    common_fields,
    seal,
    validate_seal,
)
from .contracts_v5_1 import (
    FUTURE_ADMISSION_EVIDENCE_LANE,
    REGISTERED_STATE,
    validate_candidate_registration_v5_1,
)

AUTHORITY_MATRIX_VERSION: Final = "factor-research-authority-matrix.v5.1"
RESEARCH_DIAGNOSTIC_FACTOR_LANE: Final = "RESEARCH_DIAGNOSTIC_FACTOR_LANE"
LIFECYCLE_STATES: Final = (
    "ADMITTED",
    "CORE",
    "DEGRADED",
    "PROVISIONAL",
    "REGISTERED",
    "RESEARCH_ELIGIBLE",
    "RETIRED",
    "SHADOW",
)

_DENIED_CONSUMERS: Final = (
    "B0",
    "DECISION_V2",
    "FACTOR_REGISTRY",
    "I6",
    "V17_MAINLINE",
)


def build_authority_matrix_v5_1(
    *, registered_at: str, registration: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = validate_candidate_registration_v5_1(registration)
    registration_ref = {
        "artifact_id": normalized["registration_id"],
        "artifact_version": normalized["version"],
        "byte_sha256": hashlib.sha256(canonical_bytes(normalized)).hexdigest(),
        "semantic_sha256": normalized["semantic_sha256"],
    }
    return seal(
        {
            **common_fields(timestamp_value=registered_at),
            "bayesian_posterior_available": False,
            "denied_consumers": list(_DENIED_CONSUMERS),
            "factor_protocol_extension": "factor-governance-protocol.v5.1",
            "lanes": [
                {
                    "admission_eligible": False,
                    "lane": FUTURE_ADMISSION_EVIDENCE_LANE,
                    "stock_pool_builder_available": False,
                },
                {
                    "admission_eligible": False,
                    "lane": RESEARCH_DIAGNOSTIC_FACTOR_LANE,
                    "stock_pool_builder_available": False,
                },
            ],
            "lifecycle_states": list(LIFECYCLE_STATES),
            "lifecycle_transition_engine_available": False,
            "reachable_lifecycle_states": [REGISTERED_STATE],
            "registration_ref": registration_ref,
            "version": AUTHORITY_MATRIX_VERSION,
        },
        identity_field="authority_matrix_id",
    )


def validate_authority_matrix_v5_1(
    document: Mapping[str, Any], *, registration: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="authority_matrix_id")
    expected = build_authority_matrix_v5_1(
        registered_at=normalized["timestamp"], registration=registration
    )
    if normalized != expected:
        raise FactorGovernanceV5Error("Factor v5.1 authority matrix replay mismatch")
    return normalized


__all__ = [
    "AUTHORITY_MATRIX_VERSION",
    "LIFECYCLE_STATES",
    "RESEARCH_DIAGNOSTIC_FACTOR_LANE",
    "build_authority_matrix_v5_1",
    "validate_authority_matrix_v5_1",
]
