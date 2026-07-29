"""Closed, research-only contracts for ``myquant.v17.v5``."""

from __future__ import annotations

from .canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
    seal_semantic,
    semantic_sha256,
    validate_semantic_sha,
)
from .resources import (
    load_compatibility_policy,
    load_factor_diagnostic_policy,
    load_v4_factor_evidence_adapter_policy,
    verify_package,
    verify_predecessor,
    verify_runtime_build,
)
from .schema_validation import (
    artifact_identity_field,
    load_canonical_artifact,
    validate_artifact,
)

PROTOCOL_VERSION = "myquant.v17.v5"
FORMAL_RESEARCH_PUBLICATION_AUTHORITY = False
RESEARCH_RUNTIME_DEFAULT = False
FORMAL_ACTIVATION_AUTHORITY = False
CANARY_AUTHORITY = False
PROMOTION_AUTHORITY = False
PROVIDER_AUTHORITY = False
LLM_AUTHORITY = False
FACTOR_GOVERNANCE_WRITE_AUTHORITY = False
PORTFOLIO_AUTHORITY = False
SELECTOR_AUTHORITY = False
EXECUTION_AUTHORITY = False
BROKER_AUTHORITY = False
ORDER_AUTHORITY = False
TRADE_AUTHORITY = False

__all__ = [
    "BROKER_AUTHORITY",
    "CANARY_AUTHORITY",
    "EXECUTION_AUTHORITY",
    "FACTOR_GOVERNANCE_WRITE_AUTHORITY",
    "FORMAL_ACTIVATION_AUTHORITY",
    "FORMAL_RESEARCH_PUBLICATION_AUTHORITY",
    "LLM_AUTHORITY",
    "ORDER_AUTHORITY",
    "PORTFOLIO_AUTHORITY",
    "PROMOTION_AUTHORITY",
    "PROTOCOL_VERSION",
    "PROVIDER_AUTHORITY",
    "RESEARCH_RUNTIME_DEFAULT",
    "SELECTOR_AUTHORITY",
    "TRADE_AUTHORITY",
    "artifact_identity_field",
    "canonical_bytes",
    "canonical_resource_bytes",
    "load_canonical_artifact",
    "load_canonical_resource",
    "load_compatibility_policy",
    "load_factor_diagnostic_policy",
    "load_v4_factor_evidence_adapter_policy",
    "seal_semantic",
    "semantic_sha256",
    "validate_artifact",
    "validate_semantic_sha",
    "verify_package",
    "verify_predecessor",
    "verify_runtime_build",
]
