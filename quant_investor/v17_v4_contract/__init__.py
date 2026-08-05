"""Offline research-only contracts for ``myquant.v17.v4``."""

from __future__ import annotations

from .canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    semantic_sha256,
)
from .resources import (
    verify_forward_runtime_sources,
    verify_package,
    verify_runtime_build,
)
from .schema_validation import (
    artifact_identity_field,
    load_canonical_artifact,
    validate_artifact,
)

PROTOCOL_VERSION = "myquant.v17.v4"
RESEARCH_ONLY = True
MAINLINE_AUTHORITY = False
PRODUCTION_AUTHORITY = False
EXECUTION_AUTHORITY = False
BROKER_AUTHORITY = False
ORDER_AUTHORITY = False
TRADE_AUTHORITY = False

__all__ = [
    "BROKER_AUTHORITY",
    "EXECUTION_AUTHORITY",
    "MAINLINE_AUTHORITY",
    "ORDER_AUTHORITY",
    "PROTOCOL_VERSION",
    "PRODUCTION_AUTHORITY",
    "RESEARCH_ONLY",
    "TRADE_AUTHORITY",
    "artifact_identity_field",
    "canonical_bytes",
    "canonical_resource_bytes",
    "load_canonical_artifact",
    "seal_semantic",
    "semantic_sha256",
    "validate_artifact",
    "verify_forward_runtime_sources",
    "verify_package",
    "verify_runtime_build",
]
