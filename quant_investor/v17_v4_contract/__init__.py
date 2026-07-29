"""Offline contracts for the ``myquant.v17.v4`` research successor.

This scaffold distinguishes formal-research publication from selection as the
default research runtime.  It grants neither authority at import time and
permanently grants no execution, broker, order, or trade authority.
"""

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
FORMAL_RESEARCH_PUBLICATION_AUTHORITY = False
RESEARCH_RUNTIME_DEFAULT = False
EXECUTION_AUTHORITY = False
BROKER_AUTHORITY = False
ORDER_AUTHORITY = False
TRADE_AUTHORITY = False

__all__ = [
    "BROKER_AUTHORITY",
    "EXECUTION_AUTHORITY",
    "FORMAL_RESEARCH_PUBLICATION_AUTHORITY",
    "ORDER_AUTHORITY",
    "PROTOCOL_VERSION",
    "RESEARCH_RUNTIME_DEFAULT",
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
