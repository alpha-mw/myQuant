"""Independent contracts for the research-only myQuant v17 protocol v3.

The package is deliberately a pure, offline contract surface.  It grants
formal-research publication authority only through a valid ACTIVE receipt and
never grants broker, order, trade, execution, or production-default authority.
It does not import either the v17 v2 contract or runtime packages.
"""

from __future__ import annotations

from .canonical import canonical_bytes, semantic_sha256
from .namespace import (
    FORMAL_RESEARCH_RESULTS_ROOT,
    RUNS_ROOT,
    SHADOW_RESULTS_ROOT,
    SOURCES_ROOT,
)
from .resources import verify_package
from .references import build_artifact_ref
from .schema_validation import (
    artifact_identity_field,
    load_canonical_artifact,
    validate_artifact,
)

PROTOCOL_VERSION = "myquant.v17.v3"
FORMAL_RESEARCH_PUBLICATION_AUTHORITY = False
EXECUTION_AUTHORITY = False
PRODUCTION_DEFAULT = False
BROKER_AUTHORITY = False
ORDER_AUTHORITY = False
TRADE_AUTHORITY = False

__all__ = [
    "BROKER_AUTHORITY",
    "EXECUTION_AUTHORITY",
    "FORMAL_RESEARCH_PUBLICATION_AUTHORITY",
    "FORMAL_RESEARCH_RESULTS_ROOT",
    "ORDER_AUTHORITY",
    "PRODUCTION_DEFAULT",
    "PROTOCOL_VERSION",
    "RUNS_ROOT",
    "SHADOW_RESULTS_ROOT",
    "SOURCES_ROOT",
    "TRADE_AUTHORITY",
    "artifact_identity_field",
    "build_artifact_ref",
    "canonical_bytes",
    "load_canonical_artifact",
    "semantic_sha256",
    "validate_artifact",
    "verify_package",
]
