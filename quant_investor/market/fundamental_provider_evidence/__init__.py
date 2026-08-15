"""Stable, offline Fundamental provider-evidence replay boundary.

Only deterministic validation and hardened filesystem capture are public.
Provider acquisition, credentials, HTTP transport, promotion authority, and
network fallbacks are deliberately absent.
"""

from ._codec import canonical_bytes as canonical_provider_json_bytes
from ._comparison import validate_fundamental_comparison_policy
from ._fileset import validate_provider_evidence_fileset_manifest
from ._manifest import (
    is_fundamental_provider_evidence_manifest,
    validate_fundamental_provider_manifest,
)
from ._model import FundamentalProviderEvidenceError
from ._reconciliation import validate_fundamental_reconciliation_receipt
from ._schedule import validate_fundamental_execution_closure
from ._storage import capture_provider_evidence_directory

__all__ = [
    "FundamentalProviderEvidenceError",
    "canonical_provider_json_bytes",
    "capture_provider_evidence_directory",
    "is_fundamental_provider_evidence_manifest",
    "validate_fundamental_comparison_policy",
    "validate_fundamental_execution_closure",
    "validate_fundamental_provider_manifest",
    "validate_fundamental_reconciliation_receipt",
    "validate_provider_evidence_fileset_manifest",
]
