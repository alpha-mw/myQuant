"""Read-only V17 v4 regime evidence adapter for V17 v5 diagnostics.

The current registered V4 regime-evidence artifact is provenance-only: it
contains a role and gross multiplier, but it does not contain a sealed hard
regime state, posterior probabilities, sessions, published_at, or source refs.
This adapter therefore normalizes the exact verified closure and returns an
explicit unavailable status instead of inferring a regime classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Mapping

from quant_investor.v17_v5_contract.validators import (
    V4_COMPATIBILITY_POLICY_BYTE_SHA256,
    V4_SOURCE_GIT_COMMIT,
)
from quant_investor.v17_v5_runtime.v4_compat_reader import (
    V4ClosureNode,
    V4CompatibilityRead,
)

REGIME_EVIDENCE_VERSION: Final = "myquant.v17.v4.regime-evidence.v1"
REGIME_MARKOV_ROLE: Final = "markov_evidence"
REGIME_HARD_STATE_UNAVAILABLE: Final = "REGIME_HARD_STATE_UNAVAILABLE"
_V4_REGIME_V1_LIMITATIONS: Final = (
    "regime_hard_state_absent",
    "regime_posterior_absent",
    "regime_decision_session_absent",
    "regime_effective_session_absent",
    "regime_published_at_absent",
    "regime_source_refs_absent",
)


class V4RegimeAdapterError(ValueError):
    """Raised when a verified V4 read is contradictory for regime adaptation."""

    exit_code = 2


class V4RegimeEvidenceStatus(str, Enum):
    """The only V4 regime adapter status for Sprint 1B."""

    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class NormalizedV4RegimeEvidence:
    """A normalized, read-only view over one exact V4 regime evidence closure."""

    available_at: str
    blockers: tuple[str, ...]
    causal_status: str
    created_at: str
    cutoff: str
    decision_session: None
    effective_session: None
    published_at: None
    regime_artifact_ref: Mapping[str, str]
    regime_state: None
    source_identity: str
    source_role: str
    source_version: str
    state_probabilities: None
    status: V4RegimeEvidenceStatus
    strategy_id: str


def _fail(message: str) -> None:
    raise V4RegimeAdapterError(message)


def _root_node(read: V4CompatibilityRead) -> V4ClosureNode:
    path = read.root_ref.get("relative_path")
    if type(path) is not str:
        _fail("V4 regime root reference path is absent")
    matches = [node for node in read.closure if node.relative_path == path]
    if len(matches) != 1:
        _fail("V4 regime root closure node is absent")
    return matches[0]


def _validate_verified_regime_read(
    read: V4CompatibilityRead,
) -> tuple[dict[str, Any], V4ClosureNode]:
    if not isinstance(read, V4CompatibilityRead):
        _fail("read must be a V4CompatibilityRead value")
    if (
        read.compatibility_policy_byte_sha256 != V4_COMPATIBILITY_POLICY_BYTE_SHA256
        or read.predecessor_git_commit != V4_SOURCE_GIT_COMMIT
    ):
        _fail("V4 compatibility read policy or predecessor identity mismatch")
    if read.terminal_bindings:
        _fail("V4 regime evidence must not expose terminal source bindings")

    document = dict(read.document)
    if document.get("version") != REGIME_EVIDENCE_VERSION:
        _fail("V4 regime evidence version mismatch")
    if document.get("role") != REGIME_MARKOV_ROLE:
        _fail("V4 regime evidence role is not markov_evidence")
    if document.get("status") != "AVAILABLE":
        _fail("V4 regime evidence status is not AVAILABLE")

    node = _root_node(read)
    if node.version != REGIME_EVIDENCE_VERSION:
        _fail("V4 regime root node version mismatch")
    if node.validation_mode != "V4_REGISTERED_JSON":
        _fail("V4 regime root must be a registered JSON artifact")
    if read.documents.get(node.relative_path) != document:
        _fail("V4 regime root document is absent from verified closure")
    expected_root_ref = {
        "artifact_id": str(document.get("evidence_id")),
        "artifact_version": REGIME_EVIDENCE_VERSION,
        "byte_sha256": node.byte_sha256,
        "cutoff": str(document.get("cutoff")),
        "relative_path": node.relative_path,
        "semantic_sha256": node.semantic_sha256,
        "strategy_id": str(document.get("strategy_id")),
    }
    if dict(read.root_ref) != expected_root_ref:
        _fail("V4 regime root reference identity mismatch")
    if document.get("semantic_sha256") != node.semantic_sha256:
        _fail("V4 regime semantic SHA mismatch")
    if len(read.closure) != 1:
        _fail("V4 regime evidence closure must contain only the root artifact")
    return document, node


def adapt_v4_regime_evidence(read: V4CompatibilityRead) -> NormalizedV4RegimeEvidence:
    """Normalize one exact V4 regime closure without inferring a regime state."""

    document, node = _validate_verified_regime_read(read)
    artifact_ref = MappingProxyType(
        {
            "artifact_id": str(document["evidence_id"]),
            "artifact_version": REGIME_EVIDENCE_VERSION,
            "byte_sha256": node.byte_sha256,
            "cutoff": str(document["cutoff"]),
            "relative_path": node.relative_path,
            "semantic_sha256": node.semantic_sha256,
            "strategy_id": str(document["strategy_id"]),
        }
    )
    return NormalizedV4RegimeEvidence(
        available_at=str(document["available_at"]),
        blockers=_V4_REGIME_V1_LIMITATIONS,
        causal_status=REGIME_HARD_STATE_UNAVAILABLE,
        created_at=str(document["created_at"]),
        cutoff=str(document["cutoff"]),
        decision_session=None,
        effective_session=None,
        published_at=None,
        regime_artifact_ref=artifact_ref,
        regime_state=None,
        source_identity=str(document["evidence_id"]),
        source_role=REGIME_MARKOV_ROLE,
        source_version=REGIME_EVIDENCE_VERSION,
        state_probabilities=None,
        status=V4RegimeEvidenceStatus.UNAVAILABLE,
        strategy_id=str(document["strategy_id"]),
    )


__all__ = [
    "NormalizedV4RegimeEvidence",
    "REGIME_EVIDENCE_VERSION",
    "REGIME_HARD_STATE_UNAVAILABLE",
    "REGIME_MARKOV_ROLE",
    "V4RegimeAdapterError",
    "V4RegimeEvidenceStatus",
    "adapt_v4_regime_evidence",
]
