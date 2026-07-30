"""Read-only V17 v4 regime evidence adapter for V17 v5 diagnostics.

The adapter only consumes a :class:`V4CompatibilityRead` returned by the
bounded compatibility reader.  It never scans for latest artifacts, never calls
the V4 producer, and never recomputes the posterior or hard state.  V4 v1
regime evidence remains integrity-checkable but is not conditioning eligible;
only sealed V4 regime-evidence v2 can enter V5 origin-regime binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
import re
from typing import Any, Final, Mapping

from quant_investor.v17_v5_contract.validators import (
    V4_COMPATIBILITY_POLICY_BYTE_SHA256,
    V4_PACKAGE_MANIFEST_SHA256,
    V4_RUNTIME_MANIFEST_SHA256,
    V4_SOURCE_GIT_COMMIT,
)
from quant_investor.v17_v5_runtime.v4_compat_reader import (
    V4ClosureNode,
    V4CompatibilityRead,
)

REGIME_EVIDENCE_V1_VERSION: Final = "myquant.v17.v4.regime-evidence.v1"
REGIME_EVIDENCE_V2_VERSION: Final = "myquant.v17.v4.regime-evidence.v2"
REGIME_MARKOV_ROLE: Final = "markov_evidence"
REGIME_HARD_STATE_UNAVAILABLE: Final = "REGIME_HARD_STATE_UNAVAILABLE"
REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE: Final = "REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE"
REGIME_HARD_STATE_UNKNOWN: Final = "REGIME_HARD_STATE_UNKNOWN"

REQUIRED_INFERENCE_KIND: Final = "FILTERED_CAUSAL"
REQUIRED_PUBLICATION_PHASE: Final = "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION"
REQUIRED_SCOPE_KIND: Final = "FULL_MARKET"
REQUIRED_HARD_STATE_DERIVATION: Final = "SEALED_ARGMAX_POLICY_V1"
V4_PACKAGE_MANIFEST_PATH: Final = (
    "quant_investor/v17_v4_contract/resources/package_manifest.v1.json"
)
V4_RUNTIME_MANIFEST_PATH: Final = (
    "quant_investor/v17_v4_contract/resources/runtime_build_manifest.v1.json"
)
V4_REGIME_INFERENCE_POLICY_REF: Final = {
    "byte_sha256": "006773e24f47f0b7f28d6f7707ff6f570066cb212bd83ebd9566512fda7734ef",
    "relative_path": "resources/regime_inference_policy.v1.json",
    "semantic_sha256": "8abff276d5ed217ad2cb411e26e658ac87877e6f7b03682f502d960a8487c913",
    "version": "myquant.v17.v4.regime-inference-policy.v1",
}
V4_REGIME_MODEL_IMPLEMENTATION_SHA256: Final = (
    "4e90e06eb340438e909b842a4f40e1dec7eb5ff3231e02c087499bda7646cc7a"
)
STATE_ORDER: Final = (
    "趋势上涨",
    "震荡低波",
    "震荡高波",
    "趋势下跌",
    "未知",
)
DECIMAL_ONE: Final = Decimal("1.000000000000")
_DECIMAL12_RE: Final = re.compile(r"^(?:0|1)\.[0-9]{12}$")
_NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
_V4_REGIME_V1_LIMITATIONS: Final = (
    "regime_hard_state_absent",
    "regime_posterior_absent",
    "regime_decision_session_absent",
    "regime_effective_session_absent",
    "regime_published_at_absent",
    "regime_source_refs_absent",
    REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE,
)
_REQUIRED_V2_REFS: Final = (
    "feature_snapshot_ref",
    "model_snapshot_ref",
    "scope_ref",
    "transition_matrix_ref",
)


class V4RegimeAdapterError(ValueError):
    """Raised when a verified V4 read is contradictory for regime adaptation."""

    exit_code = 2


class V4RegimeEvidenceStatus(str, Enum):
    """Read-only adapter status; not a governance or validity conclusion."""

    CONDITIONING_ELIGIBLE = "CONDITIONING_ELIGIBLE"
    CONDITIONING_INELIGIBLE = "CONDITIONING_INELIGIBLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class NormalizedV4RegimeEvidence:
    """A normalized, read-only view over one exact V4 regime evidence closure."""

    available_at: str
    blockers: tuple[str, ...]
    causal_status: str
    calendar_previous_open_session: str | None
    conditioning_eligible: bool
    conditioning_ineligibility_reason: str | None
    coverage_ratio: str | None
    created_at: str
    cutoff: str
    decision_session: str | None
    effective_session: str | None
    evidence_id: str
    hard_state: str | None
    hard_state_derivation: str | None
    inference_kind: str | None
    market_sample_count: int | None
    minimum_market_sample: int | None
    model_implementation_sha256: str | None
    observed_through_session: str | None
    publication_phase: str | None
    published_at: str | None
    regime_artifact_ref: Mapping[str, str]
    regime_state: str | None
    scope_kind: str | None
    smoothing_used: bool | None
    source_commit: str
    source_identity: str
    source_role: str | None
    source_version: str
    state_order: tuple[str, ...]
    state_probabilities: Mapping[str, str] | None
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


def _validate_read_binding(read: V4CompatibilityRead) -> None:
    if not isinstance(read, V4CompatibilityRead):
        _fail("read must be a V4CompatibilityRead value")
    if (
        read.compatibility_policy_byte_sha256 != V4_COMPATIBILITY_POLICY_BYTE_SHA256
        or read.predecessor_git_commit != V4_SOURCE_GIT_COMMIT
    ):
        _fail("V4 compatibility read policy or predecessor identity mismatch")
    if read.predecessor_package_manifest_byte_sha256 != V4_PACKAGE_MANIFEST_SHA256:
        _fail("V4 predecessor package manifest identity mismatch")
    if read.predecessor_runtime_manifest_byte_sha256 != V4_RUNTIME_MANIFEST_SHA256:
        _fail("V4 predecessor runtime manifest identity mismatch")
    if (
        read.predecessor_package_manifest_relative_path != V4_PACKAGE_MANIFEST_PATH
        or read.predecessor_runtime_manifest_relative_path != V4_RUNTIME_MANIFEST_PATH
    ):
        _fail("V4 predecessor manifest path identity mismatch")
    if read.predecessor_protocol_version != "myquant.v17.v4":
        _fail("V4 predecessor protocol identity mismatch")
    if read.terminal_bindings:
        _fail("V4 regime evidence must not expose terminal source bindings")


def _validate_root_ref(
    document: Mapping[str, Any],
    node: V4ClosureNode,
    read: V4CompatibilityRead,
    *,
    version: str,
) -> Mapping[str, str]:
    expected_root_ref = {
        "artifact_id": str(document.get("evidence_id")),
        "artifact_version": version,
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
    return MappingProxyType(expected_root_ref)


def _validate_verified_regime_read(
    read: V4CompatibilityRead,
) -> tuple[dict[str, Any], V4ClosureNode, Mapping[str, str]]:
    _validate_read_binding(read)
    document = dict(read.document)
    version = document.get("version")
    if version not in {REGIME_EVIDENCE_V1_VERSION, REGIME_EVIDENCE_V2_VERSION}:
        _fail("V4 regime evidence version mismatch")
    node = _root_node(read)
    if node.version != version:
        _fail("V4 regime root node version mismatch")
    if node.validation_mode != "V4_REGISTERED_JSON":
        _fail("V4 regime root must be a registered JSON artifact")
    if read.documents.get(node.relative_path) != document:
        _fail("V4 regime root document is absent from verified closure")
    root_ref = _validate_root_ref(document, node, read, version=str(version))
    return document, node, root_ref


def _validate_v1(document: Mapping[str, Any], read: V4CompatibilityRead) -> None:
    if document.get("role") != REGIME_MARKOV_ROLE:
        _fail("V4 regime evidence role is not markov_evidence")
    if document.get("status") != "AVAILABLE":
        _fail("V4 regime evidence status is not AVAILABLE")
    if len(read.closure) != 1:
        _fail("V4 v1 regime evidence closure must contain only the root artifact")


def _decimal12(value: Any, *, label: str) -> Decimal:
    if type(value) is not str or _DECIMAL12_RE.fullmatch(value) is None:
        _fail(f"{label} is not a canonical decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise V4RegimeAdapterError(f"{label} is not finite") from exc
    if not result.is_finite() or result < Decimal("0") or result > Decimal("1"):
        _fail(f"{label} is outside [0, 1]")
    return result


def _validate_probabilities(document: Mapping[str, Any]) -> Mapping[str, str]:
    order = document.get("state_order")
    probabilities = document.get("state_probabilities")
    if order != list(STATE_ORDER):
        _fail("V4 v2 state_order mismatch")
    if type(probabilities) is not dict or set(probabilities) != set(STATE_ORDER):
        _fail("V4 v2 state probability set mismatch")
    total = Decimal("0")
    normalized: dict[str, str] = {}
    for state in STATE_ORDER:
        raw = probabilities[state]
        total += _decimal12(raw, label=f"state_probabilities.{state}")
        normalized[state] = raw
    if total != DECIMAL_ONE:
        _fail("V4 v2 state probabilities do not sum to 1.000000000000")
    return MappingProxyType(normalized)


def _artifact_ref(value: Any, *, label: str) -> Mapping[str, str]:
    if type(value) is not dict:
        _fail(f"{label} is not an artifact ref")
    required = {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
    if set(value) != required or any(type(value[key]) is not str for key in required):
        _fail(f"{label} is not an exact artifact ref")
    return MappingProxyType({key: str(value[key]) for key in sorted(required)})


def _closure_node_for_ref(read: V4CompatibilityRead, ref: Mapping[str, str], *, label: str) -> None:
    matches = [node for node in read.closure if node.relative_path == ref["relative_path"]]
    if len(matches) != 1:
        _fail(f"{label} closure node is absent")
    node = matches[0]
    if (
        node.artifact_id != ref["artifact_id"]
        or node.version != ref["artifact_version"]
        or node.byte_sha256 != ref["byte_sha256"]
        or node.semantic_sha256 != ref["semantic_sha256"]
    ):
        _fail(f"{label} closure node identity mismatch")


def _document_for_ref(
    read: V4CompatibilityRead,
    ref: Mapping[str, str],
    *,
    label: str,
) -> Mapping[str, Any]:
    _closure_node_for_ref(read, ref, label=label)
    document = read.documents.get(ref["relative_path"])
    if type(document) is not dict:
        _fail(f"{label} verified document is absent")
    if (
        document.get("version") != ref["artifact_version"]
        or document.get("strategy_id") != ref["strategy_id"]
        or document.get("cutoff") != ref["cutoff"]
        or document.get("semantic_sha256") != ref["semantic_sha256"]
    ):
        _fail(f"{label} verified document identity mismatch")
    return document


def _sealed_previous_open_session(
    document: Mapping[str, Any],
    read: V4CompatibilityRead,
) -> str:
    feature_ref = _artifact_ref(document.get("feature_snapshot_ref"), label="feature_snapshot_ref")
    feature = _document_for_ref(read, feature_ref, label="feature_snapshot_ref")
    calendar_ref = _artifact_ref(feature.get("calendar_ref"), label="feature.calendar_ref")
    calendar = _document_for_ref(read, calendar_ref, label="feature.calendar_ref")
    sessions = calendar.get("open_sessions")
    observed = document.get("observed_through_session")
    decision = document.get("decision_session")
    if (
        type(sessions) is not list
        or any(type(session) is not str for session in sessions)
        or sessions != sorted(set(sessions))
        or observed not in sessions
        or decision not in sessions
        or sessions.index(decision) != sessions.index(observed) + 1
        or feature.get("observed_through_session") != observed
        or feature.get("effective_session") != decision
    ):
        _fail("V4 v2 sealed calendar does not prove the prior open session")
    return str(observed)


def _validate_v2(
    document: Mapping[str, Any],
    read: V4CompatibilityRead,
) -> tuple[Mapping[str, str], str]:
    if document.get("status") != "AVAILABLE":
        _fail("V4 v2 regime evidence status is not AVAILABLE")
    if document.get("authority") != _NO_AUTHORITY:
        _fail("V4 v2 regime evidence grants authority")
    if document.get("inference_kind") != REQUIRED_INFERENCE_KIND:
        _fail("V4 v2 regime inference_kind mismatch")
    if document.get("smoothing_used") is not False:
        _fail("V4 v2 regime smoothing_used mismatch")
    if document.get("publication_phase") != REQUIRED_PUBLICATION_PHASE:
        _fail("V4 v2 regime publication_phase mismatch")
    if document.get("scope_kind") != REQUIRED_SCOPE_KIND:
        _fail("V4 v2 regime scope_kind mismatch")
    if document.get("hard_state_derivation") != REQUIRED_HARD_STATE_DERIVATION:
        _fail("V4 v2 regime hard_state_derivation mismatch")
    if document.get("no_retroactive_causal_backfill") is not True:
        _fail("V4 v2 regime no-backfill flag mismatch")
    if document.get("same_session_execution_eligible") is not False:
        _fail("V4 v2 regime same-session execution flag mismatch")
    if document.get("shadow_only") is not True:
        _fail("V4 v2 regime shadow_only mismatch")
    if (
        document.get("formal_activation_eligible") is not False
        or document.get("performance_evidence_eligible") is not False
        or document.get("promotion_eligible") is not False
    ):
        _fail("V4 v2 regime grants eligibility")
    hard_state = document.get("hard_state")
    if type(hard_state) is not str or hard_state not in STATE_ORDER:
        _fail("V4 v2 hard state is not in sealed state_order")
    probabilities = _validate_probabilities(document)
    market_sample_count = document.get("market_sample_count")
    minimum_market_sample = document.get("minimum_market_sample")
    if (
        type(market_sample_count) is not int
        or type(minimum_market_sample) is not int
        or market_sample_count < minimum_market_sample
    ):
        _fail("V4 v2 market sample count is below minimum")
    if document.get("model_implementation_sha256") != V4_REGIME_MODEL_IMPLEMENTATION_SHA256:
        _fail("V4 v2 model implementation is not the pinned predecessor source")
    _decimal12(document.get("coverage_ratio"), label="coverage_ratio")
    for field in _REQUIRED_V2_REFS:
        ref = _artifact_ref(document.get(field), label=field)
        _closure_node_for_ref(read, ref, label=field)
    for index, source_ref in enumerate(document.get("source_refs") or ()):
        ref = _artifact_ref(source_ref, label=f"source_refs[{index}]")
        _closure_node_for_ref(read, ref, label=f"source_refs[{index}]")
    policy = document.get("inference_policy_ref")
    if type(policy) is not dict or policy != V4_REGIME_INFERENCE_POLICY_REF:
        _fail("V4 v2 inference policy ref mismatch")
    return probabilities, _sealed_previous_open_session(document, read)


def adapt_v4_regime_evidence(read: V4CompatibilityRead) -> NormalizedV4RegimeEvidence:
    """Normalize one exact V4 regime closure without inferring a regime state."""

    document, _, artifact_ref = _validate_verified_regime_read(read)
    version = str(document["version"])
    if version == REGIME_EVIDENCE_V1_VERSION:
        _validate_v1(document, read)
        return NormalizedV4RegimeEvidence(
            available_at=str(document["available_at"]),
            blockers=_V4_REGIME_V1_LIMITATIONS,
            causal_status=REGIME_HARD_STATE_UNAVAILABLE,
            calendar_previous_open_session=None,
            conditioning_eligible=False,
            conditioning_ineligibility_reason=REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE,
            coverage_ratio=None,
            created_at=str(document["created_at"]),
            cutoff=str(document["cutoff"]),
            decision_session=None,
            effective_session=None,
            evidence_id=str(document["evidence_id"]),
            hard_state=None,
            hard_state_derivation=None,
            inference_kind=None,
            market_sample_count=None,
            minimum_market_sample=None,
            model_implementation_sha256=None,
            observed_through_session=None,
            publication_phase=None,
            published_at=None,
            regime_artifact_ref=artifact_ref,
            regime_state=None,
            scope_kind=None,
            smoothing_used=None,
            source_commit=read.predecessor_git_commit,
            source_identity=str(document["evidence_id"]),
            source_role=REGIME_MARKOV_ROLE,
            source_version=version,
            state_order=(),
            state_probabilities=None,
            status=V4RegimeEvidenceStatus.UNAVAILABLE,
            strategy_id=str(document["strategy_id"]),
        )

    probabilities, previous_open_session = _validate_v2(document, read)
    hard_state = str(document["hard_state"])
    unknown = hard_state == "未知"
    return NormalizedV4RegimeEvidence(
        available_at=str(document["available_at"]),
        blockers=(() if not unknown else (REGIME_HARD_STATE_UNKNOWN,)),
        causal_status="CAUSAL_ORIGIN_REGIME_AVAILABLE",
        calendar_previous_open_session=previous_open_session,
        conditioning_eligible=not unknown,
        conditioning_ineligibility_reason=(REGIME_HARD_STATE_UNKNOWN if unknown else None),
        coverage_ratio=str(document["coverage_ratio"]),
        created_at=str(document["created_at"]),
        cutoff=str(document["cutoff"]),
        decision_session=str(document["decision_session"]),
        effective_session=str(document["effective_session"]),
        evidence_id=str(document["evidence_id"]),
        hard_state=hard_state,
        hard_state_derivation=str(document["hard_state_derivation"]),
        inference_kind=str(document["inference_kind"]),
        market_sample_count=int(document["market_sample_count"]),
        minimum_market_sample=int(document["minimum_market_sample"]),
        model_implementation_sha256=str(document["model_implementation_sha256"]),
        observed_through_session=str(document["observed_through_session"]),
        publication_phase=str(document["publication_phase"]),
        published_at=str(document["published_at"]),
        regime_artifact_ref=artifact_ref,
        regime_state=hard_state,
        scope_kind=str(document["scope_kind"]),
        smoothing_used=False,
        source_commit=read.predecessor_git_commit,
        source_identity=str(document["evidence_id"]),
        source_role=None,
        source_version=version,
        state_order=STATE_ORDER,
        state_probabilities=probabilities,
        status=(
            V4RegimeEvidenceStatus.CONDITIONING_INELIGIBLE
            if unknown
            else V4RegimeEvidenceStatus.CONDITIONING_ELIGIBLE
        ),
        strategy_id=str(document["strategy_id"]),
    )


__all__ = [
    "NormalizedV4RegimeEvidence",
    "REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE",
    "REGIME_EVIDENCE_V1_VERSION",
    "REGIME_EVIDENCE_V2_VERSION",
    "REGIME_HARD_STATE_UNAVAILABLE",
    "REGIME_HARD_STATE_UNKNOWN",
    "REGIME_MARKOV_ROLE",
    "V4RegimeAdapterError",
    "V4RegimeEvidenceStatus",
    "adapt_v4_regime_evidence",
]
