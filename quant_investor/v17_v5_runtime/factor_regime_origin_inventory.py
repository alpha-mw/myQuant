"""Pure origin-regime binding inventory for V17 v5 Sprint 1B.

This module accepts only caller-supplied, already verified origin and regime
evidence.  It does not read files, scan directories, call providers, write V4,
or produce governance actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import re
from typing import Any, Final, Mapping, Sequence

from quant_investor.v17_v5_contract.canonical import (
    canonical_bytes,
    seal_semantic,
    validate_semantic_sha,
)
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_git_commit,
    require_identifier,
    require_relative_path,
    require_sha256,
)
from quant_investor.v17_v5_contract.validators import (
    ArtifactContractError,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    NO_AUTHORITY,
    V4_SOURCE_GIT_COMMIT,
    validate_v3_excluded_regime_origin_row,
)

PROTOCOL_VERSION: Final = "myquant.v17.v5"
FACTOR_REGIME_ORIGIN_INVENTORY_VERSION: Final = "myquant.v17.v5.factor-regime-origin-inventory.v3"
HORIZON_SESSIONS: Final = 20
OUTPUT_SCALE: Final = Decimal("0.000000000001")
REGIME_EVIDENCE_V3_VERSION: Final = "myquant.v17.v4.regime-evidence.v3"
POLICY_V3_VERSION: Final = "myquant.v17.v5.factor-regime-diagnostic-policy.v3"
POLICY_V3_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/factor_regime_diagnostic_policy.v3.json"
)
REQUIRED_PUBLICATION_PHASE: Final = "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION"
REQUIRED_INFERENCE_KIND: Final = "FILTERED_CAUSAL"
REQUIRED_HARD_STATE_DERIVATION: Final = "SEALED_ARGMAX_POLICY_V1"
REQUIRED_SCOPE_KIND: Final = "FULL_MARKET"
UNKNOWN_REGIME_STATE: Final = "未知"
_SESSION_RE: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", re.ASCII)
_UTC_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_DECIMAL_RE: Final = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
    re.ASCII,
)
_FORBIDDEN_KEY_PARTS: Final = (
    "factor_weight",
    "recommended_weight",
    "target_weight",
    "production_weight",
    "portfolio_weight",
    "lifecycle_action",
    "validity",
)
_FORBIDDEN_KEYS: Final = {
    "tier",
    "promotion",
    "production_apply",
    "buy_signal",
    "sell_signal",
}


class FactorRegimeOriginInventoryError(ValueError):
    """Raised when a regime-origin inventory input is malformed."""

    exit_code = 2


@dataclass(frozen=True)
class ContentArtifactRef:
    """A pathless V5 content reference or explicit immutable V4 reference."""

    artifact_id: str
    version: str
    byte_sha256: str
    semantic_sha256: str
    cutoff: str
    strategy_id: str
    relative_path: str | None = None


@dataclass(frozen=True)
class RegimeEvidenceSnapshot:
    """Normalized read-only regime evidence at the origin decision point."""

    regime_artifact_ref: ContentArtifactRef
    strategy_id: str
    cutoff: str
    available_at: str
    published_at: str
    regime_state: str
    source_version: str
    decision_session: str | None = None
    effective_session: str | None = None
    state_probabilities: Mapping[str, str] | None = None
    observed_through_session: str | None = None
    calendar_previous_open_session: str | None = None
    publication_phase: str | None = None
    inference_kind: str | None = None
    smoothing_used: bool | None = None
    hard_state_derivation: str | None = None
    scope_kind: str | None = None
    no_retroactive_causal_backfill: bool | None = None
    source_commit: str | None = None
    state_order: Sequence[str] | None = None
    checkpoint_ref: ContentArtifactRef | None = None
    finalized: bool = False
    continuity_kind: str | None = None
    segment_id: str | None = None
    segment_index: int | None = None
    segment_position: int | None = None
    transition_commitment_sha256: str | None = None
    chain_digest_sha256: str | None = None
    segment_accumulator_sha256: str | None = None


@dataclass(frozen=True)
class FactorRegimeOriginInput:
    """One already verified factor origin bound to one origin-time regime."""

    origin_id: str
    strategy_id: str
    factor_name: str
    factor_implementation_sha256: str
    decision_session: str
    origin_cutoff: str
    label_end_session: str
    label_horizon_sessions: int
    label_origin_session: str
    rank_ic: str | None
    eligible_symbol_count: int
    comparable_symbol_count: int
    coverage: str
    factor_observation_ref: ContentArtifactRef
    matured_label_ref: ContentArtifactRef
    factor_evidence_ref: ContentArtifactRef
    observation_run_ref: ContentArtifactRef
    request_ref: ContentArtifactRef
    source_locator_ref: ContentArtifactRef
    regime_evidence: RegimeEvidenceSnapshot


def _fail(message: str) -> None:
    raise FactorRegimeOriginInventoryError(message)


def _timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        _fail(f"{label} must be a second-precision UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise FactorRegimeOriginInventoryError(f"{label} is not a valid UTC timestamp") from exc


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str or _SESSION_RE.fullmatch(value) is None:
        _fail(f"{label} must be an ISO session date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise FactorRegimeOriginInventoryError(f"{label} is not a valid session date") from exc
    return value


def _decimal(value: Any, *, label: str, minimum: Decimal, maximum: Decimal) -> str:
    if type(value) is not str or _DECIMAL_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical finite decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise FactorRegimeOriginInventoryError(f"{label} is not a finite decimal") from exc
    if not parsed.is_finite() or parsed < minimum or parsed > maximum:
        _fail(f"{label} is out of range")
    if parsed.is_zero() and value.startswith("-"):
        _fail(f"{label} is not canonical")
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        rendered = parsed.quantize(OUTPUT_SCALE, rounding=ROUND_HALF_EVEN)
    if rendered.is_zero():
        rendered = abs(rendered)
    return format(rendered, ".12f")


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or any(ord(character) < 32 for character in value):
        _fail(f"{label} must be a nonempty control-free string")
    return value


def _artifact_ref(ref: ContentArtifactRef, *, label: str, v4: bool) -> dict[str, Any]:
    if not isinstance(ref, ContentArtifactRef):
        _fail(f"{label} must be ContentArtifactRef")
    try:
        artifact_id = require_identifier(ref.artifact_id, label=f"{label}.artifact_id")
        version = require_identifier(ref.version, label=f"{label}.version")
        byte_sha = require_sha256(ref.byte_sha256, label=f"{label}.byte_sha256")
        semantic_sha = require_sha256(ref.semantic_sha256, label=f"{label}.semantic_sha256")
        strategy_id = require_identifier(ref.strategy_id, label=f"{label}.strategy_id")
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    cutoff = _timestamp(ref.cutoff, label=f"{label}.cutoff").strftime("%Y-%m-%dT%H:%M:%SZ")
    document: dict[str, Any] = {
        "artifact_id": artifact_id,
        "byte_sha256": byte_sha,
        "cutoff": cutoff,
        "semantic_sha256": semantic_sha,
        "strategy_id": strategy_id,
        "version": version,
    }
    if v4:
        if ref.relative_path is None:
            _fail(f"{label}.relative_path is required for V4 refs")
        try:
            document["relative_path"] = require_relative_path(
                ref.relative_path,
                label=f"{label}.relative_path",
            )
        except IdentityContractError as exc:
            raise FactorRegimeOriginInventoryError(str(exc)) from exc
    elif ref.relative_path is not None:
        _fail(f"{label}.relative_path is not allowed for V5 content refs")
    return document


def _policy_ref(policy_ref: Mapping[str, Any]) -> dict[str, Any]:
    if type(policy_ref) is not dict or set(policy_ref) != {
        "artifact_id",
        "byte_sha256",
        "relative_path",
        "semantic_sha256",
        "version",
    }:
        _fail("policy_ref must be an object")
    try:
        document = {
            "artifact_id": require_identifier(
                policy_ref["artifact_id"], label="policy artifact_id"
            ),
            "byte_sha256": require_sha256(policy_ref["byte_sha256"], label="policy byte_sha256"),
            "semantic_sha256": require_sha256(
                policy_ref["semantic_sha256"],
                label="policy semantic_sha256",
            ),
            "version": require_identifier(policy_ref["version"], label="policy version"),
        }
    except (KeyError, IdentityContractError) as exc:
        raise FactorRegimeOriginInventoryError("policy_ref is invalid") from exc
    try:
        document["relative_path"] = require_relative_path(
            policy_ref["relative_path"],
            label="policy relative_path",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    expected_current = {
        "artifact_id": FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
        "byte_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
        "relative_path": FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
        "semantic_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
        "version": FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    }
    if document == expected_current and document["version"] == POLICY_V3_VERSION:
        return document
    if document["version"] in {
        "myquant.v17.v5.factor-regime-diagnostic-policy.v1",
        "myquant.v17.v5.factor-regime-diagnostic-policy.v2",
    }:
        _fail("Sprint 1E-0B regime diagnostics must bind policy v3")
    if document["version"] != POLICY_V3_VERSION or document["relative_path"] != POLICY_V3_PATH:
        _fail("policy_ref does not bind the sealed Sprint 1E-0B policy v3")
    return document


def _probabilities(
    values: Mapping[str, str] | None,
    *,
    state_order: Sequence[str] | None = None,
) -> list[dict[str, str]] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping) or not values:
        _fail("state_probabilities must be a nonempty mapping")
    ordered_states: list[str]
    if state_order is None:
        ordered_states = sorted(values)
    else:
        if isinstance(state_order, (str, bytes)) or not isinstance(state_order, Sequence):
            _fail("state_order must be a sequence")
        ordered_states = [_text(value, label="state_order item") for value in state_order]
        if len(ordered_states) != len(set(ordered_states)):
            _fail("state_order contains duplicates")
        if set(values) != set(ordered_states):
            _fail("state probabilities must exactly match state_order")
    result: dict[str, str] = {}
    total = Decimal(0)
    for state in ordered_states:
        probability = values[state]
        name = _text(state, label="state probability regime_state")
        value = _decimal(
            probability,
            label=f"state_probabilities[{name}]",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        result[name] = value
        total += Decimal(value)
    if total.quantize(OUTPUT_SCALE, rounding=ROUND_HALF_EVEN) != Decimal("1.000000000000"):
        _fail("state probabilities must sum exactly to one")
    return [
        {
            "probability": result[key],
            "regime_state": key,
        }
        for key in ordered_states
    ]


def _forbidden_key_scan(value: Any, *, path: str = "$") -> None:
    if path == "$.authority":
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is str:
                lowered = key.lower()
                if lowered in _FORBIDDEN_KEYS or any(
                    part in lowered for part in _FORBIDDEN_KEY_PARTS
                ):
                    _fail(f"forbidden governance or weight field at {path}.{key}")
            _forbidden_key_scan(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_key_scan(child, path=f"{path}[{index}]")


def _origin_row(
    origin: FactorRegimeOriginInput,
    *,
    strategy_id: str,
    factor_name: str,
    factor_implementation_sha256: str,
    horizon_sessions: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(origin, FactorRegimeOriginInput):
        _fail("origin_rows must contain FactorRegimeOriginInput values")
    try:
        origin_id = require_identifier(origin.origin_id, label="origin_id")
        row_strategy = require_identifier(origin.strategy_id, label="origin.strategy_id")
        row_factor = require_identifier(origin.factor_name, label="origin.factor_name")
        row_implementation = require_sha256(
            origin.factor_implementation_sha256,
            label="origin.factor_implementation_sha256",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    if (
        row_strategy != strategy_id
        or row_factor != factor_name
        or row_implementation != factor_implementation_sha256
    ):
        _fail("origin row stratum mismatch")
    if (
        origin.label_horizon_sessions != horizon_sessions
        or origin.label_horizon_sessions != HORIZON_SESSIONS
    ):
        _fail("label horizon must equal 20 sessions")
    decision_session = _session(origin.decision_session, label=f"{origin_id}.decision_session")
    label_origin_session = _session(
        origin.label_origin_session,
        label=f"{origin_id}.label_origin_session",
    )
    label_end_session = _session(origin.label_end_session, label=f"{origin_id}.label_end_session")
    if label_origin_session != decision_session:
        _fail("label origin must equal factor observation origin")
    origin_cutoff = _timestamp(origin.origin_cutoff, label=f"{origin_id}.origin_cutoff")
    if type(origin.eligible_symbol_count) is not int or origin.eligible_symbol_count < 0:
        _fail("eligible_symbol_count is invalid")
    if (
        type(origin.comparable_symbol_count) is not int
        or origin.comparable_symbol_count < 0
        or origin.comparable_symbol_count > origin.eligible_symbol_count
    ):
        _fail("comparable_symbol_count is invalid")
    coverage = _decimal(
        origin.coverage,
        label=f"{origin_id}.coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if origin.eligible_symbol_count == 0:
        if origin.comparable_symbol_count != 0 or Decimal(coverage) != 0:
            _fail("zero eligible symbols require zero coverage")
    elif (Decimal(origin.comparable_symbol_count) / Decimal(origin.eligible_symbol_count)).quantize(
        OUTPUT_SCALE, rounding=ROUND_HALF_EVEN
    ) != Decimal(coverage):
        _fail("coverage must equal comparable_symbol_count / eligible_symbol_count exactly")
    rank_ic = None
    if origin.rank_ic is not None:
        rank_ic = _decimal(
            origin.rank_ic,
            label=f"{origin_id}.rank_ic",
            minimum=Decimal("-1"),
            maximum=Decimal("1"),
        )
    factor_observation_ref = _artifact_ref(
        origin.factor_observation_ref,
        label=f"{origin_id}.factor_observation_ref",
        v4=True,
    )
    label_ref = _artifact_ref(
        origin.matured_label_ref,
        label=f"{origin_id}.matured_label_ref",
        v4=True,
    )
    factor_evidence_ref = _artifact_ref(
        origin.factor_evidence_ref,
        label=f"{origin_id}.factor_evidence_ref",
        v4=True,
    )
    observation_run_ref = _artifact_ref(
        origin.observation_run_ref,
        label=f"{origin_id}.observation_run_ref",
        v4=True,
    )
    request_ref = _artifact_ref(
        origin.request_ref,
        label=f"{origin_id}.request_ref",
        v4=True,
    )
    source_locator_ref = _artifact_ref(
        origin.source_locator_ref,
        label=f"{origin_id}.source_locator_ref",
        v4=True,
    )
    for label, ref in (
        ("factor_observation_ref", factor_observation_ref),
        ("matured_label_ref", label_ref),
        ("factor_evidence_ref", factor_evidence_ref),
        ("observation_run_ref", observation_run_ref),
        ("request_ref", request_ref),
        ("source_locator_ref", source_locator_ref),
    ):
        if ref["strategy_id"] != strategy_id:
            _fail(f"{label} strategy_id mismatch")
    regime = origin.regime_evidence
    if not isinstance(regime, RegimeEvidenceSnapshot):
        _fail("regime_evidence must be RegimeEvidenceSnapshot")
    regime_ref = _artifact_ref(
        regime.regime_artifact_ref,
        label=f"{origin_id}.regime_artifact_ref",
        v4=True,
    )
    try:
        regime_strategy = require_identifier(regime.strategy_id, label="regime.strategy_id")
        regime_source_version = require_identifier(
            regime.source_version,
            label="regime.source_version",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    if regime_strategy != strategy_id or regime_ref["strategy_id"] != strategy_id:
        _fail("strategy_id mismatch")
    if regime_source_version != regime_ref["version"]:
        _fail("regime source version does not match artifact ref")
    if regime_source_version != REGIME_EVIDENCE_V3_VERSION:
        _fail("regime source version must be myquant.v17.v4.regime-evidence.v3")
    if regime.finalized is not True or regime.checkpoint_ref is None:
        _fail("REGIME_EVIDENCE_V3_NOT_FINALIZED")
    checkpoint_ref = _artifact_ref(
        regime.checkpoint_ref,
        label=f"{origin_id}.regime.checkpoint_ref",
        v4=True,
    )
    if (
        checkpoint_ref["strategy_id"] != strategy_id
        or checkpoint_ref["version"] != "myquant.v17.v4.regime-state-checkpoint.v1"
    ):
        _fail("regime checkpoint ref mismatch")
    regime_cutoff = _timestamp(regime.cutoff, label=f"{origin_id}.regime.cutoff")
    if regime_ref["cutoff"] != regime.cutoff:
        _fail("regime cutoff does not match artifact ref")
    regime_available = _timestamp(regime.available_at, label=f"{origin_id}.regime.available_at")
    regime_published = _timestamp(regime.published_at, label=f"{origin_id}.regime.published_at")
    if regime_available > origin_cutoff:
        _fail("regime available_at is later than origin cutoff")
    if regime_published > origin_cutoff:
        _fail("regime published_at is later than origin cutoff")
    if regime_cutoff > origin_cutoff:
        _fail("regime cutoff is later than origin cutoff")
    regime_decision_session = _session(
        regime.decision_session,
        label=f"{origin_id}.regime.decision_session",
    )
    regime_effective_session = _session(
        regime.effective_session,
        label=f"{origin_id}.regime.effective_session",
    )
    observed_through_session = _session(
        regime.observed_through_session,
        label=f"{origin_id}.regime.observed_through_session",
    )
    calendar_previous_open_session = _session(
        regime.calendar_previous_open_session,
        label=f"{origin_id}.regime.calendar_previous_open_session",
    )
    if regime_decision_session != decision_session:
        _fail("regime decision_session must equal factor origin")
    if regime_effective_session != decision_session:
        _fail("regime effective_session must equal factor origin")
    if observed_through_session != calendar_previous_open_session:
        _fail("regime observed_through_session must equal sealed previous open session")
    if calendar_previous_open_session >= decision_session:
        _fail("sealed previous open session must precede factor origin")
    if regime.publication_phase != REQUIRED_PUBLICATION_PHASE:
        _fail("regime publication_phase is not conditioning eligible")
    if regime.inference_kind != REQUIRED_INFERENCE_KIND:
        _fail("regime inference_kind is not FILTERED_CAUSAL")
    if regime.smoothing_used is not False:
        _fail("regime smoothing_used must be false")
    if regime.hard_state_derivation != REQUIRED_HARD_STATE_DERIVATION:
        _fail("regime hard_state_derivation is not SEALED_ARGMAX_POLICY_V1")
    if regime.scope_kind != REQUIRED_SCOPE_KIND:
        _fail("regime scope_kind is not FULL_MARKET")
    if regime.no_retroactive_causal_backfill is not True:
        _fail("regime no-backfill flag is absent")
    try:
        source_commit = require_git_commit(
            regime.source_commit,
            label=f"{origin_id}.regime.source_commit",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    if source_commit != V4_SOURCE_GIT_COMMIT:
        _fail("regime source commit does not match the pinned V4 predecessor")
    continuity_kind = _text(
        regime.continuity_kind,
        label=f"{origin_id}.regime.continuity_kind",
    )
    if continuity_kind not in {"GENESIS", "RECOVERY", "CONTIGUOUS", "ROLLOVER"}:
        _fail("regime continuity kind is invalid")
    if type(regime.segment_index) is not int or regime.segment_index < 0:
        _fail("regime segment_index is invalid")
    if (
        type(regime.segment_position) is not int
        or regime.segment_position < 0
        or regime.segment_position > 63
    ):
        _fail("regime segment_position is invalid")
    if (
        continuity_kind in {"GENESIS", "RECOVERY", "ROLLOVER"} and regime.segment_position != 0
    ) or (continuity_kind == "CONTIGUOUS" and regime.segment_position == 0):
        _fail("regime continuity and segment position mismatch")
    try:
        segment_id = require_sha256(regime.segment_id, label="regime.segment_id")
        transition_commitment = require_sha256(
            regime.transition_commitment_sha256,
            label="regime.transition_commitment_sha256",
        )
        chain_digest = require_sha256(
            regime.chain_digest_sha256,
            label="regime.chain_digest_sha256",
        )
        segment_accumulator = require_sha256(
            regime.segment_accumulator_sha256,
            label="regime.segment_accumulator_sha256",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    regime_state = _text(regime.regime_state, label=f"{origin_id}.regime_state")
    state_probabilities = _probabilities(regime.state_probabilities, state_order=regime.state_order)
    if state_probabilities is None:
        _fail("V4 regime-evidence.v3 must carry sealed state probabilities")
    if regime_state not in {row["regime_state"] for row in state_probabilities}:
        _fail("posterior does not contain the sealed hard regime state")
    # The V4 producer sealed the native state and tie-break.  V5 verifies the
    # exact state set and preserves the sealed value; it must not rerun argmax.
    base_row = {
        "decision_session": decision_session,
        "factor_name": factor_name,
        "origin_id": origin_id,
        "regime_checkpoint_ref": checkpoint_ref,
        "regime_continuity_kind": continuity_kind,
        "regime_evidence_ref": regime_ref,
        "regime_finalized": True,
        "regime_state": regime_state,
        "row_limitation_codes": [],
    }
    limitations: list[str] = []
    if continuity_kind == "GENESIS":
        limitations.append("REGIME_CONTINUITY_GENESIS")
    elif continuity_kind == "RECOVERY":
        limitations.append("REGIME_CONTINUITY_RECOVERY")
    if regime_state == UNKNOWN_REGIME_STATE:
        limitations.append("REGIME_HARD_STATE_UNKNOWN")
    if limitations:
        excluded = dict(base_row)
        excluded["row_limitation_codes"] = sorted(limitations)
        excluded["row_identity_sha256"] = hashlib.sha256(canonical_bytes(excluded)).hexdigest()
        return None, excluded
    if continuity_kind not in {"CONTIGUOUS", "ROLLOVER"}:
        _fail("regime continuity is not conditioning eligible")
    row = {
        "comparable_symbol_count": origin.comparable_symbol_count,
        "coverage": coverage,
        "decision_session": decision_session,
        "eligible_symbol_count": origin.eligible_symbol_count,
        "factor_evidence_ref": factor_evidence_ref,
        "factor_name": factor_name,
        "factor_observation_ref": factor_observation_ref,
        "label_end_session": label_end_session,
        "label_horizon_sessions": HORIZON_SESSIONS,
        "matured_label_ref": label_ref,
        "observation_run_ref": observation_run_ref,
        "origin_cutoff": origin.origin_cutoff,
        "origin_id": origin_id,
        "rank_ic": rank_ic,
        "request_ref": request_ref,
        "regime_available_at": regime.available_at,
        "regime_chain_digest_sha256": chain_digest,
        "regime_checkpoint_ref": checkpoint_ref,
        "regime_continuity_kind": continuity_kind,
        "regime_decision_session": regime_decision_session,
        "regime_effective_session": regime_effective_session,
        "regime_evidence_ref": regime_ref,
        "regime_finalized": True,
        "regime_hard_state_derivation": regime.hard_state_derivation,
        "regime_inference_kind": regime.inference_kind,
        "regime_no_retroactive_causal_backfill": regime.no_retroactive_causal_backfill,
        "regime_observed_through_session": observed_through_session,
        "regime_publication_phase": regime.publication_phase,
        "regime_published_at": regime.published_at,
        "regime_scope_kind": regime.scope_kind,
        "regime_segment_accumulator_sha256": segment_accumulator,
        "regime_segment_id": segment_id,
        "regime_segment_index": regime.segment_index,
        "regime_segment_position": regime.segment_position,
        "regime_smoothing_used": regime.smoothing_used,
        "regime_source_version": regime_source_version,
        "regime_state": regime_state,
        "regime_transition_commitment_sha256": transition_commitment,
        "regime_source_commit": source_commit,
        "source_locator_ref": source_locator_ref,
        "state_probabilities": state_probabilities,
    }
    row_identity = hashlib.sha256(canonical_bytes(row)).hexdigest()
    row["row_identity_sha256"] = row_identity
    return row, None


def _validate_inventory(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = validate_semantic_sha(document)
        require_identifier(payload["inventory_id"], label="inventory_id")
        require_identifier(payload["strategy_id"], label="strategy_id")
        require_identifier(payload["factor_name"], label="factor_name")
        require_sha256(
            payload["factor_implementation_sha256"],
            label="factor_implementation_sha256",
        )
        _timestamp(payload["cutoff"], label="cutoff")
        _timestamp(payload["created_at"], label="created_at")
    except Exception as exc:
        raise FactorRegimeOriginInventoryError("factor regime origin inventory is invalid") from exc
    if payload.get("version") != FACTOR_REGIME_ORIGIN_INVENTORY_VERSION:
        _fail("factor regime origin inventory version mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        _fail("factor regime origin inventory protocol mismatch")
    if payload.get("authority") != NO_AUTHORITY:
        _fail("factor regime origin inventory grants authority")
    _forbidden_key_scan(payload)
    rows = payload.get("origin_rows")
    if not isinstance(rows, list):
        _fail("origin_rows must be a list")
    if payload.get("origin_count") != len(rows):
        _fail("origin_count mismatch")
    excluded_rows = payload.get("excluded_origin_rows", [])
    if not isinstance(excluded_rows, list):
        _fail("excluded_origin_rows must be a list")
    if payload.get("excluded_origin_count", 0) != len(excluded_rows):
        _fail("excluded_origin_count mismatch")
    limitation_codes = payload.get("limitation_codes", [])
    if not isinstance(limitation_codes, list) or limitation_codes != sorted(set(limitation_codes)):
        _fail("limitation_codes must be canonical")
    expected_limitations = sorted(
        {code for row in excluded_rows for code in row.get("row_limitation_codes", [])}
    )
    if limitation_codes != expected_limitations:
        _fail("excluded-origin limitation codes do not close")
    expected_order = sorted(
        rows,
        key=lambda row: (
            row["decision_session"],
            row["factor_name"],
            row["regime_state"],
            row["factor_observation_ref"]["artifact_id"],
            row["matured_label_ref"]["artifact_id"],
            row["regime_evidence_ref"]["artifact_id"],
        ),
    )
    if rows != expected_order:
        _fail("origin rows are not canonical")
    seen_origins: set[tuple[str, str]] = set()
    seen_origin_ids: set[str] = set()
    regime_counts: dict[str, int] = {}
    for row in [*rows, *excluded_rows]:
        origin_key = (row["decision_session"], row["factor_name"])
        if origin_key in seen_origins or row["origin_id"] in seen_origin_ids:
            _fail("duplicate origin")
        seen_origins.add(origin_key)
        seen_origin_ids.add(row["origin_id"])
        expected_row_identity = dict(row)
        observed = expected_row_identity.pop("row_identity_sha256")
        if hashlib.sha256(canonical_bytes(expected_row_identity)).hexdigest() != observed:
            _fail("origin row identity mismatch")
        if row.get("regime_finalized") is not True:
            _fail("REGIME_EVIDENCE_V3_NOT_FINALIZED")
        if row in rows and row.get("regime_continuity_kind") not in {
            "CONTIGUOUS",
            "ROLLOVER",
        }:
            _fail("conditioning inventory contains an ineligible continuity kind")
        if row in excluded_rows:
            try:
                validate_v3_excluded_regime_origin_row(row)
            except ArtifactContractError as exc:
                raise FactorRegimeOriginInventoryError(str(exc)) from exc
    for row in rows:
        regime_counts[row["regime_state"]] = regime_counts.get(row["regime_state"], 0) + 1
    if excluded_rows != sorted(
        excluded_rows,
        key=lambda row: (
            row["decision_session"],
            row["factor_name"],
            row["regime_state"],
            row["regime_evidence_ref"]["artifact_id"],
        ),
    ):
        _fail("excluded origin rows are not canonical")
    if payload.get("regime_counts") != [
        {
            "origin_count": regime_counts[key],
            "regime_state": key,
        }
        for key in sorted(regime_counts)
    ]:
        _fail("regime_counts mismatch")
    identity_material = dict(payload)
    identity_material.pop("inventory_id")
    identity_material.pop("semantic_sha256")
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    if payload["inventory_id"] != f"factor-regime-origin-inventory-{identity[:32]}":
        _fail("factor regime origin inventory identity mismatch")
    return payload


def build_factor_regime_origin_inventory(
    *,
    strategy_id: str,
    factor_name: str,
    factor_implementation_sha256: str,
    policy_ref: Mapping[str, Any],
    cutoff: str,
    created_at: str,
    origin_rows: Sequence[FactorRegimeOriginInput],
    horizon_sessions: int = HORIZON_SESSIONS,
) -> dict[str, Any]:
    """Build a deterministic in-memory origin-regime inventory artifact."""

    try:
        subject_strategy = require_identifier(strategy_id, label="strategy_id")
        subject_factor = require_identifier(factor_name, label="factor_name")
        implementation_sha = require_sha256(
            factor_implementation_sha256,
            label="factor_implementation_sha256",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    if type(horizon_sessions) is not int or horizon_sessions != HORIZON_SESSIONS:
        _fail("horizon_sessions must equal 20")
    cutoff_dt = _timestamp(cutoff, label="cutoff")
    created_dt = _timestamp(created_at, label="created_at")
    if created_dt < cutoff_dt:
        _fail("created_at must not precede cutoff")
    if isinstance(origin_rows, (str, bytes)) or not isinstance(origin_rows, Sequence):
        _fail("origin_rows must be a sequence")
    rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for origin in origin_rows:
        row, excluded = _origin_row(
            origin,
            strategy_id=subject_strategy,
            factor_name=subject_factor,
            factor_implementation_sha256=implementation_sha,
            horizon_sessions=horizon_sessions,
        )
        if row is not None:
            rows.append(row)
        if excluded is not None:
            excluded_rows.append(excluded)
    rows = sorted(
        rows,
        key=lambda row: (
            row["decision_session"],
            row["factor_name"],
            row["regime_state"],
            row["factor_observation_ref"]["artifact_id"],
            row["matured_label_ref"]["artifact_id"],
            row["regime_evidence_ref"]["artifact_id"],
        ),
    )
    excluded_rows = sorted(
        excluded_rows,
        key=lambda row: (
            row["decision_session"],
            row["factor_name"],
            row["regime_state"],
            row["regime_evidence_ref"]["artifact_id"],
        ),
    )
    regime_counts: dict[str, int] = {}
    for row in rows:
        regime_counts[row["regime_state"]] = regime_counts.get(row["regime_state"], 0) + 1
    limitation_codes = sorted(
        {code for row in excluded_rows for code in row["row_limitation_codes"]}
    )
    document = {
        "authority": dict(NO_AUTHORITY),
        "created_at": created_at,
        "cutoff": cutoff,
        "excluded_origin_count": len(excluded_rows),
        "excluded_origin_rows": excluded_rows,
        "factor_implementation_sha256": implementation_sha,
        "factor_name": subject_factor,
        "horizon_sessions": HORIZON_SESSIONS,
        "inventory_id": "",
        "limitation_codes": limitation_codes,
        "origin_count": len(rows),
        "origin_rows": rows,
        "policy_ref": _policy_ref(policy_ref),
        "protocol_version": PROTOCOL_VERSION,
        "regime_counts": [
            {
                "origin_count": regime_counts[key],
                "regime_state": key,
            }
            for key in sorted(regime_counts)
        ],
        "strategy_id": subject_strategy,
        "version": FACTOR_REGIME_ORIGIN_INVENTORY_VERSION,
    }
    identity_material = dict(document)
    identity_material.pop("inventory_id")
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    document["inventory_id"] = f"factor-regime-origin-inventory-{identity[:32]}"
    return _validate_inventory(seal_semantic(document))


def validate_factor_regime_origin_inventory_replay(
    artifact: Mapping[str, Any],
    *,
    strategy_id: str,
    factor_name: str,
    factor_implementation_sha256: str,
    policy_ref: Mapping[str, Any],
    cutoff: str,
    created_at: str,
    origin_rows: Sequence[FactorRegimeOriginInput],
) -> dict[str, Any]:
    """Rebuild and compare an inventory without reading or writing files."""

    validated = _validate_inventory(artifact)
    rebuilt = build_factor_regime_origin_inventory(
        strategy_id=strategy_id,
        factor_name=factor_name,
        factor_implementation_sha256=factor_implementation_sha256,
        policy_ref=policy_ref,
        cutoff=cutoff,
        created_at=created_at,
        origin_rows=origin_rows,
    )
    if canonical_bytes(validated) != canonical_bytes(rebuilt):
        _fail("factor regime origin inventory replay mismatch")
    return validated


def validate_factor_regime_origin_inventory(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a sealed factor-regime origin inventory artifact."""

    return _validate_inventory(artifact)


__all__ = [
    "ContentArtifactRef",
    "FACTOR_REGIME_ORIGIN_INVENTORY_VERSION",
    "FactorRegimeOriginInput",
    "FactorRegimeOriginInventoryError",
    "RegimeEvidenceSnapshot",
    "build_factor_regime_origin_inventory",
    "validate_factor_regime_origin_inventory",
    "validate_factor_regime_origin_inventory_replay",
]
