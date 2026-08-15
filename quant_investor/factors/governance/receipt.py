"""Intrinsic, non-authorizing receipts for exact Factor evidence closures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes, seal_artifact

from .admission import (
    ADMITTED_SET_KIND,
    EVALUATION_KIND,
    validate_admitted_factor_set,
    validate_preregistration_evaluation,
)
from .bootstrap import BOOTSTRAP_SET_KIND, validate_bootstrap_factor_set
from .bootstrap_evidence import (
    BOOTSTRAP_EVIDENCE_KIND,
    validate_bootstrap_exception_evidence,
)
from .common import (
    SIGNAL_OPEN_SESSIONS,
    artifact_ref,
    business_identity,
    canonical_timestamp,
    exact_payload,
    validate_artifact_ref,
    validate_governance_artifact,
)
from .custody import COMPOSITE_STATE_KIND, validate_composite_state
from .errors import FactorGovernanceError
from .execution import EXECUTION_EVIDENCE_KIND, validate_execution_turnover_evidence
from .prospective import (
    OBSERVATION_KIND,
    PREREGISTRATION_KIND,
    SELECTION_KIND,
    SIGNAL_CAPTURE_KIND,
    _validate_observation_prevalidated,
    _validate_signal_capture_prevalidated,
    validate_configuration_selection,
    validate_preregistration,
)
from .source import SOURCE_DECODE_ATTESTATION_KIND, validate_source_decode_attestation

VALIDATION_RECEIPT_KIND: Final = "factor.validation_receipt"

_MAX_RECEIPT_BYTES: Final = 2 * 1024 * 1024
_RECEIPT_FIELDS: Final = {
    "validation_receipt_id",
    "policy_ref",
    "evidence_refs",
    "active_set_ref",
    "validated",
    "authority",
}
_REFERENCE_FIELDS: Final = (
    "kind",
    "contract_sha256",
    "artifact_id",
    "semantic_sha256",
    "byte_sha256",
)
_BOOTSTRAP_EVIDENCE_COUNTS: Final = {
    "system.release": 1,
    "system.source_bundle": 7,
}
_PROSPECTIVE_EVIDENCE_COUNTS: Final = {
    SOURCE_DECODE_ATTESTATION_KIND: 721,
    SELECTION_KIND: 1,
    SIGNAL_CAPTURE_KIND: SIGNAL_OPEN_SESSIONS,
    OBSERVATION_KIND: SIGNAL_OPEN_SESSIONS,
    EXECUTION_EVIDENCE_KIND: 1,
    EVALUATION_KIND: 1,
    COMPOSITE_STATE_KIND: 1,
}


def _reference_key(reference: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(reference[field]) for field in _REFERENCE_FIELDS)


def _validated_evidence_artifacts(
    values: Sequence[Mapping[str, Any] | bytes],
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FactorGovernanceError("validation receipt evidence must be a sequence")
    artifacts = [validate_governance_artifact(value) for value in values]
    refs = [artifact_ref(value) for value in artifacts]
    if len(refs) != len({_reference_key(value) for value in refs}):
        raise FactorGovernanceError("validation receipt evidence is duplicated")
    return artifacts


def _sorted_refs(artifacts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((artifact_ref(value) for value in artifacts), key=_reference_key)


def _receipt_evidence_counts(references: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reference in references:
        kind = str(reference["kind"])
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _expected_receipt_evidence_counts(policy_kind: str, active_kind: str) -> dict[str, int]:
    if (policy_kind, active_kind) == (BOOTSTRAP_EVIDENCE_KIND, BOOTSTRAP_SET_KIND):
        return _BOOTSTRAP_EVIDENCE_COUNTS
    if (policy_kind, active_kind) == (PREREGISTRATION_KIND, ADMITTED_SET_KIND):
        return _PROSPECTIVE_EVIDENCE_COUNTS
    raise FactorGovernanceError("Factor validation receipt policy and active-set lanes differ")


def _validate_bootstrap_closure(
    policy: Mapping[str, Any],
    active_set: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    validated_policy = validate_bootstrap_exception_evidence(policy)
    validated_active = validate_bootstrap_factor_set(active_set)
    if validated_active["payload"]["bootstrap_exception_evidence_ref"] != artifact_ref(
        validated_policy
    ):
        raise FactorGovernanceError("bootstrap set references another exception evidence")
    expected_refs = sorted(
        (row["ref"] for row in validated_policy["payload"]["source_refs"]),
        key=_reference_key,
    )
    if len(expected_refs) != 8 or _sorted_refs(evidence) != expected_refs:
        raise FactorGovernanceError("bootstrap validation evidence closure is not exact")


def _artifacts_by_kind(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for artifact in artifacts:
        by_kind.setdefault(str(artifact["kind"]), []).append(artifact)
    return by_kind


def _one(by_kind: Mapping[str, list[Mapping[str, Any]]], kind: str) -> Mapping[str, Any]:
    values = by_kind.get(kind, [])
    if len(values) != 1:
        raise FactorGovernanceError(f"prospective validation closure requires one {kind}")
    return values[0]


def _ordered_by_ordinal(
    values: Sequence[Mapping[str, Any]], *, label: str
) -> list[Mapping[str, Any]]:
    ordered = sorted(values, key=lambda value: value["payload"].get("ordinal", -1))
    if [value["payload"].get("ordinal") for value in ordered] != list(range(SIGNAL_OPEN_SESSIONS)):
        raise FactorGovernanceError(f"prospective {label} ordinals are not exact")
    return ordered


def _validated_capture_chain(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for artifact in _ordered_by_ordinal(artifacts, label="capture"):
        capture = _validate_signal_capture_prevalidated(
            artifact,
            preregistration=preregistration,
            selection=selection,
            previous_signal_capture=previous,
        )
        captures.append(capture)
        previous = capture
    return captures


def _validated_observation_chain(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    captures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for capture, artifact in zip(
        captures,
        _ordered_by_ordinal(artifacts, label="observation"),
        strict=True,
    ):
        observation = _validate_observation_prevalidated(
            artifact,
            preregistration=preregistration,
            selection=selection,
            signal_capture=capture,
            previous_observation=previous,
        )
        observations.append(observation)
        previous = observation
    return observations


def _validate_attestation_projection(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    preregistration: Mapping[str, Any],
    captures: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> None:
    attestations = [validate_source_decode_attestation(value) for value in artifacts]
    expected_refs = [preregistration["payload"]["source_decode_attestation_ref"]]
    expected_refs.extend(value["payload"]["source_decode_attestation_ref"] for value in captures)
    expected_refs.extend(
        value["payload"]["source_decode_attestation_ref"] for value in observations
    )
    if sorted(expected_refs, key=_reference_key) != _sorted_refs(attestations):
        raise FactorGovernanceError("prospective source-attestation closure is not exact")


def _validate_admitted_predecessor(
    composite: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    captures: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    active_set: Mapping[str, Any],
) -> None:
    validated = validate_composite_state(composite)
    payload = validated["payload"]
    expected = {
        "preregistration_ref": artifact_ref(preregistration),
        "selection_ref": artifact_ref(selection),
        "signal_capture_head_ref": artifact_ref(captures[-1]),
        "observation_head_ref": artifact_ref(observations[-1]),
        "execution_evidence_ref": artifact_ref(execution),
        "evaluation_ref": artifact_ref(evaluation),
        "admitted_set_ref": artifact_ref(active_set),
    }
    if any(payload[field] != value for field, value in expected.items()) or any(
        (
            payload["cycle_state"] != "ADMITTED",
            payload["transaction_sequence"] != 724,
            payload["custody_record_count"] != 725,
            payload["signal_capture_count"] != SIGNAL_OPEN_SESSIONS,
            payload["observation_count"] != SIGNAL_OPEN_SESSIONS,
            payload["resolved_signal_slot_count"] != SIGNAL_OPEN_SESSIONS,
            payload["resolved_label_slot_count"] != SIGNAL_OPEN_SESSIONS,
            payload["intrinsic_receipt_ref"] is not None,
            payload["terminal"] is not False,
            payload["blockers"] != [],
        )
    ):
        raise FactorGovernanceError("receipt predecessor composite is not exact ADMITTED state")


def _validate_prospective_closure(
    policy: Mapping[str, Any],
    active_set: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    preregistration = validate_preregistration(policy)
    by_kind = _artifacts_by_kind(evidence)
    if set(by_kind) != set(_PROSPECTIVE_EVIDENCE_COUNTS):
        raise FactorGovernanceError("prospective validation evidence kinds are not exact")
    selection = validate_configuration_selection(
        _one(by_kind, SELECTION_KIND), preregistration=preregistration
    )
    captures = _validated_capture_chain(
        by_kind[SIGNAL_CAPTURE_KIND],
        preregistration=preregistration,
        selection=selection,
    )
    observations = _validated_observation_chain(
        by_kind[OBSERVATION_KIND],
        preregistration=preregistration,
        selection=selection,
        captures=captures,
    )
    _validate_attestation_projection(
        by_kind[SOURCE_DECODE_ATTESTATION_KIND],
        preregistration=preregistration,
        captures=captures,
        observations=observations,
    )
    execution = validate_execution_turnover_evidence(
        _one(by_kind, EXECUTION_EVIDENCE_KIND),
        preregistration=preregistration,
        selection=selection,
    )
    if execution["payload"]["signal_capture_refs"] != [
        artifact_ref(value) for value in captures
    ] or execution["payload"]["observation_refs"] != [
        artifact_ref(value) for value in observations
    ]:
        raise FactorGovernanceError("execution evidence does not bind receipt observations")
    evaluation = validate_preregistration_evaluation(
        _one(by_kind, EVALUATION_KIND),
        preregistration=preregistration,
        selection=selection,
        signal_captures=captures,
        observations=observations,
        execution_turnover_evidence=execution,
    )
    admitted = validate_admitted_factor_set(
        active_set,
        evaluation=evaluation,
        preregistration=preregistration,
        selection=selection,
        signal_captures=captures,
        observations=observations,
        execution_turnover_evidence=execution,
    )
    _validate_admitted_predecessor(
        _one(by_kind, COMPOSITE_STATE_KIND),
        preregistration=preregistration,
        selection=selection,
        captures=captures,
        observations=observations,
        execution=execution,
        evaluation=evaluation,
        active_set=admitted,
    )


def _receipt_payload(
    *,
    policy: Mapping[str, Any],
    active_set: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identity_inputs = {
        "policy_ref": artifact_ref(policy),
        "evidence_refs": _sorted_refs(evidence),
        "active_set_ref": artifact_ref(active_set),
    }
    return {
        "validation_receipt_id": business_identity("factor-validation", identity_inputs),
        **identity_inputs,
        "validated": True,
        "authority": "NON_AUTHORIZING",
    }


def _build_factor_validation_receipt(
    *,
    policy: Mapping[str, Any] | bytes,
    active_set: Mapping[str, Any] | bytes,
    evidence_artifacts: Sequence[Mapping[str, Any] | bytes],
    trusted_at: str,
) -> dict[str, Any]:
    """Replay one complete lane and seal its inert intrinsic receipt."""

    stamp = canonical_timestamp(trusted_at, label="trusted_at")
    validated_policy = validate_governance_artifact(policy)
    validated_active = validate_governance_artifact(active_set)
    validated_evidence = _validated_evidence_artifacts(evidence_artifacts)
    expected_counts = _expected_receipt_evidence_counts(
        validated_policy["kind"], validated_active["kind"]
    )
    if _receipt_evidence_counts(_sorted_refs(validated_evidence)) != expected_counts:
        raise FactorGovernanceError("Factor validation receipt evidence closure is not exact")
    if expected_counts == _BOOTSTRAP_EVIDENCE_COUNTS:
        _validate_bootstrap_closure(validated_policy, validated_active, validated_evidence)
    else:
        _validate_prospective_closure(validated_policy, validated_active, validated_evidence)
    closure = [validated_policy, validated_active, *validated_evidence]
    if any(value["created_at"] > stamp for value in closure):
        raise FactorGovernanceError("Factor validation receipt predates its closure")
    artifact = seal_artifact(
        VALIDATION_RECEIPT_KIND,
        _receipt_payload(
            policy=validated_policy,
            active_set=validated_active,
            evidence=validated_evidence,
        ),
        created_at=stamp,
    )
    if len(canonical_json_bytes(artifact)) > _MAX_RECEIPT_BYTES:
        raise FactorGovernanceError("Factor validation receipt exceeds its byte limit")
    return artifact


def validate_factor_validation_receipt(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate receipt structure and identity without claiming contextual replay."""

    envelope, payload = exact_payload(
        document,
        kind=VALIDATION_RECEIPT_KIND,
        fields=_RECEIPT_FIELDS,
    )
    if len(canonical_json_bytes(envelope)) > _MAX_RECEIPT_BYTES:
        raise FactorGovernanceError("Factor validation receipt exceeds its byte limit")
    policy_ref = validate_artifact_ref(payload["policy_ref"], label="policy_ref")
    active_ref = validate_artifact_ref(payload["active_set_ref"], label="active_set_ref")
    expected_counts = _expected_receipt_evidence_counts(policy_ref["kind"], active_ref["kind"])
    raw_evidence = payload["evidence_refs"]
    if type(raw_evidence) is not list or not raw_evidence:
        raise FactorGovernanceError("Factor validation receipt evidence refs are missing")
    evidence_refs = [
        validate_artifact_ref(value, label=f"evidence_refs[{index}]")
        for index, value in enumerate(raw_evidence)
    ]
    if evidence_refs != sorted(evidence_refs, key=_reference_key) or len(evidence_refs) != len(
        {_reference_key(value) for value in evidence_refs}
    ):
        raise FactorGovernanceError("Factor validation receipt evidence refs are not canonical")
    if _receipt_evidence_counts(evidence_refs) != expected_counts:
        raise FactorGovernanceError("Factor validation receipt evidence closure is not exact")
    if payload["validated"] is not True or payload["authority"] != "NON_AUTHORIZING":
        raise FactorGovernanceError("Factor validation receipt authority differs")
    expected_id = business_identity(
        "factor-validation",
        {
            "policy_ref": policy_ref,
            "evidence_refs": evidence_refs,
            "active_set_ref": active_ref,
        },
    )
    if payload["validation_receipt_id"] != expected_id:
        raise FactorGovernanceError("Factor validation receipt business identity differs")
    return envelope


__all__ = [
    "VALIDATION_RECEIPT_KIND",
    "validate_factor_validation_receipt",
]
