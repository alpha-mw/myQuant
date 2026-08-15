"""Compact Factor readiness projection over an exact validated closure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes, seal_artifact

from .admission import ADMITTED_SET_KIND, PROSPECTIVE_ADMISSION_ROUTE
from .bootstrap import (
    BOOTSTRAP_SET_KIND,
    NOT_CLAIMED,
    validate_bootstrap_factor_set,
)
from .bootstrap_evidence import BOOTSTRAP_ADMISSION_ROUTE
from .common import (
    artifact_ref,
    business_identity,
    canonical_timestamp,
    exact_payload,
    require_sha256,
    validate_artifact_ref,
    validate_governance_artifact,
)
from .custody import validate_composite_state
from .errors import FactorGovernanceError
from .receipt import validate_factor_validation_receipt

FACTOR_STATUS_KIND: Final = "factor.status"
FACTOR_READY: Final = "READY"
FACTOR_BLOCKED: Final = "BLOCKED"

_MAX_STATUS_BYTES: Final = 256 * 1024
_STATUS_FIELDS: Final = {
    "status_id",
    "active",
    "observed",
    "readiness",
    "blockers",
    "activation_mutation_authorized",
}
_ACTIVE_FIELDS: Final = {
    "state",
    "lane",
    "admission_route",
    "producer_identity",
    "factor_set_ref",
    "factor_ids",
    "validation_receipt_ref",
    "contextual_result_ref",
    "validation_attestation_ref",
}
_OBSERVED_FIELDS: Final = {
    "composite_state_ref",
    "cycle_state",
    "terminal",
    "blockers",
}
_CONTEXT_FIELDS: Final = {
    "contextual_result_id",
    "validation_namespace_id",
    "lane",
    "intrinsic_receipt_ref",
    "policy_ref",
    "evidence_refs",
    "active_set_ref",
    "composite_state_ref",
    "factor_validator_manifest_ref",
    "contextual_validator_component_ref",
    "source_decoder_component_ref",
    "implementation_component_refs",
    "source_attestation_refs",
    "source_object_refs",
    "custody_record_refs",
    "custody_tree_sha256",
    "custody_head_ref",
    "validated",
    "blockers",
    "authority",
}
_REFERENCE_FIELDS: Final = (
    "kind",
    "contract_sha256",
    "artifact_id",
    "semantic_sha256",
    "byte_sha256",
)
_BLOCKER_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{2,95}$")


def _reference_key(reference: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(reference[field]) for field in _REFERENCE_FIELDS)


def _blockers(values: Any) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FactorGovernanceError("Factor blockers must be a sequence")
    rows: list[str] = []
    for value in values:
        if type(value) is not str or _BLOCKER_RE.fullmatch(value) is None:
            raise FactorGovernanceError("Factor blocker code is invalid")
        rows.append(value)
    if rows != sorted(set(rows)):
        raise FactorGovernanceError("Factor blocker codes are not canonical")
    return rows


def _ref_list(
    value: Any,
    *,
    label: str,
    expected_kind: str | None = None,
    sorted_unique: bool,
) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise FactorGovernanceError(f"{label} must be a list")
    rows = [
        validate_artifact_ref(row, label=f"{label}[{index}]", expected_kind=expected_kind)
        for index, row in enumerate(value)
    ]
    keys = [_reference_key(row) for row in rows]
    if len(keys) != len(set(keys)) or (sorted_unique and keys != sorted(keys)):
        raise FactorGovernanceError(f"{label} is not canonical")
    return rows


def _context_identity(payload: Mapping[str, Any]) -> str:
    return business_identity(
        "factor-contextual-result",
        {field: payload[field] for field in _CONTEXT_FIELDS if field != "contextual_result_id"},
    )


def _validate_contextual_result(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    envelope, payload = exact_payload(
        document,
        kind="factor.contextual_validation_result",
        fields=_CONTEXT_FIELDS,
    )
    for field, kind, nullable in (
        ("intrinsic_receipt_ref", "factor.validation_receipt", False),
        ("policy_ref", None, False),
        ("active_set_ref", None, False),
        ("composite_state_ref", "factor.composite_state", True),
        ("factor_validator_manifest_ref", "factor.validator_manifest", False),
        (
            "contextual_validator_component_ref",
            "system.installed_component_manifest",
            False,
        ),
        ("source_decoder_component_ref", "system.installed_component_manifest", False),
        ("custody_head_ref", "factor.custody_record", True),
    ):
        value = payload[field]
        if value is None and nullable:
            continue
        validate_artifact_ref(value, label=f"context.{field}", expected_kind=kind)
    _ref_list(payload["evidence_refs"], label="context.evidence_refs", sorted_unique=True)
    _ref_list(
        payload["implementation_component_refs"],
        label="context.implementation_component_refs",
        expected_kind="system.installed_component_manifest",
        sorted_unique=True,
    )
    _ref_list(
        payload["source_attestation_refs"],
        label="context.source_attestation_refs",
        expected_kind="factor.source_decode_attestation",
        sorted_unique=True,
    )
    _ref_list(
        payload["source_object_refs"],
        label="context.source_object_refs",
        expected_kind="system.source_object",
        sorted_unique=True,
    )
    _ref_list(
        payload["custody_record_refs"],
        label="context.custody_record_refs",
        expected_kind="factor.custody_record",
        sorted_unique=False,
    )
    require_sha256(payload["custody_tree_sha256"], label="context.custody_tree_sha256")
    _blockers(payload["blockers"])
    if (
        payload["lane"] not in {"BOOTSTRAP", "PROSPECTIVE"}
        or type(payload["validation_namespace_id"]) is not str
        or not payload["validation_namespace_id"]
        or payload["validated"] is not True
        or payload["blockers"] != []
        or payload["authority"] != "NON_AUTHORIZING"
        or payload["contextual_result_id"] != _context_identity(payload)
    ):
        raise FactorGovernanceError("Factor contextual result is not validated")
    if payload["lane"] == "BOOTSTRAP":
        if (
            payload["composite_state_ref"] is not None
            or payload["source_attestation_refs"] != []
            or payload["custody_record_refs"] != []
            or payload["custody_head_ref"] is not None
        ):
            raise FactorGovernanceError("Bootstrap contextual state is inconsistent")
    elif (
        payload["composite_state_ref"] is None
        or payload["custody_head_ref"] is None
        or not payload["source_attestation_refs"]
        or not payload["custody_record_refs"]
    ):
        raise FactorGovernanceError("Prospective contextual state is incomplete")
    return envelope


def _bootstrap_active_metadata(payload: Mapping[str, Any]) -> tuple[str, str, str, list[str]]:
    active = validate_bootstrap_factor_set(payload)
    rows = active["payload"]["factor_rows"]
    factor_ids = [row["factor_id"] for row in rows]
    return "BOOTSTRAP", BOOTSTRAP_ADMISSION_ROUTE, NOT_CLAIMED, factor_ids


def _prospective_active_metadata(payload: Mapping[str, Any]) -> tuple[str, str, str, list[str]]:
    active = validate_governance_artifact(payload)
    if active["kind"] != ADMITTED_SET_KIND:
        raise FactorGovernanceError("active prospective set kind differs")
    rows = active["payload"].get("factor_rows")
    if type(rows) is not list or not 1 <= len(rows) <= 10:
        raise FactorGovernanceError("active prospective factor rows are invalid")
    factor_ids = [row.get("factor_id") for row in rows]
    if (
        any(type(value) is not str or not value for value in factor_ids)
        or factor_ids != sorted(set(factor_ids))
        or active["payload"].get("lane") != "PROSPECTIVE"
        or active["payload"].get("activation_authorized") is not False
    ):
        raise FactorGovernanceError("active prospective set is inconsistent")
    return (
        "PROSPECTIVE",
        PROSPECTIVE_ADMISSION_ROUTE,
        "PROSPECTIVE_GOVERNANCE",
        factor_ids,
    )


def _active_metadata(active: Mapping[str, Any]) -> tuple[str, str, str, list[str]]:
    if active["kind"] == BOOTSTRAP_SET_KIND:
        return _bootstrap_active_metadata(active)
    if active["kind"] == ADMITTED_SET_KIND:
        return _prospective_active_metadata(active)
    raise FactorGovernanceError("active Factor set kind is invalid")


def _empty_active() -> dict[str, Any]:
    return {
        "state": "ABSENT",
        "lane": "NONE",
        "admission_route": "NONE",
        "producer_identity": "NONE",
        "factor_set_ref": None,
        "factor_ids": [],
        "validation_receipt_ref": None,
        "contextual_result_ref": None,
        "validation_attestation_ref": None,
    }


def _attestation_cross_check(
    attestation: Mapping[str, Any],
    *,
    lane: str,
    active_ref: Mapping[str, Any],
    receipt: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    payload = attestation["payload"]
    context_payload = context["payload"]
    expected = {
        "contextual_result_ref": artifact_ref(context),
        "intrinsic_receipt_ref": artifact_ref(receipt),
        "policy_ref": receipt["payload"]["policy_ref"],
        "evidence_refs": receipt["payload"]["evidence_refs"],
        "active_set_ref": dict(active_ref),
        "factor_validator_manifest_ref": context_payload["factor_validator_manifest_ref"],
        "contextual_validator_component_ref": context_payload["contextual_validator_component_ref"],
        "source_decoder_component_ref": context_payload["source_decoder_component_ref"],
        "implementation_component_refs": context_payload["implementation_component_refs"],
        "source_attestation_refs": context_payload["source_attestation_refs"],
        "source_object_refs": context_payload["source_object_refs"],
        "custody_record_refs": context_payload["custody_record_refs"],
        "custody_head_ref": context_payload["custody_head_ref"],
        "custody_tree_sha256": context_payload["custody_tree_sha256"],
    }
    if (
        any(payload.get(field) != value for field, value in expected.items())
        or payload.get("validation_namespace_id") != context_payload["validation_namespace_id"]
        or payload.get("validation_lane") != lane
        or payload.get("candidate_state_ref") != context_payload["composite_state_ref"]
        or payload.get("outcome") != "VALIDATED"
        or payload.get("authority") != "NON_AUTHORIZING"
    ):
        raise FactorGovernanceError("System validation attestation closure differs")


def _active_projection(
    active_factor_set: Mapping[str, Any] | bytes | None,
    active_validation_receipt: Mapping[str, Any] | bytes | None,
    active_contextual_result: Mapping[str, Any] | bytes | None,
    active_validation_attestation: Mapping[str, Any] | bytes | None,
) -> tuple[dict[str, Any], list[str]]:
    values = (
        active_factor_set,
        active_validation_receipt,
        active_contextual_result,
        active_validation_attestation,
    )
    if all(value is None for value in values):
        return _empty_active(), ["ACTIVE_FACTOR_SET_ABSENT"]
    if any(value is None for value in values):
        raise FactorGovernanceError("active Factor validation closure is incomplete")
    assert active_factor_set is not None
    assert active_validation_receipt is not None
    assert active_contextual_result is not None
    assert active_validation_attestation is not None
    active = validate_governance_artifact(active_factor_set)
    receipt = validate_factor_validation_receipt(active_validation_receipt)
    context = _validate_contextual_result(active_contextual_result)
    attestation = validate_governance_artifact(active_validation_attestation)
    if attestation["kind"] != "system.validation_attestation":
        raise FactorGovernanceError("active validation attestation kind differs")
    lane, route, producer, factor_ids = _active_metadata(active)
    active_ref = artifact_ref(active)
    receipt_payload = receipt["payload"]
    context_payload = context["payload"]
    if (
        receipt_payload["active_set_ref"] != active_ref
        or context_payload["lane"] != lane
        or context_payload["intrinsic_receipt_ref"] != artifact_ref(receipt)
        or context_payload["policy_ref"] != receipt_payload["policy_ref"]
        or context_payload["evidence_refs"] != receipt_payload["evidence_refs"]
        or context_payload["active_set_ref"] != active_ref
    ):
        raise FactorGovernanceError("active Factor contextual closure differs")
    _attestation_cross_check(
        attestation,
        lane=lane,
        active_ref=active_ref,
        receipt=receipt,
        context=context,
    )
    return (
        {
            "state": "ACTIVE",
            "lane": lane,
            "admission_route": route,
            "producer_identity": producer,
            "factor_set_ref": active_ref,
            "factor_ids": sorted(factor_ids, key=lambda value: value.encode("utf-8")),
            "validation_receipt_ref": artifact_ref(receipt),
            "contextual_result_ref": artifact_ref(context),
            "validation_attestation_ref": artifact_ref(attestation),
        },
        [],
    )


def _observed_projection(
    observed_composite_state: Mapping[str, Any] | bytes | None,
) -> dict[str, Any]:
    if observed_composite_state is None:
        return {
            "composite_state_ref": None,
            "cycle_state": "NOT_STARTED",
            "terminal": False,
            "blockers": [],
        }
    composite = validate_composite_state(observed_composite_state)
    payload = composite["payload"]
    return {
        "composite_state_ref": artifact_ref(composite),
        "cycle_state": payload["cycle_state"],
        "terminal": payload["terminal"],
        "blockers": list(payload["blockers"]),
    }


def _build_factor_status(
    *,
    active_factor_set: Mapping[str, Any] | bytes | None,
    active_validation_receipt: Mapping[str, Any] | bytes | None,
    active_contextual_result: Mapping[str, Any] | bytes | None,
    active_validation_attestation: Mapping[str, Any] | bytes | None,
    observed_composite_state: Mapping[str, Any] | bytes | None,
    trusted_at: str,
) -> dict[str, Any]:
    active, blockers = _active_projection(
        active_factor_set,
        active_validation_receipt,
        active_contextual_result,
        active_validation_attestation,
    )
    observed = _observed_projection(observed_composite_state)
    readiness = FACTOR_READY if active["state"] == "ACTIVE" and not blockers else FACTOR_BLOCKED
    identity = {
        "active": active,
        "observed": observed,
        "readiness": readiness,
        "blockers": blockers,
    }
    artifact = seal_artifact(
        FACTOR_STATUS_KIND,
        {
            "status_id": business_identity("factor-status", identity),
            **identity,
            "activation_mutation_authorized": False,
        },
        created_at=canonical_timestamp(trusted_at, label="trusted_at"),
    )
    if len(canonical_json_bytes(artifact)) > _MAX_STATUS_BYTES:
        raise FactorGovernanceError("Factor status exceeds its byte limit")
    return artifact


def _validate_absent_active(active: Mapping[str, Any]) -> None:
    if active != _empty_active():
        raise FactorGovernanceError("absent Factor active state is inconsistent")


def _validate_present_active(active: Mapping[str, Any]) -> None:
    valid_triplets = {
        ("BOOTSTRAP", BOOTSTRAP_ADMISSION_ROUTE, NOT_CLAIMED, BOOTSTRAP_SET_KIND),
        (
            "PROSPECTIVE",
            PROSPECTIVE_ADMISSION_ROUTE,
            "PROSPECTIVE_GOVERNANCE",
            ADMITTED_SET_KIND,
        ),
    }
    factor_ref = validate_artifact_ref(active["factor_set_ref"], label="active.factor_set_ref")
    validate_artifact_ref(
        active["validation_receipt_ref"],
        label="active.validation_receipt_ref",
        expected_kind="factor.validation_receipt",
    )
    validate_artifact_ref(
        active["contextual_result_ref"],
        label="active.contextual_result_ref",
        expected_kind="factor.contextual_validation_result",
    )
    validate_artifact_ref(
        active["validation_attestation_ref"],
        label="active.validation_attestation_ref",
        expected_kind="system.validation_attestation",
    )
    if (
        (active["lane"], active["admission_route"], active["producer_identity"], factor_ref["kind"])
        not in valid_triplets
        or type(active["factor_ids"]) is not list
        or not active["factor_ids"]
        or any(type(value) is not str or not value for value in active["factor_ids"])
        or active["factor_ids"] != sorted(set(active["factor_ids"]))
    ):
        raise FactorGovernanceError("active Factor state is inconsistent")


def _validate_active(active: Any) -> None:
    if type(active) is not dict or set(active) != _ACTIVE_FIELDS:
        raise FactorGovernanceError("Factor status active fields are not exact")
    if active["state"] == "ABSENT":
        _validate_absent_active(active)
    elif active["state"] == "ACTIVE":
        _validate_present_active(active)
    else:
        raise FactorGovernanceError("Factor status active state is invalid")


def _validate_observed(observed: Any) -> None:
    if type(observed) is not dict or set(observed) != _OBSERVED_FIELDS:
        raise FactorGovernanceError("Factor status observed fields are not exact")
    blockers = _blockers(observed["blockers"])
    if type(observed["terminal"]) is not bool:
        raise FactorGovernanceError("Factor observed terminal flag is invalid")
    if observed["composite_state_ref"] is None:
        if observed != {
            "composite_state_ref": None,
            "cycle_state": "NOT_STARTED",
            "terminal": False,
            "blockers": [],
        }:
            raise FactorGovernanceError("unstarted Factor observation is inconsistent")
        return
    validate_artifact_ref(
        observed["composite_state_ref"],
        label="observed.composite_state_ref",
        expected_kind="factor.composite_state",
    )
    if (
        type(observed["cycle_state"]) is not str
        or not observed["cycle_state"]
        or (observed["terminal"] is False and blockers)
    ):
        raise FactorGovernanceError("Factor observed composite projection is inconsistent")


def validate_factor_status(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate the stable projection without re-performing contextual storage reads."""

    envelope, payload = exact_payload(document, kind=FACTOR_STATUS_KIND, fields=_STATUS_FIELDS)
    if len(canonical_json_bytes(envelope)) > _MAX_STATUS_BYTES:
        raise FactorGovernanceError("Factor status exceeds its byte limit")
    active = payload["active"]
    observed = payload["observed"]
    _validate_active(active)
    _validate_observed(observed)
    blockers = _blockers(payload["blockers"])
    expected_readiness = (
        FACTOR_READY if active["state"] == "ACTIVE" and not blockers else FACTOR_BLOCKED
    )
    if (
        payload["readiness"] != expected_readiness
        or payload["activation_mutation_authorized"] is not False
    ):
        raise FactorGovernanceError("Factor readiness projection is inconsistent")
    expected_id = business_identity(
        "factor-status",
        {
            "active": active,
            "observed": observed,
            "readiness": expected_readiness,
            "blockers": blockers,
        },
    )
    if payload["status_id"] != expected_id:
        raise FactorGovernanceError("Factor status business identity differs")
    return envelope


__all__ = [
    "FACTOR_BLOCKED",
    "FACTOR_READY",
    "FACTOR_STATUS_KIND",
    "validate_factor_status",
]
