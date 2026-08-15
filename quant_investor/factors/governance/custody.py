"""Immutable Factor custody records and the sole composite candidate state.

This module deliberately contains no mutable storage implementation.  Factor
builders seal immutable records and composite states here; ``SystemStore`` owns
the single non-authorizing candidate-state CAS.  Contextual validation replays
the complete immutable closure before System may attest it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
from typing import Any, Final, TYPE_CHECKING

from quant_investor.contracts import ContractError, canonical_json_bytes, seal_artifact

from .common import (
    SIGNAL_OPEN_SESSIONS,
    artifact_ref,
    business_identity,
    canonical_timestamp,
    exact_payload,
    require_sha256,
    validate_artifact_ref,
)
from .errors import FactorGovernanceError

if TYPE_CHECKING:
    from quant_investor.system import SystemStore


CUSTODY_RECORD_KIND: Final = "factor.custody_record"
COMPOSITE_STATE_KIND: Final = "factor.composite_state"
CLOCK_SOURCE: Final = "FACTOR_VALIDATION_STORE_CLOCK"

_CUSTODY_RECORD_MAX_BYTES: Final = 64 * 1024
_COMPOSITE_STATE_MAX_BYTES: Final = 128 * 1024
_MAX_CUSTODY_RECORDS: Final = 726
_MAX_TRANSACTIONS: Final = 725

_REF_FIELDS: Final = (
    "kind",
    "contract_sha256",
    "artifact_id",
    "semantic_sha256",
    "byte_sha256",
)
_STAGE_SLOT_FIELDS: Final = {
    "stage_slot_id",
    "stage",
    "ordinal",
    "signal_session",
    "maturity_session",
    "state",
    "subject_ref",
    "blocker",
}
_CUSTODY_FIELDS: Final = {
    "custody_record_id",
    "custody_namespace_id",
    "preregistration_id",
    "sequence",
    "previous_custody_ref",
    "previous_composite_state_ref",
    "transaction_id",
    "transaction_sequence",
    "transaction_record_index",
    "transaction_record_count",
    "operation_request_sha256",
    "operation",
    "subject_refs",
    "source_attestation_refs",
    "stage_slot",
    "blockers",
    "stored_at",
    "clock_source",
    "authority",
}
_COMPOSITE_FIELDS: Final = {
    "composite_state_id",
    "custody_namespace_id",
    "preregistration_ref",
    "cycle_state",
    "transaction_sequence",
    "previous_composite_state_ref",
    "transaction_id",
    "custody_record_count",
    "custody_head_ref",
    "selection_ref",
    "signal_capture_count",
    "signal_capture_head_ref",
    "observation_count",
    "observation_head_ref",
    "execution_evidence_ref",
    "evaluation_ref",
    "admitted_set_ref",
    "intrinsic_receipt_ref",
    "resolved_signal_slot_count",
    "resolved_label_slot_count",
    "slot_tree_sha256",
    "terminal",
    "blockers",
    "last_stored_at",
    "authority",
}

_OPERATIONS: Final = frozenset(
    {
        "PREREGISTER",
        "OBSERVE_SIGNAL",
        "OBSERVE_LABEL",
        "FINALIZE_EXECUTION",
        "EVALUATE_PREREGISTRATION",
        "BUILD_ADMITTED_SET",
        "BUILD_INTRINSIC_RECEIPT",
    }
)
_FINAL_OPERATION_KINDS: Final = {
    "FINALIZE_EXECUTION": "factor.execution_turnover_evidence",
    "EVALUATE_PREREGISTRATION": "factor.prospective_evaluation",
    "BUILD_ADMITTED_SET": "factor.admitted_set",
    "BUILD_INTRINSIC_RECEIPT": "factor.validation_receipt",
}
_SUCCESS_STATES: Final = frozenset(
    {
        "PREREGISTERED",
        "OBSERVING",
        "OBSERVATIONS_MATURED",
        "EXECUTION_FINALIZED",
        "EVALUATED_ELIGIBLE",
        "ADMITTED",
        "INTRINSIC_VALIDATED",
    }
)
_FAILURE_STATES: Final = frozenset(
    {
        "EVALUATED_REJECTED",
        "SIGNAL_CAPTURE_MISSED",
        "LABEL_OBSERVATION_MISSED",
        "TERMINAL_INCOMPLETE",
    }
)
_ALL_STATES: Final = _SUCCESS_STATES | _FAILURE_STATES
_TERMINAL_STATES: Final = _FAILURE_STATES | {"INTRINSIC_VALIDATED"}
_BLOCKER_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{2,95}$")
_TRANSITION_FIELDS: Final = (
    "selection_ref",
    "signal_capture_count",
    "signal_capture_head_ref",
    "observation_count",
    "observation_head_ref",
    "execution_evidence_ref",
    "evaluation_ref",
    "admitted_set_ref",
    "intrinsic_receipt_ref",
    "resolved_signal_slot_count",
    "resolved_label_slot_count",
)


@dataclass(frozen=True)
class CustodyReplay:
    """The fully validated immutable custody closure, in chain order."""

    final_composite: dict[str, Any]
    final_composite_ref: dict[str, str]
    custody_records: tuple[dict[str, Any], ...]
    custody_record_refs: tuple[dict[str, str], ...]
    source_attestation_refs: tuple[dict[str, str], ...]
    stage_slots: tuple[dict[str, Any], ...]
    custody_tree_sha256: str
    custody_head_ref: dict[str, str]
    transaction_count: int


def _text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
    ):
        raise FactorGovernanceError(f"{label} must be canonical text")
    return value


def _integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise FactorGovernanceError(f"{label} is outside its allowed domain")
    return value


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise FactorGovernanceError(f"{label} must be an ISO date")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise FactorGovernanceError(f"{label} must be an ISO date") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise FactorGovernanceError(f"{label} must be canonical")
    return value


def _ref_key(value: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    return tuple(value[field] for field in _REF_FIELDS)  # type: ignore[return-value]


def _optional_ref(
    value: Any,
    *,
    label: str,
    expected_kind: str,
) -> dict[str, str] | None:
    if value is None:
        return None
    return validate_artifact_ref(value, label=label, expected_kind=expected_kind)


def _ref_rows(
    values: Any,
    *,
    label: str,
    expected_kind: str | None = None,
    minimum: int = 0,
    maximum: int | None = None,
    require_sorted: bool = True,
) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FactorGovernanceError(f"{label} must be a sequence")
    rows = [
        validate_artifact_ref(
            value,
            label=f"{label}[{index}]",
            expected_kind=expected_kind,
        )
        for index, value in enumerate(values)
    ]
    if len(rows) < minimum or (maximum is not None and len(rows) > maximum):
        raise FactorGovernanceError(f"{label} cardinality is invalid")
    keys = [_ref_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise FactorGovernanceError(f"{label} contains duplicate refs")
    if require_sorted and keys != sorted(keys):
        raise FactorGovernanceError(f"{label} must use canonical ref order")
    return rows


def _blockers(values: Any, *, allow_empty: bool = True) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FactorGovernanceError("blockers must be a sequence")
    rows: list[str] = []
    for index, value in enumerate(values):
        if type(value) is not str or _BLOCKER_RE.fullmatch(value) is None:
            raise FactorGovernanceError(f"blockers[{index}] is invalid")
        rows.append(value)
    if rows != sorted(set(rows)) or (not allow_empty and not rows):
        raise FactorGovernanceError("blockers must be canonical, unique, and nonempty")
    return rows


def _identity_payload(payload: Mapping[str, Any], identity_field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != identity_field}


def _sealed(
    kind: str,
    payload: dict[str, Any],
    *,
    trusted_at: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    try:
        envelope = seal_artifact(kind, payload, created_at=trusted_at)
        raw = canonical_json_bytes(envelope)
    except ContractError as exc:
        raise FactorGovernanceError("artifact sealing failed") from exc
    if len(raw) > maximum_bytes:
        raise FactorGovernanceError(
            "artifact exceeds its closed byte limit",
            code="ARTIFACT_SIZE_LIMIT_EXCEEDED",
        )
    return envelope


def _stage_slot_identity(payload: Mapping[str, Any]) -> str:
    return business_identity("factor-stage-slot", _identity_payload(payload, "stage_slot_id"))


def build_stage_slot(
    *,
    stage: str,
    ordinal: int,
    signal_session: str,
    maturity_session: str | None,
    state: str,
    subject_ref: Mapping[str, Any] | None,
    blocker: str | None,
) -> dict[str, Any]:
    """Build one exact nested stage-slot record."""

    normalized_stage = _text(stage, label="stage")
    if normalized_stage not in {"SIGNAL", "LABEL"}:
        raise FactorGovernanceError("stage is invalid")
    normalized_ordinal = _integer(
        ordinal,
        label="ordinal",
        maximum=SIGNAL_OPEN_SESSIONS - 1,
    )
    normalized_signal = _session(signal_session, label="signal_session")
    if normalized_stage == "SIGNAL":
        if maturity_session is not None:
            raise FactorGovernanceError("SIGNAL stage cannot carry maturity_session")
        normalized_maturity = None
        expected_kind = "factor.signal_capture"
        missed_blocker = "SIGNAL_WINDOW_MISSED"
    else:
        normalized_maturity = _session(maturity_session, label="maturity_session")
        if normalized_maturity <= normalized_signal:
            raise FactorGovernanceError("LABEL maturity must follow signal_session")
        expected_kind = "factor.prospective_observation"
        missed_blocker = "LABEL_WINDOW_MISSED"
    normalized_state = _text(state, label="state")
    if normalized_state == "CAPTURED":
        normalized_subject = validate_artifact_ref(
            subject_ref,
            label="subject_ref",
            expected_kind=expected_kind,
        )
        if blocker is not None:
            raise FactorGovernanceError("captured stage slot cannot carry a blocker")
        normalized_blocker = None
    elif normalized_state == "MISSED":
        if subject_ref is not None or blocker != missed_blocker:
            raise FactorGovernanceError("missed stage slot closure is invalid")
        normalized_subject = None
        normalized_blocker = missed_blocker
    else:
        raise FactorGovernanceError("stage slot state is invalid")
    payload: dict[str, Any] = {
        "stage": normalized_stage,
        "ordinal": normalized_ordinal,
        "signal_session": normalized_signal,
        "maturity_session": normalized_maturity,
        "state": normalized_state,
        "subject_ref": normalized_subject,
        "blocker": normalized_blocker,
    }
    payload["stage_slot_id"] = _stage_slot_identity(payload)
    return {"stage_slot_id": payload.pop("stage_slot_id"), **payload}


def validate_stage_slot(value: Any) -> dict[str, Any]:
    """Validate one nested stage slot and recompute its identity."""

    if type(value) is not dict or set(value) != _STAGE_SLOT_FIELDS:
        raise FactorGovernanceError("stage_slot fields are not exact")
    rebuilt = build_stage_slot(
        stage=value["stage"],
        ordinal=value["ordinal"],
        signal_session=value["signal_session"],
        maturity_session=value["maturity_session"],
        state=value["state"],
        subject_ref=value["subject_ref"],
        blocker=value["blocker"],
    )
    if rebuilt != value:
        raise FactorGovernanceError("stage_slot identity or normalization differs")
    return rebuilt


def _canonical_stage_slots(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(slots, (str, bytes)) or not isinstance(slots, Sequence):
        raise FactorGovernanceError("stage slots must be a sequence")
    rows = [validate_stage_slot(dict(value)) for value in slots]
    keys = [(row["ordinal"], 0 if row["stage"] == "SIGNAL" else 1) for row in rows]
    if len(keys) != len(set(keys)):
        raise FactorGovernanceError("stage slots contain a duplicate stage ordinal")
    expected = sorted(rows, key=lambda row: (row["ordinal"], row["stage"] != "SIGNAL"))
    if rows != expected:
        raise FactorGovernanceError("stage slots are not in canonical order")
    return rows


def custody_slot_tree_sha256(slots: Sequence[Mapping[str, Any]]) -> str:
    """Hash the canonical resolved slot projection used by composite state."""

    rows = _canonical_stage_slots(slots)
    return hashlib.sha256(
        canonical_json_bytes({"domain": "myquant-factor-stage-slots", "slots": rows})
    ).hexdigest()


def operation_request_sha256(
    *,
    operation: str,
    expected_composite_state_ref: Mapping[str, Any] | None,
    input_refs: Mapping[str, Mapping[str, Any] | None],
) -> str:
    """Derive a retry-stable operation identity from exact immutable inputs."""

    normalized_operation = _text(operation, label="operation")
    if normalized_operation not in _OPERATIONS:
        raise FactorGovernanceError("operation is invalid")
    expected = _optional_ref(
        expected_composite_state_ref,
        label="expected_composite_state_ref",
        expected_kind=COMPOSITE_STATE_KIND,
    )
    if type(input_refs) is not dict or not input_refs:
        raise FactorGovernanceError("input_refs must be a nonempty exact mapping")
    normalized_inputs: dict[str, dict[str, str] | None] = {}
    for role in sorted(input_refs, key=lambda item: item.encode("utf-8")):
        normalized_role = _text(role, label="input ref role")
        value = input_refs[role]
        normalized_inputs[normalized_role] = (
            None
            if value is None
            else validate_artifact_ref(value, label=f"input_refs.{normalized_role}")
        )
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant-factor-operation-request",
                "operation": normalized_operation,
                "expected_composite_state_ref": expected,
                "input_refs": normalized_inputs,
            }
        )
    ).hexdigest()


def custody_transaction_id(
    *,
    custody_namespace_id: str,
    transaction_sequence: int,
    previous_composite_state_ref: Mapping[str, Any] | None,
    operation_request_sha256_value: str,
) -> str:
    """Derive the deterministic transaction ID, deliberately excluding time."""

    namespace = _text(custody_namespace_id, label="custody_namespace_id")
    sequence = _integer(
        transaction_sequence,
        label="transaction_sequence",
        minimum=1,
        maximum=_MAX_TRANSACTIONS,
    )
    previous = _optional_ref(
        previous_composite_state_ref,
        label="previous_composite_state_ref",
        expected_kind=COMPOSITE_STATE_KIND,
    )
    request_sha = require_sha256(
        operation_request_sha256_value,
        label="operation_request_sha256",
    )
    return business_identity(
        "factor-custody-transaction",
        {
            "custody_namespace_id": namespace,
            "transaction_sequence": sequence,
            "previous_composite_state_ref": previous,
            "operation_request_sha256": request_sha,
        },
    )


def _validate_record_matrix(payload: Mapping[str, Any]) -> None:
    operation = payload["operation"]
    subjects = payload["subject_refs"]
    attestations = payload["source_attestation_refs"]
    slot = payload["stage_slot"]
    blockers = payload["blockers"]
    index = payload["transaction_record_index"]
    count = payload["transaction_record_count"]

    if operation == "PREREGISTER":
        valid = (
            len(subjects) == 1
            and subjects[0]["kind"] == "factor.preregistration"
            and len(attestations) == 1
            and slot is None
            and not blockers
            and index == 0
            and count == 1
        )
    elif operation == "OBSERVE_SIGNAL" and slot is None:
        valid = (
            len(subjects) == 1
            and subjects[0]["kind"] == "factor.configuration_selection"
            and len(attestations) == 1
            and not blockers
            and index == 0
            and count == 2
        )
    elif operation == "OBSERVE_SIGNAL":
        valid = slot["stage"] == "SIGNAL"
        if slot["state"] == "CAPTURED":
            valid = valid and (
                len(subjects) == 1
                and subjects[0] == slot["subject_ref"]
                and len(attestations) == 1
                and (not blockers or blockers == ["SIGNAL_COVERAGE_BELOW_MINIMUM"])
            )
        else:
            valid = valid and not subjects and not attestations and blockers == [slot["blocker"]]
        valid = valid and (
            (slot["ordinal"] == 0 and slot["state"] == "CAPTURED" and index == 1 and count == 2)
            or ((slot["ordinal"] > 0 or slot["state"] == "MISSED") and index == 0 and count == 1)
        )
    elif operation == "OBSERVE_LABEL":
        valid = slot is not None and slot["stage"] == "LABEL" and index == 0 and count == 1
        if valid and slot["state"] == "CAPTURED":
            valid = (
                len(subjects) == 1
                and subjects[0] == slot["subject_ref"]
                and len(attestations) == 1
                and not blockers
            )
        elif valid:
            valid = not subjects and not attestations and blockers == [slot["blocker"]]
    else:
        expected_kind = _FINAL_OPERATION_KINDS.get(operation)
        valid = (
            expected_kind is not None
            and len(subjects) == 1
            and subjects[0]["kind"] == expected_kind
            and not attestations
            and slot is None
            and not blockers
            and index == 0
            and count == 1
        )
    if not valid:
        raise FactorGovernanceError("custody operation closure is invalid")


def build_custody_record(
    *,
    custody_namespace_id: str,
    preregistration_id: str,
    sequence: int,
    previous_custody_ref: Mapping[str, Any] | None,
    previous_composite_state_ref: Mapping[str, Any] | None,
    transaction_id: str,
    transaction_sequence: int,
    transaction_record_index: int,
    transaction_record_count: int,
    operation_request_sha256: str,
    operation: str,
    subject_refs: Sequence[Mapping[str, Any]],
    source_attestation_refs: Sequence[Mapping[str, Any]],
    stage_slot: Mapping[str, Any] | None,
    blockers: Sequence[str],
    trusted_at: str,
) -> dict[str, Any]:
    """Seal one exact immutable custody record."""

    namespace = _text(custody_namespace_id, label="custody_namespace_id")
    preregistration = _text(preregistration_id, label="preregistration_id")
    record_sequence = _integer(
        sequence,
        label="sequence",
        maximum=_MAX_CUSTODY_RECORDS - 1,
    )
    previous_custody = _optional_ref(
        previous_custody_ref,
        label="previous_custody_ref",
        expected_kind=CUSTODY_RECORD_KIND,
    )
    previous_composite = _optional_ref(
        previous_composite_state_ref,
        label="previous_composite_state_ref",
        expected_kind=COMPOSITE_STATE_KIND,
    )
    if (record_sequence == 0) != (previous_custody is None):
        raise FactorGovernanceError("custody predecessor nullability differs")
    tx_sequence = _integer(
        transaction_sequence,
        label="transaction_sequence",
        minimum=1,
        maximum=_MAX_TRANSACTIONS,
    )
    if (tx_sequence == 1) != (previous_composite is None):
        raise FactorGovernanceError("composite predecessor nullability differs")
    request_sha = require_sha256(operation_request_sha256, label="operation_request_sha256")
    expected_transaction_id = custody_transaction_id(
        custody_namespace_id=namespace,
        transaction_sequence=tx_sequence,
        previous_composite_state_ref=previous_composite,
        operation_request_sha256_value=request_sha,
    )
    if transaction_id != expected_transaction_id:
        raise FactorGovernanceError("custody transaction identity differs")
    record_count = _integer(
        transaction_record_count,
        label="transaction_record_count",
        minimum=1,
        maximum=2,
    )
    record_index = _integer(
        transaction_record_index,
        label="transaction_record_index",
        maximum=record_count - 1,
    )
    normalized_operation = _text(operation, label="operation")
    if normalized_operation not in _OPERATIONS:
        raise FactorGovernanceError("custody operation is invalid")
    normalized_subjects = _ref_rows(
        subject_refs,
        label="subject_refs",
        maximum=1,
    )
    normalized_attestations = _ref_rows(
        source_attestation_refs,
        label="source_attestation_refs",
        expected_kind="factor.source_decode_attestation",
        maximum=1,
    )
    normalized_slot = None if stage_slot is None else validate_stage_slot(dict(stage_slot))
    normalized_blockers = _blockers(blockers)
    stored_at = canonical_timestamp(trusted_at, label="trusted_at")
    payload: dict[str, Any] = {
        "custody_namespace_id": namespace,
        "preregistration_id": preregistration,
        "sequence": record_sequence,
        "previous_custody_ref": previous_custody,
        "previous_composite_state_ref": previous_composite,
        "transaction_id": expected_transaction_id,
        "transaction_sequence": tx_sequence,
        "transaction_record_index": record_index,
        "transaction_record_count": record_count,
        "operation_request_sha256": request_sha,
        "operation": normalized_operation,
        "subject_refs": normalized_subjects,
        "source_attestation_refs": normalized_attestations,
        "stage_slot": normalized_slot,
        "blockers": normalized_blockers,
        "stored_at": stored_at,
        "clock_source": CLOCK_SOURCE,
        "authority": "NON_AUTHORIZING",
    }
    _validate_record_matrix(payload)
    payload["custody_record_id"] = business_identity("factor-custody-record", payload)
    ordered_payload = {"custody_record_id": payload.pop("custody_record_id"), **payload}
    artifact = _sealed(
        CUSTODY_RECORD_KIND,
        ordered_payload,
        trusted_at=stored_at,
        maximum_bytes=_CUSTODY_RECORD_MAX_BYTES,
    )
    return validate_custody_record(artifact)


def _validate_custody_lineage(payload: Mapping[str, Any]) -> None:
    namespace = _text(payload["custody_namespace_id"], label="custody_namespace_id")
    _text(payload["preregistration_id"], label="preregistration_id")
    sequence = _integer(
        payload["sequence"],
        label="sequence",
        maximum=_MAX_CUSTODY_RECORDS - 1,
    )
    previous_custody = _optional_ref(
        payload["previous_custody_ref"],
        label="previous_custody_ref",
        expected_kind=CUSTODY_RECORD_KIND,
    )
    if (sequence == 0) != (previous_custody is None):
        raise FactorGovernanceError("custody predecessor nullability differs")
    tx_sequence = _integer(
        payload["transaction_sequence"],
        label="transaction_sequence",
        minimum=1,
        maximum=_MAX_TRANSACTIONS,
    )
    previous_composite = _optional_ref(
        payload["previous_composite_state_ref"],
        label="previous_composite_state_ref",
        expected_kind=COMPOSITE_STATE_KIND,
    )
    if (tx_sequence == 1) != (previous_composite is None):
        raise FactorGovernanceError("composite predecessor nullability differs")
    request_sha = require_sha256(
        payload["operation_request_sha256"], label="operation_request_sha256"
    )
    expected_transaction = custody_transaction_id(
        custody_namespace_id=namespace,
        transaction_sequence=tx_sequence,
        previous_composite_state_ref=previous_composite,
        operation_request_sha256_value=request_sha,
    )
    if payload["transaction_id"] != expected_transaction:
        raise FactorGovernanceError("custody transaction identity differs")


def _validate_custody_content(payload: Mapping[str, Any]) -> None:
    count = _integer(
        payload["transaction_record_count"],
        label="transaction_record_count",
        minimum=1,
        maximum=2,
    )
    _integer(
        payload["transaction_record_index"],
        label="transaction_record_index",
        maximum=count - 1,
    )
    if payload["operation"] not in _OPERATIONS:
        raise FactorGovernanceError("custody operation is invalid")
    if (
        _ref_rows(payload["subject_refs"], label="subject_refs", maximum=1)
        != payload["subject_refs"]
    ):
        raise FactorGovernanceError("subject_refs normalization differs")
    if (
        _ref_rows(
            payload["source_attestation_refs"],
            label="source_attestation_refs",
            expected_kind="factor.source_decode_attestation",
            maximum=1,
        )
        != payload["source_attestation_refs"]
    ):
        raise FactorGovernanceError("source_attestation_refs normalization differs")
    if payload["stage_slot"] is not None:
        validate_stage_slot(payload["stage_slot"])
    if _blockers(payload["blockers"]) != payload["blockers"]:
        raise FactorGovernanceError("blocker normalization differs")


def validate_custody_record(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate one custody record independently of storage context."""

    envelope, payload = exact_payload(
        document,
        kind=CUSTODY_RECORD_KIND,
        fields=_CUSTODY_FIELDS,
    )
    _validate_custody_lineage(payload)
    _validate_custody_content(payload)
    stored_at = canonical_timestamp(payload["stored_at"], label="stored_at")
    if (
        payload["clock_source"] != CLOCK_SOURCE
        or payload["authority"] != "NON_AUTHORIZING"
        or envelope["created_at"] != stored_at
    ):
        raise FactorGovernanceError("custody clock or authority differs")
    _validate_record_matrix(payload)
    expected_id = business_identity(
        "factor-custody-record", _identity_payload(payload, "custody_record_id")
    )
    if payload["custody_record_id"] != expected_id:
        raise FactorGovernanceError("custody record business identity differs")
    if len(canonical_json_bytes(envelope)) > _CUSTODY_RECORD_MAX_BYTES:
        raise FactorGovernanceError(
            "custody record exceeds its byte limit",
            code="ARTIFACT_SIZE_LIMIT_EXCEEDED",
        )
    return envelope


def _validate_composite_state_flags(payload: Mapping[str, Any]) -> None:
    state = payload["cycle_state"]
    if state not in _ALL_STATES:
        raise FactorGovernanceError("composite cycle_state is invalid")
    terminal = payload["terminal"]
    if type(terminal) is not bool or terminal != (state in _TERMINAL_STATES):
        raise FactorGovernanceError("composite terminal state differs")
    blockers = payload["blockers"]
    if state in _SUCCESS_STATES and state != "INTRINSIC_VALIDATED" and blockers:
        raise FactorGovernanceError("successful intermediate state cannot carry blockers")
    if state == "INTRINSIC_VALIDATED" and blockers:
        raise FactorGovernanceError("validated state cannot carry blockers")
    if state in _FAILURE_STATES and not blockers:
        raise FactorGovernanceError("failed terminal state requires blockers")


def _validate_composite_stage_counters(payload: Mapping[str, Any]) -> None:
    captures = payload["signal_capture_count"]
    observations = payload["observation_count"]
    resolved_signals = payload["resolved_signal_slot_count"]
    resolved_labels = payload["resolved_label_slot_count"]
    if observations > captures or resolved_signals < captures or resolved_labels < observations:
        raise FactorGovernanceError("composite stage counters are inconsistent")
    if (captures == 0) != (payload["signal_capture_head_ref"] is None):
        raise FactorGovernanceError("signal capture head nullability differs")
    if (observations == 0) != (payload["observation_head_ref"] is None):
        raise FactorGovernanceError("observation head nullability differs")


def _composite_state_closure_is_valid(payload: Mapping[str, Any]) -> bool:
    state = payload["cycle_state"]
    captures = payload["signal_capture_count"]
    observations = payload["observation_count"]
    resolved_signals = payload["resolved_signal_slot_count"]
    resolved_labels = payload["resolved_label_slot_count"]
    selection = payload["selection_ref"]
    execution = payload["execution_evidence_ref"]
    evaluation = payload["evaluation_ref"]
    admitted = payload["admitted_set_ref"]
    receipt = payload["intrinsic_receipt_ref"]
    if state == "PREREGISTERED":
        return (
            payload["transaction_sequence"] == 1
            and payload["custody_record_count"] == 1
            and selection is None
            and captures == observations == resolved_signals == resolved_labels == 0
            and execution is evaluation is admitted is receipt is None
        )
    elif state in {
        "OBSERVING",
        "SIGNAL_CAPTURE_MISSED",
        "LABEL_OBSERVATION_MISSED",
        "TERMINAL_INCOMPLETE",
    }:
        return (
            captures <= SIGNAL_OPEN_SESSIONS
            and observations <= SIGNAL_OPEN_SESSIONS
            and execution is evaluation is admitted is receipt is None
            and (selection is not None or captures == 0)
        )
    else:
        complete = (
            selection is not None
            and captures == observations == SIGNAL_OPEN_SESSIONS
            and resolved_signals == resolved_labels == SIGNAL_OPEN_SESSIONS
        )
    if state == "OBSERVATIONS_MATURED":
        return complete and execution is evaluation is admitted is receipt is None
    if state == "EXECUTION_FINALIZED":
        return complete and execution is not None and evaluation is admitted is receipt is None
    if state in {"EVALUATED_ELIGIBLE", "EVALUATED_REJECTED"}:
        return (
            complete
            and execution is not None
            and evaluation is not None
            and admitted is receipt is None
        )
    if state == "ADMITTED":
        return (
            complete
            and execution is not None
            and evaluation is not None
            and admitted is not None
            and receipt is None
        )
    return complete and all(
        value is not None for value in (execution, evaluation, admitted, receipt)
    )


def _validate_composite_policy(payload: Mapping[str, Any]) -> None:
    _validate_composite_state_flags(payload)
    _validate_composite_stage_counters(payload)
    if not _composite_state_closure_is_valid(payload):
        raise FactorGovernanceError("composite state closure is invalid")


def build_composite_state(
    *,
    custody_namespace_id: str,
    preregistration_ref: Mapping[str, Any],
    cycle_state: str,
    transaction_sequence: int,
    previous_composite_state_ref: Mapping[str, Any] | None,
    transaction_id: str,
    custody_record_count: int,
    custody_head_ref: Mapping[str, Any],
    selection_ref: Mapping[str, Any] | None,
    signal_capture_count: int,
    signal_capture_head_ref: Mapping[str, Any] | None,
    observation_count: int,
    observation_head_ref: Mapping[str, Any] | None,
    execution_evidence_ref: Mapping[str, Any] | None,
    evaluation_ref: Mapping[str, Any] | None,
    admitted_set_ref: Mapping[str, Any] | None,
    intrinsic_receipt_ref: Mapping[str, Any] | None,
    resolved_signal_slot_count: int,
    resolved_label_slot_count: int,
    slot_tree_sha256: str,
    terminal: bool,
    blockers: Sequence[str],
    last_stored_at: str,
) -> dict[str, Any]:
    """Seal one immutable projection of the sole candidate state."""

    namespace = _text(custody_namespace_id, label="custody_namespace_id")
    preregistration = validate_artifact_ref(
        preregistration_ref,
        label="preregistration_ref",
        expected_kind="factor.preregistration",
    )
    tx_sequence = _integer(
        transaction_sequence,
        label="transaction_sequence",
        minimum=1,
        maximum=_MAX_TRANSACTIONS,
    )
    previous = _optional_ref(
        previous_composite_state_ref,
        label="previous_composite_state_ref",
        expected_kind=COMPOSITE_STATE_KIND,
    )
    if (tx_sequence == 1) != (previous is None):
        raise FactorGovernanceError("composite predecessor nullability differs")
    custody_count = _integer(
        custody_record_count,
        label="custody_record_count",
        minimum=1,
        maximum=_MAX_CUSTODY_RECORDS,
    )
    custody_head = validate_artifact_ref(
        custody_head_ref,
        label="custody_head_ref",
        expected_kind=CUSTODY_RECORD_KIND,
    )
    captures = _integer(
        signal_capture_count,
        label="signal_capture_count",
        maximum=SIGNAL_OPEN_SESSIONS,
    )
    observations = _integer(
        observation_count,
        label="observation_count",
        maximum=SIGNAL_OPEN_SESSIONS,
    )
    resolved_signals = _integer(
        resolved_signal_slot_count,
        label="resolved_signal_slot_count",
        maximum=SIGNAL_OPEN_SESSIONS,
    )
    resolved_labels = _integer(
        resolved_label_slot_count,
        label="resolved_label_slot_count",
        maximum=SIGNAL_OPEN_SESSIONS,
    )
    normalized_blockers = _blockers(blockers)
    stored_at = canonical_timestamp(last_stored_at, label="last_stored_at")
    payload: dict[str, Any] = {
        "custody_namespace_id": namespace,
        "preregistration_ref": preregistration,
        "cycle_state": _text(cycle_state, label="cycle_state"),
        "transaction_sequence": tx_sequence,
        "previous_composite_state_ref": previous,
        "transaction_id": _text(transaction_id, label="transaction_id"),
        "custody_record_count": custody_count,
        "custody_head_ref": custody_head,
        "selection_ref": _optional_ref(
            selection_ref,
            label="selection_ref",
            expected_kind="factor.configuration_selection",
        ),
        "signal_capture_count": captures,
        "signal_capture_head_ref": _optional_ref(
            signal_capture_head_ref,
            label="signal_capture_head_ref",
            expected_kind="factor.signal_capture",
        ),
        "observation_count": observations,
        "observation_head_ref": _optional_ref(
            observation_head_ref,
            label="observation_head_ref",
            expected_kind="factor.prospective_observation",
        ),
        "execution_evidence_ref": _optional_ref(
            execution_evidence_ref,
            label="execution_evidence_ref",
            expected_kind="factor.execution_turnover_evidence",
        ),
        "evaluation_ref": _optional_ref(
            evaluation_ref,
            label="evaluation_ref",
            expected_kind="factor.prospective_evaluation",
        ),
        "admitted_set_ref": _optional_ref(
            admitted_set_ref,
            label="admitted_set_ref",
            expected_kind="factor.admitted_set",
        ),
        "intrinsic_receipt_ref": _optional_ref(
            intrinsic_receipt_ref,
            label="intrinsic_receipt_ref",
            expected_kind="factor.validation_receipt",
        ),
        "resolved_signal_slot_count": resolved_signals,
        "resolved_label_slot_count": resolved_labels,
        "slot_tree_sha256": require_sha256(slot_tree_sha256, label="slot_tree_sha256"),
        "terminal": terminal,
        "blockers": normalized_blockers,
        "last_stored_at": stored_at,
        "authority": "NON_AUTHORIZING",
    }
    _transaction_request_sha_from_id(transaction_id, namespace, tx_sequence, previous)
    _validate_composite_policy(payload)
    payload["composite_state_id"] = business_identity("factor-composite-state", payload)
    ordered_payload = {"composite_state_id": payload.pop("composite_state_id"), **payload}
    artifact = _sealed(
        COMPOSITE_STATE_KIND,
        ordered_payload,
        trusted_at=stored_at,
        maximum_bytes=_COMPOSITE_STATE_MAX_BYTES,
    )
    return _validate_composite_state(artifact, verify_transaction=False)


def _transaction_request_sha_from_id(
    transaction_id: str,
    namespace: str,
    sequence: int,
    previous: Mapping[str, Any] | None,
) -> str:
    """Recovering a SHA from a one-way identity is impossible; reject bad shape only.

    Composite state intentionally stores the transaction ID, while the bound
    request SHA lives in its head custody record.  Contextual replay performs
    the exact equality check.  Here we validate the stable identity prefix and
    return a dummy value solely so callers cannot pass arbitrary text.
    """

    del namespace, sequence, previous
    prefix = "factor-custody-transaction-"
    if type(transaction_id) is not str or not transaction_id.startswith(prefix):
        raise FactorGovernanceError("transaction_id is invalid")
    digest = transaction_id.removeprefix(prefix)
    return require_sha256(digest, label="transaction_id digest")


def _validate_composite_state(
    document: Mapping[str, Any] | bytes,
    *,
    verify_transaction: bool,
) -> dict[str, Any]:
    envelope, payload = exact_payload(
        document,
        kind=COMPOSITE_STATE_KIND,
        fields=_COMPOSITE_FIELDS,
    )
    namespace = _text(payload["custody_namespace_id"], label="custody_namespace_id")
    validate_artifact_ref(
        payload["preregistration_ref"],
        label="preregistration_ref",
        expected_kind="factor.preregistration",
    )
    tx_sequence = _integer(
        payload["transaction_sequence"],
        label="transaction_sequence",
        minimum=1,
        maximum=_MAX_TRANSACTIONS,
    )
    previous = _optional_ref(
        payload["previous_composite_state_ref"],
        label="previous_composite_state_ref",
        expected_kind=COMPOSITE_STATE_KIND,
    )
    if (tx_sequence == 1) != (previous is None):
        raise FactorGovernanceError("composite predecessor nullability differs")
    _transaction_request_sha_from_id(payload["transaction_id"], namespace, tx_sequence, previous)
    validate_artifact_ref(
        payload["custody_head_ref"],
        label="custody_head_ref",
        expected_kind=CUSTODY_RECORD_KIND,
    )
    _integer(
        payload["custody_record_count"],
        label="custody_record_count",
        minimum=1,
        maximum=_MAX_CUSTODY_RECORDS,
    )
    for field, kind in (
        ("selection_ref", "factor.configuration_selection"),
        ("signal_capture_head_ref", "factor.signal_capture"),
        ("observation_head_ref", "factor.prospective_observation"),
        ("execution_evidence_ref", "factor.execution_turnover_evidence"),
        ("evaluation_ref", "factor.prospective_evaluation"),
        ("admitted_set_ref", "factor.admitted_set"),
        ("intrinsic_receipt_ref", "factor.validation_receipt"),
    ):
        _optional_ref(payload[field], label=field, expected_kind=kind)
    for field in (
        "signal_capture_count",
        "observation_count",
        "resolved_signal_slot_count",
        "resolved_label_slot_count",
    ):
        _integer(payload[field], label=field, maximum=SIGNAL_OPEN_SESSIONS)
    require_sha256(payload["slot_tree_sha256"], label="slot_tree_sha256")
    if type(payload["terminal"]) is not bool or payload["authority"] != "NON_AUTHORIZING":
        raise FactorGovernanceError("composite authority or terminal flag is invalid")
    _blockers(payload["blockers"])
    stored_at = canonical_timestamp(payload["last_stored_at"], label="last_stored_at")
    if envelope["created_at"] != stored_at:
        raise FactorGovernanceError("composite envelope time differs")
    _validate_composite_policy(payload)
    expected_id = business_identity(
        "factor-composite-state", _identity_payload(payload, "composite_state_id")
    )
    if payload["composite_state_id"] != expected_id:
        raise FactorGovernanceError("composite business identity differs")
    if len(canonical_json_bytes(envelope)) > _COMPOSITE_STATE_MAX_BYTES:
        raise FactorGovernanceError(
            "composite state exceeds its byte limit",
            code="ARTIFACT_SIZE_LIMIT_EXCEEDED",
        )
    if verify_transaction:
        _transaction_request_sha_from_id(
            payload["transaction_id"], namespace, tx_sequence, previous
        )
    return envelope


def validate_composite_state(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate one intrinsic composite state without resolving storage."""

    return _validate_composite_state(document, verify_transaction=True)


def _get_exact(system_store: SystemStore, ref: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        document = system_store.get_object(ref)
    except Exception as exc:
        raise FactorGovernanceError(
            f"{label} cannot be resolved",
            code="CUSTODY_CHAIN_BROKEN",
        ) from exc
    if artifact_ref(document) != dict(ref):
        raise FactorGovernanceError(
            f"{label} exact ref differs",
            code="CUSTODY_CHAIN_BROKEN",
        )
    return document


def _walk_ref_chain(
    *,
    system_store: SystemStore,
    head_ref: Mapping[str, Any],
    expected_count: int,
    kind: str,
    previous_field: str,
    validator: Any,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    documents: list[dict[str, Any]] = []
    refs: list[dict[str, str]] = []
    current: dict[str, str] | None = validate_artifact_ref(
        head_ref, label=f"{label}_head_ref", expected_kind=kind
    )
    seen: set[tuple[str, str, str, str, str]] = set()
    while current is not None:
        key = _ref_key(current)
        if key in seen or len(refs) >= expected_count:
            raise FactorGovernanceError(
                f"{label} chain is cyclic or longer than declared",
                code="CUSTODY_CHAIN_BROKEN",
            )
        seen.add(key)
        document = validator(_get_exact(system_store, current, label=label))
        documents.append(document)
        refs.append(current)
        current = document["payload"][previous_field]
    if len(refs) != expected_count:
        raise FactorGovernanceError(
            f"{label} chain count differs",
            code="CUSTODY_CHAIN_BROKEN",
        )
    documents.reverse()
    refs.reverse()
    return documents, refs


def _validate_record_chain(
    records: Sequence[Mapping[str, Any]], refs: Sequence[Mapping[str, str]]
) -> None:
    previous_ref: Mapping[str, str] | None = None
    previous_time: str | None = None
    terminal_seen = False
    namespace = records[0]["payload"]["custody_namespace_id"]
    preregistration_id = records[0]["payload"]["preregistration_id"]
    for sequence, (record, ref) in enumerate(zip(records, refs, strict=True)):
        payload = record["payload"]
        if (
            payload["sequence"] != sequence
            or payload["previous_custody_ref"] != previous_ref
            or payload["custody_namespace_id"] != namespace
            or payload["preregistration_id"] != preregistration_id
        ):
            raise FactorGovernanceError(
                "custody predecessor or lineage differs",
                code="CUSTODY_CHAIN_BROKEN",
            )
        if previous_time is not None and payload["stored_at"] < previous_time:
            raise FactorGovernanceError(
                "custody clock moved backwards",
                code="CUSTODY_CHAIN_BROKEN",
            )
        if terminal_seen:
            raise FactorGovernanceError(
                "custody record follows a terminal stage slot",
                code="CUSTODY_CHAIN_BROKEN",
            )
        slot = payload["stage_slot"]
        if slot is not None and (slot["state"] == "MISSED" or payload["blockers"]):
            terminal_seen = True
        previous_ref = ref
        previous_time = payload["stored_at"]


def _transaction_groups(
    records: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for record in records:
        payload = record["payload"]
        sequence = payload["transaction_sequence"]
        if sequence == len(groups) + 1:
            groups.append([dict(record)])
        elif sequence == len(groups):
            groups[-1].append(dict(record))
        else:
            raise FactorGovernanceError(
                "custody transaction sequence is not contiguous",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
    for sequence, group in enumerate(groups, start=1):
        first = group[0]["payload"]
        count = first["transaction_record_count"]
        if len(group) != count or [
            row["payload"]["transaction_record_index"] for row in group
        ] != list(range(count)):
            raise FactorGovernanceError(
                "custody transaction record set is incomplete",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
        shared = (
            "transaction_id",
            "transaction_sequence",
            "transaction_record_count",
            "operation_request_sha256",
            "operation",
            "previous_composite_state_ref",
            "stored_at",
        )
        if sequence != first["transaction_sequence"] or any(
            any(row["payload"][field] != first[field] for field in shared) for row in group[1:]
        ):
            raise FactorGovernanceError(
                "custody transaction members differ",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
    return groups


def _expected_stage_transaction(stage: str, ordinal: int) -> int:
    if stage == "SIGNAL":
        return ordinal + 2 if ordinal < 30 else 2 * ordinal - 28
    return 2 * ordinal + 33 if ordinal < 330 else ordinal + 362


def _validate_stage_schedule(groups: Sequence[Sequence[Mapping[str, Any]]]) -> None:
    signal_ordinals: list[int] = []
    label_ordinals: list[int] = []
    captured_signals: set[int] = set()
    for transaction_sequence, group in enumerate(groups, start=1):
        slots = [row["payload"]["stage_slot"] for row in group if row["payload"]["stage_slot"]]
        if not slots:
            continue
        if len(slots) != 1:
            raise FactorGovernanceError(
                "custody stage transaction has multiple slots",
                code="CUSTODY_CHAIN_BROKEN",
            )
        slot = slots[0]
        if transaction_sequence != _expected_stage_transaction(slot["stage"], slot["ordinal"]):
            raise FactorGovernanceError(
                "custody stages do not follow the calendar interleave",
                code="CUSTODY_CHAIN_BROKEN",
            )
        if slot["stage"] == "SIGNAL":
            signal_ordinals.append(slot["ordinal"])
            if slot["state"] == "CAPTURED":
                captured_signals.add(slot["ordinal"])
        else:
            if slot["ordinal"] not in captured_signals:
                raise FactorGovernanceError(
                    "label stage precedes its signal capture",
                    code="CUSTODY_CHAIN_BROKEN",
                )
            label_ordinals.append(slot["ordinal"])
    if signal_ordinals != list(range(len(signal_ordinals))) or label_ordinals != list(
        range(len(label_ordinals))
    ):
        raise FactorGovernanceError(
            "stage ordinals are not independently contiguous",
            code="CUSTODY_CHAIN_BROKEN",
        )


def _signal_transition_projection(
    previous: Mapping[str, Any],
    group: Sequence[Mapping[str, Any]],
    expected: dict[str, Any],
    slot: Mapping[str, Any],
) -> set[str]:
    record = group[-1]["payload"]
    if slot["ordinal"] != previous["resolved_signal_slot_count"]:
        raise FactorGovernanceError(
            "signal stage ordinal differs from composite progress",
            code="CUSTODY_CHAIN_BROKEN",
        )
    expected["resolved_signal_slot_count"] += 1
    if slot["state"] == "MISSED":
        return {"SIGNAL_CAPTURE_MISSED"}
    if slot["ordinal"] == 0:
        if len(group) != 2 or group[0]["payload"]["stage_slot"] is not None:
            raise FactorGovernanceError(
                "first signal transaction is not atomic",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
        expected["selection_ref"] = group[0]["payload"]["subject_refs"][0]
        if group[0]["payload"]["source_attestation_refs"] != record["source_attestation_refs"]:
            raise FactorGovernanceError(
                "first selection/capture attestation differs",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
    elif len(group) != 1 or previous["selection_ref"] is None:
        raise FactorGovernanceError(
            "later signal transaction changed selection semantics",
            code="CUSTODY_TRANSACTION_INCOMPLETE",
        )
    expected["signal_capture_count"] += 1
    expected["signal_capture_head_ref"] = slot["subject_ref"]
    return {"TERMINAL_INCOMPLETE"} if record["blockers"] else {"OBSERVING"}


def _label_transition_projection(
    previous: Mapping[str, Any],
    expected: dict[str, Any],
    slot: Mapping[str, Any],
) -> set[str]:
    if slot["ordinal"] != previous["resolved_label_slot_count"]:
        raise FactorGovernanceError(
            "label stage ordinal differs from composite progress",
            code="CUSTODY_CHAIN_BROKEN",
        )
    if previous["signal_capture_count"] <= slot["ordinal"]:
        raise FactorGovernanceError(
            "label stage has no captured signal predecessor",
            code="CUSTODY_CHAIN_BROKEN",
        )
    expected["resolved_label_slot_count"] += 1
    if slot["state"] == "MISSED":
        return {"LABEL_OBSERVATION_MISSED"}
    expected["observation_count"] += 1
    expected["observation_head_ref"] = slot["subject_ref"]
    return {"OBSERVING"}


def _stage_transition_projection(
    previous: Mapping[str, Any],
    group: Sequence[Mapping[str, Any]],
    expected: dict[str, Any],
) -> set[str]:
    slot = group[-1]["payload"]["stage_slot"]
    if slot is None:
        raise FactorGovernanceError(
            "stage transaction lacks its slot",
            code="CUSTODY_TRANSACTION_INCOMPLETE",
        )
    states = (
        _signal_transition_projection(previous, group, expected, slot)
        if slot["stage"] == "SIGNAL"
        else _label_transition_projection(previous, expected, slot)
    )
    if states == {"OBSERVING"} and (
        expected["signal_capture_count"] == SIGNAL_OPEN_SESSIONS
        and expected["observation_count"] == SIGNAL_OPEN_SESSIONS
    ):
        return {"OBSERVATIONS_MATURED"}
    return states


def _final_transition_projection(
    operation: str,
    subject_ref: Mapping[str, Any],
    expected: dict[str, Any],
    previous_state: str,
) -> set[str]:
    required_previous = {
        "FINALIZE_EXECUTION": "OBSERVATIONS_MATURED",
        "EVALUATE_PREREGISTRATION": "EXECUTION_FINALIZED",
        "BUILD_ADMITTED_SET": "EVALUATED_ELIGIBLE",
        "BUILD_INTRINSIC_RECEIPT": "ADMITTED",
    }.get(operation)
    if previous_state != required_previous:
        raise FactorGovernanceError(
            "custody finalization skips or repeats a state",
            code="CUSTODY_TRANSACTION_INCOMPLETE",
        )
    if operation == "FINALIZE_EXECUTION":
        expected["execution_evidence_ref"] = subject_ref
        return {"EXECUTION_FINALIZED"}
    if operation == "EVALUATE_PREREGISTRATION":
        expected["evaluation_ref"] = subject_ref
        return {"EVALUATED_ELIGIBLE", "EVALUATED_REJECTED"}
    if operation == "BUILD_ADMITTED_SET":
        expected["admitted_set_ref"] = subject_ref
        return {"ADMITTED"}
    if operation == "BUILD_INTRINSIC_RECEIPT":
        expected["intrinsic_receipt_ref"] = subject_ref
        return {"INTRINSIC_VALIDATED"}
    raise FactorGovernanceError(
        "custody transition operation is invalid",
        code="CUSTODY_TRANSACTION_INCOMPLETE",
    )


def _validate_transition_projection(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    group: Sequence[Mapping[str, Any]],
) -> None:
    first = group[0]["payload"]
    if previous is None:
        valid = (
            first["operation"] == "PREREGISTER"
            and len(group) == 1
            and first["subject_refs"] == [current["preregistration_ref"]]
            and first["preregistration_id"] == current["preregistration_ref"]["artifact_id"]
            and current["cycle_state"] == "PREREGISTERED"
        )
        if not valid:
            raise FactorGovernanceError(
                "initial composite transaction differs",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
        return
    if previous["terminal"]:
        raise FactorGovernanceError(
            "composite transition follows a terminal state",
            code="CUSTODY_CHAIN_BROKEN",
        )
    expected = {field: previous[field] for field in _TRANSITION_FIELDS}
    operation = first["operation"]
    if operation in {"OBSERVE_SIGNAL", "OBSERVE_LABEL"}:
        allowed_states = _stage_transition_projection(previous, group, expected)
    else:
        allowed_states = _final_transition_projection(
            operation,
            first["subject_refs"][0],
            expected,
            previous["cycle_state"],
        )
    if any(current[field] != expected[field] for field in _TRANSITION_FIELDS):
        raise FactorGovernanceError(
            "composite transition projection differs",
            code="CUSTODY_TRANSACTION_INCOMPLETE",
        )
    if current["cycle_state"] not in allowed_states:
        raise FactorGovernanceError(
            "composite transition state differs",
            code="CUSTODY_TRANSACTION_INCOMPLETE",
        )
    expected_blockers = group[-1]["payload"]["blockers"]
    evaluation_rejected = (
        operation == "EVALUATE_PREREGISTRATION" and current["cycle_state"] == "EVALUATED_REJECTED"
    )
    if not evaluation_rejected and current["blockers"] != expected_blockers:
        raise FactorGovernanceError(
            "composite blockers differ from the terminal transaction",
            code="CUSTODY_TRANSACTION_INCOMPLETE",
        )


def _validate_composite_chain(
    composites: Sequence[Mapping[str, Any]],
    composite_refs: Sequence[Mapping[str, str]],
    records: Sequence[Mapping[str, Any]],
    record_refs: Sequence[Mapping[str, str]],
    groups: Sequence[Sequence[Mapping[str, Any]]],
) -> None:
    if len(composites) != len(groups):
        raise FactorGovernanceError(
            "composite and transaction counts differ",
            code="CUSTODY_TRANSACTION_INCOMPLETE",
        )
    record_offset = 0
    accumulated_slots: list[dict[str, Any]] = []
    for index, (composite, composite_ref, group) in enumerate(
        zip(composites, composite_refs, groups, strict=True)
    ):
        payload = composite["payload"]
        previous_ref = None if index == 0 else composite_refs[index - 1]
        last_record_index = record_offset + len(group) - 1
        last_record = records[last_record_index]["payload"]
        expected_transaction_id = custody_transaction_id(
            custody_namespace_id=payload["custody_namespace_id"],
            transaction_sequence=index + 1,
            previous_composite_state_ref=previous_ref,
            operation_request_sha256_value=last_record["operation_request_sha256"],
        )
        accumulated_slots.extend(
            row["payload"]["stage_slot"]
            for row in group
            if row["payload"]["stage_slot"] is not None
        )
        canonical_slots = sorted(
            accumulated_slots,
            key=lambda row: (row["ordinal"], row["stage"] != "SIGNAL"),
        )
        if (
            payload["transaction_sequence"] != index + 1
            or payload["previous_composite_state_ref"] != previous_ref
            or payload["transaction_id"] != expected_transaction_id
            or last_record["transaction_id"] != expected_transaction_id
            or payload["custody_record_count"] != last_record_index + 1
            or payload["custody_head_ref"] != record_refs[last_record_index]
            or payload["last_stored_at"] != last_record["stored_at"]
            or payload["slot_tree_sha256"] != custody_slot_tree_sha256(canonical_slots)
        ):
            raise FactorGovernanceError(
                "composite transaction projection differs",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
        if index and (
            payload["custody_namespace_id"] != composites[0]["payload"]["custody_namespace_id"]
            or payload["preregistration_ref"] != composites[0]["payload"]["preregistration_ref"]
        ):
            raise FactorGovernanceError(
                "composite immutable roots changed",
                code="CUSTODY_CHAIN_BROKEN",
            )
        if any(row["payload"]["previous_composite_state_ref"] != previous_ref for row in group):
            raise FactorGovernanceError(
                "transaction binds the wrong previous composite",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
        if any(
            row["payload"]["custody_namespace_id"] != payload["custody_namespace_id"]
            or row["payload"]["preregistration_id"] != payload["preregistration_ref"]["artifact_id"]
            for row in group
        ):
            raise FactorGovernanceError(
                "transaction custody roots differ from the composite",
                code="CUSTODY_CHAIN_BROKEN",
            )
        _validate_transition_projection(
            None if index == 0 else composites[index - 1]["payload"],
            payload,
            group,
        )
        if artifact_ref(composite) != composite_ref:
            raise FactorGovernanceError(
                "composite exact ref differs",
                code="CUSTODY_CHAIN_BROKEN",
            )
        record_offset += len(group)


def replay_custody_chain(
    *,
    system_store: SystemStore,
    final_composite: Mapping[str, Any] | bytes,
) -> CustodyReplay:
    """Resolve and replay the complete immutable custody/composite closure."""

    final = validate_composite_state(final_composite)
    final_payload = final["payload"]
    final_ref = artifact_ref(final)
    stored_final = _get_exact(system_store, final_ref, label="final composite")
    if stored_final != final:
        raise FactorGovernanceError(
            "final composite is not the stored exact envelope",
            code="CUSTODY_CHAIN_BROKEN",
        )
    records, record_refs = _walk_ref_chain(
        system_store=system_store,
        head_ref=final_payload["custody_head_ref"],
        expected_count=final_payload["custody_record_count"],
        kind=CUSTODY_RECORD_KIND,
        previous_field="previous_custody_ref",
        validator=validate_custody_record,
        label="custody record",
    )
    composites, composite_refs = _walk_ref_chain(
        system_store=system_store,
        head_ref=final_ref,
        expected_count=final_payload["transaction_sequence"],
        kind=COMPOSITE_STATE_KIND,
        previous_field="previous_composite_state_ref",
        validator=validate_composite_state,
        label="composite state",
    )
    _validate_record_chain(records, record_refs)
    groups = _transaction_groups(records)
    _validate_composite_chain(composites, composite_refs, records, record_refs, groups)
    _validate_stage_schedule(groups)
    if final_payload["cycle_state"] == "INTRINSIC_VALIDATED":
        if len(groups) != 725 or len(records) != 726:
            raise FactorGovernanceError(
                "validated custody closure has the wrong exact counts",
                code="CONTEXT_CLOSURE_INCOMPLETE",
            )

    slots = [
        record["payload"]["stage_slot"]
        for record in records
        if record["payload"]["stage_slot"] is not None
    ]
    canonical_slots = sorted(slots, key=lambda row: (row["ordinal"], row["stage"] != "SIGNAL"))
    _canonical_stage_slots(canonical_slots)
    source_refs_by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for record in records:
        for ref in record["payload"]["subject_refs"]:
            _get_exact(system_store, ref, label="custody subject")
        for ref in record["payload"]["source_attestation_refs"]:
            _get_exact(system_store, ref, label="source attestation")
            source_refs_by_key[_ref_key(ref)] = dict(ref)
    source_refs = tuple(source_refs_by_key[key] for key in sorted(source_refs_by_key))
    if final_payload["cycle_state"] == "INTRINSIC_VALIDATED" and (
        len(source_refs) != 721 or len(canonical_slots) != 720
    ):
        raise FactorGovernanceError(
            "validated source-attestation or stage-slot count differs",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    custody_tree = hashlib.sha256(canonical_json_bytes(record_refs)).hexdigest()
    return CustodyReplay(
        final_composite=final,
        final_composite_ref=final_ref,
        custody_records=tuple(records),
        custody_record_refs=tuple(record_refs),
        source_attestation_refs=source_refs,
        stage_slots=tuple(canonical_slots),
        custody_tree_sha256=custody_tree,
        custody_head_ref=dict(record_refs[-1]),
        transaction_count=len(groups),
    )


__all__ = [
    "CLOCK_SOURCE",
    "COMPOSITE_STATE_KIND",
    "CUSTODY_RECORD_KIND",
    "CustodyReplay",
    "build_composite_state",
    "build_custody_record",
    "build_stage_slot",
    "custody_slot_tree_sha256",
    "custody_transaction_id",
    "operation_request_sha256",
    "replay_custody_chain",
    "validate_composite_state",
    "validate_custody_record",
    "validate_stage_slot",
]
