from __future__ import annotations

import copy
from typing import Any

import pytest

from quant_investor.contracts import get_contract, seal_artifact
from quant_investor.factors.governance import FactorGovernanceError
from quant_investor.factors.governance.common import artifact_ref, business_identity
from quant_investor.factors.governance.custody import (
    build_composite_state,
    build_custody_record,
    build_stage_slot,
    custody_slot_tree_sha256,
    custody_transaction_id,
    operation_request_sha256,
    replay_custody_chain,
    validate_composite_state,
    validate_custody_record,
    validate_stage_slot,
)

STAMP = "2026-08-14T00:00:00Z"
SHA = "1" * 64
NAMESPACE = "factor-validation-namespace-test"


def _ref_key(value: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        value["kind"],
        value["contract_sha256"],
        value["artifact_id"],
        value["semantic_sha256"],
        value["byte_sha256"],
    )


class _MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def add(self, *documents: dict[str, Any]) -> None:
        for document in documents:
            ref = artifact_ref(document)
            self.objects[_ref_key(ref)] = document

    def get_object(self, ref: dict[str, str]) -> dict[str, Any]:
        return copy.deepcopy(self.objects[_ref_key(dict(ref))])


def _placeholder(kind: str, artifact_id: str) -> dict[str, Any]:
    definition = get_contract(kind)
    payload: dict[str, Any] = {field: None for field in definition.required_payload_fields}
    payload[definition.identity_field] = artifact_id
    return seal_artifact(kind, payload, created_at=STAMP)


def _source_attestation(artifact_id: str) -> dict[str, Any]:
    fake_ref = {
        "kind": "system.installed_component_manifest",
        "contract_sha256": SHA,
        "artifact_id": "component",
        "semantic_sha256": SHA,
        "byte_sha256": SHA,
    }
    return seal_artifact(
        "factor.source_decode_attestation",
        {
            "source_decode_attestation_id": artifact_id,
            "purpose": "PREREGISTRATION",
            "preregistration_id": None,
            "selection_id": None,
            "ordinal": None,
            "signal_session": None,
            "maturity_session": None,
            "decoder_contract": {
                "decoder_id": "factor-strict-parquet-source-decoder",
                "factor_validator_manifest_ref": fake_ref,
                "contextual_validator_component_ref": fake_ref,
                "source_decoder_component_ref": fake_ref,
                "decoder_code_sha256": SHA,
                "implementation_component_refs": [],
                "allowed_source_formats": ["PARQUET"],
                "fallback_allowed": False,
            },
            "source_bindings": [],
            "normalized_inputs_sha256": SHA,
            "authority": "NON_AUTHORIZING",
        },
        created_at=STAMP,
    )


def _request_sha(
    operation: str,
    expected: dict[str, str] | None,
    input_ref: dict[str, str],
) -> str:
    return operation_request_sha256(
        operation=operation,
        expected_composite_state_ref=expected,
        input_refs={"input": input_ref},
    )


def _root_and_first_capture() -> tuple[
    _MemoryStore,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    store = _MemoryStore()
    preregistration = _placeholder("factor.preregistration", "preregistration-test")
    preregistration_ref = artifact_ref(preregistration)
    prereg_attestation = _source_attestation("source-attestation-prereg")
    request1 = _request_sha("PREREGISTER", None, preregistration_ref)
    transaction1 = custody_transaction_id(
        custody_namespace_id=NAMESPACE,
        transaction_sequence=1,
        previous_composite_state_ref=None,
        operation_request_sha256_value=request1,
    )
    record1 = build_custody_record(
        custody_namespace_id=NAMESPACE,
        preregistration_id=preregistration_ref["artifact_id"],
        sequence=0,
        previous_custody_ref=None,
        previous_composite_state_ref=None,
        transaction_id=transaction1,
        transaction_sequence=1,
        transaction_record_index=0,
        transaction_record_count=1,
        operation_request_sha256=request1,
        operation="PREREGISTER",
        subject_refs=[preregistration_ref],
        source_attestation_refs=[artifact_ref(prereg_attestation)],
        stage_slot=None,
        blockers=[],
        trusted_at=STAMP,
    )
    composite1 = build_composite_state(
        custody_namespace_id=NAMESPACE,
        preregistration_ref=preregistration_ref,
        cycle_state="PREREGISTERED",
        transaction_sequence=1,
        previous_composite_state_ref=None,
        transaction_id=transaction1,
        custody_record_count=1,
        custody_head_ref=artifact_ref(record1),
        selection_ref=None,
        signal_capture_count=0,
        signal_capture_head_ref=None,
        observation_count=0,
        observation_head_ref=None,
        execution_evidence_ref=None,
        evaluation_ref=None,
        admitted_set_ref=None,
        intrinsic_receipt_ref=None,
        resolved_signal_slot_count=0,
        resolved_label_slot_count=0,
        slot_tree_sha256=custody_slot_tree_sha256([]),
        terminal=False,
        blockers=[],
        last_stored_at=STAMP,
    )

    selection = _placeholder("factor.configuration_selection", "selection-test")
    capture = _placeholder("factor.signal_capture", "capture-0")
    signal_attestation = _source_attestation("source-attestation-signal-0")
    capture_slot = build_stage_slot(
        stage="SIGNAL",
        ordinal=0,
        signal_session="2026-08-17",
        maturity_session=None,
        state="CAPTURED",
        subject_ref=artifact_ref(capture),
        blocker=None,
    )
    composite1_ref = artifact_ref(composite1)
    request2 = _request_sha("OBSERVE_SIGNAL", composite1_ref, artifact_ref(capture))
    transaction2 = custody_transaction_id(
        custody_namespace_id=NAMESPACE,
        transaction_sequence=2,
        previous_composite_state_ref=composite1_ref,
        operation_request_sha256_value=request2,
    )
    selection_record = build_custody_record(
        custody_namespace_id=NAMESPACE,
        preregistration_id=preregistration_ref["artifact_id"],
        sequence=1,
        previous_custody_ref=artifact_ref(record1),
        previous_composite_state_ref=composite1_ref,
        transaction_id=transaction2,
        transaction_sequence=2,
        transaction_record_index=0,
        transaction_record_count=2,
        operation_request_sha256=request2,
        operation="OBSERVE_SIGNAL",
        subject_refs=[artifact_ref(selection)],
        source_attestation_refs=[artifact_ref(signal_attestation)],
        stage_slot=None,
        blockers=[],
        trusted_at=STAMP,
    )
    capture_record = build_custody_record(
        custody_namespace_id=NAMESPACE,
        preregistration_id=preregistration_ref["artifact_id"],
        sequence=2,
        previous_custody_ref=artifact_ref(selection_record),
        previous_composite_state_ref=composite1_ref,
        transaction_id=transaction2,
        transaction_sequence=2,
        transaction_record_index=1,
        transaction_record_count=2,
        operation_request_sha256=request2,
        operation="OBSERVE_SIGNAL",
        subject_refs=[artifact_ref(capture)],
        source_attestation_refs=[artifact_ref(signal_attestation)],
        stage_slot=capture_slot,
        blockers=[],
        trusted_at=STAMP,
    )
    composite2 = build_composite_state(
        custody_namespace_id=NAMESPACE,
        preregistration_ref=preregistration_ref,
        cycle_state="OBSERVING",
        transaction_sequence=2,
        previous_composite_state_ref=composite1_ref,
        transaction_id=transaction2,
        custody_record_count=3,
        custody_head_ref=artifact_ref(capture_record),
        selection_ref=artifact_ref(selection),
        signal_capture_count=1,
        signal_capture_head_ref=artifact_ref(capture),
        observation_count=0,
        observation_head_ref=None,
        execution_evidence_ref=None,
        evaluation_ref=None,
        admitted_set_ref=None,
        intrinsic_receipt_ref=None,
        resolved_signal_slot_count=1,
        resolved_label_slot_count=0,
        slot_tree_sha256=custody_slot_tree_sha256([capture_slot]),
        terminal=False,
        blockers=[],
        last_stored_at=STAMP,
    )
    store.add(
        preregistration,
        prereg_attestation,
        record1,
        composite1,
        selection,
        capture,
        signal_attestation,
        selection_record,
        capture_record,
        composite2,
    )
    return store, composite1, composite2, [record1, selection_record, capture_record]


def test_stage_slot_identity_tree_and_nullability_are_exact() -> None:
    capture = _placeholder("factor.signal_capture", "capture-slot")
    slot = build_stage_slot(
        stage="SIGNAL",
        ordinal=0,
        signal_session="2026-08-17",
        maturity_session=None,
        state="CAPTURED",
        subject_ref=artifact_ref(capture),
        blocker=None,
    )
    assert validate_stage_slot(slot) == slot
    assert custody_slot_tree_sha256([slot]) == custody_slot_tree_sha256([slot])

    forged = copy.deepcopy(slot)
    forged["ordinal"] = 1
    with pytest.raises(FactorGovernanceError):
        validate_stage_slot(forged)
    with pytest.raises(FactorGovernanceError):
        custody_slot_tree_sha256([slot, slot])


def test_operation_and_transaction_identity_exclude_time_and_nonce() -> None:
    preregistration = _placeholder("factor.preregistration", "prereg-id")
    request = _request_sha("PREREGISTER", None, artifact_ref(preregistration))
    first = custody_transaction_id(
        custody_namespace_id=NAMESPACE,
        transaction_sequence=1,
        previous_composite_state_ref=None,
        operation_request_sha256_value=request,
    )
    second = custody_transaction_id(
        custody_namespace_id=NAMESPACE,
        transaction_sequence=1,
        previous_composite_state_ref=None,
        operation_request_sha256_value=request,
    )
    assert first == second

    with pytest.raises(TypeError):
        operation_request_sha256(  # type: ignore[call-arg]
            operation="PREREGISTER",
            expected_composite_state_ref=None,
            input_refs={"input": artifact_ref(preregistration)},
            created_at=STAMP,
        )


def test_custody_replay_accepts_atomic_selection_and_capture_zero() -> None:
    store, _, composite, _ = _root_and_first_capture()
    replay = replay_custody_chain(system_store=store, final_composite=composite)

    assert replay.transaction_count == 2
    assert len(replay.custody_records) == 3
    assert len(replay.stage_slots) == 1
    assert len(replay.source_attestation_refs) == 2
    assert replay.final_composite_ref == artifact_ref(composite)


def test_custody_replay_rejects_label_before_frozen_calendar_transaction() -> None:
    store, _, composite2, records = _root_and_first_capture()
    observation = _placeholder("factor.prospective_observation", "observation-0")
    attestation = _source_attestation("source-attestation-label-0")
    slot = build_stage_slot(
        stage="LABEL",
        ordinal=0,
        signal_session="2026-08-17",
        maturity_session="2026-09-16",
        state="CAPTURED",
        subject_ref=artifact_ref(observation),
        blocker=None,
    )
    composite2_ref = artifact_ref(composite2)
    request = _request_sha("OBSERVE_LABEL", composite2_ref, artifact_ref(observation))
    transaction = custody_transaction_id(
        custody_namespace_id=NAMESPACE,
        transaction_sequence=3,
        previous_composite_state_ref=composite2_ref,
        operation_request_sha256_value=request,
    )
    record = build_custody_record(
        custody_namespace_id=NAMESPACE,
        preregistration_id=composite2["payload"]["preregistration_ref"]["artifact_id"],
        sequence=3,
        previous_custody_ref=artifact_ref(records[-1]),
        previous_composite_state_ref=composite2_ref,
        transaction_id=transaction,
        transaction_sequence=3,
        transaction_record_index=0,
        transaction_record_count=1,
        operation_request_sha256=request,
        operation="OBSERVE_LABEL",
        subject_refs=[artifact_ref(observation)],
        source_attestation_refs=[artifact_ref(attestation)],
        stage_slot=slot,
        blockers=[],
        trusted_at=STAMP,
    )
    composite3 = build_composite_state(
        custody_namespace_id=NAMESPACE,
        preregistration_ref=composite2["payload"]["preregistration_ref"],
        cycle_state="OBSERVING",
        transaction_sequence=3,
        previous_composite_state_ref=composite2_ref,
        transaction_id=transaction,
        custody_record_count=4,
        custody_head_ref=artifact_ref(record),
        selection_ref=composite2["payload"]["selection_ref"],
        signal_capture_count=1,
        signal_capture_head_ref=composite2["payload"]["signal_capture_head_ref"],
        observation_count=1,
        observation_head_ref=artifact_ref(observation),
        execution_evidence_ref=None,
        evaluation_ref=None,
        admitted_set_ref=None,
        intrinsic_receipt_ref=None,
        resolved_signal_slot_count=1,
        resolved_label_slot_count=1,
        slot_tree_sha256=custody_slot_tree_sha256([records[-1]["payload"]["stage_slot"], slot]),
        terminal=False,
        blockers=[],
        last_stored_at=STAMP,
    )
    store.add(observation, attestation, record, composite3)

    with pytest.raises(FactorGovernanceError, match="calendar interleave"):
        replay_custody_chain(system_store=store, final_composite=composite3)


def test_custody_replay_rejects_missing_immutable_predecessor() -> None:
    store, _, composite, records = _root_and_first_capture()
    store.objects.pop(_ref_key(artifact_ref(records[0])))
    with pytest.raises(FactorGovernanceError, match="CUSTODY_CHAIN_BROKEN"):
        replay_custody_chain(system_store=store, final_composite=composite)


def test_custody_replay_rejects_forged_composite_counter_projection() -> None:
    store, composite1, composite2, records = _root_and_first_capture()
    payload = composite2["payload"]
    forged = build_composite_state(
        custody_namespace_id=NAMESPACE,
        preregistration_ref=payload["preregistration_ref"],
        cycle_state="OBSERVING",
        transaction_sequence=2,
        previous_composite_state_ref=artifact_ref(composite1),
        transaction_id=payload["transaction_id"],
        custody_record_count=3,
        custody_head_ref=artifact_ref(records[-1]),
        selection_ref=payload["selection_ref"],
        signal_capture_count=2,
        signal_capture_head_ref=payload["signal_capture_head_ref"],
        observation_count=0,
        observation_head_ref=None,
        execution_evidence_ref=None,
        evaluation_ref=None,
        admitted_set_ref=None,
        intrinsic_receipt_ref=None,
        resolved_signal_slot_count=2,
        resolved_label_slot_count=0,
        slot_tree_sha256=payload["slot_tree_sha256"],
        terminal=False,
        blockers=[],
        last_stored_at=STAMP,
    )
    store.add(forged)
    with pytest.raises(FactorGovernanceError, match="projection differs"):
        replay_custody_chain(system_store=store, final_composite=forged)


def test_custody_replay_rejects_split_composite_namespace() -> None:
    store, composite1, _, records = _root_and_first_capture()
    payload = composite1["payload"]
    forged = build_composite_state(
        custody_namespace_id="different-factor-validation-namespace",
        preregistration_ref=payload["preregistration_ref"],
        cycle_state="PREREGISTERED",
        transaction_sequence=1,
        previous_composite_state_ref=None,
        transaction_id=payload["transaction_id"],
        custody_record_count=1,
        custody_head_ref=artifact_ref(records[0]),
        selection_ref=None,
        signal_capture_count=0,
        signal_capture_head_ref=None,
        observation_count=0,
        observation_head_ref=None,
        execution_evidence_ref=None,
        evaluation_ref=None,
        admitted_set_ref=None,
        intrinsic_receipt_ref=None,
        resolved_signal_slot_count=0,
        resolved_label_slot_count=0,
        slot_tree_sha256=custody_slot_tree_sha256([]),
        terminal=False,
        blockers=[],
        last_stored_at=STAMP,
    )
    store.add(forged)
    with pytest.raises(FactorGovernanceError, match="projection differs"):
        replay_custody_chain(system_store=store, final_composite=forged)


def test_stage_miss_has_one_state_and_exact_transaction_blockers() -> None:
    store, composite1, _, records = _root_and_first_capture()
    composite1_ref = artifact_ref(composite1)
    request = _request_sha("OBSERVE_SIGNAL", composite1_ref, artifact_ref(composite1))
    transaction = custody_transaction_id(
        custody_namespace_id=NAMESPACE,
        transaction_sequence=2,
        previous_composite_state_ref=composite1_ref,
        operation_request_sha256_value=request,
    )
    slot = build_stage_slot(
        stage="SIGNAL",
        ordinal=0,
        signal_session="2026-08-17",
        maturity_session=None,
        state="MISSED",
        subject_ref=None,
        blocker="SIGNAL_WINDOW_MISSED",
    )
    record = build_custody_record(
        custody_namespace_id=NAMESPACE,
        preregistration_id=composite1["payload"]["preregistration_ref"]["artifact_id"],
        sequence=1,
        previous_custody_ref=artifact_ref(records[0]),
        previous_composite_state_ref=composite1_ref,
        transaction_id=transaction,
        transaction_sequence=2,
        transaction_record_index=0,
        transaction_record_count=1,
        operation_request_sha256=request,
        operation="OBSERVE_SIGNAL",
        subject_refs=[],
        source_attestation_refs=[],
        stage_slot=slot,
        blockers=["SIGNAL_WINDOW_MISSED"],
        trusted_at=STAMP,
    )

    def _terminal(state: str, blockers: list[str]) -> dict[str, Any]:
        return build_composite_state(
            custody_namespace_id=NAMESPACE,
            preregistration_ref=composite1["payload"]["preregistration_ref"],
            cycle_state=state,
            transaction_sequence=2,
            previous_composite_state_ref=composite1_ref,
            transaction_id=transaction,
            custody_record_count=2,
            custody_head_ref=artifact_ref(record),
            selection_ref=None,
            signal_capture_count=0,
            signal_capture_head_ref=None,
            observation_count=0,
            observation_head_ref=None,
            execution_evidence_ref=None,
            evaluation_ref=None,
            admitted_set_ref=None,
            intrinsic_receipt_ref=None,
            resolved_signal_slot_count=1,
            resolved_label_slot_count=0,
            slot_tree_sha256=custody_slot_tree_sha256([slot]),
            terminal=True,
            blockers=blockers,
            last_stored_at=STAMP,
        )

    correct = _terminal("SIGNAL_CAPTURE_MISSED", ["SIGNAL_WINDOW_MISSED"])
    store.add(record, correct)
    assert replay_custody_chain(system_store=store, final_composite=correct).transaction_count == 2

    wrong_state = _terminal("TERMINAL_INCOMPLETE", ["SIGNAL_WINDOW_MISSED"])
    store.add(wrong_state)
    with pytest.raises(FactorGovernanceError, match="transition state differs"):
        replay_custody_chain(system_store=store, final_composite=wrong_state)

    wrong_blocker = _terminal("SIGNAL_CAPTURE_MISSED", ["ARBITRARY_BLOCKER"])
    store.add(wrong_blocker)
    with pytest.raises(FactorGovernanceError, match="blockers differ"):
        replay_custody_chain(system_store=store, final_composite=wrong_blocker)


def test_composite_validator_enforces_closed_128kib_bound() -> None:
    _, _, composite, _ = _root_and_first_capture()
    payload = copy.deepcopy(composite["payload"])
    payload["cycle_state"] = "TERMINAL_INCOMPLETE"
    payload["terminal"] = True
    payload["blockers"] = [f"B{index:05d}_" + "X" * 80 for index in range(2_000)]
    identity = {key: value for key, value in payload.items() if key != "composite_state_id"}
    payload["composite_state_id"] = business_identity("factor-composite-state", identity)
    oversized = seal_artifact("factor.composite_state", payload, created_at=STAMP)
    with pytest.raises(FactorGovernanceError, match="ARTIFACT_SIZE_LIMIT_EXCEEDED"):
        validate_composite_state(oversized)


def test_custody_validator_rejects_identity_and_transaction_forgery() -> None:
    _, _, _, records = _root_and_first_capture()
    payload = copy.deepcopy(records[-1]["payload"])
    payload["custody_record_id"] = "factor-custody-record-" + "f" * 64
    forged = seal_artifact("factor.custody_record", payload, created_at=STAMP)
    with pytest.raises(FactorGovernanceError, match="business identity"):
        validate_custody_record(forged)

    payload = copy.deepcopy(records[-1]["payload"])
    payload["transaction_id"] = "factor-custody-transaction-" + "f" * 64
    payload["custody_record_id"] = "factor-custody-record-" + "e" * 64
    forged = seal_artifact("factor.custody_record", payload, created_at=STAMP)
    with pytest.raises(FactorGovernanceError, match="transaction identity"):
        validate_custody_record(forged)
