from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from quant_investor.contracts import (
    artifact_byte_sha256,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
)
import quant_investor.factors.governance as governance
from quant_investor.factors.governance import (
    FactorGovernanceError,
    build_bootstrap_exception_evidence,
    build_bootstrap_factor_set,
    validate_factor_status,
)
from quant_investor.factors.governance.common import business_identity
from quant_investor.factors.governance.custody import build_composite_state
from quant_investor.factors.governance.status import _build_factor_status
from test_unified_factor_bootstrap import _evidence_inputs

STAMP = "2026-08-14T00:00:00Z"


def _ref(artifact: dict) -> dict[str, str]:
    return {
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "artifact_id": artifact["artifact_id"],
        "semantic_sha256": artifact["semantic_sha256"],
        "byte_sha256": artifact_byte_sha256(artifact),
    }


def _stub_ref(kind: str, artifact_id: str, marker: str = "1") -> dict[str, str]:
    return {
        "kind": kind,
        "contract_sha256": get_contract(kind).contract_sha256,
        "artifact_id": artifact_id,
        "semantic_sha256": marker * 64,
        "byte_sha256": marker * 64,
    }


def _reference_key(reference: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        reference[field]
        for field in ("kind", "contract_sha256", "artifact_id", "semantic_sha256", "byte_sha256")
    )


def _bootstrap_active_and_receipt() -> tuple[dict, dict, dict, list[dict]]:
    decision, sources, implementation_sha, _ = _evidence_inputs()
    policy = build_bootstrap_exception_evidence(
        decision_source_bytes=decision,
        source_artifacts=sources,
        implementation_source_sha256=implementation_sha,
        created_at=STAMP,
    )
    active = build_bootstrap_factor_set(
        bootstrap_exception_evidence=policy,
        created_at=STAMP,
    )
    evidence = list(sources.values())
    identity = {
        "policy_ref": _ref(policy),
        "evidence_refs": sorted((_ref(value) for value in evidence), key=_reference_key),
        "active_set_ref": _ref(active),
    }
    receipt = seal_artifact(
        "factor.validation_receipt",
        {
            "validation_receipt_id": business_identity("factor-validation", identity),
            **identity,
            "validated": True,
            "authority": "NON_AUTHORIZING",
        },
        created_at=STAMP,
    )
    return active, policy, receipt, evidence


def _context_and_attestation(
    active: dict,
    policy: dict,
    receipt: dict,
    evidence: list[dict],
) -> tuple[dict, dict]:
    contextual_component = _stub_ref(
        "system.installed_component_manifest", "contextual-component", "2"
    )
    decoder_component = _stub_ref("system.installed_component_manifest", "decoder-component", "3")
    implementation_components = [
        _stub_ref("system.installed_component_manifest", "implementation-low", "4"),
        _stub_ref("system.installed_component_manifest", "implementation-w80", "5"),
    ]
    factor_manifest = _stub_ref("factor.validator_manifest", "factor-manifest", "6")
    source_objects = [
        _stub_ref("system.source_object", f"source-{index}", "7") for index in range(7)
    ]
    context_payload = {
        "validation_namespace_id": "factor-bootstrap-namespace-test",
        "lane": "BOOTSTRAP",
        "intrinsic_receipt_ref": _ref(receipt),
        "policy_ref": _ref(policy),
        "evidence_refs": receipt["payload"]["evidence_refs"],
        "active_set_ref": _ref(active),
        "composite_state_ref": None,
        "factor_validator_manifest_ref": factor_manifest,
        "contextual_validator_component_ref": contextual_component,
        "source_decoder_component_ref": decoder_component,
        "implementation_component_refs": sorted(implementation_components, key=_reference_key),
        "source_attestation_refs": [],
        "source_object_refs": sorted(source_objects, key=_reference_key),
        "custody_record_refs": [],
        "custody_tree_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        "custody_head_ref": None,
        "validated": True,
        "blockers": [],
        "authority": "NON_AUTHORIZING",
    }
    context_payload["contextual_result_id"] = business_identity(
        "factor-contextual-result", context_payload
    )
    context = seal_artifact(
        "factor.contextual_validation_result",
        context_payload,
        created_at=STAMP,
    )
    release = next(value for value in evidence if value["kind"] == "system.release")
    attestation = seal_artifact(
        "system.validation_attestation",
        {
            "attestation_id": "validation-attestation-test",
            "validation_request_ref": _stub_ref(
                "system.validation_run_request", "validation-request", "8"
            ),
            "validation_profile_id": "factor-bootstrap-contextual-validation",
            "component_registry_sha256": "9" * 64,
            "validation_namespace_id": context_payload["validation_namespace_id"],
            "validation_lane": "BOOTSTRAP",
            "validation_intent_sha256": "a" * 64,
            "validation_plan_sha256": "b" * 64,
            "candidate_state_ref": None,
            "candidate_state_pointer_sha256": "c" * 64,
            "contextual_result_ref": _ref(context),
            "intrinsic_receipt_ref": _ref(receipt),
            "policy_ref": _ref(policy),
            "evidence_refs": receipt["payload"]["evidence_refs"],
            "active_set_ref": _ref(active),
            "source_object_refs": context_payload["source_object_refs"],
            "release_manifest_ref": _ref(release),
            "release_identity": {
                "release_id": release["payload"]["release_id"],
                "code_sha256": release["payload"]["code_sha256"],
                "wheel_sha256": release["payload"]["wheel_sha256"],
                "code_manifest_sha256": release["payload"]["code_manifest_sha256"],
            },
            "installed_code_manifest_sha256": "d" * 64,
            "compiled_contracts": [],
            "factor_validator_manifest_ref": factor_manifest,
            "contextual_validator_component_ref": contextual_component,
            "source_decoder_component_ref": decoder_component,
            "implementation_component_refs": context_payload["implementation_component_refs"],
            "source_attestation_refs": [],
            "custody_record_refs": [],
            "custody_head_ref": None,
            "custody_tree_sha256": context_payload["custody_tree_sha256"],
            "factor_source_stat_tree_sha256": "e" * 64,
            "factor_source_total_bytes": 7,
            "maximum_total_factor_source_bytes": 2 * 1024**3,
            "validated_at": STAMP,
            "clock_source": "SYSTEM_UTC",
            "outcome": "VALIDATED",
            "authority": "NON_AUTHORIZING",
        },
        created_at=STAMP,
    )
    return context, attestation


def test_minimal_factor_status_is_blocked_and_non_authorizing() -> None:
    status = _build_factor_status(
        active_factor_set=None,
        active_validation_receipt=None,
        active_contextual_result=None,
        active_validation_attestation=None,
        observed_composite_state=None,
        trusted_at=STAMP,
    )
    payload = validate_factor_status(status)["payload"]
    assert payload["readiness"] == "BLOCKED"
    assert payload["blockers"] == ["ACTIVE_FACTOR_SET_ABSENT"]
    assert payload["activation_mutation_authorized"] is False
    assert payload["active"] == {
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
    assert payload["observed"] == {
        "composite_state_ref": None,
        "cycle_state": "NOT_STARTED",
        "terminal": False,
        "blockers": [],
    }


def test_ready_status_requires_receipt_context_and_system_attestation() -> None:
    active, policy, receipt, evidence = _bootstrap_active_and_receipt()
    context, attestation = _context_and_attestation(active, policy, receipt, evidence)
    status = _build_factor_status(
        active_factor_set=active,
        active_validation_receipt=receipt,
        active_contextual_result=context,
        active_validation_attestation=attestation,
        observed_composite_state=None,
        trusted_at=STAMP,
    )
    payload = validate_factor_status(status)["payload"]
    assert payload["readiness"] == "READY"
    assert payload["blockers"] == []
    assert payload["active"]["lane"] == "BOOTSTRAP"
    assert payload["active"]["contextual_result_ref"] == _ref(context)
    assert payload["active"]["validation_attestation_ref"] == _ref(attestation)

    with pytest.raises(FactorGovernanceError, match="closure is incomplete"):
        _build_factor_status(
            active_factor_set=active,
            active_validation_receipt=receipt,
            active_contextual_result=None,
            active_validation_attestation=None,
            observed_composite_state=None,
            trusted_at=STAMP,
        )


def test_ready_status_rejects_context_or_attestation_ref_mismatch() -> None:
    active, policy, receipt, evidence = _bootstrap_active_and_receipt()
    context, attestation = _context_and_attestation(active, policy, receipt, evidence)
    forged_payload = deepcopy(context["payload"])
    forged_payload["active_set_ref"] = _stub_ref("factor.bootstrap_set", "other-set", "f")
    identity = {
        field: value for field, value in forged_payload.items() if field != "contextual_result_id"
    }
    forged_payload["contextual_result_id"] = business_identity("factor-contextual-result", identity)
    forged = seal_artifact(
        "factor.contextual_validation_result",
        forged_payload,
        created_at=STAMP,
    )
    with pytest.raises(FactorGovernanceError, match="contextual closure differs"):
        _build_factor_status(
            active_factor_set=active,
            active_validation_receipt=receipt,
            active_contextual_result=forged,
            active_validation_attestation=attestation,
            observed_composite_state=None,
            trusted_at=STAMP,
        )


def test_observed_projection_is_only_the_authoritative_composite_state() -> None:
    composite = build_composite_state(
        custody_namespace_id="factor-validation-namespace-test",
        preregistration_ref=_stub_ref("factor.preregistration", "preregistration", "1"),
        cycle_state="PREREGISTERED",
        transaction_sequence=1,
        previous_composite_state_ref=None,
        transaction_id="factor-custody-transaction-" + "2" * 64,
        custody_record_count=1,
        custody_head_ref=_stub_ref("factor.custody_record", "custody-zero", "3"),
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
        slot_tree_sha256="4" * 64,
        terminal=False,
        blockers=[],
        last_stored_at=STAMP,
    )
    status = _build_factor_status(
        active_factor_set=None,
        active_validation_receipt=None,
        active_contextual_result=None,
        active_validation_attestation=None,
        observed_composite_state=composite,
        trusted_at=STAMP,
    )
    assert validate_factor_status(status)["payload"]["observed"] == {
        "composite_state_ref": _ref(composite),
        "cycle_state": "PREREGISTERED",
        "terminal": False,
        "blockers": [],
    }


def test_old_status_builder_surface_is_absent() -> None:
    assert not hasattr(governance, "build_factor_status")
