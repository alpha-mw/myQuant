from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes, seal_semantic
from quant_investor.v17_v4_contract.schema_validation import (
    SchemaValidationError,
    artifact_identity_field,
    preflight_schema,
    schema_path_for_version,
    validate_instance_against_schema,
)
from quant_investor.v17_v4_contract.validators import (
    ArtifactContractError,
    validate_typed_artifact,
)
from quant_investor.v17_v4_runtime.forward_evidence import _allocation_roles

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "quant_investor" / "v17_v4_contract" / "schemas"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
STRATEGY_ID = "cn.forward"
CUTOFF = "2026-07-29T07:00:00Z"
SESSION = "2026-07-29"
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
FORWARD_SCHEMAS = {
    "myquant.v17.v4.existing-factor-inventory.v1": (
        "existing_factor_inventory.v1.schema.json",
        "inventory_id",
    ),
    "myquant.v17.v4.factor-universe-observation.v1": (
        "factor_universe_observation.v1.schema.json",
        "observation_id",
    ),
    "myquant.v17.v4.forward-evaluation-receipt.v1": (
        "forward_evaluation_receipt.v1.schema.json",
        "receipt_id",
    ),
    "myquant.v17.v4.forward-evidence-origin-inventory.v1": (
        "forward_evidence_origin_inventory.v1.schema.json",
        "inventory_id",
    ),
    "myquant.v17.v4.forward-factor-allocation.v1": (
        "forward_factor_allocation.v1.schema.json",
        "allocation_id",
    ),
    "myquant.v17.v4.forward-label.v1": (
        "forward_label.v1.schema.json",
        "label_id",
    ),
    "myquant.v17.v4.forward-observation-run.v1": (
        "forward_observation_run.v1.schema.json",
        "observation_run_id",
    ),
    "myquant.v17.v4.forward-observation-session-ref.v1": (
        "forward_observation_session_ref.v1.schema.json",
        "session_ref_id",
    ),
    "myquant.v17.v4.forward-run-request.v1": (
        "forward_run_request.v1.schema.json",
        "request_id",
    ),
    "myquant.v17.v4.forward-runtime-source-manifest.v1": (
        "forward_runtime_source_manifest.v1.schema.json",
        "manifest_id",
    ),
    "myquant.v17.v4.forward-stage-receipt.v1": (
        "forward_stage_receipt.v1.schema.json",
        "receipt_id",
    ),
    "myquant.v17.v4.forward-stage-output.v1": (
        "forward_stage_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v4.strategy-pool-observation.v1": (
        "strategy_pool_observation.v1.schema.json",
        "observation_id",
    ),
}


def _schema(filename: str) -> dict[str, object]:
    return json.loads((SCHEMA_ROOT / filename).read_text(encoding="utf-8"))


def _ref(
    artifact_id: str,
    artifact_version: str,
    relative_path: str,
    *,
    byte_sha256: str = SHA_A,
    semantic_sha256: str = SHA_B,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": byte_sha256,
        "cutoff": CUTOFF,
        "relative_path": relative_path,
        "semantic_sha256": semantic_sha256,
        "strategy_id": STRATEGY_ID,
    }


def _request() -> dict[str, object]:
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "factor_refs": [
                _ref(
                    "factor.one",
                    "myquant.v17.v4.factor-definition.v1",
                    "factors/factor_one.json",
                )
            ],
            "protocol_version": "myquant.v17.v4",
            "request_id": "forward.request.1",
            "request_profile": "FORWARD_EVIDENCE",
            "source_refs": [
                _ref(
                    "source.one",
                    "myquant.v17.v4.source-snapshot.v1",
                    "sources/source_one.json",
                )
            ],
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.forward-run-request.v1",
        }
    )


@pytest.mark.parametrize(
    ("version", "schema_file", "identity_field"),
    [
        (version, schema_file, identity_field)
        for version, (schema_file, identity_field) in FORWARD_SCHEMAS.items()
    ],
)
def test_forward_schemas_are_closed_canonical_and_registered(
    version: str,
    schema_file: str,
    identity_field: str,
) -> None:
    raw = (SCHEMA_ROOT / schema_file).read_bytes()
    schema = json.loads(raw)

    assert raw == canonical_resource_bytes(schema)
    preflight_schema(schema)
    assert schema_path_for_version(version) == f"schemas/{schema_file}"
    assert artifact_identity_field(version) == identity_field


def test_forward_request_accepts_exact_refs_and_closed_profile() -> None:
    request = _request()

    validate_instance_against_schema(
        request,
        _schema("forward_run_request.v1.schema.json"),
    )
    validated = validate_typed_artifact(request, schema_checked=True)

    assert validated.version == "myquant.v17.v4.forward-run-request.v1"


def test_forward_factor_allocation_uses_tier_and_selected_rows() -> None:
    request = _request()
    allocation = seal_semantic(
        {
            "allocation_id": "forward.allocation.1",
            "allocations": [
                {
                    "allocation_weight": "0.6",
                    "factor_id": "factor.core",
                    "factor_ref": _ref(
                        "factor.core",
                        "myquant.v17.v4.factor-definition.v1",
                        "factors/core.json",
                    ),
                    "factor_tier": "CORE",
                    "selected": True,
                },
                {
                    "allocation_weight": "0.4",
                    "factor_id": "factor.experimental",
                    "factor_ref": _ref(
                        "factor.experimental",
                        "myquant.v17.v4.factor-definition.v1",
                        "factors/experimental.json",
                    ),
                    "factor_tier": "EXPERIMENTAL",
                    "selected": False,
                },
            ],
            "authority": dict(NO_AUTHORITY),
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "evidence_origin_inventory_ref": _ref(
                "origin.inventory.1",
                "myquant.v17.v4.forward-evidence-origin-inventory.v1",
                "inventories/origin.json",
            ),
            "existing_factor_inventory_ref": _ref(
                "factor.inventory.1",
                "myquant.v17.v4.existing-factor-inventory.v1",
                "inventories/factors.json",
            ),
            "protocol_version": "myquant.v17.v4",
            "request_ref": _ref(
                str(request["request_id"]),
                str(request["version"]),
                "requests/forward_request.json",
                semantic_sha256=str(request["semantic_sha256"]),
            ),
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.forward-factor-allocation.v1",
        }
    )

    validated = validate_typed_artifact(allocation)

    assert validated.version == "myquant.v17.v4.forward-factor-allocation.v1"
    assert _allocation_roles(allocation) == ["CORE"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unexpected": True}),
        lambda value: value.update({"request_profile": "PRODUCTION"}),
        lambda value: value["authority"].update({"execution": True}),
    ],
)
def test_forward_request_rejects_additional_bad_enum_and_true_authority(
    mutation: object,
) -> None:
    request = _request()
    request.pop("semantic_sha256")
    mutation(request)
    request = seal_semantic(request)

    with pytest.raises(SchemaValidationError):
        validate_instance_against_schema(
            request,
            _schema("forward_run_request.v1.schema.json"),
        )


def test_stage_receipt_keeps_outcome_and_completeness_separate() -> None:
    request = _request()
    receipt = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "blockers": ["source_gap"],
            "completeness": "PARTIAL",
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "execution_outcome": "BLOCKED",
            "output_refs": [],
            "protocol_version": "myquant.v17.v4",
            "receipt_id": "forward.stage.receipt.1",
            "recorded_at": CUTOFF,
            "request_ref": _ref(
                str(request["request_id"]),
                str(request["version"]),
                "requests/forward_request.json",
                semantic_sha256=str(request["semantic_sha256"]),
            ),
            "stage_id": "factor_observation",
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.forward-stage-receipt.v1",
        }
    )

    validate_instance_against_schema(
        receipt,
        _schema("forward_stage_receipt.v1.schema.json"),
    )
    validate_typed_artifact(receipt, schema_checked=True)

    invalid = deepcopy(receipt)
    invalid.pop("semantic_sha256")
    invalid["execution_outcome"] = "PARTIAL"
    with pytest.raises(SchemaValidationError):
        validate_instance_against_schema(
            seal_semantic(invalid),
            _schema("forward_stage_receipt.v1.schema.json"),
        )


def test_stage_output_is_closed_and_schema_validated() -> None:
    request = _request()
    payload = {"rows": 3}
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    output = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "completeness": "COMPLETE",
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "lineage_receipt_refs": [],
            "output_id": "forward.stage.output.1",
            "payload_json": payload_json,
            "payload_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
            "protocol_version": "myquant.v17.v4",
            "recorded_at": CUTOFF,
            "request_ref": _ref(
                str(request["request_id"]),
                str(request["version"]),
                "requests/forward_request.json",
                semantic_sha256=str(request["semantic_sha256"]),
            ),
            "stage_id": "quant",
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.forward-stage-output.v1",
        }
    )

    validate_instance_against_schema(
        output,
        _schema("forward_stage_output.v1.schema.json"),
    )
    validate_typed_artifact(output, schema_checked=True)

    invalid = deepcopy(output)
    invalid.pop("semantic_sha256")
    invalid["unexpected"] = False
    with pytest.raises(SchemaValidationError):
        validate_instance_against_schema(
            seal_semantic(invalid),
            _schema("forward_stage_output.v1.schema.json"),
        )

    hash_mismatch = deepcopy(output)
    hash_mismatch.pop("semantic_sha256")
    hash_mismatch["payload_sha256"] = SHA_A
    with pytest.raises(ArtifactContractError, match="payload_json/hash"):
        validate_typed_artifact(
            seal_semantic(hash_mismatch),
            schema_checked=True,
        )


def test_origin_inventory_rejects_duplicate_conflict_as_unpublishable() -> None:
    request = _request()
    inventory = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "inventory_id": "forward.origin.inventory.1",
            "origins": [
                {
                    "canonical_evidence_ref": _ref(
                        "evidence.a",
                        "myquant.v17.v4.forward-label.v1",
                        "evidence/a.json",
                        byte_sha256=SHA_A,
                    ),
                    "duplicate_origin_status": "DUPLICATE_CONFLICT",
                    "evidence_refs": [
                        _ref(
                            "evidence.a",
                            "myquant.v17.v4.forward-label.v1",
                            "evidence/a.json",
                            byte_sha256=SHA_A,
                        ),
                        _ref(
                            "evidence.b",
                            "myquant.v17.v4.forward-label.v1",
                            "evidence/b.json",
                            byte_sha256=SHA_C,
                        ),
                    ],
                    "lineage_key": {
                        "factor_definition_sha256": SHA_A,
                        "factor_name": "factor.a",
                        "factor_set_sha256": SHA_B,
                        "horizon_sessions": 20,
                        "quant_policy_sha256": SHA_C,
                        "source_lineage_sha256": SHA_A,
                    },
                    "lineage_key_sha256": SHA_B,
                    "origin": SESSION,
                }
            ],
            "production_default_eligible": False,
            "promotion_eligible": False,
            "protocol_version": "myquant.v17.v4",
            "provider_authority": False,
            "provider_invoked": False,
            "request_ref": _ref(
                str(request["request_id"]),
                str(request["version"]),
                "requests/forward_request.json",
                semantic_sha256=str(request["semantic_sha256"]),
            ),
            "shadow_only": True,
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.forward-evidence-origin-inventory.v1",
        }
    )

    with pytest.raises(SchemaValidationError, match="closed enum"):
        validate_instance_against_schema(
            inventory,
            _schema("forward_evidence_origin_inventory.v1.schema.json"),
        )
