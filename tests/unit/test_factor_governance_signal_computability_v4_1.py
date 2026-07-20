from __future__ import annotations

import copy
import hashlib

import pytest

from quant_investor.factors import governance_signal_computability_v4_1 as subject
from quant_investor.monitoring import v16_run_readiness


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _descriptor(label: str) -> dict[str, object]:
    return {
        "contract": "factor-no-label-matrix-f64-le.v1",
        "shape": [1227, 5866],
        "dtype": "float64-little-endian",
        "date_axis_sha256": _digest("date-axis"),
        "symbol_axis_sha256": _digest("symbol-axis"),
        "matrix_sha256": _digest(label),
    }


def _binding(binding_id: str) -> dict[str, object]:
    return {
        "binding_id": binding_id,
        "absolute_path": f"/private/{binding_id}.json",
        "byte_sha256": _digest(f"bytes:{binding_id}"),
        "semantic_sha256": _digest(f"semantic:{binding_id}"),
    }


def _receipt() -> dict[str, object]:
    bars = {
        "tree_oid": subject.EXPECTED_BARS_TREE_OID,
        "full_inventory_sha256": subject.EXPECTED_BARS_FULL_INVENTORY_SHA256,
        "selected_inventory_sha256": subject.EXPECTED_BARS_SELECTED_INVENTORY_SHA256,
        "projected_ipc_sha256": subject.EXPECTED_BARS_IPC_SHA256,
        **subject.EXPECTED_BARS_ACCOUNTING,
    }
    financial = {
        "tree_oid": subject.EXPECTED_FINANCIAL_TREE_OID,
        "inventory_sha256": subject.EXPECTED_FINANCIAL_INVENTORY_SHA256,
        "physical_schema_manifest_sha256": (
            subject.EXPECTED_FINANCIAL_SCHEMA_MANIFEST_SHA256
        ),
        "logical_ipc_sha256": subject.EXPECTED_FINANCIAL_IPC_SHA256,
        **subject.EXPECTED_FINANCIAL_ACCOUNTING,
    }
    return subject.build_input_semantics_receipt_v4_1(
        baseline_bindings=[
            _binding("execution_baseline"),
            _binding("worktree_content_baseline"),
        ],
        code_bindings=[_binding("builder"), _binding("contract")],
        predecessor_bindings=[_binding("no_label"), _binding("operator")],
        git_identity={
            "executable_sha256": _digest("git"),
            "replacement_objects_disabled": True,
            "lazy_fetch_disabled": True,
        },
        aquant_source_blobs=[
            {
                "path": f"A_quant/source_{index}.py",
                "mode": "100644",
                "type": "blob",
                "oid": f"{index + 1:040x}",
                "size": index + 1,
                "sha256": _digest(f"source:{index}"),
            }
            for index in range(3)
        ],
        bars_inventory=bars,
        calendar_accounting=copy.deepcopy(subject.EXPECTED_CALENDAR_ACCOUNTING),
        financial_inventory=financial,
        transformation_contract={
            "ast_manifest_sha256": subject.EXPECTED_TRANSFORMATION_AST_MANIFEST_SHA256,
            "child_exec_event_count": 1,
            "child_filesystem_access_after_audit": False,
            "child_network_access_after_audit": False,
            "child_parent_all_match": True,
            "descriptor_contract": "factor-no-label-matrix-f64-le.v1",
        },
        runtime_identity={"runtime_semantic_sha256": _digest("runtime")},
        source_partition=copy.deepcopy(subject.SOURCE_PARTITION),
        resource_limits=copy.deepcopy(subject.EXPECTED_RESOURCE_LIMITS),
        protected_contexts=copy.deepcopy(subject.EXPECTED_PROTECTED_CONTEXTS),
    )


def _candidate_rows() -> list[dict[str, object]]:
    names = [f"alpha_dummy_{index:02d}" for index in range(27)] + list(
        subject.NEWLY_COMPUTED_NAMES
    )
    rows = []
    for name in sorted(names):
        newly_computed = name in subject.NEWLY_COMPUTED_NAMES
        predecessor_status = "no_label_signal_eval_diagnostic"
        if newly_computed:
            predecessor_status = (
                "turnover_data_blocked"
                if name.startswith("alpha_turnover")
                else "fundamental_semantic_blocked"
            )
        row = {
            "candidate_id": f"candidate:{name}",
            "name": name,
            "input_fields": ["turnover_rate"] if name.startswith("alpha_turnover") else ["close"],
            "source_definition_sha256": _digest(f"source:{name}"),
            "catalog_definition_sha256": _digest(f"catalog:{name}"),
            "mapping_semantic_sha256": _digest(f"mapping:{name}"),
            "normalized_ast_sha256": _digest(f"ast:{name}"),
            "predecessor_status": predecessor_status,
            "predecessor_descriptor_preserved": not newly_computed,
            "status": "source_semantic_computability_verified",
            "eligible_cell_count": subject.EXPECTED_ELIGIBLE_CELL_COUNT,
            "finite_count": 1,
            "nan_count": subject.EXPECTED_ELIGIBLE_CELL_COUNT - 1,
            "positive_inf_count": 0,
            "negative_inf_count": 0,
            "outside_mask_non_nan_count": 0,
            "signal_matrix": _descriptor(name),
        }
        row["row_semantic_sha256"] = subject.semantic_sha256_v4_1(row)
        rows.append(row)
    return rows


def _primitive_rows() -> list[dict[str, object]]:
    return [
        {
            "field": field,
            "source": "fixture",
            "matrix": _descriptor(field),
            "finite_count": 1,
            "nan_count": subject.EXPECTED_ELIGIBLE_CELL_COUNT - 1,
            "positive_inf_count": 0,
            "negative_inf_count": 0,
            "outside_mask_non_nan_count": 0,
        }
        for field in subject.PRIMITIVE_NAMES
    ]


def _proof() -> tuple[dict[str, object], dict[str, object]]:
    receipt = _receipt()
    primitives = _primitive_rows()
    rows = _candidate_rows()
    manifest = subject.semantic_sha256_v4_1(
        {"primitive_matrices": primitives, "rows": rows}
    )
    passes = [
        {
            "pass_id": pass_id,
            "result_manifest_sha256": manifest,
            "runtime_semantic_sha256": _digest("runtime"),
            "child_parent_all_match": True,
            "outside_mask_all_zero": True,
            "candidate_count": 37,
        }
        for pass_id in ("first", "fresh_readback")
    ]
    proof = subject.build_signal_computability_proof_v4_1(
        semantics_receipt=receipt,
        predecessor_proof_bindings=[_binding("no_label"), _binding("operator")],
        computation_passes=passes,
        primitive_matrices=primitives,
        rows=rows,
    )
    return receipt, proof


def _reseal(value: dict[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = subject.semantic_sha256_v4_1(result)
    return result


def test_exact_37_computability_is_bounded_and_nonauthorizing() -> None:
    receipt, proof = _proof()

    assert receipt["calendar_accounting"] == subject.EXPECTED_CALENDAR_ACCOUNTING
    assert proof["signal_computability_proven"] is True
    assert proof["candidate_count"] == 37
    assert proof["predecessor_preserved_count"] == 27
    assert proof["newly_computed_count"] == 10
    assert proof["claim_negatives"] == subject.CLAIM_NEGATIVES
    assert all(value is False for value in proof["side_effects"].values())
    assert proof["screening_authority"] is False
    assert proof["factor_apply_authority"] is False
    assert proof["portfolio_authority"] is False
    assert proof["new_risk_authorized"] is False


@pytest.mark.parametrize(
    "delta",
    (-1, 1),
)
def test_calendar_deficiency_drift_is_rejected(delta: int) -> None:
    receipt = _receipt()
    receipt["calendar_accounting"]["off_myquant_calendar_count"] += delta
    receipt = _reseal(receipt, "receipt_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_1Error,
        match="calendar accounting mismatch",
    ):
        subject.validate_input_semantics_receipt_v4_1(receipt)


def test_claim_negative_cannot_be_promoted() -> None:
    _, proof = _proof()
    proof["claim_negatives"]["source_same_snapshot"] = True
    proof = _reseal(proof, "proof_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_1Error,
        match="claim negatives mismatch",
    ):
        subject.validate_signal_computability_proof_v4_1(proof)


def test_resource_limit_reseal_is_rejected() -> None:
    receipt = _receipt()
    receipt["resource_limits"]["max_parent_rss_bytes"] += 1
    receipt = _reseal(receipt, "receipt_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_1Error,
        match="resource limits mismatch",
    ):
        subject.validate_input_semantics_receipt_v4_1(receipt)


def test_protected_context_reseal_is_rejected() -> None:
    receipt = _receipt()
    receipt["protected_contexts"]["same_snapshot_screening_bundle"] = (
        "proof_input"
    )
    receipt = _reseal(receipt, "receipt_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_1Error,
        match="protected contexts mismatch",
    ):
        subject.validate_input_semantics_receipt_v4_1(receipt)


def test_exact_ten_row_mismatch_against_fresh_pass_is_rejected() -> None:
    _, proof = _proof()
    target = next(
        row for row in proof["rows"] if row["name"] in subject.NEWLY_COMPUTED_NAMES
    )
    target["signal_matrix"]["matrix_sha256"] = "0" * 64
    target_without_hash = {
        key: copy.deepcopy(value)
        for key, value in target.items()
        if key != "row_semantic_sha256"
    }
    target["row_semantic_sha256"] = subject.semantic_sha256_v4_1(
        target_without_hash
    )
    proof["result_manifest_sha256"] = subject.semantic_sha256_v4_1(
        {"primitive_matrices": proof["primitive_matrices"], "rows": proof["rows"]}
    )
    proof = _reseal(proof, "proof_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_1Error,
        match="fresh recomputation result manifest mismatch|result manifest mismatch",
    ):
        subject.validate_signal_computability_proof_v4_1(proof)


def test_computability_schema_cannot_satisfy_v16_factor_readiness() -> None:
    _, proof = _proof()

    _, _, ready, _, blockers = v16_run_readiness._factor_contract(proof)

    assert ready is False
    assert "factor_readiness_schema_not_v4" in blockers
