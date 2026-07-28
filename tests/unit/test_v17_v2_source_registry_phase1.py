from __future__ import annotations

import copy

import pytest

from quant_investor.v17_v2_contract.resources import load_packaged_json
from quant_investor.v17_v2_contract.validators import (
    PORTFOLIO_OUTPUT_VERSION,
    RANK_OUTPUT_VERSION,
    V17V2ValidationError,
    require_runtime_usable_source_role_matrix,
    seal_semantic,
    validate_dataset_record_schema_registry,
    validate_portfolio_output,
    validate_rank_output,
)


def test_phase1_source_registries_are_exact_complete_and_runtime_usable() -> None:
    matrix = load_packaged_json("resources/source_role_matrix.v1.json")
    registry = load_packaged_json("resources/dataset_record_schema_registry.v1.json")

    assert require_runtime_usable_source_role_matrix(matrix) == matrix
    assert validate_dataset_record_schema_registry(registry) == registry
    assert matrix["completeness"] == "COMPLETE"
    assert matrix["runtime_usable"] is True
    assert matrix["pending_registry"] == []
    assert matrix["authority"] is False
    assert [record["record_schema_id"] for record in registry["records"]] == [
        "cn-open-day-calendar-dataset.v1",
        "corporate-actions-dataset.v1",
        "deep-evidence-dataset.v1",
        "fundamental-raw-tables-dataset.v1",
        "h00300-total-return-dataset.v1",
        "market-bars-dataset.v1",
        "official-delisting-cash-dataset.v1",
    ]


def test_dataset_registry_rejects_record_identity_or_key_drift() -> None:
    registry = copy.deepcopy(
        load_packaged_json("resources/dataset_record_schema_registry.v1.json")
    )
    registry["records"][0]["record_schema_id"] = "substituted.v1"
    with pytest.raises(V17V2ValidationError, match="does not derive from role"):
        validate_dataset_record_schema_registry(registry)

    registry = copy.deepcopy(
        load_packaged_json("resources/dataset_record_schema_registry.v1.json")
    )
    registry["records"][0]["primary_key"] = ["undeclared"]
    with pytest.raises(V17V2ValidationError, match="undeclared"):
        validate_dataset_record_schema_registry(registry)


def test_rank_and_portfolio_outputs_are_typed_and_cross_validated() -> None:
    rank = seal_semantic(
        {
            "protocol_version": "myquant.v17.v2",
            "version": RANK_OUTPUT_VERSION,
            "output_id": "rank-1",
            "run_id": "run-1",
            "strategy_id": "cn-shadow",
            "market": "CN",
            "cutoff": "2026-07-22T00:00:00Z",
            "status": "COMPLETE",
            "candidate_ordering": "rank-ascending-then-security_code-ascending",
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "security_code": "000001.SZ",
                    "rank": 1,
                    "fundamental_score": 0.75,
                    "fundamental_eligibility": "F_ELIGIBLE",
                    "quant_timing": "BUY_NOW",
                    "severe_red_flag": False,
                }
            ],
            "generated_at": "2026-07-22T00:01:00Z",
            "authority": False,
        }
    )
    assert validate_rank_output(rank) == rank

    ref = {
        "artifact_id": "input-1",
        "artifact_version": "myquant.v17.v2.portfolio-required-inputs.v1",
        "relative_path": (
            "data/private/v17_sources/protocol-v2/objects/"
            + "a" * 2
            + "/"
            + "a" * 64
            + ".json"
        ),
        "byte_sha256": "a" * 64,
        "semantic_sha256": "b" * 64,
    }
    risk_ref = {
        **ref,
        "artifact_id": "risk-1",
        "artifact_version": "myquant.v17.v2.risk-policy-snapshot.v1",
        "relative_path": (
            "data/private/v17_sources/protocol-v2/objects/"
            + "c" * 2
            + "/"
            + "c" * 64
            + ".json"
        ),
        "byte_sha256": "c" * 64,
    }
    portfolio = seal_semantic(
        {
            "protocol_version": "myquant.v17.v2",
            "version": PORTFOLIO_OUTPUT_VERSION,
            "output_id": "portfolio-1",
            "run_id": "run-1",
            "strategy_id": "cn-shadow",
            "market": "CN",
            "cutoff": "2026-07-22T00:00:00Z",
            "status": "FEASIBLE",
            "input_bindings": [
                {"role": "portfolio_required_inputs", "artifact_ref": ref},
                {"role": "risk_policy_snapshot", "artifact_ref": risk_ref},
            ],
            "position_ordering": "security_code-ascending",
            "positions": [
                {
                    "candidate_id": "candidate-1",
                    "security_code": "000001.SZ",
                    "target_weight": 0.5,
                }
            ],
            "cash_weight": 0.5,
            "blockers": [],
            "generated_at": "2026-07-22T00:02:00Z",
            "authority": False,
        }
    )
    assert validate_portfolio_output(portfolio) == portfolio
