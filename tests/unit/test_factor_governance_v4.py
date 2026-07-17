from __future__ import annotations

import hashlib

from quant_investor.factors.governance_protocol_v4 import (
    CONTROL_CHAIN_STAGES,
    FactorGovernanceV4Error,
    assess_candidate_admission_v4,
    assess_candidate_maturity,
    assess_factor_governance_readiness_v4,
    assess_governance_cycle_v4,
    build_health_action_proposal_v4,
    protocol_hash,
    protocol_policy,
    semantic_sha256,
    validate_candidate_admission_v4,
)
import pytest
from quant_investor.factors.governance_transaction_v4 import (
    activation_receipt_sha256,
)
from quant_investor.factors.runtime import production_factor_set_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _record(index: int, *, count: int) -> dict:
    name = f"factor_{index}"
    if count == 5:
        family = f"family_{index}"
    else:
        family = f"family_{index // 2}"
    runtime_contract = {
        "schema_version": "factor-production-runtime-contract.v4",
        "factor_name": name,
    }
    return {
        "name": name,
        "family": family,
        "slot": f"{family}::slot_{index}",
        "state": "production_factor",
        "weight": 1.0,
        "gate_results": {str(gate_id): True for gate_id in range(1, 9)},
        "maturity": {
            "month_end_rankic_dates": [f"2025-{month:02d}-28" for month in range(1, 13)],
            "forward_cohorts": [],
        },
        "bh_q_value": 0.05,
        "fdr_method": "benjamini_hochberg_by_family",
        "runtime_contract": runtime_contract,
        "runtime_contract_sha256": semantic_sha256(runtime_contract),
        "runtime_contract_status": "verified",
        "evidence": {
            "schema_version": "factor-governance-replay-evidence.v4",
            "status": "verified",
            "replay_semantic_sha256": _digest(f"replay:{name}"),
        },
        "health": {"status": "healthy", "fresh": True, "data_blocked": False},
    }


def _receipt(records: list[dict], *, factor_set_sha: str) -> dict:
    runtime_sha = semantic_sha256(sorted(record["runtime_contract_sha256"] for record in records))
    context = {
        "protocol_hash": protocol_hash(),
        "transaction_plan_sha256": _digest("transaction-plan"),
        "registry_file_sha256": _digest("registry"),
        "production_factor_set_sha256": factor_set_sha,
        "runtime_contracts_sha256": runtime_sha,
        "as_of": "2026-07-17",
    }
    payload = {
        "schema_version": "factor-governance-activation-receipt.v4",
        "protocol_version": "v4",
        "protocol_hash": protocol_hash(),
        "receipt_id": "receipt-test",
        "status": "activated",
        "authorization_scope": "factor_v4_production_activation",
        "authorized_by": "Maxwell",
        "activated_at": "2026-07-17T09:00:00+08:00",
        "as_of": "2026-07-17",
        "transaction_plan_sha256": context["transaction_plan_sha256"],
        "registry_file_sha256": context["registry_file_sha256"],
        "production_factor_set_sha256": factor_set_sha,
        "runtime_contracts_sha256": runtime_sha,
        "activation_context_sha256": semantic_sha256(context),
        "activation_performed": True,
    }
    payload["receipt_sha256"] = activation_receipt_sha256(payload)
    return payload


def _readiness(records: list[dict], receipt: dict | None = None) -> dict:
    factor_set_sha = production_factor_set_sha256(sorted(record["name"] for record in records))
    return assess_factor_governance_readiness_v4(
        records,
        as_of="2026-07-17",
        registry_file_sha256=_digest("registry"),
        production_factor_set_sha256=factor_set_sha,
        activation_receipt=receipt or _receipt(records, factor_set_sha=factor_set_sha),
    )


def test_v4_policy_is_isolated_plan_only_and_uses_exact_eight_stage_chain() -> None:
    policy = protocol_policy()
    assert policy["canonical_chain"] == list(CONTROL_CHAIN_STAGES)
    assert CONTROL_CHAIN_STAGES == (
        "eligibility",
        "quant",
        "funnel",
        "codex_s1",
        "bayesian",
        "risk_advisor",
        "codex_ic",
        "portfolio_constructor",
    )
    assert policy["legacy_evidence"] == {
        "v2": "reject",
        "v3": "reject",
        "auto_upgrade": False,
    }
    assert policy["transaction"]["production_apply_enabled"] is False
    assert policy["risk_advisor"]["positive_weight_requires_approval"] is False


def test_v4_maturity_accepts_12_months_or_8_nonoverlap_cohorts() -> None:
    month_route = assess_candidate_maturity(
        month_end_rankic_dates=[f"2025-{month:02d}-28" for month in range(1, 13)],
        forward_cohorts=[],
    )
    assert month_route["mature"] is True
    assert month_route["maturity_route"] == "month_end_rankic"

    cohorts = [
        {
            "cohort_id": f"cohort-{index}",
            "start": f"2025-{index + 1:02d}-01",
            "end": f"2025-{index + 1:02d}-28",
            "horizon_days": 30,
        }
        for index in range(8)
    ]
    cohort_route = assess_candidate_maturity(month_end_rankic_dates=[], forward_cohorts=cohorts)
    assert cohort_route["mature"] is True
    assert cohort_route["maturity_route"] == "nonoverlap_30d_forward_cohort"


def test_five_healthy_factors_require_five_families_and_accelerated_mining() -> None:
    records = [_record(index, count=5) for index in range(5)]
    readiness = _readiness(records)
    assert readiness["status"] == "underfilled_accelerated_mining"
    assert readiness["factor_governance_ready"] is True
    assert readiness["target_ready"] is False
    assert readiness["accelerated_mining_required"] is True
    assert readiness["new_risk_eligible"] is True
    assert readiness["new_risk_authorized"] is False
    assert max(readiness["normalized_abs_weights"].values()) == 0.2

    duplicate_family = [_record(index, count=5) for index in range(5)]
    duplicate_family[1]["family"] = duplicate_family[0]["family"]
    duplicate_family[1]["slot"] = "family_0::different-slot"
    blocked = _readiness(duplicate_family)
    assert blocked["status"] == "no_new_risk"
    assert "exact_five_requires_five_distinct_families" in blocked["blockers"]


def test_target_10_is_ready_but_scaffold_never_authorizes_new_risk() -> None:
    records = [_record(index, count=10) for index in range(10)]
    readiness = _readiness(records)
    assert readiness["status"] == "ready_target_10"
    assert readiness["healthy_factor_count"] == 10
    assert readiness["production_family_count"] == 5
    assert readiness["target_ready"] is True
    assert readiness["new_risk_authorized"] is False
    assert readiness["production_apply_enabled"] is False


def test_missing_receipt_and_data_blocked_both_fail_closed_without_alpha_failure() -> None:
    records = [_record(index, count=5) for index in range(5)]
    missing_receipt = assess_factor_governance_readiness_v4(
        records,
        as_of="2026-07-17",
        registry_file_sha256=_digest("registry"),
        production_factor_set_sha256=production_factor_set_sha256(
            sorted(record["name"] for record in records)
        ),
        activation_receipt=None,
    )
    assert missing_receipt["status"] == "no_new_risk"
    assert "activation_receipt_missing" in missing_receipt["blockers"]

    records[0]["health"] = {
        "status": "data_blocked",
        "fresh": True,
        "data_blocked": True,
    }
    blocked = _readiness(records)
    assert blocked["status"] == "no_new_risk"
    factor = next(row for row in blocked["factors"] if row["name"] == "factor_0")
    assert factor["data_blocked"] is True
    assert factor["data_blocked_counts_as_alpha_failure"] is False


def test_weekly_is_report_only_and_target_replacement_needs_positive_lower_bound() -> None:
    weekly = assess_governance_cycle_v4(
        cadence="weekly",
        production_factor_count=8,
        proposals=[{"action": "watch_proposal"}],
    )
    assert weekly["status"] == "proposal_blocked"
    assert "weekly_cycle_is_report_only" in weekly["blockers"]

    rejected = assess_governance_cycle_v4(
        cadence="month_end",
        production_factor_count=10,
        proposals=[
            {
                "action": "replace_proposal",
                "incumbent": "old",
                "challenger": "new",
                "incremental_edge_ci95_lower": 0.0,
            }
        ],
    )
    assert rejected["status"] == "proposal_blocked"
    accepted = assess_governance_cycle_v4(
        cadence="month_end",
        production_factor_count=10,
        proposals=[
            {
                "action": "replace_proposal",
                "incumbent": "old",
                "challenger": "new",
                "incremental_edge_ci95_lower": 0.001,
            }
        ],
    )
    assert accepted["status"] == "proposal_ready"
    assert accepted["proposal_only"] is True
    assert accepted["production_apply_enabled"] is False


def _candidate() -> dict:
    return {
        "name": "candidate-a",
        "family": "value",
        "slot": "value::primary",
        "data_source_receipt": {
            "schema_version": "factor-v4-strict-parquet-source-receipt.v1",
            "backend": "parquet",
            "mode_policy": "strict",
            "snapshot_healthy": True,
            "universe_scope": "full_a",
            "snapshot_id": "snapshot-test",
            "latest_pointer_sha256": _digest("pointer"),
            "manifest_sha256": _digest("manifest"),
        },
        "full_a_universe_verified": True,
        "duplicate_primitive": False,
        "high_correlation_dedup_passed": True,
        "initial_weight": 0.0,
        "maturity": {
            "month_end_rankic_dates": [f"2025-{month:02d}-28" for month in range(1, 13)],
            "forward_cohorts": [],
        },
        "bh_q_value": 0.05,
        "fdr_method": "benjamini_hochberg_by_family",
        "gate_results": {str(gate_id): True for gate_id in range(1, 9)},
        "evidence": {
            "schema_version": "factor-governance-replay-evidence.v4",
            "status": "verified",
            "replay_semantic_sha256": _digest("candidate-replay"),
        },
    }


def test_candidate_admission_is_machine_checkable_and_fail_closed() -> None:
    admitted = validate_candidate_admission_v4(_candidate())
    assert admitted["candidate_registry_proposal_allowed"] is True
    assert admitted["registry_write_enabled"] is False

    invalid = _candidate()
    invalid["data_source_receipt"]["backend"] = "csv"
    invalid["full_a_universe_verified"] = False
    invalid["duplicate_primitive"] = True
    invalid["high_correlation_dedup_passed"] = False
    invalid["initial_weight"] = 0.01
    invalid["gate_results"]["8"] = False
    assessment = assess_candidate_admission_v4(invalid)
    assert assessment["candidate_registry_proposal_allowed"] is False
    assert "strict_parquet_source_contract_mismatch" in assessment["blockers"]
    assert "candidate_full_a_universe_not_verified" in assessment["blockers"]
    assert "candidate_duplicate_primitive_not_false" in assessment["blockers"]
    assert "candidate_initial_weight_must_be_zero" in assessment["blockers"]
    with pytest.raises(FactorGovernanceV4Error, match="candidate admission blocked"):
        validate_candidate_admission_v4(invalid)


@pytest.mark.parametrize(
    ("failure_count", "expected_action", "expected_weight"),
    [
        (1, "watch_proposal", 0.20),
        (2, "reduce_proposal", 0.10),
        (3, "deprecate_proposal", 0.0),
        (4, "deprecate_proposal", 0.0),
    ],
)
def test_independent_alpha_failures_map_to_proposals_only(
    failure_count: int, expected_action: str, expected_weight: float
) -> None:
    proposal = build_health_action_proposal_v4(
        factor_name="factor-a",
        failure_window_ids=[f"window-{index}" for index in range(failure_count)],
        current_weight=0.20,
    )
    assert proposal["action"] == expected_action
    assert proposal["proposed_weight"] == expected_weight
    assert proposal["apply"] is False
    assert proposal["proposal_only"] is True
    assert proposal["proposal_emitted"] is True


def test_data_blocked_does_not_increment_failure_but_blocks_set_new_risk() -> None:
    proposal = build_health_action_proposal_v4(
        factor_name="factor-a",
        failure_window_ids=["same-window", "same-window"],
        current_weight=0.20,
        data_blocked_window_ids=["blocked-window"],
    )
    assert proposal["independent_alpha_failure_count"] == 1
    assert proposal["action"] == "watch_proposal"
    assert proposal["data_blocked_counts_as_alpha_failure"] is False
    assert proposal["factor_set_new_risk_blocked"] is True
    assert proposal["blocking_factor_names"] == ["factor-a"]
