from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from datetime import date, timedelta

import pytest

from quant_investor.factors.governance_candidate_admission_evidence_v4 import (
    DEDUP_READBACK_SCHEMA_VERSION,
    SCREENING_READBACK_SCHEMA_VERSION,
    build_candidate_dedup_evidence_v4,
    canonical_file_bytes,
    file_sha256_for_payload,
    readback_screening_evidence_v4,
)
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
from quant_investor.factors.governance_screening_v4 import (
    COMPUTE_FAILED_STATUS,
    EVALUATED_STATUS,
    FDR_METHOD,
    RAW_P_METHOD,
    build_candidate_catalog_v4,
    build_primitive_ontology_v4,
    build_screening_evidence_v4,
    canonical_json_bytes as screening_canonical_json_bytes,
)
from quant_investor.factors.governance_transaction_v4 import (
    activation_receipt_sha256,
)
from quant_investor.factors.runtime import production_factor_set_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _calendar() -> dict:
    cursor = date(2024, 1, 1)
    end = date(2026, 2, 1)
    sessions: list[str] = []
    while cursor < end:
        if cursor.weekday() < 5:
            sessions.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return {
        "schema_version": "factor-governance-open-session-calendar.v4",
        "market": "CN",
        "source": "strict_parquet_observed_trade_dates",
        "latest_pointer_sha256": _digest("pointer"),
        "manifest_sha256": _digest("manifest"),
        "open_session_dates": sessions,
    }


def _month_ends(calendar: dict, count: int = 12) -> list[str]:
    by_month: dict[str, str] = {}
    for item in calendar["open_session_dates"]:
        by_month[item[:7]] = item
    return list(by_month.values())[:count]


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
    calendar = _calendar()
    return {
        "name": name,
        "family": family,
        "slot": f"{family}::slot_{index}",
        "state": "production_factor",
        "weight": 1.0,
        "calendar_sha256": semantic_sha256(calendar),
        "gate_results": {str(gate_id): True for gate_id in range(1, 9)},
        "maturity": {
            "calendar": calendar,
            "month_end_rankic_dates": _month_ends(calendar),
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
    assert policy["candidate_admission"]["canonical_replay_hash_bindings"] == [
        "market_data_input_sha256",
        "candidate_catalog_sha256",
        "screening_evidence_sha256",
        "dedup_evidence_sha256",
        "quantitative_evidence_sha256",
    ]


def test_v4_maturity_accepts_12_months_or_8_nonoverlap_cohorts() -> None:
    calendar = _calendar()
    calendar_sha = semantic_sha256(calendar)
    month_route = assess_candidate_maturity(
        month_end_rankic_dates=_month_ends(calendar),
        forward_cohorts=[],
        calendar=calendar,
        expected_calendar_sha256=calendar_sha,
    )
    assert month_route["mature"] is True
    assert month_route["maturity_route"] == "month_end_rankic"

    sessions = calendar["open_session_dates"]
    cohorts = []
    for index in range(8):
        cohort_sessions = sessions[index * 40 : index * 40 + 30]
        cohorts.append(
            {
                "cohort_id": f"cohort-{index}",
                "start": cohort_sessions[0],
                "end": cohort_sessions[-1],
                "horizon_days": 30,
                "open_session_dates": cohort_sessions,
                "calendar_sha256": calendar_sha,
            }
        )
    cohort_route = assess_candidate_maturity(
        month_end_rankic_dates=[],
        forward_cohorts=cohorts,
        calendar=calendar,
        expected_calendar_sha256=calendar_sha,
    )
    assert cohort_route["mature"] is True
    assert cohort_route["maturity_route"] == "nonoverlap_30d_forward_cohort"


def test_v4_maturity_rejects_self_declared_horizon_and_weekend_calendar() -> None:
    calendar = _calendar()
    legacy_cohorts = [
        {
            "cohort_id": f"legacy-{index}",
            "start": f"2025-{index + 1:02d}-01",
            "end": f"2025-{index + 1:02d}-28",
            "horizon_days": 30,
        }
        for index in range(8)
    ]
    legacy = assess_candidate_maturity(
        month_end_rankic_dates=[],
        forward_cohorts=legacy_cohorts,
        calendar=calendar,
        expected_calendar_sha256=semantic_sha256(calendar),
    )
    assert legacy["mature"] is False
    assert legacy["nonoverlap_30d_cohort_count"] == 0

    weekend_calendar = copy.deepcopy(calendar)
    weekend_calendar["open_session_dates"].insert(5, "2024-01-06")
    weekend = assess_candidate_maturity(
        month_end_rankic_dates=_month_ends(calendar),
        forward_cohorts=[],
        calendar=weekend_calendar,
    )
    assert weekend["mature"] is False
    assert weekend["calendar_verified"] is False
    assert "weekend" in weekend["maturity_blockers"][0]


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
    runtime_contract = {
        "schema_version": "factor-production-runtime-contract.v4",
        "factor_name": "candidate-a",
    }
    calendar = _calendar()
    return {
        "name": "candidate-a",
        "family": "value",
        "slot": "value::primary",
        "registry_file_sha256": _digest("candidate-registry"),
        "calendar_sha256": semantic_sha256(calendar),
        "pit_sha256": _digest("candidate-pit"),
        "candidate_catalog_sha256": _digest("candidate-catalog"),
        "screening_evidence_sha256": _digest("candidate-screening"),
        "dedup_evidence_sha256": _digest("candidate-dedup"),
        "quantitative_evidence_sha256": _digest("candidate-quantitative"),
        "runtime_contract": runtime_contract,
        "runtime_contract_sha256": semantic_sha256(runtime_contract),
        "runtime_contract_status": "verified",
        "data_source_receipt": {
            "schema_version": "factor-v4-strict-parquet-source-receipt.v1",
            "backend": "parquet",
            "mode_policy": "strict",
            "snapshot_healthy": True,
            "universe_scope": "full_a",
            "snapshot_id": "snapshot-test",
            "latest_pointer_sha256": _digest("pointer"),
            "manifest_sha256": _digest("manifest"),
            "market_data_input_sha256": _digest("market-data-input"),
        },
        "full_a_universe_verified": True,
        "duplicate_primitive": False,
        "high_correlation_dedup_passed": True,
        "initial_weight": 0.0,
        "maturity": {
            "calendar": calendar,
            "month_end_rankic_dates": _month_ends(calendar),
            "forward_cohorts": [],
        },
        "bh_q_value": 0.05,
        "fdr_method": FDR_METHOD,
        "gate_results": {str(gate_id): True for gate_id in range(1, 9)},
        "evidence": {
            "schema_version": "factor-governance-replay-evidence.v4",
            "status": "verified",
            "replay_semantic_sha256": _digest("candidate-replay"),
        },
    }


def _write_v4_json(path: Path, payload: dict) -> str:
    path.write_bytes(canonical_file_bytes(payload))
    path.chmod(0o600)
    return file_sha256_for_payload(payload)


def _attach_candidate_screening_and_dedup(
    candidate: dict,
    tmp_path: Path,
    *,
    raw_p_value: float = 0.05,
    dedup_rows: list[dict] | None = None,
    evidence_complete: bool = True,
) -> dict:
    ontology = build_primitive_ontology_v4(
        [
            {"primitive_id": "close_return", "family": candidate["family"]},
            {"primitive_id": "volume", "family": "liquidity"},
        ]
    )
    catalog = build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            {
                "name": candidate["name"],
                "implementation": "formulaic",
                "expression": "rank(close)",
                "direction": 1,
                "params": {"window": 20},
                "lookback": 20,
                "slot": candidate["slot"],
                "input_fields": ["close"],
                "primitive_ids": ["close_return"],
            }
        ],
    )
    source = candidate["data_source_receipt"]
    screening = build_screening_evidence_v4(
        ontology=ontology,
        catalog=catalog,
        evaluations=[
            {
                "name": candidate["name"],
                "evaluation_status": EVALUATED_STATUS,
                "raw_p_value": raw_p_value,
                "failure_reason": None,
            }
        ],
        source_bindings={
            "code_sha256": _digest("code"),
            "registry_file_sha256": candidate["registry_file_sha256"],
            "latest_pointer_sha256": source["latest_pointer_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "market_data_input_sha256": source["market_data_input_sha256"],
            "pit_sha256": candidate["pit_sha256"],
            "calendar_sha256": candidate["calendar_sha256"],
            "fundamental_manifest_sha256": _digest("fundamental"),
            "run_config_sha256": _digest("run-config"),
        },
        statistic_contract={
            "raw_p_method": RAW_P_METHOD,
            "fdr_method": FDR_METHOD,
            "q": 0.1,
        },
    )
    dedup = build_candidate_dedup_evidence_v4(
        catalog=catalog,
        candidate_name=candidate["name"],
        screening_evidence_sha256=screening["semantic_sha256"],
        source_bindings=screening["source_bindings"],
        comparison_rows=dedup_rows
        if dedup_rows is not None
        else [
            {
                "existing_factor_name": "existing-low-corr",
                "existing_primitive_ids": ["volume"],
                "abs_correlation": 0.69,
                "valid_common_date_count": 12,
            }
        ],
        evidence_complete=evidence_complete,
    )
    candidate["candidate_catalog_sha256"] = catalog["semantic_sha256"]
    candidate["screening_evidence_sha256"] = screening["semantic_sha256"]
    candidate["dedup_evidence_sha256"] = dedup["semantic_sha256"]
    candidate["bh_q_value"] = next(
        row["bh_q_value"] for row in screening["rows"] if row["name"] == candidate["name"]
    )

    ontology_path = (tmp_path / "ontology.v4.json").resolve()
    catalog_path = (tmp_path / "catalog.v4.json").resolve()
    screening_path = (tmp_path / "screening.v4.json").resolve()
    dedup_path = (tmp_path / "dedup.v4.json").resolve()
    ontology_file_sha = _write_v4_json(ontology_path, ontology)
    catalog_file_sha = _write_v4_json(catalog_path, catalog)
    screening_file_sha = _write_v4_json(screening_path, screening)
    dedup_file_sha = _write_v4_json(dedup_path, dedup)
    candidate["screening_evidence"] = {
        "schema_version": SCREENING_READBACK_SCHEMA_VERSION,
        "ontology_path": str(ontology_path),
        "ontology_file_sha256": ontology_file_sha,
        "candidate_catalog_path": str(catalog_path),
        "candidate_catalog_file_sha256": catalog_file_sha,
        "screening_evidence_path": str(screening_path),
        "screening_evidence_file_sha256": screening_file_sha,
    }
    candidate["dedup_evidence"] = {
        "schema_version": DEDUP_READBACK_SCHEMA_VERSION,
        "dedup_evidence_path": str(dedup_path),
        "dedup_evidence_file_sha256": dedup_file_sha,
    }
    return {
        "ontology": ontology,
        "catalog": catalog,
        "screening": screening,
        "dedup": dedup,
        "paths": {
            "ontology": ontology_path,
            "catalog": catalog_path,
            "screening": screening_path,
            "dedup": dedup_path,
        },
    }


def _candidate_readback(candidate: dict) -> dict:
    source = candidate["data_source_receipt"]
    context = {
        "eligibility_contract_sha256": _digest("candidate-eligibility"),
        "calendar_sha256": candidate["calendar_sha256"],
        "pit_sha256": candidate["pit_sha256"],
        "runtime_contract_sha256": candidate["runtime_contract_sha256"],
        "latest_pointer_sha256": source["latest_pointer_sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "market_data_input_sha256": source["market_data_input_sha256"],
        "candidate_catalog_sha256": candidate["candidate_catalog_sha256"],
        "screening_evidence_sha256": candidate["screening_evidence_sha256"],
        "dedup_evidence_sha256": candidate["dedup_evidence_sha256"],
        "quantitative_evidence_sha256": candidate["quantitative_evidence_sha256"],
    }
    challenger_record = {
        "name": candidate["name"],
        "family": candidate["family"],
        "slot": candidate["slot"],
        "state": "mature_candidate",
        "registry_record_sha256": _digest("candidate-record"),
    }
    existing_record = {
        "name": "existing-low-corr",
        "family": "liquidity",
        "slot": "liquidity::primary",
        "state": "production_factor",
        "registry_record_sha256": _digest("existing-record"),
    }
    return {
        "evidence": {"factor_name": candidate["name"]},
        "replay": {
            "registry_file_sha256": candidate["registry_file_sha256"],
            "context": context,
            "comparison": {
                "transition_mode": "add",
                "incumbent": None,
                "challenger": candidate["name"],
                "slot": candidate["slot"],
            },
            "arms": {
                "A": {
                    "quant": {
                        "selected_factors": ["existing-low-corr"],
                        "factor_records": {"existing-low-corr": existing_record},
                    }
                },
                "B": {
                    "quant": {
                        "selected_factors": ["existing-low-corr"],
                        "factor_records": {"existing-low-corr": existing_record},
                    }
                },
                "C": {
                    "quant": {
                        "selected_factors": [candidate["name"], "existing-low-corr"],
                        "factor_records": {candidate["name"]: challenger_record}
                    }
                }
            },
        },
        "local_bytes_readback_verified": True,
        "complete_chain_hash_binding_verified": True,
    }


def test_candidate_admission_is_machine_checkable_and_fail_closed(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = _candidate()
    _attach_candidate_screening_and_dedup(candidate, tmp_path)
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: _candidate_readback(candidate),
    )
    admitted = validate_candidate_admission_v4(candidate)
    assert admitted["candidate_registry_proposal_allowed"] is True
    assert admitted["registry_write_enabled"] is False
    assert admitted["bh_q_value"] == pytest.approx(0.05)
    assert admitted["screening_exact_readback_verified"] is True
    assert admitted["dedup_exact_readback_verified"] is True
    assert admitted["evidence_exact_readback_verified"] is True

    invalid = _candidate()
    _attach_candidate_screening_and_dedup(invalid, tmp_path)
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
    assert "candidate_duplicate_primitive_caller_mismatch" in assessment["blockers"]
    assert "candidate_initial_weight_must_be_zero" in assessment["blockers"]
    with pytest.raises(FactorGovernanceV4Error, match="candidate admission blocked"):
        validate_candidate_admission_v4(invalid)


def test_screening_readback_accepts_runner_style_no_newline_bytes(tmp_path: Path) -> None:
    candidate = _candidate()
    artifacts = _attach_candidate_screening_and_dedup(candidate, tmp_path)
    runner_style_path = (tmp_path / "runner-style-screening.v4.json").resolve()
    runner_style_path.write_bytes(screening_canonical_json_bytes(artifacts["screening"]))
    runner_style_path.chmod(0o600)
    descriptor = {
        **candidate["screening_evidence"],
        "screening_evidence_path": str(runner_style_path),
        "screening_evidence_file_sha256": hashlib.sha256(
            screening_canonical_json_bytes(artifacts["screening"])
        ).hexdigest(),
    }

    readback = readback_screening_evidence_v4(
        descriptor,
        candidate_name=candidate["name"],
    )

    assert readback["exact_readback_verified"] is True
    assert readback["bh_q_value"] == pytest.approx(0.05)
    assert not runner_style_path.read_bytes().endswith(b"\n")


def test_candidate_admission_rejects_shallow_evidence_without_exact_readback() -> None:
    assessment = assess_candidate_admission_v4(_candidate())
    assert assessment["candidate_registry_proposal_allowed"] is False
    assert assessment["screening_exact_readback_verified"] is False
    assert assessment["dedup_exact_readback_verified"] is False
    assert assessment["evidence_exact_readback_verified"] is False
    assert "candidate_screening_evidence_readback_missing" in assessment["blockers"]
    assert "candidate_dedup_evidence_readback_missing" in assessment["blockers"]
    assert any(
        item.startswith("candidate_v4_evidence_exact_readback_failed:")
        for item in assessment["blockers"]
    )
    with pytest.raises(FactorGovernanceV4Error, match="exact_readback_failed"):
        validate_candidate_admission_v4(_candidate())


@pytest.mark.parametrize(
    ("field", "expected_blocker"),
    [
        ("registry_file_sha256", "candidate_v4_registry_sha_mismatch"),
        ("calendar_sha256", "candidate_v4_calendar_sha256_mismatch"),
        ("pit_sha256", "candidate_v4_pit_sha256_mismatch"),
        (
            "candidate_catalog_sha256",
            "candidate_v4_candidate_catalog_sha256_mismatch",
        ),
        (
            "screening_evidence_sha256",
            "candidate_v4_screening_evidence_sha256_mismatch",
        ),
        (
            "dedup_evidence_sha256",
            "candidate_v4_dedup_evidence_sha256_mismatch",
        ),
        (
            "quantitative_evidence_sha256",
            "candidate_v4_quantitative_evidence_sha256_mismatch",
        ),
        (
            "runtime_contract_sha256",
            "candidate_v4_runtime_contract_sha256_mismatch",
        ),
    ],
)
def test_candidate_admission_rejects_hash_binding_drift(
    monkeypatch, tmp_path: Path, field: str, expected_blocker: str
) -> None:
    trusted = _candidate()
    _attach_candidate_screening_and_dedup(trusted, tmp_path)
    trusted_readback = _candidate_readback(trusted)
    drifted = copy.deepcopy(trusted)
    drifted[field] = _digest(f"drifted:{field}")
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: trusted_readback,
    )
    assessment = assess_candidate_admission_v4(drifted)
    assert assessment["candidate_registry_proposal_allowed"] is False
    assert expected_blocker in assessment["blockers"]


@pytest.mark.parametrize(
    ("receipt_field", "expected_blocker"),
    [
        ("latest_pointer_sha256", "candidate_v4_latest_pointer_sha256_mismatch"),
        ("manifest_sha256", "candidate_v4_manifest_sha256_mismatch"),
        (
            "market_data_input_sha256",
            "candidate_v4_market_data_input_sha256_mismatch",
        ),
    ],
)
def test_candidate_admission_rejects_source_receipt_binding_drift(
    monkeypatch, tmp_path: Path, receipt_field: str, expected_blocker: str
) -> None:
    trusted = _candidate()
    _attach_candidate_screening_and_dedup(trusted, tmp_path)
    trusted_readback = _candidate_readback(trusted)
    drifted = copy.deepcopy(trusted)
    drifted["data_source_receipt"][receipt_field] = _digest(
        f"drifted:{receipt_field}"
    )
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: trusted_readback,
    )
    assessment = assess_candidate_admission_v4(drifted)
    assert assessment["candidate_registry_proposal_allowed"] is False
    assert expected_blocker in assessment["blockers"]


def test_candidate_admission_rejects_forged_low_q_and_tampered_bh_rehash(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = _candidate()
    artifacts = _attach_candidate_screening_and_dedup(
        candidate,
        tmp_path,
        raw_p_value=0.20,
    )
    candidate["bh_q_value"] = 0.01
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: _candidate_readback(candidate),
    )

    forged_low_q = assess_candidate_admission_v4(candidate)
    assert "candidate_screening_bh_not_passed" in forged_low_q["blockers"]
    assert "candidate_bh_q_value_caller_mismatch" in forged_low_q["blockers"]
    assert forged_low_q["bh_q_value"] == pytest.approx(0.20)

    tampered = copy.deepcopy(artifacts["screening"])
    tampered["rows"][0]["bh_q_value"] = 0.01
    tampered["rows"][0]["bh_pass"] = True
    tampered["semantic_sha256"] = semantic_sha256(
        {key: value for key, value in tampered.items() if key != "semantic_sha256"}
    )
    _write_v4_json(artifacts["paths"]["screening"], tampered)
    candidate["screening_evidence"]["screening_evidence_file_sha256"] = (
        file_sha256_for_payload(tampered)
    )
    candidate["screening_evidence_sha256"] = tampered["semantic_sha256"]
    rehashed = assess_candidate_admission_v4(candidate)
    assert any(
        item.startswith("candidate_screening_exact_readback_failed:")
        for item in rehashed["blockers"]
    )


def test_candidate_admission_requires_evaluated_screening_row(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = _candidate()
    artifacts = _attach_candidate_screening_and_dedup(candidate, tmp_path)
    source = candidate["data_source_receipt"]
    compute_failed = build_screening_evidence_v4(
        ontology=artifacts["ontology"],
        catalog=artifacts["catalog"],
        evaluations=[
            {
                "name": candidate["name"],
                "evaluation_status": COMPUTE_FAILED_STATUS,
                "raw_p_value": None,
                "failure_reason": "compute_failed",
            }
        ],
        source_bindings={
            "code_sha256": _digest("code"),
            "registry_file_sha256": candidate["registry_file_sha256"],
            "latest_pointer_sha256": source["latest_pointer_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "market_data_input_sha256": source["market_data_input_sha256"],
            "pit_sha256": candidate["pit_sha256"],
            "calendar_sha256": candidate["calendar_sha256"],
            "fundamental_manifest_sha256": _digest("fundamental"),
            "run_config_sha256": _digest("run-config"),
        },
        statistic_contract={
            "raw_p_method": RAW_P_METHOD,
            "fdr_method": FDR_METHOD,
            "q": 0.1,
        },
    )
    _write_v4_json(artifacts["paths"]["screening"], compute_failed)
    candidate["screening_evidence"]["screening_evidence_file_sha256"] = (
        file_sha256_for_payload(compute_failed)
    )
    candidate["screening_evidence_sha256"] = compute_failed["semantic_sha256"]
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: _candidate_readback(candidate),
    )

    assessment = assess_candidate_admission_v4(candidate)
    assert any(
        item.startswith("candidate_screening_exact_readback_failed:")
        for item in assessment["blockers"]
    )


def test_candidate_admission_rejects_missing_catalog_row_and_wrong_family(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = _candidate()
    artifacts = _attach_candidate_screening_and_dedup(candidate, tmp_path)
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: _candidate_readback(candidate),
    )

    missing_catalog = copy.deepcopy(artifacts["catalog"])
    missing_catalog["candidates"] = []
    missing_catalog["semantic_sha256"] = semantic_sha256(
        {key: value for key, value in missing_catalog.items() if key != "semantic_sha256"}
    )
    _write_v4_json(artifacts["paths"]["catalog"], missing_catalog)
    candidate["screening_evidence"]["candidate_catalog_file_sha256"] = (
        file_sha256_for_payload(missing_catalog)
    )
    missing = assess_candidate_admission_v4(candidate)
    assert any(
        item.startswith("candidate_screening_exact_readback_failed:")
        for item in missing["blockers"]
    )

    candidate = _candidate()
    _attach_candidate_screening_and_dedup(candidate, tmp_path)
    candidate["family"] = "forged-family"
    wrong_family = assess_candidate_admission_v4(candidate)
    assert "candidate_screening_family_mismatch" in wrong_family["blockers"]


@pytest.mark.parametrize("mode", ["symlink", "mode", "hash"])
def test_candidate_admission_rejects_unsafe_screening_files(
    monkeypatch, tmp_path: Path, mode: str
) -> None:
    candidate = _candidate()
    artifacts = _attach_candidate_screening_and_dedup(candidate, tmp_path)
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: _candidate_readback(candidate),
    )
    if mode == "symlink":
        link = (tmp_path / "screening-link.v4.json").resolve()
        link.symlink_to(artifacts["paths"]["screening"])
        candidate["screening_evidence"]["screening_evidence_path"] = str(link)
    elif mode == "mode":
        artifacts["paths"]["screening"].chmod(0o644)
    else:
        candidate["screening_evidence"]["screening_evidence_file_sha256"] = _digest(
            "wrong-file-sha"
        )

    assessment = assess_candidate_admission_v4(candidate)
    assert any(
        item.startswith("candidate_screening_exact_readback_failed:")
        for item in assessment["blockers"]
    )


@pytest.mark.parametrize(
    ("dedup_rows", "expected_blocker"),
    [
        (
            [
                {
                    "existing_factor_name": "existing-low-corr",
                    "existing_primitive_ids": ["volume"],
                    "abs_correlation": 0.70,
                    "valid_common_date_count": 12,
                }
            ],
            "candidate_high_correlation_dedup_not_passed",
        ),
        (
            [
                {
                    "existing_factor_name": "existing-low-corr",
                    "existing_primitive_ids": ["close_return"],
                    "abs_correlation": 0.10,
                    "valid_common_date_count": 12,
                }
            ],
            "candidate_duplicate_primitive_not_false",
        ),
    ],
)
def test_candidate_admission_rejects_high_corr_and_duplicate_primitives(
    monkeypatch, tmp_path: Path, dedup_rows: list[dict], expected_blocker: str
) -> None:
    candidate = _candidate()
    _attach_candidate_screening_and_dedup(
        candidate,
        tmp_path,
        dedup_rows=dedup_rows,
    )
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: _candidate_readback(candidate),
    )
    assessment = assess_candidate_admission_v4(candidate)
    assert expected_blocker in assessment["blockers"]
    assert assessment["dedup_exact_readback_verified"] is True
    assert assessment["candidate_registry_proposal_allowed"] is False


@pytest.mark.parametrize(
    ("dedup_rows", "expected_names"),
    [
        ([], []),
        (
            [
                {
                    "existing_factor_name": "existing-low-corr",
                    "existing_primitive_ids": ["volume"],
                    "abs_correlation": 0.10,
                    "valid_common_date_count": 12,
                },
                {
                    "existing_factor_name": "extra-not-in-replay",
                    "existing_primitive_ids": ["volume"],
                    "abs_correlation": 0.10,
                    "valid_common_date_count": 12,
                },
            ],
            ["existing-low-corr", "extra-not-in-replay"],
        ),
    ],
)
def test_candidate_admission_rejects_dedup_comparison_omission_or_extra_row(
    monkeypatch,
    tmp_path: Path,
    dedup_rows: list[dict],
    expected_names: list[str],
) -> None:
    candidate = _candidate()
    _attach_candidate_screening_and_dedup(
        candidate,
        tmp_path,
        dedup_rows=dedup_rows,
    )
    import quant_investor.factors.governance_canonical_replay_v4 as replay_v4

    monkeypatch.setattr(
        replay_v4,
        "readback_v4_evidence",
        lambda evidence: _candidate_readback(candidate),
    )

    assessment = assess_candidate_admission_v4(candidate)
    assert "candidate_dedup_comparison_factor_set_mismatch" in assessment["blockers"]
    assert assessment["dedup_exact_readback_verified"] is True
    assert assessment["candidate_registry_proposal_allowed"] is False
    assert expected_names != ["existing-low-corr"]


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
