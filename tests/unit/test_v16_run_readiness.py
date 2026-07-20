from __future__ import annotations

import copy
from hashlib import sha256
import stat
from pathlib import Path

import pytest

from quant_investor.codex_review import seal_json_payload
from quant_investor.factors.governance_protocol_v4 import (
    assess_factor_governance_readiness_v4,
    protocol_hash,
    semantic_sha256,
)
from quant_investor.factors.governance_quality_v1 import (
    factor_quality_set_identity_sha256,
)
from quant_investor.factors.governance_transaction_v4 import (
    activation_receipt_sha256,
)
from quant_investor.factors.runtime import production_factor_set_sha256
from quant_investor.monitoring.v15_run_readiness import build_v15_run_readiness
from quant_investor.monitoring.v16_run_readiness import (
    V16ReadinessError,
    build_v16_run_readiness,
    canonical_sha256,
    load_v16_run_readiness,
    validate_v16_run_readiness,
    write_v16_run_readiness,
)

SYNTHETIC_SYMBOL = "SYNTH-0001"


def _factor_governance(*, include_quality: bool = False) -> dict[str, object]:
    month_end_rankic_dates = [
        "2025-01-31",
        "2025-02-28",
        "2025-03-31",
        "2025-04-30",
        "2025-05-30",
        "2025-06-30",
        "2025-07-31",
        "2025-08-29",
        "2025-09-30",
        "2025-10-31",
        "2025-11-28",
        "2025-12-31",
    ]
    calendar = {
        "schema_version": "factor-governance-open-session-calendar.v4",
        "market": "CN",
        "source": "strict_parquet_observed_trade_dates",
        "latest_pointer_sha256": "7" * 64,
        "manifest_sha256": "8" * 64,
        "open_session_dates": month_end_rankic_dates,
    }
    records = []
    for index in range(5):
        name = f"synthetic_factor_{index}"
        runtime_contract = {
            "schema_version": "factor-production-runtime-contract.v4",
            "factor_name": name,
            "formula": f"primitive_{index}",
        }
        records.append(
            {
                "name": name,
                "family": f"synthetic_family_{index}",
                "slot": f"synthetic_family_{index}::slot",
                "state": "production_factor",
                "weight": 1.0,
                "calendar_sha256": semantic_sha256(calendar),
                "gate_results": {str(gate): True for gate in range(1, 9)},
                "maturity": {
                    "calendar": copy.deepcopy(calendar),
                    "month_end_rankic_dates": list(month_end_rankic_dates),
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
                    "replay_semantic_sha256": sha256(name.encode()).hexdigest(),
                },
                "health": {
                    "status": "healthy",
                    "fresh": True,
                    "data_blocked": False,
                },
            }
        )
    if include_quality:
        identity_sha256 = factor_quality_set_identity_sha256(records)
        for record in records:
            record["runtime_contract"]["quality_set_identity_sha256"] = identity_sha256
            record["runtime_contract_sha256"] = semantic_sha256(record["runtime_contract"])
    registry_sha = "b" * 64
    factor_set_sha = production_factor_set_sha256(sorted(record["name"] for record in records))
    runtime_sha = semantic_sha256(sorted(record["runtime_contract_sha256"] for record in records))
    context = {
        "protocol_hash": protocol_hash(),
        "transaction_plan_sha256": "9" * 64,
        "registry_file_sha256": registry_sha,
        "production_factor_set_sha256": factor_set_sha,
        "runtime_contracts_sha256": runtime_sha,
        "as_of": "2026-07-17",
    }
    receipt = {
        "schema_version": "factor-governance-activation-receipt.v4",
        "protocol_version": "v4",
        "protocol_hash": protocol_hash(),
        "receipt_id": "receipt-readiness-test",
        "status": "activated",
        "authorization_scope": "factor_v4_production_activation",
        "authorized_by": "Maxwell",
        "activated_at": "2026-07-17T09:00:00+08:00",
        "as_of": "2026-07-17",
        "transaction_plan_sha256": "9" * 64,
        "registry_file_sha256": registry_sha,
        "production_factor_set_sha256": factor_set_sha,
        "runtime_contracts_sha256": runtime_sha,
        "activation_context_sha256": semantic_sha256(context),
        "activation_performed": True,
    }
    receipt["receipt_sha256"] = activation_receipt_sha256(receipt)
    return assess_factor_governance_readiness_v4(
        records,
        as_of="2026-07-17",
        registry_file_sha256=registry_sha,
        production_factor_set_sha256=factor_set_sha,
        activation_receipt=receipt,
        quality_records=records if include_quality else None,
    )


def _human_authorization(
    *, run_id: str, stage2_response_sha256: str, capital_map_sha256: str
) -> dict[str, object]:
    return seal_json_payload(
        {
            "schema_version": "codex-review-human-authorization.v1",
            "run_id": run_id,
            "stage2_response_sha256": stage2_response_sha256,
            "capital_map_sha256": capital_map_sha256,
            "decision": "AUTHORIZED",
            "authorized_by": "Maxwell",
            "authorized_at": "2026-07-17T09:30:00+08:00",
            "expires_at": "2026-07-17T11:00:00+08:00",
            "rationale": "人工复核完成并授权。",
        },
        digest_field="receipt_sha256",
    )


def _branches(*, ready: bool = True) -> dict[str, dict[str, object]]:
    return {
        branch: {
            "status": "pass" if ready else "block",
            "blockers": [] if ready else ["synthetic_missing"],
        }
        for branch in ("quant", "fundamental", "macro", "llm")
    }


def _inputs() -> dict[str, object]:
    execution_plan = {
        "valid": True,
        "selected_symbols": [SYNTHETIC_SYMBOL],
        "broker_side_effects": False,
        "blockers": [],
    }
    plan_sha = canonical_sha256(execution_plan)
    stage2_response_sha = "e" * 64
    capital_map_sha = "f" * 64
    authorization = _human_authorization(
        run_id="synthetic-v16-run",
        stage2_response_sha256=stage2_response_sha,
        capital_map_sha256=capital_map_sha,
    )
    return {
        "run_id": "synthetic-v16-run",
        "generated_at": "2026-07-17T10:00:00+08:00",
        "analysis_trade_date": "2026-07-16",
        "market_data_ready": True,
        "market_data_blockers": [],
        "branch_readiness": _branches(),
        "branch_objects": {branch: True for branch in ("quant", "fundamental", "macro", "llm")},
        "factor_governance": _factor_governance(),
        "calibration": {
            "schema_version": "calibration-readiness.v16.four-evidence",
            "branches": {
                branch: {"samples": 300, "nonoverlap_cohorts": 8}
                for branch in ("quant", "fundamental", "macro", "llm")
            },
            "metrics": {
                "brier_bootstrap_upper": 0.18,
                "brier_baseline": 0.20,
                "logloss_bootstrap_upper": 0.58,
                "logloss_baseline": 0.60,
                "ece": 0.04,
                "interval_coverage": 0.90,
                "alpha_mae": 0.03,
                "zero_alpha_mae": 0.04,
                "top_bucket_edge_lower": 0.01,
                "lambda_fold_min": 0.30,
                "lambda_fold_max": 0.45,
            },
            "artifact_sha256": "c" * 64,
            "blockers": [],
        },
        "candidate_decision": {
            "candidate_decision_status": "complete",
            "selected_symbols": [SYNTHETIC_SYMBOL],
        },
        "eligibility": {"eligible": True, "blockers": []},
        "handoff": {
            "status": "complete",
            "execution_plan_sha256": plan_sha,
            "artifact_sha256": "d" * 64,
            "stage2_response_sha256": stage2_response_sha,
            "capital_map_sha256": capital_map_sha,
            "authorization_receipt_sha256": authorization["receipt_sha256"],
            "blockers": [],
        },
        "execution_plan": execution_plan,
        "activation_gates": {
            "codex_ready": True,
            "dashboard_ready": True,
            "blockers": [],
        },
        "human_authorization": authorization,
        "risk_reduction_quote_gate": {
            "authorized": False,
            "blockers": ["synthetic_quote_missing"],
        },
        "material_warnings": [],
    }


def _payload(**overrides: object) -> dict[str, object]:
    values = _inputs()
    values.update(overrides)
    return build_v16_run_readiness(**values)  # type: ignore[arg-type]


def test_legacy_v16_inputs_cannot_bypass_evidence_v2_migration_gates() -> None:
    payload = _payload()

    assert payload["branch_data_ready"] == {
        "quant": True,
        "fundamental": True,
        "macro": True,
        "llm": True,
    }
    assert payload["activation_candidate"] is False
    assert payload["new_risk_authorized"] is False
    assert payload["readiness_status"] == "no_new_risk"
    assert payload["activation_blockers"] == [
        "evidence_v2_disconnected_from_authorizing_consumers",
        "global_attempt_registry_authority_not_integrated",
    ]
    assert payload["execution"]["broker_side_effects"] is False
    validate_v16_run_readiness(payload)


def test_validator_rejects_removed_evidence_v2_migration_blocker() -> None:
    payload = _payload()
    blocker = "global_attempt_registry_authority_not_integrated"
    payload["activation_blockers"].remove(blocker)
    payload["blockers"].remove(blocker)

    with pytest.raises(
        V16ReadinessError,
        match="must preserve disconnected evidence-v2 migration gates",
    ):
        validate_v16_run_readiness(payload)


@pytest.mark.parametrize(
    ("override", "expected_blocker"),
    [
        (
            {
                "factor_governance": {
                    "schema_version": "factor-governance-readiness.v4",
                    "protocol_version": "v4",
                    "factor_governance_ready": True,
                    "new_risk_eligible": True,
                    "production_factor_count": 4,
                    "healthy_factor_count": 4,
                    "production_family_count": 4,
                    "activation_receipt": {
                        "valid": True,
                        "receipt_sha256": "a" * 64,
                        "blockers": [],
                    },
                    "normalized_abs_weights": {
                        f"synthetic_factor_{index}": 0.25 for index in range(4)
                    },
                    "family_normalized_abs_weights": {
                        f"synthetic_family_{index}": 0.25 for index in range(4)
                    },
                    "blockers": [],
                }
            },
            "factor_count_below_minimum:actual=4:required=5",
        ),
        ({"handoff": None}, "handoff_missing"),
        (
            {
                "calibration": {
                    **_inputs()["calibration"],  # type: ignore[dict-item]
                    "metrics": {
                        **_inputs()["calibration"]["metrics"],  # type: ignore[index]
                        "ece": 0.051,
                    },
                }
            },
            "calibration_threshold_not_met",
        ),
        (
            {"human_authorization": None},
            "new_risk_human_authorization_missing_or_invalid",
        ),
    ],
)
def test_required_v16_fail_closed_gates(override: dict[str, object], expected_blocker: str) -> None:
    payload = _payload(**override)

    assert payload["new_risk_authorized"] is False
    assert payload["readiness_status"] == "no_new_risk"
    assert expected_blocker in payload["blockers"]


def test_missing_codex_and_dashboard_activation_stays_research_only() -> None:
    payload = _payload(activation_gates=None)

    assert payload["activation_candidate"] is False
    assert payload["new_risk_authorized"] is False
    assert payload["activation_blockers"] == [
        "activation_codex_gate_not_ready",
        "activation_dashboard_gate_not_ready",
        "evidence_v2_disconnected_from_authorizing_consumers",
        "global_attempt_registry_authority_not_integrated",
    ]


def test_missing_calibration_writes_canonical_fail_closed_readiness(tmp_path: Path) -> None:
    payload = _payload(calibration=None)

    validate_v16_run_readiness(payload)
    path = tmp_path / "results/v16/missing-calibration/v16_run_readiness.json"
    reference = write_v16_run_readiness(path, payload)
    loaded = load_v16_run_readiness(path, expected_sha256=reference["sha256"])

    assert loaded["calibration_ready"] is False
    assert loaded["calibration"]["checks"]["shape"] is False
    assert "calibration_gate_failed:shape" in loaded["blockers"]
    assert loaded["new_risk_authorized"] is False


def test_llm_is_required_fourth_branch() -> None:
    branches = _branches()
    branches.pop("llm")
    objects = {branch: True for branch in ("quant", "fundamental", "macro")}
    payload = _payload(branch_readiness=branches, branch_objects=objects)

    assert payload["branch_data_ready"]["llm"] is False
    assert "branch_object_missing:llm" in payload["blockers"]
    assert payload["new_risk_authorized"] is False


def test_factor_v3_or_missing_activation_receipt_cannot_pass_v16() -> None:
    factor_v3 = copy.deepcopy(_inputs()["factor_governance"])
    factor_v3["schema_version"] = "factor-governance-readiness.v3"
    payload = _payload(factor_governance=factor_v3)
    assert payload["factor_governance_ready"] is False
    assert "factor_readiness_schema_not_v4" in payload["blockers"]

    missing_receipt = copy.deepcopy(_inputs()["factor_governance"])
    missing_receipt["activation_receipt"] = None
    payload = _payload(factor_governance=missing_receipt)
    assert payload["factor_governance_ready"] is False
    assert "factor_activation_receipt_missing" in payload["blockers"]


def test_factor_quality_projection_is_informational_only() -> None:
    without_quality = _payload(factor_governance=_factor_governance())
    with_quality = _payload(factor_governance=_factor_governance(include_quality=True))

    assert without_quality["factor_governance"]["quality_assessment"] == {
        "availability": "unavailable",
        "schema_version": "missing",
        "policy_hash": None,
        "status": "unavailable",
        "valid": False,
        "report_only": True,
        "quality_ready": False,
        "shadow_observation_eligible": False,
        "factor_count": 0,
        "family_count": 0,
        "qualified_factor_count": 0,
        "qualified_family_count": 0,
        "quality_set_sha256": None,
        "assessment_sha256": None,
        "blockers": [],
    }
    quality = with_quality["factor_governance"]["quality_assessment"]
    assert quality["availability"] == "available"
    assert quality["status"] == "ready_underfilled"
    assert quality["quality_ready"] is True
    for field in (
        "factor_governance_ready",
        "activation_candidate",
        "activation_blockers",
        "activation_gates",
        "new_risk_authorized",
        "readiness_status",
        "blockers",
    ):
        assert with_quality[field] == without_quality[field]


def test_tampered_factor_quality_normalizes_invalid_without_blocking_v16() -> None:
    baseline_factor = _factor_governance(include_quality=True)
    tampered_factor = copy.deepcopy(baseline_factor)
    tampered_factor["quality_assessment"]["quality_policy_hash"] = "a" * 64

    baseline = _payload(factor_governance=baseline_factor)
    tampered = _payload(factor_governance=tampered_factor)
    quality = tampered["factor_governance"]["quality_assessment"]

    assert quality["availability"] == "invalid"
    assert quality["valid"] is False
    assert quality["blockers"] == ["quality_assessment_invalid"]
    for field in (
        "factor_governance_ready",
        "activation_candidate",
        "activation_blockers",
        "activation_gates",
        "new_risk_authorized",
        "readiness_status",
        "blockers",
    ):
        assert tampered[field] == baseline[field]


def test_historical_v16_nested_factor_summary_without_quality_still_validates() -> None:
    historical = _payload()
    del historical["factor_governance"]["quality_assessment"]

    validate_v16_run_readiness(historical)


def test_summary_only_factor_receipt_cannot_unlock_v16() -> None:
    spoofed = copy.deepcopy(_factor_governance())
    spoofed["activation_receipt"] = {
        "valid": True,
        "receipt_sha256": "a" * 64,
        "blockers": [],
    }
    payload = _payload(factor_governance=spoofed)
    assert payload["factor_governance_ready"] is False
    assert "factor_activation_receipt_missing_or_invalid" in payload["blockers"]


def test_unsigned_authorization_mapping_cannot_authorize_new_risk() -> None:
    payload = _payload(
        human_authorization={
            "authorized": True,
            "run_id": "synthetic-v16-run",
            "analysis_trade_date": "2026-07-16",
            "execution_plan_sha256": canonical_sha256(
                _inputs()["execution_plan"]  # type: ignore[arg-type]
            ),
        }
    )
    assert payload["new_risk_authorized"] is False
    assert payload["human_authorization"]["valid"] is False
    assert "new_risk_human_authorization_missing_or_invalid" in payload["blockers"]


@pytest.mark.parametrize(
    ("metric", "value", "check"),
    [
        ("brier_bootstrap_upper", 0.20, "brier_bootstrap_upper_better_than_baseline"),
        ("logloss_bootstrap_upper", 0.60, "logloss_bootstrap_upper_better_than_baseline"),
        ("ece", 0.051, "ece_lte_0_05"),
        ("interval_coverage", 0.84, "interval_coverage_0_85_to_0_95"),
        ("alpha_mae", 0.04, "alpha_mae_better_than_zero"),
        ("top_bucket_edge_lower", 0.0, "top_bucket_edge_lower_gt_zero"),
        ("lambda_fold_max", 0.51, "lambda_fold_range_lte_0_20"),
    ],
)
def test_calibration_recomputes_every_metric_gate(metric: str, value: float, check: str) -> None:
    calibration = copy.deepcopy(_inputs()["calibration"])
    calibration["metrics"][metric] = value
    payload = _payload(calibration=calibration)

    assert payload["calibration_ready"] is False
    assert payload["calibration"]["checks"][check] is False
    assert f"calibration_gate_failed:{check}" in payload["blockers"]


def test_calibration_requires_300_samples_and_8_cohorts_per_branch() -> None:
    calibration = copy.deepcopy(_inputs()["calibration"])
    calibration["branches"]["llm"] = {
        "samples": 299,
        "nonoverlap_cohorts": 7,
    }
    payload = _payload(calibration=calibration)

    assert payload["calibration_ready"] is False
    assert payload["calibration"]["checks"]["branch_samples_and_cohorts"] is False


def test_atomic_owner_only_v16_roundtrip_and_v15_path_rejection(
    tmp_path: Path,
) -> None:
    payload = _payload()
    path = tmp_path / "results/v16/synthetic-run/v16_run_readiness.json"
    reference = write_v16_run_readiness(path, payload)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert reference["path"] == "results/v16/synthetic-run/v16_run_readiness.json"
    assert load_v16_run_readiness(path, expected_sha256=reference["sha256"]) == payload
    with pytest.raises(V16ReadinessError, match="results/v16"):
        write_v16_run_readiness(
            tmp_path / "results/v15/synthetic-run/v16_run_readiness.json",
            payload,
        )


def test_v15_payload_cannot_masquerade_as_v16() -> None:
    portfolio = {"valid": True, "target_weights": {SYNTHETIC_SYMBOL: 0.1}}
    v15 = build_v15_run_readiness(
        run_id="synthetic-v15-run",
        generated_at="2026-07-17T09:00:00+08:00",
        analysis_trade_date="2026-07-16",
        market_data_ready=True,
        market_data_blockers=[],
        branch_readiness={
            branch: {"status": "pass", "blockers": []}
            for branch in ("quant", "fundamental", "macro")
        },
        branch_objects={branch: True for branch in ("quant", "fundamental", "macro")},
        factor_governance={
            "governance_status": "ready",
            "production_eligible": True,
            "blockers": [],
        },
        candidate_decision={"candidate_decision_status": "complete"},
        portfolio_constructor=portfolio,
        human_authorization=None,
    )
    masquerade = dict(v15, schema_version="v16_run_readiness.v1")

    with pytest.raises(V16ReadinessError, match="v16 readiness fields mismatch"):
        validate_v16_run_readiness(masquerade)
