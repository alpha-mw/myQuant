from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from quant_investor.factors.governance_protocol_v4 import (
    assess_factor_governance_readiness_v4,
    protocol_hash,
    semantic_sha256,
)
from quant_investor.factors.governance_quality_v1 import (
    FactorQualityV1Error,
    assess_factor_quality_readiness_v4,
    factor_quality_policy_hash,
    factor_quality_set_identity_sha256,
    validate_factor_quality_readiness_v1,
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


def _month_ends(calendar: dict) -> list[str]:
    by_month: dict[str, str] = {}
    for item in calendar["open_session_dates"]:
        by_month[item[:7]] = item
    return list(by_month.values())[:12]


def _records(
    count: int,
    *,
    family_count: int,
    state: str = "production_candidate",
    weight: float = 0.0,
) -> list[dict]:
    calendar = _calendar()
    records = []
    for index in range(count):
        name = f"quality_factor_{index}"
        family = f"quality_family_{index % family_count}"
        records.append(
            {
                "name": name,
                "family": family,
                "slot": f"{family}::slot_{index}",
                "state": state,
                "weight": weight,
                "calendar_sha256": semantic_sha256(calendar),
                "gate_results": {str(gate_id): True for gate_id in range(1, 9)},
                "maturity": {
                    "calendar": calendar,
                    "month_end_rankic_dates": _month_ends(calendar),
                    "forward_cohorts": [],
                },
                "bh_q_value": 0.05,
                "fdr_method": "benjamini_hochberg_by_family",
                "runtime_contract": {
                    "schema_version": "factor-production-runtime-contract.v4",
                    "factor_name": name,
                },
                "runtime_contract_status": "verified",
                "evidence": {
                    "schema_version": "factor-governance-replay-evidence.v4",
                    "status": "verified",
                    "replay_semantic_sha256": _digest(f"replay:{name}"),
                },
                "health": {"status": "healthy", "fresh": True, "data_blocked": False},
            }
        )
    identity_sha256 = factor_quality_set_identity_sha256(records)
    for record in records:
        record["runtime_contract"]["quality_set_identity_sha256"] = identity_sha256
        record["runtime_contract_sha256"] = semantic_sha256(record["runtime_contract"])
    return records


def _receipt(records: list[dict], *, factor_set_sha256: str) -> dict:
    runtime_sha256 = semantic_sha256(
        sorted(record["runtime_contract_sha256"] for record in records)
    )
    context = {
        "protocol_hash": protocol_hash(),
        "transaction_plan_sha256": _digest("transaction-plan"),
        "registry_file_sha256": _digest("registry"),
        "production_factor_set_sha256": factor_set_sha256,
        "runtime_contracts_sha256": runtime_sha256,
        "as_of": "2026-07-17",
    }
    receipt = {
        "schema_version": "factor-governance-activation-receipt.v4",
        "protocol_version": "v4",
        "protocol_hash": protocol_hash(),
        "receipt_id": "quality-layer-test-receipt",
        "status": "activated",
        "authorization_scope": "factor_v4_production_activation",
        "authorized_by": "Maxwell",
        "activated_at": "2026-07-17T09:00:00+08:00",
        "as_of": "2026-07-17",
        "transaction_plan_sha256": context["transaction_plan_sha256"],
        "registry_file_sha256": context["registry_file_sha256"],
        "production_factor_set_sha256": factor_set_sha256,
        "runtime_contracts_sha256": runtime_sha256,
        "activation_context_sha256": semantic_sha256(context),
        "activation_performed": True,
    }
    receipt["receipt_sha256"] = activation_receipt_sha256(receipt)
    return receipt


def _production_kwargs(records: list[dict], receipt: dict | None) -> dict:
    factor_set_sha256 = production_factor_set_sha256(sorted(record["name"] for record in records))
    return {
        "as_of": "2026-07-17",
        "registry_file_sha256": _digest("registry"),
        "production_factor_set_sha256": factor_set_sha256,
        "activation_receipt": receipt,
    }


def test_quality_policy_is_independent_and_hashes_are_pinned() -> None:
    assert protocol_hash() == "9bf5bf2923a3ae0549f9eef570759aaa749e8fccdeedb426db0fa0298f6c0f09"
    assert (
        factor_quality_policy_hash()
        == "8484e33d2e6defdca70991898f63d2ef91bcb89f96f3b97ad09f815976c42d02"
    )


def test_zero_weight_candidate_set_can_qualify_without_production_authority() -> None:
    records = _records(5, family_count=3)
    quality = assess_factor_quality_readiness_v4(records)

    assert validate_factor_quality_readiness_v1(quality) == quality
    assert quality["status"] == "ready_underfilled"
    assert quality["quality_ready"] is True
    assert quality["shadow_observation_eligible"] is True
    assert quality["production_authority"] is False
    assert all("state" not in row and "weight" not in row for row in quality["source_records"])

    legacy = assess_factor_governance_readiness_v4(
        records,
        **_production_kwargs(records, receipt=None),
    )
    with_quality = assess_factor_governance_readiness_v4(
        records,
        **_production_kwargs(records, receipt=None),
        quality_records=records,
    )
    embedded_quality = with_quality.pop("quality_assessment")

    assert with_quality == legacy
    assert legacy["factor_governance_ready"] is False
    assert legacy["new_risk_eligible"] is False
    assert "activation_receipt_missing" in legacy["blockers"]
    assert embedded_quality["quality_ready"] is True


@pytest.mark.parametrize(
    ("count", "family_count", "status", "quality_ready", "shadow_ready"),
    [
        (0, 1, "blocked", False, False),
        (2, 2, "insufficient_for_shadow", False, False),
        (3, 3, "shadow_observation", False, True),
        (5, 3, "ready_underfilled", True, True),
        (10, 3, "ready_target_10", True, True),
        (11, 3, "shadow_observation_above_target", False, True),
    ],
)
def test_quality_status_precedence(
    count: int,
    family_count: int,
    status: str,
    quality_ready: bool,
    shadow_ready: bool,
) -> None:
    assessment = assess_factor_quality_readiness_v4(_records(count, family_count=family_count))
    assert assessment["status"] == status
    assert assessment["quality_ready"] is quality_ready
    assert assessment["shadow_observation_eligible"] is shadow_ready


def test_quality_integrity_drift_and_v2_evidence_fail_closed() -> None:
    identity_drift = _records(5, family_count=3)
    identity_drift[0]["runtime_contract"]["quality_set_identity_sha256"] = "a" * 64
    identity_drift[0]["runtime_contract_sha256"] = semantic_sha256(
        identity_drift[0]["runtime_contract"]
    )
    drifted = assess_factor_quality_readiness_v4(identity_drift)
    assert drifted["status"] == "invalid"
    assert "quality_runtime_set_identity_binding_invalid" in drifted["blockers"]

    runtime_drift = _records(5, family_count=3)
    runtime_drift[0]["runtime_contract"]["formula"] = "tampered_without_rehash"
    runtime_rejected = assess_factor_quality_readiness_v4(runtime_drift)
    assert runtime_rejected["status"] == "partially_qualified"
    assert any(
        "quality_runtime_contract_sha256_mismatch" in item for item in runtime_rejected["blockers"]
    )

    legacy_evidence = _records(5, family_count=3)
    legacy_evidence[0]["evidence"]["schema_version"] = "factor-governance-replay-evidence.v3"
    rejected = assess_factor_quality_readiness_v4(legacy_evidence)
    assert rejected["status"] == "partially_qualified"
    assert rejected["qualified_factor_count"] == 4
    assert any("quality_v4_evidence_schema_mismatch" in item for item in rejected["blockers"])


def test_quality_expected_hash_duplicate_slot_and_malformed_input_are_invalid() -> None:
    records = _records(5, family_count=3)
    hash_mismatch = assess_factor_quality_readiness_v4(
        records, expected_quality_set_sha256="b" * 64
    )
    assert hash_mismatch["status"] == "invalid"
    assert "quality_set_sha256_mismatch" in hash_mismatch["blockers"]

    duplicate = _records(5, family_count=3)
    duplicate[1]["slot"] = duplicate[0]["slot"]
    duplicate_assessment = assess_factor_quality_readiness_v4(duplicate)
    assert duplicate_assessment["status"] == "invalid"
    assert "quality_factor_slots_not_unique" in duplicate_assessment["blockers"]

    duplicate_name = _records(5, family_count=3)
    duplicate_name[1]["name"] = duplicate_name[0]["name"]
    duplicate_name_assessment = assess_factor_quality_readiness_v4(duplicate_name)
    assert duplicate_name_assessment["status"] == "invalid"
    assert "quality_factor_names_not_unique" in duplicate_name_assessment["blockers"]

    malformed = assess_factor_quality_readiness_v4({"not": "an array"})
    assert malformed["status"] == "invalid"
    assert malformed["input_valid"] is False
    assert malformed["blockers"] == ["quality_records_input_not_sequence"]


def test_validator_recomputes_claims_even_after_resealing() -> None:
    assessment = assess_factor_quality_readiness_v4(_records(5, family_count=3))
    tampered = copy.deepcopy(assessment)
    tampered["quality_ready"] = False
    tampered["assessment_sha256"] = semantic_sha256(
        {key: value for key, value in tampered.items() if key != "assessment_sha256"}
    )

    with pytest.raises(FactorQualityV1Error, match="recomputation mismatch"):
        validate_factor_quality_readiness_v1(tampered)


def test_cli_invalid_quality_input_does_not_change_production_exit(tmp_path: Path) -> None:
    records = _records(
        5,
        family_count=5,
        state="production_factor",
        weight=1.0,
    )
    factor_set_sha256 = production_factor_set_sha256(sorted(record["name"] for record in records))
    input_payload = {
        "factor_records": records,
        "as_of": "2026-07-17",
        "registry_file_sha256": _digest("registry"),
        "production_factor_set_sha256": factor_set_sha256,
        "activation_receipt": _receipt(records, factor_set_sha256=factor_set_sha256),
    }
    input_path = tmp_path / "production-input.json"
    output_path = tmp_path / "readiness.json"
    quality_path = tmp_path / "bad-quality.json"
    input_path.write_text(json.dumps(input_payload), encoding="utf-8")
    quality_path.write_text("{", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_factor_v4_readiness_plan.py",
            "--input-json",
            str(input_path),
            "--output-json",
            str(output_path),
            "--quality-records-json",
            str(quality_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert report["factor_governance_ready"] is True
    assert report["quality_assessment"]["status"] == "invalid"
    assert "factor_quality_input_error=quality_input_unreadable_or_invalid_json" in completed.stderr

    valid_output_path = tmp_path / "readiness-with-quality.json"
    quality_path.write_text(
        json.dumps(
            {
                "schema_version": "factor-quality-input.v1",
                "protocol_version": "v4",
                "quality_records": records,
                "expected_quality_set_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_factor_v4_readiness_plan.py",
            "--input-json",
            str(input_path),
            "--output-json",
            str(valid_output_path),
            "--quality-records-json",
            str(quality_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    valid_report = json.loads(valid_output_path.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert valid_report["factor_governance_ready"] is True
    assert valid_report["quality_assessment"]["status"] == "ready_underfilled"
    assert "factor_quality_status=ready_underfilled" in completed.stdout
