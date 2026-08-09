from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from quant_investor.factors.governance_v5 import (
    FactorGovernanceV5Error,
    build_admitted_factor_set,
    build_coverage_receipt,
    build_diagnostic_scan_receipt,
    build_governance_policy,
    build_preregistration,
    build_prospective_evaluation,
    build_substitution_receipt,
    canonical_bytes,
    historical_support_projection,
    strict_json_loads,
    validate_governance_policy,
    validate_preregistration,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CREATED = "2026-08-08T00:00:00Z"
LABEL_AT = "2026-09-30T00:00:00Z"


def policy(**overrides):
    values = {
        "created_at": CREATED,
        "coverage_threshold": "0.800000000000",
        "label_horizon_sessions": 20,
        "minimum_prospective_paths": 2,
    }
    values.update(overrides)
    return build_governance_policy(**values)


def candidates():
    return [
        {
            "candidate_id": "alpha",
            "expression": "cs_rank(pb)",
            "family": "value",
            "implementation_sha256": SHA_A,
            "input_fields": ["pb"],
            "parameterization": "NONE",
            "role": "PRIMARY",
            "source_sha256": SHA_B,
        },
        {
            "candidate_id": "alpha_alt",
            "expression": "cs_rank(fcf_to_price)",
            "family": "value",
            "implementation_sha256": SHA_A,
            "input_fields": ["fcf_to_price"],
            "parameterization": "NONE",
            "role": "ALTERNATE_FOR:alpha",
            "source_sha256": SHA_C,
        },
        {
            "candidate_id": "beta",
            "expression": "-cs_rank(total_mv)",
            "family": "size",
            "implementation_sha256": SHA_A,
            "input_fields": ["total_mv"],
            "parameterization": "NONE",
            "role": "PRIMARY",
            "source_sha256": SHA_B,
        },
    ]


def prereg(p=None):
    active = p or policy()
    return build_preregistration(
        policy=active,
        sealed_at="2026-08-08T01:00:00Z",
        evaluation_start_session="20260901",
        evaluation_end_session="20260930",
        label_available_at=LABEL_AT,
        candidates=candidates(),
    )


def coverage(p, registration, candidate_id, numerator):
    return build_coverage_receipt(
        policy=p,
        preregistration=registration,
        candidate_id=candidate_id,
        numerator=numerator,
        denominator=100,
        pit_universe_sha256=SHA_A,
        input_source_sha256=SHA_B,
        cutoff="2026-08-08T01:30:00Z",
        computed_at="2026-08-08T02:00:00Z",
        label_reader_permitted_at="2026-08-08T04:00:00Z",
    )


def evaluation(p, registration, candidate_id, path_values, admitted=True):
    rows = [
        {
            "path_id": f"path_{index}",
            "path_ic": value,
            "purge_proof_sha256": SHA_A,
            "split_sha256": chr(ord("a") + index) * 64,
            "test_block_ids": [index],
        }
        for index, value in enumerate(path_values)
    ]
    return build_prospective_evaluation(
        policy=p,
        preregistration=registration,
        candidate_id=candidate_id,
        path_rows=rows,
        evaluation_available_at=LABEL_AT,
        label_source_sha256=SHA_B,
        implementation_sha256=SHA_A,
        admitted=admitted,
    )


def test_policy_is_isolated_single_factor_and_replay_sealed():
    document = policy()
    assert document["factor_protocol"] == "factor-governance-protocol.v5"
    assert document["minimum_admitted_factors"] == 1
    assert document["minimum_admitted_families"] == 1
    assert document["factor_composite_max_factor_weight"] == "1.000000000000"
    assert validate_governance_policy(document) == document

    forged = deepcopy(document)
    forged["coverage_threshold"] = "0.700000000000"
    with pytest.raises(FactorGovernanceV5Error):
        validate_governance_policy(forged)


def test_canonical_json_rejects_float_duplicate_key_and_non_nfc():
    with pytest.raises(FactorGovernanceV5Error, match="binary float"):
        canonical_bytes({"value": 0.1})
    with pytest.raises(FactorGovernanceV5Error, match="duplicate JSON key"):
        strict_json_loads(b'{"a":1,"a":2}')
    with pytest.raises(FactorGovernanceV5Error, match="NFC"):
        canonical_bytes({"text": "e\u0301"})


def test_preregistration_rejects_available_label_and_invalid_alternate():
    p = policy()
    with pytest.raises(FactorGovernanceV5Error, match="already available"):
        build_preregistration(
            policy=p,
            sealed_at="2026-08-08T01:00:00Z",
            evaluation_start_session="20260901",
            evaluation_end_session="20260930",
            label_available_at="2026-08-08T01:00:00Z",
            candidates=candidates(),
        )
    rows = candidates()
    rows[1]["family"] = "other_family"
    with pytest.raises(FactorGovernanceV5Error, match="share the primary family"):
        build_preregistration(
            policy=p,
            sealed_at="2026-08-08T01:00:00Z",
            evaluation_start_session="20260901",
            evaluation_end_session="20260930",
            label_available_at=LABEL_AT,
            candidates=rows,
        )


def test_coverage_and_one_time_pre_label_substitution():
    p = policy()
    registration = prereg(p)
    primary = coverage(p, registration, "alpha", 79)
    alternate = coverage(p, registration, "alpha_alt", 80)
    receipt = build_substitution_receipt(
        policy=p,
        preregistration=registration,
        primary_coverage=primary,
        alternate_coverage=alternate,
        substituted_at="2026-08-08T03:00:00Z",
    )
    assert receipt["reason"] == "PREDEFINED_COVERAGE_GATE_FAILED"

    with pytest.raises(FactorGovernanceV5Error, match="after label access"):
        build_substitution_receipt(
            policy=p,
            preregistration=registration,
            primary_coverage=primary,
            alternate_coverage=alternate,
            substituted_at="2026-08-08T04:00:00Z",
        )


def test_prospective_and_historical_lanes_are_not_interchangeable():
    p = policy()
    registration = prereg(p)
    prospective = evaluation(p, registration, "alpha", ["0.10", "0.20"])
    assert prospective["lane"] == "PROSPECTIVE_ONLY"
    assert prospective["parameter_stability_status"] == "NOT_APPLICABLE"
    support = historical_support_projection(
        candidate_id="alpha",
        mean_path_ic="0.10",
        path_count=10,
        produced_at=LABEL_AT,
        source_sha256=SHA_A,
    )
    assert support["lane"] == "BACKTEST_SUPPORT_ONLY"
    assert support["admission_eligible"] is False
    with pytest.raises(FactorGovernanceV5Error):
        build_admitted_factor_set(
            policy=p,
            preregistration=registration,
            prospective_evaluations=[support],
            built_at="2026-09-30T01:00:00Z",
        )


def test_diagnostic_lane_is_explicitly_non_promotable():
    receipt = build_diagnostic_scan_receipt(
        scanned_at=CREATED,
        implementation_sha256=SHA_A,
        candidate_ids=[f"candidate_{index:03d}" for index in range(230)],
    )
    assert receipt["lane"] == "DIAGNOSTIC_ONLY"
    assert receipt["promotion_eligible"] is False


def test_shrunk_weights_are_decimal_order_independent_and_sum_exactly():
    p = policy()
    registration = prereg(p)
    alpha = evaluation(p, registration, "alpha", ["0.20", "0.10"])
    beta = evaluation(p, registration, "beta", ["0.10", "0.10"])
    left = build_admitted_factor_set(
        policy=p,
        preregistration=registration,
        prospective_evaluations=[alpha, beta],
        built_at="2026-09-30T01:00:00Z",
    )
    right = build_admitted_factor_set(
        policy=p,
        preregistration=registration,
        prospective_evaluations=[beta, alpha],
        built_at="2026-09-30T01:00:00Z",
    )
    assert left == right
    assert sum((Decimal(row["weight"]) for row in left["factor_rows"]), Decimal("0")) == Decimal(
        "1"
    )


def test_single_factor_receives_full_composite_budget_but_no_portfolio_authority():
    p = policy()
    registration = prereg(p)
    alpha = evaluation(p, registration, "alpha", ["0.20", "0.10"])
    document = build_admitted_factor_set(
        policy=p,
        preregistration=registration,
        prospective_evaluations=[alpha],
        built_at="2026-09-30T01:00:00Z",
    )
    assert document["factor_rows"][0]["weight"] == "1.000000000000"
    assert document["authority"]["portfolio"] is False
    assert document["authority"]["production"] is False


def test_all_zero_shrunk_values_block():
    p = policy()
    registration = prereg(p)
    alpha = evaluation(p, registration, "alpha", ["-0.20", "0.00"])
    with pytest.raises(FactorGovernanceV5Error, match="all shrunk IC"):
        build_admitted_factor_set(
            policy=p,
            preregistration=registration,
            prospective_evaluations=[alpha],
            built_at="2026-09-30T01:00:00Z",
        )


def test_replay_rejects_resealed_preregistration_forgery():
    p = policy()
    document = prereg(p)
    forged = deepcopy(document)
    forged["candidates"][0]["expression"] = "cs_rank(pe)"
    with pytest.raises(FactorGovernanceV5Error):
        validate_preregistration(forged, policy=p)
