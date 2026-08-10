from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from quant_investor.factors.governance_v5 import (
    build_admitted_factor_set,
    build_governance_policy,
    build_preregistration,
    build_prospective_evaluation,
)
from quant_investor.intelligence_v2 import (
    IntelligenceV2ContractError,
    build_initial_pool,
    build_investment_data_readiness,
    build_quant_branch_v5,
    build_quant_pool_policy,
    build_readiness_policy,
    build_subject_branch_binding,
    validate_initial_pool,
    validate_investment_data_readiness,
    validate_quant_branch_v5,
    validate_subject_branch_binding,
)
from quant_investor.intelligence_v2._core import canonical_bytes

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
AS_OF = "2026-08-07T08:00:00Z"
CUTOFF = "2026-08-07T07:00:00Z"


def exact_ref(name: str, *, cutoff: str = CUTOFF, available_at: str = CUTOFF):
    return {
        "artifact_id": name,
        "artifact_version": f"fixture.{name}.v1",
        "available_at": available_at,
        "byte_sha256": SHA_A,
        "cutoff": cutoff,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": SHA_B,
    }


def readiness_policy():
    return build_readiness_policy(
        created_at="2026-08-01T00:00:00Z",
        fundamental_max_stale_sessions=1,
        macro_stale_after_seconds=172_800,
        macro_block_after_seconds=604_800,
    )


def readiness_closure(**overrides):
    values = {
        "policy": readiness_policy(),
        "target_trade_session": "20260807",
        "market_data_cutoff": CUTOFF,
        "open_sessions": ["20260805", "20260806", "20260807"],
        "market_session": "20260807",
        "market_ref": exact_ref("market"),
        "pit_session": "20260807",
        "pit_ref": exact_ref("pit"),
        "fundamental_session": "20260806",
        "fundamental_ref": exact_ref("fundamental"),
        "macro_observed_at": "2026-08-06T00:00:00Z",
        "macro_latest_expected_release_at": "2026-08-05T00:00:00Z",
        "macro_ref": exact_ref("macro"),
        "macro_release_calendar_ref": exact_ref("macro-calendar"),
        "as_of": AS_OF,
    }
    values.update(overrides)
    return values


def factor_closure():
    policy = build_governance_policy(
        created_at="2026-06-01T00:00:00Z",
        coverage_threshold="0.800000000000",
        label_horizon_sessions=20,
        minimum_prospective_paths=2,
    )
    preregistration = build_preregistration(
        policy=policy,
        sealed_at="2026-06-02T00:00:00Z",
        evaluation_start_session="20260701",
        evaluation_end_session="20260731",
        label_available_at="2026-08-01T00:00:00Z",
        candidates=[
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
                "candidate_id": "beta",
                "expression": "-cs_rank(total_mv)",
                "family": "size",
                "implementation_sha256": SHA_A,
                "input_fields": ["total_mv"],
                "parameterization": "NONE",
                "role": "PRIMARY",
                "source_sha256": SHA_C,
            },
        ],
    )

    def evaluation(candidate_id: str, first: str, second: str):
        return build_prospective_evaluation(
            policy=policy,
            preregistration=preregistration,
            candidate_id=candidate_id,
            path_rows=[
                {
                    "path_id": "path_1",
                    "path_ic": first,
                    "purge_proof_sha256": SHA_A,
                    "split_sha256": SHA_B,
                    "test_block_ids": [1],
                },
                {
                    "path_id": "path_2",
                    "path_ic": second,
                    "purge_proof_sha256": SHA_A,
                    "split_sha256": SHA_C,
                    "test_block_ids": [2],
                },
            ],
            evaluation_available_at="2026-08-01T00:00:00Z",
            label_source_sha256=SHA_B,
            implementation_sha256=SHA_A,
            admitted=True,
        )

    evaluations = [
        evaluation("alpha", "0.05", "0.03"),
        evaluation("beta", "0.02", "0.01"),
    ]
    return {
        "policy": policy,
        "preregistration": preregistration,
        "prospective_evaluations": evaluations,
        "built_at": "2026-08-02T00:00:00Z",
    }


def factor_set_and_ref():
    closure = factor_closure()
    factor_set = build_admitted_factor_set(**closure)
    reference = {
        "artifact_id": factor_set["factor_set_id"],
        "artifact_version": factor_set["version"],
        "available_at": "2026-08-02T00:00:00Z",
        "byte_sha256": hashlib.sha256(canonical_bytes(factor_set)).hexdigest(),
        "cutoff": "2026-08-01T00:00:00Z",
        "relative_path": "research/factor-set.v5.json",
        "semantic_sha256": factor_set["semantic_sha256"],
    }
    return closure, factor_set, reference


def universe_rows():
    return [
        {
            "company_code": "000001.SZ",
            "exposures": {"alpha": "0.8", "beta": "0.2"},
            "pit_active": True,
            "security_identity_ref": exact_ref("identity-000001"),
            "tradable": True,
        },
        {
            "company_code": "000002.SZ",
            "exposures": {"alpha": "0.5", "beta": "0.5"},
            "pit_active": True,
            "security_identity_ref": exact_ref("identity-000002"),
            "tradable": True,
        },
        {
            "company_code": "000003.SZ",
            "exposures": {"alpha": "0.1", "beta": "0.9"},
            "pit_active": True,
            "security_identity_ref": exact_ref("identity-000003"),
            "tradable": True,
        },
    ]


def pool_closure(**overrides):
    ready_closure = readiness_closure()
    readiness = build_investment_data_readiness(**ready_closure)
    factor_validation, factor_set, factor_reference = factor_set_and_ref()
    values = {
        "readiness_receipt": readiness,
        "readiness_validation_closure": ready_closure,
        "factor_admitted_set": factor_set,
        "factor_validation_closure": factor_validation,
        "factor_set_ref": factor_reference,
        "policy": build_quant_pool_policy(
            created_at="2026-08-02T00:00:00Z",
            pool_size=2,
            minimum_pool_size=2,
        ),
        "universe_rows": universe_rows(),
        "market_catalog_ref": ready_closure["market_ref"],
        "pit_universe_ref": ready_closure["pit_ref"],
        "as_of": AS_OF,
    }
    values.update(overrides)
    return values


def test_readiness_exposes_one_session_fundamental_lag() -> None:
    closure = readiness_closure()
    receipt = build_investment_data_readiness(**closure)

    rows = {row["name"]: row for row in receipt["rows"]}
    assert rows["MARKET"]["status"] == "AVAILABLE"
    assert rows["PIT_UNIVERSE"]["status"] == "AVAILABLE"
    assert rows["FUNDAMENTAL"]["status"] == "STALE"
    assert receipt["overall_status"] == "STALE"
    assert receipt["quant_inputs_ready"] is True
    assert validate_investment_data_readiness(receipt, **closure) == receipt


def test_fundamental_session_mismatch_can_never_be_available() -> None:
    receipt = build_investment_data_readiness(**readiness_closure(fundamental_session="20260805"))
    row = next(row for row in receipt["rows"] if row["name"] == "FUNDAMENTAL")
    assert row["status"] == "BLOCKED"


def test_macro_release_gap_and_age_fail_closed_without_blocking_quant_inputs() -> None:
    receipt = build_investment_data_readiness(
        **readiness_closure(
            macro_observed_at="2026-08-01T00:00:00Z",
            macro_latest_expected_release_at="2026-08-05T00:00:00Z",
        )
    )
    macro = next(row for row in receipt["rows"] if row["name"] == "MACRO")
    assert macro["status"] == "BLOCKED"
    assert macro["blocker_codes"] == ["MACRO_EXPECTED_RELEASE_MISSING"]
    assert receipt["quant_inputs_ready"] is True


def test_readiness_rejects_future_source_and_resealed_forgery() -> None:
    with pytest.raises(IntelligenceV2ContractError, match="future"):
        build_investment_data_readiness(
            **readiness_closure(
                market_ref=exact_ref(
                    "market",
                    cutoff="2026-08-07T07:00:00Z",
                    available_at="2026-08-08T07:00:00Z",
                )
            )
        )

    closure = readiness_closure()
    receipt = build_investment_data_readiness(**closure)
    forged = deepcopy(receipt)
    forged["overall_status"] = "AVAILABLE"
    with pytest.raises(IntelligenceV2ContractError):
        validate_investment_data_readiness(forged, **closure)


def test_quant_producer_ranks_and_truncates_deterministically() -> None:
    closure = pool_closure()
    pool = build_initial_pool(**closure)

    assert pool["status"] == "AVAILABLE"
    assert [row["company_code"] for row in pool["pool_rows"]] == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert [row["rank"] for row in pool["pool_rows"]] == [1, 2]
    assert validate_initial_pool(pool, **closure) == pool


def test_quant_producer_rejects_binary_float_and_incomplete_target_pool() -> None:
    rows = universe_rows()
    rows[0]["exposures"]["alpha"] = 0.8
    with pytest.raises(IntelligenceV2ContractError, match="binary float"):
        build_initial_pool(**pool_closure(universe_rows=rows))

    blocked_rows = universe_rows()
    blocked_rows[1]["tradable"] = False
    blocked_rows[2]["exposures"].pop("beta")
    blocked = build_initial_pool(**pool_closure(universe_rows=blocked_rows))
    assert blocked["status"] == "BLOCKED"
    assert blocked["pool_rows"] == []


def test_quant_producer_replays_full_factor_v5_closure() -> None:
    closure = pool_closure()
    forged_factor = deepcopy(closure["factor_admitted_set"])
    forged_factor["factor_rows"][0]["weight"] = "1.000000000000"
    with pytest.raises(IntelligenceV2ContractError, match="replay mismatch"):
        build_initial_pool(**{**closure, "factor_admitted_set": forged_factor})


def test_branch_and_subject_binding_preserve_exact_sources() -> None:
    pool_validation = pool_closure()
    pool = build_initial_pool(**pool_validation)
    branch_closure = {
        "initial_pool": pool,
        "pool_validation_closure": pool_validation,
        "company_code": "000001.SZ",
        "as_of": AS_OF,
    }
    branch = build_quant_branch_v5(**branch_closure)
    assert branch["rank"] == 1
    assert validate_quant_branch_v5(branch, **branch_closure) == branch

    binding_closure = {
        "quant_branch": branch,
        "quant_branch_validation_closure": branch_closure,
        "frozen_v1_branch_ref": exact_ref("frozen-v1-quant-branch"),
        "v2_manifest_ref": {
            "artifact_id": "v2-manifest",
            "artifact_version": "myquant.v17.intelligence-v2.package-manifest.v1",
            "byte_sha256": SHA_A,
            "semantic_sha256": SHA_B,
        },
        "bound_at": AS_OF,
    }
    binding = build_subject_branch_binding(**binding_closure)
    assert binding["company_code"] == "000001.SZ"
    assert validate_subject_branch_binding(binding, **binding_closure) == binding
