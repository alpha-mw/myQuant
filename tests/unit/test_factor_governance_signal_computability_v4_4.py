from __future__ import annotations

import copy
import hashlib

import pandas as pd
import pytest

from quant_investor.factors import governance_exact_five_no_label_eval_v4_4 as evaluator
from quant_investor.factors import governance_signal_computability_v4_4 as subject


def _matrices() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    return evaluator.build_synthetic_fixture_v4_4()


def _inputs(
    *, scope: str = subject.SYNTHETIC_SCOPE
) -> tuple[dict, list[dict], dict, dict, list[dict]]:
    matrices, mask = _matrices()
    pit = evaluator.matrix_hash_descriptor_v4_4(mask.astype(float))
    snapshot = {
        "schema_version": subject.SNAPSHOT_BINDING_SCHEMA_VERSION,
        "source_kind": (
            "synthetic_fixture" if scope == subject.SYNTHETIC_SCOPE else "strict_parquet"
        ),
        "market": "CN",
        "universe": "full_a",
        "snapshot_id": "20260720T000000Z",
        "analysis_start": pit["date_axis"]["first"],
        "cutoff": "2026-07-20",
        "latest_trade_date": "2026-07-20",
        "complete_trade_date": "2026-07-20",
        "full_a_count": 3,
        "covered_count": 3,
        "coverage_ratio": 1.0,
        "full_a_semantic_sha256": hashlib.sha256(b"full-a").hexdigest(),
        "snapshot_manifest_sha256": hashlib.sha256(b"snapshot").hexdigest(),
        "table_inventory_sha256": hashlib.sha256(b"table").hexdigest(),
        "pit_membership_sha256": hashlib.sha256(b"pit").hexdigest(),
        "pit_manifest_sha256": hashlib.sha256(b"pit-manifest").hexdigest(),
        "date_axis_sha256": pit["date_axis"]["sha256"],
        "symbol_axis_sha256": pit["symbol_axis"]["sha256"],
        "eligibility_matrix_sha256": pit["matrix_sha256"],
        "pit_mask_descriptor": pit,
        "fallbacks": {
            "csv": False,
            "mock": False,
            "serving": False,
            "stale_pointer": False,
        },
        "strict_full_a_proven": scope == subject.STRICT_SCOPE,
    }
    preregistration = {
        "schema_version": subject.PREREG_BINDING_SCHEMA_VERSION,
        "binding_scope": "synthetic_fixture",
        "cycle_id": "cn_full_a_v4_4_20260720_20260720T000000Z",
        "bundle_path": "/private/tmp/cn_full_a_v4_4_20260720_20260720T000000Z",
        "artifact_count": 27,
        "readback_byte_sha256": hashlib.sha256(b"readback-bytes").hexdigest(),
        "readback_semantic_sha256": hashlib.sha256(b"readback-semantics").hexdigest(),
        "candidate_rows_semantic_sha256": (
            subject.candidate_rows_semantic_sha256_v4_4()
        ),
        "existing_signal_computability": "not_run",
        "existing_authority_false": True,
        "immutable_readback_accepted": False,
    }
    collection = subject.synthetic_fixture_collection_sha256_v4_4()
    passes: list[dict] = []
    for pass_index in (1, 2):
        pass_id = f"fresh_pass_{pass_index}"
        source_outputs = evaluator.evaluate_source_dag_v4_4(matrices, mask)
        local_outputs = evaluator.evaluate_local_formulas_v4_4(matrices, mask)
        passes.append(
            {
                "pass_id": pass_id,
                "collection_sha256": collection,
                "engines": [
                    evaluator.build_engine_pass_result_v4_4(
                        engine_id=evaluator.SOURCE_ENGINE_ID,
                        pass_id=pass_id,
                        collection_sha256=collection,
                        outputs=source_outputs,
                        pit_mask=mask,
                    ),
                    evaluator.build_engine_pass_result_v4_4(
                        engine_id=evaluator.LOCAL_ENGINE_ID,
                        pass_id=pass_id,
                        collection_sha256=collection,
                        outputs=local_outputs,
                        pit_mask=mask,
                    ),
                ],
            }
        )
    return (
        copy.deepcopy(subject.SOURCE_BINDINGS_V4_4),
        copy.deepcopy(list(subject.FIELD_ADAPTERS_V4_4)),
        snapshot,
        preregistration,
        passes,
    )


def _build(scope: str = subject.SYNTHETIC_SCOPE) -> dict:
    source, adapters, snapshot, preregistration, passes = _inputs(scope=scope)
    return subject.build_signal_computability_proof_v4_4(
        evidence_scope=scope,
        source_bindings=source,
        field_adapters=adapters,
        snapshot_binding=snapshot,
        preregistration_binding=preregistration,
        passes=passes,
    )


def _reseal(proof: dict) -> dict:
    result = copy.deepcopy(proof)
    result.pop("artifact_semantic_sha256", None)
    result["artifact_semantic_sha256"] = evaluator.semantic_sha256_v4_4(result)
    return result


def test_synthetic_proof_is_atomic_exact_and_non_authorizing() -> None:
    proof = _build()
    assert subject.validate_signal_computability_proof_v4_4(proof) == proof
    assert proof["candidate_count"] == 5
    assert proof["atomic_exact_five_passed"] is True
    assert proof["independent_engine_equivalence_proven"] is True
    assert proof["strict_snapshot_signal_computability_proven"] is False
    assert proof["measurement"] == subject.MEASUREMENT_FLAGS
    assert not any(proof["authority"].values())
    assert not any(proof["side_effects"].values())
    assert proof["preregistration_binding"]["existing_signal_computability"] == "not_run"


def test_future_strict_scope_is_unavailable_in_synthetic_only_slice() -> None:
    source, adapters, snapshot, preregistration, passes = _inputs(
        scope=subject.STRICT_SCOPE
    )
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="strict snapshot proof construction is unavailable",
    ):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.STRICT_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )


def test_cutoff_and_fallbacks_fail_closed() -> None:
    source, adapters, snapshot, preregistration, passes = _inputs()
    snapshot["cutoff"] = "2026-07-19"
    snapshot["latest_trade_date"] = "2026-07-19"
    snapshot["complete_trade_date"] = "2026-07-19"
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="strictly later",
    ):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.SYNTHETIC_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )

    source, adapters, snapshot, preregistration, passes = _inputs()
    snapshot["fallbacks"]["csv"] = True
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="fallbacks must all be false",
    ):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.SYNTHETIC_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )


def test_source_ast_and_candidate_specific_adapter_tamper_are_rejected() -> None:
    source, adapters, snapshot, preregistration, passes = _inputs()
    source["aquant"]["commit"] = "0" * 40
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="pinned source identities",
    ):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.SYNTHETIC_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )

    source, adapters, snapshot, preregistration, passes = _inputs()
    adapters[2]["physical_columns"] = ["adj_close"]
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="candidate-specific mappings",
    ):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.SYNTHETIC_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )


def test_engine_divergence_and_second_pass_drift_are_rejected() -> None:
    source, adapters, snapshot, preregistration, passes = _inputs()
    local_row = passes[0]["engines"][1]["candidates"][0]
    for descriptor_key in ("raw_matrix", "direction_adjusted_matrix"):
        local_row[descriptor_key]["matrix_sha256"] = "0" * 64
        local_row[descriptor_key]["bit_pattern_sha256"] = "0" * 64
    engine = passes[0]["engines"][1]
    body = {key: value for key, value in engine.items() if key != "result_semantic_sha256"}
    engine["result_semantic_sha256"] = evaluator.semantic_sha256_v4_4(body)
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="independent engines differ",
    ):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.SYNTHETIC_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )

    source, adapters, snapshot, preregistration, passes = _inputs()
    passes[1]["collection_sha256"] = hashlib.sha256(b"drift").hexdigest()
    for engine in passes[1]["engines"]:
        engine["collection_sha256"] = passes[1]["collection_sha256"]
        body = {
            key: value for key, value in engine.items() if key != "result_semantic_sha256"
        }
        engine["result_semantic_sha256"] = evaluator.semantic_sha256_v4_4(body)
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="collections differ",
    ):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.SYNTHETIC_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )


def test_preregistration_must_remain_immutable_not_run_and_hash_bound() -> None:
    source, adapters, snapshot, preregistration, passes = _inputs()
    preregistration["existing_signal_computability"] = "passed"
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="preregistration binding mismatch",
    ):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.SYNTHETIC_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )


def test_proof_authority_or_measurement_tamper_fails_after_reseal() -> None:
    proof = _build()
    proof["authority"]["candidate_qualified"] = True
    proof = _reseal(proof)
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="proof mismatch",
    ):
        subject.validate_signal_computability_proof_v4_4(proof)

    proof = _build()
    proof["measurement"]["family_bh"] = "passed"
    proof = _reseal(proof)
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="proof mismatch",
    ):
        subject.validate_signal_computability_proof_v4_4(proof)


def test_resealed_descriptor_substitution_is_recomputed_and_rejected() -> None:
    proof = _build()
    pit_descriptor = copy.deepcopy(proof["snapshot_binding"]["pit_mask_descriptor"])
    for pass_value in proof["passes"]:
        for engine in pass_value["engines"]:
            engine["candidates"][0]["raw_matrix"] = copy.deepcopy(pit_descriptor)
            engine["candidates"][0]["direction_adjusted_matrix"] = copy.deepcopy(
                pit_descriptor
            )
            body = {
                key: value
                for key, value in engine.items()
                if key != "result_semantic_sha256"
            }
            engine["result_semantic_sha256"] = evaluator.semantic_sha256_v4_4(body)
    forged = _reseal(proof)
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="independent deterministic fixture recomputation",
    ):
        subject.validate_signal_computability_proof_v4_4(forged)


def test_readback_is_independently_rebuilt_and_bound_to_proof_bytes() -> None:
    proof = _build()
    readback = subject.build_signal_computability_readback_v4_4(proof=proof)
    assert (
        subject.validate_signal_computability_readback_v4_4(
            readback, proof=proof
        )
        == readback
    )
    tampered = copy.deepcopy(readback)
    tampered["proof_byte_sha256"] = "0" * 64
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="independent rebuild",
    ):
        subject.validate_signal_computability_readback_v4_4(tampered, proof=proof)


def test_unexpected_outcome_surface_and_schema_fields_are_rejected() -> None:
    source, adapters, snapshot, preregistration, passes = _inputs()
    snapshot["fallbacks"]["label"] = False
    with pytest.raises(subject.FactorGovernanceSignalComputabilityV4_4Error):
        subject.build_signal_computability_proof_v4_4(
            evidence_scope=subject.SYNTHETIC_SCOPE,
            source_bindings=source,
            field_adapters=adapters,
            snapshot_binding=snapshot,
            preregistration_binding=preregistration,
            passes=passes,
        )

    proof = _build()
    proof["ic"] = 0.1
    proof = _reseal(proof)
    with pytest.raises(
        subject.FactorGovernanceSignalComputabilityV4_4Error,
        match="fields are not exact",
    ):
        subject.validate_signal_computability_proof_v4_4(proof)
