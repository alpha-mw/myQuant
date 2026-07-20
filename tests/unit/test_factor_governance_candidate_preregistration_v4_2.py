from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable

import pytest

from quant_investor.factors import governance_candidate_preregistration_v4_2 as subject
from quant_investor.factors import governance_cycle_state_v4_1 as cycle_state


ERR = subject.FactorGovernanceCandidatePreregistrationV4_2Error
CYCLE_ID = "cn_full_a_v4_2_20260720_20260720T172132Z"
CODE_BINDING_SET_SHA = hashlib.sha256(b"code-bindings").hexdigest()
STRICT_SOURCE_BINDING_SHA = hashlib.sha256(b"strict-source-binding").hexdigest()
FULL_A_SCOPE_SHA = hashlib.sha256(b"full-a-scope").hexdigest()
FULL_A_SCOPE_COUNT = 5502
SERVING_INVENTORY_COUNT = 5502


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["artifact_semantic_sha256"] = subject.semantic_sha256_v4_2(
        {
            key: value
            for key, value in result.items()
            if key != "artifact_semantic_sha256"
        }
    )
    return result


def _fixture() -> dict[str, Any]:
    aquant = subject.build_aquant_receipt_v4_2()
    myquant = subject.build_myquant_receipt_v4_2()
    operators = subject.build_operator_semantics_v4_2()
    comparison = subject.build_comparison_catalog_receipt_v4_2(
        catalog_id="synthetic-v4.1-comparison",
        catalog_byte_sha256=_sha("comparison-catalog-byte"),
        catalog_semantic_sha256=_sha("comparison-catalog"),
        primitive_count=38,
        definition_identity_inventory=[
            {
                "name": "base_alpha_001",
                "definition_identity_sha256": _sha("base-definition-001"),
            },
            {
                "name": "base_alpha_002",
                "definition_identity_sha256": _sha("base-definition-002"),
            },
        ],
    )
    selection = subject.build_selection_spec_v4_2(
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    envelope = subject.build_future_source_envelope_v4_2(
        cycle_id=CYCLE_ID,
        analysis_start="2021-06-25",
        cutoff="2026-07-20",
        snapshot_id="20260720T172132Z",
        snapshot_date="2026-07-20",
        strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
        full_a_scope_sha256=FULL_A_SCOPE_SHA,
        full_a_scope_count=FULL_A_SCOPE_COUNT,
        serving_inventory_count=SERVING_INVENTORY_COUNT,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
    )
    predecessor = cycle_state.build_genesis_cycle_state_v4_1(
        cycle_id=CYCLE_ID,
        cycle_root_sha256=_sha("cycle-root"),
        source_chain_node_sha256=envelope["artifact_semantic_sha256"],
    )
    collision = subject.build_definition_identity_collision_audit_v4_2(
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    source_node = subject.build_prereg_discovery_source_node_v4_2(
        predecessor_state=predecessor,
        future_source_envelope=envelope,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
        strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
        full_a_scope_sha256=FULL_A_SCOPE_SHA,
        full_a_scope_count=FULL_A_SCOPE_COUNT,
        serving_inventory_count=SERVING_INVENTORY_COUNT,
    )
    predecessor_byte = cycle_state.byte_sha256(predecessor)
    orchestration = subject.build_preregistration_discovery_cycle_v4_2(
        predecessor_state=predecessor,
        predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_semantic_sha256=predecessor["state_semantic_sha256"],
        future_source_envelope=envelope,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
        strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
        full_a_scope_sha256=FULL_A_SCOPE_SHA,
        full_a_scope_count=FULL_A_SCOPE_COUNT,
        serving_inventory_count=SERVING_INVENTORY_COUNT,
    )
    return {
        "envelope": envelope,
        "predecessor": predecessor,
        "predecessor_byte": predecessor_byte,
        "aquant": aquant,
        "myquant": myquant,
        "operators": operators,
        "comparison": comparison,
        "selection": selection,
        "collision": collision,
        "source_node": source_node,
        "orchestration": orchestration,
    }


def _validate_selection(value: dict[str, Any], fx: dict[str, Any]) -> dict[str, Any]:
    return subject.validate_selection_spec_v4_2(
        value,
        aquant_receipt=fx["aquant"],
        myquant_receipt=fx["myquant"],
        operator_semantics=fx["operators"],
        comparison_catalog_receipt=fx["comparison"],
    )


def _validate_collision(value: dict[str, Any], fx: dict[str, Any]) -> dict[str, Any]:
    return subject.validate_definition_identity_collision_audit_v4_2(
        value,
        selection_spec=fx["selection"],
        aquant_receipt=fx["aquant"],
        myquant_receipt=fx["myquant"],
        operator_semantics=fx["operators"],
        comparison_catalog_receipt=fx["comparison"],
    )


def _validate_envelope(value: dict[str, Any], fx: dict[str, Any]) -> dict[str, Any]:
    return subject.validate_future_source_envelope_v4_2(
        value,
        selection_spec=fx["selection"],
        aquant_receipt=fx["aquant"],
        myquant_receipt=fx["myquant"],
        operator_semantics=fx["operators"],
        comparison_catalog_receipt=fx["comparison"],
        code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
        strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
        full_a_scope_sha256=FULL_A_SCOPE_SHA,
        full_a_scope_count=FULL_A_SCOPE_COUNT,
        serving_inventory_count=SERVING_INVENTORY_COUNT,
    )


def _validate_orchestration(
    value: dict[str, Any], fx: dict[str, Any]
) -> dict[str, Any]:
    return subject.validate_preregistration_discovery_cycle_v4_2(
        value,
        predecessor_state=fx["predecessor"],
        predecessor_byte_sha256=fx["predecessor_byte"],
        expected_predecessor_byte_sha256=fx["predecessor_byte"],
        expected_predecessor_semantic_sha256=fx["predecessor"][
            "state_semantic_sha256"
        ],
        future_source_envelope=fx["envelope"],
        selection_spec=fx["selection"],
        aquant_receipt=fx["aquant"],
        myquant_receipt=fx["myquant"],
        operator_semantics=fx["operators"],
        comparison_catalog_receipt=fx["comparison"],
        definition_identity_collision_audit=fx["collision"],
        code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
        strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
        full_a_scope_sha256=FULL_A_SCOPE_SHA,
        full_a_scope_count=FULL_A_SCOPE_COUNT,
        serving_inventory_count=SERVING_INVENTORY_COUNT,
    )


def test_successful_preregistration_is_exact_four_zero_weight_and_non_authoritative() -> None:
    fx = _fixture()
    selection = _validate_selection(fx["selection"], fx)
    source_node = subject.validate_prereg_discovery_source_node_v4_2(
        fx["source_node"],
        predecessor_state=fx["predecessor"],
        future_source_envelope=fx["envelope"],
        selection_spec=selection,
        aquant_receipt=fx["aquant"],
        myquant_receipt=fx["myquant"],
        operator_semantics=fx["operators"],
        comparison_catalog_receipt=fx["comparison"],
        definition_identity_collision_audit=fx["collision"],
        code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
        strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
        full_a_scope_sha256=FULL_A_SCOPE_SHA,
        full_a_scope_count=FULL_A_SCOPE_COUNT,
        serving_inventory_count=SERVING_INVENTORY_COUNT,
    )
    orchestration = _validate_orchestration(fx["orchestration"], fx)

    assert tuple(row["name"] for row in selection["candidates"]) == (
        subject.EXPECTED_CANDIDATES
    )
    assert [row["order"] for row in selection["candidates"]] == [1, 2, 3, 4]
    assert all(row["initial_weight"] == 0 for row in selection["candidates"])
    assert selection["claims"] == {
        "artifact_and_builder_label_inputs_absent": True,
        "outcome_paths_read": [],
        "outcomes_used_as_evidence": False,
        "selection_uninfluenced_by_any_external_label": "UNPROVEN",
        "authoritative_evidence_route": (
            "prospective_post_preregistration_holdout_only"
        ),
    }
    assert selection["measurement"] == subject.MEASUREMENT_FLAGS
    assert selection["authority"] == subject.AUTHORITY_FLAGS
    assert selection["side_effects"] == subject.SIDE_EFFECT_FLAGS
    assert fx["envelope"]["healthy_source_verified"] is False
    assert source_node["dual_sha_predecessor_transition_validated"] is True
    assert source_node["predecessor_state_byte_sha256"] == fx["predecessor_byte"]
    assert source_node["exact_once_publication"] == "NOT_IMPLEMENTED"
    assert orchestration["discovery_state"]["schema_version"] == (
        cycle_state.STATE_SCHEMA_VERSION
    )
    assert orchestration["discovery_state"]["state"] == cycle_state.DISCOVERY
    assert orchestration["discovery_state"]["source_chain_node_sha256"] == (
        source_node["artifact_semantic_sha256"]
    )
    assert orchestration["persisted_state_sequence"] == [
        cycle_state.PRECOMMITTED,
        cycle_state.DISCOVERY,
    ]
    assert orchestration["precommitted_state_role"] == "INTRA_BUNDLE_LINEAGE_ONLY"
    assert orchestration["discovery_state_role"] == "FINAL_CURRENT"
    assert orchestration["external_state_pointer_mutation"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.reverse(),
        lambda rows: rows.pop(),
        lambda rows: rows.append(copy.deepcopy(rows[-1])),
        lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
        lambda rows: rows[0].__setitem__("name", "unknown_factor"),
        lambda rows: rows[0].__setitem__("initial_weight", 1),
    ],
)
def test_selection_rejects_reorder_missing_extra_duplicate_unknown_and_nonzero_weight(
    mutate: Callable[[list[dict[str, Any]]], None],
) -> None:
    fx = _fixture()
    selection = copy.deepcopy(fx["selection"])
    mutate(selection["candidates"])
    selection = _reseal(selection)

    with pytest.raises(ERR):
        _validate_selection(selection, fx)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("commit", "0" * 40, "commit"),
        ("blob_oid", "1" * 40, "blob OID"),
        ("path", "A_quant/output/factor_mining/deep_candidates_20260526.json", "path"),
        ("raw_sha256", "2" * 64, "raw SHA"),
        ("mode", "100755", "mode"),
        ("outcome_paths_read", ["forbidden.json"], "outcome_paths_read"),
        ("outcomes_used_as_evidence", True, "outcomes_used_as_evidence"),
    ],
)
def test_aquant_receipt_rejects_source_and_outcome_claim_tamper(
    field: str, value: Any, match: str
) -> None:
    receipt = copy.deepcopy(subject.build_aquant_receipt_v4_2())
    receipt[field] = value
    receipt = _reseal(receipt)

    with pytest.raises(ERR, match=match):
        subject.validate_aquant_receipt_v4_2(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate", "unknown"),
        ("expression", "cs_rank(close)"),
        ("family", "forged"),
        ("definition_sha256", "3" * 64),
    ],
)
def test_aquant_receipt_rejects_definition_tamper(field: str, value: str) -> None:
    receipt = copy.deepcopy(subject.build_aquant_receipt_v4_2())
    receipt["definition"][field] = value
    receipt = _reseal(receipt)

    with pytest.raises(ERR):
        subject.validate_aquant_receipt_v4_2(receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.__setitem__("commit", "4" * 40),
        lambda receipt: receipt.__setitem__("blob_oid", "5" * 40),
        lambda receipt: receipt.__setitem__("path", "quant_investor/other.py"),
        lambda receipt: receipt.__setitem__("full_sha256", "6" * 64),
        lambda receipt: receipt["alias_rows"][0].__setitem__("candidate", "bad"),
        lambda receipt: receipt["alias_rows"][0].__setitem__("source_factor", "BAD"),
        lambda receipt: receipt["alias_rows"][0].__setitem__("direction", 1),
        lambda receipt: receipt["alias_rows"][0].__setitem__("source_ast_sha256", "7" * 64),
        lambda receipt: receipt["alias_rows"][0].__setitem__(
            "bound_definition_sha256", "8" * 64
        ),
        lambda receipt: receipt["alias_rows"].reverse(),
    ],
)
def test_myquant_receipt_rejects_source_alias_ast_direction_and_order_tamper(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    receipt = copy.deepcopy(subject.build_myquant_receipt_v4_2())
    mutate(receipt)
    receipt = _reseal(receipt)

    with pytest.raises(ERR):
        subject.validate_myquant_receipt_v4_2(receipt)


@pytest.mark.parametrize(
    "banned_key",
    [
        "ic",
        "score",
        "pvalue",
        "q_value",
        "verdict",
        "backtest",
        "replay",
        "performance",
    ],
)
def test_selection_rejects_any_outcome_or_measurement_field(banned_key: str) -> None:
    fx = _fixture()
    selection = copy.deepcopy(fx["selection"])
    selection["claims"][banned_key] = 0.1
    selection = _reseal(selection)

    with pytest.raises(ERR, match="banned outcome/stat key"):
        _validate_selection(selection, fx)


def test_aquant_receipt_rejects_nested_replay_outcome_field() -> None:
    receipt = copy.deepcopy(subject.build_aquant_receipt_v4_2())
    receipt["definition"]["replay"] = {"status": "passed"}
    receipt = _reseal(receipt)

    with pytest.raises(ERR, match="banned outcome/stat key 'replay'"):
        subject.validate_aquant_receipt_v4_2(receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda operators: operators["semantics"].__setitem__(
            "runtime_equivalence_verified", True
        ),
        lambda operators: operators["semantics"]["std"].__setitem__("ddof", 0),
        lambda operators: operators["semantics"]["runtime"].__setitem__(
            "pandas", "0.0.0"
        ),
        lambda operators: operators.__setitem__("schema_version", "wrong"),
    ],
)
def test_operator_semantics_rejects_runtime_descriptor_tamper(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    operators = copy.deepcopy(subject.build_operator_semantics_v4_2())
    mutate(operators)
    operators = _reseal(operators)

    with pytest.raises(ERR):
        subject.validate_operator_semantics_v4_2(operators)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_count", 266),
        ("primitive_count", 0),
        (
            "definition_identity_inventory",
            [
                {
                    "name": "base_alpha_002",
                    "definition_identity_sha256": _sha("base-definition-002"),
                },
                {
                    "name": "base_alpha_001",
                    "definition_identity_sha256": _sha("base-definition-001"),
                },
            ],
        ),
        ("label_inputs_absent", False),
        ("outcome_fields_absent", False),
    ],
)
def test_comparison_catalog_receipt_rejects_binding_and_no_label_tamper(
    field: str, value: Any
) -> None:
    receipt = subject.build_comparison_catalog_receipt_v4_2(
        catalog_id="synthetic-v4.1-comparison",
        catalog_byte_sha256=_sha("comparison-catalog-byte"),
        catalog_semantic_sha256=_sha("comparison-catalog"),
        primitive_count=38,
        definition_identity_inventory=[
            {
                "name": "base_alpha_001",
                "definition_identity_sha256": _sha("base-definition-001"),
            },
            {
                "name": "base_alpha_002",
                "definition_identity_sha256": _sha("base-definition-002"),
            },
        ],
    )
    receipt[field] = value
    receipt = _reseal(receipt)

    with pytest.raises(ERR):
        subject.validate_comparison_catalog_receipt_v4_2(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("catalog_byte_sha256", "8" * 64),
        ("catalog_semantic_sha256", "9" * 64),
    ],
)
def test_comparison_catalog_byte_and_semantic_tamper_breaks_selection_binding(
    field: str, value: str
) -> None:
    fx = _fixture()
    comparison = copy.deepcopy(fx["comparison"])
    comparison[field] = value
    comparison = _reseal(comparison)

    with pytest.raises(ERR):
        subject.validate_selection_spec_v4_2(
            fx["selection"],
            aquant_receipt=fx["aquant"],
            myquant_receipt=fx["myquant"],
            operator_semantics=fx["operators"],
            comparison_catalog_receipt=comparison,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda audit: audit["selected_vs_selected"].__setitem__(
            "collisions", [{"left": "a", "right": "b"}]
        ),
        lambda audit: audit["selected_vs_comparison"].__setitem__(
            "collisions", [{"selected": "a", "comparison": "b"}]
        ),
        lambda audit: audit.__setitem__(
            "definition_identity_collision_detected", True
        ),
        lambda audit: audit.__setitem__("duplicate_primitive", False),
        lambda audit: audit.__setitem__("structural_dedup", "passed"),
        lambda audit: audit.__setitem__("formal_dedup", "passed"),
        lambda audit: audit.__setitem__("high_correlation_dedup", "passed"),
    ],
)
def test_definition_identity_audit_rejects_collisions_and_keeps_dedup_not_run(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    fx = _fixture()
    audit = copy.deepcopy(fx["collision"])
    mutate(audit)
    audit = _reseal(audit)

    with pytest.raises(ERR):
        _validate_collision(audit, fx)


@pytest.mark.parametrize("collision_kind", ["name", "definition_identity"])
def test_definition_identity_audit_recomputes_selected_vs_comparison_collisions(
    collision_kind: str,
) -> None:
    fx = _fixture()
    selected = fx["selection"]["candidates"][0]
    if collision_kind == "name":
        comparison_name = selected["name"]
        comparison_identity = _sha("different-comparison-definition")
    else:
        comparison_name = "base_alpha_collision"
        comparison_identity = selected["definition_identity_sha256"]
    comparison = subject.build_comparison_catalog_receipt_v4_2(
        catalog_id="synthetic-v4.1-comparison",
        catalog_byte_sha256=_sha("comparison-catalog-byte"),
        catalog_semantic_sha256=_sha("comparison-catalog"),
        primitive_count=38,
        definition_identity_inventory=[
            {
                "name": comparison_name,
                "definition_identity_sha256": comparison_identity,
            }
        ],
    )

    with pytest.raises(ERR, match="definition identity collision"):
        selection = subject.build_selection_spec_v4_2(
            aquant_receipt=fx["aquant"],
            myquant_receipt=fx["myquant"],
            operator_semantics=fx["operators"],
            comparison_catalog_receipt=comparison,
        )
        subject.build_definition_identity_collision_audit_v4_2(
            selection_spec=selection,
            aquant_receipt=fx["aquant"],
            myquant_receipt=fx["myquant"],
            operator_semantics=fx["operators"],
            comparison_catalog_receipt=comparison,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cutoff", "2026-07-17"),
        ("analysis_start", "2026-07-21"),
        ("snapshot_date", "2026-07-19"),
        ("snapshot_id", "20260721T172132Z"),
        ("latest_trade_date", "2026-07-21"),
        ("latest_complete_trade_date", "2026-07-21"),
        ("market", "US"),
        ("universe", "csi300"),
        ("storage_mode", "csv"),
        ("strict_source_binding_semantic_sha256", "0" * 64),
        ("full_a_scope_sha256", "1" * 64),
        ("full_a_scope_count", 5501),
        ("serving_inventory_count", 5501),
        ("selection_spec_semantic_sha256", "a" * 64),
        ("aquant_receipt_semantic_sha256", "b" * 64),
        ("myquant_receipt_semantic_sha256", "c" * 64),
        ("operator_semantics_sha256", "d" * 64),
        ("comparison_catalog_receipt_semantic_sha256", "e" * 64),
        ("code_binding_set_semantic_sha256", "f" * 64),
        ("blockers", ["missing"]),
        ("healthy_source_verified", True),
        ("source_authority", "VERIFIED"),
        ("publication_status", "PUBLISHED"),
    ],
)
def test_future_source_envelope_rejects_nonfresh_non_strict_and_blocked_inputs(
    field: str, value: Any
) -> None:
    fx = _fixture()
    envelope = copy.deepcopy(fx["envelope"])
    envelope[field] = value
    envelope = _reseal(envelope)

    with pytest.raises(ERR):
        _validate_envelope(envelope, fx)


@pytest.mark.parametrize(
    "coverage_mutation",
    [
        {"coverage_ratio": 0.999},
        {"complete_count": 5501},
        {"expected_scope_count": 0},
    ],
)
def test_future_source_envelope_rejects_incomplete_coverage(
    coverage_mutation: dict[str, Any]
) -> None:
    fx = _fixture()
    envelope = copy.deepcopy(fx["envelope"])
    envelope["coverage"].update(coverage_mutation)
    envelope = _reseal(envelope)

    with pytest.raises(ERR):
        _validate_envelope(envelope, fx)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("analysis_start", "2026-02-30"),
        ("cutoff", "2026-99-99"),
        ("snapshot_date", "2026-02-30"),
        ("snapshot_id", "20260230T172132Z"),
    ],
)
def test_future_source_envelope_rejects_impossible_dates_and_timestamps(
    field: str, value: str
) -> None:
    fx = _fixture()
    envelope = copy.deepcopy(fx["envelope"])
    envelope[field] = value
    envelope = _reseal(envelope)

    with pytest.raises(ERR, match="real ISO|real UTC"):
        _validate_envelope(envelope, fx)


@pytest.mark.parametrize(
    "cycle_id",
    [
        "cn_full_a_v4_2_20260720",
        "arbitrary-cycle-id",
    ],
)
def test_future_source_envelope_rejects_truncated_or_arbitrary_cycle_id(
    cycle_id: str,
) -> None:
    fx = _fixture()
    envelope = copy.deepcopy(fx["envelope"])
    envelope["cycle_id"] = cycle_id
    envelope = _reseal(envelope)

    with pytest.raises(ERR, match="cycle_id must exactly bind"):
        _validate_envelope(envelope, fx)


def test_definition_identity_naming_does_not_claim_structural_dedup() -> None:
    fx = _fixture()
    audit = _validate_collision(fx["collision"], fx)

    assert audit["schema_version"] == (
        subject.DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
    )
    assert audit["method"] == "definition_identity_equality_only.v1"
    assert audit["structural_dedup"] == "not_run"
    assert audit["formal_dedup"] == "not_run"
    assert audit["high_correlation_dedup"] == "not_run"
    assert all(
        "definition_identity_sha256" in row
        and "structural_fingerprint_sha256" not in row
        for row in fx["selection"]["candidates"]
    )
    assert "definition_identity_inventory" in fx["comparison"]
    assert "fingerprint_inventory" not in fx["comparison"]


def test_label_independence_remains_unproven_and_outcomes_are_not_evidence() -> None:
    fx = _fixture()
    assert fx["aquant"]["outcome_paths_read"] == []
    assert fx["aquant"]["outcomes_used_as_evidence"] is False

    selection = copy.deepcopy(fx["selection"])
    selection["claims"]["selection_uninfluenced_by_any_external_label"] = True
    selection = _reseal(selection)
    with pytest.raises(ERR, match="must be UNPROVEN"):
        _validate_selection(selection, fx)


def test_precommitted_source_envelope_binds_predecessor_source_chain() -> None:
    fx = _fixture()

    assert fx["predecessor"]["state"] == cycle_state.PRECOMMITTED
    assert fx["predecessor"]["source_chain_node_sha256"] == (
        fx["envelope"]["artifact_semantic_sha256"]
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda fx: fx.__setitem__(
            "predecessor",
            cycle_state.build_next_cycle_state_v4_1(
                predecessor=fx["predecessor"],
                predecessor_byte_sha256=fx["predecessor_byte"],
                expected_predecessor_byte_sha256=fx["predecessor_byte"],
                expected_predecessor_semantic_sha256=fx["predecessor"][
                    "state_semantic_sha256"
                ],
                cycle_id=fx["predecessor"]["cycle_id"],
                cycle_root_sha256=fx["predecessor"]["cycle_root_sha256"],
                next_state=cycle_state.DISCOVERY,
                source_chain_node_sha256=_sha("already-discovery-source"),
            ),
        ),
        lambda fx: fx["predecessor"].__setitem__(
            "source_chain_node_sha256", _sha("wrong-source-node")
        ),
    ],
)
def test_source_node_rejects_wrong_predecessor_state_or_source_node_mismatch(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    fx = _fixture()
    mutate(fx)

    with pytest.raises((ERR, cycle_state.FactorGovernanceCycleStateV4_1Error)):
        subject.build_prereg_discovery_source_node_v4_2(
            predecessor_state=fx["predecessor"],
            future_source_envelope=fx["envelope"],
            selection_spec=fx["selection"],
            aquant_receipt=fx["aquant"],
            myquant_receipt=fx["myquant"],
            operator_semantics=fx["operators"],
            comparison_catalog_receipt=fx["comparison"],
            definition_identity_collision_audit=fx["collision"],
            code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
            strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
            full_a_scope_sha256=FULL_A_SCOPE_SHA,
            full_a_scope_count=FULL_A_SCOPE_COUNT,
            serving_inventory_count=SERVING_INVENTORY_COUNT,
        )


@pytest.mark.parametrize("field", ["cycle_id", "cycle_root_sha256"])
def test_source_node_rejects_cross_cycle_or_root_substitution(field: str) -> None:
    fx = _fixture()
    source_node = copy.deepcopy(fx["source_node"])
    source_node[field] = "other-cycle" if field == "cycle_id" else _sha("other-root")
    source_node = _reseal(source_node)

    with pytest.raises(ERR, match=field):
        subject.validate_prereg_discovery_source_node_v4_2(
            source_node,
            predecessor_state=fx["predecessor"],
            future_source_envelope=fx["envelope"],
            selection_spec=fx["selection"],
            aquant_receipt=fx["aquant"],
            myquant_receipt=fx["myquant"],
            operator_semantics=fx["operators"],
            comparison_catalog_receipt=fx["comparison"],
            definition_identity_collision_audit=fx["collision"],
            code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
            strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
            full_a_scope_sha256=FULL_A_SCOPE_SHA,
            full_a_scope_count=FULL_A_SCOPE_COUNT,
            serving_inventory_count=SERVING_INVENTORY_COUNT,
        )


def test_source_node_rejects_predecessor_byte_binding_tamper() -> None:
    fx = _fixture()
    source_node = copy.deepcopy(fx["source_node"])
    source_node["predecessor_state_byte_sha256"] = _sha("forged-byte")
    source_node = _reseal(source_node)

    with pytest.raises(ERR, match="predecessor_state_byte_sha256 mismatch"):
        subject.validate_prereg_discovery_source_node_v4_2(
            source_node,
            predecessor_state=fx["predecessor"],
            future_source_envelope=fx["envelope"],
            selection_spec=fx["selection"],
            aquant_receipt=fx["aquant"],
            myquant_receipt=fx["myquant"],
            operator_semantics=fx["operators"],
            comparison_catalog_receipt=fx["comparison"],
            definition_identity_collision_audit=fx["collision"],
            code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
            strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
            full_a_scope_sha256=FULL_A_SCOPE_SHA,
            full_a_scope_count=FULL_A_SCOPE_COUNT,
            serving_inventory_count=SERVING_INVENTORY_COUNT,
        )


@pytest.mark.parametrize(
    ("byte_arg", "expected_byte_arg", "semantic_arg"),
    [
        ("stale", "valid", "valid"),
        ("valid", "stale", "valid"),
        ("valid", "valid", "stale"),
    ],
)
def test_discovery_cycle_rejects_stale_byte_and_semantic_cas(
    byte_arg: str, expected_byte_arg: str, semantic_arg: str
) -> None:
    fx = _fixture()
    predecessor_byte = fx["predecessor_byte"]
    predecessor_semantic = fx["predecessor"]["state_semantic_sha256"]

    with pytest.raises(ERR):
        subject.build_preregistration_discovery_cycle_v4_2(
            predecessor_state=fx["predecessor"],
            predecessor_byte_sha256=(
                predecessor_byte if byte_arg == "valid" else _sha("stale-byte")
            ),
            expected_predecessor_byte_sha256=(
                predecessor_byte
                if expected_byte_arg == "valid"
                else _sha("stale-expected-byte")
            ),
            expected_predecessor_semantic_sha256=(
                predecessor_semantic
                if semantic_arg == "valid"
                else _sha("stale-semantic")
            ),
            future_source_envelope=fx["envelope"],
            selection_spec=fx["selection"],
            aquant_receipt=fx["aquant"],
            myquant_receipt=fx["myquant"],
            operator_semantics=fx["operators"],
            comparison_catalog_receipt=fx["comparison"],
            definition_identity_collision_audit=fx["collision"],
            code_binding_set_semantic_sha256=CODE_BINDING_SET_SHA,
            strict_source_binding_semantic_sha256=STRICT_SOURCE_BINDING_SHA,
            full_a_scope_sha256=FULL_A_SCOPE_SHA,
            full_a_scope_count=FULL_A_SCOPE_COUNT,
            serving_inventory_count=SERVING_INVENTORY_COUNT,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda orchestration: orchestration.__setitem__(
            "precommitted_state_role", "CURRENT"
        ),
        lambda orchestration: orchestration.__setitem__(
            "external_state_pointer_mutation", True
        ),
        lambda orchestration: orchestration["source_node"].__setitem__(
            "predecessor_state_byte_sha256", _sha("nested-forged-byte")
        ),
        lambda orchestration: orchestration["discovery_state"].__setitem__(
            "source_chain_node_sha256", _sha("nested-forged-source")
        ),
    ],
)
def test_full_orchestration_validator_rebuilds_graph_and_rejects_tamper(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    fx = _fixture()
    assert _validate_orchestration(fx["orchestration"], fx) == fx["orchestration"]

    orchestration = copy.deepcopy(fx["orchestration"])
    mutate(orchestration)
    orchestration = _reseal(orchestration)
    with pytest.raises((ERR, cycle_state.FactorGovernanceCycleStateV4_1Error)):
        _validate_orchestration(orchestration, fx)


@pytest.mark.parametrize(
    ("artifact_key", "validator"),
    [
        ("selection", _validate_selection),
        ("collision", _validate_collision),
    ],
)
def test_exact_schema_rejects_missing_unknown_and_self_hash_tamper(
    artifact_key: str,
    validator: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> None:
    fx = _fixture()

    missing = copy.deepcopy(fx[artifact_key])
    missing.pop("schema_version")
    with pytest.raises(ERR, match="missing"):
        validator(missing, fx)

    unknown = copy.deepcopy(fx[artifact_key])
    unknown["registry_sha256"] = _sha("forbidden-registry")
    with pytest.raises(ERR, match="unknown|banned"):
        validator(unknown, fx)

    tampered_hash = copy.deepcopy(fx[artifact_key])
    tampered_hash["artifact_semantic_sha256"] = _sha("wrong-self-hash")
    with pytest.raises(ERR, match="artifact_semantic_sha256 mismatch"):
        validator(tampered_hash, fx)


def test_pure_builds_are_deterministic_across_repeated_invocations() -> None:
    first = _fixture()["orchestration"]
    second = _fixture()["orchestration"]

    assert first == second
    assert subject.canonical_file_bytes_v4_2(first) == (
        subject.canonical_file_bytes_v4_2(second)
    )
