from __future__ import annotations

import copy
import hashlib
import subprocess
from typing import Any, Callable

import pytest

from quant_investor.factors import governance_candidate_preregistration_v4_3 as subject
from quant_investor.factors import governance_cycle_state_v4_1 as cycle_state


ERR = subject.FactorGovernanceCandidatePreregistrationV4_3Error
PREREGISTERED_AT = "2026-07-19T12:00:00+08:00"
CYCLE_ID = "cn_full_a_v4_3_20260717_20260717T172132Z"
FULL_A_SCOPE_COUNT = 5502
SERVING_INVENTORY_COUNT = 5728


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["artifact_semantic_sha256"] = subject.semantic_sha256_v4_3(
        {
            key: value
            for key, value in result.items()
            if key != "artifact_semantic_sha256"
        }
    )
    return result


@pytest.fixture(scope="module")
def aquant_git_objects() -> dict[str, bytes]:
    return {
        row["git_tree_path"]: subprocess.check_output(
            [
                "git",
                "-C",
                subject.AQUANT_GIT_TOP,
                "cat-file",
                "blob",
                row["blob_oid"],
            ]
        )
        for row in subject.SOURCE_BINDINGS
    }


@pytest.fixture(scope="module")
def fx(aquant_git_objects: dict[str, bytes]) -> dict[str, Any]:
    source = subject.build_aquant_source_set_receipt_v4_3(
        aquant_git_objects=aquant_git_objects
    )
    operator = subject.build_operator_semantics_v4_3()
    selection = subject.build_selection_spec_v4_3(
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        preregistered_at=PREREGISTERED_AT,
    )
    comparison = subject.build_comparison_catalog_receipt_v4_3(
        catalog_id="synthetic-base230-v4.1-v4.2",
        catalog_byte_sha256=_sha("comparison-byte"),
        catalog_semantic_sha256=_sha("comparison-semantic"),
        comparison_sources=[
            {
                "name": "base230",
                "byte_sha256": _sha("base230-byte"),
                "semantic_sha256": _sha("base230-semantic"),
                "candidate_count": 230,
            },
            {
                "name": "v4_1",
                "byte_sha256": _sha("v4.1-byte"),
                "semantic_sha256": _sha("v4.1-semantic"),
                "candidate_count": 267,
            },
            {
                "name": "v4_2",
                "byte_sha256": _sha("v4.2-byte"),
                "semantic_sha256": _sha("v4.2-semantic"),
                "candidate_count": 4,
            },
        ],
        definition_identity_inventory=[
            {
                "name": "base_alpha_001",
                "definition_identity_sha256": _sha("base-alpha-001"),
            },
            {
                "name": "base_alpha_002",
                "definition_identity_sha256": _sha("base-alpha-002"),
            },
        ],
    )
    collision = subject.build_definition_identity_collision_audit_v4_3(
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
    )
    strict = {
        "name": "strict_source_binding",
        "byte_sha256": _sha("strict-byte"),
        "semantic_sha256": _sha("strict-semantic"),
    }
    code = {
        "name": "code_binding_set",
        "byte_sha256": _sha("code-byte"),
        "semantic_sha256": _sha("code-semantic"),
    }
    future = subject.build_future_source_envelope_v4_3(
        cycle_id=CYCLE_ID,
        analysis_start="2021-01-01",
        cutoff="2026-07-17",
        snapshot_id="20260717T172132Z",
        snapshot_date="2026-07-17",
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        strict_source_binding=strict,
        code_binding_set=code,
        full_a_scope_sha256=_sha("full-a-scope"),
        full_a_scope_count=FULL_A_SCOPE_COUNT,
        serving_inventory_count=SERVING_INVENTORY_COUNT,
    )
    cycle_root = {
        "name": "cycle_root",
        "byte_sha256": _sha("cycle-root-byte"),
        "semantic_sha256": _sha("cycle-root"),
    }
    predecessor = cycle_state.build_genesis_cycle_state_v4_1(
        cycle_id=CYCLE_ID,
        cycle_root_sha256=cycle_root["semantic_sha256"],
        source_chain_node_sha256=(
            subject.build_precommit_source_chain_sha256_v4_3(future, collision)
        ),
    )
    predecessor_byte = cycle_state.byte_sha256(predecessor)
    graph_kwargs = {
        "predecessor_state": predecessor,
        "predecessor_byte_sha256": predecessor_byte,
        "expected_predecessor_byte_sha256": predecessor_byte,
        "expected_predecessor_semantic_sha256": predecessor[
            "state_semantic_sha256"
        ],
        "future_source_envelope": future,
        "selection_spec": selection,
        "aquant_source_set_receipt": source,
        "operator_semantics": operator,
        "comparison_catalog_receipt": comparison,
        "definition_identity_collision_audit": collision,
        "cycle_root_binding": cycle_root,
        "strict_source_binding": strict,
        "code_binding_set": code,
        "full_a_scope_sha256": _sha("full-a-scope"),
        "full_a_scope_count": FULL_A_SCOPE_COUNT,
        "serving_inventory_count": SERVING_INVENTORY_COUNT,
    }
    orchestration = subject.build_preregistration_discovery_cycle_v4_3(
        **graph_kwargs
    )
    return {
        "source": source,
        "operator": operator,
        "selection": selection,
        "comparison": comparison,
        "collision": collision,
        "strict": strict,
        "code": code,
        "future": future,
        "cycle_root": cycle_root,
        "predecessor": predecessor,
        "graph_kwargs": graph_kwargs,
        "orchestration": orchestration,
    }


def test_exact_source_oracle_and_runtime_fingerprint() -> None:
    assert subject.AQUANT_COMMIT == (
        "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
    )
    assert subject.AQUANT_GIT_TOP == "/Users/maxwell/mySpace"
    assert len(subject.SOURCE_BINDINGS) == 8
    assert [row["order"] for row in subject.SOURCE_BINDINGS] == list(range(1, 9))
    assert {row["mode"] for row in subject.SOURCE_BINDINGS} == {"100644"}
    assert subject.runtime_fingerprint_v4_3() == subject.AST_RUNTIME_FINGERPRINT
    assert subject.AST_RUNTIME_FINGERPRINT["version_info"] == [3, 13, 7]
    assert subject.AST_RUNTIME_FINGERPRINT["ast_parse"] == {
        "mode": "exec",
        "type_comments": True,
        "feature_version": [3, 13],
        "optimize": -1,
    }
    assert subject.AST_RUNTIME_FINGERPRINT["ast_dump"] == {
        "annotate_fields": True,
        "include_attributes": False,
        "indent": None,
        "show_empty": True,
    }


def test_exact_order_aliases_and_definition_identities(fx: dict[str, Any]) -> None:
    selection = fx["selection"]
    assert tuple(row["name"] for row in selection["candidates"]) == (
        subject.EXPECTED_CANDIDATES
    )
    assert [row["order"] for row in selection["candidates"]] == [1, 2, 3, 4, 5, 6]
    assert [
        row["definition_identity_sha256"] for row in selection["candidates"]
    ] == [
        "bac5f223258ace64408c7f85ebf40f23191f1186c5777ba5d457df798077912f",
        "48301f911a56f3130afdcf893948c39739a8992ba10fe2063ccace1cb8587eb5",
        "11286ce714db5384b3f259637c6d762c070550f95816c513a5f6479890fce18f",
        "e2a7263a0972443363a37fbb49a60e7bc0567deec285db083a4a3a899b6b3ca0",
        "01efa6aa12dff950a5c1ab21b00f4231c3f0b6402c375e1cc85b1480ba855f31",
        "2f03c8d5bf6f7dc1f4508b81dc04c17cc7d5996820e6af561a35bb0f6d39fb44",
    ]
    aliases = [row["source_alias"] for row in selection["candidates"]]
    assert "_DEFAULT_LOOKBACK_DAYS=90" in aliases[0]
    assert aliases[1] == "EarningsEventDrift config 60"
    assert aliases[2] == "RoeDelta + ReportType.ANNUAL"
    assert "cs_rank(-float_market_cap)" in aliases[3]
    assert "cs_rank(book_to_price)" in aliases[4]
    assert aliases[5] == "IndustryRelativeMomentum config 20 calendar days"


def test_success_is_only_prospective_and_measurement_not_run(fx: dict[str, Any]) -> None:
    selection = fx["selection"]
    assert selection["readiness_status"] == "PROSPECTIVE_PREREGISTRATION_ONLY"
    assert selection["measurement_status"] == "measurement_not_run"
    assert all(row["initial_weight"] == 0 for row in selection["candidates"])
    assert all(row["report_only"] is True for row in selection["candidates"])
    assert all(value is False for value in subject.AUTHORITY_FLAGS.values())
    assert all(value is False for value in subject.SIDE_EFFECT_FLAGS.values())
    for field, value in subject.MEASUREMENT_FLAGS.items():
        if field == "readiness":
            assert value == "PROSPECTIVE_PREREGISTRATION_ONLY"
        elif field == "status":
            assert value == "measurement_not_run"
        else:
            assert value == "not_run"
    assert fx["source"]["source_authenticity_verified"] is True
    assert fx["source"]["definition_identity_verified"] is True
    assert fx["source"]["runtime_equivalence_verified"] is False
    assert fx["source"]["signal_computability_proven"] is False


def test_time_policy_is_exact_and_publication_date_is_excluded(fx: dict[str, Any]) -> None:
    assert fx["selection"]["publication"] == {
        "preregistered_at": PREREGISTERED_AT,
        "publication_date": "2026-07-19",
        "timezone": "Asia/Shanghai",
        "publication_time_authority": "LOCAL_UNVERIFIED",
        "measurement_anchor_status": (
            "PENDING_INDEPENDENT_POST_PUBLICATION_EVIDENCE"
        ),
    }
    assert subject.TIME_POLICY["selection_independence"] == "UNPROVEN"
    assert subject.TIME_POLICY["measurement_authorized"] is False
    assert subject.TIME_POLICY["publication_date_in_measurement_sample"] is False
    assert subject.TIME_POLICY["first_eligible_session"] == (
        "STRICTLY_LATER_THAN_PUBLICATION_DATE"
    )
    assert subject.TIME_POLICY["embargo_open_sessions"] == 30
    assert subject.TIME_POLICY["measurement_sample_begins_at_eligible_session"] == 31
    assert subject.TIME_POLICY["minimum_post_embargo_open_sessions_policy"] == 240
    assert subject.TIME_POLICY["minimum_distinct_month_ends_policy"] == 12
    assert subject.TIME_POLICY["horizon_policy_only"] is True


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-19T12:00:00Z",
        "2026-07-19T12:00:00+07:00",
        "2026-07-19T12:00:00.123456+08:00",
        "not-a-time",
    ],
)
def test_selection_rejects_noncanonical_publication_time(
    fx: dict[str, Any], timestamp: str
) -> None:
    with pytest.raises(ERR):
        subject.build_selection_spec_v4_3(
            aquant_source_set_receipt=fx["source"],
            operator_semantics=fx["operator"],
            preregistered_at=timestamp,
        )


def test_fixed_blockers_are_exact_and_immutable(fx: dict[str, Any]) -> None:
    assert fx["selection"]["blockers"] == list(subject.BLOCKERS)
    assert [row["detail"] for row in subject.BLOCKERS] == [
        "guidance missing original p_change bounds",
        "book-to-price only 1/pb proxy without PIT equivalence",
        "ROE semantic/report_type unproved",
        "earnings availability-date equivalence unproved",
        "industry historical PIT generations insufficient",
    ]
    tampered = copy.deepcopy(fx["selection"])
    tampered["blockers"][0]["detail"] = "resolved"
    with pytest.raises(ERR, match="blocker"):
        subject.validate_selection_spec_v4_3(
            _reseal(tampered),
            aquant_source_set_receipt=fx["source"],
            operator_semantics=fx["operator"],
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.reverse(),
        lambda rows: rows.pop(),
        lambda rows: rows.append(copy.deepcopy(rows[-1])),
        lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
        lambda rows: rows[0].__setitem__("name", "unknown_factor"),
        lambda rows: rows[0].__setitem__("initial_weight", 1),
        lambda rows: rows[0].__setitem__("report_only", False),
    ],
)
def test_selection_rejects_reorder_omit_extra_duplicate_and_unknown(
    fx: dict[str, Any], mutate: Callable[[list[dict[str, Any]]], None]
) -> None:
    selection = copy.deepcopy(fx["selection"])
    mutate(selection["candidates"])
    with pytest.raises(ERR):
        subject.validate_selection_spec_v4_3(
            _reseal(selection),
            aquant_source_set_receipt=fx["source"],
            operator_semantics=fx["operator"],
        )


@pytest.mark.parametrize("field", ["status", "weight", "approval", "results", "outcome"])
def test_candidate_rejects_inherited_aquant_fields(
    fx: dict[str, Any], field: str
) -> None:
    selection = copy.deepcopy(fx["selection"])
    selection["candidates"][0][field] = "inherited"
    with pytest.raises(ERR):
        subject.validate_selection_spec_v4_3(
            _reseal(selection),
            aquant_source_set_receipt=fx["source"],
            operator_semantics=fx["operator"],
        )


def test_source_blob_tamper_fails_before_ast_identity(
    aquant_git_objects: dict[str, bytes]
) -> None:
    tampered = dict(aquant_git_objects)
    path = subject.SOURCE_BINDINGS[0]["git_tree_path"]
    tampered[path] = tampered[path] + b"\n"
    with pytest.raises(ERR, match="raw SHA-256 mismatch"):
        subject.build_aquant_source_set_receipt_v4_3(
            aquant_git_objects=tampered
        )


def test_runtime_fingerprint_tamper_fails(
    aquant_git_objects: dict[str, bytes]
) -> None:
    runtime = subject.runtime_fingerprint_v4_3()
    runtime["version_info"] = [3, 13, 8]
    with pytest.raises(ERR, match="runtime fingerprint mismatch"):
        subject.build_aquant_source_set_receipt_v4_3(
            aquant_git_objects=aquant_git_objects,
            runtime_fingerprint=runtime,
        )


def test_resealed_selector_and_source_binding_tamper_fails(fx: dict[str, Any]) -> None:
    selector = copy.deepcopy(fx["source"])
    selector["selector_bindings"][0]["canonical_sha256"] = _sha("forged")
    with pytest.raises(ERR, match="selector bindings"):
        subject.validate_aquant_source_set_receipt_v4_3(_reseal(selector))

    source = copy.deepcopy(fx["source"])
    source["source_bindings"][0]["blob_oid"] = "0" * 40
    with pytest.raises(ERR, match="source bindings"):
        subject.validate_aquant_source_set_receipt_v4_3(_reseal(source))


@pytest.mark.parametrize("mode", ["zero", "multiple"])
def test_unique_float_alias_selector_rejects_zero_or_multiple_matches(
    aquant_git_objects: dict[str, bytes], mode: str
) -> None:
    path = "A_quant/scripts/run_factor_batch_screen.py"
    source = aquant_git_objects[path].decode("utf-8")
    target = (
        '    add("alpha_float_size_small", "cs_rank(-float_market_cap)", '
        '"risk_liquidity", "Size effect using float market value.")'
    )
    assert source.count(target) == 1
    replacement = target.replace("alpha_float_size_small", "other")
    if mode == "multiple":
        replacement = target + "\n" + target
    module = subject._parse_ast(source.replace(target, replacement).encode(), path)
    with pytest.raises(ERR, match="match exactly once"):
        subject._ast_node_for_selector(module, "float_candidate_call")


def test_yaml_config_selectors_are_exact_and_reject_duplicates(
    aquant_git_objects: dict[str, bytes]
) -> None:
    path = "A_quant/configs/factors.yaml"
    source = aquant_git_objects[path]
    assert subject._yaml_factor_selection(source, "earnings_event_drift") == {
        "enabled": True,
        "max_event_age_days": 60,
    }
    assert subject._yaml_factor_selection(source, "roe_delta") == {"enabled": True}
    assert subject._yaml_factor_selection(source, "industry_relative_momentum") == {
        "enabled": True,
        "lookback_days": 20,
    }
    block = (
        "  earnings_event_drift:\n"
        "    enabled: true\n"
        "    max_event_age_days: 60  # only recent reports\n"
    )
    duplicate_source = source.decode().replace(block, block + block)
    if duplicate_source == source.decode():
        duplicate_source = source.decode().replace(
            "  earnings_event_drift:\n", "  earnings_event_drift:\n  earnings_event_drift:\n"
        )
    with pytest.raises(ERR, match="match exactly once"):
        subject._yaml_factor_selection(
            duplicate_source.encode(), "earnings_event_drift"
        )


def test_yaml_config_value_tamper_fails_intent(
    aquant_git_objects: dict[str, bytes]
) -> None:
    source = aquant_git_objects["A_quant/configs/factors.yaml"].replace(
        b"max_event_age_days: 60", b"max_event_age_days: 61"
    )
    selected = subject._yaml_factor_selection(source, "earnings_event_drift")
    with pytest.raises(ERR, match="config mismatch"):
        subject._validate_selector_intent("config_earnings", selected)


@pytest.mark.parametrize(
    ("artifact_key", "validator"),
    [
        ("source", lambda value, fx: subject.validate_aquant_source_set_receipt_v4_3(value)),
        (
            "operator",
            lambda value, fx: subject.validate_operator_semantics_v4_3(value),
        ),
        (
            "selection",
            lambda value, fx: subject.validate_selection_spec_v4_3(
                value,
                aquant_source_set_receipt=fx["source"],
                operator_semantics=fx["operator"],
            ),
        ),
    ],
)
def test_exact_schema_rejects_missing_unknown_and_self_hash_tamper(
    fx: dict[str, Any],
    artifact_key: str,
    validator: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> None:
    missing = copy.deepcopy(fx[artifact_key])
    missing.pop("schema_version")
    with pytest.raises(ERR, match="missing"):
        validator(missing, fx)

    unknown = copy.deepcopy(fx[artifact_key])
    unknown["approval"] = True
    with pytest.raises(ERR, match="unknown"):
        validator(unknown, fx)

    bad_hash = copy.deepcopy(fx[artifact_key])
    bad_hash["artifact_semantic_sha256"] = _sha("wrong")
    with pytest.raises(ERR, match="artifact_semantic_sha256 mismatch"):
        validator(bad_hash, fx)


def test_comparison_collision_is_recomputed_not_trusted(fx: dict[str, Any]) -> None:
    first = fx["selection"]["candidates"][0]
    comparison = subject.build_comparison_catalog_receipt_v4_3(
        catalog_id="collision",
        catalog_byte_sha256=_sha("collision-byte"),
        catalog_semantic_sha256=_sha("collision-semantic"),
        comparison_sources=fx["comparison"]["comparison_sources"],
        definition_identity_inventory=[
            {
                "name": first["name"],
                "definition_identity_sha256": _sha("different-definition"),
            }
        ],
    )
    with pytest.raises(ERR, match="collide"):
        subject.build_definition_identity_collision_audit_v4_3(
            selection_spec=fx["selection"],
            aquant_source_set_receipt=fx["source"],
            operator_semantics=fx["operator"],
            comparison_catalog_receipt=comparison,
        )


def test_future_envelope_accepts_distinct_scope_and_serving_counts(
    fx: dict[str, Any]
) -> None:
    assert fx["future"]["full_a_scope_count"] == 5502
    assert fx["future"]["serving_inventory_count"] == 5728
    assert [row["name"] for row in fx["future"]["predecessor_bindings"]] == [
        "selection_spec",
        "strict_source_binding",
        "code_binding_set",
    ]
    assert all(
        set(row) == {"name", "byte_sha256", "semantic_sha256"}
        for row in fx["future"]["predecessor_bindings"]
    )


def test_future_envelope_rejects_publication_and_time_policy_tamper(
    fx: dict[str, Any]
) -> None:
    future = copy.deepcopy(fx["future"])
    future["time_policy"]["embargo_open_sessions"] = 29
    with pytest.raises(ERR, match="time policy"):
        subject.validate_future_source_envelope_v4_3(
            _reseal(future),
            selection_spec=fx["selection"],
            aquant_source_set_receipt=fx["source"],
            operator_semantics=fx["operator"],
            strict_source_binding=fx["strict"],
            code_binding_set=fx["code"],
            full_a_scope_sha256=_sha("full-a-scope"),
            full_a_scope_count=FULL_A_SCOPE_COUNT,
            serving_inventory_count=SERVING_INVENTORY_COUNT,
        )


def test_discovery_graph_has_dual_sha_edges_and_no_external_mutation(
    fx: dict[str, Any]
) -> None:
    orchestration = fx["orchestration"]
    assert orchestration["discovery_state"]["state"] == cycle_state.DISCOVERY
    assert orchestration["persisted_state_sequence"] == [
        cycle_state.PRECOMMITTED,
        cycle_state.DISCOVERY,
    ]
    assert orchestration["external_state_pointer_mutation"] is False
    assert len(orchestration["graph_bindings"]) == 12
    assert all(
        set(row) == {"name", "byte_sha256", "semantic_sha256"}
        for row in orchestration["graph_bindings"]
    )
    assert [
        row["name"]
        for row in orchestration["source_node"]["predecessor_bindings"]
    ] == ["precommitted_state", "selection_spec", "aquant_source_set_receipt"]


def test_orchestration_rebuild_rejects_nested_binding_tamper(
    fx: dict[str, Any]
) -> None:
    orchestration = copy.deepcopy(fx["orchestration"])
    orchestration["graph_bindings"][0]["byte_sha256"] = _sha("forged")
    with pytest.raises(ERR, match="graph mismatch"):
        subject.validate_preregistration_discovery_cycle_v4_3(
            _reseal(orchestration), **fx["graph_kwargs"]
        )


def test_pure_builds_are_deterministic(aquant_git_objects: dict[str, bytes]) -> None:
    first = subject.build_aquant_source_set_receipt_v4_3(
        aquant_git_objects=aquant_git_objects
    )
    second = subject.build_aquant_source_set_receipt_v4_3(
        aquant_git_objects=aquant_git_objects
    )
    assert first == second
    assert subject.canonical_file_bytes_v4_3(first) == (
        subject.canonical_file_bytes_v4_3(second)
    )
