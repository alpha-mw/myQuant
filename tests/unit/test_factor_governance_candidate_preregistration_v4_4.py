from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from quant_investor.factors import (
    governance_candidate_preregistration_bundle_v4_2 as v42_bundle,
)
from quant_investor.factors import governance_candidate_preregistration_v4_2 as v42
from quant_investor.factors import (
    governance_candidate_preregistration_bundle_v4_4 as bundle,
)
from quant_investor.factors import governance_candidate_preregistration_v4_4 as subject
from quant_investor.factors import governance_cycle_state_v4_1 as cycle_state
from quant_investor.factors import (
    governance_prior_diagnostic_nomination_bundle_v4_3 as diagnostic_bundle,
)
from tests.unit.test_factor_governance_candidate_preregistration_bundle_v4_4 import (
    _artifacts,
    _code_binding,
    _diagnostic_graph,
    _v42_prefixed_graph,
)


def _unprefixed(
    prefixed: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        filename: prefixed[subject.V4_2_PREDECESSOR_PREFIX + filename]
        for filename in v42_bundle.INPUT_FILENAMES_V4_2
    }


def _selection_kwargs(
    graph: dict[str, dict[str, Any]],
    diagnostic_graph: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "v4_2_selection_spec": graph[
            v42_bundle.CANDIDATE_SELECTION_SPEC_FILENAME_V4_2
        ],
        "v4_2_aquant_receipt": graph[
            v42_bundle.AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2
        ],
        "v4_2_myquant_receipt": graph[
            v42_bundle.MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2
        ],
        "v4_2_operator_semantics": graph[
            v42_bundle.OPERATOR_SEMANTICS_FILENAME_V4_2
        ],
        "v4_2_comparison_catalog_receipt": graph[
            v42_bundle.COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2
        ],
        "prior_diagnostic_nomination": diagnostic_graph[
            diagnostic_bundle.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3
        ],
        "diagnostic_artifact_bindings": [
            subject.build_artifact_binding_v4_4(
                filename=filename, artifact=diagnostic_graph[filename]
            )
            for filename in subject.PRIOR_DIAGNOSTIC_FILENAMES
        ],
    }


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        dict_result = set(value)
        for item in value.values():
            dict_result.update(_keys(item))
        return dict_result
    if isinstance(value, list):
        list_result: set[str] = set()
        for item in value:
            list_result.update(_keys(item))
        return list_result
    return set()


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(value)
    sealed.pop("artifact_semantic_sha256", None)
    sealed["artifact_semantic_sha256"] = subject.semantic_sha256_v4_4(sealed)
    return sealed


def _reseal_state(value: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(value)
    sealed.pop("state_semantic_sha256", None)
    sealed["state_semantic_sha256"] = cycle_state.semantic_sha256(sealed)
    return sealed


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


def test_exact_five_candidate_oracle_is_unique_zero_weight_and_non_authorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    selection = artifacts[bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4]
    assert selection["candidates"] == list(subject.EXPECTED_CANDIDATE_ROWS)
    for field in ("name", "definition_identity_sha256", "family", "slot"):
        assert len({row[field] for row in selection["candidates"]}) == 5
    assert {row["initial_weight"] for row in selection["candidates"]} == {0}
    assert selection["selection_claims"] == {
        "outcome_informed_selection": True,
        "external_label_independence": False,
        "prior_statistics_nomination_only": True,
        "prior_statistics_inherited_as_formal_evidence": False,
        "authoritative_evidence_route": (
            "independent_post_v4_4_publication_embargo_and_holdout_only"
        ),
    }
    assert selection["measurement"] == subject.MEASUREMENT_FLAGS
    assert selection["authority"] == subject.AUTHORITY_FLAGS
    assert selection["side_effects"] == subject.SIDE_EFFECT_FLAGS


def test_new_artifacts_do_not_copy_prior_statistics_or_formal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    new_names = (
        bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4,
        bundle.DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4,
        bundle.CYCLE_ROOT_FILENAME_V4_4,
        bundle.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4,
        bundle.DISCOVERY_SOURCE_NODE_FILENAME_V4_4,
        bundle.PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4,
    )
    forbidden = {
        "rank_ic",
        "raw_mean_ic",
        "adjusted_mean_ic",
        "icir",
        "t_statistic",
        "p_value",
        "bonferroni_p",
        "bh_q_value",
        "coverage",
        "coverage_rate",
    }
    for filename in new_names:
        assert not (_keys(artifacts[filename]) & forbidden)
    future = artifacts[bundle.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4]
    assert future["publication"]["publication_time_authority"] == "LOCAL_UNVERIFIED"
    assert future["publication"]["publication_proven"] is False
    assert future["future_measurement_policy"]["prior_diagnostic_window_inherited"] is False
    assert future["future_measurement_policy"]["measurement_authorized"] is False


def test_publication_and_embargo_policy_resealed_tamper_fails_direct_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    future = artifacts[bundle.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4]
    cases = (
        (("publication", "publication_time_authority"), "SELF_ASSERTED", "publication authority"),
        (("publication", "publication_proven"), True, "publication authority"),
        (
            ("publication", "separate_post_commit_proof_required"),
            False,
            "publication authority",
        ),
        (
            ("future_measurement_policy", "prior_diagnostic_window_inherited"),
            True,
            "future measurement policy",
        ),
        (
            (
                "future_measurement_policy",
                "prior_statistics_inherited_as_formal_evidence",
            ),
            True,
            "future measurement policy",
        ),
        (
            ("future_measurement_policy", "measurement_anchor"),
            "LOCAL_RECORDED_TIME",
            "future measurement policy",
        ),
        (
            ("future_measurement_policy", "publication_session_in_measurement_sample"),
            True,
            "future measurement policy",
        ),
        (("future_measurement_policy", "embargo_open_sessions"), 29, "future measurement policy"),
        (
            ("future_measurement_policy", "first_eligible_measurement_session"),
            30,
            "future measurement policy",
        ),
        (
            ("future_measurement_policy", "minimum_post_embargo_open_sessions"),
            239,
            "future measurement policy",
        ),
        (
            ("future_measurement_policy", "minimum_distinct_closed_month_ends"),
            11,
            "future measurement policy",
        ),
        (
            ("future_measurement_policy", "measurement_authorized"),
            True,
            "future measurement policy",
        ),
    )
    prefixed = {
        filename: artifacts[filename]
        for filename in subject.V4_2_PREDECESSOR_FILENAMES
    }
    for path, replacement, reason in cases:
        changed = copy.deepcopy(future)
        _set_path(changed, path, replacement)
        changed = _reseal(changed)
        with pytest.raises(
            subject.FactorGovernanceCandidatePreregistrationV4_4Error,
            match=reason,
        ):
            subject.validate_future_source_envelope_v4_4(
                changed,
                v4_2_predecessor_artifacts=prefixed,
                expanded_candidate_selection=artifacts[
                    bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4
                ],
                publication_at=future["publication"]["recorded_local_time"],
            )


def test_every_current_artifact_keeps_all_not_run_false_and_side_effect_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    current_names = (
        bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4,
        bundle.DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4,
        bundle.CYCLE_ROOT_FILENAME_V4_4,
        bundle.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4,
        bundle.DISCOVERY_SOURCE_NODE_FILENAME_V4_4,
        bundle.PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4,
    )
    for filename in current_names:
        artifact = artifacts[filename]
        if "measurement" in artifact:
            assert artifact["measurement"] == subject.MEASUREMENT_FLAGS
        if "authority" in artifact:
            assert artifact["authority"] == subject.AUTHORITY_FLAGS
        if "side_effects" in artifact:
            assert artifact["side_effects"] == subject.SIDE_EFFECT_FLAGS

    graph = _unprefixed(
        {
            filename: artifacts[filename]
            for filename in subject.V4_2_PREDECESSOR_FILENAMES
        }
    )
    diagnostic_graph = {
        filename: artifacts[filename]
        for filename in subject.PRIOR_DIAGNOSTIC_FILENAMES
    }
    kwargs = _selection_kwargs(graph, diagnostic_graph)
    selection = artifacts[bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4]
    for section, expected, reason in (
        ("measurement", subject.MEASUREMENT_FLAGS, "measurement flags"),
        ("authority", subject.AUTHORITY_FLAGS, "authority flags"),
        ("side_effects", subject.SIDE_EFFECT_FLAGS, "side effects"),
    ):
        for field, original in expected.items():
            changed = copy.deepcopy(selection)
            changed[section][field] = not original if type(original) is bool else "tampered"
            changed = _reseal(changed)
            with pytest.raises(
                subject.FactorGovernanceCandidatePreregistrationV4_4Error,
                match=reason,
            ):
                subject.validate_expanded_candidate_selection_v4_4(
                    changed,
                    **kwargs,
                )


def test_direct_core_rejects_nonrow_v42_selection_tamper_no_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefixed = _v42_prefixed_graph(tmp_path / "v42")
    graph = _unprefixed(prefixed)
    diagnostic_graph = _diagnostic_graph(monkeypatch)
    changed = copy.deepcopy(
        graph[v42_bundle.CANDIDATE_SELECTION_SPEC_FILENAME_V4_2]
    )
    changed["claims"]["outcomes_used_as_evidence"] = True
    changed.pop("artifact_semantic_sha256")
    changed["artifact_semantic_sha256"] = v42.semantic_sha256_v4_2(changed)
    kwargs = _selection_kwargs(graph, diagnostic_graph)
    kwargs["v4_2_selection_spec"] = changed
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationV4_4Error,
        match="embedded v4.2 selection validation failed",
    ):
        subject.build_expanded_candidate_selection_v4_4(**kwargs)


def test_future_source_consumes_fully_validated_v42_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path / "good", monkeypatch)
    prefixed = {
        filename: copy.deepcopy(artifacts[filename])
        for filename in subject.V4_2_PREDECESSOR_FILENAMES
    }
    future_name = (
        subject.V4_2_PREDECESSOR_PREFIX
        + v42_bundle.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_2
    )
    prefixed[future_name]["code_binding_set_semantic_sha256"] = "0" * 64
    prefixed[future_name].pop("artifact_semantic_sha256")
    prefixed[future_name]["artifact_semantic_sha256"] = v42.semantic_sha256_v4_2(
        prefixed[future_name]
    )
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationV4_4Error,
        match="embedded v4.2 graph validation failed",
    ):
        subject.build_future_source_envelope_v4_4(
            v4_2_predecessor_artifacts=prefixed,
            expanded_candidate_selection=artifacts[
                bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4
            ],
            publication_at="2026-07-20T18:00:00+08:00",
        )


def test_cutoff_must_be_strictly_later_than_2026_07_19(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationV4_4Error,
        match="strictly later than 2026-07-19",
    ):
        _artifacts(tmp_path, monkeypatch, cutoff="2026-07-19")


def _reseal_cycle_root(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("cycle_root_sha256", None)
    value.pop("artifact_semantic_sha256", None)
    value["cycle_root_sha256"] = subject.semantic_sha256_v4_4(value)
    value["artifact_semantic_sha256"] = subject.semantic_sha256_v4_4(value)
    return value


def test_standalone_cycle_root_rejects_stale_cutoff_and_source_summary_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    root = artifacts[bundle.CYCLE_ROOT_FILENAME_V4_4]

    stale = copy.deepcopy(root)
    stale["cutoff"] = "2026-07-19"
    stale["snapshot_id"] = "20260719T172132Z"
    stale["cycle_id"] = subject.deterministic_cycle_id_v4_4(
        cutoff=stale["cutoff"], snapshot_id=stale["snapshot_id"]
    )
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationV4_4Error,
        match="strictly later than 2026-07-19",
    ):
        subject.validate_cycle_root_v4_4(_reseal_cycle_root(stale))

    inconsistent = copy.deepcopy(root)
    inconsistent["source_summary"]["latest_complete_trade_date"] = "2026-07-19"
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationV4_4Error,
        match="source summary root crosslink mismatch",
    ):
        subject.validate_cycle_root_v4_4(_reseal_cycle_root(inconsistent))


def test_collision_audit_checks_names_identities_families_slots_and_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    audit = artifacts[bundle.DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4]
    assert {
        name: row["unique_count"] for name, row in audit["candidate_uniqueness"].items()
    } == {
        "names": 5,
        "definition_identities": 5,
        "families": 5,
        "slots": 5,
    }
    assert audit["selected_vs_comparison"]["collisions"] == []
    assert audit["declared_slot_collision"] is False
    assert audit["structural_dedup"] == "not_run"
    assert audit["formal_dedup"] == "not_run"
    assert audit["high_correlation_dedup"] == "not_run"
    for audit_name, field in (
        ("names", "name"),
        ("definition_identities", "definition_identity_sha256"),
        ("families", "family"),
        ("slots", "slot"),
    ):
        assert audit["candidate_uniqueness"][audit_name]["values"] == [
            row[field] for row in subject.EXPECTED_CANDIDATE_ROWS
        ]


@pytest.mark.parametrize("collision_kind", ("name", "identity"))
def test_comparison_catalog_name_or_identity_collision_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    graph = _unprefixed(_v42_prefixed_graph(tmp_path / collision_kind))
    diagnostic_graph = _diagnostic_graph(monkeypatch)
    selected = subject.EXPECTED_CANDIDATE_ROWS[0]
    comparison = v42.build_comparison_catalog_receipt_v4_2(
        catalog_id=f"synthetic-{collision_kind}-collision",
        catalog_byte_sha256="1" * 64,
        catalog_semantic_sha256="2" * 64,
        primitive_count=1,
        definition_identity_inventory=[
            {
                "name": selected["name"] if collision_kind == "name" else "legacy_other",
                "definition_identity_sha256": (
                    "3" * 64
                    if collision_kind == "name"
                    else selected["definition_identity_sha256"]
                ),
            }
        ],
    )
    graph[v42_bundle.COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2] = comparison
    graph[v42_bundle.CANDIDATE_SELECTION_SPEC_FILENAME_V4_2] = (
        v42.build_selection_spec_v4_2(
            aquant_receipt=graph[
                v42_bundle.AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2
            ],
            myquant_receipt=graph[
                v42_bundle.MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2
            ],
            operator_semantics=graph[
                v42_bundle.OPERATOR_SEMANTICS_FILENAME_V4_2
            ],
            comparison_catalog_receipt=comparison,
        )
    )
    kwargs = _selection_kwargs(graph, diagnostic_graph)
    expanded = subject.build_expanded_candidate_selection_v4_4(**kwargs)
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationV4_4Error,
        match="selected candidates collide with comparison catalog",
    ):
        subject.build_definition_identity_collision_audit_v4_4(
            expanded_candidate_selection=expanded,
            v4_2_selection_spec=kwargs["v4_2_selection_spec"],
            v4_2_aquant_receipt=kwargs["v4_2_aquant_receipt"],
            v4_2_myquant_receipt=kwargs["v4_2_myquant_receipt"],
            v4_2_operator_semantics=kwargs["v4_2_operator_semantics"],
            prior_diagnostic_nomination=kwargs["prior_diagnostic_nomination"],
            diagnostic_artifact_bindings=kwargs["diagnostic_artifact_bindings"],
            comparison_catalog_receipt=comparison,
        )


def test_v43_diagnostic_bindings_are_exact_and_descriptor_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    selection = artifacts[bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4]
    root = artifacts[bundle.CYCLE_ROOT_FILENAME_V4_4]
    expected = list(subject.EXPECTED_PRIOR_DIAGNOSTIC_BINDINGS)
    assert selection["prior_diagnostic_provenance"]["artifact_bindings"] == expected
    assert root["prior_diagnostic_evidence"]["artifact_bindings"] == expected

    prefixed = {
        filename: artifacts[filename]
        for filename in subject.V4_2_PREDECESSOR_FILENAMES
    }
    graph = _unprefixed(prefixed)
    diagnostic_graph = {
        filename: artifacts[filename]
        for filename in subject.PRIOR_DIAGNOSTIC_FILENAMES
    }
    kwargs = _selection_kwargs(graph, diagnostic_graph)
    binding_cases: list[list[dict[str, Any]]] = []
    for field, replacement in (
        ("byte_sha256", "0" * 64),
        ("semantic_sha256", "0" * 64),
        ("size_bytes", expected[0]["size_bytes"] + 1),
    ):
        changed = copy.deepcopy(expected)
        changed[0][field] = replacement
        binding_cases.append(changed)
    binding_cases.append(list(reversed(copy.deepcopy(expected))))
    for bindings in binding_cases:
        changed_kwargs = dict(kwargs)
        changed_kwargs["diagnostic_artifact_bindings"] = bindings
        with pytest.raises(
            subject.FactorGovernanceCandidatePreregistrationV4_4Error,
            match="diagnostic artifact bindings|filename/order mismatch",
        ):
            subject.build_expanded_candidate_selection_v4_4(**changed_kwargs)


def test_current_artifact_schemas_and_self_hashes_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    expected_schemas = {
        bundle.CODE_BINDING_SET_FILENAME_V4_4: subject.CODE_BINDING_SET_SCHEMA_VERSION,
        bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4: (
            subject.EXPANDED_SELECTION_SCHEMA_VERSION
        ),
        bundle.DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4: (
            subject.DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
        ),
        bundle.CYCLE_ROOT_FILENAME_V4_4: subject.CYCLE_ROOT_SCHEMA_VERSION,
        bundle.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4: (
            subject.SOURCE_ENVELOPE_SCHEMA_VERSION
        ),
        bundle.PRECOMMITTED_STATE_FILENAME_V4_4: cycle_state.STATE_SCHEMA_VERSION,
        bundle.DISCOVERY_SOURCE_NODE_FILENAME_V4_4: (
            subject.DISCOVERY_SOURCE_NODE_SCHEMA_VERSION
        ),
        bundle.DISCOVERY_STATE_FILENAME_V4_4: cycle_state.STATE_SCHEMA_VERSION,
        bundle.PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4: (
            subject.ORCHESTRATION_SCHEMA_VERSION
        ),
    }
    for filename, schema in expected_schemas.items():
        artifact = artifacts[filename]
        assert artifact["schema_version"] == schema
        assert artifact["protocol_version"] == subject.PROTOCOL_VERSION
        if "artifact_semantic_sha256" in artifact:
            expected_self_hash = subject.semantic_sha256_v4_4(
                {
                    key: value
                    for key, value in artifact.items()
                    if key != "artifact_semantic_sha256"
                }
            )
            assert artifact["artifact_semantic_sha256"] == expected_self_hash
        else:
            expected_state_hash = cycle_state.semantic_sha256(
                {
                    key: value
                    for key, value in artifact.items()
                    if key != "state_semantic_sha256"
                }
            )
            assert artifact["state_semantic_sha256"] == expected_state_hash

    root = artifacts[bundle.CYCLE_ROOT_FILENAME_V4_4]
    assert root["cycle_root_sha256"] == subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in root.items()
            if key not in {"cycle_root_sha256", "artifact_semantic_sha256"}
        }
    )


def test_v44_state_chain_is_independent_genesis_to_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    precommit = artifacts[bundle.PRECOMMITTED_STATE_FILENAME_V4_4]
    discovery = artifacts[bundle.DISCOVERY_STATE_FILENAME_V4_4]
    old_discovery = artifacts[
        subject.V4_2_PREDECESSOR_PREFIX
        + v42_bundle.DISCOVERY_STATE_FILENAME_V4_2
    ]
    assert precommit["state"] == "PRECOMMITTED"
    assert precommit["predecessor"]["kind"] == "genesis"
    assert discovery["state"] == "DISCOVERY"
    assert discovery["predecessor"]["semantic_sha256"] == precommit[
        "state_semantic_sha256"
    ]
    assert discovery["predecessor"]["semantic_sha256"] != old_discovery[
        "state_semantic_sha256"
    ]
    root = artifacts[bundle.CYCLE_ROOT_FILENAME_V4_4]
    assert root["embedded_v4_2_evidence_graph"]["cross_cycle_state_edge"] is False
    assert root["lineage_contract"]["v4_2_cycle_state_used_as_predecessor"] is False


def test_code_binding_inventory_and_self_hash_tamper_fail_closed() -> None:
    code = _code_binding()
    assert [row["relative_path"] for row in code["ordered_bindings"]] == list(
        subject.CODE_BINDING_PATHS_V4_4
    )
    changed = copy.deepcopy(code)
    changed["ordered_bindings"][0]["relative_path"] = "wrong.py"
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationV4_4Error,
        match="code binding path/order mismatch",
    ):
        subject.validate_code_binding_set_v4_4(changed)
