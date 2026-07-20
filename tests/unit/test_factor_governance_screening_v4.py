from __future__ import annotations

import copy
import hashlib
import inspect
import math
from typing import Any

import pytest

from quant_investor.factors import governance_screening_v4 as screening
from quant_investor.factors.governance_screening_v4 import (
    CANDIDATE_CATALOG_SCHEMA_VERSION,
    COMPUTE_FAILED_STATUS,
    EVALUATED_STATUS,
    FDR_METHOD,
    RAW_P_METHOD,
    SCREENING_EVIDENCE_SCHEMA_VERSION,
    SOURCE_BINDING_FIELDS,
    FactorGovernanceScreeningV4Error,
    build_candidate_catalog_v4,
    build_primitive_ontology_v4,
    build_screening_evidence_v4,
    canonical_json_bytes,
    canonical_semantic_sha256,
    validate_candidate_catalog_v4,
    validate_primitive_ontology_v4,
    validate_screening_evidence_v4,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_bindings() -> dict[str, str]:
    return {key: _digest(key) for key in SOURCE_BINDING_FIELDS}


def _statistic_contract() -> dict[str, Any]:
    return {
        "raw_p_method": RAW_P_METHOD,
        "fdr_method": FDR_METHOD,
        "q": 0.1,
    }


def _candidate(
    name: str,
    *,
    primitive_ids: list[str] | None = None,
    input_fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "implementation": "formulaic",
        "expression": "rank(close)",
        "direction": 1,
        "params": {"window": 20, "weights": [0.6, 0.4]},
        "lookback": 20,
        "slot": "alpha_slot",
        "input_fields": input_fields or ["close"],
        "primitive_ids": primitive_ids or ["close_return"],
    }


@pytest.fixture
def ontology() -> dict[str, Any]:
    return build_primitive_ontology_v4(
        [
            {"primitive_id": "volume", "family": "liquidity"},
            {"primitive_id": "roe", "family": "fundamental"},
            {"primitive_id": "close_return_alt", "family": "return"},
            {"primitive_id": "close_return", "family": "return"},
        ]
    )


@pytest.fixture
def catalog(ontology: dict[str, Any]) -> dict[str, Any]:
    return build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            _candidate("z_failed", primitive_ids=["close_return_alt"]),
            _candidate("a_signal"),
            _candidate("m_signal", primitive_ids=["close_return_alt"]),
            _candidate("quality", primitive_ids=["roe"]),
            _candidate(
                "blend",
                primitive_ids=["volume", "close_return", "roe"],
                input_fields=["volume", "close"],
            ),
        ],
    )


def _evaluations() -> list[dict[str, Any]]:
    return [
        {
            "name": "z_failed",
            "evaluation_status": COMPUTE_FAILED_STATUS,
            "raw_p_value": None,
            "failure_reason": "insufficient_observations",
        },
        {
            "name": "quality",
            "evaluation_status": EVALUATED_STATUS,
            "raw_p_value": 1.0,
            "failure_reason": None,
        },
        {
            "name": "a_signal",
            "evaluation_status": EVALUATED_STATUS,
            "raw_p_value": 0.01,
            "failure_reason": None,
        },
        {
            "name": "blend",
            "evaluation_status": EVALUATED_STATUS,
            "raw_p_value": 0.5,
            "failure_reason": None,
        },
        {
            "name": "m_signal",
            "evaluation_status": EVALUATED_STATUS,
            "raw_p_value": 0.04,
            "failure_reason": None,
        },
    ]


@pytest.fixture
def evidence(
    ontology: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, Any]:
    return build_screening_evidence_v4(
        ontology=ontology,
        catalog=catalog,
        evaluations=_evaluations(),
        source_bindings=_source_bindings(),
        statistic_contract=_statistic_contract(),
    )


def _rehash_artifact(value: dict[str, Any]) -> None:
    value["semantic_sha256"] = canonical_semantic_sha256(
        value,
        exclude_fields=("semantic_sha256",),
    )


def _rehash_candidate(value: dict[str, Any]) -> None:
    value["definition_sha256"] = canonical_semantic_sha256(
        value,
        exclude_fields=("definition_sha256",),
    )


def test_canonical_semantic_sha_is_stable_and_rejects_nan() -> None:
    left = {"z": [2, 1], "a": {"x": "中文"}}
    right = {"a": {"x": "中文"}, "z": [2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert not canonical_json_bytes(left).endswith(b"\n")
    assert canonical_semantic_sha256(left) == canonical_semantic_sha256(right)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="canonical JSON"):
        canonical_semantic_sha256({"bad": math.nan})


def test_exact_schema_rejects_non_string_field_names() -> None:
    with pytest.raises(FactorGovernanceScreeningV4Error, match="field names"):
        build_primitive_ontology_v4(
            [{"primitive_id": "p", "family": "p", 1: "not-json"}]
        )


def test_build_validate_exact_v4_artifacts_and_hashes(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    assert ontology["schema_version"] == "factor-primitive-ontology.v4"
    assert catalog["schema_version"] == CANDIDATE_CATALOG_SCHEMA_VERSION
    assert evidence["schema_version"] == SCREENING_EVIDENCE_SCHEMA_VERSION
    assert validate_primitive_ontology_v4(ontology) == ontology
    assert validate_candidate_catalog_v4(catalog, ontology=ontology) == catalog
    assert (
        validate_screening_evidence_v4(
            evidence,
            ontology=ontology,
            catalog=catalog,
        )
        == evidence
    )
    for artifact in (ontology, catalog, evidence):
        assert artifact["semantic_sha256"] == canonical_semantic_sha256(
            artifact,
            exclude_fields=("semantic_sha256",),
        )
    for candidate in catalog["candidates"]:
        assert candidate["definition_sha256"] == canonical_semantic_sha256(
            candidate,
            exclude_fields=("definition_sha256",),
        )


def test_builders_are_pure_and_do_not_import_legacy_governance(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = list(tmp_path.iterdir())
    ontology = build_primitive_ontology_v4(
        [{"primitive_id": "p", "family": "p"}]
    )
    catalog = build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[_candidate("factor", primitive_ids=["p"])],
    )
    build_screening_evidence_v4(
        ontology=ontology,
        catalog=catalog,
        evaluations=[
            {
                "name": "factor",
                "evaluation_status": COMPUTE_FAILED_STATUS,
                "raw_p_value": None,
                "failure_reason": "compute_error",
            }
        ],
        source_bindings=_source_bindings(),
        statistic_contract=_statistic_contract(),
    )

    assert list(tmp_path.iterdir()) == before
    source = inspect.getsource(screening)
    assert "governance_protocol_v3" not in source
    assert "scripts." not in source


def test_ontology_is_sorted_unique_and_hash_bound() -> None:
    ontology = build_primitive_ontology_v4(
        [
            {"primitive_id": "z", "family": "z_family"},
            {"primitive_id": "a", "family": "a_family"},
        ]
    )
    assert [row["primitive_id"] for row in ontology["primitives"]] == ["a", "z"]

    duplicate = [
        {"primitive_id": "a", "family": "one"},
        {"primitive_id": "a", "family": "two"},
    ]
    with pytest.raises(FactorGovernanceScreeningV4Error, match="distinct"):
        build_primitive_ontology_v4(duplicate)

    with pytest.raises(FactorGovernanceScreeningV4Error, match="reserved"):
        build_primitive_ontology_v4(
            [
                {
                    "primitive_id": "p",
                    "family": 'composite:["a","b"]',
                }
            ]
        )

    unsorted = copy.deepcopy(ontology)
    unsorted["primitives"].reverse()
    _rehash_artifact(unsorted)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="sorted"):
        validate_primitive_ontology_v4(unsorted)

    drift = copy.deepcopy(ontology)
    drift["primitives"][0]["family"] = "inflated_family"
    with pytest.raises(FactorGovernanceScreeningV4Error, match="semantic SHA"):
        validate_primitive_ontology_v4(drift)


def test_catalog_sorts_names_and_lists_and_rejects_duplicates(
    ontology: dict[str, Any],
) -> None:
    second = _candidate(
        "second",
        primitive_ids=["roe", "close_return"],
        input_fields=["roe", "close"],
    )
    first = _candidate("first")
    catalog = build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[second, first],
    )
    assert [row["name"] for row in catalog["candidates"]] == ["first", "second"]
    assert catalog["candidates"][1]["input_fields"] == ["close", "roe"]
    assert catalog["candidates"][1]["primitive_ids"] == ["close_return", "roe"]
    assert type(catalog["candidates"][0]["direction"]) is float

    with pytest.raises(FactorGovernanceScreeningV4Error, match="distinct"):
        build_candidate_catalog_v4(
            ontology=ontology,
            candidates=[first, copy.deepcopy(first)],
        )
    duplicate_primitive = _candidate(
        "duplicate_primitive", primitive_ids=["close_return", "close_return"]
    )
    with pytest.raises(FactorGovernanceScreeningV4Error, match="distinct"):
        build_candidate_catalog_v4(
            ontology=ontology,
            candidates=[duplicate_primitive],
        )


def test_catalog_rejects_unknown_primitive_and_caller_derived_fields(
    ontology: dict[str, Any],
) -> None:
    unknown = _candidate("unknown", primitive_ids=["not_registered"])
    with pytest.raises(FactorGovernanceScreeningV4Error, match="unknown primitive"):
        build_candidate_catalog_v4(ontology=ontology, candidates=[unknown])

    inflated = _candidate("inflated")
    inflated["family"] = "fresh_family_per_candidate"
    with pytest.raises(FactorGovernanceScreeningV4Error, match="fields invalid"):
        build_candidate_catalog_v4(ontology=ontology, candidates=[inflated])


def test_ontology_derived_composite_family_is_order_invariant_and_collision_safe(
    ontology: dict[str, Any],
) -> None:
    catalog = build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            _candidate(
                "first",
                primitive_ids=["roe", "volume", "close_return"],
            ),
            _candidate(
                "second",
                primitive_ids=["close_return", "roe", "volume"],
            ),
            _candidate(
                "different",
                primitive_ids=["close_return", "roe"],
            ),
        ],
    )
    by_name = {row["name"]: row for row in catalog["candidates"]}
    expected = 'composite:["fundamental","liquidity","return"]'
    assert by_name["first"]["family"] == expected
    assert by_name["second"]["family"] == expected
    assert by_name["different"]["family"] == 'composite:["fundamental","return"]'
    assert by_name["different"]["family"] != expected
    assert " " not in expected


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("name", "renamed"),
        ("implementation", "other_impl"),
        ("expression", "rank(volume)"),
        ("direction", -1.0),
        ("params", {"window": 21}),
        ("lookback", 21),
        ("slot", "other_slot"),
        ("input_fields", ["amount", "close"]),
        ("primitive_ids", ["close_return_alt"]),
    ],
)
def test_catalog_definition_fields_are_hash_frozen(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    field: str,
    replacement: Any,
) -> None:
    forged = copy.deepcopy(catalog)
    forged["candidates"][0][field] = replacement

    with pytest.raises(FactorGovernanceScreeningV4Error):
        validate_candidate_catalog_v4(forged, ontology=ontology)


def test_catalog_recomputes_family_even_when_attacker_rehashes_everything(
    ontology: dict[str, Any], catalog: dict[str, Any]
) -> None:
    forged = copy.deepcopy(catalog)
    forged["candidates"][0]["family"] = "one_candidate_one_family"
    _rehash_candidate(forged["candidates"][0])
    _rehash_artifact(forged)

    with pytest.raises(FactorGovernanceScreeningV4Error, match="family mismatch"):
        validate_candidate_catalog_v4(forged, ontology=ontology)


def test_catalog_rejects_noncanonical_order_even_after_rehash(
    ontology: dict[str, Any], catalog: dict[str, Any]
) -> None:
    forged = copy.deepcopy(catalog)
    forged["candidates"].reverse()
    _rehash_artifact(forged)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="sorted"):
        validate_candidate_catalog_v4(forged, ontology=ontology)

    unsorted_lineage = copy.deepcopy(catalog)
    blend = next(row for row in unsorted_lineage["candidates"] if row["name"] == "blend")
    blend["primitive_ids"].reverse()
    _rehash_candidate(blend)
    _rehash_artifact(unsorted_lineage)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="sorted"):
        validate_candidate_catalog_v4(unsorted_lineage, ontology=ontology)


def test_screening_keeps_failed_rows_in_full_family_denominator(
    evidence: dict[str, Any],
) -> None:
    rows = {row["name"]: row for row in evidence["rows"]}
    assert rows["a_signal"] == {
        "name": "a_signal",
        "evaluation_status": EVALUATED_STATUS,
        "raw_p_value": 0.01,
        "failure_reason": None,
        "family": "return",
        "bh_input_p_value": 0.01,
        "family_hypothesis_count": 3,
        "bh_rank": 1,
        "bh_q_value": pytest.approx(0.03),
        "bh_pass": True,
    }
    assert rows["m_signal"]["family_hypothesis_count"] == 3
    assert rows["m_signal"]["bh_rank"] == 2
    assert rows["m_signal"]["bh_q_value"] == pytest.approx(0.06)
    assert rows["m_signal"]["bh_pass"] is True
    assert rows["z_failed"]["raw_p_value"] is None
    assert rows["z_failed"]["bh_input_p_value"] == 1.0
    assert rows["z_failed"]["family_hypothesis_count"] == 3
    assert rows["z_failed"]["bh_rank"] == 3
    assert rows["z_failed"]["bh_q_value"] == 1.0
    assert rows["z_failed"]["bh_pass"] is False


def test_real_raw_p_one_is_not_conflated_with_compute_failure(
    evidence: dict[str, Any],
) -> None:
    rows = {row["name"]: row for row in evidence["rows"]}
    real_one = rows["quality"]
    failure = rows["z_failed"]

    assert real_one["evaluation_status"] == EVALUATED_STATUS
    assert real_one["raw_p_value"] == 1.0
    assert real_one["failure_reason"] is None
    assert failure["evaluation_status"] == COMPUTE_FAILED_STATUS
    assert failure["raw_p_value"] is None
    assert failure["failure_reason"] == "insufficient_observations"
    assert real_one["bh_input_p_value"] == failure["bh_input_p_value"] == 1.0


def test_all_failed_family_is_retained_and_ranked_by_name() -> None:
    ontology = build_primitive_ontology_v4(
        [{"primitive_id": "p", "family": "p"}]
    )
    catalog = build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            _candidate("b", primitive_ids=["p"]),
            _candidate("a", primitive_ids=["p"]),
        ],
    )
    evidence = build_screening_evidence_v4(
        ontology=ontology,
        catalog=catalog,
        evaluations=[
            {
                "name": name,
                "evaluation_status": COMPUTE_FAILED_STATUS,
                "raw_p_value": None,
                "failure_reason": "failed",
            }
            for name in ("b", "a")
        ],
        source_bindings=_source_bindings(),
        statistic_contract=_statistic_contract(),
    )

    assert [row["name"] for row in evidence["rows"]] == ["a", "b"]
    assert [row["bh_rank"] for row in evidence["rows"]] == [1, 2]
    assert all(row["family_hypothesis_count"] == 2 for row in evidence["rows"])
    assert all(row["bh_q_value"] == 1.0 for row in evidence["rows"])
    assert not any(row["bh_pass"] for row in evidence["rows"])


def test_bh_ties_are_deterministic_by_candidate_name() -> None:
    ontology = build_primitive_ontology_v4(
        [{"primitive_id": "p", "family": "p"}]
    )
    catalog = build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            _candidate("z", primitive_ids=["p"]),
            _candidate("a", primitive_ids=["p"]),
        ],
    )
    evidence = build_screening_evidence_v4(
        ontology=ontology,
        catalog=catalog,
        evaluations=[
            {
                "name": name,
                "evaluation_status": EVALUATED_STATUS,
                "raw_p_value": 0.04,
                "failure_reason": None,
            }
            for name in ("z", "a")
        ],
        source_bindings=_source_bindings(),
        statistic_contract=_statistic_contract(),
    )

    assert [(row["name"], row["bh_rank"]) for row in evidence["rows"]] == [
        ("a", 1),
        ("z", 2),
    ]
    assert [row["bh_q_value"] for row in evidence["rows"]] == [0.04, 0.04]
    assert all(row["bh_pass"] for row in evidence["rows"])


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate"])
def test_screening_rejects_missing_extra_and_duplicate_catalog_rows(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    evidence: dict[str, Any],
    mode: str,
) -> None:
    forged = copy.deepcopy(evidence)
    if mode == "missing":
        forged["rows"].pop()
    elif mode == "extra":
        extra = copy.deepcopy(forged["rows"][0])
        extra["name"] = "not_in_catalog"
        forged["rows"].append(extra)
    else:
        forged["rows"][-1] = copy.deepcopy(forged["rows"][0])
    _rehash_artifact(forged)

    with pytest.raises(FactorGovernanceScreeningV4Error):
        validate_screening_evidence_v4(
            forged,
            ontology=ontology,
            catalog=catalog,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("family", "inflated"),
        ("bh_input_p_value", 0.0),
        ("family_hypothesis_count", 1),
        ("bh_rank", 3),
        ("bh_q_value", 0.0),
        ("bh_pass", False),
    ],
)
def test_validator_recomputes_and_rejects_caller_bh_or_family_fields(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    evidence: dict[str, Any],
    field: str,
    replacement: Any,
) -> None:
    forged = copy.deepcopy(evidence)
    forged["rows"][0][field] = replacement
    _rehash_artifact(forged)

    with pytest.raises(FactorGovernanceScreeningV4Error, match="recomputation"):
        validate_screening_evidence_v4(
            forged,
            ontology=ontology,
            catalog=catalog,
        )


def test_build_rejects_caller_supplied_bh_fields(
    ontology: dict[str, Any], catalog: dict[str, Any]
) -> None:
    evaluations = _evaluations()
    evaluations[0]["bh_q_value"] = 0.0

    with pytest.raises(FactorGovernanceScreeningV4Error, match="fields invalid"):
        build_screening_evidence_v4(
            ontology=ontology,
            catalog=catalog,
            evaluations=evaluations,
            source_bindings=_source_bindings(),
            statistic_contract=_statistic_contract(),
        )


@pytest.mark.parametrize("bad_p", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_screening_rejects_nonfinite_or_out_of_range_raw_p(
    ontology: dict[str, Any], catalog: dict[str, Any], bad_p: float
) -> None:
    evaluations = _evaluations()
    row = next(item for item in evaluations if item["name"] == "a_signal")
    row["raw_p_value"] = bad_p

    with pytest.raises(FactorGovernanceScreeningV4Error, match="raw_p_value"):
        build_screening_evidence_v4(
            ontology=ontology,
            catalog=catalog,
            evaluations=evaluations,
            source_bindings=_source_bindings(),
            statistic_contract=_statistic_contract(),
        )


@pytest.mark.parametrize(
    "replacement",
    [
        {
            "evaluation_status": EVALUATED_STATUS,
            "raw_p_value": 0.5,
            "failure_reason": "should_be_null",
        },
        {
            "evaluation_status": COMPUTE_FAILED_STATUS,
            "raw_p_value": 1.0,
            "failure_reason": "failed",
        },
        {
            "evaluation_status": COMPUTE_FAILED_STATUS,
            "raw_p_value": None,
            "failure_reason": "",
        },
        {
            "evaluation_status": "not_evaluated",
            "raw_p_value": None,
            "failure_reason": "failed",
        },
    ],
)
def test_screening_status_and_failure_semantics_are_exact(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    replacement: dict[str, Any],
) -> None:
    evaluations = _evaluations()
    row = next(item for item in evaluations if item["name"] == "a_signal")
    row.update(replacement)

    with pytest.raises(FactorGovernanceScreeningV4Error):
        build_screening_evidence_v4(
            ontology=ontology,
            catalog=catalog,
            evaluations=evaluations,
            source_bindings=_source_bindings(),
            statistic_contract=_statistic_contract(),
        )


def test_deleting_failed_candidate_cannot_improve_bh_denominator(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    forged = copy.deepcopy(evidence)
    forged["rows"] = [
        row for row in forged["rows"] if row["name"] != "z_failed"
    ]
    for row in forged["rows"]:
        if row["family"] == "return":
            row["family_hypothesis_count"] = 2
            row["bh_q_value"] = row["raw_p_value"]
            row["bh_pass"] = bool(row["bh_q_value"] <= 0.1)
    _rehash_artifact(forged)

    with pytest.raises(FactorGovernanceScreeningV4Error, match="catalog"):
        validate_screening_evidence_v4(
            forged,
            ontology=ontology,
            catalog=catalog,
        )


def test_screening_rejects_row_order_and_self_hash_drift(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    reordered = copy.deepcopy(evidence)
    reordered["rows"].reverse()
    _rehash_artifact(reordered)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="order"):
        validate_screening_evidence_v4(
            reordered,
            ontology=ontology,
            catalog=catalog,
        )

    hash_drift = copy.deepcopy(evidence)
    hash_drift["semantic_sha256"] = "0" * 64
    with pytest.raises(FactorGovernanceScreeningV4Error, match="semantic SHA"):
        validate_screening_evidence_v4(
            hash_drift,
            ontology=ontology,
            catalog=catalog,
        )


def test_source_bindings_and_statistic_contract_are_exact_and_hash_bound(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    missing_source = _source_bindings()
    missing_source.pop("calendar_sha256")
    with pytest.raises(FactorGovernanceScreeningV4Error, match="fields invalid"):
        build_screening_evidence_v4(
            ontology=ontology,
            catalog=catalog,
            evaluations=_evaluations(),
            source_bindings=missing_source,
            statistic_contract=_statistic_contract(),
        )

    uppercase_source = _source_bindings()
    uppercase_source["code_sha256"] = uppercase_source["code_sha256"].upper()
    with pytest.raises(FactorGovernanceScreeningV4Error, match="lowercase SHA"):
        build_screening_evidence_v4(
            ontology=ontology,
            catalog=catalog,
            evaluations=_evaluations(),
            source_bindings=uppercase_source,
            statistic_contract=_statistic_contract(),
        )

    wrong_method = _statistic_contract()
    wrong_method["fdr_method"] = "benjamini_hochberg_by_family.v3"
    with pytest.raises(FactorGovernanceScreeningV4Error, match="fdr_method"):
        build_screening_evidence_v4(
            ontology=ontology,
            catalog=catalog,
            evaluations=_evaluations(),
            source_bindings=_source_bindings(),
            statistic_contract=wrong_method,
        )

    forged_contract = copy.deepcopy(evidence)
    forged_contract["statistic_contract"]["q"] = 0.2
    _rehash_artifact(forged_contract)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="canonical 0.1"):
        validate_screening_evidence_v4(
            forged_contract,
            ontology=ontology,
            catalog=catalog,
        )


@pytest.mark.parametrize("legacy_version", ["v2", "v3"])
def test_legacy_v2_v3_schemas_are_rejected(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    evidence: dict[str, Any],
    legacy_version: str,
) -> None:
    legacy_ontology = copy.deepcopy(ontology)
    legacy_ontology["schema_version"] = f"factor-primitive-ontology.{legacy_version}"
    _rehash_artifact(legacy_ontology)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="unsupported"):
        validate_primitive_ontology_v4(legacy_ontology)

    legacy_catalog = copy.deepcopy(catalog)
    legacy_catalog["schema_version"] = f"factor-candidate-catalog.{legacy_version}"
    _rehash_artifact(legacy_catalog)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="unsupported"):
        validate_candidate_catalog_v4(legacy_catalog, ontology=ontology)

    legacy_evidence = copy.deepcopy(evidence)
    legacy_evidence["schema_version"] = f"factor-screening-evidence.{legacy_version}"
    _rehash_artifact(legacy_evidence)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="unsupported"):
        validate_screening_evidence_v4(
            legacy_evidence,
            ontology=ontology,
            catalog=catalog,
        )


def test_catalog_and_screening_reject_wrong_bound_artifact_hashes(
    ontology: dict[str, Any],
    catalog: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    catalog_drift = copy.deepcopy(catalog)
    catalog_drift["ontology_sha256"] = _digest("different-ontology")
    _rehash_artifact(catalog_drift)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="ontology SHA"):
        validate_candidate_catalog_v4(catalog_drift, ontology=ontology)

    evidence_drift = copy.deepcopy(evidence)
    evidence_drift["candidate_catalog_sha256"] = _digest("different-catalog")
    _rehash_artifact(evidence_drift)
    with pytest.raises(FactorGovernanceScreeningV4Error, match="catalog SHA"):
        validate_screening_evidence_v4(
            evidence_drift,
            ontology=ontology,
            catalog=catalog,
        )
