"""Contract, replay, chronology, and numerical tests for I2 Industry Intelligence."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from quant_investor.intelligence_v2.industry import (
    IndustryContractError,
    build_industry_component_policy,
    build_industry_component_receipt,
    build_industry_evidence,
    build_industry_identity_policy,
    build_industry_membership_catalog,
    build_industry_taxonomy,
    evaluate_industry_identity,
    validate_industry_component_policy,
    validate_industry_component_receipt,
    validate_industry_evaluation_receipt,
    validate_industry_evidence,
    validate_industry_identity_policy,
    validate_industry_membership_catalog,
    validate_industry_taxonomy,
)

AS_OF = "2025-01-02T00:00:00Z"
EARLIER = "2025-01-01T00:00:00Z"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _exact_ref(
    name: str,
    *,
    available_at: str = EARLIER,
    cutoff: str = EARLIER,
    sha: str = SHA_A,
) -> dict[str, str]:
    return {
        "artifact_id": f"{name}-artifact",
        "artifact_version": f"{name}.v1",
        "available_at": available_at,
        "byte_sha256": sha,
        "cutoff": cutoff,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": sha,
    }


def _policy(taxonomy_precedence: list[str] | None = None) -> dict:
    return build_industry_identity_policy(
        created_at=EARLIER,
        provider_precedence=["PROVIDER_A", "PROVIDER_B"],
        taxonomy_precedence=taxonomy_precedence or ["TAXONOMY_A"],
        cap_taxonomy_level="LEVEL_1",
    )


def _taxonomy(
    taxonomy_id: str = "TAXONOMY_A",
    *,
    source_name: str = "taxonomy",
    sha: str = SHA_A,
) -> dict:
    return build_industry_taxonomy(
        taxonomy_id=taxonomy_id,
        rows=[
            {
                "aliases": [],
                "available_at": EARLIER,
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": None,
                "industry_id": "ROOT",
                "level": 0,
                "name": "All Industries",
                "parent_id": None,
                "status": "ACTIVE",
            },
            {
                "aliases": ["ALPHA_ALIAS"],
                "available_at": EARLIER,
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": None,
                "industry_id": "INDUSTRY_ALPHA",
                "level": 1,
                "name": "Alpha",
                "parent_id": "ROOT",
                "status": "ACTIVE",
            },
            {
                "aliases": ["BETA_ALIAS"],
                "available_at": EARLIER,
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": None,
                "industry_id": "INDUSTRY_BETA",
                "level": 1,
                "name": "Beta",
                "parent_id": "ROOT",
                "status": "ACTIVE",
            },
            {
                "aliases": ["GAMMA_ALIAS"],
                "available_at": EARLIER,
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": None,
                "industry_id": "INDUSTRY_GAMMA",
                "level": 1,
                "name": "Gamma",
                "parent_id": "ROOT",
                "status": "ACTIVE",
            },
        ],
        source_ref=_exact_ref(source_name, sha=sha),
        as_of=EARLIER,
    )


def _membership(industry_id: str, exposure: str = "1") -> dict:
    return {
        "available_at": EARLIER,
        "effective_from": "2020-01-01T00:00:00Z",
        "effective_to": None,
        "exposure": exposure,
        "industry_id": industry_id,
        "listing_identity": "LISTING_1",
        "subject_id": "SUBJECT_1",
    }


def _catalog(
    *,
    provider: str = "PROVIDER_A",
    memberships: list[dict] | None = None,
    source_name: str = "membership-a",
    sha: str = SHA_A,
    taxonomy_id: str = "TAXONOMY_A",
) -> dict:
    return build_industry_membership_catalog(
        provider_id=provider,
        taxonomy_id=taxonomy_id,
        memberships=memberships or [_membership("INDUSTRY_ALPHA")],
        source_ref=_exact_ref(source_name, sha=sha),
        cutoff=EARLIER,
        created_at=EARLIER,
    )


def _evaluation(catalogs: list[dict] | None = None) -> tuple[dict, dict, list[dict], dict]:
    policy = _policy()
    taxonomy = _taxonomy()
    rows = catalogs or [_catalog()]
    evaluation = evaluate_industry_identity(
        policy=policy,
        taxonomies=[taxonomy],
        catalogs=rows,
        subject_id="SUBJECT_1",
        listing_identity="LISTING_1",
        as_of=AS_OF,
    )
    return policy, taxonomy, rows, evaluation


def _component_policy() -> dict:
    weights = [
        "0.166666666667",
        "0.166666666667",
        "0.166666666667",
        "0.166666666667",
        "0.166666666667",
        "0.166666666665",
    ]
    dimensions = []
    for index, dimension in enumerate(
        ("CAPEX", "DEMAND", "EARNINGS_REVISION", "INVENTORY", "PRICING_POWER", "SUPPLY")
    ):
        dimensions.append(
            {
                "dimension": dimension,
                "dimension_weight": weights[index],
                "metrics": [
                    {
                        "direction": "HIGHER_IS_BETTER",
                        "metric_id": f"METRIC_{dimension}",
                        "weight": "1",
                    }
                ],
                "minimum_metric_coverage": "1",
                "missing_rule": "BLOCK_COMPONENT",
                "winsor_lower": "0.25",
                "winsor_upper": "0.75",
            }
        )
    return build_industry_component_policy(dimensions=dimensions, created_at=EARLIER)


def _evidence(taxonomy: dict) -> list[dict]:
    rows = []
    for dimension in (
        "CAPEX",
        "DEMAND",
        "EARNINGS_REVISION",
        "INVENTORY",
        "PRICING_POWER",
        "SUPPLY",
    ):
        rows.append(
            build_industry_evidence(
                taxonomy=taxonomy,
                metric_id=f"METRIC_{dimension}",
                dimension=dimension,
                direction="HIGHER_IS_BETTER",
                observations=[
                    {
                        "available_at": EARLIER,
                        "industry_id": "INDUSTRY_ALPHA",
                        "source_refs": [_exact_ref(f"{dimension.lower()}-a")],
                        "value": "10",
                    },
                    {
                        "available_at": EARLIER,
                        "industry_id": "INDUSTRY_BETA",
                        "source_refs": [_exact_ref(f"{dimension.lower()}-b", sha=SHA_B)],
                        "value": "20",
                    },
                    {
                        "available_at": EARLIER,
                        "industry_id": "INDUSTRY_GAMMA",
                        "source_refs": [_exact_ref(f"{dimension.lower()}-c", sha=SHA_C)],
                        "value": "30",
                    },
                ],
                cutoff=EARLIER,
                created_at=EARLIER,
            )
        )
    return rows


def test_all_i2_builders_are_content_addressed_and_replay_exactly() -> None:
    policy, taxonomy, catalogs, evaluation = _evaluation()
    component_policy = _component_policy()
    evidence = _evidence(taxonomy)
    component = build_industry_component_receipt(
        identity_evaluation=evaluation,
        identity_policy=policy,
        taxonomies=[taxonomy],
        catalogs=catalogs,
        component_policy=component_policy,
        evidence=evidence,
        as_of=AS_OF,
    )

    assert validate_industry_identity_policy(policy) == policy
    assert validate_industry_taxonomy(taxonomy) == taxonomy
    assert validate_industry_membership_catalog(catalogs[0]) == catalogs[0]
    assert (
        validate_industry_evaluation_receipt(
            evaluation, policy=policy, taxonomies=[taxonomy], catalogs=catalogs
        )
        == evaluation
    )
    assert validate_industry_component_policy(component_policy) == component_policy
    assert all(validate_industry_evidence(row, taxonomy=taxonomy) == row for row in evidence)
    assert (
        validate_industry_component_receipt(
            component,
            identity_evaluation=evaluation,
            identity_policy=policy,
            taxonomies=[taxonomy],
            catalogs=catalogs,
            component_policy=component_policy,
            evidence=evidence,
        )
        == component
    )
    for artifact in [
        policy,
        taxonomy,
        catalogs[0],
        evaluation,
        component_policy,
        *evidence,
        component,
    ]:
        assert artifact["research_only"] is True
        assert artifact["production"] is False
        assert artifact["authority"]["llm"] is False
        assert artifact["authority"]["provider"] is False
        assert artifact["authority"]["portfolio"] is False


def test_provider_precedence_and_weighted_primary_are_deterministic() -> None:
    lower_priority = _catalog(
        provider="PROVIDER_B",
        memberships=[_membership("INDUSTRY_BETA")],
        source_name="membership-b",
        sha=SHA_B,
    )
    weighted = _catalog(
        memberships=[
            _membership("INDUSTRY_ALPHA", "0.400000000000"),
            _membership("INDUSTRY_BETA", "0.600000000000"),
        ]
    )
    policy, taxonomy, catalogs, evaluation = _evaluation([lower_priority, weighted])

    assert evaluation["state"] == "AVAILABLE"
    assert evaluation["primary_industry_id"] == "INDUSTRY_BETA"
    assert evaluation["exposures"] == [
        {"exposure": "0.400000000000", "industry_id": "INDUSTRY_ALPHA"},
        {"exposure": "0.600000000000", "industry_id": "INDUSTRY_BETA"},
    ]
    assert validate_industry_evaluation_receipt(
        evaluation, policy=policy, taxonomies=[taxonomy], catalogs=catalogs
    )


def test_taxonomy_precedence_and_fallback_use_the_exact_taxonomy_closure() -> None:
    policy = _policy(["TAXONOMY_A", "TAXONOMY_B"])
    taxonomy_a = _taxonomy()
    taxonomy_b = _taxonomy("TAXONOMY_B", source_name="taxonomy-b", sha=SHA_B)
    catalog_a = _catalog()
    catalog_b = _catalog(
        memberships=[_membership("INDUSTRY_BETA")],
        source_name="membership-taxonomy-b",
        sha=SHA_B,
        taxonomy_id="TAXONOMY_B",
    )

    preferred = evaluate_industry_identity(
        policy=policy,
        taxonomies=[taxonomy_b, taxonomy_a],
        catalogs=[catalog_b, catalog_a],
        subject_id="SUBJECT_1",
        listing_identity="LISTING_1",
        as_of=AS_OF,
    )

    assert preferred["primary_industry_id"] == "INDUSTRY_ALPHA"
    assert preferred["taxonomy_ref"]["artifact_id"] == taxonomy_a["taxonomy_receipt_id"]
    assert len(preferred["taxonomy_refs"]) == 2
    assert preferred == evaluate_industry_identity(
        policy=policy,
        taxonomies=[taxonomy_a, taxonomy_b],
        catalogs=[catalog_a, catalog_b],
        subject_id="SUBJECT_1",
        listing_identity="LISTING_1",
        as_of=AS_OF,
    )

    catalog_a_unmapped = _catalog(
        memberships=[{**_membership("INDUSTRY_ALPHA"), "subject_id": "OTHER"}],
        source_name="membership-taxonomy-a-unmapped",
    )
    fallback = evaluate_industry_identity(
        policy=policy,
        taxonomies=[taxonomy_a, taxonomy_b],
        catalogs=[catalog_a_unmapped, catalog_b],
        subject_id="SUBJECT_1",
        listing_identity="LISTING_1",
        as_of=AS_OF,
    )

    assert fallback["primary_industry_id"] == "INDUSTRY_BETA"
    assert fallback["taxonomy_ref"]["artifact_id"] == taxonomy_b["taxonomy_receipt_id"]
    with pytest.raises(IndustryContractError, match="taxonomy closure"):
        validate_industry_evaluation_receipt(
            fallback,
            policy=policy,
            taxonomies=[taxonomy_b],
            catalogs=[catalog_a_unmapped, catalog_b],
        )

    component = build_industry_component_receipt(
        identity_evaluation=fallback,
        identity_policy=policy,
        taxonomies=[taxonomy_a, taxonomy_b],
        catalogs=[catalog_a_unmapped, catalog_b],
        component_policy=_component_policy(),
        evidence=_evidence(taxonomy_b),
        as_of=AS_OF,
    )
    assert component["taxonomy_ref"]["artifact_id"] == taxonomy_b["taxonomy_receipt_id"]
    assert (
        validate_industry_component_receipt(
            component,
            identity_evaluation=fallback,
            identity_policy=policy,
            taxonomies=[taxonomy_b, taxonomy_a],
            catalogs=[catalog_b, catalog_a_unmapped],
            component_policy=_component_policy(),
            evidence=_evidence(taxonomy_b),
        )
        == component
    )


def test_same_precedence_different_classification_is_ambiguous() -> None:
    first = _catalog()
    second = _catalog(
        memberships=[_membership("INDUSTRY_BETA")],
        source_name="membership-conflict",
        sha=SHA_B,
    )
    _policy_row, _taxonomy_row, _catalogs, evaluation = _evaluation([first, second])

    assert evaluation["state"] == "AMBIGUOUS"
    assert evaluation["primary_industry_id"] is None
    assert evaluation["exposures"] == []
    assert evaluation["reason_codes"] == ["SAME_PRECEDENCE_CLASSIFICATION_CONFLICT"]

    with pytest.raises(IndustryContractError, match="closure is duplicated"):
        evaluate_industry_identity(
            policy=_policy(),
            taxonomies=[_taxonomy()],
            catalogs=[first, first],
            subject_id="SUBJECT_1",
            listing_identity="LISTING_1",
            as_of=AS_OF,
        )


def test_no_admissible_membership_is_unmapped() -> None:
    policy = _policy()
    taxonomy = _taxonomy()
    catalog = _catalog(memberships=[{**_membership("INDUSTRY_ALPHA"), "subject_id": "OTHER"}])

    evaluation = evaluate_industry_identity(
        policy=policy,
        taxonomies=[taxonomy],
        catalogs=[catalog],
        subject_id="SUBJECT_1",
        listing_identity="LISTING_1",
        as_of=AS_OF,
    )

    assert evaluation["state"] == "UNMAPPED"
    assert evaluation["reason_codes"] == ["NO_ADMISSIBLE_MEMBERSHIP"]


def test_taxonomy_allows_nonoverlapping_retirement_chronology() -> None:
    taxonomy = _taxonomy()
    rows = [row for row in taxonomy["rows"] if row["industry_id"] != "INDUSTRY_ALPHA"]
    rows.extend(
        [
            {
                "aliases": ["OLD_ALPHA"],
                "available_at": "2023-12-31T00:00:00Z",
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": "2023-12-31T00:00:00Z",
                "industry_id": "INDUSTRY_ALPHA",
                "level": 1,
                "name": "Old Alpha",
                "parent_id": "ROOT",
                "status": "RETIRED",
            },
            {
                "aliases": ["NEW_ALPHA"],
                "available_at": EARLIER,
                "effective_from": "2024-01-01T00:00:00Z",
                "effective_to": None,
                "industry_id": "INDUSTRY_ALPHA",
                "level": 1,
                "name": "New Alpha",
                "parent_id": "ROOT",
                "status": "ACTIVE",
            },
        ]
    )

    rebuilt = build_industry_taxonomy(
        taxonomy_id="TAXONOMY_A",
        rows=rows,
        source_ref=_exact_ref("taxonomy"),
        as_of=EARLIER,
    )

    assert validate_industry_taxonomy(rebuilt) == rebuilt


def test_membership_catalog_allows_nonoverlapping_historical_bundles() -> None:
    historical = _membership("INDUSTRY_ALPHA")
    historical["effective_to"] = "2022-12-31T00:00:00Z"
    current = _membership("INDUSTRY_BETA")
    current["effective_from"] = "2023-01-01T00:00:00Z"

    catalog = _catalog(memberships=[current, historical])

    assert validate_industry_membership_catalog(catalog) == catalog
    policy, taxonomy, _catalogs, evaluation = _evaluation([catalog])
    assert evaluation["primary_industry_id"] == "INDUSTRY_BETA"
    assert validate_industry_evaluation_receipt(
        evaluation, policy=policy, taxonomies=[taxonomy], catalogs=[catalog]
    )


def test_membership_catalog_rejects_overlapping_historical_bundles() -> None:
    historical = _membership("INDUSTRY_ALPHA")
    historical["effective_to"] = "2023-06-30T00:00:00Z"
    overlapping = _membership("INDUSTRY_BETA")
    overlapping["effective_from"] = "2023-01-01T00:00:00Z"

    with pytest.raises(IndustryContractError, match="overlapping chronology"):
        _catalog(memberships=[historical, overlapping])


def test_overlapping_taxonomy_chronology_and_future_membership_fail_closed() -> None:
    taxonomy = _taxonomy()
    overlap = deepcopy(taxonomy["rows"])
    duplicate = deepcopy(next(row for row in overlap if row["industry_id"] == "INDUSTRY_ALPHA"))
    duplicate["effective_from"] = "2024-01-01T00:00:00Z"
    overlap.append(duplicate)

    with pytest.raises(IndustryContractError, match="overlaps"):
        build_industry_taxonomy(
            taxonomy_id="TAXONOMY_A",
            rows=overlap,
            source_ref=_exact_ref("taxonomy"),
            as_of=EARLIER,
        )

    future = _membership("INDUSTRY_ALPHA")
    future["available_at"] = "2025-01-03T00:00:00Z"
    with pytest.raises(IndustryContractError, match="future"):
        _catalog(memberships=[future])


def test_industry_evidence_rejects_retired_and_future_effective_peers() -> None:
    rows = list(_taxonomy()["rows"])
    rows.extend(
        [
            {
                "aliases": [],
                "available_at": EARLIER,
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": "2024-12-31T00:00:00Z",
                "industry_id": "INDUSTRY_RETIRED",
                "level": 1,
                "name": "Retired",
                "parent_id": "ROOT",
                "status": "RETIRED",
            },
            {
                "aliases": [],
                "available_at": EARLIER,
                "effective_from": "2025-02-01T00:00:00Z",
                "effective_to": None,
                "industry_id": "INDUSTRY_FUTURE",
                "level": 1,
                "name": "Future Effective",
                "parent_id": "ROOT",
                "status": "ACTIVE",
            },
        ]
    )
    taxonomy = build_industry_taxonomy(
        taxonomy_id="TAXONOMY_A",
        rows=rows,
        source_ref=_exact_ref("taxonomy-with-inadmissible-peers"),
        as_of=EARLIER,
    )

    for industry_id in ("INDUSTRY_RETIRED", "INDUSTRY_FUTURE"):
        with pytest.raises(IndustryContractError, match="identities are invalid"):
            build_industry_evidence(
                taxonomy=taxonomy,
                metric_id="METRIC_DEMAND",
                dimension="DEMAND",
                direction="HIGHER_IS_BETTER",
                observations=[
                    {
                        "available_at": EARLIER,
                        "industry_id": industry_id,
                        "source_refs": [_exact_ref(f"peer-{industry_id.lower()}")],
                        "value": "1",
                    }
                ],
                cutoff=EARLIER,
                created_at=EARLIER,
            )


def test_exposure_must_sum_exactly_to_one_and_binary_float_is_rejected() -> None:
    with pytest.raises(IndustryContractError, match="sum exactly"):
        _catalog(
            memberships=[
                _membership("INDUSTRY_ALPHA", "0.4"),
                _membership("INDUSTRY_BETA", "0.5"),
            ]
        )

    row = _membership("INDUSTRY_ALPHA")
    row["exposure"] = 1.0
    with pytest.raises(IndustryContractError, match="binary float"):
        _catalog(memberships=[row])


def test_type7_winsorized_component_has_no_llm_or_macro_input() -> None:
    policy, taxonomy, catalogs, evaluation = _evaluation()
    component_policy = _component_policy()
    evidence = _evidence(taxonomy)

    component = build_industry_component_receipt(
        identity_evaluation=evaluation,
        identity_policy=policy,
        taxonomies=[taxonomy],
        catalogs=catalogs,
        component_policy=component_policy,
        evidence=evidence,
        as_of=AS_OF,
    )

    assert component["status"] == "AVAILABLE"
    assert component["component_score"] == "0.000000000000"
    assert all(row["status"] == "AVAILABLE" for row in component["dimension_rows"])
    assert all(
        row["metric_rows"][0]["winsorized_value"] == "15.000000000000"
        for row in component["dimension_rows"]
    )
    assert "llm" not in component
    assert "macro" not in component
    assert component["authority"]["llm"] is False


def test_missing_component_stays_missing_without_zero_or_mean_imputation() -> None:
    policy, taxonomy, catalogs, evaluation = _evaluation()
    component_policy = _component_policy()
    evidence = _evidence(taxonomy)[:-1]

    component = build_industry_component_receipt(
        identity_evaluation=evaluation,
        identity_policy=policy,
        taxonomies=[taxonomy],
        catalogs=catalogs,
        component_policy=component_policy,
        evidence=evidence,
        as_of=AS_OF,
    )

    assert component["status"] == "MISSING"
    assert component["component_score"] is None
    missing = next(row for row in component["dimension_rows"] if row["status"] == "MISSING")
    assert missing["score"] is None
    assert missing["missing_metric_ids"] == ["METRIC_SUPPLY"]


def test_resealed_or_replayed_forgery_is_rejected() -> None:
    policy, taxonomy, catalogs, evaluation = _evaluation()
    forged = deepcopy(taxonomy)
    forged["rows"][0]["name"] = "Forged"

    with pytest.raises(IndustryContractError):
        validate_industry_taxonomy(forged)

    wrong_catalog = _catalog(
        memberships=[_membership("INDUSTRY_BETA")],
        source_name="wrong-replay",
        sha=SHA_B,
    )
    with pytest.raises(IndustryContractError, match="replay mismatch"):
        validate_industry_evaluation_receipt(
            evaluation,
            policy=policy,
            taxonomies=[taxonomy],
            catalogs=[wrong_catalog],
        )


def test_component_policy_has_no_implicit_defaults() -> None:
    policy = _component_policy()
    missing_dimension = deepcopy(policy["dimensions"][:-1])

    with pytest.raises(IndustryContractError, match="all six"):
        build_industry_component_policy(dimensions=missing_dimension, created_at=EARLIER)

    invalid = deepcopy(policy["dimensions"])
    invalid[0]["metrics"][0]["weight"] = "0.9"
    with pytest.raises(IndustryContractError, match="sum exactly"):
        build_industry_component_policy(dimensions=invalid, created_at=EARLIER)


def test_input_order_does_not_change_canonical_artifacts() -> None:
    taxonomy = _taxonomy()
    reversed_taxonomy = build_industry_taxonomy(
        taxonomy_id="TAXONOMY_A",
        rows=list(reversed(taxonomy["rows"])),
        source_ref=_exact_ref("taxonomy"),
        as_of=EARLIER,
    )
    first = _catalog(
        memberships=[
            _membership("INDUSTRY_ALPHA", "0.4"),
            _membership("INDUSTRY_BETA", "0.6"),
        ]
    )
    second = _catalog(
        memberships=[
            _membership("INDUSTRY_BETA", "0.6"),
            _membership("INDUSTRY_ALPHA", "0.4"),
        ]
    )

    assert reversed_taxonomy == taxonomy
    assert first == second
    assert Decimal(first["memberships"][0]["exposure"]) in {
        Decimal("0.4"),
        Decimal("0.6"),
    }
