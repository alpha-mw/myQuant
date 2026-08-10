"""I4 Fundamental policy, PIT profile, replay, and frozen-scorer boundaries."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any

import pytest

from quant_investor.intelligence_v2._core import (
    IntelligenceV2ContractError,
    common_fields,
    seal,
)
from quant_investor.intelligence_v2.fundamental import (
    FundamentalContractError,
    build_fundamental_component_policy,
    build_fundamental_profile,
    validate_fundamental_component_policy,
    validate_fundamental_profile,
)
from quant_investor.intelligence_v2.fundamental.models import (
    COMPONENT_POLICY_VERSION,
    FUNDAMENTAL_SCORER_IMPLEMENTATION_SHA256_V3,
    INDUSTRY_COMPONENT_VERSION,
    PROFILE_VERSION,
    THEME_COMPONENT_VERSION,
)
from quant_investor.intelligence_v2.industry import (
    build_industry_component_policy,
    build_industry_component_receipt,
    build_industry_evidence,
    build_industry_identity_policy,
    build_industry_membership_catalog,
    build_industry_taxonomy,
    evaluate_industry_identity,
)
import quant_investor.intelligence_v2.fundamental.profile as profile_module

AS_OF = "2025-01-02T00:00:00Z"
EARLIER = "2025-01-01T00:00:00Z"
FUTURE = "2025-01-03T00:00:00Z"
SYMBOLS = ("000001.SZ", "000002.SZ", "000003.SZ")
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


def _component(
    component: str,
    metric_id: str,
    *,
    direction: str = "HIGHER_IS_BETTER",
    missing_rule: str = "BLOCK_COMPONENT",
) -> dict[str, Any]:
    return {
        "component": component,
        "implementation_sha256": SHA_B,
        "metric_rows": [{"metric_id": metric_id, "direction": direction, "weight": "1"}],
        "minimum_coverage": "1",
        "missing_rule": missing_rule,
        "percentile_method": "TYPE_7_AVERAGE_TIE",
        "source_cutoff": EARLIER,
        "winsor_lower": "0.25",
        "winsor_upper": "0.75",
    }


def _policy(*, components: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = components or [
        _component("industry_cycle", "I2_INDUSTRY_COMPONENT_SCORE"),
        _component("earnings_revision", "EARNINGS_REVISION_3M"),
        _component("theme_narrative", "I3_THEME_COMPONENT_SCORE"),
        _component("valuation", "FCF_YIELD"),
        _component("governance", "GOVERNANCE_QUALITY"),
    ]
    return build_fundamental_component_policy(
        components=rows,
        owner_policy_ref=_exact_ref("owner-policy"),
        created_at=EARLIER,
    )


def _metric_row(
    symbol: str,
    metric_id: str,
    value: Any,
    *,
    sha: str = SHA_A,
) -> dict[str, Any]:
    return {
        "available_at": EARLIER,
        "company_code": symbol,
        "metric_id": metric_id,
        "source_ref": _exact_ref(f"{symbol}-{metric_id}", sha=sha),
        "value": value,
    }


def _financial_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = {
        "roe": ("0.10", "0.20", "0.30"),
        "ocf_to_profit": ("0.70", "0.80", "0.90"),
        "debt_to_assets": ("0.50", "0.30", "0.10"),
    }
    for metric_id, metric_values in values.items():
        for symbol, value in zip(SYMBOLS, metric_values, strict=True):
            rows.append(_metric_row(symbol, metric_id, value))
    return rows


def _owner_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inputs = {
        "earnings_revision": ("EARNINGS_REVISION_3M", ("0", "10", "100")),
        "valuation": ("FCF_YIELD", ("0.05", "0.10", "0.15")),
        "governance": ("GOVERNANCE_QUALITY", ("0.20", "0.50", "0.80")),
    }
    for component, (metric_id, values) in inputs.items():
        for symbol, value in zip(SYMBOLS, values, strict=True):
            rows.append({"component": component, **_metric_row(symbol, metric_id, value)})
    return rows


def _closure(**overrides: Any) -> dict[str, Any]:
    result = {
        "company_code": SYMBOLS[0],
        "symbols": SYMBOLS,
        "policy": _policy(),
        "financial_metric_rows": _financial_rows(),
        "component_metric_rows": _owner_rows(),
        "industry_component_receipts": {},
        "industry_component_validation_closures": {},
        "theme_component_receipts": {},
        "theme_component_validation_closures": {},
        "scorer_implementation_sha256": FUNDAMENTAL_SCORER_IMPLEMENTATION_SHA256_V3,
        "as_of": AS_OF,
    }
    result.update(overrides)
    return result


def _reseal(document: dict[str, Any], identity_field: str) -> dict[str, Any]:
    body = deepcopy(document)
    body.pop(identity_field)
    body.pop("semantic_sha256")
    return seal(body, identity_field=identity_field)


def _contains_float(value: Any) -> bool:
    if type(value) is float:
        return True
    if type(value) is list:
        return any(_contains_float(item) for item in value)
    if type(value) is dict:
        return any(_contains_float(item) for item in value.values())
    return False


def _component_receipt(*, version: str, score: str, timestamp: str = EARLIER) -> dict:
    return seal(
        {
            **common_fields(timestamp_value=timestamp),
            "component_score": score,
            "status": "AVAILABLE",
            "version": version,
        },
        identity_field="component_receipt_id",
    )


def _industry_closure(symbol: str) -> dict[str, Any]:
    return {
        "catalogs": [],
        "component_policy": {},
        "evidence": [],
        "identity_evaluation": {"subject_id": symbol},
        "identity_policy": {},
        "taxonomies": [],
    }


def _theme_closure(symbol: str) -> dict[str, Any]:
    return {
        "as_of": EARLIER,
        "component_policy": {},
        "exposure_closure": {},
        "exposure_receipt": {"company_code": symbol},
        "metric_rows": [],
    }


def _real_i2_receipt_closure(symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_policy = build_industry_identity_policy(
        created_at=EARLIER,
        provider_precedence=["PROVIDER_A"],
        taxonomy_precedence=["TAXONOMY_A"],
        cap_taxonomy_level="LEVEL_1",
    )
    taxonomy = build_industry_taxonomy(
        taxonomy_id="TAXONOMY_A",
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
                "aliases": [],
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
                "aliases": [],
                "available_at": EARLIER,
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": None,
                "industry_id": "INDUSTRY_BETA",
                "level": 1,
                "name": "Beta",
                "parent_id": "ROOT",
                "status": "ACTIVE",
            },
        ],
        source_ref=_exact_ref("i2-taxonomy"),
        as_of=EARLIER,
    )
    catalog = build_industry_membership_catalog(
        provider_id="PROVIDER_A",
        taxonomy_id="TAXONOMY_A",
        memberships=[
            {
                "available_at": EARLIER,
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": None,
                "exposure": "1",
                "industry_id": "INDUSTRY_ALPHA",
                "listing_identity": "LISTING_1",
                "subject_id": symbol,
            }
        ],
        source_ref=_exact_ref("i2-membership"),
        cutoff=EARLIER,
        created_at=EARLIER,
    )
    evaluation = evaluate_industry_identity(
        policy=identity_policy,
        taxonomies=[taxonomy],
        catalogs=[catalog],
        subject_id=symbol,
        listing_identity="LISTING_1",
        as_of=AS_OF,
    )
    dimensions = (
        "CAPEX",
        "DEMAND",
        "EARNINGS_REVISION",
        "INVENTORY",
        "PRICING_POWER",
        "SUPPLY",
    )
    weights = ("0.166666666667",) * 5 + ("0.166666666665",)
    component_policy = build_industry_component_policy(
        dimensions=[
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
            for index, dimension in enumerate(dimensions)
        ],
        created_at=EARLIER,
    )
    evidence = [
        build_industry_evidence(
            taxonomy=taxonomy,
            metric_id=f"METRIC_{dimension}",
            dimension=dimension,
            direction="HIGHER_IS_BETTER",
            observations=[
                {
                    "available_at": EARLIER,
                    "industry_id": "INDUSTRY_ALPHA",
                    "source_refs": [_exact_ref(f"i2-{dimension.lower()}-alpha")],
                    "value": "10",
                },
                {
                    "available_at": EARLIER,
                    "industry_id": "INDUSTRY_BETA",
                    "source_refs": [_exact_ref(f"i2-{dimension.lower()}-beta", sha=SHA_B)],
                    "value": "20",
                },
            ],
            cutoff=EARLIER,
            created_at=EARLIER,
        )
        for dimension in dimensions
    ]
    closure = {
        "identity_evaluation": evaluation,
        "identity_policy": identity_policy,
        "taxonomies": [taxonomy],
        "catalogs": [catalog],
        "component_policy": component_policy,
        "evidence": evidence,
    }
    receipt = build_industry_component_receipt(**closure, as_of=AS_OF)
    return receipt, closure


def test_policy_is_owner_sealed_exact_five_sorted_and_replayable() -> None:
    reversed_components = list(reversed(_policy()["components"]))
    policy = _policy(components=reversed_components)

    assert policy["version"] == COMPONENT_POLICY_VERSION
    assert [row["component"] for row in policy["components"]] == [
        "industry_cycle",
        "earnings_revision",
        "theme_narrative",
        "valuation",
        "governance",
    ]
    assert validate_fundamental_component_policy(policy) == policy
    assert policy["authority"]["llm"] is False
    assert policy["authority"]["provider"] is False
    assert policy["authority"]["portfolio"] is False


def test_policy_rejects_float_duplicate_missing_component_and_future_cutoff() -> None:
    floating = deepcopy(_policy()["components"])
    floating[0]["metric_rows"][0]["weight"] = 1.0
    with pytest.raises(Exception, match="binary float"):
        _policy(components=floating)

    duplicate = deepcopy(_policy()["components"])
    duplicate[1]["component"] = "industry_cycle"
    with pytest.raises(FundamentalContractError, match="industry_cycle|exactly all five"):
        _policy(components=duplicate)

    future = deepcopy(_policy()["components"])
    future[0]["source_cutoff"] = AS_OF
    with pytest.raises(FundamentalContractError, match="future-known"):
        _policy(components=future)


def test_policy_tamper_and_resealed_forgery_fail_replay() -> None:
    policy = _policy()
    tampered = deepcopy(policy)
    tampered["components"][1]["metric_rows"][0]["direction"] = "LOWER_IS_BETTER"
    with pytest.raises(Exception, match="mismatch"):
        validate_fundamental_component_policy(tampered)

    forged = deepcopy(policy)
    forged["authority"]["llm"] = True
    forged = _reseal(forged, "policy_id")
    assert forged["policy_id"] != policy["policy_id"]
    with pytest.raises(FundamentalContractError, match="boundary"):
        validate_fundamental_component_policy(forged)


def test_policy_metric_order_is_owner_semantic_and_changes_identity() -> None:
    components = deepcopy(_policy()["components"])
    earnings = next(row for row in components if row["component"] == "earnings_revision")
    earnings["metric_rows"] = [
        {
            "metric_id": "EARNINGS_REVISION_3M",
            "direction": "HIGHER_IS_BETTER",
            "weight": "0.5",
        },
        {
            "metric_id": "EARNINGS_REVISION_1M",
            "direction": "HIGHER_IS_BETTER",
            "weight": "0.5",
        },
    ]
    declared = _policy(components=components)
    reversed_components = deepcopy(components)
    next(row for row in reversed_components if row["component"] == "earnings_revision")[
        "metric_rows"
    ].reverse()
    reversed_policy = _policy(components=reversed_components)

    declared_ids = next(
        row for row in declared["components"] if row["component"] == "earnings_revision"
    )["metric_rows"]
    assert [row["metric_id"] for row in declared_ids] == [
        "EARNINGS_REVISION_3M",
        "EARNINGS_REVISION_1M",
    ]
    assert declared["policy_id"] != reversed_policy["policy_id"]
    assert validate_fundamental_component_policy(declared) == declared
    assert validate_fundamental_component_policy(reversed_policy) == reversed_policy


def test_profile_calls_frozen_scorer_exactly_once_and_projects_decimals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = profile_module.score_fundamental_forward_v3
    calls = 0

    def counted(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(profile_module, "score_fundamental_forward_v3", counted)
    profile = build_fundamental_profile(**_closure())

    assert calls == 1
    assert profile["version"] == PROFILE_VERSION
    assert profile["company_code"] == SYMBOLS[0]
    assert profile["status"] == "PARTIAL"
    assert profile["coverage"] == "0.650000000000"
    assert profile["score_present"] is True
    assert profile["raw_score"] == profile["subject_record"]["raw_score"]
    assert profile["effective_score"] == profile["subject_record"]["effective_score"]
    assert not _contains_float(profile)
    assert "binary_float_repr" in str(profile["raw_float_audit"])
    assert profile["authority"]["llm"] is False


def test_profile_replays_actual_i2_component_receipt_without_monkeypatch() -> None:
    receipt, industry_closure = _real_i2_receipt_closure(SYMBOLS[0])
    components = deepcopy(_policy()["components"])
    for component in components:
        component["source_cutoff"] = AS_OF
    policy = build_fundamental_component_policy(
        components=components,
        owner_policy_ref=_exact_ref("owner-policy-i2"),
        created_at=AS_OF,
    )

    profile = build_fundamental_profile(
        **_closure(
            policy=policy,
            industry_component_receipts={SYMBOLS[0]: receipt},
            industry_component_validation_closures={SYMBOLS[0]: industry_closure},
        )
    )

    assert len(profile["industry_component_refs"]) == 1
    industry_row = next(
        row
        for row in profile["component_rows"]
        if row["company_code"] == SYMBOLS[0] and row["component"] == "industry_cycle"
    )
    assert industry_row["status"] == "AVAILABLE"
    assert profile["authority"]["provider"] is False
    assert profile["authority"]["portfolio"] is False


def test_profile_replay_validator_calls_scorer_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = _closure()
    profile = build_fundamental_profile(**closure)
    original = profile_module.score_fundamental_forward_v3
    calls = 0

    def counted(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(profile_module, "score_fundamental_forward_v3", counted)
    assert validate_fundamental_profile(profile, **closure) == profile
    assert calls == 1


def test_type7_winsor_and_ascii_order_are_deterministic() -> None:
    closure = _closure()
    first = build_fundamental_profile(**closure)
    reordered = _closure(
        symbols=tuple(reversed(SYMBOLS)),
        financial_metric_rows=list(reversed(closure["financial_metric_rows"])),
        component_metric_rows=list(reversed(closure["component_metric_rows"])),
    )
    second = build_fundamental_profile(**reordered)

    assert first == second
    earnings = next(
        row
        for row in first["component_rows"]
        if row["company_code"] == SYMBOLS[0] and row["component"] == "earnings_revision"
    )
    assert earnings["metric_rows"][0]["winsorized_value"] == "5.000000000000"
    assert earnings["metric_rows"][0]["projected_value"] == "0.333333333333"


def test_missing_component_stays_missing_without_zero_or_mean_imputation() -> None:
    rows = [
        row
        for row in _owner_rows()
        if not (row["company_code"] == SYMBOLS[0] and row["component"] == "valuation")
    ]
    profile = build_fundamental_profile(**_closure(component_metric_rows=rows))
    valuation = next(
        row
        for row in profile["component_rows"]
        if row["company_code"] == SYMBOLS[0] and row["component"] == "valuation"
    )

    assert valuation["status"] == "MISSING"
    assert valuation["score"] is None
    evidence = {row["component"]: row for row in profile["subject_record"]["component_evidence"]}
    assert evidence["valuation"]["score"] is None
    assert evidence["valuation"]["status"] == "MISSING_VALUE"


def test_i2_i3_receipts_are_replayed_and_bound_as_projection_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    industry = {
        symbol: _component_receipt(version=INDUSTRY_COMPONENT_VERSION, score="0.8")
        for symbol in SYMBOLS
    }
    theme = {
        symbol: _component_receipt(version=THEME_COMPONENT_VERSION, score="0.7")
        for symbol in SYMBOLS
    }
    industry_closures = {symbol: _industry_closure(symbol) for symbol in SYMBOLS}
    theme_closures = {symbol: _theme_closure(symbol) for symbol in SYMBOLS}
    industry_calls: list[str] = []
    theme_calls: list[str] = []

    def validate_industry(document: dict, **closure: Any) -> dict:
        industry_calls.append(closure["identity_evaluation"]["subject_id"])
        return document

    def validate_theme(document: dict, **closure: Any) -> dict:
        theme_calls.append(closure["exposure_receipt"]["company_code"])
        return document

    monkeypatch.setattr(profile_module, "validate_industry_component_receipt", validate_industry)
    monkeypatch.setattr(profile_module, "validate_theme_component_receipt", validate_theme)
    profile = build_fundamental_profile(
        **_closure(
            industry_component_receipts=industry,
            industry_component_validation_closures=industry_closures,
            theme_component_receipts=theme,
            theme_component_validation_closures=theme_closures,
        )
    )

    assert industry_calls == list(SYMBOLS)
    assert theme_calls == list(SYMBOLS)
    assert len(profile["industry_component_refs"]) == len(SYMBOLS)
    assert len(profile["theme_component_refs"]) == len(SYMBOLS)
    assert profile["status"] == "COMPLETE"
    assert profile["coverage"] == "1.000000000000"


def test_future_receipt_duplicate_metric_float_and_domain_mismatch_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = _financial_rows()
    duplicate.append(deepcopy(duplicate[0]))
    with pytest.raises(FundamentalContractError, match="duplicate"):
        build_fundamental_profile(**_closure(financial_metric_rows=duplicate))

    floating = _owner_rows()
    floating[0]["value"] = 1.0
    with pytest.raises(Exception, match="binary float"):
        build_fundamental_profile(**_closure(component_metric_rows=floating))

    with pytest.raises(FundamentalContractError, match="implementation SHA"):
        build_fundamental_profile(**_closure(scorer_implementation_sha256=SHA_C))

    non_pit = _financial_rows()
    non_pit[0]["source_ref"]["cutoff"] = AS_OF
    with pytest.raises(IntelligenceV2ContractError, match="cutoff exceeds availability"):
        build_fundamental_profile(**_closure(financial_metric_rows=non_pit))

    receipt = _component_receipt(
        version=INDUSTRY_COMPONENT_VERSION,
        score="0.8",
        timestamp=FUTURE,
    )
    monkeypatch.setattr(
        profile_module,
        "validate_industry_component_receipt",
        lambda document, **_closure: document,
    )
    with pytest.raises(FundamentalContractError, match="future-known"):
        build_fundamental_profile(
            **_closure(
                industry_component_receipts={SYMBOLS[0]: receipt},
                industry_component_validation_closures={SYMBOLS[0]: _industry_closure(SYMBOLS[0])},
            )
        )

    with pytest.raises(FundamentalContractError, match="domains"):
        build_fundamental_profile(
            **_closure(
                industry_component_receipts={SYMBOLS[0]: receipt},
                industry_component_validation_closures={},
            )
        )


def test_profile_tamper_and_resealed_forgery_fail_validation() -> None:
    closure = _closure()
    profile = build_fundamental_profile(**closure)
    tampered = deepcopy(profile)
    tampered["coverage"] = "1.000000000000"
    with pytest.raises(Exception, match="mismatch"):
        validate_fundamental_profile(tampered, **closure)

    forged = _reseal(tampered, "profile_id")
    with pytest.raises(FundamentalContractError, match="deterministic replay"):
        validate_fundamental_profile(forged, **closure)


def test_artifacts_contain_no_binary_float_and_policy_weights_are_decimal() -> None:
    policy = _policy()
    profile = build_fundamental_profile(**_closure(policy=policy))

    assert not _contains_float(policy)
    assert not _contains_float(profile)
    for component in policy["components"]:
        total = sum(
            (Decimal(row["weight"]) for row in component["metric_rows"]),
            Decimal("0"),
        )
        assert total == Decimal("1.000000000000")
