from __future__ import annotations

import ast
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from quant_investor.intelligence_v2._core import (
    IntelligenceV2ContractError,
    seal,
)
from quant_investor.intelligence_v2.theme import (
    build_theme_component_policy,
    build_theme_component_receipt,
    build_theme_lifecycle_policy,
    build_theme_membership_catalog,
    build_theme_registry,
    build_theme_risk_policy,
    build_theme_risk_receipt,
    resolve_theme_exposure,
    validate_theme_component_receipt,
    validate_theme_exposure_receipt,
    validate_theme_lifecycle_policy,
    validate_theme_risk_receipt,
)

AS_OF = "2026-08-07T12:00:00Z"
SESSION = "20260807"
REPOSITORY = Path(__file__).resolve().parents[2]


def _exact_ref(name: str, *, available_at: str = "2026-08-07T10:00:00Z") -> dict:
    return {
        "artifact_id": f"source:{name}",
        "artifact_version": "myquant.test.source.v1",
        "available_at": available_at,
        "byte_sha256": (name.encode().hex() + "0" * 64)[:64],
        "cutoff": "2026-08-07T09:00:00Z",
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": (name.encode().hex() + "1" * 64)[:64],
    }


def _theme_row(
    theme_id: str,
    *,
    parent: str | None,
    level: int,
    status: str = "ACTIVE",
    effective_from: str = "20200101",
    effective_to: str | None = None,
) -> dict:
    return {
        "theme_id": theme_id,
        "display_name": theme_id.replace("_", " ").title(),
        "parent_theme_id": parent,
        "level": level,
        "status": status,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "available_at": "2026-08-07T10:00:00Z",
        "source_ref": _exact_ref(f"registry-{theme_id}"),
    }


def _registry() -> dict:
    return build_theme_registry(
        themes=[
            _theme_row("ai", parent=None, level=0),
            _theme_row("cloud", parent="ai", level=1),
            _theme_row("robotics", parent="ai", level=1),
            _theme_row(
                "retired_theme",
                parent="ai",
                level=1,
                status="RETIRED",
                effective_to="20241231",
            ),
        ],
        as_of=AS_OF,
    )


def _coverage(company_code: str, status: str = "COVERED") -> dict:
    return {
        "company_code": company_code,
        "status": status,
        "available_at": "2026-08-07T10:00:00Z",
        "source_ref": _exact_ref(f"coverage-{company_code.replace('.', '-')}"),
    }


def _membership(
    company_code: str,
    theme_id: str,
    weight: str,
    *,
    basis: str = "REVENUE",
) -> dict:
    return {
        "company_code": company_code,
        "theme_id": theme_id,
        "provider_id": "official",
        "exposure_basis": basis,
        "exposure_weight": weight,
        "effective_from": "20200101",
        "effective_to": None,
        "available_at": "2026-08-07T10:00:00Z",
        "source_ref": _exact_ref(f"membership-{company_code.replace('.', '-')}-{theme_id}"),
    }


def _catalog(
    registry: dict,
    *,
    companies: list[str],
    memberships: list[dict],
    scope_status: str = "COMPLETE",
) -> dict:
    return build_theme_membership_catalog(
        registry=registry,
        scope_status=scope_status,
        scope_ref=_exact_ref("catalog-scope"),
        coverage_rows=[_coverage(company) for company in companies],
        membership_rows=memberships,
        as_of=AS_OF,
    )


def _lifecycle(registry: dict) -> dict:
    rows = []
    for theme in registry["themes"]:
        status = "RETIRED" if theme["theme_id"] == "retired_theme" else "ACTIVE"
        rows.append(
            {
                "theme_id": theme["theme_id"],
                "provider_id": "official",
                "status": status,
                "effective_from": "20250101" if status == "RETIRED" else "20200101",
                "effective_to": None,
                "available_at": "2026-08-07T10:00:00Z",
                "source_ref": _exact_ref(f"lifecycle-{theme['theme_id']}"),
            }
        )
    return build_theme_lifecycle_policy(
        registry=registry,
        provider_precedence=["official"],
        cap_level=0,
        lifecycle_rows=rows,
        owner_policy_ref=_exact_ref("owner-lifecycle-policy"),
        as_of=AS_OF,
    )


def _available_closure() -> tuple[dict, dict, dict, dict, dict]:
    registry = _registry()
    catalog = _catalog(
        registry,
        companies=["000001.SZ", "000002.SZ"],
        memberships=[
            _membership("000001.SZ", "robotics", "0.600000000000"),
            _membership("000001.SZ", "cloud", "0.400000000000", basis="PRODUCT"),
        ],
    )
    lifecycle = _lifecycle(registry)
    exposure = resolve_theme_exposure(
        company_code="000001.SZ",
        registry=registry,
        membership_catalog=catalog,
        lifecycle_policy=lifecycle,
        as_of=AS_OF,
    )
    closure = {
        "registry": registry,
        "membership_catalog": catalog,
        "lifecycle_policy": lifecycle,
        "as_of": AS_OF,
    }
    return registry, catalog, lifecycle, exposure, closure


def _component_policy() -> dict:
    return build_theme_component_policy(
        metric_rows=[
            {
                "metric_id": "lifecycle",
                "direction": "LOWER_IS_BETTER",
                "weight": "0.250000000000",
            },
            {
                "metric_id": "revenue_exposure",
                "direction": "HIGHER_IS_BETTER",
                "weight": "0.750000000000",
            },
        ],
        minimum_coverage="1.000000000000",
        missing_rule="BLOCK_COMPONENT",
        owner_policy_ref=_exact_ref("owner-component-policy"),
        created_at=AS_OF,
    )


def _metric(
    theme_id: str,
    metric_id: str,
    value: str,
    source_kind: str,
) -> dict:
    return {
        "theme_id": theme_id,
        "metric_id": metric_id,
        "normalized_value": value,
        "available_at": "2026-08-07T10:00:00Z",
        "source_kind": source_kind,
        "source_ref": _exact_ref(f"metric-{theme_id}-{metric_id}"),
    }


def _metrics() -> list[dict]:
    return [
        _metric("cloud", "lifecycle", "0.200000000000", "SOURCE_BOUND_LIFECYCLE"),
        _metric(
            "cloud",
            "revenue_exposure",
            "0.400000000000",
            "SOURCE_BOUND_REVENUE_EXPOSURE",
        ),
        _metric("robotics", "lifecycle", "0.100000000000", "SOURCE_BOUND_LIFECYCLE"),
        _metric(
            "robotics",
            "revenue_exposure",
            "0.800000000000",
            "SOURCE_BOUND_REVENUE_EXPOSURE",
        ),
    ]


def _reseal(document: dict, identity_field: str, **changes: object) -> dict:
    body = deepcopy(document)
    body.pop(identity_field)
    body.pop("semantic_sha256")
    body.update(changes)
    return seal(body, identity_field=identity_field)


def test_available_multi_theme_exposure_sums_to_one_and_collapses_cap_bucket() -> None:
    registry, catalog, lifecycle, exposure, _ = _available_closure()
    assert exposure["status"] == "AVAILABLE"
    assert [row["theme_id"] for row in exposure["exposure_rows"]] == [
        "cloud",
        "robotics",
    ]
    assert sum(
        (Decimal(row["exposure_weight"]) for row in exposure["exposure_rows"]),
        Decimal("0"),
    ) == Decimal("1.000000000000")
    assert exposure["cap_bucket_rows"] == [
        {
            "bucket_id": "ai",
            "exposure_weight": "1.000000000000",
            "theme_ids": ["cloud", "robotics"],
        }
    ]
    assert exposure["authority"]["llm"] is False
    assert exposure["authority"]["portfolio"] is False
    assert (
        validate_theme_exposure_receipt(
            exposure,
            registry=registry,
            membership_catalog=catalog,
            lifecycle_policy=lifecycle,
            as_of=AS_OF,
        )
        == exposure
    )


def test_no_membership_requires_complete_catalog_and_never_gets_neutral_score() -> None:
    registry = _registry()
    lifecycle = _lifecycle(registry)
    complete = _catalog(registry, companies=["000002.SZ"], memberships=[])
    no_membership = resolve_theme_exposure(
        company_code="000002.SZ",
        registry=registry,
        membership_catalog=complete,
        lifecycle_policy=lifecycle,
        as_of=AS_OF,
    )
    assert no_membership["status"] == "NO_MEMBERSHIP"
    assert no_membership["exposure_rows"] == []
    assert no_membership["cap_bucket_rows"] == [
        {
            "bucket_id": "NO_THEME",
            "exposure_weight": "1.000000000000",
            "theme_ids": [],
        }
    ]
    closure = {
        "registry": registry,
        "membership_catalog": complete,
        "lifecycle_policy": lifecycle,
        "as_of": AS_OF,
    }
    component = build_theme_component_receipt(
        exposure_receipt=no_membership,
        exposure_closure=closure,
        component_policy=_component_policy(),
        metric_rows=[],
        as_of=AS_OF,
    )
    assert component["status"] == "MISSING"
    assert component["component_score"] is None
    assert component["blocker_codes"] == ["NO_MEMBERSHIP_COMPONENT_MISSING"]

    incomplete = _catalog(
        registry,
        companies=["000002.SZ"],
        memberships=[],
        scope_status="INCOMPLETE",
    )
    unresolved = resolve_theme_exposure(
        company_code="000002.SZ",
        registry=registry,
        membership_catalog=incomplete,
        lifecycle_policy=lifecycle,
        as_of=AS_OF,
    )
    assert unresolved["status"] == "UNMAPPED"
    assert unresolved["cap_bucket_rows"] == []


@pytest.mark.parametrize(
    ("memberships", "expected_status", "expected_blocker"),
    [
        (
            [
                _membership("000004.SZ", "ai", "0.500000000000"),
                _membership("000004.SZ", "robotics", "0.500000000000"),
            ],
            "AMBIGUOUS",
            "THEME_MEMBERSHIP_AMBIGUOUS",
        ),
        (
            [
                _membership("000004.SZ", "cloud", "0.400000000000"),
                _membership("000004.SZ", "robotics", "0.500000000000"),
            ],
            "AMBIGUOUS",
            "THEME_EXPOSURE_WEIGHT_INVALID",
        ),
    ],
)
def test_parent_child_and_invalid_weight_are_ambiguous(
    memberships: list[dict], expected_status: str, expected_blocker: str
) -> None:
    registry = _registry()
    catalog = _catalog(registry, companies=["000004.SZ"], memberships=memberships)
    result = resolve_theme_exposure(
        company_code="000004.SZ",
        registry=registry,
        membership_catalog=catalog,
        lifecycle_policy=_lifecycle(registry),
        as_of=AS_OF,
    )
    assert result["status"] == expected_status
    assert result["exposure_rows"] == []
    assert expected_blocker in result["blocker_codes"]


def test_owner_provider_precedence_is_ordered_semantics_and_replay_sensitive() -> None:
    registry = _registry()
    membership_rows = [
        {
            **_membership("000005.SZ", "cloud", "1.000000000000", basis="REVENUE"),
            "provider_id": "z_official",
        },
        {
            **_membership("000005.SZ", "cloud", "1.000000000000", basis="PRODUCT"),
            "provider_id": "a_alternate",
        },
    ]
    catalog = build_theme_membership_catalog(
        registry=registry,
        scope_status="COMPLETE",
        scope_ref=_exact_ref("catalog-scope-precedence"),
        coverage_rows=[_coverage("000005.SZ")],
        membership_rows=membership_rows,
        as_of=AS_OF,
    )
    catalog_reordered = build_theme_membership_catalog(
        registry=registry,
        scope_status="COMPLETE",
        scope_ref=_exact_ref("catalog-scope-precedence"),
        coverage_rows=[_coverage("000005.SZ")],
        membership_rows=list(reversed(membership_rows)),
        as_of=AS_OF,
    )
    assert catalog == catalog_reordered
    lifecycle_rows = []
    for provider in ("z_official", "a_alternate"):
        for theme in registry["themes"]:
            lifecycle_rows.append(
                {
                    "theme_id": theme["theme_id"],
                    "provider_id": provider,
                    "status": ("RETIRED" if theme["theme_id"] == "retired_theme" else "ACTIVE"),
                    "effective_from": (
                        "20250101" if theme["theme_id"] == "retired_theme" else "20200101"
                    ),
                    "effective_to": None,
                    "available_at": "2026-08-07T10:00:00Z",
                    "source_ref": _exact_ref(f"lifecycle-{provider}-{theme['theme_id']}"),
                }
            )
    owner_first = build_theme_lifecycle_policy(
        registry=registry,
        provider_precedence=["z_official", "a_alternate"],
        cap_level=0,
        lifecycle_rows=lifecycle_rows,
        owner_policy_ref=_exact_ref("owner-lifecycle-policy-precedence"),
        as_of=AS_OF,
    )
    alternate_first = build_theme_lifecycle_policy(
        registry=registry,
        provider_precedence=["a_alternate", "z_official"],
        cap_level=0,
        lifecycle_rows=lifecycle_rows,
        owner_policy_ref=_exact_ref("owner-lifecycle-policy-precedence"),
        as_of=AS_OF,
    )
    owner_first_reordered = build_theme_lifecycle_policy(
        registry=registry,
        provider_precedence=["z_official", "a_alternate"],
        cap_level=0,
        lifecycle_rows=list(reversed(lifecycle_rows)),
        owner_policy_ref=_exact_ref("owner-lifecycle-policy-precedence"),
        as_of=AS_OF,
    )
    assert owner_first["provider_precedence"] == ["z_official", "a_alternate"]
    assert owner_first["lifecycle_policy_id"] != alternate_first["lifecycle_policy_id"]
    assert owner_first == owner_first_reordered
    assert validate_theme_lifecycle_policy(owner_first, registry=registry) == owner_first
    assert validate_theme_lifecycle_policy(alternate_first, registry=registry) == alternate_first

    official = resolve_theme_exposure(
        company_code="000005.SZ",
        registry=registry,
        membership_catalog=catalog,
        lifecycle_policy=owner_first,
        as_of=AS_OF,
    )
    alternate = resolve_theme_exposure(
        company_code="000005.SZ",
        registry=registry,
        membership_catalog=catalog,
        lifecycle_policy=alternate_first,
        as_of=AS_OF,
    )
    assert official["exposure_rows"][0]["exposure_basis"] == "REVENUE"
    assert alternate["exposure_rows"][0]["exposure_basis"] == "PRODUCT"
    assert official["lifecycle_policy_ref"] != alternate["lifecycle_policy_ref"]


def test_retired_membership_is_explicit_and_cannot_enter_component() -> None:
    registry = _registry()
    catalog = _catalog(
        registry,
        companies=["000003.SZ"],
        memberships=[_membership("000003.SZ", "retired_theme", "1.000000000000")],
    )
    lifecycle = _lifecycle(registry)
    result = resolve_theme_exposure(
        company_code="000003.SZ",
        registry=registry,
        membership_catalog=catalog,
        lifecycle_policy=lifecycle,
        as_of=AS_OF,
    )
    assert result["status"] == "RETIRED"
    assert result["blocker_codes"] == ["THEME_MEMBERSHIP_RETIRED"]


def test_component_is_decimal_deterministic_and_source_bound() -> None:
    _, _, _, exposure, closure = _available_closure()
    policy = _component_policy()
    metrics = _metrics()
    before = deepcopy((exposure, closure, policy, metrics))
    component = build_theme_component_receipt(
        exposure_receipt=exposure,
        exposure_closure=closure,
        component_policy=policy,
        metric_rows=list(reversed(metrics)),
        as_of=AS_OF,
    )
    assert component["status"] == "AVAILABLE"
    assert component["component_score"] == "0.695000000000"
    assert [row["theme_id"] for row in component["theme_rows"]] == [
        "cloud",
        "robotics",
    ]
    assert (
        validate_theme_component_receipt(
            component,
            exposure_receipt=exposure,
            exposure_closure=closure,
            component_policy=policy,
            metric_rows=metrics,
            as_of=AS_OF,
        )
        == component
    )
    assert before == (exposure, closure, policy, metrics)

    forged = _reseal(
        component,
        "component_receipt_id",
        component_score="1.000000000000",
    )
    with pytest.raises(IntelligenceV2ContractError):
        validate_theme_component_receipt(
            forged,
            exposure_receipt=exposure,
            exposure_closure=closure,
            component_policy=policy,
            metric_rows=metrics,
            as_of=AS_OF,
        )


def test_llm_or_binary_float_cannot_create_theme_identity_or_score() -> None:
    registry, _, _, exposure, closure = _available_closure()
    with pytest.raises(IntelligenceV2ContractError):
        build_theme_membership_catalog(
            registry=registry,
            scope_status="COMPLETE",
            scope_ref=_exact_ref("catalog-scope-llm"),
            coverage_rows=[_coverage("000001.SZ")],
            membership_rows=[_membership("000001.SZ", "robotics", 1.0)],
            as_of=AS_OF,
        )
    metrics = _metrics()
    metrics[0]["source_kind"] = "LLM_NARRATIVE"
    with pytest.raises(IntelligenceV2ContractError):
        build_theme_component_receipt(
            exposure_receipt=exposure,
            exposure_closure=closure,
            component_policy=_component_policy(),
            metric_rows=metrics,
            as_of=AS_OF,
        )


def test_risk_receipt_is_separate_from_identity_and_has_owner_veto() -> None:
    _, _, _, exposure, closure = _available_closure()
    policy = build_theme_risk_policy(
        max_single_theme_exposure="0.500000000000",
        prohibited_theme_ids=["cloud"],
        hard_veto_codes_by_theme={"cloud": "CLOUD_POLICY_VETO"},
        owner_policy_ref=_exact_ref("owner-risk-policy"),
        created_at=AS_OF,
    )
    risk = build_theme_risk_receipt(
        exposure_receipt=exposure,
        exposure_closure=closure,
        risk_policy=policy,
        as_of=AS_OF,
    )
    assert risk["status"] == "AVAILABLE"
    assert risk["overall_severity"] == "0.600000000000"
    assert risk["hard_veto_codes"] == ["CLOUD_POLICY_VETO"]
    assert [row["theme_id"] for row in risk["risk_rows"]] == [
        "cloud",
        "robotics",
    ]
    assert (
        validate_theme_risk_receipt(
            risk,
            exposure_receipt=exposure,
            exposure_closure=closure,
            risk_policy=policy,
            as_of=AS_OF,
        )
        == risk
    )

    forged = _reseal(risk, "risk_receipt_id", hard_veto_codes=[])
    with pytest.raises(IntelligenceV2ContractError):
        validate_theme_risk_receipt(
            forged,
            exposure_receipt=exposure,
            exposure_closure=closure,
            risk_policy=policy,
            as_of=AS_OF,
        )


def test_theme_package_has_no_io_provider_model_or_mutation_imports() -> None:
    root = REPOSITORY / "quant_investor" / "intelligence_v2" / "theme"
    forbidden_roots = {
        "aiohttp",
        "httpx",
        "openai",
        "requests",
        "socket",
        "subprocess",
        "tushare",
        "urllib",
    }
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.names[0].name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        }
        imports.update(
            str(node.module or "").split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 0
        )
        assert imports.isdisjoint(forbidden_roots), (path, imports)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert calls.isdisjoint({"open", "exec", "eval", "compile"}), (path, calls)
