from __future__ import annotations

import copy
from typing import Any

import pytest

from quant_investor.intelligence_v2.industry import (
    evaluate_industry_identity,
    build_industry_identity_policy,
    validate_industry_evaluation_receipt,
)
from quant_investor.intelligence_v2.industry.models import IndustryContractError
from quant_investor.intelligence_v2.sources.tushare.industry import (
    compile_tushare_sw2021_industry_source,
)

NOW = "2026-08-09T08:00:00Z"


def source_ref(name: str) -> dict[str, str]:
    return {
        "artifact_id": name,
        "artifact_version": f"{name}.v1",
        "available_at": NOW,
        "byte_sha256": "a" * 64,
        "cutoff": NOW,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": "b" * 64,
    }


def taxonomy_row(code: str, name: str, level: str, parent: str | None) -> dict[str, Any]:
    return {
        "index_code": code,
        "industry_name": name,
        "parent_code": parent,
        "level": level,
        "industry_code": code,
        "is_pub": "1",
        "src": "SW2021",
    }


def member(symbol: str, *, flag: str = "Y") -> dict[str, Any]:
    return {
        "l1_code": "L1A",
        "l1_name": "一级A",
        "l2_code": "L2A",
        "l2_name": "二级A",
        "l3_code": "L3A",
        "l3_name": "三级A",
        "ts_code": symbol,
        "name": "公司",
        "in_date": "20210101",
        "out_date": None,
        "is_new": flag,
    }


def compile_source() -> dict[str, Any]:
    return compile_tushare_sw2021_industry_source(
        taxonomy_partitions={
            "L1": [taxonomy_row("L1A", "一级A", "L1", None)],
            "L2": [taxonomy_row("L2A", "二级A", "L2", "L1A")],
            "L3": [taxonomy_row("L3A", "三级A", "L3", "L2A")],
        },
        membership_partitions={
            "L3A|Y": [member("000001.SZ", flag="Y")],
            "L3A|N": [member("000001.SZ", flag="N")],
        },
        listing_identity_by_company={"000001.SZ": "listing-000001-sz"},
        taxonomy_source_ref=source_ref("taxonomy-source"),
        membership_source_ref=source_ref("membership-source"),
        taxonomy_effective_from="2021-12-13T00:00:00Z",
        cutoff=NOW,
        captured_at=NOW,
    )


def test_compiler_projects_one_active_l1_with_exact_exposure_and_i2_replay() -> None:
    compiled = compile_source()
    assert compiled["status"] == "AVAILABLE"
    assert compiled["blocked_subjects"] == {}
    membership = compiled["catalog"]["memberships"]
    assert len(membership) == 1
    assert membership[0]["industry_id"] == "TUSHARE_SW2021:L1A"
    assert membership[0]["exposure"] == "1.000000000000"

    policy = build_industry_identity_policy(
        created_at=NOW,
        provider_precedence=["TUSHARE_INDEX_MEMBER_ALL"],
        taxonomy_precedence=["TUSHARE_SW2021"],
        cap_taxonomy_level="TUSHARE_SW2021:L1",
    )
    evaluation = evaluate_industry_identity(
        policy=policy,
        taxonomies=[compiled["taxonomy"]],
        catalogs=[compiled["catalog"]],
        subject_id="000001.SZ",
        listing_identity="listing-000001-sz",
        as_of=NOW,
    )
    assert evaluation["state"] == "AVAILABLE"
    assert evaluation["primary_industry_id"] == "TUSHARE_SW2021:L1A"
    assert (
        validate_industry_evaluation_receipt(
            evaluation,
            policy=policy,
            taxonomies=[compiled["taxonomy"]],
            catalogs=[compiled["catalog"]],
        )
        == evaluation
    )


def test_row_limit_scope_and_display_industry_are_fail_closed() -> None:
    with pytest.raises(Exception):
        compile_tushare_sw2021_industry_source(
            taxonomy_partitions={"L1": [], "L2": [], "L3": []},
            membership_partitions={},
            listing_identity_by_company={},
            taxonomy_source_ref=source_ref("taxonomy-source"),
            membership_source_ref=source_ref("membership-source"),
            taxonomy_effective_from="2021-12-13T00:00:00Z",
            cutoff=NOW,
            captured_at=NOW,
            stock_basic_industry="银行",
        )

    forged = compile_source()
    tampered = copy.deepcopy(forged["catalog"])
    tampered["memberships"][0]["industry_id"] = "FREE_TEXT:银行"
    policy = build_industry_identity_policy(
        created_at=NOW,
        provider_precedence=["TUSHARE_INDEX_MEMBER_ALL"],
        taxonomy_precedence=["TUSHARE_SW2021"],
        cap_taxonomy_level="TUSHARE_SW2021:L1",
    )
    with pytest.raises(Exception):
        evaluate_industry_identity(
            policy=policy,
            taxonomies=[forged["taxonomy"]],
            catalogs=[tampered],
            subject_id="000001.SZ",
            listing_identity="listing-000001-sz",
            as_of=NOW,
        )

    membership_partitions = {
        "L3A|Y": [member("000001.SZ", flag="Y")] * 2000,
        "L3A|N": [],
    }
    with pytest.raises(Exception, match="row limit"):
        compile_tushare_sw2021_industry_source(
            taxonomy_partitions={
                "L1": [taxonomy_row("L1A", "一级A", "L1", None)],
                "L2": [taxonomy_row("L2A", "二级A", "L2", "L1A")],
                "L3": [taxonomy_row("L3A", "三级A", "L3", "L2A")],
            },
            membership_partitions=membership_partitions,
            listing_identity_by_company={"000001.SZ": "listing-000001-sz"},
            taxonomy_source_ref=source_ref("taxonomy-source"),
            membership_source_ref=source_ref("membership-source"),
            taxonomy_effective_from="2021-12-13T00:00:00Z",
            cutoff=NOW,
            captured_at=NOW,
        )


def test_compiler_translates_shared_ref_errors_to_industry_contract() -> None:
    invalid_ref = source_ref("taxonomy-source")
    invalid_ref["byte_sha256"] = "invalid"
    with pytest.raises(IndustryContractError):
        compile_tushare_sw2021_industry_source(
            taxonomy_partitions={
                "L1": [taxonomy_row("L1A", "一级A", "L1", None)],
                "L2": [taxonomy_row("L2A", "二级A", "L2", "L1A")],
                "L3": [taxonomy_row("L3A", "三级A", "L3", "L2A")],
            },
            membership_partitions={
                "L3A|Y": [member("000001.SZ", flag="Y")],
                "L3A|N": [member("000001.SZ", flag="N")],
            },
            listing_identity_by_company={"000001.SZ": "listing-000001-sz"},
            taxonomy_source_ref=invalid_ref,
            membership_source_ref=source_ref("membership-source"),
            taxonomy_effective_from="2021-12-13T00:00:00Z",
            cutoff=NOW,
            captured_at=NOW,
        )
