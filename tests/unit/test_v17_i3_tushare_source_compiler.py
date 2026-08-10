from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from quant_investor.intelligence_v2._core import seal
from quant_investor.intelligence_v2.sources.tushare.theme import (
    compile_tushare_theme_source,
    validate_tushare_theme_source_receipt,
)
from quant_investor.intelligence_v2.theme import (
    resolve_theme_exposure,
    validate_theme_exposure_receipt,
)
from quant_investor.intelligence_v2.theme.models import ThemeContractError

TRADE_DATE = "20260807"
AS_OF = "2026-08-07T12:00:00Z"
CAPTURED = "2026-08-07T10:00:00Z"


def source_ref(name: str) -> dict[str, str]:
    encoded = name.encode().hex()
    return {
        "artifact_id": f"source:{name}",
        "artifact_version": "myquant.test.source.v1",
        "available_at": CAPTURED,
        "byte_sha256": (encoded + "0" * 64)[:64],
        "cutoff": CAPTURED,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": (encoded + "1" * 64)[:64],
    }


def dc_registry(code: str, name: str) -> dict[str, object]:
    return {
        "ts_code": code,
        "trade_date": TRADE_DATE,
        "name": name,
        "idx_type": "概念板块",
        "level": "1",
    }


def tdx_registry(code: str, name: str) -> dict[str, object]:
    return {
        "ts_code": code,
        "trade_date": TRADE_DATE,
        "name": name,
        "idx_type": "概念板块",
        "idx_count": 1,
    }


def dc_member(company: str, code: str, name: str = "公司") -> dict[str, str]:
    return {
        "trade_date": TRADE_DATE,
        "ts_code": code,
        "con_code": company,
        "name": name,
    }


def tdx_member(company: str, code: str, name: str = "公司") -> dict[str, str]:
    return {
        "ts_code": code,
        "trade_date": TRADE_DATE,
        "con_code": company,
        "con_name": name,
    }


def capture(name: str, rows: list[dict], status: str = "COMPLETE") -> dict:
    return {"status": status, "rows": rows, "source_ref": source_ref(name)}


def compile_source(
    *,
    companies: list[str],
    dc_rows: list[dict],
    dc_captures: dict[str, dict],
    tdx_rows: list[dict] | None = None,
    tdx_captures: dict[str, dict] | None = None,
) -> dict:
    has_tdx = tdx_rows is not None
    return compile_tushare_theme_source(
        trade_date=TRADE_DATE,
        company_keyset=companies,
        dc_registry_rows=dc_rows,
        dc_registry_source_ref=source_ref("dc-registry"),
        dc_membership_captures=dc_captures,
        tdx_registry_rows=[] if tdx_rows is None else tdx_rows,
        tdx_registry_source_ref=source_ref("tdx-registry") if has_tdx else None,
        tdx_membership_captures={} if tdx_captures is None else tdx_captures,
        scope_ref=source_ref("full-a-scope"),
        owner_policy_ref=source_ref("owner-theme-policy"),
        captured_at=CAPTURED,
        as_of=AS_OF,
    )


def exposure(bundle: dict, company: str) -> dict:
    document = resolve_theme_exposure(
        company_code=company,
        registry=bundle["registry"],
        membership_catalog=bundle["catalog"],
        lifecycle_policy=bundle["lifecycle_policy"],
        as_of=AS_OF,
    )
    assert (
        validate_theme_exposure_receipt(
            document,
            registry=bundle["registry"],
            membership_catalog=bundle["catalog"],
            lifecycle_policy=bundle["lifecycle_policy"],
            as_of=AS_OF,
        )
        == document
    )
    return document


def test_dc_complete_is_primary_and_equal_membership_weights_are_exact() -> None:
    company = "000001.SZ"
    bundle = compile_source(
        companies=[company],
        dc_rows=[dc_registry("BK1001.DC", "机器人"), dc_registry("BK1002.DC", "人工智能")],
        dc_captures={
            company: capture(
                "dc-member-000001",
                [dc_member(company, "BK1002.DC"), dc_member(company, "BK1001.DC")],
            )
        },
    )
    receipt = bundle["source_receipt"]
    assert receipt["tdx_fallback_company_keyset"] == []
    assert receipt["company_rows"][0]["provider"] == "TUSHARE_DC"
    resolved = exposure(bundle, company)
    assert resolved["status"] == "AVAILABLE"
    assert [row["theme_id"] for row in resolved["exposure_rows"]] == [
        "TUSHARE_DC:BK1001.DC",
        "TUSHARE_DC:BK1002.DC",
    ]
    assert all(row["exposure_basis"] == "EQUAL_MEMBERSHIP" for row in resolved["exposure_rows"])
    assert sum(
        (Decimal(row["exposure_weight"]) for row in resolved["exposure_rows"]),
        Decimal(0),
    ) == Decimal("1.000000000000")
    assert (
        validate_tushare_theme_source_receipt(
            receipt,
            registry=bundle["registry"],
            catalog=bundle["catalog"],
            lifecycle_policy=bundle["lifecycle_policy"],
        )
        == receipt
    )


def test_equal_membership_residual_is_assigned_to_last_ascii_theme() -> None:
    company = "000001.SZ"
    codes = ["BK1001.DC", "BK1002.DC", "BK1003.DC"]
    bundle = compile_source(
        companies=[company],
        dc_rows=[dc_registry(code, f"主题{index}") for index, code in enumerate(codes)],
        dc_captures={
            company: capture(
                "dc-member-000001",
                [dc_member(company, code) for code in reversed(codes)],
            )
        },
    )
    rows = exposure(bundle, company)["exposure_rows"]
    assert [row["exposure_weight"] for row in rows] == [
        "0.333333333333",
        "0.333333333333",
        "0.333333333334",
    ]


def test_complete_dc_no_membership_forbids_tdx_and_stays_no_membership() -> None:
    company = "000001.SZ"
    bundle = compile_source(
        companies=[company],
        dc_rows=[dc_registry("BK1001.DC", "机器人")],
        dc_captures={company: capture("dc-member-000001", [])},
    )
    assert exposure(bundle, company)["status"] == "NO_MEMBERSHIP"
    with pytest.raises(ThemeContractError, match="forbidden"):
        compile_source(
            companies=[company],
            dc_rows=[dc_registry("BK1001.DC", "机器人")],
            dc_captures={company: capture("dc-member-000001", [])},
            tdx_rows=[tdx_registry("880001.TDX", "机器人")],
            tdx_captures={},
        )


def test_incomplete_dc_seals_ascii_fallback_and_uses_tdx_only_for_that_company() -> None:
    companies = ["000001.SZ", "000002.SZ"]
    bundle = compile_source(
        companies=companies,
        dc_rows=[dc_registry("BK1001.DC", "机器人")],
        dc_captures={
            "000001.SZ": capture("dc-member-000001", [dc_member("000001.SZ", "BK1001.DC")]),
            "000002.SZ": capture("dc-member-000002", [], status="INCOMPLETE"),
        },
        tdx_rows=[tdx_registry("880001.TDX", "航运概念")],
        tdx_captures={
            "000002.SZ": capture("tdx-member-000002", [tdx_member("000002.SZ", "880001.TDX")])
        },
    )
    assert bundle["source_receipt"]["tdx_fallback_company_keyset"] == ["000002.SZ"]
    assert exposure(bundle, "000001.SZ")["exposure_rows"][0]["theme_id"].startswith("TUSHARE_DC:")
    assert exposure(bundle, "000002.SZ")["exposure_rows"][0]["theme_id"].startswith("TUSHARE_TDX:")


def test_unknown_dc_code_routes_to_tdx_and_both_incomplete_is_unmapped() -> None:
    company = "000001.SZ"
    fallback = compile_source(
        companies=[company],
        dc_rows=[dc_registry("BK1001.DC", "机器人")],
        dc_captures={company: capture("dc-member-000001", [dc_member(company, "BK9999.DC")])},
        tdx_rows=[tdx_registry("880001.TDX", "航运概念")],
        tdx_captures={company: capture("tdx-member-000001", [])},
    )
    assert fallback["source_receipt"]["tdx_fallback_company_keyset"] == [company]
    assert exposure(fallback, company)["status"] == "NO_MEMBERSHIP"

    unmapped = compile_source(
        companies=[company],
        dc_rows=[dc_registry("BK1001.DC", "机器人")],
        dc_captures={company: capture("dc-member-000001", [], status="INCOMPLETE")},
        tdx_rows=[tdx_registry("880001.TDX", "航运概念")],
        tdx_captures={company: capture("tdx-member-000001", [], status="INCOMPLETE")},
    )
    assert exposure(unmapped, company)["status"] == "UNMAPPED"


def test_conflicting_selected_registry_is_ambiguous_and_receipt_forgery_is_rejected() -> None:
    company = "000001.SZ"
    bundle = compile_source(
        companies=[company],
        dc_rows=[dc_registry("BK1001.DC", "机器人"), dc_registry("BK1001.DC", "机器人冲突")],
        dc_captures={company: capture("dc-member-000001", [dc_member(company, "BK1001.DC")])},
    )
    assert exposure(bundle, company)["status"] == "AMBIGUOUS"
    forged = deepcopy(bundle["source_receipt"])
    forged.pop("source_receipt_id")
    forged.pop("semantic_sha256")
    forged["company_rows"][0]["status"] = "COVERED"
    forged = seal(forged, identity_field="source_receipt_id")
    with pytest.raises(ThemeContractError, match="projection mismatch"):
        validate_tushare_theme_source_receipt(
            forged,
            registry=bundle["registry"],
            catalog=bundle["catalog"],
            lifecycle_policy=bundle["lifecycle_policy"],
        )


def test_tdx_keyset_and_snapshot_scope_are_fail_closed() -> None:
    company = "000001.SZ"
    with pytest.raises(ThemeContractError, match="keyset"):
        compile_source(
            companies=[company],
            dc_rows=[dc_registry("BK1001.DC", "机器人")],
            dc_captures={company: capture("dc-member-000001", [], status="INCOMPLETE")},
            tdx_rows=[tdx_registry("880001.TDX", "航运概念")],
            tdx_captures={},
        )
    wrong_day = tdx_member(company, "880001.TDX")
    wrong_day["trade_date"] = "20260806"
    bundle = compile_source(
        companies=[company],
        dc_rows=[dc_registry("BK1001.DC", "机器人")],
        dc_captures={company: capture("dc-member-000001", [], status="INCOMPLETE")},
        tdx_rows=[tdx_registry("880001.TDX", "航运概念")],
        tdx_captures={company: capture("tdx-member-000001", [wrong_day])},
    )
    assert exposure(bundle, company)["status"] == "UNMAPPED"
