from __future__ import annotations

import copy
from dataclasses import asdict
from decimal import Decimal
from typing import Any

import pytest

from quant_investor.market.tushare import (
    TushareContractError,
    build_endpoint_execution_plan,
    build_tushare_endpoint_policy,
    build_tushare_request_receipt,
    build_tushare_schema_diagnostic_receipt,
    response_projection_sha256,
    validate_endpoint_execution_plan,
    validate_tushare_endpoint_policy,
    validate_tushare_request_receipt,
    validate_tushare_schema_diagnostic_receipt,
)
from quant_investor.market.tushare._core import FORBIDDEN_VERSION_FIELDS, canonical_bytes
from quant_investor.market.tushare_transport import TushareSchemaDiagnostic

NOW = "2026-08-14T08:00:00Z"
OBSERVED = "2026-08-14T07:59:00Z"


def build_plan(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "api_name": "daily_basic",
        "lane": "FUNDAMENTAL",
        "permission_class": "POINTS",
        "official_document_url": "https://tushare.pro/document/2?doc_id=32",
        "official_document_id": "tushare.doc.32",
        "document_observed_at": OBSERVED,
        "documented_min_points": 2000,
        "strict_decimal_decode": True,
        "expected_fields": ["ts_code", "trade_date", "close", "total_mv"],
        "fixed_params": {"trade_date": "20260813"},
        "partition_dimensions": ["trade_date"],
        "ordered_expected_partition_keyset": ["trade_date=20260813"],
        "documented_row_limit": 6000,
        "max_attempts": 1,
        "retry_schedule": [0],
        "empty_partition_rule": "BASELINE_IDENTITY_EMPTY",
        "completeness_proof": "EXACT_PARTITION_AND_COUNT",
        "limit_hit_action": "BLOCK",
        "planned_terminal_request_count": 1,
        "planned_max_network_attempts": 1,
        "created_at": NOW,
    }
    values.update(overrides)
    return build_endpoint_execution_plan(**values)


def _assert_stable_contract(document: dict[str, Any], *, identity_field: str) -> None:
    assert document["kind"].startswith("market.tushare.")
    assert len(document["contract_sha256"]) == 64
    assert len(document[identity_field]) == 64
    assert len(document["semantic_sha256"]) == 64
    assert FORBIDDEN_VERSION_FIELDS.isdisjoint(document)
    rendered = canonical_bytes(document)
    for field in FORBIDDEN_VERSION_FIELDS:
        assert f'"{field}"'.encode() not in rendered


def test_endpoint_plan_and_policy_use_stable_exact_contracts() -> None:
    daily = build_plan()
    income = build_plan(
        api_name="income_vip",
        official_document_id="tushare.doc.33",
        expected_fields=["ts_code", "end_date", "revenue"],
        fixed_params={"period": "20260630"},
        partition_dimensions=["period"],
        ordered_expected_partition_keyset=["period=20260630"],
        documented_row_limit=5000,
    )
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[income, daily])

    assert validate_endpoint_execution_plan(daily) == daily
    assert validate_tushare_endpoint_policy(policy) == policy
    assert [row["api_name"] for row in policy["endpoint_plans"]] == [
        "daily_basic",
        "income_vip",
    ]
    _assert_stable_contract(daily, identity_field="plan_id")
    _assert_stable_contract(policy, identity_field="policy_id")
    assert policy["authority"]["provider"] is False
    assert policy["research_only"] is True
    assert policy["production"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"strict_decimal_decode": False},
        {"fixed_params": {"token": "SECRET"}},
        {"expected_fields": ["ts_code", "ts_code"]},
        {"ordered_expected_partition_keyset": ["x", "x"]},
        {"documented_row_limit": 0},
        {"max_attempts": 2, "retry_schedule": [0]},
        {"planned_terminal_request_count": 2},
        {"planned_max_network_attempts": 2},
        {"official_document_url": "http://tushare.pro/document/2"},
        {"document_observed_at": "2026-08-14T08:00:01Z"},
    ],
)
def test_endpoint_plan_rejects_unsafe_or_unclosed_inputs(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(TushareContractError):
        build_plan(**overrides)


def test_exact_validation_rejects_unknown_and_version_fields() -> None:
    plan = build_plan()
    forged = copy.deepcopy(plan)
    forged["unknown"] = "x"
    with pytest.raises(TushareContractError):
        validate_endpoint_execution_plan(forged)

    versioned = copy.deepcopy(plan)
    versioned["version"] = "legacy"
    with pytest.raises(TushareContractError):
        validate_endpoint_execution_plan(versioned)


def test_request_and_schema_receipts_bind_semantic_five_field_refs() -> None:
    plan = build_plan(
        api_name="forecast_vip",
        official_document_id="tushare.doc.45",
        expected_fields=["ts_code", "ann_date", "end_date", "summary"],
        fixed_params={"period": "20260630"},
        partition_dimensions=["period"],
        ordered_expected_partition_keyset=["period=20260630"],
        documented_row_limit=3500,
    )
    params = {"period": "20260630"}
    request = build_tushare_request_receipt(
        plan=plan,
        partition_key="period=20260630",
        partition_ordinal=0,
        sanitized_params=params,
        requested_at=NOW,
    )
    diagnostic = asdict(
        TushareSchemaDiagnostic(
            api_name="forecast_vip",
            status="OBSERVED",
            provider_code=0,
            request_id_sha256="1" * 64,
            response_body_sha256="2" * 64,
            provider_reported_count=1,
            item_count=1,
            has_more=False,
            observed_fields=("ts_code", "ann_date", "end_date"),
            expected_fields_match=False,
            row_widths=(3,),
            cell_types=("TEXT",),
            text_cell_count=3,
            non_nfc_text_count=0,
            max_text_utf8_bytes=9,
        )
    )
    receipt = build_tushare_schema_diagnostic_receipt(
        plan=plan,
        request_receipt=request,
        sanitized_params=params,
        diagnostic=diagnostic,
        captured_at=NOW,
    )

    assert validate_tushare_request_receipt(request, plan=plan, sanitized_params=params) == request
    assert (
        validate_tushare_schema_diagnostic_receipt(
            receipt,
            plan=plan,
            request_receipt=request,
            sanitized_params=params,
            diagnostic=diagnostic,
        )
        == receipt
    )
    assert set(request["plan_ref"]) == {
        "artifact_id",
        "byte_sha256",
        "contract_sha256",
        "kind",
        "semantic_sha256",
    }
    assert request["plan_ref"]["kind"] == plan["kind"]
    assert request["plan_ref"]["contract_sha256"] == plan["contract_sha256"]
    _assert_stable_contract(request, identity_field="request_receipt_id")
    _assert_stable_contract(receipt, identity_field="schema_diagnostic_receipt_id")


def test_response_projection_is_decimal_exact_and_rejects_non_nfc() -> None:
    first = response_projection_sha256(((Decimal("1.000000000001"), "甲" * 1704),))
    second = response_projection_sha256(((Decimal("1.000000000002"), "甲" * 1704),))
    assert first != second
    with pytest.raises(TushareContractError, match="Unicode NFC"):
        response_projection_sha256((("e\N{COMBINING ACUTE ACCENT}",),))
