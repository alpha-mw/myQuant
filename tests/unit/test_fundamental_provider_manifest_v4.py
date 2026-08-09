from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pytest

from quant_investor.intelligence_v2.sources.tushare import (
    build_endpoint_execution_plan,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    FundamentalV4ContractError,
    build_fundamental_execution_closure_v4,
    build_fundamental_request_plan_v4,
    build_logical_symbol_table_coverage_v4,
    build_provider_physical_request_receipt_v4,
    build_raw_table_evidence_v4,
    validate_fundamental_request_plan_v4,
    validate_fundamental_execution_closure_v4,
    validate_logical_symbol_table_coverage_v4,
    validate_provider_physical_request_receipt_v4,
    validate_raw_table_evidence_v4,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.models import (
    SOURCE_ENDPOINTS,
)

NOW = "2026-08-09T08:00:00Z"
CUTOFF = "2026-08-07T23:59:59Z"


def exact_ref(name: str) -> dict[str, str]:
    return {
        "artifact_id": name,
        "artifact_version": f"{name}.v1",
        "available_at": CUTOFF,
        "byte_sha256": "a" * 64,
        "cutoff": CUTOFF,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": "b" * 64,
    }


def endpoint_plans(*, created_at: str = NOW) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for table, endpoint in SOURCE_ENDPOINTS.items():
        dimension = "trade_date" if table == "daily_basic" else "period"
        value = "20260807" if table == "daily_basic" else "20260630"
        result[table] = build_endpoint_execution_plan(
            api_name=endpoint,
            lane="FUNDAMENTAL",
            permission_class="POINTS",
            official_document_url="https://tushare.pro/document/2?doc_id=1",
            official_document_id=f"tushare.{endpoint}",
            document_observed_at=created_at,
            documented_min_points=2000,
            strict_decimal_decode=True,
            expected_fields=["ts_code", dimension],
            fixed_params={dimension: value},
            partition_dimensions=[dimension],
            ordered_expected_partition_keyset=[f"{dimension}={value}"],
            documented_row_limit=6000,
            max_attempts=1,
            retry_schedule=[0],
            empty_partition_rule="BASELINE_IDENTITY_EMPTY",
            completeness_proof="EXACT_PARTITION_AND_COUNT",
            limit_hit_action="BLOCK",
            planned_terminal_request_count=1,
            planned_max_network_attempts=1,
            created_at=created_at,
        )
    return result


def business_sessions(start: str, end: str) -> list[str]:
    return [value.strftime("%Y%m%d") for value in pd.bdate_range(start, end)]


def build_plan(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "as_of": "20260807",
        "pit_cutoff": CUTOFF,
        "symbols": ["000001.SZ", "600000.SH"],
        "canonical_open_sessions": business_sessions("20210807", "20260807"),
        "market_scope_ref": exact_ref("market-scope"),
        "market_calendar_ref": exact_ref("market-calendar"),
        "baseline_provider_manifest_ref": exact_ref("baseline-provider-manifest"),
        "baseline_empty_partition_keyset": [],
        "endpoint_plans": endpoint_plans(),
        "max_attempts_per_partition": 2,
        "implementation_sha256": "c" * 64,
        "created_at": NOW,
    }
    values.update(overrides)
    return build_fundamental_request_plan_v4(**values)


def test_exact_inclusive_window_schedule_and_replay() -> None:
    endpoints = endpoint_plans()
    plan = build_plan(endpoint_plans=endpoints)

    assert plan["daily_start"] == "20210807"
    assert plan["financial_start"] == "20190807"
    assert plan["financial_periods"][0] == "20190930"
    assert plan["financial_periods"][-1] == "20260630"
    assert plan["window_years"] == {"daily": 5, "financial": 7}
    assert plan["baseline_planned_network_attempts"] == 12
    assert plan["planned_terminal_request_count"] == (
        len(plan["financial_periods"]) * 5 + len(plan["daily_open_sessions"])
    )
    assert plan["planned_max_network_attempts"] == (plan["planned_terminal_request_count"] * 2)
    assert (
        validate_fundamental_request_plan_v4(
            plan,
            endpoint_plans=endpoints,
        )
        == plan
    )

    closure = build_fundamental_execution_closure_v4(
        plan=plan,
        endpoint_plans=endpoints,
        created_at=NOW,
    )
    assert closure["request_plan"] == plan
    assert set(closure["endpoint_plans"]) == set(SOURCE_ENDPOINTS)
    assert validate_fundamental_execution_closure_v4(closure) == closure


def test_dateoffset_uses_calendar_year_semantics_on_leap_day() -> None:
    plan = build_plan(
        as_of="20240229",
        pit_cutoff="2024-02-29T23:59:59Z",
        created_at="2024-03-01T00:00:00Z",
        canonical_open_sessions=business_sessions("20190228", "20240229"),
        market_scope_ref={
            **exact_ref("market-scope"),
            "available_at": "2024-02-29T23:59:59Z",
            "cutoff": "2024-02-29T23:59:59Z",
        },
        market_calendar_ref={
            **exact_ref("market-calendar"),
            "available_at": "2024-02-29T23:59:59Z",
            "cutoff": "2024-02-29T23:59:59Z",
        },
        baseline_provider_manifest_ref={
            **exact_ref("baseline-provider-manifest"),
            "available_at": "2024-02-29T23:59:59Z",
            "cutoff": "2024-02-29T23:59:59Z",
        },
        endpoint_plans=endpoint_plans(created_at="2024-03-01T00:00:00Z"),
    )
    assert plan["daily_start"] == "20190228"
    assert plan["financial_start"] == "20170228"


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbols": ["600000.SH", "000001.SZ"]},
        {"symbols": ["000001.SZ", "000001.SZ"]},
        {"canonical_open_sessions": ["20260807"]},
        {"canonical_open_sessions": ["20260807", "20210809"]},
        {"max_attempts_per_partition": 0},
        {"baseline_empty_partition_keyset": ["income|period=19990101"]},
        {"pit_cutoff": "2026-08-10T00:00:00Z"},
    ],
)
def test_plan_rejects_unclosed_scope_window_or_attempts(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(FundamentalV4ContractError):
        build_plan(**overrides)


def test_plan_resealed_forgery_is_rejected() -> None:
    plan = build_plan()
    forged = copy.deepcopy(plan)
    forged["daily_start"] = "20210808"
    with pytest.raises(FundamentalV4ContractError):
        validate_fundamental_request_plan_v4(
            forged,
            endpoint_plans=endpoint_plans(),
        )


def physical_receipt(
    plan: dict[str, Any],
    endpoints: dict[str, dict[str, Any]],
    *,
    table: str,
    partition_id: str,
    status: str = "AVAILABLE",
) -> dict[str, Any]:
    available = status == "AVAILABLE"
    incomplete = status == "INCOMPLETE"
    return build_provider_physical_request_receipt_v4(
        plan=plan,
        endpoint_plans=endpoints,
        table=table,
        partition_id=partition_id,
        sanitized_params_sha256="d" * 64,
        attempts=1,
        provider_request_id="request-1" if available or incomplete else None,
        reported_count=1 if available or incomplete else 0,
        accepted_count=1 if available or incomplete else 0,
        has_more=incomplete,
        status=status,
        blocker_codes=["HAS_MORE"] if incomplete else [],
        raw_response_projection_sha256=(
            "e" * 64 if status in {"AVAILABLE", "EMPTY", "INCOMPLETE"} else None
        ),
        captured_at=NOW,
    )


def test_physical_request_receipt_is_partition_bound_and_replayable() -> None:
    endpoints = endpoint_plans()
    plan = build_plan(endpoint_plans=endpoints)
    partition_id = next(
        row["partition_id"] for row in plan["partition_rows"] if row["table"] == "income"
    )
    receipt = physical_receipt(
        plan,
        endpoints,
        table="income",
        partition_id=partition_id,
    )

    assert receipt["endpoint"] == "income_vip"
    assert receipt["strict_decimal_decode"] is True
    assert (
        validate_provider_physical_request_receipt_v4(
            receipt,
            plan=plan,
            endpoint_plans=endpoints,
        )
        == receipt
    )


@pytest.mark.parametrize(
    ("status", "expected_blocker"),
    [
        ("EMPTY", None),
        ("INCOMPLETE", "HAS_MORE"),
        ("PROVIDER_ERROR", None),
        ("SCHEMA_MISMATCH", None),
        ("TRANSPORT_ERROR", None),
    ],
)
def test_physical_terminal_statuses_are_explicit(
    status: str,
    expected_blocker: str | None,
) -> None:
    endpoints = endpoint_plans()
    plan_overrides: dict[str, Any] = {"endpoint_plans": endpoints}
    if status == "EMPTY":
        plan_overrides["baseline_empty_partition_keyset"] = ["forecast|period=20190930"]
    plan = build_plan(**plan_overrides)
    partition_id = next(
        row["partition_id"] for row in plan["partition_rows"] if row["table"] == "forecast"
    )
    receipt = physical_receipt(
        plan,
        endpoints,
        table="forecast",
        partition_id=partition_id,
        status=status,
    )
    assert receipt["status"] == status
    if expected_blocker is not None:
        assert expected_blocker in receipt["blocker_codes"]


def income_receipts(
    plan: dict[str, Any],
    endpoints: dict[str, dict[str, Any]],
    *,
    incomplete_ordinal: int | None = None,
) -> list[dict[str, Any]]:
    rows = [row for row in plan["partition_rows"] if row["table"] == "income"]
    return [
        physical_receipt(
            plan,
            endpoints,
            table="income",
            partition_id=row["partition_id"],
            status="INCOMPLETE" if index == incomplete_ordinal else "AVAILABLE",
        )
        for index, row in enumerate(rows)
    ]


def test_logical_coverage_binds_complete_physical_partition_keyset() -> None:
    endpoints = endpoint_plans()
    plan = build_plan(endpoint_plans=endpoints)
    receipts = income_receipts(plan, endpoints)
    coverage = build_logical_symbol_table_coverage_v4(
        plan=plan,
        endpoint_plans=endpoints,
        physical_receipts=receipts,
        company_code="000001.SZ",
        table="income",
        expected_start=plan["financial_periods"][0],
        expected_end=plan["financial_periods"][-1],
        observed_start=plan["financial_periods"][0],
        observed_end=plan["financial_periods"][-1],
        row_count=10,
        missing_reason_codes=[],
        duplicate_reason_codes=[],
        restatement_reason_codes=[],
        status="COMPLETE",
        assessed_at=NOW,
    )

    assert coverage["expected_partition_refs"] == sorted([row["receipt_id"] for row in receipts])
    assert (
        validate_logical_symbol_table_coverage_v4(
            coverage,
            plan=plan,
            endpoint_plans=endpoints,
            physical_receipts=receipts,
        )
        == coverage
    )


def test_logical_coverage_rejects_missing_partition_or_false_complete() -> None:
    endpoints = endpoint_plans()
    plan = build_plan(endpoint_plans=endpoints)
    receipts = income_receipts(plan, endpoints)
    with pytest.raises(FundamentalV4ContractError):
        build_logical_symbol_table_coverage_v4(
            plan=plan,
            endpoint_plans=endpoints,
            physical_receipts=receipts[:-1],
            company_code="000001.SZ",
            table="income",
            expected_start=plan["financial_periods"][0],
            expected_end=plan["financial_periods"][-1],
            observed_start=plan["financial_periods"][0],
            observed_end=plan["financial_periods"][-1],
            row_count=10,
            missing_reason_codes=[],
            duplicate_reason_codes=[],
            restatement_reason_codes=[],
            status="COMPLETE",
            assessed_at=NOW,
        )

    incomplete = income_receipts(plan, endpoints, incomplete_ordinal=0)
    with pytest.raises(FundamentalV4ContractError):
        build_logical_symbol_table_coverage_v4(
            plan=plan,
            endpoint_plans=endpoints,
            physical_receipts=incomplete,
            company_code="000001.SZ",
            table="income",
            expected_start=plan["financial_periods"][0],
            expected_end=plan["financial_periods"][-1],
            observed_start=plan["financial_periods"][0],
            observed_end=plan["financial_periods"][-1],
            row_count=10,
            missing_reason_codes=[],
            duplicate_reason_codes=[],
            restatement_reason_codes=[],
            status="COMPLETE",
            assessed_at=NOW,
        )


def test_raw_table_evidence_binds_exact_file_and_multiset() -> None:
    endpoints = endpoint_plans()
    plan = build_plan(endpoint_plans=endpoints)
    evidence = build_raw_table_evidence_v4(
        plan=plan,
        endpoint_plans=endpoints,
        lane="VIP",
        table="income",
        file_ref={
            **exact_ref("vip-income"),
            "relative_path": "provider_evidence/vip_raw/income.parquet",
        },
        row_count=10,
        column_order=["ts_code", "end_date", "n_income"],
        canonical_multiset_sha256="f" * 64,
        duplicate_row_count=0,
        winner_implementation_sha256="1" * 64,
        evidenced_at=NOW,
    )
    assert (
        validate_raw_table_evidence_v4(
            evidence,
            plan=plan,
            endpoint_plans=endpoints,
        )
        == evidence
    )

    forged = copy.deepcopy(evidence)
    forged["row_count"] = 11
    with pytest.raises(FundamentalV4ContractError):
        validate_raw_table_evidence_v4(
            forged,
            plan=plan,
            endpoint_plans=endpoints,
        )
