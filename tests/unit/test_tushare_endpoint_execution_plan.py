from __future__ import annotations

import copy
from typing import Any

import pytest

from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_endpoint_execution_plan,
    build_tushare_endpoint_policy,
    validate_endpoint_execution_plan,
    validate_tushare_endpoint_policy,
)

NOW = "2026-08-09T08:00:00Z"
OBSERVED = "2026-08-09T07:59:00Z"


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
        "fixed_params": {"trade_date": "20260807"},
        "partition_dimensions": ["trade_date"],
        "ordered_expected_partition_keyset": ["trade_date=20260807"],
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


def test_plan_and_policy_are_sealed_sorted_and_replay_exact() -> None:
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
    policy = build_tushare_endpoint_policy(
        created_at=NOW,
        endpoint_plans=[income, daily],
    )

    assert validate_endpoint_execution_plan(daily) == daily
    assert validate_tushare_endpoint_policy(policy) == policy
    assert [row["api_name"] for row in policy["endpoint_plans"]] == [
        "daily_basic",
        "income_vip",
    ]
    assert policy["authority"]["provider"] is False
    assert policy["research_only"] is True
    assert policy["production"] is False


def test_ordered_partition_keyset_is_semantic_and_not_resorted() -> None:
    plan = build_plan(
        ordered_expected_partition_keyset=["period=20260630", "period=20260331"],
        planned_terminal_request_count=2,
        planned_max_network_attempts=2,
    )
    assert plan["ordered_expected_partition_keyset"] == [
        "period=20260630",
        "period=20260331",
    ]
    assert validate_endpoint_execution_plan(plan) == plan


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
        {"document_observed_at": "2026-08-09T08:00:01Z"},
    ],
)
def test_plan_rejects_unsafe_or_unclosed_inputs(overrides: dict[str, Any]) -> None:
    with pytest.raises(TushareContractError):
        build_plan(**overrides)


def test_separate_permission_has_zero_request_topology() -> None:
    plan = build_plan(
        api_name="stk_mins",
        lane="DIAGNOSTIC",
        permission_class="SEPARATE",
        fixed_params={},
        partition_dimensions=[],
        ordered_expected_partition_keyset=[],
        planned_terminal_request_count=0,
        planned_max_network_attempts=0,
    )
    assert plan["permission_class"] == "SEPARATE"
    assert plan["ordered_expected_partition_keyset"] == []
    assert validate_endpoint_execution_plan(plan) == plan


def test_resealed_plan_forgery_is_rejected() -> None:
    plan = build_plan()
    forged = copy.deepcopy(plan)
    forged["documented_row_limit"] = 5999
    with pytest.raises(TushareContractError):
        validate_endpoint_execution_plan(forged)
