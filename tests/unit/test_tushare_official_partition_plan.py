from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from quant_investor.intelligence_v2._core import seal
from quant_investor.intelligence_v2.sources.tushare import build_endpoint_execution_plan
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    FundamentalV4ContractError,
    build_fundamental_execution_closure_v4,
    build_fundamental_request_plan_v4,
    build_official_partition_execution_plan,
    validate_fundamental_execution_closure_v4,
    validate_official_partition_execution_plan,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.models import (
    SOURCE_ENDPOINTS,
)

NOW = "2026-08-10T00:30:00Z"
CUTOFF = "2026-08-07T23:59:59Z"


def _ref(name: str) -> dict[str, str]:
    return {
        "artifact_id": name,
        "artifact_version": f"{name}.v1",
        "available_at": CUTOFF,
        "byte_sha256": "a" * 64,
        "cutoff": CUTOFF,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": "b" * 64,
    }


def _sessions() -> list[str]:
    return [value.strftime("%Y%m%d") for value in pd.bdate_range("20210807", "20260807")]


def _periods() -> list[str]:
    return [value.strftime("%Y%m%d") for value in pd.date_range("20190930", "20260630", freq="QE")]


def _source_execution() -> dict[str, Any]:
    sessions = _sessions()
    periods = _periods()
    endpoint_plans = {}
    for table, endpoint in SOURCE_ENDPOINTS.items():
        dimension = "trade_date" if table == "daily_basic" else "period"
        values = sessions if table == "daily_basic" else periods
        keyset = [f"{dimension}={value}" for value in values]
        endpoint_plans[table] = build_endpoint_execution_plan(
            api_name=endpoint,
            lane="FUNDAMENTAL",
            permission_class="POINTS",
            official_document_url="https://tushare.pro/document/2?doc_id=1",
            official_document_id=f"tushare.{endpoint}",
            document_observed_at="2026-08-09T00:00:00Z",
            documented_min_points=2000,
            strict_decimal_decode=True,
            expected_fields=["ts_code", "trade_date" if table == "daily_basic" else "end_date"],
            fixed_params={},
            partition_dimensions=[dimension],
            ordered_expected_partition_keyset=keyset,
            documented_row_limit=6000,
            max_attempts=2,
            retry_schedule=[0, 0],
            empty_partition_rule="BASELINE_IDENTITY_EMPTY",
            completeness_proof="EXACT_PARTITION_AND_COUNT",
            limit_hit_action="BLOCK",
            planned_terminal_request_count=len(keyset),
            planned_max_network_attempts=len(keyset) * 2,
            created_at="2026-08-09T00:00:00Z",
        )
    plan = build_fundamental_request_plan_v4(
        as_of="20260807",
        pit_cutoff=CUTOFF,
        symbols=["000001.SZ", "600000.SH"],
        canonical_open_sessions=sessions,
        market_scope_ref=_ref("market-scope"),
        market_calendar_ref=_ref("market-calendar"),
        baseline_provider_manifest_ref=_ref("baseline-provider"),
        baseline_network_attempts=33012,
        baseline_empty_partition_keyset=[],
        endpoint_plans=endpoint_plans,
        max_attempts_per_partition=2,
        implementation_sha256="c" * 64,
        created_at="2026-08-09T00:00:00Z",
    )
    return build_fundamental_execution_closure_v4(
        plan=plan,
        endpoint_plans=endpoint_plans,
        created_at="2026-08-09T00:00:00Z",
    )


def _probes() -> list[dict[str, Any]]:
    values = [
        ("BALANCESHEET_COMPANY_TYPE_LIMIT", "balancesheet_vip", 7000, True),
        ("BALANCESHEET_MONTH_COMPLETE", "balancesheet_vip", 6963, False),
        ("CASHFLOW_COMPANY_TYPE_LIMIT", "cashflow_vip", 6400, True),
        ("CASHFLOW_DAY_COMPLETE", "cashflow_vip", 2683, False),
        ("CASHFLOW_MONTH_LIMIT", "cashflow_vip", 6400, True),
        ("FINA_INDICATOR_20191231_COMPLETE", "fina_indicator_vip", 11780, False),
        ("FINA_INDICATOR_20230331_COMPLETE", "fina_indicator_vip", 11308, False),
        ("INCOME_COMPANY_TYPE_COMPLETE", "income_vip", 5858, False),
    ]
    return [
        {
            "api_name": api_name,
            "case_id": case_id,
            "expected_fields_match": True,
            "has_more": has_more,
            "item_count": count,
            "observed_at": NOW,
            "params_sha256": f"{index + 1:064x}",
            "response_body_sha256": f"{index + 101:064x}",
        }
        for index, (case_id, api_name, count, has_more) in enumerate(values)
    ]


def _build() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source = _source_execution()
    probes = _probes()
    plan = build_official_partition_execution_plan(
        source_execution_closure=source,
        probe_observations=probes,
        document_observed_at=NOW,
        created_at=NOW,
    )
    return source, plan, probes


def test_official_partition_plan_is_sealed_replayable_and_within_attempt_gate() -> None:
    source, plan, probes = _build()
    assert (
        validate_official_partition_execution_plan(
            plan,
            source_execution_closure=source,
            probe_observations=probes,
        )
        == plan
    )
    assert plan["max_attempts_per_partition"] == 1
    assert plan["planned_terminal_request_count"] == len(plan["request_rows"])
    assert plan["planned_max_network_attempts"] * 10 <= 33012
    assert plan["performance_gate"]["passed"] is True
    assert len({row["request_key"] for row in plan["request_rows"]}) == len(plan["request_rows"])
    assert validate_fundamental_execution_closure_v4(source) == source


def test_cashflow_general_company_date_partitions_are_gap_free() -> None:
    _source, plan, _probes_value = _build()
    rows = [
        row
        for row in plan["request_rows"]
        if row["table"] == "cashflow"
        and row["params"].get("period") == "20230331"
        and row["params"].get("comp_type") == "1"
    ]
    intervals = [
        (
            datetime.strptime(row["params"]["start_date"], "%Y%m%d").date(),
            datetime.strptime(row["params"]["end_date"], "%Y%m%d").date(),
        )
        for row in rows
    ]
    assert intervals[0][0].strftime("%Y%m%d") == "20230331"
    assert intervals[-1][1].strftime("%Y%m%d") == "20260807"
    assert all(
        left[1] + timedelta(days=1) == right[0] for left, right in zip(intervals, intervals[1:])
    )
    april = [row for row in rows if row["params"]["start_date"].startswith("202304")]
    assert len(april) == 30
    assert all(row["params"]["start_date"] == row["params"]["end_date"] for row in april)


def test_plan_uses_baseline_exact_scope_and_provider_has_more_semantics() -> None:
    _source, plan, _probes_value = _build()
    assert plan["scope_policy"] == {
        "code_change_behavior": "BASELINE_EXACT_CODES_ONLY",
        "current_subject_scope_ref": _ref("market-scope"),
        "daily_scope": "BASELINE_EXACT_PARTITION_RECONCILIATION",
        "financial_scope": "BASELINE_EXACT_PARTITION_RECONCILIATION",
    }
    fina = [row for row in plan["request_rows"] if row["table"] == "fina_indicator"]
    assert fina
    assert all(row["official_row_limit"] is None for row in fina)
    daily = [row for row in plan["request_rows"] if row["table"] == "daily_basic"]
    assert all(row["official_row_limit"] == 6000 for row in daily)
    forecast = [row for row in plan["request_rows"] if row["table"] == "forecast"]
    assert all(row["official_row_limit"] is None for row in forecast)
    assert all(row["local_max_response_items"] == 20_000 for row in plan["request_rows"])


def test_resealed_request_topology_and_probe_forgeries_are_rejected() -> None:
    source, plan, probes = _build()
    forged = dict(plan)
    forged.pop("partition_plan_id")
    forged.pop("semantic_sha256")
    forged["request_rows"] = list(forged["request_rows"][:-1])
    forged = seal(forged, identity_field="partition_plan_id")
    with pytest.raises(FundamentalV4ContractError, match="replay mismatch"):
        validate_official_partition_execution_plan(
            forged,
            source_execution_closure=source,
            probe_observations=probes,
        )

    missing_probe = probes[:-1]
    with pytest.raises(FundamentalV4ContractError, match="case set is incomplete"):
        build_official_partition_execution_plan(
            source_execution_closure=source,
            probe_observations=missing_probe,
            document_observed_at=NOW,
            created_at=NOW,
        )
