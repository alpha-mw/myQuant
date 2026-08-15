from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_investor.market.tushare import build_endpoint_execution_plan
from quant_investor.market.fundamental_provider_contract import frame_fingerprint
from quant_investor.market.fundamental_provider_evidence._comparison import (
    build_fundamental_comparison_policy,
)
from quant_investor.market.tushare._core import (
    FORBIDDEN_VERSION_FIELDS,
    contract_sha256,
)
from quant_investor.market.tushare.fundamental import (
    FundamentalAcquisitionError,
    SOURCE_ENDPOINTS,
    acquire_official_fundamental_partitions,
    build_fundamental_execution_closure,
    build_fundamental_request_plan,
    build_official_partition_plan,
    replay_official_partition_requests,
    validate_official_partition_plan,
    validate_official_partition_request_receipt,
)

NOW = "2026-08-14T00:30:00Z"
CUTOFF = "2026-08-07T23:59:59Z"


def _exact_ref(name: str) -> dict[str, str]:
    kind = "market.fundamental.external_evidence"
    digest = hashlib.sha256(name.encode()).hexdigest()
    return {
        "artifact_id": name,
        "available_at": CUTOFF,
        "byte_sha256": digest,
        "contract_sha256": contract_sha256(kind),
        "cutoff": CUTOFF,
        "kind": kind,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": digest,
    }


def _sessions() -> list[str]:
    return [value.strftime("%Y%m%d") for value in pd.bdate_range("20210807", "20260807")]


def _periods() -> list[str]:
    return [value.strftime("%Y%m%d") for value in pd.date_range("20190930", "20260630", freq="QE")]


def _execution() -> dict[str, Any]:
    sessions = _sessions()
    periods = _periods()
    endpoint_plans: dict[str, dict[str, Any]] = {}
    for table, endpoint in SOURCE_ENDPOINTS.items():
        dimension = "trade_date" if table == "daily_basic" else "period"
        values = sessions if table == "daily_basic" else periods
        fields = ["ts_code", "trade_date" if table == "daily_basic" else "end_date"]
        endpoint_plans[table] = build_endpoint_execution_plan(
            api_name=endpoint,
            lane="FUNDAMENTAL",
            permission_class="POINTS",
            official_document_url="https://tushare.pro/document/2?doc_id=1",
            official_document_id=f"tushare.{endpoint}",
            document_observed_at=NOW,
            documented_min_points=2000,
            strict_decimal_decode=True,
            expected_fields=fields,
            fixed_params={},
            partition_dimensions=[dimension],
            ordered_expected_partition_keyset=[f"{dimension}={value}" for value in values],
            documented_row_limit=6000,
            max_attempts=1,
            retry_schedule=[0],
            empty_partition_rule="BASELINE_IDENTITY_EMPTY",
            completeness_proof="EXACT_PARTITION_AND_COUNT",
            limit_hit_action="BLOCK",
            planned_terminal_request_count=len(values),
            planned_max_network_attempts=len(values),
            created_at=NOW,
        )
    plan = build_fundamental_request_plan(
        as_of="20260807",
        pit_cutoff=CUTOFF,
        symbols=["000001.SZ", "600000.SH"],
        canonical_open_sessions=sessions,
        market_scope_ref=_exact_ref("market-scope"),
        market_calendar_ref=_exact_ref("market-calendar"),
        baseline_provider_manifest_ref=_exact_ref("baseline-provider"),
        baseline_network_attempts=34_000,
        baseline_empty_partition_keyset=[],
        endpoint_plans=endpoint_plans,
        max_attempts_per_partition=1,
        implementation_sha256="c" * 64,
        created_at=NOW,
    )
    return build_fundamental_execution_closure(
        plan=plan,
        endpoint_plans=endpoint_plans,
        created_at=NOW,
    )


def _probes() -> list[dict[str, Any]]:
    cases = {
        "BALANCESHEET_COMPANY_TYPE_LIMIT",
        "BALANCESHEET_EIGHT_DAY_1_COMPLETE",
        "BALANCESHEET_EIGHT_DAY_2_COMPLETE",
        "BALANCESHEET_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
        "BALANCESHEET_MONTH_COMPLETE",
        "BALANCESHEET_Q2_FOUR_DAY_1_COMPLETE",
        "BALANCESHEET_Q2_FOUR_DAY_2_COMPLETE",
        "BALANCESHEET_Q3_FOUR_DAY_1_COMPLETE",
        "BALANCESHEET_Q3_FOUR_DAY_2_COMPLETE",
        "BALANCESHEET_UNFILTERED_PHYSICAL_CLASSIFICATION_COMPLETE",
        "CASHFLOW_COMPANY_TYPE_LIMIT",
        "CASHFLOW_DAY_COMPLETE",
        "CASHFLOW_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
        "CASHFLOW_MONTH_LIMIT",
        "CASHFLOW_UNFILTERED_PHYSICAL_CLASSIFICATION_COMPLETE",
        "FINA_INDICATOR_20191231_COMPLETE",
        "FINA_INDICATOR_20230331_COMPLETE",
        "FINA_INDICATOR_ANN_DATE_COMPLETE",
        "INCOME_COMPANY_TYPE_COMPLETE",
        "INCOME_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
        "INCOME_UNFILTERED_PHYSICAL_CLASSIFICATION_COMPLETE",
    }
    endpoint_by_prefix = {
        "BALANCESHEET": "balancesheet_vip",
        "CASHFLOW": "cashflow_vip",
        "FINA_INDICATOR": "fina_indicator_vip",
        "INCOME": "income_vip",
    }
    rows = []
    for index, case_id in enumerate(sorted(cases), start=1):
        prefix = next(prefix for prefix in endpoint_by_prefix if case_id.startswith(prefix))
        rows.append(
            {
                "api_name": endpoint_by_prefix[prefix],
                "case_id": case_id,
                "expected_fields_match": True,
                "has_more": False,
                "item_count": 1,
                "observed_at": NOW,
                "params_sha256": f"{index:064x}",
                "response_body_sha256": f"{index + 100:064x}",
            }
        )
    return rows


def test_current_official_plan_is_semantic_compact_and_replayable() -> None:
    execution = _execution()
    probes = _probes()
    plan = build_official_partition_plan(
        source_execution_closure=execution,
        probe_observations=probes,
        document_observed_at=NOW,
        created_at=NOW,
    )

    assert (
        validate_official_partition_plan(
            plan,
            source_execution_closure=execution,
            probe_observations=probes,
        )
        == plan
    )
    requests = replay_official_partition_requests(
        plan,
        source_execution_closure=execution,
    )
    assert plan["kind"] == "market.tushare.fundamental_official_partition_plan"
    assert len(plan["contract_sha256"]) == 64
    assert FORBIDDEN_VERSION_FIELDS.isdisjoint(plan)
    assert "request_rows" not in plan
    assert plan["request_schedule"]["planned_request_count"] == len(requests)
    assert plan["planned_terminal_request_count"] == len(requests)
    assert len({row["request_key"] for row in requests}) == len(requests)


def test_official_plan_rejects_unknown_fields_and_schedule_forgery() -> None:
    execution = _execution()
    probes = _probes()
    plan = build_official_partition_plan(
        source_execution_closure=execution,
        probe_observations=probes,
        document_observed_at=NOW,
        created_at=NOW,
    )
    forged = deepcopy(plan)
    forged["request_schedule"]["planned_request_count"] -= 1
    with pytest.raises(FundamentalAcquisitionError):
        validate_official_partition_plan(
            forged,
            source_execution_closure=execution,
            probe_observations=probes,
        )


def _frozen_comparison_policy(execution: dict[str, Any]) -> dict[str, Any]:
    source_plan = execution["request_plan"]
    table_policies: dict[str, dict[str, Any]] = {}
    windows: dict[str, dict[str, Any]] = {}
    for table, endpoint in execution["endpoint_plans"].items():
        fields = list(endpoint["expected_fields"])
        date_column = "trade_date" if table == "daily_basic" else "end_date"
        table_policies[table] = {
            "baseline_source_only_columns": [],
            "baseline_source_only_reason": None,
            "baseline_source_schema_evidence_ref": None,
            "canonical_key_columns": ["ts_code", date_column],
            "column_rows": [
                {
                    "column": field,
                    "kind": "DATE" if field == date_column else "TEXT",
                }
                for field in fields
            ],
            "table": table,
            "winner_completeness_columns": [],
            "winner_implementation_sha256": "9" * 64,
            "winner_order_columns": ["ts_code", date_column],
            "winner_rule": "ASCII_CANONICAL_LAST",
        }
        windows[table] = {
            "date_column": date_column,
            "end_date": source_plan["as_of"],
            "start_date": (
                source_plan["daily_start"]
                if table == "daily_basic"
                else source_plan["financial_start"]
            ),
            "table": table,
        }
    return build_fundamental_comparison_policy(
        table_policies=table_policies,
        comparison_windows=windows,
        created_at=NOW,
    )


def test_official_acquisition_failure_is_one_call_stable_and_does_not_emit_frozen_policy(
    tmp_path: Path,
) -> None:
    execution = _execution()
    probes = _probes()
    plan = build_official_partition_plan(
        source_execution_closure=execution,
        probe_observations=probes,
        document_observed_at=NOW,
        created_at=NOW,
    )
    baseline = {
        table: pd.DataFrame(columns=endpoint["expected_fields"])
        for table, endpoint in execution["endpoint_plans"].items()
    }
    fingerprints = {table: frame_fingerprint(frame) for table, frame in baseline.items()}

    class OfflineFailure:
        calls = 0

        def request(self, **_kwargs: Any) -> Any:
            self.calls += 1
            raise RuntimeError("offline injected failure")

    client = OfflineFailure()
    checkpoint = tmp_path / "official-checkpoint"
    result = acquire_official_fundamental_partitions(
        official_plan=plan,
        source_execution_closure=execution,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=_frozen_comparison_policy(execution),
        client=client,
        captured_at=NOW,
        checkpoint_root=checkpoint,
    )

    assert result["status"] == "ACQUISITION_BLOCKED"
    assert result["transport_calls"] == client.calls == 1
    assert len(result["physical_receipts"]) == 1
    receipt = result["physical_receipts"][0]
    assert receipt["kind"] == "market.tushare.fundamental_official_partition_receipt"
    assert "version" not in receipt
    assert (
        validate_official_partition_request_receipt(
            receipt,
            official_plan=plan,
            source_execution_closure=execution,
            probe_observations=probes,
        )
        == receipt
    )
    forged_receipt = {**receipt, "unexpected": True}
    with pytest.raises(FundamentalAcquisitionError):
        validate_official_partition_request_receipt(
            forged_receipt,
            official_plan=plan,
            source_execution_closure=execution,
            probe_observations=probes,
        )
    bundle_bytes = (checkpoint / "execution_bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    assert set(bundle) == {
        "baseline_table_fingerprints",
        "captured_at",
        "comparison_policy_sha256",
        "official_plan_ref",
        "source_execution_closure_ref",
    }
    assert b'"version"' not in bundle_bytes
    assert b'"schema_version"' not in bundle_bytes
