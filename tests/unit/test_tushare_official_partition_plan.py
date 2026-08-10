from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
from typing import Any

import pandas as pd
import pytest

from quant_investor.intelligence_v2._core import canonical_bytes, seal
from quant_investor.intelligence_v2.sources.tushare import build_endpoint_execution_plan
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    FundamentalV4ContractError,
    acquire_official_partition_fundamental_vip_v4,
    build_fundamental_comparison_policy,
    build_fundamental_execution_closure_v4,
    build_fundamental_request_plan_v4,
    build_official_partition_execution_plan,
    validate_fundamental_execution_closure_v4,
    validate_official_partition_request_receipt,
    validate_official_partition_execution_plan,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.models import (
    SOURCE_ENDPOINTS,
)
from quant_investor.market.fundamental_provider_contract import frame_fingerprint
from quant_investor.v17_v4_runtime.tushare_https import TushareResponse

NOW = "2026-08-10T00:30:00Z"
CUTOFF = "2026-08-07T23:59:59Z"
EXPECTED_FIELDS = {
    "balancesheet": [
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "total_liab",
        "total_assets",
        "update_flag",
    ],
    "cashflow": [
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "n_cashflow_act",
        "c_pay_acq_const_fiolta",
        "free_cashflow",
        "update_flag",
    ],
    "daily_basic": ["ts_code", "trade_date", "total_mv", "circ_mv", "pe", "pb"],
    "fina_indicator": [
        "ts_code",
        "ann_date",
        "end_date",
        "roe_dt",
        "roe",
        "roa",
        "debt_to_assets",
        "netprofit_yoy",
    ],
    "forecast": [
        "ts_code",
        "ann_date",
        "end_date",
        "type",
        "p_change_min",
        "p_change_max",
        "net_profit_min",
        "net_profit_max",
        "last_parent_net",
        "summary",
        "change_reason",
    ],
    "income": [
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "n_income",
        "n_income_attr_p",
        "update_flag",
    ],
}


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
            expected_fields=EXPECTED_FIELDS[table],
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


def _schema_ref() -> dict[str, str]:
    return {
        "artifact_id": "d" * 64,
        "artifact_version": "myquant.v17.intelligence-v2.tushare-schema-diagnostic-receipt.v1",
        "byte_sha256": "e" * 64,
        "semantic_sha256": "f" * 64,
    }


def _comparison_policy(source: dict[str, Any]) -> dict[str, Any]:
    tables: dict[str, dict[str, Any]] = {}
    date_fields = {"ann_date", "end_date", "f_ann_date", "trade_date"}
    text_fields = {"change_reason", "summary", "ts_code", "type", "update_flag"}
    for table, endpoint in source["endpoint_plans"].items():
        fields = list(endpoint["expected_fields"])
        source_only: list[str] = []
        if table == "forecast":
            source_only = ["update_flag"]
        common = [*fields]
        tables[table] = {
            "baseline_source_only_columns": source_only,
            "baseline_source_only_reason": ("ENDPOINT_SCHEMA_NOT_EXPOSED" if source_only else None),
            "baseline_source_schema_evidence_ref": _schema_ref() if source_only else None,
            "canonical_key_columns": [
                "ts_code",
                "trade_date" if table == "daily_basic" else "end_date",
            ],
            "column_rows": [
                {
                    "column": field,
                    "kind": (
                        "DATE"
                        if field in date_fields
                        else "TEXT" if field in text_fields else "DECIMAL"
                    ),
                }
                for field in common
            ],
            "table": table,
            "winner_implementation_sha256": "9" * 64,
            "winner_order_columns": [
                "ts_code",
                "trade_date" if table == "daily_basic" else "end_date",
            ],
            "winner_rule": "ASCII_CANONICAL_LAST",
        }
    return build_fundamental_comparison_policy(table_policies=tables, created_at=NOW)


def _row(request: dict[str, Any], fields: list[str], *, changed: bool = False) -> tuple[Any, ...]:
    params = request["params"]
    values: list[Any] = []
    for field in fields:
        if field == "ts_code":
            values.append(f"{request['ordinal']:06d}.SZ")
        elif field == "trade_date":
            values.append(params["trade_date"])
        elif field == "end_date":
            values.append(params["period"])
        elif field in {"ann_date", "f_ann_date"}:
            values.append(params["start_date"] if "start_date" in params else params["period"])
        elif field == "type":
            values.append("预增")
        elif field in {"summary", "change_reason"}:
            values.append("validated text")
        elif field == "update_flag":
            values.append("1")
        else:
            values.append(Decimal("0.2") if changed and request["ordinal"] == 0 else Decimal("0.1"))
    return tuple(values)


def _baseline_tables(
    plan: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    rows: dict[str, list[tuple[Any, ...]]] = {table: [] for table in SOURCE_ENDPOINTS}
    for request in plan["request_rows"]:
        fields = list(source["endpoint_plans"][request["table"]]["expected_fields"])
        row = _row(request, fields)
        if request["table"] == "forecast":
            row = (*row, "1")
        rows[request["table"]].append(row)
    result: dict[str, pd.DataFrame] = {}
    for table, values in rows.items():
        fields = list(source["endpoint_plans"][table]["expected_fields"])
        if table == "forecast":
            fields.append("update_flag")
        result[table] = pd.DataFrame(values, columns=fields)
    return result


class _OfficialClient:
    def __init__(
        self,
        *,
        plan: dict[str, Any],
        source: dict[str, Any],
        first_has_more: bool = False,
        changed: bool = False,
    ) -> None:
        self.calls = 0
        self._changed = changed
        self._first_has_more = first_has_more
        self._requests = {
            (row["endpoint"], hashlib.sha256(canonical_bytes(row["params"])).hexdigest()): row
            for row in plan["request_rows"]
        }
        self._fields = {
            value["api_name"]: list(value["expected_fields"])
            for value in source["endpoint_plans"].values()
        }

    def request(
        self,
        *,
        api_name: str,
        params: dict[str, Any],
        expected_fields: list[str],
    ) -> TushareResponse:
        self.calls += 1
        request = self._requests[(api_name, hashlib.sha256(canonical_bytes(params)).hexdigest())]
        assert expected_fields == self._fields[api_name]
        row = _row(request, expected_fields, changed=self._changed)
        return TushareResponse(
            api_name=api_name,
            request_id=f"official-{self.calls}",
            reported_count=1,
            has_more=self._first_has_more and self.calls == 1,
            fields=tuple(expected_fields),
            rows=(row,),
        )


def _adapter_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, pd.DataFrame],
    dict[str, str],
    dict[str, Any],
]:
    source, plan, probes = _build()
    baseline = _baseline_tables(plan, source)
    fingerprints = {table: frame_fingerprint(frame) for table, frame in baseline.items()}
    return source, plan, probes, baseline, fingerprints, _comparison_policy(source)


def test_official_partition_adapter_reconciles_exact_baseline() -> None:
    source, plan, probes, baseline, fingerprints, policy = _adapter_inputs()
    client = _OfficialClient(plan=plan, source=source)
    result = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=client,
        captured_at=NOW,
    )
    assert result["status"] == "COMPLETE"
    assert result["comparison"]["passed"] is True
    assert result["transport_calls"] == len(plan["request_rows"])
    assert result["receipt_network_attempts"] == len(plan["request_rows"])
    assert client.calls == len(plan["request_rows"])
    assert (
        validate_official_partition_request_receipt(
            result["physical_receipts"][0],
            official_plan=plan,
            source_execution_closure=source,
            probe_observations=probes,
        )
        == result["physical_receipts"][0]
    )


def test_official_partition_adapter_resume_uses_no_transport(tmp_path: Any) -> None:
    source, plan, probes, baseline, fingerprints, policy = _adapter_inputs()
    checkpoint = tmp_path / "official-checkpoint"
    client = _OfficialClient(plan=plan, source=source)
    first = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=client,
        captured_at=NOW,
        checkpoint_root=checkpoint,
    )

    class _NetworkForbidden:
        def request(self, **_kwargs: Any) -> TushareResponse:
            raise AssertionError("checkpoint replay attempted transport")

    second = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=_NetworkForbidden(),
        captured_at=NOW,
        checkpoint_root=checkpoint,
    )
    assert first["status"] == second["status"] == "COMPLETE"
    assert second["transport_calls"] == 0
    assert second["receipt_network_attempts"] == len(plan["request_rows"])
    assert checkpoint.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in checkpoint.rglob("*.json"))


def test_official_partition_adapter_blocks_provider_and_baseline_mismatch(
    tmp_path: Any,
) -> None:
    source, plan, probes, baseline, fingerprints, policy = _adapter_inputs()
    checkpoint = tmp_path / "blocked-checkpoint"
    incomplete_client = _OfficialClient(plan=plan, source=source, first_has_more=True)
    incomplete = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=incomplete_client,
        captured_at=NOW,
        checkpoint_root=checkpoint,
    )
    assert incomplete["status"] == "ACQUISITION_BLOCKED"
    assert incomplete["comparison"] is None
    assert incomplete["physical_receipts"][0]["blocker_codes"] == ["HAS_MORE"]
    assert incomplete["transport_calls"] == incomplete_client.calls == 1
    assert len(incomplete["physical_receipts"]) == 1

    class _NetworkForbidden:
        def request(self, **_kwargs: Any) -> TushareResponse:
            raise AssertionError("failed checkpoint attempted more transport")

    replay = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=_NetworkForbidden(),
        captured_at=NOW,
        checkpoint_root=checkpoint,
    )
    assert replay["status"] == "ACQUISITION_BLOCKED"
    assert replay["transport_calls"] == 0
    assert replay["physical_receipts"] == incomplete["physical_receipts"]

    changed = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=_OfficialClient(plan=plan, source=source, changed=True),
        captured_at=NOW,
    )
    assert changed["status"] == "RECONCILIATION_BLOCKED"
    assert changed["comparison"]["passed"] is False

    forged_fingerprints = {**fingerprints, "income": "0" * 64}
    forbidden = _OfficialClient(plan=plan, source=source)
    with pytest.raises(FundamentalV4ContractError, match="baseline frame fingerprint mismatch"):
        acquire_official_partition_fundamental_vip_v4(
            official_plan=plan,
            source_execution_closure=source,
            probe_observations=probes,
            baseline_tables=baseline,
            baseline_table_fingerprints=forged_fingerprints,
            comparison_policy=policy,
            client=forbidden,
            captured_at=NOW,
        )
    assert forbidden.calls == 0
