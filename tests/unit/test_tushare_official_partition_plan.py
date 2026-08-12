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
    build_official_partition_execution_plan_v2,
    build_official_partition_execution_plan_v3,
    build_official_partition_execution_plan_v4,
    build_official_partition_execution_plan_v5,
    replay_official_partition_request_rows,
    validate_fundamental_execution_closure_v4,
    validate_official_partition_request_receipt,
    validate_official_partition_execution_plan,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.models import (
    SOURCE_ENDPOINTS,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    official_partition_acquisition as official_acquisition,
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
        baseline_network_attempts=34000,
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
        ("BALANCESHEET_EIGHT_DAY_1_COMPLETE", "balancesheet_vip", 1635, False),
        ("BALANCESHEET_EIGHT_DAY_2_COMPLETE", "balancesheet_vip", 5385, False),
        ("BALANCESHEET_MONTH_COMPLETE", "balancesheet_vip", 6963, False),
        ("BALANCESHEET_Q2_FOUR_DAY_1_COMPLETE", "balancesheet_vip", 2584, False),
        ("BALANCESHEET_Q2_FOUR_DAY_2_COMPLETE", "balancesheet_vip", 4845, False),
        ("BALANCESHEET_Q3_FOUR_DAY_1_COMPLETE", "balancesheet_vip", 4138, False),
        ("BALANCESHEET_Q3_FOUR_DAY_2_COMPLETE", "balancesheet_vip", 3876, False),
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


def _probes_v2() -> list[dict[str, Any]]:
    values = _probes()
    values.append(
        {
            "api_name": "fina_indicator_vip",
            "case_id": "FINA_INDICATOR_ANN_DATE_COMPLETE",
            "expected_fields_match": True,
            "has_more": False,
            "item_count": 3098,
            "observed_at": NOW,
            "params_sha256": "f" * 64,
            "response_body_sha256": "e" * 64,
        }
    )
    return sorted(values, key=lambda row: row["case_id"].encode("ascii"))


def _probes_v3() -> list[dict[str, Any]]:
    values = _probes_v2()
    values.extend(
        [
            {
                "api_name": "balancesheet_vip",
                "case_id": "BALANCESHEET_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
                "expected_fields_match": True,
                "has_more": False,
                "item_count": 1730,
                "observed_at": NOW,
                "params_sha256": "1" * 64,
                "response_body_sha256": "2" * 64,
            },
            {
                "api_name": "cashflow_vip",
                "case_id": "CASHFLOW_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
                "expected_fields_match": True,
                "has_more": False,
                "item_count": 2594,
                "observed_at": NOW,
                "params_sha256": "3" * 64,
                "response_body_sha256": "4" * 64,
            },
            {
                "api_name": "income_vip",
                "case_id": "INCOME_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
                "expected_fields_match": True,
                "has_more": False,
                "item_count": 2143,
                "observed_at": NOW,
                "params_sha256": "5" * 64,
                "response_body_sha256": "6" * 64,
            },
        ]
    )
    return sorted(values, key=lambda row: row["case_id"].encode("ascii"))


def _probes_v4() -> list[dict[str, Any]]:
    values = _probes_v3()
    for table, api_name in (
        ("BALANCESHEET", "balancesheet_vip"),
        ("CASHFLOW", "cashflow_vip"),
        ("INCOME", "income_vip"),
    ):
        for comp_type in ("1", "2", "3", "4"):
            index = len(values) + 1
            values.append(
                {
                    "api_name": api_name,
                    "case_id": (f"{table}_EXACT_ANN_DATE_REPORT_1_COMP_{comp_type}_COMPLETE"),
                    "expected_fields_match": True,
                    "has_more": False,
                    "item_count": 1,
                    "observed_at": NOW,
                    "params_sha256": f"{index:064x}",
                    "response_body_sha256": f"{index + 100:064x}",
                }
            )
    return sorted(values, key=lambda row: row["case_id"].encode("ascii"))


def _probes_v5() -> list[dict[str, Any]]:
    values = _probes_v3()
    for index, (table, api_name, item_count) in enumerate(
        (
            ("BALANCESHEET", "balancesheet_vip", 1057),
            ("CASHFLOW", "cashflow_vip", 1710),
            ("INCOME", "income_vip", 1375),
        ),
        start=1,
    ):
        values.append(
            {
                "api_name": api_name,
                "case_id": f"{table}_UNFILTERED_PHYSICAL_CLASSIFICATION_COMPLETE",
                "expected_fields_match": True,
                "has_more": False,
                "item_count": item_count,
                "observed_at": NOW,
                "params_sha256": f"{index + 200:064x}",
                "response_body_sha256": f"{index + 300:064x}",
            }
        )
    return sorted(values, key=lambda row: row["case_id"].encode("ascii"))


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


def _build_v2() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source = _source_execution()
    probes = _probes_v2()
    plan = build_official_partition_execution_plan_v2(
        source_execution_closure=source,
        probe_observations=probes,
        document_observed_at=NOW,
        created_at=NOW,
    )
    return source, plan, probes


def _build_v3() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source = _source_execution()
    probes = _probes_v3()
    plan = build_official_partition_execution_plan_v3(
        source_execution_closure=source,
        probe_observations=probes,
        document_observed_at=NOW,
        created_at=NOW,
    )
    return source, plan, probes


def _build_v4() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source = _source_execution()
    probes = _probes_v4()
    plan = build_official_partition_execution_plan_v4(
        source_execution_closure=source,
        probe_observations=probes,
        document_observed_at=NOW,
        created_at=NOW,
    )
    return source, plan, probes


def _build_v5() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source = _source_execution()
    probes = _probes_v5()
    plan = build_official_partition_execution_plan_v5(
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
    assert (
        plan["planned_max_network_attempts"] * 10
        <= source["request_plan"]["baseline_network_attempts"]
    )
    assert plan["performance_gate"]["passed"] is True
    assert len({row["request_key"] for row in plan["request_rows"]}) == len(plan["request_rows"])
    assert validate_fundamental_execution_closure_v4(source) == source


def test_v2_exhausts_every_announcement_date_without_ratio_gate() -> None:
    source, plan, probes = _build_v2()
    assert (
        validate_official_partition_execution_plan(
            plan,
            source_execution_closure=source,
            probe_observations=probes,
        )
        == plan
    )
    proof = plan["announcement_date_keyset_proof"]
    first = datetime.strptime(proof["start_date"], "%Y%m%d").date()
    last = datetime.strptime(proof["end_date"], "%Y%m%d").date()
    assert proof["domain_basis"] == "ALL_CALENDAR_DATES_INCLUSIVE"
    assert proof["date_count"] == (last - first).days + 1
    assert plan["performance_gate"] == {
        "baseline_network_attempts": source["request_plan"]["baseline_network_attempts"],
        "mode": "OWNER_AUTHORIZED_EXACT_ANN_DATE_NO_RATIO_CAP",
        "multiplier": None,
        "passed": True,
        "planned_network_attempts": plan["planned_terminal_request_count"],
    }
    assert (
        plan["planned_terminal_request_count"] * 10
        > source["request_plan"]["baseline_network_attempts"]
    )
    rows = [row for row in plan["request_rows"] if row["table"] == "fina_indicator"]
    assert len(rows) == proof["date_count"]
    assert [row["params"]["ann_date"] for row in rows] == [
        (first + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(proof["date_count"])
    ]
    assert all(
        row["params"]
        == {
            "ann_date": row["params"]["ann_date"],
            "end_date": "20260807",
            "start_date": "20190807",
        }
        for row in rows
    )
    assert {
        tuple(document["partition_parameters"])
        for document in plan["document_refs"]
        if document["api_name"] == "fina_indicator_vip"
    } == {("ann_date", "end_date", "start_date")}


def test_v2_rejects_a_resealed_announcement_date_gap() -> None:
    source, plan, probes = _build_v2()
    forged = dict(plan)
    forged.pop("partition_plan_id")
    forged.pop("semantic_sha256")
    rows = list(forged["request_rows"])
    removed = next(
        index
        for index, row in enumerate(rows)
        if row["table"] == "fina_indicator" and row["params"]["ann_date"] == "20220428"
    )
    rows.pop(removed)
    forged["request_rows"] = rows
    forged = seal(forged, identity_field="partition_plan_id")
    with pytest.raises(FundamentalV4ContractError, match="replay mismatch"):
        validate_official_partition_execution_plan(
            forged,
            source_execution_closure=source,
            probe_observations=probes,
        )


def test_v3_exhausts_statement_announcement_dates_without_empirical_ranges() -> None:
    source, plan, probes = _build_v3()
    assert (
        validate_official_partition_execution_plan(
            plan,
            source_execution_closure=source,
            probe_observations=probes,
        )
        == plan
    )
    assert plan["performance_gate"]["mode"] == (
        "OWNER_AUTHORIZED_EXACT_ANN_DATE_FULL_STATEMENT_KEYSET_NO_RATIO_CAP"
    )
    proofs = plan["announcement_date_keyset_proofs"]
    assert [row["table"] for row in proofs] == [
        "balancesheet",
        "cashflow",
        "fina_indicator",
        "income",
    ]
    assert len({row["ordered_keyset_sha256"] for row in proofs}) == 1
    date_count = proofs[0]["date_count"]
    assert all(row["date_count"] == date_count for row in proofs)
    for table in ("balancesheet", "cashflow", "income"):
        rows = [row for row in plan["request_rows"] if row["table"] == table]
        assert len(rows) == date_count
        assert all(
            row["partition_type"] == "EXACT_ANNOUNCEMENT_DATE_ALL_PERIODS"
            and row["params"]
            == {
                "end_date": row["params"]["start_date"],
                "start_date": row["params"]["start_date"],
            }
            and "period" not in row["params"]
            and "comp_type" not in row["params"]
            and "report_type" not in row["params"]
            for row in rows
        )
    assert not any("ANNOUNCEMENT_RANGE" in row["partition_type"] for row in plan["request_rows"])


def test_v3_rejects_resealed_statement_date_gap_and_incomplete_probe() -> None:
    source, plan, probes = _build_v3()
    forged = dict(plan)
    forged.pop("partition_plan_id")
    forged.pop("semantic_sha256")
    forged["request_rows"] = [
        row
        for row in forged["request_rows"]
        if not (row["table"] == "balancesheet" and row["params"].get("start_date") == "20220428")
    ]
    forged = seal(forged, identity_field="partition_plan_id")
    with pytest.raises(FundamentalV4ContractError, match="replay mismatch"):
        validate_official_partition_execution_plan(
            forged,
            source_execution_closure=source,
            probe_observations=probes,
        )
    incomplete = [
        (
            {**row, "has_more": True}
            if row["case_id"] == "INCOME_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE"
            else row
        )
        for row in probes
    ]
    with pytest.raises(FundamentalV4ContractError, match="probe does not prove completeness"):
        build_official_partition_execution_plan_v3(
            source_execution_closure=source,
            probe_observations=incomplete,
            document_observed_at=NOW,
            created_at=NOW,
        )


def test_v3_statement_leaf_enforces_exact_announcement_date() -> None:
    _source, plan, _probes_value = _build_v3()
    request = next(
        row
        for row in plan["request_rows"]
        if row["table"] == "balancesheet" and row["params"]["start_date"] == "20220428"
    )
    fields = EXPECTED_FIELDS["balancesheet"]
    valid = (
        "000001.SZ",
        "20220428",
        "20220428",
        "20211231",
        Decimal("1"),
        Decimal("2"),
        "1",
    )
    wrong_date = (valid[0], "20220429", *valid[2:])
    assert official_acquisition._baseline_partition_key(request) == "ann_date=20220428"
    assert (
        official_acquisition._response_scope_blockers(
            request=request,
            fields=fields,
            rows=(valid,),
        )
        == []
    )
    assert official_acquisition._response_scope_blockers(
        request=request,
        fields=fields,
        rows=(wrong_date,),
    ) == ["SCOPE_MISMATCH"]


def test_v4_compact_schedule_replays_complete_statement_dimensions() -> None:
    source, plan, probes = _build_v4()
    assert "request_rows" not in plan
    assert len(canonical_bytes(plan)) < 8 * 1024 * 1024
    assert (
        validate_official_partition_execution_plan(
            plan,
            source_execution_closure=source,
            probe_observations=probes,
        )
        == plan
    )
    rows = replay_official_partition_request_rows(
        plan,
        source_execution_closure=source,
    )
    statement_rows = [row for row in rows if row["table"] in {"balancesheet", "cashflow", "income"}]
    date_count = plan["announcement_date_keyset_proofs"][0]["date_count"]
    expected_count = (
        date_count * 3 * 4
        + date_count
        + len(source["request_plan"]["daily_open_sessions"])
        + len(source["request_plan"]["financial_periods"])
    )
    assert len(statement_rows) == date_count * 3 * 4
    assert len(rows) == plan["planned_terminal_request_count"] == expected_count
    assert plan["request_schedule"] == {
        "generator_version": "EXACT_ANN_DATE_REPORT_TYPE_COMP_TYPE_V1",
        "planned_request_count": expected_count,
        "request_rows_sha256": plan["request_schedule"]["request_rows_sha256"],
        "statement_comp_types": ["1", "2", "3", "4"],
        "statement_report_type": "1",
    }
    assert all(
        row["params"]["report_type"] == "1"
        and row["params"]["comp_type"] in {"1", "2", "3", "4"}
        and row["params"]["start_date"] == row["params"]["end_date"]
        and "period" not in row["params"]
        and row["exact_duplicate_mode"] == "REJECT_EXACT_DUPLICATES"
        for row in statement_rows
    )
    first = next(
        row
        for row in statement_rows
        if row["table"] == "balancesheet"
        and row["params"]["start_date"] == "20190819"
        and row["params"]["comp_type"] == "1"
    )
    assert official_acquisition._baseline_partition_key(first) == (
        "ann_date=20190819&report_type=1&comp_type=1"
    )


def test_v4_rejects_resealed_schedule_and_keeps_v3_bytes_replayable() -> None:
    source, plan, probes = _build_v4()
    forged = dict(plan)
    forged.pop("partition_plan_id")
    forged.pop("semantic_sha256")
    forged["request_schedule"] = {
        **forged["request_schedule"],
        "statement_comp_types": ["1", "2", "4"],
    }
    forged = seal(forged, identity_field="partition_plan_id")
    with pytest.raises(FundamentalV4ContractError, match="replay mismatch"):
        validate_official_partition_execution_plan(
            forged,
            source_execution_closure=source,
            probe_observations=probes,
        )
    source_v3, plan_v3, probes_v3 = _build_v3()
    assert (
        validate_official_partition_execution_plan(
            plan_v3,
            source_execution_closure=source_v3,
            probe_observations=probes_v3,
        )
        == plan_v3
    )
    with pytest.raises(FundamentalV4ContractError, match="case set is incomplete"):
        build_official_partition_execution_plan_v4(
            source_execution_closure=source,
            probe_observations=_probes_v3(),
            document_observed_at=NOW,
            created_at=NOW,
        )
    incomplete = [
        (
            {**row, "item_count": 0}
            if row["case_id"] == "BALANCESHEET_EXACT_ANN_DATE_REPORT_1_COMP_4_COMPLETE"
            else row
        )
        for row in probes
    ]
    with pytest.raises(FundamentalV4ContractError, match="does not prove completeness"):
        build_official_partition_execution_plan_v4(
            source_execution_closure=source,
            probe_observations=incomplete,
            document_observed_at=NOW,
            created_at=NOW,
        )


def test_v4_compact_schedule_is_consumed_without_expanding_the_artifact() -> None:
    source, plan, probes = _build_v4()
    _source_v1, plan_v1, _probes_v1, baseline, fingerprints, policy = _adapter_inputs()
    calls: list[tuple[str, dict[str, Any], list[str]]] = []

    class _FirstRequestBlocked:
        def request(
            self,
            *,
            api_name: str,
            params: dict[str, Any],
            expected_fields: list[str],
        ) -> TushareResponse:
            calls.append((api_name, params, expected_fields))
            request = replay_official_partition_request_rows(
                plan,
                source_execution_closure=source,
            )[0]
            row = _row(request, expected_fields)
            return TushareResponse(
                api_name=api_name,
                request_id="v4-first-blocked",
                reported_count=1,
                has_more=True,
                fields=tuple(expected_fields),
                rows=(row,),
            )

    result = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=_FirstRequestBlocked(),
        captured_at=NOW,
    )
    assert plan_v1["version"].endswith(".v1")
    assert result["status"] == "ACQUISITION_BLOCKED"
    assert result["transport_calls"] == 1
    assert result["physical_receipts"][0]["blocker_codes"] == ["HAS_MORE"]
    assert calls[0][0] == "balancesheet_vip"
    assert calls[0][1] == {
        "comp_type": "1",
        "end_date": "20190807",
        "report_type": "1",
        "start_date": "20190807",
    }


def test_v5_uses_unfiltered_dates_and_seals_physical_classification() -> None:
    source, plan, probes = _build_v5()
    assert "request_rows" not in plan
    assert (
        validate_official_partition_execution_plan(
            plan,
            source_execution_closure=source,
            probe_observations=probes,
        )
        == plan
    )
    rows = replay_official_partition_request_rows(
        plan,
        source_execution_closure=source,
    )
    statement_rows = [row for row in rows if row["table"] in {"balancesheet", "cashflow", "income"}]
    date_count = plan["announcement_date_keyset_proofs"][0]["date_count"]
    assert len(statement_rows) == date_count * 3
    assert all(
        row["params"]
        == {
            "end_date": row["params"]["start_date"],
            "start_date": row["params"]["start_date"],
        }
        for row in statement_rows
    )
    assert plan["request_schedule"] == {
        "generator_version": "EXACT_ANN_DATE_UNFILTERED_PHYSICAL_CLASSIFICATION_V1",
        "physical_statement_projection_columns": ["report_type", "comp_type"],
        "planned_request_count": len(rows),
        "request_rows_sha256": plan["request_schedule"]["request_rows_sha256"],
    }
    statement_proofs = [
        row for row in plan["announcement_date_keyset_proofs"] if row["table"] != "fina_indicator"
    ]
    assert all(
        row["physical_projection_columns"] == ["report_type", "comp_type"]
        for row in statement_proofs
    )

    with pytest.raises(FundamentalV4ContractError, match="case set is incomplete"):
        build_official_partition_execution_plan_v5(
            source_execution_closure=source,
            probe_observations=probes[:-1],
            document_observed_at=NOW,
            created_at=NOW,
        )


def test_v5_transport_projects_classification_but_accepted_raw_drops_it() -> None:
    source, plan, probes = _build_v5()
    _old_source, _old_plan, _old_probes, baseline, fingerprints, policy = _adapter_inputs()
    calls: list[tuple[str, dict[str, Any], list[str]]] = []

    class _FirstRequestBlocked:
        def request(
            self,
            *,
            api_name: str,
            params: dict[str, Any],
            expected_fields: list[str],
        ) -> TushareResponse:
            calls.append((api_name, params, expected_fields))
            request = replay_official_partition_request_rows(
                plan,
                source_execution_closure=source,
            )[0]
            row = _row(request, expected_fields)
            return TushareResponse(
                api_name=api_name,
                request_id="v5-first-blocked",
                reported_count=1,
                has_more=True,
                fields=tuple(expected_fields),
                rows=(row,),
            )

    result = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=_FirstRequestBlocked(),
        captured_at=NOW,
    )
    assert result["status"] == "ACQUISITION_BLOCKED"
    assert calls[0][0] == "balancesheet_vip"
    assert calls[0][1] == {"end_date": "20190807", "start_date": "20190807"}
    assert calls[0][2][-2:] == ["report_type", "comp_type"]


def test_accepted_projection_keeps_restatements_and_filters_forecast_ann_date() -> None:
    statement = pd.DataFrame(
        [
            ["000001.SZ", "20250829", "20250829", "20250630", 1, 2, "0", "1", "2"],
            ["000001.SZ", "20250829", "20250829", "20250630", 1, 2, "0", "5", "2"],
            ["000001.SZ", "20250829", "20250829", "20250630", 1, 2, "1", "5", "2"],
        ],
        columns=[*EXPECTED_FIELDS["balancesheet"], "report_type", "comp_type"],
    )
    accepted_statement = official_acquisition._accepted_symbol_rows(
        statement,
        table="balancesheet",
        symbol="000001.SZ",
        as_of="20260807",
        financial_start="20190807",
    )
    assert list(accepted_statement.columns) == EXPECTED_FIELDS["balancesheet"]
    assert accepted_statement["update_flag"].tolist() == ["0", "1"]

    forecast = pd.DataFrame(
        [
            ["000001.SZ", "20190806", "20191231", "预增", 1, 2, 3, 4, 5, "a", "b"],
            ["000001.SZ", "20190807", "20191231", "预增", 1, 2, 3, 4, 5, "a", "b"],
        ],
        columns=EXPECTED_FIELDS["forecast"],
    )
    accepted_forecast = official_acquisition._accepted_symbol_rows(
        forecast,
        table="forecast",
        symbol="000001.SZ",
        as_of="20260807",
        financial_start="20190807",
    )
    assert accepted_forecast["ann_date"].tolist() == ["20190807"]


def test_v2_announcement_leaf_enforces_date_and_report_period_scope() -> None:
    _source, plan, _probes_value = _build_v2()
    request = next(
        row
        for row in plan["request_rows"]
        if row["table"] == "fina_indicator" and row["params"]["ann_date"] == "20220428"
    )
    fields = EXPECTED_FIELDS["fina_indicator"]
    assert official_acquisition._baseline_partition_key(request) == "ann_date=20220428"
    valid = ("000001.SZ", "20220428", "20211231", *([Decimal("0.1")] * 5))
    wrong_announcement = ("000001.SZ", "20220429", "20211231", *([Decimal("0.1")] * 5))
    old_period = ("000001.SZ", "20220428", "20181231", *([Decimal("0.1")] * 5))
    assert (
        official_acquisition._response_scope_blockers(request=request, fields=fields, rows=(valid,))
        == []
    )
    assert official_acquisition._response_scope_blockers(
        request=request,
        fields=fields,
        rows=(wrong_announcement,),
    ) == ["SCOPE_MISMATCH"]
    assert official_acquisition._response_scope_blockers(
        request=request, fields=fields, rows=(old_period,)
    ) == ["SCOPE_MISMATCH"]


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


def test_balance_and_income_hot_partitions_are_at_most_four_days() -> None:
    _source, plan, _probes_value = _build()
    for table in ("balancesheet", "income"):
        rows = [
            row
            for row in plan["request_rows"]
            if row["table"] == table
            and row["params"].get("period") == "20200630"
            and row["params"].get("comp_type") == "1"
            and row["params"].get("start_date", "") >= "20200801"
            and row["params"].get("end_date", "") <= "20200831"
        ]
        assert rows
        for row in rows:
            start = datetime.strptime(row["params"]["start_date"], "%Y%m%d").date()
            end = datetime.strptime(row["params"]["end_date"], "%Y%m%d").date()
            assert (end - start).days + 1 <= 4


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
    assert {row["exact_duplicate_mode"] for row in fina} == {"PRESERVE_CANONICAL_MULTISET"}
    daily = [row for row in plan["request_rows"] if row["table"] == "daily_basic"]
    assert all(row["official_row_limit"] == 6000 for row in daily)
    forecast = [row for row in plan["request_rows"] if row["table"] == "forecast"]
    assert all(row["official_row_limit"] is None for row in forecast)
    assert all(row["local_max_response_items"] == 20_000 for row in plan["request_rows"])
    assert {
        row["exact_duplicate_mode"]
        for row in plan["request_rows"]
        if row["table"] != "fina_indicator"
    } == {"REJECT_EXACT_DUPLICATES"}


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
            values.append("000001.SZ")
        elif field == "trade_date":
            values.append(params["trade_date"])
        elif field == "end_date":
            values.append(params.get("period", "20181231"))
        elif field in {"ann_date", "f_ann_date"}:
            values.append(params["start_date"] if "start_date" in params else params["period"])
        elif field == "type":
            values.append("预增")
        elif field in {"summary", "change_reason"}:
            values.append("validated text")
        elif field == "update_flag":
            values.append("1")
        elif field in {"report_type", "comp_type"}:
            values.append({"report_type": "1", "comp_type": "2"}[field])
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
    from quant_investor.market.fundamental_mart import _strict_pit_cutoff

    for table, frame in result.items():
        accepted: list[pd.DataFrame] = []
        for symbol in source["request_plan"]["symbols"]:
            symbol_rows = frame.loc[frame["ts_code"] == symbol].reset_index(drop=True)
            if symbol_rows.empty:
                continue
            projected, _stats, malformed_reason = _strict_pit_cutoff(
                symbol_rows,
                table=table,
                symbol=symbol,
                as_of=source["request_plan"]["as_of"],
            )
            assert not malformed_reason
            if not projected.empty:
                accepted.append(projected)
        result[table] = (
            pd.concat(accepted, ignore_index=True) if accepted else frame.iloc[:0].copy()
        )
    return result


class _OfficialClient:
    def __init__(
        self,
        *,
        plan: dict[str, Any],
        source: dict[str, Any],
        first_has_more: bool = False,
        changed: bool = False,
        duplicate_table: str | None = None,
        include_out_of_scope: bool = False,
        include_unusable: bool = False,
    ) -> None:
        self.calls = 0
        self._changed = changed
        self._first_has_more = first_has_more
        self._duplicate_table = duplicate_table
        self._include_out_of_scope = include_out_of_scope
        self._include_unusable = include_unusable
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
        rows = [row, row] if request["table"] == self._duplicate_table else [row]
        if self._include_out_of_scope:
            rows.append(("999999.SZ", *row[1:]))
        if self._include_unusable:
            unusable = tuple(
                (
                    value
                    if field
                    in {
                        "ts_code",
                        "trade_date",
                        "ann_date",
                        "f_ann_date",
                        "end_date",
                        "type",
                        "summary",
                        "change_reason",
                        "update_flag",
                    }
                    else None
                )
                for field, value in zip(expected_fields, row)
            )
            rows.append(unusable)
        return TushareResponse(
            api_name=api_name,
            request_id=f"official-{self.calls}",
            reported_count=len(rows),
            has_more=self._first_has_more and self.calls == 1,
            fields=tuple(expected_fields),
            rows=tuple(rows),
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


def test_fina_indicator_exact_duplicates_stay_in_physical_receipts_only() -> None:
    source, plan, probes, baseline, _fingerprints, policy = _adapter_inputs()
    fingerprints = {table: frame_fingerprint(frame) for table, frame in baseline.items()}
    result = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=_OfficialClient(plan=plan, source=source, duplicate_table="fina_indicator"),
        captured_at=NOW,
    )
    assert result["status"] == "COMPLETE"
    assert result["comparison"]["passed"] is True
    assert not result["raw_tables"]["fina_indicator"].duplicated().any()
    assert any(
        receipt["accepted_count"] == 2
        for receipt in result["physical_receipts"]
        if receipt["table"] == "fina_indicator"
    )
    assert all(
        "DUPLICATE_ROWS" not in receipt["blocker_codes"]
        for receipt in result["physical_receipts"]
        if receipt["table"] == "fina_indicator"
    )


def test_official_partition_adapter_projects_target_scope_and_v3_accepted_raw() -> None:
    source, plan, probes, baseline, fingerprints, policy = _adapter_inputs()
    result = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=_OfficialClient(
            plan=plan,
            source=source,
            include_out_of_scope=True,
            include_unusable=True,
        ),
        captured_at=NOW,
    )
    assert result["status"] == "COMPLETE"
    assert result["comparison"]["passed"] is True
    assert all(
        set(frame["ts_code"].tolist()) == {"000001.SZ"} for frame in result["raw_tables"].values()
    )
    assert all(receipt["accepted_count"] == 3 for receipt in result["physical_receipts"])


def test_non_indicator_exact_duplicates_remain_incomplete() -> None:
    source, plan, probes, baseline, fingerprints, policy = _adapter_inputs()
    result = acquire_official_partition_fundamental_vip_v4(
        official_plan=plan,
        source_execution_closure=source,
        probe_observations=probes,
        baseline_tables=baseline,
        baseline_table_fingerprints=fingerprints,
        comparison_policy=policy,
        client=_OfficialClient(plan=plan, source=source, duplicate_table="balancesheet"),
        captured_at=NOW,
    )
    assert result["status"] == "ACQUISITION_BLOCKED"
    assert result["transport_calls"] == 1
    assert result["physical_receipts"][0]["blocker_codes"] == ["DUPLICATE_ROWS"]


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
