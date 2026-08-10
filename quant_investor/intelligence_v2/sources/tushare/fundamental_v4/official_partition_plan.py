"""Sealed replacement topology for officially supported Tushare partitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Final

from ...._core import (
    common_fields,
    content_ref,
    require_exact_keys,
    seal,
    session_date,
    sha256,
    timestamp,
    validate_seal,
)
from .models import FundamentalV4ContractError, fundamental_v4_contract
from .schedule import validate_fundamental_execution_closure_v4

OFFICIAL_PARTITION_PLAN_V1: Final = "myquant.v17.fundamental-official-partition-execution-plan.v1"

_PLAN_FIELDS = {
    "as_of",
    "authority",
    "baseline_network_attempts",
    "created_at",
    "decision_protocol",
    "document_refs",
    "frozen_v1_manifest_sha256",
    "local_max_response_items",
    "max_attempts_per_partition",
    "partition_plan_id",
    "performance_gate",
    "pit_cutoff",
    "planned_max_network_attempts",
    "planned_terminal_request_count",
    "probe_observations",
    "production",
    "research_only",
    "request_rows",
    "scope_policy",
    "semantic_sha256",
    "source_execution_closure_ref",
    "source_plan_id",
    "timestamp",
    "version",
}
_DOCUMENT_FIELDS = {
    "api_name",
    "document_observed_at",
    "official_document_id",
    "official_document_url",
    "partition_parameters",
    "row_limit_basis",
}
_PROBE_FIELDS = {
    "api_name",
    "case_id",
    "expected_fields_match",
    "has_more",
    "item_count",
    "observed_at",
    "params_sha256",
    "response_body_sha256",
}
_REQUEST_FIELDS = {
    "completeness_mode",
    "endpoint",
    "local_max_response_items",
    "official_row_limit",
    "ordinal",
    "params",
    "partition_id",
    "partition_type",
    "request_key",
    "scope_mode",
    "table",
}
_ENDPOINTS: Final = {
    "balancesheet": "balancesheet_vip",
    "cashflow": "cashflow_vip",
    "daily_basic": "daily_basic",
    "fina_indicator": "fina_indicator_vip",
    "forecast": "forecast_vip",
    "income": "income_vip",
}
_DOC_IDS: Final = {
    "balancesheet_vip": "tushare.doc.36",
    "cashflow_vip": "tushare.doc.44",
    "daily_basic": "tushare.doc.32",
    "fina_indicator_vip": "tushare.doc.79",
    "forecast_vip": "tushare.doc.45",
    "income_vip": "tushare.doc.33",
}
_DOC_URLS: Final = {
    endpoint: f"https://tushare.pro/document/2?doc_id={doc_id.rsplit('.', 1)[-1]}"
    for endpoint, doc_id in _DOC_IDS.items()
}
_STATEMENT_TABLES: Final = ("balancesheet", "cashflow", "income")
_SCOPE_MODE: Final = "BASELINE_EXACT_PARTITION_RECONCILIATION"
_COMPLETENESS_MODE: Final = "EXACT_PARAMS_HAS_MORE_FALSE_BASELINE_EQUAL"


def _documents(observed_at: str) -> list[dict[str, Any]]:
    observed = timestamp(observed_at, label="document_observed_at")
    rows: list[dict[str, Any]] = []
    for table, endpoint in sorted(_ENDPOINTS.items()):
        if table == "daily_basic":
            parameters = ["trade_date"]
            row_limit_basis = "OFFICIAL_NUMERIC_6000"
        elif table in _STATEMENT_TABLES:
            parameters = [
                "comp_type",
                "end_date",
                "period",
                "report_type",
                "start_date",
            ]
            row_limit_basis = "OFFICIAL_HAS_MORE_NO_NUMERIC_LIMIT"
        else:
            parameters = ["period"]
            row_limit_basis = "OFFICIAL_HAS_MORE_NO_NUMERIC_LIMIT"
        rows.append(
            {
                "api_name": endpoint,
                "document_observed_at": observed,
                "official_document_id": _DOC_IDS[endpoint],
                "official_document_url": _DOC_URLS[endpoint],
                "partition_parameters": parameters,
                "row_limit_basis": row_limit_basis,
            }
        )
    return rows


def _probes(values: Sequence[Mapping[str, Any]], *, created_at: str) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FundamentalV4ContractError("probe observations must be a sequence")
    rows: list[dict[str, Any]] = []
    for value in values:
        row = require_exact_keys(dict(value), _PROBE_FIELDS, label="partition probe")
        if (
            row["api_name"] not in set(_ENDPOINTS.values())
            or type(row["case_id"]) is not str
            or not row["case_id"].isascii()
            or not row["case_id"]
            or type(row["item_count"]) is not int
            or row["item_count"] < 0
            or type(row["has_more"]) is not bool
            or type(row["expected_fields_match"]) is not bool
            or row["expected_fields_match"] is not True
        ):
            raise FundamentalV4ContractError("partition probe observation is invalid")
        normalized = dict(row)
        normalized["observed_at"] = timestamp(row["observed_at"], label="observed_at")
        if normalized["observed_at"] > created_at:
            raise FundamentalV4ContractError("partition probe is future-dated")
        normalized["params_sha256"] = sha256(row["params_sha256"], label="params_sha256")
        normalized["response_body_sha256"] = sha256(
            row["response_body_sha256"], label="response_body_sha256"
        )
        rows.append(normalized)
    expected = sorted(rows, key=lambda row: row["case_id"].encode("ascii"))
    if rows != expected or len({row["case_id"] for row in rows}) != len(rows):
        raise FundamentalV4ContractError("partition probes must be case-id sorted unique")
    required = {
        "BALANCESHEET_COMPANY_TYPE_LIMIT",
        "BALANCESHEET_EIGHT_DAY_1_COMPLETE",
        "BALANCESHEET_EIGHT_DAY_2_COMPLETE",
        "BALANCESHEET_MONTH_COMPLETE",
        "CASHFLOW_COMPANY_TYPE_LIMIT",
        "CASHFLOW_DAY_COMPLETE",
        "CASHFLOW_MONTH_LIMIT",
        "FINA_INDICATOR_20191231_COMPLETE",
        "FINA_INDICATOR_20230331_COMPLETE",
        "INCOME_COMPANY_TYPE_COMPLETE",
    }
    if {row["case_id"] for row in rows} != required:
        raise FundamentalV4ContractError("partition probe case set is incomplete")
    return rows


def _hot_month(period: str) -> tuple[date, date]:
    parsed = datetime.strptime(period, "%Y%m%d").date()
    if parsed.month == 3:
        return date(parsed.year, 4, 1), date(parsed.year, 4, 30)
    if parsed.month == 6:
        return date(parsed.year, 8, 1), date(parsed.year, 8, 31)
    if parsed.month == 9:
        return date(parsed.year, 10, 1), date(parsed.year, 10, 31)
    return date(parsed.year + 1, 4, 1), date(parsed.year + 1, 4, 30)


def _company_one_intervals(period: str, *, table: str, as_of: str) -> list[tuple[date, date]]:
    start = datetime.strptime(period, "%Y%m%d").date()
    end = datetime.strptime(as_of, "%Y%m%d").date()
    hot_start, hot_end = _hot_month(period)
    hot_end = min(hot_end, end)
    rows: list[tuple[date, date]] = []
    if start < hot_start:
        rows.append((start, min(hot_start - timedelta(days=1), end)))
    if hot_start <= end:
        maximum_days = 1 if table == "cashflow" else 8
        cursor = hot_start
        while cursor <= hot_end:
            interval_end = min(cursor + timedelta(days=maximum_days - 1), hot_end)
            rows.append((cursor, interval_end))
            cursor = interval_end + timedelta(days=1)
    cursor = max(start, hot_end + timedelta(days=1))
    while cursor <= end:
        year_end = min(date(cursor.year, 12, 31), end)
        rows.append((cursor, year_end))
        cursor = year_end + timedelta(days=1)
    if not rows or rows[0][0] != start or rows[-1][1] != end:
        raise FundamentalV4ContractError("announcement partitions do not cover the window")
    for left, right in zip(rows, rows[1:]):
        if left[1] + timedelta(days=1) != right[0]:
            raise FundamentalV4ContractError("announcement partitions contain a gap")
    return rows


def _partition_id(params: Mapping[str, str]) -> str:
    order = ("period", "report_type", "comp_type", "start_date", "end_date", "trade_date")
    return "&".join(f"{key}={params[key]}" for key in order if key in params)


def _request_row(
    *,
    ordinal: int,
    table: str,
    params: Mapping[str, str],
    partition_type: str,
) -> dict[str, Any]:
    endpoint = _ENDPOINTS[table]
    row_limit = 6000 if table == "daily_basic" else None
    partition_id = _partition_id(params)
    return {
        "completeness_mode": _COMPLETENESS_MODE,
        "endpoint": endpoint,
        "local_max_response_items": 20_000,
        "official_row_limit": row_limit,
        "ordinal": ordinal,
        "params": dict(params),
        "partition_id": partition_id,
        "partition_type": partition_type,
        "request_key": f"{table}|{partition_id}",
        "scope_mode": _SCOPE_MODE,
        "table": table,
    }


def _request_rows(source_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in sorted(_ENDPOINTS):
        if table == "daily_basic":
            for trade_date in source_plan["daily_open_sessions"]:
                rows.append(
                    _request_row(
                        ordinal=len(rows),
                        table=table,
                        params={"trade_date": trade_date},
                        partition_type="TRADE_DATE",
                    )
                )
            continue
        if table in {"fina_indicator", "forecast"}:
            for period in source_plan["financial_periods"]:
                rows.append(
                    _request_row(
                        ordinal=len(rows),
                        table=table,
                        params={"period": period},
                        partition_type="PERIOD",
                    )
                )
            continue
        for period in source_plan["financial_periods"]:
            for interval_start, interval_end in _company_one_intervals(
                period,
                table=table,
                as_of=source_plan["as_of"],
            ):
                params = {
                    "period": period,
                    "report_type": "1",
                    "comp_type": "1",
                    "start_date": interval_start.strftime("%Y%m%d"),
                    "end_date": interval_end.strftime("%Y%m%d"),
                }
                rows.append(
                    _request_row(
                        ordinal=len(rows),
                        table=table,
                        params=params,
                        partition_type=(
                            "PERIOD_COMPANY_TYPE_ANNOUNCEMENT_DATE"
                            if interval_start == interval_end
                            else "PERIOD_COMPANY_TYPE_ANNOUNCEMENT_RANGE"
                        ),
                    )
                )
            for comp_type in ("2", "3", "4"):
                rows.append(
                    _request_row(
                        ordinal=len(rows),
                        table=table,
                        params={
                            "period": period,
                            "report_type": "1",
                            "comp_type": comp_type,
                        },
                        partition_type="PERIOD_COMPANY_TYPE",
                    )
                )
    if len({row["request_key"] for row in rows}) != len(rows):
        raise FundamentalV4ContractError("official partition request key is duplicated")
    return rows


@fundamental_v4_contract
def build_official_partition_execution_plan(
    *,
    source_execution_closure: Mapping[str, Any],
    probe_observations: Sequence[Mapping[str, Any]],
    document_observed_at: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the static official-parameter topology replacing the blocked v4 plan."""

    source = validate_fundamental_execution_closure_v4(source_execution_closure)
    source_plan = source["request_plan"]
    created = timestamp(created_at, label="created_at")
    if created < source["created_at"]:
        raise FundamentalV4ContractError("partition plan predates its source closure")
    documents = _documents(document_observed_at)
    if any(row["document_observed_at"] > created for row in documents):
        raise FundamentalV4ContractError("official documentation is future-dated")
    probes = _probes(probe_observations, created_at=created)
    requests = _request_rows(source_plan)
    terminal = len(requests)
    baseline_attempts = source_plan["baseline_network_attempts"]
    performance_passed = terminal * 10 <= baseline_attempts
    if not performance_passed:
        raise FundamentalV4ContractError("official partition plan exceeds performance gate")
    body = {
        **common_fields(timestamp_value=created),
        "as_of": session_date(source_plan["as_of"], label="as_of"),
        "baseline_network_attempts": baseline_attempts,
        "created_at": created,
        "document_refs": documents,
        "local_max_response_items": 20_000,
        "max_attempts_per_partition": 1,
        "performance_gate": {
            "baseline_network_attempts": baseline_attempts,
            "multiplier": 10,
            "passed": True,
            "planned_network_attempts": terminal,
        },
        "pit_cutoff": timestamp(source_plan["pit_cutoff"], label="pit_cutoff"),
        "planned_max_network_attempts": terminal,
        "planned_terminal_request_count": terminal,
        "probe_observations": probes,
        "request_rows": requests,
        "scope_policy": {
            "code_change_behavior": "BASELINE_EXACT_CODES_ONLY",
            "current_subject_scope_ref": source_plan["market_scope_ref"],
            "daily_scope": _SCOPE_MODE,
            "financial_scope": _SCOPE_MODE,
        },
        "source_execution_closure_ref": content_ref(source, identity_field="closure_id"),
        "source_plan_id": source_plan["plan_id"],
        "version": OFFICIAL_PARTITION_PLAN_V1,
    }
    return seal(body, identity_field="partition_plan_id")


@fundamental_v4_contract
def validate_official_partition_execution_plan(
    document: Mapping[str, Any],
    *,
    source_execution_closure: Mapping[str, Any],
    probe_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="partition_plan_id")
    require_exact_keys(value, _PLAN_FIELDS, label="official partition execution plan")
    if value.get("version") != OFFICIAL_PARTITION_PLAN_V1:
        raise FundamentalV4ContractError("official partition plan version mismatch")
    expected = build_official_partition_execution_plan(
        source_execution_closure=source_execution_closure,
        probe_observations=probe_observations,
        document_observed_at=value["document_refs"][0]["document_observed_at"],
        created_at=value["created_at"],
    )
    if value != expected:
        raise FundamentalV4ContractError("official partition plan replay mismatch")
    for row in value["document_refs"]:
        require_exact_keys(row, _DOCUMENT_FIELDS, label="official document ref")
    for row in value["request_rows"]:
        require_exact_keys(row, _REQUEST_FIELDS, label="official request row")
    return value


__all__ = [
    "OFFICIAL_PARTITION_PLAN_V1",
    "build_official_partition_execution_plan",
    "validate_official_partition_execution_plan",
]
