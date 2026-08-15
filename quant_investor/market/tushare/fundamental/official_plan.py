"""Current official Tushare partition topology for Fundamental shadow capture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import hashlib
from typing import Any, Final

from .._core import (
    canonical_bytes,
    common_fields,
    content_ref,
    require_exact_keys,
    seal,
    session_date,
    sha256,
    timestamp,
    validate_seal,
)
from .models import (
    OFFICIAL_PARTITION_PLAN_KIND,
    SOURCE_ENDPOINTS,
    FundamentalAcquisitionError,
    fundamental_contract,
)
from .schedule import validate_fundamental_execution_closure

_PLAN_FIELDS = {
    "announcement_date_keyset_proofs",
    "as_of",
    "authority",
    "baseline_network_attempts",
    "contract_sha256",
    "created_at",
    "document_refs",
    "kind",
    "local_max_response_items",
    "max_attempts_per_partition",
    "partition_plan_id",
    "performance_gate",
    "pit_cutoff",
    "planned_max_network_attempts",
    "planned_terminal_request_count",
    "probe_observations",
    "production",
    "request_schedule",
    "research_only",
    "scope_policy",
    "semantic_sha256",
    "source_execution_closure_ref",
    "source_plan_id",
    "timestamp",
}
_PROOF_FIELDS = {
    "date_count",
    "domain_basis",
    "end_date",
    "endpoint",
    "ordered_keyset_sha256",
    "partition_parameter",
    "physical_projection_columns",
    "pit_cutoff",
    "report_period_end",
    "report_period_start",
    "start_date",
    "table",
}
_SCHEDULE_FIELDS = {
    "physical_statement_projection_columns",
    "planned_request_count",
    "request_generator",
    "request_rows_sha256",
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
    "exact_duplicate_mode",
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
_DOC_IDS: Final = {
    "balancesheet_vip": "tushare.doc.36",
    "cashflow_vip": "tushare.doc.44",
    "daily_basic": "tushare.doc.32",
    "fina_indicator_vip": "tushare.doc.79",
    "forecast_vip": "tushare.doc.45",
    "income_vip": "tushare.doc.33",
}
_STATEMENT_TABLES: Final = ("balancesheet", "cashflow", "income")
_STATEMENT_PROJECTION_COLUMNS: Final = ("report_type", "comp_type")
_REQUEST_GENERATOR: Final = "EXACT_ANN_DATE_UNFILTERED_PHYSICAL_CLASSIFICATION"
_REQUEST_ROWS_HASH_DOMAIN: Final = b"market.tushare.fundamental.official_partition_request_rows\0"
_SCOPE_MODE: Final = "BASELINE_EXACT_PARTITION_RECONCILIATION"
_COMPLETENESS_MODE: Final = "EXACT_PARAMS_HAS_MORE_FALSE_BASELINE_EQUAL"
_EXACT_DUPLICATE_MODES: Final = {
    "fina_indicator": "PRESERVE_CANONICAL_MULTISET",
    "balancesheet": "REJECT_EXACT_DUPLICATES",
    "cashflow": "REJECT_EXACT_DUPLICATES",
    "daily_basic": "REJECT_EXACT_DUPLICATES",
    "forecast": "REJECT_EXACT_DUPLICATES",
    "income": "REJECT_EXACT_DUPLICATES",
}


def _documents(observed_at: str) -> list[dict[str, Any]]:
    observed = timestamp(observed_at, label="document_observed_at")
    rows: list[dict[str, Any]] = []
    for table, endpoint in sorted(SOURCE_ENDPOINTS.items()):
        if table == "daily_basic":
            parameters = ["trade_date"]
            row_limit_basis = "OFFICIAL_NUMERIC_6000"
        elif table in _STATEMENT_TABLES:
            parameters = ["end_date", "start_date"]
            row_limit_basis = "OFFICIAL_HAS_MORE_NO_NUMERIC_LIMIT"
        elif table == "fina_indicator":
            parameters = ["ann_date", "end_date", "start_date"]
            row_limit_basis = "OFFICIAL_HAS_MORE_NO_NUMERIC_LIMIT"
        else:
            parameters = ["period"]
            row_limit_basis = "OFFICIAL_HAS_MORE_NO_NUMERIC_LIMIT"
        document_id = _DOC_IDS[endpoint]
        rows.append(
            {
                "api_name": endpoint,
                "document_observed_at": observed,
                "official_document_id": document_id,
                "official_document_url": (
                    "https://tushare.pro/document/2?doc_id=" f"{document_id.rsplit('.', 1)[-1]}"
                ),
                "partition_parameters": parameters,
                "row_limit_basis": row_limit_basis,
            }
        )
    return rows


def _required_probe_cases() -> set[str]:
    return {
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


def _probes(
    values: Sequence[Mapping[str, Any]],
    *,
    created_at: str,
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FundamentalAcquisitionError("probe observations must be a sequence")
    rows: list[dict[str, Any]] = []
    for value in values:
        row = require_exact_keys(dict(value), _PROBE_FIELDS, label="partition probe")
        if (
            row["api_name"] not in set(SOURCE_ENDPOINTS.values())
            or type(row["case_id"]) is not str
            or not row["case_id"].isascii()
            or not row["case_id"]
            or type(row["item_count"]) is not int
            or row["item_count"] < 0
            or type(row["has_more"]) is not bool
            or row["expected_fields_match"] is not True
        ):
            raise FundamentalAcquisitionError("partition probe observation is invalid")
        normalized = dict(row)
        normalized["observed_at"] = timestamp(
            row["observed_at"],
            label="observed_at",
        )
        if normalized["observed_at"] > created_at:
            raise FundamentalAcquisitionError("partition probe is future-dated")
        normalized["params_sha256"] = sha256(
            row["params_sha256"],
            label="params_sha256",
        )
        normalized["response_body_sha256"] = sha256(
            row["response_body_sha256"],
            label="response_body_sha256",
        )
        rows.append(normalized)
    if (
        rows != sorted(rows, key=lambda row: row["case_id"].encode("ascii"))
        or len({row["case_id"] for row in rows}) != len(rows)
        or {row["case_id"] for row in rows} != _required_probe_cases()
    ):
        raise FundamentalAcquisitionError("partition probe case set is incomplete")
    completeness_cases = {
        "BALANCESHEET_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
        "BALANCESHEET_UNFILTERED_PHYSICAL_CLASSIFICATION_COMPLETE",
        "CASHFLOW_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
        "CASHFLOW_UNFILTERED_PHYSICAL_CLASSIFICATION_COMPLETE",
        "FINA_INDICATOR_ANN_DATE_COMPLETE",
        "INCOME_EXACT_ANN_DATE_ALL_PERIODS_COMPLETE",
        "INCOME_UNFILTERED_PHYSICAL_CLASSIFICATION_COMPLETE",
    }
    if any(
        row["case_id"] in completeness_cases and (row["has_more"] or row["item_count"] < 1)
        for row in rows
    ):
        raise FundamentalAcquisitionError("partition probe does not prove completeness")
    return rows


def _announcement_dates(*, start: str, end: str) -> list[str]:
    cursor = datetime.strptime(start, "%Y%m%d").date()
    terminal = datetime.strptime(end, "%Y%m%d").date()
    if cursor > terminal:
        raise FundamentalAcquisitionError("announcement date keyset is inverted")
    values: list[str] = []
    while cursor <= terminal:
        values.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    return values


def _partition_id(params: Mapping[str, str]) -> str:
    order = ("ann_date", "period", "start_date", "end_date", "trade_date")
    return "&".join(f"{key}={params[key]}" for key in order if key in params)


def _request_row(
    *,
    ordinal: int,
    table: str,
    params: Mapping[str, str],
    partition_type: str,
) -> dict[str, Any]:
    partition_id = _partition_id(params)
    return {
        "completeness_mode": _COMPLETENESS_MODE,
        "exact_duplicate_mode": _EXACT_DUPLICATE_MODES[table],
        "endpoint": SOURCE_ENDPOINTS[table],
        "local_max_response_items": 20_000,
        "official_row_limit": 6000 if table == "daily_basic" else None,
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
    dates = _announcement_dates(
        start=source_plan["financial_start"],
        end=source_plan["as_of"],
    )
    for table in sorted(SOURCE_ENDPOINTS):
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
        elif table in _STATEMENT_TABLES:
            for announced_at in dates:
                rows.append(
                    _request_row(
                        ordinal=len(rows),
                        table=table,
                        params={"end_date": announced_at, "start_date": announced_at},
                        partition_type="EXACT_ANNOUNCEMENT_DATE_ALL_PERIODS",
                    )
                )
        elif table == "fina_indicator":
            for announced_at in dates:
                rows.append(
                    _request_row(
                        ordinal=len(rows),
                        table=table,
                        params={
                            "ann_date": announced_at,
                            "end_date": source_plan["as_of"],
                            "start_date": source_plan["financial_start"],
                        },
                        partition_type=("EXACT_ANNOUNCEMENT_DATE_REPORT_PERIOD_WINDOW"),
                    )
                )
        else:
            for period in source_plan["financial_periods"]:
                rows.append(
                    _request_row(
                        ordinal=len(rows),
                        table=table,
                        params={"period": period},
                        partition_type="PERIOD",
                    )
                )
    if len({row["request_key"] for row in rows}) != len(rows):
        raise FundamentalAcquisitionError("official request key is duplicated")
    return rows


def _request_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(_REQUEST_ROWS_HASH_DOMAIN)
    for row in rows:
        payload = canonical_bytes(row)
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


@fundamental_contract
def replay_official_partition_requests(
    document: Mapping[str, Any],
    *,
    source_execution_closure: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = validate_fundamental_execution_closure(source_execution_closure)
    rows = _request_rows(source["request_plan"])
    schedule = document["request_schedule"]
    if (
        len(rows) != schedule["planned_request_count"]
        or len(rows) != document["planned_terminal_request_count"]
        or _request_rows_sha256(rows) != schedule["request_rows_sha256"]
    ):
        raise FundamentalAcquisitionError("official request schedule replay mismatch")
    return rows


@fundamental_contract
def build_official_partition_plan(
    *,
    source_execution_closure: Mapping[str, Any],
    probe_observations: Sequence[Mapping[str, Any]],
    document_observed_at: str,
    created_at: str,
) -> dict[str, Any]:
    """Seal the sole supported official unfiltered announcement-date topology."""

    source = validate_fundamental_execution_closure(source_execution_closure)
    source_plan = source["request_plan"]
    created = timestamp(created_at, label="created_at")
    if created < source["created_at"]:
        raise FundamentalAcquisitionError("partition plan predates source closure")
    documents = _documents(document_observed_at)
    if any(row["document_observed_at"] > created for row in documents):
        raise FundamentalAcquisitionError("official documentation is future-dated")
    probes = _probes(probe_observations, created_at=created)
    requests = _request_rows(source_plan)
    terminal = len(requests)
    dates = _announcement_dates(
        start=source_plan["financial_start"],
        end=source_plan["as_of"],
    )
    keyset_sha = hashlib.sha256(canonical_bytes(dates)).hexdigest()
    proofs = [
        {
            "date_count": len(dates),
            "domain_basis": "ALL_CALENDAR_DATES_INCLUSIVE",
            "end_date": source_plan["as_of"],
            "endpoint": SOURCE_ENDPOINTS[table],
            "ordered_keyset_sha256": keyset_sha,
            "partition_parameter": (
                "ann_date" if table == "fina_indicator" else "start_date=end_date"
            ),
            "physical_projection_columns": (
                list(_STATEMENT_PROJECTION_COLUMNS) if table in _STATEMENT_TABLES else []
            ),
            "pit_cutoff": timestamp(source_plan["pit_cutoff"], label="pit_cutoff"),
            "report_period_end": source_plan["as_of"],
            "report_period_start": source_plan["financial_start"],
            "start_date": source_plan["financial_start"],
            "table": table,
        }
        for table in sorted((*_STATEMENT_TABLES, "fina_indicator"))
    ]
    baseline_attempts = source_plan["baseline_network_attempts"]
    body = {
        **common_fields(timestamp_value=created),
        "announcement_date_keyset_proofs": proofs,
        "as_of": session_date(source_plan["as_of"], label="as_of"),
        "baseline_network_attempts": baseline_attempts,
        "created_at": created,
        "document_refs": documents,
        "kind": OFFICIAL_PARTITION_PLAN_KIND,
        "local_max_response_items": 20_000,
        "max_attempts_per_partition": 1,
        "performance_gate": {
            "baseline_network_attempts": baseline_attempts,
            "mode": "OWNER_AUTHORIZED_EXACT_ANN_DATE_FULL_PIT_KEYSET_NO_RATIO_CAP",
            "multiplier": None,
            "passed": True,
            "planned_network_attempts": terminal,
        },
        "pit_cutoff": timestamp(source_plan["pit_cutoff"], label="pit_cutoff"),
        "planned_max_network_attempts": terminal,
        "planned_terminal_request_count": terminal,
        "probe_observations": probes,
        "request_schedule": {
            "physical_statement_projection_columns": list(_STATEMENT_PROJECTION_COLUMNS),
            "planned_request_count": terminal,
            "request_generator": _REQUEST_GENERATOR,
            "request_rows_sha256": _request_rows_sha256(requests),
        },
        "scope_policy": {
            "code_change_behavior": "BASELINE_EXACT_CODES_ONLY",
            "current_subject_scope_ref": source_plan["market_scope_ref"],
            "daily_scope": _SCOPE_MODE,
            "financial_scope": _SCOPE_MODE,
        },
        "source_execution_closure_ref": content_ref(
            source,
            identity_field="closure_id",
        ),
        "source_plan_id": source_plan["plan_id"],
    }
    return seal(body, identity_field="partition_plan_id")


@fundamental_contract
def validate_official_partition_plan(
    document: Mapping[str, Any],
    *,
    source_execution_closure: Mapping[str, Any],
    probe_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="partition_plan_id")
    require_exact_keys(value, _PLAN_FIELDS, label="official partition plan")
    if value.get("kind") != OFFICIAL_PARTITION_PLAN_KIND:
        raise FundamentalAcquisitionError("official partition plan kind mismatch")
    proofs = value["announcement_date_keyset_proofs"]
    if not isinstance(proofs, list) or len(proofs) != 4:
        raise FundamentalAcquisitionError("announcement proof set is invalid")
    for proof in proofs:
        require_exact_keys(proof, _PROOF_FIELDS, label="announcement proof")
    require_exact_keys(
        value["request_schedule"],
        _SCHEDULE_FIELDS,
        label="official request schedule",
    )
    expected = build_official_partition_plan(
        source_execution_closure=source_execution_closure,
        probe_observations=probe_observations,
        document_observed_at=value["document_refs"][0]["document_observed_at"],
        created_at=value["created_at"],
    )
    if value != expected:
        raise FundamentalAcquisitionError("official partition plan replay mismatch")
    for row in value["document_refs"]:
        require_exact_keys(row, _DOCUMENT_FIELDS, label="official document ref")
    for row in replay_official_partition_requests(
        value,
        source_execution_closure=source_execution_closure,
    ):
        require_exact_keys(row, _REQUEST_FIELDS, label="official request row")
    return value


__all__ = [
    "build_official_partition_plan",
    "replay_official_partition_requests",
    "validate_official_partition_plan",
]
