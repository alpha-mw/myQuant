"""Fail-closed Tushare support capture for Fundamental successor generations.

This module is deliberately independent from the research-only Fundamental v4
baseline-comparison contracts.  It owns provider acquisition and immutable
support evidence only; it never reads or writes a canonical Fundamental
pointer and it does not derive or promote a generation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Final, NoReturn, Protocol

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.v17_v4_runtime.tushare_https import (
    TushareHttpsError,
    TushareResponse,
    replay_tushare_response_bytes,
)

FUNDAMENTAL_SUCCESSOR_SUPPORT_PLAN_SCHEMA: Final = "myquant-fundamental-successor-support-plan.v4"
FUNDAMENTAL_SUCCESSOR_REQUEST_RECEIPT_SCHEMA: Final = (
    "myquant-fundamental-successor-request-receipt.v4"
)
FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA: Final = (
    "myquant-fundamental-successor-support-fileset.v5"
)
FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY: Final = (
    "myquant-fundamental-successor-canonicalization.v4"
)
FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA: Final = (
    "cn-fundamental-successor-provider-manifest.v5"
)
FUNDAMENTAL_SUCCESSOR_BINDING_SCHEMA: Final = "myquant-fundamental-successor-support-binding.v5"
FUNDAMENTAL_SUCCESSOR_RECORD_SCHEMA: Final = "myquant-fundamental-successor-support-record.v4"
FUNDAMENTAL_SUCCESSOR_TABLE_SCHEMA: Final = "myquant-fundamental-successor-canonical-table.v4"
FUNDAMENTAL_SUCCESSOR_REQUEST_ENVELOPE_SCOPE_SCHEMA: Final = (
    "myquant-fundamental-successor-request-envelope-scope.v1"
)
FUNDAMENTAL_SUCCESSOR_CANONICAL_SUBJECT_SCOPE_SCHEMA: Final = (
    "myquant-fundamental-successor-canonical-subject-scope.v1"
)
FUNDAMENTAL_SUCCESSOR_FAILURE_EVIDENCE_SCHEMA: Final = (
    "myquant-fundamental-successor-failure-evidence.v1"
)
FUNDAMENTAL_SUCCESSOR_OPAQUE_COMP_TYPE_EVIDENCE_SCHEMA: Final = (
    "myquant-fundamental-successor-opaque-comp-type-evidence.v2"
)
FUNDAMENTAL_SUCCESSOR_OPAQUE_COMP_TYPE_ACCOUNTING_SCHEMA: Final = (
    "myquant-fundamental-successor-opaque-comp-type-accounting.v2"
)
FUNDAMENTAL_SUCCESSOR_CLASSIFICATION_PARTITION_SCHEMA: Final = (
    "myquant-fundamental-successor-observation-classification.v1"
)
FUNDAMENTAL_SUCCESSOR_UNSUPPORTED_INVENTORY_SCHEMA: Final = (
    "myquant-fundamental-successor-unsupported-inventory.v1"
)

_DEFERRED_AUTHORITY_STATE: Final = "DEFERRED_UNSUPPORTED_OBSERVATIONS"
_AUTHORITATIVE_AUTHORITY_STATE: Final = "AUTHORITATIVE_DELTA_COMPLETE"

SUCCESSOR_SUPPORT_PLAN_VERSION: Final = FUNDAMENTAL_SUCCESSOR_SUPPORT_PLAN_SCHEMA
SUCCESSOR_SUPPORT_REQUEST_RECEIPT_VERSION: Final = FUNDAMENTAL_SUCCESSOR_REQUEST_RECEIPT_SCHEMA
SUCCESSOR_SUPPORT_FILESET_VERSION: Final = FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA
SUCCESSOR_SUPPORT_CANONICALIZATION_VERSION: Final = FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY
SUCCESSOR_SUPPORT_PROVIDER_MANIFEST_VERSION: Final = FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA

_HEX_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_DATE_RE: Final = re.compile(r"^[0-9]{8}$", re.ASCII)
_TS_CODE_RE: Final = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)
_PROVIDER_EXTERNAL_TS_CODE_RE: Final = re.compile(
    r"^(?:[A-Z][0-9]{5}|[0-9]{6}![1-9][0-9]{0,2})\.(?:BJ|SH|SZ)$",
    re.ASCII,
)
_POINTER_NAMES: Final = frozenset({"market", "pit", "predecessor"})
_STATEMENT_TABLES: Final = frozenset({"balancesheet", "cashflow", "income"})
_TABLES: Final = (
    "balancesheet",
    "cashflow",
    "daily_basic",
    "fina_indicator",
    "forecast",
    "income",
)
_ENDPOINTS: Final = {
    "balancesheet": "balancesheet_vip",
    "cashflow": "cashflow_vip",
    "daily_basic": "daily_basic",
    "fina_indicator": "fina_indicator_vip",
    "forecast": "forecast_vip",
    "income": "income_vip",
}
_EXPECTED_FIELDS: Final = {
    "balancesheet": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "total_liab",
        "total_assets",
        "update_flag",
        "report_type",
        "comp_type",
    ),
    "cashflow": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "n_cashflow_act",
        "c_pay_acq_const_fiolta",
        "free_cashflow",
        "update_flag",
        "report_type",
        "comp_type",
    ),
    "daily_basic": (
        "ts_code",
        "trade_date",
        "total_mv",
        "circ_mv",
        "pe",
        "pb",
    ),
    "fina_indicator": (
        "ts_code",
        "ann_date",
        "end_date",
        "roe_dt",
        "roe",
        "roa",
        "debt_to_assets",
        "netprofit_yoy",
        "update_flag",
    ),
    "forecast": (
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
    ),
    "income": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "n_income",
        "n_income_attr_p",
        "update_flag",
        "report_type",
        "comp_type",
    ),
}
_DATE_FIELDS: Final = frozenset({"ann_date", "end_date", "f_ann_date", "trade_date"})
_CLASS_FIELDS: Final = frozenset({"comp_type", "report_type", "update_flag"})
_SUPPORTED_COMP_TYPES: Final = frozenset({"1", "2", "3", "4"})
_OPAQUE_BALANCESHEET_COMP_TYPE: Final = "7"
_TEXT_FIELDS: Final = frozenset({"change_reason", "summary", "type"})
_AVAILABILITY_POLICY: Final = {
    "balancesheet": "F_ANN_DATE_ELSE_ANN_DATE",
    "cashflow": "F_ANN_DATE_ELSE_ANN_DATE",
    "daily_basic": "TRADE_DATE",
    "fina_indicator": "ANN_DATE",
    "forecast": "ANN_DATE",
    "income": "F_ANN_DATE_ELSE_ANN_DATE",
}
_UPDATE_POLICY: Final = {
    "balancesheet": "REQUIRED_ZERO_OR_ONE_DOMINANCE",
    "cashflow": "REQUIRED_ZERO_OR_ONE_DOMINANCE",
    "daily_basic": "ABSENT",
    "fina_indicator": "OPTIONAL_ZERO_OR_ONE_DOMINANCE",
    "forecast": "ABSENT",
    "income": "REQUIRED_ZERO_OR_ONE_DOMINANCE",
}
_PHYSICAL_CLASS_POLICY: Final = {
    "balancesheet": (
        "REPORT_TYPE_ONE_COMP_TYPE_1_TO_4_PLUS_7_OPAQUE_EQUIVALENCE_ONLY"
    ),
    "cashflow": "REPORT_TYPE_ONE_COMP_TYPE_ONE_TO_FOUR",
    "daily_basic": "ABSENT",
    "fina_indicator": "ABSENT",
    "forecast": "ABSENT",
    "income": "REPORT_TYPE_ONE_COMP_TYPE_ONE_TO_FOUR",
}
_SECRET_KEY_FRAGMENTS: Final = (
    "api_key",
    "authorization",
    "bearer",
    "header",
    "secret",
    "token",
)
_DEFAULT_MINIMUM_FREE_DISK_BYTES: Final = 256 * 1024 * 1024
_DEFAULT_MAXIMUM_RECORD_BYTES: Final = 32 * 1024 * 1024
_DECODE_ESTIMATED_BYTES_PER_CELL: Final = 512
_MAX_TABLE_MEMORY_FRACTION: Final = Decimal("0.50")
_PARQUET_ROW_GROUP_ROWS: Final = 2_048
_MAX_STREAM_BATCH_ROWS: Final = 2_048
_MAX_STREAM_BATCH_BYTES: Final = 16 * 1024 * 1024
_MAX_SYMBOL_ROWS: Final = 100_000
_MAX_SYMBOL_BYTES: Final = 64 * 1024 * 1024
_PARQUET_SCHEMA: Final = pa.schema(
    [
        pa.field("ts_code", pa.string(), nullable=False),
        pa.field("sort_date", pa.string(), nullable=False),
        pa.field("end_date", pa.string(), nullable=False),
        pa.field("row_json", pa.binary(), nullable=False),
    ]
)


class FundamentalSuccessorSourceError(RuntimeError):
    """A static-code acquisition or evidence-validation failure."""

    def __init__(
        self,
        code: str,
        *,
        response_evidence: Mapping[str, Any] | None = None,
        raw_response_bytes: bytes | None = None,
    ) -> None:
        self.code = code
        self.response_evidence = (
            dict(response_evidence) if response_evidence is not None else None
        )
        self.raw_response_bytes = raw_response_bytes
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


class SuccessorTushareClient(Protocol):
    """Minimal injected transport; credentials remain outside this module."""

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareResponse: ...


def _fail(code: str) -> NoReturn:
    raise FundamentalSuccessorSourceError(code) from None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        _fail("SUCCESSOR_NON_CANONICAL_JSON")


def _sealed(body: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    result = dict(body)
    if identity_field in result:
        _fail("SUCCESSOR_IDENTITY_FIELD_PREEXISTS")
    result[identity_field] = _sha256(_canonical_json_bytes(result))
    return result


def _validate_seal(value: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("SUCCESSOR_SEALED_VALUE_INVALID")
    result = dict(value)
    identity = result.pop(identity_field, None)
    if type(identity) is not str or _HEX_SHA256_RE.fullmatch(identity) is None:
        _fail("SUCCESSOR_SEAL_INVALID")
    if _sha256(_canonical_json_bytes(result)) != identity:
        _fail("SUCCESSOR_SEAL_INVALID")
    result[identity_field] = identity
    return result


def _date(value: Any, *, label: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        _fail(f"SUCCESSOR_{label.upper()}_INVALID")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        _fail(f"SUCCESSOR_{label.upper()}_INVALID")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"SUCCESSOR_{label.upper()}_INVALID")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"SUCCESSOR_{label.upper()}_INVALID")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"SUCCESSOR_{label.upper()}_INVALID")
    normalized = parsed.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _hex_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _HEX_SHA256_RE.fullmatch(value) is None:
        _fail(f"SUCCESSOR_{label.upper()}_INVALID")
    return value


def _natural_dates(start: str, end: str) -> tuple[str, ...]:
    cursor = datetime.strptime(start, "%Y%m%d").date()
    terminal = datetime.strptime(end, "%Y%m%d").date()
    values: list[str] = []
    while cursor <= terminal:
        values.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    return tuple(values)


def _canonical_strings(
    values: Sequence[str],
    *,
    label: str,
    validator: Callable[[Any], str],
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail(f"SUCCESSOR_{label.upper()}_INVALID")
    normalized = [validator(value) for value in values]
    if len(normalized) != len(set(normalized)):
        _fail(f"SUCCESSOR_{label.upper()}_DUPLICATED")
    return tuple(sorted(normalized, key=lambda item: item.encode("ascii")))


def _symbol(value: Any) -> str:
    if type(value) is not str:
        _fail("SUCCESSOR_SYMBOL_INVALID")
    normalized = value.strip().upper()
    if _TS_CODE_RE.fullmatch(normalized) is None:
        _fail("SUCCESSOR_SYMBOL_INVALID")
    # The production successor has no implicit alias table.  Accepting case,
    # whitespace or exchange-suffix aliases here would make it possible for an
    # out-of-scope provider row to normalize onto an in-scope identity.
    if value != normalized:
        _fail("SUCCESSOR_SYMBOL_ALIAS_UNSUPPORTED")
    return normalized


def _response_symbol(value: Any, *, symbols: frozenset[str]) -> str:
    """Keep exact provider-external identities only outside subject scope."""

    if type(value) is not str:
        _fail("SUCCESSOR_SYMBOL_INVALID")
    normalized = value.strip().upper()
    if value != normalized:
        _fail("SUCCESSOR_SYMBOL_ALIAS_UNSUPPORTED")
    if _TS_CODE_RE.fullmatch(normalized) is not None:
        return normalized
    if (
        _PROVIDER_EXTERNAL_TS_CODE_RE.fullmatch(normalized) is not None
        and normalized not in symbols
    ):
        return normalized
    _fail("SUCCESSOR_SYMBOL_INVALID")


def _scope_ref(body: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed(body, identity_field="scope_sha256")


def _request_envelope_scope_ref(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    has_bounded_dependencies = any(
        request.get("partition_type") == "EXACT_SYMBOL_REPORT_PERIOD_SUPPORT"
        for request in requests
    )
    return _scope_ref(
        {
            "identity_policy": "EXACT_CANONICAL_TS_CODE_NO_ALIASES",
            "partition_contract": (
                "GLOBAL_DELTA_PARTITIONS_PLUS_EXACT_SUBJECT_PERIOD_DEPENDENCIES"
                if has_bounded_dependencies
                else "GLOBAL_PROVIDER_EXACT_PARTITION_ALL_SYMBOLS"
            ),
            "request_topology_sha256": _sha256(_canonical_json_bytes(list(requests))),
            "schema_version": FUNDAMENTAL_SUCCESSOR_REQUEST_ENVELOPE_SCOPE_SCHEMA,
        }
    )


def _canonical_subject_scope_ref(
    symbols: Sequence[str],
    *,
    authority_closure_sha256: str,
) -> dict[str, Any]:
    keyset_sha256 = _sha256(_canonical_json_bytes(list(symbols)))
    return _scope_ref(
        {
            "authority_closure_sha256": _hex_sha256(
                authority_closure_sha256,
                label="canonical_subject_scope_authority_sha256",
            ),
            "frozen_before_provider_capture": True,
            "identity_policy": "EXACT_CANONICAL_TS_CODE_NO_ALIASES",
            "projection_policy": (
                "PARENT_PREFIX_UNION_DELTA_PIT_EXPECTED_UNION_"
                "DELTA_OBSERVED_BARS_UNION_TARGET_FULL_A"
            ),
            "schema_version": FUNDAMENTAL_SUCCESSOR_CANONICAL_SUBJECT_SCOPE_SCHEMA,
            "subject_symbol_count": len(symbols),
            "subject_symbol_keyset_sha256": keyset_sha256,
        }
    )


def _endpoint_capabilities() -> dict[str, dict[str, Any]]:
    return {
        table: {
            "availability_policy": _AVAILABILITY_POLICY[table],
            "endpoint": _ENDPOINTS[table],
            "expected_fields": list(_EXPECTED_FIELDS[table]),
            "field_kinds": {
                field: (
                    "SYMBOL"
                    if field == "ts_code"
                    else (
                        "DATE"
                        if field in _DATE_FIELDS
                        else (
                            "CLASS"
                            if field in _CLASS_FIELDS
                            else "TEXT" if field in _TEXT_FIELDS else "DECIMAL"
                        )
                    )
                )
                for field in _EXPECTED_FIELDS[table]
            },
            "physical_class_policy": _PHYSICAL_CLASS_POLICY[table],
            "row_ceiling": 6000 if table == "daily_basic" else 20_000,
            "update_policy": _UPDATE_POLICY[table],
        }
        for table in _TABLES
    }


SUCCESSOR_ENDPOINT_CAPABILITIES: Final = MappingProxyType(
    {table: MappingProxyType(value) for table, value in _endpoint_capabilities().items()}
)


def _request_row(
    *,
    ordinal: int,
    table: str,
    params: Mapping[str, str],
    partition_type: str,
) -> dict[str, Any]:
    parameter_text = "&".join(f"{key}={params[key]}" for key in sorted(params))
    return {
        "endpoint": _ENDPOINTS[table],
        "expected_fields": list(_EXPECTED_FIELDS[table]),
        "ordinal": ordinal,
        "params": dict(sorted(params.items())),
        "partition_type": partition_type,
        "request_key": f"{table}|{parameter_text}",
        "row_ceiling": 6000 if table == "daily_basic" else 20_000,
        "table": table,
    }


def _build_request_rows(
    *,
    support_start: str,
    target_date: str,
    open_sessions: Sequence[str],
    financial_support_dependencies: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    dates = _natural_dates(support_start, target_date)
    rows: list[dict[str, Any]] = []
    for table in _TABLES:
        if table == "daily_basic":
            partitions = [("TRADE_DATE", {"trade_date": session}) for session in open_sessions]
        elif table in _STATEMENT_TABLES:
            partitions = [
                (
                    "EXACT_ANNOUNCEMENT_DATE_ALL_PERIODS",
                    {"end_date": announced_at, "start_date": announced_at},
                )
                for announced_at in dates
            ]
        elif table == "fina_indicator":
            partitions = [
                (
                    "EXACT_ANNOUNCEMENT_DATE_ALL_REPORT_PERIODS",
                    {"ann_date": announced_at},
                )
                for announced_at in dates
            ]
        elif table == "forecast":
            partitions = [
                ("EXACT_ANNOUNCEMENT_DATE", {"ann_date": announced_at}) for announced_at in dates
            ]
        else:  # pragma: no cover - closed by the table constant
            _fail("SUCCESSOR_TABLE_INVALID")
        for partition_type, params in partitions:
            rows.append(
                _request_row(
                    ordinal=len(rows),
                    table=table,
                    params=params,
                    partition_type=partition_type,
                )
            )
    for dependency in financial_support_dependencies:
        rows.append(
            _request_row(
                ordinal=len(rows),
                table=dependency["table"],
                params={
                    "period": dependency["end_date"],
                    "ts_code": dependency["ts_code"],
                },
                partition_type="EXACT_SYMBOL_REPORT_PERIOD_SUPPORT",
            )
        )
    return rows


def _financial_support_dependencies(
    values: Sequence[Mapping[str, Any]],
    *,
    support_start: str,
    subject_symbols: Sequence[str],
) -> tuple[dict[str, str], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("SUCCESSOR_FINANCIAL_SUPPORT_DEPENDENCIES_INVALID")
    subjects = frozenset(subject_symbols)
    normalized: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "end_date",
            "table",
            "ts_code",
        }:
            _fail("SUCCESSOR_FINANCIAL_SUPPORT_DEPENDENCIES_INVALID")
        table = str(value["table"])
        if table not in _STATEMENT_TABLES:
            _fail("SUCCESSOR_FINANCIAL_SUPPORT_TABLE_INVALID")
        symbol = _symbol(value["ts_code"])
        end_date = _date(value["end_date"], label="financial_support_end_date")
        if symbol not in subjects or end_date >= support_start:
            _fail("SUCCESSOR_FINANCIAL_SUPPORT_DEPENDENCY_OUT_OF_SCOPE")
        normalized.append({"end_date": end_date, "table": table, "ts_code": symbol})
    ordered = sorted(
        normalized,
        key=lambda row: (row["table"], row["ts_code"], row["end_date"]),
    )
    if len(
        {(row["table"], row["ts_code"], row["end_date"]) for row in ordered}
    ) != len(ordered):
        _fail("SUCCESSOR_FINANCIAL_SUPPORT_DEPENDENCY_DUPLICATE")
    return tuple(ordered)


def build_successor_support_plan(
    *,
    support_start: str,
    target_date: str,
    open_sessions: Sequence[str],
    symbols: Sequence[str],
    canonical_subject_scope_authority_sha256: str,
    income_support_dependencies: Sequence[Mapping[str, Any]] = (),
    financial_support_dependencies: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a deterministic exact-partition production acquisition plan."""

    support = _date(support_start, label="support_start")
    target = _date(target_date, label="target_date")
    if support > target:
        _fail("SUCCESSOR_DATE_RANGE_INVERTED")
    sessions = _canonical_strings(
        open_sessions,
        label="open_sessions",
        validator=lambda value: _date(value, label="open_session"),
    )
    if not sessions or target not in sessions:
        _fail("SUCCESSOR_TARGET_OPEN_SESSION_MISSING")
    if any(session < support or session > target for session in sessions):
        _fail("SUCCESSOR_OPEN_SESSION_OUT_OF_RANGE")
    subject_symbols = _canonical_strings(
        symbols,
        label="symbols",
        validator=_symbol,
    )
    if not subject_symbols:
        _fail("SUCCESSOR_SYMBOL_SCOPE_EMPTY")
    dependency_values = [
        {"table": "income", **dict(value)}
        for value in income_support_dependencies
    ] + [dict(value) for value in financial_support_dependencies]
    dependencies = _financial_support_dependencies(
        dependency_values,
        support_start=support,
        subject_symbols=subject_symbols,
    )
    requests = _build_request_rows(
        support_start=support,
        target_date=target,
        open_sessions=sessions,
        financial_support_dependencies=dependencies,
    )
    request_topology_sha256 = _sha256(_canonical_json_bytes(requests))
    request_envelope_scope_ref = _request_envelope_scope_ref(requests)
    canonical_subject_scope_ref = _canonical_subject_scope_ref(
        subject_symbols,
        authority_closure_sha256=canonical_subject_scope_authority_sha256,
    )
    body = {
        "canonicalization_policy": FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY,
        "canonical_subject_scope_ref": canonical_subject_scope_ref,
        "endpoint_capabilities": _endpoint_capabilities(),
        "report_period_envelope": {
            "capture_policy": (
                "ALL_REPORT_PERIODS_FOR_EXACT_ANNOUNCEMENT_DATE_PLUS_"
                "SEALED_EXACT_FINANCIAL_SUPPORT"
                if dependencies
                else "ALL_REPORT_PERIODS_FOR_EXACT_ANNOUNCEMENT_DATE"
            ),
            "lower_bound": None,
            "upper_bound_policy": "END_DATE_NOT_AFTER_AVAILABILITY",
        },
        "open_sessions": list(sessions),
        "planned_request_count": len(requests),
        "request_topology_sha256": request_topology_sha256,
        "request_envelope_scope_ref": request_envelope_scope_ref,
        "requests": requests,
        "schema_version": FUNDAMENTAL_SUCCESSOR_SUPPORT_PLAN_SCHEMA,
        "subject_symbol_keyset_sha256": _sha256(_canonical_json_bytes(list(subject_symbols))),
        "subject_symbols": list(subject_symbols),
        "support_start": support,
        "target_date": target,
    }
    if dependencies:
        body["financial_support_dependencies"] = list(dependencies)
    return _sealed(body, identity_field="plan_sha256")


def replay_successor_support_requests(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate and replay the exact deterministic request topology."""

    sealed = _validate_seal(plan, identity_field="plan_sha256")
    required = {
        "canonicalization_policy",
        "canonical_subject_scope_ref",
        "endpoint_capabilities",
        "open_sessions",
        "plan_sha256",
        "planned_request_count",
        "request_topology_sha256",
        "request_envelope_scope_ref",
        "report_period_envelope",
        "requests",
        "schema_version",
        "subject_symbol_keyset_sha256",
        "subject_symbols",
        "support_start",
        "target_date",
    }
    allowed_fields = required | {"financial_support_dependencies"}
    observed_fields = set(sealed)
    if observed_fields != required and observed_fields != allowed_fields:
        _fail("SUCCESSOR_PLAN_FIELDS_INVALID")
    if (
        sealed["schema_version"] != FUNDAMENTAL_SUCCESSOR_SUPPORT_PLAN_SCHEMA
        or sealed["canonicalization_policy"] != FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY
        or sealed["endpoint_capabilities"] != _endpoint_capabilities()
    ):
        _fail("SUCCESSOR_PLAN_CONTRACT_MISMATCH")
    expected = build_successor_support_plan(
        support_start=sealed["support_start"],
        target_date=sealed["target_date"],
        open_sessions=sealed["open_sessions"],
        symbols=sealed["subject_symbols"],
        canonical_subject_scope_authority_sha256=sealed[
            "canonical_subject_scope_ref"
        ]["authority_closure_sha256"],
        financial_support_dependencies=sealed.get("financial_support_dependencies", ()),
    )
    if _canonical_json_bytes(sealed) != _canonical_json_bytes(expected):
        _fail("SUCCESSOR_PLAN_REPLAY_MISMATCH")
    requests = sealed["requests"]
    if (
        type(requests) is not list
        or sealed["planned_request_count"] != len(requests)
        or sealed["request_topology_sha256"] != _sha256(_canonical_json_bytes(requests))
        or [row.get("ordinal") for row in requests] != list(range(len(requests)))
        or len({row.get("request_key") for row in requests}) != len(requests)
    ):
        _fail("SUCCESSOR_REQUEST_TOPOLOGY_INVALID")
    return tuple(dict(row) for row in requests)


def _typed_scalar(value: Any, *, logical: bool) -> dict[str, Any]:
    if value is None:
        return {"kind": "null", "value": ""}
    if type(value) is bool:
        return {"kind": "boolean", "value": "true" if value else "false"}
    if type(value) is int:
        if logical:
            return {"kind": "number", "value": str(value)}
        return {"kind": "integer", "value": str(value)}
    if type(value) is Decimal:
        if not value.is_finite():
            _fail("SUCCESSOR_RESPONSE_SCALAR_INVALID")
        if logical:
            normalized = Decimal(0) if value == 0 else value.normalize()
            return {"kind": "number", "value": normalized.to_eng_string()}
        return {"kind": "decimal", "value": str(value)}
    if type(value) is str:
        return {"kind": "text", "value": value}
    _fail("SUCCESSOR_RESPONSE_SCALAR_INVALID")


def _decode_typed_scalar(value: Any) -> Any:
    if type(value) is not dict or set(value) != {"kind", "value"}:
        _fail("SUCCESSOR_TYPED_SCALAR_INVALID")
    kind = value["kind"]
    payload = value["value"]
    if type(kind) is not str or type(payload) is not str:
        _fail("SUCCESSOR_TYPED_SCALAR_INVALID")
    if kind == "null" and payload == "":
        return None
    if kind == "boolean" and payload in {"false", "true"}:
        return payload == "true"
    if kind == "integer" and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", payload):
        return int(payload)
    if kind in {"decimal", "number"}:
        try:
            result = Decimal(payload)
        except Exception:
            _fail("SUCCESSOR_TYPED_SCALAR_INVALID")
        if not result.is_finite():
            _fail("SUCCESSOR_TYPED_SCALAR_INVALID")
        return result
    if kind == "text":
        return payload
    _fail("SUCCESSOR_TYPED_SCALAR_INVALID")


def _typed_row(row: Mapping[str, Any], fields: Sequence[str], *, logical: bool) -> list[Any]:
    return [_typed_scalar(row[field], logical=logical) for field in fields]


def _row_sort_key(row: Mapping[str, Any], fields: Sequence[str]) -> bytes:
    return _canonical_json_bytes(_typed_row(row, fields, logical=True))


def _class_text(value: Any, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if type(value) is int and value in {0, 1, 2, 3, 4, 7}:
        return str(value)
    if type(value) is str and value in {"0", "1", "2", "3", "4", "7"}:
        return value
    _fail("SUCCESSOR_CLASSIFICATION_VALUE_INVALID")


def _row_date(value: Any, *, label: str, optional: bool = False) -> str:
    if optional and (value is None or value == ""):
        return ""
    return _date(value, label=label)


def _normalize_response_row(
    *,
    table: str,
    request: Mapping[str, Any],
    fields: Sequence[str],
    values: Sequence[Any],
    symbols: frozenset[str],
    target_date: str,
    enforce_subject_scope: bool = True,
) -> dict[str, Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("SUCCESSOR_RESPONSE_ROW_INVALID")
    if len(values) != len(fields):
        _fail("SUCCESSOR_RESPONSE_ROW_INVALID")
    row = dict(zip(fields, values, strict=True))
    field_kinds = _endpoint_capabilities()[table]["field_kinds"]
    for field, value in row.items():
        _typed_scalar(value, logical=False)
        kind = field_kinds[field]
        if (
            kind == "DECIMAL"
            and value is not None
            and type(value)
            not in {
                int,
                Decimal,
            }
        ):
            _fail("SUCCESSOR_RESPONSE_FIELD_TYPE_INVALID")
        if kind == "TEXT" and value is not None and type(value) is not str:
            _fail("SUCCESSOR_RESPONSE_FIELD_TYPE_INVALID")
    symbol = _response_symbol(row["ts_code"], symbols=symbols)
    if enforce_subject_scope and symbol not in symbols:
        _fail("SUCCESSOR_RESPONSE_SYMBOL_OUT_OF_SCOPE")
    row["ts_code"] = symbol
    params = request["params"]
    if table == "daily_basic":
        trade_date = _row_date(row["trade_date"], label="trade_date")
        if trade_date != params["trade_date"]:
            _fail("SUCCESSOR_RESPONSE_PARTITION_SCOPE_MISMATCH")
        row["trade_date"] = trade_date
        return row
    end_date = _row_date(row["end_date"], label="end_date")
    row["end_date"] = end_date
    if table in _STATEMENT_TABLES:
        ann_date = _row_date(row["ann_date"], label="ann_date")
        f_ann_date = _row_date(row["f_ann_date"], label="f_ann_date", optional=True)
        if request.get("partition_type") == "EXACT_SYMBOL_REPORT_PERIOD_SUPPORT":
            if (
                table not in _STATEMENT_TABLES
                or symbol != params.get("ts_code")
                or end_date != params.get("period")
            ):
                _fail("SUCCESSOR_RESPONSE_PARTITION_SCOPE_MISMATCH")
        elif (
            params.get("start_date") != params.get("end_date")
            or ann_date != params.get("start_date")
        ):
            _fail("SUCCESSOR_RESPONSE_PARTITION_SCOPE_MISMATCH")
        availability = f_ann_date or ann_date
        if availability < ann_date or availability > target_date:
            _fail("SUCCESSOR_RESPONSE_AVAILABILITY_OUT_OF_ENVELOPE")
        if end_date > availability:
            _fail("SUCCESSOR_FINANCIAL_END_AFTER_AVAILABILITY")
        report_type = _class_text(row["report_type"], required=True)
        comp_type = _class_text(row["comp_type"], required=True)
        update_flag = _class_text(row["update_flag"], required=True)
        accepted_comp_types = set(_SUPPORTED_COMP_TYPES)
        if table == "balancesheet":
            accepted_comp_types.add(_OPAQUE_BALANCESHEET_COMP_TYPE)
        if report_type != "1" or comp_type not in accepted_comp_types:
            _fail("SUCCESSOR_STATEMENT_PHYSICAL_CLASS_INVALID")
        if update_flag not in {"0", "1"}:
            _fail("SUCCESSOR_UPDATE_FLAG_INVALID")
        row.update(
            {
                "ann_date": ann_date,
                "availability_date": availability,
                "comp_type": comp_type,
                "f_ann_date": f_ann_date,
                "report_type": report_type,
                "update_flag": update_flag,
            }
        )
        return row
    ann_date = _row_date(row["ann_date"], label="ann_date")
    if ann_date != params["ann_date"]:
        _fail("SUCCESSOR_RESPONSE_PARTITION_SCOPE_MISMATCH")
    row["ann_date"] = ann_date
    row["availability_date"] = ann_date
    if table == "fina_indicator":
        if end_date > ann_date:
            _fail("SUCCESSOR_FINANCIAL_END_AFTER_AVAILABILITY")
        update = _class_text(row["update_flag"], required=False)
        if update not in {None, "0", "1"}:
            _fail("SUCCESSOR_UPDATE_FLAG_INVALID")
        row["update_flag"] = update
    return row


def _output_fields(table: str) -> tuple[str, ...]:
    if table == "daily_basic":
        return _EXPECTED_FIELDS[table]
    return (*_EXPECTED_FIELDS[table], "availability_date")


def _business_key(table: str, row: Mapping[str, Any]) -> tuple[str, ...]:
    if table == "daily_basic":
        return (str(row["ts_code"]), str(row["trade_date"]))
    return (
        str(row["ts_code"]),
        str(row["end_date"]),
        str(row["availability_date"]),
    )


def _update_rank(table: str, row: Mapping[str, Any]) -> int:
    if table in _STATEMENT_TABLES or table == "fina_indicator":
        value = row.get("update_flag")
        return -1 if value is None else int(value)
    return 0


def _physical_update_identity(table: str, row: Mapping[str, Any]) -> tuple[str, ...]:
    if table in _STATEMENT_TABLES:
        return (str(row["report_type"]), str(row["comp_type"]))
    if table == "fina_indicator":
        return ("UNVERSIONED" if row.get("update_flag") is None else "VERSIONED",)
    return ()


def _projection_fields(table: str) -> tuple[str, ...]:
    excluded = {"comp_type", "report_type", "update_flag"}
    return tuple(field for field in _output_fields(table) if field not in excluded)


def _opaque_comp_type_evidence(
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = _output_fields(table)
    opaque_rows = [
        dict(row)
        for row in rows
        if str(row.get("comp_type") or "")
        == _OPAQUE_BALANCESHEET_COMP_TYPE
    ]
    if opaque_rows and table != "balancesheet":
        _fail("SUCCESSOR_STATEMENT_PHYSICAL_CLASS_INVALID")
    if not opaque_rows:
        return _sealed(
            {
                "schema_version": (
                    FUNDAMENTAL_SUCCESSOR_OPAQUE_COMP_TYPE_EVIDENCE_SCHEMA
                ),
                "opaque_comp_type_observation_count": 0,
                "opaque_comp_type_business_key_count": 0,
                "opaque_comp_type_observation_multiset_sha256": _sha256(
                    _canonical_json_bytes(
                        {"fields": list(fields), "rows": []}
                    )
                ),
                "opaque_to_supported_peer_pair_keyset_sha256": _sha256(
                    _canonical_json_bytes([])
                ),
                "opaque_equivalent_business_keys": [],
                "deferred_business_keys": [],
                "deferred_observations": [],
                "opaque_deferred_observation_count": 0,
                "opaque_unpaired_count": 0,
                "opaque_material_conflict_count": 0,
            },
            identity_field="evidence_sha256",
        )

    exact_seen: set[bytes] = set()
    exact_rows: list[dict[str, Any]] = []
    for value in rows:
        row = dict(value)
        token = _canonical_json_bytes(_typed_row(row, fields, logical=False))
        if token not in exact_seen:
            exact_seen.add(token)
            exact_rows.append(row)
    physical_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in exact_rows:
        physical_key = (
            *_business_key(table, row),
            *_physical_update_identity(table, row),
        )
        physical_groups.setdefault(physical_key, []).append(row)
    survivors: list[dict[str, Any]] = []
    for candidates in physical_groups.values():
        highest = max(_update_rank(table, row) for row in candidates)
        survivors.extend(
            row
            for row in candidates
            if _update_rank(table, row) == highest
        )

    logical_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in survivors:
        logical_groups.setdefault(_business_key(table, row), []).append(row)
    peer_pairs: list[dict[str, Any]] = []
    equivalent_business_keys: list[list[str]] = []
    deferred_business_keys: list[list[str]] = []
    deferred_observations: list[dict[str, Any]] = []
    projection_fields = _projection_fields(table)
    for key, candidates in logical_groups.items():
        opaque = [
            row
            for row in candidates
            if str(row.get("comp_type") or "")
            == _OPAQUE_BALANCESHEET_COMP_TYPE
        ]
        if not opaque:
            continue
        supported = [
            row
            for row in candidates
            if str(row.get("comp_type") or "") in _SUPPORTED_COMP_TYPES
        ]
        if not supported:
            opaque_projection_tokens = {
                _canonical_json_bytes(
                    _typed_row(row, projection_fields, logical=True)
                )
                for row in opaque
            }
            if len(opaque_projection_tokens) != 1:
                _fail("SUCCESSOR_OPAQUE_COMP_TYPE_MATERIAL_CONFLICT")
            deferred_business_keys.append(list(key))
            for row in sorted(opaque, key=lambda value: _row_sort_key(value, fields)):
                encoded = _typed_row(row, fields, logical=False)
                deferred_observations.append(
                    {
                        "business_key": list(key),
                        "row_sha256": _sha256(_canonical_json_bytes(encoded)),
                        "typed_row": encoded,
                    }
                )
            continue
        projection_tokens = {
            _canonical_json_bytes(
                _typed_row(row, projection_fields, logical=True)
            )
            for row in (*opaque, *supported)
        }
        if len(projection_tokens) != 1:
            _fail("SUCCESSOR_OPAQUE_COMP_TYPE_EQUIVALENCE_UNCLOSED")
        equivalent_business_keys.append(list(key))
        peer_pairs.append(
            {
                "business_key": list(key),
                "opaque_comp_type": _OPAQUE_BALANCESHEET_COMP_TYPE,
                "supported_comp_types": sorted(
                    {
                        str(row["comp_type"])
                        for row in supported
                    }
                ),
            }
        )

    opaque_observations = sorted(
        (_typed_row(row, fields, logical=False) for row in opaque_rows),
        key=_canonical_json_bytes,
    )
    opaque_business_keys = sorted(
        {tuple(_business_key(table, row)) for row in opaque_rows},
        key=lambda key: tuple(value.encode("utf-8") for value in key),
    )
    peer_pairs.sort(
        key=lambda value: _canonical_json_bytes(value)
    )
    equivalent_business_keys.sort(key=_canonical_json_bytes)
    deferred_business_keys.sort(key=_canonical_json_bytes)
    deferred_observations.sort(key=_canonical_json_bytes)
    if len(peer_pairs) + len(deferred_business_keys) != len(opaque_business_keys):
        _fail("SUCCESSOR_OPAQUE_COMP_TYPE_CLASSIFICATION_UNCLOSED")
    return _sealed(
        {
            "schema_version": (
                FUNDAMENTAL_SUCCESSOR_OPAQUE_COMP_TYPE_EVIDENCE_SCHEMA
            ),
            "opaque_comp_type_observation_count": len(opaque_rows),
            "opaque_comp_type_business_key_count": len(
                opaque_business_keys
            ),
            "opaque_comp_type_observation_multiset_sha256": _sha256(
                _canonical_json_bytes(
                    {"fields": list(fields), "rows": opaque_observations}
                )
            ),
            "opaque_to_supported_peer_pair_keyset_sha256": _sha256(
                _canonical_json_bytes(peer_pairs)
            ),
            "opaque_equivalent_business_keys": equivalent_business_keys,
            "deferred_business_keys": deferred_business_keys,
            "deferred_observations": deferred_observations,
            "opaque_deferred_observation_count": len(deferred_observations),
            "opaque_unpaired_count": len(deferred_business_keys),
            "opaque_material_conflict_count": 0,
        },
        identity_field="evidence_sha256",
    )


def _validate_opaque_comp_type_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _validate_seal(value, identity_field="evidence_sha256")
    required = {
        "deferred_business_keys",
        "deferred_observations",
        "evidence_sha256",
        "opaque_deferred_observation_count",
        "opaque_equivalent_business_keys",
        "opaque_comp_type_business_key_count",
        "opaque_comp_type_observation_count",
        "opaque_comp_type_observation_multiset_sha256",
        "opaque_material_conflict_count",
        "opaque_to_supported_peer_pair_keyset_sha256",
        "opaque_unpaired_count",
        "schema_version",
    }
    count_fields = {
        "opaque_comp_type_business_key_count",
        "opaque_comp_type_observation_count",
        "opaque_deferred_observation_count",
        "opaque_material_conflict_count",
        "opaque_unpaired_count",
    }
    if (
        set(evidence) != required
        or evidence["schema_version"]
        != FUNDAMENTAL_SUCCESSOR_OPAQUE_COMP_TYPE_EVIDENCE_SCHEMA
        or any(
            type(evidence[field]) is not int or evidence[field] < 0
            for field in count_fields
        )
        or evidence["opaque_material_conflict_count"] != 0
        or type(evidence["deferred_business_keys"]) is not list
        or type(evidence["opaque_equivalent_business_keys"]) is not list
        or type(evidence["deferred_observations"]) is not list
        or evidence["opaque_unpaired_count"]
        != len(evidence["deferred_business_keys"])
        or evidence["opaque_deferred_observation_count"]
        != len(evidence["deferred_observations"])
    ):
        _fail("SUCCESSOR_OPAQUE_COMP_TYPE_EVIDENCE_INVALID")
    deferred_keys = {
        tuple(value)
        for value in evidence["deferred_business_keys"]
        if type(value) is list and all(type(part) is str for part in value)
    }
    equivalent_keys = {
        tuple(value)
        for value in evidence["opaque_equivalent_business_keys"]
        if type(value) is list and all(type(part) is str for part in value)
    }
    if (
        len(deferred_keys) != len(evidence["deferred_business_keys"])
        or len(equivalent_keys) != len(evidence["opaque_equivalent_business_keys"])
        or deferred_keys.intersection(equivalent_keys)
    ):
        _fail("SUCCESSOR_OPAQUE_COMP_TYPE_EVIDENCE_INVALID")
    for observation in evidence["deferred_observations"]:
        if (
            type(observation) is not dict
            or set(observation) != {"business_key", "row_sha256", "typed_row"}
            or tuple(observation["business_key"]) not in deferred_keys
            or type(observation["typed_row"]) is not list
            or _sha256(_canonical_json_bytes(observation["typed_row"]))
            != observation["row_sha256"]
        ):
            _fail("SUCCESSOR_OPAQUE_COMP_TYPE_EVIDENCE_INVALID")
    for field in (
        "evidence_sha256",
        "opaque_comp_type_observation_multiset_sha256",
        "opaque_to_supported_peer_pair_keyset_sha256",
    ):
        _hex_sha256(evidence[field], label=field)
    return evidence


def _canonicalize_rows(
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    opaque_evidence = _opaque_comp_type_evidence(table, rows)
    deferred_keys = {
        tuple(value) for value in opaque_evidence["deferred_business_keys"]
    }
    fields = _output_fields(table)
    exact_seen: set[bytes] = set()
    exact_rows: list[dict[str, Any]] = []
    exact_collapsed = 0
    for value in rows:
        row = dict(value)
        token = _canonical_json_bytes(_typed_row(row, fields, logical=False))
        if token in exact_seen:
            exact_collapsed += 1
            continue
        exact_seen.add(token)
        exact_rows.append(row)
    physical_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in exact_rows:
        physical_key = (*_business_key(table, row), *_physical_update_identity(table, row))
        physical_groups.setdefault(physical_key, []).append(row)
    survivors: list[dict[str, Any]] = []
    dominated = 0
    for key in sorted(
        physical_groups,
        key=lambda item: tuple(part.encode("utf-8") for part in item),
    ):
        candidates = physical_groups[key]
        highest = max(_update_rank(table, row) for row in candidates)
        winners = [row for row in candidates if _update_rank(table, row) == highest]
        dominated += len(candidates) - len(winners)
        survivors.extend(winners)

    logical_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in survivors:
        logical_groups.setdefault(_business_key(table, row), []).append(row)
    accepted: list[dict[str, Any]] = []
    projection_collapsed = 0
    deferred = 0
    projection_fields = _projection_fields(table)
    for key in sorted(
        logical_groups,
        key=lambda item: tuple(part.encode("utf-8") for part in item),
    ):
        winners = logical_groups[key]
        projections: dict[bytes, list[dict[str, Any]]] = {}
        for row in winners:
            token = _canonical_json_bytes(_typed_row(row, projection_fields, logical=True))
            projections.setdefault(token, []).append(row)
        if len(projections) != 1:
            if any(
                str(row.get("comp_type") or "")
                == _OPAQUE_BALANCESHEET_COMP_TYPE
                for row in winners
            ):
                _fail("SUCCESSOR_OPAQUE_COMP_TYPE_EQUIVALENCE_UNCLOSED")
            _fail("SUCCESSOR_MATERIAL_DUPLICATE_CONFLICT")
        equivalent = next(iter(projections.values()))
        if key in deferred_keys:
            if any(
                str(row.get("comp_type") or "")
                != _OPAQUE_BALANCESHEET_COMP_TYPE
                for row in equivalent
            ):
                _fail("SUCCESSOR_OPAQUE_COMP_TYPE_CLASSIFICATION_UNCLOSED")
            deferred += len(equivalent)
            continue
        projection_collapsed += len(equivalent) - 1
        # All business values are equal here.  Physical classification is the
        # only tie-break input; no content hash selects a winner.
        supported = [
            row
            for row in equivalent
            if str(row.get("comp_type") or "")
            != _OPAQUE_BALANCESHEET_COMP_TYPE
        ]
        accepted.append(
            min(supported or equivalent, key=lambda row: _row_sort_key(row, fields))
        )
    accepted.sort(key=lambda row: _row_sort_key(row, fields))
    return accepted, {
        "exact_duplicates_collapsed": exact_collapsed,
        "projection_equivalent_duplicates_collapsed": projection_collapsed,
        "superseded_updates_discarded": dominated,
        "deferred_opaque_observations": deferred,
    }


def _response_identities(
    *,
    request: Mapping[str, Any],
    response: TushareResponse,
    normalized_rows: Sequence[Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    table = request["table"]
    fields = _output_fields(table)
    observation_rows = sorted(
        (_typed_row(row, fields, logical=False) for row in normalized_rows),
        key=_canonical_json_bytes,
    )
    payload_rows = sorted(
        (_typed_row(row, fields, logical=True) for row in normalized_rows),
        key=_canonical_json_bytes,
    )
    logical_values = [_typed_row(row, fields, logical=True) for row in logical_rows]
    request_id_sha256 = _sha256(response.request_id.encode("utf-8"))
    observation = {
        "fields": list(fields),
        "has_more": response.has_more,
        "item_count": response.item_count,
        "provider_request_id_sha256": request_id_sha256,
        "provider_reported_count": response.provider_reported_count,
        "rows": observation_rows,
    }
    payload = {"fields": list(fields), "rows": payload_rows}
    logical = {"fields": list(fields), "rows": logical_values}
    return (
        _sha256(_canonical_json_bytes(observation)),
        _sha256(_canonical_json_bytes(payload)),
        _sha256(_canonical_json_bytes(logical)),
    )


def _row_multiset_sha256(
    table: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    logical: bool,
) -> str:
    fields = _output_fields(table)
    encoded = sorted(
        (_typed_row(row, fields, logical=logical) for row in rows),
        key=_canonical_json_bytes,
    )
    return _sha256(_canonical_json_bytes({"fields": list(fields), "rows": encoded}))


def _row_order_sha256(
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    fields = _output_fields(table)
    encoded = [_typed_row(row, fields, logical=False) for row in rows]
    return _sha256(_canonical_json_bytes({"fields": list(fields), "rows": encoded}))


def _scope_partition_identity(
    *,
    table: str,
    normalized_rows: Sequence[Mapping[str, Any]],
    in_scope_rows: Sequence[Mapping[str, Any]],
    out_of_scope_rows: Sequence[Mapping[str, Any]],
    request_envelope_scope_ref: Mapping[str, Any],
    canonical_subject_scope_ref: Mapping[str, Any],
) -> dict[str, Any]:
    if len(normalized_rows) != len(in_scope_rows) + len(out_of_scope_rows):
        _fail("SUCCESSOR_SCOPE_PARTITION_NOT_RECONCILED")
    in_symbols = {str(row["ts_code"]) for row in in_scope_rows}
    out_symbols = {str(row["ts_code"]) for row in out_of_scope_rows}
    subject_symbols_sha256 = canonical_subject_scope_ref.get(
        "subject_symbol_keyset_sha256"
    )
    if in_symbols.intersection(out_symbols):
        _fail("SUCCESSOR_SCOPE_IDENTITY_COLLISION")
    excluded_symbols = sorted(out_symbols)
    identity = {
        "canonical_subject_scope_ref_sha256": canonical_subject_scope_ref.get(
            "scope_sha256"
        ),
        "full_response_observation_count": len(normalized_rows),
        "full_response_observation_multiset_sha256": _row_multiset_sha256(
            table, normalized_rows, logical=False
        ),
        "in_scope_canonical_payload_multiset_sha256": _row_multiset_sha256(
            table, in_scope_rows, logical=True
        ),
        "in_scope_observation_count": len(in_scope_rows),
        "in_scope_observation_multiset_sha256": _row_multiset_sha256(
            table, in_scope_rows, logical=False
        ),
        "out_of_scope_canonical_payload_multiset_sha256": _row_multiset_sha256(
            table, out_of_scope_rows, logical=True
        ),
        "out_of_scope_observation_count": len(out_of_scope_rows),
        "out_of_scope_observation_multiset_sha256": _row_multiset_sha256(
            table, out_of_scope_rows, logical=False
        ),
        "out_of_scope_symbol_count": len(excluded_symbols),
        "out_of_scope_symbol_keyset_sha256": _sha256(
            _canonical_json_bytes(excluded_symbols)
        ),
        "request_envelope_scope_ref_sha256": request_envelope_scope_ref.get(
            "scope_sha256"
        ),
        "scope_exclusion_policy": (
            "FROZEN_CANONICAL_SUBJECT_PROJECTION.v3_"
            "EXACT_PROVIDER_EXTERNAL_EVIDENCE_ONLY"
        ),
        "subject_symbol_keyset_sha256": subject_symbols_sha256,
    }
    for field in (
        "canonical_subject_scope_ref_sha256",
        "request_envelope_scope_ref_sha256",
        "subject_symbol_keyset_sha256",
    ):
        _hex_sha256(identity[field], label=field)
    identity["scope_partition_sha256"] = _sha256(_canonical_json_bytes(identity))
    return identity


def _classification_partition(
    *,
    table: str,
    normalized_rows: Sequence[Mapping[str, Any]],
    subject_symbols: frozenset[str],
    opaque_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal an exhaustive, mutually-exclusive raw-observation partition."""

    fields = _output_fields(table)
    deferred_keys = {
        tuple(value) for value in opaque_evidence["deferred_business_keys"]
    }
    equivalent_keys = {
        tuple(value)
        for value in opaque_evidence["opaque_equivalent_business_keys"]
    }
    names = (
        "out_of_scope_excluded",
        "authoritative_supported",
        "opaque_equivalent",
        "tainted_deferred",
        "source_blocking",
    )
    buckets: dict[str, list[list[dict[str, Any]]]] = {
        name: [] for name in names
    }
    for row in normalized_rows:
        encoded = _typed_row(row, fields, logical=False)
        if str(row["ts_code"]) not in subject_symbols:
            bucket = "out_of_scope_excluded"
        elif (
            table == "balancesheet"
            and str(row.get("comp_type") or "")
            == _OPAQUE_BALANCESHEET_COMP_TYPE
        ):
            key = _business_key(table, row)
            if key in deferred_keys:
                bucket = "tainted_deferred"
            elif key in equivalent_keys:
                bucket = "opaque_equivalent"
            else:
                _fail("SUCCESSOR_OBSERVATION_CLASSIFICATION_UNCLOSED")
        else:
            bucket = "authoritative_supported"
        buckets[bucket].append(encoded)
    for values in buckets.values():
        values.sort(key=_canonical_json_bytes)
    all_rows = sorted(
        (_typed_row(row, fields, logical=False) for row in normalized_rows),
        key=_canonical_json_bytes,
    )
    classified = sorted(
        (row for values in buckets.values() for row in values),
        key=_canonical_json_bytes,
    )
    if _canonical_json_bytes(classified) != _canonical_json_bytes(all_rows):
        _fail("SUCCESSOR_OBSERVATION_CLASSIFICATION_UNCLOSED")
    body = {
        "schema_version": FUNDAMENTAL_SUCCESSOR_CLASSIFICATION_PARTITION_SCHEMA,
        "table": table,
        "raw_observation_count": len(all_rows),
        "raw_observation_multiset_sha256": _sha256(
            _canonical_json_bytes({"fields": list(fields), "rows": all_rows})
        ),
        "classification_counts": {
            name: len(buckets[name]) for name in names
        },
        "classification_multiset_sha256": {
            name: _sha256(
                _canonical_json_bytes(
                    {"fields": list(fields), "rows": buckets[name]}
                )
            )
            for name in names
        },
        "partition_mutually_exclusive": True,
        "partition_union_complete": True,
    }
    return _sealed(body, identity_field="partition_sha256")


def _validate_classification_partition(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    partition = _validate_seal(value, identity_field="partition_sha256")
    names = {
        "out_of_scope_excluded",
        "authoritative_supported",
        "opaque_equivalent",
        "tainted_deferred",
        "source_blocking",
    }
    if (
        set(partition)
        != {
            "classification_counts",
            "classification_multiset_sha256",
            "partition_mutually_exclusive",
            "partition_sha256",
            "partition_union_complete",
            "raw_observation_count",
            "raw_observation_multiset_sha256",
            "schema_version",
            "table",
        }
        or partition["schema_version"]
        != FUNDAMENTAL_SUCCESSOR_CLASSIFICATION_PARTITION_SCHEMA
        or partition["table"] not in _TABLES
        or type(partition["classification_counts"]) is not dict
        or set(partition["classification_counts"]) != names
        or type(partition["classification_multiset_sha256"]) is not dict
        or set(partition["classification_multiset_sha256"]) != names
        or any(
            type(count) is not int or count < 0
            for count in partition["classification_counts"].values()
        )
        or partition["classification_counts"]["source_blocking"] != 0
        or sum(partition["classification_counts"].values())
        != partition["raw_observation_count"]
        or partition["partition_mutually_exclusive"] is not True
        or partition["partition_union_complete"] is not True
    ):
        _fail("SUCCESSOR_OBSERVATION_CLASSIFICATION_INVALID")
    _hex_sha256(
        partition["raw_observation_multiset_sha256"],
        label="raw_observation_multiset_sha256",
    )
    for digest in partition["classification_multiset_sha256"].values():
        _hex_sha256(digest, label="classification_multiset_sha256")
    return partition


def _stored_response_identities(
    *,
    table: str,
    provider_request_id_sha256: str,
    provider_reported_count: int,
    item_count: int,
    observed_rows: Sequence[Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    fields = _output_fields(table)
    observation_values = sorted(
        (_typed_row(row, fields, logical=False) for row in observed_rows),
        key=_canonical_json_bytes,
    )
    payload_values = sorted(
        (_typed_row(row, fields, logical=True) for row in observed_rows),
        key=_canonical_json_bytes,
    )
    logical_values = [_typed_row(row, fields, logical=True) for row in logical_rows]
    observation = {
        "fields": list(fields),
        "has_more": False,
        "item_count": item_count,
        "provider_request_id_sha256": provider_request_id_sha256,
        "provider_reported_count": provider_reported_count,
        "rows": observation_values,
    }
    payload = {"fields": list(fields), "rows": payload_values}
    logical = {"fields": list(fields), "rows": logical_values}
    return (
        _sha256(_canonical_json_bytes(observation)),
        _sha256(_canonical_json_bytes(payload)),
        _sha256(_canonical_json_bytes(logical)),
    )


class _Pacer:
    def __init__(
        self,
        requests_per_second: float,
        *,
        sleeper: Callable[[float], None],
        monotonic: Callable[[], float],
    ) -> None:
        if (
            type(requests_per_second) not in {int, float}
            or isinstance(requests_per_second, bool)
            or not math.isfinite(float(requests_per_second))
            or not 0 < float(requests_per_second) <= 8.0
        ):
            _fail("SUCCESSOR_PACING_POLICY_INVALID")
        self._interval = 1.0 / float(requests_per_second)
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._last_started: float | None = None

    def wait(self) -> None:
        now = float(self._monotonic())
        if not math.isfinite(now):
            _fail("SUCCESSOR_MONOTONIC_CLOCK_INVALID")
        if self._last_started is not None:
            delay = self._interval - (now - self._last_started)
            if delay > 0:
                self._sleeper(delay)
                now = float(self._monotonic())
                if not math.isfinite(now):
                    _fail("SUCCESSOR_MONOTONIC_CLOCK_INVALID")
        self._last_started = now


def _error_code(error: BaseException) -> str:
    if isinstance(error, TushareHttpsError):
        return error.code
    if isinstance(error, FundamentalSuccessorSourceError):
        return error.code
    return "SUCCESSOR_PROVIDER_CALL_FAILED"


def _validate_provider_response_envelope(
    response: TushareResponse,
    *,
    request: Mapping[str, Any],
) -> None:
    fields = tuple(request["expected_fields"])
    if (
        response.api_name != request["endpoint"]
        or response.fields != fields
        or type(response.provider_reported_count) is not int
        or type(response.item_count) is not int
        or response.item_count != len(response.rows)
        or response.provider_reported_count not in {0, response.item_count}
        or type(response.has_more) is not bool
        or type(response.request_id) is not str
        or not response.request_id
    ):
        _fail("SUCCESSOR_PROVIDER_SCHEMA_MISMATCH")
    if response.has_more:
        _fail("SUCCESSOR_PROVIDER_HAS_MORE")
    if len(response.rows) >= request["row_ceiling"]:
        _fail("SUCCESSOR_PROVIDER_ROW_CEILING_HIT")


def _receipt(
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    response: TushareResponse,
    normalized_rows: Sequence[Mapping[str, Any]],
    in_scope_rows: Sequence[Mapping[str, Any]],
    out_of_scope_rows: Sequence[Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
    raw_response_bytes: bytes,
    attempts: int,
    retry_error_codes: Sequence[str],
    counters: Mapping[str, int],
    opaque_comp_type_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    observation_sha256, payload_sha256, logical_sha256 = _response_identities(
        request=request,
        response=response,
        normalized_rows=normalized_rows,
        logical_rows=logical_rows,
    )
    scope_identity = _scope_partition_identity(
        table=request["table"],
        normalized_rows=normalized_rows,
        in_scope_rows=in_scope_rows,
        out_of_scope_rows=out_of_scope_rows,
        request_envelope_scope_ref=plan["request_envelope_scope_ref"],
        canonical_subject_scope_ref=plan["canonical_subject_scope_ref"],
    )
    classification = _classification_partition(
        table=request["table"],
        normalized_rows=normalized_rows,
        subject_symbols=frozenset(plan["subject_symbols"]),
        opaque_evidence=opaque_comp_type_evidence,
    )
    body = {
        "accepted_count": len(logical_rows),
        "attempts": attempts,
        "blocker_codes": [],
        "canonicalization_counters": dict(counters),
        "classification_partition": classification,
        "endpoint": request["endpoint"],
        "has_more": False,
        "item_count": response.item_count,
        "logical_sha256": logical_sha256,
        "observation_sha256": observation_sha256,
        "opaque_comp_type_evidence": dict(opaque_comp_type_evidence),
        "ordinal": request["ordinal"],
        "payload_sha256": payload_sha256,
        "plan_sha256": plan["plan_sha256"],
        "provider_count_policy": "ZERO_SENTINEL_OR_EXACT_ITEM_COUNT.v1",
        "provider_reported_count": response.provider_reported_count,
        "provider_request_id_sha256": _sha256(response.request_id.encode("utf-8")),
        "raw_item_order_sha256": _row_order_sha256(
            request["table"], normalized_rows
        ),
        "raw_response_byte_length": len(raw_response_bytes),
        "raw_response_sha256": _sha256(raw_response_bytes),
        "request_key": request["request_key"],
        "retry_error_codes": list(retry_error_codes),
        **scope_identity,
        "schema_version": FUNDAMENTAL_SUCCESSOR_REQUEST_RECEIPT_SCHEMA,
        "status": "EMPTY" if not logical_rows else "AVAILABLE",
        "table": request["table"],
    }
    return _sealed(body, identity_field="receipt_sha256")


def _fetch_request(
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    client: SuccessorTushareClient,
    symbols: frozenset[str],
    max_attempts: int,
    retry_backoff_seconds: Sequence[float],
    pacer: _Pacer,
    sleeper: Callable[[float], None],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    bytes,
]:
    retry_errors: list[str] = []
    response: TushareResponse | None = None
    for attempt in range(1, max_attempts + 1):
        pacer.wait()
        try:
            candidate = client.request(
                api_name=request["endpoint"],
                params=request["params"],
                expected_fields=request["expected_fields"],
            )
        except BaseException as error:
            code = _error_code(error)
            if code in {
                "TUSHARE_API_ERROR",
                "TUSHARE_CLIENT_CONFIG_INVALID",
                "TUSHARE_ENDPOINT_BLOCKED",
                "TUSHARE_REQUEST_INVALID",
                "TUSHARE_RESPONSE_INVALID",
                "TUSHARE_TOKEN_MISSING",
            }:
                _fail("SUCCESSOR_PROVIDER_REQUEST_FAILED")
            retry_errors.append(code)
            if attempt == max_attempts:
                _fail("SUCCESSOR_PROVIDER_REQUEST_FAILED")
            sleeper(float(retry_backoff_seconds[attempt - 1]))
            continue
        if not isinstance(candidate, TushareResponse):
            _fail("SUCCESSOR_PROVIDER_RESPONSE_TYPE_INVALID")
        response = candidate
        break
    if response is None:  # pragma: no cover - loop either succeeds or raises
        _fail("SUCCESSOR_PROVIDER_REQUEST_FAILED")
    raw_response_bytes = response.raw_body
    if type(raw_response_bytes) is not bytes or not raw_response_bytes:
        _fail("SUCCESSOR_PROVIDER_RAW_RESPONSE_MISSING")
    fields = tuple(request["expected_fields"])
    _validate_provider_response_envelope(response, request=request)
    try:
        normalized = [
            _normalize_response_row(
                table=request["table"],
                request=request,
                fields=fields,
                values=row,
                symbols=symbols,
                target_date=plan["target_date"],
                enforce_subject_scope=False,
            )
            for row in response.rows
        ]
        in_scope = [
            row for row in normalized if str(row["ts_code"]) in symbols
        ]
        scope_excluded = [
            row for row in normalized if str(row["ts_code"]) not in symbols
        ]
        opaque_comp_type_evidence = _opaque_comp_type_evidence(
            request["table"], in_scope
        )
        logical, counters = _canonicalize_rows(request["table"], in_scope)
    except FundamentalSuccessorSourceError as exc:
        raise FundamentalSuccessorSourceError(
            exc.code,
            response_evidence={
                "api_name": response.api_name,
                "fields": list(response.fields),
                "has_more": response.has_more,
                "item_count": response.item_count,
                "provider_reported_count": response.provider_reported_count,
                "request_id": response.request_id,
                "raw_response_byte_length": len(raw_response_bytes),
                "raw_response_sha256": _sha256(raw_response_bytes),
            },
            raw_response_bytes=raw_response_bytes,
        ) from None
    receipt = _receipt(
        plan=plan,
        request=request,
        response=response,
        normalized_rows=normalized,
        in_scope_rows=in_scope,
        out_of_scope_rows=scope_excluded,
        logical_rows=logical,
        raw_response_bytes=raw_response_bytes,
        attempts=attempt,
        retry_error_codes=retry_errors,
        counters=counters,
        opaque_comp_type_evidence=opaque_comp_type_evidence,
    )
    return receipt, normalized, logical, raw_response_bytes


def _assert_no_secret_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = key.casefold() if type(key) is str else ""
            if type(key) is not str or any(
                fragment in folded for fragment in _SECRET_KEY_FRAGMENTS
            ):
                _fail("SUCCESSOR_IMMUTABLE_REF_INVALID")
            _assert_no_secret_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_secret_keys(child)
    elif value is None or type(value) in {bool, int, str}:
        return
    else:
        _fail("SUCCESSOR_IMMUTABLE_REF_INVALID")


def _captured_pointers(values: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    if type(values) is not dict or set(values) != _POINTER_NAMES:
        _fail("SUCCESSOR_CAPTURED_POINTER_SET_INVALID")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(_POINTER_NAMES):
        payload = values[name]
        if type(payload) is not bytes or not payload:
            _fail("SUCCESSOR_CAPTURED_POINTER_BYTES_INVALID")
        result[name] = {
            "byte_length": len(payload),
            "bytes_base64": base64.b64encode(payload).decode("ascii"),
            "sha256": _sha256(payload),
        }
    return result


def _validate_captured_pointers(value: Any) -> dict[str, dict[str, Any]]:
    if type(value) is not dict or set(value) != _POINTER_NAMES:
        _fail("SUCCESSOR_CAPTURED_POINTER_SET_INVALID")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(_POINTER_NAMES):
        row = value[name]
        if type(row) is not dict or set(row) != {
            "byte_length",
            "bytes_base64",
            "sha256",
        }:
            _fail("SUCCESSOR_CAPTURED_POINTER_EVIDENCE_INVALID")
        if type(row["byte_length"]) is not int or row["byte_length"] < 1:
            _fail("SUCCESSOR_CAPTURED_POINTER_EVIDENCE_INVALID")
        if type(row["bytes_base64"]) is not str:
            _fail("SUCCESSOR_CAPTURED_POINTER_EVIDENCE_INVALID")
        try:
            payload = base64.b64decode(row["bytes_base64"], validate=True)
        except Exception:
            _fail("SUCCESSOR_CAPTURED_POINTER_EVIDENCE_INVALID")
        if (
            len(payload) != row["byte_length"]
            or _sha256(payload) != row["sha256"]
            or _HEX_SHA256_RE.fullmatch(str(row["sha256"])) is None
        ):
            _fail("SUCCESSOR_CAPTURED_POINTER_EVIDENCE_INVALID")
        result[name] = dict(row)
    return result


def _private_root(path: str | Path, *, create: bool) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        _fail("SUCCESSOR_FILESET_ROOT_INVALID")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if current != candidate or not create:
                _fail("SUCCESSOR_FILESET_ROOT_INVALID")
            try:
                os.mkdir(current, mode=0o700)
            except OSError:
                _fail("SUCCESSOR_FILESET_ROOT_INVALID")
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            _fail("SUCCESSOR_FILESET_SYMLINK_BLOCKED")
        if current != candidate and not stat.S_ISDIR(metadata.st_mode):
            _fail("SUCCESSOR_FILESET_ROOT_INVALID")
    try:
        metadata = os.lstat(candidate)
    except OSError:
        _fail("SUCCESSOR_FILESET_ROOT_INVALID")
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        _fail("SUCCESSOR_FILESET_ROOT_NOT_PRIVATE")
    return candidate


def _safe_directory(root: Path, relative: str, *, create: bool) -> Path:
    path = root / relative
    if path.exists():
        metadata = os.lstat(path)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            _fail("SUCCESSOR_FILESET_ENTRY_INVALID")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            _fail("SUCCESSOR_FILESET_ENTRY_NOT_PRIVATE")
        return path
    if not create:
        _fail("SUCCESSOR_FILESET_ENTRY_INVALID")
    try:
        os.mkdir(path, mode=0o700)
    except OSError:
        _fail("SUCCESSOR_FILESET_ENTRY_INVALID")
    return path


def _regular_bytes(path: Path) -> bytes:
    try:
        before = os.lstat(path)
    except OSError:
        _fail("SUCCESSOR_EVIDENCE_FILE_MISSING")
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        _fail("SUCCESSOR_EVIDENCE_FILE_INVALID")
    try:
        with path.open("rb") as handle:
            payload = handle.read()
            descriptor = os.fstat(handle.fileno())
    except OSError:
        _fail("SUCCESSOR_EVIDENCE_FILE_INVALID")
    after = os.lstat(path)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_descriptor = (
        descriptor.st_dev,
        descriptor.st_ino,
        descriptor.st_size,
        descriptor.st_mtime_ns,
    )
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_descriptor or identity_before != identity_after:
        _fail("SUCCESSOR_EVIDENCE_FILE_CHANGED")
    return payload


def _regular_file_identity(path: Path) -> tuple[str, int]:
    """Hash one private regular file without materialising it in memory."""

    try:
        before = os.lstat(path)
    except OSError:
        _fail("SUCCESSOR_EVIDENCE_FILE_MISSING")
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        _fail("SUCCESSOR_EVIDENCE_FILE_INVALID")
    digest = hashlib.sha256()
    observed = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            opened = os.fstat(handle.fileno())
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                observed += len(chunk)
    except OSError:
        _fail("SUCCESSOR_EVIDENCE_FILE_INVALID")
    finally:
        if "descriptor" in locals() and descriptor >= 0:
            os.close(descriptor)
    try:
        after = os.lstat(path)
    except OSError:
        _fail("SUCCESSOR_EVIDENCE_FILE_CHANGED")
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(opened) or identity(before) != identity(after):
        _fail("SUCCESSOR_EVIDENCE_FILE_CHANGED")
    if observed != int(before.st_size):
        _fail("SUCCESSOR_EVIDENCE_FILE_CHANGED")
    return digest.hexdigest(), observed


def _canonical_file_mapping(path: Path) -> dict[str, Any]:
    payload = _regular_bytes(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        _fail("SUCCESSOR_EVIDENCE_JSON_INVALID")
    if type(value) is not dict or _canonical_json_bytes(value) != payload:
        _fail("SUCCESSOR_EVIDENCE_JSON_NON_CANONICAL")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        if _regular_bytes(path) != payload:
            _fail("SUCCESSOR_IMMUTABLE_FILE_CONFLICT")
        return
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FundamentalSuccessorSourceError:
        raise
    except OSError:
        _fail("SUCCESSOR_ATOMIC_WRITE_FAILED")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    if _regular_bytes(path) != payload:
        _fail("SUCCESSOR_ATOMIC_WRITE_READBACK_FAILED")


def _persist_failure_evidence(
    *,
    root: Path,
    binding: Mapping[str, Any],
    request: Mapping[str, Any],
    error: FundamentalSuccessorSourceError,
    maximum_record_bytes: int,
) -> None:
    raw_response_bytes = error.raw_response_bytes
    response_evidence = error.response_evidence
    if raw_response_bytes is None or response_evidence is None:
        return
    if not raw_response_bytes or len(raw_response_bytes) > maximum_record_bytes:
        _fail("SUCCESSOR_FAILURE_EVIDENCE_RESOURCE_LIMIT_EXCEEDED")
    ordinal = int(request["ordinal"])
    failure_root = _private_root(
        root.parent / f"{root.name}-failures",
        create=True,
    )
    raw_name = f"{ordinal:06d}.raw.json"
    raw_path = failure_root / raw_name
    _atomic_write(raw_path, raw_response_bytes)
    raw_readback = _regular_bytes(raw_path)
    failure = _sealed(
        {
            "schema_version": FUNDAMENTAL_SUCCESSOR_FAILURE_EVIDENCE_SCHEMA,
            "status": "BLOCKED",
            "error_code": error.code,
            "binding_sha256": binding["binding_sha256"],
            "request": dict(request),
            "response_evidence": dict(response_evidence),
            "raw_response_ref": {
                "path": raw_name,
                "byte_length": len(raw_readback),
                "sha256": _sha256(raw_readback),
            },
        },
        identity_field="failure_sha256",
    )
    _assert_no_secret_keys(failure)
    _atomic_write(
        failure_root / f"{ordinal:06d}.failure.json",
        _canonical_json_bytes(failure),
    )
    validate_successor_failure_evidence(failure_root, ordinal=ordinal)


def validate_successor_failure_evidence(
    failure_root: str | Path,
    *,
    ordinal: int,
) -> dict[str, Any]:
    """Validate one blocked provider response without making it resumable."""

    if type(ordinal) is not int or ordinal < 0:
        _fail("SUCCESSOR_FAILURE_EVIDENCE_ORDINAL_INVALID")
    root = _private_root(failure_root, create=False)
    failure = _validate_seal(
        _canonical_file_mapping(root / f"{ordinal:06d}.failure.json"),
        identity_field="failure_sha256",
    )
    required = {
        "binding_sha256",
        "error_code",
        "failure_sha256",
        "raw_response_ref",
        "request",
        "response_evidence",
        "schema_version",
        "status",
    }
    if (
        set(failure) != required
        or failure["schema_version"]
        != FUNDAMENTAL_SUCCESSOR_FAILURE_EVIDENCE_SCHEMA
        or failure["status"] != "BLOCKED"
        or type(failure["error_code"]) is not str
        or re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", failure["error_code"])
        is None
    ):
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    _hex_sha256(failure["binding_sha256"], label="binding_sha256")
    request = failure["request"]
    response_evidence = failure["response_evidence"]
    raw_ref = failure["raw_response_ref"]
    if (
        not isinstance(request, Mapping)
        or request.get("ordinal") != ordinal
        or not isinstance(response_evidence, Mapping)
        or type(raw_ref) is not dict
        or set(raw_ref) != {"byte_length", "path", "sha256"}
        or raw_ref["path"] != f"{ordinal:06d}.raw.json"
    ):
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    if not root.name.endswith("-failures"):
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    fileset_name = root.name.removesuffix("-failures")
    if not fileset_name:
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    try:
        fileset_root = _private_root(
            root.parent / fileset_name,
            create=False,
        )
        binding = _validate_binding(
            _canonical_file_mapping(fileset_root / "binding.json")
        )
        planned_requests = replay_successor_support_requests(binding["plan"])
    except FundamentalSuccessorSourceError:
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    if (
        ordinal >= len(planned_requests)
        or failure["binding_sha256"] != binding["binding_sha256"]
        or _canonical_json_bytes(request)
        != _canonical_json_bytes(planned_requests[ordinal])
    ):
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    raw_bytes = _regular_bytes(root / raw_ref["path"])
    if (
        type(raw_ref["byte_length"]) is not int
        or raw_ref["byte_length"] != len(raw_bytes)
        or _hex_sha256(raw_ref["sha256"], label="raw_response_sha256")
        != _sha256(raw_bytes)
    ):
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    expected_response_fields = {
        "api_name",
        "fields",
        "has_more",
        "item_count",
        "provider_reported_count",
        "request_id",
        "raw_response_byte_length",
        "raw_response_sha256",
    }
    if set(response_evidence) != expected_response_fields:
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    try:
        replayed = replay_tushare_response_bytes(
            raw_bytes,
            api_name=request["endpoint"],
            expected_fields=request["expected_fields"],
            strict_decimal_decode=True,
            max_response_items=request["row_ceiling"],
        )
    except (KeyError, TypeError, TushareHttpsError):
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    try:
        _validate_provider_response_envelope(replayed, request=request)
    except FundamentalSuccessorSourceError:
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    expected_response = {
        "api_name": replayed.api_name,
        "fields": list(replayed.fields),
        "has_more": replayed.has_more,
        "item_count": replayed.item_count,
        "provider_reported_count": replayed.provider_reported_count,
        "request_id": replayed.request_id,
        "raw_response_byte_length": len(raw_bytes),
        "raw_response_sha256": _sha256(raw_bytes),
    }
    if _canonical_json_bytes(response_evidence) != _canonical_json_bytes(
        expected_response
    ):
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    subject_symbols = frozenset(binding["plan"]["subject_symbols"])
    try:
        normalized = [
            _normalize_response_row(
                table=request["table"],
                request=request,
                fields=request["expected_fields"],
                values=row,
                symbols=subject_symbols,
                target_date=binding["plan"]["target_date"],
                enforce_subject_scope=False,
            )
            for row in replayed.rows
        ]
        in_scope = [
            row
            for row in normalized
            if str(row["ts_code"]) in subject_symbols
        ]
        _opaque_comp_type_evidence(request["table"], in_scope)
        _canonicalize_rows(request["table"], in_scope)
    except FundamentalSuccessorSourceError as replay_error:
        if replay_error.code != failure["error_code"]:
            _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    else:
        _fail("SUCCESSOR_FAILURE_EVIDENCE_INVALID")
    return failure


def _physical_memory_bytes() -> int:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, TypeError, ValueError):
        _fail("SUCCESSOR_PHYSICAL_MEMORY_UNAVAILABLE")
    total = page_size * page_count
    if page_size < 1 or page_count < 1 or total < 1024 * 1024 * 1024:
        _fail("SUCCESSOR_PHYSICAL_MEMORY_INVALID")
    return total


def _available_memory_bytes() -> int:
    """Return currently available memory without treating total RAM as headroom."""

    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    value = int(line.split()[1]) * 1024
                    if value > 0:
                        return value
        except (OSError, UnicodeError, ValueError, IndexError):
            pass
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["/usr/bin/vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            first, *lines = completed.stdout.splitlines()
            match = re.search(r"page size of ([0-9]+) bytes", first)
            if match is None:
                _fail("SUCCESSOR_AVAILABLE_MEMORY_UNAVAILABLE")
            page_size = int(match.group(1))
            counts: dict[str, int] = {}
            for line in lines:
                if ":" not in line:
                    continue
                name, raw = line.split(":", 1)
                value = raw.strip().rstrip(".")
                if value.isdigit():
                    counts[name] = int(value)
            pages = sum(
                counts.get(name, 0)
                for name in (
                    "Pages free",
                    "Pages inactive",
                    "Pages speculative",
                    "Pages purgeable",
                )
            )
            available = pages * page_size
            if available > 0:
                return available
        except FundamentalSuccessorSourceError:
            raise
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    _fail("SUCCESSOR_AVAILABLE_MEMORY_UNAVAILABLE")


def _resident_memory_bytes() -> int:
    try:
        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        _fail("SUCCESSOR_RESIDENT_MEMORY_UNAVAILABLE")
    if usage < 0:
        _fail("SUCCESSOR_RESIDENT_MEMORY_UNAVAILABLE")
    return usage if sys.platform == "darwin" else usage * 1024


def _rlimit_headroom_bytes(*, resident_memory_bytes: int) -> int:
    candidates: list[int] = []
    for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
        limit_id = getattr(resource, limit_name, None)
        if limit_id is None:
            continue
        try:
            soft, _hard = resource.getrlimit(limit_id)
        except (OSError, ValueError):
            _fail("SUCCESSOR_MEMORY_RLIMIT_UNAVAILABLE")
        if soft not in {resource.RLIM_INFINITY, -1}:
            candidates.append(max(0, int(soft) - resident_memory_bytes))
    return min(candidates) if candidates else 2**63 - 1


def _resource_policy(
    *,
    physical_memory_bytes: int,
    available_memory_bytes: int,
    rlimit_headroom_bytes: int,
    table_memory_limit_bytes: int,
    minimum_free_disk_bytes: int,
    maximum_record_bytes: int,
) -> dict[str, Any]:
    values = (
        physical_memory_bytes,
        available_memory_bytes,
        rlimit_headroom_bytes,
        table_memory_limit_bytes,
        minimum_free_disk_bytes,
        maximum_record_bytes,
    )
    if any(type(value) is not int or value < 1 for value in values):
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    effective_headroom = min(
        physical_memory_bytes,
        available_memory_bytes,
        rlimit_headroom_bytes,
    )
    maximum_table_memory = int(Decimal(effective_headroom) * _MAX_TABLE_MEMORY_FRACTION)
    if (
        effective_headroom < 2 * _MAX_STREAM_BATCH_BYTES
        or table_memory_limit_bytes > maximum_table_memory
        or maximum_record_bytes > min(table_memory_limit_bytes, _MAX_STREAM_BATCH_BYTES * 2)
        or minimum_free_disk_bytes < 64 * 1024 * 1024
    ):
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    return {
        "available_memory_bytes": available_memory_bytes,
        "effective_memory_headroom_bytes": effective_headroom,
        "maximum_stream_batch_bytes": _MAX_STREAM_BATCH_BYTES,
        "maximum_stream_batch_rows": _MAX_STREAM_BATCH_ROWS,
        "decode_estimated_bytes_per_cell": _DECODE_ESTIMATED_BYTES_PER_CELL,
        "maximum_record_bytes": maximum_record_bytes,
        "minimum_free_disk_bytes": minimum_free_disk_bytes,
        "parquet_row_group_rows": _PARQUET_ROW_GROUP_ROWS,
        "physical_memory_bytes": physical_memory_bytes,
        "rlimit_headroom_bytes": rlimit_headroom_bytes,
        "schema_version": "myquant-fundamental-successor-resource-policy.v2",
        "table_memory_limit_bytes": table_memory_limit_bytes,
    }


def _validate_resource_policy(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "available_memory_bytes",
        "effective_memory_headroom_bytes",
        "maximum_stream_batch_bytes",
        "maximum_stream_batch_rows",
        "decode_estimated_bytes_per_cell",
        "maximum_record_bytes",
        "minimum_free_disk_bytes",
        "parquet_row_group_rows",
        "physical_memory_bytes",
        "rlimit_headroom_bytes",
        "schema_version",
        "table_memory_limit_bytes",
    }:
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    if (
        value["schema_version"]
        != "myquant-fundamental-successor-resource-policy.v2"
        or value["decode_estimated_bytes_per_cell"]
        != _DECODE_ESTIMATED_BYTES_PER_CELL
        or value["maximum_stream_batch_rows"] != _MAX_STREAM_BATCH_ROWS
        or value["maximum_stream_batch_bytes"] != _MAX_STREAM_BATCH_BYTES
        or value["parquet_row_group_rows"] != _PARQUET_ROW_GROUP_ROWS
    ):
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    expected = _resource_policy(
        physical_memory_bytes=value["physical_memory_bytes"],
        available_memory_bytes=value["available_memory_bytes"],
        rlimit_headroom_bytes=value["rlimit_headroom_bytes"],
        table_memory_limit_bytes=value["table_memory_limit_bytes"],
        minimum_free_disk_bytes=value["minimum_free_disk_bytes"],
        maximum_record_bytes=value["maximum_record_bytes"],
    )
    if expected != value:
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    return expected


def _resume_binding_matches(
    installed: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    def static(value: Mapping[str, Any]) -> dict[str, Any]:
        body = dict(value)
        body.pop("binding_sha256", None)
        policy = dict(body.pop("resource_policy", {}) or {})
        return {
            "binding": body,
            "resource_policy": {
                key: policy.get(key)
                for key in (
                    "decode_estimated_bytes_per_cell",
                    "maximum_record_bytes",
                    "maximum_stream_batch_bytes",
                    "maximum_stream_batch_rows",
                    "minimum_free_disk_bytes",
                    "parquet_row_group_rows",
                    "physical_memory_bytes",
                    "schema_version",
                )
            },
        }

    return _canonical_json_bytes(static(installed)) == _canonical_json_bytes(
        static(current)
    )


def _require_disk_reserve(
    root: Path,
    *,
    minimum_free_disk_bytes: int,
    pending_bytes: int = 0,
) -> int:
    if type(pending_bytes) is not int or pending_bytes < 0:
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    try:
        free = int(shutil.disk_usage(root).free)
    except OSError:
        _fail("SUCCESSOR_DISK_PREFLIGHT_FAILED")
    # Atomic writes temporarily require both the new payload and filesystem
    # bookkeeping.  Two payload lengths plus the sealed 25%-style reserve is
    # deliberately conservative and is rechecked before every material write.
    required = minimum_free_disk_bytes + pending_bytes * 2
    if free < required:
        _fail("SUCCESSOR_DISK_RESERVE_EXHAUSTED")
    return free


def _external_sort_budget(
    *,
    record_bytes: int,
    accepted_rows: int,
) -> dict[str, int]:
    if (
        type(record_bytes) is not int
        or record_bytes < 0
        or type(accepted_rows) is not int
        or accepted_rows < 0
    ):
        _fail("SUCCESSOR_EXTERNAL_SORT_BUDGET_INVALID")
    sqlite_store = max(
        8 * 1024 * 1024,
        record_bytes * 6 + accepted_rows * 512,
    )
    sqlite_indexes = max(8 * 1024 * 1024, sqlite_store * 2)
    sqlite_journal = max(8 * 1024 * 1024, sqlite_store)
    parquet_temp = max(8 * 1024 * 1024, record_bytes * 2 + accepted_rows * 256)
    fsync_reserve = 64 * 1024 * 1024
    subtotal = (
        sqlite_store
        + sqlite_indexes
        + sqlite_journal
        + parquet_temp
        + fsync_reserve
    )
    margin = (subtotal + 3) // 4
    return {
        "fsync_reserve_bytes": fsync_reserve,
        "margin_25_percent_bytes": margin,
        "parquet_temp_bytes": parquet_temp,
        "sqlite_index_bytes": sqlite_indexes,
        "sqlite_journal_bytes": sqlite_journal,
        "sqlite_store_bytes": sqlite_store,
        "total_bytes": subtotal + margin,
    }


def _require_external_sort_reserve(
    root: Path,
    *,
    minimum_free_disk_bytes: int,
    required_working_bytes: int,
) -> int:
    if (
        type(minimum_free_disk_bytes) is not int
        or minimum_free_disk_bytes < 1
        or type(required_working_bytes) is not int
        or required_working_bytes < 0
    ):
        _fail("SUCCESSOR_EXTERNAL_SORT_BUDGET_INVALID")
    try:
        free = int(shutil.disk_usage(root).free)
    except OSError:
        _fail("SUCCESSOR_DISK_PREFLIGHT_FAILED")
    if free < minimum_free_disk_bytes + required_working_bytes:
        _fail("SUCCESSOR_EXTERNAL_SORT_DISK_RESERVE_EXHAUSTED")
    return free


def _binding(
    *,
    plan: Mapping[str, Any],
    captured_pointer_bytes: Mapping[str, bytes],
    immutable_refs: Mapping[str, Any],
    implementation_sha256: str,
    captured_at: str,
    max_attempts: int,
    retry_backoff_seconds: Sequence[float],
    requests_per_second: float,
    resource_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(immutable_refs, Mapping):
        _fail("SUCCESSOR_IMMUTABLE_REF_INVALID")
    refs = dict(immutable_refs)
    _assert_no_secret_keys(refs)
    _canonical_json_bytes(refs)
    body = {
        "canonicalization_policy": FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY,
        "captured_at": _timestamp(captured_at, label="captured_at"),
        "captured_pointers": _captured_pointers(dict(captured_pointer_bytes)),
        "execution_policy": {
            "max_attempts": max_attempts,
            "requests_per_second": float(requests_per_second),
            "retry_backoff_seconds": [float(value) for value in retry_backoff_seconds],
        },
        "immutable_refs": refs,
        "implementation_sha256": _hex_sha256(implementation_sha256, label="implementation_sha256"),
        "plan": dict(plan),
        "plan_sha256": plan["plan_sha256"],
        "resource_policy": dict(resource_policy),
        "schema_version": FUNDAMENTAL_SUCCESSOR_BINDING_SCHEMA,
    }
    return _sealed(body, identity_field="binding_sha256")


def _validate_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = _validate_seal(value, identity_field="binding_sha256")
    required = {
        "binding_sha256",
        "canonicalization_policy",
        "captured_at",
        "captured_pointers",
        "execution_policy",
        "immutable_refs",
        "implementation_sha256",
        "plan",
        "plan_sha256",
        "resource_policy",
        "schema_version",
    }
    if set(binding) != required:
        _fail("SUCCESSOR_BINDING_FIELDS_INVALID")
    if (
        binding["schema_version"] != FUNDAMENTAL_SUCCESSOR_BINDING_SCHEMA
        or binding["canonicalization_policy"] != FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY
    ):
        _fail("SUCCESSOR_BINDING_CONTRACT_MISMATCH")
    plan = dict(binding["plan"])
    replay_successor_support_requests(plan)
    if binding["plan_sha256"] != plan["plan_sha256"]:
        _fail("SUCCESSOR_BINDING_PLAN_MISMATCH")
    _timestamp(binding["captured_at"], label="captured_at")
    _hex_sha256(binding["implementation_sha256"], label="implementation_sha256")
    _validate_captured_pointers(binding["captured_pointers"])
    _assert_no_secret_keys(binding["immutable_refs"])
    _validate_resource_policy(binding["resource_policy"])
    policy = binding["execution_policy"]
    if type(policy) is not dict or set(policy) != {
        "max_attempts",
        "requests_per_second",
        "retry_backoff_seconds",
    }:
        _fail("SUCCESSOR_EXECUTION_POLICY_INVALID")
    _validate_execution_policy(
        max_attempts=policy["max_attempts"],
        retry_backoff_seconds=policy["retry_backoff_seconds"],
        requests_per_second=policy["requests_per_second"],
    )
    return binding


def _validate_execution_policy(
    *,
    max_attempts: Any,
    retry_backoff_seconds: Any,
    requests_per_second: Any,
) -> tuple[int, tuple[float, ...], float]:
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        _fail("SUCCESSOR_RETRY_POLICY_INVALID")
    if (
        isinstance(retry_backoff_seconds, (str, bytes))
        or not isinstance(retry_backoff_seconds, Sequence)
        or len(retry_backoff_seconds) != max_attempts - 1
    ):
        _fail("SUCCESSOR_RETRY_POLICY_INVALID")
    backoffs: list[float] = []
    for value in retry_backoff_seconds:
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 60
        ):
            _fail("SUCCESSOR_RETRY_POLICY_INVALID")
        backoffs.append(float(value))
    if (
        type(requests_per_second) not in {int, float}
        or isinstance(requests_per_second, bool)
        or not math.isfinite(float(requests_per_second))
        or not 0 < float(requests_per_second) <= 8.0
    ):
        _fail("SUCCESSOR_PACING_POLICY_INVALID")
    return max_attempts, tuple(backoffs), float(requests_per_second)


def _record(
    *,
    binding_sha256: str,
    request: Mapping[str, Any],
    receipt: Mapping[str, Any],
    observed_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    raw_response_bytes: bytes,
) -> dict[str, Any]:
    fields = _output_fields(request["table"])
    encoded_observed = sorted(
        (_typed_row(row, fields, logical=False) for row in observed_rows),
        key=_canonical_json_bytes,
    )
    body = {
        "binding_sha256": binding_sha256,
        "fields": list(fields),
        "observed_rows": encoded_observed,
        "receipt": dict(receipt),
        "raw_response_bytes_base64": base64.b64encode(raw_response_bytes).decode("ascii"),
        "raw_response_byte_length": len(raw_response_bytes),
        "raw_response_sha256": _sha256(raw_response_bytes),
        "request": dict(request),
        "rows": [_typed_row(row, fields, logical=False) for row in rows],
        "schema_version": FUNDAMENTAL_SUCCESSOR_RECORD_SCHEMA,
        "table": request["table"],
    }
    return _sealed(body, identity_field="record_sha256")


def _decode_record(
    value: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    expected_request: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    record = _validate_seal(value, identity_field="record_sha256")
    required = {
        "binding_sha256",
        "fields",
        "observed_rows",
        "raw_response_bytes_base64",
        "raw_response_byte_length",
        "raw_response_sha256",
        "receipt",
        "record_sha256",
        "request",
        "rows",
        "schema_version",
        "table",
    }
    if set(record) != required or record["schema_version"] != FUNDAMENTAL_SUCCESSOR_RECORD_SCHEMA:
        _fail("SUCCESSOR_RECORD_FIELDS_INVALID")
    if (
        record["binding_sha256"] != binding["binding_sha256"]
        or _canonical_json_bytes(record["request"]) != _canonical_json_bytes(expected_request)
        or record["table"] != expected_request["table"]
    ):
        _fail("SUCCESSOR_RECORD_BINDING_MISMATCH")
    fields = _output_fields(record["table"])
    if (
        record["fields"] != list(fields)
        or type(record["observed_rows"]) is not list
        or type(record["rows"]) is not list
    ):
        _fail("SUCCESSOR_RECORD_ROWS_INVALID")

    def decode_rows(values: Sequence[Any]) -> list[dict[str, Any]]:
        decoded: list[dict[str, Any]] = []
        for encoded in values:
            if type(encoded) is not list or len(encoded) != len(fields):
                _fail("SUCCESSOR_RECORD_ROWS_INVALID")
            decoded.append(
                dict(
                    zip(
                        fields,
                        (_decode_typed_scalar(value) for value in encoded),
                        strict=True,
                    )
                )
            )
        return decoded

    observed_rows = decode_rows(record["observed_rows"])
    rows = decode_rows(record["rows"])
    if (
        type(record["raw_response_bytes_base64"]) is not str
        or type(record["raw_response_byte_length"]) is not int
        or record["raw_response_byte_length"] < 1
    ):
        _fail("SUCCESSOR_RECORD_RAW_RESPONSE_INVALID")
    try:
        raw_response_bytes = base64.b64decode(
            record["raw_response_bytes_base64"], validate=True
        )
    except Exception:
        _fail("SUCCESSOR_RECORD_RAW_RESPONSE_INVALID")
    if (
        len(raw_response_bytes) != record["raw_response_byte_length"]
        or _sha256(raw_response_bytes)
        != _hex_sha256(record["raw_response_sha256"], label="raw_response_sha256")
    ):
        _fail("SUCCESSOR_RECORD_RAW_RESPONSE_INVALID")
    try:
        replayed = replay_tushare_response_bytes(
            raw_response_bytes,
            api_name=expected_request["endpoint"],
            expected_fields=expected_request["expected_fields"],
            strict_decimal_decode=True,
            max_response_items=expected_request["row_ceiling"],
        )
    except TushareHttpsError:
        _fail("SUCCESSOR_RECORD_RAW_RESPONSE_REPLAY_FAILED")
    replayed_rows = [
        _normalize_response_row(
            table=record["table"],
            request=expected_request,
            fields=expected_request["expected_fields"],
            values=row,
            symbols=frozenset(binding["plan"]["subject_symbols"]),
            target_date=binding["plan"]["target_date"],
            enforce_subject_scope=False,
        )
        for row in replayed.rows
    ]
    encoded_observed = sorted(
        (_typed_row(row, fields, logical=False) for row in observed_rows),
        key=_canonical_json_bytes,
    )
    if _canonical_json_bytes(encoded_observed) != _canonical_json_bytes(record["observed_rows"]):
        _fail("SUCCESSOR_RECORD_OBSERVATION_NON_CANONICAL")
    replayed_observed = sorted(
        (_typed_row(row, fields, logical=False) for row in replayed_rows),
        key=_canonical_json_bytes,
    )
    if _canonical_json_bytes(replayed_observed) != _canonical_json_bytes(
        record["observed_rows"]
    ):
        _fail("SUCCESSOR_RECORD_RAW_ROWS_MISMATCH")
    subject_symbols = frozenset(binding["plan"]["subject_symbols"])
    in_scope_rows = [
        row for row in observed_rows if str(row["ts_code"]) in subject_symbols
    ]
    out_of_scope_rows = [
        row for row in observed_rows if str(row["ts_code"]) not in subject_symbols
    ]
    opaque_comp_type_evidence = _opaque_comp_type_evidence(
        record["table"], in_scope_rows
    )
    canonical_rows, counters = _canonicalize_rows(record["table"], in_scope_rows)
    if _canonical_json_bytes(
        [_typed_row(row, fields, logical=False) for row in canonical_rows]
    ) != _canonical_json_bytes(record["rows"]):
        _fail("SUCCESSOR_RECORD_ROWS_NON_CANONICAL")
    receipt = _validate_seal(record["receipt"], identity_field="receipt_sha256")
    receipt_fields = {
        "accepted_count",
        "attempts",
        "blocker_codes",
        "canonicalization_counters",
        "classification_partition",
        "canonical_subject_scope_ref_sha256",
        "endpoint",
        "full_response_observation_count",
        "full_response_observation_multiset_sha256",
        "has_more",
        "in_scope_canonical_payload_multiset_sha256",
        "in_scope_observation_count",
        "in_scope_observation_multiset_sha256",
        "item_count",
        "logical_sha256",
        "observation_sha256",
        "opaque_comp_type_evidence",
        "ordinal",
        "out_of_scope_canonical_payload_multiset_sha256",
        "out_of_scope_observation_count",
        "out_of_scope_observation_multiset_sha256",
        "out_of_scope_symbol_count",
        "out_of_scope_symbol_keyset_sha256",
        "payload_sha256",
        "plan_sha256",
        "provider_request_id_sha256",
        "provider_count_policy",
        "provider_reported_count",
        "raw_item_order_sha256",
        "raw_response_byte_length",
        "raw_response_sha256",
        "receipt_sha256",
        "request_envelope_scope_ref_sha256",
        "request_key",
        "retry_error_codes",
        "schema_version",
        "scope_exclusion_policy",
        "scope_partition_sha256",
        "status",
        "subject_symbol_keyset_sha256",
        "table",
    }
    if set(receipt) != receipt_fields:
        _fail("SUCCESSOR_RECEIPT_INVALID")
    retry_errors = receipt["retry_error_codes"]
    integer_receipt_fields = (
        "accepted_count",
        "attempts",
        "full_response_observation_count",
        "in_scope_observation_count",
        "item_count",
        "ordinal",
        "out_of_scope_observation_count",
        "out_of_scope_symbol_count",
        "raw_response_byte_length",
        "provider_reported_count",
    )
    if (
        type(retry_errors) is not list
        or any(type(receipt[field]) is not int for field in integer_receipt_fields)
        or any(
            type(value) is not str or re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", value) is None
            for value in retry_errors
        )
        or type(receipt["attempts"]) is not int
        or receipt["attempts"] != len(retry_errors) + 1
        or receipt["item_count"] != len(observed_rows)
        or receipt["provider_reported_count"] not in {0, receipt["item_count"]}
        or receipt["provider_count_policy"]
        != "ZERO_SENTINEL_OR_EXACT_ITEM_COUNT.v1"
        or replayed.provider_reported_count != receipt["provider_reported_count"]
        or replayed.item_count != receipt["item_count"]
        or replayed.has_more is not False
        or _sha256(replayed.request_id.encode("utf-8"))
        != receipt["provider_request_id_sha256"]
        or _row_order_sha256(record["table"], replayed_rows)
        != receipt["raw_item_order_sha256"]
        or receipt["raw_response_byte_length"] != len(raw_response_bytes)
        or receipt["raw_response_sha256"] != _sha256(raw_response_bytes)
        or receipt["canonicalization_counters"] != counters
        or _canonical_json_bytes(receipt["opaque_comp_type_evidence"])
        != _canonical_json_bytes(opaque_comp_type_evidence)
        or receipt["endpoint"] != expected_request["endpoint"]
        or receipt["table"] != expected_request["table"]
        or receipt["accepted_count"] != len(rows)
        or receipt["status"] != ("EMPTY" if not rows else "AVAILABLE")
        or receipt["has_more"] is not False
        or receipt["blocker_codes"] != []
    ):
        _fail("SUCCESSOR_RECEIPT_INVALID")
    _validate_opaque_comp_type_evidence(
        receipt["opaque_comp_type_evidence"]
    )
    expected_classification = _classification_partition(
        table=record["table"],
        normalized_rows=observed_rows,
        subject_symbols=subject_symbols,
        opaque_evidence=opaque_comp_type_evidence,
    )
    if _canonical_json_bytes(receipt["classification_partition"]) != (
        _canonical_json_bytes(expected_classification)
    ):
        _fail("SUCCESSOR_RECEIPT_CLASSIFICATION_MISMATCH")
    _validate_classification_partition(receipt["classification_partition"])
    expected_scope_identity = _scope_partition_identity(
        table=record["table"],
        normalized_rows=observed_rows,
        in_scope_rows=in_scope_rows,
        out_of_scope_rows=out_of_scope_rows,
        request_envelope_scope_ref=binding["plan"]["request_envelope_scope_ref"],
        canonical_subject_scope_ref=binding["plan"]["canonical_subject_scope_ref"],
    )
    if any(receipt.get(field) != value for field, value in expected_scope_identity.items()):
        _fail("SUCCESSOR_RECEIPT_SCOPE_IDENTITY_MISMATCH")
    provider_request_id_sha256 = _hex_sha256(
        receipt["provider_request_id_sha256"],
        label="provider_request_id_sha256",
    )
    observation, payload, logical = _stored_response_identities(
        table=record["table"],
        provider_request_id_sha256=provider_request_id_sha256,
        provider_reported_count=receipt["provider_reported_count"],
        item_count=receipt["item_count"],
        observed_rows=observed_rows,
        logical_rows=rows,
    )
    if (
        receipt["observation_sha256"] != observation
        or receipt["payload_sha256"] != payload
        or receipt["logical_sha256"] != logical
    ):
        _fail("SUCCESSOR_RECEIPT_IDENTITY_MISMATCH")
    if (
        receipt.get("schema_version") != FUNDAMENTAL_SUCCESSOR_REQUEST_RECEIPT_SCHEMA
        or receipt.get("plan_sha256") != binding["plan_sha256"]
        or receipt.get("request_key") != expected_request["request_key"]
        or receipt.get("ordinal") != expected_request["ordinal"]
    ):
        _fail("SUCCESSOR_RECEIPT_INVALID")
    return receipt, observed_rows, rows


def _frame_scalar_token(value: Any) -> tuple[str, str]:
    if value is None:
        return ("null", "")
    if type(value) is bool:
        return ("boolean", "true" if value else "false")
    if type(value) is int:
        return ("integer", str(value))
    if type(value) is Decimal:
        if not value.is_finite():
            _fail("SUCCESSOR_TABLE_SCALAR_INVALID")
        return ("decimal", value.normalize().to_eng_string())
    if type(value) is str:
        return ("string", value)
    _fail("SUCCESSOR_TABLE_SCALAR_INVALID")


def _decode_row_token(token: bytes, *, fields: Sequence[str]) -> dict[str, Any]:
    try:
        encoded = json.loads(token.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        _fail("SUCCESSOR_TABLE_ROW_TOKEN_INVALID")
    if type(encoded) is not list or len(encoded) != len(fields):
        _fail("SUCCESSOR_TABLE_ROW_TOKEN_INVALID")
    return dict(
        zip(
            fields,
            (_decode_typed_scalar(value) for value in encoded),
            strict=True,
        )
    )


def _accepted_sql() -> str:
    return """
        WITH survivors AS (
            SELECT row_token, business_key, projection_token, sort_token,
                   ts_code, sort_date, end_date
            FROM source_rows AS candidate
            WHERE update_rank = (
                SELECT MAX(peer.update_rank)
                FROM source_rows AS peer
                WHERE peer.physical_key = candidate.physical_key
            )
        ), ranked AS (
            SELECT row_token, business_key, projection_token, sort_token,
                   ts_code, sort_date, end_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY business_key ORDER BY sort_token, row_token
                   ) AS ordinal
            FROM survivors
        )
        SELECT row_token, ts_code, sort_date, end_date
        FROM ranked
        WHERE ordinal = 1
        ORDER BY sort_token, row_token
    """


def _insert_external_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    fields = _output_fields(table)
    attempted = 0
    inserted = 0
    for value in rows:
        attempted += 1
        row = dict(value)
        row_token = _canonical_json_bytes(_typed_row(row, fields, logical=False))
        business_key = _canonical_json_bytes(list(_business_key(table, row)))
        physical_key = _canonical_json_bytes(
            [*_business_key(table, row), *_physical_update_identity(table, row)]
        )
        projection_token = _canonical_json_bytes(
            _typed_row(row, _projection_fields(table), logical=True)
        )
        sort_token = _row_sort_key(row, fields)
        sort_date = str(
            row["trade_date"] if table == "daily_basic" else row["availability_date"]
        )
        end_date = "" if table == "daily_basic" else str(row["end_date"])
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO source_rows (
                row_token, business_key, physical_key, projection_token,
                update_rank, sort_token, ts_code, sort_date, end_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_token,
                business_key,
                physical_key,
                projection_token,
                _update_rank(table, row),
                sort_token,
                str(row["ts_code"]),
                sort_date,
                end_date,
            ),
        )
        inserted += max(0, int(cursor.rowcount))
    connection.commit()
    return attempted, inserted


def _table_frame_schema(
    connection: sqlite3.Connection,
    *,
    fields: Sequence[str],
) -> tuple[int, list[dict[str, Any]]]:
    types = [set() for _field in fields]
    nullable = [False for _field in fields]
    rows = 0
    for (row_token, _symbol, _sort_date, _end_date) in connection.execute(
        _accepted_sql()
    ):
        row = _decode_row_token(bytes(row_token), fields=fields)
        rows += 1
        for position, field in enumerate(fields):
            kind, _payload = _frame_scalar_token(row[field])
            if kind == "null":
                nullable[position] = True
            else:
                types[position].add(kind)
    schema = [
        {
            "position": position,
            "name": ["string", field],
            "logical_scalar_types": sorted(types[position]),
            "nullable": nullable[position],
        }
        for position, field in enumerate(fields)
    ]
    return rows, schema


def _flush_parquet_rows(
    writer: pq.ParquetWriter,
    rows: list[tuple[str, str, str, bytes]],
) -> tuple[int, int]:
    if not rows:
        return 0, 0
    table = pa.Table.from_pydict(
        {
            "ts_code": [row[0] for row in rows],
            "sort_date": [row[1] for row in rows],
            "end_date": [row[2] for row in rows],
            "row_json": [row[3] for row in rows],
        },
        schema=_PARQUET_SCHEMA,
    )
    if len(table) > _MAX_STREAM_BATCH_ROWS or table.nbytes > _MAX_STREAM_BATCH_BYTES:
        _fail("SUCCESSOR_STREAM_BATCH_LIMIT_EXCEEDED")
    writer.write_table(table, row_group_size=len(table))
    return len(table), int(table.nbytes)


def _install_streamed_file(temporary: Path, destination: Path) -> tuple[str, int]:
    try:
        os.chmod(temporary, 0o600)
        descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temporary_sha, temporary_size = _regular_file_identity(temporary)
        if destination.exists():
            installed_sha, installed_size = _regular_file_identity(destination)
            if (installed_sha, installed_size) != (temporary_sha, temporary_size):
                _fail("SUCCESSOR_IMMUTABLE_FILE_CONFLICT")
            os.unlink(temporary)
        else:
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        if _regular_file_identity(destination) != (temporary_sha, temporary_size):
            _fail("SUCCESSOR_ATOMIC_WRITE_READBACK_FAILED")
        return temporary_sha, temporary_size
    finally:
        if temporary.exists():
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _build_streamed_table(
    *,
    table: str,
    row_batches: Iterator[Sequence[Mapping[str, Any]]],
    destination: Path,
    minimum_free_disk_bytes: int,
    external_sort_budget: Mapping[str, int],
) -> dict[str, Any]:
    budget = dict(external_sort_budget)
    required_budget_fields = {
        "fsync_reserve_bytes",
        "margin_25_percent_bytes",
        "parquet_temp_bytes",
        "sqlite_index_bytes",
        "sqlite_journal_bytes",
        "sqlite_store_bytes",
        "total_bytes",
    }
    if (
        set(budget) != required_budget_fields
        or any(type(value) is not int or value < 0 for value in budget.values())
        or budget["total_bytes"]
        != sum(
            budget[field]
            for field in required_budget_fields
            if field not in {"total_bytes"}
        )
    ):
        _fail("SUCCESSOR_EXTERNAL_SORT_BUDGET_INVALID")
    _require_external_sort_reserve(
        destination.parent,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        required_working_bytes=budget["total_bytes"],
    )
    database_fd, database_name = tempfile.mkstemp(
        prefix=f".{table}.sort.",
        suffix=".sqlite3",
        dir=destination.parent,
    )
    os.close(database_fd)
    database = Path(database_name)
    parquet_fd, parquet_name = tempfile.mkstemp(
        prefix=f".{table}.",
        suffix=".parquet",
        dir=destination.parent,
    )
    os.close(parquet_fd)
    temporary = Path(parquet_name)
    connection: sqlite3.Connection | None = None
    try:
        os.chmod(database, 0o600)
        os.chmod(temporary, 0o600)
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TABLE source_rows (
                row_token BLOB PRIMARY KEY,
                business_key BLOB NOT NULL,
                physical_key BLOB NOT NULL,
                projection_token BLOB NOT NULL,
                update_rank INTEGER NOT NULL,
                sort_token BLOB NOT NULL,
                ts_code TEXT NOT NULL,
                sort_date TEXT NOT NULL,
                end_date TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        attempted = 0
        inserted = 0
        for batch in row_batches:
            if len(batch) > _MAX_STREAM_BATCH_ROWS:
                _fail("SUCCESSOR_SOURCE_BATCH_ROW_LIMIT_EXCEEDED")
            batch_bytes = sum(
                len(_canonical_json_bytes(_typed_row(dict(row), _output_fields(table), logical=False)))
                for row in batch
            )
            if batch_bytes > _MAX_STREAM_BATCH_BYTES:
                _fail("SUCCESSOR_SOURCE_BATCH_BYTE_LIMIT_EXCEEDED")
            current_attempted, current_inserted = _insert_external_rows(
                connection,
                table=table,
                rows=batch,
            )
            attempted += current_attempted
            inserted += current_inserted
        _require_external_sort_reserve(
            destination.parent,
            minimum_free_disk_bytes=minimum_free_disk_bytes,
            required_working_bytes=(
                budget["sqlite_index_bytes"]
                + budget["sqlite_journal_bytes"]
                + budget["parquet_temp_bytes"]
                + budget["fsync_reserve_bytes"]
                + budget["margin_25_percent_bytes"]
            ),
        )
        connection.execute(
            "CREATE INDEX source_rows_physical ON source_rows (physical_key, update_rank)"
        )
        connection.execute(
            "CREATE INDEX source_rows_business ON source_rows (business_key, projection_token)"
        )
        connection.commit()
        survivors = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_rows AS candidate
                WHERE update_rank = (
                    SELECT MAX(peer.update_rank) FROM source_rows AS peer
                    WHERE peer.physical_key = candidate.physical_key
                )
                """
            ).fetchone()[0]
        )
        conflict = connection.execute(
            """
            WITH survivors AS (
                SELECT business_key, projection_token
                FROM source_rows AS candidate
                WHERE update_rank = (
                    SELECT MAX(peer.update_rank) FROM source_rows AS peer
                    WHERE peer.physical_key = candidate.physical_key
                )
            )
            SELECT business_key FROM survivors
            GROUP BY business_key
            HAVING COUNT(DISTINCT projection_token) > 1
            LIMIT 1
            """
        ).fetchone()
        if conflict is not None:
            _fail("SUCCESSOR_MATERIAL_DUPLICATE_CONFLICT")
        fields = _output_fields(table)
        row_count, logical_schema = _table_frame_schema(connection, fields=fields)
        _require_external_sort_reserve(
            destination.parent,
            minimum_free_disk_bytes=minimum_free_disk_bytes,
            required_working_bytes=(
                budget["parquet_temp_bytes"]
                + budget["fsync_reserve_bytes"]
                + budget["margin_25_percent_bytes"]
            ),
        )
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {"rows": row_count, "schema": logical_schema},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        writer = pq.ParquetWriter(
            temporary,
            _PARQUET_SCHEMA,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=False,
            version="2.6",
        )
        buffered: list[tuple[str, str, str, bytes]] = []
        buffered_bytes = 0
        observed_rows = 0
        high_rows = 0
        high_bytes = 0
        try:
            for row_token, symbol, sort_date, end_date in connection.execute(
                _accepted_sql()
            ):
                token = bytes(row_token)
                row = _decode_row_token(token, fields=fields)
                frame_tokens = [list(_frame_scalar_token(row[field])) for field in fields]
                digest.update(b"\x00")
                digest.update(
                    json.dumps(
                        frame_tokens,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                estimate = len(token) + len(symbol) + len(sort_date) + len(end_date) + 64
                if buffered and (
                    len(buffered) >= _PARQUET_ROW_GROUP_ROWS
                    or buffered_bytes + estimate > _MAX_STREAM_BATCH_BYTES // 2
                ):
                    written, byte_count = _flush_parquet_rows(writer, buffered)
                    observed_rows += written
                    high_rows = max(high_rows, written)
                    high_bytes = max(high_bytes, byte_count)
                    buffered = []
                    buffered_bytes = 0
                if estimate > _MAX_STREAM_BATCH_BYTES // 2:
                    _fail("SUCCESSOR_STREAM_ROW_BYTE_LIMIT_EXCEEDED")
                buffered.append((str(symbol), str(sort_date), str(end_date), token))
                buffered_bytes += estimate
            written, byte_count = _flush_parquet_rows(writer, buffered)
            observed_rows += written
            high_rows = max(high_rows, written)
            high_bytes = max(high_bytes, byte_count)
        finally:
            writer.close()
        if observed_rows != row_count:
            _fail("SUCCESSOR_STREAMED_TABLE_ROWCOUNT_MISMATCH")
        table_sha, byte_length = _install_streamed_file(temporary, destination)
        metadata: dict[str, Any] = {
            "canonicalization_counters": {
                "exact_duplicates_collapsed": attempted - inserted,
                "projection_equivalent_duplicates_collapsed": survivors - row_count,
                "superseded_updates_discarded": inserted - survivors,
            },
            "fields": list(fields),
            "external_sort_budget": budget,
            "external_sort_reserve_checks": [
                "PRE_POPULATE",
                "PRE_INDEX",
                "PRE_PARQUET",
            ],
            "file_format": "PARQUET",
            "fingerprint_sha256": digest.hexdigest(),
            "layout": "canonical_typed_row.v1",
            "maximum_batch_bytes": _MAX_STREAM_BATCH_BYTES,
            "maximum_batch_rows": _MAX_STREAM_BATCH_ROWS,
            "observed_maximum_batch_bytes": high_bytes,
            "observed_maximum_batch_rows": high_rows,
            "parquet_row_group_rows": _PARQUET_ROW_GROUP_ROWS,
            "row_count": row_count,
            "schema_version": FUNDAMENTAL_SUCCESSOR_TABLE_SCHEMA,
            "table": table,
            "table_sha256": table_sha,
            "byte_length": byte_length,
        }
        return _sealed(metadata, identity_field="metadata_sha256")
    finally:
        if connection is not None:
            connection.close()
        for path in (database, Path(f"{database}-journal"), temporary):
            if path.exists():
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _scope_projection_manifest(
    *,
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    table_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    full_count = sum(int(row["full_response_observation_count"]) for row in receipts)
    in_scope_count = sum(int(row["in_scope_observation_count"]) for row in receipts)
    out_of_scope_count = sum(
        int(row["out_of_scope_observation_count"]) for row in receipts
    )
    body: dict[str, Any] = {
        "canonical_subject_scope_ref": dict(plan["canonical_subject_scope_ref"]),
        "full_response_observation_count": full_count,
        "in_scope_candidate_table_fingerprints": dict(table_fingerprints),
        "in_scope_observation_count": in_scope_count,
        "out_of_scope_observation_count": out_of_scope_count,
        "raw_response_fileset_sha256": _sha256(
            _canonical_json_bytes(
                [
                    {
                        "ordinal": row["ordinal"],
                        "raw_response_byte_length": row["raw_response_byte_length"],
                        "raw_response_sha256": row["raw_response_sha256"],
                        "request_key": row["request_key"],
                    }
                    for row in receipts
                ]
            )
        ),
        "request_envelope_scope_ref": dict(plan["request_envelope_scope_ref"]),
        "schema_version": "myquant-fundamental-successor-scope-projection.v1",
    }
    if full_count != in_scope_count + out_of_scope_count:
        _fail("SUCCESSOR_SCOPE_PARTITION_NOT_RECONCILED")
    return _sealed(body, identity_field="projection_sha256")


def _iter_parquet_rows(
    path: Path,
    *,
    table: str,
    maximum_batch_rows: int = _MAX_STREAM_BATCH_ROWS,
    maximum_batch_bytes: int = _MAX_STREAM_BATCH_BYTES,
) -> Iterator[list[dict[str, Any]]]:
    fields = _output_fields(table)
    try:
        parquet = pq.ParquetFile(path)
    except (OSError, pa.ArrowException):
        _fail("SUCCESSOR_TABLE_PARQUET_INVALID")
    if parquet.schema_arrow != _PARQUET_SCHEMA:
        _fail("SUCCESSOR_TABLE_PARQUET_SCHEMA_MISMATCH")
    for group in range(parquet.num_row_groups):
        if parquet.metadata.row_group(group).num_rows > _PARQUET_ROW_GROUP_ROWS:
            _fail("SUCCESSOR_TABLE_ROW_GROUP_LIMIT_EXCEEDED")
    previous_sort_token: bytes | None = None
    try:
        batches = parquet.iter_batches(batch_size=maximum_batch_rows)
        for batch in batches:
            if len(batch) > maximum_batch_rows or batch.nbytes > maximum_batch_bytes:
                _fail("SUCCESSOR_STREAM_BATCH_LIMIT_EXCEEDED")
            frame = batch.to_pydict()
            rows: list[dict[str, Any]] = []
            for symbol, sort_date, end_date, raw_token in zip(
                frame["ts_code"],
                frame["sort_date"],
                frame["end_date"],
                frame["row_json"],
                strict=True,
            ):
                token = bytes(raw_token)
                row = _decode_row_token(token, fields=fields)
                sort_token = _row_sort_key(row, fields)
                if previous_sort_token is not None and sort_token < previous_sort_token:
                    _fail("SUCCESSOR_TABLE_SORT_ORDER_INVALID")
                previous_sort_token = sort_token
                expected_sort_date = str(
                    row["trade_date"]
                    if table == "daily_basic"
                    else row["availability_date"]
                )
                expected_end_date = "" if table == "daily_basic" else str(row["end_date"])
                if (
                    str(row["ts_code"]) != symbol
                    or expected_sort_date != sort_date
                    or expected_end_date != end_date
                ):
                    _fail("SUCCESSOR_TABLE_INDEX_COLUMN_MISMATCH")
                rows.append(row)
            yield rows
    except (OSError, pa.ArrowException):
        _fail("SUCCESSOR_TABLE_PARQUET_INVALID")


def _validate_streamed_table(
    path: Path,
    *,
    table: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_seal(metadata, identity_field="metadata_sha256")
    required = {
        "byte_length",
        "canonicalization_counters",
        "external_sort_budget",
        "external_sort_reserve_checks",
        "fields",
        "file_format",
        "fingerprint_sha256",
        "layout",
        "maximum_batch_bytes",
        "maximum_batch_rows",
        "metadata_sha256",
        "observed_maximum_batch_bytes",
        "observed_maximum_batch_rows",
        "parquet_row_group_rows",
        "row_count",
        "schema_version",
        "table",
        "table_sha256",
    }
    fields = _output_fields(table)
    if (
        set(value) != required
        or value["schema_version"] != FUNDAMENTAL_SUCCESSOR_TABLE_SCHEMA
        or value["table"] != table
        or value["fields"] != list(fields)
        or value["file_format"] != "PARQUET"
        or value["layout"] != "canonical_typed_row.v1"
        or value["external_sort_reserve_checks"]
        != ["PRE_POPULATE", "PRE_INDEX", "PRE_PARQUET"]
        or value["maximum_batch_rows"] != _MAX_STREAM_BATCH_ROWS
        or value["maximum_batch_bytes"] != _MAX_STREAM_BATCH_BYTES
        or value["parquet_row_group_rows"] != _PARQUET_ROW_GROUP_ROWS
    ):
        _fail("SUCCESSOR_TABLE_ARTIFACT_INVALID")
    budget = value["external_sort_budget"]
    if not isinstance(budget, Mapping):
        _fail("SUCCESSOR_TABLE_ARTIFACT_INVALID")
    budget_values = dict(budget)
    if (
        set(budget_values)
        != {
            "fsync_reserve_bytes",
            "margin_25_percent_bytes",
            "parquet_temp_bytes",
            "sqlite_index_bytes",
            "sqlite_journal_bytes",
            "sqlite_store_bytes",
            "total_bytes",
        }
        or budget_values["total_bytes"]
        != sum(
            item
            for key, item in budget_values.items()
            if key != "total_bytes"
        )
    ):
        _fail("SUCCESSOR_TABLE_ARTIFACT_INVALID")
    sha, size = _regular_file_identity(path)
    if sha != value["table_sha256"] or size != value["byte_length"]:
        _fail("SUCCESSOR_TABLE_ARTIFACT_MISMATCH")
    types = [set() for _field in fields]
    nullable = [False for _field in fields]
    row_count = 0
    observed_rows = 0
    observed_bytes = 0
    for rows in _iter_parquet_rows(path, table=table):
        observed_rows = max(observed_rows, len(rows))
        encoded_bytes = sum(
            len(_canonical_json_bytes(_typed_row(row, fields, logical=False)))
            for row in rows
        )
        observed_bytes = max(observed_bytes, encoded_bytes)
        for row in rows:
            row_count += 1
            for position, field in enumerate(fields):
                kind, _payload = _frame_scalar_token(row[field])
                if kind == "null":
                    nullable[position] = True
                else:
                    types[position].add(kind)
    logical_schema = [
        {
            "position": position,
            "name": ["string", field],
            "logical_scalar_types": sorted(types[position]),
            "nullable": nullable[position],
        }
        for position, field in enumerate(fields)
    ]
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"rows": row_count, "schema": logical_schema},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for rows in _iter_parquet_rows(path, table=table):
        for row in rows:
            digest.update(b"\x00")
            digest.update(
                json.dumps(
                    [list(_frame_scalar_token(row[field])) for field in fields],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    if (
        row_count != value["row_count"]
        or digest.hexdigest() != value["fingerprint_sha256"]
        or observed_rows > value["maximum_batch_rows"]
        or observed_bytes > value["maximum_batch_bytes"]
        or value["observed_maximum_batch_rows"] > value["maximum_batch_rows"]
        or value["observed_maximum_batch_bytes"] > value["maximum_batch_bytes"]
    ):
        _fail("SUCCESSOR_TABLE_ARTIFACT_MISMATCH")
    return value


def _table_memory_estimates(
    *,
    requests: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    record_refs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    if not (len(requests) == len(receipts) == len(record_refs)):
        _fail("SUCCESSOR_RESOURCE_ACCOUNTING_INVALID")
    rows_by_table = {table: 0 for table in _TABLES}
    largest_record_by_table = {table: 0 for table in _TABLES}
    record_bytes_by_table = {table: 0 for table in _TABLES}
    for request, receipt, ref in zip(requests, receipts, record_refs, strict=True):
        table = str(request.get("table") or "")
        if table not in rows_by_table or receipt.get("table") != table:
            _fail("SUCCESSOR_RESOURCE_ACCOUNTING_INVALID")
        accepted = receipt.get("accepted_count")
        byte_length = ref.get("byte_length")
        if (
            type(accepted) is not int
            or accepted < 0
            or type(byte_length) is not int
            or byte_length < 1
        ):
            _fail("SUCCESSOR_RESOURCE_ACCOUNTING_INVALID")
        rows_by_table[table] += accepted
        largest_record_by_table[table] = max(
            largest_record_by_table[table], byte_length
        )
        record_bytes_by_table[table] += byte_length
    estimates: dict[str, dict[str, int]] = {}
    for table in _TABLES:
        row_count = rows_by_table[table]
        # Cardinality is externalised to SQLite/Parquet.  The resident bound is
        # one independently byte-capped request plus one Arrow batch and codec
        # workspace; it does not grow with aggregate table row count.
        estimated = min(
            _DEFAULT_MAXIMUM_RECORD_BYTES,
            largest_record_by_table[table],
        ) + 2 * _MAX_STREAM_BATCH_BYTES
        estimates[table] = {
            "accepted_row_count": row_count,
            "estimated_peak_memory_bytes": estimated,
            "largest_record_bytes": largest_record_by_table[table],
            "record_bytes": record_bytes_by_table[table],
            "stream_batch_bytes": _MAX_STREAM_BATCH_BYTES,
            "stream_batch_rows": _MAX_STREAM_BATCH_ROWS,
        }
    return estimates


def _resource_accounting(
    *,
    resource_policy: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    record_refs: Sequence[Mapping[str, Any]],
    table_refs: Mapping[str, Mapping[str, Any]],
    minimum_observed_free_disk_bytes: int,
) -> dict[str, Any]:
    policy = _validate_resource_policy(dict(resource_policy))
    estimates = _table_memory_estimates(
        requests=requests,
        receipts=receipts,
        record_refs=record_refs,
    )
    peak = max(
        (row["estimated_peak_memory_bytes"] for row in estimates.values()),
        default=0,
    )
    # Tables are replayed sequentially.  Aggregate cardinality is on disk, so
    # simultaneous resident memory equals the maximum per-table bound.
    aggregate = peak
    if (
        peak > policy["table_memory_limit_bytes"]
        or minimum_observed_free_disk_bytes < policy["minimum_free_disk_bytes"]
    ):
        _fail("SUCCESSOR_RESOURCE_PREFLIGHT_BLOCKED")
    source_payload_bytes = sum(int(ref["byte_length"]) for ref in record_refs)
    source_payload_bytes += sum(
        int(dict(ref)["byte_length"]) for ref in table_refs.values()
    )
    body: dict[str, Any] = {
        "aggregate_estimated_memory_bytes": aggregate,
        "maximum_estimated_table_memory_bytes": peak,
        "minimum_observed_free_disk_bytes": minimum_observed_free_disk_bytes,
        "policy": policy,
        "schema_version": "myquant-fundamental-successor-resource-accounting.v2",
        "source_payload_bytes": source_payload_bytes,
        "status": "PASS",
        "table_estimates": estimates,
    }
    return _sealed(body, identity_field="resource_sha256")


def _file_ref(path: str, payload: bytes) -> dict[str, Any]:
    return {"byte_length": len(payload), "path": path, "sha256": _sha256(payload)}


def _path_ref(path: str, absolute: Path) -> dict[str, Any]:
    digest, byte_length = _regular_file_identity(absolute)
    return {"byte_length": byte_length, "path": path, "sha256": digest}


def _validate_file_ref(root: Path, value: Any) -> bytes:
    required = {"byte_length", "path", "sha256"}
    optional = {"ordinal", "request_key", "row_count", "table_sha256"}
    if (
        type(value) is not dict
        or not required.issubset(value)
        or not set(value).issubset(required | optional)
    ):
        _fail("SUCCESSOR_FILE_REF_INVALID")
    path_text = value["path"]
    if (
        type(path_text) is not str
        or not path_text
        or Path(path_text).is_absolute()
        or ".." in Path(path_text).parts
    ):
        _fail("SUCCESSOR_FILE_REF_INVALID")
    payload = _regular_bytes(root / path_text)
    if (
        type(value["byte_length"]) is not int
        or value["byte_length"] != len(payload)
        or _hex_sha256(value["sha256"], label="file_sha256") != _sha256(payload)
    ):
        _fail("SUCCESSOR_FILE_REF_MISMATCH")
    return payload


def _validate_streamed_file_ref(root: Path, value: Any) -> Path:
    required = {"byte_length", "path", "sha256"}
    optional = {"metadata", "ordinal", "request_key", "row_count", "table_sha256"}
    if (
        type(value) is not dict
        or not required.issubset(value)
        or not set(value).issubset(required | optional)
    ):
        _fail("SUCCESSOR_FILE_REF_INVALID")
    path_text = value["path"]
    if (
        type(path_text) is not str
        or not path_text
        or Path(path_text).is_absolute()
        or ".." in Path(path_text).parts
    ):
        _fail("SUCCESSOR_FILE_REF_INVALID")
    path = root / path_text
    digest, byte_length = _regular_file_identity(path)
    if (
        type(value["byte_length"]) is not int
        or value["byte_length"] != byte_length
        or _hex_sha256(value["sha256"], label="file_sha256") != digest
    ):
        _fail("SUCCESSOR_FILE_REF_MISMATCH")
    return path


def _record_row_batches(
    *,
    table: str,
    requests: Sequence[Mapping[str, Any]],
    records_root: Path,
    binding: Mapping[str, Any],
) -> Iterator[Sequence[Mapping[str, Any]]]:
    for request in requests:
        if request["table"] != table:
            continue
        record_path = records_root / f"{request['ordinal']:06d}.json"
        _receipt_value, _observed_rows, rows = _decode_record(
            _canonical_file_mapping(record_path),
            binding=binding,
            expected_request=request,
        )
        if len(rows) >= int(request["row_ceiling"]):
            _fail("SUCCESSOR_PROVIDER_ROW_CEILING_HIT")
        fields = _output_fields(table)
        batch: list[Mapping[str, Any]] = []
        batch_bytes = 0
        for row in rows:
            row_bytes = len(
                _canonical_json_bytes(_typed_row(dict(row), fields, logical=False))
            )
            if row_bytes > _MAX_STREAM_BATCH_BYTES:
                _fail("SUCCESSOR_SOURCE_ROW_BYTE_LIMIT_EXCEEDED")
            if batch and (
                len(batch) >= _MAX_STREAM_BATCH_ROWS
                or batch_bytes + row_bytes > _MAX_STREAM_BATCH_BYTES
            ):
                yield batch
                batch = []
                batch_bytes = 0
            batch.append(row)
            batch_bytes += row_bytes
        if batch:
            yield batch


def _opaque_comp_type_accounting(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observation_components: list[dict[str, Any]] = []
    peer_components: list[dict[str, Any]] = []
    observation_count = 0
    business_key_count = 0
    deferred_observation_count = 0
    unpaired_count = 0
    material_conflict_count = 0
    for receipt in receipts:
        raw_evidence = receipt.get("opaque_comp_type_evidence")
        if not isinstance(raw_evidence, Mapping):
            _fail("SUCCESSOR_OPAQUE_COMP_TYPE_EVIDENCE_INVALID")
        evidence = _validate_opaque_comp_type_evidence(raw_evidence)
        ordinal = int(receipt["ordinal"])
        observation_count += int(
            evidence["opaque_comp_type_observation_count"]
        )
        business_key_count += int(
            evidence["opaque_comp_type_business_key_count"]
        )
        unpaired_count += int(evidence["opaque_unpaired_count"])
        deferred_observation_count += int(
            evidence["opaque_deferred_observation_count"]
        )
        material_conflict_count += int(
            evidence["opaque_material_conflict_count"]
        )
        if evidence["opaque_comp_type_observation_count"]:
            observation_components.append(
                {
                    "ordinal": ordinal,
                    "sha256": evidence[
                        "opaque_comp_type_observation_multiset_sha256"
                    ],
                }
            )
            peer_components.append(
                {
                    "ordinal": ordinal,
                    "sha256": evidence[
                        "opaque_to_supported_peer_pair_keyset_sha256"
                    ],
                }
            )
    return _sealed(
        {
            "schema_version": (
                FUNDAMENTAL_SUCCESSOR_OPAQUE_COMP_TYPE_ACCOUNTING_SCHEMA
            ),
            "opaque_comp_type_observation_count": observation_count,
            "opaque_comp_type_business_key_count": business_key_count,
            "opaque_deferred_observation_count": deferred_observation_count,
            "opaque_comp_type_observation_multiset_sha256": _sha256(
                _canonical_json_bytes(observation_components)
            ),
            "opaque_to_supported_peer_pair_keyset_sha256": _sha256(
                _canonical_json_bytes(peer_components)
            ),
            "opaque_unpaired_count": unpaired_count,
            "opaque_material_conflict_count": material_conflict_count,
        },
        identity_field="accounting_sha256",
    )


def _validate_opaque_comp_type_accounting(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    accounting = _validate_seal(value, identity_field="accounting_sha256")
    required = {
        "accounting_sha256",
        "opaque_comp_type_business_key_count",
        "opaque_comp_type_observation_count",
        "opaque_deferred_observation_count",
        "opaque_comp_type_observation_multiset_sha256",
        "opaque_material_conflict_count",
        "opaque_to_supported_peer_pair_keyset_sha256",
        "opaque_unpaired_count",
        "schema_version",
    }
    count_fields = {
        "opaque_comp_type_business_key_count",
        "opaque_comp_type_observation_count",
        "opaque_deferred_observation_count",
        "opaque_material_conflict_count",
        "opaque_unpaired_count",
    }
    if (
        set(accounting) != required
        or accounting["schema_version"]
        != FUNDAMENTAL_SUCCESSOR_OPAQUE_COMP_TYPE_ACCOUNTING_SCHEMA
        or any(
            type(accounting[field]) is not int or accounting[field] < 0
            for field in count_fields
        )
        or accounting["opaque_material_conflict_count"] != 0
    ):
        _fail("SUCCESSOR_OPAQUE_COMP_TYPE_ACCOUNTING_INVALID")
    for field in (
        "accounting_sha256",
        "opaque_comp_type_observation_multiset_sha256",
        "opaque_to_supported_peer_pair_keyset_sha256",
    ):
        _hex_sha256(accounting[field], label=field)
    return accounting


def _unsupported_inventory(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for receipt in receipts:
        evidence = _validate_opaque_comp_type_evidence(
            receipt["opaque_comp_type_evidence"]
        )
        for observation in evidence["deferred_observations"]:
            entries.append(
                {
                    "business_key": list(observation["business_key"]),
                    "classification": "TAINTED_PENDING_ANALYSIS",
                    "ordinal": int(receipt["ordinal"]),
                    "record_path": f"requests/{int(receipt['ordinal']):06d}.json",
                    "request_key": str(receipt["request_key"]),
                    "row_sha256": str(observation["row_sha256"]),
                    "table": str(receipt["table"]),
                    "typed_row": list(observation["typed_row"]),
                }
            )
    entries.sort(key=_canonical_json_bytes)
    keys = sorted(
        {tuple(entry["business_key"]) for entry in entries},
        key=lambda value: tuple(part.encode("utf-8") for part in value),
    )
    body = {
        "schema_version": FUNDAMENTAL_SUCCESSOR_UNSUPPORTED_INVENTORY_SCHEMA,
        "authority_state": _DEFERRED_AUTHORITY_STATE,
        "deferred_observation_count": len(entries),
        "deferred_business_key_count": len(keys),
        "deferred_business_keyset_sha256": _sha256(
            _canonical_json_bytes([list(value) for value in keys])
        ),
        "deferred_observation_multiset_sha256": _sha256(
            _canonical_json_bytes(entries)
        ),
        "entries": entries,
    }
    return _sealed(body, identity_field="inventory_sha256")


def _validate_unsupported_inventory(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = _validate_seal(value, identity_field="inventory_sha256")
    if (
        set(inventory)
        != {
            "authority_state",
            "deferred_business_key_count",
            "deferred_business_keyset_sha256",
            "deferred_observation_count",
            "deferred_observation_multiset_sha256",
            "entries",
            "inventory_sha256",
            "schema_version",
        }
        or inventory["schema_version"]
        != FUNDAMENTAL_SUCCESSOR_UNSUPPORTED_INVENTORY_SCHEMA
        or inventory["authority_state"] != _DEFERRED_AUTHORITY_STATE
        or type(inventory["entries"]) is not list
        or type(inventory["deferred_observation_count"]) is not int
        or inventory["deferred_observation_count"] != len(inventory["entries"])
    ):
        _fail("SUCCESSOR_UNSUPPORTED_INVENTORY_INVALID")
    expected = _unsupported_inventory_from_entries(inventory["entries"])
    if _canonical_json_bytes(expected) != _canonical_json_bytes(inventory):
        _fail("SUCCESSOR_UNSUPPORTED_INVENTORY_INVALID")
    return inventory


def _unsupported_inventory_from_entries(
    raw_entries: Sequence[Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        if (
            type(raw) is not dict
            or set(raw)
            != {
                "business_key",
                "classification",
                "ordinal",
                "record_path",
                "request_key",
                "row_sha256",
                "table",
                "typed_row",
            }
            or raw["classification"] != "TAINTED_PENDING_ANALYSIS"
            or raw["table"] != "balancesheet"
            or type(raw["ordinal"]) is not int
            or raw["ordinal"] < 0
            or raw["record_path"] != f"requests/{raw['ordinal']:06d}.json"
            or type(raw["business_key"]) is not list
            or len(raw["business_key"]) != 3
            or any(type(part) is not str for part in raw["business_key"])
            or type(raw["typed_row"]) is not list
            or _sha256(_canonical_json_bytes(raw["typed_row"]))
            != raw["row_sha256"]
        ):
            _fail("SUCCESSOR_UNSUPPORTED_INVENTORY_INVALID")
        entries.append(dict(raw))
    entries.sort(key=_canonical_json_bytes)
    keys = sorted(
        {tuple(entry["business_key"]) for entry in entries},
        key=lambda value: tuple(part.encode("utf-8") for part in value),
    )
    body = {
        "schema_version": FUNDAMENTAL_SUCCESSOR_UNSUPPORTED_INVENTORY_SCHEMA,
        "authority_state": _DEFERRED_AUTHORITY_STATE,
        "deferred_observation_count": len(entries),
        "deferred_business_key_count": len(keys),
        "deferred_business_keyset_sha256": _sha256(
            _canonical_json_bytes([list(value) for value in keys])
        ),
        "deferred_observation_multiset_sha256": _sha256(
            _canonical_json_bytes(entries)
        ),
        "entries": entries,
    }
    return _sealed(body, identity_field="inventory_sha256")


def acquire_successor_support(
    *,
    plan: Mapping[str, Any],
    client: SuccessorTushareClient,
    fileset_root: str | Path,
    captured_pointer_bytes: Mapping[str, bytes],
    immutable_refs: Mapping[str, Any],
    implementation_sha256: str,
    captured_at: str,
    max_attempts: int = 3,
    retry_backoff_seconds: Sequence[float] = (0.5, 1.0),
    requests_per_second: float = 8.0,
    physical_memory_bytes: int | None = None,
    available_memory_bytes: int | None = None,
    rlimit_headroom_bytes: int | None = None,
    table_memory_limit_bytes: int | None = None,
    minimum_free_disk_bytes: int = _DEFAULT_MINIMUM_FREE_DISK_BYTES,
    maximum_record_bytes: int = _DEFAULT_MAXIMUM_RECORD_BYTES,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Acquire every support partition into a resumable private fileset.

    Any provider, schema, scope, ceiling, ``has_more``, duplicate-conflict, or
    readback failure raises a static-code error and no COMPLETE manifest exists.
    """

    requests = replay_successor_support_requests(plan)
    attempts, backoffs, rate = _validate_execution_policy(
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        requests_per_second=requests_per_second,
    )
    resolved_physical_memory = (
        _physical_memory_bytes()
        if physical_memory_bytes is None
        else physical_memory_bytes
    )
    resolved_available_memory = (
        _available_memory_bytes()
        if available_memory_bytes is None
        else available_memory_bytes
    )
    resolved_rlimit_headroom = (
        _rlimit_headroom_bytes(resident_memory_bytes=_resident_memory_bytes())
        if rlimit_headroom_bytes is None
        else rlimit_headroom_bytes
    )
    effective_headroom = min(
        resolved_physical_memory,
        resolved_available_memory,
        resolved_rlimit_headroom,
    )
    resolved_table_memory_limit = (
        int(Decimal(effective_headroom) * _MAX_TABLE_MEMORY_FRACTION)
        if table_memory_limit_bytes is None
        else table_memory_limit_bytes
    )
    resource_policy = _resource_policy(
        physical_memory_bytes=resolved_physical_memory,
        available_memory_bytes=resolved_available_memory,
        rlimit_headroom_bytes=resolved_rlimit_headroom,
        table_memory_limit_bytes=resolved_table_memory_limit,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        maximum_record_bytes=maximum_record_bytes,
    )
    expected_binding = _binding(
        plan=plan,
        captured_pointer_bytes=captured_pointer_bytes,
        immutable_refs=immutable_refs,
        implementation_sha256=implementation_sha256,
        captured_at=captured_at,
        max_attempts=attempts,
        retry_backoff_seconds=backoffs,
        requests_per_second=rate,
        resource_policy=resource_policy,
    )
    root = _private_root(fileset_root, create=True)
    records_root = _safe_directory(root, "requests", create=True)
    tables_root = _safe_directory(root, "tables", create=True)
    binding_path = root / "binding.json"
    manifest_path = root / "provider_manifest.json"
    if manifest_path.exists():
        manifest = validate_successor_capture_fileset(root)
        installed_binding = _validate_binding(_canonical_file_mapping(binding_path))
        if not _resume_binding_matches(installed_binding, expected_binding):
            _fail("SUCCESSOR_RESUME_BINDING_MISMATCH")
        return manifest
    if binding_path.exists():
        installed_binding = _validate_binding(_canonical_file_mapping(binding_path))
        if not _resume_binding_matches(installed_binding, expected_binding):
            _fail("SUCCESSOR_RESUME_BINDING_MISMATCH")
    else:
        _require_disk_reserve(
            root,
            minimum_free_disk_bytes=resource_policy["minimum_free_disk_bytes"],
            pending_bytes=len(_canonical_json_bytes(expected_binding)),
        )
        _atomic_write(binding_path, _canonical_json_bytes(expected_binding))
    binding = _validate_binding(_canonical_file_mapping(binding_path))
    expected_record_names = {f"{request['ordinal']:06d}.json" for request in requests}
    observed_record_names = {entry.name for entry in records_root.iterdir()}
    if not observed_record_names.issubset(expected_record_names):
        _fail("SUCCESSOR_CHECKPOINT_UNEXPECTED_RECORD")
    pacer = _Pacer(rate, sleeper=sleeper, monotonic=monotonic)
    receipts: list[dict[str, Any]] = []
    minimum_observed_free_disk = _require_disk_reserve(
        root,
        minimum_free_disk_bytes=resource_policy["minimum_free_disk_bytes"],
        pending_bytes=resource_policy["maximum_record_bytes"],
    )
    symbols = frozenset(plan["subject_symbols"])
    for request in requests:
        record_path = records_root / f"{request['ordinal']:06d}.json"
        if record_path.exists():
            receipt, _observed_rows, _rows = _decode_record(
                _canonical_file_mapping(record_path),
                binding=binding,
                expected_request=request,
            )
        else:
            try:
                receipt, observed_rows, rows, raw_response_bytes = _fetch_request(
                    plan=plan,
                    request=request,
                    client=client,
                    symbols=symbols,
                    max_attempts=attempts,
                    retry_backoff_seconds=backoffs,
                    pacer=pacer,
                    sleeper=sleeper,
                )
            except FundamentalSuccessorSourceError as error:
                _persist_failure_evidence(
                    root=root,
                    binding=binding,
                    request=request,
                    error=error,
                    maximum_record_bytes=resource_policy[
                        "maximum_record_bytes"
                    ],
                )
                raise
            record = _record(
                binding_sha256=binding["binding_sha256"],
                request=request,
                receipt=receipt,
                observed_rows=observed_rows,
                rows=rows,
                raw_response_bytes=raw_response_bytes,
            )
            record_payload = _canonical_json_bytes(record)
            if len(record_payload) > resource_policy["maximum_record_bytes"]:
                _fail("SUCCESSOR_RECORD_RESOURCE_LIMIT_EXCEEDED")
            observed_free = _require_disk_reserve(
                root,
                minimum_free_disk_bytes=resource_policy["minimum_free_disk_bytes"],
                pending_bytes=len(record_payload),
            )
            minimum_observed_free_disk = min(
                minimum_observed_free_disk, observed_free
            )
            _atomic_write(record_path, record_payload)
            receipt, _observed_rows, _rows = _decode_record(
                _canonical_file_mapping(record_path),
                binding=binding,
                expected_request=request,
            )
        receipts.append(receipt)
    record_refs = []
    for request in requests:
        relative = f"requests/{request['ordinal']:06d}.json"
        payload = _regular_bytes(root / relative)
        if len(payload) > resource_policy["maximum_record_bytes"]:
            _fail("SUCCESSOR_RECORD_RESOURCE_LIMIT_EXCEEDED")
        ref = _file_ref(relative, payload)
        ref.update(
            {
                "ordinal": request["ordinal"],
                "request_key": request["request_key"],
            }
        )
        record_refs.append(ref)
    unsupported_root = root / "unsupported_observations"
    if not unsupported_root.exists():
        unsupported_root.mkdir(mode=0o700)
    unsupported_inventory = _unsupported_inventory(receipts)
    unsupported_inventory_path = unsupported_root / "inventory.json"
    unsupported_inventory_payload = _canonical_json_bytes(
        unsupported_inventory
    )
    _atomic_write(unsupported_inventory_path, unsupported_inventory_payload)
    unsupported_inventory_ref = _file_ref(
        "unsupported_observations/inventory.json",
        _regular_bytes(unsupported_inventory_path),
    )
    table_estimates = _table_memory_estimates(
        requests=requests,
        receipts=receipts,
        record_refs=record_refs,
    )
    if (
        max(
            row["estimated_peak_memory_bytes"]
            for row in table_estimates.values()
        )
        > resource_policy["table_memory_limit_bytes"]
    ):
        _fail("SUCCESSOR_RESOURCE_PREFLIGHT_BLOCKED")
    table_refs: dict[str, dict[str, Any]] = {}
    table_fingerprints: dict[str, str] = {}
    for table in _TABLES:
        path = tables_root / f"{table}.parquet"
        external_sort_budget = _external_sort_budget(
            record_bytes=table_estimates[table]["record_bytes"],
            accepted_rows=table_estimates[table]["accepted_row_count"],
        )
        artifact = _build_streamed_table(
            table=table,
            row_batches=_record_row_batches(
                table=table,
                requests=requests,
                records_root=records_root,
                binding=binding,
            ),
            destination=path,
            minimum_free_disk_bytes=resource_policy["minimum_free_disk_bytes"],
            external_sort_budget=external_sort_budget,
        )
        observed_free = _require_disk_reserve(
            root,
            minimum_free_disk_bytes=resource_policy["minimum_free_disk_bytes"],
            pending_bytes=int(artifact["byte_length"]),
        )
        minimum_observed_free_disk = min(minimum_observed_free_disk, observed_free)
        _validate_streamed_table(path, table=table, metadata=artifact)
        table_refs[table] = _path_ref(f"tables/{table}.parquet", path)
        table_refs[table]["row_count"] = artifact["row_count"]
        table_refs[table]["table_sha256"] = artifact["table_sha256"]
        table_refs[table]["metadata"] = artifact
        table_fingerprints[table] = artifact["fingerprint_sha256"]
        del artifact
    request_attempts = sum(int(receipt["attempts"]) for receipt in receipts)
    retry_failures = sum(len(receipt["retry_error_codes"]) for receipt in receipts)
    scope_projection = _scope_projection_manifest(
        plan=plan,
        receipts=receipts,
        table_fingerprints=table_fingerprints,
    )
    resource_accounting = _resource_accounting(
        resource_policy=resource_policy,
        requests=requests,
        receipts=receipts,
        record_refs=record_refs,
        table_refs=table_refs,
        minimum_observed_free_disk_bytes=minimum_observed_free_disk,
    )
    authoritative_source_ready = (
        int(unsupported_inventory["deferred_observation_count"]) == 0
    )
    body = {
        "authority_state": (
            _AUTHORITATIVE_AUTHORITY_STATE
            if authoritative_source_ready
            else _DEFERRED_AUTHORITY_STATE
        ),
        "authoritative_source_ready": authoritative_source_ready,
        "binding": binding,
        "binding_ref": _file_ref("binding.json", _regular_bytes(binding_path)),
        "captured_pointers": binding["captured_pointers"],
        "fileset_schema_version": FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA,
        "implementation_sha256": binding["implementation_sha256"],
        "immutable_refs": binding["immutable_refs"],
        "plan_sha256": plan["plan_sha256"],
        "provider_accounting": {
            "full_response_observation_rows": sum(
                int(row["full_response_observation_count"]) for row in receipts
            ),
            "has_more_requests": 0,
            "in_scope_observation_rows": sum(
                int(row["in_scope_observation_count"]) for row in receipts
            ),
            "malformed_requests": 0,
            "opaque_comp_type": _opaque_comp_type_accounting(receipts),
            "raw_response_bytes": sum(
                int(row["raw_response_byte_length"]) for row in receipts
            ),
            "request_attempts": request_attempts,
            "requests_empty": sum(row["status"] == "EMPTY" for row in receipts),
            "requests_failed": 0,
            "requests_succeeded": sum(row["status"] == "AVAILABLE" for row in receipts),
            "requests_terminal": len(receipts),
            "retryable_attempt_failures": retry_failures,
            "row_ceiling_hits": 0,
            "schema_mismatches": 0,
            "scope_excluded_rows": sum(
                int(row["out_of_scope_observation_count"]) for row in receipts
            ),
            "scope_exclusion_requests": sum(
                int(row["out_of_scope_observation_count"]) > 0 for row in receipts
            ),
        },
        "record_files": record_refs,
        "request_receipts": receipts,
        "request_topology_sha256": plan["request_topology_sha256"],
        "resource_accounting": resource_accounting,
        "schema_version": FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA,
        "scope_projection": scope_projection,
        "staging_eligible": authoritative_source_ready,
        "status": "COMPLETE",
        "table_files": table_refs,
        "table_fingerprints": table_fingerprints,
        "unsupported_inventory_ref": unsupported_inventory_ref,
        "promotion_eligible": authoritative_source_ready,
        "canonical_write_authorized": False,
        "usable_for_investment_research": False,
    }
    manifest = _sealed(body, identity_field="manifest_sha256")
    manifest_payload = _canonical_json_bytes(manifest)
    if len(manifest_payload) > resource_policy["table_memory_limit_bytes"]:
        _fail("SUCCESSOR_MANIFEST_RESOURCE_LIMIT_EXCEEDED")
    _require_disk_reserve(
        root,
        minimum_free_disk_bytes=resource_policy["minimum_free_disk_bytes"],
        pending_bytes=len(manifest_payload),
    )
    _atomic_write(manifest_path, manifest_payload)
    validated = validate_successor_capture_fileset(root)
    return validated


def _validate_manifest_shape(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _validate_seal(value, identity_field="manifest_sha256")
    required = {
        "authority_state",
        "authoritative_source_ready",
        "binding",
        "binding_ref",
        "captured_pointers",
        "fileset_schema_version",
        "implementation_sha256",
        "immutable_refs",
        "manifest_sha256",
        "plan_sha256",
        "provider_accounting",
        "record_files",
        "request_receipts",
        "request_topology_sha256",
        "resource_accounting",
        "schema_version",
        "scope_projection",
        "staging_eligible",
        "status",
        "table_files",
        "table_fingerprints",
        "unsupported_inventory_ref",
        "promotion_eligible",
        "canonical_write_authorized",
        "usable_for_investment_research",
    }
    if set(manifest) != required:
        _fail("SUCCESSOR_PROVIDER_MANIFEST_FIELDS_INVALID")
    authority_tuple = (
        manifest["authority_state"],
        manifest["authoritative_source_ready"],
        manifest["staging_eligible"],
        manifest["promotion_eligible"],
    )
    if (
        manifest["schema_version"] != FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA
        or manifest["fileset_schema_version"] != FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA
        or manifest["status"] != "COMPLETE"
        or authority_tuple
        not in {
            (_DEFERRED_AUTHORITY_STATE, False, False, False),
            (_AUTHORITATIVE_AUTHORITY_STATE, True, True, True),
        }
        or manifest["canonical_write_authorized"] is not False
        or manifest["usable_for_investment_research"] is not False
    ):
        _fail("SUCCESSOR_PROVIDER_MANIFEST_CONTRACT_MISMATCH")
    accounting = manifest["provider_accounting"]
    if type(accounting) is not dict or set(accounting) != {
        "full_response_observation_rows",
        "has_more_requests",
        "in_scope_observation_rows",
        "malformed_requests",
        "opaque_comp_type",
        "raw_response_bytes",
        "request_attempts",
        "requests_empty",
        "requests_failed",
        "requests_succeeded",
        "requests_terminal",
        "retryable_attempt_failures",
        "row_ceiling_hits",
        "schema_mismatches",
        "scope_excluded_rows",
        "scope_exclusion_requests",
    }:
        _fail("SUCCESSOR_PROVIDER_ACCOUNTING_INVALID")
    if any(
        type(value) is not int or value < 0
        for key, value in accounting.items()
        if key != "opaque_comp_type"
    ):
        _fail("SUCCESSOR_PROVIDER_ACCOUNTING_INVALID")
    opaque_accounting = accounting["opaque_comp_type"]
    if not isinstance(opaque_accounting, Mapping):
        _fail("SUCCESSOR_PROVIDER_ACCOUNTING_INVALID")
    _validate_opaque_comp_type_accounting(opaque_accounting)
    if any(
        accounting[field] != 0
        for field in (
            "has_more_requests",
            "malformed_requests",
            "requests_failed",
            "row_ceiling_hits",
            "schema_mismatches",
        )
    ):
        _fail("SUCCESSOR_PROVIDER_ACCOUNTING_NOT_CLOSED")
    if (
        accounting["requests_succeeded"] + accounting["requests_empty"]
        != accounting["requests_terminal"]
        or accounting["request_attempts"]
        != accounting["requests_terminal"] + accounting["retryable_attempt_failures"]
    ):
        _fail("SUCCESSOR_PROVIDER_ACCOUNTING_NOT_RECONCILED")
    projection = manifest["scope_projection"]
    if not isinstance(projection, Mapping):
        _fail("SUCCESSOR_SCOPE_PROJECTION_INVALID")
    validated_projection = _validate_seal(
        projection,
        identity_field="projection_sha256",
    )
    if set(validated_projection) != {
        "canonical_subject_scope_ref",
        "full_response_observation_count",
        "in_scope_candidate_table_fingerprints",
        "in_scope_observation_count",
        "out_of_scope_observation_count",
        "projection_sha256",
        "raw_response_fileset_sha256",
        "request_envelope_scope_ref",
        "schema_version",
    } or validated_projection["schema_version"] != (
        "myquant-fundamental-successor-scope-projection.v1"
    ):
        _fail("SUCCESSOR_SCOPE_PROJECTION_INVALID")
    resource = manifest["resource_accounting"]
    if not isinstance(resource, Mapping):
        _fail("SUCCESSOR_RESOURCE_ACCOUNTING_INVALID")
    validated_resource = _validate_seal(
        resource,
        identity_field="resource_sha256",
    )
    if set(validated_resource) != {
        "aggregate_estimated_memory_bytes",
        "maximum_estimated_table_memory_bytes",
        "minimum_observed_free_disk_bytes",
        "policy",
        "resource_sha256",
        "schema_version",
        "source_payload_bytes",
        "status",
        "table_estimates",
    } or (
        validated_resource["schema_version"]
        != "myquant-fundamental-successor-resource-accounting.v2"
        or validated_resource["status"] != "PASS"
    ):
        _fail("SUCCESSOR_RESOURCE_ACCOUNTING_INVALID")
    _validate_resource_policy(validated_resource["policy"])
    return manifest


def validate_successor_capture_fileset(
    fileset_root: str | Path,
    *,
    expected_implementation_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a complete, capture-only, promotion-ineligible fileset."""

    root = _private_root(fileset_root, create=False)
    records_root = _safe_directory(root, "requests", create=False)
    tables_root = _safe_directory(root, "tables", create=False)
    unsupported_root = _safe_directory(
        root,
        "unsupported_observations",
        create=False,
    )
    if {entry.name for entry in root.iterdir()} != {
        "binding.json",
        "provider_manifest.json",
        "requests",
        "tables",
        "unsupported_observations",
    }:
        _fail("SUCCESSOR_FILESET_ENTRY_SET_INVALID")
    manifest_path = root / "provider_manifest.json"
    binding_path = root / "binding.json"
    manifest = _validate_manifest_shape(_canonical_file_mapping(manifest_path))
    binding = _validate_binding(_canonical_file_mapping(binding_path))
    if (
        _canonical_json_bytes(manifest["binding"]) != _canonical_json_bytes(binding)
        or manifest["plan_sha256"] != binding["plan_sha256"]
        or manifest["implementation_sha256"] != binding["implementation_sha256"]
        or manifest["captured_pointers"] != binding["captured_pointers"]
        or manifest["immutable_refs"] != binding["immutable_refs"]
    ):
        _fail("SUCCESSOR_MANIFEST_BINDING_MISMATCH")
    _validate_file_ref(root, manifest["binding_ref"])
    if expected_implementation_sha256 is not None and binding[
        "implementation_sha256"
    ] != _hex_sha256(expected_implementation_sha256, label="expected_implementation_sha256"):
        _fail("SUCCESSOR_IMPLEMENTATION_IDENTITY_MISMATCH")
    plan = binding["plan"]
    requests = replay_successor_support_requests(plan)
    if manifest["request_topology_sha256"] != plan["request_topology_sha256"]:
        _fail("SUCCESSOR_MANIFEST_TOPOLOGY_MISMATCH")
    record_refs = manifest["record_files"]
    receipts = manifest["request_receipts"]
    if (
        type(record_refs) is not list
        or type(receipts) is not list
        or len(record_refs) != len(requests)
        or len(receipts) != len(requests)
    ):
        _fail("SUCCESSOR_MANIFEST_RECORD_SET_INVALID")
    expected_record_names = {f"{request['ordinal']:06d}.json" for request in requests}
    if {entry.name for entry in records_root.iterdir()} != expected_record_names:
        _fail("SUCCESSOR_MANIFEST_RECORD_SET_INVALID")
    validated_receipts: list[dict[str, Any]] = []
    for request, ref, manifest_receipt in zip(requests, record_refs, receipts, strict=True):
        payload = _validate_file_ref(root, ref)
        if (
            ref.get("ordinal") != request["ordinal"]
            or ref.get("request_key") != request["request_key"]
            or ref.get("path") != f"requests/{request['ordinal']:06d}.json"
        ):
            _fail("SUCCESSOR_MANIFEST_RECORD_REF_INVALID")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError):
            _fail("SUCCESSOR_EVIDENCE_JSON_INVALID")
        receipt, _observed_rows, _rows = _decode_record(
            value,
            binding=binding,
            expected_request=request,
        )
        if _canonical_json_bytes(receipt) != _canonical_json_bytes(manifest_receipt):
            _fail("SUCCESSOR_MANIFEST_RECEIPT_MISMATCH")
        validated_receipts.append(receipt)
    if {entry.name for entry in unsupported_root.iterdir()} != {"inventory.json"}:
        _fail("SUCCESSOR_UNSUPPORTED_INVENTORY_SET_INVALID")
    inventory_payload = _validate_file_ref(
        root,
        manifest["unsupported_inventory_ref"],
    )
    try:
        inventory_value = json.loads(inventory_payload.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        _fail("SUCCESSOR_UNSUPPORTED_INVENTORY_INVALID")
    inventory = _validate_unsupported_inventory(inventory_value)
    expected_inventory = _unsupported_inventory(validated_receipts)
    if _canonical_json_bytes(inventory) != _canonical_json_bytes(
        expected_inventory
    ):
        _fail("SUCCESSOR_UNSUPPORTED_INVENTORY_RECORD_CLOSURE_MISMATCH")
    expected_authoritative = inventory["deferred_observation_count"] == 0
    if (
        manifest["authoritative_source_ready"] is not expected_authoritative
        or manifest["staging_eligible"] is not expected_authoritative
        or manifest["promotion_eligible"] is not expected_authoritative
        or manifest["authority_state"]
        != (
            _AUTHORITATIVE_AUTHORITY_STATE
            if expected_authoritative
            else _DEFERRED_AUTHORITY_STATE
        )
    ):
        _fail("SUCCESSOR_SOURCE_AUTHORITY_STATE_MISMATCH")
    if set(manifest["table_files"]) != set(_TABLES):
        _fail("SUCCESSOR_MANIFEST_TABLE_SET_INVALID")
    if {entry.name for entry in tables_root.iterdir()} != {
        f"{table}.parquet" for table in _TABLES
    }:
        _fail("SUCCESSOR_MANIFEST_TABLE_SET_INVALID")
    validation_estimates = _table_memory_estimates(
        requests=requests,
        receipts=validated_receipts,
        record_refs=record_refs,
    )
    validation_budgets = {
        table: _external_sort_budget(
            record_bytes=validation_estimates[table]["record_bytes"],
            accepted_rows=validation_estimates[table]["accepted_row_count"],
        )
        for table in _TABLES
    }
    validation_parent = root.parent
    if os.stat(validation_parent).st_dev != os.stat(root).st_dev:
        _fail("SUCCESSOR_VALIDATION_DEVICE_MISMATCH")
    _require_external_sort_reserve(
        validation_parent,
        minimum_free_disk_bytes=binding["resource_policy"]["minimum_free_disk_bytes"],
        required_working_bytes=max(
            budget["total_bytes"] for budget in validation_budgets.values()
        ),
    )
    validation_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.validation-",
            dir=validation_parent,
        )
    )
    os.chmod(validation_directory, 0o700)
    try:
        for table in _TABLES:
            ref = manifest["table_files"][table]
            if ref.get("path") != f"tables/{table}.parquet":
                _fail("SUCCESSOR_MANIFEST_TABLE_REF_INVALID")
            path = _validate_streamed_file_ref(root, ref)
            metadata = ref.get("metadata")
            if not isinstance(metadata, Mapping):
                _fail("SUCCESSOR_MANIFEST_TABLE_REF_INVALID")
            validated_metadata = _validate_streamed_table(
                path,
                table=table,
                metadata=metadata,
            )
            expected_path = validation_directory / f"{table}.parquet"
            expected_metadata = _build_streamed_table(
                table=table,
                row_batches=_record_row_batches(
                    table=table,
                    requests=requests,
                    records_root=records_root,
                    binding=binding,
                ),
                destination=expected_path,
                minimum_free_disk_bytes=binding["resource_policy"][
                    "minimum_free_disk_bytes"
                ],
                external_sort_budget=validation_budgets[table],
            )
            if (
                _canonical_json_bytes(validated_metadata)
                != _canonical_json_bytes(expected_metadata)
                or _regular_file_identity(path)
                != _regular_file_identity(expected_path)
            ):
                _fail("SUCCESSOR_TABLE_RECORD_CLOSURE_MISMATCH")
            if (
                ref.get("row_count") != validated_metadata["row_count"]
                or ref.get("table_sha256") != validated_metadata["table_sha256"]
                or manifest["table_fingerprints"].get(table)
                != validated_metadata["fingerprint_sha256"]
            ):
                _fail("SUCCESSOR_MANIFEST_TABLE_REF_INVALID")
            os.unlink(expected_path)
    finally:
        for table in _TABLES:
            expected_path = validation_directory / f"{table}.parquet"
            if expected_path.exists():
                try:
                    os.unlink(expected_path)
                except OSError:
                    pass
        try:
            os.rmdir(validation_directory)
        except OSError:
            pass
    accounting = manifest["provider_accounting"]
    expected_scope_projection = _scope_projection_manifest(
        plan=plan,
        receipts=validated_receipts,
        table_fingerprints=manifest["table_fingerprints"],
    )
    if (
        accounting["requests_terminal"] != len(validated_receipts)
        or accounting["requests_succeeded"]
        != sum(receipt["status"] == "AVAILABLE" for receipt in validated_receipts)
        or accounting["requests_empty"]
        != sum(receipt["status"] == "EMPTY" for receipt in validated_receipts)
        or accounting["request_attempts"]
        != sum(receipt["attempts"] for receipt in validated_receipts)
        or accounting["retryable_attempt_failures"]
        != sum(len(receipt["retry_error_codes"]) for receipt in validated_receipts)
        or accounting["full_response_observation_rows"]
        != sum(
            receipt["full_response_observation_count"]
            for receipt in validated_receipts
        )
        or accounting["in_scope_observation_rows"]
        != sum(receipt["in_scope_observation_count"] for receipt in validated_receipts)
        or accounting["scope_excluded_rows"]
        != sum(
            receipt["out_of_scope_observation_count"]
            for receipt in validated_receipts
        )
        or accounting["scope_exclusion_requests"]
        != sum(
            receipt["out_of_scope_observation_count"] > 0
            for receipt in validated_receipts
        )
        or accounting["raw_response_bytes"]
        != sum(receipt["raw_response_byte_length"] for receipt in validated_receipts)
        or _canonical_json_bytes(accounting["opaque_comp_type"])
        != _canonical_json_bytes(
            _opaque_comp_type_accounting(validated_receipts)
        )
        or _canonical_json_bytes(manifest["scope_projection"])
        != _canonical_json_bytes(expected_scope_projection)
    ):
        _fail("SUCCESSOR_PROVIDER_ACCOUNTING_NOT_RECONCILED")
    expected_resource = _resource_accounting(
        resource_policy=binding["resource_policy"],
        requests=requests,
        receipts=validated_receipts,
        record_refs=record_refs,
        table_refs=manifest["table_files"],
        minimum_observed_free_disk_bytes=manifest["resource_accounting"][
            "minimum_observed_free_disk_bytes"
        ],
    )
    if _canonical_json_bytes(expected_resource) != _canonical_json_bytes(
        manifest["resource_accounting"]
    ):
        _fail("SUCCESSOR_RESOURCE_ACCOUNTING_INVALID")
    return manifest


def validate_successor_support_fileset(
    fileset_root: str | Path,
    *,
    expected_implementation_sha256: str | None = None,
) -> dict[str, Any]:
    """Reject deferred capture evidence at the authoritative source boundary."""

    manifest = validate_successor_capture_fileset(
        fileset_root,
        expected_implementation_sha256=expected_implementation_sha256,
    )
    if (
        manifest.get("authority_state") != _AUTHORITATIVE_AUTHORITY_STATE
        or manifest.get("authoritative_source_ready") is not True
        or manifest.get("staging_eligible") is not True
        or manifest.get("promotion_eligible") is not True
    ):
        _fail("SUCCESSOR_DEFERRED_CAPTURE_NOT_AUTHORITATIVE")
    return manifest


class _LazySupportTables(Mapping[str, object]):
    """Validated path-backed store; aggregate frame access is forbidden."""

    def __init__(
        self,
        fileset_root: str | Path,
        *,
        capture_only: bool = False,
    ) -> None:
        validator = (
            validate_successor_capture_fileset
            if capture_only
            else validate_successor_support_fileset
        )
        self._manifest = validator(fileset_root)
        self._root = _private_root(fileset_root, create=False)

    def __iter__(self) -> Iterator[str]:
        return iter(_TABLES)

    def __len__(self) -> int:
        return len(_TABLES)

    def __getitem__(self, table: str) -> object:
        if table not in _TABLES:
            raise KeyError(table)
        _fail("SUCCESSOR_FULL_TABLE_ACCESS_FORBIDDEN")

    @property
    def table_fingerprints(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._manifest["table_fingerprints"]))

    def table_metadata(self, table: str) -> Mapping[str, Any]:
        if table not in _TABLES:
            raise KeyError(table)
        metadata = dict(self._manifest["table_files"][table].get("metadata", {}) or {})
        _validate_streamed_table(
            _validate_streamed_file_ref(
                self._root,
                self._manifest["table_files"][table],
            ),
            table=table,
            metadata=metadata,
        )
        return MappingProxyType(metadata)

    def iter_batches(
        self,
        table: str,
        *,
        batch_rows: int = _MAX_STREAM_BATCH_ROWS,
        batch_bytes: int = _MAX_STREAM_BATCH_BYTES,
    ) -> Iterator[pd.DataFrame]:
        if table not in _TABLES:
            raise KeyError(table)
        if (
            type(batch_rows) is not int
            or not 0 < batch_rows <= _MAX_STREAM_BATCH_ROWS
            or type(batch_bytes) is not int
            or not 0 < batch_bytes <= _MAX_STREAM_BATCH_BYTES
        ):
            _fail("SUCCESSOR_STREAM_BATCH_POLICY_INVALID")
        path = _validate_streamed_file_ref(
            self._root,
            self._manifest["table_files"][table],
        )
        for rows in _iter_parquet_rows(
            path,
            table=table,
            maximum_batch_rows=batch_rows,
            maximum_batch_bytes=batch_bytes,
        ):
            yield pd.DataFrame(rows, columns=_output_fields(table))

    def iter_rows(self, table: str) -> Iterator[dict[str, Any]]:
        for batch in self.iter_batches(table):
            yield from batch.to_dict("records")

    def materialize_table(self, table: str) -> pd.DataFrame:
        frames = list(self.iter_batches(table))
        if not frames:
            return pd.DataFrame(columns=_output_fields(table))
        return pd.concat(frames, ignore_index=True, sort=False)


def open_support_tables(fileset_root: str | Path) -> Mapping[str, object]:
    """Open a validated streaming store with no aggregate ``__getitem__``."""

    return _LazySupportTables(fileset_root)


def open_capture_support_tables(fileset_root: str | Path) -> Mapping[str, object]:
    """Open accepted-only tables for diagnostic taint replay only."""

    return _LazySupportTables(fileset_root, capture_only=True)


def load_support_tables(fileset_root: str | Path) -> dict[str, pd.DataFrame]:
    """Test-only compatibility helper that materialises all support tables."""

    tables = open_support_tables(fileset_root)
    if not isinstance(tables, _LazySupportTables):  # pragma: no cover
        _fail("SUCCESSOR_SUPPORT_STORE_INVALID")
    return {table: tables.materialize_table(table) for table in _TABLES}


def load_capture_support_tables(
    fileset_root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Test helper for a validated capture-only table store."""

    tables = open_capture_support_tables(fileset_root)
    if not isinstance(tables, _LazySupportTables):  # pragma: no cover
        _fail("SUCCESSOR_CAPTURE_STORE_INVALID")
    return {table: tables.materialize_table(table) for table in _TABLES}


def load_capture_symbol_rows(
    fileset_root: str | Path,
    *,
    table: str,
    symbol: str,
    maximum_rows: int = 100_000,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Decode one symbol from a validated capture without a full-table load."""

    if (
        table not in _TABLES
        or _TS_CODE_RE.fullmatch(symbol) is None
        or type(maximum_rows) is not int
        or maximum_rows < 1
        or type(maximum_bytes) is not int
        or maximum_bytes < 1
    ):
        _fail("SUCCESSOR_CAPTURE_SYMBOL_QUERY_INVALID")
    manifest = validate_successor_capture_fileset(fileset_root)
    root = _private_root(fileset_root, create=False)
    path = _validate_streamed_file_ref(root, manifest["table_files"][table])
    try:
        physical = pq.read_table(path, filters=[("ts_code", "=", symbol)])
    except (OSError, pa.ArrowException):
        _fail("SUCCESSOR_CAPTURE_SYMBOL_QUERY_FAILED")
    if physical.num_rows > maximum_rows or physical.nbytes > maximum_bytes:
        _fail("SUCCESSOR_CAPTURE_SYMBOL_RESOURCE_LIMIT")
    fields = _output_fields(table)
    rows: list[dict[str, Any]] = []
    values = physical.to_pydict()
    for observed_symbol, sort_date, end_date, raw_token in zip(
        values["ts_code"],
        values["sort_date"],
        values["end_date"],
        values["row_json"],
        strict=True,
    ):
        row = _decode_row_token(bytes(raw_token), fields=fields)
        expected_sort = str(
            row["trade_date"]
            if table == "daily_basic"
            else row["availability_date"]
        )
        expected_end = "" if table == "daily_basic" else str(row["end_date"])
        if (
            observed_symbol != symbol
            or row["ts_code"] != symbol
            or sort_date != expected_sort
            or end_date != expected_end
        ):
            _fail("SUCCESSOR_TABLE_INDEX_COLUMN_MISMATCH")
        rows.append(row)
    rows.sort(key=lambda value: _row_sort_key(value, fields))
    return rows


def load_unsupported_inventory(
    fileset_root: str | Path,
) -> dict[str, Any]:
    """Read the independently validated deferred-observation inventory."""

    manifest = validate_successor_capture_fileset(fileset_root)
    root = _private_root(fileset_root, create=False)
    payload = _validate_file_ref(root, manifest["unsupported_inventory_ref"])
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        _fail("SUCCESSOR_UNSUPPORTED_INVENTORY_INVALID")
    return _validate_unsupported_inventory(value)


def iter_unsupported_observations(
    fileset_root: str | Path,
) -> Iterator[dict[str, Any]]:
    """Yield decoded deferred winners without materialising provider tables."""

    inventory = load_unsupported_inventory(fileset_root)
    fields = _output_fields("balancesheet")
    for entry in inventory["entries"]:
        typed = entry["typed_row"]
        if len(typed) != len(fields):
            _fail("SUCCESSOR_UNSUPPORTED_INVENTORY_INVALID")
        row = dict(
            zip(
                fields,
                (_decode_typed_scalar(value) for value in typed),
                strict=True,
            )
        )
        yield {**dict(entry), "row": row}


def successor_support_evidence_paths(
    fileset_root: str | Path,
) -> dict[str, Path]:
    """Return validated absolute evidence paths without loading their bytes."""

    manifest = validate_successor_capture_fileset(fileset_root)
    root = _private_root(fileset_root, create=False)
    paths: dict[str, Path] = {"binding.json": root / "binding.json"}
    for ref in manifest["record_files"]:
        paths[str(ref["path"])] = root / str(ref["path"])
    for table in _TABLES:
        ref = manifest["table_files"][table]
        paths[str(ref["path"])] = root / str(ref["path"])
    unsupported_ref = manifest["unsupported_inventory_ref"]
    paths[str(unsupported_ref["path"])] = root / str(
        unsupported_ref["path"]
    )
    paths["provider_manifest.json"] = root / "provider_manifest.json"
    return paths


def iter_successor_support_evidence(
    fileset_root: str | Path,
) -> Iterator[tuple[str, bytes]]:
    """Yield every validated evidence file without retaining the bundle."""

    manifest = validate_successor_support_fileset(fileset_root)
    root = _private_root(fileset_root, create=False)
    yield "binding.json", _validate_file_ref(root, manifest["binding_ref"])
    for ref in manifest["record_files"]:
        yield ref["path"], _validate_file_ref(root, ref)
    for table in _TABLES:
        ref = manifest["table_files"][table]
        path = _validate_streamed_file_ref(root, ref)
        yield ref["path"], _regular_bytes(path)
    manifest_bytes = _regular_bytes(root / "provider_manifest.json")
    if _canonical_json_bytes(manifest) != manifest_bytes:
        _fail("SUCCESSOR_PROVIDER_MANIFEST_CHANGED")
    yield "provider_manifest.json", manifest_bytes


def capture_successor_support_evidence(
    fileset_root: str | Path,
) -> dict[str, bytes]:
    """Return a validated exact-byte evidence mapping for permanent staging."""

    return dict(iter_successor_support_evidence(fileset_root))


__all__ = [
    "FUNDAMENTAL_SUCCESSOR_BINDING_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_CLASSIFICATION_PARTITION_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY",
    "FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_RECORD_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_REQUEST_RECEIPT_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_SUPPORT_PLAN_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_TABLE_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_UNSUPPORTED_INVENTORY_SCHEMA",
    "FundamentalSuccessorSourceError",
    "SUCCESSOR_ENDPOINT_CAPABILITIES",
    "SUCCESSOR_SUPPORT_CANONICALIZATION_VERSION",
    "SUCCESSOR_SUPPORT_FILESET_VERSION",
    "SUCCESSOR_SUPPORT_PLAN_VERSION",
    "SUCCESSOR_SUPPORT_PROVIDER_MANIFEST_VERSION",
    "SUCCESSOR_SUPPORT_REQUEST_RECEIPT_VERSION",
    "SuccessorTushareClient",
    "acquire_successor_support",
    "build_successor_support_plan",
    "capture_successor_support_evidence",
    "iter_successor_support_evidence",
    "iter_unsupported_observations",
    "load_capture_support_tables",
    "load_capture_symbol_rows",
    "load_support_tables",
    "load_unsupported_inventory",
    "open_capture_support_tables",
    "open_support_tables",
    "replay_successor_support_requests",
    "successor_support_evidence_paths",
    "validate_successor_failure_evidence",
    "validate_successor_capture_fileset",
    "validate_successor_support_fileset",
]
