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
import shutil
import stat
import tempfile
import time
from types import MappingProxyType
from typing import Any, Final, NoReturn, Protocol

import pandas as pd

from quant_investor.market.tushare_transport import (
    TushareHttpsError,
    TushareResponse,
    replay_tushare_response_bytes,
)

FUNDAMENTAL_SUCCESSOR_SUPPORT_PLAN_SCHEMA: Final = "myquant-fundamental-successor-support-plan.v3"
FUNDAMENTAL_SUCCESSOR_REQUEST_RECEIPT_SCHEMA: Final = (
    "myquant-fundamental-successor-request-receipt.v3"
)
FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA: Final = (
    "myquant-fundamental-successor-support-fileset.v3"
)
FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY: Final = (
    "myquant-fundamental-successor-canonicalization.v2"
)
FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA: Final = (
    "cn-fundamental-successor-provider-manifest.v3"
)
FUNDAMENTAL_SUCCESSOR_BINDING_SCHEMA: Final = "myquant-fundamental-successor-support-binding.v3"
FUNDAMENTAL_SUCCESSOR_RECORD_SCHEMA: Final = "myquant-fundamental-successor-support-record.v3"
FUNDAMENTAL_SUCCESSOR_TABLE_SCHEMA: Final = "myquant-fundamental-successor-canonical-table.v2"
FUNDAMENTAL_SUCCESSOR_REQUEST_ENVELOPE_SCOPE_SCHEMA: Final = (
    "myquant-fundamental-successor-request-envelope-scope.v1"
)
FUNDAMENTAL_SUCCESSOR_CANONICAL_SUBJECT_SCOPE_SCHEMA: Final = (
    "myquant-fundamental-successor-canonical-subject-scope.v1"
)

SUCCESSOR_SUPPORT_PLAN_VERSION: Final = FUNDAMENTAL_SUCCESSOR_SUPPORT_PLAN_SCHEMA
SUCCESSOR_SUPPORT_REQUEST_RECEIPT_VERSION: Final = FUNDAMENTAL_SUCCESSOR_REQUEST_RECEIPT_SCHEMA
SUCCESSOR_SUPPORT_FILESET_VERSION: Final = FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA
SUCCESSOR_SUPPORT_CANONICALIZATION_VERSION: Final = FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY
SUCCESSOR_SUPPORT_PROVIDER_MANIFEST_VERSION: Final = FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA

_HEX_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_DATE_RE: Final = re.compile(r"^[0-9]{8}$", re.ASCII)
_TS_CODE_RE: Final = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)
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
    "balancesheet": "REPORT_TYPE_ONE_COMP_TYPE_ONE_TO_FOUR",
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
_DEFAULT_MAXIMUM_RECORD_BYTES: Final = 128 * 1024 * 1024
_DECODE_ESTIMATED_BYTES_PER_CELL: Final = 512
_MAX_TABLE_MEMORY_FRACTION: Final = Decimal("0.50")


class FundamentalSuccessorSourceError(RuntimeError):
    """A static-code acquisition or evidence-validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
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


def _scope_ref(body: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed(body, identity_field="scope_sha256")


def _request_envelope_scope_ref(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _scope_ref(
        {
            "identity_policy": "EXACT_CANONICAL_TS_CODE_NO_ALIASES",
            "partition_contract": "GLOBAL_PROVIDER_EXACT_PARTITION_ALL_SYMBOLS",
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
    return rows


def build_successor_support_plan(
    *,
    support_start: str,
    target_date: str,
    open_sessions: Sequence[str],
    symbols: Sequence[str],
    canonical_subject_scope_authority_sha256: str,
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
    requests = _build_request_rows(
        support_start=support,
        target_date=target,
        open_sessions=sessions,
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
            "capture_policy": "ALL_REPORT_PERIODS_FOR_EXACT_ANNOUNCEMENT_DATE",
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
    if set(sealed) != required:
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
    if type(value) is int and value in {0, 1, 2, 3, 4}:
        return str(value)
    if type(value) is str and value in {"0", "1", "2", "3", "4"}:
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
    symbol = _symbol(row["ts_code"])
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
        if (
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
        if report_type != "1" or comp_type not in {"1", "2", "3", "4"}:
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


def _canonicalize_rows(
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
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
            _fail("SUCCESSOR_MATERIAL_DUPLICATE_CONFLICT")
        equivalent = next(iter(projections.values()))
        projection_collapsed += len(equivalent) - 1
        # All business values are equal here.  Physical classification is the
        # only tie-break input; no content hash selects a winner.
        accepted.append(min(equivalent, key=lambda row: _row_sort_key(row, fields)))
    accepted.sort(key=lambda row: _row_sort_key(row, fields))
    return accepted, {
        "exact_duplicates_collapsed": exact_collapsed,
        "projection_equivalent_duplicates_collapsed": projection_collapsed,
        "superseded_updates_discarded": dominated,
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
        "scope_exclusion_policy": "FROZEN_CANONICAL_SUBJECT_PROJECTION.v2",
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
    body = {
        "accepted_count": len(logical_rows),
        "attempts": attempts,
        "blocker_codes": [],
        "canonicalization_counters": dict(counters),
        "endpoint": request["endpoint"],
        "has_more": False,
        "item_count": response.item_count,
        "logical_sha256": logical_sha256,
        "observation_sha256": observation_sha256,
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
    in_scope = [row for row in normalized if str(row["ts_code"]) in symbols]
    scope_excluded = [row for row in normalized if str(row["ts_code"]) not in symbols]
    logical, counters = _canonicalize_rows(request["table"], in_scope)
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


def _resource_policy(
    *,
    physical_memory_bytes: int,
    table_memory_limit_bytes: int,
    minimum_free_disk_bytes: int,
    maximum_record_bytes: int,
) -> dict[str, Any]:
    values = (
        physical_memory_bytes,
        table_memory_limit_bytes,
        minimum_free_disk_bytes,
        maximum_record_bytes,
    )
    if any(type(value) is not int or value < 1 for value in values):
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    maximum_table_memory = int(
        Decimal(physical_memory_bytes) * _MAX_TABLE_MEMORY_FRACTION
    )
    if (
        table_memory_limit_bytes > maximum_table_memory
        or maximum_record_bytes > table_memory_limit_bytes
        or minimum_free_disk_bytes < 64 * 1024 * 1024
    ):
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    return {
        "aggregate_table_memory_limit_bytes": int(
            Decimal(physical_memory_bytes) * Decimal("0.75")
        ),
        "decode_estimated_bytes_per_cell": _DECODE_ESTIMATED_BYTES_PER_CELL,
        "maximum_record_bytes": maximum_record_bytes,
        "minimum_free_disk_bytes": minimum_free_disk_bytes,
        "physical_memory_bytes": physical_memory_bytes,
        "schema_version": "myquant-fundamental-successor-resource-policy.v1",
        "table_memory_limit_bytes": table_memory_limit_bytes,
    }


def _validate_resource_policy(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "aggregate_table_memory_limit_bytes",
        "decode_estimated_bytes_per_cell",
        "maximum_record_bytes",
        "minimum_free_disk_bytes",
        "physical_memory_bytes",
        "schema_version",
        "table_memory_limit_bytes",
    }:
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    if (
        value["schema_version"]
        != "myquant-fundamental-successor-resource-policy.v1"
        or value["decode_estimated_bytes_per_cell"]
        != _DECODE_ESTIMATED_BYTES_PER_CELL
        or type(value["aggregate_table_memory_limit_bytes"]) is not int
        or value["aggregate_table_memory_limit_bytes"]
        != int(Decimal(value["physical_memory_bytes"]) * Decimal("0.75"))
    ):
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    expected = _resource_policy(
        physical_memory_bytes=value["physical_memory_bytes"],
        table_memory_limit_bytes=value["table_memory_limit_bytes"],
        minimum_free_disk_bytes=value["minimum_free_disk_bytes"],
        maximum_record_bytes=value["maximum_record_bytes"],
    )
    if expected != value:
        _fail("SUCCESSOR_RESOURCE_POLICY_INVALID")
    return expected


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
        or receipt["endpoint"] != expected_request["endpoint"]
        or receipt["table"] != expected_request["table"]
        or receipt["accepted_count"] != len(rows)
        or receipt["status"] != ("EMPTY" if not rows else "AVAILABLE")
        or receipt["has_more"] is not False
        or receipt["blocker_codes"] != []
    ):
        _fail("SUCCESSOR_RECEIPT_INVALID")
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


def _table_artifact(table: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = _output_fields(table)
    canonical_rows, counters = _canonicalize_rows(table, rows)
    encoded = [_typed_row(row, fields, logical=False) for row in canonical_rows]
    body = {
        "canonicalization_counters": counters,
        "fields": list(fields),
        "fingerprint_sha256": _sha256(
            _canonical_json_bytes(
                {
                    "fields": list(fields),
                    "rows": [_typed_row(row, fields, logical=True) for row in canonical_rows],
                }
            )
        ),
        "row_count": len(canonical_rows),
        "rows": encoded,
        "schema_version": FUNDAMENTAL_SUCCESSOR_TABLE_SCHEMA,
        "table": table,
    }
    return _sealed(body, identity_field="table_sha256")


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


def _decode_table_artifact(value: Mapping[str, Any], *, table: str) -> list[dict[str, Any]]:
    artifact = _validate_seal(value, identity_field="table_sha256")
    required = {
        "canonicalization_counters",
        "fields",
        "fingerprint_sha256",
        "row_count",
        "rows",
        "schema_version",
        "table",
        "table_sha256",
    }
    fields = _output_fields(table)
    if (
        set(artifact) != required
        or artifact["schema_version"] != FUNDAMENTAL_SUCCESSOR_TABLE_SCHEMA
        or artifact["table"] != table
        or artifact["fields"] != list(fields)
        or type(artifact["rows"]) is not list
        or artifact["row_count"] != len(artifact["rows"])
    ):
        _fail("SUCCESSOR_TABLE_ARTIFACT_INVALID")
    rows: list[dict[str, Any]] = []
    for encoded in artifact["rows"]:
        if type(encoded) is not list or len(encoded) != len(fields):
            _fail("SUCCESSOR_TABLE_ARTIFACT_INVALID")
        rows.append(
            dict(
                zip(
                    fields,
                    (_decode_typed_scalar(value) for value in encoded),
                    strict=True,
                )
            )
        )
    expected = _table_artifact(table, rows)
    if _canonical_json_bytes(expected) != _canonical_json_bytes(artifact):
        _fail("SUCCESSOR_TABLE_ARTIFACT_MISMATCH")
    return rows


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
        cell_count = row_count * len(_output_fields(table))
        # The allowance covers decoded scalar objects, row dictionaries,
        # canonical typed-row copies, JSON encoding, and one replayed record.
        estimated = (
            cell_count * _DECODE_ESTIMATED_BYTES_PER_CELL
            + largest_record_by_table[table] * 4
            + 16 * 1024 * 1024
        )
        estimates[table] = {
            "accepted_row_count": row_count,
            "estimated_peak_memory_bytes": estimated,
            "largest_record_bytes": largest_record_by_table[table],
            "record_bytes": record_bytes_by_table[table],
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
    aggregate = sum(
        row["estimated_peak_memory_bytes"] for row in estimates.values()
    )
    if (
        peak > policy["table_memory_limit_bytes"]
        or aggregate > policy["aggregate_table_memory_limit_bytes"]
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
        "schema_version": "myquant-fundamental-successor-resource-accounting.v1",
        "source_payload_bytes": source_payload_bytes,
        "status": "PASS",
        "table_estimates": estimates,
    }
    return _sealed(body, identity_field="resource_sha256")


def _file_ref(path: str, payload: bytes) -> dict[str, Any]:
    return {"byte_length": len(payload), "path": path, "sha256": _sha256(payload)}


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
    resolved_table_memory_limit = (
        int(Decimal(resolved_physical_memory) * _MAX_TABLE_MEMORY_FRACTION)
        if table_memory_limit_bytes is None
        else table_memory_limit_bytes
    )
    resource_policy = _resource_policy(
        physical_memory_bytes=resolved_physical_memory,
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
        manifest = validate_successor_support_fileset(root)
        installed_binding = _canonical_file_mapping(binding_path)
        if _canonical_json_bytes(installed_binding) != _canonical_json_bytes(expected_binding):
            _fail("SUCCESSOR_RESUME_BINDING_MISMATCH")
        return manifest
    if binding_path.exists():
        installed_binding = _validate_binding(_canonical_file_mapping(binding_path))
        if _canonical_json_bytes(installed_binding) != _canonical_json_bytes(expected_binding):
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
        or sum(
            row["estimated_peak_memory_bytes"]
            for row in table_estimates.values()
        )
        > resource_policy["aggregate_table_memory_limit_bytes"]
    ):
        _fail("SUCCESSOR_RESOURCE_PREFLIGHT_BLOCKED")
    table_refs: dict[str, dict[str, Any]] = {}
    table_fingerprints: dict[str, str] = {}
    for table in _TABLES:
        table_rows: list[dict[str, Any]] = []
        for request in requests:
            if request["table"] != table:
                continue
            record_path = records_root / f"{request['ordinal']:06d}.json"
            _receipt_value, _observed_rows, rows = _decode_record(
                _canonical_file_mapping(record_path),
                binding=binding,
                expected_request=request,
            )
            table_rows.extend(rows)
        artifact = _table_artifact(table, table_rows)
        path = tables_root / f"{table}.json"
        payload = _canonical_json_bytes(artifact)
        if len(payload) > resource_policy["table_memory_limit_bytes"]:
            _fail("SUCCESSOR_TABLE_RESOURCE_LIMIT_EXCEEDED")
        observed_free = _require_disk_reserve(
            root,
            minimum_free_disk_bytes=resource_policy["minimum_free_disk_bytes"],
            pending_bytes=len(payload),
        )
        minimum_observed_free_disk = min(minimum_observed_free_disk, observed_free)
        _atomic_write(path, payload)
        readback = _canonical_file_mapping(path)
        _decode_table_artifact(readback, table=table)
        table_refs[table] = _file_ref(f"tables/{table}.json", payload)
        table_refs[table]["row_count"] = artifact["row_count"]
        table_refs[table]["table_sha256"] = artifact["table_sha256"]
        table_fingerprints[table] = artifact["fingerprint_sha256"]
        del artifact, payload, table_rows
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
    body = {
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
        "status": "COMPLETE",
        "table_files": table_refs,
        "table_fingerprints": table_fingerprints,
    }
    manifest = _sealed(body, identity_field="manifest_sha256")
    _atomic_write(manifest_path, _canonical_json_bytes(manifest))
    validated = validate_successor_support_fileset(root)
    return validated


def _validate_manifest_shape(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _validate_seal(value, identity_field="manifest_sha256")
    required = {
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
        "schema_version",
        "scope_projection",
        "status",
        "table_files",
        "table_fingerprints",
    }
    if set(manifest) != required:
        _fail("SUCCESSOR_PROVIDER_MANIFEST_FIELDS_INVALID")
    if (
        manifest["schema_version"] != FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA
        or manifest["fileset_schema_version"] != FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA
        or manifest["status"] != "COMPLETE"
    ):
        _fail("SUCCESSOR_PROVIDER_MANIFEST_CONTRACT_MISMATCH")
    accounting = manifest["provider_accounting"]
    if type(accounting) is not dict or set(accounting) != {
        "full_response_observation_rows",
        "has_more_requests",
        "in_scope_observation_rows",
        "malformed_requests",
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
    if any(type(value) is not int or value < 0 for value in accounting.values()):
        _fail("SUCCESSOR_PROVIDER_ACCOUNTING_INVALID")
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
    return manifest


def validate_successor_support_fileset(
    fileset_root: str | Path,
    *,
    expected_implementation_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently validate the complete fileset and exact byte closure."""

    root = _private_root(fileset_root, create=False)
    records_root = _safe_directory(root, "requests", create=False)
    tables_root = _safe_directory(root, "tables", create=False)
    if {entry.name for entry in root.iterdir()} != {
        "binding.json",
        "provider_manifest.json",
        "requests",
        "tables",
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
    rows_by_table: dict[str, list[dict[str, Any]]] = {table: [] for table in _TABLES}
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
        receipt, _observed_rows, rows = _decode_record(
            value,
            binding=binding,
            expected_request=request,
        )
        if _canonical_json_bytes(receipt) != _canonical_json_bytes(manifest_receipt):
            _fail("SUCCESSOR_MANIFEST_RECEIPT_MISMATCH")
        validated_receipts.append(receipt)
        rows_by_table[request["table"]].extend(rows)
    if set(manifest["table_files"]) != set(_TABLES):
        _fail("SUCCESSOR_MANIFEST_TABLE_SET_INVALID")
    if {entry.name for entry in tables_root.iterdir()} != {f"{table}.json" for table in _TABLES}:
        _fail("SUCCESSOR_MANIFEST_TABLE_SET_INVALID")
    for table in _TABLES:
        ref = manifest["table_files"][table]
        payload = _validate_file_ref(root, ref)
        if ref.get("path") != f"tables/{table}.json":
            _fail("SUCCESSOR_MANIFEST_TABLE_REF_INVALID")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError):
            _fail("SUCCESSOR_EVIDENCE_JSON_INVALID")
        table_rows = _decode_table_artifact(value, table=table)
        expected_artifact = _table_artifact(table, rows_by_table[table])
        if _canonical_json_bytes(value) != _canonical_json_bytes(expected_artifact):
            _fail("SUCCESSOR_TABLE_RECORD_CLOSURE_MISMATCH")
        if (
            ref.get("row_count") != value["row_count"]
            or ref.get("table_sha256") != value["table_sha256"]
            or manifest["table_fingerprints"].get(table) != value["fingerprint_sha256"]
            or len(table_rows) != value["row_count"]
        ):
            _fail("SUCCESSOR_MANIFEST_TABLE_REF_INVALID")
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
        or _canonical_json_bytes(manifest["scope_projection"])
        != _canonical_json_bytes(expected_scope_projection)
    ):
        _fail("SUCCESSOR_PROVIDER_ACCOUNTING_NOT_RECONCILED")
    return manifest


def load_support_tables(fileset_root: str | Path) -> dict[str, pd.DataFrame]:
    """Return validated canonical support tables as independent DataFrames."""

    manifest = validate_successor_support_fileset(fileset_root)
    root = _private_root(fileset_root, create=False)
    result: dict[str, pd.DataFrame] = {}
    for table in _TABLES:
        payload = _validate_file_ref(root, manifest["table_files"][table])
        value = json.loads(payload.decode("utf-8"))
        rows = _decode_table_artifact(value, table=table)
        result[table] = pd.DataFrame(rows, columns=_output_fields(table))
    return result


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
        yield ref["path"], _validate_file_ref(root, ref)
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
    "FUNDAMENTAL_SUCCESSOR_CANONICALIZATION_POLICY",
    "FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_RECORD_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_REQUEST_RECEIPT_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_SUPPORT_FILESET_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_SUPPORT_PLAN_SCHEMA",
    "FUNDAMENTAL_SUCCESSOR_TABLE_SCHEMA",
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
    "load_support_tables",
    "replay_successor_support_requests",
    "validate_successor_support_fileset",
]
