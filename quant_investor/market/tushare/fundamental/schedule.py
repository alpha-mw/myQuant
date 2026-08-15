"""Exact five-year daily and seven-year financial request schedule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import hashlib
from typing import Any

import pandas as pd

from .._core import (
    common_fields,
    content_ref,
    exact_ref,
    require_exact_keys,
    seal,
    session_date,
    sha256,
    timestamp,
    validate_seal,
)
from ..contracts import validate_endpoint_execution_plan
from .models import (
    EXECUTION_CLOSURE_KIND,
    FINANCIAL_ENDPOINTS,
    REQUEST_PLAN_KIND,
    SOURCE_ENDPOINTS,
    FundamentalAcquisitionError,
    fundamental_contract,
)

_EXECUTION_FIELDS = {
    "authority",
    "closure_id",
    "contract_sha256",
    "created_at",
    "endpoint_plans",
    "kind",
    "production",
    "request_plan",
    "research_only",
    "semantic_sha256",
    "timestamp",
}

_FIELDS = {
    "as_of",
    "authority",
    "baseline_empty_partition_keyset",
    "baseline_network_attempts",
    "baseline_provider_manifest_ref",
    "contract_sha256",
    "created_at",
    "daily_open_sessions",
    "daily_start",
    "endpoint_plan_refs",
    "financial_periods",
    "financial_start",
    "implementation_sha256",
    "kind",
    "market_calendar_ref",
    "market_scope_ref",
    "max_attempts_per_partition",
    "partition_rows",
    "pit_cutoff",
    "plan_id",
    "planned_max_network_attempts",
    "planned_terminal_request_count",
    "production",
    "research_only",
    "semantic_sha256",
    "strict_decimal_decode",
    "symbol_set_sha256",
    "symbols",
    "timestamp",
    "window_years",
}


def _date(value: Any, *, label: str) -> str:
    return session_date(value, label=label)


def _symbols(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FundamentalAcquisitionError("symbols must be a sequence")
    rows = list(values)
    if not rows or len(rows) > 10_000:
        raise FundamentalAcquisitionError("symbol scope cardinality is invalid")
    normalized: list[str] = []
    for value in rows:
        if type(value) is not str or not value or not value.isascii():
            raise FundamentalAcquisitionError("symbol scope contains an invalid value")
        normalized.append(value)
    expected = sorted(normalized, key=lambda item: item.encode("ascii"))
    if normalized != expected or len(normalized) != len(set(normalized)):
        raise FundamentalAcquisitionError("symbol scope must be ASCII-sorted unique")
    return normalized


def _open_sessions(values: Sequence[str], *, start: str, end: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FundamentalAcquisitionError("daily_open_sessions must be a sequence")
    rows = [_date(value, label="daily_open_session") for value in values]
    parsed = [datetime.strptime(value, "%Y%m%d").date() for value in rows]
    if (
        not rows
        or len(rows) < 900
        or len(rows) > 1500
        or rows != sorted(rows)
        or len(rows) != len(set(rows))
        or rows[0] < start
        or rows[-1] != end
        or any(value < start or value > end for value in rows)
        or any((right - left).days > 14 for left, right in zip(parsed, parsed[1:]))
    ):
        raise FundamentalAcquisitionError("daily open-session closure is invalid")
    return rows


def _quarter_ends(*, start: str, end: str) -> list[str]:
    lower = datetime.strptime(start, "%Y%m%d").date()
    upper = datetime.strptime(end, "%Y%m%d").date()
    rows: list[str] = []
    for year in range(lower.year, upper.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            value = date(year, month, day)
            if lower <= value <= upper:
                rows.append(value.strftime("%Y%m%d"))
    return rows


def _validated_endpoint_plans(
    value: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    if type(value) is not dict or set(value) != set(SOURCE_ENDPOINTS):
        raise FundamentalAcquisitionError("endpoint plan set is invalid")
    plans: dict[str, dict[str, Any]] = {}
    refs: dict[str, dict[str, str]] = {}
    for table in sorted(SOURCE_ENDPOINTS):
        plan = validate_endpoint_execution_plan(value[table])
        if (
            plan["api_name"] != SOURCE_ENDPOINTS[table]
            or plan["permission_class"] != "POINTS"
            or plan["lane"] != "FUNDAMENTAL"
            or plan["strict_decimal_decode"] is not True
        ):
            raise FundamentalAcquisitionError("endpoint plan authority mismatch")
        plans[table] = plan
        refs[table] = content_ref(plan, identity_field="plan_id")
    return plans, refs


def _partition_rows(
    *, financial_periods: Sequence[str], daily_sessions: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for table in sorted(FINANCIAL_ENDPOINTS):
        for period in financial_periods:
            rows.append(
                {
                    "endpoint": FINANCIAL_ENDPOINTS[table],
                    "ordinal": ordinal,
                    "partition_id": f"period={period}",
                    "partition_type": "PERIOD",
                    "table": table,
                }
            )
            ordinal += 1
    for trade_date in daily_sessions:
        rows.append(
            {
                "endpoint": "daily_basic",
                "ordinal": ordinal,
                "partition_id": f"trade_date={trade_date}",
                "partition_type": "TRADE_DATE",
                "table": "daily_basic",
            }
        )
        ordinal += 1
    return rows


def _baseline_empty_keyset(
    values: Sequence[str],
    *,
    partitions: Sequence[Mapping[str, Any]],
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FundamentalAcquisitionError("baseline_empty_partition_keyset must be a sequence")
    rows = list(values)
    expected = sorted(rows, key=lambda item: item.encode("ascii"))
    admitted = {f"{row['table']}|{row['partition_id']}" for row in partitions}
    if (
        rows != expected
        or len(rows) != len(set(rows))
        or any(type(item) is not str or item not in admitted for item in rows)
    ):
        raise FundamentalAcquisitionError("baseline empty partition keyset is invalid")
    return rows


@fundamental_contract
def build_fundamental_request_plan(
    *,
    as_of: str,
    pit_cutoff: str,
    symbols: Sequence[str],
    canonical_open_sessions: Sequence[str],
    market_scope_ref: Mapping[str, Any],
    market_calendar_ref: Mapping[str, Any],
    baseline_provider_manifest_ref: Mapping[str, Any],
    baseline_network_attempts: int,
    baseline_empty_partition_keyset: Sequence[str],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    max_attempts_per_partition: int,
    implementation_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the exact inclusive Fundamental request schedule."""

    target = _date(as_of, label="as_of")
    cutoff = timestamp(pit_cutoff, label="pit_cutoff")
    created = timestamp(created_at, label="created_at")
    if cutoff > created or cutoff[:10].replace("-", "") > target:
        raise FundamentalAcquisitionError("PIT cutoff is future-dated")
    normalized_symbols = _symbols(symbols)
    target_ts = pd.Timestamp(datetime.strptime(target, "%Y%m%d"))
    daily_start = (target_ts - pd.DateOffset(years=5)).strftime("%Y%m%d")
    financial_start = (
        pd.Timestamp(datetime.strptime(daily_start, "%Y%m%d")) - pd.DateOffset(years=2)
    ).strftime("%Y%m%d")
    sessions = _open_sessions(
        canonical_open_sessions,
        start=daily_start,
        end=target,
    )
    periods = _quarter_ends(start=financial_start, end=target)
    if type(max_attempts_per_partition) is not int or not (1 <= max_attempts_per_partition <= 64):
        raise FundamentalAcquisitionError("max_attempts_per_partition is invalid")
    if type(baseline_network_attempts) is not int or baseline_network_attempts < 1:
        raise FundamentalAcquisitionError("baseline network attempts are invalid")
    plans, endpoint_refs = _validated_endpoint_plans(endpoint_plans)
    if any(plan["created_at"] > created for plan in plans.values()):
        raise FundamentalAcquisitionError("request plan contains future endpoint policy")
    scope_ref = exact_ref(market_scope_ref, label="market_scope_ref")
    calendar_ref = exact_ref(market_calendar_ref, label="market_calendar_ref")
    baseline_ref = exact_ref(
        baseline_provider_manifest_ref,
        label="baseline_provider_manifest_ref",
    )
    if (
        scope_ref["cutoff"] > cutoff
        or calendar_ref["cutoff"] > cutoff
        or baseline_ref["cutoff"] > cutoff
        or scope_ref["available_at"] > created
        or calendar_ref["available_at"] > created
        or baseline_ref["available_at"] > created
    ):
        raise FundamentalAcquisitionError("market closure contains future evidence")
    partitions = _partition_rows(
        financial_periods=periods,
        daily_sessions=sessions,
    )
    terminal_count = len(partitions)
    body = {
        **common_fields(timestamp_value=created),
        "as_of": target,
        "baseline_empty_partition_keyset": _baseline_empty_keyset(
            baseline_empty_partition_keyset,
            partitions=partitions,
        ),
        "baseline_network_attempts": baseline_network_attempts,
        "baseline_provider_manifest_ref": baseline_ref,
        "created_at": created,
        "daily_open_sessions": sessions,
        "daily_start": daily_start,
        "endpoint_plan_refs": endpoint_refs,
        "financial_periods": periods,
        "financial_start": financial_start,
        "implementation_sha256": sha256(
            implementation_sha256,
            label="implementation_sha256",
        ),
        "market_calendar_ref": calendar_ref,
        "market_scope_ref": scope_ref,
        "max_attempts_per_partition": max_attempts_per_partition,
        "partition_rows": partitions,
        "pit_cutoff": cutoff,
        "planned_max_network_attempts": terminal_count * max_attempts_per_partition,
        "planned_terminal_request_count": terminal_count,
        "strict_decimal_decode": True,
        "symbol_set_sha256": hashlib.sha256(
            "\n".join(normalized_symbols).encode("utf-8")
        ).hexdigest(),
        "symbols": normalized_symbols,
        "kind": REQUEST_PLAN_KIND,
        "window_years": {"daily": 5, "financial": 7},
    }
    return seal(body, identity_field="plan_id")


@fundamental_contract
def validate_fundamental_request_plan(
    document: Mapping[str, Any],
    *,
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="plan_id")
    require_exact_keys(value, _FIELDS, label="Fundamental request plan")
    if value.get("kind") != REQUEST_PLAN_KIND:
        raise FundamentalAcquisitionError("Fundamental request plan kind mismatch")
    expected = build_fundamental_request_plan(
        as_of=value["as_of"],
        pit_cutoff=value["pit_cutoff"],
        symbols=value["symbols"],
        canonical_open_sessions=value["daily_open_sessions"],
        market_scope_ref=value["market_scope_ref"],
        market_calendar_ref=value["market_calendar_ref"],
        baseline_provider_manifest_ref=value["baseline_provider_manifest_ref"],
        baseline_network_attempts=value["baseline_network_attempts"],
        baseline_empty_partition_keyset=value["baseline_empty_partition_keyset"],
        endpoint_plans=endpoint_plans,
        max_attempts_per_partition=value["max_attempts_per_partition"],
        implementation_sha256=value["implementation_sha256"],
        created_at=value["created_at"],
    )
    if value != expected:
        raise FundamentalAcquisitionError("Fundamental request plan replay mismatch")
    return value


@fundamental_contract
def build_fundamental_execution_closure(
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    """Persist every policy document required to replay the request plan."""

    validated = validate_fundamental_request_plan(
        plan,
        endpoint_plans=endpoint_plans,
    )
    plans, _refs = _validated_endpoint_plans(endpoint_plans)
    created = timestamp(created_at, label="created_at")
    if created < validated["created_at"]:
        raise FundamentalAcquisitionError("execution closure predates request plan")
    return seal(
        {
            **common_fields(timestamp_value=created),
            "created_at": created,
            "endpoint_plans": plans,
            "request_plan": validated,
            "kind": EXECUTION_CLOSURE_KIND,
        },
        identity_field="closure_id",
    )


@fundamental_contract
def validate_fundamental_execution_closure(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="closure_id")
    require_exact_keys(value, _EXECUTION_FIELDS, label="Fundamental execution closure")
    if value.get("kind") != EXECUTION_CLOSURE_KIND:
        raise FundamentalAcquisitionError("execution closure kind mismatch")
    expected = build_fundamental_execution_closure(
        plan=value["request_plan"],
        endpoint_plans=value["endpoint_plans"],
        created_at=value["created_at"],
    )
    if value != expected:
        raise FundamentalAcquisitionError("execution closure replay mismatch")
    return value


__all__ = [
    "build_fundamental_execution_closure",
    "build_fundamental_request_plan",
    "validate_fundamental_execution_closure",
    "validate_fundamental_request_plan",
]
