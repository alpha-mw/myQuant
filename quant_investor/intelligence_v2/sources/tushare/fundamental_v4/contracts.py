"""Physical, logical, raw-table, and provider evidence contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...._core import (
    common_fields,
    content_ref,
    exact_ref,
    identifier,
    require_exact_keys,
    seal,
    session_date,
    sha256,
    timestamp,
    validate_seal,
)
from ..contracts import validate_endpoint_execution_plan
from .models import (
    LOGICAL_COVERAGE_V4,
    LOGICAL_STATUSES,
    PHYSICAL_REQUEST_RECEIPT_V4,
    PHYSICAL_STATUSES,
    RAW_TABLE_EVIDENCE_V4,
    SOURCE_TABLES,
    FundamentalV4ContractError,
    fundamental_v4_contract,
)
from .schedule import validate_fundamental_request_plan_v4

_COMMON_FIELDS = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "version",
}
_PHYSICAL_FIELDS = _COMMON_FIELDS | {
    "accepted_count",
    "attempts",
    "blocker_codes",
    "endpoint",
    "has_more",
    "partition_id",
    "partition_type",
    "plan_ref",
    "provider_request_id",
    "raw_response_projection_sha256",
    "receipt_id",
    "reported_count",
    "sanitized_params_sha256",
    "status",
    "strict_decimal_decode",
    "table",
}
_LOGICAL_FIELDS = _COMMON_FIELDS | {
    "company_code",
    "coverage_id",
    "duplicate_reason_codes",
    "expected_end",
    "expected_partition_refs",
    "expected_start",
    "missing_reason_codes",
    "observed_end",
    "observed_start",
    "physical_receipt_refs",
    "plan_ref",
    "restatement_reason_codes",
    "row_count",
    "status",
    "table",
}
_RAW_FIELDS = _COMMON_FIELDS | {
    "canonical_multiset_sha256",
    "column_order",
    "duplicate_row_count",
    "evidence_id",
    "file_ref",
    "lane",
    "plan_ref",
    "row_count",
    "table",
    "winner_implementation_sha256",
}
_INCOMPLETE_BLOCKERS = {
    "COUNT_MISMATCH",
    "DUPLICATE_ROWS",
    "HAS_MORE",
    "ROW_LIMIT_HIT",
    "SCHEMA_MISMATCH",
    "SCOPE_MISMATCH",
}


def _sequence(value: Any, *, label: str, maximum: int) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FundamentalV4ContractError(f"{label} must be a sequence")
    rows = list(value)
    if len(rows) > maximum:
        raise FundamentalV4ContractError(f"{label} exceeds maximum cardinality")
    return rows


def _codes(value: Any, *, label: str, maximum: int = 64) -> list[str]:
    rows = _sequence(value, label=label, maximum=maximum)
    normalized: list[str] = []
    for item in rows:
        if (
            type(item) is not str
            or not item
            or not item.isascii()
            or not item.replace("_", "").isalnum()
            or item.upper() != item
        ):
            raise FundamentalV4ContractError(f"{label} contains an invalid code")
        normalized.append(item)
    expected = sorted(normalized, key=lambda item: item.encode("ascii"))
    if normalized != expected or len(normalized) != len(set(normalized)):
        raise FundamentalV4ContractError(f"{label} must be ASCII-sorted unique")
    return normalized


def _receipt_ids(value: Any, *, label: str) -> list[str]:
    rows = _sequence(value, label=label, maximum=2_000)
    normalized = [sha256(item, label=f"{label} item") for item in rows]
    expected = sorted(normalized, key=lambda item: item.encode("ascii"))
    if normalized != expected or len(normalized) != len(set(normalized)):
        raise FundamentalV4ContractError(f"{label} must be ASCII-sorted unique")
    return normalized


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FundamentalV4ContractError(f"{label} must be a nonnegative integer")
    return value


def _plan_and_endpoints(
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    validated_plan = validate_fundamental_request_plan_v4(
        plan,
        endpoint_plans=endpoint_plans,
    )
    endpoints = {
        table: validate_endpoint_execution_plan(document)
        for table, document in endpoint_plans.items()
    }
    return validated_plan, endpoints


def _partition(
    plan: Mapping[str, Any],
    *,
    table: str,
    partition_id: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in plan["partition_rows"]
        if row["table"] == table and row["partition_id"] == partition_id
    ]
    if len(rows) != 1:
        raise FundamentalV4ContractError("physical partition is not in the sealed plan")
    return dict(rows[0])


def _validate_physical_status(
    *,
    status: str,
    attempts: int,
    max_attempts: int,
    reported: int,
    accepted: int,
    has_more: bool,
    blockers: list[str],
    projection_sha: str | None,
    row_limit: int,
) -> None:
    if attempts < 1 or attempts > max_attempts:
        raise FundamentalV4ContractError("physical request attempts exceed policy")
    if status == "AVAILABLE" and (
        reported != accepted
        or accepted < 1
        or accepted >= row_limit
        or has_more
        or blockers
        or projection_sha is None
    ):
        raise FundamentalV4ContractError("AVAILABLE physical request is not closed")
    if status == "EMPTY" and (
        reported or accepted or has_more or blockers or projection_sha is None
    ):
        raise FundamentalV4ContractError("EMPTY physical request is not closed")
    if status == "INCOMPLETE" and (
        not blockers
        or any(code not in _INCOMPLETE_BLOCKERS for code in blockers)
        or projection_sha is None
    ):
        raise FundamentalV4ContractError("INCOMPLETE physical blockers are invalid")
    if status in {"PROVIDER_ERROR", "SCHEMA_MISMATCH", "TRANSPORT_ERROR"} and (
        reported or accepted or has_more or blockers or projection_sha is not None
    ):
        raise FundamentalV4ContractError("failed physical request claims response data")


def _build_physical_request_receipt_validated(
    *,
    validated_plan: Mapping[str, Any],
    validated_plan_ref: Mapping[str, str],
    endpoints: Mapping[str, Mapping[str, Any]],
    table: str,
    partition_id: str,
    sanitized_params_sha256: str,
    attempts: int,
    provider_request_id: str | None,
    reported_count: int,
    accepted_count: int,
    has_more: bool,
    status: str,
    blocker_codes: Sequence[str],
    raw_response_projection_sha256: str | None,
    captured_at: str,
) -> dict[str, Any]:
    if table not in SOURCE_TABLES or status not in PHYSICAL_STATUSES:
        raise FundamentalV4ContractError("physical table or status is invalid")
    partition = _partition(validated_plan, table=table, partition_id=partition_id)
    captured = timestamp(captured_at, label="captured_at")
    if captured < validated_plan["created_at"]:
        raise FundamentalV4ContractError("physical receipt predates its plan")
    normalized_attempts = _nonnegative_int(attempts, label="attempts")
    reported = _nonnegative_int(reported_count, label="reported_count")
    accepted = _nonnegative_int(accepted_count, label="accepted_count")
    if type(has_more) is not bool:
        raise FundamentalV4ContractError("has_more must be boolean")
    blockers = _codes(blocker_codes, label="blocker_codes")
    projection_sha = (
        None
        if raw_response_projection_sha256 is None
        else sha256(
            raw_response_projection_sha256,
            label="raw_response_projection_sha256",
        )
    )
    if provider_request_id is not None:
        provider_request_id = identifier(
            provider_request_id,
            label="provider_request_id",
        )
    _validate_physical_status(
        status=status,
        attempts=normalized_attempts,
        max_attempts=validated_plan["max_attempts_per_partition"],
        reported=reported,
        accepted=accepted,
        has_more=has_more,
        blockers=blockers,
        projection_sha=projection_sha,
        row_limit=endpoints[table]["documented_row_limit"],
    )
    if (
        status == "EMPTY"
        and f"{table}|{partition_id}" not in validated_plan["baseline_empty_partition_keyset"]
    ):
        raise FundamentalV4ContractError(
            "EMPTY physical request lacks sealed baseline identity proof"
        )
    body = {
        **common_fields(timestamp_value=captured),
        "accepted_count": accepted,
        "attempts": normalized_attempts,
        "blocker_codes": blockers,
        "endpoint": partition["endpoint"],
        "has_more": has_more,
        "partition_id": partition["partition_id"],
        "partition_type": partition["partition_type"],
        "plan_ref": dict(validated_plan_ref),
        "provider_request_id": provider_request_id,
        "raw_response_projection_sha256": projection_sha,
        "reported_count": reported,
        "sanitized_params_sha256": sha256(
            sanitized_params_sha256,
            label="sanitized_params_sha256",
        ),
        "status": status,
        "strict_decimal_decode": True,
        "table": table,
        "version": PHYSICAL_REQUEST_RECEIPT_V4,
    }
    return seal(body, identity_field="receipt_id")


@fundamental_v4_contract
def build_provider_physical_request_receipt_v4(
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    table: str,
    partition_id: str,
    sanitized_params_sha256: str,
    attempts: int,
    provider_request_id: str | None,
    reported_count: int,
    accepted_count: int,
    has_more: bool,
    status: str,
    blocker_codes: Sequence[str],
    raw_response_projection_sha256: str | None,
    captured_at: str,
) -> dict[str, Any]:
    """Seal one actual VIP physical request without embedding params or secrets."""

    validated_plan, endpoints = _plan_and_endpoints(
        plan=plan,
        endpoint_plans=endpoint_plans,
    )
    return _build_physical_request_receipt_validated(
        validated_plan=validated_plan,
        validated_plan_ref=content_ref(validated_plan, identity_field="plan_id"),
        endpoints=endpoints,
        table=table,
        partition_id=partition_id,
        sanitized_params_sha256=sanitized_params_sha256,
        attempts=attempts,
        provider_request_id=provider_request_id,
        reported_count=reported_count,
        accepted_count=accepted_count,
        has_more=has_more,
        status=status,
        blocker_codes=blocker_codes,
        raw_response_projection_sha256=raw_response_projection_sha256,
        captured_at=captured_at,
    )


@fundamental_v4_contract
def validate_provider_physical_request_receipt_v4(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validated_plan, endpoints = _plan_and_endpoints(
        plan=plan,
        endpoint_plans=endpoint_plans,
    )
    return _validate_physical_request_receipt_validated(
        document,
        validated_plan=validated_plan,
        validated_plan_ref=content_ref(validated_plan, identity_field="plan_id"),
        endpoints=endpoints,
    )


def _validate_physical_request_receipt_validated(
    document: Mapping[str, Any],
    *,
    validated_plan: Mapping[str, Any],
    validated_plan_ref: Mapping[str, str],
    endpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="receipt_id")
    require_exact_keys(value, _PHYSICAL_FIELDS, label="physical request receipt v4")
    if value.get("version") != PHYSICAL_REQUEST_RECEIPT_V4:
        raise FundamentalV4ContractError("physical request receipt version mismatch")
    expected = _build_physical_request_receipt_validated(
        validated_plan=validated_plan,
        validated_plan_ref=validated_plan_ref,
        endpoints=endpoints,
        table=value["table"],
        partition_id=value["partition_id"],
        sanitized_params_sha256=value["sanitized_params_sha256"],
        attempts=value["attempts"],
        provider_request_id=value["provider_request_id"],
        reported_count=value["reported_count"],
        accepted_count=value["accepted_count"],
        has_more=value["has_more"],
        status=value["status"],
        blocker_codes=value["blocker_codes"],
        raw_response_projection_sha256=value["raw_response_projection_sha256"],
        captured_at=value["timestamp"],
    )
    if value != expected:
        raise FundamentalV4ContractError("physical request receipt replay mismatch")
    return value


def _physical_receipts(
    values: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    table: str,
    expected_start: str,
    expected_end: str,
) -> list[dict[str, Any]]:
    rows = [
        validate_provider_physical_request_receipt_v4(
            value,
            plan=plan,
            endpoint_plans=endpoint_plans,
        )
        for value in _sequence(values, label="physical_receipts", maximum=2_000)
    ]
    rows = [row for row in rows if row["table"] == table]
    if table == "daily_basic":
        rows = [
            row
            for row in rows
            if expected_start <= row["partition_id"].split("=", 1)[1] <= expected_end
        ]
    else:
        rows = [
            row
            for row in rows
            if expected_start <= row["partition_id"].split("=", 1)[1] <= expected_end
        ]
    return sorted(rows, key=lambda row: row["receipt_id"].encode("ascii"))


def _observed_interval(
    *,
    expected_start: str,
    expected_end: str,
    observed_start: str | None,
    observed_end: str | None,
) -> tuple[str | None, str | None]:
    normalized_start = (
        None if observed_start is None else session_date(observed_start, label="observed_start")
    )
    normalized_end = (
        None if observed_end is None else session_date(observed_end, label="observed_end")
    )
    if (normalized_start is None) != (normalized_end is None):
        raise FundamentalV4ContractError("logical observed interval is partial")
    if normalized_start is not None and (
        normalized_start < expected_start
        or normalized_end is None
        or normalized_end > expected_end
        or normalized_start > normalized_end
    ):
        raise FundamentalV4ContractError("logical observed interval is invalid")
    return normalized_start, normalized_end


@fundamental_v4_contract
def build_logical_symbol_table_coverage_v4(
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    physical_receipts: Sequence[Mapping[str, Any]],
    company_code: str,
    table: str,
    expected_start: str,
    expected_end: str,
    observed_start: str | None,
    observed_end: str | None,
    row_count: int,
    missing_reason_codes: Sequence[str],
    duplicate_reason_codes: Sequence[str],
    restatement_reason_codes: Sequence[str],
    status: str,
    assessed_at: str,
) -> dict[str, Any]:
    """Project batch partitions into one explicit logical symbol/table closure."""

    validated_plan = validate_fundamental_request_plan_v4(
        plan,
        endpoint_plans=endpoint_plans,
    )
    if company_code not in validated_plan["symbols"] or table not in SOURCE_TABLES:
        raise FundamentalV4ContractError("logical coverage subject is outside plan")
    if status not in LOGICAL_STATUSES:
        raise FundamentalV4ContractError("logical coverage status is invalid")
    start = session_date(expected_start, label="expected_start")
    end = session_date(expected_end, label="expected_end")
    if start > end:
        raise FundamentalV4ContractError("logical expected interval is reversed")
    observed_start_value, observed_end_value = _observed_interval(
        expected_start=start,
        expected_end=end,
        observed_start=observed_start,
        observed_end=observed_end,
    )
    receipts = _physical_receipts(
        physical_receipts,
        plan=validated_plan,
        endpoint_plans=endpoint_plans,
        table=table,
        expected_start=start,
        expected_end=end,
    )
    expected_partitions = [
        row
        for row in validated_plan["partition_rows"]
        if row["table"] == table and start <= row["partition_id"].split("=", 1)[1] <= end
    ]
    expected_partition_ids = {row["partition_id"] for row in expected_partitions}
    if {row["partition_id"] for row in receipts} != expected_partition_ids:
        raise FundamentalV4ContractError("logical physical partition keyset is incomplete")
    receipt_ids = [row["receipt_id"] for row in receipts]
    missing = _codes(missing_reason_codes, label="missing_reason_codes")
    duplicates = _codes(duplicate_reason_codes, label="duplicate_reason_codes")
    restatements = _codes(
        restatement_reason_codes,
        label="restatement_reason_codes",
    )
    count = _nonnegative_int(row_count, label="row_count")
    physical_failed = any(row["status"] not in {"AVAILABLE", "EMPTY"} for row in receipts)
    complete = not missing and not duplicates and not restatements and not physical_failed
    if (status == "COMPLETE") != complete:
        raise FundamentalV4ContractError("logical coverage status does not match evidence")
    if count == 0 and observed_start_value is not None:
        raise FundamentalV4ContractError("zero-row coverage claims an observed interval")
    if count > 0 and observed_start_value is None:
        raise FundamentalV4ContractError("nonempty coverage lacks an observed interval")
    assessed = timestamp(assessed_at, label="assessed_at")
    if assessed < validated_plan["created_at"]:
        raise FundamentalV4ContractError("logical coverage predates its plan")
    body = {
        **common_fields(timestamp_value=assessed),
        "company_code": company_code,
        "duplicate_reason_codes": duplicates,
        "expected_end": end,
        "expected_partition_refs": receipt_ids,
        "expected_start": start,
        "missing_reason_codes": missing,
        "observed_end": observed_end_value,
        "observed_start": observed_start_value,
        "physical_receipt_refs": receipt_ids,
        "plan_ref": content_ref(validated_plan, identity_field="plan_id"),
        "restatement_reason_codes": restatements,
        "row_count": count,
        "status": status,
        "table": table,
        "version": LOGICAL_COVERAGE_V4,
    }
    return seal(body, identity_field="coverage_id")


@fundamental_v4_contract
def validate_logical_symbol_table_coverage_v4(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    physical_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="coverage_id")
    require_exact_keys(value, _LOGICAL_FIELDS, label="logical coverage v4")
    if value.get("version") != LOGICAL_COVERAGE_V4:
        raise FundamentalV4ContractError("logical coverage version mismatch")
    _receipt_ids(value["expected_partition_refs"], label="expected_partition_refs")
    _receipt_ids(value["physical_receipt_refs"], label="physical_receipt_refs")
    expected = build_logical_symbol_table_coverage_v4(
        plan=plan,
        endpoint_plans=endpoint_plans,
        physical_receipts=physical_receipts,
        company_code=value["company_code"],
        table=value["table"],
        expected_start=value["expected_start"],
        expected_end=value["expected_end"],
        observed_start=value["observed_start"],
        observed_end=value["observed_end"],
        row_count=value["row_count"],
        missing_reason_codes=value["missing_reason_codes"],
        duplicate_reason_codes=value["duplicate_reason_codes"],
        restatement_reason_codes=value["restatement_reason_codes"],
        status=value["status"],
        assessed_at=value["timestamp"],
    )
    if value != expected:
        raise FundamentalV4ContractError("logical coverage replay mismatch")
    return value


def _columns(value: Any) -> list[str]:
    rows = _sequence(value, label="column_order", maximum=512)
    normalized: list[str] = []
    for item in rows:
        if type(item) is not str or not item or not item.isascii():
            raise FundamentalV4ContractError("column_order contains an invalid name")
        normalized.append(item)
    if not normalized or len(normalized) != len(set(normalized)):
        raise FundamentalV4ContractError("column_order cardinality is invalid")
    return normalized


@fundamental_v4_contract
def build_raw_table_evidence_v4(
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    lane: str,
    table: str,
    file_ref: Mapping[str, Any],
    row_count: int,
    column_order: Sequence[str],
    canonical_multiset_sha256: str,
    duplicate_row_count: int,
    winner_implementation_sha256: str,
    evidenced_at: str,
) -> dict[str, Any]:
    validated_plan = validate_fundamental_request_plan_v4(
        plan,
        endpoint_plans=endpoint_plans,
    )
    if lane not in {"BASELINE", "VIP"} or table not in SOURCE_TABLES:
        raise FundamentalV4ContractError("raw table lane or table is invalid")
    evidenced = timestamp(evidenced_at, label="evidenced_at")
    normalized_ref = exact_ref(file_ref, label="file_ref")
    if (
        normalized_ref["cutoff"] > validated_plan["pit_cutoff"]
        or normalized_ref["available_at"] > evidenced
    ):
        raise FundamentalV4ContractError("raw table evidence contains future input")
    body = {
        **common_fields(timestamp_value=evidenced),
        "canonical_multiset_sha256": sha256(
            canonical_multiset_sha256,
            label="canonical_multiset_sha256",
        ),
        "column_order": _columns(column_order),
        "duplicate_row_count": _nonnegative_int(
            duplicate_row_count,
            label="duplicate_row_count",
        ),
        "file_ref": normalized_ref,
        "lane": lane,
        "plan_ref": content_ref(validated_plan, identity_field="plan_id"),
        "row_count": _nonnegative_int(row_count, label="row_count"),
        "table": table,
        "version": RAW_TABLE_EVIDENCE_V4,
        "winner_implementation_sha256": sha256(
            winner_implementation_sha256,
            label="winner_implementation_sha256",
        ),
    }
    return seal(body, identity_field="evidence_id")


@fundamental_v4_contract
def validate_raw_table_evidence_v4(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="evidence_id")
    require_exact_keys(value, _RAW_FIELDS, label="raw table evidence v4")
    if value.get("version") != RAW_TABLE_EVIDENCE_V4:
        raise FundamentalV4ContractError("raw table evidence version mismatch")
    expected = build_raw_table_evidence_v4(
        plan=plan,
        endpoint_plans=endpoint_plans,
        lane=value["lane"],
        table=value["table"],
        file_ref=value["file_ref"],
        row_count=value["row_count"],
        column_order=value["column_order"],
        canonical_multiset_sha256=value["canonical_multiset_sha256"],
        duplicate_row_count=value["duplicate_row_count"],
        winner_implementation_sha256=value["winner_implementation_sha256"],
        evidenced_at=value["timestamp"],
    )
    if value != expected:
        raise FundamentalV4ContractError("raw table evidence replay mismatch")
    return value


__all__ = [
    "build_logical_symbol_table_coverage_v4",
    "build_provider_physical_request_receipt_v4",
    "build_raw_table_evidence_v4",
    "validate_logical_symbol_table_coverage_v4",
    "validate_provider_physical_request_receipt_v4",
    "validate_raw_table_evidence_v4",
]
