"""Stable physical receipt contracts for Fundamental partition acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .._core import (
    common_fields,
    content_ref,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_content_ref,
    validate_seal,
)
from ..contracts import validate_endpoint_execution_plan
from .models import (
    PHYSICAL_RECEIPT_KIND,
    PHYSICAL_STATUSES,
    SOURCE_TABLES,
    FundamentalAcquisitionError,
    fundamental_contract,
)
from .schedule import validate_fundamental_request_plan

_PHYSICAL_FIELDS = {
    "accepted_count",
    "attempts",
    "authority",
    "blocker_codes",
    "contract_sha256",
    "endpoint",
    "has_more",
    "kind",
    "partition_id",
    "partition_type",
    "plan_ref",
    "production",
    "provider_request_id",
    "raw_response_projection_sha256",
    "receipt_id",
    "reported_count",
    "research_only",
    "sanitized_params_sha256",
    "semantic_sha256",
    "status",
    "strict_decimal_decode",
    "table",
    "timestamp",
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
        raise FundamentalAcquisitionError(f"{label} must be a sequence")
    rows = list(value)
    if len(rows) > maximum:
        raise FundamentalAcquisitionError(f"{label} exceeds maximum cardinality")
    return rows


def _codes(value: Any, *, label: str) -> list[str]:
    rows = _sequence(value, label=label, maximum=64)
    normalized: list[str] = []
    for item in rows:
        if (
            type(item) is not str
            or not item
            or not item.isascii()
            or not item.replace("_", "").isalnum()
            or item.upper() != item
        ):
            raise FundamentalAcquisitionError(f"{label} contains an invalid code")
        normalized.append(item)
    expected = sorted(normalized, key=lambda item: item.encode("ascii"))
    if normalized != expected or len(normalized) != len(set(normalized)):
        raise FundamentalAcquisitionError(f"{label} must be ASCII-sorted unique")
    return normalized


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FundamentalAcquisitionError(f"{label} must be a nonnegative integer")
    return value


def _plan_and_endpoints(
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    validated_plan = validate_fundamental_request_plan(
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
        raise FundamentalAcquisitionError("physical partition is not in the sealed plan")
    return dict(rows[0])


def _validate_status(
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
        raise FundamentalAcquisitionError("physical request attempts exceed policy")
    if status == "AVAILABLE" and (
        reported != accepted
        or accepted < 1
        or accepted >= row_limit
        or has_more
        or blockers
        or projection_sha is None
    ):
        raise FundamentalAcquisitionError("AVAILABLE physical request is not closed")
    if status == "EMPTY" and (
        reported or accepted or has_more or blockers or projection_sha is None
    ):
        raise FundamentalAcquisitionError("EMPTY physical request is not closed")
    if status == "INCOMPLETE" and (
        not blockers
        or any(code not in _INCOMPLETE_BLOCKERS for code in blockers)
        or projection_sha is None
    ):
        raise FundamentalAcquisitionError("INCOMPLETE physical blockers are invalid")
    if status in {"PROVIDER_ERROR", "SCHEMA_MISMATCH", "TRANSPORT_ERROR"} and (
        reported or accepted or has_more or blockers or projection_sha is not None
    ):
        raise FundamentalAcquisitionError("failed physical request claims response data")


def _build_validated(
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
        raise FundamentalAcquisitionError("physical table or status is invalid")
    partition = _partition(validated_plan, table=table, partition_id=partition_id)
    captured = timestamp(captured_at, label="captured_at")
    if captured < validated_plan["created_at"]:
        raise FundamentalAcquisitionError("physical receipt predates its plan")
    normalized_attempts = _nonnegative_int(attempts, label="attempts")
    reported = _nonnegative_int(reported_count, label="reported_count")
    accepted = _nonnegative_int(accepted_count, label="accepted_count")
    if type(has_more) is not bool:
        raise FundamentalAcquisitionError("has_more must be boolean")
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
    _validate_status(
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
        raise FundamentalAcquisitionError(
            "EMPTY physical request lacks sealed baseline identity proof"
        )
    body = {
        **common_fields(timestamp_value=captured),
        "accepted_count": accepted,
        "attempts": normalized_attempts,
        "blocker_codes": blockers,
        "endpoint": partition["endpoint"],
        "has_more": has_more,
        "kind": PHYSICAL_RECEIPT_KIND,
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
    }
    return seal(body, identity_field="receipt_id")


@fundamental_contract
def build_fundamental_partition_receipt(
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
    """Seal one request receipt without parameters, credentials, or authority."""

    validated_plan, endpoints = _plan_and_endpoints(
        plan=plan,
        endpoint_plans=endpoint_plans,
    )
    return _build_validated(
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


def _validate_validated(
    document: Mapping[str, Any],
    *,
    validated_plan: Mapping[str, Any],
    validated_plan_ref: Mapping[str, str],
    endpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="receipt_id")
    require_exact_keys(value, _PHYSICAL_FIELDS, label="Fundamental partition receipt")
    if value.get("kind") != PHYSICAL_RECEIPT_KIND:
        raise FundamentalAcquisitionError("Fundamental partition receipt kind mismatch")
    if validate_content_ref(value["plan_ref"], label="plan_ref") != dict(validated_plan_ref):
        raise FundamentalAcquisitionError("Fundamental partition plan ref mismatch")
    expected = _build_validated(
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
        raise FundamentalAcquisitionError("Fundamental partition receipt replay mismatch")
    return value


@fundamental_contract
def validate_fundamental_partition_receipt(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validated_plan, endpoints = _plan_and_endpoints(
        plan=plan,
        endpoint_plans=endpoint_plans,
    )
    return _validate_validated(
        document,
        validated_plan=validated_plan,
        validated_plan_ref=content_ref(validated_plan, identity_field="plan_id"),
        endpoints=endpoints,
    )


__all__ = [
    "build_fundamental_partition_receipt",
    "validate_fundamental_partition_receipt",
]
