"""Exact, resumable partition contracts for SW2021 industry membership."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
import hashlib
import re
from typing import Any, Callable, Final, ParamSpec, TypeVar

from ....v17_v4_runtime.tushare_https import TushareHttpsError
from ..._core import (
    IntelligenceV2ContractError,
    canonical_bytes,
    common_fields,
    content_ref,
    require_exact_keys,
    seal,
    timestamp,
    validate_seal,
)
from .contracts import response_projection_sha256
from .industry_taxonomy import validate_industry_membership_execution_plan
from .models import TushareContractError, TushareRequestClient

INDUSTRY_MEMBERSHIP_PARTITION_VERSION: Final = (
    "myquant.v17.intelligence-v2.tushare-industry-membership-partition.v1"
)
INDUSTRY_MEMBERSHIP_CAPTURE_VERSION: Final = (
    "myquant.v17.intelligence-v2.tushare-industry-membership-capture.v1"
)
INDEX_MEMBER_ALL_FIELDS: Final = (
    "l1_code",
    "l1_name",
    "l2_code",
    "l2_name",
    "l3_code",
    "l3_name",
    "ts_code",
    "name",
    "in_date",
    "out_date",
    "is_new",
)
_PARTITION_KEY_RE: Final = re.compile(
    r"^l3_code=([A-Za-z0-9.]+)\|is_new=([YN])$",
    re.ASCII,
)
_PARTITION_FIELDS: Final = set(common_fields(timestamp_value="2000-01-01T00:00:00Z")) | {
    "membership_plan_ref",
    "partition_capture_id",
    "partition_key",
    "partition_ordinal",
    "provider_request_id",
    "reported_count",
    "response_projection_sha256",
    "rows",
    "semantic_sha256",
    "status",
    "version",
}
_CAPTURE_FIELDS: Final = set(common_fields(timestamp_value="2000-01-01T00:00:00Z")) | {
    "capture_id",
    "empty_partition_count",
    "membership_plan_ref",
    "partition_rows",
    "semantic_sha256",
    "status",
    "total_row_count",
    "version",
}
_CAPTURE_ROW_FIELDS: Final = {
    "byte_sha256",
    "partition_capture_ref",
    "partition_key",
    "partition_ordinal",
    "relative_path",
}
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _fail(message: str) -> None:
    raise TushareContractError(message)


def _contract(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except TushareContractError:
            raise
        except IntelligenceV2ContractError as exc:
            raise TushareContractError(str(exc)) from exc

    return wrapped


def membership_partition_params(partition_key: str) -> dict[str, str]:
    match = _PARTITION_KEY_RE.fullmatch(partition_key) if type(partition_key) is str else None
    if match is None:
        _fail("industry membership partition key is invalid")
    return {"l3_code": match.group(1), "is_new": match.group(2)}


def _row(raw: Any, *, params: Mapping[str, str], index: int) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != set(INDEX_MEMBER_ALL_FIELDS):
        _fail(f"industry membership row {index} shape is invalid")
    row = dict(raw)
    text_fields = set(INDEX_MEMBER_ALL_FIELDS) - {"out_date"}
    if any(type(row[field]) is not str or not row[field] for field in text_fields):
        _fail("industry membership row text is invalid")
    if row["out_date"] is not None and type(row["out_date"]) is not str:
        _fail("industry membership out_date is invalid")
    if row["l3_code"] != params["l3_code"] or row["is_new"] != params["is_new"]:
        _fail("industry membership row scope mismatch")
    return row


def _validated_rows(
    value: Any,
    *,
    params: Mapping[str, str],
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("industry membership rows must be a sequence")
    rows = [_row(raw, params=params, index=index) for index, raw in enumerate(value)]
    if len(rows) >= 2000:
        _fail("industry membership row limit was reached")
    identities = [tuple(row[field] for field in INDEX_MEMBER_ALL_FIELDS) for row in rows]
    if len(identities) != len(set(identities)):
        _fail("industry membership duplicate rows")
    return rows


@_contract
def build_industry_membership_partition_capture(
    *,
    membership_plan: Mapping[str, Any],
    taxonomy_plan: Mapping[str, Any],
    taxonomy_capture: Mapping[str, Any],
    partition_key: str,
    partition_ordinal: int,
    provider_request_id: str,
    reported_count: int,
    rows: Sequence[Mapping[str, Any]],
    captured_at: str,
) -> dict[str, Any]:
    plan = validate_industry_membership_execution_plan(
        membership_plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
    )
    keyset = plan["endpoint_plan"]["ordered_expected_partition_keyset"]
    if (
        type(partition_ordinal) is not int
        or not 0 <= partition_ordinal < len(keyset)
        or keyset[partition_ordinal] != partition_key
    ):
        _fail("industry membership partition ordinal mismatch")
    params = membership_partition_params(partition_key)
    normalized = _validated_rows(rows, params=params)
    if type(reported_count) is not int or reported_count != len(normalized):
        _fail("industry membership reported count mismatch")
    if type(provider_request_id) is not str or not provider_request_id:
        _fail("industry membership provider request id is invalid")
    captured = timestamp(captured_at, label="captured_at")
    if captured < plan["created_at"]:
        _fail("industry membership partition predates its plan")
    projection = response_projection_sha256(
        tuple(tuple(row[field] for field in INDEX_MEMBER_ALL_FIELDS) for row in normalized)
    )
    return seal(
        {
            "version": INDUSTRY_MEMBERSHIP_PARTITION_VERSION,
            **common_fields(timestamp_value=captured),
            "membership_plan_ref": content_ref(
                plan,
                identity_field="membership_plan_id",
            ),
            "partition_key": partition_key,
            "partition_ordinal": partition_ordinal,
            "provider_request_id": provider_request_id,
            "reported_count": reported_count,
            "response_projection_sha256": projection,
            "rows": normalized,
            "status": "EMPTY" if not normalized else "AVAILABLE",
        },
        identity_field="partition_capture_id",
    )


@_contract
def validate_industry_membership_partition_capture(
    document: Mapping[str, Any],
    *,
    membership_plan: Mapping[str, Any],
    taxonomy_plan: Mapping[str, Any],
    taxonomy_capture: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="partition_capture_id")
    require_exact_keys(value, _PARTITION_FIELDS, label="industry membership partition")
    if value.get("version") != INDUSTRY_MEMBERSHIP_PARTITION_VERSION:
        _fail("industry membership partition version mismatch")
    expected = build_industry_membership_partition_capture(
        membership_plan=membership_plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        partition_key=value["partition_key"],
        partition_ordinal=value["partition_ordinal"],
        provider_request_id=value["provider_request_id"],
        reported_count=value["reported_count"],
        rows=value["rows"],
        captured_at=value["timestamp"],
    )
    if value != expected:
        _fail("industry membership partition replay mismatch")
    return value


@_contract
def capture_industry_membership_partition(
    *,
    membership_plan: Mapping[str, Any],
    taxonomy_plan: Mapping[str, Any],
    taxonomy_capture: Mapping[str, Any],
    partition_ordinal: int,
    captured_at: str,
    client: TushareRequestClient,
) -> dict[str, Any]:
    plan = validate_industry_membership_execution_plan(
        membership_plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
    )
    keyset = plan["endpoint_plan"]["ordered_expected_partition_keyset"]
    if type(partition_ordinal) is not int or not 0 <= partition_ordinal < len(keyset):
        _fail("industry membership partition ordinal mismatch")
    partition_key = keyset[partition_ordinal]
    params = membership_partition_params(partition_key)
    try:
        response = client.request(
            api_name="index_member_all",
            params=params,
            expected_fields=INDEX_MEMBER_ALL_FIELDS,
        )
    except TushareHttpsError as exc:
        _fail(f"industry membership transport failed: {exc.code}")
    except Exception as exc:
        raise TushareContractError("industry membership transport failed") from exc
    if (
        response.api_name != "index_member_all"
        or tuple(response.fields) != INDEX_MEMBER_ALL_FIELDS
        or response.has_more
    ):
        _fail("industry membership response closure mismatch")
    mapped = [dict(zip(INDEX_MEMBER_ALL_FIELDS, values)) for values in response.rows]
    return build_industry_membership_partition_capture(
        membership_plan=plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        partition_key=partition_key,
        partition_ordinal=partition_ordinal,
        provider_request_id=response.request_id,
        reported_count=response.reported_count,
        rows=mapped,
        captured_at=captured_at,
    )


@_contract
def build_industry_membership_capture(
    *,
    membership_plan: Mapping[str, Any],
    taxonomy_plan: Mapping[str, Any],
    taxonomy_capture: Mapping[str, Any],
    partition_documents: Sequence[Mapping[str, Any]],
    completed_at: str,
) -> dict[str, Any]:
    """Seal COMPLETE only when every planned partition fully replays."""

    plan = validate_industry_membership_execution_plan(
        membership_plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
    )
    keyset = plan["endpoint_plan"]["ordered_expected_partition_keyset"]
    if len(partition_documents) != len(keyset):
        _fail("industry membership capture keyset is incomplete")
    rows: list[dict[str, Any]] = []
    total = 0
    empty = 0
    latest = plan["created_at"]
    for ordinal, (partition_key, raw_document) in enumerate(zip(keyset, partition_documents)):
        document = validate_industry_membership_partition_capture(
            raw_document,
            membership_plan=plan,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=taxonomy_capture,
        )
        if document["partition_ordinal"] != ordinal or document["partition_key"] != partition_key:
            _fail("industry membership capture keyset mismatch")
        raw = canonical_bytes(document)
        rows.append(
            {
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "partition_capture_ref": content_ref(
                    document,
                    identity_field="partition_capture_id",
                ),
                "partition_key": partition_key,
                "partition_ordinal": ordinal,
                "relative_path": f"partitions/{ordinal:04d}.json",
            }
        )
        total += document["reported_count"]
        empty += document["status"] == "EMPTY"
        latest = max(latest, document["timestamp"])
    completed = timestamp(completed_at, label="completed_at")
    if completed < latest:
        _fail("industry membership capture completion predates a partition")
    return seal(
        {
            "version": INDUSTRY_MEMBERSHIP_CAPTURE_VERSION,
            **common_fields(timestamp_value=completed),
            "empty_partition_count": empty,
            "membership_plan_ref": content_ref(
                plan,
                identity_field="membership_plan_id",
            ),
            "partition_rows": rows,
            "status": "COMPLETE",
            "total_row_count": total,
        },
        identity_field="capture_id",
    )


@_contract
def validate_industry_membership_capture(
    document: Mapping[str, Any],
    *,
    membership_plan: Mapping[str, Any],
    taxonomy_plan: Mapping[str, Any],
    taxonomy_capture: Mapping[str, Any],
    partition_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="capture_id")
    require_exact_keys(value, _CAPTURE_FIELDS, label="industry membership capture")
    if value.get("version") != INDUSTRY_MEMBERSHIP_CAPTURE_VERSION:
        _fail("industry membership capture version mismatch")
    for row in value.get("partition_rows", []):
        require_exact_keys(row, _CAPTURE_ROW_FIELDS, label="industry membership capture row")
    expected = build_industry_membership_capture(
        membership_plan=membership_plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        partition_documents=partition_documents,
        completed_at=value["timestamp"],
    )
    if value != expected:
        _fail("industry membership capture replay mismatch")
    return value


__all__ = [
    "INDEX_MEMBER_ALL_FIELDS",
    "INDUSTRY_MEMBERSHIP_CAPTURE_VERSION",
    "INDUSTRY_MEMBERSHIP_PARTITION_VERSION",
    "build_industry_membership_capture",
    "build_industry_membership_partition_capture",
    "capture_industry_membership_partition",
    "membership_partition_params",
    "validate_industry_membership_capture",
    "validate_industry_membership_partition_capture",
]
