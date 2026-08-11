"""Exact three-partition capture contract for the SW2021 taxonomy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
import hashlib
from typing import Any, Callable, Final, ParamSpec, TypeVar

from ....v17_v4_runtime.tushare_https import TushareHttpsError
from ..._core import (
    canonical_bytes,
    common_fields,
    content_ref,
    IntelligenceV2ContractError,
    require_exact_keys,
    seal,
    timestamp,
    validate_seal,
)
from .contracts import (
    build_endpoint_execution_plan,
    response_projection_sha256,
    validate_endpoint_execution_plan,
)
from .models import TushareContractError, TushareRequestClient

INDUSTRY_TAXONOMY_PLAN_VERSION: Final = (
    "myquant.v17.intelligence-v2.tushare-industry-taxonomy-plan.v1"
)
INDUSTRY_TAXONOMY_CAPTURE_VERSION: Final = (
    "myquant.v17.intelligence-v2.tushare-industry-taxonomy-capture.v1"
)
INDUSTRY_MEMBERSHIP_PLAN_VERSION: Final = (
    "myquant.v17.intelligence-v2.tushare-industry-membership-plan.v1"
)
INDEX_CLASSIFY_FIELDS: Final = (
    "index_code",
    "industry_name",
    "parent_code",
    "level",
    "industry_code",
    "is_pub",
    "src",
)
OFFICIAL_PARTITIONS: Final = (
    ("L1", 31),
    ("L2", 134),
    ("L3", 346),
)
_PLAN_FIELDS: Final = set(common_fields(timestamp_value="2000-01-01T00:00:00Z")) | {
    "api_name",
    "created_at",
    "document_observed_at",
    "expected_fields",
    "official_document_id",
    "official_document_url",
    "partition_rows",
    "plan_id",
    "planned_max_network_attempts",
    "planned_terminal_request_count",
    "semantic_sha256",
    "src",
    "strict_decimal_decode",
    "version",
}
_CAPTURE_FIELDS: Final = set(common_fields(timestamp_value="2000-01-01T00:00:00Z")) | {
    "capture_id",
    "partition_rows",
    "plan_ref",
    "semantic_sha256",
    "version",
}
_PARTITION_FIELDS: Final = {"expected_count", "level", "params"}
_CAPTURE_PARTITION_FIELDS: Final = {
    "level",
    "reported_count",
    "response_projection_sha256",
    "rows",
}
_MEMBERSHIP_PLAN_FIELDS: Final = set(common_fields(timestamp_value="2000-01-01T00:00:00Z")) | {
    "created_at",
    "endpoint_plan",
    "l3_keyset",
    "membership_plan_id",
    "semantic_sha256",
    "taxonomy_capture_ref",
    "version",
}
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _fail(message: str) -> None:
    raise TushareContractError(message)


def _tushare_contract(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except TushareContractError:
            raise
        except IntelligenceV2ContractError as exc:
            raise TushareContractError(str(exc)) from exc

    return wrapped


def _partition_plan_rows() -> list[dict[str, Any]]:
    return [
        {
            "expected_count": expected_count,
            "level": level,
            "params": {"level": level, "src": "SW2021"},
        }
        for level, expected_count in OFFICIAL_PARTITIONS
    ]


@_tushare_contract
def build_industry_taxonomy_execution_plan(
    *,
    document_observed_at: str,
    created_at: str,
) -> dict[str, Any]:
    """Seal the official SW2021 L1/L2/L3 cardinality closure."""

    observed = timestamp(document_observed_at, label="document_observed_at")
    created = timestamp(created_at, label="created_at")
    if observed > created:
        _fail("taxonomy documentation observation is future-dated")
    return seal(
        {
            "version": INDUSTRY_TAXONOMY_PLAN_VERSION,
            **common_fields(timestamp_value=created),
            "api_name": "index_classify",
            "created_at": created,
            "document_observed_at": observed,
            "expected_fields": list(INDEX_CLASSIFY_FIELDS),
            "official_document_id": "tushare.doc-181.index_classify",
            "official_document_url": "https://tushare.pro/document/2?doc_id=181",
            "partition_rows": _partition_plan_rows(),
            "planned_max_network_attempts": 3,
            "planned_terminal_request_count": 3,
            "src": "SW2021",
            "strict_decimal_decode": True,
        },
        identity_field="plan_id",
    )


@_tushare_contract
def validate_industry_taxonomy_execution_plan(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="plan_id")
    require_exact_keys(value, _PLAN_FIELDS, label="industry taxonomy execution plan")
    if value.get("version") != INDUSTRY_TAXONOMY_PLAN_VERSION:
        _fail("industry taxonomy execution plan version mismatch")
    for row in value.get("partition_rows", []):
        require_exact_keys(row, _PARTITION_FIELDS, label="taxonomy plan partition")
    expected = build_industry_taxonomy_execution_plan(
        document_observed_at=value["document_observed_at"],
        created_at=value["created_at"],
    )
    if value != expected:
        _fail("industry taxonomy execution plan replay mismatch")
    return value


def _row_mapping(raw: Any, *, index: int) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != set(INDEX_CLASSIFY_FIELDS):
        _fail(f"taxonomy row {index} shape is invalid")
    return dict(raw)


def _normalize_capture_partition(
    raw_partition: Any,
    *,
    expected: Mapping[str, Any],
    all_index_codes: set[str],
    all_industry_codes: set[str],
) -> tuple[dict[str, Any], set[str]]:
    partition = require_exact_keys(
        raw_partition,
        _CAPTURE_PARTITION_FIELDS,
        label="taxonomy capture partition",
    )
    level = expected["level"]
    rows_raw = partition["rows"]
    if isinstance(rows_raw, (str, bytes)) or not isinstance(rows_raw, Sequence):
        _fail("taxonomy partition rows must be a sequence")
    rows = [_row_mapping(row, index=index) for index, row in enumerate(rows_raw)]
    if (
        partition["level"] != level
        or type(partition["reported_count"]) is not int
        or partition["reported_count"] != len(rows)
        or len(rows) != expected["expected_count"]
    ):
        _fail("taxonomy partition cardinality mismatch")
    projection = response_projection_sha256(
        tuple(tuple(row[field] for field in INDEX_CLASSIFY_FIELDS) for row in rows)
    )
    if partition["response_projection_sha256"] != projection:
        _fail("taxonomy response projection mismatch")
    industry_codes: set[str] = set()
    index_codes: set[str] = set()
    for row in rows:
        index_code = row["index_code"]
        industry_code = row["industry_code"]
        if (
            type(index_code) is not str
            or not index_code
            or not index_code.isascii()
            or type(industry_code) is not str
            or not industry_code
            or not industry_code.isascii()
            or row["level"] != level
            or row["src"] != "SW2021"
            or index_code in index_codes
            or index_code in all_index_codes
            or industry_code in industry_codes
            or industry_code in all_industry_codes
        ):
            _fail("taxonomy row identity or scope mismatch")
        index_codes.add(index_code)
        all_index_codes.add(index_code)
        industry_codes.add(industry_code)
        all_industry_codes.add(industry_code)
    return (
        {
            "level": level,
            "reported_count": len(rows),
            "response_projection_sha256": projection,
            "rows": rows,
        },
        industry_codes,
    )


def _validate_capture_hierarchy(
    partitions: Sequence[Mapping[str, Any]],
    *,
    codes_by_level: Mapping[str, set[str]],
) -> None:
    for partition in partitions:
        level = partition["level"]
        for row in partition["rows"]:
            parent = row["parent_code"]
            if level == "L1":
                if parent not in {None, "", "0"}:
                    _fail("L1 taxonomy parent mismatch")
                continue
            parent_level = "L1" if level == "L2" else "L2"
            if type(parent) is not str or parent not in codes_by_level[parent_level]:
                _fail("taxonomy hierarchy is incomplete")


def _validate_partition_rows(
    raw_partitions: Any,
    *,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(raw_partitions, (str, bytes)) or not isinstance(raw_partitions, Sequence):
        _fail("taxonomy capture partitions must be a sequence")
    if len(raw_partitions) != len(plan["partition_rows"]):
        _fail("taxonomy capture partition keyset is incomplete")
    result: list[dict[str, Any]] = []
    all_index_codes: set[str] = set()
    all_industry_codes: set[str] = set()
    industry_codes_by_level: dict[str, set[str]] = {}
    for expected, raw_partition in zip(plan["partition_rows"], raw_partitions):
        partition, industry_codes = _normalize_capture_partition(
            raw_partition,
            expected=expected,
            all_index_codes=all_index_codes,
            all_industry_codes=all_industry_codes,
        )
        industry_codes_by_level[partition["level"]] = industry_codes
        result.append(partition)
    _validate_capture_hierarchy(result, codes_by_level=industry_codes_by_level)
    return result


@_tushare_contract
def build_industry_taxonomy_capture(
    *,
    plan: Mapping[str, Any],
    partition_rows: Sequence[Mapping[str, Any]],
    captured_at: str,
) -> dict[str, Any]:
    validated_plan = validate_industry_taxonomy_execution_plan(plan)
    captured = timestamp(captured_at, label="captured_at")
    if captured < validated_plan["created_at"]:
        _fail("taxonomy capture predates its plan")
    rows = _validate_partition_rows(partition_rows, plan=validated_plan)
    return seal(
        {
            "version": INDUSTRY_TAXONOMY_CAPTURE_VERSION,
            **common_fields(timestamp_value=captured),
            "partition_rows": rows,
            "plan_ref": content_ref(validated_plan, identity_field="plan_id"),
        },
        identity_field="capture_id",
    )


@_tushare_contract
def validate_industry_taxonomy_capture(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="capture_id")
    require_exact_keys(value, _CAPTURE_FIELDS, label="industry taxonomy capture")
    if value.get("version") != INDUSTRY_TAXONOMY_CAPTURE_VERSION:
        _fail("industry taxonomy capture version mismatch")
    expected = build_industry_taxonomy_capture(
        plan=plan,
        partition_rows=value["partition_rows"],
        captured_at=value["timestamp"],
    )
    if value != expected:
        _fail("industry taxonomy capture replay mismatch")
    return value


@_tushare_contract
def capture_tushare_industry_taxonomy(
    *,
    plan: Mapping[str, Any],
    captured_at: str,
    client: TushareRequestClient,
) -> dict[str, Any]:
    """Make exactly three official requests and return one replayable capture."""

    validated_plan = validate_industry_taxonomy_execution_plan(plan)
    partitions: list[dict[str, Any]] = []
    for row in validated_plan["partition_rows"]:
        try:
            response = client.request(
                api_name="index_classify",
                params=row["params"],
                expected_fields=INDEX_CLASSIFY_FIELDS,
            )
        except TushareHttpsError as exc:
            _fail(f"industry taxonomy transport failed: {exc.code}")
        except Exception as exc:
            raise TushareContractError("industry taxonomy transport failed") from exc
        if (
            response.api_name != "index_classify"
            or tuple(response.fields) != INDEX_CLASSIFY_FIELDS
            or response.has_more
            or response.reported_count != len(response.rows)
        ):
            _fail("industry taxonomy response closure mismatch")
        mapped = [dict(zip(INDEX_CLASSIFY_FIELDS, values)) for values in response.rows]
        partitions.append(
            {
                "level": row["level"],
                "reported_count": response.reported_count,
                "response_projection_sha256": response_projection_sha256(response.rows),
                "rows": mapped,
            }
        )
    if len(partitions) != validated_plan["planned_terminal_request_count"]:
        _fail("industry taxonomy request topology mismatch")
    return build_industry_taxonomy_capture(
        plan=validated_plan,
        partition_rows=partitions,
        captured_at=captured_at,
    )


def taxonomy_capture_byte_sha256(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _membership_partition_keyset(l3_keyset: Sequence[str]) -> list[str]:
    return [f"l3_code={l3_code}|is_new={flag}" for l3_code in l3_keyset for flag in ("Y", "N")]


@_tushare_contract
def build_industry_membership_execution_plan(
    *,
    taxonomy_plan: Mapping[str, Any],
    taxonomy_capture: Mapping[str, Any],
    document_observed_at: str,
    created_at: str,
) -> dict[str, Any]:
    """Bind all official L3 identities to the exact Y/N request keyset."""

    validated_taxonomy_plan = validate_industry_taxonomy_execution_plan(taxonomy_plan)
    validated_capture = validate_industry_taxonomy_capture(
        taxonomy_capture,
        plan=validated_taxonomy_plan,
    )
    created = timestamp(created_at, label="created_at")
    if created < validated_capture["timestamp"]:
        _fail("industry membership plan predates taxonomy capture")
    l3_partition = next(row for row in validated_capture["partition_rows"] if row["level"] == "L3")
    l3_keyset = sorted(
        (row["index_code"] for row in l3_partition["rows"]),
        key=lambda item: item.encode("ascii"),
    )
    partition_keyset = _membership_partition_keyset(l3_keyset)
    endpoint_plan = build_endpoint_execution_plan(
        api_name="index_member_all",
        lane="INDUSTRY",
        permission_class="POINTS",
        official_document_url="https://tushare.pro/document/2?doc_id=335",
        official_document_id="tushare.doc-335.index_member_all",
        document_observed_at=document_observed_at,
        documented_min_points=2000,
        strict_decimal_decode=True,
        expected_fields=(
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
        ),
        fixed_params={},
        partition_dimensions=("l3_code", "is_new"),
        ordered_expected_partition_keyset=partition_keyset,
        documented_row_limit=2000,
        max_attempts=1,
        retry_schedule=(0,),
        empty_partition_rule="EXACT_PARTITION_EMPTY_ALLOWED",
        completeness_proof="L3_IS_NEW_EXACT_KEYSET",
        limit_hit_action="BLOCK",
        planned_terminal_request_count=len(partition_keyset),
        planned_max_network_attempts=len(partition_keyset),
        created_at=created,
    )
    return seal(
        {
            "version": INDUSTRY_MEMBERSHIP_PLAN_VERSION,
            **common_fields(timestamp_value=created),
            "created_at": created,
            "endpoint_plan": endpoint_plan,
            "l3_keyset": l3_keyset,
            "taxonomy_capture_ref": content_ref(
                validated_capture,
                identity_field="capture_id",
            ),
        },
        identity_field="membership_plan_id",
    )


@_tushare_contract
def validate_industry_membership_execution_plan(
    document: Mapping[str, Any],
    *,
    taxonomy_plan: Mapping[str, Any],
    taxonomy_capture: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="membership_plan_id")
    require_exact_keys(value, _MEMBERSHIP_PLAN_FIELDS, label="industry membership plan")
    if value.get("version") != INDUSTRY_MEMBERSHIP_PLAN_VERSION:
        _fail("industry membership plan version mismatch")
    validate_endpoint_execution_plan(value["endpoint_plan"])
    expected = build_industry_membership_execution_plan(
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        document_observed_at=value["endpoint_plan"]["document_observed_at"],
        created_at=value["created_at"],
    )
    if value != expected:
        _fail("industry membership execution plan replay mismatch")
    return value


__all__ = [
    "INDEX_CLASSIFY_FIELDS",
    "INDUSTRY_TAXONOMY_CAPTURE_VERSION",
    "INDUSTRY_TAXONOMY_PLAN_VERSION",
    "INDUSTRY_MEMBERSHIP_PLAN_VERSION",
    "OFFICIAL_PARTITIONS",
    "build_industry_taxonomy_capture",
    "build_industry_taxonomy_execution_plan",
    "build_industry_membership_execution_plan",
    "capture_tushare_industry_taxonomy",
    "taxonomy_capture_byte_sha256",
    "validate_industry_taxonomy_capture",
    "validate_industry_taxonomy_execution_plan",
    "validate_industry_membership_execution_plan",
]
