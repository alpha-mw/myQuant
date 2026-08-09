"""Canonical contracts for governed Tushare endpoint execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import wraps
import hashlib
import re
from typing import Any, Callable, Final, ParamSpec, TypeVar
from urllib.parse import urlsplit

from ..._core import (
    IntelligenceV2ContractError,
    canonical_bytes,
    code,
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
from .models import (
    CAPABILITY_RECEIPT_VERSION,
    CAPABILITY_STATUSES,
    ENDPOINT_POLICY_VERSION,
    EXECUTION_PLAN_VERSION,
    EXECUTION_RECEIPT_VERSION,
    INCOMPLETE_BLOCKERS,
    LANES,
    PERMISSION_CLASSES,
    REQUEST_RECEIPT_VERSION,
    TushareContractError,
)

_FIELD_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$", re.ASCII)
_API_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$", re.ASCII)
_FORBIDDEN_PARAM_KEYS: Final = frozenset({"api_key", "authorization", "bearer", "secret", "token"})
_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "semantic_sha256",
    "timestamp",
}

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _tushare_contract(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Keep every public contract failure inside the Tushare error domain."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except TushareContractError:
            raise
        except IntelligenceV2ContractError as exc:
            raise TushareContractError(str(exc)) from exc

    return wrapped


def _transport_implementation_sha256(*, strict_decimal_decode: bool) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "decoder": "json.loads",
                "integer_decode": "int",
                "nonfinite": "REJECT",
                "strict_decimal_decode": strict_decimal_decode,
                "transport": "OfficialTushareHttpsClient.v1",
            }
        )
    ).hexdigest()


def _sequence(value: Any, *, label: str, maximum: int = 10_000) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TushareContractError(f"{label} must be a sequence")
    rows = list(value)
    if len(rows) > maximum:
        raise TushareContractError(f"{label} exceeds its maximum cardinality")
    return rows


def _unique_text_rows(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
    ascii_only: bool = True,
) -> list[str]:
    rows = _sequence(value, label=label)
    result: list[str] = []
    for index, item in enumerate(rows):
        if (
            type(item) is not str
            or not item
            or (ascii_only and not item.isascii())
            or len(item.encode("utf-8")) > 4000
        ):
            raise TushareContractError(f"{label}[{index}] is invalid")
        result.append(item)
    if (not allow_empty and not result) or len(result) != len(set(result)):
        raise TushareContractError(f"{label} cardinality or uniqueness is invalid")
    return result


def _validate_params_value(value: Any, *, label: str, depth: int = 0) -> Any:
    if depth > 16:
        raise TushareContractError(f"{label} exceeds maximum depth")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is list:
        return [
            _validate_params_value(item, label=f"{label}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if (
                type(key) is not str
                or not key
                or not key.isascii()
                or key.casefold() in _FORBIDDEN_PARAM_KEYS
            ):
                raise TushareContractError(f"{label} contains forbidden key")
            result[key] = _validate_params_value(
                item,
                label=f"{label}.{key}",
                depth=depth + 1,
            )
        canonical_bytes(result)
        return result
    raise TushareContractError(f"{label} contains an unsupported value")


def _params(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TushareContractError(f"{label} must be an object")
    return _validate_params_value(dict(value), label=label)


def _document_url(value: Any) -> str:
    if type(value) is not str or len(value.encode("utf-8")) > 4000:
        raise TushareContractError("official_document_url is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise TushareContractError("official_document_url must be exact HTTPS")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TushareContractError(f"{label} must be a nonnegative integer")
    return value


def _plan_fields() -> set[str]:
    return _COMMON_FIELDS | {
        "api_name",
        "completeness_proof",
        "created_at",
        "document_observed_at",
        "documented_min_points",
        "documented_row_limit",
        "empty_partition_rule",
        "expected_fields",
        "fixed_params",
        "lane",
        "limit_hit_action",
        "max_attempts",
        "official_document_id",
        "official_document_url",
        "ordered_expected_partition_keyset",
        "partition_dimensions",
        "permission_class",
        "plan_id",
        "planned_max_network_attempts",
        "planned_terminal_request_count",
        "retry_schedule",
        "strict_decimal_decode",
        "version",
    }


def _validate_plan_identity(
    *,
    api_name: str,
    lane: str,
    permission_class: str,
    strict_decimal_decode: bool,
) -> None:
    if type(api_name) is not str or _API_RE.fullmatch(api_name) is None:
        raise TushareContractError("api_name is invalid")
    if lane not in LANES or permission_class not in PERMISSION_CLASSES:
        raise TushareContractError("lane or permission_class is invalid")
    if type(strict_decimal_decode) is not bool or strict_decimal_decode is not True:
        raise TushareContractError("v2 endpoint plans require strict decimal decode")


def _validate_retry_schedule(value: Any, *, max_attempts: int) -> list[int]:
    retries = _sequence(value, label="retry_schedule", maximum=64)
    if (
        type(max_attempts) is not int
        or max_attempts < 1
        or max_attempts > 64
        or len(retries) != max_attempts
        or any(type(item) is not int or item < 0 or item > 3600 for item in retries)
        or retries[0] != 0
    ):
        raise TushareContractError("retry schedule is invalid")
    return retries


def _validate_request_topology(
    *,
    permission_class: str,
    dimensions: list[str],
    keyset: list[str],
    terminal_count: int,
    planned_attempts: int,
    max_attempts: int,
) -> None:
    if permission_class == "SEPARATE":
        if keyset or dimensions or terminal_count != 0 or planned_attempts != 0:
            raise TushareContractError("SEPARATE endpoint must plan zero requests")
        return
    if terminal_count != len(keyset) or planned_attempts != terminal_count * max_attempts:
        raise TushareContractError("POINTS request counts do not close")


@_tushare_contract
def build_endpoint_execution_plan(
    *,
    api_name: str,
    lane: str,
    permission_class: str,
    official_document_url: str,
    official_document_id: str,
    document_observed_at: str,
    documented_min_points: int,
    strict_decimal_decode: bool,
    expected_fields: Sequence[str],
    fixed_params: Mapping[str, Any],
    partition_dimensions: Sequence[str],
    ordered_expected_partition_keyset: Sequence[str],
    documented_row_limit: int,
    max_attempts: int,
    retry_schedule: Sequence[int],
    empty_partition_rule: str,
    completeness_proof: str,
    limit_hit_action: str,
    planned_terminal_request_count: int,
    planned_max_network_attempts: int,
    created_at: str,
) -> dict[str, Any]:
    """Build one exact, bounded endpoint execution plan."""

    created = timestamp(created_at, label="created_at")
    observed = timestamp(document_observed_at, label="document_observed_at")
    if observed > created:
        raise TushareContractError("official documentation observation is future-dated")
    _validate_plan_identity(
        api_name=api_name,
        lane=lane,
        permission_class=permission_class,
        strict_decimal_decode=strict_decimal_decode,
    )
    fields = _unique_text_rows(expected_fields, label="expected_fields")
    if any(_FIELD_RE.fullmatch(field) is None for field in fields):
        raise TushareContractError("expected_fields contains an invalid field")
    dimensions = _unique_text_rows(
        partition_dimensions,
        label="partition_dimensions",
        allow_empty=permission_class == "SEPARATE",
    )
    keyset = _unique_text_rows(
        ordered_expected_partition_keyset,
        label="ordered_expected_partition_keyset",
        allow_empty=permission_class == "SEPARATE",
    )
    retries = _validate_retry_schedule(retry_schedule, max_attempts=max_attempts)
    terminal_count = _nonnegative_int(
        planned_terminal_request_count,
        label="planned_terminal_request_count",
    )
    planned_attempts = _nonnegative_int(
        planned_max_network_attempts,
        label="planned_max_network_attempts",
    )
    _validate_request_topology(
        permission_class=permission_class,
        dimensions=dimensions,
        keyset=keyset,
        terminal_count=terminal_count,
        planned_attempts=planned_attempts,
        max_attempts=max_attempts,
    )
    row_limit = _nonnegative_int(documented_row_limit, label="documented_row_limit")
    if row_limit < 1:
        raise TushareContractError("documented_row_limit must be positive")
    body = {
        **common_fields(timestamp_value=created),
        "api_name": api_name,
        "completeness_proof": code(completeness_proof, label="completeness_proof"),
        "created_at": created,
        "document_observed_at": observed,
        "documented_min_points": _nonnegative_int(
            documented_min_points,
            label="documented_min_points",
        ),
        "documented_row_limit": row_limit,
        "empty_partition_rule": code(
            empty_partition_rule,
            label="empty_partition_rule",
        ),
        "expected_fields": fields,
        "fixed_params": _params(fixed_params, label="fixed_params"),
        "lane": lane,
        "limit_hit_action": code(limit_hit_action, label="limit_hit_action"),
        "max_attempts": max_attempts,
        "official_document_id": identifier(
            official_document_id,
            label="official_document_id",
        ),
        "official_document_url": _document_url(official_document_url),
        "ordered_expected_partition_keyset": keyset,
        "partition_dimensions": dimensions,
        "permission_class": permission_class,
        "planned_max_network_attempts": planned_attempts,
        "planned_terminal_request_count": terminal_count,
        "retry_schedule": retries,
        "strict_decimal_decode": True,
        "version": EXECUTION_PLAN_VERSION,
    }
    return seal(body, identity_field="plan_id")


@_tushare_contract
def validate_endpoint_execution_plan(document: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_seal(document, identity_field="plan_id")
    require_exact_keys(value, _plan_fields(), label="endpoint execution plan")
    if value.get("version") != EXECUTION_PLAN_VERSION:
        raise TushareContractError("endpoint execution plan version mismatch")
    expected = build_endpoint_execution_plan(
        api_name=value["api_name"],
        lane=value["lane"],
        permission_class=value["permission_class"],
        official_document_url=value["official_document_url"],
        official_document_id=value["official_document_id"],
        document_observed_at=value["document_observed_at"],
        documented_min_points=value["documented_min_points"],
        strict_decimal_decode=value["strict_decimal_decode"],
        expected_fields=value["expected_fields"],
        fixed_params=value["fixed_params"],
        partition_dimensions=value["partition_dimensions"],
        ordered_expected_partition_keyset=value["ordered_expected_partition_keyset"],
        documented_row_limit=value["documented_row_limit"],
        max_attempts=value["max_attempts"],
        retry_schedule=value["retry_schedule"],
        empty_partition_rule=value["empty_partition_rule"],
        completeness_proof=value["completeness_proof"],
        limit_hit_action=value["limit_hit_action"],
        planned_terminal_request_count=value["planned_terminal_request_count"],
        planned_max_network_attempts=value["planned_max_network_attempts"],
        created_at=value["created_at"],
    )
    if value != expected:
        raise TushareContractError("endpoint execution plan replay mismatch")
    return value


@_tushare_contract
def build_tushare_endpoint_policy(
    *,
    created_at: str,
    endpoint_plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    created = timestamp(created_at, label="created_at")
    plans = [
        validate_endpoint_execution_plan(plan)
        for plan in _sequence(
            endpoint_plans,
            label="endpoint_plans",
            maximum=128,
        )
    ]
    if not plans:
        raise TushareContractError("endpoint policy requires plans")
    if any(plan["created_at"] > created for plan in plans):
        raise TushareContractError("endpoint policy contains a future plan")
    names = [plan["api_name"] for plan in plans]
    if len(names) != len(set(names)):
        raise TushareContractError("endpoint policy contains duplicate api_name")
    plans = sorted(plans, key=lambda row: row["api_name"].encode("ascii"))
    body = {
        **common_fields(timestamp_value=created),
        "created_at": created,
        "endpoint_plans": plans,
        "version": ENDPOINT_POLICY_VERSION,
    }
    return seal(body, identity_field="policy_id")


@_tushare_contract
def validate_tushare_endpoint_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_seal(document, identity_field="policy_id")
    require_exact_keys(
        value,
        _COMMON_FIELDS | {"created_at", "endpoint_plans", "policy_id", "version"},
        label="Tushare endpoint policy",
    )
    if value.get("version") != ENDPOINT_POLICY_VERSION:
        raise TushareContractError("Tushare endpoint policy version mismatch")
    expected = build_tushare_endpoint_policy(
        created_at=value["created_at"],
        endpoint_plans=value["endpoint_plans"],
    )
    if value != expected:
        raise TushareContractError("Tushare endpoint policy replay mismatch")
    return value


@_tushare_contract
def build_tushare_request_receipt(
    *,
    plan: Mapping[str, Any],
    partition_key: str,
    partition_ordinal: int,
    sanitized_params: Mapping[str, Any],
    requested_at: str,
) -> dict[str, Any]:
    validated_plan = validate_endpoint_execution_plan(plan)
    requested = timestamp(requested_at, label="requested_at")
    if requested < validated_plan["created_at"]:
        raise TushareContractError("request predates its endpoint plan")
    if validated_plan["permission_class"] != "POINTS":
        raise TushareContractError("SEPARATE endpoint cannot build a request receipt")
    keyset = validated_plan["ordered_expected_partition_keyset"]
    if (
        type(partition_ordinal) is not int
        or partition_ordinal < 0
        or partition_ordinal >= len(keyset)
        or keyset[partition_ordinal] != partition_key
    ):
        raise TushareContractError("partition identity is invalid")
    params = _params(sanitized_params, label="sanitized_params")
    for key, item in validated_plan["fixed_params"].items():
        if params.get(key) != item:
            raise TushareContractError("fixed_params were not preserved")
    body = {
        **common_fields(timestamp_value=requested),
        "api_name": validated_plan["api_name"],
        "partition_key": partition_key,
        "partition_ordinal": partition_ordinal,
        "plan_ref": content_ref(validated_plan, identity_field="plan_id"),
        "sanitized_params_sha256": hashlib.sha256(canonical_bytes(params)).hexdigest(),
        "strict_decimal_decode": True,
        "transport_implementation_sha256": _transport_implementation_sha256(
            strict_decimal_decode=True
        ),
        "version": REQUEST_RECEIPT_VERSION,
    }
    return seal(body, identity_field="request_receipt_id")


@_tushare_contract
def validate_tushare_request_receipt(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    sanitized_params: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="request_receipt_id")
    require_exact_keys(
        value,
        _COMMON_FIELDS
        | {
            "api_name",
            "partition_key",
            "partition_ordinal",
            "plan_ref",
            "request_receipt_id",
            "sanitized_params_sha256",
            "strict_decimal_decode",
            "transport_implementation_sha256",
            "version",
        },
        label="Tushare request receipt",
    )
    if value.get("version") != REQUEST_RECEIPT_VERSION:
        raise TushareContractError("Tushare request receipt version mismatch")
    expected = build_tushare_request_receipt(
        plan=plan,
        partition_key=value["partition_key"],
        partition_ordinal=value["partition_ordinal"],
        sanitized_params=sanitized_params,
        requested_at=value["timestamp"],
    )
    if value != expected:
        raise TushareContractError("Tushare request receipt replay mismatch")
    return value


def _blocker_codes(value: Any) -> list[str]:
    rows = [
        code(item, label="blocker_code")
        for item in _sequence(
            value,
            label="blocker_codes",
            maximum=64,
        )
    ]
    if len(rows) != len(set(rows)):
        raise TushareContractError("blocker_codes contain duplicates")
    return sorted(rows, key=lambda item: item.encode("ascii"))


def _validate_capability_authority(
    *,
    permission_class: str,
    status: str,
    calls: int,
    reported: int,
    accepted: int,
    request_ref: Mapping[str, Any] | None,
    projection_sha: str | None,
) -> None:
    if permission_class == "SEPARATE":
        if (
            status != "NOT_PROBED"
            or calls != 0
            or reported != 0
            or accepted != 0
            or request_ref is not None
            or projection_sha is not None
        ):
            raise TushareContractError("SEPARATE capability must remain NOT_PROBED")
        return
    if status == "NOT_PROBED" or calls != 1 or request_ref is None:
        raise TushareContractError("POINTS capability must have one request")


def _validate_capability_status(
    *,
    status: str,
    blockers: list[str],
    reported: int,
    accepted: int,
    row_limit: int,
    projection_sha: str | None,
) -> None:
    if status == "AVAILABLE" and (
        blockers
        or reported != accepted
        or accepted < 1
        or accepted >= row_limit
        or projection_sha is None
    ):
        raise TushareContractError("AVAILABLE capability is not closed")
    if status == "EMPTY" and (reported or accepted or blockers or projection_sha is None):
        raise TushareContractError("EMPTY capability is not closed")
    if status == "INCOMPLETE" and (
        not blockers or any(item not in INCOMPLETE_BLOCKERS for item in blockers)
    ):
        raise TushareContractError("INCOMPLETE capability blockers are invalid")
    if status == "INCOMPLETE" and projection_sha is None:
        raise TushareContractError("INCOMPLETE capability lacks response projection")
    if status in {"PROVIDER_ERROR", "SCHEMA_MISMATCH", "TRANSPORT_ERROR"} and (
        blockers or reported or accepted or projection_sha is not None
    ):
        raise TushareContractError("failed capability must not claim response data")


@_tushare_contract
def build_tushare_capability_receipt(
    *,
    plan: Mapping[str, Any],
    status: str,
    transport_calls: int,
    reported_count: int,
    accepted_count: int,
    blocker_codes: Sequence[str],
    request_ref: Mapping[str, Any] | None,
    response_projection_sha256: str | None,
    probed_at: str,
) -> dict[str, Any]:
    validated_plan = validate_endpoint_execution_plan(plan)
    probed = timestamp(probed_at, label="probed_at")
    if probed < validated_plan["created_at"]:
        raise TushareContractError("capability probe predates its endpoint plan")
    if status not in CAPABILITY_STATUSES:
        raise TushareContractError("capability status is invalid")
    calls = _nonnegative_int(transport_calls, label="transport_calls")
    reported = _nonnegative_int(reported_count, label="reported_count")
    accepted = _nonnegative_int(accepted_count, label="accepted_count")
    blockers = _blocker_codes(blocker_codes)
    if calls > 1:
        raise TushareContractError("capability probe permits at most one transport call")
    normalized_request_ref = (
        None if request_ref is None else validate_content_ref(request_ref, label="request_ref")
    )
    if (
        normalized_request_ref is not None
        and normalized_request_ref["artifact_version"] != REQUEST_RECEIPT_VERSION
    ):
        raise TushareContractError("request_ref version mismatch")
    normalized_projection_sha = (
        None
        if response_projection_sha256 is None
        else sha256(response_projection_sha256, label="response_projection_sha256")
    )
    _validate_capability_authority(
        permission_class=validated_plan["permission_class"],
        status=status,
        calls=calls,
        reported=reported,
        accepted=accepted,
        request_ref=normalized_request_ref,
        projection_sha=normalized_projection_sha,
    )
    _validate_capability_status(
        status=status,
        blockers=blockers,
        reported=reported,
        accepted=accepted,
        row_limit=validated_plan["documented_row_limit"],
        projection_sha=normalized_projection_sha,
    )
    body = {
        **common_fields(timestamp_value=probed),
        "accepted_count": accepted,
        "api_name": validated_plan["api_name"],
        "blocker_codes": blockers,
        "plan_ref": content_ref(validated_plan, identity_field="plan_id"),
        "reported_count": reported,
        "request_ref": normalized_request_ref,
        "response_projection_sha256": normalized_projection_sha,
        "status": status,
        "transport_calls": calls,
        "version": CAPABILITY_RECEIPT_VERSION,
    }
    return seal(body, identity_field="capability_receipt_id")


@_tushare_contract
def validate_tushare_capability_receipt(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="capability_receipt_id")
    require_exact_keys(
        value,
        _COMMON_FIELDS
        | {
            "accepted_count",
            "api_name",
            "blocker_codes",
            "capability_receipt_id",
            "plan_ref",
            "reported_count",
            "request_ref",
            "response_projection_sha256",
            "status",
            "transport_calls",
            "version",
        },
        label="Tushare capability receipt",
    )
    if value.get("version") != CAPABILITY_RECEIPT_VERSION:
        raise TushareContractError("Tushare capability receipt version mismatch")
    expected = build_tushare_capability_receipt(
        plan=plan,
        status=value["status"],
        transport_calls=value["transport_calls"],
        reported_count=value["reported_count"],
        accepted_count=value["accepted_count"],
        blocker_codes=value["blocker_codes"],
        request_ref=value["request_ref"],
        response_projection_sha256=value["response_projection_sha256"],
        probed_at=value["timestamp"],
    )
    if value != expected:
        raise TushareContractError("Tushare capability receipt replay mismatch")
    return value


@_tushare_contract
def build_tushare_execution_receipt(
    *,
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    request_refs: Sequence[Mapping[str, Any]],
    capability_receipt: Mapping[str, Any],
    network_attempts: int,
    completed_partition_keys: Sequence[str],
    missing_partition_keys: Sequence[str],
    failed_partition_keys: Sequence[str],
    executed_at: str,
) -> dict[str, Any]:
    validated_policy = validate_tushare_endpoint_policy(policy)
    validated_plan = validate_endpoint_execution_plan(plan)
    capability = validate_tushare_capability_receipt(
        capability_receipt,
        plan=validated_plan,
    )
    executed = timestamp(executed_at, label="executed_at")
    if executed < max(validated_policy["created_at"], capability["timestamp"]):
        raise TushareContractError("execution receipt predates its validation closure")
    refs = [
        validate_content_ref(item, label="request_ref")
        for item in _sequence(request_refs, label="request_refs", maximum=64)
    ]
    if any(item["artifact_version"] != REQUEST_RECEIPT_VERSION for item in refs):
        raise TushareContractError("request_refs contain an unsupported version")
    if len(refs) != len({tuple(sorted(item.items())) for item in refs}):
        raise TushareContractError("request_refs contain duplicates")
    completed = _unique_text_rows(
        completed_partition_keys,
        label="completed_partition_keys",
        allow_empty=True,
    )
    missing = _unique_text_rows(
        missing_partition_keys,
        label="missing_partition_keys",
        allow_empty=True,
    )
    failed = _unique_text_rows(
        failed_partition_keys,
        label="failed_partition_keys",
        allow_empty=True,
    )
    if set(completed) & set(missing) or set(completed) & set(failed) or set(missing) & set(failed):
        raise TushareContractError("partition terminal sets overlap")
    expected_keyset = set(validated_plan["ordered_expected_partition_keyset"])
    if set(completed) | set(missing) | set(failed) != expected_keyset:
        raise TushareContractError("partition terminal sets do not close")
    attempts = _nonnegative_int(network_attempts, label="network_attempts")
    if (
        attempts != capability["transport_calls"]
        or attempts > validated_plan["planned_max_network_attempts"]
    ):
        raise TushareContractError("network attempt accounting mismatch")
    terminal_state = (
        "COMPLETE"
        if not missing
        and not failed
        and capability["status"] in {"AVAILABLE", "EMPTY", "NOT_PROBED"}
        else "INCOMPLETE"
    )
    body = {
        **common_fields(timestamp_value=executed),
        "capability_ref": content_ref(
            capability,
            identity_field="capability_receipt_id",
        ),
        "completed_partition_keys": completed,
        "failed_partition_keys": failed,
        "missing_partition_keys": missing,
        "network_attempts": attempts,
        "plan_ref": content_ref(validated_plan, identity_field="plan_id"),
        "policy_ref": content_ref(validated_policy, identity_field="policy_id"),
        "request_refs": refs,
        "terminal_state": terminal_state,
        "version": EXECUTION_RECEIPT_VERSION,
    }
    return seal(body, identity_field="execution_receipt_id")


@_tushare_contract
def validate_tushare_execution_receipt(
    document: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    capability_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="execution_receipt_id")
    require_exact_keys(
        value,
        _COMMON_FIELDS
        | {
            "capability_ref",
            "completed_partition_keys",
            "execution_receipt_id",
            "failed_partition_keys",
            "missing_partition_keys",
            "network_attempts",
            "plan_ref",
            "policy_ref",
            "request_refs",
            "terminal_state",
            "version",
        },
        label="Tushare execution receipt",
    )
    if value.get("version") != EXECUTION_RECEIPT_VERSION:
        raise TushareContractError("Tushare execution receipt version mismatch")
    expected = build_tushare_execution_receipt(
        policy=policy,
        plan=plan,
        request_refs=value["request_refs"],
        capability_receipt=capability_receipt,
        network_attempts=value["network_attempts"],
        completed_partition_keys=value["completed_partition_keys"],
        missing_partition_keys=value["missing_partition_keys"],
        failed_partition_keys=value["failed_partition_keys"],
        executed_at=value["timestamp"],
    )
    if value != expected:
        raise TushareContractError("Tushare execution receipt replay mismatch")
    return value


def response_projection_sha256(rows: Sequence[Sequence[Any]]) -> str:
    projections: list[list[dict[str, Any]]] = []
    for row in rows:
        projected_row: list[dict[str, Any]] = []
        for value in row:
            if value is None:
                projected_row.append({"kind": "NULL", "value": None})
            elif type(value) is bool:
                projected_row.append({"kind": "BOOL", "value": value})
            elif type(value) is int:
                projected_row.append({"kind": "INT", "value": str(value)})
            elif type(value) is Decimal and value.is_finite():
                projected_row.append({"kind": "DECIMAL", "value": format(value, "f")})
            elif type(value) is str:
                projected_row.append({"kind": "TEXT", "value": value})
            else:
                raise TushareContractError("response projection contains invalid scalar")
        projections.append(projected_row)
    return hashlib.sha256(canonical_bytes(projections)).hexdigest()


__all__ = [
    "build_endpoint_execution_plan",
    "build_tushare_capability_receipt",
    "build_tushare_endpoint_policy",
    "build_tushare_execution_receipt",
    "build_tushare_request_receipt",
    "response_projection_sha256",
    "validate_endpoint_execution_plan",
    "validate_tushare_capability_receipt",
    "validate_tushare_endpoint_policy",
    "validate_tushare_execution_receipt",
    "validate_tushare_request_receipt",
]
