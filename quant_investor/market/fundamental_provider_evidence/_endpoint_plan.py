"""Pure replay contract for persisted provider endpoint plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
import hashlib
import re
from typing import Any, Callable, Final, ParamSpec, TypeVar
from urllib.parse import urlsplit

from ._codec import (
    FundamentalProviderEvidenceError,
    canonical_bytes,
    code,
    common_fields,
    identifier,
    require_exact_keys,
    seal,
    timestamp,
    validate_seal,
)
from ._model import (
    ENDPOINT_EXECUTION_PLAN_SCHEMA,
    ENDPOINT_LANES,
    PERMISSION_CLASSES,
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
        except FundamentalProviderEvidenceError:
            raise

    return wrapped


def _transport_implementation_sha256(*, strict_decimal_decode: bool) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "decoder": "json.loads",
                "integer_decode": "int",
                "nonfinite": "REJECT",
                "reported_count": "ZERO_SENTINEL_TO_ITEMS_LENGTH_IN_STRICT_MODE",
                "strict_decimal_decode": strict_decimal_decode,
                "transport": "OfficialTushareHttpsClient.v2",
            }
        )
    ).hexdigest()


def _sequence(value: Any, *, label: str, maximum: int = 10_000) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FundamentalProviderEvidenceError(f"{label} must be a sequence")
    rows = list(value)
    if len(rows) > maximum:
        raise FundamentalProviderEvidenceError(f"{label} exceeds its maximum cardinality")
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
            raise FundamentalProviderEvidenceError(f"{label}[{index}] is invalid")
        result.append(item)
    if (not allow_empty and not result) or len(result) != len(set(result)):
        raise FundamentalProviderEvidenceError(f"{label} cardinality or uniqueness is invalid")
    return result


def _validate_params_value(value: Any, *, label: str, depth: int = 0) -> Any:
    if depth > 16:
        raise FundamentalProviderEvidenceError(f"{label} exceeds maximum depth")
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
                raise FundamentalProviderEvidenceError(f"{label} contains forbidden key")
            result[key] = _validate_params_value(
                item,
                label=f"{label}.{key}",
                depth=depth + 1,
            )
        canonical_bytes(result)
        return result
    raise FundamentalProviderEvidenceError(f"{label} contains an unsupported value")


def _params(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise FundamentalProviderEvidenceError(f"{label} must be an object")
    return _validate_params_value(dict(value), label=label)


def _document_url(value: Any) -> str:
    if type(value) is not str or len(value.encode("utf-8")) > 4000:
        raise FundamentalProviderEvidenceError("official_document_url is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise FundamentalProviderEvidenceError("official_document_url must be exact HTTPS")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FundamentalProviderEvidenceError(f"{label} must be a nonnegative integer")
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
        raise FundamentalProviderEvidenceError("api_name is invalid")
    if lane not in ENDPOINT_LANES or permission_class not in PERMISSION_CLASSES:
        raise FundamentalProviderEvidenceError("lane or permission_class is invalid")
    if type(strict_decimal_decode) is not bool or strict_decimal_decode is not True:
        raise FundamentalProviderEvidenceError("v2 endpoint plans require strict decimal decode")


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
        raise FundamentalProviderEvidenceError("retry schedule is invalid")
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
            raise FundamentalProviderEvidenceError("SEPARATE endpoint must plan zero requests")
        return
    if terminal_count != len(keyset) or planned_attempts != terminal_count * max_attempts:
        raise FundamentalProviderEvidenceError("POINTS request counts do not close")


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
        raise FundamentalProviderEvidenceError("official documentation observation is future-dated")
    _validate_plan_identity(
        api_name=api_name,
        lane=lane,
        permission_class=permission_class,
        strict_decimal_decode=strict_decimal_decode,
    )
    fields = _unique_text_rows(expected_fields, label="expected_fields")
    if any(_FIELD_RE.fullmatch(field) is None for field in fields):
        raise FundamentalProviderEvidenceError("expected_fields contains an invalid field")
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
        raise FundamentalProviderEvidenceError("documented_row_limit must be positive")
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
        "version": ENDPOINT_EXECUTION_PLAN_SCHEMA,
    }
    return seal(body, identity_field="plan_id")


@_tushare_contract
def validate_endpoint_execution_plan(document: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_seal(document, identity_field="plan_id")
    require_exact_keys(value, _plan_fields(), label="endpoint execution plan")
    if value.get("version") != ENDPOINT_EXECUTION_PLAN_SCHEMA:
        raise FundamentalProviderEvidenceError("endpoint execution plan version mismatch")
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
        raise FundamentalProviderEvidenceError("endpoint execution plan replay mismatch")
    return value


__all__ = [
    "build_endpoint_execution_plan",
    "validate_endpoint_execution_plan",
]
