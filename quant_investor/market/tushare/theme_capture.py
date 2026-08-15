"""Sealed, resumable DC/TDX Theme snapshot acquisition contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
import re
from typing import Any, Callable, cast, Final, ParamSpec, TypeVar

from quant_investor.market.tushare_transport import TushareHttpsError

from ._core import (
    TushareDataContractError,
    canonical_bytes,
    common_fields,
    content_ref,
    require_exact_keys,
    seal,
    timestamp,
    validate_seal,
)
from .contracts import build_endpoint_execution_plan, validate_endpoint_execution_plan
from .models import TushareContractError, TushareRequestClient

THEME_PROVIDER_PLAN_KIND: Final = "market.tushare.theme_provider_plan"
THEME_PARTITION_KIND: Final = "market.tushare.theme_partition"
THEME_PROVIDER_CAPTURE_KIND: Final = "market.tushare.theme_provider_capture"

_PROVIDERS: Final = {
    "TUSHARE_DC": {
        "registry_api": "dc_index",
        "registry_document_id": "tushare.doc-362.dc_index",
        "registry_document_url": "https://tushare.pro/document/2?doc_id=362",
        "registry_fields": ("ts_code", "trade_date", "name", "idx_type", "level"),
        "registry_limit": 5000,
        "member_api": "dc_member",
        "member_document_id": "tushare.doc-363.dc_member",
        "member_document_url": "https://tushare.pro/document/2?doc_id=363",
        "member_fields": ("trade_date", "ts_code", "con_code", "name"),
        "member_limit": 5000,
    },
    "TUSHARE_TDX": {
        "registry_api": "tdx_index",
        "registry_document_id": "tushare.doc-376.tdx_index",
        "registry_document_url": "https://tushare.pro/document/2?doc_id=376",
        "registry_fields": ("ts_code", "trade_date", "name", "idx_type", "idx_count"),
        "registry_limit": 1000,
        "member_api": "tdx_member",
        "member_document_id": "tushare.doc-377.tdx_member",
        "member_document_url": "https://tushare.pro/document/2?doc_id=377",
        "member_fields": ("ts_code", "trade_date", "con_code", "con_name"),
        "member_limit": 3000,
    },
}
_PARTITION_KEY_RE: Final = re.compile(
    r"^(registry|member):([A-Z0-9.]+):([0-9]{8})$",
    re.ASCII,
)
_PLAN_FIELDS: Final = set(common_fields(timestamp_value="2000-01-01T00:00:00Z")) | {
    "company_keyset",
    "contract_sha256",
    "created_at",
    "membership_plan",
    "kind",
    "plan_id",
    "provider",
    "registry_plan",
    "semantic_sha256",
    "trade_date",
}
_PARTITION_FIELDS: Final = set(common_fields(timestamp_value="2000-01-01T00:00:00Z")) | {
    "api_name",
    "contract_sha256",
    "blocker_codes",
    "partition_capture_id",
    "kind",
    "partition_key",
    "partition_ordinal",
    "plan_ref",
    "provider_request_id",
    "reported_count",
    "rows",
    "semantic_sha256",
    "status",
}
_CAPTURE_FIELDS: Final = set(common_fields(timestamp_value="2000-01-01T00:00:00Z")) | {
    "capture_id",
    "contract_sha256",
    "incomplete_partition_count",
    "kind",
    "partition_rows",
    "plan_ref",
    "provider",
    "semantic_sha256",
    "status",
    "total_row_count",
}
_CAPTURE_ROW_FIELDS: Final = {
    "partition_capture_ref",
    "partition_key",
    "partition_ordinal",
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
        except TushareDataContractError as exc:
            raise TushareContractError(str(exc)) from exc

    return wrapped


def _company_keyset(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        _fail("theme company keyset must be a nonempty sequence")
    rows = list(value)
    if (
        any(type(row) is not str or not row or not row.isascii() for row in rows)
        or len(rows) != len(set(rows))
        or rows != sorted(rows, key=lambda row: row.encode("ascii"))
    ):
        _fail("theme company keyset must be unique and ASCII sorted")
    return rows


def _provider(value: Any) -> tuple[str, Mapping[str, Any]]:
    if type(value) is not str or value not in _PROVIDERS:
        _fail("theme source provider is invalid")
    return value, _PROVIDERS[value]


def _endpoint_plan(
    *,
    api_name: str,
    document_id: str,
    document_url: str,
    documented_row_limit: int,
    expected_fields: Sequence[str],
    trade_date: str,
    company_keyset: Sequence[str],
    created_at: str,
    document_observed_at: str,
) -> dict[str, Any]:
    registry = api_name.endswith("_index")
    keyset = (
        [f"registry:ALL:{trade_date}"]
        if registry
        else [f"member:{company}:{trade_date}" for company in company_keyset]
    )
    return build_endpoint_execution_plan(
        api_name=api_name,
        lane="THEME",
        permission_class="POINTS",
        official_document_url=document_url,
        official_document_id=document_id,
        document_observed_at=document_observed_at,
        documented_min_points=6000,
        strict_decimal_decode=True,
        expected_fields=expected_fields,
        fixed_params={"trade_date": trade_date, **({"idx_type": "概念板块"} if registry else {})},
        partition_dimensions=["trade_date"] if registry else ["con_code", "trade_date"],
        ordered_expected_partition_keyset=keyset,
        documented_row_limit=documented_row_limit,
        max_attempts=1,
        retry_schedule=[0],
        empty_partition_rule=(
            "REGISTRY_MUST_BE_NONEMPTY" if registry else "EMPTY_IS_COMPLETE_NO_MEMBERSHIP"
        ),
        completeness_proof="EXACT_KEYSET_SCHEMA_SCOPE_AND_ROW_LIMIT",
        limit_hit_action="BLOCK",
        planned_terminal_request_count=len(keyset),
        planned_max_network_attempts=len(keyset),
        created_at=created_at,
    )


@_contract
def build_theme_provider_execution_plan(
    *,
    provider: str,
    trade_date: str,
    company_keyset: Sequence[str],
    document_observed_at: str,
    created_at: str,
) -> dict[str, Any]:
    provider_id, config = _provider(provider)
    companies = _company_keyset(company_keyset)
    if type(trade_date) is not str or not re.fullmatch(r"[0-9]{8}", trade_date):
        _fail("theme trade date is invalid")
    observed = timestamp(document_observed_at, label="document_observed_at")
    created = timestamp(created_at, label="created_at")
    if observed > created:
        _fail("theme documentation observation is future-dated")
    return seal(
        {
            "kind": THEME_PROVIDER_PLAN_KIND,
            **common_fields(timestamp_value=created),
            "provider": provider_id,
            "trade_date": trade_date,
            "company_keyset": companies,
            "created_at": created,
            "registry_plan": _endpoint_plan(
                api_name=config["registry_api"],
                document_id=config["registry_document_id"],
                document_url=config["registry_document_url"],
                documented_row_limit=config["registry_limit"],
                expected_fields=config["registry_fields"],
                trade_date=trade_date,
                company_keyset=companies,
                created_at=created,
                document_observed_at=observed,
            ),
            "membership_plan": _endpoint_plan(
                api_name=config["member_api"],
                document_id=config["member_document_id"],
                document_url=config["member_document_url"],
                documented_row_limit=config["member_limit"],
                expected_fields=config["member_fields"],
                trade_date=trade_date,
                company_keyset=companies,
                created_at=created,
                document_observed_at=observed,
            ),
        },
        identity_field="plan_id",
    )


@_contract
def validate_theme_provider_execution_plan(document: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_seal(document, identity_field="plan_id")
    require_exact_keys(value, _PLAN_FIELDS, label="theme provider plan")
    if value.get("kind") != THEME_PROVIDER_PLAN_KIND:
        _fail("theme provider plan kind mismatch")
    validate_endpoint_execution_plan(value["registry_plan"])
    validate_endpoint_execution_plan(value["membership_plan"])
    expected = build_theme_provider_execution_plan(
        provider=value["provider"],
        trade_date=value["trade_date"],
        company_keyset=value["company_keyset"],
        document_observed_at=value["registry_plan"]["document_observed_at"],
        created_at=value["created_at"],
    )
    if value != expected:
        _fail("theme provider plan replay mismatch")
    return value


def _partition_identity(plan: Mapping[str, Any], ordinal: int) -> tuple[str, str, str, int]:
    if type(ordinal) is not int or ordinal < 0:
        _fail("theme partition ordinal is invalid")
    registry_plan = plan["registry_plan"]
    membership_plan = plan["membership_plan"]
    if ordinal == 0:
        return (
            registry_plan["api_name"],
            registry_plan["ordered_expected_partition_keyset"][0],
            "ALL",
            registry_plan["documented_row_limit"],
        )
    member_ordinal = ordinal - 1
    keyset = membership_plan["ordered_expected_partition_keyset"]
    if member_ordinal >= len(keyset):
        _fail("theme partition ordinal is outside the plan")
    match = _PARTITION_KEY_RE.fullmatch(keyset[member_ordinal])
    if match is None or match.group(1) != "member":
        _fail("theme membership partition key is invalid")
    return (
        membership_plan["api_name"],
        keyset[member_ordinal],
        match.group(2),
        membership_plan["documented_row_limit"],
    )


def _normalize_rows(
    rows: Any,
    *,
    fields: Sequence[str],
    api_name: str,
    company: str,
    trade_date: str,
    limit: int,
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        _fail("theme partition rows must be a sequence")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if type(raw) is not dict or set(raw) != set(fields):
            _fail(f"theme partition row {index} shape is invalid")
        row = dict(raw)
        if row.get("trade_date") != trade_date:
            _fail("theme partition trade date mismatch")
        if api_name.endswith("_index"):
            if row.get("idx_type") != "概念板块":
                _fail("theme registry type mismatch")
        elif row.get("con_code") != company:
            _fail("theme membership company mismatch")
        result.append(row)
    if len(result) >= limit:
        _fail("theme partition row limit was reached")
    identities = [canonical_bytes(row) for row in result]
    if len(identities) != len(set(identities)):
        _fail("theme partition contains duplicate rows")
    return result


def _build_theme_partition_capture(
    *,
    validated_plan: Mapping[str, Any],
    validated_plan_ref: Mapping[str, str],
    partition_ordinal: int,
    provider_request_id: str | None,
    reported_count: int,
    rows: Sequence[Mapping[str, Any]],
    blocker_codes: Sequence[str],
    captured_at: str,
) -> dict[str, Any]:
    api_name, partition_key, company, limit = _partition_identity(
        validated_plan,
        partition_ordinal,
    )
    config = _PROVIDERS[validated_plan["provider"]]
    fields = cast(
        tuple[str, ...],
        config["registry_fields"] if api_name.endswith("_index") else config["member_fields"],
    )
    captured = timestamp(captured_at, label="captured_at")
    if captured < validated_plan["created_at"]:
        _fail("theme partition predates its execution plan")
    blockers = list(blocker_codes)
    if blockers:
        if (
            blockers != sorted(set(blockers), key=lambda item: item.encode("ascii"))
            or any(type(item) is not str or not item or not item.isascii() for item in blockers)
            or provider_request_id is not None
            or reported_count != 0
            or rows
        ):
            _fail("incomplete theme partition closure is invalid")
        normalized: list[dict[str, Any]] = []
        status = "INCOMPLETE"
    else:
        if type(provider_request_id) is not str or not provider_request_id:
            _fail("theme provider request id is invalid")
        normalized = _normalize_rows(
            rows,
            fields=fields,
            api_name=api_name,
            company=company,
            trade_date=validated_plan["trade_date"],
            limit=limit,
        )
        if type(reported_count) is not int or reported_count != len(normalized):
            _fail("theme reported count mismatch")
        if api_name.endswith("_index") and not normalized:
            _fail("theme registry must be nonempty")
        status = "EMPTY" if not normalized else "AVAILABLE"
    return seal(
        {
            "kind": THEME_PARTITION_KIND,
            **common_fields(timestamp_value=captured),
            "plan_ref": dict(validated_plan_ref),
            "partition_key": partition_key,
            "partition_ordinal": partition_ordinal,
            "api_name": api_name,
            "provider_request_id": provider_request_id,
            "reported_count": reported_count,
            "rows": normalized,
            "blocker_codes": blockers,
            "status": status,
        },
        identity_field="partition_capture_id",
    )


@_contract
def build_theme_partition_capture(
    *,
    plan: Mapping[str, Any],
    partition_ordinal: int,
    provider_request_id: str | None,
    reported_count: int,
    rows: Sequence[Mapping[str, Any]],
    blocker_codes: Sequence[str],
    captured_at: str,
) -> dict[str, Any]:
    validated_plan = validate_theme_provider_execution_plan(plan)
    return _build_theme_partition_capture(
        validated_plan=validated_plan,
        validated_plan_ref=content_ref(validated_plan, identity_field="plan_id"),
        partition_ordinal=partition_ordinal,
        provider_request_id=provider_request_id,
        reported_count=reported_count,
        rows=rows,
        blocker_codes=blocker_codes,
        captured_at=captured_at,
    )


def _validate_theme_partition_capture(
    document: Mapping[str, Any],
    *,
    validated_plan: Mapping[str, Any],
    validated_plan_ref: Mapping[str, str],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="partition_capture_id")
    require_exact_keys(value, _PARTITION_FIELDS, label="theme partition")
    if value.get("kind") != THEME_PARTITION_KIND:
        _fail("theme partition kind mismatch")
    expected = _build_theme_partition_capture(
        validated_plan=validated_plan,
        validated_plan_ref=validated_plan_ref,
        partition_ordinal=value["partition_ordinal"],
        provider_request_id=value["provider_request_id"],
        reported_count=value["reported_count"],
        rows=value["rows"],
        blocker_codes=value["blocker_codes"],
        captured_at=value["timestamp"],
    )
    if value != expected:
        _fail("theme partition replay mismatch")
    return value


@_contract
def validate_theme_partition_capture(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validated_plan = validate_theme_provider_execution_plan(plan)
    return _validate_theme_partition_capture(
        document,
        validated_plan=validated_plan,
        validated_plan_ref=content_ref(validated_plan, identity_field="plan_id"),
    )


@_contract
def capture_theme_partition(
    *,
    plan: Mapping[str, Any],
    partition_ordinal: int,
    captured_at: str,
    client: TushareRequestClient,
) -> dict[str, Any]:
    validated_plan = validate_theme_provider_execution_plan(plan)
    validated_plan_ref = content_ref(validated_plan, identity_field="plan_id")
    api_name, _, company, _ = _partition_identity(validated_plan, partition_ordinal)
    config = _PROVIDERS[validated_plan["provider"]]
    fields = cast(
        tuple[str, ...],
        config["registry_fields"] if api_name.endswith("_index") else config["member_fields"],
    )
    params = {"trade_date": validated_plan["trade_date"]}
    if api_name.endswith("_index"):
        params["idx_type"] = "概念板块"
    else:
        params["con_code"] = company
    try:
        response = client.request(api_name=api_name, params=params, expected_fields=fields)
    except TushareHttpsError as exc:
        return _build_theme_partition_capture(
            validated_plan=validated_plan,
            validated_plan_ref=validated_plan_ref,
            partition_ordinal=partition_ordinal,
            provider_request_id=None,
            reported_count=0,
            rows=[],
            blocker_codes=[f"TRANSPORT_{exc.code}"],
            captured_at=captured_at,
        )
    except Exception:
        return _build_theme_partition_capture(
            validated_plan=validated_plan,
            validated_plan_ref=validated_plan_ref,
            partition_ordinal=partition_ordinal,
            provider_request_id=None,
            reported_count=0,
            rows=[],
            blocker_codes=["TRANSPORT_ERROR"],
            captured_at=captured_at,
        )
    if response.api_name != api_name or tuple(response.fields) != tuple(fields):
        return _build_theme_partition_capture(
            validated_plan=validated_plan,
            validated_plan_ref=validated_plan_ref,
            partition_ordinal=partition_ordinal,
            provider_request_id=None,
            reported_count=0,
            rows=[],
            blocker_codes=["SCHEMA_MISMATCH"],
            captured_at=captured_at,
        )
    if response.has_more:
        return _build_theme_partition_capture(
            validated_plan=validated_plan,
            validated_plan_ref=validated_plan_ref,
            partition_ordinal=partition_ordinal,
            provider_request_id=None,
            reported_count=0,
            rows=[],
            blocker_codes=["HAS_MORE"],
            captured_at=captured_at,
        )
    mapped = [dict(zip(fields, values)) for values in response.rows]
    try:
        return _build_theme_partition_capture(
            validated_plan=validated_plan,
            validated_plan_ref=validated_plan_ref,
            partition_ordinal=partition_ordinal,
            provider_request_id=response.request_id,
            reported_count=response.reported_count,
            rows=mapped,
            blocker_codes=[],
            captured_at=captured_at,
        )
    except TushareContractError:
        return _build_theme_partition_capture(
            validated_plan=validated_plan,
            validated_plan_ref=validated_plan_ref,
            partition_ordinal=partition_ordinal,
            provider_request_id=None,
            reported_count=0,
            rows=[],
            blocker_codes=["CONTENT_INCOMPLETE"],
            captured_at=captured_at,
        )


@_contract
def build_theme_provider_capture(
    *,
    plan: Mapping[str, Any],
    partition_documents: Sequence[Mapping[str, Any]],
    completed_at: str,
) -> dict[str, Any]:
    validated_plan = validate_theme_provider_execution_plan(plan)
    validated_plan_ref = content_ref(validated_plan, identity_field="plan_id")
    expected_count = 1 + len(validated_plan["company_keyset"])
    if len(partition_documents) != expected_count:
        _fail("theme provider partition keyset is incomplete")
    partitions = [
        _validate_theme_partition_capture(
            document,
            validated_plan=validated_plan,
            validated_plan_ref=validated_plan_ref,
        )
        for document in partition_documents
    ]
    if [row["partition_ordinal"] for row in partitions] != list(range(expected_count)):
        _fail("theme provider partition ordering is invalid")
    completed = timestamp(completed_at, label="completed_at")
    if any(row["timestamp"] > completed for row in partitions):
        _fail("theme provider capture predates a partition")
    incomplete = sum(row["status"] == "INCOMPLETE" for row in partitions)
    return seal(
        {
            "kind": THEME_PROVIDER_CAPTURE_KIND,
            **common_fields(timestamp_value=completed),
            "provider": validated_plan["provider"],
            "plan_ref": content_ref(validated_plan, identity_field="plan_id"),
            "status": "COMPLETE" if incomplete == 0 else "PARTIAL",
            "incomplete_partition_count": incomplete,
            "total_row_count": sum(row["reported_count"] for row in partitions),
            "partition_rows": [
                {
                    "partition_ordinal": row["partition_ordinal"],
                    "partition_key": row["partition_key"],
                    "partition_capture_ref": content_ref(
                        row,
                        identity_field="partition_capture_id",
                    ),
                }
                for row in partitions
            ],
        },
        identity_field="capture_id",
    )


@_contract
def validate_theme_provider_capture(
    document: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    partition_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="capture_id")
    require_exact_keys(value, _CAPTURE_FIELDS, label="theme provider capture")
    if value.get("kind") != THEME_PROVIDER_CAPTURE_KIND:
        _fail("theme provider capture kind mismatch")
    for row in value.get("partition_rows", []):
        require_exact_keys(row, _CAPTURE_ROW_FIELDS, label="theme capture partition row")
    expected = build_theme_provider_capture(
        plan=plan,
        partition_documents=partition_documents,
        completed_at=value["timestamp"],
    )
    if value != expected:
        _fail("theme provider capture replay mismatch")
    return value


def _registry_identity(
    rows: Sequence[Mapping[str, Any]], provider: str
) -> tuple[set[str], set[str]]:
    identities: dict[str, bytes] = {}
    conflicts: set[str] = set()
    for row in rows:
        theme_id = f"{provider}:{row['ts_code']}"
        identity = canonical_bytes(row)
        if theme_id in identities and identities[theme_id] != identity:
            conflicts.add(theme_id)
        identities[theme_id] = identity
    return set(identities), conflicts


def _partition_exact_ref(
    partition: Mapping[str, Any],
    *,
    trade_date: str,
) -> dict[str, str]:
    reference = content_ref(partition, identity_field="partition_capture_id")
    cutoff = f"{trade_date[0:4]}-{trade_date[4:6]}-{trade_date[6:8]}T23:59:59Z"
    return {
        **reference,
        "available_at": partition["timestamp"],
        "cutoff": cutoff,
        "relative_path": f"partitions/{partition['partition_ordinal']:05d}.json",
    }


@_contract
def derive_tdx_fallback_company_keyset(
    *,
    dc_plan: Mapping[str, Any],
    dc_capture: Mapping[str, Any],
    dc_partition_documents: Sequence[Mapping[str, Any]],
) -> list[str]:
    plan = validate_theme_provider_execution_plan(dc_plan)
    if plan["provider"] != "TUSHARE_DC":
        _fail("TDX fallback requires a DC plan")
    plan_ref = content_ref(plan, identity_field="plan_id")
    validate_theme_provider_capture(
        dc_capture,
        plan=plan,
        partition_documents=dc_partition_documents,
    )
    partitions = [
        _validate_theme_partition_capture(
            row,
            validated_plan=plan,
            validated_plan_ref=plan_ref,
        )
        for row in dc_partition_documents
    ]
    registry_ids, registry_conflicts = _registry_identity(partitions[0]["rows"], "TUSHARE_DC")
    fallback: list[str] = []
    for company, partition in zip(plan["company_keyset"], partitions[1:]):
        ids = {f"TUSHARE_DC:{row['ts_code']}" for row in partition["rows"]}
        if (
            partition["status"] == "INCOMPLETE"
            or any(theme_id not in registry_ids for theme_id in ids)
            or any(theme_id in registry_conflicts for theme_id in ids)
        ):
            fallback.append(company)
    return fallback


@_contract
def project_theme_provider_capture(
    *,
    plan: Mapping[str, Any],
    capture: Mapping[str, Any],
    partition_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated_plan = validate_theme_provider_execution_plan(plan)
    validated_plan_ref = content_ref(validated_plan, identity_field="plan_id")
    validated_capture = validate_theme_provider_capture(
        capture,
        plan=validated_plan,
        partition_documents=partition_documents,
    )
    partitions = [
        _validate_theme_partition_capture(
            row,
            validated_plan=validated_plan,
            validated_plan_ref=validated_plan_ref,
        )
        for row in partition_documents
    ]
    membership_captures = {}
    for company, partition in zip(validated_plan["company_keyset"], partitions[1:]):
        membership_captures[company] = {
            "status": "COMPLETE" if partition["status"] != "INCOMPLETE" else "INCOMPLETE",
            "rows": partition["rows"],
            "source_ref": _partition_exact_ref(
                partition,
                trade_date=validated_plan["trade_date"],
            ),
        }
    return {
        "captured_at": validated_capture["timestamp"],
        "membership_captures": membership_captures,
        "registry_rows": partitions[0]["rows"],
        "registry_source_ref": _partition_exact_ref(
            partitions[0],
            trade_date=validated_plan["trade_date"],
        ),
    }


__all__ = [
    "THEME_PARTITION_KIND",
    "THEME_PROVIDER_CAPTURE_KIND",
    "THEME_PROVIDER_PLAN_KIND",
    "build_theme_partition_capture",
    "build_theme_provider_capture",
    "build_theme_provider_execution_plan",
    "capture_theme_partition",
    "derive_tdx_fallback_company_keyset",
    "project_theme_provider_capture",
    "validate_theme_partition_capture",
    "validate_theme_provider_capture",
    "validate_theme_provider_execution_plan",
]
