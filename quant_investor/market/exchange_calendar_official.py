"""Fail-closed admission boundary for official CN exchange calendars.

No production wire decoder is registered until exact issuer response bytes,
endpoint semantics, response metadata, and a reviewed decoder are admitted as
one immutable source contract. In particular, project-authored JSON must never
be relabelled as a native SSE, SZSE, or BSE response.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any, Final, Literal
from urllib.parse import parse_qsl, urlsplit

from quant_investor.contracts import ContractError, validate_artifact
from quant_investor.system.errors import SystemContractError

EvidenceRole = Literal[
    "TRADING_WEEK_RULE",
    "SESSION_RULE",
    "ANNUAL_HOLIDAY_NOTICE",
    "TEMPORARY_CLOSURE_NOTICE",
    "SESSION_CHANGE_NOTICE",
    "NOTICE_INDEX_SNAPSHOT",
]
EVIDENCE_ROLES: Final = frozenset(
    {
        "TRADING_WEEK_RULE",
        "SESSION_RULE",
        "ANNUAL_HOLIDAY_NOTICE",
        "TEMPORARY_CLOSURE_NOTICE",
        "SESSION_CHANGE_NOTICE",
        "NOTICE_INDEX_SNAPSHOT",
    }
)

# Adding an entry is a production authority change and requires a retained
# native issuer capture plus an exact endpoint/response/decoder admission
# artifact. Tests may inject synthetic decoders into the assembler module, but
# they must never mutate this registry.
DECODER_IDS: Final[dict[tuple[str, str, str | None], str]] = {}
DECODER_ADMISSIONS: Final[dict[tuple[str, str, str | None], Mapping[str, Any]]] = {}
REQUIRED_CATEGORY_SETS: Final[dict[tuple[str, str], Mapping[str, Any]]] = {}

_UNVERIFIED = "OFFICIAL_CALENDAR_WIRE_CONTRACT_UNVERIFIED"
_ADMISSION_KIND = "system.exchange_calendar_decoder_admission"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_EXCHANGE_AUTHORITIES: Final = {
    "SSE": ("SSE_OFFICIAL", "www.sse.com.cn"),
    "SZSE": ("SZSE_OFFICIAL", "www.szse.cn"),
    "BSE": ("BSE_OFFICIAL", "www.bse.cn"),
}
_SAFE_RESPONSE_HEADERS: Final = frozenset(
    {"cache-control", "content-length", "content-type", "etag", "last-modified"}
)
_ADMISSION_FIELDS: Final = frozenset(
    {
        "decoder_admission_id",
        "state",
        "exchange_id",
        "evidence_role",
        "issuer",
        "endpoint_scheme",
        "endpoint_host",
        "endpoint_path_query_template",
        "issuer_category_id",
        "category_scope",
        "category_completeness_policy",
        "query_window_semantics",
        "required_query_parameters",
        "page_parameter",
        "cursor_parameter",
        "required_category_set_id",
        "discovery_start_date",
        "fixture_request_url",
        "fixture_effective_url",
        "fixture_redirect_chain",
        "fixture_tls_verified",
        "redirect_policy",
        "http_status",
        "raw_media_type",
        "response_headers",
        "fixture_raw_file_ref",
        "fixture_raw_sha256",
        "fixture_captured_at",
        "decoder_id",
        "decoder_sha256",
        "fixture_projection_sha256",
        "review_basis",
    }
)
_INDEX_POLICY_FIELDS: Final = frozenset(
    {
        "exchange_id",
        "required_category_set_id",
        "issuer",
        "category_scope",
        "category_completeness_policy",
        "query_window_semantics",
        "required_query_parameters",
        "page_parameter",
        "cursor_parameter",
        "required_issuer_category_ids",
        "category_role_rows",
        "discovery_start_date",
        "maximum_page_count",
        "maximum_body_count",
    }
)
_QUERY_VALUE_SOURCES: Final = frozenset(
    {
        "ISSUER_CATEGORY_ID",
        "PAGE_NUMBER",
        "DISCOVERY_PUBLISH_START_DATE",
        "DISCOVERY_PUBLISH_END_DATE",
    }
)
_MAXIMUM_REDIRECTS: Final = 5


def decoder_code_sha256() -> str:
    """Return the exact installed admission-boundary module byte identity."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _reject(exchange: str, role: str) -> SystemContractError:
    return SystemContractError(f"{_UNVERIFIED}: {exchange}/{role}")


def _validate_endpoint(payload: Mapping[str, Any]) -> None:
    exchange = payload["exchange_id"]
    role = payload["evidence_role"]
    if exchange not in _EXCHANGE_AUTHORITIES or role not in EVIDENCE_ROLES:
        raise SystemContractError("official calendar decoder admission subject differs")
    issuer, hostname = _EXCHANGE_AUTHORITIES[exchange]
    endpoint = payload["endpoint_path_query_template"]
    parsed = urlsplit(endpoint) if type(endpoint) is str else None
    if (
        payload["issuer"] != issuer
        or payload["endpoint_scheme"] != "https"
        or payload["endpoint_host"] != hostname
        or parsed is None
        or not endpoint.startswith("/")
        or parsed.scheme
        or parsed.netloc
        or parsed.fragment
    ):
        raise SystemContractError("official calendar decoder endpoint admission differs")
    if role != "NOTICE_INDEX_SNAPSHOT" and ("{" in endpoint or "}" in endpoint):
        raise SystemContractError("non-index calendar endpoint admission may not contain wildcards")


def _canonical_date(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not an ISO date")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemContractError(f"{label} is not an ISO date") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise SystemContractError(f"{label} is not an ISO date")
    return value


def _query_parameters(value: Any) -> list[dict[str, str]]:
    if type(value) is not list:
        raise SystemContractError("official calendar query-parameter policy differs")
    rows: list[dict[str, str]] = []
    for row in value:
        if (
            type(row) is not dict
            or set(row) != {"name", "value_source"}
            or type(row["name"]) is not str
            or not row["name"]
            or row["name"].strip() != row["name"]
            or row["value_source"] not in _QUERY_VALUE_SOURCES
        ):
            raise SystemContractError("official calendar query-parameter policy differs")
        rows.append(dict(row))
    if rows != sorted(rows, key=lambda row: row["name"]) or len(
        {row["name"] for row in rows}
    ) != len(rows):
        raise SystemContractError("official calendar query-parameter policy is not canonical")
    return rows


def _validate_index_policy_fields(payload: Mapping[str, Any]) -> None:  # noqa: C901
    role = payload["evidence_role"]
    policy_fields = (
        "issuer_category_id",
        "category_scope",
        "category_completeness_policy",
        "query_window_semantics",
        "page_parameter",
        "cursor_parameter",
        "required_category_set_id",
        "discovery_start_date",
    )
    if role != "NOTICE_INDEX_SNAPSHOT":
        if (
            any(payload[field] is not None for field in policy_fields)
            or payload["required_query_parameters"] != []
        ):
            raise SystemContractError("non-index calendar admission has index authority")
        return
    for field in (
        "issuer_category_id",
        "category_scope",
        "required_category_set_id",
    ):
        value = payload[field]
        if type(value) is not str or not value or value.strip() != value:
            raise SystemContractError("official calendar index admission identity differs")
    if (
        payload["category_completeness_policy"] != "COMPLETE_ISSUER_CATEGORY_PAGINATION"
        or payload["query_window_semantics"] != "PUBLISH_DATE_INCLUSIVE"
        or type(payload["page_parameter"]) is not str
        or not payload["page_parameter"]
        or payload["cursor_parameter"] is not None
    ):
        raise SystemContractError("official calendar index admission semantics differ")
    _canonical_date(payload["discovery_start_date"], label="discovery_start_date")
    parameters = _query_parameters(payload["required_query_parameters"])
    if {row["value_source"] for row in parameters} != _QUERY_VALUE_SOURCES:
        raise SystemContractError("official calendar index query semantics are incomplete")
    page_rows = [row for row in parameters if row["value_source"] == "PAGE_NUMBER"]
    if len(page_rows) != 1 or page_rows[0]["name"] != payload["page_parameter"]:
        raise SystemContractError("official calendar index page mapping differs")
    template = urlsplit(payload["endpoint_path_query_template"])
    try:
        query = parse_qsl(template.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise SystemContractError("official calendar index endpoint query differs") from exc
    if len(query) != len(parameters) or {name for name, _ in query} != {
        row["name"] for row in parameters
    }:
        raise SystemContractError("official calendar index endpoint query differs")
    source_by_name = {row["name"]: row["value_source"] for row in parameters}
    if any(value != "{" + source_by_name[name] + "}" for name, value in query):
        raise SystemContractError("official calendar index endpoint mapping differs")


def _validate_redirects(
    *,
    policy: str,
    exchange: str,
    request_url: str,
    effective_url: str,
    redirects: Any,
    label: str,
) -> None:
    if type(redirects) is not list or any(type(item) is not str or not item for item in redirects):
        raise SystemContractError(f"{label} redirect chain differs")
    if len(redirects) > _MAXIMUM_REDIRECTS:
        raise SystemContractError(f"{label} redirect chain exceeds its bound")
    expected_host = _EXCHANGE_AUTHORITIES[exchange][1]
    urls = [request_url, *redirects, effective_url]
    for url in urls:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != expected_host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise SystemContractError(f"{label} redirect authority differs")
    hops = [request_url, *redirects]
    if len(hops) != len(set(hops)):
        raise SystemContractError(f"{label} redirect chain contains a loop")
    if policy == "NO_REDIRECTS":
        if redirects != [] or effective_url != request_url:
            raise SystemContractError(f"{label} violates NO_REDIRECTS")
    elif policy == "SAME_ISSUER_HOST_ONLY":
        if (redirects and redirects[-1] != effective_url) or (
            not redirects and effective_url != request_url
        ):
            raise SystemContractError(f"{label} redirect chain is incomplete")
    else:
        raise SystemContractError(f"{label} redirect policy differs")


def _validate_response(payload: Mapping[str, Any]) -> None:
    if (
        payload["redirect_policy"] not in {"NO_REDIRECTS", "SAME_ISSUER_HOST_ONLY"}
        or payload["http_status"] != 200
        or type(payload["raw_media_type"]) is not str
        or not payload["raw_media_type"]
    ):
        raise SystemContractError("official calendar decoder response admission differs")
    headers = payload["response_headers"]
    if (
        type(headers) is not list
        or not headers
        or any(
            type(row) is not dict
            or set(row) != {"name", "value"}
            or type(row["name"]) is not str
            or row["name"] != row["name"].lower()
            or type(row["value"]) is not str
            or row["name"] not in _SAFE_RESPONSE_HEADERS
            for row in headers
        )
        or not any(
            row["name"] == "content-type" and row["value"] == payload["raw_media_type"]
            for row in headers
        )
    ):
        raise SystemContractError("official calendar response-header admission differs")
    request_url = payload["fixture_request_url"]
    effective_url = payload["fixture_effective_url"]
    redirects = payload["fixture_redirect_chain"]
    if (
        type(request_url) is not str
        or type(effective_url) is not str
        or type(redirects) is not list
        or any(type(item) is not str for item in redirects)
        or payload["fixture_tls_verified"] is not True
    ):
        raise SystemContractError("official calendar fixture transport admission differs")
    _validate_redirects(
        policy=payload["redirect_policy"],
        exchange=payload["exchange_id"],
        request_url=request_url,
        effective_url=effective_url,
        redirects=redirects,
        label="official calendar fixture",
    )


def _validate_fixture(payload: Mapping[str, Any]) -> None:
    fixture_ref = payload["fixture_raw_file_ref"]
    if (
        type(fixture_ref) is not dict
        or set(fixture_ref) != {"relative_path", "byte_sha256", "size"}
        or type(fixture_ref["relative_path"]) is not str
        or not fixture_ref["relative_path"]
        or type(fixture_ref["size"]) is not int
        or fixture_ref["size"] <= 0
        or fixture_ref["byte_sha256"] != payload["fixture_raw_sha256"]
    ):
        raise SystemContractError("official calendar native fixture admission differs")
    for field in ("fixture_raw_sha256", "decoder_sha256", "fixture_projection_sha256"):
        if type(payload[field]) is not str or _SHA256_RE.fullmatch(payload[field]) is None:
            raise SystemContractError("official calendar decoder SHA admission differs")
    captured_at = payload["fixture_captured_at"]
    try:
        parsed_at = datetime.strptime(captured_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError) as exc:
        raise SystemContractError("official calendar fixture capture time differs") from exc
    if parsed_at.strftime("%Y-%m-%dT%H:%M:%SZ") != captured_at:
        raise SystemContractError("official calendar fixture capture time differs")
    for field in ("decoder_admission_id", "decoder_id", "review_basis"):
        if (
            type(payload[field]) is not str
            or not payload[field]
            or payload[field].strip() != payload[field]
        ):
            raise SystemContractError("official calendar decoder text admission differs")


def validate_decoder_admission(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate the mandatory source contract for a future native decoder."""

    try:
        artifact = validate_artifact(document, expected_kind=_ADMISSION_KIND)
    except ContractError as exc:
        raise SystemContractError("official calendar decoder admission contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _ADMISSION_FIELDS or payload["state"] != "ADMITTED":
        raise SystemContractError("official calendar decoder admission fields differ")
    _validate_endpoint(payload)
    _validate_index_policy_fields(payload)
    _validate_response(payload)
    _validate_fixture(payload)
    return artifact


def decoder_admission(
    exchange: str, role: EvidenceRole, issuer_category_id: str | None = None
) -> dict[str, Any]:
    try:
        document = DECODER_ADMISSIONS[(exchange, role, issuer_category_id)]
    except KeyError as exc:
        raise _reject(exchange, role) from exc
    artifact = validate_decoder_admission(document)
    payload = artifact["payload"]
    if (
        payload["exchange_id"] != exchange
        or payload["evidence_role"] != role
        or payload["issuer_category_id"] != issuer_category_id
    ):
        raise SystemContractError("official calendar decoder registry subject differs")
    return artifact


def decoder_id(exchange: str, role: EvidenceRole, issuer_category_id: str | None = None) -> str:
    """Return an admitted production decoder identity or fail closed."""

    artifact = decoder_admission(exchange, role, issuer_category_id)
    value = artifact["payload"]["decoder_id"]
    if DECODER_IDS.get((exchange, role, issuer_category_id)) != value:
        raise SystemContractError("official calendar decoder registry identity differs")
    return value


def validate_required_category_set(document: Mapping[str, Any]) -> dict[str, Any]:
    if type(document) is not dict or set(document) != _INDEX_POLICY_FIELDS:
        raise SystemContractError("official calendar required-category set fields differ")
    result = dict(document)
    exchange = result["exchange_id"]
    if (
        exchange not in _EXCHANGE_AUTHORITIES
        or result["issuer"] != _EXCHANGE_AUTHORITIES[exchange][0]
    ):
        raise SystemContractError("official calendar required-category set authority differs")
    for field in ("required_category_set_id", "category_scope"):
        if (
            type(result[field]) is not str
            or not result[field]
            or result[field].strip() != result[field]
        ):
            raise SystemContractError("official calendar required-category set identity differs")
    if (
        result["category_completeness_policy"] != "COMPLETE_ISSUER_CATEGORY_PAGINATION"
        or result["query_window_semantics"] != "PUBLISH_DATE_INCLUSIVE"
        or type(result["page_parameter"]) is not str
        or not result["page_parameter"]
        or result["cursor_parameter"] is not None
    ):
        raise SystemContractError("official calendar required-category set semantics differ")
    parameters = _query_parameters(result["required_query_parameters"])
    if {row["value_source"] for row in parameters} != _QUERY_VALUE_SOURCES:
        raise SystemContractError("official calendar required-category query semantics differ")
    page_rows = [row for row in parameters if row["value_source"] == "PAGE_NUMBER"]
    if len(page_rows) != 1 or page_rows[0]["name"] != result["page_parameter"]:
        raise SystemContractError("official calendar required-category page mapping differs")
    categories = result["required_issuer_category_ids"]
    if (
        type(categories) is not list
        or not categories
        or categories != sorted(set(categories))
        or any(
            type(value) is not str or not value or value.strip() != value for value in categories
        )
    ):
        raise SystemContractError("official calendar required categories are not canonical")
    category_role_rows = result["category_role_rows"]
    if type(category_role_rows) is not list:
        raise SystemContractError("official calendar category-role policy differs")
    normalized_role_rows: list[dict[str, Any]] = []
    body_roles = EVIDENCE_ROLES - {"NOTICE_INDEX_SNAPSHOT", "TRADING_WEEK_RULE", "SESSION_RULE"}
    for row in category_role_rows:
        if (
            type(row) is not dict
            or set(row) != {"issuer_category_id", "allowed_evidence_roles"}
            or type(row["issuer_category_id"]) is not str
            or not row["issuer_category_id"]
            or type(row["allowed_evidence_roles"]) is not list
            or not row["allowed_evidence_roles"]
            or row["allowed_evidence_roles"] != sorted(set(row["allowed_evidence_roles"]))
            or any(role not in body_roles for role in row["allowed_evidence_roles"])
        ):
            raise SystemContractError("official calendar category-role policy differs")
        normalized_role_rows.append(dict(row))
    if (
        normalized_role_rows
        != sorted(normalized_role_rows, key=lambda row: row["issuer_category_id"])
        or [row["issuer_category_id"] for row in normalized_role_rows] != categories
    ):
        raise SystemContractError("official calendar category-role union differs")
    _canonical_date(result["discovery_start_date"], label="category.discovery_start_date")
    if (
        type(result["maximum_page_count"]) is not int
        or not 1 <= result["maximum_page_count"] <= 256
        or type(result["maximum_body_count"]) is not int
        or not 1 <= result["maximum_body_count"] <= 4096
    ):
        raise SystemContractError("official calendar required-category resource bound differs")
    return result


def required_category_set(exchange: str, set_id: str) -> dict[str, Any]:
    try:
        policy = REQUIRED_CATEGORY_SETS[(exchange, set_id)]
    except KeyError as exc:
        raise SystemContractError(f"{_UNVERIFIED}: {exchange}/{set_id}") from exc
    result = validate_required_category_set(policy)
    if result["exchange_id"] != exchange or result["required_category_set_id"] != set_id:
        raise SystemContractError("official calendar required-category registry subject differs")
    return result


validate_redirect_chain = _validate_redirects


def decode_session_intervals(exchange: str, raw: bytes, *, media_type: str) -> list[dict[str, str]]:
    """Reject session evidence while no native issuer contract is admitted."""

    del raw, media_type
    raise _reject(exchange, "SESSION_RULE")


def decode_capture_projection(
    exchange: str,
    role: EvidenceRole,
    raw: bytes,
    *,
    media_type: str,
) -> Mapping[str, object]:
    """Reject every unadmitted issuer body before it can gain authority."""

    del raw, media_type
    decoder_id(exchange, role)
    raise _reject(exchange, role)  # pragma: no cover - registry is empty


__all__ = [
    "DECODER_IDS",
    "DECODER_ADMISSIONS",
    "REQUIRED_CATEGORY_SETS",
    "EVIDENCE_ROLES",
    "decode_capture_projection",
    "decode_session_intervals",
    "decoder_code_sha256",
    "decoder_admission",
    "decoder_id",
    "required_category_set",
    "validate_decoder_admission",
    "validate_redirect_chain",
    "validate_required_category_set",
]
