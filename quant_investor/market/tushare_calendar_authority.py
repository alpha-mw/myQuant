"""Fail-closed trusted-provider authority for the unified daily CN calendar.

This module admits Tushare ``trade_cal`` only as a degraded daily Factor
processing authority.  It never claims exchange-official provenance or
intraday/session authority.  SSE and SZSE are replayed directly; BSE is an
explicit policy projection whose direct provider probe must be exact-empty.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
import ctypes
from datetime import date, datetime, time, timedelta, timezone
import errno
import hashlib
from html.parser import HTMLParser
import http.client
import fcntl
import os
from pathlib import Path
import re
import secrets
import ssl
import stat
from typing import Any, Final, NoReturn

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    parse_canonical_json_bytes,
    seal_artifact,
    validate_artifact,
)
from quant_investor.system.errors import (
    SystemContractError,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
)
from quant_investor.system.release_install import (
    validate_release_install_evidence,
    verify_release_install_input,
)
from quant_investor.system.store import object_ref_for_artifact, validate_object_ref

from .exchange_calendar_closure import runtime_json_bytes, runtime_parquet_bytes
from .tushare_transport import (
    OFFICIAL_TUSHARE_URL,
    TushareHttpsError,
    OfficialTushareHttpsClient,
    replay_tushare_response_bytes,
)

POLICY_KIND: Final = "system.calendar_authority_policy"
CAPTURE_KIND: Final = "system.trusted_provider_calendar_capture"
CAPABILITY_KIND: Final = "system.trusted_provider_calendar_capability"
COMPILATION_KIND: Final = "system.trusted_provider_calendar_compilation"
CAPTURE_TRANSACTION_KIND: Final = "system.trusted_provider_calendar_capture_transaction"
CAPTURE_EXECUTION_KIND: Final = "system.trusted_provider_calendar_capture_execution"
CAPTURE_SUCCESS_KIND: Final = "system.trusted_provider_calendar_capture_success"
CAPTURE_FAILURE_KIND: Final = "system.trusted_provider_calendar_capture_failure"
OFFICIAL_COMPILATION_KIND: Final = "system.exchange_calendar_compilation"
AUTHORITY_ROUTE: Final = "TRUSTED_PROVIDER_DEGRADED"
AUTHORITY_TIER: Final = "TIER_1_TRUSTED_PROVIDER"
CONFIDENCE: Final = "DEGRADED"
PROVIDER: Final = "TUSHARE"
API_NAME: Final = "trade_cal"
DOCS_URL: Final = "https://tushare.pro/document/2?doc_id=26"
TIMEZONE: Final = "Asia/Shanghai"
RUNTIME_START_DATE: Final = date(2024, 1, 1)
CAPTURE_PREHISTORY_DAYS: Final = 31
EXPECTED_FIELDS: Final = ("exchange", "cal_date", "is_open", "pretrade_date")
DIRECT_EXCHANGES: Final = ("SSE", "SZSE")
PROBE_EXCHANGES: Final = ("BSE",)
PROJECTED_EXCHANGES: Final = ("BSE",)
PROJECTION_SOURCE_EXCHANGES: Final = ("SSE", "SZSE")
DEGRADED_PIT_EXCHANGES: Final = ("BSE", "SSE", "SZSE")
SOURCE_LIMITATIONS: Final = (
    "BSE_CALENDAR_POLICY_PROJECTED_FROM_SSE_SZSE",
    "CALENDAR_AUTHORITY_DEGRADED",
)
TIME_SEMANTICS: Final = "DAILY_FACTOR_PROCESSING_ENVELOPE"
ENVELOPE_SOURCE: Final = "CODE_OWNED_POLICY"
PROCESSING_OPEN_LOCAL: Final = "09:30:00"
PROCESSING_CLOSE_LOCAL: Final = "15:00:00"
COMPILER_RELATIVE_PATH: Final = "quant_investor/market/tushare_calendar_authority.py"
DOCS_DECODER_ID: Final = "tushare-trade-cal-documentation-html"
CAPABILITY_CONCLUSION: Final = "BSE_DIRECT_CALENDAR_SUPPORT_NOT_ESTABLISHED"
POLICY_USER_AUTHORIZATION_BASIS: Final = (
    "USER_ACCEPTED_TUSHARE_TRUSTED_PROVIDER_DEGRADED_CALENDAR_AUTHORITY"
)

_FILE_REF_FIELDS: Final = frozenset({"relative_path", "byte_sha256"})
_SAFE_HEADERS: Final = frozenset(
    {"cache-control", "content-length", "content-type", "etag", "last-modified"}
)
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_FAILURE_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,159}$")
_FAILURE_ROOT_SUFFIX: Final = ".failure"
_FAILURE_LEAF: Final = "capture-failure.json"
_CALENDAR_RESPONSE_MAX_BYTES: Final = 4 * 1024 * 1024
_CALENDAR_AGGREGATE_MAX_BYTES: Final = 16 * 1024 * 1024
_OPERATOR_SPEC: Final = {
    "api_name": API_NAME,
    "documentation_url": DOCS_URL,
    "endpoint_url": OFFICIAL_TUSHARE_URL,
    "exchange_ids": [*DIRECT_EXCHANGES, *PROBE_EXCHANGES],
    "expected_fields": list(EXPECTED_FIELDS),
    "per_response_max_bytes": _CALENDAR_RESPONSE_MAX_BYTES,
    "aggregate_max_bytes": _CALENDAR_AGGREGATE_MAX_BYTES,
}

RawResolver = Callable[[Mapping[str, Any]], bytes]


_CAPTURE_FAILURE_PHASE_CODES: Final = {
    "DOCUMENTATION_FETCH": "TRUSTED_PROVIDER_CALENDAR_DOCUMENTATION_FETCH_FAILED",
    "PROVIDER_SSE": "TRUSTED_PROVIDER_CALENDAR_SSE_REQUEST_FAILED",
    "PROVIDER_SZSE": "TRUSTED_PROVIDER_CALENDAR_SZSE_REQUEST_FAILED",
    "PROVIDER_BSE": "TRUSTED_PROVIDER_CALENDAR_BSE_REQUEST_FAILED",
    "EVIDENCE_BUILD": "TRUSTED_PROVIDER_CALENDAR_EVIDENCE_INVALID",
    "PUBLICATION": "TRUSTED_PROVIDER_CALENDAR_PUBLICATION_FAILED",
    "SUCCESS_VALIDATION": "TRUSTED_PROVIDER_CALENDAR_SUCCESS_VALIDATION_FAILED",
}
_TUSHARE_FAILURE_CODES: Final = frozenset(
    {
        "TUSHARE_API_ERROR",
        "TUSHARE_CLIENT_CONFIG_INVALID",
        "TUSHARE_ENDPOINT_BLOCKED",
        "TUSHARE_HTTP_STATUS_ERROR",
        "TUSHARE_REDIRECT_BLOCKED",
        "TUSHARE_REQUEST_INVALID",
        "TUSHARE_RESPONSE_INVALID",
        "TUSHARE_RESPONSE_TOO_LARGE",
        "TUSHARE_TOKEN_MISSING",
        "TUSHARE_TRANSPORT_ERROR",
    }
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def compiler_code_sha256() -> str:
    return _sha256(Path(__file__).read_bytes())


def compiler_ast_sha256() -> str:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=COMPILER_RELATIVE_PATH)
    return _sha256(ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8"))


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not a canonical identifier")
    return value


def _capture_root_name(value: Any) -> str:
    root_name = _identifier(value, label="capture_root_name")
    if (
        "/" in root_name
        or root_name.startswith(".")
        or len(root_name) + len(_FAILURE_ROOT_SUFFIX) > 200
    ):
        raise SystemSecurityError("trusted-provider capture root name is invalid")
    return root_name


def _failure_root_name(capture_root_name: str) -> str:
    return _identifier(
        _capture_root_name(capture_root_name) + _FAILURE_ROOT_SUFFIX,
        label="capture_failure_root_name",
    )


def _failure_code(value: Any) -> str:
    if type(value) is not str or _FAILURE_CODE_RE.fullmatch(value) is None:
        raise SystemContractError("capture failure error code is not controlled")
    lowered = value.casefold()
    if any(secret in lowered for secret in ("authorization", "bearer", "secret", "token=")):
        raise SystemSecurityError("capture failure error code is sensitive")
    return value


def _text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(ord(character) < 0x20 for character in value)
    ):
        raise SystemContractError(f"{label} is not canonical text")
    return value


def _date(value: Any, *, label: str) -> date:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not an ISO date") from exc
    if parsed.isoformat() != value:
        raise SystemContractError(f"{label} is not an ISO date")
    return parsed


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} is not canonical UTC seconds")
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not lowercase SHA-256")
    return value


def _file_ref(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != _FILE_REF_FIELDS:
        raise SystemContractError(f"{label} fields are not exact")
    relative = value.get("relative_path")
    if (
        type(relative) is not str
        or not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise SystemContractError(f"{label}.relative_path is invalid")
    return {
        "relative_path": relative,
        "byte_sha256": _sha(value.get("byte_sha256"), label=f"{label}.byte_sha256"),
    }


def _headers(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict:
        raise SystemContractError(f"{label} is not an exact object")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if (
            type(raw_key) is not str
            or raw_key != raw_key.lower()
            or raw_key not in _SAFE_HEADERS
            or type(raw_value) is not str
            or any(ord(character) < 0x20 for character in raw_value)
        ):
            raise SystemSecurityError(f"{label} contains a sensitive or invalid header")
        normalized[raw_key] = raw_value
    if list(normalized) != sorted(normalized):
        raise SystemContractError(f"{label} is not key-sorted")
    return normalized


def _artifact(document: Mapping[str, Any] | bytes, kind: str) -> dict[str, Any]:
    try:
        return validate_artifact(
            document,
            expected_kind=kind,
            expected_contract_sha256=get_contract(kind).contract_sha256,
        )
    except ContractError as exc:
        raise SystemContractError(f"{kind} contract failed") from exc


class _DocumentationParser(HTMLParser):
    """Collect headings and table cells without interpreting page scripts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = []
        self.rows: list[tuple[str, list[str]]] = []
        self._heading_tag: str | None = None
        self._heading_text: list[str] = []
        self._section = ""
        self._in_cell = False
        self._cell_text: list[str] = []
        self._row: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"h1", "h2", "h3", "h4", "strong"}:
            self._heading_tag = tag
            self._heading_text = []
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._in_cell = True
            self._cell_text = []

    def handle_data(self, data: str) -> None:
        if self._heading_tag is not None:
            self._heading_text.append(data)
        if self._in_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._heading_tag:
            text = " ".join("".join(self._heading_text).split())
            if text:
                self.headings.append(text)
                if "输入参数" in text:
                    self._section = "INPUT"
                elif "输出参数" in text:
                    self._section = "OUTPUT"
            self._heading_tag = None
            self._heading_text = []
        elif tag in {"td", "th"} and self._in_cell:
            text = " ".join("".join(self._cell_text).split())
            assert self._row is not None
            self._row.append(text)
            self._in_cell = False
            self._cell_text = []
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append((self._section, list(self._row)))
            self._row = None


def _exchange_ids(text: str) -> list[str]:
    order = re.findall(r"(?<![A-Z])(?:SSE|SZSE|BSE|CFFEX|SHFE|CZCE|DCE|INE)(?![A-Z])", text)
    return list(dict.fromkeys(order))


def decode_trade_cal_documentation(raw: bytes) -> dict[str, Any]:  # noqa: C901
    """Machine-project the retained official Tushare documentation bytes."""

    if type(raw) is not bytes or not raw or len(raw) > 4 * 1024 * 1024:
        raise SystemSecurityError("Tushare documentation bytes are invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise SystemContractError("Tushare documentation is not strict UTF-8") from exc
    parser = _DocumentationParser()
    try:
        parser.feed(text)
        parser.close()
    except (AssertionError, ValueError) as exc:
        raise SystemContractError("Tushare documentation HTML is malformed") from exc
    if sum(heading == "交易日历" for heading in parser.headings) != 1:
        raise SystemContractError("trade_cal documentation heading is absent or ambiguous")
    if len(re.findall(r"接口\s*[：:]\s*trade_cal", text)) != 1:
        raise SystemContractError("trade_cal API declaration is absent or ambiguous")
    input_exchange_rows = [
        row for section, row in parser.rows if section == "INPUT" and row and row[0] == "exchange"
    ]
    output_exchange_rows = [
        row for section, row in parser.rows if section == "OUTPUT" and row and row[0] == "exchange"
    ]
    output_field_rows = [
        row[0]
        for section, row in parser.rows
        if section == "OUTPUT" and row and row[0] in EXPECTED_FIELDS
    ]
    if len(input_exchange_rows) != 1 or len(output_exchange_rows) != 1:
        raise SystemContractError("trade_cal exchange documentation is absent or ambiguous")
    input_ids = _exchange_ids(" ".join(input_exchange_rows[0][1:]))
    output_ids = _exchange_ids(" ".join(output_exchange_rows[0][1:]))
    if input_ids != ["SSE", "SZSE", "CFFEX", "SHFE", "CZCE", "DCE", "INE"]:
        raise SystemContractError("trade_cal input exchange documentation drifted")
    if output_ids != ["SSE", "SZSE"]:
        raise SystemContractError("trade_cal stock output exchange documentation drifted")
    if output_field_rows != list(EXPECTED_FIELDS):
        raise SystemContractError("trade_cal documented output fields drifted")
    if "BSE" in input_ids or "BSE" in output_ids:
        bse_state = "LISTED"
    else:
        bse_state = "NOT_LISTED"
    projection = {
        "api_name": API_NAME,
        "documented_input_exchange_ids": input_ids,
        "documented_stock_output_exchange_ids": output_ids,
        "documented_fields": list(EXPECTED_FIELDS),
        "bse_documentation_state": bse_state,
    }
    return projection


def build_trusted_provider_calendar_capability(
    *,
    docs_raw: bytes,
    docs_raw_file_ref: Mapping[str, Any],
    docs_captured_at: str,
    docs_http_status: int,
    docs_tls_verified: bool,
    docs_redirect_chain: Sequence[str],
    docs_response_headers: Mapping[str, str],
    created_at: str,
) -> dict[str, Any]:
    raw_ref = _file_ref(docs_raw_file_ref, label="docs_raw_file_ref")
    if _sha256(docs_raw) != raw_ref["byte_sha256"]:
        raise SystemSecurityError("Tushare documentation raw SHA differs")
    if docs_http_status != 200 or docs_tls_verified is not True or list(docs_redirect_chain):
        raise SystemSecurityError("Tushare documentation transport is not exact HTTPS")
    projection = decode_trade_cal_documentation(docs_raw)
    if projection["bse_documentation_state"] != "NOT_LISTED":
        raise SystemPreconditionError("Tushare documentation does not prove the degraded route")
    captured_at = _timestamp(docs_captured_at, label="docs_captured_at")
    if _timestamp(created_at, label="created_at") != captured_at:
        raise SystemContractError("capability created_at/captured_at differs")
    body = {
        "state": "VERIFIED",
        "provider": PROVIDER,
        "api_name": API_NAME,
        "docs_url": DOCS_URL,
        "docs_captured_at": captured_at,
        "docs_http_status": docs_http_status,
        "docs_tls_verified": docs_tls_verified,
        "docs_redirect_chain": [],
        "docs_response_headers": _headers(
            dict(docs_response_headers), label="docs_response_headers"
        ),
        "docs_raw_file_ref": raw_ref,
        "docs_raw_sha256": _sha256(docs_raw),
        "docs_raw_byte_length": len(docs_raw),
        "decoder_id": DOCS_DECODER_ID,
        "decoder_relative_path": COMPILER_RELATIVE_PATH,
        "decoder_code_sha256": compiler_code_sha256(),
        "decoder_ast_sha256": compiler_ast_sha256(),
        "decoder_projection_sha256": _sha256(canonical_json_bytes(projection)),
        **projection,
        "conclusion": CAPABILITY_CONCLUSION,
    }
    identity = "tushare-calendar-capability-" + _sha256(canonical_json_bytes(body))
    return validate_trusted_provider_calendar_capability(
        seal_artifact(
            CAPABILITY_KIND,
            {"calendar_capability_id": identity, **body},
            created_at=created_at,
        ),
        docs_raw=docs_raw,
    )


def validate_trusted_provider_calendar_capability(
    document: Mapping[str, Any] | bytes,
    *,
    docs_raw: bytes,
    historical: bool = False,
) -> dict[str, Any]:
    artifact = _artifact(document, CAPABILITY_KIND)
    payload = artifact["payload"]
    raw_ref = _file_ref(payload["docs_raw_file_ref"], label="docs_raw_file_ref")
    if (
        payload["state"] != "VERIFIED"
        or payload["provider"] != PROVIDER
        or payload["api_name"] != API_NAME
        or payload["docs_url"] != DOCS_URL
        or payload["docs_http_status"] != 200
        or payload["docs_tls_verified"] is not True
        or payload["docs_redirect_chain"] != []
        or payload["docs_raw_sha256"] != raw_ref["byte_sha256"]
        or payload["docs_raw_sha256"] != _sha256(docs_raw)
        or payload["docs_raw_byte_length"] != len(docs_raw)
        or payload["decoder_id"] != DOCS_DECODER_ID
        or payload["decoder_relative_path"] != COMPILER_RELATIVE_PATH
        or payload["conclusion"] != CAPABILITY_CONCLUSION
    ):
        raise SystemContractError("trusted-provider capability binding differs")
    _headers(payload["docs_response_headers"], label="docs_response_headers")
    _timestamp(payload["docs_captured_at"], label="docs_captured_at")
    projection = decode_trade_cal_documentation(docs_raw)
    expected = {
        "documented_input_exchange_ids": projection["documented_input_exchange_ids"],
        "documented_stock_output_exchange_ids": projection["documented_stock_output_exchange_ids"],
        "documented_fields": projection["documented_fields"],
        "bse_documentation_state": "NOT_LISTED",
        "decoder_projection_sha256": _sha256(canonical_json_bytes(projection)),
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise SystemContractError("trusted-provider capability projection differs")
    if not historical and (
        payload["decoder_code_sha256"] != compiler_code_sha256()
        or payload["decoder_ast_sha256"] != compiler_ast_sha256()
    ):
        raise SystemSecurityError("trusted-provider documentation decoder drifted")
    identity_body = dict(payload)
    identity = identity_body.pop("calendar_capability_id")
    if identity != "tushare-calendar-capability-" + _sha256(canonical_json_bytes(identity_body)):
        raise SystemContractError("trusted-provider capability identity differs")
    return artifact


def build_calendar_authority_policy(
    *,
    created_at: str,
    authority_route: str = AUTHORITY_ROUTE,
    pit_exchange_ids: Sequence[str] = (),
    provider_capability: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    _timestamp(created_at, label="created_at")
    capability_ref = None
    if provider_capability is not None:
        capability_ref = object_ref_for_artifact(_artifact(provider_capability, CAPABILITY_KIND))
    body = build_policy_payload(
        authority_route=authority_route,
        pit_exchange_ids=pit_exchange_ids,
        provider_capability_ref=capability_ref,
    )
    identity = "calendar-policy-" + _sha256(canonical_json_bytes(body))
    return validate_calendar_authority_policy(
        seal_artifact(
            POLICY_KIND,
            {"calendar_authority_policy_id": identity, **body},
            created_at=created_at,
        )
    )


def validate_calendar_authority_policy(
    document: Mapping[str, Any] | bytes,
    *,
    pit_exchange_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    artifact = _artifact(document, POLICY_KIND)
    payload = artifact["payload"]
    route = payload.get("authority_route")
    if route == AUTHORITY_ROUTE:
        expected = build_policy_payload(
            provider_capability_ref=payload.get("provider_capability_ref")
        )
    elif route == "EXCHANGE_OFFICIAL":
        exchanges = payload.get("direct_exchange_official_calendar_exchange_ids")
        if type(exchanges) is not list:
            raise SystemContractError("official calendar policy exchange set is invalid")
        expected = build_policy_payload(
            authority_route="EXCHANGE_OFFICIAL",
            pit_exchange_ids=exchanges,
        )
    else:
        raise SystemContractError("calendar authority policy route is invalid")
    if any(payload.get(field) != value for field, value in expected.items()):
        raise SystemContractError("calendar authority policy semantics differ")
    if "production_allowed" in payload or "compilation_ref" in payload:
        raise SystemContractError("calendar policy may not self-authorize or create a cycle")
    policy_scope = sorted(
        {
            *payload["direct_exchange_official_calendar_exchange_ids"],
            *payload["direct_provider_calendar_exchange_ids"],
            *payload["policy_projected_calendar_exchange_ids"],
        }
    )
    if pit_exchange_ids is not None and policy_scope != list(pit_exchange_ids):
        raise SystemContractError("calendar policy/PIT exchange set differs")
    identity_body = dict(payload)
    identity = identity_body.pop("calendar_authority_policy_id")
    if identity != "calendar-policy-" + _sha256(canonical_json_bytes(identity_body)):
        raise SystemContractError("calendar authority policy identity differs")
    return artifact


def build_policy_payload(
    *,
    authority_route: str = AUTHORITY_ROUTE,
    pit_exchange_ids: Sequence[str] = (),
    provider_capability_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if authority_route == AUTHORITY_ROUTE:
        capability_ref = validate_object_ref(
            provider_capability_ref,
            label="provider_capability_ref",
        )
        if capability_ref["kind"] != CAPABILITY_KIND or (
            pit_exchange_ids and list(pit_exchange_ids) != list(DEGRADED_PIT_EXCHANGES)
        ):
            raise SystemContractError("degraded calendar policy scope/capability differs")
        return {
            "state": "SEALED",
            "authority_route": AUTHORITY_ROUTE,
            "requested_scope": "PRODUCTION_FACTOR_DAILY",
            "authority_tier": AUTHORITY_TIER,
            "confidence": CONFIDENCE,
            "expected_compilation_kind": COMPILATION_KIND,
            "direct_exchange_official_calendar_exchange_ids": [],
            "direct_provider_calendar_exchange_ids": list(DIRECT_EXCHANGES),
            "unsupported_or_undocumented_probe_exchange_ids": list(PROBE_EXCHANGES),
            "policy_projected_calendar_exchange_ids": list(PROJECTED_EXCHANGES),
            "provider_capability_ref": capability_ref,
            "source_limitations": list(SOURCE_LIMITATIONS),
            "requires_explicit_final_cutover_authorization": True,
            "user_authorization_basis": POLICY_USER_AUTHORIZATION_BASIS,
            "human_signature_claimed": False,
            "time_semantics": TIME_SEMANTICS,
            "envelope_source": ENVELOPE_SOURCE,
            "timezone": TIMEZONE,
            "processing_open_local": PROCESSING_OPEN_LOCAL,
            "processing_close_local": PROCESSING_CLOSE_LOCAL,
            "full_exchange_session_authority_available": False,
        }
    exchanges = list(pit_exchange_ids)
    if (
        authority_route != "EXCHANGE_OFFICIAL"
        or exchanges != sorted(set(exchanges))
        or not exchanges
        or any(exchange not in {"SSE", "SZSE", "BSE"} for exchange in exchanges)
    ):
        raise SystemContractError("official calendar policy exchange set is invalid")
    return {
        "state": "SEALED",
        "authority_route": "EXCHANGE_OFFICIAL",
        "requested_scope": "PRODUCTION_FACTOR_DAILY",
        "authority_tier": "TIER_2_EXCHANGE_OFFICIAL",
        "confidence": "OFFICIAL",
        "expected_compilation_kind": OFFICIAL_COMPILATION_KIND,
        "direct_exchange_official_calendar_exchange_ids": exchanges,
        "direct_provider_calendar_exchange_ids": [],
        "unsupported_or_undocumented_probe_exchange_ids": [],
        "policy_projected_calendar_exchange_ids": [],
        "provider_capability_ref": None,
        "source_limitations": [],
        "requires_explicit_final_cutover_authorization": True,
        "user_authorization_basis": "EXCHANGE_OFFICIAL_SOURCE_CLOSURE",
        "human_signature_claimed": False,
        "time_semantics": "FULL_EXCHANGE_SESSION_AUTHORITY",
        "envelope_source": "EXCHANGE_ISSUER",
        "timezone": TIMEZONE,
        "processing_open_local": "NOT_APPLICABLE",
        "processing_close_local": "NOT_APPLICABLE",
        "full_exchange_session_authority_available": True,
    }


def _normalized_calendar_rows(  # noqa: C901
    response_rows: Sequence[Sequence[Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(response_rows):
        exchange, raw_date, raw_open, raw_pretrade = row
        if type(exchange) is not str or type(raw_date) is not str:
            raise SystemContractError(f"trade_cal row {index} identity fields are invalid")
        if type(raw_open) is bool or type(raw_open) is not int or raw_open not in {0, 1}:
            raise SystemContractError(f"trade_cal row {index} is_open is invalid")
        if type(raw_pretrade) is not str:
            raise SystemContractError(f"trade_cal row {index} pretrade_date is invalid")
        try:
            parsed = datetime.strptime(raw_date, "%Y%m%d").date()
        except ValueError as exc:
            raise SystemContractError(f"trade_cal row {index} cal_date is invalid") from exc
        pretrade: str | None = None
        if raw_pretrade:
            try:
                pretrade = datetime.strptime(raw_pretrade, "%Y%m%d").date().isoformat()
            except ValueError as exc:
                raise SystemContractError(
                    f"trade_cal row {index} pretrade_date is invalid"
                ) from exc
        rows.append(
            {
                "exchange_id": exchange,
                "date": parsed.isoformat(),
                "status": "OPEN" if raw_open == 1 else "CLOSED",
                "pretrade_date": pretrade,
            }
        )
    rows.sort(key=lambda value: value["date"])
    if len({row["date"] for row in rows}) != len(rows):
        raise SystemContractError("trade_cal rows contain duplicate dates")
    return rows


def _validate_finite_pretrade_chain(
    rows: Sequence[Mapping[str, Any]], *, capture_start: date, cutoff: date
) -> tuple[str, str]:
    expected_dates: list[str] = []
    cursor = capture_start
    while cursor <= cutoff:
        expected_dates.append(cursor.isoformat())
        cursor += timedelta(days=1)
    if [row["date"] for row in rows] != expected_dates:
        raise SystemPreconditionError("trade_cal capture does not cover every natural date")
    precoverage_opens = [
        row["date"] for row in rows if row["status"] == "OPEN" and row["date"] < "2024-01-01"
    ]
    if len(precoverage_opens) < 2:
        raise SystemPreconditionError("trade_cal capture has fewer than two precoverage opens")
    anchor = precoverage_opens[-1]
    anchor_row = next(row for row in rows if row["date"] == anchor)
    predecessor = anchor_row["pretrade_date"]
    if predecessor not in precoverage_opens[:-1]:
        raise SystemPreconditionError("trade_cal anchor predecessor is not a captured prior open")
    open_dates: list[str] = []
    for row in rows:
        if row["date"] >= anchor:
            expected_pretrade = open_dates[-1] if open_dates else predecessor
            if row["pretrade_date"] != expected_pretrade:
                raise SystemPreconditionError("trade_cal finite predecessor chain differs")
        if row["status"] == "OPEN":
            open_dates.append(row["date"])
    return anchor, predecessor


def build_trusted_provider_calendar_capture(  # noqa: C901
    *,
    exchange_id: str,
    raw: bytes,
    raw_file_ref: Mapping[str, Any],
    capability: Mapping[str, Any] | bytes,
    docs_raw: bytes,
    captured_at: str,
    capture_start_date: str,
    cutoff_date: str,
    request_parameters_sanitized: Mapping[str, Any],
    response_headers: Mapping[str, str],
    created_at: str,
) -> dict[str, Any]:
    capability_artifact = validate_trusted_provider_calendar_capability(
        capability, docs_raw=docs_raw
    )
    raw_ref = _file_ref(raw_file_ref, label="raw_file_ref")
    if _sha256(raw) != raw_ref["byte_sha256"]:
        raise SystemSecurityError("trusted-provider response SHA differs")
    exchange = _identifier(exchange_id, label="exchange_id")
    if exchange not in {*DIRECT_EXCHANGES, *PROBE_EXCHANGES}:
        raise SystemContractError("trusted-provider exchange is outside the exact route")
    capture_start = _date(capture_start_date, label="capture_start_date")
    cutoff = _date(cutoff_date, label="cutoff_date")
    if capture_start != RUNTIME_START_DATE - timedelta(days=CAPTURE_PREHISTORY_DAYS):
        raise SystemPreconditionError("trusted-provider capture prehistory differs")
    if cutoff < RUNTIME_START_DATE:
        raise SystemPreconditionError("trusted-provider cutoff precedes runtime start")
    try:
        response = replay_tushare_response_bytes(
            raw,
            api_name=API_NAME,
            expected_fields=EXPECTED_FIELDS,
            strict_decimal_decode=True,
        )
    except TushareHttpsError as exc:
        raise SystemContractError("trusted-provider response replay failed") from exc
    parameters = dict(request_parameters_sanitized)
    expected_parameters = {
        "end_date": cutoff.strftime("%Y%m%d"),
        "exchange": exchange,
        "start_date": capture_start.strftime("%Y%m%d"),
    }
    if parameters != expected_parameters:
        raise SystemContractError("trusted-provider request parameters differ")
    if exchange in DIRECT_EXCHANGES:
        rows = _normalized_calendar_rows(response.rows)
        if (
            response.provider_reported_count != 0
            or response.item_count <= 0
            or response.reported_count != response.item_count
            or response.has_more
            or any(row["exchange_id"] != exchange for row in rows)
        ):
            raise SystemContractError("trusted-provider direct response exchange differs")
        _validate_finite_pretrade_chain(rows, capture_start=capture_start, cutoff=cutoff)
        evidence_role = "DIRECT_PROVIDER_CALENDAR"
        authority_conferred = True
        projection = rows
    else:
        if (
            response.provider_reported_count != 0
            or response.reported_count != 0
            or response.rows
            or response.item_count != 0
            or response.has_more
        ):
            raise SystemPreconditionError("BSE capability probe must be exact-empty")
        evidence_role = "PROVIDER_CAPABILITY_PROBE"
        authority_conferred = False
        projection = []
    captured = _timestamp(captured_at, label="captured_at")
    if _timestamp(created_at, label="created_at") != captured:
        raise SystemContractError("capture created_at/captured_at differs")
    body = {
        "state": "VERIFIED",
        "evidence_role": evidence_role,
        "provider": PROVIDER,
        "api_name": API_NAME,
        "exchange_id": exchange,
        "endpoint_url": OFFICIAL_TUSHARE_URL,
        "request_parameters_sanitized": parameters,
        "request_parameters_sha256": _sha256(canonical_json_bytes(parameters)),
        "expected_fields": list(EXPECTED_FIELDS),
        "captured_at": captured,
        "http_status": 200,
        "tls_verified": True,
        "redirect_chain": [],
        "response_headers": _headers(dict(response_headers), label="response_headers"),
        "raw_file_ref": raw_ref,
        "raw_sha256": _sha256(raw),
        "raw_byte_length": len(raw),
        "request_id_sha256": _sha256(response.request_id.encode("utf-8")),
        "provider_reported_count": response.provider_reported_count,
        "item_count": response.item_count,
        "normalized_count": response.reported_count,
        "has_more": response.has_more,
        "capture_start_date": capture_start.isoformat(),
        "cutoff_date": cutoff.isoformat(),
        "projection_sha256": _sha256(canonical_json_bytes(projection)),
        "calendar_authority_conferred": authority_conferred,
        "capability_ref": object_ref_for_artifact(capability_artifact),
    }
    identity = f"tushare-calendar-{exchange.lower()}-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        CAPTURE_KIND,
        {"calendar_capture_id": identity, **body},
        created_at=created_at,
    )
    validate_trusted_provider_calendar_capture(
        artifact,
        raw=raw,
        capability=capability_artifact,
        docs_raw=docs_raw,
    )
    return artifact


def validate_trusted_provider_calendar_capture(  # noqa: C901
    document: Mapping[str, Any] | bytes,
    *,
    raw: bytes,
    capability: Mapping[str, Any] | bytes,
    docs_raw: bytes,
    historical: bool = False,
) -> dict[str, Any]:
    artifact = _artifact(document, CAPTURE_KIND)
    payload = artifact["payload"]
    capability_artifact = validate_trusted_provider_calendar_capability(
        capability, docs_raw=docs_raw, historical=historical
    )
    expected_ref = object_ref_for_artifact(capability_artifact)
    raw_ref = _file_ref(payload["raw_file_ref"], label="raw_file_ref")
    exchange = payload["exchange_id"]
    if (
        payload["state"] != "VERIFIED"
        or payload["provider"] != PROVIDER
        or payload["api_name"] != API_NAME
        or exchange not in {*DIRECT_EXCHANGES, *PROBE_EXCHANGES}
        or payload["endpoint_url"] != OFFICIAL_TUSHARE_URL
        or payload["expected_fields"] != list(EXPECTED_FIELDS)
        or payload["http_status"] != 200
        or payload["tls_verified"] is not True
        or payload["redirect_chain"] != []
        or payload["raw_sha256"] != raw_ref["byte_sha256"]
        or payload["raw_sha256"] != _sha256(raw)
        or payload["raw_byte_length"] != len(raw)
        or payload["capability_ref"] != expected_ref
    ):
        raise SystemContractError("trusted-provider capture binding differs")
    _headers(payload["response_headers"], label="response_headers")
    _timestamp(payload["captured_at"], label="captured_at")
    try:
        response = replay_tushare_response_bytes(
            raw,
            api_name=API_NAME,
            expected_fields=EXPECTED_FIELDS,
            strict_decimal_decode=True,
        )
    except TushareHttpsError as exc:
        raise SystemContractError("trusted-provider response replay failed") from exc
    parameters = payload["request_parameters_sanitized"]
    if (
        type(parameters) is not dict
        or set(parameters) != {"exchange", "start_date", "end_date"}
        or parameters["exchange"] != exchange
        or payload["request_parameters_sha256"] != _sha256(canonical_json_bytes(parameters))
    ):
        raise SystemContractError("trusted-provider request binding differs")
    capture_start = _date(payload["capture_start_date"], label="capture_start_date")
    cutoff = _date(payload["cutoff_date"], label="cutoff_date")
    if parameters != {
        "end_date": cutoff.strftime("%Y%m%d"),
        "exchange": exchange,
        "start_date": capture_start.strftime("%Y%m%d"),
    }:
        raise SystemContractError("trusted-provider request date binding differs")
    if exchange in DIRECT_EXCHANGES:
        projection = _normalized_calendar_rows(response.rows)
        if (
            response.provider_reported_count != 0
            or response.item_count <= 0
            or response.reported_count != response.item_count
            or response.has_more
            or any(row["exchange_id"] != exchange for row in projection)
        ):
            raise SystemContractError("trusted-provider direct response exchange differs")
        _validate_finite_pretrade_chain(projection, capture_start=capture_start, cutoff=cutoff)
        expected_role = "DIRECT_PROVIDER_CALENDAR"
        expected_authority = True
    else:
        if (
            response.provider_reported_count != 0
            or response.reported_count != 0
            or response.rows
            or response.item_count != 0
            or response.has_more
        ):
            raise SystemPreconditionError("BSE capability probe must be exact-empty")
        projection = []
        expected_role = "PROVIDER_CAPABILITY_PROBE"
        expected_authority = False
    expected = {
        "evidence_role": expected_role,
        "provider_reported_count": response.provider_reported_count,
        "item_count": response.item_count,
        "normalized_count": response.reported_count,
        "has_more": response.has_more,
        "projection_sha256": _sha256(canonical_json_bytes(projection)),
        "calendar_authority_conferred": expected_authority,
        "request_id_sha256": _sha256(response.request_id.encode("utf-8")),
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise SystemContractError("trusted-provider capture replay differs")
    identity_body = dict(payload)
    identity = identity_body.pop("calendar_capture_id")
    expected_identity = f"tushare-calendar-{exchange.lower()}-" + _sha256(
        canonical_json_bytes(identity_body)
    )
    if identity != expected_identity:
        raise SystemContractError("trusted-provider capture identity differs")
    return artifact


def validate_trusted_provider_calendar_capture_transaction(
    document: Mapping[str, Any] | bytes,
    *,
    documentation_raw_file_ref: Mapping[str, Any],
    capability_file_ref: Mapping[str, Any],
    policy_file_ref: Mapping[str, Any],
    provider_raw_file_refs: Sequence[Mapping[str, Any]],
    provider_capture_file_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the immutable, all-or-nothing capture publication envelope."""

    artifact = _artifact(document, CAPTURE_TRANSACTION_KIND)
    payload = artifact["payload"]
    docs_ref = _file_ref(
        documentation_raw_file_ref,
        label="documentation_raw_file_ref",
    )
    capability_ref = _file_ref(capability_file_ref, label="capability_file_ref")
    policy_ref = _file_ref(policy_file_ref, label="policy_file_ref")
    raw_refs = [
        _file_ref(reference, label=f"provider_raw_file_refs[{ordinal}]")
        for ordinal, reference in enumerate(provider_raw_file_refs)
    ]
    capture_refs = [
        _file_ref(reference, label=f"provider_capture_file_refs[{ordinal}]")
        for ordinal, reference in enumerate(provider_capture_file_refs)
    ]
    if (
        payload["state"] != "COMPLETE"
        or payload["network_call_count"] != 4
        or payload["source_limitations"] != list(SOURCE_LIMITATIONS)
        or payload["documentation_raw_file_ref"] != docs_ref
        or payload["capability_file_ref"] != capability_ref
        or payload["policy_file_ref"] != policy_ref
        or payload["provider_raw_file_refs"] != raw_refs
        or payload["provider_capture_file_refs"] != capture_refs
    ):
        raise SystemContractError("trusted-provider capture transaction binding differs")
    _identifier(payload["capture_root_name"], label="capture_root_name")
    capture_start = _date(payload["capture_start_date"], label="capture_start_date")
    cutoff = _date(payload["cutoff_date"], label="cutoff_date")
    _timestamp(payload["captured_at"], label="captured_at")
    if capture_start >= RUNTIME_START_DATE or cutoff < RUNTIME_START_DATE:
        raise SystemContractError("trusted-provider capture transaction dates differ")
    leaves = sorted(
        [docs_ref, capability_ref, policy_ref, *raw_refs, *capture_refs],
        key=lambda row: row["relative_path"],
    )
    if payload["all_leaves_sha256"] != _sha256(canonical_json_bytes(leaves)):
        raise SystemContractError("trusted-provider capture transaction leaves differ")
    identity_body = dict(payload)
    identity = identity_body.pop("capture_transaction_id")
    expected_identity = "tushare-calendar-capture-" + _sha256(canonical_json_bytes(identity_body))
    if identity != expected_identity:
        raise SystemContractError("trusted-provider capture transaction identity differs")
    return artifact


def build_trusted_provider_calendar_capture_transaction(
    *,
    capture_root_name: str,
    capture_start_date: str,
    cutoff_date: str,
    captured_at: str,
    documentation_raw_file_ref: Mapping[str, Any],
    capability_file_ref: Mapping[str, Any],
    policy_file_ref: Mapping[str, Any],
    provider_raw_file_refs: Sequence[Mapping[str, Any]],
    provider_capture_file_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    docs_ref = _file_ref(
        documentation_raw_file_ref,
        label="documentation_raw_file_ref",
    )
    capability_ref = _file_ref(capability_file_ref, label="capability_file_ref")
    policy_ref = _file_ref(policy_file_ref, label="policy_file_ref")
    raw_refs = sorted(
        (
            _file_ref(reference, label="provider_raw_file_ref")
            for reference in provider_raw_file_refs
        ),
        key=lambda row: row["relative_path"],
    )
    capture_refs = sorted(
        (
            _file_ref(reference, label="provider_capture_file_ref")
            for reference in provider_capture_file_refs
        ),
        key=lambda row: row["relative_path"],
    )
    leaves = sorted(
        [docs_ref, capability_ref, policy_ref, *raw_refs, *capture_refs],
        key=lambda row: row["relative_path"],
    )
    body = {
        "state": "COMPLETE",
        "capture_root_name": _identifier(
            capture_root_name,
            label="capture_root_name",
        ),
        "capture_start_date": _date(
            capture_start_date,
            label="capture_start_date",
        ).isoformat(),
        "cutoff_date": _date(cutoff_date, label="cutoff_date").isoformat(),
        "captured_at": _timestamp(captured_at, label="captured_at"),
        "documentation_raw_file_ref": docs_ref,
        "capability_file_ref": capability_ref,
        "policy_file_ref": policy_ref,
        "provider_raw_file_refs": raw_refs,
        "provider_capture_file_refs": capture_refs,
        "network_call_count": 4,
        "source_limitations": list(SOURCE_LIMITATIONS),
        "all_leaves_sha256": _sha256(canonical_json_bytes(leaves)),
    }
    artifact = seal_artifact(
        CAPTURE_TRANSACTION_KIND,
        {
            "capture_transaction_id": "tushare-calendar-capture-"
            + _sha256(canonical_json_bytes(body)),
            **body,
        },
        created_at=captured_at,
    )
    validate_trusted_provider_calendar_capture_transaction(
        artifact,
        documentation_raw_file_ref=docs_ref,
        capability_file_ref=capability_ref,
        policy_file_ref=policy_ref,
        provider_raw_file_refs=raw_refs,
        provider_capture_file_refs=capture_refs,
    )
    return artifact


def _capture_projection(raw: bytes) -> list[dict[str, Any]]:
    response = replay_tushare_response_bytes(
        raw,
        api_name=API_NAME,
        expected_fields=EXPECTED_FIELDS,
        strict_decimal_decode=True,
    )
    return _normalized_calendar_rows(response.rows)


def build_trusted_provider_calendar_compilation(  # noqa: C901
    *,
    compilation_id: str,
    policy: Mapping[str, Any] | bytes,
    capability: Mapping[str, Any] | bytes,
    capture_documents: Sequence[Mapping[str, Any] | bytes],
    docs_raw: bytes,
    raw_resolver: RawResolver,
    release_ref: Mapping[str, Any],
    pit_exchange_ids: Sequence[str],
    market_session_dates: Sequence[str],
    cutoff_date: str,
    calendar_json_file_ref: Mapping[str, Any],
    calendar_parquet_file_ref: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    policy_artifact = validate_calendar_authority_policy(policy)
    capability_artifact = validate_trusted_provider_calendar_capability(
        capability, docs_raw=docs_raw
    )
    release = validate_object_ref(release_ref, label="release_ref")
    if release["kind"] != "system.release":
        raise SystemContractError("trusted-provider release authority differs")
    if policy_artifact["payload"]["provider_capability_ref"] != object_ref_for_artifact(
        capability_artifact
    ):
        raise SystemContractError("trusted-provider policy/capability binding differs")
    exchanges = list(pit_exchange_ids)
    if exchanges != list(DEGRADED_PIT_EXCHANGES):
        raise SystemContractError("PIT exchange set is not canonical")
    cutoff = _date(cutoff_date, label="cutoff_date")
    capture_start = RUNTIME_START_DATE - timedelta(days=CAPTURE_PREHISTORY_DAYS)
    captures: dict[str, dict[str, Any]] = {}
    raw_by_exchange: dict[str, bytes] = {}
    for document in capture_documents:
        shallow = _artifact(document, CAPTURE_KIND)
        exchange = shallow["payload"]["exchange_id"]
        if exchange in captures:
            raise SystemContractError("trusted-provider capture exchange is duplicated")
        raw = raw_resolver(shallow["payload"]["raw_file_ref"])
        captures[exchange] = validate_trusted_provider_calendar_capture(
            shallow,
            raw=raw,
            capability=capability_artifact,
            docs_raw=docs_raw,
        )
        raw_by_exchange[exchange] = raw
    if set(captures) != {"SSE", "SZSE", "BSE"}:
        raise SystemPreconditionError("trusted-provider capture set is incomplete")
    direct_rows = {
        exchange: _capture_projection(raw_by_exchange[exchange]) for exchange in DIRECT_EXCHANGES
    }
    normalized = {
        exchange: [
            {
                "date": row["date"],
                "status": row["status"],
                "pretrade_date": row["pretrade_date"],
            }
            for row in direct_rows[exchange]
        ]
        for exchange in DIRECT_EXCHANGES
    }
    if normalized["SSE"] != normalized["SZSE"]:
        raise SystemPreconditionError("SSE/SZSE trusted-provider projections differ")
    anchor, predecessor = _validate_finite_pretrade_chain(
        direct_rows["SSE"], capture_start=capture_start, cutoff=cutoff
    )
    runtime: list[dict[str, Any]] = []
    for row in normalized["SSE"]:
        if row["date"] < RUNTIME_START_DATE.isoformat():
            continue
        opened = row["status"] == "OPEN"
        session_date = date.fromisoformat(row["date"])
        runtime.append(
            {
                "date": row["date"],
                "status": row["status"],
                "opens_at_utc": (
                    datetime.combine(session_date, time(1, 30), tzinfo=timezone.utc).isoformat()
                    if opened
                    else None
                ),
                "closes_at_utc": (
                    datetime.combine(session_date, time(7, 0), tzinfo=timezone.utc).isoformat()
                    if opened
                    else None
                ),
            }
        )
    open_dates = [row["date"] for row in runtime if row["status"] == "OPEN"]
    if len(open_dates) < 391:
        raise SystemPreconditionError("trusted-provider calendar has fewer than 391 opens")
    sessions = [
        _date(value, label=f"market_session_dates[{index}]").isoformat()
        for index, value in enumerate(market_session_dates)
    ]
    if sessions != sorted(set(sessions)) or not sessions or sessions[-1] != cutoff.isoformat():
        raise SystemContractError("market sessions are not canonical to cutoff")
    closed = sorted(set(sessions) - set(open_dates))
    if closed:
        raise SystemPreconditionError("market bars contradict trusted-provider CLOSED rows")
    json_ref = _file_ref(calendar_json_file_ref, label="calendar_json_file_ref")
    parquet_ref = _file_ref(calendar_parquet_file_ref, label="calendar_parquet_file_ref")
    json_raw = raw_resolver(json_ref)
    parquet_raw = raw_resolver(parquet_ref)
    if (
        _sha256(json_raw) != json_ref["byte_sha256"]
        or json_raw != runtime_json_bytes(runtime)
        or _sha256(parquet_raw) != parquet_ref["byte_sha256"]
        or parquet_raw != runtime_parquet_bytes(runtime)
    ):
        raise SystemSecurityError("trusted-provider runtime exact bytes differ")
    exchange_rows = []
    for exchange in exchanges:
        projected = exchange == "BSE"
        exchange_rows.append(
            {
                "exchange_id": exchange,
                "authority_route": AUTHORITY_ROUTE,
                "calendar_row_origin": "POLICY_PROJECTED" if projected else "PROVIDER_DIRECT",
                "projection_source_exchange_ids": (
                    list(PROJECTION_SOURCE_EXCHANGES) if projected else [exchange]
                ),
                "provider_direct": not projected,
                "exchange_official": False,
                "user_authorized_cross_exchange_assumption": projected,
                "open_session_count": len(open_dates),
                "open_session_sha256": _sha256(canonical_json_bytes(open_dates)),
            }
        )
    body = {
        "state": "COMPILED",
        "authority_route": AUTHORITY_ROUTE,
        "authority_tier": AUTHORITY_TIER,
        "confidence": CONFIDENCE,
        "policy_ref": object_ref_for_artifact(policy_artifact),
        "provider_capability_ref": object_ref_for_artifact(capability_artifact),
        "release_ref": release,
        "coverage_start_date": RUNTIME_START_DATE.isoformat(),
        "capture_start_date": capture_start.isoformat(),
        "cutoff_date": cutoff.isoformat(),
        "timezone": TIMEZONE,
        "pit_exchange_ids": exchanges,
        "direct_provider_calendar_exchange_ids": list(DIRECT_EXCHANGES),
        "unsupported_or_undocumented_probe_exchange_ids": list(PROBE_EXCHANGES),
        "policy_projected_calendar_exchange_ids": list(PROJECTED_EXCHANGES),
        "provider_capture_refs": sorted(
            (object_ref_for_artifact(row) for row in captures.values()),
            key=canonical_json_bytes,
        ),
        "source_limitations": list(SOURCE_LIMITATIONS),
        "time_semantics": TIME_SEMANTICS,
        "envelope_source": ENVELOPE_SOURCE,
        "processing_open_local": PROCESSING_OPEN_LOCAL,
        "processing_close_local": PROCESSING_CLOSE_LOCAL,
        "full_exchange_session_authority_available": False,
        "projection_source_exchange_ids": list(PROJECTION_SOURCE_EXCHANGES),
        "anchor_open_date": anchor,
        "predecessor_open_date": predecessor,
        "capture_projection_sha256": _sha256(canonical_json_bytes(normalized["SSE"])),
        "market_session_dates_sha256": _sha256(canonical_json_bytes(sessions)),
        "exchange_rows": exchange_rows,
        "runtime_projection": runtime,
        "calendar_json_file_ref": json_ref,
        "calendar_parquet_file_ref": parquet_ref,
        "contradiction_rows": [],
        "compiler_relative_path": COMPILER_RELATIVE_PATH,
        "compiler_code_sha256": compiler_code_sha256(),
        "compiler_ast_sha256": compiler_ast_sha256(),
    }
    identity = _identifier(compilation_id, label="compilation_id")
    return _artifact(
        seal_artifact(
            COMPILATION_KIND,
            {"compilation_id": identity, **body},
            created_at=created_at,
        ),
        COMPILATION_KIND,
    )


def validate_trusted_provider_calendar_compilation(  # noqa: C901
    document: Mapping[str, Any] | bytes,
    *,
    policy: Mapping[str, Any] | bytes,
    capability: Mapping[str, Any] | bytes,
    capture_documents: Sequence[Mapping[str, Any] | bytes],
    docs_raw: bytes,
    raw_resolver: RawResolver,
    expected_release_ref: Mapping[str, Any],
    pit_exchange_ids: Sequence[str],
    market_session_dates: Sequence[str],
    historical: bool = False,
) -> dict[str, Any]:
    artifact = _artifact(document, COMPILATION_KIND)
    payload = artifact["payload"]
    rebuilt = build_trusted_provider_calendar_compilation(
        compilation_id=payload["compilation_id"],
        policy=policy,
        capability=capability,
        capture_documents=capture_documents,
        docs_raw=docs_raw,
        raw_resolver=raw_resolver,
        release_ref=expected_release_ref,
        pit_exchange_ids=pit_exchange_ids,
        market_session_dates=market_session_dates,
        cutoff_date=payload["cutoff_date"],
        calendar_json_file_ref=payload["calendar_json_file_ref"],
        calendar_parquet_file_ref=payload["calendar_parquet_file_ref"],
        created_at=artifact["created_at"],
    )
    if historical:
        # Historical validation uses the artifact's frozen compiler identity.
        rebuilt["payload"]["compiler_code_sha256"] = payload["compiler_code_sha256"]
        rebuilt["payload"]["compiler_ast_sha256"] = payload["compiler_ast_sha256"]
        rebuilt = seal_artifact(
            COMPILATION_KIND,
            rebuilt["payload"],
            created_at=artifact["created_at"],
        )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        differing = sorted(
            field
            for field in set(payload) | set(rebuilt["payload"])
            if payload.get(field) != rebuilt["payload"].get(field)
        )
        raise SystemContractError(
            "trusted-provider compilation semantic replay differs:" + ",".join(differing)
        )
    return artifact


def calendar_source_limitations(
    policy: Mapping[str, Any] | bytes,
    compilation: Mapping[str, Any] | bytes,
) -> list[str]:
    policy_artifact = validate_calendar_authority_policy(policy)
    compilation_artifact = _artifact(compilation, COMPILATION_KIND)
    if compilation_artifact["payload"]["policy_ref"] != object_ref_for_artifact(policy_artifact):
        raise SystemContractError("calendar policy/compilation binding differs")
    limitations = compilation_artifact["payload"]["source_limitations"]
    if limitations != list(SOURCE_LIMITATIONS):
        raise SystemContractError("calendar source limitations differ")
    return list(limitations)


def _release_install_components(
    raw: bytes,
    *,
    repository_root: str | os.PathLike[str],
    require_current_operator: bool = False,
    historical: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(raw) is not bytes or not raw or len(raw) > 64 * 1024 * 1024:
        raise SystemSecurityError("release install input is absent or exceeds its bound")
    try:
        value = parse_canonical_json_bytes(raw)
    except ContractError as exc:
        raise SystemContractError("release install input is not canonical") from exc
    if type(value) is not dict or set(value) != {
        "release_install_evidence",
        "deployed_release",
    }:
        raise SystemContractError("release install input fields are not exact")
    evidence = validate_release_install_evidence(value["release_install_evidence"])
    release = _artifact(value["deployed_release"], "system.release")
    if historical:
        payload = evidence["payload"]
        verification = {
            "state": "PASS",
            "release_ref": payload["release_ref"],
            "source_archive_sha256": payload["source_archive"]["byte_sha256"],
            "wheel_sha256": payload["wheel"]["byte_sha256"],
            "code_tree_sha256": payload["code_tree_sha256"],
            "installed_code_manifest_sha256": payload["installed_code_manifest_sha256"],
            "contract_catalog_sha256": payload["contract_catalog_sha256"],
            "import_origin": payload["import_origin"],
        }
    else:
        verification = verify_release_install_input(raw, repository_root=repository_root)
    if (
        verification.get("state") != "PASS"
        or verification.get("release_ref") != object_ref_for_artifact(release)
        or evidence["payload"]["release_ref"] != object_ref_for_artifact(release)
    ):
        raise SystemPreconditionError("release install closure did not pass")
    if require_current_operator:
        expected_operator = (
            Path(verification["import_origin"]).resolve(strict=True).parent
            / "market"
            / "tushare_calendar_authority.py"
        ).resolve(strict=True)
        if Path(__file__).resolve(strict=True) != expected_operator:
            raise SystemPreconditionError("calendar capture is not running installed release")
    return evidence, release, verification


def _rooted_file_refs(
    values: Sequence[Mapping[str, Any]],
    *,
    root_name: str,
    label: str,
) -> list[dict[str, str]]:
    refs = [_file_ref(value, label=f"{label}[{ordinal}]") for ordinal, value in enumerate(values)]
    if (
        refs != sorted(refs, key=lambda row: row["relative_path"])
        or len({row["relative_path"] for row in refs}) != len(refs)
        or any(not row["relative_path"].startswith(f"{root_name}/") for row in refs)
    ):
        raise SystemContractError(f"{label} topology differs")
    return refs


def _build_trusted_provider_calendar_capture_execution(
    *,
    capture_root_name: str,
    release_install_input_raw: bytes,
    release_install_input_file_ref: Mapping[str, Any],
    release_repository_root: str | os.PathLike[str],
    documentation_raw_file_ref: Mapping[str, Any],
    capability_file_ref: Mapping[str, Any],
    policy_file_ref: Mapping[str, Any],
    provider_raw_file_refs: Sequence[Mapping[str, Any]],
    provider_capture_file_refs: Sequence[Mapping[str, Any]],
    capture_transaction_file_ref: Mapping[str, Any],
    observed_started_at: str,
    observed_completed_at: str,
) -> dict[str, Any]:
    root_name = _identifier(capture_root_name, label="capture_root_name")
    if "/" in root_name or root_name.startswith("."):
        raise SystemSecurityError("trusted-provider capture root name is invalid")
    input_ref = _file_ref(
        release_install_input_file_ref,
        label="release_install_input_file_ref",
    )
    if input_ref["relative_path"] != f"{root_name}/release-install-input.json" or input_ref[
        "byte_sha256"
    ] != _sha256(release_install_input_raw):
        raise SystemSecurityError("release install input file binding differs")
    release_root = Path(release_repository_root).resolve(strict=True)
    if not release_root.is_absolute() or str(release_root) != str(release_repository_root):
        raise SystemSecurityError("release repository root is not canonical absolute")
    evidence, release, verification = _release_install_components(
        release_install_input_raw,
        repository_root=release_root,
        require_current_operator=True,
    )
    started = _timestamp(observed_started_at, label="observed_started_at")
    completed = _timestamp(observed_completed_at, label="observed_completed_at")
    if completed < started:
        raise SystemContractError("trusted-provider capture time order differs")
    docs_ref = _file_ref(
        documentation_raw_file_ref,
        label="documentation_raw_file_ref",
    )
    capability_ref = _file_ref(capability_file_ref, label="capability_file_ref")
    policy_ref = _file_ref(policy_file_ref, label="policy_file_ref")
    transaction_ref = _file_ref(
        capture_transaction_file_ref,
        label="capture_transaction_file_ref",
    )
    raw_refs = _rooted_file_refs(
        provider_raw_file_refs,
        root_name=root_name,
        label="provider_raw_file_refs",
    )
    capture_refs = _rooted_file_refs(
        provider_capture_file_refs,
        root_name=root_name,
        label="provider_capture_file_refs",
    )
    scalar_refs = [docs_ref, capability_ref, policy_ref, transaction_ref, input_ref]
    if any(not row["relative_path"].startswith(f"{root_name}/") for row in scalar_refs):
        raise SystemContractError("trusted-provider execution root binding differs")
    evidence_payload = evidence["payload"]
    body = {
        "state": "EXECUTED",
        "capture_root_name": root_name,
        "deployed_release_ref": object_ref_for_artifact(release),
        "release_install_input_file_ref": input_ref,
        "release_install_evidence_ref": object_ref_for_artifact(evidence),
        "release_install_verification_sha256": _sha256(canonical_json_bytes(verification)),
        "release_repository_root": str(release_root),
        "final_commit": evidence_payload["final_commit"],
        "final_tree": evidence_payload["final_tree"],
        "wheel_sha256": verification["wheel_sha256"],
        "installed_code_manifest_sha256": verification["installed_code_manifest_sha256"],
        "contract_catalog_sha256": verification["contract_catalog_sha256"],
        "installed_import_origin": verification["import_origin"],
        "operator_relative_path": COMPILER_RELATIVE_PATH,
        "operator_code_sha256": compiler_code_sha256(),
        "operator_ast_sha256": compiler_ast_sha256(),
        "documentation_raw_file_ref": docs_ref,
        "capability_file_ref": capability_ref,
        "policy_file_ref": policy_ref,
        "provider_raw_file_refs": raw_refs,
        "provider_capture_file_refs": capture_refs,
        "capture_transaction_file_ref": transaction_ref,
        "network_call_count": 4,
        "operation_spec": dict(_OPERATOR_SPEC),
        "observed_started_at": started,
        "observed_completed_at": completed,
        "source_limitations": list(SOURCE_LIMITATIONS),
    }
    artifact = seal_artifact(
        CAPTURE_EXECUTION_KIND,
        {
            "capture_execution_id": "tushare-calendar-execution-"
            + _sha256(canonical_json_bytes(body)),
            **body,
        },
        created_at=completed,
    )
    return validate_trusted_provider_calendar_capture_execution(
        artifact,
        release_install_input_raw=release_install_input_raw,
        documentation_raw_file_ref=docs_ref,
        capability_file_ref=capability_ref,
        policy_file_ref=policy_ref,
        provider_raw_file_refs=raw_refs,
        provider_capture_file_refs=capture_refs,
        capture_transaction_file_ref=transaction_ref,
    )


def validate_trusted_provider_calendar_capture_execution(
    document: Mapping[str, Any] | bytes,
    *,
    release_install_input_raw: bytes,
    documentation_raw_file_ref: Mapping[str, Any],
    capability_file_ref: Mapping[str, Any],
    policy_file_ref: Mapping[str, Any],
    provider_raw_file_refs: Sequence[Mapping[str, Any]],
    provider_capture_file_refs: Sequence[Mapping[str, Any]],
    capture_transaction_file_ref: Mapping[str, Any],
    historical: bool = False,
) -> dict[str, Any]:
    artifact = _artifact(document, CAPTURE_EXECUTION_KIND)
    payload = artifact["payload"]
    root_name = _identifier(payload["capture_root_name"], label="capture_root_name")
    input_ref = _file_ref(
        payload["release_install_input_file_ref"],
        label="release_install_input_file_ref",
    )
    if (
        "/" in root_name
        or root_name.startswith(".")
        or input_ref["relative_path"] != f"{root_name}/release-install-input.json"
        or input_ref["byte_sha256"] != _sha256(release_install_input_raw)
    ):
        raise SystemSecurityError("capture execution input binding differs")
    release_root_value = payload.get("release_repository_root")
    if type(release_root_value) is not str:
        raise SystemSecurityError("release repository root binding is absent")
    release_root = Path(release_root_value)
    if (
        not release_root.is_absolute()
        or any(part in {"", ".", ".."} for part in release_root.parts[1:])
        or (not historical and release_root.resolve(strict=True) != release_root)
    ):
        raise SystemSecurityError("release repository root binding is invalid")
    evidence, release, verification = _release_install_components(
        release_install_input_raw,
        repository_root=release_root,
        historical=historical,
    )
    docs_ref = _file_ref(
        documentation_raw_file_ref,
        label="documentation_raw_file_ref",
    )
    capability_ref = _file_ref(capability_file_ref, label="capability_file_ref")
    policy_ref = _file_ref(policy_file_ref, label="policy_file_ref")
    transaction_ref = _file_ref(
        capture_transaction_file_ref,
        label="capture_transaction_file_ref",
    )
    raw_refs = _rooted_file_refs(
        provider_raw_file_refs,
        root_name=root_name,
        label="provider_raw_file_refs",
    )
    capture_refs = _rooted_file_refs(
        provider_capture_file_refs,
        root_name=root_name,
        label="provider_capture_file_refs",
    )
    verification_sha = _sha256(canonical_json_bytes(verification))
    evidence_payload = evidence["payload"]
    expected = {
        "state": "EXECUTED",
        "deployed_release_ref": object_ref_for_artifact(release),
        "release_install_evidence_ref": object_ref_for_artifact(evidence),
        "release_install_verification_sha256": verification_sha,
        "release_repository_root": str(release_root),
        "final_commit": evidence_payload["final_commit"],
        "final_tree": evidence_payload["final_tree"],
        "wheel_sha256": verification["wheel_sha256"],
        "installed_code_manifest_sha256": verification["installed_code_manifest_sha256"],
        "contract_catalog_sha256": verification["contract_catalog_sha256"],
        "installed_import_origin": verification["import_origin"],
        "operator_relative_path": COMPILER_RELATIVE_PATH,
        "documentation_raw_file_ref": docs_ref,
        "capability_file_ref": capability_ref,
        "policy_file_ref": policy_ref,
        "provider_raw_file_refs": raw_refs,
        "provider_capture_file_refs": capture_refs,
        "capture_transaction_file_ref": transaction_ref,
        "network_call_count": 4,
        "operation_spec": dict(_OPERATOR_SPEC),
        "source_limitations": list(SOURCE_LIMITATIONS),
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise SystemContractError("trusted-provider capture execution binding differs")
    started = _timestamp(payload["observed_started_at"], label="observed_started_at")
    completed = _timestamp(
        payload["observed_completed_at"],
        label="observed_completed_at",
    )
    if completed < started or artifact["created_at"] != completed:
        raise SystemContractError("trusted-provider capture execution time differs")
    if not historical and (
        payload["operator_code_sha256"] != compiler_code_sha256()
        or payload["operator_ast_sha256"] != compiler_ast_sha256()
    ):
        raise SystemSecurityError("trusted-provider capture operator drifted")
    _sha(payload["operator_code_sha256"], label="operator_code_sha256")
    _sha(payload["operator_ast_sha256"], label="operator_ast_sha256")
    identity_body = dict(payload)
    identity = identity_body.pop("capture_execution_id")
    if identity != "tushare-calendar-execution-" + _sha256(canonical_json_bytes(identity_body)):
        raise SystemContractError("trusted-provider capture execution identity differs")
    return artifact


def _build_trusted_provider_calendar_capture_success(
    *,
    capture_root_name: str,
    capture_transaction_file_ref: Mapping[str, Any],
    capture_execution_file_ref: Mapping[str, Any],
    published_leaf_file_refs: Sequence[Mapping[str, Any]],
    published_root_device: int,
    published_root_inode: int,
    observed_completed_at: str,
) -> dict[str, Any]:
    root_name = _identifier(capture_root_name, label="capture_root_name")
    transaction_ref = _file_ref(
        capture_transaction_file_ref,
        label="capture_transaction_file_ref",
    )
    execution_ref = _file_ref(
        capture_execution_file_ref,
        label="capture_execution_file_ref",
    )
    leaves = _rooted_file_refs(
        published_leaf_file_refs,
        root_name=root_name,
        label="published_leaf_file_refs",
    )
    if transaction_ref not in leaves or execution_ref not in leaves:
        raise SystemContractError("capture success omits authority leaves")
    if (
        type(published_root_device) is not int
        or published_root_device < 0
        or type(published_root_inode) is not int
        or published_root_inode <= 0
    ):
        raise SystemSecurityError("capture success root identity is invalid")
    body = {
        "state": "COMPLETE",
        "capture_root_name": root_name,
        "capture_transaction_file_ref": transaction_ref,
        "capture_execution_file_ref": execution_ref,
        "published_leaf_file_refs": leaves,
        "published_leaves_sha256": _sha256(canonical_json_bytes(leaves)),
        "published_root_device": published_root_device,
        "published_root_inode": published_root_inode,
        "observed_completed_at": _timestamp(
            observed_completed_at,
            label="observed_completed_at",
        ),
    }
    artifact = seal_artifact(
        CAPTURE_SUCCESS_KIND,
        {
            "capture_success_id": "tushare-calendar-success-" + _sha256(canonical_json_bytes(body)),
            **body,
        },
        created_at=str(body["observed_completed_at"]),
    )
    return validate_trusted_provider_calendar_capture_success(
        artifact,
        capture_transaction_file_ref=transaction_ref,
        capture_execution_file_ref=execution_ref,
        published_leaf_file_refs=leaves,
    )


def validate_trusted_provider_calendar_capture_success(
    document: Mapping[str, Any] | bytes,
    *,
    capture_transaction_file_ref: Mapping[str, Any],
    capture_execution_file_ref: Mapping[str, Any],
    published_leaf_file_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact = _artifact(document, CAPTURE_SUCCESS_KIND)
    payload = artifact["payload"]
    root_name = _identifier(payload["capture_root_name"], label="capture_root_name")
    transaction_ref = _file_ref(
        capture_transaction_file_ref,
        label="capture_transaction_file_ref",
    )
    execution_ref = _file_ref(
        capture_execution_file_ref,
        label="capture_execution_file_ref",
    )
    leaves = _rooted_file_refs(
        published_leaf_file_refs,
        root_name=root_name,
        label="published_leaf_file_refs",
    )
    expected = {
        "state": "COMPLETE",
        "capture_transaction_file_ref": transaction_ref,
        "capture_execution_file_ref": execution_ref,
        "published_leaf_file_refs": leaves,
        "published_leaves_sha256": _sha256(canonical_json_bytes(leaves)),
        "published_root_device": payload["published_root_device"],
        "published_root_inode": payload["published_root_inode"],
    }
    if (
        any(payload[field] != value for field, value in expected.items())
        or transaction_ref not in leaves
        or execution_ref not in leaves
        or type(payload["published_root_device"]) is not int
        or payload["published_root_device"] < 0
        or type(payload["published_root_inode"]) is not int
        or payload["published_root_inode"] <= 0
    ):
        raise SystemContractError("trusted-provider capture success binding differs")
    completed = _timestamp(
        payload["observed_completed_at"],
        label="observed_completed_at",
    )
    if artifact["created_at"] != completed:
        raise SystemContractError("trusted-provider capture success time differs")
    identity_body = dict(payload)
    identity = identity_body.pop("capture_success_id")
    if identity != "tushare-calendar-success-" + _sha256(canonical_json_bytes(identity_body)):
        raise SystemContractError("trusted-provider capture success identity differs")
    return artifact


def _build_trusted_provider_calendar_capture_failure(
    *,
    capture_root_name: str,
    failed_at: str,
    error_code: str,
    success_root_published: bool,
    published_root_device: int,
    published_root_inode: int,
) -> dict[str, Any]:
    """Seal the only non-sensitive terminal record for a failed capture.

    A failure document is diagnostic custody only.  It can never substitute
    for the success-last execution/root evidence required by production
    assembly.
    """

    root_name = _capture_root_name(capture_root_name)
    timestamp = _timestamp(failed_at, label="failed_at")
    code = _failure_code(error_code)
    if type(success_root_published) is not bool:
        raise SystemContractError("success_root_published is not boolean")
    if (
        type(published_root_device) is not int
        or published_root_device < 0
        or type(published_root_inode) is not int
        or published_root_inode <= 0
    ):
        raise SystemSecurityError("capture failure root identity is invalid")
    body = {
        "state": "FAILED",
        "capture_root_name": root_name,
        "failed_at": timestamp,
        "error_code": code,
        "success_root_published": success_root_published,
        "published_root_device": published_root_device,
        "published_root_inode": published_root_inode,
    }
    artifact = seal_artifact(
        CAPTURE_FAILURE_KIND,
        {
            "capture_failure_id": "tushare-calendar-failure-" + _sha256(canonical_json_bytes(body)),
            **body,
        },
        created_at=timestamp,
    )
    return validate_trusted_provider_calendar_capture_failure(artifact)


def validate_trusted_provider_calendar_capture_failure(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(document, CAPTURE_FAILURE_KIND)
    payload = artifact["payload"]
    root_name = _capture_root_name(payload["capture_root_name"])
    failed_at = _timestamp(payload["failed_at"], label="failed_at")
    code = _failure_code(payload["error_code"])
    published = payload["success_root_published"]
    device = payload["published_root_device"]
    inode = payload["published_root_inode"]
    if (
        type(published) is not bool
        or payload["state"] != "FAILED"
        or type(device) is not int
        or device < 0
        or type(inode) is not int
        or inode <= 0
    ):
        raise SystemContractError("trusted-provider capture failure state differs")
    if artifact["created_at"] != failed_at:
        raise SystemContractError("trusted-provider capture failure time differs")
    body = {
        "state": "FAILED",
        "capture_root_name": root_name,
        "failed_at": failed_at,
        "error_code": code,
        "success_root_published": published,
        "published_root_device": device,
        "published_root_inode": inode,
    }
    expected_id = "tushare-calendar-failure-" + _sha256(canonical_json_bytes(body))
    if payload["capture_failure_id"] != expected_id:
        raise SystemContractError("trusted-provider capture failure identity differs")
    return artifact


def _official_documentation_fetch() -> tuple[bytes, int, Mapping[str, str], bool, Sequence[str]]:
    connection: http.client.HTTPSConnection | None = None
    try:
        connection = http.client.HTTPSConnection(
            "tushare.pro",
            443,
            timeout=20.0,
            context=ssl.create_default_context(),
        )
        connection.request(
            "GET",
            "/document/2?doc_id=26",
            headers={"Accept": "text/html", "User-Agent": "myquant-calendar-authority"},
        )
        response = connection.getresponse()
        if 300 <= response.status < 400:
            raise SystemSecurityError("TRUSTED_PROVIDER_DOCUMENTATION_REDIRECT_BLOCKED")
        if response.status != 200:
            raise SystemPreconditionError("TRUSTED_PROVIDER_DOCUMENTATION_HTTP_FAILED")
        raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise SystemSecurityError("TRUSTED_PROVIDER_DOCUMENTATION_TOO_LARGE")
        headers = {
            key.lower(): value
            for key, value in response.getheaders()
            if key.lower() in _SAFE_HEADERS
        }
        return raw, response.status, dict(sorted(headers.items())), True, []
    except (SystemSecurityError, SystemPreconditionError):
        raise
    except BaseException as exc:
        raise SystemPreconditionError("TRUSTED_PROVIDER_DOCUMENTATION_FETCH_FAILED") from exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException:
                pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_nlink,
        value.st_size,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid)


def _verify_owner_directory(value: os.stat_result, *, exact_mode: bool) -> None:
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISDIR(value.st_mode):
        raise SystemSecurityError("trusted-provider capture directory is unsafe")
    if exact_mode:
        if value.st_uid != os.geteuid() or mode != 0o700:
            raise SystemSecurityError("trusted-provider capture directory is unsafe")
        return
    root_sticky_directory = value.st_uid == 0 and bool(mode & stat.S_ISVTX)
    if value.st_uid not in {0, os.geteuid()} or (mode & 0o022 and not root_sticky_directory):
        raise SystemSecurityError("trusted-provider capture ancestor is unsafe")


def _verify_owner_file(value: os.stat_result) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or value.st_nlink != 1
        or stat.S_IMODE(value.st_mode) != 0o600
    ):
        raise SystemSecurityError("trusted-provider capture file is unsafe")


def _open_pinned_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise SystemSecurityError("trusted-provider capture parent must be canonical absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            _verify_owner_directory(os.fstat(child), exact_mode=False)
            os.close(descriptor)
            descriptor = child
        _verify_owner_directory(os.fstat(descriptor), exact_mode=True)
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise SystemSecurityError("trusted-provider capture parent traversal is unsafe") from exc
    except BaseException:
        os.close(descriptor)
        raise


def _read_fd_exact(descriptor: int, *, expected: bytes) -> None:
    before = os.fstat(descriptor)
    _verify_owner_file(before)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while total <= len(expected):
        chunk = os.read(descriptor, min(1024 * 1024, len(expected) + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    after = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(after) or b"".join(chunks) != expected:
        raise SystemSecurityError("trusted-provider capture descriptor readback differs")


def _write_fd_leaf(directory_fd: int, leaf: str, raw: bytes) -> None:
    if "/" in leaf or leaf in {"", ".", ".."} or type(raw) is not bytes or not raw:
        raise SystemSecurityError("trusted-provider capture leaf is invalid")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(leaf, flags, 0o600, dir_fd=directory_fd)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SystemStorageError("trusted-provider capture write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        _read_fd_exact(descriptor, expected=raw)
        path_stat = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_identity(path_stat) != _stat_identity(os.fstat(descriptor)):
            raise SystemSecurityError("trusted-provider capture leaf path drifted")
    finally:
        os.close(descriptor)


def _read_directory_files(
    directory_fd: int,
    *,
    expected: Mapping[str, bytes],
) -> None:
    with os.scandir(directory_fd) as entries:
        names = sorted(entry.name for entry in entries)
    if names != sorted(expected):
        raise SystemSecurityError("trusted-provider capture directory topology differs")
    for leaf, raw in sorted(expected.items()):
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            _read_fd_exact(descriptor, expected=raw)
            path_stat = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
            if _stat_identity(path_stat) != _stat_identity(os.fstat(descriptor)):
                raise SystemSecurityError("trusted-provider capture path identity differs")
        finally:
            os.close(descriptor)


def _read_ref_leaf(
    directory_fd: int,
    *,
    leaf: str,
    expected_sha256: str,
) -> bytes:
    descriptor = os.open(
        leaf,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        _verify_owner_file(before)
        chunks: list[bytes] = []
        total = 0
        while total <= _CALENDAR_AGGREGATE_MAX_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _CALENDAR_AGGREGATE_MAX_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        path_stat = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        raw = b"".join(chunks)
        if (
            not raw
            or total > _CALENDAR_AGGREGATE_MAX_BYTES
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(path_stat)
            or _sha256(raw) != expected_sha256
        ):
            raise SystemSecurityError("trusted-provider capture leaf readback differs")
        return raw
    finally:
        os.close(descriptor)


def validate_published_trusted_provider_calendar_capture_root(
    *,
    capture_parent: str | os.PathLike[str],
    capture_execution: Mapping[str, Any] | bytes,
    capture_execution_file_ref: Mapping[str, Any],
    capture_success: Mapping[str, Any] | bytes,
    capture_success_file_ref: Mapping[str, Any],
) -> dict[str, bytes]:
    """Pin and replay the sole success-last capture custody topology.

    This proves exact local bytes, reviewed operator identity, and immutable
    owner-only custody.  Tushare responses are not provider-signed, so this is
    deliberately not a cryptographic proof that a network peer emitted them.
    """

    execution = _artifact(capture_execution, CAPTURE_EXECUTION_KIND)
    execution_payload = execution["payload"]
    root_name = _identifier(execution_payload["capture_root_name"], label="capture_root_name")
    if "/" in root_name or root_name.startswith("."):
        raise SystemSecurityError("trusted-provider capture root name is invalid")
    execution_ref = _file_ref(
        capture_execution_file_ref,
        label="capture_execution_file_ref",
    )
    success_ref = _file_ref(
        capture_success_file_ref,
        label="capture_success_file_ref",
    )
    expected_paths = {
        "documentation.raw": execution_payload["documentation_raw_file_ref"],
        "release-install-input.json": execution_payload["release_install_input_file_ref"],
        "capability.json": execution_payload["capability_file_ref"],
        "policy.json": execution_payload["policy_file_ref"],
        "capture-transaction.json": execution_payload["capture_transaction_file_ref"],
        "capture-execution.json": execution_ref,
    }
    raw_refs = {
        row["relative_path"].removeprefix(f"{root_name}/"): row
        for row in execution_payload["provider_raw_file_refs"]
    }
    capture_refs = {
        row["relative_path"].removeprefix(f"{root_name}/"): row
        for row in execution_payload["provider_capture_file_refs"]
    }
    expected_paths.update(raw_refs)
    expected_paths.update(capture_refs)
    expected_leaf_names = {
        "documentation.raw",
        "release-install-input.json",
        "capability.json",
        "policy.json",
        "response-sse.raw",
        "response-szse.raw",
        "response-bse.raw",
        "capture-sse.json",
        "capture-szse.json",
        "capture-bse.json",
        "capture-transaction.json",
        "capture-execution.json",
    }
    if (
        set(expected_paths) != expected_leaf_names
        or any(
            _file_ref(reference, label=f"capture root {leaf}")["relative_path"]
            != f"{root_name}/{leaf}"
            for leaf, reference in expected_paths.items()
        )
        or execution_ref["relative_path"] != f"{root_name}/capture-execution.json"
        or success_ref["relative_path"] != f"{root_name}/capture-success.json"
    ):
        raise SystemContractError("trusted-provider capture exact topology differs")
    success = validate_trusted_provider_calendar_capture_success(
        capture_success,
        capture_transaction_file_ref=execution_payload["capture_transaction_file_ref"],
        capture_execution_file_ref=execution_ref,
        published_leaf_file_refs=sorted(
            expected_paths.values(), key=lambda row: row["relative_path"]
        ),
    )
    if success["payload"]["capture_root_name"] != root_name:
        raise SystemContractError("trusted-provider capture success root differs")

    parent = Path(capture_parent)
    parent_fd = _open_pinned_absolute_directory(parent)
    root_fd: int | None = None
    try:
        try:
            os.stat(
                _failure_root_name(root_name),
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise SystemPreconditionError("trusted-provider capture has terminal failure evidence")
        root_fd = os.open(
            root_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        root_stat = os.fstat(root_fd)
        _verify_owner_directory(root_stat, exact_mode=True)
        if (
            success["payload"]["published_root_device"] != root_stat.st_dev
            or success["payload"]["published_root_inode"] != root_stat.st_ino
        ):
            raise SystemSecurityError("trusted-provider published root identity differs")
        with os.scandir(root_fd) as entries:
            observed_names = sorted(entry.name for entry in entries)
        if observed_names != sorted({*expected_leaf_names, "capture-success.json"}):
            raise SystemSecurityError("trusted-provider capture directory topology differs")
        result: dict[str, bytes] = {}
        aggregate = 0
        for leaf, reference in sorted(
            {**expected_paths, "capture-success.json": success_ref}.items()
        ):
            raw = _read_ref_leaf(
                root_fd,
                leaf=leaf,
                expected_sha256=reference["byte_sha256"],
            )
            aggregate += len(raw)
            if aggregate > _CALENDAR_AGGREGATE_MAX_BYTES:
                raise SystemSecurityError("trusted-provider capture root exceeds byte bound")
            result[leaf] = raw
        if result["capture-execution.json"] != canonical_json_bytes(execution) or result[
            "capture-success.json"
        ] != canonical_json_bytes(success):
            raise SystemSecurityError("trusted-provider capture authority bytes differ")
        return result
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _rename_no_replace(parent_fd: int, source: str, target: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    result: int
    if hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(parent_fd, source.encode(), parent_fd, target.encode(), 0x00000004)
    elif hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(parent_fd, source.encode(), parent_fd, target.encode(), 1)
    else:
        raise SystemSecurityError("atomic no-replace rename is unavailable")
    if result != 0:
        observed = ctypes.get_errno()
        if observed in {errno.EEXIST, errno.ENOTEMPTY}:
            raise SystemPreconditionError("trusted-provider capture destination exists")
        raise SystemStorageError("trusted-provider capture no-replace publication failed")


def _file_reference(root_name: str, leaf: str, raw: bytes) -> dict[str, str]:
    return {
        "relative_path": f"{root_name}/{leaf}",
        "byte_sha256": _sha256(raw),
    }


def _open_capture_lock(parent_fd: int) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    for _attempt in range(3):
        try:
            return os.open(
                ".tushare-calendar-capture.lock",
                flags,
                0o600,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            continue
    raise SystemStorageError("trusted-provider capture lock could not be opened")


def _publish_capture_tree(
    *,
    parent: Path,
    root_name: str,
    files: Mapping[str, bytes],
    success_builder: Callable[[str, os.stat_result], bytes],
) -> bytes:
    parent_fd = _open_pinned_absolute_directory(parent)
    parent_identity = _directory_identity(os.fstat(parent_fd))
    lock_fd: int | None = None
    staging_fd: int | None = None
    final_fd: int | None = None
    staging = f".{root_name}.staging-{os.getpid()}-{secrets.token_hex(8)}"
    published = False
    try:
        lock_fd = _open_capture_lock(parent_fd)
        os.fchmod(lock_fd, 0o600)
        _verify_owner_file(os.fstat(lock_fd))
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        lock_path_stat = os.stat(
            ".tushare-calendar-capture.lock",
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if _stat_identity(lock_path_stat) != _stat_identity(os.fstat(lock_fd)):
            raise SystemSecurityError("trusted-provider capture lock path drifted")
        try:
            os.stat(root_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise SystemPreconditionError("trusted-provider capture destination exists")
        try:
            os.stat(
                _failure_root_name(root_name),
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise SystemPreconditionError("trusted-provider capture failure destination exists")
        os.mkdir(staging, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(
            staging,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        os.fchmod(staging_fd, 0o700)
        _verify_owner_directory(os.fstat(staging_fd), exact_mode=True)
        if os.fstat(staging_fd).st_dev != os.fstat(parent_fd).st_dev:
            raise SystemSecurityError("trusted-provider capture crosses a device")
        for leaf, raw in sorted(files.items()):
            _write_fd_leaf(staging_fd, leaf, raw)
        _read_directory_files(staging_fd, expected=files)
        os.fsync(staging_fd)
        staging_identity = _stat_identity(os.fstat(staging_fd))
        if _directory_identity(os.fstat(parent_fd)) != parent_identity:
            raise SystemSecurityError("trusted-provider capture parent drifted")
        _rename_no_replace(parent_fd, staging, root_name)
        published = True
        os.fsync(parent_fd)
        final_fd = os.open(
            root_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        if _stat_identity(os.fstat(final_fd)) != staging_identity:
            raise SystemSecurityError("trusted-provider capture directory inode drifted")
        _read_directory_files(final_fd, expected=files)
        success_raw = success_builder(_utc_now(), os.fstat(final_fd))
        if type(success_raw) is not bytes or not success_raw:
            raise SystemSecurityError("trusted-provider capture success bytes are invalid")
        _write_fd_leaf(final_fd, "capture-success.json", success_raw)
        os.fsync(final_fd)
        _read_directory_files(
            final_fd,
            expected={**files, "capture-success.json": success_raw},
        )
        os.fsync(parent_fd)
        return success_raw
    except BaseException as exc:
        try:
            setattr(exc, "_trusted_provider_capture_root_published", published)
        except BaseException:
            pass
        raise
    finally:
        if final_fd is not None:
            os.close(final_fd)
        if staging_fd is not None:
            os.close(staging_fd)
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        if not published:
            try:
                cleanup_fd = os.open(
                    staging,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    with os.scandir(cleanup_fd) as entries:
                        names = [entry.name for entry in entries]
                    for name in names:
                        os.unlink(name, dir_fd=cleanup_fd)
                finally:
                    os.close(cleanup_fd)
                os.rmdir(staging, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _publish_capture_failure_root(
    *,
    parent: Path,
    capture_root_name: str,
    failed_at: str,
    error_code: str,
    success_root_published: bool,
) -> dict[str, Any]:
    """Atomically publish one exact owner-only sibling failure root."""

    success_root_name = _capture_root_name(capture_root_name)
    failure_root_name = _failure_root_name(success_root_name)
    parent_fd = _open_pinned_absolute_directory(parent)
    parent_identity = _directory_identity(os.fstat(parent_fd))
    lock_fd: int | None = None
    staging_fd: int | None = None
    final_fd: int | None = None
    staging = f".{failure_root_name}.staging-{os.getpid()}-{secrets.token_hex(8)}"
    published = False
    try:
        lock_fd = _open_capture_lock(parent_fd)
        os.fchmod(lock_fd, 0o600)
        _verify_owner_file(os.fstat(lock_fd))
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        lock_path_stat = os.stat(
            ".tushare-calendar-capture.lock",
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if _stat_identity(lock_path_stat) != _stat_identity(os.fstat(lock_fd)):
            raise SystemSecurityError("trusted-provider capture lock path drifted")
        try:
            os.stat(failure_root_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise SystemPreconditionError("trusted-provider capture failure destination exists")
        os.mkdir(staging, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(
            staging,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        os.fchmod(staging_fd, 0o700)
        staging_stat = os.fstat(staging_fd)
        _verify_owner_directory(staging_stat, exact_mode=True)
        if staging_stat.st_dev != os.fstat(parent_fd).st_dev:
            raise SystemSecurityError("trusted-provider capture failure crosses a device")
        failure = _build_trusted_provider_calendar_capture_failure(
            capture_root_name=success_root_name,
            failed_at=failed_at,
            error_code=error_code,
            success_root_published=success_root_published,
            published_root_device=staging_stat.st_dev,
            published_root_inode=staging_stat.st_ino,
        )
        failure_raw = canonical_json_bytes(failure)
        failure_ref = _file_reference(failure_root_name, _FAILURE_LEAF, failure_raw)
        _write_fd_leaf(staging_fd, _FAILURE_LEAF, failure_raw)
        _read_directory_files(staging_fd, expected={_FAILURE_LEAF: failure_raw})
        os.fsync(staging_fd)
        staging_identity = _stat_identity(os.fstat(staging_fd))
        if _directory_identity(os.fstat(parent_fd)) != parent_identity:
            raise SystemSecurityError("trusted-provider capture parent drifted")
        _rename_no_replace(parent_fd, staging, failure_root_name)
        published = True
        os.fsync(parent_fd)
        final_fd = os.open(
            failure_root_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        final_stat = os.fstat(final_fd)
        _verify_owner_directory(final_stat, exact_mode=True)
        if _stat_identity(final_stat) != staging_identity:
            raise SystemSecurityError("trusted-provider capture failure directory inode drifted")
        _read_directory_files(final_fd, expected={_FAILURE_LEAF: failure_raw})
        os.fsync(final_fd)
        os.fsync(parent_fd)
        return {
            "status": "FAILED",
            "capture_failure_root": str(parent / failure_root_name),
            "capture_failure": failure,
            "capture_failure_file_ref": failure_ref,
        }
    finally:
        if final_fd is not None:
            os.close(final_fd)
        if staging_fd is not None:
            os.close(staging_fd)
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        if not published:
            try:
                cleanup_fd = os.open(
                    staging,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    with os.scandir(cleanup_fd) as entries:
                        names = [entry.name for entry in entries]
                    for name in names:
                        os.unlink(name, dir_fd=cleanup_fd)
                finally:
                    os.close(cleanup_fd)
                os.rmdir(staging, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def validate_published_trusted_provider_calendar_capture_failure_root(
    *,
    capture_parent: str | os.PathLike[str],
    capture_failure: Mapping[str, Any] | bytes,
    capture_failure_file_ref: Mapping[str, Any],
) -> bytes:
    """Replay the exact one-leaf failure topology without touching success."""

    failure = validate_trusted_provider_calendar_capture_failure(capture_failure)
    success_root_name = failure["payload"]["capture_root_name"]
    failure_root_name = _failure_root_name(success_root_name)
    reference = _file_ref(
        capture_failure_file_ref,
        label="capture_failure_file_ref",
    )
    if reference["relative_path"] != f"{failure_root_name}/{_FAILURE_LEAF}":
        raise SystemContractError("trusted-provider capture failure topology differs")
    expected_raw = canonical_json_bytes(failure)
    if reference["byte_sha256"] != _sha256(expected_raw):
        raise SystemSecurityError("trusted-provider capture failure SHA differs")
    parent_fd = _open_pinned_absolute_directory(Path(capture_parent))
    root_fd: int | None = None
    try:
        root_fd = os.open(
            failure_root_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        root_stat = os.fstat(root_fd)
        _verify_owner_directory(root_stat, exact_mode=True)
        if (
            failure["payload"]["published_root_device"] != root_stat.st_dev
            or failure["payload"]["published_root_inode"] != root_stat.st_ino
        ):
            raise SystemSecurityError("trusted-provider capture failure root identity differs")
        with os.scandir(root_fd) as entries:
            names = sorted(entry.name for entry in entries)
        if names != [_FAILURE_LEAF]:
            raise SystemSecurityError("trusted-provider capture failure directory topology differs")
        observed = _read_ref_leaf(
            root_fd,
            leaf=_FAILURE_LEAF,
            expected_sha256=reference["byte_sha256"],
        )
        if observed != expected_raw:
            raise SystemSecurityError("trusted-provider capture failure bytes differ")
        return observed
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _publish_trusted_provider_calendar_capture_failure(
    *,
    capture_parent: str | os.PathLike[str],
    capture_root_name: str,
    failed_at: str,
    error_code: str,
    success_root_published: bool,
) -> dict[str, Any]:
    """Build, publish, and read back one terminal capture failure."""

    parent = Path(capture_parent)
    result = _publish_capture_failure_root(
        parent=parent,
        capture_root_name=capture_root_name,
        failed_at=failed_at,
        error_code=error_code,
        success_root_published=success_root_published,
    )
    validate_published_trusted_provider_calendar_capture_failure_root(
        capture_parent=parent,
        capture_failure=result["capture_failure"],
        capture_failure_file_ref=result["capture_failure_file_ref"],
    )
    return result


def _controlled_capture_failure_code(exc: BaseException, *, phase: str) -> str:
    phase_code = _CAPTURE_FAILURE_PHASE_CODES.get(
        phase,
        "TRUSTED_PROVIDER_CALENDAR_CAPTURE_FAILED",
    )
    if isinstance(exc, TushareHttpsError) and exc.code in _TUSHARE_FAILURE_CODES:
        exchange = phase.removeprefix("PROVIDER_")
        if phase in {"PROVIDER_SSE", "PROVIDER_SZSE", "PROVIDER_BSE"}:
            return _failure_code(f"TRUSTED_PROVIDER_CALENDAR_{exchange}_{exc.code}")
        return _failure_code(exc.code)
    return _failure_code(phase_code)


def _raise_recorded_capture_failure(
    *,
    parent: Path,
    capture_root_name: str,
    phase: str,
    exc: BaseException,
    success_root_published: bool,
) -> NoReturn:
    code = _controlled_capture_failure_code(exc, phase=phase)
    result = _publish_trusted_provider_calendar_capture_failure(
        capture_parent=parent,
        capture_root_name=capture_root_name,
        failed_at=_utc_now(),
        error_code=code,
        success_root_published=success_root_published,
    )
    failure = result["capture_failure"]
    recorded = SystemPreconditionError(
        "trusted-provider calendar capture failed with immutable evidence",
        code=code,
    )
    recorded.public_fields = {
        "capture_failure_id": failure["payload"]["capture_failure_id"],
        "capture_failure_root": result["capture_failure_root"],
        "capture_failure_file_ref": result["capture_failure_file_ref"],
        "success_root_published": success_root_published,
    }
    raise recorded from exc


def capture_trusted_provider_calendar_evidence(  # noqa: C901
    *,
    capture_parent: str | os.PathLike[str],
    capture_root_name: str,
    cutoff_date: str,
    release_install_input_raw: bytes,
    expected_release_install_input_sha256: str,
    release_repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Capture through the fixed production operator and publish exact evidence."""

    parent = Path(capture_parent)
    if not parent.is_absolute():
        raise SystemSecurityError("trusted-provider capture parent must be absolute")
    root_name = _capture_root_name(capture_root_name)
    expected_input_sha = _sha(
        expected_release_install_input_sha256,
        label="expected_release_install_input_sha256",
    )
    if _sha256(release_install_input_raw) != expected_input_sha:
        raise SystemSecurityError("release install input SHA differs")
    observed_started_at = _utc_now()
    _release_install_components(
        release_install_input_raw,
        repository_root=release_repository_root,
        require_current_operator=True,
    )
    capture_start = RUNTIME_START_DATE - timedelta(days=CAPTURE_PREHISTORY_DAYS)
    cutoff = _date(cutoff_date, label="cutoff_date")
    if cutoff < RUNTIME_START_DATE:
        raise SystemPreconditionError("trusted-provider capture cutoff is invalid")
    phase = "DOCUMENTATION_FETCH"
    success_root_published = False
    try:
        docs_raw, docs_status, docs_headers, docs_tls, docs_redirects = (
            _official_documentation_fetch()
        )
        if type(docs_raw) is not bytes or not docs_raw or len(docs_raw) > 4 * 1024 * 1024:
            raise SystemSecurityError("trusted-provider documentation exceeds capture bound")
        provider = OfficialTushareHttpsClient(
            strict_decimal_decode=True,
            max_response_items=2_000,
            max_response_bytes=_CALENDAR_RESPONSE_MAX_BYTES,
        )
        raw_values: dict[str, bytes] = {}
        aggregate = len(docs_raw)
        responses: dict[str, Any] = {}
        for exchange in (*DIRECT_EXCHANGES, *PROBE_EXCHANGES):
            phase = f"PROVIDER_{exchange}"
            parameters = {
                "end_date": cutoff.strftime("%Y%m%d"),
                "exchange": exchange,
                "start_date": capture_start.strftime("%Y%m%d"),
            }
            response = provider.request(
                api_name=API_NAME,
                params=parameters,
                expected_fields=EXPECTED_FIELDS,
            )
            raw = response.raw_body
            if type(raw) is not bytes or not raw or len(raw) > _CALENDAR_RESPONSE_MAX_BYTES:
                raise SystemSecurityError("trusted-provider response exceeds capture bound")
            aggregate += len(raw)
            if aggregate > _CALENDAR_AGGREGATE_MAX_BYTES:
                raise SystemSecurityError("trusted-provider aggregate capture bound exceeded")
            raw_values[exchange] = raw
            responses[exchange] = parameters
        phase = "EVIDENCE_BUILD"
        captured_at = _utc_now()
        docs_ref = _file_reference(root_name, "documentation.raw", docs_raw)
        release_input_ref = _file_reference(
            root_name,
            "release-install-input.json",
            release_install_input_raw,
        )
        capability = build_trusted_provider_calendar_capability(
            docs_raw=docs_raw,
            docs_raw_file_ref=docs_ref,
            docs_captured_at=captured_at,
            docs_http_status=docs_status,
            docs_tls_verified=docs_tls,
            docs_redirect_chain=docs_redirects,
            docs_response_headers=docs_headers,
            created_at=captured_at,
        )
        capability_raw = canonical_json_bytes(capability)
        capability_ref = _file_reference(root_name, "capability.json", capability_raw)
        policy = build_calendar_authority_policy(
            created_at=captured_at,
            pit_exchange_ids=DEGRADED_PIT_EXCHANGES,
            provider_capability=capability,
        )
        policy_raw = canonical_json_bytes(policy)
        policy_ref = _file_reference(root_name, "policy.json", policy_raw)
        raw_refs: list[dict[str, str]] = []
        capture_refs: list[dict[str, str]] = []
        captures: dict[str, bytes] = {}
        for exchange in (*DIRECT_EXCHANGES, *PROBE_EXCHANGES):
            raw = raw_values[exchange]
            raw_ref = _file_reference(
                root_name,
                f"response-{exchange.lower()}.raw",
                raw,
            )
            capture = build_trusted_provider_calendar_capture(
                exchange_id=exchange,
                raw=raw,
                raw_file_ref=raw_ref,
                capability=capability,
                docs_raw=docs_raw,
                captured_at=captured_at,
                capture_start_date=capture_start.isoformat(),
                cutoff_date=cutoff.isoformat(),
                request_parameters_sanitized=responses[exchange],
                response_headers={},
                created_at=captured_at,
            )
            capture_raw = canonical_json_bytes(capture)
            capture_ref = _file_reference(
                root_name,
                f"capture-{exchange.lower()}.json",
                capture_raw,
            )
            raw_refs.append(raw_ref)
            capture_refs.append(capture_ref)
            captures[exchange] = capture_raw
        raw_refs.sort(key=lambda row: row["relative_path"])
        capture_refs.sort(key=lambda row: row["relative_path"])
        transaction = build_trusted_provider_calendar_capture_transaction(
            capture_root_name=root_name,
            capture_start_date=capture_start.isoformat(),
            cutoff_date=cutoff.isoformat(),
            captured_at=captured_at,
            documentation_raw_file_ref=docs_ref,
            capability_file_ref=capability_ref,
            policy_file_ref=policy_ref,
            provider_raw_file_refs=raw_refs,
            provider_capture_file_refs=capture_refs,
        )
        transaction_raw = canonical_json_bytes(transaction)
        transaction_ref = _file_reference(
            root_name,
            "capture-transaction.json",
            transaction_raw,
        )
        observed_completed_at = _utc_now()
        execution = _build_trusted_provider_calendar_capture_execution(
            capture_root_name=root_name,
            release_install_input_raw=release_install_input_raw,
            release_install_input_file_ref=release_input_ref,
            release_repository_root=release_repository_root,
            documentation_raw_file_ref=docs_ref,
            capability_file_ref=capability_ref,
            policy_file_ref=policy_ref,
            provider_raw_file_refs=raw_refs,
            provider_capture_file_refs=capture_refs,
            capture_transaction_file_ref=transaction_ref,
            observed_started_at=observed_started_at,
            observed_completed_at=observed_completed_at,
        )
        execution_raw = canonical_json_bytes(execution)
        execution_ref = _file_reference(root_name, "capture-execution.json", execution_raw)
        files = {
            "documentation.raw": docs_raw,
            "release-install-input.json": release_install_input_raw,
            "capability.json": capability_raw,
            "policy.json": policy_raw,
            "response-sse.raw": raw_values["SSE"],
            "response-szse.raw": raw_values["SZSE"],
            "response-bse.raw": raw_values["BSE"],
            "capture-sse.json": captures["SSE"],
            "capture-szse.json": captures["SZSE"],
            "capture-bse.json": captures["BSE"],
            "capture-transaction.json": transaction_raw,
            "capture-execution.json": execution_raw,
        }
        published_refs = sorted(
            (_file_reference(root_name, leaf, raw) for leaf, raw in files.items()),
            key=lambda row: row["relative_path"],
        )

        def success_builder(
            observed_completed_at: str,
            published_root_stat: os.stat_result,
        ) -> bytes:
            return canonical_json_bytes(
                _build_trusted_provider_calendar_capture_success(
                    capture_root_name=root_name,
                    capture_transaction_file_ref=transaction_ref,
                    capture_execution_file_ref=execution_ref,
                    published_leaf_file_refs=published_refs,
                    published_root_device=published_root_stat.st_dev,
                    published_root_inode=published_root_stat.st_ino,
                    observed_completed_at=observed_completed_at,
                )
            )

        phase = "PUBLICATION"
        success_raw = _publish_capture_tree(
            parent=parent,
            root_name=root_name,
            files=files,
            success_builder=success_builder,
        )
        success_root_published = True
        phase = "SUCCESS_VALIDATION"
        success = validate_trusted_provider_calendar_capture_success(
            success_raw,
            capture_transaction_file_ref=transaction_ref,
            capture_execution_file_ref=execution_ref,
            published_leaf_file_refs=published_refs,
        )
        success_ref = _file_reference(root_name, "capture-success.json", success_raw)
        return {
            "status": "CAPTURED",
            "capture_root": str(parent / root_name),
            "capture_transaction": transaction,
            "capture_transaction_file_ref": transaction_ref,
            "capture_execution": execution,
            "capture_execution_file_ref": execution_ref,
            "capture_success": success,
            "capture_success_file_ref": success_ref,
            "release_install_input_file_ref": release_input_ref,
            "calendar_authority_policy_file_ref": policy_ref,
            "trusted_provider_calendar_capability_file_ref": capability_ref,
            "trusted_provider_calendar_raw_file_refs": [docs_ref, *raw_refs],
            "trusted_provider_calendar_capture_file_refs": capture_refs,
            "network_call_count": 4,
        }
    except BaseException as exc:
        success_root_published = bool(
            success_root_published
            or getattr(exc, "_trusted_provider_capture_root_published", False)
        )
        _raise_recorded_capture_failure(
            parent=parent,
            capture_root_name=root_name,
            phase=phase,
            exc=exc,
            success_root_published=success_root_published,
        )


__all__ = [
    "AUTHORITY_ROUTE",
    "AUTHORITY_TIER",
    "CAPABILITY_KIND",
    "CAPTURE_EXECUTION_KIND",
    "CAPTURE_FAILURE_KIND",
    "CAPTURE_KIND",
    "CAPTURE_SUCCESS_KIND",
    "COMPILATION_KIND",
    "DIRECT_EXCHANGES",
    "DOCS_URL",
    "EXPECTED_FIELDS",
    "POLICY_KIND",
    "PROJECTED_EXCHANGES",
    "PROBE_EXCHANGES",
    "RUNTIME_START_DATE",
    "SOURCE_LIMITATIONS",
    "build_calendar_authority_policy",
    "build_trusted_provider_calendar_capability",
    "build_trusted_provider_calendar_capture",
    "build_trusted_provider_calendar_capture_transaction",
    "build_trusted_provider_calendar_compilation",
    "calendar_source_limitations",
    "capture_trusted_provider_calendar_evidence",
    "compiler_ast_sha256",
    "compiler_code_sha256",
    "decode_trade_cal_documentation",
    "validate_calendar_authority_policy",
    "validate_published_trusted_provider_calendar_capture_root",
    "validate_published_trusted_provider_calendar_capture_failure_root",
    "validate_trusted_provider_calendar_capability",
    "validate_trusted_provider_calendar_capture",
    "validate_trusted_provider_calendar_capture_execution",
    "validate_trusted_provider_calendar_capture_failure",
    "validate_trusted_provider_calendar_capture_success",
    "validate_trusted_provider_calendar_capture_transaction",
    "validate_trusted_provider_calendar_compilation",
]
