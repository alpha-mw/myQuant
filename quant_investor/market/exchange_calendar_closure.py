"""Deep official-source closure for the unified CN exchange calendar.

This module never accepts caller-authored calendar rules. It replays admitted
native issuer captures, derives category closure and precedence, and proves
the exact runtime JSON and strict Parquet bytes before they can enter a System
generation.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, cast
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    seal_artifact,
    validate_artifact,
)
from quant_investor.factors.governance.source import role_schema
from quant_investor.system.errors import (
    SystemContractError,
    SystemPreconditionError,
    SystemSecurityError,
)
from quant_investor.system.store import object_ref_for_artifact, validate_object_ref

from . import exchange_calendar_official as official

COMPILATION_KIND: Final = "system.exchange_calendar_compilation"
CAPTURE_KIND: Final = "system.exchange_calendar_capture"
INDEX_CLOSURE_KIND: Final = "system.exchange_calendar_index_closure"
ADMISSION_KIND: Final = "system.exchange_calendar_decoder_admission"
COMPILER_RELATIVE_PATH: Final = "quant_investor/market/exchange_calendar_closure.py"
TIMEZONE: Final = "Asia/Shanghai"
EXCHANGE_ISSUERS: Final = {
    "SSE": "SSE_OFFICIAL",
    "SZSE": "SZSE_OFFICIAL",
    "BSE": "BSE_OFFICIAL",
}
EXCHANGE_HOSTS: Final = {
    "SSE": "www.sse.com.cn",
    "SZSE": "www.szse.cn",
    "BSE": "www.bse.cn",
}
PRECEDENCE_RULES: Final = [
    "SESSION_CHANGE_NOTICE",
    "TEMPORARY_CLOSURE_NOTICE",
    "ANNUAL_HOLIDAY_NOTICE",
    "SESSION_RULE",
    "TRADING_WEEK_RULE",
]
BODY_ROLES: Final = frozenset(
    {"ANNUAL_HOLIDAY_NOTICE", "TEMPORARY_CLOSURE_NOTICE", "SESSION_CHANGE_NOTICE"}
)
DIRECT_ROLES: Final = frozenset({"TRADING_WEEK_RULE", "SESSION_RULE"})
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RESPONSE_HEADERS: Final = frozenset(
    {"cache-control", "content-length", "content-type", "etag", "last-modified"}
)
_CAPTURE_FIELDS: Final = frozenset(
    {
        "calendar_capture_id",
        "state",
        "evidence_role",
        "exchange_id",
        "issuer",
        "request_url",
        "effective_url",
        "redirect_chain",
        "request_headers",
        "response_headers",
        "http_status",
        "tls_verified",
        "captured_at",
        "raw_file_ref",
        "raw_sha256",
        "raw_byte_length",
        "raw_media_type",
        "decoder_admission_ref",
        "decoder_id",
        "decoder_sha256",
        "projection_sha256",
    }
)
_INDEX_FIELDS: Final = frozenset(
    {
        "index_closure_id",
        "state",
        "exchange_id",
        "issuer",
        "category",
        "root_capture_ref",
        "page_capture_refs",
        "reported_page_count",
        "reported_item_count",
        "observed_item_count",
        "window_start_date",
        "window_end_date",
        "entry_rows",
        "body_capture_refs",
        "pagination_complete",
        "date_window_complete",
        "unknown_relevant_count",
    }
)
_INDEX_ENTRY_FIELDS: Final = frozenset(
    {"entry_id", "publish_date", "title", "body_url", "relevant", "evidence_role"}
)
_RULE_FIELDS: Final = frozenset({"start_date", "end_date", "weekdays"})
_SESSION_RULE_FIELDS: Final = frozenset({"start_date", "end_date", "session_intervals"})
_SESSION_FIELDS: Final = frozenset({"phase", "opens_local", "closes_local"})
_COMPILATION_FIELDS: Final = frozenset(
    {
        "compilation_id",
        "state",
        "coverage_start_date",
        "cutoff_date",
        "timezone",
        "compiler_relative_path",
        "compiler_code_sha256",
        "compiler_ast_sha256",
        "release_ref",
        "pit_exchange_ids",
        "market_session_dates_sha256",
        "source_exchange_rows",
        "source_capture_refs",
        "decoder_admission_refs",
        "index_closure_refs",
        "precedence_rules",
        "exchange_rows",
        "runtime_projection",
        "calendar_json_file_ref",
        "calendar_parquet_file_ref",
        "contradiction_rows",
    }
)

RawResolver = Callable[[Mapping[str, Any]], bytes]
ArtifactResolver = Callable[[Mapping[str, Any]], Mapping[str, Any]]
AdmissionResolver = Callable[[str, official.EvidenceRole], Mapping[str, Any]]
DecoderIdResolver = Callable[[str, official.EvidenceRole], str]
ProjectionDecoder = Callable[..., Mapping[str, object]]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def compiler_code_sha256() -> str:
    return _sha256(Path(__file__).read_bytes())


def compiler_ast_sha256() -> str:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=COMPILER_RELATIVE_PATH)
    return _sha256(ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8"))


def _identifier(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or len(value) > 300
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


def _time(value: Any, *, label: str) -> time:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not a local time")
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not a local time") from exc
    if parsed.strftime("%H:%M:%S") != value:
        raise SystemContractError(f"{label} is not a second-precision local time")
    return parsed


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not canonical UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} is not canonical UTC")
    return value


def _file_ref(value: Any, *, label: str, with_size: bool = False) -> dict[str, Any]:
    fields = (
        {"relative_path", "byte_sha256", "size"}
        if with_size
        else {
            "relative_path",
            "byte_sha256",
        }
    )
    if type(value) is not dict or set(value) != fields:
        raise SystemContractError(f"{label} fields are not exact")
    relative = value["relative_path"]
    path = PurePosixPath(relative) if type(relative) is str else None
    if (
        path is None
        or path.is_absolute()
        or str(path) != relative
        or any(part in {"", ".", ".."} for part in path.parts)
        or type(value["byte_sha256"]) is not str
        or _SHA_RE.fullmatch(value["byte_sha256"]) is None
        or (with_size and (type(value["size"]) is not int or value["size"] <= 0))
    ):
        raise SystemContractError(f"{label} is invalid")
    return dict(value)


def _object_refs(
    value: Any, *, label: str, expected_kind: str, minimum: int = 1
) -> list[dict[str, str]]:
    if type(value) is not list or len(value) < minimum:
        raise SystemContractError(f"{label} is incomplete")
    refs = [validate_object_ref(row, label=f"{label}[{index}]") for index, row in enumerate(value)]
    if any(row["kind"] != expected_kind for row in refs):
        raise SystemContractError(f"{label} contains the wrong authority kind")
    encoded = [canonical_json_bytes(row) for row in refs]
    if encoded != sorted(encoded) or len(encoded) != len(set(encoded)):
        raise SystemContractError(f"{label} is not canonical sorted unique")
    return refs


def _official_url(value: Any, *, exchange: str, label: str) -> str:
    url = _identifier(value, label=label)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != EXCHANGE_HOSTS[exchange]
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise SystemSecurityError(f"{label} is outside the official issuer authority")
    return url


def _header_rows(value: Any, *, label: str) -> list[dict[str, str]]:
    if type(value) is not list:
        raise SystemContractError(f"{label} is not a list")
    rows: list[dict[str, str]] = []
    for index, row in enumerate(value):
        if (
            type(row) is not dict
            or set(row) != {"name", "value"}
            or type(row["name"]) is not str
            or row["name"] != row["name"].lower()
            or row["name"] not in _SAFE_RESPONSE_HEADERS
            or type(row["value"]) is not str
            or row["value"].strip() != row["value"]
        ):
            raise SystemSecurityError(f"{label}[{index}] is not safe retained metadata")
        rows.append(dict(row))
    if rows != sorted(rows, key=lambda item: (item["name"], item["value"])):
        raise SystemContractError(f"{label} is not canonical")
    return rows


def _endpoint_matches(template: str, request_url: str) -> bool:
    parsed = urlsplit(request_url)
    target = parsed.path + (("?" + parsed.query) if parsed.query else "")
    pattern = re.escape(template)
    pattern = re.sub(r"\\\{[A-Za-z_][A-Za-z0-9_]*\\\}", r"[^&/?#]+", pattern)
    return re.fullmatch(pattern, target) is not None


def _default_decoder(
    exchange: str,
    role: official.EvidenceRole,
    raw: bytes,
    *,
    media_type: str,
) -> Mapping[str, object]:
    return official.decode_capture_projection(exchange, role, raw, media_type=media_type)


def _projection(  # noqa: C901
    *,
    exchange: str,
    role: str,
    raw: bytes,
    media_type: str,
    decoder: ProjectionDecoder,
) -> dict[str, Any]:
    if role not in official.EVIDENCE_ROLES:
        raise SystemContractError("calendar capture role is not compiled")
    result = dict(
        decoder(
            exchange,
            cast(official.EvidenceRole, role),
            raw,
            media_type=media_type,
        )
    )
    if role == "TRADING_WEEK_RULE":
        if set(result) != {"weekly_rule_intervals"}:
            raise SystemContractError("weekly-rule decoder projection fields differ")
        intervals: list[dict[str, Any]] = []
        value = result["weekly_rule_intervals"]
        if type(value) is not list:
            raise SystemContractError("weekly-rule intervals are not a list")
        for index, item in enumerate(value):
            if type(item) is not dict or set(item) != _RULE_FIELDS:
                raise SystemContractError("weekly-rule interval fields differ")
            start = _date(item["start_date"], label=f"weekly[{index}].start_date")
            end = _date(item["end_date"], label=f"weekly[{index}].end_date")
            weekdays = item["weekdays"]
            if (
                start > end
                or type(weekdays) is not list
                or weekdays != sorted(set(weekdays))
                or any(type(day) is not int or not 1 <= day <= 7 for day in weekdays)
            ):
                raise SystemContractError("weekly-rule interval is invalid")
            intervals.append(
                {
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "weekdays": list(weekdays),
                }
            )
        if not intervals:
            raise SystemPreconditionError("weekly-rule evidence is empty")
        return {"weekly_rule_intervals": intervals}
    if role in {"SESSION_RULE", "SESSION_CHANGE_NOTICE"}:
        if set(result) != {"session_rule_intervals"}:
            raise SystemContractError("session-rule decoder projection fields differ")
        value = result["session_rule_intervals"]
        if type(value) is not list:
            raise SystemContractError("session-rule intervals are not a list")
        rules: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if type(item) is not dict or set(item) != _SESSION_RULE_FIELDS:
                raise SystemContractError("session-rule interval fields differ")
            start = _date(item["start_date"], label=f"session_rule[{index}].start_date")
            end = _date(item["end_date"], label=f"session_rule[{index}].end_date")
            raw_sessions = item["session_intervals"]
            if type(raw_sessions) is not list:
                raise SystemContractError("session-rule sessions are not a list")
            sessions: list[dict[str, str]] = []
            for ordinal, session in enumerate(raw_sessions):
                if type(session) is not dict or set(session) != _SESSION_FIELDS:
                    raise SystemContractError("session interval fields differ")
                opens = _time(
                    session["opens_local"],
                    label=f"session[{index}][{ordinal}].opens_local",
                )
                closes = _time(
                    session["closes_local"],
                    label=f"session[{index}][{ordinal}].closes_local",
                )
                if opens >= closes:
                    raise SystemContractError("session interval is inverted")
                sessions.append(
                    {
                        "phase": _identifier(session["phase"], label="session.phase"),
                        "opens_local": opens.strftime("%H:%M:%S"),
                        "closes_local": closes.strftime("%H:%M:%S"),
                    }
                )
            if sessions != sorted(sessions, key=lambda row: (row["opens_local"], row["phase"])):
                raise SystemContractError("session intervals are not canonical")
            if not any(
                row["opens_local"] == "09:30:00" and row["closes_local"] == "11:30:00"
                for row in sessions
            ) or not any(
                row["opens_local"] == "13:00:00" and row["closes_local"] == "15:00:00"
                for row in sessions
            ):
                raise SystemPreconditionError("continuous-auction session evidence is incomplete")
            rules.append(
                {
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "session_intervals": sessions,
                }
            )
        if not rules:
            raise SystemPreconditionError("session-rule evidence is empty")
        return {"session_rule_intervals": rules}
    if role in {"ANNUAL_HOLIDAY_NOTICE", "TEMPORARY_CLOSURE_NOTICE"}:
        if set(result) != {"closure_dates"} or type(result["closure_dates"]) is not list:
            raise SystemContractError("closure decoder projection fields differ")
        rows = [
            _date(item, label=f"closure_dates[{index}]").isoformat()
            for index, item in enumerate(result["closure_dates"])
        ]
        if rows != sorted(set(rows)):
            raise SystemContractError("closure dates are not canonical")
        return {"closure_dates": rows}
    if role == "NOTICE_INDEX_SNAPSHOT":
        expected = {
            "category",
            "page_number",
            "page_count",
            "reported_item_count",
            "window_start_date",
            "window_end_date",
            "entries",
        }
        if set(result) != expected:
            raise SystemContractError("notice-index decoder projection fields differ")
        page = result["page_number"]
        count = result["page_count"]
        total = result["reported_item_count"]
        if (
            type(page) is not int
            or type(count) is not int
            or type(total) is not int
            or not 1 <= page <= count
            or total < 0
            or type(result["entries"]) is not list
        ):
            raise SystemContractError("notice-index pagination fields differ")
        entries: list[dict[str, Any]] = []
        for index, item in enumerate(result["entries"]):
            if type(item) is not dict or set(item) != _INDEX_ENTRY_FIELDS:
                raise SystemContractError("notice-index entry fields differ")
            relevant = item["relevant"]
            evidence_role = item["evidence_role"]
            if (
                type(relevant) is not bool
                or (evidence_role is not None and evidence_role not in BODY_ROLES)
                or (not relevant and evidence_role is not None)
            ):
                raise SystemContractError("notice-index relevance fields differ")
            entries.append(
                {
                    "entry_id": _identifier(item["entry_id"], label=f"entry[{index}].entry_id"),
                    "publish_date": _date(
                        item["publish_date"], label=f"entry[{index}].publish_date"
                    ).isoformat(),
                    "title": _identifier(item["title"], label=f"entry[{index}].title"),
                    "body_url": _official_url(
                        item["body_url"],
                        exchange=exchange,
                        label=f"entry[{index}].body_url",
                    ),
                    "relevant": relevant,
                    "evidence_role": evidence_role,
                }
            )
        if entries != sorted(entries, key=lambda row: (row["publish_date"], row["entry_id"])):
            raise SystemContractError("notice-index entries are not canonical")
        return {
            "category": _identifier(result["category"], label="notice category"),
            "page_number": page,
            "page_count": count,
            "reported_item_count": total,
            "window_start_date": _date(
                result["window_start_date"], label="index.window_start_date"
            ).isoformat(),
            "window_end_date": _date(
                result["window_end_date"], label="index.window_end_date"
            ).isoformat(),
            "entries": entries,
        }
    raise SystemContractError("calendar capture role is unsupported")


def _validate_capture(  # noqa: C901
    document: Mapping[str, Any],
    *,
    raw_resolver: RawResolver,
    artifact_resolver: ArtifactResolver,
    decoder: ProjectionDecoder,
    admission_resolver: AdmissionResolver,
    decoder_id_resolver: DecoderIdResolver,
    decoder_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        capture = validate_artifact(document, expected_kind=CAPTURE_KIND)
    except ContractError as exc:
        raise SystemContractError("official calendar capture contract failed") from exc
    payload = capture["payload"]
    if set(payload) != _CAPTURE_FIELDS or payload["state"] != "IMMUTABLE":
        raise SystemContractError("official calendar capture fields differ")
    exchange = payload["exchange_id"]
    role = payload["evidence_role"]
    if (
        exchange not in EXCHANGE_ISSUERS
        or role not in official.EVIDENCE_ROLES
        or payload["issuer"] != EXCHANGE_ISSUERS[exchange]
    ):
        raise SystemContractError("official calendar capture authority differs")
    _identifier(payload["calendar_capture_id"], label="calendar_capture_id")
    _timestamp(payload["captured_at"], label="capture.captured_at")
    request_url = _official_url(payload["request_url"], exchange=exchange, label="request_url")
    effective_url = _official_url(
        payload["effective_url"], exchange=exchange, label="effective_url"
    )
    redirects = payload["redirect_chain"]
    if type(redirects) is not list or any(type(item) is not str for item in redirects):
        raise SystemContractError("official calendar redirect chain differs")
    for index, url in enumerate(redirects):
        _official_url(url, exchange=exchange, label=f"redirect_chain[{index}]")
    if (redirects and redirects[-1] != effective_url) or (
        not redirects and effective_url != request_url
    ):
        raise SystemContractError("official calendar redirect chain is incomplete")
    if payload["request_headers"] != []:
        raise SystemSecurityError("official calendar capture used request headers")
    response_headers = _header_rows(payload["response_headers"], label="response_headers")
    if payload["http_status"] != 200 or payload["tls_verified"] is not True:
        raise SystemPreconditionError("official calendar HTTP/TLS capture failed")
    raw_ref = _file_ref(payload["raw_file_ref"], label="capture.raw_file_ref")
    raw = raw_resolver(raw_ref)
    if (
        type(raw) is not bytes
        or not raw
        or _sha256(raw) != raw_ref["byte_sha256"]
        or payload["raw_sha256"] != raw_ref["byte_sha256"]
        or payload["raw_byte_length"] != len(raw)
    ):
        raise SystemSecurityError("official calendar raw bytes differ")
    media_type = _identifier(payload["raw_media_type"], label="raw_media_type")
    if not any(row == {"name": "content-type", "value": media_type} for row in response_headers):
        raise SystemContractError("official calendar response content type differs")

    admission_ref = validate_object_ref(
        payload["decoder_admission_ref"], label="capture.decoder_admission_ref"
    )
    if admission_ref["kind"] != ADMISSION_KIND:
        raise SystemContractError("calendar capture decoder admission kind differs")
    supplied = official.validate_decoder_admission(artifact_resolver(admission_ref))
    registered = official.validate_decoder_admission(
        admission_resolver(exchange, cast(official.EvidenceRole, role))
    )
    if canonical_json_bytes(supplied) != canonical_json_bytes(registered):
        raise SystemSecurityError("calendar decoder admission is not registered")
    if object_ref_for_artifact(supplied) != admission_ref:
        raise SystemContractError("calendar decoder admission ref differs")
    admission_payload = supplied["payload"]
    fixture_ref = _file_ref(
        admission_payload["fixture_raw_file_ref"],
        label="decoder.fixture_raw_file_ref",
        with_size=True,
    )
    fixture = raw_resolver(fixture_ref)
    fixture_projection = _projection(
        exchange=exchange,
        role=role,
        raw=fixture,
        media_type=admission_payload["raw_media_type"],
        decoder=decoder,
    )
    if (
        len(fixture) != fixture_ref["size"]
        or _sha256(fixture) != fixture_ref["byte_sha256"]
        or admission_payload["fixture_projection_sha256"]
        != _sha256(canonical_json_bytes(fixture_projection))
        or admission_payload["exchange_id"] != exchange
        or admission_payload["evidence_role"] != role
        or admission_payload["issuer"] != payload["issuer"]
        or admission_payload["decoder_id"]
        != decoder_id_resolver(exchange, cast(official.EvidenceRole, role))
        or admission_payload["decoder_sha256"] != decoder_sha256
        or payload["decoder_id"] != admission_payload["decoder_id"]
        or payload["decoder_sha256"] != decoder_sha256
        or payload["http_status"] != admission_payload["http_status"]
        or media_type != admission_payload["raw_media_type"]
        or not _endpoint_matches(admission_payload["endpoint_path_query_template"], request_url)
    ):
        raise SystemSecurityError("calendar capture/admission binding differs")
    projection = _projection(
        exchange=exchange,
        role=role,
        raw=raw,
        media_type=media_type,
        decoder=decoder,
    )
    if payload["projection_sha256"] != _sha256(canonical_json_bytes(projection)):
        raise SystemSecurityError("calendar capture projection binding differs")
    return capture, projection


def _resolve_capture_map(
    capture_documents: Sequence[Mapping[str, Any]],
    **validation: Any,
) -> dict[bytes, tuple[dict[str, Any], dict[str, Any]]]:
    result: dict[bytes, tuple[dict[str, Any], dict[str, Any]]] = {}
    for document in capture_documents:
        capture, projection = _validate_capture(document, **validation)
        key = canonical_json_bytes(object_ref_for_artifact(capture))
        if key in result:
            raise SystemContractError("calendar capture authority is duplicated")
        result[key] = (capture, projection)
    return result


def _capture_for_ref(
    capture_map: Mapping[bytes, tuple[dict[str, Any], dict[str, Any]]],
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = validate_object_ref(value, label=label)
    if ref["kind"] != CAPTURE_KIND:
        raise SystemContractError(f"{label} has the wrong kind")
    try:
        return capture_map[canonical_json_bytes(ref)]
    except KeyError as exc:
        raise SystemContractError(f"{label} is outside the sealed capture closure") from exc


def _derive_index_closure(  # noqa: C901
    document: Mapping[str, Any],
    *,
    coverage: date,
    cutoff: date,
    capture_map: Mapping[bytes, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    try:
        supplied = validate_artifact(document, expected_kind=INDEX_CLOSURE_KIND)
    except ContractError as exc:
        raise SystemContractError("calendar index closure contract failed") from exc
    supplied_payload = supplied["payload"]
    if set(supplied_payload) != _INDEX_FIELDS:
        raise SystemContractError("calendar index closure fields differ")
    exchange = supplied_payload["exchange_id"]
    if exchange not in EXCHANGE_ISSUERS:
        raise SystemContractError("calendar index exchange differs")
    page_refs = _object_refs(
        supplied_payload["page_capture_refs"],
        label="page_capture_refs",
        expected_kind=CAPTURE_KIND,
    )
    body_refs = _object_refs(
        supplied_payload["body_capture_refs"],
        label="body_capture_refs",
        expected_kind=CAPTURE_KIND,
        minimum=0,
    )
    root_ref = validate_object_ref(supplied_payload["root_capture_ref"], label="root_capture_ref")
    pages: list[tuple[dict[str, str], dict[str, Any], dict[str, Any]]] = []
    for ref in page_refs:
        capture, projection = _capture_for_ref(
            capture_map, ref, label="calendar index page capture"
        )
        payload = capture["payload"]
        if (
            payload["exchange_id"] != exchange
            or payload["evidence_role"] != "NOTICE_INDEX_SNAPSHOT"
        ):
            raise SystemContractError("calendar index page subject differs")
        pages.append((ref, capture, projection))
    pages.sort(key=lambda item: item[2]["page_number"])
    if [row[2]["page_number"] for row in pages] != list(range(1, len(pages) + 1)):
        raise SystemPreconditionError("calendar index pagination is incomplete")
    first = pages[0][2]
    if (
        root_ref != pages[0][0]
        or first["page_count"] != len(pages)
        or any(
            row[2]["page_count"] != len(pages)
            or row[2]["reported_item_count"] != first["reported_item_count"]
            or row[2]["category"] != first["category"]
            or row[2]["window_start_date"] != first["window_start_date"]
            or row[2]["window_end_date"] != first["window_end_date"]
            for row in pages
        )
    ):
        raise SystemPreconditionError("calendar index pagination metadata differs")
    entries = [entry for _, _, page in pages for entry in page["entries"]]
    entries.sort(key=lambda row: (row["publish_date"], row["entry_id"]))
    if len(entries) != first["reported_item_count"] or len(
        {row["entry_id"] for row in entries}
    ) != len(entries):
        raise SystemPreconditionError("calendar index item closure differs")
    window_start = _date(first["window_start_date"], label="index.window_start_date")
    window_end = _date(first["window_end_date"], label="index.window_end_date")
    date_window_complete = window_start <= coverage and window_end >= cutoff
    relevant = [row for row in entries if row["relevant"]]
    unknown_count = sum(row["evidence_role"] is None for row in relevant)
    body_by_subject: dict[tuple[str, str], dict[str, str]] = {}
    for ref in body_refs:
        capture, _ = _capture_for_ref(capture_map, ref, label="calendar index body capture")
        payload = capture["payload"]
        key = (payload["effective_url"], payload["evidence_role"])
        if payload["exchange_id"] != exchange or payload["evidence_role"] not in BODY_ROLES:
            raise SystemContractError("calendar index body subject differs")
        if key in body_by_subject:
            raise SystemContractError("calendar index body subject is duplicated")
        body_by_subject[key] = ref
    expected_subjects = {
        (row["body_url"], row["evidence_role"])
        for row in relevant
        if row["evidence_role"] is not None
    }
    if set(body_by_subject) != expected_subjects:
        raise SystemPreconditionError("calendar index relevant-body closure differs")
    payload = {
        "index_closure_id": _identifier(
            supplied_payload["index_closure_id"], label="index_closure_id"
        ),
        "state": "COMPLETE" if unknown_count == 0 and date_window_complete else "BLOCKED",
        "exchange_id": exchange,
        "issuer": EXCHANGE_ISSUERS[exchange],
        "category": first["category"],
        "root_capture_ref": root_ref,
        "page_capture_refs": page_refs,
        "reported_page_count": len(pages),
        "reported_item_count": first["reported_item_count"],
        "observed_item_count": len(entries),
        "window_start_date": window_start.isoformat(),
        "window_end_date": window_end.isoformat(),
        "entry_rows": entries,
        "body_capture_refs": body_refs,
        "pagination_complete": True,
        "date_window_complete": date_window_complete,
        "unknown_relevant_count": unknown_count,
    }
    rebuilt = seal_artifact(INDEX_CLOSURE_KIND, payload, created_at=supplied["created_at"])
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(supplied):
        raise SystemSecurityError("calendar index closure is not decoder-derived")
    if payload["state"] != "COMPLETE":
        raise SystemPreconditionError("calendar index/category closure is incomplete")
    return rebuilt


def _interval_cover(
    rows: Sequence[Mapping[str, Any]], *, coverage: date, cutoff: date, label: str
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["start_date"])
    previous: date | None = None
    for row in ordered:
        start = _date(row["start_date"], label=f"{label}.start_date")
        end = _date(row["end_date"], label=f"{label}.end_date")
        if start > end or (previous is not None and start != previous + timedelta(days=1)):
            raise SystemPreconditionError(f"{label} intervals overlap or contain a gap")
        previous = end
    if (
        not ordered
        or ordered[0]["start_date"] > coverage.isoformat()
        or ordered[-1]["end_date"] < cutoff.isoformat()
    ):
        raise SystemPreconditionError(f"{label} intervals do not cover the compilation window")
    return ordered


def _derive_exchange_source(  # noqa: C901
    *,
    exchange: str,
    coverage: date,
    cutoff: date,
    captures: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    index_closures: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    indexed_refs = {
        canonical_json_bytes(ref)
        for closure in index_closures
        for ref in closure["payload"]["body_capture_refs"]
    }
    for capture, _ in captures:
        role = capture["payload"]["evidence_role"]
        if (
            role in BODY_ROLES
            and canonical_json_bytes(object_ref_for_artifact(capture)) not in indexed_refs
        ):
            raise SystemPreconditionError("calendar notice body is outside index closure")
    weekly_rows: list[dict[str, Any]] = []
    session_base: list[dict[str, Any]] = []
    session_changes: list[dict[str, Any]] = []
    closure_sources: dict[str, list[dict[str, str]]] = {}
    for capture, projection in captures:
        role = capture["payload"]["evidence_role"]
        ref = object_ref_for_artifact(capture)
        if role == "TRADING_WEEK_RULE":
            weekly_rows.extend(
                {**row, "source_capture_ref": ref} for row in projection["weekly_rule_intervals"]
            )
        elif role == "SESSION_RULE":
            session_base.extend(
                {**row, "source_capture_refs": [ref]}
                for row in projection["session_rule_intervals"]
            )
        elif role == "SESSION_CHANGE_NOTICE":
            session_changes.extend(
                {**row, "source_capture_refs": [ref]}
                for row in projection["session_rule_intervals"]
            )
        elif role in {"ANNUAL_HOLIDAY_NOTICE", "TEMPORARY_CLOSURE_NOTICE"}:
            for value in projection["closure_dates"]:
                closure_sources.setdefault(value, []).append(ref)
    weekly = _interval_cover(
        weekly_rows, coverage=coverage, cutoff=cutoff, label=f"{exchange} weekly rule"
    )
    base = _interval_cover(
        session_base, coverage=coverage, cutoff=cutoff, label=f"{exchange} session rule"
    )
    effective_daily: list[dict[str, Any]] = []
    cursor = coverage
    while cursor <= cutoff:
        text = cursor.isoformat()
        base_matches = [row for row in base if row["start_date"] <= text <= row["end_date"]]
        changes = [row for row in session_changes if row["start_date"] <= text <= row["end_date"]]
        if len(base_matches) != 1 or len(changes) > 1:
            raise SystemPreconditionError("calendar session-rule applicability is ambiguous")
        selected = changes[0] if changes else base_matches[0]
        effective_daily.append(
            {
                "date": text,
                "session_intervals": selected["session_intervals"],
                "source_capture_refs": selected["source_capture_refs"],
            }
        )
        cursor += timedelta(days=1)
    effective: list[dict[str, Any]] = []
    for row in effective_daily:
        if effective and (
            effective[-1]["session_intervals"] == row["session_intervals"]
            and effective[-1]["source_capture_refs"] == row["source_capture_refs"]
        ):
            effective[-1]["end_date"] = row["date"]
        else:
            effective.append(
                {
                    "start_date": row["date"],
                    "end_date": row["date"],
                    "session_intervals": row["session_intervals"],
                    "source_capture_refs": row["source_capture_refs"],
                }
            )
    closures = [
        {"date": value, "source_capture_refs": sorted(refs, key=canonical_json_bytes)}
        for value, refs in sorted(closure_sources.items())
    ]
    return {
        "exchange_id": exchange,
        "issuer": EXCHANGE_ISSUERS[exchange],
        "weekly_rule_intervals": weekly,
        "closure_dates": closures,
        "session_rule_intervals": effective,
        "native_capture_refs": sorted(
            (object_ref_for_artifact(row[0]) for row in captures),
            key=canonical_json_bytes,
        ),
        "decoder_admission_refs": sorted(
            {
                canonical_json_bytes(row[0]["payload"]["decoder_admission_ref"]): row[0]["payload"][
                    "decoder_admission_ref"
                ]
                for row in captures
            }.values(),
            key=canonical_json_bytes,
        ),
        "index_closure_refs": sorted(
            (object_ref_for_artifact(row) for row in index_closures),
            key=canonical_json_bytes,
        ),
    }


def _compile_exchange_row(
    source: Mapping[str, Any], *, coverage: date, cutoff: date
) -> dict[str, Any]:
    closure_map = {row["date"]: row["source_capture_refs"] for row in source["closure_dates"]}
    zone = ZoneInfo(TIMEZONE)
    daily: list[dict[str, Any]] = []
    cursor = coverage
    while cursor <= cutoff:
        text = cursor.isoformat()
        weeks = [
            row
            for row in source["weekly_rule_intervals"]
            if row["start_date"] <= text <= row["end_date"]
        ]
        sessions = [
            row
            for row in source["session_rule_intervals"]
            if row["start_date"] <= text <= row["end_date"]
        ]
        if len(weeks) != 1 or len(sessions) != 1:
            raise SystemPreconditionError("calendar rule applicability is ambiguous")
        is_open = cursor.isoweekday() in weeks[0]["weekdays"] and text not in closure_map
        provenance = [weeks[0]["source_capture_ref"]]
        if text in closure_map:
            provenance.extend(closure_map[text])
        else:
            provenance.extend(sessions[0]["source_capture_refs"])
        provenance = sorted(
            {canonical_json_bytes(ref): ref for ref in provenance}.values(),
            key=canonical_json_bytes,
        )
        row: dict[str, Any] = {
            "date": text,
            "status": "OPEN" if is_open else "CLOSED",
            "provenance_refs": provenance,
        }
        if is_open:
            row["opens_at_utc"] = (
                datetime.combine(cursor, time(9, 30), tzinfo=zone)
                .astimezone(timezone.utc)
                .isoformat()
            )
            row["closes_at_utc"] = (
                datetime.combine(cursor, time(15, 0), tzinfo=zone)
                .astimezone(timezone.utc)
                .isoformat()
            )
        else:
            row["opens_at_utc"] = None
            row["closes_at_utc"] = None
        daily.append(row)
        cursor += timedelta(days=1)
    open_dates = [row["date"] for row in daily if row["status"] == "OPEN"]
    if len(open_dates) < 391:
        raise SystemPreconditionError("exchange calendar has fewer than 391 open sessions")
    return {
        "exchange_id": source["exchange_id"],
        "issuer": source["issuer"],
        "daily_rows": daily,
        "open_session_count": len(open_dates),
        "open_session_sha256": _sha256(canonical_json_bytes(open_dates)),
        "session_rule_intervals": source["session_rule_intervals"],
    }


def _runtime_without_provenance(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "date": row["date"],
            "status": row["status"],
            "opens_at_utc": row["opens_at_utc"],
            "closes_at_utc": row["closes_at_utc"],
        }
        for row in rows
    ]


def runtime_json_bytes(runtime_projection: Sequence[Mapping[str, Any]]) -> bytes:
    return canonical_json_bytes(
        {
            "domain": "myquant-unified-exchange-calendar-runtime",
            "schema_version": "system.exchange-calendar-runtime",
            "timezone": TIMEZONE,
            "rows": [dict(row) for row in runtime_projection],
        }
    )


def runtime_parquet_bytes(runtime_projection: Sequence[Mapping[str, Any]]) -> bytes:
    rows = [row for row in runtime_projection if row["status"] == "OPEN"]
    arrow_rows = [
        {
            "ordinal": ordinal,
            "open_session": date.fromisoformat(row["date"]),
            "opens_at_utc": datetime.fromisoformat(row["opens_at_utc"]),
            "closes_at_utc": datetime.fromisoformat(row["closes_at_utc"]),
        }
        for ordinal, row in enumerate(rows)
    ]
    table = pa.Table.from_pylist(arrow_rows, schema=role_schema("exchange_calendar"))
    sink = BytesIO()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )
    return sink.getvalue()


def build_exchange_calendar_compilation(  # noqa: C901
    *,
    compilation_id: str,
    coverage_start_date: str,
    cutoff_date: str,
    release_ref: Mapping[str, Any],
    pit_exchange_ids: Sequence[str],
    market_session_dates: Sequence[str],
    capture_documents: Sequence[Mapping[str, Any]],
    admission_documents: Sequence[Mapping[str, Any]],
    index_closure_documents: Sequence[Mapping[str, Any]],
    raw_resolver: RawResolver,
    calendar_json_file_ref: Mapping[str, Any],
    calendar_parquet_file_ref: Mapping[str, Any],
    created_at: str,
    decoder: ProjectionDecoder = _default_decoder,
    admission_resolver: AdmissionResolver = official.decoder_admission,
    decoder_id_resolver: DecoderIdResolver = official.decoder_id,
    decoder_sha256: str | None = None,
) -> dict[str, Any]:
    coverage = _date(coverage_start_date, label="coverage_start_date")
    cutoff = _date(cutoff_date, label="cutoff_date")
    if coverage > date(2024, 1, 1) or coverage > cutoff:
        raise SystemPreconditionError("calendar compilation coverage is insufficient")
    exchanges = list(pit_exchange_ids)
    if (
        exchanges != sorted(set(exchanges))
        or not exchanges
        or any(exchange not in EXCHANGE_ISSUERS for exchange in exchanges)
    ):
        raise SystemContractError("PIT exchange set is not canonical")
    sessions = [
        _date(value, label=f"market_session_dates[{index}]").isoformat()
        for index, value in enumerate(market_session_dates)
    ]
    if sessions != sorted(set(sessions)) or not sessions or sessions[-1] != cutoff.isoformat():
        raise SystemContractError("market session dates are not canonical to cutoff")
    normalized_release = validate_object_ref(release_ref, label="release_ref")
    if normalized_release["kind"] != "system.release":
        raise SystemContractError("calendar compilation release authority kind differs")
    admissions: dict[bytes, Mapping[str, Any]] = {}
    for document in admission_documents:
        artifact = official.validate_decoder_admission(document)
        key = canonical_json_bytes(object_ref_for_artifact(artifact))
        if key in admissions:
            raise SystemContractError("calendar decoder admission is duplicated")
        admissions[key] = artifact

    def artifact_resolver(ref: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return admissions[canonical_json_bytes(validate_object_ref(ref))]
        except KeyError as exc:
            raise SystemContractError(
                "calendar decoder admission is outside request closure"
            ) from exc

    code_sha = decoder_sha256 or official.decoder_code_sha256()
    capture_map = _resolve_capture_map(
        capture_documents,
        raw_resolver=raw_resolver,
        artifact_resolver=artifact_resolver,
        decoder=decoder,
        admission_resolver=admission_resolver,
        decoder_id_resolver=decoder_id_resolver,
        decoder_sha256=code_sha,
    )
    index_closures = [
        _derive_index_closure(document, coverage=coverage, cutoff=cutoff, capture_map=capture_map)
        for document in index_closure_documents
    ]
    if len({canonical_json_bytes(object_ref_for_artifact(row)) for row in index_closures}) != len(
        index_closures
    ):
        raise SystemContractError("calendar index closure is duplicated")
    source_rows: list[dict[str, Any]] = []
    for exchange in exchanges:
        exchange_captures = [
            row for row in capture_map.values() if row[0]["payload"]["exchange_id"] == exchange
        ]
        exchange_indexes = [
            row for row in index_closures if row["payload"]["exchange_id"] == exchange
        ]
        if not exchange_indexes:
            raise SystemPreconditionError("calendar exchange index closure is absent")
        source_rows.append(
            _derive_exchange_source(
                exchange=exchange,
                coverage=coverage,
                cutoff=cutoff,
                captures=exchange_captures,
                index_closures=exchange_indexes,
            )
        )
    compiled = [_compile_exchange_row(row, coverage=coverage, cutoff=cutoff) for row in source_rows]
    runtime = _runtime_without_provenance(compiled[0]["daily_rows"])
    if any(_runtime_without_provenance(row["daily_rows"]) != runtime for row in compiled[1:]):
        raise SystemPreconditionError("exchange-less runtime projections differ")
    open_dates = {row["date"] for row in runtime if row["status"] == "OPEN"}
    contradictions = [
        {"date": value, "reason": "MARKET_SESSION_DECLARED_CLOSED"}
        for value in sessions
        if value not in open_dates
    ]
    if contradictions:
        raise SystemPreconditionError("calendar compilation contains market contradictions")
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
        raise SystemSecurityError("compiled calendar JSON/Parquet exact readback differs")
    return seal_artifact(
        COMPILATION_KIND,
        {
            "compilation_id": _identifier(compilation_id, label="compilation_id"),
            "state": "COMPILED",
            "coverage_start_date": coverage.isoformat(),
            "cutoff_date": cutoff.isoformat(),
            "timezone": TIMEZONE,
            "compiler_relative_path": COMPILER_RELATIVE_PATH,
            "compiler_code_sha256": compiler_code_sha256(),
            "compiler_ast_sha256": compiler_ast_sha256(),
            "release_ref": normalized_release,
            "pit_exchange_ids": exchanges,
            "market_session_dates_sha256": _sha256(canonical_json_bytes(sessions)),
            "source_exchange_rows": source_rows,
            "source_capture_refs": sorted(
                (object_ref_for_artifact(row[0]) for row in capture_map.values()),
                key=canonical_json_bytes,
            ),
            "decoder_admission_refs": sorted(
                (object_ref_for_artifact(row) for row in admissions.values()),
                key=canonical_json_bytes,
            ),
            "index_closure_refs": sorted(
                (object_ref_for_artifact(row) for row in index_closures),
                key=canonical_json_bytes,
            ),
            "precedence_rules": PRECEDENCE_RULES,
            "exchange_rows": compiled,
            "runtime_projection": runtime,
            "calendar_json_file_ref": json_ref,
            "calendar_parquet_file_ref": parquet_ref,
            "contradiction_rows": [],
        },
        created_at=created_at,
    )


def validate_exchange_calendar_compilation(
    document: Mapping[str, Any] | bytes,
    *,
    pit_exchange_ids: Sequence[str],
    market_session_dates: Sequence[str],
    capture_documents: Sequence[Mapping[str, Any]],
    admission_documents: Sequence[Mapping[str, Any]],
    index_closure_documents: Sequence[Mapping[str, Any]],
    raw_resolver: RawResolver,
    expected_release_ref: Mapping[str, Any],
    decoder: ProjectionDecoder = _default_decoder,
    admission_resolver: AdmissionResolver = official.decoder_admission,
    decoder_id_resolver: DecoderIdResolver = official.decoder_id,
    decoder_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay a compilation with the currently installed admitted decoders."""

    try:
        artifact = validate_artifact(document, expected_kind=COMPILATION_KIND)
    except ContractError as exc:
        raise SystemContractError("exchange calendar compilation contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _COMPILATION_FIELDS:
        raise SystemContractError("exchange calendar compilation fields differ")
    release_ref = validate_object_ref(expected_release_ref, label="expected_release_ref")
    if payload["release_ref"] != release_ref:
        raise SystemContractError("calendar compilation release binding differs")
    if (
        payload["compiler_relative_path"] != COMPILER_RELATIVE_PATH
        or payload["compiler_code_sha256"] != compiler_code_sha256()
        or payload["compiler_ast_sha256"] != compiler_ast_sha256()
    ):
        raise SystemContractError("calendar compiler code identity differs")
    rebuilt = build_exchange_calendar_compilation(
        compilation_id=payload["compilation_id"],
        coverage_start_date=payload["coverage_start_date"],
        cutoff_date=payload["cutoff_date"],
        release_ref=release_ref,
        pit_exchange_ids=pit_exchange_ids,
        market_session_dates=market_session_dates,
        capture_documents=capture_documents,
        admission_documents=admission_documents,
        index_closure_documents=index_closure_documents,
        raw_resolver=raw_resolver,
        calendar_json_file_ref=payload["calendar_json_file_ref"],
        calendar_parquet_file_ref=payload["calendar_parquet_file_ref"],
        created_at=artifact["created_at"],
        decoder=decoder,
        admission_resolver=admission_resolver,
        decoder_id_resolver=decoder_id_resolver,
        decoder_sha256=decoder_sha256,
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemSecurityError("exchange calendar compilation replay differs")
    return artifact


def validate_historical_compilation_envelope(
    document: Mapping[str, Any] | bytes,
    *,
    expected_release_ref: Mapping[str, Any],
    expected_compiler_code_sha256: str,
) -> dict[str, Any]:
    """Authenticate frozen initial bytes without executing descendant code.

    Historical marker replay is intentionally non-executable. The initial
    receipt already bound a successful current-code deep replay; descendants
    authenticate that immutable compilation and its source-object graph.
    """

    try:
        artifact = validate_artifact(document, expected_kind=COMPILATION_KIND)
    except ContractError as exc:
        raise SystemContractError("historical calendar compilation contract failed") from exc
    payload = artifact["payload"]
    if (
        set(payload) != _COMPILATION_FIELDS
        or payload["state"] != "COMPILED"
        or payload["release_ref"]
        != validate_object_ref(expected_release_ref, label="historical release_ref")
        or payload["compiler_relative_path"] != COMPILER_RELATIVE_PATH
        or payload["compiler_code_sha256"] != expected_compiler_code_sha256
        or payload["precedence_rules"] != PRECEDENCE_RULES
        or payload["contradiction_rows"] != []
    ):
        raise SystemContractError("historical calendar compilation envelope differs")
    for field, kind in (
        ("source_capture_refs", CAPTURE_KIND),
        ("decoder_admission_refs", ADMISSION_KIND),
        ("index_closure_refs", INDEX_CLOSURE_KIND),
    ):
        _object_refs(payload[field], label=field, expected_kind=kind)
    _file_ref(payload["calendar_json_file_ref"], label="calendar_json_file_ref")
    _file_ref(payload["calendar_parquet_file_ref"], label="calendar_parquet_file_ref")
    return artifact


__all__ = [
    "COMPILATION_KIND",
    "COMPILER_RELATIVE_PATH",
    "INDEX_CLOSURE_KIND",
    "PRECEDENCE_RULES",
    "build_exchange_calendar_compilation",
    "compiler_ast_sha256",
    "compiler_code_sha256",
    "runtime_json_bytes",
    "runtime_parquet_bytes",
    "validate_exchange_calendar_compilation",
    "validate_historical_compilation_envelope",
]
