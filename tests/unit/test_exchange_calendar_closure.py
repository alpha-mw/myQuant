from __future__ import annotations

import copy
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.market.exchange_calendar_closure import (
    build_exchange_calendar_compilation,
    runtime_json_bytes,
    runtime_parquet_bytes,
    validate_exchange_calendar_compilation,
)
from quant_investor.system import (
    SystemContractError,
    SystemPreconditionError,
    SystemSecurityError,
    object_ref_for_artifact,
)

BASE = "2026-08-16T00:00:00Z"
COVERAGE = date(2024, 1, 1)
CUTOFF = date(2026, 8, 14)
ISSUERS = {"BSE": "BSE_OFFICIAL", "SSE": "SSE_OFFICIAL", "SZSE": "SZSE_OFFICIAL"}
HOSTS = {"BSE": "www.bse.cn", "SSE": "www.sse.com.cn", "SZSE": "www.szse.cn"}
DECODER_SHA = "d" * 64


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_ref(relative: str, raw: bytes, *, size: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"relative_path": relative, "byte_sha256": _sha(raw)}
    if size:
        result["size"] = len(raw)
    return result


def _decoder_id(exchange: str, role: str, category: str | None = None) -> str:
    suffix = "" if category is None else "." + category.lower()
    return f"test-only.native.{exchange.lower()}.{role.lower()}{suffix}"


def _decoder(exchange: str, role: str, raw: bytes, *, media_type: str) -> dict[str, object]:
    assert media_type == "application/json"
    document = json.loads(raw)
    assert document["native_exchange"] == exchange
    assert document["native_role"] == role
    return document["projection"]


def _native(exchange: str, role: str, projection: dict[str, object]) -> bytes:
    return canonical_json_bytes(
        {"native_exchange": exchange, "native_role": role, "projection": projection}
    )


def _runtime(
    closures: set[str], *, coverage: date = COVERAGE, cutoff: date = CUTOFF
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    zone = ZoneInfo("Asia/Shanghai")
    cursor = coverage
    while cursor <= cutoff:
        text = cursor.isoformat()
        is_open = cursor.isoweekday() <= 5 and text not in closures
        rows.append(
            {
                "date": text,
                "status": "OPEN" if is_open else "CLOSED",
                "opens_at_utc": (
                    datetime.combine(cursor, time(9, 30), tzinfo=zone)
                    .astimezone(timezone.utc)
                    .isoformat()
                    if is_open
                    else None
                ),
                "closes_at_utc": (
                    datetime.combine(cursor, time(15, 0), tzinfo=zone)
                    .astimezone(timezone.utc)
                    .isoformat()
                    if is_open
                    else None
                ),
            }
        )
        cursor += timedelta(days=1)
    return rows


def _case(
    *,
    coverage: date = COVERAGE,
    cutoff: date = CUTOFF,
    exchanges: tuple[str, ...] = ("BSE", "SSE", "SZSE"),
    discovery_start: str = "2023-01-01",
) -> dict[str, Any]:  # noqa: C901
    closures = {"2024-01-01", "2025-01-01", "2026-01-01"}
    raw_by_ref: dict[bytes, bytes] = {}
    admissions: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    admission_by_subject: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    category_sets: dict[tuple[str, str], dict[str, Any]] = {}
    category_id = "TRADING_CALENDAR_NOTICES"
    category_set_id = "ALL_CALENDAR_NOTICE_CATEGORIES"
    query_parameters = [
        {"name": "category", "value_source": "ISSUER_CATEGORY_ID"},
        {"name": "end", "value_source": "DISCOVERY_PUBLISH_END_DATE"},
        {"name": "page", "value_source": "PAGE_NUMBER"},
        {"name": "start", "value_source": "DISCOVERY_PUBLISH_START_DATE"},
    ]
    sessions = [
        {
            "phase": "MORNING_CONTINUOUS_AUCTION",
            "opens_local": "09:30:00",
            "closes_local": "11:30:00",
        },
        {
            "phase": "AFTERNOON_CONTINUOUS_AUCTION",
            "opens_local": "13:00:00",
            "closes_local": "15:00:00",
        },
    ]
    for exchange in exchanges:
        projections: dict[str, dict[str, object]] = {
            "TRADING_WEEK_RULE": {
                "weekly_rule_intervals": [
                    {
                        "start_date": coverage.isoformat(),
                        "end_date": cutoff.isoformat(),
                        "weekdays": [1, 2, 3, 4, 5],
                    }
                ]
            },
            "SESSION_RULE": {
                "session_rule_intervals": [
                    {
                        "start_date": coverage.isoformat(),
                        "end_date": cutoff.isoformat(),
                        "session_intervals": [
                            *(
                                [
                                    {
                                        "phase": "OPENING_CALL_AUCTION",
                                        "opens_local": "09:15:00",
                                        "closes_local": "09:25:00",
                                    }
                                ]
                                if exchange == "BSE"
                                else []
                            ),
                            *sessions,
                        ],
                    }
                ]
            },
            "ANNUAL_HOLIDAY_NOTICE": {"closure_dates": sorted(closures)},
        }
        body_url = f"https://{HOSTS[exchange]}/calendar/annual"
        projections["NOTICE_INDEX_SNAPSHOT"] = {
            "category": category_id,
            "page_number": 1,
            "page_count": 1,
            "reported_item_count": 1,
            "discovery_publish_start_date": discovery_start,
            "discovery_publish_end_date": cutoff.isoformat(),
            "entries": [
                {
                    "entry_id": f"{exchange.lower()}-annual",
                    "publish_date": "2023-12-29",
                    "title": "Official annual market closure notice",
                    "body_url": body_url,
                    "relevant": True,
                    "evidence_role": "ANNUAL_HOLIDAY_NOTICE",
                }
            ],
        }
        by_role: dict[str, dict[str, Any]] = {}
        for role, projection in projections.items():
            index_category = category_id if role == "NOTICE_INDEX_SNAPSHOT" else None
            raw = _native(exchange, role, projection)
            raw_ref = _file_ref(f"raw/{exchange.lower()}-{role.lower()}.json", raw)
            raw_by_ref[canonical_json_bytes(raw_ref)] = raw
            fixture_ref = _file_ref(
                f"raw/{exchange.lower()}-{role.lower()}-fixture.json", raw, size=True
            )
            raw_by_ref[canonical_json_bytes(fixture_ref)] = raw
            if role == "ANNUAL_HOLIDAY_NOTICE":
                request_url = body_url
            elif role == "NOTICE_INDEX_SNAPSHOT":
                request_url = (
                    f"https://{HOSTS[exchange]}/calendar/notice-index"
                    f"?category={category_id}&end={cutoff.isoformat()}&page=1"
                    f"&start={discovery_start}"
                )
            else:
                request_url = f"https://{HOSTS[exchange]}/calendar/{role.lower()}"
            endpoint = (
                "/calendar/annual"
                if role == "ANNUAL_HOLIDAY_NOTICE"
                else (
                    "/calendar/notice-index"
                    "?category={ISSUER_CATEGORY_ID}"
                    "&end={DISCOVERY_PUBLISH_END_DATE}"
                    "&page={PAGE_NUMBER}"
                    "&start={DISCOVERY_PUBLISH_START_DATE}"
                    if role == "NOTICE_INDEX_SNAPSHOT"
                    else f"/calendar/{role.lower()}"
                )
            )
            admission = seal_artifact(
                "system.exchange_calendar_decoder_admission",
                {
                    "decoder_admission_id": f"{exchange.lower()}-{role.lower()}-admission",
                    "state": "ADMITTED",
                    "exchange_id": exchange,
                    "evidence_role": role,
                    "issuer": ISSUERS[exchange],
                    "endpoint_scheme": "https",
                    "endpoint_host": HOSTS[exchange],
                    "endpoint_path_query_template": endpoint,
                    "issuer_category_id": index_category,
                    "category_scope": (
                        "ALL_MARKET_CLOSURE_AND_SESSION_CHANGE_NOTICES"
                        if index_category is not None
                        else None
                    ),
                    "category_completeness_policy": (
                        "COMPLETE_ISSUER_CATEGORY_PAGINATION"
                        if index_category is not None
                        else None
                    ),
                    "query_window_semantics": (
                        "PUBLISH_DATE_INCLUSIVE" if index_category is not None else None
                    ),
                    "required_query_parameters": (
                        query_parameters if index_category is not None else []
                    ),
                    "page_parameter": "page" if index_category is not None else None,
                    "cursor_parameter": None,
                    "required_category_set_id": (
                        category_set_id if index_category is not None else None
                    ),
                    "discovery_start_date": (
                        discovery_start if index_category is not None else None
                    ),
                    "fixture_request_url": request_url,
                    "fixture_effective_url": request_url,
                    "fixture_redirect_chain": [],
                    "fixture_tls_verified": True,
                    "redirect_policy": "NO_REDIRECTS",
                    "http_status": 200,
                    "raw_media_type": "application/json",
                    "response_headers": [{"name": "content-type", "value": "application/json"}],
                    "fixture_raw_file_ref": fixture_ref,
                    "fixture_raw_sha256": fixture_ref["byte_sha256"],
                    "fixture_captured_at": BASE,
                    "decoder_id": _decoder_id(exchange, role, index_category),
                    "decoder_sha256": DECODER_SHA,
                    "fixture_projection_sha256": _sha(canonical_json_bytes(projection)),
                    "review_basis": "test-only-native-wire-contract",
                },
                created_at=BASE,
            )
            admissions.append(admission)
            admission_by_subject[(exchange, role, index_category)] = admission
            capture = seal_artifact(
                "system.exchange_calendar_capture",
                {
                    "calendar_capture_id": f"{exchange.lower()}-{role.lower()}-capture",
                    "state": "IMMUTABLE",
                    "evidence_role": role,
                    "exchange_id": exchange,
                    "issuer": ISSUERS[exchange],
                    "request_url": request_url,
                    "effective_url": request_url,
                    "redirect_chain": [],
                    "request_headers": [],
                    "response_headers": [{"name": "content-type", "value": "application/json"}],
                    "http_status": 200,
                    "tls_verified": True,
                    "captured_at": BASE,
                    "raw_file_ref": raw_ref,
                    "raw_sha256": raw_ref["byte_sha256"],
                    "raw_byte_length": len(raw),
                    "raw_media_type": "application/json",
                    "decoder_admission_ref": object_ref_for_artifact(admission),
                    "decoder_id": _decoder_id(exchange, role, index_category),
                    "decoder_sha256": DECODER_SHA,
                    "projection_sha256": _sha(canonical_json_bytes(projection)),
                },
                created_at=BASE,
            )
            captures.append(capture)
            by_role[role] = capture
        page_ref = object_ref_for_artifact(by_role["NOTICE_INDEX_SNAPSHOT"])
        body_ref = object_ref_for_artifact(by_role["ANNUAL_HOLIDAY_NOTICE"])
        indexes.append(
            seal_artifact(
                "system.exchange_calendar_index_closure",
                {
                    "index_closure_id": f"{exchange.lower()}-calendar-index",
                    "state": "COMPLETE",
                    "exchange_id": exchange,
                    "issuer": ISSUERS[exchange],
                    "issuer_category_id": category_id,
                    "required_category_set_id": category_set_id,
                    "category_scope": "ALL_MARKET_CLOSURE_AND_SESSION_CHANGE_NOTICES",
                    "category_completeness_policy": "COMPLETE_ISSUER_CATEGORY_PAGINATION",
                    "query_window_semantics": "PUBLISH_DATE_INCLUSIVE",
                    "root_capture_ref": page_ref,
                    "page_capture_refs": [page_ref],
                    "reported_page_count": 1,
                    "reported_item_count": 1,
                    "observed_item_count": 1,
                    "discovery_publish_start_date": discovery_start,
                    "discovery_publish_end_date": cutoff.isoformat(),
                    "calendar_effective_coverage_start_date": coverage.isoformat(),
                    "calendar_effective_coverage_end_date": cutoff.isoformat(),
                    "entry_rows": projections["NOTICE_INDEX_SNAPSHOT"]["entries"],
                    "body_capture_refs": [body_ref],
                    "pagination_complete": True,
                    "discovery_window_complete": True,
                    "calendar_coverage_bound": True,
                    "unknown_relevant_count": 0,
                },
                created_at=BASE,
            )
        )
        category_sets[(exchange, category_set_id)] = {
            "exchange_id": exchange,
            "required_category_set_id": category_set_id,
            "issuer": ISSUERS[exchange],
            "category_scope": "ALL_MARKET_CLOSURE_AND_SESSION_CHANGE_NOTICES",
            "category_completeness_policy": "COMPLETE_ISSUER_CATEGORY_PAGINATION",
            "query_window_semantics": "PUBLISH_DATE_INCLUSIVE",
            "required_query_parameters": query_parameters,
            "page_parameter": "page",
            "cursor_parameter": None,
            "required_issuer_category_ids": [category_id],
            "category_role_rows": [
                {
                    "issuer_category_id": category_id,
                    "allowed_evidence_roles": [
                        "ANNUAL_HOLIDAY_NOTICE",
                        "SESSION_CHANGE_NOTICE",
                        "TEMPORARY_CLOSURE_NOTICE",
                    ],
                }
            ],
            "discovery_start_date": discovery_start,
            "maximum_page_count": 8,
            "maximum_body_count": 64,
        }
    runtime = _runtime(closures, coverage=coverage, cutoff=cutoff)
    json_raw = runtime_json_bytes(runtime)
    parquet_raw = runtime_parquet_bytes(runtime)
    json_ref = _file_ref("strict/exchange_calendar.json", json_raw)
    parquet_ref = _file_ref("strict/exchange_calendar.parquet", parquet_raw)
    raw_by_ref[canonical_json_bytes(json_ref)] = json_raw
    raw_by_ref[canonical_json_bytes(parquet_ref)] = parquet_raw
    market_sessions = [row["date"] for row in runtime if row["status"] == "OPEN"]

    def resolve_raw(reference: dict[str, Any]) -> bytes:
        return raw_by_ref[canonical_json_bytes(reference)]

    release = seal_artifact(
        "system.release",
        {
            "release_id": "calendar-closure-test-release",
            "state": "OPERATIONAL",
            "code_sha256": "1" * 64,
            "wheel_sha256": "2" * 64,
            "code_manifest_sha256": "3" * 64,
        },
        created_at=BASE,
    )
    return {
        "release_ref": object_ref_for_artifact(release),
        "captures": captures,
        "admissions": admissions,
        "indexes": indexes,
        "admission_by_subject": admission_by_subject,
        "category_sets": category_sets,
        "raw_resolver": resolve_raw,
        "json_ref": json_ref,
        "parquet_ref": parquet_ref,
        "market_sessions": market_sessions,
        "coverage": coverage,
        "cutoff": cutoff,
        "exchanges": list(exchanges),
        "raw_by_ref": raw_by_ref,
    }


def _build(case: dict[str, Any]) -> dict[str, Any]:
    return build_exchange_calendar_compilation(
        compilation_id="official-cn-calendar-test",
        coverage_start_date=case["coverage"].isoformat(),
        cutoff_date=case["cutoff"].isoformat(),
        release_ref=case["release_ref"],
        pit_exchange_ids=case["exchanges"],
        market_session_dates=case["market_sessions"],
        capture_documents=case["captures"],
        admission_documents=case["admissions"],
        index_closure_documents=case["indexes"],
        raw_resolver=case["raw_resolver"],
        calendar_json_file_ref=case["json_ref"],
        calendar_parquet_file_ref=case["parquet_ref"],
        created_at=BASE,
        decoder=_decoder,
        admission_resolver=lambda exchange, role, category: case["admission_by_subject"][
            (exchange, role, category)
        ],
        decoder_id_resolver=_decoder_id,
        category_set_resolver=lambda exchange, set_id: case["category_sets"][(exchange, set_id)],
        decoder_sha256=DECODER_SHA,
    )


def test_native_closure_replays_three_exchanges_and_exact_outputs() -> None:
    case = _case()
    artifact = _build(case)
    assert (
        validate_exchange_calendar_compilation(
            artifact,
            pit_exchange_ids=case["exchanges"],
            market_session_dates=case["market_sessions"],
            capture_documents=case["captures"],
            admission_documents=case["admissions"],
            index_closure_documents=case["indexes"],
            raw_resolver=case["raw_resolver"],
            expected_release_ref=case["release_ref"],
            decoder=_decoder,
            admission_resolver=lambda exchange, role, category: case["admission_by_subject"][
                (exchange, role, category)
            ],
            decoder_id_resolver=_decoder_id,
            category_set_resolver=lambda exchange, set_id: case["category_sets"][
                (exchange, set_id)
            ],
            decoder_sha256=DECODER_SHA,
        )
        == artifact
    )
    bse_sessions = artifact["payload"]["exchange_rows"][0]["session_rule_intervals"]
    assert bse_sessions[0]["session_intervals"][0]["phase"] == "OPENING_CALL_AUCTION"
    assert all(row["open_session_count"] >= 391 for row in artifact["payload"]["exchange_rows"])


def test_request_headers_and_projection_tamper_fail_closed() -> None:
    case = _case()
    case["captures"][0]["payload"]["request_headers"] = [{"name": "cookie", "value": "secret"}]
    case["captures"][0] = seal_artifact(
        "system.exchange_calendar_capture",
        case["captures"][0]["payload"],
        created_at=BASE,
    )
    with pytest.raises(SystemSecurityError, match="request headers"):
        _build(case)


def test_index_claim_and_market_contradiction_fail_closed() -> None:
    case = _case()
    tampered = copy.deepcopy(case)
    tampered["indexes"][0]["payload"]["reported_item_count"] = 2
    tampered["indexes"][0] = seal_artifact(
        "system.exchange_calendar_index_closure",
        tampered["indexes"][0]["payload"],
        created_at=BASE,
    )
    with pytest.raises((SystemSecurityError, SystemPreconditionError)):
        _build(tampered)
    case["market_sessions"] = sorted([*case["market_sessions"], "2026-08-08"])
    with pytest.raises(SystemPreconditionError, match="market contradictions"):
        _build(case)


def test_exact_runtime_output_tamper_fails_closed() -> None:
    case = _case()
    original = case["raw_resolver"]
    parquet_ref = case["parquet_ref"]

    def tampered(reference: dict[str, Any]) -> bytes:
        raw = original(reference)
        return raw + b"x" if reference == parquet_ref else raw

    case["raw_resolver"] = tampered
    with pytest.raises(SystemSecurityError, match="readback"):
        _build(case)


def _mutate_capture_url(
    case: dict[str, Any],
    *,
    exchange: str,
    role: str,
    request_url: str | None = None,
    effective_url: str | None = None,
    redirect_chain: list[str] | None = None,
) -> None:
    for index, capture in enumerate(case["captures"]):
        payload = capture["payload"]
        if payload["exchange_id"] == exchange and payload["evidence_role"] == role:
            updated = copy.deepcopy(payload)
            if request_url is not None:
                updated["request_url"] = request_url
            if effective_url is not None:
                updated["effective_url"] = effective_url
            if redirect_chain is not None:
                updated["redirect_chain"] = redirect_chain
            case["captures"][index] = seal_artifact(
                "system.exchange_calendar_capture",
                updated,
                created_at=BASE,
            )
            return
    raise AssertionError("capture subject not found")


def test_required_category_union_rejects_missing_extra_and_omitted_temporary_category() -> None:
    missing = _case(exchanges=("SSE",))
    policy = missing["category_sets"][("SSE", "ALL_CALENDAR_NOTICE_CATEGORIES")]
    policy["required_issuer_category_ids"] = [
        "TEMPORARY_CLOSURE_NOTICES",
        "TRADING_CALENDAR_NOTICES",
    ]
    policy["category_role_rows"] = [
        {
            "issuer_category_id": "TEMPORARY_CLOSURE_NOTICES",
            "allowed_evidence_roles": ["TEMPORARY_CLOSURE_NOTICE"],
        },
        {
            "issuer_category_id": "TRADING_CALENDAR_NOTICES",
            "allowed_evidence_roles": [
                "ANNUAL_HOLIDAY_NOTICE",
                "SESSION_CHANGE_NOTICE",
                "TEMPORARY_CLOSURE_NOTICE",
            ],
        },
    ]
    with pytest.raises(SystemPreconditionError, match="required-category set"):
        _build(missing)

    extra = _case(exchanges=("SSE",))
    duplicate = copy.deepcopy(extra["indexes"][0])
    duplicate["payload"]["index_closure_id"] = "sse-calendar-index-extra"
    extra["indexes"].append(
        seal_artifact(
            "system.exchange_calendar_index_closure",
            duplicate["payload"],
            created_at=BASE,
        )
    )
    with pytest.raises(SystemPreconditionError, match="required-category set"):
        _build(extra)


def test_index_query_category_page_and_window_are_bidirectionally_bound() -> None:
    cases = (
        "category=WRONG&end=2026-08-14&page=1&start=2023-01-01",
        "category=TRADING_CALENDAR_NOTICES&end=2026-08-14&page=2&start=2023-01-01",
        "category=TRADING_CALENDAR_NOTICES&end=2026-08-13&page=1&start=2023-01-01",
    )
    for query in cases:
        case = _case(exchanges=("SSE",))
        _mutate_capture_url(
            case,
            exchange="SSE",
            role="NOTICE_INDEX_SNAPSHOT",
            request_url=f"https://www.sse.com.cn/calendar/notice-index?{query}",
            effective_url=f"https://www.sse.com.cn/calendar/notice-index?{query}",
        )
        with pytest.raises(SystemSecurityError, match="query projection binding"):
            _build(case)


def test_discovery_window_must_include_prior_year_annual_notice() -> None:
    case = _case(exchanges=("SSE",), discovery_start="2024-01-01")
    with pytest.raises((SystemPreconditionError, SystemSecurityError)):
        _build(case)


def test_category_scope_policy_is_code_owned() -> None:
    case = _case(exchanges=("SSE",))
    case["category_sets"][("SSE", "ALL_CALENDAR_NOTICE_CATEGORIES")][
        "category_scope"
    ] = "ANNUAL_ONLY"
    with pytest.raises((SystemPreconditionError, SystemSecurityError)):
        _build(case)


def test_cross_page_discovery_windows_must_be_identical() -> None:
    case = _case(exchanges=("SSE",))
    capture_index = next(
        index
        for index, capture in enumerate(case["captures"])
        if capture["payload"]["exchange_id"] == "SSE"
        and capture["payload"]["evidence_role"] == "NOTICE_INDEX_SNAPSHOT"
    )
    original = case["captures"][capture_index]
    first_projection = json.loads(case["raw_resolver"](original["payload"]["raw_file_ref"]))[
        "projection"
    ]
    first_projection["page_count"] = 2
    first_raw = _native("SSE", "NOTICE_INDEX_SNAPSHOT", first_projection)
    first_ref = _file_ref("raw/sse-notice-index-page-1.json", first_raw)
    case["raw_by_ref"][canonical_json_bytes(first_ref)] = first_raw
    first_payload = copy.deepcopy(original["payload"])
    first_payload["raw_file_ref"] = first_ref
    first_payload["raw_sha256"] = first_ref["byte_sha256"]
    first_payload["raw_byte_length"] = len(first_raw)
    first_payload["projection_sha256"] = _sha(canonical_json_bytes(first_projection))
    first_capture = seal_artifact(
        "system.exchange_calendar_capture",
        first_payload,
        created_at=BASE,
    )
    case["captures"][capture_index] = first_capture

    second_projection = copy.deepcopy(first_projection)
    second_projection["page_number"] = 2
    second_projection["discovery_publish_end_date"] = "2026-08-13"
    second_projection["entries"] = []
    second_raw = _native("SSE", "NOTICE_INDEX_SNAPSHOT", second_projection)
    second_ref = _file_ref("raw/sse-notice-index-page-2.json", second_raw)
    case["raw_by_ref"][canonical_json_bytes(second_ref)] = second_raw
    second_payload = copy.deepcopy(first_payload)
    second_payload["calendar_capture_id"] = "sse-notice-index-page-2-capture"
    second_payload["request_url"] = (
        "https://www.sse.com.cn/calendar/notice-index"
        "?category=TRADING_CALENDAR_NOTICES&end=2026-08-13&page=2&start=2023-01-01"
    )
    second_payload["effective_url"] = second_payload["request_url"]
    second_payload["raw_file_ref"] = second_ref
    second_payload["raw_sha256"] = second_ref["byte_sha256"]
    second_payload["raw_byte_length"] = len(second_raw)
    second_payload["projection_sha256"] = _sha(canonical_json_bytes(second_projection))
    second_capture = seal_artifact(
        "system.exchange_calendar_capture",
        second_payload,
        created_at=BASE,
    )
    case["captures"].append(second_capture)

    index_payload = copy.deepcopy(case["indexes"][0]["payload"])
    first_capture_ref = object_ref_for_artifact(first_capture)
    second_capture_ref = object_ref_for_artifact(second_capture)
    index_payload["root_capture_ref"] = first_capture_ref
    index_payload["page_capture_refs"] = sorted(
        [first_capture_ref, second_capture_ref],
        key=canonical_json_bytes,
    )
    index_payload["reported_page_count"] = 2
    case["indexes"][0] = seal_artifact(
        "system.exchange_calendar_index_closure",
        index_payload,
        created_at=BASE,
    )

    with pytest.raises(SystemPreconditionError, match="pagination metadata"):
        _build(case)


def test_actual_no_redirects_policy_rejects_same_host_redirect() -> None:
    case = _case(exchanges=("SSE",))
    redirected = "https://www.sse.com.cn/calendar/trading_week_rule-final"
    _mutate_capture_url(
        case,
        exchange="SSE",
        role="TRADING_WEEK_RULE",
        effective_url=redirected,
        redirect_chain=[redirected],
    )
    with pytest.raises(SystemContractError, match="NO_REDIRECTS"):
        _build(case)
