"""Frozen official-source compiler for the disconnected CN 2026 calendar.

The compiler has no network or discovery fallback.  It accepts only the
reviewed source inventory, proves byte identity before parsing, and emits a
permanently nonauthorizing canonical projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import stat
import unicodedata
from typing import Any

from .contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    seal_semantic,
    semantic_sha256,
    sha256_bytes,
    validate_semantic_seal,
)
from .secure_io import PRIVATE_EVIDENCE_POLICY, load_bound_raw_artifact

SOURCE_BINDING_SCHEMA = "v16.calendar-source-binding.v1"
SOURCE_DOCUMENT_SET_SCHEMA = "v16.calendar-source-document-set.v1"
OPEN_SESSION_CALENDAR_SCHEMA = "v16.open-session-calendar.v1"
OPEN_SESSION_CALENDAR_ID = "cn.open-session-calendar.2026.v1"
PRIVATE_ROOT_POLICY = PRIVATE_EVIDENCE_POLICY.policy_id
SOURCE_AUTHORITY_SCOPE = "declared_exchange_url_semantic_correspondence_only"
CALENDAR_AUTHORITY_SCOPE = "official_exchange_rules_and_closure_notices_only"
CALENDAR_YEAR = 2026
CALENDAR_TIMEZONE = "Asia/Shanghai"

ANNUAL_CLOSURE_PROJECTION_SCHEMA = "v16.annual-closure-notice-projection.v1"
ACTIVE_CLOSURE_YEAR_PROJECTION_SCHEMA = (
    "v16.active-closure-explicit-year-projection.v1"
)
ACTIVE_CLOSURE_MONTH_DAY_PROJECTION_SCHEMA = (
    "v16.active-closure-month-day-projection.v1"
)
RULE_EFFECTIVE_EVENT_PROJECTION_SCHEMA = "v16.rule-effective-event-projection.v1"
RULE_NOTICE_TRIGGER_PROJECTION_SCHEMA = "v16.rule-notice-trigger-projection.v1"
RULE_NOTICE_EXPLICIT_PROJECTION_SCHEMA = (
    "v16.rule-notice-explicit-date-projection.v1"
)
WAYBACK_CDX_PROJECTION_SCHEMA = "v16.wayback-cdx-projection.v1"
INLINE_CALENDAR_RULE_PROJECTION_SCHEMA = "v16.inline-calendar-rule-projection.v1"
INLINE_SESSION_CLOCK_PROJECTION_SCHEMA = (
    "v16.inline-session-clock-rule-projection.v1"
)
RULE_BINARY_PROFILE_SCHEMA = "v16.rule-binary-semantic-profile.v1"

WEEKDAY_OPEN_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
CLOSURE_RULE = "national_statutory_holidays_and_exchange_announced_closures"
CLOCK_SCOPE_ID = "listed_equity_auction_trading"
CLOCK_SEGMENTS = (
    {
        "segment_id": "opening_call_auction",
        "start_local_time": "09:15:00",
        "end_local_time": "09:25:00",
    },
    {
        "segment_id": "continuous_auction_am",
        "start_local_time": "09:30:00",
        "end_local_time": "11:30:00",
    },
    {
        "segment_id": "continuous_auction_pm",
        "start_local_time": "13:00:00",
        "end_local_time": "14:57:00",
    },
    {
        "segment_id": "closing_call_auction",
        "start_local_time": "14:57:00",
        "end_local_time": "15:00:00",
    },
)
CLOCK_EXCLUDED_SCOPES = (
    "after_hours_fixed_price",
    "block_trading",
    "bond_trading",
    "fund_trading",
)

CLOSED_WEEKDAY_DATES = (
    "2026-01-01",
    "2026-01-02",
    "2026-02-16",
    "2026-02-17",
    "2026-02-18",
    "2026-02-19",
    "2026-02-20",
    "2026-02-23",
    "2026-04-06",
    "2026-05-01",
    "2026-05-04",
    "2026-05-05",
    "2026-06-19",
    "2026-09-25",
    "2026-10-01",
    "2026-10-02",
    "2026-10-05",
    "2026-10-06",
    "2026-10-07",
)
REOPENING_DATES = (
    "2026-01-05",
    "2026-02-24",
    "2026-04-07",
    "2026-05-06",
    "2026-06-22",
    "2026-09-28",
    "2026-10-08",
)
CLOSED_MONTH_DAYS = tuple(value[5:] for value in CLOSED_WEEKDAY_DATES)
REOPENING_MONTH_DAYS = tuple(value[5:] for value in REOPENING_DATES)


def _open_sessions_2026() -> tuple[str, ...]:
    current = date(CALENDAR_YEAR, 1, 1)
    end = date(CALENDAR_YEAR + 1, 1, 1)
    closed = set(CLOSED_WEEKDAY_DATES)
    result: list[str] = []
    while current < end:
        text = current.isoformat()
        if current.weekday() < 5 and text not in closed:
            result.append(text)
        current += timedelta(days=1)
    if len(result) != 242:
        raise RuntimeError("frozen 2026 calendar does not contain 242 sessions")
    return tuple(result)


OPEN_SESSIONS = _open_sessions_2026()


@dataclass(frozen=True)
class SourceInventoryEntry:
    filename: str
    size: int
    byte_sha256: str
    consumed: bool


def _inventory(
    filename: str,
    size: int,
    byte_sha256: str,
    *,
    consumed: bool = True,
) -> SourceInventoryEntry:
    return SourceInventoryEntry(filename, size, byte_sha256, consumed)


SOURCE_INVENTORY = (
    _inventory(
        "myquant-csrc-2023-mainboard-registration-implementation.html",
        11883,
        "84bd60d15279d4de874c80c35d5f6003a57bb4fa0faabe1da6784297206fb619",
    ),
    _inventory(
        "myquant-sse-2023-mainboard-registration-implementation-en.html",
        70138,
        "e36d7cfd43a4996e4a3e92a8abe7036de2c8a4c845e99c35b97a03ae258d3f77",
    ),
    _inventory(
        "myquant-sse-2023-trading-rules-notice.html",
        35521,
        "df5e1671823fc1b3bf94ec9ce4536cf846bdeb0f6781bbee21df4e8df7ff060f",
    ),
    _inventory(
        "myquant-sse-2023-trading-rules.docx",
        51948,
        "7aa2319f6dcf597be1e86b3b69d7c2ad0e6acb2a5d0cc6be48a01af602fded40",
    ),
    _inventory(
        "myquant-sse-2026-closures.html",
        29060,
        "ec22ea038b51fad00069a9370934056384ecc67aaae0988d610632f94fa84411",
    ),
    _inventory(
        "myquant-sse-2026-active-closures.html",
        30633,
        "458e21da31490dba7aee6630bfc2ccd026657cb138c365dcc17fbaeeb27d5538",
    ),
    _inventory(
        "myquant-sse-2026-trading-rules.html",
        34632,
        "83b98c8c02bb237d00a16f4eaa1bc9d4496ea81385d2feae52e8e7892347687b",
    ),
    _inventory(
        "myquant-sse-2026-trading-rules.docx",
        65302,
        "fc922c433438b2636cb631eab25cca405209712acbb6aaded768c45456ff8888",
    ),
    _inventory(
        "myquant-szse-2023-events.html",
        125239,
        "3c2d08dbcc90adc753e71971076b1b6ec647ad4cdc77cd0c423664668ffa82df",
    ),
    _inventory(
        "myquant-szse-2023-trading-rules-notice-wayback-cdx.json",
        213,
        "0b5eb4951cf4ae7d8c8021011cbecd74574181c32cc3e03ad392c8710a30ebeb",
    ),
    _inventory(
        "myquant-szse-2023-trading-rules-notice-wayback.html",
        14488,
        "119a4cf1c510289b94718e5a565a7a7229bf4a01e71b29a9de4e505ecf9cf2cf",
    ),
    _inventory(
        "myquant-szse-2023-trading-rules-wayback-alias-cdx.json",
        215,
        "af9f661df127ab650bb0f94f520891cc55ec7d8ce7ecd1b8c6efdd2e7d679168",
    ),
    _inventory(
        "myquant-szse-2023-trading-rules-wayback-cdx.json",
        208,
        "9899cf193888c3a83cae4f65f1455f1e3364a3bb23551a4690b71bc823b5cc59",
    ),
    _inventory(
        "myquant-szse-2023-trading-rules-wayback.pdf",
        620843,
        "7018114a6e11deb239c2a72e71e49defc6e8841b3e2c093b3bbf809282c67222",
    ),
    _inventory(
        "myquant-szse-2026-active-calendar-en.html",
        47153,
        "f8b2b687dc55fbd41dc8cdd562e7f7988dda8e022b0ba83cb27516d5c35a2bc1",
    ),
    _inventory(
        "myquant-szse-2026-closures-20260719.html",
        24892,
        "adfcdc1c285121a58ad3a55060a7125812d542cf9fdb92058aac1b1bc99c8243",
    ),
    _inventory(
        "myquant-szse-2026-trading-rules-notice.html",
        19405,
        "98ec21eee37d254bb0f3a4bb37f7eec700a1e301754ae79d88c52f5674fa691c",
    ),
    _inventory(
        "myquant-szse-2026-trading-rules.pdf",
        282084,
        "9b66f8b0db70f84a25ef1ccb4ee2351001724e408117552d75f6d8993483c586",
    ),
    _inventory(
        "myquant-bse-2026-closures-annual.html",
        21440,
        "1d6178157837f021dcb98c1206ee2f6f8cafcadbbb5b1cb0ac3637bbe82387ab",
    ),
    _inventory(
        "myquant-bse-2026-closures-active.html",
        20482,
        "031c2af35a824249a8f310d026668e60ffe96652f023fa5879855f09b3f034bd",
    ),
    _inventory(
        "myquant-bse-2021-trading-rules-prior.html",
        119658,
        "9e2271bef634fd9337c4ee1cef8679a5cb99f813f6e1aae4d099b57d2a01ad18",
    ),
    _inventory(
        "myquant-bse-2026-trading-rules-current.html",
        158945,
        "e800ab4de9688136aff039239fbb8e48b33a732d43a79b302cbdb848d532c7ba",
    ),
    _inventory(
        "myquant-szse-2023-rule-implementation-report.pdf",
        5562125,
        "9540ba342ab92b6cb464d24cd2e7acfa45db2847ea3a6832a8767bc60f6c9029",
        consumed=False,
    ),
    _inventory(
        "myquant-szse-2026-closures.html",
        24946,
        "1656d8708d5faed33c7588538639d96128f39fdface02cfa6eb66d796c40f14e",
        consumed=False,
    ),
)
SOURCE_INVENTORY_BY_NAME = {entry.filename: entry for entry in SOURCE_INVENTORY}
CONSUMED_SOURCE_NAMES = frozenset(
    entry.filename for entry in SOURCE_INVENTORY if entry.consumed
)
EXCLUDED_SOURCE_NAMES = frozenset(
    entry.filename for entry in SOURCE_INVENTORY if not entry.consumed
)


@dataclass(frozen=True)
class _BindingSpec:
    binding_id: str
    parser_contract_id: str
    filename: str
    declared_origin_url: str
    retrieval_url: str
    retrieval_method: str
    semantic: Mapping[str, Any]
    marker_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class BoundSource:
    binding_id: str
    artifact: BoundRawArtifact

    def __post_init__(self) -> None:
        if not self.binding_id or not isinstance(self.artifact, BoundRawArtifact):
            raise EvidenceV2Error("bound source has an invalid envelope")


@dataclass(frozen=True)
class CalendarEvidenceBundle:
    calendar: BoundCanonicalArtifact
    sources: tuple[BoundSource, ...]

    def read(self) -> dict[str, Any]:
        if self.calendar.reference.root_policy != PRIVATE_ROOT_POLICY:
            raise EvidenceV2Error("calendar evidence bundle must use the private root")
        declared = validate_open_session_calendar(self.calendar.read())
        rebuilt = build_open_session_calendar(self.sources)
        if rebuilt != declared:
            raise EvidenceV2Error("calendar evidence bundle does not recompute exactly")
        return declared


class _SemanticHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_fragments: list[str] = []
        self.attribute_fragments: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        for _key, value in attrs:
            if value:
                self.attribute_fragments.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.text_fragments.append(data)

    @property
    def compact_text(self) -> str:
        return "|".join(
            _compact("".join(fragments))
            for fragments in (self.text_fragments, self.attribute_fragments)
        )


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    return seal_semantic(dict(value))


def _annual_projection(
    exchange: str,
    title: str,
    document_number: str | None,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": ANNUAL_CLOSURE_PROJECTION_SCHEMA,
            "exchange": exchange,
            "document_role": "annual_exchange_closure_notice",
            "document_title": title,
            "document_number": document_number,
            "publication_date": "2025-12-22",
            "calendar_year": CALENDAR_YEAR,
            "closed_weekday_dates": list(CLOSED_WEEKDAY_DATES),
            "reopening_dates": list(REOPENING_DATES),
        }
    )


def _active_year_projection(exchange: str) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": ACTIVE_CLOSURE_YEAR_PROJECTION_SCHEMA,
            "exchange": exchange,
            "document_role": "active_exchange_closure_schedule",
            "calendar_year": CALENDAR_YEAR,
            "closed_weekday_dates": list(CLOSED_WEEKDAY_DATES),
            "reopening_dates": list(REOPENING_DATES),
        }
    )


def _active_month_day_projection() -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": ACTIVE_CLOSURE_MONTH_DAY_PROJECTION_SCHEMA,
            "exchange": "BSE",
            "document_role": "active_exchange_closure_schedule",
            "closed_month_day_values": list(CLOSED_MONTH_DAYS),
            "reopening_month_day_values": list(REOPENING_MONTH_DAYS),
        }
    )


def _event_projection(
    publisher: str,
    title: str,
    publication_date: str,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": RULE_EFFECTIVE_EVENT_PROJECTION_SCHEMA,
            "event_id": "cn.mainboard-registration-first-listing.2023",
            "publisher": publisher,
            "document_title": title,
            "publication_date": publication_date,
            "event_date": "2023-04-10",
            "asserted_fact_id": "mainboard_registration_first_listings_held",
            "exchange_scope": ["SSE", "SZSE"],
        }
    )


def _trigger_notice_projection(
    *,
    exchange: str,
    title: str,
    document_number: str,
    attachment_url: str,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": RULE_NOTICE_TRIGGER_PROJECTION_SCHEMA,
            "exchange": exchange,
            "document_role": "trading_rule_notice",
            "notice_title": title,
            "document_number": document_number,
            "publication_date": "2023-02-17",
            "declared_effective_trigger_id": (
                "cn.mainboard-registration-effective-date.2023.v1"
            ),
            "primary_rule_attachment_url": attachment_url,
        }
    )


def _explicit_notice_projection(
    *,
    exchange: str,
    title: str,
    document_number: str,
    superseded_document_number: str,
    attachment_url: str,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": RULE_NOTICE_EXPLICIT_PROJECTION_SCHEMA,
            "exchange": exchange,
            "document_role": "trading_rule_notice",
            "notice_title": title,
            "document_number": document_number,
            "publication_date": "2026-04-24",
            "declared_effective_date": "2026-07-06",
            "superseded_document_number": superseded_document_number,
            "primary_rule_attachment_url": attachment_url,
        }
    )


def _inline_rule_projection(
    *,
    binding_id: str,
    exchange: str,
    title: str,
    document_number: str,
    publication_date: str,
    effective_from: str,
    effective_to_exclusive: str | None,
    legal_status: str,
    domain: str,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "exchange": exchange,
        "document_role": "trading_rule",
        "notice_title": title,
        "document_number": document_number,
        "publication_date": publication_date,
        "effective_from": effective_from,
        "effective_to_exclusive": effective_to_exclusive,
        "legal_status": legal_status,
    }
    if domain == "calendar":
        return _sealed(
            {
                "schema_version": INLINE_CALENDAR_RULE_PROJECTION_SCHEMA,
                **common,
                "clause_locator": "2.3.1",
                "weekday_open_days": list(WEEKDAY_OPEN_DAYS),
                "exchange_announced_closure_rule": CLOSURE_RULE,
            }
        )
    if domain == "clock":
        return _sealed(
            {
                "schema_version": INLINE_SESSION_CLOCK_PROJECTION_SCHEMA,
                **common,
                "clause_locator": "2.3.2",
                "scope_id": CLOCK_SCOPE_ID,
                "segments": [dict(item) for item in CLOCK_SEGMENTS],
                "excluded_scopes": list(CLOCK_EXCLUDED_SCOPES),
            }
        )
    raise EvidenceV2Error(f"unsupported inline rule domain for {binding_id}")


def _binary_profile(
    *,
    binding_id: str,
    exchange: str,
    notice_binding_id: str,
    document_title: str,
    document_number: str,
    publication_date: str,
    effective_from: str,
    effective_to_exclusive: str | None,
    legal_status: str,
    attachment_url: str,
    clause_locator: str,
    domain: str,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": RULE_BINARY_PROFILE_SCHEMA,
            "profile_id": f"{binding_id}.profile",
            "binding_id": binding_id,
            "exchange": exchange,
            "domain": domain,
            "notice_binding_id": notice_binding_id,
            "document_title": document_title,
            "document_number": document_number,
            "publication_date": publication_date,
            "effective_from": effective_from,
            "effective_to_exclusive": effective_to_exclusive,
            "legal_status": legal_status,
            "primary_official_attachment_url": attachment_url,
            "clause_locator": clause_locator,
            "scope_id": None if domain == "calendar" else CLOCK_SCOPE_ID,
            "weekday_open_days": (
                list(WEEKDAY_OPEN_DAYS) if domain == "calendar" else None
            ),
            "exchange_announced_closure_rule": (
                CLOSURE_RULE if domain == "calendar" else None
            ),
            "segments": (
                None if domain == "calendar" else [dict(item) for item in CLOCK_SEGMENTS]
            ),
            "excluded_scopes": (
                None if domain == "calendar" else list(CLOCK_EXCLUDED_SCOPES)
            ),
        }
    )


def _wayback_projection(
    *,
    timestamp: str,
    original_url: str,
    mime_type: str,
    digest: str,
    length: int,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": WAYBACK_CDX_PROJECTION_SCHEMA,
            "archive_timestamp": timestamp,
            "original_url": original_url,
            "status_code": 200,
            "mime_type": mime_type,
            "digest": digest,
            "length": length,
        }
    )


SSE_PRIOR_ATTACHMENT = (
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/"
    "10824490/files/dcbe58edb194451d93f19b1f7dd8fb4c.docx"
)
SSE_CURRENT_ATTACHMENT = (
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/"
    "10816492/files/704204728fe74fff89de4f16efda4791.docx"
)
SZSE_PRIOR_ATTACHMENT = (
    "http://docs.static.szse.cn/www/lawrules/index/rule/"
    "W020230217564423808793.pdf"
)
SZSE_CURRENT_ATTACHMENT = (
    "http://docs.static.szse.cn/www/lawrules/rule/allrules/bussiness/"
    "W020260424690713155663.pdf"
)

_CHINESE_CLOSURE_MARKERS = (
    ("1月1日（星期四）至1月3日（星期六）休市",),
    ("1月5日（星期一）起照常开市",),
    ("2月15日（星期日）至2月23日（星期一）休市",),
    ("2月24日（星期二）起照常开市",),
    ("4月4日（星期六）至4月6日（星期一）休市",),
    ("4月7日（星期二）起照常开市",),
    ("5月1日（星期五）至5月5日（星期二）休市",),
    ("5月6日（星期三）起照常开市",),
    ("6月19日（星期五）至6月21日（星期日）休市",),
    ("6月22日（星期一）起照常开市",),
    ("9月25日（星期五）至9月27日（星期日）休市",),
    ("9月28日（星期一）起照常开市",),
    ("10月1日（星期四）至10月7日（星期三）休市",),
    ("10月8日（星期四）起照常开市",),
)
_ENGLISH_CLOSURE_MARKERS = (
    ("StockMarketHolidaySchedule(2026)",),
    ("January1st(Thursday)toJanuary2nd(Friday)",),
    ("resume trading on January 5th (Monday)",),
    ("February16th(Monday)toFebruary23rd(Monday)",),
    ("resume trading on February 24th (Tuesday)",),
    ("April6th(Monday)",),
    ("resume trading on April 7th (Tuesday)",),
    ("May1st(Friday)toMay5th(Tuesday)",),
    ("resume trading on May 6th (Wednesday)",),
    ("June19th(Friday)",),
    ("resume trading on June 22nd (Monday)",),
    ("September25th(Friday)",),
    ("resume trading on September 28th (Monday)",),
    ("October1st(Thursday)toOctober7th(Wednesday)",),
    ("resume trading on October 8th (Thursday)",),
)


def _binding_specs() -> tuple[_BindingSpec, ...]:
    sse_prior_calendar = _binary_profile(
        binding_id="cn.sse.rule-binary.prior.2023.calendar.v1",
        exchange="SSE",
        notice_binding_id="cn.sse.rule-notice.prior.2023.v1",
        document_title="上海证券交易所交易规则（2023年修订）",
        document_number="上证发〔2023〕32号",
        publication_date="2023-02-17",
        effective_from="2023-04-10",
        effective_to_exclusive="2026-07-06",
        legal_status="superseded",
        attachment_url=SSE_PRIOR_ATTACHMENT,
        clause_locator="2.4.1",
        domain="calendar",
    )
    sse_prior_clock = _binary_profile(
        binding_id="cn.sse.rule-binary.prior.2023.clock.v1",
        exchange="SSE",
        notice_binding_id="cn.sse.rule-notice.prior.2023.v1",
        document_title="上海证券交易所交易规则（2023年修订）",
        document_number="上证发〔2023〕32号",
        publication_date="2023-02-17",
        effective_from="2023-04-10",
        effective_to_exclusive="2026-07-06",
        legal_status="superseded",
        attachment_url=SSE_PRIOR_ATTACHMENT,
        clause_locator="2.4.2",
        domain="clock",
    )
    sse_current_calendar = _binary_profile(
        binding_id="cn.sse.rule-binary.current.2026.calendar.v1",
        exchange="SSE",
        notice_binding_id="cn.sse.rule-notice.current.2026.v1",
        document_title="上海证券交易所交易规则（2026年修订）",
        document_number="上证发〔2026〕41号",
        publication_date="2026-04-24",
        effective_from="2026-07-06",
        effective_to_exclusive=None,
        legal_status="effective",
        attachment_url=SSE_CURRENT_ATTACHMENT,
        clause_locator="2.4.1",
        domain="calendar",
    )
    sse_current_clock = _binary_profile(
        binding_id="cn.sse.rule-binary.current.2026.clock.v1",
        exchange="SSE",
        notice_binding_id="cn.sse.rule-notice.current.2026.v1",
        document_title="上海证券交易所交易规则（2026年修订）",
        document_number="上证发〔2026〕41号",
        publication_date="2026-04-24",
        effective_from="2026-07-06",
        effective_to_exclusive=None,
        legal_status="effective",
        attachment_url=SSE_CURRENT_ATTACHMENT,
        clause_locator="2.4.2",
        domain="clock",
    )
    szse_prior_calendar = _binary_profile(
        binding_id="cn.szse.rule-binary.prior.2023.calendar.v1",
        exchange="SZSE",
        notice_binding_id="cn.szse.rule-notice.prior.2023.v1",
        document_title="深圳证券交易所交易规则（2023年修订）",
        document_number="深证上〔2023〕98号",
        publication_date="2023-02-17",
        effective_from="2023-04-10",
        effective_to_exclusive="2026-07-06",
        legal_status="superseded",
        attachment_url=SZSE_PRIOR_ATTACHMENT,
        clause_locator="2.3.1",
        domain="calendar",
    )
    szse_prior_clock = _binary_profile(
        binding_id="cn.szse.rule-binary.prior.2023.clock.v1",
        exchange="SZSE",
        notice_binding_id="cn.szse.rule-notice.prior.2023.v1",
        document_title="深圳证券交易所交易规则（2023年修订）",
        document_number="深证上〔2023〕98号",
        publication_date="2023-02-17",
        effective_from="2023-04-10",
        effective_to_exclusive="2026-07-06",
        legal_status="superseded",
        attachment_url=SZSE_PRIOR_ATTACHMENT,
        clause_locator="2.3.2",
        domain="clock",
    )
    szse_current_calendar = _binary_profile(
        binding_id="cn.szse.rule-binary.current.2026.calendar.v1",
        exchange="SZSE",
        notice_binding_id="cn.szse.rule-notice.current.2026.v1",
        document_title="深圳证券交易所交易规则（2026年修订）",
        document_number="深证上〔2026〕551号",
        publication_date="2026-04-24",
        effective_from="2026-07-06",
        effective_to_exclusive=None,
        legal_status="effective",
        attachment_url=SZSE_CURRENT_ATTACHMENT,
        clause_locator="2.3.1",
        domain="calendar",
    )
    szse_current_clock = _binary_profile(
        binding_id="cn.szse.rule-binary.current.2026.clock.v1",
        exchange="SZSE",
        notice_binding_id="cn.szse.rule-notice.current.2026.v1",
        document_title="深圳证券交易所交易规则（2026年修订）",
        document_number="深证上〔2026〕551号",
        publication_date="2026-04-24",
        effective_from="2026-07-06",
        effective_to_exclusive=None,
        legal_status="effective",
        attachment_url=SZSE_CURRENT_ATTACHMENT,
        clause_locator="2.3.2",
        domain="clock",
    )

    return (
        _BindingSpec(
            "cn.csrc.mainboard-registration-first-listing.2023.v1",
            "cn.csrc.mainboard-registration-event-html.2023.v1",
            "myquant-csrc-2023-mainboard-registration-implementation.html",
            "https://www.csrc.gov.cn/shanghai/c105566/c7402810/content.shtml",
            "https://www.csrc.gov.cn/shanghai/c105566/c7402810/content.shtml",
            "direct_https",
            _event_projection(
                "CSRC Shanghai Regulatory Bureau",
                "沪深交易所主板注册制首批企业上市仪式举行",
                "2023-04-10",
            ),
            (
                ("沪深交易所主板注册制首批企业上市仪式举行",),
                ("日期：2023-04-10",),
                ("2023年4月10日",),
                ("股票发行注册制改革全面落地",),
            ),
        ),
        _BindingSpec(
            "cn.sse.mainboard-registration-first-listing.2023.v1",
            "cn.sse.mainboard-registration-event-html.2023.v1",
            "myquant-sse-2023-mainboard-registration-implementation-en.html",
            "https://english.sse.com.cn/news/newsrelease/c/5719418.shtml",
            "https://english.sse.com.cn/news/newsrelease/c/5719418.shtml",
            "direct_https",
            _event_projection(
                "Shanghai Stock Exchange",
                (
                    "SSE Meets the Press on Listing Arrangements for the First Batch "
                    "of Companies under SSE and SZSE Main Board Registration System"
                ),
                "2023-04-04",
            ),
            (
                ("data-time=2023-04-04", "2023-04-04"),
                ("listing ceremony for the first batch of companies",),
                ("Monday, April 10, 2023",),
            ),
        ),
        _BindingSpec(
            "cn.sse.rule-notice.prior.2023.v1",
            "cn.sse.rule-notice-trigger-html.2023.v1",
            "myquant-sse-2023-trading-rules-notice.html",
            (
                "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/"
                "c_20250612_10824490.shtml"
            ),
            (
                "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/"
                "c_20250612_10824490.shtml"
            ),
            "direct_https",
            _trigger_notice_projection(
                exchange="SSE",
                title="上海证券交易所交易规则（2023年修订）",
                document_number="上证发〔2023〕32号",
                attachment_url=SSE_PRIOR_ATTACHMENT,
            ),
            (
                ("上海证券交易所交易规则（2023年修订）",),
                ("上证发〔2023〕32号",),
                ("2023年02月17日", "二〇二三年二月十七日"),
                ("首只主板股票上市首日起施行",),
                ("dcbe58edb194451d93f19b1f7dd8fb4c.docx",),
            ),
        ),
        _BindingSpec(
            "cn.sse.rule-binary.prior.2023.calendar.v1",
            "exact_byte_sha_to_code_frozen_profile_v1",
            "myquant-sse-2023-trading-rules.docx",
            SSE_PRIOR_ATTACHMENT,
            SSE_PRIOR_ATTACHMENT,
            "direct_https",
            sse_prior_calendar,
        ),
        _BindingSpec(
            "cn.sse.rule-binary.prior.2023.clock.v1",
            "exact_byte_sha_to_code_frozen_profile_v1",
            "myquant-sse-2023-trading-rules.docx",
            SSE_PRIOR_ATTACHMENT,
            SSE_PRIOR_ATTACHMENT,
            "direct_https",
            sse_prior_clock,
        ),
        _BindingSpec(
            "cn.sse.annual-closure-notice.2026.v1",
            "cn.sse.annual-closure-html.2026.v1",
            "myquant-sse-2026-closures.html",
            "https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml",
            "https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml",
            "direct_https",
            _annual_projection(
                "SSE",
                "关于上海证券交易所2026年部分节假日休市安排的通知",
                None,
            ),
            (
                ("关于上海证券交易所2026年部分节假日休市安排的通知",),
                ("2025-12-22", "2025年12月22日"),
                *_CHINESE_CLOSURE_MARKERS,
            ),
        ),
        _BindingSpec(
            "cn.sse.active-closure-schedule.2026.v1",
            "cn.sse.active-closure-html.2026.v1",
            "myquant-sse-2026-active-closures.html",
            "https://www.sse.com.cn/disclosure/dealinstruc/closed/",
            "https://www.sse.com.cn/disclosure/dealinstruc/closed/",
            "direct_https",
            _active_year_projection("SSE"),
            (("2026年休市安排",), *_CHINESE_CLOSURE_MARKERS),
        ),
        _BindingSpec(
            "cn.sse.rule-notice.current.2026.v1",
            "cn.sse.rule-notice-explicit-html.2026.v1",
            "myquant-sse-2026-trading-rules.html",
            (
                "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/"
                "c_20260424_10816492.shtml"
            ),
            (
                "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/"
                "c_20260424_10816492.shtml"
            ),
            "direct_https",
            _explicit_notice_projection(
                exchange="SSE",
                title="上海证券交易所交易规则（2026年修订）",
                document_number="上证发〔2026〕41号",
                superseded_document_number="上证发〔2023〕32号",
                attachment_url=SSE_CURRENT_ATTACHMENT,
            ),
            (
                ("上海证券交易所交易规则（2026年修订）",),
                ("上证发〔2026〕41号",),
                ("2026年04月24日", "2026年4月24日"),
                ("自2026年7月6日起施行",),
                ("上证发〔2023〕32号",),
                ("704204728fe74fff89de4f16efda4791.docx",),
            ),
        ),
        _BindingSpec(
            "cn.sse.rule-binary.current.2026.calendar.v1",
            "exact_byte_sha_to_code_frozen_profile_v1",
            "myquant-sse-2026-trading-rules.docx",
            SSE_CURRENT_ATTACHMENT,
            SSE_CURRENT_ATTACHMENT,
            "direct_https",
            sse_current_calendar,
        ),
        _BindingSpec(
            "cn.sse.rule-binary.current.2026.clock.v1",
            "exact_byte_sha_to_code_frozen_profile_v1",
            "myquant-sse-2026-trading-rules.docx",
            SSE_CURRENT_ATTACHMENT,
            SSE_CURRENT_ATTACHMENT,
            "direct_https",
            sse_current_clock,
        ),
        _BindingSpec(
            "cn.szse.mainboard-registration-first-listing.2023.v1",
            "cn.szse.mainboard-registration-event-html.2023.v1",
            "myquant-szse-2023-events.html",
            "https://www.szse.cn/aboutus/sse/events/t20240110_605593.html",
            "https://www.szse.cn/aboutus/sse/events/t20240110_605593.html",
            "direct_https",
            _event_projection(
                "Shenzhen Stock Exchange",
                "2023年深圳证券交易所大事记",
                "2024-01-10",
            ),
            (
                ("2023年深圳证券交易所大事记",),
                ("4月10日",),
                ("沪深交易所主板注册制首批企业上市仪式",),
                ("深交所主板注册制首批5家上市企业",),
            ),
        ),
        _BindingSpec(
            "cn.szse.rule-notice-prior.cdx.2023.v1",
            "cn.wayback-cdx-json.v1",
            "myquant-szse-2023-trading-rules-notice-wayback-cdx.json",
            "https://www.szse.cn/lawrules/index/rule/t20230217_598773.html",
            (
                "https://web.archive.org/cdx/search/cdx?url=https%3A%2F%2Fwww.szse.cn%2F"
                "lawrules%2Findex%2Frule%2Ft20230217_598773.html&output=json&fl="
                "timestamp%2Coriginal%2Cstatuscode%2Cmimetype%2Cdigest%2Clength&"
                "filter=statuscode%3A200&from=20241007104029&to=20241007104029"
            ),
            "wayback_cdx_api_exact_row_capture",
            _wayback_projection(
                timestamp="20241007104029",
                original_url="https://www.szse.cn/lawrules/index/rule/t20230217_598773.html",
                mime_type="text/html",
                digest="77WDTIVWIE4KTVCCRWDJT6VR5XY3Y6LP",
                length=4648,
            ),
        ),
        _BindingSpec(
            "cn.szse.rule-notice.prior.2023.v1",
            "cn.szse.rule-notice-trigger-html.2023.v1",
            "myquant-szse-2023-trading-rules-notice-wayback.html",
            "https://www.szse.cn/lawrules/index/rule/t20230217_598773.html",
            (
                "https://web.archive.org/web/20241007104029id_/https://www.szse.cn/"
                "lawrules/index/rule/t20230217_598773.html"
            ),
            "wayback_replay_exact_capture",
            _trigger_notice_projection(
                exchange="SZSE",
                title=(
                    "关于发布《深圳证券交易所交易规则"
                    "（2023年修订）》的通知"
                ),
                document_number="深证上〔2023〕98号",
                attachment_url=SZSE_PRIOR_ATTACHMENT,
            ),
            (
                (
                    "关于发布《深圳证券交易所交易规则"
                    "（2023年修订）》的通知",
                ),
                ("深证上〔2023〕98号",),
                ("2023-02-17",),
                ("首只主板股票上市首日起施行",),
                ("W020230217564423808793.pdf",),
            ),
        ),
        _BindingSpec(
            "cn.szse.rule-binary-prior.alias-cdx.2023.v1",
            "cn.wayback-cdx-json.v1",
            "myquant-szse-2023-trading-rules-wayback-alias-cdx.json",
            (
                "https://docs.static.szse.cn/www/lawrules/rule/stock/trade/"
                "W020230217564423808793.pdf"
            ),
            (
                "https://web.archive.org/cdx/search/cdx?url=https%3A%2F%2Fdocs.static."
                "szse.cn%2Fwww%2Flawrules%2Frule%2Fstock%2Ftrade%2F"
                "W020230217564423808793.pdf&output=json&fl=timestamp%2Coriginal%2C"
                "statuscode%2Cdigest%2Clength&filter=statuscode%3A200&from="
                "20240816054023&to=20240816054023"
            ),
            "wayback_cdx_api_exact_row_capture",
            _wayback_projection(
                timestamp="20240816054023",
                original_url=(
                    "https://docs.static.szse.cn/www/lawrules/rule/stock/trade/"
                    "W020230217564423808793.pdf"
                ),
                mime_type="application/pdf",
                digest="K7HEG6VII757DG3JCQGZGI57CBFUN3FA",
                length=537780,
            ),
        ),
        _BindingSpec(
            "cn.szse.rule-binary-prior.origin-cdx.2023.v1",
            "cn.wayback-cdx-json.v1",
            "myquant-szse-2023-trading-rules-wayback-cdx.json",
            SZSE_PRIOR_ATTACHMENT,
            (
                "https://web.archive.org/cdx/search/cdx?url=http%3A%2F%2Fdocs.static."
                "szse.cn%2Fwww%2Flawrules%2Findex%2Frule%2F"
                "W020230217564423808793.pdf&output=json&fl=timestamp%2Coriginal%2C"
                "statuscode%2Cdigest%2Clength&filter=statuscode%3A200&from="
                "20260116100725&to=20260116100725"
            ),
            "wayback_cdx_api_exact_row_capture",
            _wayback_projection(
                timestamp="20260116100725",
                original_url=SZSE_PRIOR_ATTACHMENT,
                mime_type="application/pdf",
                digest="K7HEG6VII757DG3JCQGZGI57CBFUN3FA",
                length=538071,
            ),
        ),
        _BindingSpec(
            "cn.szse.rule-binary.prior.2023.calendar.v1",
            "exact_byte_sha_to_code_frozen_profile_v1",
            "myquant-szse-2023-trading-rules-wayback.pdf",
            SZSE_PRIOR_ATTACHMENT,
            (
                "https://web.archive.org/web/20260116100725id_/http://docs.static."
                "szse.cn/www/lawrules/index/rule/W020230217564423808793.pdf"
            ),
            "wayback_replay_exact_capture",
            szse_prior_calendar,
        ),
        _BindingSpec(
            "cn.szse.rule-binary.prior.2023.clock.v1",
            "exact_byte_sha_to_code_frozen_profile_v1",
            "myquant-szse-2023-trading-rules-wayback.pdf",
            SZSE_PRIOR_ATTACHMENT,
            (
                "https://web.archive.org/web/20260116100725id_/http://docs.static."
                "szse.cn/www/lawrules/index/rule/W020230217564423808793.pdf"
            ),
            "wayback_replay_exact_capture",
            szse_prior_clock,
        ),
        _BindingSpec(
            "cn.szse.active-closure-schedule.2026.v1",
            "cn.szse.active-closure-html.2026.v1",
            "myquant-szse-2026-active-calendar-en.html",
            "https://www.szse.cn/English/services/trading/calendar/index.html",
            "https://www.szse.cn/English/services/trading/calendar/index.html",
            "direct_https",
            _active_year_projection("SZSE"),
            _ENGLISH_CLOSURE_MARKERS,
        ),
        _BindingSpec(
            "cn.szse.annual-closure-notice.2026.v1",
            "cn.szse.annual-closure-html.2026.v1",
            "myquant-szse-2026-closures-20260719.html",
            "https://www.szse.cn/disclosure/notice/t20251222_618087.html",
            "https://www.szse.cn/disclosure/notice/t20251222_618087.html",
            "direct_https",
            _annual_projection(
                "SZSE",
                "关于2026年部分节假日休市安排的通知",
                None,
            ),
            (
                ("关于2026年部分节假日休市安排的通知",),
                ("2025-12-22", "2025年12月22日"),
                *_CHINESE_CLOSURE_MARKERS,
            ),
        ),
        _BindingSpec(
            "cn.szse.rule-notice.current.2026.v1",
            "cn.szse.rule-notice-explicit-html.2026.v1",
            "myquant-szse-2026-trading-rules-notice.html",
            "https://www.szse.cn/lawrules/rule/allrules/bussiness/t20260424_620190.html",
            "https://www.szse.cn/lawrules/rule/allrules/bussiness/t20260424_620190.html",
            "direct_https",
            _explicit_notice_projection(
                exchange="SZSE",
                title=(
                    "关于发布《深圳证券交易所交易规则"
                    "（2026年修订）》的通知"
                ),
                document_number="深证上〔2026〕551号",
                superseded_document_number="深证上〔2023〕98号",
                attachment_url=SZSE_CURRENT_ATTACHMENT,
            ),
            (
                (
                    "关于发布《深圳证券交易所交易规则"
                    "（2026年修订）》的通知",
                ),
                ("深证上〔2026〕551号",),
                ("2026-04-24", "2026年4月24日"),
                ("2026年7月6日起施行",),
                ("深证上〔2023〕98号",),
                ("W020260424690713155663.pdf",),
            ),
        ),
        _BindingSpec(
            "cn.szse.rule-binary.current.2026.calendar.v1",
            "exact_byte_sha_to_code_frozen_profile_v1",
            "myquant-szse-2026-trading-rules.pdf",
            SZSE_CURRENT_ATTACHMENT,
            SZSE_CURRENT_ATTACHMENT.replace("http://", "https://", 1),
            "direct_https_scheme_upgrade_same_host_path",
            szse_current_calendar,
        ),
        _BindingSpec(
            "cn.szse.rule-binary.current.2026.clock.v1",
            "exact_byte_sha_to_code_frozen_profile_v1",
            "myquant-szse-2026-trading-rules.pdf",
            SZSE_CURRENT_ATTACHMENT,
            SZSE_CURRENT_ATTACHMENT.replace("http://", "https://", 1),
            "direct_https_scheme_upgrade_same_host_path",
            szse_current_clock,
        ),
        _BindingSpec(
            "cn.bse.annual-closure-notice.2026.v1",
            "cn.bse.annual-closure-html.2026.v1",
            "myquant-bse-2026-closures-annual.html",
            "https://www.bse.cn/important_news/200027428.html",
            "https://www.bse.cn/important_news/200027428.html",
            "controlled_browser_cdp_response_body",
            _annual_projection(
                "BSE",
                "关于2026年部分节假日休市安排的公告",
                "北证公告〔2025〕58号",
            ),
            (
                ("关于2026年部分节假日休市安排的公告",),
                ("北证公告〔2025〕58号",),
                ("2025-12-22", "2025年12月22日"),
                *_CHINESE_CLOSURE_MARKERS,
            ),
        ),
        _BindingSpec(
            "cn.bse.active-closure-schedule.2026.v1",
            "cn.bse.active-closure-html.month-day.2026.v1",
            "myquant-bse-2026-closures-active.html",
            "https://www.bse.cn/disclosure/Rest_arrangement.html",
            "https://www.bse.cn/disclosure/Rest_arrangement.html",
            "controlled_browser_cdp_response_body",
            _active_month_day_projection(),
            (("休市安排",), *_CHINESE_CLOSURE_MARKERS),
        ),
        _BindingSpec(
            "cn.bse.rule-inline.prior.2021.calendar.v1",
            "cn.bse.inline-calendar-rule-html.v1",
            "myquant-bse-2021-trading-rules-prior.html",
            "https://www.bse.cn/jygl_list/200010919.html",
            "https://www.bse.cn/jygl_list/200010919.html",
            "controlled_browser_cdp_response_body",
            _inline_rule_projection(
                binding_id="cn.bse.rule-inline.prior.2021.calendar.v1",
                exchange="BSE",
                title="关于发布《北京证券交易所交易规则（试行）》的公告",
                document_number="北证公告〔2021〕15号",
                publication_date="2021-11-02",
                effective_from="2021-11-15",
                effective_to_exclusive="2026-07-06",
                legal_status="modified",
                domain="calendar",
            ),
            (
                ("关于发布《北京证券交易所交易规则（试行）》的公告",),
                ("北证公告〔2021〕15号",),
                ("2021-11-02",),
                ("2021-11-15",),
                ("已被修改",),
                ("2.3.1",),
                ("本所交易日为每周一至周五",),
                ("国家法定节假日和本所公告的休市日",),
            ),
        ),
        _BindingSpec(
            "cn.bse.rule-inline.prior.2021.clock.v1",
            "cn.bse.inline-session-clock-rule-html.v1",
            "myquant-bse-2021-trading-rules-prior.html",
            "https://www.bse.cn/jygl_list/200010919.html",
            "https://www.bse.cn/jygl_list/200010919.html",
            "controlled_browser_cdp_response_body",
            _inline_rule_projection(
                binding_id="cn.bse.rule-inline.prior.2021.clock.v1",
                exchange="BSE",
                title="关于发布《北京证券交易所交易规则（试行）》的公告",
                document_number="北证公告〔2021〕15号",
                publication_date="2021-11-02",
                effective_from="2021-11-15",
                effective_to_exclusive="2026-07-06",
                legal_status="modified",
                domain="clock",
            ),
            (
                ("关于发布《北京证券交易所交易规则（试行）》的公告",),
                ("北证公告〔2021〕15号",),
                ("2.3.2",),
                ("9:15至9:25",),
                ("9:30至11:30",),
                ("13:00至14:57",),
                ("14:57至15:00",),
            ),
        ),
        _BindingSpec(
            "cn.bse.rule-inline.current.2026.calendar.v1",
            "cn.bse.inline-calendar-rule-html.v1",
            "myquant-bse-2026-trading-rules-current.html",
            "https://www.bse.cn/jygl_list/200028217.html",
            "https://www.bse.cn/jygl_list/200028217.html",
            "controlled_browser_cdp_response_body",
            _inline_rule_projection(
                binding_id="cn.bse.rule-inline.current.2026.calendar.v1",
                exchange="BSE",
                title="关于发布《北京证券交易所交易规则》的公告",
                document_number="北证公告〔2026〕17号",
                publication_date="2026-04-24",
                effective_from="2026-07-06",
                effective_to_exclusive=None,
                legal_status="effective",
                domain="calendar",
            ),
            (
                ("关于发布《北京证券交易所交易规则》的公告",),
                ("北证公告〔2026〕17号",),
                ("2026-04-24",),
                ("2026-07-06", "2026年7月6日起施行"),
                ("现行有效",),
                ("2.3.1",),
                ("本所交易日为每周一至周五",),
                ("国家法定节假日和本所公告的休市日",),
            ),
        ),
        _BindingSpec(
            "cn.bse.rule-inline.current.2026.clock.v1",
            "cn.bse.inline-session-clock-rule-html.v1",
            "myquant-bse-2026-trading-rules-current.html",
            "https://www.bse.cn/jygl_list/200028217.html",
            "https://www.bse.cn/jygl_list/200028217.html",
            "controlled_browser_cdp_response_body",
            _inline_rule_projection(
                binding_id="cn.bse.rule-inline.current.2026.clock.v1",
                exchange="BSE",
                title="关于发布《北京证券交易所交易规则》的公告",
                document_number="北证公告〔2026〕17号",
                publication_date="2026-04-24",
                effective_from="2026-07-06",
                effective_to_exclusive=None,
                legal_status="effective",
                domain="clock",
            ),
            (
                ("关于发布《北京证券交易所交易规则》的公告",),
                ("北证公告〔2026〕17号",),
                ("2.3.2",),
                ("9:15至9:25",),
                ("9:30至11:30",),
                ("13:00至14:57",),
                ("14:57至15:00",),
            ),
        ),
    )


BINDING_SPECS = _binding_specs()
BINDING_SPEC_BY_ID = {spec.binding_id: spec for spec in BINDING_SPECS}
if len(BINDING_SPECS) != 28 or len(BINDING_SPEC_BY_ID) != 28:
    raise RuntimeError("calendar source binding inventory must contain 28 unique rows")

_EXPECTED_BINARY_PROFILE_HASHES = {
    "cn.sse.rule-binary.prior.2023.calendar.v1": (
        "fdea863493328353a4f10b5872531ce06f344c6a5eb388b9a4872c16fc1ebea3"
    ),
    "cn.sse.rule-binary.prior.2023.clock.v1": (
        "961135960997a547f8edbd9866cee5d00f42872bd71b26c80e8c8d6e7165f753"
    ),
    "cn.sse.rule-binary.current.2026.calendar.v1": (
        "bad73f071527cf82954a9916e968232addac68ad6f9d7800dd5095c7a013b1ea"
    ),
    "cn.sse.rule-binary.current.2026.clock.v1": (
        "27cd90a2d7fdf8ee28928acf37e39aaf2ec8d4d047d38575a146f15a0317b1af"
    ),
    "cn.szse.rule-binary.prior.2023.calendar.v1": (
        "b3f67a7adc63396fdadbd32afd19c4072a8df79eb0df86b43ef7d6cbe728db03"
    ),
    "cn.szse.rule-binary.prior.2023.clock.v1": (
        "1332aa8a2f667c79481074d2bf7c4a97d93aaabc79f6643af774cacec52d816e"
    ),
    "cn.szse.rule-binary.current.2026.calendar.v1": (
        "eeda8bf8b67da5cd5829d8a369bed9b960b8dfe563fc9ff2eb25a3901bfe9bf1"
    ),
    "cn.szse.rule-binary.current.2026.clock.v1": (
        "1bac35b7ce4e82a1305c8d892d05fca94bc804cdc8f59b2afab231457e5108c0"
    ),
}
for _binding_id, _expected_hash in _EXPECTED_BINARY_PROFILE_HASHES.items():
    if BINDING_SPEC_BY_ID[_binding_id].semantic["semantic_sha256"] != _expected_hash:
        raise RuntimeError(f"frozen binary profile hash drift: {_binding_id}")


def _source_set(
    source_document_set_id: str,
    exchange_scope: Sequence[str],
    purpose: str,
    required_source_document_set_ids: Sequence[str],
    ordered_binding_ids: Sequence[str],
    combination_policy_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_DOCUMENT_SET_SCHEMA,
        "source_document_set_id": source_document_set_id,
        "exchange_scope": list(exchange_scope),
        "purpose": purpose,
        "required_source_document_set_ids": list(required_source_document_set_ids),
        "ordered_binding_ids": list(ordered_binding_ids),
        "combination_policy_id": combination_policy_id,
    }


SOURCE_DOCUMENT_SETS = (
    _source_set(
        "cn.sse.closure-sources.2026.v1",
        ("SSE",),
        "resolve_2026_sse_closed_weekdays",
        (),
        (
            "cn.sse.annual-closure-notice.2026.v1",
            "cn.sse.active-closure-schedule.2026.v1",
        ),
        "explicit_annual_and_active_schedule_exact_agreement_v1",
    ),
    _source_set(
        "cn.szse.closure-sources.2026.v1",
        ("SZSE",),
        "resolve_2026_szse_closed_weekdays",
        (),
        (
            "cn.szse.annual-closure-notice.2026.v1",
            "cn.szse.active-closure-schedule.2026.v1",
        ),
        "explicit_annual_and_active_schedule_exact_agreement_v1",
    ),
    _source_set(
        "cn.bse.closure-sources.2026.v1",
        ("BSE",),
        "resolve_2026_bse_closed_weekdays",
        (),
        (
            "cn.bse.annual-closure-notice.2026.v1",
            "cn.bse.active-closure-schedule.2026.v1",
        ),
        "explicit_annual_and_active_schedule_exact_agreement_v1",
    ),
    _source_set(
        "cn.mainboard-registration-effective-date.2023.v1",
        ("SSE", "SZSE"),
        "resolve_2023_mainboard_rule_effective_date",
        (),
        (
            "cn.csrc.mainboard-registration-first-listing.2023.v1",
            "cn.sse.mainboard-registration-first-listing.2023.v1",
            "cn.szse.mainboard-registration-first-listing.2023.v1",
        ),
        "three_official_events_unanimous_exact_date_v1",
    ),
    _source_set(
        "cn.sse.rule-notice-history.2023-2026.v1",
        ("SSE",),
        "resolve_sse_rule_notice_legal_intervals",
        ("cn.mainboard-registration-effective-date.2023.v1",),
        (
            "cn.sse.rule-notice.prior.2023.v1",
            "cn.sse.rule-notice.current.2026.v1",
        ),
        "trigger_resolved_gapless_notice_intervals_v1",
    ),
    _source_set(
        "cn.szse.rule-notice-history.2023-2026.v1",
        ("SZSE",),
        "resolve_szse_rule_notice_legal_intervals",
        ("cn.mainboard-registration-effective-date.2023.v1",),
        (
            "cn.szse.rule-notice-prior.cdx.2023.v1",
            "cn.szse.rule-notice.prior.2023.v1",
            "cn.szse.rule-binary-prior.alias-cdx.2023.v1",
            "cn.szse.rule-binary-prior.origin-cdx.2023.v1",
            "cn.szse.rule-notice.current.2026.v1",
        ),
        "wayback_digest_alias_and_trigger_resolved_gapless_notice_intervals_v1",
    ),
    _source_set(
        "cn.sse.calendar-rule-history.2023-2026.v1",
        ("SSE",),
        "resolve_sse_calendar_rule_history",
        ("cn.sse.rule-notice-history.2023-2026.v1",),
        (
            "cn.sse.rule-binary.prior.2023.calendar.v1",
            "cn.sse.rule-binary.current.2026.calendar.v1",
        ),
        "binary_profile_notice_attachment_and_gapless_interval_match_v1",
    ),
    _source_set(
        "cn.szse.calendar-rule-history.2023-2026.v1",
        ("SZSE",),
        "resolve_szse_calendar_rule_history",
        ("cn.szse.rule-notice-history.2023-2026.v1",),
        (
            "cn.szse.rule-binary.prior.2023.calendar.v1",
            "cn.szse.rule-binary.current.2026.calendar.v1",
        ),
        "binary_profile_notice_attachment_and_gapless_interval_match_v1",
    ),
    _source_set(
        "cn.bse.calendar-rule-history.2021-2026.v1",
        ("BSE",),
        "resolve_bse_calendar_rule_history",
        (),
        (
            "cn.bse.rule-inline.prior.2021.calendar.v1",
            "cn.bse.rule-inline.current.2026.calendar.v1",
        ),
        "inline_notice_clause_gapless_interval_match_v1",
    ),
    _source_set(
        "cn.sse.clock-rule-history.2023-2026.v1",
        ("SSE",),
        "resolve_sse_clock_rule_history",
        ("cn.sse.rule-notice-history.2023-2026.v1",),
        (
            "cn.sse.rule-binary.prior.2023.clock.v1",
            "cn.sse.rule-binary.current.2026.clock.v1",
        ),
        "binary_profile_notice_attachment_and_gapless_interval_match_v1",
    ),
    _source_set(
        "cn.szse.clock-rule-history.2023-2026.v1",
        ("SZSE",),
        "resolve_szse_clock_rule_history",
        ("cn.szse.rule-notice-history.2023-2026.v1",),
        (
            "cn.szse.rule-binary.prior.2023.clock.v1",
            "cn.szse.rule-binary.current.2026.clock.v1",
        ),
        "binary_profile_notice_attachment_and_gapless_interval_match_v1",
    ),
    _source_set(
        "cn.bse.clock-rule-history.2021-2026.v1",
        ("BSE",),
        "resolve_bse_clock_rule_history",
        (),
        (
            "cn.bse.rule-inline.prior.2021.clock.v1",
            "cn.bse.rule-inline.current.2026.clock.v1",
        ),
        "inline_notice_clause_gapless_interval_match_v1",
    ),
    _source_set(
        OPEN_SESSION_CALENDAR_ID,
        ("SSE", "SZSE", "BSE"),
        "compile_2026_cn_open_sessions",
        (
            "cn.sse.closure-sources.2026.v1",
            "cn.szse.closure-sources.2026.v1",
            "cn.bse.closure-sources.2026.v1",
            "cn.sse.calendar-rule-history.2023-2026.v1",
            "cn.szse.calendar-rule-history.2023-2026.v1",
            "cn.bse.calendar-rule-history.2021-2026.v1",
        ),
        (),
        "three_exchange_2026_calendar_exact_intersection_v1",
    ),
    _source_set(
        "cn.listed-equity-auction-clock.2026.v1",
        ("SSE", "SZSE", "BSE"),
        "compile_cn_listed_equity_auction_clock",
        (
            "cn.sse.clock-rule-history.2023-2026.v1",
            "cn.szse.clock-rule-history.2023-2026.v1",
            "cn.bse.clock-rule-history.2021-2026.v1",
        ),
        (),
        "three_exchange_current_clock_exact_agreement_v1",
    ),
)
SOURCE_DOCUMENT_SET_BY_ID = {
    value["source_document_set_id"]: value for value in SOURCE_DOCUMENT_SETS
}
if len(SOURCE_DOCUMENT_SETS) != 14 or len(SOURCE_DOCUMENT_SET_BY_ID) != 14:
    raise RuntimeError("calendar document-set inventory must contain 14 unique rows")

CALENDAR_SOURCE_SET_IDS = (
    "cn.sse.closure-sources.2026.v1",
    "cn.szse.closure-sources.2026.v1",
    "cn.bse.closure-sources.2026.v1",
    "cn.mainboard-registration-effective-date.2023.v1",
    "cn.sse.rule-notice-history.2023-2026.v1",
    "cn.szse.rule-notice-history.2023-2026.v1",
    "cn.sse.calendar-rule-history.2023-2026.v1",
    "cn.szse.calendar-rule-history.2023-2026.v1",
    "cn.bse.calendar-rule-history.2021-2026.v1",
    OPEN_SESSION_CALENDAR_ID,
)
CALENDAR_BINDING_IDS = tuple(
    spec.binding_id
    for spec in BINDING_SPECS
    if not spec.binding_id.endswith(".clock.v1")
)


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _compact(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFC", value)
        if not character.isspace()
    )


def _parse_html_semantics(spec: _BindingSpec, payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceV2Error(f"{spec.binding_id} HTML is not UTF-8") from exc
    parser = _SemanticHTMLParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise EvidenceV2Error(f"{spec.binding_id} HTML parse failed") from exc
    compact = parser.compact_text
    if not compact:
        raise EvidenceV2Error(f"{spec.binding_id} HTML has no semantic text")
    for index, alternatives in enumerate(spec.marker_groups):
        if not alternatives or not any(_compact(marker) in compact for marker in alternatives):
            raise EvidenceV2Error(
                f"{spec.binding_id} semantic marker group {index} is missing"
            )
    return validate_semantic_seal(spec.semantic)


def _parse_cdx_semantics(spec: _BindingSpec, payload: bytes) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceV2Error(f"{spec.binding_id} CDX JSON is not UTF-8") from exc
    try:
        raw = json.loads(decoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceV2Error(f"{spec.binding_id} CDX JSON is invalid") from exc
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not all(isinstance(row, list) for row in raw)
        or len(raw[0]) != len(raw[1])
        or not all(isinstance(item, str) for row in raw for item in row)
    ):
        raise EvidenceV2Error(f"{spec.binding_id} CDX response shape mismatch")
    header = raw[0]
    if len(header) != len(set(header)):
        raise EvidenceV2Error(f"{spec.binding_id} CDX header is not unique")
    row = dict(zip(header, raw[1]))
    expected_fields = {"timestamp", "original", "statuscode", "digest", "length"}
    if set(row) != expected_fields and set(row) != expected_fields | {"mimetype"}:
        raise EvidenceV2Error(f"{spec.binding_id} CDX fields mismatch")
    if row["statuscode"] != "200" or not row["length"].isdigit():
        raise EvidenceV2Error(f"{spec.binding_id} CDX selected row is not HTTP 200")
    mime_type = row.get("mimetype")
    if mime_type is None:
        if not row["original"].lower().endswith(".pdf"):
            raise EvidenceV2Error(f"{spec.binding_id} CDX MIME type is absent")
        mime_type = "application/pdf"
    projection = _wayback_projection(
        timestamp=row["timestamp"],
        original_url=row["original"],
        mime_type=mime_type,
        digest=row["digest"],
        length=int(row["length"]),
    )
    if projection != dict(spec.semantic):
        raise EvidenceV2Error(f"{spec.binding_id} CDX semantic projection drift")
    return projection


def parse_source_semantics(binding_id: str, payload: bytes) -> dict[str, Any]:
    """Parse one reviewed source without accepting caller-supplied semantics."""

    spec = BINDING_SPEC_BY_ID.get(str(binding_id))
    if spec is None:
        raise EvidenceV2Error("source binding ID is not in the frozen inventory")
    if not isinstance(payload, bytes) or not payload:
        raise EvidenceV2Error(f"{binding_id} source bytes are empty")
    if spec.parser_contract_id == "exact_byte_sha_to_code_frozen_profile_v1":
        inventory = SOURCE_INVENTORY_BY_NAME[spec.filename]
        if sha256_bytes(payload) != inventory.byte_sha256:
            raise EvidenceV2Error(f"{binding_id} binary source byte SHA drift")
        return validate_semantic_seal(spec.semantic)
    if spec.parser_contract_id == "cn.wayback-cdx-json.v1":
        return _parse_cdx_semantics(spec, payload)
    return _parse_html_semantics(spec, payload)


def _canonical_root(root: str | Path) -> Path:
    path = Path(root)
    text = str(path)
    if (
        not path.is_absolute()
        or "\x00" in text
        or os.path.normpath(text) != text
        or text.startswith("//")
        or text.endswith("/")
    ):
        raise EvidenceV2Error("calendar source root must be a canonical absolute path")
    return path


def _declared_reference(root: Path, spec: _BindingSpec) -> EvidenceRef:
    inventory = SOURCE_INVENTORY_BY_NAME[spec.filename]
    semantic = validate_semantic_seal(spec.semantic)
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=str(semantic["schema_version"]),
        absolute_path=str(root / spec.filename),
        byte_sha256=inventory.byte_sha256,
        semantic_sha256=str(semantic["semantic_sha256"]),
        root_policy=PRIVATE_ROOT_POLICY,
    )


def _binding_payload(spec: _BindingSpec, reference: EvidenceRef) -> dict[str, Any]:
    semantic = validate_semantic_seal(spec.semantic)
    is_profile = semantic["schema_version"] == RULE_BINARY_PROFILE_SCHEMA
    return {
        "schema_version": SOURCE_BINDING_SCHEMA,
        "source_binding_id": spec.binding_id,
        "parser_contract_id": spec.parser_contract_id,
        "raw_ref": reference.to_dict(),
        "declared_origin_url": spec.declared_origin_url,
        "retrieval_url": spec.retrieval_url,
        "retrieval_method": spec.retrieval_method,
        "semantic_projection": None if is_profile else semantic,
        "semantic_projection_sha256": semantic["semantic_sha256"],
        "selected_profile": semantic if is_profile else None,
        "authority_scope": SOURCE_AUTHORITY_SCOPE,
    }


def declared_source_bindings(root: str | Path) -> tuple[dict[str, Any], ...]:
    """Return the exact 28-row declaration without opening any source file."""

    root_path = _canonical_root(root)
    return tuple(
        _binding_payload(spec, _declared_reference(root_path, spec))
        for spec in BINDING_SPECS
    )


def validate_source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "source_binding_id",
        "parser_contract_id",
        "raw_ref",
        "declared_origin_url",
        "retrieval_url",
        "retrieval_method",
        "semantic_projection",
        "semantic_projection_sha256",
        "selected_profile",
        "authority_scope",
    }
    payload = _exact(value, fields, label="source binding")
    if payload["schema_version"] != SOURCE_BINDING_SCHEMA:
        raise EvidenceV2Error("unsupported source binding schema")
    binding_id = str(payload["source_binding_id"])
    spec = BINDING_SPEC_BY_ID.get(binding_id)
    if spec is None:
        raise EvidenceV2Error("source binding is outside the frozen registry")
    for field in (
        "parser_contract_id",
        "declared_origin_url",
        "retrieval_url",
        "retrieval_method",
    ):
        if payload[field] != getattr(spec, field):
            raise EvidenceV2Error(f"{binding_id} {field} drift")
    if payload["authority_scope"] != SOURCE_AUTHORITY_SCOPE:
        raise EvidenceV2Error(f"{binding_id} authority scope drift")
    reference = EvidenceRef.from_dict(payload["raw_ref"])
    inventory = SOURCE_INVENTORY_BY_NAME[spec.filename]
    semantic = validate_semantic_seal(spec.semantic)
    if (
        Path(reference.absolute_path).name != spec.filename
        or reference.byte_sha256 != inventory.byte_sha256
        or reference.semantic_sha256 != semantic["semantic_sha256"]
        or reference.artifact_schema != semantic["schema_version"]
        or reference.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error(f"{binding_id} raw EvidenceRef drift")
    is_profile = semantic["schema_version"] == RULE_BINARY_PROFILE_SCHEMA
    if is_profile:
        if payload["semantic_projection"] is not None:
            raise EvidenceV2Error(f"{binding_id} binary projection must be null")
        selected = validate_semantic_seal(payload["selected_profile"])
        if selected != semantic:
            raise EvidenceV2Error(f"{binding_id} selected profile drift")
    else:
        if payload["selected_profile"] is not None:
            raise EvidenceV2Error(f"{binding_id} selected profile must be null")
        selected = validate_semantic_seal(payload["semantic_projection"])
        if selected != semantic:
            raise EvidenceV2Error(f"{binding_id} semantic projection drift")
    if payload["semantic_projection_sha256"] != semantic["semantic_sha256"]:
        raise EvidenceV2Error(f"{binding_id} semantic projection SHA drift")
    payload["raw_ref"] = reference.to_dict()
    payload["semantic_projection"] = None if is_profile else selected
    payload["selected_profile"] = selected if is_profile else None
    return payload


def validate_source_document_set(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "source_document_set_id",
        "exchange_scope",
        "purpose",
        "required_source_document_set_ids",
        "ordered_binding_ids",
        "combination_policy_id",
    }
    payload = _exact(value, fields, label="source document set")
    set_id = str(payload["source_document_set_id"])
    expected = SOURCE_DOCUMENT_SET_BY_ID.get(set_id)
    if expected is None or payload != expected:
        raise EvidenceV2Error(f"source document set drift: {set_id}")
    return payload


def _binding_from_source(source: BoundSource) -> dict[str, Any]:
    spec = BINDING_SPEC_BY_ID.get(source.binding_id)
    if spec is None:
        raise EvidenceV2Error("bound source is outside the frozen registry")
    reference = source.artifact.reference
    expected_reference = _declared_reference(
        Path(reference.absolute_path).parent,
        spec,
    )
    if reference != expected_reference:
        raise EvidenceV2Error(f"{spec.binding_id} bound raw reference drift")
    semantic = parse_source_semantics(spec.binding_id, source.artifact.payload)
    if semantic_sha256(semantic) != reference.semantic_sha256:
        raise EvidenceV2Error(f"{spec.binding_id} parsed semantic SHA drift")
    return _binding_payload(spec, reference)


def _calendar_payload_from_bindings(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = [validate_source_binding(value) for value in bindings]
    by_id = {str(value["source_binding_id"]): value for value in normalized}
    if len(by_id) != len(normalized) or tuple(by_id) != CALENDAR_BINDING_IDS:
        raise EvidenceV2Error("calendar must bind the exact ordered 22-source registry")
    source_roots = {
        str(Path(EvidenceRef.from_dict(value["raw_ref"]).absolute_path).parent)
        for value in normalized
    }
    if len(source_roots) != 1:
        raise EvidenceV2Error("calendar source bindings must share one explicit root")

    closure_pairs = (
        (
            "cn.sse.annual-closure-notice.2026.v1",
            "cn.sse.active-closure-schedule.2026.v1",
        ),
        (
            "cn.szse.annual-closure-notice.2026.v1",
            "cn.szse.active-closure-schedule.2026.v1",
        ),
    )
    for annual_id, active_id in closure_pairs:
        annual = by_id[annual_id]["semantic_projection"]
        active = by_id[active_id]["semantic_projection"]
        if (
            annual["closed_weekday_dates"] != active["closed_weekday_dates"]
            or annual["reopening_dates"] != active["reopening_dates"]
        ):
            raise EvidenceV2Error("annual and active closure sources disagree")
    bse_annual = by_id["cn.bse.annual-closure-notice.2026.v1"][
        "semantic_projection"
    ]
    bse_active = by_id["cn.bse.active-closure-schedule.2026.v1"][
        "semantic_projection"
    ]
    if (
        [value[5:] for value in bse_annual["closed_weekday_dates"]]
        != bse_active["closed_month_day_values"]
        or [value[5:] for value in bse_annual["reopening_dates"]]
        != bse_active["reopening_month_day_values"]
    ):
        raise EvidenceV2Error("BSE annual and active closure sources disagree")

    event_ids = (
        "cn.csrc.mainboard-registration-first-listing.2023.v1",
        "cn.sse.mainboard-registration-first-listing.2023.v1",
        "cn.szse.mainboard-registration-first-listing.2023.v1",
    )
    if {
        by_id[binding_id]["semantic_projection"]["event_date"]
        for binding_id in event_ids
    } != {"2023-04-10"}:
        raise EvidenceV2Error("official mainboard effective-date events disagree")

    document_sets = [
        dict(SOURCE_DOCUMENT_SET_BY_ID[set_id]) for set_id in CALENDAR_SOURCE_SET_IDS
    ]
    return seal_semantic(
        {
            "schema_version": OPEN_SESSION_CALENDAR_SCHEMA,
            "calendar_id": OPEN_SESSION_CALENDAR_ID,
            "market": "CN",
            "calendar_year": CALENDAR_YEAR,
            "timezone": CALENDAR_TIMEZONE,
            "open_sessions": list(OPEN_SESSIONS),
            "closed_weekday_dates": list(CLOSED_WEEKDAY_DATES),
            "reopening_dates": list(REOPENING_DATES),
            "source_document_sets": document_sets,
            "source_bindings": normalized,
            "authority_scope": CALENDAR_AUTHORITY_SCOPE,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def build_open_session_calendar(
    sources: Sequence[BoundSource],
) -> dict[str, Any]:
    by_id: dict[str, BoundSource] = {}
    for source in sources:
        if not isinstance(source, BoundSource) or source.binding_id in by_id:
            raise EvidenceV2Error("calendar bound sources are invalid or duplicated")
        by_id[source.binding_id] = source
    if set(by_id) != set(CALENDAR_BINDING_IDS) and set(by_id) != set(
        BINDING_SPEC_BY_ID
    ):
        raise EvidenceV2Error("calendar bound source set is incomplete or contains extras")
    if len({str(Path(source.artifact.reference.absolute_path).parent) for source in sources}) != 1:
        raise EvidenceV2Error("calendar bound sources must share one explicit root")
    bindings = [_binding_from_source(by_id[binding_id]) for binding_id in CALENDAR_BINDING_IDS]
    return _calendar_payload_from_bindings(bindings)


def build_declared_open_session_calendar(root: str | Path) -> dict[str, Any]:
    """Build the declaration only; this is not source acceptance."""

    all_bindings = declared_source_bindings(root)
    by_id = {str(item["source_binding_id"]): item for item in all_bindings}
    return _calendar_payload_from_bindings(
        [by_id[binding_id] for binding_id in CALENDAR_BINDING_IDS]
    )


def validate_open_session_calendar(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "calendar_id",
        "market",
        "calendar_year",
        "timezone",
        "open_sessions",
        "closed_weekday_dates",
        "reopening_dates",
        "source_document_sets",
        "source_bindings",
        "authority_scope",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="open-session calendar")
    if (
        payload["schema_version"] != OPEN_SESSION_CALENDAR_SCHEMA
        or payload["calendar_id"] != OPEN_SESSION_CALENDAR_ID
        or payload["market"] != "CN"
        or payload["calendar_year"] != CALENDAR_YEAR
        or payload["timezone"] != CALENDAR_TIMEZONE
        or payload["open_sessions"] != list(OPEN_SESSIONS)
        or payload["closed_weekday_dates"] != list(CLOSED_WEEKDAY_DATES)
        or payload["reopening_dates"] != list(REOPENING_DATES)
        or payload["authority_scope"] != CALENDAR_AUTHORITY_SCOPE
    ):
        raise EvidenceV2Error("open-session calendar frozen values drift")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("open-session calendar must be permanently nonauthorizing")
    sets = [validate_source_document_set(item) for item in payload["source_document_sets"]]
    if [item["source_document_set_id"] for item in sets] != list(
        CALENDAR_SOURCE_SET_IDS
    ):
        raise EvidenceV2Error("open-session calendar source-set order drift")
    bindings = [validate_source_binding(item) for item in payload["source_bindings"]]
    if [item["source_binding_id"] for item in bindings] != list(CALENDAR_BINDING_IDS):
        raise EvidenceV2Error("open-session calendar binding order drift")
    rebuilt = _calendar_payload_from_bindings(bindings)
    if rebuilt != payload:
        raise EvidenceV2Error("open-session calendar does not recompute exactly")
    return payload


_EXPECTED_ALIAS_GROUPS = frozenset(
    {
        frozenset(
            {
                "cn.sse.rule-binary.prior.2023.calendar.v1",
                "cn.sse.rule-binary.prior.2023.clock.v1",
            }
        ),
        frozenset(
            {
                "cn.sse.rule-binary.current.2026.calendar.v1",
                "cn.sse.rule-binary.current.2026.clock.v1",
            }
        ),
        frozenset(
            {
                "cn.szse.rule-binary.prior.2023.calendar.v1",
                "cn.szse.rule-binary.prior.2023.clock.v1",
            }
        ),
        frozenset(
            {
                "cn.szse.rule-binary.current.2026.calendar.v1",
                "cn.szse.rule-binary.current.2026.clock.v1",
            }
        ),
        frozenset(
            {
                "cn.bse.rule-inline.prior.2021.calendar.v1",
                "cn.bse.rule-inline.prior.2021.clock.v1",
            }
        ),
        frozenset(
            {
                "cn.bse.rule-inline.current.2026.calendar.v1",
                "cn.bse.rule-inline.current.2026.clock.v1",
            }
        ),
    }
)


def _check_inventory_directory(root: Path) -> None:
    try:
        names = set(os.listdir(root))
    except OSError as exc:
        raise EvidenceV2Error("calendar source inventory directory is unreadable") from exc
    expected = set(SOURCE_INVENTORY_BY_NAME)
    if names != expected:
        raise EvidenceV2Error("calendar source inventory names drift")
    for filename in EXCLUDED_SOURCE_NAMES:
        entry = SOURCE_INVENTORY_BY_NAME[filename]
        try:
            metadata = os.lstat(root / filename)
        except OSError as exc:
            raise EvidenceV2Error(f"excluded source metadata missing: {filename}") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size != entry.size
        ):
            raise EvidenceV2Error(f"excluded source metadata drift: {filename}")


def load_private_source_bindings(root: str | Path) -> tuple[BoundSource, ...]:
    """Open the 22 consumed files once and distribute the six approved aliases."""

    root_path = _canonical_root(root)
    _check_inventory_directory(root_path)
    groups: dict[tuple[str, str, str], list[_BindingSpec]] = {}
    for spec in BINDING_SPECS:
        inventory = SOURCE_INVENTORY_BY_NAME[spec.filename]
        key = (spec.filename, inventory.byte_sha256, PRIVATE_ROOT_POLICY)
        groups.setdefault(key, []).append(spec)
    alias_groups = frozenset(
        frozenset(spec.binding_id for spec in specs)
        for specs in groups.values()
        if len(specs) > 1
    )
    if alias_groups != _EXPECTED_ALIAS_GROUPS or len(groups) != 22:
        raise EvidenceV2Error("physical source alias registry drift")

    by_id: dict[str, BoundSource] = {}
    for specs in groups.values():
        first = specs[0]
        first_reference = _declared_reference(root_path, first)
        loaded = load_bound_raw_artifact(
            root=root_path,
            reference=first_reference,
            policy=PRIVATE_EVIDENCE_POLICY,
            max_bytes=SOURCE_INVENTORY_BY_NAME[first.filename].size,
        )
        for spec in specs:
            reference = _declared_reference(root_path, spec)
            by_id[spec.binding_id] = BoundSource(
                binding_id=spec.binding_id,
                artifact=BoundRawArtifact(reference=reference, payload=loaded.payload),
            )
    if tuple(by_id) != tuple(spec.binding_id for spec in BINDING_SPECS):
        raise EvidenceV2Error("loaded source binding order drift")
    return tuple(by_id[binding_id] for binding_id in by_id)


def validate_private_calendar_acceptance(root: str | Path) -> dict[str, Any]:
    return validate_open_session_calendar(
        build_open_session_calendar(load_private_source_bindings(root))
    )


def validate_private_calendar_clock_acceptance(
    root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate calendar and clock from one shared, single-read source load."""

    from .session_clock import build_session_clock, validate_session_clock

    sources = load_private_source_bindings(root)
    calendar = validate_open_session_calendar(build_open_session_calendar(sources))
    clock = validate_session_clock(build_session_clock(sources))
    return calendar, clock


def bind_calendar_artifact(
    value: Mapping[str, Any],
    *,
    absolute_path: str,
) -> BoundCanonicalArtifact:
    payload = validate_open_session_calendar(value)
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=OPEN_SESSION_CALENDAR_SCHEMA,
            absolute_path=absolute_path,
            byte_sha256=sha256_bytes(raw),
            semantic_sha256=str(payload["semantic_sha256"]),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=raw,
    )


__all__ = [
    "ACTIVE_CLOSURE_MONTH_DAY_PROJECTION_SCHEMA",
    "ACTIVE_CLOSURE_YEAR_PROJECTION_SCHEMA",
    "ANNUAL_CLOSURE_PROJECTION_SCHEMA",
    "BINDING_SPECS",
    "BoundSource",
    "CALENDAR_BINDING_IDS",
    "CALENDAR_SOURCE_SET_IDS",
    "CALENDAR_TIMEZONE",
    "CalendarEvidenceBundle",
    "CLOCK_EXCLUDED_SCOPES",
    "CLOCK_SCOPE_ID",
    "CLOCK_SEGMENTS",
    "CLOSED_WEEKDAY_DATES",
    "INLINE_CALENDAR_RULE_PROJECTION_SCHEMA",
    "INLINE_SESSION_CLOCK_PROJECTION_SCHEMA",
    "OPEN_SESSIONS",
    "OPEN_SESSION_CALENDAR_ID",
    "OPEN_SESSION_CALENDAR_SCHEMA",
    "REOPENING_DATES",
    "RULE_BINARY_PROFILE_SCHEMA",
    "RULE_EFFECTIVE_EVENT_PROJECTION_SCHEMA",
    "RULE_NOTICE_EXPLICIT_PROJECTION_SCHEMA",
    "RULE_NOTICE_TRIGGER_PROJECTION_SCHEMA",
    "SOURCE_AUTHORITY_SCOPE",
    "SOURCE_BINDING_SCHEMA",
    "SOURCE_DOCUMENT_SETS",
    "SOURCE_DOCUMENT_SET_SCHEMA",
    "SOURCE_INVENTORY",
    "WAYBACK_CDX_PROJECTION_SCHEMA",
    "WEEKDAY_OPEN_DAYS",
    "bind_calendar_artifact",
    "build_declared_open_session_calendar",
    "build_open_session_calendar",
    "declared_source_bindings",
    "load_private_source_bindings",
    "parse_source_semantics",
    "validate_open_session_calendar",
    "validate_private_calendar_acceptance",
    "validate_private_calendar_clock_acceptance",
    "validate_source_binding",
    "validate_source_document_set",
]
