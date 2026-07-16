"""Compile fixed official CN macro HTML captures into replayable observations.

This module is deliberately offline.  It accepts an immutable plan, a
hash-bound capture manifest, and the exact response entities captured from the
National Bureau of Statistics (NBS) and the People's Bank of China (PBC).  It
does not discover URLs, fetch pages, publish observations, or substitute a
different provider.

The compiler is intentionally narrow: four versioned parser contracts produce
twelve official national indicators, with three periods per indicator.  A
bundle is publishable only when the complete 36-observation official scope
recompiles without ambiguity.  Rounded social-financing cumulative values are
validated as co-page evidence but are never differenced into an observation.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

from quant_investor.macro.contracts import (
    MacroObservation,
    canonical_hash,
    normalize_source_url,
    parse_timestamp,
)
from quant_investor.macro.nbs_pmi import (
    NBS_PMI_PARSER_CONTRACT_SHA256 as NBS_PMI_V2_CONTRACT_SHA256,
)
from quant_investor.macro.registry import NATIONAL_INDICATORS

OFFICIAL_WEB_PLAN_SCHEMA = "macro-official-web-plan.v1"
OFFICIAL_WEB_CAPTURE_SCHEMA = "macro-official-web-capture.v1"
OFFICIAL_WEB_NORMALIZATION_SCHEMA = "macro-official-web-normalization.v1"
OFFICIAL_WEB_RECEIPT_SCHEMA = "macro-official-web-receipt.v1"
OFFICIAL_WEB_EVIDENCE_SCHEMA = "macro-official-web-evidence.v1"

NBS_NATIONAL_ECONOMY_PARSER = "nbs-national-economy-html.v1"
NBS_OFFICIAL_PMI_PARSER = "nbs-cn-pmi-html.v3"
NBS_QUARTERLY_GDP_PARSER = "nbs-quarterly-gdp-html.v1"
PBC_MONEY_STOCK_PARSER = "pbc-financial-statistics-html.v1"
PBC_FINANCIAL_STATISTICS_PARSER = PBC_MONEY_STOCK_PARSER

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_NBS_RECORD_RE = re.compile(r"^(t(?P<date>20\d{6})_(?P<serial>\d+))\.html$")
_PBC_RECORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{3,79}$")
_PUB_DATE_RE = re.compile(
    r"(?P<year>20\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"
    r"[ T](?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
)
_NUMBER_RE = r"(?:0|[1-9]\d*)(?:\.\d+)?"
_PERCENT_RE = rf"(?P<direction>增长|上升|上涨|下降|下跌)(?P<value>{_NUMBER_RE})[%％]"
_MAX_HTML_BYTES = 4 * 1024 * 1024
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_JSONL_BYTES = 8 * 1024 * 1024


class OfficialWebCompilerError(RuntimeError):
    """Raised when an official page or evidence bundle fails closed."""


def _contract_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_NBS_NATIONAL_ECONOMY_CONTRACT = {
    "article_meta": ["ArticleTitle", "PubDate"],
    "body_encoding": "utf-8-strict",
    "issuer": "nbs_official",
    "measurement_basis": {
        "cn.cpi_yoy": "current-month national CPI YoY",
        "cn.exports_yoy": "current-month CNY exports YoY",
        "cn.fixed_asset_investment_yoy": "Jan-to-month FAI ex-rural cumulative YoY",
        "cn.gdp_yoy": "Q2 real GDP YoY only from formal half-year quarter sequence",
        "cn.imports_yoy": "current-month CNY imports YoY",
        "cn.industrial_value_added_yoy": "current-month real industrial VA YoY",
        "cn.ppi_yoy": "current-month national producer output-price YoY",
        "cn.property_investment_yoy": "Jan-to-month property investment cumulative YoY",
        "cn.retail_sales_yoy": "current-month nominal retail sales YoY",
    },
    "period_source": "formal release paragraphs",
    "publication_source": "PubDate Asia/Shanghai with URL record-date equality",
    "unique_value_per_indicator": True,
    "version": NBS_NATIONAL_ECONOMY_PARSER,
}
_NBS_QUARTERLY_GDP_CONTRACT = {
    "article_meta": ["ArticleTitle", "PubDate"],
    "body_encoding": "utf-8-strict",
    "issuer": "nbs_official",
    "measurement_basis": "current-quarter real GDP YoY, never cumulative YTD",
    "period_source": "formal ArticleTitle target quarter",
    "publication_source": "PubDate Asia/Shanghai with URL record-date equality",
    "unique_value": True,
    "version": NBS_QUARTERLY_GDP_PARSER,
}
_NBS_OFFICIAL_PMI_CONTRACT = {
    "article_title": "exact YYYY年M月中国采购经理指数运行情况",
    "body_encoding": "utf-8-strict",
    "issuer": "nbs_official",
    "legacy_compatibility": {
        "prior_contract_sha256": NBS_PMI_V2_CONTRACT_SHA256,
        "title_source": "title element when ArticleTitle is absent",
        "publication_source": "visible exact timestamp when PubDate is absent",
    },
    "measurement_basis": "headline manufacturing PMI",
    "publication_source": "exact official timestamp with URL record-date equality",
    "unique_value": True,
    "version": NBS_OFFICIAL_PMI_PARSER,
}
_PBC_MONEY_STOCK_CONTRACT = {
    "article_meta": ["ArticleTitle", "PubDate"],
    "body_encoding": "utf-8-strict",
    "issuer": "pbc_official",
    "measurement_basis": {
        "cn.m1_yoy": "month-end narrow money M1 YoY",
        "cn.m2_yoy": "month-end broad money M2 YoY",
    },
    "co_page_non_output_evidence": (
        "rounded social-financing cumulative is structurally validated but excluded"
    ),
    "excluded_observation": {
        "indicator_id": "cn.social_financing_flow",
        "reason": "rounded_cumulative_difference_not_exact",
    },
    "period_source": "formal ArticleTitle year and month",
    "publication_source": "PubDate Asia/Shanghai",
    "unique_value_per_indicator": True,
    "version": PBC_MONEY_STOCK_PARSER,
}

NBS_NATIONAL_ECONOMY_PARSER_CONTRACT_SHA256 = _contract_hash(_NBS_NATIONAL_ECONOMY_CONTRACT)
NBS_QUARTERLY_GDP_PARSER_CONTRACT_SHA256 = _contract_hash(_NBS_QUARTERLY_GDP_CONTRACT)
NBS_OFFICIAL_PMI_PARSER_CONTRACT_SHA256 = _contract_hash(_NBS_OFFICIAL_PMI_CONTRACT)
PBC_MONEY_STOCK_PARSER_CONTRACT_SHA256 = _contract_hash(_PBC_MONEY_STOCK_CONTRACT)

PARSER_CONTRACT_SHA256: Mapping[str, str] = {
    NBS_NATIONAL_ECONOMY_PARSER: NBS_NATIONAL_ECONOMY_PARSER_CONTRACT_SHA256,
    NBS_OFFICIAL_PMI_PARSER: NBS_OFFICIAL_PMI_PARSER_CONTRACT_SHA256,
    NBS_QUARTERLY_GDP_PARSER: NBS_QUARTERLY_GDP_PARSER_CONTRACT_SHA256,
    PBC_MONEY_STOCK_PARSER: PBC_MONEY_STOCK_PARSER_CONTRACT_SHA256,
}

_PARSER_SOURCE_SYSTEM: Mapping[str, str] = {
    NBS_NATIONAL_ECONOMY_PARSER: "nbs_official",
    NBS_OFFICIAL_PMI_PARSER: "nbs_official",
    NBS_QUARTERLY_GDP_PARSER: "nbs_official",
    PBC_MONEY_STOCK_PARSER: "pbc_official",
}

_NATIONAL_ECONOMY_IDS = frozenset(
    {
        "cn.industrial_value_added_yoy",
        "cn.retail_sales_yoy",
        "cn.fixed_asset_investment_yoy",
        "cn.property_investment_yoy",
        "cn.exports_yoy",
        "cn.imports_yoy",
        "cn.cpi_yoy",
        "cn.ppi_yoy",
    }
)
_MONEY_IDS = frozenset({"cn.m1_yoy", "cn.m2_yoy"})
_SUPPORTED_IDS = frozenset(
    {
        *_NATIONAL_ECONOMY_IDS,
        "cn.pmi_manufacturing",
        "cn.gdp_yoy",
        *_MONEY_IDS,
    }
)

_MEASUREMENT_BASIS: Mapping[str, str] = {
    "cn.industrial_value_added_yoy": "current_month_real_yoy",
    "cn.retail_sales_yoy": "current_month_nominal_yoy",
    "cn.fixed_asset_investment_yoy": "jan_to_month_cumulative_yoy",
    "cn.property_investment_yoy": "jan_to_month_cumulative_yoy",
    "cn.exports_yoy": "current_month_cny_yoy",
    "cn.imports_yoy": "current_month_cny_yoy",
    "cn.cpi_yoy": "current_month_yoy",
    "cn.ppi_yoy": "current_month_yoy",
    "cn.pmi_manufacturing": "headline_manufacturing_pmi",
    "cn.gdp_yoy": "current_quarter_real_yoy",
    "cn.m1_yoy": "month_end_yoy",
    "cn.m2_yoy": "month_end_yoy",
}

_FREQUENCY_UNIT: Mapping[str, tuple[str, str]] = {
    indicator.indicator_id: (indicator.frequency, indicator.unit)
    for indicator in NATIONAL_INDICATORS
    if indicator.indicator_id in _SUPPORTED_IDS
}


@dataclass(frozen=True)
class _Value:
    indicator_id: str
    value: Decimal
    period: str = ""


@dataclass(frozen=True)
class _ParsedDocument:
    period: str
    release_at: datetime
    source_record_id: str
    article_title: str
    values: tuple[_Value, ...]


@dataclass(frozen=True)
class _PageEvidence:
    page_id: str
    parser_id: str
    parser_contract_sha256: str
    source_system: str
    source_url: str
    source_record_id: str
    period: str
    release_at: datetime
    fetched_at: datetime
    body_sha256: str
    body_size_bytes: int
    article_title: str
    values: tuple[_Value, ...]


@dataclass(frozen=True)
class OfficialWebCompilationResult:
    """A complete but not-yet-published official macro bundle."""

    observations: tuple[MacroObservation, ...]
    receipts: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]


class _OfficialHtmlParser(HTMLParser):
    """Collect metadata and fixed structural text blocks without a DOM."""

    _BLOCK_TAGS = frozenset({"p", "td", "th", "h1", "h2", "h3", "li"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.blocks: list[str] = []
        self.all_parts: list[str] = []
        self.titles: list[str] = []
        self._block_stack: list[str] = []
        self._block_parts: list[str] = []
        self._title_depth = 0
        self._title_parts: list[str] = []

    def _meta(self, attrs: Sequence[tuple[str, str | None]]) -> None:
        values = {str(key).casefold(): value for key, value in attrs}
        name = values.get("name")
        content = values.get("content")
        if name is not None and content is not None:
            self.meta.setdefault(str(name).casefold(), []).append(str(content))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized == "meta":
            self._meta(attrs)
        if normalized == "title":
            if self._title_depth == 0:
                self._title_parts = []
            self._title_depth += 1
        if normalized in self._BLOCK_TAGS:
            if not self._block_stack:
                self._block_parts = []
            self._block_stack.append(normalized)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "meta":
            self._meta(attrs)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized == "title" and self._title_depth:
            self._title_depth -= 1
            if self._title_depth == 0:
                self.titles.append("".join(self._title_parts))
                self._title_parts = []
        if not self._block_stack or normalized not in self._BLOCK_TAGS:
            return
        if normalized not in self._block_stack:
            return
        while self._block_stack:
            opened = self._block_stack.pop()
            if opened == normalized:
                break
        if not self._block_stack:
            self.blocks.append("".join(self._block_parts))
            self._block_parts = []

    def handle_data(self, data: str) -> None:
        self.all_parts.append(data)
        if self._title_depth:
            self._title_parts.append(data)
        if self._block_stack:
            self._block_parts.append(data)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_hash(value: Any, error: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise OfficialWebCompilerError(error)
    return normalized


def _required_mapping(value: Any, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialWebCompilerError(error)
    return value


def _required_list(value: Any, error: str) -> list[Any]:
    if not isinstance(value, list):
        raise OfficialWebCompilerError(error)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], error: str) -> None:
    if set(value) != expected:
        raise OfficialWebCompilerError(error)


def _normalize_text(value: str) -> str:
    translated = value.translate(
        {
            ord("\u00a0"): "",
            ord("\u3000"): "",
            ord("\u2002"): "",
            ord("\u2003"): "",
            ord("\u2009"): "",
            ord("\ufeff"): "",
            ord("－"): "—",
            ord("-"): "—",
            ord("%"): "%",
            ord("％"): "%",
            ord("("): "（",
            ord(")"): "）",
        }
    )
    return re.sub(r"\s+", "", translated).strip()


def _parse_html(body_bytes: bytes) -> _OfficialHtmlParser:
    if not isinstance(body_bytes, bytes):
        raise OfficialWebCompilerError("official_web_body_bytes_required")
    if not body_bytes or len(body_bytes) > _MAX_HTML_BYTES:
        raise OfficialWebCompilerError("official_web_body_size_invalid")
    try:
        text = body_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OfficialWebCompilerError("official_web_body_not_utf8") from exc
    parser = _OfficialHtmlParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise OfficialWebCompilerError("official_web_html_parse_failed") from exc
    return parser


def _single_meta(parser: _OfficialHtmlParser, name: str) -> str:
    raw_values = parser.meta.get(name.casefold(), [])
    if name.casefold() == "articletitle":
        values = [_normalize_text(item) for item in raw_values]
    else:
        values = [re.sub(r"\s+", " ", item).strip() for item in raw_values]
    unique = tuple(dict.fromkeys(item for item in values if item))
    if len(unique) != 1:
        raise OfficialWebCompilerError(f"official_web_meta_{name.casefold()}_not_unique")
    return unique[0]


def _publication_timestamp(value: str) -> datetime:
    match = _PUB_DATE_RE.fullmatch(value)
    if match is None:
        raise OfficialWebCompilerError("official_web_pubdate_invalid")
    try:
        return datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second") or 0),
            tzinfo=_SHANGHAI,
        )
    except ValueError as exc:
        raise OfficialWebCompilerError("official_web_pubdate_invalid") from exc


def _date_only(value: str) -> date | None:
    match = re.fullmatch(r"(?P<year>20\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})", value)
    if match is None:
        return None
    try:
        return date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError as exc:
        raise OfficialWebCompilerError("official_web_pubdate_invalid") from exc


def _visible_publication_timestamp(
    parser: _OfficialHtmlParser,
    *,
    expected_date: date,
) -> datetime:
    body_text = " ".join(parser.all_parts)
    matches: list[datetime] = []
    for match in re.finditer(
        r"(?P<year>20\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"
        r"\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?",
        body_text,
    ):
        try:
            parsed = datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second") or 0),
                tzinfo=_SHANGHAI,
            )
        except ValueError as exc:
            raise OfficialWebCompilerError("official_web_pubdate_invalid") from exc
        if parsed.date() == expected_date:
            matches.append(parsed)
    unique = tuple(dict.fromkeys(matches))
    if not unique:
        raise OfficialWebCompilerError("official_web_pubdate_exact_time_missing")
    if len(unique) != 1:
        raise OfficialWebCompilerError("official_web_pubdate_exact_time_ambiguous")
    return unique[0]


def _source_record_id(source_url: str, source_system: str) -> tuple[str, date | None]:
    parsed = urlsplit(source_url)
    raw_segments = [unquote(item) for item in parsed.path.split("/") if item]
    if not raw_segments:
        raise OfficialWebCompilerError("official_web_source_path_invalid")
    if any(
        item in {".", ".."} or "\\" in item or any(ord(character) < 32 for character in item)
        for item in raw_segments
    ):
        raise OfficialWebCompilerError("official_web_source_path_invalid")
    if source_system == "nbs_official":
        match = _NBS_RECORD_RE.fullmatch(raw_segments[-1])
        if match is None:
            raise OfficialWebCompilerError("official_web_nbs_record_id_invalid")
        try:
            record_date = datetime.strptime(match.group("date"), "%Y%m%d").date()
        except ValueError as exc:
            raise OfficialWebCompilerError("official_web_nbs_record_date_invalid") from exc
        return match.group(1), record_date
    if source_system != "pbc_official":
        raise OfficialWebCompilerError("official_web_source_system_invalid")
    basename = raw_segments[-1]
    if basename.casefold() == "index.html":
        if len(raw_segments) < 2:
            raise OfficialWebCompilerError("official_web_pbc_record_id_invalid")
        record_id = raw_segments[-2]
    elif basename.casefold().endswith(".html"):
        record_id = basename[:-5]
    else:
        raise OfficialWebCompilerError("official_web_pbc_record_id_invalid")
    if not _PBC_RECORD_RE.fullmatch(record_id):
        raise OfficialWebCompilerError("official_web_pbc_record_id_invalid")
    encoded_dates = tuple(dict.fromkeys(re.findall(r"20\d{6}", record_id)))
    if len(encoded_dates) > 1:
        raise OfficialWebCompilerError("official_web_pbc_record_date_ambiguous")
    if encoded_dates:
        try:
            record_date = datetime.strptime(encoded_dates[0], "%Y%m%d").date()
        except ValueError as exc:
            raise OfficialWebCompilerError("official_web_pbc_record_date_invalid") from exc
    else:
        record_date = None
    return record_id, record_date


def _official_metadata(
    parser: _OfficialHtmlParser,
    *,
    source_url: str,
    source_system: str,
) -> tuple[str, datetime, str]:
    article_title = _single_meta(parser, "ArticleTitle")
    raw_pubdate = _single_meta(parser, "PubDate")
    pubdate_date = _date_only(raw_pubdate)
    if pubdate_date is None:
        release_at = _publication_timestamp(raw_pubdate)
    else:
        release_at = _visible_publication_timestamp(
            parser,
            expected_date=pubdate_date,
        )
    record_id, record_date = _source_record_id(source_url, source_system)
    if record_date is not None and release_at.date() != record_date:
        raise OfficialWebCompilerError("official_web_pubdate_record_mismatch")
    return article_title, release_at, record_id


def _decimal(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise OfficialWebCompilerError("official_web_numeric_value_invalid") from exc
    if not result.is_finite():
        raise OfficialWebCompilerError("official_web_numeric_value_invalid")
    return result


def _signed_value(direction: str, value: str) -> Decimal:
    parsed = _decimal(value)
    if direction in {"下降", "下跌"}:
        return -parsed
    if direction not in {"增长", "上升", "上涨"}:
        raise OfficialWebCompilerError("official_web_direction_invalid")
    return parsed


def _unique_pattern_match(
    texts: Sequence[str],
    pattern: re.Pattern[str] | Sequence[re.Pattern[str]],
    *,
    error_prefix: str,
) -> tuple[int, Decimal]:
    matches: list[tuple[int, Decimal]] = []
    patterns = (pattern,) if isinstance(pattern, re.Pattern) else tuple(pattern)
    for text in texts:
        for compiled in patterns:
            for match in compiled.finditer(text):
                month_text = match.groupdict().get("month")
                if month_text is None and match.groupdict().get("half_year") is not None:
                    month_text = "6"
                if month_text is None:
                    raise OfficialWebCompilerError(f"{error_prefix}_period_missing")
                matches.append(
                    (
                        int(month_text),
                        _signed_value(match.group("direction"), match.group("value")),
                    )
                )
    unique = tuple(dict.fromkeys(matches))
    if not unique:
        raise OfficialWebCompilerError(f"{error_prefix}_missing")
    periods = {item[0] for item in unique}
    values = {item[1] for item in unique}
    if len(periods) != 1:
        raise OfficialWebCompilerError(f"{error_prefix}_period_ambiguous")
    if len(values) != 1:
        raise OfficialWebCompilerError(f"{error_prefix}_value_ambiguous")
    return unique[0]


def _period_not_after_release(period: str, release_at: datetime) -> None:
    if re.fullmatch(r"20\d{4}", period):
        year, month = int(period[:4]), int(period[4:])
        if (year, month) > (release_at.year, release_at.month):
            raise OfficialWebCompilerError("official_web_period_after_pubdate")
        period_end = date(year, month, calendar.monthrange(year, month)[1])
        if (release_at.date() - period_end).days > 62:
            raise OfficialWebCompilerError("official_web_release_lag_invalid")
        return
    match = re.fullmatch(r"(20\d{2})Q([1-4])", period)
    if match is None:
        raise OfficialWebCompilerError("official_web_period_invalid")
    year, quarter = int(match.group(1)), int(match.group(2))
    month = quarter * 3
    period_end = date(year, month, calendar.monthrange(year, month)[1])
    if period_end > release_at.date():
        raise OfficialWebCompilerError("official_web_period_after_pubdate")
    if (release_at.date() - period_end).days > 150:
        raise OfficialWebCompilerError("official_web_release_lag_invalid")


_INDUSTRIAL_RE = re.compile(
    rf"(?<![—])(?P<month>1[0-2]|[1-9])月份，?(?:全国)?规模以上工业增加值"
    rf"同比(?:实际)?{_PERCENT_RE}"
)
_RETAIL_RE = re.compile(
    rf"(?<![—])(?P<month>1[0-2]|[1-9])月份，?(?:全国)?社会消费品零售总额"
    rf"[^。；]{{0,80}}?同比{_PERCENT_RE}"
)
_FAI_RE = re.compile(
    rf"(?:20\d{{2}}年)?1—(?P<month>1[0-2]|[1-9])月份，?全国固定资产投资"
    rf"（不含农户）[^。；]{{0,80}}?同比{_PERCENT_RE}"
)
_FAI_HALF_YEAR_RE = re.compile(
    rf"上半年(?P<half_year>)，?全国固定资产投资（不含农户）" rf"[^。；]{{0,80}}?同比{_PERCENT_RE}"
)
_PROPERTY_RE = re.compile(
    rf"(?:20\d{{2}}年)?1—(?P<month>1[0-2]|[1-9])月份，?全国固定资产投资"
    rf"（不含农户）.{{0,320}}?(?<!扣除)房地产开发投资{_PERCENT_RE}"
)
_PROPERTY_HALF_YEAR_RE = re.compile(
    rf"上半年(?P<half_year>)，?全国固定资产投资（不含农户）"
    rf".{{0,320}}?(?<!扣除)房地产开发投资{_PERCENT_RE}"
)
_EXPORT_RE = re.compile(
    rf"(?<![—])(?P<month>1[0-2]|[1-9])月份，?(?:货物)?进出口总额.{{0,200}}?"
    rf"其中，?出口[^；。]{{0,80}}?(?:同比)?{_PERCENT_RE}"
)
_IMPORT_RE = re.compile(
    rf"(?<![—])(?P<month>1[0-2]|[1-9])月份，?(?:货物)?进出口总额.{{0,200}}?"
    rf"其中，?出口[^；。]{{0,100}}?[；;，,]进口[^；。]{{0,80}}?(?:同比)?{_PERCENT_RE}"
)
_CPI_RE = re.compile(
    rf"(?<![—])(?P<month>1[0-2]|[1-9])月份，?(?:全国)?居民消费价格"
    rf"(?:指数)?(?:（CPI）)?[^。；]{{0,40}}?(?:同比)?{_PERCENT_RE}"
)
_PPI_RE = re.compile(
    rf"(?<![—])(?P<month>1[0-2]|[1-9])月份，?(?:全国)?工业生产者出厂价格"
    rf"(?:指数)?(?:（PPI）)?[^。；]{{0,40}}?(?:同比)?{_PERCENT_RE}"
)
_PPI_CUMULATIVE_LEAD_RE = re.compile(
    rf"(?:20\d{{2}}年)?1—(?P<month>1[0-2]|[1-9])月份，?(?:全国)?"
    rf"工业生产者出厂价格(?:指数)?(?:（PPI）)?同比"
    rf"(?:增长|上升|上涨|下降|下跌){_NUMBER_RE}[%％]。其中，?"
    rf"(?P=month)月份同比{_PERCENT_RE}"
)
_PPI_HALF_YEAR_LEAD_RE = re.compile(
    rf"上半年，?(?:全国)?工业生产者出厂价格(?:指数)?(?:（PPI）)?同比"
    rf"(?:增长|上升|上涨|下降|下跌){_NUMBER_RE}[%％]。其中，?"
    rf"(?P<month>6)月份同比{_PERCENT_RE}"
)
_HALF_YEAR_Q2_GDP_RE = re.compile(
    rf"二季度(?:国内生产总值)?(?:同比)?"
    rf"(?P<direction>增长|上升|下降|下跌)(?P<value>{_NUMBER_RE})[%％]"
)


def _parse_half_year_q2_gdp(texts: Sequence[str], *, year: int, article_title: str) -> _Value:
    if "上半年" not in article_title:
        raise OfficialWebCompilerError("official_web_half_year_gdp_title_invalid")
    values: list[Decimal] = []
    for text in texts:
        if not (
            "初步核算，上半年国内生产总值" in text
            and "按不变价格计算" in text
            and "分季度看" in text
            and "从环比看" in text
        ):
            continue
        segment = text.split("分季度看", 1)[1].split("从环比看", 1)[0]
        for match in _HALF_YEAR_Q2_GDP_RE.finditer(segment):
            values.append(_signed_value(match.group("direction"), match.group("value")))
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise OfficialWebCompilerError("official_web_half_year_q2_gdp_missing")
    if len(unique) != 1:
        raise OfficialWebCompilerError("official_web_half_year_q2_gdp_ambiguous")
    period = f"{year:04d}Q2"
    return _Value("cn.gdp_yoy", unique[0], period)


def _parse_nbs_national_economy(body_bytes: bytes, *, source_url: str) -> _ParsedDocument:
    parser = _parse_html(body_bytes)
    title, release_at, record_id = _official_metadata(
        parser,
        source_url=source_url,
        source_system="nbs_official",
    )
    texts = tuple(_normalize_text(item) for item in parser.blocks if item.strip())
    patterns: dict[str, re.Pattern[str] | tuple[re.Pattern[str], ...]] = {
        "cn.industrial_value_added_yoy": _INDUSTRIAL_RE,
        "cn.retail_sales_yoy": _RETAIL_RE,
        "cn.fixed_asset_investment_yoy": (_FAI_RE, _FAI_HALF_YEAR_RE),
        "cn.property_investment_yoy": (_PROPERTY_RE, _PROPERTY_HALF_YEAR_RE),
        "cn.exports_yoy": _EXPORT_RE,
        "cn.imports_yoy": _IMPORT_RE,
        "cn.cpi_yoy": _CPI_RE,
        "cn.ppi_yoy": (
            _PPI_RE,
            _PPI_CUMULATIVE_LEAD_RE,
            _PPI_HALF_YEAR_LEAD_RE,
        ),
    }
    parsed_values: list[_Value] = []
    parsed_months: set[int] = set()
    for indicator_id, pattern in patterns.items():
        month, value = _unique_pattern_match(
            texts,
            pattern,
            error_prefix=f"official_web_{indicator_id.replace('.', '_')}",
        )
        parsed_months.add(month)
        parsed_values.append(_Value(indicator_id, value))
    if len(parsed_months) != 1:
        raise OfficialWebCompilerError("official_web_nbs_economy_period_ambiguous")
    month = next(iter(parsed_months))
    year = release_at.year if month <= release_at.month else release_at.year - 1
    period = f"{year:04d}{month:02d}"
    _period_not_after_release(period, release_at)
    if month == 6:
        q2_gdp = _parse_half_year_q2_gdp(
            texts,
            year=year,
            article_title=title,
        )
        _period_not_after_release(q2_gdp.period, release_at)
        parsed_values.append(q2_gdp)
    return _ParsedDocument(
        period,
        release_at,
        record_id,
        title,
        tuple(sorted(parsed_values, key=lambda item: item.indicator_id)),
    )


_PMI_TITLE_RE = re.compile(
    r"^(?P<year>20\d{2})年(?P<month>1[0-2]|[1-9])月中国采购经理指数运行情况$"
)
_PMI_VALUE_RE = re.compile(
    rf"(?P<month>1[0-2]|[1-9])月份，?(?:中国)?制造业采购经理指数（PMI）"
    rf"为(?P<value>{_NUMBER_RE})[%％]"
)


def _single_document_title(parser: _OfficialHtmlParser) -> str:
    values = []
    for item in parser.titles:
        normalized = _normalize_text(item)
        normalized = re.sub(r"-国家统计局$", "", normalized)
        if normalized:
            values.append(normalized)
    unique = tuple(dict.fromkeys(values))
    if len(unique) != 1:
        raise OfficialWebCompilerError("official_web_document_title_not_unique")
    return unique[0]


def _parse_nbs_official_pmi(body_bytes: bytes, *, source_url: str) -> _ParsedDocument:
    parser = _parse_html(body_bytes)
    title_values = parser.meta.get("articletitle", [])
    title = _single_meta(parser, "ArticleTitle") if title_values else _single_document_title(parser)
    title_match = _PMI_TITLE_RE.fullmatch(title)
    if title_match is None:
        raise OfficialWebCompilerError("official_web_pmi_article_title_invalid")
    record_id, record_date = _source_record_id(source_url, "nbs_official")
    raw_pubdates = parser.meta.get("pubdate", [])
    if raw_pubdates:
        raw_pubdate = _single_meta(parser, "PubDate")
        date_value = _date_only(raw_pubdate)
        release_at = (
            _publication_timestamp(raw_pubdate)
            if date_value is None
            else _visible_publication_timestamp(parser, expected_date=date_value)
        )
    else:
        if record_date is None:  # pragma: no cover - NBS records always encode it
            raise OfficialWebCompilerError("official_web_pmi_record_date_missing")
        release_at = _visible_publication_timestamp(
            parser,
            expected_date=record_date,
        )
    if release_at.date() != record_date:
        raise OfficialWebCompilerError("official_web_pubdate_record_mismatch")
    period = f"{int(title_match.group('year')):04d}" f"{int(title_match.group('month')):02d}"
    texts = tuple(_normalize_text(item) for item in parser.blocks if item.strip())
    matches: list[tuple[int, Decimal]] = []
    for text in texts:
        for match in _PMI_VALUE_RE.finditer(text):
            matches.append((int(match.group("month")), _decimal(match.group("value"))))
    unique = tuple(dict.fromkeys(matches))
    if not unique:
        raise OfficialWebCompilerError("official_web_pmi_value_missing")
    if len({item[0] for item in unique}) != 1:
        raise OfficialWebCompilerError("official_web_pmi_period_ambiguous")
    if len({item[1] for item in unique}) != 1:
        raise OfficialWebCompilerError("official_web_pmi_value_ambiguous")
    if unique[0][0] != int(title_match.group("month")):
        raise OfficialWebCompilerError("official_web_pmi_period_mismatch")
    if not Decimal("0") <= unique[0][1] <= Decimal("100"):
        raise OfficialWebCompilerError("official_web_pmi_value_out_of_range")
    _period_not_after_release(period, release_at)
    return _ParsedDocument(
        period,
        release_at,
        record_id,
        title,
        (_Value("cn.pmi_manufacturing", unique[0][1]),),
    )


_QUARTER_NUMBER = {"一": 1, "二": 2, "三": 3, "四": 4}


def _gdp_primary_period(title: str, texts: Sequence[str], release_at: datetime) -> str:
    year_match = re.match(r"^(?P<year>20\d{2})年", title)
    year = int(year_match.group("year")) if year_match else release_at.year
    quarter: int | None = None
    for text, value in (("一季度", 1), ("上半年", 2), ("前三季度", 3)):
        if text in title:
            quarter = value
            break
    if quarter is None and (
        "全年" in title or any("全年国内生产总值" in text and "分季度看" in text for text in texts)
    ):
        quarter = 4
    if quarter is None:
        raise OfficialWebCompilerError("official_web_gdp_article_title_invalid")
    return f"{year:04d}Q{quarter}"


def _parse_nbs_quarterly_gdp(body_bytes: bytes, *, source_url: str) -> _ParsedDocument:
    parser = _parse_html(body_bytes)
    title, release_at, record_id = _official_metadata(
        parser,
        source_url=source_url,
        source_system="nbs_official",
    )
    texts = tuple(_normalize_text(item) for item in parser.blocks if item.strip())
    primary_period = _gdp_primary_period(title, texts, release_at)
    year = int(primary_period[:4])
    by_quarter: dict[int, list[Decimal]] = {}
    sequence_pattern = re.compile(
        rf"(?P<quarter>[一二三四])季度(?:国内生产总值)?(?:同比)?"
        rf"(?P<direction>增长|上升|下降|下跌)(?P<value>{_NUMBER_RE})[%％]"
    )
    direct_pattern = re.compile(
        rf"(?P<quarter>[一二三四])季度国内生产总值[^。；]{{0,100}}?"
        rf"按不变价格计算，?同比(?P<direction>增长|上升|下降|下跌)"
        rf"(?P<value>{_NUMBER_RE})[%％]"
    )
    for text in texts:
        if "分季度看" in text:
            segment = text.split("分季度看", 1)[1].split("从环比看", 1)[0]
            for match in sequence_pattern.finditer(segment):
                quarter = _QUARTER_NUMBER[match.group("quarter")]
                by_quarter.setdefault(quarter, []).append(
                    _signed_value(match.group("direction"), match.group("value"))
                )
        for match in direct_pattern.finditer(text):
            quarter = _QUARTER_NUMBER[match.group("quarter")]
            by_quarter.setdefault(quarter, []).append(
                _signed_value(match.group("direction"), match.group("value"))
            )
    if not by_quarter:
        raise OfficialWebCompilerError("official_web_gdp_value_missing")
    parsed_values: list[_Value] = []
    primary_index = _quarter_index(primary_period)
    for quarter, raw_values in sorted(by_quarter.items()):
        unique = tuple(dict.fromkeys(raw_values))
        if len(unique) != 1:
            raise OfficialWebCompilerError("official_web_gdp_value_ambiguous")
        period = f"{year:04d}Q{quarter}"
        if _quarter_index(period) not in {primary_index - 1, primary_index}:
            continue
        _period_not_after_release(period, release_at)
        parsed_values.append(_Value("cn.gdp_yoy", unique[0], period))
    if primary_period not in {item.period for item in parsed_values}:
        raise OfficialWebCompilerError("official_web_gdp_primary_value_missing")
    return _ParsedDocument(
        primary_period,
        release_at,
        record_id,
        title,
        tuple(parsed_values),
    )


_PBC_MONEY_TITLE_RE = re.compile(
    r"^(?P<year>20\d{2})年(?:(?P<month>1[0-2]|[1-9])月|"
    r"(?P<quarter>[一二三四])季度)"
    r"(?:金融统计数据报告|金融统计数据|货币金融数据)$"
)
_M1_RE = re.compile(rf"(?:狭义货币（M1）|M1余额)[^。；]{{0,100}}?同比{_PERCENT_RE}")
_M2_RE = re.compile(rf"(?:广义货币（M2）|M2余额)[^。；]{{0,100}}?同比{_PERCENT_RE}")


def _unique_signed_value(
    texts: Sequence[str], pattern: re.Pattern[str], *, error_prefix: str
) -> Decimal:
    values: list[Decimal] = []
    for text in texts:
        for match in pattern.finditer(text):
            values.append(_signed_value(match.group("direction"), match.group("value")))
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise OfficialWebCompilerError(f"{error_prefix}_missing")
    if len(unique) != 1:
        raise OfficialWebCompilerError(f"{error_prefix}_ambiguous")
    return unique[0]


def _parse_pbc_money_stock(body_bytes: bytes, *, source_url: str) -> _ParsedDocument:
    parser = _parse_html(body_bytes)
    title, release_at, record_id = _official_metadata(
        parser,
        source_url=source_url,
        source_system="pbc_official",
    )
    title_match = _PBC_MONEY_TITLE_RE.fullmatch(title)
    if title_match is None:
        raise OfficialWebCompilerError("official_web_money_article_title_invalid")
    texts = tuple(_normalize_text(item) for item in parser.blocks if item.strip())
    if title_match.group("month"):
        month = int(title_match.group("month"))
    else:
        month = _QUARTER_NUMBER[title_match.group("quarter")] * 3
    period = f"{int(title_match.group('year')):04d}{month:02d}"
    _period_not_after_release(period, release_at)
    # The same official bytes also contain a rounded cumulative social-
    # financing figure.  Parse it to prove structural integrity, but never
    # convert adjacent rounded totals into an allegedly exact monthly flow.
    _unique_social_financing_cumulative(texts)
    return _ParsedDocument(
        period,
        release_at,
        record_id,
        title,
        (
            _Value(
                "cn.m1_yoy",
                _unique_signed_value(texts, _M1_RE, error_prefix="official_web_m1_value"),
            ),
            _Value(
                "cn.m2_yoy",
                _unique_signed_value(texts, _M2_RE, error_prefix="official_web_m2_value"),
            ),
        ),
    )


_PBC_SF_CUMULATIVE_RE = re.compile(
    rf"社会融资规模增量累计为(?P<value>{_NUMBER_RE})(?P<unit>万亿元|亿元)"
)


def _unique_social_financing_cumulative(texts: Sequence[str]) -> Decimal:
    values: list[Decimal] = []
    for text in texts:
        for match in _PBC_SF_CUMULATIVE_RE.finditer(text):
            value = _decimal(match.group("value"))
            if match.group("unit") == "万亿元":
                value *= Decimal("10000")
            values.append(value)
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise OfficialWebCompilerError("official_web_sf_cumulative_missing")
    if len(unique) != 1:
        raise OfficialWebCompilerError("official_web_sf_cumulative_ambiguous")
    return unique[0]


def _parse_page(
    parser_id: str,
    body_bytes: bytes,
    *,
    source_url: str,
) -> _ParsedDocument:
    if parser_id == NBS_NATIONAL_ECONOMY_PARSER:
        return _parse_nbs_national_economy(body_bytes, source_url=source_url)
    if parser_id == NBS_OFFICIAL_PMI_PARSER:
        return _parse_nbs_official_pmi(body_bytes, source_url=source_url)
    if parser_id == NBS_QUARTERLY_GDP_PARSER:
        return _parse_nbs_quarterly_gdp(body_bytes, source_url=source_url)
    if parser_id == PBC_MONEY_STOCK_PARSER:
        return _parse_pbc_money_stock(body_bytes, source_url=source_url)
    raise OfficialWebCompilerError("official_web_parser_id_unsupported")


def _period_end(period: str, frequency: str) -> str:
    if frequency == "monthly":
        if re.fullmatch(r"20\d{4}", period) is None:
            raise OfficialWebCompilerError("official_web_month_period_invalid")
        year, month = int(period[:4]), int(period[4:])
        try:
            return date(year, month, calendar.monthrange(year, month)[1]).isoformat()
        except ValueError as exc:
            raise OfficialWebCompilerError("official_web_month_period_invalid") from exc
    if frequency == "quarterly":
        match = re.fullmatch(r"(20\d{2})Q([1-4])", period)
        if match is None:
            raise OfficialWebCompilerError("official_web_quarter_period_invalid")
        year, month = int(match.group(1)), int(match.group(2)) * 3
        return date(year, month, calendar.monthrange(year, month)[1]).isoformat()
    raise OfficialWebCompilerError("official_web_frequency_invalid")


def _scope_period(indicator_id: str, period_end: str) -> str:
    try:
        parsed = date.fromisoformat(period_end)
    except ValueError as exc:
        raise OfficialWebCompilerError("official_web_scope_period_end_invalid") from exc
    frequency = _FREQUENCY_UNIT[indicator_id][0]
    if frequency == "monthly":
        period = f"{parsed.year:04d}{parsed.month:02d}"
    elif frequency == "quarterly" and parsed.month in {3, 6, 9, 12}:
        period = f"{parsed.year:04d}Q{parsed.month // 3}"
    else:
        raise OfficialWebCompilerError("official_web_scope_frequency_invalid")
    if _period_end(period, frequency) != parsed.isoformat():
        raise OfficialWebCompilerError("official_web_scope_period_not_period_end")
    return period


def _month_index(period: str) -> int:
    if re.fullmatch(r"20\d{4}", period) is None:
        raise OfficialWebCompilerError("official_web_month_period_invalid")
    year, month = int(period[:4]), int(period[4:])
    if not 1 <= month <= 12:
        raise OfficialWebCompilerError("official_web_month_period_invalid")
    return year * 12 + month - 1


def _quarter_index(period: str) -> int:
    match = re.fullmatch(r"(20\d{2})Q([1-4])", period)
    if match is None:
        raise OfficialWebCompilerError("official_web_quarter_period_invalid")
    return int(match.group(1)) * 4 + int(match.group(2)) - 1


def _require_consecutive(periods: Sequence[str], *, quarterly: bool) -> None:
    indices = sorted(_quarter_index(item) if quarterly else _month_index(item) for item in periods)
    if len(indices) != len(set(indices)) or any(
        right != left + 1 for left, right in zip(indices, indices[1:])
    ):
        raise OfficialWebCompilerError("official_web_scope_periods_not_consecutive")


def _validate_requested_scope(plan: Mapping[str, Any]) -> set[tuple[str, str]]:
    rows = _required_list(plan.get("requested_scope"), "official_web_requested_scope_not_list")
    if len(rows) != 36:
        raise OfficialWebCompilerError("official_web_requested_scope_count_invalid")
    scope: set[tuple[str, str]] = set()
    periods_by_id: dict[str, set[str]] = {}
    for raw_row in rows:
        row = _required_mapping(raw_row, "official_web_scope_row_not_object")
        _exact_keys(
            row,
            {"indicator_id", "period_end"},
            "official_web_scope_row_shape_invalid",
        )
        indicator_id = str(row.get("indicator_id") or "").strip()
        if indicator_id not in _SUPPORTED_IDS:
            raise OfficialWebCompilerError("official_web_scope_indicator_unsupported")
        period_end = str(row.get("period_end") or "").strip()
        compact = _scope_period(indicator_id, period_end)
        key = (indicator_id, period_end)
        if key in scope:
            raise OfficialWebCompilerError("official_web_scope_duplicate")
        scope.add(key)
        periods_by_id.setdefault(indicator_id, set()).add(compact)
    if set(periods_by_id) != _SUPPORTED_IDS or any(
        len(periods) != 3 for periods in periods_by_id.values()
    ):
        raise OfficialWebCompilerError("official_web_scope_indicator_history_invalid")
    economy_periods = {
        tuple(sorted(periods_by_id[indicator_id])) for indicator_id in _NATIONAL_ECONOMY_IDS
    }
    if len(economy_periods) != 1:
        raise OfficialWebCompilerError("official_web_economy_scope_period_mismatch")
    money_periods = {tuple(sorted(periods_by_id[item])) for item in _MONEY_IDS}
    if len(money_periods) != 1:
        raise OfficialWebCompilerError("official_web_money_scope_period_mismatch")
    _require_consecutive(next(iter(economy_periods)), quarterly=False)
    _require_consecutive(tuple(periods_by_id["cn.pmi_manufacturing"]), quarterly=False)
    _require_consecutive(next(iter(money_periods)), quarterly=False)
    _require_consecutive(tuple(periods_by_id["cn.gdp_yoy"]), quarterly=True)
    return scope


def _validated_source_url(value: Any, *, source_system: str) -> str:
    raw = str(value or "").strip()
    try:
        normalized = normalize_source_url(raw, source_system=source_system)
    except ValueError as exc:
        raise OfficialWebCompilerError(str(exc)) from exc
    if raw != normalized:
        raise OfficialWebCompilerError("official_web_source_url_not_normalized")
    return normalized


def _safe_raw_path(value: Any) -> str:
    raw = str(value or "")
    if not raw or "\\" in raw:
        raise OfficialWebCompilerError("official_web_raw_path_unsafe")
    candidate = PurePosixPath(raw)
    if (
        candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.suffix.casefold() not in {".html", ".htm"}
        or str(candidate) != raw
    ):
        raise OfficialWebCompilerError("official_web_raw_path_unsafe")
    return raw


def _validate_plan_pages(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = _required_list(plan.get("pages"), "official_web_plan_pages_not_list")
    if len(rows) != 12:
        raise OfficialWebCompilerError("official_web_plan_page_count_invalid")
    pages: dict[str, Mapping[str, Any]] = {}
    parser_counts: dict[str, int] = {}
    urls: set[str] = set()
    periods_by_parser: dict[str, list[str]] = {}
    for raw_row in rows:
        row = _required_mapping(raw_row, "official_web_plan_page_not_object")
        _exact_keys(
            row,
            {
                "page_id",
                "parser_id",
                "parser_contract_sha256",
                "source_system",
                "source_url",
                "expected_period",
            },
            "official_web_plan_page_shape_invalid",
        )
        page_id = str(row.get("page_id") or "")
        if not _PAGE_ID_RE.fullmatch(page_id) or page_id in {".", ".."} or page_id in pages:
            raise OfficialWebCompilerError("official_web_page_id_invalid")
        parser_id = str(row.get("parser_id") or "")
        expected_contract = PARSER_CONTRACT_SHA256.get(parser_id)
        if expected_contract is None:
            raise OfficialWebCompilerError("official_web_parser_id_unsupported")
        if (
            _required_hash(
                row.get("parser_contract_sha256"),
                "official_web_parser_contract_hash_invalid",
            )
            != expected_contract
        ):
            raise OfficialWebCompilerError("official_web_parser_contract_mismatch")
        source_system = str(row.get("source_system") or "").strip().lower()
        if source_system != _PARSER_SOURCE_SYSTEM[parser_id]:
            raise OfficialWebCompilerError("official_web_parser_issuer_mismatch")
        source_url = _validated_source_url(row.get("source_url"), source_system=source_system)
        if source_url in urls:
            raise OfficialWebCompilerError("official_web_source_url_duplicate")
        urls.add(source_url)
        expected_period = str(row.get("expected_period") or "")
        if parser_id == NBS_QUARTERLY_GDP_PARSER:
            _quarter_index(expected_period)
        else:
            _month_index(expected_period)
        pages[page_id] = dict(row)
        parser_counts[parser_id] = parser_counts.get(parser_id, 0) + 1
        periods_by_parser.setdefault(parser_id, []).append(expected_period)
    expected_counts = {
        NBS_NATIONAL_ECONOMY_PARSER: 3,
        NBS_OFFICIAL_PMI_PARSER: 3,
        NBS_QUARTERLY_GDP_PARSER: 2,
        PBC_MONEY_STOCK_PARSER: 4,
    }
    if parser_counts != expected_counts:
        raise OfficialWebCompilerError("official_web_plan_parser_counts_invalid")
    _require_consecutive(periods_by_parser[NBS_NATIONAL_ECONOMY_PARSER], quarterly=False)
    _require_consecutive(periods_by_parser[NBS_OFFICIAL_PMI_PARSER], quarterly=False)
    _require_consecutive(periods_by_parser[NBS_QUARTERLY_GDP_PARSER], quarterly=True)
    _require_consecutive(periods_by_parser[PBC_MONEY_STOCK_PARSER], quarterly=False)
    return pages


def _validate_plan(
    plan: Mapping[str, Any],
) -> tuple[set[tuple[str, str]], dict[str, Mapping[str, Any]]]:
    _exact_keys(
        plan,
        {"schema_version", "market", "requested_scope", "pages"},
        "official_web_plan_shape_invalid",
    )
    if plan.get("schema_version") != OFFICIAL_WEB_PLAN_SCHEMA:
        raise OfficialWebCompilerError("official_web_plan_schema_invalid")
    if plan.get("market") != "CN":
        raise OfficialWebCompilerError("official_web_plan_market_invalid")
    scope = _validate_requested_scope(plan)
    pages = _validate_plan_pages(plan)
    scope_periods: dict[str, set[str]] = {}
    for indicator_id, period_end in scope:
        scope_periods.setdefault(indicator_id, set()).add(_scope_period(indicator_id, period_end))
    page_periods: dict[str, list[str]] = {}
    for page in pages.values():
        page_periods.setdefault(str(page["parser_id"]), []).append(str(page["expected_period"]))
    economy_periods = set(page_periods[NBS_NATIONAL_ECONOMY_PARSER])
    if any(scope_periods[item] != economy_periods for item in _NATIONAL_ECONOMY_IDS):
        raise OfficialWebCompilerError("official_web_economy_plan_scope_mismatch")
    if scope_periods["cn.pmi_manufacturing"] != set(page_periods[NBS_OFFICIAL_PMI_PARSER]):
        raise OfficialWebCompilerError("official_web_pmi_plan_scope_mismatch")
    pbc_periods = sorted(page_periods[PBC_MONEY_STOCK_PARSER], key=_month_index)
    latest_three = set(pbc_periods[-3:])
    if any(scope_periods[item] != latest_three for item in _MONEY_IDS):
        raise OfficialWebCompilerError("official_web_money_plan_scope_mismatch")
    return scope, pages


def _validate_capture_pages(
    capture_manifest: Mapping[str, Any],
    *,
    plan_pages: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    rows = _required_list(capture_manifest.get("pages"), "official_web_capture_pages_not_list")
    if len(rows) != len(plan_pages):
        raise OfficialWebCompilerError("official_web_capture_page_count_invalid")
    pages: dict[str, Mapping[str, Any]] = {}
    raw_paths: set[str] = set()
    for raw_row in rows:
        row = _required_mapping(raw_row, "official_web_capture_page_not_object")
        _exact_keys(
            row,
            {
                "page_id",
                "source_system",
                "source_url",
                "effective_url",
                "raw_path",
                "raw_sha256",
                "size_bytes",
                "fetch_started_at",
                "fetch_completed_at",
                "content_type",
                "charset",
                "redirect_chain",
            },
            "official_web_capture_page_shape_invalid",
        )
        page_id = str(row.get("page_id") or "")
        if page_id in pages or page_id not in plan_pages:
            raise OfficialWebCompilerError("official_web_capture_page_id_invalid")
        planned = plan_pages[page_id]
        source_system = str(row.get("source_system") or "").strip().lower()
        if source_system != planned["source_system"]:
            raise OfficialWebCompilerError("official_web_capture_issuer_mismatch")
        source_url = _validated_source_url(row.get("source_url"), source_system=source_system)
        effective_url = _validated_source_url(row.get("effective_url"), source_system=source_system)
        if effective_url != planned["source_url"]:
            raise OfficialWebCompilerError("official_web_effective_url_mismatch")
        chain = _required_list(row.get("redirect_chain"), "official_web_redirect_chain_not_list")
        if not 1 <= len(chain) <= 4:
            raise OfficialWebCompilerError("official_web_redirect_chain_invalid")
        normalized_chain = [
            _validated_source_url(item, source_system=source_system) for item in chain
        ]
        if normalized_chain[0] != source_url or normalized_chain[-1] != effective_url:
            raise OfficialWebCompilerError("official_web_redirect_chain_mismatch")
        if len(normalized_chain) != len(set(normalized_chain)):
            raise OfficialWebCompilerError("official_web_redirect_chain_loop")
        raw_path = _safe_raw_path(row.get("raw_path"))
        if raw_path in raw_paths:
            raise OfficialWebCompilerError("official_web_raw_path_duplicate")
        raw_paths.add(raw_path)
        _required_hash(row.get("raw_sha256"), "official_web_raw_hash_invalid")
        size = row.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= _MAX_HTML_BYTES:
            raise OfficialWebCompilerError("official_web_raw_size_invalid")
        started = parse_timestamp(row.get("fetch_started_at"), field_name="fetch_started_at")
        completed = parse_timestamp(row.get("fetch_completed_at"), field_name="fetch_completed_at")
        if completed < started:
            raise OfficialWebCompilerError("official_web_fetch_clock_invalid")
        if str(row.get("content_type") or "").strip().casefold() != "text/html":
            raise OfficialWebCompilerError("official_web_content_type_invalid")
        if str(row.get("charset") or "").strip().casefold() not in {"utf-8", "utf8"}:
            raise OfficialWebCompilerError("official_web_charset_invalid")
        pages[page_id] = dict(row)
    if set(pages) != set(plan_pages):
        raise OfficialWebCompilerError("official_web_capture_page_set_mismatch")
    return pages


def _validate_capture_manifest(
    capture_manifest: Mapping[str, Any],
    *,
    plan_pages: Mapping[str, Mapping[str, Any]],
    plan_file_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    _exact_keys(
        capture_manifest,
        {"schema_version", "market", "plan_sha256", "pages"},
        "official_web_capture_shape_invalid",
    )
    if capture_manifest.get("schema_version") != OFFICIAL_WEB_CAPTURE_SCHEMA:
        raise OfficialWebCompilerError("official_web_capture_schema_invalid")
    if capture_manifest.get("market") != "CN":
        raise OfficialWebCompilerError("official_web_capture_market_invalid")
    if (
        _required_hash(
            capture_manifest.get("plan_sha256"),
            "official_web_capture_plan_hash_invalid",
        )
        != plan_file_sha256
    ):
        raise OfficialWebCompilerError("official_web_capture_plan_hash_mismatch")
    return _validate_capture_pages(capture_manifest, plan_pages=plan_pages)


def _utc_iso(value: datetime) -> str:
    return parse_timestamp(value.isoformat(), field_name="timestamp").isoformat()


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    result = format(normalized, "f")
    return "0" if result in {"-0", ""} else result


def _page_semantic_evidence(page: _PageEvidence) -> dict[str, Any]:
    return {
        "page_id": page.page_id,
        "parser_id": page.parser_id,
        "parser_contract_sha256": page.parser_contract_sha256,
        "source_system": page.source_system,
        "source_url": page.source_url,
        "source_record_id": page.source_record_id,
        "period": page.period,
        "release_at": _utc_iso(page.release_at),
        "body_sha256": page.body_sha256,
        "body_size_bytes": page.body_size_bytes,
    }


def _build_observation(
    indicator_id: str,
    period: str,
    value: Decimal,
    *,
    page: _PageEvidence,
) -> tuple[MacroObservation, Mapping[str, Any]]:
    frequency, unit = _FREQUENCY_UNIT[indicator_id]
    period_end = _period_end(period, frequency)
    evidence = {
        "schema_version": OFFICIAL_WEB_EVIDENCE_SCHEMA,
        "indicator_id": indicator_id,
        "period_end": period_end,
        "measurement_basis": _MEASUREMENT_BASIS[indicator_id],
        "value_decimal": _decimal_text(value),
        "pages": [_page_semantic_evidence(page)],
    }
    evidence_hash = canonical_hash(evidence)
    lineage_source_system = (
        "pboc_official" if page.source_system == "pbc_official" else page.source_system
    )
    observation = MacroObservation.from_mapping(
        {
            "indicator_id": indicator_id,
            "dimension_type": "national",
            "period_end": period_end,
            "release_at": page.release_at.isoformat(),
            "available_at": page.release_at.isoformat(),
            "vintage_id": f"official-web.v1:{evidence_hash}",
            "value": float(value),
            "unit": unit,
            "frequency": frequency,
            "source_system": lineage_source_system,
            "source_record_id": page.source_record_id,
            "source_url": page.source_url,
            "fetched_at": page.fetched_at.isoformat(),
            "quality_status": "pass",
        }
    )
    receipt = {
        "schema_version": OFFICIAL_WEB_RECEIPT_SCHEMA,
        "status": "accepted",
        "indicator_id": indicator_id,
        "period_end": period_end,
        "content_hash": observation.content_hash,
        "evidence_semantic_sha256": evidence_hash,
        "measurement_basis": _MEASUREMENT_BASIS[indicator_id],
        "value_decimal": _decimal_text(value),
        "evidence_pages": [_page_semantic_evidence(page)],
    }
    return observation, receipt


def compile_official_web_bundle(
    plan: Mapping[str, Any],
    *,
    capture_manifest: Mapping[str, Any],
    raw_pages: Mapping[str, bytes],
    plan_file_sha256: str,
    capture_manifest_sha256: str,
) -> OfficialWebCompilationResult:
    """Compile an in-memory, hash-bound official capture without I/O."""

    plan_hash = _required_hash(plan_file_sha256, "official_web_plan_file_hash_invalid")
    capture_hash = _required_hash(
        capture_manifest_sha256,
        "official_web_capture_file_hash_invalid",
    )
    scope, plan_pages = _validate_plan(_required_mapping(plan, "official_web_plan_not_object"))
    capture_pages = _validate_capture_manifest(
        _required_mapping(capture_manifest, "official_web_capture_manifest_not_object"),
        plan_pages=plan_pages,
        plan_file_sha256=plan_hash,
    )
    if set(raw_pages) != set(plan_pages):
        raise OfficialWebCompilerError("official_web_raw_page_set_mismatch")

    parsed_pages: list[_PageEvidence] = []
    for page_id in sorted(plan_pages):
        planned = plan_pages[page_id]
        captured = capture_pages[page_id]
        body = raw_pages[page_id]
        if not isinstance(body, bytes):
            raise OfficialWebCompilerError("official_web_body_bytes_required")
        body_hash = _sha256(body)
        if body_hash != captured["raw_sha256"]:
            raise OfficialWebCompilerError("official_web_raw_hash_mismatch")
        if len(body) != captured["size_bytes"]:
            raise OfficialWebCompilerError("official_web_raw_size_mismatch")
        parsed = _parse_page(
            str(planned["parser_id"]),
            body,
            source_url=str(planned["source_url"]),
        )
        if parsed.period != planned["expected_period"]:
            raise OfficialWebCompilerError("official_web_page_period_mismatch")
        fetched_at = parse_timestamp(
            captured["fetch_completed_at"], field_name="fetch_completed_at"
        )
        if fetched_at < parse_timestamp(parsed.release_at.isoformat(), field_name="release_at"):
            raise OfficialWebCompilerError("official_web_fetched_before_release")
        normalized_values = tuple(
            _Value(item.indicator_id, item.value, item.period or parsed.period)
            for item in parsed.values
        )
        parsed_pages.append(
            _PageEvidence(
                page_id=page_id,
                parser_id=str(planned["parser_id"]),
                parser_contract_sha256=str(planned["parser_contract_sha256"]),
                source_system=str(planned["source_system"]),
                source_url=str(planned["source_url"]),
                source_record_id=parsed.source_record_id,
                period=parsed.period,
                release_at=parsed.release_at,
                fetched_at=fetched_at,
                body_sha256=body_hash,
                body_size_bytes=len(body),
                article_title=parsed.article_title,
                values=normalized_values,
            )
        )

    candidates: dict[tuple[str, str], list[tuple[Decimal, _PageEvidence]]] = {}
    for page in parsed_pages:
        for item in page.values:
            if item.indicator_id not in _SUPPORTED_IDS:
                continue
            frequency = _FREQUENCY_UNIT[item.indicator_id][0]
            period_end = _period_end(item.period, frequency)
            key = (item.indicator_id, period_end)
            if key in scope:
                candidates.setdefault(key, []).append((item.value, page))
    if set(candidates) != scope:
        raise OfficialWebCompilerError("official_web_compiled_scope_missing")
    if any(len(items) != 1 for items in candidates.values()):
        raise OfficialWebCompilerError("official_web_compiled_scope_ambiguous")

    observations: list[MacroObservation] = []
    receipts: list[Mapping[str, Any]] = []
    for key in sorted(scope):
        value, page = candidates[key][0]
        period = _scope_period(*key)
        observation, receipt = _build_observation(key[0], period, value, page=page)
        observations.append(observation)
        receipts.append(receipt)
    normalized_observations = tuple(observations)
    normalized_receipts = tuple(receipts)
    if len(normalized_observations) != 36 or len(normalized_receipts) != 36:
        raise OfficialWebCompilerError("official_web_observation_count_invalid")
    if len({item.content_hash for item in normalized_observations}) != 36:
        raise OfficialWebCompilerError("official_web_observation_hash_collision")

    scope_rows = [
        {"indicator_id": indicator_id, "period_end": period_end}
        for indicator_id, period_end in sorted(scope)
    ]
    raw_set = [
        {
            "page_id": page.page_id,
            "raw_sha256": page.body_sha256,
            "size_bytes": page.body_size_bytes,
        }
        for page in sorted(parsed_pages, key=lambda item: item.page_id)
    ]
    registry_ids = {item.indicator_id for item in NATIONAL_INDICATORS}
    unsupported = sorted(registry_ids - _SUPPORTED_IDS)
    manifest = {
        "schema_version": OFFICIAL_WEB_NORMALIZATION_SCHEMA,
        "status": "OK",
        "market": "CN",
        "plan_file_sha256": plan_hash,
        "capture_manifest_sha256": capture_hash,
        "parser_contract_sha256": {
            key: PARSER_CONTRACT_SHA256[key] for key in sorted(_PARSER_SOURCE_SYSTEM)
        },
        "observation_count": len(normalized_observations),
        "receipt_count": len(normalized_receipts),
        "quarantine_count": 0,
        "supported_indicator_ids": sorted(_SUPPORTED_IDS),
        "unsupported_indicator_ids": unsupported,
        "unsupported_indicator_reasons": {
            "cn.fiscal_expenditure_yoy": "no_exact_timestamp_official_capture_in_bundle",
            "cn.social_financing_flow": "rounded_cumulative_difference_not_exact",
            "market.breadth": "strict_local_market_observation_required",
            "market.volatility_percentile": "strict_local_market_observation_required",
        },
        "national_registry_coverage": len(_SUPPORTED_IDS) / len(registry_ids),
        "expected_scope": scope_rows,
        "expected_scope_hash": canonical_hash({"scope": scope_rows}),
        "missing_scope": [],
        "unexpected_scope": [],
        "raw_page_count": len(raw_set),
        "raw_set_hash": canonical_hash({"raw_pages": raw_set}),
        "receipt_set_hash": canonical_hash(
            {"receipts": [dict(item) for item in normalized_receipts]}
        ),
        "social_financing_flow_emitted": False,
        "publishable": True,
        "production_eligible": True,
        "promoted": False,
        "applied": False,
    }
    return OfficialWebCompilationResult(
        normalized_observations,
        normalized_receipts,
        manifest,
    )


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_file_bytes(
    path: str | Path,
    *,
    error_prefix: str,
    max_bytes: int,
    require_private: bool = False,
) -> bytes:
    source = Path(path).expanduser()
    try:
        absolute = source.absolute()
        if absolute.is_symlink():
            raise OfficialWebCompilerError(f"{error_prefix}_symlink_rejected")
        resolved = absolute.resolve(strict=True)
        before = os.lstat(resolved)
    except OfficialWebCompilerError:
        raise
    except OSError as exc:
        raise OfficialWebCompilerError(f"{error_prefix}_missing_or_unsafe") from exc
    if not stat.S_ISREG(before.st_mode):
        raise OfficialWebCompilerError(f"{error_prefix}_not_regular_file")
    if require_private and before.st_mode & 0o077:
        raise OfficialWebCompilerError(f"{error_prefix}_permissions_unsafe")
    if before.st_size < 0 or before.st_size > max_bytes:
        raise OfficialWebCompilerError(f"{error_prefix}_size_invalid")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise OfficialWebCompilerError(f"{error_prefix}_open_failed") from exc
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(before):
            raise OfficialWebCompilerError(f"{error_prefix}_toctou_detected")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise OfficialWebCompilerError(f"{error_prefix}_size_invalid")
        opened_after = os.fstat(descriptor)
        try:
            path_after = os.lstat(resolved)
        except OSError as exc:
            raise OfficialWebCompilerError(f"{error_prefix}_toctou_detected") from exc
        if _stat_signature(opened_after) != _stat_signature(opened) or _stat_signature(
            path_after
        ) != _stat_signature(before):
            raise OfficialWebCompilerError(f"{error_prefix}_toctou_detected")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_object_bytes(
    path: str | Path,
    *,
    error_prefix: str,
    require_private: bool = False,
) -> tuple[Mapping[str, Any], bytes]:
    payload = _stable_file_bytes(
        path,
        error_prefix=error_prefix,
        max_bytes=_MAX_JSON_BYTES,
        require_private=require_private,
    )
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise OfficialWebCompilerError(f"{error_prefix}_json_invalid") from exc
    if not isinstance(decoded, Mapping):
        raise OfficialWebCompilerError(f"{error_prefix}_not_object")
    return decoded, payload


def _resolved_directory(path: str | Path, *, error_prefix: str) -> Path:
    source = Path(path).expanduser()
    try:
        if source.absolute().is_symlink():
            raise OfficialWebCompilerError(f"{error_prefix}_symlink_rejected")
        resolved = source.absolute().resolve(strict=True)
        metadata = os.lstat(resolved)
    except OfficialWebCompilerError:
        raise
    except OSError as exc:
        raise OfficialWebCompilerError(f"{error_prefix}_missing_or_unsafe") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise OfficialWebCompilerError(f"{error_prefix}_not_directory")
    return resolved


def _raw_input_path(raw_root: Path, raw_path: str) -> Path:
    relative = PurePosixPath(_safe_raw_path(raw_path))
    candidate = raw_root.joinpath(*relative.parts)
    try:
        candidate.relative_to(raw_root)
    except ValueError as exc:  # pragma: no cover - guarded by PurePosixPath checks
        raise OfficialWebCompilerError("official_web_raw_path_escape") from exc
    cursor = raw_root
    for component in relative.parts[:-1]:
        cursor = cursor / component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise OfficialWebCompilerError("official_web_raw_parent_missing_or_unsafe") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OfficialWebCompilerError("official_web_raw_parent_unsafe")
    return candidate


def _prepare_private_root(path: str | Path) -> Path:
    source = Path(path).expanduser().absolute()
    if source.exists():
        if source.is_symlink():
            raise OfficialWebCompilerError("official_web_output_root_symlink_rejected")
        root = source.resolve(strict=True)
        metadata = os.lstat(root)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
            raise OfficialWebCompilerError("official_web_output_root_unsafe")
        return root
    try:
        source.mkdir(parents=True, mode=0o700)
        os.chmod(source, 0o700)
        return source.resolve(strict=True)
    except OSError as exc:
        raise OfficialWebCompilerError("official_web_output_root_create_failed") from exc


def _mkdir_private(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise OfficialWebCompilerError("official_web_output_directory_create_failed") from exc


def _write_new_private_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise OfficialWebCompilerError("official_web_output_file_create_failed") from exc
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:  # pragma: no cover - defensive OS contract check
                raise OfficialWebCompilerError("official_web_output_write_failed")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _observation_jsonl_bytes(observations: Sequence[MacroObservation]) -> bytes:
    return _jsonl_bytes([item.to_dict() for item in observations])


def _capture_by_page(capture_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _required_list(capture_manifest.get("pages"), "official_web_capture_pages_not_list"):
        row = _required_mapping(raw, "official_web_capture_page_not_object")
        page_id = str(row.get("page_id") or "")
        if page_id in result:
            raise OfficialWebCompilerError("official_web_capture_page_id_invalid")
        result[page_id] = row
    return result


def persist_official_web_compilation(
    result: OfficialWebCompilationResult,
    *,
    plan_bytes: bytes,
    capture_manifest_bytes: bytes,
    raw_pages: Mapping[str, bytes],
    output_root: str | Path,
    run_id: str,
) -> dict[str, str]:
    """Persist exact input bytes and compiler artifacts with no clobber."""

    if not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise OfficialWebCompilerError("official_web_run_id_unsafe")
    if _sha256(plan_bytes) != result.manifest["plan_file_sha256"]:
        raise OfficialWebCompilerError("official_web_persist_plan_hash_mismatch")
    if _sha256(capture_manifest_bytes) != result.manifest["capture_manifest_sha256"]:
        raise OfficialWebCompilerError("official_web_persist_capture_hash_mismatch")
    try:
        capture_manifest = json.loads(capture_manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise OfficialWebCompilerError("official_web_persist_capture_invalid") from exc
    capture_pages = _capture_by_page(
        _required_mapping(capture_manifest, "official_web_capture_not_object")
    )
    if set(raw_pages) != set(capture_pages):
        raise OfficialWebCompilerError("official_web_persist_raw_page_set_mismatch")

    root = _prepare_private_root(output_root)
    market_root = root / "CN"
    if market_root.exists():
        metadata = os.lstat(market_root)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_mode & 0o077
        ):
            raise OfficialWebCompilerError("official_web_market_root_unsafe")
    else:
        _mkdir_private(market_root)
    final = market_root / run_id
    if final.exists() or final.is_symlink():
        raise OfficialWebCompilerError("official_web_output_exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=market_root))
    os.chmod(staging, 0o700)
    raw_directory = staging / "raw"
    _mkdir_private(raw_directory)
    try:
        artifacts: dict[str, bytes] = {
            "plan.json": plan_bytes,
            "capture_manifest.json": capture_manifest_bytes,
            "observations.jsonl": _observation_jsonl_bytes(result.observations),
            "normalization_receipts.jsonl": _jsonl_bytes(result.receipts),
            "quarantine.jsonl": b"",
        }
        raw_artifacts: dict[str, str] = {}
        for page_id, body in sorted(raw_pages.items()):
            if not _PAGE_ID_RE.fullmatch(page_id):
                raise OfficialWebCompilerError("official_web_page_id_invalid")
            name = f"raw/{page_id}.html"
            raw_artifacts[page_id] = name
            artifacts[name] = body
        for name, payload in artifacts.items():
            _write_new_private_file(staging / name, payload)
        artifact_hashes = {name: _sha256(payload) for name, payload in sorted(artifacts.items())}
        persisted_manifest = {
            **dict(result.manifest),
            "raw_artifacts": raw_artifacts,
            "artifact_sha256": artifact_hashes,
        }
        manifest_bytes = (
            json.dumps(
                persisted_manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_new_private_file(staging / "normalization_manifest.json", manifest_bytes)
        _fsync_directory(raw_directory)
        _fsync_directory(staging)
        os.replace(staging, final)
        _fsync_directory(market_root)
        return {
            "bundle": str(final),
            "manifest": str(final / "normalization_manifest.json"),
            "manifest_sha256": _sha256(manifest_bytes),
            "plan_sha256": str(result.manifest["plan_file_sha256"]),
            "observations": str(final / "observations.jsonl"),
            "receipts": str(final / "normalization_receipts.jsonl"),
            "quarantine": str(final / "quarantine.jsonl"),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def compile_official_web_bundle_file(
    plan_path: str | Path,
    *,
    capture_manifest_path: str | Path,
    raw_root: str | Path,
    output_root: str | Path = "results/macro_official_web",
    run_id: str,
) -> dict[str, Any]:
    """Stable-read, compile, and persist one complete official bundle."""

    plan, plan_bytes = _json_object_bytes(plan_path, error_prefix="official_web_plan_file")
    capture, capture_bytes = _json_object_bytes(
        capture_manifest_path,
        error_prefix="official_web_capture_file",
    )
    plan_hash = _sha256(plan_bytes)
    capture_hash = _sha256(capture_bytes)
    _, plan_pages = _validate_plan(plan)
    capture_pages = _validate_capture_manifest(
        capture,
        plan_pages=plan_pages,
        plan_file_sha256=plan_hash,
    )
    root = _resolved_directory(raw_root, error_prefix="official_web_raw_root")
    raw_pages: dict[str, bytes] = {}
    for page_id, row in sorted(capture_pages.items()):
        path = _raw_input_path(root, str(row["raw_path"]))
        raw_pages[page_id] = _stable_file_bytes(
            path,
            error_prefix="official_web_raw_file",
            max_bytes=_MAX_HTML_BYTES,
        )
    result = compile_official_web_bundle(
        plan,
        capture_manifest=capture,
        raw_pages=raw_pages,
        plan_file_sha256=plan_hash,
        capture_manifest_sha256=capture_hash,
    )
    persisted = persist_official_web_compilation(
        result,
        plan_bytes=plan_bytes,
        capture_manifest_bytes=capture_bytes,
        raw_pages=raw_pages,
        output_root=output_root,
        run_id=run_id,
    )
    return {
        **dict(result.manifest),
        "normalization_manifest_sha256": persisted["manifest_sha256"],
        "artifacts": persisted,
    }


def _load_jsonl(payload: bytes, *, error_prefix: str) -> list[Mapping[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OfficialWebCompilerError(f"{error_prefix}_not_utf8") from exc
    rows: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except Exception as exc:
            raise OfficialWebCompilerError(f"{error_prefix}_json_invalid") from exc
        if not isinstance(decoded, Mapping):
            raise OfficialWebCompilerError(f"{error_prefix}_row_not_object")
        rows.append(dict(decoded))
    return rows


def _safe_bundle_artifact(parent: Path, name: Any) -> Path:
    raw = str(name or "")
    if not raw or "\\" in raw:
        raise OfficialWebCompilerError("official_web_artifact_path_unsafe")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or str(relative) != raw
    ):
        raise OfficialWebCompilerError("official_web_artifact_path_unsafe")
    path = parent.joinpath(*relative.parts)
    try:
        path.relative_to(parent)
    except ValueError as exc:  # pragma: no cover - guarded above
        raise OfficialWebCompilerError("official_web_artifact_path_escape") from exc
    return path


def recompile_official_web_bundle(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_plan_sha256: str,
) -> OfficialWebCompilationResult:
    """Hash-check and deterministically recompile a persisted bundle."""

    expected_manifest = _required_hash(
        expected_manifest_sha256,
        "official_web_expected_manifest_hash_invalid",
    )
    expected_plan = _required_hash(
        expected_plan_sha256,
        "official_web_expected_plan_hash_invalid",
    )
    manifest, manifest_bytes = _json_object_bytes(
        manifest_path,
        error_prefix="official_web_manifest_file",
        require_private=True,
    )
    if _sha256(manifest_bytes) != expected_manifest:
        raise OfficialWebCompilerError("official_web_manifest_hash_mismatch")
    if manifest.get("schema_version") != OFFICIAL_WEB_NORMALIZATION_SCHEMA:
        raise OfficialWebCompilerError("official_web_manifest_schema_invalid")
    if manifest.get("status") != "OK" or manifest.get("publishable") is not True:
        raise OfficialWebCompilerError("official_web_bundle_not_publishable")
    parent = Path(manifest_path).expanduser().absolute().resolve(strict=True).parent
    raw_artifacts = _required_mapping(
        manifest.get("raw_artifacts"), "official_web_raw_artifacts_missing"
    )
    artifact_hashes = _required_mapping(
        manifest.get("artifact_sha256"), "official_web_artifact_hashes_missing"
    )
    fixed_names = {
        "plan.json",
        "capture_manifest.json",
        "observations.jsonl",
        "normalization_receipts.jsonl",
        "quarantine.jsonl",
    }
    declared_names = fixed_names | {str(value) for value in raw_artifacts.values()}
    if set(artifact_hashes) != declared_names:
        raise OfficialWebCompilerError("official_web_artifact_set_mismatch")
    payloads: dict[str, bytes] = {}
    for name in sorted(declared_names):
        path = _safe_bundle_artifact(parent, name)
        payload = _stable_file_bytes(
            path,
            error_prefix="official_web_artifact",
            max_bytes=(
                _MAX_JSONL_BYTES
                if name.endswith(".jsonl")
                else (_MAX_HTML_BYTES if name.startswith("raw/") else _MAX_JSON_BYTES)
            ),
            require_private=True,
        )
        expected_hash = _required_hash(
            artifact_hashes.get(name), "official_web_artifact_hash_invalid"
        )
        if _sha256(payload) != expected_hash:
            raise OfficialWebCompilerError("official_web_artifact_hash_mismatch")
        payloads[name] = payload
    if _sha256(payloads["plan.json"]) != expected_plan:
        raise OfficialWebCompilerError("official_web_plan_hash_mismatch")
    try:
        plan = json.loads(payloads["plan.json"].decode("utf-8"))
        capture = json.loads(payloads["capture_manifest.json"].decode("utf-8"))
    except Exception as exc:
        raise OfficialWebCompilerError("official_web_input_artifact_invalid") from exc
    if not isinstance(plan, Mapping) or not isinstance(capture, Mapping):
        raise OfficialWebCompilerError("official_web_input_artifact_shape_invalid")
    capture_pages = _capture_by_page(capture)
    if set(raw_artifacts) != set(capture_pages):
        raise OfficialWebCompilerError("official_web_raw_artifact_page_set_mismatch")
    raw_pages = {
        page_id: payloads[str(raw_artifacts[page_id])] for page_id in sorted(raw_artifacts)
    }
    recomputed = compile_official_web_bundle(
        plan,
        capture_manifest=capture,
        raw_pages=raw_pages,
        plan_file_sha256=_sha256(payloads["plan.json"]),
        capture_manifest_sha256=_sha256(payloads["capture_manifest.json"]),
    )
    persisted_observations = _load_jsonl(
        payloads["observations.jsonl"], error_prefix="official_web_observations"
    )
    persisted_receipts = _load_jsonl(
        payloads["normalization_receipts.jsonl"],
        error_prefix="official_web_receipts",
    )
    persisted_quarantine = _load_jsonl(
        payloads["quarantine.jsonl"], error_prefix="official_web_quarantine"
    )
    if persisted_quarantine:
        raise OfficialWebCompilerError("official_web_quarantine_not_empty")
    if persisted_observations != [item.to_dict() for item in recomputed.observations]:
        raise OfficialWebCompilerError("official_web_observation_recompile_mismatch")
    if persisted_receipts != [dict(item) for item in recomputed.receipts]:
        raise OfficialWebCompilerError("official_web_receipt_recompile_mismatch")
    core_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"raw_artifacts", "artifact_sha256"}
    }
    if core_manifest != dict(recomputed.manifest):
        raise OfficialWebCompilerError("official_web_manifest_recompile_mismatch")
    return recomputed


__all__ = [
    "NBS_NATIONAL_ECONOMY_PARSER",
    "NBS_NATIONAL_ECONOMY_PARSER_CONTRACT_SHA256",
    "NBS_OFFICIAL_PMI_PARSER",
    "NBS_OFFICIAL_PMI_PARSER_CONTRACT_SHA256",
    "NBS_QUARTERLY_GDP_PARSER",
    "NBS_QUARTERLY_GDP_PARSER_CONTRACT_SHA256",
    "OFFICIAL_WEB_CAPTURE_SCHEMA",
    "OFFICIAL_WEB_NORMALIZATION_SCHEMA",
    "OFFICIAL_WEB_PLAN_SCHEMA",
    "OfficialWebCompilationResult",
    "OfficialWebCompilerError",
    "PARSER_CONTRACT_SHA256",
    "PBC_FINANCIAL_STATISTICS_PARSER",
    "PBC_MONEY_STOCK_PARSER",
    "PBC_MONEY_STOCK_PARSER_CONTRACT_SHA256",
    "compile_official_web_bundle",
    "compile_official_web_bundle_file",
    "persist_official_web_compilation",
    "recompile_official_web_bundle",
]
