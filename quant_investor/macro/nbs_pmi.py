"""Strict official NBS HTML capture for the CN manufacturing PMI.

The adapter is intentionally narrow.  It accepts only a canonical National
Bureau of Statistics article URL, follows redirects itself, and extracts the
headline PMI from the formal monthly release paragraph.  It does not discover
URLs, credentials, or substitute another provider.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, urljoin, urlsplit
from zoneinfo import ZoneInfo

import requests

NBS_PMI_PARSER_VERSION = "nbs-cn-pmi-html.v2"
NBS_PMI_MAX_BODY_BYTES = 2 * 1024 * 1024
NBS_PMI_MAX_REDIRECTS = 3
NBS_PMI_MAX_ATTEMPTS = 3
NBS_PMI_MAX_TRANSFER_SECONDS = 30.0

_ALLOWED_HOSTS = frozenset({"stats.gov.cn", "www.stats.gov.cn"})
_MANAGED_INTERCEPT_NETWORK = ipaddress.ip_network("198.18.0.0/15")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_RECORD_RE = re.compile(r"^(t(?P<date>\d{8})_(?P<serial>\d+))\.html$")
_TITLE_RE = re.compile(
    r"^(?P<year>20\d{2})年(?P<month>1[0-2]|[1-9])月中国采购经理指数运行情况$"
)
_PMI_RE = re.compile(
    r"(?P<month>1[0-2]|[1-9])月份，(?:中国)?制造业采购经理指数（PMI）"
    r"为(?P<value>(?:100(?:\.0+)?|\d{1,2}(?:\.\d+)?))[%％]"
)
_PUB_DATE_RE = re.compile(
    r"(?P<year>20\d{2})[-/](?P<month>\d{2})[-/](?P<day>\d{2})"
    r"[ T](?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
)
_SECRET_QUERY_FRAGMENTS = frozenset(
    {
        "accesskey",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "signature",
        "token",
    }
)
_PARSER_CONTRACT = {
    "article_title": "exact YYYY年M月中国采购经理指数运行情况",
    "body_encoding": "utf-8-strict",
    "issuer_hosts": sorted(_ALLOWED_HOSTS),
    "period_source": "ArticleTitle",
    "period_release_order": "period month must not follow PubDate month",
    "pmi_source": "formal paragraph 制造业采购经理指数（PMI）为N%",
    "publication_source": "PubDate Asia/Shanghai",
    "record_id": "URL basename tYYYYMMDD_N.html",
    "unique_value": True,
    "value_range": [0.0, 100.0],
    "version": NBS_PMI_PARSER_VERSION,
}
NBS_PMI_PARSER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _PARSER_CONTRACT,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class NbsPmiError(RuntimeError):
    """Base class for a rejected or unavailable NBS PMI capture."""


class NbsPmiPermanentError(NbsPmiError):
    """A request or response violated the formal source contract."""


class NbsPmiTransientError(NbsPmiError):
    """The official source was temporarily unavailable."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class NbsPmiParsed:
    """Deterministic fields extracted from one exact response entity."""

    month: str
    value: float
    source_url: str
    source_record_id: str
    article_title: str
    source_release_at: str
    body_sha256: str
    body_size_bytes: int
    parser_version: str
    parser_contract_sha256: str


@dataclass(frozen=True)
class NbsPmiCapture:
    """Official response capture ready for a provenance sidecar."""

    month: str
    value: float
    source_url: str
    source_record_id: str
    article_title: str
    source_release_at: str
    fetch_started_at: str
    fetch_completed_at: str
    content_type: str
    charset: str
    body_bytes: bytes
    body_sha256: str
    body_size_bytes: int
    parser_version: str
    parser_contract_sha256: str
    redirect_chain: tuple[str, ...]


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, Any]
    content: bytes


class _Transport(Protocol):
    def __call__(
        self,
        url: str,
        *,
        allow_redirects: bool,
        timeout: tuple[float, float],
        headers: Mapping[str, str],
    ) -> _Response:
        ...


Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]


@dataclass
class _BufferedResponse:
    status_code: int
    headers: Mapping[str, Any]
    content: bytes


class _DefaultTransport:
    def __call__(
        self,
        url: str,
        *,
        allow_redirects: bool,
        timeout: tuple[float, float],
        headers: Mapping[str, str],
    ) -> _Response:
        session = requests.Session()
        session.trust_env = False
        response = None
        try:
            response = session.get(
                url,
                allow_redirects=allow_redirects,
                timeout=timeout,
                headers=dict(headers),
                stream=True,
            )
            response_headers = dict(response.headers)
            status = int(response.status_code)
            if status != 200:
                return _BufferedResponse(
                    status_code=status,
                    headers=response_headers,
                    content=b"",
                )
            declared = response_headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(str(declared))
                except ValueError as exc:
                    raise NbsPmiPermanentError(
                        "nbs_pmi_content_length_invalid"
                    ) from exc
                if not 0 <= declared_size <= NBS_PMI_MAX_BODY_BYTES:
                    raise NbsPmiPermanentError(
                        "nbs_pmi_body_size_invalid"
                    )
            deadline = time.monotonic() + NBS_PMI_MAX_TRANSFER_SECONDS
            body = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if time.monotonic() > deadline:
                    raise NbsPmiTransientError(
                        "nbs_pmi_transfer_deadline_exceeded"
                    )
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > NBS_PMI_MAX_BODY_BYTES:
                    raise NbsPmiPermanentError(
                        "nbs_pmi_body_size_invalid"
                    )
            return _BufferedResponse(
                status_code=status,
                headers=response_headers,
                content=bytes(body),
            )
        finally:
            if response is not None:
                response.close()
            session.close()

    def resolve(self, host: str, port: int) -> tuple[str, ...]:
        addresses = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )
        return tuple(str(item[4][0]) for item in addresses)


class _NbsHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.paragraphs: list[str] = []
        self._paragraph_depth = 0
        self._paragraph_parts: list[str] = []

    def _handle_meta(self, attrs: Sequence[tuple[str, str | None]]) -> None:
        values = {str(key).lower(): value for key, value in attrs}
        name = values.get("name")
        content = values.get("content")
        if name is not None and content is not None:
            self.meta.setdefault(str(name).casefold(), []).append(str(content))

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.casefold() == "meta":
            self._handle_meta(attrs)
        if tag.casefold() == "p":
            if self._paragraph_depth == 0:
                self._paragraph_parts = []
            self._paragraph_depth += 1

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.casefold() == "meta":
            self._handle_meta(attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "p" or self._paragraph_depth == 0:
            return
        self._paragraph_depth -= 1
        if self._paragraph_depth == 0:
            self.paragraphs.append("".join(self._paragraph_parts))
            self._paragraph_parts = []

    def handle_data(self, data: str) -> None:
        if self._paragraph_depth:
            self._paragraph_parts.append(data)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("\u3000", "")).strip()


def _single_meta(parser: _NbsHtmlParser, name: str) -> str:
    if name.casefold() == "articletitle":
        values = [
            _normalized_text(item)
            for item in parser.meta.get(name.casefold(), [])
        ]
    else:
        values = [
            re.sub(r"\s+", " ", item).strip()
            for item in parser.meta.get(name.casefold(), [])
        ]
    unique = tuple(dict.fromkeys(item for item in values if item))
    if len(unique) != 1:
        raise NbsPmiPermanentError(f"nbs_pmi_meta_{name.lower()}_not_unique")
    return unique[0]


def _query_is_sensitive(query: str) -> bool:
    if not query:
        return False
    if re.search(r"%(?![0-9A-Fa-f]{2})", query):
        return True
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        return True
    for key, _value in pairs:
        normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
        if any(fragment in normalized for fragment in _SECRET_QUERY_FRAGMENTS):
            return True
    return False


def _validate_url_structure(url: str) -> tuple[str, str, int]:
    candidate = str(url or "").strip()
    if candidate != url or not candidate:
        raise NbsPmiPermanentError("nbs_pmi_url_invalid")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise NbsPmiPermanentError("nbs_pmi_url_invalid") from exc
    host = str(parsed.hostname or "").casefold()
    if parsed.scheme.casefold() != "https":
        raise NbsPmiPermanentError("nbs_pmi_url_https_required")
    if host not in _ALLOWED_HOSTS:
        raise NbsPmiPermanentError("nbs_pmi_url_host_rejected")
    if parsed.username is not None or parsed.password is not None:
        raise NbsPmiPermanentError("nbs_pmi_url_userinfo_rejected")
    if port not in (None, 443):
        raise NbsPmiPermanentError("nbs_pmi_url_port_rejected")
    if parsed.fragment:
        raise NbsPmiPermanentError("nbs_pmi_url_fragment_rejected")
    if _query_is_sensitive(parsed.query):
        raise NbsPmiPermanentError("nbs_pmi_url_sensitive_query_rejected")
    basename = parsed.path.rsplit("/", 1)[-1]
    record = _RECORD_RE.fullmatch(basename)
    if record is None:
        raise NbsPmiPermanentError("nbs_pmi_url_record_id_invalid")
    return host, record.group(1), port or 443


def validate_nbs_pmi_url(url: str) -> str:
    """Validate and return one exact issuer-bound NBS PMI article URL."""

    _validate_url_structure(url)
    return url


def _resolved_ips(
    transport: _Transport,
    *,
    host: str,
    port: int,
) -> tuple[str, ...]:
    resolver = getattr(transport, "resolve", None)
    try:
        if callable(resolver):
            raw_addresses = resolver(host, port)
        else:
            raw_addresses = tuple(
                str(item[4][0])
                for item in socket.getaddrinfo(
                    host,
                    port,
                    type=socket.SOCK_STREAM,
                )
            )
    except Exception as exc:
        raise NbsPmiTransientError("nbs_pmi_dns_unavailable") from exc
    addresses = tuple(dict.fromkeys(str(item) for item in raw_addresses))
    if not addresses:
        raise NbsPmiTransientError("nbs_pmi_dns_empty")
    return addresses


def _validate_public_resolution(
    transport: _Transport,
    *,
    host: str,
    port: int,
) -> None:
    for raw_address in _resolved_ips(transport, host=host, port=port):
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise NbsPmiPermanentError("nbs_pmi_dns_address_invalid") from exc
        # Some managed macOS network stacks return RFC 2544 benchmark-space
        # addresses for an authenticated transparent TLS intercept.  The
        # destination host remains issuer-allowlisted and certificate-checked.
        managed_intercept = (
            isinstance(address, ipaddress.IPv4Address)
            and address in _MANAGED_INTERCEPT_NETWORK
        )
        if (
            address.is_multicast
            or (not address.is_global and not managed_intercept)
        ):
            raise NbsPmiPermanentError("nbs_pmi_url_ip_rejected")


def _record_date(source_url: str) -> tuple[str, datetime]:
    _host, record_id, _port = _validate_url_structure(source_url)
    match = _RECORD_RE.fullmatch(urlsplit(source_url).path.rsplit("/", 1)[-1])
    if match is None:  # pragma: no cover - guarded by _validate_url_structure
        raise NbsPmiPermanentError("nbs_pmi_url_record_id_invalid")
    try:
        date_value = datetime.strptime(match.group("date"), "%Y%m%d")
    except ValueError as exc:
        raise NbsPmiPermanentError("nbs_pmi_url_record_date_invalid") from exc
    return record_id, date_value


def _publication_timestamp(value: str) -> datetime:
    match = _PUB_DATE_RE.fullmatch(value)
    if match is None:
        raise NbsPmiPermanentError("nbs_pmi_pubdate_invalid")
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
        raise NbsPmiPermanentError("nbs_pmi_pubdate_invalid") from exc


def parse_nbs_cn_pmi_html(
    body_bytes: bytes,
    *,
    source_url: str,
) -> NbsPmiParsed:
    """Parse one decompressed NBS HTML entity without network or clock I/O."""

    if not isinstance(body_bytes, bytes):
        raise NbsPmiPermanentError("nbs_pmi_body_bytes_required")
    if not body_bytes or len(body_bytes) > NBS_PMI_MAX_BODY_BYTES:
        raise NbsPmiPermanentError("nbs_pmi_body_size_invalid")
    record_id, record_date = _record_date(source_url)
    try:
        html = body_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise NbsPmiPermanentError("nbs_pmi_body_not_utf8") from exc
    parser = _NbsHtmlParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        raise NbsPmiPermanentError("nbs_pmi_html_parse_failed") from exc

    article_title = _single_meta(parser, "ArticleTitle")
    title_match = _TITLE_RE.fullmatch(article_title)
    if title_match is None:
        raise NbsPmiPermanentError("nbs_pmi_article_title_invalid")
    year = int(title_match.group("year"))
    month_number = int(title_match.group("month"))
    month = f"{year:04d}{month_number:02d}"

    pubdate = _publication_timestamp(_single_meta(parser, "PubDate"))
    if pubdate.date() != record_date.date():
        raise NbsPmiPermanentError("nbs_pmi_pubdate_record_mismatch")
    if (year, month_number) > (pubdate.year, pubdate.month):
        raise NbsPmiPermanentError("nbs_pmi_period_after_pubdate")

    matches: list[tuple[int, float]] = []
    for paragraph in parser.paragraphs:
        normalized = _normalized_text(paragraph)
        for match in _PMI_RE.finditer(normalized):
            value = float(match.group("value"))
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise NbsPmiPermanentError("nbs_pmi_value_out_of_range")
            matches.append((int(match.group("month")), value))
    if not matches:
        raise NbsPmiPermanentError("nbs_pmi_formal_paragraph_missing")
    if any(item_month != month_number for item_month, _value in matches):
        raise NbsPmiPermanentError("nbs_pmi_paragraph_period_mismatch")
    unique_values = tuple(dict.fromkeys(value for _item_month, value in matches))
    if len(unique_values) != 1:
        raise NbsPmiPermanentError("nbs_pmi_value_not_unique")

    return NbsPmiParsed(
        month=month,
        value=unique_values[0],
        source_url=source_url,
        source_record_id=record_id,
        article_title=article_title,
        source_release_at=pubdate.isoformat(timespec="seconds"),
        body_sha256=hashlib.sha256(body_bytes).hexdigest(),
        body_size_bytes=len(body_bytes),
        parser_version=NBS_PMI_PARSER_VERSION,
        parser_contract_sha256=NBS_PMI_PARSER_CONTRACT_SHA256,
    )


def _response_headers(response: _Response) -> dict[str, str]:
    try:
        return {
            str(key).strip().casefold(): str(value).strip()
            for key, value in response.headers.items()
        }
    except Exception as exc:
        raise NbsPmiPermanentError("nbs_pmi_response_headers_invalid") from exc


def _html_content_type(headers: Mapping[str, str]) -> tuple[str, str]:
    raw = str(headers.get("content-type") or "")
    parts = [item.strip() for item in raw.split(";")]
    if not parts or parts[0].casefold() != "text/html":
        raise NbsPmiPermanentError("nbs_pmi_content_type_invalid")
    charsets: list[str] = []
    for item in parts[1:]:
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.strip().casefold() == "charset":
            normalized = value.strip()
            if normalized.startswith(("\"", "'")) or normalized.endswith(
                ("\"", "'")
            ):
                if (
                    len(normalized) < 2
                    or normalized[0] != normalized[-1]
                    or normalized[0] not in {"\"", "'"}
                ):
                    raise NbsPmiPermanentError("nbs_pmi_charset_invalid")
                normalized = normalized[1:-1]
            charsets.append(normalized.casefold())
    unique_charsets = tuple(dict.fromkeys(charsets))
    if len(unique_charsets) > 1 or (
        unique_charsets and unique_charsets[0] not in {"utf-8", "utf8"}
    ):
        raise NbsPmiPermanentError("nbs_pmi_charset_invalid")
    return "text/html", "utf-8"


def _response_body(response: _Response, headers: Mapping[str, str]) -> bytes:
    declared = headers.get("content-length")
    if declared:
        try:
            declared_size = int(declared)
        except ValueError as exc:
            raise NbsPmiPermanentError("nbs_pmi_content_length_invalid") from exc
        if declared_size < 0 or declared_size > NBS_PMI_MAX_BODY_BYTES:
            raise NbsPmiPermanentError("nbs_pmi_body_size_invalid")
    try:
        content = response.content
    except Exception as exc:
        raise NbsPmiTransientError("nbs_pmi_response_body_unavailable") from exc
    if not isinstance(content, bytes):
        raise NbsPmiPermanentError("nbs_pmi_response_body_invalid")
    if not content or len(content) > NBS_PMI_MAX_BODY_BYTES:
        raise NbsPmiPermanentError("nbs_pmi_body_size_invalid")
    return content


def _bounded_retry_after(headers: Mapping[str, str]) -> float | None:
    raw = str(headers.get("retry-after") or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0.0:
        return None
    return min(seconds, 1.0)


def _request(transport: _Transport, url: str) -> _Response:
    try:
        response = transport(
            url,
            allow_redirects=False,
            timeout=(5.0, 20.0),
            headers={
                "Accept": "text/html",
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": "QuantInvestor/14 official-macro-capture",
            },
        )
    except (NbsPmiPermanentError, NbsPmiTransientError):
        raise
    except requests.exceptions.SSLError as exc:
        raise NbsPmiPermanentError(
            "nbs_pmi_tls_verification_failed"
        ) from exc
    except Exception as exc:
        raise NbsPmiTransientError("nbs_pmi_network_unavailable") from exc
    if not hasattr(response, "status_code"):
        raise NbsPmiPermanentError("nbs_pmi_response_contract_invalid")
    return response


def _fetch_once(
    initial_url: str,
    *,
    transport: _Transport,
) -> tuple[bytes, str, str, str, tuple[str, ...]]:
    current_url = initial_url
    chain: list[str] = []
    redirect_count = 0
    while True:
        host, _record_id, port = _validate_url_structure(current_url)
        _validate_public_resolution(transport, host=host, port=port)
        chain.append(current_url)
        response = _request(transport, current_url)
        status = response.status_code
        if isinstance(status, bool) or not isinstance(status, int):
            raise NbsPmiPermanentError("nbs_pmi_response_status_invalid")
        headers = _response_headers(response)
        if status in _REDIRECT_STATUSES:
            if redirect_count >= NBS_PMI_MAX_REDIRECTS:
                raise NbsPmiPermanentError("nbs_pmi_redirect_limit_exceeded")
            location = headers.get("location")
            if not location:
                raise NbsPmiPermanentError("nbs_pmi_redirect_location_missing")
            target = urljoin(current_url, location)
            _validate_url_structure(target)
            current_url = target
            redirect_count += 1
            continue
        if status == 429 or 500 <= status <= 599:
            raise NbsPmiTransientError(
                f"nbs_pmi_http_transient:{status}",
                retry_after_seconds=_bounded_retry_after(headers),
            )
        if status != 200:
            raise NbsPmiPermanentError(f"nbs_pmi_http_permanent:{status}")
        content_type, charset = _html_content_type(headers)
        body = _response_body(response, headers)
        return body, current_url, content_type, charset, tuple(chain)


def _clock_value(clock: Clock, field: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise NbsPmiPermanentError(f"nbs_pmi_{field}_clock_invalid")
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_nbs_cn_pmi(
    url: str,
    *,
    transport: _Transport | None = None,
    clock: Clock = _utc_now,
    sleeper: Sleeper = time.sleep,
    max_attempts: int = NBS_PMI_MAX_ATTEMPTS,
) -> NbsPmiCapture:
    """Fetch and capture one official NBS manufacturing-PMI release.

    Only transport/DNS failures, HTTP 429, and HTTP 5xx are retried.  URL,
    redirect, response-contract, encoding, and parsing failures are permanent.
    """

    if isinstance(max_attempts, bool) or not 1 <= max_attempts <= NBS_PMI_MAX_ATTEMPTS:
        raise ValueError("max_attempts must be between 1 and 3")
    _validate_url_structure(url)
    active_transport = transport or _DefaultTransport()
    started = _clock_value(clock, "started")
    last_error: NbsPmiTransientError | None = None
    response_bundle: tuple[bytes, str, str, str, tuple[str, ...]] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response_bundle = _fetch_once(url, transport=active_transport)
            break
        except NbsPmiTransientError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            delay = (
                exc.retry_after_seconds
                if exc.retry_after_seconds is not None
                else min(0.25 * (2 ** (attempt - 1)), 1.0)
            )
            sleeper(delay)
    if response_bundle is None:  # pragma: no cover - loop always returns or raises
        if last_error is not None:
            raise last_error
        raise NbsPmiTransientError("nbs_pmi_fetch_unavailable")
    body, final_url, content_type, charset, redirect_chain = response_bundle
    completed = _clock_value(clock, "completed")
    if completed < started:
        raise NbsPmiPermanentError("nbs_pmi_fetch_clock_reversed")
    parsed = parse_nbs_cn_pmi_html(body, source_url=final_url)
    return NbsPmiCapture(
        month=parsed.month,
        value=parsed.value,
        source_url=parsed.source_url,
        source_record_id=parsed.source_record_id,
        article_title=parsed.article_title,
        source_release_at=parsed.source_release_at,
        fetch_started_at=started.isoformat(),
        fetch_completed_at=completed.isoformat(),
        content_type=content_type,
        charset=charset,
        body_bytes=body,
        body_sha256=parsed.body_sha256,
        body_size_bytes=parsed.body_size_bytes,
        parser_version=parsed.parser_version,
        parser_contract_sha256=parsed.parser_contract_sha256,
        redirect_chain=redirect_chain,
    )


__all__ = [
    "NBS_PMI_MAX_ATTEMPTS",
    "NBS_PMI_MAX_BODY_BYTES",
    "NBS_PMI_MAX_REDIRECTS",
    "NBS_PMI_MAX_TRANSFER_SECONDS",
    "NBS_PMI_PARSER_CONTRACT_SHA256",
    "NBS_PMI_PARSER_VERSION",
    "NbsPmiCapture",
    "NbsPmiError",
    "NbsPmiParsed",
    "NbsPmiPermanentError",
    "NbsPmiTransientError",
    "fetch_nbs_cn_pmi",
    "parse_nbs_cn_pmi_html",
    "validate_nbs_pmi_url",
]
