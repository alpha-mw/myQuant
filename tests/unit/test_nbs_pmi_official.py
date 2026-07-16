from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest
import requests

import quant_investor.macro.nbs_pmi as nbs_pmi_module
from quant_investor.macro.nbs_pmi import (
    NBS_PMI_MAX_BODY_BYTES,
    NBS_PMI_PARSER_CONTRACT_SHA256,
    NBS_PMI_PARSER_VERSION,
    NbsPmiCapture,
    NbsPmiPermanentError,
    NbsPmiTransientError,
    fetch_nbs_cn_pmi,
    parse_nbs_cn_pmi_html,
)

URL = "https://www.stats.gov.cn/sj/zxfb/202606/t20260630_1964032.html"
FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "macro"
    / "nbs_cn_pmi_202606_minimal.html"
)
PUBLIC_IP = "8.8.8.8"


@dataclass(frozen=True)
class _FakeResponse:
    status_code: int
    headers: Mapping[str, Any]
    content: bytes = b""


class _FakeTransport:
    def __init__(
        self,
        outcomes: list[_FakeResponse | Exception],
        *,
        resolved_ips: tuple[str, ...] = (PUBLIC_IP,),
    ) -> None:
        self.outcomes = list(outcomes)
        self.resolved_ips = resolved_ips
        self.calls: list[dict[str, object]] = []
        self.resolve_calls: list[tuple[str, int]] = []

    def resolve(self, host: str, port: int) -> tuple[str, ...]:
        self.resolve_calls.append((host, port))
        return self.resolved_ips

    def __call__(
        self,
        url: str,
        *,
        allow_redirects: bool,
        timeout: tuple[float, float],
        headers: Mapping[str, str],
    ) -> _FakeResponse:
        self.calls.append(
            {
                "url": url,
                "allow_redirects": allow_redirects,
                "timeout": timeout,
                "headers": dict(headers),
            }
        )
        if not self.outcomes:
            raise AssertionError("unexpected transport call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _StreamingResponse:
    def __init__(self, body: bytes) -> None:
        self.status_code = 200
        self.headers = {
            "Content-Type": "text/html; charset=UTF-8",
            "Content-Length": str(len(body)),
        }
        self._body = body
        self.closed = False

    def iter_content(self, *, chunk_size: int):
        for offset in range(0, len(self._body), chunk_size):
            yield self._body[offset : offset + chunk_size]

    def close(self) -> None:
        self.closed = True


class _StreamingSession:
    def __init__(self, response: _StreamingResponse) -> None:
        self.response = response
        self.trust_env = True
        self.closed = False
        self.call: dict[str, object] = {}

    def get(self, url: str, **kwargs: object) -> _StreamingResponse:
        self.call = {"url": url, **kwargs}
        return self.response

    def close(self) -> None:
        self.closed = True


def _body() -> bytes:
    return FIXTURE.read_bytes()


def _ok_response(
    body: bytes | None = None,
    *,
    content_type: str = "text/html; charset=UTF-8",
    headers: Mapping[str, str] | None = None,
) -> _FakeResponse:
    entity = _body() if body is None else body
    return _FakeResponse(
        200,
        {
            "Content-Type": content_type,
            "Content-Length": str(len(entity)),
            **dict(headers or {}),
        },
        entity,
    )


def _clock(*values: datetime):
    iterator = iter(values)
    return lambda: next(iterator)


def _replace(old: str, new: str) -> bytes:
    return _body().decode("utf-8").replace(old, new).encode("utf-8")


def test_pure_parser_extracts_exact_official_period_value_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_investor.macro.nbs_pmi.socket.getaddrinfo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pure parser must not resolve DNS")
        ),
    )
    body = _body()
    assert len(body) < 10 * 1024

    parsed = parse_nbs_cn_pmi_html(body, source_url=URL)

    assert parsed.month == "202606"
    assert parsed.value == 50.3
    assert parsed.source_url == URL
    assert parsed.source_record_id == "t20260630_1964032"
    assert parsed.article_title == "2026年6月中国采购经理指数运行情况"
    assert parsed.source_release_at == "2026-06-30T09:30:00+08:00"
    assert parsed.body_sha256 == hashlib.sha256(body).hexdigest()
    assert parsed.body_size_bytes == len(body)
    assert parsed.parser_version == NBS_PMI_PARSER_VERSION
    assert parsed.parser_contract_sha256 == NBS_PMI_PARSER_CONTRACT_SHA256
    assert len(parsed.parser_contract_sha256) == 64


def test_fetch_returns_complete_capture_from_decompressed_response_content() -> None:
    body = _body()
    start = datetime(2026, 6, 30, 1, 29, 59, tzinfo=timezone.utc)
    end = start + timedelta(seconds=2)
    transport = _FakeTransport([_ok_response(body)])

    capture = fetch_nbs_cn_pmi(
        URL,
        transport=transport,
        clock=_clock(start, end),
        sleeper=lambda _seconds: None,
    )

    assert isinstance(capture, NbsPmiCapture)
    assert capture.month == "202606"
    assert capture.value == 50.3
    assert capture.source_url == URL
    assert capture.source_record_id == "t20260630_1964032"
    assert capture.article_title == "2026年6月中国采购经理指数运行情况"
    assert capture.source_release_at == "2026-06-30T09:30:00+08:00"
    assert capture.fetch_started_at == start.isoformat()
    assert capture.fetch_completed_at == end.isoformat()
    assert capture.content_type == "text/html"
    assert capture.charset == "utf-8"
    assert capture.body_bytes is body
    assert capture.body_sha256 == hashlib.sha256(body).hexdigest()
    assert capture.body_size_bytes == len(body)
    assert capture.parser_version == NBS_PMI_PARSER_VERSION
    assert capture.parser_contract_sha256 == NBS_PMI_PARSER_CONTRACT_SHA256
    assert capture.redirect_chain == (URL,)
    assert transport.resolve_calls == [("www.stats.gov.cn", 443)]
    assert transport.calls[0]["allow_redirects"] is False
    assert transport.calls[0]["timeout"] == (5.0, 20.0)


def test_default_transport_streams_with_environment_proxies_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StreamingResponse(_body())
    session = _StreamingSession(response)
    monkeypatch.setattr(
        nbs_pmi_module.requests,
        "Session",
        lambda: session,
    )
    monkeypatch.setattr(
        nbs_pmi_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", (PUBLIC_IP, 443)),
        ],
    )
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    capture = fetch_nbs_cn_pmi(
        URL,
        clock=_clock(start, start),
        sleeper=lambda _seconds: None,
    )

    assert capture.value == 50.3
    assert session.trust_env is False
    assert session.call["allow_redirects"] is False
    assert session.call["stream"] is True
    assert response.closed is True
    assert session.closed is True


def test_three_manual_redirects_are_allowed_and_bound_to_final_url() -> None:
    first = URL
    second = "https://stats.gov.cn/sj/zxfb/202606/t20260630_1964032.html"
    third = "https://www.stats.gov.cn/sj/zxfb/202606/t20260630_1964032.html?view=full"
    final = "https://stats.gov.cn/sj/zxfb/202606/t20260630_1964032.html?view=full"
    transport = _FakeTransport(
        [
            _FakeResponse(302, {"Location": second}),
            _FakeResponse(301, {"Location": third}),
            _FakeResponse(307, {"Location": final}),
            _ok_response(),
        ]
    )
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    capture = fetch_nbs_cn_pmi(
        first,
        transport=transport,
        clock=_clock(start, start),
        sleeper=lambda _seconds: None,
    )

    assert capture.source_url == final
    assert capture.redirect_chain == (first, second, third, final)
    assert len(transport.calls) == 4


def test_fourth_redirect_is_permanent_and_not_retried() -> None:
    redirects = [
        _FakeResponse(
            302,
            {"Location": f"https://www.stats.gov.cn/sj/zxfb/202606/t20260630_{index}.html"},
        )
        for index in range(1, 5)
    ]
    transport = _FakeTransport(redirects)
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(NbsPmiPermanentError, match="redirect_limit_exceeded"):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(start),
            sleeper=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("permanent failure must not sleep")
            ),
        )

    assert len(transport.calls) == 4


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        (URL.replace("https://", "http://"), "https_required"),
        (URL.replace("www.stats.gov.cn", "stats.gov.cn.evil.test"), "host_rejected"),
        (URL.replace("www.stats.gov.cn", "www.stats.gov.cn:444"), "port_rejected"),
        (URL.replace("https://", "https://user:pass@"), "userinfo_rejected"),
        (f"{URL}?access_token=secret", "sensitive_query_rejected"),
        (f"{URL}#body", "fragment_rejected"),
        (URL.replace("t20260630_1964032.html", "latest.html"), "record_id_invalid"),
    ],
)
def test_unsafe_source_url_is_rejected_before_transport(url: str, reason: str) -> None:
    transport = _FakeTransport([_ok_response()])

    with pytest.raises(NbsPmiPermanentError, match=reason):
        fetch_nbs_cn_pmi(url, transport=transport)

    assert transport.calls == []
    assert transport.resolve_calls == []


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.1.1",
        "192.0.2.1",
        "224.0.0.1",
        "::1",
        "ff02::1",
    ],
)
def test_non_public_and_multicast_dns_are_rejected(
    address: str,
) -> None:
    transport = _FakeTransport([_ok_response()], resolved_ips=(address,))
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(NbsPmiPermanentError, match="url_ip_rejected"):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(start),
            sleeper=lambda _seconds: None,
        )

    assert transport.calls == []


def test_managed_rfc2544_tls_intercept_address_is_allowed() -> None:
    transport = _FakeTransport(
        [_ok_response()],
        resolved_ips=("198.18.5.43",),
    )
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    capture = fetch_nbs_cn_pmi(
        URL,
        transport=transport,
        clock=_clock(start, start),
        sleeper=lambda _seconds: None,
    )

    assert capture.value == 50.3
    assert transport.calls[0]["url"] == URL


def test_redirect_to_non_official_host_is_permanent() -> None:
    transport = _FakeTransport(
        [_FakeResponse(302, {"Location": "https://example.com/t20260630_1.html"})]
    )
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(NbsPmiPermanentError, match="host_rejected"):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(start),
            sleeper=lambda _seconds: None,
        )

    assert len(transport.calls) == 1


def test_network_429_and_5xx_are_retried_with_bounded_backoff() -> None:
    transport = _FakeTransport(
        [
            OSError("temporary DNS path failure"),
            _FakeResponse(429, {}),
            _ok_response(),
        ]
    )
    sleeps: list[float] = []
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    capture = fetch_nbs_cn_pmi(
        URL,
        transport=transport,
        clock=_clock(start, start),
        sleeper=sleeps.append,
    )

    assert capture.value == 50.3
    assert len(transport.calls) == 3
    assert sleeps == [0.25, 0.5]


def test_retry_after_delta_seconds_is_honored_but_capped() -> None:
    transport = _FakeTransport(
        [
            _FakeResponse(429, {"Retry-After": "9"}),
            _ok_response(),
        ]
    )
    sleeps: list[float] = []
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    capture = fetch_nbs_cn_pmi(
        URL,
        transport=transport,
        clock=_clock(start, start),
        sleeper=sleeps.append,
    )

    assert capture.value == 50.3
    assert sleeps == [1.0]


def test_transient_error_remains_distinct_after_retry_budget() -> None:
    transport = _FakeTransport(
        [_FakeResponse(503, {}), _FakeResponse(500, {})]
    )
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(NbsPmiTransientError, match="http_transient:500"):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(start),
            sleeper=lambda _seconds: None,
            max_attempts=2,
        )


def test_tls_certificate_failure_is_permanent_and_never_retried() -> None:
    transport = _FakeTransport(
        [requests.exceptions.SSLError("certificate verify failed")]
    )
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(
        NbsPmiPermanentError,
        match="tls_verification_failed",
    ):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(start),
            sleeper=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("TLS failure must not sleep or retry")
            ),
        )

    assert len(transport.calls) == 1


@pytest.mark.parametrize("status", [304, 400, 401, 403, 404])
def test_other_http_statuses_are_permanent_and_not_retried(status: int) -> None:
    transport = _FakeTransport([_FakeResponse(status, {})])
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(NbsPmiPermanentError, match=f"http_permanent:{status}"):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(start),
            sleeper=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("permanent failure must not sleep")
            ),
        )

    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_ok_response(content_type="application/json"), "content_type_invalid"),
        (_ok_response(content_type="text/html; charset=gb2312"), "charset_invalid"),
        (_ok_response(b"\xff\xfe"), "body_not_utf8"),
        (
            _ok_response(headers={"Content-Length": str(NBS_PMI_MAX_BODY_BYTES + 1)}),
            "body_size_invalid",
        ),
    ],
)
def test_response_entity_contract_fails_closed(
    response: _FakeResponse,
    reason: str,
) -> None:
    transport = _FakeTransport([response])
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(NbsPmiPermanentError, match=reason):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(start, start),
            sleeper=lambda _seconds: None,
        )


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            _replace(
                "2026年6月中国采购经理指数运行情况",
                "2026年6月中国采购经理指数专家解读",
            ),
            "article_title_invalid",
        ),
        (
            _replace("2026/06/30 09:30", "2026/06/29 09:30"),
            "pubdate_record_mismatch",
        ),
        (
            _replace("6月份，制造业采购经理指数（PMI）", "5月份，制造业采购经理指数（PMI）"),
            "paragraph_period_mismatch",
        ),
        (
            _replace(
                "</main>",
                "<p>6月份，制造业采购经理指数（PMI）为49.9%。</p></main>",
            ),
            "value_not_unique",
        ),
        (
            _replace("制造业采购经理指数（PMI）为", "制造业采购经理指数为"),
            "formal_paragraph_missing",
        ),
    ],
)
def test_parser_rejects_nonformal_or_conflicting_evidence(
    body: bytes,
    reason: str,
) -> None:
    with pytest.raises(NbsPmiPermanentError, match=reason):
        parse_nbs_cn_pmi_html(body, source_url=URL)


def test_record_date_must_match_pubdate() -> None:
    wrong_url = URL.replace("t20260630_1964032", "t20260701_1964032")

    with pytest.raises(NbsPmiPermanentError, match="pubdate_record_mismatch"):
        parse_nbs_cn_pmi_html(_body(), source_url=wrong_url)


def test_article_period_may_precede_cross_month_publication() -> None:
    body = _replace("2026/06/30 09:30", "2026/07/01 09:30")
    source_url = URL.replace("t20260630_1964032", "t20260701_1964032")

    parsed = parse_nbs_cn_pmi_html(body, source_url=source_url)

    assert parsed.month == "202606"
    assert parsed.source_release_at == "2026-07-01T09:30:00+08:00"


def test_parser_rejects_period_after_publication_month() -> None:
    body = (
        _body()
        .decode("utf-8")
        .replace(
            "2026年6月中国采购经理指数运行情况",
            "2026年7月中国采购经理指数运行情况",
        )
        .replace(
            "6月份，制造业采购经理指数（PMI）",
            "7月份，制造业采购经理指数（PMI）",
        )
        .encode("utf-8")
    )

    with pytest.raises(NbsPmiPermanentError, match="period_after_pubdate"):
        parse_nbs_cn_pmi_html(body, source_url=URL)


def test_official_fetch_rejects_period_after_publication_without_retry() -> None:
    body = (
        _body()
        .decode("utf-8")
        .replace(
            "2026年6月中国采购经理指数运行情况",
            "2026年7月中国采购经理指数运行情况",
        )
        .replace(
            "6月份，制造业采购经理指数（PMI）",
            "7月份，制造业采购经理指数（PMI）",
        )
        .encode("utf-8")
    )
    transport = _FakeTransport([_ok_response(body)])
    started = datetime(2026, 7, 1, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(NbsPmiPermanentError, match="period_after_pubdate"):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(started, started),
            sleeper=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("semantic failure must not retry")
            ),
        )

    assert len(transport.calls) == 1


@pytest.mark.parametrize("max_attempts", [0, 4, True])
def test_retry_budget_is_hard_bounded(max_attempts: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        fetch_nbs_cn_pmi(URL, max_attempts=max_attempts)


def test_reversed_fetch_clock_is_permanent() -> None:
    transport = _FakeTransport([_ok_response()])
    start = datetime(2026, 6, 30, 1, 30, tzinfo=timezone.utc)

    with pytest.raises(NbsPmiPermanentError, match="fetch_clock_reversed"):
        fetch_nbs_cn_pmi(
            URL,
            transport=transport,
            clock=_clock(start, start - timedelta(seconds=1)),
            sleeper=lambda _seconds: None,
        )
