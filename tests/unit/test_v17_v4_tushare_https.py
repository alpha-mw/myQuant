from __future__ import annotations

from collections.abc import Iterator, Mapping
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v4_runtime import tushare_https
from quant_investor.v17_v4_runtime.tushare_https import (
    MAX_RESPONSE_BYTES,
    OFFICIAL_TUSHARE_HOST,
    OFFICIAL_TUSHARE_PATH,
    OFFICIAL_TUSHARE_PORT,
    OFFICIAL_TUSHARE_URL,
    OfficialTushareHttpsClient,
    TushareHttpsError,
    validate_official_endpoint,
)

TOKEN = "A" * 40
FIELDS = ("ts_code", "trade_date", "close")


class _Response:
    def __init__(self, status: int, body: bytes | BaseException) -> None:
        self.status = status
        self._body = body

    def read(self, amount: int) -> bytes:
        if isinstance(self._body, BaseException):
            raise self._body
        return self._body[:amount]


class _Connection:
    response = _Response(200, b"")
    constructor_calls: list[dict[str, Any]] = []
    request_calls: list[dict[str, Any]] = []
    close_count = 0

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float,
        context: object,
    ) -> None:
        self.constructor_calls.append(
            {
                "host": host,
                "port": port,
                "timeout": timeout,
                "context": context,
            }
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        self.request_calls.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "headers": headers,
            }
        )

    def getresponse(self) -> _Response:
        return self.response

    def close(self) -> None:
        type(self).close_count += 1


@pytest.fixture(autouse=True)
def _reset_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    _Connection.constructor_calls = []
    _Connection.request_calls = []
    _Connection.close_count = 0
    monkeypatch.setattr(tushare_https, "_HTTPS_CONNECTION", _Connection)


def _body(
    *,
    fields: tuple[str, ...] = FIELDS,
    items: list[list[Any]] | None = None,
    code: int = 0,
    message: Any = "",
) -> bytes:
    return json.dumps(
        {
            "code": code,
            "data": {
                "count": 0,
                "fields": list(fields),
                "has_more": False,
                "items": (
                    [["000001.SZ", "20260727", 10.5]]
                    if items is None
                    else items
                ),
            },
            "detail": "",
            "msg": message,
            "request_id": "request-1",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _install_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: bytes | None = None,
) -> object:
    context = object()
    context_calls: list[tuple[object, ...]] = []

    def create_context(*args: object, **kwargs: object) -> object:
        context_calls.append((*args, *kwargs.items()))
        return context

    monkeypatch.setattr(
        tushare_https,
        "_CREATE_DEFAULT_CONTEXT",
        create_context,
    )
    _Connection.response = _Response(200, _body() if body is None else body)
    return context


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.tushare.pro/",
        "https://api.tushare.pro",
        "https://API.TUSHARE.PRO/",
        "https://api.tushare.pro:443/",
        "https://api.tushare.pro:444/",
        "https://api.tushare.pro/other",
        "https://api.tushare.pro/?query=1",
        "https://api.tushare.pro/#fragment",
        "https://user@api.tushare.pro/",
        "https://127.0.0.1/",
        "https://api.tushare.pro.evil/",
    ],
)
def test_endpoint_allowlist_is_exact_and_pre_socket(endpoint: str) -> None:
    with pytest.raises(TushareHttpsError) as captured:
        validate_official_endpoint(endpoint)
    assert str(captured.value) == "TUSHARE_ENDPOINT_BLOCKED"
    assert not _Connection.constructor_calls
    assert not _Connection.request_calls


def test_official_https_request_uses_default_tls_and_exact_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    context = _install_success(monkeypatch)

    result = OfficialTushareHttpsClient(timeout_seconds=8).request(
        api_name="daily",
        params={"start_date": "20260701", "end_date": "20260727"},
        expected_fields=FIELDS,
    )

    assert OFFICIAL_TUSHARE_URL == "https://api.tushare.pro/"
    assert result.fields == FIELDS
    assert result.rows == (("000001.SZ", "20260727", 10.5),)
    assert result.reported_count == 0
    assert result.has_more is False
    assert _Connection.constructor_calls == [
        {
            "host": OFFICIAL_TUSHARE_HOST,
            "port": OFFICIAL_TUSHARE_PORT,
            "timeout": 8.0,
            "context": context,
        }
    ]
    request = _Connection.request_calls[0]
    assert request["method"] == "POST"
    assert request["path"] == OFFICIAL_TUSHARE_PATH
    assert request["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "myquant-v17-v4/1",
    }
    decoded = json.loads(request["body"])
    assert decoded == {
        "api_name": "daily",
        "fields": "ts_code,trade_date,close",
        "params": {
            "end_date": "20260727",
            "start_date": "20260701",
        },
        "token": TOKEN,
    }
    assert _Connection.close_count == 1


def test_only_tushare_token_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setenv("TUSHARE_FALLBACK_TOKEN", "B" * 40)
    with pytest.raises(TushareHttpsError) as captured:
        OfficialTushareHttpsClient().request(
            api_name="daily",
            params={},
            expected_fields=FIELDS,
        )
    assert str(captured.value) == "TUSHARE_TOKEN_MISSING"
    assert not _Connection.constructor_calls


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (301, "TUSHARE_REDIRECT_BLOCKED"),
        (302, "TUSHARE_REDIRECT_BLOCKED"),
        (307, "TUSHARE_REDIRECT_BLOCKED"),
        (308, "TUSHARE_REDIRECT_BLOCKED"),
        (403, "TUSHARE_HTTP_STATUS_ERROR"),
        (500, "TUSHARE_HTTP_STATUS_ERROR"),
    ],
)
def test_redirects_and_non_success_statuses_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected_code: str,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    _install_success(monkeypatch)
    _Connection.response = _Response(status, b"must-not-be-read")
    with pytest.raises(TushareHttpsError) as captured:
        OfficialTushareHttpsClient().request(
            api_name="daily",
            params={},
            expected_fields=FIELDS,
        )
    assert str(captured.value) == expected_code
    assert _Connection.close_count == 1


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        _body(fields=("trade_date", "ts_code", "close")),
        _body(items=[["000001.SZ", "20260727"]]),
        _body(items=[["000001.SZ", "20260727", {"nested": True}]]),
        _body(code=-1, message="provider detail"),
        json.dumps(
            {
                "code": 0,
                "data": {
                    "count": 0,
                    "fields": list(FIELDS),
                    "has_more": False,
                    "items": [],
                },
                "detail": "",
                "extra": True,
                "msg": None,
                "request_id": "request-1",
            },
            separators=(",", ":"),
        ).encode(),
    ],
)
def test_response_envelope_and_expected_fields_are_closed(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    _install_success(monkeypatch, body=body)
    with pytest.raises(TushareHttpsError) as captured:
        OfficialTushareHttpsClient().request(
            api_name="daily",
            params={},
            expected_fields=FIELDS,
        )
    assert captured.value.code in {
        "TUSHARE_API_ERROR",
        "TUSHARE_RESPONSE_INVALID",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"detail": None}),
        lambda payload: payload["data"].update({"count": True}),
        lambda payload: payload["data"].update({"count": -1}),
        lambda payload: payload["data"].update({"has_more": 0}),
        lambda payload: payload["data"].update({"unknown": False}),
    ],
)
def test_official_response_metadata_is_strict(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    payload = json.loads(_body())
    mutation(payload)
    _install_success(
        monkeypatch,
        body=json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(),
    )
    with pytest.raises(TushareHttpsError) as captured:
        OfficialTushareHttpsClient().request(
            api_name="daily",
            params={},
            expected_fields=FIELDS,
        )
    assert captured.value.code == "TUSHARE_RESPONSE_INVALID"


def test_oversized_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    _install_success(monkeypatch, body=b"x" * (MAX_RESPONSE_BYTES + 1))
    with pytest.raises(TushareHttpsError) as captured:
        OfficialTushareHttpsClient().request(
            api_name="daily",
            params={},
            expected_fields=FIELDS,
        )
    assert captured.value.code == "TUSHARE_RESPONSE_TOO_LARGE"


def test_secret_samples_never_reach_exception_text_or_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = {
        "environment": "ENVIRONMENTSECRET0123456789",
        "request": "REQUESTSECRET01234567890123",
        "response": "RESPONSESECRET0123456789012",
        "transport": "TRANSPORTSECRET012345678901",
        "nested": "NESTEDSECRET012345678901234",
    }
    monkeypatch.setenv("TUSHARE_TOKEN", samples["environment"])
    _install_success(
        monkeypatch,
        body=json.dumps(
            {"echo": samples["response"]},
            separators=(",", ":"),
        ).encode(),
    )
    failures: list[TushareHttpsError] = []

    for params in (
        {"token": samples["request"]},
        {"safe": {"authorization": samples["nested"]}},
    ):
        with pytest.raises(TushareHttpsError) as captured:
            OfficialTushareHttpsClient().request(
                api_name="daily",
                params=params,
                expected_fields=FIELDS,
            )
        failures.append(captured.value)

    with pytest.raises(TushareHttpsError) as captured:
        OfficialTushareHttpsClient().request(
            api_name="daily",
            params={},
            expected_fields=FIELDS,
        )
    failures.append(captured.value)

    _Connection.response = _Response(
        200,
        RuntimeError(samples["transport"]),
    )
    with pytest.raises(TushareHttpsError) as captured:
        OfficialTushareHttpsClient().request(
            api_name="daily",
            params={},
            expected_fields=FIELDS,
        )
    failures.append(captured.value)

    rendered = "\n".join(
        f"{failure!s}\n{failure!r}" for failure in failures
    )
    for sample in samples.values():
        assert sample not in rendered
    assert all(failure.__cause__ is None for failure in failures)


def test_request_validation_exception_is_static_and_pre_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "MAPPINGEXCEPTIONSECRET0123456"

    class ExplodingMapping(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError(secret)

        def __len__(self) -> int:
            return 1

    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    with pytest.raises(TushareHttpsError) as captured:
        OfficialTushareHttpsClient().request(
            api_name="daily",
            params=ExplodingMapping(),
            expected_fields=FIELDS,
        )
    assert str(captured.value) == "TUSHARE_REQUEST_INVALID"
    assert secret not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert not _Connection.constructor_calls


def test_client_exposes_no_endpoint_tls_proxy_or_redirect_override() -> None:
    constructor = inspect.signature(OfficialTushareHttpsClient)
    request = inspect.signature(OfficialTushareHttpsClient.request)
    forbidden = {
        "allow_redirects",
        "cert",
        "endpoint",
        "headers",
        "host",
        "proxy",
        "url",
        "verify",
    }
    assert forbidden.isdisjoint(constructor.parameters)
    assert forbidden.isdisjoint(request.parameters)


def test_transport_creates_no_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    _install_success(monkeypatch)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    OfficialTushareHttpsClient().request(
        api_name="daily",
        params={},
        expected_fields=FIELDS,
    )
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before == []
