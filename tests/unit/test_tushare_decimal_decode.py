from __future__ import annotations

from decimal import Decimal
import json
from typing import Any

import pytest

from quant_investor.v17_v4_runtime import tushare_https
from quant_investor.v17_v4_runtime.tushare_https import (
    OfficialTushareHttpsClient,
    TushareHttpsError,
)

TOKEN = "A" * 40
FIELDS = ("small", "large", "scientific", "negative_zero")


class Response:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self, amount: int) -> bytes:
        return self.body[:amount]


class Connection:
    body = b""
    request_body = b""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def request(self, *args: Any, **kwargs: Any) -> None:
        type(self).request_body = kwargs["body"]

    def getresponse(self) -> Response:
        return Response(self.body)

    def close(self) -> None:
        pass


def envelope(items: str) -> bytes:
    return (
        '{"code":0,"data":{"count":1,"fields":'
        + json.dumps(list(FIELDS), separators=(",", ":"))
        + ',"has_more":false,"items":'
        + items
        + '},"detail":"","msg":"","request_id":"request-1"}'
    ).encode()


@pytest.fixture(autouse=True)
def install(monkeypatch: pytest.MonkeyPatch) -> None:
    Connection.request_body = b""
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    monkeypatch.setattr(tushare_https, "_HTTPS_CONNECTION", Connection)
    monkeypatch.setattr(tushare_https, "_CREATE_DEFAULT_CONTEXT", object)


def test_default_decode_remains_binary_float_compatible() -> None:
    Connection.body = envelope("[[0.1,12345678901234567890,1.25e2,-0.0]]")
    result = OfficialTushareHttpsClient().request(
        api_name="daily_basic",
        params={},
        expected_fields=FIELDS,
    )
    row = result.rows[0]
    assert type(row[0]) is float
    assert type(row[1]) is int
    assert type(row[2]) is float
    assert type(row[3]) is float
    assert Connection.request_body == (
        b'{"api_name":"daily_basic","fields":"small,large,scientific,negative_zero",'
        b'"params":{},"token":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}'
    )


def test_strict_decode_preserves_decimal_and_integer_exactness() -> None:
    Connection.body = envelope("[[0.1000000000001,12345678901234567890,1.234567890123e2,-0.0]]")
    result = OfficialTushareHttpsClient(strict_decimal_decode=True).request(
        api_name="daily_basic",
        params={},
        expected_fields=FIELDS,
    )
    row = result.rows[0]
    assert row == (
        Decimal("0.1000000000001"),
        12345678901234567890,
        Decimal("123.4567890123"),
        Decimal("-0.0"),
    )
    assert type(row[1]) is int


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_decode_rejects_nonfinite_constants(constant: str) -> None:
    Connection.body = envelope(f"[[{constant},1,2,3]]")
    with pytest.raises(TushareHttpsError, match="TUSHARE_RESPONSE_INVALID"):
        OfficialTushareHttpsClient(strict_decimal_decode=True).request(
            api_name="daily_basic",
            params={},
            expected_fields=FIELDS,
        )


def test_strict_decode_configuration_requires_exact_bool() -> None:
    with pytest.raises(TushareHttpsError, match="TUSHARE_CLIENT_CONFIG_INVALID"):
        OfficialTushareHttpsClient(strict_decimal_decode=1)  # type: ignore[arg-type]
