from __future__ import annotations

from decimal import Decimal
import json
from typing import Any

import pytest

from quant_investor.market import tushare_transport
from quant_investor.market.tushare_transport import (
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


def envelope(items: str, *, count: int = 1) -> bytes:
    return (
        f'{{"code":0,"data":{{"count":{count},"fields":'
        + json.dumps(list(FIELDS), separators=(",", ":"))
        + ',"has_more":false,"items":'
        + items
        + '},"detail":"","msg":"","request_id":"request-1"}'
    ).encode()


@pytest.fixture(autouse=True)
def install(monkeypatch: pytest.MonkeyPatch) -> None:
    Connection.request_body = b""
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    monkeypatch.setattr(tushare_transport, "_HTTPS_CONNECTION", Connection)
    monkeypatch.setattr(tushare_transport, "_CREATE_DEFAULT_CONTEXT", object)


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


def test_strict_decode_normalizes_official_zero_count_sentinel() -> None:
    Connection.body = envelope(
        '[["first",1,2,3],["second",4,5,6]]',
        count=0,
    )
    strict = OfficialTushareHttpsClient(strict_decimal_decode=True).request(
        api_name="daily_basic",
        params={},
        expected_fields=FIELDS,
    )
    assert strict.reported_count == 2

    Connection.body = envelope(
        '[["first",1,2,3],["second",4,5,6]]',
        count=0,
    )
    legacy = OfficialTushareHttpsClient().request(
        api_name="daily_basic",
        params={},
        expected_fields=FIELDS,
    )
    assert legacy.reported_count == 0


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


def test_response_item_limit_is_explicit_and_default_compatible() -> None:
    items = json.dumps([[index, 1, 2, 3] for index in range(10_001)], separators=(",", ":"))
    Connection.body = envelope(items, count=10_001)
    with pytest.raises(TushareHttpsError, match="TUSHARE_RESPONSE_INVALID"):
        OfficialTushareHttpsClient(strict_decimal_decode=True).request(
            api_name="fina_indicator_vip",
            params={"period": "20191231"},
            expected_fields=FIELDS,
        )

    Connection.body = envelope(items, count=10_001)
    result = OfficialTushareHttpsClient(
        strict_decimal_decode=True,
        max_response_items=20_000,
    ).request(
        api_name="fina_indicator_vip",
        params={"period": "20191231"},
        expected_fields=FIELDS,
    )
    assert result.reported_count == len(result.rows) == 10_001


@pytest.mark.parametrize("value", [True, 0, 20_001])
def test_response_item_limit_rejects_invalid_configuration(value: object) -> None:
    with pytest.raises(TushareHttpsError, match="TUSHARE_CLIENT_CONFIG_INVALID"):
        OfficialTushareHttpsClient(max_response_items=value)  # type: ignore[arg-type]
