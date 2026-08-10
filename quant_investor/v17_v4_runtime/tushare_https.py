"""Official HTTPS-only Tushare transport for the V17 v4 research runtime."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import http.client
import json
import math
import os
import re
import ssl
import unicodedata
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn, Sequence

OFFICIAL_TUSHARE_URL: Final = "https://api.tushare.pro/"
OFFICIAL_TUSHARE_HOST: Final = "api.tushare.pro"
OFFICIAL_TUSHARE_PORT: Final = 443
OFFICIAL_TUSHARE_PATH: Final = "/"
DEFAULT_TIMEOUT_SECONDS: Final = 15.0
MAX_RESPONSE_BYTES: Final = 64 * 1024 * 1024
MAX_REQUEST_BYTES: Final = 1024 * 1024
MAX_CONTAINER_ITEMS: Final = 10_000
_MAX_CONFIGURED_RESPONSE_ITEMS: Final = 20_000
MAX_DEPTH: Final = 16

_API_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$", re.ASCII)
_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$", re.ASCII)
_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{20,128}$", re.ASCII)
_FORBIDDEN_PARAM_KEYS = frozenset(
    {"api_key", "authorization", "bearer", "secret", "token"}
)
_HTTPS_CONNECTION = http.client.HTTPSConnection
_CREATE_DEFAULT_CONTEXT = ssl.create_default_context


class TushareHttpsError(RuntimeError):
    """A static-code provider failure that never renders response or secrets."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True)
class TushareResponse:
    """Validated, immutable response rows without raw transport material."""

    api_name: str
    request_id: str
    reported_count: int
    has_more: bool
    fields: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class TushareSchemaDiagnostic:
    """Sanitized response-shape metadata with no business cell values."""

    api_name: str
    status: str
    provider_code: int
    request_id_sha256: str
    response_body_sha256: str
    provider_reported_count: int
    item_count: int
    has_more: bool
    observed_fields: tuple[str, ...]
    expected_fields_match: bool
    row_widths: tuple[int, ...]
    cell_types: tuple[str, ...]
    text_cell_count: int
    non_nfc_text_count: int
    max_text_utf8_bytes: int


def _fail(code: str) -> NoReturn:
    raise TushareHttpsError(code) from None


def validate_official_endpoint(value: str) -> None:
    """Reject every endpoint except the exact official HTTPS root."""

    if type(value) is not str or value != OFFICIAL_TUSHARE_URL:
        _fail("TUSHARE_ENDPOINT_BLOCKED")


def _validate_json_value(value: Any, *, depth: int = 1) -> None:
    if depth > MAX_DEPTH:
        _fail("TUSHARE_REQUEST_INVALID")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail("TUSHARE_REQUEST_INVALID")
        return
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_REQUEST_BYTES:
            _fail("TUSHARE_REQUEST_INVALID")
        return
    if type(value) is list:
        if len(value) > MAX_CONTAINER_ITEMS:
            _fail("TUSHARE_REQUEST_INVALID")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if type(value) is dict:
        if len(value) > MAX_CONTAINER_ITEMS:
            _fail("TUSHARE_REQUEST_INVALID")
        for key, item in value.items():
            if (
                type(key) is not str
                or not key
                or key.casefold() in _FORBIDDEN_PARAM_KEYS
            ):
                _fail("TUSHARE_REQUEST_INVALID")
            _validate_json_value(item, depth=depth + 1)
        return
    _fail("TUSHARE_REQUEST_INVALID")


def _request_bytes(
    *,
    api_name: str,
    token: str,
    params: Mapping[str, Any],
    fields: tuple[str, ...],
) -> bytes:
    payload = {
        "api_name": api_name,
        "fields": ",".join(fields),
        "params": dict(params),
        "token": token,
    }
    try:
        raw = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        _fail("TUSHARE_REQUEST_INVALID")
    if len(raw) > MAX_REQUEST_BYTES:
        _fail("TUSHARE_REQUEST_INVALID")
    return raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("TUSHARE_RESPONSE_INVALID")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    _fail("TUSHARE_RESPONSE_INVALID")


def _validate_response_cell(
    value: Any,
    *,
    strict_decimal_decode: bool,
) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if strict_decimal_decode and type(value) is Decimal and value.is_finite():
        return
    if type(value) is float and math.isfinite(value):
        if strict_decimal_decode:
            _fail("TUSHARE_RESPONSE_INVALID")
        return
    _fail("TUSHARE_RESPONSE_INVALID")


def _diagnostic_cell_type(value: Any) -> str:
    if value is None:
        return "NULL"
    if type(value) is bool:
        return "BOOLEAN"
    if type(value) is int:
        return "INTEGER"
    if type(value) is Decimal and value.is_finite():
        return "DECIMAL"
    if type(value) is str:
        return "TEXT"
    _fail("TUSHARE_RESPONSE_INVALID")


def _decode_schema_diagnostic(
    raw: bytes,
    *,
    api_name: str,
    expected_fields: tuple[str, ...],
    max_response_items: int = MAX_CONTAINER_ITEMS,
) -> TushareSchemaDiagnostic:
    """Project response shape while irreversibly discarding business values."""

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=Decimal,
        )
    except TushareHttpsError:
        raise
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        _fail("TUSHARE_RESPONSE_INVALID")
    if type(payload) is not dict or set(payload) != {
        "code",
        "data",
        "detail",
        "msg",
        "request_id",
    }:
        _fail("TUSHARE_RESPONSE_INVALID")
    if (
        type(payload["code"]) is not int
        or type(payload["detail"]) is not str
        or type(payload["msg"]) is not str
        or type(payload["request_id"]) is not str
        or not payload["request_id"]
    ):
        _fail("TUSHARE_RESPONSE_INVALID")
    if payload["code"] != 0:
        _fail("TUSHARE_API_ERROR")
    data = payload["data"]
    if type(data) is not dict or set(data) != {
        "count",
        "fields",
        "has_more",
        "items",
    }:
        _fail("TUSHARE_RESPONSE_INVALID")
    reported_count = data["count"]
    has_more = data["has_more"]
    response_fields = data["fields"]
    items = data["items"]
    if (
        type(reported_count) is not int
        or reported_count < 0
        or type(has_more) is not bool
        or type(response_fields) is not list
        or any(
            type(field) is not str or _FIELD_RE.fullmatch(field) is None
            for field in response_fields
        )
        or len(response_fields) != len(set(response_fields))
        or type(items) is not list
        or len(items) > max_response_items
    ):
        _fail("TUSHARE_RESPONSE_INVALID")
    row_widths: set[int] = set()
    cell_types: set[str] = set()
    text_cell_count = 0
    non_nfc_text_count = 0
    max_text_utf8_bytes = 0
    for row in items:
        if type(row) is not list or len(row) > MAX_CONTAINER_ITEMS:
            _fail("TUSHARE_RESPONSE_INVALID")
        row_widths.add(len(row))
        for value in row:
            cell_types.add(_diagnostic_cell_type(value))
            if type(value) is str:
                text_cell_count += 1
                encoded_length = len(value.encode("utf-8", errors="strict"))
                max_text_utf8_bytes = max(max_text_utf8_bytes, encoded_length)
                if unicodedata.normalize("NFC", value) != value:
                    non_nfc_text_count += 1
    request_id_sha256 = hashlib.sha256(payload["request_id"].encode("utf-8")).hexdigest()
    return TushareSchemaDiagnostic(
        api_name=api_name,
        status="OBSERVED",
        provider_code=payload["code"],
        request_id_sha256=request_id_sha256,
        response_body_sha256=hashlib.sha256(raw).hexdigest(),
        provider_reported_count=reported_count,
        item_count=len(items),
        has_more=has_more,
        observed_fields=tuple(response_fields),
        expected_fields_match=tuple(response_fields) == expected_fields,
        row_widths=tuple(sorted(row_widths)),
        cell_types=tuple(sorted(cell_types)),
        text_cell_count=text_cell_count,
        non_nfc_text_count=non_nfc_text_count,
        max_text_utf8_bytes=max_text_utf8_bytes,
    )


def _decode_response(
    raw: bytes,
    *,
    api_name: str,
    expected_fields: tuple[str, ...],
    strict_decimal_decode: bool = False,
    max_response_items: int = MAX_CONTAINER_ITEMS,
) -> TushareResponse:
    try:
        decode_options: dict[str, Any] = {
            "object_pairs_hook": _unique_object,
            "parse_constant": _reject_constant,
        }
        if strict_decimal_decode:
            decode_options["parse_float"] = Decimal
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            **decode_options,
        )
    except TushareHttpsError:
        raise
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        _fail("TUSHARE_RESPONSE_INVALID")
    if type(payload) is not dict or set(payload) != {
        "code",
        "data",
        "detail",
        "msg",
        "request_id",
    }:
        _fail("TUSHARE_RESPONSE_INVALID")
    if type(payload["code"]) is not int:
        _fail("TUSHARE_RESPONSE_INVALID")
    if payload["code"] != 0:
        _fail("TUSHARE_API_ERROR")
    if (
        type(payload["detail"]) is not str
        or type(payload["msg"]) is not str
    ):
        _fail("TUSHARE_RESPONSE_INVALID")
    request_id = payload["request_id"]
    if type(request_id) is not str or not request_id:
        _fail("TUSHARE_RESPONSE_INVALID")
    data = payload["data"]
    if type(data) is not dict or set(data) != {
        "count",
        "fields",
        "has_more",
        "items",
    }:
        _fail("TUSHARE_RESPONSE_INVALID")
    reported_count = data["count"]
    has_more = data["has_more"]
    response_fields = data["fields"]
    items = data["items"]
    if (
        type(reported_count) is not int
        or reported_count < 0
        or type(has_more) is not bool
        or type(response_fields) is not list
        or any(type(field) is not str for field in response_fields)
        or tuple(response_fields) != expected_fields
        or len(response_fields) != len(set(response_fields))
        or type(items) is not list
        or len(items) > max_response_items
    ):
        _fail("TUSHARE_RESPONSE_INVALID")
    rows: list[tuple[Any, ...]] = []
    for row in items:
        if type(row) is not list or len(row) != len(expected_fields):
            _fail("TUSHARE_RESPONSE_INVALID")
        for value in row:
            _validate_response_cell(
                value,
                strict_decimal_decode=strict_decimal_decode,
            )
        rows.append(tuple(row))
    normalized_reported_count = reported_count
    if strict_decimal_decode and reported_count == 0 and rows:
        # Official batch endpoints use zero as a count placeholder even when
        # ``items`` is non-empty. Strict v2 consumers bind the accepted row
        # count to item cardinality; legacy decoding retains the provider value.
        normalized_reported_count = len(rows)
    return TushareResponse(
        api_name=api_name,
        request_id=request_id,
        reported_count=normalized_reported_count,
        has_more=has_more,
        fields=expected_fields,
        rows=tuple(rows),
    )


class OfficialTushareHttpsClient:
    """No-redirect client whose credential source is only ``TUSHARE_TOKEN``."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        strict_decimal_decode: bool = False,
        max_response_items: int = MAX_CONTAINER_ITEMS,
    ) -> None:
        if (
            type(timeout_seconds) not in {int, float}
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
            or float(timeout_seconds) > 120
        ):
            _fail("TUSHARE_CLIENT_CONFIG_INVALID")
        if type(strict_decimal_decode) is not bool:
            _fail("TUSHARE_CLIENT_CONFIG_INVALID")
        if (
            type(max_response_items) is not int
            or not 1 <= max_response_items <= _MAX_CONFIGURED_RESPONSE_ITEMS
        ):
            _fail("TUSHARE_CLIENT_CONFIG_INVALID")
        self._timeout_seconds = float(timeout_seconds)
        self._strict_decimal_decode = strict_decimal_decode
        self._max_response_items = max_response_items

    def _prepare_request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> tuple[tuple[str, ...], bytes]:
        try:
            validate_official_endpoint(OFFICIAL_TUSHARE_URL)
            if (
                type(api_name) is not str
                or _API_NAME_RE.fullmatch(api_name) is None
                or not isinstance(params, Mapping)
                or isinstance(expected_fields, (str, bytes))
            ):
                _fail("TUSHARE_REQUEST_INVALID")
            params_copy = dict(params)
            _validate_json_value(params_copy)
            fields = tuple(expected_fields)
            if (
                not fields
                or len(fields) != len(set(fields))
                or any(
                    type(field) is not str
                    or _FIELD_RE.fullmatch(field) is None
                    for field in fields
                )
            ):
                _fail("TUSHARE_REQUEST_INVALID")
            token = os.environ.get("TUSHARE_TOKEN")
            if type(token) is not str or _TOKEN_RE.fullmatch(token) is None:
                _fail("TUSHARE_TOKEN_MISSING")
            body = _request_bytes(
                api_name=api_name,
                token=token,
                params=MappingProxyType(params_copy),
                fields=fields,
            )
            return fields, body
        except TushareHttpsError:
            raise
        except BaseException:
            _fail("TUSHARE_REQUEST_INVALID")

    def _fetch_raw(self, body: bytes) -> bytes:
        connection: http.client.HTTPSConnection | None = None
        try:
            context = _CREATE_DEFAULT_CONTEXT()
            connection = _HTTPS_CONNECTION(
                OFFICIAL_TUSHARE_HOST,
                OFFICIAL_TUSHARE_PORT,
                timeout=self._timeout_seconds,
                context=context,
            )
            connection.request(
                "POST",
                OFFICIAL_TUSHARE_PATH,
                body=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json; charset=utf-8",
                    "User-Agent": "myquant-v17-v4/1",
                },
            )
            response = connection.getresponse()
            status = response.status
            if 300 <= status < 400:
                _fail("TUSHARE_REDIRECT_BLOCKED")
            if status != 200:
                _fail("TUSHARE_HTTP_STATUS_ERROR")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                _fail("TUSHARE_RESPONSE_TOO_LARGE")
            return raw
        except TushareHttpsError:
            raise
        except BaseException:
            _fail("TUSHARE_TRANSPORT_ERROR")
        finally:
            if connection is not None:
                try:
                    connection.close()
                except BaseException:
                    pass

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareResponse:
        fields, body = self._prepare_request(
            api_name=api_name,
            params=params,
            expected_fields=expected_fields,
        )
        raw = self._fetch_raw(body)
        return _decode_response(
            raw,
            api_name=api_name,
            expected_fields=fields,
            strict_decimal_decode=self._strict_decimal_decode,
            max_response_items=self._max_response_items,
        )

    def diagnose_schema(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareSchemaDiagnostic:
        """Make one strict request and discard all business values after projection."""

        if self._strict_decimal_decode is not True:
            _fail("TUSHARE_CLIENT_CONFIG_INVALID")
        fields, body = self._prepare_request(
            api_name=api_name,
            params=params,
            expected_fields=expected_fields,
        )
        raw = self._fetch_raw(body)
        return _decode_schema_diagnostic(
            raw,
            api_name=api_name,
            expected_fields=fields,
            max_response_items=self._max_response_items,
        )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_RESPONSE_BYTES",
    "OFFICIAL_TUSHARE_HOST",
    "OFFICIAL_TUSHARE_PATH",
    "OFFICIAL_TUSHARE_PORT",
    "OFFICIAL_TUSHARE_URL",
    "OfficialTushareHttpsClient",
    "TushareHttpsError",
    "TushareResponse",
    "TushareSchemaDiagnostic",
    "validate_official_endpoint",
]
