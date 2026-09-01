"""Official HTTPS-only Tushare transport for the stable market data plane."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import errno
import hashlib
import http.client
import json
import math
import os
import re
import socket
import ssl
import time
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
_FORBIDDEN_PARAM_KEYS = frozenset({"api_key", "authorization", "bearer", "secret", "token"})
_RESPONSE_ENVELOPE_FIELDS = frozenset({"code", "data", "detail", "msg", "request_id"})
_RESPONSE_DATA_FIELDS = frozenset({"count", "fields", "has_more", "items"})
_HTTPS_CONNECTION = http.client.HTTPSConnection
_CREATE_DEFAULT_CONTEXT = ssl.create_default_context
_MONOTONIC = time.monotonic

_TRANSPORT_FAILURE_CLASSES: Final = frozenset(
    {
        "CONNECT",
        "CONNECTION_RESET",
        "DNS",
        "READ",
        "TIMEOUT",
        "TLS",
        "UNKNOWN",
    }
)
_TRANSPORT_FAILURE_PHASES: Final = frozenset(
    {
        "RESPONSE_BODY",
        "RESPONSE_HEADERS",
        "REQUEST_SEND",
        "TLS_CONTEXT",
        "UNKNOWN",
    }
)
_CONNECT_ERRNOS: Final = frozenset(
    {
        errno.ECONNREFUSED,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
    }
)


@dataclass(frozen=True, slots=True)
class TushareTransportDiagnostic:
    """Allowlisted transport metadata with no source exception material."""

    failure_class: str
    failure_phase: str
    elapsed_ms: int

    def __post_init__(self) -> None:
        if (
            self.failure_class not in _TRANSPORT_FAILURE_CLASSES
            or self.failure_phase not in _TRANSPORT_FAILURE_PHASES
            or type(self.elapsed_ms) is not int
            or self.elapsed_ms < 0
        ):
            raise ValueError("TUSHARE_TRANSPORT_DIAGNOSTIC_INVALID")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "failure_class": self.failure_class,
            "failure_phase": self.failure_phase,
            "elapsed_ms": self.elapsed_ms,
        }


class TushareHttpsError(RuntimeError):
    """A static-code provider failure that never renders response or secrets."""

    def __init__(
        self,
        code: str,
        *,
        transport_diagnostic: TushareTransportDiagnostic | None = None,
    ) -> None:
        if transport_diagnostic is not None and (
            code != "TUSHARE_TRANSPORT_ERROR"
            or not isinstance(transport_diagnostic, TushareTransportDiagnostic)
        ):
            raise ValueError("TUSHARE_ERROR_DIAGNOSTIC_INVALID")
        self.code = code
        self.transport_diagnostic = transport_diagnostic
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True)
class TushareResponse:
    """Validated response rows with mandatory exact response evidence bytes."""

    api_name: str
    request_id: str
    reported_count: int
    has_more: bool
    fields: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    raw_body: bytes
    provider_reported_count: int
    item_count: int


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


def _elapsed_ms(started_at: float) -> int:
    try:
        elapsed = (_MONOTONIC() - started_at) * 1000
    except Exception:
        return 0
    if not math.isfinite(elapsed) or elapsed <= 0:
        return 0
    return int(elapsed)


def _transport_failure_class(exc: Exception, *, phase: str) -> str:
    if isinstance(exc, socket.gaierror):
        return "DNS"
    if isinstance(exc, (ssl.SSLError, ssl.CertificateError)):
        return "TLS"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "TIMEOUT"
    if isinstance(
        exc,
        (ConnectionResetError, ConnectionAbortedError, BrokenPipeError),
    ):
        return "CONNECTION_RESET"
    if isinstance(exc, ConnectionRefusedError):
        return "CONNECT"
    if isinstance(exc, OSError) and exc.errno in _CONNECT_ERRNOS:
        return "CONNECT"
    if isinstance(exc, http.client.IncompleteRead):
        return "READ"
    if isinstance(exc, OSError) and phase == "RESPONSE_BODY":
        return "READ"
    return "UNKNOWN"


def _transport_diagnostic(
    exc: Exception,
    *,
    phase: str,
    started_at: float,
) -> TushareTransportDiagnostic:
    safe_phase = phase if phase in _TRANSPORT_FAILURE_PHASES else "UNKNOWN"
    return TushareTransportDiagnostic(
        failure_class=_transport_failure_class(exc, phase=safe_phase),
        failure_phase=safe_phase,
        elapsed_ms=_elapsed_ms(started_at),
    )


def validate_official_endpoint(value: str) -> None:
    """Reject every endpoint except the exact official HTTPS root."""

    if type(value) is not str or value != OFFICIAL_TUSHARE_URL:
        _fail("TUSHARE_ENDPOINT_BLOCKED")


def _validate_json_list(value: list[Any], *, depth: int) -> None:
    if len(value) > MAX_CONTAINER_ITEMS:
        _fail("TUSHARE_REQUEST_INVALID")
    for item in value:
        _validate_json_value(item, depth=depth + 1)


def _validate_json_object(value: dict[Any, Any], *, depth: int) -> None:
    if len(value) > MAX_CONTAINER_ITEMS:
        _fail("TUSHARE_REQUEST_INVALID")
    for key, item in value.items():
        if type(key) is not str or not key or key.casefold() in _FORBIDDEN_PARAM_KEYS:
            _fail("TUSHARE_REQUEST_INVALID")
        _validate_json_value(item, depth=depth + 1)


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
        _validate_json_list(value, depth=depth)
        return
    if type(value) is dict:
        _validate_json_object(value, depth=depth)
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
    raw: bytes | None = None
    try:
        raw = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        pass
    if raw is None:
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


def _decode_provider_payload(
    raw: bytes,
    *,
    strict_decimal_decode: bool,
) -> dict[str, Any]:
    decode_options: dict[str, Any] = {
        "object_pairs_hook": _unique_object,
        "parse_constant": _reject_constant,
    }
    if strict_decimal_decode:
        decode_options["parse_float"] = Decimal
    payload: Any = None
    decode_failed = False
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            **decode_options,
        )
    except TushareHttpsError:
        raise
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        decode_failed = True
    if decode_failed:
        _fail("TUSHARE_RESPONSE_INVALID")
    if type(payload) is not dict or set(payload) != _RESPONSE_ENVELOPE_FIELDS:
        _fail("TUSHARE_RESPONSE_INVALID")
    return payload


def _validated_diagnostic_metadata(payload: dict[str, Any]) -> tuple[int, str]:
    provider_code = payload["code"]
    request_id = payload["request_id"]
    if (
        type(provider_code) is not int
        or type(payload["detail"]) is not str
        or type(payload["msg"]) is not str
        or type(request_id) is not str
        or not request_id
    ):
        _fail("TUSHARE_RESPONSE_INVALID")
    if provider_code != 0:
        _fail("TUSHARE_API_ERROR")
    return provider_code, request_id


def _validated_response_request_id(payload: dict[str, Any]) -> str:
    provider_code = payload["code"]
    if type(provider_code) is not int:
        _fail("TUSHARE_RESPONSE_INVALID")
    if provider_code != 0:
        _fail("TUSHARE_API_ERROR")
    if type(payload["detail"]) is not str or type(payload["msg"]) is not str:
        _fail("TUSHARE_RESPONSE_INVALID")
    request_id = payload["request_id"]
    if type(request_id) is not str or not request_id:
        _fail("TUSHARE_RESPONSE_INVALID")
    return request_id


def _response_data_object(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload["data"]
    if type(data) is not dict or set(data) != _RESPONSE_DATA_FIELDS:
        _fail("TUSHARE_RESPONSE_INVALID")
    return data


def _validated_diagnostic_data(
    data: dict[str, Any],
    *,
    max_response_items: int,
) -> tuple[int, bool, list[str], list[Any]]:
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
    return reported_count, has_more, response_fields, items


def _validated_response_data(
    data: dict[str, Any],
    *,
    expected_fields: tuple[str, ...],
    max_response_items: int,
) -> tuple[int, bool, list[Any]]:
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
    return reported_count, has_more, items


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


def _diagnostic_shape(
    items: list[Any],
) -> tuple[tuple[int, ...], tuple[str, ...], int, int, int]:
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
    return (
        tuple(sorted(row_widths)),
        tuple(sorted(cell_types)),
        text_cell_count,
        non_nfc_text_count,
        max_text_utf8_bytes,
    )


def _validated_rows(
    items: list[Any],
    *,
    expected_fields: tuple[str, ...],
    strict_decimal_decode: bool,
) -> tuple[tuple[Any, ...], ...]:
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
    return tuple(rows)


def _decode_schema_diagnostic(
    raw: bytes,
    *,
    api_name: str,
    expected_fields: tuple[str, ...],
    max_response_items: int = MAX_CONTAINER_ITEMS,
) -> TushareSchemaDiagnostic:
    """Project response shape while irreversibly discarding business values."""

    payload = _decode_provider_payload(raw, strict_decimal_decode=True)
    provider_code, request_id = _validated_diagnostic_metadata(payload)
    data = _response_data_object(payload)
    reported_count, has_more, response_fields, items = _validated_diagnostic_data(
        data,
        max_response_items=max_response_items,
    )
    (
        row_widths,
        cell_types,
        text_cell_count,
        non_nfc_text_count,
        max_text_utf8_bytes,
    ) = _diagnostic_shape(items)
    request_id_sha256 = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return TushareSchemaDiagnostic(
        api_name=api_name,
        status="OBSERVED",
        provider_code=provider_code,
        request_id_sha256=request_id_sha256,
        response_body_sha256=hashlib.sha256(raw).hexdigest(),
        provider_reported_count=reported_count,
        item_count=len(items),
        has_more=has_more,
        observed_fields=tuple(response_fields),
        expected_fields_match=tuple(response_fields) == expected_fields,
        row_widths=row_widths,
        cell_types=cell_types,
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
    payload = _decode_provider_payload(
        raw,
        strict_decimal_decode=strict_decimal_decode,
    )
    request_id = _validated_response_request_id(payload)
    data = _response_data_object(payload)
    reported_count, has_more, items = _validated_response_data(
        data,
        expected_fields=expected_fields,
        max_response_items=max_response_items,
    )
    rows = _validated_rows(
        items,
        expected_fields=expected_fields,
        strict_decimal_decode=strict_decimal_decode,
    )
    normalized_reported_count = reported_count
    if strict_decimal_decode and reported_count == 0 and rows:
        # Official batch endpoints use zero as a count placeholder even when
        # ``items`` is non-empty. Strict consumers bind the accepted row
        # count to item cardinality; legacy decoding retains the provider value.
        normalized_reported_count = len(rows)
    return TushareResponse(
        api_name=api_name,
        request_id=request_id,
        reported_count=normalized_reported_count,
        has_more=has_more,
        fields=expected_fields,
        rows=rows,
        raw_body=bytes(raw),
        provider_reported_count=reported_count,
        item_count=len(rows),
    )


def replay_tushare_response_bytes(
    raw: bytes,
    *,
    api_name: str,
    expected_fields: Sequence[str],
    strict_decimal_decode: bool = True,
    max_response_items: int = MAX_CONTAINER_ITEMS,
) -> TushareResponse:
    """Independently replay one exact response entity under the public policy."""

    if type(raw) is not bytes or not raw or len(raw) > MAX_RESPONSE_BYTES:
        _fail("TUSHARE_RESPONSE_INVALID")
    if (
        type(api_name) is not str
        or _API_NAME_RE.fullmatch(api_name) is None
        or isinstance(expected_fields, (str, bytes))
        or type(strict_decimal_decode) is not bool
        or type(max_response_items) is not int
        or not 1 <= max_response_items <= _MAX_CONFIGURED_RESPONSE_ITEMS
    ):
        _fail("TUSHARE_RESPONSE_INVALID")
    fields: tuple[Any, ...] | None = None
    try:
        fields = tuple(expected_fields)
    except TypeError:
        pass
    if fields is None:
        _fail("TUSHARE_RESPONSE_INVALID")
    if (
        not fields
        or len(fields) != len(set(fields))
        or any(type(field) is not str or _FIELD_RE.fullmatch(field) is None for field in fields)
    ):
        _fail("TUSHARE_RESPONSE_INVALID")
    return _decode_response(
        raw,
        api_name=api_name,
        expected_fields=fields,
        strict_decimal_decode=strict_decimal_decode,
        max_response_items=max_response_items,
    )


class OfficialTushareHttpsClient:
    """No-redirect client whose credential source is only ``TUSHARE_TOKEN``."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        strict_decimal_decode: bool = False,
        max_response_items: int = MAX_CONTAINER_ITEMS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
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
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= MAX_RESPONSE_BYTES:
            _fail("TUSHARE_CLIENT_CONFIG_INVALID")
        self._timeout_seconds = float(timeout_seconds)
        self._strict_decimal_decode = strict_decimal_decode
        self._max_response_items = max_response_items
        self._max_response_bytes = max_response_bytes

    def _prepare_request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> tuple[tuple[str, ...], bytes]:
        request_invalid = False
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
                    type(field) is not str or _FIELD_RE.fullmatch(field) is None for field in fields
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
        except Exception:
            request_invalid = True
        if request_invalid:
            _fail("TUSHARE_REQUEST_INVALID")
        raise AssertionError("validated request did not return")

    def _fetch_raw(self, body: bytes) -> bytes:
        connection: http.client.HTTPSConnection | None = None
        started_at = _MONOTONIC()
        phase = "TLS_CONTEXT"
        diagnostic: TushareTransportDiagnostic | None = None
        try:
            context = _CREATE_DEFAULT_CONTEXT()
            connection = _HTTPS_CONNECTION(
                OFFICIAL_TUSHARE_HOST,
                OFFICIAL_TUSHARE_PORT,
                timeout=self._timeout_seconds,
                context=context,
            )
            phase = "REQUEST_SEND"
            connection.request(
                "POST",
                OFFICIAL_TUSHARE_PATH,
                body=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json; charset=utf-8",
                    "User-Agent": "myquant-market-data",
                },
            )
            phase = "RESPONSE_HEADERS"
            response = connection.getresponse()
            status = response.status
            if 300 <= status < 400:
                _fail("TUSHARE_REDIRECT_BLOCKED")
            if status != 200:
                _fail("TUSHARE_HTTP_STATUS_ERROR")
            phase = "RESPONSE_BODY"
            raw = response.read(self._max_response_bytes + 1)
            if len(raw) > self._max_response_bytes:
                _fail("TUSHARE_RESPONSE_TOO_LARGE")
            return raw
        except TushareHttpsError:
            raise
        except Exception as exc:
            diagnostic = _transport_diagnostic(
                exc,
                phase=phase,
                started_at=started_at,
            )
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
        if diagnostic is not None:
            raise TushareHttpsError(
                "TUSHARE_TRANSPORT_ERROR",
                transport_diagnostic=diagnostic,
            ) from None
        raise AssertionError("transport failure missing diagnostic")

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
    "TushareTransportDiagnostic",
    "replay_tushare_response_bytes",
    "validate_official_endpoint",
]
