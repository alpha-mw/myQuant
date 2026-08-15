"""Exact-byte Tushare response fixtures shared by stable transport tests."""

from __future__ import annotations

from decimal import Decimal
import json
import math
from typing import Any, Sequence

from quant_investor.market.tushare_transport import (
    TushareResponse,
    replay_tushare_response_bytes,
)


def _json_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("fixture float must be finite")
        return repr(value)
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("fixture Decimal must be finite")
        return str(value)
    if type(value) is str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_json_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported fixture value: {type(value).__name__}")


def make_tushare_response(
    *,
    api_name: str,
    request_id: str,
    fields: Sequence[str],
    rows: Sequence[Sequence[Any]],
    reported_count: int | None = None,
    has_more: bool = False,
) -> TushareResponse:
    """Build through the real byte replay validator, never the dataclass directly."""

    normalized_fields = tuple(fields)
    normalized_rows = tuple(tuple(row) for row in rows)
    provider_count = len(normalized_rows) if reported_count is None else reported_count
    raw = (
        "{"
        '"code":0,'
        '"data":{'
        f'"count":{provider_count},'
        f'"fields":{_json_scalar(normalized_fields)},'
        f'"has_more":{_json_scalar(has_more)},'
        f'"items":{_json_scalar(normalized_rows)}'
        "},"
        '"detail":"",'
        '"msg":"",'
        f'"request_id":{_json_scalar(request_id)}'
        "}"
    ).encode("utf-8")
    return replay_tushare_response_bytes(
        raw,
        api_name=api_name,
        expected_fields=normalized_fields,
        strict_decimal_decode=True,
    )


__all__ = ["make_tushare_response"]
