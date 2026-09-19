"""Pure reconstruction of Sina CN quote wire bytes; no network or filesystem access."""

import re


class SinaQuoteParseError(ValueError):
    """The wire response or its deterministic symbol mapping is invalid."""


def _validated_mappings(mappings: list[dict[str, str]]) -> dict[str, str]:
    symbols = []
    for row in mappings:
        if (
            type(row) is not dict
            or set(row) != {"provider_symbol", "symbol"}
            or type(row.get("symbol")) is not str
            or re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", row["symbol"]) is None
            or row.get("provider_symbol") != row["symbol"][-2:].lower() + row["symbol"][:6]
        ):
            raise SinaQuoteParseError("SINA_SYMBOL_MAPPING_INVALID")
        symbols.append(row["symbol"])
    if not symbols or symbols != sorted(set(symbols)):
        raise SinaQuoteParseError("SINA_SYMBOL_MAPPING_INVALID")
    return {row["provider_symbol"]: row["symbol"] for row in mappings}


def parse_sina_quote_response(raw: bytes, mappings: list[dict[str, str]]) -> list[dict[str, str]]:
    by_provider = _validated_mappings(mappings)
    try:
        text = raw.decode("gb18030", errors="strict")
    except UnicodeError as exc:
        raise SinaQuoteParseError("SINA_RESPONSE_DECODE_FAILED") from exc
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        prefix = "var hq_str_"
        if not line.strip():
            continue
        if not line.startswith(prefix) or '="' not in line or not line.endswith('";'):
            raise SinaQuoteParseError("SINA_RESPONSE_LINE_INVALID")
        offset = len(prefix)
        provider_symbol, payload = line[offset:].split('="', 1)
        if provider_symbol not in by_provider:
            raise SinaQuoteParseError("SINA_RESPONSE_SYMBOL_SET_EXTRA")
        if by_provider[provider_symbol] in rows:
            raise SinaQuoteParseError("SINA_RESPONSE_SYMBOL_DUPLICATE")
        fields = payload[:-2].split(",")
        if len(fields) < 32 or not fields[0]:
            raise SinaQuoteParseError("SINA_RESPONSE_FIELDS_INVALID")
        symbol = by_provider[provider_symbol]
        rows[symbol] = {
            "symbol": symbol,
            "name": fields[0],
            "open": fields[1],
            "previous_close": fields[2],
            "price": fields[3],
            "high": fields[4],
            "low": fields[5],
            "volume": fields[8],
            "amount": fields[9],
            "provider_date": fields[30],
            "provider_time": fields[31],
        }
    ordered = [rows[row["symbol"]] for row in mappings if row["symbol"] in rows]
    if [row["symbol"] for row in ordered] != [row["symbol"] for row in mappings]:
        raise SinaQuoteParseError("SINA_RESPONSE_SYMBOL_SET_INCOMPLETE")
    return ordered
