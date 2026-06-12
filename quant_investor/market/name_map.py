"""Cached market symbol-name lookup helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.market.config import get_market_settings


_STOCK_NAME_CACHE: dict[str, dict[str, str]] = {"CN": {}, "US": {}}


def clear_stock_name_cache(market: str | None = None) -> None:
    """Clear cached symbol-name maps for tests and explicit refreshes."""

    if market is None:
        for key in list(_STOCK_NAME_CACHE):
            _STOCK_NAME_CACHE[key] = {}
        return
    settings = get_market_settings(market)
    _STOCK_NAME_CACHE[settings.market] = {}


def is_unknown_stock_name(value: Any) -> bool:
    text = str(value or "").strip()
    return not text or text.upper() in {
        "N/A",
        "NA",
        "NONE",
        "NULL",
        "UNKNOWN",
        "未知",
    }


def _load_us_stock_names_from_local_sources() -> dict[str, str]:
    """Build a US symbol-name map from local metadata only."""

    names: dict[str, str] = {}
    metadata_paths = [
        Path("data/us_universe/us_market_caps.json"),
        Path("data/us_universe/complete_us_universe.json"),
    ]
    for path in metadata_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        symbols = (
            payload.get("symbols")
            if isinstance(payload.get("symbols"), dict)
            else payload
        )
        if not isinstance(symbols, dict):
            continue
        for symbol, value in symbols.items():
            normalized_symbol = str(symbol or "").strip().upper()
            if not normalized_symbol or normalized_symbol in names:
                continue
            candidate = ""
            if isinstance(value, dict):
                candidate = str(
                    value.get("name")
                    or value.get("company_name")
                    or value.get("long_name")
                    or value.get("short_name")
                    or ""
                ).strip()
            elif isinstance(value, str):
                candidate = value.strip()
            if not is_unknown_stock_name(candidate):
                names[normalized_symbol] = candidate
    return names


def _load_cached_name_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(symbol).strip().upper(): str(name).strip()
        for symbol, name in payload.items()
        if str(symbol).strip() and str(name).strip()
    }


def load_stock_names(
    market: str,
    refresh: bool = False,
    *,
    allow_provider: bool = True,
) -> dict[str, str]:
    settings = get_market_settings(market)
    cache = _STOCK_NAME_CACHE.setdefault(settings.market, {})
    if cache and not refresh:
        return cache

    cache_path = Path(settings.name_cache_file)
    if cache_path.exists() and not refresh:
        _STOCK_NAME_CACHE[settings.market] = _load_cached_name_file(cache_path)
        return _STOCK_NAME_CACHE[settings.market]

    if settings.market == "US":
        payload = _load_us_stock_names_from_local_sources()
        if payload:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            _STOCK_NAME_CACHE[settings.market] = payload
            return payload
        return cache

    if not allow_provider:
        return cache

    try:
        import tushare as ts

        pro = create_tushare_pro(ts, config.TUSHARE_TOKEN, config.TUSHARE_URL)
        if pro is None:
            return cache
        df = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,name",
        )
        if df is None or df.empty:
            return cache
        payload = dict(zip(df["ts_code"], df["name"]))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _STOCK_NAME_CACHE[settings.market] = payload
        return payload
    except Exception:
        return cache


def load_cn_stock_names(
    refresh: bool = False,
    *,
    allow_provider: bool = True,
) -> dict[str, str]:
    return load_stock_names(
        "CN",
        refresh=refresh,
        allow_provider=allow_provider,
    )


def load_us_stock_names(refresh: bool = False) -> dict[str, str]:
    return load_stock_names("US", refresh=refresh)


def get_stock_name(symbol: str, market: str = "CN") -> str:
    settings = get_market_settings(market)
    if not _STOCK_NAME_CACHE.get(settings.market):
        load_stock_names(settings.market)
    fallback = "未知" if settings.market == "CN" else "N/A"
    normalized_symbol = str(symbol or "").strip().upper()
    return _STOCK_NAME_CACHE[settings.market].get(normalized_symbol, fallback)


def load_company_name_map(
    market: str,
    *,
    allow_provider: bool = False,
) -> dict[str, str]:
    """Return normalized company names for DAG lookups."""

    settings = get_market_settings(market)
    return {
        str(symbol).strip().upper(): str(name).strip()
        for symbol, name in load_stock_names(
            settings.market,
            allow_provider=allow_provider,
        ).items()
        if str(symbol).strip() and str(name).strip()
    }


__all__ = [
    "_STOCK_NAME_CACHE",
    "clear_stock_name_cache",
    "get_stock_name",
    "is_unknown_stock_name",
    "load_cn_stock_names",
    "load_company_name_map",
    "load_stock_names",
    "load_us_stock_names",
]
