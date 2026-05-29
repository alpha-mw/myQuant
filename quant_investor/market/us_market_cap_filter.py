#!/usr/bin/env python3
"""US market-cap universe filtering."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.request import Request, urlopen

DEFAULT_US_MIN_MARKET_CAP_USD = 10_000_000_000
DEFAULT_US_MARKET_CAP_CACHE_FILE = "data/us_universe/us_market_caps.json"


def get_us_min_market_cap_usd(default: int = DEFAULT_US_MIN_MARKET_CAP_USD) -> int:
    """Return the configured US market-cap floor in USD."""
    raw = str(os.getenv("MYQUANT_US_MIN_MARKET_CAP_USD", "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw.replace(",", "")))
    except ValueError:
        return int(default)


def get_us_market_cap_cache_file(default: str = DEFAULT_US_MARKET_CAP_CACHE_FILE) -> str:
    """Return the configured US market-cap cache path."""
    return str(os.getenv("MYQUANT_US_MARKET_CAP_CACHE_FILE", "") or default).strip() or default


class USMarketCapFilter:
    """Filter US symbols by cached/fetched market capitalization."""

    def __init__(
        self,
        threshold_usd: int | float | None = None,
        cache_file: str | Path | None = None,
        max_workers: int = 8,
        market_cap_fetcher: Callable[[str], Any] | None = None,
    ) -> None:
        self.threshold_usd = int(
            get_us_min_market_cap_usd()
            if threshold_usd is None
            else float(threshold_usd)
        )
        self.cache_file = Path(get_us_market_cap_cache_file() if cache_file is None else cache_file)
        self.max_workers = max(1, int(max_workers or 1))
        self._market_cap_fetcher = market_cap_fetcher

    @property
    def enabled(self) -> bool:
        return self.threshold_usd > 0

    @staticmethod
    def normalize_symbols(symbols: Iterable[str]) -> list[str]:
        return list(
            dict.fromkeys(
                str(symbol or "").strip().upper()
                for symbol in symbols
                if str(symbol or "").strip()
            )
        )

    @staticmethod
    def normalize_market_cap(value: Any) -> int | None:
        if value is None:
            return None
        try:
            cap = int(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            return None
        return cap if cap > 0 else None

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_file.exists():
            return {"symbols": {}}
        try:
            payload = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except Exception:
            return {"symbols": {}}
        if not isinstance(payload, dict):
            return {"symbols": {}}
        symbols = payload.get("symbols")
        if not isinstance(symbols, dict):
            payload["symbols"] = {}
        return payload

    def _save_cache(self, cache: dict[str, Any]) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache["threshold_usd"] = self.threshold_usd
        cache["updated_at"] = datetime.now().isoformat(timespec="seconds")
        cache.setdefault("symbols", {})
        self.cache_file.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _entry_market_cap(self, entry: Any) -> int | None:
        if isinstance(entry, dict):
            return self.normalize_market_cap(entry.get("market_cap"))
        return self.normalize_market_cap(entry)

    def _fetch_market_cap(self, symbol: str) -> int | None:
        if self._market_cap_fetcher is not None:
            return self.normalize_market_cap(self._market_cap_fetcher(symbol))

        try:
            import yfinance as yf  # type: ignore
        except Exception:
            return None

        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None)
            for key in ("market_cap", "marketCap"):
                if hasattr(fast_info, "get"):
                    cap = self.normalize_market_cap(fast_info.get(key))
                    if cap is not None:
                        return cap
            info_getter = getattr(ticker, "get_info", None)
            info = info_getter() if callable(info_getter) else getattr(ticker, "info", {})
            if isinstance(info, dict):
                return self.normalize_market_cap(info.get("marketCap") or info.get("market_cap"))
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_company_name(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text.upper() in {"N/A", "NA", "NONE", "NULL"}:
            return ""
        return text

    def _fetch_nasdaq_screener_profiles(self) -> dict[str, dict[str, Any]]:
        url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&offset=0&download=true"
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com",
                "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return {}
        results = payload.get("data", {}).get("rows", []) if isinstance(payload, dict) else []
        fetched: dict[str, dict[str, Any]] = {}
        if not isinstance(results, list):
            return fetched
        for item in results:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "") or "").strip().upper()
            if not symbol:
                continue
            cap = self.normalize_market_cap(item.get("marketCap"))
            profile = {
                "market_cap": cap,
                "name": self._normalize_company_name(
                    item.get("name") or item.get("companyName") or item.get("securityName")
                ),
                "source": "nasdaq_screener",
            }
            fetched[symbol] = profile
            fetched[symbol.replace("/", "-")] = profile
        return fetched

    def _fetch_nasdaq_screener_caps(self) -> dict[str, int | None]:
        return {
            symbol: self.normalize_market_cap(profile.get("market_cap"))
            for symbol, profile in self._fetch_nasdaq_screener_profiles().items()
        }

    def _fetch_many(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        if not symbols:
            return {}
        if self._market_cap_fetcher is None:
            screener_profiles = self._fetch_nasdaq_screener_profiles()
            fetched = {
                symbol: dict(screener_profiles.get(symbol, {"market_cap": None, "name": ""}))
                for symbol in symbols
            }
            missing = [symbol for symbol, profile in fetched.items() if profile.get("market_cap") is None]
            if str(os.getenv("MYQUANT_US_MARKET_CAP_INDIVIDUAL_FALLBACK", "")).strip() == "1":
                fetched.update(
                    {
                        symbol: {"market_cap": cap, "name": "", "source": "yfinance"}
                        for symbol, cap in self._fetch_many_individual(missing).items()
                    }
                )
            return fetched
        return {
            symbol: {"market_cap": cap, "name": "", "source": "yfinance"}
            for symbol, cap in self._fetch_many_individual(symbols).items()
        }

    def _fetch_many_individual(self, symbols: list[str]) -> dict[str, int | None]:
        if not symbols:
            return {}
        fetched: dict[str, int | None] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_market_cap, symbol): symbol
                for symbol in symbols
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    fetched[symbol] = self.normalize_market_cap(future.result())
                except Exception:
                    fetched[symbol] = None
        return fetched

    def filter_symbols(
        self,
        symbols: Iterable[str],
        *,
        fetch_missing: bool = True,
    ) -> tuple[list[str], dict[str, Any]]:
        normalized = self.normalize_symbols(symbols)
        if not self.enabled:
            return normalized, {
                "enabled": False,
                "threshold_usd": self.threshold_usd,
                "input_count": len(normalized),
                "included_count": len(normalized),
                "excluded_count": 0,
                "unknown_count": 0,
                "fetched_count": 0,
                "cache_file": str(self.cache_file),
            }

        cache = self._load_cache()
        cache_symbols = cache.setdefault("symbols", {})
        cache_changed = False
        enrich_names = str(os.getenv("MYQUANT_US_ENRICH_NAMES", "") or "").strip() == "1"
        if enrich_names and fetch_missing and self._market_cap_fetcher is None:
            needs_name = [
                symbol
                for symbol in normalized
                if isinstance(cache_symbols.get(symbol), dict)
                and not self._normalize_company_name(cache_symbols[symbol].get("name"))
            ]
            if needs_name:
                screener_profiles = self._fetch_nasdaq_screener_profiles()
                for symbol in needs_name:
                    profile = screener_profiles.get(symbol, {})
                    name = self._normalize_company_name(profile.get("name"))
                    if name:
                        cache_symbols[symbol]["name"] = name
                        cache_changed = True
        accepted: list[str] = []
        rejected: list[str] = []
        unknown: list[str] = []
        missing: list[str] = []
        cap_by_symbol: dict[str, int | None] = {}

        for symbol in normalized:
            cap = self._entry_market_cap(cache_symbols.get(symbol))
            if cap is None:
                missing.append(symbol)
                continue
            cap_by_symbol[symbol] = cap

        fetched: dict[str, dict[str, Any]] = {}
        if missing and fetch_missing:
            fetched = self._fetch_many(missing)
            now = datetime.now().isoformat(timespec="seconds")
            for symbol, profile in fetched.items():
                cap = self.normalize_market_cap(profile.get("market_cap") if isinstance(profile, dict) else profile)
                name = self._normalize_company_name(profile.get("name") if isinstance(profile, dict) else "")
                raw_source = profile.get("source") if isinstance(profile, dict) else ""
                source = str(raw_source or "").strip()
                cache_symbols[symbol] = {
                    "market_cap": cap,
                    "currency": "USD",
                    "updated_at": now,
                    "source": source if cap is not None and source else "unavailable",
                }
                if name:
                    cache_symbols[symbol]["name"] = name
                cap_by_symbol[symbol] = cap
            self._save_cache(cache)
        elif cache_changed:
            self._save_cache(cache)
        else:
            unknown.extend(missing)

        for symbol in normalized:
            cap = cap_by_symbol.get(symbol)
            if cap is None:
                if symbol not in unknown:
                    unknown.append(symbol)
                continue
            if cap >= self.threshold_usd:
                accepted.append(symbol)
            else:
                rejected.append(symbol)

        metadata = {
            "enabled": True,
            "threshold_usd": self.threshold_usd,
            "input_count": len(normalized),
            "included_count": len(accepted),
            "excluded_count": len(rejected) + len(unknown),
            "below_threshold_count": len(rejected),
            "unknown_count": len(unknown),
            "fetched_count": len(fetched),
            "cache_file": str(self.cache_file),
            "excluded_symbols_sample": rejected[:20],
            "unknown_symbols_sample": unknown[:20],
        }
        return accepted, metadata
