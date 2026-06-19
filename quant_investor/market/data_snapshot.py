"""
Local-only market data snapshot helpers.

This module is intentionally analysis-safe:
- no completeness report
- no downloader invocation
- no Tushare/network calls
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from quant_investor.config import config
from quant_investor.market.config import get_market_settings, normalize_categories, normalize_universe
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.us_market_cap_filter import USMarketCapFilter

_CN_PHYSICAL_DIRECTORIES: tuple[str, ...] = ("hs300", "zz500", "zz1000", "other")


def _normalize_symbols(symbols: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for symbol in symbols or []:
        text = str(symbol or "").strip().upper()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _freshness_mode_for_market(market: str) -> str:
    if market.upper() == "CN":
        mode = str(getattr(config, "CN_FRESHNESS_MODE", "stable") or "stable").strip().lower()
        return mode if mode in {"stable", "strict"} else "stable"
    return "local_only"


def _parquet_data_root_from_market_dir(base_dir: Path, *, market: str) -> Path:
    market_key = str(market or "").strip().lower()
    for candidate in [
        base_dir,
        base_dir.parent,
        base_dir.parent.parent,
        base_dir.parent.parent.parent,
    ]:
        if (candidate / "parquet" / market_key).exists():
            return candidate
        if (candidate / "parquet_serving" / market_key).exists():
            return candidate
    return Path("data")


def _load_cn_freshness_index(data_dir: Path) -> dict[str, str]:
    path = data_dir / ".cache" / "freshness_index.json"
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        symbols = payload.get("symbols", {}) if isinstance(payload, dict) else {}
        if not isinstance(symbols, dict):
            return {}
        return {
            str(symbol).strip().upper(): str(trade_date)
            for symbol, trade_date in symbols.items()
            if str(symbol or "").strip() and str(trade_date or "").strip()
        }
    except Exception:
        return {}


def _selected_universe_key(
    *,
    market: str,
    universe: str | None,
    categories: list[str] | None,
) -> tuple[str, list[str]]:
    settings = get_market_settings(market)
    selected_categories = (
        normalize_universe(settings.market, universe)
        if universe is not None
        else normalize_categories(settings.market, categories)
    )
    universe_key = universe or (selected_categories[0] if len(selected_categories) == 1 else "custom")
    return universe_key, selected_categories


def _count_csvs(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.glob("*.csv") if path.is_file())


def _build_cn_snapshot(
    *,
    universe_key: str,
    selected_categories: list[str],
    requested_symbols: list[str],
    data_dir: Path,
) -> dict[str, Any]:
    parquet_data_root = (
        data_dir
        if (data_dir / "parquet" / "cn").exists()
        else data_dir.parent
        if (data_dir.parent / "parquet" / "cn").exists()
        else Path("data")
    )
    reader = MarketDataReader(market="CN", data_root=parquet_data_root)
    gate = reader.clean_snapshot_gate()
    if not gate.get("healthy"):
        blockers = list(gate.get("blockers", []) or [])
        return {
            "market": "CN",
            "universe_key": universe_key,
            "local_latest_trade_date": "",
            "freshness_mode": _freshness_mode_for_market("CN"),
            "category_symbol_counts": {},
            "date_distribution_top": [],
            "data_directories": [],
            "resolver_priority": ["parquet_canonical", "parquet_serving"],
            "data_quality_issue_count": len(blockers),
            "summary_text": "本地 Parquet canonical snapshot 未通过 strict 校验；分析应 fail closed。",
            "missing_requested_symbols": list(requested_symbols),
            "unreadable_requested_symbols": [],
            "stale_requested_symbols": [],
            "requested_symbol_count": len(requested_symbols),
            "inventory_symbol_count": 0,
            "storage_backend": "parquet",
            "strict_parquet_gate": gate,
            "fail_closed": True,
        }

    category_keys = (
        ["hs300", "zz500", "zz1000"]
        if universe_key in {"full_a", "custom"} or "full_a" in selected_categories
        else list(selected_categories)
    )
    category_symbol_counts = {
        category: len(reader.list_symbols(category))
        for category in category_keys
    }
    inventory_symbols = reader.list_symbols("full_a")
    local_latest_trade_date = reader.latest_trade_date(universe_key)

    missing_requested_symbols: list[str] = []
    unreadable_requested_symbols: list[str] = []
    stale_requested_symbols: list[str] = []
    observed_dates: dict[str, str] = {}
    for symbol in requested_symbols:
        latest = reader.peek_symbol_latest_date(symbol, universe_key="full_a", category="full_a")
        if latest:
            observed_dates[symbol] = latest
        elif symbol in requested_symbols:
            resolved = reader.resolve_symbol_path(symbol, universe_key="full_a", category="full_a")
            if resolved is None:
                missing_requested_symbols.append(symbol)
            else:
                unreadable_requested_symbols.append(symbol)
    if requested_symbols:
        for symbol, latest in observed_dates.items():
            if latest and local_latest_trade_date and latest < local_latest_trade_date:
                stale_requested_symbols.append(symbol)

    if observed_dates:
        date_distribution = Counter(date for date in observed_dates.values() if str(date).strip())
        date_distribution_top = [
            {"trade_date": trade_date, "symbol_count": int(symbol_count)}
            for trade_date, symbol_count in sorted(date_distribution.items(), key=lambda item: (-item[1], item[0]))[:5]
        ]
    else:
        date_distribution_top = [
            {
                "trade_date": local_latest_trade_date,
                "symbol_count": len(set(inventory_symbols)),
            }
        ] if local_latest_trade_date else []

    data_directories = [str(Path(gate.get("serving_root", "")))] if gate.get("serving_root") else []
    resolver_priority = ["parquet_serving", "parquet_canonical"]
    summary_parts = [
        f"本地 A 股数据更新至 {local_latest_trade_date or '未知日期'}",
        "分析默认使用 Parquet canonical + serving layer",
    ]
    if category_symbol_counts:
        summary_parts.append(
            "目录结构: "
            + " / ".join(f"{category}={count}" for category, count in category_symbol_counts.items())
        )
    if missing_requested_symbols or unreadable_requested_symbols:
        summary_parts.append(
            f"请求标的缺失/不可读 {len(missing_requested_symbols) + len(unreadable_requested_symbols)} 只"
        )
    elif stale_requested_symbols:
        summary_parts.append(f"请求标的中存在陈旧样本 {len(stale_requested_symbols)} 只")

    return {
        "market": "CN",
        "universe_key": universe_key,
        "local_latest_trade_date": local_latest_trade_date,
        "freshness_mode": _freshness_mode_for_market("CN"),
        "category_symbol_counts": category_symbol_counts,
        "date_distribution_top": date_distribution_top,
        "data_directories": data_directories,
        "resolver_priority": resolver_priority,
        "data_quality_issue_count": len(missing_requested_symbols) + len(unreadable_requested_symbols),
        "summary_text": "；".join(summary_parts) + "。",
        "missing_requested_symbols": missing_requested_symbols,
        "unreadable_requested_symbols": unreadable_requested_symbols,
        "stale_requested_symbols": stale_requested_symbols,
        "requested_symbol_count": len(requested_symbols),
        "inventory_symbol_count": len(set(inventory_symbols)),
        "storage_backend": "parquet",
        "strict_parquet_gate": gate,
        "fail_closed": False,
    }


def _build_us_snapshot(
    *,
    universe_key: str,
    selected_categories: list[str],
    requested_symbols: list[str],
    data_dir: Path,
) -> dict[str, Any]:
    data_root = _parquet_data_root_from_market_dir(data_dir, market="US")
    reader = MarketDataReader(market="US", data_root=data_root)
    gate = reader.clean_snapshot_gate()
    if not gate.get("healthy"):
        blockers = [str(item) for item in gate.get("blockers", []) if str(item).strip()]
        return {
            "market": "US",
            "universe_key": universe_key,
            "local_latest_trade_date": "",
            "freshness_mode": _freshness_mode_for_market("US"),
            "category_symbol_counts": {},
            "market_cap_filter": {},
            "date_distribution_top": [],
            "data_directories": [],
            "resolver_priority": list(selected_categories),
            "data_quality_issue_count": len(blockers) or 1,
            "summary_text": "本地 US Parquet 数据不可用；生产读取已禁止 CSV 回退。",
            "missing_requested_symbols": list(requested_symbols),
            "unreadable_requested_symbols": [],
            "stale_requested_symbols": [],
            "requested_symbol_count": len(requested_symbols),
            "observed_symbol_count": 0,
            "inventory_symbol_count": 0,
            "storage_backend": "parquet",
            "strict_parquet_gate": gate,
            "fail_closed": True,
        }
    symbols_by_category = {
        category: reader.list_symbols(category)
        for category in selected_categories
    }
    market_cap_filter_metadata: dict[str, Any] = {}
    if not requested_symbols:
        filtered_by_category: dict[str, list[str]] = {}
        filter_metadata: dict[str, Any] = {}
        market_cap_filter = USMarketCapFilter()
        for category, symbols in symbols_by_category.items():
            filtered_symbols, metadata = market_cap_filter.filter_symbols(symbols, fetch_missing=False)
            filtered_by_category[category] = filtered_symbols
            filter_metadata[category] = metadata
        symbols_by_category = filtered_by_category
        market_cap_filter_metadata = filter_metadata
    category_symbol_counts = {
        category: len(symbols)
        for category, symbols in symbols_by_category.items()
    }
    symbols_to_check = requested_symbols or list(
        dict.fromkeys(
            symbol
            for symbols in symbols_by_category.values()
            for symbol in symbols
            if str(symbol or "").strip()
        )
    )
    observed_dates: dict[str, str] = {}
    missing_requested_symbols: list[str] = []
    unreadable_requested_symbols: list[str] = []
    for symbol in symbols_to_check:
        latest = reader.peek_symbol_latest_date(symbol, universe_key=universe_key)
        if latest:
            observed_dates[symbol] = latest
        else:
            path = reader.resolve_symbol_path(symbol, universe_key=universe_key)
            if path is None:
                missing_requested_symbols.append(symbol)
            else:
                unreadable_requested_symbols.append(symbol)

    local_latest_trade_date = max(observed_dates.values(), default="")
    date_distribution = Counter(date for date in observed_dates.values() if str(date).strip())
    date_distribution_top = [
        {"trade_date": trade_date, "symbol_count": int(symbol_count)}
        for trade_date, symbol_count in sorted(date_distribution.items(), key=lambda item: (-item[1], item[0]), reverse=False)[:5]
    ]
    data_directories = [
        str(gate.get("table_root", "")),
        str(gate.get("serving_root", "")),
    ]
    return {
        "market": "US",
        "universe_key": universe_key,
        "local_latest_trade_date": local_latest_trade_date,
        "freshness_mode": _freshness_mode_for_market("US"),
        "category_symbol_counts": category_symbol_counts,
        "market_cap_filter": market_cap_filter_metadata,
        "date_distribution_top": date_distribution_top,
        "data_directories": data_directories,
        "resolver_priority": list(selected_categories),
        "data_quality_issue_count": len(missing_requested_symbols) + len(unreadable_requested_symbols),
        "summary_text": f"本地 US 数据更新至 {local_latest_trade_date or '未知日期'}；分析默认直接使用现有本地数据。",
        "missing_requested_symbols": missing_requested_symbols,
        "unreadable_requested_symbols": unreadable_requested_symbols,
        "stale_requested_symbols": [],
        "requested_symbol_count": len(requested_symbols),
        "observed_symbol_count": len(observed_dates),
        "inventory_symbol_count": sum(category_symbol_counts.values()),
        "storage_backend": "parquet",
        "strict_parquet_gate": gate,
        "fail_closed": False,
    }


def build_market_data_snapshot(
    *,
    market: str,
    universe: str | None = None,
    categories: list[str] | None = None,
    requested_symbols: list[str] | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    settings = get_market_settings(market)
    base_dir = Path(data_dir or settings.data_dir)
    universe_key, selected_categories = _selected_universe_key(
        market=settings.market,
        universe=universe,
        categories=categories,
    )
    normalized_requested = _normalize_symbols(requested_symbols)
    if settings.market == "CN":
        return _build_cn_snapshot(
            universe_key=universe_key,
            selected_categories=selected_categories,
            requested_symbols=normalized_requested,
            data_dir=base_dir,
        )
    return _build_us_snapshot(
        universe_key=universe_key,
        selected_categories=selected_categories,
        requested_symbols=normalized_requested,
        data_dir=base_dir,
    )
