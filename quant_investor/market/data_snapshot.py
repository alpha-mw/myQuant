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
from quant_investor.data.storage.csv_reader import peek_latest_date
from quant_investor.market.cn_resolver import CNUniverseResolver
from quant_investor.market.cn_symbol_status import evaluate_symbol_local_status
from quant_investor.market.config import get_market_settings, normalize_categories, normalize_universe
from quant_investor.market.shared_csv_reader import SharedCSVReader

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
    resolver = CNUniverseResolver(data_dir=str(data_dir))
    reader = SharedCSVReader(market="CN", data_dir=data_dir, resolver=resolver)
    freshness_index = _load_cn_freshness_index(data_dir)

    physical_directories = [path for path in resolver.physical_directories_for_full_a() if path.exists()]
    if not physical_directories:
        physical_directories = [data_dir / category for category in _CN_PHYSICAL_DIRECTORIES if (data_dir / category).exists()]

    category_symbol_counts: dict[str, int]
    if universe_key == "full_a" or "full_a" in selected_categories or universe_key == "custom":
        category_symbol_counts = {
            category: _count_csvs(data_dir / category)
            for category in _CN_PHYSICAL_DIRECTORIES
            if (data_dir / category).exists()
        }
        inventory_symbols, resolved_paths = resolver.collect_full_a_inventory(local_union_fallback_used=True)
    else:
        category_symbol_counts = {
            category: _count_csvs(data_dir / category)
            for category in selected_categories
            if (data_dir / category).exists()
        }
        inventory_symbols = []
        resolved_paths: dict[str, str] = {}
        for category in selected_categories:
            for symbol in reader.list_symbols(category, category=category):
                inventory_symbols.append(symbol)
                resolved = reader.resolve_symbol_path(symbol, universe_key=category, category=category)
                if resolved is not None:
                    resolved_paths[symbol] = str(resolved)

    if requested_symbols:
        for symbol in requested_symbols:
            if symbol not in resolved_paths:
                resolved = reader.resolve_symbol_path(symbol, universe_key="full_a", category="full_a")
                if resolved is not None:
                    resolved_paths[symbol] = str(resolved)

    observed_dates: dict[str, str] = {}
    for symbol, path_str in resolved_paths.items():
        indexed = freshness_index.get(symbol, "")
        if indexed:
            observed_dates[symbol] = indexed
            continue
        latest = peek_latest_date(path_str)
        if latest:
            observed_dates[symbol] = latest

    local_latest_trade_date = max(observed_dates.values(), default="")
    if not local_latest_trade_date and requested_symbols:
        for symbol in requested_symbols:
            latest = reader.peek_symbol_latest_date(symbol, universe_key="full_a", category="full_a")
            if latest:
                local_latest_trade_date = max(local_latest_trade_date, latest)

    date_distribution = Counter(date for date in observed_dates.values() if str(date).strip())
    date_distribution_top = [
        {"trade_date": trade_date, "symbol_count": int(symbol_count)}
        for trade_date, symbol_count in sorted(date_distribution.items(), key=lambda item: (-item[1], item[0]), reverse=False)[:5]
    ]

    missing_requested_symbols: list[str] = []
    unreadable_requested_symbols: list[str] = []
    stale_requested_symbols: list[str] = []
    for symbol in requested_symbols:
        target_trade_date = local_latest_trade_date or reader.peek_symbol_latest_date(
            symbol,
            universe_key="full_a",
            category="full_a",
        )
        status = evaluate_symbol_local_status(
            symbol,
            category="full_a",
            resolver=resolver,
            csv_reader=reader,
            latest_trade_date=target_trade_date,
            allowed_stale_symbols=[],
            suspended_symbols=[],
            freshness_mode=_freshness_mode_for_market("CN"),
            strict_trade_date=target_trade_date,
            stable_trade_date=target_trade_date,
            fast_date_peek=True,
        )
        if status.local_status == "missing":
            missing_requested_symbols.append(symbol)
        elif status.local_status == "unreadable":
            unreadable_requested_symbols.append(symbol)
        elif status.local_status in {"stale", "stale_cached", "suspended_stale"}:
            stale_requested_symbols.append(symbol)

    data_directories = [str(path) for path in physical_directories]
    resolver_priority = list((resolver.snapshot() or {}).get("directory_priority", [])) or list(_CN_PHYSICAL_DIRECTORIES)
    summary_parts = [
        f"本地 A 股数据更新至 {local_latest_trade_date or '未知日期'}",
        "分析默认直接使用现有本地数据",
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
        "inventory_symbol_count": len(set(inventory_symbols or resolved_paths.keys())),
    }


def _build_us_snapshot(
    *,
    universe_key: str,
    selected_categories: list[str],
    requested_symbols: list[str],
    data_dir: Path,
) -> dict[str, Any]:
    reader = SharedCSVReader(market="US", data_dir=data_dir)
    category_symbol_counts = {
        category: _count_csvs(data_dir / category)
        for category in selected_categories
        if (data_dir / category).exists()
    }
    observed_dates: dict[str, str] = {}
    missing_requested_symbols: list[str] = []
    unreadable_requested_symbols: list[str] = []
    for symbol in requested_symbols:
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
    data_directories = [str(data_dir / category) for category in selected_categories if (data_dir / category).exists()]
    return {
        "market": "US",
        "universe_key": universe_key,
        "local_latest_trade_date": local_latest_trade_date,
        "freshness_mode": _freshness_mode_for_market("US"),
        "category_symbol_counts": category_symbol_counts,
        "date_distribution_top": date_distribution_top,
        "data_directories": data_directories,
        "resolver_priority": list(selected_categories),
        "data_quality_issue_count": len(missing_requested_symbols) + len(unreadable_requested_symbols),
        "summary_text": f"本地 US 数据更新至 {local_latest_trade_date or '未知日期'}；分析默认直接使用现有本地数据。",
        "missing_requested_symbols": missing_requested_symbols,
        "unreadable_requested_symbols": unreadable_requested_symbols,
        "stale_requested_symbols": [],
        "requested_symbol_count": len(requested_symbols),
        "inventory_symbol_count": sum(category_symbol_counts.values()),
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
