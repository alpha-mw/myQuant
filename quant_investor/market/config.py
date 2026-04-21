"""
市场级共享配置。
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_investor.config import config


@dataclass(frozen=True)
class MarketSettings:
    market: str
    market_name: str
    report_flag: str
    currency_symbol: str
    data_dir: str
    analysis_output_dir: str
    backtest_output_dir: str
    name_cache_file: str
    default_batch_size: int
    categories: tuple[str, ...]
    category_labels: dict[str, str]
    lot_size: int


MARKET_SETTINGS: dict[str, MarketSettings] = {
    "CN": MarketSettings(
        market="CN",
        market_name="A股",
        report_flag="🇨🇳",
        currency_symbol="¥",
        data_dir=config.CN_MARKET_DATA_DIR,
        analysis_output_dir="results/cn_analysis_full",
        backtest_output_dir="results/cn_backtest",
        name_cache_file="data/cn_universe/stock_names.json",
        default_batch_size=30,
        categories=("full_a",),
        category_labels={
            "full_a": "全 A 股 (Full A-share Universe)",
            "all_a": "全 A 股 (Alias)",
            "full_market": "全市场 (Alias)",
            "hs300": "沪深300 (兼容)",
            "zz500": "中证500 (兼容)",
            "zz1000": "中证1000 (兼容)",
            "all": "默认全市场",
            "full": "默认全市场",
        },
        lot_size=100,
    ),
    "US": MarketSettings(
        market="US",
        market_name="美股",
        report_flag="🇺🇸",
        currency_symbol="$",
        data_dir="data/us_market_full",
        analysis_output_dir="results/us_analysis_full",
        backtest_output_dir="results/us_backtest",
        name_cache_file="data/us_universe/stock_names.json",
        default_batch_size=25,
        categories=("full_us",),
        category_labels={
            "full_us": "全美股 (Full US Universe)",
            "all_us": "全美股 (Alias)",
            "full_market": "全市场 (Alias)",
            "large_cap": "大盘股 (S&P 500)",
            "mid_cap": "中盘股 (Mid Cap)",
            "small_cap": "小盘股 (Small Cap)",
            "all": "默认全市场",
            "full": "默认全市场",
        },
        lot_size=1,
    ),
}


def get_market_settings(market: str) -> MarketSettings:
    normalized = market.upper()
    if normalized not in MARKET_SETTINGS:
        raise ValueError(f"不支持的市场: {market!r}，可选 {sorted(MARKET_SETTINGS)}")
    return MARKET_SETTINGS[normalized]


def normalize_categories(market: str, categories: list[str] | None) -> list[str]:
    settings = get_market_settings(market)
    canonical_full = "full_a" if settings.market == "CN" else "full_us"
    if not categories:
        return list(settings.categories)

    normalized: list[str] = []
    for category in categories:
        key = str(category).strip().lower()
        if key in {"all", "full", "core", "full_a", "all_a", "full_us", "all_us", "full_market"}:
            key = canonical_full
        if key == "all":
            return list(settings.categories)
        if key not in settings.category_labels:
            raise ValueError(
                f"不支持的类别: {category!r}，{settings.market} 可选 {list(settings.category_labels)}"
            )
        if key == canonical_full and key not in normalized:
            normalized.append(key)
            continue
        if key not in normalized:
            normalized.append(key)
    return normalized


def normalize_universe(market: str, universe: str | None) -> list[str]:
    settings = get_market_settings(market)
    canonical_full = "full_a" if settings.market == "CN" else "full_us"
    key = str(universe or "").strip().lower()
    if not key:
        return list(settings.categories)
    if key in {"core", "full_a", "all_a", "full_us", "all_us", "full_market", "full", "all"}:
        return [canonical_full]
    if key in settings.category_labels:
        return [key]
    raise ValueError(
        f"不支持的 universe: {universe!r}，{settings.market} 可选 {[canonical_full, *sorted(settings.category_labels) ]}"
    )
