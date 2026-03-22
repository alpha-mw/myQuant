"""
市场级共享配置。
"""

from __future__ import annotations

from dataclasses import dataclass


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
        data_dir="data/cn_market_full",
        analysis_output_dir="results/cn_analysis_full",
        backtest_output_dir="results/cn_backtest",
        name_cache_file="data/cn_universe/stock_names.json",
        default_batch_size=30,
        categories=("hs300", "zz500", "zz1000"),
        category_labels={
            "hs300": "沪深300 (大盘股)",
            "zz500": "中证500 (中盘股)",
            "zz1000": "中证1000 (小盘股)",
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
        categories=("large_cap", "mid_cap", "small_cap"),
        category_labels={
            "large_cap": "大盘股 (S&P 500)",
            "mid_cap": "中盘股 (Mid Cap)",
            "small_cap": "小盘股 (Small Cap)",
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
    if not categories:
        return list(settings.categories)

    normalized: list[str] = []
    for category in categories:
        if category == "all":
            return list(settings.categories)
        if category not in settings.category_labels:
            raise ValueError(
                f"不支持的类别: {category!r}，{settings.market} 可选 {list(settings.category_labels)}"
            )
        if category not in normalized:
            normalized.append(category)
    return normalized
