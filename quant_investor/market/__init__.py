"""
Unified market workflow public surface with lazy imports.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CNFullMarketDownloader",
    "USFullMarketDownloader",
    "MarketDownloader",
    "create_downloader",
    "run_download",
    "run_market_maintenance",
    "load_stock_names",
    "get_stock_name",
    "category_name",
    "analyze_batch",
    "analyze_category_full",
    "build_full_market_trade_plan",
    "generate_full_report",
    "run_market_analysis",
    "run_market_backtest",
]


def __getattr__(name: str) -> Any:
    if name in {
        "analyze_batch",
        "analyze_category_full",
        "build_full_market_trade_plan",
        "category_name",
        "generate_full_report",
        "get_stock_name",
        "load_stock_names",
        "run_market_analysis",
    }:
        from quant_investor.market import analyze as analyze_mod

        return getattr(analyze_mod, name)
    if name in {"run_market_backtest"}:
        from quant_investor.market import backtest as backtest_mod

        return getattr(backtest_mod, name)
    if name in {
        "CNFullMarketDownloader",
        "USFullMarketDownloader",
        "MarketDownloader",
        "create_downloader",
        "run_download",
        "run_market_maintenance",
    }:
        from quant_investor.market import download as download_mod

        return getattr(download_mod, name)
    raise AttributeError(name)
