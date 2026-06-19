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
    "run_staged_maintenance",
    "load_stock_names",
    "get_stock_name",
    "category_name",
    "analyze_batch",
    "analyze_category_full",
    "build_full_market_trade_plan",
    "generate_full_report",
    "run_market_analysis",
    "run_market_backtest",
    "run_materialize_features",
    "run_materialize_serving",
    "run_storage_diff",
    "run_storage_validate",
    "run_storage_validate_clean",
]


def __getattr__(name: str) -> Any:
    if name in {
        "get_stock_name",
        "load_stock_names",
    }:
        from quant_investor.market import name_map as name_map_mod

        return getattr(name_map_mod, name)
    if name in {
        "analyze_batch",
        "analyze_category_full",
    }:
        from quant_investor.market import legacy_batch_analysis as batch_mod

        return getattr(batch_mod, name)
    if name in {
        "build_full_market_trade_plan",
        "category_name",
        "generate_full_report",
    }:
        from quant_investor.market import full_report as report_mod

        return getattr(report_mod, name)
    if name in {
        "run_market_analysis",
    }:
        from quant_investor.market import analyze as analyze_mod

        return getattr(analyze_mod, name)
    if name in {"run_market_backtest"}:
        from quant_investor.market import backtest as backtest_mod

        return getattr(backtest_mod, name)
    if name in {
        "run_materialize_features",
        "run_materialize_serving",
        "run_storage_diff",
        "run_storage_validate",
        "run_storage_validate_clean",
    }:
        from quant_investor.market import market_data_store as store_mod

        return getattr(store_mod, name)
    if name in {
        "CNFullMarketDownloader",
        "USFullMarketDownloader",
        "MarketDownloader",
        "create_downloader",
        "run_download",
        "run_market_maintenance",
        "run_staged_maintenance",
    }:
        from quant_investor.market import download as download_mod

        if name == "run_staged_maintenance":
            from quant_investor.market import staged_maintenance as staged_mod

            return getattr(staged_mod, name)
        return getattr(download_mod, name)
    raise AttributeError(name)
