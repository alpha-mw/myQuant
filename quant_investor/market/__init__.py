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
    "run_materialize_features",
    "run_materialize_serving",
    "run_storage_diff",
    "run_storage_reactivate_snapshot",
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
        "run_materialize_features",
        "run_materialize_serving",
        "run_storage_diff",
        "run_storage_reactivate_snapshot",
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
