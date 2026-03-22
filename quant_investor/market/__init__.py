"""
统一市场工作流入口。
"""

from quant_investor.market.analyze import (
    analyze_batch,
    analyze_category_full,
    build_full_market_trade_plan,
    category_name,
    generate_full_report,
    get_stock_name,
    load_stock_names,
    run_market_analysis,
)
from quant_investor.market.backtest import run_market_backtest
from quant_investor.market.download import (
    CNFullMarketDownloader,
    MarketDownloader,
    USFullMarketDownloader,
    create_downloader,
    run_download,
)

__all__ = [
    "CNFullMarketDownloader",
    "USFullMarketDownloader",
    "MarketDownloader",
    "create_downloader",
    "run_download",
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
