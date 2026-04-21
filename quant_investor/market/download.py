"""
统一市场下载入口。
"""

from __future__ import annotations

from typing import Any

from quant_investor.config import config
from quant_investor.market.config import get_market_settings, normalize_categories
from quant_investor.market.download_cn import CNFullMarketDownloader
from quant_investor.market.download_us import FullMarketDownloader as USFullMarketDownloader


class MarketDownloader:
    """按市场选择具体下载 provider 的统一外壳。"""

    def __init__(self, market: str, **kwargs: Any) -> None:
        settings = get_market_settings(market)
        data_dir = kwargs.pop("data_dir", settings.data_dir)
        self.market = settings.market
        if self.market == "CN":
            self._impl = CNFullMarketDownloader(data_dir=data_dir, **kwargs)
        else:
            self._impl = USFullMarketDownloader(data_dir=data_dir, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


def create_downloader(market: str, **kwargs: Any) -> MarketDownloader:
    return MarketDownloader(market, **kwargs)


def run_market_maintenance(
    market: str,
    categories: list[str] | None = None,
    max_rounds: int = 1,
    fail_on_incomplete: bool = False,
    allowed_stale_symbols: list[str] | None = None,
    deprecated_alias: bool = False,
    **kwargs: Any,
) -> Any:
    settings = get_market_settings(market)
    selected_categories = normalize_categories(settings.market, categories)
    if deprecated_alias:
        print("⚠️ `quant-investor market download` 已兼容保留；请改用 `quant-investor market maintain`。")

    if settings.market == "CN":
        if not str(getattr(config, "TUSHARE_TOKEN", "") or "").strip():
            raise RuntimeError("CN maintenance requires TUSHARE_TOKEN；请先配置主 Tushare Pro Token。")
        if not str(getattr(config, "TUSHARE_URL", "") or "").strip():
            raise RuntimeError("CN maintenance requires TUSHARE_URL；请先配置高积分 Tushare Pro URL。")

    downloader = create_downloader(settings.market, **kwargs)

    if settings.market == "CN":
        components = downloader.load_components()
        if getattr(downloader, "pro", None) is None:
            raise RuntimeError("CN maintenance 无法初始化 Tushare Pro；请检查主 Token 和高积分 URL。")
        download_results = downloader.download_all(
            components=components,
            max_rounds=max_rounds,
            fail_on_incomplete=fail_on_incomplete,
            allowed_stale_symbols=allowed_stale_symbols,
            categories=selected_categories,
        )
        completeness = downloader.build_completeness_report(
            components=components,
            allowed_stale_symbols=allowed_stale_symbols,
            categories=selected_categories,
        )
        downloader._print_completeness_summary(completeness)
        return {
            "status": "maintained",
            "download_results": download_results,
            "completeness": completeness,
            "categories": selected_categories,
        }

    universe = downloader.load_universe()
    scoped_universe = {
        key: value
        for key, value in universe.items()
        if key in selected_categories or key == "stats"
    }
    return {
        "status": "maintained",
        "download_results": downloader.download_all(scoped_universe),
        "categories": selected_categories,
    }


def run_download(
    market: str,
    categories: list[str] | None = None,
    check_complete: bool = False,
    max_rounds: int = 1,
    fail_on_incomplete: bool = False,
    allowed_stale_symbols: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    settings = get_market_settings(market)
    selected_categories = normalize_categories(settings.market, categories)
    downloader = create_downloader(settings.market, **kwargs)

    if check_complete and settings.market == "CN":
        components = downloader.load_components()
        completeness = downloader.build_completeness_report(
            components=components,
            allowed_stale_symbols=allowed_stale_symbols,
            categories=selected_categories,
        )
        downloader._print_completeness_summary(completeness)
        return completeness

    return run_market_maintenance(
        market=market,
        categories=categories,
        max_rounds=max_rounds,
        fail_on_incomplete=fail_on_incomplete,
        allowed_stale_symbols=allowed_stale_symbols,
        deprecated_alias=True,
        **kwargs,
    )
