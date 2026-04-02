"""
统一市场流水线：检查数据新鲜度，按需下载，再执行分析与报告生成。
"""

from __future__ import annotations

import time
from typing import Any

from quant_investor.market.analyze import get_all_local_symbols, run_market_analysis
from quant_investor.market.config import get_market_settings, normalize_categories
from quant_investor.market.download import create_downloader


def _print_stage_header(index: int, title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"Stage {index}: {title}")
    print(f"{'=' * 80}")


def _print_completeness_summary(completeness: dict[str, Any]) -> None:
    print(f"目标最新交易日: {completeness.get('latest_trade_date')}")
    print(f"完整性状态: {'通过' if completeness.get('complete') else '未通过'}")
    print(f"阻塞缺口总数: {completeness.get('blocking_incomplete_count', 0)}")
    for category, payload in completeness.get("categories", {}).items():
        date_counts = payload.get("date_counts", {})
        latest_trade_date = payload.get("latest_trade_date") or completeness.get("latest_trade_date")
        latest_count = int(date_counts.get(latest_trade_date, 0)) if latest_trade_date else 0
        print(
            f"  - {category}: 最新 {latest_count}/{payload.get('expected', 0)} | "
            f"阻塞缺口 {payload.get('blocking_incomplete_count', 0)}"
        )


def _resolve_analysis_symbols(
    *,
    market: str,
    categories: list[str],
    mode: str,
    batch_size: int | None,
    fallback_symbols_by_category: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    settings = get_market_settings(market)
    resolved: dict[str, list[str]] = {}
    for category in categories:
        symbols = get_all_local_symbols(category, market=settings.market)
        if not symbols and fallback_symbols_by_category:
            symbols = list(fallback_symbols_by_category.get(category, []))
        if mode == "sample":
            scoped_batch_size = batch_size or settings.default_batch_size
            symbols = symbols[:scoped_batch_size]
        resolved[category] = symbols
    return resolved


def _print_forecast_snapshot_summary(report: dict[str, Any]) -> None:
    print(f"预测快照目标日期: {report.get('requested_as_of')}")
    print(f"预测快照最新/可用: {report.get('fresh_count', 0)}/{report.get('expected', 0)}")
    print(f"预测快照待刷新: {len(report.get('missing_symbols', [])) + len(report.get('stale_symbols', []))}")


def _build_skipped_download_result(
    *,
    reason: str,
    completeness: dict[str, Any] | None = None,
    forecast: dict[str, Any] | None = None,
    warning: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "skipped",
        "reason": reason,
        "completeness_before": completeness,
        "completeness_after": completeness,
        "forecast_before": forecast,
        "forecast_after": forecast,
        "download_results": None,
        "forecast_refresh": None,
    }
    if warning:
        result["warning"] = warning
    return result


def _build_download_context(
    downloader: Any,
    market: str,
    categories: list[str],
) -> dict[str, Any]:
    if market == "CN":
        components = downloader.load_components()
        completeness = downloader.build_completeness_report(
            components=components,
            categories=categories,
        )
        return {
            "components": components,
            "completeness": completeness,
        }

    universe = downloader.load_universe()
    required_latest_trade_date = downloader.detect_latest_available_trade_date()
    completeness = downloader.build_completeness_report(
        universe=universe,
        categories=categories,
        required_latest_trade_date=required_latest_trade_date,
    )
    return {
        "universe": universe,
        "required_latest_trade_date": required_latest_trade_date,
        "completeness": completeness,
    }


def _refresh_completeness(
    downloader: Any,
    market: str,
    categories: list[str],
    context: dict[str, Any],
) -> dict[str, Any]:
    if market == "CN":
        return downloader.build_completeness_report(
            components=context["components"],
            categories=categories,
        )

    return downloader.build_completeness_report(
        universe=context["universe"],
        categories=categories,
        required_latest_trade_date=context.get("required_latest_trade_date"),
    )


def _build_force_refresh_map(
    market: str,
    categories: list[str],
    context: dict[str, Any],
    *,
    force_download: bool,
) -> dict[str, list[str]]:
    if market == "CN":
        return {}

    universe = context["universe"]
    completeness = context["completeness"]
    refresh_map: dict[str, list[str]] = {}
    for category in categories:
        if force_download:
            refresh_map[category] = list(universe.get(category, []))
            continue
        stale_symbols = [
            item["symbol"]
            for item in completeness.get("categories", {}).get(category, {}).get("blocking_stale_symbols", [])
        ]
        if stale_symbols:
            refresh_map[category] = stale_symbols
    return refresh_map


def _is_missing_downloader_dependency(error: RuntimeError) -> bool:
    message = str(error)
    lowered = message.lower()
    return "tushare" in lowered and ("未安装" in message or "not installed" in lowered)


def _should_block_formal_analysis(download_stage: dict[str, Any]) -> bool:
    completeness = download_stage.get("completeness_after") or download_stage.get("completeness_before")
    return bool(completeness) and not bool(completeness.get("complete", False))


def _run_download_stage(
    *,
    market: str,
    categories: list[str],
    years: int,
    workers: int,
    batch_size: int | None,
    skip_download: bool,
    force_download: bool,
    max_download_rounds: int,
    mode: str,
) -> tuple[dict[str, Any], float]:
    stage_started = time.time()
    downloader_kwargs: dict[str, Any] = {
        "years": years,
        "max_workers": workers,
    }
    if batch_size is not None:
        downloader_kwargs["batch_size"] = batch_size

    try:
        downloader = create_downloader(market, **downloader_kwargs)
        context = _build_download_context(downloader, market, categories)
        completeness_before = context["completeness"]
        _print_completeness_summary(completeness_before)
        analysis_symbols_by_category = _resolve_analysis_symbols(
            market=market,
            categories=categories,
            mode=mode,
            batch_size=batch_size,
            fallback_symbols_by_category=context.get("components"),
        )
        analysis_symbols = [symbol for symbols in analysis_symbols_by_category.values() for symbol in symbols]
        forecast_before: dict[str, Any] | None = None
        if market == "CN" and hasattr(downloader, "build_forecast_snapshot_report"):
            forecast_before = downloader.build_forecast_snapshot_report(analysis_symbols)
            _print_forecast_snapshot_summary(forecast_before)
    except RuntimeError as error:
        if skip_download and _is_missing_downloader_dependency(error):
            print(f"⏭️ 已按请求跳过下载阶段；下载依赖不可用: {error}")
            return (
                _build_skipped_download_result(
                    reason="skip_download_dependency_unavailable",
                    warning=str(error),
                ),
                time.time() - stage_started,
            )
        raise

    if skip_download:
        print("⏭️ 已按请求跳过下载阶段。")
        return (
            _build_skipped_download_result(
                reason="skip_download",
                completeness=completeness_before,
                forecast=forecast_before,
            ),
            time.time() - stage_started,
        )

    needs_forecast_refresh = bool(forecast_before and forecast_before.get("refresh_needed", False))
    needs_ohlcv_download = force_download or not completeness_before.get("complete", False)

    download_results = None
    if needs_ohlcv_download:
        if market == "CN":
            download_results = downloader.download_all(
                components=context["components"],
                max_rounds=max_download_rounds,
                fail_on_incomplete=False,
                categories=categories,
            )
        else:
            download_results = downloader.download_all(
                universe=context["universe"],
                categories=categories,
                force_refresh_by_category=_build_force_refresh_map(
                    market,
                    categories,
                    context,
                    force_download=force_download,
                ),
            )

    forecast_after = forecast_before
    forecast_refresh_results = None
    if market == "CN" and hasattr(downloader, "refresh_forecast_snapshots") and (
        force_download or needs_forecast_refresh
    ):
        forecast_refresh_results = downloader.refresh_forecast_snapshots(
            analysis_symbols,
            force_refresh=force_download,
        )
        forecast_after = forecast_refresh_results.get("after", forecast_before)
        print("📌 预测快照刷新后的复核")
        _print_forecast_snapshot_summary(forecast_after or {})

    if not needs_ohlcv_download and not force_download and not needs_forecast_refresh:
        print("✅ 数据已是最新，跳过下载。")
        return (
            {
                "status": "up_to_date",
                "reason": "already_latest",
                "completeness_before": completeness_before,
                "completeness_after": completeness_before,
                "forecast_before": forecast_before,
                "forecast_after": forecast_before,
                "download_results": None,
                "forecast_refresh": None,
            },
            time.time() - stage_started,
        )

    completeness_after = _refresh_completeness(downloader, market, categories, context)
    print("📌 下载后的完整性复核")
    _print_completeness_summary(completeness_after)

    return (
        {
            "status": (
                "forced_download"
                if force_download
                else "downloaded"
                if needs_ohlcv_download
                else "refreshed_snapshots"
            ),
            "reason": (
                "force_download"
                if force_download
                else "stale_or_incomplete"
                if needs_ohlcv_download
                else "forecast_snapshots_refreshed"
            ),
            "completeness_before": completeness_before,
            "completeness_after": completeness_after,
            "forecast_before": forecast_before,
            "forecast_after": forecast_after,
            "download_results": download_results,
            "forecast_refresh": forecast_refresh_results,
        },
        time.time() - stage_started,
    )


def run_unified_pipeline(
    market: str,
    categories: list[str] | None = None,
    mode: str = "batch",
    batch_size: int | None = None,
    total_capital: float = 1_000_000,
    top_k: int = 12,
    years: int = 3,
    workers: int = 4,
    max_download_rounds: int = 2,
    skip_download: bool = False,
    force_download: bool = False,
    verbose: bool = True,
    enable_agent_layer: bool = True,
    agent_model: str = "",
    master_model: str = "",
    agent_timeout: float = 15.0,
    master_timeout: float = 30.0,
) -> dict[str, Any]:
    settings = get_market_settings(market)
    selected_categories = normalize_categories(settings.market, categories)
    total_started = time.time()

    _print_stage_header(1, "数据新鲜度检查与条件下载")
    download_stage, download_duration = _run_download_stage(
        market=settings.market,
        categories=selected_categories,
        years=years,
        workers=workers,
        batch_size=batch_size,
        skip_download=skip_download,
        force_download=force_download,
        max_download_rounds=max_download_rounds,
        mode=mode,
    )

    if _should_block_formal_analysis(download_stage):
        _print_stage_header(2, "正式分析已阻塞")
        print("❌ 数据完整性校验未通过，已停止正式分析。")
        completeness = download_stage.get("completeness_after") or download_stage.get("completeness_before")
        _print_completeness_summary(completeness or {})
        return {
            "market": settings.market,
            "categories": selected_categories,
            "download": download_stage,
            "analysis": {},
            "reports": {},
            "blocked": {
                "reason": "data_incomplete",
                "completeness": completeness,
            },
            "timing": {
                "download_seconds": download_duration,
                "analysis_seconds": 0.0,
                "total_seconds": time.time() - total_started,
            },
        }

    _print_stage_header(2, "全市场分析与报告生成")
    analysis_started = time.time()
    analysis_output = run_market_analysis(
        market=settings.market,
        mode=mode,
        categories=selected_categories,
        batch_size=batch_size,
        total_capital=total_capital,
        top_k=top_k,
        verbose=verbose,
        enable_agent_layer=enable_agent_layer,
        agent_model=agent_model,
        master_model=master_model,
        agent_timeout=agent_timeout,
        master_timeout=master_timeout,
    )
    analysis_duration = time.time() - analysis_started

    _print_stage_header(3, "流水线完成")
    print(f"下载阶段: {download_duration:.2f}s")
    print(f"分析与报告阶段: {analysis_duration:.2f}s")
    print(f"总耗时: {time.time() - total_started:.2f}s")

    return {
        "market": settings.market,
        "categories": selected_categories,
        "download": download_stage,
        "analysis": analysis_output["results"],
        "reports": analysis_output["reports"],
        "timing": {
            "download_seconds": download_duration,
            "analysis_seconds": analysis_duration,
            "total_seconds": time.time() - total_started,
        },
    }
