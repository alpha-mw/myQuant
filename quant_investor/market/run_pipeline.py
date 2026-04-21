"""
统一市场流水线：构建本地数据快照，再执行分析与报告生成。
"""

from __future__ import annotations

import time
from typing import Any

from quant_investor.config import config
from quant_investor.llm_provider_priority import resolve_runtime_role_models
from quant_investor.market.analyze import run_market_analysis
from quant_investor.market.config import get_market_settings, normalize_categories, normalize_universe
from quant_investor.market.data_snapshot import build_market_data_snapshot


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


def _run_download_stage(
    *,
    market: str,
    categories: list[str],
    years: int,
    workers: int,
    batch_size: int | None,
    skip_download: bool,
    skip_stage1: bool,
    force_download: bool,
    max_download_rounds: int,
    mode: str,
) -> tuple[dict[str, Any], float]:
    stage_started = time.time()
    warnings: list[str] = []
    if skip_stage1:
        print("ℹ️ `skip_stage1` 已兼容保留；分析路径当前始终执行本地数据快照披露。")
        warnings.append("skip_stage1_ignored")
    if skip_download:
        print("ℹ️ `skip_download` 已兼容保留；分析路径当前不会自动补数。")
        warnings.append("skip_download_ignored")
    if force_download:
        print("ℹ️ `force_download` 已兼容保留；需要补数请改用 `quant-investor market maintain`。")
        warnings.append("force_download_ignored")

    data_snapshot = build_market_data_snapshot(
        market=market,
        categories=categories,
    )
    print(f"本地数据最新日期: {data_snapshot.get('local_latest_trade_date') or '未知'}")
    print(f"快照摘要: {data_snapshot.get('summary_text', '')}")
    result: dict[str, Any] = {
        "status": "snapshot_only",
        "reason": "analysis_uses_local_data_snapshot",
        "data_snapshot": data_snapshot,
    }
    if warnings:
        result["warning"] = warnings[0] if len(warnings) == 1 else ",".join(warnings)
    return result, time.time() - stage_started


def run_unified_pipeline(
    market: str,
    universe: str | None = None,
    categories: list[str] | None = None,
    mode: str = "batch",
    batch_size: int | None = None,
    total_capital: float = 1_000_000,
    top_k: int = 12,
    years: int = 3,
    workers: int = 4,
    max_download_rounds: int = 2,
    skip_stage1: bool = False,
    skip_download: bool = False,
    force_download: bool = False,
    verbose: bool = True,
    enable_agent_layer: bool = True,
    review_model_priority: list[str] | None = None,
    agent_model: str = "",
    agent_fallback_model: str = "",
    master_model: str = "",
    master_fallback_model: str = "",
    master_reasoning_effort: str = "high",
    agent_timeout: float = config.DEFAULT_AGENT_TIMEOUT_SECONDS,
    master_timeout: float = config.DEFAULT_MASTER_TIMEOUT_SECONDS,
    funnel_profile: str = config.FUNNEL_PROFILE,
    max_candidates: int = config.FUNNEL_MAX_CANDIDATES,
    trend_windows: list[int] | tuple[int, ...] | None = None,
    volume_spike_threshold: float = config.FUNNEL_VOLUME_SPIKE_THRESHOLD,
    breakout_distance_pct: float = config.FUNNEL_BREAKOUT_DISTANCE_PCT,
    recall_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_market_settings(market)
    selected_categories = (
        normalize_universe(settings.market, universe)
        if universe is not None
        else normalize_categories(settings.market, categories)
    )
    total_started = time.time()

    _print_stage_header(1, "数据快照与来源披露")
    download_stage, download_duration = _run_download_stage(
        market=settings.market,
        categories=selected_categories,
        years=years,
        workers=workers,
        batch_size=batch_size,
        skip_stage1=skip_stage1,
        skip_download=skip_download,
        force_download=force_download,
        max_download_rounds=max_download_rounds,
        mode=mode,
    )

    _print_stage_header(2, "全市场分析与报告生成")
    analysis_started = time.time()
    branch_config, master_config = resolve_runtime_role_models(
        review_model_priority=review_model_priority,
        agent_model=agent_model,
        agent_fallback_model=agent_fallback_model,
        master_model=master_model,
        master_fallback_model=master_fallback_model,
    )
    analysis_output = run_market_analysis(
        market=settings.market,
        universe=universe,
        mode=mode,
        categories=selected_categories,
        batch_size=batch_size,
        total_capital=total_capital,
        top_k=top_k,
        verbose=verbose,
        enable_agent_layer=enable_agent_layer,
        review_model_priority=list(review_model_priority or []),
        agent_model=branch_config.primary_model,
        agent_fallback_model=branch_config.fallback_model,
        master_model=master_config.primary_model,
        master_fallback_model=master_config.fallback_model,
        master_reasoning_effort=master_reasoning_effort,
        agent_timeout=agent_timeout,
        master_timeout=master_timeout,
        funnel_profile=funnel_profile,
        max_candidates=max_candidates,
        trend_windows=list(trend_windows or config.FUNNEL_TREND_WINDOWS),
        volume_spike_threshold=volume_spike_threshold,
        breakout_distance_pct=breakout_distance_pct,
        recall_context=recall_context,
        data_snapshot=download_stage.get("data_snapshot"),
    )
    analysis_duration = time.time() - analysis_started

    _print_stage_header(3, "流水线完成")
    print(f"下载阶段: {download_duration:.2f}s")
    print(f"分析与报告阶段: {analysis_duration:.2f}s")
    print(f"总耗时: {time.time() - total_started:.2f}s")

    return {
        "market": settings.market,
        "categories": selected_categories,
        "universe": universe or (selected_categories[0] if len(selected_categories) == 1 else "custom"),
        "download": download_stage,
        "analysis": analysis_output["results"],
        "reports": analysis_output["reports"],
        "analysis_meta": analysis_output.get("analysis_meta", {}),
        "timing": {
            "download_seconds": download_duration,
            "analysis_seconds": analysis_duration,
            "total_seconds": time.time() - total_started,
        },
    }
