from __future__ import annotations

import os
import time
from typing import Any

from quant_investor.automation import daily_runner as _runner
from quant_investor.config import config as runtime_config
from quant_investor.research_run_config import ResolvedReviewModels


def _maintenance_categories(config: dict[str, Any]) -> list[str]:
    categories = config.get("categories")
    if isinstance(categories, (list, tuple)):
        normalized = [str(item or "").strip() for item in categories if str(item or "").strip()]
        if normalized:
            return normalized
    universe = str(config.get("universe") or "full_a").strip()
    return [universe or "full_a"]


def _run_automation_data_update_preflight(config: dict[str, Any]) -> dict[str, Any]:
    market = str(config.get("market") or "CN").strip().upper()
    if market != "CN":
        return {
            "status": "skipped",
            "maintenance_status": "skipped_non_cn",
            "market": market,
            "reason": "staged_batch_maintenance_cn_only",
        }

    try:
        return _runner.run_staged_maintenance(
            market="CN",
            categories=_maintenance_categories(config),
            batch_size=max(1, int(config.get("maintenance_batch_size", 200))),
            max_batches_per_run=max(0, int(config.get("maintenance_max_batches_per_run", 200))),
            min_symbol_success_rate=float(config.get("maintenance_min_symbol_success_rate", 0.95)),
            target_date=str(config.get("maintenance_target_date", "auto") or "auto"),
            daily_window=bool(config.get("maintenance_daily_window", True)),
            resume=True,
            fail_on_incomplete=False,
            allowed_stale_symbols=list(config.get("allowed_stale_symbols", []) or []),
            years=int(config.get("years", 3)),
            max_workers=int(config.get("workers", 4)),
        )
    except Exception as exc:
        _runner.log.warning("staged batch 数据维护失败，继续使用本地快照: %s", exc, exc_info=True)
        return {
            "status": "failed_non_blocking",
            "maintenance_status": "failed_non_blocking",
            "market": market,
            "non_blocking": True,
            "error": str(exc),
        }


class AnalysisRunner:
    """包装 run_unified_pipeline，执行全量 A 股分析。"""

    def run(self, config: dict[str, Any], recall_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """执行全量市场分析，返回 pipeline 结果字典。"""
        config = _runner._normalize_runtime_config(config)
        # 确保工作目录正确（pipeline 依赖相对路径）
        os.chdir(_runner.ROOT)

        from quant_investor.market.run_pipeline import run_unified_pipeline
        from quant_investor.model_roles import resolve_model_role

        review_models = ResolvedReviewModels.from_mapping(config)
        branch_resolution = resolve_model_role(
            role="branch",
            primary_model=review_models.branch_primary_model,
            fallback_model=review_models.branch_fallback_model,
        )
        master_resolution = resolve_model_role(
            role="master",
            primary_model=review_models.master_primary_model,
            fallback_model=review_models.master_fallback_model,
        )
        config = review_models.apply_to_mapping(config)
        config["agent_model"] = branch_resolution.resolved_model
        config["master_model"] = master_resolution.resolved_model
        config["agent_fallback_model"] = branch_resolution.fallback_model
        config["master_fallback_model"] = master_resolution.fallback_model
        config.setdefault("universe", "full_a")
        config.setdefault("skip_stage1", bool(config.get("skip_data_check", False)))
        config["model_role_resolution"] = {
            "branch": branch_resolution.to_dict(),
            "master": master_resolution.to_dict(),
        }

        os.environ.setdefault(
            "FUNNEL_MAX_CANDIDATES",
            str(config.get("funnel_max_candidates", runtime_config.FUNNEL_MAX_CANDIDATES)),
        )
        os.environ.setdefault("FUNNEL_PROFILE", str(config.get("funnel_profile", runtime_config.FUNNEL_PROFILE)))
        os.environ.setdefault(
            "FUNNEL_TREND_WINDOWS",
            ",".join(str(int(item)) for item in config.get("trend_windows", runtime_config.FUNNEL_TREND_WINDOWS)),
        )
        os.environ.setdefault(
            "FUNNEL_VOLUME_SPIKE_THRESHOLD",
            str(config.get("volume_spike_threshold", runtime_config.FUNNEL_VOLUME_SPIKE_THRESHOLD)),
        )
        os.environ.setdefault(
            "FUNNEL_BREAKOUT_DISTANCE_PCT",
            str(config.get("breakout_distance_pct", runtime_config.FUNNEL_BREAKOUT_DISTANCE_PCT)),
        )
        os.environ.setdefault("BAYESIAN_SHORTLIST_SIZE", str(config.get("bayesian_shortlist_size", 50)))
        os.environ.setdefault("CN_FRESHNESS_MODE", config.get("freshness_mode", "stable"))

        _runner.log.info(
            "开始分析 | market=%s | universe=%s | review_model_priority=%s | branch_model=%s%s | master_model=%s%s | master_reasoning_effort=%s | top_k=%s | skip_stage1=%s",
            config["market"],
            config.get("universe", "full_a"),
            " -> ".join(config["review_model_priority"]),
            config["agent_model"] or "(默认)",
            " [fallback]" if branch_resolution.fallback_used else "",
            config["master_model"] or "(默认)",
            " [fallback]" if master_resolution.fallback_used else "",
            config.get("master_reasoning_effort", "") or "(默认)",
            config["top_k"],
            bool(config.get("skip_stage1", False)),
        )
        if branch_resolution.fallback_used:
            _runner.log.warning(
                "branch model fallback activated: primary=%s fallback=%s reason=%s",
                branch_resolution.primary_model,
                branch_resolution.fallback_model,
                branch_resolution.fallback_reason,
            )
        if master_resolution.fallback_used:
            _runner.log.warning(
                "master model fallback activated: primary=%s fallback=%s reason=%s",
                master_resolution.primary_model,
                master_resolution.fallback_model,
                master_resolution.fallback_reason,
            )
        started = time.time()

        def _call_pipeline(skip_dl: bool) -> dict[str, Any]:
            return run_unified_pipeline(
                market=config["market"],
                universe=config.get("universe", "full_a"),
                mode="batch",
                skip_stage1=bool(config.get("skip_stage1", False)),
                skip_download=skip_dl,
                total_capital=config["total_capital"],
                top_k=config["top_k"],
                shortlist_size=max(1, int(config.get("bayesian_shortlist_size", 50))),
                years=config["years"],
                workers=config["workers"],
                enable_agent_layer=config["enable_agent_layer"],
                review_model_priority=config["review_model_priority"],
                agent_model=config["agent_model"],
                agent_fallback_model=config["agent_fallback_model"],
                master_model=config["master_model"],
                master_fallback_model=config["master_fallback_model"],
                master_reasoning_effort=config.get("master_reasoning_effort", ""),
                agent_timeout=config["agent_timeout"],
                master_timeout=config["master_timeout"],
                recall_context=dict(recall_context or {}),
                verbose=True,
            )

        automation_data_update: dict[str, Any] | None = None
        if not bool(config["skip_download"]):
            _runner.log.info(
                "执行 staged batch 数据维护 | market=%s | categories=%s | batch_size=%s | max_batches=%s",
                config["market"],
                _maintenance_categories(config),
                config.get("maintenance_batch_size", 200),
                config.get("maintenance_max_batches_per_run", 200),
            )
            automation_data_update = _run_automation_data_update_preflight(config)

        try:
            result = _call_pipeline(skip_dl=True if automation_data_update is not None else bool(config["skip_download"]))
        except RuntimeError as exc:
            msg = str(exc)
            if "tushare" in msg and ("未安装" in msg or "not installed" in msg):
                _runner.log.warning(
                    "tushare 下载阶段失败（%s），自动切换到 skip_download=True 使用本地数据...",
                    msg,
                )
                result = _call_pipeline(skip_dl=True)
            else:
                raise

        if automation_data_update is not None:
            result["automation_data_update"] = automation_data_update
            download_stage = dict(result.get("download") or {})
            download_stage["automation_data_update"] = automation_data_update
            result["download"] = download_stage

        elapsed = time.time() - started
        _runner.log.info("分析完成，耗时 %.0f 秒（%.1f 分钟）", elapsed, elapsed / 60)
        return result
