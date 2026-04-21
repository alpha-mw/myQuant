"""
Quant-Investor single mainline.

Current mainline = deterministic research core + structured review layer +
single control chain (RiskGuard -> ICCoordinator -> PortfolioConstructor -> NarratorAgent).
"""

from __future__ import annotations

import time
from typing import Any

from quant_investor.model_roles import resolve_model_role
from quant_investor.config import config
from quant_investor.llm_provider_priority import resolve_runtime_role_models
from quant_investor.llm_gateway import (
    build_effective_usage_summary,
    current_usage_session_id,
    end_usage_session,
    resolve_default_model,
    snapshot_usage,
    start_usage_session,
)
from quant_investor.logger import get_logger
from quant_investor.market.data_snapshot import build_market_data_snapshot
from quant_investor.pipeline.result_types import QuantInvestorPipelineResult
from quant_investor.pipeline.result_builder import build_pipeline_result_from_dag

_logger = get_logger("QuantInvestor")


def _execute_market_dag(**kwargs: Any) -> dict[str, Any]:
    from quant_investor.market.dag_executor import execute_market_dag

    return execute_market_dag(**kwargs)

_CONVICTION_SCORE = {
    "strong_buy": 0.80,
    "buy": 0.35,
    "neutral": 0.0,
    "sell": -0.35,
    "strong_sell": -0.80,
}

class QuantInvestor:
    """Single supported public mainline."""

    def __init__(
        self,
        stock_pool: list[str],
        market: str = "CN",
        lookback_years: float = 1.0,
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        enable_macro: bool = True,
        enable_backtest: bool = False,
        enable_alpha_mining: bool = True,
        enable_quant: bool = True,
        enable_kline: bool = True,
        enable_fundamental: bool = True,
        enable_intelligence: bool = True,
        kline_backend: str = "hybrid",
        allow_synthetic_for_research: bool = False,
        enable_document_semantics: bool = True,
        verbose: bool = True,
        enable_kronos: bool | None = None,
        enable_agent_layer: bool = True,
        review_model_priority: list[str] | None = None,
        agent_model: str = "",
        master_model: str = "",
        agent_fallback_model: str = "",
        master_fallback_model: str = "",
        master_reasoning_effort: str = "",
        agent_timeout: float = config.DEFAULT_AGENT_TIMEOUT_SECONDS,
        master_timeout: float = config.DEFAULT_MASTER_TIMEOUT_SECONDS,
        agent_total_timeout: float = config.DEFAULT_AGENT_TOTAL_TIMEOUT_SECONDS,
        universe_key: str = "full_a",
        funnel_profile: str = config.FUNNEL_PROFILE,
        max_candidates: int = config.FUNNEL_MAX_CANDIDATES,
        trend_windows: list[int] | tuple[int, ...] | None = None,
        volume_spike_threshold: float = config.FUNNEL_VOLUME_SPIKE_THRESHOLD,
        breakout_distance_pct: float = config.FUNNEL_BREAKOUT_DISTANCE_PCT,
        sector_bucket_limit: int = config.FUNNEL_SECTOR_BUCKET_LIMIT,
        recall_context: dict[str, Any] | None = None,
    ) -> None:
        self.stock_pool = stock_pool
        self.market = market
        self.lookback_years = lookback_years
        self.total_capital = total_capital
        self.risk_level = risk_level
        self.enable_macro = enable_macro
        self.enable_backtest = enable_backtest
        self.enable_alpha_mining = enable_alpha_mining
        self.enable_quant = enable_quant
        self.enable_kline = enable_kline if enable_kronos is None else enable_kronos
        self.kline_backend = kline_backend
        self.enable_fundamental = enable_fundamental
        self.enable_intelligence = enable_intelligence
        self.allow_synthetic_for_research = allow_synthetic_for_research
        self.enable_document_semantics = enable_document_semantics
        self.verbose = verbose
        self.enable_agent_layer = enable_agent_layer
        branch_config, master_config = resolve_runtime_role_models(
            review_model_priority=review_model_priority,
            agent_model=agent_model,
            agent_fallback_model=agent_fallback_model,
            master_model=master_model,
            master_fallback_model=master_fallback_model,
        )
        self.primary_agent_model = branch_config.primary_model or resolve_default_model(preferred_model="")
        self.primary_master_model = master_config.primary_model or resolve_default_model(preferred_model="")
        self.agent_fallback_model = branch_config.fallback_model
        self.master_fallback_model = master_config.fallback_model
        self.review_model_priority = {
            "branch": list(branch_config.candidate_models),
            "master": list(master_config.candidate_models),
            "branch_source": branch_config.source,
            "master_source": master_config.source,
        }
        self.agent_resolution = resolve_model_role(
            role="branch",
            primary_model=self.primary_agent_model,
            fallback_model=self.agent_fallback_model,
        )
        self.master_resolution = resolve_model_role(
            role="master",
            primary_model=self.primary_master_model,
            fallback_model=self.master_fallback_model,
        )
        self.agent_model = self.agent_resolution.resolved_model
        self.master_model = self.master_resolution.resolved_model
        self.master_reasoning_effort = str(master_reasoning_effort or "").strip() or "high"
        self.agent_timeout = agent_timeout
        self.master_timeout = master_timeout
        self.agent_total_timeout = agent_total_timeout
        self.universe_key = str(universe_key or "").strip() or "full_a"
        self.funnel_profile = str(funnel_profile or config.FUNNEL_PROFILE).strip().lower() or config.FUNNEL_PROFILE
        self.max_candidates = max(1, int(max_candidates or config.FUNNEL_MAX_CANDIDATES))
        windows = list(trend_windows or config.FUNNEL_TREND_WINDOWS or (20, 60, 120))
        self.trend_windows = [int(item) for item in windows if int(item) > 0][:3] or [20, 60, 120]
        self.volume_spike_threshold = float(volume_spike_threshold or config.FUNNEL_VOLUME_SPIKE_THRESHOLD)
        self.breakout_distance_pct = float(breakout_distance_pct or config.FUNNEL_BREAKOUT_DISTANCE_PCT)
        self.sector_bucket_limit = max(0, int(sector_bucket_limit if sector_bucket_limit is not None else config.FUNNEL_SECTOR_BUCKET_LIMIT))
        self.recall_context = dict(recall_context or {})
        self.result = QuantInvestorPipelineResult()

    def _log(self, message: str) -> None:
        self.result.execution_log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if self.verbose:
            _logger.info(message)

    def run(self) -> QuantInvestorPipelineResult:
        t0 = time.time()
        managed_session = None
        if current_usage_session_id() is None:
            managed_session = start_usage_session(label="mainline")

        try:
            self._log("=" * 60)
            self._log("🚀 Quant-Investor 主线启动")
            self._log(f"标的: {self.stock_pool}")
            self._log(f"市场: {self.market}  universe={self.universe_key}  资金: ¥{self.total_capital:,.0f}")
            self._log(
                "Selection Profile: profile=%s max_candidates=%d trend_windows=%s volume_spike_threshold=%.2f breakout_distance_pct=%.2f sector_bucket_limit=%d"
                % (
                    self.funnel_profile,
                    self.max_candidates,
                    self.trend_windows,
                    self.volume_spike_threshold,
                    self.breakout_distance_pct,
                    self.sector_bucket_limit,
                )
            )
            self._log(
                "Review Layer: %s | branch_model=%s | master_model=%s | master_reasoning_effort=%s"
                % (
                    "启用" if self.enable_agent_layer else "禁用",
                    self.agent_model,
                    self.master_model,
                    self.master_reasoning_effort,
                )
            )
            if self.agent_resolution.fallback_used:
                self._log(
                    "branch model fallback: primary=%s fallback=%s reason=%s"
                    % (
                        self.agent_resolution.primary_model,
                        self.agent_resolution.fallback_model,
                        self.agent_resolution.fallback_reason,
                    )
                )
            if self.master_resolution.fallback_used:
                self._log(
                    "master model fallback: primary=%s fallback=%s reason=%s"
                    % (
                        self.master_resolution.primary_model,
                        self.master_resolution.fallback_model,
                        self.master_resolution.fallback_reason,
                    )
                )
            self._log("=" * 60)

            if hasattr(config, "validate_runtime"):
                validation = config.validate_runtime(
                    market=self.market,
                    enable_agent_layer=self.enable_agent_layer,
                    agent_model=self.agent_model,
                    master_model=self.master_model,
                    master_reasoning_effort=self.master_reasoning_effort,
                    kline_backend=self.kline_backend,
                )
                for message in validation["errors"]:
                    self._log(f"⚠️ {message}")
                for message in validation["warnings"]:
                    self._log(f"⚠️ {message}")
            data_snapshot = build_market_data_snapshot(
                market=self.market,
                universe=self.universe_key,
                requested_symbols=list(self.stock_pool),
            )
            missing_requested = list(data_snapshot.get("missing_requested_symbols", []) or [])
            unreadable_requested = list(data_snapshot.get("unreadable_requested_symbols", []) or [])
            if missing_requested or unreadable_requested:
                missing_text = ", ".join(missing_requested + unreadable_requested)
                raise ValueError(f"本地数据不可用于研究：{missing_text}")
            if data_snapshot.get("summary_text"):
                self._log(f"数据快照: {data_snapshot['summary_text']}")
            dag_artifacts = _execute_market_dag(
                market=self.market,
                symbols=list(self.stock_pool),
                universe=self.universe_key,
                mode="sample",
                batch_size=len(self.stock_pool),
                total_capital=self.total_capital,
                top_k=max(1, min(len(self.stock_pool), int(getattr(config, "BAYESIAN_SHORTLIST_SIZE", 20)))),
                verbose=self.verbose,
                enable_agent_layer=self.enable_agent_layer,
                review_model_priority=list(self.review_model_priority.get("branch", [])),
                agent_model=self.primary_agent_model,
                agent_fallback_model=self.agent_fallback_model,
                master_model=self.primary_master_model,
                master_fallback_model=self.master_fallback_model,
                master_reasoning_effort=self.master_reasoning_effort,
                agent_timeout=self.agent_timeout,
                master_timeout=self.master_timeout,
                funnel_profile=self.funnel_profile,
                max_candidates=self.max_candidates,
                trend_windows=list(self.trend_windows),
                volume_spike_threshold=self.volume_spike_threshold,
                breakout_distance_pct=self.breakout_distance_pct,
                sector_bucket_limit=self.sector_bucket_limit,
                recall_context=self.recall_context,
                data_snapshot=data_snapshot,
            )
            records, summary = snapshot_usage(current_usage_session_id() or "")
            effective_records = [record for record in records if bool(getattr(record, "success", False))]
            effective_summary = build_effective_usage_summary(records)
            result = build_pipeline_result_from_dag(
                dag_artifacts=dag_artifacts,
                usage_records=records,
                usage_summary=summary,
                effective_usage_records=effective_records,
                effective_usage_summary=effective_summary,
                total_time=time.time() - t0,
                enable_agent_layer=self.enable_agent_layer,
                execution_log=list(self.result.execution_log),
            )
            self.result = result
            if summary.call_count:
                self._log(
                    f"LLM 可观测性: calls={summary.call_count}, tokens={summary.total_tokens}, cost=${summary.estimated_cost_usd:.6f}"
                )
            self._log(f"✅ 主线分析完成，总耗时 {result.total_time:.1f}s")
            return result
        finally:
            if managed_session is not None:
                end_usage_session(managed_session)

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print(self.result.final_report)
        print("=" * 70)

    def save_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.result.final_report)
        _logger.info(f"报告已保存到: {path}")
