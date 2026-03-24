#!/usr/bin/env python3
"""
五路并行研究编排器

主流程：
数据层 -> 五大研究分支并行 -> 风控层 -> 集成裁判层
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime, timedelta
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import pandas as pd

from quant_investor.branch_contracts import (
    BranchResult,
    DebateVerdict,
    EvidencePacket,
    PortfolioStrategy,
    ResearchPipelineResult,
    TradeRecommendation,
    UnifiedDataBundle,
)
from quant_investor.branch_debate_engine import BranchDebateEngine
from quant_investor.config import config
from quant_investor.debate_scheduler import DebateScheduler, DebateRetryPolicy
from quant_investor.debate_templates import BRANCH_DEBATE_ADJUSTMENT_CAPS
from quant_investor.enhanced_data_layer import EnhancedDataLayer
from quant_investor.ensemble_judge import EnsembleJudge
from quant_investor.fundamental_branch import FundamentalBranch
from quant_investor.logger import get_logger
from quant_investor.macro_terminal_tushare import create_terminal
from quant_investor.risk_management_layer import PortfolioOptimizer, RiskLayerResult, RiskManagementLayer
from quant_investor.signal_calibration import SignalCalibrator
from quant_investor.versioning import (
    ARCHITECTURE_VERSION_V9,
    BRANCH_SCHEMA_VERSION_V9,
    BRANCH_TRACKER_SCHEMA_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    CURRENT_BRANCH_ORDER,
    CURRENT_BRANCH_WEIGHTS,
    DEBATE_TEMPLATE_VERSION,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
    output_version_payload,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


_BRANCH_METHOD_NAMES = {
    "kline": "_run_kline_branch",
    "quant": "_run_quant_branch",
    "fundamental": "_run_fundamental_branch",
    "intelligence": "_run_intelligence_branch",
    "macro": "_run_macro_branch",
}


def _make_ipc_safe(value: Any) -> Any:
    """将分支返回值裁剪为可跨进程传输的安全结构。"""
    try:
        pickle.dumps(value)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(key): _make_ipc_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_make_ipc_safe(item) for item in value]
        if hasattr(value, "__dict__"):
            return {
                str(key): _make_ipc_safe(item)
                for key, item in vars(value).items()
                if not callable(item)
            }
        return repr(value)


def _ipc_safe_branch_result(result: BranchResult) -> BranchResult:
    """裁剪分支结果中的不可 pickle 字段，避免 IPC 传输失败。"""
    evidence = result.evidence if isinstance(result.evidence, EvidencePacket) else EvidencePacket(
        branch_name=str(result.branch_name)
    )
    verdict = result.debate_verdict if isinstance(result.debate_verdict, DebateVerdict) else DebateVerdict()
    return BranchResult(
        branch_name=str(result.branch_name),
        score=float(result.score),
        confidence=float(result.confidence),
        signals=_make_ipc_safe(result.signals),
        risks=[str(item) for item in result.risks],
        explanation=str(result.explanation),
        symbol_scores={str(symbol): float(score) for symbol, score in result.symbol_scores.items()},
        success=bool(result.success),
        metadata=_make_ipc_safe(result.metadata),
        base_score=float(result.base_score if result.base_score is not None else result.score),
        final_score=float(result.final_score if result.final_score is not None else result.score),
        base_confidence=float(
            result.base_confidence if result.base_confidence is not None else result.confidence
        ),
        final_confidence=float(
            result.final_confidence if result.final_confidence is not None else result.confidence
        ),
        horizon_days=int(result.horizon_days),
        evidence=EvidencePacket(
            branch_name=str(evidence.branch_name),
            as_of=str(evidence.as_of),
            scope=str(evidence.scope),
            summary=str(evidence.summary),
            symbols=[str(item) for item in evidence.symbols],
            top_symbols=[str(item) for item in evidence.top_symbols],
            bull_points=[str(item) for item in evidence.bull_points],
            bear_points=[str(item) for item in evidence.bear_points],
            risk_points=[str(item) for item in evidence.risk_points],
            unknowns=[str(item) for item in evidence.unknowns],
            used_features=[str(item) for item in evidence.used_features],
            feature_values=_make_ipc_safe(evidence.feature_values),
            symbol_context=_make_ipc_safe(evidence.symbol_context),
            metadata=_make_ipc_safe(evidence.metadata),
        ),
        debate_verdict=DebateVerdict(
            direction=str(verdict.direction),
            confidence=float(verdict.confidence),
            score_adjustment=float(verdict.score_adjustment),
            bull_points=[str(item) for item in verdict.bull_points],
            bear_points=[str(item) for item in verdict.bear_points],
            risk_flags=[str(item) for item in verdict.risk_flags],
            unknowns=[str(item) for item in verdict.unknowns],
            used_features=[str(item) for item in verdict.used_features],
            hard_veto=bool(verdict.hard_veto),
            metadata=_make_ipc_safe(verdict.metadata),
        ),
        data_quality=_make_ipc_safe(result.data_quality),
        conclusion=str(result.conclusion),
        thesis_points=[str(item) for item in result.thesis_points],
        investment_risks=[str(item) for item in result.investment_risks],
        coverage_notes=[str(item) for item in result.coverage_notes],
        diagnostic_notes=[str(item) for item in result.diagnostic_notes],
        support_drivers=[str(item) for item in result.support_drivers],
        drag_drivers=[str(item) for item in result.drag_drivers],
        weight_cap_reasons=[str(item) for item in result.weight_cap_reasons],
        module_coverage=_make_ipc_safe(result.module_coverage),
    )


def _branch_process_entry(
    branch_name: str,
    pipeline_kwargs: dict[str, Any],
    market_regime: str | None,
    data_bundle: UnifiedDataBundle,
    send_conn: Connection,
) -> None:
    """在独立子进程中执行单个研究分支，便于超时后强制回收。"""
    try:
        pipeline = ParallelResearchPipeline(**pipeline_kwargs)
        pipeline._market_regime = market_regime
        branch_method = getattr(pipeline, _BRANCH_METHOD_NAMES[branch_name])
        branch_result = branch_method(data_bundle)
        send_conn.send({"ok": True, "result": _ipc_safe_branch_result(branch_result)})
    except Exception as exc:
        send_conn.send({"ok": False, "error": str(exc)})
    finally:
        send_conn.close()


# ---------------------------------------------------------------------------
# 自适应集成权重追踪器
# ---------------------------------------------------------------------------

class BranchPerformanceTracker:
    """
    基于滚动 IC（信息系数）动态调整五路分支权重。

    每次运行后将分支得分与实际收益存入 JSON 历史文件，
    下次运行时读取历史 IC 计算自适应权重：
      w_i = max(IC_i_rolling, 0) / sum(max(IC_j, 0))
    若所有分支 IC 均非正，回退到预设静态权重。

    历史文件：data/branch_ic_history.json（相对于工作目录）
    """

    MIN_WEIGHT = 0.05      # 分支 IC ≤ 0 时的最低保底权重
    ROLLING_WINDOW = 60    # 滚动 IC 窗口（天/次）
    HISTORY_PATH = "data/branch_ic_history.json"

    def __init__(
        self,
        default_weights: dict[str, float],
        architecture_version: str,
        branch_schema_version: str,
    ) -> None:
        self.default_weights = dict(default_weights)
        self.architecture_version = architecture_version
        self.branch_schema_version = branch_schema_version
        self._history: dict[str, list[float]] = {k: [] for k in self.default_weights}
        self._archived_history: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        try:
            import json
            if os.path.exists(self.HISTORY_PATH):
                with open(self.HISTORY_PATH) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "history" in data:
                    raw_history = dict(data.get("history", {}))
                    self._archived_history = {
                        str(key): [float(item) for item in value][-self.ROLLING_WINDOW:]
                        for key, value in dict(data.get("archived_history", {})).items()
                    }
                else:
                    raw_history = dict(data)
                if "kronos" in raw_history and "kline" not in raw_history and "kline" in self._history:
                    raw_history["kline"] = list(raw_history.pop("kronos"))
                elif "kronos" in raw_history:
                    self._archived_history["kronos"] = list(raw_history.pop("kronos"))
                if "llm_debate" in raw_history and "llm_debate" not in self._history:
                    self._archived_history["llm_debate"] = list(raw_history.pop("llm_debate"))
                for branch in self._history:
                    self._history[branch] = raw_history.get(branch, [])[-self.ROLLING_WINDOW:]
        except Exception:
            pass

    def save(self, branch_scores: dict[str, float], realized_return: float) -> None:
        """记录本次运行的分支得分 × 实际收益（IC 代理）"""
        import json
        for branch, score in branch_scores.items():
            if branch in self._history:
                ic_proxy = score * np.sign(realized_return)
                self._history[branch].append(float(ic_proxy))
                self._history[branch] = self._history[branch][-self.ROLLING_WINDOW:]
        try:
            os.makedirs(os.path.dirname(self.HISTORY_PATH) or ".", exist_ok=True)
            with open(self.HISTORY_PATH, "w") as f:
                json.dump(
                    {
                        "schema_version": BRANCH_TRACKER_SCHEMA_VERSION,
                        "architecture_version": self.architecture_version,
                        "branch_schema_version": self.branch_schema_version,
                        "ic_protocol_version": IC_PROTOCOL_VERSION,
                        "report_protocol_version": REPORT_PROTOCOL_VERSION,
                        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
                        "active_branches": list(self.default_weights.keys()),
                        "history": self._history,
                        "archived_history": self._archived_history,
                    },
                    f,
                )
        except Exception:
            pass

    def get_adaptive_weights(self) -> dict[str, float]:
        """
        计算自适应权重。历史记录不足时回退到静态权重。
        """
        ic_means: dict[str, float] = {}
        for branch, history in self._history.items():
            if len(history) >= 5:
                ic_means[branch] = float(np.mean(history[-self.ROLLING_WINDOW:]))
            else:
                ic_means[branch] = self.default_weights.get(branch, 0.1)

        positive_ics = {k: max(v, 0.0) for k, v in ic_means.items()}
        total = sum(positive_ics.values())
        if total < 1e-8:
            return dict(self.default_weights)

        weights = {k: max(v / total, self.MIN_WEIGHT) for k, v in positive_ics.items()}
        # 重新归一化（保底权重可能使总和 > 1）
        total_w = sum(weights.values())
        return {k: v / total_w for k, v in weights.items()}


def _safe_mean(values: list[float]) -> float:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def _series_or_empty(values: dict[str, float]) -> pd.Series:
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float)


class ParallelResearchPipeline:
    """并行研究主流程编排器。"""

    BRANCH_ORDER = list(CURRENT_BRANCH_ORDER)
    ARCHITECTURE_VERSION = ARCHITECTURE_VERSION_V9
    BRANCH_SCHEMA_VERSION = BRANCH_SCHEMA_VERSION_V9

    def __init__(
        self,
        stock_pool: list[str],
        market: str = "CN",
        lookback_years: float = 1.0,
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        enable_alpha_mining: bool = True,
        enable_quant: bool = True,
        enable_kline: bool = True,
        enable_fundamental: bool = True,
        enable_intelligence: bool = True,
        enable_branch_debate: bool = True,
        enable_macro: bool = True,
        kline_backend: str = "hybrid",
        allow_synthetic_for_research: bool = False,
        branch_timeout: float = 120.0,
        debate_top_k: int = 3,
        debate_min_abs_score: float = 0.08,
        debate_timeout_sec: float = 8.0,
        debate_model: str = "gpt-5.4-mini",
        enable_document_semantics: bool = True,
        verbose: bool = True,
        # 向后兼容
        enable_kronos: bool | None = None,
        enable_llm_debate: bool | None = None,
    ) -> None:
        self.stock_pool = stock_pool
        self.market = market.upper()
        self.lookback_years = lookback_years
        self.total_capital = total_capital
        self.risk_level = risk_level
        self.enable_alpha_mining = enable_alpha_mining
        self.enable_quant = enable_quant
        # 向后兼容：enable_kronos 映射到 enable_kline
        self.enable_kline = enable_kline if enable_kronos is None else enable_kronos
        self.enable_fundamental = enable_fundamental
        self.enable_intelligence = enable_intelligence
        self.enable_branch_debate = (
            enable_branch_debate if enable_llm_debate is None else enable_llm_debate
        )
        self.enable_llm_debate = self.enable_branch_debate
        self.enable_macro = enable_macro
        self.kline_backend = kline_backend
        self.allow_synthetic_for_research = allow_synthetic_for_research
        self.branch_timeout = max(float(branch_timeout), 0.0)
        self.debate_top_k = max(int(debate_top_k), 1)
        self.debate_min_abs_score = max(float(debate_min_abs_score), 0.0)
        self.debate_timeout_sec = max(float(debate_timeout_sec), 0.1)
        self.debate_model = debate_model
        self.enable_document_semantics = enable_document_semantics
        self.verbose = verbose
        self.architecture_version = self.ARCHITECTURE_VERSION
        self.branch_schema_version = self.BRANCH_SCHEMA_VERSION
        self.calibration_schema_version = CALIBRATION_SCHEMA_VERSION
        self.debate_template_version = DEBATE_TEMPLATE_VERSION
        self.data_layer = EnhancedDataLayer(market=self.market, verbose=verbose)
        self.risk_layer = RiskManagementLayer(verbose=verbose)
        self.branch_debate_engine = BranchDebateEngine(
            enabled=self.enable_branch_debate,
            model=self.debate_model,
            timeout_sec=self.debate_timeout_sec,
            min_abs_score=self.debate_min_abs_score,
        )
        self.debate_scheduler = DebateScheduler(
            engine=self.branch_debate_engine,
            per_batch_max_llm_calls=12,
            per_branch_max_calls=3,
            single_symbol_max_calls=1,
            timeout_sec=self.debate_timeout_sec,
            retry_policy=DebateRetryPolicy(max_attempts=1),
        )
        self._logger = get_logger("ParallelResearchPipeline", verbose)
        self._branch_tracker = BranchPerformanceTracker(
            default_weights=CURRENT_BRANCH_WEIGHTS,
            architecture_version=self.architecture_version,
            branch_schema_version=self.branch_schema_version,
        )
        self._market_regime: str | None = None  # 由 run() 中 RegimeDetector 设定
        self.enable_agent_orchestrator_bridge = True

    def _log(self, message: str) -> None:
        self._logger.info(message)

    def _enabled_branch_flags(self) -> dict[str, bool]:
        return {
            "kline": self.enable_kline,
            "quant": self.enable_quant,
            "fundamental": self.enable_fundamental,
            "intelligence": self.enable_intelligence,
            "macro": self.enable_macro,
        }

    def _enabled_branch_names(self) -> list[str]:
        return [name for name in self.BRANCH_ORDER if self._enabled_branch_flags().get(name, False)]

    def _branch_runtime_kwargs(self) -> dict[str, Any]:
        """提取可复用的分支运行参数，供子进程重建轻量上下文。"""
        return {
            "stock_pool": list(self.stock_pool),
            "market": self.market,
            "lookback_years": self.lookback_years,
            "total_capital": self.total_capital,
            "risk_level": self.risk_level,
            "enable_alpha_mining": self.enable_alpha_mining,
            "enable_quant": self.enable_quant,
            "enable_kline": self.enable_kline,
            "enable_fundamental": self.enable_fundamental,
            "enable_intelligence": self.enable_intelligence,
            "enable_branch_debate": self.enable_branch_debate,
            "enable_macro": self.enable_macro,
            "kline_backend": self.kline_backend,
            "allow_synthetic_for_research": self.allow_synthetic_for_research,
            "branch_timeout": self.branch_timeout,
            "debate_top_k": self.debate_top_k,
            "debate_min_abs_score": self.debate_min_abs_score,
            "debate_timeout_sec": self.debate_timeout_sec,
            "debate_model": self.debate_model,
            "enable_document_semantics": self.enable_document_semantics,
            "verbose": self.verbose,
        }

    @staticmethod
    def _branch_process_context() -> mp.context.BaseContext:
        """优先选择 fork，确保测试 monkeypatch 和大对象共享在 POSIX 下可用。"""
        if os.name != "nt":
            start_methods = mp.get_all_start_methods()
            if "fork" in start_methods:
                return mp.get_context("fork")
        return mp.get_context("spawn")

    @staticmethod
    def _terminate_branch_process(process: mp.Process) -> None:
        """终止超时分支对应的子进程。"""
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)

    def _degraded_branch_result(
        self,
        branch_name: str,
        explanation: str,
        degraded_reason: str,
        risks: list[str] | None = None,
    ) -> BranchResult:
        return self._annotate_branch_result(
            BranchResult(
            branch_name=branch_name,
            score=0.0,
            confidence=0.0,
            risks=[],
            explanation=explanation,
            symbol_scores={symbol: 0.0 for symbol in self.stock_pool},
            success=False,
            metadata={
                "data_source_status": "degraded_branch",
                "is_synthetic": False,
                "degraded_reason": degraded_reason,
                "branch_mode": self._default_branch_mode(branch_name),
                "reliability": 0.2,
            },
            data_quality={"status": "degraded_branch", "reason": degraded_reason},
            conclusion=f"{branch_name} 分支本轮未形成可直接执行的增量结论，当前按中性结果处理。",
            diagnostic_notes=risks or [f"{branch_name} 分支降级: {degraded_reason}"],
            )
        )

    def _version_payload(self) -> dict[str, str]:
        return output_version_payload(
            architecture_version=self.architecture_version,
            branch_schema_version=self.branch_schema_version,
        )

    def _annotate_branch_result(self, branch_result: BranchResult) -> BranchResult:
        versions = self._version_payload()
        branch_result.architecture_version = versions["architecture_version"]
        branch_result.branch_schema_version = versions["branch_schema_version"]
        branch_result.calibration_schema_version = versions["calibration_schema_version"]
        branch_result.debate_template_version = versions["debate_template_version"]
        branch_result.metadata.setdefault("architecture_version", branch_result.architecture_version)
        branch_result.metadata.setdefault("branch_schema_version", branch_result.branch_schema_version)
        branch_result.metadata.setdefault(
            "calibration_schema_version",
            branch_result.calibration_schema_version,
        )
        branch_result.metadata.setdefault("debate_template_version", branch_result.debate_template_version)
        if not branch_result.conclusion:
            if branch_result.score >= 0.15:
                branch_result.conclusion = f"{branch_result.branch_name} 分支当前给出偏正面结论。"
            elif branch_result.score <= -0.15:
                branch_result.conclusion = f"{branch_result.branch_name} 分支当前给出偏谨慎结论。"
            else:
                branch_result.conclusion = f"{branch_result.branch_name} 分支当前维持中性结论。"
        if not branch_result.thesis_points:
            branch_result.thesis_points = [branch_result.conclusion]
        if not branch_result.investment_risks and branch_result.risks:
            branch_result.investment_risks = [str(item) for item in branch_result.risks[:4]]
        if not branch_result.support_drivers and branch_result.score > 0.05:
            branch_result.support_drivers = [f"{branch_result.branch_name} 分支综合评分为正。"]
        if not branch_result.drag_drivers and branch_result.score < -0.05:
            branch_result.drag_drivers = [f"{branch_result.branch_name} 分支综合评分偏弱。"]
        return branch_result

    def run(self) -> ResearchPipelineResult:
        """执行完整并行研究流程。"""
        t0 = time.time()
        data_bundle = self._build_data_bundle()
        result = ResearchPipelineResult(
            data_bundle=data_bundle,
            **self._version_payload(),
        )
        result.execution_log.append("并行研究流程启动")
        result.timings["data_layer"] = time.time() - t0

        # 在分支执行前检测市场状态，供各分支自适应使用
        try:
            from quant_investor.regime_detector import RegimeDetector
            combined_for_regime = data_bundle.combined_frame()
            market_ret = combined_for_regime.groupby("date")["close"].mean().pct_change().dropna()
            if len(market_ret) >= 20:
                regime, _ = RegimeDetector().detect(market_ret)
                self._market_regime = regime.value
                self._log(f"市场状态识别：{self._market_regime}")
        except Exception as regime_exc:
            self._log(f"市场状态识别失败，使用默认权重: {regime_exc}")

        branch_start = time.time()
        result.branch_results = self._run_branches(data_bundle)
        result.timings["research_branches"] = time.time() - branch_start
        result.calibrated_signals = self._calibrate_signals(data_bundle, result.branch_results)

        # 分级 Quorum 检查
        enabled_branch_count = len(self._enabled_branch_names())
        successful_branches = [b for b in result.branch_results.values() if b.success]
        n_success = len(successful_branches)
        self._enabled_branch_count = enabled_branch_count
        # 分级可靠度乘数：0-1=禁止交易, 2=0.6, 3=0.85, 4=0.95, 5=1.0
        self._quorum_reliability = {0: 0.0, 1: 0.0, 2: 0.6, 3: 0.85, 4: 0.95, 5: 1.0}.get(n_success, 1.0)
        self._quorum_max_exposure = {0: 0.0, 1: 0.0, 2: 0.30, 3: 0.95, 4: 0.95, 5: 0.95}.get(n_success, 0.95)
        if enabled_branch_count == 0:
            self._log("警告：未启用任何研究分支，进入 research_only 模式。")
            result.execution_log.append("[CRITICAL] 未启用任何研究分支，研究模式，不生成交易。")
        elif n_success <= 1:
            self._log(
                f"警告：仅 {n_success}/{enabled_branch_count} 个启用分支成功，"
                "进入 research_only 模式，禁止生成交易建议。"
            )
            result.execution_log.append(
                f"[CRITICAL] 分支成功数严重不足（{n_success}/{enabled_branch_count}），研究模式，不生成交易。"
            )
        elif n_success == 2:
            self._log(
                f"警告：仅 {n_success}/{enabled_branch_count} 个启用分支成功，"
                "仓位上限降至 30%，可靠度 ×0.6。"
            )
            result.execution_log.append(
                f"[WARN] 分支成功数偏低（{n_success}/{enabled_branch_count}），仓位受限，结果仅供参考。"
            )
        elif n_success == 3:
            result.execution_log.append(
                f"[INFO] {n_success}/{enabled_branch_count} 个启用分支成功，正常运行（轻度警告）。"
            )

        risk_start = time.time()
        result.risk_result = self._run_risk_layer(data_bundle, result.branch_results, result.calibrated_signals)
        result.timings["risk_layer"] = time.time() - risk_start

        ensemble_start = time.time()
        result.final_strategy = self._run_ensemble_layer(
            data_bundle,
            result.branch_results,
            result.risk_result,
            calibrated_signals=result.calibrated_signals,
        )
        result.final_report = self._build_markdown_report(result)
        self._run_agent_orchestrator_bridge(result)
        result.timings["ensemble_layer"] = time.time() - ensemble_start
        result.timings["total"] = time.time() - t0
        result.execution_log.append("并行研究流程完成")
        return result

    def _run_agent_orchestrator_bridge(self, result: ResearchPipelineResult) -> None:
        """兼容过渡：复用旧 branch_results 生成新协议层 artifacts。"""
        if not self.enable_agent_orchestrator_bridge:
            return
        try:
            from quant_investor.agent_orchestrator import AgentOrchestrator

            orchestrator = AgentOrchestrator()
            bridge_output = orchestrator.run_with_precomputed_research(
                data_bundle=result.data_bundle,
                branch_results=result.branch_results,
                constraints=self._build_agent_bridge_constraints(result),
                existing_portfolio={"current_weights": {}},
                tradability_snapshot=self._build_agent_bridge_tradability(result.data_bundle),
                persist_outputs=False,
            )
            result.agent_orchestration = bridge_output
            result.agent_portfolio_plan = bridge_output["portfolio_plan"]
            result.agent_report_bundle = bridge_output["report_bundle"]
            result.agent_ic_decisions = bridge_output["ic_by_symbol"]
            result.execution_log.append(
                f"[INFO] AgentOrchestrator bridge 完成，生成 {len(bridge_output['ic_by_symbol'])}/{len(result.data_bundle.symbols)} 个 ICDecision。"
            )
        except Exception as exc:
            result.execution_log.append(f"[WARN] AgentOrchestrator bridge 跳过: {exc}")

    def _build_agent_bridge_constraints(self, result: ResearchPipelineResult) -> dict[str, Any]:
        """把 legacy 风控输出映射为新 orchestrator 的最小约束集。"""
        risk_level = str(getattr(result.risk_result, "risk_level", "normal"))
        target_exposure = float(getattr(result.final_strategy, "target_exposure", 0.0))
        max_weight = float(
            result.final_strategy.risk_summary.get("max_single_position", self.risk_layer.max_position_size)
        )
        if risk_level == "danger":
            gross_exposure_cap = min(target_exposure, 0.25)
        elif risk_level == "warning":
            gross_exposure_cap = min(target_exposure, 0.55)
        else:
            gross_exposure_cap = max(target_exposure, 0.0)

        return {
            "gross_exposure_cap": max(0.0, min(gross_exposure_cap, 1.0)),
            "max_weight": max(0.0, min(max_weight, 1.0)),
            "blocked_symbols": list(
                result.final_strategy.provenance_summary.get("synthetic_symbols", [])
            ),
        }

    @staticmethod
    def _build_agent_bridge_tradability(
        data_bundle: UnifiedDataBundle,
    ) -> dict[str, dict[str, Any]]:
        """为 bridge 模式构造最小可交易快照。"""
        tradability: dict[str, dict[str, Any]] = {}
        for symbol in data_bundle.symbols:
            frame = data_bundle.symbol_data.get(symbol)
            if frame is None or frame.empty:
                tradability[symbol] = {"is_tradable": True, "sector": "unknown", "liquidity_score": 0.5}
                continue
            avg_volume = float(frame.get("volume", pd.Series([1_000_000])).tail(20).mean() or 1_000_000)
            tradability[symbol] = {
                "is_tradable": True,
                "sector": str(data_bundle.fundamentals.get(symbol, {}).get("sector", "unknown")),
                "liquidity_score": max(0.2, min(1.0, avg_volume / 5_000_000)),
            }
        return tradability

    # ---------------------------------------------------------------------
    # 数据层
    # ---------------------------------------------------------------------

    def _build_data_bundle(self) -> UnifiedDataBundle:
        """构建统一数据包。"""
        self._log("构建统一数据包...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(365 * self.lookback_years))

        symbol_data: dict[str, pd.DataFrame] = {}
        fundamentals: dict[str, dict[str, Any]] = {}
        event_data: dict[str, list[dict[str, Any]]] = {}
        sentiment_data: dict[str, dict[str, Any]] = {}
        symbol_provenance: dict[str, dict[str, Any]] = {}

        for symbol in self.stock_pool:
            df, provenance = self._fetch_symbol_frame(symbol, start_date, end_date)
            symbol_data[symbol] = df
            symbol_provenance[symbol] = provenance
            fundamentals[symbol] = self._extract_fundamental_snapshot(df)
            event_data[symbol] = self._build_event_snapshot(symbol, df)
            sentiment_data[symbol] = self._build_sentiment_snapshot(symbol, df)

        macro_data = {
            "market": self.market,
            "as_of": end_date.strftime("%Y-%m-%d"),
            "risk_level": "中风险",
            "signal": "🟡",
        }

        return UnifiedDataBundle(
            market=self.market,
            symbols=self.stock_pool,
            symbol_data=symbol_data,
            fundamentals=fundamentals,
            event_data=event_data,
            sentiment_data=sentiment_data,
            macro_data=macro_data,
            metadata={
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
                "total_capital": self.total_capital,
                "risk_level": self.risk_level,
                "symbol_provenance": symbol_provenance,
                "data_source_status": self._bundle_data_source_status(symbol_provenance),
                "research_mode": self._bundle_research_mode(symbol_provenance),
            },
        )

    def _fetch_symbol_frame(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """优先拉取真实数据，失败时降级到模拟数据。"""
        try:
            df = self.data_layer.fetch_and_process(
                symbol=symbol,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                label_periods=5,
            )
            if df is not None and not df.empty:
                if "forward_ret_5d" not in df.columns:
                    df["forward_ret_5d"] = df["close"].shift(-5) / df["close"] - 1
                price_source = str(getattr(self.data_layer, "last_ohlcv_source", "unknown"))
                fundamental_source = str(getattr(self.data_layer, "last_fundamental_source", "unknown"))
                return df, {
                    "symbol": symbol,
                    "data_source_status": "real",
                    "is_synthetic": False,
                    "degraded_reason": "",
                    "branch_mode": "real_market_data",
                    "reliability": 1.0,
                    "price_source": price_source,
                    "fundamental_source": fundamental_source,
                }
        except Exception as exc:
            self._log(f"{symbol} 真实数据获取失败，使用模拟数据: {exc}")
            degraded_reason = str(exc)
        else:
            degraded_reason = "empty_frame"

        return self._generate_mock_symbol_frame(symbol, start_date, end_date), {
            "symbol": symbol,
            "data_source_status": "synthetic_fallback",
            "is_synthetic": True,
            "degraded_reason": degraded_reason,
            "branch_mode": "synthetic_fallback",
            "reliability": 0.35,
        }

    def _generate_mock_symbol_frame(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """生成可用于测试和离线运行的模拟数据。"""
        seed = abs(hash(symbol)) % 10000
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start_date, end_date)
        drift = 0.0005 + (seed % 7) * 0.00008
        shocks = rng.normal(drift, 0.02, len(dates))
        close = 100 * np.exp(np.cumsum(shocks))
        open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, len(dates)))
        volume = rng.integers(2_000_000, 12_000_000, len(dates))
        amount = close * volume

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            }
        )
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        df["momentum_12_1"] = df["close"].pct_change(120) - df["close"].pct_change(20)
        df["volatility_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
        df["rsi_14"] = self._compute_rsi(df["close"], 14)
        df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
        df["ma_bias_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).mean()
        df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
        df["label_return"] = df["close"].shift(-5) / df["close"] - 1
        df["forward_ret_5d"] = df["label_return"]
        df["symbol"] = symbol
        df["market"] = self.market
        df["roe"] = 0.08 + (seed % 9) * 0.01
        df["gross_margin"] = 0.2 + (seed % 5) * 0.03
        df["profit_growth"] = -0.05 + (seed % 8) * 0.03
        df["revenue_growth"] = -0.02 + (seed % 6) * 0.025
        df["debt_ratio"] = 0.25 + (seed % 4) * 0.08
        df["pe"] = 10 + (seed % 15)
        df["pb"] = 1.0 + (seed % 6) * 0.3
        df["ps"] = 1.0 + (seed % 5) * 0.25
        return df

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def _extract_fundamental_snapshot(self, df: pd.DataFrame) -> dict[str, Any]:
        """从数据表抽取基本面快照。"""
        if df is None or df.empty:
            return {}
        latest = df.iloc[-1]
        fields = [
            "roe",
            "gross_margin",
            "profit_growth",
            "revenue_growth",
            "debt_ratio",
            "pe",
            "pb",
            "ps",
        ]
        snapshot = {field: float(latest[field]) for field in fields if field in latest and pd.notna(latest[field])}
        snapshot["latest_close"] = float(latest.get("close", 0.0))
        return snapshot

    def _build_event_snapshot(self, symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
        """构造占位事件摘要。"""
        if df is None or df.empty:
            return []
        latest = df.iloc[-1]
        recent_return = float(latest.get("return_20d", 0.0) or 0.0)
        volume_ratio = float(latest.get("volume_ratio_20d", 1.0) or 1.0)
        events = []
        if recent_return > 0.12:
            events.append({"type": "price_breakout", "headline": f"{symbol} 近期趋势显著增强", "impact": 0.2})
        if recent_return < -0.12:
            events.append({"type": "drawdown", "headline": f"{symbol} 近期出现较大回撤", "impact": -0.25})
        if volume_ratio > 1.5:
            events.append({"type": "abnormal_volume", "headline": f"{symbol} 成交量显著放大", "impact": 0.1})
        return events

    def _build_sentiment_snapshot(self, symbol: str, df: pd.DataFrame) -> dict[str, Any]:
        """构造情绪和广度占位数据。"""
        if df is None or df.empty:
            return {}
        ret = df["close"].pct_change().dropna()

        # --- 增强版恐贪指数（三维度组合） ---
        # 1) 动量信号：20 日 Sharpe
        momentum = float(ret.tail(20).mean() / (ret.tail(20).std() + 1e-8))
        # 2) 波动率排位（低波 = 偏贪婪）
        vol_20 = float(ret.tail(20).std()) if len(ret) >= 20 else float(ret.std())
        vol_252 = float(ret.tail(252).std()) if len(ret) >= 252 else vol_20
        vol_rank = 1.0 - min(vol_20 / (vol_252 + 1e-8), 2.0) / 2.0
        # 3) 价格在 52 周高低间的位置
        close = df["close"]
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        low_52w = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())
        price_position = (float(close.iloc[-1]) - low_52w) / (high_52w - low_52w + 1e-8)
        # 组合
        raw_fg = 0.35 * momentum + 0.25 * vol_rank + 0.40 * (price_position * 2 - 1)
        fear_greed = _clamp(raw_fg / 2.5, -1.0, 1.0)

        money_flow = float(df["volume"].tail(5).mean() / (df["volume"].tail(20).mean() + 1e-8) - 1)
        breadth = float((ret.tail(20) > 0).mean() * 2 - 1) if not ret.empty else 0.0
        return {
            "symbol": symbol,
            "fear_greed": fear_greed,
            "money_flow": _clamp(money_flow, -1.0, 1.0),
            "breadth": _clamp(breadth, -1.0, 1.0),
        }

    # ---------------------------------------------------------------------
    # 分支层
    # ---------------------------------------------------------------------

    def _run_branches(self, data_bundle: UnifiedDataBundle) -> dict[str, BranchResult]:
        enabled = self._enabled_branch_flags()

        results: dict[str, BranchResult] = {}
        enabled_names = [name for name in self.BRANCH_ORDER if enabled.get(name, False)]
        if not enabled_names:
            return results

        process_context = self._branch_process_context()
        process_map: dict[str, tuple[mp.Process, Connection]] = {}
        deadline = time.monotonic() + self.branch_timeout
        pipeline_kwargs = self._branch_runtime_kwargs()
        try:
            for name in enabled_names:
                recv_conn, send_conn = process_context.Pipe(duplex=False)
                process = process_context.Process(
                    target=_branch_process_entry,
                    args=(name, pipeline_kwargs, self._market_regime, data_bundle, send_conn),
                    name=f"qi-branch-{name}",
                )
                process.start()
                send_conn.close()
                process_map[name] = (process, recv_conn)
            for name in enabled_names:
                process, recv_conn = process_map[name]
                remaining = max(deadline - time.monotonic(), 0.0)
                process.join(timeout=remaining)

                if process.is_alive():
                    recv_conn.close()
                    self._terminate_branch_process(process)
                    self._log(f"{name} 分支超时（>{self.branch_timeout:.1f}s），已强制回收并降级。")
                    results[name] = self._degraded_branch_result(
                        branch_name=name,
                        explanation=f"{name} 分支执行超时，已降级为中性结果。",
                        degraded_reason="branch_timeout",
                        risks=[
                            f"{name} 分支执行超过 {self.branch_timeout:.1f}s，当前忽略该分支结果。"
                        ],
                    )
                    continue

                payload: dict[str, Any] | None = None
                try:
                    if recv_conn.poll(0.1):
                        payload = recv_conn.recv()
                except (EOFError, BrokenPipeError):
                    payload = None
                finally:
                    recv_conn.close()
                    process.join(timeout=0.1)

                if not payload:
                    degraded_reason = (
                        f"branch_process_exit_{process.exitcode}"
                        if process.exitcode not in (None, 0)
                        else "branch_process_no_result"
                    )
                    self._log(f"{name} 分支未返回结果，使用降级结果: {degraded_reason}")
                    results[name] = self._degraded_branch_result(
                        branch_name=name,
                        explanation=f"{name} 分支未返回有效结果，已降级为中性结果。",
                        degraded_reason=degraded_reason,
                    )
                    continue

                if not payload.get("ok", False):
                    error_message = str(payload.get("error", "branch_process_failed"))
                    self._log(f"{name} 分支失败，使用降级结果: {error_message}")
                    results[name] = self._degraded_branch_result(
                        branch_name=name,
                        explanation=f"{name} 分支异常，已降级为中性结果。",
                        degraded_reason=error_message,
                        risks=[f"{name} 分支失败: {error_message}"],
                    )
                    continue

                branch_result = payload["result"]
                branch_result.metadata.setdefault(
                    "data_source_status",
                    self._branch_data_source_status(data_bundle),
                )
                branch_result.metadata.setdefault("is_synthetic", False)
                branch_result.metadata.setdefault("degraded_reason", "")
                branch_result.metadata.setdefault("branch_mode", self._default_branch_mode(name))
                branch_result.metadata.setdefault("reliability", 1.0 if branch_result.success else 0.25)
                if name == "kline":
                    branch_result.metadata.setdefault("requested_backend", self.kline_backend)
                    effective_backend = str(
                        branch_result.metadata.get("effective_backend", self.kline_backend)
                    ).strip().lower() or "hybrid"
                    branch_result.metadata.setdefault("effective_backend", effective_backend)
                    branch_result.metadata.setdefault(
                        "llm_interface_reserved",
                        effective_backend in {"chronos", "hybrid"},
                    )
                branch_result.horizon_days = int(
                    branch_result.metadata.get("horizon_days", branch_result.horizon_days or 5)
                )
                branch_result = self._annotate_branch_result(branch_result)
                evidence_packet = self._build_branch_evidence_packet(
                    branch_name=name,
                    data_bundle=data_bundle,
                    base_result=branch_result,
                )
                branch_result.evidence = evidence_packet
                branch_result = self._apply_branch_debate(name, evidence_packet, branch_result, data_bundle)
                results[name] = branch_result
        finally:
            for process, recv_conn in process_map.values():
                if not recv_conn.closed:
                    recv_conn.close()
                if process.is_alive():
                    self._terminate_branch_process(process)

        for name in self.BRANCH_ORDER:
            if name not in results and enabled.get(name, False):
                results[name] = self._degraded_branch_result(
                    branch_name=name,
                    explanation=f"{name} 分支未执行，返回中性结果。",
                    degraded_reason="branch_not_executed",
                )

        return results

    @staticmethod
    def _dedupe_list(items: list[str]) -> list[str]:
        deduped: list[str] = []
        for item in items:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    def _build_branch_evidence_packet(
        self,
        branch_name: str,
        data_bundle: UnifiedDataBundle,
        base_result: BranchResult,
    ) -> EvidencePacket:
        """把 deterministic/base 分支结果提炼为统一证据包。"""
        ranked = sorted(
            base_result.symbol_scores.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        top_symbols = [
            symbol
            for symbol, score in ranked
            if abs(float(score)) >= self.debate_min_abs_score
        ][: self.debate_top_k]
        if not top_symbols:
            top_symbols = [symbol for symbol, _ in ranked[: self.debate_top_k]]

        bull_points: list[str] = []
        bear_points: list[str] = []
        risk_points = list(base_result.risks[:4])
        unknowns: list[str] = []
        used_features = list(base_result.signals.keys())[:10]
        symbol_context: dict[str, dict[str, Any]] = {}

        if branch_name == "kline":
            predicted_returns = base_result.signals.get("predicted_return", {})
            regimes = base_result.signals.get("trend_regime", {})
            for symbol in top_symbols:
                score = float(base_result.symbol_scores.get(symbol, 0.0))
                pred = float(predicted_returns.get(symbol, 0.0))
                regime = str(regimes.get(symbol, "未知"))
                symbol_context[symbol] = {"score": score, "predicted_return": pred, "regime": regime}
                if score > 0:
                    bull_points.append(f"{symbol} 趋势状态 {regime}，预测收益 {pred:+.2%}。")
                elif score < 0:
                    bear_points.append(f"{symbol} 趋势偏弱，预测收益 {pred:+.2%}。")
        elif branch_name == "quant":
            alpha_factors = base_result.signals.get("alpha_factors", [])
            if alpha_factors:
                bull_points.append("量化因子组: " + " / ".join(str(item) for item in alpha_factors[:5]))
            for symbol in top_symbols:
                score = float(base_result.symbol_scores.get(symbol, 0.0))
                symbol_context[symbol] = {
                    "score": score,
                    "factor_exposure": base_result.signals.get("factor_exposures", {}).get(symbol, {}),
                }
                if score > 0:
                    bull_points.append(f"{symbol} 量化打分为正。")
                elif score < 0:
                    bear_points.append(f"{symbol} 量化打分为负。")
        elif branch_name == "fundamental":
            bull_case = base_result.signals.get("bull_case", {})
            bear_case = base_result.signals.get("bear_case", {})
            quality = base_result.signals.get("quality_breakdown", {})
            for symbol in top_symbols:
                symbol_context[symbol] = _make_ipc_safe(quality.get(symbol, {}))
                bull_points.extend(str(item) for item in bull_case.get(symbol, [])[:2])
                bear_points.extend(str(item) for item in bear_case.get(symbol, [])[:2])
            if base_result.data_quality.get("documents_missing_symbols"):
                unknowns.append("部分标的缺少离线文档语义快照。")
            if base_result.data_quality.get("coverage_ratio", 0.0) < 0.5:
                unknowns.append("基本面覆盖度偏低，分支自动回退为偏中性解释。")
        elif branch_name == "intelligence":
            event_scores = base_result.signals.get("event_risk_score", {})
            sentiment_scores = base_result.signals.get("sentiment_score", {})
            flow_scores = base_result.signals.get("money_flow_score", {})
            for symbol in top_symbols:
                score = float(base_result.symbol_scores.get(symbol, 0.0))
                symbol_context[symbol] = {
                    "score": score,
                    "event": float(event_scores.get(symbol, 0.0)),
                    "sentiment": float(sentiment_scores.get(symbol, 0.0)),
                    "flow": float(flow_scores.get(symbol, 0.0)),
                }
                if score > 0:
                    bull_points.append(f"{symbol} 事件/情绪/资金流共振偏正。")
                elif score < 0:
                    bear_points.append(f"{symbol} 事件/情绪/资金流偏弱。")
        elif branch_name == "macro":
            bull_points.append(f"宏观 regime: {base_result.signals.get('macro_regime', '未知')}")
            bull_points.append(f"流动性信号: {base_result.signals.get('liquidity_signal', '🟡')}")
            symbol_context["market"] = _make_ipc_safe(base_result.signals)
            if base_result.success is False:
                unknowns.append("宏观分支当前使用降级统计。")

        if not bull_points and base_result.score > 0:
            bull_points.append(f"{branch_name} base score 为正。")
        if not bear_points and base_result.score < 0:
            bear_points.append(f"{branch_name} base score 为负。")
        if not used_features:
            used_features = ["score", "confidence"]

        return EvidencePacket(
            branch_name=branch_name,
            as_of=str(data_bundle.metadata.get("end_date", "")),
            scope="market" if branch_name == "macro" else "symbol",
            summary=base_result.explanation,
            symbols=list(self.stock_pool),
            top_symbols=top_symbols,
            bull_points=self._dedupe_list(bull_points)[:6],
            bear_points=self._dedupe_list(bear_points)[:6],
            risk_points=self._dedupe_list(risk_points)[:6],
            unknowns=self._dedupe_list(unknowns)[:4],
            used_features=self._dedupe_list([str(item) for item in used_features])[:10],
            feature_values={
                "branch_score": float(base_result.base_score if base_result.base_score is not None else base_result.score),
                "branch_confidence": float(
                    base_result.base_confidence
                    if base_result.base_confidence is not None
                    else base_result.confidence
                ),
            },
            symbol_context=symbol_context,
            metadata={
                "data_quality": _make_ipc_safe(base_result.data_quality),
                "branch_mode": base_result.metadata.get("branch_mode", self._default_branch_mode(branch_name)),
                "debate_model": self.debate_model,
                "template_version": self.debate_template_version,
            },
        )

    def _apply_branch_debate(
        self,
        branch_name: str,
        evidence_packet: EvidencePacket,
        base_result: BranchResult,
        data_bundle: UnifiedDataBundle | None = None,
    ) -> BranchResult:
        """执行 branch-local debate，并把 verdict 融入分支最终输出。"""
        base_score = float(base_result.base_score if base_result.base_score is not None else base_result.score)
        base_conf = float(
            base_result.base_confidence if base_result.base_confidence is not None else base_result.confidence
        )
        self.debate_scheduler.engine = self.branch_debate_engine
        verdict = self.debate_scheduler.evaluate_branch(
            branch_name=branch_name,
            evidence=evidence_packet,
            base_result=base_result,
            debate_top_k=self.debate_top_k,
            debate_min_abs_score=self.debate_min_abs_score,
            data_bundle=data_bundle,
        )

        cap = BRANCH_DEBATE_ADJUSTMENT_CAPS.get(branch_name, 0.10)
        bounded_adjustment = _clamp(float(verdict.score_adjustment), -cap, cap)
        final_score = _clamp(base_score + bounded_adjustment, -1.0, 1.0)
        disagreement = (base_score * bounded_adjustment) < 0

        if not verdict.hard_veto and base_score != 0.0 and final_score != 0.0 and base_score * final_score < 0:
            final_score = 0.0

        if verdict.hard_veto:
            if base_score > 0:
                final_score = min(abs(final_score), 0.10)
            elif base_score < 0:
                final_score = -min(abs(final_score), 0.10)
            else:
                final_score = 0.0

        if verdict.metadata.get("reason") in {"llm_provider_missing", "scheduler_gate_not_met"}:
            final_conf = base_conf
        elif verdict.hard_veto:
            final_conf = _clamp(min(base_conf, verdict.confidence or base_conf) * 0.78, 0.15, 1.0)
        elif disagreement:
            final_conf = _clamp(min(base_conf, verdict.confidence or base_conf) * 0.82, 0.15, 1.0)
        else:
            debate_conf = verdict.confidence if verdict.confidence > 0 else base_conf
            final_conf = _clamp(base_conf * 0.85 + debate_conf * 0.15, 0.0, 1.0)

        final_risks = self._dedupe_list([*base_result.risks, *verdict.risk_flags])
        explanation = base_result.explanation
        if verdict.metadata.get("status") == "deterministic_stub":
            explanation += " 已通过 branch-local debate 做 bounded review。"
        elif verdict.metadata.get("reason") == "llm_provider_missing":
            explanation += " branch-local debate 缺少 provider，保持 deterministic 结果。"

        metadata = dict(base_result.metadata)
        metadata.update(
            {
                "debate_enabled": self.enable_branch_debate,
                "debate_model": self.debate_model,
                "debate_status": verdict.metadata.get("status", "skipped"),
                "debate_reason": verdict.metadata.get("reason", ""),
                "bounded_adjustment_cap": cap,
                "bounded_adjustment": bounded_adjustment,
                **self._version_payload(),
            }
        )
        signals = dict(base_result.signals)
        signals["debate_verdict"] = verdict.to_dict()
        signals["base_score"] = base_score
        signals["final_score"] = final_score
        signals["base_confidence"] = base_conf
        signals["final_confidence"] = final_conf
        signals["bounded_adjustment"] = bounded_adjustment

        return self._annotate_branch_result(
            BranchResult(
            branch_name=base_result.branch_name,
            score=final_score,
            confidence=final_conf,
            signals=signals,
            risks=final_risks,
            explanation=explanation,
            symbol_scores=dict(base_result.symbol_scores),
            success=base_result.success,
            metadata=metadata,
            base_score=base_score,
            final_score=final_score,
            base_confidence=base_conf,
            final_confidence=final_conf,
            horizon_days=int(base_result.horizon_days),
            evidence=evidence_packet,
            debate_verdict=verdict,
            data_quality=dict(base_result.data_quality),
            )
        )

    def _calibrate_signals(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
    ) -> dict[str, Any]:
        """对全部分支结果做统一量纲校准。"""
        calibrator = SignalCalibrator(
            data_bundle.symbol_provenance(),
            architecture_version=self.architecture_version,
            branch_schema_version=self.branch_schema_version,
        )
        return calibrator.calibrate_all(branch_results, self.stock_pool)

    @staticmethod
    def _normalize_kline_backend_name(name: str | None) -> str:
        normalized = str(name or "hybrid").strip().lower()
        if normalized not in {"heuristic", "kronos", "chronos", "hybrid"}:
            return "hybrid"
        return normalized

    @staticmethod
    def _kline_branch_mode_for_backend(name: str) -> str:
        return "kline_heuristic" if name == "heuristic" else "kline_dual_model"

    def _fallback_kline_result(
        self,
        data_bundle: UnifiedDataBundle,
        reason: str,
    ) -> BranchResult:
        from quant_investor.kline_backends import get_backend

        base_backend = get_backend("heuristic")
        result = base_backend.predict(data_bundle.symbol_data, self.stock_pool)
        lowered_confidence = max(float(result.confidence) * 0.72, 0.25)
        result.confidence = lowered_confidence
        result.base_confidence = min(float(result.base_confidence or lowered_confidence), lowered_confidence)
        note = (
            "Chronos 深度模型阶段超时，已自动保留 base result。"
            if "timeout" in str(reason).lower()
            else "Chronos 深度模型失败，已自动保留 base result。"
        )
        if note not in result.diagnostic_notes:
            result.diagnostic_notes.append(note)
        result.conclusion = (
            str(result.conclusion).strip()
            or "深度模型未完成本轮推理，已保留 K 线基础结论，趋势判断仍然有效。"
        )
        result.metadata["fallback_reason"] = str(reason)
        result.metadata["branch_mode"] = "kline_heuristic"
        result.metadata["reliability"] = base_backend.reliability
        result.metadata["horizon_days"] = base_backend.horizon_days
        return result

    def _run_kline_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """K线分析分支。

        `kline_backend` 是唯一的运行语义开关：
        - `heuristic`: full-market / batch fast-screen
        - `hybrid` / `chronos` / `kronos`: shortlist / second-pass deep analysis
        """
        from quant_investor.kline_backends import get_backend

        requested_backend = self._normalize_kline_backend_name(self.kline_backend)
        effective_backend = requested_backend
        backend_kwargs: dict[str, Any] = {
            "kronos_path": config.KRONOS_MODEL_PATH,
            "kronos_model_size": config.KRONOS_MODEL_SIZE,
            "model_name": config.CHRONOS_MODEL_NAME,
            "evaluator_name": config.KLINE_EVALUATOR,
            "allow_remote_download": config.KLINE_ALLOW_REMOTE_MODEL_DOWNLOAD,
        }

        runtime_backend = effective_backend
        try:
            if effective_backend == "heuristic":
                backend = get_backend("heuristic")
            else:
                backend = get_backend(effective_backend, **backend_kwargs)
            result = backend.predict(data_bundle.symbol_data, self.stock_pool)
        except ModuleNotFoundError as exc:
            # 最小测试环境允许缺少 torch 等重依赖，此时回退到轻量 heuristic runtime。
            result = self._fallback_kline_result(data_bundle, f"missing_dependency:{exc.name}")
            runtime_backend = "heuristic_fallback"
            result.metadata.setdefault("degraded_reason", f"missing_dependency:{exc.name}")
            result.metadata.setdefault("runtime_fallback_dependency", exc.name)
            backend = get_backend("heuristic")
        except TimeoutError:
            result = self._fallback_kline_result(data_bundle, "timeout")
            runtime_backend = "heuristic_fallback"
            backend = get_backend("heuristic")
        except Exception as exc:
            result = self._fallback_kline_result(data_bundle, str(exc))
            runtime_backend = "heuristic_fallback"
            backend = get_backend("heuristic")
        result.branch_name = "kline"
        result.metadata["branch_mode"] = self._kline_branch_mode_for_backend(
            "heuristic" if runtime_backend.startswith("heuristic") else effective_backend
        )
        result.metadata.setdefault("reliability", backend.reliability)
        result.metadata.setdefault("horizon_days", backend.horizon_days)
        result.metadata["requested_backend"] = requested_backend
        result.metadata["effective_backend"] = effective_backend
        result.metadata["runtime_backend"] = runtime_backend
        result.metadata["llm_interface_reserved"] = effective_backend in {"chronos", "hybrid"}
        if not str(result.conclusion or "").strip():
            result.conclusion = (
                "启发式 K 线快筛已形成完整趋势结论。"
                if runtime_backend.startswith("heuristic")
                else "K 线深模型链路已形成完整趋势结论。"
            )
        return result

    def _run_quant_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """传统量化分支：Alpha 挖掘优先，失败时回退到经典因子。"""
        try:
            from quant_investor.alpha_mining import AlphaMiner, FactorLibrary
        except ModuleNotFoundError as exc:
            return BranchResult(
                branch_name="quant",
                explanation=f"量化依赖缺失，分支降级为中性结果: {exc}",
                symbol_scores={symbol: 0.0 for symbol in self.stock_pool},
                success=False,
                metadata={
                    "branch_mode": "alpha_research",
                    "reliability": 0.2,
                    "degraded_reason": f"missing_dependency:{exc.name}",
                },
                data_quality={
                    "status": "degraded_branch",
                    "reason": "missing_quant_dependency",
                    "dependency": exc.name,
                },
            )

        combined = data_bundle.combined_frame()
        if combined.empty:
            return BranchResult(
                branch_name="quant",
                explanation="量化分支没有可用数据，返回中性结果。",
                symbol_scores={symbol: 0.0 for symbol in self.stock_pool},
                success=False,
            )

        if "forward_ret_5d" not in combined.columns:
            combined["forward_ret_5d"] = combined["close"].shift(-5) / combined["close"] - 1

        selected_factors: list[str] = []
        explanation = ""
        if self.enable_alpha_mining:
            try:
                unique_dates = combined["date"].nunique() if "date" in combined.columns else 60
                gen_count = max(10, min(20, unique_dates // 15))
                miner = AlphaMiner(
                    combined.copy(),
                    forward_col="forward_ret_5d",
                    enable_genetic=True,
                    enable_llm=False,
                    genetic_generations=gen_count,
                )
                mining_result = miner.mine()
                selected_factors = [profile.name for profile in mining_result.selected_factors[:6]]
                explanation = (
                    f"Alpha 挖掘完成，优先采用 {len(selected_factors)} 个经过 IC/IR 和正交筛选的因子。"
                )
            except Exception as exc:
                explanation = f"Alpha 挖掘失败，使用经典因子回退: {exc}"
        else:
            explanation = "已关闭 Alpha 挖掘，当前直接使用经典量价因子组合作为量化研究分支。"

        if not selected_factors:
            selected_factors = [
                factor
                for factor in ["momentum_3m", "realized_vol_20d", "rsi_14", "macd_signal", "volume_ratio"]
                if factor in FactorLibrary.all_factor_funcs()
            ]
            explanation = "Alpha 挖掘未筛出稳定因子，当前回退到经典动量/低波/技术/量能因子组合。"

        factor_exposures: dict[str, dict[str, float]] = {symbol: {} for symbol in self.stock_pool}
        factor_scores = pd.DataFrame(index=combined.index)
        factor_funcs = FactorLibrary.all_factor_funcs()

        for factor_name in selected_factors:
            func = factor_funcs.get(factor_name)
            if func is None:
                continue
            try:
                factor_scores[factor_name] = func(combined)
            except Exception:
                continue

        if factor_scores.empty:
            factor_scores["fallback_momentum"] = combined.groupby("symbol")["close"].pct_change(20)
            selected_factors = ["fallback_momentum"]

        ranked = factor_scores.groupby(combined["date"]).transform(
            lambda series: (series - series.mean()) / (series.std(ddof=0) + 1e-8)
        )
        combined_scores = ranked.mean(axis=1).fillna(0.0)
        combined = combined.assign(quant_alpha_score=combined_scores)

        latest_rows = combined.sort_values("date").groupby("symbol").tail(1)
        symbol_scores = {
            str(row["symbol"]): float(_clamp(row["quant_alpha_score"] / 3, -1.0, 1.0))
            for _, row in latest_rows.iterrows()
        }

        for _, row in latest_rows.iterrows():
            symbol = str(row["symbol"])
            for factor_name in selected_factors[:5]:
                if factor_name in factor_scores.columns:
                    factor_exposures[symbol][factor_name] = float(factor_scores.loc[row.name, factor_name])

        expected_returns = {
            symbol: float(_clamp(score * 0.12, -0.25, 0.25))
            for symbol, score in symbol_scores.items()
        }
        score = _safe_mean(list(symbol_scores.values()))
        confidence = _clamp(0.4 + len(selected_factors) * 0.05, 0.4, 0.85)
        risks = []
        if len(selected_factors) <= 2:
            risks.append("可用 Alpha 因子较少，量化信号稳定性有限。")

        return BranchResult(
            branch_name="quant",
            score=score,
            confidence=confidence,
            signals={
                "alpha_factors": selected_factors,
                "factor_exposures": factor_exposures,
                "expected_return": expected_returns,
                "feature_reasoning": explanation,
            },
            risks=risks,
            explanation=explanation,
            symbol_scores=symbol_scores,
            metadata={
                "factor_exposures": factor_exposures,
                "expected_return": expected_returns,
                "branch_mode": "alpha_research",
                "reliability": 0.86 if len(selected_factors) >= 3 else 0.68,
                "horizon_days": 5,
            },
        )

    def _run_fundamental_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """公司基本面分支。"""
        branch = FundamentalBranch(
            data_layer=self.data_layer,
            stock_pool=self.stock_pool,
            enable_document_semantics=self.enable_document_semantics,
        )
        return branch.run(data_bundle)

    def _run_llm_debate_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """Deprecated: 旧 llm_debate 顶层分支已并入 V9 fundamental + branch-local debate。"""
        result = self._run_fundamental_branch(data_bundle)
        result.metadata["deprecated_alias"] = "llm_debate"
        result.metadata["deprecated_alias_target"] = "fundamental"
        return result

    def _run_intelligence_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """多维智能融合分支，仅覆盖事件/情绪/资金流/广度/行业轮动。"""
        event_risk: dict[str, float] = {}
        sentiment_score: dict[str, float] = {}
        money_flow_score: dict[str, float] = {}
        breadth_score: dict[str, float] = {}
        rotation_score: dict[str, float] = {}
        alerts: list[str] = []
        symbol_scores: dict[str, float] = {}

        advancing = 0
        declining = 0
        for symbol, df in data_bundle.symbol_data.items():
            sentiments = data_bundle.sentiment_data.get(symbol, {})
            events = data_bundle.event_data.get(symbol, [])

            event_bias = _safe_mean([float(item.get("impact", 0.0)) for item in events])
            event_score = _clamp(event_bias, -1.0, 1.0)
            senti = _clamp(
                0.5 * float(sentiments.get("fear_greed", 0.0))
                + 0.3 * float(sentiments.get("money_flow", 0.0))
                + 0.2 * float(sentiments.get("breadth", 0.0)),
                -1.0,
                1.0,
            )
            flow = _clamp(float(sentiments.get("money_flow", 0.0)), -1.0, 1.0)

            if df["close"].pct_change().iloc[-1] >= 0:
                advancing += 1
            else:
                declining += 1

            breadth = 0.0
            if advancing + declining > 0:
                breadth = (advancing - declining) / (advancing + declining)

            returns = df["close"].pct_change().dropna()
            short_trend = float(returns.tail(10).mean()) if len(returns) >= 5 else 0.0
            medium_trend = float(returns.tail(30).mean()) if len(returns) >= 10 else short_trend
            rotation = _clamp((short_trend - medium_trend) / (returns.tail(20).std() + 1e-8), -1.0, 1.0)

            if event_bias < -0.15:
                alerts.append(f"{symbol} 近期事件流显著偏负面。")
            if flow < -0.25:
                alerts.append(f"{symbol} 资金流持续走弱。")

            event_risk[symbol] = event_score
            sentiment_score[symbol] = senti
            money_flow_score[symbol] = flow
            breadth_score[symbol] = breadth
            rotation_score[symbol] = rotation
            # 根据市场状态自适应调整融合权重
            regime = self._market_regime
            if regime == "趋势上涨" or regime == "趋势下跌":
                w_evt, w_sen, w_flow, w_rot = 0.25, 0.30, 0.25, 0.20
            elif regime == "震荡低波":
                w_evt, w_sen, w_flow, w_rot = 0.20, 0.25, 0.25, 0.30
            elif regime == "震荡高波":
                w_evt, w_sen, w_flow, w_rot = 0.35, 0.20, 0.25, 0.20
            else:
                w_evt, w_sen, w_flow, w_rot = 0.30, 0.25, 0.25, 0.20
            symbol_scores[symbol] = _clamp(
                w_evt * event_score + w_sen * senti + w_flow * flow + w_rot * rotation,
                -1.0,
                1.0,
            )

        return BranchResult(
            branch_name="intelligence",
            score=_safe_mean(list(symbol_scores.values())),
            confidence=0.62,
            signals={
                "intelligence_score": symbol_scores,
                "event_risk_score": event_risk,
                "sentiment_score": sentiment_score,
                "money_flow_score": money_flow_score,
                "breadth_score": breadth_score,
                "rotation_score": rotation_score,
                "alerts": alerts,
            },
            risks=alerts[:5],
            explanation="多维智能融合分支只整合新闻/公告事件、情绪、资金流、市场广度与行业轮动信号。",
            symbol_scores=symbol_scores,
            metadata={
                "event_risk_score": event_risk,
                "sentiment_score": sentiment_score,
                "money_flow_score": money_flow_score,
                "breadth_score": breadth_score,
                "rotation_score": rotation_score,
                "branch_mode": "structured_intelligence_fusion",
                "reliability": 0.72,
                "horizon_days": 10,
            },
        )

    def _run_macro_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """宏观分支：运行一次，同一得分广播至所有标的。

        NOTE: 本分支设计为 portfolio 级别分析，整个 pipeline 仅执行一次（非逐股调用）。
        产出的 macro_score 统一赋值给所有标的的 symbol_scores。
        V10 Agent 层的 MacroAgentOutput.uniform_score_appropriateness 字段
        会判断此统一处理是否合理。
        """
        try:
            terminal = create_terminal(self.market)
            report = terminal.generate_risk_report()
            macro_score = {"🔴": -0.8, "🟡": 0.0, "🟢": 0.7, "🔵": 0.9}.get(report.overall_signal, 0.0)
            signal_map = {
                "macro_score": macro_score,
                "macro_regime": report.overall_risk_level,
                "liquidity_signal": report.overall_signal,
                "policy_signal": report.recommendation,
                "risk_level": report.overall_risk_level,
            }
            return BranchResult(
                branch_name="macro",
                score=macro_score,
                confidence=0.75,
                signals=signal_map,
                risks=[] if macro_score >= 0 else ["宏观环境偏谨慎，需控制总仓位。"],
                explanation="宏观分支基于多模块风控终端输出流动性、政策与风险状态。",
                symbol_scores={symbol: macro_score for symbol in self.stock_pool},
                metadata={
                    "report": report,
                    "branch_mode": "macro_terminal",
                    "reliability": 0.82,
                    "horizon_days": 20,
                },
            )
        except Exception as exc:
            self._log(f"宏观终端失败，使用增强降级统计: {exc}")

        combined = data_bundle.combined_frame()
        market_return = combined.groupby("date")["close"].mean().pct_change().dropna()

        # --- 增强降级回退：三维度宏观替代 ---
        # 1) 波动率分位数（低波 = 偏乐观）
        vol_20 = float(market_return.tail(20).std()) if len(market_return) >= 20 else 0.01
        vol_252 = float(market_return.tail(252).std()) if len(market_return) >= 252 else vol_20
        vol_percentile = min(vol_20 / (vol_252 + 1e-8), 2.0) / 2.0
        vol_score = 1.0 - vol_percentile  # 低波 = 积极

        # 2) 市场广度（涨跌比）
        latest_date = combined["date"].max() if "date" in combined.columns else None
        if latest_date is not None:
            latest_slice = combined[combined["date"] == latest_date]
            adv = int((latest_slice["close"].pct_change().fillna(0) >= 0).sum())
            dec = max(int(len(latest_slice) - adv), 1)
        else:
            adv, dec = 1, 1
        breadth_score = (adv - dec) / (adv + dec)

        # 3) 多周期动量结构
        ret_5d = float(market_return.tail(5).mean()) if len(market_return) >= 5 else 0.0
        ret_20d = float(market_return.tail(20).mean()) if len(market_return) >= 20 else 0.0
        ret_60d = float(market_return.tail(60).mean()) if len(market_return) >= 60 else ret_20d
        momentum_structure = 0.5 * ret_5d + 0.3 * ret_20d + 0.2 * ret_60d
        momentum_norm = _clamp(momentum_structure / (vol_20 + 1e-8), -1.0, 1.0)

        # 综合降级宏观得分
        score = _clamp(0.40 * vol_score + 0.35 * breadth_score + 0.25 * momentum_norm, -1.0, 1.0)

        # 集成 RegimeDetector 输出状态标签
        from quant_investor.regime_detector import RegimeDetector
        try:
            detector = RegimeDetector()
            regime_enum, _ = detector.detect(market_return)
            regime = regime_enum.value
        except Exception:
            regime = "低风险" if score > 0.2 else "高风险" if score < -0.2 else "中风险"

        signal = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "🟡"
        return BranchResult(
            branch_name="macro",
            score=score,
            confidence=0.50,
            signals={
                "macro_score": score,
                "macro_regime": regime,
                "liquidity_signal": signal,
                "policy_signal": "增强降级统计",
                "risk_level": regime,
                "vol_score": round(vol_score, 3),
                "breadth_score": round(breadth_score, 3),
                "momentum_norm": round(momentum_norm, 3),
            },
            risks=["宏观分支使用增强降级统计（波动率+广度+动量），建议后续补齐真实宏观数据。"],
            explanation="宏观分支降级为市场波动率分位、广度和多周期动量的增强估计。",
            symbol_scores={symbol: score for symbol in self.stock_pool},
            metadata={
                "branch_mode": "macro_enhanced_degraded",
                "reliability": 0.55,
                "horizon_days": 20,
            },
            success=False,
        )

    # ---------------------------------------------------------------------
    # 风控层
    # ---------------------------------------------------------------------

    def _run_risk_layer(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any] | None = None,
    ) -> RiskLayerResult:
        """将统一分支结果适配到现有风控层。"""
        if calibrated_signals is None:
            calibrated_signals = self._calibrate_signals(data_bundle, branch_results)
        consensus_scores = self._aggregate_symbol_scores(calibrated_signals)
        current_prices = data_bundle.latest_prices()
        predicted_returns = self._aggregate_expected_returns(calibrated_signals)
        macro_overlay = self._macro_overlay_factor(calibrated_signals.get("macro"))
        predicted_returns = {
            symbol: value * macro_overlay
            for symbol, value in predicted_returns.items()
        }
        predicted_volatilities = self._estimate_symbol_volatilities(data_bundle)
        portfolio_returns = self._build_portfolio_returns(data_bundle)
        covariance_matrix = self._build_covariance_matrix(data_bundle, list(predicted_returns.keys()))
        macro_signal = self._macro_signal_from_branch(branch_results.get("macro"))
        return self.risk_layer.run_risk_management(
            portfolio_returns=portfolio_returns,
            predicted_returns=predicted_returns,
            predicted_volatilities=predicted_volatilities,
            current_prices=current_prices,
            macro_signal=macro_signal,
            conviction_scores=consensus_scores,
            covariance_matrix=covariance_matrix,
        )

    def _aggregate_symbol_scores(
        self,
        calibrated_signals: dict[str, Any],
    ) -> dict[str, float]:
        """将多个分支的 symbol_scores 聚合为统一共识分数。"""
        branch_weights = self._branch_tracker.get_adaptive_weights()
        self._log("分支权重: " + ", ".join(f"{k}={v:.2f}" for k, v in branch_weights.items()))
        return self._aggregate_symbol_metric(
            calibrated_signals=calibrated_signals,
            metric_name="symbol_convictions",
            clamp_range=(-1.0, 1.0),
            branch_weights=branch_weights,
        )

    def _aggregate_expected_returns(
        self,
        calibrated_signals: dict[str, Any],
    ) -> dict[str, float]:
        """汇总各分支的校准后预期收益。"""
        branch_weights = self._branch_tracker.get_adaptive_weights()
        return self._aggregate_symbol_metric(
            calibrated_signals=calibrated_signals,
            metric_name="symbol_expected_returns",
            clamp_range=(-0.3, 0.3),
            branch_weights=branch_weights,
        )

    def _aggregate_symbol_metric(
        self,
        calibrated_signals: dict[str, Any],
        metric_name: str,
        clamp_range: tuple[float, float],
        branch_weights: dict[str, float],
    ) -> dict[str, float]:
        """按统一链路聚合个股级信号。"""
        weighted_sum = {symbol: 0.0 for symbol in self.stock_pool}
        weight_sum = {symbol: 0.0 for symbol in self.stock_pool}
        lower, upper = clamp_range

        for branch_name, branch in calibrated_signals.items():
            if branch_name == "macro":
                continue
            branch_weight = branch_weights.get(branch_name, 0.1)
            metric = getattr(branch, metric_name, {})
            for symbol in self.stock_pool:
                value = float(metric.get(symbol, 0.0))
                confidence = float(branch.symbol_confidences.get(symbol, 0.0))
                reliability_penalty = max(float(branch.reliability), 0.05)
                weight = branch_weight * reliability_penalty * max(confidence, 0.05)
                weighted_sum[symbol] += value * weight
                weight_sum[symbol] += weight

        return {
            symbol: _clamp(weighted_sum[symbol] / (weight_sum[symbol] + 1e-8), lower, upper)
            for symbol in self.stock_pool
        }

    @staticmethod
    def _macro_overlay_factor(macro_signal: Any | None) -> float:
        """宏观 overlay 只影响总预期收益和总仓位，不参与个股排序。"""
        if macro_signal is None:
            return 1.0
        macro_score = float(getattr(macro_signal, "aggregate_expected_return", 0.0))
        return _clamp(1.0 + macro_score * 1.5, 0.65, 1.15)

    def _estimate_symbol_volatilities(self, data_bundle: UnifiedDataBundle) -> dict[str, float]:
        vols: dict[str, float] = {}
        for symbol, df in data_bundle.symbol_data.items():
            if df.empty:
                vols[symbol] = 0.25
                continue
            ret = df["close"].pct_change().dropna()
            vols[symbol] = float(ret.tail(60).std() * np.sqrt(252)) if len(ret) >= 5 else 0.25
            if not np.isfinite(vols[symbol]) or vols[symbol] <= 0:
                vols[symbol] = 0.25
        return vols

    def _build_portfolio_returns(self, data_bundle: UnifiedDataBundle) -> pd.Series:
        aligned: list[pd.Series] = []
        for symbol, df in data_bundle.symbol_data.items():
            if df.empty:
                continue
            ret = df.set_index("date")["close"].pct_change().rename(symbol)
            aligned.append(ret)
        if not aligned:
            return pd.Series([0.0] * 30)
        combined = pd.concat(aligned, axis=1).dropna(how="all").fillna(0.0)
        return combined.mean(axis=1).tail(252)

    def _build_returns_matrix(self, data_bundle: UnifiedDataBundle) -> pd.DataFrame:
        aligned: list[pd.Series] = []
        for symbol, df in data_bundle.symbol_data.items():
            if df.empty:
                continue
            aligned.append(df.set_index("date")["close"].pct_change().rename(symbol))
        if not aligned:
            return pd.DataFrame()
        return pd.concat(aligned, axis=1).sort_index().fillna(0.0)

    def _build_covariance_matrix(
        self,
        data_bundle: UnifiedDataBundle,
        symbols: list[str],
    ) -> pd.DataFrame | None:
        returns_matrix = self._build_returns_matrix(data_bundle)
        if returns_matrix.empty or not symbols:
            return None
        usable = [symbol for symbol in symbols if symbol in returns_matrix.columns]
        if not usable:
            return None
        cov = PortfolioOptimizer.build_cov_from_returns(returns_matrix, usable)
        return pd.DataFrame(cov, index=usable, columns=usable)

    def _macro_signal_from_branch(self, branch: BranchResult | None) -> str:
        if branch is None:
            return "🟡"
        signal = branch.signals.get("liquidity_signal")
        if signal in {"🔴", "🟡", "🟢", "🔵"}:
            return str(signal)
        if branch.score >= 0.5:
            return "🟢"
        if branch.score <= -0.3:
            return "🔴"
        return "🟡"

    # ---------------------------------------------------------------------
    # 集成裁判层
    # ---------------------------------------------------------------------

    def _run_ensemble_layer(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        risk_result: RiskLayerResult,
        calibrated_signals: dict[str, Any] | None = None,
    ) -> PortfolioStrategy:
        """生成最终组合级策略。"""
        if calibrated_signals is None:
            calibrated_signals = self._calibrate_signals(data_bundle, branch_results)
        judge_output = EnsembleJudge.combine(branch_results, market_regime=self._market_regime)
        branch_consensus = judge_output["branch_consensus"]
        raw_symbol_convictions = self._aggregate_symbol_scores(calibrated_signals)
        expected_returns = self._aggregate_expected_returns(calibrated_signals)
        synthetic_symbols = set(data_bundle.synthetic_symbols())
        degraded_symbols = set(data_bundle.degraded_symbols())
        macro_overlay = self._macro_overlay_factor(calibrated_signals.get("macro"))
        candidate_symbols = [
            symbol
            for symbol, score in sorted(raw_symbol_convictions.items(), key=lambda item: item[1], reverse=True)
            if score > 0 and (self.allow_synthetic_for_research or symbol not in synthetic_symbols)
        ][: min(8, len(data_bundle.real_symbols()) or len(self.stock_pool))]

        research_score = float(judge_output["aggregate_score"])
        disagreement = (
            np.std(
                [
                    float(branch.final_score if branch.final_score is not None else branch.score)
                    for branch in branch_results.values()
                ]
            )
            if branch_results
            else 0.0
        )
        base_exposure = 1 - risk_result.position_sizing.cash_ratio
        exposure_penalty = _clamp(disagreement, 0.0, 0.35)
        target_exposure = _clamp(base_exposure * (1 - exposure_penalty), 0.1, 0.95)
        if synthetic_symbols and not self.allow_synthetic_for_research:
            target_exposure = min(target_exposure, 0.45)
        if risk_result.risk_level == "danger":
            target_exposure = min(target_exposure, 0.25)
        elif risk_result.risk_level == "warning":
            target_exposure = min(target_exposure, 0.55)
        if not data_bundle.real_symbols():
            target_exposure = 0.0
        target_exposure = _clamp(target_exposure * macro_overlay, 0.0, 0.95)

        # 分级 quorum 约束
        quorum_max = getattr(self, "_quorum_max_exposure", 0.95)
        quorum_rel = getattr(self, "_quorum_reliability", 1.0)
        target_exposure = min(target_exposure, quorum_max)
        if quorum_rel <= 0.0:
            target_exposure = 0.0

        style_bias = self._infer_style_bias(branch_results, research_score)
        sector_preferences = self._infer_sector_preferences(style_bias, branch_results.get("macro"))
        stop_loss_policy = {
            "portfolio_drawdown_limit": abs(self.risk_layer.max_drawdown_limit),
            "default_stop_loss_pct": abs(self.risk_layer.stop_loss_pct),
            "default_take_profit_pct": self.risk_layer.take_profit_pct,
        }
        position_limits = self._optimize_positions(
            data_bundle=data_bundle,
            candidate_symbols=candidate_symbols,
            expected_returns=expected_returns,
            convictions=raw_symbol_convictions,
            target_exposure=target_exposure,
            fallback_weights=risk_result.position_sizing.risk_adjusted_weights,
        )

        execution_notes = [
            f"研究共识得分 {research_score:+.2f}，分支分歧度 {disagreement:.2f}。",
            f"当前风险等级为 {risk_result.risk_level}，目标总仓位调整为 {target_exposure:.0%}。",
        ]
        if synthetic_symbols:
            execution_notes.append(
                "存在降级/模拟数据标的，已默认排除其进入候选池。"
                if not self.allow_synthetic_for_research else
                "存在降级/模拟数据标的，当前以研究模式允许其保留。"
            )
        for branch in branch_results.values():
            execution_notes.extend(branch.risks[:1])
        research_mode = "production"
        research_only_reason = ""
        if quorum_rel <= 0.0:
            research_mode = "research_only"
            research_only_reason = "quorum_insufficient"
            target_exposure = 0.0
            execution_notes.append("分支成功数不足（≤1），进入 research_only 模式。")
        elif synthetic_symbols or any(not branch.success for branch in branch_results.values()):
            research_mode = "degraded"
        if not candidate_symbols:
            execution_notes.append("暂无明确优势标的，建议以现金和防御仓位为主。")
            if not data_bundle.real_symbols():
                research_mode = "research_only"
                research_only_reason = "no_real_candidates_after_provenance_filter"
                target_exposure = 0.0
                position_limits = {}

        trade_recommendations = self._build_trade_recommendations(
            data_bundle=data_bundle,
            branch_results=branch_results,
            calibrated_signals=calibrated_signals,
            risk_result=risk_result,
            candidate_symbols=candidate_symbols,
            position_limits=position_limits,
            expected_returns=expected_returns,
            convictions=raw_symbol_convictions,
        )

        return PortfolioStrategy(
            **self._version_payload(),
            target_exposure=target_exposure,
            style_bias=style_bias,
            sector_preferences=sector_preferences,
            candidate_symbols=candidate_symbols,
            position_limits=position_limits,
            stop_loss_policy=stop_loss_policy,
            execution_notes=execution_notes[:8],
            branch_consensus=branch_consensus,
            symbol_convictions=raw_symbol_convictions,
            risk_summary={
                "risk_level": risk_result.risk_level,
                "volatility": risk_result.risk_metrics.volatility,
                "max_drawdown": risk_result.risk_metrics.max_drawdown,
                "warnings": risk_result.risk_warnings,
                "planned_investment": self.total_capital * target_exposure,
                "cash_reserve": self.total_capital * (1 - target_exposure),
                "max_single_position": self.risk_layer.max_position_size,
            },
            provenance_summary={
                "synthetic_symbols": sorted(synthetic_symbols),
                "degraded_symbols": sorted(degraded_symbols),
                "branch_modes": {
                    name: str(signal.branch_mode) for name, signal in calibrated_signals.items()
                },
                "branch_reliability": {
                    name: float(signal.reliability) for name, signal in calibrated_signals.items()
                },
                "ensemble_weights": judge_output["weights"],
                "ensemble_confidence": float(judge_output["aggregate_confidence"]),
                "macro_overlay": macro_overlay,
                "research_only_reason": research_only_reason,
                "symbol_provenance": data_bundle.symbol_provenance(),
                **self._version_payload(),
            },
            research_mode=research_mode,
            trade_recommendations=trade_recommendations,
        )

    def _optimize_positions(
        self,
        data_bundle: UnifiedDataBundle,
        candidate_symbols: list[str],
        expected_returns: dict[str, float],
        convictions: dict[str, float],
        target_exposure: float,
        fallback_weights: dict[str, float],
    ) -> dict[str, float]:
        """使用组合优化器生成最终仓位上限。"""
        if not candidate_symbols or target_exposure <= 0:
            return {}
        returns_matrix = self._build_returns_matrix(data_bundle)
        if returns_matrix.empty:
            return {
                symbol: round(min(fallback_weights.get(symbol, 0.0), self.risk_layer.max_position_size), 4)
                for symbol in candidate_symbols
            }
        optimizer = PortfolioOptimizer(
            max_position=self.risk_layer.max_position_size,
            method="risk_parity",
        )
        covariance_matrix = PortfolioOptimizer.build_cov_from_returns(returns_matrix, candidate_symbols)
        optimized = optimizer.optimize(
            candidate_symbols,
            expected_returns=expected_returns,
            cov_matrix=covariance_matrix,
            conviction_scores=convictions,
            base_position=target_exposure,
        )
        return {
            symbol: round(min(weight, self.risk_layer.max_position_size), 4)
            for symbol, weight in optimized.items()
            if weight > 0
        }

    def _build_trade_recommendations(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        risk_result: RiskLayerResult,
        candidate_symbols: list[str],
        position_limits: dict[str, float],
        expected_returns: dict[str, float],
        convictions: dict[str, float],
    ) -> list[TradeRecommendation]:
        """把组合结论展开为单票可执行交易建议。"""
        if not candidate_symbols or not position_limits:
            return []

        current_prices = data_bundle.latest_prices()
        stop_levels = risk_result.stop_loss_take_profit.stop_loss_levels
        take_profit_levels = risk_result.stop_loss_take_profit.take_profit_levels
        symbol_provenance = data_bundle.symbol_provenance()
        active_branch_count = max(len(branch_results), 1)
        recommendations: list[TradeRecommendation] = []

        for symbol in candidate_symbols:
            target_weight = float(position_limits.get(symbol, 0.0))
            current_price = float(current_prices.get(symbol, 0.0))
            if target_weight <= 0 or current_price <= 0:
                continue

            df = data_bundle.symbol_data.get(symbol, pd.DataFrame())
            support, resistance, atr, trend = self._derive_price_levels(df, current_price)
            model_expected_return = max(float(expected_returns.get(symbol, 0.0)), 0.0)
            conviction = float(convictions.get(symbol, 0.0))
            entry_price = self._suggest_entry_price(
                current_price=current_price,
                support_price=support,
                atr_value=atr,
                conviction=conviction,
                trend_regime=trend,
            )
            buy_zone_low = max(0.01, min(entry_price, support) * 0.995)
            buy_zone_high = max(entry_price, min(current_price * 1.01, entry_price + atr * 0.35))

            stop_loss_price = float(
                stop_levels.get(symbol, current_price * (1 + self.risk_layer.stop_loss_pct))
            )
            if support > 0:
                stop_loss_price = min(stop_loss_price, support * 0.99)
            stop_loss_price = _clamp(stop_loss_price, current_price * 0.75, current_price * 0.985)

            target_price = max(
                float(take_profit_levels.get(symbol, current_price * (1 + self.risk_layer.take_profit_pct))),
                current_price * (1 + max(model_expected_return, 0.08)),
                max(resistance, current_price + atr * 1.5),
            )
            target_price = max(target_price, entry_price * 1.05)

            lot_size = self._market_lot_size(symbol)
            budget_amount = self.total_capital * target_weight
            suggested_shares = int(budget_amount // max(entry_price, 0.01) // lot_size) * lot_size
            suggested_amount = suggested_shares * entry_price
            suggested_weight = suggested_amount / self.total_capital if self.total_capital > 0 else 0.0

            confidence = _safe_mean(
                [
                    float(signal.symbol_confidences.get(symbol, 0.0))
                    for name, signal in calibrated_signals.items()
                    if name != "macro"
                ]
            )
            horizon_days = int(
                max(
                    5,
                    round(
                        _safe_mean(
                            [
                                float(signal.horizon_days)
                                for name, signal in calibrated_signals.items()
                                if name != "macro"
                            ]
                        )
                    ),
                )
            )
            branch_scores = {
                name: float(branch.symbol_scores.get(symbol, branch.score))
                for name, branch in branch_results.items()
            }
            branch_expected_returns = {
                name: float(
                    signal.symbol_expected_returns.get(symbol, signal.aggregate_expected_return)
                )
                for name, signal in calibrated_signals.items()
            }
            branch_positive_count = sum(1 for score in branch_scores.values() if score > 0.05)

            expected_upside = target_price / max(entry_price, 0.01) - 1
            expected_drawdown = max(0.0, 1 - stop_loss_price / max(entry_price, 0.01))
            risk_reward_ratio = expected_upside / (expected_drawdown + 1e-8)
            risk_flags = self._collect_symbol_risks(
                symbol=symbol,
                data_bundle=data_bundle,
                branch_results=branch_results,
                branch_positive_count=branch_positive_count,
                active_branch_count=active_branch_count,
                confidence=confidence,
                expected_drawdown=expected_drawdown,
                trend_regime=trend,
            )
            position_management = self._build_position_management(
                symbol=symbol,
                suggested_weight=suggested_weight,
                suggested_amount=suggested_amount,
                buy_zone_low=buy_zone_low,
                buy_zone_high=buy_zone_high,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                branch_positive_count=branch_positive_count,
                active_branch_count=active_branch_count,
            )

            action = "buy" if suggested_shares > 0 else "watch"
            if suggested_shares == 0:
                risk_flags.append("按当前资金与整手规则暂无法成交，需提高单票预算或等待回落。")

            recommendations.append(
                TradeRecommendation(
                    symbol=symbol,
                    action=action,
                    category=self.market,
                    current_price=round(current_price, 2),
                    recommended_entry_price=round(entry_price, 2),
                    entry_price_range={
                        "low": round(buy_zone_low, 2),
                        "high": round(buy_zone_high, 2),
                    },
                    target_price=round(target_price, 2),
                    stop_loss_price=round(stop_loss_price, 2),
                    support_price=round(support, 2),
                    resistance_price=round(resistance, 2),
                    model_expected_return=round(model_expected_return, 4),
                    expected_upside=round(expected_upside, 4),
                    expected_drawdown=round(expected_drawdown, 4),
                    risk_reward_ratio=round(risk_reward_ratio, 2),
                    suggested_weight=round(suggested_weight, 4),
                    suggested_amount=round(suggested_amount, 2),
                    suggested_shares=suggested_shares,
                    lot_size=lot_size,
                    confidence=round(confidence, 4),
                    consensus_score=round(conviction, 4),
                    branch_positive_count=branch_positive_count,
                    branch_scores={k: round(v, 4) for k, v in branch_scores.items()},
                    branch_expected_returns={
                        k: round(v, 4) for k, v in branch_expected_returns.items()
                    },
                    risk_flags=risk_flags[:4],
                    position_management=position_management,
                    horizon_days=horizon_days,
                    trend_regime=trend,
                    data_source_status=str(
                        symbol_provenance.get(symbol, {}).get("data_source_status", "unknown")
                    ),
                )
            )

        return sorted(
            recommendations,
            key=lambda item: (
                item.suggested_weight,
                item.consensus_score,
                item.expected_upside,
            ),
            reverse=True,
        )

    def _derive_price_levels(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> tuple[float, float, float, str]:
        """基于近期价格行为估计支撑、阻力、ATR 与趋势状态。"""
        if df is None or df.empty or "close" not in df.columns:
            return current_price * 0.96, current_price * 1.08, max(current_price * 0.02, 0.01), "震荡"

        recent = df.sort_values("date").tail(60).copy()
        close = recent["close"].astype(float)
        high = recent["high"].astype(float) if "high" in recent.columns else close
        low = recent["low"].astype(float) if "low" in recent.columns else close
        prev_close = close.shift(1).fillna(close)
        true_range = np.maximum(
            high - low,
            np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
        )

        atr = float(true_range.tail(14).mean()) if len(true_range) >= 2 else current_price * 0.02
        atr = max(atr, current_price * 0.005)
        ma10 = float(close.tail(10).mean()) if len(close) >= 10 else current_price
        ma20 = float(close.tail(20).mean()) if len(close) >= 20 else ma10
        low20 = float(low.tail(20).min()) if len(low) >= 5 else current_price * 0.96
        high20 = float(high.tail(20).max()) if len(high) >= 5 else current_price * 1.06

        support = min(current_price, max(low20, ma20 - 0.75 * atr))
        resistance = max(high20, current_price + 1.5 * atr)

        if current_price > ma10 > ma20:
            trend = "上行"
        elif current_price < ma10 < ma20:
            trend = "承压"
        else:
            trend = "震荡"

        return support, resistance, atr, trend

    @staticmethod
    def _suggest_entry_price(
        current_price: float,
        support_price: float,
        atr_value: float,
        conviction: float,
        trend_regime: str,
    ) -> float:
        """结合趋势和回撤空间给出执行参考买点。"""
        conviction = max(conviction, 0.0)
        if trend_regime == "上行":
            raw_entry = current_price - max(atr_value * 0.45, current_price * 0.01)
        elif trend_regime == "承压":
            raw_entry = min(current_price * 0.98, support_price * 1.01)
        else:
            raw_entry = (current_price + support_price) / 2

        lower = max(0.01, support_price * (1.002 if conviction >= 0.25 else 0.995))
        upper = current_price * 1.005
        return round(_clamp(raw_entry, lower, upper), 2)

    def _collect_symbol_risks(
        self,
        symbol: str,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        branch_positive_count: int,
        active_branch_count: int,
        confidence: float,
        expected_drawdown: float,
        trend_regime: str,
    ) -> list[str]:
        """收集个股级风险观察。"""
        risk_flags: list[str] = []
        symbol_meta = data_bundle.symbol_provenance().get(symbol, {})
        if symbol_meta.get("data_source_status") != "real":
            risk_flags.append("数据链路并非纯真实行情，当前建议仅作研究参考。")
        consensus_floor = max(2, math.ceil(active_branch_count * 0.6))
        if branch_positive_count < consensus_floor:
            risk_flags.append("启用分支共识不足，执行上宜降级为观察或轻仓试错。")
        if confidence < 0.35:
            risk_flags.append("综合置信度一般，建议严格分批建仓。")
        if expected_drawdown > 0.10:
            risk_flags.append("止损空间偏大，单票仓位不宜激进。")
        if trend_regime == "承压":
            risk_flags.append("短线趋势仍承压，优先等待企稳信号。")

        fundamental_branch = branch_results.get("fundamental")
        if fundamental_branch is not None:
            fundamental_risks = fundamental_branch.signals.get("bear_case", {}).get(symbol, [])
            risk_flags.extend(str(item) for item in fundamental_risks[:2])
            risk_flags.extend(str(item) for item in fundamental_branch.debate_verdict.risk_flags[:1])

        intelligence_branch = branch_results.get("intelligence")
        if intelligence_branch is not None:
            alerts = intelligence_branch.signals.get("alerts", [])
            risk_flags.extend(str(item) for item in alerts if symbol in str(item))

        deduped: list[str] = []
        for item in risk_flags:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _build_position_management(
        symbol: str,
        suggested_weight: float,
        suggested_amount: float,
        buy_zone_low: float,
        buy_zone_high: float,
        target_price: float,
        stop_loss_price: float,
        branch_positive_count: int,
        active_branch_count: int,
    ) -> list[str]:
        """生成简洁的分批建仓与止盈止损动作。"""
        return [
            (
                f"{symbol} 建议总仓位 {suggested_weight:.1%}，预算约 ¥{suggested_amount:,.0f}，"
                "首次建仓 60%，余下 40% 仅在回踩买点区间时补齐。"
            ),
            (
                f"买入区间参考 ¥{buy_zone_low:.2f}-¥{buy_zone_high:.2f}；"
                f"若跌破 ¥{stop_loss_price:.2f} 则执行止损。"
            ),
            (
                f"若上行至 ¥{target_price:.2f} 附近先兑现 50%，"
                f"剩余仓位按分支共识 {branch_positive_count}/{active_branch_count} 持续跟踪。"
            ),
        ]

    def _market_lot_size(self, symbol: str) -> int:
        """返回市场整手单位。"""
        if self.market == "CN":
            return 100
        return 1

    def _infer_style_bias(
        self,
        branch_results: dict[str, BranchResult],
        research_score: float,
    ) -> str:
        quant_score = branch_results.get("quant", BranchResult("quant")).score
        fundamental_score = branch_results.get("fundamental", BranchResult("fundamental")).score
        macro_score = branch_results.get("macro", BranchResult("macro")).score
        if research_score < -0.2 or macro_score < -0.2:
            return "防御"
        if fundamental_score > 0.2 and quant_score < 0:
            return "高质量"
        if quant_score > 0.2:
            return "成长"
        return "均衡"

    def _infer_sector_preferences(
        self,
        style_bias: str,
        macro_branch: BranchResult | None,
    ) -> list[str]:
        if style_bias == "成长":
            prefs = ["科技成长", "先进制造", "景气消费"]
        elif style_bias == "高质量":
            prefs = ["高股息", "央国企龙头", "现金流稳健"]
        elif style_bias == "防御":
            prefs = ["公用事业", "必选消费", "红利低波"]
        else:
            prefs = ["均衡配置", "盈利改善", "估值合理"]
        if macro_branch and macro_branch.score < -0.2:
            return prefs[:2] + ["现金管理"]
        return prefs

    # ---------------------------------------------------------------------
    # 报告
    # ---------------------------------------------------------------------

    def _build_markdown_report(self, result: ResearchPipelineResult) -> str:
        strategy = result.final_strategy
        risk_result = result.risk_result
        active_branch_count = max(len(result.branch_results), 1)
        lines = [
            "# V9 五路并行研究投资策略报告",
            "",
            f"**市场**: {result.data_bundle.market}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**架构版本**: {result.architecture_version}",
            f"**分支 Schema**: {result.branch_schema_version}",
            f"**校准 Schema**: {result.calibration_schema_version}",
            f"**Debate 模板版本**: {result.debate_template_version}",
            "",
            "## 组合级策略结论",
            f"- 目标总仓位: {strategy.target_exposure:.0%}",
            f"- 风格偏好: {strategy.style_bias}",
            f"- 候选标的: {', '.join(strategy.candidate_symbols) if strategy.candidate_symbols else '暂无'}",
            f"- 行业偏好: {', '.join(strategy.sector_preferences) if strategy.sector_preferences else '均衡配置'}",
            "",
            "## 五大研究分支",
        ]

        for name in self.BRANCH_ORDER:
            branch = result.branch_results.get(name)
            if branch is None:
                continue
            lines.extend(
                [
                    f"### {branch.branch_name}",
                    f"- Base/Final 得分: {branch.base_score:+.2f} -> {branch.final_score:+.2f}",
                    f"- Base/Final 置信度: {branch.base_confidence:.0%} -> {branch.final_confidence:.0%}",
                    f"- 分支模式: {branch.metadata.get('branch_mode', 'unknown')}",
                    f"- 可靠度: {float(branch.metadata.get('reliability', 0.0)):.0%}",
                    f"- Debate 状态: {branch.metadata.get('debate_status', 'n/a') or 'n/a'}",
                    f"- 说明: {branch.explanation}",
                ]
            )
            if branch.risks:
                lines.append(f"- 关键风险: {'；'.join(branch.risks[:3])}")
            top_symbols = sorted(branch.symbol_scores.items(), key=lambda item: item[1], reverse=True)[:3]
            if top_symbols:
                lines.append(
                    "- 高优先级标的: "
                    + " / ".join([f"{symbol}({score:+.2f})" for symbol, score in top_symbols])
                )
            lines.append("")

        if risk_result is not None:
            lines.extend(
                [
                    "## 风控层",
                    f"- 风险等级: {risk_result.risk_level}",
                    f"- 年化波动率: {risk_result.risk_metrics.volatility:.2%}",
                    f"- 最大回撤: {risk_result.risk_metrics.max_drawdown:.2%}",
                    f"- 夏普比率: {risk_result.risk_metrics.sharpe_ratio:.2f}",
                    f"- 计划投入资金: ¥{strategy.risk_summary.get('planned_investment', 0.0):,.0f}",
                    f"- 计划保留现金: ¥{strategy.risk_summary.get('cash_reserve', 0.0):,.0f}",
                    "",
                ]
            )

        lines.extend(
            [
                "## 可信度与降级状态",
                f"- 研究模式: {strategy.research_mode}",
                f"- 使用模拟/降级数据的标的: {', '.join(strategy.provenance_summary.get('synthetic_symbols', [])) or '无'}",
                f"- 宏观 Overlay: {float(strategy.provenance_summary.get('macro_overlay', 1.0)):.2f}",
            ]
        )
        reason = strategy.provenance_summary.get("research_only_reason", "")
        if reason:
            lines.append(f"- 仅研究参考原因: {reason}")
        lines.append("- 分支模式与可靠度:")
        for branch_name in self.BRANCH_ORDER:
            if branch_name not in strategy.provenance_summary.get("branch_modes", {}):
                continue
            lines.append(
                "  "
                + f"{branch_name}: "
                + f"{strategy.provenance_summary['branch_modes'][branch_name]} / "
                + f"{float(strategy.provenance_summary.get('branch_reliability', {}).get(branch_name, 0.0)):.0%}"
            )
        for symbol, meta in sorted(result.data_bundle.symbol_provenance().items()):
            status = meta.get("data_source_status", "unknown")
            reason = meta.get("degraded_reason", "")
            reliability = float(meta.get("reliability", 0.0))
            suffix = f"；原因: {reason}" if reason else ""
            lines.append(f"- {symbol}: {status}，可靠度 {reliability:.0%}{suffix}")
        lines.append("")

        if strategy.trade_recommendations:
            lines.append("## 可执行交易计划")
            lines.append("| 标的 | 建议仓位 | 建议金额 | 建议股数 | 现价 | 建议买入 | 目标价 | 止损价 | 预期空间 | 分支支持 |")
            lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for recommendation in strategy.trade_recommendations:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            recommendation.symbol,
                            f"{recommendation.suggested_weight:.1%}",
                            f"¥{recommendation.suggested_amount:,.0f}",
                            f"{recommendation.suggested_shares:,}",
                            f"¥{recommendation.current_price:.2f}",
                            f"¥{recommendation.recommended_entry_price:.2f}",
                            f"¥{recommendation.target_price:.2f}",
                            f"¥{recommendation.stop_loss_price:.2f}",
                            f"{recommendation.expected_upside:.1%}",
                            f"{recommendation.branch_positive_count}/{active_branch_count}",
                        ]
                    )
                    + " |"
                )
            lines.append("")
            lines.append("## 个股执行与风险观察")
            for recommendation in strategy.trade_recommendations[:5]:
                lines.append(f"### {recommendation.symbol}")
                lines.append(
                    f"- 买入区间: ¥{recommendation.entry_price_range.get('low', 0.0):.2f}"
                    f" - ¥{recommendation.entry_price_range.get('high', 0.0):.2f}"
                )
                lines.append(
                    f"- 建议配置: {recommendation.suggested_weight:.1%}"
                    f" / ¥{recommendation.suggested_amount:,.0f}"
                    f" / {recommendation.suggested_shares:,} 股"
                )
                lines.append(
                    f"- 目标/止损: ¥{recommendation.target_price:.2f}"
                    f" / ¥{recommendation.stop_loss_price:.2f}"
                )
                lines.append(
                    f"- 分支共识: {recommendation.consensus_score:+.2f}，"
                    f"支持分支 {recommendation.branch_positive_count}/{active_branch_count}，"
                    f"综合置信度 {recommendation.confidence:.0%}"
                )
                if recommendation.risk_flags:
                    lines.append(f"- 风险观察: {'；'.join(recommendation.risk_flags[:3])}")
                if recommendation.position_management:
                    lines.append(f"- 仓位管理: {'；'.join(recommendation.position_management[:2])}")
                lines.append("")

        lines.append("## 执行建议")
        for note in strategy.execution_notes:
            lines.append(f"- {note}")

        lines.append("")
        lines.append("## 风控约束")
        lines.append(f"- 单票上限: {self.risk_layer.max_position_size:.0%}")
        lines.append(f"- 组合回撤红线: {abs(self.risk_layer.max_drawdown_limit):.0%}")
        lines.append(f"- 默认止损: {abs(self.risk_layer.stop_loss_pct):.0%}")

        return "\n".join(lines)

    @staticmethod
    def _default_branch_mode(branch_name: str) -> str:
        return {
            "kline": "kline_dual_model",
            "quant": "alpha_research",
            "llm_debate": "structured_research_debate",
            "fundamental": "fundamental_snapshot_fusion",
            "intelligence": "structured_intelligence_fusion",
            "macro": "macro_terminal",
        }.get(branch_name, "unknown")

    @staticmethod
    def _bundle_data_source_status(symbol_provenance: dict[str, dict[str, Any]]) -> str:
        statuses = {meta.get("data_source_status", "unknown") for meta in symbol_provenance.values()}
        if not statuses:
            return "unknown"
        if statuses == {"real"}:
            return "real"
        if statuses <= {"synthetic_fallback"}:
            return "synthetic_only"
        return "mixed"

    def _bundle_research_mode(self, symbol_provenance: dict[str, dict[str, Any]]) -> str:
        if not symbol_provenance:
            return "research_only"
        if all(meta.get("is_synthetic") for meta in symbol_provenance.values()):
            return "research_only"
        if any(meta.get("is_synthetic") for meta in symbol_provenance.values()):
            return "degraded"
        return "production"

    @staticmethod
    def _branch_data_source_status(data_bundle: UnifiedDataBundle) -> str:
        if not data_bundle.symbol_provenance():
            return "unknown"
        statuses = {meta.get("data_source_status", "unknown") for meta in data_bundle.symbol_provenance().values()}
        if statuses == {"real"}:
            return "real"
        if statuses <= {"synthetic_fallback"}:
            return "synthetic_only"
        return "mixed"
