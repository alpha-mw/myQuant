"""
V12 Agent 编排器。

移除了分支 SubAgent 层，Master Agent 直接接收5个分支的原始量化数据、
风控结果和过往交易记录，在内部进行五轮多空辩论后产出最终投资决策。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Any

from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    MasterAgentInput,
    MasterAgentOutput,
)
from quant_investor.agents.llm_client import LLMClient, has_any_provider
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.branch_contracts import BranchResult, PortfolioStrategy, UnifiedDataBundle
from quant_investor.logger import get_logger

_logger = get_logger("AgentOrchestrator")


class AgentOrchestrator:
    """V12 Agent 层编排器：直接调用 IC Master Agent，无中间 SubAgent 层。"""

    _MASTER_TIMEOUT_CUSHION_SECONDS = 15.0

    def __init__(
        self,
        branch_model: str,
        master_model: str,
        timeout_per_agent: float = 15.0,
        master_timeout: float = 30.0,
        total_timeout: float = 120.0,
        max_tokens_branch: int = 800,
        max_tokens_master: int = 2000,
    ) -> None:
        self.branch_model = branch_model
        self.master_model = master_model
        self.timeout_per_agent = timeout_per_agent
        self.master_timeout = master_timeout
        self.total_timeout = total_timeout
        self.max_tokens_branch = max_tokens_branch
        self.max_tokens_master = max_tokens_master

    @staticmethod
    def compute_outer_timeout(
        timeout_seconds: float,
        *,
        max_retries: int = 2,
        cushion_seconds: float = 10.0,
    ) -> float:
        attempts = max(1, int(max_retries))
        return float(timeout_seconds) * attempts + 1.0 + float(cushion_seconds)

    @classmethod
    def compute_recommended_total_timeout(
        cls,
        *,
        timeout_per_agent: float,
        master_timeout: float,
        existing_total_timeout: float,
        branch_max_retries: int = 2,
        master_max_retries: int = 2,
    ) -> float:
        master_budget = cls.compute_outer_timeout(
            master_timeout,
            max_retries=master_max_retries,
            cushion_seconds=cls._MASTER_TIMEOUT_CUSHION_SECONDS,
        )
        recommended = master_budget + max(float(timeout_per_agent), 30.0)
        return max(float(existing_total_timeout), recommended)

    async def enhance(
        self,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        risk_result: Any,
        ensemble_output: dict[str, Any],
        data_bundle: UnifiedDataBundle,
        market_regime: str,
        algorithmic_strategy: PortfolioStrategy | None = None,
        recall_context: dict[str, Any] | None = None,
    ) -> AgentEnhancedStrategy:
        """异步执行 IC Master Agent 并返回增强策略。"""
        t0 = time.monotonic()
        timings: dict[str, float] = {}

        if not has_any_provider():
            _logger.warning("No LLM provider available, skipping agent layer")
            return self._fallback_strategy(algorithmic_strategy)

        master_llm = LLMClient(timeout=self.master_timeout)

        # --- 直接调用 IC Master Agent ---
        master_output: MasterAgentOutput | None = None
        t_master = time.monotonic()
        try:
            master_output = await self._run_master_agent(
                branch_results=branch_results,
                risk_result=risk_result,
                ensemble_output=ensemble_output,
                market_regime=market_regime,
                candidate_symbols=list(data_bundle.symbol_data.keys()) if data_bundle else [],
                llm_client=master_llm,
                recall_context=dict(recall_context or {}),
            )
        except Exception as exc:
            _logger.warning(f"Master agent (IC) failed: {exc}")
        timings["master_agent"] = time.monotonic() - t_master
        timings["total_agent_layer"] = time.monotonic() - t0
        _logger.info(f"Agent layer completed in {timings['total_agent_layer']:.1f}s")

        algo_dict: dict[str, Any] = {}
        if algorithmic_strategy:
            try:
                algo_dict = asdict(algorithmic_strategy)
            except Exception:
                algo_dict = {"note": "serialization_failed"}

        return AgentEnhancedStrategy(
            algorithmic_strategy=algo_dict,
            agent_strategy=master_output,
            agent_layer_success=master_output is not None,
            agent_layer_timings=timings,
            fallback_used=master_output is None,
            branch_agent_outputs={},
            risk_agent_output=None,
        )

    def enhance_sync(self, **kwargs: Any) -> AgentEnhancedStrategy:
        """Sync wrapper for enhance(), compatible with V9's synchronous pipeline."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.enhance(**kwargs))
                return future.result(timeout=self.total_timeout)
        else:
            return asyncio.run(self.enhance(**kwargs))

    # --- Internal helpers ---

    @staticmethod
    def _serialize_branch_result(br: BranchResult) -> dict[str, Any]:
        """将 BranchResult 序列化为适合传给 LLM 的字典。"""
        evidence: dict[str, Any] = {}
        if br.evidence:
            evidence = {
                "summary": str(br.evidence.summary)[:600],
                "bull_points": list(br.evidence.bull_points[:5]),
                "bear_points": list(br.evidence.bear_points[:5]),
                "risk_points": list(br.evidence.risk_points[:5]),
            }

        signals: dict[str, Any] = {}
        if isinstance(br.signals, dict):
            for k, v in br.signals.items():
                if isinstance(v, (int, float, str, bool)):
                    signals[k] = v
                elif isinstance(v, dict):
                    signals[k] = {
                        sk: sv for sk, sv in v.items()
                        if isinstance(sv, (int, float, str, bool))
                    }

        return {
            "branch_name": br.branch_name,
            "base_score": float(br.base_score if br.base_score is not None else br.score),
            "final_score": float(br.final_score if br.final_score is not None else br.score),
            "confidence": float(
                br.final_confidence if br.final_confidence is not None else br.confidence
            ),
            "symbol_scores": dict(br.symbol_scores or {}),
            "signals": signals,
            "evidence": evidence,
        }

    @staticmethod
    def _serialize_risk_result(risk_result: Any) -> dict[str, Any]:
        """将风控层结果序列化为适合传给 LLM 的字典。"""
        if risk_result is None:
            return {}
        try:
            if hasattr(risk_result, "__dict__"):
                return {
                    k: v for k, v in risk_result.__dict__.items()
                    if isinstance(v, (int, float, str, bool, list, dict))
                }
            if isinstance(risk_result, dict):
                return risk_result
        except Exception:
            pass
        return {}

    async def _run_master_agent(
        self,
        branch_results: dict[str, BranchResult],
        risk_result: Any,
        ensemble_output: dict[str, Any],
        market_regime: str,
        candidate_symbols: list[str],
        llm_client: LLMClient,
        recall_context: dict[str, Any] | None = None,
    ) -> MasterAgentOutput:
        serialized_branches = {
            name: self._serialize_branch_result(br)
            for name, br in branch_results.items()
            if br is not None
        }

        master_input = MasterAgentInput(
            branch_results=serialized_branches,
            risk_result=self._serialize_risk_result(risk_result),
            ensemble_baseline=ensemble_output,
            market_regime=market_regime,
            candidate_symbols=candidate_symbols,
            recall_context=dict(recall_context or {}),
        )

        agent = MasterAgent(
            llm_client=llm_client,
            model=self.master_model,
            timeout=self.master_timeout,
            max_tokens=self.max_tokens_master,
        )
        return await asyncio.wait_for(
            agent.deliberate(master_input),
            timeout=self.master_timeout + 10.0,
        )

    @staticmethod
    def _fallback_strategy(
        algorithmic_strategy: PortfolioStrategy | None = None,
    ) -> AgentEnhancedStrategy:
        algo_dict: dict[str, Any] = {}
        if algorithmic_strategy:
            try:
                algo_dict = asdict(algorithmic_strategy)
            except Exception:
                algo_dict = {}
        return AgentEnhancedStrategy(
            algorithmic_strategy=algo_dict,
            agent_strategy=None,
            agent_layer_success=False,
            agent_layer_timings={},
            fallback_used=True,
            branch_agent_outputs={},
            risk_agent_output=None,
        )
