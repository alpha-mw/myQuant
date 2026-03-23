"""
V10 Agent 编排器。

异步协调 5 个分支 SubAgent、1 个风控 SubAgent 和 1 个 IC Master Agent，
将量化管线输出增强为 Agent-Enhanced 投资策略。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Any

from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BranchAgentInput,
    BranchAgentOutput,
    MasterAgentInput,
    MasterAgentOutput,
    RiskAgentInput,
    RiskAgentOutput,
)
from quant_investor.agents.llm_client import LLMClient, LLMCallError, has_any_provider
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.subagent import BranchSubAgent, RiskSubAgent
from quant_investor.branch_contracts import BranchResult, PortfolioStrategy, UnifiedDataBundle
from quant_investor.logger import get_logger
from quant_investor.versioning import CURRENT_BRANCH_ORDER

_logger = get_logger("AgentOrchestrator")


class AgentOrchestrator:
    """V10 Agent 层编排器：协调所有 SubAgent 并汇总到 IC。"""

    def __init__(
        self,
        branch_model: str,
        master_model: str,
        timeout_per_agent: float = 15.0,
        master_timeout: float = 30.0,
        total_timeout: float = 120.0,
        max_tokens_branch: int = 800,
        max_tokens_master: int = 1500,
    ) -> None:
        self.branch_model = branch_model
        self.master_model = master_model
        self.timeout_per_agent = timeout_per_agent
        self.master_timeout = master_timeout
        self.total_timeout = total_timeout
        self.max_tokens_branch = max_tokens_branch
        self.max_tokens_master = max_tokens_master

    async def enhance(
        self,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        risk_result: Any,
        ensemble_output: dict[str, Any],
        data_bundle: UnifiedDataBundle,
        market_regime: str,
        algorithmic_strategy: PortfolioStrategy | None = None,
    ) -> AgentEnhancedStrategy:
        """异步执行全部 agent 层并返回增强策略。"""
        t0 = time.monotonic()
        timings: dict[str, float] = {}

        if not has_any_provider():
            _logger.warning("No LLM provider available, skipping agent layer")
            return self._fallback_strategy(algorithmic_strategy)

        llm_client = LLMClient(timeout=self.timeout_per_agent)
        master_llm = LLMClient(timeout=self.master_timeout)

        # --- Phase 1: 5 branch SubAgents concurrently ---
        branch_agent_outputs: dict[str, BranchAgentOutput | None] = {}
        branch_inputs = self._prepare_branch_inputs(
            branch_results, calibrated_signals, market_regime,
        )

        t_branch = time.monotonic()
        branch_tasks = {
            name: self._run_branch_agent(name, inp, llm_client)
            for name, inp in branch_inputs.items()
        }
        results = await asyncio.gather(*branch_tasks.values(), return_exceptions=True)
        for name, result in zip(branch_tasks.keys(), results):
            if isinstance(result, Exception):
                _logger.warning(f"Branch agent [{name}] failed: {result}")
                branch_agent_outputs[name] = None
            else:
                branch_agent_outputs[name] = result
        timings["branch_agents"] = time.monotonic() - t_branch

        successful_branches = {k: v for k, v in branch_agent_outputs.items() if v is not None}
        _logger.info(f"Branch agents: {len(successful_branches)}/{len(branch_inputs)} succeeded")

        # --- Phase 2: Risk SubAgent ---
        risk_agent_output: RiskAgentOutput | None = None
        t_risk = time.monotonic()
        try:
            risk_agent_output = await self._run_risk_agent(
                risk_result, market_regime, successful_branches, llm_client,
            )
        except Exception as exc:
            _logger.warning(f"Risk agent failed: {exc}")
        timings["risk_agent"] = time.monotonic() - t_risk

        # --- Phase 3: Master Agent (IC) ---
        master_output: MasterAgentOutput | None = None
        t_master = time.monotonic()
        if successful_branches:
            try:
                master_output = await self._run_master_agent(
                    successful_branches,
                    risk_agent_output,
                    ensemble_output,
                    market_regime,
                    list(data_bundle.symbol_data.keys()) if data_bundle else [],
                    master_llm,
                )
            except Exception as exc:
                _logger.warning(f"Master agent (IC) failed: {exc}")
        else:
            _logger.warning("No successful branch agents, skipping IC deliberation")
        timings["master_agent"] = time.monotonic() - t_master

        timings["total_agent_layer"] = time.monotonic() - t0
        _logger.info(f"Agent layer completed in {timings['total_agent_layer']:.1f}s")

        algo_dict = {}
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
            branch_agent_outputs=branch_agent_outputs,
            risk_agent_output=risk_agent_output,
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

    def _prepare_branch_inputs(
        self,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        market_regime: str,
    ) -> dict[str, BranchAgentInput]:
        """将 BranchResult 转换为 BranchAgentInput。"""
        inputs: dict[str, BranchAgentInput] = {}
        for name in CURRENT_BRANCH_ORDER:
            br = branch_results.get(name)
            if br is None:
                continue

            base_score = float(br.base_score if br.base_score is not None else br.score)
            final_score = float(br.final_score if br.final_score is not None else br.score)
            confidence = float(br.final_confidence if br.final_confidence is not None else br.confidence)

            evidence = br.evidence
            evidence_summary = evidence.summary if evidence else (br.explanation or "")
            bull_points = list(evidence.bull_points) if evidence else []
            bear_points = list(evidence.bear_points) if evidence else []
            risk_points = list(evidence.risk_points) if evidence else []
            used_features = list(evidence.used_features) if evidence else []

            cal = calibrated_signals.get(name, {})
            expected_return = 0.0
            if hasattr(cal, "expected_return"):
                expected_return = float(cal.expected_return)
            elif isinstance(cal, dict):
                expected_return = float(cal.get("expected_return", 0.0))

            # Compact signals summary (avoid sending full dataframes)
            signals_summary: dict[str, Any] = {}
            if isinstance(br.signals, dict):
                for k, v in br.signals.items():
                    if isinstance(v, (int, float, str, bool)):
                        signals_summary[k] = v
                    elif isinstance(v, dict) and len(v) <= 20:
                        signals_summary[k] = {
                            sk: sv for sk, sv in v.items()
                            if isinstance(sv, (int, float, str, bool))
                        }

            inputs[name] = BranchAgentInput(
                branch_name=name,
                base_score=base_score,
                final_score=final_score,
                confidence=confidence,
                evidence_summary=str(evidence_summary)[:500],
                bull_points=bull_points[:5],
                bear_points=bear_points[:5],
                risk_points=risk_points[:5],
                used_features=used_features[:10],
                symbol_scores=dict(br.symbol_scores) if br.symbol_scores else {},
                market_regime=market_regime,
                calibrated_expected_return=expected_return,
                branch_signals=signals_summary,
            )
        return inputs

    async def _run_branch_agent(
        self,
        branch_name: str,
        agent_input: BranchAgentInput,
        llm_client: LLMClient,
    ) -> BranchAgentOutput:
        agent = BranchSubAgent(
            branch_name=branch_name,
            llm_client=llm_client,
            model=self.branch_model,
            timeout=self.timeout_per_agent,
            max_tokens=self.max_tokens_branch,
        )
        return await asyncio.wait_for(
            agent.analyze(agent_input),
            timeout=self.timeout_per_agent + 5.0,
        )

    async def _run_risk_agent(
        self,
        risk_result: Any,
        market_regime: str,
        branch_outputs: dict[str, BranchAgentOutput],
        llm_client: LLMClient,
    ) -> RiskAgentOutput:
        # Extract risk metrics summary
        risk_summary: dict[str, Any] = {}
        if risk_result is not None:
            try:
                if hasattr(risk_result, "__dict__"):
                    for k, v in risk_result.__dict__.items():
                        if isinstance(v, (int, float, str, bool)):
                            risk_summary[k] = v
                elif isinstance(risk_result, dict):
                    risk_summary = {
                        k: v for k, v in risk_result.items()
                        if isinstance(v, (int, float, str, bool))
                    }
            except Exception:
                risk_summary = {"note": "risk_result_extraction_failed"}

        risk_input = RiskAgentInput(
            risk_metrics_summary=risk_summary,
            regime=market_regime,
            position_sizing={},
            branch_agent_summaries=branch_outputs,
            portfolio_level_risks=[],
        )

        agent = RiskSubAgent(
            llm_client=llm_client,
            model=self.branch_model,
            timeout=self.timeout_per_agent,
            max_tokens=self.max_tokens_branch,
        )
        return await asyncio.wait_for(
            agent.analyze(risk_input),
            timeout=self.timeout_per_agent + 5.0,
        )

    async def _run_master_agent(
        self,
        branch_outputs: dict[str, BranchAgentOutput],
        risk_output: RiskAgentOutput | None,
        ensemble_output: dict[str, Any],
        market_regime: str,
        candidate_symbols: list[str],
        llm_client: LLMClient,
    ) -> MasterAgentOutput:
        master_input = MasterAgentInput(
            branch_reports=branch_outputs,
            risk_report=risk_output,
            ensemble_baseline=ensemble_output,
            market_regime=market_regime,
            candidate_symbols=candidate_symbols,
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
        algo_dict = {}
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
