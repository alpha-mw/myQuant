"""
V10.1 Agent 编排器。

异步协调 5 个专属分支 SubAgent、1 个风控 SubAgent 和 1 个 IC Master Agent，
将量化管线输出增强为 Agent-Enhanced 投资策略。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from itertools import combinations
from typing import Any

from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    FundamentalAgentInput,
    IntelligenceAgentInput,
    KLineAgentInput,
    MacroAgentInput,
    MasterAgentInput,
    MasterAgentOutput,
    QuantAgentInput,
    RiskAgentInput,
    RiskAgentOutput,
)
from quant_investor.agents.llm_client import (
    LLMClient,
    has_any_provider,
    has_provider_for_model,
)
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.prompts import MASTER_AGENT_PROFILE, RISK_AGENT_PROFILE, format_agent_display_name
from quant_investor.agents.subagent import BaseSubAgent
from quant_investor.agents.subagents import (
    FundamentalSubAgent,
    IntelligenceSubAgent,
    KLineSubAgent,
    MacroSubAgent,
    QuantSubAgent,
    SpecializedRiskSubAgent,
)
from quant_investor.branch_contracts import BranchResult, PortfolioStrategy, UnifiedDataBundle
from quant_investor.logger import get_logger
from quant_investor.versioning import CURRENT_BRANCH_ORDER

_logger = get_logger("AgentOrchestrator")

# ---------------------------------------------------------------------------
# Agent registry: branch_name → specialized SubAgent class
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: dict[str, type[BaseSubAgent]] = {
    "kline": KLineSubAgent,
    "quant": QuantSubAgent,
    "fundamental": FundamentalSubAgent,
    "intelligence": IntelligenceSubAgent,
    "macro": MacroSubAgent,
}


class AgentOrchestrator:
    """V10.1 Agent 层编排器：协调所有专属 SubAgent 并汇总到 IC。"""

    def __init__(
        self,
        branch_model: str,
        master_model: str,
        timeout_per_agent: float = 15.0,
        master_timeout: float = 30.0,
        total_timeout: float = 120.0,
        max_tokens_branch: int = 1000,
        max_tokens_master: int = 2000,
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
        if not has_provider_for_model(self.branch_model):
            _logger.warning(
                f"Provider for branch agent model [{self.branch_model}] is not configured, skipping agent layer"
            )
            return self._fallback_strategy(algorithmic_strategy)
        if not has_provider_for_model(self.master_model):
            _logger.warning(
                f"Provider for master agent model [{self.master_model}] is not configured, skipping agent layer"
            )
            return self._fallback_strategy(algorithmic_strategy)

        llm_client = LLMClient(timeout=self.timeout_per_agent)
        master_llm = LLMClient(timeout=self.master_timeout)

        # --- Phase 1: 5 specialized branch SubAgents concurrently ---
        branch_agent_outputs: dict[str, BaseBranchAgentOutput | None] = {}
        branch_inputs = self._prepare_specialized_inputs(
            branch_results, calibrated_signals, market_regime,
        )

        t_branch = time.monotonic()
        branch_tasks = {
            name: self._run_branch_agent(name, inp, llm_client)
            for name, inp in branch_inputs.items()
        }
        if branch_tasks:
            branch_labels = ", ".join(format_agent_display_name(name) for name in branch_tasks)
            _logger.info(f"Launching specialized SubAgents: {branch_labels}")

        results = await asyncio.gather(*branch_tasks.values(), return_exceptions=True)
        for name, result in zip(branch_tasks.keys(), results):
            if isinstance(result, BaseException):
                _logger.warning(f"Branch agent [{name}] failed: {result}")
                branch_agent_outputs[name] = None
            else:
                branch_agent_outputs[name] = result
        timings["branch_agents"] = time.monotonic() - t_branch

        successful_branches = {k: v for k, v in branch_agent_outputs.items() if v is not None}
        _logger.info(f"Branch agents: {len(successful_branches)}/{len(branch_inputs)} succeeded")

        # --- Phase 2: Risk SubAgent (with branch disagreement) ---
        risk_agent_output: RiskAgentOutput | None = None
        t_risk = time.monotonic()
        try:
            _logger.info(
                "Launching %s（%s）",
                RISK_AGENT_PROFILE["agent_name"],
                RISK_AGENT_PROFILE["specialist"],
            )
            risk_agent_output = await self._run_risk_agent(
                risk_result, market_regime, successful_branches, llm_client,
            )
        except Exception as exc:
            _logger.warning(f"Risk agent failed: {exc}")
        timings["risk_agent"] = time.monotonic() - t_risk

        # --- Phase 3: Compute disagreement matrix + Master Agent (IC) ---
        disagreement_matrix = self._compute_disagreement_matrix(successful_branches)
        master_output: MasterAgentOutput | None = None
        t_master = time.monotonic()
        if successful_branches:
            try:
                _logger.info(
                    "Launching %s（%s）",
                    MASTER_AGENT_PROFILE["agent_name"],
                    MASTER_AGENT_PROFILE["specialist"],
                )
                master_output = await self._run_master_agent(
                    successful_branches,
                    risk_agent_output,
                    ensemble_output,
                    market_regime,
                    list(data_bundle.symbol_data.keys()) if data_bundle else [],
                    master_llm,
                    disagreement_matrix,
                )
            except Exception as exc:
                _logger.warning(f"Master agent (IC) failed: {exc}")
        else:
            _logger.warning("No successful branch agents, skipping IC deliberation")
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

    # -----------------------------------------------------------------------
    # Specialized input preparation
    # -----------------------------------------------------------------------

    def _prepare_specialized_inputs(
        self,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        market_regime: str,
    ) -> dict[str, BaseBranchAgentInput]:
        """将 BranchResult 转换为各分支的专属 AgentInput。"""
        inputs: dict[str, BaseBranchAgentInput] = {}
        for name in CURRENT_BRANCH_ORDER:
            br = branch_results.get(name)
            if br is None:
                continue

            # Common fields
            base_fields = self._extract_common_fields(br, calibrated_signals.get(name), market_regime)

            # Dispatch to specialized builder
            builder = {
                "kline": self._build_kline_input,
                "quant": self._build_quant_input,
                "fundamental": self._build_fundamental_input,
                "intelligence": self._build_intelligence_input,
                "macro": self._build_macro_input,
            }.get(name)

            if builder:
                inputs[name] = builder(base_fields, br)
            else:
                inputs[name] = BaseBranchAgentInput(**base_fields)

        return inputs

    def _extract_common_fields(
        self,
        br: BranchResult,
        cal: Any,
        market_regime: str,
    ) -> dict[str, Any]:
        """Extract common fields from BranchResult."""
        base_score = float(br.base_score if br.base_score is not None else br.score)
        final_score = float(br.final_score if br.final_score is not None else br.score)
        confidence = float(br.final_confidence if br.final_confidence is not None else br.confidence)

        evidence = br.evidence
        evidence_summary = evidence.summary if evidence else (br.explanation or "")
        bull_points = list(evidence.bull_points) if evidence else []
        bear_points = list(evidence.bear_points) if evidence else []
        risk_points = list(evidence.risk_points) if evidence else []
        used_features = list(evidence.used_features) if evidence else []

        expected_return = 0.0
        if hasattr(cal, "expected_return"):
            expected_return = float(cal.expected_return)
        elif isinstance(cal, dict):
            expected_return = float(cal.get("expected_return", 0.0))

        # Compact signals summary
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

        return {
            "branch_name": br.branch_name,
            "base_score": base_score,
            "final_score": final_score,
            "confidence": confidence,
            "evidence_summary": str(evidence_summary)[:500],
            "bull_points": bull_points[:5],
            "bear_points": bear_points[:5],
            "risk_points": risk_points[:5],
            "used_features": used_features[:10],
            "symbol_scores": dict(br.symbol_scores) if br.symbol_scores else {},
            "market_regime": market_regime,
            "calibrated_expected_return": expected_return,
            "branch_signals": signals_summary,
        }

    @staticmethod
    def _safe_get_signal(signals: dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely extract a signal value, returning default if not scalar-compatible."""
        val = signals.get(key, default)
        if val is None:
            return default
        return val

    def _build_kline_input(self, base: dict[str, Any], br: BranchResult) -> KLineAgentInput:
        signals = br.signals if isinstance(br.signals, dict) else {}
        predicted_returns = signals.get("predicted_return", {})
        if not isinstance(predicted_returns, dict):
            predicted_returns = {}

        return KLineAgentInput(
            **base,
            predicted_returns={k: float(v) for k, v in predicted_returns.items() if isinstance(v, (int, float))},
            kronos_confidence=float(signals.get("kronos_confidence", 0.0)),
            chronos_confidence=float(signals.get("chronos_confidence", 0.0)),
            model_agreement=float(signals.get("model_agreement", 0.0)),
            detected_regimes={k: str(v) for k, v in signals.get("detected_regimes", {}).items()} if isinstance(signals.get("detected_regimes"), dict) else {},
            trend_strength={k: float(v) for k, v in signals.get("trend_strength", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("trend_strength"), dict) else {},
            momentum_signals={k: float(v) for k, v in signals.get("momentum_signals", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("momentum_signals"), dict) else {},
            volatility_percentile=float(signals.get("volatility_percentile", 0.0) or signals.get("volatility", 50.0) or 50.0),
        )

    def _build_quant_input(self, base: dict[str, Any], br: BranchResult) -> QuantAgentInput:
        signals = br.signals if isinstance(br.signals, dict) else {}
        factor_exposures = signals.get("factor_exposures", {})
        if not isinstance(factor_exposures, dict):
            factor_exposures = {}

        return QuantAgentInput(
            **base,
            factor_exposures={k: v for k, v in factor_exposures.items() if isinstance(v, dict)},
            ic_metrics={k: float(v) for k, v in signals.get("ic_metrics", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("ic_metrics"), dict) else {},
            ir_metrics={k: float(v) for k, v in signals.get("ir_metrics", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("ir_metrics"), dict) else {},
            factor_decay_info={k: int(v) for k, v in signals.get("factor_decay_info", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("factor_decay_info"), dict) else {},
            crowding_signals={k: float(v) for k, v in signals.get("crowding_signals", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("crowding_signals"), dict) else {},
            alpha_candidates=signals.get("alpha_candidates", []) if isinstance(signals.get("alpha_candidates"), list) else [],
            regime_factor_effectiveness={k: float(v) for k, v in signals.get("regime_factor_effectiveness", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("regime_factor_effectiveness"), dict) else {},
        )

    def _build_fundamental_input(self, base: dict[str, Any], br: BranchResult) -> FundamentalAgentInput:
        signals = br.signals if isinstance(br.signals, dict) else {}

        return FundamentalAgentInput(
            **base,
            module_scores={k: float(v) for k, v in signals.get("component_scores", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("component_scores"), dict) else {},
            module_confidences={k: float(v) for k, v in signals.get("module_confidences", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("module_confidences"), dict) else {},
            module_coverages={k: str(v) for k, v in signals.get("module_coverages", {}).items()} if isinstance(signals.get("module_coverages"), dict) else {},
            financial_quality={k: v for k, v in signals.get("financial_quality", {}).items() if isinstance(v, dict)} if isinstance(signals.get("financial_quality"), dict) else {},
            forecast_revisions={k: v for k, v in signals.get("forecast_revisions", {}).items() if isinstance(v, dict)} if isinstance(signals.get("forecast_revisions"), dict) else {},
            valuation_metrics={k: v for k, v in signals.get("valuation_metrics", {}).items() if isinstance(v, dict)} if isinstance(signals.get("valuation_metrics"), dict) else {},
            governance_scores={k: float(v) for k, v in signals.get("governance_scores", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("governance_scores"), dict) else {},
            ownership_signals={k: v for k, v in signals.get("ownership_signals", {}).items() if isinstance(v, dict)} if isinstance(signals.get("ownership_signals"), dict) else {},
            doc_sentiment={k: float(v) for k, v in signals.get("doc_sentiment", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("doc_sentiment"), dict) else {},
            data_staleness_days={k: int(v) for k, v in signals.get("data_staleness_days", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("data_staleness_days"), dict) else {},
        )

    def _build_intelligence_input(self, base: dict[str, Any], br: BranchResult) -> IntelligenceAgentInput:
        signals = br.signals if isinstance(br.signals, dict) else {}

        # Extract fear-greed index (composite of sentiment signals)
        fear_greed = 50.0
        sentiment_score = signals.get("sentiment_score", {})
        if isinstance(sentiment_score, dict) and sentiment_score:
            avg_sentiment = sum(float(v) for v in sentiment_score.values() if isinstance(v, (int, float))) / max(len(sentiment_score), 1)
            fear_greed = max(0.0, min(100.0, (avg_sentiment + 1.0) * 50.0))
        elif isinstance(sentiment_score, (int, float)):
            fear_greed = max(0.0, min(100.0, (float(sentiment_score) + 1.0) * 50.0))

        return IntelligenceAgentInput(
            **base,
            event_risk_score=float(signals.get("event_risk_score", 0.0)) if isinstance(signals.get("event_risk_score"), (int, float)) else 0.0,
            event_catalysts=signals.get("alerts", []) if isinstance(signals.get("alerts"), list) else [],
            fear_greed_index=fear_greed,
            sentiment_extremes={k: float(v) for k, v in signals.get("sentiment_extremes", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("sentiment_extremes"), dict) else {},
            money_flow_signal=float(signals.get("money_flow_score", 0.0)) if isinstance(signals.get("money_flow_score"), (int, float)) else 0.0,
            smart_money_indicators={k: float(v) for k, v in signals.get("smart_money_indicators", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("smart_money_indicators"), dict) else {},
            market_breadth={k: float(v) for k, v in signals.get("breadth_score", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("breadth_score"), dict) else {},
            sector_rotation_signal=str(signals.get("rotation_signal", "neutral")),
        )

    def _build_macro_input(self, base: dict[str, Any], br: BranchResult) -> MacroAgentInput:
        signals = br.signals if isinstance(br.signals, dict) else {}

        return MacroAgentInput(
            **base,
            liquidity_score=float(signals.get("liquidity_score", 0.0) or signals.get("macro_score", 0.0) or 0.0) if isinstance(signals.get("liquidity_score", signals.get("macro_score", 0.0)), (int, float)) else 0.0,
            monetary_policy_signal=str(signals.get("policy_signal", "neutral")),
            macro_volatility_percentile=float(signals.get("volatility_percentile", 50.0)) if isinstance(signals.get("volatility_percentile"), (int, float)) else 50.0,
            volatility_term_structure=str(signals.get("vol_term_structure", "normal")),
            breadth_score=float(signals.get("breadth_score", 0.0)) if isinstance(signals.get("breadth_score"), (int, float)) else 0.0,
            momentum_structure={k: float(v) for k, v in signals.get("momentum_structure", {}).items() if isinstance(v, (int, float))} if isinstance(signals.get("momentum_structure"), dict) else {},
            cross_asset_signals={k: str(v) for k, v in signals.get("cross_asset_signals", {}).items()} if isinstance(signals.get("cross_asset_signals"), dict) else {},
            yield_curve_signal=str(signals.get("yield_curve_signal", "normal")),
            overall_risk_level=str(signals.get("risk_level", "normal")),
        )

    # -----------------------------------------------------------------------
    # Disagreement matrix
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_disagreement_matrix(
        branch_outputs: dict[str, BaseBranchAgentOutput],
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise disagreement between branch conviction scores."""
        matrix: dict[str, dict[str, float]] = {}
        branch_names = list(branch_outputs.keys())
        for b1, b2 in combinations(branch_names, 2):
            o1 = branch_outputs[b1]
            o2 = branch_outputs[b2]
            disagreement = abs(o1.conviction_score - o2.conviction_score)
            matrix.setdefault(b1, {})[b2] = round(disagreement, 4)
            matrix.setdefault(b2, {})[b1] = round(disagreement, 4)
        return matrix

    @staticmethod
    def _compute_branch_disagreement_level(
        branch_outputs: dict[str, BaseBranchAgentOutput],
    ) -> float:
        """Compute overall branch disagreement as std of conviction scores."""
        if len(branch_outputs) < 2:
            return 0.0
        scores = [o.conviction_score for o in branch_outputs.values()]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return float(min(1.0, variance ** 0.5))

    # -----------------------------------------------------------------------
    # Agent execution
    # -----------------------------------------------------------------------

    async def _run_branch_agent(
        self,
        branch_name: str,
        agent_input: BaseBranchAgentInput,
        llm_client: LLMClient,
    ) -> BaseBranchAgentOutput:
        agent_cls = _AGENT_REGISTRY.get(branch_name)
        if agent_cls is None:
            from quant_investor.agents.subagent import BranchSubAgent
            agent_cls = BranchSubAgent

        agent = agent_cls(
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
        branch_outputs: dict[str, BaseBranchAgentOutput],
        llm_client: LLMClient,
    ) -> RiskAgentOutput:
        # Extract risk metrics
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

        branch_disagreement = self._compute_branch_disagreement_level(branch_outputs)

        risk_input = RiskAgentInput(
            risk_metrics_summary=risk_summary,
            regime=market_regime,
            position_sizing={},
            branch_agent_summaries=branch_outputs,
            portfolio_level_risks=[],
            branch_disagreement_level=branch_disagreement,
        )

        agent = SpecializedRiskSubAgent(
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
        branch_outputs: dict[str, BaseBranchAgentOutput],
        risk_output: RiskAgentOutput | None,
        ensemble_output: dict[str, Any],
        market_regime: str,
        candidate_symbols: list[str],
        llm_client: LLMClient,
        disagreement_matrix: dict[str, dict[str, float]],
    ) -> MasterAgentOutput:
        master_input = MasterAgentInput(
            branch_reports=branch_outputs,
            risk_report=risk_output,
            ensemble_baseline=ensemble_output,
            market_regime=market_regime,
            candidate_symbols=candidate_symbols,
            branch_disagreement_matrix=disagreement_matrix,
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
