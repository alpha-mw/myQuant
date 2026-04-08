"""
V12 Agent 编排器。

移除了分支 SubAgent 层，Master Agent 直接接收5个分支的原始量化数据、
风控结果和过往交易记录，在内部进行五轮多空辩论后产出最终投资决策。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchOverlayVerdict,
    BranchVerdict,
    ConfidenceLabel,
    Direction,
    MasterICHint,
    ReviewTelemetry,
    RiskDecision,
    StockReviewBundle,
)
from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BranchAgentInput,
    BranchAgentOutput,
    MasterAgentInput,
    MasterAgentOutput,
    RiskAgentInput,
    RiskAgentOutput,
)
from quant_investor.agents.llm_client import LLMClient as LegacyLLMClient, has_any_provider
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.agents.stock_reviewers import (
    BranchOverlayPacket,
    BranchOverlayReviewer,
    MasterICAgent,
    MasterSymbolPacket,
)
from quant_investor.agents.subagent import BranchSubAgent, RiskSubAgent
from quant_investor.branch_contracts import BranchResult, PortfolioStrategy, UnifiedDataBundle
from quant_investor.llm_gateway import LLMClient as GatewayLLMClient
from quant_investor.logger import get_logger
from quant_investor.versioning import CURRENT_BRANCH_ORDER

_logger = get_logger("AgentOrchestrator")

_AGENT_REGISTRY: dict[str, type[BranchSubAgent]] = {
    name: BranchSubAgent
    for name in CURRENT_BRANCH_ORDER
}


class AgentOrchestrator:
    """V12 Agent 层编排器：直接调用 IC Master Agent，无中间 SubAgent 层。"""

    _MASTER_TIMEOUT_CUSHION_SECONDS = 15.0

    def __init__(
        self,
        branch_model: str,
        master_model: str,
        master_reasoning_effort: str = "high",
        branch_primary_model: str = "",
        branch_fallback_model: str = "",
        branch_fallback_used: bool = False,
        branch_fallback_reason: str = "",
        master_primary_model: str = "",
        master_fallback_model: str = "",
        master_fallback_used: bool = False,
        master_fallback_reason: str = "",
        universe_key: str = "full_a",
        timeout_per_agent: float = 15.0,
        master_timeout: float = 30.0,
        total_timeout: float = 120.0,
        max_tokens_branch: int = 800,
        max_tokens_master: int = 2000,
    ) -> None:
        self.branch_model = branch_model
        self.master_model = master_model
        self.master_reasoning_effort = str(master_reasoning_effort or "").strip() or "high"
        self.branch_primary_model = str(branch_primary_model or "").strip() or self.branch_model
        self.branch_fallback_model = str(branch_fallback_model or "").strip()
        self.branch_fallback_used = bool(branch_fallback_used)
        self.branch_fallback_reason = str(branch_fallback_reason or "").strip()
        self.master_primary_model = str(master_primary_model or "").strip() or self.master_model
        self.master_fallback_model = str(master_fallback_model or "").strip()
        self.master_fallback_used = bool(master_fallback_used)
        self.master_fallback_reason = str(master_fallback_reason or "").strip()
        self.universe_key = str(universe_key or "").strip() or "full_a"
        self.timeout_per_agent = timeout_per_agent
        self.master_timeout = master_timeout
        self.total_timeout = total_timeout
        self.max_tokens_branch = max_tokens_branch
        self.max_tokens_master = max_tokens_master
        self.portfolio_constructor = PortfolioConstructor()

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

    @staticmethod
    def branch_request_timeout(branch_name: str, timeout_per_agent: float) -> float:
        multiplier = 1.5 if branch_name == "kline" else 1.0
        return float(timeout_per_agent) * multiplier

    @staticmethod
    def branch_max_tokens(branch_name: str, max_tokens_branch: int) -> int:
        if branch_name == "kline":
            return min(int(max_tokens_branch), 600)
        return int(max_tokens_branch)

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
        candidate_symbols: list[str] | None = None,
    ) -> AgentEnhancedStrategy:
        """异步执行 IC Master Agent 并返回增强策略。"""
        t0 = time.monotonic()
        timings: dict[str, float] = {}

        if not has_any_provider():
            _logger.warning("No LLM provider available, skipping agent layer")
            return self._fallback_strategy(algorithmic_strategy)

        llm_client = LegacyLLMClient(timeout=self.timeout_per_agent)
        master_llm = LegacyLLMClient(
            timeout=self.master_timeout,
            default_reasoning_effort=self.master_reasoning_effort,
        )
        review_llm = GatewayLLMClient(timeout=self.timeout_per_agent)
        review_master_llm = GatewayLLMClient(
            timeout=self.master_timeout,
            default_reasoning_effort=self.master_reasoning_effort,
        )

        # Initialize review-layer variables
        branch_agent_outputs: dict[str, BranchAgentOutput | None] = {}
        risk_agent_output: RiskAgentOutput | None = None
        review_recall_context = dict(recall_context or {})

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

        # --- Phase 4: symbol-level review bundle ---
        review_bundle: StockReviewBundle | None = None
        symbol_review_bundle: dict[str, dict[str, Any]] = {}
        ic_hints_by_symbol: dict[str, dict[str, Any]] = {}
        t_review = time.monotonic()
        try:
            review_bundle = await self._run_symbol_review_layer(
                branch_results=branch_results,
                calibrated_signals=calibrated_signals,
                risk_result=risk_result,
                risk_agent_output=risk_agent_output,
                ensemble_output=ensemble_output,
                data_bundle=data_bundle,
                market_regime=market_regime,
                review_llm=review_llm,
                review_master_llm=review_master_llm,
                recall_context=review_recall_context,
                candidate_symbols=candidate_symbols,
            )
            symbol_review_bundle = self._serialize_review_map(
                review_bundle.branch_overlay_verdicts_by_symbol,
            )
            symbol_review_bundle = {
                symbol: {
                    "branch_overlays": symbol_review_bundle.get(symbol, {}),
                    "master_hint": self._serialize_review_item(review_bundle.master_hints_by_symbol.get(symbol)),
                    "ic_hint": dict(review_bundle.ic_hints_by_symbol.get(symbol, {})),
                }
                for symbol in sorted(review_bundle.ic_hints_by_symbol or review_bundle.master_hints_by_symbol)
            }
            ic_hints_by_symbol = dict(review_bundle.ic_hints_by_symbol)
        except Exception as exc:
            _logger.warning(f"Symbol review bundle failed: {exc}")
        timings["symbol_review"] = time.monotonic() - t_review

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
            review_bundle=review_bundle,
            symbol_review_bundle=symbol_review_bundle,
            ic_hints_by_symbol=ic_hints_by_symbol,
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

    async def _run_branch_agent(
        self,
        branch_name: str,
        agent_input: BranchAgentInput,
        llm_client: LegacyLLMClient,
    ) -> BranchAgentOutput:
        agent_cls = _AGENT_REGISTRY.get(branch_name, BranchSubAgent)
        branch_timeout = self.branch_request_timeout(branch_name, self.timeout_per_agent)
        agent = agent_cls(
            branch_name=branch_name,
            llm_client=LegacyLLMClient(timeout=branch_timeout),
            model=self.branch_model,
            timeout=branch_timeout,
            max_tokens=self.branch_max_tokens(branch_name, self.max_tokens_branch),
        )
        return await asyncio.wait_for(
            agent.analyze(agent_input),
            timeout=branch_timeout + 5.0,
        )

    async def _run_risk_agent(
        self,
        risk_result: Any,
        market_regime: str,
        branch_outputs: dict[str, BranchAgentOutput],
        llm_client: LegacyLLMClient,
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
        branch_results: dict[str, BranchResult],
        risk_result: Any,
        ensemble_output: dict[str, Any],
        market_regime: str,
        candidate_symbols: list[str],
        llm_client: LegacyLLMClient,
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
            reasoning_effort=self.master_reasoning_effort,
            timeout=self.master_timeout,
            max_tokens=self.max_tokens_master,
        )
        return await asyncio.wait_for(
            agent.deliberate(master_input),
            timeout=self.master_timeout + 10.0,
        )

    async def _run_symbol_review_layer(
        self,
        *,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        risk_result: Any,
        risk_agent_output: RiskAgentOutput | None,
        ensemble_output: dict[str, Any],
        data_bundle: UnifiedDataBundle,
        market_regime: str,
        review_llm: GatewayLLMClient,
        review_master_llm: GatewayLLMClient,
        recall_context: dict[str, Any] | None = None,
        candidate_symbols: list[str] | None = None,
    ) -> StockReviewBundle:
        macro_verdict = self._build_macro_verdict_from_branch_results(branch_results, data_bundle)
        risk_decision = self._coerce_risk_decision(risk_result)
        macro_summary = self._build_macro_summary(data_bundle, macro_verdict, ensemble_output, market_regime)
        risk_summary = self._build_risk_summary(risk_result, risk_agent_output, risk_decision)
        branch_names = [name for name in CURRENT_BRANCH_ORDER if name != "macro"]
        branch_overlay_verdicts_by_symbol: dict[str, dict[str, BranchOverlayVerdict]] = {}
        master_hints_by_symbol: dict[str, MasterICHint] = {}
        ic_hints_by_symbol: dict[str, dict[str, Any]] = {}
        telemetry: list[ReviewTelemetry] = []
        fallback_reasons: list[str] = []

        # When candidate_symbols is provided, only review those symbols.
        review_symbols = candidate_symbols if candidate_symbols is not None else data_bundle.symbols
        _logger.info(
            "Symbol review layer: reviewing %d/%d symbols",
            len(review_symbols),
            len(data_bundle.symbols),
        )

        branch_tasks: dict[tuple[str, str], asyncio.Task[BranchOverlayVerdict]] = {}
        for symbol in review_symbols:
            branch_overlay_verdicts_by_symbol[symbol] = {}
            for branch_name in branch_names:
                branch_result = branch_results.get(branch_name)
                if branch_result is None:
                    continue
                packet = self._build_branch_overlay_packet(
                    symbol=symbol,
                    branch_name=branch_name,
                    branch_result=branch_result,
                    calibrated_signal=calibrated_signals.get(branch_name, {}),
                    macro_summary=macro_summary,
                    risk_summary=risk_summary,
                    recall_context=recall_context,
                )
                reviewer = BranchOverlayReviewer(
                    branch_name=branch_name,
                    llm_client=review_llm,
                    model=self.branch_model,
                    timeout=self.timeout_per_agent,
                    max_tokens=self.branch_max_tokens(branch_name, self.max_tokens_branch),
                )
                branch_tasks[(symbol, branch_name)] = asyncio.create_task(reviewer.review(packet))

        if branch_tasks:
            results = await asyncio.gather(*branch_tasks.values(), return_exceptions=True)
            for (symbol, branch_name), result in zip(branch_tasks.keys(), results):
                if isinstance(result, Exception):
                    _logger.warning(f"Symbol branch overlay failed [{symbol}/{branch_name}]: {result}")
                    fallback_reasons.append(f"{symbol}/{branch_name}: {result}")
                    continue
                branch_overlay_verdicts_by_symbol[symbol][branch_name] = result
                telemetry.append(result.telemetry)
                if result.telemetry.fallback and result.telemetry.fallback_reason:
                    fallback_reasons.append(f"{symbol}/{branch_name}: {result.telemetry.fallback_reason}")

        for symbol in review_symbols:
            overlays = branch_overlay_verdicts_by_symbol.get(symbol, {})
            packet = self._build_master_symbol_packet(
                symbol=symbol,
                overlays=overlays,
                macro_summary=macro_summary,
                risk_summary=risk_summary,
                risk_decision=risk_decision,
                recall_context=recall_context,
            )
            reviewer = MasterICAgent(
                llm_client=review_master_llm,
                model=self.master_model,
                reasoning_effort=self.master_reasoning_effort,
                timeout=self.master_timeout,
                max_tokens=self.max_tokens_master,
            )
            try:
                hint = await reviewer.deliberate(packet)
            except Exception as exc:
                _logger.warning(f"Master symbol hint failed [{symbol}]: {exc}")
                fallback_reasons.append(f"{symbol}: {exc}")
                hint = self._deterministic_master_hint(packet, reason=str(exc))
            master_hints_by_symbol[symbol] = hint
            telemetry.append(hint.telemetry)
            if hint.telemetry.fallback and hint.telemetry.fallback_reason:
                fallback_reasons.append(f"{symbol}: {hint.telemetry.fallback_reason}")
            ic_hints_by_symbol[symbol] = self._master_hint_to_ic_hint(hint)

        return StockReviewBundle(
            agent_name="StockReviewOrchestrator",
            branch_overlay_verdicts_by_symbol=branch_overlay_verdicts_by_symbol,
            master_hints_by_symbol=master_hints_by_symbol,
            ic_hints_by_symbol=ic_hints_by_symbol,
            branch_summaries={},
            macro_verdict=macro_verdict,
            risk_decision=risk_decision,
            telemetry=telemetry,
            fallback_reasons=self._dedupe_texts(fallback_reasons),
            metadata={
                "branch_model": self.branch_model,
                "master_model": self.master_model,
                "branch_primary_model": self.branch_primary_model,
                "branch_fallback_model": self.branch_fallback_model,
                "branch_fallback_used": self.branch_fallback_used,
                "branch_fallback_reason": self.branch_fallback_reason,
                "master_primary_model": self.master_primary_model,
                "master_fallback_model": self.master_fallback_model,
                "master_fallback_used": self.master_fallback_used,
                "master_fallback_reason": self.master_fallback_reason,
                "master_reasoning_effort": self.master_reasoning_effort,
                "universe_key": self.universe_key,
                "market_regime": market_regime,
                "symbol_count": len(data_bundle.symbols),
                "branch_count": len(branch_names),
                "recall_context_keys": sorted((recall_context or {}).keys()),
                "macro_summary": macro_summary,
                "risk_summary": risk_summary,
            },
        )

    def _build_branch_overlay_packet(
        self,
        *,
        symbol: str,
        branch_name: str,
        branch_result: BranchResult,
        calibrated_signal: Any,
        macro_summary: dict[str, Any],
        risk_summary: dict[str, Any],
        recall_context: dict[str, Any] | None = None,
    ) -> BranchOverlayPacket:
        base_score = float(branch_result.symbol_scores.get(symbol, branch_result.final_score or branch_result.score))
        base_confidence = float(branch_result.final_confidence if branch_result.final_confidence is not None else branch_result.confidence)
        direction = self.portfolio_constructor.score_to_direction(base_score).value
        action = self.portfolio_constructor.score_to_action(base_score).value
        evidence = branch_result.evidence
        agreement_points = list(branch_result.support_drivers or branch_result.thesis_points or [])
        conflict_points = list(branch_result.drag_drivers or branch_result.diagnostic_notes or [])
        risk_points = list(branch_result.investment_risks or branch_result.risks or [])
        branch_signals = self._compact_mapping(branch_result.signals)
        if hasattr(calibrated_signal, "__dict__"):
            branch_signals.update(self._compact_mapping(calibrated_signal.__dict__))
        elif isinstance(calibrated_signal, Mapping):
            branch_signals.update(self._compact_mapping(calibrated_signal))
        if evidence and evidence.summary:
            branch_signals.setdefault("evidence_summary", str(evidence.summary)[:300])
        return BranchOverlayPacket(
            symbol=symbol,
            branch_name=branch_name,
            base_score=base_score,
            base_confidence=base_confidence,
            thesis=str(branch_result.conclusion or branch_result.explanation or branch_name).strip(),
            direction=direction,
            action=action,
            agreement_points=self._dedupe_texts(agreement_points[:4]),
            conflict_points=self._dedupe_texts(conflict_points[:4]),
            risk_points=self._dedupe_texts(risk_points[:4]),
            branch_signals=branch_signals,
            macro_summary=dict(macro_summary),
            risk_summary=dict(risk_summary),
            metadata={
                "branch_result_score": branch_result.final_score,
                "branch_result_confidence": branch_result.final_confidence,
                "symbol_score": base_score,
                "recall_context": dict(recall_context or {}),
                "source_branch": branch_name,
            },
        )

    def _build_master_symbol_packet(
        self,
        *,
        symbol: str,
        overlays: Mapping[str, BranchOverlayVerdict],
        macro_summary: dict[str, Any],
        risk_summary: dict[str, Any],
        risk_decision: RiskDecision | None,
        recall_context: dict[str, Any] | None = None,
    ) -> MasterSymbolPacket:
        overlay_dicts = [overlay.to_dict() for overlay in overlays.values()]
        scores = [float(item.get("adjusted_score", item.get("base_score", 0.0))) for item in overlay_dicts]
        confidences = [float(item.get("adjusted_confidence", item.get("base_confidence", 0.0))) for item in overlay_dicts]
        hard_veto = bool(risk_decision.hard_veto or risk_decision.veto) if risk_decision is not None else False
        return MasterSymbolPacket(
            symbol=symbol,
            branch_overlay_summaries=overlay_dicts,
            macro_summary=dict(macro_summary),
            risk_summary=dict(risk_summary),
            baseline_score=sum(scores) / len(scores) if scores else 0.0,
            baseline_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            hard_veto=hard_veto,
            metadata={
                "recall_context": dict(recall_context or {}),
                "overlay_count": len(overlay_dicts),
            },
        )

    @staticmethod
    def _build_macro_summary(
        data_bundle: UnifiedDataBundle,
        macro_verdict: BranchVerdict,
        ensemble_output: Mapping[str, Any],
        market_regime: str,
    ) -> dict[str, Any]:
        macro_data = dict(data_bundle.macro_data or {})
        summary = {
            "market_regime": market_regime,
            "policy_signal": macro_data.get("policy_signal", macro_data.get("signal", "neutral")),
            "regime": macro_data.get("regime", market_regime),
            "macro_score": macro_data.get("macro_score"),
            "liquidity_score": macro_data.get("liquidity_score"),
            "volatility_percentile": macro_data.get("volatility_percentile"),
            "target_gross_exposure": macro_verdict.metadata.get("target_gross_exposure", 0.0),
            "style_bias": macro_verdict.metadata.get("style_bias", "balanced"),
            "ensemble_aggregate_score": ensemble_output.get("aggregate_score", 0.0),
        }
        return {key: value for key, value in summary.items() if value is not None}

    def _build_macro_verdict_from_branch_results(
        self,
        branch_results: Mapping[str, BranchResult],
        data_bundle: UnifiedDataBundle,
    ) -> BranchVerdict:
        macro_branch = branch_results.get("macro")
        if isinstance(macro_branch, BranchResult):
            score = float(macro_branch.final_score if macro_branch.final_score is not None else macro_branch.score)
            confidence = float(
                macro_branch.final_confidence if macro_branch.final_confidence is not None else macro_branch.confidence
            )
            verdict = BranchVerdict(
                agent_name="MacroAgent",
                thesis=str(macro_branch.conclusion or macro_branch.explanation or "宏观分支已生成结构化判断。"),
                symbol=None,
                status=AgentStatus.SUCCESS,
                direction=self._score_to_direction(score),
                action=self._score_to_action(score),
                confidence_label=self._confidence_to_label(confidence),
                final_score=score,
                final_confidence=confidence,
                investment_risks=list(macro_branch.investment_risks[:5]),
                coverage_notes=list(macro_branch.coverage_notes[:5]),
                diagnostic_notes=list(macro_branch.diagnostic_notes[:5]),
                metadata={
                    "regime": str(
                        macro_branch.signals.get("macro_regime")
                        or macro_branch.signals.get("risk_level")
                        or data_bundle.macro_data.get("regime")
                        or "neutral"
                    ),
                    "target_gross_exposure": self._macro_target_exposure_from_score(score),
                    "style_bias": self._style_bias_from_macro_score(score),
                    "symbol": None,
                },
            )
            return verdict

        regime = str(data_bundle.macro_data.get("regime", "neutral"))
        return BranchVerdict(
            agent_name="MacroAgent",
            thesis="宏观分支已采用默认中性判断。",
            symbol=None,
            status=AgentStatus.DEGRADED,
            direction=Direction.NEUTRAL,
            action=ActionLabel.HOLD,
            confidence_label=ConfidenceLabel.MEDIUM,
            final_score=0.0,
            final_confidence=0.5,
            investment_risks=[],
            coverage_notes=[],
            diagnostic_notes=[],
            metadata={
                "regime": regime,
                "target_gross_exposure": 0.55,
                "style_bias": "balanced",
                "symbol": None,
            },
        )

    @staticmethod
    def _build_risk_summary(
        risk_result: Any,
        risk_agent_output: RiskAgentOutput | None,
        risk_decision: RiskDecision | None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if risk_decision is not None:
            summary.update(
                {
                    "hard_veto": risk_decision.hard_veto or risk_decision.veto,
                    "risk_level": risk_decision.risk_level.value if isinstance(risk_decision.risk_level, Enum) else str(risk_decision.risk_level),
                    "action_cap": risk_decision.action_cap.value if isinstance(risk_decision.action_cap, Enum) else str(risk_decision.action_cap),
                    "gross_exposure_cap": risk_decision.gross_exposure_cap,
                    "max_weight": risk_decision.max_weight,
                    "blocked_symbols": list(risk_decision.blocked_symbols[:10]),
                    "reasons": list(risk_decision.reasons[:5]),
                }
            )
        elif isinstance(risk_result, Mapping):
            summary.update({k: v for k, v in risk_result.items() if isinstance(v, (int, float, str, bool, list, dict))})
        elif hasattr(risk_result, "__dict__"):
            summary.update(
                {
                    k: v
                    for k, v in risk_result.__dict__.items()
                    if isinstance(v, (int, float, str, bool, list, dict))
                }
            )
        if risk_agent_output is not None:
            summary.setdefault("risk_assessment", risk_agent_output.risk_assessment)
            summary.setdefault("max_recommended_exposure", risk_agent_output.max_recommended_exposure)
            summary.setdefault("risk_warnings", list(risk_agent_output.risk_warnings[:5]))
            summary.setdefault("hedging_suggestions", list(risk_agent_output.hedging_suggestions[:3]))
        return summary

    def _coerce_risk_decision(self, risk_result: Any) -> RiskDecision | None:
        if isinstance(risk_result, RiskDecision):
            return risk_result
        if isinstance(risk_result, Mapping):
            try:
                return RiskDecision(**{k: v for k, v in risk_result.items() if k in {field.name for field in RiskDecision.__dataclass_fields__.values()}})
            except Exception:
                return None
        return None

    @staticmethod
    def _score_to_direction(score: float) -> Direction:
        if score >= 0.15:
            return Direction.BULLISH
        if score <= -0.15:
            return Direction.BEARISH
        return Direction.NEUTRAL

    @staticmethod
    def _score_to_action(score: float) -> ActionLabel:
        if score >= 0.25:
            return ActionLabel.BUY
        if score <= -0.35:
            return ActionLabel.SELL
        return ActionLabel.HOLD

    @staticmethod
    def _confidence_to_label(confidence: float) -> ConfidenceLabel:
        if confidence >= 0.75:
            return ConfidenceLabel.VERY_HIGH
        if confidence >= 0.60:
            return ConfidenceLabel.HIGH
        if confidence >= 0.40:
            return ConfidenceLabel.MEDIUM
        if confidence >= 0.20:
            return ConfidenceLabel.LOW
        return ConfidenceLabel.VERY_LOW

    @staticmethod
    def _macro_target_exposure_from_score(score: float) -> float:
        clamped = max(-1.0, min(1.0, float(score)))
        exposure = 0.55 + clamped * 0.25
        return round(max(0.10, min(0.90, exposure)), 4)

    @staticmethod
    def _style_bias_from_macro_score(score: float) -> str:
        if score >= 0.35:
            return "growth"
        if score <= -0.35:
            return "defensive"
        return "balanced"

    @staticmethod
    def _dedupe_texts(values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    def _deterministic_master_hint(
        self,
        packet: MasterSymbolPacket,
        *,
        reason: str,
    ) -> MasterICHint:
        avg_score = float(packet.baseline_score)
        avg_confidence = float(packet.baseline_confidence)
        if packet.hard_veto:
            avg_score = min(avg_score, 0.0)
        action = self.portfolio_constructor.score_to_action(avg_score)
        direction = self.portfolio_constructor.score_to_direction(avg_score)
        telemetry = ReviewTelemetry(
            stage="review_master_symbol",
            model=self.master_model,
            provider="fallback",
            latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            success=False,
            fallback=True,
            fallback_reason=reason,
            score_delta=0.0,
            confidence_delta=0.0,
            metadata={"actor_name": f"IC:{packet.symbol}", "deterministic_fallback": True},
        )
        return MasterICHint(
            symbol=packet.symbol,
            status=AgentStatus.VETOED if packet.hard_veto else AgentStatus.DEGRADED,
            thesis=f"{packet.symbol} deterministic master fallback.",
            action=action,
            direction=direction,
            score_hint=avg_score,
            confidence_hint=avg_confidence,
            score_delta=0.0,
            confidence_delta=0.0,
            agreement_points=["deterministic fallback"],
            conflict_points=[],
            rationale_points=[f"fallback_reason={reason}"],
            risk_flags=list(packet.risk_summary.get("risk_flags", []))[:5] if isinstance(packet.risk_summary, Mapping) else [],
            telemetry=telemetry,
            metadata={"fallback_reason": reason, "hard_veto": packet.hard_veto, "deterministic_fallback": True},
        )

    @staticmethod
    def _master_hint_to_ic_hint(hint: MasterICHint) -> dict[str, Any]:
        return {
            "score": float(hint.score_hint),
            "confidence": float(hint.confidence_hint),
            "action": hint.action.value if isinstance(hint.action, Enum) else str(hint.action),
            "direction": hint.direction.value if isinstance(hint.direction, Enum) else str(hint.direction),
            "rationale_points": list(hint.rationale_points[:4]),
            "agreement_points": list(hint.agreement_points[:3]),
            "conflict_points": list(hint.conflict_points[:3]),
            "risk_flags": list(hint.risk_flags[:5]),
            "score_delta": float(hint.score_delta),
            "confidence_delta": float(hint.confidence_delta),
            "status": hint.status.value if isinstance(hint.status, Enum) else str(hint.status),
            "telemetry": hint.telemetry.to_dict() if hasattr(hint.telemetry, "to_dict") else asdict(hint.telemetry),
            "thesis": hint.thesis,
        }

    @staticmethod
    def _compact_mapping(payload: Mapping[str, Any] | Any) -> dict[str, Any]:
        if isinstance(payload, Mapping):
            result: dict[str, Any] = {}
            for key, value in payload.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    result[str(key)] = value
                elif isinstance(value, Mapping):
                    nested = {k: v for k, v in value.items() if isinstance(v, (int, float, str, bool)) or v is None}
                    if nested:
                        result[str(key)] = nested
                elif isinstance(value, list):
                    items = [item for item in value if isinstance(item, (int, float, str, bool))]
                    if items:
                        result[str(key)] = items[:10]
            return result
        if hasattr(payload, "__dict__"):
            return AgentOrchestrator._compact_mapping(payload.__dict__)
        return {}

    @staticmethod
    def _serialize_review_item(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Mapping):
            return {
                str(key): AgentOrchestrator._serialize_review_item(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [AgentOrchestrator._serialize_review_item(item) for item in value]
        return value

    @staticmethod
    def _serialize_review_map(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): AgentOrchestrator._serialize_review_item(item)
            for key, item in value.items()
        }

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
            review_bundle=StockReviewBundle(),
            symbol_review_bundle={},
            ic_hints_by_symbol={},
        )
