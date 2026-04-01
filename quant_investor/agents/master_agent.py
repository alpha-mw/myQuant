"""
V12 Master Agent（IC 投资委员会主席）。

直接接收5个分支的原始量化数据 + 风控结果 + 历史交易回顾，
主持五轮多空辩论，产出最终投资决策（含交易决策和投资逻辑存档）。
"""

from __future__ import annotations

import time
from typing import Any

from quant_investor.agents.agent_contracts import (
    MasterAgentInput,
    MasterAgentOutput,
    SymbolRecommendation,
    TradeDecision,
)
from quant_investor.agents.llm_client import LLMClient, LLMCallError
from quant_investor.agents.prompts import build_master_agent_messages
from quant_investor.logger import get_logger

_logger = get_logger("MasterAgent")

# IC 对算法 baseline 的最大偏离
_IC_DEVIATION_CAP = 0.30


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class MasterAgent:
    """IC 主席：直接读取5个分支原始量化数据，主持五轮多空辩论，做出最终决策。"""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        timeout: float = 30.0,
        max_tokens: int = 1500,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    async def deliberate(self, agent_input: MasterAgentInput) -> MasterAgentOutput:
        """主持 IC 会议并产出最终决策。"""
        t0 = time.monotonic()
        input_json = agent_input.model_dump_json(indent=2)
        messages = build_master_agent_messages(input_json)

        try:
            raw = await self.llm_client.complete(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                response_json=True,
            )
            output = self._parse_and_bound(raw, agent_input)
            elapsed = time.monotonic() - t0
            _logger.info(f"[IC] Master Agent deliberation completed in {elapsed:.1f}s")
            return output
        except (LLMCallError, Exception) as exc:
            elapsed = time.monotonic() - t0
            _logger.warning(f"[IC] Master Agent failed after {elapsed:.1f}s: {exc}")
            raise

    def _parse_and_bound(
        self,
        raw: dict[str, Any],
        agent_input: MasterAgentInput,
    ) -> MasterAgentOutput:
        """Parse, validate, and apply IC deviation bounds."""
        # Bound final_score relative to algorithmic baseline
        baseline_score = float(agent_input.ensemble_baseline.get("aggregate_score", 0.0))
        raw_score = float(raw.get("final_score", baseline_score))
        bounded_score = _clamp(
            raw_score,
            baseline_score - _IC_DEVIATION_CAP,
            baseline_score + _IC_DEVIATION_CAP,
        )
        bounded_score = _clamp(bounded_score, -1.0, 1.0)
        raw["final_score"] = bounded_score

        # Bound confidence
        raw["confidence"] = _clamp(float(raw.get("confidence", 0.5)), 0.0, 1.0)

        # Bound risk_adjusted_exposure
        raw["risk_adjusted_exposure"] = _clamp(
            float(raw.get("risk_adjusted_exposure", 0.5)), 0.0, 1.0,
        )

        # Validate conviction label
        valid_convictions = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        if raw.get("final_conviction") not in valid_convictions:
            raw["final_conviction"] = self._score_to_conviction(bounded_score)

        # Apply risk veto from risk_result (if VaR / risk level is extreme)
        risk_result = agent_input.risk_result or {}
        risk_level = str(risk_result.get("risk_level", risk_result.get("risk_assessment", ""))).lower()
        if risk_level == "extreme":
            if raw["final_conviction"] in ("strong_buy", "buy"):
                raw["final_conviction"] = "neutral"
                raw["final_score"] = min(bounded_score, 0.1)
                raw.setdefault("debate_resolution", []).append(
                    "风控层一票否决：风险评估为 extreme，conviction 已降级至 neutral"
                )

        # Parse top_picks into SymbolRecommendation
        raw_picks = raw.get("top_picks", [])
        parsed_picks = []
        for pick in raw_picks:
            if isinstance(pick, dict):
                try:
                    parsed_picks.append(SymbolRecommendation.model_validate(pick))
                except Exception:
                    pass
        raw["top_picks"] = parsed_picks

        # Parse trade_decisions into TradeDecision
        raw_decisions = raw.get("trade_decisions", [])
        parsed_decisions = []
        for decision in raw_decisions:
            if isinstance(decision, dict):
                try:
                    parsed_decisions.append(TradeDecision.model_validate(decision))
                except Exception:
                    pass
        raw["trade_decisions"] = parsed_decisions

        # Ensure list fields
        for field in (
            "debate_rounds", "consensus_areas", "disagreement_areas",
            "debate_resolution", "conviction_drivers", "dissenting_views",
        ):
            if not isinstance(raw.get(field), list):
                raw[field] = []

        # Ensure string fields
        for field in ("bull_case", "bear_case", "investment_thesis", "portfolio_narrative"):
            if not isinstance(raw.get(field), str):
                raw[field] = ""

        return MasterAgentOutput.model_validate(raw)

    @staticmethod
    def _score_to_conviction(score: float) -> str:
        if score >= 0.5:
            return "strong_buy"
        if score >= 0.15:
            return "buy"
        if score <= -0.5:
            return "strong_sell"
        if score <= -0.15:
            return "sell"
        return "neutral"
