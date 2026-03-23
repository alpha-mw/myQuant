"""
V10 分支 SubAgent 和风控 SubAgent 实现。

每个 SubAgent 将量化分支结果提交给 LLM 进行定性审阅，
返回结构化的研判报告。
"""

from __future__ import annotations

import time
from typing import Any

from quant_investor.agents.agent_contracts import (
    BranchAgentInput,
    BranchAgentOutput,
    RiskAgentInput,
    RiskAgentOutput,
)
from quant_investor.agents.llm_client import LLMClient, LLMCallError
from quant_investor.agents.prompts import (
    CONVICTION_DEVIATION_CAP,
    build_branch_agent_messages,
    build_risk_agent_messages,
)
from quant_investor.logger import get_logger

_logger = get_logger("SubAgent")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class BranchSubAgent:
    """分支专属 SubAgent：解读量化结果，输出定性研判。"""

    def __init__(
        self,
        branch_name: str,
        llm_client: LLMClient,
        model: str,
        timeout: float = 15.0,
        max_tokens: int = 800,
    ) -> None:
        self.branch_name = branch_name
        self.llm_client = llm_client
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    async def analyze(self, agent_input: BranchAgentInput) -> BranchAgentOutput:
        """调用 LLM 审阅分支结果并返回研判。"""
        t0 = time.monotonic()
        input_json = agent_input.model_dump_json(indent=2)
        messages = build_branch_agent_messages(self.branch_name, input_json)

        try:
            raw = await self.llm_client.complete(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                response_json=True,
            )
            output = self._parse_and_bound(raw, agent_input)
            elapsed = time.monotonic() - t0
            _logger.info(f"[{self.branch_name}] SubAgent completed in {elapsed:.1f}s")
            return output
        except (LLMCallError, Exception) as exc:
            elapsed = time.monotonic() - t0
            _logger.warning(f"[{self.branch_name}] SubAgent failed after {elapsed:.1f}s: {exc}")
            raise

    def _parse_and_bound(
        self,
        raw: dict[str, Any],
        agent_input: BranchAgentInput,
    ) -> BranchAgentOutput:
        """Parse LLM output, validate, and apply bounding."""
        raw.setdefault("branch_name", self.branch_name)

        # Bound conviction_score
        cap = CONVICTION_DEVIATION_CAP.get(self.branch_name, 0.25)
        raw_score = float(raw.get("conviction_score", agent_input.final_score))
        bounded_score = _clamp(
            raw_score,
            agent_input.final_score - cap,
            agent_input.final_score + cap,
        )
        bounded_score = _clamp(bounded_score, -1.0, 1.0)
        raw["conviction_score"] = bounded_score

        # Bound confidence
        raw["confidence"] = _clamp(float(raw.get("confidence", 0.5)), 0.0, 1.0)

        # Validate conviction label consistency
        valid_convictions = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        if raw.get("conviction") not in valid_convictions:
            raw["conviction"] = self._score_to_conviction(bounded_score)

        return BranchAgentOutput.model_validate(raw)

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


class RiskSubAgent:
    """风控专属 SubAgent：评估组合风险并提出风控建议。"""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        timeout: float = 15.0,
        max_tokens: int = 800,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    async def analyze(self, agent_input: RiskAgentInput) -> RiskAgentOutput:
        """调用 LLM 评估组合风险。"""
        t0 = time.monotonic()
        input_json = agent_input.model_dump_json(indent=2)
        messages = build_risk_agent_messages(input_json)

        try:
            raw = await self.llm_client.complete(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                response_json=True,
            )
            output = self._parse_and_validate(raw)
            elapsed = time.monotonic() - t0
            _logger.info(f"[risk] SubAgent completed in {elapsed:.1f}s")
            return output
        except (LLMCallError, Exception) as exc:
            elapsed = time.monotonic() - t0
            _logger.warning(f"[risk] SubAgent failed after {elapsed:.1f}s: {exc}")
            raise

    @staticmethod
    def _parse_and_validate(raw: dict[str, Any]) -> RiskAgentOutput:
        """Parse and validate risk agent output."""
        valid_assessments = {"acceptable", "elevated", "high", "extreme"}
        if raw.get("risk_assessment") not in valid_assessments:
            raw["risk_assessment"] = "elevated"

        raw["max_recommended_exposure"] = _clamp(
            float(raw.get("max_recommended_exposure", 0.6)), 0.0, 1.0,
        )

        return RiskAgentOutput.model_validate(raw)
