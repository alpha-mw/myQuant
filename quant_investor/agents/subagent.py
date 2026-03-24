"""
V10.1 SubAgent 基类和向后兼容别名。

BaseSubAgent 提供通用 LLM 调用/解析/bounding 逻辑，
每个专属 SubAgent 继承并重写输入准备、输出验证和领域校验。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from quant_investor.agents.agent_contracts import (
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    RiskAgentInput,
    RiskAgentOutput,
)
from quant_investor.agents.llm_client import LLMCallError, LLMClient
from quant_investor.agents.prompts import CONVICTION_DEVIATION_CAP
from quant_investor.logger import get_logger

_logger = get_logger("SubAgent")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class BaseSubAgent(ABC):
    """所有分支 SubAgent 的抽象基类。

    提供通用的 LLM 调用流程：
    1. _get_system_prompt() → 获取专属 system prompt
    2. _get_output_schema_hint() → 获取专属输出 schema 提示
    3. LLM call → JSON response
    4. _parse_and_bound() → 通用 bounding
    5. _validate_specialized_output() → 专属输出解析
    """

    branch_name: str = ""

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

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """返回该分支的专属 system prompt（含输出 schema）。"""

    @abstractmethod
    def _validate_specialized_output(
        self,
        raw: dict[str, Any],
        agent_input: BaseBranchAgentInput,
    ) -> BaseBranchAgentOutput:
        """解析 LLM 输出为专属 Output 类型，含领域校验。"""

    async def analyze(self, agent_input: BaseBranchAgentInput) -> BaseBranchAgentOutput:
        """调用 LLM 审阅分支结果并返回研判。"""
        t0 = time.monotonic()
        input_json = agent_input.model_dump_json(indent=2)
        system_prompt = self._get_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是 {self.branch_name} 分支的量化分析结果，请审阅并给出你的研判：\n\n{input_json}"},
        ]

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
        agent_input: BaseBranchAgentInput,
    ) -> BaseBranchAgentOutput:
        """Parse LLM output, apply common bounding, then delegate to specialized validation."""
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

        # Validate conviction label
        valid_convictions = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        if raw.get("conviction") not in valid_convictions:
            raw["conviction"] = _score_to_conviction(bounded_score)

        return self._validate_specialized_output(raw, agent_input)

    @staticmethod
    def _score_to_conviction(score: float) -> str:
        return _score_to_conviction(score)


def _score_to_conviction(score: float) -> str:
    """将 score 映射到 conviction label。"""
    if score >= 0.5:
        return "strong_buy"
    if score >= 0.15:
        return "buy"
    if score <= -0.5:
        return "strong_sell"
    if score <= -0.15:
        return "sell"
    return "neutral"


# ---------------------------------------------------------------------------
# Backward-compatible BranchSubAgent (delegates to BaseSubAgent logic)
# ---------------------------------------------------------------------------

class BranchSubAgent(BaseSubAgent):
    """通用分支 SubAgent，向后兼容 V10.0。

    新代码应使用 subagents/ 下的专属 SubAgent 类。
    """

    def _get_system_prompt(self) -> str:
        from quant_investor.agents.prompts import BRANCH_SYSTEM_PROMPTS
        prompt = BRANCH_SYSTEM_PROMPTS.get(self.branch_name, "")
        if not prompt:
            raise ValueError(f"No system prompt defined for branch: {self.branch_name}")
        return prompt

    def _validate_specialized_output(
        self,
        raw: dict[str, Any],
        agent_input: BaseBranchAgentInput,
    ) -> BaseBranchAgentOutput:
        return BaseBranchAgentOutput.model_validate(raw)


# ---------------------------------------------------------------------------
# RiskSubAgent (kept here for backward compat, also available via subagents/)
# ---------------------------------------------------------------------------

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
        from quant_investor.agents.prompts import build_risk_agent_messages

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

        # Enhanced fields with defaults
        valid_tail = {"normal", "elevated", "critical"}
        if raw.get("tail_risk_assessment") not in valid_tail:
            raw.setdefault("tail_risk_assessment", "normal")

        raw["correlation_breakdown_risk"] = _clamp(
            float(raw.get("correlation_breakdown_risk", 0.0)), 0.0, 1.0,
        )

        return RiskAgentOutput.model_validate(raw)
