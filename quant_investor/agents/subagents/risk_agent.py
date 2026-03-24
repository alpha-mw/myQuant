"""风控专属 SubAgent（增强版）。"""

from __future__ import annotations

import time
from typing import Any

from quant_investor.agents.agent_contracts import RiskAgentInput, RiskAgentOutput
from quant_investor.agents.llm_client import LLMCallError, LLMClient
from quant_investor.agents.subagent import _clamp
from quant_investor.logger import get_logger

_logger = get_logger("RiskSubAgent")


class SpecializedRiskSubAgent:
    """增强版风控 SubAgent：深度尾部风险分析、相关性崩溃检测和仓位覆盖。"""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        timeout: float = 15.0,
        max_tokens: int = 1000,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _get_system_prompt(self) -> str:
        return """\
你是首席风控官（Risk SubAgent 增强版），负责全方位评估投资组合的风险水平并提出风控建议。

你的专业领域：
- 风险度量：VaR、CVaR（条件风险价值）、最大回撤、波动率分析
- 尾部风险分析：极端事件概率评估、黑天鹅识别、fat-tail 检测
- 相关性风险：相关性崩溃预警（危机时资产相关性趋向 1.0）
- 仓位管理：基于风险预算的仓位分配、集中度风险
- 压力测试：极端情景下的组合表现评估
- 对冲策略：降低组合风险的具体对冲手段

专属分析能力：
1. **尾部风险评估**：
   - CVaR/VaR 比值 > 1.5 → tail_risk_assessment = "elevated"
   - 波动率急剧扩大 + 偏度变负 → "critical"
2. **相关性崩溃预警**：
   - 近期资产间相关性急剧上升 → correlation_breakdown_risk > 0.5
   - 多数标的同向运动加剧 → 分散化失效风险
3. **分支分歧风险**：
   - 5 个分支 SubAgent 的 conviction_score 标准差 > 0.4 → 不确定性极高
   - 分歧度高时应主动降低 max_recommended_exposure
4. **回撤情景描述**：具体描述最可能的回撤路径和预期幅度

决策原则：先求不败，再求胜。宁可错过机会，也不要承担不可控的风险。

你必须以 **纯 JSON** 格式回复，不要加 markdown 代码块。
JSON schema:
{
  "risk_assessment": "acceptable" | "elevated" | "high" | "extreme",
  "max_recommended_exposure": <float, 0.0 ~ 1.0>,
  "position_adjustments": {"<symbol>": <multiplier float>, ...},
  "risk_warnings": ["<warning_1>", ...],
  "hedging_suggestions": ["<suggestion_1>", ...],
  "tail_risk_assessment": "normal" | "elevated" | "critical",
  "correlation_breakdown_risk": <float, 0.0 ~ 1.0>,
  "position_sizing_overrides": {"<symbol>": {"max_weight": <float>, "reason_score": <float>}, ...},
  "drawdown_scenario": "<预期回撤情景描述>",
  "reasoning": "<2-3 句总结>"
}

约束:
- risk_assessment 为 "extreme" 时，max_recommended_exposure 应 < 0.3
- 分支分歧度 > 0.6 时，应在 risk_warnings 中明确标注
- tail_risk_assessment 为 "critical" 时，应建议减仓或对冲
"""

    async def analyze(self, agent_input: RiskAgentInput) -> RiskAgentOutput:
        """调用 LLM 评估组合风险（增强版）。"""
        t0 = time.monotonic()
        input_json = agent_input.model_dump_json(indent=2)

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"以下是风险管理层的量化输出和各分支 SubAgent 的研判汇总，请评估整体风险：\n\n{input_json}"},
        ]

        try:
            raw = await self.llm_client.complete(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                response_json=True,
            )
            output = self._parse_and_validate(raw)
            elapsed = time.monotonic() - t0
            _logger.info(f"[risk] Specialized SubAgent completed in {elapsed:.1f}s")
            return output
        except (LLMCallError, Exception) as exc:
            elapsed = time.monotonic() - t0
            _logger.warning(f"[risk] Specialized SubAgent failed after {elapsed:.1f}s: {exc}")
            raise

    @staticmethod
    def _parse_and_validate(raw: dict[str, Any]) -> RiskAgentOutput:
        """Parse and validate enhanced risk agent output."""
        valid_assessments = {"acceptable", "elevated", "high", "extreme"}
        if raw.get("risk_assessment") not in valid_assessments:
            raw["risk_assessment"] = "elevated"

        raw["max_recommended_exposure"] = _clamp(
            float(raw.get("max_recommended_exposure", 0.6)), 0.0, 1.0,
        )

        valid_tail = {"normal", "elevated", "critical"}
        if raw.get("tail_risk_assessment") not in valid_tail:
            raw.setdefault("tail_risk_assessment", "normal")

        raw["correlation_breakdown_risk"] = _clamp(
            float(raw.get("correlation_breakdown_risk", 0.0)), 0.0, 1.0,
        )

        if not isinstance(raw.get("position_sizing_overrides"), dict):
            raw["position_sizing_overrides"] = {}

        return RiskAgentOutput.model_validate(raw)
