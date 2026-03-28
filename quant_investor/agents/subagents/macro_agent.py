"""宏观分支专属 SubAgent。"""

from __future__ import annotations

from typing import Any

from quant_investor.agents.agent_contracts import (
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    MacroAgentInput,
    MacroAgentOutput,
)
from quant_investor.agents.prompts import CONVICTION_DEVIATION_CAP
from quant_investor.agents.subagent import BaseSubAgent, _clamp


class MacroSubAgent(BaseSubAgent):
    """宏观策略首席分析师：解读流动性环境、波动率结构和跨资产联动。

    Design: 本 agent 每次 pipeline 执行仅运行一次（非逐股），产出 portfolio 级别的
    统一宏观评估，广播至所有标的。uniform_score_appropriateness 字段标记此处理是否合理。
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("branch_name", "macro")
        super().__init__(**kwargs)

    def _get_system_prompt(self) -> str:
        cap = CONVICTION_DEVIATION_CAP.get("macro", 0.25)
        return f"""\
你是一位宏观策略首席分析师（宏观分支专属 SubAgent），专注于市场整体环境、流动性周期和跨资产联动分析。

你的专业领域：
- 流动性环境分析：央行政策方向、利率走势、M2 增速、信贷脉冲
- 波动率结构：VIX/波动率百分位、期限结构（contango vs backwardation）
- 市场广度分析：上涨/下跌比率、新高新低比率、参与度
- 多周期动量结构：短/中/长期动量的一致性与背离
- 跨资产联动：股-债-汇-商的相关性变化与传导路径

专属分析能力：
1. **流动性周期判断**：
   - 央行宽松 + M2 加速 → liquidity_outlook = "supportive"
   - 央行紧缩 + 信贷收紧 → liquidity_outlook = "restrictive"
   - 政策转向初期 → 信号最强，需特别关注
2. **系统性风险评估**：
   - 波动率百分位 > 90% + 信用利差扩大 → systemic_risk_level > 0.7
   - 跨资产相关性急剧变化 → regime 转换预警
3. **Regime 转换概率**：
   - 短期动量与长期动量方向相反 → regime_transition_risk > 0.5
   - 波动率期限结构从 contango 翻转为 backwardation → 风险信号
4. **统一打分合理性**：
   - 宏观评分对所有标的一致，评估这是否合理
   - 如果个股受行业政策影响差异大 → uniform_score_appropriateness = "questionable"

你必须以 **纯 JSON** 格式回复，不要加 markdown 代码块。
JSON schema:
{{
  "branch_name": "macro",
  "conviction": "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell",
  "conviction_score": <float, -1.0 ~ 1.0>,
  "confidence": <float, 0.0 ~ 1.0>,
  "key_insights": ["<insight_1>", ...],
  "risk_flags": ["<risk_1>", ...],
  "disagreements_with_algo": ["<disagreement_1>", ...],
  "symbol_views": {{"<symbol>": "<one-line view>", ...}},
  "reasoning": "<2-3 句总结>",
  "macro_regime_assessment": "<当前宏观 regime 判断>",
  "liquidity_outlook": "supportive" | "neutral" | "restrictive",
  "systemic_risk_level": <float, 0.0 ~ 1.0>,
  "cross_asset_implications": ["<implication_1>", ...],
  "uniform_score_appropriateness": "appropriate" | "questionable" | "inappropriate",
  "regime_transition_risk": <float, 0.0 ~ 1.0>
}}

约束:
- conviction_score 不得偏离输入的 final_score 超过 ±{cap}
- 宏观分析的时间框架最长（20 天），避免对短期波动过度反应
- 波动率百分位 > 90% 时，systemic_risk_level 应 > 0.5
"""

    def _validate_specialized_output(
        self,
        raw: dict[str, Any],
        agent_input: BaseBranchAgentInput,
    ) -> BaseBranchAgentOutput:
        valid_liquidity = {"supportive", "neutral", "restrictive"}
        if raw.get("liquidity_outlook") not in valid_liquidity:
            raw.setdefault("liquidity_outlook", "neutral")

        valid_uniform = {"appropriate", "questionable", "inappropriate"}
        if raw.get("uniform_score_appropriateness") not in valid_uniform:
            raw.setdefault("uniform_score_appropriateness", "appropriate")

        raw["systemic_risk_level"] = _clamp(float(raw.get("systemic_risk_level", 0.0)), 0.0, 1.0)
        raw["regime_transition_risk"] = _clamp(float(raw.get("regime_transition_risk", 0.0)), 0.0, 1.0)

        if not isinstance(raw.get("cross_asset_implications"), list):
            raw["cross_asset_implications"] = []

        if not isinstance(raw.get("macro_regime_assessment"), str):
            raw["macro_regime_assessment"] = "neutral"

        return MacroAgentOutput.model_validate(raw)
