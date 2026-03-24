"""量化因子分支专属 SubAgent。"""

from __future__ import annotations

from typing import Any

from quant_investor.agents.agent_contracts import (
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    QuantAgentInput,
    QuantAgentOutput,
)
from quant_investor.agents.prompts import CONVICTION_DEVIATION_CAP
from quant_investor.agents.subagent import BaseSubAgent, _clamp


class QuantSubAgent(BaseSubAgent):
    """量化因子研究专家：解读因子暴露、Alpha 信号和统计指标。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("branch_name", "quant")
        super().__init__(**kwargs)

    def _get_system_prompt(self) -> str:
        cap = CONVICTION_DEVIATION_CAP.get("quant", 0.25)
        return f"""\
你是一位量化因子研究首席分析师（量化分支专属 SubAgent），专注于因子投资、Alpha 挖掘和风险因子解读。

你的专业领域：
- 多因子模型：动量、价值、质量、低波动、规模等经典因子
- Alpha mining：遗传算法挖掘的非线性因子表达式的经济学解读
- 因子衰减与拥挤度：判断因子信号的时效性和市场拥挤程度
- 因子 regime 适配性：不同市场环境（趋势/震荡/高波/低波）下因子有效性的差异
- IC/IR 分析：信息系数和信息比率的统计意义判断

专属分析能力：
1. **因子质量评估**：对每个使用的因子给出 robust/decaying/crowded 判断
   - IC 绝对值 > 0.05 且 IR > 0.5 → robust
   - IC 在近期下降 → decaying
   - 因子暴露集中度过高 → crowded
2. **过拟合风险检测**：遗传算法 Alpha 的样本外有效性判断
   - Alpha 候选表达式过于复杂（>5 个因子组合）→ 过拟合风险升高
   - IC 极高（>0.15）但样本量小 → 可疑
3. **因子矛盾识别**：当动量与价值因子方向相反时，应明确标注为 factor_conflict
4. **Regime 适配判断**：评估当前市场状态下因子组合的预期有效性

你必须以 **纯 JSON** 格式回复，不要加 markdown 代码块。
JSON schema:
{{
  "branch_name": "quant",
  "conviction": "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell",
  "conviction_score": <float, -1.0 ~ 1.0>,
  "confidence": <float, 0.0 ~ 1.0>,
  "key_insights": ["<insight_1>", ...],
  "risk_flags": ["<risk_1>", ...],
  "disagreements_with_algo": ["<disagreement_1>", ...],
  "symbol_views": {{"<symbol>": "<one-line view>", ...}},
  "reasoning": "<2-3 句总结>",
  "factor_quality_assessment": {{"<factor>": "robust" | "decaying" | "crowded", ...}},
  "regime_suitability": <float, 0.0 ~ 1.0>,
  "overfitting_risk": <float, 0.0 ~ 1.0>,
  "factor_conflicts": ["<conflict_1>", ...],
  "recommended_factor_tilts": {{"<factor>": "overweight" | "neutral" | "underweight", ...}}
}}

约束:
- conviction_score 不得偏离输入的 final_score 超过 ±{cap}
- 关注因子的经济学逻辑，而非仅看统计显著性
- 遗传算法 Alpha 表达式过于复杂时，overfitting_risk 应 > 0.6
"""

    def _validate_specialized_output(
        self,
        raw: dict[str, Any],
        agent_input: BaseBranchAgentInput,
    ) -> BaseBranchAgentOutput:
        raw["regime_suitability"] = _clamp(float(raw.get("regime_suitability", 0.5)), 0.0, 1.0)
        raw["overfitting_risk"] = _clamp(float(raw.get("overfitting_risk", 0.0)), 0.0, 1.0)

        if not isinstance(raw.get("factor_quality_assessment"), dict):
            raw["factor_quality_assessment"] = {}
        if not isinstance(raw.get("factor_conflicts"), list):
            raw["factor_conflicts"] = []
        if not isinstance(raw.get("recommended_factor_tilts"), dict):
            raw["recommended_factor_tilts"] = {}

        return QuantAgentOutput.model_validate(raw)
