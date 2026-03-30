"""智能融合分支专属 SubAgent。"""

from __future__ import annotations

from typing import Any

from quant_investor.agents.agent_contracts import (
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    IntelligenceAgentInput,
    IntelligenceAgentOutput,
)
from quant_investor.agents.prompts import CONVICTION_DEVIATION_CAP
from quant_investor.agents.subagent import BaseSubAgent, _clamp


class IntelligenceSubAgent(BaseSubAgent):
    """多维信息情报首席分析师：解读事件驱动、市场情绪和资金流向。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("branch_name", "intelligence")
        super().__init__(**kwargs)

    def _get_system_prompt(self) -> str:
        cap = CONVICTION_DEVIATION_CAP.get("intelligence", 0.30)
        return f"""\
你是一位多维信息情报首席分析师（智能分支专属 SubAgent），专注于事件驱动、市场情绪和资金流向的深度解读。

你的专业领域：
- 事件风险评估：重大新闻、公告、政策变化的影响判断与时间窗口评估
- 市场情绪分析：恐惧-贪婪指数、情绪极端值的识别与反转信号判断
- 资金流向分析：日内量能比率、主力资金动向、「聪明钱」行为识别
- 市场广度与轮动：涨跌比、板块轮动节奏、领涨领跌结构

专属分析能力：
1. **情绪反转判断**：
   - 恐惧贪婪指数 < 20 → sentiment_regime = "extreme_fear"，通常是 contrarian buy 信号
   - 恐惧贪婪指数 > 80 → sentiment_regime = "extreme_greed"，通常是 contrarian sell 信号
   - 在极端情绪区间，contrarian_signal 应与情绪方向相反
2. **催化剂时间窗口**：
   - 识别即将发生的重大事件（财报、政策会议、经济数据发布）
   - 事件前 3 天内，event_timing_risk 应 > 0.5
3. **聪明钱识别**：
   - 放量不涨（大量成交但价格横盘）→ smart_money_accumulating/distributing
   - 尾盘异常成交 → 可能的信息不对称
4. **板块轮动评估**：判断当前轮动方向是否有利于持仓标的

你必须以 **纯 JSON** 格式回复，不要加 markdown 代码块。
JSON schema:
{{
  "branch_name": "intelligence",
  "conviction": "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell",
  "conviction_score": <float, -1.0 ~ 1.0>,
  "confidence": <float, 0.0 ~ 1.0>,
  "key_insights": ["<insight_1>", ...],
  "risk_flags": ["<risk_1>", ...],
  "disagreements_with_algo": ["<disagreement_1>", ...],
  "symbol_views": {{"<symbol>": "<one-line view>", ...}},
  "reasoning": "<2-3 句总结>",
  "sentiment_regime": "extreme_fear" | "fear" | "neutral" | "greed" | "extreme_greed",
  "contrarian_signal": "strong_buy_contrarian" | "buy_contrarian" | "none" | "sell_contrarian" | "strong_sell_contrarian",
  "catalyst_assessment": ["<catalyst_1>", ...],
  "information_asymmetry": "smart_money_accumulating" | "distributing" | "none",
  "event_timing_risk": <float, 0.0 ~ 1.0>
}}

约束:
- conviction_score 不得偏离输入的 final_score 超过 ±{cap}
- 情绪极端值是反转信号：极度恐惧常是买入机会，极度贪婪常是卖出信号
- 在有重大事件催化剂的标的上，event_timing_risk 必须如实反映
"""

    def _validate_specialized_output(
        self,
        raw: dict[str, Any],
        agent_input: BaseBranchAgentInput,
    ) -> BaseBranchAgentOutput:
        valid_sentiment = {"extreme_fear", "fear", "neutral", "greed", "extreme_greed"}
        if raw.get("sentiment_regime") not in valid_sentiment:
            raw.setdefault("sentiment_regime", "neutral")

        valid_contrarian = {
            "strong_buy_contrarian", "buy_contrarian", "none",
            "sell_contrarian", "strong_sell_contrarian",
        }
        if raw.get("contrarian_signal") not in valid_contrarian:
            raw.setdefault("contrarian_signal", "none")

        valid_asymmetry = {"smart_money_accumulating", "distributing", "none"}
        if raw.get("information_asymmetry") not in valid_asymmetry:
            raw.setdefault("information_asymmetry", "none")

        raw["event_timing_risk"] = _clamp(float(raw.get("event_timing_risk", 0.0)), 0.0, 1.0)

        if not isinstance(raw.get("catalyst_assessment"), list):
            raw["catalyst_assessment"] = []

        return IntelligenceAgentOutput.model_validate(raw)
