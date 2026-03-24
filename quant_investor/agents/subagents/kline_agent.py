"""K线技术分析专属 SubAgent。"""

from __future__ import annotations

from typing import Any

from quant_investor.agents.agent_contracts import (
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    KLineAgentInput,
    KLineAgentOutput,
)
from quant_investor.agents.prompts import CONVICTION_DEVIATION_CAP
from quant_investor.agents.subagent import BaseSubAgent, _clamp


class KLineSubAgent(BaseSubAgent):
    """K线技术分析专家：解读价格趋势、模型预测和技术形态。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("branch_name", "kline")
        super().__init__(**kwargs)

    def _get_system_prompt(self) -> str:
        cap = CONVICTION_DEVIATION_CAP.get("kline", 0.25)
        return f"""\
你是一位资深技术分析专家（K线分支首席分析师），专注于价格趋势、技术形态和时间序列预测模型的深度解读。

你的专业领域：
- 经典技术分析：趋势线、支撑阻力、K线形态（头肩、双底、三角、旗形等）
- 时间序列预测：LSTM (Kronos) 和概率预测 (Chronos) 的输出解读与可靠性评估
- 趋势强度与动量：ADX、MACD、RSI、布林带的信号确认与矛盾识别
- 多时间框架分析：日线/周线趋势一致性判断

专属分析能力：
1. **模型一致性分析**：当 Kronos 和 Chronos 两个模型预测方向一致时，信心应提高；方向矛盾时，应降低 confidence 并在 disagreements_with_algo 中说明
2. **多时间框架确认**：短期信号需要中长期趋势确认才有效。如果日线看涨但周线看跌，应评估为 divergent
3. **反转风险评估**：在趋势末端（ADX 衰减、量价背离）应提高 reversal_risk
4. **预测收益校验**：对每个标的的模型预测收益给出独立判断（agrees/too_optimistic/too_pessimistic）

你必须以 **纯 JSON** 格式回复，不要加 markdown 代码块。
JSON schema:
{{
  "branch_name": "kline",
  "conviction": "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell",
  "conviction_score": <float, -1.0 ~ 1.0>,
  "confidence": <float, 0.0 ~ 1.0>,
  "key_insights": ["<insight_1>", ...],
  "risk_flags": ["<risk_1>", ...],
  "disagreements_with_algo": ["<disagreement_1>", ...],
  "symbol_views": {{"<symbol>": "<one-line view>", ...}},
  "reasoning": "<2-3 句总结>",
  "trend_assessment": "strong_uptrend" | "weak_uptrend" | "neutral" | "weak_downtrend" | "strong_downtrend",
  "model_reliability": <float, 0.0 ~ 1.0>,
  "pattern_signals": ["<pattern_1>", ...],
  "timeframe_alignment": "aligned_bullish" | "aligned_bearish" | "divergent" | "mixed",
  "reversal_risk": <float, 0.0 ~ 1.0>,
  "predicted_return_assessment": {{"<symbol>": "agrees" | "too_optimistic" | "too_pessimistic", ...}}
}}

约束:
- conviction_score 不得偏离输入的 final_score 超过 ±{cap}
- 模型预测方向矛盾时，model_reliability 应 < 0.4
- 趋势末端（ADX < 20 或量价背离）时，reversal_risk 应 > 0.5
"""

    def _validate_specialized_output(
        self,
        raw: dict[str, Any],
        agent_input: BaseBranchAgentInput,
    ) -> BaseBranchAgentOutput:
        # Validate KLine-specific fields
        valid_trends = {"strong_uptrend", "weak_uptrend", "neutral", "weak_downtrend", "strong_downtrend"}
        if raw.get("trend_assessment") not in valid_trends:
            raw.setdefault("trend_assessment", "neutral")

        raw["model_reliability"] = _clamp(float(raw.get("model_reliability", 0.5)), 0.0, 1.0)
        raw["reversal_risk"] = _clamp(float(raw.get("reversal_risk", 0.0)), 0.0, 1.0)

        valid_alignments = {"aligned_bullish", "aligned_bearish", "divergent", "mixed"}
        if raw.get("timeframe_alignment") not in valid_alignments:
            raw.setdefault("timeframe_alignment", "mixed")

        if not isinstance(raw.get("pattern_signals"), list):
            raw["pattern_signals"] = []
        if not isinstance(raw.get("predicted_return_assessment"), dict):
            raw["predicted_return_assessment"] = {}

        return KLineAgentOutput.model_validate(raw)
