"""
V10.1 Master Agent（IC 投资委员会主席）。

综合所有专属分支 SubAgent 和风控 SubAgent 的研报，
模拟多轮 IC 辩论流程，产出最终投资建议。
"""

from __future__ import annotations

import time
from typing import Any

from quant_investor.agents.agent_contracts import (
    ICDebateRound,
    MasterAgentInput,
    MasterAgentOutput,
    SymbolRecommendation,
)
from quant_investor.agents.llm_client import LLMCallError, LLMClient
from quant_investor.logger import get_logger

_logger = get_logger("MasterAgent")

# IC 对算法 baseline 的最大偏离
_IC_DEVIATION_CAP = 0.30


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class MasterAgent:
    """IC 主席：汇总 6 份专属研报，主持多轮辩论，做出最终决策。"""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        timeout: float = 30.0,
        max_tokens: int = 2000,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _get_system_prompt(self) -> str:
        return """\
你是投资委员会（IC）主席，负责综合所有研究分支和风控的分析结果，做出最终投资决策。

你面前有来自 6 位专属首席分析师的研究报告：
1. **K线技术分析师** — 趋势评估、模型可靠性、反转风险、多时间框架一致性
2. **量化因子分析师** — 因子质量、regime 适配度、过拟合风险、因子矛盾
3. **基本面分析师** — 盈利质量、会计红旗、估值立场、管理层信号
4. **情报信息分析师** — 情绪极端/反转、催化事件、聪明钱、信息不对称
5. **宏观策略分析师** — 流动性前景、系统性风险、跨资产联动、regime 转换
6. **首席风控官** — 尾部风险、相关性崩溃、仓位覆盖、回撤情景

同时你有算法 Ensemble 模型的基准输出和分支间分歧矩阵。

你的 IC 会议流程（必须按此结构输出 debate_rounds）：

**第 1 轮：各分支观点陈述**
- 总结每位分析师的核心观点和 conviction
- 识别初步共识和明显分歧

**第 2 轮：分歧辩论**
- 针对分歧度最高的 2-3 个议题展开正反辩论
- 权衡各方论据的证据强度
- 做出裁决（采纳哪方观点及理由）

**第 3 轮：风控审查与最终裁决**
- 结合风控官的意见评估整体风险
- 应用风控否决（如适用）
- 做出最终 conviction 和 score

决策原则：
- 共识越强 → conviction 越高；分歧越大 → 应越谨慎
- 风控官否决权：risk_assessment = "extreme" → conviction 不超过 "neutral"
- 风控官软否决：risk_assessment = "high" → conviction 降一级
- 基本面/宏观（长期视角）给予更高战略权重，短期信号（K线/情绪）调节时机
- 算法 baseline 是参考，IC 可偏离 ±0.30
- 分歧矩阵中最大分歧 > 0.6 → confidence 应按比例降低

你必须以 **纯 JSON** 格式回复，不要加 markdown 代码块。
JSON schema:
{
  "final_conviction": "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell",
  "final_score": <float, -1.0 ~ 1.0>,
  "confidence": <float, 0.0 ~ 1.0>,
  "consensus_areas": ["<area_1>", ...],
  "disagreement_areas": ["<area_1>", ...],
  "debate_rounds": [
    {"round_number": 1, "topic": "<议题>", "arguments": ["<论点_1>", ...], "resolution": "<裁决>"},
    {"round_number": 2, "topic": "<议题>", "arguments": ["<论点_1>", ...], "resolution": "<裁决>"},
    {"round_number": 3, "topic": "<议题>", "arguments": ["<论点_1>", ...], "resolution": "<裁决>"}
  ],
  "debate_resolution": ["<resolution_1>", ...],
  "top_picks": [
    {"symbol": "<code>", "action": "buy"|"hold"|"sell", "conviction": "<level>", "rationale": "<why>", "target_weight": <0.0~1.0>}
  ],
  "portfolio_narrative": "<3-5 句投资论点因果链>",
  "risk_adjusted_exposure": <float, 0.0 ~ 1.0>,
  "dissenting_views": ["<minority_opinion_1>", ...],
  "conviction_drivers": ["<driver_1>", ...],
  "time_horizon_weights": {"short_term": <float>, "medium_term": <float>, "long_term": <float>}
}

约束:
- final_score 不得偏离算法 ensemble baseline 的 aggregate_score 超过 ±0.30
- debate_rounds 必须包含 3 轮辩论
- 必须保留少数派意见（dissenting_views），即使不采纳
- conviction_drivers 应清晰列出最终决策的 2-3 个核心驱动因素
- time_horizon_weights 三项之和应为 1.0
- portfolio_narrative 必须清晰说明投资逻辑的因果链
"""

    async def deliberate(self, agent_input: MasterAgentInput) -> MasterAgentOutput:
        """主持 IC 会议并产出最终决策。"""
        t0 = time.monotonic()
        input_json = agent_input.model_dump_json(indent=2)
        system_prompt = self._get_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是本次 IC 会议的全部研究材料（含各分支专属报告和分歧矩阵），请主持会议并做出最终决策：\n\n{input_json}"},
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
        """Parse, validate, apply IC deviation bounds and risk veto."""
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
        raw_confidence = _clamp(float(raw.get("confidence", 0.5)), 0.0, 1.0)

        # Reduce confidence based on max disagreement
        max_disagreement = 0.0
        for pairs in agent_input.branch_disagreement_matrix.values():
            for d in pairs.values():
                max_disagreement = max(max_disagreement, d)
        if max_disagreement > 0.6:
            penalty = (max_disagreement - 0.6) * 0.5
            raw_confidence = max(0.1, raw_confidence - penalty)
        raw["confidence"] = raw_confidence

        # Bound risk_adjusted_exposure
        raw["risk_adjusted_exposure"] = _clamp(
            float(raw.get("risk_adjusted_exposure", 0.5)), 0.0, 1.0,
        )

        # Validate conviction label
        valid_convictions = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        if raw.get("final_conviction") not in valid_convictions:
            raw["final_conviction"] = self._score_to_conviction(bounded_score)

        # Risk veto logic
        risk_report = agent_input.risk_report
        if risk_report:
            if risk_report.risk_assessment == "extreme":
                # Hard veto
                if raw["final_conviction"] in ("strong_buy", "buy"):
                    raw["final_conviction"] = "neutral"
                    raw["final_score"] = min(bounded_score, 0.1)
                    raw.setdefault("debate_resolution", []).append(
                        "风控官一票否决：风险评估为 extreme，conviction 已降级至 neutral"
                    )
            elif risk_report.risk_assessment == "high":
                # Soft veto: downgrade one level
                conviction_downgrade = {
                    "strong_buy": "buy",
                    "strong_sell": "sell",
                }
                current = raw["final_conviction"]
                if current in conviction_downgrade:
                    raw["final_conviction"] = conviction_downgrade[current]
                    raw.setdefault("debate_resolution", []).append(
                        f"风控官软否决：风险评估为 high，conviction 从 {current} 降级至 {raw['final_conviction']}"
                    )

        # Parse debate_rounds
        raw_rounds = raw.get("debate_rounds", [])
        parsed_rounds: list[ICDebateRound] = []
        if isinstance(raw_rounds, list):
            for r in raw_rounds:
                if isinstance(r, dict):
                    try:
                        parsed_rounds.append(ICDebateRound.model_validate(r))
                    except Exception:
                        pass
        raw["debate_rounds"] = parsed_rounds

        # Parse top_picks
        raw_picks = raw.get("top_picks", [])
        parsed_picks: list[SymbolRecommendation] = []
        if isinstance(raw_picks, list):
            for pick in raw_picks:
                if isinstance(pick, dict):
                    try:
                        parsed_picks.append(SymbolRecommendation.model_validate(pick))
                    except Exception:
                        pass
        raw["top_picks"] = parsed_picks

        # Ensure list fields
        for field in ("consensus_areas", "disagreement_areas", "debate_resolution",
                       "dissenting_views", "conviction_drivers"):
            if not isinstance(raw.get(field), list):
                raw[field] = []

        # Ensure time_horizon_weights
        if not isinstance(raw.get("time_horizon_weights"), dict):
            raw["time_horizon_weights"] = {"short_term": 0.3, "medium_term": 0.4, "long_term": 0.3}

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
