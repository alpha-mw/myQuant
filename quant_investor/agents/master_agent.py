"""
V10 Master Agent（IC 投资委员会主席）。

综合所有分支 SubAgent 和风控 SubAgent 的研报，
模拟 IC 会议流程，产出最终投资建议。
"""

from __future__ import annotations

import time
from typing import Any

from quant_investor.agents.agent_contracts import (
    MasterAgentInput,
    MasterAgentOutput,
    SymbolRecommendation,
    WhatIfScenario,
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
    """IC 主席：汇总 6 份研报，主持辩论，做出最终决策。"""

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

        # Apply risk veto: if risk assessment is extreme, cap conviction
        risk_report = agent_input.risk_report
        if risk_report and risk_report.risk_assessment == "extreme":
            if raw["final_conviction"] in ("strong_buy", "buy"):
                raw["final_conviction"] = "neutral"
                raw["final_score"] = min(bounded_score, 0.1)
                if "risk_warnings" not in raw:
                    raw.setdefault("disagreement_areas", [])
                raw.setdefault("debate_resolution", []).append(
                    "风控官一票否决：风险评估为 extreme，conviction 已降级至 neutral"
                )

        # Parse top_picks into SymbolRecommendation
        raw_picks = raw.get("top_picks", [])
        parsed_picks = []
        for pick in raw_picks:
            if isinstance(pick, dict):
                try:
                    parsed_picks.append(self._normalize_symbol_recommendation(pick))
                except Exception:
                    pass
        raw["top_picks"] = parsed_picks

        # Ensure list fields
        for field in ("consensus_areas", "disagreement_areas", "debate_resolution", "dissenting_views"):
            if not isinstance(raw.get(field), list):
                raw[field] = []

        return MasterAgentOutput.model_validate(raw)

    def _normalize_symbol_recommendation(self, payload: dict[str, Any]) -> SymbolRecommendation:
        recommendation = SymbolRecommendation.model_validate(payload)
        recommendation = self._attach_position_size_hint(recommendation)
        recommendation = self._sanitize_price_structure(recommendation)
        return recommendation

    @staticmethod
    def _attach_position_size_hint(recommendation: SymbolRecommendation) -> SymbolRecommendation:
        payload = recommendation.model_dump()
        if payload.get("position_size_pct") is None and payload.get("target_weight") is not None:
            payload["position_size_pct"] = float(payload.get("target_weight", 0.0))
        return SymbolRecommendation.model_validate(payload)

    @staticmethod
    def _sanitize_price_structure(recommendation: SymbolRecommendation) -> SymbolRecommendation:
        payload = recommendation.model_dump()
        action = str(payload.get("action", "hold")).strip().lower()
        entry_price = payload.get("entry_price")
        target_price = payload.get("target_price")
        stop_loss = payload.get("stop_loss")

        if not MasterAgent._prices_are_valid(entry_price, target_price, stop_loss, action):
            payload["entry_price"] = None
            payload["target_price"] = None
            payload["stop_loss"] = None
            payload["risk_reward_ratio"] = None
            payload["what_if_scenarios"] = MasterAgent._normalize_what_if_scenarios(
                payload.get("what_if_scenarios", [])
            )
            return SymbolRecommendation.model_validate(payload)

        ratio = payload.get("risk_reward_ratio")
        if ratio is None:
            payload["risk_reward_ratio"] = MasterAgent._compute_risk_reward_ratio(
                entry_price=float(entry_price),
                target_price=float(target_price),
                stop_loss=float(stop_loss),
                action=action,
            )

        payload["what_if_scenarios"] = MasterAgent._normalize_what_if_scenarios(
            payload.get("what_if_scenarios", [])
        )
        return SymbolRecommendation.model_validate(payload)

    @staticmethod
    def _normalize_what_if_scenarios(payload: Any) -> list[WhatIfScenario]:
        if not isinstance(payload, list):
            return []
        scenarios: list[WhatIfScenario] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                scenarios.append(WhatIfScenario.model_validate(item))
            except Exception:
                continue
        return scenarios

    @staticmethod
    def _prices_are_valid(
        entry_price: Any,
        target_price: Any,
        stop_loss: Any,
        action: str,
    ) -> bool:
        try:
            entry = float(entry_price)
            target = float(target_price)
            stop = float(stop_loss)
        except (TypeError, ValueError):
            return False
        if min(entry, target, stop) <= 0:
            return False
        if action == "buy":
            return stop < entry < target
        if action == "sell":
            return target < entry < stop
        return True

    @staticmethod
    def _compute_risk_reward_ratio(
        *,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        action: str,
    ) -> float | None:
        try:
            if action == "sell":
                risk = stop_loss - entry_price
                reward = entry_price - target_price
            else:
                risk = entry_price - stop_loss
                reward = target_price - entry_price
            if risk <= 0:
                return None
            return _clamp(reward / risk, -1.0, 100.0)
        except Exception:
            return None

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
