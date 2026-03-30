"""
QuantAgent：对现有量化分支做 agent 化包装。
"""

from __future__ import annotations

from typing import Any, Mapping

from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline
from quant_investor.agents.base import BaseAgent


class QuantAgent(BaseAgent):
    """以 deterministic 信号为主的量化 research agent。"""

    agent_name = "QuantAgent"
    MAX_SCORE_ADJUSTMENT = 0.10
    MAX_CONFIDENCE_ADJUSTMENT = 0.15

    def run(self, payload: Mapping[str, Any]) -> Any:
        envelope = self.ensure_payload(payload)
        data_bundle = envelope.get("data_bundle")
        if not isinstance(data_bundle, UnifiedDataBundle):
            raise TypeError("QuantAgent 需要 `data_bundle: UnifiedDataBundle`")

        stock_pool = list(envelope.get("stock_pool") or data_bundle.symbols)
        pipeline = ParallelResearchPipeline(
            stock_pool=stock_pool,
            market=str(envelope.get("market", data_bundle.market or "CN")),
            enable_alpha_mining=bool(envelope.get("enable_alpha_mining", True)),
            verbose=bool(envelope.get("verbose", False)),
        )
        result = pipeline._run_quant_branch(data_bundle)

        score_adjustment = self.clamp(
            float(envelope.get("score_adjustment", 0.0)),
            -self.MAX_SCORE_ADJUSTMENT,
            self.MAX_SCORE_ADJUSTMENT,
        )
        confidence_adjustment = self.clamp(
            float(envelope.get("confidence_adjustment", 0.0)),
            -self.MAX_CONFIDENCE_ADJUSTMENT,
            self.MAX_CONFIDENCE_ADJUSTMENT,
        )
        if score_adjustment or confidence_adjustment:
            result.final_score = self.clamp(float(result.score) + score_adjustment, -1.0, 1.0)
            result.final_confidence = self.clamp(
                float(result.confidence) + confidence_adjustment,
                0.0,
                1.0,
            )
            result.diagnostic_notes.append(
                "QuantAgent 仅对 deterministic 量化结论施加了 bounded adjustment / confidence 修正。"
            )

        thesis = self._build_thesis(result)
        return self.branch_result_to_verdict(
            result,
            thesis=thesis,
            metadata={
                "alpha_factors": list(result.signals.get("alpha_factors", [])),
                "deterministic_primary": True,
            },
        )

    def _build_thesis(self, result) -> str:
        factors = [str(item) for item in result.signals.get("alpha_factors", []) if str(item).strip()]
        if factors:
            factor_text = "、".join(factors[:3])
            return f"量化分支当前主要依据 {factor_text} 等 deterministic 因子形成判断。"
        if str(result.explanation or "").strip():
            return str(result.explanation).strip()
        return "量化分支当前以 deterministic 因子信号形成中性判断。"
