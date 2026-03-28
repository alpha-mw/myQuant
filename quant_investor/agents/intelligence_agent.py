"""
IntelligenceAgent：对现有智能融合分支做 agent 化包装。
"""

from __future__ import annotations

from typing import Any, Mapping

from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline
from quant_investor.agents.base import BaseAgent


class IntelligenceAgent(BaseAgent):
    """只包装事件/情绪/资金流/广度/行业轮动，不复用财务主分。"""

    agent_name = "IntelligenceAgent"
    ALLOWED_SIGNAL_KEYS = {
        "intelligence_score",
        "event_risk_score",
        "sentiment_score",
        "money_flow_score",
        "breadth_score",
        "rotation_score",
        "alerts",
    }

    def run(self, payload: Mapping[str, Any]) -> Any:
        envelope = self.ensure_payload(payload)
        data_bundle = envelope.get("data_bundle")
        if not isinstance(data_bundle, UnifiedDataBundle):
            raise TypeError("IntelligenceAgent 需要 `data_bundle: UnifiedDataBundle`")

        stock_pool = list(envelope.get("stock_pool") or data_bundle.symbols)
        pipeline = ParallelResearchPipeline(
            stock_pool=stock_pool,
            market=str(envelope.get("market", data_bundle.market or "CN")),
            verbose=bool(envelope.get("verbose", False)),
        )
        pipeline._market_regime = envelope.get("market_regime")
        result = pipeline._run_intelligence_branch(data_bundle)

        filtered_signals = {
            key: value
            for key, value in dict(result.signals).items()
            if key in self.ALLOWED_SIGNAL_KEYS
        }
        thesis = self._build_thesis(result)
        return self.branch_result_to_verdict(
            result,
            thesis=thesis,
            metadata={
                "allowed_signal_keys": sorted(filtered_signals.keys()),
                "branch_mode": result.metadata.get("branch_mode", "structured_intelligence_fusion"),
                "no_financial_primary_scoring": True,
            },
        )

    @staticmethod
    def _build_thesis(result) -> str:
        alerts = [str(item) for item in result.signals.get("alerts", []) if str(item).strip()]
        if alerts:
            return (
                "智能融合分支当前仅根据新闻、公告事件、情绪、资金流、广度与行业轮动形成判断，"
                f"重点预警包括 {alerts[0]}"
            )
        return (
            "智能融合分支当前仅根据新闻、公告事件、情绪、资金流、广度与行业轮动形成结构化判断。"
        )
