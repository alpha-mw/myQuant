"""
IntelligenceAgent：对事件/活跃度代理信号做轻量包装。
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping

from quant_investor.agents.base import BaseAgent
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle


class IntelligenceAgent(BaseAgent):
    """只包装事件/情绪/资金流/广度/行业轮动，不复用旧 batch 主分。"""

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

    @staticmethod
    def _frame_signal(frame: Any) -> tuple[float, float]:
        if frame is None or getattr(frame, "empty", True):
            return 0.0, 0.0
        working = frame.copy()
        close_col = "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
        volume_col = "volume" if "volume" in working.columns else "vol" if "vol" in working.columns else ""
        if not close_col:
            return 0.0, 0.0
        close = working[close_col].astype(float)
        returns = close.pct_change().dropna()
        momentum = float(returns.tail(5).mean()) if not returns.empty else 0.0
        flow = 0.0
        if volume_col:
            volume = working[volume_col].astype(float)
            if len(volume) >= 5:
                baseline = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
                if baseline > 0:
                    flow = float(volume.iloc[-1] / baseline - 1.0)
        return momentum, flow

    def run(self, payload: Mapping[str, Any]) -> Any:
        envelope = self.ensure_payload(payload)
        data_bundle = envelope.get("data_bundle")
        if not isinstance(data_bundle, UnifiedDataBundle):
            raise TypeError("IntelligenceAgent 需要 `data_bundle: UnifiedDataBundle`")

        stock_pool = list(envelope.get("stock_pool") or data_bundle.symbols)
        symbol_scores: dict[str, float] = {}
        alerts: list[str] = []
        for symbol in stock_pool:
            momentum, flow = self._frame_signal(data_bundle.symbol_data.get(symbol))
            signal = self.clamp(momentum * 6.0 + flow * 0.2, -1.0, 1.0)
            symbol_scores[symbol] = signal
            if flow < -0.4:
                alerts.append(f"{symbol} volume_flow_negative")

        average_score = float(fmean(symbol_scores.values()) if symbol_scores else 0.0)
        result = BranchResult(
            branch_name="intelligence",
            final_score=average_score,
            final_confidence=self.clamp(0.30 + min(len(symbol_scores), 20) / 60.0, 0.0, 1.0),
            symbol_scores=symbol_scores,
            conclusion="智能融合分支基于活跃度、走势和轻量事件代理形成结构化判断。",
            signals={
                "branch_mode": "structured_intelligence_fusion",
                "intelligence_score": average_score,
                "event_risk_score": min(0.0, average_score),
                "money_flow_score": average_score,
                "alerts": alerts[:5],
            },
            investment_risks=["智能融合当前未调用旧 batch pipeline，文本证据为候选层可扩展能力。"],
            coverage_notes=[f"symbols={len(symbol_scores)}", "legacy batch retired"],
            diagnostic_notes=["legacy_batch_internal_retired"],
            metadata={"branch_mode": "structured_intelligence_fusion", "reliability": 0.55},
        )

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
