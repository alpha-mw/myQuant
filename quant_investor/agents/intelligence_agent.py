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
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except Exception:
            return default
        return number if number == number else default

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
        structured_count = 0
        proxy_count = 0
        required = {
            "intelligence_score",
            "event_risk_score",
            "sentiment_score",
            "money_flow_score",
            "breadth_score",
            "rotation_score",
        }
        for symbol in stock_pool:
            sentiment = dict(data_bundle.sentiment_data.get(symbol, {}) or {})
            if required.issubset(set(sentiment)):
                structured_count += 1
                intelligence_score = self._number(sentiment.get("intelligence_score"))
                event_risk = self._number(sentiment.get("event_risk_score"))
                sentiment_score = self._number(sentiment.get("sentiment_score"))
                money_flow = self._number(sentiment.get("money_flow_score"))
                breadth = self._number(sentiment.get("breadth_score"))
                rotation = self._number(sentiment.get("rotation_score"))
                signal = self.clamp(
                    0.40 * intelligence_score
                    + 0.20 * sentiment_score
                    + 0.20 * money_flow
                    + 0.10 * breadth
                    + 0.10 * rotation
                    - 0.25 * max(event_risk, 0.0),
                    -1.0,
                    1.0,
                )
            else:
                proxy_count += 1
                momentum, flow = self._frame_signal(data_bundle.symbol_data.get(symbol))
                signal = self.clamp(momentum * 6.0 + flow * 0.2, -1.0, 1.0)
            symbol_scores[symbol] = signal
            flow_for_alert = self._number(sentiment.get("money_flow_score"), 0.0) if sentiment else signal
            if flow_for_alert < -0.4:
                alerts.append(f"{symbol} volume_flow_negative")

        average_score = float(fmean(symbol_scores.values()) if symbol_scores else 0.0)
        mode = "local_intelligence_mart" if structured_count else "price_volume_proxy"
        result = BranchResult(
            branch_name="intelligence",
            final_score=average_score,
            final_confidence=self.clamp(0.30 + min(len(symbol_scores), 20) / 60.0, 0.0, 1.0),
            symbol_scores=symbol_scores,
            conclusion=(
                "智能融合分支基于本地 intelligence mart 的资金流、事件风险、情绪、广度和轮动字段形成判断。"
                if structured_count
                else "智能融合分支未发现本地 intelligence mart 记录，仅使用价格/成交量代理并标记降级。"
            ),
            signals={
                "branch_mode": mode,
                "intelligence_score": average_score,
                "event_risk_score": min(0.0, average_score),
                "money_flow_score": average_score,
                "alerts": alerts[:5],
            },
            investment_risks=(
                ["intelligence mart 缺失时不调用旧 NewsAnalyzer/网页/LLM；当前仅价格成交量代理。"]
                if proxy_count
                else ["intelligence mart 为本地离线快照；需依赖 readiness provenance 判断来源。"]
            ),
            coverage_notes=[
                f"symbols={len(symbol_scores)}",
                f"structured_symbols={structured_count}",
                f"proxy_symbols={proxy_count}",
                "legacy batch retired",
            ],
            diagnostic_notes=["legacy_batch_internal_retired", f"branch_mode={mode}"],
            metadata={"branch_mode": mode, "reliability": 0.65 if structured_count else 0.35},
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
                "branch_mode": result.metadata.get("branch_mode", "price_volume_proxy"),
                "no_financial_primary_scoring": True,
            },
        )

    @staticmethod
    def _build_thesis(result) -> str:
        alerts = [str(item) for item in result.signals.get("alerts", []) if str(item).strip()]
        if alerts:
            return (
                "智能融合分支当前根据本地结构化 intelligence mart 字段形成判断，"
                f"重点预警包括 {alerts[0]}"
            )
        return (
            str(result.conclusion or "智能融合分支已完成结构化判断。")
        )
