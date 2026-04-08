"""
QuantAgent：对 deterministic 量化信号做轻量包装。
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping

from quant_investor.agents.base import BaseAgent
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle


class QuantAgent(BaseAgent):
    """以 deterministic 信号为主的量化 research agent。"""

    agent_name = "QuantAgent"
    MAX_SCORE_ADJUSTMENT = 0.10
    MAX_CONFIDENCE_ADJUSTMENT = 0.15

    @staticmethod
    def _frame_summary(frame: Any) -> dict[str, float]:
        if frame is None or getattr(frame, "empty", True):
            return {"average_return": 0.0, "volatility": 0.0}
        working = frame.copy()
        close_col = "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
        if not close_col:
            return {"average_return": 0.0, "volatility": 0.0}
        close = working[close_col].astype(float)
        returns = close.pct_change().dropna()
        return {
            "average_return": float(returns.tail(20).mean()) if not returns.empty else 0.0,
            "volatility": float(returns.tail(60).std()) if len(returns) >= 3 else 0.0,
        }

    def run(self, payload: Mapping[str, Any]) -> Any:
        envelope = self.ensure_payload(payload)
        data_bundle = envelope.get("data_bundle")
        if not isinstance(data_bundle, UnifiedDataBundle):
            raise TypeError("QuantAgent 需要 `data_bundle: UnifiedDataBundle`")

        stock_pool = list(envelope.get("stock_pool") or data_bundle.symbols)
        symbol_scores: dict[str, float] = {}
        for symbol in stock_pool:
            summary = self._frame_summary(data_bundle.symbol_data.get(symbol))
            score = summary["average_return"] * 8.0 - summary["volatility"] * 2.0
            symbol_scores[symbol] = self.clamp(score, -1.0, 1.0)

        result = BranchResult(
            branch_name="quant",
            final_score=float(fmean(symbol_scores.values()) if symbol_scores else 0.0),
            final_confidence=self.clamp(0.35 + min(len(symbol_scores), 20) / 50.0, 0.0, 1.0),
            symbol_scores=symbol_scores,
            conclusion="量化分支基于收益/波动率代理形成 deterministic 结论。",
            signals={
                "branch_mode": "deterministic_cross_section",
                "alpha_factors": ["short_term_return", "volatility_penalty"],
            },
            investment_risks=["量化分支当前未调用旧 batch pipeline，只使用轻量横截面代理。"],
            coverage_notes=[f"symbols={len(symbol_scores)}", "legacy batch retired"],
            diagnostic_notes=["legacy_batch_internal_retired"],
            metadata={"deterministic_primary": True, "reliability": 0.70},
        )

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
