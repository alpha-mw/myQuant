"""
QuantAgent：对 deterministic 量化信号做轻量包装。
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping

from quant_investor.agents.base import BaseAgent
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.factors.runtime import score_with_mined_factors


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
        close_col = (
            "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
        )
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
        frames = {symbol: data_bundle.symbol_data.get(symbol) for symbol in stock_pool}
        mined = score_with_mined_factors(frames)
        if mined.factor_count > 0:
            symbol_scores = dict(mined.symbol_scores)
            factors_used = list(mined.factors_used)
            factor_mode = "governed_mined_factors"
            conclusion = (
                "量化分支基于通过 8 道门的 production mined factors 形成 deterministic 结论。"
            )
            investment_risks = [
                "只消费 production_factor；paper/research 因子不会进入选股。",
                f"mined_factor_coverage={mined.coverage_rate:.2%}",
            ]
            coverage_notes = [
                f"symbols={len(symbol_scores)}",
                f"production_factors={mined.factor_count}",
                f"factor_coverage={mined.coverage_rate:.2%}",
            ]
            diagnostic_notes = ["mined_factor_registry_enforced"]
            reliability = self.clamp(0.72 + min(mined.factor_count, 5) * 0.03, 0.0, 0.90)
        else:
            symbol_scores = {}
            for symbol in stock_pool:
                summary = self._frame_summary(data_bundle.symbol_data.get(symbol))
                score = summary["average_return"] * 8.0 - summary["volatility"] * 2.0
                symbol_scores[symbol] = self.clamp(score, -1.0, 1.0)
            factors_used = ["short_term_return", "volatility_penalty"]
            factor_mode = "legacy_proxy_fallback"
            conclusion = "量化分支未发现可用 production mined factors，回退到收益/波动率代理。"
            investment_risks = [
                "没有已通过 8 道门并被人工确认为 production_factor 的 mined factor。",
                "当前仅使用 legacy short-term-return / volatility-penalty proxy。",
            ]
            coverage_notes = [
                f"symbols={len(symbol_scores)}",
                "legacy_fallback_until_factor_approval",
            ]
            diagnostic_notes = [
                "legacy_proxy_fallback",
                "mined_factor_registry_empty_or_not_selectable",
            ]
            reliability = 0.55

        result = BranchResult(
            branch_name="quant",
            final_score=float(fmean(symbol_scores.values()) if symbol_scores else 0.0),
            final_confidence=self.clamp(
                0.38 + min(len(symbol_scores), 20) / 50.0 + min(mined.factor_count, 5) * 0.02,
                0.0,
                1.0,
            ),
            symbol_scores=symbol_scores,
            conclusion=conclusion,
            signals={
                "branch_mode": "deterministic_cross_section",
                "factor_mode": factor_mode,
                "alpha_factors": factors_used,
            },
            investment_risks=investment_risks,
            coverage_notes=coverage_notes,
            diagnostic_notes=diagnostic_notes,
            metadata={
                "deterministic_primary": True,
                "reliability": reliability,
                "factor_mode": factor_mode,
                "mined_factor_runtime": mined.to_metadata(),
            },
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
                "factor_mode": result.signals.get("factor_mode", "legacy_proxy_fallback"),
            },
        )

    def _build_thesis(self, result) -> str:
        factors = [
            str(item) for item in result.signals.get("alpha_factors", []) if str(item).strip()
        ]
        if factors:
            factor_text = "、".join(factors[:3])
            return f"量化分支当前主要依据 {factor_text} 等 deterministic 因子形成判断。"
        if str(result.explanation or "").strip():
            return str(result.explanation).strip()
        return "量化分支当前以 deterministic 因子信号形成中性判断。"
