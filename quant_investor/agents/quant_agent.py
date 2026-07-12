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
        runtime_ready = bool(
            mined.factor_count > 0
            and mined.production_eligible
            and mined.governance_status == "ready"
        )
        if runtime_ready:
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
            symbol_scores = {symbol: 0.0 for symbol in stock_pool}
            factors_used = []
            factor_mode = "governance_blocked"
            conclusion = (
                "量化分支没有通过 FactorGovernanceProtocol v2 完整运行时契约的因子，"
                "按治理协议阻断量化证据；"
                "不会回退到 legacy proxy。"
            )
            investment_risks = [
                "当前 selectable 记录未通过 v2 protocol/set/slot/budget/evidence 完整门禁。",
                "量化分支置信度为 0；不得以收益/波动率代理替代缺失 alpha。",
            ]
            coverage_notes = [
                f"symbols={len(symbol_scores)}",
                "governance_blocked_no_protocol_eligible_production_factor",
            ]
            diagnostic_notes = [
                "mined_factor_runtime_contract_not_ready",
                "legacy_fallback_forbidden",
                *[f"factor_runtime_blocker:{item}" for item in mined.runtime_blockers],
            ]
            reliability = 0.0

        branch_confidence = (
            self.clamp(
                0.38
                + min(len(symbol_scores), 20) / 50.0
                + min(mined.factor_count, 5) * 0.02,
                0.0,
                1.0,
            )
            if runtime_ready
            else 0.0
        )

        result = BranchResult(
            branch_name="quant",
            final_score=float(fmean(symbol_scores.values()) if symbol_scores else 0.0),
            final_confidence=branch_confidence,
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
        if (score_adjustment or confidence_adjustment) and runtime_ready:
            result.final_score = self.clamp(float(result.score) + score_adjustment, -1.0, 1.0)
            result.final_confidence = self.clamp(
                float(result.confidence) + confidence_adjustment,
                0.0,
                1.0,
            )
            result.diagnostic_notes.append(
                "QuantAgent 仅对 deterministic 量化结论施加了 bounded adjustment / confidence 修正。"
            )
        elif score_adjustment or confidence_adjustment:
            result.diagnostic_notes.append(
                "governance_blocked_adjustment_ignored"
            )

        thesis = self._build_thesis(result)
        return self.branch_result_to_verdict(
            result,
            thesis=thesis,
            metadata={
                "alpha_factors": list(result.signals.get("alpha_factors", [])),
                "deterministic_primary": True,
                "factor_mode": result.signals.get("factor_mode", "governance_blocked"),
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
