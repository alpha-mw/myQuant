"""
市场级 MacroAgent。
"""

from __future__ import annotations

from typing import Any, Mapping

from quant_investor.agent_protocol import (
    AgentStatus,
    BranchVerdict,
    CoverageScope,
    EvidenceItem,
)
from quant_investor.agents.base import BaseAgent


class MacroAgent(BaseAgent):
    """一次运行只生成一个市场级宏观 verdict。"""

    agent_name = "MacroAgent"

    def run(self, payload: Mapping[str, Any]) -> BranchVerdict:
        envelope = self.ensure_payload(payload)
        snapshot = envelope.get("market_snapshot", envelope)
        if not isinstance(snapshot, Mapping):
            raise TypeError("market_snapshot 必须是 Mapping")

        regime = str(
            snapshot.get("regime")
            or snapshot.get("market_regime")
            or snapshot.get("macro_regime")
            or "neutral"
        )
        macro_score = self._get_float(snapshot, "macro_score", default=0.0)
        liquidity_score = self._get_float(snapshot, "liquidity_score", default=0.0)
        volatility_pct = self._get_float(
            snapshot,
            "volatility_percentile",
            "macro_volatility_percentile",
            default=50.0,
        )
        policy_signal = str(
            snapshot.get("policy_signal")
            or snapshot.get("monetary_policy_signal")
            or "neutral"
        ).lower()

        score = 0.55 * macro_score + 0.25 * liquidity_score - 0.25 * max(0.0, (volatility_pct - 50.0) / 50.0)
        if any(token in policy_signal for token in ("tight", "restrict", "hawk")):
            score -= 0.15
        if any(token in policy_signal for token in ("ease", "support", "dovish")):
            score += 0.1
        score = self.clamp(score, -1.0, 1.0)

        target_gross_exposure = self.clamp(0.55 + score * 0.35, 0.1, 1.0)
        if volatility_pct >= 80.0:
            target_gross_exposure = self.clamp(target_gross_exposure - 0.15, 0.05, 1.0)
        if volatility_pct <= 25.0:
            target_gross_exposure = self.clamp(target_gross_exposure + 0.05, 0.05, 1.0)

        style_bias = self._infer_style_bias(regime=regime, score=score, liquidity_score=liquidity_score)
        confidence = self._infer_confidence(snapshot)
        status = AgentStatus.SUCCESS if self._coverage_ratio(snapshot) >= 0.6 else AgentStatus.DEGRADED

        evidence = [
            EvidenceItem(
                source=self.agent_name,
                summary=(
                    f"regime={regime}, liquidity_score={liquidity_score:.2f}, "
                    f"volatility_pct={volatility_pct:.1f}, policy={policy_signal}"
                ),
                direction=self.score_to_direction(score),
                score=score,
                confidence=confidence,
                scope=CoverageScope.MARKET,
            )
        ]

        coverage_notes = []
        if "macro_score" not in snapshot:
            coverage_notes.append("market_snapshot 未提供 macro_score，已按 0.0 处理。")
        if "liquidity_score" not in snapshot:
            coverage_notes.append("market_snapshot 未提供 liquidity_score，已按 0.0 处理。")

        thesis = (
            f"当前宏观 regime 为 {regime}，建议将总暴露控制在 "
            f"{target_gross_exposure:.0%} 左右，风格偏向 {style_bias}。"
        )

        return BranchVerdict(
            agent_name=self.agent_name,
            thesis=thesis,
            symbol=None,
            status=status,
            direction=self.score_to_direction(score),
            action=self.score_to_action(score),
            confidence_label=self.confidence_to_label(confidence),
            final_score=score,
            final_confidence=confidence,
            evidence=evidence,
            coverage_notes=coverage_notes,
            metadata={
                "symbol": None,
                "regime": regime,
                "target_gross_exposure": target_gross_exposure,
                "style_bias": style_bias,
            },
        )

    @staticmethod
    def _get_float(snapshot: Mapping[str, Any], *keys: str, default: float) -> float:
        for key in keys:
            if key in snapshot and snapshot[key] is not None:
                return float(snapshot[key])
        return float(default)

    @staticmethod
    def _coverage_ratio(snapshot: Mapping[str, Any]) -> float:
        required = ("regime", "macro_score", "liquidity_score")
        present = sum(1 for key in required if snapshot.get(key) is not None)
        return present / len(required)

    def _infer_confidence(self, snapshot: Mapping[str, Any]) -> float:
        ratio = self._coverage_ratio(snapshot)
        extra = 0.1 if snapshot.get("policy_signal") or snapshot.get("monetary_policy_signal") else 0.0
        return self.clamp(0.35 + ratio * 0.45 + extra, 0.2, 0.9)

    @staticmethod
    def _infer_style_bias(regime: str, score: float, liquidity_score: float) -> str:
        regime_text = regime.lower()
        if score <= -0.25 or any(token in regime_text for token in ("risk_off", "defensive", "tight")):
            return "defensive_quality"
        if score >= 0.25 and liquidity_score >= 0.0:
            return "cyclical_growth"
        return "balanced_quality"
