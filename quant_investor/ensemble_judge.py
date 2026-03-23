#!/usr/bin/env python3
"""
V9 集成裁判辅助模块。

只消费各分支 final_score / final_confidence，不把 debate 当成独立分支输入。
"""

from __future__ import annotations

from typing import Any

from quant_investor.branch_contracts import BranchResult


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class EnsembleJudge:
    """V9 分支级集成裁判。"""

    REGIME_WEIGHTS: dict[str, dict[str, float]] = {
        "default": {
            "kline": 0.23,
            "quant": 0.24,
            "fundamental": 0.21,
            "intelligence": 0.18,
            "macro": 0.14,
        },
        "趋势上涨": {
            "kline": 0.29,
            "quant": 0.23,
            "fundamental": 0.18,
            "intelligence": 0.16,
            "macro": 0.14,
        },
        "趋势下跌": {
            "kline": 0.21,
            "quant": 0.18,
            "fundamental": 0.16,
            "intelligence": 0.17,
            "macro": 0.28,
        },
        "震荡低波": {
            "kline": 0.16,
            "quant": 0.26,
            "fundamental": 0.24,
            "intelligence": 0.19,
            "macro": 0.15,
        },
        "震荡高波": {
            "kline": 0.18,
            "quant": 0.23,
            "fundamental": 0.20,
            "intelligence": 0.17,
            "macro": 0.22,
        },
    }

    @classmethod
    def from_fundamental(cls, branch_results: dict[str, BranchResult]) -> BranchResult | None:
        return branch_results.get("fundamental")

    @classmethod
    def branch_weights(cls, market_regime: str | None = None) -> dict[str, float]:
        regime_key = market_regime if market_regime in cls.REGIME_WEIGHTS else "default"
        return dict(cls.REGIME_WEIGHTS[regime_key])

    @classmethod
    def combine(
        cls,
        branch_results: dict[str, BranchResult],
        market_regime: str | None = None,
    ) -> dict[str, Any]:
        weights = cls.branch_weights(market_regime)
        branch_consensus: dict[str, float] = {}
        weighted_score_sum = 0.0
        weighted_conf_sum = 0.0
        weighted_total = 0.0
        base_total = 0.0

        for branch_name, base_weight in weights.items():
            branch = branch_results.get(branch_name)
            if branch is None:
                continue
            final_score = float(branch.final_score if branch.final_score is not None else branch.score)
            final_confidence = float(
                branch.final_confidence if branch.final_confidence is not None else branch.confidence
            )
            branch_consensus[branch_name] = round(final_score, 4)
            effective_weight = base_weight * max(final_confidence, 0.05)
            weighted_score_sum += final_score * effective_weight
            weighted_conf_sum += final_confidence * base_weight
            weighted_total += effective_weight
            base_total += base_weight

        aggregate_score = weighted_score_sum / weighted_total if weighted_total > 0 else 0.0
        aggregate_confidence = weighted_conf_sum / base_total if base_total > 0 else 0.0
        return {
            "branch_consensus": branch_consensus,
            "aggregate_score": _clamp(aggregate_score, -1.0, 1.0),
            "aggregate_confidence": _clamp(aggregate_confidence, 0.0, 1.0),
            "weights": weights,
        }
