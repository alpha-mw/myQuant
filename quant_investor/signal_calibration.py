#!/usr/bin/env python3
"""
信号校准层

统一将五个研究分支映射为可比较的预期收益、置信度和可靠度。
"""

from __future__ import annotations

from typing import Any

from quant_investor.contracts import BranchResult, CalibratedBranchSignal


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class SignalCalibrator:
    """把不同研究分支映射到统一的信号量纲。"""

    HORIZON_BY_BRANCH = {
        "kline": 5,
        "quant": 5,
        "llm_debate": 10,
        "intelligence": 10,
        "macro": 20,
    }

    RETURN_SCALE_BY_BRANCH = {
        "kline": 0.14,
        "quant": 0.12,
        "llm_debate": 0.07,
        "intelligence": 0.08,
        "macro": 0.05,
    }

    DEFAULT_MODE_BY_BRANCH = {
        "kline": "kline_dual_model",
        "quant": "alpha_research",
        "llm_debate": "structured_research_debate",
        "intelligence": "structured_intelligence_fusion",
        "macro": "macro_terminal",
    }

    def __init__(self, symbol_provenance: dict[str, dict[str, Any]] | None = None) -> None:
        self.symbol_provenance = symbol_provenance or {}

    def calibrate_all(
        self,
        branch_results: dict[str, BranchResult],
        symbols: list[str],
    ) -> dict[str, CalibratedBranchSignal]:
        """批量校准全部分支。"""
        return {
            name: self.calibrate_branch(name, branch, symbols)
            for name, branch in branch_results.items()
        }

    def calibrate_branch(
        self,
        branch_name: str,
        branch: BranchResult,
        symbols: list[str],
    ) -> CalibratedBranchSignal:
        """校准单个分支。"""
        horizon = int(branch.metadata.get("horizon_days", self.HORIZON_BY_BRANCH.get(branch_name, 5)))
        mode = str(branch.metadata.get("branch_mode", self.DEFAULT_MODE_BY_BRANCH.get(branch_name, "unknown")))
        branch_reliability = self._branch_reliability(branch)
        scale = self.RETURN_SCALE_BY_BRANCH.get(branch_name, 0.08)

        expected_from_signal = branch.signals.get("expected_return") or branch.signals.get("predicted_return")
        symbol_expected_returns: dict[str, float] = {}
        symbol_confidences: dict[str, float] = {}
        symbol_convictions: dict[str, float] = {}

        for symbol in symbols:
            symbol_reliability = self._symbol_reliability(symbol)
            conviction = float(branch.symbol_scores.get(symbol, branch.score or 0.0))
            conviction = _clamp(conviction, -1.0, 1.0)
            symbol_convictions[symbol] = conviction * branch_reliability * symbol_reliability
            symbol_confidences[symbol] = _clamp(branch.confidence * branch_reliability * symbol_reliability, 0.0, 1.0)

            if isinstance(expected_from_signal, dict) and symbol in expected_from_signal:
                expected_return = float(expected_from_signal[symbol])
            else:
                expected_return = conviction * scale
            expected_return *= branch_reliability * symbol_reliability
            symbol_expected_returns[symbol] = _clamp(expected_return, -0.35, 0.35)

        aggregate_expected_return = (
            sum(symbol_expected_returns.values()) / len(symbol_expected_returns)
            if symbol_expected_returns else 0.0
        )
        return CalibratedBranchSignal(
            branch_name=branch_name,
            branch_mode=mode,
            horizon_days=horizon,
            reliability=branch_reliability,
            aggregate_expected_return=aggregate_expected_return,
            symbol_expected_returns=symbol_expected_returns,
            symbol_confidences=symbol_confidences,
            symbol_convictions=symbol_convictions,
            data_source_status=str(branch.metadata.get("data_source_status", "unknown")),
            metadata={
                "branch_mode": mode,
                "reliability": branch_reliability,
                "data_source_status": branch.metadata.get("data_source_status", "unknown"),
                "horizon_days": horizon,
            },
        )

    def _branch_reliability(self, branch: BranchResult) -> float:
        base = float(branch.metadata.get("reliability", 1.0 if branch.success else 0.2))
        if not branch.success:
            base = min(base, 0.25)
        if branch.metadata.get("branch_mode") in {"heuristic_adapter", "kline_heuristic", "structured_research_debate"}:
            base *= 0.85
        return _clamp(base, 0.0, 1.0)

    def _symbol_reliability(self, symbol: str) -> float:
        meta = self.symbol_provenance.get(symbol, {})
        reliability = float(meta.get("reliability", 1.0))
        if meta.get("is_synthetic"):
            reliability = min(reliability, 0.35)
        elif meta.get("data_source_status") == "degraded_real":
            reliability = min(reliability, 0.7)
        return _clamp(reliability, 0.0, 1.0)
