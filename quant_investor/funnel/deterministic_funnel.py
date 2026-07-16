"""Deterministic full-market compression for the production research DAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.candidate_filter import DataQualityGate, LiquidityGate, TradabilityGate
from quant_investor.logger import get_logger

_logger = get_logger("DeterministicFunnel")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class FunnelConfig:
    max_candidates: int = 500
    liquidity_percentile_min: float = 0.10
    min_composite_score: float = -1.0
    profile: str = "classic"
    trend_windows: tuple[int, ...] = (20, 60, 120)
    volume_spike_threshold: float = 1.35
    breakout_distance_pct: float = 0.06
    sector_bucket_limit: int = 0


@dataclass
class FunnelOutput:
    """Result of the deterministic funnel pass."""

    candidates: list[str] = field(default_factory=list)
    candidate_scores: dict[str, float] = field(default_factory=dict)
    excluded_symbols: dict[str, str] = field(default_factory=dict)
    funnel_metadata: dict[str, Any] = field(default_factory=dict)


class DeterministicFunnel:
    """Apply hard data/tradability/liquidity gates and deterministic ranking."""

    def __init__(self, config: FunnelConfig | None = None) -> None:
        self.config = config or FunnelConfig()

    @staticmethod
    def _symbol_state(global_context: GlobalContext, symbol: str) -> dict[str, Any]:
        metadata = dict(global_context.metadata or {})
        states = metadata.get("symbol_market_state", {})
        if isinstance(states, dict):
            return dict(states.get(symbol, {}) or {})
        return {}

    @staticmethod
    def _sector_name(global_context: GlobalContext, symbol: str) -> str:
        industry_map = dict(global_context.industry_map or {})
        if symbol in industry_map and str(industry_map[symbol]).strip():
            return str(industry_map[symbol]).strip()
        state = DeterministicFunnel._symbol_state(global_context, symbol)
        return str(state.get("industry") or state.get("sector") or "").strip()

    def _momentum_leader_score(
        self,
        *,
        symbol: str,
        quant_scores: dict[str, float],
        global_context: GlobalContext,
    ) -> float:
        qs = float(quant_scores.get(symbol, 0.0))
        state = self._symbol_state(global_context, symbol)
        breakout = float(state.get("breakout_readiness", 0.0))
        volume = float(state.get("volume_confirmation", 0.0))
        fake_breakout = float(state.get("fake_breakout_risk", 0.0))
        score = (
            0.34 * float(state.get("momentum_strength", 0.0))
            + 0.24 * _clamp((qs + 1.0) / 2.0)
            + 0.13 * breakout
            + 0.10 * volume
            + 0.07 * float(state.get("trend_stability", 0.0))
            + 0.08 * _clamp((float(state.get("return_20d", 0.0)) + 0.12) / 0.30)
        )
        score -= 0.20 * _clamp(
            float(state.get("distance_from_high_pct", 1.0))
            / max(float(self.config.breakout_distance_pct) * 1.5, 0.01)
        )
        score -= 0.18 * _clamp(float(state.get("max_drawdown_pct", 0.0)) / 0.18)
        score -= 0.22 * fake_breakout
        if breakout >= 0.75 and volume <= 0.15:
            score -= 0.10 * fake_breakout
        if breakout >= 0.75 and volume >= 0.50:
            score += 0.05
        return round(score, 6)

    def _apply_sector_bucket_limit(
        self,
        *,
        ranked: list[tuple[str, float]],
        global_context: GlobalContext,
        excluded: dict[str, str],
    ) -> list[tuple[str, float]]:
        limit = max(int(self.config.sector_bucket_limit or 0), 0)
        if limit <= 0 or not global_context.industry_map:
            return ranked[: self.config.max_candidates]
        counts: dict[str, int] = {}
        selected: list[tuple[str, float]] = []
        for symbol, score in ranked:
            sector = self._sector_name(global_context, symbol)
            if sector and sector != "unknown":
                if counts.get(sector, 0) >= limit:
                    excluded.setdefault(symbol, "sector_bucket_limit")
                    continue
                counts[sector] = counts.get(sector, 0) + 1
            selected.append((symbol, score))
            if len(selected) >= self.config.max_candidates:
                break
        return selected

    def run(self, *, quant_result: BranchResult, global_context: GlobalContext) -> FunnelOutput:
        all_symbols = list(
            global_context.universe_tiers.get("researchable", global_context.universe_symbols)
        )
        all_excluded: dict[str, str] = {}
        symbols, excluded = DataQualityGate().filter(all_symbols, global_context)
        all_excluded.update(excluded)
        symbols, excluded = TradabilityGate().filter(symbols, global_context)
        all_excluded.update(excluded)
        symbols, excluded = LiquidityGate(
            percentile_min=self.config.liquidity_percentile_min
        ).filter(symbols, global_context)
        all_excluded.update(excluded)
        after_hard_gates = len(symbols)

        quant_scores = quant_result.symbol_scores or {}
        profile = str(self.config.profile or "classic").strip().lower() or "classic"
        composite = {
            symbol: (
                self._momentum_leader_score(
                    symbol=symbol, quant_scores=quant_scores, global_context=global_context
                )
                if profile == "momentum_leader"
                else float(quant_scores.get(symbol, 0.0))
            )
            for symbol in symbols
        }
        if self.config.min_composite_score > -1.0:
            for symbol in list(composite):
                if composite[symbol] < self.config.min_composite_score:
                    all_excluded[symbol] = f"below_min_score_{self.config.min_composite_score}"
                    del composite[symbol]

        ranked = sorted(composite.items(), key=lambda item: (-item[1], item[0]))
        top_n = (
            self._apply_sector_bucket_limit(
                ranked=ranked, global_context=global_context, excluded=all_excluded
            )
            if profile == "momentum_leader"
            else ranked[: self.config.max_candidates]
        )
        candidate_scores = dict(top_n)
        for symbol, _score in ranked:
            if symbol not in candidate_scores and symbol not in all_excluded:
                all_excluded[symbol] = "rank_cutoff"
        _logger.info(
            "Funnel[%s]: %d total -> %d after gates -> %d candidates (max %d)",
            profile,
            len(all_symbols),
            len(composite),
            len(top_n),
            self.config.max_candidates,
        )
        return FunnelOutput(
            candidates=list(candidate_scores),
            candidate_scores=candidate_scores,
            excluded_symbols=all_excluded,
            funnel_metadata={
                "total_universe": len(all_symbols),
                "after_hard_gates": after_hard_gates,
                "after_gates": len(composite),
                "final_candidates": len(top_n),
                "max_candidates": self.config.max_candidates,
                "factor_mode": "quant_only",
                "profile": profile,
                "trend_windows": list(self.config.trend_windows),
                "volume_spike_threshold": float(self.config.volume_spike_threshold),
                "breakout_distance_pct": float(self.config.breakout_distance_pct),
                "sector_bucket_limit": int(self.config.sector_bucket_limit),
                "excluded_count": len(all_excluded),
            },
        )
