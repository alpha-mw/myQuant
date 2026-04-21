"""Deterministic funnel — compress full market to candidate set.

Consumes quant + kline BranchResults (full-market) and a GlobalContext,
then applies a pipeline of gates and ranking to produce a compressed
candidate set of ~200 symbols by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.candidate_filter import (
    DataQualityGate,
    LiquidityGate,
    TradabilityGate,
)
from quant_investor.logger import get_logger

_logger = get_logger("DeterministicFunnel")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclass
class FunnelConfig:
    """Tuning knobs for the deterministic funnel."""

    max_candidates: int = 200
    liquidity_percentile_min: float = 0.10
    min_composite_score: float = -1.0  # disabled by default
    quant_weight: float = 0.55
    kline_weight: float = 0.45
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
    """Full-market first-pass engine.

    Pipeline:
    1. Data quality gate
    2. Tradability gate
    3. Liquidity gate
    4. Composite score ranking (quant + kline)
    5. Top-N cutoff
    """

    def __init__(self, config: FunnelConfig | None = None) -> None:
        self.config = config or FunnelConfig()

    @staticmethod
    def _symbol_state(global_context: GlobalContext, symbol: str) -> dict[str, Any]:
        metadata = dict(global_context.metadata or {})
        symbol_market_state = metadata.get("symbol_market_state", {})
        if isinstance(symbol_market_state, dict):
            return dict(symbol_market_state.get(symbol, {}) or {})
        return {}

    @staticmethod
    def _sector_name(global_context: GlobalContext, symbol: str) -> str:
        industry_map = dict(global_context.industry_map or {})
        if symbol in industry_map and str(industry_map[symbol]).strip():
            return str(industry_map[symbol]).strip()
        state = DeterministicFunnel._symbol_state(global_context, symbol)
        return str(state.get("industry") or state.get("sector") or "").strip()

    @staticmethod
    def _classic_score(symbol: str, quant_scores: dict[str, float], kline_scores: dict[str, float], *, quant_weight: float, kline_weight: float) -> float:
        qs = float(quant_scores.get(symbol, 0.0))
        ks = float(kline_scores.get(symbol, 0.0))
        return quant_weight * qs + kline_weight * ks

    def _momentum_leader_score(
        self,
        *,
        symbol: str,
        quant_scores: dict[str, float],
        kline_scores: dict[str, float],
        global_context: GlobalContext,
    ) -> float:
        qs = float(quant_scores.get(symbol, 0.0))
        ks = float(kline_scores.get(symbol, 0.0))
        state = self._symbol_state(global_context, symbol)
        momentum_strength = float(state.get("momentum_strength", 0.0))
        breakout_readiness = float(state.get("breakout_readiness", 0.0))
        volume_confirmation = float(state.get("volume_confirmation", 0.0))
        trend_stability = float(state.get("trend_stability", 0.0))
        distance_from_high = float(state.get("distance_from_high_pct", 1.0))
        fake_breakout_risk = float(state.get("fake_breakout_risk", 0.0))
        max_drawdown = float(state.get("max_drawdown_pct", 0.0))
        recent_return = float(state.get("return_20d", 0.0))
        quant_component = _clamp((qs + 1.0) / 2.0, 0.0, 1.0)
        kline_component = _clamp((ks + 1.0) / 2.0, 0.0, 1.0)
        recent_return_component = _clamp((recent_return + 0.12) / 0.30, 0.0, 1.0)
        distance_penalty = _clamp(
            distance_from_high / max(float(self.config.breakout_distance_pct) * 1.5, 0.01),
            0.0,
            1.0,
        )
        drawdown_penalty = _clamp(max_drawdown / 0.18, 0.0, 1.0)

        score = (
            0.30 * momentum_strength
            + 0.20 * kline_component
            + 0.15 * quant_component
            + 0.13 * breakout_readiness
            + 0.10 * volume_confirmation
            + 0.07 * trend_stability
            + 0.05 * recent_return_component
        )
        score -= 0.20 * distance_penalty
        score -= 0.18 * drawdown_penalty
        score -= 0.22 * fake_breakout_risk
        if breakout_readiness >= 0.75 and volume_confirmation <= 0.15:
            score -= 0.10 * fake_breakout_risk
        if breakout_readiness >= 0.75 and volume_confirmation >= 0.50:
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
                current = counts.get(sector, 0)
                if current >= limit:
                    excluded.setdefault(symbol, "sector_bucket_limit")
                    continue
                counts[sector] = current + 1
            selected.append((symbol, score))
            if len(selected) >= self.config.max_candidates:
                break
        return selected

    def run(
        self,
        *,
        quant_result: BranchResult,
        kline_result: BranchResult,
        global_context: GlobalContext,
    ) -> FunnelOutput:
        all_symbols = list(global_context.universe_tiers.get("researchable", global_context.universe_symbols))
        all_excluded: dict[str, str] = {}

        # Gate 1: data quality
        symbols, excluded = DataQualityGate().filter(all_symbols, global_context)
        all_excluded.update(excluded)

        # Gate 2: tradability
        symbols, excluded = TradabilityGate().filter(symbols, global_context)
        all_excluded.update(excluded)

        # Gate 3: liquidity
        symbols, excluded = LiquidityGate(
            percentile_min=self.config.liquidity_percentile_min,
        ).filter(symbols, global_context)
        all_excluded.update(excluded)

        # Score: weighted composite of quant + kline symbol_scores
        quant_scores = quant_result.symbol_scores or {}
        kline_scores = kline_result.symbol_scores or {}
        qw = self.config.quant_weight
        kw = self.config.kline_weight
        profile = str(self.config.profile or "classic").strip().lower() or "classic"

        composite: dict[str, float] = {}
        for symbol in symbols:
            if profile == "momentum_leader":
                composite[symbol] = self._momentum_leader_score(
                    symbol=symbol,
                    quant_scores=quant_scores,
                    kline_scores=kline_scores,
                    global_context=global_context,
                )
            else:
                composite[symbol] = self._classic_score(
                    symbol,
                    quant_scores,
                    kline_scores,
                    quant_weight=qw,
                    kline_weight=kw,
                )

        # Filter by minimum composite score
        if self.config.min_composite_score > -1.0:
            for symbol in list(composite):
                if composite[symbol] < self.config.min_composite_score:
                    all_excluded[symbol] = f"below_min_score_{self.config.min_composite_score}"
                    del composite[symbol]

        # Rank and cutoff
        ranked = sorted(composite.items(), key=lambda item: (-item[1], item[0]))
        if profile == "momentum_leader":
            top_n = self._apply_sector_bucket_limit(
                ranked=ranked,
                global_context=global_context,
                excluded=all_excluded,
            )
        else:
            top_n = ranked[: self.config.max_candidates]
        candidates = [symbol for symbol, _ in top_n]
        candidate_scores = {symbol: score for symbol, score in top_n}
        for symbol, _score in ranked:
            if symbol in candidate_scores or symbol in all_excluded:
                continue
            all_excluded[symbol] = "rank_cutoff"

        _logger.info(
            "Funnel[%s]: %d total -> %d after gates -> %d candidates (max %d)",
            profile,
            len(all_symbols),
            len(composite),
            len(candidates),
            self.config.max_candidates,
        )

        return FunnelOutput(
            candidates=candidates,
            candidate_scores=candidate_scores,
            excluded_symbols=all_excluded,
            funnel_metadata={
                "total_universe": len(all_symbols),
                "after_gates": len(composite),
                "final_candidates": len(candidates),
                "max_candidates": self.config.max_candidates,
                "quant_weight": qw,
                "kline_weight": kw,
                "profile": profile,
                "trend_windows": list(self.config.trend_windows),
                "volume_spike_threshold": float(self.config.volume_spike_threshold),
                "breakout_distance_pct": float(self.config.breakout_distance_pct),
                "sector_bucket_limit": int(self.config.sector_bucket_limit),
                "excluded_count": len(all_excluded),
            },
        )
