"""Deterministic funnel — compress full market to candidate set.

Consumes the quant BranchResult and a GlobalContext, then applies gates and
ranking to produce a compressed candidate set of ~500 symbols by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.candidate_filter import (
    DataQualityGate,
    LiquidityGate,
    TradabilityGate,
)
from quant_investor.logger import get_logger

_logger = get_logger("DeterministicFunnel")


_THEME_PHASE_ADJUSTMENTS: dict[str, float] = {
    "accumulation": 0.01,
    "early_acceleration": 0.03,
    "confirmed_rotation": 0.04,
    "overextended": -0.05,
    "distribution": -0.07,
}
_THEME_RISK_FLAG_PENALTIES: dict[str, float] = {
    "theme_overextended": -0.03,
    "theme_overextended_no_chase": -0.03,
    "theme_fake_breakout_risk": -0.03,
    "theme_low_breadth": -0.02,
    "theme_distribution_risk": -0.04,
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dedupe_flags(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        return []
    try:
        items = list(value or [])
    except TypeError:
        return []
    flags: list[str] = []
    seen: set[str] = set()
    for item in items:
        flag = str(item or "").strip()
        if not flag or flag in seen:
            continue
        seen.add(flag)
        flags.append(flag)
    return flags


@dataclass
class FunnelConfig:
    """Tuning knobs for the deterministic funnel."""

    max_candidates: int = 500
    liquidity_percentile_min: float = 0.10
    min_composite_score: float = -1.0  # disabled by default
    profile: str = "classic"
    trend_windows: tuple[int, ...] = (20, 60, 120)
    volume_spike_threshold: float = 1.35
    breakout_distance_pct: float = 0.06
    sector_bucket_limit: int = 0
    theme_boost_enabled: bool = False
    theme_boost_cap: float = 0.10


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
    4. Quant score ranking with deterministic tradability context
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
    def _classic_score(symbol: str, quant_scores: dict[str, float]) -> float:
        return float(quant_scores.get(symbol, 0.0))

    def _theme_boost_for_symbol(
        self,
        *,
        symbol: str,
        global_context: GlobalContext,
    ) -> tuple[float, dict[str, Any]]:
        if not bool(self.config.theme_boost_enabled):
            return 0.0, {"enabled": False, "reason": "disabled"}

        cap = max(
            0.0,
            _safe_float(getattr(self.config, "theme_boost_cap", 0.0), 0.0),
        )
        boost_metadata: dict[str, Any] = {
            "enabled": True,
            "available": False,
            "symbol_score": 0.0,
            "theme_strength": 0.0,
            "primary_theme_id": "",
            "phase": "",
            "risk_flags": [],
            "raw_boost": 0.0,
            "phase_adjustment": 0.0,
            "risk_penalty": 0.0,
            "final_boost": 0.0,
            "cap": cap,
            "reason": "",
        }
        metadata = getattr(global_context, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            boost_metadata["reason"] = "theme_metadata_missing"
            return 0.0, boost_metadata

        payload: Mapping[str, Any]
        rotation_payload = metadata.get("theme_rotation")
        if rotation_payload is not None:
            if not isinstance(rotation_payload, Mapping):
                boost_metadata["reason"] = "theme_rotation_malformed"
                return 0.0, boost_metadata
            status = str(rotation_payload.get("status") or "").strip().lower()
            if status != "success":
                boost_metadata["reason"] = "theme_rotation_not_success"
                return 0.0, boost_metadata
            payload = rotation_payload
        else:
            if not any(
                key in metadata
                for key in (
                    "symbol_theme_score",
                    "symbol_primary_theme",
                    "symbol_theme_phase",
                    "theme_scores",
                )
            ):
                boost_metadata["reason"] = "theme_rotation_missing"
                return 0.0, boost_metadata
            payload = {
                "symbol_scores": metadata.get("symbol_theme_score"),
                "symbol_primary_theme": metadata.get("symbol_primary_theme"),
                "symbol_phase": metadata.get("symbol_theme_phase"),
                "symbol_risk_flags": metadata.get("symbol_risk_flags")
                or metadata.get("symbol_theme_risk_flags"),
                "theme_scores": metadata.get("theme_scores"),
            }

        symbol_scores = _mapping_or_empty(payload.get("symbol_scores"))
        if symbol not in symbol_scores:
            boost_metadata["reason"] = "symbol_theme_missing"
            return 0.0, boost_metadata

        symbol_score = _clamp(_safe_float(symbol_scores.get(symbol, 0.0)), 0.0, 1.0)
        primary_theme_id = str(
            _mapping_or_empty(payload.get("symbol_primary_theme")).get(symbol, "") or ""
        )
        phase = (
            str(_mapping_or_empty(payload.get("symbol_phase")).get(symbol, "") or "")
            .strip()
            .lower()
        )
        if not phase and primary_theme_id:
            theme_score = _mapping_or_empty(payload.get("theme_scores")).get(primary_theme_id)
            if isinstance(theme_score, Mapping):
                phase = str(theme_score.get("phase") or "").strip().lower()
        risk_flags = _dedupe_flags(
            _mapping_or_empty(payload.get("symbol_risk_flags")).get(symbol, [])
        )

        theme_strength = _clamp((symbol_score - 0.50) / 0.50, 0.0, 1.0)
        raw_boost = 0.06 * theme_strength
        phase_adjustment = float(_THEME_PHASE_ADJUSTMENTS.get(phase, 0.0))
        risk_penalty = sum(
            _THEME_RISK_FLAG_PENALTIES.get(flag, 0.0)
            for flag in risk_flags
        )
        final_boost = _clamp(
            raw_boost + phase_adjustment + risk_penalty,
            -0.06,
            cap,
        )
        boost_metadata.update(
            {
                "available": True,
                "symbol_score": symbol_score,
                "theme_strength": theme_strength,
                "primary_theme_id": primary_theme_id,
                "phase": phase,
                "risk_flags": risk_flags,
                "raw_boost": raw_boost,
                "phase_adjustment": phase_adjustment,
                "risk_penalty": risk_penalty,
                "final_boost": final_boost,
                "reason": "applied" if final_boost != 0.0 else "no_theme_boost",
            }
        )
        return final_boost, boost_metadata

    def _momentum_leader_score(
        self,
        *,
        symbol: str,
        quant_scores: dict[str, float],
        global_context: GlobalContext,
    ) -> float:
        qs = float(quant_scores.get(symbol, 0.0))
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
        recent_return_component = _clamp((recent_return + 0.12) / 0.30, 0.0, 1.0)
        distance_penalty = _clamp(
            distance_from_high / max(float(self.config.breakout_distance_pct) * 1.5, 0.01),
            0.0,
            1.0,
        )
        drawdown_penalty = _clamp(max_drawdown / 0.18, 0.0, 1.0)

        score = (
            0.34 * momentum_strength
            + 0.24 * quant_component
            + 0.13 * breakout_readiness
            + 0.10 * volume_confirmation
            + 0.07 * trend_stability
            + 0.08 * recent_return_component
        )
        score -= 0.20 * distance_penalty
        score -= 0.18 * drawdown_penalty
        score -= 0.22 * fake_breakout_risk
        if breakout_readiness >= 0.75 and volume_confirmation <= 0.15:
            score -= 0.10 * fake_breakout_risk
        if breakout_readiness >= 0.75 and volume_confirmation >= 0.50:
            score += 0.05
        theme_boost, _theme_meta = self._theme_boost_for_symbol(
            symbol=symbol,
            global_context=global_context,
        )
        score += theme_boost
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

        # Score: quant branch plus deterministic tradability/momentum context.
        quant_scores = quant_result.symbol_scores or {}
        profile = str(self.config.profile or "classic").strip().lower() or "classic"

        composite: dict[str, float] = {}
        for symbol in symbols:
            if profile == "momentum_leader":
                composite[symbol] = self._momentum_leader_score(
                    symbol=symbol,
                    quant_scores=quant_scores,
                    global_context=global_context,
                )
            else:
                composite[symbol] = self._classic_score(symbol, quant_scores)

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
        theme_boost_available_count = 0
        theme_boost_applied_count = 0
        if profile == "momentum_leader" and bool(self.config.theme_boost_enabled):
            for symbol in candidates:
                theme_boost, theme_meta = self._theme_boost_for_symbol(
                    symbol=symbol,
                    global_context=global_context,
                )
                if bool(theme_meta.get("available")) or theme_boost != 0.0:
                    theme_boost_available_count += 1
                if theme_boost != 0.0:
                    theme_boost_applied_count += 1

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
                "factor_mode": "quant_only",
                "profile": profile,
                "trend_windows": list(self.config.trend_windows),
                "volume_spike_threshold": float(self.config.volume_spike_threshold),
                "breakout_distance_pct": float(self.config.breakout_distance_pct),
                "sector_bucket_limit": int(self.config.sector_bucket_limit),
                "excluded_count": len(all_excluded),
                "theme_boost_enabled": bool(self.config.theme_boost_enabled),
                "theme_boost_cap": max(
                    0.0,
                    _safe_float(getattr(self.config, "theme_boost_cap", 0.0), 0.0),
                ),
                "theme_boost_profile": "momentum_leader_only",
                "theme_boost_note": "disabled_by_default_capped_deterministic",
                "theme_boost_available_count": theme_boost_available_count,
                "theme_boost_applied_count": theme_boost_applied_count,
            },
        )
