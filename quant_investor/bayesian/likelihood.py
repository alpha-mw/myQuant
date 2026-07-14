"""Signal likelihood mapper — converts branch evidence into calibrated likelihoods.

Each branch's (final_score, final_confidence, reliability) is mapped to a
likelihood value in [0, 1] representing the probability of observing this
signal given the stock will outperform.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from quant_investor.bayesian.calibration import CalibrationStore
from quant_investor.bayesian.types import LikelihoodSet
from quant_investor.branch_contracts import BranchResult
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER

# Default branch reliabilities used when metadata is missing.
_DEFAULT_RELIABILITY: dict[str, float] = {
    "quant": 0.70,
    "fundamental": 0.60,
}

# Preserve the surviving v13 effective coefficients exactly.  These values are
# already normalized against the former three-source mean; the posterior must
# not re-normalize them after the v14 source-count reduction.
_MOMENTUM_EFFECTIVE_WEIGHTS: dict[str, dict[str, float]] = {
    "strong": {"quant": 1.0161290322580645, "fundamental": 0.7258064516129031},
    "weak": {"quant": 1.0714285714285716, "fundamental": 0.8035714285714286},
    "neutral": {"quant": 1.0327868852459017, "fundamental": 0.8360655737704918},
}
_MOMENTUM_BREAKOUT_EFFECTIVE_WEIGHTS: dict[str, dict[str, float]] = {
    "strong": {"quant": 1.0606060606060608, "fundamental": 0.7575757575757577},
    "weak": {"quant": 1.1131725417439704, "fundamental": 0.8348794063079777},
    "neutral": {"quant": 1.0732538330494037, "fundamental": 0.8688245315161839},
}


def _finite_float(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}; got {value!r}.")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}; got {value!r}.")
    return resolved


def _score_to_likelihood(
    score: float,
    confidence: float,
    reliability: float,
    *,
    calibrated_probability: float | None = None,
) -> float:
    """Map (score, confidence, reliability) to a likelihood in [0.05, 0.95].

    The mapping uses a soft sigmoid-like transform:
    - score in [-1, 1] drives the center
    - confidence and reliability moderate the extremity
    """
    score = _finite_float(score, "score")
    confidence = _finite_float(confidence, "confidence", minimum=0.0, maximum=1.0)
    reliability = _finite_float(reliability, "reliability", minimum=0.0, maximum=1.0)
    if calibrated_probability is not None:
        calibrated_probability = _finite_float(
            calibrated_probability,
            "calibrated_probability",
            minimum=0.0,
            maximum=1.0,
        )

    # Normalize score to [0, 1] range
    raw = (score + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    raw = max(0.0, min(1.0, raw))

    # Blend with 0.5 (neutral) based on confidence and reliability
    blend_strength = confidence * reliability
    blend_strength = max(0.0, min(1.0, blend_strength))

    likelihood = 0.50 + (raw - 0.50) * blend_strength
    if calibrated_probability is not None:
        calibration_blend = 0.35 + 0.35 * blend_strength
        likelihood = likelihood * (1.0 - calibration_blend) + calibrated_probability * calibration_blend
    return max(0.05, min(0.95, likelihood))


class SignalLikelihoodMapper:
    """Map branch results to calibrated likelihood values."""

    def __init__(
        self,
        *,
        calibration_store: CalibrationStore | None = None,
        recall_context: Mapping[str, Any] | None = None,
        global_context: Any | None = None,
    ) -> None:
        self.calibration_store = calibration_store or CalibrationStore()
        self.recall_context = dict(recall_context or {})
        self.global_context = global_context

    def _selection_profile(self) -> str:
        if self.global_context is None:
            return "classic"
        metadata = dict(getattr(self.global_context, "metadata", {}) or {})
        selection_profile = dict(metadata.get("selection_profile", {}) or {})
        return str(selection_profile.get("funnel_profile", "classic") or "classic").strip().lower() or "classic"

    def _symbol_state(self, symbol: str) -> dict[str, Any]:
        if self.global_context is None:
            return {}
        metadata = dict(getattr(self.global_context, "metadata", {}) or {})
        states = metadata.get("symbol_market_state", {})
        if isinstance(states, Mapping):
            return dict(states.get(symbol, {}) or {})
        return {}

    def _recall_bias(self, symbol: str) -> float:
        if self._selection_profile() != "momentum_leader":
            return 0.0
        top_picks = self.recall_context.get("top_picks", [])
        net = 0.0
        if isinstance(top_picks, list):
            for pick in top_picks:
                if not isinstance(pick, Mapping):
                    continue
                if str(pick.get("symbol", "")).strip() != symbol:
                    continue
                action = str(pick.get("action", "")).strip().lower()
                if action in {"buy", "hold"}:
                    net += 1.0
                elif action in {"sell", "avoid", "watch"}:
                    net -= 1.0
        recent_symbols = {
            str(item).strip()
            for item in self.recall_context.get("recent_symbols", [])
            if str(item).strip()
        }
        pending_counts = self.recall_context.get("pending_trade_counts", {})
        if symbol in recent_symbols and isinstance(pending_counts, Mapping):
            pending_buy = int(pending_counts.get("buy", 0) or 0)
            pending_sell = int(pending_counts.get("sell", 0) or 0)
            if pending_buy >= max(pending_sell + 3, 5):
                net -= 0.5
        return max(-0.08, min(0.08, net * 0.04))

    def _crowding_penalty(self, symbol: str) -> float:
        if self.global_context is None or self._selection_profile() != "momentum_leader":
            return 0.0
        metadata = dict(getattr(self.global_context, "metadata", {}) or {})
        sector_counts = metadata.get("candidate_sector_counts", {})
        if not isinstance(sector_counts, Mapping):
            return 0.0
        sector_bucket_limit = int(getattr(self.global_context, "risk_budget", {}).get("sector_bucket_limit", 0) or 0)
        if sector_bucket_limit <= 0:
            return 0.0
        sector = str(getattr(self.global_context, "industry_map", {}).get(symbol) or self._symbol_state(symbol).get("industry") or "").strip()
        if not sector:
            return 0.0
        count = int(sector_counts.get(sector, 0) or 0)
        if count <= sector_bucket_limit:
            return 0.0
        return max(0.0, min(1.0, (count - sector_bucket_limit) / max(sector_bucket_limit, 1)))

    def _branch_weight_map(self, symbol: str) -> dict[str, float]:
        if self.global_context is None or self._selection_profile() != "momentum_leader":
            return {}
        regime = str(getattr(self.global_context, "macro_regime", "") or "")
        breadth = _finite_float(
            getattr(self.global_context, "cross_section_quant", {}).get("breadth", 0.0) or 0.0,
            "cross_section_quant.breadth",
        )
        state = self._symbol_state(symbol)
        breakout_risk = _finite_float(
            state.get("fake_breakout_risk", 0.0) or 0.0,
            "fake_breakout_risk",
            minimum=0.0,
            maximum=1.0,
        )
        strong_regime = regime == "趋势上涨" or breadth >= 0.55
        weak_regime = regime in {"趋势下跌", "震荡高波"} or breadth <= 0.48
        if strong_regime:
            bucket = "strong"
        elif weak_regime:
            bucket = "weak"
        else:
            bucket = "neutral"
        source = (
            _MOMENTUM_BREAKOUT_EFFECTIVE_WEIGHTS
            if breakout_risk >= 0.65
            else _MOMENTUM_EFFECTIVE_WEIGHTS
        )
        return dict(source[bucket])

    def _branch_likelihood(
        self,
        branch_name: str,
        branch_results: dict[str, BranchResult],
        symbol: str,
    ) -> tuple[float, dict[str, float]]:
        result = branch_results.get(branch_name)
        if result is None:
            return 0.50, {
                "reliability": float(_DEFAULT_RELIABILITY.get(branch_name, 0.50)),
                "sample_size": 0.0,
                "calibration_probability": 0.50,
            }

        score = _finite_float(
            result.symbol_scores.get(symbol, result.final_score),
            f"{branch_name}.score",
        )
        confidence = _finite_float(
            result.final_confidence,
            f"{branch_name}.confidence",
            minimum=0.0,
            maximum=1.0,
        )
        reliability = _finite_float(
            result.metadata.get("reliability", _DEFAULT_RELIABILITY.get(branch_name, 0.50)),
            f"{branch_name}.reliability",
            minimum=0.0,
            maximum=1.0,
        )
        profile = self._selection_profile()
        calibration = (
            self.calibration_store.calibration_stats(branch_name, score)
            if profile == "momentum_leader"
            else {"probability": 0.50, "sample_size": 0.0, "recent_failure_rate": 0.0}
        )
        likelihood = _score_to_likelihood(
            score,
            confidence,
            reliability,
            calibrated_probability=(
                float(calibration.get("probability", 0.50))
                if profile == "momentum_leader"
                else None
            ),
        )

        if profile == "momentum_leader":
            recall_bias = self._recall_bias(symbol)
            if branch_name == "fundamental" and score < 0.0:
                likelihood -= 0.04
            likelihood += recall_bias
        likelihood = max(0.05, min(0.95, likelihood))
        return likelihood, {
            "reliability": reliability,
            "sample_size": float(calibration.get("sample_size", 0.0) or 0.0),
            "calibration_probability": float(calibration.get("probability", 0.50) or 0.50),
            "recent_failure_rate": float(calibration.get("recent_failure_rate", 0.0) or 0.0),
        }

    def compute_likelihoods(
        self,
        *,
        branch_results: dict[str, BranchResult],
        symbol: str,
        candidate_symbols: set[str] | None = None,
    ) -> LikelihoodSet:
        """Compute likelihoods for a single symbol from all available branches.

        Fundamental only runs on candidates, while Quant covers the funnel.
        non-candidate symbols get a neutral likelihood of 0.50.
        """
        unexpected_branches = sorted(
            set(branch_results) - set(CANONICAL_BRANCH_ORDER)
        )
        if unexpected_branches:
            raise ValueError(
                "Unexpected branch results for v14 likelihood mapping: "
                + ", ".join(unexpected_branches)
            )
        candidate_only_branches = {"fundamental"}
        is_candidate = candidate_symbols is None or symbol in (candidate_symbols or set())
        profile = self._selection_profile()
        branch_meta: dict[str, dict[str, float]] = {}

        quant_l, branch_meta["quant"] = self._branch_likelihood("quant", branch_results, symbol)

        if is_candidate:
            fundamental_l, branch_meta["fundamental"] = self._branch_likelihood("fundamental", branch_results, symbol)
        else:
            fundamental_l = 0.50  # neutral for non-candidates
            branch_meta["fundamental"] = {
                "reliability": float(_DEFAULT_RELIABILITY["fundamental"]),
                "sample_size": 0.0,
                "calibration_probability": 0.50,
                "recent_failure_rate": 0.0,
            }

        evidence_sources = []
        for name in ("quant", "fundamental"):
            if name in candidate_only_branches and not is_candidate:
                continue
            if name in branch_results:
                evidence_sources.append(name)

        history_confidence = 0.0
        if profile == "momentum_leader" and evidence_sources:
            sample_confidences = [
                min(float(branch_meta[name].get("sample_size", 0.0)) / 16.0, 1.0)
                for name in evidence_sources
            ]
            history_confidence = sum(sample_confidences) / max(len(sample_confidences), 1)
            if history_confidence <= 0.0:
                history_confidence = 0.35
        avg_reliability = (
            sum(float(branch_meta[name].get("reliability", 0.0)) for name in evidence_sources) / max(len(evidence_sources), 1)
            if evidence_sources
            else 0.0
        )
        state = self._symbol_state(symbol)
        fake_breakout_penalty = (
            _finite_float(
                state.get("fake_breakout_risk", 0.0) or 0.0,
                "fake_breakout_risk",
                minimum=0.0,
                maximum=1.0,
            )
            if profile == "momentum_leader"
            else 0.0
        )
        market_pressure = 0.0
        if self.global_context is not None and profile == "momentum_leader":
            regime = str(getattr(self.global_context, "macro_regime", "") or "")
            breadth = _finite_float(
                getattr(self.global_context, "cross_section_quant", {}).get("breadth", 0.0) or 0.0,
                "cross_section_quant.breadth",
            )
            if regime in {"趋势下跌", "震荡高波"} or breadth < 0.48:
                market_pressure = 1.0
            elif regime == "趋势上涨" or breadth > 0.55:
                market_pressure = -0.5

        return LikelihoodSet(
            quant_likelihood=quant_l,
            fundamental_likelihood=fundamental_l,
            correlation_matrix={},
            metadata={
                "evidence_sources": evidence_sources,
                "profile": profile,
                "branch_weights": self._branch_weight_map(symbol),
                "history_confidence": history_confidence,
                "avg_reliability": avg_reliability,
                "recall_bias": self._recall_bias(symbol) if profile == "momentum_leader" else 0.0,
                "momentum_strength": _finite_float(
                    state.get("momentum_strength", 0.0) or 0.0,
                    "momentum_strength",
                ),
                "fake_breakout_penalty": fake_breakout_penalty,
                "crowding_penalty": self._crowding_penalty(symbol) if profile == "momentum_leader" else 0.0,
                "calibration_samples": {
                    name: int(float(meta.get("sample_size", 0.0)))
                    for name, meta in branch_meta.items()
                },
                "market_pressure": market_pressure,
                "sector": str(
                    (
                        getattr(self.global_context, "industry_map", {}).get(symbol)
                        if self.global_context is not None
                        else ""
                    )
                    or state.get("industry")
                    or state.get("sector")
                    or ""
                ),
            },
        )
