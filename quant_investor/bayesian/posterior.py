"""Bayesian posterior engine — the core decision logic.

Computes posterior probability of outperformance using log-odds Bayesian
update with correlation-aware discounting.
"""

from __future__ import annotations

import math
from typing import Any

from quant_investor.bayesian.types import LikelihoodSet, PosteriorResult, PriorSet

# Correlation discount: when two branches are correlated, reduce
# the joint information weight.
_CORRELATION_PAIRS: list[tuple[str, str, float]] = [
    ("kline", "quant", 0.50),
    ("fundamental", "intelligence", 0.35),
]

# Branch weight in log-odds space.  Equal by default; correlation discount
# reduces effective weight for correlated pairs.
_BASE_BRANCH_WEIGHT: float = 1.0

# Action thresholds by regime.
_ACTION_THRESHOLDS: dict[str, dict[str, float]] = {
    "趋势上涨": {"buy": 0.52, "strong_buy": 0.65},
    "趋势下跌": {"buy": 0.60, "strong_buy": 0.72},
    "震荡低波": {"buy": 0.55, "strong_buy": 0.68},
    "震荡高波": {"buy": 0.58, "strong_buy": 0.70},
    "未知": {"buy": 0.55, "strong_buy": 0.68},
}

_EPSILON = 1e-8


def _safe_log_odds(p: float) -> float:
    """Convert probability to log-odds, clamped to avoid infinities."""
    p = max(_EPSILON, min(1.0 - _EPSILON, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Convert log-odds back to probability."""
    if x > 20:
        return 1.0 - _EPSILON
    if x < -20:
        return _EPSILON
    return 1.0 / (1.0 + math.exp(-x))


class BayesianPosteriorEngine:
    """Compute posterior from prior + likelihoods with correlation discount."""

    def compute_posterior(
        self,
        prior: PriorSet,
        likelihoods: LikelihoodSet,
        *,
        symbol: str = "",
        company_name: str = "",
        regime: str = "未知",
        is_degraded: dict[str, bool] | None = None,
    ) -> PosteriorResult:
        """Run the full Bayesian update.

        Steps:
        1. Convert composite prior to log-odds
        2. For each branch likelihood, compute log-likelihood-ratio
        3. Apply correlation discount to correlated pairs
        4. Sum discounted log-LR and add to prior log-odds
        5. Convert back to posterior probability
        6. Apply coverage, fallback, and data-quality penalties
        7. Compute action score using regime-aware thresholds
        """
        degraded = is_degraded or {}
        prior_log_odds = _safe_log_odds(prior.composite_prior)
        metadata = dict(likelihoods.metadata or {})
        profile = str(metadata.get("profile", "classic") or "classic").strip().lower() or "classic"
        history_confidence = float(metadata.get("history_confidence", 0.0) or 0.0)
        avg_reliability = float(metadata.get("avg_reliability", 0.0) or 0.0)
        recall_bias = float(metadata.get("recall_bias", 0.0) or 0.0)
        momentum_strength = float(metadata.get("momentum_strength", 0.0) or 0.0)
        fake_breakout_penalty = float(metadata.get("fake_breakout_penalty", 0.0) or 0.0)
        setup_failure_penalty = float(metadata.get("setup_failure_penalty", 0.0) or 0.0)
        crowding_penalty = float(metadata.get("crowding_penalty", 0.0) or 0.0)
        market_pressure = float(metadata.get("market_pressure", 0.0) or 0.0)

        # Build per-branch log-likelihood ratios
        branch_llr: dict[str, float] = {}
        evidence_sources: list[str] = []
        for name, likelihood in likelihoods.as_list():
            if abs(likelihood - 0.50) < 0.01:
                # Neutral likelihood — no information
                branch_llr[name] = 0.0
                continue
            branch_llr[name] = _safe_log_odds(likelihood)
            evidence_sources.append(name)

        # Compute effective weights with correlation discount
        dynamic_weights = {
            str(name): float(weight)
            for name, weight in dict(metadata.get("branch_weights", {}) or {}).items()
        }
        if profile == "momentum_leader" and dynamic_weights:
            weight_values = [max(value, 0.10) for value in dynamic_weights.values()]
            mean_weight = sum(weight_values) / max(len(weight_values), 1)
            weights = {
                name: max(0.10, float(dynamic_weights.get(name, mean_weight))) / max(mean_weight, 1e-6)
                for name in branch_llr
            }
        else:
            weights = {name: _BASE_BRANCH_WEIGHT for name in branch_llr}
        total_correlation_discount = 0.0
        for branch_a, branch_b, rho in _CORRELATION_PAIRS:
            if branch_a in branch_llr and branch_b in branch_llr:
                # Both branches have non-trivial evidence
                if abs(branch_llr[branch_a]) > 0.01 and abs(branch_llr[branch_b]) > 0.01:
                    # Reduce the weight of the less informative branch
                    discount = rho * 0.5  # max 25% reduction per pair
                    lesser = branch_b if abs(branch_llr[branch_a]) >= abs(branch_llr[branch_b]) else branch_a
                    weights[lesser] *= (1.0 - discount)
                    total_correlation_discount += discount

        # Fallback penalty: degraded backends contribute less
        total_fallback_penalty = 0.0
        for name in branch_llr:
            if degraded.get(name, False):
                weights[name] *= 0.60
                total_fallback_penalty += 0.40

        # Compute raw posterior (before coverage/quality penalties)
        weighted_llr_sum = sum(branch_llr[name] * weights[name] for name in branch_llr)
        raw_log_odds = prior_log_odds + weighted_llr_sum
        raw_posterior = _sigmoid(raw_log_odds)

        # Coverage discount: fewer evidence sources -> lower confidence
        num_sources = len(evidence_sources)
        max_sources = 4  # kline, quant, fundamental, intelligence
        coverage_ratio = num_sources / max_sources if max_sources > 0 else 1.0
        coverage_discount = 1.0 - coverage_ratio

        # Data quality penalty
        data_quality_penalty = 0.0
        if prior.data_quality_prior < 0.30:
            data_quality_penalty = 0.15

        # Final posterior with penalties
        if profile == "momentum_leader":
            confidence = 0.25 * coverage_ratio + 0.50 * history_confidence + 0.25 * avg_reliability
            confidence -= data_quality_penalty
            confidence -= fake_breakout_penalty * 0.20
            confidence -= setup_failure_penalty * 0.12
            confidence -= crowding_penalty * 0.08
            confidence = max(0.10, min(0.98, confidence))
        else:
            confidence = max(0.10, coverage_ratio * (1.0 - data_quality_penalty))
        correlation_discounted_posterior = raw_posterior

        # Posterior win rate — the final probability estimate
        posterior_win_rate = raw_posterior
        if profile == "momentum_leader":
            posterior_win_rate = max(
                _EPSILON,
                min(
                    1.0 - _EPSILON,
                    raw_posterior
                    + recall_bias * 0.25
                    + momentum_strength * 0.04
                    - fake_breakout_penalty * 0.06
                    - setup_failure_penalty * 0.05
                    - crowding_penalty * 0.03,
                ),
            )

        # Expected alpha — rough estimate based on distance from neutral
        posterior_expected_alpha = (posterior_win_rate - 0.50) * 0.10  # scale to ~5% max
        posterior_capacity_penalty = max(0.0, total_correlation_discount * 0.05 + total_fallback_penalty * 0.03)
        if profile == "momentum_leader":
            posterior_capacity_penalty += fake_breakout_penalty * 0.04
            posterior_capacity_penalty += setup_failure_penalty * 0.03
            posterior_capacity_penalty += crowding_penalty * 0.02
        posterior_edge_after_costs = posterior_expected_alpha - posterior_capacity_penalty

        # Regime adjustment
        thresholds = _ACTION_THRESHOLDS.get(regime, _ACTION_THRESHOLDS["未知"])
        buy_threshold = thresholds["buy"]
        if profile == "momentum_leader":
            buy_threshold += max(0.0, market_pressure) * 0.04
            buy_threshold += fake_breakout_penalty * 0.02
            buy_threshold += setup_failure_penalty * 0.015
            buy_threshold = min(0.80, buy_threshold)
        regime_adjustment = buy_threshold - 0.55  # relative to neutral threshold

        # Action score: combines win rate, confidence, and regime awareness
        if profile == "momentum_leader":
            posterior_action_score = (
                posterior_win_rate * 0.42
                + confidence * 0.23
                + max(0.0, posterior_edge_after_costs) * 5.0 * 0.18
                + momentum_strength * 0.17
            )
            posterior_action_score -= fake_breakout_penalty * 0.14
            posterior_action_score -= setup_failure_penalty * 0.10
            posterior_action_score -= crowding_penalty * 0.06
            posterior_action_score -= max(0.0, buy_threshold - posterior_win_rate) * 0.45
        else:
            posterior_action_score = (
                posterior_win_rate * 0.60
                + confidence * 0.25
                + max(0.0, posterior_expected_alpha) * 5.0 * 0.15
            )
        posterior_action_score = max(-1.0, min(1.0, posterior_action_score))

        return PosteriorResult(
            symbol=symbol,
            company_name=company_name,
            prior=prior,
            likelihoods=likelihoods,
            raw_posterior=raw_posterior,
            correlation_discounted_posterior=correlation_discounted_posterior,
            posterior_win_rate=posterior_win_rate,
            posterior_expected_alpha=posterior_expected_alpha,
            posterior_confidence=confidence,
            posterior_action_score=posterior_action_score,
            posterior_edge_after_costs=posterior_edge_after_costs,
            posterior_capacity_penalty=posterior_capacity_penalty,
            coverage_discount=coverage_discount,
            fallback_penalty=total_fallback_penalty,
            correlation_discount=total_correlation_discount,
            data_quality_penalty=data_quality_penalty,
            regime_adjustment=regime_adjustment,
            evidence_sources=evidence_sources,
            action_threshold_used=buy_threshold,
            metadata={
                "profile": profile,
                "history_confidence": history_confidence,
                "avg_reliability": avg_reliability,
                "recall_bias": recall_bias,
                "momentum_strength": momentum_strength,
                "fake_breakout_penalty": fake_breakout_penalty,
                "setup_failure_penalty": setup_failure_penalty,
                "crowding_penalty": crowding_penalty,
                "market_pressure": market_pressure,
                "calibration_samples": dict(metadata.get("calibration_samples", {}) or {}),
                "branch_weights": dict(dynamic_weights or {}),
                "kill_switch": bool(
                    profile == "momentum_leader"
                    and (
                        fake_breakout_penalty >= 0.88
                        or setup_failure_penalty >= 0.75
                        or (posterior_edge_after_costs < -0.01 and momentum_strength >= 0.60)
                    )
                ),
                "sector": str(metadata.get("sector", "") or ""),
            },
        )
