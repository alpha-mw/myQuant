"""Signal likelihood mapper — converts branch evidence into calibrated likelihoods.

Each branch's (final_score, final_confidence, reliability) is mapped to a
likelihood value in [0, 1] representing the probability of observing this
signal given the stock will outperform.
"""

from __future__ import annotations

from typing import Any

from quant_investor.bayesian.types import LikelihoodSet
from quant_investor.branch_contracts import BranchResult

# Default branch reliabilities used when metadata is missing.
_DEFAULT_RELIABILITY: dict[str, float] = {
    "kline": 0.65,
    "quant": 0.70,
    "fundamental": 0.60,
    "intelligence": 0.55,
    "macro": 0.50,
}

# Default pairwise correlations between branch signals.
# These are used to discount the joint information content.
_DEFAULT_CORRELATIONS: dict[tuple[str, str], float] = {
    ("kline", "quant"): 0.50,
    ("fundamental", "intelligence"): 0.35,
    ("kline", "fundamental"): 0.15,
    ("kline", "intelligence"): 0.10,
    ("quant", "fundamental"): 0.20,
    ("quant", "intelligence"): 0.15,
}


def _score_to_likelihood(
    score: float,
    confidence: float,
    reliability: float,
) -> float:
    """Map (score, confidence, reliability) to a likelihood in [0.05, 0.95].

    The mapping uses a soft sigmoid-like transform:
    - score in [-1, 1] drives the center
    - confidence and reliability moderate the extremity
    """
    # Normalize score to [0, 1] range
    raw = (score + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    raw = max(0.0, min(1.0, raw))

    # Blend with 0.5 (neutral) based on confidence and reliability
    blend_strength = confidence * reliability
    blend_strength = max(0.0, min(1.0, blend_strength))

    likelihood = 0.50 + (raw - 0.50) * blend_strength
    return max(0.05, min(0.95, likelihood))


class SignalLikelihoodMapper:
    """Map branch results to calibrated likelihood values."""

    def compute_likelihoods(
        self,
        *,
        branch_results: dict[str, BranchResult],
        symbol: str,
        candidate_symbols: set[str] | None = None,
    ) -> LikelihoodSet:
        """Compute likelihoods for a single symbol from all available branches.

        For branches that only run on candidates (fundamental, intelligence),
        non-candidate symbols get a neutral likelihood of 0.50.
        """
        candidate_only_branches = {"fundamental", "intelligence"}
        is_candidate = candidate_symbols is None or symbol in (candidate_symbols or set())

        kline_l = self._branch_likelihood("kline", branch_results, symbol)
        quant_l = self._branch_likelihood("quant", branch_results, symbol)

        if is_candidate:
            fundamental_l = self._branch_likelihood("fundamental", branch_results, symbol)
            intelligence_l = self._branch_likelihood("intelligence", branch_results, symbol)
        else:
            fundamental_l = 0.50  # neutral for non-candidates
            intelligence_l = 0.50

        evidence_sources = []
        for name in ("kline", "quant", "fundamental", "intelligence"):
            if name in candidate_only_branches and not is_candidate:
                continue
            if name in branch_results:
                evidence_sources.append(name)

        return LikelihoodSet(
            kline_likelihood=kline_l,
            quant_likelihood=quant_l,
            fundamental_likelihood=fundamental_l,
            intelligence_likelihood=intelligence_l,
            correlation_matrix={
                f"{a}_{b}": corr for (a, b), corr in _DEFAULT_CORRELATIONS.items()
            },
            metadata={"evidence_sources": evidence_sources},
        )

    @staticmethod
    def _branch_likelihood(
        branch_name: str,
        branch_results: dict[str, BranchResult],
        symbol: str,
    ) -> float:
        result = branch_results.get(branch_name)
        if result is None:
            return 0.50  # neutral

        score = float(result.symbol_scores.get(symbol, result.final_score))
        confidence = float(result.final_confidence)
        reliability = float(result.metadata.get("reliability", _DEFAULT_RELIABILITY.get(branch_name, 0.50)))

        return _score_to_likelihood(score, confidence, reliability)
