"""Audited pure algorithms for the isolated v17 protocol-v2 runtime.

The package is deliberately data-in/data-out.  It has no storage, provider,
LLM, broker, order, or trade integration and never imports ``quant_investor.v17``.
"""

from .deep_research import DeepResearchEvaluation, evaluate_deep_research
from .forward_calibration import (
    FundamentalEligibility,
    assess_fundamental_eligibility,
    calibrate_forward_returns,
)
from .fundamental_scoring import FundamentalCandidateSet, score_fundamental_universe
from .optimizer import (
    FeasiblePortfolio,
    OptimizerResult,
    ProposedTrade,
    optimize_lexicographic,
)
from .permissions import (
    apply_permission_restrictions,
    build_permission_restriction,
    determine_trade_permission,
)
from .quant_timing import (
    TimingCalibration,
    calibrate_timing_probabilities,
    compute_latest_scores,
    decide_timing,
)
from .regime_overlay import (
    build_available_overlay_input,
    build_disabled_overlay_input,
    build_unavailable_overlay_input,
    compute_regime_portfolio_overlay,
)
from .transaction_cost import TransactionCostEstimate, estimate_transaction_cost

__all__ = [
    "DeepResearchEvaluation",
    "FeasiblePortfolio",
    "FundamentalCandidateSet",
    "FundamentalEligibility",
    "OptimizerResult",
    "ProposedTrade",
    "TimingCalibration",
    "TransactionCostEstimate",
    "apply_permission_restrictions",
    "assess_fundamental_eligibility",
    "build_available_overlay_input",
    "build_disabled_overlay_input",
    "build_permission_restriction",
    "build_unavailable_overlay_input",
    "calibrate_forward_returns",
    "calibrate_timing_probabilities",
    "compute_latest_scores",
    "compute_regime_portfolio_overlay",
    "decide_timing",
    "determine_trade_permission",
    "estimate_transaction_cost",
    "evaluate_deep_research",
    "optimize_lexicographic",
    "score_fundamental_universe",
]
