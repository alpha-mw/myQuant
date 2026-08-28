"""Stable, research-only Investment Intelligence.

The package is intentionally unversioned at the Python surface.  Artifact
contract identity is hash-bound by :mod:`quant_investor.contracts`, and every
builder remains inactive until a separately verified System generation is
activated.
"""

from ._common import IntelligenceError, NO_AUTHORITY
from .advisory import replay_advisory, review_advisory
from .decision_context import build_decision_context, validate_decision_context
from .daily import (
    build_daily_research_policy,
    build_factor_research_rank,
    compile_daily_intelligence,
    project_tushare_industry_source,
    project_tushare_theme_source,
    rank_factor_signals,
    validate_daily_research_policy,
    validate_factor_research_rank,
)
from .daily_evidence import (
    build_company_source_evidence,
    build_fundamental_assessments_from_frame,
    build_source_bound_economic_exposure_projection,
    theme_assessment_from_exposure,
    validate_company_source_evidence,
)
from .fundamental import assess_fundamental, validate_fundamental_assessment
from .industry import assess_industry, validate_industry_assessment
from .investment_decision import (
    DECISION_STATES,
    make_investment_decision,
    validate_investment_decision,
)
from .morning import evaluate_morning_cutover, run_morning_strategy
from .portfolio import (
    assess_graduation,
    construct_research_portfolio,
    observe_paper_portfolio,
    validate_research_portfolio,
)
from .runtime import (
    assess_readiness,
    compile_evidence,
    evaluate,
    forward,
    inspect,
    validate_readiness,
)
from .storage import (
    approved_phase_a_policy,
    approved_theme_policy_v2,
    publish_phase_a_policy,
    publish_theme_policy_v2,
)
from .theme import assess_theme, validate_theme_assessment
from .theme_governance import (
    approved_theme_governance_policy,
    build_unverified_economic_exposure_projection,
    validate_theme_governance_policy,
)

__all__ = [
    "DECISION_STATES",
    "IntelligenceError",
    "NO_AUTHORITY",
    "assess_fundamental",
    "assess_graduation",
    "assess_industry",
    "assess_readiness",
    "assess_theme",
    "approved_phase_a_policy",
    "approved_theme_policy_v2",
    "approved_theme_governance_policy",
    "build_decision_context",
    "build_company_source_evidence",
    "build_daily_research_policy",
    "build_factor_research_rank",
    "build_fundamental_assessments_from_frame",
    "build_source_bound_economic_exposure_projection",
    "build_unverified_economic_exposure_projection",
    "compile_evidence",
    "compile_daily_intelligence",
    "construct_research_portfolio",
    "evaluate",
    "evaluate_morning_cutover",
    "forward",
    "inspect",
    "make_investment_decision",
    "observe_paper_portfolio",
    "project_tushare_industry_source",
    "project_tushare_theme_source",
    "publish_phase_a_policy",
    "publish_theme_policy_v2",
    "rank_factor_signals",
    "replay_advisory",
    "review_advisory",
    "run_morning_strategy",
    "theme_assessment_from_exposure",
    "validate_company_source_evidence",
    "validate_decision_context",
    "validate_daily_research_policy",
    "validate_factor_research_rank",
    "validate_fundamental_assessment",
    "validate_industry_assessment",
    "validate_investment_decision",
    "validate_research_portfolio",
    "validate_readiness",
    "validate_theme_assessment",
    "validate_theme_governance_policy",
]
