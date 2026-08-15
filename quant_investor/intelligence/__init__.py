"""Stable, research-only Investment Intelligence.

The package is intentionally unversioned at the Python surface.  Artifact
contract identity is hash-bound by :mod:`quant_investor.contracts`, and every
builder remains inactive until a separately verified System generation is
activated.
"""

from ._common import IntelligenceError, NO_AUTHORITY
from .advisory import replay_advisory, review_advisory
from .decision_context import build_decision_context, validate_decision_context
from .fundamental import assess_fundamental, validate_fundamental_assessment
from .industry import assess_industry, validate_industry_assessment
from .investment_decision import (
    DECISION_STATES,
    make_investment_decision,
    validate_investment_decision,
)
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
from .theme import assess_theme, validate_theme_assessment

__all__ = [
    "DECISION_STATES",
    "IntelligenceError",
    "NO_AUTHORITY",
    "assess_fundamental",
    "assess_graduation",
    "assess_industry",
    "assess_readiness",
    "assess_theme",
    "build_decision_context",
    "compile_evidence",
    "construct_research_portfolio",
    "evaluate",
    "forward",
    "inspect",
    "make_investment_decision",
    "observe_paper_portfolio",
    "replay_advisory",
    "review_advisory",
    "validate_decision_context",
    "validate_fundamental_assessment",
    "validate_industry_assessment",
    "validate_investment_decision",
    "validate_research_portfolio",
    "validate_readiness",
    "validate_theme_assessment",
]
