"""Research-only contracts for the myQuant v16 candidate-decision pipeline.

The v16 package is deliberately separate from the v15 production runtime.  Its
types and validators can be exercised offline while activation evidence is
being accumulated; importing this package does not mutate registries or create
orders.
"""

from .candidate_pipeline import (
    CandidateUnion,
    CapitalMapping,
    CapitalTarget,
    FormalBranchEvidence,
    FourBranchEvidence,
    LLMBranchVerdict,
    PosteriorMenuItem,
    RetrievalEvidence,
    Stage2Decision,
    Stage2PortfolioDecision,
    build_candidate_union,
    build_posterior_menu,
    map_portfolio_capital,
    seal_four_branch_evidence,
    validate_stage1_review,
    validate_stage2_portfolio,
)
from .stage1_contract import PITFactRow, Stage1FactPackage, build_stage1_fact_package
from .protocol_matrix import PROTOCOL_VERSIONS, protocol_envelope, require_exact_v16_protocol
from .diagnostic import V16NoAgentDiagnostic

__all__ = [
    "CandidateUnion",
    "CapitalMapping",
    "CapitalTarget",
    "FormalBranchEvidence",
    "FourBranchEvidence",
    "LLMBranchVerdict",
    "PosteriorMenuItem",
    "RetrievalEvidence",
    "Stage2Decision",
    "Stage2PortfolioDecision",
    "build_candidate_union",
    "build_posterior_menu",
    "map_portfolio_capital",
    "seal_four_branch_evidence",
    "validate_stage1_review",
    "validate_stage2_portfolio",
    "PITFactRow",
    "Stage1FactPackage",
    "build_stage1_fact_package",
    "PROTOCOL_VERSIONS",
    "protocol_envelope",
    "require_exact_v16_protocol",
    "V16NoAgentDiagnostic",
]
