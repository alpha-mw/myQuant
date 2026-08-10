"""Closed constants and errors for I2 Industry Intelligence."""

from __future__ import annotations

from typing import Final

from .._core import IntelligenceV2ContractError

IDENTITY_POLICY_VERSION: Final = "myquant.v17.research-intelligence-v2.industry-identity-policy.v1"
TAXONOMY_VERSION: Final = "myquant.v17.research-intelligence-v2.industry-taxonomy.v1"
MEMBERSHIP_CATALOG_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.industry-membership-catalog.v1"
)
EVIDENCE_VERSION: Final = "myquant.v17.research-intelligence-v2.industry-evidence.v1"
COMPONENT_POLICY_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.industry-component-policy.v1"
)
COMPONENT_RECEIPT_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.industry-component-receipt.v1"
)
EVALUATION_RECEIPT_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.industry-evaluation-receipt.v1"
)

INDUSTRY_STATES: Final = frozenset({"AVAILABLE", "UNMAPPED", "AMBIGUOUS"})
COMPONENT_DIMENSIONS: Final = (
    "CAPEX",
    "DEMAND",
    "EARNINGS_REVISION",
    "INVENTORY",
    "PRICING_POWER",
    "SUPPLY",
)
DIRECTIONS: Final = frozenset({"HIGHER_IS_BETTER", "LOWER_IS_BETTER"})
MISSING_RULES: Final = frozenset({"BLOCK_COMPONENT", "DROP_METRIC"})
TAXONOMY_STATUSES: Final = frozenset({"ACTIVE", "RETIRED"})


class IndustryContractError(IntelligenceV2ContractError):
    """Fail-closed I2 contract error."""

    exit_code = 2


__all__ = [
    "COMPONENT_DIMENSIONS",
    "COMPONENT_POLICY_VERSION",
    "COMPONENT_RECEIPT_VERSION",
    "DIRECTIONS",
    "EVALUATION_RECEIPT_VERSION",
    "EVIDENCE_VERSION",
    "IDENTITY_POLICY_VERSION",
    "INDUSTRY_STATES",
    "IndustryContractError",
    "MEMBERSHIP_CATALOG_VERSION",
    "MISSING_RULES",
    "TAXONOMY_STATUSES",
    "TAXONOMY_VERSION",
]
