"""Closed constants and errors for I3 Theme Intelligence."""

from __future__ import annotations

from typing import Final

from .._core import IntelligenceV2ContractError

REGISTRY_VERSION: Final = "myquant.v17.research-intelligence-v2.theme-registry.v1"
MEMBERSHIP_CATALOG_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.theme-membership-catalog.v1"
)
LIFECYCLE_POLICY_VERSION: Final = "myquant.v17.research-intelligence-v2.theme-lifecycle-policy.v1"
EXPOSURE_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence-v2.theme-exposure-receipt.v1"
COMPONENT_POLICY_VERSION: Final = "myquant.v17.research-intelligence-v2.theme-component-policy.v1"
COMPONENT_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence-v2.theme-component-receipt.v1"
RISK_POLICY_VERSION: Final = "myquant.v17.research-intelligence-v2.theme-risk-policy.v1"
RISK_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence-v2.theme-risk-receipt.v1"

THEME_STATES: Final = frozenset({"AVAILABLE", "NO_MEMBERSHIP", "UNMAPPED", "AMBIGUOUS", "RETIRED"})
CATALOG_SCOPE_STATES: Final = frozenset({"COMPLETE", "INCOMPLETE"})
COVERAGE_STATES: Final = frozenset({"COVERED", "UNMAPPED", "AMBIGUOUS"})
LIFECYCLE_STATES: Final = frozenset({"ACTIVE", "RETIRED"})
EXPOSURE_BASES: Final = frozenset({"DECLARED_MEMBERSHIP", "REVENUE", "PRODUCT", "CUSTOMER"})
COMPONENT_SOURCE_KINDS: Final = frozenset(
    {
        "SOURCE_BOUND_CUSTOMER_EXPOSURE",
        "SOURCE_BOUND_LIFECYCLE",
        "SOURCE_BOUND_PRODUCT_EXPOSURE",
        "SOURCE_BOUND_REVENUE_EXPOSURE",
    }
)
COMPONENT_DIRECTIONS: Final = frozenset({"HIGHER_IS_BETTER", "LOWER_IS_BETTER"})
COMPONENT_MISSING_RULES: Final = frozenset({"BLOCK_COMPONENT", "DROP_METRIC"})
COMPONENT_STATES: Final = frozenset({"AVAILABLE", "MISSING", "BLOCKED"})
RISK_STATES: Final = frozenset({"AVAILABLE", "NO_MEMBERSHIP", "BLOCKED"})


class ThemeContractError(IntelligenceV2ContractError):
    """Fail-closed I3 contract error."""

    exit_code = 2


__all__ = [
    "CATALOG_SCOPE_STATES",
    "COMPONENT_DIRECTIONS",
    "COMPONENT_MISSING_RULES",
    "COMPONENT_POLICY_VERSION",
    "COMPONENT_RECEIPT_VERSION",
    "COMPONENT_SOURCE_KINDS",
    "COMPONENT_STATES",
    "COVERAGE_STATES",
    "EXPOSURE_BASES",
    "EXPOSURE_RECEIPT_VERSION",
    "LIFECYCLE_POLICY_VERSION",
    "LIFECYCLE_STATES",
    "MEMBERSHIP_CATALOG_VERSION",
    "REGISTRY_VERSION",
    "RISK_POLICY_VERSION",
    "RISK_RECEIPT_VERSION",
    "RISK_STATES",
    "THEME_STATES",
    "ThemeContractError",
]
