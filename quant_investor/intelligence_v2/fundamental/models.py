"""Closed constants for I4 Fundamental Intelligence Profile v1."""

from __future__ import annotations

from typing import Final

from .._core import IntelligenceV2ContractError

COMPONENT_POLICY_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.fundamental-component-policy.v1"
)
PROFILE_VERSION: Final = "myquant.v17.research-intelligence-v2.fundamental-intelligence-profile.v1"
COMPONENTS: Final = (
    "industry_cycle",
    "earnings_revision",
    "theme_narrative",
    "valuation",
    "governance",
)
FINANCIAL_QUALITY_METRICS: Final = (
    "roe",
    "ocf_to_profit",
    "debt_to_assets",
)
COMPONENT_DIRECTIONS: Final = frozenset({"HIGHER_IS_BETTER", "LOWER_IS_BETTER"})
MISSING_RULES: Final = frozenset({"BLOCK_COMPONENT", "DROP_METRIC"})
PERCENTILE_METHOD: Final = "TYPE_7_AVERAGE_TIE"
PROFILE_STATUSES: Final = frozenset({"COMPLETE", "PARTIAL", "UNAVAILABLE"})
FUNDAMENTAL_SCORER_IMPLEMENTATION_SHA256_V3: Final = (
    "35fcd9ac98bb1ef51b244c95f20db4489dc6ffdf5adcd51b1bde69ab5369f417"
)

INDUSTRY_COMPONENT_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.industry-component-receipt.v1"
)
THEME_COMPONENT_VERSION: Final = "myquant.v17.research-intelligence-v2.theme-component-receipt.v1"
INDUSTRY_PROJECTION_METRIC: Final = "I2_INDUSTRY_COMPONENT_SCORE"
THEME_PROJECTION_METRIC: Final = "I3_THEME_COMPONENT_SCORE"


class FundamentalContractError(IntelligenceV2ContractError):
    """Fail-closed I4 contract error."""

    exit_code = 2


__all__ = [
    "COMPONENTS",
    "COMPONENT_DIRECTIONS",
    "COMPONENT_POLICY_VERSION",
    "FINANCIAL_QUALITY_METRICS",
    "FUNDAMENTAL_SCORER_IMPLEMENTATION_SHA256_V3",
    "FundamentalContractError",
    "INDUSTRY_COMPONENT_VERSION",
    "INDUSTRY_PROJECTION_METRIC",
    "MISSING_RULES",
    "PERCENTILE_METHOD",
    "PROFILE_STATUSES",
    "PROFILE_VERSION",
    "THEME_COMPONENT_VERSION",
    "THEME_PROJECTION_METRIC",
]
