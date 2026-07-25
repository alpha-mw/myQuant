"""v17 shadow-only investment research contracts.

The package deliberately exports no broker, order, execution, production, or
live-provider surface.  v15 remains the production/default protocol.
"""

from .contracts import (
    Availability,
    FundamentalEligibility,
    QuantTiming,
    TradeSide,
    V17ContractError,
    V17_PACKAGE_VERSION,
)
from .holdings import (
    HOLDINGS_VERSION,
    HoldingsSnapshot,
    build_available_holdings_snapshot,
    build_unavailable_holdings_snapshot,
    validate_holdings_snapshot,
)
from .permissions import (
    TRADE_PERMISSION_VERSION,
    apply_permission_restrictions,
    build_permission_restriction,
    determine_trade_permission,
    validate_trade_permission,
)
from .pretrade import (
    COST_FIELD_ORDER,
    EXECUTION_COST_POLICY_VERSION,
    PRETRADE_RESULT_VERSION,
    build_execution_cost_policy,
    estimate_transaction_cost,
    evaluate_pretrade,
    validate_execution_cost_policy,
    validate_pretrade_result,
)
from .regime_overlay import (
    REGIME_OVERLAY_VERSION,
    RegimePortfolioOverlay,
    build_available_overlay_input,
    build_disabled_overlay_input,
    build_unavailable_overlay_input,
    compute_regime_portfolio_overlay,
    validate_regime_portfolio_overlay,
)
from .risk_policy import (
    OWNER_MANDATE_VERSION,
    RISK_POLICY_VERSION,
    PortfolioRiskPolicySnapshot,
    build_available_risk_policy_snapshot,
    build_unavailable_risk_policy_snapshot,
    seal_risk_policy_from_owner_mandate,
    validate_portfolio_risk_policy_snapshot,
)
from .semantic import (
    canonical_json_bytes,
    seal_semantic,
    semantic_sha256,
    validate_semantic_seal,
)
from .storage import (
    atomic_write_bytes,
    atomic_write_bytes_exact_once,
    atomic_write_json,
    atomic_write_json_exact_once,
    ensure_private_directory,
    ensure_v17_shadow_layout,
    file_sha256,
    read_json,
)

__all__ = [
    "Availability",
    "COST_FIELD_ORDER",
    "EXECUTION_COST_POLICY_VERSION",
    "FundamentalEligibility",
    "HOLDINGS_VERSION",
    "HoldingsSnapshot",
    "OWNER_MANDATE_VERSION",
    "PRETRADE_RESULT_VERSION",
    "PortfolioRiskPolicySnapshot",
    "QuantTiming",
    "REGIME_OVERLAY_VERSION",
    "RISK_POLICY_VERSION",
    "RegimePortfolioOverlay",
    "TRADE_PERMISSION_VERSION",
    "TradeSide",
    "V17ContractError",
    "V17_PACKAGE_VERSION",
    "apply_permission_restrictions",
    "atomic_write_bytes",
    "atomic_write_bytes_exact_once",
    "atomic_write_json",
    "atomic_write_json_exact_once",
    "build_available_holdings_snapshot",
    "build_available_overlay_input",
    "build_available_risk_policy_snapshot",
    "build_disabled_overlay_input",
    "build_execution_cost_policy",
    "build_permission_restriction",
    "build_unavailable_holdings_snapshot",
    "build_unavailable_overlay_input",
    "build_unavailable_risk_policy_snapshot",
    "canonical_json_bytes",
    "compute_regime_portfolio_overlay",
    "determine_trade_permission",
    "ensure_private_directory",
    "ensure_v17_shadow_layout",
    "estimate_transaction_cost",
    "evaluate_pretrade",
    "file_sha256",
    "read_json",
    "seal_risk_policy_from_owner_mandate",
    "seal_semantic",
    "semantic_sha256",
    "validate_execution_cost_policy",
    "validate_holdings_snapshot",
    "validate_portfolio_risk_policy_snapshot",
    "validate_pretrade_result",
    "validate_regime_portfolio_overlay",
    "validate_semantic_seal",
    "validate_trade_permission",
]
