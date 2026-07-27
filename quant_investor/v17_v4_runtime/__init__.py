"""Production-research controls for ``myquant.v17.v4``.

Formal research publication is available only through the exact, crash-safe
activation service.  Importing the package still grants no publication,
default-selector, execution, broker, order, or trade authority.
"""

from __future__ import annotations

from .authority import (
    BROKER_AUTHORITY,
    DELIVERY_STATUS,
    EXECUTION_AUTHORITY,
    FORMAL_RESEARCH_PUBLICATION_AUTHORITY,
    ORDER_AUTHORITY,
    PROTOCOL_VERSION,
    RESEARCH_RUNTIME_DEFAULT,
    TRADE_AUTHORITY,
    authority_envelope,
)
from .canary_control import (
    CanaryError,
    CanaryService,
    build_canary_transition_intent,
    build_dual_run_comparison,
    build_historical_canary_policy,
    evaluate_operational_canary,
)
from .eligibility_control import (
    EligibilityError,
    EligibilityService,
    build_rollback_drill_receipt,
    build_validation_receipt,
)
from .public_surfaces import (
    PublicSurfaceError,
    build_dashboard_contract_v4,
    build_public_surface_compatibility_receipts,
    publish_canary_snapshot,
    resolve_public_run,
)

__all__ = [
    "BROKER_AUTHORITY",
    "CanaryError",
    "CanaryService",
    "DELIVERY_STATUS",
    "EXECUTION_AUTHORITY",
    "EligibilityError",
    "EligibilityService",
    "FORMAL_RESEARCH_PUBLICATION_AUTHORITY",
    "ORDER_AUTHORITY",
    "PROTOCOL_VERSION",
    "RESEARCH_RUNTIME_DEFAULT",
    "TRADE_AUTHORITY",
    "PublicSurfaceError",
    "authority_envelope",
    "build_canary_transition_intent",
    "build_dashboard_contract_v4",
    "build_dual_run_comparison",
    "build_historical_canary_policy",
    "build_public_surface_compatibility_receipts",
    "build_rollback_drill_receipt",
    "build_validation_receipt",
    "evaluate_operational_canary",
    "publish_canary_snapshot",
    "resolve_public_run",
]
