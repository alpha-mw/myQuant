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
from .public_surfaces import (
    PublicSurfaceError,
    build_dashboard_contract_v4,
    build_public_surface_compatibility_receipts,
    publish_canary_snapshot,
    resolve_public_run,
)

__all__ = [
    "BROKER_AUTHORITY",
    "DELIVERY_STATUS",
    "EXECUTION_AUTHORITY",
    "FORMAL_RESEARCH_PUBLICATION_AUTHORITY",
    "ORDER_AUTHORITY",
    "PROTOCOL_VERSION",
    "RESEARCH_RUNTIME_DEFAULT",
    "TRADE_AUTHORITY",
    "PublicSurfaceError",
    "authority_envelope",
    "build_dashboard_contract_v4",
    "build_public_surface_compatibility_receipts",
    "publish_canary_snapshot",
    "resolve_public_run",
]
