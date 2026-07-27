"""Production-research control scaffold for ``myquant.v17.v4``.

The package is intentionally incomplete: it exposes the versioned authority
boundary and an explicit verification CLI, but it cannot activate formal
research, change the default research selector, or call a provider.
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

__all__ = [
    "BROKER_AUTHORITY",
    "DELIVERY_STATUS",
    "EXECUTION_AUTHORITY",
    "FORMAL_RESEARCH_PUBLICATION_AUTHORITY",
    "ORDER_AUTHORITY",
    "PROTOCOL_VERSION",
    "RESEARCH_RUNTIME_DEFAULT",
    "TRADE_AUTHORITY",
    "authority_envelope",
]
