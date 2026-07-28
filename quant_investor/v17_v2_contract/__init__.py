"""Contracts for the research-only myQuant v17 lifecycle protocol v2.

Importing this package must never import the legacy ``quant_investor.v17``
runtime.  A separately packaged shadow runtime may use these contracts, but
this package grants no production, trading, broker, order, or portfolio
decision authority.
"""

from __future__ import annotations

PROTOCOL_VERSION = "myquant.v17.v2"
SHADOW_RUNTIME_USABLE = True
PRODUCTION_AUTHORITY = False
RUNTIME_AUTHORITY = False

__all__ = [
    "PRODUCTION_AUTHORITY",
    "PROTOCOL_VERSION",
    "RUNTIME_AUTHORITY",
    "SHADOW_RUNTIME_USABLE",
]
