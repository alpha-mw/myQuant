"""Pure, unwired contracts for the myQuant v17 lifecycle protocol v2.

Importing this package must never import the legacy ``quant_investor.v17``
runtime.  Phase 0 intentionally exposes contracts only; it grants no runtime,
production, trading, or publication authority.
"""

from __future__ import annotations

PROTOCOL_VERSION = "myquant.v17.v2"
RUNTIME_AUTHORITY = False

__all__ = ["PROTOCOL_VERSION", "RUNTIME_AUTHORITY"]
