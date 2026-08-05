"""Research-only runtime for V17 v4 forward evidence and dynamic Shadow."""

from __future__ import annotations

PROTOCOL_VERSION = "myquant.v17.v4"
RESEARCH_ONLY = True
MAINLINE_AUTHORITY = False
PRODUCTION_AUTHORITY = False
EXECUTION_AUTHORITY = False
BROKER_AUTHORITY = False
ORDER_AUTHORITY = False
TRADE_AUTHORITY = False

__all__ = [
    "BROKER_AUTHORITY",
    "EXECUTION_AUTHORITY",
    "MAINLINE_AUTHORITY",
    "ORDER_AUTHORITY",
    "PRODUCTION_AUTHORITY",
    "PROTOCOL_VERSION",
    "RESEARCH_ONLY",
    "TRADE_AUTHORITY",
]
