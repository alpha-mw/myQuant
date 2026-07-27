"""Authority boundary for the V17 v4 production-research scaffold."""

from __future__ import annotations

from typing import Any, Final

from quant_investor.v17_v4_contract import (
    BROKER_AUTHORITY,
    EXECUTION_AUTHORITY,
    FORMAL_RESEARCH_PUBLICATION_AUTHORITY,
    ORDER_AUTHORITY,
    PROTOCOL_VERSION,
    RESEARCH_RUNTIME_DEFAULT,
    TRADE_AUTHORITY,
)

DELIVERY_STATUS: Final = "CONTRACT_SCAFFOLD_NOT_ACTIVATED"
STATE: Final = "V15_DEFAULT"


def authority_envelope() -> dict[str, Any]:
    """Return the only authority exposed before formal activation."""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "state": STATE,
        "formal_research_publication": FORMAL_RESEARCH_PUBLICATION_AUTHORITY,
        "research_runtime_default": RESEARCH_RUNTIME_DEFAULT,
        "execution": EXECUTION_AUTHORITY,
        "broker": BROKER_AUTHORITY,
        "order": ORDER_AUTHORITY,
        "trade": TRADE_AUTHORITY,
    }


__all__ = [
    "BROKER_AUTHORITY",
    "DELIVERY_STATUS",
    "EXECUTION_AUTHORITY",
    "FORMAL_RESEARCH_PUBLICATION_AUTHORITY",
    "ORDER_AUTHORITY",
    "PROTOCOL_VERSION",
    "RESEARCH_RUNTIME_DEFAULT",
    "STATE",
    "TRADE_AUTHORITY",
    "authority_envelope",
]
