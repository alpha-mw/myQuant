"""Authority declarations for the isolated V17 protocol-v3 runtime."""

from __future__ import annotations

from typing import Any, Final

from quant_investor.v17_v3_contract import (
    BROKER_AUTHORITY,
    EXECUTION_AUTHORITY,
    FORMAL_RESEARCH_PUBLICATION_AUTHORITY,
    ORDER_AUTHORITY,
    PRODUCTION_DEFAULT,
    PROTOCOL_VERSION,
    TRADE_AUTHORITY,
)

DELIVERY_STATUS: Final = "NOT_ACTIVATED_DATA_BLOCKED"


def authority_envelope(
    *,
    formal_research_active: bool = False,
) -> dict[str, Any]:
    """Return the only public authority declaration exposed by this runtime."""

    if type(formal_research_active) is not bool:
        raise TypeError("formal_research_active must be boolean")
    return {
        "formal_research_publication_authority": formal_research_active,
        "execution_authority": EXECUTION_AUTHORITY,
        "production_default": PRODUCTION_DEFAULT,
        "broker_authority": BROKER_AUTHORITY,
        "order_authority": ORDER_AUTHORITY,
        "trade_authority": TRADE_AUTHORITY,
    }


__all__ = [
    "BROKER_AUTHORITY",
    "DELIVERY_STATUS",
    "EXECUTION_AUTHORITY",
    "FORMAL_RESEARCH_PUBLICATION_AUTHORITY",
    "ORDER_AUTHORITY",
    "PRODUCTION_DEFAULT",
    "PROTOCOL_VERSION",
    "TRADE_AUTHORITY",
    "authority_envelope",
]
