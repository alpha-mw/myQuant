"""Permanent authority boundary for the V17 v5 research extension."""

from __future__ import annotations

from typing import Any, Final

from quant_investor.v17_v5_contract import (
    BROKER_AUTHORITY,
    CANARY_AUTHORITY,
    EXECUTION_AUTHORITY,
    FACTOR_GOVERNANCE_WRITE_AUTHORITY,
    FORMAL_ACTIVATION_AUTHORITY,
    FORMAL_RESEARCH_PUBLICATION_AUTHORITY,
    LLM_AUTHORITY,
    ORDER_AUTHORITY,
    PORTFOLIO_AUTHORITY,
    PROMOTION_AUTHORITY,
    PROTOCOL_VERSION,
    PROVIDER_AUTHORITY,
    RESEARCH_RUNTIME_DEFAULT,
    SELECTOR_AUTHORITY,
    TRADE_AUTHORITY,
)

DELIVERY_STATUS: Final = "SPRINT1D_CAUSAL_REGIME_EVIDENCE_ADAPTER_AVAILABLE_NOT_OPERATIONAL"
STATE: Final = "V15_DEFAULT"
GLOBAL_ACTIVATION_STATE: Final = "INACTIVE"
RUN_STATE: Final = "INACTIVE"


def authority_envelope() -> dict[str, Any]:
    return {
        "broker": BROKER_AUTHORITY,
        "canary": CANARY_AUTHORITY,
        "execution": EXECUTION_AUTHORITY,
        "factor_governance_write": FACTOR_GOVERNANCE_WRITE_AUTHORITY,
        "formal_activation": FORMAL_ACTIVATION_AUTHORITY,
        "formal_research_publication": FORMAL_RESEARCH_PUBLICATION_AUTHORITY,
        "llm": LLM_AUTHORITY,
        "order": ORDER_AUTHORITY,
        "portfolio": PORTFOLIO_AUTHORITY,
        "promotion": PROMOTION_AUTHORITY,
        "provider": PROVIDER_AUTHORITY,
        "research_runtime_default": RESEARCH_RUNTIME_DEFAULT,
        "selector": SELECTOR_AUTHORITY,
        "trade": TRADE_AUTHORITY,
    }


__all__ = [
    "DELIVERY_STATUS",
    "GLOBAL_ACTIVATION_STATE",
    "RUN_STATE",
    "STATE",
    "authority_envelope",
]
