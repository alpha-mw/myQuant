from __future__ import annotations

from typing import Any

from quant_investor.llm_gateway import has_provider_for_model


def detect_provider_health(
    *,
    agent_model: str,
    master_model: str,
) -> dict[str, dict[str, Any]]:
    return {
        "agent": {
            "model": str(agent_model or ""),
            "available": bool(agent_model) and has_provider_for_model(agent_model),
        },
        "master": {
            "model": str(master_model or ""),
            "available": bool(master_model) and has_provider_for_model(master_model),
        },
    }
