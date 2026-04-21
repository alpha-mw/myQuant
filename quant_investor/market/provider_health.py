from __future__ import annotations

from typing import Any

from quant_investor.llm_gateway import has_provider_for_model
from quant_investor.kline_backends.hybrid_engine import KlineHybridEngine


def probe_kline_model_capabilities() -> dict[str, Any]:
    try:
        return KlineHybridEngine().health_check()
    except Exception as exc:
        return {
            "kronos_available": False,
            "chronos_available": False,
            "mode": "statistical_only_fallback",
            "error": str(exc),
        }


def detect_provider_health(
    *,
    agent_model: str,
    master_model: str,
) -> dict[str, dict[str, Any]]:
    kline = probe_kline_model_capabilities()
    return {
        "agent": {
            "model": str(agent_model or ""),
            "available": bool(agent_model) and has_provider_for_model(agent_model),
        },
        "master": {
            "model": str(master_model or ""),
            "available": bool(master_model) and has_provider_for_model(master_model),
        },
        "kline": dict(kline),
    }
