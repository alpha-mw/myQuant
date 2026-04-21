"""
Compatibility shim for the legacy agent-layer LLM client.

The canonical implementation now lives in `quant_investor.llm_gateway`.
Keep this module importable for callers and tests during the migration.
"""

from quant_investor.llm_gateway import (
    LLMCallError,
    LLMClient,
    LLMProviderResponseError,
    analyze_provider_failure,
    current_usage_session_id,
    detect_provider as _detect_provider,
    estimate_cost_usd,
    estimate_message_tokens,
    estimate_text_tokens,
    get_provider_cooldown,
    has_any_provider,
    has_provider_for_model,
    record_fallback_event,
    record_usage,
    resolve_default_model,
)

__all__ = [
    "LLMCallError",
    "LLMClient",
    "LLMProviderResponseError",
    "_detect_provider",
    "analyze_provider_failure",
    "current_usage_session_id",
    "estimate_cost_usd",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "get_provider_cooldown",
    "has_any_provider",
    "has_provider_for_model",
    "record_fallback_event",
    "record_usage",
    "resolve_default_model",
]
