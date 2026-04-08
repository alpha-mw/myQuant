"""
Shared LLM transport policy helpers.

This module keeps observability labels separate from provider request bodies and
encodes model-specific completion token field selection in one place.
"""

from __future__ import annotations

from typing import Any


# Telemetry fields that must never leak into provider request bodies.
_TELEMETRY_KEYS: frozenset[str] = frozenset({
    "stage",
    "actor_name",
    "reasoning_effort",
    "session_id",
    "run_id",
})

# Provider prefix mapping for model routing.
KNOWN_PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "deepseek": ("deepseek-",),
    "qwen": ("qwen-",),
    "kimi": ("moonshot-",),
}

# Models / providers that support JSON mode (response_format: json_object).
_JSON_MODE_PROVIDERS: frozenset[str] = frozenset({"deepseek", "kimi"})

# Models / providers that accept reasoning_effort parameter.
_REASONING_EFFORT_PREFIXES: dict[str, tuple[str, ...]] = {
    "deepseek": ("deepseek-reasoner",),
}


def normalize_label(value: str | None) -> str:
    return str(value or "").strip()


def should_use_max_completion_tokens(provider: str, model: str) -> bool:
    """Return True when a request should use the modern completion token field."""
    return False


def completion_token_field(provider: str, model: str) -> str:
    return "max_tokens"


def supports_json_mode(provider: str, model: str) -> bool:
    """Return True if the provider/model combination supports JSON response mode."""
    normalized_provider = normalize_label(provider).lower()
    return normalized_provider in _JSON_MODE_PROVIDERS


def supports_reasoning_effort(provider: str, model: str) -> bool:
    """Return True if the provider/model accepts a reasoning_effort parameter."""
    normalized_provider = normalize_label(provider).lower()
    prefixes = _REASONING_EFFORT_PREFIXES.get(normalized_provider, ())
    if not prefixes:
        return False
    normalized_model = normalize_label(model).lower()
    return normalized_model.startswith(prefixes)


def telemetry_safe_body(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *body* with telemetry-only keys stripped.

    Use this before sending a request body to a provider API to ensure
    observability fields like ``stage``, ``actor_name``, and ``reasoning_effort``
    do not pollute the wire payload.
    """
    return {key: value for key, value in body.items() if key not in _TELEMETRY_KEYS}


def build_openai_compatible_completion_body(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    response_json: bool,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        completion_token_field(provider, model): max_tokens,
        "temperature": 0.3,
    }
    if response_json and supports_json_mode(provider, model):
        body["response_format"] = {"type": "json_object"}
    if extra_body:
        body.update(extra_body)
    return telemetry_safe_body(body)
