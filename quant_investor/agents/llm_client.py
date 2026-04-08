"""
统一异步 LLM 客户端。

基于 aiohttp 实现，根据 model 前缀自动路由到对应 provider，
不引入额外 SDK 依赖。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from quant_investor.branch_contracts import LLMUsageRecord
from quant_investor.llm_gateway import (
    estimate_cost_usd,
    estimate_message_tokens,
    estimate_text_tokens,
    record_fallback_event,
    record_usage,
)
from quant_investor.logger import get_logger
from quant_investor.llm_transport import build_openai_compatible_completion_body, normalize_label

try:
    import aiohttp
    _AiohttpClientError: type[Exception] = aiohttp.ClientError
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    _AiohttpClientError = OSError  # fallback exception type

_logger = get_logger("LLMClient")


class LLMCallError(Exception):
    """LLM 调用失败。"""


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

_PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "qwen": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "kimi": {
        "env_key": "KIMI_API_KEY",
        "base_url": "https://api.moonshot.cn/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
}


def _detect_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("qwen"):
        return "qwen"
    if m.startswith("moonshot"):
        return "kimi"
    raise LLMCallError(f"Cannot detect provider for model: {model}")


def has_any_provider() -> bool:
    """Check if at least one LLM provider key is configured."""
    return any(bool(os.getenv(cfg["env_key"])) for cfg in _PROVIDER_CONFIGS.values())


def has_provider_for_model(model: str) -> bool:
    """Check whether the provider required by a model currently has credentials."""
    try:
        provider = _detect_provider(model)
    except LLMCallError:
        return False
    return bool(os.getenv(_PROVIDER_CONFIGS[provider]["env_key"], ""))


def resolve_default_model(preferred_model: str = "") -> str:
    """Return the preferred model when available, otherwise the first configured default."""
    preferred = str(preferred_model or "").strip()
    if preferred and has_provider_for_model(preferred):
        return preferred
    for candidate in ("moonshot-v1-128k", "deepseek-chat", "qwen-plus"):
        if has_provider_for_model(candidate):
            return candidate
    return preferred or "moonshot-v1-128k"


def _get_api_key(provider: str) -> str:
    cfg = _PROVIDER_CONFIGS[provider]
    key = os.getenv(cfg["env_key"], "")
    if not key:
        raise LLMCallError(f"Missing env var {cfg['env_key']} for provider {provider}")
    return key


# ---------------------------------------------------------------------------
# Request builders per provider
# ---------------------------------------------------------------------------

def _build_openai_compatible(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    response_json: bool,
    api_key: str,
    base_url: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = build_openai_compatible_completion_body(
        provider=provider,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        response_json=response_json,
    )
    return base_url, headers, body


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_openai_response(data: dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMCallError(f"Unexpected OpenAI response structure: {exc}") from exc


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Async LLM client with provider auto-routing and retry."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 2) -> None:
        self.timeout = timeout
        self.max_retries = max_retries

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        response_json: bool = True,
        stage: str = "",
        actor_name: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        """Send a chat completion request and return parsed JSON."""
        provider = _detect_provider(model)
        api_key = _get_api_key(provider)
        stage_name = normalize_label(stage) or "unlabeled_stage"
        branch_or_agent_name = normalize_label(actor_name)
        prompt_tokens_est = estimate_message_tokens(messages)

        cfg = _PROVIDER_CONFIGS[provider]
        url, headers, body = _build_openai_compatible(
            provider, model, messages, max_tokens, response_json, api_key, cfg["base_url"],
        )

        parse_fn = {
            "deepseek": _parse_openai_response,
            "qwen": _parse_openai_response,
            "kimi": _parse_openai_response,
        }[provider]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = await self._http_post(url, headers, body)
                response_data = json.loads(raw_text)
                content_text = parse_fn(response_data)
                completion_tokens = estimate_text_tokens(content_text)
                record_usage(
                    LLMUsageRecord(
                        stage=stage_name,
                        branch_or_agent_name=branch_or_agent_name,
                        provider=provider,
                        model=model,
                        prompt_tokens=prompt_tokens_est,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens_est + completion_tokens,
                        latency_ms=0,
                        success=True,
                        fallback=False,
                        estimated_cost_usd=estimate_cost_usd(model, prompt_tokens_est, completion_tokens),
                    )
                )
                return self._parse_json_content(content_text)
            except (_AiohttpClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                _logger.warning(f"LLM call attempt {attempt}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * attempt)
            except LLMCallError as exc:
                last_exc = exc
                _logger.warning(f"LLM call attempt {attempt}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * attempt)
            except json.JSONDecodeError as exc:
                last_exc = exc
                _logger.warning(f"JSON parse failed on attempt {attempt}/{self.max_retries}: {exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * attempt)

        record_fallback_event(
            stage=stage_name,
            branch_or_agent_name=branch_or_agent_name,
            model=model,
            reason=str(last_exc or "llm_call_failed"),
            provider=provider,
            prompt_tokens=prompt_tokens_est,
            latency_ms=0,
        )
        raise LLMCallError(f"All {self.max_retries} attempts failed for model={model}: {last_exc}")

    async def _http_post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> str:
        if aiohttp is None:
            raise LLMCallError("aiohttp is required for LLM calls. Install with: pip install aiohttp")
        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, headers=headers, json=body) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise LLMCallError(f"HTTP {resp.status}: {text[:500]}")
                return text

    @staticmethod
    def _parse_json_content(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # remove opening ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)
