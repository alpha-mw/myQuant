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

from quant_investor.logger import get_logger

try:
    import aiohttp
    _AiohttpClientError = aiohttp.ClientError
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
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/messages",
        "auth_header": "x-api-key",
        "auth_prefix": "",
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "auth_header": "",
        "auth_prefix": "",
    },
}


def _detect_provider(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("gemini"):
        return "google"
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
    for candidate in ("gpt-5.4-mini", "claude-sonnet-4-6", "deepseek-chat", "gemini-2.5-flash"):
        if has_provider_for_model(candidate):
            return candidate
    return preferred or "gpt-5.4-mini"


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
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if response_json:
        body["response_format"] = {"type": "json_object"}
    return base_url, headers, body


def _build_anthropic(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    response_json: bool,
    api_key: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    system_msg = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append(m)

    if response_json and system_msg:
        system_msg += "\n\nYou MUST respond with valid JSON only. No markdown, no extra text."

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": user_messages,
    }
    if system_msg:
        body["system"] = system_msg
    return _PROVIDER_CONFIGS["anthropic"]["base_url"], headers, body


def _build_google(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    response_json: bool,
    api_key: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    url = _PROVIDER_CONFIGS["google"]["base_url"].format(model=model) + f"?key={api_key}"
    contents = []
    system_instruction = None
    for m in messages:
        if m["role"] == "system":
            system_instruction = {"parts": [{"text": m["content"]}]}
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

    body: dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = system_instruction
    generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens, "temperature": 0.3}
    if response_json:
        generation_config["responseMimeType"] = "application/json"
    body["generationConfig"] = generation_config
    return url, {"Content-Type": "application/json"}, body


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_openai_response(data: dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMCallError(f"Unexpected OpenAI response structure: {exc}") from exc


def _parse_anthropic_response(data: dict[str, Any]) -> str:
    try:
        for block in data["content"]:
            if block.get("type") == "text":
                return block["text"]
        raise LLMCallError("No text block in Anthropic response")
    except (KeyError, IndexError) as exc:
        raise LLMCallError(f"Unexpected Anthropic response structure: {exc}") from exc


def _parse_google_response(data: dict[str, Any]) -> str:
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise LLMCallError(f"Unexpected Google response structure: {exc}") from exc


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
    ) -> dict[str, Any]:
        """Send a chat completion request and return parsed JSON."""
        provider = _detect_provider(model)
        api_key = _get_api_key(provider)

        if provider == "anthropic":
            url, headers, body = _build_anthropic(model, messages, max_tokens, response_json, api_key)
        elif provider == "google":
            url, headers, body = _build_google(model, messages, max_tokens, response_json, api_key)
        else:
            cfg = _PROVIDER_CONFIGS[provider]
            url, headers, body = _build_openai_compatible(
                model, messages, max_tokens, response_json, api_key, cfg["base_url"],
            )

        parse_fn = {
            "openai": _parse_openai_response,
            "deepseek": _parse_openai_response,
            "anthropic": _parse_anthropic_response,
            "google": _parse_google_response,
        }[provider]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = await self._http_post(url, headers, body)
                response_data = json.loads(raw_text)
                content_text = parse_fn(response_data)
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
