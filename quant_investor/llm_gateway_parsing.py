"""Pure parsing helpers for LLM gateway responses."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from quant_investor.llm_gateway_types import LLMCallError


def parse_openai_response(data: dict[str, Any]) -> str:
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMCallError(f"Unexpected OpenAI response structure: {exc}") from exc


def extract_usage(
    provider: str,
    response_data: dict[str, Any],
    *,
    prompt_fallback: int,
    completion_fallback: int,
) -> tuple[int, int, int]:
    prompt_tokens = prompt_fallback
    completion_tokens = completion_fallback
    total_tokens = prompt_fallback + completion_fallback

    if provider in {"deepseek", "qwen", "kimi"}:
        usage = response_data.get("usage", {})
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens", prompt_fallback) or prompt_fallback)
            completion_tokens = int(usage.get("completion_tokens", completion_fallback) or completion_fallback)
            total_tokens = int(
                usage.get("total_tokens", prompt_tokens + completion_tokens)
                or (prompt_tokens + completion_tokens)
            )

    return prompt_tokens, completion_tokens, total_tokens


def parse_json_content(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    def _ensure_dict(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMCallError("LLM JSON response must be a JSON object")
        return payload

    candidates = [cleaned]
    trailing_comma_repaired = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    if trailing_comma_repaired != cleaned:
        candidates.append(trailing_comma_repaired)

    for candidate in candidates:
        try:
            return _ensure_dict(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    python_like_candidates = list(candidates)
    normalized_literals = re.sub(r"\bnull\b", "None", cleaned, flags=re.IGNORECASE)
    normalized_literals = re.sub(r"\btrue\b", "True", normalized_literals, flags=re.IGNORECASE)
    normalized_literals = re.sub(r"\bfalse\b", "False", normalized_literals, flags=re.IGNORECASE)
    if normalized_literals not in python_like_candidates:
        python_like_candidates.append(normalized_literals)
    repaired_literals = re.sub(r",(\s*[}\]])", r"\1", normalized_literals)
    if repaired_literals not in python_like_candidates:
        python_like_candidates.append(repaired_literals)

    for candidate in python_like_candidates:
        try:
            return _ensure_dict(ast.literal_eval(candidate))
        except (SyntaxError, ValueError):
            continue

    raise LLMCallError("LLM JSON response must be a JSON object")


__all__ = [
    "extract_usage",
    "parse_json_content",
    "parse_openai_response",
]
