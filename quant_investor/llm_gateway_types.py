"""
LLM gateway contracts and static provider registries.

This module is intentionally pure: it defines the data contracts, public
registry tables, and static policy constants used by ``llm_gateway`` without
owning runtime state or network behavior.
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass, field
from pathlib import Path


class LLMCallError(Exception):
    """LLM 调用失败。"""


class LLMProviderResponseError(LLMCallError):
    """Provider returned a non-200 response."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = int(status_code or 0)
        self.headers = dict(headers or {})


@dataclass(frozen=True)
class LLMProviderSpec:
    name: str
    env_key: str
    base_url: str
    auth_header: str
    auth_prefix: str = ""
    default_model: str = ""


@dataclass(frozen=True)
class LLMModelPricing:
    model: str
    prompt_usd_per_1m: float
    completion_usd_per_1m: float


@dataclass
class LLMUsageSessionHandle:
    session_id: str
    _token: contextvars.Token[str | None] | None = field(repr=False, default=None)


@dataclass(frozen=True)
class ProviderFailureAnalysis:
    status_code: int
    reason: str
    retry_after_seconds: float = 0.0
    is_rate_limited: bool = False
    should_cooldown_provider: bool = False
    cooldown_seconds: float = 0.0


@dataclass
class ProviderCooldown:
    until_monotonic: float
    reason: str
    status_code: int = 0


LLM_PROVIDER_REGISTRY: dict[str, LLMProviderSpec] = {
    "deepseek": LLMProviderSpec(
        name="deepseek",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1/chat/completions",
        auth_header="Authorization",
        auth_prefix="Bearer ",
        default_model="deepseek-chat",
    ),
    "qwen": LLMProviderSpec(
        name="qwen",
        env_key="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        auth_header="Authorization",
        auth_prefix="Bearer ",
        default_model="qwen3.5-plus",
    ),
    "kimi": LLMProviderSpec(
        name="kimi",
        env_key="KIMI_API_KEY",
        base_url="https://api.moonshot.cn/v1/chat/completions",
        auth_header="Authorization",
        auth_prefix="Bearer ",
        default_model="moonshot-v1-128k",
    ),
}

LLM_MODEL_PRICING_REGISTRY: dict[str, LLMModelPricing] = {
    "deepseek-chat": LLMModelPricing(
        "deepseek-chat",
        prompt_usd_per_1m=0.27,
        completion_usd_per_1m=1.10,
    ),
    "deepseek-reasoner": LLMModelPricing(
        "deepseek-reasoner",
        prompt_usd_per_1m=0.55,
        completion_usd_per_1m=2.19,
    ),
    "qwen3.5-plus": LLMModelPricing("qwen3.5-plus", prompt_usd_per_1m=0.11, completion_usd_per_1m=0.28),
    "qwen3.5-flash": LLMModelPricing("qwen3.5-flash", prompt_usd_per_1m=0.04, completion_usd_per_1m=0.08),
    "qwen3.6-plus": LLMModelPricing("qwen3.6-plus", prompt_usd_per_1m=0.11, completion_usd_per_1m=0.28),
    "qwen-turbo": LLMModelPricing("qwen-turbo", prompt_usd_per_1m=0.04, completion_usd_per_1m=0.08),
    "moonshot-v1-8k": LLMModelPricing(
        "moonshot-v1-8k",
        prompt_usd_per_1m=1.64,
        completion_usd_per_1m=1.64,
    ),
    "moonshot-v1-32k": LLMModelPricing(
        "moonshot-v1-32k",
        prompt_usd_per_1m=3.28,
        completion_usd_per_1m=3.28,
    ),
    "moonshot-v1-128k": LLMModelPricing(
        "moonshot-v1-128k",
        prompt_usd_per_1m=8.20,
        completion_usd_per_1m=8.20,
    ),
}

LLM_STAGE_NAMES: dict[str, str] = {
    "review_branch_subagent": "Review branch subagent",
    "review_risk_subagent": "Review risk subagent",
    "review_master_agent": "Review master agent",
    "review_branch_overlay": "Review branch overlay",
    "review_master_symbol": "Review master symbol",
    "intelligence_summary": "Intelligence synthesis",
    "news_sentiment": "News sentiment analysis",
    "factor_brainstorm": "Factor brainstorm",
}

LLM_PROVIDER_ENV_KEYS: tuple[str, ...] = tuple(spec.env_key for spec in LLM_PROVIDER_REGISTRY.values())
USAGE_LOG_PATH = Path("data") / "llm_usage.jsonl"
PROVIDER_CONCURRENCY_LIMITS: dict[str, int] = {
    "deepseek": 4,
    "qwen": 4,
    "kimi": 2,
}
_RATE_LIMIT_RETRY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"try again after\s+(\d+(?:\.\d+)?)\s*seconds?", re.IGNORECASE),
    re.compile(r"retry after\s+(\d+(?:\.\d+)?)\s*seconds?", re.IGNORECASE),
)
_COOLDOWN_REASON_KEYWORDS: tuple[str, ...] = (
    "arrearage",
    "invalid api key",
    "unauthorized",
    "forbidden",
    "account disabled",
    "deactivated",
    "quota exhausted",
    "insufficient balance",
)


__all__ = [
    "LLMCallError",
    "LLMProviderResponseError",
    "LLMProviderSpec",
    "LLMModelPricing",
    "LLMUsageSessionHandle",
    "ProviderFailureAnalysis",
    "ProviderCooldown",
    "LLM_PROVIDER_REGISTRY",
    "LLM_MODEL_PRICING_REGISTRY",
    "LLM_STAGE_NAMES",
    "LLM_PROVIDER_ENV_KEYS",
    "USAGE_LOG_PATH",
    "PROVIDER_CONCURRENCY_LIMITS",
]
