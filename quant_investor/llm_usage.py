"""Token and cost accounting records for LLM calls.

These two dataclasses used to live in ``quant_investor.branch_contracts``, which
is a V15 three-branch contract module: importing it drags in
``ARCHITECTURE_VERSION = "15.0.0-stable"``, ``CANONICAL_BRANCH_ORDER`` and the
rest of the three-branch architecture. Commit 389562a deliberately removed that
architecture from ``versioning.py``, which left ``llm_gateway`` importing a
symbol that no longer exists anywhere - so every call into it raised
ImportError.

The records themselves have nothing to do with branches or architecture
versions; they only count tokens, latency and cost. Keeping them here lets the
LLM gateway account for usage without reaching back into V15. The definitions
are carried over verbatim so existing call sites keep their behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMUsageRecord:
    stage: str = ""
    branch_or_agent_name: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    success: bool = True
    fallback: bool = False
    estimated_cost_usd: float = 0.0
    timestamp_utc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMUsageSummary:
    call_count: int = 0
    success_count: int = 0
    fallback_count: int = 0
    failed_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_stage: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_calls(self) -> int:
        return int(self.call_count)

    @total_calls.setter
    def total_calls(self, value: int) -> None:
        self.call_count = int(value)

    @property
    def total_prompt_tokens(self) -> int:
        return int(self.prompt_tokens)

    @total_prompt_tokens.setter
    def total_prompt_tokens(self, value: int) -> None:
        self.prompt_tokens = int(value)

    @property
    def total_completion_tokens(self) -> int:
        return int(self.completion_tokens)

    @total_completion_tokens.setter
    def total_completion_tokens(self, value: int) -> None:
        self.completion_tokens = int(value)
