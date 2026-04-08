"""
统一 LLM 网关与使用量观测。

目标：
1. 集中维护 provider / env key / 默认模型 / 价格表 / stage 名称
2. 为每次在线 LLM 调用写入 data/llm_usage.jsonl
3. 为当前主线暴露统一的 usage record / summary
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import json
import math
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from quant_investor.branch_contracts import LLMUsageRecord, LLMUsageSummary
from quant_investor.logger import get_logger
from quant_investor.llm_transport import build_openai_compatible_completion_body, normalize_label

try:
    import aiohttp

    _AiohttpClientError: type[Exception] = aiohttp.ClientError
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    _AiohttpClientError = OSError

_logger = get_logger("LLMGateway")


class LLMCallError(Exception):
    """LLM 调用失败。"""


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
        default_model="qwen-plus",
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
    "deepseek-chat": LLMModelPricing("deepseek-chat", prompt_usd_per_1m=0.27, completion_usd_per_1m=1.10),
    "deepseek-reasoner": LLMModelPricing("deepseek-reasoner", prompt_usd_per_1m=0.55, completion_usd_per_1m=2.19),
    "qwen-plus": LLMModelPricing("qwen-plus", prompt_usd_per_1m=0.11, completion_usd_per_1m=0.28),
    "qwen-turbo": LLMModelPricing("qwen-turbo", prompt_usd_per_1m=0.04, completion_usd_per_1m=0.08),
    "moonshot-v1-8k": LLMModelPricing("moonshot-v1-8k", prompt_usd_per_1m=1.64, completion_usd_per_1m=1.64),
    "moonshot-v1-32k": LLMModelPricing("moonshot-v1-32k", prompt_usd_per_1m=3.28, completion_usd_per_1m=3.28),
    "moonshot-v1-128k": LLMModelPricing("moonshot-v1-128k", prompt_usd_per_1m=8.20, completion_usd_per_1m=8.20),
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

_ACTIVE_USAGE_SESSION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "quant_investor_active_llm_usage_session",
    default=None,
)
_USAGE_LOCK = threading.Lock()
_SESSION_RECORDS: dict[str, list[LLMUsageRecord]] = {}


def detect_provider(model: str) -> str:
    normalized = str(model or "").strip().lower()
    if normalized.startswith("deepseek"):
        return "deepseek"
    if normalized.startswith("qwen"):
        return "qwen"
    if normalized.startswith("moonshot"):
        return "kimi"
    raise LLMCallError(f"Cannot detect provider for model: {model}")


def has_any_provider() -> bool:
    return any(bool(os.getenv(env_key)) for env_key in LLM_PROVIDER_ENV_KEYS)


def has_provider_for_model(model: str) -> bool:
    try:
        provider = detect_provider(model)
    except LLMCallError:
        return False
    return bool(os.getenv(LLM_PROVIDER_REGISTRY[provider].env_key))


def resolve_default_model(preferred_model: str = "") -> str:
    preferred = str(preferred_model or "").strip()
    if preferred and has_provider_for_model(preferred):
        return preferred

    for provider_name in ("kimi", "deepseek", "qwen"):
        provider = LLM_PROVIDER_REGISTRY[provider_name]
        if os.getenv(provider.env_key):
            return provider.default_model

    return preferred or LLM_PROVIDER_REGISTRY["kimi"].default_model


def current_usage_session_id() -> str | None:
    return _ACTIVE_USAGE_SESSION.get()


def start_usage_session(label: str = "") -> LLMUsageSessionHandle:
    session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    with _USAGE_LOCK:
        _SESSION_RECORDS.setdefault(session_id, [])
    token = _ACTIVE_USAGE_SESSION.set(session_id)
    _logger.debug(f"LLM usage session started: {session_id} ({label or 'unlabeled'})")
    return LLMUsageSessionHandle(session_id=session_id, _token=token)


def end_usage_session(handle: LLMUsageSessionHandle | None) -> None:
    if handle is None:
        return
    if handle._token is not None:
        _ACTIVE_USAGE_SESSION.reset(handle._token)
        handle._token = None


@contextmanager
def usage_session(label: str = "") -> Iterator[LLMUsageSessionHandle]:
    handle = start_usage_session(label=label)
    try:
        yield handle
    finally:
        end_usage_session(handle)


def get_usage_records(session_id: str | None = None) -> list[LLMUsageRecord]:
    target = session_id or current_usage_session_id()
    if not target:
        return []
    with _USAGE_LOCK:
        return [LLMUsageRecord(**asdict(record)) for record in _SESSION_RECORDS.get(target, [])]


def build_usage_summary(records: list[LLMUsageRecord]) -> LLMUsageSummary:
    summary = LLMUsageSummary()
    for record in records:
        summary.call_count += 1
        summary.success_count += int(record.success)
        summary.fallback_count += int(record.fallback)
        summary.failed_count += int(not record.success)
        summary.prompt_tokens += int(record.prompt_tokens)
        summary.completion_tokens += int(record.completion_tokens)
        summary.total_tokens += int(record.total_tokens)
        summary.estimated_cost_usd = round(summary.estimated_cost_usd + float(record.estimated_cost_usd), 8)

        stage_bucket = summary.by_stage.setdefault(
            record.stage,
            {
                "call_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "success_count": 0,
                "fallback_count": 0,
            },
        )
        stage_bucket["call_count"] += 1
        stage_bucket["prompt_tokens"] += int(record.prompt_tokens)
        stage_bucket["completion_tokens"] += int(record.completion_tokens)
        stage_bucket["total_tokens"] += int(record.total_tokens)
        stage_bucket["estimated_cost_usd"] = round(
            float(stage_bucket["estimated_cost_usd"]) + float(record.estimated_cost_usd),
            8,
        )
        stage_bucket["success_count"] += int(record.success)
        stage_bucket["fallback_count"] += int(record.fallback)

        model_bucket = summary.by_model.setdefault(
            record.model,
            {
                "call_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        model_bucket["call_count"] += 1
        model_bucket["prompt_tokens"] += int(record.prompt_tokens)
        model_bucket["completion_tokens"] += int(record.completion_tokens)
        model_bucket["total_tokens"] += int(record.total_tokens)
        model_bucket["estimated_cost_usd"] = round(
            float(model_bucket["estimated_cost_usd"]) + float(record.estimated_cost_usd),
            8,
        )

    return summary


def snapshot_usage(session_id: str | None = None) -> tuple[list[LLMUsageRecord], LLMUsageSummary]:
    records = get_usage_records(session_id)
    return records, build_usage_summary(records)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = LLM_MODEL_PRICING_REGISTRY.get(str(model or "").strip())
    if pricing is None:
        return 0.0
    prompt_cost = (max(int(prompt_tokens), 0) / 1_000_000.0) * pricing.prompt_usd_per_1m
    completion_cost = (max(int(completion_tokens), 0) / 1_000_000.0) * pricing.completion_usd_per_1m
    return round(prompt_cost + completion_cost, 8)


def estimate_text_tokens(text: str) -> int:
    return max(1, int(math.ceil(len(str(text or "")) / 4.0)))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    total = 0
    for message in messages:
        total += 4
        total += estimate_text_tokens(message.get("content", ""))
        total += estimate_text_tokens(message.get("role", ""))
    return max(total, 1)


def _usage_log_abspath() -> Path:
    return Path.cwd() / USAGE_LOG_PATH


def _append_usage_record(record: LLMUsageRecord, session_id: str | None = None) -> None:
    payload = asdict(record)
    payload["session_id"] = session_id or ""
    log_path = _usage_log_abspath()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if session_id:
        with _USAGE_LOCK:
            _SESSION_RECORDS.setdefault(session_id, []).append(record)


def record_usage(record: LLMUsageRecord, session_id: str | None = None) -> LLMUsageRecord:
    session = session_id or current_usage_session_id()
    normalized = LLMUsageRecord(
        stage=str(record.stage or ""),
        branch_or_agent_name=str(record.branch_or_agent_name or ""),
        provider=str(record.provider or ""),
        model=str(record.model or ""),
        prompt_tokens=int(record.prompt_tokens or 0),
        completion_tokens=int(record.completion_tokens or 0),
        total_tokens=int(record.total_tokens or 0),
        latency_ms=int(record.latency_ms or 0),
        success=bool(record.success),
        fallback=bool(record.fallback),
        estimated_cost_usd=float(record.estimated_cost_usd or 0.0),
        timestamp_utc=str(record.timestamp_utc or datetime.now(timezone.utc).isoformat()),
        metadata=dict(record.metadata or {}),
    )
    _append_usage_record(normalized, session_id=session)
    return normalized


def record_fallback_event(
    *,
    stage: str,
    branch_or_agent_name: str,
    model: str,
    reason: str,
    provider: str = "",
    prompt_tokens: int = 0,
    latency_ms: int = 0,
    session_id: str | None = None,
) -> LLMUsageRecord:
    resolved_provider = provider
    if not resolved_provider:
        try:
            resolved_provider = detect_provider(model)
        except LLMCallError:
            resolved_provider = "unknown"

    prompt_tokens = max(int(prompt_tokens or 0), 0)
    return record_usage(
        LLMUsageRecord(
            stage=stage,
            branch_or_agent_name=branch_or_agent_name,
            provider=resolved_provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
            latency_ms=max(int(latency_ms or 0), 0),
            success=False,
            fallback=True,
            estimated_cost_usd=estimate_cost_usd(model, prompt_tokens, 0),
            metadata={"reason": str(reason or "")},
        ),
        session_id=session_id,
    )


def _normalize_failure_reason(exc: Exception | None) -> str:
    if exc is None:
        return "llm_call_failed"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__


def render_usage_markdown(summary: LLMUsageSummary, title: str = "## LLM 可观测性") -> str:
    lines = [
        "<!-- llm_usage:start -->",
        title,
    ]
    if summary.call_count <= 0:
        lines.extend(
            [
                "- 本轮未发生在线 LLM 调用。",
                "<!-- llm_usage:end -->",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"- 调用次数: {summary.call_count}",
            f"- 成功次数: {summary.success_count}",
            f"- 降级/回退次数: {summary.fallback_count}",
            f"- Prompt Tokens: {summary.prompt_tokens}",
            f"- Completion Tokens: {summary.completion_tokens}",
            f"- Total Tokens: {summary.total_tokens}",
            f"- 估算成本: ${summary.estimated_cost_usd:.6f}",
        ]
    )

    if summary.by_stage:
        lines.append("- Stage 明细:")
        for stage_name in sorted(summary.by_stage):
            bucket = summary.by_stage[stage_name]
            lines.append(
                "  "
                + f"{stage_name}: calls={bucket['call_count']}, "
                + f"tokens={bucket['total_tokens']}, "
                + f"fallbacks={bucket['fallback_count']}, "
                + f"cost=${float(bucket['estimated_cost_usd']):.6f}"
            )

    lines.append("<!-- llm_usage:end -->")
    return "\n".join(lines)


def update_usage_markdown(report: str, summary: LLMUsageSummary, title: str = "## LLM 可观测性") -> str:
    section = render_usage_markdown(summary, title=title)
    marker_start = "<!-- llm_usage:start -->"
    marker_end = "<!-- llm_usage:end -->"
    if marker_start in report and marker_end in report:
        start = report.index(marker_start)
        end = report.index(marker_end) + len(marker_end)
        return report[:start].rstrip() + "\n\n" + section + report[end:]
    if report.strip():
        return report.rstrip() + "\n\n" + section
    return section


def _get_api_key(provider: str) -> str:
    spec = LLM_PROVIDER_REGISTRY[provider]
    api_key = os.getenv(spec.env_key, "")
    if not api_key:
        raise LLMCallError(f"Missing env var {spec.env_key} for provider {provider}")
    return api_key


def _build_openai_compatible_request(
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


def _parse_openai_response(data: dict[str, Any]) -> str:
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMCallError(f"Unexpected OpenAI response structure: {exc}") from exc


def _extract_usage(
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
            total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

    return prompt_tokens, completion_tokens, total_tokens


class LLMClient:
    """统一异步 LLM 客户端。"""

    def __init__(self, timeout: float = 30.0, max_retries: int = 2, default_reasoning_effort: str = "") -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_reasoning_effort = str(default_reasoning_effort or "").strip()

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
        content_text = await self.complete_text(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            response_json=response_json,
            stage=stage,
            actor_name=actor_name,
            reasoning_effort=reasoning_effort,
        )
        return self._parse_json_content(content_text)

    async def complete_text(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        response_json: bool = False,
        stage: str = "",
        actor_name: str = "",
        reasoning_effort: str = "",
    ) -> str:
        provider = detect_provider(model)
        stage_name = normalize_label(stage) or "unlabeled_stage"
        branch_or_agent_name = normalize_label(actor_name)
        prompt_tokens_est = estimate_message_tokens(messages)
        t0 = time.monotonic()

        try:
            api_key = _get_api_key(provider)
        except LLMCallError as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            record_fallback_event(
                stage=stage_name,
                branch_or_agent_name=branch_or_agent_name,
                model=model,
                reason=str(exc),
                provider=provider,
                prompt_tokens=prompt_tokens_est,
                latency_ms=latency_ms,
            )
            raise

        url, headers, body = _build_openai_compatible_request(
            provider=provider,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_json=response_json,
            api_key=api_key,
            base_url=LLM_PROVIDER_REGISTRY[provider].base_url,
        )

        parser = {
            "deepseek": _parse_openai_response,
            "qwen": _parse_openai_response,
            "kimi": _parse_openai_response,
        }[provider]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = await self._http_post(url=url, headers=headers, body=body)
                response_data = json.loads(raw_text)
                content_text = parser(response_data)
                completion_est = estimate_text_tokens(content_text)
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(
                    provider=provider,
                    response_data=response_data,
                    prompt_fallback=prompt_tokens_est,
                    completion_fallback=completion_est,
                )
                latency_ms = int((time.monotonic() - t0) * 1000)
                record_usage(
                    LLMUsageRecord(
                        stage=stage_name,
                        branch_or_agent_name=branch_or_agent_name,
                        provider=provider,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency_ms=latency_ms,
                        success=True,
                        fallback=False,
                        estimated_cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
                    )
                )
                return content_text
            except (_AiohttpClientError, asyncio.TimeoutError, json.JSONDecodeError, LLMCallError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * attempt)
                continue

        latency_ms = int((time.monotonic() - t0) * 1000)
        failure_reason = _normalize_failure_reason(last_exc)
        record_fallback_event(
            stage=stage_name,
            branch_or_agent_name=branch_or_agent_name,
            model=model,
            reason=failure_reason,
            provider=provider,
            prompt_tokens=prompt_tokens_est,
            latency_ms=latency_ms,
        )
        raise LLMCallError(f"All {self.max_retries} attempts failed for model={model}: {failure_reason}")

    async def _http_post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> str:
        if aiohttp is None:
            raise LLMCallError("aiohttp is required for LLM calls. Install with: pip install aiohttp")
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=body) as response:
                text = await response.text()
                if response.status != 200:
                    raise LLMCallError(f"HTTP {response.status}: {text[:500]}")
                return text

    @staticmethod
    def _parse_json_content(text: str) -> dict[str, Any]:
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


def _run_sync(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


def complete_json_sync(
    *,
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int = 1024,
    timeout: float = 30.0,
    max_retries: int = 2,
    stage: str = "",
    actor_name: str = "",
) -> dict[str, Any]:
    client = LLMClient(timeout=timeout, max_retries=max_retries)
    return _run_sync(
        client.complete(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            response_json=True,
            stage=stage,
            actor_name=actor_name,
        )
    )


def complete_text_sync(
    *,
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int = 1024,
    timeout: float = 30.0,
    max_retries: int = 2,
    stage: str = "",
    actor_name: str = "",
) -> str:
    client = LLMClient(timeout=timeout, max_retries=max_retries)
    return _run_sync(
        client.complete_text(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            response_json=False,
            stage=stage,
            actor_name=actor_name,
        )
    )


__all__ = [
    "LLMCallError",
    "LLMClient",
    "LLMProviderSpec",
    "LLMModelPricing",
    "LLMUsageSessionHandle",
    "LLM_PROVIDER_REGISTRY",
    "LLM_MODEL_PRICING_REGISTRY",
    "LLM_STAGE_NAMES",
    "LLM_PROVIDER_ENV_KEYS",
    "USAGE_LOG_PATH",
    "build_usage_summary",
    "complete_json_sync",
    "complete_text_sync",
    "current_usage_session_id",
    "detect_provider",
    "end_usage_session",
    "estimate_cost_usd",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "get_usage_records",
    "has_any_provider",
    "has_provider_for_model",
    "record_fallback_event",
    "record_usage",
    "render_usage_markdown",
    "resolve_default_model",
    "snapshot_usage",
    "start_usage_session",
    "update_usage_markdown",
    "usage_session",
]
