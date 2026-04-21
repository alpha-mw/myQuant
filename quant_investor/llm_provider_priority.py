from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_ORDERED_MODELS: tuple[str, ...] = (
    "deepseek-chat",
    "moonshot-v1-128k",
    "qwen3.5-plus",
)
DEFAULT_BRANCH_MODELS: tuple[str, str] = (
    "deepseek-chat",
    "moonshot-v1-128k",
)
DEFAULT_MASTER_MODELS: tuple[str, str] = (
    "moonshot-v1-128k",
    "deepseek-reasoner",
)

MODEL_ALIASES: dict[str, str] = {
    "qwen-plus": "qwen3.5-plus",
    "qwen-3.6": "qwen3.5-plus",
    "qwen3.6-plus": "qwen3.5-plus",
    "qwen-3.5": "qwen3.5-plus",
    "qwen-flash": "qwen3.5-flash",
    "deepseek reasoning": "deepseek-reasoner",
}

MODEL_ENV_KEYS: dict[str, str] = {
    "qwen3.5-plus": "DASHSCOPE_API_KEY",
    "qwen3.5-flash": "DASHSCOPE_API_KEY",
    "qwen3.6-plus": "DASHSCOPE_API_KEY",
    "qwen-plus": "DASHSCOPE_API_KEY",
    "qwen-turbo": "DASHSCOPE_API_KEY",
    "moonshot-v1-8k": "KIMI_API_KEY",
    "moonshot-v1-32k": "KIMI_API_KEY",
    "moonshot-v1-128k": "KIMI_API_KEY",
    "deepseek-chat": "DEEPSEEK_API_KEY",
    "deepseek-reasoner": "DEEPSEEK_API_KEY",
}

PRIORITY_CACHE_PATH = Path("data") / "llm_provider_priority.json"
BENCHMARK_RESULTS_DIR = Path("results") / "diagnostics" / "llm_provider_benchmarks"


@dataclass(frozen=True)
class ReviewModelPriority:
    primary_model: str
    fallback_model: str
    ordered_models: list[str]
    source: str = "default"
    benchmark_timestamp_utc: str = ""
    benchmark_path: str = ""


@dataclass(frozen=True)
class RoleModelConfig:
    role: str
    primary_model: str
    fallback_model: str
    candidate_models: list[str]
    source: str = "default"


def coerce_review_model_priority(
    preferred_models: Sequence[str] | None = None,
    *,
    legacy_models: Sequence[str] | None = None,
) -> list[str]:
    explicit_order = _dedupe_models(list(preferred_models or []))
    if explicit_order:
        return explicit_order

    legacy_order = _dedupe_models(list(legacy_models or []))
    if legacy_order:
        return legacy_order

    return list(DEFAULT_ORDERED_MODELS)


def default_role_model_chain(role: str) -> tuple[str, str]:
    normalized_role = str(role or "").strip().lower()
    if normalized_role == "master":
        return DEFAULT_MASTER_MODELS
    return DEFAULT_BRANCH_MODELS


def coerce_role_model_config(
    *,
    role: str,
    primary_model: str = "",
    fallback_model: str = "",
    preferred_models: Sequence[str] | None = None,
) -> RoleModelConfig:
    default_primary, default_fallback = default_role_model_chain(role)
    explicit_primary = normalize_model_name(primary_model)
    explicit_fallback = normalize_model_name(fallback_model)

    legacy_candidates = _dedupe_models(list(preferred_models or []))

    if explicit_primary or explicit_fallback:
        candidate_models = _dedupe_models(
            [
                explicit_primary or default_primary,
                explicit_fallback or default_fallback,
                *legacy_candidates,
            ]
        )
        source = "explicit_role_with_priority" if legacy_candidates else "explicit_role"
    else:
        candidate_models = list(
            legacy_candidates
            or _dedupe_models([default_primary, default_fallback, *DEFAULT_ORDERED_MODELS])
        )
        source = "legacy_priority" if legacy_candidates else "default"

    primary = candidate_models[0] if candidate_models else ""
    fallback = candidate_models[1] if len(candidate_models) > 1 else ""
    return RoleModelConfig(
        role=str(role or ""),
        primary_model=primary,
        fallback_model=fallback,
        candidate_models=list(candidate_models),
        source=source,
    )


def resolve_runtime_role_models(
    *,
    review_model_priority: Sequence[str] | None = None,
    agent_model: str = "",
    agent_fallback_model: str = "",
    master_model: str = "",
    master_fallback_model: str = "",
) -> tuple[RoleModelConfig, RoleModelConfig]:
    legacy_priority = _dedupe_models(list(review_model_priority or []))
    branch = coerce_role_model_config(
        role="branch",
        primary_model=agent_model,
        fallback_model=agent_fallback_model,
        preferred_models=legacy_priority,
    )
    master = coerce_role_model_config(
        role="master",
        primary_model=master_model,
        fallback_model=master_fallback_model,
        preferred_models=legacy_priority,
    )
    return branch, master


def build_candidate_model_chain(
    requested_model: str = "",
    fallback_model: str = "",
    candidate_models: Sequence[str] | None = None,
) -> list[str]:
    requested = normalize_model_name(requested_model)
    fallback = normalize_model_name(fallback_model)
    ordered_candidates = _dedupe_models(list(candidate_models or []))
    if not (requested or fallback):
        return ordered_candidates

    anchor = ""
    if requested and requested in ordered_candidates:
        anchor = requested
    elif fallback and fallback in ordered_candidates:
        anchor = fallback
    tail = ordered_candidates[ordered_candidates.index(anchor) :] if anchor else ordered_candidates
    return _dedupe_models([requested, fallback] + tail)


def _dedupe_models(models: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for model in models:
        text = normalize_model_name(model)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def normalize_model_name(model: str) -> str:
    text = str(model or "").strip()
    if not text:
        return ""
    return MODEL_ALIASES.get(text, text)


def has_credentials_for_model(model: str) -> bool:
    env_key = MODEL_ENV_KEYS.get(normalize_model_name(model), "")
    return bool(env_key and os.getenv(env_key))


def load_ranked_models(cache_path: Path | None = None) -> tuple[list[str], str, str, str]:
    path = cache_path or PRIORITY_CACHE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return list(DEFAULT_ORDERED_MODELS), "default", "", ""
    if not isinstance(payload, dict):
        return list(DEFAULT_ORDERED_MODELS), "default", "", ""
    ordered = _dedupe_models(payload.get("ordered_models", []))
    if not ordered:
        ordered = list(DEFAULT_ORDERED_MODELS)
    return (
        ordered,
        "cache",
        str(payload.get("generated_at_utc", "") or ""),
        str(payload.get("report_path", "") or ""),
    )


def rank_benchmark_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in results:
        model = str(item.get("model", "")).strip()
        if not model:
            continue
        normalized.append(
            {
                **dict(item),
                "model": model,
                "success_rate": float(item.get("success_rate", 0.0) or 0.0),
                "burst_success_rate": float(item.get("burst_success_rate", 0.0) or 0.0),
                "burst_mean_latency_ms": float(item.get("burst_mean_latency_ms", 0.0) or 0.0),
                "mean_latency_ms": float(item.get("mean_latency_ms", 0.0) or 0.0),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            -float(item.get("success_rate", 0.0) or 0.0),
            -float(item.get("burst_success_rate", 0.0) or 0.0),
            float(item.get("burst_mean_latency_ms", 10**12) or 10**12),
            float(item.get("mean_latency_ms", 10**12) or 10**12),
            str(item.get("model", "")),
        ),
    )


def resolve_review_model_priority(
    *,
    ranked_models: Sequence[str] | None = None,
    preferred_models: Sequence[str] | None = None,
    preferred_primary: str = "",
    preferred_fallback: str = "",
    available_only: bool = True,
) -> ReviewModelPriority:
    if ranked_models is not None:
        ordered = _dedupe_models(ranked_models)
        source = "explicit"
        benchmark_timestamp_utc = ""
        benchmark_path = ""
    else:
        ordered = list(DEFAULT_ORDERED_MODELS)
        source = "default"
        benchmark_timestamp_utc = ""
        benchmark_path = ""
    if not ordered:
        ordered = list(DEFAULT_ORDERED_MODELS)

    preferred_order = _dedupe_models(
        list(preferred_models or []) + [preferred_primary, preferred_fallback] + ordered
    )
    available_models = [model for model in preferred_order if has_credentials_for_model(model)]
    chosen_pool = available_models if available_only and available_models else preferred_order

    primary = chosen_pool[0] if chosen_pool else ""
    fallback = chosen_pool[1] if len(chosen_pool) > 1 else ""

    return ReviewModelPriority(
        primary_model=primary,
        fallback_model=fallback,
        ordered_models=preferred_order,
        source=source,
        benchmark_timestamp_utc=benchmark_timestamp_utc,
        benchmark_path=benchmark_path,
    )


def benchmark_provider_models(
    *,
    models: Sequence[str] | None = None,
    sequential_rounds: int = 3,
    burst_concurrency: int = 3,
    timeout: float = 15.0,
    max_retries: int = 1,
) -> dict[str, Any]:
    from quant_investor.llm_gateway import complete_text_sync

    benchmark_models = _dedupe_models(models or DEFAULT_ORDERED_MODELS)
    prompt = "Reply with exactly one lowercase word: pong"

    def _call_once(model: str, actor_name: str) -> tuple[bool, float, str]:
        started = time.monotonic()
        try:
            text = complete_text_sync(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=8,
                timeout=timeout,
                max_retries=max_retries,
                stage="diagnostic_provider_benchmark",
                actor_name=actor_name,
            )
            return True, (time.monotonic() - started) * 1000.0, str(text).strip()
        except Exception as exc:
            return False, (time.monotonic() - started) * 1000.0, str(exc)

    results: list[dict[str, Any]] = []
    for model in benchmark_models:
        available = has_credentials_for_model(model)
        sequential_latencies: list[float] = []
        sequential_failures: list[str] = []
        sequential_successes = 0
        if available:
            for idx in range(max(int(sequential_rounds), 0)):
                ok, latency_ms, detail = _call_once(model, actor_name=f"seq:{model}:{idx}")
                if ok:
                    sequential_successes += 1
                    sequential_latencies.append(latency_ms)
                else:
                    sequential_failures.append(detail[:240])

        burst_latencies: list[float] = []
        burst_failures: list[str] = []
        burst_successes = 0
        if available and burst_concurrency > 0:
            with ThreadPoolExecutor(max_workers=int(burst_concurrency)) as pool:
                futures = [
                    pool.submit(_call_once, model, f"burst:{model}:{idx}")
                    for idx in range(int(burst_concurrency))
                ]
                for future in as_completed(futures):
                    ok, latency_ms, detail = future.result()
                    if ok:
                        burst_successes += 1
                        burst_latencies.append(latency_ms)
                    else:
                        burst_failures.append(detail[:240])

        results.append(
            {
                "model": model,
                "available": available,
                "sequential_rounds": int(sequential_rounds),
                "sequential_successes": sequential_successes,
                "sequential_failures": sequential_failures,
                "success_rate": round(sequential_successes / max(int(sequential_rounds), 1), 4),
                "mean_latency_ms": round(sum(sequential_latencies) / len(sequential_latencies), 2) if sequential_latencies else 0.0,
                "burst_concurrency": int(burst_concurrency),
                "burst_successes": burst_successes,
                "burst_failures": burst_failures,
                "burst_success_rate": round(burst_successes / max(int(burst_concurrency), 1), 4),
                "burst_mean_latency_ms": round(sum(burst_latencies) / len(burst_latencies), 2) if burst_latencies else 0.0,
            }
        )

    ranked = rank_benchmark_results(results)
    benchmark_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ordered_models": [item["model"] for item in ranked],
        "results": ranked,
        "benchmark_id": benchmark_id,
    }


def persist_benchmark_results(payload: dict[str, Any]) -> dict[str, str]:
    benchmark_id = str(payload.get("benchmark_id", "") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRIORITY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = BENCHMARK_RESULTS_DIR / f"llm_provider_benchmark_{benchmark_id}.json"
    cache_payload = {
        "generated_at_utc": payload.get("generated_at_utc", ""),
        "ordered_models": list(payload.get("ordered_models", []) or []),
        "report_path": str(report_path),
        "results": list(payload.get("results", []) or []),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    PRIORITY_CACHE_PATH.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report_path": str(report_path), "cache_path": str(PRIORITY_CACHE_PATH)}
