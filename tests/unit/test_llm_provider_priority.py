from __future__ import annotations

from quant_investor.llm_provider_priority import (
    build_candidate_model_chain,
    coerce_review_model_priority,
    normalize_model_name,
    rank_benchmark_results,
    resolve_review_model_priority,
    resolve_runtime_role_models,
)


def test_normalize_model_name_maps_legacy_qwen_alias_to_qwen35_plus():
    assert normalize_model_name("qwen-plus") == "qwen3.5-plus"
    assert normalize_model_name(" qwen3.6-plus ") == "qwen3.5-plus"
    assert normalize_model_name("moonshot-v1-128k") == "moonshot-v1-128k"


def test_coerce_review_model_priority_defaults_to_execution_chain():
    assert coerce_review_model_priority([]) == [
        "deepseek-chat",
        "moonshot-v1-128k",
        "qwen3.5-plus",
    ]


def test_rank_benchmark_results_prefers_success_then_latency():
    ranked = rank_benchmark_results(
        [
            {
                "model": "moonshot-v1-128k",
                "success_rate": 1.0,
                "burst_success_rate": 0.8,
                "mean_latency_ms": 1800.0,
            },
            {
                "model": "qwen3.5-plus",
                "success_rate": 1.0,
                "burst_success_rate": 1.0,
                "mean_latency_ms": 950.0,
            },
            {
                "model": "deepseek-chat",
                "success_rate": 0.0,
                "burst_success_rate": 0.0,
                "mean_latency_ms": 0.0,
            },
        ]
    )

    assert [item["model"] for item in ranked] == ["qwen3.5-plus", "moonshot-v1-128k", "deepseek-chat"]


def test_resolve_review_model_priority_uses_ranked_models():
    resolved = resolve_review_model_priority(
        ranked_models=["qwen3.5-flash", "deepseek-chat", "moonshot-v1-128k", "qwen3.5-plus"],
    )

    assert resolved.primary_model == "qwen3.5-flash"
    assert resolved.fallback_model == "deepseek-chat"
    assert resolved.ordered_models == [
        "qwen3.5-flash",
        "deepseek-chat",
        "moonshot-v1-128k",
        "qwen3.5-plus",
    ]


def test_resolve_review_model_priority_respects_explicit_ordered_override():
    resolved = resolve_review_model_priority(
        ranked_models=["qwen3.5-flash", "deepseek-chat", "moonshot-v1-128k", "qwen3.5-plus"],
        preferred_models=["deepseek-chat", "qwen-plus"],
        available_only=False,
    )

    assert resolved.primary_model == "deepseek-chat"
    assert resolved.fallback_model == "qwen3.5-plus"
    assert resolved.ordered_models == [
        "deepseek-chat",
        "qwen3.5-plus",
        "qwen3.5-flash",
        "moonshot-v1-128k",
    ]


def test_resolve_review_model_priority_ignores_cached_benchmark_order_for_runtime(monkeypatch):
    monkeypatch.setattr(
        "quant_investor.llm_provider_priority.load_ranked_models",
        lambda *args, **kwargs: (
            ["moonshot-v1-8k", "qwen3.6-plus", "deepseek-reasoner"],
            "cache",
            "cached-ts",
            "cached-report",
        ),
    )

    resolved = resolve_review_model_priority()

    assert resolved.primary_model == "deepseek-chat"
    assert resolved.fallback_model == "moonshot-v1-128k"
    assert resolved.ordered_models == [
        "deepseek-chat",
        "moonshot-v1-128k",
        "qwen3.5-plus",
    ]
    assert resolved.source == "default"
    assert resolved.benchmark_timestamp_utc == ""
    assert resolved.benchmark_path == ""


def test_resolve_runtime_role_models_defaults_follow_daily_config_roles():
    branch, master = resolve_runtime_role_models(
        review_model_priority=[],
    )

    assert branch.primary_model == "deepseek-chat"
    assert branch.fallback_model == "moonshot-v1-128k"
    assert branch.candidate_models == [
        "deepseek-chat",
        "moonshot-v1-128k",
        "qwen3.5-plus",
    ]
    assert master.primary_model == "moonshot-v1-128k"
    assert master.fallback_model == "deepseek-reasoner"
    assert master.candidate_models == [
        "moonshot-v1-128k",
        "deepseek-reasoner",
        "deepseek-chat",
        "qwen3.5-plus",
    ]


def test_build_candidate_model_chain_keeps_cold_standby_models_after_resolved_fallback():
    resolved = build_candidate_model_chain(
        requested_model="deepseek-chat",
        fallback_model="deepseek-chat",
        candidate_models=["qwen3.5-flash", "deepseek-chat", "moonshot-v1-128k", "qwen3.5-plus"],
    )

    assert resolved == [
        "deepseek-chat",
        "moonshot-v1-128k",
        "qwen3.5-plus",
    ]


def test_rank_benchmark_results_prefers_burst_latency_when_success_is_tied():
    ranked = rank_benchmark_results(
        [
            {
                "model": "moonshot-v1-128k",
                "success_rate": 1.0,
                "burst_success_rate": 1.0,
                "burst_mean_latency_ms": 2100.0,
                "mean_latency_ms": 2200.0,
            },
            {
                "model": "qwen3.5-plus",
                "success_rate": 1.0,
                "burst_success_rate": 1.0,
                "burst_mean_latency_ms": 2000.0,
                "mean_latency_ms": 2400.0,
            },
        ]
    )

    assert [item["model"] for item in ranked] == ["qwen3.5-plus", "moonshot-v1-128k"]
