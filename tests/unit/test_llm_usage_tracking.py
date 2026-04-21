from __future__ import annotations

import asyncio
import json

import pytest

from quant_investor.llm_gateway import (
    LLMCallError,
    LLMClient,
    end_usage_session,
    reset_provider_runtime_state,
    snapshot_usage,
    start_usage_session,
)


def test_llm_usage_success_records_tokens_and_cost(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KIMI_API_KEY", "test-key")

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        return json.dumps(
            {
                "choices": [{"message": {"content": "{\"ok\": true}"}}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                },
            }
        )

    monkeypatch.setattr(LLMClient, "_http_post", _fake_http_post)

    handle = start_usage_session(label="usage-success")
    try:
        payload = asyncio.run(
            LLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="moonshot-v1-8k",
                max_tokens=128,
                stage="review_branch_subagent",
                actor_name="quant",
            )
        )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert payload == {"ok": True}
    assert summary.call_count == 1
    assert summary.success_count == 1
    assert summary.fallback_count == 0
    assert summary.total_tokens == 150
    assert summary.estimated_cost_usd > 0
    assert records[0].stage == "review_branch_subagent"
    assert records[0].branch_or_agent_name == "quant"
    assert (tmp_path / "data" / "llm_usage.jsonl").exists()


def test_llm_usage_records_provider_missing_fallback(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    handle = start_usage_session(label="usage-missing-provider")
    try:
        with pytest.raises(LLMCallError):
            asyncio.run(
                LLMClient(timeout=1.0, max_retries=1).complete(
                    messages=[{"role": "user", "content": "return json"}],
                    model="moonshot-v1-8k",
                    max_tokens=64,
                    stage="review_branch_subagent",
                    actor_name="macro",
                )
            )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert summary.call_count == 1
    assert summary.success_count == 0
    assert summary.failed_count == 1
    assert summary.fallback_count == 1
    assert [record.model for record in records] == ["moonshot-v1-8k"]
    assert all(record.fallback is True for record in records)
    assert "Missing env var KIMI_API_KEY" in records[0].metadata["reason"]


def test_llm_usage_records_timeout_fallback(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KIMI_API_KEY", "test-key")

    async def _timeout_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        raise asyncio.TimeoutError("timed out")

    monkeypatch.setattr(LLMClient, "_http_post", _timeout_http_post)

    handle = start_usage_session(label="usage-timeout")
    try:
        with pytest.raises(LLMCallError):
            asyncio.run(
                LLMClient(timeout=0.1, max_retries=1).complete(
                    messages=[{"role": "user", "content": "return json"}],
                    model="moonshot-v1-8k",
                    max_tokens=64,
                    stage="review_master_agent",
                    actor_name="ICCoordinator",
                )
            )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert summary.call_count == 1
    assert summary.success_count == 0
    assert summary.failed_count == 1
    assert summary.fallback_count == 1
    assert records[0].stage == "review_master_agent"
    assert all(record.fallback is True for record in records)
    assert records[0].metadata["reason"] == "timeout"


def test_gateway_client_timeout_does_not_retry_same_model(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    attempts = {"count": 0}

    async def _timeout_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        attempts["count"] += 1
        raise asyncio.TimeoutError()

    monkeypatch.setattr(LLMClient, "_http_post", _timeout_http_post)

    handle = start_usage_session(label="usage-timeout-no-retry")
    try:
        with pytest.raises(LLMCallError, match="timeout"):
            asyncio.run(
                LLMClient(timeout=0.1, max_retries=2).complete(
                    messages=[{"role": "user", "content": "return json"}],
                    model="deepseek-reasoner",
                    max_tokens=64,
                    stage="review_master_agent",
                    actor_name="ICCoordinator",
                )
            )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert attempts["count"] == 1
    assert summary.call_count == 1
    assert summary.failed_count == 1
    assert records[0].metadata["reason"] == "timeout"


def test_gateway_client_uses_fallback_model_after_primary_json_failure(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key")

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        if body["model"] == "qwen3.5-plus":
            return json.dumps({"choices": [{"message": {"content": "{\"broken\": "}}]})
        return json.dumps(
            {
                "choices": [{"message": {"content": "{\"ok\": true, \"model\": \"moonshot\"}"}}],
                "usage": {
                    "prompt_tokens": 90,
                    "completion_tokens": 20,
                    "total_tokens": 110,
                },
            }
        )

    monkeypatch.setattr(LLMClient, "_http_post", _fake_http_post)

    handle = start_usage_session(label="usage-fallback-model")
    try:
        payload = asyncio.run(
            LLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="qwen3.5-plus",
                fallback_model="moonshot-v1-128k",
                max_tokens=64,
                stage="review_branch_overlay",
                actor_name="000001.SZ:kline",
            )
        )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert payload == {"ok": True, "model": "moonshot"}
    assert summary.call_count == 2
    assert summary.success_count == 1
    assert summary.failed_count == 1
    assert summary.fallback_count == 1
    assert records[0].model == "qwen3.5-plus"
    assert records[0].fallback is True
    assert records[-1].model == "moonshot-v1-128k"


def test_gateway_client_uses_only_requested_and_fallback_models(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        if body["model"] == "qwen3.5-plus":
            return json.dumps({"choices": [{"message": {"content": "{\"broken\": "}}]})
        return json.dumps(
            {
                "choices": [{"message": {"content": "{\"ok\": true, \"model\": \"moonshot\"}"}}],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 12,
                    "total_tokens": 62,
                },
            }
        )

    monkeypatch.setattr(LLMClient, "_http_post", _fake_http_post)

    handle = start_usage_session(label="usage-ordered-models")
    try:
        payload = asyncio.run(
            LLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="qwen3.5-plus",
                fallback_model="moonshot-v1-128k",
                candidate_models=["qwen3.5-plus", "moonshot-v1-128k", "deepseek-chat"],
                max_tokens=64,
                stage="review_branch_overlay",
                actor_name="000001.SZ:kline",
            )
        )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert payload == {"ok": True, "model": "moonshot"}
    assert summary.call_count == 2
    assert summary.success_count == 1
    assert summary.failed_count == 1
    assert [record.model for record in records] == [
        "qwen3.5-plus",
        "moonshot-v1-128k",
    ]


def test_gateway_client_uses_third_candidate_after_primary_and_fallback_fail(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        model = body["model"]
        if model == "moonshot-v1-128k":
            return json.dumps(
                {
                    "choices": [{"message": {"content": "{\"ok\": true, \"model\": \"moonshot\"}"}}],
                    "usage": {
                        "prompt_tokens": 44,
                        "completion_tokens": 11,
                        "total_tokens": 55,
                    },
                }
            )
        return json.dumps({"choices": [{"message": {"content": "{\"broken\": "}}]})

    monkeypatch.setattr(LLMClient, "_http_post", _fake_http_post)

    handle = start_usage_session(label="usage-third-candidate")
    try:
        payload = asyncio.run(
            LLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="qwen3.5-flash",
                fallback_model="deepseek-chat",
                candidate_models=["qwen3.5-flash", "deepseek-chat", "moonshot-v1-128k", "qwen3.5-plus"],
                max_tokens=64,
                stage="review_branch_overlay",
                actor_name="000001.SZ:kline",
            )
        )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert payload == {"ok": True, "model": "moonshot"}
    assert summary.call_count == 3
    assert summary.success_count == 1
    assert summary.failed_count == 2
    assert summary.fallback_count == 2
    assert [record.model for record in records] == [
        "qwen3.5-flash",
        "deepseek-chat",
        "moonshot-v1-128k",
    ]


def test_gateway_client_retries_same_model_on_rate_limit_before_fallback(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    attempts = {"moonshot-v1-128k": 0}
    sleeps: list[float] = []

    async def _fake_sleep(delay: float):  # type: ignore[no-untyped-def]
        sleeps.append(delay)

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        attempts[body["model"]] = attempts.get(body["model"], 0) + 1
        if attempts[body["model"]] == 1:
            raise LLMCallError(
                "HTTP 429: max organization concurrency: 3, please try again after 1 seconds"
            )
        return json.dumps(
            {
                "choices": [{"message": {"content": "{\"ok\": true, \"model\": \"moonshot\"}"}}],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 10,
                    "total_tokens": 50,
                },
            }
        )

    monkeypatch.setattr("quant_investor.llm_gateway.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr(LLMClient, "_http_post", _fake_http_post)

    handle = start_usage_session(label="usage-rate-limit-retry")
    try:
        payload = asyncio.run(
            LLMClient(timeout=1.0, max_retries=2).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="moonshot-v1-128k",
                fallback_model="deepseek-chat",
                max_tokens=64,
                stage="review_master_agent",
                actor_name="ICCoordinator",
            )
        )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert payload == {"ok": True, "model": "moonshot"}
    assert attempts["moonshot-v1-128k"] == 2
    assert sleeps == [1.0]
    assert summary.call_count == 1
    assert records[0].metadata["rate_limit_retry_count"] == 1
    assert records[0].metadata["retry_after_seconds"] == 1.0


def test_gateway_client_marks_provider_unhealthy_after_arrearage(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key")

    attempts: dict[str, int] = {"qwen3.5-plus": 0, "moonshot-v1-128k": 0}

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        model = body["model"]
        attempts[model] = attempts.get(model, 0) + 1
        if model == "qwen3.5-plus":
            raise LLMCallError("HTTP 400: Arrearage")
        return json.dumps(
            {
                "choices": [{"message": {"content": "{\"ok\": true, \"model\": \"moonshot\"}"}}],
                "usage": {
                    "prompt_tokens": 35,
                    "completion_tokens": 8,
                    "total_tokens": 43,
                },
            }
        )

    monkeypatch.setattr(LLMClient, "_http_post", _fake_http_post)

    handle = start_usage_session(label="usage-provider-cooldown")
    try:
        first_payload = asyncio.run(
            LLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="qwen3.5-plus",
                fallback_model="moonshot-v1-128k",
                max_tokens=64,
                stage="review_branch_overlay",
                actor_name="000001.SZ:kline",
            )
        )
        second_payload = asyncio.run(
            LLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="qwen3.5-plus",
                fallback_model="moonshot-v1-128k",
                max_tokens=64,
                stage="review_branch_overlay",
                actor_name="000001.SZ:kline",
            )
        )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert first_payload == {"ok": True, "model": "moonshot"}
    assert second_payload == {"ok": True, "model": "moonshot"}
    assert attempts["qwen3.5-plus"] == 1
    assert attempts["moonshot-v1-128k"] == 2
    assert summary.call_count == 4
    assert records[0].metadata["reason"] == "HTTP 400: Arrearage"
    assert records[2].metadata["cooldown_hit"] is True
