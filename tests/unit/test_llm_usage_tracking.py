from __future__ import annotations

import asyncio
import json

import pytest

from quant_investor.llm_gateway import LLMCallError, LLMClient, end_usage_session, snapshot_usage, start_usage_session


def test_llm_usage_success_records_tokens_and_cost(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

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
                model="gpt-5.4-mini",
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
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    handle = start_usage_session(label="usage-missing-provider")
    try:
        with pytest.raises(LLMCallError):
            asyncio.run(
                LLMClient(timeout=1.0, max_retries=1).complete(
                    messages=[{"role": "user", "content": "return json"}],
                    model="gpt-5.4-mini",
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
    assert summary.fallback_count == 1
    assert records[0].fallback is True
    assert "Missing env var OPENAI_API_KEY" in records[0].metadata["reason"]


def test_llm_usage_records_timeout_fallback(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def _timeout_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        raise asyncio.TimeoutError("timed out")

    monkeypatch.setattr(LLMClient, "_http_post", _timeout_http_post)

    handle = start_usage_session(label="usage-timeout")
    try:
        with pytest.raises(LLMCallError):
            asyncio.run(
                LLMClient(timeout=0.1, max_retries=1).complete(
                    messages=[{"role": "user", "content": "return json"}],
                    model="gpt-5.4-mini",
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
    assert summary.fallback_count == 1
    assert records[0].stage == "review_master_agent"
    assert records[0].fallback is True
    assert records[0].metadata["reason"] == "timeout"
