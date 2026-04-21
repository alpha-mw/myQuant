from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from quant_investor.agents.agent_contracts import BranchAgentInput, MasterAgentInput, RiskAgentInput
from quant_investor.agents.llm_client import LLMCallError, LLMClient as LegacyLLMClient
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.subagent import BranchSubAgent, RiskSubAgent
from quant_investor.llm_gateway import (
    LLMClient as GatewayLLMClient,
    end_usage_session,
    reset_provider_runtime_state,
    snapshot_usage,
    start_usage_session,
)


def _openai_response() -> str:
    return json.dumps(
        {
            "choices": [{"message": {"content": "{\"ok\": true}"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )


@pytest.mark.parametrize(
    ("client_cls", "model", "api_key_env", "expected_token_field", "expected_reasoning_effort"),
    [
        (LegacyLLMClient, "moonshot-v1-128k", "KIMI_API_KEY", "max_tokens", False),
        (LegacyLLMClient, "moonshot-v1-8k", "KIMI_API_KEY", "max_tokens", False),
        (GatewayLLMClient, "moonshot-v1-128k", "KIMI_API_KEY", "max_tokens", False),
        (GatewayLLMClient, "deepseek-chat", "DEEPSEEK_API_KEY", "max_tokens", False),
        (GatewayLLMClient, "deepseek-reasoner", "DEEPSEEK_API_KEY", "max_tokens", True),
    ],
)
def test_completion_request_policy_splits_telemetry_and_selects_token_field(
    monkeypatch,
    tmp_path,
    client_cls,
    model,
    api_key_env,
    expected_token_field,
    expected_reasoning_effort,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(api_key_env, "test-key")

    captured: dict[str, object] = {}

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        captured.update(body)
        return _openai_response()

    monkeypatch.setattr(client_cls, "_http_post", _fake_http_post)

    client = client_cls(timeout=1.0, max_retries=1)
    payload = asyncio.run(
        client.complete(
            messages=[{"role": "user", "content": "return json"}],
            model=model,
            max_tokens=128,
            response_json=True,
            stage="review_branch_overlay",
            actor_name="IC:000001.SZ",
            reasoning_effort="high",
        )
    )

    assert payload == {"ok": True}
    assert expected_token_field in captured
    assert captured[expected_token_field] == 128
    if expected_token_field == "max_completion_tokens":
        assert "max_tokens" not in captured
    else:
        assert "max_completion_tokens" not in captured
    assert "stage" not in captured
    assert "actor_name" not in captured
    if expected_reasoning_effort:
        assert captured["reasoning_effort"] == "high"
    else:
        assert "reasoning_effort" not in captured


def test_legacy_client_records_telemetry_labels(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KIMI_API_KEY", "test-key")

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        return _openai_response()

    monkeypatch.setattr(LegacyLLMClient, "_http_post", _fake_http_post)

    payload = asyncio.run(
        LegacyLLMClient(timeout=1.0, max_retries=1).complete(
            messages=[{"role": "user", "content": "return json"}],
            model="moonshot-v1-128k",
            max_tokens=128,
            response_json=True,
            stage="review_branch_subagent",
            actor_name="quant",
        )
    )

    assert payload == {"ok": True}
    log_path = Path(tmp_path) / "data" / "llm_usage.jsonl"
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["stage"] == "review_branch_subagent"
    assert record["branch_or_agent_name"] == "quant"


def test_legacy_client_uses_fallback_model_after_primary_json_failure(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key")

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        if body["model"] == "qwen3.5-plus":
            return "{\"choices\": [{\"message\": {\"content\": \"{\"}}]}"
        return _openai_response()

    monkeypatch.setattr(LegacyLLMClient, "_http_post", _fake_http_post)

    handle = start_usage_session(label="legacy-fallback-model")
    try:
        payload = asyncio.run(
            LegacyLLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="qwen3.5-plus",
                fallback_model="moonshot-v1-128k",
                max_tokens=128,
                response_json=True,
                stage="review_master_agent",
                actor_name="ICCoordinator",
            )
        )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert payload == {"ok": True}
    assert summary.call_count == 2
    assert summary.success_count == 1
    assert summary.failed_count == 1
    assert summary.fallback_count == 1
    assert records[0].model == "qwen3.5-plus"
    assert records[-1].model == "moonshot-v1-128k"


def test_legacy_client_uses_third_candidate_after_primary_and_fallback_fail(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key")

    async def _fake_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        if body["model"] == "moonshot-v1-128k":
            return _openai_response()
        return "{\"choices\": [{\"message\": {\"content\": \"{\"}}]}"

    monkeypatch.setattr(LegacyLLMClient, "_http_post", _fake_http_post)

    handle = start_usage_session(label="legacy-third-candidate")
    try:
        payload = asyncio.run(
            LegacyLLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="qwen3.5-flash",
                fallback_model="deepseek-chat",
                candidate_models=["qwen3.5-flash", "deepseek-chat", "moonshot-v1-128k", "qwen3.5-plus"],
                max_tokens=128,
                response_json=True,
                stage="review_master_agent",
                actor_name="ICCoordinator",
            )
        )
        records, summary = snapshot_usage(handle.session_id)
    finally:
        end_usage_session(handle)

    assert payload == {"ok": True}
    assert summary.call_count == 3
    assert summary.success_count == 1
    assert summary.failed_count == 2
    assert summary.fallback_count == 2
    assert [record.model for record in records] == [
        "qwen3.5-flash",
        "deepseek-chat",
        "moonshot-v1-128k",
    ]


def test_legacy_client_candidate_models_none_does_not_raise_type_error(monkeypatch, tmp_path):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(LLMCallError, match="Missing env var DASHSCOPE_API_KEY"):
        asyncio.run(
            LegacyLLMClient(timeout=1.0, max_retries=1).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="qwen3.5-plus",
                fallback_model="",
                candidate_models=None,
                max_tokens=64,
                response_json=True,
                stage="review_branch_overlay",
                actor_name="IC:000001.SZ",
            )
        )


def test_legacy_client_timeout_logs_reason_and_skips_same_model_retry(
    monkeypatch,
    tmp_path,
):
    reset_provider_runtime_state()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    attempts = {"count": 0}

    async def _timeout_http_post(self, url, headers, body):  # type: ignore[no-untyped-def]
        attempts["count"] += 1
        raise asyncio.TimeoutError()

    monkeypatch.setattr(LegacyLLMClient, "_http_post", _timeout_http_post)

    with pytest.raises(LLMCallError) as exc_info:
        asyncio.run(
            LegacyLLMClient(timeout=0.1, max_retries=2).complete(
                messages=[{"role": "user", "content": "return json"}],
                model="deepseek-reasoner",
                max_tokens=64,
                response_json=True,
                stage="review_master_agent",
                actor_name="ICCoordinator",
            )
        )

    assert attempts["count"] == 1
    assert "timeout" in str(exc_info.value)


def test_branch_subagent_forwards_stage_and_actor_name(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "conviction": "buy",
            "conviction_score": 0.2,
            "confidence": 0.7,
            "key_insights": ["insight"],
            "risk_flags": [],
            "disagreements_with_algo": [],
            "symbol_views": {},
            "reasoning": "ok",
        }

    monkeypatch.setattr(LegacyLLMClient, "complete", _fake_complete)

    agent = BranchSubAgent(
        branch_name="quant",
        llm_client=LegacyLLMClient(),
        model="moonshot-v1-8k",
        timeout=1.0,
    )
    output = asyncio.run(
        agent.analyze(
            BranchAgentInput(
                branch_name="quant",
                base_score=0.0,
                final_score=0.1,
                confidence=0.5,
            )
        )
    )

    assert output.branch_name == "quant"
    assert captured["stage"] == "review_branch_subagent"
    assert captured["actor_name"] == "quant"


def test_risk_subagent_forwards_stage_and_actor_name(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "risk_assessment": "acceptable",
            "max_recommended_exposure": 0.7,
            "position_adjustments": {},
            "risk_warnings": [],
            "hedging_suggestions": [],
            "tail_risk_assessment": "normal",
            "correlation_breakdown_risk": 0.0,
            "position_sizing_overrides": {},
            "drawdown_scenario": "",
            "reasoning": "ok",
        }

    monkeypatch.setattr(LegacyLLMClient, "complete", _fake_complete)

    agent = RiskSubAgent(
        llm_client=LegacyLLMClient(),
        model="moonshot-v1-8k",
        timeout=1.0,
    )
    output = asyncio.run(agent.analyze(RiskAgentInput()))

    assert output.risk_assessment == "acceptable"
    assert captured["stage"] == "review_risk_subagent"
    assert captured["actor_name"] == "RiskGuard"


def test_supports_json_mode():
    from quant_investor.llm_transport import supports_json_mode

    assert supports_json_mode("kimi", "moonshot-v1-128k") is True
    assert supports_json_mode("deepseek", "deepseek-chat") is True
    assert supports_json_mode("kimi", "moonshot-v1-8k") is True


def test_supports_reasoning_effort():
    from quant_investor.llm_transport import supports_reasoning_effort

    assert supports_reasoning_effort("deepseek", "deepseek-reasoner") is True
    assert supports_reasoning_effort("deepseek", "deepseek-chat") is False
    assert supports_reasoning_effort("kimi", "moonshot-v1-128k") is False


def test_telemetry_safe_body_strips_telemetry_keys():
    from quant_investor.llm_transport import telemetry_safe_body

    body = {
        "model": "moonshot-v1-128k",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 128,
        "stage": "review_branch_overlay",
        "actor_name": "IC:000001.SZ",
        "reasoning_effort": "high",
        "temperature": 0.3,
    }
    safe = telemetry_safe_body(body)
    assert "model" in safe
    assert "messages" in safe
    assert "temperature" in safe
    assert "stage" not in safe
    assert "actor_name" not in safe
    assert "reasoning_effort" not in safe


def test_build_body_strips_telemetry_and_respects_json_mode():
    from quant_investor.llm_transport import build_openai_compatible_completion_body

    body = build_openai_compatible_completion_body(
        provider="kimi",
        model="moonshot-v1-128k",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        response_json=True,
        extra_body={"stage": "test_stage", "actor_name": "test_actor"},
    )
    assert "stage" not in body
    assert "actor_name" not in body
    assert body.get("response_format") == {"type": "json_object"}

def test_master_agent_forwards_stage_and_actor_name(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "final_conviction": "buy",
            "final_score": 0.2,
            "confidence": 0.7,
            "consensus_areas": [],
            "disagreement_areas": [],
            "debate_rounds": [],
            "debate_resolution": [],
            "conviction_drivers": [],
            "top_picks": [],
            "portfolio_narrative": "narrative",
            "risk_adjusted_exposure": 0.5,
            "dissenting_views": [],
        }

    monkeypatch.setattr(LegacyLLMClient, "complete", _fake_complete)

    agent = MasterAgent(
        llm_client=LegacyLLMClient(),
        model="moonshot-v1-8k",
        reasoning_effort="high",
        timeout=1.0,
    )
    output = asyncio.run(
        agent.deliberate(
            MasterAgentInput(
                branch_reports={},
                risk_report=None,
                ensemble_baseline={"aggregate_score": 0.0},
                market_regime="neutral",
                candidate_symbols=[],
            )
        )
    )

    assert output.final_conviction == "buy"
    assert captured["stage"] == "review_master_agent"
    assert captured["actor_name"] == "ICCoordinator"


def test_master_agent_forwards_fallback_model(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "final_conviction": "buy",
            "final_score": 0.2,
            "confidence": 0.7,
            "consensus_areas": [],
            "disagreement_areas": [],
            "debate_rounds": [],
            "debate_resolution": [],
            "conviction_drivers": [],
            "top_picks": [],
            "portfolio_narrative": "narrative",
            "risk_adjusted_exposure": 0.5,
            "dissenting_views": [],
        }

    monkeypatch.setattr(LegacyLLMClient, "complete", _fake_complete)

    agent = MasterAgent(
        llm_client=LegacyLLMClient(),
        model="qwen-plus",
        fallback_model="moonshot-v1-128k",
        reasoning_effort="high",
        timeout=1.0,
    )
    asyncio.run(
        agent.deliberate(
            MasterAgentInput(
                branch_reports={},
                risk_report=None,
                ensemble_baseline={"aggregate_score": 0.0},
                market_regime="neutral",
                candidate_symbols=[],
            )
        )
    )

    assert captured["fallback_model"] == "moonshot-v1-128k"
