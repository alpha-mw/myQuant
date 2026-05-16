from __future__ import annotations

import pytest

from quant_investor.llm_gateway import LLMCallError, LLMClient, has_any_provider, has_provider_for_model
from quant_investor.llm_policy import apply_local_llm_policy, local_llm_disabled


def test_codex_handoff_disables_local_provider_detection(monkeypatch):
    monkeypatch.delenv("MYQUANT_ENABLE_LOCAL_LLM", raising=False)
    monkeypatch.setenv("MYQUANT_LLM_HANDOFF", "codex")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    assert local_llm_disabled() is True
    assert apply_local_llm_policy(True) is False
    assert has_any_provider() is False
    assert has_provider_for_model("deepseek-chat") is False


@pytest.mark.asyncio
async def test_codex_handoff_blocks_direct_llm_completion(monkeypatch):
    monkeypatch.delenv("MYQUANT_ENABLE_LOCAL_LLM", raising=False)
    monkeypatch.setenv("MYQUANT_DISABLE_LOCAL_LLM", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    client = LLMClient()
    with pytest.raises(LLMCallError, match="Codex handoff"):
        await client.complete_text(
            messages=[{"role": "user", "content": "hello"}],
            model="deepseek-chat",
        )
