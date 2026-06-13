from __future__ import annotations

from quant_investor import llm_gateway
from quant_investor import llm_gateway_parsing
from quant_investor import llm_gateway_types


def test_llm_gateway_reexports_contract_types() -> None:
    assert llm_gateway.LLMCallError is llm_gateway_types.LLMCallError
    assert llm_gateway.LLMProviderResponseError is llm_gateway_types.LLMProviderResponseError
    assert llm_gateway.LLMProviderSpec is llm_gateway_types.LLMProviderSpec
    assert llm_gateway.LLMModelPricing is llm_gateway_types.LLMModelPricing
    assert llm_gateway.LLMUsageSessionHandle is llm_gateway_types.LLMUsageSessionHandle
    assert llm_gateway.ProviderFailureAnalysis is llm_gateway_types.ProviderFailureAnalysis
    assert llm_gateway.ProviderCooldown is llm_gateway_types.ProviderCooldown


def test_llm_gateway_reexports_contract_registries() -> None:
    assert llm_gateway.LLM_PROVIDER_REGISTRY is llm_gateway_types.LLM_PROVIDER_REGISTRY
    assert llm_gateway.LLM_MODEL_PRICING_REGISTRY is llm_gateway_types.LLM_MODEL_PRICING_REGISTRY
    assert llm_gateway.LLM_STAGE_NAMES is llm_gateway_types.LLM_STAGE_NAMES
    assert llm_gateway.LLM_PROVIDER_ENV_KEYS is llm_gateway_types.LLM_PROVIDER_ENV_KEYS
    assert llm_gateway.USAGE_LOG_PATH == llm_gateway_types.USAGE_LOG_PATH
    assert llm_gateway.PROVIDER_CONCURRENCY_LIMITS is llm_gateway_types.PROVIDER_CONCURRENCY_LIMITS
    assert "deepseek" in llm_gateway.LLM_PROVIDER_REGISTRY


def test_llm_client_uses_shared_json_parser() -> None:
    assert llm_gateway.LLMClient._parse_json_content is llm_gateway_parsing.parse_json_content
