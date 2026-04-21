from __future__ import annotations

from quant_investor.research_run_config import ResearchRunConfig, ResolvedReviewModels


def test_resolved_review_models_preserve_explicit_priority_and_role_overrides():
    resolved = ResolvedReviewModels.from_mapping(
        {
            "review_model_priority": ["qwen3.5-plus", "moonshot-v1-128k"],
            "agent_model": "deepseek-chat",
            "agent_fallback_model": "qwen3.5-flash",
            "master_model": "moonshot-v1-128k",
            "master_fallback_model": "deepseek-reasoner",
            "master_reasoning_effort": "medium",
            "agent_timeout": 25,
            "master_timeout": 55,
        }
    )

    assert resolved.review_model_priority == ["qwen3.5-plus", "moonshot-v1-128k"]
    assert resolved.branch.primary_model == "deepseek-chat"
    assert resolved.branch.fallback_model == "qwen3.5-flash"
    assert resolved.master.primary_model == "moonshot-v1-128k"
    assert resolved.master.fallback_model == "deepseek-reasoner"
    assert resolved.master_reasoning_effort == "medium"
    assert resolved.agent_timeout == 25.0
    assert resolved.master_timeout == 55.0


def test_research_run_config_builds_quant_investor_kwargs_from_workspace_payload():
    config = ResearchRunConfig.from_mapping(
        {
            "stock_pool": ["000001.SZ", "600519.SH"],
            "market": "CN",
            "capital": 2_000_000,
            "risk_level": "积极",
            "lookback_years": 2,
            "kline_backend": "hybrid",
            "enable_macro": True,
            "enable_quant": True,
            "enable_kline": True,
            "enable_fundamental": False,
            "enable_intelligence": True,
            "enable_agent_layer": True,
            "review_model_priority": ["deepseek-chat"],
            "agent_model": "deepseek-chat",
            "master_model": "moonshot-v1-128k",
            "agent_timeout": 30,
            "master_timeout": 90,
            "funnel_profile": "momentum_leader",
            "max_candidates": 120,
            "trend_windows": [15, 45, 120],
            "volume_spike_threshold": 1.5,
            "breakout_distance_pct": 0.05,
        },
        recall_context={"source": "workspace"},
    )

    kwargs = config.to_quant_investor_kwargs(verbose=False)

    assert kwargs["stock_pool"] == ["000001.SZ", "600519.SH"]
    assert kwargs["market"] == "CN"
    assert kwargs["total_capital"] == 2_000_000
    assert kwargs["risk_level"] == "积极"
    assert kwargs["enable_fundamental"] is False
    assert kwargs["enable_agent_layer"] is True
    assert kwargs["review_model_priority"] == ["deepseek-chat"]
    assert kwargs["agent_model"] == "deepseek-chat"
    assert kwargs["master_model"] == "moonshot-v1-128k"
    assert kwargs["agent_timeout"] == 30.0
    assert kwargs["master_timeout"] == 90.0
    assert kwargs["funnel_profile"] == "momentum_leader"
    assert kwargs["max_candidates"] == 120
    assert kwargs["trend_windows"] == [15, 45, 120]
    assert kwargs["volume_spike_threshold"] == 1.5
    assert kwargs["breakout_distance_pct"] == 0.05
    assert kwargs["recall_context"] == {"source": "workspace"}
    assert kwargs["verbose"] is False


def test_research_run_config_honors_cli_negative_toggles():
    config = ResearchRunConfig.from_mapping(
        {
            "stocks": ["600000.SH"],
            "market": "CN",
            "capital": 1_500_000,
            "risk": "中等",
            "lookback": 1.5,
            "kline_backend": "heuristic",
            "no_macro": True,
            "no_quant": False,
            "no_kline": True,
            "no_fundamental": True,
            "no_intelligence": False,
            "no_agent_layer": True,
            "disable_document_semantics": True,
            "allow_synthetic_for_research": True,
        }
    )

    kwargs = config.to_quant_investor_kwargs(verbose=True)

    assert kwargs["stock_pool"] == ["600000.SH"]
    assert kwargs["lookback_years"] == 1.5
    assert kwargs["enable_macro"] is False
    assert kwargs["enable_quant"] is True
    assert kwargs["enable_kline"] is False
    assert kwargs["enable_fundamental"] is False
    assert kwargs["enable_intelligence"] is True
    assert kwargs["enable_agent_layer"] is False
    assert kwargs["enable_document_semantics"] is False
    assert kwargs["allow_synthetic_for_research"] is True
