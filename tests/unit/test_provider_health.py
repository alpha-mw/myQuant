from __future__ import annotations


def test_provider_health_detects_capabilities_without_network(monkeypatch):
    from quant_investor.market.provider_health import detect_provider_health

    monkeypatch.setattr(
        "quant_investor.market.provider_health.has_provider_for_model",
        lambda model: model in {"deepseek-reasoner", "moonshot-v1-128k"},
    )
    monkeypatch.setattr(
        "quant_investor.market.provider_health.probe_kline_model_capabilities",
        lambda: {
            "kronos_available": True,
            "chronos_available": False,
            "mode": "kronos_only_degraded",
        },
    )

    result = detect_provider_health(
        agent_model="deepseek-reasoner",
        master_model="moonshot-v1-128k",
    )

    assert result["agent"]["available"] is True
    assert result["master"]["available"] is True
    assert result["kline"]["kronos_available"] is True
    assert result["kline"]["chronos_available"] is False

