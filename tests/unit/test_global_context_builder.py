"""Tests for GlobalContext builder."""

from __future__ import annotations

import pytest

from quant_investor.agent_protocol import DataQualityIssue, GlobalContext
from quant_investor.global_context.builder import GlobalContextBuilder


def test_builder_produces_global_context_with_universe_tiers(monkeypatch):
    monkeypatch.setattr(
        "quant_investor.global_context.builder._detect_macro_regime",
        lambda *_a, **_kw: ("震荡低波", {"max_position": 0.80}),
    )
    monkeypatch.setattr(
        "quant_investor.global_context.builder._load_stock_name_map",
        lambda market: {"000001.SZ": "平安银行", "600519.SH": "贵州茅台"},
    )
    monkeypatch.setattr(
        "quant_investor.model_roles.has_provider_for_model",
        lambda _model: True,
    )

    builder = GlobalContextBuilder()
    ctx = builder.build(
        stock_pool=["000001.SZ", "600519.SH", "BAD.SZ"],
        market="CN",
        universe_key="full_a",
        total_capital=500_000,
        data_quality_issues=[
            DataQualityIssue(symbol="BAD.SZ", severity="error", message="unreadable"),
        ],
        agent_model="deepseek-reasoner",
        master_model="moonshot-v1-128k",
        freshness_mode="stable",
        latest_trade_date="20260403",
    )

    assert isinstance(ctx, GlobalContext)
    assert ctx.market == "CN"
    assert ctx.macro_regime == "震荡低波"
    assert ctx.freshness_mode == "stable"
    assert ctx.latest_trade_date == "20260403"
    assert ctx.effective_target_trade_date == "20260403"

    # Universe tiers
    assert len(ctx.universe_tiers["total"]) == 3
    assert "BAD.SZ" not in ctx.universe_tiers["researchable"]
    assert len(ctx.universe_tiers["researchable"]) == 2

    # Quarantine
    assert "BAD.SZ" in ctx.data_quality_quarantine

    # Symbol names
    assert ctx.symbol_name_map["000001.SZ"] == "平安银行"

    # Model capability
    assert "branch" in ctx.model_capability_map
    assert "master" in ctx.model_capability_map

    # Hash is stable
    assert len(ctx.universe_hash) == 16


def test_builder_handles_empty_pool(monkeypatch):
    monkeypatch.setattr(
        "quant_investor.global_context.builder._detect_macro_regime",
        lambda *_a, **_kw: ("未知", {}),
    )
    monkeypatch.setattr(
        "quant_investor.global_context.builder._load_stock_name_map",
        lambda market: {},
    )
    monkeypatch.setattr(
        "quant_investor.global_context.builder._build_model_capability_map",
        lambda **_kw: {"branch": {}, "master": {}},
    )

    builder = GlobalContextBuilder()
    ctx = builder.build(stock_pool=[], market="CN")
    assert ctx.universe_tiers["total"] == []
    assert ctx.universe_tiers["researchable"] == []
    assert ctx.macro_regime == "未知"


def test_builder_integrates_provider_health_into_capability_map(monkeypatch):
    monkeypatch.setattr(
        "quant_investor.global_context.builder._detect_macro_regime",
        lambda *_a, **_kw: ("震荡低波", {}),
    )
    monkeypatch.setattr(
        "quant_investor.global_context.builder._load_stock_name_map",
        lambda market: {},
    )
    monkeypatch.setattr(
        "quant_investor.global_context.builder._build_model_capability_map",
        lambda **_kw: {
            "branch": {"resolved_model": "deepseek-reasoner", "fallback_used": False},
            "master": {"resolved_model": "moonshot-v1-128k", "fallback_used": False},
        },
    )
    monkeypatch.setattr(
        "quant_investor.global_context.builder.detect_provider_health",
        lambda **_kw: {
            "deepseek": {"available": True},
            "master": {"available": False},
            "kline": {"kronos_available": True, "chronos_available": False},
        },
    )

    builder = GlobalContextBuilder()
    ctx = builder.build(
        stock_pool=["000001.SZ"],
        market="CN",
        agent_model="deepseek-reasoner",
        master_model="moonshot-v1-128k",
    )

    # detect_provider_health results get merged into model_capability_map
    assert ctx.model_capability_map["deepseek"]["available"] is True
    assert ctx.model_capability_map["master"]["available"] is False
    assert ctx.model_capability_map["kline"]["kronos_available"] is True
    # Original branch/master entries preserved
    assert ctx.model_capability_map["branch"]["resolved_model"] == "deepseek-reasoner"


def test_bayesian_decision_record_exists():
    from quant_investor.agent_protocol import BayesianDecisionRecord

    record = BayesianDecisionRecord(
        symbol="000001.SZ",
        company_name="平安银行",
        posterior_win_rate=0.62,
        posterior_expected_alpha=0.03,
        posterior_confidence=0.71,
        posterior_action_score=0.55,
    )
    d = record.to_dict()
    assert d["symbol"] == "000001.SZ"
    assert d["posterior_win_rate"] == 0.62
    assert "correlation_discount" in d
    assert "coverage_discount" in d
