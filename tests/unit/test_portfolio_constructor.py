"""
PortfolioConstructor 单元测试。
"""

from __future__ import annotations

from quant_investor.agent_protocol import ActionLabel, BranchVerdict, ICDecision
from quant_investor.agents.portfolio_constructor import PortfolioConstructor


def _macro_verdict(target_gross_exposure: float = 0.55) -> BranchVerdict:
    return BranchVerdict(
        agent_name="MacroAgent",
        thesis="宏观要求控制总暴露。",
        final_score=0.1,
        final_confidence=0.8,
        metadata={
            "regime": "balanced",
            "target_gross_exposure": target_gross_exposure,
            "style_bias": "balanced_quality",
        },
    )


def test_portfolio_constructor_is_deterministic_and_respects_gross_cap():
    constructor = PortfolioConstructor()
    payload = {
        "ic_decisions": [
            ICDecision(
                thesis="消费龙头与银行龙头优先保留。",
                final_score=0.7,
                final_confidence=0.85,
                action=ActionLabel.BUY,
                metadata={
                    "symbol_candidates": [
                        {
                            "symbol": "600519.SH",
                            "score": 0.8,
                            "confidence": 0.9,
                            "action": "buy",
                            "position_mode": "target",
                            "sector": "consumer",
                        },
                        {
                            "symbol": "000001.SZ",
                            "score": 0.55,
                            "confidence": 0.75,
                            "action": "hold",
                            "position_mode": "target",
                            "sector": "bank",
                        },
                    ]
                },
            )
        ],
        "macro_verdict": _macro_verdict(0.55),
        "risk_limits": {
            "gross_exposure_cap": 0.6,
            "max_weight": 0.3,
            "position_limits": {
                "600519.SH": 0.28,
                "000001.SZ": 0.25,
            },
            "sector_caps": {
                "consumer": 0.30,
                "bank": 0.30,
            },
        },
        "existing_portfolio": {
            "current_weights": {
                "600519.SH": 0.10,
                "000001.SZ": 0.05,
            }
        },
        "tradability_snapshot": {
            "600519.SH": {
                "is_tradable": True,
                "sector": "consumer",
                "liquidity_score": 1.0,
            },
            "000001.SZ": {
                "is_tradable": True,
                "sector": "bank",
                "liquidity_score": 0.9,
            },
        },
    }

    plan_a = constructor.run(payload)
    plan_b = constructor.run(payload)

    assert plan_a == plan_b
    assert plan_a.target_positions == plan_a.target_weights
    assert plan_a.target_gross_exposure <= 0.55 + 1e-9
    assert sum(plan_a.target_positions.values()) <= 0.55 + 1e-9


def test_watch_reject_and_research_only_do_not_enter_target_positions():
    plan = PortfolioConstructor().run(
        {
            "ic_decisions": [
                ICDecision(
                    thesis="只保留确定性更高的标的。",
                    final_score=0.6,
                    final_confidence=0.8,
                    action=ActionLabel.BUY,
                    rejected_symbols=["300750.SZ"],
                    metadata={
                        "symbol_candidates": [
                            {
                                "symbol": "600519.SH",
                                "score": 0.7,
                                "confidence": 0.8,
                                "action": "buy",
                                "position_mode": "target",
                                "sector": "consumer",
                            },
                            {
                                "symbol": "300750.SZ",
                                "score": 0.5,
                                "confidence": 0.7,
                                "action": "watch",
                                "position_mode": "watch",
                                "sector": "battery",
                            },
                            {
                                "symbol": "688111.SH",
                                "score": 0.6,
                                "confidence": 0.75,
                                "action": "buy",
                                "position_mode": "research_only",
                                "sector": "semis",
                            },
                        ]
                    },
                )
            ],
            "macro_verdict": _macro_verdict(0.7),
            "risk_limits": {
                "gross_exposure_cap": 0.8,
                "max_weight": 0.4,
            },
            "existing_portfolio": {"current_weights": {}},
            "tradability_snapshot": {
                "600519.SH": {
                    "is_tradable": True,
                    "sector": "consumer",
                    "liquidity_score": 1.0,
                },
                "300750.SZ": {
                    "is_tradable": True,
                    "sector": "battery",
                    "liquidity_score": 1.0,
                },
                "688111.SH": {
                    "is_tradable": True,
                    "sector": "semis",
                    "liquidity_score": 1.0,
                },
            },
        }
    )

    assert "600519.SH" in plan.target_positions
    assert "300750.SZ" not in plan.target_positions
    assert "688111.SH" not in plan.target_positions
    assert "300750.SZ" in plan.rejected_symbols
    assert "688111.SH" in plan.rejected_symbols


def test_turnover_cap_is_applied_deterministically():
    plan = PortfolioConstructor().run(
        {
            "ic_decisions": [
                ICDecision(
                    thesis="组合需要从单一持仓平滑过渡到双标的。",
                    final_score=0.65,
                    final_confidence=0.8,
                    action=ActionLabel.BUY,
                    metadata={
                        "symbol_candidates": [
                            {
                                "symbol": "600519.SH",
                                "score": 0.8,
                                "confidence": 0.85,
                                "action": "buy",
                                "position_mode": "target",
                                "sector": "consumer",
                            },
                            {
                                "symbol": "000001.SZ",
                                "score": 0.55,
                                "confidence": 0.7,
                                "action": "buy",
                                "position_mode": "target",
                                "sector": "bank",
                            },
                        ]
                    },
                )
            ],
            "macro_verdict": _macro_verdict(0.8),
            "risk_limits": {
                "gross_exposure_cap": 0.8,
                "max_weight": 0.4,
                "turnover_cap": 0.05,
                "sector_caps": {
                    "consumer": 0.5,
                    "bank": 0.5,
                },
            },
            "existing_portfolio": {
                "current_weights": {
                    "000001.SZ": 0.4,
                }
            },
            "tradability_snapshot": {
                "600519.SH": {
                    "is_tradable": True,
                    "sector": "consumer",
                    "liquidity_score": 1.0,
                },
                "000001.SZ": {
                    "is_tradable": True,
                    "sector": "bank",
                    "liquidity_score": 1.0,
                },
            },
        }
    )

    assert plan.turnover_estimate <= 0.05 + 1e-6
    assert plan.metadata["deterministic"] is True
