#!/usr/bin/env python3
"""
branch-local debate 模板与限制。
"""

from __future__ import annotations

from typing import Any

from quant_investor.branch_contracts import EvidencePacket
from quant_investor.versioning import DEBATE_TEMPLATE_VERSION

DEBATE_OUTPUT_SCHEMA: dict[str, Any] = {
    "direction": "bullish|bearish|neutral",
    "confidence": 0.0,
    "score_adjustment": 0.0,
    "bull_points": [],
    "bear_points": [],
    "risk_flags": [],
    "unknowns": [],
    "used_features": [],
    "hard_veto": False,
}

BRANCH_DEBATE_ADJUSTMENT_CAPS: dict[str, float] = {
    "kline": 0.10,
    "quant": 0.10,
    "fundamental": 0.20,
    "intelligence": 0.15,
    "macro": 0.10,
}

BRANCH_DEBATE_INTENTS: dict[str, str] = {
    "kline": "校验趋势/波动/图形结构是否支持 base signal，并只做小幅置信度与风险修正。",
    "quant": "校验因子/模型/稳定性是否支持 base signal，并只做小幅修正。",
    "fundamental": "校验财务、预测、估值、管理层、股东结构和文档语义是否支持 base signal。",
    "intelligence": "校验新闻、事件、情绪、资金流、广度、行业轮动是否支持 base signal。",
    "macro": "校验全市场宏观 overlay 是否支持 base signal，只允许市场级一次性审查。",
}


def build_debate_prompt(branch_name: str, evidence: EvidencePacket) -> str:
    """构造标准化 debate 上下文。"""
    intent = BRANCH_DEBATE_INTENTS.get(branch_name, "校验证据与 base signal 的一致性。")
    summary = evidence.summary or "无额外摘要"
    bull_points = "；".join(evidence.bull_points[:5]) or "无"
    bear_points = "；".join(evidence.bear_points[:5]) or "无"
    risks = "；".join(evidence.risk_points[:5]) or "无"
    unknowns = "；".join(evidence.unknowns[:5]) or "无"
    features = "；".join(evidence.used_features[:8]) or "无"
    return (
        f"branch={branch_name}\n"
        f"template_version={DEBATE_TEMPLATE_VERSION}\n"
        f"intent={intent}\n"
        f"scope={evidence.scope}\n"
        f"summary={summary}\n"
        f"bull_points={bull_points}\n"
        f"bear_points={bear_points}\n"
        f"risk_points={risks}\n"
        f"unknowns={unknowns}\n"
        f"used_features={features}\n"
        f"schema={DEBATE_OUTPUT_SCHEMA}\n"
    )
