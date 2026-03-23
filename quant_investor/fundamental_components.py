#!/usr/bin/env python3
"""
Fundamental Branch 子组件。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_investor.branch_contracts import (
    CorporateDocumentSnapshot,
    ForecastSnapshot,
    FundamentalSnapshot,
    ManagementSnapshot,
    OwnershipSnapshot,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class FundamentalComponentResult:
    name: str
    score: float = 0.0
    available: bool = False
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    used_features: list[str] = field(default_factory=list)


def financial_quality_analyzer(snapshot: FundamentalSnapshot) -> FundamentalComponentResult:
    result = FundamentalComponentResult(
        name="financial_quality",
        available=snapshot.available,
        used_features=["roe", "gross_margin", "profit_growth", "revenue_growth", "debt_ratio", "current_ratio"],
    )
    if not snapshot.available:
        result.risks.append("财务快照缺失，财务质量回退中性。")
        return result

    score = 0.0
    if snapshot.roe >= 0.12:
        result.evidence.append("ROE 处于较优区间。")
        score += 0.30
    elif snapshot.roe <= 0.05:
        result.risks.append("ROE 偏低。")
        score -= 0.20

    if snapshot.gross_margin >= 0.28:
        result.evidence.append("毛利率稳健。")
        score += 0.20
    elif snapshot.gross_margin <= 0.15:
        result.risks.append("毛利率偏弱。")
        score -= 0.10

    if snapshot.profit_growth >= 0.10 or snapshot.revenue_growth >= 0.12:
        result.evidence.append("成长性仍在延续。")
        score += 0.25
    elif snapshot.profit_growth <= -0.05:
        result.risks.append("盈利增速转弱。")
        score -= 0.20

    if snapshot.debt_ratio >= 0.60:
        result.risks.append("负债率偏高。")
        score -= 0.20
    elif 0.0 < snapshot.debt_ratio <= 0.35:
        result.evidence.append("杠杆水平温和。")
        score += 0.12

    if snapshot.current_ratio >= 1.2:
        result.evidence.append("短期偿债能力尚可。")
        score += 0.08

    result.score = _clamp(score, -1.0, 1.0)
    return result


def forecast_revision_analyzer(snapshot: ForecastSnapshot) -> FundamentalComponentResult:
    result = FundamentalComponentResult(
        name="forecast_revision",
        available=snapshot.available,
        used_features=["eps_growth", "revenue_growth_forecast", "forecast_revision", "coverage_count"],
    )
    if not snapshot.available:
        result.risks.append("盈利预测 provider 缺失，预测修正回退中性。")
        return result

    score = 0.0
    if snapshot.forecast_revision > 0.03:
        result.evidence.append("一致预期修正向上。")
        score += 0.30
    elif snapshot.forecast_revision < -0.03:
        result.risks.append("一致预期修正向下。")
        score -= 0.30

    if snapshot.eps_growth > 0.12:
        result.evidence.append("EPS 预期增速积极。")
        score += 0.18
    elif snapshot.eps_growth < 0:
        result.risks.append("EPS 预期增速为负。")
        score -= 0.18

    if snapshot.coverage_count <= 1:
        result.risks.append("分析师覆盖稀少，预测可信度有限。")
        score -= 0.05

    result.score = _clamp(score, -1.0, 1.0)
    return result


def valuation_analyzer(snapshot: FundamentalSnapshot) -> FundamentalComponentResult:
    result = FundamentalComponentResult(
        name="valuation",
        available=snapshot.available,
        used_features=["pe", "pb", "ps", "dividend_yield"],
    )
    if not snapshot.available:
        result.risks.append("估值快照缺失，估值模块回退中性。")
        return result

    score = 0.0
    if 0 < snapshot.pe <= 14:
        result.evidence.append("PE 估值处于相对合理区间。")
        score += 0.22
    elif snapshot.pe >= 35:
        result.risks.append("PE 偏高。")
        score -= 0.18

    if 0 < snapshot.pb <= 2.0:
        result.evidence.append("PB 不算拥挤。")
        score += 0.18
    elif snapshot.pb >= 5.0:
        result.risks.append("PB 偏高。")
        score -= 0.15

    if 0 < snapshot.ps <= 2.5:
        score += 0.08
    elif snapshot.ps >= 6.0:
        result.risks.append("PS 偏高。")
        score -= 0.08

    if snapshot.dividend_yield >= 0.03:
        result.evidence.append("股息率提供一定保护。")
        score += 0.06

    result.score = _clamp(score, -1.0, 1.0)
    return result


def management_governance_analyzer(snapshot: ManagementSnapshot) -> FundamentalComponentResult:
    result = FundamentalComponentResult(
        name="management_governance",
        available=snapshot.available,
        used_features=["management_stability", "governance_score", "management_alignment"],
    )
    if not snapshot.available:
        result.risks.append("管理层数据缺失，治理模块回退中性。")
        return result

    score = (
        0.40 * snapshot.management_stability
        + 0.35 * snapshot.governance_score
        + 0.25 * snapshot.management_alignment
    )
    if snapshot.governance_score >= 0.5:
        result.evidence.append("治理结构偏稳健。")
    elif snapshot.governance_score <= -0.2:
        result.risks.append("治理结构存在隐忧。")
    if snapshot.key_executive_changes:
        result.risks.append("近期存在核心管理层变动。")
        score -= 0.10

    result.score = _clamp(score, -1.0, 1.0)
    return result


def ownership_analyzer(snapshot: OwnershipSnapshot) -> FundamentalComponentResult:
    result = FundamentalComponentResult(
        name="ownership",
        available=snapshot.available,
        used_features=["concentration_score", "top_holder_pct", "institutional_holding_pct", "ownership_change_signal"],
    )
    if not snapshot.available:
        result.risks.append("股东结构数据缺失，ownership 模块回退中性。")
        return result

    score = 0.50 * snapshot.concentration_score + 0.30 * snapshot.ownership_change_signal
    if snapshot.institutional_holding_pct >= 0.20:
        result.evidence.append("机构持股占比具备一定支撑。")
        score += 0.12
    if snapshot.top_holder_pct >= 0.60:
        result.risks.append("股权过度集中。")
        score -= 0.10

    result.score = _clamp(score, -1.0, 1.0)
    return result


def document_semantic_analyzer(snapshot: CorporateDocumentSnapshot) -> FundamentalComponentResult:
    result = FundamentalComponentResult(
        name="document_semantics",
        available=snapshot.available,
        used_features=["semantic_sentiment", "execution_confidence", "governance_red_flag"],
    )
    if not snapshot.available:
        result.risks.append("离线文档语义快照缺失，文档模块回退中性。")
        return result

    score = (
        0.55 * snapshot.semantic_sentiment
        + 0.25 * snapshot.execution_confidence
        - 0.45 * snapshot.governance_red_flag
    )
    if snapshot.semantic_sentiment > 0.2:
        result.evidence.append("文档语义整体偏积极。")
    elif snapshot.semantic_sentiment < -0.2:
        result.risks.append("文档语义偏谨慎。")
    if snapshot.governance_red_flag > 0.35:
        result.risks.append("文档语义揭示治理红旗。")

    result.score = _clamp(score, -1.0, 1.0)
    return result
