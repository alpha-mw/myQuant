from __future__ import annotations

from typing import Any

from quant_investor.branch_contracts import TradeRecommendation
from quant_investor.contracts import GlobalContext, PortfolioDecision, ShortlistItem


def build_portfolio_decisions(
    *,
    shortlist: list[ShortlistItem],
    trade_recommendations: list[TradeRecommendation],
    global_context: GlobalContext,
) -> list[PortfolioDecision]:
    shortlist_map = {item.symbol: item for item in shortlist}
    decisions: list[PortfolioDecision] = []
    for recommendation in trade_recommendations:
        shortlist_item = shortlist_map.get(recommendation.symbol)
        if shortlist_item is None:
            continue
        decisions.append(
            PortfolioDecision(
                symbol=recommendation.symbol,
                action=str(recommendation.action),
                target_weight=float(recommendation.suggested_weight or recommendation.weight or 0.0),
                entry_price=_optional_float(
                    getattr(recommendation, "recommended_entry_price", None)
                    or getattr(recommendation, "entry_price", None)
                ),
                target_price=_optional_float(getattr(recommendation, "target_price", None)),
                stop_loss=_optional_float(
                    getattr(recommendation, "stop_loss_price", None)
                    or getattr(recommendation, "stop_loss", None)
                ),
                thesis=str(
                    getattr(recommendation, "one_line_conclusion", "")
                    or getattr(recommendation, "rationale", "")
                    or shortlist_item.rationale
                ),
                what_if_scenarios=_build_what_if_scenarios(recommendation),
                confidence=float(getattr(recommendation, "confidence", 0.0)),
                packet_ref=str(shortlist_item.packet_ref or global_context.cache_key),
                metadata={
                    "rank_score": float(shortlist_item.rank_score),
                    "expected_return": float(shortlist_item.expected_return),
                    "downside_risk": float(shortlist_item.downside_risk),
                    "liquidity_score": float(shortlist_item.liquidity_score),
                },
            )
        )
    return decisions


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_what_if_scenarios(recommendation: TradeRecommendation) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    position_notes = list(recommendation.position_management or [])
    if position_notes:
        scenarios.append(
            {
                "scenario": "加仓",
                "trigger_condition": position_notes[0],
                "expected_outcome": "按计划追加仓位",
                "probability": 0.45,
            }
        )
    if len(position_notes) > 1:
        scenarios.append(
            {
                "scenario": "减仓",
                "trigger_condition": position_notes[1],
                "expected_outcome": "降低风险敞口",
                "probability": 0.35,
            }
        )
    if len(position_notes) > 2:
        scenarios.append(
            {
                "scenario": "离场",
                "trigger_condition": position_notes[2],
                "expected_outcome": "执行纪律性止盈或止损",
                "probability": 0.20,
            }
        )
    if not scenarios:
        scenarios = [
            {
                "scenario": "加仓",
                "trigger_condition": "价格与分支共识继续改善",
                "expected_outcome": "继续按计划分批买入",
                "probability": 0.34,
            },
            {
                "scenario": "减仓",
                "trigger_condition": "风险信号上升或权重超限",
                "expected_outcome": "降低仓位并锁定收益",
                "probability": 0.33,
            },
            {
                "scenario": "离场",
                "trigger_condition": "跌破止损或基本面/事件转弱",
                "expected_outcome": "退出持仓并保留现金",
                "probability": 0.33,
            },
        ]
    return scenarios
