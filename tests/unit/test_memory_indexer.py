"""
Memory indexer 单元测试。
"""

from __future__ import annotations

from datetime import datetime

from quant_investor.learning.memory_indexer import MemoryIndexer
from quant_investor.learning.trade_case_store import (
    AttributionSnapshot,
    ExecutionSnapshot,
    HumanDecisionSnapshot,
    OutcomeSnapshot,
    PreTradeSnapshot,
    TradeCase,
)


def _build_trade_case(
    *,
    case_id: str,
    symbol: str = "000001.SZ",
    human_action: str = "executed",
    support_agents: list[str] | None = None,
    market_regime: str = "risk_on",
    t20_return: float | None = 0.12,
    outcome_status: str = "closed",
    error_tags: list[str] | None = None,
    style_bias: str = "balanced_quality",
    sector: str = "financials",
    volatility_regime: str = "medium",
) -> TradeCase:
    support = support_agents or ["KlineAgent", "QuantAgent"]
    return TradeCase(
        case_id=case_id,
        symbol=symbol,
        decision_time=datetime.fromisoformat("2026-03-24T09:30:00+08:00"),
        pretrade_snapshot=PreTradeSnapshot(
            market_regime=market_regime,
            target_gross_exposure=0.65,
            branch_scores={"kline": 0.61, "quant": 0.58},
            branch_confidences={"kline": 0.72, "quant": 0.68},
            ic_action="buy",
            ic_thesis="研究共识支持试仓。",
            support_agents=support,
            dissenting_agents=["MacroAgent"] if market_regime == "risk_off" else [],
        ),
        human_decision=HumanDecisionSnapshot(
            human_action=human_action,
            human_reason=["desk_review"],
            manual_override=human_action == "overridden",
            override_direction="sell" if human_action == "overridden" else None,
            override_notes="人工覆写记录。" if human_action == "overridden" else "",
        ),
        execution_snapshot=ExecutionSnapshot(
            executed=human_action == "executed",
            entry_price=12.34 if human_action == "executed" else None,
            entry_time=(
                datetime.fromisoformat("2026-03-24T09:35:00+08:00")
                if human_action == "executed"
                else None
            ),
            quantity=1_000 if human_action == "executed" else None,
            manual_notes="execution log",
        ),
        outcomes=OutcomeSnapshot(
            t20_return=t20_return,
            outcome_status=outcome_status,
            stop_loss_hit=True if t20_return is not None and t20_return < -0.05 else False,
            take_profit_hit=True if t20_return is not None and t20_return > 0.08 else False,
        ),
        attribution=AttributionSnapshot(
            helpful_agents=["QuantAgent"],
            misleading_agents=[],
            correct_risk_controls=["gross_exposure_cap"],
            missed_risks=["liquidity_gap"] if t20_return is not None and t20_return < -0.05 else [],
            summary="原始交易案件已进入本地记忆索引。",
        ),
        error_tags=error_tags or [],
        lesson_draft=[],
        memory_status="raw",
        metadata={
            "style_bias": style_bias,
            "sector": sector,
            "holding_horizon": "t20",
            "volatility_regime": volatility_regime,
        },
    )


def test_memory_indexer_builds_memory_item_from_trade_case(tmp_path):
    indexer = MemoryIndexer(tmp_path)
    trade_case = _build_trade_case(case_id="case-001")

    memory_item = indexer.build_memory_item_from_case(trade_case)

    assert memory_item.memory_id == "memory::case-001"
    assert memory_item.source_case_id == "case-001"
    assert memory_item.memory_type == "episodic"
    assert memory_item.tags.market_regime == "risk_on"
    assert memory_item.tags.outcome_bucket == "big_win"
    assert memory_item.created_at == trade_case.decision_time


def test_memory_indexer_extracts_complete_tags(tmp_path):
    indexer = MemoryIndexer(tmp_path)
    trade_case = _build_trade_case(
        case_id="case-002",
        error_tags=["slippage_watch"],
        support_agents=["QuantAgent", "KlineAgent"],
        style_bias="balanced",
        sector="banks",
        volatility_regime="high",
    )

    tags = indexer.extract_tags(trade_case)

    assert tags.market_regime == "risk_on"
    assert tags.style_bias == "balanced"
    assert tags.sector == "banks"
    assert tags.branch_support_pattern == "klineagent+quantagent"
    assert tags.consensus_count == 2
    assert tags.human_action == "executed"
    assert tags.outcome_bucket == "big_win"
    assert tags.error_tags == ["slippage_watch"]
    assert tags.holding_horizon == "t20"
    assert tags.volatility_regime == "high"


def test_memory_indexer_can_search_by_regime_branch_pattern_and_outcome_bucket(tmp_path):
    indexer = MemoryIndexer(tmp_path)
    cases = [
        _build_trade_case(case_id="case-101", market_regime="risk_on", t20_return=0.12),
        _build_trade_case(
            case_id="case-102",
            market_regime="risk_off",
            support_agents=["MacroAgent"],
            t20_return=-0.09,
        ),
        _build_trade_case(case_id="case-103", market_regime="risk_on", t20_return=0.03),
    ]
    indexer.rebuild_index(cases)

    regime_hits = indexer.search_by_tags(market_regime="risk_on")
    pattern_hits = indexer.search_by_tags(branch_support_pattern="klineagent+quantagent")
    outcome_hits = indexer.search_by_tags(outcome_bucket="big_win")

    assert [record.source_case_id for record in regime_hits] == ["case-101", "case-103"]
    assert [record.source_case_id for record in pattern_hits] == ["case-101", "case-103"]
    assert [record.source_case_id for record in outcome_hits] == ["case-101"]


def test_memory_indexer_skipped_and_overridden_cases_can_be_indexed(tmp_path):
    indexer = MemoryIndexer(tmp_path)

    skipped = indexer.index_case(
        _build_trade_case(
            case_id="case-skipped",
            human_action="skipped",
            t20_return=None,
            outcome_status="pending",
        )
    )
    overridden = indexer.index_case(
        _build_trade_case(
            case_id="case-overridden",
            human_action="overridden",
            t20_return=0.02,
            outcome_status="closed",
        )
    )

    skipped_hits = indexer.search_by_tags(human_action="skipped")
    overridden_hits = indexer.search_by_tags(human_action="overridden")

    assert skipped.memory_type == "counterfactual_case"
    assert overridden.memory_type == "preference_case"
    assert [record.source_case_id for record in skipped_hits] == ["case-skipped"]
    assert [record.source_case_id for record in overridden_hits] == ["case-overridden"]


def test_memory_indexer_error_tags_can_participate_in_search(tmp_path):
    indexer = MemoryIndexer(tmp_path)
    indexer.rebuild_index(
        [
            _build_trade_case(case_id="case-201", error_tags=["slippage_watch"]),
            _build_trade_case(case_id="case-202", error_tags=["data_gap"]),
        ]
    )

    results = indexer.search_by_tags(error_tags=["slippage_watch"])

    assert [record.source_case_id for record in results] == ["case-201"]


def test_memory_indexer_can_reload_local_index_and_find_related_cases(tmp_path):
    indexer = MemoryIndexer(tmp_path)
    indexer.rebuild_index(
        [
            _build_trade_case(case_id="case-301", market_regime="risk_on", t20_return=0.12),
            _build_trade_case(case_id="case-302", market_regime="risk_on", t20_return=0.03),
            _build_trade_case(
                case_id="case-303",
                market_regime="risk_off",
                support_agents=["MacroAgent"],
                t20_return=-0.08,
                error_tags=["risk_spike"],
            ),
        ]
    )

    reloaded = MemoryIndexer(tmp_path)
    related = reloaded.get_related_cases("case-301", top_k=1)

    assert [record.source_case_id for record in related] == ["case-302"]
