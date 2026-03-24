"""
Trade Case Store 单元测试。
"""

from __future__ import annotations

import json
from datetime import datetime

from quant_investor.learning.trade_case_store import (
    AttributionSnapshot,
    ExecutionSnapshot,
    HumanDecisionSnapshot,
    OutcomeSnapshot,
    PreTradeSnapshot,
    TradeCase,
    TradeCaseStore,
    TRADE_CASE_SCHEMA_VERSION,
)


def _build_trade_case(
    *,
    case_id: str = "case-0001",
    symbol: str = "000001.SZ",
    decision_time: datetime | None = None,
    human_action: str = "executed",
) -> TradeCase:
    return TradeCase(
        case_id=case_id,
        symbol=symbol,
        decision_time=decision_time or datetime.fromisoformat("2026-03-24T09:30:00+08:00"),
        pretrade_snapshot=PreTradeSnapshot(
            market_regime="risk_on",
            target_gross_exposure=0.65,
            branch_scores={"kline": 0.62, "quant": 0.58},
            branch_confidences={"kline": 0.7, "quant": 0.66},
            ic_action="buy",
            ic_thesis="量价与因子结论一致，先做中等强度试仓。",
            support_agents=["KlineAgent", "QuantAgent"],
            dissenting_agents=["MacroAgent"],
        ),
        human_decision=HumanDecisionSnapshot(
            human_action=human_action,
            human_reason=["desk_validation"],
            manual_override=human_action == "overridden",
            override_direction="sell" if human_action == "overridden" else None,
            override_notes="人工台账确认。"
            if human_action == "overridden"
            else "保持系统建议。",
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
            manual_notes="原始交易执行记录。",
        ),
        outcomes=OutcomeSnapshot(outcome_status="pending"),
        attribution=AttributionSnapshot(
            helpful_agents=["QuantAgent"],
            misleading_agents=[],
            correct_risk_controls=["gross_exposure_cap"],
            missed_risks=[],
            summary="先记录原始案件，后续再补结果归因。",
        ),
        error_tags=["initial_capture"],
        lesson_draft=["等待 outcomes 更新后补 lessons。"],
        memory_status="raw",
        metadata={"source": "unit_test"},
    )


def test_trade_case_store_can_save_and_load_case(tmp_path):
    store = TradeCaseStore(tmp_path)
    trade_case = _build_trade_case()

    path = store.save_case(trade_case)
    loaded = store.load_case(trade_case.case_id)

    assert path.exists()
    assert loaded.case_id == trade_case.case_id
    assert loaded.symbol == "000001.SZ"
    assert loaded.pretrade_snapshot.branch_scores["kline"] == 0.62
    assert loaded.human_decision.human_action == "executed"
    assert loaded.execution_snapshot.entry_price == 12.34
    assert loaded.schema_version == TRADE_CASE_SCHEMA_VERSION


def test_trade_case_store_can_update_outcomes(tmp_path):
    store = TradeCaseStore(tmp_path)
    trade_case = _build_trade_case()
    store.save_case(trade_case)

    updated = store.update_outcomes(
        trade_case.case_id,
        OutcomeSnapshot(
            t1_return=0.01,
            t5_return=0.03,
            t10_return=0.05,
            t20_return=0.08,
            mfe=0.09,
            mae=-0.02,
            stop_loss_hit=False,
            take_profit_hit=True,
            outcome_status="closed_positive",
        ),
    )

    reloaded = store.load_case(trade_case.case_id)
    assert updated.outcomes.outcome_status == "closed_positive"
    assert reloaded.outcomes.t20_return == 0.08
    assert reloaded.outcomes.take_profit_hit is True


def test_trade_case_store_list_cases_can_filter_by_symbol(tmp_path):
    store = TradeCaseStore(tmp_path)
    store.save_case(_build_trade_case(case_id="case-0001", symbol="000001.SZ"))
    store.save_case(_build_trade_case(case_id="case-0002", symbol="600519.SH"))

    filtered = store.list_cases(symbol="600519.SH")

    assert len(filtered) == 1
    assert filtered[0].case_id == "case-0002"
    assert filtered[0].symbol == "600519.SH"


def test_trade_case_store_persists_schema_version(tmp_path):
    store = TradeCaseStore(tmp_path)
    trade_case = _build_trade_case()

    path = store.save_case(trade_case)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == TRADE_CASE_SCHEMA_VERSION


def test_trade_case_store_supports_human_skipped_case(tmp_path):
    store = TradeCaseStore(tmp_path)
    trade_case = _build_trade_case(case_id="case-skipped", human_action="skipped")

    store.save_case(trade_case)
    loaded = store.load_case("case-skipped")

    assert loaded.human_decision.human_action == "skipped"
    assert loaded.execution_snapshot.executed is False
    assert loaded.execution_snapshot.entry_price is None
    assert loaded.outcomes.outcome_status == "pending"


def test_trade_case_store_can_append_lessons_and_error_tags(tmp_path):
    store = TradeCaseStore(tmp_path)
    trade_case = _build_trade_case(case_id="case-ops")
    store.save_case(trade_case)

    store.append_lesson_draft("case-ops", "二次确认后发现执行窗口过窄。")
    updated = store.add_error_tags("case-ops", ["slippage_watch", "initial_capture"])

    reloaded = store.load_case("case-ops")
    assert reloaded.lesson_draft[-1] == "二次确认后发现执行窗口过窄。"
    assert reloaded.error_tags == ["initial_capture", "slippage_watch"]
    assert updated.error_tags == reloaded.error_tags
