from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_investor.monitoring.cn_aggressive_portfolio_tracker as tracker


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_FACTOR_FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "factor_library_shadow"


class _Sentinel(Exception):
    pass


def test_build_parser_accepts_allowed_stale_symbols():
    parser = tracker.build_parser()

    args = parser.parse_args(["--allowed-stale-symbols", "601989.SH", "603000.SH"])

    assert args.allowed_stale_symbols == ["601989.SH", "603000.SH"]


def test_switch_plan_accepts_previous_day_realtime_decision_data():
    holdings_review = pd.DataFrame(
        [
            {
                "symbol": "688295.SH",
                "name": "中复神鹰",
                "position_role": "降级观察",
                "rank_full_market": 752,
                "score_full_market": 0.538854,
                "today_change_pct": 0.36,
                "ret20": 0.019102,
            }
        ]
    )
    candidate_pool = pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "name": "奕瑞科技",
                "theme_label": "中盘制造主线",
                "rank_full_market": 68,
                "score_full_market": 0.903194,
                "ret20": 0.384691,
                "candidate_source": "latest_formal_screening",
                "candidate_priority": 1,
                "evidence_quality": "中",
            }
        ]
    )

    switch_plan = tracker._build_switch_plan(
        holdings_review=holdings_review,
        candidate_pool=candidate_pool,
        completeness_passed=False,
        decision_data_sufficient=True,
    )

    row = switch_plan.iloc[0]
    assert row["action"] == "switch_now"
    assert row["switch_ratio_hint"] == "20%"
    assert "strict 完整性" not in row["no_switch_condition"]


def test_run_tracker_forwards_allowed_stale_symbols(monkeypatch, tmp_path):
    ledger = pd.DataFrame(
        [
            {
                "symbol": "601869.SH",
                "name": "长飞光纤",
                "shares": 300,
                "avg_cost": 221.22,
                "cost_basis": 66366.0,
                "current_price": 332.49,
                "current_value": 99747.0,
                "unrealized_pnl": 33381.0,
                "unrealized_pnl_pct": 0.502983,
                "market_weight": 0.107873,
                "stage_target_price": 339.19,
                "stage_stop_price": 231.0,
                "thesis_status": "核心持有",
            }
        ]
    )
    manifest = {"timestamp": "20260408_1118", "capital_cny": 1_000_000.0}
    pnl = pd.DataFrame([{"cash_after": 100000.0, "total_value_after": 975421.0}])

    monkeypatch.setattr(
        tracker,
        "_load_previous_record",
        lambda base_dir, source_record=None: (ledger, manifest, pnl),
    )
    monkeypatch.setattr(
        tracker,
        "get_market_settings",
        lambda _market: SimpleNamespace(data_dir=str(tmp_path / "cn_market_full")),
    )

    class _FakeDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def load_components(self):
            return {"full_a": []}

        def build_completeness_report(self, components=None, allowed_stale_symbols=None):
            assert allowed_stale_symbols == ["601989.SH"]
            raise _Sentinel

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)

    args = argparse.Namespace(
        base_dir=str(tmp_path / "strategy_records"),
        years=7,
        max_rounds=0,
        source_record=None,
        allowed_stale_symbols=["601989.SH"],
    )

    with pytest.raises(_Sentinel):
        tracker.run_tracker(args)


def test_run_unified_review_mainline_degrades_when_llm_usage_missing(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setenv("MYQUANT_ENABLE_LOCAL_LLM", "true")
    monkeypatch.setattr(
        tracker,
        "_load_daily_config_llm_settings",
        lambda: {
            "review_model_priority": ["deepseek-chat", "moonshot-v1-128k", "qwen3.5-plus"],
            "agent_model": "",
            "agent_fallback_model": "",
            "master_model": "moonshot-v1-128k",
            "master_fallback_model": "deepseek-reasoner",
            "master_reasoning_effort": "high",
        },
    )

    class _FakeInvestor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(
                llm_usage_summary=SimpleNamespace(
                    call_count=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                    success_count=0,
                    fallback_count=0,
                    failed_count=0,
                ),
                llm_effective_summary=SimpleNamespace(
                    call_count=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                    success_count=0,
                    fallback_count=0,
                    failed_count=0,
                ),
                llm_usage_session_id="session-0",
                model_role_metadata=SimpleNamespace(
                    branch_model="moonshot-v1-128k",
                    master_model="moonshot-v1-128k",
                    resolved_branch_model="moonshot-v1-128k",
                    resolved_master_model="moonshot-v1-128k",
                    branch_fallback_used=False,
                    master_fallback_used=False,
                ),
                ic_hints_by_symbol={},
                final_strategy=SimpleNamespace(recommendations=[]),
                final_report="",
                review_bundle=SimpleNamespace(fallback_reasons=[]),
            )

    monkeypatch.setattr(tracker, "QuantInvestor", _FakeInvestor)

    ledger = pd.DataFrame(
        [
            {
                "symbol": "601869.SH",
                "name": "长飞光纤",
                "shares": 300,
                "avg_cost": 221.22,
                "cost_basis": 66366.0,
                "current_price": 332.49,
                "current_value": 99747.0,
            }
        ]
    )

    payload = tracker._run_unified_review_mainline_for_holdings(
        source_ledger=ledger,
        latest_trade_date="20260407",
        source_record="20260408_1403",
    )

    assert captured["stock_pool"] == ["601869.SH"]
    assert captured["enable_agent_layer"] is True
    assert captured["market"] == "CN"
    assert captured["review_model_priority"] == ["deepseek-chat", "moonshot-v1-128k", "qwen3.5-plus"]
    assert captured["agent_model"] == "deepseek-chat"
    assert captured["agent_fallback_model"] == "moonshot-v1-128k"
    assert captured["master_model"] == "moonshot-v1-128k"
    assert captured["master_fallback_model"] == "deepseek-reasoner"
    assert captured["master_reasoning_effort"] == "high"
    assert payload["llm_attempt_summary"]["call_count"] == 0
    assert "601869.SH" in payload["degraded_symbols"]
    assert payload["by_symbol"]["601869.SH"]["llm_degraded"] is True


def test_run_unified_review_mainline_aggregates_attempt_and_effective_usage(monkeypatch):
    monkeypatch.setenv("MYQUANT_ENABLE_LOCAL_LLM", "true")
    monkeypatch.setattr(
        tracker,
        "_load_daily_config_llm_settings",
        lambda: {
            "review_model_priority": ["deepseek-chat"],
            "agent_model": "deepseek-chat",
            "agent_fallback_model": "",
            "master_model": "moonshot-v1-128k",
            "master_fallback_model": "",
            "master_reasoning_effort": "high",
            "enable_agent_layer": True,
        },
    )

    class _FakeInvestor:
        def __init__(self, **kwargs):
            self.symbol = kwargs["stock_pool"][0]

        def run(self):
            return SimpleNamespace(
                llm_usage_summary=SimpleNamespace(
                    call_count=2,
                    total_tokens=40,
                    estimated_cost_usd=0.002,
                    success_count=1,
                    fallback_count=1,
                    failed_count=1,
                ),
                llm_effective_summary=SimpleNamespace(
                    call_count=1,
                    total_tokens=18,
                    estimated_cost_usd=0.001,
                    success_count=1,
                    fallback_count=0,
                    failed_count=0,
                ),
                llm_usage_session_id=f"session-{self.symbol}",
                model_role_metadata=SimpleNamespace(
                    branch_model="qwen-plus",
                    master_model="qwen-plus",
                    resolved_branch_model="qwen-plus",
                    resolved_master_model="qwen-plus",
                    branch_fallback_used=False,
                    master_fallback_used=False,
                ),
                ic_hints_by_symbol={self.symbol: {"action": "buy"}},
                final_strategy=SimpleNamespace(recommendations=[]),
                final_report="report",
                review_bundle=SimpleNamespace(fallback_reasons=[]),
            )

    monkeypatch.setattr(tracker, "QuantInvestor", _FakeInvestor)
    ledger = pd.DataFrame(
        [
            {"symbol": "601869.SH", "cost_basis": 100000.0, "current_value": 120000.0},
            {"symbol": "600487.SH", "cost_basis": 100000.0, "current_value": 110000.0},
        ]
    )

    payload = tracker._run_unified_review_mainline_for_holdings(
        source_ledger=ledger,
        latest_trade_date="20260407",
        source_record="20260408_1403",
    )

    assert payload["llm_attempt_summary"]["call_count"] == 4
    assert payload["llm_attempt_summary"]["fallback_count"] == 2
    assert payload["llm_effective_summary"]["call_count"] == 2
    assert payload["llm_effective_summary"]["success_count"] == 2
    assert payload["session_ids"] == {
        "601869.SH": "session-601869.SH",
        "600487.SH": "session-600487.SH",
    }


def test_run_unified_review_mainline_hands_off_to_codex_when_disabled(monkeypatch):
    monkeypatch.delenv("MYQUANT_ENABLE_LOCAL_LLM", raising=False)
    monkeypatch.setenv("MYQUANT_DISABLE_LOCAL_LLM", "true")

    class _UnexpectedInvestor:
        def __init__(self, **kwargs):
            raise AssertionError("local LLM review should be handed off to Codex")

    monkeypatch.setattr(tracker, "QuantInvestor", _UnexpectedInvestor)
    ledger = pd.DataFrame([{"symbol": "601869.SH", "cost_basis": 100000.0, "current_value": 120000.0}])

    payload = tracker._run_unified_review_mainline_for_holdings(
        source_ledger=ledger,
        latest_trade_date="20260407",
        source_record="20260408_1403",
    )

    assert payload["codex_handoff"] is True
    assert payload["local_llm_disabled"] is True
    assert payload["llm_attempt_summary"]["call_count"] == 0
    assert payload["degraded_symbols"] == {}
    assert payload["by_symbol"]["601869.SH"]["codex_handoff"] is True


def test_run_tracker_invokes_unified_review_mainline(monkeypatch, tmp_path):
    ledger = pd.DataFrame(
        [
            {
                "symbol": "601869.SH",
                "name": "长飞光纤",
                "shares": 300,
                "avg_cost": 221.22,
                "cost_basis": 66366.0,
                "current_price": 332.49,
                "current_value": 99747.0,
                "unrealized_pnl": 33381.0,
                "unrealized_pnl_pct": 0.502983,
                "market_weight": 0.107873,
                "stage_target_price": 339.19,
                "stage_stop_price": 231.0,
                "thesis_status": "核心持有",
            }
        ]
    )
    manifest = {"timestamp": "20260408_1118", "capital_cny": 1_000_000.0}
    pnl = pd.DataFrame([{"cash_after": 100000.0, "total_value_after": 975421.0}])

    monkeypatch.setattr(
        tracker,
        "_load_previous_record",
        lambda base_dir, source_record=None: (ledger, manifest, pnl),
    )
    monkeypatch.setattr(
        tracker,
        "get_market_settings",
        lambda _market: SimpleNamespace(data_dir=str(tmp_path / "cn_market_full")),
    )

    class _FakeDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def load_components(self):
            return {"full_a": []}

        def build_completeness_report(self, components=None, allowed_stale_symbols=None):
            return {
                "complete": True,
                "latest_trade_date": "20260407",
                "blocking_incomplete_count": 0,
                "suspension_evidence": {},
            }

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)

    captured: dict[str, object] = {}

    def _fake_review(source_ledger, latest_trade_date, source_record):
        captured["symbols"] = source_ledger["symbol"].astype(str).tolist()
        captured["latest_trade_date"] = latest_trade_date
        captured["source_record"] = source_record
        raise _Sentinel

    monkeypatch.setattr(tracker, "_run_unified_review_mainline_for_holdings", _fake_review)

    args = argparse.Namespace(
        base_dir=str(tmp_path / "strategy_records"),
        years=7,
        max_rounds=0,
        source_record=None,
        allowed_stale_symbols=[],
    )

    with pytest.raises(_Sentinel):
        tracker.run_tracker(args)

    assert captured == {
        "symbols": ["601869.SH"],
        "latest_trade_date": "20260407",
        "source_record": "20260408_1118",
    }


def test_run_tracker_keeps_formal_review_when_completeness_incomplete(monkeypatch, tmp_path):
    ledger = pd.DataFrame(
        [
            {
                "symbol": "601869.SH",
                "name": "长飞光纤",
                "shares": 300,
                "avg_cost": 221.22,
                "cost_basis": 66366.0,
                "current_price": 332.49,
                "current_value": 99747.0,
                "unrealized_pnl": 33381.0,
                "unrealized_pnl_pct": 0.502983,
                "market_weight": 0.107873,
                "stage_target_price": 339.19,
                "stage_stop_price": 231.0,
                "thesis_status": "核心持有",
            }
        ]
    )
    manifest = {"timestamp": "20260408_1118", "capital_cny": 1_000_000.0}
    pnl = pd.DataFrame([{"cash_after": 100000.0, "total_value_after": 975421.0}])

    monkeypatch.setattr(
        tracker,
        "_load_previous_record",
        lambda base_dir, source_record=None: (ledger, manifest, pnl),
    )
    monkeypatch.setattr(
        tracker,
        "get_market_settings",
        lambda _market: SimpleNamespace(data_dir=str(tmp_path / "cn_market_full")),
    )

    calls = {"download_all": 0}

    class _FakeDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def load_components(self):
            return {"full_a": []}

        def build_completeness_report(self, components=None, allowed_stale_symbols=None):
            return {
                "complete": False,
                "latest_trade_date": "20260407",
                "blocking_incomplete_count": 12,
            }

        def download_all(self, *args, **kwargs):
            calls["download_all"] += 1
            raise AssertionError("run_tracker should not auto-backfill before formal review")

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)

    captured: dict[str, object] = {}

    def _fake_review(source_ledger, latest_trade_date, source_record):
        captured["symbols"] = source_ledger["symbol"].astype(str).tolist()
        captured["latest_trade_date"] = latest_trade_date
        captured["source_record"] = source_record
        raise _Sentinel

    monkeypatch.setattr(tracker, "_run_unified_review_mainline_for_holdings", _fake_review)

    args = argparse.Namespace(
        base_dir=str(tmp_path / "strategy_records"),
        years=7,
        max_rounds=3,
        source_record=None,
        allowed_stale_symbols=[],
    )

    with pytest.raises(_Sentinel):
        tracker.run_tracker(args)

    assert calls["download_all"] == 0
    assert captured == {
        "symbols": ["601869.SH"],
        "latest_trade_date": "20260407",
        "source_record": "20260408_1118",
    }


def test_format_top_holdings_by_unrealized_pnl_filters_sign():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA.SH", "name": "甲", "unrealized_pnl": 1200.0},
            {"symbol": "BBB.SH", "name": "乙", "unrealized_pnl": -800.0},
            {"symbol": "CCC.SH", "name": "丙", "unrealized_pnl": -300.0},
        ]
    )

    losers = tracker._format_top_holdings_by_unrealized_pnl(frame, positive=False)

    assert losers == "BBB.SH(乙) -800.00 元；CCC.SH(丙) -300.00 元"


def test_format_top_delta_vs_source_record_filters_sign():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA.SH", "delta_vs_source_record": 1200.0},
            {"symbol": "BBB.SH", "delta_vs_source_record": 230.0},
            {"symbol": "CCC.SH", "delta_vs_source_record": -800.0},
            {"symbol": "DDD.SH", "delta_vs_source_record": -300.0},
        ]
    )

    winners = tracker._format_top_delta_vs_source_record(frame, positive=True)
    detractors = tracker._format_top_delta_vs_source_record(frame, positive=False)
    no_detractors = tracker._format_top_delta_vs_source_record(
        frame[frame["delta_vs_source_record"] > 0],
        positive=False,
    )

    assert winners == "AAA.SH 1,200.00 元；BBB.SH 230.00 元"
    assert detractors == "CCC.SH -800.00 元；DDD.SH -300.00 元"
    assert no_detractors == "无"


def test_build_notes_payload_includes_names_in_switch_detail():
    notes = tracker._build_notes_payload(
        trade_date="2026-05-19",
        data_status="数据状态",
        market_core_view="市场判断",
        pnl_summary={
            "quote_snapshot": "20260519132031",
            "total_value_after": 1_411_455.0,
            "portfolio_pnl_after": 411_455.0,
            "portfolio_pnl_pct_after": 0.411455,
            "delta_vs_source_record": 8_153.0,
        },
        orders=[],
        switch_plan_df=pd.DataFrame(
            [
                {
                    "sell_symbol": "688295.SH",
                    "sell_name": "中复神鹰",
                    "buy_symbol": "688301.SH",
                    "buy_name": "奕瑞科技",
                    "priority": "high",
                    "action": "prepare_switch",
                    "trigger_threshold": "候选继续留在本地强度前120",
                }
            ]
        ),
        candidate_pool=pd.DataFrame(
            [
                {
                    "symbol": "688301.SH",
                    "name": "奕瑞科技",
                    "theme_label": "中盘制造主线",
                    "evidence_quality": "中等偏弱",
                }
            ]
        ),
        tomorrow_focus=["继续观察"],
    )

    assert "换仓内容：688295.SH 中复神鹰 -> 688301.SH 奕瑞科技" in notes


def test_holding_advice_line_includes_cost_and_pnl():
    line = tracker._format_holding_advice_line(
        SimpleNamespace(
            symbol="002008.SZ",
            name="大族激光",
            recommended_action="继续持有",
            position_role="稳定核心",
            current_price=148.66,
            buy_price=65.3,
            buy_value=137130.0,
            unrealized_pnl=175056.0,
            unrealized_pnl_pct=1.27657,
            stage_stop_price=112.65,
            stage_target_price=156.61,
            delta_vs_source_record=-1974.0,
            rank_full_market=62,
            today_change_pct=4.26,
        )
    )

    assert "持有成本 `65.30`（成本金额 `137,130.00 元`）" in line
    assert "浮动 PNL `+175,056.00 元`（+127.66%）" in line


def test_holding_snapshot_set_includes_cost_and_pnl():
    frame = pd.DataFrame(
        [
            {
                "symbol": "002008.SZ",
                "name": "大族激光",
                "buy_price": 65.3,
                "unrealized_pnl": 175056.0,
                "unrealized_pnl_pct": 1.27657,
            }
        ]
    )

    text = tracker._format_holding_snapshot_set(frame)

    assert "002008.SZ(大族激光) 持有成本 `65.30`" in text
    assert "PNL `+175,056.00 元`（+127.66%）" in text


def test_run_tracker_renders_formal_diagnostics_without_changing_action(monkeypatch, tmp_path):
    ledger = pd.DataFrame(
        [
            {
                "symbol": "601869.SH",
                "name": "长飞光纤",
                "shares": 300,
                "avg_cost": 100.0,
                "cost_basis": 30000.0,
                "current_price": 120.0,
                "current_value": 36000.0,
                "unrealized_pnl": 6000.0,
                "unrealized_pnl_pct": 0.2,
                "market_weight": 0.2,
                "stage_target_price": 140.0,
                "stage_stop_price": 100.0,
                "thesis_status": "核心持有",
            }
        ]
    )
    manifest = {"timestamp": "20260422_0942", "capital_cny": 1_000_000.0}
    pnl = pd.DataFrame([{"cash_after": 100000.0, "total_value_after": 1_100_000.0}])

    monkeypatch.setattr(
        tracker,
        "_load_previous_record",
        lambda base_dir, source_record=None: (ledger, manifest, pnl),
    )
    monkeypatch.setattr(
        tracker,
        "get_market_settings",
        lambda _market: SimpleNamespace(data_dir=str(tmp_path / "cn_market_full")),
    )

    completeness = {
        "complete": False,
        "latest_trade_date": "20260427",
        "strict_trade_date": "20260427",
        "stable_trade_date": "20260426",
        "effective_target_trade_date": "20260427",
        "freshness_mode": "strict",
        "coverage_ratio": 0.0,
        "coverage_complete_count": 0,
        "expected_scope_count": 7302,
        "blocking_incomplete_count": 7302,
        "pre_listing_symbols": [],
        "categories_checked": ["full_a", "hs300", "zz500", "zz1000"],
        "categories": {
            "full_a": {
                "expected": 1,
                "latest_trade_date": "20260427",
                "date_counts": {"20260424": 1},
                "coverage_complete_count": 0,
                "blocking_incomplete_count": 1,
                "suspended_stale_symbols": [],
                "blocking_missing_symbols": [],
                "blocking_stale_symbols": [{"symbol": "601869.SH", "latest_local_date": "20260424"}],
            },
            "hs300": {
                "expected": 1,
                "latest_trade_date": "20260427",
                "date_counts": {"20260424": 1},
                "coverage_complete_count": 0,
                "blocking_incomplete_count": 1,
                "suspended_stale_symbols": [],
                "blocking_missing_symbols": [],
                "blocking_stale_symbols": [{"symbol": "601869.SH", "latest_local_date": "20260424"}],
            },
            "zz500": {
                "expected": 0,
                "latest_trade_date": "20260427",
                "date_counts": {},
                "coverage_complete_count": 0,
                "blocking_incomplete_count": 0,
                "suspended_stale_symbols": [],
                "blocking_missing_symbols": [],
                "blocking_stale_symbols": [],
            },
            "zz1000": {
                "expected": 0,
                "latest_trade_date": "20260427",
                "date_counts": {},
                "coverage_complete_count": 0,
                "blocking_incomplete_count": 0,
                "suspended_stale_symbols": [],
                "blocking_missing_symbols": [],
                "blocking_stale_symbols": [],
            },
        },
    }

    class _FakeDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def load_components(self):
            return {"full_a": ["601869.SH"], "hs300": ["601869.SH"], "zz500": [], "zz1000": []}

        def build_completeness_report(self, components=None, allowed_stale_symbols=None):
            return completeness

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)
    monkeypatch.setattr(
        tracker,
        "_compute_full_market_metrics",
        lambda components, data_root, latest_trade_date: pd.DataFrame(
            [
                {
                    "symbol": "601869.SH",
                    "name": "长飞光纤",
                    "category": "hs300",
                    "ret1": 0.02,
                    "ret5": 0.03,
                    "ret20": 0.12,
                    "ret60": 0.20,
                    "close_vs_ma20": 0.05,
                    "ma20_vs_ma60": 0.03,
                    "ma60_vs_ma120": 0.01,
                    "dd20": -0.02,
                    "latest_close": 120.0,
                    "stage_target_price": 140.0,
                    "stage_stop_price": 100.0,
                    "score_full_market": 0.91,
                    "rank_full_market": 8,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        tracker,
        "_compute_category_breadth",
        lambda category, symbols, data_root, latest_trade_date, completeness_report: {
            "ret1_positive_ratio": 0.2,
            "ret20_positive_ratio": 0.5,
            "ma20_gt_ma60_ratio": 0.3,
            "avg_ret1": 0.0,
            "avg_ret20": 0.0,
            "avg_ret60": 0.0,
            "latest_count": len(symbols),
            "expected": len(symbols),
            "suspended_stale_count": 0,
        },
    )
    monkeypatch.setattr(
        tracker,
        "_fetch_tencent_quotes",
        lambda codes: {
            **{
                code: {
                    "quote_code": code,
                    "name": code,
                    "current": 100.0,
                    "prev_close": 99.0,
                    "open": 99.5,
                    "high": 101.0,
                    "low": 98.5,
                    "time": "20260427101603",
                    "change": 1.0,
                    "change_pct": 1.01,
                }
                for code in tracker.INDEX_QUOTES.keys()
            },
            "sh601869": {
                "quote_code": "sh601869",
                "name": "长飞光纤",
                "current": 120.0,
                "prev_close": 118.0,
                "open": 119.0,
                "high": 121.0,
                "low": 117.0,
                "time": "20260427101603",
                "change": 2.0,
                "change_pct": 1.69,
            },
        },
    )
    monkeypatch.setattr(
        tracker,
        "_run_unified_review_mainline_for_holdings",
        lambda source_ledger, latest_trade_date, source_record: {
            "reviewed_symbols": ["601869.SH"],
            "by_symbol": {
                "601869.SH": {
                    "recommendation": {
                        "action": "hold",
                        "confidence": 0.0,
                        "one_line_conclusion": "继续持有。",
                        "risk_flags": [],
                    },
                    "ic_hint": {"action": "hold", "confidence_hint": 0.0, "thesis": "继续持有。"},
                    "master_hint": {"action": "hold", "confidence_hint": 0.0},
                    "llm_attempt_summary": {"call_count": 1, "success_count": 1, "failed_count": 0, "fallback_count": 0, "total_tokens": 8, "estimated_cost_usd": 0.001},
                    "llm_effective_summary": {"call_count": 1, "success_count": 1, "failed_count": 0, "fallback_count": 0, "total_tokens": 8, "estimated_cost_usd": 0.001},
                    "llm_session_id": "session-1",
                    "llm_degraded": False,
                    "llm_degraded_reason": "",
                    "reviewed_branch_verdicts": {
                        "fundamental": {
                            "metadata": {
                                "data_quality": {
                                    "coverage_ratio": 0.33,
                                    "missing_modules": {"601869.SH": ["forecast_revision", "ownership"]},
                                    "snapshot_quality_by_symbol": {
                                        "601869.SH": {
                                            "forecast_revision": {"provider_missing": True, "missing_scope": "global"},
                                            "ownership": {"snapshot_missing": True, "missing_scope": "symbol"},
                                        }
                                    },
                                },
                                "module_coverage": {},
                            }
                        },
                        "kline": {
                            "action": "sell",
                            "metadata": {
                                "evaluator_name": "placeholder_llm_reviewer",
                                "llm_ready": False,
                                "model_components": {"chronos": {"runtime_mode": "error_fallback"}},
                            },
                            "diagnostic_notes": ["fallback path engaged"],
                        },
                        "intelligence": {
                            "action": "hold",
                            "metadata": {"branch_mode": "structured_intelligence_fusion"},
                            "coverage_notes": ["legacy batch retired"],
                            "investment_risks": ["智能融合当前未调用旧 batch pipeline，文本证据为候选层可扩展能力。"],
                        },
                    },
                    "branch_overlays": {"kline": {"action": "sell"}},
                    "report_excerpt": "bearish conclusion",
                }
            },
            "degraded_symbols": {},
            "llm_attempt_summary": {"call_count": 1, "success_count": 1, "failed_count": 0, "fallback_count": 0, "total_tokens": 8, "estimated_cost_usd": 0.001},
            "llm_effective_summary": {"call_count": 1, "success_count": 1, "failed_count": 0, "fallback_count": 0, "total_tokens": 8, "estimated_cost_usd": 0.001},
            "model_role_metadata": {"resolved_branch_model": "deepseek-chat", "resolved_master_model": "moonshot-v1-128k"},
            "fallback_reasons": [],
            "session_ids": {"601869.SH": "session-1"},
        },
    )

    args = argparse.Namespace(
        base_dir=str(tmp_path / "strategy_records"),
        years=7,
        max_rounds=0,
        source_record=None,
        allowed_stale_symbols=[],
    )
    factor_shadow_status = tracker.load_factor_library_shadow_status(
        root_dir=CANONICAL_FACTOR_FIXTURE_ROOT,
        as_of="2026-05-01",
    )
    monkeypatch.setattr(
        tracker,
        "load_factor_library_shadow_status",
        lambda **_kwargs: factor_shadow_status,
    )
    result = tracker.run_tracker(args)

    assert result["action_taken_today"] is False
    assert result["report_guardrail_label"] == "no_action_evidence_impaired"
    run_dir = tmp_path / "strategy_records" / result["timestamp"]
    report_text = (run_dir / "analysis_report.md").read_text(encoding="utf-8")
    manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "#### 5.4.1 决策诊断" in report_text
    assert "stale_snapshot" in report_text
    assert "placeholder_kline_evaluator" in report_text
    assert "601869.SH(长飞光纤) 持有成本 `100.00`，PNL `+6,000.00 元`（+20.00%）" in report_text
    assert "#### 5.4.3 证据质量与工程诊断" in report_text
    assert "### 5.7 因子库状态（只读影子观察）" in report_text
    assert "This factor library status is read-only" in report_text
    assert "does not alter stock selection, portfolio construction, RiskGuard" in report_text
    assert "| Verdict | `fail` |" in report_text
    assert "| `production_factor_count` | 2 |" in report_text
    assert "factor_expired_value_v1" in report_text
    advice_section = report_text.split("#### 5.4.2 正式建议", 1)[1].split(
        "#### 5.4.3 证据质量与工程诊断",
        1,
    )[0]
    assert "诊断码" not in advice_section
    assert "仲裁说明" not in advice_section
    assert "placeholder_kline_evaluator" not in advice_section
    assert "provider_missing" not in advice_section
    assert manifest_payload["action_taken_today"] is False
    assert manifest_payload["formal_diagnostics"]["decision_guardrail"]["display_label"] == "no_action_evidence_impaired"
    orders_payload = (run_dir / "orders.csv").read_text(encoding="utf-8-sig")
    assert orders_payload == "timestamp,action,symbol,name,shares,price,trade_value,realized_pnl,reason\n"
    assert (run_dir / "orders.csv").exists()
    assert (run_dir / "holdings_review.csv").exists()
