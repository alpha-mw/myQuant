from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_investor.monitoring.cn_aggressive_portfolio_tracker as tracker


class _Sentinel(Exception):
    pass


def test_build_parser_accepts_allowed_stale_symbols():
    parser = tracker.build_parser()

    args = parser.parse_args(["--allowed-stale-symbols", "601989.SH", "603000.SH"])

    assert args.allowed_stale_symbols == ["601989.SH", "603000.SH"]


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
