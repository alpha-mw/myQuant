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


def _fake_market_metrics_bundle(
    *,
    full_metrics: pd.DataFrame | None = None,
    breadth: dict[str, object] | None = None,
    status: str = "blocking_generated",
) -> tracker.MarketMetricsBundle:
    if full_metrics is None:
        full_metrics = pd.DataFrame(columns=sorted(tracker.MARKET_METRICS_REQUIRED_COLUMNS))
    if breadth is None:
        breadth = {
            category: {
                "ret1_positive_ratio": 0.0,
                "ret20_positive_ratio": 0.0,
                "ma20_gt_ma60_ratio": 0.0,
                "avg_ret1": 0.0,
                "avg_ret20": 0.0,
                "avg_ret60": 0.0,
                "latest_count": 0,
                "expected": 0,
                "suspended_stale_count": 0,
            }
            for category in tracker.MARKET_METRICS_CATEGORIES
        }
    return tracker.MarketMetricsBundle(
        full_metrics=full_metrics,
        breadth=breadth,
        cache_meta={
            "schema_version": tracker.MARKET_METRICS_CACHE_SCHEMA_VERSION,
            "status": status,
            "cache_hit": status == "cache_hit",
            "cache_dir": "/tmp/fake-market-metrics-cache",
            "row_count": int(len(full_metrics)),
            "compute_elapsed_sec": 0.0,
            "snapshot_id": "snap-test",
            "analysis_trade_date": "20260407",
            "components_fingerprint": "fingerprint",
        },
    )


def _fake_empty_candidate_dag_status() -> dict[str, object]:
    return {
        "candidate_generation_status": "blocked",
        "blocker": "candidate_dag_incomplete",
        "required_branches": list(tracker.REQUIRED_DAG_BRANCHES),
        "candidate_source": "v13_full_market_dag",
        "dag_pipeline": {
            "universe": "full_a",
            "deterministic_funnel": True,
            "candidate_level_four_branch": True,
            "bayesian_shortlist": True,
            "riskguard_ic_portfolio_constructor": True,
            "bayesian_record_count": 0,
            "shortlist_count": 0,
            "portfolio_target_count": 0,
        },
        "candidate_dag_four_branch_compliance": {
            "complete": False,
            "evaluated_symbols": [],
            "accepted_symbols": [],
            "present_branch_by_symbol": {},
            "missing_branch_by_symbol": {},
            "required_branches": list(tracker.REQUIRED_DAG_BRANCHES),
        },
        "error": "unit_test_stub",
    }


def _patch_parquet_completeness(
    monkeypatch,
    completeness: dict[str, object],
    *,
    assert_allowed_stale_symbols: list[str] | None = None,
) -> None:
    def _fake_parquet_completeness(**kwargs):
        if assert_allowed_stale_symbols is not None:
            assert kwargs.get("allowed_stale_symbols") == assert_allowed_stale_symbols
        return dict(completeness)

    monkeypatch.setattr(
        tracker,
        "build_parquet_canonical_completeness_report",
        _fake_parquet_completeness,
    )


@pytest.fixture(autouse=True)
def _stub_candidate_level_v13_dag(monkeypatch):
    monkeypatch.setattr(
        tracker,
        "_run_candidate_level_v13_dag",
        lambda **_kwargs: (pd.DataFrame(dtype=object), _fake_empty_candidate_dag_status(), {}),
    )


def test_build_parser_accepts_allowed_stale_symbols():
    parser = tracker.build_parser()

    args = parser.parse_args(["--allowed-stale-symbols", "601989.SH", "603000.SH"])

    assert args.allowed_stale_symbols == ["601989.SH", "603000.SH"]

    debug_args = parser.parse_args(["--skip-market-metrics-prewarm"])
    assert debug_args.skip_market_metrics_prewarm is True


def test_parquet_canonical_completeness_drives_analysis_date_from_reader():
    class _FakeReader:
        def snapshot(self):
            return {
                "healthy": True,
                "snapshot_id": "snap-20260618",
                "latest_complete_trade_date": "20260618",
                "latest_trade_date": "20260618",
                "table_root": "/tmp/parquet/cn/bars",
                "serving_root": "/tmp/parquet_serving/cn/bars",
                "manifest_path": "/tmp/parquet/cn/_snapshots/snap-20260618.json",
                "latest_pointer_path": "/tmp/parquet/cn/_latest.json",
            }

        def read_cross_section(self, trade_date, **kwargs):
            assert trade_date == "20260618"
            assert kwargs["universe_key"] == "full_a"
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                    "trade_date": ["20260618", "20260618", "20260618"],
                }
            )

    report = tracker.build_parquet_canonical_completeness_report(
        reader=_FakeReader(),
        components={
            "full_a": ["000001.SZ", "000002.SZ"],
            "hs300": ["000001.SZ"],
            "zz500": ["000002.SZ"],
            "zz1000": [],
        },
        categories=["full_a", "hs300", "zz500", "zz1000"],
    )

    assert report["source"] == "strict_parquet_canonical"
    assert report["complete"] is True
    assert report["latest_complete_trade_date"] == "20260618"
    assert report["coverage_ratio"] == 1.0
    assert tracker._resolve_analysis_trade_date(report) == "20260618"
    assert report["resolver"]["physical_directories_used_for_full_a"] == []


def test_realtime_execution_price_accepts_non_current_realtime_field():
    quote = {
        "source": "unit_test_quote",
        "quote_timestamp": "2026-06-17 10:30:00",
        "last": 42.8,
        "open": 41.9,
        "high": 43.2,
        "low": 41.6,
        "prev_close": 41.7,
    }

    price, field = tracker._resolve_realtime_execution_price(quote)

    assert price == pytest.approx(42.8)
    assert field == "last"


def test_realtime_execution_price_accepts_declared_realtime_field():
    quote = {
        "source": "unit_test_quote",
        "quote_timestamp": "2026-06-17 10:30:00",
        "execution_price_field": "deal_price",
        "deal_price": 42.8,
        "open": 41.9,
        "high": 43.2,
        "low": 41.6,
        "prev_close": 41.7,
    }

    price, field = tracker._resolve_realtime_execution_price(quote)

    assert price == pytest.approx(42.8)
    assert field == "deal_price"


def test_realtime_execution_price_rejects_static_daily_price_only():
    quote = {
        "latest_close": 42.8,
        "prev_close": 41.7,
        "open": 41.9,
        "high": 43.2,
        "low": 41.6,
    }

    price, field = tracker._resolve_realtime_execution_price(quote)

    assert price == 0.0
    assert field == ""


def test_parse_quote_payload_marks_current_as_realtime_price():
    parts = [""] * 35
    parts[1] = "测试股份"
    parts[3] = "10.50"
    parts[4] = "10.00"
    parts[5] = "10.10"
    parts[30] = "2026-06-17 10:30:00"
    parts[31] = "0.50"
    parts[32] = "5.00"
    parts[33] = "10.80"
    parts[34] = "10.00"

    parsed = tracker._parse_quote_payload('v_sh600000="' + "~".join(parts) + '";')

    assert parsed is not None
    assert parsed["source"] == "tencent_realtime_quote"
    assert parsed["realtime_price"] == pytest.approx(10.5)
    assert parsed["realtime_price_field"] == "current"
    assert parsed["quote_timestamp"] == "2026-06-17 10:30:00"


def test_load_previous_record_ignores_cache_directories(tmp_path):
    base_dir = tmp_path / "strategy_records"
    base_dir.mkdir()
    (base_dir / "_cache").mkdir()
    record_dir = base_dir / "20260612_1042"
    record_dir.mkdir()
    (record_dir / "manifest.json").write_text(
        json.dumps({"timestamp": "20260612_1042", "capital_cny": 1_000_000.0}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "symbol": "600487.SH",
                "name": "亨通光电",
                "shares": 100,
                "avg_cost": 1.0,
                "current_value": 120.0,
            }
        ]
    ).to_parquet(record_dir / "ledger_after_manual_switch.parquet", index=False)
    (record_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "rejected_no_execution_data_gate_and_prepare_switch_only",
                "next_ledger_path": "ledger_after_manual_switch.parquet",
                "cash_after": 25.0,
                "total_value_after": 145.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"cash_after": 10.0, "total_value_after": 130.0}]).to_parquet(
        record_dir / "pnl_summary.parquet",
        index=False,
    )

    ledger, manifest, pnl_summary = tracker._load_previous_record(base_dir)

    assert manifest["timestamp"] == "20260612_1042"
    assert manifest["effective_manual_ledger_path"].endswith("ledger_after_manual_switch.parquet")
    assert ledger.iloc[0]["symbol"] == "600487.SH"
    assert pnl_summary.iloc[0]["cash_after"] == 25.0
    assert pnl_summary.iloc[0]["total_value_after"] == 145.0


def test_load_previous_record_rejects_legacy_pnl_summary_csv(tmp_path):
    base_dir = tmp_path / "strategy_records"
    base_dir.mkdir()
    record_dir = base_dir / "20260617_0932"
    record_dir.mkdir()
    (record_dir / "manifest.json").write_text(
        json.dumps({"timestamp": "20260617_0932", "capital_cny": 1_000_000.0}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "name": "奕瑞科技",
                "shares": 700,
                "avg_cost": 164.24,
            }
        ]
    ).to_parquet(record_dir / "ledger_after_manual_switch.parquet", index=False)
    (record_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "rejected_no_fill_carry_forward",
                "next_ledger_path": "ledger_after_manual_switch.parquet",
                "cash_after": 76_796.0,
                "total_value_after": 1_410_977.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"cash_after": 10.0, "total_value_after": 130.0}]).to_csv(
        record_dir / "pnl_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pytest.raises(RuntimeError, match="正式记录"):
        tracker._load_previous_record(base_dir)


def test_load_previous_record_resolves_manual_ledger_parquet_sidecar_from_legacy_manifest(tmp_path):
    base_dir = tmp_path / "strategy_records"
    base_dir.mkdir()
    record_dir = base_dir / "20260617_0932"
    record_dir.mkdir()
    (record_dir / "manifest.json").write_text(
        json.dumps({"timestamp": "20260617_0932", "capital_cny": 1_000_000.0}),
        encoding="utf-8",
    )
    ledger_df = pd.DataFrame(
        [
            {
                "symbol": "600487.SH",
                "name": "亨通光电",
                "shares": 1200,
                "avg_cost": 50.83,
            }
        ]
    )
    ledger_df.to_csv(record_dir / "ledger_after_manual_switch.csv", index=False, encoding="utf-8-sig")
    ledger_df.to_parquet(record_dir / "ledger_after_manual_switch.parquet", index=False)
    (record_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "rejected_no_execution_data_gate_dag_incomplete_quote_unverified",
                "next_ledger_path": str(record_dir / "ledger_after_manual_switch.csv"),
                "cash_after": 76_796.0,
                "total_value_after": 1_410_977.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"cash_after": 10.0, "total_value_after": 130.0}]).to_parquet(
        record_dir / "pnl_summary.parquet",
        index=False,
    )

    ledger, manifest, pnl_summary = tracker._load_previous_record(base_dir)

    assert manifest["effective_manual_ledger_path"].endswith("ledger_after_manual_switch.parquet")
    assert ledger.iloc[0]["symbol"] == "600487.SH"
    assert int(ledger.iloc[0]["shares"]) == 1200
    assert pnl_summary.iloc[0]["cash_after"] == 76_796.0


def test_load_previous_record_accepts_effective_manual_ledger_csv_without_parquet_sidecar(tmp_path):
    base_dir = tmp_path / "strategy_records"
    base_dir.mkdir()
    record_dir = base_dir / "20260618_0934"
    record_dir.mkdir()
    (record_dir / "manifest.json").write_text(
        json.dumps({"timestamp": "20260618_0934", "capital_cny": 1_000_000.0}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "name": "奕瑞科技",
                "shares": 700,
                "avg_cost": 164.24,
            }
        ]
    ).to_csv(record_dir / "ledger_after_manual_switch.csv", index=False, encoding="utf-8-sig")
    (record_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "rejected_no_execution_data_gate_quote_unverified_carry_forward",
                "next_ledger_path": str(record_dir / "ledger_after_manual_switch.csv"),
                "cash_after": 276_709.0,
                "total_value_after": 1_684_374.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"cash_after": 10.0, "total_value_after": 130.0}]).to_parquet(
        record_dir / "pnl_summary.parquet",
        index=False,
    )

    ledger, manifest, pnl_summary = tracker._load_previous_record(base_dir)

    assert manifest["effective_manual_ledger_path"].endswith("ledger_after_manual_switch.csv")
    assert ledger.iloc[0]["symbol"] == "688301.SH"
    assert int(ledger.iloc[0]["shares"]) == 700
    assert pnl_summary.iloc[0]["cash_after"] == 276_709.0
    assert pnl_summary.iloc[0]["total_value_after"] == 1_684_374.0


def test_load_previous_record_skips_invalidated_manual_manifest(tmp_path):
    base_dir = tmp_path / "strategy_records"
    base_dir.mkdir()
    valid_dir = base_dir / "20260604_1302"
    valid_dir.mkdir()
    invalid_dir = base_dir / "20260605_1025"
    invalid_dir.mkdir()

    for record_dir in (valid_dir, invalid_dir):
        (record_dir / "manifest.json").write_text(
            json.dumps({"timestamp": record_dir.name, "capital_cny": 1_000_000.0}),
            encoding="utf-8",
        )
        pd.DataFrame([{"cash_after": 10.0, "total_value_after": 130.0}]).to_parquet(
            record_dir / "pnl_summary.parquet",
            index=False,
        )

    pd.DataFrame([{"symbol": "688301.SH", "shares": 700, "avg_cost": 164.24}]).to_parquet(
        valid_dir / "ledger_after_manual_switch.parquet",
        index=False,
    )
    (valid_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "filled_local_manual",
                "next_ledger_path": "ledger_after_manual_switch.parquet",
                "cash_after": 42.0,
                "total_value_after": 200.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"symbol": "600903.SH", "shares": 100, "avg_cost": 9.83}]).to_parquet(
        invalid_dir / "ledger_after_manual_switch.parquet",
        index=False,
    )
    (invalid_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "invalidated_price_basis_no_execution",
                "next_ledger_path": "ledger_after_manual_switch.parquet",
                "cash_after": 99.0,
                "total_value_after": 999.0,
            }
        ),
        encoding="utf-8",
    )

    ledger, manifest, pnl_summary = tracker._load_previous_record(base_dir)

    assert manifest["timestamp"] == "20260605_1025"
    assert manifest["effective_manual_manifest_path"].endswith(
        "20260604_1302/manual_execution_manifest.json"
    )
    assert ledger.iloc[0]["symbol"] == "688301.SH"
    assert pnl_summary.iloc[0]["cash_after"] == 42.0


def test_load_previous_record_prefers_latest_manual_manifest_recorded_at(tmp_path):
    base_dir = tmp_path / "strategy_records"
    base_dir.mkdir()
    filled_dir = base_dir / "20260703_0932"
    carry_dir = base_dir / "20260703_1100"
    filled_dir.mkdir()
    carry_dir.mkdir()

    for record_dir in (filled_dir, carry_dir):
        (record_dir / "manifest.json").write_text(
            json.dumps({"timestamp": record_dir.name, "capital_cny": 1_000_000.0}),
            encoding="utf-8",
        )
        pd.DataFrame([{"cash_after": 10.0, "total_value_after": 130.0}]).to_parquet(
            record_dir / "pnl_summary.parquet",
            index=False,
        )

    pd.DataFrame(
        [
            {"symbol": "603078.SH", "name": "江化微", "shares": 1300, "avg_cost": 52.10},
            {"symbol": "002008.SZ", "name": "大族激光", "shares": 1400, "avg_cost": 65.30},
        ]
    ).to_parquet(filled_dir / "ledger_after_manual_switch.parquet", index=False)
    (filled_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "filled_local_manual_paper_rebalance",
                "recorded_at": "2026-07-03 11:06:56 CST",
                "next_ledger_path": "ledger_after_manual_switch.parquet",
                "cash_after": 1_001_341.0,
                "total_value_after": 1_674_627.0,
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"symbol": "688519.SH", "name": "南亚新材", "shares": 900, "avg_cost": 130.2973},
            {"symbol": "688525.SH", "name": "佰维存储", "shares": 700, "avg_cost": 238.77},
        ]
    ).to_parquet(carry_dir / "ledger_after_manual_switch.parquet", index=False)
    (carry_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "no_action_carry_forward",
                "recorded_at": "2026-07-03 11:00:39 CST",
                "next_ledger_path": "ledger_after_manual_switch.parquet",
                "cash_after": 429_343.0,
                "total_value_after": 1_668_845.0,
            }
        ),
        encoding="utf-8",
    )

    ledger, manifest, pnl_summary = tracker._load_previous_record(
        base_dir,
        source_record="20260703_1100",
    )

    assert manifest["timestamp"] == "20260703_1100"
    assert manifest["effective_manual_manifest_path"].endswith(
        "20260703_0932/manual_execution_manifest.json"
    )
    assert set(ledger["symbol"]) == {"603078.SH", "002008.SZ"}
    assert pnl_summary.iloc[0]["cash_after"] == 1_001_341.0
    assert pnl_summary.iloc[0]["total_value_after"] == 1_674_627.0


def test_switch_plan_rejects_price_strength_candidate_without_candidate_dag():
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

    assert switch_plan.empty


def test_candidate_pool_from_v13_dag_requires_candidate_level_four_branches():
    complete_branches = {
        branch: {"branch_name": branch, "final_score": 0.8, "final_confidence": 0.7}
        for branch in tracker.REQUIRED_DAG_BRANCHES
    }
    incomplete_branches = {
        "quant": {"branch_name": "quant", "final_score": 0.9, "final_confidence": 0.8},
        "macro": {"branch_name": "macro", "final_score": 0.7, "final_confidence": 0.6},
    }
    dag_artifacts = {
        "funnel_summary": {
            "candidate_count": 2,
            "funnel_metadata": {
                "theme_pool_status": "applied",
                "theme_pool": {
                    "enabled": True,
                    "required": True,
                    "status": "applied",
                    "policy": {"regime": "趋势下跌", "top_themes": 4},
                    "admitted_theme_count": 1,
                    "rejected_theme_count": 1,
                    "core_symbol_count": 1,
                    "residual_symbol_count": 1,
                    "excluded_symbol_count": 1,
                    "admitted_themes": [
                        {
                            "theme_id": "industry::半导体",
                            "theme_rank_score": 0.91,
                            "theme_score": 88.0,
                            "phase": "expansion",
                        }
                    ],
                    "rejected_themes": [
                        {
                            "theme_id": "industry::白酒",
                            "reason": "below_policy_threshold",
                        }
                    ],
                    "score_source": "smoothed",
                    "fallback_to_raw_score": True,
                    "symbols": {
                        "688301.SH": {
                            "admitted": True,
                            "source": "core",
                            "primary_theme_id": "industry::半导体",
                            "theme_pool_score": 0.88,
                            "theme_policy_regime": "趋势下跌",
                            "theme_pool_reason": "admitted",
                        },
                        "002409.SZ": {
                            "admitted": True,
                            "source": "residual",
                            "primary_theme_id": "industry::半导体",
                            "theme_pool_score": 0.66,
                            "theme_policy_regime": "趋势下跌",
                            "theme_pool_reason": "admitted",
                        },
                        "600519.SH": {
                            "admitted": False,
                            "source": "none",
                            "primary_theme_id": "industry::白酒",
                            "theme_pool_score": 0.20,
                            "theme_policy_regime": "趋势下跌",
                            "theme_pool_reason": "theme_pool_theme_not_admitted",
                        },
                    },
                },
            },
        },
        "symbol_research_packets": {
            "688301.SH": {
                "symbol": "688301.SH",
                "company_name": "奕瑞科技",
                "category": "full_a",
                "branch_verdicts": complete_branches,
            },
            "002409.SZ": {
                "symbol": "002409.SZ",
                "company_name": "雅克科技",
                "category": "full_a",
                "branch_verdicts": incomplete_branches,
            },
        },
        "shortlist": [
            {
                "symbol": "688301.SH",
                "company_name": "奕瑞科技",
                "category": "full_a",
                "rank_score": 0.86,
                "confidence": 0.72,
                "expected_upside": 0.16,
                "suggested_weight": 0.08,
                "risk_flags": ["估值波动"],
                "rationale": ["四分支共振"],
            },
            {
                "symbol": "002409.SZ",
                "company_name": "雅克科技",
                "category": "full_a",
                "rank_score": 0.93,
                "confidence": 0.81,
                "expected_upside": 0.22,
                "suggested_weight": 0.10,
            },
        ],
        "bayesian_records": [
            {
                "symbol": "688301.SH",
                "posterior_action_score": 0.74,
                "posterior_win_rate": 0.62,
                "posterior_expected_alpha": 0.08,
                "posterior_confidence": 0.71,
                "rank": 2,
            },
            {
                "symbol": "002409.SZ",
                "posterior_action_score": 0.92,
                "posterior_win_rate": 0.68,
                "posterior_expected_alpha": 0.12,
                "posterior_confidence": 0.80,
                "rank": 1,
            },
        ],
        "portfolio_decision": {
            "target_weights": {"688301.SH": 0.08, "002409.SZ": 0.10},
            "target_positions": {"688301.SH": 0.08, "002409.SZ": 0.10},
            "risk_constraints": {"risk_decision": {"status": "success"}},
        },
    }

    candidate_pool, status = tracker._build_candidate_pool_from_v13_dag(
        dag_artifacts=dag_artifacts,
        held_symbols=[],
    )

    assert candidate_pool["symbol"].tolist() == ["688301.SH"]
    row = candidate_pool.iloc[0]
    assert row["candidate_source"] == "v13_full_market_dag"
    assert row["candidate_dag_four_branch_complete"] is True
    assert row["present_branches"] == "quant,fundamental,intelligence,macro"
    assert row["portfolio_target_weight"] == pytest.approx(0.08)
    assert "score_full_market" not in candidate_pool.columns
    assert "ret20" not in candidate_pool.columns
    assert status["candidate_generation_status"] == "complete"
    assert status["candidate_dag_four_branch_compliance"]["missing_branch_by_symbol"] == {
        "002409.SZ": ["fundamental", "intelligence"]
    }
    theme_pool = status["theme_candidate_pool"]
    assert theme_pool["status"] == "applied"
    assert theme_pool["policy_regime"] == "趋势下跌"
    assert theme_pool["admitted_theme_count"] == 1
    assert theme_pool["rejected_theme_count"] == 1
    assert theme_pool["core_symbol_count"] == 1
    assert theme_pool["residual_symbol_count"] == 1
    assert theme_pool["excluded_symbol_count"] == 1
    assert theme_pool["excluded_reason_counts"] == {"theme_pool_theme_not_admitted": 1}
    assert theme_pool["admitted_symbols_by_source_sample"]["core"] == ["688301.SH"]
    assert theme_pool["admitted_symbols_by_source_sample"]["residual"] == ["002409.SZ"]
    assert theme_pool["excluded_symbols_by_reason_sample"][
        "theme_pool_theme_not_admitted"
    ] == ["600519.SH"]


def test_write_outputs_persists_theme_pool_audit(tmp_path: Path) -> None:
    run_dir = tmp_path / "strategy_records" / "20260630_theme_pool"
    manifest = {
        "timestamp": "20260630_theme_pool",
        "files": {"analysis_report": "analysis_report.md"},
        "raw_exports": {},
    }
    market_snapshot = {"analysis_trade_date": "20260629"}
    theme_pool_audit = {
        "schema_version": tracker.THEME_POOL_AUDIT_SCHEMA_VERSION,
        "summary": {
            "status": "applied",
            "admitted_theme_count": 1,
            "rejected_theme_count": 1,
            "core_symbol_count": 1,
            "residual_symbol_count": 0,
            "excluded_symbol_count": 1,
            "excluded_reason_counts": {"theme_pool_theme_not_admitted": 1},
        },
        "symbols": {
            "688301.SH": {"admitted": True, "source": "core"},
            "600519.SH": {
                "admitted": False,
                "source": "none",
                "theme_pool_reason": "theme_pool_theme_not_admitted",
            },
        },
    }

    tracker._write_outputs(
        base_dir=tmp_path / "strategy_records",
        run_dir=run_dir,
        report_text="# report\n",
        holdings_review=pd.DataFrame([{"symbol": "688301.SH"}]),
        candidate_pool=pd.DataFrame([{"symbol": "688301.SH"}]),
        switch_plan_df=pd.DataFrame([{"buy_symbol": "688301.SH"}]),
        ledger=pd.DataFrame([{"symbol": "688301.SH"}]),
        orders_df=pd.DataFrame([{"symbol": "688301.SH"}]),
        pnl_summary_df=pd.DataFrame([{"total_value_after": 1.0}]),
        manifest=manifest,
        market_snapshot=market_snapshot,
        theme_pool_audit=theme_pool_audit,
    )

    written_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    written_snapshot = json.loads((run_dir / "market_snapshot.json").read_text(encoding="utf-8"))
    written_audit = json.loads((run_dir / "theme_pool_audit.json").read_text(encoding="utf-8"))

    assert written_manifest["files"]["theme_pool_audit"] == "theme_pool_audit.json"
    assert written_manifest["raw_exports"]["theme_pool_audit"] == (
        "raw_exports/aggressive_portfolio_20260630_theme_pool_formal_theme_pool_audit.json"
    )
    assert written_manifest["theme_pool_audit"]["file"] == "theme_pool_audit.json"
    assert written_snapshot["theme_pool_audit"]["file"] == "theme_pool_audit.json"
    assert written_audit["summary"]["excluded_reason_counts"] == {
        "theme_pool_theme_not_admitted": 1
    }
    assert (
        run_dir
        / "raw_exports"
        / "aggressive_portfolio_20260630_theme_pool_formal_theme_pool_audit.json"
    ).exists()


def test_theme_pool_report_lines_expose_forced_core_candidates() -> None:
    theme_pool_summary = {
        "status": "applied",
        "policy_regime": "震荡高波",
        "admitted_theme_count": 2,
        "natural_admitted_theme_count": 0,
        "forced_theme_count": 2,
        "min_admitted_themes": 2,
        "rejected_theme_count": 18,
        "core_symbol_count": 274,
        "residual_symbol_count": 0,
        "excluded_symbol_count": 5267,
        "excluded_reason_counts": {
            "theme_pool_missing_theme_membership": 5241,
            "theme_pool_theme_not_admitted": 26,
        },
        "admitted_symbols_by_source_sample": {
            "core": ["603078.SH", "688046.SH"],
        },
        "admitted_themes": [
            {
                "theme_id": "industry::半导体",
                "theme_name": "半导体",
                "theme_score": 0.4718297977,
                "theme_rank_score": 0.4639796609,
                "phase": "accumulation",
                "forced": True,
                "force_reason": "forced_top_theme_min_admitted_themes",
                "original_rejection_reason": "theme_pool_theme_quality_gate_failed",
                "quality_flags": ["theme_score_below_threshold", "phase_not_allowed"],
            },
            {
                "theme_id": "industry::生物制药",
                "theme_name": "生物制药",
                "theme_score": 0.4320757745,
                "phase": "accumulation",
                "forced": True,
                "force_reason": "forced_top_theme_min_admitted_themes",
                "original_rejection_reason": "theme_pool_theme_quality_gate_failed",
                "quality_flags": ["theme_score_below_threshold", "phase_not_allowed"],
            },
        ],
        "rejected_themes": [
            {
                "theme_name": "证券",
                "reason": "theme_pool_theme_quality_gate_failed",
                "score": 0.357023681,
                "phase": "unclassified",
            }
        ],
    }
    candidate_pool = pd.DataFrame(
        [
            {
                "symbol": "603078.SH",
                "name": "江化微",
                "codex_recommendation_score": 68,
            }
        ]
    )
    theme_symbols = {
        "603078.SH": {
            "admitted": True,
            "source": "core",
            "primary_theme_id": "industry::半导体",
            "theme_pool_score": 0.395369,
            "theme_pool_reason": "admitted",
        }
    }

    lines = tracker._format_theme_pool_report_lines(
        theme_pool_summary,
        candidate_pool=candidate_pool,
        theme_symbols=theme_symbols,
    )
    text = "\n".join(lines)

    assert "Theme 强制准入说明" in text
    assert "未自然通过 ThemeGatePolicy" in text
    assert "不等同于自然合格主题" in text
    assert "自然通过=false" in text
    assert "强制准入=true" in text
    assert "硬加入=true" in text
    assert "半导体" in text
    assert "生物制药" in text
    assert "original_rejection=theme_pool_theme_quality_gate_failed" in text
    assert "core样本=`603078.SH, 688046.SH`" in text
    assert "603078.SH 江化微" in text
    assert "theme=`industry::半导体`" in text
    assert "source=`core`" in text
    assert "证券" in text


def test_candidate_pool_from_v13_dag_falls_back_to_report_bundle_shortlist():
    complete_branches = {
        branch: {"branch_name": branch, "final_score": 0.8, "final_confidence": 0.7}
        for branch in tracker.REQUIRED_DAG_BRANCHES
    }
    shortlist = [
        {
            "symbol": "688301.SH",
            "company_name": "奕瑞科技",
            "category": "full_a",
            "rank_score": 0.86,
            "confidence": 0.72,
            "expected_upside": 0.16,
            "suggested_weight": 0.08,
            "risk_flags": ["估值波动"],
            "rationale": ["四分支共振"],
        }
    ]
    dag_artifacts = {
        "symbol_research_packets": {
            "688301.SH": {
                "symbol": "688301.SH",
                "company_name": "奕瑞科技",
                "category": "full_a",
                "branch_verdicts": complete_branches,
            },
        },
        "report_bundle": SimpleNamespace(shortlist=shortlist),
        "bayesian_records": [
            {
                "symbol": "688301.SH",
                "posterior_action_score": 0.74,
                "posterior_win_rate": 0.62,
                "posterior_expected_alpha": 0.08,
                "posterior_confidence": 0.71,
                "rank": 1,
            }
        ],
        "portfolio_decision": {
            "target_weights": {"688301.SH": 0.08},
            "target_positions": {"688301.SH": 0.08},
        },
    }

    candidate_pool, status = tracker._build_candidate_pool_from_v13_dag(
        dag_artifacts=dag_artifacts,
        held_symbols=[],
    )

    assert candidate_pool["symbol"].tolist() == ["688301.SH"]
    assert status["candidate_generation_status"] == "complete"
    assert status["blocker"] == ""
    assert status["dag_pipeline"]["shortlist_source"] == "report_bundle.shortlist"
    assert status["dag_pipeline"]["shortlist_fallback_used"] is True
    assert status["dag_pipeline"]["shortlist_artifact_missing"] is False


def test_candidate_pool_from_v13_dag_blocks_when_shortlist_artifact_missing():
    dag_artifacts = {
        "symbol_research_packets": {},
        "bayesian_records": [
            {
                "symbol": "688301.SH",
                "posterior_action_score": 0.74,
                "posterior_win_rate": 0.62,
                "posterior_expected_alpha": 0.08,
                "posterior_confidence": 0.71,
                "rank": 1,
            }
        ],
        "portfolio_decision": {
            "target_weights": {"688301.SH": 0.08},
        },
    }

    candidate_pool, status = tracker._build_candidate_pool_from_v13_dag(
        dag_artifacts=dag_artifacts,
        held_symbols=[],
    )

    assert candidate_pool.empty
    assert status["candidate_generation_status"] == "blocked"
    assert status["blocker"] == "candidate_artifact_shortlist_missing"
    assert status["dag_pipeline"]["shortlist_artifact_missing"] is True
    assert status["dag_pipeline"]["shortlist_source"] == ""
    assert "shortlist artifact missing" in status["error"]


def test_candidate_pool_from_v13_dag_empty_when_no_shortlist_or_positive_targets():
    complete_branches = {
        branch: {"branch_name": branch, "final_score": 0.5, "final_confidence": 0.6}
        for branch in tracker.REQUIRED_DAG_BRANCHES
    }
    dag_artifacts = {
        "symbol_research_packets": {
            "688301.SH": {
                "symbol": "688301.SH",
                "company_name": "奕瑞科技",
                "category": "full_a",
                "branch_verdicts": complete_branches,
            },
        },
        "bayesian_records": [
            {
                "symbol": "688301.SH",
                "posterior_action_score": 0.34,
                "posterior_win_rate": 0.48,
                "posterior_expected_alpha": -0.02,
                "posterior_confidence": 0.52,
                "rank": 1,
            }
        ],
        "portfolio_decision": {
            "target_weights": {"688301.SH": 0.0},
        },
    }

    candidate_pool, status = tracker._build_candidate_pool_from_v13_dag(
        dag_artifacts=dag_artifacts,
        held_symbols=[],
    )

    assert candidate_pool.empty
    assert status["candidate_generation_status"] == "empty"
    assert status["blocker"] == "no_candidate_selected_by_portfolio_constructor"
    assert status["dag_pipeline"]["bayesian_record_count"] == 1
    assert status["dag_pipeline"]["portfolio_target_count"] == 0
    assert status["dag_pipeline"]["shortlist_artifact_missing"] is False
    assert status["error"] == ""


def test_trailing_take_profit_review_sets_explicit_watch_from_entry_date():
    holdings_review = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "shares_before": 100,
                "buy_price": 100.0,
                "current_price": 150.0,
                "stage_stop_price": 120.0,
                "score_full_market": 0.82,
                "market_weight": 0.10,
                "realtime_quote_valid": True,
                "manual_entry_trade_date": "20260101",
            }
        ]
    )

    class _FakeReader:
        def read_symbol_frames(self, symbols, **_kwargs):
            assert symbols == ["000001.SZ"]
            return {
                "000001.SZ": SimpleNamespace(
                    frame=pd.DataFrame(
                        [
                            {"trade_date": "20260101", "high": 100.0, "close": 100.0},
                            {"trade_date": "20260110", "high": 170.0, "close": 166.0},
                            {"trade_date": "20260120", "high": 151.0, "close": 150.0},
                        ]
                    )
                )
            }

    reviewed = tracker._apply_trailing_take_profit_review(
        holdings_review,
        reader=_FakeReader(),
        analysis_trade_date="20260120",
    )
    row = reviewed.iloc[0]

    assert row["trailing_take_profit_status"] == "hold_with_trailing_stop"
    assert bool(row["trailing_take_profit_confirmed"]) is True
    assert row["trailing_profit_peak_price"] == pytest.approx(170.0)
    assert row["profit_giveback_ratio"] == pytest.approx((70.0 - 50.0) / 70.0)
    assert row["trailing_stop_price"] == pytest.approx(156.0)
    assert tracker._position_action(row) == "移动止盈观察"


def test_trailing_take_profit_review_falls_back_to_symbol_serving_history():
    holdings_review = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "shares_before": 100,
                "buy_price": 100.0,
                "current_price": 150.0,
                "stage_stop_price": 120.0,
                "score_full_market": 0.82,
                "market_weight": 0.10,
                "realtime_quote_valid": True,
                "manual_entry_trade_date": "20260101",
            }
        ]
    )

    class _FakeReader:
        def read_symbol_frames(self, symbols, **_kwargs):
            return {"000001.SZ": SimpleNamespace(frame=pd.DataFrame())}

        def read_symbol_frame(self, symbol, **_kwargs):
            assert symbol == "000001.SZ"
            return SimpleNamespace(
                frame=pd.DataFrame(
                    [
                        {"trade_date": "20260101", "high": 100.0, "close": 100.0},
                        {"trade_date": "20260110", "high": 170.0, "close": 166.0},
                    ]
                )
            )

    reviewed = tracker._apply_trailing_take_profit_review(
        holdings_review,
        reader=_FakeReader(),
        analysis_trade_date="20260120",
    )

    assert reviewed.iloc[0]["trailing_profit_peak_price"] == pytest.approx(170.0)
    assert reviewed.iloc[0]["trailing_take_profit_status"] == "hold_with_trailing_stop"


def test_trailing_take_profit_review_missing_entry_date_is_unanchored_watch_not_reduce():
    holdings_review = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "shares_before": 100,
                "buy_price": 100.0,
                "current_price": 120.0,
                "stage_stop_price": 105.0,
                "score_full_market": 0.80,
                "market_weight": 0.10,
                "realtime_quote_valid": True,
            }
        ]
    )

    class _FakeReader:
        def read_symbol_frames(self, symbols, **_kwargs):
            return {
                "000001.SZ": SimpleNamespace(
                    frame=pd.DataFrame(
                        [
                            {"trade_date": "20250101", "high": 200.0, "close": 198.0},
                            {"trade_date": "20260120", "high": 121.0, "close": 120.0},
                        ]
                    )
                )
            }

    reviewed = tracker._apply_trailing_take_profit_review(
        holdings_review,
        reader=_FakeReader(),
        analysis_trade_date="20260120",
    )
    row = reviewed.iloc[0]

    assert row["trailing_take_profit_status"] == "hold_with_trailing_stop"
    assert bool(row["trailing_take_profit_confirmed"]) is False
    assert "unanchored" in row["trailing_take_profit_basis"]
    assert "缺少 ledger 买入日期" in row["trailing_take_profit_reason"]
    assert tracker._position_action(row) == "移动止盈观察"


def test_trailing_take_profit_review_unconfirmed_when_peak_history_unavailable():
    holdings_review = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "shares_before": 100,
                "buy_price": 100.0,
                "current_price": 120.0,
                "stage_stop_price": 105.0,
                "score_full_market": 0.80,
                "market_weight": 0.10,
                "realtime_quote_valid": True,
            }
        ]
    )

    class _BrokenReader:
        def read_symbol_frames(self, symbols, **_kwargs):
            raise RuntimeError("history unavailable")

    reviewed = tracker._apply_trailing_take_profit_review(
        holdings_review,
        reader=_BrokenReader(),
        analysis_trade_date="20260120",
    )
    row = reviewed.iloc[0]

    assert row["trailing_take_profit_status"] == "unconfirmed"
    assert row["trailing_profit_peak_price"] == 0.0
    assert "history unavailable" in row["trailing_take_profit_basis"]


def test_risk_reduction_sell_gate_accepts_sell_only_broken_stop():
    order = tracker.ProposedOrder(
        symbol="688301.SH",
        action="sell",
        shares=100,
        price=114.81,
        trade_value=11481.0,
        realized_pnl=-4943.0,
        reason="formal reduce",
    )
    effective_ledger = pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "name": "奕瑞科技",
                "shares": 700,
                "avg_cost": 164.24,
            }
        ]
    )
    holdings_review = pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "current_price": 114.81,
                "stage_stop_price": 161.04,
                "score_full_market": 0.0,
                "recommended_action": "减仓待确认",
                "reason": "仍低于阶段止损位",
            }
        ]
    )

    allowed, reason = tracker._risk_reduction_sell_gate(
        order=order,
        effective_ledger=effective_ledger,
        holdings_review=holdings_review,
    )

    assert allowed is True
    assert reason == "risk_reduction_sell_eligible_pending_realtime_quote"


def test_risk_reduction_sell_gate_rejects_new_risk_and_invalid_sell_legs():
    effective_ledger = pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "shares": 100,
                "avg_cost": 164.24,
            }
        ]
    )
    holdings_review = pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "current_price": 114.81,
                "stage_stop_price": 161.04,
                "score_full_market": 0.0,
                "recommended_action": "减仓待确认",
            }
        ]
    )

    buy_order = tracker.ProposedOrder(
        symbol="688301.SH",
        action="buy",
        shares=100,
        price=114.81,
        trade_value=11481.0,
        realized_pnl=0.0,
        reason="not allowed",
    )
    allowed, reason = tracker._risk_reduction_sell_gate(
        order=buy_order,
        effective_ledger=effective_ledger,
        holdings_review=holdings_review,
    )
    assert allowed is False
    assert reason == "not_sell_order"

    oversell_order = tracker.ProposedOrder(
        symbol="688301.SH",
        action="sell",
        shares=200,
        price=114.81,
        trade_value=22962.0,
        realized_pnl=-9886.0,
        reason="oversell",
    )
    allowed, reason = tracker._risk_reduction_sell_gate(
        order=oversell_order,
        effective_ledger=effective_ledger,
        holdings_review=holdings_review,
    )
    assert allowed is False
    assert reason == "sell_exceeds_effective_ledger_shares"

    missing_symbol_order = tracker.ProposedOrder(
        symbol="600903.SH",
        action="sell",
        shares=100,
        price=8.75,
        trade_value=875.0,
        realized_pnl=-108.0,
        reason="not in effective ledger",
    )
    allowed, reason = tracker._risk_reduction_sell_gate(
        order=missing_symbol_order,
        effective_ledger=effective_ledger,
        holdings_review=holdings_review,
    )
    assert allowed is False
    assert reason == "symbol_not_in_effective_ledger"


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

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)

    def _fake_parquet_completeness(**kwargs):
        assert kwargs.get("allowed_stale_symbols") == ["601989.SH"]
        raise _Sentinel

    monkeypatch.setattr(
        tracker,
        "build_parquet_canonical_completeness_report",
        _fake_parquet_completeness,
    )

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


def test_run_unified_review_mainline_keeps_non_llm_dag_when_local_llm_disabled(monkeypatch):
    monkeypatch.delenv("MYQUANT_ENABLE_LOCAL_LLM", raising=False)
    monkeypatch.setenv("MYQUANT_DISABLE_LOCAL_LLM", "true")
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
    captured: dict[str, object] = {}

    class _FakeInvestor:
        def __init__(self, **kwargs):
            captured["enable_agent_layer"] = kwargs["enable_agent_layer"]
            self.symbol = kwargs["stock_pool"][0]

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
                llm_usage_session_id="",
                model_role_metadata=SimpleNamespace(
                    local_llm_disabled=True,
                    agent_layer_enabled=False,
                ),
                reviewed_research_by_symbol={
                    self.symbol: {
                        "fundamental": {"branch_name": "fundamental", "action": "hold"},
                        "intelligence": {"branch_name": "intelligence", "action": "hold"},
                    }
                },
                reviewed_branch_summaries={
                    "quant": {
                        "branch_name": "quant",
                        "factor_mode": "governed_mined_factors",
                    },
                    "macro": {"branch_name": "macro", "regime": "neutral"},
                },
                ic_hints_by_symbol={self.symbol: {"action": "hold"}},
                final_strategy=SimpleNamespace(recommendations=[]),
                final_report="deterministic report",
                review_bundle=SimpleNamespace(
                    fallback_reasons=[],
                    branch_overlay_verdicts_by_symbol={},
                    master_hints_by_symbol={},
                ),
            )

    monkeypatch.setattr(tracker, "QuantInvestor", _FakeInvestor)
    ledger = pd.DataFrame([{"symbol": "601869.SH", "cost_basis": 100000.0, "current_value": 120000.0}])

    payload = tracker._run_unified_review_mainline_for_holdings(
        source_ledger=ledger,
        latest_trade_date="20260407",
        source_record="20260408_1403",
    )

    assert captured["enable_agent_layer"] is True
    assert payload["codex_handoff"] is True
    assert payload["local_llm_disabled"] is True
    assert payload["non_llm_dag_executed"] is True
    assert payload["llm_attempt_summary"]["call_count"] == 0
    assert payload["degraded_symbols"] == {}
    assert payload["by_symbol"]["601869.SH"]["codex_handoff"] is True
    assert set(payload["by_symbol"]["601869.SH"]["reviewed_branch_verdicts"]) == set(
        tracker.REQUIRED_DAG_BRANCHES
    )


def test_build_dag_four_branch_compliance_marks_complete_when_all_branches_present():
    branch_signals = {
        "601869.SH": {
            "reviewed_branch_verdicts": {
                branch_name: {"branch_name": branch_name}
                for branch_name in tracker.REQUIRED_DAG_BRANCHES
            }
        }
    }

    result = tracker._build_dag_four_branch_compliance(
        review_symbols=["601869.SH"],
        effective_local_holding_symbols=["601869.SH"],
        branch_signals_by_symbol=branch_signals,
    )

    assert result["complete"] is True
    assert result["status"] == "DAG四分支完整执行"
    assert result["missing_branch_by_symbol"]["601869.SH"] == []


def test_serialize_reviewed_branch_verdicts_backfills_all_canonical_summaries():
    result = SimpleNamespace(
        reviewed_research_by_symbol={},
        reviewed_branch_summaries={
            branch_name: {"branch_name": branch_name, "status": "success"}
            for branch_name in tracker.REQUIRED_DAG_BRANCHES
        },
    )

    reviewed = tracker._serialize_reviewed_branch_verdicts(result, "601869.SH")

    assert set(reviewed) == set(tracker.REQUIRED_DAG_BRANCHES)


def test_serialize_reviewed_branch_verdicts_does_not_invent_missing_branches():
    result = SimpleNamespace(
        reviewed_research_by_symbol={},
        reviewed_branch_summaries={
            "quant": {"branch_name": "quant", "status": "success"},
            "macro": {"branch_name": "macro", "status": "success"},
        },
    )

    reviewed = tracker._serialize_reviewed_branch_verdicts(result, "601869.SH")

    assert set(reviewed) == {"quant", "macro"}


def test_dag_compliance_does_not_mark_governed_quant_limited():
    branch_signals = {
        "688519.SH": {
            "reviewed_branch_verdicts": {
                "quant": {
                    "status": "success",
                    "final_score": 0.0,
                    "final_confidence": 0.58,
                    "evidence": [],
                    "coverage_notes": [
                        "symbols=1",
                        "production_factors=14",
                        "factor_coverage=100.00%",
                    ],
                    "investment_risks": [
                        (
                            "量化分支只消费 production_factor；"
                            "paper/research 因子权重为 0 且不进入选股。"
                        ),
                        "mined_factor_coverage=100.00%",
                    ],
                    "diagnostic_notes": [
                        "global_quant_branch_result",
                        "mined_factor_registry_enforced",
                    ],
                    "metadata": {
                        "factor_mode": "governed_mined_factors",
                        "mined_factor_runtime": {
                            "factor_count": 14,
                            "factors_used": ["pv_volume_stability_15d"],
                            "factor_coverages": {
                                "pv_volume_stability_15d": 1.0,
                            },
                            "applied_to_score": True,
                            "score_weight": 0.05,
                        },
                    },
                },
                "fundamental": {"status": "success"},
                "intelligence": {"status": "success"},
                "macro": {"status": "success"},
            }
        }
    }

    result = tracker._build_dag_four_branch_compliance(
        review_symbols=["688519.SH"],
        effective_local_holding_symbols=[],
        branch_signals_by_symbol=branch_signals,
    )

    assert "quant" not in result["limited_evidence_branch_by_symbol"].get(
        "688519.SH", []
    )


def test_dag_compliance_does_not_mark_substantive_fundamental_notes_limited():
    branch_signals = {
        "688519.SH": {
            "reviewed_branch_verdicts": {
                "quant": {
                    "status": "success",
                    "final_score": 0.1,
                    "final_confidence": 0.5,
                },
                "fundamental": {
                    "status": "success",
                    "final_score": 0.35,
                    "final_confidence": 0.695,
                    "evidence": [{"summary": "FundamentalAgent evidence"}],
                    "coverage_notes": [
                        "盈利预测 全局不可用，已从评分分母剔除（0/1 标的）。",
                        "估值 当前覆盖 0/1 标的，缺失部分仅计入覆盖说明。",
                    ],
                    "metadata": {
                        "module_coverage": {
                            "financial_quality": {
                                "status": "active",
                                "coverage_ratio": 1.0,
                            },
                            "forecast_revision": {
                                "status": "disabled_global",
                                "coverage_ratio": 0.0,
                            },
                            "valuation": {
                                "status": "active",
                                "coverage_ratio": 0.0,
                            },
                            "management_governance": {
                                "status": "active",
                                "coverage_ratio": 1.0,
                            },
                            "ownership": {
                                "status": "active",
                                "coverage_ratio": 1.0,
                            },
                            "document_semantics": {
                                "status": "disabled_global",
                                "coverage_ratio": 0.0,
                            },
                        },
                    },
                },
                "intelligence": {"status": "success"},
                "macro": {"status": "success"},
            }
        }
    }

    result = tracker._build_dag_four_branch_compliance(
        review_symbols=["688519.SH"],
        effective_local_holding_symbols=[],
        branch_signals_by_symbol=branch_signals,
    )

    assert "fundamental" not in result["limited_evidence_branch_by_symbol"].get(
        "688519.SH", []
    )


def test_dag_compliance_keeps_legacy_quant_proxy_limited():
    branch_signals = {
        "688519.SH": {
            "reviewed_branch_verdicts": {
                "quant": {
                    "status": "success",
                    "final_score": 0.0,
                    "final_confidence": 0.5,
                    "diagnostic_notes": [
                        "legacy_proxy_fallback",
                        "mined_factor_registry_empty_or_not_selectable",
                    ],
                    "metadata": {
                        "factor_mode": "legacy_proxy_fallback",
                        "mined_factor_runtime": {
                            "factor_count": 0,
                            "factors_used": [],
                            "applied_to_score": False,
                        },
                    },
                },
                "fundamental": {"status": "success"},
                "intelligence": {"status": "success"},
                "macro": {"status": "success"},
            }
        }
    }

    result = tracker._build_dag_four_branch_compliance(
        review_symbols=["688519.SH"],
        effective_local_holding_symbols=[],
        branch_signals_by_symbol=branch_signals,
    )

    assert "quant" in result["limited_evidence_branch_by_symbol"]["688519.SH"]


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

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)
    _patch_parquet_completeness(
        monkeypatch,
        {
            "complete": True,
            "latest_trade_date": "20260407",
            "blocking_incomplete_count": 0,
            "suspension_evidence": {},
        },
    )
    monkeypatch.setattr(
        tracker,
        "_load_or_compute_market_metrics_bundle",
        lambda **_kwargs: _fake_market_metrics_bundle(),
    )

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

        def download_all(self, *args, **kwargs):
            calls["download_all"] += 1
            raise AssertionError("run_tracker should not auto-backfill before formal review")

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)
    _patch_parquet_completeness(
        monkeypatch,
        {
            "complete": False,
            "latest_trade_date": "20260407",
            "blocking_incomplete_count": 12,
        },
    )
    monkeypatch.setattr(
        tracker,
        "_load_or_compute_market_metrics_bundle",
        lambda **_kwargs: _fake_market_metrics_bundle(),
    )

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

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)
    _patch_parquet_completeness(monkeypatch, completeness)
    prewarmed_metrics = pd.DataFrame(
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
    )
    prewarmed_breadth = {
        category: {
            "ret1_positive_ratio": 0.2,
            "ret20_positive_ratio": 0.5,
            "ma20_gt_ma60_ratio": 0.3,
            "avg_ret1": 0.0,
            "avg_ret20": 0.0,
            "avg_ret60": 0.0,
            "latest_count": len(symbols),
            "expected": len(symbols),
            "suspended_stale_count": 0,
        }
        for category, symbols in {"hs300": ["601869.SH"], "zz500": [], "zz1000": []}.items()
    }
    monkeypatch.setattr(
        tracker,
        "_load_or_compute_market_metrics_bundle",
        lambda **_kwargs: _fake_market_metrics_bundle(
            full_metrics=prewarmed_metrics,
            breadth=prewarmed_breadth,
        ),
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
                        "intelligence": {
                            "action": "hold",
                            "metadata": {"branch_mode": "structured_intelligence_fusion"},
                            "coverage_notes": ["legacy batch retired"],
                            "investment_risks": ["智能融合当前未调用旧 batch pipeline，文本证据为候选层可扩展能力。"],
                        },
                        "quant": {
                            "branch_name": "quant",
                            "factor_mode": "governed_mined_factors",
                            "factor_count": 12,
                            "applied_to_score": True,
                        },
                        "macro": {
                            "branch_name": "macro",
                            "regime": "neutral",
                        },
                    },
                    "branch_overlays": {"fundamental": {"action": "sell"}},
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
    market_snapshot_payload = json.loads((run_dir / "market_snapshot.json").read_text(encoding="utf-8"))
    runtime_profile_payload = json.loads((run_dir / "runtime_profile.json").read_text(encoding="utf-8"))

    assert "#### 5.4.1 决策诊断" in report_text
    assert "#### 5.3.1 DAG 四分支执行验收" in report_text
    assert "DAG四分支完整执行" in report_text
    assert "stale_snapshot" in report_text
    assert "601869.SH(长飞光纤) 持有成本 `100.00`，PNL `+6,000.00 元`（+20.00%）" in report_text
    assert "Codex 评分 `" in report_text
    assert "#### 5.4.3 证据质量与工程诊断" in report_text
    assert "provider、snapshot、旧 intelligence batch" in report_text
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
    assert "provider_missing" not in advice_section
    assert manifest_payload["action_taken_today"] is False
    assert manifest_payload["formal_diagnostics"]["decision_guardrail"]["display_label"] == "no_action_evidence_impaired"
    assert manifest_payload["dag_four_branch_compliance"]["complete"] is True
    assert manifest_payload["market_metrics_prewarm"]["status"] == "blocking_generated"
    assert manifest_payload["data_snapshot"]["market_metrics_cache"]["status"] == "blocking_generated"
    assert manifest_payload["files"]["runtime_profile"] == "runtime_profile.json"
    assert manifest_payload["raw_exports"]["runtime_profile"] == "raw_exports/runtime_profile.json"
    assert manifest_payload["runtime_profile"]["stages"][0]["name"] == "market_metrics_prewarm"
    assert market_snapshot_payload["market_metrics_prewarm"]["status"] == "blocking_generated"
    assert runtime_profile_payload["stages"][0]["status"] == "blocking_generated"
    assert result["full_market_metrics_cache"]["status"] == "blocking_generated"
    assert (
        manifest_payload["formal_diagnostics"]["dag_four_branch_compliance"]["status"]
        == "DAG四分支完整执行"
    )
    orders_payload = (run_dir / "orders.csv").read_text(encoding="utf-8-sig")
    assert orders_payload == "timestamp,action,symbol,name,shares,price,trade_value,realized_pnl,reason\n"
    assert (run_dir / "orders.csv").exists()
    assert (run_dir / "holdings_review.csv").exists()
    assert (run_dir / "raw_exports" / "runtime_profile.json").exists()
    holdings_review = pd.read_csv(run_dir / "holdings_review.csv", encoding="utf-8-sig")
    assert "codex_recommendation_score" in holdings_review.columns
    assert "codex_recommendation_rating" in holdings_review.columns


def test_legacy_overweight_holding_warning_only_does_not_force_sell():
    review = pd.DataFrame(
        [
            {
                "symbol": "688519.SH",
                "name": "南亚新材",
                "shares_before": 1000,
                "buy_price": 20.0,
                "current_price": 50.0,
                "stage_stop_price": 30.0,
                "score_full_market": 0.90,
                "today_change_pct": 2.0,
                "market_weight": 0.22,
            }
        ]
    )

    assert tracker._build_rebalance_plan(review) == []
    lines = tracker._format_legacy_overweight_holding_lines(review, cap=0.15)

    assert len(lines) == 1
    assert "超限存量持仓" in lines[0]
    assert "当前权重 22.00%" in lines[0]
    assert "上限 15.00%" in lines[0]
    assert "只提示，不强制卖出" in lines[0]


def test_prune_holdings_review_to_effective_ledger_removes_exited_symbols():
    holdings_review = pd.DataFrame(
        [
            {"symbol": "688301.SH", "name": "奕瑞科技", "current_value": 100.0},
            {"symbol": "300285.SZ", "name": "国瓷材料", "current_value": 50.0},
        ]
    )
    effective_ledger = pd.DataFrame(
        [
            {"symbol": "688301.SH", "name": "奕瑞科技", "current_value": 100.0},
        ]
    )

    pruned, invariant = tracker._prune_holdings_review_to_effective_ledger(
        holdings_review,
        effective_ledger,
    )

    assert pruned["symbol"].tolist() == ["688301.SH"]
    assert invariant["status"] == "ok"
    assert invariant["pruned_extra_review_symbols"] == ["300285.SZ"]
    assert invariant["pre_prune"]["status"] == "warning"


def test_run_tracker_auto_fills_risk_reduction_sell_with_realtime_quote(monkeypatch, tmp_path):
    ledger = pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "name": "奕瑞科技",
                "shares": 500,
                "avg_cost": 164.24,
                "cost_basis": 82120.0,
                "current_price": 145.0,
                "current_value": 72500.0,
                "unrealized_pnl": -9620.0,
                "unrealized_pnl_pct": -0.117145,
                "market_weight": 0.5,
                "stage_target_price": 200.46,
                "stage_stop_price": 137.90,
                "thesis_status": "降级观察",
            }
        ]
    )
    manifest = {"timestamp": "20260617_0932", "capital_cny": 1_000_000.0}
    pnl = pd.DataFrame([{"cash_after": 100000.0, "total_value_after": 172500.0}])
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
    monkeypatch.setattr(
        tracker,
        "is_previous_day_realtime_decision_sufficient",
        lambda **_kwargs: False,
    )

    completeness = {
        "complete": False,
        "latest_trade_date": "20260618",
        "strict_trade_date": "20260618",
        "stable_trade_date": "20260617",
        "effective_target_trade_date": "20260618",
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
                "latest_trade_date": "20260618",
                "date_counts": {"20260617": 1},
                "coverage_complete_count": 0,
                "blocking_incomplete_count": 1,
                "suspended_stale_symbols": [],
                "blocking_missing_symbols": [],
                "blocking_stale_symbols": [{"symbol": "688301.SH", "latest_local_date": "20260617"}],
            },
            "hs300": {
                "expected": 0,
                "latest_trade_date": "20260618",
                "date_counts": {},
                "coverage_complete_count": 0,
                "blocking_incomplete_count": 0,
                "suspended_stale_symbols": [],
                "blocking_missing_symbols": [],
                "blocking_stale_symbols": [],
            },
            "zz500": {
                "expected": 0,
                "latest_trade_date": "20260618",
                "date_counts": {},
                "coverage_complete_count": 0,
                "blocking_incomplete_count": 0,
                "suspended_stale_symbols": [],
                "blocking_missing_symbols": [],
                "blocking_stale_symbols": [],
            },
            "zz1000": {
                "expected": 0,
                "latest_trade_date": "20260618",
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
            return {"full_a": ["688301.SH"], "hs300": [], "zz500": [], "zz1000": []}

    monkeypatch.setattr(tracker, "CNFullMarketDownloader", _FakeDownloader)
    _patch_parquet_completeness(monkeypatch, completeness)
    prewarmed_metrics = pd.DataFrame(
        [
            {
                "symbol": "688301.SH",
                "name": "奕瑞科技",
                "category": "full_a",
                "ret1": -0.02,
                "ret5": -0.05,
                "ret20": -0.12,
                "ret60": -0.20,
                "close_vs_ma20": -0.08,
                "ma20_vs_ma60": -0.04,
                "ma60_vs_ma120": -0.02,
                "dd20": -0.22,
                "latest_close": 120.0,
                "stage_target_price": 200.46,
                "stage_stop_price": 137.90,
                "score_full_market": 0.38,
                "rank_full_market": 900,
            }
        ]
    )
    prewarmed_breadth = {
        category: {
            "ret1_positive_ratio": 0.2,
            "ret20_positive_ratio": 0.3,
            "ma20_gt_ma60_ratio": 0.25,
            "avg_ret1": 0.0,
            "avg_ret20": 0.0,
            "avg_ret60": 0.0,
            "latest_count": len(symbols),
            "expected": len(symbols),
            "suspended_stale_count": 0,
        }
        for category, symbols in {"hs300": [], "zz500": [], "zz1000": []}.items()
    }
    monkeypatch.setattr(
        tracker,
        "_load_or_compute_market_metrics_bundle",
        lambda **_kwargs: _fake_market_metrics_bundle(
            full_metrics=prewarmed_metrics,
            breadth=prewarmed_breadth,
        ),
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
                    "time": "20260618101603",
                    "change": 1.0,
                    "change_pct": 1.01,
                }
                for code in tracker.INDEX_QUOTES.keys()
            },
            "sh688301": {
                "quote_code": "sh688301",
                "name": "奕瑞科技",
                "current": 120.75,
                "prev_close": 122.0,
                "open": 121.5,
                "high": 122.2,
                "low": 120.0,
                "time": "20260618101603",
                "change": -1.25,
                "change_pct": -1.02,
            },
        },
    )
    monkeypatch.setattr(
        tracker,
        "_run_unified_review_mainline_for_holdings",
        lambda source_ledger, latest_trade_date, source_record: {
            "reviewed_symbols": ["688301.SH"],
            "by_symbol": {
                "688301.SH": {
                    "recommendation": {
                        "action": "sell",
                        "confidence": 0.7,
                        "one_line_conclusion": "跌破止损，减仓。",
                        "risk_flags": ["broken_stop"],
                    },
                    "ic_hint": {"action": "sell", "confidence_hint": 0.7, "thesis": "跌破止损"},
                    "master_hint": {"action": "sell", "confidence_hint": 0.7},
                    "llm_attempt_summary": {"call_count": 0, "success_count": 0, "failed_count": 0, "fallback_count": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
                    "llm_effective_summary": {"call_count": 0, "success_count": 0, "failed_count": 0, "fallback_count": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
                    "reviewed_branch_verdicts": {
                        "quant": {"branch_name": "quant"},
                        "fundamental": {"branch_name": "fundamental", "metadata": {"data_quality": {}}},
                        "intelligence": {"branch_name": "intelligence"},
                        "macro": {"branch_name": "macro"},
                    },
                    "branch_overlays": {},
                    "report_excerpt": "sell",
                }
            },
            "degraded_symbols": {},
            "llm_attempt_summary": {"call_count": 0, "success_count": 0, "failed_count": 0, "fallback_count": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
            "llm_effective_summary": {"call_count": 0, "success_count": 0, "failed_count": 0, "fallback_count": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
            "model_role_metadata": {},
            "fallback_reasons": [],
            "session_ids": {},
        },
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

    args = argparse.Namespace(
        base_dir=str(tmp_path / "strategy_records"),
        years=7,
        max_rounds=0,
        source_record=None,
        allowed_stale_symbols=[],
    )
    result = tracker.run_tracker(args)

    assert result["action_taken_today"] is True
    assert result["decision_data_sufficient"] is False
    assert result["manual_execution"]["status"] == "filled_local_manual_paper_rebalance"
    run_dir = tmp_path / "strategy_records" / result["timestamp"]
    manual_manifest = json.loads((run_dir / "manual_execution_manifest.json").read_text(encoding="utf-8"))
    manual_orders = pd.read_csv(run_dir / "manual_switch_and_take_profit_orders.csv")
    next_ledger = pd.read_csv(run_dir / "ledger_after_manual_switch.csv")
    formal_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manual_manifest["price_basis"] == "execution_time_realtime_quote"
    assert manual_manifest["decision_data_sufficient"] is False
    assert manual_manifest["applied_local_trades"][0]["symbol"] == "688301.SH"
    assert manual_orders.iloc[0]["status"] == "filled"
    assert manual_orders.iloc[0]["execution_price"] == pytest.approx(120.75)
    assert int(next_ledger.iloc[0]["shares"]) == 400
    assert formal_manifest["manual_execution"]["status"] == "filled_local_manual_paper_rebalance"
    assert formal_manifest["execution_price_gate"]["data_gate_allows_new_risk"] is False
