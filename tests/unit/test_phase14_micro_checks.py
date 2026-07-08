from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_phase14_micro_checks",
        ROOT / "scripts" / "run_phase14_micro_checks.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_strict_session_match_requires_symbol_near_advice_verb(tmp_path):
    mod = _load_module()
    session_dir = tmp_path / "sessions" / "2026" / "07" / "03"
    session_dir.mkdir(parents=True)
    lines = ["noise"] * 30
    lines[10] = "symbol 688519.SH appears here"
    lines[18] = "建议 卖出 part of the position"
    lines[29] = "symbol 688525.SH appears without advice nearby"
    (session_dir / "rollout-2026-07-03.jsonl").write_text("\n".join(lines), encoding="utf-8")

    rows = mod.strict_session_match_matrix(
        [
            {"date": "2026-07-03", "symbol": "688519.SH", "shares": 100, "price": 10},
            {"date": "2026-07-03", "symbol": "688525.SH", "shares": 100, "price": 10},
        ],
        tmp_path / "sessions" / "2026",
    )

    assert rows[0]["match_level"] == "symbol_plus_advice_verb"
    assert rows[0]["advice_verb_found"] is True
    assert rows[1]["match_level"] == "symbol_only_no_advice_verb"
    assert rows[1]["advice_verb_found"] is False


def test_trade_tradability_flags_one_price_limit_and_missing_bar(tmp_path):
    mod = _load_module()
    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "AAA.SZ",
                "trade_date": "20260703",
                "open": 11.0,
                "high": 11.0,
                "low": 11.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
                "vol": 100.0,
                "amount": 1100.0,
            },
            {
                "ts_code": "BBB.SZ",
                "trade_date": "20260703",
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "pre_close": 10.0,
                "pct_chg": 5.0,
                "vol": 100.0,
                "amount": 1000.0,
            },
        ]
    ).to_parquet(bars_root / "part.parquet", index=False)

    result = mod.trade_tradability_check(
        [
            {"date": "2026-07-03", "symbol": "AAA.SZ", "action": "buy", "status": "filled", "shares": 100, "price": 11},
            {"date": "2026-07-03", "symbol": "BBB.SZ", "action": "sell", "status": "filled", "shares": 100, "price": 10.5},
            {"date": "2026-07-03", "symbol": "CCC.SZ", "action": "sell", "status": "filled", "shares": 100, "price": 10.5},
            {"date": "2026-07-04", "symbol": "DDD.SZ", "action": "sell", "status": "filled", "shares": 100, "price": 10.5},
        ],
        bars_root,
    )

    assert result["trade_count"] == 4
    assert [row["symbol"] for row in result["violations"]] == ["AAA.SZ", "CCC.SZ", "DDD.SZ"]
    assert "buy_blocked_by_one_price_limit_up" in result["violations"][0]["flags"]
    assert "missing_bar_or_possible_suspension" in result["violations"][1]["flags"]
    assert "weekend_paper_fill" in result["violations"][2]["flags"]


def test_machine_exit_breakdown_adds_status_and_contribution_fields():
    mod = _load_module()
    result = mod.machine_exit_breakdown(
        {
            "nav_rows": [{"initial_capital": 1000.0}],
            "shadow_ledgers": {
                "shadow_nav_machine_exit": {
                    "rows": [
                        {
                            "date": "2026-07-03",
                            "symbol": "AAA.SZ",
                            "shares": 100,
                            "stage_stop_price": 9.0,
                            "machine_exit_price": 8.5,
                            "delta_vs_manual_pnl": -150.0,
                        },
                        {
                            "date": "2026-07-03",
                            "symbol": "BBB.SZ",
                            "shares": 100,
                            "stage_stop_price": 9.0,
                            "machine_exit_price": 12.0,
                            "delta_vs_manual_pnl": 300.0,
                        },
                    ]
                }
            },
        }
    )

    assert result["total_difference_vs_actual"] == 0.15
    assert result["rows"][0]["exit_status"] == "stop_triggered_exit"
    assert result["rows"][1]["exit_status"] == "not_closed_marked_to_window_end"
    assert result["rows"][1]["contribution_pct_of_initial"] == 0.3


def test_price_audit_688301_uses_raw_close_and_reports_adjusted_mismatch(tmp_path):
    mod = _load_module()
    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "688301.SH", "trade_date": "20260617", "close": 120.85, "adj_close": 483.702125},
            {"ts_code": "688301.SH", "trade_date": "20260706", "close": 110.47, "adj_close": 442.156175},
        ]
    ).to_parquet(bars_root / "part.parquet", index=False)
    metrics = {
        "nav_rows": [{"date": "2026-06-17", "initial_capital": 1_000_000}, {"date": "2026-07-06"}],
        "shadow_ledgers": {
            "shadow_nav_machine_exit": {"rows": []},
            "machine_exit_sensitivity_including_non_trading": {
                "rows": [{"date": "2026-06-20", "symbol": "688301.SH"}]
            },
        },
    }

    result = mod.price_audit_688301(
        metrics,
        [
            {
                "date": "2026-06-23",
                "symbol": "688301.SH",
                "action": "sell",
                "status": "filled",
                "shares": 400,
                "price": 112.14,
            },
            {
                "date": "2026-06-20",
                "symbol": "688301.SH",
                "action": "sell",
                "status": "filled",
                "shares": 100,
                "price": 120.6,
                "calendar_status": "weekend_paper_fill",
            },
        ],
        bars_root,
    )

    recompute = result["manual_recompute"]
    assert recompute["shares_unit"] == "股"
    assert abs(recompute["raw_contribution_pct"] - (-0.000668)) < 1e-9
    assert abs(recompute["adjusted_close_mismatch_contribution_pct"] - 0.13200647) < 1e-9
    assert result["ghost_20260620"]["included_in_current_default_shadow"] is False
    assert result["ghost_20260620"]["included_in_sensitivity_shadow"] is True


def test_theme_guardrail_replay_finds_first_hard_filter_residual_trigger(tmp_path):
    mod = _load_module()
    record_root = tmp_path / "records"
    first = record_root / "20260702_1357"
    first.mkdir(parents=True)
    (first / "theme_pool_audit.json").write_text(
        json.dumps(
            {
                "summary": {
                    "policy": {"hard_theme_constraint": True, "residual_enabled": True},
                    "core_symbol_count": 0,
                    "residual_symbol_count": 6,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (first / "market_snapshot.json").write_text(
        json.dumps({"blocker": "theme_pool_hard_filter_regression"}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = mod.theme_guardrail_replay(record_root)

    assert result["first_trigger"]["record_id"] == "20260702_1357"
    assert "zero upper bound" in result["residual_6_violation"]


def test_exit_record_audit_requires_manifest_and_order_rows(tmp_path):
    mod = _load_module()
    record_root = tmp_path / "records"
    run_dir = record_root / "20260707_1046"
    run_dir.mkdir(parents=True)
    (run_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "applied_local_trades": [
                    {"symbol": "300285.SZ", "action": "clear_risk_sell", "shares": 700, "price": 88.28}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "manual_switch_and_take_profit_orders.csv").write_text(
        "timestamp,status,action,symbol,shares,price\n"
        "2026-07-07,filled_local_manual,clear_risk_sell,300285.SZ,700,88.28\n",
        encoding="utf-8",
    )

    result = mod.exit_record_audit(record_root, symbols=["300285.SZ", "600487.SH"])

    by_symbol = {row["symbol"]: row for row in result["rows"]}
    assert by_symbol["300285.SZ"]["status"] == "complete_exit_record"
    assert by_symbol["300285.SZ"]["nav_breakpoint_risk"] is False
    assert by_symbol["600487.SH"]["status"] == "missing_exit_record"
    assert "600487.SH" in result["nav_breakpoint_risk_symbols"]


def _theme_payload(theme_id: str, score: float) -> dict[str, object]:
    return {
        "theme_id": theme_id,
        "theme_name": theme_id.split("::")[-1],
        "score": score,
        "phase": "accumulation",
        "breadth": 0.6,
        "confidence": 0.8,
        "member_count": 8,
        "risk_flags": [],
    }


def test_phase14_3_residual_source_audit_and_current_replay_classify_stale_artifact(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "records" / "20260707_1046"
    run_dir.mkdir(parents=True)
    (run_dir / "theme_pool_audit.json").write_text(
        json.dumps(
            {
                "summary": {
                    "policy": {"hard_theme_constraint": True, "residual_enabled": True},
                    "policy_regime": "趋势下跌",
                    "residual_symbol_count": 1,
                    "admitted_symbol_count": 2,
                    "admitted_symbols_by_source_sample": {"residual_theme": ["AAA.SZ"]},
                },
                "symbols": {
                    "AAA.SZ": {
                        "admitted": True,
                        "source": "residual_theme",
                        "primary_theme_id": "industry::农业综合",
                        "theme_pool_score": 0.12,
                        "theme_policy_regime": "趋势下跌",
                        "theme_pool_reason": "admitted",
                    },
                    "BBB.SZ": {
                        "admitted": True,
                        "source": "core",
                        "primary_theme_id": "industry::生物制药",
                        "theme_pool_score": 0.90,
                    },
                },
                "admitted_symbols_by_source": {"residual_theme": ["AAA.SZ"], "core": ["BBB.SZ"]},
                "excluded_symbols_by_reason": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "market_snapshot.json").write_text(
        json.dumps(
            {
                "candidate_level_dag_status": {
                    "theme_candidate_pool": {
                        "admitted_symbols_by_source_sample": {"residual_theme": ["AAA.SZ"]}
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "theme_snapshot.json").write_text(
        json.dumps(
            {
                "stored_snapshot": {
                    "theme_rotation": {
                        "status": "success",
                        "universe_key": "full_a",
                        "theme_scores": {
                            "industry::农业综合": _theme_payload("industry::农业综合", 0.42),
                            "industry::生物制药": _theme_payload("industry::生物制药", 0.52),
                        },
                        "symbol_scores": {"AAA.SZ": 0.12, "BBB.SZ": 0.90},
                        "symbol_smoothed_scores": {"AAA.SZ": 0.12, "BBB.SZ": 0.90},
                        "symbol_primary_theme": {
                            "AAA.SZ": "industry::农业综合",
                            "BBB.SZ": "industry::生物制药",
                        },
                        "symbol_phase": {"AAA.SZ": "accumulation", "BBB.SZ": "accumulation"},
                        "symbol_risk_flags": {"AAA.SZ": [], "BBB.SZ": []},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    source_audit = mod.residual_symbol_source_audit(run_dir)
    replay = mod.current_theme_pool_replay(run_dir)

    assert source_audit["residual_symbols"] == ["AAA.SZ"]
    assert any(
        path == "theme_pool_audit.json:admitted_symbols_by_source.residual_theme"
        for path in source_audit["rows"][0]["source_paths"]
    )
    assert replay["classification"] == "stale_artifact_pollution_or_historical_old_code_behavior"
    assert replay["residual_symbol_count"] == 0


def test_daily_pipeline_proof_distinguishes_hard_filter_pool_from_final_shortlist(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "records" / "20260708_0910"
    run_dir.mkdir(parents=True)
    (run_dir / "theme_pool_audit.json").write_text(
        json.dumps(
            {
                "summary": {
                    "policy": {"hard_theme_constraint": True, "residual_enabled": False},
                    "core_symbol_count": 228,
                    "residual_symbol_count": 0,
                    "forced_theme_count": 2,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "market_snapshot.json").write_text(
        json.dumps(
            {
                "candidate_level_dag_status": {
                    "candidate_generation_status": "empty",
                    "blocker": "no_candidate_selected_by_portfolio_constructor",
                    "candidate_pool": [],
                    "dag_pipeline": {"shortlist_count": 0, "portfolio_target_count": 0},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = mod.daily_pipeline_proof(tmp_path / "records", "20260708_0910")

    assert result["hard_filter_pool_restored_nonempty"] is True
    assert result["final_shortlist_restored_nonempty"] is False
    assert result["blocker"] == "no_candidate_selected_by_portfolio_constructor"


def test_shadow_denominator_check_explains_window_return_gap():
    mod = _load_module()
    result = mod.shadow_denominator_check(
        {
            "nav_rows": [
                {
                    "date": "2026-03-18",
                    "initial_capital": 1_000_000,
                    "nav": 1.023097,
                    "total_value_after": 1_023_097,
                },
                {
                    "date": "2026-07-08",
                    "initial_capital": 1_000_000,
                    "nav": 1.649461,
                    "total_value_after": 1_649_461,
                },
            ]
        }
    )

    assert result["initial_capital"] == 1_000_000
    assert result["last_nav_times_initial_capital"] == 1_649_461
    assert round(result["window_return_from_first_nav"], 6) == 0.612223
    assert result["gap_vs_last_total_if_window_return_is_misused"] > 37_000
    assert "No deposit-injection" in result["conclusion"]


def test_session_forensics_688301_excerpt_requires_advice_verb_near_symbol(tmp_path):
    mod = _load_module()
    session_dir = tmp_path / "sessions" / "2026" / "06" / "19"
    session_dir.mkdir(parents=True)
    lines = ["noise"] * 30
    lines[8] = "688301.SH appears without advice nearby"
    lines[20] = "复盘 688301 奕瑞，出现减仓建议，触发止盈纪律"
    (session_dir / "rollout-2026-06-19.jsonl").write_text("\n".join(lines), encoding="utf-8")

    result = mod.session_forensics_688301_excerpt(
        tmp_path / "sessions" / "2026",
        context_lines=5,
        max_matches=3,
    )

    assert result["match_count"] == 1
    assert result["matches"][0]["date"] == "2026-06-19"
