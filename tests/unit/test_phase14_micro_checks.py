from __future__ import annotations

import importlib.util
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
        ],
        bars_root,
    )

    assert result["trade_count"] == 3
    assert [row["symbol"] for row in result["violations"]] == ["AAA.SZ", "CCC.SZ"]
    assert "buy_blocked_by_one_price_limit_up" in result["violations"][0]["flags"]
    assert "missing_bar_or_possible_suspension" in result["violations"][1]["flags"]


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
