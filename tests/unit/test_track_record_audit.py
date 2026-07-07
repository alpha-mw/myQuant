from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "track_record_audit",
        ROOT / "scripts" / "run_track_record_audit.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_record(root: Path, run_id: str, total: float, *, metric_pnl: bool = False) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    record_date = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "timestamp": run_id,
                "recorded_at": f"{record_date} 10:00:00 CST",
                "capital_cny": 100.0,
                "strategy": "aggressive_tech_manufacturing",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if metric_pnl:
        (run_dir / "pnl_summary.csv").write_text(
            "\n".join(
                [
                    "metric,value",
                    "initial_capital,100",
                    f"total_value_after,{total}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (run_dir / "pnl_summary.csv").write_text(
            "record_time,initial_capital,total_value_after,cash_after,market_value_after\n"
            f"{record_date},100,{total},0,{total}\n",
            encoding="utf-8",
        )
    (run_dir / "ledger_after_manual_switch.csv").write_text(
        "symbol,name,shares,avg_cost,current_price,current_value,unrealized_pnl,market_weight\n"
        f"AAA.SZ,Alpha,100,10,{total / 10:.2f},{total},0,1\n",
        encoding="utf-8",
    )
    (run_dir / "orders.csv").write_text(
        "timestamp,action,symbol,name,shares,price,trade_value,realized_pnl,reason\n",
        encoding="utf-8",
    )
    (run_dir / "manual_switch_and_take_profit_orders.csv").write_text(
        "timestamp,action,symbol,name,shares,execution_price,trade_value,realized_pnl,status,reason\n",
        encoding="utf-8",
    )
    (run_dir / "market_snapshot.json").write_text("{}", encoding="utf-8")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    record_root = tmp_path / "records"
    _write_record(record_root, "20260102_1000", 100.0)
    _write_record(record_root, "20260105_1000", 110.0, metric_pnl=True)
    _write_record(record_root, "20260106_1000", 121.0)

    benchmark = tmp_path / "cn_index_benchmark.csv"
    benchmark.write_text(
        "\n".join(
            [
                "date,ts_code,close,source_system,value_date,coverage",
                "2026-01-02,000300.SH,100,unit,2026-01-02,exact_close",
                "2026-01-05,000300.SH,100,unit,2026-01-05,exact_close",
                "2026-01-06,000300.SH,100,unit,2026-01-06,exact_close",
                "2026-01-02,000688.SH,100,unit,2026-01-02,exact_close",
                "2026-01-05,000688.SH,105,unit,2026-01-05,exact_close",
                "2026-01-06,000688.SH,110,unit,2026-01-06,exact_close",
                "2026-01-02,399006.SZ,100,unit,2026-01-02,exact_close",
                "2026-01-05,399006.SZ,100,unit,2026-01-05,exact_close",
                "2026-01-06,399006.SZ,100,unit,2026-01-06,exact_close",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "AAA.SZ", "trade_date": "20260102", "close": 10.0, "adj_close": 10.0},
            {"ts_code": "AAA.SZ", "trade_date": "20260105", "close": 11.0, "adj_close": 11.0},
            {"ts_code": "AAA.SZ", "trade_date": "20260106", "close": 12.1, "adj_close": 12.1},
            {"ts_code": "BBB.SZ", "trade_date": "20260102", "close": 20.0, "adj_close": 20.0},
            {"ts_code": "BBB.SZ", "trade_date": "20260105", "close": 20.0, "adj_close": 20.0},
            {"ts_code": "BBB.SZ", "trade_date": "20260106", "close": 20.0, "adj_close": 20.0},
        ]
    ).to_parquet(bars_root / "part.parquet", index=False)
    stock_basic = tmp_path / "stock_basic"
    stock_basic.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "AAA.SZ", "industry": "IT设备"},
            {"ts_code": "BBB.SZ", "industry": "半导体"},
        ]
    ).to_parquet(stock_basic / "part.parquet", index=False)
    fundamentals_root = tmp_path / "fundamental_raw"
    indicator = fundamentals_root / "table=fina_indicator"
    indicator.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "AAA.SZ",
                "end_date": "20260331",
                "ann_date": "20260430",
                "tr_yoy": 12.0,
                "netprofit_yoy": 8.0,
            }
        ]
    ).to_parquet(indicator / "part.parquet", index=False)
    regime = tmp_path / "markov_regime_history.jsonl"
    regime.write_text(
        json.dumps(
            {
                "as_of": "20260706",
                "dominant_regime": "趋势下跌",
                "confidence": 0.5,
                "transition_risk": 0.8,
                "suggested_gross_exposure_cap": 0.36,
                "turnover_cap": 0.3,
                "regime_scope": "full_market",
                "feature_snapshot": {"breadth": 0.3},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return record_root, benchmark, bars_root, stock_basic, fundamentals_root, regime


def test_track_record_audit_decomposition_read_only_and_deterministic(tmp_path):
    audit = _load_audit()
    record_root, benchmark, bars_root, stock_basic, fundamentals_root, regime = _write_inputs(tmp_path)
    output_root = tmp_path / "audit"

    for path in [record_root, *record_root.iterdir()]:
        path.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        metrics = audit.run_audit(
            record_root=record_root,
            output_root=output_root,
            benchmark_file=benchmark,
            bars_root=bars_root,
            stock_basic_root=stock_basic,
            fundamentals_root=fundamentals_root,
            regime_history=regime,
            as_of_date="20260107",
            generate_plots=False,
        )
        first_json = (output_root / "20260107" / "audit_metrics.json").read_bytes()
        first_md = (output_root / "20260107" / "audit_report.md").read_bytes()
        audit.run_audit(
            record_root=record_root,
            output_root=output_root,
            benchmark_file=benchmark,
            bars_root=bars_root,
            stock_basic_root=stock_basic,
            fundamentals_root=fundamentals_root,
            regime_history=regime,
            as_of_date="20260107",
            generate_plots=False,
        )
    finally:
        for path in [record_root, *record_root.iterdir()]:
            path.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)

    second_json = (output_root / "20260107" / "audit_metrics.json").read_bytes()
    second_md = (output_root / "20260107" / "audit_report.md").read_bytes()
    assert first_json == second_json
    assert first_md == second_md
    decomp = metrics["decomposition"]
    assert abs(decomp["total_excess_vs_csi300"] - decomp["explained_sum"]) < 1e-12
    assert abs(decomp["reconciliation_residual"]) < 1e-12
    report = first_md.decode("utf-8")
    assert "## A. 基础绩效" in report
    assert "## B. 基准与超额" in report
    assert "## C. 三分解" in report
    assert metrics["fundamentals_appendix"]["rows"][0]["h1_schedule"] == "披露日需人工补充"


def test_track_record_audit_refuses_output_inside_record_root(tmp_path):
    audit = _load_audit()
    record_root, benchmark, bars_root, stock_basic, fundamentals_root, regime = _write_inputs(tmp_path)

    try:
        audit.run_audit(
            record_root=record_root,
            output_root=record_root / "audit",
            benchmark_file=benchmark,
            bars_root=bars_root,
            stock_basic_root=stock_basic,
            fundamentals_root=fundamentals_root,
            regime_history=regime,
            as_of_date="20260107",
            generate_plots=False,
        )
    except ValueError as exc:
        assert "must not be inside strategy record root" in str(exc)
    else:
        raise AssertionError("expected output-inside-record-root guard")


def test_track_record_audit_missing_optional_fields_degrades(tmp_path):
    audit = _load_audit()
    record_root = tmp_path / "records"
    _write_record(record_root, "20260102_1000", 100.0, metric_pnl=True)
    _write_record(record_root, "20260105_1000", 101.0, metric_pnl=True)
    for run_dir in record_root.iterdir():
        os.remove(run_dir / "ledger_after_manual_switch.csv")
        os.remove(run_dir / "manual_switch_and_take_profit_orders.csv")
        os.remove(run_dir / "orders.csv")
    benchmark = tmp_path / "cn_index_benchmark.csv"
    benchmark.write_text(
        "date,ts_code,close,source_system\n"
        "2026-01-02,000300.SH,100,unit\n"
        "2026-01-05,000300.SH,101,unit\n",
        encoding="utf-8",
    )

    metrics = audit.run_audit(
        record_root=record_root,
        output_root=tmp_path / "audit",
        benchmark_file=benchmark,
        bars_root=tmp_path / "missing_bars",
        stock_basic_root=tmp_path / "missing_stock_basic",
        fundamentals_root=tmp_path / "missing_fundamentals",
        regime_history=tmp_path / "missing_regime.jsonl",
        as_of_date="20260107",
        generate_plots=False,
    )

    assert metrics["trade_count"] == 0
    assert metrics["funding_nature"]["funding_nature"] == "undetermined_early_records"
    assert metrics["markov_diagnostics"]["available"] is False
