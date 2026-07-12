from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "dashboard_export_checker",
        ROOT / "scripts" / "check_cn_dashboard_export.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _csv_text(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    from io import StringIO

    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().rstrip("\r\n")


def _write_generated_js(
    path: Path,
    *,
    nav_csv: str,
    positions_csv: str,
    trades_csv: str,
    latest_record: str = "20260701_1032",
    record_count: int = 1,
    generated_at: str = "2026-07-01 16:48:50",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "window.DashboardGeneratedRecords = {",
                f"  generatedAt: {json.dumps(generated_at)},",
                f"  sourceRoot: {json.dumps('/tmp/records')},",
                f"  latestRecord: {json.dumps(latest_record)},",
                f"  recordCount: {record_count},",
                "  warnings: [],",
                "  csv: {",
                f"    nav: {json.dumps(nav_csv)},",
                f"    positions: {json.dumps(positions_csv)},",
                f"    trades: {json.dumps(trades_csv)}",
                "  }",
                "};",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_summary(
    path: Path,
    *,
    nav_rows: int = 1,
    positions_rows: int = 1,
    trade_rows: int = 0,
    source_status: str = "production_source_partial_missing_dates",
    source_system: str = "tushare.index_daily",
    production_grade: bool = False,
    trade_completeness: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-01 16:48:50",
                "source_root": "/tmp/records",
                "latest_record": "20260701_1032",
                "record_count": 1,
                "nav_rows": nav_rows,
                "positions_rows": positions_rows,
                "trade_rows": trade_rows,
                "warnings": [],
                "trade_record_completeness": trade_completeness
                or {"status": "complete", "skipped_incomplete_rows": 0},
                "benchmark_source": {
                    "benchmark_fields": [
                        "benchmark_main_nav",
                        "benchmark_nav",
                        "csi300_nav",
                        "csi500_nav",
                        "csi1000_nav",
                        "star50_nav",
                        "chinext_nav",
                    ],
                    "benchmark_source_status": source_status,
                    "source_system": source_system,
                    "production_grade": production_grade,
                    "first_valid_date": "2026-03-18",
                    "last_valid_date": "2026-06-25",
                    "missing_date_count": 1,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _valid_csvs() -> tuple[str, str, str]:
    nav_fields = [
        "date",
        "portfolio_nav",
        "benchmark_main_nav",
        "benchmark_nav",
        "csi300_nav",
        "csi500_nav",
        "csi1000_nav",
        "star50_nav",
        "chinext_nav",
    ]
    nav_csv = _csv_text(
        [
            {
                "date": "2026-03-18",
                "portfolio_nav": "1.0",
                "benchmark_main_nav": "1.0",
                "benchmark_nav": "1.0",
                "csi300_nav": "1.0",
                "csi500_nav": "1.0",
                "csi1000_nav": "1.0",
                "star50_nav": "1.0",
                "chinext_nav": "1.0",
            }
        ],
        nav_fields,
    )
    positions_csv = _csv_text(
        [
            {
                "date": "2026-03-18",
                "ticker": "688525.SH",
                "name": "佰维存储",
                "weight": "0.1",
            }
        ],
        ["date", "ticker", "name", "weight"],
    )
    trades_csv = "trade_date,ticker,name,side,price,quantity,trade_amount,fee,reason,theme"
    return nav_csv, positions_csv, trades_csv


def test_dashboard_export_check_allows_partial_benchmark_by_default(tmp_path):
    checker = _load_checker()
    summary_file = tmp_path / "export_summary.json"
    generated_js = tmp_path / "generated_records.js"
    nav_csv, positions_csv, trades_csv = _valid_csvs()
    _write_summary(summary_file)
    _write_generated_js(
        generated_js,
        nav_csv=nav_csv,
        positions_csv=positions_csv,
        trades_csv=trades_csv,
    )

    result = checker.check_dashboard_export(summary_file, generated_js)

    assert result["ok"] is True
    assert result["fallback_to_sample"] is False
    assert result["nav_rows"] == 1
    assert result["positions_rows"] == 1
    assert result["trade_rows"] == 0
    assert result["benchmark_source"]["production_grade"] is False
    assert any("not formal" in warning for warning in result["warnings"])


def test_dashboard_export_check_can_require_production_benchmark(tmp_path):
    checker = _load_checker()
    summary_file = tmp_path / "export_summary.json"
    generated_js = tmp_path / "generated_records.js"
    nav_csv, positions_csv, trades_csv = _valid_csvs()
    _write_summary(summary_file)
    _write_generated_js(
        generated_js,
        nav_csv=nav_csv,
        positions_csv=positions_csv,
        trades_csv=trades_csv,
    )

    result = checker.check_dashboard_export(
        summary_file,
        generated_js,
        require_production_benchmark=True,
    )

    assert result["ok"] is False
    assert any("not production_grade" in error for error in result["errors"])


def test_dashboard_export_check_rejects_non_unitized_funding_and_legacy_ledger(tmp_path):
    checker = _load_checker()
    summary_file = tmp_path / "export_summary.json"
    generated_js = tmp_path / "generated_records.js"
    nav_csv, positions_csv, trades_csv = _valid_csvs()
    _write_summary(summary_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["portfolio_nav_source"] = {
        "method": "cash_flow_subtraction",
        "historical_return_preserved": True,
        "capital_base_start": 100.0,
        "capital_base_end": 1000.0,
        "funding_events": [
            {
                "date": "2026-07-01",
                "amount": 900.0,
                "total_value_before": 110.0,
                "total_value_after": 1010.0,
            }
        ],
    }
    summary["effective_manual_ledger_status"] = {
        "status": "valid",
        "ledger_path": "/tmp/records/ledger.csv",
        "manifest_path": "",
        "legacy_ledger_fallback_used": True,
    }
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    _write_generated_js(
        generated_js,
        nav_csv=nav_csv,
        positions_csv=positions_csv,
        trades_csv=trades_csv,
    )

    result = checker.check_dashboard_export(summary_file, generated_js)

    assert result["ok"] is False
    assert any("time_weighted_unitization" in error for error in result["errors"])
    assert any("portfolio_units" in error for error in result["errors"])
    assert any("legacy ledger.csv fallback" in error for error in result["errors"])
    assert any("ledger_after_manual_switch.csv" in error for error in result["errors"])
    assert any("manual_execution_manifest.json" in error for error in result["errors"])


def test_dashboard_export_check_allows_previous_trading_day_ffill_as_production(tmp_path):
    checker = _load_checker()
    summary_file = tmp_path / "export_summary.json"
    generated_js = tmp_path / "generated_records.js"
    nav_csv, positions_csv, trades_csv = _valid_csvs()
    _write_summary(
        summary_file,
        source_status="production_source_with_previous_trading_day_ffill",
        source_system="tushare.index_daily+eastmoney.push2his.kline",
        production_grade=True,
    )
    _write_generated_js(
        generated_js,
        nav_csv=nav_csv,
        positions_csv=positions_csv,
        trades_csv=trades_csv,
    )

    result = checker.check_dashboard_export(
        summary_file,
        generated_js,
        require_production_benchmark=True,
    )

    assert result["ok"] is True
    assert result["warnings"] == []


def test_dashboard_export_check_fails_when_generated_records_are_empty(tmp_path):
    checker = _load_checker()
    summary_file = tmp_path / "export_summary.json"
    generated_js = tmp_path / "generated_records.js"
    _write_summary(summary_file, nav_rows=0, positions_rows=0, trade_rows=0)
    _write_generated_js(generated_js, nav_csv="", positions_csv="", trades_csv="")

    result = checker.check_dashboard_export(summary_file, generated_js)

    assert result["ok"] is False
    assert result["fallback_to_sample"] is True
    assert any("fall back to sample" in error for error in result["errors"])


def test_dashboard_export_check_rejects_sample_source_system(tmp_path):
    checker = _load_checker()
    summary_file = tmp_path / "export_summary.json"
    generated_js = tmp_path / "generated_records.js"
    nav_csv, positions_csv, trades_csv = _valid_csvs()
    _write_summary(
        summary_file,
        source_status="production_grade",
        source_system="sample.benchmark",
        production_grade=True,
    )
    _write_generated_js(
        generated_js,
        nav_csv=nav_csv,
        positions_csv=positions_csv,
        trades_csv=trades_csv,
    )

    result = checker.check_dashboard_export(summary_file, generated_js)

    assert result["ok"] is False
    assert any("sample/mock/demo" in error for error in result["errors"])


def test_dashboard_export_check_rejects_incomplete_trade_records(tmp_path):
    checker = _load_checker()
    summary_file = tmp_path / "export_summary.json"
    generated_js = tmp_path / "generated_records.js"
    nav_csv, positions_csv, trades_csv = _valid_csvs()
    _write_summary(
        summary_file,
        trade_completeness={"status": "partial", "skipped_incomplete_rows": 1},
    )
    _write_generated_js(
        generated_js,
        nav_csv=nav_csv,
        positions_csv=positions_csv,
        trades_csv=trades_csv,
    )

    result = checker.check_dashboard_export(summary_file, generated_js)

    assert result["ok"] is False
    assert any("trade_record_completeness" in error for error in result["errors"])


def test_dashboard_export_check_cli_accepts_dashboard_root(tmp_path, monkeypatch, capsys):
    checker = _load_checker()
    dashboard_root = tmp_path / "portfolio_dashboard"
    summary_file = dashboard_root / "generated" / "export_summary.json"
    generated_js = dashboard_root / "js" / "generated_records.js"
    nav_csv, positions_csv, trades_csv = _valid_csvs()
    _write_summary(summary_file)
    _write_generated_js(
        generated_js,
        nav_csv=nav_csv,
        positions_csv=positions_csv,
        trades_csv=trades_csv,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_cn_dashboard_export.py", "--dashboard-root", str(dashboard_root)],
    )

    checker.main()

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["summary_file"] == str(summary_file)
    assert result["generated_js"] == str(generated_js)
