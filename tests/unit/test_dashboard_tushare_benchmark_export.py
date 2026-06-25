from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "dashboard_exporter",
        ROOT / "scripts" / "export_cn_aggressive_dashboard_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTusharePro:
    def index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        del start_date, end_date
        base = {
            "000300.SH": 1000.0,
            "000905.SH": 2000.0,
            "000852.SH": 3000.0,
            "000688.SH": 4000.0,
            "399006.SZ": 5000.0,
        }[ts_code]
        return pd.DataFrame(
            [
                {"ts_code": ts_code, "trade_date": "20260318", "close": base},
                {"ts_code": ts_code, "trade_date": "20260323", "close": base * 1.10},
            ]
        )


def test_tushare_benchmark_partial_missing_dates_are_not_production_grade():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260318_0930", "2026-03-18", Path("r1"), "", 100.0, 100.0, {}),
        exporter.RecordRun("20260320_0930", "2026-03-20", Path("r2"), "", 100.0, 101.0, {}),
        exporter.RecordRun("20260323_0930", "2026-03-23", Path("r3"), "", 100.0, 102.0, {}),
    ]

    benchmark_export, warnings = exporter.build_tushare_benchmark_export(runs, _FakeTusharePro())
    assert warnings == []
    assert benchmark_export is not None
    assert len(benchmark_export.raw_rows) == 10

    nav_rows, nav_warnings, fieldnames = exporter.build_nav_rows(runs, benchmark_export)
    assert nav_warnings == []
    summary = exporter.benchmark_source_summary(nav_rows, fieldnames, benchmark_export)

    assert summary["source_system"] == "tushare.index_daily"
    assert summary["benchmark_source_status"] == "production_source_partial_missing_dates"
    assert summary["production_grade"] is False
    assert summary["missing_dates"] == ["2026-03-20"]
    assert summary["normalization"] == "tushare_index_daily_close_divided_by_first_valid_close"
    assert nav_rows[0]["csi300_nav"] == "1.00000000"
    assert nav_rows[2]["csi300_nav"] == "1.10000000"
