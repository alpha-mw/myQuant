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

    def trade_cal(self, exchange: str, start_date: str, end_date: str) -> pd.DataFrame:
        del exchange, start_date, end_date
        return pd.DataFrame(
            [
                {"cal_date": "20260318", "is_open": 1, "pretrade_date": "20260317"},
                {"cal_date": "20260320", "is_open": 1, "pretrade_date": "20260319"},
                {"cal_date": "20260323", "is_open": 1, "pretrade_date": "20260320"},
            ]
        )


class _FakeTushareProWithWeekendFill(_FakeTusharePro):
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
                {"ts_code": ts_code, "trade_date": "20260320", "close": base * 1.05},
                {"ts_code": ts_code, "trade_date": "20260323", "close": base * 1.10},
            ]
        )

    def trade_cal(self, exchange: str, start_date: str, end_date: str) -> pd.DataFrame:
        del exchange, start_date, end_date
        return pd.DataFrame(
            [
                {"cal_date": "20260318", "is_open": 1, "pretrade_date": "20260317"},
                {"cal_date": "20260321", "is_open": 0, "pretrade_date": "20260320"},
                {"cal_date": "20260323", "is_open": 1, "pretrade_date": "20260320"},
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
    assert summary["display_continuity_grade"] is False
    assert summary["missing_dates"] == ["2026-03-20"]
    assert summary["previous_trading_day_ffill_dates"] == []
    assert summary["normalization"] == "tushare_index_daily_close_divided_by_first_valid_close_with_trade_cal_previous_trading_day_ffill"
    assert nav_rows[0]["csi300_nav"] == "1.00000000"
    assert nav_rows[2]["csi300_nav"] == "1.10000000"


def test_tushare_benchmark_non_trading_record_uses_previous_trading_day_ffill():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260318_0930", "2026-03-18", Path("r1"), "", 100.0, 100.0, {}),
        exporter.RecordRun("20260321_0930", "2026-03-21", Path("r2"), "", 100.0, 101.0, {}),
        exporter.RecordRun("20260323_0930", "2026-03-23", Path("r3"), "", 100.0, 102.0, {}),
    ]

    benchmark_export, warnings = exporter.build_tushare_benchmark_export(runs, _FakeTushareProWithWeekendFill())
    assert warnings == []
    assert benchmark_export is not None

    nav_rows, nav_warnings, fieldnames = exporter.build_nav_rows(runs, benchmark_export)
    assert nav_warnings == []
    summary = exporter.benchmark_source_summary(nav_rows, fieldnames, benchmark_export)

    assert summary["benchmark_source_status"] == "production_source_with_previous_trading_day_ffill"
    assert summary["production_grade"] is False
    assert summary["display_continuity_grade"] is True
    assert summary["missing_dates"] == []
    assert summary["previous_trading_day_ffill_dates"] == ["2026-03-21"]
    assert summary["coverage_by_date"]["2026-03-21"]["csi300_nav"] == "previous_trading_day_ffill"
    assert summary["value_date_by_date"]["2026-03-21"]["csi300_nav"] == "2026-03-20"
    assert nav_rows[1]["csi300_nav"] == "1.05000000"


def test_tushare_benchmark_snapshot_gap_fill_is_not_production_grade():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun(
            "20260318_0930",
            "2026-03-18",
            Path("r1"),
            "",
            100.0,
            100.0,
            {
                "csi300_nav": 1000.0,
                "csi500_nav": 2000.0,
                "csi1000_nav": 3000.0,
                "star50_nav": 4000.0,
                "chinext_nav": 5000.0,
            },
        ),
        exporter.RecordRun(
            "20260320_0930",
            "2026-03-20",
            Path("r2"),
            "",
            100.0,
            101.0,
            {
                "csi300_nav": 1020.0,
                "csi500_nav": 2040.0,
                "csi1000_nav": 3060.0,
                "star50_nav": 4080.0,
                "chinext_nav": 5100.0,
            },
        ),
        exporter.RecordRun(
            "20260323_0930",
            "2026-03-23",
            Path("r3"),
            "",
            100.0,
            102.0,
            {
                "csi300_nav": 1100.0,
                "csi500_nav": 2200.0,
                "csi1000_nav": 3300.0,
                "star50_nav": 4400.0,
                "chinext_nav": 5500.0,
            },
        ),
    ]

    benchmark_export, warnings = exporter.build_tushare_benchmark_export(
        runs,
        _FakeTusharePro(),
        snapshot_gap_fill=True,
    )
    assert warnings == []
    assert benchmark_export is not None

    nav_rows, nav_warnings, fieldnames = exporter.build_nav_rows(runs, benchmark_export)
    assert nav_warnings == []
    summary = exporter.benchmark_source_summary(nav_rows, fieldnames, benchmark_export)

    assert summary["benchmark_source_status"] == "not_production_grade"
    assert summary["production_grade"] is False
    assert summary["display_continuity_grade"] is True
    assert summary["missing_dates"] == []
    assert summary["snapshot_gap_fill_dates"] == ["2026-03-20"]
    assert "2026-03-20" not in summary["exact_dates"]
    assert summary["snapshot_gap_fill_by_date"]["2026-03-20"] == [
        "csi300_nav",
        "csi500_nav",
        "csi1000_nav",
        "star50_nav",
        "chinext_nav",
    ]
    assert (
        summary["coverage_by_date"]["2026-03-20"]["csi300_nav"]
        == "strategy_record_snapshot_gap_fill"
    )
    assert nav_rows[1]["csi300_nav"] == "1.02000000"
