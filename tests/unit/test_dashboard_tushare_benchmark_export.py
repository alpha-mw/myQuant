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
    assert summary["production_grade"] is True
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


def test_local_benchmark_file_exact_closes_can_be_production_grade(tmp_path):
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260318_0930", "2026-03-18", Path("r1"), "", 100.0, 100.0, {}),
        exporter.RecordRun("20260320_0930", "2026-03-20", Path("r2"), "", 100.0, 101.0, {}),
        exporter.RecordRun("20260323_0930", "2026-03-23", Path("r3"), "", 100.0, 102.0, {}),
    ]
    benchmark_path = tmp_path / "cn_index_benchmark.csv"
    benchmark_path.write_text(
        "\n".join(
            [
                "date,ts_code,close,source_system",
                "2026-03-18,000300.SH,1000,Wind",
                "2026-03-20,000300.SH,1050,Wind",
                "2026-03-23,000300.SH,1100,Wind",
                "2026-03-18,000905.SH,2000,Wind",
                "2026-03-20,000905.SH,2100,Wind",
                "2026-03-23,000905.SH,2200,Wind",
                "2026-03-18,000852.SH,3000,Wind",
                "2026-03-20,000852.SH,3150,Wind",
                "2026-03-23,000852.SH,3300,Wind",
                "2026-03-18,000688.SH,4000,Wind",
                "2026-03-20,000688.SH,4200,Wind",
                "2026-03-23,000688.SH,4400,Wind",
                "2026-03-18,399006.SZ,5000,Wind",
                "2026-03-20,399006.SZ,5250,Wind",
                "2026-03-23,399006.SZ,5500,Wind",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    benchmark_export, warnings = exporter.load_local_benchmark_export(runs, benchmark_path)
    assert warnings == []
    assert benchmark_export is not None
    assert len(benchmark_export.raw_rows) == 15

    nav_rows, nav_warnings, fieldnames = exporter.build_nav_rows(runs, benchmark_export)
    assert nav_warnings == []
    summary = exporter.benchmark_source_summary(nav_rows, fieldnames, benchmark_export)

    assert summary["source_system"] == "Wind"
    assert summary["benchmark_source_status"] == "production_grade"
    assert summary["production_grade"] is True
    assert summary["display_continuity_grade"] is True
    assert summary["calendar_source_system"] == ""
    assert summary["missing_dates"] == []
    assert summary["previous_trading_day_ffill_dates"] == []
    assert summary["normalization"] == "local_index_close_divided_by_first_valid_close"
    assert nav_rows[0]["csi300_nav"] == "1.00000000"
    assert nav_rows[1]["csi300_nav"] == "1.05000000"
    assert nav_rows[2]["csi300_nav"] == "1.10000000"


def test_local_benchmark_file_rejects_sample_or_mock_sources(tmp_path):
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260318_0930", "2026-03-18", Path("r1"), "", 100.0, 100.0, {}),
    ]
    benchmark_path = tmp_path / "cn_index_benchmark.csv"
    benchmark_path.write_text(
        "\n".join(
            [
                "date,ts_code,close,source_system",
                "2026-03-18,000300.SH,1000,sample",
                "2026-03-18,000905.SH,2000,sample",
                "2026-03-18,000852.SH,3000,sample",
                "2026-03-18,000688.SH,4000,sample",
                "2026-03-18,399006.SZ,5000,sample",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    benchmark_export, warnings = exporter.load_local_benchmark_export(runs, benchmark_path)

    assert benchmark_export is None
    assert any("sample/mock" in warning for warning in warnings)


def test_default_benchmark_source_prefers_auto_local_file():
    exporter = _load_exporter()

    assert exporter.DEFAULT_BENCHMARK_SOURCE == "auto"


def test_resolve_local_benchmark_file_falls_back_to_repo_input_for_temp_output(tmp_path, monkeypatch):
    exporter = _load_exporter()
    default_dashboard_root = tmp_path / "portfolio_dashboard"
    default_input = default_dashboard_root / "inputs" / "cn_index_benchmark.csv"
    default_input.parent.mkdir(parents=True)
    default_input.write_text("date,ts_code,close,source_system\n", encoding="utf-8")
    temp_dashboard_root = tmp_path / "temp_dashboard"

    monkeypatch.setattr(exporter, "DEFAULT_DASHBOARD_ROOT", default_dashboard_root)

    assert exporter.resolve_local_benchmark_file(temp_dashboard_root, None) == default_input


def test_industry_equal_weight_nav_is_built_from_local_parquet(tmp_path):
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260318_0930", "2026-03-18", Path("r1"), "", 100.0, 100.0, {}),
        exporter.RecordRun("20260319_0930", "2026-03-19", Path("r2"), "", 100.0, 101.0, {}),
        exporter.RecordRun("20260320_0930", "2026-03-20", Path("r3"), "", 100.0, 102.0, {}),
    ]
    benchmark_path = tmp_path / "cn_index_benchmark.csv"
    lines = ["date,ts_code,close,source_system"]
    for ts_code, close in [
        ("000300.SH", 1000),
        ("000905.SH", 2000),
        ("000852.SH", 3000),
        ("000688.SH", 4000),
        ("399006.SZ", 5000),
    ]:
        lines.extend(
            [
                f"2026-03-18,{ts_code},{close},Wind",
                f"2026-03-19,{ts_code},{close * 1.01},Wind",
                f"2026-03-20,{ts_code},{close * 1.02},Wind",
            ]
        )
    benchmark_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stock_basic_root = tmp_path / "stock_basic"
    stock_basic_root.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry": "IT设备"},
            {"ts_code": "000002.SZ", "industry": "半导体"},
            {"ts_code": "000003.SZ", "industry": "银行"},
        ]
    ).to_parquet(stock_basic_root / "part.parquet", index=False)
    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260318", "close": 100.0, "adj_close": 100.0},
            {"ts_code": "000001.SZ", "trade_date": "20260319", "close": 110.0, "adj_close": 110.0},
            {"ts_code": "000001.SZ", "trade_date": "20260320", "close": 121.0, "adj_close": 121.0},
            {"ts_code": "000002.SZ", "trade_date": "20260318", "close": 50.0, "adj_close": 50.0},
            {"ts_code": "000002.SZ", "trade_date": "20260319", "close": 45.0, "adj_close": 45.0},
            {"ts_code": "000002.SZ", "trade_date": "20260320", "close": 49.5, "adj_close": 49.5},
            {"ts_code": "000003.SZ", "trade_date": "20260318", "close": 10.0, "adj_close": 10.0},
            {"ts_code": "000003.SZ", "trade_date": "20260319", "close": 20.0, "adj_close": 20.0},
            {"ts_code": "000003.SZ", "trade_date": "20260320", "close": 30.0, "adj_close": 30.0},
        ]
    ).to_parquet(bars_root / "part.parquet", index=False)

    benchmark_export, warnings = exporter.load_local_benchmark_export(runs, benchmark_path)
    assert warnings == []
    assert benchmark_export is not None

    enhanced_export, industry_warnings = exporter.attach_industry_equal_weight_nav(
        runs,
        benchmark_export,
        bars_root=bars_root,
        stock_basic_root=stock_basic_root,
        industries=("IT设备", "半导体"),
    )

    assert industry_warnings == []
    assert enhanced_export is not None
    nav_rows, nav_warnings, fieldnames = exporter.build_nav_rows(runs, enhanced_export)
    assert nav_warnings == []
    summary = exporter.benchmark_source_summary(nav_rows, fieldnames, enhanced_export)

    assert "industry_ew_nav" in fieldnames
    assert nav_rows[0]["industry_ew_nav"] == "1.00000000"
    assert nav_rows[1]["industry_ew_nav"] == "1.00000000"
    assert nav_rows[2]["industry_ew_nav"] == "1.10000000"
    assert summary["production_grade"] is True
    assert summary["industry_equal_weight"]["member_count"] == 2
    assert summary["industry_equal_weight"]["industry_list"] == ["IT设备", "半导体"]
    assert any(row["field"] == "industry_ew_nav" for row in enhanced_export.raw_rows)
