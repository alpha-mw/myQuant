from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


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
    assert all("UNSPECIFIED_RECORD_THEME" not in warning for warning in warnings)
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


def test_build_nav_rows_preserves_raw_portfolio_nav_and_uses_star50_as_main_benchmark():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun(
            "20260701_0930",
            "2026-07-01",
            Path("r1"),
            "",
            100.0,
            150.0,
            {},
        ),
        exporter.RecordRun(
            "20260702_0930",
            "2026-07-02",
            Path("r2"),
            "",
            100.0,
            165.0,
            {},
        ),
    ]
    benchmark_export = exporter.BenchmarkExport(
        values_by_date={
            "2026-07-01": {"star50_nav": 1.0, "csi300_nav": 1.0, "chinext_nav": 1.0},
            "2026-07-02": {"star50_nav": 1.10, "csi300_nav": 0.90, "chinext_nav": 1.20},
        },
        raw_rows=[],
        source_system="local",
        normalization="local_index_close_divided_by_first_valid_close",
        status_hint="production_source",
        notes=[],
        coverage_by_date={},
        value_date_by_date={},
    )

    nav_rows, warnings, fieldnames = exporter.build_nav_rows(runs, benchmark_export)

    assert warnings == []
    assert "portfolio_nav_raw" in fieldnames
    assert "portfolio_nav_rebased" in fieldnames
    assert nav_rows[0]["portfolio_nav"] == "1.50000000"
    assert nav_rows[0]["portfolio_nav_raw"] == "1.50000000"
    assert nav_rows[0]["portfolio_nav_rebased"] == "1.00000000"
    assert nav_rows[1]["portfolio_nav"] == "1.65000000"
    assert nav_rows[1]["portfolio_nav_raw"] == "1.65000000"
    assert nav_rows[1]["portfolio_nav_rebased"] == "1.10000000"
    assert nav_rows[1]["benchmark_main_nav"] == "1.10000000"
    assert nav_rows[1]["benchmark_nav"] == "0.90000000"


def test_build_nav_rows_chain_links_manifest_funding_without_diluting_return():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260102_1000", "2026-01-02", Path("r1"), "", 100.0, 100.0, {}),
        exporter.RecordRun("20260105_1000", "2026-01-05", Path("r2"), "", 100.0, 110.0, {}),
        exporter.RecordRun(
            "20260106_1000",
            "2026-01-06",
            Path("r3"),
            "",
            1000.0,
            1010.0,
            {},
            (
                {
                    "date": "2026-01-06",
                    "amount": 900.0,
                    "source": "unit_test",
                    "total_value_before": 110.0,
                    "total_value_after": 1010.0,
                },
            ),
        ),
    ]

    nav_rows, warnings, fieldnames = exporter.build_nav_rows(runs)

    assert warnings == ["记录中未找到可用 benchmark 指数快照，benchmark_main_nav 使用 1.0 平线占位。"]
    assert "external_funding_cash_flow" in fieldnames
    assert nav_rows[1]["portfolio_nav"] == "1.10000000"
    assert nav_rows[2]["portfolio_nav"] == "1.10000000"
    assert nav_rows[2]["portfolio_nav_raw"] == "1.01000000"
    assert nav_rows[2]["portfolio_nav_rebased"] == "1.10000000"
    assert nav_rows[2]["portfolio_return"] == "0.00000000"
    assert nav_rows[2]["initial_capital"] == "1000.00"
    assert nav_rows[2]["total_value_after"] == "1010.00"
    assert nav_rows[2]["external_funding_cash_flow"] == "900.00"
    assert nav_rows[2]["portfolio_units"] == "918.18181818"


def test_build_nav_rows_uses_expanded_units_for_post_funding_return():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260102_1000", "2026-01-02", Path("r1"), "", 100.0, 100.0, {}),
        exporter.RecordRun("20260105_1000", "2026-01-05", Path("r2"), "", 100.0, 110.0, {}),
        exporter.RecordRun(
            "20260106_1000",
            "2026-01-06",
            Path("r3"),
            "",
            1000.0,
            1020.1,
            {},
            (
                {
                    "date": "2026-01-06",
                    "amount": 900.0,
                    "source": "unit_test",
                    "total_value_before": 110.0,
                    "total_value_after": 1010.0,
                },
            ),
        ),
    ]

    nav_rows, _, _ = exporter.build_nav_rows(runs)

    assert nav_rows[1]["portfolio_nav"] == "1.10000000"
    assert nav_rows[2]["portfolio_nav"] == "1.11100000"
    assert nav_rows[2]["portfolio_return"] == "0.01000000"


def test_build_nav_rows_fails_closed_when_funding_lacks_timing_valuations():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260102_1000", "2026-01-02", Path("r1"), "", 100.0, 110.0, {}),
        exporter.RecordRun(
            "20260105_1000",
            "2026-01-05",
            Path("r2"),
            "",
            1000.0,
            1010.0,
            {},
            ({"date": "2026-01-05", "amount": 900.0, "source": "unit_test"},),
        ),
    ]

    with pytest.raises(ValueError, match="requires total_value_before and total_value_after"):
        exporter.build_nav_rows(runs)


def test_build_nav_rows_fails_closed_when_capital_changes_without_funding_evidence():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260102_1000", "2026-01-02", Path("r1"), "", 100.0, 110.0, {}),
        exporter.RecordRun("20260105_1000", "2026-01-05", Path("r2"), "", 1000.0, 1010.0, {}),
    ]

    with pytest.raises(ValueError, match="without manifest-backed funding evidence"):
        exporter.build_nav_rows(runs)


def test_discover_record_runs_retains_earlier_same_day_funding_evidence(tmp_path):
    exporter = _load_exporter()
    record_root = tmp_path / "records"
    funding_run = record_root / "20260105_0900"
    latest_run = record_root / "20260105_1400"
    funding_run.mkdir(parents=True)
    latest_run.mkdir(parents=True)
    (funding_run / "pnl_summary.csv").write_text(
        "initial_capital,total_value_after,record_time\n1000,1000,2026-01-05 09:00:00\n",
        encoding="utf-8",
    )
    (latest_run / "pnl_summary.csv").write_text(
        "initial_capital,total_value_after,record_time\n1000,1001,2026-01-05 14:00:00\n",
        encoding="utf-8",
    )
    supplement = {
        "amount": 900.0,
        "capital_base_effective_from": "2026-01-05",
        "total_value_before": 100.0,
        "total_value_after": 1000.0,
        "source": "unit_test_confirmed_funding",
    }
    (funding_run / "manual_execution_manifest.json").write_text(
        json.dumps({"manual_funding_supplement": supplement}),
        encoding="utf-8",
    )

    runs, warnings = exporter.discover_record_runs(record_root, require_effective_manual=False)

    assert warnings == []
    assert [run.run_id for run in runs] == ["20260105_1400"]
    assert runs[0].funding_cash_flows == (
        {
            "date": "2026-01-05",
            "amount": 900.0,
            "source": "unit_test_confirmed_funding",
            "schema_version": "",
            "evidence_path": "",
            "capital_base_after": None,
            "total_value_before": 100.0,
            "total_value_after": 1000.0,
        },
    )


def test_discover_record_runs_requires_effective_manual_ledger_and_manifest(tmp_path):
    exporter = _load_exporter()
    record_root = tmp_path / "records"
    legacy_run = record_root / "20260707_0910"
    valid_run = record_root / "20260708_0933"
    legacy_run.mkdir(parents=True)
    valid_run.mkdir(parents=True)
    pnl_text = "initial_capital,total_value_after,record_time\n100,101,2026-07-08 09:33:00\n"
    for run_dir in (legacy_run, valid_run):
        (run_dir / "pnl_summary.csv").write_text(pnl_text, encoding="utf-8")
    (legacy_run / "ledger.csv").write_text(
        "symbol,name,current_value,market_weight\n000001.SZ,Legacy,101,1\n",
        encoding="utf-8",
    )
    (valid_run / "ledger_after_manual_switch.csv").write_text(
        "symbol,name,current_value,market_weight\n000002.SZ,Manual,101,1\n",
        encoding="utf-8",
    )
    (valid_run / "manual_execution_manifest.json").write_text('{"status":"ok"}\n', encoding="utf-8")

    runs, warnings = exporter.discover_record_runs(record_root)

    assert [run.run_id for run in runs] == ["20260708_0933"]
    assert any("20260707_0910" in warning and "ledger_after_manual_switch.csv" in warning for warning in warnings)


def test_build_positions_rows_never_falls_back_to_legacy_ledger(tmp_path):
    exporter = _load_exporter()
    run_dir = tmp_path / "20260708_0910"
    run_dir.mkdir()
    (run_dir / "ledger.csv").write_text(
        "symbol,name,current_value,market_weight\n000001.SZ,Legacy,101,1\n",
        encoding="utf-8",
    )
    run = exporter.RecordRun("20260708_0910", "2026-07-08", run_dir, "", 100.0, 101.0, {})

    rows, warnings = exporter.build_positions_rows(
        [run],
        {"000002.SZ": {"sector": "测试行业", "sub_sector": "测试子行业"}},
    )

    assert rows == []
    assert any("ledger_after_manual_switch.csv" in warning for warning in warnings)


def test_build_positions_rows_exports_fifo_opening_cost_fields(tmp_path):
    exporter = _load_exporter()
    run_dir = tmp_path / "20260708_0910"
    run_dir.mkdir()
    (run_dir / "ledger_after_manual_switch.csv").write_text(
        (
            "symbol,name,shares,avg_cost,cost_basis,current_price,current_value,market_weight\n"
            "000002.SZ,Manual,300,12.50,3750,14.20,4260,0.25\n"
        ),
        encoding="utf-8",
    )
    run = exporter.RecordRun("20260708_0910", "2026-07-08", run_dir, "", 100.0, 101.0, {})

    rows, warnings = exporter.build_positions_rows(
        [run],
        {"000002.SZ": {"sector": "测试行业", "sub_sector": "测试子行业"}},
    )

    assert all("UNSPECIFIED_RECORD_THEME" not in warning for warning in warnings)
    assert rows[0]["quantity"] == "300"
    assert rows[0]["avg_cost"] == "12.5000"
    assert rows[0]["cost_basis"] == "3750.00"
    assert rows[0]["current_price"] == "14.2000"


def test_positions_carry_forward_prior_explicit_theme_without_lookahead(tmp_path):
    exporter = _load_exporter()
    early_dir = tmp_path / "20260102_1000"
    source_dir = tmp_path / "20260105_1000"
    later_dir = tmp_path / "20260106_1000"
    for run_dir in (early_dir, source_dir, later_dir):
        run_dir.mkdir()
    (source_dir / "market_snapshot.json").write_text(
        json.dumps({"theme_strength": [{"theme": "先进材料", "symbols": ["000002.SZ"]}]}),
        encoding="utf-8",
    )
    (later_dir / "ledger_after_manual_switch.csv").write_text(
        "symbol,name,current_value,market_weight\n000002.SZ,Manual,110,1\n",
        encoding="utf-8",
    )
    runs = [
        exporter.RecordRun("20260102_1000", "2026-01-02", early_dir, "", 100.0, 100.0, {}),
        exporter.RecordRun("20260105_1000", "2026-01-05", source_dir, "", 100.0, 105.0, {}),
        exporter.RecordRun("20260106_1000", "2026-01-06", later_dir, "", 100.0, 110.0, {}),
    ]

    historical = exporter.build_historical_theme_maps(runs)
    rows, warnings = exporter.build_positions_rows(
        [runs[-1]],
        {"000002.SZ": {"sector": "测试行业", "sub_sector": ""}},
        historical_theme_by_date=historical,
    )

    assert "000002.SZ" not in historical["2026-01-02"]
    assert historical["2026-01-05"]["000002.SZ"] == ("先进材料", "20260105_1000")
    assert rows[0]["theme"] == "先进材料"
    assert rows[0]["theme_source"] == "prior_strategy_record:20260105_1000"
    assert not any("stock_basic industry" in warning for warning in warnings)


def test_export_summary_reports_effective_manual_ledger_status(tmp_path, monkeypatch):
    exporter = _load_exporter()
    record_root = tmp_path / "records"
    run_dir = record_root / "20260708_0933"
    run_dir.mkdir(parents=True)
    (run_dir / "pnl_summary.csv").write_text(
        "initial_capital,total_value_after,record_time\n100,110,2026-07-08 09:33:00\n",
        encoding="utf-8",
    )
    ledger_path = run_dir / "ledger_after_manual_switch.csv"
    ledger_path.write_text(
        "symbol,name,current_value,market_weight\n000001.SZ,Manual,110,1\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manual_execution_manifest.json"
    manifest_path.write_text('{"status":"ok"}\n', encoding="utf-8")
    benchmark_path = tmp_path / "cn_index_benchmark.csv"
    benchmark_path.write_text(
        "\n".join(
            [
                "date,ts_code,close,source_system",
                "2026-07-08,000300.SH,1000,Wind",
                "2026-07-08,000905.SH,2000,Wind",
                "2026-07-08,000852.SH,3000,Wind",
                "2026-07-08,000688.SH,4000,Wind",
                "2026-07-08,399006.SZ,5000,Wind",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dashboard_root = tmp_path / "dashboard"
    monkeypatch.setattr(exporter, "DEFAULT_STOCK_BASIC_ROOT", tmp_path / "missing_stock_basic")

    summary = exporter.export(
        record_root,
        dashboard_root,
        benchmark_source="local",
        benchmark_file=benchmark_path,
    )
    written_summary = json.loads(
        (dashboard_root / "generated" / "export_summary.json").read_text(encoding="utf-8")
    )

    status = summary["effective_manual_ledger_status"]
    assert written_summary["effective_manual_ledger_status"] == status
    assert status == {
        "status": "valid",
        "record_id": "20260708_0933",
        "ledger_path": str(ledger_path),
        "manifest_path": str(manifest_path),
        "legacy_ledger_fallback_used": False,
    }


def test_export_preserves_full_pnl_nav_history_separate_from_manual_position_baseline(tmp_path, monkeypatch):
    exporter = _load_exporter()
    record_root = tmp_path / "records"
    early_run = record_root / "20260318_1451"
    manual_run = record_root / "20260526_0934"
    early_run.mkdir(parents=True)
    manual_run.mkdir(parents=True)
    (early_run / "pnl_summary.csv").write_text(
        "initial_capital,total_value_after,record_time\n1000000,1023097,2026-03-18 14:51:18\n",
        encoding="utf-8",
    )
    (manual_run / "pnl_summary.csv").write_text(
        "initial_capital,total_value_after,record_time\n1000000,1425877,2026-05-26 09:34:29\n",
        encoding="utf-8",
    )
    ledger_path = manual_run / "ledger_after_manual_switch.csv"
    ledger_path.write_text(
        "symbol,name,current_value,market_weight\n000001.SZ,Manual,1425877,1\n",
        encoding="utf-8",
    )
    manifest_path = manual_run / "manual_execution_manifest.json"
    manifest_path.write_text('{"status":"ok"}\n', encoding="utf-8")
    benchmark_path = tmp_path / "cn_index_benchmark.csv"
    benchmark_lines = ["date,ts_code,close,source_system"]
    for date, scale in [("2026-03-18", 1.0), ("2026-05-26", 1.1)]:
        for ts_code, close in [
            ("000300.SH", 1000),
            ("000905.SH", 2000),
            ("000852.SH", 3000),
            ("000688.SH", 4000),
            ("399006.SZ", 5000),
        ]:
            benchmark_lines.append(f"{date},{ts_code},{close * scale},Wind")
    benchmark_path.write_text("\n".join(benchmark_lines) + "\n", encoding="utf-8")
    dashboard_root = tmp_path / "dashboard"
    monkeypatch.setattr(exporter, "DEFAULT_STOCK_BASIC_ROOT", tmp_path / "missing_stock_basic")

    summary = exporter.export(
        record_root,
        dashboard_root,
        benchmark_source="local",
        benchmark_file=benchmark_path,
    )
    nav_rows = exporter.read_csv_rows(dashboard_root / "generated" / "nav_records.csv")
    positions_rows = exporter.read_csv_rows(dashboard_root / "generated" / "positions_records.csv")
    generated_js = (dashboard_root / "js" / "generated_records.js").read_text(encoding="utf-8")

    assert [row["date"] for row in nav_rows] == ["2026-03-18", "2026-05-26"]
    assert nav_rows[0]["portfolio_nav"] == "1.02309700"
    assert nav_rows[0]["portfolio_nav_raw"] == "1.02309700"
    assert nav_rows[0]["portfolio_nav_rebased"] == "1.00000000"
    assert nav_rows[1]["portfolio_nav"] == "1.42587700"
    assert nav_rows[1]["portfolio_nav_raw"] == "1.42587700"
    assert nav_rows[1]["portfolio_nav_rebased"] == "1.39368701"
    assert summary["record_count"] == 2
    assert summary["manual_record_count"] == 1
    assert summary["effective_manual_ledger_status"]["record_id"] == "20260526_0934"
    assert positions_rows and positions_rows[0]["date"] == "2026-05-26"
    assert "缺少可读 ledger_after_manual_switch.csv" not in generated_js


def test_build_trade_rows_includes_local_manual_and_orders_csv_fills(tmp_path):
    exporter = _load_exporter()
    run_dir = tmp_path / "20260707_1046"
    run_dir.mkdir()
    (run_dir / "manual_switch_and_take_profit_orders.csv").write_text(
        "\n".join(
            [
                "timestamp,status,action,symbol,name,shares,price,trade_value,realized_pnl,reason",
                "2026-07-07 11:09:35 CST,filled_local_manual,clear_risk_sell,603078.SH,江化微,1300,53.66,69758.0,2028.0,user_confirmed_clear_risk",
                "2026-07-07 11:09:35 CST,watch_only_no_execution,watch_only_sell,600000.SH,未成交,100,10,1000,0,not executed",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "orders.csv").write_text(
        "\n".join(
            [
                "timestamp,action,symbol,name,shares,price,trade_value,realized_pnl,reason",
                "2026-07-07 10:00:00 CST,buy,300285.SZ,国瓷材料,700,88.28,61796.0,0.0,legacy manual buy",
                "2026-07-07 11:09:35 CST,sell,603078.SH,江化微,1300,53.66,69758.0,2028.0,duplicate legacy sell",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run = exporter.RecordRun("20260707_1046", "2026-07-07", run_dir, "", 100.0, 101.0, {})

    rows = exporter.build_trade_rows([run], {})

    assert [(row["trade_date"], row["ticker"], row["side"], row["quantity"]) for row in rows] == [
        ("2026-07-07", "300285.SZ", "buy", "700"),
        ("2026-07-07", "603078.SH", "sell", "1300"),
    ]
    assert rows[1]["price"] == "53.6600"
    assert rows[1]["trade_amount"] == "69758.00"


def test_build_trade_rows_skips_incomplete_executed_records(tmp_path):
    exporter = _load_exporter()
    run_dir = tmp_path / "20260707_1046"
    run_dir.mkdir()
    (run_dir / "manual_switch_and_take_profit_orders.csv").write_text(
        "\n".join(
            [
                "timestamp,status,action,symbol,name,shares,price,trade_value,realized_pnl,reason",
                "2026-07-07 11:09:35 CST,filled_local_manual,buy,300285.SZ,国瓷材料,700,88.28,61796.0,0.0,complete",
                "2026-07-07 11:10:00 CST,filled_local_manual,sell,603078.SH,江化微,1300,,69758.0,2028.0,missing price",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run = exporter.RecordRun("20260707_1046", "2026-07-07", run_dir, "", 100.0, 101.0, {})
    warnings: list[str] = []
    completeness: dict[str, object] = {}

    rows = exporter.build_trade_rows([run], {}, warnings=warnings, completeness=completeness)

    assert [(row["ticker"], row["side"], row["quantity"]) for row in rows] == [
        ("300285.SZ", "buy", "700")
    ]
    assert completeness["status"] == "partial"
    assert completeness["executed_source_rows"] == 2
    assert completeness["exported_rows"] == 1
    assert completeness["skipped_incomplete_rows"] == 1
    assert any("trade_record_incomplete" in warning and "price" in warning for warning in warnings)


def test_dashboard_display_warnings_aggregate_skipped_legacy_records():
    exporter = _load_exporter()
    warnings = [
        "20260318_1146: 缺少可读 ledger_after_manual_switch.csv，已跳过。",
        "20260318_1417: 缺少可读 ledger_after_manual_switch.csv，已跳过。",
        "benchmark_source_status=production_source_partial_latest_unavailable；source_system=local；Dashboard benchmark 仅供临时展示，不是正式投委会口径。",
    ]

    display = exporter.dashboard_display_warnings(warnings)

    assert display == [
        "benchmark_source_status=production_source_partial_latest_unavailable；source_system=local；Dashboard benchmark 仅供临时展示，不是正式投委会口径。",
        "2 个历史记录因缺少可读 ledger_after_manual_switch.csv 已跳过；详见 export_summary.json。",
    ]


def test_dashboard_display_infos_carry_nonblocking_theme_fallback():
    exporter = _load_exporter()
    theme_warning = "133 条持仓记录缺少显式 theme，已使用 stock_basic industry 生成 '行业: <sector>' 回退标签。"
    benchmark_warning = (
        "benchmark_source_status=production_source_partial_latest_unavailable；"
        "source_system=local；Dashboard benchmark 仅供临时展示，不是正式投委会口径。"
    )

    assert exporter.dashboard_display_warnings([theme_warning, benchmark_warning]) == [benchmark_warning]
    assert exporter.dashboard_display_infos([theme_warning, benchmark_warning]) == [theme_warning]


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
