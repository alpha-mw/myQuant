from __future__ import annotations

import csv
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market.pit_universe import (
    LIST_STATUS_LISTED,
    PITUniverseRecord,
    PITUniverseStore,
)


ROOT = Path(__file__).resolve().parents[2]


def _formal_trading_calendar(open_dates: list[str]) -> dict[str, object]:
    return {
        "status": "available",
        "source_system": "strict_parquet.cn_bars.trade_date",
        "expected_open_dates": open_dates,
        "prior_open_date": None,
        "market_snapshot": {
            "latest_pointer_path_summary": "<data_root>/parquet/cn/_latest.json",
            "latest_pointer_sha256": "a" * 64,
            "snapshot_id": "unit-test",
            "table_root_path_summary": (
                "<data_root>/parquet/cn/_snapshots/unit-test/table/bars"
            ),
            "fallback_used": False,
        },
    }


def _write_active_cn_market(
    data_root: Path,
    rows: list[dict[str, object]],
    *,
    snapshot_id: str = "dashboard-snapshot-v4",
) -> dict[str, Path]:
    """Create a minimal healthy v4 pointer plus deliberately stale legacy bars."""

    pit_store = PITUniverseStore(
        root_dir=data_root / "parquet" / "cn" / "reference",
        raw_root=data_root / "pit_raw",
        compatibility_path=data_root / "pit_compat.json",
    )
    symbols = sorted({str(row["ts_code"]) for row in rows})
    observed_at = (
        "2026-07-16T00:00:"
        f"{int(hashlib.sha256(snapshot_id.encode()).hexdigest()[:2], 16) % 60:02d}Z"
    )
    published = pit_store.write_snapshot(
        raw_records=[
            PITUniverseRecord(
                symbol=symbol,
                source_list_status=LIST_STATUS_LISTED,
                list_date="20200101",
                observed_at=observed_at,
                source_run_id=snapshot_id,
            )
            for symbol in symbols
        ],
        observed_at=observed_at,
        source_run_id=snapshot_id,
    )
    snapshot_root = data_root / "parquet" / "cn" / "_snapshots" / snapshot_id
    table_root = snapshot_root / "table" / "bars"
    serving_root = snapshot_root / "serving" / "bars"
    table_root.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(table_root / "part.parquet", index=False)
    for symbol in symbols:
        symbol_root = serving_root / f"symbol={symbol}"
        symbol_root.mkdir(parents=True)
        frame[frame["ts_code"].eq(symbol)].to_parquet(
            symbol_root / "bars.parquet",
            index=False,
        )
    legacy_root = data_root / "parquet" / "cn" / "bars"
    legacy_root.mkdir(parents=True, exist_ok=True)
    legacy_frame = frame.copy()
    legacy_frame["trade_date"] = "19990101"
    legacy_frame.to_parquet(legacy_root / "stale.parquet", index=False)

    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": False,
        "pit_membership_path": str(published["canonical_path"]),
        "pit_membership_sha256": str(published["canonical_sha256"]),
        "pit_generation_id": str(published["generation_id"]),
        "pit_generation_manifest_path": str(
            published["generation_manifest_path"]
        ),
        "pit_generation_manifest_sha256": str(
            published["generation_manifest_sha256"]
        ),
    }
    manifest_path = data_root / "parquet" / "cn" / "_snapshots" / f"{snapshot_id}.json"
    pointer = {
        "status": "OK",
        "snapshot_id": snapshot_id,
        "latest_complete_trade_date": max(str(row["trade_date"]) for row in rows),
        "latest_trade_date": max(str(row["trade_date"]) for row in rows),
        "manifest_path": str(manifest_path),
        "table_root": str(table_root),
        "derived_serving_root": str(serving_root),
        "coverage": coverage,
        "blockers": [],
    }
    manifest_path.write_text(json.dumps(pointer), encoding="utf-8")
    latest_path = data_root / "parquet" / "cn" / "_latest.json"
    latest_path.write_text(json.dumps(pointer), encoding="utf-8")
    return {
        "data_root": data_root,
        "latest": latest_path,
        "table_root": table_root,
        "legacy_root": legacy_root,
    }


def _write_valid_manual_manifest(
    run_dir: Path,
    *,
    recorded_at: str,
    total_value_after: float,
    status: str = "filled_local_manual_paper_rebalance",
) -> Path:
    ledger_path = run_dir / "ledger_after_manual_switch.csv"
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    manifest_path = run_dir / "manual_execution_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cn_aggressive_manual_execution.v2",
                "status": status,
                "execution_status": status,
                "recorded_at": recorded_at,
                "next_ledger_path": ledger_path.name,
                "ledger_after_manual_switch_csv_sha256": ledger_sha,
                "effective_manual_holding_count": 1,
                "total_value_after": total_value_after,
                "no_broker_api_called": True,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


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
        raise AssertionError("Dashboard benchmark must not call Tushare trade_cal")


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
        raise AssertionError("Dashboard benchmark must not call Tushare trade_cal")


def test_tushare_benchmark_partial_missing_dates_are_not_production_grade():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260318_0930", "2026-03-18", Path("r1"), "", 100.0, 100.0, {}),
        exporter.RecordRun("20260320_0930", "2026-03-20", Path("r2"), "", 100.0, 101.0, {}),
        exporter.RecordRun("20260323_0930", "2026-03-23", Path("r3"), "", 100.0, 102.0, {}),
    ]

    benchmark_export, warnings = exporter.build_tushare_benchmark_export(
        runs,
        _FakeTusharePro(),
        formal_trading_calendar=_formal_trading_calendar(
            ["2026-03-18", "2026-03-20", "2026-03-23"]
        ),
    )
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
    assert summary["normalization"] == "tushare_index_daily_close_divided_by_first_valid_close_with_strict_parquet_calendar_ffill"
    assert nav_rows[0]["csi300_nav"] == "1.00000000"
    assert nav_rows[2]["csi300_nav"] == "1.10000000"


def test_tushare_benchmark_non_trading_record_uses_previous_trading_day_ffill():
    exporter = _load_exporter()
    runs = [
        exporter.RecordRun("20260318_0930", "2026-03-18", Path("r1"), "", 100.0, 100.0, {}),
        exporter.RecordRun("20260321_0930", "2026-03-21", Path("r2"), "", 100.0, 101.0, {}),
        exporter.RecordRun("20260323_0930", "2026-03-23", Path("r3"), "", 100.0, 102.0, {}),
    ]

    benchmark_export, warnings = exporter.build_tushare_benchmark_export(
        runs,
        _FakeTushareProWithWeekendFill(),
        formal_trading_calendar=_formal_trading_calendar(
            ["2026-03-18", "2026-03-20", "2026-03-23"]
        ),
    )
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
        formal_trading_calendar=_formal_trading_calendar(
            ["2026-03-18", "2026-03-20", "2026-03-23"]
        ),
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

    assert exporter.DEFAULT_BENCHMARK_SOURCE == "local"


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
        "symbol,name,current_value,market_weight,shares,avg_cost,current_price\n"
        "000002.SZ,Manual,101,1,10,10,10.1\n",
        encoding="utf-8",
    )
    _write_valid_manual_manifest(
        valid_run,
        recorded_at="2026-07-08T09:33:00+08:00",
        total_value_after=101.0,
    )

    runs, warnings = exporter.discover_record_runs(record_root)

    assert [run.run_id for run in runs] == ["20260708_0933"]
    assert any("20260707_0910" in warning and "manual_execution_manifest.json" in warning for warning in warnings)


def test_manual_baseline_selection_uses_manifest_time_not_later_directory_name(tmp_path):
    exporter = _load_exporter()
    record_root = tmp_path / "records"
    valid_run = record_root / "20260703_0932"
    stale_run = record_root / "20260703_1100"
    invalid_run = record_root / "20260703_1200"
    for run_dir, symbol in (
        (valid_run, "VALID.SZ"),
        (stale_run, "STALE.SZ"),
        (invalid_run, "INVALID.SZ"),
    ):
        run_dir.mkdir(parents=True)
        (run_dir / "pnl_summary.csv").write_text(
            "initial_capital,total_value_after,record_time\n"
            "100,110,2026-07-03 09:32:00\n",
            encoding="utf-8",
        )
        (run_dir / "ledger_after_manual_switch.csv").write_text(
            "symbol,name,current_value,market_weight,shares,avg_cost,current_price\n"
            f"{symbol},{symbol},110,1,10,10,11\n",
            encoding="utf-8",
        )
    _write_valid_manual_manifest(
        valid_run,
        recorded_at="2026-07-03T11:06:56+08:00",
        total_value_after=110.0,
    )
    _write_valid_manual_manifest(
        stale_run,
        recorded_at="2026-07-03T11:00:39+08:00",
        total_value_after=110.0,
        status="no_action_carry_forward",
    )
    invalid_ledger = invalid_run / "ledger_after_manual_switch.csv"
    (invalid_run / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "invalidated_price_basis_no_execution",
                "execution_status": "invalidated_price_basis_no_execution",
                "recorded_at": "2026-07-03T12:00:00+08:00",
                "next_ledger_path": invalid_ledger.name,
                "ledger_after_manual_switch_csv_sha256": hashlib.sha256(
                    invalid_ledger.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    runs, warnings = exporter.discover_record_runs(record_root)

    assert [run.run_id for run in runs] == ["20260703_0932"]
    assert runs[0].manual_order_key == ("20260703110656", "20260703_0932")
    assert any("manual_manifest_status_invalid" in warning for warning in warnings)


def test_manual_baseline_rejects_status_ok_traversal_and_hash_mismatch(tmp_path):
    exporter = _load_exporter()
    record_root = tmp_path / "records"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ledger_after_manual_switch.csv").write_text(
        "symbol,name,current_value,market_weight,shares,avg_cost,current_price\n"
        "OUT.SZ,Outside,110,1,10,10,11\n",
        encoding="utf-8",
    )
    cases = (
        ("20260703_1000", "ok", "ledger_after_manual_switch.csv", None),
        (
            "20260703_1100",
            "filled_local_manual_paper_rebalance",
            "../../outside/ledger_after_manual_switch.csv",
            None,
        ),
        (
            "20260703_1200",
            "filled_local_manual_paper_rebalance",
            "ledger_after_manual_switch.csv",
            "0" * 64,
        ),
    )
    for run_id, status, next_path, forced_sha in cases:
        run_dir = record_root / run_id
        run_dir.mkdir(parents=True)
        ledger = run_dir / "ledger_after_manual_switch.csv"
        ledger.write_text(
            "symbol,name,current_value,market_weight,shares,avg_cost,current_price\n"
            "TEST.SZ,Test,110,1,10,10,11\n",
            encoding="utf-8",
        )
        (run_dir / "pnl_summary.csv").write_text(
            "initial_capital,total_value_after,record_time\n"
            "100,110,2026-07-03 10:00:00\n",
            encoding="utf-8",
        )
        (run_dir / "manual_execution_manifest.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "execution_status": status,
                    "recorded_at": "2026-07-03T10:00:00+08:00",
                    "next_ledger_path": next_path,
                    "ledger_after_manual_switch_csv_sha256": forced_sha
                    or hashlib.sha256(ledger.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

    runs, warnings = exporter.discover_record_runs(record_root)

    assert runs == []
    assert any("manual_manifest_status_invalid" in warning for warning in warnings)
    assert any("manual_manifest_next_ledger_path_unsafe" in warning for warning in warnings)
    assert any("manual_ledger_sha256_mismatch" in warning for warning in warnings)


def test_manual_ledger_undeclared_sha_warnings_are_aggregated_without_paths(tmp_path):
    exporter = _load_exporter()
    ledger = tmp_path / "ledger_after_manual_switch.csv"
    ledger.write_text(
        "symbol,shares,avg_cost\nTEST.SZ,10,10\n",
        encoding="utf-8",
    )
    ledger_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    run = exporter.RecordRun(
        "20260708_0933",
        "2026-07-08",
        tmp_path,
        "2026-07-08T09:33:00+08:00",
        100.0,
        110.0,
        {},
        manual_ledger_path=ledger,
        manual_ledger_sha256=ledger_sha,
        manual_ledger_sha_declared=False,
        manual_manifest={"status": "filled_local_manual_paper_rebalance"},
    )
    warnings = [
        f"run-{index}: manual_ledger_sha_not_declared；private/path/{index}"
        for index in range(3)
    ]

    summarized = exporter.aggregate_manual_record_warnings(warnings, run)

    assert len(summarized) == 1
    assert "count=3" in summarized[0]
    assert "effective_ledger_sha_declared=false" in summarized[0]
    assert "effective_computed_sha_readback_verified=true" in summarized[0]
    assert "private/path" not in summarized[0]
    assert "run-" not in summarized[0]


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


def test_positions_use_strict_industry_without_historical_theme_fallback(tmp_path):
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

    rows, warnings = exporter.build_positions_rows(
        [runs[-1]],
        {
            "000002.SZ": {
                "industry": "测试行业",
                "industry_source": "strict_parquet.stock_basic.industry",
                "industry_as_of": None,
                "industry_generation_sha256": "a" * 64,
            }
        },
    )

    assert rows[0]["industry"] == "测试行业"
    assert rows[0]["industry_source"] == "strict_parquet.stock_basic.industry"
    assert rows[0]["industry_generation_sha256"] == "a" * 64
    assert not warnings


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
        "symbol,name,current_value,market_weight,shares,avg_cost,current_price\n"
        "000001.SZ,Manual,110,1,10,10,11\n",
        encoding="utf-8",
    )
    manifest_path = _write_valid_manual_manifest(
        run_dir,
        recorded_at="2026-07-08T09:33:00+08:00",
        total_value_after=110.0,
    )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload.pop("ledger_after_manual_switch_csv_sha256")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
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
    market = _write_active_cn_market(
        tmp_path / "market_data",
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260708",
                "close": 11.0,
                "adj_close": 11.0,
            }
        ],
    )

    summary = exporter.export(
        record_root,
        dashboard_root,
        benchmark_source="local",
        benchmark_file=benchmark_path,
        data_root=market["data_root"],
    )
    written_summary = json.loads(
        (dashboard_root / "generated" / "export_summary.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (dashboard_root / "private" / "dashboard_snapshot.v3.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot_evidence = contract["trading_calendar"]["market_snapshot"]
    assert contract["sources"]["cn_market_snapshot"] == snapshot_evidence
    assert snapshot_evidence["latest_pointer_sha256"] == hashlib.sha256(
        market["latest"].read_bytes()
    ).hexdigest()
    assert snapshot_evidence["table_root_path_summary"].endswith(
        "/parquet/cn/_snapshots/dashboard-snapshot-v4/table/bars"
    )

    status = summary["effective_manual_ledger_status"]
    assert written_summary["effective_manual_ledger_status"] == status
    assert status == {
        "status": "valid",
        "record_id": "20260708_0933",
        "manifest_status": "filled_local_manual_paper_rebalance",
        "manifest_recorded_at": "2026-07-08T09:33:00+08:00",
        "manifest_order_key": ["20260708093300", "20260708_0933"],
        "ledger_path": "<external>/20260708_0933/ledger_after_manual_switch.csv",
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "ledger_sha_declared": False,
        "ledger_readback_verified": True,
        "manifest_path": "<external>/20260708_0933/manual_execution_manifest.json",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_readback_verified": True,
        "legacy_ledger_fallback_used": False,
    }
    assert "manual_ledger_sha_not_declared" in summary["blockers"]
    undeclared_warnings = [
        warning
        for warning in summary["warnings"]
        if "manual_ledger_sha_not_declared" in warning
    ]
    assert len(undeclared_warnings) == 1
    assert "count=1" in undeclared_warnings[0]

    for table, summary_key, fieldnames in (
        ("industries", "industry_rows", exporter.INDUSTRY_RECORD_FIELDNAMES),
        ("factors", "factor_rows", exporter.FACTOR_RECORD_FIELDNAMES),
    ):
        path = dashboard_root / "generated" / f"{table}_records.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            written_rows = list(csv.DictReader(handle))
        assert path.stat().st_mode & 0o777 == 0o600
        assert written_rows == [
            {
                field: "" if row.get(field) is None else str(row.get(field))
                for field in fieldnames
            }
            for row in contract[table]
        ]
        assert summary[summary_key] == len(contract[table])
        assert summary["generated_table_hashes"][table] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


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
        "symbol,name,current_value,market_weight,shares,avg_cost,current_price\n"
        "000001.SZ,Manual,1425877,1,1000,1000,1425.877\n",
        encoding="utf-8",
    )
    _write_valid_manual_manifest(
        manual_run,
        recorded_at="2026-05-26T09:34:29+08:00",
        total_value_after=1_425_877.0,
    )
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
    market = _write_active_cn_market(
        tmp_path / "market_data",
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "close": close,
                "adj_close": close,
            }
            for trade_date, close in [
                ("20260318", 10.0),
                ("20260526", 11.0),
            ]
        ],
    )

    summary = exporter.export(
        record_root,
        dashboard_root,
        benchmark_source="local",
        benchmark_file=benchmark_path,
        data_root=market["data_root"],
    )
    nav_rows = exporter.read_csv_rows(dashboard_root / "generated" / "nav_records.csv")
    positions_rows = exporter.read_csv_rows(dashboard_root / "generated" / "positions_records.csv")
    generated_js = (dashboard_root / "private" / "dashboard_snapshot.v3.js").read_text(encoding="utf-8")

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
        ("2026-07-07", "603078.SH", "sell", "1300"),
    ]
    assert rows[0]["price"] == "53.6600"
    assert rows[0]["trade_amount"] == "69758.00"


def test_build_trade_rows_partial_fill_uses_only_actual_fill_metrics(tmp_path):
    exporter = _load_exporter()
    run_dir = tmp_path / "20260707_1046"
    run_dir.mkdir()
    (run_dir / "manual_switch_and_take_profit_orders.csv").write_text(
        "\n".join(
            [
                (
                    "timestamp,status,action,symbol,name,shares,price,trade_value,"
                    "filled_quantity,fill_price,fill_value,reason"
                ),
                (
                    "2026-07-07 11:09:35 CST,partial_fill,buy,300285.SZ,国瓷材料,"
                    "700,88.28,61796,100,88.10,8810,partial execution"
                ),
                (
                    "2026-07-07 11:10:00 CST,partially_filled,sell,603078.SH,江化微,"
                    "1300,53.66,69758,,,,order-only values must not be exported"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run = exporter.RecordRun(
        "20260707_1046", "2026-07-07", run_dir, "", 100.0, 101.0, {}
    )
    warnings: list[str] = []
    completeness: dict[str, object] = {}

    rows = exporter.build_trade_rows(
        [run], {}, warnings=warnings, completeness=completeness
    )

    assert len(rows) == 1
    assert rows[0]["ticker"] == "300285.SZ"
    assert rows[0]["quantity"] == "100"
    assert rows[0]["price"] == "88.1000"
    assert rows[0]["trade_amount"] == "8810.00"
    assert completeness["status"] == "partial"
    assert completeness["skipped_incomplete_rows"] == 1
    assert any("trade_record_incomplete" in warning for warning in warnings)


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


def test_dashboard_display_infos_carry_nonblocking_industry_warning():
    exporter = _load_exporter()
    theme_warning = "133 条持仓记录缺少strict industry，已使用 stock_basic industry 生成 '行业: <sector>' 回退标签。"
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
    market_rows = [
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
    market = _write_active_cn_market(tmp_path / "data", market_rows)
    market_binding = exporter.resolve_strict_cn_market_binding(market["data_root"])

    benchmark_export, warnings = exporter.load_local_benchmark_export(runs, benchmark_path)
    assert warnings == []
    assert benchmark_export is not None

    enhanced_export, industry_warnings = exporter.attach_industry_equal_weight_nav(
        runs,
        benchmark_export,
        market_binding=market_binding,
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
    snapshot_evidence = summary["industry_equal_weight"]["market_snapshot"]
    assert snapshot_evidence["snapshot_id"] == "dashboard-snapshot-v4"
    assert snapshot_evidence["latest_pointer_path_summary"].endswith(
        "/parquet/cn/_latest.json"
    )
    assert snapshot_evidence["table_root_path_summary"].endswith(
        "/parquet/cn/_snapshots/dashboard-snapshot-v4/table/bars"
    )
    assert snapshot_evidence["fallback_used"] is False
    assert any(row["field"] == "industry_ew_nav" for row in enhanced_export.raw_rows)


def test_dashboard_calendar_reads_pointer_bound_immutable_root_not_stale_legacy(tmp_path):
    exporter = _load_exporter()
    first = _write_active_cn_market(
        tmp_path / "data",
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260714",
                "close": 10.0,
                "adj_close": 10.0,
            }
        ],
        snapshot_id="dashboard-snapshot-v4-a",
    )
    assert json.loads(first["latest"].read_text(encoding="utf-8"))[
        "snapshot_id"
    ] == "dashboard-snapshot-v4-a"
    market = _write_active_cn_market(
        tmp_path / "data",
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260715",
                "close": 11.0,
                "adj_close": 11.0,
            }
        ],
        snapshot_id="dashboard-snapshot-v4-b",
    )
    binding = exporter.resolve_strict_cn_market_binding(market["data_root"])

    calendar, warnings = exporter.load_strict_parquet_trading_calendar(
        binding,
        "2026-07-15",
        "2026-07-15",
    )

    assert warnings == []
    assert calendar["expected_open_dates"] == ["2026-07-15"]
    assert calendar["market_snapshot"] == binding.audit_metadata()
    assert binding.snapshot_id == "dashboard-snapshot-v4-b"
    assert "1999-01-01" not in calendar["expected_open_dates"]


def test_dashboard_calendar_rejects_active_pointer_drift(tmp_path):
    exporter = _load_exporter()
    market = _write_active_cn_market(
        tmp_path / "data",
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260715",
                "close": 11.0,
                "adj_close": 11.0,
            }
        ],
    )
    binding = exporter.resolve_strict_cn_market_binding(market["data_root"])
    pointer = json.loads(market["latest"].read_text(encoding="utf-8"))
    pointer["published_after_binding"] = True
    market["latest"].write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(
        exporter.MarketDataUnavailableError,
        match="pointer changed after Dashboard source binding",
    ):
        exporter.load_strict_parquet_trading_calendar(
            binding,
            "2026-07-15",
            "2026-07-15",
        )


def test_dashboard_calendar_rejects_immutable_root_symlink_drift(tmp_path):
    exporter = _load_exporter()
    market = _write_active_cn_market(
        tmp_path / "data",
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260715",
                "close": 11.0,
                "adj_close": 11.0,
            }
        ],
    )
    binding = exporter.resolve_strict_cn_market_binding(market["data_root"])
    original_root = market["table_root"].with_name("bars-original")
    market["table_root"].rename(original_root)
    market["table_root"].symlink_to(market["legacy_root"], target_is_directory=True)

    with pytest.raises(
        exporter.MarketDataUnavailableError,
        match="symlink rejected",
    ):
        exporter.load_strict_parquet_trading_calendar(
            binding,
            "2026-07-15",
            "2026-07-15",
        )


def test_dashboard_binding_rejects_pointer_declared_fixed_bars_root(tmp_path):
    exporter = _load_exporter()
    market = _write_active_cn_market(
        tmp_path / "data",
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260715",
                "close": 11.0,
                "adj_close": 11.0,
            }
        ],
    )
    pointer = json.loads(market["latest"].read_text(encoding="utf-8"))
    pointer["table_root"] = str(market["legacy_root"])
    market["latest"].write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(
        exporter.MarketDataUnavailableError,
        match="v4 snapshot table_root invalid",
    ):
        exporter.resolve_strict_cn_market_binding(market["data_root"])
