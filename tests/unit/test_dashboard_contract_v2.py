from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_valid_manual_manifest(
    run_dir: Path,
    *,
    recorded_at: str,
    total_value_after: float,
) -> Path:
    ledger_path = run_dir / "ledger_after_manual_switch.csv"
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    manifest_path = run_dir / "manual_execution_manifest.json"
    status = "filled_local_manual_paper_rebalance"
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _exporter():
    return _load_module(
        "dashboard_contract_v2_exporter",
        ROOT / "scripts" / "export_cn_aggressive_dashboard_data.py",
    )


def _checker():
    return _load_module(
        "dashboard_contract_v2_checker",
        ROOT / "scripts" / "check_cn_dashboard_export.py",
    )


def test_percent_point_return_is_exported_once_as_decimal(tmp_path: Path) -> None:
    exporter = _exporter()
    run_dir = tmp_path / "20260712_0939"
    run_dir.mkdir()
    (run_dir / "ledger_after_manual_switch.csv").write_text(
        "symbol,name,current_value,market_weight,shares,current_price\n"
        "TEST.SZ,示例公司,426,1,10,42.6\n",
        encoding="utf-8",
    )
    (run_dir / "holdings_review.csv").write_text(
        "symbol,name,today_change_pct,recommended_action,quote_at\n"
        "TEST.SZ,示例公司,-0.46,hold,2026-07-10T15:00:00+08:00\n",
        encoding="utf-8",
    )
    run = exporter.RecordRun(
        "20260712_0939",
        "2026-07-12",
        run_dir,
        "2026-07-12 09:39:00",
        10_000.0,
        10_000.0,
        {},
    )

    rows, warnings = exporter.build_positions_rows([run], {})

    assert warnings
    assert len(rows) == 1
    assert rows[0]["daily_return"] == "-0.00460000"
    assert rows[0]["nav_weight"] == "0.04260000"
    assert rows[0]["equity_sleeve_weight"] == "1.00000000"
    assert rows[0]["weight"] == rows[0]["nav_weight"]
    assert rows[0]["contribution"] == "-0.00019596"
    assert rows[0]["contribution_effective_date"] == "2026-07-10"
    assert rows[0]["contribution_date_source"] == "holdings_review.quote_at"


def test_dashboard_contract_v2_keeps_unknown_fees_null_and_paths_sanitized(
    tmp_path: Path,
) -> None:
    exporter = _exporter()
    source = tmp_path / "records" / "20260712_0939"
    source.mkdir(parents=True)
    ledger = source / "ledger_after_manual_switch.csv"
    manifest = source / "manual_execution_manifest.json"
    ledger.write_text("symbol,name\nTEST.SZ,示例公司\n", encoding="utf-8")
    manifest.write_text('{"status":"ok"}\n', encoding="utf-8")
    nav_rows = [
        {
            "date": "2026-07-12",
            "portfolio_nav": "1.00000000",
            "portfolio_return": "",
            "total_value_after": "10000.00",
            "gross_exposure": "0.04260000",
            "net_exposure": "0.04260000",
            "cash_weight": "0.95740000",
            "benchmark_main_nav": "1.23450000",
        }
    ]
    position_rows = [
        {
            "date": "2026-07-12",
            "ticker": "TEST.SZ",
            "name": "示例公司",
            "weight": "0.04260000",
            "nav_weight": "0.04260000",
            "equity_sleeve_weight": "1.00000000",
            "daily_return": "-0.00460000",
            "contribution": "-0.00019596",
        }
    ]
    trade_rows = [
        {
            "trade_date": "2026-07-12",
            "ticker": "TEST.SZ",
            "side": "buy",
            "price": "42.6000",
            "quantity": "10",
            "trade_amount": "426.00",
            "fee": "",
            "fee_source": "unknown",
        }
    ]
    contract = exporter.build_dashboard_contract_v2(
        run_id="dashboard_20260712_0939",
        generated_at="2026-07-12T09:40:00+08:00",
        record_root=tmp_path / "records",
        latest_run=exporter.RecordRun(
            "20260712_0939",
            "2026-07-12",
            source,
            "2026-07-12 09:39:00",
            10_000.0,
            10_000.0,
            {},
        ),
        nav_rows=nav_rows,
        position_rows=position_rows,
        trade_rows=trade_rows,
        benchmark_summary={"value_date_by_date": {}},
        ledger_path=ledger,
        manifest_path=manifest,
        warnings=[],
    )

    assert contract["schema_version"] == "dashboard_contract.v2"
    assert contract["trades"][0]["fee"] is None
    assert contract["as_of_matrix"]["strategy_record_date"] == "2026-07-12"
    assert contract["positions"][0]["nav_weight"] == 0.0426
    assert contract["positions"][0]["equity_sleeve_weight"] == 1.0
    assert contract["nav"][0]["benchmark_main_nav"] == 1.2345
    encoded = json.dumps(contract, ensure_ascii=False)
    assert str(tmp_path) not in encoded
    assert contract["sources"]["ledger"]["sha256"]
    assert contract["sources"]["manual_manifest"]["sha256"]


def test_semantic_validator_rejects_contribution_mismatch_and_non_decimal_return() -> None:
    checker = _checker()
    contract = {
        "schema_version": "dashboard_contract.v2",
        "run_id": "run",
        "generated_at": "2026-07-12T09:40:00+08:00",
        "status": "partial",
        "blockers": [],
        "as_of_matrix": {},
        "sources": {},
        "nav": [{"date": "2026-07-12", "portfolio_nav": 1.0}],
        "positions": [
            {
                "date": "2026-07-12",
                "ticker": "TEST.SZ",
                "nav_weight": 0.0426,
                "equity_sleeve_weight": 1.0,
                "daily_return": -0.46,
                "contribution": 0.0,
            }
        ],
        "trades": [],
        "themes": [],
        "factors": [],
        "reconciliation": {"daily": []},
    }

    errors, _warnings = checker.validate_dashboard_contract_v2(contract)

    assert any("decimal" in error for error in errors)
    assert any("contribution" in error for error in errors)


def test_checker_rejects_reconciled_attribution_with_extremely_low_coverage() -> None:
    checker = _checker()
    policy = {
        "returns_unit": "decimal",
        "contribution_formula": "nav_weight * daily_return",
        "annualization_min_open_day_coverage": 0.95,
        "annualization_min_valid_daily_returns": 60,
        "annualization_insufficient_status": "insufficient_daily_history",
        "rolling_window_min_open_day_coverage": 0.95,
        "trading_calendar_required": True,
        "excess_curve": "relative_wealth_ratio",
        "monthly_return": "previous_month_end_anchor",
        "unknown_numeric": None,
    }
    contract = {
        "schema_version": "dashboard_contract.v2",
        "schema_sha256": None,
        "protocol_hash": checker._canonical_json_sha256(
            {
                "schema_version": "dashboard_contract.v2",
                "metric_policy": policy,
                "required_tables": ["nav", "positions", "trades", "themes", "factors"],
            }
        ),
        "run_id": "sample",
        "generated_at": "2026-07-12T09:40:00+08:00",
        "status": "sample",
        "blockers": [],
        "as_of_matrix": {},
        "sources": {},
        "trading_calendar": {"status": "missing", "expected_open_dates": []},
        "nav_return_provenance": {"gross_or_net": "unknown", "secondary_fee_adjustment_allowed": False},
        "nav": [
            {"date": "2026-07-10", "portfolio_nav": 1.0, "portfolio_return": 0.01},
            {"date": "2026-07-11", "portfolio_nav": 1.01, "portfolio_return": 0.01},
        ],
        "positions": [
            {
                "date": "2026-07-10",
                "ticker": "TEST.SZ",
                "nav_weight": 0.1,
                "equity_sleeve_weight": 1.0,
                "daily_return": 0.1,
                "contribution": 0.01,
            }
        ],
        "trades": [],
        "themes": [],
        "factors": [],
        "reconciliation": {
            "tolerance": 0.0001,
            "valid_nav_return_days": 2,
            "covered_days": 1,
            "coverage_ratio": 0.5,
            "daily": [
                {
                    "date": "2026-07-10",
                    "portfolio_return": 0.01,
                    "position_contribution": 0.01,
                    "explicit_cash_fee_residual": None,
                    "unexplained_residual": 0.0,
                    "within_1bp": True,
                }
            ],
            "status": "reconciled",
        },
        "metric_policy": policy,
    }

    errors, _warnings = checker.validate_dashboard_contract_v2(contract)

    assert any("attribution coverage" in error for error in errors)


def test_as_of_matrix_does_not_guess_from_run_id_or_latest_nav(tmp_path: Path) -> None:
    exporter = _exporter()
    run_dir = tmp_path / "20260712_0939"
    run_dir.mkdir()
    ledger = run_dir / "ledger_after_manual_switch.csv"
    manifest = run_dir / "manual_execution_manifest.json"
    ledger.write_text("symbol,name\nTEST.SZ,示例公司\n", encoding="utf-8")
    manifest.write_text('{"status":"ok"}\n', encoding="utf-8")
    run = exporter.RecordRun(
        "20260712_0939", "2026-07-12", run_dir, "", 100.0, 100.0, {}
    )

    contract = exporter.build_dashboard_contract_v2(
        run_id="dashboard_20260712_0939",
        generated_at="2026-07-12T09:40:00+08:00",
        record_root=tmp_path,
        latest_run=run,
        nav_rows=[{"date": "2026-07-12", "portfolio_nav": "1", "portfolio_return": ""}],
        position_rows=[
            {
                "date": "2026-07-12",
                "ticker": "TEST.SZ",
                "nav_weight": "0.1",
                "equity_sleeve_weight": "1",
                "daily_return": "",
                "contribution": "",
            }
        ],
        trade_rows=[],
        benchmark_summary={},
        ledger_path=ledger,
        manifest_path=manifest,
        warnings=[],
    )

    assert contract["as_of_matrix"]["strategy_record_date"] is None
    assert contract["as_of_matrix"]["analysis_trading_date"] is None
    assert contract["as_of_matrix"]["quote_at"] is None
    assert contract["as_of_matrix"]["theme_date"] is None
    assert "as_of_strategy_record_missing" in contract["blockers"]
    assert exporter.explicit_iso_date("latest_20260712") is None
    assert exporter.explicit_iso_date("2026-02-30") is None
    assert exporter.explicit_iso_timestamp("20260712_bad") is None


def test_attribution_aggregate_cash_fee_residual_takes_precedence_over_components() -> None:
    exporter = _exporter()

    reconciliation = exporter.build_attribution_reconciliation(
        [
            {
                "date": "2026-07-10",
                "portfolio_return": 0.01,
                "cash_return_contribution": 0.001,
                "fee_return_contribution": -0.0002,
                "explicit_cash_fee_residual": 0.002,
            }
        ],
        [
            {
                "date": "2026-07-10",
                "ticker": "TEST.SZ",
                "nav_weight": 0.08,
                "daily_return": 0.1,
                "contribution": 0.008,
                "equity_sleeve_weight": 1.0,
                "contribution_effective_date": "2026-07-10",
                "contribution_date_source": "holdings_review.quote_at",
            }
        ],
        trading_calendar={
            "status": "available",
            "expected_open_dates": ["2026-07-10"],
        },
    )

    assert reconciliation["daily"][0]["explicit_cash_fee_residual"] == 0.002
    assert reconciliation["daily"][0]["unexplained_residual"] == 0.0
    assert reconciliation["daily"][0]["position_snapshot_complete"] is True
    assert reconciliation["coverage_ratio"] == 1.0
    assert reconciliation["status"] == "reconciled"


def test_cash_fee_residual_cannot_create_attribution_coverage() -> None:
    exporter = _exporter()

    reconciliation = exporter.build_attribution_reconciliation(
        [
            {
                "date": "2026-07-10",
                "portfolio_return": 0.01,
                "explicit_cash_fee_residual": 0.01,
            }
        ],
        [
            {
                "date": "2026-07-10",
                "ticker": "TEST.SZ",
                "contribution": 0.0,
            }
        ],
        trading_calendar={
            "status": "available",
            "expected_open_dates": ["2026-07-10"],
        },
    )

    assert reconciliation["daily"][0]["covered"] is False
    assert reconciliation["daily"][0]["unexplained_residual"] is None
    assert reconciliation["coverage_ratio"] == 0.0
    assert reconciliation["status"] == "partial"


def test_attribution_excludes_non_open_nav_and_position_effective_dates() -> None:
    exporter = _exporter()

    reconciliation = exporter.build_attribution_reconciliation(
        [
            {"date": "2026-07-10", "portfolio_return": 0.01},
            {"date": "2026-07-11", "portfolio_return": 0.50},
        ],
        [
            {
                "date": "2026-07-12",
                "ticker": "OPEN.SZ",
                "nav_weight": 0.1,
                "daily_return": 0.1,
                "contribution": 0.01,
                "equity_sleeve_weight": 1.0,
                "contribution_effective_date": "2026-07-10",
                "contribution_date_source": "holdings_review.quote_at",
            },
            {
                "date": "2026-07-12",
                "ticker": "WEEKEND.SZ",
                "nav_weight": 0.1,
                "daily_return": 5.0,
                "contribution": 0.5,
                "equity_sleeve_weight": 1.0,
                "contribution_effective_date": "2026-07-11",
                "contribution_date_source": "holdings_review.quote_at",
            },
        ],
        trading_calendar={
            "status": "available",
            "expected_open_dates": ["2026-07-10"],
        },
    )

    assert reconciliation["valid_nav_return_days"] == 1
    assert reconciliation["covered_days"] == 1
    assert reconciliation["daily"][0]["date"] == "2026-07-10"
    assert reconciliation["diagnostics"]["excluded_nav_return_dates"] == [
        "2026-07-11"
    ]
    assert reconciliation["diagnostics"][
        "excluded_position_effective_dates"
    ] == ["2026-07-11"]
    assert "attribution_non_open_dates_excluded" in reconciliation["blockers"]


def test_checker_rejects_string_benchmark_nav_and_schema_constrains_dynamic_nav() -> None:
    checker = _checker()
    schema = json.loads(
        (ROOT / "portfolio_dashboard/schema/dashboard_contract.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["$defs"]["nav"]["patternProperties"]["^.*_nav$"] == {
        "$ref": "#/$defs/nullableNumber"
    }
    contract = {
        "schema_version": "dashboard_contract.v2",
        "schema_sha256": None,
        "protocol_hash": "0" * 64,
        "run_id": "sample",
        "generated_at": "2026-07-12T09:40:00+08:00",
        "status": "sample",
        "blockers": [],
        "as_of_matrix": {},
        "sources": {},
        "trading_calendar": {"status": "missing", "expected_open_dates": []},
        "nav_return_provenance": {},
        "nav": [
            {
                "date": "2026-07-10",
                "portfolio_nav": 1.0,
                "benchmark_main_nav": "1.0",
            }
        ],
        "positions": [],
        "trades": [],
        "themes": [],
        "theme_protocol": {},
        "factors": [],
        "factor_protocol": {},
        "reconciliation": {"daily": []},
        "metric_policy": {},
    }

    errors, _warnings = checker.validate_dashboard_contract_v2(contract)

    assert "nav[0].benchmark_main_nav must be numeric or null." in errors


def test_future_benchmark_value_date_is_an_explicit_contract_blocker(
    tmp_path: Path,
) -> None:
    exporter = _exporter()
    checker = _checker()
    run_dir = tmp_path / "20260710_0939"
    run_dir.mkdir()
    (run_dir / "market_snapshot.json").write_text(
        '{"analysis_trade_date":"2026-07-10"}\n', encoding="utf-8"
    )
    ledger = run_dir / "ledger_after_manual_switch.csv"
    manifest = run_dir / "manual_execution_manifest.json"
    ledger.write_text("symbol,name\nTEST.SZ,示例公司\n", encoding="utf-8")
    manifest.write_text('{"status":"ok"}\n', encoding="utf-8")
    contract = exporter.build_dashboard_contract_v2(
        run_id="dashboard_20260710_0939",
        generated_at="2026-07-10T09:40:00+08:00",
        record_root=tmp_path,
        latest_run=exporter.RecordRun(
            "20260710_0939",
            "2026-07-10",
            run_dir,
            "2026-07-10T09:39:00+08:00",
            10_000.0,
            10_000.0,
            {},
        ),
        nav_rows=[{"date": "2026-07-10", "portfolio_nav": "1.0"}],
        position_rows=[
            {
                "date": "2026-07-10",
                "ticker": "TEST.SZ",
                "nav_weight": "0.1",
                "equity_sleeve_weight": "1.0",
                "daily_return": "0.0",
                "contribution": "0.0",
            }
        ],
        trade_rows=[],
        benchmark_summary={
            "production_grade": True,
            "value_date_by_date": {
                "2026-07-10": {"benchmark_main_nav": "2026-07-11"}
            },
        },
        ledger_path=ledger,
        manifest_path=manifest,
        warnings=[],
    )

    assert "as_of_benchmark_after_analysis_date" in contract["blockers"]
    errors, _warnings = checker.validate_dashboard_contract_v2(contract)
    assert not any("future benchmark value date" in error for error in errors)
    contract["blockers"].remove("as_of_benchmark_after_analysis_date")
    errors, _warnings = checker.validate_dashboard_contract_v2(contract)
    assert any("future benchmark value date" in error for error in errors)


def test_applied_factor_artifact_is_blocked_without_canonical_producer(
    tmp_path: Path,
) -> None:
    exporter = _exporter()
    from quant_investor.factors.governance_protocol_v2 import (
        PROTOCOL_HASH,
        PROTOCOL_SCHEMA_VERSION,
    )

    run_dir = tmp_path / "20260710_0939"
    run_dir.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_text('{"factors":[]}\n', encoding="utf-8")
    registry_sha = exporter.sha256_file(registry)
    inverse_wal = tmp_path / "inverse.wal.json"
    inverse_wal.write_text("{}\n", encoding="utf-8")
    inverse_wal.chmod(0o600)
    artifact = tmp_path / "factor_protocol_v2.json"
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_version": "v2",
        "protocol_hash": PROTOCOL_HASH,
        "status": "applied",
        "evidence_hash": "a" * 64,
        "transition_hash": "b" * 64,
        "mutation_plan_hash": "c" * 64,
        "transition_id": "transition-1",
        "before_registry_sha256": "d" * 64,
        "after_registry_sha256": registry_sha,
        "registry_mutation_manifest": {
            "applied": True,
            "after_registry_sha256": registry_sha,
        },
        "inverse_wal_path": str(inverse_wal),
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    protocol, _path = exporter._load_factor_protocol_v2_readback(
        exporter.RecordRun(
            "20260710_0939", "2026-07-10", run_dir, "", 1.0, 1.0, {}
        ),
        registry_sha=registry_sha,
        protocol_file=artifact,
        expected_sha256=exporter.sha256_file(artifact) or "",
    )

    assert protocol["canonical_producer_available"] is False
    assert protocol["status"] == "blocked"
    assert protocol["transition_applied"] is False
    assert "forward_factor_apply_not_authorized_pr4" in protocol[
        "blockers"
    ]


def test_theme_dashboard_preserves_only_available_trajectory_supply_and_review() -> None:
    exporter = _exporter()

    row = exporter._normalize_theme_state_row(
        "tech::ai",
        {
            "theme_name": "AI",
            "attention_5d": 0.8,
            "attention_20d": 0.7,
            "attention_60d": 0.6,
            "attention_120d": 0.5,
            "attention_history_coverage": 1.0,
            "supply_chain_roles": ["compute", "optical_interconnect"],
            "thesis_review": {
                "status": "due",
                "review_by": "2026-07-31",
            },
        },
    )
    missing = exporter._normalize_theme_state_row("tech::space", {})

    assert row["attention_trajectory_120d"] == {
        "5d": 0.8,
        "20d": 0.7,
        "60d": 0.6,
        "120d": 0.5,
        "history_coverage": 1.0,
    }
    assert row["supply_chain_roles"] == ["compute", "optical_interconnect"]
    assert row["thesis_review"] == {
        "status": "due",
        "review_by": "2026-07-31",
    }
    assert missing["supply_chain_roles"] is None
    assert missing["thesis_review"] is None


def test_formal_trading_calendar_comes_only_from_strict_parquet_dates(tmp_path: Path) -> None:
    exporter = _exporter()
    import pandas as pd

    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "TEST.SZ", "trade_date": "20260708", "close": 10.0},
            {"ts_code": "TEST.SZ", "trade_date": "20260710", "close": 10.1},
        ]
    ).to_parquet(bars_root / "part.parquet", index=False)

    calendar, warnings = exporter.load_strict_parquet_trading_calendar(
        bars_root, "2026-07-08", "2026-07-12"
    )

    assert warnings == []
    assert calendar["status"] == "available"
    assert calendar["expected_open_dates"] == ["2026-07-08", "2026-07-10"]
    assert "2026-07-09" not in calendar["expected_open_dates"]
    assert calendar["source_system"] == "strict_parquet.cn_bars.trade_date"


def test_schema_and_public_sample_do_not_contain_private_paths() -> None:
    schema_path = ROOT / "portfolio_dashboard" / "schema" / "dashboard_contract.v2.schema.json"
    sample_path = ROOT / "portfolio_dashboard" / "sample" / "dashboard_snapshot.v2.json"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))

    assert schema["$id"].endswith("dashboard_contract.v2.schema.json")
    assert sample["schema_version"] == "dashboard_contract.v2"
    assert sample["status"] == "sample"
    text = sample_path.read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert "ledger_after_manual_switch" not in text


def test_exported_private_snapshot_v2_round_trips_through_semantic_checker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exporter = _exporter()
    checker = _checker()
    record_root = tmp_path / "records"
    run_dir = record_root / "20260712_0939"
    run_dir.mkdir(parents=True)
    (run_dir / "pnl_summary.csv").write_text(
        "initial_capital,total_value_after,record_time\n"
        "10000,10000,2026-07-12 09:39:00\n",
        encoding="utf-8",
    )
    (run_dir / "ledger_after_manual_switch.csv").write_text(
        "symbol,name,current_value,market_weight,shares,avg_cost,current_price\n"
        "TEST.SZ,示例公司,426,1,10,40,42.6\n",
        encoding="utf-8",
    )
    _write_valid_manual_manifest(
        run_dir,
        recorded_at="2026-07-12T09:39:00+08:00",
        total_value_after=10_000.0,
    )
    (run_dir / "holdings_review.csv").write_text(
        "symbol,name,today_change_pct,recommended_action\n"
        "TEST.SZ,示例公司,-0.46,hold\n",
        encoding="utf-8",
    )
    benchmark_file = tmp_path / "benchmark.csv"
    benchmark_file.write_text(
        "date,ts_code,close,source_system\n"
        "2026-07-12,000300.SH,1000,TEST_VENDOR\n"
        "2026-07-12,000905.SH,2000,TEST_VENDOR\n"
        "2026-07-12,000852.SH,3000,TEST_VENDOR\n"
        "2026-07-12,000688.SH,4000,TEST_VENDOR\n"
        "2026-07-12,399006.SZ,5000,TEST_VENDOR\n",
        encoding="utf-8",
    )
    dashboard_root = tmp_path / "dashboard"
    monkeypatch.setattr(exporter, "DEFAULT_STOCK_BASIC_ROOT", tmp_path / "missing")

    summary = exporter.export(
        record_root,
        dashboard_root,
        benchmark_source="local",
        benchmark_file=benchmark_file,
    )
    result = checker.check_dashboard_export(
        dashboard_root / "generated" / "export_summary.json",
        dashboard_root / "private" / "dashboard_snapshot.v2.js",
    )

    assert summary["schema_version"] == "dashboard_contract.v2"
    assert summary["status"] == "partial"
    assert result["ok"] is True
    assert result["dashboard_contract"]["positions"][0]["nav_weight"] == 0.0426
    assert not (dashboard_root / "js" / "generated_records.js").exists()
    snapshot_text = (dashboard_root / "private" / "dashboard_snapshot.v2.json").read_text(
        encoding="utf-8"
    )
    assert str(tmp_path) not in snapshot_text
    private_json = dashboard_root / "private" / "dashboard_snapshot.v2.json"
    private_js = dashboard_root / "private" / "dashboard_snapshot.v2.js"
    assert private_json.stat().st_mode & 0o777 == 0o600
    assert private_js.stat().st_mode & 0o777 == 0o600
    parsed_js = checker.parse_generated_records(private_js)
    assert parsed_js.contract == json.loads(snapshot_text)
    for generated_artifact in (dashboard_root / "generated").iterdir():
        assert generated_artifact.stat().st_mode & 0o777 == 0o600

    mismatched_payload = json.loads(snapshot_text)
    mismatched_payload["run_id"] = "tampered"
    private_json.write_text(json.dumps(mismatched_payload), encoding="utf-8")
    private_json.chmod(0o600)
    mismatch = checker.check_dashboard_export(
        dashboard_root / "generated" / "export_summary.json",
        private_js,
    )
    assert mismatch["ok"] is False
    assert "private dashboard JSON/JS contract payload mismatch." in mismatch[
        "errors"
    ]
    private_json.write_text(snapshot_text, encoding="utf-8")
    private_json.chmod(0o600)

    private_json.chmod(0o644)
    private_js.chmod(0o644)
    insecure = checker.check_dashboard_export(
        dashboard_root / "generated" / "export_summary.json",
        private_js,
    )
    assert insecure["ok"] is False
    permission_errors = [
        error for error in insecure["errors"] if "permissions must be 0600" in error
    ]
    assert any("dashboard_snapshot.v2.json" in error for error in permission_errors)
    assert any("dashboard_snapshot.v2.js" in error for error in permission_errors)


def test_private_snapshot_atomic_replacement_restores_owner_only_mode(tmp_path: Path) -> None:
    exporter = _exporter()
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    private_json = private_dir / "dashboard_snapshot.v2.json"
    private_js = private_dir / "dashboard_snapshot.v2.js"
    for path in (private_json, private_js):
        path.write_text("stale", encoding="utf-8")
        path.chmod(0o644)

    exporter.write_dashboard_snapshot_v2(
        private_json,
        private_js,
        {"schema_version": "dashboard_contract.v2"},
        generated_records_js="window.DashboardGeneratedRecords = {};\n",
    )

    assert private_json.stat().st_mode & 0o777 == 0o600
    assert private_js.stat().st_mode & 0o777 == 0o600
    assert not list(private_dir.glob(".*.tmp"))


def test_static_dashboard_privacy_xss_and_workspace_contracts() -> None:
    app = (ROOT / "portfolio_dashboard" / "app.js").read_text(encoding="utf-8")
    charts = (ROOT / "portfolio_dashboard" / "js" / "charts.js").read_text(
        encoding="utf-8"
    )
    loader = (
        ROOT / "portfolio_dashboard" / "js" / "generated_records.js"
    ).read_text(encoding="utf-8")
    index = (ROOT / "portfolio_dashboard" / "index.html").read_text(encoding="utf-8")

    assert "localStorage" not in app
    assert "csvBundle" not in app[app.find("function syncHashState") : app.find("function applyWorkspaceVisibility")]
    assert "escapeHtml(data.label)" in charts
    assert "/Users/" not in loader
    assert index.count("data-workspace-tab=") == 6
    assert 'src="private/dashboard_snapshot.v2.js"' in index
    assert 'id="attributionSection" data-workspace="audit"' in index
