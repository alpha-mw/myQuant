from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cn_dashboard_common import (  # noqa: E402
    DashboardInputError,
    build_bundle,
    canonical_json_bytes,
    validate_bundle_shape,
    verify_source_refs,
)
from export_cn_aggressive_dashboard_data import publish_bundle  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_record(
    project_root: Path,
    record: str,
    data_date: str,
    total_value: float,
    *,
    source_record: str | None,
    funding: dict | None = None,
    capital_base: float = 10000,
) -> Path:
    record_root = (
        project_root
        / "results"
        / "strategy_records"
        / "CN"
        / "aggressive_tech_manufacturing"
    )
    record_dir = record_root / record
    record_dir.mkdir(parents=True)
    market_value = 1100.0
    cash = total_value - market_value
    ledger_path = record_dir / "ledger_after_manual_switch.csv"
    _write_csv(
        ledger_path,
        [
            "symbol",
            "name",
            "shares",
            "avg_cost",
            "cost_basis",
            "current_price",
            "current_value",
            "unrealized_pnl",
            "equity_sleeve_weight",
            "nav_weight",
            "thesis_status",
        ],
        [
            {
                "symbol": "000001.SZ",
                "name": "合成样例A",
                "shares": 100,
                "avg_cost": 10,
                "cost_basis": 1000,
                "current_price": 11,
                "current_value": market_value,
                "unrealized_pnl": 100,
                "equity_sleeve_weight": 1,
                "nav_weight": market_value / total_value,
                "thesis_status": "SYNTHETIC",
            }
        ],
    )
    ledger_sha = _sha(ledger_path)
    pnl_path = record_dir / "pnl_summary.csv"
    pnl_fields = [
        "initial_capital",
        "cash_after",
        "market_value_after",
        "total_value_after",
        "portfolio_pnl_after",
        "realized_pnl_from_rebalance",
    ]
    _write_csv(
        pnl_path,
        pnl_fields,
        [
            {
                "initial_capital": capital_base,
                "cash_after": cash,
                "market_value_after": market_value,
                "total_value_after": total_value,
                "portfolio_pnl_after": total_value - capital_base,
                "realized_pnl_from_rebalance": 0,
            }
        ],
    )
    manual = {
        "schema_version": "cn_aggressive_manual_execution.v3",
        "status": "no_action_carry_forward",
        "execution_status": "no_action_carry_forward",
        "record_timestamp": record,
        "no_broker_api_called": True,
        "effective_manual_ledger_path": "ledger_after_manual_switch.csv",
        "next_ledger_path": "ledger_after_manual_switch.csv",
        "next_ledger_sha256": ledger_sha,
        "ledger_after_manual_switch_csv_sha256": ledger_sha,
        "ledger_provenance": {
            "contained_in_run_directory": True,
            "regular_non_symlink_file": True,
            "stable_double_read": True,
            "declared_sha256": ledger_sha,
        },
        "effective_manual_holding_count": 1,
        "cash_after": cash,
        "market_value_after": market_value,
        "total_value_after": total_value,
        "portfolio_pnl_after": total_value - capital_base,
        "realized_pnl_from_rebalance": 0,
        "financial_state_sha256": "f" * 64,
    }
    if funding is not None:
        supplement_path = record_dir / "funding.json"
        supplement_path.write_text(
            json.dumps(funding, sort_keys=True), encoding="utf-8"
        )
        manual["manual_funding_supplement"] = funding
        manual["manual_funding_supplement_path"] = "funding.json"
    manual_path = record_dir / "manual_execution_manifest.json"
    manual_path.write_text(
        json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": record,
        "recorded_at": (
            f"{record[:4]}-{record[4:6]}-{record[6:8]} 12:00:00 CST"
        ),
        "capital_cny": capital_base,
        "source_record": source_record,
        "files": {
            "manual_execution_manifest": "manual_execution_manifest.json",
            "pnl_summary": "pnl_summary.csv",
        },
        "manual_execution": manual,
        "data_snapshot": {"analysis_trade_date": data_date.replace("-", "")},
    }
    (record_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record_dir


def _write_benchmark(project_root: Path, dates: list[str]) -> Path:
    path = (
        project_root
        / "portfolio_dashboard"
        / "inputs"
        / "cn_index_benchmark.csv"
    )
    _write_csv(
        path,
        [
            "date",
            "ts_code",
            "close",
            "source_system",
            "value_date",
            "coverage",
        ],
        [
            {
                "date": value,
                "ts_code": "000300.SH",
                "close": 100 + index,
                "source_system": "tushare.index_daily",
                "value_date": value,
                "coverage": "exact_close",
            }
            for index, value in enumerate(dates)
        ],
    )
    return path


def _write_legacy_baseline(
    project_root: Path, record: str, *, capital: float
) -> None:
    record_dir = (
        project_root
        / "results"
        / "strategy_records"
        / "CN"
        / "aggressive_tech_manufacturing"
        / record
    )
    record_dir.mkdir(parents=True)
    ledger_path = record_dir / "ledger.csv"
    _write_csv(
        ledger_path,
        ["symbol", "name", "shares", "current_value"],
        [
            {
                "symbol": "000001.SZ",
                "name": "合成样例A",
                "shares": 100,
                "current_value": 1100,
            }
        ],
    )
    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": record,
        "recorded_at": "2098-12-30 12:00:00 CST",
        "source_record": None,
        "capital_cny": capital,
        "files": {"ledger": "ledger.csv"},
        "data_snapshot": {
            "intraday_quote_snapshot": "2098-12-30 12:00 CST"
        },
    }
    (record_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    record_root = (
        tmp_path
        / "results"
        / "strategy_records"
        / "CN"
        / "aggressive_tech_manufacturing"
    )
    _write_legacy_baseline(tmp_path, "20981230_1200", capital=10000)
    _write_record(
        tmp_path,
        "20990101_1200",
        "2099-01-01",
        10000,
        source_record="20981230_1200",
    )
    funding = {
        "amount": 10000,
        "cash_before": 9000,
        "cash_after": 19000,
        "total_value_before": 11000,
        "total_value_after": 21000,
        "record_id": "20990102_1200",
        "schema_version": "cn_aggressive_manual_funding_supplement.v1",
        "status": "local_manual_funding_recorded_no_broker_api",
    }
    _write_record(
        tmp_path,
        "20990102_1200",
        "2099-01-02",
        21000,
        source_record="20990101_1200",
        funding=funding,
        capital_base=20000,
    )
    _write_record(
        tmp_path,
        "20990103_1200",
        "2099-01-03",
        23100,
        source_record="20990102_1200",
        capital_base=20000,
    )
    benchmark = _write_benchmark(
        tmp_path,
        ["2098-12-30", "2099-01-01", "2099-01-02", "2099-01-03"],
    )
    return tmp_path, record_root, benchmark


def test_build_bundle_is_funding_aware_and_read_only(tmp_path: Path) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )
    assert bundle["latest_valid_record"] == "20990103_1200"
    assert bundle["previous_valid_record"] == "20990102_1200"
    assert bundle["history"]["archive_start_record"] == "20981230_1200"
    assert bundle["history"]["archive_start_date"] == "2098-12-30"
    assert bundle["history"]["first_pnl_record"] == "20990101_1200"
    assert bundle["history"]["legacy_exact_byte_record_count"] == 1
    assert bundle["history"]["net_external_flow"] == 10000
    assert bundle["history"]["funding_events"][0]["record"] == (
        "20990102_1200"
    )
    assert bundle["portfolio"]["performance_start_date"] == "2098-12-30"
    assert bundle["portfolio"]["performance_points"][0][
        "evidence_status"
    ] == "ARCHIVE_INCEPTION_EXACT_BYTES_NO_DECLARED_SHA"
    assert bundle["portfolio"]["cumulative_twr"] == pytest.approx(0.21)
    assert (
        bundle["portfolio"]["return_method"]
        == "funding_aware_time_weighted_unitization"
    )
    assert bundle["benchmarks"][0]["id"] == "CSI300"
    assert bundle["benchmarks"][0]["missing_dates"] == []
    assert all(value is False for value in bundle["authority_flags"].values())
    assert bundle["i1_research"] is None
    assert validate_bundle_shape(bundle) == []
    assert verify_source_refs(bundle, project_root) == []
    without_hash = dict(bundle)
    content_hash = without_hash.pop("content_sha256")
    assert (
        hashlib.sha256(canonical_json_bytes(without_hash)).hexdigest()
        == content_hash
    )


def test_source_sha_change_blocks_publish_and_preserves_existing_bundle(
    tmp_path: Path,
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )
    output_dir = project_root / "portfolio_dashboard" / "private" / "generated"
    output_dir.mkdir(parents=True)
    json_output = output_dir / "cn_aggressive_dashboard.v1.json"
    js_output = output_dir / "cn_aggressive_dashboard.v1.js"
    json_output.write_bytes(b"old-json-bytes\n")
    js_output.write_bytes(b"old-js-bytes\n")
    benchmark.write_text(
        benchmark.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8"
    )
    with pytest.raises(DashboardInputError, match="source_ref_sha_mismatch"):
        publish_bundle(bundle, json_output, js_output, project_root)
    assert json_output.read_bytes() == b"old-json-bytes\n"
    assert js_output.read_bytes() == b"old-js-bytes\n"


def test_capital_base_change_without_exact_funding_blocks_history(
    tmp_path: Path,
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    record_dir = record_root / "20990102_1200"
    manual_path = record_dir / "manual_execution_manifest.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    manual.pop("manual_funding_supplement")
    manual.pop("manual_funding_supplement_path")
    manual_path.write_text(
        json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = record_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manual_execution"] = manual
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pytest.raises(
        DashboardInputError,
        match="historical_capital_change_without_funding:20990102_1200",
    ):
        build_bundle(
            project_root=project_root,
            record_root=record_root,
            benchmark_path=benchmark,
            generated_at="2099-01-03T12:00:00+08:00",
            today=date(2099, 1, 3),
        )


def test_missing_primary_benchmark_date_blocks_export(tmp_path: Path) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    rows = list(
        csv.DictReader(benchmark.read_text(encoding="utf-8-sig").splitlines())
    )
    _write_csv(
        benchmark,
        [
            "date",
            "ts_code",
            "close",
            "source_system",
            "value_date",
            "coverage",
        ],
        [row for row in rows if row["date"] != "2099-01-02"],
    )
    with pytest.raises(
        DashboardInputError, match="csi300_benchmark_missing_dates"
    ):
        build_bundle(
            project_root=project_root,
            record_root=record_root,
            benchmark_path=benchmark,
            generated_at="2099-01-03T12:00:00+08:00",
            today=date(2099, 1, 3),
        )


def test_effective_ledger_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    latest_ledger = (
        record_root / "20990103_1200" / "ledger_after_manual_switch.csv"
    )
    latest_ledger.write_bytes(latest_ledger.read_bytes() + b"\n")
    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )
    assert bundle["latest_valid_record"] == "20990102_1200"
    assert (
        bundle["rejected_record_reason_counts"][
            "effective_ledger_sha_mismatch"
        ]
        == 1
    )


def test_hash_bound_effective_parquet_ledger_is_supported(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    project_root, record_root, benchmark = _fixture(tmp_path)
    latest_dir = record_root / "20990103_1200"
    parquet_path = latest_dir / "ledger_after_manual_switch.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "symbol": "000001.SZ",
                    "name": "合成样例A",
                    "shares": 100,
                    "avg_cost": 10,
                    "cost_basis": 1000,
                    "current_price": 11,
                    "current_value": 1100,
                    "unrealized_pnl": 100,
                    "equity_sleeve_weight": 1,
                    "nav_weight": 1100 / 23100,
                    "thesis_status": "SYNTHETIC",
                }
            ]
        ),
        parquet_path,
    )
    parquet_sha = _sha(parquet_path)
    manual_path = latest_dir / "manual_execution_manifest.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    manual["effective_manual_ledger_path"] = (
        "ledger_after_manual_switch.parquet"
    )
    manual["next_ledger_path"] = "ledger_after_manual_switch.parquet"
    manual["next_ledger_sha256"] = parquet_sha
    manual["ledger_after_manual_switch_parquet"] = (
        "ledger_after_manual_switch.parquet"
    )
    manual["ledger_after_manual_switch_parquet_sha256"] = parquet_sha
    manual["ledger_provenance"]["declared_sha256"] = parquet_sha
    manual_path.write_text(
        json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = latest_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manual_execution"] = manual
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )
    assert bundle["latest_valid_record"] == "20990103_1200"
    assert bundle["current_evidence"]["ledger_path"].endswith(".parquet")
    assert bundle["current_evidence"]["ledger_sha256"] == parquet_sha
