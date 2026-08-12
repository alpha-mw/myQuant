from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from quant_investor.strategy_records.store import (
    canonical_json_bytes as store_canonical_json_bytes,
    content_sha256 as store_content_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cn_dashboard_common as dashboard_common  # noqa: E402
from cn_dashboard_common import (  # noqa: E402
    DashboardInputError,
    build_bundle,
    build_history_integrity_registry,
    canonical_json_bytes,
    scan_historical_performance_records,
    scan_valid_records,
    validate_record,
    validate_bundle_shape,
    verify_source_refs,
)
from export_cn_aggressive_dashboard_data import publish_bundle  # noqa: E402
from build_cn_dashboard_public_site import sanitize_bundle  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_store_document(path: Path, body: dict) -> dict:
    payload = dict(body)
    payload["content_sha256"] = store_content_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(store_canonical_json_bytes(payload))
    return payload


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
    funding_correction: dict | None = None,
    capital_base: float = 10000,
    official_valuation: bool | None = None,
    valuation_status: str | None = None,
    fallback_price_date: str | None = None,
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
        "capital_cny": capital_base,
        "no_broker_api_called": True,
        "no_trade_performed": True,
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
    if funding_correction is not None:
        manual["correction_type"] = (
            "reverse_erroneous_20260709_manual_funding"
        )
        manual["owner_correction"] = funding_correction
    if official_valuation is not None:
        manual["official_valuation"] = official_valuation
        manual["valuation_completeness_passed"] = official_valuation
    if valuation_status is not None:
        manual["valuation_status"] = valuation_status
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
        "data_snapshot": {
            "analysis_trade_date": data_date.replace("-", ""),
            **(
                {"valuation_status": valuation_status}
                if valuation_status is not None
                else {}
            ),
            **(
                {
                    "last_strict_completed_trade_date_for_untouched_marks": (
                        fallback_price_date.replace("-", "")
                    )
                }
                if fallback_price_date is not None
                else {}
            ),
        },
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
                "ts_code": ts_code,
                "close": base + index,
                "source_system": "tushare.index_daily",
                "value_date": value,
                "coverage": "exact_close",
            }
            for ts_code, base in (
                ("000300.SH", 100),
                ("000688.SH", 200),
                ("399006.SZ", 300),
            )
            for index, value in enumerate(dates)
        ],
    )
    return path


def _write_risk_free(project_root: Path, dates: list[str]) -> Path:
    path = (
        project_root
        / "portfolio_dashboard"
        / "inputs"
        / "cn_govt_bond_yield.csv"
    )
    _write_csv(
        path,
        [
            "date",
            "tenor",
            "annual_yield_percent",
            "source_system",
            "source_url",
        ],
        [
            {
                "date": value,
                "tenor": "1Y",
                "annual_yield_percent": "2.0",
                "source_system": "chinabond.mof_govt_yield_curve",
                "source_url": (
                    "https://yield.chinabond.com.cn/cbweb-mn/pgxh/"
                    "showHistory"
                ),
            }
            for value in dates
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
    _write_risk_free(
        tmp_path,
        ["2098-12-30", "2099-01-01", "2099-01-02", "2099-01-03"],
    )
    return tmp_path, record_root, benchmark


def _registered_projection_stub(
    *,
    project_root: Path,
    record_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_record_id: str = "20990103_1200",
    previous_record_id: str = "20990102_1200",
) -> tuple[dict, dict]:
    valid, rejected, latest_seen = scan_valid_records(
        record_root, project_root
    )
    historical, historical_rejected = (
        scan_historical_performance_records(
            record_root=record_root,
            project_root=project_root,
            strict_records=valid,
        )
    )
    projection = {
        "valid_records": valid,
        "rejected": rejected,
        "latest_seen": latest_seen,
        "historical_records": historical,
        "historical_rejected": historical_rejected,
    }
    store_root = record_root / "_record_store"
    catalog_path = store_root / "catalogs" / "fixture" / "catalog.v1.json"
    catalog = {
        "records": [
            {
                "record_id": path.name,
                "relative_path": path.name,
                "state": "ONLINE",
                "inventory": [
                    {
                        "path": source.relative_to(path).as_posix(),
                        "type": "file",
                        "size": source.stat().st_size,
                        "sha256": _sha(source),
                    }
                    for source in sorted(path.rglob("*"))
                    if source.is_file() and not source.is_symlink()
                ],
            }
            for path in sorted(record_root.iterdir())
            if path.is_dir() and path.name[:1].isdigit()
        ],
        "dashboard_projection": projection,
    }
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    pointer = {
        "active_record_id": active_record_id,
        "previous_record_id": previous_record_id,
        "catalog_path": catalog_path.relative_to(record_root).as_posix(),
        "catalog_sha256": _sha(catalog_path),
    }
    pointer_path = store_root / "current.v1.json"
    pointer_path.write_text(
        json.dumps(pointer, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_common,
        "load_registered_catalog",
        lambda _record_root: (pointer, catalog),
    )
    return pointer, catalog


def _archive_registered_record(
    *,
    project_root: Path,
    record_root: Path,
    pointer: dict,
    catalog: dict,
    record_id: str,
) -> None:
    projection_rows = [
        row
        for key in ("valid_records", "historical_records")
        for row in catalog["dashboard_projection"][key]
        if row["record"] == record_id
    ]
    assert projection_rows
    logical_refs = sorted(
        projection_rows[0]["source_refs"], key=lambda item: item["path"]
    )
    record_dir = record_root / record_id
    inventory = [
        {
            "path": source.relative_to(record_dir).as_posix(),
            "type": "file",
            "size": source.stat().st_size,
            "sha256": _sha(source),
        }
        for source in sorted(record_dir.rglob("*"))
        if source.is_file() and not source.is_symlink()
    ]
    inventory_sha = hashlib.sha256(
        store_canonical_json_bytes(inventory)
    ).hexdigest()
    archive_id = "cn-aggressive-2098-12-v1"
    archive_root = (
        project_root
        / "results/strategy_record_archives/CN/"
        "aggressive_tech_manufacturing/monthly/v1/2098-12"
    )
    archive_path = archive_root / "records.tar.zst"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"synthetic immutable archive bytes")
    archive_rel = archive_path.relative_to(project_root).as_posix()
    manifest_path = archive_root / "manifest.v1.json"
    manifest_rel = manifest_path.relative_to(project_root).as_posix()
    manifest = _write_store_document(
        manifest_path,
        {
            "schema_id": "myquant.strategy_record_archive_manifest.v1",
            "archive_id": archive_id,
            "archive_path": archive_rel,
            "archive_sha256": _sha(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "record_count": 1,
            "file_count": len(inventory),
            "logical_bytes": sum(item["size"] for item in inventory),
            "records": [
                {
                    "record_id": record_id,
                    "relative_path": record_id,
                    "member_prefix": record_id,
                    "inventory": inventory,
                    "inventory_sha256": inventory_sha,
                    "file_count": len(inventory),
                    "total_bytes": sum(item["size"] for item in inventory),
                    "logical_source_refs": logical_refs,
                }
            ],
        },
    )
    receipt_path = archive_root / "restore_receipt.v1.json"
    receipt_rel = receipt_path.relative_to(project_root).as_posix()
    _write_store_document(
        receipt_path,
        {
            "schema_id": (
                "myquant.strategy_record_archive_restore_receipt.v1"
            ),
            "archive_id": archive_id,
            "manifest_path": manifest_rel,
            "manifest_sha256": _sha(manifest_path),
            "archive_path": archive_rel,
            "archive_sha256": _sha(archive_path),
            "record_ids": [record_id],
            "record_count": 1,
            "all_inventory_matched": True,
            "restored_file_count": manifest["file_count"],
            "restored_logical_bytes": manifest["logical_bytes"],
        },
    )
    for row in catalog["records"]:
        row["storage_state"] = row["state"]
        row_inventory = row["inventory"]
        row["inventory_sha256"] = hashlib.sha256(
            store_canonical_json_bytes(row_inventory)
        ).hexdigest()
        row["file_count"] = len(
            [item for item in row_inventory if item["type"] == "file"]
        )
        row["total_bytes"] = sum(
            item["size"] for item in row_inventory if item["type"] == "file"
        )
        if row["record_id"] == record_id:
            row.update(
                {
                    "state": "ARCHIVED",
                    "storage_state": "ARCHIVED",
                    "logical_source_refs": logical_refs,
                    "archive_locator": {
                        "schema_id": (
                            "myquant.strategy_record_archive_locator.v1"
                        ),
                        "archive_id": archive_id,
                        "archive_path": archive_rel,
                        "archive_sha256": _sha(archive_path),
                        "archive_bytes": archive_path.stat().st_size,
                        "manifest_path": manifest_rel,
                        "manifest_sha256": _sha(manifest_path),
                        "restore_receipt_path": receipt_rel,
                        "restore_receipt_sha256": _sha(receipt_path),
                        "member_prefix": record_id,
                    },
                }
            )
    catalog["schema_id"] = "myquant.strategy_record_catalog.v2"
    catalog["generation_id"] = "fixture"
    catalog["active_record_id"] = pointer["active_record_id"]
    catalog["previous_record_id"] = pointer["previous_record_id"]
    normalized_projection = {
        **catalog["dashboard_projection"],
        "valid_records": [
            {
                **row,
                "source_refs": sorted(
                    row["source_refs"], key=lambda ref: ref["path"]
                ),
            }
            for row in catalog["dashboard_projection"]["valid_records"]
        ],
        "historical_records": [
            {
                **row,
                "source_refs": sorted(
                    row["source_refs"], key=lambda ref: ref["path"]
                ),
            }
            for row in catalog["dashboard_projection"][
                "historical_records"
            ]
        ],
    }
    projection_sha = hashlib.sha256(
        canonical_json_bytes(normalized_projection)
    ).hexdigest()
    catalog_by_id = {row["record_id"]: row for row in catalog["records"]}
    registry_records = []
    archive_bindings = {}
    for row in normalized_projection["historical_records"]:
        catalog_row = catalog_by_id[row["record"]]
        registry_row = {
            **row,
            "storage_state": catalog_row["state"],
            "record_inventory_sha256": catalog_row["inventory_sha256"],
            "logical_source_refs": row["source_refs"],
        }
        registry_records.append(registry_row)
        if catalog_row["state"] == "ARCHIVED":
            locator = catalog_row["archive_locator"]
            archive_bindings[row["record"]] = {
                "record_inventory_sha256": catalog_row[
                    "inventory_sha256"
                ],
                "archive_storage_refs": [
                    {
                        "path": locator["manifest_path"],
                        "sha256": locator["manifest_sha256"],
                        "bytes": (project_root / locator["manifest_path"])
                        .stat()
                        .st_size,
                        "media_type": "application/json",
                    },
                    {
                        "path": locator["restore_receipt_path"],
                        "sha256": locator["restore_receipt_sha256"],
                        "bytes": (
                            project_root / locator["restore_receipt_path"]
                        )
                        .stat()
                        .st_size,
                        "media_type": "application/json",
                    },
                    {
                        "path": locator["archive_path"],
                        "sha256": locator["archive_sha256"],
                        "bytes": locator["archive_bytes"],
                        "media_type": "application/zstd",
                    },
                ],
            }
    history_registry = build_history_integrity_registry(
        registry_records,
        generated_at="2099-01-03T11:00:00+08:00",
        intended_generation_id="fixture",
        dashboard_projection_sha256=projection_sha,
        archive_bindings=archive_bindings,
    )
    registry_path = (
        project_root
        / "portfolio_dashboard/private/generated/"
        "cn_aggressive_history_integrity.v2.json"
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(history_registry, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    catalog["history_registry"] = history_registry
    catalog["history_registry_ref"] = {
        "path": registry_path.relative_to(project_root).as_posix(),
        "sha256": _sha(registry_path),
    }
    catalog_path = record_root / pointer["catalog_path"]
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    pointer["generation_id"] = "fixture"
    pointer["catalog_sha256"] = _sha(catalog_path)
    (record_root / "_record_store/current.v1.json").write_text(
        json.dumps(pointer, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    for source in sorted(record_dir.rglob("*"), reverse=True):
        if source.is_file():
            source.unlink()
        elif source.is_dir():
            source.rmdir()
    record_dir.rmdir()


def _publish_fixture_metadata_generation(
    *,
    record_root: Path,
    pointer: dict,
    catalog: dict,
    generation_id: str,
) -> None:
    catalog["generation_id"] = generation_id
    catalog.setdefault("receipts", []).append(
        {
            "schema_id": "myquant.strategy_record_no_action_receipt.v1",
            "receipt_id": generation_id,
            "status": "NO_ACTION",
        }
    )
    catalog_path = record_root / pointer["catalog_path"]
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    pointer["generation_id"] = generation_id
    pointer["catalog_sha256"] = _sha(catalog_path)
    (record_root / "_record_store/current.v1.json").write_text(
        json.dumps(pointer, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def test_build_bundle_excludes_external_funding_and_is_read_only(
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
    assert bundle["latest_valid_record"] == "20990103_1200"
    assert bundle["status"] == "PARTIAL"
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
    assert bundle["portfolio"]["performance_initial_capital"] == 10000
    assert bundle["portfolio"]["excluded_external_flow"] == 10000
    assert bundle["portfolio"]["adjusted_total_value"] == 13100
    assert bundle["portfolio"]["cash"] == 12000
    assert bundle["portfolio"]["market_value"] == 1100
    assert bundle["portfolio"]["total_value"] == 13100
    assert bundle["portfolio"]["portfolio_pnl"] == 3100
    assert bundle["portfolio"]["cash_weight"] == pytest.approx(
        12000 / 13100
    )
    assert bundle["portfolio"]["gross_exposure"] == pytest.approx(
        1100 / 13100
    )
    assert bundle["positions"][0]["nav_weight"] == pytest.approx(
        1100 / 13100
    )
    assert bundle["changes"][0]["nav_weight_delta"] == pytest.approx(
        1100 / 13100 - 1100 / 11000
    )
    assert bundle["portfolio"][
        "cumulative_profit_excluding_external_flow"
    ] == 3100
    assert bundle["portfolio"]["cumulative_return"] == pytest.approx(0.31)
    assert bundle["portfolio"]["performance_points"][2][
        "adjusted_total_value"
    ] == 11000
    assert bundle["portfolio"]["performance_points"][-1][
        "excluded_external_flow"
    ] == 10000
    assert bundle["portfolio"]["latest_interval_turnover"] == 0
    assert (
        bundle["portfolio"]["return_method"]
        == "initial_capital_return_excluding_external_flows"
    )
    assert bundle["benchmarks"][0]["id"] == "CSI300"
    assert bundle["benchmarks"][0]["missing_dates"] == []
    assert bundle["risk_free"]["tenor"] == "1Y"
    assert bundle["risk_free"]["latest_annual_yield"] == pytest.approx(0.02)
    assert bundle["portfolio"]["performance_points"][-1][
        "risk_free_annual_yield"
    ] == pytest.approx(0.02)
    assert [row["id"] for row in bundle["benchmarks"]] == [
        "CSI300",
        "STAR50",
        "CHINEXT",
    ]
    assert bundle["portfolio"]["performance_points"][-1][
        "star50_nav"
    ] == pytest.approx(203 / 200)
    assert bundle["portfolio"]["performance_points"][-1][
        "chinext_nav"
    ] == pytest.approx(303 / 300)
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


def test_owner_correction_reverses_false_funding_once(
    tmp_path: Path,
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    correction = {
        "declared_on": "2099-01-04",
        "declaration": "synthetic owner correction",
        "initial_capital_cny": 10000,
        "erroneous_funding_amount_reversed_cny": 10000,
        "erroneous_funding_record": "20990102_1200",
        "position_events_after_20260709_preserved": True,
    }
    _write_record(
        tmp_path,
        "20990104_1200",
        "2099-01-04",
        13100,
        source_record="20990103_1200",
        funding_correction=correction,
        capital_base=10000,
    )
    dates = [
        "2098-12-30",
        "2099-01-01",
        "2099-01-02",
        "2099-01-03",
        "2099-01-04",
    ]
    benchmark = _write_benchmark(tmp_path, dates)
    _write_risk_free(tmp_path, dates)

    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-04T12:00:00+08:00",
        today=date(2099, 1, 4),
    )

    assert bundle["latest_valid_record"] == "20990104_1200"
    assert bundle["history"]["funding_events"] == []
    assert bundle["history"]["net_external_flow"] == 0
    assert bundle["portfolio"]["excluded_external_flow"] == 0
    assert bundle["portfolio"]["cash"] == 12000
    assert bundle["portfolio"]["market_value"] == 1100
    assert bundle["portfolio"]["total_value"] == 13100
    assert bundle["portfolio"]["cumulative_return"] == pytest.approx(0.31)
    assert bundle["portfolio"]["performance_points"][-2][
        "adjusted_total_value"
    ] == 13100
    assert bundle["portfolio"]["performance_points"][-1][
        "adjusted_total_value"
    ] == 13100


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


def test_missing_risk_free_prior_date_blocks_export(tmp_path: Path) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    _write_risk_free(project_root, ["2099-01-03"])
    with pytest.raises(
        DashboardInputError,
        match="risk_free_missing_prior_date:2098-12-30",
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


def test_exact_backfill_provenance_can_close_current_carry_forward(
    tmp_path: Path,
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    latest_dir = record_root / "20990103_1200"
    manifest_path = latest_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manual_execution")
    manifest["data_snapshot"] = {
        "valuation_trade_date": "20990103",
        "freshness_mode": (
            "strict_parquet_canonical_historical_revaluation"
        ),
        "snapshot_id_at_backfill": "20990103T120000Z",
        "market_pointer_sha256_at_backfill": "d" * 64,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_path = (
        project_root / ".codex" / "automations" / "history.md"
    )
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        "# Synthetic automation history\n2099 current carry forward\n",
        encoding="utf-8",
    )
    inventory = []
    for path in sorted(latest_dir.iterdir()):
        if path.is_file():
            inventory.append(
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
    provenance = {
        "schema_version": "cn_aggressive_transaction_backfill_provenance.v1",
        "record": latest_dir.name,
        "backfilled_at": "2099-01-03T12:30:00+08:00",
        "source_record": "20990102_1200",
        "history_path": str(history_path.resolve()),
        "history_sha256": _sha(history_path),
        "history_section": "2099 current carry forward",
        "record_inventory_sha256_before_provenance": "e" * 64,
        "record_files_before_provenance": inventory,
    }
    (latest_dir / "backfill_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T13:00:00+08:00",
        today=date(2099, 1, 3),
    )

    assert bundle["latest_valid_record"] == "20990103_1200"
    assert bundle["latest_data_date"] == "2099-01-03"
    assert any(
        ref["path"].endswith("backfill_provenance.json")
        for ref in bundle["source_refs"]
    )
    assert verify_source_refs(bundle, project_root) == []

    history_path.write_text("changed\n", encoding="utf-8")
    assert "backfill_history_sha_mismatch" in verify_source_refs(
        bundle, project_root
    )


def test_public_bundle_keeps_performance_and_removes_holdings_detail(
    tmp_path: Path,
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    source = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )

    public = sanitize_bundle(source)

    assert public["public_redacted"] is True
    assert public["portfolio"]["cumulative_return"] == source["portfolio"][
        "cumulative_return"
    ]
    assert public["portfolio"]["performance_initial_capital"] == 0
    assert public["portfolio"]["excluded_external_flow"] == 0
    assert public["portfolio"]["adjusted_total_value"] == 0
    assert public["portfolio"]["cash_weight"] == 0
    assert public["portfolio"]["gross_exposure"] == 0
    assert public["benchmarks"][0]["return"] == source["benchmarks"][0][
        "return"
    ]
    assert [row["id"] for row in public["benchmarks"]] == [
        "CSI300",
        "STAR50",
        "CHINEXT",
    ]
    assert public["portfolio"]["performance_points"][-1][
        "star50_nav"
    ] == source["portfolio"]["performance_points"][-1]["star50_nav"]
    assert public["portfolio"]["performance_points"][-1][
        "chinext_nav"
    ] == source["portfolio"]["performance_points"][-1]["chinext_nav"]
    assert public["positions"] == []
    assert public["changes"] == []
    assert public["concentration"] == {
        "equity_hhi": 0,
        "holding_count": 0,
        "thesis_status_counts": {},
        "top1_equity_weight": 0,
        "top3_equity_weight": 0,
    }
    assert public["source_refs"] == []
    assert public["risks"] == []
    assert public["warnings"] == ["public_pages_redacted_snapshot"]
    assert set(public["current_evidence"].values()).issubset(
        {"PUBLIC_REDACTED", "0" * 64, False, None}
    )
    assert validate_bundle_shape(public) == []


def test_history_integrity_registry_upgrades_legacy_exact_bytes(
    tmp_path: Path,
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    valid, _, _ = scan_valid_records(record_root, project_root)
    history, _ = scan_historical_performance_records(
        record_root=record_root,
        project_root=project_root,
        strict_records=valid,
    )
    registry = build_history_integrity_registry(
        history, generated_at="2099-01-03T11:00:00+08:00"
    )
    registry_path = (
        project_root
        / "portfolio_dashboard"
        / "private"
        / "generated"
        / "cn_aggressive_history_integrity.v1.json"
    )
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
        history_integrity_path=registry_path,
    )

    assert bundle["history"]["legacy_exact_byte_record_count"] == 0
    assert (
        bundle["history"]["dashboard_integrity_registry_record_count"] == 1
    )
    assert bundle["history"]["evidence_status"] == (
        "DASHBOARD_POST_HOC_SHA_REGISTRY_BOUND"
    )
    assert bundle["status"] == "FRESH"
    assert "trade_fee_and_net_of_fee_basis_unknown" in bundle["warnings"]
    assert not any(
        warning.startswith(
            "historical_performance_legacy_exact_bytes_without_declared_sha"
        )
        for warning in bundle["warnings"]
    )
    assert not any(
        risk["code"] == "LEGACY_HISTORY_EVIDENCE_PARTIAL"
        for risk in bundle["risks"]
    )

    stale_bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-04T12:00:00+08:00",
        today=date(2099, 1, 4),
        history_integrity_path=registry_path,
    )
    assert stale_bundle["status"] == "PARTIAL"
    assert (
        "latest_performance_stale_calendar_days:1"
        in stale_bundle["warnings"]
    )


def test_unvalued_current_holdings_do_not_create_performance_point(
    tmp_path: Path,
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    _write_record(
        tmp_path,
        "20990104_1200",
        "2099-01-04",
        23200,
        source_record="20990103_1200",
        capital_base=20000,
        official_valuation=False,
        valuation_status="BLOCKED_PENDING_STRICT_CLOSE",
        fallback_price_date="2099-01-03",
    )

    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-04T12:00:00+08:00",
        today=date(2099, 1, 4),
    )

    assert bundle["latest_valid_record"] == "20990104_1200"
    assert bundle["latest_data_date"] == "2099-01-04"
    assert bundle["portfolio"]["performance_end_date"] == "2099-01-03"
    assert bundle["portfolio"]["total_value"] == 13200
    assert bundle["positions"][0]["price_date"] == "2099-01-03"
    assert bundle["portfolio"]["performance_points"][-1]["record"] == (
        "20990103_1200"
    )
    assert bundle["status"] == "PARTIAL"
    assert (
        "latest_performance_stale_calendar_days:1" in bundle["warnings"]
    )
    assert (
        "latest_current_valuation_incomplete:BLOCKED_PENDING_STRICT_CLOSE"
        in bundle["warnings"]
    )


def test_official_tushare_valuation_requires_hash_bound_close_evidence(
    tmp_path: Path,
) -> None:
    project_root, record_root, _ = _fixture(tmp_path)
    record = "20990104_1200"
    record_dir = _write_record(
        tmp_path,
        record,
        "2099-01-04",
        23200,
        source_record="20990103_1200",
        capital_base=20000,
        official_valuation=True,
        valuation_status="OFFICIAL_TUSHARE_EOD_CLOSE_COMPLETE",
    )
    source_dir = record_root / "20990103_1200"
    source_manual_path = source_dir / "manual_execution_manifest.json"
    source_manual = json.loads(source_manual_path.read_text(encoding="utf-8"))
    source_ledger = source_dir / source_manual["effective_manual_ledger_path"]
    evidence = {
        "schema_version": "cn_dashboard_tushare_close_evidence.v1",
        "provider": "tushare.pro",
        "stock_api": "daily",
        "index_api": "index_daily",
        "trade_date": "20990104",
        "coverage": "exact_close",
        "previous_trading_day_ffill": False,
        "stocks": [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20990104",
                "close": 11,
            }
        ],
        "indices": [
            {
                "ts_code": code,
                "trade_date": "20990104",
                "close": 100,
            }
            for code in ("000300.SH", "000688.SH", "399006.SZ")
        ],
    }
    evidence_path = record_dir / "tushare_close_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    evidence_sha = _sha(evidence_path)
    manual_path = record_dir / "manual_execution_manifest.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    manual.update(
        {
            "valuation_evidence_sha256": evidence_sha,
            "source_manifest_sha256": _sha(source_dir / "manifest.json"),
            "source_manual_manifest_sha256": _sha(source_manual_path),
            "source_contained_ledger_sha256": _sha(source_ledger),
        }
    )
    manual_path.write_text(
        json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = record_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_manifest_sha256"] = _sha(source_dir / "manifest.json")
    manifest["manual_execution"] = manual
    manifest["files"]["valuation_evidence"] = evidence_path.name
    manifest["data_snapshot"]["valuation_evidence_sha256"] = evidence_sha
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    closed = validate_record(record_dir, record_root, project_root)
    assert closed["official_valuation"] is True
    assert closed["positions"][0]["price_date"] == "2099-01-04"
    assert any(
        ref["sha256"] == evidence_sha for ref in closed["source_refs"]
    )

    evidence["stocks"][0]["close"] = 12
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        DashboardInputError,
        match="official_valuation_evidence_sha_mismatch",
    ):
        validate_record(record_dir, record_root, project_root)


def test_registered_catalog_projection_preserves_dashboard_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    legacy = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )
    pointer, catalog = _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
    )
    candidate = dashboard_common.load_dashboard_candidate_projection(
        record_root=record_root,
        project_root=project_root,
        pointer=pointer,
        catalog=catalog,
    )
    assert candidate["selection"]["active_record_id"] == "20990103_1200"

    registered = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )

    for key in (
        "latest_valid_record",
        "previous_valid_record",
        "latest_data_date",
        "positions",
        "portfolio",
        "history",
        "changes",
    ):
        assert registered[key] == legacy[key]
    store_refs = {
        Path(ref["path"]).name for ref in registered["source_refs"]
    }
    assert {"current.v1.json", "catalog.v1.json"}.issubset(store_refs)
    assert verify_source_refs(registered, project_root) == []


def test_registered_archive_projection_survives_hot_record_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    legacy = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )
    pointer, catalog = _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
    )
    _archive_registered_record(
        project_root=project_root,
        record_root=record_root,
        pointer=pointer,
        catalog=catalog,
        record_id="20981230_1200",
    )

    projection = dashboard_common.load_dashboard_catalog_projection(
        record_root, project_root
    )
    candidate = dashboard_common.load_dashboard_candidate_projection(
        record_root=record_root,
        project_root=project_root,
        pointer=pointer,
        catalog=catalog,
    )
    assert (
        candidate["integrity_context"]["dashboard_projection_sha256"]
        == projection["integrity_context"]["dashboard_projection_sha256"]
    )
    context = projection["integrity_context"]
    registry_path = project_root / context["history_registry_ref"]["path"]
    with pytest.raises(
        DashboardInputError, match="catalog_history_registry_path_required"
    ):
        build_bundle(
            project_root=project_root,
            record_root=record_root,
            benchmark_path=benchmark,
            generated_at="2099-01-03T12:00:00+08:00",
            today=date(2099, 1, 3),
        )
    registered = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
        history_integrity_path=registry_path,
    )

    for key in ("positions", "changes"):
        assert registered[key] == legacy[key]
    registered_portfolio = json.loads(json.dumps(registered["portfolio"]))
    legacy_portfolio = json.loads(json.dumps(legacy["portfolio"]))
    for point in registered_portfolio["performance_points"]:
        point.pop("evidence_status")
    for point in legacy_portfolio["performance_points"]:
        point.pop("evidence_status")
    assert registered_portfolio == legacy_portfolio
    assert not (record_root / "20981230_1200").exists()
    assert all(
        "20981230_1200" not in ref["path"]
        for ref in registered["source_refs"]
    )
    assert any(
        ref["path"].endswith("records.tar.zst")
        for ref in registered["source_refs"]
    )
    assert verify_source_refs(registered, project_root) == []

    archive_ref = next(
        ref
        for ref in registered["source_refs"]
        if ref["path"].endswith("records.tar.zst")
    )
    (project_root / archive_ref["path"]).write_bytes(b"tampered")
    with pytest.raises(DashboardInputError, match="archive_binding_invalid"):
        build_bundle(
            project_root=project_root,
            record_root=record_root,
            benchmark_path=benchmark,
            generated_at="2099-01-03T12:00:00+08:00",
            today=date(2099, 1, 3),
        )


def test_registered_history_binding_survives_consecutive_no_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    pointer, catalog = _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
    )
    _archive_registered_record(
        project_root=project_root,
        record_root=record_root,
        pointer=pointer,
        catalog=catalog,
        record_id="20981230_1200",
    )
    registry_path = project_root / catalog["history_registry_ref"]["path"]
    registry_bytes = registry_path.read_bytes()
    original_records = json.loads(json.dumps(catalog["records"]))
    original_projection = json.loads(
        json.dumps(catalog["dashboard_projection"])
    )
    original_registry = json.loads(json.dumps(catalog["history_registry"]))
    original_registry_ref = dict(catalog["history_registry_ref"])

    for generation_id in ("g-noop-1", "g-noop-2"):
        _publish_fixture_metadata_generation(
            record_root=record_root,
            pointer=pointer,
            catalog=catalog,
            generation_id=generation_id,
        )
        projection = dashboard_common.load_dashboard_catalog_projection(
            record_root, project_root
        )
        assert projection["integrity_context"][
            "publication_generation_id"
        ] == generation_id
        assert projection["integrity_context"][
            "intended_generation_id"
        ] == "fixture"

    assert catalog["records"] == original_records
    assert catalog["dashboard_projection"] == original_projection
    assert catalog["history_registry"] == original_registry
    assert catalog["history_registry_ref"] == original_registry_ref
    assert registry_path.read_bytes() == registry_bytes
    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
        history_integrity_path=registry_path,
    )
    assert bundle["latest_valid_record"] == pointer["active_record_id"]
    assert verify_source_refs(bundle, project_root) == []


def test_registered_history_binding_rejects_malformed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, _benchmark = _fixture(tmp_path)
    pointer, catalog = _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
    )
    _archive_registered_record(
        project_root=project_root,
        record_root=record_root,
        pointer=pointer,
        catalog=catalog,
        record_id="20981230_1200",
    )
    registry = dict(catalog["history_registry"])
    registry["intended_generation_id"] = "../malformed"
    registry_without_hash = dict(registry)
    registry_without_hash.pop("content_sha256")
    registry["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(registry_without_hash)
    ).hexdigest()
    registry_path = project_root / catalog["history_registry_ref"]["path"]
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    catalog["history_registry"] = registry
    catalog["history_registry_ref"]["sha256"] = _sha(registry_path)
    _publish_fixture_metadata_generation(
        record_root=record_root,
        pointer=pointer,
        catalog=catalog,
        generation_id="g-noop",
    )

    with pytest.raises(
        DashboardInputError,
        match="catalog_history_registry_generation_invalid",
    ):
        dashboard_common.load_dashboard_catalog_projection(
            record_root, project_root
        )


def test_registered_history_binding_rejects_projection_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, _benchmark = _fixture(tmp_path)
    pointer, catalog = _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
    )
    _archive_registered_record(
        project_root=project_root,
        record_root=record_root,
        pointer=pointer,
        catalog=catalog,
        record_id="20981230_1200",
    )
    catalog["dashboard_projection"]["historical_rejected"].append(
        "20980101_1200:changed"
    )
    _publish_fixture_metadata_generation(
        record_root=record_root,
        pointer=pointer,
        catalog=catalog,
        generation_id="g-noop",
    )

    with pytest.raises(
        DashboardInputError,
        match="history_integrity_registry_projection_mismatch",
    ):
        dashboard_common.load_dashboard_catalog_projection(
            record_root, project_root
        )


def test_registered_archive_rejects_active_row_and_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, _benchmark = _fixture(tmp_path)
    pointer, catalog = _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
    )
    _archive_registered_record(
        project_root=project_root,
        record_root=record_root,
        pointer=pointer,
        catalog=catalog,
        record_id="20981230_1200",
    )
    archived = next(
        row for row in catalog["records"] if row["state"] == "ARCHIVED"
    )
    pointer["active_record_id"] = archived["record_id"]
    with pytest.raises(
        DashboardInputError, match="record_catalog_pointer_projection_mismatch"
    ):
        dashboard_common.load_dashboard_catalog_projection(
            record_root, project_root
        )

    pointer["active_record_id"] = "20990103_1200"
    original_registry = catalog["history_registry"]
    catalog["history_registry"] = {
        **original_registry,
        "generated_at": "2099-01-03T11:00:01+08:00",
    }
    with pytest.raises(
        DashboardInputError, match="catalog_history_registry_body_mismatch"
    ):
        dashboard_common.load_dashboard_catalog_projection(
            record_root, project_root
        )
    catalog["history_registry"] = original_registry
    archived["archive_locator"]["manifest_path"] = "../escape.json"
    with pytest.raises(DashboardInputError, match="archive_binding_invalid"):
        dashboard_common.load_dashboard_catalog_projection(
            record_root, project_root
        )


def test_catalog_projection_canonicalizes_duplicate_source_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_ref = {"path": "records/run/ledger.parquet", "sha256": "a" * 64}
    row = {
        "record": "20990101_0900",
        "source_refs": [dict(source_ref), dict(source_ref)],
    }
    monkeypatch.setattr(
        dashboard_common, "load_registered_catalog", lambda _root: None
    )
    monkeypatch.setattr(
        dashboard_common,
        "scan_valid_records",
        lambda *_args, **_kwargs: ([row], [], "20990101_0900"),
    )
    monkeypatch.setattr(
        dashboard_common,
        "scan_historical_performance_records",
        lambda **_kwargs: ([dict(row)], []),
    )

    projection = dashboard_common.build_dashboard_catalog_projection(
        tmp_path / "records", tmp_path
    )

    assert projection["valid_records"][0]["source_refs"] == [source_ref]
    assert projection["historical_records"][0]["source_refs"] == [source_ref]


def test_registered_catalog_never_calls_legacy_raw_scanners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
    )

    def unexpected_scan(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("registered Dashboard called legacy raw scanner")

    monkeypatch.setattr(
        dashboard_common, "scan_valid_records", unexpected_scan
    )
    monkeypatch.setattr(
        dashboard_common,
        "scan_historical_performance_records",
        unexpected_scan,
    )

    projection = dashboard_common.load_dashboard_catalog_projection(
        record_root, project_root
    )
    assert projection["valid_records"][-1]["record"] == "20990103_1200"
    assert projection["historical_records"]

    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )
    assert bundle["latest_valid_record"] == "20990103_1200"


def test_registered_pointer_ids_override_projection_sort_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
        active_record_id="20990102_1200",
        previous_record_id="20990103_1200",
    )

    bundle = build_bundle(
        project_root=project_root,
        record_root=record_root,
        benchmark_path=benchmark,
        generated_at="2099-01-03T12:00:00+08:00",
        today=date(2099, 1, 3),
    )

    assert bundle["latest_valid_record"] == "20990102_1200"
    assert bundle["previous_valid_record"] == "20990103_1200"


def test_registered_pointer_active_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)
    _registered_projection_stub(
        project_root=project_root,
        record_root=record_root,
        monkeypatch=monkeypatch,
        active_record_id="20990109_1200",
    )

    with pytest.raises(
        DashboardInputError,
        match="record_catalog_pointer_projection_mismatch",
    ):
        build_bundle(
            project_root=project_root,
            record_root=record_root,
            benchmark_path=benchmark,
            generated_at="2099-01-03T12:00:00+08:00",
            today=date(2099, 1, 3),
        )


def test_corrupt_registered_pointer_fails_without_legacy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, record_root, benchmark = _fixture(tmp_path)

    def corrupt_pointer(_record_root: Path) -> None:
        raise dashboard_common.StrategyRecordStoreError(
            "pointer_json_unreadable"
        )

    monkeypatch.setattr(
        dashboard_common, "load_registered_catalog", corrupt_pointer
    )
    monkeypatch.setattr(
        dashboard_common,
        "build_dashboard_catalog_projection",
        lambda *_args, **_kwargs: pytest.fail("legacy fallback called"),
    )

    with pytest.raises(
        DashboardInputError,
        match="record_catalog_invalid:pointer_json_unreadable",
    ):
        build_bundle(
            project_root=project_root,
            record_root=record_root,
            benchmark_path=benchmark,
            generated_at="2099-01-03T12:00:00+08:00",
            today=date(2099, 1, 3),
        )
