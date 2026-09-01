from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.strategy_records.performance import (
    MAX_PERFORMANCE_JSON_BYTES,
    build_manifest as build_performance_manifest,
    build_owner_declaration,
    build_performance_history_ref,
    immutable_write,
    load_performance_history,
    write_deterministic_parquet,
)
from quant_investor.strategy_records.store import (
    CATALOG_SCHEMA_V2,
    CATALOG_SCHEMA_V3,
    StrategyRecordConflict,
    StrategyRecordStoreError,
    bootstrap_catalog,
    canonical_json_bytes,
    content_sha256,
    load_registered_catalog,
    publish_catalog,
)
from scripts import manage_cn_strategy_records as manager
from scripts.cn_dashboard_common import DashboardInputError


def _args(**values: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "record_root": "",
        "record_dir": [],
        "active_record_id": None,
        "previous_record_id": None,
        "dashboard_projection_json": None,
        "generation_id": None,
        "published_at": "2026-08-10T00:00:00Z",
        "expected_current_id": None,
        "expected_previous_id": None,
        "expected_inventory_sha": None,
        "project_root": None,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def _bootstrap(root: Path) -> dict[str, object]:
    run = root / "20260809_1000"
    run.mkdir(parents=True)
    (run / "holdings.json").write_text('{"cash":1}\n', encoding="utf-8")
    return manager.command_bootstrap(
        _args(
            record_root=str(root),
            generation_id="g1",
            record_dir=[run.name],
        )
    )


def _pointer_sha(root: Path) -> str:
    return hashlib.sha256((root / "_record_store" / "current.v1.json").read_bytes()).hexdigest()


def test_inventory_reports_unregistered_legacy_and_registered_orphans(
    tmp_path: Path,
) -> None:
    (tmp_path / "legacy").mkdir()
    before = manager.command_inventory(_args(record_root=str(tmp_path)))
    assert before["registered"] is False
    assert before["orphan_record_dirs"] == ["legacy"]
    _bootstrap(tmp_path)
    after = manager.command_inventory(_args(record_root=str(tmp_path)))
    assert after["registered"] is True
    assert after["orphan_record_dirs"] == ["legacy"]
    assert after["orphans_preserved"] is True


def test_batch_record_identity_is_versioned_without_changing_legacy_identity() -> None:
    assert manager._strict_run_id("20260821_1200") == "20260821_1200"
    assert manager._strict_run_id("20260901_090000-b01") == "20260901_090000-b01"
    with pytest.raises(StrategyRecordStoreError, match="YYYYMMDD"):
        manager._strict_run_id("20260901_090000")


def test_reselect_catalog_command_requires_explicit_owner_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _args(
        record_root="/governed/root",
        expected_current_pointer_sha="a" * 64,
        target_generation_id="g-pre-cutover",
        target_catalog_path=("_record_store/catalogs/g-pre-cutover/catalog.v2.json"),
        target_catalog_sha="b" * 64,
        published_at="2026-08-15T08:00:00Z",
        owner_approved_by="not-owner",
        approval_reason="owner-approved rollback",
    )
    with pytest.raises(StrategyRecordStoreError, match="explicit owner approval"):
        manager.command_reselect_catalog(values)

    captured: dict[str, object] = {}

    def fake_reselect(record_root: str, **kwargs: object) -> dict[str, object]:
        captured.update({"record_root": record_root, **kwargs})
        return {
            "pointer_sha256": "c" * 64,
            "pointer": {
                "generation_id": "g-pre-cutover",
                "catalog_path": ("_record_store/catalogs/g-pre-cutover/catalog.v2.json"),
                "catalog_sha256": "b" * 64,
            },
            "catalog": {"schema_id": CATALOG_SCHEMA_V2},
        }

    monkeypatch.setattr(manager, "reselect_catalog", fake_reselect)
    values.owner_approved_by = "maxwell"
    result = manager.command_reselect_catalog(values)
    assert result["catalog_reselected"] is True
    assert result["catalog_created"] is False
    assert result["performance_contract_ready"] is False
    assert captured["expected_current_pointer_sha256"] == "a" * 64

    parsed = manager.build_parser().parse_args(
        [
            "reselect-catalog",
            "--record-root",
            "/governed/root",
            "--expected-current-pointer-sha",
            "a" * 64,
            "--target-generation-id",
            "g-pre-cutover",
            "--target-catalog-path",
            "_record_store/catalogs/g-pre-cutover/catalog.v2.json",
            "--target-catalog-sha",
            "b" * 64,
            "--published-at",
            "2026-08-15T08:00:00Z",
            "--owner-approved-by",
            "maxwell",
            "--approval-reason",
            "owner-approved rollback",
        ]
    )
    assert parsed.handler is manager.command_reselect_catalog
    assert parsed.mutating is True


def test_identity_declaration_is_create_once_and_exact(tmp_path: Path) -> None:
    project = tmp_path / "project"
    record_root = project / ("results/strategy_records/CN/aggressive_tech_manufacturing")
    record_root.mkdir(parents=True)
    values = _args(
        record_root=str(record_root),
        project_root=str(project),
        identity_path=manager.IDENTITY_RELATIVE_PATH,
        declared_at="2026-08-15T06:00:00Z",
        provenance=None,
    )
    first = manager.command_declare_strategy_identity(values)
    second = manager.command_declare_strategy_identity(values)
    identity = project / manager.IDENTITY_RELATIVE_PATH

    assert first == second
    assert hashlib.sha256(identity.read_bytes()).hexdigest() == first["identity_sha256"]
    assert oct(identity.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(StrategyRecordConflict, match="conflict"):
        manager.command_declare_strategy_identity(
            _args(
                record_root=str(record_root),
                project_root=str(project),
                identity_path=manager.IDENTITY_RELATIVE_PATH,
                declared_at="2026-08-15T06:00:00Z",
                provenance="different exact owner declaration",
            )
        )


def _performance_record(record_id: str, *, active: bool) -> dict[str, object]:
    return {
        "record_id": record_id,
        "relative_path": record_id,
        "state": "ONLINE",
        "storage_state": "ONLINE",
        "sealed_at": "2026-08-15T06:00:00Z",
        "inventory": [],
        "inventory_sha256": hashlib.sha256(b"[]\n").hexdigest(),
        "file_count": 0,
        "total_bytes": 0,
        "manifest_path": f"{record_id}/manifest.json",
        "manifest_sha256": ("7" if active else "8") * 64,
        "manual_manifest_path": f"{record_id}/manual_execution_manifest.json",
        "manual_manifest_sha256": ("a" if active else "b") * 64,
        "ledger_path": f"{record_id}/ledger_after_manual_switch.parquet",
        "ledger_sha256": ("c" if active else "d") * 64,
        "financial_state_sha256": ("e" if active else "f") * 64,
    }


def test_prepare_performance_migration_writes_tmp_candidate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    record_root = project / ("results/strategy_records/CN/aggressive_tech_manufacturing")
    previous_id = "20260813_1800"
    active_id = "20260814_1800"
    for record_id in (previous_id, active_id):
        (record_root / record_id).mkdir(parents=True)
    records = [
        _performance_record(previous_id, active=False),
        _performance_record(active_id, active=True),
    ]
    projection = {
        "valid_records": [
            {
                "record": previous_id,
                "source_record": None,
                "data_date": "2026-08-13",
                "execution_kind": "carry_forward",
                "execution_status": "no_action_carry_forward_official_valuation",
                "official_valuation": True,
            },
            {
                "record": active_id,
                "source_record": previous_id,
                "data_date": "2026-08-14",
                "execution_kind": "carry_forward",
                "execution_status": "no_action_carry_forward_official_valuation",
                "official_valuation": True,
            },
        ],
        "historical_records": [
            {
                "record": previous_id,
                "valuation_date": "2026-08-13",
                "accounting": {
                    "cash_after": 1_000_000,
                    "market_value_after": 0,
                    "total_value_after": 1_000_000,
                    "portfolio_pnl_after": 0,
                },
                "capital_base": 1_000_000,
                "funding": None,
                "funding_correction": None,
                "evidence_status": "REGISTERED",
                "source_refs": [
                    {
                        "path": "forbidden/ledger.csv",
                        "sha256": "9" * 64,
                    }
                ],
            },
            {
                "record": active_id,
                "valuation_date": "2026-08-14",
                "accounting": {
                    "cash_after": 1_010_000,
                    "market_value_after": 0,
                    "total_value_after": 1_010_000,
                    "portfolio_pnl_after": 10_000,
                },
                "capital_base": 1_000_000,
                "funding": None,
                "funding_correction": None,
                "evidence_status": "REGISTERED",
                "source_refs": [
                    {
                        "path": "forbidden/ledger.csv",
                        "sha256": "8" * 64,
                    }
                ],
            },
        ],
        "rejected": [],
        "historical_rejected": [],
        "latest_seen": active_id,
    }
    initial = bootstrap_catalog(
        record_root,
        records=records,
        dashboard_projection=projection,
        active_record_id=active_id,
        previous_record_id=previous_id,
        generation_id="g-v2",
        published_at="2026-08-15T06:00:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    identity = manager.command_declare_strategy_identity(
        _args(
            record_root=str(record_root),
            project_root=str(project),
            identity_path=manager.IDENTITY_RELATIVE_PATH,
            declared_at="2026-08-15T06:00:00Z",
            provenance=None,
        )
    )
    output = tmp_path / "candidate"
    monkeypatch.setattr(manager, "assert_private_tmp", lambda path: Path(path))
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name == "ledger.csv":
            raise AssertionError("migration attempted to read the disabled ledger")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    result = manager.command_prepare_performance_migration(
        _args(
            record_root=str(record_root),
            project_root=str(project),
            expected_pointer_sha=initial["pointer_sha256"],
            performance_generation_id="p-seed",
            identity_path=manager.IDENTITY_RELATIVE_PATH,
            identity_sha=identity["identity_sha256"],
            generated_at="2026-08-15T06:01:00Z",
            owner_declared_at="2026-08-15T06:01:00Z",
            output_dir=str(output),
        )
    )

    assert result["prepared"] is True
    assert result["published"] is False
    assert result["store_pointer_modified"] is False
    assert result["row_count"] == 2
    assert _pointer_sha(record_root) == initial["pointer_sha256"]
    assert not (output / "owner_declaration.v1.json").exists()
    for candidate in output.iterdir():
        assert b"ledger.csv" not in candidate.read_bytes().lower()

    owner = manager.command_seal_performance_owner_declaration(
        _args(
            candidate_dir=str(output),
            candidate_receipt_sha=result["candidate_receipt_sha256"],
            approved_manifest_sha=result["candidate_manifest_sha256"],
            approved_series_sha=result["series_sha256"],
            approved_owner_declaration_sha=result["prospective_owner_declaration_sha256"],
        )
    )
    published = manager.command_publish_performance_migration(
        _args(
            record_root=str(record_root),
            expected_pointer_sha=initial["pointer_sha256"],
            candidate_dir=str(output),
            candidate_receipt_sha=result["candidate_receipt_sha256"],
            candidate_manifest_sha=result["candidate_manifest_sha256"],
            series_sha=result["series_sha256"],
            owner_declaration_sha=owner["owner_declaration_sha256"],
            lineage_document_sha=result["lineage_document_sha256"],
            performance_generation_id="p-seed",
            catalog_generation_id="g-v3",
            published_at="2026-08-15T06:02:00Z",
        )
    )
    verified = manager.command_verify(_args(record_root=str(record_root)))

    assert published["performance_contract_ready"] is True
    assert verified["catalog_schema"] == "myquant.strategy_record_catalog.v3"
    assert verified["performance_contract_ready"] is True


def _initial_v3_store(project: Path) -> tuple[Path, dict[str, object]]:
    root = project / "results/strategy_records/CN/aggressive_tech_manufacturing"
    previous_id = "20260101_1000"
    active_id = "20260102_1000"
    sha = {
        previous_id: {"manifest": "1" * 64, "manual": "2" * 64, "ledger": "3" * 64, "fs": "4" * 64},
        active_id: {"manifest": "5" * 64, "manual": "6" * 64, "ledger": "7" * 64, "fs": "8" * 64},
    }
    records: list[dict[str, object]] = []
    for record_id in (previous_id, active_id):
        record_dir = root / record_id
        record_dir.mkdir(parents=True)
        inventory = manager.build_inventory(record_dir, enforce_new_record_budget=True)
        records.append(
            {
                "record_id": record_id,
                "relative_path": record_id,
                "state": "ONLINE",
                "storage_state": "ONLINE",
                "sealed_at": "2026-01-02T02:00:00Z",
                **inventory,
                "manifest_path": f"{record_id}/manifest.json",
                "manifest_sha256": sha[record_id]["manifest"],
                "manual_manifest_path": f"{record_id}/manual_execution_manifest.json",
                "manual_manifest_sha256": sha[record_id]["manual"],
                "ledger_path": f"{record_id}/ledger_after_manual_switch.parquet",
                "ledger_sha256": sha[record_id]["ledger"],
                "financial_state_sha256": sha[record_id]["fs"],
            }
        )
    initial = bootstrap_catalog(
        root,
        records=records,
        active_record_id=active_id,
        previous_record_id=previous_id,
        generation_id="g-v2",
        published_at="2026-01-02T02:00:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    rows = []
    for sequence, record_id, day, nav in (
        (1, previous_id, "2026-01-01", Decimal("1000000.0000")),
        (2, active_id, "2026-01-02", Decimal("1010000.0000")),
    ):
        unit_nav = nav / Decimal("1000000")
        rows.append(
            {
                "sequence_no": sequence,
                "record_id": record_id,
                "valuation_at": f"{day}T02:00:00Z",
                "valuation_date": day,
                "cash_cny": nav,
                "equity_market_value_cny": Decimal("0.0000"),
                "raw_nav_cny": nav,
                "portfolio_pnl_cny": nav - Decimal("1000000"),
                "excluded_external_flow_cny": Decimal("0.0000"),
                "adjusted_nav_cny": nav,
                "unit_count": Decimal("1000000.000000000000"),
                "unit_nav": unit_nav,
                "interval_return": (
                    Decimal("0.000000000000") if sequence == 1 else Decimal("0.010000000000")
                ),
                "cumulative_return": (
                    Decimal("0.000000000000") if sequence == 1 else Decimal("0.010000000000")
                ),
                "drawdown": Decimal("0.000000000000"),
                "evidence_kind": "OWNER_DECLARED_REGISTERED_PROJECTION_MIGRATION",
                "manual_manifest_sha256": sha[record_id]["manual"],
                "ledger_parquet_sha256": sha[record_id]["ledger"],
                "financial_state_sha256": sha[record_id]["fs"],
            }
        )
    performance_generation = "p-parent"
    prefix = root / "_record_store/performance" / performance_generation
    prefix.mkdir(parents=True)
    series_sha, series_bytes = write_deterministic_parquet(rows, prefix / "series.parquet")
    owner = build_owner_declaration(
        performance_generation_id=performance_generation,
        declared_at="2026-01-02T02:01:00Z",
        series_path=f"_record_store/performance/{performance_generation}/series.parquet",
        series_sha256=series_sha,
        series_bytes=series_bytes,
        source_pointer_sha256=initial["pointer_sha256"],
        source_catalog_sha256=initial["pointer"]["catalog_sha256"],
        normalized_projection_semantic_sha256="9" * 64,
    )
    owner_raw = canonical_json_bytes(owner)
    owner_sha = immutable_write(
        prefix / "owner_declaration.v1.json",
        owner_raw,
        max_bytes=MAX_PERFORMANCE_JSON_BYTES,
    )
    manifest = build_performance_manifest(
        performance_generation_id=performance_generation,
        generated_at="2026-01-02T02:01:00Z",
        identity_path=manager.IDENTITY_RELATIVE_PATH,
        identity_sha256="a" * 64,
        parent_performance_manifest_sha256=None,
        source_pointer_sha256=initial["pointer_sha256"],
        source_catalog_generation_id="g-v2",
        source_catalog_sha256=initial["pointer"]["catalog_sha256"],
        dashboard_projection_sha256="b" * 64,
        normalized_projection_semantic_sha256="9" * 64,
        series_path=f"_record_store/performance/{performance_generation}/series.parquet",
        series_sha256=series_sha,
        series_bytes=series_bytes,
        owner_path=f"_record_store/performance/{performance_generation}/owner_declaration.v1.json",
        owner_sha256=owner_sha,
        owner_bytes=len(owner_raw),
        rows=rows,
    )
    manifest_raw = canonical_json_bytes(manifest)
    manifest_sha = immutable_write(
        prefix / "manifest.v1.json",
        manifest_raw,
        max_bytes=MAX_PERFORMANCE_JSON_BYTES,
    )
    performance_ref = build_performance_history_ref(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        manifest_bytes=len(manifest_raw),
    )
    lineage: list[dict[str, object]] = []
    parent: str | None = None
    for record_id, day in ((previous_id, "2026-01-01"), (active_id, "2026-01-02")):
        lineage.append(
            {
                "record_id": record_id,
                "source_record_id": parent,
                "supersedes_record_id": None,
                "valuation_date": day,
                "execution_class": "NO_TRADE",
                "publication_class": "OFFICIAL_FINANCIAL_STATE",
                "storage_state": "ONLINE",
                "manifest_ref": {
                    "path": f"{record_id}/manifest.json",
                    "sha256": sha[record_id]["manifest"],
                },
                "manual_manifest_ref": {
                    "path": f"{record_id}/manual_execution_manifest.json",
                    "sha256": sha[record_id]["manual"],
                },
                "effective_ledger_ref": {
                    "path": f"{record_id}/ledger_after_manual_switch.parquet",
                    "sha256": sha[record_id]["ledger"],
                },
                "financial_state_sha256": sha[record_id]["fs"],
                "ledger_parquet_sha256": sha[record_id]["ledger"],
            }
        )
        parent = record_id
    published = publish_catalog(
        root,
        expected_pointer_sha256=initial["pointer_sha256"],
        records=records,
        active_record_id=active_id,
        previous_record_id=previous_id,
        generation_id="g-v3",
        published_at="2026-01-02T02:02:00Z",
        catalog_schema=CATALOG_SCHEMA_V3,
        inherit_history_registry=False,
        lineage_index=lineage,
        performance_history_ref=performance_ref,
    )
    return root, published


def _write_applied_parquet_record(stage: Path, *, record_id: str, source_record_id: str) -> None:
    ledger_path = stage / "ledger_after_manual_switch.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "symbol": "000001.SZ",
                    "name": "平安银行",
                    "shares": 100,
                    "avg_cost": 10.0,
                    "cost_basis": 1000.0,
                    "current_price": 11.0,
                    "current_value": 1100.0,
                    "unrealized_pnl": 100.0,
                    "equity_sleeve_weight": 1.0,
                    "nav_weight": 1100.0 / 1_020_000.0,
                    "thesis_status": "FORMAL_RESEARCH",
                }
            ]
        ),
        ledger_path,
    )
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    pnl_path = stage / "pnl_summary.csv"
    with pnl_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "initial_capital",
                "cash_after",
                "market_value_after",
                "total_value_after",
                "portfolio_pnl_after",
                "realized_pnl_from_rebalance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "initial_capital": 1_000_000,
                "cash_after": 1_018_900,
                "market_value_after": 1100,
                "total_value_after": 1_020_000,
                "portfolio_pnl_after": 20_000,
                "realized_pnl_from_rebalance": 0,
            }
        )
    manual = {
        "schema_version": "cn_aggressive_manual_execution.v3",
        "status": "manual_execution_applied",
        "execution_status": "manual_execution_applied",
        "record_timestamp": record_id,
        "capital_cny": 1_000_000,
        "no_broker_api_called": True,
        "no_trade_performed": False,
        "effective_manual_ledger_path": "ledger_after_manual_switch.parquet",
        "next_ledger_path": "ledger_after_manual_switch.parquet",
        "next_ledger_sha256": ledger_sha,
        "ledger_after_manual_switch_parquet_sha256": ledger_sha,
        "ledger_provenance": {
            "contained_in_run_directory": True,
            "regular_non_symlink_file": True,
            "stable_double_read": True,
            "declared_sha256": ledger_sha,
        },
        "effective_manual_holding_count": 1,
        "cash_after": 1_018_900,
        "market_value_after": 1100,
        "total_value_after": 1_020_000,
        "portfolio_pnl_after": 20_000,
        "realized_pnl_from_rebalance": 0,
        "financial_state_sha256": "c" * 64,
        "applied_owner_declared_trades": [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "side": "BUY",
                "shares": 100,
                "execution_price": 10.0,
                "trade_date": "2026-01-03",
            }
        ],
    }
    (stage / "manual_execution_manifest.json").write_text(
        json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": record_id,
        "recorded_at": "2026-01-03 10:00:00 CST",
        "source_record": source_record_id,
        "files": {
            "manual_execution_manifest": "manual_execution_manifest.json",
            "pnl_summary": "pnl_summary.csv",
        },
        "manual_execution": manual,
        "data_snapshot": {"analysis_trade_date": "20260103"},
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def test_catalog_v3_seal_publish_advances_performance_and_no_action_inherits(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    root, initial = _initial_v3_store(project)
    record_id = "20260103_1000"
    stage = Path(
        manager.command_stage_init(_args(record_root=str(root), record_id=record_id))["staging_dir"]
    )
    _write_applied_parquet_record(stage, record_id=record_id, source_record_id="20260102_1000")
    published = manager.command_seal_publish(
        _args(
            record_root=str(root),
            record_id=record_id,
            expected_pointer_sha=initial["pointer_sha256"],
            project_root=str(project),
            generation_id="g-v3-next",
            performance_generation_id="p-next",
            cash_flow_artifact=[],
            published_at="2026-01-03T02:01:00Z",
        )
    )
    pointer, catalog = load_registered_catalog(root) or ({}, {})
    performance = load_performance_history(root, catalog["performance_history_ref"])

    assert pointer["active_record_id"] == record_id
    assert pointer["previous_record_id"] == "20260102_1000"
    assert performance["rows"][-1]["record_id"] == record_id
    assert performance["rows"][-1]["cumulative_return"] == Decimal("0.020000000000")
    assert catalog["lineage_index"][-1]["execution_class"] == "APPLIED_TRADES"

    replay = manager.command_seal_publish(
        _args(
            record_root=str(root),
            record_id=record_id,
            expected_pointer_sha=published["pointer_sha256"],
            project_root=str(project),
            generation_id="unused-replay",
            performance_generation_id="unused-replay",
            cash_flow_artifact=[],
            published_at="2026-01-03T02:02:00Z",
        )
    )
    assert replay["idempotent"] is True

    performance_ref_before = catalog["performance_history_ref"]
    no_action = manager.command_no_action(
        _args(
            record_root=str(root),
            receipt_id="weekly-no-action-2026-w01",
            reason="No new official financial state",
            expected_pointer_sha=published["pointer_sha256"],
            generation_id="g-v3-receipt",
            published_at="2026-01-03T02:03:00Z",
        )
    )
    _, after = load_registered_catalog(root) or ({}, {})
    assert no_action["catalog"]["schema_id"] == CATALOG_SCHEMA_V3
    assert after["performance_history_ref"] == performance_ref_before


def test_catalog_v3_rejects_named_or_symlinked_legacy_ledger_before_hashing(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "ledger.csv").write_text("disabled", encoding="utf-8")
    with pytest.raises(StrategyRecordStoreError, match="legacy ledger"):
        manager._reject_disabled_ledger_candidates(direct)

    linked = tmp_path / "linked"
    linked.mkdir()
    target = tmp_path / "outside.csv"
    target.write_text("disabled", encoding="utf-8")
    (linked / "ledger_after_manual_switch.csv").symlink_to(target)
    with pytest.raises(StrategyRecordStoreError, match="legacy ledger"):
        manager._reject_disabled_ledger_candidates(linked)


def test_bootstrap_live_expectations_fail_before_registration(
    tmp_path: Path,
) -> None:
    (tmp_path / "20260809_1000").mkdir()
    (tmp_path / "20260810_1000").mkdir()
    projection_path = tmp_path.parent / "projection.json"
    projection_path.write_text(
        json.dumps(
            {
                "valid_records": [
                    {"record": "20260809_1000"},
                    {"record": "20260810_1000"},
                ],
                "historical_records": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StrategyRecordStoreError, match="active record"):
        manager.command_bootstrap_live(
            _args(
                record_root=str(tmp_path),
                record_dir=[],
                dashboard_projection_json=str(projection_path),
                expected_current_id="other",
                generation_id="g1",
            )
        )
    assert not (tmp_path / "_record_store").exists()


def test_stage_seal_exact_once_and_collision(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    stage_args = _args(record_root=str(tmp_path), record_id="20260810_1000")
    stage = Path(manager.command_stage_init(stage_args)["staging_dir"])
    (stage / "payload.json").write_text('{"value":1}\n', encoding="utf-8")
    sealed = manager.command_seal_publish(
        _args(
            record_root=str(tmp_path),
            record_id="20260810_1000",
            expected_pointer_sha=_pointer_sha(tmp_path),
            generation_id="g2",
            published_at="2026-08-10T00:01:00Z",
        )
    )
    target = tmp_path / "20260810_1000"
    assert target.is_dir()
    assert not stage.exists()
    idempotent = manager.command_seal_publish(
        _args(
            record_root=str(tmp_path),
            record_id="20260810_1000",
            expected_pointer_sha=sealed["pointer_sha256"],
            generation_id="unused",
        )
    )
    assert idempotent["idempotent"] is True
    stage.mkdir()
    (stage / "payload.json").write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(StrategyRecordConflict, match="different bytes"):
        manager.command_seal_publish(
            _args(
                record_root=str(tmp_path),
                record_id="20260810_1000",
                expected_pointer_sha=sealed["pointer_sha256"],
                generation_id="g3",
            )
        )


def test_seal_publish_normalizes_relative_record_root_for_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "records"
    _bootstrap(root)
    target = root / "20260810_1000"
    target.mkdir()
    (target / "payload.json").write_text('{"value":1}\n', encoding="utf-8")
    observed: list[Path] = []

    def scan_valid(record_root: Path, project_root: Path) -> tuple[list, list, str]:
        observed.append(record_root)
        return (
            [
                {"record": "20260809_1000", "source_refs": []},
                {"record": "20260810_1000", "source_refs": []},
            ],
            [],
            "20260810_1000",
        )

    def scan_historical(**values: object) -> tuple[list, list]:
        observed.append(values["record_root"])
        return ([{"record": "20260809_1000", "source_refs": []}], [])

    monkeypatch.setattr("scripts.cn_dashboard_common.scan_valid_records", scan_valid)
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.scan_historical_performance_records",
        scan_historical,
    )
    monkeypatch.setattr(manager, "_attach_dashboard_closure", lambda *a, **k: None)
    relative_root = Path(os.path.relpath(root, Path.cwd()))
    result = manager.command_seal_publish(
        _args(
            record_root=str(relative_root),
            record_id="20260810_1000",
            expected_pointer_sha=_pointer_sha(root),
            project_root=str(tmp_path),
            generation_id="g-relative",
        )
    )
    assert result["pointer"]["active_record_id"] == "20260810_1000"
    assert observed == [root.resolve(), root.resolve()]


def test_seal_publish_atomically_adopts_existing_source_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "records"
    _bootstrap(root)
    for record_id in ("20260810_1943", "20260810_1948"):
        target = root / record_id
        target.mkdir()
        (target / "payload.json").write_text('{"value":1}\n', encoding="utf-8")

    valid = [
        {
            "record": "20260809_1000",
            "source_record": None,
            "source_refs": [],
        },
        {
            "record": "20260810_1943",
            "source_record": "20260809_1000",
            "source_refs": [],
        },
        {
            "record": "20260810_1948",
            "source_record": "20260810_1943",
            "source_refs": [],
        },
    ]
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.scan_valid_records",
        lambda *args: (valid, [], "20260810_1948"),
    )
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.scan_historical_performance_records",
        lambda **kwargs: ([], []),
    )
    monkeypatch.setattr(manager, "_attach_dashboard_closure", lambda *args, **kwargs: None)
    result = manager.command_seal_publish(
        _args(
            record_root=str(root),
            record_id="20260810_1948",
            expected_pointer_sha=_pointer_sha(root),
            project_root=str(tmp_path),
            generation_id="g-adopt-chain",
        )
    )
    assert result["pointer"]["active_record_id"] == "20260810_1948"
    assert result["pointer"]["previous_record_id"] == "20260810_1943"
    by_id = {row["record_id"]: row for row in result["catalog"]["records"]}
    assert {"20260810_1943", "20260810_1948"} <= set(by_id)


def test_dashboard_projection_source_refs_deduplicate_and_conflict() -> None:
    projection = {
        "valid_records": [
            {
                "record": "20260810_1948",
                "source_refs": [
                    {"path": "b", "sha256": "2" * 64},
                    {"path": "a", "sha256": "1" * 64},
                    {"path": "a", "sha256": "1" * 64},
                ],
            }
        ],
        "historical_records": [],
    }
    manager._normalize_dashboard_projection_source_refs(projection)
    assert projection["valid_records"][0]["source_refs"] == [
        {"path": "a", "sha256": "1" * 64},
        {"path": "b", "sha256": "2" * 64},
    ]
    projection["valid_records"][0]["source_refs"].append({"path": "a", "sha256": "3" * 64})
    with pytest.raises(StrategyRecordStoreError, match="refs conflict"):
        manager._normalize_dashboard_projection_source_refs(projection)


def test_archive_aware_seal_publish_extends_catalog_without_hot_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "records"
    target = root / "20260810_2000"
    target.mkdir(parents=True)
    (target / "payload").write_text("new", encoding="utf-8")
    pointer = {
        "active_record_id": "20260810_1948",
        "previous_record_id": "20260810_1943",
    }
    catalog = {
        "schema_id": "myquant.strategy_record_catalog.v2",
        "records": [
            {
                "record_id": "20260810_1948",
                "relative_path": "20260810_1948",
                "state": "ONLINE",
                "storage_state": "ONLINE",
            },
            {
                "record_id": "20260630_1614",
                "relative_path": "20260630_1614",
                "state": "ARCHIVED",
                "storage_state": "ARCHIVED",
            },
        ],
        "dashboard_projection": {
            "valid_records": [
                {
                    "record": "20260810_1948",
                    "source_record": "20260810_1943",
                    "source_refs": [],
                }
            ],
            "rejected": [],
            "latest_seen": "20260810_1948",
            "historical_records": [{"record": "20260630_1614", "source_refs": []}],
            "historical_rejected": [],
        },
    }
    monkeypatch.setattr(manager, "load_registered_catalog", lambda value: (pointer, catalog))
    monkeypatch.setattr(manager, "_pointer_sha", lambda value: "a" * 64)
    new_valid = {
        "record": "20260810_2000",
        "source_record": "20260810_1948",
        "source_refs": [{"path": "new", "sha256": "b" * 64}],
    }
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.validate_record",
        lambda *args: new_valid,
    )
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.validate_historical_record",
        lambda **kwargs: (_ for _ in ()).throw(
            DashboardInputError("historical_official_valuation_incomplete")
        ),
    )
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.validate_historical_performance_sequence",
        lambda rows: None,
    )
    monkeypatch.setattr(manager, "_attach_dashboard_closure", lambda *a, **k: None)
    registry_calls: list[dict[str, object]] = []

    def build_registry(**kwargs: object) -> tuple[dict, dict]:
        registry_calls.append(kwargs)
        return ({"schema_id": "registry"}, {"path": "registry", "sha256": "c" * 64})

    monkeypatch.setattr(manager, "_build_candidate_history_registry", build_registry)
    published: dict[str, object] = {}

    def publish(*args: object, **kwargs: object) -> dict:
        published.update(kwargs)
        return {"pointer": {"active_record_id": "20260810_2000"}}

    monkeypatch.setattr(manager, "publish_catalog", publish)
    result = manager.command_seal_publish(
        _args(
            record_root=str(root),
            record_id="20260810_2000",
            expected_pointer_sha="a" * 64,
            project_root=str(tmp_path),
            generation_id="g-archive-aware",
        )
    )
    assert result["pointer"]["active_record_id"] == "20260810_2000"
    assert published["catalog_schema"] == manager.CATALOG_SCHEMA_V2
    assert published["inherit_history_registry"] is False
    assert published["previous_record_id"] == "20260810_1948"
    projection = published["dashboard_projection"]
    assert [row["record"] for row in projection["historical_records"]] == ["20260630_1614"]
    assert projection["historical_rejected"] == [
        "20260810_2000:historical_official_valuation_incomplete"
    ]
    assert registry_calls[0]["generation_id"] == "g-archive-aware"


def test_stage_rejects_noncanonical_new_record_id(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    with pytest.raises(
        StrategyRecordStoreError,
        match="new record_id must use YYYYMMDD_HHMM",
    ):
        manager.command_stage_init(_args(record_root=str(tmp_path), record_id="research-output"))


def test_new_record_budgets_reject_file_count_and_total_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "record"
    record.mkdir()
    (record / "a").write_bytes(b"1234")
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_FILE_BYTES", 3)
    with pytest.raises(StrategyRecordStoreError, match="file exceeds"):
        manager.build_inventory(record, enforce_new_record_budget=True)
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_FILE_BYTES", 10)
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_TOTAL_BYTES", 3)
    with pytest.raises(StrategyRecordStoreError, match="total byte"):
        manager.build_inventory(record, enforce_new_record_budget=True)
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_TOTAL_BYTES", 10)
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_FILES", 0)
    with pytest.raises(StrategyRecordStoreError, match="file-count"):
        manager.build_inventory(record, enforce_new_record_budget=True)


def test_inventory_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    record = tmp_path / "record"
    record.mkdir()
    source = record / "source"
    source.write_bytes(b"x")
    os.symlink("source", record / "link")
    with pytest.raises(StrategyRecordStoreError, match="symlink"):
        manager.build_inventory(record, enforce_new_record_budget=False)
    (record / "link").unlink()
    os.link(source, record / "hard")
    with pytest.raises(StrategyRecordStoreError, match="hard links"):
        manager.build_inventory(record, enforce_new_record_budget=False)


def test_no_action_receipt_references_checkpoint_without_copying_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bootstrap(tmp_path)
    result = manager.command_no_action(
        _args(
            record_root=str(tmp_path),
            receipt_id="no-action-1",
            reason="readiness blocked",
            expected_pointer_sha=_pointer_sha(tmp_path),
            generation_id="g2",
            published_at="2026-08-10T00:01:00Z",
        )
    )
    receipt = result["catalog"]["receipts"][-1]
    assert receipt["active_checkpoint"] == result["pointer"]["active_closure"]
    assert receipt["payload_copied"] is False
    assert "inventory" not in receipt["active_checkpoint"]
    replay = manager.command_no_action(
        _args(
            record_root=str(tmp_path),
            receipt_id="no-action-1",
            reason="readiness blocked",
            expected_pointer_sha=result["pointer_sha256"],
            generation_id="unused-replay",
            published_at="2026-08-10T00:01:00Z",
        )
    )
    assert replay["idempotent"] is True
    assert replay["pointer_sha256"] == result["pointer_sha256"]
    assert len(replay["catalog"]["receipts"]) == len(result["catalog"]["receipts"])
    monkeypatch.setattr(manager, "NO_ACTION_RECEIPT_MAX_BYTES", 200)
    with pytest.raises(StrategyRecordStoreError, match="receipt exceeds"):
        manager.command_no_action(
            _args(
                record_root=str(tmp_path),
                receipt_id="no-action-2",
                reason="x" * 500,
                expected_pointer_sha=result["pointer_sha256"],
                generation_id="g3",
            )
        )


def test_consecutive_no_actions_preserve_history_state_binding(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    root = project / ("results/strategy_records/CN/aggressive_tech_manufacturing")
    record_ids = ("20260809_1000", "20260810_1000")
    for record_id in record_ids:
        (root / record_id).mkdir(parents=True)
    empty_inventory_sha = hashlib.sha256(b"[]\n").hexdigest()
    records = [
        {
            "record_id": record_id,
            "relative_path": record_id,
            "state": "ONLINE",
            "storage_state": "ONLINE",
            "sealed_at": "2026-08-10T00:00:00Z",
            "inventory": [],
            "inventory_sha256": empty_inventory_sha,
            "file_count": 0,
            "total_bytes": 0,
        }
        for record_id in record_ids
    ]
    projection = {
        "valid_records": [],
        "rejected": [],
        "latest_seen": None,
        "historical_records": [],
        "historical_rejected": [],
    }
    registry = {
        "schema_version": "cn_aggressive_dashboard_history_integrity.v2",
        "market": "CN",
        "strategy_label": "aggressive_tech_manufacturing",
        "generated_at": "2026-08-10T00:00:00Z",
        "authority": "DASHBOARD_POST_HOC_INTEGRITY_DECLARATION",
        "intended_generation_id": "g-history",
        "dashboard_projection_sha256": hashlib.sha256(canonical_json_bytes(projection)).hexdigest(),
        "record_count": 0,
        "records": [],
    }
    registry["content_sha256"] = content_sha256(registry)
    registry_path = project / "results/history-integrity/g-history.v2.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_bytes(canonical_json_bytes(registry))
    registry_ref = {
        "path": registry_path.relative_to(project).as_posix(),
        "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    }
    initial = bootstrap_catalog(
        root,
        records=records,
        dashboard_projection=projection,
        active_record_id=record_ids[1],
        previous_record_id=record_ids[0],
        generation_id="g-history",
        published_at="2026-08-10T00:00:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
        history_registry=registry,
        history_registry_ref=registry_ref,
    )
    registry_bytes = registry_path.read_bytes()
    first = manager.command_no_action(
        _args(
            record_root=str(root),
            receipt_id="no-action-1",
            reason="metadata only",
            expected_pointer_sha=initial["pointer_sha256"],
            generation_id="g-noop-1",
            published_at="2026-08-10T00:01:00Z",
        )
    )
    second = manager.command_no_action(
        _args(
            record_root=str(root),
            receipt_id="no-action-2",
            reason="metadata only",
            expected_pointer_sha=first["pointer_sha256"],
            generation_id="g-noop-2",
            published_at="2026-08-10T00:02:00Z",
        )
    )

    assert first["pointer"]["generation_id"] == "g-noop-1"
    assert second["pointer"]["generation_id"] == "g-noop-2"
    assert second["catalog"]["generation_id"] == "g-noop-2"
    for key in (
        "active_record_id",
        "previous_record_id",
        "active_closure",
    ):
        assert second["pointer"][key] == initial["pointer"][key]
    for key in (
        "records",
        "dashboard_projection",
        "history_registry",
        "history_registry_ref",
    ):
        assert first["catalog"][key] == initial["catalog"][key]
        assert second["catalog"][key] == initial["catalog"][key]
    assert second["catalog"]["history_registry"]["intended_generation_id"] == "g-history"
    assert registry_path.read_bytes() == registry_bytes
    assert [row["receipt_id"] for row in second["catalog"]["receipts"]] == [
        "no-action-1",
        "no-action-2",
    ]


def test_restore_rejects_malicious_archive_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    restore = tmp_path / "restore"
    restore.mkdir()
    with pytest.raises(StrategyRecordStoreError, match="canonical relative"):
        manager._restore_tar(archive_path, restore)
    assert not (tmp_path / "escape").exists()

    symlink_archive = tmp_path / "symlink.tar"
    with tarfile.open(symlink_archive, "w") as archive:
        info = tarfile.TarInfo("record/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(StrategyRecordStoreError, match="member type"):
        manager._restore_tar(symlink_archive, restore)


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd executable unavailable")
def test_archive_rehearsal_is_copy_only_and_full_restore_verified(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    root.mkdir()
    _bootstrap(root)
    source = root / "20260809_1000" / "holdings.json"
    before = source.read_bytes()
    output = tmp_path / "archive"
    result = manager.command_archive_rehearsal(
        _args(record_root=str(root), before="2027", output_root=str(output))
    )
    assert Path(result["archive_path"]).is_file()
    assert source.read_bytes() == before
    assert result["source_records_preserved"] is True
    assert result["moved"] is False
    assert result["deleted"] is False
    assert load_registered_catalog(root) is not None


def test_main_inventory_json_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "legacy").mkdir()
    assert manager.main(["inventory", "--record-root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["registered"] is False


def test_quarantine_move_is_resume_safe_and_rollback_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "records"
    root.mkdir()
    rows = []
    for record_id in ("20260601_1000", "20260602_1000"):
        record_dir = root / record_id
        record_dir.mkdir()
        (record_dir / "payload").write_text(record_id, encoding="utf-8")
        rows.append(
            {
                "record_id": record_id,
                "relative_path": record_id,
                **manager.build_inventory(record_dir, enforce_new_record_budget=False),
            }
        )
    plan = {
        "transaction_id": "tx-resume",
        "record_root": str(root),
        "quarantine_root": str(tmp_path / "quarantine/tx-resume/records"),
        "record_ids": [row["record_id"] for row in rows],
        "source_catalog": {"records": rows},
    }
    real_rename = os.rename
    calls = 0

    def crash_on_second(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated crash")
        real_rename(source, target)

    monkeypatch.setattr(manager.os, "rename", crash_on_second)
    with pytest.raises(OSError, match="simulated crash"):
        manager._move_records(plan, direction="cutover")
    monkeypatch.setattr(manager.os, "rename", real_rename)
    assert manager._move_records(plan, direction="cutover") == 1
    assert all(not (root / row["record_id"]).exists() for row in rows)
    assert manager._move_records(plan, direction="rollback") == 2
    assert all((root / row["record_id"]).is_dir() for row in rows)


def test_quarantine_refuses_dual_or_lost_locations(tmp_path: Path) -> None:
    root = tmp_path / "records"
    quarantine = tmp_path / "quarantine"
    source = root / "20260601_1000"
    target = quarantine / "20260601_1000"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    row = {
        "record_id": source.name,
        "relative_path": source.name,
        **manager.build_inventory(source, enforce_new_record_budget=False),
    }
    with pytest.raises(StrategyRecordStoreError, match="dual"):
        manager._verify_location(root, quarantine, row)
    source.rmdir()
    target.rmdir()
    with pytest.raises(StrategyRecordStoreError, match="lost"):
        manager._verify_location(root, quarantine, row)


def test_archive_candidate_rejects_unregistered_strict_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results/strategy_records/CN/aggressive_tech_manufacturing"
    root.mkdir(parents=True)
    _bootstrap(root)
    (root / "20260810_1948").mkdir()
    args = _args(
        record_root=str(root),
        project_root=str(tmp_path),
        expected_pointer_sha=_pointer_sha(root),
        archive_manifest=[],
    )
    with pytest.raises(
        StrategyRecordStoreError,
        match=("unregistered strict record directory blocks archive candidate: " "20260810_1948"),
    ):
        manager.command_archive_candidate(args)


def _continuity_receipt_fixture() -> tuple[dict[str, object], dict[str, object], str]:
    closure: dict[str, object] = {
        "record_id": "20260820_1321",
        "relative_path": "20260820_1321",
        "inventory_sha256": "1" * 64,
        "total_bytes": 10,
        "file_count": 1,
        "manifest_path": "20260820_1321/manifest.json",
        "manifest_sha256": "2" * 64,
        "manual_manifest_path": "20260820_1321/manual_execution_manifest.json",
        "manual_manifest_sha256": "3" * 64,
        "ledger_path": "20260820_1321/ledger_after_manual_switch.parquet",
        "ledger_sha256": "4" * 64,
        "pnl_path": "20260820_1321/pnl_summary.csv",
        "pnl_sha256": "5" * 64,
        "financial_state_sha256": "6" * 64,
    }
    pointer: dict[str, object] = {
        "active_record_id": "20260820_1321",
        "active_closure": closure,
    }
    receipt: dict[str, object] = {
        "schema_id": manager.NO_ACTION_RECEIPT_SCHEMA,
        "receipt_id": "automation-20260821-daily-review-v1",
        "created_at": "2026-08-21T01:30:00Z",
        "status": "NO_ACTION",
        "active_record_id": "20260820_1321",
        "active_checkpoint": closure,
        "payload_copied": False,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
    }
    receipt["content_sha256"] = manager.content_sha256(receipt)
    catalog: dict[str, object] = {"receipts": [receipt]}
    return pointer, catalog, str(receipt["content_sha256"])


def test_seal_parser_exposes_governed_v3_continuity_bindings() -> None:
    parsed = manager.build_parser().parse_args(
        [
            "seal-publish",
            "--record-root",
            "/records",
            "--record-id",
            "20260821_1200",
            "--expected-pointer-sha",
            "a" * 64,
            "--expected-catalog-sha",
            "b" * 64,
            "--continuity-receipt-id",
            "automation-20260821-daily-review-v1",
            "--expected-continuity-receipt-sha",
            "c" * 64,
        ]
    )
    assert parsed.expected_catalog_sha == "b" * 64
    assert parsed.continuity_receipt_id == "automation-20260821-daily-review-v1"
    assert parsed.expected_continuity_receipt_sha == "c" * 64

    late = manager.build_parser().parse_args(
        [
            "seal-publish",
            "--record-root",
            "/records",
            "--record-id",
            "20260822_0930",
            "--expected-pointer-sha",
            "a" * 64,
            "--publication-class",
            manager.LATE_OFFICIAL_VALUATION_PUBLICATION,
            "--expected-valuation-date",
            "2026-08-21",
            "--expected-publication-date",
            "2026-08-22",
            "--publication-delay-reason",
            manager.LATE_PUBLICATION_REASON,
        ]
    )
    assert late.publication_class == manager.LATE_OFFICIAL_VALUATION_PUBLICATION
    assert late.expected_valuation_date == "2026-08-21"
    assert late.expected_publication_date == "2026-08-22"


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("missing", "continuity receipt must be unique"),
        ("duplicate", "continuity receipt must be unique"),
        ("hash", "continuity receipt content hash mismatch"),
        ("date", "continuity receipt date mismatch"),
        ("checkpoint", "continuity receipt active checkpoint mismatch"),
        ("source", "candidate source record is not active"),
    ],
)
def test_continuity_receipt_validation_fail_closed(mutation: str, pattern: str) -> None:
    pointer, catalog, receipt_sha = _continuity_receipt_fixture()
    receipt = catalog["receipts"][0]
    assert isinstance(receipt, dict)
    receipt_id = receipt["receipt_id"]
    candidate_date = "2026-08-21"
    source_record = "20260820_1321"
    if mutation == "missing":
        receipt_id = "missing"
    elif mutation == "duplicate":
        catalog["receipts"].append(dict(receipt))
    elif mutation == "hash":
        receipt["content_sha256"] = "f" * 64
    elif mutation == "date":
        receipt["created_at"] = "2026-08-20T01:30:00Z"
        receipt["content_sha256"] = manager.content_sha256(receipt)
        receipt_sha = str(receipt["content_sha256"])
    elif mutation == "checkpoint":
        receipt["active_checkpoint"] = {**receipt["active_checkpoint"], "file_count": 2}
        receipt["content_sha256"] = manager.content_sha256(receipt)
        receipt_sha = str(receipt["content_sha256"])
    elif mutation == "source":
        source_record = "20260819_1200"
    with pytest.raises(StrategyRecordStoreError, match=pattern):
        manager._find_and_validate_continuity_receipt(
            catalog=catalog,
            pointer=pointer,
            receipt_id=receipt_id,
            expected_sha=receipt_sha,
            candidate_date=candidate_date,
            source_record=source_record,
        )


def _late_validation_fixture() -> tuple[argparse.Namespace, dict, dict, dict, dict]:
    declaration = {
        "publication_class": manager.LATE_OFFICIAL_VALUATION_PUBLICATION,
        "expected_valuation_date": "2026-08-21",
        "expected_publication_date": "2026-08-22",
        "publication_delay_reason": manager.LATE_PUBLICATION_REASON,
    }
    recorded_at = "2026-08-22T09:30:00+08:00"
    checkpoint = {"record_id": "20260820_1321"}
    delay = {
        "schema_id": manager.PUBLICATION_DELAY_SCHEMA,
        **declaration,
        "evidence_date": "2026-08-21",
        "source_record": "20260820_1321",
        "continuity_receipt_id": "automation-20260821-daily-review-v1",
        "continuity_receipt_sha256": "a" * 64,
        "continuity_receipt_created_at": "2026-08-21T13:27:37Z",
        "continuity_checkpoint_digest": manager.content_sha256(checkpoint),
        "recorded_at_iso": recorded_at,
        "delay_days": 1,
        "historical_holdings_storage_authority": True,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
    }
    manifest = {
        **declaration,
        "recorded_at_iso": recorded_at,
        "publication_delay": delay,
    }
    manual = {
        **declaration,
        "recorded_at_iso": recorded_at,
        "publication_delay": dict(delay),
    }
    strict_record = {
        "data_date": "2026-08-21",
        "source_record": "20260820_1321",
    }
    receipt = {
        "receipt_id": "automation-20260821-daily-review-v1",
        "content_sha256": "a" * 64,
        "created_at": "2026-08-21T13:27:37Z",
        "active_checkpoint": checkpoint,
    }
    args = _args(
        publication_class=manager.LATE_OFFICIAL_VALUATION_PUBLICATION,
        expected_valuation_date="2026-08-21",
        expected_publication_date="2026-08-22",
        publication_delay_reason=manager.LATE_PUBLICATION_REASON,
        published_at=None,
    )
    return args, strict_record, manifest, manual, receipt


def test_late_publication_contract_uses_one_manager_instant() -> None:
    args, strict_record, manifest, manual, receipt = _late_validation_fixture()
    delay = manager._validate_late_publication(
        args=args,
        record_id="20260822_0930",
        sealed_at="2026-08-22T01:31:00Z",
        strict_record=strict_record,
        manifest=manifest,
        manual=manual,
        receipt=receipt,
    )
    assert delay["publication_class"] == manager.LATE_OFFICIAL_VALUATION_PUBLICATION
    assert delay["delay_days"] == 1
    assert delay["actual_sealed_at"] == delay["actual_published_at"]
    assert delay["actual_publication_local_date"] == "2026-08-22"
    assert delay["v17_mainline_authority"] is False
    assert delay["broker_order_trade_authority"] is False


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("clock_date", "manager local date mismatch"),
        ("timezone", "timezone is missing"),
        ("reason", "late publication reason mismatch"),
        ("class", "publication class declarations conflict"),
        ("multi_day", "delay must be exactly one day"),
        ("record_id", "record id does not match"),
        ("ordering", "timestamp ordering"),
    ],
)
def test_late_publication_contract_rejects_invalid_timing_and_declarations(
    mutation: str, pattern: str
) -> None:
    args, strict_record, manifest, manual, receipt = _late_validation_fixture()
    record_id = "20260822_0930"
    sealed_at = "2026-08-22T01:31:00Z"
    if mutation == "clock_date":
        sealed_at = "2026-08-23T01:31:00Z"
    elif mutation == "timezone":
        manifest["recorded_at_iso"] = manual["recorded_at_iso"] = "2026-08-22T09:30:00"
        manifest["publication_delay"]["recorded_at_iso"] = "2026-08-22T09:30:00"
        manual["publication_delay"]["recorded_at_iso"] = "2026-08-22T09:30:00"
    elif mutation == "reason":
        args.publication_delay_reason = "OTHER"
    elif mutation == "class":
        manifest["publication_class"] = "OFFICIAL_FINANCIAL_STATE"
    elif mutation == "multi_day":
        args.expected_valuation_date = "2026-08-20"
        manifest["expected_valuation_date"] = "2026-08-20"
        manual["expected_valuation_date"] = "2026-08-20"
        manifest["publication_delay"]["expected_valuation_date"] = "2026-08-20"
        manual["publication_delay"]["expected_valuation_date"] = "2026-08-20"
    elif mutation == "record_id":
        record_id = "20260822_0931"
    elif mutation == "ordering":
        receipt["created_at"] = "2026-08-22T01:30:30Z"
        manifest["publication_delay"]["continuity_receipt_created_at"] = "2026-08-22T01:30:30Z"
        manual["publication_delay"]["continuity_receipt_created_at"] = "2026-08-22T01:30:30Z"
    with pytest.raises(StrategyRecordStoreError, match=pattern):
        manager._validate_late_publication(
            args=args,
            record_id=record_id,
            sealed_at=sealed_at,
            strict_record=strict_record,
            manifest=manifest,
            manual=manual,
            receipt=receipt,
        )


def test_late_publication_freshness_rejects_existing_final_and_generations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    stage = root / "_record_store/staging/20260822_0930"
    stage.mkdir(parents=True)
    manager._validate_late_freshness(
        root=root,
        catalog={"records": []},
        stage=stage,
        target=root / "20260822_0930",
        record_id="20260822_0930",
        generation_id="g-late",
        performance_generation_id="p-late",
    )
    (root / "20260822_0930").mkdir()
    with pytest.raises(StrategyRecordConflict, match="fresh final"):
        manager._validate_late_freshness(
            root=root,
            catalog={"records": []},
            stage=stage,
            target=root / "20260822_0930",
            record_id="20260822_0930",
            generation_id="g-late",
            performance_generation_id="p-late",
        )


def test_manager_clock_is_aware_utc() -> None:
    observed = manager._manager_utc_now()
    assert observed.tzinfo is not None
    assert observed.utcoffset() == timezone.utc.utcoffset(datetime.now(timezone.utc))


def test_late_publication_rejects_caller_published_at_before_stage_adoption(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    root, initial = _initial_v3_store(project)
    with pytest.raises(StrategyRecordStoreError, match="rejects caller-supplied"):
        manager.command_seal_publish(
            _args(
                record_root=str(root),
                record_id="20260822_0930",
                expected_pointer_sha=initial["pointer_sha256"],
                publication_class=manager.LATE_OFFICIAL_VALUATION_PUBLICATION,
                published_at="2026-08-22T01:31:00Z",
            )
        )
    assert not (root / "20260822_0930").exists()
