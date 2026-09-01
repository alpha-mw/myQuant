from __future__ import annotations

import builtins
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from quant_investor.strategy_records.performance import (
    MAX_PERFORMANCE_JSON_BYTES,
    PERFORMANCE_INITIAL_CAPITAL,
    UNIT_QUANTUM,
    apply_flow_neutral_unitization,
    build_manifest,
    build_owner_declaration,
    build_performance_history_ref,
    build_seed_rows,
    decimal_text,
    extend_performance_rows,
    immutable_write,
    load_performance_history,
    normalize_registered_projection,
    seal_semantic,
    validate_cash_flow_artifact,
    validate_lineage_index,
    write_deterministic_parquet,
)
from quant_investor.strategy_records.store import (
    CATALOG_SCHEMA_V2,
    CATALOG_SCHEMA_V3,
    StrategyRecordStoreError,
    bootstrap_catalog,
    canonical_json_bytes,
    load_registered_catalog,
    publish_catalog,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _projection_catalog() -> dict:
    rows = [
        {
            "record": "20260101_1000",
            "valuation_date": "2026-01-01",
            "accounting": {
                "cash_after": 1_000_000,
                "market_value_after": 0,
                "total_value_after": 1_000_000,
                "portfolio_pnl_after": 0,
            },
            "capital_base": 1_000_000,
            "funding": None,
            "funding_correction": None,
            "evidence_status": "LEGACY",
        },
        {
            "record": "20260102_1000",
            "valuation_date": "2026-01-02",
            "accounting": {
                "cash_after": 1_110_000,
                "market_value_after": 0,
                "total_value_after": 1_110_000,
                "portfolio_pnl_after": 10_000,
            },
            "capital_base": 1_100_000,
            "funding": {"amount": 100_000},
            "funding_correction": None,
            "evidence_status": "LEGACY",
        },
        {
            "record": "20260103_1000",
            "valuation_date": "2026-01-03",
            "accounting": {
                "cash_after": 1_020_000,
                "market_value_after": 0,
                "total_value_after": 1_020_000,
                "portfolio_pnl_after": 20_000,
            },
            "capital_base": 1_000_000,
            "funding": None,
            "funding_correction": {
                "reversed_record": "20260102_1000",
                "reversed_amount": 100_000,
                "initial_capital": 1_000_000,
            },
            "evidence_status": "CURRENT",
        },
    ]
    return {
        "dashboard_projection": {"historical_records": rows},
        "records": [{"record_id": row["record"]} for row in rows[:-1]]
        + [
            {
                "record_id": rows[-1]["record"],
                "manual_manifest_sha256": SHA_C,
                "ledger_sha256": SHA_D,
                "financial_state_sha256": SHA_E,
            }
        ],
    }


def test_registered_projection_seed_reproduces_owner_correction() -> None:
    catalog = _projection_catalog()
    normalized, projection_sha, normalized_sha = normalize_registered_projection(catalog)
    rows = build_seed_rows(normalized, catalog=catalog)

    assert len(rows) == 3
    assert projection_sha != normalized_sha
    assert rows[0]["adjusted_nav_cny"] == PERFORMANCE_INITIAL_CAPITAL
    assert rows[1]["excluded_external_flow_cny"] == Decimal("100000.0000")
    assert rows[1]["adjusted_nav_cny"] == Decimal("1010000.0000")
    assert rows[2]["excluded_external_flow_cny"] == Decimal("0.0000")
    assert rows[2]["adjusted_nav_cny"] == Decimal("1020000.0000")
    assert rows[2]["cumulative_return"] == Decimal("0.020000000000")
    assert rows[2]["manual_manifest_sha256"] == SHA_C
    assert rows[2]["ledger_parquet_sha256"] == SHA_D
    assert rows[2]["financial_state_sha256"] == SHA_E


def test_migration_adapter_is_pure_and_never_consults_logical_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("migration adapter attempted external I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(tarfile, "open", forbidden)

    catalog = _projection_catalog()
    for row in catalog["dashboard_projection"]["historical_records"]:
        row["source_refs"] = [{"path": "forbidden/ledger.csv", "sha256": "0" * 64}]
    normalized, _, _ = normalize_registered_projection(catalog)

    assert len(normalized) == 3
    assert all("source_refs" not in row for row in normalized)


def test_performance_parquet_is_decimal_and_deterministic(tmp_path: Path) -> None:
    catalog = _projection_catalog()
    normalized, _, _ = normalize_registered_projection(catalog)
    rows = build_seed_rows(normalized, catalog=catalog)
    first = tmp_path / "first.parquet"
    replay = tmp_path / "replay.parquet"
    digest, size = write_deterministic_parquet(rows, first, replay_path=replay)

    assert first.read_bytes() == replay.read_bytes()
    assert digest == hashlib.sha256(first.read_bytes()).hexdigest()
    assert size == first.stat().st_size
    assert b"ledger.csv" not in first.read_bytes().lower()


def _lineage() -> list[dict]:
    base = {
        "execution_class": "NO_TRADE",
        "publication_class": "OFFICIAL_FINANCIAL_STATE",
        "storage_state": "ONLINE",
        "manifest_ref": None,
        "manual_manifest_ref": None,
        "effective_ledger_ref": None,
        "financial_state_sha256": None,
        "ledger_parquet_sha256": None,
    }
    return [
        {
            **base,
            "record_id": "20260101_1000",
            "source_record_id": None,
            "supersedes_record_id": None,
            "valuation_date": "2026-01-01",
        },
        {
            **base,
            "record_id": "20260102_1000",
            "source_record_id": "20260101_1000",
            "supersedes_record_id": None,
            "valuation_date": "2026-01-02",
        },
    ]


def test_lineage_rejects_cycle_and_fork() -> None:
    lineage = _lineage()
    assert validate_lineage_index(lineage, active_record_id="20260102_1000") == (
        "20260101_1000",
        "20260102_1000",
    )

    cycle = [dict(row) for row in lineage]
    cycle[0]["source_record_id"] = "20260102_1000"
    with pytest.raises(StrategyRecordStoreError, match="cycle"):
        validate_lineage_index(cycle, active_record_id="20260102_1000")

    fork = [dict(row) for row in lineage]
    fork.append(
        {
            **lineage[1],
            "record_id": "20260103_1000",
            "valuation_date": "2026-01-03",
        }
    )
    with pytest.raises(StrategyRecordStoreError, match="fork"):
        validate_lineage_index(fork, active_record_id="20260103_1000")


def test_lineage_accepts_typed_late_official_valuation_publication() -> None:
    lineage = _lineage()
    lineage[-1]["publication_class"] = "LATE_OFFICIAL_VALUATION_PUBLICATION"
    assert validate_lineage_index(lineage, active_record_id="20260102_1000") == (
        "20260101_1000",
        "20260102_1000",
    )


def test_flow_neutral_unitization_and_cash_flow_proof() -> None:
    pre = {
        "record_id": "20260101_1000",
        "raw_nav_cny": Decimal("1000000.0000"),
        "financial_state_sha256": SHA_A,
    }
    post = {
        "record_id": "20260101_1001",
        "raw_nav_cny": Decimal("1100000.0000"),
        "financial_state_sha256": SHA_B,
        "manual_manifest_path": "records/manual.json",
        "manual_manifest_sha256": SHA_C,
    }
    declaration = seal_semantic(
        {
            "schema_id": "myquant.strategy_performance_cash_flow.v1",
            "event_id": "flow-1",
            "historical_label": "aggressive_tech_manufacturing",
            "canonical_strategy_id": "cn-aggressive-tech-manufacturing",
            "effective_at": "2026-01-01T02:01:00Z",
            "shanghai_trade_date": "2026-01-01",
            "direction": "CONTRIBUTION",
            "amount_cny": "100000.0000",
            "timing_convention": "between_exact_pre_and_post_financial_states",
            "pre_flow_record_id": pre["record_id"],
            "pre_flow_nav_cny": "1000000.0000",
            "pre_flow_financial_state_sha256": SHA_A,
            "post_flow_record_id": post["record_id"],
            "post_flow_nav_cny": "1100000.0000",
            "post_flow_financial_state_sha256": SHA_B,
            "matching_manual_manifest_path": "records/manual.json",
            "matching_manual_manifest_sha256": SHA_C,
            "declared_by": "maxwell",
            "declared_at": "2026-01-01T02:02:00Z",
            "authority_kind": "owner_declaration",
            "v17_activation_authority": False,
            "broker_authority": False,
            "order_authority": False,
            "execution_authority": False,
            "trade_authority": False,
        }
    )
    amount = validate_cash_flow_artifact(
        declaration,
        pre_row=pre,
        post_row=post,
        pre_positions={"000001.SZ": 100},
        post_positions={"000001.SZ": 100},
    )
    units, unit_nav = apply_flow_neutral_unitization(
        pre_nav=Decimal("1000000.0000"),
        pre_units=Decimal("1000000.000000000000"),
        pre_unit_nav=Decimal("1.000000000000"),
        amount=amount,
    )
    assert units == Decimal("1100000.000000000000")
    assert unit_nav == Decimal("1.000000000000")


def test_extension_appends_new_valuation_and_rejects_same_date() -> None:
    catalog = _projection_catalog()
    normalized, _, _ = normalize_registered_projection(catalog)
    rows = build_seed_rows(normalized, catalog=catalog)
    extended = extend_performance_rows(
        rows,
        strict_record={
            "record": "20260104_1000",
            "data_date": "2026-01-04",
            "execution_kind": "carry_forward",
            "execution_status": "no_action_carry_forward_official_valuation",
            "accounting": {
                "cash_after": 1_030_000,
                "market_value_after": 0,
                "total_value_after": 1_030_000,
                "portfolio_pnl_after": 30_000,
            },
        },
        manual_manifest_sha256=SHA_A,
        ledger_parquet_sha256=SHA_B,
        financial_state_sha256=SHA_C,
    )
    assert extended[-1]["cumulative_return"] == Decimal("0.030000000000")

    with pytest.raises(StrategyRecordStoreError, match="SAME_DATE"):
        extend_performance_rows(
            rows,
            strict_record={
                "record": "20260103_1100",
                "data_date": "2026-01-03",
                "execution_kind": "carry_forward",
                "execution_status": "no_action_carry_forward",
                "accounting": {
                    "cash_after": 1_020_000,
                    "market_value_after": 0,
                    "total_value_after": 1_020_000,
                    "portfolio_pnl_after": 20_000,
                },
            },
            manual_manifest_sha256=SHA_A,
            ledger_parquet_sha256=SHA_B,
            financial_state_sha256=SHA_C,
        )


def test_batch_extension_uses_each_historical_market_close_as_valuation_time() -> None:
    catalog = _projection_catalog()
    normalized, _, _ = normalize_registered_projection(catalog)
    rows = build_seed_rows(normalized, catalog=catalog)

    for ordinal, valuation_date, nav in (
        (1, "2026-01-04", 1_030_000),
        (2, "2026-01-05", 1_040_000),
    ):
        rows = extend_performance_rows(
            rows,
            strict_record={
                "record": f"20260901_115917-b{ordinal:02d}",
                "data_date": valuation_date,
                "execution_kind": "carry_forward",
                "execution_status": "no_action_carry_forward_official_valuation",
                "accounting": {
                    "cash_after": nav,
                    "market_value_after": 0,
                    "total_value_after": nav,
                    "portfolio_pnl_after": nav - 1_000_000,
                },
            },
            manual_manifest_sha256=SHA_A,
            ledger_parquet_sha256=SHA_B,
            financial_state_sha256=SHA_C,
        )

    assert rows[-2]["valuation_at"] == "2026-01-04T07:00:00Z"
    assert rows[-1]["valuation_at"] == "2026-01-05T07:00:00Z"


def test_extension_carries_external_flow_without_counting_it_as_return() -> None:
    catalog = _projection_catalog()
    normalized, _, _ = normalize_registered_projection(catalog)
    rows = build_seed_rows(normalized, catalog=catalog)
    previous = rows[-1]
    amount = Decimal("100000.0000")
    post_units, post_unit_nav = apply_flow_neutral_unitization(
        pre_nav=previous["raw_nav_cny"],
        pre_units=previous["unit_count"],
        pre_unit_nav=previous["unit_nav"],
        amount=amount,
    )
    extended = extend_performance_rows(
        rows,
        strict_record={
            "record": "20260104_1000",
            "data_date": "2026-01-04",
            "execution_kind": "carry_forward",
            "execution_status": "no_action_carry_forward_official_valuation",
            "accounting": {
                "cash_after": 1_120_000,
                "market_value_after": 0,
                "total_value_after": 1_120_000,
                "portfolio_pnl_after": 20_000,
            },
        },
        manual_manifest_sha256=SHA_A,
        ledger_parquet_sha256=SHA_B,
        financial_state_sha256=SHA_C,
        post_flow_unit_count=post_units,
        external_flow_amount=amount,
    )

    assert extended[-1]["unit_nav"] == post_unit_nav
    assert extended[-1]["interval_return"] == Decimal("0.000000000000")
    assert extended[-1]["cumulative_return"] == Decimal("0.020000000000")
    assert extended[-1]["excluded_external_flow_cny"] == amount
    assert extended[-1]["adjusted_nav_cny"] == Decimal("1020000.0000")


def _store_record(record_id: str, *, last: bool) -> dict:
    return {
        "record_id": record_id,
        "relative_path": record_id,
        "state": "ONLINE",
        "storage_state": "ONLINE",
        "sealed_at": "2026-01-02T02:00:00Z",
        "inventory": [],
        "inventory_sha256": hashlib.sha256(b"[]\n").hexdigest(),
        "file_count": 0,
        "total_bytes": 0,
        "manifest_path": f"{record_id}/manifest.json",
        "manifest_sha256": SHA_A,
        "manual_manifest_path": f"{record_id}/manual_execution_manifest.json",
        "manual_manifest_sha256": SHA_C if last else SHA_A,
        "ledger_path": f"{record_id}/ledger_after_manual_switch.parquet",
        "ledger_sha256": SHA_D if last else SHA_A,
        "financial_state_sha256": SHA_E if last else SHA_A,
    }


def test_catalog_v3_binds_and_reads_performance_closure(tmp_path: Path) -> None:
    root = tmp_path / "records"
    for record_id in ("20260101_1000", "20260102_1000"):
        (root / record_id).mkdir(parents=True)
    records = [
        _store_record("20260101_1000", last=False),
        _store_record("20260102_1000", last=True),
    ]
    initial = bootstrap_catalog(
        root,
        records=records,
        active_record_id="20260102_1000",
        previous_record_id="20260101_1000",
        generation_id="g-v2",
        published_at="2026-01-02T02:00:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    rows = [
        {
            "sequence_no": 1,
            "record_id": "20260101_1000",
            "valuation_at": "2026-01-01T02:00:00Z",
            "valuation_date": "2026-01-01",
            "cash_cny": Decimal("1000000.0000"),
            "equity_market_value_cny": Decimal("0.0000"),
            "raw_nav_cny": Decimal("1000000.0000"),
            "portfolio_pnl_cny": Decimal("0.0000"),
            "excluded_external_flow_cny": Decimal("0.0000"),
            "adjusted_nav_cny": Decimal("1000000.0000"),
            "unit_count": Decimal("1000000.000000000000"),
            "unit_nav": Decimal("1.000000000000"),
            "interval_return": Decimal("0.000000000000"),
            "cumulative_return": Decimal("0.000000000000"),
            "drawdown": Decimal("0.000000000000"),
            "evidence_kind": "OWNER_DECLARED_REGISTERED_PROJECTION_MIGRATION",
            "manual_manifest_sha256": SHA_A,
            "ledger_parquet_sha256": SHA_A,
            "financial_state_sha256": SHA_A,
        },
        {
            "sequence_no": 2,
            "record_id": "20260102_1000",
            "valuation_at": "2026-01-02T02:00:00Z",
            "valuation_date": "2026-01-02",
            "cash_cny": Decimal("1010000.0000"),
            "equity_market_value_cny": Decimal("0.0000"),
            "raw_nav_cny": Decimal("1010000.0000"),
            "portfolio_pnl_cny": Decimal("10000.0000"),
            "excluded_external_flow_cny": Decimal("0.0000"),
            "adjusted_nav_cny": Decimal("1010000.0000"),
            "unit_count": Decimal("1000000.000000000000"),
            "unit_nav": Decimal("1.010000000000"),
            "interval_return": Decimal("0.010000000000"),
            "cumulative_return": Decimal("0.010000000000"),
            "drawdown": Decimal("0.000000000000"),
            "evidence_kind": "OWNER_DECLARED_REGISTERED_PROJECTION_MIGRATION",
            "manual_manifest_sha256": SHA_C,
            "ledger_parquet_sha256": SHA_D,
            "financial_state_sha256": SHA_E,
        },
    ]
    generation = "p1"
    prefix = root / "_record_store/performance" / generation
    prefix.mkdir(parents=True)
    series_path = prefix / "series.parquet"
    series_sha, series_bytes = write_deterministic_parquet(rows, series_path)
    owner = build_owner_declaration(
        performance_generation_id=generation,
        declared_at="2026-01-02T02:01:00Z",
        series_path=f"_record_store/performance/{generation}/series.parquet",
        series_sha256=series_sha,
        series_bytes=series_bytes,
        source_pointer_sha256=initial["pointer_sha256"],
        source_catalog_sha256=initial["pointer"]["catalog_sha256"],
        normalized_projection_semantic_sha256=SHA_A,
    )
    owner_raw = canonical_json_bytes(owner)
    owner_sha = immutable_write(
        prefix / "owner_declaration.v1.json",
        owner_raw,
        max_bytes=MAX_PERFORMANCE_JSON_BYTES,
    )
    manifest = build_manifest(
        performance_generation_id=generation,
        generated_at="2026-01-02T02:00:00Z",
        identity_path="governance/identity.json",
        identity_sha256=SHA_A,
        parent_performance_manifest_sha256=None,
        source_pointer_sha256=initial["pointer_sha256"],
        source_catalog_generation_id="g-v2",
        source_catalog_sha256=initial["pointer"]["catalog_sha256"],
        dashboard_projection_sha256=SHA_A,
        normalized_projection_semantic_sha256=SHA_A,
        series_path=f"_record_store/performance/{generation}/series.parquet",
        series_sha256=series_sha,
        series_bytes=series_bytes,
        owner_path=(f"_record_store/performance/{generation}/owner_declaration.v1.json"),
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
    ref = build_performance_history_ref(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        manifest_bytes=len(manifest_raw),
    )
    lineage = _lineage()
    for index, record in enumerate(records):
        lineage[index]["manifest_ref"] = {
            "path": record["manifest_path"],
            "sha256": record["manifest_sha256"],
        }
        lineage[index]["manual_manifest_ref"] = {
            "path": record["manual_manifest_path"],
            "sha256": record["manual_manifest_sha256"],
        }
        lineage[index]["effective_ledger_ref"] = {
            "path": record["ledger_path"],
            "sha256": record["ledger_sha256"],
        }
        lineage[index]["financial_state_sha256"] = record["financial_state_sha256"]
        lineage[index]["ledger_parquet_sha256"] = record["ledger_sha256"]
    result = publish_catalog(
        root,
        expected_pointer_sha256=initial["pointer_sha256"],
        records=records,
        active_record_id="20260102_1000",
        previous_record_id="20260101_1000",
        generation_id="g-v3",
        published_at="2026-01-02T02:02:00Z",
        catalog_schema=CATALOG_SCHEMA_V3,
        inherit_history_registry=False,
        lineage_index=lineage,
        performance_history_ref=ref,
    )

    pointer, catalog = load_registered_catalog(root) or ({}, {})
    assert result["pointer"] == pointer
    assert catalog["performance_contract_ready"] is True
    assert "dashboard_projection" not in catalog
    assert load_performance_history(root, ref)["rows"][-1]["record_id"] == ("20260102_1000")
    assert (
        decimal_text(
            load_performance_history(root, ref)["rows"][-1]["cumulative_return"],
            quantum=UNIT_QUANTUM,
        )
        == "0.010000000000"
    )
