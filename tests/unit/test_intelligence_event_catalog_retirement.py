from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.retire_event_score_catalog_residual as residual
from quant_investor.market.market_data_reader import MarketDataReader
from tests.fixtures.strict_cn_snapshot import coverage_v4, v4_snapshot_paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict:
    repo_root = tmp_path / "repo"
    market_root = repo_root / "data" / "parquet" / "cn"
    catalog_path = market_root / "_catalog.json"
    pointer_path = market_root / "_latest.json"
    source_path = market_root / "event_daily_score" / "part.parquet"
    source_path.parent.mkdir(parents=True)
    source_frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "trade_date": ["20260714", "20260714"],
            "money_flow_score": [0.2, 0.3],
            "top_list_score": [0.1, 0.4],
            "rotation_score": [0.5, 0.6],
            "breadth_score": [0.7, 0.8],
            "event_risk_score": [0.9, 0.2],
            "sentiment_score": [0.3, 0.4],
            "intelligence_score": [0.55, 0.65],
            "source": ["fixture", "fixture"],
            "source_priority": ["primary", "primary"],
            "pit_status": ["pit", "pit"],
            "source_snapshot_id": ["fixture", "fixture"],
            "transform_version": ["v14", "v14"],
            "fetched_at": ["2026-07-15T00:00:00Z"] * 2,
        }
    )
    source_frame.to_parquet(source_path, index=False)
    source_sha = _sha256(source_path)
    catalog = {
        "schema_version": "strict-parquet-catalog.v1",
        "required_tables": ["event_daily_score"],
        "tables": {
            "event_daily_score": {
                "logical_table": "event_daily_score",
                "path": "event_daily_score/part.parquet",
                "table_root": "event_daily_score",
                "date_column": "trade_date",
                "key_columns": ["ts_code", "trade_date"],
                "columns": list(source_frame.columns),
                "row_count": len(source_frame),
                "sha256": source_sha,
                "size_bytes": source_path.stat().st_size,
                "status": "ok",
            }
        },
    }
    _write_json(catalog_path, catalog)

    bars_root, serving_root, manifest_path = v4_snapshot_paths(
        repo_root / "data",
        "fixture",
    )
    bars_root.mkdir(parents=True)
    (bars_root / "bars.parquet").write_bytes(b"snapshot probe")
    serving_symbol = serving_root / "symbol=000001.SZ"
    serving_symbol.mkdir(parents=True)
    (serving_symbol / "bars.parquet").write_bytes(b"snapshot probe")
    coverage = coverage_v4(
        repo_root / "data",
        ["000001.SZ"],
        trade_date="20260714",
    )
    _write_json(manifest_path, {"snapshot_id": "fixture", "coverage": coverage})
    _write_json(
        pointer_path,
        {
            "status": "OK",
            "snapshot_id": "fixture",
            "latest_complete_trade_date": "20260714",
            "latest_trade_date": "20260714",
            "table_root": str(bars_root),
            "derived_serving_root": str(serving_root),
            "manifest_path": str(manifest_path),
            "coverage": coverage,
            "blockers": [],
        },
    )
    return {
        "repo_root": repo_root,
        "catalog_path": catalog_path,
        "market_pointer_path": pointer_path,
        "source_table_path": source_path,
        "generation_id": "event-schema-v14-fixture-0001",
        "expected_catalog_sha256": _sha256(catalog_path),
        "expected_market_pointer_sha256": _sha256(pointer_path),
        "expected_source_table_sha256": source_sha,
        "apply": False,
        "confirm_token": None,
    }


def _apply(fixture: dict) -> dict:
    return residual.retire_event_score_catalog_residual(
        **{
            **fixture,
            "apply": True,
            "confirm_token": residual.CONFIRM_TOKEN,
        }
    )


def test_dry_run_is_repo_read_only_and_requires_exact_residual(tmp_path):
    fixture = _fixture(tmp_path)
    before = {
        "catalog": fixture["catalog_path"].read_bytes(),
        "pointer": fixture["market_pointer_path"].read_bytes(),
        "source": fixture["source_table_path"].read_bytes(),
    }

    report = residual.retire_event_score_catalog_residual(**fixture)

    assert report["status"] == "would_retire_event_score_residual"
    assert report["repository_writes"] == []
    assert report["catalog_residuals"] == [
        {
            "path": "tables.event_daily_score.columns[8]",
            "value": "intelligence_score",
            "kind": "value",
        }
    ]
    assert not Path(report["generation_path"]).exists()
    assert not Path(report["transaction_path"]).exists()
    assert fixture["catalog_path"].read_bytes() == before["catalog"]
    assert fixture["market_pointer_path"].read_bytes() == before["pointer"]
    assert fixture["source_table_path"].read_bytes() == before["source"]


def test_apply_writes_new_generation_and_commits_bound_wal(tmp_path):
    fixture = _fixture(tmp_path)
    old_source = fixture["source_table_path"].read_bytes()
    old_pointer = fixture["market_pointer_path"].read_bytes()

    report = _apply(fixture)

    assert report["status"] == "retired_event_score_residual"
    assert report["fresh_reader_probe"]["passed"] is True
    assert fixture["source_table_path"].read_bytes() == old_source
    assert fixture["market_pointer_path"].read_bytes() == old_pointer
    catalog = json.loads(fixture["catalog_path"].read_text(encoding="utf-8"))
    assert "intelligence" not in json.dumps(catalog, ensure_ascii=False).lower()
    entry = catalog["tables"]["event_daily_score"]
    assert entry["sha256"] == report["new_generation_sha256"]
    assert entry["generation_manifest_sha256"] == report["generation_manifest_sha256"]
    assert entry["path"].startswith("event_daily_score/_generations/event-schema-v14-fixture-0001/")
    generation = fixture["catalog_path"].parent / entry["path"]
    assert _sha256(generation) == entry["sha256"]
    journal_path = (
        fixture["catalog_path"].parent
        / "event_daily_score"
        / "_transactions"
        / fixture["generation_id"]
        / "journal.json"
    )
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["state"] == "committed"
    assert journal["old_catalog_sha256"] == fixture["expected_catalog_sha256"]
    assert journal["new_catalog_sha256"] == report["output_catalog_sha256"]
    assert journal["expected_market_pointer_sha256"] == fixture["expected_market_pointer_sha256"]
    assert journal["new_generation_sha256"] == report["new_generation_sha256"]
    fresh = MarketDataReader(
        market="CN",
        data_root=fixture["repo_root"] / "data",
        mode_policy="strict",
    ).read_table("event_daily_score")
    assert "intelligence_score" not in fresh.columns
    assert len(fresh) == 2

    repeated = _apply(fixture)
    assert repeated["status"] == "already_retired"
    assert repeated["recovery_attempted"] is True
    assert repeated["fresh_reader_probe"]["passed"] is True
    assert fixture["source_table_path"].read_bytes() == old_source


def test_failed_fresh_reader_probe_rolls_back_exact_catalog(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    old_catalog = fixture["catalog_path"].read_bytes()
    old_source = fixture["source_table_path"].read_bytes()
    monkeypatch.setattr(
        residual,
        "_fresh_reader_probe",
        lambda *_args, **_kwargs: (False, "injected probe failure"),
    )

    report = _apply(fixture)

    assert report["status"] == "rolled_back_post_switch_validation_failed"
    assert report["rollback_verified"] is True
    assert fixture["catalog_path"].read_bytes() == old_catalog
    assert fixture["source_table_path"].read_bytes() == old_source
    journal = json.loads(
        (
            fixture["catalog_path"].parent
            / "event_daily_score"
            / "_transactions"
            / fixture["generation_id"]
            / "journal.json"
        ).read_text(encoding="utf-8")
    )
    assert journal["state"] == "rolled_back"


def test_recovery_commits_after_crash_between_catalog_switch_and_wal_update(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    old_source = fixture["source_table_path"].read_bytes()
    real_transition = residual._journal_transition

    def crash_before_switched_record(*args, state: str, **kwargs):
        if state == "switched":
            raise SystemExit("injected crash")
        return real_transition(*args, state=state, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(residual, "_journal_transition", crash_before_switched_record)
        with pytest.raises(SystemExit, match="injected crash"):
            _apply(fixture)

    switched_catalog = json.loads(fixture["catalog_path"].read_text(encoding="utf-8"))
    assert "intelligence" not in json.dumps(switched_catalog, ensure_ascii=False).lower()
    journal_path = (
        fixture["catalog_path"].parent
        / "event_daily_score"
        / "_transactions"
        / fixture["generation_id"]
        / "journal.json"
    )
    assert json.loads(journal_path.read_text(encoding="utf-8"))["state"] == "prepared"

    report = _apply(fixture)

    assert report["status"] == "recovered_committed"
    assert report["recovery_attempted"] is True
    assert report["fresh_reader_probe"]["passed"] is True
    assert json.loads(journal_path.read_text(encoding="utf-8"))["state"] == "committed"
    assert fixture["source_table_path"].read_bytes() == old_source


def test_apply_requires_confirmation_and_does_not_create_transaction(tmp_path):
    fixture = _fixture(tmp_path)
    report = residual.retire_event_score_catalog_residual(
        **{**fixture, "apply": True, "confirm_token": "WRONG"}
    )

    assert report["status"] == "blocked_confirmation_required"
    assert not Path(report["transaction_path"]).exists()


def test_cli_requires_explicit_production_hashes_and_new_run_id():
    parser = residual._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-id", "event-schema-v14-missing-hashes"])

    args = parser.parse_args(
        [
            "--expected-catalog-sha256",
            "a" * 64,
            "--expected-market-pointer-sha256",
            "b" * 64,
            "--expected-source-table-sha256",
            "c" * 64,
            "--run-id",
            "event-schema-v14-explicit-0001",
        ]
    )
    assert args.expected_catalog_sha256 == "a" * 64
    assert args.expected_market_pointer_sha256 == "b" * 64
    assert args.expected_source_table_sha256 == "c" * 64
    assert args.run_id == "event-schema-v14-explicit-0001"
