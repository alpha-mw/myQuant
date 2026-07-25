from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market import market_data_store
from quant_investor.market.market_data_store import (
    MarketDataStore,
    run_storage_reactivate_snapshot,
)
from quant_investor.market.pit_universe import (
    LIST_STATUS_LISTED,
    PITUniverseRecord,
    PITUniverseStore,
)
from quant_investor.market import snapshot_recovery_binding


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_reactivation_fixture(
    tmp_path: Path,
    *,
    repository_layout: bool = False,
) -> dict[str, object]:
    data_root = tmp_path / "data" if repository_layout else tmp_path
    snapshot_id = "snapshot-good-v4"
    snapshot_root = data_root / "parquet" / "cn" / "_snapshots"
    table_root = snapshot_root / snapshot_id / "table" / "bars"
    serving_root = snapshot_root / snapshot_id / "serving" / "bars"
    table_path = table_root / "year=2026" / "month=01" / "part.parquet"
    serving_a = serving_root / "symbol=000001.SZ" / "bars.parquet"
    serving_b = serving_root / "symbol=000002.SZ" / "bars.parquet"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    serving_a.parent.mkdir(parents=True, exist_ok=True)
    serving_b.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260102",
                "close": 10.0,
                "adj_factor": 1.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260103",
                "close": 11.0,
                "adj_factor": 1.0,
            },
            {
                "ts_code": "000002.SZ",
                "trade_date": "20260103",
                "close": 21.0,
                "adj_factor": 1.0,
            },
        ]
    )
    table_frame = frame.copy()
    table_frame["symbol"] = table_frame["ts_code"]
    table_frame.to_parquet(table_path, index=False)
    frame.loc[frame["ts_code"].eq("000001.SZ")].to_parquet(
        serving_a,
        index=False,
    )
    frame.loc[frame["ts_code"].eq("000002.SZ")].to_parquet(
        serving_b,
        index=False,
    )

    pit_store = PITUniverseStore(
        root_dir=data_root / "parquet" / "cn" / "reference",
        raw_root=data_root / "pit_raw",
        compatibility_path=data_root / "pit_compat.json",
    )
    pit = pit_store.write_snapshot(
        raw_records=[
            PITUniverseRecord(
                symbol=symbol,
                source_list_status=LIST_STATUS_LISTED,
                list_date="20200101",
                observed_at="2026-01-03T10:00:00Z",
                source_run_id="reactivation-fixture",
            )
            for symbol in ("000001.SZ", "000002.SZ")
        ],
        observed_at="2026-01-03T10:00:00Z",
        source_run_id="reactivation-fixture",
    )
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "coverage_ratio": 1.0,
        "coverage_complete_count": 2,
        "expected_scope_count": 2,
        "observed_bar_count": 2,
        "blocking_incomplete_count": 0,
        "latest_available_trade_date": "20260103",
        "latest_complete_trade_date": "20260103",
        "coverage_trade_date": "20260103",
        "expected_scope_sha256": "a" * 64,
        "suspended_symbols": [],
        "inactive_symbols": [],
        "verified_nontrading_bak_daily_zero_symbols": [],
        "verified_terminal_delisting_symbols": [],
        "allowed_stale_symbols": [],
        "non_blocking_absent_symbols": [],
        "true_missing_symbols": [],
        "classification_sets_disjoint": True,
        "pit_membership_path": str(pit["canonical_path"]),
        "pit_membership_sha256": str(pit["canonical_sha256"]),
        "pit_generation_id": str(pit["generation_id"]),
        "pit_generation_manifest_path": str(pit["generation_manifest_path"]),
        "pit_generation_manifest_sha256": str(pit["generation_manifest_sha256"]),
        "daily_basic_coverage": {
            "status": "OK",
            "covered_count": 2,
            "expected_count": 2,
            "coverage_ratio": 1.0,
            "error": "",
        },
        "adj_factor_coverage": {
            "status": "OK",
            "covered_count": 2,
            "expected_count": 2,
            "coverage_ratio": 1.0,
            "error": "",
        },
    }
    manifest_path = snapshot_root / f"{snapshot_id}.json"
    if repository_layout:
        declared_manifest_path = f"data/parquet/cn/_snapshots/{snapshot_id}.json"
        declared_table_root = f"data/parquet/cn/_snapshots/{snapshot_id}/table/bars"
        declared_serving_root = f"data/parquet/cn/_snapshots/{snapshot_id}/serving/bars"
    else:
        declared_manifest_path = str(manifest_path)
        declared_table_root = str(table_root)
        declared_serving_root = str(serving_root)
    manifest = {
        "snapshot_id": snapshot_id,
        "market": "CN",
        "status": "OK",
        "source": "unit-test",
        "row_count": len(frame),
        "symbol_count": 2,
        "latest_trade_date": "20260103",
        "latest_available_trade_date": "20260103",
        "latest_complete_trade_date": "20260103",
        "table_root": declared_table_root,
        "derived_serving_root": declared_serving_root,
        "manifest_path": declared_manifest_path,
        "readback_validated": True,
        "parquet_size_bytes": table_path.stat().st_size,
        "quarantined_tail_dates": [],
        "coverage": coverage,
        "blockers": [],
        "metadata": {},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    latest_path = data_root / "parquet" / "cn" / "_latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_bytes(b'{"status":"BROKEN","snapshot_id":"damaged-current"}\n')
    return {
        "snapshot_id": snapshot_id,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "latest_path": latest_path,
        "latest_sha256": _sha256(latest_path),
        "table_path": table_path,
        "serving_a": serving_a,
        "serving_b": serving_b,
        "table_root": table_root,
        "serving_root": serving_root,
        "data_root": data_root,
    }


def _run(tmp_path: Path, fixture: dict[str, object], *, commit: bool = False):
    return run_storage_reactivate_snapshot(
        market="CN",
        snapshot_id=str(fixture["snapshot_id"]),
        expected_snapshot_manifest_sha256=str(fixture["manifest_sha256"]),
        expected_market_pointer_sha256=str(fixture["latest_sha256"]),
        acknowledge_trade_date="20260103",
        reason="operator-authorized immutable snapshot recovery",
        commit=commit,
        data_root=fixture.get("data_root", tmp_path),
    )


def test_cn_full_history_writer_is_retired_before_any_write(tmp_path: Path) -> None:
    store = MarketDataStore(market="CN", data_root=tmp_path)

    with pytest.raises(
        ValueError,
        match="cn_full_history_writer_retired_use_parquet_direct",
    ):
        store.write_full_history_bars(
            pd.DataFrame(),
            source="must-not-write",
        )

    assert not (tmp_path / "parquet" / "cn" / "_latest.json").exists()


def test_reactivate_dry_run_is_zero_write(tmp_path: Path) -> None:
    fixture = _write_reactivation_fixture(tmp_path)
    latest_path = Path(fixture["latest_path"])
    before_pointer = latest_path.read_bytes()
    before_manifest = Path(fixture["manifest_path"]).read_bytes()

    result = _run(tmp_path, fixture)

    assert result["status"] == "validated_dry_run"
    assert result["commit"] is False
    assert latest_path.read_bytes() == before_pointer
    assert Path(fixture["manifest_path"]).read_bytes() == before_manifest
    assert not (tmp_path / "parquet" / "cn" / "_recoveries").exists()
    assert not (tmp_path / "parquet" / "cn" / ".market_writer.lock").exists()


def test_reactivate_ignores_stable_orphaned_atomic_parquet_temp(
    tmp_path: Path,
) -> None:
    fixture = _write_reactivation_fixture(tmp_path)
    table_path = Path(fixture["table_path"])
    residual = table_path.with_name(f".{table_path.name}.tmp-1234")
    residual.write_bytes(table_path.read_bytes())

    result = _run(tmp_path, fixture)

    assert result["status"] == "validated_dry_run"
    assert residual.exists()


def test_reactivate_rejects_unexpected_noncanonical_file(tmp_path: Path) -> None:
    fixture = _write_reactivation_fixture(tmp_path)
    table_path = Path(fixture["table_path"])
    (table_path.parent / "notes.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="non_parquet_file_rejected"):
        _run(tmp_path, fixture)


def test_logical_normalization_allows_blank_redundant_symbol() -> None:
    store = MarketDataStore(market="CN")
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "symbol": [None, "000002.SZ"],
            "trade_date": ["20260103", "20260103"],
        }
    )

    normalized = store._normalize_logical_frame(
        frame,
        logical_columns=["ts_code", "trade_date"],
    )

    assert normalized["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]


def test_logical_normalization_rejects_conflicting_redundant_symbol() -> None:
    store = MarketDataStore(market="CN")
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "symbol": ["000002.SZ"],
            "trade_date": ["20260103"],
        }
    )

    with pytest.raises(ValueError, match="snapshot_redundant_symbol_mismatch"):
        store._normalize_logical_frame(
            frame,
            logical_columns=["ts_code", "trade_date"],
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("manifest_sha256", "source_snapshot_manifest_sha256_mismatch"),
        ("latest_sha256", "market_pointer_cas_mismatch"),
        ("trade_date", "acknowledged_trade_date_mismatch"),
    ],
)
def test_reactivate_rejects_sha_date_and_cas_mismatches(
    tmp_path: Path,
    override: str,
    match: str,
) -> None:
    fixture = _write_reactivation_fixture(tmp_path)
    kwargs = {
        "market": "CN",
        "snapshot_id": fixture["snapshot_id"],
        "expected_snapshot_manifest_sha256": fixture["manifest_sha256"],
        "expected_market_pointer_sha256": fixture["latest_sha256"],
        "acknowledge_trade_date": "20260103",
        "reason": "unit test",
        "data_root": tmp_path,
    }
    if override == "manifest_sha256":
        kwargs["expected_snapshot_manifest_sha256"] = "f" * 64
    elif override == "latest_sha256":
        kwargs["expected_market_pointer_sha256"] = "e" * 64
    else:
        kwargs["acknowledge_trade_date"] = "20260102"

    with pytest.raises(ValueError, match=match):
        run_storage_reactivate_snapshot(**kwargs)

    assert not (tmp_path / "parquet" / "cn" / "_recoveries").exists()


def test_absolute_repository_data_root_cross_binds_neutral_relative_recovery_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(market_data_store, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(snapshot_recovery_binding, "REPO_ROOT", tmp_path)
    fixture = _write_reactivation_fixture(tmp_path, repository_layout=True)
    before_manifest = Path(fixture["manifest_path"]).read_bytes()
    before_table = Path(fixture["table_path"]).read_bytes()
    before_serving_a = Path(fixture["serving_a"]).read_bytes()

    result = _run(tmp_path, fixture, commit=True)

    assert result["status"] == "activated"
    latest_path = Path(fixture["latest_path"])
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    recovery = pointer["recovery"]
    assert pointer["manifest_path"] == ("data/parquet/cn/_snapshots/snapshot-good-v4.json")
    assert pointer["table_root"] == ("data/parquet/cn/_snapshots/snapshot-good-v4/table/bars")
    assert pointer["derived_serving_root"] == (
        "data/parquet/cn/_snapshots/snapshot-good-v4/serving/bars"
    )
    assert set(recovery) == {
        "schema_version",
        "recovery_id",
        "previous_market_pointer_sha256",
        "source_snapshot_manifest_sha256",
        "acknowledged_trade_date",
        "reason",
        "intent_path",
        "intent_sha256",
        "receipt_path",
    }
    assert recovery["schema_version"] == ("cn-market-snapshot-recovery-pointer.v1")
    assert "receipt_sha256" not in recovery
    assert not Path(recovery["intent_path"]).is_absolute()
    assert not Path(recovery["receipt_path"]).is_absolute()
    intent_path = tmp_path / recovery["intent_path"]
    receipt_path = tmp_path / recovery["receipt_path"]
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert _sha256(intent_path) == recovery["intent_sha256"]
    assert intent["intent_path"] == recovery["intent_path"]
    assert intent["receipt_path"] == recovery["receipt_path"]
    assert receipt["status"] == "activated"
    assert receipt["intent_sha256"] == recovery["intent_sha256"]
    assert receipt["new_market_pointer_sha256"] == _sha256(latest_path)
    assert receipt["new_market_pointer_sha256"] == result["new_market_pointer_sha256"]
    assert receipt["source_validation"]["table_logical_rowset_sha256"] == (
        receipt["source_validation"]["serving_logical_rowset_sha256"]
    )
    binding = snapshot_recovery_binding.validate_recovery_pointer_binding(
        pointer,
        pointer_sha256=_sha256(latest_path),
    )
    assert binding is not None
    assert binding["intent_path"] == recovery["intent_path"]
    assert binding["receipt_path"] == recovery["receipt_path"]
    assert Path(fixture["manifest_path"]).read_bytes() == before_manifest
    assert Path(fixture["table_path"]).read_bytes() == before_table
    assert Path(fixture["serving_a"]).read_bytes() == before_serving_a
    assert (
        MarketDataStore(
            market="CN",
            data_root=fixture["data_root"],
        ).validate_latest()["status"]
        == "passed"
    )


def test_reactivate_external_root_commit_is_rejected_before_write(
    tmp_path: Path,
) -> None:
    fixture = _write_reactivation_fixture(tmp_path)
    latest_path = Path(fixture["latest_path"])
    before_pointer = latest_path.read_bytes()

    with pytest.raises(
        ValueError,
        match="snapshot_reactivation_commit_requires_repository_data_root",
    ):
        _run(tmp_path, fixture, commit=True)

    assert latest_path.read_bytes() == before_pointer
    assert not (tmp_path / "parquet" / "cn" / "_recoveries").exists()
    assert not (tmp_path / "parquet" / "cn" / ".market_writer.lock").exists()


def test_reactivate_rejects_noncanonical_source_manifest_before_recovery_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(market_data_store, "_REPOSITORY_ROOT", tmp_path)
    fixture = _write_reactivation_fixture(tmp_path, repository_layout=True)
    manifest_path = Path(fixture["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fixture["manifest_sha256"] = _sha256(manifest_path)
    latest_path = Path(fixture["latest_path"])
    before_pointer = latest_path.read_bytes()

    with pytest.raises(
        ValueError,
        match="source_snapshot_manifest_path_not_repository_relative",
    ):
        _run(tmp_path, fixture, commit=True)

    assert latest_path.read_bytes() == before_pointer
    assert not (tmp_path / "data" / "parquet" / "cn" / "_recoveries").exists()


@pytest.mark.parametrize("mutation", ["value", "key"])
def test_reactivate_rejects_table_serving_value_or_key_difference(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _write_reactivation_fixture(tmp_path)
    serving_path = Path(fixture["serving_a"])
    serving = pd.read_parquet(serving_path)
    if mutation == "value":
        serving.loc[serving["trade_date"].eq("20260103"), "close"] = 999.0
    else:
        serving = serving.loc[serving["trade_date"].ne("20260102")].copy()
    serving.to_parquet(serving_path, index=False)

    with pytest.raises(
        ValueError,
        match="snapshot_table_serving_.*_mismatch",
    ):
        _run(tmp_path, fixture)

    assert not (tmp_path / "parquet" / "cn" / "_recoveries").exists()


def test_reactivate_rolls_back_only_its_own_attempted_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(market_data_store, "_REPOSITORY_ROOT", tmp_path)
    fixture = _write_reactivation_fixture(tmp_path, repository_layout=True)
    latest_path = Path(fixture["latest_path"])
    before = latest_path.read_bytes()
    store = MarketDataStore(market="CN", data_root=fixture["data_root"])

    def fail_post_validation(**_kwargs):
        raise ValueError("forced_post_activation_failure")

    monkeypatch.setattr(store, "_validate_reactivated_pointer", fail_post_validation)
    with pytest.raises(ValueError, match="forced_post_activation_failure"):
        store.reactivate_snapshot(
            snapshot_id=str(fixture["snapshot_id"]),
            expected_snapshot_manifest_sha256=str(fixture["manifest_sha256"]),
            expected_market_pointer_sha256=str(fixture["latest_sha256"]),
            acknowledge_trade_date="20260103",
            reason="force rollback",
            commit=True,
        )

    assert latest_path.read_bytes() == before
    receipts = list((tmp_path / "data" / "parquet" / "cn" / "_recoveries").glob("*/receipt.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["status"] == ("rolled_back")


def test_reactivate_does_not_rollback_a_concurrent_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(market_data_store, "_REPOSITORY_ROOT", tmp_path)
    fixture = _write_reactivation_fixture(tmp_path, repository_layout=True)
    latest_path = Path(fixture["latest_path"])
    concurrent_pointer = b'{"status":"EXTERNAL-CONCURRENT-WRITER"}\n'
    store = MarketDataStore(market="CN", data_root=fixture["data_root"])

    def replace_then_fail(**_kwargs):
        store._atomic_write_bytes(concurrent_pointer, latest_path)
        raise ValueError("forced_concurrent_change")

    monkeypatch.setattr(store, "_validate_reactivated_pointer", replace_then_fail)
    with pytest.raises(ValueError, match="forced_concurrent_change"):
        store.reactivate_snapshot(
            snapshot_id=str(fixture["snapshot_id"]),
            expected_snapshot_manifest_sha256=str(fixture["manifest_sha256"]),
            expected_market_pointer_sha256=str(fixture["latest_sha256"]),
            acknowledge_trade_date="20260103",
            reason="preserve concurrent pointer",
            commit=True,
        )

    assert latest_path.read_bytes() == concurrent_pointer
    receipt_path = next(
        (tmp_path / "data" / "parquet" / "cn" / "_recoveries").glob("*/receipt.json")
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "activation_failed_pointer_changed"
    assert receipt["rolled_back"] is False
