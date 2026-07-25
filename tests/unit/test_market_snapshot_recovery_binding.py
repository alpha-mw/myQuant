from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from quant_investor.cli import main as cli_main
from quant_investor.market import snapshot_recovery_binding
from quant_investor.market.snapshot_recovery_binding import (
    MarketSnapshotRecoveryBindingError,
    canonical_json_bytes,
    file_sha256,
    read_json,
)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _build_recovery_fixture(tmp_path: Path) -> dict[str, Any]:
    snapshot_id = "snapshot-recovery-test"
    trade_date = "20260720"
    recovery_id = "recovery-test"
    old_pointer_sha = "1" * 64
    pit_membership_path = Path(
        "data/parquet/cn/reference/_generations/pit-test/stock_basic_membership.parquet"
    )
    pit_manifest_path = Path("data/parquet/cn/reference/_generations/pit-test/manifest.json")
    membership_file = tmp_path / pit_membership_path
    membership_file.parent.mkdir(parents=True, exist_ok=True)
    membership_file.write_bytes(b"immutable-pit-membership")
    membership_sha = file_sha256(membership_file)
    pit_manifest_payload = {
        "generation_id": "pit-test",
        "canonical_sha256": membership_sha,
    }
    pit_manifest_sha = _write_json(tmp_path / pit_manifest_path, pit_manifest_payload)

    source_manifest_relative = Path("data/parquet/cn/_snapshots") / f"{snapshot_id}.json"
    source_manifest_payload = {
        "snapshot_id": snapshot_id,
        "market": "CN",
        "status": "OK",
        "row_count": 10,
        "symbol_count": 3,
        "latest_trade_date": trade_date,
        "latest_available_trade_date": trade_date,
        "latest_complete_trade_date": trade_date,
        "manifest_path": str(source_manifest_relative),
        "table_root": f"data/parquet/cn/_snapshots/{snapshot_id}/table/bars",
        "derived_serving_root": (f"data/parquet/cn/_snapshots/{snapshot_id}/serving/bars"),
        "coverage": {
            "coverage_schema_version": "cn-full-a-coverage.v4",
            "complete": True,
            "blocking_incomplete_count": 0,
            "pit_membership_path": str(pit_membership_path),
            "pit_membership_sha256": membership_sha,
            "pit_generation_id": "pit-test",
            "pit_generation_manifest_path": str(pit_manifest_path),
            "pit_generation_manifest_sha256": pit_manifest_sha,
        },
    }
    source_manifest_file = tmp_path / source_manifest_relative
    source_manifest_sha = _write_json(source_manifest_file, source_manifest_payload)
    source_validation = {
        "table_inventory_sha256": "2" * 64,
        "serving_inventory_sha256": "3" * 64,
        "table_logical_rowset_sha256": "4" * 64,
        "serving_logical_rowset_sha256": "4" * 64,
        "logical_column_names": ["ts_code", "trade_date", "close"],
        "row_count": 10,
        "key_count": 10,
        "symbol_count": 3,
        "latest_trade_date": trade_date,
        "exact_date_symbol_count": 3,
        "pit_membership_path": str(pit_membership_path),
        "pit_membership_sha256": membership_sha,
        "pit_generation_manifest_path": str(pit_manifest_path),
        "pit_generation_manifest_sha256": pit_manifest_sha,
    }
    recovery_root = Path("data/parquet/cn/_recoveries") / recovery_id
    intent_relative = recovery_root / "intent.json"
    receipt_relative = recovery_root / "receipt.json"
    reason = "restore last internally complete immutable snapshot for offline research"
    intent_payload = {
        "schema_version": snapshot_recovery_binding.RECOVERY_INTENT_SCHEMA,
        "recovery_id": recovery_id,
        "market": "CN",
        "snapshot_id": snapshot_id,
        "created_at": "2026-07-22T01:00:00Z",
        "previous_market_pointer_sha256": old_pointer_sha,
        "source_snapshot_manifest_path": str(source_manifest_relative),
        "source_snapshot_manifest_sha256": source_manifest_sha,
        "acknowledged_trade_date": trade_date,
        "reason": reason,
        "intent_path": str(intent_relative),
        "receipt_path": str(receipt_relative),
        "source_validation": source_validation,
    }
    intent_sha = _write_json(tmp_path / intent_relative, intent_payload)
    pointer_payload = {
        "snapshot_id": snapshot_id,
        "status": "OK",
        "manifest_path": str(source_manifest_relative),
        "table_root": source_manifest_payload["table_root"],
        "derived_serving_root": source_manifest_payload["derived_serving_root"],
        "latest_available_trade_date": trade_date,
        "latest_complete_trade_date": trade_date,
        "latest_trade_date": trade_date,
        "coverage": source_manifest_payload["coverage"],
        "blockers": [],
        "recovery": {
            "schema_version": snapshot_recovery_binding.RECOVERY_POINTER_SCHEMA,
            "recovery_id": recovery_id,
            "previous_market_pointer_sha256": old_pointer_sha,
            "source_snapshot_manifest_sha256": source_manifest_sha,
            "acknowledged_trade_date": trade_date,
            "reason": reason,
            "intent_path": str(intent_relative),
            "intent_sha256": intent_sha,
            "receipt_path": str(receipt_relative),
        },
    }
    pointer_relative = Path("data/parquet/cn/_latest.json")
    pointer_sha = _write_json(tmp_path / pointer_relative, pointer_payload)
    receipt_payload = {
        "schema_version": snapshot_recovery_binding.RECOVERY_RECEIPT_SCHEMA,
        "status": "activated",
        "recovery_id": recovery_id,
        "market": "CN",
        "snapshot_id": snapshot_id,
        "activated_at": "2026-07-22T01:01:00Z",
        "previous_market_pointer_sha256": old_pointer_sha,
        "new_market_pointer_sha256": pointer_sha,
        "source_snapshot_manifest_path": str(source_manifest_relative),
        "source_snapshot_manifest_sha256": source_manifest_sha,
        "acknowledged_trade_date": trade_date,
        "reason": reason,
        "intent_path": str(intent_relative),
        "intent_sha256": intent_sha,
        "receipt_path": str(receipt_relative),
        "source_validation": source_validation,
    }
    receipt_file = tmp_path / receipt_relative
    _write_json(receipt_file, receipt_payload)
    return {
        "pointer_payload": pointer_payload,
        "pointer_path": pointer_relative,
        "pointer_sha256": pointer_sha,
        "receipt_path": receipt_file,
        "source_manifest_path": source_manifest_relative,
        "source_manifest_sha256": source_manifest_sha,
        "pit_membership_path": pit_membership_path,
        "pit_membership_sha256": membership_sha,
        "pit_manifest_path": pit_manifest_path,
        "pit_manifest_sha256": pit_manifest_sha,
        "snapshot_id": snapshot_id,
        "trade_date": trade_date,
    }


def test_recovery_binding_is_version_neutral_and_sealable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_recovery_fixture(tmp_path)
    monkeypatch.setattr(snapshot_recovery_binding, "REPO_ROOT", tmp_path)

    binding = snapshot_recovery_binding.validate_recovery_pointer_binding(
        fixture["pointer_payload"],
        pointer_sha256=fixture["pointer_sha256"],
    )

    assert binding is not None
    assert binding["schema_version"] == "cn-market-snapshot-recovery-binding.v1"
    assert binding["new_market_pointer_sha256"] == fixture["pointer_sha256"]
    assert binding["acknowledged_trade_date"] == fixture["trade_date"]
    assert binding["restored_trade_date"] == fixture["trade_date"]
    assert (
        binding["semantic_digests"]["table_sha256"] == binding["semantic_digests"]["serving_sha256"]
    )
    assert (
        snapshot_recovery_binding.validate_recovery_pointer_binding(
            fixture["pointer_payload"],
            pointer_sha256=fixture["pointer_sha256"],
            expected_binding=binding,
        )
        == binding
    )


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b'{"status":"activated","status":"activated"}', "duplicate JSON key"),
        (b'{"activated_at":NaN}', "invalid JSON constant"),
    ],
)
def test_recovery_binding_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
    match: str,
) -> None:
    fixture = _build_recovery_fixture(tmp_path)
    monkeypatch.setattr(snapshot_recovery_binding, "REPO_ROOT", tmp_path)
    fixture["receipt_path"].write_bytes(raw)

    with pytest.raises(MarketSnapshotRecoveryBindingError, match=match):
        snapshot_recovery_binding.validate_recovery_pointer_binding(
            fixture["pointer_payload"],
            pointer_sha256=fixture["pointer_sha256"],
        )


def test_recovery_binding_rejects_leaf_and_parent_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_recovery_fixture(tmp_path)
    monkeypatch.setattr(snapshot_recovery_binding, "REPO_ROOT", tmp_path)
    receipt_path = fixture["receipt_path"]
    receipt_bytes = receipt_path.read_bytes()
    receipt_path.unlink()
    external_receipt = tmp_path / "external-receipt.json"
    external_receipt.write_bytes(receipt_bytes)
    receipt_path.symlink_to(external_receipt)

    with pytest.raises(
        MarketSnapshotRecoveryBindingError,
        match="recovery receipt symlink rejected",
    ):
        snapshot_recovery_binding.validate_recovery_pointer_binding(
            fixture["pointer_payload"],
            pointer_sha256=fixture["pointer_sha256"],
        )

    receipt_path.unlink()
    receipt_path.write_bytes(receipt_bytes)
    recoveries = tmp_path / "data/parquet/cn/_recoveries"
    real_recoveries = tmp_path / "real-recoveries"
    recoveries.rename(real_recoveries)
    recoveries.symlink_to(real_recoveries, target_is_directory=True)
    with pytest.raises(
        MarketSnapshotRecoveryBindingError,
        match="recovery intent symlink rejected",
    ):
        snapshot_recovery_binding.validate_recovery_pointer_binding(
            fixture["pointer_payload"],
            pointer_sha256=fixture["pointer_sha256"],
        )


def test_secure_json_reader_rejects_file_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stable.json"
    source.write_bytes(b'{"status":"OK"}\n')
    real_fstat = os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        observed = real_fstat(descriptor)
        calls += 1
        if calls != 2:
            return observed
        changed = list(observed)
        changed[6] += 1
        return os.stat_result(changed)

    monkeypatch.setattr(snapshot_recovery_binding.os, "fstat", drifting_fstat)
    with pytest.raises(
        MarketSnapshotRecoveryBindingError,
        match="input changed during read",
    ):
        read_json(source)


def test_recovery_binding_rejects_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_recovery_fixture(tmp_path)
    monkeypatch.setattr(snapshot_recovery_binding, "REPO_ROOT", tmp_path)
    receipt_path = fixture["receipt_path"]
    replacement = receipt_path.with_name("replacement.json")
    replacement.write_bytes(receipt_path.read_bytes())
    real_reader = snapshot_recovery_binding._read_stable_regular_file

    def replacing_reader(
        descriptor: int,
        *,
        source: str,
        max_bytes: int | None,
    ) -> bytes:
        raw = real_reader(
            descriptor,
            source=source,
            max_bytes=max_bytes,
        )
        if source.endswith("receipt.json"):
            os.replace(replacement, receipt_path)
        return raw

    monkeypatch.setattr(
        snapshot_recovery_binding,
        "_read_stable_regular_file",
        replacing_reader,
    )
    with pytest.raises(
        MarketSnapshotRecoveryBindingError,
        match="recovery receipt changed during read",
    ):
        snapshot_recovery_binding.validate_recovery_pointer_binding(
            fixture["pointer_payload"],
            pointer_sha256=fixture["pointer_sha256"],
        )


def test_sealed_recovery_binding_rejects_missing_or_replaced_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_recovery_fixture(tmp_path)
    monkeypatch.setattr(snapshot_recovery_binding, "REPO_ROOT", tmp_path)
    binding = snapshot_recovery_binding.validate_recovery_pointer_binding(
        fixture["pointer_payload"],
        pointer_sha256=fixture["pointer_sha256"],
    )
    assert binding is not None

    receipt = read_json(fixture["receipt_path"])
    receipt["activated_at"] = "2026-07-22T01:02:00Z"
    _write_json(fixture["receipt_path"], receipt)
    with pytest.raises(
        MarketSnapshotRecoveryBindingError,
        match="sealed recovery binding mismatch",
    ):
        snapshot_recovery_binding.validate_recovery_pointer_binding(
            fixture["pointer_payload"],
            pointer_sha256=fixture["pointer_sha256"],
            expected_binding=binding,
        )

    fixture["receipt_path"].unlink()
    with pytest.raises(
        MarketSnapshotRecoveryBindingError,
        match="recovery receipt unavailable",
    ):
        snapshot_recovery_binding.validate_recovery_pointer_binding(
            fixture["pointer_payload"],
            pointer_sha256=fixture["pointer_sha256"],
            expected_binding=binding,
        )


def test_recovery_binding_rejects_forged_receipt_pointer_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_recovery_fixture(tmp_path)
    monkeypatch.setattr(snapshot_recovery_binding, "REPO_ROOT", tmp_path)
    receipt = read_json(fixture["receipt_path"])
    receipt["new_market_pointer_sha256"] = "f" * 64
    _write_json(fixture["receipt_path"], receipt)

    with pytest.raises(
        MarketSnapshotRecoveryBindingError,
        match="receipt/current pointer hash mismatch",
    ):
        snapshot_recovery_binding.validate_recovery_pointer_binding(
            fixture["pointer_payload"],
            pointer_sha256=fixture["pointer_sha256"],
        )


def test_storage_reactivate_cli_defaults_to_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"status": "dry_run"}

    monkeypatch.setattr(cli_main, "run_storage_reactivate_snapshot", fake_run)
    monkeypatch.setattr(cli_main, "_print_json", lambda payload: None)
    cli_main.main(
        [
            "market",
            "storage-reactivate-snapshot",
            "--market",
            "CN",
            "--snapshot-id",
            "snapshot-recovery-test",
            "--expected-snapshot-manifest-sha256",
            "a" * 64,
            "--expected-market-pointer-sha256",
            "b" * 64,
            "--acknowledge-trade-date",
            "20260720",
            "--reason",
            "operator recovery",
        ]
    )

    assert observed["commit"] is False
    assert observed["data_root"] is None
    assert observed["acknowledge_trade_date"] == "20260720"
