"""Native plan-phase tests with synthetic upstream seams, not full accounting proof."""

from datetime import datetime, timedelta, timezone
import hashlib
import json

import pandas as pd
import pytest

from _native_daily_store_fixture import NativeStoreFixture, batch as native


def fixture(tmp_path, monkeypatch):
    # Exercise current native Market/Calendar/Event/Store contracts rather than
    # extending the retired partial manifest and empty-pointer placeholders.
    book = NativeStoreFixture(tmp_path, stock_symbols=("002463.SZ",))
    return {
        **book.advance("2026-08-24"),
        "execute": False,
        "now": datetime(2026, 9, 7, 8, tzinfo=timezone.utc),
    }


def test_readonly_plan_has_no_writes_and_prepared_bytes_survive_clock_change(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    read = native.close_through_latest(**args)
    assert read["status"] == "PLAN_READY"
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    prepared = native.close_through_latest(**args, prepare_only=True)
    assert prepared["status"] == "PLAN_PREPARED"
    plan = tmp_path / prepared["plan_path"]
    raw = plan.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prepared["plan_sha256"]
    args["now"] = datetime(2026, 9, 7, 10, tzinfo=timezone.utc)
    replay = native.close_through_latest(**args, expected_plan_sha=prepared["plan_sha256"])
    assert replay["plan_sha256"] == prepared["plan_sha256"]
    assert plan.read_bytes() == raw
    pointer = args["record_root"] / "_record_store/current.v1.json"
    assert hashlib.sha256(pointer.read_bytes()).hexdigest() == args["expected_store_pointer_sha"]


def test_wrong_prepared_sha_rejected_before_business_writer(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    native.close_through_latest(**args, prepare_only=True)
    monkeypatch.setattr(native, "build_record", lambda **_: pytest.fail("business writer called"))
    args["execute"] = True
    with pytest.raises(native.StrategyRecordConflict, match="prepared plan SHA differs"):
        native.close_through_latest(**args, expected_plan_sha="0" * 64)


def test_prepare_execute_cannot_be_combined(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args["execute"] = True
    with pytest.raises(native.StrategyRecordStoreError, match="distinct phases"):
        native.close_through_latest(**args, prepare_only=True)


def test_completion_retains_exact_pointer_before_terminal_and_refuses_drift(tmp_path):
    root = tmp_path / "records"
    p = root / "_record_store/current.v1.json"
    p.parent.mkdir(parents=True)
    raw = native.canonical_json_bytes({"synthetic_pointer": "committed"})
    p.write_bytes(raw)
    p.chmod(0o600)
    plan = {
        "schema_id": native.BATCH_PLAN_SCHEMA,
        "transaction_id": "daily-close-20260904-" + "a" * 16,
        "input_fingerprint": "a" * 64,
        "requested_target": "2026-09-04",
        "effective_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    sha = hashlib.sha256(raw).hexdigest()
    result = native._write_completion(
        record_root=root, plan=plan, pointer_sha=sha, status="COMMITTED"
    )
    folder = native._completion_path(root, plan["transaction_id"]).parent
    assert (folder / "committed-pointer.v1.json").read_bytes() == raw
    assert result["pointer_sha256"] == sha
    with pytest.raises(native.StrategyRecordConflict, match="moved before completion custody"):
        native._write_completion(
            record_root=root, plan=plan, pointer_sha="0" * 64, status="COMMITTED"
        )


def test_commit_identity_reads_hash_bound_parquet_and_rejects_nonfinite(tmp_path):
    directory = tmp_path / "record"
    directory.mkdir()
    ledger = directory / "ledger_after_manual_switch.parquet"
    manual = directory / "manual.json"
    frame = pd.DataFrame(
        [{"symbol": "000001.SZ", "shares": 100, "avg_cost": 10.0, "cost_basis": 1000.0}]
    )
    frame.to_parquet(ledger, index=False)
    ledger.chmod(0o600)
    manual.write_bytes(native.canonical_json_bytes({"cash_after": 9000}))
    manual.chmod(0o600)
    record = {
        "ledger_path": str(ledger.relative_to(tmp_path)),
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "manual_manifest_path": str(manual.relative_to(tmp_path)),
        "manual_manifest_sha256": hashlib.sha256(manual.read_bytes()).hexdigest(),
    }
    got, cash = native._holdings_identity(tmp_path, record)
    assert got.to_dict("records") == frame.to_dict("records") and cash == 9000
    frame["shares"] = frame["shares"].astype(float)
    frame.to_parquet(ledger, index=False)
    record["ledger_sha256"] = hashlib.sha256(ledger.read_bytes()).hexdigest()
    same, _ = native._holdings_identity(tmp_path, record)
    assert same.equals(got)
    frame.loc[0, "avg_cost"] = float("inf")
    frame.to_parquet(ledger, index=False)
    record["ledger_sha256"] = hashlib.sha256(ledger.read_bytes()).hexdigest()
    with pytest.raises(native.StrategyRecordStoreError, match="nonfinite"):
        native._holdings_identity(tmp_path, record)


def test_metadata_recovery_never_repeats_catalog_cas(tmp_path, monkeypatch):
    root = tmp_path / "records"
    current = root / "_record_store/current.v1.json"
    current.parent.mkdir(parents=True)
    raw = native.canonical_json_bytes({"synthetic_committed_pointer": True})
    current.write_bytes(raw)
    current.chmod(0o600)
    sha = hashlib.sha256(raw).hexdigest()
    tx = "daily-close-20260904-" + "a" * 16
    completion_path = native._completion_path(root, tx)
    plan = {
        "schema_id": native.BATCH_PLAN_SCHEMA,
        "transaction_id": tx,
        "requested_target": "2026-09-04",
        "input_fingerprint": "a" * 64,
        "effective_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }

    def inspect(**kwargs):
        value = json.loads(completion_path.read_bytes()) if completion_path.exists() else None
        return {
            "plan": plan,
            "completion": value,
            "pointer_sha256": sha,
            "pointer_ref": {"path": "_record_store/current.v1.json", "sha256": sha},
            "write_performed": False,
        }

    monkeypatch.setattr(native, "inspect_close_commit", inspect)
    monkeypatch.setattr(
        native, "publish_catalog", lambda *args, **kwargs: pytest.fail("CAS repeated")
    )
    args = {
        "record_root": root,
        "transaction_id": tx,
        "expected_plan_sha": "b" * 64,
        "expected_source_pointer_sha": "c" * 64,
        "expected_target": "2026-09-04",
    }
    first = native.recover_close_completion(**args)
    saved = completion_path.read_bytes()
    assert first["metadata_write_performed"] is True and first["catalog_cas_performed"] is False
    assert json.loads(saved)["status"] == "RECOVERED_AFTER_CAS"
    second = native.recover_close_completion(**args)
    assert second["metadata_write_performed"] is False
    assert current.read_bytes() == raw and completion_path.read_bytes() == saved


def test_interrupted_metadata_write_never_exposes_partial_final_file(tmp_path, monkeypatch):
    path = tmp_path / "transaction/plan.v1.json"
    original = native.os.write
    count = []

    def interrupted(fd, raw):
        count.append(1)
        if len(count) > 1:
            raise OSError("injected short-write interruption")
        return original(fd, raw[:1])

    monkeypatch.setattr(native.os, "write", interrupted)
    with pytest.raises(OSError, match="interruption"):
        native._write_exact_json(path, {"plan": "complete value"})
    assert not path.exists()
    monkeypatch.setattr(native.os, "write", original)
    sha = native._write_exact_json(path, {"plan": "complete value"})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
