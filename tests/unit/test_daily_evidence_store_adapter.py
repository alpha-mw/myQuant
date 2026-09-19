"""Store adapter phase/authority guards with native plan writer and fixture inputs."""

import hashlib
import pytest

from test_daily_evidence_store_plan import fixture, native
from scripts import daily_production_store_adapter as adapter_module


def setup(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args.pop("now")
    args.pop("execute")
    planned = adapter_module.prepare_store_plan(args)
    adapter = adapter_module.StoreCloseAdapter(
        arguments=args,
        trade_date="20260824",
        plan_ref={"path": planned["plan_path"], "sha256": planned["plan_sha256"]},
        release_ref={"path": "release.json", "sha256": "a" * 64},
    )
    return adapter, args, planned


def test_native_prepared_store_adapter_probe_is_readonly(tmp_path, monkeypatch):
    adapter, args, planned = setup(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = adapter.probe(adapter.template())
    assert result.safe_to_execute is True and result.recovery_only is False
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_metadata_recovery_uses_original_preimage_not_current_sha(tmp_path, monkeypatch):
    adapter, args, planned = setup(tmp_path, monkeypatch)
    current = args["record_root"] / "_record_store/current.v1.json"
    current.write_bytes(b'{"synthetic_committed":true}')
    proof = {"completion": None, "pointer_ref": {"path": "_record_store/current.v1.json"}}
    monkeypatch.setattr(native, "inspect_close_commit", lambda **kwargs: proof)
    calls = []
    monkeypatch.setattr(native, "recover_close_completion", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        native, "close_through_latest", lambda **kwargs: pytest.fail("CAS writer repeated")
    )
    probe = adapter.probe(adapter.template())
    assert probe.recovery_only is True and probe.safe_to_execute is False
    adapter.execute(adapter.template())
    assert calls[0]["expected_source_pointer_sha"] == args["expected_store_pointer_sha"]
    assert (
        calls[0]["expected_source_pointer_sha"] != hashlib.sha256(current.read_bytes()).hexdigest()
    )


def test_changed_prepared_plan_blocks_before_execute(tmp_path, monkeypatch):
    adapter, args, planned = setup(tmp_path, monkeypatch)
    (tmp_path / planned["plan_path"]).write_bytes(b"{}")
    monkeypatch.setattr(
        native, "close_through_latest", lambda **kwargs: pytest.fail("writer called")
    )
    with pytest.raises(native.StrategyRecordStoreError, match="SHA mismatch"):
        adapter.execute(adapter.template())
