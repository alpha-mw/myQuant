from concurrent.futures import ThreadPoolExecutor

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, GRAPH_SHA256, NodeState
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.journal_storage import JournalStorage, ROOT
from quant_investor.system.errors import SystemImmutableConflict, SystemSecurityError


def request():
    return {
        "schema_version": "cn-daily-node-request.v1",
        "trade_date": "20260904",
        "node_id": "top100",
        "graph_sha256": GRAPH_SHA256,
        "release_ref": {"path": "release.json", "sha256": "a" * 64},
        "adapter_sha256": "b" * 64,
        "policy_refs": {},
        "input_refs": {"factor": {"path": "factor.json", "sha256": "c" * 64}},
    }


def output():
    return {"manifest": {"path": "pool/manifest.json", "sha256": "d" * 64}}


def test_crash_cannot_blindly_restart_writer_and_adoption_preserves_start(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        first = journal.begin(request())
    resumed = DailyJournal(str(tmp_path), "20260904")
    with resumed.locked():
        state = resumed.inspect(request())
        assert state["recovery_state"] == "IN_DOUBT"
        assert state["start_ref"] == first["start_ref"]
        with pytest.raises(ContractError, match="RECONCILE_BEFORE_RETRY"):
            resumed.begin(request(), reconciled_no_write=True)
        # Native adapter validation is required before this internal adoption call.
        terminal = resumed.finish(
            request(), state=NodeState.SUCCEEDED, output_refs=output(), recovered=True
        )
        assert terminal["terminal"]["recovered"] is True
        assert terminal["start_ref"] == first["start_ref"]
        with pytest.raises(ContractError, match="SUCCESS_REPLAY_REQUIRED"):
            resumed.begin(request())
        assert resumed.inspect(request())["terminal_ref"] == terminal["terminal_ref"]


def test_retry_requires_typed_retryability_and_reconciliation(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        journal.begin(request())
        first = journal.finish(
            request(), state=NodeState.FAILED, output_refs={}, failure_code="PROVIDER_UNAVAILABLE"
        )
        with pytest.raises(ContractError, match="RETRY_NOT_AUTHORIZED"):
            journal.begin(request())
        assert journal.begin(request(), reconciled_no_write=True)["attempt"] == 2
        journal.finish(
            request(), state=NodeState.FAILED, output_refs={}, failure_code="WRITER_FAILED"
        )
        with pytest.raises(ContractError, match="RETRY_NOT_AUTHORIZED"):
            journal.begin(request(), reconciled_no_write=True)
        assert (tmp_path / first["terminal_ref"]["path"]).exists()


def test_node_input_conflict_does_not_hide_prior_success(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        journal.begin(request())
        journal.finish(request(), state=NodeState.SUCCEEDED, output_refs=output())
        altered = request()
        altered["adapter_sha256"] = "e" * 64
        with pytest.raises(ContractError, match="NODE_INPUT_CONFLICT"):
            journal.begin(altered)


def test_day_lock_serializes_competing_native_writer_claims(tmp_path):
    def claim(_):
        journal = DailyJournal(str(tmp_path), "20260904")
        with journal.locked():
            if journal.inspect(request())["state"] != "NOT_STARTED":
                return "RECONCILE"
            journal.begin(request())
            return "WRITE"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(claim, range(2))) == ["RECONCILE", "WRITE"]


def test_lock_required_and_success_without_refs_rejected(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with pytest.raises(ContractError, match="LOCK_REQUIRED"):
        journal.begin(request())
    with journal.locked():
        journal.begin(request())
        with pytest.raises(ContractError, match="SUCCESS_WITHOUT_OUTPUT"):
            journal.finish(request(), state=NodeState.SUCCEEDED, output_refs={})


def test_storage_is_immutable_except_exact_derived_status_path(tmp_path):
    storage = JournalStorage(str(tmp_path))
    path = str(ROOT / "20260904/receipt.json")
    first = storage.write(path, b'{"ok":true}')
    assert storage.write(path, first.data).byte_sha256 == first.byte_sha256
    with pytest.raises(SystemImmutableConflict):
        storage.write(path, b'{"ok":false}')
    with pytest.raises(SystemSecurityError):
        storage.write(path, b'{"ok":false}', projection=True)
    projection = str(ROOT / "20260904/dag-status.v1.json")
    storage.write(projection, b'{"revision":1}', projection=True)
    assert storage.write(projection, b'{"revision":2}', projection=True).data == b'{"revision":2}'
    with pytest.raises(SystemSecurityError, match="ROOT_INVALID"):
        storage.write("results/system/_active.json", b'{"bad":true}')


def test_symlink_and_hardlink_cannot_supply_governed_receipt(tmp_path):
    storage = JournalStorage(str(tmp_path))
    path = str(ROOT / "20260904/receipt.json")
    stored = storage.write(path, b'{"ok":true}')
    import os

    os.link(tmp_path / path, tmp_path / "hardlink")
    with pytest.raises(SystemSecurityError):
        storage.read(path)
    (tmp_path / "hardlink").unlink()
    (tmp_path / path).unlink()
    (tmp_path / "outside").write_bytes(stored.data)
    (tmp_path / path).symlink_to(tmp_path / "outside")
    with pytest.raises(SystemSecurityError):
        storage.read(path)


def test_resealed_authority_and_chronology_tampering_rejected(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        journal.begin(request())
        state = journal.finish(request(), state=NodeState.SUCCEEDED, output_refs=output())
        path = tmp_path / state["terminal_ref"]["path"]
        altered = state["terminal"]
        altered["authority"]["broker"] = 0
        path.write_bytes(canonical_json_bytes(altered))
        with pytest.raises(ContractError, match="TERMINAL_INVALID"):
            journal.inspect(request())
        altered["authority"]["broker"] = False
        altered["finished_at"] = "2000-01-01T00:00:00Z"
        path.write_bytes(canonical_json_bytes(altered))
        with pytest.raises(ContractError, match="CHRONOLOGY_INVALID"):
            journal.inspect(request())


def test_production_calendar_or_pointer_not_created_by_journal(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        journal.begin(request())
        journal.finish(
            request(), state=NodeState.BLOCKED, output_refs={}, failure_code="INPUT_MISSING"
        )
    assert not (tmp_path / "results/factors").exists()
    assert not (tmp_path / "results/system").exists()
    assert not (tmp_path / "results/strategy_records").exists()
