"""Read-only status ignores green projections and rejects changed output bytes."""

import hashlib
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_status import read_daily_status
from quant_investor.operations.daily_contract import NodeState
from test_daily_evidence_dag_journal import request


def test_absent_status_does_not_create_results(tmp_path):
    result = read_daily_status(str(tmp_path), "20260904")
    assert result["status"] == "NOT_STARTED"
    assert list(tmp_path.iterdir()) == []


def test_status_replays_terminal_and_output_without_writes(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    output = tmp_path / "output.json"
    output.write_bytes(b"{}")
    output.chmod(0o600)
    with journal.locked():
        journal.begin(request())
        journal.finish(
            request(),
            state=NodeState.SUCCEEDED,
            output_refs={
                "native": {"path": "output.json", "sha256": hashlib.sha256(b"{}").hexdigest()}
            },
        )
    projection = tmp_path / journal.root / "dag-status.v1.json"
    projection.write_bytes(b'{"status":"SUCCEEDED"}')
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = read_daily_status(str(tmp_path), "20260904")
    assert result["nodes"]["top100"]["state"] == "SUCCEEDED"
    assert result["status"] == "PARTIAL" and result["completion_ref"] is None
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    output.write_bytes(b'{"changed":true}')
    result = read_daily_status(str(tmp_path), "20260904")
    assert result["nodes"]["top100"]["state"] == "STALE"
    assert result["status"] == "FAILED"


def test_store_native_public_read_mode_is_supported_without_symlink_fallback(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    req = {**request(), "node_id": "store"}
    relative = "results/strategy_records/CN/aggressive_tech_manufacturing/example/ledger.parquet"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture")
    path.chmod(0o644)
    with journal.locked():
        journal.begin(req)
        journal.finish(
            req,
            state=NodeState.SUCCEEDED,
            output_refs={
                "ledger": {"path": relative, "sha256": hashlib.sha256(b"fixture").hexdigest()}
            },
        )
    assert read_daily_status(str(tmp_path), "20260904")["nodes"]["store"]["state"] == "SUCCEEDED"
    target = path.with_name("elsewhere.parquet")
    path.rename(target)
    path.symlink_to(target)
    assert read_daily_status(str(tmp_path), "20260904")["nodes"]["store"]["state"] == "STALE"
