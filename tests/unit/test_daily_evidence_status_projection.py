"""Phase 1 observational fields retain real journal evidence and no authority."""

from copy import deepcopy
import hashlib
import json

import pytest

from quant_investor.operations.daily_contract import NodeState, failure
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_status import read_daily_status
from quant_investor.operations.status_projection import project_node_status
from test_daily_evidence_dag_journal import request
from test_daily_evidence_runner import setup

FIELDS = {
    "trigger_reason",
    "blocking_reason",
    "retryable",
    "upstream_refs",
    "output_refs",
    "started_at",
    "finished_at",
    "attempt",
}


def inventory(root):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_missing_request_does_not_invent_zero_attempt_or_ready_inputs(tmp_path):
    value = read_daily_status(str(tmp_path), "20260904")
    for row in value["nodes"].values():
        assert FIELDS <= set(row)
        assert row["state"] == "NOT_STARTED"
        assert row["trigger_reason"] is row["attempt"] is None
        assert row["started_at"] is row["finished_at"] is None
        assert row["blocking_reason"] is None and row["retryable"] is False
        assert row["upstream_refs"] == row["output_refs"] == {}
    assert not list(tmp_path.iterdir())


def test_exact_request_absence_and_running_start_use_original_journal_values(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    req = request()
    absent = journal.readonly_inspect(req)
    projected = project_node_status(absent)
    assert projected["attempt"] == 0 and projected["trigger_reason"] == "NOT_STARTED"
    assert projected["upstream_refs"] == {}
    with journal.locked():
        started = journal.begin(req)
    before = inventory(tmp_path)
    row = read_daily_status(str(tmp_path), "20260904")["nodes"]["top100"]
    assert row["trigger_reason"] == "ATTEMPT_RUNNING" and row["attempt"] == started["attempt"]
    assert row["started_at"] == started["start"]["started_at"]
    assert row["finished_at"] is None and row["output_refs"] == {}
    assert inventory(tmp_path) == before


def test_recorded_readback_ignores_persisted_invocation_action(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    output = tmp_path / "output.json"
    output.write_bytes(b"{}")
    output.chmod(0o600)
    upstream = {"path": "upstream.json", "sha256": "d" * 64}
    req = {**request(), "input_refs": {"upstream.factor": upstream}}
    refs = {"native": {"path": "output.json", "sha256": hashlib.sha256(b"{}").hexdigest()}}
    with journal.locked():
        journal.begin(req)
        original = journal.finish(req, state=NodeState.SUCCEEDED, output_refs=refs)
    projection = tmp_path / journal.root / "dag-status.v1.json"
    projection.write_text(
        json.dumps(
            {
                "nodes": {
                    "top100": {
                        "command_status": "EXECUTED",
                        "trigger_reason": "NATIVE_EXECUTED",
                        "started_at": "2099-01-01T00:00:00Z",
                        "attempt": 99,
                    }
                }
            }
        )
    )
    before = inventory(tmp_path)
    row = read_daily_status(str(tmp_path), "20260904")["nodes"]["top100"]
    assert row["trigger_reason"] == "RECORDED_TERMINAL"
    assert row["started_at"] == original["start"]["started_at"]
    assert row["finished_at"] == original["terminal"]["finished_at"]
    assert row["attempt"] == 1 and row["output_refs"] == refs
    assert row["upstream_refs"] == {"upstream.factor": upstream}
    assert "command_status" not in row
    assert (FIELDS - {"attempt", "started_at"}).isdisjoint(original["start"])
    assert (FIELDS - {"attempt", "output_refs", "finished_at"}).isdisjoint(original["terminal"])
    assert inventory(tmp_path) == before


def test_invalid_output_retains_diagnostic_without_blessing_refs_or_times(tmp_path):
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
    output.write_bytes(b"changed")
    before = inventory(tmp_path)
    row = read_daily_status(str(tmp_path), "20260904")["nodes"]["top100"]
    assert row["state"] == "STALE" and row["trigger_reason"] == "EVIDENCE_INVALID"
    assert row["blocking_reason"] == "VALIDATION_FAILED" and row["retryable"] is False
    assert "SHA_MISMATCH" in row["reason"]
    assert row["attempt"] is row["started_at"] is row["finished_at"] is None
    assert row["upstream_refs"] == row["output_refs"] == {}
    assert inventory(tmp_path) == before


def test_runner_exposes_running_before_writer_then_local_execute_and_replay(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    real = adapters["calendar"].execute
    observed = []

    def checking(req):
        value = json.loads((tmp_path / runner.journal.root / "dag-status.v1.json").read_bytes())
        row = value["nodes"]["calendar"]
        assert row["state"] == "RUNNING" and row["trigger_reason"] == "ATTEMPT_RUNNING"
        assert row["started_at"] and row["finished_at"] is None
        assert row["attempt"] == 1 and row["output_refs"] == {}
        observed.append(row["started_at"])
        real(req)

    adapters["calendar"].execute = checking
    first = runner.run(templates)
    assert len(observed) == 1
    assert first["nodes"]["calendar"]["started_at"] == observed[0]
    assert all(row["trigger_reason"] == "NATIVE_EXECUTED" for row in first["nodes"].values())
    assert first["status"] == "PARTIAL" and first["completion_ref"] is None
    before = {
        p: data
        for p, data in inventory(tmp_path).items()
        if p.endswith(("start.json", "terminal.json", "request.json"))
    }
    second = runner.run(templates)
    assert all(row["trigger_reason"] == "NATIVE_REPLAY" for row in second["nodes"].values())
    assert len(calls) == 16
    assert all(inventory(tmp_path)[p] == value for p, value in before.items())
    assert all(row["attempt"] == 1 for row in second["nodes"].values())


def test_runner_reports_typed_failure_and_skips_without_fabricated_attempt(tmp_path):
    runner, adapters, templates, _ = setup(tmp_path)
    adapters["exposure"].blocked = True
    result = runner.run(templates)
    blocked, skipped = result["nodes"]["exposure"], result["nodes"]["decision"]
    assert blocked["blocking_reason"] == "INPUT_MISSING" and blocked["retryable"] is True
    assert blocked["attempt"] == 1
    assert skipped["trigger_reason"] == "WAITING_UPSTREAM"
    assert skipped["blocking_reason"] == "UPSTREAM_INCOMPLETE" and skipped["retryable"] is True
    assert skipped["attempt"] is None and skipped["started_at"] is None
    assert skipped["upstream_refs"] == skipped["output_refs"] == {}


def test_indeterminate_write_preserves_start_and_recovery_keeps_attempt(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    adapters["top100"].crash_once = True
    first = runner.run(templates)["nodes"]["top100"]
    assert first["trigger_reason"] == "BLOCKING_FAILURE"
    assert first["blocking_reason"] == "POST_WRITE_IN_DOUBT" and first["retryable"] is False
    assert first["attempt"] == 1 and first["started_at"] is not None
    assert first["finished_at"] is None
    second = runner.run(templates, resume=True)["nodes"]["top100"]
    assert second["trigger_reason"] == "NATIVE_RECOVERY"
    assert second["attempt"] == 1 and second["started_at"] == first["started_at"]
    assert calls.count("top100") == 1


@pytest.mark.parametrize("state", ["NOT_STARTED", "READY", "SKIPPED"])
def test_uninspected_state_has_no_invented_trigger_or_attempt(state):
    row = {"state": state}
    assert project_node_status(row)["trigger_reason"] is None
    assert project_node_status(row)["attempt"] is None


def test_helper_is_pure_and_uses_fixed_retry_taxonomy():
    row = {"state": "BLOCKED", "failure": failure("INPUT_MISSING", next_node="top100")}
    row["failure"]["retryable"] = False
    original = deepcopy(row)
    value = project_node_status(row)
    assert value["trigger_reason"] == "BLOCKING_FAILURE" and value["retryable"] is True
    assert row == original
    value["failure"]["code"] = "different"
    assert row == original
