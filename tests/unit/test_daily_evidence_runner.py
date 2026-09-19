"""Fixed-graph engine tests; fixture adapters are not full production DAG proof."""

import hashlib
import json

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import EOD_NODE_IDS, NodeState
from quant_investor.operations.daily_runner import DayRunner, NativeOutcome, Probe
from test_daily_evidence_dag_journal import request


class FixtureAdapter:
    resume_safe = True

    def __init__(self, root, node, calls):
        self.root, self.node, self.calls = root, node, calls
        self.blocked = False
        self.crash_once = False

    def probe(self, request):
        if self.blocked:
            return Probe(NativeOutcome(NodeState.BLOCKED, {}, "INPUT_MISSING"))
        path = self.root / "fixture-native" / (self.node + ".json")
        if not path.exists():
            return Probe(None, safe_to_execute=True)
        raw = path.read_bytes()
        value = json.loads(raw)
        assert value["request_sha256"] == hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        return Probe(
            NativeOutcome(
                NodeState.SUCCEEDED,
                {
                    "native": {
                        "path": str(path.relative_to(self.root)),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                },
            )
        )

    def execute(self, request):
        self.calls.append(self.node)
        path = self.root / "fixture-native" / (self.node + ".json")
        path.parent.mkdir(exist_ok=True)
        raw = canonical_json_bytes(
            {
                "synthetic": True,
                "request_sha256": hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
            }
        )
        path.write_bytes(raw)
        path.chmod(0o600)
        if self.crash_once:
            self.crash_once = False
            raise RuntimeError("simulated interruption after writer")


def setup(root):
    calls = []
    adapters = {node: FixtureAdapter(root, node, calls) for node in EOD_NODE_IDS}
    templates = {node: {**request(), "node_id": node} for node in EOD_NODE_IDS}
    return DayRunner(str(root), "20260904", adapters), adapters, templates, calls


def test_all_success_is_only_completion_candidate_and_replay_has_zero_writes(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    first = runner.run(templates)
    assert first["status"] == "PARTIAL"
    assert first["completion_status"] == "AWAITING_NATIVE_COMPLETION_VALIDATION"
    assert first["completion_ref"] is None
    assert len(calls) == 16
    hashes = {node: row["terminal_ref"] for node, row in first["nodes"].items()}
    second = runner.run(templates)
    assert len(calls) == 16
    assert {node: row["terminal_ref"] for node, row in second["nodes"].items()} == hashes
    assert not (
        tmp_path / "results/operations/daily_production/CN/20260904/completion.v1.json"
    ).exists()


def test_missing_research_does_not_stop_independent_financial_branch(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    adapters["exposure"].blocked = True
    result = runner.run(templates)
    assert result["nodes"]["exposure"]["state"] == "BLOCKED"
    assert result["nodes"]["decision"]["state"] == "SKIPPED"
    assert result["nodes"]["store"]["state"] == "SUCCEEDED"
    assert result["nodes"]["dashboard"]["state"] == "SUCCEEDED"
    assert result["status"] == "PARTIAL"


def test_crashed_pool_resumes_without_reexecuting_successful_factor(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    adapters["top100"].crash_once = True
    first = runner.run(templates)
    assert first["nodes"]["top100"]["state"] == "FAILED"
    second = runner.run(templates, resume=True)
    assert second["nodes"]["top100"]["state"] == "SUCCEEDED"
    assert second["nodes"]["top100"]["terminal"]["recovered"] is True
    assert calls.count("factor") == calls.count("top100") == 1


def test_interrupted_store_projection_and_immutable_custody_remain_distinct(tmp_path):
    from _native_missing_store_case import validate_interrupted_boundary
    from quant_investor.operations.daily_status import read_daily_status

    runner, adapters, templates, _ = setup(tmp_path)
    adapters["store"].crash_once = True
    invocation = runner.run(templates)
    recorded = read_daily_status(str(tmp_path), "20260904")
    assert invocation["nodes"]["store"]["state"] == "FAILED"
    assert recorded["nodes"]["store"]["state"] == "RUNNING"
    assert recorded["nodes"]["store"]["recovery_state"] == "IN_DOUBT"
    validate_interrupted_boundary(recorded, invocation)


def test_preparation_directory_does_not_imply_a_started_business_day(tmp_path):
    import pytest
    from _native_configured_successor import require_unstarted_day
    from quant_investor.operations.daily_journal import DailyJournal

    journal = DailyJournal(str(tmp_path), "20260904")
    journal.storage.write(
        str(journal.root / "preparation" / ("a" * 64) / "request.json"),
        canonical_json_bytes({"synthetic_preparation": True}),
    )
    require_unstarted_day(tmp_path, "20260904")
    with journal.locked():
        journal.begin({**request(), "node_id": "calendar"})
    with pytest.raises(ValueError, match="business execution evidence"):
        require_unstarted_day(tmp_path, "20260904")


def test_resume_cannot_initiate_maintenance(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    adapters["calendar"].resume_safe = False
    result = runner.run(templates, resume=True)
    assert result["nodes"]["calendar"]["state"] == "BLOCKED"
    assert calls == []


def test_arrived_external_evidence_revises_only_failed_node(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    adapters["fundamental"].blocked = True
    first = runner.run(templates)
    old = first["nodes"]["fundamental"]["terminal_ref"]
    adapters["fundamental"].blocked = False
    templates["fundamental"]["input_refs"] = {
        "new_evidence": {"path": "f.json", "sha256": "e" * 64}
    }
    second = runner.run(templates, resume=True)
    assert second["nodes"]["fundamental"]["state"] == "SUCCEEDED"
    assert calls.count("factor") == 1
    assert (tmp_path / old["path"]).exists()


def test_known_committed_metadata_recovery_keeps_original_attempt(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)

    class CommittedAdapter(FixtureAdapter):
        def probe(self, request):
            native = self.root / "fixture-native" / (self.node + ".json")
            if native.exists() and not (self.root / "metadata.json").exists():
                return Probe(None, recovery_only=True)
            return super().probe(request)

        def execute(self, request):
            native = self.root / "fixture-native" / (self.node + ".json")
            if native.exists():
                (self.root / "metadata.json").write_text("{}")
                return
            self.crash_once = True
            super().execute(request)

    adapters["store"] = CommittedAdapter(tmp_path, "store", calls)
    runner.adapters = adapters
    first = runner.run(templates)
    start = first["nodes"]["store"]["start_ref"]
    second = runner.run(templates, resume=True)
    result = second["nodes"]["store"]
    assert result["state"] == "SUCCEEDED" and result["attempt"] == 1
    assert result["start_ref"] == start and result["terminal"]["recovered"] is True
    assert calls.count("store") == 1
