"""Recorded integrity is distinct from native EOD/admission proof."""

import hashlib
from datetime import datetime, timezone
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.daily_contract import EOD_NODE_IDS, GRAPH_SHA256, ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.daily_runner import DayRunner
from test_daily_evidence_runner import FixtureAdapter
from test_daily_evidence_dag_journal import request


def put(root, path, value):
    raw = canonical_json_bytes(value)
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    p.chmod(0o600)
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize(
    "valid_store_location,bad_upstream", [(False, False), (True, False), (True, True)]
)
def test_recorded_closure_does_not_grant_consumer_admission(
    tmp_path, valid_store_location, bad_upstream
):
    calls = []
    release_ref = put(tmp_path, "release.json", {"synthetic": True})
    runner = DayRunner(str(tmp_path), "20260904", {})
    # Store output uses its own governed root, so this unit test replaces it with
    # a reference inside that root; these are explicitly not native completion proofs.
    adapters = {node: FixtureAdapter(tmp_path, node, calls) for node in EOD_NODE_IDS}
    if valid_store_location:
        from quant_investor.operations.daily_runner import Probe, NativeOutcome

        class StoreFixture(FixtureAdapter):
            def execute(self, req):
                super().execute(req)
                source = self.root / "fixture-native/store.json"
                self.target = (
                    "results/strategy_records/CN/aggressive_tech_manufacturing/fixture.json"
                )
                p = self.root / self.target
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(source.read_bytes())
                p.chmod(0o644)

            def probe(self, req):
                result = super().probe(req)
                if result.outcome is None:
                    return result
                old = result.outcome
                return Probe(
                    NativeOutcome(
                        old.state,
                        {
                            "native": {
                                "path": self.target,
                                "sha256": old.output_refs["native"]["sha256"],
                            }
                        },
                    )
                )

        adapters["store"] = StoreFixture(tmp_path, "store", calls)
    runner.adapters = adapters
    if bad_upstream:
        original_request = runner._request

        def mismatched(base, node, completed):
            value = original_request(base, node, completed)
            if node == "factor":
                value["input_refs"]["upstream.calendar"] = completed["pit"]["terminal_ref"]
            return value

        runner._request = mismatched
    result = runner.run(
        {n: {**request(), "node_id": n, "release_ref": release_ref} for n in EOD_NODE_IDS}
    )
    document = {
        "schema_version": "cn-daily-eod-completion.v1",
        "status": "SUCCEEDED",
        "market": "CN",
        "strategy_id": "aggressive_tech_manufacturing",
        "trade_date": "20260904",
        "graph_sha256": GRAPH_SHA256,
        "release_ref": release_ref,
        "native_inputs_ref": put(tmp_path, "input.json", {"synthetic": True}),
        "node_terminal_refs": {n: row["terminal_ref"] for n, row in result["nodes"].items()},
        "synthetic": True,
        "prospective_admission_state": "NOT_CLAIMED",
        "authority": FALSE_AUTHORITY,
        "native_validation_completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ref = put(tmp_path, str(runner.journal.root / "completion.v1.json"), document)
    if bad_upstream:
        with pytest.raises(ContractError, match="UPSTREAM_BINDING_INVALID"):
            inspect_recorded_completion(
                workspace=str(tmp_path), trade_date="20260904", completion_ref=ref
            )
    elif valid_store_location:
        before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        inspected = inspect_recorded_completion(
            workspace=str(tmp_path), trade_date="20260904", completion_ref=ref
        )
        assert inspected["native_replay_required"] is True
        assert inspected["consumer_admission"] is False
        assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    else:
        with pytest.raises(ContractError, match="TERMINAL_CHANGED:store"):
            inspect_recorded_completion(
                workspace=str(tmp_path), trade_date="20260904", completion_ref=ref
            )
    with pytest.raises(ContractError, match="PATH_INVALID"):
        inspect_recorded_completion(
            workspace=str(tmp_path), trade_date="20260905", completion_ref=ref
        )
