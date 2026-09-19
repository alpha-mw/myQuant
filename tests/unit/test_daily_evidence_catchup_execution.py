"""Real Calendar planning with controlled native execution seams; no live providers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS
from scripts import daily_catchup as catchup
from test_daily_evidence_catchup import fixture, put, day_input


def context(root, monkeypatch):
    args = fixture(root)
    args.pop("previous_trade_date")
    anchor = {
        "path": "results/operations/daily_production/CN/20260825/completion.v1.json",
        "sha256": "a" * 64,
    }
    inputs = {}
    previous = "20260825"
    for day in ("20260826", "20260827", "20260828"):
        value = day_input(day, v2=True)
        value["previous_trade_date"] = previous
        inputs[day] = put(root, day + ".json", canonical_json_bytes(value))
        previous = day
    calls = []

    def replay(**kw):
        calls.append(("replay", kw["trade_date"]))
        return {
            "native_replay_validated": True,
            "completion_ref": kw["completion_ref"],
            "trade_date": kw["trade_date"],
            "validated_nodes": sorted(EOD_NODE_IDS),
        }

    def run(**kw):
        assert kw["resume"] is True
        day = kw["input_ref"]["path"][:8]
        calls.append(("run", day))
        return {
            "status": "COMPLETE",
            "completion_ref": {
                "path": f"results/operations/daily_production/CN/{day}/completion.v1.json",
                "sha256": "b" * 64,
            },
        }

    monkeypatch.setattr(catchup, "replay_native_completion", replay)
    monkeypatch.setattr(catchup, "run_materialized_native_input", run)
    from scripts import daily_dashboard_publication

    # These ordering tests have no native EOD documents; publication admission
    # is part of the same explicit controlled execution seam.
    monkeypatch.setattr(
        daily_dashboard_publication,
        "complete_serving_result",
        lambda **kw: {"status": "COMPLETE", "completion_ref": kw["completion_ref"]},
    )
    return (
        {**args, "previous_completion_ref": anchor, "day_input_refs": inputs, "synthetic": True},
        calls,
        run,
    )


def test_catchup_calls_native_days_in_calendar_order(tmp_path, monkeypatch):
    args, calls, _ = context(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.iterdir()}
    result = catchup.run_native_catchup(**args)
    assert result["execution_state"] == "SUCCEEDED"
    assert calls == [
        ("replay", "20260825"),
        ("run", "20260826"),
        ("run", "20260827"),
        ("run", "20260828"),
    ]
    # This test uses no-op writer seams; the coordinator itself creates no files.
    assert before == {str(p): p.read_bytes() for p in tmp_path.iterdir()}


def test_failed_day_stops_successors_without_erasing_prior_completion(tmp_path, monkeypatch):
    args, calls, run = context(tmp_path, monkeypatch)

    def fail(**kw):
        if kw["input_ref"]["path"] == "20260827.json":
            calls.append(("failed", "20260827"))
            return {"status": "PARTIAL", "completion_ref": None}
        return run(**kw)

    monkeypatch.setattr(catchup, "run_materialized_native_input", fail)
    result = catchup.run_native_catchup(**args)
    assert result["execution_state"] == "PARTIAL"
    assert result["days"][0]["completion_ref"] is not None
    assert [r["execution_state"] for r in result["days"]] == ["SUCCEEDED", "PARTIAL", "BLOCKED"]
    assert ("run", "20260828") not in calls


def test_missing_input_or_bad_predecessor_starts_no_writers(tmp_path, monkeypatch):
    args, calls, _ = context(tmp_path, monkeypatch)
    missing = dict(args["day_input_refs"])
    missing.pop("20260826")
    result = catchup.run_native_catchup(**{**args, "day_input_refs": missing})
    assert result["execution_state"] == "BLOCKED"
    assert calls == [("replay", "20260825")]
    bad = day_input("20260827", v2=True)
    bad["previous_trade_date"] = "20260825"
    args["day_input_refs"]["20260827"] = put(tmp_path, "bad.json", canonical_json_bytes(bad))
    with pytest.raises(ContractError, match="PREDECESSOR_BINDING_INVALID"):
        catchup.run_native_catchup(**args)
    assert not any(kind == "run" for kind, day in calls)


def test_existing_completion_uses_replay_without_native_writer(tmp_path, monkeypatch):
    args, calls, _ = context(tmp_path, monkeypatch)

    def completed(workspace, day, input_ref):
        calls.append(("completed_replay", day))
        return {
            "path": f"results/operations/daily_production/CN/{day}/completion.v1.json",
            "sha256": "b" * 64,
        }

    monkeypatch.setattr(catchup, "_completed_day", completed)
    result = catchup.run_native_catchup(**args)
    assert result["execution_state"] == "NO_ACTION"
    assert not any(kind == "run" for kind, day in calls)


def test_target_already_equals_anchor_is_completed_not_nontrading(tmp_path, monkeypatch):
    args, calls, _ = context(tmp_path, monkeypatch)
    result = catchup.run_native_catchup(
        **{**args, "target_trade_date": "20260825", "day_input_refs": {}}
    )
    assert result["execution_state"] == "NO_ACTION"
    assert result["business_state"] == "COMPLETE"
    assert result["days"][0]["completion_ref"] == args["previous_completion_ref"]
    assert calls == [("replay", "20260825")]


def test_invalid_anchor_starts_no_daily_writer(tmp_path, monkeypatch):
    args, calls, _ = context(tmp_path, monkeypatch)

    def invalid(**kwargs):
        return {
            "native_replay_validated": False,
            "completion_ref": kwargs["completion_ref"],
            "trade_date": kwargs["trade_date"],
            "validated_nodes": sorted(EOD_NODE_IDS),
        }

    monkeypatch.setattr(catchup, "replay_native_completion", invalid)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ContractError, match="CATCHUP_NATIVE_ANCHOR_INVALID"):
        catchup.run_native_catchup(**args)
    assert calls == []
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_input_changed_during_anchor_replay_starts_no_daily_writer(tmp_path, monkeypatch):
    args, calls, _ = context(tmp_path, monkeypatch)
    original = catchup.replay_native_completion

    def mutate(**kwargs):
        result = original(**kwargs)
        # Calendar planning already read this input. Mutation during the next
        # native boundary must be detected before the first day's writer starts.
        path = tmp_path / args["day_input_refs"]["20260828"]["path"]
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(catchup, "replay_native_completion", mutate)
    with pytest.raises(ContractError, match="CATCHUP_INPUT_CHANGED"):
        catchup.run_native_catchup(**args)
    assert calls == [("replay", "20260825")]
    assert not (tmp_path / "results").exists()


def test_publication_pending_stops_successor_days(tmp_path, monkeypatch):
    args, calls, _ = context(tmp_path, monkeypatch)
    from scripts import daily_dashboard_publication

    seen = []

    def pending(**kwargs):
        seen.append(kwargs["completion_ref"])
        return {
            "status": "PARTIAL",
            "completion_ref": None,
            "sealed_evidence_ref": kwargs["completion_ref"],
        }

    monkeypatch.setattr(daily_dashboard_publication, "complete_serving_result", pending)
    result = catchup.run_native_catchup(**args)
    assert result["execution_state"] == "PARTIAL"
    assert result["days"][0]["completion_ref"] is None
    assert seen[0]["path"].endswith("/20260826/completion.v1.json")
    assert ("run", "20260827") not in calls and ("run", "20260828") not in calls
