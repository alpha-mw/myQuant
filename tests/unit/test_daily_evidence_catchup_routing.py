"""V2 Calendar/lineage routing; EOD and business execution explicitly controlled."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest
from _public_catchup_fixture import completion, put, routed_collection
from test_daily_evidence_public_catchup import snapshot
from test_daily_evidence_store_materialization import context as _scripts_context  # noqa: F401
from quant_investor.operations import catchup_binding as binding
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.dashboard_serving_contract import POLICY
from quant_investor.operations.native_input_contract import FIELDS, EXTRAS
from scripts import daily_catchup as batch
from scripts import daily_materialization as materialization
from scripts import daily_dashboard_publication as publication


@pytest.mark.parametrize("policy", sorted(binding.PUBLICATION_POLICIES))
@pytest.mark.parametrize("weekend", [False, True])
def test_immutable_binding_routes_current_and_history(tmp_path, policy, weekend):
    now = datetime(2026, 8, 29 if weekend else 28, 13, tzinfo=timezone.utc)
    root_ref, request = routed_collection(tmp_path, policy=policy, now=now)
    template_before = (tmp_path / request["recipe_ref"]["path"]).read_bytes()
    previous = request["previous_completion_ref"]
    for day in ("20260827", "20260828"):
        derived = binding.derive_catchup_binding(
            workspace=str(tmp_path), request_ref=root_ref, day=day, previous_completion_ref=previous
        )
        current = day == "20260828" and not weekend
        publishing = current and policy == "CURRENT_OBSERVED_CLOSE_ONLY"
        assert derived["binding"]["maintenance_mode"] == ("CURRENT" if current else "HISTORICAL")
        assert derived["binding"]["dashboard_mode"] == (
            "CURRENT_LATEST_EOD" if publishing else "HISTORICAL_CAPTURE"
        )
        assert derived["recipe"]["publish_current_dashboard"] is publishing
        assert derived["request"]["schema_version"] == (
            "cn-daily-production-request.v1" if current else "cn-daily-production-request.v2"
        )
        journal = DailyJournal(str(tmp_path), day)
        with journal.locked():
            ref = binding.persist_catchup_binding(journal=journal, derived=derived)
        assert ref["path"].endswith("/binding.v2.json")
        before = snapshot(tmp_path)
        assert (
            binding.read_catchup_binding(workspace=str(tmp_path), binding_ref=ref)["binding"]
            == derived["binding"]
        )
        assert snapshot(tmp_path) == before
        previous = completion(tmp_path, day, request)
    assert (tmp_path / request["recipe_ref"]["path"]).read_bytes() == template_before


@pytest.mark.parametrize("fault", ["mode", "policy", "extra", "recipe_flag", "path"])
def test_rehashed_binding_tampering_is_rejected(tmp_path, fault):
    root_ref, request = routed_collection(tmp_path)
    derived = binding.derive_catchup_binding(
        workspace=str(tmp_path),
        request_ref=root_ref,
        day="20260827",
        previous_completion_ref=request["previous_completion_ref"],
    )
    journal = DailyJournal(str(tmp_path), "20260827")
    with journal.locked():
        ref = binding.persist_catchup_binding(journal=journal, derived=derived)
    value = deepcopy(derived["binding"])
    if fault == "mode":
        value["maintenance_mode"] = "CURRENT"
    elif fault == "policy":
        value["publication_policy"] = "HISTORICAL_ONLY"
    elif fault == "extra":
        value["allow_unchecked"] = True
    elif fault == "recipe_flag":
        recipe = deepcopy(derived["recipe"])
        recipe["publish_current_dashboard"] = True
        value["recipe_ref"] = put(tmp_path, value["recipe_ref"]["path"], recipe)
    ref = put(tmp_path, ref["path"] if fault != "path" else "binding.v2.json", value)
    before = snapshot(tmp_path)
    with pytest.raises(ContractError):
        binding.read_catchup_binding(workspace=str(tmp_path), binding_ref=ref)
    assert snapshot(tmp_path) == before


def native_inputs(request, *, day, previous, current):
    value = dict.fromkeys(FIELDS | EXTRAS)
    value.update(
        schema_version="cn-daily-native-inputs.v5",
        trade_date=day,
        previous_trade_date=previous,
        calendar_ref=request["calendar_ref"],
        publish_current_dashboard=current,
        decision_recipe_ref=request["release_install_ref"],
        corporate_action_context_ref=request["release_install_ref"],
        dashboard_publication_policy=POLICY,
    )
    return value


@pytest.mark.parametrize("fault", ["flag", "day", "previous", "schema", "policy"])
def test_incomplete_supplied_native_inputs_cannot_override_routing(tmp_path, monkeypatch, fault):
    root_ref, request = routed_collection(tmp_path)
    declaration = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    del declaration["recipes"]["20260827"]
    value = native_inputs(request, day="20260827", previous="20260826", current=False)
    if fault == "flag":
        value["publish_current_dashboard"] = True
    elif fault == "day":
        value["trade_date"] = "20260828"
    elif fault == "previous":
        value["previous_trade_date"] = "20260825"
    elif fault == "schema":
        value["schema_version"] = "cn-daily-native-inputs.v4"
        del value["dashboard_publication_policy"]
    else:
        value["dashboard_publication_policy"] = "unchecked"
    request["recipe_ref"] = put(tmp_path, request["recipe_ref"]["path"], declaration)
    request["day_input_refs"] = {"20260827": put(tmp_path, "supplied.json", value)}
    root_ref = put(tmp_path, root_ref["path"], request)
    _, calls, serving_calls, _ = controlled_routed_native(tmp_path, monkeypatch, request)
    before = snapshot(tmp_path)
    with pytest.raises(ContractError):
        batch.run_upstream_catchup(workspace=str(tmp_path), request_ref=root_ref, synthetic=True)
    assert snapshot(tmp_path) == before and calls == serving_calls == []


def controlled_routed_native(root, monkeypatch, request):
    """Full EOD admission/business callbacks controlled; exact Calendar edge is real."""
    records, calls, serving_calls = {}, [], []

    def seal(day, previous_ref, *, current, request_ref=None):
        previous = PurePosixPath(previous_ref["path"]).parent.name
        native = native_inputs(request, day=day, previous=previous, current=current)
        native_ref = put(root, f"inputs/{day}.json", native)
        ref = completion(root, day, request)
        eod = json.loads((root / ref["path"]).read_bytes())
        eod["native_inputs_ref"] = native_ref
        ref = put(root, ref["path"], eod)
        documents = {
            "handoff": {
                "request_ref": request_ref,
                "calendar_ref": request["calendar_ref"],
                "raw_calendar_ref": request["raw_calendar_ref"],
            },
            "recipe": {"previous_completion_ref": previous_ref},
        }
        records[day] = {
            "recorded_completion": eod,
            "completed_handoff_snapshot": SimpleNamespace(
                document=documents.__getitem__,
                recheck=lambda: None,
            ),
        }
        return ref

    def replay(**kw):
        assert (root / kw["completion_ref"]["path"]).is_file()
        return {
            **kw,
            "native_replay_validated": True,
            "validated_nodes": sorted(EOD_NODE_IDS),
            "synthetic": True,
        }

    def inspect(**kw):
        return records[kw["trade_date"]]

    def execute(**kw):
        req = json.loads((root / kw["request_ref"]["path"]).read_bytes())
        recipe = json.loads((root / req["recipe_ref"]["path"]).read_bytes())
        day = req["target_trade_date"]
        historical = req["schema_version"] == "cn-daily-production-request.v2"
        assert (kw["_catchup_binding_ref"] is not None) is historical
        calls.append((day, historical, recipe["publish_current_dashboard"]))
        ref = seal(
            day,
            recipe["previous_completion_ref"],
            current=recipe["publish_current_dashboard"],
            request_ref=kw["request_ref"],
        )
        return {"status": "COMPLETE", "completion_ref": ref}

    def serving(**kw):
        day = PurePosixPath(kw["completion_ref"]["path"]).parent.name
        serving_calls.append(day)
        return {"status": "PARTIAL", "completion_ref": None}

    from quant_investor.operations import completion_readback

    monkeypatch.setattr(batch, "replay_native_completion", replay)
    monkeypatch.setattr(batch, "inspect_recorded_completion", inspect)
    monkeypatch.setattr(completion_readback, "inspect_recorded_completion", inspect)
    monkeypatch.setattr(materialization, "execute_daily_recipe", execute)
    monkeypatch.setattr(publication, "complete_serving_result", serving)
    return seal, calls, serving_calls, records


@pytest.mark.parametrize("completed_prefix", [False, True])
@pytest.mark.parametrize("policy", sorted(binding.PUBLICATION_POLICIES))
def test_only_current_latest_requires_serving(tmp_path, monkeypatch, policy, completed_prefix):
    root_ref, request = routed_collection(tmp_path, policy=policy)
    seal, calls, serving_calls, _ = controlled_routed_native(tmp_path, monkeypatch, request)
    if completed_prefix:
        # Earlier native EOD originally captured current mode, now consumed as history.
        seal("20260827", request["previous_completion_ref"], current=True)
    result = batch.run_upstream_catchup(
        workspace=str(tmp_path), request_ref=root_ref, synthetic=True
    )
    assert result["days"][0]["business_state"] == "COMPLETE"
    assert result["days"][0]["execution_state"] == (
        "NO_ACTION" if completed_prefix else "SUCCEEDED"
    )
    publishing = policy == "CURRENT_OBSERVED_CLOSE_ONLY"
    assert serving_calls == (["20260828"] if publishing else [])
    assert calls == ([] if completed_prefix else [("20260827", True, False)]) + [
        ("20260828", False, publishing)
    ]
    assert result["business_state"] == ("INCOMPLETE" if publishing else "COMPLETE")
    # Repeat consumes exact completed history; no business callback repeats.
    before, old_calls = snapshot(tmp_path), list(calls)
    repeated = batch.run_upstream_catchup(
        workspace=str(tmp_path), request_ref=root_ref, synthetic=True
    )
    assert repeated["business_state"] == result["business_state"]
    assert snapshot(tmp_path) == before and calls == old_calls


@pytest.mark.parametrize("fault", ["previous_date", "previous_ref", "calendar", "gap"])
def test_completed_history_needs_exact_ancestry_before_writers(tmp_path, monkeypatch, fault):
    root_ref, request = routed_collection(tmp_path)
    seal, calls, serving_calls, records = controlled_routed_native(tmp_path, monkeypatch, request)
    day = "20260828" if fault == "gap" else "20260827"
    seal(day, request["previous_completion_ref"], current=True)
    record = records[day]
    if fault == "previous_date":
        ref = record["recorded_completion"]["native_inputs_ref"]
        value = json.loads((tmp_path / ref["path"]).read_bytes())
        value["previous_trade_date"] = "20260825"
        record["recorded_completion"]["native_inputs_ref"] = put(tmp_path, ref["path"], value)
    elif fault == "previous_ref":
        record["completed_handoff_snapshot"].document("recipe")["previous_completion_ref"] = {
            **request["previous_completion_ref"],
            "sha256": "b" * 64,
        }
    elif fault == "calendar":
        record["completed_handoff_snapshot"].document("handoff")["calendar_ref"] = request[
            "release_install_ref"
        ]
    before = snapshot(tmp_path)
    with pytest.raises(ContractError, match="CATCHUP_COMPLETED_"):
        batch.run_upstream_catchup(workspace=str(tmp_path), request_ref=root_ref, synthetic=True)
    assert calls == serving_calls == [] and snapshot(tmp_path) == before


@pytest.mark.parametrize("current", [False, True])
def test_no_gap_target_cannot_masquerade_as_current_publication(tmp_path, monkeypatch, current):
    root_ref, request = routed_collection(tmp_path)
    seal, calls, serving_calls, _ = controlled_routed_native(tmp_path, monkeypatch, request)
    parent = seal("20260827", request["previous_completion_ref"], current=False)
    target = seal("20260828", parent, current=current)
    request["previous_completion_ref"] = target
    request["recipe_ref"] = put(
        tmp_path,
        "empty-collection.json",
        {
            "schema_version": binding.COLLECTION_SCHEMA_V2,
            "recipes": {},
            "publication_policy": "CURRENT_OBSERVED_CLOSE_ONLY",
        },
    )
    root_ref = put(tmp_path, root_ref["path"], request)
    before = snapshot(tmp_path)
    if current:
        result = batch.run_upstream_catchup(
            workspace=str(tmp_path), request_ref=root_ref, synthetic=True
        )
        assert result["business_state"] == "INCOMPLETE"
        assert serving_calls == ["20260828"]
    else:
        with pytest.raises(ContractError, match="CATCHUP_NATIVE_INPUT_ROUTING_CONFLICT"):
            batch.run_upstream_catchup(
                workspace=str(tmp_path), request_ref=root_ref, synthetic=True
            )
        assert serving_calls == []
    assert calls == [] and snapshot(tmp_path) == before
