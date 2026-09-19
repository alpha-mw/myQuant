"""Public batch orchestration with real Calendar/derivation; native writers controlled."""

from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations import catchup_binding as binding
from _public_catchup_fixture import collection, completion, put, routed_collection
from test_daily_evidence_store_materialization import context as _scripts_context  # noqa: F401
from scripts import daily_catchup as batch
from scripts import daily_materialization as materialization
from scripts.daily_production import dispatch_daily_request


def snapshot(root):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("after", [1, 2, 3])
@pytest.mark.parametrize("factory", [collection, routed_collection])
def test_each_interrupted_binding_write_recovers_same_bytes(tmp_path, monkeypatch, after, factory):
    root_ref, request = factory(tmp_path)
    derived = binding.derive_catchup_binding(
        workspace=str(tmp_path),
        request_ref=root_ref,
        day="20260827",
        previous_completion_ref=request["previous_completion_ref"],
    )
    journal = DailyJournal(str(tmp_path), "20260827")
    original = journal.storage.write
    count = 0

    def interrupted(path, raw):
        nonlocal count
        result = original(path, raw)
        count += 1
        if count == after:
            raise OSError("synthetic interruption after durable write")
        return result

    with journal.locked():
        monkeypatch.setattr(journal.storage, "write", interrupted)
        with pytest.raises(OSError):
            binding.persist_catchup_binding(journal=journal, derived=derived)
        monkeypatch.setattr(journal.storage, "write", original)
        recovered = binding.derive_catchup_binding(
            workspace=str(tmp_path),
            request_ref=root_ref,
            day="20260827",
            previous_completion_ref=request["previous_completion_ref"],
        )
        ref = binding.persist_catchup_binding(journal=journal, derived=recovered)
    assert (
        binding.read_catchup_binding(workspace=str(tmp_path), binding_ref=ref)["binding"]
        == derived["binding"]
    )
    before = snapshot(tmp_path)
    # A later mutable Store head cannot change recorded derivation.
    put(tmp_path, binding.STORE_CURRENT, {"unrelated_later_head": True})
    assert (
        binding.read_catchup_binding(workspace=str(tmp_path), binding_ref=ref)["binding"]
        == derived["binding"]
    )
    with journal.locked():
        assert binding.persist_catchup_binding(journal=journal, derived=recovered) == ref
    assert all(snapshot(tmp_path)[p] == raw for p, raw in before.items())


def test_orphan_conflict_is_not_overwritten(tmp_path):
    root_ref, request = collection(tmp_path)
    derived = binding.derive_catchup_binding(
        workspace=str(tmp_path),
        request_ref=root_ref,
        day="20260827",
        previous_completion_ref=request["previous_completion_ref"],
    )
    put(tmp_path, derived["binding"]["recipe_ref"]["path"], {"wrong": "orphan"})
    before = snapshot(tmp_path)
    journal = DailyJournal(str(tmp_path), "20260827")
    with journal.locked(), pytest.raises(ContractError, match="IMMUTABLE_CONFLICT"):
        binding.persist_catchup_binding(journal=journal, derived=derived)
    assert all(snapshot(tmp_path)[p] == raw for p, raw in before.items())
    assert not (tmp_path / derived["binding"]["execution_request_ref"]["path"]).exists()


def controlled_native(root, monkeypatch, scope, *, fail_day=None, publication_failure=False):
    owners, calls = {}, []
    fail = [fail_day]

    def replay(**kwargs):
        assert (root / kwargs["completion_ref"]["path"]).is_file()
        return {
            **kwargs,
            "native_replay_validated": True,
            "validated_nodes": sorted(EOD_NODE_IDS),
            "synthetic": False,
        }

    def inspect(**kwargs):
        return {
            "completed_handoff_snapshot": SimpleNamespace(
                document=lambda role: {"request_ref": owners[kwargs["trade_date"]]}
            )
        }

    def execute(**kwargs):
        checked = binding.read_catchup_binding(
            workspace=str(root), binding_ref=kwargs["_catchup_binding_ref"]
        )
        assert kwargs["request_ref"] == checked["binding"]["execution_request_ref"]
        day = checked["binding"]["trade_date"]
        calls.append(day)
        if day == fail[0]:
            return {"status": "BLOCKED", "completion_ref": None}
        owners[day] = kwargs["request_ref"]
        ref = completion(root, day, scope)
        return {"status": "COMPLETE", "completion_ref": ref}

    monkeypatch.setattr(batch, "replay_native_completion", replay)
    monkeypatch.setattr(batch, "inspect_recorded_completion", inspect)
    # This fixture controls native EOD admission and has no native-input closure.
    # Its post-EOD serving admission belongs to the same explicit native seam.
    from scripts import daily_dashboard_publication as publication

    def serving(**kwargs):
        return {
            "status": "PARTIAL" if publication_failure else "COMPLETE",
            "completion_ref": None if publication_failure else kwargs["completion_ref"],
        }

    monkeypatch.setattr(publication, "complete_serving_result", serving)
    monkeypatch.setattr(materialization, "execute_daily_recipe", execute)
    return calls, fail, owners, replay


def test_public_cli_two_missing_days_interruption_repeat_and_no_writes(tmp_path, monkeypatch):
    from quant_investor.cli import daily_production as public

    root_ref, request = collection(tmp_path)
    calls, fail, _, replay = controlled_native(tmp_path, monkeypatch, request, fail_day="20260828")

    @contextmanager
    def installed(**kwargs):
        yield {"daily_close": dispatch_daily_request, "completion_replay": replay}

    monkeypatch.setattr(public, "verified_native_context", installed)

    def run():
        return public.run_daily_close(
            workspace=str(tmp_path),
            request_path=root_ref["path"],
            expected_request_sha256=root_ref["sha256"],
            release_repository_root="/controlled-install",
            release_install_input_path=request["release_install_ref"]["path"],
            expected_release_install_input_sha256=request["release_install_ref"]["sha256"],
        )

    first = run()
    assert first["execution_state"] == "PARTIAL"
    assert [r["execution_state"] for r in first["days"]] == ["SUCCEEDED", "BLOCKED"]
    prefix = DailyJournal(str(tmp_path), "20260827").root
    before = {p: raw for p, raw in snapshot(tmp_path).items() if str(tmp_path / prefix) in p}
    fail[0] = None
    second = run()
    assert second["business_state"] == "COMPLETE"
    assert [r["execution_state"] for r in second["days"]] == ["NO_ACTION", "SUCCEEDED"]
    assert all(snapshot(tmp_path)[p] == raw for p, raw in before.items())
    completed = snapshot(tmp_path)
    assert run()["execution_state"] == "NO_ACTION"
    assert snapshot(tmp_path) == completed
    assert calls == ["20260827", "20260828", "20260828"]


def test_missing_day_ownership_stops_before_any_generated_input(tmp_path, monkeypatch):
    root_ref, request = collection(tmp_path, days=("20260827",))
    calls, _, _, _ = controlled_native(tmp_path, monkeypatch, request)
    before = snapshot(tmp_path)
    result = dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=root_ref,
        release_install_ref=request["release_install_ref"],
    )
    assert result["execution_state"] == "BLOCKED" and calls == []
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "field",
    [
        "root_request_ref",
        "collection_ref",
        "calendar_ref",
        "raw_calendar_ref",
        "previous_completion_ref",
        "previous_store_terminal_ref",
        "previous_store_pointer_ref",
        "recipe_ref",
        "execution_request_ref",
    ],
)
def test_binding_drift_cannot_execute_or_adopt_completion(tmp_path, monkeypatch, field):
    root_ref, request = collection(tmp_path)
    derived = binding.derive_catchup_binding(
        workspace=str(tmp_path),
        request_ref=root_ref,
        day="20260827",
        previous_completion_ref=request["previous_completion_ref"],
    )
    journal = DailyJournal(str(tmp_path), "20260827")
    with journal.locked():
        ref = binding.persist_catchup_binding(journal=journal, derived=derived)
    corrupted = deepcopy(derived["binding"])
    corrupted[field]["sha256"] = "0" * 64
    ref = put(tmp_path, ref["path"], corrupted)
    before = snapshot(tmp_path)
    with pytest.raises((ContractError, OSError)):
        materialization.execute_daily_recipe(
            workspace=str(tmp_path),
            request_ref=derived["binding"]["execution_request_ref"],
            _catchup_binding_ref=ref,
        )
    assert snapshot(tmp_path) == before
    calls, _, _, _ = controlled_native(tmp_path, monkeypatch, request)
    completion(tmp_path, "20260827", request)
    with pytest.raises((ContractError, OSError)):
        dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=root_ref,
            release_install_ref=request["release_install_ref"],
        )
    assert calls == []


def test_direct_historical_execute_and_free_calendar_reject_before_writes(tmp_path):
    root_ref, request = collection(tmp_path)
    derived = binding.derive_catchup_binding(
        workspace=str(tmp_path),
        request_ref=root_ref,
        day="20260827",
        previous_completion_ref=request["previous_completion_ref"],
    )
    ref = put(tmp_path, derived["binding"]["execution_request_ref"]["path"], derived["request"])
    before = snapshot(tmp_path)
    with pytest.raises(ContractError, match="CATCHUP_BINDING_REQUIRED"):
        materialization.execute_daily_recipe(workspace=str(tmp_path), request_ref=ref)
    with pytest.raises(TypeError):
        materialization.execute_daily_recipe(
            workspace=str(tmp_path),
            request_ref=ref,
            _historical_calendar_input={"calendar_ref": request["calendar_ref"]},
        )
    assert snapshot(tmp_path) == before


def test_bound_execute_uses_native_historical_maintenance_and_handoff(tmp_path, monkeypatch):
    from test_daily_evidence_execute_wiring import setup
    from test_daily_evidence_requested_session import capture
    from _public_catchup_fixture import bind_existing_recipe

    ref, events = setup(tmp_path, monkeypatch)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    recipe = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    cal = capture("2026-09-07T20:20:00+08:00")
    raw_ref = put(tmp_path, "original-calendar.raw", cal.raw_response_bytes)
    cal_ref = put(
        tmp_path,
        "original-calendar.json",
        {**cal.receipt, "raw_response_path": str(tmp_path / raw_ref["path"])},
    )
    generated, binding_ref, derived = bind_existing_recipe(
        tmp_path, recipe_value=recipe, calendar_ref=cal_ref, raw_calendar_ref=raw_ref
    )
    handoff = {"path": "handoff.json", "sha256": "a" * 64}

    def publish(**kwargs):
        assert kwargs["_catchup_binding_ref"] == binding_ref
        assert kwargs["request_ref"] == generated
        events.append("bound-handoff")
        return handoff

    def maintenance(**kwargs):
        assert "_expected_target_trade_date" not in kwargs
        assert kwargs["_historical_calendar_input"] == binding.historical_input(derived["binding"])
        events.append("historical-maintenance")
        kwargs["core_completed"]({})
        return {"status": "READY"}

    monkeypatch.setattr(
        "quant_investor.operations.maintenance_handoff.publish_maintenance_handoff", publish
    )
    monkeypatch.setattr(
        "quant_investor.market.daily_maintenance.run_cn_daily_maintenance", maintenance
    )
    monkeypatch.setattr(
        materialization,
        "read_maintenance_handoff",
        lambda **kwargs: {"handoff": {"request_ref": generated}, "recipe": derived["recipe"]},
    )
    result = materialization.execute_daily_recipe(
        workspace=str(tmp_path),
        request_ref=generated,
        synthetic=True,
        _catchup_binding_ref=binding_ref,
    )
    assert result["status"] == "COMPLETE"
    assert events == [
        "install",
        "store",
        "previous",
        "loop",
        "historical-maintenance",
        "top100",
        "bound-handoff",
        "settlement",
        "report",
        "materialize",
        "seal",
    ]


def test_completed_day_without_controller_binding_is_adopted(tmp_path, monkeypatch):
    root_ref, request = collection(tmp_path)
    calls, _, _, _ = controlled_native(tmp_path, monkeypatch, request)
    completion(tmp_path, "20260827", request)
    result = dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=root_ref,
        release_install_ref=request["release_install_ref"],
    )
    assert [r["execution_state"] for r in result["days"]] == ["NO_ACTION", "SUCCEEDED"]
    assert calls == ["20260828"]
    assert not (tmp_path / binding.binding_path("20260827", root_ref)).exists()


def test_wrong_binding_root_or_v1_private_route_reject_before_native_calls(tmp_path):
    root_ref, request = collection(tmp_path)
    derived = binding.derive_catchup_binding(
        workspace=str(tmp_path),
        request_ref=root_ref,
        day="20260827",
        previous_completion_ref=request["previous_completion_ref"],
    )
    journal = DailyJournal(str(tmp_path), "20260827")
    with journal.locked():
        bound_ref = binding.persist_catchup_binding(journal=journal, derived=derived)
    other = put(
        tmp_path,
        "other-root.json",
        {
            **request,
            "target_trade_date": "20260827",
            "recipe_ref": put(
                tmp_path,
                "other-collection.json",
                {
                    "schema_version": binding.COLLECTION_SCHEMA,
                    "recipes": {
                        "20260827": json.loads(
                            (tmp_path / request["recipe_ref"]["path"]).read_bytes()
                        )["recipes"]["20260827"]
                    },
                },
            ),
        },
    )
    other_derived = binding.derive_catchup_binding(
        workspace=str(tmp_path),
        request_ref=other,
        day="20260827",
        previous_completion_ref=request["previous_completion_ref"],
    )
    with journal.locked():
        other_ref = binding.persist_catchup_binding(journal=journal, derived=other_derived)
    with pytest.raises(ContractError, match="CATCHUP_REQUEST_MISMATCH"):
        materialization.execute_daily_recipe(
            workspace=str(tmp_path),
            request_ref=derived["binding"]["execution_request_ref"],
            _catchup_binding_ref=other_ref,
        )
    ordinary = put(
        tmp_path,
        "ordinary.json",
        {**derived["request"], "schema_version": "cn-daily-production-request.v1"},
    )
    before = snapshot(tmp_path)
    with pytest.raises(ContractError, match="CATCHUP_BINDING_REQUIRED"):
        materialization.execute_daily_recipe(
            workspace=str(tmp_path), request_ref=ordinary, _catchup_binding_ref=bound_ref
        )
    assert snapshot(tmp_path) == before


def test_finalized_historical_maintenance_recovers_under_original_lock(tmp_path, monkeypatch):
    from test_daily_evidence_historical_dispatch import context
    from test_daily_evidence_production_request import recipe
    from _public_catchup_fixture import bind_existing_recipe
    from quant_investor.market import daily_maintenance as daily
    from quant_investor.operations.maintenance_readback import locked_finalized_maintenance_replay

    args, seen, _ = context(tmp_path)
    root = args["workspace_root"]
    result = daily.run_cn_daily_maintenance(**args)
    value = recipe()
    value["target_trade_date"] = "20260820"
    value["research_sources"]["as_of"] = "2026-08-20T13:30:00Z"
    value["previous_completion_ref"][
        "path"
    ] = "results/operations/daily_production/CN/20260819/completion.v1.json"
    historical = args["_historical_calendar_input"]
    _, bound_ref, _ = bind_existing_recipe(
        root,
        recipe_value=value,
        calendar_ref=historical["calendar_ref"],
        raw_calendar_ref=historical["raw_calendar_ref"],
    )
    # The released real maintenance lock/claim/core are read, never reacquired recursively.
    seen.clear()
    before = snapshot(root)
    with locked_finalized_maintenance_replay(
        workspace=str(root),
        run_root="data/private/cn_daily_maintenance",
        run_date="20260820",
        _catchup_binding_ref=bound_ref,
    ) as recovered:
        assert recovered["core_completion_ref"] == result["core_completion_ref"]
        assert recovered["target_date"] == "20260820"
    assert seen == [] and snapshot(root) == before


def test_missing_successor_reports_completed_prefix_without_writers(tmp_path, monkeypatch):
    root_ref, request = collection(tmp_path, days=("20260827",))
    completion(tmp_path, "20260827", request)
    calls, _, _, _ = controlled_native(tmp_path, monkeypatch, request)
    before = snapshot(tmp_path)
    result = dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=root_ref,
        release_install_ref=request["release_install_ref"],
    )
    assert [r["execution_state"] for r in result["days"]] == ["NO_ACTION", "BLOCKED"]
    assert calls == [] and snapshot(tmp_path) == before


def test_publication_failure_stops_catchup_before_successor(tmp_path, monkeypatch):
    root_ref, request = collection(tmp_path)
    calls, _, _, _ = controlled_native(tmp_path, monkeypatch, request, publication_failure=True)
    result = dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=root_ref,
        release_install_ref=request["release_install_ref"],
    )
    assert result["business_state"] == "INCOMPLETE"
    assert [row["execution_state"] for row in result["days"]] == ["PARTIAL", "BLOCKED"]
    assert all(row["completion_ref"] is None for row in result["days"])
    assert calls == ["20260827"]
