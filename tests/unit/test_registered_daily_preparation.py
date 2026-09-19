"""Native registered source -> v6 request; outer EOD/install admission is controlled."""

from datetime import timedelta
import json

import pytest

from _registered_event_fixture import build, ref, NOW
from _daily_preparation_fixture import config, put, snapshot
from test_daily_evidence_requested_session import capture
from quant_investor.operations import daily_preparation, source_slot_inputs as inputs
from quant_investor.operations.source_slot_contract import paths, SourceSlotError
from quant_investor.operations.daily_preparation_contract import PreparationError
from quant_investor.operations.daily_journal import DailyJournal
from scripts import (
    daily_completion_store,
    daily_source_inputs,
    manage_cn_strategy_records as manager,
)
from scripts.daily_production_store_adapter import StoreCloseAdapter


def fixture(root, monkeypatch, *, seed=True):
    data = build(root)
    book = data["book"]
    monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW)
    declaration = manager.command_publish_registered_event_declaration(data["args"])
    book.advance("2026-08-25", publish_events=False)
    baseline_plan = (root / data["baseline_ref"]["path"]).with_name("plan.v1.json")
    release = put(root, "fixtures/release.json", {"synthetic": True})
    adapter = StoreCloseAdapter(
        arguments=data["baseline_arguments"],
        trade_date="20260824",
        plan_ref=ref(root, baseline_plan),
        release_ref=release,
    )
    journal = DailyJournal(str(root), "20260824")
    with journal.locked():
        request = adapter.template()
        journal.begin(request)
        outcome = adapter.probe(request).outcome
        terminal = journal.finish(request, state=outcome.state, output_refs=outcome.output_refs)
    recorded = {
        # Only the native Store reader consumes this bounded test projection.
        # Full native input and all-node EOD admission remain outside this fixture.
        "native_inputs_ref": put(
            root,
            "fixtures/prior-store-reader.json",
            {
                "schema_version": "cn-daily-native-inputs.v4",
                "store_plan_ref": ref(root, baseline_plan),
            },
        ),
        "node_terminal_refs": {"store": terminal["terminal_ref"]},
    }
    completion = put(
        root, "results/operations/daily_production/CN/20260824/completion.v1.json", recorded
    )
    monkeypatch.setattr(
        daily_completion_store,
        "inspect_recorded_completion",
        lambda **kw: {"recorded_completion": recorded},
    )
    cfg, _ = config(root, seed=completion if seed else None)
    policy = json.loads((root / book.policy_path).read_bytes())
    policy.update(
        effective_from="2026-08-01T00:00:00Z",
        event_inbox={
            "pointer_path": str(book.root.relative_to(root)) + "/_event_store/current.v1.json",
            "owner_append_cutoff_local": "15:30:00",
            "timezone": "Asia/Shanghai",
            "sealed_empty_inventory_is_owner_authorized_closure": True,
            "late_event_behavior": "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED",
        },
    )
    cfg["store_policy_ref"] = put(root, "fixtures/standing-policy.json", policy)
    cfg["risk_free_ref"] = put(
        root,
        "fixtures/risk-free.csv",
        (root / "portfolio_dashboard/inputs/cn_govt_bond_yield.csv").read_bytes(),
    )
    config_ref = put(root, "fixtures/registered-source-config.json", cfg)
    monkeypatch.setattr(inputs, "utc_now", lambda: NOW)
    monkeypatch.setattr(inputs, "verify_recipe_static_controls", lambda **kw: {})
    monkeypatch.setattr(daily_preparation, "verify_recipe_static_controls", lambda **kw: {})
    calls = []

    def calendar(**kwargs):
        calls.append("Calendar")
        return capture(kwargs["now"].isoformat())

    monkeypatch.setattr(inputs, "acquire_close_session_authority", calendar)
    data.update(config_ref=config_ref, config=cfg, declaration=declaration, calls=calls)
    return data


def run(root, data, **kwargs):
    return daily_source_inputs.configured_source_inputs(
        workspace=str(root),
        config_ref=data["config_ref"],
        release_install_ref=data["config"]["release_install_ref"],
        synthetic=True,
        **kwargs,
    )


def forbidden(*args, **kwargs):
    pytest.fail("source producer or financial writer was called")


def test_configured_registered_buy_reaches_recipe_without_empty_event(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    current = data["book"].root / "_record_store/current.v1.json"
    event = data["book"].root / "_event_store/current.v1.json"
    before = current.read_bytes(), event.read_bytes()
    monkeypatch.setattr(manager, "command_publish_daily_event_closure", forbidden)
    monkeypatch.setattr(manager, "recover_planned_daily_event", forbidden)
    monkeypatch.setattr(daily_source_inputs.capture, "capture_benchmark_close", forbidden)
    result = run(tmp_path, data, mode="provision")
    assert result["mode"] == "REQUEST_AVAILABLE" and data["calls"] == ["Calendar"]
    selected = paths(data["config_ref"], "20260825", 2)
    assert not (tmp_path / paths(data["config_ref"], "20260825")["plan"]).exists()
    plan = json.loads((tmp_path / selected["plan"]).read_bytes())
    assert plan["event_operation"] == "USE_REGISTERED_DECLARATION"
    assert plan["registered_event_declaration_ref"] == data["declaration"]["declaration_ref"]
    commitment = json.loads((tmp_path / result["preparation_commitment_ref"]["path"]).read_bytes())
    assert commitment["schema_version"] == "cn-daily-preparation-commitment.v2"
    request = json.loads((tmp_path / result["request_ref"]["path"]).read_bytes())
    collection = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    recipe = collection["recipes"]["20260825"]
    assert recipe["schema_version"] == "cn-daily-execute-recipe.v6"
    assert recipe["registered_event_declaration_ref"] == plan["registered_event_declaration_ref"]
    assert recipe["store_preimages"]["store_pointer_ref"] is None
    assert before == (current.read_bytes(), event.read_bytes())
    after = snapshot(tmp_path)
    assert run(tmp_path, data) == result
    assert snapshot(tmp_path) == after


def test_registered_source_without_previous_eod_blocks_before_producers(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch, seed=False)
    monkeypatch.setattr(daily_source_inputs, "_produce_sources", forbidden)
    with pytest.raises(PreparationError, match="PREVIOUS_EOD_REQUIRED"):
        run(tmp_path, data, mode="provision")


def test_registered_request_crash_repairs_original_bytes_next_day(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    original = daily_preparation.JournalStorage.write

    def crash(storage, path, raw, **kwargs):
        if path.endswith("/preparation/" + data["config_ref"]["sha256"] + "/request.json"):
            raise OSError("synthetic request-write interruption")
        return original(storage, path, raw, **kwargs)

    monkeypatch.setattr(daily_preparation.JournalStorage, "write", crash)
    with pytest.raises(OSError, match="synthetic"):
        run(tmp_path, data, mode="provision")
    selected = paths(data["config_ref"], "20260825", 2)
    original_commitment = (tmp_path / selected["commitment"]).read_bytes()
    monkeypatch.setattr(daily_preparation.JournalStorage, "write", original)
    monkeypatch.setattr(inputs, "utc_now", lambda: NOW + timedelta(days=1))
    monkeypatch.setattr(inputs, "acquire_close_session_authority", forbidden)
    monkeypatch.setattr(daily_preparation, "_native_sources", forbidden)
    result = run(tmp_path, data, mode="provision", no_providers=True)
    assert result["trade_date"] == "20260825"
    assert (tmp_path / selected["commitment"]).read_bytes() == original_commitment


@pytest.mark.parametrize("fault", ["plan", "commitment"])
def test_source_versions_cannot_be_combined(tmp_path, monkeypatch, fault):
    data = fixture(tmp_path, monkeypatch)
    run(tmp_path, data, mode="provision")
    leaf = paths(data["config_ref"], "20260825")[fault]
    put(tmp_path, leaf, {"schema_version": "synthetic-conflicting-version"})
    with pytest.raises((SourceSlotError, PreparationError), match="VERSION_CONFLICT"):
        if fault == "plan":
            run(tmp_path, data)
        else:
            cal = inputs.read_calendar(
                inputs.SourceContext(
                    str(tmp_path), data["config_ref"], data["config"]["release_install_ref"]
                ),
                "20260825",
            )
            daily_preparation.prepare_daily_request(
                workspace=str(tmp_path),
                config_ref=data["config_ref"],
                calendar_ref=cal["calendar_ref"],
                raw_calendar_ref=cal["raw_calendar_ref"],
                now=NOW,
            )
