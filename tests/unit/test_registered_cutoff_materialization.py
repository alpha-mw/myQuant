"""Native registered BUY -> cutoff2/native7/corporate3; explicit outer EOD/Factor seams."""

from datetime import datetime, timedelta, timezone
import json

import pytest

from _native_cutoff_sources_fixture import build
from test_registered_close_native import inventory
from quant_investor.operations import research_cutoff, research_timing
from quant_investor.operations.research_request import load_research_request
from scripts import daily_materialization
from scripts.daily_native_inputs import load_native_inputs
from scripts.daily_native_registry import NativeDailyRegistry
from _native_corporate_fixture import put


def clock(monkeypatch):
    state = {"now": datetime(2026, 8, 28, 13, 30, 0, 250000, tzinfo=timezone.utc)}

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return state["now"]

    monkeypatch.setattr(research_timing, "datetime", Clock)
    monkeypatch.setattr(daily_materialization, "datetime", Clock)
    monkeypatch.setattr(research_cutoff, "_clock", lambda: state["now"])

    def align(seconds):
        assert 0 < seconds <= 1
        state["now"] += timedelta(seconds=seconds)

    monkeypatch.setattr(research_cutoff.time, "sleep", align)
    return state


def test_automatic_v4_origin_is_in_cutoff_custody(tmp_path, monkeypatch):
    from quant_investor.operations import automatic_origin
    from quant_investor.operations.catchup_binding import BindingSources
    from quant_investor.operations.daily_contract import ContractError
    from quant_investor.operations.research_file_readback import ResearchFileReadback

    case = build(tmp_path, monkeypatch, registered_buy=True)
    clock(monkeypatch)
    journal, workspace = case["journal"], case["workspace"]
    recovered = case["recovered"]
    handoff = recovered["handoff"]
    request = json.loads((workspace / handoff["request_ref"]["path"]).read_bytes())
    ref = put(
        workspace,
        str(journal.root / "origin-fixture.json"),
        {"synthetic": "outer automatic-resolution seam"},
    )
    source = BindingSources(str(workspace))
    source.raw(ref)
    context = {
        "origin": {"execution_request_ref": handoff["request_ref"]},
        "bound": {"request": request, "recipe": recovered["recipe"], "sources": source},
        "resolution": {"resolution": {"resolved_at": "2026-08-28T13:19:00Z"}, "sources": source},
        "sources": source,
    }

    def read_origin(**kwargs):
        assert kwargs["reference"] == ref
        source.recheck()
        return context

    # This extends the existing controlled Core fixture before cutoff creation.
    # Native origin publication/derivation has separate real financial coverage.
    # No claim of a real automatic controller or real Core handoff is made here.
    handoff.update(schema_version="cn-daily-maintenance-handoff.v4", automatic_origin_ref=ref)
    recovered["handoff_ref"] = put(workspace, recovered["handoff_ref"]["path"], handoff)
    monkeypatch.setattr(automatic_origin, "read_automatic_origin", read_origin)
    observed = []
    original = ResearchFileReadback.source_file

    def record(self, reference, **kwargs):
        if reference == ref:
            observed.append(reference)
        return original(self, reference, **kwargs)

    monkeypatch.setattr(ResearchFileReadback, "source_file", record)
    with journal.locked():
        result = daily_materialization.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    assert result.materialization_ref["path"].endswith("/materialization.v6.json")
    assert observed
    before = inventory(workspace)
    with journal.locked():
        assert (
            daily_materialization.materialize_locked(
                journal=journal, recovered=recovered, auxiliary={"stages": {}}
            ).materialization_ref
            == result.materialization_ref
        )
    assert inventory(workspace) == before
    put(workspace, ref["path"], {"changed": "original automatic evidence"})
    before = inventory(workspace)
    with journal.locked(), pytest.raises(ContractError, match="SOURCE_CHANGED"):
        daily_materialization.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    assert inventory(workspace) == before


@pytest.mark.parametrize("new_position", [False, True])
def test_registered_sources_reach_native_cutoff_and_corporate_node(
    tmp_path, monkeypatch, new_position
):
    case = build(tmp_path, monkeypatch, registered_buy=True, new_position=new_position)
    state = clock(monkeypatch)
    journal, workspace = case["journal"], case["workspace"]
    current = case["book"].root / "_record_store/current.v1.json"
    original = current.read_bytes()
    with journal.locked():
        materialized = daily_materialization.materialize_locked(
            journal=journal,
            recovered=case["recovered"],
            auxiliary={"stages": {}},
        )
    assert materialized.materialization_ref["path"].endswith("/materialization.v6.json")
    value = json.loads((workspace / materialized.native_inputs_ref["path"]).read_bytes())
    assert value["schema_version"] == "cn-daily-native-inputs.v7"
    assert value["registered_event_declaration_ref"] == case["registered"]["declaration_ref"]
    request = load_research_request(workspace=workspace, reference=value["research_request_ref"])
    receipt = request["cutoff"]["receipt"]
    assert receipt["schema_version"] == "cn-daily-research-cutoff.v2"
    assert request["cutoff"]["source_bundle"]["schema_version"] == "cn-daily-acquired-sources.v2"
    roles = {r["role"]: r for r in receipt["source_times"] if r["role"].startswith("REGISTERED_")}
    assert roles["REGISTERED_OWNER_FACT"]["original_time"] == "2026-08-28T02:02:00Z"
    assert roles["REGISTERED_DECLARATION"]["original_time"] == "2026-08-28T12:20:00Z"
    assert roles["REGISTERED_STORE_PUBLICATION"]["original_time"] == "2026-08-28T02:01:00Z"
    assert receipt["prospective"] is False
    assert value["research_request_ref"]["path"].endswith("/research.v3.json")
    day, loaded = load_native_inputs(
        workspace=str(workspace), input_ref=materialized.native_inputs_ref
    )
    assert (
        loaded.store_arguments["registered_event_declaration_ref"]
        == value["registered_event_declaration_ref"]
    )
    with journal.locked():
        registry = NativeDailyRegistry(str(workspace), day, loaded, journal=journal)
        node_request, adapter = registry.resolve("corporate_action_recon", {})
        journal.begin(node_request)
        adapter.execute(node_request)
        outcome = adapter.probe(node_request).outcome
        corporate_terminal = journal.finish(
            node_request, state=outcome.state, output_refs=outcome.output_refs
        )
    assert outcome.state.value == "SUCCEEDED"
    transition = json.loads(
        (workspace / outcome.output_refs["registered_transition"]["path"]).read_bytes()
    )
    changed = [
        r for r in transition["payload"]["position_rows"] if r["policy_revalidation_required"]
    ]
    assert len(changed) == 1
    assert changed[0]["change_kind"] == (
        "NEW_POSITION" if new_position else "EXISTING_POSITION_ADD"
    )
    assert set(value["adjustment_market_refs"]) == {
        r["symbol"] for r in transition["payload"]["position_rows"]
    }
    assert current.read_bytes() == original
    with journal.locked():
        store_request, store_adapter = registry.resolve("store", {})
        journal.begin(store_request)
        store_adapter.execute(store_request)
        store_outcome = store_adapter.probe(store_request).outcome
        store_terminal = journal.finish(
            store_request, state=store_outcome.state, output_refs=store_outcome.output_refs
        )
    _morning_readback(
        case, value, materialized.native_inputs_ref, corporate_terminal, store_terminal, transition
    )
    # Move only the logical fixture clock. Replay must preserve original cutoff
    # bytes and must not require fresh provider sources or a new alignment sleep.
    state["now"] = datetime(2026, 9, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(
        research_cutoff.time, "sleep", lambda *a: pytest.fail("fresh alignment on replay")
    )
    before = inventory(workspace)
    with journal.locked():
        repeated = daily_materialization.materialize_locked(
            journal=journal,
            recovered=case["recovered"],
            auxiliary={"stages": {}},
        )
    assert (
        repeated.status == "NO_ACTION"
        and repeated.native_inputs_ref == materialized.native_inputs_ref
    )
    assert inventory(workspace) == before


def _morning_readback(case, inputs, native_ref, corporate_terminal, store_terminal, transition):
    from quant_investor.operations.morning_risk_sources import MorningRiskSources

    workspace = case["workspace"]
    body = transition["payload"]
    declaration = json.loads(
        (workspace / body["registered_event_declaration_ref"]["path"]).read_bytes()
    )
    writer_manual = json.loads(
        (workspace / declaration["writer_record_refs"]["manual_manifest"]["path"]).read_bytes()
    )
    context = json.loads((workspace / inputs["corporate_action_context_ref"]["path"]).read_bytes())
    trailing = json.loads((workspace / context["tracking_policy_ref"]["path"]).read_bytes())
    changed = next(r for r in body["position_rows"] if r["policy_revalidation_required"])
    symbol = changed["symbol"]
    binding = {
        "pointer_path": str(case["book"].root.relative_to(workspace))
        + "/_record_store/current.v1.json",
        "pointer_sha256": body["writer_pointer_ref"]["sha256"],
        "active_record_id": body["writer_record_id"],
        "ledger_path": declaration["writer_record_refs"]["ledger"]["path"],
        "ledger_sha256": declaration["writer_record_refs"]["ledger"]["sha256"],
        "valuation_trade_date": "20260828",
        "total_value_after_cny": str(writer_manual["total_value_after"]),
    }
    policy = {
        "schema_version": "initial-risk-stop.v1",
        "policy_id": "synthetic-registered-stop",
        "strategy_domain": "aggressive_tech_manufacturing",
        "owner": trailing["owner"],
        "owner_confirmation_recorded_at": "2026-08-28T02:05:00Z",
        "effective_from": "2026-08-28T02:05:00Z",
        "authority": {
            "risk_policy_confirmation": True,
            **dict.fromkeys(
                (
                    "broker",
                    "live_order",
                    "live_execution",
                    "trade",
                    "actual_holdings_mutation",
                    "automatic_human_action",
                ),
                False,
            ),
        },
        "store_binding": binding,
        "market_evidence": {"synthetic": True},
        "stops": [
            {
                "symbol": symbol,
                "company_name": "Synthetic",
                "current_shares": changed["shares_after"],
                "effective_entry_event": "SYNTHETIC_REGISTERED_BUY",
                "effective_entry_trade_date": "20260828",
                "fee_inclusive_avg_cost_cny": changed["avg_cost_after"],
                "initial_stop_state": "CONFIRMED",
                "stop_policy_ref": "synthetic-registered-stop:" + symbol,
                "setting_authority": "OWNER_DELEGATED_TO_CODEX",
                "initial_stop_price_cny": "1.00",
                "price_tick_cny": "0.01",
                "trigger": {
                    "observation": "STRICT_CN_DAILY_CLOSE",
                    "operator": "LESS_THAN_OR_EQUAL",
                    "price_cny": "1.00",
                    "intraday_touch_only": "WARNING_NOT_BREACH",
                    "confirmed_breach_state": "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW",
                    "review_quantity_shares": changed["shares_after"],
                    "automatic_order": False,
                    "automatic_execution": False,
                },
                "calibration": {"synthetic": True},
                "add_policy": "NO_ADD_UNTIL_SEPARATE_I6_ELIGIBILITY_AND_NEW_ANCHOR_CONTRACT",
                "valid_until": (
                    "EARLIEST_OF_OWNER_REVISION_POSITION_REMOVAL_OR_NEW_ENTRY_ADD_ANCHOR"
                ),
            }
        ],
    }
    stop_ref = put(workspace, "cutoff-controls/registered-stop.json", policy)
    recorded = {
        "native_inputs_ref": native_ref,
        "node_terminal_refs": {
            "corporate_action_recon": corporate_terminal["terminal_ref"],
            "store": store_terminal["terminal_ref"],
        },
    }
    completion = put(
        workspace, "results/operations/daily_production/CN/20260828/completion.v1.json", recorded
    )

    def read(reference):
        return MorningRiskSources(
            workspace=workspace,
            recorded=recorded,
            completion_ref=completion,
            policy_refs={"trailing": context["tracking_policy_ref"], "initial_stop": reference},
            quote_requested_at="2026-08-31T01:45:00Z",
        )

    before = inventory(workspace)
    source = read(stop_ref)
    assert {r["position"]["symbol"] for r in source.rows} == set(inputs["adjustment_market_refs"])
    row = next(r for r in source.rows if r["position"]["symbol"] == symbol)
    assert "OWNER_POLICY_REVALIDATION_REQUIRED" in row["trailing_blockers"]
    assert row["owner_stop_blockers"] == [] and row["owner_stop"] == "1.00"
    assert inventory(workspace) == before
    policy["store_binding"]["pointer_sha256"] = "f" * 64
    wrong = put(workspace, "cutoff-controls/registered-stop-wrong-pointer.json", policy)
    row = next(r for r in read(wrong).rows if r["position"]["symbol"] == symbol)
    assert "OWNER_POLICY_REVALIDATION_REQUIRED" in row["owner_stop_blockers"]
    # Only a partial completion document was used for the Morning source boundary;
    # it cannot stand in for EOD admission or remain for materialization recovery.
    (workspace / completion["path"]).unlink()
