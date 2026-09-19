"""Synthetic native BUY and two-book corporate evidence; no providers or full DAG."""

from copy import deepcopy
from decimal import Decimal
import json

import pytest

from _registered_event_fixture import build, ref, NOW
from _native_corporate_fixture import context_ref, put, STRATEGY
from test_daily_evidence_requested_session import capture
from test_registered_close_native import inventory
from scripts import manage_cn_strategy_records as manager
from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter
from quant_investor.operations.corporate_adapter import (
    CorporateReconciliationAdapter,
    derive_corporate_projection,
)
from quant_investor.operations.corporate_actions import CorporateActionEvidence
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.decision_recipe import publish_decision_recipe
from quant_investor.intelligence.registered_transition import KIND
from quant_investor.contracts import validate_artifact
from quant_investor.strategy_records import registered_event_contracts as contracts

AS_OF = "2026-08-25T13:30:00Z"


def prepare(root, monkeypatch, *, buy_symbol="002463.SZ"):
    fixture = build(root, buy_symbol=buy_symbol)
    book = fixture["book"]
    monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW)
    declared = manager.command_publish_registered_event_declaration(fixture["args"])
    args = book.advance("2026-08-25", publish_events=False)
    args["calendar_receipt_sha"] = put(
        root, "fixtures/registered-calendar.json", capture(NOW.isoformat()).receipt
    )["sha256"]
    args["calendar_receipt_path"] = root / "fixtures/registered-calendar.json"
    args["registered_event_declaration_ref"] = declared["declaration_ref"]
    planned = prepare_store_plan(args)
    plan = {"path": planned["plan_path"], "sha256": planned["plan_sha256"]}
    request = put(root, "fixtures/research.json", {"as_of": AS_OF, "strategy_id": STRATEGY})
    journal = DailyJournal(str(root), "20260825")
    with journal.locked():
        decision = publish_decision_recipe(
            journal=journal, research_request_ref=request, store_plan_ref=plan
        )
    context = context_ref(root, book)
    context_value = json.loads((root / context["path"]).read_bytes())
    context_value["as_of"] = AS_OF
    context = put(root, context["path"], context_value)
    snapshot = root / "data/parquet/cn/_snapshots/synthetic-20260825.json"
    adapter = CorporateReconciliationAdapter(
        workspace=str(root),
        journal=journal,
        event_pointer_sha256=book.event_pointer,
        release_ref=put(root, "fixtures/release.json", {"synthetic": True}),
        previous_trade_date="20260824",
        market_refs={
            s: ref(root, snapshot.with_suffix("") / f"serving/bars/symbol={s}/bars.parquet")
            for s in book.stocks
        },
        calendar_ref=ref(root, args["calendar_receipt_path"]),
        market_snapshot_ref=ref(root, snapshot),
        corporate_action_context_ref=context,
        decision_recipe_ref=decision,
        research_request_ref=request,
        store_plan_ref=plan,
        registered_event_declaration_ref=declared["declaration_ref"],
    )
    return fixture, args, adapter


def execute(adapter):
    with adapter.journal.locked():
        adapter.prepare()
        request = adapter.template()
        assert adapter.probe(request).safe_to_execute
        adapter.execute(request)
        outcome = adapter.probe(request).outcome
    documents = {
        role: json.loads((adapter.workspace / reference["path"]).read_bytes())
        for role, reference in outcome.output_refs.items()
        if role != "event_generation"
    }
    return request, outcome, documents


@pytest.mark.parametrize("symbol", ["002463.SZ", "300308.SZ"])
def test_native_two_books_preserve_baseline_and_expose_changed_positions(
    tmp_path, monkeypatch, symbol
):
    fixture, args, adapter = prepare(tmp_path, monkeypatch, buy_symbol=symbol)
    current = fixture["book"].root / "_record_store/current.v1.json"
    before = current.read_bytes()
    _, outcome, documents = execute(adapter)
    assert outcome.state.value == "SUCCEEDED"
    assert current.read_bytes() == before
    transition = validate_artifact(documents["registered_transition"], expected_kind=KIND)[
        "payload"
    ]
    rows = {r["symbol"]: r for r in transition["position_rows"]}
    assert rows[symbol]["change_kind"] == (
        "EXISTING_POSITION_ADD" if symbol == "002463.SZ" else "NEW_POSITION"
    )
    assert Decimal(rows[symbol]["shares_delta"]) == 100
    assert rows[symbol]["risk_execution_state"] == "NON_EXECUTABLE"
    assert transition["financial_admission_state"] == "VALIDATED_REGISTERED_TRANSITION"
    assert transition["risk_readiness_state"] == "OWNER_POLICY_REVALIDATION_REQUIRED"
    assert transition["evidence_level"] == "OWNER_DECLARED"
    assert transition["broker_statement_verified"] is False
    assert transition["fee_evidence_level"] == contracts.FEE_STATUS
    assert transition["prospective"] is False
    assert all(v is False for v in transition["authority"].values())
    assert Decimal(transition["cash_delta_cny"]) < 0
    baseline = documents["reconciliation"]["payload"]
    assert [r["symbol"] for r in baseline["company_rows"]] == ["002463.SZ"]
    assert transition["decision_baseline_pointer_ref"] == fixture["baseline_ref"]
    projection = documents["financial_events"]
    assert projection["schema_version"] == "cn-corporate-financial-projection.v3"
    assert projection["event_closure"] is None
    assert projection["financial_event_state"] == "REGISTERED_OWNER_DECLARED_FACTS"
    if symbol != "002463.SZ":
        assert rows[symbol]["baseline_position_state"] == "ABSENT"
        assert rows[symbol]["avg_cost_before"] is None
        assert rows["002463.SZ"]["change_kind"] == "UNCHANGED"
        assert rows["002463.SZ"]["risk_execution_state"] == "NOT_EVALUATED"
    # The unchanged native C1 writer can close valuation despite policy revalidation.
    store_adapter = StoreCloseAdapter(
        arguments=args,
        trade_date=adapter.trade_date,
        plan_ref=adapter.binding["store_plan_ref"],
        release_ref=adapter.release_ref,
    )
    store_adapter.execute(store_adapter.template())
    assert store_adapter.probe(store_adapter.template()).outcome.state.value == "SUCCEEDED"


def test_frozen_replay_does_not_consult_heads_or_reset_clock(tmp_path, monkeypatch):
    fixture, _, adapter = prepare(tmp_path, monkeypatch, buy_symbol="300308.SZ")
    request, outcome, documents = execute(adapter)
    for name in (
        "_record_store/current.v1.json",
        "_event_store/current.v1.json",
    ):
        (fixture["book"].root / name).unlink()
    (tmp_path / "data/parquet/cn/_latest.json").unlink()
    (tmp_path / "data/parquet/cn/benchmarks/_latest.json").unlink()
    before = inventory(tmp_path)
    evidence = CorporateActionEvidence(
        workspace=str(tmp_path), trade_date=adapter.trade_date, recipe=adapter.corporate_recipe
    )
    derived = derive_corporate_projection(evidence=evidence, recipe=adapter.corporate_recipe)
    assert derived[0] == documents["financial_events"]
    assert derived[3] == documents["registered_transition"]
    with adapter.journal.locked():
        adapter.prepare()
    assert adapter.probe(request).outcome == outcome
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "fault",
    [
        "declaration",
        "projection",
        "transition",
        "legacy_schema",
        "future_custody",
        "before_registration",
        "market_union",
    ],
)
def test_registered_report_rejects_missing_or_changed_evidence(tmp_path, monkeypatch, fault):
    _, _, adapter = prepare(tmp_path, monkeypatch, buy_symbol="300308.SZ")
    request, outcome, _ = execute(adapter)
    recipe = deepcopy(adapter.corporate_recipe)
    if fault in {"projection", "transition"}:
        role = "financial_events" if fault == "projection" else "registered_transition"
        path = tmp_path / outcome.output_refs[role]["path"]
        raw = json.loads(path.read_bytes())
        (raw if role == "financial_events" else raw["payload"])["risk_readiness_state"] = "READY"
        put(tmp_path, str(path.relative_to(tmp_path)), raw)
        with pytest.raises(ContractError, match="READBACK_CONFLICT"):
            adapter.probe(request)
        return
    if fault == "declaration":
        recipe["registered_event_declaration_ref"] = put(tmp_path, "fixtures/unbound.json", {})
    elif fault == "legacy_schema":
        recipe["schema_version"] = "cn-corporate-recipe.v2"
    elif fault == "future_custody":
        recipe["custody_at"] = "2099-01-01T00:00:00Z"
    elif fault == "before_registration":
        recipe["custody_at"] = "2026-08-25T12:19:59Z"
    elif fault == "market_union":
        recipe["market_refs"].pop("300308.SZ")
    evidence = CorporateActionEvidence(
        workspace=str(tmp_path), trade_date=adapter.trade_date, recipe=recipe
    )
    with pytest.raises(ContractError):
        derive_corporate_projection(evidence=evidence, recipe=recipe)


def test_named_corporate_event_for_new_position_conflicts_before_store(tmp_path, monkeypatch):
    fixture, _, adapter = prepare(tmp_path, monkeypatch, buy_symbol="300308.SZ")
    announcement = put(tmp_path, "fixtures/announcement.json", {"synthetic": True})
    event_ref = put(
        tmp_path,
        "fixtures/current-corporate.json",
        {
            "schema_version": "cn-corporate-action-event.v1",
            "event_id": "synthetic-new-position-action",
            "symbol": "300308.SZ",
            "kind": "DIVIDEND",
            "effective_trade_date": "20260825",
            "announced_at": "2026-08-24T08:00:00Z",
            "announcement_ref": announcement,
            "accounting_records": None,
        },
    )
    events = put(
        tmp_path,
        "fixtures/current-corporate-events.json",
        {
            "schema_version": "cn-corporate-action-events.v1",
            "strategy_id": STRATEGY,
            "as_of": AS_OF,
            "event_refs": [event_ref],
        },
    )
    context = json.loads(
        (tmp_path / adapter.binding["corporate_action_context_ref"]["path"]).read_bytes()
    )
    context["named_events_ref"] = events
    adapter.binding["corporate_action_context_ref"] = put(
        tmp_path, "fixtures/conflicting-context.json", context
    )
    pointer = fixture["book"].root / "_record_store/current.v1.json"
    before = pointer.read_bytes(), pointer.stat().st_mtime_ns
    with adapter.journal.locked(), pytest.raises(ContractError, match="NAMED_CORPORATE_CONFLICT"):
        adapter.prepare()
    assert (pointer.read_bytes(), pointer.stat().st_mtime_ns) == before


def test_current_empty_and_registered_facts_cannot_share_projection(tmp_path, monkeypatch):
    from quant_investor.strategy_records import event_store

    fixture, _, adapter = prepare(tmp_path, monkeypatch)
    # Publish deliberately contradictory synthetic evidence with the fixture's
    # native Event helper. Production declaration/empty producers forbid this.
    book = fixture["book"]
    owner = put(tmp_path, "fixtures/conflicting-empty-owner.json", {"synthetic": True})
    policy = {"path": book.policy_path, "sha256": book.policy_sha}
    closure = event_store.build_empty_closure(
        trade_date="2026-08-25",
        sealed_at="2026-08-25T12:30:00Z",
        cutoff_at="2026-08-25T07:30:00Z",
        policy_ref=policy,
        owner_declaration_ref=owner,
        source_receipt_ref=None,
    )
    result = event_store.publish_generation(
        book.root / "_event_store",
        generation_id="synthetic-conflicting-empty",
        generated_at="2026-08-25T12:30:00Z",
        expected_pointer_sha256=book.event_pointer,
        closures=[*book.closures, closure],
        policy_ref=policy,
    )
    adapter.event_sha = result["pointer_sha256"]
    with adapter.journal.locked(), pytest.raises(ContractError, match="EMPTY_EVENT_CONFLICT"):
        adapter.prepare()
