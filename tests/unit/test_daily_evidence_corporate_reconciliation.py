"""New full-window corporate node over native pre-close book and Market fixtures."""

import hashlib
import json

import pytest

from _native_corporate_fixture import corporate_book, context_ref, put, AS_OF, STRATEGY
from _native_daily_store_fixture import DAYS
from quant_investor.operations.corporate_adapter import CorporateReconciliationAdapter
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.decision_recipe import publish_decision_recipe
from scripts.daily_production_store_adapter import prepare_store_plan


def prepare(
    root,
    *,
    event=True,
    posting=True,
    review=False,
    changed=False,
    cash_delta=100,
    mixed=False,
    share_multiplier=1,
    prepared_arguments=None,
):
    book, event_ref = corporate_book(
        root,
        posting=posting,
        cash_delta=cash_delta,
        mixed=mixed,
        share_multiplier=share_multiplier,
        kind="SPLIT" if share_multiplier != 1 else "DIVIDEND",
    )
    args = book.advance(
        DAYS[0],
        adjustment_factors={
            book.stocks[0]: {
                "2026-08-20": 1,
                "2026-08-21": 2 if changed else 1,
                DAYS[0]: 2 if changed else 1,
            }
        },
    )
    if prepared_arguments is not None:
        prepared_arguments.update(args)
    context = context_ref(root, book, event_ref if event else None, review=review)
    planned = prepare_store_plan(args)
    plan = {"path": planned["plan_path"], "sha256": planned["plan_sha256"]}
    request = put(root, "fixtures/research.json", {"as_of": AS_OF, "strategy_id": STRATEGY})
    journal = DailyJournal(str(root), "20260824")
    with journal.locked():
        decision = publish_decision_recipe(
            journal=journal, research_request_ref=request, store_plan_ref=plan
        )

    def ref(path):
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    market = root / "data/parquet/cn/_snapshots/synthetic-20260824.json"
    bars = (
        root
        / "data/parquet/cn/_snapshots/synthetic-20260824/serving/bars"
        / f"symbol={book.stocks[0]}/bars.parquet"
    )
    adapter = CorporateReconciliationAdapter(
        workspace=str(root),
        journal=journal,
        event_pointer_sha256=book.event_pointer,
        release_ref=put(root, "fixtures/release.json", {"synthetic": True}),
        previous_trade_date="20260821",
        market_refs={book.stocks[0]: ref(bars)},
        calendar_ref=ref(book.calendar_path),
        market_snapshot_ref=ref(market),
        corporate_action_context_ref=context,
        decision_recipe_ref=decision,
        research_request_ref=request,
        store_plan_ref=plan,
    )
    return book, adapter


def execute(adapter):
    with adapter.journal.locked():
        adapter.prepare()
        request = adapter.template()
        adapter.execute(request)
        outcome = adapter.probe(request).outcome
    raw = (adapter.workspace / outcome.output_refs["reconciliation"]["path"]).read_bytes()
    return json.loads(raw), request, outcome


@pytest.mark.parametrize(
    "posting,review,state",
    [
        (False, False, "UNCONFIRMED"),
        (True, False, "OWNER_REVIEW_REQUIRED"),
        (True, True, "RECONCILED_RESEARCH_ONLY"),
    ],
)
def test_native_accounting_and_owner_review_are_independent(tmp_path, posting, review, state):
    book, adapter = prepare(tmp_path, posting=posting, review=review)
    current = book.root / "_record_store/current.v1.json"
    before = current.read_bytes()
    report, request, outcome = execute(adapter)
    event = report["payload"]["company_rows"][0]["events"][0]
    assert event["reconciliation_state"] == state
    assert report["payload"]["company_rows"][0]["threshold_state"] == "NON_EXECUTABLE"
    if posting:
        assert event["financial"]["cash_adjustment"]["delta"] == "100.0"
    assert report["payload"]["prospective"] is False
    assert current.read_bytes() == before
    current.unlink()
    (book.root / "_event_store/current.v1.json").unlink()
    inventory = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    with adapter.journal.locked():
        adapter.prepare()
    assert adapter.probe(request).outcome == outcome
    assert inventory == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_full_window_finds_change_even_when_last_daily_pair_is_equal(tmp_path):
    _, adapter = prepare(tmp_path, event=False, posting=False, changed=True)
    report, _, _ = execute(adapter)
    row = report["payload"]["company_rows"][0]
    assert [(r["previous_trade_date"], r["trade_date"]) for r in row["transitions"]] == [
        ("20260820", "20260821")
    ]
    assert row["window_state"] == "VERIFIED"
    assert "NAMED_EVENT_EVIDENCE_MISSING" in row["blocker_codes"]
    assert report["payload"]["summary_state"] == "UNCONFIRMED"


def test_historical_unresolved_action_allows_native_valuation(tmp_path):
    from scripts.daily_production_store_adapter import StoreCloseAdapter

    arguments = {}
    book, corporate = prepare(tmp_path, posting=False, prepared_arguments=arguments)
    before = (book.root / "_record_store/current.v1.json").read_bytes()
    report, _, outcome = execute(corporate)
    row = report["payload"]["company_rows"][0]
    assert outcome.state.value == "SUCCEEDED"
    assert row["events"][0]["financial"]["blocker_codes"] == ["ACCOUNTING_EVIDENCE_MISSING"]
    assert row["events"][0]["reconciliation_state"] == "UNCONFIRMED"
    assert row["threshold_state"] == "NON_EXECUTABLE"
    store = StoreCloseAdapter(
        arguments=arguments,
        trade_date="20260824",
        plan_ref=corporate.binding["store_plan_ref"],
        release_ref=corporate.release_ref,
    )
    store.execute(store.template())
    result = store.probe(store.template()).outcome
    assert result.state.value == "SUCCEEDED"
    committed = json.loads((tmp_path / result.output_refs["completion"]["path"]).read_bytes())
    assert committed["status"] == "COMMITTED"
    assert committed["committed_through"] == "2026-08-24"
    assert (book.root / "_record_store/current.v1.json").read_bytes() != before
    manual = json.loads((tmp_path / result.output_refs["manual"]["path"]).read_bytes())
    assert manual["valuation_trade_date"] == "20260824"
    assert all(manual[k] == 0 for k in ("trade_count", "order_count", "fill_count"))
    assert corporate.probe(corporate.template()).outcome == outcome


@pytest.mark.parametrize("version", [3, 4])
def test_materialization_v3_native_v4_and_frozen_corporate_replay(tmp_path, monkeypatch, version):
    from test_daily_evidence_daily_materialization import context
    from scripts.daily_materialization import materialize_locked
    from scripts.daily_native_registry import NativeDailyRegistry
    from quant_investor.operations import completion_corporate as replay

    journal, recovered, book = context(tmp_path, version=2)
    recipe = recovered["recipe"]
    recipe["schema_version"] = f"cn-daily-execute-recipe.v{version}"
    if version == 4:
        from quant_investor.operations.dashboard_serving_contract import POLICY

        recipe["dashboard_publication_policy"] = POLICY
    recipe["corporate_action_context_ref"] = context_ref(tmp_path, book)
    recovered["handoff"]["recipe_ref"] = put(tmp_path, "fixtures/recipe.json", recipe)
    recovered["handoff_ref"] = put(tmp_path, recovered["handoff_ref"]["path"], recovered["handoff"])
    with journal.locked():
        materialized = materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
        registry = NativeDailyRegistry(
            str(tmp_path), journal.trade_date, materialized.inputs, journal=journal
        )
        request, adapter = registry.resolve("corporate_action_recon", {})
        journal.begin(request)
        adapter.execute(request)
        outcome = adapter.probe(request).outcome
        journal.finish(request, state=outcome.state, output_refs=outcome.output_refs)
    assert materialized.materialization_ref["path"].endswith(f"materialization.v{version}.json")
    document = json.loads((tmp_path / materialized.native_inputs_ref["path"]).read_bytes())
    assert document["schema_version"] == f"cn-daily-native-inputs.v{version + 1}"
    assert document["corporate_action_context_ref"] == recipe["corporate_action_context_ref"]
    from quant_investor.operations.corporate_adapter import corporate_report_is_late

    saved_report = json.loads(
        (tmp_path / outcome.output_refs["reconciliation"]["path"]).read_bytes()
    )
    assert corporate_report_is_late(inputs=document, report=saved_report) is True
    # Outer all-node EOD admission is isolated; node request/terminal, native
    # source replay, report and materialization are real, not a full-DAG claim.
    terminal_paths = list(
        (tmp_path / journal.root / "nodes/corporate_action_recon").rglob("terminal.json")
    )
    assert len(terminal_paths) == 1
    terminal_path = terminal_paths[0]
    recorded = {
        "native_inputs_ref": materialized.native_inputs_ref,
        "node_terminal_refs": {
            "corporate_action_recon": {
                "path": str(terminal_path.relative_to(tmp_path)),
                "sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
            }
        },
    }
    monkeypatch.setattr(
        replay, "inspect_recorded_completion", lambda **kwargs: {"recorded_completion": recorded}
    )
    completion = {"path": "fixture-completion.json", "sha256": "a" * 64}
    current = book.root / "_record_store/current.v1.json"
    current.unlink()
    event_current = book.root / "_event_store/current.v1.json"
    event_current.unlink()
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    projected = replay.replay_completed_corporate_actions(
        workspace=str(tmp_path), trade_date=journal.trade_date, completion_ref=completion
    )
    assert projected["projection"]["schema_version"] == "cn-corporate-financial-projection.v2"
    assert projected["consumer_admission"] is False
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    with journal.locked():
        assert (
            materialize_locked(
                journal=journal, recovered=recovered, auxiliary={"stages": {}}
            ).status
            == "NO_ACTION"
        )


def test_same_day_declared_action_blocks_store_in_fixed_dag(tmp_path):
    from quant_investor.operations.daily_runner import DayRunner
    from quant_investor.operations.daily_contract import EOD_NODE_IDS
    from test_daily_evidence_runner import FixtureAdapter
    from test_daily_evidence_dag_journal import request as fixture_request

    book, adapter = prepare(tmp_path, posting=False)
    event = json.loads((tmp_path / "fixtures/corporate-event.json").read_bytes())
    event["effective_trade_date"] = "20260824"
    ref = put(tmp_path, "fixtures/corporate-event.json", event)
    adapter.binding["corporate_action_context_ref"] = context_ref(tmp_path, book, ref)
    with adapter.journal.locked():
        adapter.prepare()
    calls = []
    adapters = {node: FixtureAdapter(tmp_path, node, calls) for node in EOD_NODE_IDS}
    adapters["corporate_action_recon"] = adapter
    templates = {
        node: {**fixture_request(), "node_id": node, "trade_date": "20260824"}
        for node in EOD_NODE_IDS
    }
    templates["corporate_action_recon"] = adapter.template()
    with adapter.journal.locked():
        runner = DayRunner(str(tmp_path), "20260824", adapters, journal=adapter.journal)
        result = runner.run_locked(templates)
    assert result["nodes"]["corporate_action_recon"]["state"] == "BLOCKED"
    assert "store" not in calls and "dashboard" not in calls
    terminal = result["nodes"]["corporate_action_recon"]["terminal"]
    report = json.loads((tmp_path / terminal["output_refs"]["reconciliation"]["path"]).read_bytes())
    assert report["payload"]["summary_state"] == "CURRENT_FINANCIAL_CONFLICT"


def test_extra_caller_custody_and_duplicate_event_identity_reject(tmp_path):
    from quant_investor.operations.daily_contract import ContractError

    book, adapter = prepare(tmp_path, posting=False)
    context = json.loads((tmp_path / "fixtures/corporate-context.json").read_bytes())
    context["custody_at"] = AS_OF
    adapter.binding["corporate_action_context_ref"] = put(
        tmp_path, "fixtures/invalid-context.json", context
    )
    with adapter.journal.locked(), pytest.raises(ContractError, match="SCHEMA"):
        adapter.prepare()
    del context["custody_at"]
    event = json.loads((tmp_path / "fixtures/corporate-event.json").read_bytes())
    refs = [
        put(tmp_path, "fixtures/event-copy-a.json", event),
        put(tmp_path, "fixtures/event-copy-b.json", event),
    ]
    context["named_events_ref"] = put(
        tmp_path,
        "fixtures/duplicated-events.json",
        {
            "schema_version": "cn-corporate-action-events.v1",
            "strategy_id": STRATEGY,
            "as_of": AS_OF,
            "event_refs": refs,
        },
    )
    adapter.binding["corporate_action_context_ref"] = put(
        tmp_path, "fixtures/duplicate-context.json", context
    )
    with adapter.journal.locked(), pytest.raises(ContractError, match="DUPLICATE_ID"):
        adapter.prepare()


@pytest.mark.parametrize(
    "cash_delta,mixed,code",
    [(0, False, "ACCOUNTING_ZERO_DELTA"), (100, True, "ACCOUNTING_MIXED_TRANSITION")],
)
def test_native_zero_or_mixed_transition_cannot_be_attributed(tmp_path, cash_delta, mixed, code):
    _, adapter = prepare(tmp_path, cash_delta=cash_delta, mixed=mixed)
    report, _, _ = execute(adapter)
    financial = report["payload"]["company_rows"][0]["events"][0]["financial"]
    assert financial["state"] == "UNCONFIRMED"
    assert financial["cost_basis_adjustment"] is None
    assert financial["unattributed_transition"] is not None
    assert financial["blocker_codes"] == [code]


def test_native_share_adjustment_preserves_observed_total_cost_basis(tmp_path):
    from decimal import Decimal

    _, adapter = prepare(tmp_path, cash_delta=0, share_multiplier=2, review=True)
    report, _, _ = execute(adapter)
    event = report["payload"]["company_rows"][0]["events"][0]
    assert event["kind"] == "SPLIT" and event["source_state"] == "SOURCE_DECLARED"
    assert event["reconciliation_state"] == "RECONCILED_RESEARCH_ONLY"
    financial = event["financial"]
    assert Decimal(financial["shares_adjustment"]["delta"]) == 100
    assert Decimal(financial["cost_basis_adjustment"]["delta"]) == 0
    assert Decimal(financial["cash_adjustment"]["delta"]) == 0
    assert report["payload"]["company_rows"][0]["threshold_state"] == "NON_EXECUTABLE"


@pytest.mark.parametrize(
    "fault,code",
    [
        ("ancestry", "ACCOUNTING_ANCESTRY_UNCONFIRMED"),
        ("link", "ACCOUNTING_EVENT_LINK_MISSING"),
        ("policy", "POLICY_BASELINE_UNCONFIRMED"),
    ],
)
def test_native_reconciliation_rejects_wrong_record_link_or_policy_ancestry(tmp_path, fault, code):
    book, adapter = prepare(tmp_path, review=False)
    if fault == "policy":
        context = json.loads((tmp_path / "fixtures/corporate-context.json").read_bytes())
        policy = json.loads((tmp_path / "fixtures/tracking-policy.json").read_bytes())
        policy["store_binding"]["active_record_id"] = "not-registered"
        context["tracking_policy_ref"] = put(tmp_path, "fixtures/unregistered-policy.json", policy)
        adapter.binding["corporate_action_context_ref"] = put(
            tmp_path, "fixtures/policy-context.json", context
        )
    else:
        event = json.loads((tmp_path / "fixtures/corporate-event.json").read_bytes())
        if fault == "ancestry":
            event["accounting_records"]["before_record_id"] = "not-registered"
        else:
            event["event_id"] = "changed-source-event"
        ref = put(tmp_path, "fixtures/changed-event.json", event)
        adapter.binding["corporate_action_context_ref"] = context_ref(tmp_path, book, ref)
    report, _, _ = execute(adapter)
    row = report["payload"]["company_rows"][0]
    assert code in row["blocker_codes"]
    assert row["threshold_state"] == "NON_EXECUTABLE"
    assert report["payload"]["summary_state"] == "UNCONFIRMED"


@pytest.mark.parametrize("fault", ["sha", "symlink"])
def test_bound_announcement_cannot_be_substituted(tmp_path, fault):
    book, adapter = prepare(tmp_path, posting=False)
    path = tmp_path / "fixtures/announcement.json"
    if fault == "sha":
        path.write_bytes(b"{}")
    else:
        target = tmp_path / "fixtures/aliased-announcement.json"
        path.rename(target)
        path.symlink_to(target)
    pointer = book.root / "_record_store/current.v1.json"
    original = (pointer.read_bytes(), pointer.stat().st_mtime_ns)
    from quant_investor.strategy_records.event_store import StrategyEventStoreError

    with adapter.journal.locked(), pytest.raises(StrategyEventStoreError):
        adapter.prepare()
    assert original == (pointer.read_bytes(), pointer.stat().st_mtime_ns)


def test_native_v4_corporate_type_reaches_eod_control_gate(tmp_path):
    from types import SimpleNamespace
    from scripts.daily_completion import _validate_native_context
    from scripts.daily_native_registry import NativeDailyRegistry
    from quant_investor.operations.core_pool import (
        CORE_NODES,
        CoreEvidenceAdapter,
        NativePoolAdapter,
    )
    from quant_investor.operations.research_sources import SOURCE_NODES, ResearchSourceAdapter
    from quant_investor.operations.research_decision import ResearchDecisionAdapter
    from quant_investor.operations.corporate_actions import CorporateActionAdapter
    from scripts.daily_production_store_adapter import StoreCloseAdapter
    from scripts.daily_dashboard_adapter import HistoricalDashboardAdapter
    from quant_investor.operations.daily_contract import ContractError

    _, corporate = prepare(tmp_path, posting=False)
    registry = object.__new__(NativeDailyRegistry)
    registry.runner = SimpleNamespace(journal=corporate.journal)
    registry.inputs = SimpleNamespace(
        corporate_action_context_ref=corporate.binding["corporate_action_context_ref"],
        publish_current_dashboard=False,
    )
    classes = {n: CoreEvidenceAdapter for n in CORE_NODES}
    classes.update({n: ResearchSourceAdapter for n in SOURCE_NODES})
    classes.update(
        top100=NativePoolAdapter,
        decision=ResearchDecisionAdapter,
        store=StoreCloseAdapter,
        dashboard=HistoricalDashboardAdapter,
    )
    # Only exercise the exact-type admission gate; unrelated adapters are
    # uninitialized type tokens and do not stand in for full native replay.
    registry.adapters = {n: object.__new__(kind) for n, kind in classes.items()}
    registry.adapters["corporate_action_recon"] = corporate
    registry.templates = dict.fromkeys(registry.adapters)
    with corporate.journal.locked():
        _validate_native_context(
            registry, native_inputs_ref={"path": "input.json", "sha256": "a" * 64}, synthetic=True
        )
        registry.adapters["corporate_action_recon"] = object.__new__(CorporateActionAdapter)
        with pytest.raises(ContractError, match="ADAPTER_TYPE"):
            _validate_native_context(
                registry,
                native_inputs_ref={"path": "input.json", "sha256": "a" * 64},
                synthetic=True,
            )
