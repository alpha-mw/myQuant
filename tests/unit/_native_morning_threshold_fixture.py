"""Isolated native prior-EOD source segment and synthetic owner policies.

Store/Market/Event/Calendar/corporate publishers are real. No full EOD, external
provider, owner approval or live Morning success is claimed by this fixture.
"""

from pathlib import Path
import hashlib

import pandas as pd

from _native_daily_store_fixture import NativeStoreFixture, DAYS
from _native_corporate_fixture import put
from quant_investor.strategy_records.store import load_registered_catalog
from quant_investor.strategy_records.corporate_contracts import POLICY_CALCULATION
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.decision_recipe import publish_decision_recipe
from quant_investor.operations.corporate_adapter import CorporateReconciliationAdapter
from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter


def ref(root, path):
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build(root, *, adjustment_change=False):
    root = Path(root).resolve()
    book = NativeStoreFixture(root, stock_symbols=("002463.SZ",))
    for day in DAYS[:4]:
        args = book.advance(
            day,
            adjustment_factors={
                "002463.SZ": {
                    d: 2 if adjustment_change and d >= "2026-08-26" else 1
                    for d in ["2026-08-20", "2026-08-21", *DAYS[:4]]
                }
            },
        )
    prepared = prepare_store_plan(args)
    plan = {"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]}
    pointer, catalog = load_registered_catalog(book.root)
    baseline = next(r for r in catalog["records"] if r["record_id"] == book.seed.name)
    ledger = pd.read_parquet(book.root / baseline["ledger_path"]).iloc[0]
    binding = {
        "pointer_path": str((book.root / "_record_store/current.v1.json").relative_to(root)),
        "pointer_sha256": ref(root, book.root / "_record_store/current.v1.json")["sha256"],
        "active_record_id": baseline["record_id"],
        "ledger_path": str((book.root / baseline["ledger_path"]).relative_to(root)),
        "ledger_sha256": baseline["ledger_sha256"],
    }
    policy = put(
        root,
        "fixtures/morning/trailing.json",
        {
            "schema_version": "owner-trailing-anchor-policy.v1",
            "policy_id": "synthetic-morning-trailing",
            "strategy_domain": "aggressive_tech_manufacturing",
            "owner": "Synthetic Owner",
            "effective_from": "2026-08-21T11:00:00Z",
            "authority": {
                "research_threshold_calculation": True,
                "paper_risk_reduction_input": True,
                **dict.fromkeys(
                    (
                        "store_mutation",
                        "actual_holdings_mutation",
                        "broker",
                        "live_order",
                        "live_execution",
                        "trade",
                    ),
                    False,
                ),
            },
            "store_binding": binding,
            "calculation": POLICY_CALCULATION,
            "anchors": [
                {
                    "symbol": "002463.SZ",
                    "company_name": "Synthetic",
                    "tracking_start_date": "20260821",
                    "anchor_state": "OWNER_APPROVED_RESET_NO_STRUCTURED_BUY_FOUND",
                    "exclude_pre_anchor_peaks": True,
                }
            ],
            "invalidation": (
                "EARLIEST_OF_OWNER_REVISION_POSITION_REMOVAL_NEW_ENTRY_ADD_OR" "_CORPORATE_ACTION"
            ),
        },
    )
    stop = put(
        root,
        "fixtures/morning/initial-stop.json",
        {
            "schema_version": "initial-risk-stop.v1",
            "policy_id": "synthetic-morning-stop",
            "strategy_domain": "aggressive_tech_manufacturing",
            "owner": "Synthetic Owner",
            "owner_confirmation_recorded_at": "2026-08-21T11:00:00Z",
            "effective_from": "2026-08-21T11:00:00Z",
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
            "store_binding": {
                **binding,
                "valuation_trade_date": "20260821",
                "total_value_after_cny": "1000000",
            },
            "market_evidence": {"synthetic_calibration_context": True},
            "stops": [
                {
                    "symbol": "002463.SZ",
                    "company_name": "Synthetic",
                    "current_shares": int(ledger["shares"]),
                    "effective_entry_event": "SYNTHETIC_OWNER_BASELINE",
                    "effective_entry_trade_date": "20260821",
                    "fee_inclusive_avg_cost_cny": str(ledger["avg_cost"]),
                    "initial_stop_state": "CONFIRMED",
                    "stop_policy_ref": "synthetic-morning-stop:002463.SZ",
                    "setting_authority": "OWNER_DELEGATED_TO_CODEX",
                    "initial_stop_price_cny": "9.00",
                    "price_tick_cny": "0.01",
                    "trigger": {
                        "observation": "STRICT_CN_DAILY_CLOSE",
                        "operator": "LESS_THAN_OR_EQUAL",
                        "price_cny": "9.00",
                        "intraday_touch_only": "WARNING_NOT_BREACH",
                        "confirmed_breach_state": "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW",
                        "review_quantity_shares": int(ledger["shares"]),
                        "automatic_order": False,
                        "automatic_execution": False,
                    },
                    "calibration": {"synthetic": True},
                    "add_policy": "NO_ADD_UNTIL_SEPARATE_I6_ELIGIBILITY_AND_NEW_ANCHOR_CONTRACT",
                    "valid_until": (
                        "EARLIEST_OF_OWNER_REVISION_POSITION_REMOVAL_OR_NEW_ENTRY_ADD" "_ANCHOR"
                    ),
                }
            ],
        },
    )
    as_of = "2026-08-27T13:30:00Z"
    context = put(
        root,
        "fixtures/morning/corporate-context.json",
        {
            "schema_version": "cn-corporate-action-context.v1",
            "strategy_id": "aggressive_tech_manufacturing",
            "as_of": as_of,
            "tracking_policy_ref": policy,
            "named_events_ref": None,
            "anchor_reviews_ref": None,
        },
    )
    research = put(
        root,
        "fixtures/morning/research.json",
        {"as_of": as_of, "strategy_id": "aggressive_tech_manufacturing"},
    )
    release = put(root, "fixtures/morning/release.json", {"synthetic": True})
    journal = DailyJournal(str(root), "20260827")
    with journal.locked():
        decision = publish_decision_recipe(
            journal=journal, research_request_ref=research, store_plan_ref=plan
        )
    market = ref(root, root / "data/parquet/cn/_snapshots/synthetic-20260827.json")
    frame = ref(
        root,
        root
        / (
            "data/parquet/cn/_snapshots/synthetic-20260827/serving/bars/s"
            "ymbol=002463.SZ/bars.parquet"
        ),
    )
    corporate = CorporateReconciliationAdapter(
        workspace=str(root),
        journal=journal,
        release_ref=release,
        event_pointer_sha256=book.event_pointer,
        previous_trade_date="20260826",
        market_refs={"002463.SZ": frame},
        calendar_ref=ref(root, book.calendar_path),
        market_snapshot_ref=market,
        corporate_action_context_ref=context,
        decision_recipe_ref=decision,
        research_request_ref=research,
        store_plan_ref=plan,
    )
    with journal.locked():
        corporate.prepare()
        request = corporate.template()
        journal.begin(request)
        corporate.execute(request)
        outcome = corporate.probe(request).outcome
        corporate_terminal = journal.finish(
            request, state=outcome.state, output_refs=outcome.output_refs
        )["terminal_ref"]
    store = StoreCloseAdapter(
        arguments=args, trade_date="20260827", plan_ref=plan, release_ref=release
    )
    with journal.locked():
        journal.begin(store.template())
        store.execute(store.template())
        store_outcome = store.probe(store.template()).outcome
        store_terminal = journal.finish(
            store.template(), state=store_outcome.state, output_refs=store_outcome.output_refs
        )["terminal_ref"]
    return {
        "terminal_refs": {"store": store_terminal, "corporate_action_recon": corporate_terminal},
        "root": root,
        "book": book,
        "journal": journal,
        "store_outputs": store.probe(store.template()).outcome.output_refs,
        "corporate_request": request,
        "corporate_outputs": outcome.output_refs,
        "policy_refs": {"trailing": policy, "initial_stop": stop},
        "native_inputs": {
            "schema_version": "cn-daily-native-inputs.v5",
            "trade_date": "20260827",
            "previous_trade_date": "20260826",
            "store_plan_ref": plan,
            "calendar_ref": ref(root, book.calendar_path),
            "market_snapshot_ref": market,
            "adjustment_market_refs": {"002463.SZ": frame},
            "corporate_action_context_ref": context,
            "decision_recipe_ref": decision,
            "research_request_ref": research,
            "dashboard_publication_policy": "native-eod-first.v1",
        },
    }
