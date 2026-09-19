"""Synthetic issuer declarations and pre-registration native accounting transitions."""

import hashlib
import json

import pandas as pd

from quant_investor.contracts import canonical_json_bytes
from quant_investor.strategy_records.store import canonical_json_bytes as native_bytes
from _native_daily_store_fixture import NativeStoreFixture, write

STRATEGY = "aggressive_tech_manufacturing"
AS_OF = "2026-08-24T13:30:00Z"


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def corporate_book(
    root,
    *,
    symbol="601899.SH",
    kind="DIVIDEND",
    posting=True,
    cash_delta=100,
    mixed=False,
    share_multiplier=1,
):
    announcement = put(root, "fixtures/announcement.json", {"synthetic": True, "kind": kind})
    event = put(
        root,
        "fixtures/corporate-event.json",
        {
            "schema_version": "cn-corporate-action-event.v1",
            "event_id": "synthetic-action",
            "symbol": symbol,
            "kind": kind,
            "effective_trade_date": "20260820",
            "announced_at": "2026-08-19T08:00:00Z",
            "announcement_ref": announcement,
            "accounting_records": (
                {"before_record_id": "20260819_1321", "after_record_id": "20260820_1321"}
                if posting
                else None
            ),
        },
    )

    def transition(previous, after):
        path = after / "manual_execution_manifest.json"
        manual = json.loads(path.read_bytes())
        if share_multiplier != 1:
            ledger_path = after / "ledger_after_manual_switch.parquet"
            ledger = pd.read_parquet(ledger_path)
            mask = ledger["symbol"] == symbol
            ledger.loc[mask, "shares"] *= share_multiplier
            ledger.loc[mask, "avg_cost"] /= share_multiplier
            ledger.loc[mask, "current_price"] /= share_multiplier
            ledger.to_parquet(ledger_path, index=False)
            digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
            manual["next_ledger_sha256"] = digest
            manual["ledger_after_manual_switch_parquet_sha256"] = digest
            manual["financial_state"]["ledger_sha256"] = digest
        for value in (manual, manual["financial_state"]):
            value["cash_after"] += cash_delta
            value["total_value_after"] += cash_delta
            value["portfolio_pnl_after"] += cash_delta
            value["portfolio_return_after"] = value["portfolio_pnl_after"] / 1_000_000
        manual["financial_state_sha256"] = hashlib.sha256(
            native_bytes(manual["financial_state"])
        ).hexdigest()
        for key in (
            "applied_local_trades",
            "applied_owner_declared_trades",
            "reconciled_source_trades",
            "rejected_or_pending_trades",
            "funding_events",
            "fills",
            "orders",
            "manual_changes",
        ):
            manual[key] = []
        manual["gross_trade_value"] = 0
        manual["corporate_actions"] = [
            {
                "schema_version": "cn-corporate-action-application.v1",
                "event_ref": event,
                "symbol": symbol,
            }
        ]
        if mixed:
            other = json.loads((root / event["path"]).read_bytes())
            other["event_id"] = "synthetic-second-action"
            other_ref = put(root, "fixtures/other-corporate-event.json", other)
            manual["corporate_actions"].append(
                {
                    "schema_version": "cn-corporate-action-application.v1",
                    "event_ref": other_ref,
                    "symbol": symbol,
                }
            )
        write(path, manual)
        manifest = json.loads((after / "manifest.json").read_bytes())
        manifest.update(manual_execution=manual, trade_count=0, order_count=0, fill_count=0)
        write(after / "manifest.json", manifest)
        pd.DataFrame(
            [
                {
                    key: manual[key]
                    for key in (
                        "cash_after",
                        "market_value_after",
                        "total_value_after",
                        "portfolio_pnl_after",
                    )
                }
            ]
        ).to_csv(after / "pnl_summary.csv", index=False)

    book = NativeStoreFixture(
        root, stock_symbols=(symbol,), seed_transition=transition if posting else None
    )
    return book, event


def tracking_policy(root, book):
    from quant_investor.strategy_records.store import load_registered_catalog
    from quant_investor.strategy_records.corporate_contracts import POLICY_CALCULATION

    pointer, catalog = load_registered_catalog(book.root)
    base = next(r for r in catalog["records"] if r["record_id"] == "20260819_1321")
    return put(
        root,
        "fixtures/tracking-policy.json",
        {
            "schema_version": "owner-trailing-anchor-policy.v1",
            "policy_id": "synthetic-policy",
            "strategy_domain": STRATEGY,
            "owner": "Synthetic Owner",
            "effective_from": "2026-08-19T00:00:00+08:00",
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
            "store_binding": {
                "pointer_path": str(
                    (book.root / "_record_store/current.v1.json").relative_to(root)
                ),
                "pointer_sha256": hashlib.sha256(
                    (book.root / "_record_store/current.v1.json").read_bytes()
                ).hexdigest(),
                "active_record_id": base["record_id"],
                "ledger_path": str(book.root.relative_to(root)) + "/" + base["ledger_path"],
                "ledger_sha256": base["ledger_sha256"],
            },
            "calculation": POLICY_CALCULATION,
            "anchors": [
                {
                    "symbol": symbol,
                    "company_name": "Synthetic",
                    "tracking_start_date": "20260820",
                    "anchor_state": "OWNER_APPROVED_RESET_NO_STRUCTURED_BUY_FOUND",
                    "exclude_pre_anchor_peaks": True,
                }
                for symbol in book.stocks
            ],
            "invalidation": (
                "EARLIEST_OF_OWNER_REVISION_POSITION_REMOVAL_" "NEW_ENTRY_ADD_OR_CORPORATE_ACTION"
            ),
        },
    )


def context_ref(root, book, event_ref=None, *, review=False):
    from quant_investor.strategy_records.corporate_contracts import REVIEW_AUTHORITY

    policy = tracking_policy(root, book)
    events = (
        None
        if event_ref is None
        else put(
            root,
            "fixtures/events.json",
            {
                "schema_version": "cn-corporate-action-events.v1",
                "strategy_id": STRATEGY,
                "as_of": AS_OF,
                "event_refs": [event_ref],
            },
        )
    )
    reviews = None
    if review:
        reviews = put(
            root,
            "fixtures/reviews.json",
            {
                "schema_version": "cn-corporate-anchor-reviews.v1",
                "strategy_id": STRATEGY,
                "owner": "Synthetic Owner",
                "declaration_id": "synthetic-review",
                "declared_at": "2026-08-22T08:00:00Z",
                "authority": REVIEW_AUTHORITY,
                "reviews": [
                    {
                        "event_ref": event_ref,
                        "policy_ref": policy,
                        "source_record_id": "20260820_1321",
                        "reviewed_at": "2026-08-22T07:00:00Z",
                        "disposition": "OWNER_DECLARED_RESET",
                        "tracking_start_date": "20260821",
                    }
                ],
            },
        )
    return put(
        root,
        "fixtures/corporate-context.json",
        {
            "schema_version": "cn-corporate-action-context.v1",
            "strategy_id": STRATEGY,
            "as_of": AS_OF,
            "tracking_policy_ref": policy,
            "named_events_ref": events,
            "anchor_reviews_ref": reviews,
        },
    )
