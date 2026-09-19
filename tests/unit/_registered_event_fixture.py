"""Synthetic owner BUY registered through the existing native manager and Store."""

from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from _native_daily_store_fixture import NativeStoreFixture, write
from scripts import cn_official_close_batch as batch
from scripts import manage_cn_strategy_records as manager
from quant_investor.strategy_records import store
from quant_investor.strategy_records import registered_event_contracts as contracts

NOW = datetime(2026, 8, 25, 12, 20, tzinfo=timezone.utc)


def ref(project, path):
    path = Path(path)
    return {
        "path": path.relative_to(project).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build(project, *, empty_first=False, buy_symbol="002463.SZ"):
    project = Path(project).resolve()
    book = NativeStoreFixture(project, stock_symbols=("002463.SZ",))
    args = book.advance("2026-08-24")
    clock = datetime(2026, 8, 24, 12, 20, tzinfo=timezone.utc)
    with manager._operation_lock(book.root):
        planned = batch.close_through_latest(**args, execute=False, prepare_only=True, now=clock)
        batch.close_through_latest(
            **args, execute=True, expected_plan_sha=planned["plan_sha256"], now=clock
        )
    baseline_ref = ref(
        project, (project / planned["plan_path"]).with_name("committed-pointer.v1.json")
    )
    if empty_first:
        book.advance("2026-08-25")
    return apply_buy(
        project,
        book=book,
        baseline_ref=baseline_ref,
        baseline_arguments=args,
        trade_date="2026-08-25",
        buy_symbol=buy_symbol,
    )


def apply_buy(project, *, book, baseline_ref, baseline_arguments, trade_date, buy_symbol):
    """Apply one synthetic owner BUY before native sealing, on an existing native book."""
    project = Path(project).resolve()
    compact = trade_date.replace("-", "")
    pointer, catalog = store.load_registered_catalog(book.root)
    parent = next(r for r in catalog["records"] if r["record_id"] == pointer["active_record_id"])
    record_id = compact + "_1000"
    staged = manager.command_stage_init(Namespace(record_root=str(book.root), record_id=record_id))
    stage = Path(staged["staging_dir"])
    for item in (book.root / pointer["active_record_id"]).iterdir():
        if item.is_file():
            shutil.copy2(item, stage / item.name)
    manual = json.loads((stage / "manual_execution_manifest.json").read_bytes())
    previous_trade_date = manual["valuation_trade_date"]
    manifest = json.loads((stage / "manifest.json").read_bytes())
    ledger = pd.read_parquet(stage / "ledger_after_manual_switch.parquet")
    # Synthetic executions use cent prices; the continuous research-price
    # fixture is not an executable fill and must not manufacture fractional cash.
    shares, price, fee = 100, round(float(ledger.iloc[0]["current_price"]), 2), 5.02
    value = shares * price
    cost = value + fee
    trade = {
        "trade_id": "synthetic-buy-1",
        "symbol": buy_symbol,
        "name": "Synthetic",
        "side": "BUY",
        "shares": shares,
        "execution_price": price,
        "trade_value": value,
        "commission_cny": 5.0,
        "stamp_duty_cny": 0.0,
        "transfer_fee_cny": 0.02,
        "final_total_fee_cny": fee,
        "cost_basis_cny": cost,
        "avg_cost_cny_per_share": cost / shares,
        "commission_rate": 0.0001,
        "commission_minimum_cny": 5.0,
        "commission_includes_regulatory_and_handling": True,
        "transfer_fee_rate": 0.00001,
        "fill_cost_status": contracts.FEE_STATUS,
        "reported_by": "SyntheticOwner",
        "source_channel": "synthetic owner fixture",
        "trade_date": compact,
    }
    if buy_symbol in set(ledger["symbol"]):
        index = ledger.index[ledger["symbol"] == buy_symbol][0]
        ledger.loc[index, "shares"] += shares
        ledger.loc[index, "cost_basis"] += cost
        ledger.loc[index, "avg_cost"] = (
            ledger.loc[index, "cost_basis"] / ledger.loc[index, "shares"]
        )
    else:
        added = ledger.iloc[0].copy()
        added.update(
            {"symbol": buy_symbol, "shares": shares, "cost_basis": cost, "avg_cost": cost / shares}
        )
        ledger = pd.concat([ledger, added.to_frame().T], ignore_index=True)
        book.stocks = (*book.stocks, buy_symbol)
    ledger["current_value"] = ledger["shares"] * ledger["current_price"]
    ledger["unrealized_pnl"] = ledger["current_value"] - ledger["cost_basis"]
    ledger["unrealized_pnl_pct"] = ledger["unrealized_pnl"] / ledger["cost_basis"]
    cash = float(manual["cash_after"]) - cost
    equity = float(ledger["current_value"].sum())
    nav = cash + equity
    ledger["equity_sleeve_weight"] = ledger["current_value"] / equity
    ledger["nav_weight"] = ledger["current_value"] / nav
    ledger_path = stage / "ledger_after_manual_switch.parquet"
    ledger.to_parquet(ledger_path, index=False)
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    for document in (manual, manifest):
        document.pop("publication_class", None)
        document.pop("publication_delay", None)
    manual.update(
        status="owner_declared_manual_execution_applied",
        execution_status="owner_declared_manual_execution_applied",
        record_timestamp=record_id,
        recorded_at=trade_date + " 10:00:00 CST",
        recorded_at_iso=trade_date + "T10:00:00+08:00",
        source_record=pointer["active_record_id"],
        trade_date=compact,
        valuation_trade_date=compact,
        official_valuation=False,
        valuation_completeness_passed=False,
        valuation_status="OWNER_REPORTED_FILLS_OFFICIAL_CLOSE_PENDING",
        no_trade_performed=False,
        owner_reported_external_fills=True,
        applied_owner_declared_trades=[trade],
        applied_local_trades=[],
        funding_events=[],
        rejected_or_pending_trades=[],
        owner_declaration={"approved_by": "SyntheticOwner", "synthetic": True},
        no_broker_api_called=True,
        net_external_flow=0.0,
        excluded_external_flow=0.0,
        trade_count=1,
        fill_count=1,
        order_count=0,
        effective_manual_holding_count=len(ledger),
        cash_before=float(manual["cash_after"]),
        gross_trade_value=value,
        fees_cny=fee,
        cash_after=cash,
        market_value_after=equity,
        total_value_after=nav,
        portfolio_pnl_after=nav - 1_000_000,
        portfolio_return_after=nav / 1_000_000 - 1,
        next_ledger_sha256=ledger_sha,
        ledger_after_manual_switch_parquet_sha256=ledger_sha,
        source_manifest_sha256=parent["manifest_sha256"],
        source_manual_manifest_sha256=parent["manual_manifest_sha256"],
        source_contained_ledger_sha256=parent["ledger_sha256"],
    )
    manual["ledger_provenance"].update(
        declared_sha256=ledger_sha, source_ledger_sha256=parent["ledger_sha256"]
    )
    manual["financial_state"] = {
        **{
            k: manual[k]
            for k in (
                "capital_cny",
                "cash_after",
                "market_value_after",
                "total_value_after",
                "portfolio_pnl_after",
                "portfolio_return_after",
            )
        },
        "ledger_sha256": ledger_sha,
    }
    manual["financial_state_sha256"] = hashlib.sha256(
        store.canonical_json_bytes(manual["financial_state"])
    ).hexdigest()
    pnl_path = stage / "pnl_summary.csv"
    pnl = pd.read_csv(pnl_path)
    for field in (
        "cash_after",
        "market_value_after",
        "total_value_after",
        "portfolio_pnl_after",
        "portfolio_return_after",
    ):
        pnl[field] = manual[field]
    pnl.to_csv(pnl_path, index=False)
    manual["pnl_summary_sha256"] = hashlib.sha256(pnl_path.read_bytes()).hexdigest()
    write(stage / "manual_execution_manifest.json", manual)
    manifest.update(
        timestamp=record_id,
        source_record=pointer["active_record_id"],
        source_manifest_sha256=parent["manifest_sha256"],
        recorded_at=manual["recorded_at"],
        recorded_at_iso=manual["recorded_at_iso"],
        manual_execution=manual,
        action_taken_today=True,
        trade_count=1,
        fill_count=1,
        order_count=0,
    )
    manifest["data_snapshot"].update(
        analysis_trade_date=compact,
        valuation_trade_date=compact,
        valuation_status=manual["valuation_status"],
        last_strict_completed_trade_date_for_untouched_marks=previous_trade_date,
    )
    write(stage / "manifest.json", manifest)
    with manager._operation_lock(book.root):
        result = manager.command_seal_publish(
            Namespace(
                project_root=str(project),
                record_root=str(book.root),
                record_id=record_id,
                expected_pointer_sha=baseline_ref["sha256"],
                generation_id="registered-buy-fixture",
                performance_generation_id="registered-buy-performance",
                published_at=trade_date + "T02:01:00Z",
            )
        )
    writer_sha = result["pointer_sha256"]
    current, catalog = store.load_registered_catalog(book.root)
    row = next(r for r in catalog["records"] if r["record_id"] == current["active_record_id"])
    manual_ref = {
        "path": contracts.RECORD_ROOT + "/" + row["manual_manifest_path"],
        "sha256": row["manual_manifest_sha256"],
    }
    domains = {
        name: {"state": contracts.NONE, "fact_refs": []} for name in contracts.EVENT_DIMENSIONS
    }
    for name in ("executions", "fills", "cost_basis_changes"):
        domains[name] = {"state": contracts.FACTS, "fact_refs": [manual_ref]}
    fact = contracts.seal(
        {
            "schema_version": contracts.FACT_SCHEMA,
            "owner_fact_id": "synthetic-owner-fact",
            "strategy_id": contracts.STRATEGY,
            "trade_date": trade_date,
            "owner": "SyntheticOwner",
            "owner_declared_at": trade_date + "T02:02:00Z",
            "baseline_store_pointer_ref": baseline_ref,
            "writer_pointer_sha256": writer_sha,
            "writer_record_id": record_id,
            "domains": domains,
            "evidence_level": "OWNER_DECLARED",
            "broker_statement_verified": False,
            "authority": dict(contracts.AUTHORITY),
        }
    )
    owner_path = project / "fixtures/owner-facts.json"
    owner_sha = write(owner_path, fact)
    return {
        "book": book,
        "baseline_ref": baseline_ref,
        "baseline_arguments": baseline_arguments,
        "writer_sha": writer_sha,
        "fact": fact,
        "fact_ref": ref(project, owner_path),
        "args": Namespace(
            project_root=str(project),
            record_root=str(book.root),
            owner_fact=str(owner_path),
            owner_fact_sha256=owner_sha,
            expected_pointer_sha=writer_sha,
        ),
    }
