"""Registered Dashboard source closure; native Store/Corporate and synthetic owners."""

from datetime import datetime, timezone
import json

import pytest

from test_registered_transition_report import prepare
from test_registered_close_native import inventory
from _native_corporate_fixture import put
from scripts.daily_production_store_adapter import StoreCloseAdapter
from quant_investor.operations.registered_dashboard import RegisteredDashboardSources
from quant_investor.operations.daily_contract import ContractError
from quant_investor.strategy_records.event_store import StrategyEventStoreError


def case(root, monkeypatch, *, symbol="002463.SZ"):
    fixture, args, corporate = prepare(root, monkeypatch, buy_symbol=symbol)
    journal = corporate.journal
    with journal.locked():
        corporate.prepare()
        request = corporate.template()
        journal.begin(request)
        corporate.execute(request)
        outcome = corporate.probe(request).outcome
        cterminal = journal.finish(request, state=outcome.state, output_refs=outcome.output_refs)
        store = StoreCloseAdapter(
            arguments=args,
            trade_date=journal.trade_date,
            plan_ref=corporate.binding["store_plan_ref"],
            release_ref=corporate.release_ref,
        )
        request = store.template()
        journal.begin(request)
        store.execute(request)
        soutcome = store.probe(request).outcome
        sterminal = journal.finish(request, state=soutcome.state, output_refs=soutcome.output_refs)
    kwargs = {
        "workspace": root,
        "trade_date": journal.trade_date,
        "release_ref": corporate.release_ref,
        "corporate_terminal_ref": cterminal["terminal_ref"],
        "store_terminal_ref": sterminal["terminal_ref"],
        "store_plan_ref": store.plan_ref,
        "registered_event_declaration_ref": args["registered_event_declaration_ref"],
    }
    return fixture, corporate, kwargs, outcome.output_refs, soutcome.output_refs


@pytest.mark.parametrize("symbol", ["002463.SZ", "300308.SZ"])
def test_registered_dashboard_proves_writer_and_final_close_without_writes(
    tmp_path, monkeypatch, symbol
):
    fixture, corporate, kwargs, cout, sout = case(tmp_path, monkeypatch, symbol=symbol)
    before = inventory(tmp_path)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    value = RegisteredDashboardSources(**kwargs).result(stamp)
    assert value["registered_transition_ref"] == cout["registered_transition"]
    summary = value["registered_close_summary"]
    assert summary["decision_baseline_pointer_ref"] == fixture["baseline_ref"]
    assert summary["writer_pointer_ref"]["sha256"] == fixture["writer_sha"]
    assert summary["final_pointer_ref"] == sout["pointer"]
    assert summary["writer_record_id"] == "20260825_1000"
    assert summary["official_valuation"] is True
    assert [summary["close_writer_" + k + "_count"] for k in ("trade", "order", "fill")] == [
        0,
        0,
        0,
    ]
    assert value["registered_transition"]["payload"]["broker_statement_verified"] is False
    assert inventory(tmp_path) == before
    for path in (
        fixture["book"].root / "_record_store/current.v1.json",
        fixture["book"].root / "_event_store/current.v1.json",
        tmp_path / "data/parquet/cn/_latest.json",
    ):
        path.unlink()
    before = inventory(tmp_path)
    assert RegisteredDashboardSources(**kwargs).result(stamp) == value
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "fault", ["declaration", "plan", "release", "terminal", "report", "cash", "custody"]
)
def test_registered_dashboard_rejects_cross_binding_and_tampering(tmp_path, monkeypatch, fault):
    _, _, kwargs, cout, sout = case(tmp_path, monkeypatch)
    if fault in {"declaration", "plan", "release"}:
        field = {
            "declaration": "registered_event_declaration_ref",
            "plan": "store_plan_ref",
            "release": "release_ref",
        }[fault]
        kwargs[field] = put(tmp_path, "fixtures/unbound.json", {"synthetic": True})
    elif fault == "terminal":
        kwargs["corporate_terminal_ref"] = kwargs["store_terminal_ref"]
    elif fault == "report":
        reference = cout["registered_transition"]
        value = json.loads((tmp_path / reference["path"]).read_bytes())
        value["payload"]["broker_statement_verified"] = True
        put(tmp_path, reference["path"], value)
    elif fault == "cash":
        reference = sout["manual"]
        value = json.loads((tmp_path / reference["path"]).read_bytes())
        value["cash_after"] += 1
        put(tmp_path, reference["path"], value)
    stamp = (
        "2026-08-25T12:20:00Z"
        if fault == "custody"
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    with pytest.raises((ValueError, ContractError, StrategyEventStoreError)):
        RegisteredDashboardSources(**kwargs).result(stamp)
