"""Read-only source selection uses exact native custody, with synthetic books."""

import pytest

from _registered_event_fixture import build, ref, NOW
from test_registered_close_native import case, prepared_case, inventory
from scripts import manage_cn_strategy_records as manager
from scripts.registered_daily_event_sources import select_daily_source
from quant_investor.strategy_records.registered_event_contracts import RegisteredEventError


def select(root, fixture, day="2026-08-25"):
    return select_daily_source(
        workspace=root,
        store_pointer_ref=ref(root, fixture["book"].root / "_record_store/current.v1.json"),
        trade_date=day,
    )


def test_current_registered_source_selects_exact_decl_without_writes(tmp_path, monkeypatch):
    fixture, args = case(tmp_path, monkeypatch)
    before = inventory(tmp_path)
    selected = select(tmp_path, fixture)
    assert selected["state"] == "REGISTERED_INTRADAY"
    assert selected["registered_event_declaration_ref"] == args["registered_event_declaration_ref"]
    assert selected["writer_pointer_ref"]["sha256"] == fixture["writer_sha"]
    assert selected["decision_baseline_pointer_ref"] == fixture["baseline_ref"]
    assert inventory(tmp_path) == before


def test_undeclared_writer_blocks_before_any_source_publication(tmp_path):
    fixture = build(tmp_path)
    before = inventory(tmp_path)
    with pytest.raises(RegisteredEventError, match="DECLARATION_REQUIRED"):
        select(tmp_path, fixture)
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "day,code",
    [
        ("2026-08-26", "HISTORICAL_FINALIZATION_UNSUPPORTED"),
        ("2026-08-24", "TARGET_BEFORE_STORE"),
    ],
)
def test_registered_source_does_not_skip_calendar_dates(tmp_path, monkeypatch, day, code):
    fixture = build(tmp_path)
    monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW)
    manager.command_publish_registered_event_declaration(fixture["args"])
    before = inventory(tmp_path)
    with pytest.raises(RegisteredEventError, match=code):
        select(tmp_path, fixture, day)
    assert inventory(tmp_path) == before


def test_standalone_finalized_store_is_not_fresh_daily_request_custody(tmp_path, monkeypatch):
    fixture, args, prepared, adapter = prepared_case(tmp_path, monkeypatch)
    adapter.execute(adapter.template())
    before = inventory(tmp_path)
    selected = select(tmp_path, fixture)
    assert selected == {
        "state": "BLOCKED_FINALIZED_V2_INPUT_CUSTODY_MISSING",
        "registered_store_plan_ref": {
            "path": prepared["plan_path"],
            "sha256": prepared["plan_sha256"],
        },
        "registered_event_declaration_ref": args["registered_event_declaration_ref"],
    }
    # A completed prior day is an ordinary source for the next day's own events.
    assert select(tmp_path, fixture, "2026-08-26") == {"state": "ORDINARY"}
    assert inventory(tmp_path) == before
