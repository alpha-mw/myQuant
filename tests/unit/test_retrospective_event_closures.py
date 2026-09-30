"""Retrospective owner declarations append to, never replace, the event closure set."""

from argparse import Namespace

import pytest

from _native_daily_store_fixture import DAYS, NativeStoreFixture, sha, write
from quant_investor.strategy_records import event_store
from quant_investor.strategy_records.store import StrategyRecordStoreError
from scripts import manage_cn_strategy_records as manager


def _declaration(fixture, days, name):
    path = f"fixtures/{name}.json"
    digest = write(
        fixture.project / path,
        {
            "schema_id": "myquant.cn_official_close_retrospective_owner_declaration.v1",
            "declaration_id": name,
            "owner": "Maxwell",
            "authorized_at": "2026-09-30T08:00:00Z",
            "policy_id": "cn-daily-official-close-policy-v1",
            "strategy_label": "aggressive_tech_manufacturing",
            "retrospective_empty_event_closure_authorized": True,
            "broker_order_trade_authority": False,
            "actual_holdings_mutation_authority": False,
            "cash_mutation_authority": False,
            "dates": [
                {
                    "trade_date": day,
                    "source_receipt_id": None,
                    **dict.fromkeys(event_store.EVENT_DIMENSIONS, []),
                }
                for day in days
            ],
        },
    )
    return path, digest


def _publish(fixture, path, digest, generation_id):
    return manager.command_publish_event_closures(
        Namespace(
            project_root=str(fixture.project),
            record_root=str(fixture.root),
            policy_path=fixture.policy_path,
            policy_sha256=fixture.policy_sha,
            retrospective_declaration=path,
            retrospective_declaration_sha256=digest,
            expected_event_pointer_sha256=sha(fixture.root / "_event_store/current.v1.json"),
            generation_id=generation_id,
            published_at="2026-09-30T08:05:00Z",
        )
    )


def test_retrospective_declaration_appends_to_existing_closures(tmp_path):
    fixture = NativeStoreFixture(tmp_path)
    fixture.advance(DAYS[0])
    before = event_store.load_generation(fixture.root / "_event_store")["closures"]
    path, digest = _declaration(fixture, DAYS[1:3], "retro-append")

    result = _publish(fixture, path, digest, "event-retro-append-v1")

    assert result["trade_dates"] == DAYS[:3]
    after = event_store.load_generation(fixture.root / "_event_store")["closures"]
    assert after[0] == before[0]
    assert [row["trade_date"] for row in after] == DAYS[:3]


def test_retrospective_declaration_refuses_already_closed_date(tmp_path):
    fixture = NativeStoreFixture(tmp_path)
    fixture.advance(DAYS[0])
    pointer = (fixture.root / "_event_store/current.v1.json").read_bytes()
    path, digest = _declaration(fixture, DAYS[:2], "retro-repeat")

    with pytest.raises(StrategyRecordStoreError, match="repeats closed dates:" + DAYS[0]):
        _publish(fixture, path, digest, "event-retro-repeat-v1")

    assert (fixture.root / "_event_store/current.v1.json").read_bytes() == pointer
