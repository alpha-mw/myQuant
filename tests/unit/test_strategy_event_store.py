from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.strategy_records.event_store import (
    EMPTY_POINTER_SHA256,
    EVENT_DIMENSIONS,
    StrategyEventStoreError,
    build_empty_closure,
    load_generation,
    publish_generation,
)


def _closure(day: str = "2026-08-24") -> dict:
    return build_empty_closure(
        trade_date=day,
        sealed_at="2026-09-01T01:00:00Z",
        cutoff_at=f"{day}T07:30:00Z",
        policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
        owner_declaration_ref={"path": "operations/declaration.json", "sha256": "b" * 64},
        source_receipt_ref=None,
    )


def test_event_store_requires_explicit_all_dimension_empty_closure(tmp_path: Path) -> None:
    result = publish_generation(
        tmp_path,
        generation_id="event-20260824-test",
        generated_at="2026-09-01T01:00:00Z",
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        closures=[_closure()],
        policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
    )
    loaded = load_generation(tmp_path)

    assert loaded == result
    assert set(loaded["closures"][0]["dimensions"]) == set(EVENT_DIMENSIONS)
    assert loaded["closures"][0]["status"] == "CLOSED_EMPTY"


def test_event_store_rejects_missing_dimension(tmp_path: Path) -> None:
    closure = _closure()
    del closure["dimensions"]["funding"]
    closure.pop("content_sha256")
    with pytest.raises(StrategyEventStoreError, match="content SHA|dimensions"):
        publish_generation(
            tmp_path,
            generation_id="event-invalid-test",
            generated_at="2026-09-01T01:00:00Z",
            expected_pointer_sha256=EMPTY_POINTER_SHA256,
            closures=[closure],
            policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
        )


def test_event_successor_preserves_closures_and_retains_exact_pointer(tmp_path: Path) -> None:
    first = publish_generation(
        tmp_path,
        generation_id="event-first-test",
        generated_at="2026-09-01T01:00:00Z",
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        closures=[_closure()],
        policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
    )
    first_pointer = (tmp_path / "current.v1.json").read_bytes()
    second = publish_generation(
        tmp_path,
        generation_id="event-second-test",
        generated_at="2026-09-02T01:00:00Z",
        expected_pointer_sha256=first["pointer_sha256"],
        closures=[_closure(), _closure("2026-08-25")],
        policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
    )

    assert second["pointer"]["trade_dates"] == ["2026-08-24", "2026-08-25"]
    assert (
        tmp_path / "pointer_history" / f"{first['pointer_sha256']}.json"
    ).read_bytes() == first_pointer


def test_event_successor_rejects_loss_conflict_and_exact_replay(tmp_path: Path) -> None:
    first = publish_generation(
        tmp_path,
        generation_id="event-first-test",
        generated_at="2026-09-01T01:00:00Z",
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        closures=[_closure(), _closure("2026-08-25")],
        policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
    )
    with pytest.raises(StrategyEventStoreError, match="dropped current closure"):
        publish_generation(
            tmp_path,
            generation_id="event-subset-test",
            generated_at="2026-09-02T01:00:00Z",
            expected_pointer_sha256=first["pointer_sha256"],
            closures=[_closure("2026-08-25")],
            policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
        )
    assert not (tmp_path / "generations/event-subset-test.v1.json").exists()

    conflicting = _closure()
    conflicting["sealed_at"] = "2026-09-02T02:00:00Z"
    conflicting.pop("content_sha256")
    from quant_investor.strategy_records.store import content_sha256

    conflicting["content_sha256"] = content_sha256(conflicting)
    with pytest.raises(StrategyEventStoreError, match="OFFICIAL_CLOSE_RESTATEMENT_REQUIRED"):
        publish_generation(
            tmp_path,
            generation_id="event-conflict-test",
            generated_at="2026-09-02T01:00:00Z",
            expected_pointer_sha256=first["pointer_sha256"],
            closures=[conflicting, _closure("2026-08-25")],
            policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
        )

    replay = publish_generation(
        tmp_path,
        generation_id="event-replay-test",
        generated_at="2026-09-02T01:00:00Z",
        expected_pointer_sha256=first["pointer_sha256"],
        closures=[_closure(), _closure("2026-08-25")],
        policy_ref={"path": "operations/policy.json", "sha256": "a" * 64},
    )
    assert replay["no_action"] is True
    assert not (tmp_path / "generations/event-replay-test.v1.json").exists()
