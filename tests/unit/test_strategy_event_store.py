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
