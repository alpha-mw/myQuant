"""Malformed rehashed event evidence cannot authorize a no-position-change close."""

from copy import deepcopy
import hashlib
import json

import pytest

from quant_investor.strategy_records import event_store as store
from quant_investor.strategy_records.store import canonical_json_bytes, content_sha256
from quant_investor.system.storage import SecureSystemStorage
from test_strategy_event_store import _closure


def reseal(value):
    value["content_sha256"] = content_sha256(value)
    return value


def publish(root, rows=None, generated="2026-09-01T01:00:00Z"):
    return store.publish_generation(
        root,
        generation_id="test",
        generated_at=generated,
        expected_pointer_sha256=store.EMPTY_POINTER_SHA256,
        closures=rows or [_closure()],
        policy_ref={"path": "policy.json", "sha256": "a" * 64},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("sealed_at", "2026-08-24T07:00:00Z"),
        ("sealed_at", "not-a-time"),
        ("cutoff_at", None),
        ("cutoff_at", "2026-08-25T07:30:00Z"),
        ("sealed_at", "2026-09-01T01:00:00"),
        ("sealed_at", True),
        ("trade_date", "20260824"),
        ("unexpected_override", True),
        ("actual_holdings_mutation_authority", 0),
        ("late_event_behavior", "IGNORE"),
    ],
)
def test_rehashed_invalid_closure_rejects_before_any_write(tmp_path, field, value):
    invalid = reseal({**_closure(), field: value})
    with pytest.raises(store.StrategyEventStoreError):
        store.validate_closure(invalid)
    with pytest.raises(store.StrategyEventStoreError):
        publish(tmp_path, [invalid])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dimension", store.EVENT_DIMENSIONS)
def test_every_nonempty_financial_dimension_blocks(tmp_path, dimension):
    invalid = deepcopy(_closure())
    invalid["dimensions"][dimension]["events"] = [{"event_id": "unreconciled"}]
    reseal(invalid)
    with pytest.raises(store.StrategyEventStoreError, match="unclosed dimension"):
        publish(tmp_path, [invalid])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("generated", [None, "bad", "2026-08-31T00:00:00Z", "2026-09-01T01:00:00"])
def test_generation_clock_rejects_before_write(tmp_path, generated):
    with pytest.raises(store.StrategyEventStoreError):
        publish(tmp_path, generated=generated)
    assert list(tmp_path.iterdir()) == []


def test_offsets_subseconds_and_equality_preserved(tmp_path):
    value = store.build_empty_closure(
        trade_date="2026-08-24",
        sealed_at="2026-08-24T15:30:00.000001+08:00",
        cutoff_at="2026-08-24T07:30:00.000001Z",
        policy_ref={"path": "policy.json", "sha256": "a" * 64},
        owner_declaration_ref={"path": "owner.json", "sha256": "b" * 64},
        source_receipt_ref=None,
    )
    published = publish(tmp_path, [value], generated=value["sealed_at"])
    assert published["closures"] == [value]


@pytest.mark.parametrize("current", ["missing", "corrupt", "advanced"])
def test_frozen_reader_has_no_current_head_or_writer_dependency(tmp_path, monkeypatch, current):
    first = publish(tmp_path)
    raw = (tmp_path / "current.v1.json").read_bytes()
    if current == "missing":
        (tmp_path / "current.v1.json").unlink()
    elif current == "corrupt":
        (tmp_path / "current.v1.json").write_bytes(b"corrupt current")
    else:
        store.publish_generation(
            tmp_path,
            generation_id="second",
            generated_at="2026-09-02T00:00:00Z",
            expected_pointer_sha256=first["pointer_sha256"],
            closures=[_closure(), _closure("2026-08-25")],
            policy_ref={"path": "new-policy.json", "sha256": "c" * 64},
        )
    original = SecureSystemStorage.read_workspace_file_bytes

    def no_current(self, path, **kwargs):
        assert str(path) != "current.v1.json"
        return original(self, path, **kwargs)

    monkeypatch.setattr(SecureSystemStorage, "read_workspace_file_bytes", no_current)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    result = store.load_frozen_generation(
        tmp_path, pointer_bytes=raw, expected_pointer_sha256=first["pointer_sha256"]
    )
    assert result["closures"] == first["closures"]
    assert result["pointer_bytes"] == raw
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation_id", "../escape"),
        ("trade_dates", ["2026-08-25"]),
        ("previous_pointer_sha256", "bad"),
        ("broker_order_trade_authority", 0),
        ("schema_id", "unknown"),
        ("extra", False),
    ],
)
def test_rehashed_pointer_cannot_waive_native_closure(tmp_path, field, value):
    publish(tmp_path)
    pointer = json.loads((tmp_path / "current.v1.json").read_bytes())
    raw = canonical_json_bytes(reseal({**pointer, field: value}))
    with pytest.raises(store.StrategyEventStoreError):
        store.load_frozen_generation(
            tmp_path, pointer_bytes=raw, expected_pointer_sha256=hashlib.sha256(raw).hexdigest()
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("generated_at", "bad"),
        ("trade_dates", ["2026-08-25"]),
        ("broker_order_trade_authority", True),
    ],
)
def test_current_and_frozen_loaders_share_generation_validation(tmp_path, field, value):
    publish(tmp_path)
    pointer_path = tmp_path / "current.v1.json"
    pointer = json.loads(pointer_path.read_bytes())
    generation_path = tmp_path / pointer["generation"]["path"]
    generation = json.loads(generation_path.read_bytes())
    generation_raw = canonical_json_bytes(reseal({**generation, field: value}))
    generation_path.write_bytes(generation_raw)
    pointer["generation"]["sha256"] = hashlib.sha256(generation_raw).hexdigest()
    raw = canonical_json_bytes(reseal(pointer))
    pointer_path.write_bytes(raw)
    with pytest.raises(store.StrategyEventStoreError):
        store.load_generation(tmp_path)
    with pytest.raises(store.StrategyEventStoreError):
        store.load_frozen_generation(
            tmp_path, pointer_bytes=raw, expected_pointer_sha256=hashlib.sha256(raw).hexdigest()
        )
