"""Receipt persistence/shape tests; these do not claim native live admission."""

from copy import deepcopy
import pytest
from quant_investor.intelligence.morning import classify_sina_quote_timing
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.morning_receipt import (
    MorningReceiptStorage,
    receipt_path,
    _MorningIO,
)
from quant_investor.system.errors import SystemImmutableConflict, SystemSecurityError


def receipt():
    ref = {"path": "input.json", "sha256": "a" * 64}
    return {
        "schema_version": "morning-strategy-run.v2",
        "run_date": "20260827",
        "previous_trade_date": "20260826",
        "request_ref": ref,
        "previous_completion_ref": {
            **ref,
            "path": "results/operations/daily_production/CN/20260826/completion.v1.json",
        },
        "quote_capture_ref": ref,
        "quote_raw_ref": ref,
        "owner_policy_ref": ref,
        "output_ref": {
            **ref,
            "path": "results/operations/morning_strategy/CN/20260827/0945-strategy.v2.md",
        },
        "validated_at": "2026-08-27T01:50:00Z",
        "status": "COMPLETE",
        "admission": "LIVE_RESEARCH_CONSUMER",
        "synthetic": False,
        "prospective_admission_state": "NOT_CLAIMED",
        "expected_symbols": ["000001.SZ"],
        "quote_timing": classify_sina_quote_timing("2026-08-27T01:45:00Z", run_date="20260827"),
        "authority": dict(FALSE_AUTHORITY),
    }


def test_fixed_immutable_receipt_has_no_lock_or_other_outputs(tmp_path):
    storage = MorningReceiptStorage(str(tmp_path))
    assert storage.read("20260827") is None
    assert list(tmp_path.iterdir()) == []
    value = receipt()
    first = storage.write(value)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert list(before) == [str(tmp_path / receipt_path("20260827"))]
    assert storage.write(value) == first == storage.read("20260827")
    changed = deepcopy(value)
    changed["validated_at"] = "2026-08-27T01:51:00Z"
    with pytest.raises(SystemImmutableConflict):
        storage.write(changed)
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "morning-strategy-replay.v2"),
        ("status", "PARTIAL"),
        ("synthetic", True),
        ("synthetic", 0),
        ("admission", "RESEARCH_ONLY"),
        ("prospective_admission_state", "ELIGIBLE"),
        ("expected_symbols", []),
        ("expected_symbols", ["000001.SZ", "000001.SZ"]),
        ("validated_at", "2026-08-27T01:40:00Z"),
        ("validated_at", "2026-08-28T01:50:00Z"),
        ("request_ref", None),
    ],
)
def test_invalid_receipt_never_creates_directories(tmp_path, field, value):
    document = receipt()
    document[field] = value
    with pytest.raises((ContractError, ValueError)):
        MorningReceiptStorage(str(tmp_path)).write(document)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "path",
    [
        "results/system/_active.json",
        "results/operations/morning_strategy/CN/20260827/0945-run.json",
        "results/operations/morning_strategy/CN/20260827/0945-strategy.v2.md",
    ],
)
def test_io_cannot_write_other_artifacts(tmp_path, path):
    with pytest.raises(SystemSecurityError):
        _MorningIO(str(tmp_path)).write(path, b"{}")
    assert list(tmp_path.iterdir()) == []


def test_symlink_receipt_rejected_without_touching_target(tmp_path):
    storage = MorningReceiptStorage(str(tmp_path))
    value = receipt()
    storage.write(value)
    path = tmp_path / receipt_path("20260827")
    original = path.read_bytes()
    target = tmp_path / "retained.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(SystemSecurityError):
        storage.read("20260827")
    assert target.read_bytes() == original
