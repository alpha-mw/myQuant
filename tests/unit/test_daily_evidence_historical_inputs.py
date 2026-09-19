"""Selection guard tests; source validation remains the existing history resolver."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from quant_investor.factors import production_authority as module
from quant_investor.factors.governance.errors import FactorGovernanceError


def store(monkeypatch):
    value = object.__new__(module.FactorProductionStore)
    value._active_lock = nullcontext
    row = {
        "factor_pointer_sha256": "a" * 64,
        "signal_date": "20260903",
        "factor_generation_id": "old-generation",
    }
    value.read_observation_history = lambda: [row]
    value.read = lambda path: SimpleNamespace(
        byte_sha256="b" * 64 if str(path).endswith("_active.json") else "a" * 64, data=b"{}"
    )
    signals = {
        module.LOW_DOLLAR_VOLUME: {"000001.SZ": "0x1.0p+0"},
        module.BLEND_W80: {"000001.SZ": "0x1.0p-1"},
    }
    value._read_generation_for_pointer = lambda *args, **kwargs: {
        "payload": {
            "as_of": "20260903",
            "factor_production_generation_id": "old-generation",
            "signal_values": signals,
            "active_factor_rows": [],
        }
    }
    monkeypatch.setattr(module, "_artifact_ref", lambda _: {"synthetic": True})
    return value, signals


def test_historical_selection_copies_original_values_not_active_values(monkeypatch):
    reader, signals = store(monkeypatch)
    result = reader.read_historical_research_inputs(
        expected_pointer_sha256="a" * 64, expected_trade_date="20260903"
    )
    assert result["signal_date"] == "20260903"
    result["signal_values"][module.LOW_DOLLAR_VOLUME]["000001.SZ"] = "changed"
    assert signals[module.LOW_DOLLAR_VOLUME]["000001.SZ"] == "0x1.0p+0"


@pytest.mark.parametrize(
    "pointer,day,reason",
    [
        ("c" * 64, "20260903", "UNSUPPORTED_LINEAGE_GAP"),
        ("a" * 64, "20260904", "trade date differs"),
    ],
)
def test_unavailable_history_and_date_mismatch_fail_closed(monkeypatch, pointer, day, reason):
    reader, _ = store(monkeypatch)
    with pytest.raises(FactorGovernanceError, match=reason):
        reader.read_historical_research_inputs(
            expected_pointer_sha256=pointer, expected_trade_date=day
        )


def test_head_change_during_historical_copy_rejected(monkeypatch):
    reader, _ = store(monkeypatch)
    count = []

    def read(path):
        if str(path).endswith("_active.json"):
            count.append(1)
            return SimpleNamespace(byte_sha256=("b" if len(count) == 1 else "c") * 64, data=b"{}")
        return SimpleNamespace(byte_sha256="a" * 64, data=b"{}")

    reader.read = read
    with pytest.raises(FactorGovernanceError, match="changed during historical"):
        reader.read_historical_research_inputs(
            expected_pointer_sha256="a" * 64, expected_trade_date="20260903"
        )


def test_recorded_reader_rejects_pointer_sha_before_any_read():
    reader = object.__new__(module.FactorProductionStore)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input must not enter storage")

    reader.read = forbidden
    reader._active_lock = forbidden
    with pytest.raises(FactorGovernanceError, match="SHA differs"):
        reader.inspect_recorded_research_inputs(
            pointer_raw=b"{}", expected_pointer_sha256="a" * 64, expected_trade_date="20260903"
        )


def test_recorded_reader_has_no_current_head_or_lock_dependency(monkeypatch):
    import hashlib

    raw = b"{}"
    digest = hashlib.sha256(raw).hexdigest()
    reader, _ = store(monkeypatch)
    marker = SimpleNamespace(byte_sha256="e" * 64, data=b"marker")

    def read(path):
        assert path == module.FACTOR_PRODUCTION_MARKER_PATH
        return marker

    def forbidden(*args, **kwargs):
        raise AssertionError("recorded replay must not lock")

    reader.read = read
    reader._active_lock = forbidden
    reader._verify_pointer_lineage = lambda pointer, marker: {
        "as_of": "20260903",
        "genesis_pointer_sha256": "f" * 64,
    }
    reader._observation_inputs_for_pointer = lambda *args, **kwargs: {"signal_date": "20260903"}
    result = reader.inspect_recorded_research_inputs(
        pointer_raw=raw, expected_pointer_sha256=digest, expected_trade_date="20260903"
    )
    assert result["consumer_admission"] is False
    with pytest.raises(FactorGovernanceError, match="trade date differs"):
        reader.inspect_recorded_research_inputs(
            pointer_raw=raw, expected_pointer_sha256=digest, expected_trade_date="20260904"
        )
