"""Recorded v2 cross-ref checks; full native ledger replay is separately required."""

from copy import deepcopy
from datetime import datetime, timezone
import pytest
from quant_investor.operations.ledger_readback import inspect_eod_ledger_bindings
from quant_investor.operations.daily_contract import ContractError
from test_daily_evidence_ledger_assembly import context
from scripts.daily_ledger import publish_native_ledger


def fixture(root, monkeypatch, version=1):
    registry, materialized = context(root, monkeypatch, version)
    with registry.runner.journal.locked():
        published = publish_native_ledger(registry, materialized=materialized, synthetic=True)
    ledger = published["ledger"]
    completion = {
        key: ledger[key]
        for key in (
            "trade_date",
            "graph_sha256",
            "release_ref",
            "native_inputs_ref",
            "materialization_ref",
            "node_terminal_refs",
            "synthetic",
        )
    }
    completion.update(
        prospective_ledger_ref=published["ledger_ref"],
        prospective_admission_state="LEDGER_INELIGIBLE",
        native_validation_completed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return completion, ledger


@pytest.mark.parametrize("version", [1, 2])
def test_recorded_bindings_read_without_writes(tmp_path, monkeypatch, version):
    completion, ledger = fixture(tmp_path, monkeypatch, version)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    actual = inspect_eod_ledger_bindings(
        workspace=str(tmp_path), completion=completion, custody=ledger["node_custody"]
    )
    assert actual == ledger
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["eligible", "input", "terminal", "custody", "premature"])
def test_bound_evidence_mismatch_rejects(tmp_path, monkeypatch, fault):
    completion, ledger = fixture(tmp_path, monkeypatch)
    custody = deepcopy(ledger["node_custody"])
    if fault == "eligible":
        completion["prospective_admission_state"] = "LEDGER_ELIGIBLE"
    elif fault == "input":
        completion["native_inputs_ref"] = {"path": "other.json", "sha256": "f" * 64}
    elif fault == "terminal":
        completion["node_terminal_refs"] = {}
    elif fault == "custody":
        custody["recovered_unknown"] = True
    else:
        completion["native_validation_completed_at"] = "2026-09-07T00:00:00Z"
    with pytest.raises(ContractError):
        inspect_eod_ledger_bindings(workspace=str(tmp_path), completion=completion, custody=custody)
